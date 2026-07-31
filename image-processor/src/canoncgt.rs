// CanonCGT: Reference-Based Color Grading via Canonical Pivot Representation
// (Jinwon Ko, Keunsoo Ko, Chang-Su Kim, CVPR 2026). Apache-2.0, including the
// pretrained weights: https://github.com/Jinwon-Ko/CanonCGT
//
// The network predicts two 17^3 3D LUTs from 224x224 views of the target and the
// reference: `canonicalize_lut` maps the target to a style-neutral pivot (removes
// *its own* grade), and `restylize_lut` applies the reference's grade from that
// pivot. Composing the two and upsampling to the 33-cube the rest of this app
// already speaks gives one target-specific LUT, applied non-destructively at full
// resolution by the existing shader -- this module never touches a pixel outside
// its own 224x224 working copies.
//
// This replaced a hand-built statistics-based transfer (see `processor.rs`'s
// `LookProfile`/`LookTransfer`/region-anchor machinery, kept as a fallback for
// when this model fails to load). That system went through five rounds of
// measured failure -- oversaturation, washed-out range, no hue awareness, skin
// drift, a search that gamed a learned metric into posterizing pavement pink --
// each fixed in turn, but a real reference-conditioned network with actual
// content understanding (cross-attention between the LUT grid and both images'
// features) simply does the job better: a side-by-side on the model's own sample
// photos showed richer, more faithful color transfer with full structural
// fidelity, no artifacts, in a single forward pass.
//
// ONNX export note: `torch.onnx.export` refuses 5D (volumetric) `grid_sample`
// outright ("Unsupported: ONNX export of operator GridSample with 5D volumetric
// input"), which the upstream model uses to render pixels from a LUT. Since this
// module only wants the *LUT*, not rendered pixels, the exported graph reimplements
// that one lookup with Gather/Mul/Add (verified to match the original via
// `torch.allclose` before export) and never calls the unsupported op.

use tract::prelude::*;

use crate::processor::{sample_cube, smooth_lut};

const MODEL: &[u8] = include_bytes!("../models/canoncgt_lut.onnx");
const INPUT_SIZE: u32 = 224;
const GRID: usize = 17;
const OUTPUT_GRID: usize = 33;

pub struct CanonCGT {
    runnable: tract::Runnable,
}

impl CanonCGT {
    pub fn load() -> Option<Self> {
        let runnable = tract::onnx()
            .ok()?
            .load_buffer(MODEL)
            .ok()?
            .into_model()
            .ok()?
            .into_runnable()
            .ok()?;
        Some(Self { runnable })
    }

    // Predict a target-specific 33^3 LUT (RGB-interleaved, matching
    // `combined_photo_lut`'s layout) that carries `reference`'s color grade onto
    // `target`. Neither image needs to be any particular size; both are resized to
    // the network's fixed 224x224 input here.
    pub fn predict_lut(
        &self,
        target: &image::RgbaImage,
        reference: &image::RgbaImage,
    ) -> Option<Vec<f32>> {
        if target.width() == 0 || target.height() == 0 {
            return None;
        }
        let img = to_input_tensor(target)?;
        let refimg = to_input_tensor(reference)?;
        let outputs = self.runnable.run([img, refimg]).ok()?;

        let canon = planar_to_interleaved(outputs[0].as_slice::<f32>().ok()?, GRID)?;
        let restyle = planar_to_interleaved(outputs[1].as_slice::<f32>().ok()?, GRID)?;
        if canon.iter().chain(&restyle).any(|v| !v.is_finite()) {
            return None;
        }

        // Compose: evaluate the restyle LUT at the colors the canonicalize LUT
        // produces, so a single pass through the composed table does what the
        // network does in two. Then upsample by sampling that composed cube at
        // the 33-grid's own coordinates -- the standard meaning of upsampling a
        // LUT, and the same trilinear helper both steps and the shader all share.
        let composed = compose_luts(&canon, &restyle, GRID);
        let upsampled = upsample_lut(&composed, GRID, OUTPUT_GRID);
        // The 17-cube network output isn't perfectly smooth cell-to-cell (see
        // the reversal-count diagnostic that motivated this); left alone that
        // shows up as visible blotching/banding on smooth gradients like skin.
        Some(smooth_lut(&upsampled, OUTPUT_GRID))
    }

    // Combine `pre_lut` (a caller-supplied, independently-derived correction --
    // see `processor::derive_look_chain`) with the network's own direct
    // prediction for this target/reference pair, into a single LUT the caller
    // can apply once at full resolution.
    //
    // This used to render `target` through `pre_lut` first and show the
    // network *that* image, then compose its response with `pre_lut`
    // sequentially -- the same instinct behind the rejected iterative
    // approach below, on a smaller scale. Measured twice: a bark color that
    // neither stage alone did much with had its blue channel collapse under
    // the composition; a strong, correct desaturation `pre_lut` alone got
    // right nearly vanished after the network reacted to the pre-desaturated
    // image and the composition cancelled it back out. Clamping the
    // composed result's magnitude (an earlier version of this function)
    // fixed the first failure but not the second -- a clamp only bounds how
    // far a result can run away, it can't stop one stage's correction from
    // being undone by the other. Both failures trace to the same cause:
    // asking the network to react to a color transform it never saw as a
    // genuine photo produces a response with no guaranteed relationship to
    // either stage's own, independently-reasonable opinion.
    //
    // A per-channel blend, weighted by how strongly each stage itself wants
    // to move a given color, sidesteps this rather than bounding it after
    // the fact: the network only ever sees the real, untouched target, and
    // the combined result is a true convex combination of two values that
    // are each already sane on their own -- it cannot run further from
    // identity than the stronger opinion, and it cannot cancel a strong
    // opinion down to nothing just because the weaker one disagrees.
    pub fn predict_lut_prenormalized(
        &self,
        target: &image::RgbaImage,
        reference: &image::RgbaImage,
        pre_lut: &[f32],
    ) -> Option<Vec<f32>> {
        let direct = self.predict_lut(target, reference)?;
        Some(blend_by_confidence(pre_lut, &direct, OUTPUT_GRID))
    }
}

fn blend_by_confidence(pre_lut: &[f32], direct: &[f32], n: usize) -> Vec<f32> {
    let cell_count = n * n * n;
    let mut out = vec![0f32; cell_count * 3];
    for i in 0..cell_count {
        let id = [
            (i % n) as f32 / (n - 1) as f32,
            ((i / n) % n) as f32 / (n - 1) as f32,
            (i / (n * n)) as f32 / (n - 1) as f32,
        ];
        for ch in 0..3 {
            let pre_dev = pre_lut[i * 3 + ch] - id[ch];
            let direct_dev = direct[i * 3 + ch] - id[ch];
            let (pre_w, direct_w) = (pre_dev.abs(), direct_dev.abs());
            let total_w = pre_w + direct_w;
            let blended_dev = if total_w > 1e-6 {
                (pre_dev * pre_w + direct_dev * direct_w) / total_w
            } else {
                0.0
            };
            out[i * 3 + ch] = id[ch] + blended_dev;
        }
    }
    out
}

// Tried and rejected: re-deriving over several rounds by feeding each round's
// own rendered output back in as the next round's target. Measured worse than
// turning up `look_strength` (which was already rejected for the same reason,
// see main.rs) -- by round 2 the image diverged into incoherent noise, not a
// closer match. Each round's output is a full LUT prediction rather than a
// small delta, so composing them compounds whatever the model got even
// slightly wrong, and the round-tripped 224x224 image drifts further from a
// natural photo each round rather than closer to the reference -- pushing the
// *input* further out of the network's training distribution, not back into
// it. Left as a note rather than dead code: if a future attempt revisits this
// direction, it should feed the network a still-*natural* image at each step
// (e.g. a small, independently-validated correction, not the network's own
// compounding output).

// Evaluate `outer` at the colors `inner` produces, so a single pass through the
// result does what applying `inner` then `outer` in sequence would do.
fn compose_luts(inner: &[f32], outer: &[f32], n: usize) -> Vec<f32> {
    inner
        .chunks_exact(3)
        .flat_map(|rgb| sample_cube(outer, n, [rgb[0], rgb[1], rgb[2]]))
        .collect()
}

// Resample a `from`-cube LUT onto a `to`-sized grid by sampling it at the new
// grid's own coordinates -- the standard meaning of resizing a 3D LUT, and the
// same trilinear helper the shader itself uses to apply one.
fn upsample_lut(lut: &[f32], from: usize, to: usize) -> Vec<f32> {
    (0..to.pow(3))
        .flat_map(|i| {
            let r = (i % to) as f32 / (to - 1) as f32;
            let g = ((i / to) % to) as f32 / (to - 1) as f32;
            let b = (i / (to * to)) as f32 / (to - 1) as f32;
            sample_cube(lut, from, [r, g, b])
        })
        .collect()
}

fn to_input_tensor(img: &image::RgbaImage) -> Option<tract::Tensor> {
    let resized =
        image::imageops::resize(img, INPUT_SIZE, INPUT_SIZE, image::imageops::FilterType::Triangle);
    let plane = (INPUT_SIZE * INPUT_SIZE) as usize;
    let mut chw = vec![0f32; plane * 3];
    for (i, px) in resized.pixels().enumerate() {
        for c in 0..3 {
            chw[c * plane + i] = f32::from(px[c]) / 255.0;
        }
    }
    tract::Tensor::from_slice(&[1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize], &chw).ok()
}

// The exported tensor is channel-planar [1,3,N,N,N] (PyTorch's native layout):
// flat index = channel*N^3 + (d*N^2 + h*N + w). This project's LUT convention is
// voxel-interleaved with R fastest, then G, then B -- and by construction of the
// model's identity LUT (R output varies along the tensor's W axis, G along H, B
// along D), `w + h*N + d*N^2` is exactly that same R-fastest voxel index. So the
// two layouts share their spatial indexing; only channel-vs-voxel grouping differs.
fn planar_to_interleaved(planar: &[f32], n: usize) -> Option<Vec<f32>> {
    let cell = n * n * n;
    if planar.len() != cell * 3 {
        return None;
    }
    let mut out = vec![0f32; cell * 3];
    for voxel in 0..cell {
        for channel in 0..3 {
            out[voxel * 3 + channel] = planar[channel * cell + voxel];
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_model_predicts_a_finite_in_range_lut() {
        let model = CanonCGT::load().expect("embedded model should load");
        let target = image::RgbaImage::from_fn(64, 64, |x, y| {
            image::Rgba([(x * 4) as u8, (y * 4) as u8, 128, 255])
        });
        let reference = image::RgbaImage::from_pixel(64, 64, image::Rgba([200, 140, 90, 255]));

        let lut = model
            .predict_lut(&target, &reference)
            .expect("prediction should succeed");
        assert_eq!(lut.len(), OUTPUT_GRID.pow(3) * 3);
        assert!(lut.iter().all(|v| v.is_finite()));
        // The composition step clamps into `sample_cube`, which itself clamps its
        // *input* coordinate but not its output value -- a LUT genuinely can (and
        // should) push saturation/contrast past the input range; the shader is
        // what clamps to displayable color. So this only checks for a plausible
        // range, not a hard [0,1] bound.
        assert!(lut.iter().all(|v| (-1.0..=2.0).contains(v)));
    }

    #[test]
    fn planar_to_interleaved_round_trips_the_identity() {
        // A 2x2x2 "identity-like" cube: channel c holds a distinct constant, so
        // decoding is easy to check by hand.
        let n = 2;
        let cell = n * n * n;
        let mut planar = vec![0f32; cell * 3];
        for channel in 0..3 {
            for voxel in 0..cell {
                planar[channel * cell + voxel] = (channel * 100 + voxel) as f32;
            }
        }
        let interleaved = planar_to_interleaved(&planar, n).unwrap();
        for voxel in 0..cell {
            for channel in 0..3 {
                assert_eq!(interleaved[voxel * 3 + channel], (channel * 100 + voxel) as f32);
            }
        }
    }
}
