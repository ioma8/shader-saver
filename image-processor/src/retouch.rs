//! Healing-brush source search.
//!
//! A retouch dab needs a patch to copy. Photoshop's spot healing brush picks
//! one automatically and then corrects its colour so the join is invisible;
//! this does the same in two steps. It scores candidate offsets around the
//! target (sum-of-squared-differences between the two patches, with a mild pull
//! towards nearby sources) and keeps the best. Then it records the mean colour
//! difference between the two patches, which the shader adds back to the copy,
//! so a patch taken from a slightly darker or cooler area still blends.
//!
//! The search reads the *rendered* image -- what the user clicked on -- so a
//! dab matches what is actually on screen, whatever produced it.

/// A dab's source, in the units [`crate::processor::RetouchSpot`] stores.
pub struct Sample {
    /// Offset to the source patch, as a fraction of width/height.
    pub source: [f32; 2],
    /// Additive RGB correction (0..1 units) to apply to the copied patch.
    pub bias: [f32; 3],
}

/// Candidate offsets, in brush radii, out to which the search looks.
const SEARCH_RADII: [f32; 4] = [2.2, 3.0, 4.2, 6.0];
/// Angles sampled around the target; a full circle, so direction is free.
const SEARCH_ANGLES: usize = 32;
/// Patch samples per axis inside the dab.
const PATCH_SAMPLES: usize = 9;
/// Weight per brush-radius of distance, biasing the score towards nearby
/// patches so the result stays related to its surroundings.
const DISTANCE_WEIGHT: f32 = 0.02;
/// Largest colour correction allowed, so a pathological match cannot paint a
/// wild cast.
const MAX_BIAS: f32 = 0.5;

/// Choose a source patch for a dab centred at normalised `(x, y)` with `radius`
/// (a fraction of the shorter side) from `pixels` (RGBA8, `w`x`h`). Returns
/// `None` when no candidate fits inside the image.
pub fn find_sample(pixels: &[u8], w: u32, h: u32, x: f32, y: f32, radius: f32) -> Option<Sample> {
    if w < 4 || h < 4 || pixels.len() < (w as usize) * (h as usize) * 4 {
        return None;
    }
    let (wf, hf) = (w as f32, h as f32);
    let short = wf.min(hf);
    let r = (radius * short).max(1.5);
    let cx = x * wf;
    let cy = y * hf;

    // Lattice of sample offsets inside the dab disc, relative to a centre.
    let mut patch: Vec<(f32, f32)> = Vec::with_capacity(PATCH_SAMPLES * PATCH_SAMPLES);
    for iy in 0..PATCH_SAMPLES {
        for ix in 0..PATCH_SAMPLES {
            let u = (ix as f32 + 0.5) / PATCH_SAMPLES as f32 * 2.0 - 1.0;
            let v = (iy as f32 + 0.5) / PATCH_SAMPLES as f32 * 2.0 - 1.0;
            if u * u + v * v <= 1.0 {
                patch.push((u * r, v * r));
            }
        }
    }
    if patch.is_empty() {
        return None;
    }

    let at = |px: f32, py: f32| -> [f32; 3] {
        let xi = px.round().clamp(0.0, wf - 1.0) as usize;
        let yi = py.round().clamp(0.0, hf - 1.0) as usize;
        let o = (yi * w as usize + xi) * 4;
        [
            f32::from(pixels[o]) / 255.0,
            f32::from(pixels[o + 1]) / 255.0,
            f32::from(pixels[o + 2]) / 255.0,
        ]
    };

    // Destination patch, read once and reused for every candidate.
    let dest: Vec<[f32; 3]> = patch.iter().map(|&(u, v)| at(cx + u, cy + v)).collect();

    // A source on top of the target would copy the very thing being removed;
    // anywhere past two radii is fair game.
    let min_offset = r * 2.0;
    let max_offset = (short * 0.55).min(r * SEARCH_RADII[SEARCH_RADII.len() - 1]);

    let mut best: Option<(f32, f32, f32)> = None; // (score, dx, dy)
    for &multiple in &SEARCH_RADII {
        let dist = r * multiple;
        if dist < min_offset || dist > max_offset {
            continue;
        }
        for a in 0..SEARCH_ANGLES {
            let angle = a as f32 / SEARCH_ANGLES as f32 * std::f32::consts::TAU;
            let (dx, dy) = (angle.cos() * dist, angle.sin() * dist);
            let (sx, sy) = (cx + dx, cy + dy);
            // Skip candidates whose patch would fall off the image.
            if sx - r < 0.0 || sx + r > wf || sy - r < 0.0 || sy + r > hf {
                continue;
            }
            let mut score = 0.0f32;
            for (k, &(u, v)) in patch.iter().enumerate() {
                let s = at(sx + u, sy + v);
                let d = dest[k];
                let (d0, d1, d2) = (s[0] - d[0], s[1] - d[1], s[2] - d[2]);
                score += d0 * d0 + d1 * d1 + d2 * d2;
            }
            score /= patch.len() as f32;
            score += (dist / short) * DISTANCE_WEIGHT;
            if best.is_none_or(|(b, _, _)| score < b) {
                best = Some((score, dx, dy));
            }
        }
    }

    let (_, dx, dy) = best?;

    let mut bias = [0.0f32; 3];
    for (k, &(u, v)) in patch.iter().enumerate() {
        let s = at(cx + dx + u, cy + dy + v);
        for (c, b) in bias.iter_mut().enumerate() {
            *b += dest[k][c] - s[c];
        }
    }
    let n = patch.len() as f32;
    for b in &mut bias {
        *b = (*b / n).clamp(-MAX_BIAS, MAX_BIAS);
    }

    Some(Sample {
        source: [dx / wf, dy / hf],
        bias,
    })
}

#[cfg(test)]
mod tests {
    use super::find_sample;

    /// A grey field with a black dot at `(dot_x, dot_y)`; a dab centred on the
    /// dot has to find its source somewhere else.
    fn grey_with_dot(w: u32, h: u32, dot_x: u32, dot_y: u32, dot_r: u32) -> Vec<u8> {
        let mut px = vec![255u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let dx = x as i64 - dot_x as i64;
                let dy = y as i64 - dot_y as i64;
                if dx * dx + dy * dy <= (dot_r * dot_r) as i64 {
                    let o = ((y * w + x) * 4) as usize;
                    px[o] = 20;
                    px[o + 1] = 20;
                    px[o + 2] = 20;
                }
            }
        }
        px
    }

    #[test]
    fn source_never_overlaps_the_dab_it_replaces() {
        let (w, h) = (256u32, 256u32);
        let px = grey_with_dot(w, h, 128, 128, 12);
        let radius = 12.0 / 256.0;
        let sample = find_sample(&px, w, h, 0.5, 0.5, radius).expect("a source should be found");
        let dx = sample.source[0] * w as f32;
        let dy = sample.source[1] * h as f32;
        let r = radius * 256.0;
        assert!(
            (dx * dx + dy * dy).sqrt() >= r * 2.0 - 0.5,
            "source offset {dx},{dy} overlaps the dab of radius {r}"
        );
    }

    #[test]
    fn bias_corrects_a_brightness_mismatch() {
        // A horizontal gradient, so the chosen source patch sits at a different
        // brightness from the target and the recorded bias is what cancels it.
        let (w, h) = (256u32, 256u32);
        let mut px = vec![255u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let v = (30 + x * 190 / (w - 1)) as u8;
                let o = ((y * w + x) * 4) as usize;
                px[o] = v;
                px[o + 1] = v;
                px[o + 2] = v;
            }
        }
        let radius = 16.0 / 256.0;
        let sample = find_sample(&px, w, h, 0.3, 0.5, radius).expect("a source should be found");
        let (cx, cy) = (0.3 * 256.0, 128.0);
        let r = radius * 256.0;
        let (sx, sy) = (cx + sample.source[0] * 256.0, cy + sample.source[1] * 256.0);
        let dest = disc_mean(&px, w, h, cx, cy, r);
        let src = disc_mean(&px, w, h, sx, sy, r);
        for c in 0..3 {
            assert!(
                (src[c] + sample.bias[c] - dest[c]).abs() < 0.02,
                "channel {c}: {} + {} should land on {}",
                src[c],
                sample.bias[c],
                dest[c]
            );
        }
    }

    #[test]
    fn a_dab_on_a_dark_side_is_sourced_from_the_dark_side() {
        // Two flat tones side by side. The bright half is the more eye-catching
        // match for a stray dab, but a source taken from it would need a large
        // correction; the nearby same-tone patch is the right answer, and the
        // bias it records is tiny.
        let (w, h) = (256u32, 256u32);
        let mut px = vec![255u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let v = if x < w / 2 { 60u8 } else { 200u8 };
                let o = ((y * w + x) * 4) as usize;
                px[o] = v;
                px[o + 1] = v;
                px[o + 2] = v;
            }
        }
        let sample =
            find_sample(&px, w, h, 0.25, 0.5, 16.0 / 256.0).expect("a source should be found");
        for (c, b) in sample.bias.iter().enumerate() {
            assert!(b.abs() < 0.1, "bias[{c}] = {b} should stay small");
        }
    }

    /// Mean RGB over the disc of radius `r` at `(cx, cy)`, in 0..1 units.
    fn disc_mean(px: &[u8], w: u32, h: u32, cx: f32, cy: f32, r: f32) -> [f32; 3] {
        let mut sum = [0.0f64; 3];
        let mut n = 0.0f64;
        const STEPS: usize = 24;
        for iy in 0..STEPS {
            for ix in 0..STEPS {
                let u = (ix as f32 + 0.5) / STEPS as f32 * 2.0 - 1.0;
                let v = (iy as f32 + 0.5) / STEPS as f32 * 2.0 - 1.0;
                if u * u + v * v > 1.0 {
                    continue;
                }
                let x = (cx + u * r).round().clamp(0.0, w as f32 - 1.0) as usize;
                let y = (cy + v * r).round().clamp(0.0, h as f32 - 1.0) as usize;
                let o = (y * w as usize + x) * 4;
                for (c, s) in sum.iter_mut().enumerate() {
                    *s += f64::from(px[o + c]) / 255.0;
                }
                n += 1.0;
            }
        }
        [
            (sum[0] / n) as f32,
            (sum[1] / n) as f32,
            (sum[2] / n) as f32,
        ]
    }

    #[test]
    fn a_flat_field_needs_no_correction() {
        let (w, h) = (128u32, 128u32);
        let px = vec![128u8; (w * h * 4) as usize];
        let sample =
            find_sample(&px, w, h, 0.5, 0.5, 8.0 / 128.0).expect("a source should be found");
        for (c, b) in sample.bias.iter().enumerate() {
            assert!(b.abs() < 1e-3, "bias[{c}] = {b} on a flat field");
        }
    }

    #[test]
    fn tiny_images_do_not_panic() {
        assert!(find_sample(&[0u8; 4 * 4 * 4], 4, 4, 0.5, 0.5, 0.2).is_none());
    }
}
