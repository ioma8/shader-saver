//! Healing-brush source search.
//!
//! A retouch dab needs a patch to copy. Photoshop's spot healing brush picks
//! one automatically and then corrects its colour so the join is invisible;
//! this does the same. It scores candidate offsets around the target, keeps the
//! best, then fits the colour correction that lands the winner on the pixels
//! around the dab.
//!
//! Three things decide whether that works, and each is easy to get wrong:
//!
//! * **What is scored.** Only the ring the shader feathers is compared. The
//!   middle of a dab is *replaced*, so scoring it rewards a source that looks
//!   like the blemish -- point the brush at a dust spot and it pastes a second
//!   dust spot.
//! * **What is pasted.** The interior of the source is not scored either, so it
//!   must be checked separately: whatever sits there is about to be copied onto
//!   the photo. A source whose interior holds something the surroundings do not
//!   is rejected.
//! * **What is read.** The search reads the same pixels the shader samples
//!   ([`View`]), never the finished display, or it would score candidates
//!   against pixels the render never uses.

/// The region of the image the dab search is allowed to look at. Holding only
/// this (rather than the whole frame) keeps a click on a 24 MP image cheap, so
/// it must be sized to cover the dab *plus* the farthest source; candidates
/// outside it would be scored against missing samples.
pub struct View<'a> {
    /// RGBA8, `dims` pixels per row.
    pub pixels: &'a [u8],
    pub dims: (u32, u32),
    /// Where `pixels` sits in the full image, in image pixels.
    pub origin: (f32, f32),
    /// The full image size, in image pixels.
    pub full: (f32, f32),
}

impl<'a> View<'a> {
    /// A view covering a whole image, for callers that already hold it all.
    #[cfg(test)]
    pub fn whole(pixels: &'a [u8], w: u32, h: u32) -> Self {
        Self {
            pixels,
            dims: (w, h),
            origin: (0.0, 0.0),
            full: (w as f32, h as f32),
        }
    }
}

/// A dab's source: the offset to the patch that replaces it, and the affine
/// colour correction that lands that patch on the target.
pub struct Sample {
    /// Offset to the source patch, in image pixels.
    pub offset: [f32; 2],
    /// Per-channel multiplier applied to the copied patch.
    pub gain: [f32; 3],
    /// Per-channel offset added after the gain.
    pub bias: [f32; 3],
}

/// Candidate offsets, in brush radii, out to which the search looks.
const SEARCH_RADII: [f32; 4] = [2.2, 3.0, 4.2, 6.0];
/// Angles sampled around the target; a full circle, so direction is free.
const SEARCH_ANGLES: usize = 32;
/// Patch samples per axis inside the dab.
const PATCH_SAMPLES: usize = 9;
/// Fraction of the dab's radius that is replaced outright. This is the shader's
/// plateau: samples inside it carry no information about the seam, so they are
/// excluded from the match. The two constants must agree.
const BAND_INNER: f32 = 0.45;
/// Weight per brush-radius of distance, pulling the choice towards nearby
/// sources so the replacement stays related to its surroundings. Measured in
/// brush radii, so it means the same thing whatever the brush size.
const DISTANCE_WEIGHT: f32 = 0.03;
/// How far a source's interior may differ from the surroundings before it looks
/// like a second copy of the blemish, as a multiple of the variation those
/// surroundings themselves show. Texture (which naturally varies) is therefore
/// not penalised, while an anomaly on a smooth background is.
const INTERIOR_TOLERANCE: f32 = 4.0;
/// Weight of that interior term against the match, which is a mean squared
/// colour difference per channel, as is the term itself.
const INTERIOR_WEIGHT: f32 = 1.0;
/// Largest colour correction accepted, so a pathological match cannot paint a
/// wild cast.
const MAX_BIAS: f32 = 0.5;
/// Bounds on the fitted gain, for the same reason.
const MIN_GAIN: f32 = 0.5;
const MAX_GAIN: f32 = 2.0;
/// A source patch flatter than this (weighted variance) gets no gain fitted --
/// there is nothing to fit against, so the correction stays a plain offset.
const VAR_EPS: f32 = 1e-4;

/// The image region a dab at normalised `(x, y)` with `radius` (a fraction of
/// the shorter side) can reach, in image pixels: `(x, y, w, h)`.
pub fn search_rect(x: f32, y: f32, radius: f32, full_w: u32, full_h: u32) -> (u32, u32, u32, u32) {
    let r = radius_px(radius, full_w, full_h);
    let half = r * SEARCH_RADII[SEARCH_RADII.len() - 1] + r + 1.0;
    let (fx, fy) = (full_w as f32, full_h as f32);
    let cx = x * fx;
    let cy = y * fy;
    let x0 = (cx - half).max(0.0) as u32;
    let y0 = (cy - half).max(0.0) as u32;
    let x1 = ((cx + half).min(fx)).ceil() as u32;
    let y1 = ((cy + half).min(fy)).ceil() as u32;
    (x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0))
}

/// A dab's radius in pixels from the normalised radius stored on the spot.
pub fn radius_px(radius: f32, full_w: u32, full_h: u32) -> f32 {
    (radius * (full_w.min(full_h) as f32)).max(1.5)
}

/// Choose a source patch for a dab centred at `centre` (image pixels) with
/// radius `r` (pixels), from `view`. Returns `None` only when the view cannot
/// produce a single usable sample.
pub fn find_sample(view: &View, centre: [f32; 2], r: f32) -> Option<Sample> {
    let (vw, vh) = view.dims;
    if vw < 2 || vh < 2 || r < 1.0 {
        return None;
    }
    if view.pixels.len() < (vw as usize) * (vh as usize) * 4 {
        return None;
    }
    let (full_w, full_h) = view.full;
    // The dab centre in the view's own coordinates.
    let local = [centre[0] - view.origin.0, centre[1] - view.origin.1];

    // Sample lattice: `(u, v)` in pixels relative to a centre, plus the weight
    // the sample carries when matching the seam. Samples inside the plateau get
    // weight 0 -- they are replaced, so they say nothing about the join. They
    // are kept, because what a candidate would *paste* has to be checked.
    let mut band: Vec<(f32, f32, f32)> = Vec::with_capacity(PATCH_SAMPLES * PATCH_SAMPLES);
    let mut interior: Vec<(f32, f32, f32)> = Vec::with_capacity(PATCH_SAMPLES * PATCH_SAMPLES);
    for iy in 0..PATCH_SAMPLES {
        for ix in 0..PATCH_SAMPLES {
            let u = (ix as f32 + 0.5) / PATCH_SAMPLES as f32 * 2.0 - 1.0;
            let v = (iy as f32 + 0.5) / PATCH_SAMPLES as f32 * 2.0 - 1.0;
            let norm = (u * u + v * v).sqrt();
            if norm > 1.0 {
                continue;
            }
            if norm < BAND_INNER {
                interior.push((u * r, v * r, norm.max(0.05)));
            } else {
                // 0 at the plateau edge, 1 at the rim: the rim blends hardest.
                let weight = (norm - BAND_INNER) / (1.0 - BAND_INNER);
                band.push((u * r, v * r, weight));
            }
        }
    }
    if band.is_empty() {
        return None;
    }

    // Read one sample in view coordinates; `None` outside the view.
    let at = |lx: f32, ly: f32| -> Option<[f32; 3]> {
        if lx < 0.0 || ly < 0.0 || lx >= vw as f32 || ly >= vh as f32 {
            return None;
        }
        let o = ((ly as usize) * vw as usize + lx as usize) * 4;
        Some([
            f32::from(view.pixels[o]) / 255.0,
            f32::from(view.pixels[o + 1]) / 255.0,
            f32::from(view.pixels[o + 2]) / 255.0,
        ])
    };

    // Destination band, read once and reused for every candidate, plus its mean
    // and how much variation the seam itself shows (which sets what counts as
    // an anomaly in a source).
    let dest: Vec<Option<[f32; 3]>> = band
        .iter()
        .map(|&(u, v, _)| at(local[0] + u, local[1] + v))
        .collect();
    let mut dest_mean = [0.0f32; 3];
    let mut wsum = 0.0f32;
    for (k, &(_, _, w)) in band.iter().enumerate() {
        let Some(c) = dest[k] else { continue };
        for (m, x) in dest_mean.iter_mut().zip(c) {
            *m += w * x;
        }
        wsum += w;
    }
    if wsum <= 0.0 {
        return None;
    }
    for m in &mut dest_mean {
        *m /= wsum;
    }
    let mut variance = 0.0f32;
    for (k, &(_, _, w)) in band.iter().enumerate() {
        let Some(c) = dest[k] else { continue };
        let mut e = 0.0;
        for (x, m) in c.iter().zip(dest_mean) {
            e += (x - m) * (x - m);
        }
        variance += w * e / 3.0;
    }
    let variance = variance / wsum;

    // A source on top of the target would copy the very thing being removed;
    // anywhere past two radii is fair game.
    let min_offset = r * 2.0;
    let max_offset = (full_w.min(full_h) * 0.55).min(r * SEARCH_RADII[SEARCH_RADII.len() - 1]);

    // Score the candidate ring. `clamp` pulls a candidate's centre to where its
    // patch still fits instead of discarding it -- used as a fallback so a dab
    // against a border, or a brush nearly as large as the frame, still lands.
    let search = |clamp: bool| -> Option<(f32, f32, f32)> {
        let mut best: Option<(f32, f32, f32)> = None;
        for &multiple in &SEARCH_RADII {
            let dist = r * multiple;
            if dist < min_offset || dist > max_offset {
                continue;
            }
            for a in 0..SEARCH_ANGLES {
                let angle = a as f32 / SEARCH_ANGLES as f32 * std::f32::consts::TAU;
                let (dx, dy) = (angle.cos() * dist, angle.sin() * dist);
                let (mut sx, mut sy) = (centre[0] + dx, centre[1] + dy);
                if clamp {
                    let (lo_x, hi_x) = fit_range(view.origin.0, vw as f32, r, full_w);
                    let (lo_y, hi_y) = fit_range(view.origin.1, vh as f32, r, full_h);
                    sx = sx.clamp(lo_x, hi_x);
                    sy = sy.clamp(lo_y, hi_y);
                } else if sx - r < 0.0 || sx + r > full_w || sy - r < 0.0 || sy + r > full_h {
                    continue;
                }
                let (sx_local, sy_local) = (sx - view.origin.0, sy - view.origin.1);

                // Does the candidate's own rim match the destination's?
                let mut score = 0.0f32;
                let mut weight = 0.0f32;
                for (k, &(u, v, w)) in band.iter().enumerate() {
                    let (Some(d), Some(s)) = (dest[k], at(sx_local + u, sy_local + v)) else {
                        continue;
                    };
                    let mut e = 0.0;
                    for i in 0..3 {
                        let x = s[i] - d[i];
                        e += x * x;
                    }
                    score += w * e / 3.0;
                    weight += w;
                }
                if weight <= 0.0 {
                    continue;
                }
                score /= weight;

                // Would pasting it drop something the surroundings do not have?
                let mut deviation = 0.0f32;
                let mut dw = 0.0f32;
                for &(u, v, w) in &interior {
                    let Some(s) = at(sx_local + u, sy_local + v) else {
                        continue;
                    };
                    let mut e = 0.0;
                    for i in 0..3 {
                        let x = s[i] - dest_mean[i];
                        e += x * x;
                    }
                    deviation += w * e / 3.0;
                    dw += w;
                }
                if dw > 0.0 {
                    let excess = deviation / dw - INTERIOR_TOLERANCE * variance;
                    score += INTERIOR_WEIGHT * excess.max(0.0);
                }

                score += (dist / r) * DISTANCE_WEIGHT;
                if best.is_none_or(|(b, _, _)| score < b) {
                    best = Some((score, dx, dy));
                }
            }
        }
        best
    };

    let (_, dx, dy) = search(false).or_else(|| search(true))?;
    let (sx_local, sy_local) = (local[0] + dx, local[1] + dy);

    // Affine fit of destination onto source over the band: `dst ~= src * g + b`.
    // Fitting the gain as well as the offset is what keeps a dab usable after a
    // later exposure change, because the source is re-read at render time and
    // scales with it -- a bare offset would be left behind.
    let mut pairs: Vec<([f32; 3], [f32; 3], f32)> = Vec::with_capacity(band.len());
    for (k, &(u, v, w)) in band.iter().enumerate() {
        let (Some(d), Some(s)) = (dest[k], at(sx_local + u, sy_local + v)) else {
            continue;
        };
        pairs.push((s, d, w));
    }
    if pairs.is_empty() {
        return None;
    }
    let wsum: f32 = pairs.iter().map(|p| p.2).sum();
    if wsum <= 0.0 {
        return None;
    }
    let (mut mean_src, mut mean_dst) = ([0.0f32; 3], [0.0f32; 3]);
    for (s, d, w) in &pairs {
        for c in 0..3 {
            mean_src[c] += w * s[c];
            mean_dst[c] += w * d[c];
        }
    }
    for c in 0..3 {
        mean_src[c] /= wsum;
        mean_dst[c] /= wsum;
    }
    let (mut var, mut cov) = ([0.0f32; 3], [0.0f32; 3]);
    for (s, d, w) in &pairs {
        for c in 0..3 {
            let ds = s[c] - mean_src[c];
            var[c] += w * ds * ds;
            cov[c] += w * ds * (d[c] - mean_dst[c]);
        }
    }
    let mut gain = [1.0f32; 3];
    let mut bias = [0.0f32; 3];
    for c in 0..3 {
        gain[c] = if var[c] > VAR_EPS * wsum {
            (cov[c] / var[c]).clamp(MIN_GAIN, MAX_GAIN)
        } else {
            1.0
        };
        bias[c] = (mean_dst[c] - gain[c] * mean_src[c]).clamp(-MAX_BIAS, MAX_BIAS);
    }

    Some(Sample {
        offset: [dx, dy],
        gain,
        bias,
    })
}

/// Centre positions, within a region starting at `origin` and `len` long, at
/// which a patch of radius `r` still fits inside the image `full`.
fn fit_range(origin: f32, len: f32, r: f32, full: f32) -> (f32, f32) {
    let lo = (origin + r).min(origin + len * 0.5);
    let hi = (origin + len - r).max(lo).min((full - r).max(lo));
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::{find_sample, radius_px, search_rect, View};

    const SPOT_DARK: u8 = 30;
    const BG: u8 = 150;

    /// A flat field of `value` with dark circular spots at the given centres.
    fn field(w: u32, h: u32, value: u8, spots: &[(u32, u32, f32)]) -> Vec<u8> {
        let mut px = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let mut v = value;
                for &(cx, cy, r) in spots {
                    let dx = x as f32 - cx as f32;
                    let dy = y as f32 - cy as f32;
                    if dx * dx + dy * dy <= r * r {
                        v = SPOT_DARK;
                    }
                }
                let o = ((y * w + x) * 4) as usize;
                px[o] = v;
                px[o + 1] = v;
                px[o + 2] = v;
            }
        }
        px
    }

    fn luma(px: &[u8], w: u32, x: u32, y: u32) -> u8 {
        px[((y * w + x) as usize) * 4]
    }

    #[test]
    fn source_never_overlaps_the_dab_it_replaces() {
        let (w, h) = (256u32, 256u32);
        let px = field(w, h, 255, &[(128, 128, 12.0)]);
        let r = radius_px(12.0 / 256.0, w, h);
        let view = View::whole(&px, w, h);
        let sample = find_sample(&view, [128.0, 128.0], r).expect("a source should be found");
        let (dx, dy) = (sample.offset[0], sample.offset[1]);
        assert!(
            (dx * dx + dy * dy).sqrt() >= r * 2.0 - 0.5,
            "source offset {dx},{dy} overlaps the dab of radius {r}"
        );
    }

    // The regression this module exists for. Scoring the whole dab made a
    // source that itself contains a blemish match perfectly, so healing a dust
    // spot pasted a second dust spot in its place. A clean source has to win.
    #[test]
    fn a_dust_spot_is_not_healed_from_another_dust_spot() {
        let (w, h) = (240u32, 240u32);
        // One spot to heal, and a second exactly 48 px away -- which is the
        // nearest offset the search will try (2.2 brush radii at r = 22).
        let px = field(w, h, BG, &[(60, 120, 5.0), (108, 120, 5.0)]);
        let view = View::whole(&px, w, h);
        let sample = find_sample(&view, [60.0, 120.0], 22.0).expect("a source should be found");
        let (sx, sy) = (60.0 + sample.offset[0], 120.0 + sample.offset[1]);
        // Nothing dark may be pasted: the whole interior must be clean.
        for (dx, dy) in [(0i32, 0i32), (5, 0), (-5, 0), (0, 5), (0, -5)] {
            let (x, y) = ((sx as i32 + dx) as u32, (sy as i32 + dy) as u32);
            assert_eq!(
                luma(&px, w, x, y),
                BG,
                "pasted a blemish at ({x}, {y}); source was ({sx:.0}, {sy:.0})"
            );
        }
    }

    #[test]
    fn bias_corrects_a_brightness_mismatch() {
        // A horizontal gradient, so the chosen source patch sits at a different
        // brightness from the target and the correction is what cancels it.
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
        let r = 16.0;
        let view = View::whole(&px, w, h);
        let sample = find_sample(&view, [77.0, 128.0], r).expect("a source should be found");
        let (sx, sy) = (77.0 + sample.offset[0], 128.0 + sample.offset[1]);
        let src = band_mean(&px, w, h, sx, sy, r);
        let dst = band_mean(&px, w, h, 77.0, 128.0, r);
        for c in 0..3 {
            let corrected = src[c] * sample.gain[c] + sample.bias[c];
            assert!(
                (corrected - dst[c]).abs() < 0.03,
                "channel {c}: {} * {} + {} = {corrected} should land on {}",
                src[c],
                sample.gain[c],
                sample.bias[c],
                dst[c]
            );
        }
    }

    #[test]
    fn a_dab_on_a_dark_side_is_sourced_from_the_dark_side() {
        // Two flat tones side by side. A source taken from the bright half
        // would need a large correction; the nearby same-tone patch is right.
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
        let view = View::whole(&px, w, h);
        let sample = find_sample(&view, [64.0, 128.0], 16.0).expect("a source should be found");
        for (c, b) in sample.bias.iter().enumerate() {
            assert!(b.abs() < 0.1, "bias[{c}] = {b} should stay small");
        }
    }

    #[test]
    fn a_flat_field_needs_no_correction() {
        let (w, h) = (128u32, 128u32);
        let px = vec![128u8; (w * h * 4) as usize];
        let view = View::whole(&px, w, h);
        let sample = find_sample(&view, [64.0, 64.0], 8.0).expect("a source should be found");
        for c in 0..3 {
            assert!(
                (sample.gain[c] - 1.0).abs() < 1e-3 && sample.bias[c].abs() < 1e-3,
                "channel {c} should need no correction: gain {} bias {}",
                sample.gain[c],
                sample.bias[c]
            );
        }
    }

    // A brush as large as the UI allows, in the middle of a 3:2 frame: every
    // candidate starts out of bounds, and before the fallback this returned
    // `None` and the click did nothing at all.
    #[test]
    fn a_brush_nearly_as_large_as_the_frame_still_finds_a_source() {
        let (w, h) = (1500u32, 1000u32);
        let mut px = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let v = if x < w / 2 { 40u8 } else { 210u8 };
                let o = ((y * w + x) * 4) as usize;
                px[o] = v;
                px[o + 1] = v;
                px[o + 2] = v;
            }
        }
        let radius = 0.25;
        let r = radius_px(radius, w, h);
        let view = View::whole(&px, w, h);
        let sample = find_sample(&view, [750.0, 500.0], r)
            .expect("a max-size brush must still resolve a source");
        assert!(sample.offset[0].is_finite() && sample.offset[1].is_finite());
    }

    // The search only ever sees the rect it is given, so that rect has to cover
    // every reachable source; a cropped view must agree with the whole image.
    #[test]
    fn a_cropped_view_agrees_with_the_whole_image() {
        let (w, h) = (240u32, 240u32);
        let px = field(w, h, BG, &[(60, 120, 5.0), (108, 120, 5.0)]);
        let r = 22.0;
        let centre = [60.0, 120.0];
        let whole = find_sample(&View::whole(&px, w, h), centre, r).expect("source");
        let (x0, y0, vw, vh) = search_rect(60.0 / 240.0, 120.0 / 240.0, r / 240.0, w, h);
        let mut crop = Vec::with_capacity((vw * vh * 4) as usize);
        for y in y0..y0 + vh {
            let start = ((y * w + x0) * 4) as usize;
            crop.extend_from_slice(&px[start..start + (vw * 4) as usize]);
        }
        let view = View {
            pixels: &crop,
            dims: (vw, vh),
            origin: (x0 as f32, y0 as f32),
            full: (w as f32, h as f32),
        };
        let cropped = find_sample(&view, centre, r).expect("source");
        assert_eq!(whole.offset, cropped.offset);
    }

    // The band the search scores and the plateau the shader replaces have to be
    // the same ring: scoring a seam the render never draws would pick sources
    // for the wrong reason. The two constants live in different files, so this
    // pins them together.
    #[test]
    fn the_scored_band_matches_the_shaders_plateau() {
        let shader = include_str!("shaders.wgsl");
        let expected = format!("const RETOUCH_INNER: f32 = {};", super::BAND_INNER);
        assert!(
            shader.contains(&expected),
            "shaders.wgsl must declare `{expected}` to match the search's BAND_INNER"
        );
    }

    #[test]
    fn degenerate_views_do_not_panic() {
        let px = [0u8; 4 * 4 * 4];
        assert!(find_sample(&View::whole(&px, 1, 1), [0.0, 0.0], 1.0).is_none());
        assert!(find_sample(
            &View {
                pixels: &[],
                dims: (0, 0),
                origin: (0.0, 0.0),
                full: (4.0, 4.0),
            },
            [1.0, 1.0],
            1.0,
        )
        .is_none());
        let (_, _, w, h) = search_rect(0.5, 0.5, 0.2, 0, 0);
        assert_eq!((w, h), (0, 0));
        // A tiny image has no room for a patch, but must not panic either.
        let _ = find_sample(&View::whole(&px, 4, 4), [2.0, 2.0], 3.0);
    }

    /// Mean RGB over the *band* the search scores (the shader's feather ring),
    /// weighted the same way, in 0..1 units.
    fn band_mean(px: &[u8], w: u32, h: u32, cx: f32, cy: f32, r: f32) -> [f32; 3] {
        let mut sum = [0.0f64; 3];
        let mut n = 0.0f64;
        const STEPS: usize = 32;
        for iy in 0..STEPS {
            for ix in 0..STEPS {
                let u = (ix as f32 + 0.5) / STEPS as f32 * 2.0 - 1.0;
                let v = (iy as f32 + 0.5) / STEPS as f32 * 2.0 - 1.0;
                let norm = (u * u + v * v).sqrt();
                if norm > 1.0 || norm < super::BAND_INNER {
                    continue;
                }
                let weight = f64::from((norm - super::BAND_INNER) / (1.0 - super::BAND_INNER));
                let x = (cx + u * r).round().clamp(0.0, w as f32 - 1.0) as usize;
                let y = (cy + v * r).round().clamp(0.0, h as f32 - 1.0) as usize;
                let o = (y * w as usize + x) * 4;
                for (c, s) in sum.iter_mut().enumerate() {
                    *s += f64::from(px[o + c]) / 255.0 * weight;
                }
                n += weight;
            }
        }
        [
            (sum[0] / n) as f32,
            (sum[1] / n) as f32,
            (sum[2] / n) as f32,
        ]
    }
}
