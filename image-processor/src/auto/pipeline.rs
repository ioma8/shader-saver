//! CPU replica of the editor's automatic-slider transform, mirroring
//! `src/shaders.wgsl` (`contrast_pass` then `tonal_pass`) for the sliders Auto
//! uses.
//!
//! This exists so the offline training renderer (torch) can be checked against
//! the exact math the GPU will apply at run time — a mismatch there would bias
//! every learned slider. It is also a convenient way to preview a slider state
//! without a GPU.
//!
//! It sits off the runtime path, so only the tests exercise every helper.
#![allow(dead_code)]

use crate::processor::EditState;
use image::RgbaImage;

const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

fn luma(rgb: [f32; 3]) -> f32 {
    rgb[0] * LUMA[0] + rgb[1] * LUMA[1] + rgb[2] * LUMA[2]
}

/// Apply the editor's automatic transform to one gamma-encoded `[0, 1]` pixel.
pub fn transform(mut rgb: [f32; 3], s: &EditState) -> [f32; 3] {
    // contrast_pass: exposure, white balance, levels, gamma, contrast.
    let gain = (s.exposure * 0.4545).exp2();
    for channel in &mut rgb {
        *channel *= gain;
    }

    let t = s.wb_temp * 0.01;
    let g = s.wb_tint * 0.01;
    if t.abs() > 0.001 || g.abs() > 0.001 {
        let mut gains = [
            1.0 + 0.30 * t + 0.10 * g,
            1.0 - 0.20 * g,
            1.0 - 0.30 * t + 0.10 * g,
        ];
        let norm = gains[0] * LUMA[0] + gains[1] * LUMA[1] + gains[2] * LUMA[2];
        for channel in &mut gains {
            *channel = (*channel / norm).max(0.0).powf(0.4545);
        }
        for i in 0..3 {
            rgb[i] *= gains[i];
        }
    }

    let lum = luma(rgb);
    let black = s.levels_black / 255.0;
    let white = s.levels_white / 255.0;
    let range = (white - black).max(0.001);
    let mut l = ((lum - black) / range).clamp(0.0, 1.0);
    l = l.powf(1.0 / s.levels_gamma.max(0.01));

    let cs = s.contrast * 0.01;
    if cs.abs() > 0.001 {
        let k = cs.abs() * 10.0;
        let s0 = 1.0 / (1.0 + (0.5 * k).exp());
        let s1 = 1.0 / (1.0 + (-0.5 * k).exp());
        if cs > 0.0 {
            l = (1.0 / (1.0 + (-k * (l - 0.5)).exp()) - s0) / (s1 - s0);
        } else {
            let u = (s0 + l * (s1 - s0)).clamp(1e-5, 1.0 - 1e-5);
            l = 0.5 - (1.0 / u - 1.0).ln() / k;
        }
    }

    let scale = if lum < 0.001 { 1.0 } else { l / lum };
    for channel in &mut rgb {
        *channel = (*channel * scale).clamp(0.0, 1.0);
    }

    tonal(rgb, s)
}

/// tonal_pass: brightness, blacks/shadows/highlights/whites, vibrance/saturation.
fn tonal(mut rgb: [f32; 3], s: &EditState) -> [f32; 3] {
    let blacks = s.blacks * 0.01;
    let shadows = s.shadows * 0.01;
    let highlights = s.highlights * 0.01;
    let whites = s.whites * 0.01;

    let lum = luma(rgb);
    let bias = 0.5 + s.brightness * 0.0025;
    let lum_b = lum / ((1.0 / bias - 2.0) * (1.0 - lum) + 1.0);
    let inv = 1.0 - lum_b;

    let w_shadows = 9.4815 * lum_b * inv * inv * inv;
    let w_highlights = 9.4815 * lum_b * lum_b * lum_b * inv;
    let delta = blacks * 0.25 * inv.powi(6)
        + shadows * 0.35 * w_shadows
        + highlights * 0.35 * w_highlights
        + whites * 0.25 * lum_b.powi(6);
    let new_lum = (lum_b + delta).clamp(0.0, 1.0);

    let scale = if lum < 0.001 { 1.0 } else { new_lum / lum };
    for channel in &mut rgb {
        *channel = (*channel * scale).clamp(0.0, 1.0);
    }

    let mid = luma(rgb);
    let chroma = ((rgb[0] - mid).powi(2) + (rgb[1] - mid).powi(2) + (rgb[2] - mid).powi(2)).sqrt();
    let sat = 1.0 + s.saturation * 0.01;
    let vib = 1.0 + s.vibrance * 0.01 * (1.0 - (chroma * 2.0).clamp(0.0, 1.0));
    for channel in &mut rgb {
        *channel = (mid + (*channel - mid) * sat * vib).clamp(0.0, 1.0);
    }
    rgb
}

/// Apply the transform to a whole 8-bit image (returns a new image).
pub fn render(img: &RgbaImage, s: &EditState) -> RgbaImage {
    let mut out = img.clone();
    for px in out.pixels_mut() {
        let input = [
            f32::from(px[0]) / 255.0,
            f32::from(px[1]) / 255.0,
            f32::from(px[2]) / 255.0,
        ];
        let result = transform(input, s);
        for i in 0..3 {
            px[i] = (result[i].clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> EditState {
        EditState::default()
    }

    #[test]
    fn neutral_state_is_identity() {
        for value in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let out = transform([value, value, value], &state());
            for channel in out {
                assert!((channel - value).abs() < 1e-5, "{channel} != {value}");
            }
        }
    }

    #[test]
    fn contrast_s_curve_keeps_endpoints_and_adds_contrast() {
        let mut s = state();
        s.contrast = 20.0;
        assert!(transform([0.0, 0.0, 0.0], &s)[0].abs() < 1e-6);
        assert!((transform([1.0, 1.0, 1.0], &s)[0] - 1.0).abs() < 1e-6);
        // A dark grey gets darker and a light grey lighter.
        assert!(transform([0.25, 0.25, 0.25], &s)[0] < 0.25);
        assert!(transform([0.75, 0.75, 0.75], &s)[0] > 0.75);
    }

    #[test]
    fn levels_stretch_maps_black_and_white_to_the_ends() {
        let mut s = state();
        s.levels_black = 50.0;
        s.levels_white = 200.0;
        assert!(transform([50.0 / 255.0; 3], &s)[0].abs() < 1e-5);
        assert!((transform([200.0 / 255.0; 3], &s)[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn gamma_lifts_or_lowers_midtones() {
        let mut up = state();
        up.levels_gamma = 2.0;
        assert!(transform([0.25, 0.25, 0.25], &up)[0] > 0.25);
        let mut down = state();
        down.levels_gamma = 0.5;
        assert!(transform([0.25, 0.25, 0.25], &down)[0] < 0.25);
    }

    #[test]
    fn white_balance_moves_channels_apart() {
        let mut s = state();
        s.wb_temp = 50.0;
        let out = transform([0.5, 0.5, 0.5], &s);
        assert!(out[0] > out[2], "warm should raise R above B: {out:?}");
    }

    #[test]
    fn luminance_ops_preserve_hue_ratio() {
        // Levels/contrast only scale RGB uniformly, so channel ratios hold.
        let mut s = state();
        s.contrast = 25.0;
        s.levels_gamma = 1.4;
        let input = [0.6, 0.4, 0.3];
        let out = transform(input, &s);
        let (a, b) = (out[0] / input[0], out[1] / input[1]);
        assert!((a - b).abs() < 1e-4, "scale differs per channel: {a} {b}");
    }

    #[test]
    fn vibrance_increases_chroma() {
        let mut s = state();
        s.vibrance = 20.0;
        let input = [0.55, 0.5, 0.45];
        let out = transform(input, &s);
        let spread = |c: [f32; 3]| c[0] - c[2];
        assert!(spread(out) > spread(input));
    }

    #[test]
    fn render_keeps_size_and_alpha() {
        let img = RgbaImage::from_pixel(3, 2, image::Rgba([10, 20, 30, 128]));
        let out = render(&img, &state());
        assert_eq!(out.dimensions(), (3, 2));
        assert_eq!(out.get_pixel(0, 0)[3], 128);
    }

    /// `AUTO_GOLDEN=path` writes the transform's output over a small grid, so
    /// the torch training renderer can be checked against it.
    #[test]
    fn golden_probe() {
        let Ok(path) = std::env::var("AUTO_GOLDEN") else {
            return;
        };
        let mut out = String::new();
        let sliders = [
            state(),
            {
                let mut s = state();
                s.wb_temp = 22.0;
                s.wb_tint = -11.0;
                s.levels_black = 12.0;
                s.levels_gamma = 1.7;
                s.levels_white = 240.0;
                s.contrast = 24.0;
                s.shadows = 18.0;
                s.highlights = -12.0;
                s.vibrance = 16.0;
                s
            },
        ];
        for (index, s) in sliders.iter().enumerate() {
            for r in 0..=8 {
                for c in 0..=8 {
                    let rgb = [r as f32 / 8.0, c as f32 / 8.0, ((r + c) % 9) as f32 / 8.0];
                    let o = transform(rgb, s);
                    out.push_str(&format!(
                        "{index} {r} {c} {:.8} {:.8} {:.8}\n",
                        o[0], o[1], o[2]
                    ));
                }
            }
        }
        std::fs::write(path, out).unwrap();
    }
}
