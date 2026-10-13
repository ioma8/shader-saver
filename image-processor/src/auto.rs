//! Analytic "Auto": derive the editor's visible sliders from a photo's own
//! statistics so any image — RAW-developed or JPEG — opens well exposed, white
//! balanced and lively.
//!
//! This is deliberately *not* a learned transform. It is the seed the learned
//! residual refines (see `docs/auto-enhance.md`), the fallback when that model
//! is absent, and the offline fitter used to build its training targets.
//!
//! Every tonal stage in `shaders.wgsl` runs on luminance and is applied back as
//! a uniform RGB scale, so a 256-bin luma histogram is a sufficient statistic
//! for choosing the tonal sliders. White balance is the only per-channel stage;
//! it is solved against the shader's own gain model.

use crate::processor::EditState;
use image::RgbaImage;

pub(crate) mod model;
pub(crate) mod pipeline;

/// The order `model.rs` and `tools/fivek/train.py` use for the slider vector.
pub(crate) fn state_to_vector(state: &EditState) -> [f32; 14] {
    [
        state.wb_temp,
        state.wb_tint,
        state.levels_black,
        state.levels_gamma,
        state.levels_white,
        state.contrast,
        state.blacks,
        state.shadows,
        state.highlights,
        state.whites,
        state.brightness,
        state.saturation,
        state.vibrance,
        state.exposure,
    ]
}

pub(crate) fn apply_vector(state: &mut EditState, values: &[f32; 14]) {
    state.wb_temp = values[0];
    state.wb_tint = values[1];
    state.levels_black = values[2];
    state.levels_gamma = values[3];
    state.levels_white = values[4];
    state.contrast = values[5];
    state.blacks = values[6];
    state.shadows = values[7];
    state.highlights = values[8];
    state.whites = values[9];
    state.brightness = values[10];
    state.saturation = values[11];
    state.vibrance = values[12];
    state.exposure = values[13];
    if state.levels_black >= state.levels_white {
        state.levels_black = state.levels_white - 1.0;
    }
}

/// Auto with the learned residual, when the model is available. The seed alone
/// is a complete edit, so a missing or failing model just yields the seed.
pub fn suggest_edit_with(img: &RgbaImage, residual: Option<&model::Residual>) -> EditState {
    let mut state = suggest_edit(img);
    if let Some(residual) = residual {
        if let Some(values) = residual.residual(img, &state_to_vector(&state)) {
            model::refine(&mut state, &values);
        }
    }
    state
}

/// Statistics are gathered on a small copy; the sliders do not need detail.
const STAT_SIZE: u32 = 256;

/// Where the median should land after levels + gamma.
const MEDIAN_TARGET: f64 = 0.45;
/// Inter-quartile range a well-finished photo tends to have. Contrast is added
/// to reach it, measured on the *input* before the levels normalise it away: a
/// camera JPEG is already contrasty (the camera applies its own curve), while
/// our develop is deliberately flat, so the same measurement gives the RAW more
/// and the JPEG almost none.
const CONTRAST_TARGET_IQR: f64 = 0.48;
/// How much S-curve to add per unit of missing inter-quartile range.
const CONTRAST_SLOPE: f64 = 400.0;
/// How much of the gray-world white-balance correction to apply. A RAW is
/// already white-balanced as shot, and a strong grey-world reads a blue sky or
/// green field as a cast, so apply only a little.
const WB_STRENGTH: f32 = 0.35;
/// Below this share of usable neutral pixels, leave white balance alone.
const WB_MIN_SAMPLE: f64 = 0.05;
/// Hard cap on the automatic white-balance move.
const WB_LIMIT: f32 = 30.0;

pub fn suggest_edit(img: &RgbaImage) -> EditState {
    let mut state = EditState::default();
    if img.width() == 0 || img.height() == 0 {
        return state;
    }
    let (width, height) = img.dimensions();
    let small = if width.max(height) > STAT_SIZE {
        let scale = f64::from(STAT_SIZE) / f64::from(width.max(height));
        let target = (
            ((f64::from(width) * scale).round() as u32).max(1),
            ((f64::from(height) * scale).round() as u32).max(1),
        );
        std::borrow::Cow::Owned(image::imageops::resize(
            img,
            target.0,
            target.1,
            image::imageops::FilterType::Triangle,
        ))
    } else {
        std::borrow::Cow::Borrowed(img)
    };
    let stats = Stats::gather(&small);

    // White balance first: it changes the channel means everything else sees.
    let (temp, tint) = wb_sliders(&stats);
    state.wb_temp = temp;
    state.wb_tint = tint;

    let (black, white) = levels_range(&stats);
    state.levels_black = (black * 255.0).round() as f32;
    state.levels_white = (white * 255.0)
        .round()
        .clamp(f64::from(state.levels_black) + 1.0, 255.0) as f32;
    let range = (white - black).max(0.001);

    // Midtones: pick the gamma that moves the post-levels median to the target.
    let median = ((stats.quantile(0.5) - black) / range).clamp(0.0, 1.0);
    let gamma = if median > 0.02 && median < 0.98 {
        (median.ln() / MEDIAN_TARGET.ln()).clamp(0.4, 3.0)
    } else {
        1.0
    };
    state.levels_gamma = gamma as f32;

    // Remap helper mirrors the shader: levels stretch, then 1/gamma exponent.
    let remap = |v: f64| (((v - black) / range).clamp(0.0, 1.0)).powf(1.0 / gamma);

    // Contrast: give the input the spread it is missing. Measured before the
    // level stretch, because after it every image looks equally wide; a camera
    // JPEG needs almost none, a flat develop needs a lot.
    let input_iqr = stats.quantile(0.75) - stats.quantile(0.25);
    state.contrast = ((CONTRAST_TARGET_IQR - input_iqr) * CONTRAST_SLOPE).clamp(0.0, 60.0) as f32;

    // Shadows/highlights: recover only when a tail is genuinely bunched.
    let q10 = remap(stats.quantile(0.10));
    if q10 < 0.08 {
        state.shadows = ((0.08 - q10) * 350.0).clamp(0.0, 45.0) as f32;
    }
    let q90 = remap(stats.quantile(0.90));
    if q90 > 0.94 {
        state.highlights = (-(q90 - 0.94) * 350.0).clamp(-45.0, 0.0) as f32;
    }

    // Vibrance: gentle, and larger when the image carries little colour already.
    state.vibrance = ((0.25 - stats.mean_chroma) * 120.0).clamp(0.0, 25.0) as f32;

    state
}

/// White-balance sliders that neutralise `mean_rgb`, solved against the shader's
/// gain model so the chosen TEMP/TINT really do cancel the cast.
fn wb_sliders(stats: &Stats) -> (f32, f32) {
    if stats.neutral_n < WB_MIN_SAMPLE * stats.total {
        return (0.0, 0.0);
    }
    let luma_w = [0.2126, 0.7152, 0.0722];
    let mut best = (0.0f32, 0.0f32, f64::MAX);
    let mut t = -60.0f64;
    while t <= 60.0 {
        let mut g = -60.0f64;
        while g <= 60.0 {
            let mut gains = [
                1.0 + 0.30 * (t / 100.0) + 0.10 * (g / 100.0),
                1.0 - 0.20 * (g / 100.0),
                1.0 - 0.30 * (t / 100.0) + 0.10 * (g / 100.0),
            ];
            let norm = luma_w[0] * gains[0] + luma_w[1] * gains[1] + luma_w[2] * gains[2];
            for value in &mut gains {
                *value = (*value / norm).max(1e-6).powf(0.4545);
            }
            let cast = |k: usize| stats.mean_rgb[k] * gains[k];
            let (mx, mn) = {
                let p = [cast(0), cast(1), cast(2)];
                (
                    p[0].max(p[1]).max(p[2]),
                    p[0].min(p[1]).min(p[2]),
                )
            };
            let error = if mx > 1e-9 { mx / mn - 1.0 } else { 0.0 };
            if error < best.2 {
                best = (t as f32, g as f32, error);
            }
            g += 5.0;
        }
        t += 5.0;
    }
    (
        (best.0 * WB_STRENGTH).clamp(-WB_LIMIT, WB_LIMIT),
        (best.1 * WB_STRENGTH).clamp(-WB_LIMIT, WB_LIMIT),
    )
}

/// Black/white points at the 0.3/99.7 percentiles, widened if the image is flat.
fn levels_range(stats: &Stats) -> (f64, f64) {
    let mut black = stats.quantile(0.003);
    let mut white = stats.quantile(0.997);
    if white - black < 0.08 {
        black = (black - 0.04).max(0.0);
        white = (white + 0.04).min(1.0);
    }
    (black, white)
}

struct Stats {
    hist: [u32; 256],
    total: f64,
    /// Channel means over near-neutral mid-tone pixels, for white balance.
    mean_rgb: [f64; 3],
    neutral_n: f64,
    /// Mean channel spread — a rough "how colourful is this" measure.
    mean_chroma: f64,
}

impl Stats {
    fn gather(img: &RgbaImage) -> Self {
        let mut hist = [0u32; 256];
        let mut sum = [0.0f64; 3];
        let mut neutral_n = 0.0f64;
        let mut chroma = 0.0f64;
        let mut total = 0.0f64;
        for px in img.pixels() {
            let r = f64::from(px[0]) / 255.0;
            let g = f64::from(px[1]) / 255.0;
            let b = f64::from(px[2]) / 255.0;
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            hist[(luma * 255.0).round().clamp(0.0, 255.0) as usize] += 1;
            let (mx, mn) = (r.max(g).max(b), r.min(g).min(b));
            chroma += mx - mn;
            total += 1.0;
            let saturation = if mx > 1e-6 { (mx - mn) / mx } else { 0.0 };
            if (0.15..0.85).contains(&luma) && saturation < 0.22 {
                sum[0] += r;
                sum[1] += g;
                sum[2] += b;
                neutral_n += 1.0;
            }
        }
        let mean_rgb = if neutral_n > 0.0 {
            [sum[0] / neutral_n, sum[1] / neutral_n, sum[2] / neutral_n]
        } else {
            [0.5, 0.5, 0.5]
        };
        Self {
            hist,
            total,
            mean_rgb,
            neutral_n,
            mean_chroma: chroma / total.max(1.0),
        }
    }

    fn quantile(&self, q: f64) -> f64 {
        let target = q * self.total;
        let mut acc = 0.0;
        for (value, &count) in self.hist.iter().enumerate() {
            acc += f64::from(count);
            if acc >= target {
                return value as f64 / 255.0;
            }
        }
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    fn image_from(width: u32, height: u32, f: impl Fn(u32, u32) -> [u8; 3]) -> RgbaImage {
        let mut img = RgbaImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let [r, g, b] = f(x, y);
                img.put_pixel(x, y, Rgba([r, g, b, 255]));
            }
        }
        img
    }

    #[test]
    fn empty_image_is_neutral() {
        let state = suggest_edit(&RgbaImage::new(0, 0));
        assert_eq!(state.levels_black, 0.0);
        assert_eq!(state.levels_white, 255.0);
    }

    #[test]
    fn flat_dark_image_trims_levels_to_its_range() {
        // A dark, low-contrast grey field: levels should close onto its range.
        let img = image_from(64, 64, |x, _| {
            let v = 40 + u8::try_from(x % 24).unwrap();
            [v, v, v]
        });
        let state = suggest_edit(&img);
        assert!(
            (30.0..=45.0).contains(&state.levels_black),
            "black {}",
            state.levels_black
        );
        assert!(
            (60.0..=70.0).contains(&state.levels_white),
            "white {}",
            state.levels_white
        );
    }

    #[test]
    fn mostly_dark_image_lifts_midtones_and_adds_contrast() {
        // 80% dark midtones with a bright fringe: the median sits far below the
        // middle after levels, so the gamma must pull it up, and the spread is
        // narrow enough to want contrast.
        let img = image_from(64, 64, |x, y| {
            let v = if (x + y) % 5 == 0 {
                200
            } else {
                20 + u8::try_from(x % 40).unwrap()
            };
            [v, v, v]
        });
        let state = suggest_edit(&img);
        assert!(state.levels_gamma > 1.1, "gamma {}", state.levels_gamma);
        assert!(state.contrast > 0.0, "contrast {}", state.contrast);
    }

    #[test]
    fn bright_image_does_not_over_brighten() {
        let img = image_from(64, 64, |_, _| [220, 220, 220]);
        let state = suggest_edit(&img);
        assert!(state.levels_gamma <= 1.2, "gamma {}", state.levels_gamma);
    }

    #[test]
    fn warm_cast_is_cooled() {
        // A near-neutral scene pushed warm: red high, blue low, but muted enough
        // to still count as neutral. TEMP cools it.
        let img = image_from(64, 64, |x, _| {
            let v = 120 + u8::try_from(x % 40).unwrap();
            [v.saturating_add(12), v, v.saturating_sub(12)]
        });
        let state = suggest_edit(&img);
        assert!(state.wb_temp < 0.0, "temp {}", state.wb_temp);
        assert!((-60.0..=60.0).contains(&state.wb_temp));
    }

    #[test]
    fn colourful_image_leaves_white_balance_alone() {
        // Nearly every pixel is saturated, so there is no neutral to trust.
        let img = image_from(64, 64, |x, y| {
            if (x + y) % 2 == 0 {
                [255, 0, 0]
            } else {
                [0, 0, 255]
            }
        });
        let state = suggest_edit(&img);
        assert_eq!(state.wb_temp, 0.0);
        assert_eq!(state.wb_tint, 0.0);
    }

    #[test]
    fn grey_image_gets_no_white_balance() {
        let img = image_from(64, 64, |_, _| [128, 128, 128]);
        let state = suggest_edit(&img);
        assert!(state.wb_temp.abs() < 1.0, "temp {}", state.wb_temp);
        assert!(state.wb_tint.abs() < 1.0, "tint {}", state.wb_tint);
    }

    /// `AUTO_SEED_DIR=<pairs dir>` writes `seeds.csv` — the slider vector the
    /// seed picks for each `*_in.jpg`. The trainer uses it as the base the
    /// network predicts a residual on.
    #[test]
    fn seed_probe() {
        let Ok(dir) = std::env::var("AUTO_SEED_DIR") else {
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        let mut entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .map(|name| name.to_string_lossy().ends_with("_in.jpg"))
                    .unwrap_or(false)
            })
            .collect();
        entries.sort();
        let mut rows = String::from(
            "id,wb_temp,wb_tint,levels_black,levels_gamma,levels_white,contrast,blacks,\
             shadows,highlights,whites,brightness,saturation,vibrance,exposure\n",
        );
        for path in entries {
            let Some(img) = crate::imgload::load_rgba(&path, 512) else {
                continue;
            };
            let stem = path.file_stem().unwrap().to_string_lossy();
            let id = stem.trim_end_matches("_in");
            let s = suggest_edit(&img);
            rows.push_str(&format!(
                "{id},{:.2},{:.2},{:.0},{:.3},{:.0},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.3}\n",
                s.wb_temp,
                s.wb_tint,
                s.levels_black,
                s.levels_gamma,
                s.levels_white,
                s.contrast,
                s.blacks,
                s.shadows,
                s.highlights,
                s.whites,
                s.brightness,
                s.saturation,
                s.vibrance,
                s.exposure
            ));
        }
        std::fs::write(dir.join("seeds.csv"), rows).unwrap();
    }
    #[test]
    fn auto_probe() {
        let Ok(spec) = std::env::var("AUTO_IMAGES") else {
            return;
        };
        let model = model::Residual::load();
        if model.is_none() {
            eprintln!("(no models/auto_residual.onnx — seed only)");
        }
        for path in spec.split(':') {
            let Some(img) = crate::imgload::load_rgba(std::path::Path::new(path), 512) else {
                eprintln!("{path}: load failed");
                continue;
            };
            let seed = suggest_edit(&img);
            let show = |tag: &str, s: &EditState| {
                eprintln!(
                    "  {tag}: temp={:+.0} tint={:+.0} levels={:.0}/{:.2}/{:.0} contrast={:.0} \
                     bright={:.0} sat={:.0} vib={:.0}",
                    s.wb_temp,
                    s.wb_tint,
                    s.levels_black,
                    s.levels_gamma,
                    s.levels_white,
                    s.contrast,
                    s.brightness,
                    s.saturation,
                    s.vibrance
                );
            };
            eprintln!("{path}:");
            show("seed", &seed);
            let refined = if let Some(model) = model.as_ref() {
                let refined = suggest_edit_with(&img, Some(model));
                show("refined", &refined);
                Some(refined)
            } else {
                None
            };
            if let Ok(dir) = std::env::var("AUTO_SAVE_DIR") {
                let stem = std::path::Path::new(path)
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "image".into());
                let dir = std::path::Path::new(&dir);
                let _ = std::fs::create_dir_all(dir);
                let _ = img.save(dir.join(format!("{stem}_in.png")));
                let _ = pipeline::render(&img, &seed).save(dir.join(format!("{stem}_seed.png")));
                if let Some(refined) = refined {
                    let _ = pipeline::render(&img, &refined).save(dir.join(format!("{stem}_auto.png")));
                }
            }
        }
    }
}
