//! Learned residual on top of the analytic seed (see `docs/auto-enhance.md`).
//!
//! A small CNN, trained offline on MIT-Adobe FiveK (`tools/fivek/`), predicts a
//! bounded correction to the seed's sliders. The runtime then applies those
//! sliders through the normal GPU shader — no LUT, no image output.
//!
//! The slider order, scales and limits here MUST match `tools/fivek/train.py`.

use crate::processor::EditState;
use image::RgbaImage;
use tract::prelude::*;

const SIZE: u32 = 256;

/// Where the exported model lives, relative to the crate. Overridable so a
/// retrained model can be swapped in without rebuilding.
fn model_path() -> std::path::PathBuf {
    if let Ok(path) = std::env::var("AUTO_MODEL_ONNX") {
        return path.into();
    }
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join("auto_residual.onnx")
}

/// Number of sliders the model predicts.
pub const COUNT: usize = 14;

/// How far the model may move each seeded slider. Kept tight so an
/// out-of-distribution photo gets a polish, not an overhaul.
const RESIDUAL_SCALE: [f32; COUNT] = [
    16.0, 16.0, // wb_temp, wb_tint
    32.0, 0.6, 32.0, // levels_black, levels_gamma, levels_white
    20.0, // contrast
    24.0, 24.0, 24.0, 24.0, // blacks, shadows, highlights, whites
    28.0, // brightness
    14.0, 12.0, // saturation, vibrance
    0.32, // exposure
];

/// Per-slider magnitude used to hand the seed to the network normalised.
const SEED_NORM: [f32; COUNT] = [
    50.0, 50.0, 128.0, 1.5, 128.0, 50.0, 60.0, 60.0, 60.0, 60.0, 70.0, 35.0, 30.0, 0.8,
];

const LIMITS: [(f32, f32); COUNT] = [
    (-100.0, 100.0),
    (-100.0, 100.0),
    (0.0, 254.0),
    (0.1, 5.0),
    (1.0, 255.0),
    (-100.0, 100.0),
    (-100.0, 100.0),
    (-100.0, 100.0),
    (-100.0, 100.0),
    (-100.0, 100.0),
    (-100.0, 100.0),
    (-100.0, 100.0),
    (-100.0, 100.0),
    (-3.0, 3.0),
];

pub struct Residual {
    runnable: tract::Runnable,
}

impl Residual {
    pub fn load() -> Option<Self> {
        let path = model_path();
        let Ok(bytes) = std::fs::read(&path) else {
            eprintln!("auto: cannot read {}", path.display());
            return None;
        };
        let onnx = match tract::onnx() {
            Ok(onnx) => onnx,
            Err(error) => {
                eprintln!("auto: onnx backend unavailable: {error}");
                return None;
            }
        };
        let model = match onnx.load_buffer(&bytes) {
            Ok(model) => model,
            Err(error) => {
                eprintln!("auto: could not parse model: {error}");
                return None;
            }
        };
        let typed = match model.into_model() {
            Ok(typed) => typed,
            Err(error) => {
                eprintln!("auto: could not type model: {error}");
                return None;
            }
        };
        let runnable = match typed.into_runnable() {
            Ok(runnable) => runnable,
            Err(error) => {
                eprintln!("auto: could not build runnable: {error}");
                return None;
            }
        };
        Some(Self { runnable })
    }

    pub fn residual(&self, img: &RgbaImage, seed: &[f32; COUNT]) -> Option<[f32; COUNT]> {
        if img.width() == 0 || img.height() == 0 {
            return None;
        }
        let resized = image::imageops::resize(
            img,
            SIZE,
            SIZE,
            image::imageops::FilterType::Triangle,
        );
        let plane = (SIZE * SIZE) as usize;
        let mut image = vec![0.0f32; plane * 3];
        for (index, px) in resized.pixels().enumerate() {
            for channel in 0..3 {
                image[channel * plane + index] = f32::from(px[channel]) / 255.0;
            }
        }
        let mut seed_norm = vec![0.0f32; COUNT];
        for (index, value) in seed.iter().enumerate() {
            seed_norm[index] = value / SEED_NORM[index];
        }
        let shape = [1usize, 3, SIZE as usize, SIZE as usize];
        let image = tract::Tensor::from_slice(&shape, &image).ok()?;
        let seed = tract::Tensor::from_slice(&[1usize, COUNT], &seed_norm).ok()?;
        let output = self.runnable.run(vec![image, seed]).ok()?;
        let values = output.first()?.as_slice::<f32>().ok()?;
        if values.len() != COUNT || values.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let mut out = [0.0f32; COUNT];
        out.copy_from_slice(values);
        Some(out)
    }
}

/// `clamp(seed + residual * scale)` written back into the state.
///
/// `AUTO_STRENGTH` (0..1, default 1) scales the whole correction, for tuning how
/// far past the seed Auto is allowed to go.
pub fn refine(state: &mut EditState, residual: &[f32; COUNT]) {
    let strength: f32 = std::env::var("AUTO_STRENGTH")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1.0);
    let seed = super::state_to_vector(state);
    let mut values = seed;
    for index in 0..COUNT {
        let moved = seed[index] + residual[index] * RESIDUAL_SCALE[index] * strength;
        values[index] = moved.clamp(LIMITS[index].0, LIMITS[index].1);
    }
    super::apply_vector(state, &values);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trained_model_predicts_finite_residual() {
        let Some(model) = Residual::load() else {
            eprintln!("skipping: no models/auto_residual.onnx yet");
            return;
        };
        let img = RgbaImage::from_pixel(64, 64, image::Rgba([96, 120, 140, 255]));
        let residual = model
            .residual(&img, &[0.0; COUNT])
            .expect("model should run");
        assert!(residual.iter().all(|value| value.is_finite()));
    }

    /// `AUTO_MODEL_CHECK=/tmp/model_input.png` prints the residual for a zero
    /// seed, for a cross-check against onnxruntime and the training renderer.
    #[test]
    fn model_probe() {
        let Ok(path) = std::env::var("AUTO_MODEL_CHECK") else {
            return;
        };
        let Some(model) = Residual::load() else {
            eprintln!("skipping: no model");
            return;
        };
        let Some(img) = image::open(&path).ok().map(|image| image.to_rgba8()) else {
            eprintln!("could not load {path}");
            return;
        };
        eprintln!("residual {:?}", model.residual(&img, &[0.0; COUNT]));
    }
}
