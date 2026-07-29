// Apache-2.0 Image-Adaptive-3DLUT, trained for general sRGB photo retouching
// on MIT-Adobe FiveK. The network predicts three weights; the GPU applies the
// resulting 33³ LUT exactly, at the source image's full resolution.
// Upstream: https://github.com/HuiZeng/Image-Adaptive-3DLUT
// Classifier SHA-256: 4f7775c22512fbb6f6c8caecffd60b74b57be454bd49d47eb72a7ea0350ecf14
use crate::processor::EditState;
use image::RgbaImage;
use tract::prelude::*;

const MODEL: &[u8] = include_bytes!("../models/photo_lut_classifier.onnx");
const SIZE: u32 = 256;

pub struct Enhancer {
    runnable: tract::Runnable,
}

impl Enhancer {
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

    pub fn suggest_edits(&self, img: &RgbaImage) -> Option<EditState> {
        if img.width() == 0 || img.height() == 0 {
            return None;
        }
        let resized =
            image::imageops::resize(img, SIZE, SIZE, image::imageops::FilterType::Triangle);
        let plane = (SIZE * SIZE) as usize;
        let mut input = vec![0.0f32; plane * 3];
        for (i, px) in resized.pixels().enumerate() {
            for c in 0..3 {
                input[c * plane + i] = f32::from(px[c]) / 255.0;
            }
        }
        let output = self
            .runnable
            .run([tract::Tensor::from_slice(&[1, 3, 256, 256], &input).ok()?])
            .ok()?;
        let values = output.first()?.as_slice::<f32>().ok()?;
        if values.len() != 3 || values.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(EditState {
            ai_lut_enabled: true,
            ai_lut_weights: [values[0], values[1], values[2]],
            ..EditState::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_model_predicts_finite_lut_weights() {
        let enhancer = Enhancer::load().expect("embedded model should load");
        let img = RgbaImage::from_pixel(64, 64, image::Rgba([64, 96, 128, 255]));
        let state = enhancer.suggest_edits(&img).expect("model should run");
        assert!(state.ai_lut_enabled);
        assert!(state.ai_lut_weights.iter().all(|v| v.is_finite()));
        let reference = [1.964_759_8, 2.110_048_3, -1.398_129_5];
        for (actual, expected) in state.ai_lut_weights.iter().zip(reference) {
            assert!((actual - expected).abs() < 1e-4);
        }
    }
}
