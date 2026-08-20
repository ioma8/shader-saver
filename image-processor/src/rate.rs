// On-device photo quality/aesthetic rating: NIMA (Neural Image Assessment,
// MobileNet backbone), converted from idealo/image-quality-assessment's
// official Apache-2.0 Keras weights (github.com/idealo/image-quality-assessment,
// models/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5) to ONNX and run
// locally via `tract`, same as the tagger in classify.rs.
use tract::prelude::*;

const MODEL_BYTES: &[u8] = include_bytes!("../models/nima_mobilenet_aesthetic.onnx");
const INPUT_SIZE: u32 = 224;

pub struct Rater {
    runnable: tract::Runnable,
}

impl Rater {
    pub fn load() -> Option<Self> {
        let runnable = tract::onnx()
            .ok()?
            .load_buffer(MODEL_BYTES)
            .ok()?
            .into_model()
            .ok()?
            .into_runnable()
            .ok()?;
        Some(Self { runnable })
    }

    // 1-5 star rating from NIMA's predicted quality distribution.
    pub fn rate(&self, img: &image::RgbaImage) -> Option<u8> {
        self.mean_score(img).map(stars_from_mean)
    }

    fn mean_score(&self, img: &image::RgbaImage) -> Option<f32> {
        let resized = image::imageops::resize(
            img,
            INPUT_SIZE,
            INPUT_SIZE,
            image::imageops::FilterType::Triangle,
        );
        // NHWC layout, [-1, 1] scaling — matches Keras's mobilenet.preprocess_input,
        // which is what the original model was trained and exported with.
        let mut hwc = vec![0f32; (INPUT_SIZE * INPUT_SIZE * 3) as usize];
        for y in 0..INPUT_SIZE {
            for x in 0..INPUT_SIZE {
                let px = resized.get_pixel(x, y).0;
                let idx = ((y * INPUT_SIZE + x) * 3) as usize;
                for (c, channel) in px.iter().take(3).enumerate() {
                    hwc[idx + c] = f32::from(*channel) / 127.5 - 1.0;
                }
            }
        }

        let shape = [1, INPUT_SIZE as usize, INPUT_SIZE as usize, 3];
        let input = tract::Tensor::from_slice(&shape, &hwc).ok()?;
        let outputs = self.runnable.run([input]).ok()?;
        let dist = outputs[0].as_slice::<f32>().ok()?;
        Some(mean_from_distribution(dist))
    }
}

// NIMA outputs a 10-way distribution over quality scores 1..10; its mean is
// the standard NIMA "mean score". These endpoints are the 5th and 95th
// percentiles from a uniformly spaced 400-image sample of COCO 2017 val.
const STAR_SCORE_MIN: f32 = 3.6477;
const STAR_SCORE_MAX: f32 = 4.8441;

fn mean_from_distribution(dist: &[f32]) -> f32 {
    let sum: f32 = dist.iter().sum();
    if sum <= 0.0 {
        return 0.0;
    }
    dist.iter()
        .enumerate()
        .map(|(i, &p)| (i as f32 + 1.0) * p)
        .sum::<f32>()
        / sum
}

fn stars_from_mean(mean: f32) -> u8 {
    (((mean - STAR_SCORE_MIN) / (STAR_SCORE_MAX - STAR_SCORE_MIN) * 4.0) + 1.0)
        .round()
        .clamp(1.0, 5.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stars_from_distribution(dist: &[f32]) -> u8 {
        stars_from_mean(mean_from_distribution(dist))
    }

    #[test]
    fn stars_from_distribution_uses_coco_calibration() {
        assert_eq!(stars_from_mean(STAR_SCORE_MIN), 1);
        assert_eq!(stars_from_mean(4.1274), 3); // COCO sample median
        assert_eq!(stars_from_mean(STAR_SCORE_MAX), 5);

        let mut score_four = [0f32; 10];
        score_four[3] = 1.0;
        assert_eq!(stars_from_distribution(&score_four), 2);
    }

    // Loads and runs the real embedded model on a neutral-gray image — the
    // one true test that the bundled, converted ONNX bytes are valid and the
    // preprocess/run/postprocess path works end to end.
    #[test]
    fn embedded_model_loads_and_rates() {
        let rater = Rater::load().expect("embedded model should load");
        let img = image::RgbaImage::from_pixel(32, 32, image::Rgba([128, 128, 128, 255]));
        let stars = rater.rate(&img).expect("rating should succeed");
        assert!((1..=5).contains(&stars));
    }
}
