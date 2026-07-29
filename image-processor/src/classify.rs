// On-device controlled photo tagging: MobileCLIP-S0's 22 MB FP16 vision
// encoder, run locally through `tract`. It is compared to embeddings of a
// small fixed photo taxonomy rather than ImageNet's unwieldy object labels.
use tract::prelude::*;

const MODEL_BYTES: &[u8] = include_bytes!("../models/mobileclip_s0_vision_fp16.onnx");
const TAG_EMBEDDINGS: &[u8] = include_bytes!("../models/mobileclip_tags.f32");
const INPUT_SIZE: u32 = 256;
const EMBEDDING_SIZE: usize = 512;
const MIN_CONFIDENCE: f32 = 0.05;
const TAGS: [&str; 20] = [
    "people", "animals", "wedding", "cars", "indoors", "nature", "landscape", "city",
    "travel", "food", "sports", "architecture", "portrait", "family", "party", "beach",
    "mountains", "night", "computer graphics", "document",
];

pub struct Classifier {
    runnable: tract::Runnable,
    tag_embeddings: Vec<f32>,
}

impl Classifier {
    pub fn load() -> Option<Self> {
        let model = tract::onnx()
            .ok()?
            .load_buffer(MODEL_BYTES)
            .ok()?
            .into_model()
            .ok()?
            .into_runnable()
            .ok()?;
        let tag_embeddings: Vec<f32> = TAG_EMBEDDINGS
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        (tag_embeddings.len() == TAGS.len() * EMBEDDING_SIZE).then_some(Self {
            runnable: model,
            tag_embeddings,
        })
    }

    // At most two controlled tags, most confident first.
    pub fn classify(&self, img: &image::RgbaImage) -> Vec<String> {
        let scale = INPUT_SIZE as f32 / img.width().min(img.height()) as f32;
        let resized = image::imageops::resize(
            img,
            (img.width() as f32 * scale).round() as u32,
            (img.height() as f32 * scale).round() as u32,
            image::imageops::FilterType::Triangle,
        );
        let x = (resized.width() - INPUT_SIZE) / 2;
        let y = (resized.height() - INPUT_SIZE) / 2;
        let resized = image::imageops::crop_imm(&resized, x, y, INPUT_SIZE, INPUT_SIZE).to_image();
        let mut chw = vec![0f32; 3 * (INPUT_SIZE * INPUT_SIZE) as usize];
        let plane = (INPUT_SIZE * INPUT_SIZE) as usize;
        for y in 0..INPUT_SIZE {
            for x in 0..INPUT_SIZE {
                let px = resized.get_pixel(x, y).0;
                let idx = (y * INPUT_SIZE + x) as usize;
                for c in 0..3 {
                    chw[c * plane + idx] = f32::from(px[c]) / 255.0;
                }
            }
        }

        let shape = [1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize];
        let Some(input) = tract::Tensor::from_slice(&shape, &chw).ok() else {
            return Vec::new();
        };
        let Some(outputs) = self.runnable.run([input]).ok() else {
            return Vec::new();
        };
        let Some(embedding) = outputs[0].as_slice::<f32>().ok() else {
            return Vec::new();
        };

        top_tags(embedding, &self.tag_embeddings)
    }
}

fn top_tags(embedding: &[f32], tag_embeddings: &[f32]) -> Vec<String> {
    if embedding.len() != EMBEDDING_SIZE || tag_embeddings.len() != TAGS.len() * EMBEDDING_SIZE {
        return Vec::new();
    }
    let image_norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
    if image_norm == 0.0 {
        return Vec::new();
    }
    let scores: Vec<f32> = tag_embeddings
        .chunks_exact(EMBEDDING_SIZE)
        .map(|tag| {
            let norm = tag.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm == 0.0 {
                f32::NEG_INFINITY
            } else {
                embedding.iter().zip(tag).map(|(a, b)| a * b).sum::<f32>() / (image_norm * norm)
            }
        })
        .collect();
    let max = scores.iter().copied().fold(f32::MIN, f32::max);
    // CLIP's temperature keeps unrelated tags from looking confident merely
    // because the vocabulary is small.
    let exp: Vec<f32> = scores.iter().map(|&s| ((s - max) * 14.3).exp()).collect();
    let sum: f32 = exp.iter().sum();
    if sum <= 0.0 {
        return Vec::new();
    }

    let mut ranked: Vec<(usize, f32)> = exp.iter().map(|&e| e / sum).enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

    ranked
        .into_iter()
        .take(2)
        .filter(|&(_, p)| p >= MIN_CONFIDENCE)
        .filter_map(|(i, _)| TAGS.get(i))
        .map(|tag| (*tag).to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taxonomy_is_small_and_unique() {
        assert_eq!(TAGS.len(), 20);
        let mut tags = TAGS.to_vec();
        tags.sort_unstable();
        tags.dedup();
        assert_eq!(tags.len(), TAGS.len());
    }

    #[test]
    fn top_tags_picks_the_confident_class_first() {
        let mut embedding = vec![0.0f32; EMBEDDING_SIZE];
        embedding[0] = 1.0;
        let mut tags = vec![0.0f32; TAGS.len() * EMBEDDING_SIZE];
        tags[0] = 1.0;
        assert_eq!(top_tags(&embedding, &tags), vec!["people".to_string()]);
    }

    #[test]
    fn top_tags_caps_at_two() {
        let mut embedding = vec![0.0f32; EMBEDDING_SIZE];
        embedding[0] = 1.0;
        embedding[1] = 1.0;
        let mut tags = vec![0.0f32; TAGS.len() * EMBEDDING_SIZE];
        tags[0] = 1.0;
        tags[EMBEDDING_SIZE] = 1.0;
        tags[EMBEDDING_SIZE + 1] = 0.8;
        assert!(top_tags(&embedding, &tags).len() <= 2);
    }

    #[test]
    fn top_tags_empty_when_nothing_confident() {
        assert!(top_tags(&[], &[]).is_empty());
    }

    // Loads and runs the real embedded model — the one true test that the
    // bundled ONNX bytes are valid and the preprocess/run/postprocess path
    // works end to end.
    #[test]
    fn embedded_model_loads_and_classifies() {
        let classifier = Classifier::load().expect("embedded model should load");
        let img = image::RgbaImage::from_pixel(32, 32, image::Rgba([128, 128, 128, 255]));
        let tags = classifier.classify(&img);
        assert!(tags.len() <= 2);
    }
}
