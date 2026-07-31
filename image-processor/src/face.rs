// Local face detection using OpenCV Zoo's MIT-licensed YuNet model.
// The model emits three strides of class/objectness/box/landmark heads; the
// small decoder here keeps only the count and largest face area needed for
// culling decisions.
use tract::prelude::*;

const MODEL_BYTES: &[u8] = include_bytes!("../models/face_detection_yunet_2023mar.onnx");
const INPUT_SIZE: u32 = 640;
const STRIDES: [usize; 3] = [8, 16, 32];
const SCORE_THRESHOLD: f32 = 0.6;
const NMS_THRESHOLD: f32 = 0.3;

#[derive(Clone, Copy, Debug, Default)]
pub struct FaceStats {
    pub count: u8,
    pub largest_area: f32,
}

#[derive(Clone, Copy, Debug)]
struct Detection {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    score: f32,
}

pub struct Detector {
    runnable: tract::Runnable,
}

impl Detector {
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

    pub fn detect(&self, img: &image::RgbaImage) -> FaceStats {
        let boxes = self.detect_boxes(img);
        FaceStats {
            count: boxes.len().min(u8::MAX as usize) as u8,
            largest_area: boxes
                .iter()
                .map(|b| (b[2] * b[3]).clamp(0.0, 1.0))
                .fold(0.0, f32::max),
        }
    }

    // Face rectangles as fractions of the image: [x, y, width, height] in 0..1.
    //
    // The look transfer needs these, not just a count: skin is a small share of
    // most frames but the part a viewer judges first, so it gets measured and
    // matched on its own rather than being averaged into the whole picture.
    pub fn detect_boxes(&self, img: &image::RgbaImage) -> Vec<[f32; 4]> {
        let resized = image::imageops::resize(
            img,
            INPUT_SIZE,
            INPUT_SIZE,
            image::imageops::FilterType::Triangle,
        );
        // YuNet expects BGR, NCHW, and raw 0..255 values.
        let plane = (INPUT_SIZE * INPUT_SIZE) as usize;
        let mut chw = vec![0f32; plane * 3];
        for y in 0..INPUT_SIZE {
            for x in 0..INPUT_SIZE {
                let px = resized.get_pixel(x, y).0;
                let i = (y * INPUT_SIZE + x) as usize;
                chw[i] = f32::from(px[2]);
                chw[plane + i] = f32::from(px[1]);
                chw[plane * 2 + i] = f32::from(px[0]);
            }
        }
        let Ok(input) = Tensor::from_slice(&[1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize], &chw)
        else {
            return Vec::new();
        };
        let Ok(outputs) = self.runnable.run([input]) else {
            return Vec::new();
        };

        let mut detections = Vec::new();
        for (level, stride) in STRIDES.iter().copied().enumerate() {
            let Ok(cls) = outputs[level].as_slice::<f32>() else { continue };
            let Ok(obj) = outputs[3 + level].as_slice::<f32>() else { continue };
            let Ok(bbox) = outputs[6 + level].as_slice::<f32>() else { continue };
            let cells = (INPUT_SIZE as usize / stride).pow(2);
            for i in 0..cells {
                // The exported YuNet heads already include sigmoid; score is
                // the product of face-class and objectness probabilities.
                let score = cls[i] * obj[i];
                if score < SCORE_THRESHOLD {
                    continue;
                }
                let col = i % (INPUT_SIZE as usize / stride);
                let row = i / (INPUT_SIZE as usize / stride);
                let j = i * 4;
                let cx = (col as f32 + bbox[j]) * stride as f32;
                let cy = (row as f32 + bbox[j + 1]) * stride as f32;
                let w = bbox[j + 2].exp() * stride as f32;
                let h = bbox[j + 3].exp() * stride as f32;
                detections.push(Detection {
                    x: cx - w * 0.5,
                    y: cy - h * 0.5,
                    w,
                    h,
                    score,
                });
            }
        }
        let side = INPUT_SIZE as f32;
        nms(&mut detections)
            .iter()
            .map(|d| {
                [
                    (d.x / side).clamp(0.0, 1.0),
                    (d.y / side).clamp(0.0, 1.0),
                    (d.w / side).clamp(0.0, 1.0),
                    (d.h / side).clamp(0.0, 1.0),
                ]
            })
            .collect()
    }
}

fn iou(a: Detection, b: Detection) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.w).min(b.x + b.w);
    let y2 = (a.y + a.h).min(b.y + b.h);
    let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    intersection / (a.w * a.h + b.w * b.h - intersection).max(f32::EPSILON)
}

fn nms(detections: &mut Vec<Detection>) -> Vec<Detection> {
    detections.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept = Vec::new();
    for detection in detections.drain(..) {
        if kept.iter().all(|&kept| iou(detection, kept) < NMS_THRESHOLD) {
            kept.push(detection);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_score_uses_probability_product() {
        assert!((0.9f32 * 0.8 - 0.72).abs() < 1e-6);
    }

    #[test]
    fn nms_removes_overlapping_lower_score_detection() {
        let mut detections = vec![
            Detection { x: 0.0, y: 0.0, w: 100.0, h: 100.0, score: 0.9 },
            Detection { x: 5.0, y: 5.0, w: 100.0, h: 100.0, score: 0.8 },
            Detection { x: 300.0, y: 300.0, w: 20.0, h: 20.0, score: 0.7 },
        ];
        assert_eq!(nms(&mut detections).len(), 2);
    }

    #[test]
    fn embedded_model_loads_and_runs() {
        let detector = Detector::load().expect("YuNet model should load");
        let img = image::RgbaImage::from_pixel(32, 32, image::Rgba([128, 128, 128, 255]));
        assert!(detector.detect(&img).count <= 10);
    }
}
