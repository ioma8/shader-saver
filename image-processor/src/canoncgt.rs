//! CanonCGT reference grading through a canonical color pivot.
//!
//! The pretrained network predicts a target-to-neutral 17³ LUT and a
//! neutral-to-reference 17³ LUT from 224px views. They are composed, blended
//! with the conservative profile transfer, smoothed, and upsampled to the
//! editor's 33³ LUT. Full-resolution pixels never pass through the network.

use tract::prelude::*;

use crate::processor::{compose_luts, sample_cube, voxel_rgb};

const MODEL: &[u8] = include_bytes!("../models/canoncgt_lut.onnx");
// The encoders still work at 224px internally. A larger intermediate preserves
// local color mixtures when the canonical LUT is rendered before restyling.
const INPUT_SIZE: u32 = 448;
const GRID: usize = 17;
const OUTPUT_GRID: usize = 33;
const CANON_STRENGTH: f32 = 1.0;

pub struct CanonCgt {
    runnable: Runnable,
}

impl CanonCgt {
    pub fn load() -> Option<Self> {
        Some(Self {
            runnable: tract::onnx()
                .ok()?
                .load_buffer(MODEL)
                .ok()?
                .into_model()
                .ok()?
                .into_runnable()
                .ok()?,
        })
    }

    pub fn predict_lut(
        &self,
        target: &image::RgbaImage,
        reference: &image::RgbaImage,
        fallback: &[f32],
    ) -> Option<Vec<f32>> {
        if target.width() == 0 || target.height() == 0 {
            return None;
        }
        let predicted = self.predict_direct_lut(target, reference)?;
        let predicted = scale_lut(&predicted, OUTPUT_GRID, CANON_STRENGTH);
        let blended = blend_by_confidence(fallback, &predicted, OUTPUT_GRID);
        Some(smooth_lut(&blended, OUTPUT_GRID))
    }

    fn predict_direct_lut(
        &self,
        target: &image::RgbaImage,
        reference: &image::RgbaImage,
    ) -> Option<Vec<f32>> {
        let outputs = self
            .runnable
            .run([to_tensor(target)?, to_tensor(reference)?])
            .ok()?;
        let canonical = planar_to_interleaved(outputs[0].as_slice::<f32>().ok()?, GRID)?;
        let restyle = planar_to_interleaved(outputs[1].as_slice::<f32>().ok()?, GRID)?;
        if canonical
            .iter()
            .chain(&restyle)
            .any(|value| !value.is_finite())
        {
            return None;
        }
        let composed = compose_luts(&canonical, &restyle, GRID);
        Some(
            (0..OUTPUT_GRID.pow(3))
                .flat_map(|index| sample_cube(&composed, GRID, voxel_rgb(index, OUTPUT_GRID)))
                .collect(),
        )
    }
}

fn scale_lut(lut: &[f32], n: usize, strength: f32) -> Vec<f32> {
    lut.chunks_exact(3)
        .enumerate()
        .flat_map(|(index, value)| {
            let identity = voxel_rgb(index, n);
            std::array::from_fn::<_, 3, _>(|channel| {
                identity[channel] + (value[channel] - identity[channel]) * strength
            })
        })
        .collect()
}

fn blend_by_confidence(fallback: &[f32], predicted: &[f32], n: usize) -> Vec<f32> {
    let identity: Vec<f32> = (0..n.pow(3))
        .flat_map(|index| voxel_rgb(index, n))
        .collect();
    identity
        .chunks_exact(3)
        .zip(fallback.chunks_exact(3))
        .zip(predicted.chunks_exact(3))
        .flat_map(|((identity, fallback), predicted)| {
            let fallback_delta = std::array::from_fn::<_, 3, _>(|i| fallback[i] - identity[i]);
            let predicted_delta = std::array::from_fn::<_, 3, _>(|i| predicted[i] - identity[i]);
            let magnitude = |delta: [f32; 3]| delta.iter().map(|v| v * v).sum::<f32>().sqrt();
            let (a, b) = (magnitude(fallback_delta), magnitude(predicted_delta));
            let total = a + b;
            std::array::from_fn::<_, 3, _>(|i| {
                let delta = if total > 1e-6 {
                    (fallback_delta[i] * a + predicted_delta[i] * b) / total
                } else {
                    0.0
                };
                (identity[i] + delta).clamp(0.0, 1.0)
            })
        })
        .collect()
}

fn smooth_lut(lut: &[f32], n: usize) -> Vec<f32> {
    let at = |r: usize, g: usize, b: usize, c: usize| (r + g * n + b * n * n) * 3 + c;
    let clamp = |value: isize| value.clamp(0, n as isize - 1) as usize;
    let mut output = vec![0.0; lut.len()];
    for b in 0..n {
        for g in 0..n {
            for r in 0..n {
                for channel in 0..3 {
                    let mut sum = 0.0;
                    for db in -1..=1 {
                        for dg in -1..=1 {
                            for dr in -1..=1 {
                                sum += lut[at(
                                    clamp(r as isize + dr),
                                    clamp(g as isize + dg),
                                    clamp(b as isize + db),
                                    channel,
                                )];
                            }
                        }
                    }
                    output[at(r, g, b, channel)] = sum / 27.0;
                }
            }
        }
    }
    output
}

fn to_tensor(image: &image::RgbaImage) -> Option<Tensor> {
    let resized = image::imageops::resize(
        image,
        INPUT_SIZE,
        INPUT_SIZE,
        image::imageops::FilterType::Triangle,
    );
    let plane = (INPUT_SIZE * INPUT_SIZE) as usize;
    let mut channels = vec![0.0; plane * 3];
    for (index, pixel) in resized.pixels().enumerate() {
        for channel in 0..3 {
            channels[channel * plane + index] = f32::from(pixel[channel]) / 255.0;
        }
    }
    Tensor::from_slice(&[1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize], &channels).ok()
}

fn planar_to_interleaved(planar: &[f32], n: usize) -> Option<Vec<f32>> {
    let cells = n.pow(3);
    if planar.len() != cells * 3 {
        return None;
    }
    Some(
        (0..cells)
            .flat_map(|cell| std::array::from_fn::<_, 3, _>(|c| planar[c * cells + cell]))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_model_produces_a_safe_lut() {
        let model = CanonCgt::load().expect("embedded CanonCGT model");
        let target = image::RgbaImage::from_fn(64, 64, |x, y| {
            image::Rgba([(x * 4) as u8, (y * 4) as u8, 128, 255])
        });
        let reference = image::RgbaImage::from_pixel(64, 64, image::Rgba([200, 140, 90, 255]));
        let fallback: Vec<f32> = (0..OUTPUT_GRID.pow(3))
            .flat_map(|index| voxel_rgb(index, OUTPUT_GRID))
            .collect();
        let lut = model.predict_lut(&target, &reference, &fallback).unwrap();
        assert_eq!(lut.len(), OUTPUT_GRID.pow(3) * 3);
        assert!(lut
            .iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value)));
    }

    #[test]
    fn calibrate_strength_on_real_photos() {
        let Ok(folder) = std::env::var("LOOK_TRAIN_FOLDER") else {
            return;
        };
        let mut paths: Vec<_> = std::fs::read_dir(folder)
            .unwrap()
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| {
                        matches!(extension.to_ascii_lowercase().as_str(), "jpg" | "jpeg")
                    })
            })
            .take(40)
            .collect();
        paths.sort();
        let images: Vec<_> = paths
            .iter()
            .filter_map(|path| crate::imgload::load_rgba(path, 256))
            .collect();
        assert!(images.len() >= 12, "need at least 12 JPEGs");
        let model = CanonCgt::load().unwrap();
        let fallback_model = crate::look_model::LookModel::train();
        let strengths: Vec<f32> = (4..=16).map(|value| value as f32 / 10.0).collect();
        let mut errors = vec![0.0; strengths.len()];
        let mut baseline = 0.0;
        let mut fallback_error = 0.0;
        let trials = images.len().min(20);
        for index in 0..trials {
            let target = perturb(&images[index], index);
            let reference = grade(&images[(index * 7 + 3) % images.len()], index);
            let desired = grade(&target, index);
            let direct = model.predict_direct_lut(&target, &reference).unwrap();
            let current_profile = crate::processor::LookProfile::measure(
                target.as_raw(),
                target.width(),
                target.height(),
                &[],
            )
            .unwrap();
            let reference_profile = crate::processor::LookProfile::measure(
                reference.as_raw(),
                reference.width(),
                reference.height(),
                &[],
            )
            .unwrap();
            let state = crate::processor::EditState {
                look: vec![fallback_model.predict(&current_profile, &reference_profile)],
                ..Default::default()
            };
            let fallback = crate::processor::baked_lut(&state).unwrap();
            baseline += image_error(&target, &desired);
            fallback_error += rendered_error(&target, &desired, &fallback);
            for (error, &strength) in errors.iter_mut().zip(&strengths) {
                let direct = scale_lut(&direct, OUTPUT_GRID, strength);
                let lut = smooth_lut(
                    &blend_by_confidence(&fallback, &direct, OUTPUT_GRID),
                    OUTPUT_GRID,
                );
                *error += rendered_error(&target, &desired, &lut);
            }
        }
        baseline /= trials as f32;
        fallback_error /= trials as f32;
        for error in &mut errors {
            *error /= trials as f32;
        }
        let (best_index, best_error) = errors
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        eprintln!(
            "CanonCGT calibration over {trials} real JPEGs: baseline {baseline:.5}; fallback {fallback_error:.5}; strength/error {:?}; best {:.1} -> {:.5}",
            strengths.iter().copied().zip(errors.iter().copied()).collect::<Vec<_>>(),
            strengths[best_index],
            best_error
        );
        assert!(*best_error < fallback_error);
    }

    #[test]
    fn benchmark_official_examples() {
        let Ok(folder) = std::env::var("CANONCGT_EXAMPLES") else {
            return;
        };
        let model = CanonCgt::load().unwrap();
        let mut direct_error = 0.0;
        let mut production_error = 0.0;
        let mut baseline_error = 0.0;
        let mut count = 0;
        let identity: Vec<f32> = (0..OUTPUT_GRID.pow(3))
            .flat_map(|index| voxel_rgb(index, OUTPUT_GRID))
            .collect();
        for index in 0..8 {
            let load = |part: &str| {
                crate::imgload::load_rgba(
                    &std::path::Path::new(&folder)
                        .join("samples")
                        .join(part)
                        .join(format!("{index:02}.png")),
                    512,
                )
            };
            let (Some(target), Some(reference), Some(expected)) =
                (load("inp"), load("ref"), load("out"))
            else {
                continue;
            };
            let direct = model.predict_direct_lut(&target, &reference).unwrap();
            direct_error += rendered_error(&target, &expected, &direct);
            let production = model.predict_lut(&target, &reference, &identity).unwrap();
            production_error += rendered_error(&target, &expected, &production);
            baseline_error += image_error(&target, &expected);
            count += 1;
        }
        assert_eq!(count, 8);
        eprintln!(
            "official CanonCGT examples: unchanged MAE {:.5}; direct LUT MAE {:.5}; production MAE {:.5}",
            baseline_error / count as f32,
            direct_error / count as f32,
            production_error / count as f32
        );
        assert!(direct_error < baseline_error);
    }

    fn perturb(image: &image::RgbaImage, seed: usize) -> image::RgbaImage {
        let exposure = [0.72, 0.86, 1.12, 1.28][seed % 4];
        let wb = [[1.10, 1.0, 0.90], [0.92, 1.0, 1.09], [1.04, 0.97, 1.0]][seed % 3];
        map_image(image, |rgb| {
            std::array::from_fn(|channel| (rgb[channel] * exposure * wb[channel]).clamp(0.0, 1.0))
        })
    }

    fn grade(image: &image::RgbaImage, seed: usize) -> image::RgbaImage {
        let gamma = [0.78, 0.90, 1.12, 1.28][seed % 4];
        let saturation = [0.72, 0.90, 1.18, 1.35][(seed / 2) % 4];
        let gains = [[1.08, 1.0, 0.91], [0.93, 1.0, 1.08], [1.04, 0.97, 1.02]][seed % 3];
        map_image(image, |rgb| {
            let toned = rgb.map(|value| value.powf(gamma));
            let luma = 0.2126 * toned[0] + 0.7152 * toned[1] + 0.0722 * toned[2];
            std::array::from_fn(|channel| {
                ((luma + (toned[channel] - luma) * saturation) * gains[channel]).clamp(0.0, 1.0)
            })
        })
    }

    fn map_image(
        image: &image::RgbaImage,
        transform: impl Fn([f32; 3]) -> [f32; 3],
    ) -> image::RgbaImage {
        image::RgbaImage::from_fn(image.width(), image.height(), |x, y| {
            let pixel = image.get_pixel(x, y);
            let output = transform(std::array::from_fn(|channel| {
                f32::from(pixel[channel]) / 255.0
            }));
            image::Rgba([
                (output[0] * 255.0).round() as u8,
                (output[1] * 255.0).round() as u8,
                (output[2] * 255.0).round() as u8,
                pixel[3],
            ])
        })
    }

    fn rendered_error(target: &image::RgbaImage, desired: &image::RgbaImage, lut: &[f32]) -> f32 {
        let rendered = map_image(target, |rgb| sample_cube(lut, OUTPUT_GRID, rgb));
        image_error(&rendered, desired)
    }

    fn image_error(a: &image::RgbaImage, b: &image::RgbaImage) -> f32 {
        a.pixels()
            .zip(b.pixels())
            .flat_map(|(a, b)| {
                (0..3).map(move |channel| u8::abs_diff(a[channel], b[channel]) as f32 / 255.0)
            })
            .sum::<f32>()
            / (a.width() * a.height() * 3) as f32
    }
}
