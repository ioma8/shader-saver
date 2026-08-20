//! Universal RAW development: rawler sensor decode plus the embedded
//! learned rendering model. Offline fitting of those models lives in
//! `raw_fit` (CLI-only: `--fit-raw-scurve` / `--fit-raw-render-model`).

use image::RgbaImage;
use std::path::Path;

/// 16-bit gamma-encoded RGBA image (dimensions and pixels travel together).
pub type Rgba16Image = image::ImageBuffer<image::Rgba<u16>, Vec<u16>>;

// --- Development ------------------------------------------------------------

const RENDER_QUANTILES: [f32; 44] = [
    0.0, 0.001, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12,
    0.14, 0.16, 0.18, 0.2, 0.22, 0.25, 0.28, 0.32, 0.36, 0.4, 0.44, 0.48, 0.5, 0.52, 0.56, 0.6,
    0.64, 0.68, 0.72, 0.76, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0,
];

pub(crate) const RENDER_FEATURES: usize = RENDER_QUANTILES.len() * 3;

pub(crate) const RENDER_HIDDEN: usize = 128;

pub(crate) const SPATIAL_GRID: usize = 64;

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct RawRenderModel {
    pub(crate) mean: Vec<f32>,
    pub(crate) scale: Vec<f32>,
    pub(crate) w1: Vec<f32>,
    pub(crate) b1: Vec<f32>,
    pub(crate) w2: Vec<f32>,
    pub(crate) b2: Vec<f32>,
    #[serde(default)]
    pub(crate) prototypes: Vec<Vec<f32>>,
    #[serde(default)]
    pub(crate) prototype_targets: Vec<Vec<f32>>,
    #[serde(default)]
    pub(crate) spatial_targets: Vec<Vec<f32>>,
}

impl RawRenderModel {
    pub(crate) fn embedded() -> Option<&'static Self> {
        static MODEL: std::sync::OnceLock<Option<RawRenderModel>> = std::sync::OnceLock::new();
        MODEL
            .get_or_init(|| {
                serde_json::from_str(include_str!("../models/raw_render_model.json")).ok()
            })
            .as_ref()
    }

    pub(crate) fn predict(&self, input: &[f32; RENDER_FEATURES]) -> [f32; RENDER_FEATURES] {
        if self.prototypes.len() == self.prototype_targets.len() && !self.prototypes.is_empty() {
            let mut nearest = Vec::new();
            for (index, prototype) in self.prototypes.iter().enumerate() {
                if prototype.len() != RENDER_FEATURES
                    || self.prototype_targets[index].len() != RENDER_FEATURES
                {
                    continue;
                }
                let distance = prototype
                    .iter()
                    .zip(input)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>();
                nearest.push((distance, index));
            }
            nearest.sort_by(|a, b| a.0.total_cmp(&b.0));
            if !nearest.is_empty() {
                let mut output = [0.0; RENDER_FEATURES];
                let count = nearest.len().min(3);
                let mut total = 0.0;
                for &(distance, index) in nearest.iter().take(count) {
                    let weight = 1.0 / (distance.sqrt() + 1e-4);
                    total += weight;
                    for (value, target) in output.iter_mut().zip(&self.prototype_targets[index]) {
                        *value += weight * target;
                    }
                }
                for value in &mut output {
                    *value /= total;
                }
                return output;
            }
        }
        if self.mean.len() != RENDER_FEATURES
            || self.scale.len() != RENDER_FEATURES
            || self.w1.len() != RENDER_HIDDEN * RENDER_FEATURES
            || self.b1.len() != RENDER_HIDDEN
            || self.w2.len() != RENDER_FEATURES * RENDER_HIDDEN
            || self.b2.len() != RENDER_FEATURES
        {
            return *input;
        }
        let mut hidden = [0.0; RENDER_HIDDEN];
        for (h, value) in hidden.iter_mut().enumerate() {
            let mut sum = self.b1[h];
            for i in 0..RENDER_FEATURES {
                sum += self.w1[h * RENDER_FEATURES + i] * (input[i] - self.mean[i]) * self.scale[i];
            }
            *value = sum.tanh();
        }
        let mut output = [0.0; RENDER_FEATURES];
        for (o, value) in output.iter_mut().enumerate() {
            let mut sum = self.b2[o];
            for h in 0..RENDER_HIDDEN {
                sum += self.w2[o * RENDER_HIDDEN + h] * hidden[h];
            }
            *value = 1.0 / (1.0 + (-sum).exp());
        }
        for channel in 0..3 {
            let values = &mut output
                [channel * RENDER_QUANTILES.len()..(channel + 1) * RENDER_QUANTILES.len()];
            values[0] = values[0].min(0.02);
            for index in 1..values.len() {
                values[index] = values[index].max(values[index - 1]);
            }
        }
        output
    }

    fn nearest_prototype(&self, input: &[f32; RENDER_FEATURES]) -> Option<(usize, f32)> {
        self.prototypes
            .iter()
            .enumerate()
            .filter(|(index, prototype)| {
                prototype.len() == RENDER_FEATURES
                    && self.prototype_targets.get(*index).is_some_and(|target| target.len() == RENDER_FEATURES)
            })
            .map(|(index, prototype)| {
                let distance = prototype
                    .iter()
                    .zip(input)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                (index, distance)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
    }
}

pub(crate) fn render_quantiles_u16(image: &Rgba16Image) -> [f32; RENDER_FEATURES] {
    let mut histograms = [[0u32; 4096]; 3];
    for pixel in image.pixels() {
        for channel in 0..3 {
            histograms[channel][pixel[channel] as usize >> 4] += 1;
        }
    }
    render_quantiles(&histograms)
}

pub(crate) fn render_quantiles_u8(image: &RgbaImage) -> [f32; RENDER_FEATURES] {
    let mut histograms = [[0u32; 4096]; 3];
    for pixel in image.pixels() {
        for channel in 0..3 {
            histograms[channel][pixel[channel] as usize * 16] += 1;
        }
    }
    render_quantiles(&histograms)
}

fn render_quantiles(histograms: &[[u32; 4096]; 3]) -> [f32; RENDER_FEATURES] {
    let mut output = [0.0; RENDER_FEATURES];
    for channel in 0..3 {
        let total = histograms[channel].iter().sum::<u32>() as f32;
        for (index, quantile) in RENDER_QUANTILES.iter().enumerate() {
            let target = total * quantile;
            let mut count = 0u32;
            let value = histograms[channel]
                .iter()
                .position(|amount| {
                    count += amount;
                    count as f32 >= target
                })
                .unwrap_or(4095);
            output[channel * RENDER_QUANTILES.len() + index] = value as f32 / 4095.0;
        }
    }
    output
}

/// Fully develop a RAW file to an 8-bit sRGB RGBA image.
pub fn develop_raw(path: &Path, max_dim: u32) -> Option<RgbaImage> {
    develop_generic_u16(path, max_dim).map(|image| {
        image::RgbaImage::from_fn(image.width(), image.height(), |x, y| {
            let pixel = image.get_pixel(x, y);
            image::Rgba([
                (pixel[0] >> 8) as u8,
                (pixel[1] >> 8) as u8,
                (pixel[2] >> 8) as u8,
                255,
            ])
        })
    })
}

/// Like `develop_raw`, but gamma-encoded 16-bit RGBA so the sensor's
/// precision survives into the 16-bit editor input.
pub fn develop_raw_u16(path: &Path, max_dim: u32) -> Option<Rgba16Image> {
    develop_generic_u16(path, max_dim)
}

fn develop_generic_u16(path: &Path, max_dim: u32) -> Option<Rgba16Image> {
    let mut image = decode_raw_u16(path, max_dim)?;
    if let Some(model) = RawRenderModel::embedded() {
        let source = render_quantiles_u16(&image);
        let target = model.predict(&source);
        apply_render_quantiles(&mut image, &source, &target);
        apply_spatial_residual(&mut image, model, &source);
        apply_local_render_contrast(&mut image, 0.0);
    }
    Some(image)
}

fn apply_spatial_residual(image: &mut Rgba16Image, model: &RawRenderModel, source: &[f32; RENDER_FEATURES]) {
    let Some((index, distance)) = model.nearest_prototype(source) else { return; };
    if distance > 0.18 || model.spatial_targets.get(index).is_none() {
        return;
    }
    let residual = &model.spatial_targets[index];
    let expected = SPATIAL_GRID * SPATIAL_GRID * 3;
    if residual.len() != expected { return; }
    let width = image.width().max(1) as f32;
    let height = image.height().max(1) as f32;
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let gx = (x as f32 / width * SPATIAL_GRID as f32 - 0.5).clamp(0.0, (SPATIAL_GRID - 1) as f32);
        let gy = (y as f32 / height * SPATIAL_GRID as f32 - 0.5).clamp(0.0, (SPATIAL_GRID - 1) as f32);
        let x0 = gx.floor() as usize; let y0 = gy.floor() as usize;
        let x1 = (x0 + 1).min(SPATIAL_GRID - 1); let y1 = (y0 + 1).min(SPATIAL_GRID - 1);
        let tx = gx - x0 as f32; let ty = gy - y0 as f32;
        for channel in 0..3 {
            let at = |xx: usize, yy: usize| residual[(yy * SPATIAL_GRID + xx) * 3 + channel];
            let value = at(x0, y0) * (1.0 - tx) * (1.0 - ty)
                + at(x1, y0) * tx * (1.0 - ty)
                + at(x0, y1) * (1.0 - tx) * ty
                + at(x1, y1) * tx * ty;
            pixel[channel] = (f32::from(pixel[channel]) + value * 65535.0).clamp(0.0, 65535.0) as u16;
        }
    }
}

/// Integral-image box blur (mean of a `radius`-squared window). Shared by the
/// runtime local-contrast pass and the offline rendering fits.
pub(crate) fn box_blur(values: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let stride = width + 1;
    let mut integral = vec![0.0f64; (width + 1) * (height + 1)];
    for y in 0..height {
        let mut row = 0.0f64;
        for x in 0..width {
            row += f64::from(values[y * width + x]);
            integral[(y + 1) * stride + x + 1] = integral[y * stride + x + 1] + row;
        }
    }
    let mut out = vec![0.0; values.len()];
    for y in 0..height {
        let y0 = y.saturating_sub(radius);
        let y1 = (y + radius + 1).min(height);
        for x in 0..width {
            let x0 = x.saturating_sub(radius);
            let x1 = (x + radius + 1).min(width);
            let sum = integral[y1 * stride + x1]
                - integral[y0 * stride + x1]
                - integral[y1 * stride + x0]
                + integral[y0 * stride + x0];
            out[y * width + x] = (sum / ((x1 - x0) * (y1 - y0)) as f64) as f32;
        }
    }
    out
}

fn apply_local_render_contrast(image: &mut Rgba16Image, amount: f32) {
    if amount <= 0.0 || image.width() < 8 || image.height() < 8 {
        return;
    }
    let width = image.width() as usize;
    let height = image.height() as usize;
    let luma: Vec<f32> = image
        .pixels()
        .map(|p| {
            (0.2126 * f32::from(p[0]) + 0.7152 * f32::from(p[1]) + 0.0722 * f32::from(p[2]))
                / 65535.0
        })
        .collect();
    let radius = (width.max(height) / 64).max(2);
    let blurred = box_blur(&luma, width, height, radius);
    for (pixel, (&value, &local)) in image.pixels_mut().zip(luma.iter().zip(&blurred)) {
        let adjusted =
            (value + amount * (value - local)).clamp(value * 0.7, (value * 1.4).min(1.0));
        let scale = if value > 1e-5 { adjusted / value } else { 1.0 };
        for channel in 0..3 {
            pixel[channel] = (f32::from(pixel[channel]) * scale).clamp(0.0, 65535.0) as u16;
        }
    }
}

pub(crate) fn decode_raw_u16(path: &Path, max_dim: u32) -> Option<Rgba16Image> {
    // Some third-party DNGs advertise CFA layouts that rawler does not yet
    // implement and panic inside its decoder. Reject those files cleanly.
    let raw = std::panic::catch_unwind(|| rawler::decode_file(path))
        .ok()?
        .ok()?;
    let developed = rawler::imgop::develop::RawDevelop::default()
        .develop_intermediate(&raw)
        .ok()?
        .to_dynamic_image()?;
    let developed =
        crate::imgload::orient_preview(developed, crate::imgload::exif_orientation(path));
    let image = developed.to_rgba16();
    let scale = max_dim as f32 / image.width().max(image.height()) as f32;
    if scale >= 1.0 {
        Some(image)
    } else {
        Some(image::imageops::resize(
            &image,
            (image.width() as f32 * scale).round().max(1.0) as u32,
            (image.height() as f32 * scale).round().max(1.0) as u32,
            image::imageops::FilterType::Triangle,
        ))
    }
}

pub(crate) fn apply_render_quantiles(
    image: &mut Rgba16Image,
    source: &[f32; RENDER_FEATURES],
    target: &[f32; RENDER_FEATURES],
) {
    for pixel in image.pixels_mut() {
        for channel in 0..3 {
            let value = f32::from(pixel[channel]) / 65535.0;
            let offset = channel * RENDER_QUANTILES.len();
            let upper = source[offset..offset + RENDER_QUANTILES.len()]
                .iter()
                .position(|&x| x >= value)
                .unwrap_or(RENDER_QUANTILES.len() - 1)
                .max(1);
            let (x0, x1) = (source[offset + upper - 1], source[offset + upper]);
            let (y0, y1) = (target[offset + upper - 1], target[offset + upper]);
            let t = if x1 > x0 {
                (value - x0) / (x1 - x0)
            } else {
                0.0
            };
            pixel[channel] = ((y0 + (y1 - y0) * t).clamp(0.0, 1.0) * 65535.0) as u16;
        }
    }
}

pub(crate) fn load_render_target(path: &Path, max_dim: u32) -> Option<RgbaImage> {
    let sibling = path
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| {
            let stem = name.strip_suffix(".RAW-02.ORIGINAL.dng")?;
            let parent = path.parent()?;
            [
                format!("{stem}.RAW-01.jpg"),
                format!("{stem}.RAW-01.COVER.jpg"),
                format!("{stem}.RAW-01.MP.jpg"),
            ]
            .into_iter()
            .map(|candidate| parent.join(candidate))
            .find(|candidate| candidate.exists())
        });
    let external = path
        .parent()
        .and_then(Path::parent)
        .map(|root| root.join("tiff16_c_srgb"))
        .and_then(|folder| {
            path.file_stem()
                .map(|name| folder.join(name).with_extension("jpg"))
        });
    sibling
        .or(external)
        .filter(|path| path.exists())
        .and_then(|path| crate::imgload::load_rgba(&path, max_dim))
        .or_else(|| crate::imgload::load_preview_rgba(path, max_dim))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

        #[test]
        fn embedded_raw_render_model_is_valid_and_monotone() {
            let model = RawRenderModel::embedded().expect("embedded RAW render model");
            let input: [f32; RENDER_FEATURES] =
                std::array::from_fn(|index| RENDER_QUANTILES[index % RENDER_QUANTILES.len()]);
            let output = model.predict(&input);
            assert!(output.iter().all(|value| value.is_finite()));
            for channel in 0..3 {
                assert!(output
                    [channel * RENDER_QUANTILES.len()..(channel + 1) * RENDER_QUANTILES.len()]
                    .windows(2)
                    .all(|pair| pair[0] <= pair[1]));
            }
        }

        #[test]
        fn u16_develop_matches_8bit_develop() {
            let path =
                Path::new("/Users/jakubkolcar/Downloads/PXL_20260806_114652240.RAW-02.ORIGINAL.dng");
            if !path.exists() {
                return;
            }
            let dev8 = develop_raw(path, 512).unwrap();
            let dev16 = develop_raw_u16(path, 512).unwrap();
            assert_eq!(dev16.dimensions(), dev8.dimensions());
            // u16 output is the same srgb-encoded value at 16-bit resolution.
            let mut checked = 0;
            for (pixel, px16) in dev8.pixels().zip(dev16.pixels()) {
                for channel in 0..3 {
                    let expected = u32::from(pixel[channel]) * 257;
                    let actual = u32::from(px16[channel]);
                    assert!(
                        actual.abs_diff(expected) <= 257,
                        "u16 develop diverges at {channel}: {actual} vs {expected}"
                    );
                }
                assert_eq!(px16[3], 65535);
                checked += 1;
            }
            assert!(checked > 1000);
        }

        #[test]
        fn universal_raw_development_matches_camera_preview() {
            let Ok(input) = std::env::var("RAW_CAMERA_PROBE") else {
                return;
            };
            let input = std::path::Path::new(&input);
            let paths: Vec<PathBuf> = if input.is_dir() {
                std::fs::read_dir(input)
                    .unwrap()
                    .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                    .filter(|path| crate::imgload::is_raw(path))
                    // Paired camera RAWs only: for other DNGs `load_render_target`
                    // falls back to the low-res embedded preview, and comparing
                    // development against that is too noisy to gate on.
                    .filter(|path| {
                        path.file_name()
                            .and_then(|name| name.to_str())
                            .is_some_and(|name| name.ends_with(".RAW-02.ORIGINAL.dng"))
                    })
                    .collect()
            } else {
                vec![input.to_owned()]
            };
            let mut custom_errors = Vec::new();
            for path in paths {
                let developed = develop_raw(&path, 512);
                let Some(developed) = developed else {
                    continue;
                };
                let Some(camera) = load_render_target(&path, 512) else {
                    continue;
                };
                let error = |candidate: &image::RgbaImage| {
                    let candidate = image::imageops::resize(
                        candidate,
                        camera.width(),
                        camera.height(),
                        image::imageops::FilterType::Triangle,
                    );
                    candidate
                        .pixels()
                        .zip(camera.pixels())
                        .flat_map(|(a, b)| (0..3).map(move |c| u8::abs_diff(a[c], b[c]) as f32 / 255.0))
                        .sum::<f32>()
                        / (camera.width() * camera.height() * 3) as f32
                };
            custom_errors.push(error(&developed));
            }
            assert!(!custom_errors.is_empty(), "no usable camera RAWs");
            let mean = |values: &[f32]| values.iter().sum::<f32>() / values.len() as f32;
            let custom_error = mean(&custom_errors);
            let worst = custom_errors.iter().copied().fold(0.0, f32::max);
            eprintln!(
                "camera RAWs: universal pipeline MAE {custom_error:.4}, worst {worst:.4}, n={}",
                custom_errors.len(),
            );
            assert!(custom_error < 0.08, "mean camera-render error exceeds 8%");
            assert!(worst < 0.23, "at least one camera RAW exceeds 23% error");
        }
}
