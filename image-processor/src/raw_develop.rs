//! Universal RAW development: rawler sensor decode, one exposure gain, and one
//! hue-preserving tone curve. No trained look is applied at runtime. The
//! embedded render model is kept only for the offline fitting tools in
//! `raw_fit` (CLI-only: `--fit-raw-scurve` / `--train-raw-render-model`).

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
    #[cfg(test)]
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

/// Universal develop: one exposure gain and one hue-preserving tone curve for
/// all channels. No trained look is involved, so unseen cameras get the same
/// well-behaved render and there is no per-channel color cast.
/// Chroma-only denoise, applied after the metering. Lifting shadows multiplies
/// sensor chroma noise; cameras smooth it, so we do too. Luma is untouched, and
/// the strength grows as the pixel gets darker (where the lift, and the noise,
/// is largest).
const CHROMA_DENOISE: f32 = 0.85;

fn chroma_denoise_u16(image: &mut Rgba16Image) {
    let (width, height) = (image.width() as usize, image.height() as usize);
    if width < 4 || height < 4 {
        return;
    }
    let pixels: Vec<[f32; 3]> = image
        .pixels()
        .map(|p| std::array::from_fn(|k| f32::from(p[k]) / 65535.0))
        .collect();
    let luma: Vec<f32> = pixels.iter().map(|&c| luma_of(c)).collect();
    let u: Vec<f32> = pixels.iter().zip(&luma).map(|(c, &y)| c[2] - y).collect();
    let v: Vec<f32> = pixels.iter().zip(&luma).map(|(c, &y)| c[0] - y).collect();
    let radius = (width.max(height) / 256).max(1);
    let (ub, vb) = (box_blur(&u, width, height, radius), box_blur(&v, width, height, radius));
    for (index, pixel) in image.pixels_mut().enumerate() {
        let y = luma[index];
        let weight = (CHROMA_DENOISE * (1.0 - y).max(0.0).powi(2)).clamp(0.0, 1.0);
        let u = u[index] + (ub[index] - u[index]) * weight;
        let v = v[index] + (vb[index] - v[index]) * weight;
        let (r, b) = (y + v, y + u);
        let g = (y - 0.2126 * r - 0.0722 * b) / 0.7152;
        pixel[0] = (r.clamp(0.0, 1.0) * 65535.0) as u16;
        pixel[1] = (g.clamp(0.0, 1.0) * 65535.0) as u16;
        pixel[2] = (b.clamp(0.0, 1.0) * 65535.0) as u16;
    }
}

fn develop_generic_u16(path: &Path, max_dim: u32) -> Option<Rgba16Image> {
    let mut image = decode_raw_u16(path, max_dim)?;
    let raw = raw_features(&image);
    tone_u16(&mut image);
    meter_u16(&mut image, &raw);
    chroma_denoise_u16(&mut image);
    Some(image)
}

/// The whole develop look is these five numbers. Kept in a struct so the bench
/// below can sweep them against camera JPEGs.
#[derive(Clone, Copy)]
pub(crate) struct ToneParams {
    pub black: f32,
    pub white: f32,
    pub contrast: f32,
    pub gamma: f32,
    pub knee: f32,
    /// 0 = tone on each pixel, 1 = tone on the local average and keep detail.
    pub local: f32,
    /// Local base radius divisor: radius = max(w, h) / radius_div.
    pub radius_div: u32,
}

pub(crate) const TONE: ToneParams = ToneParams {
    black: 0.005,
    white: 0.97,
    contrast: 0.2,
    gamma: 0.60,
    knee: 0.1,
    local: 0.4,
    radius_div: 4,
};

fn luma_of(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// Soft shoulder: identity below `knee`, then an exponential roll-off that
/// approaches 1.0 but never reaches it, so highlights do not go flat white.
fn shoulder(y: f32, knee: f32) -> f32 {
    if y <= knee {
        y
    } else {
        knee + (1.0 - knee) * (1.0 - (-(y - knee) / (1.0 - knee)).exp())
    }
}

/// S-curve on [0, 1]: a blend of identity and smoothstep. Monotone, with fixed
/// endpoints, and `contrast` sets how much midtone contrast is added.
fn s_curve(x: f32, contrast: f32) -> f32 {
    let smooth = x * x * (3.0 - 2.0 * x);
    x + contrast * (smooth - x)
}

/// Black and white points for the given parameters, from pre-sorted luminance.
fn tone_levels(luma_sorted: &[f32], p: &ToneParams) -> (f32, f32) {
    let at = |q: f32| luma_sorted[((luma_sorted.len() - 1) as f32 * q) as usize];
    let low = at(p.black);
    (low, at(p.white).max(low + 1e-4))
}

/// Tone for one pixel, expressed on its brightest channel `m`.
fn tone_map(m: f32, p: &ToneParams) -> f32 {
    s_curve(shoulder(m, p.knee), p.contrast).powf(p.gamma)
}

/// Blur radius for the local tone base, relative to the image.
fn local_radius(width: usize, height: usize, p: &ToneParams) -> usize {
    (width.max(height) / p.radius_div.max(1) as usize).max(2)
}

fn tone_u16_with(image: &mut Rgba16Image, p: &ToneParams) {
    let (width, height) = (image.width() as usize, image.height() as usize);
    let encoded: Vec<[f32; 3]> = image
        .pixels()
        .map(|pixel| std::array::from_fn(|c| f32::from(pixel[c]) / 65535.0))
        .collect();
    let mut luma: Vec<f32> = encoded.iter().map(|&c| luma_of(c)).collect();
    luma.sort_unstable_by(f32::total_cmp);
    let (low, high) = tone_levels(&luma, p);
    let inv = 1.0 / (high - low);
    let normalized: Vec<[f32; 3]> = encoded.iter().map(|c| c.map(|v| ((v - low) * inv).max(0.0))).collect();
    let m: Vec<f32> = normalized.iter().map(|c| c[0].max(c[1]).max(c[2])).collect();
    let base = if p.local > 0.0 {
        box_blur(&m, width, height, local_radius(width, height, p))
    } else {
        Vec::new()
    };
    for (index, (pixel, c)) in image.pixels_mut().zip(&normalized).enumerate() {
        let mm = m[index];
        let global = if mm > 1e-5 { tone_map(mm, p) / mm } else { 1.0 };
        let scale = if p.local > 0.0 {
            let b = base[index].max(1e-4);
            let local = tone_map(b, p) / b;
            global * (1.0 - p.local) + local * p.local
        } else {
            global
        };
        for k in 0..3 {
            pixel[k] = ((c[k] * scale).clamp(0.0, 1.0) * 65535.0) as u16;
        }
    }
}

fn tone_u16(image: &mut Rgba16Image) {
    tone_u16_with(image, &TONE)
}

// --- Learned metering ---------------------------------------------------------
//
// One exposure gain and one contrast offset per frame, predicted from the
// frame's own statistics: nine from the toned frame plus five from the raw
// decode at absolute level. The raw statistics are what carry the scene's
// absolute brightness, which the levels otherwise normalize away. No reference
// image, no camera branch, and no dataset at runtime: the coefficients below are
// fitted offline by the `metering_probe` test over RAW/JPEG pairs and baked in.

fn frame_luma(image: &Rgba16Image) -> Vec<f32> {
    image
        .pixels()
        .map(|p| (0.2126 * f32::from(p[0]) + 0.7152 * f32::from(p[1]) + 0.0722 * f32::from(p[2])) / 257.0)
        .collect()
}

/// Five features from the raw decode, before our tone, at absolute level.
fn raw_features(image: &Rgba16Image) -> [f32; 5] {
    let luma = frame_luma(image);
    let mut sorted = luma.clone();
    sorted.sort_unstable_by(f32::total_cmp);
    let at = |q: f32| sorted[((sorted.len() - 1) as f32 * q) as usize] / 255.0;
    let mean = luma.iter().sum::<f32>() / luma.len() as f32;
    let std = (luma.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / luma.len() as f32).sqrt();
    [at(0.05), at(0.50), at(0.95), std / 255.0, mean / 255.0]
}

/// Fifteen features: intercept, nine from the toned frame, five raw.
/// Keep in sync with `metering_probe`.
fn meter_features(luma_sorted: &[f32], std: f32, skew: f32, raw: &[f32; 5]) -> [f32; 15] {
    let at = |q: f32| luma_sorted[((luma_sorted.len() - 1) as f32 * q) as usize] / 255.0;
    [
        1.0,
        at(0.02),
        at(0.10),
        at(0.25),
        at(0.50),
        at(0.75),
        at(0.90),
        at(0.98),
        std / 255.0,
        skew,
        raw[0],
        raw[1],
        raw[2],
        raw[3],
        raw[4],
    ]
}

/// Baked coefficients: gain (dimensionless) and offset (0..255 luma units).
const METER_GAIN: [f32; 15] = [0.927340, 0.107458, 0.256339, 0.054291, -0.027627, 0.197642, -0.407719, -0.127172, -0.108627, -0.019321, -1.181472, -0.131208, 0.539130, 1.032186, 0.108172];
/// Per-frame midtone curve exponent (below 1 lifts midtones, above 1 adds contrast).
const METER_GAMMA: [f32; 15] = [1.369316, 0.597708, -0.219898, -0.590255, 0.389556, 0.367973, 0.677099, -0.906502, 0.155715, 0.039787, 0.808022, 0.112798, -0.765814, -0.141391, 0.335810];
const METER_OFFSET: [f32; 15] = [-22.968147, -34.602035, -70.784434, -23.354040, 28.857493, 13.171703, 64.228898, -22.402481, 12.473469, 12.846639, 146.482698, 39.159393, -59.684663, -65.754446, 47.243694];

/// Clamps keep a weird frame from being wrecked by an out-of-distribution fit.
const METER_GAIN_RANGE: (f32, f32) = (0.6, 1.8);
const METER_OFFSET_RANGE: (f32, f32) = (-60.0, 60.0);
const METER_GAMMA_RANGE: (f32, f32) = (0.70, 1.40);

fn dotn(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The mapping a frame gets: a gamma, a gain and an offset, finished with the
/// same soft shoulder the tone uses. The shoulder means a brightening that would
/// otherwise clip the sky rolls off instead, so midtones can be lifted without
/// buying it with blown highlights.
const METER_KNEE: f32 = 0.95;

fn meter_apply(y: f32, gain: f32, gamma: f32, offset: f32) -> f32 {
    let lifted = gain * (y / 255.0).max(0.0).powf(gamma) + offset / 255.0;
    255.0 * shoulder(lifted.max(0.0), METER_KNEE)
}

/// The metering prediction (gain, offset, gamma) for a frame. Exposed for tests.
pub(crate) fn meter_predict(image: &Rgba16Image, raw: &[f32; 5]) -> (f32, f32, f32) {
    let luma = frame_luma(image);
    let mut sorted = luma.clone();
    sorted.sort_unstable_by(f32::total_cmp);
    let mean = luma.iter().sum::<f32>() / luma.len() as f32;
    let std = (luma.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / luma.len() as f32).sqrt();
    let skew = luma.iter().map(|x| ((x - mean) / std.max(1e-6)).powi(3)).sum::<f32>() / luma.len() as f32;
    let features = meter_features(&sorted, std, skew, raw);
    let gain = dotn(&features, &METER_GAIN).clamp(METER_GAIN_RANGE.0, METER_GAIN_RANGE.1);
    let offset = dotn(&features, &METER_OFFSET).clamp(METER_OFFSET_RANGE.0, METER_OFFSET_RANGE.1);
    let gamma = dotn(&features, &METER_GAMMA).clamp(METER_GAMMA_RANGE.0, METER_GAMMA_RANGE.1);
    (gain, offset, gamma)
}

fn meter_u16(image: &mut Rgba16Image, raw: &[f32; 5]) {
    let (gain, offset, gamma) = meter_predict(image, raw);
    let luma = frame_luma(image);
    for (pixel, &y) in image.pixels_mut().zip(&luma) {
        if y < 1e-3 {
            continue;
        }
        let scale = meter_apply(y, gain, gamma, offset).max(0.0) / y;
        for k in 0..3 {
            pixel[k] = (f32::from(pixel[k]) * scale).clamp(0.0, 65535.0) as u16;
        }
    }
}

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
mod tests;
#[cfg(test)]
mod dataset_tests;
#[cfg(test)]
mod reference_bench;
#[cfg(test)]
mod tune;
#[cfg(test)]
mod decompose;
#[cfg(test)]
mod profile_probe;
#[cfg(test)]
mod metering_probe;
#[cfg(test)]
mod frames_probe;
