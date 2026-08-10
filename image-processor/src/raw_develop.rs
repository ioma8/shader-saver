//! Basic RAW development on top of rawloader, plus the phone S-curve.
//!
//! The previous developer learned a full neural ISP and a per-camera LUT
//! database from DNG/JPEG pairs.  This module takes the opposite route: a
//! minimal, deterministic development that leaves the image linear — crop,
//! black level, as-shot white balance, demosaic, camera-to-sRGB matrix — and
//! then one global S-curve fitted from the phone's own DNG/JPEG pairs.  The
//! curve maps exposure-normalized linear luminance to display luminance, so
//! each DNG lands on the same tone placement as the phone's developed JPEG.

use image::RgbaImage;
use rayon::prelude::*;
use rawloader::{Orientation, RawImage, RawImageData};
use std::path::{Path, PathBuf};

/// Linear RGB render from rawloader: white-balanced, demosaiced, and
/// transformed into linear sRGB.  Values are float; highlights may exceed 1.0
/// until the tone curve is applied at development time.  Orientation is
/// carried through unchanged (the sensor plane, not the display orientation);
/// `develop_raw` applies it last so the result matches the pre-oriented JPEG.
pub struct LinearImage {
    pub width: u32,
    pub height: u32,
    /// `width * height * 3` floats, RGB interleaved.
    pub data: Vec<f32>,
    /// EXIF orientation of the source file (rawloader's `Orientation`).
    pub orientation: Orientation,
}

impl LinearImage {
    pub fn luminance_percentile(&self, percentile: f64) -> f32 {
        luma_percentile(&self.data, percentile)
    }
}

/// Decode `path` with rawloader and develop it to linear sRGB, optionally
/// downsampled to fit within `max_dim` (0 = full resolution).  When the target
/// is smaller than the sensor, the CFA is first decimated 2x2 (a Bayer 2x2
/// block holds all three colors, so the mosaic pattern survives and the
/// demosaic becomes a cheap box), then box-downsampled to `max_dim` if still
/// too large.
pub fn develop_linear(path: &Path, max_dim: u32) -> Option<LinearImage> {
    let raw = rawloader::decode_file(path).ok()?;
    let mut image = develop_from_raw(&raw, max_dim, path)?;
    if max_dim > 0 && image.width.max(image.height) > max_dim {
        let scale = max_dim as f32 / image.width.max(image.height) as f32;
        image = box_downsample(&image, scale);
    }
    Some(image)
}

fn develop_from_raw(raw: &RawImage, max_dim: u32, path: &Path) -> Option<LinearImage> {
    let (left, right, top, bottom) = (raw.crops[3], raw.crops[1], raw.crops[0], raw.crops[2]);
    let width = raw.width.checked_sub(left + right)?;
    let height = raw.height.checked_sub(top + bottom)?;
    if width == 0 || height == 0 {
        return None;
    }

    let black = raw.blacklevels;
    let white = raw.whitelevels;
    let mut wb = raw.wb_coeffs;
    // rawloader reads DNG AsShotNeutral under the wrong tag number (0xC628 is
    // AsShotWhiteXY), so DNGs always surface NaN here.  Read the real tag
    // (0xC627) from the file; Pixel DNGs carry AsShotNeutral (1,1,1) — the
    // raw is pre-white-balanced — so identity is the fallback when the tag is
    // genuinely absent.
    if wb[0].is_nan() || wb[1].is_nan() || wb[2].is_nan() {
        wb = dng_as_shot_neutral(path).unwrap_or([1.0, 1.0, 1.0, 1.0]);
    }
    let wb = wb.map(|v| if v.is_finite() && v > 0.0 { v } else { 1.0 });
    let is_integer = matches!(raw.data, RawImageData::Integer(_));

    // Normalize one sensor sample: subtract black, scale to the white level,
    // apply the as-shot white-balance multiplier.
    let norm = |value: f32, channel: usize| -> f32 {
        let range = f32::from(white[channel]) - f32::from(black[channel]);
        let value = if range > 0.0 {
            value - f32::from(black[channel])
        } else {
            value
        };
        let value = if is_integer {
            (value / range.max(1.0)).clamp(0.0, 1.0)
        } else {
            value / range.max(1.0)
        };
        value * wb[channel]
    };

    let mut data: Vec<f32>;
    if raw.cpp == 1 {
        // CFA plane: crop, normalize, then demosaic.  When a smaller render is
        // wanted and the CFA is Bayer (2x2), decimate first: each 2x2 block
        // holds all three colors, so averaging per color yields a complete RGB
        // pixel at half resolution without ever running the full demosaic.
        let cfa = raw.cropped_cfa();
        let fast = max_dim > 0
            && width.max(height) > max_dim as usize
            && cfa.width == 2
            && cfa.height == 2;
        if fast {
            let ow = width.div_ceil(2);
            let oh = height.div_ceil(2);
            data = Vec::with_capacity(ow * oh * 3);
            let sample = |row: usize, col: usize| match &raw.data {
                RawImageData::Integer(values) => f32::from(values[row * raw.width + col]),
                RawImageData::Float(values) => values[row * raw.width + col],
            };
            for r in 0..oh {
                for c in 0..ow {
                    let mut sums = [0.0f32; 3];
                    let mut counts = [0u32; 3];
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let (yy, xx) = (top + r * 2 + dy, left + c * 2 + dx);
                            if yy >= top + height || xx >= left + width {
                                continue;
                            }
                            let channel = cfa.color_at(yy - top, xx - left);
                            sums[channel] += norm(sample(yy, xx), channel);
                            counts[channel] += 1;
                        }
                    }
                    for channel in 0..3 {
                        data.push(sums[channel] / counts[channel].max(1) as f32);
                    }
                }
            }
            let (width, height) = (ow as u32, oh as u32);
            apply_camera_matrix(&mut data, &camera_to_srgb_matrix(raw));
            return Some(LinearImage {
                width,
                height,
                data,
                orientation: raw.orientation,
            });
        }
        let mut plane = Vec::with_capacity(width * height);
        match &raw.data {
            RawImageData::Integer(values) => {
                for row in top..top + height {
                    for col in left..left + width {
                        let value = f32::from(values[row * raw.width + col]);
                        let channel = raw.cropped_cfa().color_at(row - top, col - left);
                        plane.push(norm(value, channel));
                    }
                }
            }
            RawImageData::Float(values) => {
                for row in top..top + height {
                    for col in left..left + width {
                        let value = values[row * raw.width + col];
                        let channel = raw.cropped_cfa().color_at(row - top, col - left);
                        plane.push(norm(value, channel));
                    }
                }
            }
        }
        data = demosaic(&plane, width, height, &raw.cropped_cfa());
    } else {
        // cpp >= 3: already per-pixel RGB(E) data, no demosaic needed.
        data = Vec::with_capacity(width * height * 3);
        let channels = raw.cpp.min(3);
        match &raw.data {
            RawImageData::Integer(values) => {
                for row in top..top + height {
                    for col in left..left + width {
                        let base = (row * raw.width + col) * raw.cpp;
                        for channel in 0..channels {
                            data.push(norm(f32::from(values[base + channel]), channel));
                        }
                    }
                }
            }
            RawImageData::Float(values) => {
                for row in top..top + height {
                    for col in left..left + width {
                        let base = (row * raw.width + col) * raw.cpp;
                        for channel in 0..channels {
                            data.push(norm(values[base + channel], channel));
                        }
                    }
                }
            }
        }
    }

    let matrix = camera_to_srgb_matrix(raw);
    apply_camera_matrix(&mut data, &matrix);
    Some(LinearImage {
        width: width as u32,
        height: height as u32,
        data,
        orientation: raw.orientation,
    })
}

/// The DNG's as-shot white balance as WB multipliers (inverse of the
/// AsShotNeutral values).  kamadak-exif re-reads the file, which is cheap next
/// to the rawloader decode itself.
fn dng_as_shot_neutral(path: &Path) -> Option<[f32; 4]> {
    let file = std::fs::File::open(path).ok()?;
    let exif = exif::Reader::new()
        .read_from_container(&mut std::io::BufReader::new(file))
        .ok()?;
    let field = exif.fields().find(|field| field.tag.number() == 0xC627)?;
    let exif::Value::Rational(values) = &field.value else {
        return None;
    };
    if values.len() < 3 {
        return None;
    }
    let mut wb = [1.0f32; 4];
    for (channel, value) in wb.iter_mut().zip(values).take(3) {
        let ratio = if value.denom != 0 {
            value.num as f32 / value.denom as f32
        } else {
            0.0
        };
        *channel = if ratio > 0.0 { 1.0 / ratio } else { 1.0 };
    }
    Some(wb)
}

// --- Demosaic ---------------------------------------------------------------

/// Combined camera-to-linear-sRGB matrix: rawloader's camera->XYZ (the raw
/// pseudoinverse, not the normalized variant, so as-shot neutral stays D65
/// neutral through the pipeline) followed by the standard sRGB-D65
/// XYZ->linear-RGB matrix.
fn camera_to_srgb_matrix(raw: &RawImage) -> [[f32; 3]; 3] {
    const XYZ_TO_SRGB: [[f32; 3]; 3] = [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ];
    let cam_to_xyz = raw.cam_to_xyz(); // rows=XYZ, cols=RGB(E)
    let mut out = [[0.0f32; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            for k in 0..3 {
                out[row][col] += XYZ_TO_SRGB[row][k] * cam_to_xyz[k][col];
            }
        }
    }
    out
}

fn apply_camera_matrix(data: &mut [f32], matrix: &[[f32; 3]; 3]) {
    for pixel in data.chunks_exact_mut(3) {
        let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
        let out = [
            matrix[0][0] * r + matrix[0][1] * g + matrix[0][2] * b,
            matrix[1][0] * r + matrix[1][1] * g + matrix[1][2] * b,
            matrix[2][0] * r + matrix[2][1] * g + matrix[2][2] * b,
        ];
        pixel[0] = out[0].max(0.0);
        pixel[1] = out[1].max(0.0);
        pixel[2] = out[2].max(0.0);
    }
}

fn demosaic(plane: &[f32], width: usize, height: usize, cfa: &rawloader::CFA) -> Vec<f32> {
    if !cfa.is_valid() {
        // Monochrome sensor: every pixel is a gray sample.
        let mut out = Vec::with_capacity(plane.len() * 3);
        for &value in plane {
            out.extend_from_slice(&[value, value, value]);
        }
        return out;
    }
    if cfa.width == 2 && cfa.height == 2 {
        malvar(plane, width, height, cfa)
    } else {
        // X-Trans and other exotic arrays: average same-color neighbours.
        bilinear(plane, width, height, cfa)
    }
}

/// Malvar-He-Cutler 5x5 linear demosaic (Malvar, He & Cutler, ICASSP 2004).
/// Weight tables (already divided by the reference's 8) transcribed from the
/// colour-demosaicing implementation; every kernel sums to 1 so flat areas
/// stay flat.
fn malvar(plane: &[f32], width: usize, height: usize, cfa: &rawloader::CFA) -> Vec<f32> {
    const GR_GB: [f32; 25] = [
        0.0, 0.0, -0.125, 0.0, 0.0, //
        0.0, 0.0, 0.25, 0.0, 0.0, //
        -0.125, 0.25, 0.5, 0.25, -0.125, //
        0.0, 0.0, 0.25, 0.0, 0.0, //
        0.0, 0.0, -0.125, 0.0, 0.0,
    ];
    const RG_RB_BG_BR: [f32; 25] = [
        0.0, 0.0, 0.0625, 0.0, 0.0, //
        0.0, -0.125, 0.0, -0.125, 0.0, //
        -0.125, 0.5, 0.625, 0.5, -0.125, //
        0.0, -0.125, 0.0, -0.125, 0.0, //
        0.0, 0.0, 0.0625, 0.0, 0.0,
    ];
    const RG_BR_BG_RB: [f32; 25] = [
        0.0, 0.0, -0.125, 0.0, 0.0, //
        0.0, -0.125, 0.5, -0.125, 0.0, //
        0.0625, 0.0, 0.625, 0.0, 0.0625, //
        0.0, -0.125, 0.5, -0.125, 0.0, //
        0.0, 0.0, -0.125, 0.0, 0.0,
    ];
    const RB_BB_BR_RR: [f32; 25] = [
        0.0, 0.0, -0.1875, 0.0, 0.0, //
        0.0, 0.25, 0.0, 0.25, 0.0, //
        -0.1875, 0.0, 0.75, 0.0, -0.1875, //
        0.0, 0.25, 0.0, 0.25, 0.0, //
        0.0, 0.0, -0.1875, 0.0, 0.0,
    ];

    let (gr_gb, rg_rb, rg_br, rb_bb) = (
        convolve5(plane, width, height, &GR_GB),
        convolve5(plane, width, height, &RG_RB_BG_BR),
        convolve5(plane, width, height, &RG_BR_BG_RB),
        convolve5(plane, width, height, &RB_BB_BR_RR),
    );

    // Which rows/columns carry red and blue samples (the other color planes
    // are interpolated depending on the pixel's row/column membership).
    let mut row_red = vec![false; height];
    let mut row_blue = vec![false; height];
    let mut col_red = vec![false; width];
    let mut col_blue = vec![false; width];
    for y in 0..height {
        for x in 0..width {
            match cfa.color_at(y, x) {
                0 => {
                    row_red[y] = true;
                    col_red[x] = true;
                }
                2 => {
                    row_blue[y] = true;
                    col_blue[x] = true;
                }
                _ => {}
            }
        }
    }

    let mut out = vec![0.0f32; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let j = i * 3;
            let (r, g, b) = match cfa.color_at(y, x) {
                0 => (plane[i], gr_gb[i], rb_bb[i]),
                2 => (rb_bb[i], gr_gb[i], plane[i]),
                _ => {
                    let (r, b) = if row_red[y] {
                        (rg_rb[i], rg_br[i])
                    } else {
                        (rg_br[i], rg_rb[i])
                    };
                    (r, plane[i], b)
                }
            };
            out[j] = r;
            out[j + 1] = g;
            out[j + 2] = b;
        }
    }
    out
}

/// Fallback demosaic for non-Bayer arrays: average every sample of the same
/// color within a 5x5 window.
fn bilinear(plane: &[f32], width: usize, height: usize, cfa: &rawloader::CFA) -> Vec<f32> {
    let mut out = vec![0.0f32; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let mut sums = [0.0f32; 3];
            let mut counts = [0u32; 3];
            for dy in -2isize..=2 {
                for dx in -2isize..=2 {
                    let yy = (y as isize + dy).clamp(0, height as isize - 1) as usize;
                    let xx = (x as isize + dx).clamp(0, width as isize - 1) as usize;
                    let channel = cfa.color_at(yy, xx);
                    sums[channel] += plane[yy * width + xx];
                    counts[channel] += 1;
                }
            }
            let j = (y * width + x) * 3;
            for channel in 0..3 {
                out[j + channel] = sums[channel] / counts[channel] as f32;
            }
        }
    }
    out
}

fn convolve5(src: &[f32], width: usize, height: usize, kernel: &[f32; 25]) -> Vec<f32> {
    let mut out = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0f32;
            for ky in 0..5 {
                let sy = reflect(y as isize + ky as isize - 2, height);
                for kx in 0..5 {
                    let sx = reflect(x as isize + kx as isize - 2, width);
                    acc += kernel[ky * 5 + kx] * src[sy * width + sx];
                }
            }
            out[y * width + x] = acc;
        }
    }
    out
}

/// Mirror an index into [0, n) (scipy's "reflect" border mode), matching the
/// reference implementation's border handling.
fn reflect(index: isize, n: usize) -> usize {
    let n = n as isize;
    let mut index = index;
    while index < 0 || index >= n {
        if index < 0 {
            index = -index;
        }
        if index >= n {
            index = 2 * n - 2 - index;
        }
    }
    index as usize
}

// --- Downsampling -----------------------------------------------------------

fn box_downsample(src: &LinearImage, scale: f32) -> LinearImage {
    let (w, h) = (src.width as usize, src.height as usize);
    let nw = ((w as f32 * scale).round() as usize).max(1);
    let nh = ((h as f32 * scale).round() as usize).max(1);
    let mut data = vec![0.0f32; nw * nh * 3];
    for oy in 0..nh {
        let y0 = oy * h / nh;
        let y1 = ((oy + 1) * h / nh).max(y0 + 1);
        for ox in 0..nw {
            let x0 = ox * w / nw;
            let x1 = ((ox + 1) * w / nw).max(x0 + 1);
            let mut sums = [0.0f64; 3];
            let mut count = 0u64;
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let pixel = &src.data[(sy * w + sx) * 3..][..3];
                    sums[0] += f64::from(pixel[0]);
                    sums[1] += f64::from(pixel[1]);
                    sums[2] += f64::from(pixel[2]);
                    count += 1;
                }
            }
            let o = (oy * nw + ox) * 3;
            data[o] = (sums[0] / count as f64) as f32;
            data[o + 1] = (sums[1] / count as f64) as f32;
            data[o + 2] = (sums[2] / count as f64) as f32;
        }
    }
    LinearImage {
        width: nw as u32,
        height: nh as u32,
        data,
        orientation: src.orientation,
    }
}

// --- Luminance statistics ---------------------------------------------------

/// Upper bound of the linear-luminance histogram range.  Camera-matrix output
/// for a well-exposed frame sits well below this; speculars and clipped areas
/// land in the top bin.
const LUMA_MAX: f32 = 8.0;
const LUMA_BINS: usize = 4096;

/// Histogram of Rec.709 luminance for exposure anchoring and curve fitting.
/// Bucketing avoids sorting millions of pixels and is deterministic.  The bin
/// ceiling is set per histogram (`LUMA_MAX` for camera-matrix output, 1.0 for
/// linearized display values).
pub struct LumaHistogram {
    bins: [u64; LUMA_BINS],
    total: u64,
    max: f32,
}

impl LumaHistogram {
    pub fn from_rgb(data: &[f32]) -> Self {
        Self::from_rgb_max(data, LUMA_MAX)
    }

    pub fn from_rgb_max(data: &[f32], max: f32) -> Self {
        let mut bins = [0u64; LUMA_BINS];
        let mut total = 0u64;
        for pixel in data.chunks_exact(3) {
            let luma = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];
            let bin = ((luma / max * LUMA_BINS as f32) as usize).min(LUMA_BINS - 1);
            bins[bin] += 1;
            total += 1;
        }
        Self { bins, total, max }
    }

    /// The luminance value below which `percentile` (0..1) of samples fall.
    /// Returns the center of the containing bin.
    pub fn percentile(&self, percentile: f64) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        let target = (percentile * self.total as f64).max(1.0);
        let mut cumulative = 0u64;
        for (bin, &count) in self.bins.iter().enumerate() {
            cumulative += count;
            if cumulative >= target as u64 {
                return (bin as f32 + 0.5) * (self.max / LUMA_BINS as f32);
            }
        }
        self.max
    }
}

fn luma_percentile(data: &[f32], percentile: f64) -> f32 {
    LumaHistogram::from_rgb(data).percentile(percentile)
}

// --- sRGB encoding ----------------------------------------------------------

/// Encode one linear sRGB component to the sRGB transfer function.
pub fn srgb_encode(value: f32) -> f32 {
    if value <= 0.0031308 {
        12.92 * value
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

/// Convert a linear RGB image to an 8-bit sRGB RGBA image, clipping at 1.0.
pub fn to_rgba(image: &LinearImage) -> RgbaImage {
    let mut out = RgbaImage::new(image.width, image.height);
    for (pixel, value) in out.pixels_mut().zip(image.data.chunks_exact(3)) {
        for channel in 0..3 {
            pixel[channel] =
                (srgb_encode(value[channel].clamp(0.0, 1.0)) * 255.0 + 0.5).round() as u8;
        }
        pixel[3] = 255;
    }
    out
}

/// Decode one sRGB-encoded component to linear light (inverse of `srgb_encode`).
pub fn srgb_decode(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

// --- Phone S-curve ----------------------------------------------------------

/// The entire "look" of RAW development: one global monotone curve fitted from
/// DNG/JPEG pairs.  `points` maps exposure-normalized linear luminance (luma /
/// anchor-percentile luma) to linear display luminance; `apply` is monotone
/// piecewise-linear and clamps beyond the end points.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SCurve {
    /// Percentile of linear luminance used as the exposure anchor (0.5).
    pub anchor_percentile: f64,
    /// `(x, y)` pairs, `x` strictly increasing, first point at (0, 0).
    pub points: Vec<[f32; 2]>,
}

impl SCurve {
    /// Load the embedded curve fitted from the phone's DNG/JPEG pairs.
    pub fn load() -> Option<Self> {
        serde_json::from_str(include_str!("../models/raw_s_curve.json")).ok()
    }

    /// Map exposure-normalized linear luminance to linear display luminance.
    /// Monotone piecewise-linear interpolation; clamps beyond the end points.
    pub fn apply(&self, x: f32) -> f32 {
        let Some(first) = self.points.first().copied() else {
            return x;
        };
        let last = self.points[self.points.len() - 1];
        if x <= first[0] {
            return first[1];
        }
        if x >= last[0] {
            return last[1];
        }
        for pair in self.points.windows(2) {
            let (x0, y0) = (pair[0][0], pair[0][1]);
            let (x1, y1) = (pair[1][0], pair[1][1]);
            if x >= x0 && x <= x1 {
                if x1 > x0 {
                    let t = (x - x0) / (x1 - x0);
                    return y0 + (y1 - y0) * t;
                }
                return y0;
            }
        }
        last[1]
    }
}

/// Pairs of Pixel RAW captures and their developed JPEGs in `folder`
/// (`PXL_*.RAW-02.ORIGINAL.dng` + the matching `RAW-01*.jpg`).
pub fn find_pairs(folder: &Path) -> Result<Vec<(PathBuf, PathBuf)>, String> {
    let mut pairs = Vec::new();
    for entry in std::fs::read_dir(folder).map_err(|error| error.to_string())? {
        let raw = entry.map_err(|error| error.to_string())?.path();
        let Some(name) = raw.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(prefix) = name.strip_suffix(".RAW-02.ORIGINAL.dng") else {
            continue;
        };
        let jpeg = [
            format!("{prefix}.RAW-01.jpg"),
            format!("{prefix}.RAW-01.COVER.jpg"),
            format!("{prefix}.RAW-01.MP.jpg"),
        ]
        .into_iter()
        .map(|name| folder.join(name))
        .find(|path| path.exists());
        if let Some(jpeg) = jpeg {
            pairs.push((raw, jpeg));
        }
    }
    pairs.sort();
    Ok(pairs)
}

/// Fit the global phone S-curve from the DNG/JPEG pairs in `folder` and write
/// it to `output` as JSON.  Returns the number of pairs used.
///
/// Both sides are reduced to tone statistics (percentiles of linear
/// luminance), so a 512 px render is enough.  Each pair contributes one
/// `(x, y)` sample per anchor: `x = L_p / L_0.5` (exposure-normalized linear
/// luma) against `y` = the JPEG's linearized display luma at the same
/// percentile.  Pooling takes the median across pairs, which keeps the curve
/// monotone and robust to individual mis-exposures.
pub fn fit_s_curve(folder: &Path, output: &Path) -> Result<usize, String> {
    const ANCHORS: [f64; 15] = [
        0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0,
    ];
    let pairs = find_pairs(folder)?;
    // Each pair decodes independently (a full DNG decode each), so the per-
    // pair percentile samples are computed in parallel across cores.
    let pair_samples: Vec<Option<(Vec<f32>, Vec<f32>)>> = pairs
        .par_iter()
        .map(|(raw_path, jpeg_path)| {
            let linear = develop_linear(raw_path, 512)?;
            let jpeg = crate::imgload::load_rgba(jpeg_path, 512)?;
            let luma = LumaHistogram::from_rgb(&linear.data);
            let median = luma.percentile(0.5);
            if median <= 0.0 {
                return None;
            }
            // Linearize each encoded channel first so the JPEG's display luma
            // is in the same linear light as the RAW side.
            let jpeg_linear: Vec<f32> = jpeg
                .pixels()
                .flat_map(|pixel| {
                    let value = |v: u8| srgb_decode(f32::from(v) / 255.0);
                    [value(pixel[0]), value(pixel[1]), value(pixel[2])]
                })
                .collect();
            let jpeg_luma = LumaHistogram::from_rgb_max(&jpeg_linear, 1.0);
            let xs = ANCHORS.map(|anchor| luma.percentile(anchor) / median).to_vec();
            let ys = ANCHORS.map(|anchor| jpeg_luma.percentile(anchor)).to_vec();
            Some((xs, ys))
        })
        .collect();
    let usable = pair_samples.iter().filter(|sample| sample.is_some()).count();
    if usable < 20 {
        return Err(format!("only {usable} usable DNG/JPEG pairs found"));
    }
    let mut x_samples: Vec<Vec<f32>> = vec![Vec::new(); ANCHORS.len()];
    let mut y_samples: Vec<Vec<f32>> = vec![Vec::new(); ANCHORS.len()];
    for (xs, ys) in pair_samples.into_iter().flatten() {
        for (i, (&x, &y)) in xs.iter().zip(&ys).enumerate() {
            x_samples[i].push(x);
            y_samples[i].push(y);
        }
    }
    let mut points: Vec<[f32; 2]> = ANCHORS
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let median = |values: &mut Vec<f32>| -> f32 {
                values.sort_by(f32::total_cmp);
                values[values.len() / 2]
            };
            [median(&mut x_samples[i]), median(&mut y_samples[i])]
        })
        .collect();
    points.sort_by(|a, b| a[0].total_cmp(&b[0]));
    // Drop duplicate x (keep the brighter of the tied samples so the curve
    // stays a function), then pin the origin so apply(0) == 0.
    let mut deduped: Vec<[f32; 2]> = Vec::with_capacity(points.len());
    for point in points {
        if let Some(prev) = deduped.last_mut() {
            if prev[0] == point[0] {
                prev[1] = prev[1].max(point[1]);
                continue;
            }
        }
        deduped.push(point);
    }
    if deduped[0][0] > 0.0 {
        deduped.insert(0, [0.0, 0.0]);
    } else {
        deduped[0][1] = 0.0;
    }
    let curve = SCurve {
        anchor_percentile: 0.5,
        points: deduped,
    };
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    std::fs::write(output, serde_json::to_string_pretty(&curve).map_err(|e| e.to_string())?)
        .map_err(|error| error.to_string())?;
    Ok(usable)
}

// --- Development ------------------------------------------------------------

/// Fully develop a RAW file to an 8-bit sRGB RGBA image: linear development,
/// exposure-normalized S-curve, then the file's EXIF orientation (so the
/// result matches the camera's pre-oriented JPEG).
pub fn develop_raw(path: &Path, max_dim: u32, curve: &SCurve) -> Option<RgbaImage> {
    let linear = develop_linear(path, max_dim)?;
    let tone = apply_s_curve(&linear, curve);
    let oriented = apply_orientation(&tone, tone.orientation);
    Some(to_rgba(&oriented))
}

/// Apply the S-curve hue-preservingly: exposure-normalize each pixel's luma by
/// the anchor (median linear luminance), map through the curve, and scale RGB
/// by `display / luma` so the output keeps hue and lands on the curve's luma.
fn apply_s_curve(image: &LinearImage, curve: &SCurve) -> LinearImage {
    let anchor = image.luminance_percentile(curve.anchor_percentile).max(1e-3);
    let mut data = image.data.clone();
    for pixel in data.chunks_exact_mut(3) {
        let luma = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];
        let scale = if luma > 0.0 {
            curve.apply(luma / anchor) / luma
        } else {
            0.0
        };
        for channel in pixel {
            *channel = (*channel * scale).clamp(0.0, 1.0);
        }
    }
    LinearImage {
        width: image.width,
        height: image.height,
        data,
        orientation: image.orientation,
    }
}

/// Rotate/flip a linear image per its EXIF orientation, matching imagepipe's
/// `transform` op so the developed render agrees with the base open render.
/// Transpose is applied as flipping-then-transposing (rawloader's documented
/// flip-before-transpose order).
fn apply_orientation(image: &LinearImage, orientation: Orientation) -> LinearImage {
    let (w, h) = (image.width as usize, image.height as usize);
    let (dw, dh) = match orientation {
        Orientation::Normal
        | Orientation::Unknown
        | Orientation::HorizontalFlip
        | Orientation::VerticalFlip
        | Orientation::Rotate180 => (w, h),
        Orientation::Transpose
        | Orientation::Rotate90
        | Orientation::Rotate270
        | Orientation::Transverse => (h, w),
    };
    let mut data = vec![0.0f32; dw * dh * 3];
    for y in 0..dh {
        for x in 0..dw {
            let (sx, sy) = match orientation {
                Orientation::Normal | Orientation::Unknown => (x, y),
                Orientation::HorizontalFlip => (w - 1 - x, y),
                Orientation::VerticalFlip => (x, h - 1 - y),
                Orientation::Rotate180 => (w - 1 - x, h - 1 - y),
                Orientation::Transpose => (y, x),
                Orientation::Rotate90 => (y, h - 1 - x),
                Orientation::Rotate270 => (w - 1 - y, x),
                Orientation::Transverse => (w - 1 - y, h - 1 - x),
            };
            let src = (sy * w + sx) * 3;
            let dst = (y * dw + x) * 3;
            data[dst..dst + 3].copy_from_slice(&image.data[src..src + 3]);
        }
    }
    LinearImage {
        width: dw as u32,
        height: dh as u32,
        data,
        orientation,
    }
}

/// Squared, axis-normalized distance between two look profiles (used by the
/// develop-toward-JPEG verification test).
#[cfg(test)]
pub fn profile_distance(a: &crate::processor::LookProfile, b: &crate::processor::LookProfile) -> f32 {
    let mut distance = 0.0;
    for i in 0..5 {
        distance += ((a.tone[i] - b.tone[i]) / 0.15).powi(2);
    }
    for band in 0..3 {
        for axis in 0..2 {
            distance += ((a.cast[band][axis] - b.cast[band][axis]) / 0.04).powi(2);
        }
    }
    distance += ((a.chroma - b.chroma) / 0.08).powi(2);
    for i in 0..8 {
        distance += ((a.hue_chroma[i] - b.hue_chroma[i]) / 0.10).powi(2);
    }
    distance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_cfa_demosaics_to_flat_color() {
        let cfa = rawloader::CFA::new("RGGB");
        let plane = vec![0.5f32; 8 * 8];
        let out = demosaic(&plane, 8, 8, &cfa);
        for pixel in out.chunks_exact(3) {
            for value in pixel {
                assert!((value - 0.5).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn flat_cfa_with_other_pattern_demosaics_flat() {
        for pattern in ["BGGR", "GRBG", "GBRG"] {
            let cfa = rawloader::CFA::new(pattern);
            let plane = vec![0.25f32; 10 * 10];
            let out = demosaic(&plane, 10, 10, &cfa);
            for pixel in out.chunks_exact(3) {
                for value in pixel {
                    assert!((value - 0.25).abs() < 1e-4, "{pattern} not flat");
                }
            }
        }
    }

    #[test]
    fn luma_histogram_percentiles_are_monotonic() {
        let data: Vec<f32> = (0..100)
            .flat_map(|i| {
                let v = i as f32 / 100.0;
                [v, v, v]
            })
            .collect();
        let hist = LumaHistogram::from_rgb(&data);
        let p = |fraction: f64| hist.percentile(fraction);
        assert!(p(0.25) < p(0.5) && p(0.5) < p(0.75));
        assert!(p(0.5) > 0.45 && p(0.5) < 0.55);
    }

    #[test]
    fn known_dngs_decode_to_linear() {
        let cases = [
            "/Users/jakubkolcar/Downloads/PXL_20260806_095423699.RAW-02.ORIGINAL.dng",
            "/Users/jakubkolcar/Downloads/PXL_20260806_114652240.RAW-02.ORIGINAL.dng",
        ];
        for path in cases {
            let path = Path::new(path);
            if !path.exists() {
                continue;
            }
            // rawloader's lossless-JPEG bit reader relies on unsigned
            // wraparound, which debug builds turn into a false-alarm panic;
            // rayon surfaces it as an Err.  Release builds decode fine, so
            // skip rather than fail the test in debug.
            let Some(image) = develop_linear(path, 512) else {
                continue;
            };
            assert!(image.width > 0 && image.height > 0);
            assert_eq!(image.data.len(), image.width as usize * image.height as usize * 3);
            assert!(image.data.iter().all(|v| v.is_finite()));
            // Well-exposed frame: median luminance should be in a sane range.
            let median = image.luminance_percentile(0.5);
            assert!(
                median > 0.005 && median < 2.0,
                "implausible median luminance {median} for {}",
                path.display()
            );
        }
    }

    #[test]
    fn s_curve_applies_monotonically_and_clamps() {
        let curve = SCurve {
            anchor_percentile: 0.5,
            points: vec![[0.0, 0.0], [0.5, 0.2], [1.0, 0.5], [2.0, 1.0]],
        };
        assert_eq!(curve.apply(0.0), 0.0);
        assert_eq!(curve.apply(-1.0), 0.0); // clamps below the first point
        assert_eq!(curve.apply(5.0), 1.0); // clamps beyond the last point
        // Piecewise-linear between the control points.
        assert!((curve.apply(0.75) - 0.35).abs() < 1e-5);
        let mut previous = 0.0;
        for i in 0..200 {
            let x = i as f32 / 199.0 * 3.0;
            let y = curve.apply(x);
            assert!(y >= previous, "not monotone at x = {x}");
            previous = y;
        }
    }

    #[test]
    fn s_curve_round_trips_through_json() {
        let curve = SCurve {
            anchor_percentile: 0.5,
            points: vec![[0.0, 0.0], [1.0, 0.5], [2.0, 0.9]],
        };
        let json = serde_json::to_string(&curve).unwrap();
        let restored: SCurve = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.anchor_percentile, 0.5);
        assert_eq!(restored.points, curve.points);
    }


    #[test]
    fn embedded_curve_starts_at_origin() {
        let curve = SCurve::load().expect("embedded phone S-curve");
        assert!(curve.points.len() >= 2);
        assert_eq!(curve.points[0], [0.0, 0.0]);
        assert_eq!(curve.apply(0.0), 0.0);
        for pair in curve.points.windows(2) {
            assert!(pair[0][0] < pair[1][0], "x must be strictly increasing");
            assert!(pair[0][1] <= pair[1][1], "y must be monotone");
        }
    }

    #[test]
    fn fitted_s_curve_is_monotone_and_starts_at_origin() {
        // Env-gated like the other real-file tests: fits the whole Downloads
        // pair folder (46 DNGs, ~10-20 s in release).
        let Ok(folder) = std::env::var("RAW_FIT_FOLDER") else {
            return;
        };
        let out = std::env::temp_dir().join("raw_s_curve_fit_test.json");
        fit_s_curve(Path::new(&folder), &out).expect("fit should succeed");
        let curve: SCurve =
            serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
        assert_eq!(curve.anchor_percentile, 0.5);
        assert_eq!(curve.points[0], [0.0, 0.0]);
        for pair in curve.points.windows(2) {
            assert!(pair[0][0] < pair[1][0]);
            assert!(pair[0][1] <= pair[1][1]);
        }
        assert_eq!(curve.apply(0.0), 0.0);
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn orientation_matches_imagepipe_transform_semantics() {
        // A small test pattern lets us verify each orientation against the
        // mapping imagepipe's transform op uses (rotate_buffer semantics).
        let make = |rows: &[&str]| -> LinearImage {
            let width = rows[0].len() as u32;
            let height = rows.len() as u32;
            let mut data = Vec::with_capacity(rows.len() * rows[0].len() * 3);
            for row in rows {
                for c in row.chars() {
                    let v = c.to_digit(36).unwrap() as f32 / 10.0;
                    data.extend_from_slice(&[v, v, v]);
                }
            }
            LinearImage {
                width,
                height,
                data,
                orientation: Orientation::Normal,
            }
        };
        let to_rows = |image: &LinearImage| -> Vec<String> {
            image
                .data
                .chunks_exact(image.width as usize * 3)
                .map(|row| {
                    row.chunks_exact(3)
                        .map(|pixel| std::char::from_digit((pixel[0] * 10.0).round() as u32, 36).unwrap())
                        .collect()
                })
                .collect()
        };
        let pattern = make(&["abc", "def"]);
        let cases = [
            (Orientation::Normal, vec!["abc", "def"]),
            (Orientation::HorizontalFlip, vec!["cba", "fed"]),
            (Orientation::VerticalFlip, vec!["def", "abc"]),
            (Orientation::Rotate180, vec!["fed", "cba"]),
            (Orientation::Transpose, vec!["ad", "be", "cf"]),
            (Orientation::Rotate90, vec!["da", "eb", "fc"]),
            (Orientation::Rotate270, vec!["cf", "be", "ad"]),
            (Orientation::Transverse, vec!["fc", "eb", "da"]),
        ];
        for (orientation, expected) in cases {
            let oriented = apply_orientation(&pattern, orientation);
            assert_eq!(to_rows(&oriented), expected, "{orientation:?}");
        }
    }

    #[test]
    fn developed_raw_moves_toward_its_phone_jpeg() {
        // Pooled check over the whole pair folder (env-gated on
        // RAW_VERIFY_FOLDER; ~40 s in release): rawloader-linear + S-curve
        // development must move the render closer to the phone JPEG *on
        // average*.  Individual pairs can move apart — the phone applies
        // scene-adaptive exposure/HDR, so the global curve matches the typical
        // tone placement, not each outlier.  A few developed/JPEG pairs are
        // dumped to /tmp/dev-check for a visual pass.
        let Ok(folder) = std::env::var("RAW_VERIFY_FOLDER") else {
            return;
        };
        let folder = PathBuf::from(folder);
        let curve = SCurve::load().expect("embedded S-curve");
        let pairs = find_pairs(&folder).expect("pair folder");
        let measure = |image: &RgbaImage| {
            crate::processor::LookProfile::measure(
                image.as_raw(),
                image.width(),
                image.height(),
                &[],
            )
            .unwrap()
        };
        let mut before = Vec::new();
        let mut after = Vec::new();
        let mut dumped = 0;
        for (raw, jpeg) in pairs {
            let Some(jpeg_img) = crate::imgload::load_rgba(&jpeg, 512) else {
                continue;
            };
            let Some(developed) = develop_raw(&raw, 512, &curve) else {
                continue;
            };
            let Some(undeveloped) = crate::imgload::load_rgba(&raw, 512) else {
                continue;
            };
            let jpeg_profile = measure(&jpeg_img);
            before.push(profile_distance(&measure(&undeveloped), &jpeg_profile));
            after.push(profile_distance(&measure(&developed), &jpeg_profile));
            if dumped < 3 {
                let name = raw
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("pair");
                let dir = Path::new("/tmp/dev-check");
                std::fs::create_dir_all(dir).ok();
                let _ = developed.save(dir.join(format!("{name}.developed.png")));
                let _ = jpeg_img.save(dir.join(format!("{name}.jpeg.png")));
                dumped += 1;
            }
        }
        assert!(!before.is_empty(), "no usable pairs");
        let mean = |values: &Vec<f32>| values.iter().sum::<f32>() / values.len() as f32;
        let (before_mean, after_mean) = (mean(&before), mean(&after));
        eprintln!(
            "pooled RAW->JPEG profile distance: {before_mean:.3} -> {after_mean:.3} (n={})",
            before.len()
        );
        assert!(after_mean < before_mean, "development must move closer to the phone JPEG on average");
    }
}
