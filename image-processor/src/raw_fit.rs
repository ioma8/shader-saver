//! Offline fitting of the RAW development models from camera DNG/JPEG
//! pairs (CLI: `--fit-raw-scurve`, `--fit-raw-render-model`). Nothing
//! here runs in the editor; the fitted JSON models are embedded at
//! compile time by `raw_develop`.

use crate::raw_develop::{
    apply_render_quantiles, box_blur, decode_raw_u16, load_render_target,
    render_quantiles_u16, render_quantiles_u8, RawRenderModel, Rgba16Image,
    RENDER_FEATURES, RENDER_HIDDEN, SPATIAL_GRID,
};
use image::RgbaImage;
use rawloader::{Orientation, RawImage, RawImageData};
use rayon::prelude::*;
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
    // AsShotWhiteXY), so DNGs surface NaN here.  The real tag (0xC627) is read
    // from the file; note that Pixel DNGs carry AsShotNeutral (1,1,1) as a
    // *placeholder* — the raw data is NOT pre-white-balanced (the sensor's
    // green response runs ~2x red / ~1.5x blue), so identity would leave the
    // image green-dominant.  A flat (1,1,1) or a missing tag falls back to
    // rawloader's D65-neutral camera gains, which is what matches the phone's
    // JPEGs.
    if wb[0].is_nan() || wb[1].is_nan() || wb[2].is_nan() {
        match dng_as_shot_neutral(path) {
            Some(v) if !(v[0] == 1.0 && v[1] == 1.0 && v[2] == 1.0) => wb = v,
            _ => wb = raw.neutralwb(),
        }
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
/// AsShotNeutral values).  kamadak-exif re-reads the whole file, which is a
/// few tens of ms per image next to the rawloader decode itself.
/// # ponytail: full-file re-read per develop; switch to a header-only IFD0
/// scan if batch work (the 46-pair fit) ever grows past a few seconds.
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

/// Combined camera-to-linear-sRGB matrix.  rawloader's *normalized* camera->XYZ
/// keeps camera (1,1,1) on the neutral axis (rows scaled so it maps to equal
/// XYZ), but equal-energy XYZ lands warm in sRGB (sRGB's white is D65, not E),
/// so the rows are re-referenced to the D65 white point.  The composite maps
/// the as-shot white balance's neutral to sRGB D65 white, matching what the
/// phone's JPEGs do.  The raw pseudoinverse alone would not even keep the
/// neutral axis neutral for cameras whose ColorMatrix is not row-balanced.
// Canonical sRGB/XYZ matrices — the literal digits document the standard
// values (Lindbloom); truncating them changes nothing numerically.
#[allow(clippy::excessive_precision)]
fn camera_to_srgb_matrix(raw: &RawImage) -> [[f32; 3]; 3] {
    const XYZ_TO_SRGB: [[f32; 3]; 3] = [
        [3.2404542f32, -1.5371385f32, -0.4985314f32],
        [-0.9692660f32, 1.8760108f32, 0.0415560f32],
        [0.0556434f32, -0.2040259f32, 1.0572252f32],
    ];
    // D65 white point (Y = 1), used to reference camera-neutral to sRGB white.
    const D65_WHITE: [f32; 3] = [0.9504559f32, 1.0f32, 1.088754f32];
    let cam_to_xyz = raw.cam_to_xyz_normalized(); // rows=XYZ, cols=RGB(E)
    let mut out = [[0.0f32; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            for k in 0..3 {
                out[row][col] += XYZ_TO_SRGB[row][k] * cam_to_xyz[k][col] * D65_WHITE[k];
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
    for (y, row) in row_red.iter_mut().enumerate() {
        for (x, col) in col_red.iter_mut().enumerate() {
            match cfa.color_at(y, x) {
                0 => {
                    *row = true;
                    *col = true;
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
    // Flat-buffer indexing (i = y * width + x) plus per-coordinate CFA lookup;
    // clippy's enumerate rewrite would break both.
    #[allow(clippy::needless_range_loop)]
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

struct ChannelHistogram {
    bins: [u64; LUMA_BINS],
    total: u64,
}

impl ChannelHistogram {
    fn new(data: &[f32], channel: usize) -> Self {
        let mut bins = [0u64; LUMA_BINS];
        let mut total = 0;
        for pixel in data.chunks_exact(3) {
            let bin = (pixel[channel].clamp(0.0, 1.0) * (LUMA_BINS - 1) as f32) as usize;
            bins[bin] += 1;
            total += 1;
        }
        Self { bins, total }
    }

    fn percentile(&self, percentile: f64) -> f32 {
        let target = (percentile * self.total as f64).max(1.0) as u64;
        let mut cumulative = 0;
        for (bin, &count) in self.bins.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                return bin as f32 / (LUMA_BINS - 1) as f32;
            }
        }
        1.0
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
/// DNG/JPEG pairs, plus a per-image exposure placement.  `points` maps
/// exposure-normalized linear luminance (luma / anchor-percentile luma) to
/// linear display luminance; `apply` is monotone piecewise-linear and clamps
/// beyond the end points.  `exposure` compensates each image before the curve
/// (input-scaling, like the phone's auto-exposure) so the median lands where
/// the phone's own AE put its JPEG — a fixed curve alone always centers the
/// tone at the pool median, which is why dark scenes came out too bright and
/// bright scenes too dark.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SCurve {
    /// Percentile of linear luminance used as the exposure anchor (0.5).
    pub anchor_percentile: f64,
    /// Per-image exposure placement, fitted from the pairs.
    pub exposure: Exposure,
    /// `(x, y)` pairs, `x` strictly increasing, first point at (0, 0).
    pub points: Vec<[f32; 2]>,
    /// Per-channel calibration after the luminance curve. Pixel DNGs expose a
    /// placeholder AsShotNeutral, so these compact monotone curves learn the
    /// phone's missing color rendering and channel contrast from paired JPEGs.
    #[serde(default = "identity_color_curves")]
    pub color_curves: [Vec<[f32; 2]>; 3],
    #[serde(default)]
    pub rendering: Rendering,
}

#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Rendering {
    pub saturation: f32,
    pub local_contrast: f32,
    pub sharpen: f32,
}

impl Default for Rendering {
    fn default() -> Self {
        Self {
            saturation: 1.0,
            local_contrast: 0.0,
            sharpen: 0.0,
        }
    }
}

fn identity_color_curves() -> [Vec<[f32; 2]>; 3] {
    std::array::from_fn(|_| vec![[0.0, 0.0], [1.0, 1.0]])
}

/// Per-image exposure placement: the phone's JPEG median (linear display
/// luma) as a power law of the RAW's median linear luma, `a * m^b`, fitted
/// from the DNG/JPEG pairs in log-log space.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Exposure {
    pub scale_a: f32,
    pub scale_b: f32,
    #[serde(default)]
    pub samples: Vec<ExposureSample>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct ExposureSample {
    features: [f32; 7],
    target: f32,
    channel_targets: [f32; 3],
    tone_targets: [f32; 5],
    #[serde(default = "identity_white_balance")]
    white_balance: [f32; 3],
    #[serde(default = "one")]
    saturation_scale: f32,
    #[serde(default)]
    spatial_tone: Vec<f32>,
}

fn identity_white_balance() -> [f32; 3] {
    [1.0; 3]
}

fn one() -> f32 {
    1.0
}

impl Exposure {
    /// The samples nearest to `features`, sorted by distance. Computed once
    /// per image and shared by every prediction below.
    fn nearest_of(&self, features: [f32; 7]) -> Vec<(f32, &ExposureSample)> {
        let mut nearest: Vec<_> = self
            .samples
            .iter()
            .map(|sample| {
                let distance = features
                    .iter()
                    .zip(&sample.features)
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                (distance, sample)
            })
            .collect();
        nearest.sort_by(|a, b| a.0.total_cmp(&b.0));
        nearest
    }

    /// Predicted JPEG median (linear display luma) from the RAW's features.
    /// Falls back to the power law `a * m^b` when no samples were fitted.
    fn predict_median_from(&self, nearest: &[(f32, &ExposureSample)], median_luma: f32) -> f32 {
        let predicted = if self.samples.is_empty() {
            None
        } else {
            blend_nearest_scalar(nearest, |sample| sample.target).map(|v| v.clamp(0.001, 1.0))
        };
        predicted.unwrap_or_else(|| self.scale_a * median_luma.powf(self.scale_b))
    }
}

// Inverse-distance weighted average of `extract(sample)` over the nearest
// neighbors; an exact match (distance ~ 0) short-circuits to that sample.
fn blend_nearest<const N: usize>(
    nearest: &[(f32, &ExposureSample)],
    extract: impl Fn(&ExposureSample) -> [f32; N],
) -> Option<[f32; N]> {
    let &(first_distance, first) = nearest.first()?;
    if first_distance < 1e-5 {
        return Some(extract(first));
    }
    let mut output = [0.0; N];
    let mut weights = 0.0;
    for &(distance, sample) in nearest.iter().take(4) {
        let weight = 1.0 / distance.max(1e-3);
        for (value, target) in output.iter_mut().zip(extract(sample)) {
            *value += target * weight;
        }
        weights += weight;
    }
    Some(output.map(|value| value / weights))
}

fn blend_nearest_scalar(
    nearest: &[(f32, &ExposureSample)],
    extract: impl Fn(&ExposureSample) -> f32,
) -> Option<f32> {
    let &(first_distance, first) = nearest.first()?;
    if first_distance < 1e-5 {
        return Some(extract(first));
    }
    let mut output = 0.0;
    let mut weights = 0.0;
    for &(distance, sample) in nearest.iter().take(4) {
        let weight = 1.0 / distance.max(1e-3);
        output += extract(sample) * weight;
        weights += weight;
    }
    Some(output / weights)
}

impl SCurve {
    /// Load the embedded curve fitted from the phone's DNG/JPEG pairs.
    #[cfg(test)]
    pub fn load() -> Option<Self> {
        serde_json::from_str(include_str!("../models/raw_s_curve.json")).ok()
    }

    /// Map exposure-normalized linear luminance to linear display luminance.
    /// Monotone piecewise-linear interpolation; clamps beyond the end points.
    pub fn apply(&self, x: f32) -> f32 {
        apply_curve(&self.points, x)
    }

    fn apply_color(&self, channel: usize, x: f32) -> f32 {
        apply_curve(&self.color_curves[channel], x)
    }

    /// Inverse of `apply`: the x whose mapped value is `y`, clamped to the
    /// curve's domain.  Used for exposure compensation (place the image
    /// median at a predicted display value).
    pub fn inverse_x(&self, y: f32) -> f32 {
        let Some(last) = self.points.last().copied() else {
            return 0.0;
        };
        if y >= last[1] {
            return last[0];
        }
        for pair in self.points.windows(2) {
            let (x0, y0) = (pair[0][0], pair[0][1]);
            let (x1, y1) = (pair[1][0], pair[1][1]);
            if y >= y0 && y < y1 {
                if y1 > y0 {
                    return x0 + (x1 - x0) * (y - y0) / (y1 - y0);
                }
                return x0;
            }
        }
        last[0]
    }
}

fn apply_curve(points: &[[f32; 2]], x: f32) -> f32 {
    let Some(first) = points.first().copied() else {
        return x;
    };
    let last = points[points.len() - 1];
    if x <= first[0] {
        return first[1];
    }
    if x >= last[0] {
        return last[1];
    }
    for pair in points.windows(2) {
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

fn median(values: &mut [f32]) -> f32 {
    values.sort_by(f32::total_cmp);
    values[values.len() / 2]
}

// Sort by x, drop duplicate x keeping the brighter of the tied samples (so the
// curve stays a function), force y monotone non-decreasing, then pin the
// origin so apply(0) == 0.
fn monotone_points(points: Vec<[f32; 2]>) -> Vec<[f32; 2]> {
    let mut points = points;
    points.sort_by(|a, b| a[0].total_cmp(&b[0]));
    let mut out: Vec<[f32; 2]> = Vec::with_capacity(points.len() + 1);
    for mut point in points {
        if let Some(previous) = out.last_mut() {
            if point[0] <= previous[0] {
                previous[1] = previous[1].max(point[1]);
                continue;
            }
            point[1] = point[1].max(previous[1]);
        }
        out.push(point);
    }
    if out.first().is_none_or(|point| point[0] > 0.0) {
        out.insert(0, [0.0, 0.0]);
    } else {
        out[0][1] = 0.0;
    }
    out
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
    // One pair's samples: anchor-normalized x values, y values, and the raw /
    // jpeg medians for the exposure fit.
    type PairSample = (Vec<f32>, Vec<f32>, f32, f32, LinearImage, Vec<f32>);
    // Each pair decodes independently (a full DNG decode each), so the per-
    // pair percentile samples are computed in parallel across cores.
    let pair_samples: Vec<Option<PairSample>> = pairs
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
            let xs = ANCHORS
                .map(|anchor| luma.percentile(anchor) / median)
                .to_vec();
            let ys = ANCHORS.map(|anchor| jpeg_luma.percentile(anchor)).to_vec();
            Some((
                xs,
                ys,
                median,
                jpeg_luma.percentile(0.5),
                linear,
                jpeg_linear,
            ))
        })
        .collect();
    let usable = pair_samples
        .iter()
        .filter(|sample| sample.is_some())
        .count();
    if usable < 20 {
        return Err(format!("only {usable} usable DNG/JPEG pairs found"));
    }
    let mut x_samples: Vec<Vec<f32>> = vec![Vec::new(); ANCHORS.len()];
    let mut y_samples: Vec<Vec<f32>> = vec![Vec::new(); ANCHORS.len()];
    let mut raw_medians = Vec::new();
    let mut jpeg_medians = Vec::new();
    let mut images = Vec::new();
    for (xs, ys, m, j, linear, jpeg_linear) in pair_samples.into_iter().flatten() {
        for (i, (&x, &y)) in xs.iter().zip(&ys).enumerate() {
            x_samples[i].push(x);
            y_samples[i].push(y);
        }
        raw_medians.push(m);
        jpeg_medians.push(j);
        images.push((linear, jpeg_linear));
    }
    let points: Vec<[f32; 2]> = ANCHORS
        .iter()
        .enumerate()
        .map(|(i, _)| [median(&mut x_samples[i]), median(&mut y_samples[i])])
        .collect();
    // Drop duplicate x (keep the brighter of the tied samples so the curve
    // stays a function), then pin the origin so apply(0) == 0.
    let deduped = monotone_points(points);
    // The shape curve's value at the anchor (x = 1), i.e. the pooled JPEG
    // median; used to turn the exposure predictor into a scale.
    let anchor_idx = ANCHORS.iter().position(|&p| p == 0.5).unwrap_or(7);
    let mut shape_median_y = y_samples[anchor_idx].clone();
    let shape_median_y = median(&mut shape_median_y);
    let mut curve = SCurve {
        anchor_percentile: 0.5,
        exposure: fitted_exposure(&raw_medians, &jpeg_medians, shape_median_y),
        points: deduped,
        color_curves: identity_color_curves(),
        rendering: Rendering::default(),
    };
    curve.exposure.samples = images
        .iter()
        .zip(&jpeg_medians)
        .map(|((image, jpeg), &target)| ExposureSample {
            features: exposure_features(image),
            target,
            channel_targets: std::array::from_fn(|channel| {
                ChannelHistogram::new(jpeg, channel).percentile(0.5)
            }),
            tone_targets: {
                let histogram = LumaHistogram::from_rgb_max(jpeg, 1.0);
                [0.05, 0.25, 0.5, 0.75, 0.95].map(|p| histogram.percentile(p))
            },
            white_balance: identity_white_balance(),
            saturation_scale: 1.0,
            spatial_tone: Vec::new(),
        })
        .collect();
    curve.color_curves = fit_color_curves(&images, &curve, &ANCHORS);
    curve.rendering = fit_rendering(&images, &curve);
    curve.rendering.saturation = 1.0;
    let white_balances: Vec<_> = images
        .iter()
        .map(|(linear, jpeg)| {
            let toned = apply_s_curve(linear, &curve);
            let developed = apply_orientation(&toned, toned.orientation);
            fit_white_balance(&developed.data, jpeg)
        })
        .collect();
    for (sample, white_balance) in curve.exposure.samples.iter_mut().zip(white_balances) {
        sample.white_balance = white_balance;
    }
    let saturation_scales: Vec<_> = images
        .iter()
        .map(|(linear, jpeg)| {
            let toned = apply_s_curve(linear, &curve);
            let developed = apply_orientation(&toned, toned.orientation);
            (mean_encoded_saturation(jpeg) / mean_encoded_saturation(&developed.data).max(1e-5))
                .clamp(0.75, 1.25)
        })
        .collect();
    for (sample, saturation_scale) in curve.exposure.samples.iter_mut().zip(saturation_scales) {
        sample.saturation_scale = saturation_scale;
    }
    let spatial_tones: Vec<_> = images
        .iter()
        .map(|(linear, jpeg)| {
            let toned = apply_s_curve(linear, &curve);
            let developed = apply_orientation(&toned, toned.orientation);
            fit_spatial_tone(
                &developed.data,
                jpeg,
                developed.width as usize,
                developed.height as usize,
            )
        })
        .collect();
    for (sample, spatial_tone) in curve.exposure.samples.iter_mut().zip(spatial_tones) {
        sample.spatial_tone = spatial_tone;
    }
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    std::fs::write(
        output,
        serde_json::to_string_pretty(&curve).map_err(|e| e.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(usable)
}

fn fit_color_curves(
    images: &[(LinearImage, Vec<f32>)],
    curve: &SCurve,
    anchors: &[f64],
) -> [Vec<[f32; 2]>; 3] {
    let mut input = std::array::from_fn::<_, 3, _>(|_| vec![Vec::new(); anchors.len()]);
    let mut target = std::array::from_fn::<_, 3, _>(|_| vec![Vec::new(); anchors.len()]);
    for (linear, jpeg) in images {
        let toned = apply_s_curve(linear, curve);
        for channel in 0..3 {
            let a = ChannelHistogram::new(&toned.data, channel);
            let b = ChannelHistogram::new(jpeg, channel);
            for (i, &p) in anchors.iter().enumerate() {
                input[channel][i].push(a.percentile(p));
                target[channel][i].push(b.percentile(p));
            }
        }
    }
    std::array::from_fn(|channel| {
        let points: Vec<[f32; 2]> = (0..anchors.len())
            .map(|i| [median(&mut input[channel][i]), median(&mut target[channel][i])])
            .collect();
        monotone_points(points)
    })
}

fn fit_rendering(images: &[(LinearImage, Vec<f32>)], curve: &SCurve) -> Rendering {
    let mut local = Vec::new();
    let mut detail = Vec::new();
    for (linear, jpeg) in images {
        let developed = apply_s_curve(linear, curve);
        let radius = (developed.width.max(developed.height) / 64).max(2) as usize;
        let ratio = |target: f32, source: f32| {
            if source > 1e-5 {
                target / source
            } else {
                1.0
            }
        };
        local.push(ratio(
            high_pass_energy(jpeg, developed.width, developed.height, radius),
            high_pass_energy(&developed.data, developed.width, developed.height, radius),
        ));
        detail.push(ratio(
            high_pass_energy(jpeg, developed.width, developed.height, 1),
            high_pass_energy(&developed.data, developed.width, developed.height, 1),
        ));
    }
    let local_contrast = ((median(&mut local) - 1.0) * 0.1).clamp(0.0, 0.1);
    let sharpen = ((median(&mut detail) - 1.0) * 0.125).clamp(0.0, 0.15);
    Rendering {
        saturation: 1.0,
        local_contrast,
        sharpen,
    }
}

fn fit_white_balance(source: &[f32], target: &[f32]) -> [f32; 3] {
    let mut ratios = [Vec::new(), Vec::new(), Vec::new()];
    for (source, target) in source.chunks_exact(3).zip(target.chunks_exact(3)) {
        let encoded = |pixel: &[f32]| [pixel[0], pixel[1], pixel[2]].map(srgb_encode);
        let a = encoded(source);
        let b = encoded(target);
        let saturation = |pixel: [f32; 3]| {
            let max = pixel[0].max(pixel[1]).max(pixel[2]);
            let min = pixel[0].min(pixel[1]).min(pixel[2]);
            (max - min) / max.max(1e-4)
        };
        let luma = 0.2126 * a[0] + 0.7152 * a[1] + 0.0722 * a[2];
        if saturation(a) >= 0.12
            || saturation(b) >= 0.12
            || !(0.08..=0.8).contains(&luma)
            || source.iter().any(|&value| value < 1e-4)
            || target.iter().any(|&value| value < 1e-4)
        {
            continue;
        }
        let green_ratio = target[1] / source[1];
        for channel in 0..3 {
            ratios[channel].push((target[channel] / source[channel]) / green_ratio);
        }
    }
    std::array::from_fn(|channel| {
        let values = &mut ratios[channel];
        if values.is_empty() {
            return 1.0;
        }
        median(values).clamp(0.75, 1.25)
    })
}

const SPATIAL_TONE_GRID: usize = 64;

fn fit_spatial_tone(source: &[f32], target: &[f32], width: usize, height: usize) -> Vec<f32> {
    fit_spatial_residual(source, target, width, height, SPATIAL_TONE_GRID, 0.08, 0.12)
}

fn fit_spatial_residual(
    source: &[f32],
    target: &[f32],
    width: usize,
    height: usize,
    grid: usize,
    max_alignment_error: f32,
    max_residual: f32,
) -> Vec<f32> {
    if source.len() != target.len() || source.len() != width * height * 3 {
        return Vec::new();
    }
    let alignment_error = source
        .iter()
        .zip(target)
        .map(|(&a, &b)| (srgb_encode(a) - srgb_encode(b)).abs())
        .sum::<f32>()
        / source.len().max(1) as f32;
    if alignment_error > max_alignment_error {
        return Vec::new();
    }
    let mut sums = vec![0.0f64; grid * grid * 3];
    let mut counts = vec![0u64; grid * grid];
    for y in 0..height {
        for x in 0..width {
            let cell = (y * grid / height).min(grid - 1) * grid + (x * grid / width).min(grid - 1);
            let pixel = y * width + x;
            for channel in 0..3 {
                sums[cell * 3 + channel] += f64::from(
                    srgb_encode(target[pixel * 3 + channel])
                        - srgb_encode(source[pixel * 3 + channel]),
                );
            }
            counts[cell] += 1;
        }
    }
    sums.iter()
        .enumerate()
        .map(|(index, &sum)| {
            ((sum / counts[index / 3].max(1) as f64) as f32).clamp(-max_residual, max_residual)
        })
        .collect()
}

fn mean_encoded_saturation(data: &[f32]) -> f32 {
    data.chunks_exact(3)
        .map(|pixel| {
            let pixel = [pixel[0], pixel[1], pixel[2]].map(srgb_encode);
            let max = pixel[0].max(pixel[1]).max(pixel[2]);
            let min = pixel[0].min(pixel[1]).min(pixel[2]);
            (max - min) / max.max(1e-4)
        })
        .sum::<f32>()
        / (data.len() / 3).max(1) as f32
}

fn high_pass_energy(data: &[f32], width: u32, height: u32, radius: usize) -> f32 {
    let luma: Vec<f32> = data
        .chunks_exact(3)
        .map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2])
        .collect();
    let blurred = box_blur(&luma, width as usize, height as usize, radius);
    luma.iter()
        .zip(blurred)
        .map(|(&value, blur)| (value - blur).abs())
        .sum::<f32>()
        / luma.len().max(1) as f32
}


fn fit_render_spatial_residual(raw: &Rgba16Image, target: &RgbaImage, source: &[f32; RENDER_FEATURES], curve: &[f32; RENDER_FEATURES]) -> Vec<f32> {
    let mut base = raw.clone();
    apply_render_quantiles(&mut base, source, curve);
    let target = align_render_target(&base, target);
    let mut sums = vec![0.0f64; SPATIAL_GRID * SPATIAL_GRID * 3];
    let mut counts = vec![0u32; SPATIAL_GRID * SPATIAL_GRID];
    let width = raw.width().max(1) as usize;
    let height = raw.height().max(1) as usize;
    for y in 0..height {
        for x in 0..width {
            let gx = x * SPATIAL_GRID / width;
            let gy = y * SPATIAL_GRID / height;
            let cell = gy * SPATIAL_GRID + gx;
            counts[cell] += 1;
            let a = base.get_pixel(x as u32, y as u32);
            let b = target.get_pixel(x as u32, y as u32);
            for channel in 0..3 {
                sums[cell * 3 + channel] += f64::from(u16::from(b[channel]) * 257) - f64::from(a[channel]);
            }
        }
    }
    sums.into_iter().enumerate().map(|(index, value)| {
        let count = counts[index / 3].max(1) as f64;
        (value / count / 65535.0).clamp(-0.25, 0.25) as f32
    }).collect()
}

fn align_render_target(base: &Rgba16Image, target: &RgbaImage) -> RgbaImage {
    let mut candidates = vec![target.clone(), image::imageops::rotate180(target)];
    if target.width() != target.height() {
        candidates.push(image::imageops::rotate90(target));
        candidates.push(image::imageops::rotate270(target));
    }
    candidates
        .into_iter()
        .map(|candidate| {
            let candidate = if candidate.dimensions() == base.dimensions() {
                candidate
            } else {
                image::imageops::resize(&candidate, base.width(), base.height(), image::imageops::FilterType::Triangle)
            };
            let error = base.pixels().zip(candidate.pixels()).map(|(a, b)| {
                (0..3).map(|channel| (f32::from(a[channel]) - f32::from(u16::from(b[channel]) * 257)).abs()).sum::<f32>()
            }).sum::<f32>();
            (error, candidate)
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, candidate)| candidate)
        .unwrap_or_else(|| image::imageops::resize(target, base.width(), base.height(), image::imageops::FilterType::Triangle))
}

/// Train the compact fallback rendering network from RAW/JPEG pairs below
/// `folder` and write the deployable JSON model. Runtime uses this model only
/// when no exact paired render is available.
pub fn fit_raw_render_model(folder: &Path, output: &Path) -> Result<usize, String> {
    fn visit(folder: &Path, paths: &mut Vec<PathBuf>) -> Result<(), String> {
        for entry in std::fs::read_dir(folder).map_err(|error| error.to_string())? {
            let path = entry.map_err(|error| error.to_string())?.path();
            if path.is_dir() {
                visit(&path, paths)?;
            } else if crate::imgload::is_raw(&path) {
                paths.push(path);
            }
        }
        Ok(())
    }

    let mut paths = Vec::new();
    visit(folder, &mut paths)?;
    paths.sort();
    let samples: Vec<_> = paths
        .par_iter()
        .filter_map(|path| {
            let Some(raw) = decode_raw_u16(path, 512) else {
                eprintln!("RAW decode failed: {}", path.display());
                return None;
            };
            let Some(jpeg) = load_render_target(path, 512) else {
                eprintln!("training target failed: {}", path.display());
                return None;
            };
            let input = render_quantiles_u16(&raw);
            let target = render_quantiles_u8(&jpeg);
            let mut target_curve = [0.0f32; RENDER_FEATURES];
            target_curve.copy_from_slice(&target);
            let spatial = fit_render_spatial_residual(&raw, &jpeg, &input, &target_curve);
            Some((input, target, spatial))
        })
        .collect();
    if samples.len() < 20 {
        return Err(format!(
            "only {} usable RAW/JPEG pairs found",
            samples.len()
        ));
    }
    let sample_count = samples.len();

    let mut mean = vec![0.0; RENDER_FEATURES];
    for (input, _, _) in &samples {
        for (value, input) in mean.iter_mut().zip(input) {
            *value += input;
        }
    }
    for value in &mut mean {
        *value /= samples.len() as f32;
    }
    let mut scale = vec![0.0; RENDER_FEATURES];
    for (input, _, _) in &samples {
        for i in 0..RENDER_FEATURES {
            scale[i] += (input[i] - mean[i]).powi(2);
        }
    }
    for value in &mut scale {
        *value = (samples.len() as f32 / value.max(1e-6)).sqrt().min(20.0);
    }

    let mut seed = 0x6d2b_79f5u32;
    let mut random = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed as f32 / u32::MAX as f32
    };
    let mut model = RawRenderModel {
        mean,
        scale,
        w1: (0..RENDER_HIDDEN * RENDER_FEATURES)
            .map(|_| (random() * 2.0 - 1.0) * 0.08)
            .collect(),
        b1: vec![0.0; RENDER_HIDDEN],
        w2: (0..RENDER_FEATURES * RENDER_HIDDEN)
            .map(|_| (random() * 2.0 - 1.0) * 0.08)
            .collect(),
        b2: vec![0.0; RENDER_FEATURES],
        prototypes: Vec::new(),
        prototype_targets: Vec::new(),
        spatial_targets: Vec::new(),
    };

    for step in 0..500_000 {
        let sample = &samples[(random() * samples.len() as f32) as usize % samples.len()];
        let mut normalized = [0.0; RENDER_FEATURES];
        for i in 0..RENDER_FEATURES {
            normalized[i] = (sample.0[i] - model.mean[i]) * model.scale[i];
        }
        let mut hidden = [0.0; RENDER_HIDDEN];
        for (h, value) in hidden.iter_mut().enumerate() {
            let mut sum = model.b1[h];
            for i in 0..RENDER_FEATURES {
                sum += model.w1[h * RENDER_FEATURES + i] * normalized[i];
            }
            *value = sum.tanh();
        }
        let mut predicted = [0.0; RENDER_FEATURES];
        for (o, value) in predicted.iter_mut().enumerate() {
            let mut sum = model.b2[o];
            for h in 0..RENDER_HIDDEN {
                sum += model.w2[o * RENDER_HIDDEN + h] * hidden[h];
            }
            *value = 1.0 / (1.0 + (-sum).exp());
        }
        let learning_rate = 0.015 * (1.0 - step as f32 / 650_000.0);
        let mut gradient = [0.0; RENDER_FEATURES];
        for o in 0..RENDER_FEATURES {
                gradient[o] = (predicted[o] - sample.1[o]) * predicted[o] * (1.0 - predicted[o]);
        }
        let hidden_gradient: [f32; RENDER_HIDDEN] = std::array::from_fn(|h| {
            (0..RENDER_FEATURES)
                .map(|o| gradient[o] * model.w2[o * RENDER_HIDDEN + h])
                .sum::<f32>()
                * (1.0 - hidden[h].powi(2))
        });
        for o in 0..RENDER_FEATURES {
            for h in 0..RENDER_HIDDEN {
                model.w2[o * RENDER_HIDDEN + h] -= learning_rate * gradient[o] * hidden[h];
            }
            model.b2[o] -= learning_rate * gradient[o];
        }
        for h in 0..RENDER_HIDDEN {
            for (i, input) in normalized.iter().enumerate() {
                model.w1[h * RENDER_FEATURES + i] -= learning_rate * hidden_gradient[h] * input;
            }
            model.b1[h] -= learning_rate * hidden_gradient[h];
        }
    }

    let error = samples
        .iter()
        .map(|(input, target, _)| {
            model
                .predict(input)
                .iter()
                .zip(target)
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / RENDER_FEATURES as f32
        })
        .sum::<f32>()
        / samples.len() as f32;
    model.prototypes = samples.iter().map(|(input, _, _)| input.to_vec()).collect();
    model.prototype_targets = samples.iter().map(|(_, target, _)| target.to_vec()).collect();
    model.spatial_targets = samples.iter().map(|(_, _, spatial)| spatial.clone()).collect();
    eprintln!("trained RAW render model: quantile MAE {error:.4}");

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    std::fs::write(
        output,
        serde_json::to_string(&model).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(sample_count)
}

/// Apply the S-curve hue-preservingly: exposure-normalize each pixel's luma by
/// the anchor (median linear luminance), map through the curve, and scale RGB
/// by `display / luma` so the output keeps hue and lands on the curve's luma.
/// Fit the phone's per-image exposure placement: JPEG median (linear display
/// luma) as a power law of the RAW's median linear luma, least-squares in
/// log-log space over all pairs.  `shape_median_y` is the shape curve's value
/// at the anchor, used as the fallback scale for degenerate fits.
fn exposure_features(image: &LinearImage) -> [f32; 7] {
    let histogram = LumaHistogram::from_rgb(&image.data);
    let median_luma = histogram.percentile(0.5).max(1e-5);
    let channels = std::array::from_fn::<_, 3, _>(|channel| {
        ChannelHistogram::new(&image.data, channel)
            .percentile(0.5)
            .max(1e-5)
    });
    [
        median_luma.ln(),
        (histogram.percentile(0.1).max(1e-5) / median_luma).ln(),
        (histogram.percentile(0.25).max(1e-5) / median_luma).ln(),
        (histogram.percentile(0.75).max(1e-5) / median_luma).ln(),
        (histogram.percentile(0.9).max(1e-5) / median_luma).ln(),
        (channels[0] / channels[1]).ln(),
        (channels[2] / channels[1]).ln(),
    ]
}

fn fitted_exposure(raw_medians: &[f32], jpeg_medians: &[f32], shape_median_y: f32) -> Exposure {
    let mut log_x = Vec::new();
    let mut log_y = Vec::new();
    for (&m, &j) in raw_medians.iter().zip(jpeg_medians) {
        if m > 0.0 && j > 0.0 {
            log_x.push((m as f64).ln());
            log_y.push((j as f64).ln());
        }
    }
    let n = log_x.len() as f64;
    if n < 2.0 {
        return Exposure {
            scale_a: shape_median_y,
            scale_b: 0.0,
            samples: Vec::new(),
        };
    }
    let mean_x = log_x.iter().sum::<f64>() / n;
    let mean_y = log_y.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var = 0.0;
    for (&x, &y) in log_x.iter().zip(&log_y) {
        cov += (x - mean_x) * (y - mean_y);
        var += (x - mean_x).powi(2);
    }
    let scale_b = (cov / var) as f32;
    let scale_a = (mean_y - scale_b as f64 * mean_x).exp() as f32;
    Exposure {
        scale_a,
        scale_b,
        samples: Vec::new(),
    }
}

fn apply_s_curve(image: &LinearImage, curve: &SCurve) -> LinearImage {
    let anchor = image
        .luminance_percentile(curve.anchor_percentile)
        .max(1e-3);
    // Per-image exposure compensation, applied to the linear values *before*
    // the curve (as the phone's AE does): pick the curve input k whose output
    // is the predicted JPEG median for this exposure level, then map through
    // the shape curve.  Scaling before the curve keeps the neutral axis intact
    // and clips highlights the way a real exposure change would.
    // The exposure features, the neighbor list and the median are computed
    // once and shared by every prediction stage.
    let features = exposure_features(image);
    let median_luma = features[0].exp();
    let nearest = curve.exposure.nearest_of(features);
    let target = curve.exposure.predict_median_from(&nearest, median_luma);
    let k = curve.inverse_x(target);
    let mut data = image.data.clone();
    for pixel in data.chunks_exact_mut(3) {
        let luma = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];
        let scale = if luma > 0.0 {
            curve.apply(luma / anchor * k) / luma
        } else {
            0.0
        };
        for (channel, value) in pixel.iter_mut().enumerate() {
            *value = curve
                .apply_color(channel, (*value * scale).clamp(0.0, 1.0))
                .clamp(0.0, 1.0);
        }
    }
    apply_rendering(
        &mut data,
        image.width as usize,
        image.height as usize,
        curve.rendering,
    );
    if let Some(targets) = blend_nearest(&nearest, |sample| sample.channel_targets) {
        for channel in 0..3 {
            let current = ChannelHistogram::new(&data, channel)
                .percentile(0.5)
                .max(1e-5);
            let scale = targets[channel] / current;
            for pixel in data.chunks_exact_mut(3) {
                pixel[channel] = (pixel[channel] * scale).clamp(0.0, 1.0);
            }
        }
    }
    if let Some(white_balance) = blend_nearest(&nearest, |sample| sample.white_balance) {
        for pixel in data.chunks_exact_mut(3) {
            for channel in 0..3 {
                pixel[channel] = (pixel[channel] * white_balance[channel]).clamp(0.0, 1.0);
            }
        }
    }
    if let Some(saturation) = blend_nearest_scalar(&nearest, |sample| sample.saturation_scale) {
        for pixel in data.chunks_exact_mut(3) {
            let luma = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];
            for value in pixel {
                *value = (luma + (*value - luma) * saturation).clamp(0.0, 1.0);
            }
        }
    }
    if let Some(targets) = blend_nearest(&nearest, |sample| sample.tone_targets) {
        let histogram = LumaHistogram::from_rgb_max(&data, 1.0);
        let mut points = vec![[0.0, 0.0]];
        for (p, target) in [0.05, 0.25, 0.5, 0.75, 0.95].into_iter().zip(targets) {
            let x = histogram.percentile(p);
            if x > points.last().unwrap()[0] {
                points.push([x, target.max(points.last().unwrap()[1])]);
            }
        }
        if points.last().is_none_or(|point| point[0] < 1.0) {
            points.push([1.0, 1.0]);
        }
        for pixel in data.chunks_exact_mut(3) {
            let luma = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];
            let scale = if luma > 0.0 {
                apply_curve(&points, luma) / luma
            } else {
                0.0
            };
            for value in pixel {
                *value = (*value * scale).clamp(0.0, 1.0);
            }
        }
    }
    LinearImage {
        width: image.width,
        height: image.height,
        data,
        orientation: image.orientation,
    }
}

fn apply_rendering(data: &mut [f32], width: usize, height: usize, rendering: Rendering) {
    if (rendering.saturation - 1.0).abs() > 1e-4 {
        for pixel in data.chunks_exact_mut(3) {
            let luma = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];
            for value in pixel {
                *value = (luma + (*value - luma) * rendering.saturation).clamp(0.0, 1.0);
            }
        }
    }
    let mut enhance = |radius: usize, amount: f32| {
        if amount <= 1e-4 {
            return;
        }
        let luma: Vec<f32> = data
            .chunks_exact(3)
            .map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2])
            .collect();
        let blurred = box_blur(&luma, width, height, radius);
        for ((pixel, &value), blur) in data.chunks_exact_mut(3).zip(&luma).zip(blurred) {
            let adjusted =
                (value + amount * (value - blur)).clamp(value * 0.6, (value * 1.6).min(1.0));
            let scale = if value > 1e-5 { adjusted / value } else { 0.0 };
            for channel in pixel {
                *channel = (*channel * scale).clamp(0.0, 1.0);
            }
        }
    };
    enhance((width.max(height) / 64).max(2), rendering.local_contrast);
    enhance(1, rendering.sharpen);
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
pub fn profile_distance(
    a: &crate::processor::LookProfile,
    b: &crate::processor::LookProfile,
) -> f32 {
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
                assert_eq!(
                    image.data.len(),
                    image.width as usize * image.height as usize * 3
                );
                assert!(image.data.iter().all(|v| v.is_finite()));
                // Well-exposed frame: median luminance should be in a sane range.
                let median = image.luminance_percentile(0.5);
                assert!(
                    median > 0.001 && median < 2.0,
                    "implausible median luminance {median} for {}",
                    path.display()
                );
            }
        }

        #[test]
        fn s_curve_applies_monotonically_and_clamps() {
            let curve = SCurve {
                anchor_percentile: 0.5,
                exposure: Exposure {
                    scale_a: 0.158,
                    scale_b: 0.0,
                    samples: Vec::new(),
                },
                points: vec![[0.0, 0.0], [0.5, 0.2], [1.0, 0.5], [2.0, 1.0]],
                color_curves: identity_color_curves(),
                rendering: Rendering::default(),
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
                exposure: Exposure {
                    scale_a: 0.158,
                    scale_b: 0.0,
                    samples: Vec::new(),
                },
                points: vec![[0.0, 0.0], [1.0, 0.5], [2.0, 0.9]],
                color_curves: identity_color_curves(),
                rendering: Rendering::default(),
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
            for sample in &curve.exposure.samples {
                assert!(
                    sample.spatial_tone.is_empty()
                        || sample.spatial_tone.len() == SPATIAL_TONE_GRID.pow(2) * 3
                );
                assert!(sample
                    .spatial_tone
                    .iter()
                    .all(|value| value.is_finite() && (-0.12..=0.12).contains(value)));
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
            let curve: SCurve = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
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
                            .map(|pixel| {
                                std::char::from_digit((pixel[0] * 10.0).round() as u32, 36).unwrap()
                            })
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
            // RAW_VERIFY_FOLDER; ~40 s in release): universal development must
            // closely approach the phone JPEG. The old "must beat the embedded
            // preview" comparison was dropped when the RAW browse render became
            // the development itself (375aa00): for Pixel pairs the embedded
            // preview *is* the reference JPEG, so nothing can beat it and the
            // baseline was vacuous. The absolute gate still catches a broken
            // pipeline. A few pairs are dumped to /tmp/dev-check for a visual
            // pass.
            let Ok(folder) = std::env::var("RAW_VERIFY_FOLDER") else {
                return;
            };
            let folder = PathBuf::from(folder);
            let pairs = find_pairs(&folder).expect("pair folder");
            let pair_filter = std::env::var("RAW_VERIFY_PAIR").ok();
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
            let mut pixel_errors = Vec::new();
            let mut dumped = 0;
            for (raw, jpeg) in pairs {
                if pair_filter
                    .as_ref()
                    .is_some_and(|filter| !raw.to_string_lossy().contains(filter))
                {
                    continue;
                }
                let Some(jpeg_img) = crate::imgload::load_rgba(&jpeg, 512) else {
                    continue;
                };
                let Some(developed) = crate::raw_develop::develop_raw(&raw, 512) else {
                    continue;
                };
                let Some(undeveloped) = crate::imgload::load_preview_rgba(&raw, 512) else {
                    continue;
                };
                let jpeg_profile = measure(&jpeg_img);
                let developed_profile = measure(&developed);
                let pixel_error = developed
                    .as_raw()
                    .chunks_exact(4)
                    .zip(jpeg_img.as_raw().chunks_exact(4))
                    .flat_map(|(a, b)| (0..3).map(move |channel| (a[channel], b[channel])))
                    .map(|(a, b)| u8::abs_diff(a, b) as f32 / 255.0)
                    .sum::<f32>()
                    / (developed.width() * developed.height() * 3).max(1) as f32;
                before.push(profile_distance(&measure(&undeveloped), &jpeg_profile));
                after.push(profile_distance(&developed_profile, &jpeg_profile));
                pixel_errors.push(pixel_error);
                eprintln!(
                    "{}: {:.3} -> {:.3}, pixel MAE {:.2}%",
                    raw.file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("pair"),
                    before.last().unwrap(),
                    after.last().unwrap(),
                    pixel_error * 100.0
                );
                if dumped < 3 || pair_filter.is_some() {
                    let name = raw.file_name().and_then(|n| n.to_str()).unwrap_or("pair");
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
            let worst_profile_error = after.iter().copied().fold(0.0, f32::max);
            let worst_pixel_error = pixel_errors.iter().copied().fold(0.0, f32::max);
            eprintln!(
                "pooled RAW->JPEG profile distance: {before_mean:.3} -> {after_mean:.3}; pixel MAE mean {:.2}%, worst {:.2}% (n={})",
                mean(&pixel_errors) * 100.0,
                worst_pixel_error * 100.0,
                before.len()
            );
            assert!(
                after_mean < 1.0,
                "development must closely approach the phone JPEG on average"
            );
            assert!(
                worst_profile_error < 3.1,
                "a Pixel pair is visibly divergent"
            );
        }
}
