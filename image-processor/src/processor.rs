use half::f16;
use std::path::Path;

#[repr(align(4))]
struct AlignedPhotoLuts([u8; 1_293_732]);

static PHOTO_LUTS: AlignedPhotoLuts = AlignedPhotoLuts(*include_bytes!("../models/photo_luts.f32"));

fn photo_luts() -> &'static [f32] {
    bytemuck::cast_slice(&PHOTO_LUTS.0)
}

fn combined_photo_lut(weights: [f32; 3]) -> Vec<f32> {
    const GRID: usize = 33 * 33 * 33;
    const BASIS: usize = 3 * GRID;
    const TONE_STRENGTH: f32 = 0.82;
    const COLOR_STRENGTH: f32 = 0.78;
    const LUM: [f32; 3] = [0.2126, 0.7152, 0.0722];

    let bases = photo_luts();
    let sample = |i: usize, channel: usize| {
        let offset = channel * GRID + i;
        weights[0] * bases[offset]
            + weights[1] * bases[BASIS + offset]
            + weights[2] * bases[2 * BASIS + offset]
    };
    let mid = 16 + 16 * 33 + 16 * 33 * 33;
    let mid_lum = (0..3).map(|c| sample(mid, c) * LUM[c]).sum::<f32>();
    let exposure_gain = (mid_lum / 0.5).clamp(0.5, 2.0);

    let mut result = Vec::with_capacity(3 * GRID);
    for i in 0..GRID {
        let input = voxel_rgb(i, 33);
        let output = [sample(i, 0), sample(i, 1), sample(i, 2)];
        let input_lum = input.iter().zip(LUM).map(|(v, w)| v * w).sum::<f32>();
        let output_lum = output.iter().zip(LUM).map(|(v, w)| v * w).sum::<f32>();
        let target_lum = input_lum * exposure_gain;
        let softened_lum = target_lum + (output_lum - target_lum) * TONE_STRENGTH;
        for channel in 0..3 {
            let source_chroma = input[channel] - input_lum;
            let model_chroma = output[channel] - output_lum;
            result.push(
                softened_lum + source_chroma + (model_chroma - source_chroma) * COLOR_STRENGTH,
            );
        }
    }
    result
}

/// Put the settings that describe how an image is *developed* onto another one,
/// leaving alone the things that belong to a single photograph.
///
/// Framing and spot removal stay: a crop or a retouch dab pasted onto a
/// neighbour would be wrong in a way the user would have to undo one frame at a
/// time. Everything else travels, including `raw_isp_enabled` -- whether the
/// pixels are a sensor render or the camera's preview decides what the settings
/// are applied *to*, so a develop pasted onto a frame still rendered from a
/// preview would not look like the frame it came from.
pub fn copy_develop_settings(to: &mut EditState, from: &EditState) {
    let crop = to.crop;
    let retouch_spots = std::mem::take(&mut to.retouch_spots);
    *to = from.clone();
    to.crop = crop;
    to.retouch_spots = retouch_spots;
}

/// Most spots one image can carry. Each is a fixed slot in a uniform buffer,
/// so this is a hard cap rather than a growing allocation.
pub const MAX_RETOUCH_SPOTS: usize = 64;

/// Uniform buffer that carries the dab list to the shader: a four-float header
/// (count + padding) followed by twelve floats per spot (three `vec4`s).
fn retouch_buffer_bytes() -> u64 {
    ((4 + MAX_RETOUCH_SPOTS * 12) * std::mem::size_of::<f32>()) as u64
}

/// Pull a loaded dab list back into the range the shader can be trusted with.
///
/// Nothing the tool produces is ever out of range, but the list travels through
/// a JSON column that a person (or an older build) can write anything into, and
/// a single NaN would NaN every pixel the pass touches -- a blank image with no
/// obvious cause. Out-of-range values are clamped and unusable dabs dropped.
fn sanitize_retouch_spots(spots: &mut Vec<RetouchSpot>) {
    spots.retain(|s| {
        s.x.is_finite()
            && s.y.is_finite()
            && s.radius.is_finite()
            && s.source.iter().all(|v| v.is_finite())
            && s.gain.iter().all(|v| v.is_finite())
            && s.bias.iter().all(|v| v.is_finite())
    });
    for spot in spots.iter_mut() {
        spot.x = spot.x.clamp(0.0, 1.0);
        spot.y = spot.y.clamp(0.0, 1.0);
        spot.radius = spot.radius.clamp(0.0005, 0.5);
        spot.source[0] = spot.source[0].clamp(-1.0, 1.0);
        spot.source[1] = spot.source[1].clamp(-1.0, 1.0);
        spot.gain = spot.gain.map(|g| g.clamp(0.25, 4.0));
        spot.bias = spot.bias.map(|b| b.clamp(-1.0, 1.0));
    }
    spots.truncate(MAX_RETOUCH_SPOTS);
}

/// Pack the dab list into the shader's uniform layout: a four-float header
/// (count plus padding) then three `vec4`s per spot. A free function so the
/// layout the shader reads can be asserted without a GPU.
fn pack_retouch_spots(spots: &[RetouchSpot]) -> Vec<f32> {
    let mut data = vec![0f32; 4 + MAX_RETOUCH_SPOTS * 12];
    let count = spots.len().min(MAX_RETOUCH_SPOTS);
    data[0] = count as f32;
    for (slot, spot) in spots[..count].iter().enumerate() {
        let o = 4 + slot * 12;
        // a = (x, y, radius, gain.r), b = (dx, dy, gain.g, gain.b),
        // c = (bias.r, bias.g, bias.b, unused)
        data[o] = spot.x;
        data[o + 1] = spot.y;
        data[o + 2] = spot.radius;
        data[o + 3] = spot.gain[0];
        data[o + 4] = spot.source[0];
        data[o + 5] = spot.source[1];
        data[o + 6] = spot.gain[1];
        data[o + 7] = spot.gain[2];
        data[o + 8] = spot.bias[0];
        data[o + 9] = spot.bias[1];
        data[o + 10] = spot.bias[2];
        data[o + 11] = 0.0;
    }
    data
}

/// One healing-brush dab. Stored resolution-independently so a spot means the
/// same thing whichever render resolution the image is reopened at: the centre
/// is a fraction of width/height, `radius` a fraction of the shorter side, and
/// `source` the offset (also fractions) to the patch that replaces this one.
///
/// `gain`/`bias` are the affine colour correction that makes the sampled patch
/// match the pixels around the dab -- the "healing" half of a healing brush --
/// fitted from the *rim* of the two patches when the spot was placed. The gain
/// carries most of it, so a later exposure change (which scales the source
/// patch) still lands near the target; a bare additive offset would go stale.
///
/// `serde(default)` matters: dabs written before `gain` existed must keep
/// loading, or the whole edit state for that image would be discarded.
#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RetouchSpot {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub source: [f32; 2],
    pub gain: [f32; 3],
    pub bias: [f32; 3],
}

impl Default for RetouchSpot {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            radius: 0.05,
            source: [0.0; 2],
            // Identity correction: an old dab keeps its former meaning.
            gain: [1.0; 3],
            bias: [0.0; 3],
        }
    }
}

// Everything that defines an image's edit, serialized to SQLite as JSON.
// serde(default) keeps old DB rows loadable when new fields are added.
#[derive(serde::Serialize, serde::Deserialize, Clone, PartialEq)]
#[serde(default)]
pub struct EditState {
    pub camera_preset: Option<crate::camera_presets::CameraPreset>,
    pub camera_style: u8,
    pub camera_strength: f32,
    pub exposure: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub vibrance: f32,
    pub wb_temp: f32,
    pub wb_tint: f32,
    pub levels_black: f32,
    pub levels_white: f32,
    pub levels_gamma: f32,
    pub blur_radius: f32,
    pub unsharp_strength: f32,
    pub unsharp_blur_radius: f32,
    /// Whether the RAW-specific sharpening default has been initialized.
    /// This lets an explicit slider value of zero survive reopening.
    pub raw_sharpening_initialized: bool,
    pub blacks: f32,
    pub shadows: f32,
    pub highlights: f32,
    pub whites: f32,
    pub vignette: f32,
    pub vignette_mid: f32,
    pub curve_points: Vec<[f32; 2]>,
    pub ai_lut_enabled: bool,
    pub ai_lut_weights: [f32; 3],
    pub ai_lut_strength: f32,
    // Full-resolution neural RAW development has already been baked into the
    // processor input texture.  It is persisted as a flag so reopening an
    // image repeats the deterministic inference.
    pub raw_isp_enabled: bool,
    // Device rendering predicted from paired camera RAW/JPEG captures.
    pub raw_development: Option<Vec<f32>>,
    // The constrained model's captured-look transform.
    pub look: Vec<LookTransfer>,
    // Reference-conditioned canonicalize/restylize transform.
    pub look_lut: Option<Vec<f32>>,
    pub look_strength: f32,
    // Healing-brush layer: removable dabs composited after the edit chain.
    pub retouch_spots: Vec<RetouchSpot>,
    /// Non-destructive crop, as `(x, y, w, h)` fractions of the input image.
    /// The full frame `[0, 0, 1, 1]` means no crop. Snapped to whole pixels
    /// whenever it is set, so the exported pixels are exactly the ones the
    /// panel showed -- see `crop_px`.
    pub crop: [f32; 4],
}

impl Default for EditState {
    fn default() -> Self {
        Self {
            camera_preset: None,
            camera_style: 0,
            camera_strength: 1.0,
            exposure: 0.0,
            brightness: 0.0,
            contrast: 0.0,
            saturation: 0.0,
            vibrance: 0.0,
            wb_temp: 0.0,
            wb_tint: 0.0,
            levels_black: 0.0,
            levels_white: 255.0,
            levels_gamma: 1.0,
            blur_radius: 0.0,
            unsharp_strength: 0.0,
            unsharp_blur_radius: 2.0,
            raw_sharpening_initialized: false,
            blacks: 0.0,
            shadows: 0.0,
            highlights: 0.0,
            whites: 0.0,
            vignette: 0.0,
            vignette_mid: 50.0,
            curve_points: vec![[0.0, 0.0], [1.0, 1.0]],
            ai_lut_enabled: false,
            ai_lut_weights: [0.0; 3],
            ai_lut_strength: 1.0,
            raw_isp_enabled: false,
            raw_development: None,
            look: Vec::new(),
            look_lut: None,
            look_strength: 1.0,
            retouch_spots: Vec::new(),
            crop: FULL_CROP,
        }
    }
}

/// No crop.
pub const FULL_CROP: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

/// What an export is written as.
#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub enum ExportFormat {
    /// 16-bit PNG: keeps the whole depth of the edit chain, for further work.
    Png16,
    /// 8-bit JPEG with a quality setting: what a client or a print shop wants.
    Jpeg,
}

impl ExportFormat {
    pub fn label(self) -> &'static str {
        match self {
            Self::Png16 => "PNG (16-bit)",
            Self::Jpeg => "JPEG",
        }
    }
    pub fn extension(self) -> &'static str {
        match self {
            Self::Png16 => "png",
            Self::Jpeg => "jpg",
        }
    }
    pub fn key(self) -> &'static str {
        match self {
            Self::Png16 => "png16",
            Self::Jpeg => "jpeg",
        }
    }
    pub fn from_key(key: &str) -> Self {
        match key {
            "jpeg" => Self::Jpeg,
            _ => Self::Png16,
        }
    }
}

/// How to write an export.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ExportOptions {
    pub format: ExportFormat,
    /// JPEG quality, 1..=100. Ignored for PNG.
    pub quality: u8,
    /// Longest side of the output, or 0 to keep the edited size.
    pub max_dim: u32,
    /// Resolution in pixels per inch to record in the file, or 0 to record
    /// none. A JPEG written without one says nothing about how big it is, so
    /// whoever prints it has to guess: the app's own print path sets this, and
    /// an exported file is better off saying it too. Ignored for PNG, whose
    /// encoder here writes no resolution chunk.
    pub dpi: u16,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: ExportFormat::Png16,
            quality: 90,
            max_dim: 0,
            // The conventional JPEG resolution; a 2048 px export is then a
            // 6.8 inch print rather than an arbitrary size.
            dpi: 300,
        }
    }
}

/// Fit `(w, h)` inside a square `max`, keeping the aspect. `max == 0` means no
/// limit.
pub fn fit_within(w: u32, h: u32, max: u32) -> (u32, u32) {
    if max == 0 || w.max(h) <= max {
        return (w, h);
    }
    let scale = max as f32 / w.max(h) as f32;
    (
        ((w as f32 * scale).round() as u32).max(1),
        ((h as f32 * scale).round() as u32).max(1),
    )
}

/// An export whose pixels are already off the GPU, waiting to be written.
///
/// Holds a full-frame 16-bit readback -- about 190 MB for a 24 MP frame -- which
/// is why callers keep at most one of these alive at a time.
pub struct ExportJob {
    path: std::path::PathBuf,
    options: ExportOptions,
    pixels: Vec<u16>,
    width: u32,
    crop: (u32, u32, u32, u32),
}

impl ExportJob {
    /// Crop, scale and encode. Pure CPU and file I/O, safe to run anywhere.
    pub fn write(self) {
        let ExportJob {
            path,
            options,
            pixels,
            width,
            crop,
        } = self;
        let (cw, ch) = (crop.2, crop.3);
        let (ow, oh) = fit_within(cw, ch, options.max_dim);
        // The whole frame is the common case; do not copy it to crop it.
        let height = pixels.len() as u32 / 4 / width.max(1);
        let cropped = if crop == (0, 0, width, height) {
            None
        } else {
            Some(crop_pixels(&pixels, width, crop))
        };
        let source: &[u16] = cropped.as_deref().unwrap_or(&pixels);
        let scaled = if (ow, oh) != (cw, ch) {
            // Through a float buffer, so a big reduction does not band the way
            // integer truncation would.
            Some(resize_u16(source, cw, ch, ow, oh))
        } else {
            None
        };
        let out: &[u16] = scaled.as_deref().unwrap_or(source);
        match options.format {
            ExportFormat::Png16 => {
                let saved = image::ImageBuffer::<image::Rgba<u16>, Vec<u16>>::from_raw(
                    ow,
                    oh,
                    out.to_vec(),
                )
                .map(|img| img.save(&path).is_ok());
                // A half-written file at the target is worse than none: say so
                // by leaving nothing behind.
                if saved != Some(true) {
                    let _ = std::fs::remove_file(&path);
                }
            }
            ExportFormat::Jpeg => {
                // JPEG has no alpha and no room for 16 bits a channel.
                let rgb: Vec<u8> = out
                    .chunks_exact(4)
                    .flat_map(|px| px[..3].iter().map(|v| (u32::from(*v) * 255 / 65535) as u8))
                    .collect();
                let wrote = std::fs::File::create(&path)
                    .ok()
                    .map(|file| {
                        let mut writer = std::io::BufWriter::new(file);
                        let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(
                            &mut writer,
                            options.quality.clamp(1, 100),
                        );
                        if options.dpi > 0 {
                            // Without this the file claims only a pixel aspect
                            // ratio, and its printed size is anyone's guess.
                            encoder.set_pixel_density(image::codecs::jpeg::PixelDensity::dpi(
                                options.dpi,
                            ));
                        }
                        encoder
                            .encode(&rgb, ow, oh, image::ExtendedColorType::Rgb8)
                            .is_ok()
                    });
                if wrote != Some(true) {
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
    }
}

/// Take the crop out of a full-frame RGBA buffer, row by row. Split out of
/// `export` so the pixels that reach a file can be checked without writing one.
fn crop_pixels(pixels: &[u16], width: u32, crop: (u32, u32, u32, u32)) -> Vec<u16> {
    let (x, y, w, h) = crop;
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for row in 0..h {
        let start = (((y + row) * width + x) * 4) as usize;
        let end = start + (w * 4) as usize;
        if end > pixels.len() {
            break;
        }
        out.extend_from_slice(&pixels[start..end]);
    }
    out
}

/// Smallest span a crop may keep, in fractions of the image. The tool never
/// goes near this; it exists so a hand-edited row cannot collapse a crop to
/// nothing.
pub const MIN_CROP_FRACTION: f32 = 0.0005;

/// Pull a loaded crop into a shape the pipeline can draw. The tool only ever
/// writes a sane rect, but the crop travels through a JSON column, and a
/// non-finite value used to reach `f32::clamp`'s assertion and abort the
/// process -- the same hazard retouch dabs are sanitised against.
fn sanitize_crop(crop: &mut [f32; 4]) {
    if crop.iter().any(|v| !v.is_finite()) {
        *crop = FULL_CROP;
        return;
    }
    // A crop the tool could have written is left exactly as it is: the rest of
    // this only exists for values the tool cannot produce.
    let inside = (0.0..=1.0).contains(&crop[0])
        && (0.0..=1.0).contains(&crop[1])
        && crop[2] >= MIN_CROP_FRACTION
        && crop[3] >= MIN_CROP_FRACTION
        && crop[0] + crop[2] <= 1.0
        && crop[1] + crop[3] <= 1.0;
    if inside {
        return;
    }
    let (mut x0, mut y0) = (crop[0].clamp(0.0, 1.0), crop[1].clamp(0.0, 1.0));
    let (mut x1, mut y1) = (
        (crop[0] + crop[2]).clamp(0.0, 1.0),
        (crop[1] + crop[3]).clamp(0.0, 1.0),
    );
    if x1 < x0 {
        std::mem::swap(&mut x0, &mut x1);
    }
    if y1 < y0 {
        std::mem::swap(&mut y0, &mut y1);
    }
    let w = (x1 - x0).clamp(MIN_CROP_FRACTION, 1.0);
    let h = (y1 - y0).clamp(MIN_CROP_FRACTION, 1.0);
    *crop = [
        x0.min(1.0 - w),
        y0.min(1.0 - h),
        w,
        h,
    ];
}

/// The crop as whole pixels `(x, y, w, h)` inside an image of `input` pixels.
///
/// Rounding to whole pixels is what makes a crop exact: the display draws this
/// rect with no resampling and the exporter copies the same pixels, so what
/// gets written out is what the panel showed.
///
/// Total: a non-finite crop or a zero-sized image would otherwise reach
/// `f32::clamp` with its bounds the wrong way round and panic.
pub fn crop_px(crop: [f32; 4], input: (u32, u32)) -> (u32, u32, u32, u32) {
    let crop = if crop.iter().all(|v| v.is_finite()) {
        crop
    } else {
        FULL_CROP
    };
    let (wf, hf) = (input.0.max(1) as f32, input.1.max(1) as f32);
    let x0 = (crop[0] * wf).round().clamp(0.0, wf - 1.0);
    let y0 = (crop[1] * hf).round().clamp(0.0, hf - 1.0);
    let x1 = ((crop[0] + crop[2]) * wf).round().clamp(x0 + 1.0, wf);
    let y1 = ((crop[1] + crop[3]) * hf).round().clamp(y0 + 1.0, hf);
    (x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32)
}

/// The crop covering the whole frame?
pub fn is_full_crop(crop: [f32; 4], input: (u32, u32)) -> bool {
    let (_, _, w, h) = crop_px(crop, input);
    w == input.0 && h == input.1
}

/// The stored form of a whole-pixel crop rect given in pixels: rounded,
/// clamped to the image, and back to fractions.
pub fn crop_from_px(rect: (f32, f32, f32, f32), input: (u32, u32)) -> [f32; 4] {
    let (wf, hf) = (input.0 as f32, input.1 as f32);
    let x0 = rect.0.round().clamp(0.0, wf - 1.0);
    let y0 = rect.1.round().clamp(0.0, hf - 1.0);
    let x1 = rect.2.round().clamp(x0 + 1.0, wf);
    let y1 = rect.3.round().clamp(y0 + 1.0, hf);
    crop_fractions(
        (x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32),
        input,
    )
}

/// Fractions of the input image for a whole-pixel rect, so the stored crop and
/// the pixel rect always agree.
pub fn crop_fractions(px: (u32, u32, u32, u32), input: (u32, u32)) -> [f32; 4] {
    let (wf, hf) = (input.0 as f32, input.1 as f32);
    [
        px.0 as f32 / wf,
        px.1 as f32 / hf,
        px.2 as f32 / wf,
        px.3 as f32 / hf,
    ]
}

pub struct Processor {
    contrast_pipeline: wgpu::ComputePipeline,
    tonal_pipeline: wgpu::ComputePipeline,
    sharpen_pipeline: wgpu::ComputePipeline,
    blur_h_pipeline: wgpu::ComputePipeline,
    blur_v_pipeline: wgpu::ComputePipeline,
    compute_bgl: wgpu::BindGroupLayout,
    retouch_pipeline: wgpu::ComputePipeline,
    retouch_bgl: wgpu::BindGroupLayout,
    display_pipeline: wgpu::ComputePipeline,
    display_bgl: wgpu::BindGroupLayout,
    // True when the device supports 16-bit integer input textures; false falls
    // back to 16-bit half-float input.
    input_u16: bool,

    input_tex: Option<wgpu::Texture>,
    source_tex: Option<wgpu::Texture>,
    source_image: Option<image::RgbaImage>,
    tex1: Option<wgpu::Texture>,            // contrast output
    tex2: Option<wgpu::Texture>,            // tonal output
    tex3: Option<wgpu::Texture>,            // sharpen output
    tex4: Option<wgpu::Texture>,            // blur_h output
    pre_retouch_tex: Option<wgpu::Texture>, // blur_v output, before the retouch layer
    output_tex: Option<wgpu::Texture>,      // retouch output (final, half-float)
    display_tex: Option<wgpu::Texture>,     // 8-bit sRGB copy for the UI

    contrast_buf: wgpu::Buffer,
    tonal_buf: wgpu::Buffer,
    blur_buf: wgpu::Buffer,
    sharpen_buf: wgpu::Buffer,
    curve_buf: wgpu::Buffer,
    retouch_buf: wgpu::Buffer,

    histogram_bgl: wgpu::BindGroupLayout,
    histogram_pipeline: wgpu::ComputePipeline,
    histogram_buf: wgpu::Buffer,
    histogram_staging: wgpu::Buffer,
    /// Origin and size of the region the histogram measures (the crop).
    histogram_window: wgpu::Buffer,
    pub histogram: [u32; 256],

    pub image_size: Option<(u32, u32)>,
    /// All user-editable image parameters — the single source of truth.
    /// The GPU-adjacent fields above plus this state are the whole Processor.
    pub state: EditState,
}

impl std::ops::Deref for Processor {
    type Target = EditState;

    fn deref(&self) -> &EditState {
        &self.state
    }
}

impl std::ops::DerefMut for Processor {
    fn deref_mut(&mut self) -> &mut EditState {
        &mut self.state
    }
}

fn create_compute_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

// Retouch layer: the edit chain's result in, the same result with the healing
// dabs composited on top out, and the dab list as a uniform.
fn create_retouch_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

// Final pass: half-float result -> 8-bit sRGB texture for the UI.
fn create_display_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    })
}

fn create_histogram_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

// Luminance value (0-255) below which fraction p of all pixels fall.
// `total` is the histogram's sum, passed in since callers already have it.
fn percentile(histogram: &[u32; 256], total: f64, p: f64) -> f32 {
    let target = total * p;
    let mut cum = 0.0f64;
    for (bucket, &c) in (0u8..=255).zip(histogram.iter()) {
        cum += f64::from(c);
        if cum >= target {
            return f32::from(bucket);
        }
    }
    255.0
}

// Map a buffer for CPU readback and return its bytes. Synchronous (blocks on
// `device.poll`); all three call sites run on the UI thread where a blocking
// readback is acceptable.
fn map_read(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Option<Vec<u8>> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    if rx.recv().unwrap().is_err() {
        return None;
    }
    let bytes = slice.get_mapped_range().to_vec();
    buffer.unmap();
    Some(bytes)
}

/// Resize 16-bit gamma-encoded RGBA through a float intermediate (used when a
/// replacement image's dimensions differ slightly from the loaded render).
fn resize_u16(pixels: &[u16], w: u32, h: u32, nw: u32, nh: u32) -> Vec<u16> {
    let img = image::Rgba32FImage::from_raw(
        w,
        h,
        pixels.iter().map(|&v| f32::from(v) / 65535.0).collect(),
    )
    .expect("u16 pixel buffer matches dimensions");
    let resized = image::imageops::resize(&img, nw, nh, image::imageops::FilterType::Triangle);
    resized
        .pixels()
        .flat_map(|p| {
            [p[0], p[1], p[2], p[3]].map(|v| (v.clamp(0.0, 1.0) * 65535.0).round() as u16)
        })
        .collect()
}

impl Processor {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders.wgsl").into()),
        });

        let compute_bgl = create_compute_bgl(device);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&compute_bgl],
            push_constant_ranges: &[],
        });

        let make_pipeline = |ep: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: ep,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        let make_buf = |size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let histogram_bgl = create_histogram_bgl(device);
        let display_bgl = create_display_bgl(device);
        let retouch_bgl = create_retouch_bgl(device);
        let retouch_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&retouch_bgl],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader,
            entry_point: "retouch_pass",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let display_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&display_bgl],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader,
            entry_point: "display_pass",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let histogram_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("histogram.wgsl").into()),
        });
        let histogram_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&histogram_bgl],
            push_constant_ranges: &[],
        });
        let histogram_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&histogram_pl),
            module: &histogram_shader,
            entry_point: "histogram_pass",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let histogram_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 1024, // 256 bins × 4 bytes
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let histogram_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 1024,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let histogram_window = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            contrast_pipeline: make_pipeline("contrast_pass"),
            tonal_pipeline: make_pipeline("tonal_pass"),
            sharpen_pipeline: make_pipeline("sharpen_pass"),
            blur_h_pipeline: make_pipeline("blur_h_pass"),
            blur_v_pipeline: make_pipeline("blur_v_pass"),
            compute_bgl,
            retouch_pipeline,
            retouch_bgl,
            display_pipeline,
            display_bgl,
            input_u16: device
                .features()
                .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM),
            input_tex: None,
            source_tex: None,
            source_image: None,
            tex1: None,
            tex2: None,
            tex3: None,
            tex4: None,
            pre_retouch_tex: None,
            output_tex: None,
            display_tex: None,
            contrast_buf: make_buf(48),
            tonal_buf: make_buf(48),
            blur_buf: make_buf(48),
            sharpen_buf: make_buf(48),
            retouch_buf: make_buf(retouch_buffer_bytes()),
            curve_buf: device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: ((256 + 33 * 33 * 33 * 3) * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            histogram_bgl,
            histogram_pipeline,
            histogram_buf,
            histogram_staging,
            histogram_window,
            histogram: [0u32; 256],
            image_size: None,
            state: EditState::default(),
        }
    }

    // Natural cubic spline through the control points, sampled into a 256-entry
    // LUT. Flat extension outside the endpoint x-range, values clamped to [0,1].
    pub fn curve_lut(&self) -> [f32; 256] {
        let pts = &self.curve_points;
        let n_pts = pts.len();
        let mut lut = [0f32; 256];
        if n_pts < 2 {
            for (lut_i, v) in (0u16..).zip(lut.iter_mut()) {
                *v = f32::from(lut_i) / 255.0;
            }
            return lut;
        }

        // Second derivatives with natural boundary (d2[0] = d2[n-1] = 0),
        // solved with the Thomas algorithm.
        let mut d2 = vec![0f32; n_pts];
        if n_pts > 2 {
            let mut diag = vec![1f32; n_pts];
            let mut upper = vec![0f32; n_pts];
            let mut rhs = vec![0f32; n_pts];
            for idx in 1..n_pts - 1 {
                let h0 = (pts[idx][0] - pts[idx - 1][0]).max(1e-4);
                let h1 = (pts[idx + 1][0] - pts[idx][0]).max(1e-4);
                let a_idx = h0;
                diag[idx] = 2.0 * (h0 + h1);
                upper[idx] = h1;
                rhs[idx] = 6.0
                    * ((pts[idx + 1][1] - pts[idx][1]) / h1 - (pts[idx][1] - pts[idx - 1][1]) / h0);
                let fac = a_idx / diag[idx - 1];
                diag[idx] -= fac * upper[idx - 1];
                rhs[idx] -= fac * rhs[idx - 1];
            }
            for idx in (1..n_pts - 1).rev() {
                d2[idx] = (rhs[idx] - upper[idx] * d2[idx + 1]) / diag[idx];
            }
        }

        let mut seg = 0;
        for (lut_i, v) in (0u16..).zip(lut.iter_mut()) {
            let x = f32::from(lut_i) / 255.0;
            let y = if x <= pts[0][0] {
                pts[0][1]
            } else if x >= pts[n_pts - 1][0] {
                pts[n_pts - 1][1]
            } else {
                while seg < n_pts - 2 && pts[seg + 1][0] < x {
                    seg += 1;
                }
                let (x0, y0) = (pts[seg][0], pts[seg][1]);
                let (x1, y1) = (pts[seg + 1][0], pts[seg + 1][1]);
                let h = (x1 - x0).max(1e-4);
                let t0 = x1 - x;
                let t1 = x - x0;
                d2[seg] * t0 * t0 * t0 / (6.0 * h)
                    + d2[seg + 1] * t1 * t1 * t1 / (6.0 * h)
                    + (y0 / h - d2[seg] * h / 6.0) * t0
                    + (y1 / h - d2[seg + 1] * h / 6.0) * t1
            };
            *v = y.clamp(0.0, 1.0);
        }
        lut
    }

    pub fn edit_state(&self) -> EditState {
        self.state.clone()
    }

    pub fn apply_edit_state(&mut self, s: &EditState) {
        self.state = s.clone();
        if self.state.curve_points.len() < 2 {
            self.state.curve_points = vec![[0.0, 0.0], [1.0, 1.0]];
        }
        sanitize_retouch_spots(&mut self.state.retouch_spots);
        sanitize_crop(&mut self.state.crop);
    }

    // Derive auto adjustments from the luminance histogram. Call with the
    // histogram of the UNEDITED image (reset + process first).
    pub fn auto_adjust(&mut self) {
        let total_f: f64 = self.histogram.iter().map(|&c| f64::from(c)).sum();
        if total_f == 0.0 {
            return;
        }
        let histogram = self.histogram;
        let pct = |p: f64| percentile(&histogram, total_f, p);

        // Levels: trim the empty tails, clipping 0.1% of pixels per side
        let mut black = pct(0.001);
        let mut white = pct(0.999);
        if white - black < 32.0 {
            // Degenerate histogram (flat/synthetic image) — clamp gently
            black = black.min(64.0);
            white = white.max(192.0);
        }
        self.levels_black = black.clamp(0.0, 254.0);
        self.levels_white = white.clamp(self.levels_black + 1.0, 255.0);

        // Where key percentiles land AFTER the levels remap
        let remap = |v: f32| ((v - black) / (white - black).max(1.0)).clamp(0.0, 1.0);

        // Brightness: pick the Schlick bias that moves the median to ~0.45
        let m = remap(pct(0.5));
        if m > 0.02 && m < 0.98 {
            let target = 0.45;
            let b = 1.0 / (2.0 + (m / target - 1.0) / (1.0 - m));
            self.brightness = ((b - 0.5) / 0.0025).clamp(-60.0, 60.0);
        }

        // Open shadows if the lower quartile is crushed; recover highlights
        // if the upper quartile crowds the white point
        let q1 = remap(pct(0.25));
        if q1 < 0.20 {
            self.shadows = ((0.20 - q1) * 250.0).clamp(0.0, 50.0);
        }
        let q3 = remap(pct(0.75));
        if q3 > 0.80 {
            self.highlights = (-(q3 - 0.80) * 250.0).clamp(-50.0, 0.0);
        }

        // Contrast: a flat image bunches its quartiles together. Widen the
        // interquartile range toward ~0.32 with the S-curve; never reduce.
        let iqr = (q3 - q1).max(0.01);
        if iqr < 0.32 {
            self.contrast = ((0.32 / iqr - 1.0) * 45.0).clamp(0.0, 70.0);
        }
    }

    pub fn upload_rgba(
        &mut self,
        img: &image::RgbaImage,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let (width, height) = img.dimensions();
        // 8-bit source -> 16-bit input (u8 -> u16 on the way in); the working
        // pipeline below runs in 16-bit from here on.
        let pixels: Vec<u16> = img
            .pixels()
            .flat_map(|p| {
                let widen = |v: u8| u16::from(v) * 257;
                [widen(p[0]), widen(p[1]), widen(p[2]), widen(p[3])]
            })
            .collect();
        self.upload_u16(width, height, &pixels, device, queue);
        self.source_image = Some(img.clone());
    }

    /// Upload 16-bit gamma-encoded RGBA (u16 per channel) as the new image.
    /// RAW development feeds this directly so the sensor's precision survives
    /// into the editor.
    pub fn upload_u16(
        &mut self,
        width: u32,
        height: u32,
        pixels: &[u16],
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let make_input = |format: wgpu::TextureFormat,
                          view_formats: &'static [wgpu::TextureFormat]| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats,
            })
        };
        // 16-bit integer input (65536 levels) when the device supports it,
        // half-float otherwise; 8-bit sRGB original for display.
        let source_tex = make_input(
            wgpu::TextureFormat::Rgba8Unorm,
            &[wgpu::TextureFormat::Rgba8UnormSrgb],
        );
        let input_format = if self.input_u16 {
            wgpu::TextureFormat::Rgba16Unorm
        } else {
            wgpu::TextureFormat::Rgba16Float
        };
        let input_tex = make_input(input_format, &[]);
        // The original is an 8-bit copy of the source bytes; the input carries
        // the full 16-bit data.
        let source_pixels: Vec<u8> = pixels
            .chunks_exact(4)
            .flat_map(|p| {
                // Scale 16-bit -> 8-bit; low-byte truncation would show e.g.
                // 0xC800 (78% white) as black.
                [p[0], p[1], p[2], p[3]].map(|v| (u32::from(v) * 255 / 65535) as u8)
            })
            .collect();
        queue.write_texture(
            source_tex.as_image_copy(),
            &source_pixels,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let input_data: Vec<u16> = if self.input_u16 {
            pixels.to_vec()
        } else {
            pixels
                .iter()
                .map(|&v| f16::from_f32(f32::from(v) / 65535.0).to_bits())
                .collect()
        };
        queue.write_texture(
            input_tex.as_image_copy(),
            bytemuck::cast_slice(&input_data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(8 * width),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let make_intermediate = |extra: wgpu::TextureUsages| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | extra,
                view_formats: &[],
            })
        };

        self.source_tex = Some(source_tex);
        self.input_tex = Some(input_tex);
        self.tex1 = Some(make_intermediate(wgpu::TextureUsages::empty()));
        self.tex2 = Some(make_intermediate(wgpu::TextureUsages::empty()));
        self.tex3 = Some(make_intermediate(wgpu::TextureUsages::empty()));
        self.tex4 = Some(make_intermediate(wgpu::TextureUsages::empty()));
        // COPY_SRC: the dab search reads this texture back (it is the image the
        // retouch pass samples, so the search must see the same pixels).
        self.pre_retouch_tex = Some(make_intermediate(wgpu::TextureUsages::COPY_SRC));
        self.output_tex = Some(make_intermediate(wgpu::TextureUsages::COPY_SRC));
        self.display_tex = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
        }));
        self.image_size = Some((width, height));

        self.process(device, queue);
    }

    /// Replace the effective input with a full-resolution development result;
    /// the untouched source remains available for the original-image view and
    /// for Reset All Edits.  The replacement is decoded from the RAW file
    /// independently, so its dimensions can differ slightly from the editor's
    /// loaded render; it is resized to the input texture instead of being
    /// dropped.
    pub fn replace_input_rgba(&mut self, img: &image::RgbaImage, queue: &wgpu::Queue) -> bool {
        let (width, height) = img.dimensions();
        let pixels: Vec<u16> = img
            .pixels()
            .flat_map(|p| {
                let widen = |v: u8| u16::from(v) * 257;
                [widen(p[0]), widen(p[1]), widen(p[2]), widen(p[3])]
            })
            .collect();
        self.replace_input_u16(width, height, &pixels, queue)
    }

    /// Replace the effective input with 16-bit gamma-encoded RGBA pixels.
    pub fn replace_input_u16(
        &mut self,
        width: u32,
        height: u32,
        pixels: &[u16],
        queue: &wgpu::Queue,
    ) -> bool {
        let Some(input) = self.input_tex.as_ref() else {
            return false;
        };
        let Some((tw, th)) = self.image_size else {
            return false;
        };
        let pixels = if (width, height) == (tw, th) {
            pixels.to_vec()
        } else {
            resize_u16(pixels, width, height, tw, th)
        };
        let input_data: Vec<u16> = if self.input_u16 {
            pixels
        } else {
            pixels
                .iter()
                .map(|&v| f16::from_f32(f32::from(v) / 65535.0).to_bits())
                .collect()
        };
        queue.write_texture(
            input.as_image_copy(),
            bytemuck::cast_slice(&input_data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(8 * tw),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: tw,
                height: th,
                depth_or_array_layers: 1,
            },
        );
        true
    }

    /// Restore the original sensor render after Reset All Edits.
    pub fn restore_source(&mut self, queue: &wgpu::Queue) -> bool {
        let Some(source) = self.source_image.clone() else {
            return false;
        };
        self.replace_input_rgba(&source, queue)
    }
    pub fn output_view(&self) -> Option<wgpu::TextureView> {
        self.display_tex.as_ref().map(|t| {
            t.create_view(&wgpu::TextureViewDescriptor {
                format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
                ..Default::default()
            })
        })
    }

    pub fn input_view(&self) -> Option<wgpu::TextureView> {
        self.source_tex.as_ref().map(|t| {
            t.create_view(&wgpu::TextureViewDescriptor {
                format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
                ..Default::default()
            })
        })
    }

    pub fn has_image(&self) -> bool {
        self.input_tex.is_some()
    }

    // Pipeline: contrast → sharpen → blur_h → blur_v → display
    // Sharpen operates on the contrast-adjusted image (pre-blur) so the unsharp
    fn write_uniforms(&self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.contrast_buf,
            0,
            bytemuck::cast_slice(&[
                self.contrast,
                self.levels_black,
                self.levels_white,
                self.levels_gamma,
                self.exposure,
                self.wb_temp,
                self.wb_tint,
                if photo_lut_enabled(&self.state) {
                    1.0
                } else {
                    0.0
                },
            ]),
        );
        queue.write_buffer(
            &self.tonal_buf,
            0,
            bytemuck::cast_slice(&[
                self.blacks,
                self.shadows,
                self.highlights,
                self.whites,
                self.brightness,
                self.vignette,
                self.vignette_mid,
                self.saturation,
                self.vibrance,
                0f32,
                0f32,
            ]),
        );
        queue.write_buffer(
            &self.blur_buf,
            0,
            bytemuck::cast_slice(&[self.blur_radius, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32]),
        );
        queue.write_buffer(
            &self.sharpen_buf,
            0,
            bytemuck::cast_slice(&[
                self.unsharp_strength,
                self.unsharp_blur_radius,
                0f32,
                0f32,
                0f32,
                0f32,
                0f32,
                0f32,
            ]),
        );
        let mut data = Vec::with_capacity(256 + 33 * 33 * 33 * 3);
        data.extend(self.curve_lut());
        data.extend(baked_lut(&self.state).unwrap_or_else(identity_photo_lut));
        queue.write_buffer(&self.curve_buf, 0, bytemuck::cast_slice(&data));
    }

    fn read_histogram(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        hist_view: &wgpu::TextureView,
        window: (u32, u32, u32, u32),
    ) {
        let (x, y, w, h) = window;
        queue.write_buffer(&self.histogram_buf, 0, &[0u8; 1024]);
        queue.write_buffer(
            &self.histogram_window,
            0,
            bytemuck::cast_slice(&[x, y, w, h]),
        );
        let hist_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.histogram_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(hist_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.histogram_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.histogram_window.as_entire_binding(),
                },
            ],
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.histogram_pipeline);
            pass.set_bind_group(0, &hist_bg, &[]);
            pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
        }
        enc.copy_buffer_to_buffer(&self.histogram_buf, 0, &self.histogram_staging, 0, 1024);
        queue.submit([enc.finish()]);

        if let Some(data) = map_read(device, &self.histogram_staging) {
            self.histogram.copy_from_slice(bytemuck::cast_slice(&data));
        }
    }

    // Write the retouch dab list into the uniform the shader reads. With no
    // dabs the pass is skipped entirely (see `process`), so only the count has
    // to be reset -- no per-frame allocation.
    fn write_retouch(&self, queue: &wgpu::Queue) {
        if self.retouch_spots.is_empty() {
            queue.write_buffer(&self.retouch_buf, 0, bytemuck::cast_slice(&[0f32]));
            return;
        }
        let data = pack_retouch_spots(&self.retouch_spots);
        queue.write_buffer(&self.retouch_buf, 0, bytemuck::cast_slice(&data));
    }

    // mask anchors off clean signal, independent of the box-blur slider.
    pub fn process(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let (
            Some(input),
            Some(t1),
            Some(t2),
            Some(t3),
            Some(t4),
            Some(pre),
            Some(output),
            Some(display),
        ) = (
            self.input_tex.as_ref(),
            self.tex1.as_ref(),
            self.tex2.as_ref(),
            self.tex3.as_ref(),
            self.tex4.as_ref(),
            self.pre_retouch_tex.as_ref(),
            self.output_tex.as_ref(),
            self.display_tex.as_ref(),
        ) else {
            return;
        };

        self.write_uniforms(queue);
        self.write_retouch(queue);

        let iv = input.create_view(&wgpu::TextureViewDescriptor::default());
        let t1v = t1.create_view(&wgpu::TextureViewDescriptor::default());
        let t2v = t2.create_view(&wgpu::TextureViewDescriptor::default());
        let t3v = t3.create_view(&wgpu::TextureViewDescriptor::default());
        let t4v = t4.create_view(&wgpu::TextureViewDescriptor::default());
        let pv = pre.create_view(&wgpu::TextureViewDescriptor::default());
        let ov = output.create_view(&wgpu::TextureViewDescriptor::default());
        let dv = display.create_view(&wgpu::TextureViewDescriptor::default());

        let bgl = &self.compute_bgl;
        let curve_buf = &self.curve_buf;
        let make_bg =
            |in_view: &wgpu::TextureView, out_view: &wgpu::TextureView, buf: &wgpu::Buffer| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(in_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(out_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: curve_buf.as_entire_binding(),
                        },
                    ],
                })
            };

        let contrast_bg = make_bg(&iv, &t1v, &self.contrast_buf);
        let tonal_bg = make_bg(&t1v, &t2v, &self.tonal_buf);
        let sharpen_bg = make_bg(&t2v, &t3v, &self.sharpen_buf);
        let blur_h_bg = make_bg(&t3v, &t4v, &self.blur_buf);
        let blur_vert_bg = make_bg(&t4v, &pv, &self.blur_buf);
        // With no dabs there is nothing to composite, so the display reads the
        // pre-retouch texture directly and the pass is skipped.
        let retouching = !self.retouch_spots.is_empty();
        let retouch_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.retouch_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&pv),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&ov),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.retouch_buf.as_entire_binding(),
                },
            ],
        });
        let display_src = if retouching { &ov } else { &pv };
        let display_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.display_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(display_src),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&dv),
                },
            ],
        });

        let (w, h) = self.image_size.unwrap();
        let wg = (w.div_ceil(8), h.div_ceil(8));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let mut dispatch = |pipeline: &wgpu::ComputePipeline, bg: &wgpu::BindGroup| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(wg.0, wg.1, 1);
        };
        dispatch(&self.contrast_pipeline, &contrast_bg);
        dispatch(&self.tonal_pipeline, &tonal_bg);
        dispatch(&self.sharpen_pipeline, &sharpen_bg);
        dispatch(&self.blur_h_pipeline, &blur_h_bg);
        dispatch(&self.blur_v_pipeline, &blur_vert_bg);
        if retouching {
            dispatch(&self.retouch_pipeline, &retouch_bg);
        }
        dispatch(&self.display_pipeline, &display_bg);
        queue.submit([encoder.finish()]);

        // Separate submission: display write must be visible before histogram reads.
        // The histogram describes the crop, not the whole sensor frame.
        self.read_histogram(device, queue, &dv, crop_px(self.crop, (w, h)));
    }

    /// The pixels the retouch pass will sample from -- the image after every
    /// edit but *before* the retouch layer -- cropped to `[x, y, w, h)`.
    ///
    /// The dab search must read exactly this: reading the finished display
    /// instead would let it score candidates against pixels the render never
    /// samples (a dab drawn over an earlier one, for instance).
    ///
    /// Only the region a dab can reach is copied: the dab itself plus the
    /// farthest source the search looks at, which keeps a click on a 24 MP
    /// frame from reading the whole image back.
    pub fn retouch_source_pixels(
        &self,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Option<Vec<u8>> {
        let (tex, (full_w, full_h)) = (self.pre_retouch_tex.as_ref()?, self.image_size?);
        if w == 0 || h == 0 || x + w > full_w || y + h > full_h {
            return None;
        }
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let bytes_per_row = (w * 8).div_ceil(align) * align;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(bytes_per_row * h),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &staging,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);

        let data = map_read(device, &staging)?;
        // Half-float RGBA -> 8-bit RGBA. The search works on rendered values
        // (0..1), which is what this texture holds; 8 bits is plenty for a
        // coarse patch match.
        let mut pixels = Vec::with_capacity((w * h * 4) as usize);
        for row in 0..h {
            let start = (row * bytes_per_row) as usize;
            let row_bytes = &data[start..start + (w * 8) as usize];
            for channel in row_bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([channel[0], channel[1]]);
                let value = f16::from_bits(bits).to_f32().clamp(0.0, 1.0);
                pixels.push((value * 255.0).round() as u8);
            }
        }
        Some(pixels)
    }

    // Read the rendered output back to CPU RGBA8 bytes, synchronously. Used
    // where the caller needs the *actual* rendered pixels rather than the
    // edit sliders — e.g. capturing a look, since an AI-adjusted photo keeps
    // its whole look in the LUT, not in any slider value.  Reads the 8-bit
    // display copy (what the user sees).
    pub fn output_pixels(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Vec<u8>> {
        let (display, (width, height)) = (self.display_tex.as_ref()?, self.image_size?);

        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let bytes_per_row = (width * 4).div_ceil(align) * align;

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(bytes_per_row * height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            display.as_image_copy(),
            wgpu::ImageCopyBuffer {
                buffer: &staging,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);

        let data = map_read(device, &staging)?;
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for row in 0..height {
            let start = (row * bytes_per_row) as usize;
            pixels.extend_from_slice(&data[start..start + (width * 4) as usize]);
        }
        Some(pixels)
    }

    /// The texture holding the finished image: the retouch output when there
    /// are dabs, the pre-retouch result when that pass is skipped.
    fn final_tex(&self) -> Option<&wgpu::Texture> {
        if self.retouch_spots.is_empty() {
            self.pre_retouch_tex.as_ref()
        } else {
            self.output_tex.as_ref()
        }
    }

    // Export the half-float result as a 16-bit sRGB PNG (gamma-encoded u16).
    /// The size this export will be written at: after the crop, and after any
    /// size limit.
    pub fn export_size(&self, options: &ExportOptions) -> Option<(u32, u32)> {
        let (width, height) = self.image_size?;
        let crop = crop_px(self.crop, (width, height));
        Some(fit_within(crop.2, crop.3, options.max_dim))
    }

    /// Ready an export: everything that needs the GPU is done here, so writing
    /// it can happen anywhere afterwards -- on a thread, or one at a time.
    ///
    /// Split out from [`Processor::export`] so a folder export can hold exactly
    /// one in flight: each job carries a full-frame 16-bit readback, which is
    /// far too big to stack up.
    pub fn export_job(
        &self,
        path: &Path,
        options: &ExportOptions,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Option<ExportJob> {
        let (Some(output), Some((width, height))) = (self.final_tex(), self.image_size) else {
            return None;
        };

        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let bytes_per_row = (width * 8).div_ceil(align) * align;

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(bytes_per_row * height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            output.as_image_copy(),
            wgpu::ImageCopyBuffer {
                buffer: &staging,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);

        let data = map_read(device, &staging)?;

        // Half-float RGBA -> gamma-encoded u16 RGBA (values are 0..1).
        let pixels: Vec<u16> = {
            let mut out = Vec::with_capacity((width * height * 4) as usize);
            for row in 0..height {
                let start = (row * bytes_per_row) as usize;
                let row_bytes = &data[start..start + (width * 8) as usize];
                for channel in row_bytes.chunks_exact(2) {
                    let bits = u16::from_le_bytes([channel[0], channel[1]]);
                    let value = f16::from_bits(bits).to_f32();
                    out.push((value.clamp(0.0, 1.0) * 65535.0).round() as u16);
                }
            }
            out
        };

        // File I/O on a background thread so the main thread is unblocked
        let path = path.to_owned();
        // Only the crop is written out; otherwise the file would not match what
        // the panel shows. The rect is the same whole-pixel one used there, so
        // no resampling is involved.
        Some(ExportJob {
            path: path.to_path_buf(),
            options: *options,
            pixels,
            width,
            crop: crop_px(self.crop, (width, height)),
        })
    }

    /// Export one image, on a thread of its own. A single export is small
    /// enough for that; a folder of them goes through [`ExportJob`] instead.
    pub fn export(
        &self,
        path: &Path,
        options: &ExportOptions,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        if let Some(job) = self.export_job(path, options, device, queue) {
            std::thread::spawn(move || job.write());
        }
    }
}

// The normalized RGB coordinate voxel `i` of an `n`-cube grid sits at, laid
// out R-fastest -- the one indexing scheme every LUT in this app shares,
// whether it's being built, resampled, or walked for a post-process pass.
pub(crate) fn voxel_rgb(i: usize, n: usize) -> [f32; 3] {
    [
        (i % n) as f32 / (n - 1) as f32,
        ((i / n) % n) as f32 / (n - 1) as f32,
        (i / (n * n)) as f32 / (n - 1) as f32,
    ]
}

pub fn identity_photo_lut() -> Vec<f32> {
    const N: usize = 33;
    (0..N * N * N).flat_map(|i| voxel_rgb(i, N)).collect()
}

pub fn sample_cube(lut: &[f32], n: usize, rgb: [f32; 3]) -> [f32; 3] {
    let max_idx = (n - 1) as f32;
    let p = rgb.map(|v| v.clamp(0.0, 1.0) * max_idx);
    let lo = p.map(|v| v.floor() as usize);
    let hi = lo.map(|v| (v + 1).min(n - 1));
    let d = [
        p[0] - lo[0] as f32,
        p[1] - lo[1] as f32,
        p[2] - lo[2] as f32,
    ];
    let cell =
        |r: usize, g: usize, b: usize, channel: usize| lut[(r + g * n + b * n * n) * 3 + channel];
    let mut out = [0.0f32; 3];
    for (channel, slot) in out.iter_mut().enumerate() {
        let mix = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let c00 = mix(
            cell(lo[0], lo[1], lo[2], channel),
            cell(hi[0], lo[1], lo[2], channel),
            d[0],
        );
        let c10 = mix(
            cell(lo[0], hi[1], lo[2], channel),
            cell(hi[0], hi[1], lo[2], channel),
            d[0],
        );
        let c01 = mix(
            cell(lo[0], lo[1], hi[2], channel),
            cell(hi[0], lo[1], hi[2], channel),
            d[0],
        );
        let c11 = mix(
            cell(lo[0], hi[1], hi[2], channel),
            cell(hi[0], hi[1], hi[2], channel),
            d[0],
        );
        *slot = mix(mix(c00, c10, d[1]), mix(c01, c11, d[1]), d[2]);
    }
    out
}

fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}
fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}
fn linear_to_oklab(rgb: [f32; 3]) -> [f32; 3] {
    let l = 0.41222147 * rgb[0] + 0.53633255 * rgb[1] + 0.051445995 * rgb[2];
    let m = 0.2119035 * rgb[0] + 0.6806995 * rgb[1] + 0.10739696 * rgb[2];
    let s = 0.08830246 * rgb[0] + 0.28171885 * rgb[1] + 0.6299787 * rgb[2];
    let (l, m, s) = (l.cbrt(), m.cbrt(), s.cbrt());
    [
        0.21045426 * l + 0.7936178 * m - 0.004072047 * s,
        1.9779985 * l - 2.4285922 * m + 0.4505937 * s,
        0.025904037 * l + 0.78277177 * m - 0.80867577 * s,
    ]
}
fn oklab_to_linear(lab: [f32; 3]) -> [f32; 3] {
    let l = lab[0] + 0.39633778 * lab[1] + 0.21580376 * lab[2];
    let m = lab[0] - 0.105561346 * lab[1] - 0.06385417 * lab[2];
    let s = lab[0] - 0.08948418 * lab[1] - 1.2914855 * lab[2];
    let (l, m, s) = (l * l * l, m * m * m, s * s * s);
    [
        4.0767417 * l - 3.3077116 * m + 0.23096994 * s,
        -1.268438 * l + 2.6097574 * m - 0.34131938 * s,
        -0.0041960863 * l - 0.7034186 * m + 1.7076147 * s,
    ]
}
fn oklab_to_srgb_in_gamut(lab: [f32; 3]) -> [f32; 3] {
    let lt = lab[0].clamp(0.0, 1.0);
    let at = |s: f32| oklab_to_linear([lt, lab[1] * s, lab[2] * s]);
    let fits = |rgb: &[f32; 3]| rgb.iter().all(|v| *v >= -1e-4 && *v <= 1.0001);
    let mut scale = 1.0;
    if !fits(&at(1.0)) {
        let (mut lo, mut hi) = (0.0f32, 1.0f32);
        for _ in 0..18 {
            let mid = 0.5 * (lo + hi);
            if fits(&at(mid)) {
                lo = mid
            } else {
                hi = mid
            }
        }
        scale = lo;
    }
    at(scale).map(|v| linear_to_srgb(v.clamp(0.0, 1.0)).clamp(0.0, 1.0))
}
pub const REGION_COUNT: usize = 3;

struct RegionPrior {
    hue: (f32, f32),
    chroma: (f32, f32),
    lightness: (f32, f32),
}

const REGION_PRIORS: [RegionPrior; REGION_COUNT] = [
    RegionPrior {
        hue: (10.0, 85.0),
        chroma: (0.02, 0.20),
        lightness: (0.30, 0.98),
    },
    RegionPrior {
        hue: (95.0, 175.0),
        chroma: (0.03, 0.40),
        lightness: (0.15, 0.95),
    },
    RegionPrior {
        hue: (200.0, 285.0),
        chroma: (0.02, 0.40),
        lightness: (0.45, 1.00),
    },
];

#[derive(Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct RegionTone {
    pub lightness: f32,
    pub chroma: f32,
    pub hue_axis: [f32; 2],
    pub share: f32,
}

const REGION_SHARE_FLOOR: [f32; REGION_COUNT] = [0.002, 0.02, 0.02];

fn in_prior(prior: &RegionPrior, lightness: f32, chroma: f32, hue_degrees: f32) -> bool {
    hue_degrees >= prior.hue.0
        && hue_degrees <= prior.hue.1
        && chroma >= prior.chroma.0
        && chroma <= prior.chroma.1
        && lightness >= prior.lightness.0
        && lightness <= prior.lightness.1
}

pub fn measure_regions(
    pixels: &[u8],
    width: u32,
    height: u32,
    faces: &[[f32; 4]],
) -> [RegionTone; REGION_COUNT] {
    let mut sum_lightness = [0.0f64; REGION_COUNT];
    let mut sum_chroma = [0.0f64; REGION_COUNT];
    let mut sum_axis = [[0.0f64; 2]; REGION_COUNT];
    let mut counted = [0.0f64; REGION_COUNT];

    #[derive(Clone, Copy)]
    struct SkinEllipse {
        cx: f32,
        cy: f32,
        rx: f32,
        ry: f32,
    }
    let skin_ellipses: Vec<SkinEllipse> = faces
        .iter()
        .filter(|b| {
            b[2].abs() >= 0.03 && b[2].abs() <= 0.80 && b[3].abs() >= 0.03 && b[3].abs() <= 0.80
        })
        .map(|b| SkinEllipse {
            cx: b[0] + b[2] * 0.5,
            cy: b[1] + b[3] * 0.45,
            rx: b[2] * 0.20,
            ry: b[3] * 0.22,
        })
        .collect();
    let has_faces = !skin_ellipses.is_empty();
    let mut skin_box = [0.0f64; 5];
    let mut skin_global = [0.0f64; 5];

    let total = f64::from(width) * f64::from(height);
    if total == 0.0 || pixels.len() < (width * height * 4) as usize {
        return [RegionTone::default(); REGION_COUNT];
    }

    for (index, px) in pixels.chunks_exact(4).enumerate() {
        let x = (index as u32 % width) as f32 / width as f32;
        let y = (index as u32 / width) as f32 / height as f32;
        let lab = linear_to_oklab([
            srgb_to_linear(f32::from(px[0]) / 255.0),
            srgb_to_linear(f32::from(px[1]) / 255.0),
            srgb_to_linear(f32::from(px[2]) / 255.0),
        ]);
        let chroma = (lab[1] * lab[1] + lab[2] * lab[2]).sqrt();
        if chroma <= 1e-5 {
            continue;
        }
        let hue = lab[2].atan2(lab[1]).to_degrees().rem_euclid(360.0);
        let passes_skin_prior = in_prior(&REGION_PRIORS[0], lab[0], chroma, hue);

        for region in 0..REGION_COUNT {
            let allowed = match region {
                0 => passes_skin_prior,
                2 => y < 0.5,
                _ => true,
            };
            if !allowed {
                continue;
            }
            if region != 0 && !in_prior(&REGION_PRIORS[region], lab[0], chroma, hue) {
                continue;
            }

            if region == 0 {
                skin_global[0] += f64::from(lab[0]);
                skin_global[1] += f64::from(chroma);
                skin_global[2] += f64::from(lab[1] / chroma);
                skin_global[3] += f64::from(lab[2] / chroma);
                skin_global[4] += 1.0;
                if has_faces
                    && skin_ellipses.iter().any(|e| {
                        let dx = (x - e.cx) / e.rx;
                        let dy = (y - e.cy) / e.ry;
                        dx * dx + dy * dy <= 1.0
                    })
                {
                    skin_box[0] += f64::from(lab[0]);
                    skin_box[1] += f64::from(chroma);
                    skin_box[2] += f64::from(lab[1] / chroma);
                    skin_box[3] += f64::from(lab[2] / chroma);
                    skin_box[4] += 1.0;
                }
            } else {
                sum_lightness[region] += f64::from(lab[0]);
                sum_chroma[region] += f64::from(chroma);
                sum_axis[region][0] += f64::from(lab[1] / chroma);
                sum_axis[region][1] += f64::from(lab[2] / chroma);
                counted[region] += 1.0;
            }
        }
    }

    let box_share = skin_box[4] / total;
    let use_box = has_faces && box_share >= REGION_SHARE_FLOOR[0] as f64;
    let skin_stats = if use_box { skin_box } else { skin_global };
    let skin_share = skin_stats[4] / total;
    let floor = if !has_faces {
        REGION_SHARE_FLOOR[0] * 0.1
    } else {
        REGION_SHARE_FLOOR[0]
    };

    let mut tones = [RegionTone::default(); REGION_COUNT];
    if skin_stats[4] > 0.0 && skin_share as f32 >= floor {
        let length = (skin_stats[2].powi(2) + skin_stats[3].powi(2))
            .sqrt()
            .max(1e-9);
        tones[0] = RegionTone {
            lightness: (skin_stats[0] / skin_stats[4]) as f32,
            chroma: (skin_stats[1] / skin_stats[4]) as f32,
            hue_axis: [
                (skin_stats[2] / length) as f32,
                (skin_stats[3] / length) as f32,
            ],
            share: skin_share as f32,
        };
    }
    for region in 1..REGION_COUNT {
        if counted[region] == 0.0 {
            continue;
        }
        let share = (counted[region] / total) as f32;
        if share < REGION_SHARE_FLOOR[region] {
            continue;
        }
        let length = (sum_axis[region][0].powi(2) + sum_axis[region][1].powi(2))
            .sqrt()
            .max(1e-9);
        tones[region] = RegionTone {
            lightness: (sum_lightness[region] / counted[region]) as f32,
            chroma: (sum_chroma[region] / counted[region]) as f32,
            hue_axis: [
                (sum_axis[region][0] / length) as f32,
                (sum_axis[region][1] / length) as f32,
            ],
            share,
        };
    }
    tones
}

// Captured-look statistics used by the constrained model.
//
// Deliberately *shape*-based, not absolute-level: `tone` records the image's
// own lightness at fixed percentiles of its own histogram (not fixed pixel
// values), so two photos shot under different exposure/light still compare
// on their relative contrast/shape rather than their absolute brightness --
// copying absolute values washed results out or blew them out depending on
// which photo was brighter to start with.
//
// Percentiles of the Oklab lightness histogram sampled as tone anchors,
// splined between at bake time.
pub const LOOK_ANCHORS: [f64; 5] = [0.05, 0.25, 0.5, 0.75, 0.95];
const HUE_SECTORS: usize = 8;

#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LookProfile {
    pub tone: [f32; LOOK_ANCHORS.len()],
    // Average Oklab (a,b) per tone band (shadows/mids/highlights).
    pub cast: [[f32; 2]; 3],
    pub cast_evidence: [f32; 3],
    pub chroma: f32,
    pub hue_chroma: [f32; HUE_SECTORS],
    pub hue_axis: [[f32; 2]; HUE_SECTORS],
    pub hue_evidence: [f32; HUE_SECTORS],
    pub regions: [RegionTone; REGION_COUNT],
}

impl LookProfile {
    pub fn measure(pixels: &[u8], width: u32, height: u32, faces: &[[f32; 4]]) -> Option<Self> {
        let total = (width as usize) * (height as usize);
        if total == 0 || pixels.len() < total * 4 {
            return None;
        }

        let mut ls = Vec::with_capacity(total);
        let mut cast_sum = [[0f64; 2]; 3];
        let mut cast_count = [0f64; 3];
        let mut chroma_sum = 0f64;
        let mut chroma_count = 0u32;
        let mut hue_chroma_sum = [0f64; HUE_SECTORS];
        let mut hue_axis_sum = [[0f64; 2]; HUE_SECTORS];
        let mut hue_count = [0u32; HUE_SECTORS];

        for px in pixels.chunks_exact(4) {
            let lab = linear_to_oklab([
                srgb_to_linear(f32::from(px[0]) / 255.0),
                srgb_to_linear(f32::from(px[1]) / 255.0),
                srgb_to_linear(f32::from(px[2]) / 255.0),
            ]);
            ls.push(lab[0]);
            let band = tone_band(lab[0]);
            let chroma = (lab[1] * lab[1] + lab[2] * lab[2]).sqrt();

            // `cast` is meant to capture a systematic tint on this tone
            // band's *neutral* tones (a white-balance/split-tone style grade
            // a colorist applied on purpose) -- not the average color of
            // everything at that lightness. Weighted by raw pixel count, a
            // large saturated object (a field of grass, a red wall) that
            // happens to sit in this band outvotes the actual neutral tones
            // by sheer color intensity, and its own object color gets pasted
            // as a uniform cast across every hue in the band on a target
            // photo that may not contain that object at all -- the reported
            // symptom is exactly this: transferring from a grassy reference
            // tints a grass-free target green all over. Down-weighting each
            // pixel's contribution by how far it already is from neutral
            // keeps genuinely near-neutral tones in charge of the average.
            const CAST_NEUTRAL_RADIUS: f32 = 0.05;
            let cast_weight = 1.0 / (1.0 + (chroma / CAST_NEUTRAL_RADIUS).powi(2));
            cast_sum[band][0] += f64::from(lab[1] * cast_weight);
            cast_sum[band][1] += f64::from(lab[2] * cast_weight);
            cast_count[band] += f64::from(cast_weight);

            if chroma <= 1e-4 {
                continue;
            }
            chroma_sum += f64::from(chroma);
            chroma_count += 1;
            let hue = lab[2].atan2(lab[1]).to_degrees().rem_euclid(360.0);
            let sector = hue_sector(hue);
            hue_chroma_sum[sector] += f64::from(chroma);
            hue_axis_sum[sector][0] += f64::from(lab[1] / chroma);
            hue_axis_sum[sector][1] += f64::from(lab[2] / chroma);
            hue_count[sector] += 1;
        }

        ls.sort_by(|a, b| a.total_cmp(b));
        let mut tone = [0f32; LOOK_ANCHORS.len()];
        for (slot, p) in tone.iter_mut().zip(LOOK_ANCHORS) {
            let idx = ((p * (ls.len() - 1) as f64).round() as usize).min(ls.len() - 1);
            *slot = ls[idx];
        }

        let mut cast = [[0f32; 2]; 3];
        let mut cast_evidence = [0f32; 3];
        for band in 0..3 {
            if cast_count[band] > 1e-6 {
                cast[band] = [
                    (cast_sum[band][0] / cast_count[band]) as f32,
                    (cast_sum[band][1] / cast_count[band]) as f32,
                ];
            }
            // Weighted by the same neutral-tone weighting as `cast_sum` --
            // low if this band's evidence is mostly saturated object color
            // rather than reliably-neutral content, so the model can treat
            // that measurement as uncertain.
            cast_evidence[band] = (cast_count[band] / total as f64) as f32;
        }

        let chroma = if chroma_count > 0 {
            (chroma_sum / f64::from(chroma_count)) as f32
        } else {
            0.0
        };

        let mut hue_chroma = [0f32; HUE_SECTORS];
        let mut hue_axis = [[0f32; 2]; HUE_SECTORS];
        let mut hue_evidence = [0f32; HUE_SECTORS];
        for s in 0..HUE_SECTORS {
            if hue_count[s] == 0 {
                continue;
            }
            hue_chroma[s] = (hue_chroma_sum[s] / f64::from(hue_count[s])) as f32;
            let len = (hue_axis_sum[s][0].powi(2) + hue_axis_sum[s][1].powi(2))
                .sqrt()
                .max(1e-9);
            hue_axis[s] = [
                (hue_axis_sum[s][0] / len) as f32,
                (hue_axis_sum[s][1] / len) as f32,
            ];
            hue_evidence[s] = hue_count[s] as f32 / total as f32;
        }

        let regions = measure_regions(pixels, width, height, faces);

        Some(Self {
            tone,
            cast,
            cast_evidence,
            chroma,
            hue_chroma,
            hue_axis,
            hue_evidence,
            regions,
        })
    }
}

// The constrained, display-safe transform predicted by the look model.
#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LookTransfer {
    pub(crate) tone_delta: [f32; LOOK_ANCHORS.len()],
    pub(crate) cast_delta: [[f32; 2]; 3],
    pub(crate) hue_chroma_scale: [f32; HUE_SECTORS],
    // Degrees, applied as a rotation of the (a,b) chroma-plane vector rather
    // than an additive offset: a rotation of a near-zero vector stays near
    // zero, so a near-neutral pixel (grey pavement, an overcast sky) is left
    // alone. An earlier version of this field added a fixed (a,b) offset
    // instead, which shifted low-chroma neutrals by the same absolute amount
    // as fully-saturated colors -- grey stone came out visibly purple.
    pub(crate) hue_rotate: [f32; HUE_SECTORS],
}

fn tone_band(l: f32) -> usize {
    if l < 1.0 / 3.0 {
        0
    } else if l < 2.0 / 3.0 {
        1
    } else {
        2
    }
}
fn hue_sector(hue_degrees: f32) -> usize {
    ((hue_degrees / (360.0 / HUE_SECTORS as f32)) as usize).min(HUE_SECTORS - 1)
}

// Smoothly blend between the two hue sectors nearest `hue_degrees`, rather
// than snapping to whichever one it falls in -- `values[s]` is treated as the
// measurement at sector `s`'s *center*, so a pixel exactly on a 45-degree
// sector boundary reads an even 50/50 mix of its neighbors instead of jumping.
// A hard per-sector lookup here measurably reintroduced the same kind of
// discontinuity `hue_rotate` (see its doc comment) was built to avoid: walking
// a smooth skin-tone gradient through a sector boundary produced a real,
// non-monotonic step in the output color -- not an 8-bit rounding artifact,
// a genuine jump in the baked LUT that read as blotching on skin.
fn interp_hue_sectors(hue_degrees: f32, values: &[f32; HUE_SECTORS]) -> f32 {
    let width = 360.0 / HUE_SECTORS as f32;
    let pos = hue_degrees / width - 0.5;
    let i0f = pos.floor();
    let t = pos - i0f;
    let idx0 = (i0f as i64).rem_euclid(HUE_SECTORS as i64) as usize;
    let idx1 = (idx0 + 1) % HUE_SECTORS;
    values[idx0] + (values[idx1] - values[idx0]) * t
}

// Same idea as `interp_hue_sectors` but for the 3 shadow/mid/highlight cast
// bands, which aren't cyclic -- band `i`'s value is anchored at its center and
// clamped (not wrapped) past the first/last band.
fn interp_tone_band(l: f32, values: &[[f32; 2]; 3]) -> [f32; 2] {
    const CENTERS: [f32; 3] = [1.0 / 6.0, 0.5, 5.0 / 6.0];
    if l <= CENTERS[0] {
        return values[0];
    }
    if l >= CENTERS[2] {
        return values[2];
    }
    for i in 0..2 {
        if l <= CENTERS[i + 1] {
            let t = (l - CENTERS[i]) / (CENTERS[i + 1] - CENTERS[i]);
            return [
                values[i][0] + (values[i + 1][0] - values[i][0]) * t,
                values[i][1] + (values[i + 1][1] - values[i][1]) * t,
            ];
        }
    }
    values[2]
}

fn spline_tone_delta(l: f32, tone_delta: &[f32; LOOK_ANCHORS.len()]) -> f32 {
    let last = LOOK_ANCHORS.len() - 1;
    if l <= LOOK_ANCHORS[0] as f32 {
        return tone_delta[0];
    }
    if l >= LOOK_ANCHORS[last] as f32 {
        return tone_delta[last];
    }
    for i in 0..last {
        let (x0, x1) = (LOOK_ANCHORS[i] as f32, LOOK_ANCHORS[i + 1] as f32);
        if l <= x1 {
            let t = (l - x0) / (x1 - x0).max(1e-6);
            return tone_delta[i] + (tone_delta[i + 1] - tone_delta[i]) * t;
        }
    }
    tone_delta[last]
}

fn apply_transfer_to_lab(lab: [f32; 3], t: &LookTransfer) -> [f32; 3] {
    let l = lab[0].clamp(0.0, 1.0);
    let new_l = (l + spline_tone_delta(l, &t.tone_delta)).clamp(0.0, 1.0);
    let chroma = (lab[1] * lab[1] + lab[2] * lab[2]).sqrt();
    if chroma <= 1e-5 {
        return [new_l, lab[1], lab[2]];
    }
    let hue = lab[2].atan2(lab[1]).to_degrees().rem_euclid(360.0);
    let cast = interp_tone_band(l, &t.cast_delta);
    let scale = interp_hue_sectors(hue, &t.hue_chroma_scale);
    let a0 = lab[1] + cast[0];
    let b0 = lab[2] + cast[1];
    // Rotating (a,b) rather than adding a fixed offset keeps a near-neutral
    // pixel near-neutral -- its magnitude, and so its shift, stays small.
    let theta = interp_hue_sectors(hue, &t.hue_rotate).to_radians();
    let (sin_t, cos_t) = theta.sin_cos();
    let a = (a0 * cos_t - b0 * sin_t) * scale;
    let b = (a0 * sin_t + b0 * cos_t) * scale;
    [new_l, a, b]
}

fn bake_look_chain(chain: &[LookTransfer]) -> Vec<f32> {
    const N: usize = 33;
    let mut result = Vec::with_capacity(3 * N * N * N);
    for i in 0..N * N * N {
        let rgb = voxel_rgb(i, N);
        let mut lab = linear_to_oklab([
            srgb_to_linear(rgb[0]),
            srgb_to_linear(rgb[1]),
            srgb_to_linear(rgb[2]),
        ]);
        for t in chain {
            lab = apply_transfer_to_lab(lab, t);
        }
        result.extend_from_slice(&oklab_to_srgb_in_gamut(lab));
    }
    result
}

fn blend_lut(id: &[f32], v: &[f32], t: f32) -> Vec<f32> {
    id.iter().zip(v).map(|(a, b)| a + (b - a) * t).collect()
}

// Evaluate `outer` at the colors `inner` produces, so a single pass through
// the result does what applying `inner` then `outer` in sequence would do.
pub(crate) fn compose_luts(inner: &[f32], outer: &[f32], n: usize) -> Vec<f32> {
    inner
        .chunks_exact(3)
        .flat_map(|rgb| sample_cube(outer, n, [rgb[0], rgb[1], rgb[2]]))
        .collect()
}

// The single LUT `state` currently produces, composing the optional AI
// auto-adjust LUT with the constrained look-model transform.
pub fn baked_lut(state: &EditState) -> Option<Vec<f32>> {
    let mut lut = state.raw_development.clone();
    if state.ai_lut_enabled {
        let mut ai = combined_photo_lut(state.ai_lut_weights);
        if (state.ai_lut_strength - 1.0).abs() >= 1e-3 {
            ai = blend_lut(&identity_photo_lut(), &ai, state.ai_lut_strength);
        }
        lut = Some(ai);
    }
    if let Some(preset) = state.camera_preset {
        let mut camera = crate::camera_presets::lut(preset, state.camera_style);
        let strength = state.camera_strength.clamp(0.0, 2.0);
        if (strength - 1.0).abs() >= 1e-3 {
            camera = blend_lut(&identity_photo_lut(), &camera, strength);
        }
        lut = Some(match lut {
            Some(base) => compose_luts(&base, &camera, 33),
            None => camera,
        });
    }
    if !state.look.is_empty() {
        let mut look_lut = bake_look_chain(&state.look);
        let strength = state.look_strength.clamp(0.0, 1.0);
        if (strength - 1.0).abs() >= 1e-3 {
            look_lut = blend_lut(&identity_photo_lut(), &look_lut, strength);
        }
        lut = Some(match lut {
            Some(base) => compose_luts(&base, &look_lut, 33),
            None => look_lut,
        });
    }
    if let Some(predicted) = &state.look_lut {
        let strength = state.look_strength.clamp(0.0, 1.0);
        let predicted = if (strength - 1.0).abs() >= 1e-3 {
            blend_lut(&identity_photo_lut(), predicted, strength)
        } else {
            predicted.clone()
        };
        lut = Some(match lut {
            Some(base) => compose_luts(&base, &predicted, 33),
            None => predicted,
        });
    }
    lut
}

fn photo_lut_enabled(state: &EditState) -> bool {
    state.camera_preset.is_some()
        || state.ai_lut_enabled
        || state.raw_development.is_some()
        || !state.look.is_empty()
        || state.look_lut.is_some()
}

#[cfg(test)]
mod tests {
    use super::{
        combined_photo_lut, identity_photo_lut, percentile, photo_luts, EditState, ExportFormat,
        ExportOptions, LookProfile, Processor, RetouchSpot, MAX_RETOUCH_SPOTS,
    };

    // The shader reads the dab list as `vec4 a / vec4 b` per spot after a
    // four-float header; this pins the packing those declarations decode.
    // Regression: a non-finite crop used to reach f32::clamp with its bounds
    // the wrong way round and abort the process, and a zero-sized image did the
    // same. Nothing from a JSON column may be able to do that.
    #[test]
    fn a_corrupt_crop_neither_panics_nor_collapses() {
        let input = (100u32, 100u32);
        for crop in [
            [f32::NAN, 0.0, f32::NAN, 1.0],
            [0.1, f32::NAN, 0.5, 1.0],
            [0.1, 0.0, f32::NAN, f32::NAN],
            [0.1, 0.0, f32::INFINITY, 1.0],
            [f32::NEG_INFINITY, 0.0, 0.5, 1.0],
            [5.0, -5.0, 0.1, 0.1],
            [0.8, 0.8, -0.4, -0.4],
            [0.5, 0.5, 0.0, 0.0],
        ] {
            let px = super::crop_px(crop, input);
            assert!(px.2 >= 1 && px.3 >= 1, "{crop:?} collapsed to {px:?}");
            assert!(
                px.0 + px.2 <= input.0 && px.1 + px.3 <= input.1,
                "{crop:?} escaped the image: {px:?}"
            );
        }
        // No image at all is not a crash either.
        assert!(super::crop_px(super::FULL_CROP, (0, 0)).2 >= 1);

        // Sanitising turns a wild row into something renderable.
        let mut nan = [f32::NAN, 0.0, 0.5, 0.5];
        super::sanitize_crop(&mut nan);
        assert_eq!(nan, super::FULL_CROP);
        let mut wild = [0.99, 0.99, 0.99, 0.99];
        super::sanitize_crop(&mut wild);
        assert!(wild.iter().all(|v| v.is_finite()));
        assert!(wild[0] + wild[2] <= 1.0 + 1e-6 && wild[1] + wild[3] <= 1.0 + 1e-6);
        assert!(wild[2] >= super::MIN_CROP_FRACTION && wild[3] >= super::MIN_CROP_FRACTION);
        // A sane crop survives untouched.
        let mut fine = [0.1, 0.2, 0.5, 0.4];
        super::sanitize_crop(&mut fine);
        assert_eq!(fine, [0.1, 0.2, 0.5, 0.4]);
    }

    // The crop has to survive the round trip through whole pixels, or the panel
    // and the export would drift apart by a pixel at a time.
    #[test]
    fn crop_pixel_rects_are_stable_and_clamped() {
        let input = (6000u32, 4000u32);
        assert!(super::is_full_crop(super::FULL_CROP, input));
        assert_eq!(super::crop_px(super::FULL_CROP, input), (0, 0, 6000, 4000));

        // An arbitrary rect, and then its own fractions again: idempotent.
        let crop = super::crop_from_px((1500.2, 900.7, 4500.4, 3300.1), input);
        let px = super::crop_px(crop, input);
        assert_eq!(px, (1500, 901, 3000, 2399));
        assert_eq!(super::crop_px(super::crop_from_px(
            (px.0 as f32, px.1 as f32, (px.0 + px.2) as f32, (px.1 + px.3) as f32),
            input,
        ), input), px);
        assert!(!super::is_full_crop(crop, input));

        // Silly rects come back inside the image at a usable size.
        let (x, y, w, h) = super::crop_px(super::crop_from_px((-50.0, -50.0, 10.0, 10.0), input), input);
        assert!(x + w <= input.0 && y + h <= input.1);
        assert!(w >= 1 && h >= 1);
        let (_, _, w, h) = super::crop_px(super::crop_from_px((0.0, 0.0, 99999.0, 99999.0), input), input);
        assert_eq!((w, h), input);
    }

    // End to end through the real export path: a JPEG of the cropped region at
    // the requested size. Needs a GPU; skips quietly without one.
    #[test]
    fn exporting_writes_the_crop_at_the_requested_size() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let Some(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        else {
            eprintln!("no GPU adapter; skipping the export test");
            return;
        };
        let Ok((device, queue)) = pollster::block_on(
            adapter.request_device(&wgpu::DeviceDescriptor::default(), None),
        ) else {
            eprintln!("no GPU device; skipping the export test");
            return;
        };
        // Left half dark, right half bright.
        let (w, h) = (64u32, 64u32);
        let img = image::RgbaImage::from_fn(w, h, |x, _| {
            let v = if x < 32 { 30u8 } else { 220 };
            image::Rgba([v, v, v, 255])
        });
        let mut proc = Processor::new(&device);
        proc.upload_rgba(&img, &device, &queue);
        // Crop the bright half, then ask for a quarter of it.
        proc.crop = super::crop_from_px((32.0, 0.0, 64.0, 64.0), (w, h));
        let options = ExportOptions {
            format: ExportFormat::Jpeg,
            quality: 90,
            max_dim: 16,
            dpi: 600,
        };
        // The crop is 32x64; the longest side is what shrinks to 16.
        assert_eq!(proc.export_size(&options), Some((8, 16)));

        let path = std::env::temp_dir().join("image_processor_export_test.jpg");
        let _ = std::fs::remove_file(&path);
        // Written synchronously here, so the test does not race the worker the
        // real export would use.
        proc.export_job(&path, &options, &device, &queue)
            .expect("a job should be readied")
            .write();
        // The file has to say how big it is, or whoever prints it has to
        // guess: this is the JFIF density the printer's driver reads.
        let bytes = std::fs::read(&path).expect("the export should have been written");
        let app0 = bytes.windows(5).position(|w| w == b"JFIF\0");
        let app0 = app0.expect("a JFIF segment");
        let units = bytes[app0 + 7];
        let density = u16::from_be_bytes([bytes[app0 + 8], bytes[app0 + 9]]);
        assert_eq!(units, 1, "units should be pixels per inch");
        assert_eq!(density, 600, "the recorded resolution");

        let decoded = image::open(&path).expect("the export should have been written");
        assert_eq!((decoded.width(), decoded.height()), (8, 16));
        let gray = decoded.to_luma8();
        let mean = gray.pixels().map(|p| f64::from(p[0])).sum::<f64>() / f64::from(8 * 16);
        assert!(
            mean > 150.0,
            "the exported crop should be the bright half, mean was {mean}"
        );

        // A 16-bit PNG stays the full cropped size.
        let png_path = std::env::temp_dir().join("image_processor_export_test.png");
        let _ = std::fs::remove_file(&png_path);
        let png_options = ExportOptions {
            format: ExportFormat::Png16,
            quality: 90,
            max_dim: 0,
            dpi: 0,
        };
        assert_eq!(proc.export_size(&png_options), Some((32, 64)));
        proc.export_job(&png_path, &png_options, &device, &queue)
            .expect("a job should be readied")
            .write();
        let png = image::open(&png_path).expect("the PNG should have been written");
        assert_eq!((png.width(), png.height()), (32, 64));
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&png_path);
    }

    // What we hand to a printer has to be the picture on screen. A gamma or
    // scaling slip anywhere between the display texture and the encoded file
    // would show up as a brightness shift across the whole frame -- which is
    // exactly what a bleached print looks like. Needs a GPU.
    #[test]
    fn the_export_matches_what_the_panel_shows() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let Some(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        else {
            eprintln!("no GPU adapter; skipping the export fidelity test");
            return;
        };
        let Ok((device, queue)) = pollster::block_on(
            adapter.request_device(&wgpu::DeviceDescriptor::default(), None),
        ) else {
            eprintln!("no GPU device; skipping the export fidelity test");
            return;
        };
        // A smooth mid-tone gradient, where a gamma mistake is unmistakable.
        let (w, h) = (256u32, 256u32);
        let img = image::RgbaImage::from_fn(w, h, |x, y| {
            let v = (20 + x * 200 / w).saturating_add(y * 30 / h) as u8;
            image::Rgba([v, v, v, 255])
        });
        let mut proc = Processor::new(&device);
        proc.upload_rgba(&img, &device, &queue);
        let shown = proc.output_pixels(&device, &queue).unwrap();

        let path = std::env::temp_dir().join("image_processor_fidelity_test.jpg");
        let _ = std::fs::remove_file(&path);
        proc.export_job(
            &path,
            &ExportOptions {
                format: ExportFormat::Jpeg,
                quality: 95,
                max_dim: 0,
                dpi: 600,
            },
            &device,
            &queue,
        )
        .expect("a job should be readied")
        .write();
        let written = image::open(&path).expect("the export should exist").to_rgba8();
        let _ = std::fs::remove_file(&path);

        let luma = |px: &[u8], i: usize| {
            0.2126 * f64::from(px[i * 4])
                + 0.7152 * f64::from(px[i * 4 + 1])
                + 0.0722 * f64::from(px[i * 4 + 2])
        };
        let n = (w * h) as usize;
        let panel: f64 = (0..n).map(|i| luma(&shown, i)).sum::<f64>() / n as f64;
        let file: f64 = (0..n).map(|i| luma(written.as_raw(), i)).sum::<f64>() / n as f64;
        assert!(
            (panel - file).abs() < 2.0,
            "the file is not the picture we showed: panel mean {panel:.2}, file mean {file:.2}"
        );
        let worst = (0..n)
            .map(|i| (luma(&shown, i) - luma(written.as_raw(), i)).abs())
            .fold(0.0f64, f64::max);
        assert!(worst < 6.0, "a pixel differs by {worst:.1}: not just JPEG noise");
    }

    // What reaches the file is exactly the crop rect, pixel for pixel.
    #[test]
    fn exporting_takes_exactly_the_crop() {
        // 4x3 image, each pixel tagged with its index in red.
        let (w, h) = (4u32, 3u32);
        let mut pixels = vec![0u16; (w * h * 4) as usize];
        for i in 0..(w * h) as usize {
            pixels[i * 4] = (i * 100) as u16;
        }
        let (x, y, cw, ch) = super::crop_px([0.25, 0.0, 0.5, 2.0 / 3.0], (w, h));
        assert_eq!((x, y, cw, ch), (1, 0, 2, 2));
        let cropped = super::crop_pixels(&pixels, w, (x, y, cw, ch));
        assert_eq!(cropped.len(), (cw * ch * 4) as usize);
        assert_eq!(
            cropped.iter().step_by(4).copied().collect::<Vec<_>>(),
            vec![100, 200, 500, 600]
        );
        // The whole frame is a straight copy.
        assert_eq!(super::crop_pixels(&pixels, w, (0, 0, w, h)), pixels);
    }

    // What a paste carries and what it leaves: everything that describes how
    // the image is developed, and nothing that belongs to one photograph.
    #[test]
    fn pasting_carries_the_develop_and_leaves_the_photograph() {
        let source = EditState {
            exposure: 0.8,
            contrast: 31.0,
            wb_temp: -12.0,
            curve_points: vec![[0.0, 0.0], [0.5, 0.6], [1.0, 1.0]],
            camera_preset: Some(crate::camera_presets::CameraPreset::Fxs),
            ai_lut_enabled: true,
            raw_isp_enabled: true,
            ..EditState::default()
        };
        let mut target = EditState {
            exposure: -0.3,
            crop: [0.2, 0.1, 0.5, 0.6],
            retouch_spots: vec![RetouchSpot::default()],
            ..EditState::default()
        };
        super::copy_develop_settings(&mut target, &source);

        // The develop arrives.
        assert_eq!(target.exposure, 0.8);
        assert_eq!(target.contrast, 31.0);
        assert_eq!(target.wb_temp, -12.0);
        assert_eq!(target.curve_points.len(), 3);
        assert!(target.camera_preset == source.camera_preset);
        assert!(target.ai_lut_enabled);
        // Including what the settings are applied to: a develop on a frame
        // still rendered from a preview would not match the frame it came from.
        assert!(target.raw_isp_enabled);
        // The photograph does not come along.
        assert_eq!(target.crop, [0.2, 0.1, 0.5, 0.6], "framing is per frame");
        assert_eq!(target.retouch_spots.len(), 1, "dabs are per frame");
        // Pasting onto itself changes nothing.
        let mut same = source.clone();
        super::copy_develop_settings(&mut same, &source);
        assert!(same == source, "copying onto the same state must not change it");
    }

    #[test]
    fn retouch_uniform_layout_matches_the_shader() {
        let spot = RetouchSpot {
            x: 0.25,
            y: 0.5,
            radius: 0.125,
            source: [0.01, -0.02],
            gain: [1.1, 1.2, 1.3],
            bias: [0.1, 0.2, 0.3],
        };
        let packed = super::pack_retouch_spots(std::slice::from_ref(&spot));
        assert_eq!(packed.len() as u64 * 4, super::retouch_buffer_bytes());
        assert_eq!(packed[0], 1.0);
        // a = (x, y, radius, gain.r), b = (dx, dy, gain.g, gain.b),
        // c = (bias.r, bias.g, bias.b, unused)
        assert_eq!(
            &packed[4..16],
            &[
                0.25, 0.5, 0.125, 1.1, // a
                0.01, -0.02, 1.2, 1.3, // b
                0.1, 0.2, 0.3, 0.0, // c
            ]
        );
        // Slot 1 starts 12 floats in, on a 16-byte boundary -- the stride the
        // WGSL `array<Spot, 64>` has to agree with.
        assert_eq!(packed.len(), 4 + super::MAX_RETOUCH_SPOTS * 12);
        assert!(packed[16..].iter().all(|v| *v == 0.0));
    }

    // Integration test for the retouch pass: a dab must copy its source over
    // the dab, honour the colour correction, leave the rest alone, and vanish
    // when the spot is removed. Needs a GPU; skips quietly without one.
    #[test]
    fn retouch_layer_applies_and_reverses_a_dab() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let Some(adapter) = pollster::block_on(
            instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
        ) else {
            eprintln!("no GPU adapter; skipping the retouch render test");
            return;
        };
        let Ok((device, queue)) = pollster::block_on(
            adapter.request_device(&wgpu::DeviceDescriptor::default(), None),
        ) else {
            eprintln!("no GPU device; skipping the retouch render test");
            return;
        };

        // 64x64, dark grey with a bright 13x13 patch at (14..=26, 14..=26).
        let (w, h) = (64u32, 64u32);
        let img = image::RgbaImage::from_fn(w, h, |x, y| {
            let bright = (14..=26).contains(&x) && (14..=26).contains(&y);
            let v = if bright { 230u8 } else { 40 };
            image::Rgba([v, v, v, 255])
        });
        let mut proc = Processor::new(&device);
        proc.upload_rgba(&img, &device, &queue);

        let at = |px: &[u8], x: u32, y: u32| px[(y * w + x) as usize * 4];
        let before = proc.output_pixels(&device, &queue).unwrap();
        assert!(at(&before, 20, 20) > 200, "the bright patch should be present first");

        // Dab over the patch, sourcing 32 px to the right (dark grey, 40/255)
        // with a +0.2 correction: 40/255 + 0.2 lands on 91.
        proc.retouch_spots.push(RetouchSpot {
            x: 20.0 / 64.0,
            y: 20.0 / 64.0,
            radius: 8.0 / 64.0,
            source: [0.5, 0.0],
            gain: [1.0; 3],
            bias: [0.2, 0.2, 0.2],
        });
        proc.process(&device, &queue);
        let after = proc.output_pixels(&device, &queue).unwrap();
        let healed = at(&after, 20, 20);
        // Inside the plateau the replacement is total, so the dab centre is
        // exactly the corrected source. (Before the plateau existed this came
        // out at 94: the blend was already easing off halfway across.)
        assert!(
            (89..=93).contains(&healed),
            "dab centre should be fully replaced by the corrected source, got {healed}"
        );
        // (20, 23) is 3.5 px out of 8 -- still inside the plateau, so it must
        // read the same as the centre rather than feathering from it.
        assert_eq!(
            at(&after, 20, 23),
            healed,
            "the plateau should replace at full strength, not feather from the centre"
        );
        // Well outside the dab nothing may change.
        assert_eq!(at(&after, 2, 2), at(&before, 2, 2), "outside the dab changed");
        // Past the plateau the blend eases off rather than stepping.
        let rim = at(&after, 20, 26);
        assert!(
            healed < rim && rim < at(&before, 20, 26),
            "the rim should sit between the dab and the original: {healed} {rim}"
        );

        // Add a second dab so the uniform's array stride is exercised: a wrong
        // stride would have this one read the first one's fields. It sources
        // 30 px to the left, onto the bright patch.
        proc.retouch_spots.push(RetouchSpot {
            x: 44.0 / 64.0,
            y: 20.0 / 64.0,
            radius: 6.0 / 64.0,
            source: [-30.0 / 64.0, 0.0],
            gain: [1.0; 3],
            bias: [0.0; 3],
        });
        proc.process(&device, &queue);
        let two = proc.output_pixels(&device, &queue).unwrap();
        assert!(
            at(&two, 44, 20) > 200,
            "the second dab should have copied the bright patch, got {}",
            at(&two, 44, 20)
        );
        assert_eq!(at(&two, 20, 20), healed, "the first dab should be unaffected");

        // Removing both restores the original pixels exactly.
        proc.retouch_spots.clear();
        proc.process(&device, &queue);
        let restored = proc.output_pixels(&device, &queue).unwrap();
        assert_eq!(restored, before, "clearing the dabs should restore the render");
    }

    // Edits saved before the retouch layer existed must still load, and a dab
    // must survive the SQLite JSON round trip unchanged.
    #[test]
    fn retouch_layer_round_trips_and_old_rows_still_load() {
        let legacy: EditState = serde_json::from_str("{}").expect("a legacy row should load");
        assert!(legacy.retouch_spots.is_empty());

        let mut state = EditState::default();
        state.retouch_spots.push(RetouchSpot {
            x: 0.4,
            y: 0.6,
            radius: 0.05,
            source: [-0.02, 0.03],
            gain: [1.05, 0.95, 1.0],
            bias: [0.01, -0.02, 0.0],
        });
        let json = serde_json::to_string(&state).unwrap();
        let back: EditState = serde_json::from_str(&json).unwrap();
        assert_eq!(back.retouch_spots, state.retouch_spots);

        // A dab saved before `gain` existed must still load (and keep its old,
        // offset-only meaning) rather than taking the whole edit with it.
        let older = r#"{"retouch_spots":[{"x":0.4,"y":0.6,"radius":0.05,
            "source":[-0.02,0.03],"bias":[0.01,-0.02,0.0]}]}"#;
        let parsed: EditState = serde_json::from_str(older).expect("an older dab should load");
        assert_eq!(parsed.retouch_spots.len(), 1);
        assert_eq!(parsed.retouch_spots[0].gain, [1.0; 3]);
        assert_eq!(parsed.retouch_spots[0].bias, [0.01, -0.02, 0.0]);
    }

    // The dab list travels through a JSON column anyone can edit, and a single
    // NaN would NaN every pixel the pass touches.
    #[test]
    fn a_corrupt_dab_list_is_pulled_back_into_range() {
        let mut spots = vec![
            RetouchSpot {
                x: f32::NAN,
                y: 0.5,
                radius: 0.05,
                source: [0.0, 0.0],
                gain: [1.0; 3],
                bias: [0.0; 3],
            },
            RetouchSpot {
                x: 4.0,
                y: -2.0,
                radius: 9.0,
                source: [50.0, -50.0],
                gain: [8.0, 0.0, 1.0],
                bias: [4.0, -4.0, 0.0],
            },
        ];
        // Plus more dabs than the uniform can carry.
        for _ in 0..MAX_RETOUCH_SPOTS + 4 {
            spots.push(RetouchSpot::default());
        }
        super::sanitize_retouch_spots(&mut spots);
        assert_eq!(spots.len(), MAX_RETOUCH_SPOTS);
        for spot in &spots {
            assert!(spot.x.is_finite() && spot.y.is_finite() && spot.radius.is_finite());
            assert!((0.0..=1.0).contains(&spot.x) && (0.0..=1.0).contains(&spot.y));
            assert!((0.0005..=0.5).contains(&spot.radius));
            assert!(spot.source.iter().all(|v| (-1.0..=1.0).contains(v)));
            assert!(spot.gain.iter().all(|v| (0.25..=4.0).contains(v)));
            assert!(spot.bias.iter().all(|v| (-1.0..=1.0).contains(v)));
        }
    }

    #[test]
    fn retouch_packing_caps_the_slot_count() {
        let spots = vec![
            RetouchSpot {
                x: 0.0,
                y: 0.0,
                radius: 0.05,
                source: [0.0, 0.0],
                gain: [1.0; 3],
                bias: [0.0; 3],
            };
            MAX_RETOUCH_SPOTS + 5
        ];
        let packed = super::pack_retouch_spots(&spots);
        assert_eq!(packed[0], MAX_RETOUCH_SPOTS as f32);
    }

    #[test]
    fn embedded_photo_luts_are_readable() {
        assert_eq!(photo_luts().len(), 3 * 3 * 33 * 33 * 33);
    }

    #[test]
    fn softened_photo_lut_is_complete_and_finite() {
        let lut = combined_photo_lut([1.0, 0.0, 0.0]);
        assert_eq!(lut.len(), 3 * 33 * 33 * 33);
        assert!(lut.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn raw_development_enables_the_gpu_photo_lut_pass() {
        let mut state = EditState::default();
        assert!(!super::photo_lut_enabled(&state));
        state.raw_development = Some(identity_photo_lut());
        assert!(super::photo_lut_enabled(&state));
    }

    #[test]
    fn camera_preset_enables_the_gpu_photo_lut_pass() {
        let mut state = EditState::default();
        state.camera_preset = Some(crate::camera_presets::CameraPreset::Fxs);
        assert!(super::photo_lut_enabled(&state));
    }

    #[test]
    fn camera_preset_bakes_a_non_identity_lut() {
        let mut state = EditState::default();
        state.camera_preset = Some(crate::camera_presets::CameraPreset::Fxs);
        let lut = super::baked_lut(&state).unwrap();
        assert_ne!(lut, identity_photo_lut());
    }

    // Regression test: a histogram with mass concentrated in the top bucket
    // (e.g. a photo with blown highlights) used to panic with "attempt to add
    // with overflow" because the walk used `(0u8..)`, whose iterator advances
    // past u8::MAX before yielding bucket 255.
    #[test]
    fn percentile_reaches_last_bucket_without_overflow() {
        let mut histogram = [0u32; 256];
        histogram[0] = 1;
        histogram[255] = 999;
        let total: f64 = histogram.iter().map(|&c| f64::from(c)).sum();

        assert_eq!(percentile(&histogram, total, 0.001), 0.0);
        assert_eq!(percentile(&histogram, total, 0.999), 255.0);
    }

    #[test]
    fn percentile_finds_midpoint() {
        let mut histogram = [0u32; 256];
        histogram[10] = 50;
        histogram[20] = 50;
        let total: f64 = histogram.iter().map(|&c| f64::from(c)).sum();

        assert_eq!(percentile(&histogram, total, 0.5), 10.0);
    }

    // Regression test for a reported symptom: transferring a look from a
    // reference photo containing a lot of grass tinted grass-free target
    // photos green all over. Root cause: `cast` averaged raw (a,b) over
    // every pixel in a tone band with no regard for how saturated it was, so
    // a large vivid-green object outvoted the band's actually-neutral tones
    // by sheer color intensity, and that green got applied as `cast_delta`
    // to every hue in the band on the target -- including hues nothing like
    // grass. A majority-green synthetic image should still measure a small
    // cast, because `cast`'s job is the tint of the *neutral* tones, not the
    // average color of everything at that lightness.
    #[test]
    fn cast_measurement_is_not_dominated_by_a_large_saturated_object() {
        let w = 64;
        let h = 64;
        let img = image::RgbaImage::from_fn(w, h, |x, _y| {
            if x < (w * 7 / 10) {
                image::Rgba([70, 150, 60, 255]) // vivid grass green, 70% of pixels
            } else {
                image::Rgba([128, 128, 128, 255]) // neutral grey, 30% of pixels
            }
        });
        let profile =
            LookProfile::measure(img.as_raw(), w, h, &[]).expect("measure should succeed");

        for (band, evidence) in profile.cast_evidence.iter().enumerate() {
            if *evidence < 0.01 {
                continue;
            }
            let [a, b] = profile.cast[band];
            let cast_chroma = (a * a + b * b).sqrt();
            assert!(
                cast_chroma < 0.05,
                "band {band} cast chroma {cast_chroma} should stay small even though \
                 70% of the image is vivid green -- an unweighted average would land \
                 well above 0.1 here"
            );
        }
    }
}
