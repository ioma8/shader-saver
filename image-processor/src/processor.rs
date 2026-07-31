use std::path::Path;

#[repr(align(4))]
struct AlignedPhotoLuts([u8; 1_293_732]);

static PHOTO_LUTS: AlignedPhotoLuts =
    AlignedPhotoLuts(*include_bytes!("../models/photo_luts.f32"));

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
        let input = [
            (i % 33) as f32 / 32.0,
            ((i / 33) % 33) as f32 / 32.0,
            (i / (33 * 33)) as f32 / 32.0,
        ];
        let output = [sample(i, 0), sample(i, 1), sample(i, 2)];
        let input_lum = input.iter().zip(LUM).map(|(v, w)| v * w).sum::<f32>();
        let output_lum = output.iter().zip(LUM).map(|(v, w)| v * w).sum::<f32>();
        let target_lum = input_lum * exposure_gain;
        let softened_lum = target_lum + (output_lum - target_lum) * TONE_STRENGTH;
        for channel in 0..3 {
            let source_chroma = input[channel] - input_lum;
            let model_chroma = output[channel] - output_lum;
            result.push(
                softened_lum
                    + source_chroma
                    + (model_chroma - source_chroma) * COLOR_STRENGTH,
            );
        }
    }
    result
}

// Everything that defines an image's edit, serialized to SQLite as JSON.
// serde(default) keeps old DB rows loadable when new fields are added.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
#[serde(default)]
pub struct EditState {
    pub exposure: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub wb_temp: f32,
    pub wb_tint: f32,
    pub levels_black: f32,
    pub levels_white: f32,
    pub levels_gamma: f32,
    pub blur_radius: f32,
    pub unsharp_strength: f32,
    pub unsharp_blur_radius: f32,
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
    // A captured-look transfer, applied on top of the AI LUT above. See
    // `derive_look_chain`/`baked_lut`.
    pub look: Vec<LookTransfer>,
    pub look_strength: f32,
    // Set instead of `look` when CanonCGT (see `canoncgt.rs`) produced the
    // transfer; takes priority over `look` in `baked_lut`. Already
    // gamut-mapped before being stored here.
    pub canon_lut: Option<Vec<f32>>,
}

impl Default for EditState {
    fn default() -> Self {
        Self {
            exposure: 0.0,
            brightness: 0.0,
            contrast: 0.0,
            wb_temp: 0.0,
            wb_tint: 0.0,
            levels_black: 0.0,
            levels_white: 255.0,
            levels_gamma: 1.0,
            blur_radius: 0.0,
            unsharp_strength: 0.0,
            unsharp_blur_radius: 2.0,
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
            look: Vec::new(),
            look_strength: 1.0,
            canon_lut: None,
        }
    }
}

pub struct Processor {
    contrast_pipeline: wgpu::ComputePipeline,
    tonal_pipeline: wgpu::ComputePipeline,
    sharpen_pipeline: wgpu::ComputePipeline,
    blur_h_pipeline: wgpu::ComputePipeline,
    blur_v_pipeline: wgpu::ComputePipeline,
    compute_bgl: wgpu::BindGroupLayout,

    input_tex: Option<wgpu::Texture>,
    tex1: Option<wgpu::Texture>,       // contrast output
    tex2: Option<wgpu::Texture>,       // tonal output
    tex3: Option<wgpu::Texture>,       // sharpen output
    tex4: Option<wgpu::Texture>,       // blur_h output
    output_tex: Option<wgpu::Texture>, // blur_v output (final)

    contrast_buf: wgpu::Buffer,
    tonal_buf: wgpu::Buffer,
    blur_buf: wgpu::Buffer,
    sharpen_buf: wgpu::Buffer,
    curve_buf: wgpu::Buffer,

    histogram_bgl: wgpu::BindGroupLayout,
    histogram_pipeline: wgpu::ComputePipeline,
    histogram_buf: wgpu::Buffer,
    histogram_staging: wgpu::Buffer,
    pub histogram: [u32; 256],

    pub image_size: Option<(u32, u32)>,
    pub exposure: f32,     // stops
    pub contrast: f32,     // -100..100, 0 = neutral
    pub wb_temp: f32,      // -100..100, blue ↔ yellow
    pub wb_tint: f32,      // -100..100, green ↔ magenta
    pub levels_black: f32, // 0–255
    pub levels_white: f32, // 0–255
    pub levels_gamma: f32, // 0.1–10.0
    pub blur_radius: f32,
    pub unsharp_strength: f32,
    pub unsharp_blur_radius: f32,
    pub blacks: f32,
    pub shadows: f32,
    pub highlights: f32,
    pub whites: f32,
    pub brightness: f32,
    pub vignette: f32,     // -100..100, negative darkens corners
    pub vignette_mid: f32, // 0..100, where the falloff starts
    // Tone curve control points, normalized [0,1]², sorted by x.
    // Always at least the two endpoints; identity = [[0,0],[1,1]].
    pub curve_points: Vec<[f32; 2]>,
    pub ai_lut_enabled: bool,
    pub ai_lut_weights: [f32; 3],
    pub ai_lut_strength: f32,
    pub look: Vec<LookTransfer>,
    pub look_strength: f32,
    pub canon_lut: Option<Vec<f32>>,
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
                    format: wgpu::TextureFormat::Rgba8Unorm,
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

        Self {
            contrast_pipeline: make_pipeline("contrast_pass"),
            tonal_pipeline: make_pipeline("tonal_pass"),
            sharpen_pipeline: make_pipeline("sharpen_pass"),
            blur_h_pipeline: make_pipeline("blur_h_pass"),
            blur_v_pipeline: make_pipeline("blur_v_pass"),
            compute_bgl,
            input_tex: None,
            tex1: None,
            tex2: None,
            tex3: None,
            tex4: None,
            output_tex: None,
            contrast_buf: make_buf(32),
            tonal_buf: make_buf(32),
            blur_buf: make_buf(32),
            sharpen_buf: make_buf(32),
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
            histogram: [0u32; 256],
            image_size: None,
            exposure: 0.0,
            contrast: 0.0,
            wb_temp: 0.0,
            wb_tint: 0.0,
            levels_black: 0.0,
            levels_white: 255.0,
            levels_gamma: 1.0,
            blur_radius: 0.0,
            unsharp_strength: 0.0,
            unsharp_blur_radius: 2.0,
            blacks: 0.0,
            shadows: 0.0,
            highlights: 0.0,
            whites: 0.0,
            brightness: 0.0,
            vignette: 0.0,
            vignette_mid: 50.0,
            curve_points: vec![[0.0, 0.0], [1.0, 1.0]],
            ai_lut_enabled: false,
            ai_lut_weights: [0.0; 3],
            ai_lut_strength: 1.0,
            look: Vec::new(),
            look_strength: 1.0,
            canon_lut: None,
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
                rhs[idx] = 6.0 * ((pts[idx + 1][1] - pts[idx][1]) / h1 - (pts[idx][1] - pts[idx - 1][1]) / h0);
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
        EditState {
            exposure: self.exposure,
            brightness: self.brightness,
            contrast: self.contrast,
            wb_temp: self.wb_temp,
            wb_tint: self.wb_tint,
            levels_black: self.levels_black,
            levels_white: self.levels_white,
            levels_gamma: self.levels_gamma,
            blur_radius: self.blur_radius,
            unsharp_strength: self.unsharp_strength,
            unsharp_blur_radius: self.unsharp_blur_radius,
            blacks: self.blacks,
            shadows: self.shadows,
            highlights: self.highlights,
            whites: self.whites,
            vignette: self.vignette,
            vignette_mid: self.vignette_mid,
            curve_points: self.curve_points.clone(),
            ai_lut_enabled: self.ai_lut_enabled,
            ai_lut_weights: self.ai_lut_weights,
            ai_lut_strength: self.ai_lut_strength,
            look: self.look.clone(),
            look_strength: self.look_strength,
            canon_lut: self.canon_lut.clone(),
        }
    }

    pub fn apply_edit_state(&mut self, s: &EditState) {
        self.exposure = s.exposure;
        self.brightness = s.brightness;
        self.contrast = s.contrast;
        self.wb_temp = s.wb_temp;
        self.wb_tint = s.wb_tint;
        self.levels_black = s.levels_black;
        self.levels_white = s.levels_white;
        self.levels_gamma = s.levels_gamma;
        self.blur_radius = s.blur_radius;
        self.unsharp_strength = s.unsharp_strength;
        self.unsharp_blur_radius = s.unsharp_blur_radius;
        self.blacks = s.blacks;
        self.shadows = s.shadows;
        self.highlights = s.highlights;
        self.whites = s.whites;
        self.vignette = s.vignette;
        self.vignette_mid = s.vignette_mid;
        self.curve_points = if s.curve_points.len() >= 2 {
            s.curve_points.clone()
        } else {
            vec![[0.0, 0.0], [1.0, 1.0]]
        };
        self.ai_lut_enabled = s.ai_lut_enabled;
        self.ai_lut_weights = s.ai_lut_weights;
        self.ai_lut_strength = s.ai_lut_strength;
        self.look = s.look.clone();
        self.look_strength = s.look_strength;
        self.canon_lut = s.canon_lut.clone();
    }

    // Derive auto adjustments from the luminance histogram. Call with the
    // histogram of the UNEDITED image (reset + process first).
    pub fn auto_adjust(&mut self) {
        let total_f: f64 = self.histogram.iter().map(|&c| f64::from(c)).sum();
        if total_f == 0.0 {
            return;
        }
        let pct = |p: f64| percentile(&self.histogram, total_f, p);

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

        let input_tex = device.create_texture(&wgpu::TextureDescriptor {
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
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
        });
        queue.write_texture(
            input_tex.as_image_copy(),
            img,
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
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | extra,
                view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
            })
        };

        self.input_tex = Some(input_tex);
        self.tex1 = Some(make_intermediate(wgpu::TextureUsages::empty()));
        self.tex2 = Some(make_intermediate(wgpu::TextureUsages::empty()));
        self.tex3 = Some(make_intermediate(wgpu::TextureUsages::empty()));
        self.tex4 = Some(make_intermediate(wgpu::TextureUsages::empty()));
        self.output_tex = Some(make_intermediate(wgpu::TextureUsages::COPY_SRC));
        self.image_size = Some((width, height));

        self.process(device, queue);
    }

    pub fn output_view(&self) -> Option<wgpu::TextureView> {
        self.output_tex.as_ref().map(|t| {
            t.create_view(&wgpu::TextureViewDescriptor {
                format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
                ..Default::default()
            })
        })
    }

    pub fn input_view(&self) -> Option<wgpu::TextureView> {
        self.input_tex.as_ref().map(|t| {
            t.create_view(&wgpu::TextureViewDescriptor {
                format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
                ..Default::default()
            })
        })
    }

    pub fn has_image(&self) -> bool {
        self.input_tex.is_some()
    }

    // Pipeline: contrast → sharpen → blur_h → blur_v
    // Sharpen operates on the contrast-adjusted image (pre-blur) so the unsharp
    fn write_uniforms(&self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.contrast_buf,
            0,
            bytemuck::cast_slice(&[
                self.contrast, self.levels_black, self.levels_white, self.levels_gamma,
                self.exposure, self.wb_temp, self.wb_tint,
                if self.ai_lut_enabled || !self.look.is_empty() || self.canon_lut.is_some() { 1.0 } else { 0.0 },
            ]),
        );
        queue.write_buffer(
            &self.tonal_buf,
            0,
            bytemuck::cast_slice(&[
                self.blacks, self.shadows, self.highlights, self.whites,
                self.brightness, self.vignette, self.vignette_mid, 0f32,
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
                self.unsharp_strength, self.unsharp_blur_radius,
                0f32, 0f32, 0f32, 0f32, 0f32, 0f32,
            ]),
        );
        let mut data = Vec::with_capacity(256 + 33 * 33 * 33 * 3);
        data.extend(self.curve_lut());
        data.extend(baked_lut(&self.edit_state()).unwrap_or_else(identity_photo_lut));
        queue.write_buffer(&self.curve_buf, 0, bytemuck::cast_slice(&data));
    }

    fn read_histogram(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        hist_view: &wgpu::TextureView,
        w: u32,
        h: u32,
    ) {
        queue.write_buffer(&self.histogram_buf, 0, &[0u8; 1024]);
        let hist_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.histogram_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(hist_view) },
                wgpu::BindGroupEntry { binding: 1, resource: self.histogram_buf.as_entire_binding() },
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

        let slice = self.histogram_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        device.poll(wgpu::Maintain::Wait);
        if rx.recv().unwrap().is_ok() {
            let data = slice.get_mapped_range();
            self.histogram.copy_from_slice(bytemuck::cast_slice(&data));
            drop(data);
        }
        self.histogram_staging.unmap();
    }

    // mask anchors off clean signal, independent of the box-blur slider.
    pub fn process(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let (Some(input), Some(t1), Some(t2), Some(t3), Some(t4), Some(output)) = (
            self.input_tex.as_ref(),
            self.tex1.as_ref(),
            self.tex2.as_ref(),
            self.tex3.as_ref(),
            self.tex4.as_ref(),
            self.output_tex.as_ref(),
        ) else {
            return;
        };

        self.write_uniforms(queue);

        let iv = input.create_view(&wgpu::TextureViewDescriptor::default());
        let t1v = t1.create_view(&wgpu::TextureViewDescriptor::default());
        let t2v = t2.create_view(&wgpu::TextureViewDescriptor::default());
        let t3v = t3.create_view(&wgpu::TextureViewDescriptor::default());
        let t4v = t4.create_view(&wgpu::TextureViewDescriptor::default());
        let ov = output.create_view(&wgpu::TextureViewDescriptor::default());

        let bgl = &self.compute_bgl;
        let curve_buf = &self.curve_buf;
        let make_bg =
            |in_view: &wgpu::TextureView, out_view: &wgpu::TextureView, buf: &wgpu::Buffer| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(in_view) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(out_view) },
                        wgpu::BindGroupEntry { binding: 2, resource: buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: curve_buf.as_entire_binding() },
                    ],
                })
            };

        let contrast_bg  = make_bg(&iv,  &t1v, &self.contrast_buf);
        let tonal_bg     = make_bg(&t1v, &t2v, &self.tonal_buf);
        let sharpen_bg   = make_bg(&t2v, &t3v, &self.sharpen_buf);
        let blur_h_bg    = make_bg(&t3v, &t4v, &self.blur_buf);
        let blur_vert_bg = make_bg(&t4v, &ov,  &self.blur_buf);

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
        dispatch(&self.tonal_pipeline,    &tonal_bg);
        dispatch(&self.sharpen_pipeline,  &sharpen_bg);
        dispatch(&self.blur_h_pipeline,   &blur_h_bg);
        dispatch(&self.blur_v_pipeline,   &blur_vert_bg);
        queue.submit([encoder.finish()]);

        // Separate submission: blur_v write must be visible before histogram reads.
        self.read_histogram(device, queue, &ov, w, h);
    }

    // Read the rendered output back to CPU RGBA8 bytes, synchronously. Used
    // where the caller needs the *actual* rendered pixels rather than the
    // edit sliders — e.g. capturing a look, since an AI-adjusted photo keeps
    // its whole look in the LUT, not in any slider value.
    pub fn output_pixels(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Vec<u8>> {
        let (output, (width, height)) = (self.output_tex.as_ref()?, self.image_size?);

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
            output.as_image_copy(),
            wgpu::ImageCopyBuffer {
                buffer: &staging,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        if rx.recv().unwrap().is_err() {
            return None;
        }

        let pixels = {
            let data = slice.get_mapped_range();
            let mut out = Vec::with_capacity((width * height * 4) as usize);
            for row in 0..height {
                let start = (row * bytes_per_row) as usize;
                out.extend_from_slice(&data[start..start + (width * 4) as usize]);
            }
            out
        };
        staging.unmap();
        Some(pixels)
    }

    pub fn export(&self, path: &Path, device: &wgpu::Device, queue: &wgpu::Queue) {
        let (Some(output), Some((width, height))) = (self.output_tex.as_ref(), self.image_size)
        else {
            return;
        };

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

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        if rx.recv().unwrap().is_err() {
            return;
        }

        // Copy pixel data out of the mapped range so we can unmap before the file write
        let pixels: Vec<u8> = {
            let data = slice.get_mapped_range();
            let mut out = Vec::with_capacity((width * height * 4) as usize);
            for row in 0..height {
                let start = (row * bytes_per_row) as usize;
                out.extend_from_slice(&data[start..start + (width * 4) as usize]);
            }
            out
        };
        staging.unmap();

        // File I/O on a background thread so the main thread is unblocked
        let path = path.to_owned();
        std::thread::spawn(move || {
            if let Some(img) = image::RgbaImage::from_raw(width, height, pixels) {
                img.save(path).ok();
            }
        });
    }
}



pub fn identity_photo_lut() -> Vec<f32> {
    const GRID: usize = 33 * 33 * 33;
    let mut result = Vec::with_capacity(3 * GRID);
    for i in 0..GRID {
        result.push((i % 33) as f32 / 32.0);
        result.push(((i / 33) % 33) as f32 / 32.0);
        result.push((i / (33 * 33)) as f32 / 32.0);
    }
    result
}

pub fn sample_cube(lut: &[f32], n: usize, rgb: [f32; 3]) -> [f32; 3] {
    let max_idx = (n - 1) as f32;
    let p = rgb.map(|v| v.clamp(0.0, 1.0) * max_idx);
    let lo = p.map(|v| v.floor() as usize);
    let hi = lo.map(|v| (v + 1).min(n - 1));
    let d = [p[0] - lo[0] as f32, p[1] - lo[1] as f32, p[2] - lo[2] as f32];
    let cell = |r: usize, g: usize, b: usize, channel: usize| {
        lut[(r + g * n + b * n * n) * 3 + channel]
    };
    let mut out = [0.0f32; 3];
    for (channel, slot) in out.iter_mut().enumerate() {
        let mix = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let c00 = mix(cell(lo[0], lo[1], lo[2], channel), cell(hi[0], lo[1], lo[2], channel), d[0]);
        let c10 = mix(cell(lo[0], hi[1], lo[2], channel), cell(hi[0], hi[1], lo[2], channel), d[0]);
        let c01 = mix(cell(lo[0], lo[1], hi[2], channel), cell(hi[0], lo[1], hi[2], channel), d[0]);
        let c11 = mix(cell(lo[0], hi[1], hi[2], channel), cell(hi[0], hi[1], hi[2], channel), d[0]);
        *slot = mix(mix(c00, c10, d[1]), mix(c01, c11, d[1]), d[2]);
    }
    out
}

fn sample_photo_lut(lut: &[f32], rgb: [f32; 3]) -> [f32; 3] { sample_cube(lut, 33, rgb) }

fn lut_pixels(img: &image::RgbaImage, lut: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(img.as_raw().len());
    for px in img.as_raw().chunks_exact(4) {
        let rgb = sample_photo_lut(lut, [f32::from(px[0])/255.0, f32::from(px[1])/255.0, f32::from(px[2])/255.0]);
        for v in rgb { out.push((v.clamp(0.0, 1.0) * 255.0).round() as u8); }
        out.push(px[3]);
    }
    out
}

pub const LUM: [f32; 3] = [0.2126, 0.7152, 0.0722];
fn luminance(rgb: [f32; 3]) -> f32 { rgb[0]*LUM[0] + rgb[1]*LUM[1] + rgb[2]*LUM[2] }

fn srgb_to_linear(v: f32) -> f32 { if v <= 0.04045 { v/12.92 } else { ((v+0.055)/1.055).powf(2.4) } }
fn linear_to_srgb(v: f32) -> f32 { if v <= 0.0031308 { v*12.92 } else { 1.055*v.powf(1.0/2.4)-0.055 } }
fn linear_to_oklab(rgb: [f32; 3]) -> [f32; 3] {
    let l=0.41222147*rgb[0]+0.53633255*rgb[1]+0.051445995*rgb[2];
    let m=0.2119035*rgb[0]+0.6806995*rgb[1]+0.10739696*rgb[2];
    let s=0.08830246*rgb[0]+0.28171885*rgb[1]+0.6299787*rgb[2];
    let (l,m,s)=(l.cbrt(),m.cbrt(),s.cbrt());
    [0.21045426*l+0.7936178*m-0.004072047*s, 1.9779985*l-2.4285922*m+0.4505937*s, 0.025904037*l+0.78277177*m-0.80867577*s]
}
fn oklab_to_linear(lab: [f32; 3]) -> [f32; 3] {
    let l=lab[0]+0.39633778*lab[1]+0.21580376*lab[2];
    let m=lab[0]-0.105561346*lab[1]-0.06385417*lab[2];
    let s=lab[0]-0.08948418*lab[1]-1.2914855*lab[2];
    let (l,m,s)=(l*l*l,m*m*m,s*s*s);
    [4.0767417*l-3.3077116*m+0.23096994*s, -1.268438*l+2.6097574*m-0.34131938*s, -0.0041960863*l-0.7034186*m+1.7076147*s]
}
fn oklab_to_srgb_in_gamut(lab: [f32; 3]) -> [f32; 3] {
    let lt=lab[0].clamp(0.0,1.0);
    let at=|s:f32| oklab_to_linear([lt,lab[1]*s,lab[2]*s]);
    let fits=|rgb:&[f32;3]| rgb.iter().all(|v| *v>=-1e-4 && *v<=1.0001);
    let mut scale=1.0;
    if !fits(&at(1.0)) { let (mut lo,mut hi)=(0.0f32,1.0f32); for _ in 0..18 { let mid=0.5*(lo+hi); if fits(&at(mid)){lo=mid}else{hi=mid} } scale=lo; }
    at(scale).map(|v| linear_to_srgb(v.clamp(0.0,1.0)).clamp(0.0,1.0))
}
// Radius-1 box blur over the cube's own neighbors (edge-clamped), per channel.
// Meant to remove cell-to-cell jaggedness in a *predicted* LUT (e.g. a
// network's per-cell output that isn't perfectly smooth) without touching the
// grade's actual low-frequency shape -- a deliberate color grade varies slowly
// over the cube; single-cell noise doesn't, so this averaging suppresses the
// noise far more than the grade.
pub fn smooth_lut(lut: &[f32], n: usize) -> Vec<f32> {
    let idx = |r: usize, g: usize, b: usize, c: usize| (r + g * n + b * n * n) * 3 + c;
    let clamp = |v: isize| v.clamp(0, n as isize - 1) as usize;
    let mut out = vec![0f32; lut.len()];
    for b in 0..n {
        for g in 0..n {
            for r in 0..n {
                for c in 0..3 {
                    let mut sum = 0f32;
                    let mut count = 0f32;
                    for db in -1..=1isize {
                        for dg in -1..=1isize {
                            for dr in -1..=1isize {
                                let rr = clamp(r as isize + dr);
                                let gg = clamp(g as isize + dg);
                                let bb = clamp(b as isize + db);
                                sum += lut[idx(rr, gg, bb, c)];
                                count += 1.0;
                            }
                        }
                    }
                    out[idx(r, g, b, c)] = sum / count;
                }
            }
        }
    }
    out
}

pub fn gamut_map_lut(lut: &mut [f32]) {
    for cell in lut.chunks_exact_mut(3) {
        let rgb=[cell[0],cell[1],cell[2]];
        if rgb.iter().all(|v| *v>=0.0 && *v<=1.0) { continue; }
        let mapped = oklab_to_srgb_in_gamut(linear_to_oklab([srgb_to_linear(rgb[0]),srgb_to_linear(rgb[1]),srgb_to_linear(rgb[2])]));
        cell.copy_from_slice(&mapped);
    }
}


pub const REGION_COUNT: usize = 3;

struct RegionPrior { hue: (f32, f32), chroma: (f32, f32), lightness: (f32, f32) }

const REGION_PRIORS: [RegionPrior; REGION_COUNT] = [
    RegionPrior { hue: (10.0, 85.0), chroma: (0.02, 0.20), lightness: (0.30, 0.98) },
    RegionPrior { hue: (95.0, 175.0), chroma: (0.03, 0.40), lightness: (0.15, 0.95) },
    RegionPrior { hue: (200.0, 285.0), chroma: (0.02, 0.40), lightness: (0.45, 1.00) },
];

#[derive(Clone, Copy, PartialEq, Default)]
pub struct RegionTone {
    pub lightness: f32,
    pub chroma: f32,
    pub hue_axis: [f32; 2],
    pub share: f32,
}

const REGION_SHARE_FLOOR: [f32; REGION_COUNT] = [0.002, 0.02, 0.02];

fn in_prior(prior: &RegionPrior, lightness: f32, chroma: f32, hue_degrees: f32) -> bool {
    hue_degrees >= prior.hue.0 && hue_degrees <= prior.hue.1
        && chroma >= prior.chroma.0 && chroma <= prior.chroma.1
        && lightness >= prior.lightness.0 && lightness <= prior.lightness.1
}

pub fn measure_regions(
    pixels: &[u8], width: u32, height: u32, faces: &[[f32; 4]],
) -> [RegionTone; REGION_COUNT] {
    let mut sum_lightness = [0.0f64; REGION_COUNT];
    let mut sum_chroma = [0.0f64; REGION_COUNT];
    let mut sum_axis = [[0.0f64; 2]; REGION_COUNT];
    let mut counted = [0.0f64; REGION_COUNT];

    #[derive(Clone, Copy)]
    struct SkinEllipse { cx: f32, cy: f32, rx: f32, ry: f32 }
    let skin_ellipses: Vec<SkinEllipse> = faces
        .iter()
        .filter(|b| b[2].abs() >= 0.03 && b[2].abs() <= 0.80
                 && b[3].abs() >= 0.03 && b[3].abs() <= 0.80)
        .map(|b| SkinEllipse {
            cx: b[0] + b[2] * 0.5, cy: b[1] + b[3] * 0.45,
            rx: b[2] * 0.20, ry: b[3] * 0.22,
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
        if chroma <= 1e-5 { continue; }
        let hue = lab[2].atan2(lab[1]).to_degrees().rem_euclid(360.0);
        let passes_skin_prior = in_prior(&REGION_PRIORS[0], lab[0], chroma, hue);

        for region in 0..REGION_COUNT {
            let allowed = match region {
                0 => passes_skin_prior,
                2 => y < 0.5,
                _ => true,
            };
            if !allowed { continue; }
            if region != 0 && !in_prior(&REGION_PRIORS[region], lab[0], chroma, hue) { continue; }

            if region == 0 {
                skin_global[0] += f64::from(lab[0]);
                skin_global[1] += f64::from(chroma);
                skin_global[2] += f64::from(lab[1] / chroma);
                skin_global[3] += f64::from(lab[2] / chroma);
                skin_global[4] += 1.0;
                if has_faces && skin_ellipses.iter().any(|e| {
                    let dx = (x - e.cx) / e.rx;
                    let dy = (y - e.cy) / e.ry;
                    dx * dx + dy * dy <= 1.0
                }) {
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
    let floor = if !has_faces { REGION_SHARE_FLOOR[0] * 0.1 } else { REGION_SHARE_FLOOR[0] };

    let mut tones = [RegionTone::default(); REGION_COUNT];
    if skin_stats[4] > 0.0 && skin_share as f32 >= floor {
        let length = (skin_stats[2].powi(2) + skin_stats[3].powi(2)).sqrt().max(1e-9);
        tones[0] = RegionTone {
            lightness: (skin_stats[0] / skin_stats[4]) as f32,
            chroma: (skin_stats[1] / skin_stats[4]) as f32,
            hue_axis: [(skin_stats[2] / length) as f32, (skin_stats[3] / length) as f32],
            share: skin_share as f32,
        };
    }
    for region in 1..REGION_COUNT {
        if counted[region] == 0.0 { continue; }
        let share = (counted[region] / total) as f32;
        if share < REGION_SHARE_FLOOR[region] { continue; }
        let length = (sum_axis[region][0].powi(2) + sum_axis[region][1].powi(2)).sqrt().max(1e-9);
        tones[region] = RegionTone {
            lightness: (sum_lightness[region] / counted[region]) as f32,
            chroma: (sum_chroma[region] / counted[region]) as f32,
            hue_axis: [(sum_axis[region][0] / length) as f32, (sum_axis[region][1] / length) as f32],
            share,
        };
    }
    tones
}

// Captured-look statistics and transfer.
//
// Deliberately *shape*-based, not absolute-level: `tone` records the image's
// own lightness at fixed percentiles of its own histogram (not fixed pixel
// values), so two photos shot under different exposure/light still compare
// on their relative contrast/shape rather than their absolute brightness --
// copying absolute values washed results out or blew them out depending on
// which photo was brighter to start with.
//
// This is the fallback used only when CanonCGT (see `canoncgt.rs`) fails to
// load, and as the pre-normalization step feeding CanonCGT's input for large
// lighting gaps (see `look_chain_for` in main.rs).

// Percentiles of the Oklab lightness histogram sampled as tone anchors,
// splined between at bake time.
pub const LOOK_ANCHORS: [f64; 5] = [0.05, 0.25, 0.5, 0.75, 0.95];
const HUE_SECTORS: usize = 8;

#[derive(Clone, PartialEq)]
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

impl serde::Serialize for LookProfile {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        Raw::from(self).serialize(s)
    }
}
impl<'de> serde::Deserialize<'de> for LookProfile {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(Raw::deserialize(d)?.into())
    }
}

// `RegionTone` and the fixed-size arrays above don't derive
// Serialize/Deserialize themselves, so `LookProfile` round-trips through this
// plain-data mirror instead.
#[derive(serde::Serialize, serde::Deserialize)]
struct Raw {
    tone: Vec<f32>,
    cast: Vec<[f32; 2]>,
    cast_evidence: Vec<f32>,
    chroma: f32,
    hue_chroma: Vec<f32>,
    hue_axis: Vec<[f32; 2]>,
    hue_evidence: Vec<f32>,
    regions: Vec<(f32, f32, [f32; 2], f32)>,
}
impl From<&LookProfile> for Raw {
    fn from(p: &LookProfile) -> Self {
        Self {
            tone: p.tone.to_vec(),
            cast: p.cast.to_vec(),
            cast_evidence: p.cast_evidence.to_vec(),
            chroma: p.chroma,
            hue_chroma: p.hue_chroma.to_vec(),
            hue_axis: p.hue_axis.to_vec(),
            hue_evidence: p.hue_evidence.to_vec(),
            regions: p.regions.iter().map(|r| (r.lightness, r.chroma, r.hue_axis, r.share)).collect(),
        }
    }
}
impl From<Raw> for LookProfile {
    fn from(r: Raw) -> Self {
        let arr = |v: Vec<f32>, n: usize| -> Vec<f32> { let mut v = v; v.resize(n, 0.0); v };
        let tone: [f32; LOOK_ANCHORS.len()] = arr(r.tone, LOOK_ANCHORS.len()).try_into().unwrap();
        let mut cast = [[0f32; 2]; 3];
        for (slot, v) in cast.iter_mut().zip(r.cast) { *slot = v; }
        let cast_evidence: [f32; 3] = arr(r.cast_evidence, 3).try_into().unwrap();
        let mut hue_chroma = [0f32; HUE_SECTORS];
        for (slot, v) in hue_chroma.iter_mut().zip(arr(r.hue_chroma, HUE_SECTORS)) { *slot = v; }
        let mut hue_axis = [[0f32; 2]; HUE_SECTORS];
        for (slot, v) in hue_axis.iter_mut().zip(r.hue_axis) { *slot = v; }
        let mut hue_evidence = [0f32; HUE_SECTORS];
        for (slot, v) in hue_evidence.iter_mut().zip(arr(r.hue_evidence, HUE_SECTORS)) { *slot = v; }
        let mut regions = [RegionTone::default(); REGION_COUNT];
        for (slot, v) in regions.iter_mut().zip(r.regions) {
            *slot = RegionTone { lightness: v.0, chroma: v.1, hue_axis: v.2, share: v.3 };
        }
        Self { tone, cast, cast_evidence, chroma: r.chroma, hue_chroma, hue_axis, hue_evidence, regions }
    }
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
            // rather than reliably-neutral content, which correctly makes
            // `derive_transfer`'s evidence gate more reluctant to trust it.
            cast_evidence[band] = (cast_count[band] / total as f64) as f32;
        }

        let chroma = if chroma_count > 0 { (chroma_sum / f64::from(chroma_count)) as f32 } else { 0.0 };

        let mut hue_chroma = [0f32; HUE_SECTORS];
        let mut hue_axis = [[0f32; 2]; HUE_SECTORS];
        let mut hue_evidence = [0f32; HUE_SECTORS];
        for s in 0..HUE_SECTORS {
            if hue_count[s] == 0 {
                continue;
            }
            hue_chroma[s] = (hue_chroma_sum[s] / f64::from(hue_count[s])) as f32;
            let len = (hue_axis_sum[s][0].powi(2) + hue_axis_sum[s][1].powi(2)).sqrt().max(1e-9);
            hue_axis[s] = [(hue_axis_sum[s][0] / len) as f32, (hue_axis_sum[s][1] / len) as f32];
            hue_evidence[s] = hue_count[s] as f32 / total as f32;
        }

        let regions = measure_regions(pixels, width, height, faces);

        Some(Self { tone, cast, cast_evidence, chroma, hue_chroma, hue_axis, hue_evidence, regions })
    }
}

// One conservative step of a look-transfer chain -- see `derive_look_chain`.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct LookTransfer {
    tone_delta: [f32; LOOK_ANCHORS.len()],
    cast_delta: [[f32; 2]; 3],
    hue_chroma_scale: [f32; HUE_SECTORS],
    // Degrees, applied as a rotation of the (a,b) chroma-plane vector rather
    // than an additive offset: a rotation of a near-zero vector stays near
    // zero, so a near-neutral pixel (grey pavement, an overcast sky) is left
    // alone. An earlier version of this field added a fixed (a,b) offset
    // instead, which shifted low-chroma neutrals by the same absolute amount
    // as fully-saturated colors -- grey stone came out visibly purple.
    hue_rotate: [f32; HUE_SECTORS],
}

fn tone_band(l: f32) -> usize {
    if l < 1.0 / 3.0 { 0 } else if l < 2.0 / 3.0 { 1 } else { 2 }
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

// Conservative gain per refinement pass -- see `derive_look_chain`.
const LOOK_GAIN: f32 = 0.6;
// A hue sector overlapping the skin tone is corrected far more gently: skin
// drift was the single most visible failure mode of earlier versions of this
// pipeline, worse than under-correcting the rest of the image.
const SKIN_SECTOR_DAMP: f32 = 0.3;

pub fn skin_hue_degrees(profile: &LookProfile) -> Option<f32> {
    if profile.regions[0].share <= 0.0 {
        return None;
    }
    let [a, b] = profile.regions[0].hue_axis;
    Some(b.atan2(a).to_degrees().rem_euclid(360.0))
}

fn skin_sector(profile: &LookProfile) -> Option<usize> {
    skin_hue_degrees(profile).map(hue_sector)
}

// Damp a baked 33-cube LUT's deviation from identity within a smooth hue
// neighborhood around `hue_degrees`, blending affected cells back toward
// true identity by up to `1 - damp`. Skin drift was the single most visible
// failure mode this pipeline has had (see `SKIN_SECTOR_DAMP`), but that
// protection lives inside the Oklab statistical stage's own transfer math --
// it doesn't apply to a LUT that arrived some other way (CanonCGT's direct
// prediction, blended in `canoncgt::blend_by_confidence` with no knowledge
// of skin at all). Re-asserting it here, on whatever LUT a caller is about
// to store, closes that gap regardless of which stage's correction ends up
// stronger for a given cell.
pub fn damp_lut_skin_hue(lut: &mut [f32], n: usize, reference: &LookProfile) {
    let Some(hue_degrees) = skin_hue_degrees(reference) else { return };
    let radius = 60.0f32;
    for i in 0..n * n * n {
        let rgb = [
            (i % n) as f32 / (n - 1) as f32,
            ((i / n) % n) as f32 / (n - 1) as f32,
            (i / (n * n)) as f32 / (n - 1) as f32,
        ];
        let lab = linear_to_oklab([srgb_to_linear(rgb[0]), srgb_to_linear(rgb[1]), srgb_to_linear(rgb[2])]);
        let chroma = (lab[1] * lab[1] + lab[2] * lab[2]).sqrt();
        if chroma <= 1e-4 {
            continue;
        }
        let hue = lab[2].atan2(lab[1]).to_degrees().rem_euclid(360.0);
        let angdiff = ((hue - hue_degrees + 540.0) % 360.0 - 180.0).abs();
        let weight = (1.0 - angdiff / radius).clamp(0.0, 1.0);
        if weight <= 0.0 {
            continue;
        }
        let local_damp = 1.0 - weight * (1.0 - SKIN_SECTOR_DAMP);
        let idx = i * 3;
        for (c, &identity_v) in rgb.iter().enumerate() {
            lut[idx + c] = identity_v + (lut[idx + c] - identity_v) * local_damp;
        }
    }
}

fn derive_transfer(current: &LookProfile, reference: &LookProfile) -> LookTransfer {
    let mut tone_delta = [0f32; LOOK_ANCHORS.len()];
    for (slot, (r, c)) in tone_delta.iter_mut().zip(reference.tone.iter().zip(current.tone)) {
        *slot = (r - c) * LOOK_GAIN;
    }

    let mut cast_delta = [[0f32; 2]; 3];
    for (band, slot) in cast_delta.iter_mut().enumerate() {
        if current.cast_evidence[band].min(reference.cast_evidence[band]) > 0.005 {
            *slot = [
                (reference.cast[band][0] - current.cast[band][0]) * LOOK_GAIN,
                (reference.cast[band][1] - current.cast[band][1]) * LOOK_GAIN,
            ];
        }
    }

    // Damp whichever sector holds the *reference's* skin tone: pulling this
    // image's matching hue range that far would be the most visible thing a
    // viewer notices going wrong.
    let damped_sector = skin_sector(reference);

    let mut hue_chroma_scale = [1f32; HUE_SECTORS];
    let mut hue_rotate = [0f32; HUE_SECTORS];
    for s in 0..HUE_SECTORS {
        if current.hue_evidence[s].min(reference.hue_evidence[s]) <= 0.003 {
            continue;
        }
        let damp = if Some(s) == damped_sector { SKIN_SECTOR_DAMP } else { 1.0 };
        let ratio = (reference.hue_chroma[s] / current.hue_chroma[s].max(1e-4)).clamp(0.4, 2.2);
        hue_chroma_scale[s] = (1.0 + (ratio - 1.0) * LOOK_GAIN * damp).clamp(0.4, 1.8);
        let cur_angle = current.hue_axis[s][1].atan2(current.hue_axis[s][0]).to_degrees();
        let ref_angle = reference.hue_axis[s][1].atan2(reference.hue_axis[s][0]).to_degrees();
        let diff = (ref_angle - cur_angle + 540.0).rem_euclid(360.0) - 180.0;
        hue_rotate[s] = diff * LOOK_GAIN * damp * 0.5;
    }

    LookTransfer { tone_delta, cast_delta, hue_chroma_scale, hue_rotate }
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

fn render_through_transfer(img: &image::RgbaImage, t: &LookTransfer) -> image::RgbaImage {
    let mut out = image::RgbaImage::new(img.width(), img.height());
    for (x, y, px) in img.enumerate_pixels() {
        let lab = linear_to_oklab([
            srgb_to_linear(f32::from(px[0]) / 255.0),
            srgb_to_linear(f32::from(px[1]) / 255.0),
            srgb_to_linear(f32::from(px[2]) / 255.0),
        ]);
        let rgb = oklab_to_srgb_in_gamut(apply_transfer_to_lab(lab, t));
        out.put_pixel(x, y, image::Rgba([
            (rgb[0] * 255.0).round() as u8,
            (rgb[1] * 255.0).round() as u8,
            (rgb[2] * 255.0).round() as u8,
            px[3],
        ]));
    }
    out
}

fn render_through_lut33(img: &image::RgbaImage, lut: &[f32]) -> image::RgbaImage {
    image::RgbaImage::from_raw(img.width(), img.height(), lut_pixels(img, lut)).unwrap()
}

fn bake_look_chain(chain: &[LookTransfer]) -> Vec<f32> {
    const N: usize = 33;
    let mut result = Vec::with_capacity(3 * N * N * N);
    for i in 0..N * N * N {
        let rgb = [(i % N) as f32 / 32.0, ((i / N) % N) as f32 / 32.0, (i / (N * N)) as f32 / 32.0];
        let mut lab = linear_to_oklab([srgb_to_linear(rgb[0]), srgb_to_linear(rgb[1]), srgb_to_linear(rgb[2])]);
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

fn compose_lut33(inner: &[f32], outer: &[f32]) -> Vec<f32> {
    inner.chunks_exact(3).flat_map(|rgb| sample_cube(outer, 33, [rgb[0], rgb[1], rgb[2]])).collect()
}

// Derive a chain of conservative corrections that carries `img` (as it
// currently renders under `base`) toward `reference`. Each pass re-measures
// the *previous pass's own preview*, so it only ever corrects what the prior,
// deliberately partial (see `LOOK_GAIN`) pass conservatively left behind --
// jumping straight to the full measured gap in one step is what produced
// oversaturated, clipped results in earlier iterations of this pipeline.
pub fn derive_look_chain(
    img: &image::RgbaImage,
    base: &EditState,
    reference: &LookProfile,
    faces: &[[f32; 4]],
    passes: usize,
) -> Vec<LookTransfer> {
    let mut preview = match baked_lut(base) {
        Some(lut) => render_through_lut33(img, &lut),
        None => img.clone(),
    };
    let mut chain = Vec::new();
    for _ in 0..passes {
        let Some(current) = LookProfile::measure(preview.as_raw(), preview.width(), preview.height(), faces)
        else {
            break;
        };
        let transfer = derive_transfer(&current, reference);
        preview = render_through_transfer(&preview, &transfer);
        chain.push(transfer);
    }
    chain
}

// The single LUT `state` currently produces, composing the AI auto-adjust LUT
// (if enabled) with the look-transfer chain (if any) -- or, if CanonCGT
// produced a transfer, that instead (it supersedes the chain; see
// `look_chain_for` in main.rs). `look_strength`/`ai_lut_strength` blend
// their respective LUT toward the identity, so 0 is a no-op and 1 is the full
// effect; `look_strength` can go up to 2 to carry a transfer further than the
// captured look itself.
pub fn baked_lut(state: &EditState) -> Option<Vec<f32>> {
    if let Some(canon) = &state.canon_lut {
        return Some(if (state.look_strength - 1.0).abs() < 1e-3 {
            canon.clone()
        } else {
            blend_lut(&identity_photo_lut(), canon, state.look_strength)
        });
    }

    let mut lut: Option<Vec<f32>> = None;
    if state.ai_lut_enabled {
        let mut ai = combined_photo_lut(state.ai_lut_weights);
        if (state.ai_lut_strength - 1.0).abs() >= 1e-3 {
            ai = blend_lut(&identity_photo_lut(), &ai, state.ai_lut_strength);
        }
        lut = Some(ai);
    }
    if !state.look.is_empty() {
        let mut look_lut = bake_look_chain(&state.look);
        if (state.look_strength - 1.0).abs() >= 1e-3 {
            look_lut = blend_lut(&identity_photo_lut(), &look_lut, state.look_strength);
        }
        lut = Some(match lut {
            Some(base) => compose_lut33(&base, &look_lut),
            None => look_lut,
        });
    }
    lut
}

#[cfg(test)]
mod tests {
    use super::{
        combined_photo_lut, damp_lut_skin_hue, identity_photo_lut, linear_to_oklab, percentile,
        photo_luts, skin_hue_degrees, srgb_to_linear, LookProfile, RegionTone, HUE_SECTORS,
        LOOK_ANCHORS, REGION_COUNT,
    };

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
        let profile = LookProfile::measure(img.as_raw(), w, h, &[]).expect("measure should succeed");

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

    // Regression test for a reported symptom: after CanonCGT's direct
    // prediction started getting blended into the final LUT (see
    // `canoncgt::blend_by_confidence`), faces came out desaturated. Root
    // cause: `SKIN_SECTOR_DAMP` only ever lived inside the Oklab statistical
    // stage's own transfer math, so a magnitude-weighted blend could still
    // hand back a LUT that pushed skin hard, if CanonCGT's own (undamped)
    // opinion about skin was the stronger of the two. `damp_lut_skin_hue`
    // re-asserts the same protection directly on the LUT a caller is about
    // to use, regardless of which stage produced it.
    #[test]
    fn damp_lut_skin_hue_protects_skin_but_leaves_other_hues_alone() {
        let profile = LookProfile {
            tone: [0.0; LOOK_ANCHORS.len()],
            cast: [[0.0; 2]; 3],
            cast_evidence: [0.0; 3],
            chroma: 0.0,
            hue_chroma: [0.0; HUE_SECTORS],
            hue_axis: [[0.0; 2]; HUE_SECTORS],
            hue_evidence: [0.0; HUE_SECTORS],
            regions: {
                let mut r = [RegionTone::default(); REGION_COUNT];
                r[0] = RegionTone { lightness: 0.6, chroma: 0.05, hue_axis: [1.0, 0.0], share: 0.2 };
                r
            },
        };
        assert_eq!(skin_hue_degrees(&profile), Some(0.0));

        let n = 33usize;
        // A uniform, aggressive shift on every cell -- stands in for an
        // upstream stage (like CanonCGT's own opinion) with no idea skin
        // needs protecting.
        let before: Vec<f32> = identity_photo_lut().iter().map(|v| (v + 0.3).min(1.0)).collect();
        let mut lut = before.clone();

        damp_lut_skin_hue(&mut lut, n, &profile);

        let mut near = Vec::new();
        let mut far = Vec::new();
        for i in 0..n * n * n {
            let rgb = [
                (i % n) as f32 / (n - 1) as f32,
                ((i / n) % n) as f32 / (n - 1) as f32,
                (i / (n * n)) as f32 / (n - 1) as f32,
            ];
            let lab = linear_to_oklab([srgb_to_linear(rgb[0]), srgb_to_linear(rgb[1]), srgb_to_linear(rgb[2])]);
            let chroma = (lab[1] * lab[1] + lab[2] * lab[2]).sqrt();
            if chroma <= 0.02 {
                continue; // hue is meaningless this close to neutral
            }
            let hue = lab[2].atan2(lab[1]).to_degrees().rem_euclid(360.0);
            let idx = i * 3;
            let before_dev = (before[idx] - rgb[0]).abs();
            if before_dev <= 1e-4 {
                continue;
            }
            let retained = (lut[idx] - rgb[0]).abs() / before_dev;
            if !(15.0..=345.0).contains(&hue) {
                near.push(retained);
            } else if (170.0..190.0).contains(&hue) {
                far.push(retained);
            }
        }
        assert!(!near.is_empty() && !far.is_empty(), "test setup should sample both hue neighborhoods");
        let avg = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let (near_avg, far_avg) = (avg(&near), avg(&far));
        println!("near-skin retained={near_avg:.3}  far-from-skin retained={far_avg:.3}");
        assert!(near_avg < 0.6, "skin-hue cells should have most of their deviation damped away, retained {near_avg}");
        assert!(far_avg > 0.9, "cells far from skin hue should be essentially untouched, retained {far_avg}");
    }
}
