use std::path::Path;

pub struct Processor {
    contrast_pipeline: wgpu::ComputePipeline,
    sharpen_pipeline: wgpu::ComputePipeline,
    blur_h_pipeline: wgpu::ComputePipeline,
    blur_v_pipeline: wgpu::ComputePipeline,
    compute_bgl: wgpu::BindGroupLayout,

    input_tex: Option<wgpu::Texture>,
    tex1: Option<wgpu::Texture>, // contrast output
    tex2: Option<wgpu::Texture>, // sharpen output
    tex3: Option<wgpu::Texture>, // blur_h output
    output_tex: Option<wgpu::Texture>, // blur_v output (final)

    contrast_buf: wgpu::Buffer,
    blur_buf: wgpu::Buffer,
    sharpen_buf: wgpu::Buffer,

    pub image_size: Option<(u32, u32)>,
    pub contrast: f32,
    pub blur_radius: f32,
    pub unsharp_strength: f32,
    pub unsharp_blur_radius: f32,
}

impl Processor {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders.wgsl").into()),
        });

        let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            ],
        });

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
                compilation_options: Default::default(),
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

        Self {
            contrast_pipeline: make_pipeline("contrast_pass"),
            sharpen_pipeline: make_pipeline("sharpen_pass"),
            blur_h_pipeline: make_pipeline("blur_h_pass"),
            blur_v_pipeline: make_pipeline("blur_v_pass"),
            compute_bgl,
            input_tex: None,
            tex1: None,
            tex2: None,
            tex3: None,
            output_tex: None,
            contrast_buf: make_buf(8),
            blur_buf: make_buf(8),
            sharpen_buf: make_buf(8),
            image_size: None,
            contrast: 1.0,
            blur_radius: 0.0,
            unsharp_strength: 0.0,
            unsharp_blur_radius: 2.0,
        }
    }

    pub fn load_image(&mut self, path: &Path, device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
        let img = match image::open(path) {
            Ok(i) => i.to_rgba8(),
            Err(_) => return false,
        };
        let (width, height) = img.dimensions();

        let input_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            input_tex.as_image_copy(),
            &img,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: None,
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        let make_intermediate = |extra: wgpu::TextureUsages| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | extra,
                view_formats: &[],
            })
        };

        self.input_tex = Some(input_tex);
        self.tex1 = Some(make_intermediate(wgpu::TextureUsages::empty()));
        self.tex2 = Some(make_intermediate(wgpu::TextureUsages::empty()));
        self.tex3 = Some(make_intermediate(wgpu::TextureUsages::empty()));
        self.output_tex = Some(make_intermediate(wgpu::TextureUsages::COPY_SRC));
        self.image_size = Some((width, height));

        self.process(device, queue);
        true
    }

    pub fn output_view(&self) -> Option<wgpu::TextureView> {
        self.output_tex.as_ref().map(|t| t.create_view(&Default::default()))
    }

    pub fn input_view(&self) -> Option<wgpu::TextureView> {
        self.input_tex.as_ref().map(|t| t.create_view(&Default::default()))
    }

    pub fn has_image(&self) -> bool {
        self.input_tex.is_some()
    }

    // Pipeline: contrast → sharpen → blur_h → blur_v
    // Sharpen operates on the contrast-adjusted image (pre-blur) so the unsharp
    // mask anchors off clean signal, independent of the box-blur slider.
    pub fn process(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let (Some(input), Some(t1), Some(t2), Some(t3), Some(output)) = (
            self.input_tex.as_ref(),
            self.tex1.as_ref(),
            self.tex2.as_ref(),
            self.tex3.as_ref(),
            self.output_tex.as_ref(),
        ) else {
            return;
        };

        queue.write_buffer(&self.contrast_buf, 0, bytemuck::cast_slice(&[self.contrast, 0f32]));
        queue.write_buffer(&self.blur_buf, 0, bytemuck::cast_slice(&[self.blur_radius, 0f32]));
        queue.write_buffer(&self.sharpen_buf, 0, bytemuck::cast_slice(&[self.unsharp_strength, self.unsharp_blur_radius]));

        let iv  = input.create_view(&Default::default());
        let t1v = t1.create_view(&Default::default());
        let t2v = t2.create_view(&Default::default());
        let t3v = t3.create_view(&Default::default());
        let ov  = output.create_view(&Default::default());

        let bgl = &self.compute_bgl;
        let make_bg = |in_view: &wgpu::TextureView, out_view: &wgpu::TextureView, buf: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(in_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(out_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: buf.as_entire_binding() },
                ],
            })
        };

        let contrast_bg = make_bg(&iv,  &t1v, &self.contrast_buf); // input  → t1
        let sharpen_bg  = make_bg(&t1v, &t2v, &self.sharpen_buf);  // t1     → t2
        let blur_h_bg   = make_bg(&t2v, &t3v, &self.blur_buf);     // t2     → t3
        let blur_v_bg   = make_bg(&t3v, &ov,  &self.blur_buf);     // t3     → output

        let (w, h) = self.image_size.unwrap();
        let wg = ((w + 7) / 8, (h + 7) / 8);

        let mut encoder = device.create_command_encoder(&Default::default());

        let mut dispatch = |pipeline: &wgpu::ComputePipeline, bg: &wgpu::BindGroup| {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(wg.0, wg.1, 1);
        };

        dispatch(&self.contrast_pipeline, &contrast_bg);
        dispatch(&self.sharpen_pipeline,  &sharpen_bg);
        dispatch(&self.blur_h_pipeline,   &blur_h_bg);
        dispatch(&self.blur_v_pipeline,   &blur_v_bg);

        queue.submit([encoder.finish()]);
    }

    pub fn export(&self, path: &Path, device: &wgpu::Device, queue: &wgpu::Queue) {
        let (Some(output), Some((width, height))) = (self.output_tex.as_ref(), self.image_size) else {
            return;
        };

        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let bytes_per_row = (width * 4 + align - 1) / align * align;

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (bytes_per_row * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
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
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        device.poll(wgpu::Maintain::Wait);
        if rx.recv().unwrap().is_err() { return; }

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
