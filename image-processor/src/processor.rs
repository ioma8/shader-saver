use std::path::Path;

pub struct Processor {
    contrast_pipeline: wgpu::ComputePipeline,
    blur_pipeline: wgpu::ComputePipeline,
    sharpen_pipeline: wgpu::ComputePipeline,
    compute_bgl: wgpu::BindGroupLayout,

    input_tex: Option<wgpu::Texture>,
    tex1: Option<wgpu::Texture>,
    tex2: Option<wgpu::Texture>,
    output_tex: Option<wgpu::Texture>,

    contrast_buf: wgpu::Buffer,
    blur_buf: wgpu::Buffer,
    sharpen_buf: wgpu::Buffer,

    pub image_size: Option<(u32, u32)>,
    pub contrast: f32,
    pub blur_radius: f32,
    pub unsharp_strength: f32,
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

        let make_buf = || {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 4,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        Self {
            contrast_pipeline: make_pipeline("contrast_pass"),
            blur_pipeline: make_pipeline("blur_pass"),
            sharpen_pipeline: make_pipeline("sharpen_pass"),
            compute_bgl,
            input_tex: None,
            tex1: None,
            tex2: None,
            output_tex: None,
            contrast_buf: make_buf(),
            blur_buf: make_buf(),
            sharpen_buf: make_buf(),
            image_size: None,
            contrast: 1.0,
            blur_radius: 0.0,
            unsharp_strength: 0.0,
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
        self.output_tex = Some(make_intermediate(wgpu::TextureUsages::COPY_SRC));
        self.image_size = Some((width, height));

        self.process(device, queue);
        true
    }

    pub fn output_view(&self) -> Option<wgpu::TextureView> {
        self.output_tex.as_ref().map(|t| t.create_view(&Default::default()))
    }

    pub fn has_image(&self) -> bool {
        self.input_tex.is_some()
    }

    pub fn process(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let (Some(input), Some(t1), Some(t2), Some(output)) = (
            self.input_tex.as_ref(),
            self.tex1.as_ref(),
            self.tex2.as_ref(),
            self.output_tex.as_ref(),
        ) else {
            return;
        };

        queue.write_buffer(&self.contrast_buf, 0, bytemuck::cast_slice(&[self.contrast]));
        queue.write_buffer(&self.blur_buf, 0, bytemuck::cast_slice(&[self.blur_radius]));
        queue.write_buffer(&self.sharpen_buf, 0, bytemuck::cast_slice(&[self.unsharp_strength]));

        let iv = input.create_view(&Default::default());
        let t1v = t1.create_view(&Default::default());
        let t2v = t2.create_view(&Default::default());
        let ov = output.create_view(&Default::default());

        let bgl = &self.compute_bgl;
        let make_bg = |input_view: &wgpu::TextureView, out_view: &wgpu::TextureView, buf: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(out_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: buf.as_entire_binding() },
                ],
            })
        };

        let contrast_bg = make_bg(&iv, &t1v, &self.contrast_buf);
        let blur_bg = make_bg(&t1v, &t2v, &self.blur_buf);
        let sharpen_bg = make_bg(&t2v, &ov, &self.sharpen_buf);

        let (w, h) = self.image_size.unwrap();
        let wg = ((w + 7) / 8, (h + 7) / 8);

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.contrast_pipeline);
            pass.set_bind_group(0, &contrast_bg, &[]);
            pass.dispatch_workgroups(wg.0, wg.1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, &blur_bg, &[]);
            pass.dispatch_workgroups(wg.0, wg.1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.sharpen_pipeline);
            pass.set_bind_group(0, &sharpen_bg, &[]);
            pass.dispatch_workgroups(wg.0, wg.1, 1);
        }
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

        let data = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for row in 0..height {
            let start = (row * bytes_per_row) as usize;
            pixels.extend_from_slice(&data[start..start + (width * 4) as usize]);
        }
        drop(data);
        staging.unmap();

        if let Some(img) = image::RgbaImage::from_raw(width, height, pixels) {
            img.save(path).ok();
        }
    }
}
