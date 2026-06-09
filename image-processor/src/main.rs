mod processor;

use std::path::PathBuf;
use std::sync::Arc;

use egui_wgpu::ScreenDescriptor;
use processor::Processor;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

struct GpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    egui_ctx: egui::Context,
    egui_state: Option<egui_winit::State>,
    egui_renderer: Option<egui_wgpu::Renderer>,
    processor: Option<Processor>,
    image_tex_id: Option<egui::TextureId>,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            gpu: None,
            egui_ctx: egui::Context::default(),
            egui_state: None,
            egui_renderer: None,
            processor: None,
            image_tex_id: None,
        }
    }

    fn load_image(&mut self, path: &std::path::Path) {
        let Some(gpu) = &self.gpu else { return };
        let Some(processor) = &mut self.processor else { return };
        let Some(egui_renderer) = &mut self.egui_renderer else { return };

        if processor.load_image(path, &gpu.device, &gpu.queue) {
            let view = processor.output_view().unwrap();
            if let Some(old_id) = self.image_tex_id.take() {
                egui_renderer.free_texture(&old_id);
            }
            self.image_tex_id = Some(egui_renderer.register_native_texture(
                &gpu.device,
                &view,
                wgpu::FilterMode::Linear,
            ));
            if let Some(window) = &self.window {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("image-processor");
                window.set_title(name);
            }
        }
    }

}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("Image Processor")
            .with_inner_size(PhysicalSize::new(980u32, 600u32))
            .with_min_inner_size(PhysicalSize::new(600u32, 400u32));
        let window = Arc::new(event_loop.create_window(attrs).unwrap());

        let gpu = pollster::block_on(init_gpu(Arc::clone(&window)));
        let format = gpu.config.format;

        let egui_state = egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            Some(gpu.device.limits().max_texture_dimension_2d as usize),
        );

        let egui_renderer = egui_wgpu::Renderer::new(&gpu.device, format, None, 1, false);
        let processor = Processor::new(&gpu.device);

        self.egui_state = Some(egui_state);
        self.egui_renderer = Some(egui_renderer);
        self.processor = Some(processor);
        self.window = Some(window);
        self.gpu = Some(gpu);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        // Forward events to egui first
        if let (Some(state), Some(window)) = (&mut self.egui_state, &self.window) {
            let response = state.on_window_event(window, &event);
            if response.consumed {
                return;
            }
        }

        match &event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput { event, .. } => {
                if event.physical_key == PhysicalKey::Code(KeyCode::Escape) {
                    event_loop.exit();
                }
            }

            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    resize_surface(gpu, *size);
                }
            }

            WindowEvent::DroppedFile(path) => {
                let path = path.clone();
                self.load_image(&path);
            }

            WindowEvent::RedrawRequested => {
                self.render();
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

impl App {
    fn render(&mut self) {
        let (Some(gpu), Some(window), Some(egui_state), Some(egui_renderer), Some(processor)) = (
            self.gpu.as_mut(),
            self.window.as_ref(),
            self.egui_state.as_mut(),
            self.egui_renderer.as_mut(),
            self.processor.as_mut(),
        ) else {
            return;
        };

        let frame = match gpu.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                gpu.surface.configure(&gpu.device, &gpu.config);
                return;
            }
            Err(e) => {
                eprintln!("surface error: {e}");
                return;
            }
        };

        let frame_view = frame.texture.create_view(&Default::default());
        let mut encoder = gpu.device.create_command_encoder(&Default::default());

        // --- Build egui UI ---
        let raw_input = egui_state.take_egui_input(window);
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::SidePanel::right("controls")
                .exact_width(260.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.add_space(16.0);
                    ui.label(egui::RichText::new("IMAGE").small().weak());
                    if ui.button("Open Image…").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Image", &["png", "jpg", "jpeg", "tiff", "tif", "bmp"])
                            .pick_file()
                        {
                            // defer to after egui frame — store path
                            ctx.data_mut(|d| d.insert_temp(egui::Id::new("open_path"), path));
                        }
                    }
                    ui.separator();

                    ui.label(egui::RichText::new("CONTRAST").small().weak());
                    let cr = ui.add(egui::Slider::new(&mut processor.contrast, 0.1..=3.0).show_value(true));
                    if cr.changed() {
                        processor.process(&gpu.device, &gpu.queue);
                    }

                    ui.add_space(8.0);
                    ui.label(egui::RichText::new("BOX BLUR RADIUS").small().weak());
                    let br = ui.add(egui::Slider::new(&mut processor.blur_radius, 0.0..=15.0).integer().show_value(true));
                    if br.changed() {
                        processor.process(&gpu.device, &gpu.queue);
                    }

                    ui.add_space(8.0);
                    ui.label(egui::RichText::new("UNSHARP STRENGTH").small().weak());
                    let sr = ui.add(egui::Slider::new(&mut processor.unsharp_strength, 0.0..=3.0).show_value(true));
                    if sr.changed() {
                        processor.process(&gpu.device, &gpu.queue);
                    }

                    ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                        ui.add_space(8.0);
                        if ui.add_enabled(processor.has_image(), egui::Button::new("Export PNG…").min_size(egui::vec2(228.0, 0.0))).clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("PNG", &["png"])
                                .set_file_name("output.png")
                                .save_file()
                            {
                                ctx.data_mut(|d| d.insert_temp(egui::Id::new("export_path"), path));
                            }
                        }
                        ui.separator();
                    });
                });

            egui::CentralPanel::default()
                .frame(egui::Frame::none().fill(egui::Color32::from_gray(30)))
                .show(ctx, |ui| {
                    if let Some(tex_id) = self.image_tex_id {
                        let available = ui.available_size();
                        if let Some((iw, ih)) = processor.image_size {
                            let scale = (available.x / iw as f32).min(available.y / ih as f32);
                            let display = egui::vec2(iw as f32 * scale, ih as f32 * scale);
                            ui.centered_and_justified(|ui| {
                                ui.image(egui::load::SizedTexture::new(tex_id, display));
                            });
                        }
                    } else {
                        ui.centered_and_justified(|ui| {
                            ui.label(egui::RichText::new("Drop an image here or use Open Image…").weak());
                        });
                    }
                });
        });

        egui_state.handle_platform_output(window, full_output.platform_output);

        // Handle deferred file operations
        let open_path: Option<PathBuf> = self.egui_ctx.data_mut(|d| d.remove_temp(egui::Id::new("open_path")));
        let export_path: Option<PathBuf> = self.egui_ctx.data_mut(|d| d.remove_temp(egui::Id::new("export_path")));

        if let Some(path) = open_path {
            let (device, queue) = (&gpu.device, &gpu.queue);
            if let Some(proc) = &mut self.processor {
                if proc.load_image(&path, device, queue) {
                    let view = proc.output_view().unwrap();
                    if let Some(old) = self.image_tex_id.take() {
                        egui_renderer.free_texture(&old);
                    }
                    self.image_tex_id = Some(egui_renderer.register_native_texture(device, &view, wgpu::FilterMode::Linear));
                    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("image-processor");
                    window.set_title(name);
                }
            }
        }

        if let Some(path) = export_path {
            if let Some(proc) = &self.processor {
                proc.export(&path, &gpu.device, &gpu.queue);
            }
        }

        // After process(), refresh texture binding if image exists
        // (output texture contents changed, egui needs to know the view may be stale)
        if let (Some(proc), Some(tex_id)) = (&self.processor, self.image_tex_id) {
            if let Some(view) = proc.output_view() {
                egui_renderer.update_egui_texture_from_wgpu_texture(
                    &gpu.device,
                    &view,
                    wgpu::FilterMode::Linear,
                    tex_id,
                );
            }
        }

        // Render egui
        let size = window.inner_size();
        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [size.width, size.height],
            pixels_per_point: window.scale_factor() as f32,
        };
        let tris = self.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        for (id, delta) in full_output.textures_delta.set {
            egui_renderer.update_texture(&gpu.device, &gpu.queue, id, &delta);
        }
        egui_renderer.update_buffers(&gpu.device, &gpu.queue, &mut encoder, &tris, &screen_descriptor);

        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &frame_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.12, g: 0.12, b: 0.12, a: 1.0 }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                })
                .forget_lifetime();
            egui_renderer.render(&mut pass, &tris, &screen_descriptor);
        }

        for id in full_output.textures_delta.free {
            egui_renderer.free_texture(&id);
        }

        gpu.queue.submit([encoder.finish()]);
        frame.present();
    }
}

async fn init_gpu(window: Arc<Window>) -> GpuState {
    let size = window.inner_size();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let surface = instance.create_surface(window).unwrap();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        })
        .await
        .expect("no GPU adapter");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .expect("failed to get device");

    let caps = surface.get_capabilities(&adapter);
    let format = caps.formats.iter().find(|f| f.is_srgb()).copied().unwrap_or(caps.formats[0]);
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);

    GpuState { device, queue, surface, config }
}

fn resize_surface(gpu: &mut GpuState, size: PhysicalSize<u32>) {
    if size.width == 0 || size.height == 0 { return; }
    gpu.config.width = size.width;
    gpu.config.height = size.height;
    gpu.surface.configure(&gpu.device, &gpu.config);
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
