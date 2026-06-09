mod processor;

use std::path::PathBuf;
use std::sync::Arc;

use egui_wgpu::ScreenDescriptor;
use processor::Processor;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const KEY_OPEN_PATH: &str = "open_path";
const KEY_EXPORT_PATH: &str = "export_path";

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
    original_tex_id: Option<egui::TextureId>,
    output_dirty: bool,
    // Zoom state
    zoom_fit: bool,
    zoom_scale: f32,
    zoom_offset: egui::Vec2, // image top-left relative to panel top-left
    // Hold-to-show-original: only activates after 300 ms so double-clicks don't trigger it
    preview_hold_start: Option<f64>,
    // Which levels handle is being dragged: 0=black, 1=gamma, 2=white
    levels_drag: Option<usize>,
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
            original_tex_id: None,
            output_dirty: false,
            zoom_fit: true,
            zoom_scale: 1.0,
            zoom_offset: egui::Vec2::ZERO,
            preview_hold_start: None,
            levels_drag: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("Image Processor")
            .with_inner_size(LogicalSize::new(1600.0, 1000.0))
            .with_min_inner_size(LogicalSize::new(900.0, 700.0));
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

        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = egui::Color32::from_gray(28);
        visuals.window_fill = egui::Color32::from_gray(28);
        visuals.override_text_color = Some(egui::Color32::from_gray(220));
        self.egui_ctx.set_visuals(visuals);

        let egui_renderer = egui_wgpu::Renderer::new(&gpu.device, format, None, 1, false);
        let processor = Processor::new(&gpu.device);

        self.egui_state = Some(egui_state);
        self.egui_renderer = Some(egui_renderer);
        self.processor = Some(processor);
        self.window = Some(window);
        self.gpu = Some(gpu);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
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
                self.register_image(&path);
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
    fn register_image(&mut self, path: &std::path::Path) {
        let Some(gpu) = &self.gpu else { return };
        let Some(processor) = &mut self.processor else { return };
        let Some(egui_renderer) = &mut self.egui_renderer else { return };

        if processor.load_image(path, &gpu.device, &gpu.queue) {
            if let Some(id) = self.image_tex_id.take() { egui_renderer.free_texture(&id); }
            if let Some(id) = self.original_tex_id.take() { egui_renderer.free_texture(&id); }

            let output_view = processor.output_view().unwrap();
            let input_view  = processor.input_view().unwrap();

            self.image_tex_id = Some(egui_renderer.register_native_texture(
                &gpu.device, &output_view, wgpu::FilterMode::Linear,
            ));
            self.original_tex_id = Some(egui_renderer.register_native_texture(
                &gpu.device, &input_view, wgpu::FilterMode::Linear,
            ));

            if let Some(window) = &self.window {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("image-processor");
                window.set_title(name);
            }

            self.output_dirty = true;
            self.zoom_fit = true; // reset to fit when a new image is loaded
        }
    }

    fn render(&mut self) {
        if self.gpu.is_none() || self.window.is_none()
            || self.egui_state.is_none() || self.egui_renderer.is_none()
            || self.processor.is_none()
        {
            return;
        }

        let frame = {
            let gpu = self.gpu.as_mut().unwrap();
            match gpu.surface.get_current_texture() {
                Ok(f) => f,
                Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                    gpu.surface.configure(&gpu.device, &gpu.config);
                    return;
                }
                Err(e) => { eprintln!("surface error: {e}"); return; }
            }
        };
        let frame_view = frame.texture.create_view(&Default::default());

        let image_tex_id    = self.image_tex_id;
        let original_tex_id = self.original_tex_id;

        // Scoped egui frame — all field borrows dropped at end of block
        let (shapes, textures_delta, pixels_per_point, open_path, export_path, needs_process) = {
            let window      = self.window.as_ref().unwrap();
            let egui_state  = self.egui_state.as_mut().unwrap();
            let processor   = self.processor.as_mut().unwrap();
            let zoom_fit          = &mut self.zoom_fit;
            let zoom_scale        = &mut self.zoom_scale;
            let zoom_offset       = &mut self.zoom_offset;
            let preview_hold_start = &mut self.preview_hold_start;
            let levels_drag       = &mut self.levels_drag;
            let raw_input   = egui_state.take_egui_input(window);
            let mut needs_process = false;

            let full_output = self.egui_ctx.run(raw_input, |ctx| {
                egui::SidePanel::right("controls")
                    .exact_width(260.0)
                    .resizable(false)
                    .show(ctx, |ui| {
                        // Luminance histogram
                        let hist_size = egui::vec2(ui.available_width(), 90.0);
                        let (hist_rect, _) = ui.allocate_exact_size(hist_size, egui::Sense::hover());
                        let painter = ui.painter_at(hist_rect);
                        painter.rect_filled(hist_rect, 0.0, egui::Color32::from_gray(12));

                        if processor.has_image() {
                            let hist = &processor.histogram;
                            let max = hist.iter().copied().max().unwrap_or(1).max(1) as f32;
                            let log_max = (max + 1.0).ln().max(1.0);
                            let bar_w  = hist_rect.width() / 256.0;
                            let b      = hist_rect.bottom();
                            let color  = egui::Color32::from_gray(190);
                            let uv     = egui::epaint::WHITE_UV;

                            let hs: Vec<f32> = hist.iter().map(|&c| {
                                ((c as f32 + 1.0).ln() / log_max) * hist_rect.height()
                            }).collect();
                            let cx = |i: usize| hist_rect.left() + (i as f32 + 0.5) * bar_w;

                            // Explicit mesh: left cap + 255 trapezoids + right cap.
                            // Bypasses PathShape tessellation so fill is always correct.
                            let mut mesh = egui::Mesh::default();
                            let push3 = |m: &mut egui::Mesh, p: [egui::Pos2; 3]| {
                                let v = m.vertices.len() as u32;
                                for pos in p { m.vertices.push(egui::epaint::Vertex { pos, uv, color }); }
                                m.indices.extend_from_slice(&[v, v+1, v+2]);
                            };
                            let push4 = |m: &mut egui::Mesh, p: [egui::Pos2; 4]| {
                                let v = m.vertices.len() as u32;
                                for pos in p { m.vertices.push(egui::epaint::Vertex { pos, uv, color }); }
                                m.indices.extend_from_slice(&[v, v+1, v+2, v, v+2, v+3]);
                            };

                            // Left cap triangle
                            push3(&mut mesh, [
                                egui::pos2(hist_rect.left(), b),
                                egui::pos2(cx(0), b - hs[0]),
                                egui::pos2(cx(0), b),
                            ]);
                            // Trapezoids between adjacent bin centers
                            for i in 0..255 {
                                push4(&mut mesh, [
                                    egui::pos2(cx(i),   b - hs[i]),
                                    egui::pos2(cx(i+1), b - hs[i+1]),
                                    egui::pos2(cx(i+1), b),
                                    egui::pos2(cx(i),   b),
                                ]);
                            }
                            // Right cap triangle
                            push3(&mut mesh, [
                                egui::pos2(cx(255), b - hs[255]),
                                egui::pos2(hist_rect.right(), b),
                                egui::pos2(cx(255), b),
                            ]);

                            painter.add(egui::Shape::Mesh(mesh));
                        }

                        // Levels: gradient strip with draggable handles (black / gamma / white)
                        const GAMMA_LOG_RANGE: f32 = 1.609_438; // ln(5): handle maps gamma 0.2..5, center = 1.0
                        let strip_h  = 10.0;
                        let marker_h = 12.0;
                        let (strip_area, strip_resp) = ui.allocate_exact_size(
                            egui::vec2(ui.available_width(), strip_h + marker_h),
                            egui::Sense::click_and_drag(),
                        );
                        let sp = ui.painter_at(strip_area);
                        let grad = egui::Rect::from_min_size(strip_area.min, egui::vec2(strip_area.width(), strip_h));

                        let w = strip_area.width();
                        let x_of = |v: f32| strip_area.left() + (v / 255.0) * w;
                        let black_x = x_of(processor.levels_black);
                        let white_x = x_of(processor.levels_white);
                        // Gamma handle position between black/white, log-symmetric so center = 1.0
                        let gamma_t = (0.5 - processor.levels_gamma.ln() / (2.0 * GAMMA_LOG_RANGE)).clamp(0.0, 1.0);
                        let gamma_x = black_x + gamma_t * (white_x - black_x);

                        // Interaction: grab nearest handle on drag start, follow pointer while dragging
                        if strip_resp.drag_started() {
                            if let Some(p) = strip_resp.interact_pointer_pos() {
                                let dists = [(p.x - black_x).abs(), (p.x - gamma_x).abs(), (p.x - white_x).abs()];
                                *levels_drag = dists
                                    .iter()
                                    .enumerate()
                                    .min_by(|a, b| a.1.total_cmp(b.1))
                                    .map(|(i, _)| i);
                            }
                        }
                        if strip_resp.dragged() {
                            if let (Some(h), Some(p)) = (*levels_drag, strip_resp.interact_pointer_pos()) {
                                let v = ((p.x - strip_area.left()) / w * 255.0).clamp(0.0, 255.0);
                                match h {
                                    0 => processor.levels_black = v.round().clamp(0.0, processor.levels_white - 1.0),
                                    2 => processor.levels_white = v.round().clamp(processor.levels_black + 1.0, 255.0),
                                    _ => {
                                        let t = ((p.x - black_x) / (white_x - black_x).max(1.0)).clamp(0.0, 1.0);
                                        processor.levels_gamma = ((0.5 - t) * 2.0 * GAMMA_LOG_RANGE).exp();
                                    }
                                }
                                needs_process = true;
                            }
                        }
                        if strip_resp.drag_stopped() {
                            *levels_drag = None;
                        }
                        if strip_resp.double_clicked() {
                            processor.levels_black = 0.0;
                            processor.levels_white = 255.0;
                            processor.levels_gamma = 1.0;
                            needs_process = true;
                        }
                        if strip_resp.hovered() || levels_drag.is_some() {
                            ctx.set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
                        }

                        // Black → white gradient
                        let uv = egui::epaint::WHITE_UV;
                        let mut gm = egui::Mesh::default();
                        let gv = gm.vertices.len() as u32;
                        gm.vertices.extend([
                            egui::epaint::Vertex { pos: grad.left_top(),     uv, color: egui::Color32::BLACK },
                            egui::epaint::Vertex { pos: grad.right_top(),    uv, color: egui::Color32::WHITE },
                            egui::epaint::Vertex { pos: grad.right_bottom(), uv, color: egui::Color32::WHITE },
                            egui::epaint::Vertex { pos: grad.left_bottom(),  uv, color: egui::Color32::BLACK },
                        ]);
                        gm.indices.extend_from_slice(&[gv, gv+1, gv+2, gv, gv+2, gv+3]);
                        sp.add(egui::Shape::Mesh(gm));

                        // Which handle to highlight: dragged one, or nearest within reach when hovering
                        let highlight = (*levels_drag).or_else(|| {
                            strip_resp.hover_pos().and_then(|p| {
                                let dists = [(p.x - black_x).abs(), (p.x - gamma_x).abs(), (p.x - white_x).abs()];
                                dists
                                    .iter()
                                    .enumerate()
                                    .min_by(|a, b| a.1.total_cmp(b.1))
                                    .filter(|(_, d)| **d < 14.0)
                                    .map(|(i, _)| i)
                            })
                        });

                        // Triangle handles (pointing up, sitting below the gradient strip)
                        let ty = grad.bottom();
                        let by = strip_area.bottom();
                        let mk = |cx: f32, fill: egui::Color32, hot: bool| {
                            let cx = cx.clamp(strip_area.left() + 6.0, strip_area.right() - 6.0);
                            let stroke = if hot {
                                egui::Stroke::new(1.5, egui::Color32::from_gray(220))
                            } else {
                                egui::Stroke::new(1.0, egui::Color32::from_gray(110))
                            };
                            egui::Shape::convex_polygon(
                                vec![egui::pos2(cx, ty), egui::pos2(cx - 6.0, by), egui::pos2(cx + 6.0, by)],
                                fill,
                                stroke,
                            )
                        };
                        sp.add(mk(black_x, egui::Color32::from_gray(20),  highlight == Some(0)));
                        sp.add(mk(gamma_x, egui::Color32::from_gray(128), highlight == Some(1)));
                        sp.add(mk(white_x, egui::Color32::WHITE,          highlight == Some(2)));

                        // Numeric value boxes: black | gamma | white (single compact row)
                        ui.add_space(2.0);
                        ui.columns(3, |cols| {
                            let mut changed = false;
                            cols[0].with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                                changed |= ui.add(
                                    egui::DragValue::new(&mut processor.levels_black)
                                        .range(0.0..=254.0).speed(1.0).max_decimals(0),
                                ).changed();
                            });
                            cols[1].with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                                changed |= ui.add(
                                    egui::DragValue::new(&mut processor.levels_gamma)
                                        .range(0.1..=5.0).speed(0.01).fixed_decimals(2),
                                ).changed();
                            });
                            cols[2].with_layout(egui::Layout::top_down(egui::Align::Max), |ui| {
                                changed |= ui.add(
                                    egui::DragValue::new(&mut processor.levels_white)
                                        .range(1.0..=255.0).speed(1.0).max_decimals(0),
                                ).changed();
                            });
                            if changed {
                                processor.levels_black = processor.levels_black.min(processor.levels_white - 1.0);
                                needs_process = true;
                            }
                        });

                        ui.add_space(10.0);
                        ui.label(egui::RichText::new("IMAGE").small().color(egui::Color32::from_gray(140)));
                        if ui.button("Open Image…").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("Image", &["png", "jpg", "jpeg", "tiff", "tif", "bmp"])
                                .pick_file()
                            {
                                ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_OPEN_PATH), path));
                            }
                        }
                        ui.separator();
                        ui.add_space(4.0);

                        macro_rules! slider_row {
                            ($label:expr, $field:expr, $range:expr, $default:expr, $integer:expr) => {{
                                ui.label(egui::RichText::new($label).small().color(egui::Color32::from_gray(140)));
                                let mut s = egui::Slider::new(&mut $field, $range).show_value(true);
                                if $integer { s = s.integer(); }
                                let r = ui.add(s);
                                // r.double_clicked() only fires on the text input (click sense).
                                // Also check raw input so double-clicking the track/thumb resets too.
                                let double_clicked = r.double_clicked()
                                    || ctx.input(|i| {
                                        i.pointer.button_double_clicked(egui::PointerButton::Primary)
                                            && i.pointer
                                                .interact_pos()
                                                .map(|p| r.rect.contains(p))
                                                .unwrap_or(false)
                                    });
                                if double_clicked {
                                    $field = $default;
                                    needs_process = true;
                                } else if r.changed() {
                                    needs_process = true;
                                }
                                ui.add_space(8.0);
                            }};
                        }

                        slider_row!("CONTRAST",            processor.contrast,            0.5..=2.0,    1.0, false);
                        slider_row!("BOX BLUR RADIUS",     processor.blur_radius,         0.0..=15.0,   0.0, true);
                        slider_row!("UNSHARP STRENGTH",    processor.unsharp_strength,    0.0..=3.0,    0.0, false);
                        slider_row!("UNSHARP BLUR RADIUS", processor.unsharp_blur_radius, 1.0..=10.0,   2.0, true);

                        ui.separator();
                        ui.add_space(4.0);
                        ui.label(egui::RichText::new("TONE").small().color(egui::Color32::from_gray(140)));
                        ui.add_space(4.0);
                        slider_row!("BLACKS",     processor.blacks,     -100.0..=100.0, 0.0, true);
                        slider_row!("SHADOWS",    processor.shadows,    -100.0..=100.0, 0.0, true);
                        slider_row!("HIGHLIGHTS", processor.highlights, -100.0..=100.0, 0.0, true);
                        slider_row!("WHITES",     processor.whites,     -100.0..=100.0, 0.0, true);

                        ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                            ui.add_space(8.0);
                            if ui.add_enabled(
                                processor.has_image(),
                                egui::Button::new("Export PNG…").min_size(egui::vec2(228.0, 0.0)),
                            ).clicked() {
                                if let Some(path) = rfd::FileDialog::new()
                                    .add_filter("PNG", &["png"])
                                    .set_file_name("output.png")
                                    .save_file()
                                {
                                    ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_EXPORT_PATH), path));
                                }
                            }
                            ui.separator();

                            if processor.has_image() {
                                ui.label(
                                    egui::RichText::new("Scroll: zoom · Double-click: 100% / fit · Hold/Space: original")
                                        .small()
                                        .color(egui::Color32::from_gray(120)),
                                );
                            }
                        });
                    });

                egui::CentralPanel::default()
                    .frame(egui::Frame::none().fill(egui::Color32::from_gray(30)))
                    .show(ctx, |ui| {
                        let panel_rect = ui.max_rect();
                        let panel_size = panel_rect.size();

                        // Full-panel interaction: captures clicks, double-clicks, hover for zoom
                        let response = ui.interact(
                            panel_rect,
                            ui.id().with("img_area"),
                            egui::Sense::click_and_drag(),
                        );

                        let is_dragging = response.dragged();
                        let now = ctx.input(|i| i.time);

                        // Track how long the mouse button has been held (without dragging).
                        // Reset on drag so panning never accidentally shows the original.
                        if is_dragging {
                            *preview_hold_start = None;
                        } else if response.is_pointer_button_down_on() && preview_hold_start.is_none() {
                            *preview_hold_start = Some(now);
                        } else if !response.is_pointer_button_down_on() {
                            *preview_hold_start = None;
                        }

                        // Only show original after 300 ms — fast double-clicks finish in ~200–400 ms
                        // and won't reach the threshold, so they don't flash the original.
                        let held_long_enough = preview_hold_start
                            .map(|t| now - t > 0.3)
                            .unwrap_or(false);
                        let show_original = ctx.input(|i| i.key_down(egui::Key::Space))
                            || held_long_enough;
                        let tex_id = if show_original { original_tex_id } else { image_tex_id };

                        if let Some(tid) = tex_id {
                            if let Some((iw, ih)) = processor.image_size {
                                let iw = iw as f32;
                                let ih = ih as f32;
                                let fit_scale = (panel_size.x / iw).min(panel_size.y / ih);

                                let (img_offset, img_scale) = if *zoom_fit {
                                    let fw = iw * fit_scale;
                                    let fh = ih * fit_scale;
                                    (egui::vec2(
                                        (panel_size.x - fw) / 2.0,
                                        (panel_size.y - fh) / 2.0,
                                    ), fit_scale)
                                } else {
                                    (*zoom_offset, *zoom_scale)
                                };

                                // Pan when zoomed — drag translates the image offset
                                if is_dragging && !*zoom_fit {
                                    *zoom_offset += response.drag_delta();
                                }

                                let img_rect = egui::Rect::from_min_size(
                                    panel_rect.min + img_offset,
                                    egui::vec2(iw * img_scale, ih * img_scale),
                                );

                                ui.painter()
                                    .with_clip_rect(panel_rect)
                                    .image(
                                        tid,
                                        img_rect,
                                        egui::Rect::from_min_max(
                                            egui::pos2(0.0, 0.0),
                                            egui::pos2(1.0, 1.0),
                                        ),
                                        egui::Color32::WHITE,
                                    );

                                // Double-click: toggle fit ↔ 100%
                                if response.double_clicked() {
                                    if *zoom_fit {
                                        let cursor = response.hover_pos()
                                            .unwrap_or(panel_rect.center());
                                        let c = cursor - panel_rect.min;
                                        // Image pixel under cursor
                                        let img_px = (c - img_offset) / img_scale;
                                        // At 100%, top-left = cursor - img_px * 1.0
                                        *zoom_offset = c - img_px;
                                        *zoom_scale = 1.0;
                                        *zoom_fit = false;
                                    } else {
                                        *zoom_fit = true;
                                    }
                                }

                                // Scroll: zoom at cursor
                                let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
                                if scroll.abs() > 0.5 && response.hovered() {
                                    let cursor = response.hover_pos()
                                        .unwrap_or(panel_rect.center());
                                    let c = cursor - panel_rect.min;
                                    let factor = (1.0_f32 + scroll * 0.003).clamp(0.8, 1.25);
                                    // Clamp minimum to fit_scale so scrolling out never goes smaller than fit
                                    let new_scale = (img_scale * factor).clamp(fit_scale, 20.0);
                                    let ratio = new_scale / img_scale;
                                    *zoom_offset = c - (c - img_offset) * ratio;
                                    *zoom_scale = new_scale;
                                    // Snap to fit mode when at or very near the fit scale
                                    *zoom_fit = new_scale <= fit_scale * 1.03;
                                }
                            }
                        } else {
                            ui.painter().text(
                                panel_rect.center(),
                                egui::Align2::CENTER_CENTER,
                                "Drop an image here or use Open Image…",
                                egui::FontId::proportional(14.0),
                                egui::Color32::from_gray(140),
                            );
                        }
                    });
            });

            egui_state.handle_platform_output(window, full_output.platform_output);
            let open_path: Option<PathBuf> =
                self.egui_ctx.data_mut(|d| d.remove_temp(egui::Id::new(KEY_OPEN_PATH)));
            let export_path: Option<PathBuf> =
                self.egui_ctx.data_mut(|d| d.remove_temp(egui::Id::new(KEY_EXPORT_PATH)));

            (full_output.shapes, full_output.textures_delta, full_output.pixels_per_point,
             open_path, export_path, needs_process)
        }; // all field borrows dropped here

        // File ops
        if let Some(path) = open_path {
            self.register_image(&path);
        }
        if let Some(path) = export_path {
            if let (Some(proc), Some(gpu)) = (&self.processor, &self.gpu) {
                proc.export(&path, &gpu.device, &gpu.queue);
            }
        }

        // Process once per frame if any slider changed
        if needs_process {
            if let (Some(proc), Some(gpu)) = (self.processor.as_mut(), self.gpu.as_ref()) {
                proc.process(&gpu.device, &gpu.queue);
            }
            self.output_dirty = true;
        }

        // Refresh output texture only when GPU output changed
        if self.output_dirty {
            if let (Some(proc), Some(tex_id)) = (&self.processor, self.image_tex_id) {
                if let (Some(view), Some(er), Some(gpu)) = (
                    proc.output_view(),
                    self.egui_renderer.as_mut(),
                    self.gpu.as_ref(),
                ) {
                    er.update_egui_texture_from_wgpu_texture(
                        &gpu.device, &view, wgpu::FilterMode::Linear, tex_id,
                    );
                }
            }
            self.output_dirty = false;
        }

        // wgpu render
        let gpu           = self.gpu.as_mut().unwrap();
        let window        = self.window.as_ref().unwrap();
        let egui_renderer = self.egui_renderer.as_mut().unwrap();

        let size = window.inner_size();
        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [size.width, size.height],
            pixels_per_point: window.scale_factor() as f32,
        };
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let tris = self.egui_ctx.tessellate(shapes, pixels_per_point);
        for (id, delta) in textures_delta.set {
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
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.11, g: 0.11, b: 0.11, a: 1.0 }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                })
                .forget_lifetime();
            egui_renderer.render(&mut pass, &tris, &screen_descriptor);
        }

        for id in textures_delta.free {
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
