mod camera;
mod parser;
mod renderer;

use std::path::PathBuf;
use std::sync::Arc;

use renderer::Renderer;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

struct App {
    vertices: Vec<parser::Vertex>,
    title: String,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    mouse_pressed: bool,
    last_cursor: Option<PhysicalPosition<f64>>,
}

impl App {
    fn new(vertices: Vec<parser::Vertex>, title: String) -> Self {
        Self {
            vertices,
            title,
            window: None,
            renderer: None,
            mouse_pressed: false,
            last_cursor: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title(&self.title)
            .with_inner_size(winit::dpi::PhysicalSize::new(800u32, 600u32));
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        let renderer = pollster::block_on(Renderer::new(Arc::clone(&window), &self.vertices));
        self.window = Some(window);
        self.renderer = Some(renderer);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput { event, .. } => {
                if event.physical_key == PhysicalKey::Code(KeyCode::Escape) {
                    event_loop.exit();
                }
            }

            WindowEvent::Resized(size) => {
                if let Some(r) = &mut self.renderer {
                    r.resize(size);
                }
            }

            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.mouse_pressed = state == ElementState::Pressed;
                if !self.mouse_pressed {
                    self.last_cursor = None;
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_pressed {
                    if let (Some(last), Some(r)) = (self.last_cursor, &mut self.renderer) {
                        let dx = (position.x - last.x) as f32;
                        let dy = (position.y - last.y) as f32;
                        r.camera.rotation_y += dx * 0.01;
                        r.camera.rotation_x += dy * 0.01;
                    }
                    self.last_cursor = Some(position);
                } else {
                    self.last_cursor = Some(position);
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(r) = &mut self.renderer {
                    let dy = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                    };
                    r.camera.distance = (r.camera.distance - dy * 0.3).clamp(0.5, 20.0);
                }
            }

            WindowEvent::DroppedFile(path) => {
                if let Some(vertices) = parser::load_stl(&path) {
                    let title = path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("stl-viewer")
                        .to_string();
                    if let Some(window) = &self.window {
                        window.set_title(&title);
                        let renderer = pollster::block_on(Renderer::new(Arc::clone(window), &vertices));
                        self.renderer = Some(renderer);
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                if let Some(r) = &mut self.renderer {
                    r.render();
                }
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

fn main() {
    let path = std::env::args().nth(1).map(PathBuf::from).or_else(|| {
        rfd::FileDialog::new()
            .add_filter("STL", &["stl"])
            .pick_file()
    });

    let path = match path {
        Some(p) => p,
        None => {
            eprintln!("no file selected");
            std::process::exit(1);
        }
    };

    let vertices = match parser::load_stl(&path) {
        Some(v) => v,
        None => {
            eprintln!("failed to load STL: {}", path.display());
            std::process::exit(1);
        }
    };

    let triangle_count = vertices.len() / 3;
    println!("loaded {} triangles from {}", triangle_count, path.display());

    let title = path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("stl-viewer")
        .to_string();

    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(vertices, title);
    event_loop.run_app(&mut app).unwrap();
}
