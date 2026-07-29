mod classify;
mod cull;
mod enhance;
mod face;
mod imgload;
mod presets;
mod processor;
mod rate;
mod similar;
mod tags;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use egui_wgpu::ScreenDescriptor;
use processor::{EditState, Processor};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const KEY_OPEN_PATH: &str = "open_path";
const KEY_EXPORT_PATH: &str = "export_path";
const KEY_SCAN_DIR: &str = "scan_dir";
const KEY_AUTO: &str = "auto_adjust";
const KEY_TRASH_REJECTS: &str = "trash_rejects";
const KEY_MOVE_REJECTS: &str = "move_rejects";
const KEY_COPY_PICKS: &str = "copy_picks";
const KEY_CLASSIFY_FOLDER: &str = "classify_folder";
const KEY_RATE_FOLDER: &str = "rate_folder";
const KEY_ADJUST_FOLDER: &str = "adjust_folder";
const KEY_SIMILAR_CULL: &str = "similar_cull";
const CULL_HELP_KEY: &str = "browse_cull_help_visible";

// ---- Edit persistence (SQLite, one JSON row per image path) ----

fn app_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".image-processor"))
}

fn open_db() -> Option<rusqlite::Connection> {
    let dir = app_dir()?;
    std::fs::create_dir_all(&dir).ok()?;
    let conn = rusqlite::Connection::open(dir.join("edits.db")).ok()?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS edits (
            path    TEXT PRIMARY KEY,
            params  TEXT NOT NULL,
            updated INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS app_kv (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );",
    )
    .ok()?;
    cull::init_meta_table(&conn);
    tags::init_table(&conn);
    Some(conn)
}

fn save_edits(conn: &rusqlite::Connection, path: &Path, state: &EditState) {
    let Ok(json) = serde_json::to_string(state) else {
        return;
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs().cast_signed());
    let _ = conn.execute(
        "INSERT INTO edits (path, params, updated) VALUES (?1, ?2, ?3)
         ON CONFLICT(path) DO UPDATE SET params = ?2, updated = ?3",
        rusqlite::params![path.to_string_lossy(), json, now],
    );
}

fn load_edits(conn: &rusqlite::Connection, path: &Path) -> Option<EditState> {
    let json: String = conn
        .query_row(
            "SELECT params FROM edits WHERE path = ?1",
            rusqlite::params![path.to_string_lossy()],
            |row| row.get(0),
        )
        .ok()?;
    serde_json::from_str(&json).ok()
}

fn load_bool_pref(conn: &rusqlite::Connection, key: &str, default: bool) -> bool {
    conn.query_row(
        "SELECT value FROM app_kv WHERE key = ?1",
        rusqlite::params![key],
        |row| row.get::<_, String>(0),
    )
    .ok()
    .and_then(|v| match v.as_str() {
        "0" => Some(false),
        "1" => Some(true),
        _ => None,
    })
    .unwrap_or(default)
}

fn save_bool_pref(conn: &rusqlite::Connection, key: &str, value: bool) {
    let _ = conn.execute(
        "INSERT INTO app_kv (key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        rusqlite::params![key, if value { "1" } else { "0" }],
    );
}

fn label_color(label: cull::Label) -> Option<egui::Color32> {
    match label {
        cull::Label::None => None,
        cull::Label::Red => Some(egui::Color32::from_rgb(224, 92, 92)),
        cull::Label::Yellow => Some(egui::Color32::from_rgb(225, 204, 72)),
        cull::Label::Green => Some(egui::Color32::from_rgb(94, 186, 113)),
        cull::Label::Blue => Some(egui::Color32::from_rgb(96, 152, 233)),
    }
}

fn paint_badges(p: &egui::Painter, rect: egui::Rect, meta: cull::CullMeta) {
    if meta.flag == cull::Flag::Reject {
        p.rect_filled(
            egui::Rect::from_min_size(rect.min + egui::vec2(4.0, 4.0), egui::vec2(14.0, 14.0)),
            3.0,
            egui::Color32::from_rgb(170, 60, 60),
        );
        let c = rect.min + egui::vec2(11.0, 11.0);
        let s = egui::Stroke::new(1.5_f32, egui::Color32::WHITE);
        p.line_segment([c + egui::vec2(-3.5, -3.5), c + egui::vec2(3.5, 3.5)], s);
        p.line_segment([c + egui::vec2(3.5, -3.5), c + egui::vec2(-3.5, 3.5)], s);
    } else if meta.flag == cull::Flag::Pick {
        p.rect_filled(
            egui::Rect::from_min_size(rect.min + egui::vec2(4.0, 4.0), egui::vec2(14.0, 14.0)),
            3.0,
            egui::Color32::from_rgb(66, 128, 235),
        );
        let c = rect.min + egui::vec2(11.0, 11.0);
        let s = egui::Stroke::new(1.5_f32, egui::Color32::WHITE);
        p.line_segment([c + egui::vec2(-3.5, 0.5), c + egui::vec2(-1.0, 3.5)], s);
        p.line_segment([c + egui::vec2(-1.0, 3.5), c + egui::vec2(4.0, -3.5)], s);
    }
    if meta.rating > 0 {
        p.text(
            rect.left_bottom() + egui::vec2(6.0, -6.0),
            egui::Align2::LEFT_BOTTOM,
            "★".repeat(meta.rating as usize),
            egui::FontId::proportional(10.0),
            egui::Color32::from_rgb(235, 196, 55),
        );
    }
    if let Some(color) = label_color(meta.label) {
        p.circle_filled(rect.right_top() - egui::vec2(11.0, -11.0), 4.0, color);
    }
}

fn show_dir_item(
    ui: &mut egui::Ui,
    path: &Path,
    depth: u8,
    expanded: &mut std::collections::HashSet<PathBuf>,
    current: Option<&Path>,
    ctx: &egui::Context,
) {
    let is_exp = expanded.contains(path);
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("/");
    let selected = current == Some(path);
    ui.horizontal(|ui| {
        ui.add_space(f32::from(depth) * 10.0);
        let (tri_rect, tri_resp) = ui.allocate_exact_size(egui::vec2(12.0, 14.0), egui::Sense::click());
        if ui.is_rect_visible(tri_rect) {
            let c = tri_rect.center();
            let color = egui::Color32::from_gray(110);
            let pts = if is_exp {
                vec![c + egui::vec2(-4.0, -2.0), c + egui::vec2(4.0, -2.0), c + egui::vec2(0.0, 3.0)]
            } else {
                vec![c + egui::vec2(-2.0, -4.0), c + egui::vec2(3.0, 0.0), c + egui::vec2(-2.0, 4.0)]
            };
            ui.painter().add(egui::Shape::convex_polygon(pts, color, egui::Stroke::NONE));
        }
        if tri_resp.clicked() {
            if is_exp { expanded.remove(path); } else { expanded.insert(path.to_owned()); }
        }
        let color = if selected {
            egui::Color32::from_rgb(90, 140, 255)
        } else {
            egui::Color32::from_gray(195)
        };
        if ui.add(
            egui::Label::new(egui::RichText::new(name).small().color(color))
                .sense(egui::Sense::click()),
        ).clicked() {
            expanded.insert(path.to_owned());
            ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_SCAN_DIR), path.to_owned()));
        }
    });
    if is_exp {
        let Ok(rd) = std::fs::read_dir(path) else { return };
        let mut dirs: Vec<PathBuf> = rd
            .filter_map(std::result::Result::ok)
            .filter(|e| e.file_type().is_ok_and(|ft| ft.is_dir()))
            .map(|e| e.path())
            .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|s| !s.starts_with('.')))
            .collect();
        dirs.sort_unstable();
        for dir in dirs {
            show_dir_item(ui, &dir, depth.saturating_add(1), expanded, current, ctx);
        }
    }
}

#[derive(PartialEq, Clone, Copy)]
enum View {
    Browse,
    Edit,
}

#[derive(PartialEq, Clone, Copy)]
enum BrowseFilter {
    All,
    Picks,
    Rejects,
    Unflagged,
}

#[derive(PartialEq, Clone, Copy)]
enum BrowseSort {
    Name,
    Date,
    CaptureTime,
}

#[derive(Default)]
struct ThumbExif {
    capture_time: Option<i64>,      // YYYYMMDDHHMMSS for sorting
    shutter: Option<String>,        // pre-formatted: "1/200" or "2s"
    aperture: Option<(u32, u32)>,   // rational (numerator, denominator), displayed as "f/2.8"
    iso: Option<u32>,
}

struct ThumbEntry {
    path: PathBuf,
    tex: Option<egui::TextureHandle>,
    mtime: Option<std::time::SystemTime>,
    exif: ThumbExif,
}

fn read_exif(path: &Path) -> ThumbExif {
    let mut out = ThumbExif::default();
    let Ok(file) = std::fs::File::open(path) else { return out };
    let Ok(exif) = exif::Reader::new().read_from_container(&mut std::io::BufReader::new(file)) else { return out };

    if let Some(f) = exif.get_field(exif::Tag::DateTimeOriginal, exif::In::PRIMARY) {
        if let exif::Value::Ascii(v) = &f.value {
            if let Some(s) = v.first().and_then(|b| std::str::from_utf8(b).ok()) {
                if let Some((date, time)) = s.split_once(' ') {
                    let mut dp = date.split(':').filter_map(|p| p.parse::<i64>().ok());
                    let mut tp = time.split(':').filter_map(|p| p.parse::<i64>().ok());
                    if let (Some(y), Some(mo), Some(d), Some(h), Some(mi), Some(s)) =
                        (dp.next(), dp.next(), dp.next(), tp.next(), tp.next(), tp.next())
                    {
                        out.capture_time = Some(
                            y * 10_000_000_000 + mo * 100_000_000 + d * 1_000_000
                                + h * 10_000 + mi * 100 + s,
                        );
                    }
                }
            }
        }
    }

    if let Some(f) = exif.get_field(exif::Tag::ExposureTime, exif::In::PRIMARY) {
        if let exif::Value::Rational(v) = &f.value {
            if let Some(r) = v.first().filter(|r| r.denom > 0 && r.num > 0) {
                out.shutter = Some(if r.num >= r.denom {
                    // >= 1s: integer whole seconds + optional one decimal place
                    let whole = r.num / r.denom;
                    let tenths = (u64::from(r.num) * 10 / u64::from(r.denom)) % 10;
                    if tenths < 1 { format!("{whole}s") } else { format!("{whole}.{tenths}s") }
                } else {
                    // sub-second: display as 1/N using integer division
                    format!("1/{}", r.denom / r.num)
                });
            }
        }
    }

    if let Some(f) = exif.get_field(exif::Tag::FNumber, exif::In::PRIMARY) {
        if let exif::Value::Rational(v) = &f.value {
            if let Some(r) = v.first().filter(|r| r.denom > 0) {
                out.aperture = Some((r.num, r.denom));
            }
        }
    }

    if let Some(f) = exif.get_field(exif::Tag::PhotographicSensitivity, exif::In::PRIMARY) {
        if let exif::Value::Short(v) = &f.value {
            out.iso = v.first().map(|&i| u32::from(i));
        }
    }

    out
}


struct GpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
}

struct AppFlags {
    output_dirty: bool,
    zoom_fit: bool,
    strip_scroll: bool,
    grid_scroll: bool,
    confirm_trash: bool,
    show_cull_help: bool,
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
    flags: AppFlags,
    // Zoom state
    zoom_scale: f32,
    zoom_offset: egui::Vec2, // image top-left relative to panel top-left
    // Hold-to-show-original: only activates after 300 ms so double-clicks don't trigger it
    preview_hold_start: Option<f64>,
    // Which levels handle is being dragged: 0=black, 1=gamma, 2=white
    levels_drag: Option<usize>,
    // Index of the curve control point being dragged
    curve_drag: Option<usize>,
    // Tabs: thumbnail browser / edit view
    view: View,
    browse_dir: Option<PathBuf>,
    thumbs: Vec<ThumbEntry>,
    thumb_rx: Option<std::sync::mpsc::Receiver<(usize, egui::ColorImage, ThumbExif)>>,
    current_path: Option<PathBuf>,
    selected: Option<PathBuf>,
    meta: HashMap<PathBuf, cull::CullMeta>,
    filter: BrowseFilter,
    min_rating: u8,
    sort: BrowseSort,
    grid_cols: usize,
    tree_expanded: std::collections::HashSet<PathBuf>,
    full_tx: std::sync::mpsc::Sender<(PathBuf, image::RgbaImage)>,
    full_rx: std::sync::mpsc::Receiver<(PathBuf, image::RgbaImage)>,
    full_pending: Option<PathBuf>,
    wants_full: Option<(PathBuf, std::time::Instant)>,
    db: Option<rusqlite::Connection>,
    presets_dir: Option<PathBuf>,
    presets: Vec<String>,
    preset_name: String,
    classifier: Option<Arc<classify::Classifier>>,
    enhancer: Option<Arc<enhance::Enhancer>>,
    tags: HashMap<PathBuf, Vec<String>>,
    tag_edit: String,
    classify_tx: std::sync::mpsc::Sender<(PathBuf, Option<Vec<String>>)>,
    classify_rx: std::sync::mpsc::Receiver<(PathBuf, Option<Vec<String>>)>,
    // (done, total) while a folder tagging batch is running
    classify_progress: Option<(usize, usize)>,
    rater: Option<Arc<rate::Rater>>,
    rate_tx: std::sync::mpsc::Sender<(PathBuf, Option<u8>)>,
    rate_rx: std::sync::mpsc::Receiver<(PathBuf, Option<u8>)>,
    // (done, total) while a folder rating batch is running
    rate_progress: Option<(usize, usize)>,
    adjust_tx: std::sync::mpsc::Sender<(PathBuf, Option<EditState>)>,
    adjust_rx: std::sync::mpsc::Receiver<(PathBuf, Option<EditState>)>,
    adjust_progress: Option<(usize, usize)>,
    face_detector: Option<Arc<face::Detector>>,
    similar_tx: std::sync::mpsc::Sender<(PathBuf, Option<similar::Analysis>)>,
    similar_rx: std::sync::mpsc::Receiver<(PathBuf, Option<similar::Analysis>)>,
    // (done, total) while semantic burst analysis is running
    similar_progress: Option<(usize, usize)>,
    similar_candidates: HashMap<PathBuf, similar::Analysis>,
    similar_summary: Option<(usize, usize)>,
}

impl App {
    fn new() -> Self {
        let db = open_db();
        let meta = db.as_ref().map(cull::load_all_meta).unwrap_or_default();
        let show_cull_help = db.as_ref().is_none_or(|db| load_bool_pref(db, CULL_HELP_KEY, true));
        let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
        let mut tree_expanded = std::collections::HashSet::new();
        let mut p = PathBuf::from("/");
        tree_expanded.insert(p.clone());
        for comp in home.components().skip(1) {
            p.push(comp);
            tree_expanded.insert(p.clone());
        }
        let (full_tx, full_rx) = std::sync::mpsc::channel();
        let presets_dir = app_dir().map(|d| d.join("presets"));
        let presets = presets_dir.as_deref().map(presets::list).unwrap_or_default();
        let tags = db.as_ref().map(tags::load_all_tags).unwrap_or_default();
        let (classify_tx, classify_rx) = std::sync::mpsc::channel();
        let (rate_tx, rate_rx) = std::sync::mpsc::channel();
        let (adjust_tx, adjust_rx) = std::sync::mpsc::channel();
        let (similar_tx, similar_rx) = std::sync::mpsc::channel();
        Self {
            window: None,
            gpu: None,
            egui_ctx: egui::Context::default(),
            egui_state: None,
            egui_renderer: None,
            processor: None,
            image_tex_id: None,
            original_tex_id: None,
            flags: AppFlags {
                output_dirty: false,
                zoom_fit: true,
                strip_scroll: false,
                grid_scroll: false,
                confirm_trash: false,
                show_cull_help,
            },
            zoom_scale: 1.0,
            zoom_offset: egui::Vec2::ZERO,
            preview_hold_start: None,
            levels_drag: None,
            curve_drag: None,
            view: View::Browse,
            browse_dir: None,
            thumbs: Vec::new(),
            thumb_rx: None,
            current_path: None,
            selected: None,
            meta,
            filter: BrowseFilter::All,
            min_rating: 0,
            sort: BrowseSort::Name,
            grid_cols: 1,
            tree_expanded,
            full_tx,
            full_rx,
            full_pending: None,
            wants_full: None,
            db,
            presets_dir,
            presets,
            preset_name: String::new(),
            classifier: classify::Classifier::load().map(Arc::new),
            enhancer: enhance::Enhancer::load().map(Arc::new),
            tags,
            tag_edit: String::new(),
            classify_tx,
            classify_rx,
            classify_progress: None,
            rater: rate::Rater::load().map(Arc::new),
            rate_tx,
            rate_rx,
            rate_progress: None,
            adjust_tx,
            adjust_rx,
            adjust_progress: None,
            face_detector: face::Detector::load().map(Arc::new),
            similar_tx,
            similar_rx,
            similar_progress: None,
            similar_candidates: HashMap::new(),
            similar_summary: None,
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
            None, // egui-winit detects DPI from the window itself
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

        // CLI: open an image (or browse a folder) passed as the first argument
        if self.current_path.is_none() && self.browse_dir.is_none() {
            if let Some(arg) = std::env::args().nth(1) {
                let p = PathBuf::from(arg);
                if p.is_file() {
                    self.register_image(&p);
                } else if p.is_dir() {
                    self.scan_folder(&p);
                }
            }
        }
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
                if event.physical_key == PhysicalKey::Code(KeyCode::Escape)
                    && event.state == winit::event::ElementState::Pressed
                    && self.view == View::Edit {
                        self.view = View::Browse;
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
    // List the folder's images and decode thumbnails on a background thread;
    // results stream in through thumb_rx and are uploaded as egui textures.
    fn scan_folder(&mut self, dir: &std::path::Path) {
        self.similar_candidates.clear();
        self.similar_progress = None;
        self.similar_summary = None;
        while self.similar_rx.try_recv().is_ok() {}
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .map(|rd| {
                rd.filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| imgload::is_supported(p))
                    .collect()
            })
            .unwrap_or_default();
        paths.sort();

        self.browse_dir = Some(dir.to_owned());
        self.thumbs = paths
            .iter()
            .map(|p| ThumbEntry {
                path: p.clone(),
                tex: None,
                mtime: std::fs::metadata(p).and_then(|m| m.modified()).ok(),
                exif: ThumbExif::default(),
            })
            .collect();

        let (tx, rx) = std::sync::mpsc::channel::<(usize, egui::ColorImage, ThumbExif)>();
        self.thumb_rx = Some(rx); // dropping the old rx makes a stale loader thread stop
        let ctx = self.egui_ctx.clone();
        let n_workers = std::thread::available_parallelism().map_or(4, std::num::NonZero::get).min(8);
        let (work_tx, work_rx) = std::sync::mpsc::channel::<(usize, PathBuf)>();
        let work_rx = std::sync::Arc::new(std::sync::Mutex::new(work_rx));
        for _ in 0..n_workers {
            let work_rx = work_rx.clone();
            let tx = tx.clone();
            let ctx = ctx.clone();
            std::thread::spawn(move || {
                loop {
                    let Ok((idx, path)) = work_rx.lock().unwrap().recv() else { break };
                    let exif = read_exif(&path);
                    let Some(img) = imgload::load_preview_rgba(&path, 220) else { continue };
                    let (img_w, img_h) = img.dimensions();
                    let ci = egui::ColorImage::from_rgba_unmultiplied([img_w as usize, img_h as usize], &img);
                    if tx.send((idx, ci, exif)).is_err() { break; }
                    ctx.request_repaint();
                }
            });
        }
        for (i, p) in paths.into_iter().enumerate() {
            let _ = work_tx.send((i, p));
        }
    }

    fn rebind_image_textures(&mut self) {
        let (Some(gpu), Some(proc), Some(er)) = (
            self.gpu.as_ref(),
            self.processor.as_ref(),
            self.egui_renderer.as_mut(),
        ) else {
            return;
        };
        if let Some(id) = self.image_tex_id.take() {
            er.free_texture(&id);
        }
        if let Some(id) = self.original_tex_id.take() {
            er.free_texture(&id);
        }
        let output_view = proc.output_view().unwrap();
        let input_view = proc.input_view().unwrap();
        self.image_tex_id =
            Some(er.register_native_texture(&gpu.device, &output_view, wgpu::FilterMode::Linear));
        self.original_tex_id =
            Some(er.register_native_texture(&gpu.device, &input_view, wgpu::FilterMode::Linear));
    }

    fn apply_cull_action(&mut self, path: &Path, action: cull::CullAction) {
        let mut meta = self.meta.get(path).copied().unwrap_or_default();
        cull::apply_action(&mut meta, action);
        if meta.is_default() {
            self.meta.remove(path);
        } else {
            self.meta.insert(path.to_owned(), meta);
        }
        if let Some(db) = &self.db {
            cull::save_meta(db, path, meta);
        }
    }

    fn finish_similar_cull(&mut self) {
        let candidates: Vec<similar::Candidate> = self
            .similar_candidates
            .drain()
            .map(|(path, analysis)| similar::Candidate { path, analysis })
            .collect();
        let groups = similar::groups(&candidates);
        let mut rejected = 0;
        for group in &groups {
            let manual_pick = group
                .iter()
                .filter(|&&i| self.meta.get(&candidates[i].path).is_some_and(|m| m.flag == cull::Flag::Pick))
                .copied()
                .collect::<Vec<_>>();
            let keep = manual_pick.first().copied().unwrap_or_else(|| {
                group
                    .iter()
                    .copied()
                    .max_by(|&a, &b| {
                        let aa = &candidates[a].analysis;
                        let bb = &candidates[b].analysis;
                        aa.rating
                            .cmp(&bb.rating)
                            .then_with(|| (aa.face_count > 0).cmp(&(bb.face_count > 0)))
                            .then_with(|| aa.largest_face.total_cmp(&bb.largest_face))
                            .then_with(|| candidates[b].path.cmp(&candidates[a].path))
                    })
                    .unwrap()
            });
            for &i in group {
                if i == keep || self.meta.get(&candidates[i].path).is_some_and(|m| m.flag == cull::Flag::Pick) {
                    continue;
                }
                let path = &candidates[i].path;
                let mut meta = self.meta.get(path).copied().unwrap_or_default();
                if meta.flag != cull::Flag::Reject {
                    meta.flag = cull::Flag::Reject;
                    self.meta.insert(path.clone(), meta);
                    if let Some(db) = &self.db {
                        cull::save_meta(db, path, meta);
                    }
                    rejected += 1;
                }
            }
        }
        self.similar_summary = Some((groups.len(), rejected));
    }

    fn flag_count(&self, flag: cull::Flag) -> usize {
        self.thumbs
            .iter()
            .filter(|t| self.meta.get(&t.path).is_some_and(|m| m.flag == flag))
            .count()
    }

    fn flagged_paths(&self, flag: cull::Flag) -> Vec<PathBuf> {
        self.thumbs
            .iter()
            .filter(|t| self.meta.get(&t.path).is_some_and(|m| m.flag == flag))
            .map(|t| t.path.clone())
            .collect()
    }

    fn after_files_removed(&mut self, removed: &[PathBuf]) {
        if removed.is_empty() {
            return;
        }
        if self
            .current_path
            .as_ref()
            .is_some_and(|path| removed.contains(path))
        {
            self.current_path = None;
            self.view = View::Browse;
            self.wants_full = None;
            self.full_pending = None;
        }
        if self
            .selected
            .as_ref()
            .is_some_and(|path| removed.contains(path))
        {
            self.selected = None;
        }
        if let Some(dir) = self.browse_dir.clone() {
            self.scan_folder(&dir);
        }
    }

    fn register_image(&mut self, path: &std::path::Path) {
        let Some(gpu) = &self.gpu else { return };
        let Some(processor) = &mut self.processor else {
            return;
        };
        let Some(_egui_renderer) = &mut self.egui_renderer else {
            return;
        };

        let state = self
            .db
            .as_ref()
            .and_then(|db| load_edits(db, path))
            .unwrap_or_default();
        processor.apply_edit_state(&state);

        let Some((img, preview_quality)) = imgload::load_edit_rgba(path) else {
            return;
        };

        processor.upload_rgba(&img, &gpu.device, &gpu.queue);
        self.rebind_image_textures();
        self.full_pending = None;
        self.wants_full = preview_quality.then(|| (path.to_owned(), std::time::Instant::now()));

        if let Some(window) = &self.window {
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("image-processor");
            window.set_title(name);
        }

        self.tag_edit = self.tags.get(path).map(|t| t.join(", ")).unwrap_or_default();

        self.flags.output_dirty = true;
        self.flags.zoom_fit = true;
        self.view = View::Edit;
        self.current_path = Some(path.to_owned());
        self.selected = Some(path.to_owned());
        self.flags.strip_scroll = true;

        if let Some(parent) = path.parent() {
            if self.browse_dir.as_deref() != Some(parent) {
                self.scan_folder(parent);
            }
        }
    }

    fn render(&mut self) {
        if self.gpu.is_none()
            || self.window.is_none()
            || self.egui_state.is_none()
            || self.egui_renderer.is_none()
            || self.processor.is_none()
        {
            return;
        }

        // Upload any thumbnails the loader thread has finished
        if let Some(rx) = &self.thumb_rx {
            while let Ok((i, img, exif)) = rx.try_recv() {
                if let Some(entry) = self.thumbs.get_mut(i) {
                    entry.tex = Some(self.egui_ctx.load_texture(
                        format!("thumb{i}"),
                        img,
                        egui::TextureOptions::LINEAR,
                    ));
                    entry.exif = exif;
                }
            }
        }
        // Pick up finished classification results. `None` means that image
        // failed to load — still counts toward progress, but nothing to save.
        while let Ok((path, result)) = self.classify_rx.try_recv() {
            if let Some(new_tags) = result {
                if let Some(db) = &self.db {
                    tags::save_tags(db, &path, &new_tags);
                }
                if self.current_path.as_deref() == Some(path.as_path()) {
                    self.tag_edit = new_tags.join(", ");
                }
                self.tags.insert(path, new_tags);
            }
            if let Some((done, total)) = &mut self.classify_progress {
                *done += 1;
                if *done >= *total {
                    self.classify_progress = None;
                }
            }
        }

        // Pick up finished rating results, same shape as classification above.
        while let Ok((path, result)) = self.rate_rx.try_recv() {
            if let Some(stars) = result {
                let mut meta = self.meta.get(&path).copied().unwrap_or_default();
                meta.rating = stars;
                self.meta.insert(path.clone(), meta);
                if let Some(db) = &self.db {
                    cull::save_meta(db, &path, meta);
                }
            }
            if let Some((done, total)) = &mut self.rate_progress {
                *done += 1;
                if *done >= *total {
                    self.rate_progress = None;
                }
            }
        }
        while let Ok((path, result)) = self.adjust_rx.try_recv() {
            if let Some(state) = result {
                if let Some(db) = &self.db {
                    save_edits(db, &path, &state);
                }
                if self.current_path.as_deref() == Some(path.as_path()) {
                    if let (Some(proc), Some(gpu)) = (self.processor.as_mut(), self.gpu.as_ref()) {
                        proc.apply_edit_state(&state);
                        proc.process(&gpu.device, &gpu.queue);
                        self.flags.output_dirty = true;
                    }
                }
            }
            if let Some((done, total)) = &mut self.adjust_progress {
                *done += 1;
                if *done >= *total {
                    self.adjust_progress = None;
                }
            }
        }
        let mut similar_finished = false;
        while let Ok((path, result)) = self.similar_rx.try_recv() {
            if let Some(analysis) = result {
                self.similar_candidates.insert(path, analysis);
            }
            if let Some((done, total)) = &mut self.similar_progress {
                *done += 1;
                if *done >= *total {
                    similar_finished = true;
                }
            }
        }
        if similar_finished {
            self.similar_progress = None;
            self.finish_similar_cull();
        }
        if let Some((path, started)) = self.wants_full.clone() {
            if self.current_path.as_deref() != Some(path.as_path()) {
                self.wants_full = None;
            } else if self.full_pending.is_none() && started.elapsed().as_secs_f32() > 0.6 {
                self.full_pending = Some(path.clone());
                self.wants_full = None;
                let tx = self.full_tx.clone();
                std::thread::spawn(move || {
                    if let Some(img) = imgload::load_rgba(&path, 0) {
                        let _ = tx.send((path, img));
                    }
                });
            }
        }
        let full_results: Vec<(PathBuf, image::RgbaImage)> =
            std::iter::from_fn(|| self.full_rx.try_recv().ok()).collect();
        for (path, img) in full_results {
            if self.full_pending.as_deref() == Some(path.as_path()) {
                self.full_pending = None;
            }
            if self.current_path.as_deref() == Some(path.as_path()) {
                if let (Some(gpu), Some(processor)) = (self.gpu.as_ref(), self.processor.as_mut()) {
                    processor.upload_rgba(&img, &gpu.device, &gpu.queue);
                    self.rebind_image_textures();
                    self.flags.output_dirty = true;
                }
            }
        }
        let mut visible: Vec<usize> = self
            .thumbs
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                let meta = self.meta.get(&entry.path).copied().unwrap_or_default();
                let flag_ok = match self.filter {
                    BrowseFilter::All => true,
                    BrowseFilter::Picks => meta.flag == cull::Flag::Pick,
                    BrowseFilter::Rejects => meta.flag == cull::Flag::Reject,
                    BrowseFilter::Unflagged => meta.flag == cull::Flag::None,
                };
                flag_ok && meta.rating >= self.min_rating
            })
            .map(|(i, _)| i)
            .collect();
        match self.sort {
            BrowseSort::Name => {
                visible.sort_by(|&a, &b| self.thumbs[a].path.cmp(&self.thumbs[b].path));
            }
            BrowseSort::Date => visible.sort_by(|&a, &b| {
                let am = self.thumbs[a]
                    .mtime
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                let bm = self.thumbs[b]
                    .mtime
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                am.cmp(&bm)
                    .then_with(|| self.thumbs[a].path.cmp(&self.thumbs[b].path))
            }),
            BrowseSort::CaptureTime => visible.sort_by(|&a, &b| {
                self.thumbs[a].exif.capture_time.unwrap_or(i64::MAX)
                    .cmp(&self.thumbs[b].exif.capture_time.unwrap_or(i64::MAX))
                    .then_with(|| self.thumbs[a].path.cmp(&self.thumbs[b].path))
            }),
        }
        let n_picks = self.flag_count(cull::Flag::Pick);
        let n_rejects = self.flag_count(cull::Flag::Reject);

        let frame = {
            let gpu = self.gpu.as_mut().unwrap();
            match gpu.surface.get_current_texture() {
                Ok(f) => f,
                Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                    gpu.surface.configure(&gpu.device, &gpu.config);
                    return;
                }
                Err(e) => {
                    eprintln!("surface error: {e}");
                    return;
                }
            }
        };
        let frame_view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let image_tex_id = self.image_tex_id;
        let original_tex_id = self.original_tex_id;
        let mut cull_actions: Vec<(PathBuf, cull::CullAction)> = Vec::new();

        // Scoped egui frame — all field borrows dropped at end of block
        let (
            shapes,
            textures_delta,
            pixels_per_point,
            open_path,
            export_path,
            scan_dir,
            auto_req,
            trash_req,
            move_rejects_dir,
            copy_picks_dir,
            classify_folder_req,
            rate_folder_req,
            adjust_folder_req,
            similar_cull_req,
            mut needs_process,
        ) = {
            let window = self.window.as_ref().unwrap();
            let egui_state = self.egui_state.as_mut().unwrap();
            let processor = self.processor.as_mut().unwrap();
            let zoom_fit = &mut self.flags.zoom_fit;
            let zoom_scale = &mut self.zoom_scale;
            let zoom_offset = &mut self.zoom_offset;
            let preview_hold_start = &mut self.preview_hold_start;
            let levels_drag = &mut self.levels_drag;
            let curve_drag = &mut self.curve_drag;
            let view = &mut self.view;
            let selected = &mut self.selected;
            let filter = &mut self.filter;
            let min_rating = &mut self.min_rating;
            let sort = &mut self.sort;
            let strip_scroll = &mut self.flags.strip_scroll;
            let grid_scroll = &mut self.flags.grid_scroll;
            let grid_cols = &mut self.grid_cols;
            let tree_expanded = &mut self.tree_expanded;
            let confirm_trash = &mut self.flags.confirm_trash;
            let show_cull_help = &mut self.flags.show_cull_help;
            let cull_actions = &mut cull_actions;
            let thumbs = &self.thumbs;
            let browse_dir = &self.browse_dir;
            let current_path = &self.current_path;
            let meta_map = &self.meta;
            let db = self.db.as_ref();
            let presets_dir = self.presets_dir.as_deref();
            let presets = &mut self.presets;
            let preset_name = &mut self.preset_name;
            let classifier_available = self.classifier.is_some();
            let classify_progress = self.classify_progress;
            let rater_available = self.rater.is_some();
            let rate_progress = self.rate_progress;
            let enhancer_available = self.enhancer.is_some();
            let adjust_progress = self.adjust_progress;
            let similar_available = self.classifier.is_some()
                && self.rater.is_some()
                && self.face_detector.is_some();
            let similar_progress = self.similar_progress;
            let similar_summary = self.similar_summary;
            let tags_map = &mut self.tags;
            let tag_edit = &mut self.tag_edit;
            let raw_input = egui_state.take_egui_input(window);
            let mut needs_process = false;

            let full_output = self.egui_ctx.run(raw_input, |ctx| {
                egui::TopBottomPanel::top("tabs").exact_height(34.0).show(ctx, |ui| {
                    ui.horizontal_centered(|ui| {
                        ui.add_space(4.0);
                        ui.selectable_value(view, View::Browse, "  Browse  ");
                        ui.selectable_value(view, View::Edit, "  Edit  ");
                    });
                });
                if *confirm_trash {
                    egui::Window::new("Trash rejected photos?")
                        .collapsible(false)
                        .resizable(false)
                        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                        .show(ctx, |ui| {
                            ui.label(format!("Move {n_rejects} rejected photo(s) to the system trash?"));
                            ui.add_space(8.0);
                            ui.horizontal_wrapped(|ui| {
                                if ui.button("Trash").clicked() {
                                    ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_TRASH_REJECTS), true));
                                    *confirm_trash = false;
                                }
                                if ui.button("Cancel").clicked() {
                                    *confirm_trash = false;
                                }
                            });
                        });
                }

                if !ctx.wants_keyboard_input() {
                    let (left, right, up, down, enter, space) = ctx.input(|i| {
                        (
                            i.key_pressed(egui::Key::ArrowLeft),
                            i.key_pressed(egui::Key::ArrowRight),
                            i.key_pressed(egui::Key::ArrowUp),
                            i.key_pressed(egui::Key::ArrowDown),
                            i.key_pressed(egui::Key::Enter),
                            i.key_pressed(egui::Key::Space),
                        )
                    });
                    let cols = *grid_cols as i32;
                    let delta = (i32::from(right) - i32::from(left)) + (i32::from(down) - i32::from(up)) * cols;
                    if delta != 0 && !visible.is_empty() {
                        let anchor = if *view == View::Edit {
                            current_path.as_deref()
                        } else {
                            selected.as_deref().or(current_path.as_deref())
                        };
                        let pos = anchor
                            .and_then(|a| visible.iter().position(|&i| thumbs[i].path == a));
                        let next = match pos {
                            Some(c) => (c as i32 + delta).clamp(0, visible.len() as i32 - 1) as usize,
                            None => 0,
                        };
                        let p = thumbs[visible[next]].path.clone();
                        if *view == View::Edit {
                            if current_path.as_deref() != Some(p.as_path()) {
                                ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_OPEN_PATH), p));
                            }
                        } else {
                            *selected = Some(p);
                            *grid_scroll = true;
                        }
                    }
                    if *view == View::Browse && enter {
                        if let Some(p) = selected.clone().or(current_path.clone()) {
                            ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_OPEN_PATH), p));
                        }
                    }
                    if *view == View::Browse && space && !visible.is_empty() {
                        // Pick current + advance to next
                        if let Some(path) = selected.clone().or(current_path.clone()) {
                            cull_actions.push((path.clone(), cull::CullAction::TogglePick));
                        }
                        let anchor = selected.as_deref().or(current_path.as_deref());
                        let pos = anchor.and_then(|a| visible.iter().position(|&i| thumbs[i].path == a));
                        let next = match pos {
                            Some(c) => (c + 1).min(visible.len() - 1),
                            None => 0,
                        };
                        *selected = Some(thumbs[visible[next]].path.clone());
                        *grid_scroll = true;
                    }
                    if let Some(path) = selected.clone().or(current_path.clone()) {
                        let keys = ctx.input(|i| (
                            i.key_pressed(egui::Key::Num0),
                            i.key_pressed(egui::Key::Num1),
                            i.key_pressed(egui::Key::Num2),
                            i.key_pressed(egui::Key::Num3),
                            i.key_pressed(egui::Key::Num4),
                            i.key_pressed(egui::Key::Num5),
                            i.key_pressed(egui::Key::Num6),
                            i.key_pressed(egui::Key::Num7),
                            i.key_pressed(egui::Key::Num8),
                            i.key_pressed(egui::Key::Num9),
                            i.key_pressed(egui::Key::P),
                            i.key_pressed(egui::Key::X),
                        ));
                        if keys.0 {
                            cull_actions.push((path.clone(), cull::CullAction::Rating(0)));
                        }
                        for (pressed, rating) in [
                            (keys.1, 1u8),
                            (keys.2, 2),
                            (keys.3, 3),
                            (keys.4, 4),
                            (keys.5, 5),
                        ] {
                            if pressed {
                                cull_actions.push((path.clone(), cull::CullAction::Rating(rating)));
                            }
                        }
                        if keys.10 {
                            cull_actions.push((path.clone(), cull::CullAction::TogglePick));
                        }
                        if keys.11 {
                            cull_actions.push((path.clone(), cull::CullAction::ToggleReject));
                        }
                        for (pressed, label) in [
                            (keys.6, cull::Label::Red),
                            (keys.7, cull::Label::Yellow),
                            (keys.8, cull::Label::Green),
                            (keys.9, cull::Label::Blue),
                        ] {
                            if pressed {
                                cull_actions.push((path.clone(), cull::CullAction::ToggleLabel(label)));
                            }
                        }
                    }
                }

                if *view == View::Browse {
                    egui::Area::new("browse_cull_help".into())
                        .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-16.0, -16.0))
                        .interactable(true)
                        .show(ctx, |ui| {
                            if *show_cull_help {
                                egui::Frame::window(ui.style())
                                    .fill(egui::Color32::from_rgba_premultiplied(24, 24, 24, 238))
                                    .rounding(egui::Rounding::same(8.0))
                                    .inner_margin(egui::Margin::same(10.0))
                                    .show(ui, |ui| {
                                        ui.set_max_width(260.0);
                                        ui.horizontal(|ui| {
                                            ui.label(
                                                egui::RichText::new("Keyboard shortcuts")
                                                    .strong()
                                                    .color(egui::Color32::from_gray(230)),
                                            );
                                            ui.with_layout(
                                                egui::Layout::right_to_left(egui::Align::Center),
                                                |ui| {
                                                    if ui.button("×").clicked() {
                                                        *show_cull_help = false;
                                                        if let Some(db) = db {
                                                            save_bool_pref(db, CULL_HELP_KEY, false);
                                                        }
                                                    }
                                                },
                                            );
                                        });
                                        ui.add_space(4.0);
                                        ui.label("Rate: `1-5`, `0` clear");
                                        ui.label("Flag: `P` pick, `X` reject, `Space` pick+next");
                                        ui.label("Label: `6` red, `7` yellow, `8` green, `9` blue");
                                        ui.label("Move: `←→` / `↑↓`, `Enter` opens");
                                        ui.add_space(4.0);
                                        ui.label(
                                            egui::RichText::new("Bulk actions stay in the top toolbar.")
                                                .small()
                                                .color(egui::Color32::from_gray(150)),
                                        );
                                    });
                            } else if ui.small_button("? Keys").clicked() {
                                *show_cull_help = true;
                                if let Some(db) = db {
                                    save_bool_pref(db, CULL_HELP_KEY, true);
                                }
                            }
                        });

                    egui::SidePanel::left("folder_tree")
                        .default_width(200.0)
                        .width_range(120.0..=380.0)
                        .frame(egui::Frame::none()
                            .fill(egui::Color32::from_gray(18))
                            .inner_margin(egui::Margin::symmetric(4.0, 4.0)))
                        .show(ctx, |ui| {
                            ui.spacing_mut().item_spacing.y = 2.0;
                            egui::ScrollArea::vertical().show(ui, |ui| {
                                let home = std::env::var_os("HOME")
                                    .map(PathBuf::from)
                                    .unwrap_or_default();
                                show_dir_item(ui, &home, 0, tree_expanded, browse_dir.as_deref(), ctx);
                                ui.add_space(4.0);
                                if let Ok(rd) = std::fs::read_dir("/Volumes") {
                                    let mut vols: Vec<PathBuf> = rd
                                        .filter_map(std::result::Result::ok)
                                        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
                                        .map(|e| e.path())
                                        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|s| !s.starts_with('.')))
                                        .collect();
                                    vols.sort_unstable();
                                    for vol in vols {
                                        show_dir_item(ui, &vol, 0, tree_expanded, browse_dir.as_deref(), ctx);
                                    }
                                }
                            });
                        });

                    egui::CentralPanel::default()
                        .frame(egui::Frame::none().fill(egui::Color32::from_gray(24)))
                        .show(ctx, |ui| {
                            ui.add_space(6.0);
                            ui.horizontal(|ui| {
                                ui.add_space(6.0);
                                if ui.button("Open Folder…").clicked() {
                                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                                        ctx.data_mut(|d| {
                                            d.insert_temp(egui::Id::new(KEY_SCAN_DIR), dir);
                                        });
                                    }
                                }
                                if let Some(dir) = browse_dir {
                                    ui.label(
                                        egui::RichText::new(dir.display().to_string())
                                            .small()
                                            .color(egui::Color32::from_gray(140)),
                                    );
                                }
                            });
                            ui.add_space(6.0);
                            ui.separator();

                            ui.horizontal_wrapped(|ui| {
                                ui.add_space(6.0);
                                for (f, lbl) in [
                                    (BrowseFilter::All, "All"),
                                    (BrowseFilter::Picks, "Picks"),
                                    (BrowseFilter::Rejects, "Rejects"),
                                    (BrowseFilter::Unflagged, "Unflagged"),
                                ] {
                                    if ui.selectable_label(*filter == f, lbl).clicked() {
                                        *filter = f;
                                    }
                                }
                                ui.separator();
                                ui.label(egui::RichText::new("★ ≥").small());
                                egui::ComboBox::from_id_salt("min_rating")
                                    .selected_text(if *min_rating == 0 { "Any".to_string() } else { "★".repeat(*min_rating as usize) })
                                    .show_ui(ui, |ui| {
                                        for (v, lbl) in [(0u8, "Any"), (1, "★"), (2, "★★"), (3, "★★★"), (4, "★★★★"), (5, "★★★★★")] {
                                            ui.selectable_value(min_rating, v, lbl);
                                        }
                                    });
                                ui.separator();
                                ui.label(egui::RichText::new("Sort").small());
                                for (s, lbl) in [(BrowseSort::Name, "Name"), (BrowseSort::Date, "Date"), (BrowseSort::CaptureTime, "Time")] {
                                    if ui.selectable_label(*sort == s, lbl).clicked() {
                                        *sort = s;
                                    }
                                }
                                ui.separator();
                                ui.label(
                                    egui::RichText::new(format!(
                                        "{} shown · {n_picks} picks · {n_rejects} rejects",
                                        visible.len()
                                    ))
                                    .small()
                                    .color(egui::Color32::from_gray(140)),
                                );
                                ui.separator();
                                if ui
                                    .add_enabled(n_rejects > 0, egui::Button::new(format!("Trash Rejects ({n_rejects})")))
                                    .clicked()
                                {
                                    *confirm_trash = true;
                                }
                                if ui
                                    .add_enabled(n_rejects > 0, egui::Button::new("Move Rejects..."))
                                    .clicked()
                                {
                                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                                        ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_MOVE_REJECTS), dir));
                                    }
                                }
                                if ui
                                    .add_enabled(n_picks > 0, egui::Button::new(format!("Copy Picks ({n_picks})...")))
                                    .clicked()
                                {
                                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                                        ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_COPY_PICKS), dir));
                                    }
                                }
                                ui.separator();
                                ui.menu_button("🤖 AI Actions", |ui| {
                                    ui.set_min_width(190.0);
                                    if let Some((done, total)) = classify_progress {
                                        ui.label(format!("Tagging {done}/{total}…"));
                                    } else if ui
                                        .add_enabled(
                                            classifier_available && !thumbs.is_empty(),
                                            egui::Button::new("🏷 Tag Folder"),
                                        )
                                        .on_hover_text("Classify every photo in this folder and add tags")
                                        .clicked()
                                    {
                                        ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_CLASSIFY_FOLDER), true));
                                        ui.close_menu();
                                    }
                                    if let Some((done, total)) = rate_progress {
                                        ui.label(format!("Rating {done}/{total}…"));
                                    } else if ui
                                        .add_enabled(
                                            rater_available && !thumbs.is_empty(),
                                            egui::Button::new("⭐ Rate Folder"),
                                        )
                                        .on_hover_text("Auto-rate every photo in this folder by quality (1-5 stars)")
                                        .clicked()
                                    {
                                        ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_RATE_FOLDER), true));
                                        ui.close_menu();
                                    }
                                    if let Some((done, total)) = adjust_progress {
                                        ui.label(format!("Adjusting {done}/{total}…"));
                                    } else if ui
                                        .add_enabled(
                                            enhancer_available && !thumbs.is_empty(),
                                            egui::Button::new("✨ Auto Adjust Folder"),
                                        )
                                        .on_hover_text("Apply AI color and tone adjustment to every photo")
                                        .clicked()
                                    {
                                        ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_ADJUST_FOLDER), true));
                                        ui.close_menu();
                                    }
                                    if let Some((done, total)) = similar_progress {
                                        ui.label(format!("Grouping {done}/{total}…"));
                                    } else if ui
                                        .add_enabled(
                                            similar_available && !thumbs.is_empty(),
                                            egui::Button::new("✨ Cull Similar"),
                                        )
                                        .on_hover_text("Group near-duplicate photos, then keep the best-rated frame; face size breaks ties")
                                        .clicked()
                                    {
                                        ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_SIMILAR_CULL), true));
                                        ui.close_menu();
                                    }
                                });
                                if let Some((done, total)) = classify_progress {
                                    ui.label(
                                        egui::RichText::new(format!("Tagging {done}/{total}…"))
                                            .small()
                                            .color(egui::Color32::from_gray(160)),
                                    );
                                } else if let Some((done, total)) = rate_progress {
                                    ui.label(
                                        egui::RichText::new(format!("Rating {done}/{total}…"))
                                            .small()
                                            .color(egui::Color32::from_gray(160)),
                                    );
                                } else if let Some((done, total)) = adjust_progress {
                                    ui.label(
                                        egui::RichText::new(format!("Adjusting {done}/{total}…"))
                                            .small()
                                            .color(egui::Color32::from_gray(160)),
                                    );
                                } else if let Some((done, total)) = similar_progress {
                                    ui.label(
                                        egui::RichText::new(format!("Grouping {done}/{total}…"))
                                            .small()
                                            .color(egui::Color32::from_gray(160)),
                                    );
                                }
                                if let Some((groups, rejected)) = similar_summary {
                                    ui.label(
                                        egui::RichText::new(format!("{groups} groups · {rejected} rejected"))
                                            .small()
                                            .color(egui::Color32::from_gray(140)),
                                    );
                                }
                            });
                            ui.separator();

                            if thumbs.is_empty() || visible.is_empty() {
                                ui.centered_and_justified(|ui| {
                                    let msg = if thumbs.is_empty() {
                                        "Open a folder (or an image) to browse thumbnails"
                                    } else {
                                        "No images match the filter"
                                    };
                                    ui.label(egui::RichText::new(msg).color(egui::Color32::from_gray(140)));
                                });
                                return;
                            }

                            egui::ScrollArea::vertical().show(ui, |ui| {
                                ui.add_space(6.0);
                                ui.horizontal_wrapped(|ui| {
                                    let cell = egui::vec2(168.0, 152.0);
                                    *grid_cols = (ui.available_width() / cell.x).max(1.0) as usize;
                                    for &ti in &visible {
                                        let entry = &thumbs[ti];
                                        let (rect, resp) =
                                            ui.allocate_exact_size(cell, egui::Sense::click());
                                        if *grid_scroll && selected.as_deref() == Some(entry.path.as_path()) {
                                            ui.scroll_to_rect(rect, Some(egui::Align::Center));
                                            *grid_scroll = false;
                                        }
                                        if !ui.is_rect_visible(rect) {
                                            continue;
                                        }
                                        let p = ui.painter_at(rect);
                                        p.rect_filled(rect.shrink(2.0), 4.0, egui::Color32::from_gray(16));

                                        let img_area = egui::Rect::from_min_max(
                                            rect.min + egui::vec2(6.0, 6.0),
                                            egui::pos2(rect.max.x - 6.0, rect.max.y - 36.0),
                                        );
                                        if let Some(tex) = &entry.tex {
                                            let ts = tex.size_vec2();
                                            let s = (img_area.width() / ts.x)
                                                .min(img_area.height() / ts.y)
                                                .min(1.0);
                                            let ir = egui::Rect::from_center_size(
                                                img_area.center(),
                                                ts * s,
                                            );
                                            p.image(
                                                tex.id(),
                                                ir,
                                                egui::Rect::from_min_max(
                                                    egui::pos2(0.0, 0.0),
                                                    egui::pos2(1.0, 1.0),
                                                ),
                                                egui::Color32::WHITE,
                                            );
                                        } else {
                                            p.text(
                                                img_area.center(),
                                                egui::Align2::CENTER_CENTER,
                                                "…",
                                                egui::FontId::proportional(18.0),
                                                egui::Color32::from_gray(90),
                                            );
                                        }

                                        let meta = meta_map.get(&entry.path).copied().unwrap_or_default();
                                        if meta.flag == cull::Flag::Reject {
                                            p.rect_filled(
                                                img_area,
                                                0.0,
                                                egui::Color32::from_rgba_premultiplied(140, 20, 20, 48),
                                            );
                                        }
                                        paint_badges(&p, img_area, meta);

                                        let name = entry
                                            .path
                                            .file_name()
                                            .and_then(|n| n.to_str())
                                            .unwrap_or("");
                                        let mut label = name.to_owned();
                                        if label.len() > 22 {
                                            label.truncate(20);
                                            label.push('…');
                                        }
                                        // Row 1: filename
                                        p.text(
                                            egui::pos2(rect.center().x, rect.max.y - 24.0),
                                            egui::Align2::CENTER_CENTER,
                                            label,
                                            egui::FontId::proportional(10.0),
                                            egui::Color32::from_gray(170),
                                        );
                                        // Row 2: EXIF (left) + time (right)
                                        let exif_line: String = [
                                            entry.exif.shutter.clone(),
                                            entry.exif.aperture.map(|(n, d)| {
                                                let whole = n / d;
                                                let tenths = (u64::from(n) * 10 / u64::from(d)) % 10;
                                                if tenths < 1 { format!("f/{whole}") } else { format!("f/{whole}.{tenths}") }
                                            }),
                                            entry.exif.iso.map(|i| format!("ISO{i}")),
                                        ].into_iter().flatten().collect::<Vec<_>>().join("  ");
                                        if !exif_line.is_empty() {
                                            p.text(
                                                egui::pos2(rect.min.x + 6.0, rect.max.y - 11.0),
                                                egui::Align2::LEFT_CENTER,
                                                exif_line,
                                                egui::FontId::proportional(9.0),
                                                egui::Color32::from_gray(120),
                                            );
                                        }
                                        if let Some(ct) = entry.exif.capture_time {
                                            p.text(
                                                egui::pos2(rect.max.x - 6.0, rect.max.y - 11.0),
                                                egui::Align2::RIGHT_CENTER,
                                                format!("{}-{:02}-{:02} {:02}:{:02}", ct / 10_000_000_000, (ct / 100_000_000) % 100, (ct / 1_000_000) % 100, (ct / 10_000) % 100, (ct / 100) % 100),
                                                egui::FontId::proportional(9.0),
                                                egui::Color32::from_gray(120),
                                            );
                                        }

                                        let selected_item = selected
                                            .as_deref()
                                            .or(current_path.as_deref())
                                            == Some(entry.path.as_path());
                                        if selected_item {
                                            p.rect_stroke(
                                                rect.shrink(2.0),
                                                4.0,
                                                egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(90, 140, 255)),
                                            );
                                        } else if resp.hovered() {
                                            p.rect_stroke(
                                                rect.shrink(2.0),
                                                4.0,
                                                egui::Stroke::new(1.0_f32, egui::Color32::from_gray(110)),
                                            );
                                        }

                                        if resp.clicked() {
                                            *selected = Some(entry.path.clone());
                                        }
                                        if resp.double_clicked() {
                                            *selected = Some(entry.path.clone());
                                            ctx.data_mut(|d| {
                                                d.insert_temp(
                                                    egui::Id::new(KEY_OPEN_PATH),
                                                    entry.path.clone(),
                                                );
                                            });
                                        }
                                    }
                                });
                                ui.add_space(6.0);
                            });
                        });
                    return;
                }

                egui::SidePanel::right("controls")
                    .exact_width(260.0)
                    .resizable(false)
                    .show(ctx, |ui| {
                        const MIN_DX: f32 = 0.01;
                        const GAMMA_LOG_RANGE: f32 = 1.609_438; // ln(5): gamma handle maps 0.2..5
                        // Luminance histogram + curves editor
                        let hist_size = egui::vec2(ui.available_width(), 150.0);
                        let (hist_rect, hist_resp) =
                            ui.allocate_exact_size(hist_size, egui::Sense::click_and_drag());
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

                        // ---- Curves editor (Photoshop-style) overlaid on the histogram ----
                        let to_screen = |p: [f32; 2]| {
                            egui::pos2(
                                hist_rect.left() + p[0] * hist_rect.width(),
                                hist_rect.bottom() - p[1] * hist_rect.height(),
                            )
                        };
                        let to_norm = |p: egui::Pos2| {
                            [
                                ((p.x - hist_rect.left()) / hist_rect.width()).clamp(0.0, 1.0),
                                ((hist_rect.bottom() - p.y) / hist_rect.height()).clamp(0.0, 1.0),
                            ]
                        };
                        let nearest_point = |pts: &[[f32; 2]], mp: egui::Pos2| {
                            pts.iter()
                                .enumerate()
                                .map(|(i, &p)| (i, to_screen(p).distance(mp)))
                                .min_by(|a, b| a.1.total_cmp(&b.1))
                                .filter(|(_, d)| *d < 10.0)
                                .map(|(i, _)| i)
                        };

                        // Grab an existing point or create one (click or drag start)
                        if hist_resp.drag_started() || hist_resp.clicked() {
                            if let Some(mp) = hist_resp.interact_pointer_pos() {
                                let pts = &mut processor.curve_points;
                                let idx = nearest_point(pts, mp).unwrap_or_else(|| {
                                    let np = to_norm(mp);
                                    let idx = pts.iter().position(|p| p[0] > np[0]).unwrap_or(pts.len());
                                    pts.insert(idx, np);
                                    needs_process = true;
                                    idx
                                });
                                *curve_drag = Some(idx);
                            }
                        }
                        if hist_resp.dragged() {
                            if let (Some(i), Some(mp)) = (*curve_drag, hist_resp.interact_pointer_pos()) {
                                let pts = &mut processor.curve_points;
                                let np = to_norm(mp);
                                let last = pts.len() - 1;
                                let x = if i == 0 {
                                    np[0].min(pts[1][0] - MIN_DX).max(0.0)
                                } else if i == last {
                                    np[0].max(pts[last - 1][0] + MIN_DX).min(1.0)
                                } else {
                                    np[0].clamp(pts[i - 1][0] + MIN_DX, pts[i + 1][0] - MIN_DX)
                                };
                                pts[i] = [x, np[1]];
                                needs_process = true;
                            }
                        }
                        if hist_resp.drag_stopped() {
                            // Releasing a point well outside the box deletes it (endpoints stay)
                            if let (Some(i), Some(mp)) = (*curve_drag, ctx.pointer_interact_pos()) {
                                let pts = &mut processor.curve_points;
                                if i != 0 && i != pts.len() - 1 && !hist_rect.expand(25.0).contains(mp) {
                                    pts.remove(i);
                                    needs_process = true;
                                }
                            }
                            *curve_drag = None;
                        }
                        // Right-click removes a point; double-click resets the whole curve
                        if hist_resp.secondary_clicked() {
                            if let Some(mp) = hist_resp.interact_pointer_pos() {
                                let pts = &mut processor.curve_points;
                                if let Some(i) = nearest_point(pts, mp) {
                                    if i != 0 && i != pts.len() - 1 {
                                        pts.remove(i);
                                        needs_process = true;
                                    }
                                }
                            }
                        }
                        if hist_resp.double_clicked() {
                            processor.curve_points = vec![[0.0, 0.0], [1.0, 1.0]];
                            *curve_drag = None;
                            needs_process = true;
                        }

                        // Quarter grid lines
                        let grid_stroke = egui::Stroke::new(1.0_f32, egui::Color32::from_gray(34));
                        for f in [0.25f32, 0.5, 0.75] {
                            let x = hist_rect.left() + f * hist_rect.width();
                            let y = hist_rect.top() + f * hist_rect.height();
                            painter.line_segment([egui::pos2(x, hist_rect.top()), egui::pos2(x, hist_rect.bottom())], grid_stroke);
                            painter.line_segment([egui::pos2(hist_rect.left(), y), egui::pos2(hist_rect.right(), y)], grid_stroke);
                        }

                        // The curve itself, sampled from the same LUT the shader uses
                        let lut = processor.curve_lut();
                        let curve_pts: Vec<egui::Pos2> = (0..=128)
                            .map(|i| {
                                let t = i as f32 / 128.0;
                                to_screen([t, lut[((t * 255.0) as usize).min(255)]])
                            })
                            .collect();
                        painter.add(egui::Shape::line(
                            curve_pts,
                            egui::Stroke::new(1.5_f32, egui::Color32::from_gray(230)),
                        ));

                        // Control points
                        let hover_idx = hist_resp
                            .hover_pos()
                            .and_then(|mp| nearest_point(&processor.curve_points, mp));
                        for (i, &p) in processor.curve_points.iter().enumerate() {
                            let hot = *curve_drag == Some(i) || (curve_drag.is_none() && hover_idx == Some(i));
                            let (r, fill) = if hot {
                                (4.5, egui::Color32::WHITE)
                            } else {
                                (3.5, egui::Color32::from_gray(200))
                            };
                            painter.circle(to_screen(p), r, fill, egui::Stroke::new(1.0_f32, egui::Color32::from_gray(60)));
                        }
                        if hover_idx.is_some() || curve_drag.is_some() {
                            ctx.set_cursor_icon(egui::CursorIcon::Grab);
                        }

                        // Levels: gradient strip with draggable handles (black / gamma / white)
                        let strip_h  = 10.0;
                        let marker_h = 12.0;
                        let (strip_area, strip_resp) = ui.allocate_exact_size(
                            egui::vec2(ui.available_width(), strip_h + marker_h),
                            egui::Sense::click_and_drag(),
                        );
                        let sp = ui.painter_at(strip_area);
                        let gradient_rect = egui::Rect::from_min_size(strip_area.min, egui::vec2(strip_area.width(), strip_h));

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
                            egui::epaint::Vertex { pos: gradient_rect.left_top(),     uv, color: egui::Color32::BLACK },
                            egui::epaint::Vertex { pos: gradient_rect.right_top(),    uv, color: egui::Color32::WHITE },
                            egui::epaint::Vertex { pos: gradient_rect.right_bottom(), uv, color: egui::Color32::WHITE },
                            egui::epaint::Vertex { pos: gradient_rect.left_bottom(),  uv, color: egui::Color32::BLACK },
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
                        let ty = gradient_rect.bottom();
                        let by = strip_area.bottom();
                        let mk = |cx: f32, fill: egui::Color32, hot: bool| {
                            let cx = cx.clamp(strip_area.left() + 6.0, strip_area.right() - 6.0);
                            let stroke = if hot {
                                egui::Stroke::new(1.5_f32, egui::Color32::from_gray(220))
                            } else {
                                egui::Stroke::new(1.0_f32, egui::Color32::from_gray(110))
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

                        ui.add_space(6.0);
                        ui.horizontal(|ui| {
                            if ui.button("Open Image…").clicked() {
                                if let Some(path) = rfd::FileDialog::new()
                                    .add_filter("Images", &imgload::all_exts())
                                    .pick_file()
                                {
                                    ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_OPEN_PATH), path));
                                }
                            }
                            if ui
                                .add_enabled(processor.has_image(), egui::Button::new("Auto"))
                                .on_hover_text("Auto levels + brightness/shadows/highlights")
                                .clicked()
                            {
                                ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_AUTO), true));
                            }
                        });
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            let path = current_path.clone();
                            if let Some(path) = path {
                                ui.label(egui::RichText::new("Rating").small().color(egui::Color32::from_gray(140)));
                                for rating in 1..=5u8 {
                                    let filled = meta_map.get(&path).is_some_and(|m| m.rating >= rating);
                                    let label = if filled { "★" } else { "☆" };
                                    if ui.selectable_label(filled, label).clicked() {
                                        cull_actions.push((path.clone(), cull::CullAction::Rating(rating)));
                                    }
                                }
                                if ui.button("0").clicked() {
                                    cull_actions.push((path.clone(), cull::CullAction::Rating(0)));
                                }
                                ui.separator();
                                let pick = meta_map.get(&path).is_some_and(|m| m.flag == cull::Flag::Pick);
                                if ui.selectable_label(pick, "P").clicked() {
                                    cull_actions.push((path.clone(), cull::CullAction::TogglePick));
                                }
                                let reject = meta_map.get(&path).is_some_and(|m| m.flag == cull::Flag::Reject);
                                if ui.selectable_label(reject, "X").clicked() {
                                    cull_actions.push((path.clone(), cull::CullAction::ToggleReject));
                                }
                                ui.separator();
                                for label in [cull::Label::Red, cull::Label::Yellow, cull::Label::Green, cull::Label::Blue] {
                                    let c = label_color(label).unwrap();
                                    let dot = egui::RichText::new("●").color(c);
                                    let active = meta_map.get(&path).is_some_and(|m| m.label == label);
                                    if ui.selectable_label(active, dot).clicked() {
                                        cull_actions.push((path.clone(), cull::CullAction::ToggleLabel(label)));
                                    }
                                }
                            }
                        });
                        ui.separator();
                        if let Some(path) = current_path.clone() {
                            ui.horizontal(|ui| {
                                ui.label(egui::RichText::new("TAGS").small().color(egui::Color32::from_gray(140)));
                                let resp = ui.add(
                                    egui::TextEdit::singleline(tag_edit)
                                        .hint_text("comma, separated, tags")
                                        .desired_width(ui.available_width()),
                                );
                                if resp.changed() {
                                    let parsed: Vec<String> = tag_edit
                                        .split(',')
                                        .map(str::trim)
                                        .filter(|s| !s.is_empty())
                                        .map(str::to_string)
                                        .collect();
                                    if let Some(db) = db {
                                        tags::save_tags(db, &path, &parsed);
                                    }
                                    if parsed.is_empty() {
                                        tags_map.remove(&path);
                                    } else {
                                        tags_map.insert(path.clone(), parsed);
                                    }
                                }
                            });
                            ui.separator();
                        }
                        ui.spacing_mut().item_spacing.y = 3.0;

                        // Compact one-line rows: fixed-width label left, slider right
                        macro_rules! slider_row {
                            ($label:expr, $field:expr, $range:expr, $default:expr, $integer:expr) => {{
                                ui.horizontal(|ui| {
                                    // Fixed-size label slot painted directly, so every
                                    // slider starts and ends at the same x.
                                    let (lrect, _) = ui.allocate_exact_size(
                                        egui::vec2(74.0, 18.0),
                                        egui::Sense::hover(),
                                    );
                                    ui.painter().text(
                                        lrect.left_center(),
                                        egui::Align2::LEFT_CENTER,
                                        $label,
                                        egui::FontId::proportional(10.0),
                                        egui::Color32::from_gray(140),
                                    );
                                    ui.spacing_mut().interact_size.x = 46.0;
                                    ui.spacing_mut().slider_width =
                                        ui.available_width() - 46.0 - ui.spacing().item_spacing.x * 2.0;
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
                                });
                            }};
                        }

                        slider_row!("TEMP",       processor.wb_temp,    -100.0..=100.0, 0.0, true);
                        slider_row!("TINT",       processor.wb_tint,    -100.0..=100.0, 0.0, true);
                        ui.separator();
                        slider_row!("EXPOSURE",   processor.exposure,   -3.0..=3.0,     0.0, false);
                        slider_row!("BRIGHTNESS", processor.brightness, -100.0..=100.0, 0.0, true);
                        slider_row!("CONTRAST",   processor.contrast,   -100.0..=100.0, 0.0, true);
                        ui.separator();
                        slider_row!("BLACKS",     processor.blacks,     -100.0..=100.0, 0.0, true);
                        slider_row!("SHADOWS",    processor.shadows,    -100.0..=100.0, 0.0, true);
                        slider_row!("HIGHLIGHTS", processor.highlights, -100.0..=100.0, 0.0, true);
                        slider_row!("WHITES",     processor.whites,     -100.0..=100.0, 0.0, true);
                        ui.separator();
                        slider_row!("BLUR",       processor.blur_radius,         0.0..=15.0, 0.0, true);
                        slider_row!("SHARPEN",    processor.unsharp_strength,    0.0..=3.0,  0.0, false);
                        slider_row!("SHARP RAD",  processor.unsharp_blur_radius, 1.0..=10.0, 2.0, true);
                        ui.separator();
                        slider_row!("VIGNETTE",   processor.vignette,     -100.0..=100.0, 0.0,  true);
                        slider_row!("VIG MID",    processor.vignette_mid, 0.0..=100.0,    50.0, true);
                        ui.separator();

                        ui.label(egui::RichText::new("PRESET").small().color(egui::Color32::from_gray(140)));
                        ui.horizontal(|ui| {
                            egui::ComboBox::from_id_salt("preset_picker")
                                .selected_text(if preset_name.is_empty() { "Choose…" } else { preset_name.as_str() })
                                .width(150.0)
                                .show_ui(ui, |ui| {
                                    for name in presets.iter() {
                                        if ui.selectable_label(preset_name == name, name).clicked() {
                                            preset_name.clone_from(name);
                                            if let Some(state) = presets_dir.and_then(|d| presets::load(d, name)) {
                                                processor.apply_edit_state(&state);
                                                needs_process = true;
                                            }
                                        }
                                    }
                                });
                            let can_delete = presets.iter().any(|n| n == preset_name);
                            if ui.add_enabled(can_delete, egui::Button::new("🗑")).clicked() {
                                if let Some(dir) = presets_dir {
                                    let _ = presets::delete(dir, preset_name);
                                    *presets = presets::list(dir);
                                    preset_name.clear();
                                }
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::TextEdit::singleline(preset_name)
                                    .hint_text("name…")
                                    .desired_width(150.0),
                            );
                            if ui
                                .add_enabled(
                                    processor.has_image() && !preset_name.trim().is_empty(),
                                    egui::Button::new("Save"),
                                )
                                .clicked()
                            {
                                if let Some(dir) = presets_dir {
                                    if presets::save(dir, preset_name.trim(), &processor.edit_state()).is_ok() {
                                        *presets = presets::list(dir);
                                    }
                                }
                            }
                        });

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
                            // bottom_up layout: added after Export, so it sits above it
                            if ui.add_enabled(
                                processor.has_image(),
                                egui::Button::new("Reset All Edits").min_size(egui::vec2(228.0, 0.0)),
                            ).clicked() {
                                processor.apply_edit_state(&EditState::default());
                                needs_process = true; // re-process + persist the reset
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

                if *view == View::Edit {
                    egui::TopBottomPanel::bottom("filmstrip")
                        .exact_height(92.0)
                        .frame(egui::Frame::none().fill(egui::Color32::from_gray(20)))
                        .show(ctx, |ui| {
                            egui::ScrollArea::horizontal().show(ui, |ui| {
                                ui.add_space(4.0);
                                ui.horizontal(|ui| {
                                    for &ti in &visible {
                                        let entry = &thumbs[ti];
                                        let (rect, resp) = ui.allocate_exact_size(
                                            egui::vec2(104.0, 80.0),
                                            egui::Sense::click(),
                                        );
                                        let is_cur = current_path.as_deref() == Some(entry.path.as_path());
                                        if is_cur && *strip_scroll {
                                            ui.scroll_to_rect(rect, Some(egui::Align::Center));
                                            *strip_scroll = false;
                                        }
                                        if !ui.is_rect_visible(rect) {
                                            continue;
                                        }
                                        let p = ui.painter_at(rect);
                                        p.rect_filled(rect.shrink(2.0), 3.0, egui::Color32::from_gray(14));
                                        let img_area = rect.shrink(4.0);
                                        if let Some(tex) = &entry.tex {
                                            let ts = tex.size_vec2();
                                            let s = (img_area.width() / ts.x)
                                                .min(img_area.height() / ts.y)
                                                .min(1.0);
                                            let ir = egui::Rect::from_center_size(img_area.center(), ts * s);
                                            p.image(
                                                tex.id(),
                                                ir,
                                                egui::Rect::from_min_max(
                                                    egui::pos2(0.0, 0.0),
                                                    egui::pos2(1.0, 1.0),
                                                ),
                                                egui::Color32::WHITE,
                                            );
                                        } else {
                                            p.text(
                                                img_area.center(),
                                                egui::Align2::CENTER_CENTER,
                                                "…",
                                                egui::FontId::proportional(14.0),
                                                egui::Color32::from_gray(90),
                                            );
                                        }
                                        paint_badges(
                                            &p,
                                            img_area,
                                            meta_map.get(&entry.path).copied().unwrap_or_default(),
                                        );
                                        if is_cur {
                                            p.rect_stroke(
                                                rect.shrink(2.0),
                                                3.0,
                                                egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(90, 140, 255)),
                                            );
                                        } else if resp.hovered() {
                                            p.rect_stroke(
                                                rect.shrink(2.0),
                                                3.0,
                                                egui::Stroke::new(1.0_f32, egui::Color32::from_gray(110)),
                                            );
                                        }
                                        if resp.clicked() {
                                            *selected = Some(entry.path.clone());
                                            ctx.data_mut(|d| {
                                                d.insert_temp(
                                                    egui::Id::new(KEY_OPEN_PATH),
                                                    entry.path.clone(),
                                                );
                                            });
                                        }
                                    }
                                });
                            });
                        });
                }

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
                            .is_some_and(|t| now - t > 0.3);
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
            let open_path: Option<PathBuf> = self
                .egui_ctx
                .data_mut(|d| d.remove_temp(egui::Id::new(KEY_OPEN_PATH)));
            let export_path: Option<PathBuf> = self
                .egui_ctx
                .data_mut(|d| d.remove_temp(egui::Id::new(KEY_EXPORT_PATH)));
            let scan_dir: Option<PathBuf> = self
                .egui_ctx
                .data_mut(|d| d.remove_temp(egui::Id::new(KEY_SCAN_DIR)));
            let auto_req: Option<bool> = self
                .egui_ctx
                .data_mut(|d| d.remove_temp(egui::Id::new(KEY_AUTO)));
            let trash_req: Option<bool> = self
                .egui_ctx
                .data_mut(|d| d.remove_temp(egui::Id::new(KEY_TRASH_REJECTS)));
            let move_rejects_dir: Option<PathBuf> = self
                .egui_ctx
                .data_mut(|d| d.remove_temp(egui::Id::new(KEY_MOVE_REJECTS)));
            let copy_picks_dir: Option<PathBuf> = self
                .egui_ctx
                .data_mut(|d| d.remove_temp(egui::Id::new(KEY_COPY_PICKS)));
            let classify_folder_req: Option<bool> = self
                .egui_ctx
                .data_mut(|d| d.remove_temp(egui::Id::new(KEY_CLASSIFY_FOLDER)));
            let rate_folder_req: Option<bool> = self
                .egui_ctx
                .data_mut(|d| d.remove_temp(egui::Id::new(KEY_RATE_FOLDER)));
            let adjust_folder_req: Option<bool> = self
                .egui_ctx
                .data_mut(|d| d.remove_temp(egui::Id::new(KEY_ADJUST_FOLDER)));
            let similar_cull_req: Option<bool> = self
                .egui_ctx
                .data_mut(|d| d.remove_temp(egui::Id::new(KEY_SIMILAR_CULL)));
            (
                full_output.shapes,
                full_output.textures_delta,
                full_output.pixels_per_point,
                open_path,
                export_path,
                scan_dir,
                auto_req.is_some(),
                trash_req.is_some(),
                move_rejects_dir,
                copy_picks_dir,
                classify_folder_req.is_some(),
                rate_folder_req.is_some(),
                adjust_folder_req.is_some(),
                similar_cull_req.is_some(),
                needs_process,
            )
        }; // all field borrows dropped here

        for (path, action) in cull_actions {
            self.apply_cull_action(&path, action);
        }
        if trash_req {
            let rejects = self.flagged_paths(cull::Flag::Reject);
            if trash::delete_all(&rejects).is_ok() {
                if let Some(db) = &self.db {
                    for path in &rejects {
                        cull::delete_rows(db, path);
                        tags::delete_rows(db, path);
                    }
                }
                self.after_files_removed(&rejects);
            }
        }
        if let Some(dest) = move_rejects_dir {
            let rejects = self.flagged_paths(cull::Flag::Reject);
            let mut moved = Vec::new();
            for path in &rejects {
                let Some(name) = path.file_name() else {
                    continue;
                };
                let new_path = dest.join(name);
                if move_file(path, &new_path).is_ok() {
                    if let Some(db) = &self.db {
                        cull::rekey_rows(db, path, &new_path);
                        tags::rekey_rows(db, path, &new_path);
                    }
                    moved.push(path.clone());
                }
            }
            self.after_files_removed(&moved);
        }
        if let Some(dest) = copy_picks_dir {
            let picks = self.flagged_paths(cull::Flag::Pick);
            for path in &picks {
                let Some(name) = path.file_name() else {
                    continue;
                };
                let new_path = dest.join(name);
                if std::fs::copy(path, &new_path).is_ok() {
                    if let Some(db) = &self.db {
                        cull::copy_rows(db, path, &new_path);
                        tags::copy_rows(db, path, &new_path);
                    }
                }
            }
        }

        // File ops
        if let Some(dir) = scan_dir {
            self.scan_folder(&dir);
        }
        if let Some(path) = open_path {
            self.register_image(&path);
        }
        if let Some(path) = export_path {
            if let (Some(proc), Some(gpu)) = (&self.processor, &self.gpu) {
                proc.export(&path, &gpu.device, &gpu.queue);
            }
        }
        if classify_folder_req {
            let paths: Vec<PathBuf> = self.thumbs.iter().map(|t| t.path.clone()).collect();
            if let (Some(classifier), false) = (&self.classifier, paths.is_empty()) {
                self.classify_progress = Some((0, paths.len()));
                let classifier = Arc::clone(classifier);
                let work = move |path: &Path| {
                    imgload::load_edit_rgba(path).map(|(img, _)| classifier.classify(&img))
                };
                spawn_folder_workers(paths, work, self.classify_tx.clone());
            }
        }
        if rate_folder_req {
            let paths: Vec<PathBuf> = self.thumbs.iter().map(|t| t.path.clone()).collect();
            if let (Some(rater), false) = (&self.rater, paths.is_empty()) {
                self.rate_progress = Some((0, paths.len()));
                let rater = Arc::clone(rater);
                let work = move |path: &Path| {
                    imgload::load_edit_rgba(path).and_then(|(img, _)| rater.rate(&img))
                };
                spawn_folder_workers(paths, work, self.rate_tx.clone());
            }
        }
        if adjust_folder_req {
            let paths: Vec<PathBuf> = self.thumbs.iter().map(|t| t.path.clone()).collect();
            if let (Some(enhancer), false) = (&self.enhancer, paths.is_empty()) {
                self.adjust_progress = Some((0, paths.len()));
                let enhancer = Arc::clone(enhancer);
                let work = move |path: &Path| {
                    imgload::load_rgba(path, 512).and_then(|img| enhancer.suggest_edits(&img))
                };
                spawn_folder_workers(paths, work, self.adjust_tx.clone());
            }
        }
        if similar_cull_req {
            let paths: Vec<PathBuf> = self.thumbs.iter().map(|t| t.path.clone()).collect();
            if let (Some(classifier), Some(rater), Some(detector), false) = (
                &self.classifier,
                &self.rater,
                &self.face_detector,
                paths.is_empty(),
            ) {
                self.similar_candidates.clear();
                self.similar_summary = None;
                self.similar_progress = Some((0, paths.len()));
                let classifier = Arc::clone(classifier);
                let rater = Arc::clone(rater);
                let detector = Arc::clone(detector);
                let existing_meta = self.meta.clone();
                let work = move |path: &Path| {
                    let img = imgload::load_rgba(path, 512)?;
                    let embedding = classifier.embedding(&img)?;
                    let model_rating = rater.rate(&img)?;
                    let rating = existing_meta
                        .get(path)
                        .map_or(model_rating, |meta| meta.rating.max(model_rating));
                    let faces = detector.detect(&img);
                    Some(similar::Analysis {
                        embedding,
                        rating,
                        face_count: faces.count,
                        largest_face: faces.largest_area,
                    })
                };
                spawn_folder_workers(paths, work, self.similar_tx.clone());
            }
        }

        // Auto adjust: reset to neutral, refresh the histogram from the
        // original image, derive AI or histogram-based values, then let the
        // normal needs_process path re-process and persist them.
        if auto_req {
            let ai_state = self
                .current_path
                .as_deref()
                .and_then(|path| imgload::load_rgba(path, 512))
                .and_then(|img| self.enhancer.as_ref()?.suggest_edits(&img));
            if let (Some(proc), Some(gpu)) = (self.processor.as_mut(), self.gpu.as_ref()) {
                if proc.has_image() {
                    proc.apply_edit_state(&EditState::default());
                    if let Some(state) = ai_state {
                        proc.apply_edit_state(&state);
                    } else {
                        proc.process(&gpu.device, &gpu.queue);
                        proc.auto_adjust();
                    }
                    needs_process = true;
                }
            }
        }

        // Process once per frame if any slider changed, and persist the edit
        if needs_process {
            if let (Some(proc), Some(gpu)) = (self.processor.as_mut(), self.gpu.as_ref()) {
                proc.process(&gpu.device, &gpu.queue);
            }
            if let (Some(db), Some(path), Some(proc)) =
                (&self.db, &self.current_path, &self.processor)
            {
                save_edits(db, path, &proc.edit_state());
            }
            self.flags.output_dirty = true;
        }

        // Refresh output texture only when GPU output changed
        if self.flags.output_dirty {
            if let (Some(proc), Some(tex_id)) = (&self.processor, self.image_tex_id) {
                if let (Some(view), Some(er), Some(gpu)) = (
                    proc.output_view(),
                    self.egui_renderer.as_mut(),
                    self.gpu.as_ref(),
                ) {
                    er.update_egui_texture_from_wgpu_texture(
                        &gpu.device,
                        &view,
                        wgpu::FilterMode::Linear,
                        tex_id,
                    );
                }
            }
            self.flags.output_dirty = false;
        }

        // wgpu render
        let gpu = self.gpu.as_mut().unwrap();
        let window = self.window.as_ref().unwrap();
        let egui_renderer = self.egui_renderer.as_mut().unwrap();

        let size = window.inner_size();
        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [size.width, size.height],
            pixels_per_point: window.scale_factor() as f32,
        };
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let tris = self.egui_ctx.tessellate(shapes, pixels_per_point);
        for (id, delta) in textures_delta.set {
            egui_renderer.update_texture(&gpu.device, &gpu.queue, id, &delta);
        }
        egui_renderer.update_buffers(
            &gpu.device,
            &gpu.queue,
            &mut encoder,
            &tris,
            &screen_descriptor,
        );

        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &frame_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.11,
                                g: 0.11,
                                b: 0.11,
                                a: 1.0,
                            }),
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
    let format = caps
        .formats
        .iter()
        .find(|f| f.is_srgb())
        .copied()
        .unwrap_or(caps.formats[0]);
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

    GpuState {
        device,
        queue,
        surface,
        config,
    }
}

fn resize_surface(gpu: &mut GpuState, size: PhysicalSize<u32>) {
    if size.width == 0 || size.height == 0 {
        return;
    }
    gpu.config.width = size.width;
    gpu.config.height = size.height;
    gpu.surface.configure(&gpu.device, &gpu.config);
}

fn move_file(from: &Path, to: &Path) -> std::io::Result<()> {
    if let Ok(()) = std::fs::rename(from, to) { Ok(()) } else {
        std::fs::copy(from, to)?;
        std::fs::remove_file(from)
    }
}

// Runs `work` over `paths` on a small thread pool, sending one (path, result)
// per file back over `tx` as it finishes. Used for folder-wide batch actions
// (tagging, rating) that shouldn't block the UI thread.
fn spawn_folder_workers<T: Send + 'static>(
    paths: Vec<PathBuf>,
    work: impl Fn(&Path) -> T + Send + Sync + 'static,
    tx: std::sync::mpsc::Sender<(PathBuf, T)>,
) {
    let n_workers = std::thread::available_parallelism().map_or(4, std::num::NonZero::get).min(8);
    let (work_tx, work_rx) = std::sync::mpsc::channel::<PathBuf>();
    let work_rx = std::sync::Arc::new(std::sync::Mutex::new(work_rx));
    let work = std::sync::Arc::new(work);
    for _ in 0..n_workers {
        let work_rx = std::sync::Arc::clone(&work_rx);
        let work = std::sync::Arc::clone(&work);
        let tx = tx.clone();
        std::thread::spawn(move || {
            loop {
                let Ok(path) = work_rx.lock().unwrap().recv() else { break };
                let result = work(&path);
                if tx.send((path, result)).is_err() {
                    break;
                }
            }
        });
    }
    for path in paths {
        let _ = work_tx.send(path);
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}

#[cfg(test)]
mod tests {
    use super::{load_bool_pref, move_file, save_bool_pref};

    #[test]
    fn move_file_moves() {
        let dir = std::env::temp_dir().join(format!("ip_move_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let from = dir.join("a.txt");
        let to = dir.join("b.txt");
        std::fs::write(&from, b"hi").unwrap();
        move_file(&from, &to).unwrap();
        assert!(!from.exists());
        assert_eq!(std::fs::read(&to).unwrap(), b"hi");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bool_pref_roundtrip() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE app_kv (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );",
        )
        .unwrap();
        assert!(load_bool_pref(&conn, "x", true));
        save_bool_pref(&conn, "x", false);
        assert!(!load_bool_pref(&conn, "x", true));
        save_bool_pref(&conn, "x", true);
        assert!(load_bool_pref(&conn, "x", false));
    }
}
