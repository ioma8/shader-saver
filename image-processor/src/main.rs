mod canoncgt;
mod classify;
mod cull;
mod enhance;
mod face;
mod imgload;
#[cfg(test)]
mod judge;
mod look_model;
mod presets;
mod processor;
mod rate;
mod raw_develop;
mod raw_fit;
mod similar;
mod tags;
mod ui;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use egui_wgpu::ScreenDescriptor;
use processor::{EditState, LookProfile, Processor};
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
const KEY_DEVELOP_RAW: &str = "develop_raw";
const KEY_CAPTURE_LOOK: &str = "capture_look";
const KEY_APPLY_LOOK: &str = "apply_look";
const KEY_TEACH_LOOK_MODEL: &str = "teach_look_model";
const KEY_LOOK_FOLDER: &str = "look_folder";
const KEY_TRASH_REJECTS: &str = "trash_rejects";
const KEY_MOVE_REJECTS: &str = "move_rejects";
const KEY_COPY_PICKS: &str = "copy_picks";
const KEY_CLASSIFY_FOLDER: &str = "classify_folder";
const KEY_RATE_FOLDER: &str = "rate_folder";
const KEY_ADJUST_FOLDER: &str = "adjust_folder";
const KEY_SIMILAR_CULL: &str = "similar_cull";
const CULL_HELP_KEY: &str = "browse_cull_help_visible";
const RAW_DEFAULT_SHARPEN: f32 = 0.65;
const RAW_DEFAULT_SHARPEN_RADIUS: f32 = 2.0;

// RAW capture sharpening is an initialization, not a user setting: the first
// open of a RAW seeds the sliders so the sensor render reads sharp, while an
// explicit slider value of zero survives reopening. Returns whether it ran.
fn init_raw_sharpening(state: &mut EditState) -> bool {
    if state.raw_sharpening_initialized {
        return false;
    }
    if state.unsharp_strength == 0.0 {
        state.unsharp_strength = RAW_DEFAULT_SHARPEN;
        state.unsharp_blur_radius = RAW_DEFAULT_SHARPEN_RADIUS;
    }
    state.raw_sharpening_initialized = true;
    true
}

// ---- Edit persistence (SQLite, one JSON row per image path) ----

pub fn app_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".image-processor"))
}

fn existing_look_examples_path(dir: &Path) -> PathBuf {
    let current = dir.join("look-model-examples.json");
    if current.exists() {
        current
    } else {
        // One-time compatibility with examples captured by the prototype.
        dir.join("look-student-examples.json")
    }
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
        );
        CREATE TABLE IF NOT EXISTS captured_look (
            id      INTEGER PRIMARY KEY CHECK (id = 1),
            profile TEXT NOT NULL,
            thumb   BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS captured_look_full (
            id      INTEGER PRIMARY KEY CHECK (id = 1),
            image   BLOB NOT NULL
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

// A captured look keeps a small model reference and a separate full-resolution
// overlay reference. The neural path never needs the latter.
//
// The pixels are kept because statistics alone cannot serve as an objective. When
// the test harness scores results against the reference itself, and that needs the
// reference, not a summary of it.
struct CapturedLook {
    profile: LookProfile,
    reference: image::RgbaImage,
    reference_full: image::RgbaImage,
}

// The captured look outlives the session: it is measured data, not GPU state.
fn save_look(conn: &rusqlite::Connection, look: &CapturedLook) {
    let Ok(profile) = serde_json::to_string(&look.profile) else {
        return;
    };
    // JPEG carries no alpha, and a look reference has no use for one.
    let opaque = image::DynamicImage::ImageRgba8(look.reference.clone()).to_rgb8();
    let mut thumb = Vec::new();
    if image::codecs::jpeg::JpegEncoder::new_with_quality(&mut thumb, 88)
        .encode(
            opaque.as_raw(),
            opaque.width(),
            opaque.height(),
            image::ExtendedColorType::Rgb8,
        )
        .is_err()
    {
        return;
    }
    let _ = conn.execute(
        "INSERT INTO captured_look (id, profile, thumb) VALUES (1, ?1, ?2)
         ON CONFLICT(id) DO UPDATE SET profile = ?1, thumb = ?2",
        rusqlite::params![profile, thumb],
    );
    let full = image::DynamicImage::ImageRgba8(look.reference_full.clone()).to_rgb8();
    let mut full_bytes = Vec::new();
    if image::codecs::jpeg::JpegEncoder::new_with_quality(&mut full_bytes, 92)
        .encode(
            full.as_raw(),
            full.width(),
            full.height(),
            image::ExtendedColorType::Rgb8,
        )
        .is_ok()
    {
        let _ = conn.execute(
            "INSERT INTO captured_look_full (id, image) VALUES (1, ?1)
             ON CONFLICT(id) DO UPDATE SET image = ?1",
            rusqlite::params![full_bytes],
        );
    }
}

fn load_look(conn: &rusqlite::Connection) -> Option<CapturedLook> {
    let (profile, thumb): (String, Vec<u8>) = conn
        .query_row(
            "SELECT profile, thumb FROM captured_look WHERE id = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .ok()?;
    let reference = image::load_from_memory(&thumb).ok()?.to_rgba8();
    let reference_full = conn
        .query_row(
            "SELECT image FROM captured_look_full WHERE id = 1",
            [],
            |row| row.get::<_, Vec<u8>>(0),
        )
        .ok()
        .and_then(|bytes| image::load_from_memory(&bytes).ok())
        .map(|image| image.to_rgba8())
        .unwrap_or_else(|| reference.clone());
    Some(CapturedLook {
        profile: serde_json::from_str(&profile).ok()?,
        reference,
        reference_full,
    })
}

// Carry the captured look onto `img` with the single constrained look model.
// Its output is one photographic transform, baked and applied by the existing
// LUT path. There is deliberately no model selection, blend, or fallback.
fn look_chain_for(
    img: &image::RgbaImage,
    state: &mut EditState,
    look: &CapturedLook,
    model: &look_model::LookModel,
    canon: Option<&canoncgt::CanonCgt>,
    faces: &[[f32; 4]],
) {
    if let Some(current) =
        processor::LookProfile::measure(img.as_raw(), img.width(), img.height(), faces)
    {
        state.ai_lut_enabled = false;
        state.look = vec![model.predict(&current, &look.profile)];
        if let Some(canon) = canon {
            let fallback =
                processor::baked_lut(state).unwrap_or_else(processor::identity_photo_lut);
            if let Some(lut) = canon.predict_lut(img, &look.reference_full, &fallback) {
                state.look.clear();
                state.look_lut = Some(lut);
            }
        }
    }
}

// Downscale for the judge, which works at 512 square regardless.
fn look_reference_thumb(img: &image::RgbaImage) -> image::RgbaImage {
    image::imageops::thumbnail(img, 512, 512)
}

fn load_look_input(path: &Path, max_dim: u32) -> Option<image::RgbaImage> {
    if imgload::is_raw(path) {
        raw_develop::develop_raw(path, max_dim)
    } else {
        imgload::load_rgba(path, max_dim)
    }
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
    capture_time: Option<i64>,    // YYYYMMDDHHMMSS for sorting
    shutter: Option<String>,      // pre-formatted: "1/200" or "2s"
    aperture: Option<(u32, u32)>, // rational (numerator, denominator), displayed as "f/2.8"
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
    let Ok(file) = std::fs::File::open(path) else {
        return out;
    };
    let Ok(exif) = exif::Reader::new().read_from_container(&mut std::io::BufReader::new(file))
    else {
        return out;
    };

    if let Some(f) = exif.get_field(exif::Tag::DateTimeOriginal, exif::In::PRIMARY) {
        if let exif::Value::Ascii(v) = &f.value {
            if let Some(s) = v.first().and_then(|b| std::str::from_utf8(b).ok()) {
                if let Some((date, time)) = s.split_once(' ') {
                    let mut dp = date.split(':').filter_map(|p| p.parse::<i64>().ok());
                    let mut tp = time.split(':').filter_map(|p| p.parse::<i64>().ok());
                    if let (Some(y), Some(mo), Some(d), Some(h), Some(mi), Some(s)) = (
                        dp.next(),
                        dp.next(),
                        dp.next(),
                        tp.next(),
                        tp.next(),
                        tp.next(),
                    ) {
                        out.capture_time = Some(
                            y * 10_000_000_000
                                + mo * 100_000_000
                                + d * 1_000_000
                                + h * 10_000
                                + mi * 100
                                + s,
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
                    if tenths < 1 {
                        format!("{whole}s")
                    } else {
                        format!("{whole}.{tenths}s")
                    }
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
    reference_tex: Option<egui::TextureHandle>,
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
    thumb_rx: Option<std::sync::mpsc::Receiver<(
        usize,
        PathBuf,
        Option<(egui::ColorImage, ThumbExif)>,
    )>>,
    current_path: Option<PathBuf>,
    selected: Option<PathBuf>,
    meta: HashMap<PathBuf, cull::CullMeta>,
    filter: BrowseFilter,
    min_rating: u8,
    sort: BrowseSort,
    grid_cols: usize,
    tree_expanded: std::collections::HashSet<PathBuf>,
    db: Option<rusqlite::Connection>,
    presets_dir: Option<PathBuf>,
    presets: Vec<String>,
    preset_name: String,
    classifier: Option<Arc<classify::Classifier>>,
    enhancer: Option<Arc<enhance::Enhancer>>,
    tags: HashMap<PathBuf, Vec<String>>,
    tag_edit: String,
    classify_tx: std::sync::mpsc::Sender<(usize, PathBuf, Option<Vec<String>>)>,
    classify_rx: std::sync::mpsc::Receiver<(usize, PathBuf, Option<Vec<String>>)>,
    // (done, total) while a folder tagging batch is running
    classify_progress: Option<(usize, usize)>,
    rater: Option<Arc<rate::Rater>>,
    rate_tx: std::sync::mpsc::Sender<(usize, PathBuf, Option<u8>)>,
    rate_rx: std::sync::mpsc::Receiver<(usize, PathBuf, Option<u8>)>,
    // (done, total) while a folder rating batch is running
    rate_progress: Option<(usize, usize)>,
    adjust_tx: std::sync::mpsc::Sender<(usize, PathBuf, Option<EditState>)>,
    adjust_rx: std::sync::mpsc::Receiver<(usize, PathBuf, Option<EditState>)>,
    adjust_progress: Option<(usize, usize)>,
    look: Option<Arc<CapturedLook>>,
    face_detector: Option<Arc<face::Detector>>,
    look_model: Arc<look_model::LookModel>,
    canon: Option<Arc<canoncgt::CanonCgt>>,
    look_examples: Vec<look_model::TrainingExample>,
    similar_tx: std::sync::mpsc::Sender<(usize, PathBuf, Option<similar::Analysis>)>,
    similar_rx: std::sync::mpsc::Receiver<(usize, PathBuf, Option<similar::Analysis>)>,
    // (done, total) while semantic burst analysis is running
    similar_progress: Option<(usize, usize)>,
    similar_candidates: HashMap<PathBuf, similar::Analysis>,
    similar_summary: Option<(usize, usize)>,
}

impl App {
    fn new() -> Self {
        let db = open_db();
        let meta = db.as_ref().map(cull::load_all_meta).unwrap_or_default();
        let show_cull_help = db
            .as_ref()
            .is_none_or(|db| load_bool_pref(db, CULL_HELP_KEY, true));
        let look = db.as_ref().and_then(load_look).map(Arc::new);
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_default();
        let mut tree_expanded = std::collections::HashSet::new();
        let mut p = PathBuf::from("/");
        tree_expanded.insert(p.clone());
        for comp in home.components().skip(1) {
            p.push(comp);
            tree_expanded.insert(p.clone());
        }
        let presets_dir = app_dir().map(|d| d.join("presets"));
        let presets = presets_dir
            .as_deref()
            .map(presets::list)
            .unwrap_or_default();
        let look_examples_path = app_dir().map(|d| existing_look_examples_path(&d));
        let look_examples = look_examples_path
            .as_deref()
            .map(look_model::load_examples)
            .unwrap_or_default();
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
            reference_tex: None,
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
            look,
            face_detector: face::Detector::load().map(Arc::new),
            look_model: Arc::new(look_model::LookModel::train_with_examples(&look_examples)),
            canon: canoncgt::CanonCgt::load().map(Arc::new),
            look_examples,
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
                    && self.view == View::Edit
                {
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

        let (tx, rx) = std::sync::mpsc::channel::<(
            usize,
            PathBuf,
            Option<(egui::ColorImage, ThumbExif)>,
        )>();
        // dropping the old rx makes a stale loader thread stop
        self.thumb_rx = Some(rx);
        let ctx = self.egui_ctx.clone();
        let work = move |_idx: usize, path: &Path| -> Option<(egui::ColorImage, ThumbExif)> {
            let exif = read_exif(path);
            let img = imgload::load_preview_rgba(path, 220)?;
            let (img_w, img_h) = img.dimensions();
            let ci =
                egui::ColorImage::from_rgba_unmultiplied([img_w as usize, img_h as usize], &img);
            ctx.request_repaint();
            Some((ci, exif))
        };
        spawn_folder_workers(paths, work, tx);
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

    // Carry the captured look onto `paths`, off the UI thread.
    //
    // Deriving a chain renders the photo several times, so even a single photo goes
    // through the worker pool. Results arrive on the same channel as the folder
    // batch and are applied to whichever photo is open, with the existing progress
    // indicator.
    fn spawn_look_transfer(&mut self, paths: Vec<PathBuf>) {
        let Some(look) = self.look.clone() else {
            return;
        };
        if paths.is_empty() {
            return;
        }
        self.adjust_progress = Some((0, paths.len()));
        let detector = self.face_detector.clone();
        let look_model = self.look_model.clone();
        let canon = self.canon.clone();
        // Workers have no GPU context, so each renders its own preview on the CPU
        // to derive the transfer. The LUT stage is everything a look-transferred
        // state contains, so that preview matches what the GPU shows at full size.
        let work = move |_idx: usize, path: &Path| {
            let mut state = EditState::default();
            // Develop straight from the RAW file; the generic browse render is
            // only an open preview and would stack its own tone curve.
            state.raw_isp_enabled = imgload::is_raw(path);
            let img = load_look_input(path, 768)?;
            let faces = detector
                .as_ref()
                .map(|d| d.detect_boxes(&img))
                .unwrap_or_default();
            look_chain_for(
                &img,
                &mut state,
                &look,
                look_model.as_ref(),
                canon.as_deref(),
                &faces,
            );
            Some(state)
        };
        spawn_folder_workers(paths, work, self.adjust_tx.clone());
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
                .filter(|&&i| {
                    self.meta
                        .get(&candidates[i].path)
                        .is_some_and(|m| m.flag == cull::Flag::Pick)
                })
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
                if i == keep
                    || self
                        .meta
                        .get(&candidates[i].path)
                        .is_some_and(|m| m.flag == cull::Flag::Pick)
                {
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

        let saved_state = self.db.as_ref().and_then(|db| load_edits(db, path));
        let state = saved_state.clone().unwrap_or_default();
        processor.apply_edit_state(&state);

        // RAWs arrive from a sensor decode rather than a camera preview, so
        // give them a restrained capture-sharpening starting point. Persist an
        // initialization marker so a user who moves the slider to zero keeps
        // that explicit choice on the next open.
        if imgload::is_raw(path) && init_raw_sharpening(processor) {
            if let Some(db) = &self.db {
                save_edits(db, path, &processor.edit_state());
            }
        }

        let Some(img) = imgload::load_edit_rgba(path) else {
            return;
        };

        processor.upload_rgba(&img, &gpu.device, &gpu.queue);
        if state.raw_isp_enabled && imgload::is_raw(path) {
            if let Some(developed) = raw_develop::develop_raw_u16(path, 2048) {
                if processor.replace_input_u16(
                    developed.width(),
                    developed.height(),
                    developed.as_raw(),
                    &gpu.queue,
                ) {
                    // `upload_rgba` processed the temporary embedded preview.
                    // Render the restored edits again after swapping in the
                    // persisted RAW development input.
                    processor.process(&gpu.device, &gpu.queue);
                }
            }
        }
        self.rebind_image_textures();

        if let Some(window) = &self.window {
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("image-processor");
            window.set_title(name);
        }

        self.tag_edit = self
            .tags
            .get(path)
            .map(|t| t.join(", "))
            .unwrap_or_default();

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

    // Apply one-shot UI requests that need `self` beyond the frame: file dialogs,
    // batch AI actions, look capture/teach, trash/move/copy.
    fn handle_ui_requests(&mut self, requests: ui::UiRequests, needs_process: &mut bool) {
        // Capture measures the reference as it is actually rendered on screen,
        // whatever produced that rendering — AI adjustment, hand-set sliders or
        // both. Reading the sliders instead would miss the AI LUT entirely,
        // which is where an auto-adjusted photo keeps its whole look.
        if requests.capture_look {
            if let (Some(proc), Some(gpu)) = (&self.processor, &self.gpu) {
                if let Some((width, height)) = proc.image_size {
                    let faces = self.face_detector.as_ref();
                    let captured = proc
                        .output_pixels(&gpu.device, &gpu.queue)
                        .and_then(|px| image::RgbaImage::from_raw(width, height, px))
                        .and_then(|rendered| {
                            let boxes =
                                faces.map(|d| d.detect_boxes(&rendered)).unwrap_or_default();
                            Some(CapturedLook {
                                profile: LookProfile::measure(
                                    rendered.as_raw(),
                                    rendered.width(),
                                    rendered.height(),
                                    &boxes,
                                )?,
                                reference: look_reference_thumb(&rendered),
                                reference_full: rendered,
                            })
                        });
                    if let Some(captured) = captured {
                        if let Some(db) = &self.db {
                            save_look(db, &captured);
                        }
                        let image = &captured.reference_full;
                        let color = egui::ColorImage::from_rgba_unmultiplied(
                            [image.width() as usize, image.height() as usize],
                            image.as_raw(),
                        );
                        self.reference_tex = Some(self.egui_ctx.load_texture(
                            "captured-look-reference",
                            color,
                            egui::TextureOptions::LINEAR,
                        ));
                        self.look = Some(Arc::new(captured));
                    }
                }
            }
        }
        if requests.teach_look_model {
            let example = (|| {
                let path = self.current_path.as_ref()?;
                let look = self.look.as_ref()?;
                let proc = self.processor.as_ref()?;
                let gpu = self.gpu.as_ref()?;
                let target = load_look_input(path, 768)?;
                let faces = self
                    .face_detector
                    .as_ref()
                    .map(|d| d.detect_boxes(&target))
                    .unwrap_or_default();
                let current =
                    LookProfile::measure(target.as_raw(), target.width(), target.height(), &faces)?;
                let (width, height) = proc.image_size?;
                let pixels = proc.output_pixels(&gpu.device, &gpu.queue)?;
                let desired_image = image::RgbaImage::from_raw(width, height, pixels)?;
                let desired = LookProfile::measure(
                    desired_image.as_raw(),
                    desired_image.width(),
                    desired_image.height(),
                    &faces,
                )?;
                Some(look_model::TrainingExample {
                    current,
                    reference: look.profile.clone(),
                    desired,
                })
            })();
            if let Some(example) = example {
                self.look_examples.push(example);
                if let Some(path) = app_dir().map(|d| d.join("look-model-examples.json")) {
                    look_model::save_examples(&path, &self.look_examples);
                }
                self.look_model = Arc::new(look_model::LookModel::train_with_examples(
                    &self.look_examples,
                ));
            }
        }
        if requests.apply_look {
            if let Some(path) = self.current_path.clone() {
                self.spawn_look_transfer(vec![path]);
            }
        }
        if requests.develop_raw {
            // Decode at the editor's current resolution so the swap is near 1:1.
            let max_dim = self
                .processor
                .as_ref()
                .and_then(|p| p.image_size.map(|(w, h)| w.max(h)))
                .unwrap_or(2048);
            let development = self
                .current_path
                .as_deref()
                .filter(|path| imgload::is_raw(path))
                .and_then(|path| raw_develop::develop_raw_u16(path, max_dim));
            if development.is_none() {
                eprintln!(
                    "Develop RAW: could not develop {}",
                    self.current_path
                        .as_deref()
                        .map_or_else(|| "<none>".to_string(), |p| p.display().to_string())
                );
            }
            let mut applied = false;
            if let (Some(processor), Some(gpu), Some(developed)) =
                (self.processor.as_mut(), self.gpu.as_ref(), development)
            {
                if init_raw_sharpening(processor) {
                    *needs_process = true;
                }
                processor.raw_isp_enabled = true;
                processor.raw_development = None;
                // Re-upload through the owning texture path. `replace_input_u16`
                // intentionally refuses an uninitialized/stale input texture;
                // a button action must be able to establish the RAW input.
                processor.upload_u16(
                    developed.width(),
                    developed.height(),
                    developed.as_raw(),
                    &gpu.device,
                    &gpu.queue,
                );
                processor.process(&gpu.device, &gpu.queue);
                applied = true;
            }
            if applied {
                self.rebind_image_textures();
                self.flags.output_dirty = true;
                *needs_process = false;
            }
        }
        if requests.trash_rejects {
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
        if let Some(dest) = requests.move_rejects_dir {
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
        if let Some(dest) = requests.copy_picks_dir {
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
        if let Some(dir) = requests.scan_dir {
            self.scan_folder(&dir);
        }
        if let Some(path) = requests.open_path {
            self.register_image(&path);
        }
        if let Some(path) = requests.export_path {
            if let (Some(proc), Some(gpu)) = (&self.processor, &self.gpu) {
                proc.export(&path, &gpu.device, &gpu.queue);
            }
        }
        if requests.classify_folder {
            let paths: Vec<PathBuf> = self.thumbs.iter().map(|t| t.path.clone()).collect();
            if let (Some(classifier), false) = (&self.classifier, paths.is_empty()) {
                self.classify_progress = Some((0, paths.len()));
                let classifier = Arc::clone(classifier);
                let work = move |_: usize, path: &Path| {
                    imgload::load_edit_rgba(path).map(|img| classifier.classify(&img))
                };
                spawn_folder_workers(paths, work, self.classify_tx.clone());
            }
        }
        if requests.rate_folder {
            let paths: Vec<PathBuf> = self.thumbs.iter().map(|t| t.path.clone()).collect();
            if let (Some(rater), false) = (&self.rater, paths.is_empty()) {
                self.rate_progress = Some((0, paths.len()));
                let rater = Arc::clone(rater);
                let work = move |_: usize, path: &Path| {
                    imgload::load_edit_rgba(path).and_then(|img| rater.rate(&img))
                };
                spawn_folder_workers(paths, work, self.rate_tx.clone());
            }
        }
        if requests.adjust_folder {
            let paths: Vec<PathBuf> = self.thumbs.iter().map(|t| t.path.clone()).collect();
            if let (Some(enhancer), false) = (&self.enhancer, paths.is_empty()) {
                self.adjust_progress = Some((0, paths.len()));
                let enhancer = Arc::clone(enhancer);
                let work = move |_: usize, path: &Path| {
                    imgload::load_rgba(path, 512).and_then(|img| enhancer.suggest_edits(&img))
                };
                spawn_folder_workers(paths, work, self.adjust_tx.clone());
            }
        }
        if requests.look_folder {
            let paths: Vec<PathBuf> = self.thumbs.iter().map(|t| t.path.clone()).collect();
            self.spawn_look_transfer(paths);
        }
        if requests.similar_cull {
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
                let work = move |_: usize, path: &Path| {
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
        if requests.auto {
            let ai_state = self
                .current_path
                .as_deref()
                .and_then(|path| imgload::load_rgba(path, 512))
                .and_then(|img| self.enhancer.as_ref()?.suggest_edits(&img));
            if let (Some(proc), Some(gpu)) = (self.processor.as_mut(), self.gpu.as_ref()) {
                if proc.has_image() {
                    proc.apply_edit_state(&EditState::default());
                    proc.restore_source(&gpu.queue);
                    if let Some(state) = ai_state {
                        proc.apply_edit_state(&state);
                    } else {
                        proc.process(&gpu.device, &gpu.queue);
                        proc.auto_adjust();
                    }
                    *needs_process = true;
                }
            }
        }
    }

    fn render(&mut self) {
        if self.reference_tex.is_none() {
            if let Some(look) = &self.look {
                let image = &look.reference_full;
                let color = egui::ColorImage::from_rgba_unmultiplied(
                    [image.width() as usize, image.height() as usize],
                    image.as_raw(),
                );
                self.reference_tex = Some(self.egui_ctx.load_texture(
                    "captured-look-reference",
                    color,
                    egui::TextureOptions::LINEAR,
                ));
            }
        }
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
            while let Ok((i, _, result)) = rx.try_recv() {
                if let Some((img, exif)) = result {
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
        }
        // Pick up finished classification results. `None` means that image
        // failed to load — still counts toward progress, but nothing to save.
        while let Ok((_, path, result)) = self.classify_rx.try_recv() {
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
        while let Ok((_, path, result)) = self.rate_rx.try_recv() {
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
        while let Ok((_, path, result)) = self.adjust_rx.try_recv() {
            if let Some(mut state) = result {
                if imgload::is_raw(&path) {
                    init_raw_sharpening(&mut state);
                }
                if let Some(db) = &self.db {
                    save_edits(db, &path, &state);
                }
                if self.current_path.as_deref() == Some(path.as_path()) {
                    if let (Some(proc), Some(gpu)) = (self.processor.as_mut(), self.gpu.as_ref()) {
                        proc.apply_edit_state(&state);
                        if state.raw_isp_enabled {
                            if let Some(path) = self.current_path.as_deref() {
                                let max_dim = proc.image_size.map_or(2048, |(w, h)| w.max(h));
                                if let Some(developed) = raw_develop::develop_raw_u16(path, max_dim)
                                {
                                    proc.replace_input_u16(
                                        developed.width(),
                                        developed.height(),
                                        developed.as_raw(),
                                        &gpu.queue,
                                    );
                                }
                            }
                        }
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
        while let Ok((_, path, result)) = self.similar_rx.try_recv() {
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
                self.thumbs[a]
                    .exif
                    .capture_time
                    .unwrap_or(i64::MAX)
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
        let frame_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut cull_actions: Vec<(PathBuf, cull::CullAction)> = Vec::new();

        // Draw the UI. `egui_ctx` is a cheap clone so the frame closure can borrow
        // all of `self` while the context stays usable after the frame.
        let (shapes, textures_delta, pixels_per_point, requests, mut needs_process) = {
            let egui_ctx = self.egui_ctx.clone();
            let raw_input = {
                let egui_state = self.egui_state.as_mut().unwrap();
                let window = self.window.as_ref().unwrap();
                egui_state.take_egui_input(window)
            };
            let mut needs_process = false;
            let full_output = egui_ctx.run(raw_input, |ctx| {
                ui::tabs(ctx, &mut self.view);
                if self.flags.confirm_trash {
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
                                    self.flags.confirm_trash = false;
                                }
                                if ui.button("Cancel").clicked() {
                                    self.flags.confirm_trash = false;
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
                    let cols = self.grid_cols as i32;
                    let delta = (i32::from(right) - i32::from(left)) + (i32::from(down) - i32::from(up)) * cols;
                    if delta != 0 && !visible.is_empty() {
                        let anchor = if self.view == View::Edit {
                            self.current_path.as_deref()
                        } else {
                            self.selected.as_deref().or(self.current_path.as_deref())
                        };
                        let pos = anchor
                            .and_then(|a| visible.iter().position(|&i| self.thumbs[i].path == a));
                        let next = match pos {
                            Some(c) => (c as i32 + delta).clamp(0, visible.len() as i32 - 1) as usize,
                            None => 0,
                        };
                        let p = self.thumbs[visible[next]].path.clone();
                        if self.view == View::Edit {
                            if self.current_path.as_deref() != Some(p.as_path()) {
                                ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_OPEN_PATH), p));
                            }
                        } else {
                            self.selected = Some(p);
                            self.flags.grid_scroll = true;
                        }
                    }
                    if self.view == View::Browse && enter {
                        if let Some(p) = self.selected.clone().or(self.current_path.clone()) {
                            ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_OPEN_PATH), p));
                        }
                    }
                    if self.view == View::Browse && space && !visible.is_empty() {
                        // Pick current + advance to next
                        if let Some(path) = self.selected.clone().or(self.current_path.clone()) {
                            cull_actions.push((path.clone(), cull::CullAction::TogglePick));
                        }
                        let anchor = self.selected.as_deref().or(self.current_path.as_deref());
                        let pos = anchor.and_then(|a| visible.iter().position(|&i| self.thumbs[i].path == a));
                        let next = match pos {
                            Some(c) => (c + 1).min(visible.len() - 1),
                            None => 0,
                        };
                        self.selected = Some(self.thumbs[visible[next]].path.clone());
                        self.flags.grid_scroll = true;
                    }
                    if let Some(path) = self.selected.clone().or(self.current_path.clone()) {
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

                if self.view == View::Browse {
                    ui::browse(ctx, self, &visible, n_picks, n_rejects);
                } else {
                    ui::edit(ctx, self, &mut cull_actions, &mut needs_process, &visible);
                }
            });
            let egui_state = self.egui_state.as_mut().unwrap();
            let window = self.window.as_ref().unwrap();
            egui_state.handle_platform_output(window, full_output.platform_output);
            let requests = ui::take_requests(&egui_ctx);
            (
                full_output.shapes,
                full_output.textures_delta,
                full_output.pixels_per_point,
                requests,
                needs_process,
            )
        };

        for (path, action) in cull_actions {
            self.apply_cull_action(&path, action);
        }
        self.handle_ui_requests(requests, &mut needs_process);
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
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
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
        .request_device(
            &wgpu::DeviceDescriptor {
                // 16-bit integer input textures (65536 levels per channel).
                required_features: if adapter
                    .features()
                    .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
                {
                    wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
                } else {
                    wgpu::Features::empty()
                },
                ..Default::default()
            },
            None,
        )
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
    if let Ok(()) = std::fs::rename(from, to) {
        Ok(())
    } else {
        std::fs::copy(from, to)?;
        std::fs::remove_file(from)
    }
}

// Runs `work` over `paths` on a small thread pool, sending one
// `(index, path, result)` per file back over `tx` as it finishes. Used for
// folder-wide batch actions (thumbnails, tagging, rating) that shouldn't
// block the UI thread. Results carry the original index so the receiver can
// match them to the source list regardless of completion order.
fn spawn_folder_workers<T: Send + 'static>(
    paths: Vec<PathBuf>,
    work: impl Fn(usize, &Path) -> T + Send + Sync + 'static,
    tx: std::sync::mpsc::Sender<(usize, PathBuf, T)>,
) {
    let n_workers = std::thread::available_parallelism()
        .map_or(4, std::num::NonZero::get)
        .min(8);
    let (work_tx, work_rx) = std::sync::mpsc::channel::<(usize, PathBuf)>();
    let work_rx = std::sync::Arc::new(std::sync::Mutex::new(work_rx));
    let work = std::sync::Arc::new(work);
    for _ in 0..n_workers {
        let work_rx = std::sync::Arc::clone(&work_rx);
        let work = std::sync::Arc::clone(&work);
        let tx = tx.clone();
        std::thread::spawn(move || loop {
            let Ok((idx, path)) = work_rx.lock().unwrap().recv() else {
                break;
            };
            let result = work(idx, &path);
            if tx.send((idx, path, result)).is_err() {
                break;
            }
        });
    }
    for (idx, path) in paths.into_iter().enumerate() {
        let _ = work_tx.send((idx, path));
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args
        .get(1)
        .is_some_and(|arg| arg == "--train-raw-render-model")
    {
        let folder = args.get(2).map(PathBuf::from).unwrap_or_default();
        let output = args
            .get(3)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("models/raw_render_model.json"));
        match raw_fit::fit_raw_render_model(&folder, &output) {
            Ok(count) => println!("trained RAW render model from {count} camera renders"),
            Err(error) => {
                eprintln!("RAW render model training failed: {error}");
                std::process::exit(1);
            }
        }
        return;
    }
    if args.get(1).is_some_and(|arg| arg == "--fit-raw-scurve") {
        let folder = args.get(2).map(PathBuf::from).unwrap_or_default();
        let output = args
            .get(3)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("models/raw_s_curve.json"));
        match raw_fit::fit_s_curve(&folder, &output) {
            Ok(count) => println!("fitted phone S-curve from {count} DNG/JPEG pairs"),
            Err(error) => {
                eprintln!("S-curve fit failed: {error}");
                std::process::exit(1);
            }
        }
        return;
    }
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}

#[cfg(test)]
mod tests {
    use super::{
        load_bool_pref, load_edits, load_look, look_reference_thumb, move_file, save_bool_pref,
        save_edits, save_look, CapturedLook,
    };
    use crate::processor::{EditState, LookProfile};

    #[test]
    fn raw_development_and_applied_look_round_trip_through_database() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE edits (
                path    TEXT PRIMARY KEY,
                params  TEXT NOT NULL,
                updated INTEGER NOT NULL
            );",
        )
        .unwrap();
        let path = std::path::Path::new("/photos/image.dng");
        let state = EditState {
            raw_isp_enabled: true,
            raw_development: Some(vec![0.0, 0.25, 0.5, 0.75, 1.0]),
            look_lut: Some(vec![1.0, 0.8, 0.6, 0.4, 0.2, 0.0]),
            look_strength: 0.73,
            ..Default::default()
        };

        save_edits(&conn, path, &state);
        let restored = load_edits(&conn, path).expect("saved edit state");

        assert!(restored.raw_isp_enabled);
        assert_eq!(restored.raw_development, state.raw_development);
        assert_eq!(restored.look_lut, state.look_lut);
        assert_eq!(restored.look_strength, state.look_strength);
    }

    #[test]
    fn captured_look_round_trips_through_the_database() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE captured_look (
                id      INTEGER PRIMARY KEY CHECK (id = 1),
                profile TEXT NOT NULL,
                thumb   BLOB NOT NULL
            );
            CREATE TABLE captured_look_full (
                id      INTEGER PRIMARY KEY CHECK (id = 1),
                image   BLOB NOT NULL
            );",
        )
        .unwrap();
        assert!(load_look(&conn).is_none());

        let rendered = image::RgbaImage::from_fn(64, 48, |x, y| {
            image::Rgba([(x * 3) as u8, (y * 4) as u8, 140, 255])
        });
        let captured = CapturedLook {
            profile: LookProfile::measure(
                rendered.as_raw(),
                rendered.width(),
                rendered.height(),
                &[[0.2, 0.2, 0.5, 0.5]],
            )
            .unwrap(),
            reference: look_reference_thumb(&rendered),
            reference_full: rendered.clone(),
        };
        save_look(&conn, &captured);

        let restored = load_look(&conn).expect("a captured look survives the session");
        assert_eq!(restored.profile.tone, captured.profile.tone);
        assert_eq!(restored.profile.cast, captured.profile.cast);
        assert_eq!(restored.profile.chroma, captured.profile.chroma);
        // The reference's skin/foliage/sky tones travel with it, since the anchors
        // are derived against them.
        assert_eq!(
            restored.profile.regions[0].share,
            captured.profile.regions[0].share
        );
        assert_eq!(
            restored.profile.regions[0].lightness,
            captured.profile.regions[0].lightness
        );
        // The reference pixels come back too, because the judge scores against
        // them rather than against the statistics.
        assert_eq!(restored.reference.dimensions(), (512, 512));
    }

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