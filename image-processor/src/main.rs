mod auto;
mod canoncgt;
mod camera_presets;
mod classify;
mod cull;
mod enhance;
mod face;
mod imgload;
#[cfg(test)]
mod judge;
mod look_model;
mod presets;
mod print;
#[cfg(target_os = "macos")]
mod print_macos;
mod processor;
mod rate;
mod raw_develop;
mod raw_fit;
mod retouch;
mod similar;
mod tags;
mod ui;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use egui_wgpu::ScreenDescriptor;
use processor::{
    EditState, ExportFormat, ExportOptions, LookProfile, Processor, RetouchSpot, MAX_RETOUCH_SPOTS,
};
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
const KEY_RETOUCH: &str = "retouch_dab";
const KEY_RETOUCH_REMOVE: &str = "retouch_remove";
const KEY_RETOUCH_CLEAR: &str = "retouch_clear";
const CULL_HELP_KEY: &str = "browse_cull_help_visible";
// UI state that survives a restart (see `SavedUi`).
const UI_DIR_KEY: &str = "ui_last_dir";
const UI_ZOOM_FIT_KEY: &str = "ui_zoom_fit";
const UI_FILTER_KEY: &str = "ui_browse_filter";
const UI_SORT_KEY: &str = "ui_browse_sort";
const UI_MIN_RATING_KEY: &str = "ui_min_rating";
const KEY_UNDO: &str = "undo_edit";
const KEY_SHOW_EXPORT: &str = "show_export_dialog";
const KEY_SHOW_PRINT: &str = "show_print_dialog";
const KEY_EXPORT_PDF: &str = "export_pdf";
const KEY_REDO: &str = "redo_edit";
// Undo history: how many steps an image keeps, how many images are remembered,
// and how long a burst of changes (a slider drag, one change per frame) counts
// as a single step.
const HISTORY_STEPS: usize = 24;
const HISTORY_IMAGES: usize = 8;
const HISTORY_MERGE_SECS: f64 = 0.3;
const UI_PRINT_ORIENTATION_KEY: &str = "ui_print_orientation";
const UI_PRINT_QUALITY_KEY: &str = "ui_print_quality";
const UI_PRINT_FIT_KEY: &str = "ui_print_fit";
const UI_PRINT_TYPE_KEY: &str = "ui_print_paper_type";
const UI_EXPORT_FORMAT_KEY: &str = "ui_export_format";
const UI_EXPORT_QUALITY_KEY: &str = "ui_export_quality";
const UI_EXPORT_MAX_KEY: &str = "ui_export_max_dim";
const UI_EXPORT_DPI_KEY: &str = "ui_export_dpi";
const UI_WINDOW_W_KEY: &str = "ui_window_width";
const UI_WINDOW_H_KEY: &str = "ui_window_height";
const DEFAULT_WINDOW: (f32, f32) = (1600.0, 1000.0);
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

/// Generic `app_kv` access. A missing or unparsable row always falls back to
/// the caller's default, so a stale preference can never fail a launch.
fn load_pref(conn: &rusqlite::Connection, key: &str) -> Option<String> {
    conn.query_row(
        "SELECT value FROM app_kv WHERE key = ?1",
        rusqlite::params![key],
        |row| row.get::<_, String>(0),
    )
    .ok()
}

fn save_pref(conn: &rusqlite::Connection, key: &str, value: &str) {
    let _ = conn.execute(
        "INSERT INTO app_kv (key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        rusqlite::params![key, value],
    );
}

fn load_bool_pref(conn: &rusqlite::Connection, key: &str, default: bool) -> bool {
    load_pref(conn, key)
        .and_then(|v| match v.as_str() {
            "0" => Some(false),
            "1" => Some(true),
            _ => None,
        })
        .unwrap_or(default)
}

fn save_bool_pref(conn: &rusqlite::Connection, key: &str, value: bool) {
    save_pref(conn, key, if value { "1" } else { "0" });
}

fn load_f32_pref(conn: &rusqlite::Connection, key: &str, default: f32) -> f32 {
    load_pref(conn, key)
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn save_f32_pref(conn: &rusqlite::Connection, key: &str, value: f32) {
    save_pref(conn, key, &value.to_string());
}

/// One image's undo history. The entry at `at` is the state on screen, entries
/// behind it are older, and entries ahead of it are the redo branch.
///
/// A burst of changes -- a slider drag arrives as one change per frame -- keeps
/// replacing the entry it started, instead of filling the stack with
/// intermediate values. Stacks hold whole edit states, which is cheap until a
/// look transform puts a 33-cubed LUT in one; hence the step cap.
#[derive(Default)]
struct ImageHistory {
    states: Vec<EditState>,
    at: usize,
    last_push: f64,
}

impl ImageHistory {
    /// Start over from `state`, as when an image is opened.
    fn reset(&mut self, state: EditState) {
        self.states = vec![state];
        self.at = 0;
        self.last_push = 0.0;
    }

    /// Record `state` as the new current one.
    fn commit(&mut self, state: EditState, now: f64) {
        if self.states.is_empty() {
            self.reset(state);
            return;
        }
        // Controls that set the dirty flag without changing anything (a preset
        // that is already selected, say) must not spend an undo step -- pressing
        // undo and watching nothing happen is worse than doing nothing.
        if self.states[self.at] == state {
            self.last_push = now;
            return;
        }
        // Only ever merge into an entry that has something behind it: `states[0]`
        // is the state the image was opened with and must stay reachable.
        if now - self.last_push < HISTORY_MERGE_SECS && self.at > 0 {
            self.states[self.at] = state;
        } else {
            self.states.truncate(self.at + 1);
            self.states.push(state);
            self.at = self.states.len() - 1;
            if self.states.len() > HISTORY_STEPS {
                self.states.remove(0);
                self.at -= 1;
            }
        }
        self.last_push = now;
    }

    /// The state to go back to, if there is one.
    fn undo(&mut self) -> Option<EditState> {
        if self.at == 0 {
            return None;
        }
        self.at -= 1;
        // Any edit after this starts a fresh step rather than merging backwards.
        self.last_push = 0.0;
        Some(self.states[self.at].clone())
    }

    fn redo(&mut self) -> Option<EditState> {
        if self.at + 1 >= self.states.len() {
            return None;
        }
        self.at += 1;
        self.last_push = 0.0;
        Some(self.states[self.at].clone())
    }
}

/// What a finished print render is for. macOS prints through AppKit, so the
/// work splits in two: the render happens on a worker (it is slow), then the
/// main thread opens the sheet or writes the PDF (AppKit insists, and the sheet
/// is modal).
#[cfg(target_os = "macos")]
#[derive(Clone)]
enum PrintTarget {
    Sheet,
    Pdf(PathBuf),
}

/// Where a batch writes each source, resolved up front so collisions are
/// settled before anything is written. Two files in one folder can share a
/// stem -- a RAW and its JPEG, say -- and both used to be written to the same
/// name, the second quietly replacing the first.
fn export_targets(dir: &Path, sources: &[PathBuf], extension: &str) -> Vec<PathBuf> {
    let mut used = std::collections::HashSet::new();
    sources
        .iter()
        .map(|source| {
            let stem = source
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("image");
            let mut name = format!("{stem}.{extension}");
            let mut n = 2;
            while !used.insert(name.clone()) {
                name = format!("{stem}-{n}.{extension}");
                n += 1;
            }
            dir.join(name)
        })
        .collect()
}

/// Per-image undo histories, capped in both directions.
#[derive(Default)]
struct Histories {
    images: HashMap<PathBuf, ImageHistory>,
    order: std::collections::VecDeque<PathBuf>,
}

impl Histories {
    fn entry(&mut self, path: &Path) -> &mut ImageHistory {
        if !self.images.contains_key(path) {
            if self.images.len() >= HISTORY_IMAGES {
                // Evict the least recently added image that is not this one.
                while let Some(oldest) = self.order.pop_front() {
                    if oldest.as_path() != path && self.images.remove(&oldest).is_some() {
                        break;
                    }
                }
            }
            self.order.push_back(path.to_path_buf());
            self.images.insert(path.to_path_buf(), ImageHistory::default());
        }
        self.images.get_mut(path).expect("just inserted")
    }

    fn commit(&mut self, path: &Path, state: EditState, now: f64) {
        self.entry(path).commit(state, now);
    }

    fn reset(&mut self, path: &Path, state: EditState) {
        self.entry(path).reset(state);
    }
}

/// The bits of UI state that survive a restart, gathered into one value so a
/// single comparison each frame decides when to write them back -- mutating a
/// filter somewhere deep in the browse view still gets saved, without every
/// call site having to remember.
#[derive(Clone, PartialEq, Debug)]
struct SavedUi {
    dir: Option<PathBuf>,
    zoom_fit: bool,
    filter: BrowseFilter,
    sort: BrowseSort,
    min_rating: u8,
    export: ExportOptions,
    print: print::PrintOptions,
    /// The paper *type* the user last chose, by value, so it survives a
    /// restart: the paper in the tray is the same paper next time, which is
    /// not true of a printer-specific index.
    print_paper_type: Option<String>,
}

impl SavedUi {
    fn load(db: Option<&rusqlite::Connection>) -> Self {
        let Some(db) = db else {
            return Self::default();
        };
        Self {
            dir: load_pref(db, UI_DIR_KEY)
                .filter(|dir| !dir.is_empty())
                .map(PathBuf::from),
            zoom_fit: load_bool_pref(db, UI_ZOOM_FIT_KEY, true),
            filter: BrowseFilter::from_key(&load_pref(db, UI_FILTER_KEY).unwrap_or_default()),
            sort: BrowseSort::from_key(&load_pref(db, UI_SORT_KEY).unwrap_or_default()),
            min_rating: load_pref(db, UI_MIN_RATING_KEY)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            print_paper_type: load_pref(db, UI_PRINT_TYPE_KEY).filter(|v| !v.is_empty()),
            print: print::PrintOptions {
                orientation: print::Orientation::from_key(
                    &load_pref(db, UI_PRINT_ORIENTATION_KEY).unwrap_or_default(),
                ),
                quality: print::Quality::from_key(
                    &load_pref(db, UI_PRINT_QUALITY_KEY).unwrap_or_default(),
                ),
                fit_to_page: load_bool_pref(db, UI_PRINT_FIT_KEY, true),
                // The paper depends on the printer, so it is chosen each time.
                ..print::PrintOptions::default()
            },
            export: ExportOptions {
                format: ExportFormat::from_key(
                    &load_pref(db, UI_EXPORT_FORMAT_KEY).unwrap_or_default(),
                ),
                quality: load_pref(db, UI_EXPORT_QUALITY_KEY)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(90)
                    .clamp(1, 100),
                max_dim: load_pref(db, UI_EXPORT_MAX_KEY)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0),
                dpi: load_pref(db, UI_EXPORT_DPI_KEY)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(ExportOptions::default().dpi),
            },
        }
    }

    fn save(&self, db: Option<&rusqlite::Connection>) {
        let Some(db) = db else { return };
        // A missing key clears the stored value rather than leaving a stale one.
        match self.dir.as_deref().and_then(Path::to_str) {
            Some(dir) => save_pref(db, UI_DIR_KEY, dir),
            None => save_pref(db, UI_DIR_KEY, ""),
        }
        save_bool_pref(db, UI_ZOOM_FIT_KEY, self.zoom_fit);
        save_pref(db, UI_FILTER_KEY, self.filter.key());
        save_pref(db, UI_SORT_KEY, self.sort.key());
        save_pref(db, UI_MIN_RATING_KEY, &self.min_rating.to_string());
        save_pref(db, UI_PRINT_ORIENTATION_KEY, self.print.orientation.key());
        save_pref(db, UI_PRINT_QUALITY_KEY, self.print.quality.key());
        save_bool_pref(db, UI_PRINT_FIT_KEY, self.print.fit_to_page);
        save_pref(
            db,
            UI_PRINT_TYPE_KEY,
            self.print_paper_type.as_deref().unwrap_or(""),
        );
        save_pref(db, UI_EXPORT_FORMAT_KEY, self.export.format.key());
        save_pref(db, UI_EXPORT_QUALITY_KEY, &self.export.quality.to_string());
        save_pref(db, UI_EXPORT_MAX_KEY, &self.export.max_dim.to_string());
        save_pref(db, UI_EXPORT_DPI_KEY, &self.export.dpi.to_string());
    }
}

impl Default for SavedUi {
    fn default() -> Self {
        Self {
            dir: None,
            zoom_fit: true,
            filter: BrowseFilter::All,
            sort: BrowseSort::Name,
            min_rating: 0,
            export: ExportOptions::default(),
            print: print::PrintOptions::default(),
            print_paper_type: None,
        }
    }
}

/// Window size, in logical points, as it was on the last clean exit.
fn saved_window(db: Option<&rusqlite::Connection>) -> (f32, f32) {
    let Some(db) = db else {
        return DEFAULT_WINDOW;
    };
    let (w, h) = (
        load_f32_pref(db, UI_WINDOW_W_KEY, DEFAULT_WINDOW.0),
        load_f32_pref(db, UI_WINDOW_H_KEY, DEFAULT_WINDOW.1),
    );
    // Guard against a corrupt or absurd row making the window unusable.
    if (400.0..=10000.0).contains(&w) && (300.0..=10000.0).contains(&h) {
        (w, h)
    } else {
        DEFAULT_WINDOW
    }
}

fn save_window(db: Option<&rusqlite::Connection>, w: f32, h: f32) {
    if let Some(db) = db {
        save_f32_pref(db, UI_WINDOW_W_KEY, w);
        save_f32_pref(db, UI_WINDOW_H_KEY, h);
    }
}


#[derive(PartialEq, Clone, Copy)]
enum View {
    Browse,
    Edit,
}

#[derive(PartialEq, Clone, Copy, Debug)]
enum BrowseFilter {
    All,
    Picks,
    Rejects,
    Unflagged,
}

impl BrowseFilter {
    fn key(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Picks => "picks",
            Self::Rejects => "rejects",
            Self::Unflagged => "unflagged",
        }
    }
    fn from_key(key: &str) -> Self {
        match key {
            "picks" => Self::Picks,
            "rejects" => Self::Rejects,
            "unflagged" => Self::Unflagged,
            _ => Self::All,
        }
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
enum BrowseSort {
    Name,
    Date,
    CaptureTime,
}

impl BrowseSort {
    fn key(self) -> &'static str {
        match self {
            Self::Name => "name",
            Self::Date => "date",
            Self::CaptureTime => "capture_time",
        }
    }
    fn from_key(key: &str) -> Self {
        match key {
            "date" => Self::Date,
            "capture_time" => Self::CaptureTime,
            _ => Self::Name,
        }
    }
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
    /// Centre the view on the next frame. An image opens where the last one was
    /// left otherwise, which is fine when fitting and nonsense at 100%.
    centre_view: bool,
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
    camera_category: camera_presets::CameraCategory,
    classifier: Option<Arc<classify::Classifier>>,
    enhancer: Option<Arc<enhance::Enhancer>>,
    auto_model: Option<Arc<auto::model::Residual>>,
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
    // Retouch brush: whether the tool is armed, and the dab radius in image
    // pixels (the `[` / `]` keys step it).
    retouch_active: bool,
    brush_radius: f32,
    // A short-lived note for the panel ("no source found…"), with the time it
    // was raised so it can expire.
    retouch_status: Option<(String, f64)>,
    // Two-step guard on "clear every dab".
    retouch_confirm_clear: bool,
    // Crop tool: whether it is armed, which aspect preset it locks to, and the
    // drag in flight.
    crop_active: bool,
    crop_aspect: usize,
    crop_drag: Option<ui::CropDrag>,
    // The UI state as it was last written to the database, so a change made
    // anywhere can be detected in one comparison.
    saved_ui: SavedUi,
    // Per-image undo/redo.
    history: Histories,
    // Export: the settings dialog, and a batch walking out one image at a time.
    show_export: bool,
    export_opts: ExportOptions,
    /// Sources still to write, with the target each was assigned when the batch
    /// was set up.
    export_queue: Vec<(PathBuf, PathBuf)>,
    /// `Some` while a batch is running; the directory is only kept so the batch
    /// knows it is in progress.
    export_dir: Option<PathBuf>,
    export_progress: Option<(usize, usize)>,
    /// Signalled by the worker when a file is written. Only one export is ever
    /// in flight: each job holds the whole frame's readback.
    export_done: Option<std::sync::mpsc::Receiver<()>>,
    // Print: the dialog, what CUPS said it has, and the last job's outcome.
    show_print: bool,
    // The CUPS dialog's state; macOS uses the system sheet instead.
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    print_opts: print::PrintOptions,
    /// `None` means whatever CUPS has as its default.
    // The CUPS dialog's state; macOS uses the system sheet instead.
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    print_printer: Option<String>,
    // The CUPS dialog's state; macOS uses the system sheet instead.
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    print_printers: Vec<String>,
    /// Which of them CUPS would use, so the dialog can say so.
    // The CUPS dialog's state; macOS uses the system sheet instead.
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    print_default: Option<String>,
    // The CUPS dialog's state; macOS uses the system sheet instead.
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    print_papers: Vec<print::Paper>,
    // The CUPS dialog's state; macOS uses the system sheet instead.
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    print_paper_types: Vec<print::PaperType>,
    /// The paper type last chosen, kept by value: the paper in the tray is
    // The CUPS dialog's state; macOS uses the system sheet instead.
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    /// the same paper next time, which is not true of a printer-specific index.
    print_paper_type: Option<String>,
    print_pending: Option<std::sync::mpsc::Receiver<Result<String, String>>>,
    /// Set while a render is on its way to AppKit. Where there is no system
    /// sheet the worker submits the job itself, so there is nothing to carry.
    #[cfg(target_os = "macos")]
    print_target: Option<PrintTarget>,
    /// A CUPS query in flight. The scheduler is asked on a worker so opening
    /// the dialog never waits on it; the payload says whether it was the
    /// opening query, and which printer the papers belong to.
    // The CUPS dialog's state; macOS uses the system sheet instead.
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    print_probe: Option<std::sync::mpsc::Receiver<(bool, print::Probe)>>,
    /// The last thing to report, and whether it went wrong.
    print_status: Option<(String, bool)>,
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
        let saved_ui = SavedUi::load(db.as_ref());
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
                centre_view: true,
                zoom_fit: saved_ui.zoom_fit,
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
            browse_dir: saved_ui.dir.clone(),
            thumbs: Vec::new(),
            thumb_rx: None,
            current_path: None,
            selected: None,
            meta,
            filter: saved_ui.filter,
            min_rating: saved_ui.min_rating,
            sort: saved_ui.sort,
            grid_cols: 1,
            tree_expanded,
            db,
            presets_dir,
            presets,
            preset_name: String::new(),
            camera_category: camera_presets::CameraCategory::Popular,
            classifier: classify::Classifier::load().map(Arc::new),
            enhancer: enhance::Enhancer::load().map(Arc::new),
            auto_model: auto::model::Residual::load().map(Arc::new),
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
            retouch_active: false,
            brush_radius: 40.0,
            retouch_status: None,
            retouch_confirm_clear: false,
            crop_active: false,
            crop_aspect: 0,
            crop_drag: None,
            export_opts: saved_ui.export,
            print_opts: saved_ui.print,
            print_paper_type: saved_ui.print_paper_type.clone(),
            saved_ui,
            history: Histories::default(),
            show_export: false,
            export_queue: Vec::new(),
            export_dir: None,
            export_progress: None,
            export_done: None,
            show_print: false,
            print_printer: None,
            print_printers: Vec::new(),
            print_default: None,
            print_papers: Vec::new(),
            print_paper_types: Vec::new(),
            print_pending: None,
            #[cfg(target_os = "macos")]
            print_target: None,
            print_probe: None,
            print_status: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let (width, height) = saved_window(self.db.as_ref());
        let attrs = Window::default_attributes()
            .with_title("Image Processor")
            .with_inner_size(LogicalSize::new(width, height))
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

        // CLI: open an image (or browse a folder) passed as the first argument.
        // It wins over the folder remembered from last session: asking for a
        // particular image on the command line and being shown a folder instead
        // is what the `browse_dir.is_none()` guard used to do.
        if self.current_path.is_none() {
            if let Some(arg) = std::env::args().nth(1).map(PathBuf::from) {
                if arg.is_file() {
                    self.register_image(&arg);
                } else if arg.is_dir() {
                    self.scan_folder(&arg);
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
            WindowEvent::CloseRequested => {
                if let Some(window) = &self.window {
                    let size = window.inner_size().to_logical::<f32>(window.scale_factor());
                    save_window(self.db.as_ref(), size.width, size.height);
                }
                event_loop.exit()
            }

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
        // Write the UI state back when it changes, rather than from every call
        // site that can change it.
        let ui = SavedUi {
            dir: self.browse_dir.clone(),
            zoom_fit: self.flags.zoom_fit,
            filter: self.filter,
            sort: self.sort,
            min_rating: self.min_rating,
            export: self.export_opts,
            print: self.print_opts,
            print_paper_type: self.print_paper_type.clone(),
        };
        if ui != self.saved_ui {
            ui.save(self.db.as_ref());
            self.saved_ui = ui;
        }
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

    /// Record the state on screen as an undo step. Called wherever an edit is
    /// persisted, so anything that can change an image can be undone.
    fn commit_history(&mut self, path: &Path, state: EditState) {
        let now = self.egui_ctx.input(|i| i.time);
        self.history.commit(path, state, now);
    }

    /// Forget the past for this image and start from `state` -- a fresh open,
    /// or an edit that should not be undoable.
    fn reset_history(&mut self, path: &Path, state: EditState) {
        self.history.reset(path, state);
    }

    /// Step one place through the open image's history.
    fn step_history(&mut self, forward: bool) {
        let Some(path) = self.current_path.clone() else {
            return;
        };
        let state = {
            let history = self.history.entry(&path);
            if forward {
                history.redo()
            } else {
                history.undo()
            }
        };
        let Some(state) = state else {
            return;
        };
        self.load_edit_state(&path, state);
    }

    /// Put `state` into the processor and redraw, mirroring what opening an
    /// image does -- including re-developing a RAW when the state calls for it,
    /// since that is part of the state and not just a slider.
    fn load_edit_state(&mut self, path: &Path, state: EditState) {
        let raw = imgload::is_raw(path);
        let Some(gpu) = self.gpu.as_ref() else {
            return;
        };
        let Some(proc) = self.processor.as_mut() else {
            return;
        };
        proc.apply_edit_state(&state);
        if state.raw_isp_enabled && raw {
            let max_dim = proc.image_size.map_or(2048, |(w, h)| w.max(h));
            if let Some(developed) = raw_develop::develop_raw_u16(path, max_dim) {
                proc.replace_input_u16(
                    developed.width(),
                    developed.height(),
                    developed.as_raw(),
                    &gpu.queue,
                );
            }
        }
        proc.process(&gpu.device, &gpu.queue);
        if let Some(db) = &self.db {
            save_edits(db, path, &proc.edit_state());
        }
        self.flags.output_dirty = true;
    }

    /// Render `path` with the edits saved for it, so a batch export writes what
    /// the editor would show rather than the camera's preview.
    fn render_saved(&mut self, path: &Path) -> bool {
        let state = self
            .db
            .as_ref()
            .and_then(|db| load_edits(db, path))
            .unwrap_or_default();
        let Some(img) = imgload::load_edit_rgba(path) else {
            return false;
        };
        let (Some(gpu), Some(proc)) = (self.gpu.as_ref(), self.processor.as_mut()) else {
            return false;
        };
        proc.apply_edit_state(&state);
        proc.upload_rgba(&img, &gpu.device, &gpu.queue);
        if state.raw_isp_enabled && imgload::is_raw(path) {
            let max_dim = proc.image_size.map_or(2048, |(w, h)| w.max(h));
            if let Some(developed) = raw_develop::develop_raw_u16(path, max_dim) {
                proc.replace_input_u16(
                    developed.width(),
                    developed.height(),
                    developed.as_raw(),
                    &gpu.queue,
                );
            }
        }
        proc.process(&gpu.device, &gpu.queue);
        true
    }

    /// Put the open image back on the GPU after a batch export went through the
    /// processor. Like opening it, but without disturbing its undo history.
    fn reopen_current(&mut self) {
        let Some(path) = self.current_path.clone() else {
            return;
        };
        // From the database, not from the processor: by now the processor is
        // holding the *last exported image's* state, and taking that as this
        // image's own is how a batch export used to overwrite the edits of
        // whatever was open.
        let state = self
            .db
            .as_ref()
            .and_then(|db| load_edits(db, &path))
            .unwrap_or_default();
        let Some(img) = imgload::load_edit_rgba(&path) else {
            return;
        };
        if let Some(gpu) = self.gpu.as_ref() {
            if let Some(proc) = self.processor.as_mut() {
                proc.upload_rgba(&img, &gpu.device, &gpu.queue);
            }
        }
        self.load_edit_state(&path, state);
        self.rebind_image_textures();
        self.flags.output_dirty = true;
    }

    /// Open the print dialog, asking CUPS what it has first.
    ///
    /// `lpstat` and `lpoptions` are one-off queries, so they run here rather
    /// than every frame. The paper list belongs to a printer, so it is fetched
    /// per printer and the choice starts again at the printer's own default.
    #[cfg(not(target_os = "macos"))]
    fn open_print_dialog(&mut self) {
        self.show_print = true;
        self.print_status = None;
        self.probe_printer(None, true);
    }

    /// Ask CUPS about the queues and/or one printer's papers, on a worker: a
    /// query takes ~70 ms here, and a scheduler that is not answering would
    /// otherwise block the window for as long as it liked. A second call
    /// replaces the first, which drops its receiver and so its answer with it.
    #[cfg(not(target_os = "macos"))]
    fn probe_printer(&mut self, printer: Option<String>, with_printers: bool) {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let probe = print::probe(printer.as_deref(), with_printers);
            let _ = tx.send((with_printers, probe));
        });
        self.print_probe = Some(rx);
    }

    /// Render the crop and hand it to CUPS, on a thread: the render and encode
    /// take a moment, and the outcome comes back through a channel so the
    /// dialog can report what `lp` said.
    #[cfg(not(target_os = "macos"))]
    fn start_print(&mut self) {
        // A folder export is driving the processor through other images, so
        // what is loaded is not what is on screen: rendering now would print
        // the wrong photograph. The dialog says so and disables the button;
        // this is the belt to that braces.
        if self.exporting() {
            self.print_status = Some((
                "An export is running — printing is paused until it finishes".to_owned(),
                true,
            ));
            return;
        }
        if self.print_pending.is_some() {
            return;
        }
        let Some(dir) = app_dir().filter(|dir| std::fs::create_dir_all(dir).is_ok()) else {
            self.print_status = Some(("No place to write the print file".to_owned(), true));
            return;
        };
        let Some((w, h)) = self.processor.as_ref().and_then(|p| p.image_size) else {
            return;
        };
        let options = self.print_opts;
        let paper = self
            .print_papers
            .get(options.paper)
            .cloned()
            .unwrap_or_else(print::Paper::printer_default);
        let paper_type = self
            .print_paper_types
            .get(options.paper_type)
            .cloned()
            .unwrap_or_else(|| self.print_paper_types[0].clone());
        // Render at most what the paper can use: this is a bound on the spool
        // file, not a claim about the printed resolution -- the printer scales
        // what it is given onto the page.
        let box_px = print::page_box(&paper, options.orientation.landscape(), print::PRINT_DPI);
        let (_, _, cw, ch) = self
            .processor
            .as_ref()
            .map(|p| processor::crop_px(p.crop, (w, h)))
            .unwrap_or((0, 0, w, h));
        let (tw, th) = print::fit_inside(cw, ch, box_px);
        let export = ExportOptions {
            format: ExportFormat::Jpeg,
            quality: 92,
            max_dim: tw.max(th),
            // The image was laid out for this paper at this resolution, so say
            // so: with the resolution recorded, "do not scale" means the size
            // this was prepared at rather than whatever the driver assumes. A
            // 4961 px frame for A4 is 8.27 inches at 600, which is the page.
            dpi: print::PRINT_DPI as u16,
        };
        // One fixed name, overwritten each time: prints are serialised by the
        // disabled Print button, so nothing can be reading it while it is
        // rewritten, and there is nothing left on disk to clean up.
        let path = dir.join("print.jpg");
        let job = match (self.processor.as_ref(), self.gpu.as_ref()) {
            (Some(proc), Some(gpu)) => {
                proc.export_job(&path, &export, &gpu.device, &gpu.queue)
            }
            _ => None,
        };
        let Some(job) = job else {
            self.print_status = Some(("Could not render the image".to_owned(), true));
            return;
        };
        let printer = self.print_printer.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            job.write();
            // Whatever CUPS says, including its own error text.
            let _ = tx.send(print::send(
                printer.as_deref(),
                &options,
                &paper,
                &paper_type,
                &path,
            ));
        });
        self.print_pending = Some(rx);
        self.print_status = Some(("Sending to the printer…".to_owned(), false));
    }

    /// Render the crop and then open the system print sheet (or write a PDF).
    ///
    /// The render is a job on a worker, as the export is; `finish_print` picks
    /// the bytes up on the main thread, because AppKit views may not be built
    /// anywhere else.
    #[cfg(target_os = "macos")]
    fn start_print(&mut self, target: PrintTarget) {
        if self.exporting() {
            self.print_status = Some((
                "An export is running — printing is paused until it finishes".to_owned(),
                true,
            ));
            return;
        }
        if self.print_pending.is_some() {
            return;
        }
        let Some(dir) = app_dir().filter(|dir| std::fs::create_dir_all(dir).is_ok()) else {
            self.print_status = Some(("No place to write the print file".to_owned(), true));
            return;
        };
        let Some((w, h)) = self.processor.as_ref().and_then(|p| p.image_size) else {
            return;
        };
        // How many pixels to render. The sheet chooses the paper and the
        // scaling, so this only has to be generous enough for any of them: a
        // page's worth at 600 dpi, and the image's own size if it is smaller.
        let box_px = print::page_box(&print::Paper::printer_default(), false, print::PRINT_DPI);
        let (_, _, cw, ch) = self
            .processor
            .as_ref()
            .map(|p| processor::crop_px(p.crop, (w, h)))
            .unwrap_or((0, 0, w, h));
        let (tw, th) = print::fit_inside(cw, ch, box_px);
        let options = ExportOptions {
            format: ExportFormat::Jpeg,
            quality: 92,
            max_dim: tw.max(th),
            // The resolution is what tells AppKit how big the image is, so
            // "do not scale" in the sheet means the size prepared here.
            dpi: print::PRINT_DPI as u16,
        };
        let path = dir.join("print.jpg");
        let job = match (self.processor.as_ref(), self.gpu.as_ref()) {
            (Some(proc), Some(gpu)) => proc.export_job(&path, &options, &gpu.device, &gpu.queue),
            _ => None,
        };
        let Some(job) = job else {
            self.print_status = Some(("Could not render the image".to_owned(), true));
            return;
        };
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            job.write();
            let _ = tx.send(Ok(path.to_string_lossy().into_owned()));
        });
        self.print_target = Some(target);
        self.print_pending = Some(rx);
        self.print_status = Some(("Preparing the print…".to_owned(), false));
    }

    /// The render is ready: hand it to AppKit, on this thread.
    #[cfg(target_os = "macos")]
    fn finish_print(&mut self, rendered: Result<String, String>) {
        let target = self.print_target.take().unwrap_or(PrintTarget::Sheet);
        let said = rendered.and_then(|path| {
            std::fs::read(&path)
                .map_err(|why| format!("could not read the render: {why}"))
                .and_then(|encoded| {
                    let page = print_macos::Page {
                        encoded,
                        // The sheet has its own scaling control; this is what
                        // the app asks for by default.
                        fit_to_page: true,
                    };
                    match target {
                        PrintTarget::Sheet => print_macos::run_dialog(&page),
                        PrintTarget::Pdf(out) => print_macos::to_pdf(&page)
                            .and_then(|bytes| {
                                std::fs::write(&out, bytes)
                                    .map_err(|why| format!("could not write the PDF: {why}"))
                            })
                            .map(|()| format!("Wrote {}", out.display())),
                    }
                })
        });
        self.print_status = Some(match said {
            Ok(message) => (message, false),
            Err(why) => (why, true),
        });
    }

    /// A batch export is running: the processor is being driven through other
    /// people's images, so nothing about the state on screen may be persisted.
    fn exporting(&self) -> bool {
        self.export_dir.is_some()
    }

    /// Write one queued image, then wait for it before starting the next.
    ///
    /// One at a time, because an [`processor::ExportJob`] carries a full-frame
    /// 16-bit readback -- around 190 MB for a 24 MP frame -- and letting those
    /// stack up is how a folder export runs out of memory. One *per frame*, so
    /// the window keeps drawing while it works.
    fn step_export_queue(&mut self) {
        if !self.exporting() {
            return;
        }
        if let Some(done_rx) = self.export_done.take() {
            match done_rx.try_recv() {
                Ok(()) | Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    let (done, total) = self.export_progress.unwrap_or((0, 1));
                    self.export_progress = Some((done + 1, total));
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    self.export_done = Some(done_rx);
                    return;
                }
            }
        }
        if self.export_queue.is_empty() {
            // Everything is written: put the open image back.
            self.export_dir = None;
            self.export_progress = None;
            // In its own image, this was the last one exported.
            self.reopen_current();
            return;
        }
        let (path, target) = self.export_queue.remove(0);
        let options = self.export_opts;
        let mut wrote = false;
        if self.render_saved(&path) {
            let job = match (self.processor.as_ref(), self.gpu.as_ref()) {
                (Some(proc), Some(gpu)) => {
                    proc.export_job(&target, &options, &gpu.device, &gpu.queue)
                }
                _ => None,
            };
            if let Some(job) = job {
                let (tx, rx) = std::sync::mpsc::channel();
                std::thread::spawn(move || {
                    job.write();
                    // A closed receiver just means the batch was stopped.
                    let _ = tx.send(());
                });
                self.export_done = Some(rx);
                wrote = true;
            }
        }
        if !wrote {
            // Unreadable, or the GPU could not ready it: count it and move on.
            let (done, total) = self.export_progress.unwrap_or((0, 1));
            self.export_progress = Some((done + 1, total));
        }
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

        // Start this image's undo history from the state it opened at, after
        // the RAW sharpening seeding -- undoing that is not a thing.
        if let Some(proc) = self.processor.as_ref() {
            self.reset_history(path, proc.edit_state());
        }
        self.flags.output_dirty = true;
        // Whatever zoom mode was last used, not always fit -- but a crop is
        // placed against the whole frame, so arm-fit wins while it is open.
        self.flags.zoom_fit = self.saved_ui.zoom_fit || self.crop_active;
        // 100% is meaningless at the last image's scale and offset.
        self.flags.centre_view = !self.flags.zoom_fit;
        self.retouch_status = None;
        self.retouch_confirm_clear = false;
        self.crop_drag = None;
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
        // Undo/redo run outside the egui frame, where re-rendering is safe.
        if requests.undo {
            self.step_history(false);
        }
        if requests.redo {
            self.step_history(true);
        }
        // Retouch layer: place a dab, drop one, or clear them all. Placing a
        // dab searches the image that is on screen right now for a patch to
        // copy, then stores the offset (and the colour correction that makes it
        // blend) so every later render repeats the gesture from the live pixels.
        if requests.retouch_clear {
            if let Some(proc) = self.processor.as_mut() {
                if !proc.retouch_spots.is_empty() {
                    proc.retouch_spots.clear();
                    *needs_process = true;
                }
            }
        }
        if let Some(index) = requests.retouch_remove {
            if let Some(proc) = self.processor.as_mut() {
                if index < proc.retouch_spots.len() {
                    proc.retouch_spots.remove(index);
                    *needs_process = true;
                }
            }
        }
        if let Some([x, y, radius]) = requests.retouch {
            let mut placed = false;
            let mut note = None;
            if let (Some(proc), Some(gpu)) = (self.processor.as_mut(), self.gpu.as_ref()) {
                if proc.retouch_spots.len() >= MAX_RETOUCH_SPOTS {
                    note = Some(format!("Retouch layer is full ({MAX_RETOUCH_SPOTS} dabs)"));
                } else if let Some((w, h)) = proc.image_size {
                    // Only the region the dab can reach is read back; that view
                    // covers every source the search will consider, so cropping
                    // loses nothing.
                    let (rx, ry, rw, rh) = retouch::search_rect(x, y, radius, w, h);
                    let r = retouch::radius_px(radius, w, h);
                    let centre = [x * w as f32, y * h as f32];
                    let sample = proc
                        .retouch_source_pixels(rx, ry, rw, rh, &gpu.device, &gpu.queue)
                        .and_then(|pixels| {
                            let view = retouch::View {
                                pixels: &pixels,
                                dims: (rw, rh),
                                origin: (rx as f32, ry as f32),
                                full: (w as f32, h as f32),
                            };
                            retouch::find_sample(&view, centre, r)
                        });
                    match sample {
                        Some(sample) => {
                            let (fw, fh) = (w as f32, h as f32);
                            proc.retouch_spots.push(RetouchSpot {
                                x,
                                y,
                                radius,
                                source: [sample.offset[0] / fw, sample.offset[1] / fh],
                                gain: sample.gain,
                                bias: sample.bias,
                            });
                            placed = true;
                        }
                        // Never fail silently: a click that did nothing used to
                        // read as the tool being broken.
                        None => note = Some("No source found — try a smaller brush".to_owned()),
                    }
                }
            }
            self.retouch_confirm_clear = false;
            self.retouch_status = note.map(|msg| (msg, self.egui_ctx.input(|i| i.time)));
            if placed {
                *needs_process = true;
            }
        }
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
        if requests.show_export {
            // One at a time: both are centred, and both act on "the image".
            self.show_print = false;
            self.show_export = true;
        }
        if requests.show_print {
            self.show_export = false;
            #[cfg(target_os = "macos")]
            self.start_print(PrintTarget::Sheet);
            #[cfg(not(target_os = "macos"))]
            self.open_print_dialog();
        }
        #[cfg(target_os = "macos")]
        if let Some(path) = requests.export_pdf {
            self.start_print(PrintTarget::Pdf(path));
        }
        if let Some(path) = requests.export_path {
            // Same reason as printing: a batch has another image loaded, so
            // this would write that one to the file the user chose.
            if self.exporting() {
                eprintln!("export ignored: a folder export is running");
            } else {
                let options = self.export_opts;
                if let (Some(proc), Some(gpu)) = (&self.processor, &self.gpu) {
                    proc.export(&path, &options, &gpu.device, &gpu.queue);
                }
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
            // Analytic Auto: choose the visible sliders from the photo's own
            // statistics. Falls back to the histogram heuristic only if the
            // source cannot be re-read for statistics.
            let auto_state = self
                .current_path
                .as_deref()
                .and_then(|path| imgload::load_rgba(path, 512))
                .map(|img| auto::suggest_edit_with(&img, self.auto_model.as_deref()));
            if let (Some(proc), Some(gpu)) = (self.processor.as_mut(), self.gpu.as_ref()) {
                if proc.has_image() {
                    proc.apply_edit_state(&EditState::default());
                    proc.restore_source(&gpu.queue);
                    if let Some(state) = auto_state {
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
                if !self.exporting() {
                    self.commit_history(&path, state.clone());
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
        self.step_export_queue();
        // What CUPS said about the printers, if the query has come back.
        if let Some(pending) = self.print_probe.take() {
            match pending.try_recv() {
                Ok((with_printers, probe)) => {
                    if with_printers {
                        self.print_printers = probe.printers;
                        self.print_default = probe.default;
                        // Paper belongs to a printer, so the choice starts again
                        // at the one these papers came from.
                        self.print_opts.paper = 0;
                    }
                    self.print_printer = probe.printer;
                    self.print_papers = probe.papers;
                    self.print_paper_types = probe.paper_types;
                    // A printer-specific index means nothing on another
                    // printer, so the remembered choice is a value.
                    self.print_opts.paper_type = self
                        .print_paper_type
                        .as_deref()
                        .and_then(|wanted| {
                            self.print_paper_types
                                .iter()
                                .position(|t| t.value.as_deref() == Some(wanted))
                        })
                        .unwrap_or(0);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => self.print_probe = Some(pending),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {}
            }
        }
        // Whatever `lp` had to say about the last job.
        if let Some(pending) = self.print_pending.take() {
            match pending.try_recv() {
                Ok(result) => {
                    #[cfg(target_os = "macos")]
                    self.finish_print(result);
                    #[cfg(not(target_os = "macos"))]
                    {
                        self.print_status = Some(match result {
                            Ok(said) => (said, false),
                            Err(why) => (why, true),
                        });
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => self.print_pending = Some(pending),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {}
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
                if self.show_export {
                    let mut close = false;
                    let mut export_path: Option<PathBuf> = None;
                    let mut batch: Option<(Vec<PathBuf>, PathBuf)> = None;
                    // Picks if there are any, otherwise everything on screen.
                    let picks: Vec<PathBuf> = visible
                        .iter()
                        .map(|&i| self.thumbs[i].path.clone())
                        .filter(|p| {
                            self.meta
                                .get(p)
                                .is_some_and(|m| m.flag == cull::Flag::Pick)
                        })
                        .collect();
                    let batch_paths: Vec<PathBuf> = if picks.is_empty() {
                        visible
                            .iter()
                            .map(|&i| self.thumbs[i].path.clone())
                            .collect()
                    } else {
                        picks.clone()
                    };
                    egui::Window::new("Export")
                        .collapsible(false)
                        .resizable(false)
                        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                        .show(ctx, |ui| {
                            ui.set_min_width(320.0);
                            ui.horizontal(|ui| {
                                for format in [ExportFormat::Png16, ExportFormat::Jpeg] {
                                    ui.selectable_value(
                                        &mut self.export_opts.format,
                                        format,
                                        format.label(),
                                    );
                                }
                            });
                            if self.export_opts.format == ExportFormat::Jpeg {
                                ui.add(
                                    egui::Slider::new(&mut self.export_opts.quality, 1..=100)
                                        .text("Quality"),
                                );
                            }
                            ui.horizontal(|ui| {
                                ui.label("Longest side");
                                for (label, value) in
                                    [("Original", 0u32), ("1024", 1024), ("2048", 2048), ("4096", 4096)]
                                {
                                    ui.selectable_value(&mut self.export_opts.max_dim, value, label);
                                }
                            });
                            if self.export_opts.format == ExportFormat::Jpeg {
                                ui.horizontal(|ui| {
                                    ui.label("Resolution");
                                    ui.add(
                                        egui::DragValue::new(&mut self.export_opts.dpi)
                                            .range(0..=1200)
                                            .suffix(" dpi"),
                                    )
                                    .on_hover_text(
                                        "Recorded in the file so its print size is known. 0 \
                                         records nothing, and the size is then whatever the \
                                         software opening it assumes.",
                                    );
                                });
                            }
                            if let Some(proc) = self.processor.as_ref() {
                                if let Some((w, h)) = proc.export_size(&self.export_opts) {
                                    let cropped = proc
                                        .image_size
                                        .is_some_and(|size| !processor::is_full_crop(proc.crop, size));
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "{w} × {h} px{}",
                                            if cropped { " · cropped" } else { "" }
                                        ))
                                        .small()
                                        .color(egui::Color32::from_gray(150)),
                                    );
                                }
                            }
                            ui.add_space(4.0);
                            let has_image = self.processor.as_ref().is_some_and(Processor::has_image)
                                && !self.exporting();
                            if self.exporting() {
                                ui.label(
                                    egui::RichText::new(
                                        "A folder export is running — exporting waits for it",
                                    )
                                    .small()
                                    .color(egui::Color32::from_rgb(230, 170, 90)),
                                );
                            }
                            ui.horizontal_wrapped(|ui| {
                                if ui
                                    .add_enabled(has_image, egui::Button::new("Export this image…"))
                                    .clicked()
                                {
                                    let stem = self
                                        .current_path
                                        .as_ref()
                                        .and_then(|p| p.file_stem())
                                        .and_then(|s| s.to_str())
                                        .unwrap_or("output")
                                        .to_owned();
                                    export_path = rfd::FileDialog::new()
                                        .add_filter(
                                            self.export_opts.format.label(),
                                            &[self.export_opts.format.extension()],
                                        )
                                        .set_file_name(format!(
                                            "{stem}.{}",
                                            self.export_opts.format.extension()
                                        ))
                                        .save_file();
                                }
                                let label = if picks.is_empty() {
                                    format!("Export {} visible…", batch_paths.len())
                                } else {
                                    format!("Export {} picked…", batch_paths.len())
                                };
                                if ui
                                    .add_enabled(!batch_paths.is_empty(), egui::Button::new(label))
                                    .clicked()
                                {
                                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                                        batch = Some((batch_paths.clone(), dir));
                                    }
                                }
                                if ui.button("Close").clicked() {
                                    close = true;
                                }
                            });
                            if let Some((done, total)) = self.export_progress {
                                ui.label(format!("Exporting {done} of {total}…"));
                                if ui.button("Stop").clicked() {
                                    // Anything already sent still lands; the
                                    // rest of the queue is dropped.
                                    self.export_queue.clear();
                                }
                            }
                        });
                    if let Some(path) = export_path {
                        ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_EXPORT_PATH), path));
                        close = true;
                    }
                    if let Some((paths, dir)) = batch {
                        // Targets are settled up front, so two sources that
                        // share a stem cannot land on one file.
                        let targets =
                            export_targets(&dir, &paths, self.export_opts.format.extension());
                        self.export_progress = Some((0, paths.len()));
                        self.export_dir = Some(dir);
                        self.export_queue = paths.into_iter().zip(targets).collect();
                        self.export_done = None;
                        close = true;
                    }
                    if close {
                        self.show_export = false;
                    }
                }
                #[cfg(not(target_os = "macos"))]
                ui::print_dialog(ctx, self);
                if self.exporting() && !self.show_export {
                    // Say so: the processor is being driven through other
                    // images, and edits made now are deliberately not kept.
                    egui::Window::new("exporting")
                        .title_bar(false)
                        .collapsible(false)
                        .resizable(false)
                        .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-16.0, 40.0))
                        .show(ctx, |ui| {
                            let (done, total) = self.export_progress.unwrap_or((0, 1));
                            ui.label(format!("Exporting image {done} of {total}…"));
                            ui.label(
                                egui::RichText::new("Editing is paused until it finishes")
                                    .small()
                                    .color(egui::Color32::from_gray(160)),
                            );
                            ui.add_space(4.0);
                            if ui.button("Stop").clicked() {
                                self.export_queue.clear();
                            }
                        });
                }
                if !ctx.wants_keyboard_input() {
                    // Undo/redo. `command` is Cmd on macOS and Ctrl elsewhere,
                    // so both the platform conventions work; Ctrl+Y redoes for
                    // the Windows habit.
                    let (undo, redo) = ctx.input(|i| {
                        let chord = |key: egui::Key| {
                            (i.modifiers.command || i.modifiers.ctrl) && i.key_pressed(key)
                        };
                        (
                            chord(egui::Key::Z) && !i.modifiers.shift,
                            (chord(egui::Key::Z) && i.modifiers.shift) || chord(egui::Key::Y),
                        )
                    });
                    if undo {
                        ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_UNDO), true));
                    }
                    if redo {
                        ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_REDO), true));
                    }
                    if ctx.input(|i| (i.modifiers.command || i.modifiers.ctrl) && i.key_pressed(egui::Key::E))
                    {
                        ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_SHOW_EXPORT), true));
                    }
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
            // While a batch export is running the processor holds one of *its*
            // images, so saving now would write that state over the open
            // image's -- and push it onto the open image's undo stack.
            if !self.exporting() {
                let state = self.processor.as_ref().map(Processor::edit_state);
                if let (Some(db), Some(path), Some(state)) =
                    (&self.db, &self.current_path, state.as_ref())
                {
                    save_edits(db, path, state);
                }
                let path = self.current_path.clone();
                if let (Some(path), Some(state)) = (path, state) {
                    self.commit_history(&path, state);
                }
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
        export_targets, print, ExportFormat, ExportOptions, Histories, ImageHistory, HISTORY_IMAGES,
        HISTORY_STEPS,
        save_edits, save_f32_pref, save_look, save_pref, saved_window, BrowseFilter, BrowseSort,
        CapturedLook, SavedUi, DEFAULT_WINDOW, UI_FILTER_KEY, UI_MIN_RATING_KEY, UI_WINDOW_H_KEY,
        UI_WINDOW_W_KEY,
    };
    use std::path::{Path, PathBuf};
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

    // The UI state has to come back the way it was left, and a row that is
    // missing or garbage must fall back to defaults rather than fail a launch.
    #[test]
    fn ui_state_round_trips_and_survives_a_bad_row() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE app_kv (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );",
        )
        .unwrap();
        assert_eq!(SavedUi::load(Some(&conn)), SavedUi::default());

        let saved = SavedUi {
            dir: Some(PathBuf::from("/Users/someone/Pictures")),
            zoom_fit: false,
            filter: BrowseFilter::Picks,
            sort: BrowseSort::CaptureTime,
            min_rating: 4,
            export: ExportOptions {
                format: ExportFormat::Jpeg,
                quality: 72,
                max_dim: 2048,
                dpi: 240,
            },
            print: print::PrintOptions {
                // Both indices are per-printer: only the type's *value* is
                // remembered (below), and the index is resolved from it when
                // the printer's options arrive.
                paper: 0,
                paper_type: 0,
                orientation: print::Orientation::Landscape,
                quality: print::Quality::High,
                copies: 1,
                fit_to_page: false,
            },
            print_paper_type: Some("com.canon-mtglossy".to_owned()),
        };
        saved.save(Some(&conn));
        assert_eq!(SavedUi::load(Some(&conn)), saved);

        // A cleared directory stays cleared rather than resurrecting the old one.
        let cleared = SavedUi {
            dir: None,
            ..saved.clone()
        };
        cleared.save(Some(&conn));
        assert_eq!(SavedUi::load(Some(&conn)), cleared);

        // The paper *size* is a per-printer choice and is deliberately not
        // remembered -- an index saved against one printer's list would name a
        // different sheet on another's -- but the paper *type* is remembered by
        // value, because the paper in the tray is the same paper next time.
        assert_eq!(
            SavedUi::load(Some(&conn)).print_paper_type.as_deref(),
            Some("com.canon-mtglossy")
        );
        let mut with_paper = saved.clone();
        with_paper.print.paper = 5;
        with_paper.save(Some(&conn));
        assert_eq!(SavedUi::load(Some(&conn)).print.paper, 0);

        // Garbage in the numeric rows is ignored, and an absurd window size is
        // refused.
        save_pref(&conn, UI_MIN_RATING_KEY, "not a number");
        save_pref(&conn, UI_FILTER_KEY, "nonsense");
        assert_eq!(SavedUi::load(Some(&conn)).min_rating, 0);
        assert_eq!(SavedUi::load(Some(&conn)).filter, BrowseFilter::All);
        save_f32_pref(&conn, UI_WINDOW_W_KEY, -5.0);
        assert_eq!(saved_window(Some(&conn)), DEFAULT_WINDOW);
        save_f32_pref(&conn, UI_WINDOW_W_KEY, 1280.0);
        save_f32_pref(&conn, UI_WINDOW_H_KEY, 800.0);
        assert_eq!(saved_window(Some(&conn)), (1280.0, 800.0));
        // No database at all is not an error.
        assert_eq!(SavedUi::load(None), SavedUi::default());
        assert_eq!(saved_window(None), DEFAULT_WINDOW);
    }

    #[test]
    fn browse_filter_and_sort_keys_round_trip() {
        for filter in [
            BrowseFilter::All,
            BrowseFilter::Picks,
            BrowseFilter::Rejects,
            BrowseFilter::Unflagged,
        ] {
            assert_eq!(BrowseFilter::from_key(filter.key()), filter);
        }
        for sort in [BrowseSort::Name, BrowseSort::Date, BrowseSort::CaptureTime] {
            assert_eq!(BrowseSort::from_key(sort.key()), sort);
        }
    }

    fn state_with_exposure(exposure: f32) -> EditState {
        let mut state = EditState::default();
        state.exposure = exposure;
        state
    }

    #[test]
    fn undo_history_walks_back_and_forth() {
        let mut history = ImageHistory::default();
        history.reset(state_with_exposure(0.0));
        // Distinct edits, spaced out beyond the merge window.
        for (i, exposure) in [1.0f32, 2.0, 3.0].iter().enumerate() {
            history.commit(state_with_exposure(*exposure), 10.0 + i as f64);
        }
        assert_eq!(history.undo().unwrap().exposure, 2.0);
        assert_eq!(history.undo().unwrap().exposure, 1.0);
        assert_eq!(history.undo().unwrap().exposure, 0.0);
        assert!(history.undo().is_none(), "nothing before the opened state");
        assert_eq!(history.redo().unwrap().exposure, 1.0);
        assert_eq!(history.redo().unwrap().exposure, 2.0);
        assert_eq!(history.redo().unwrap().exposure, 3.0);
        assert!(history.redo().is_none());
    }

    // A slider drag arrives as one change per frame; that is one undo step, not
    // sixty, and the state it started from has to stay reachable.
    #[test]
    fn a_burst_of_changes_is_a_single_step() {
        let mut history = ImageHistory::default();
        history.reset(state_with_exposure(0.0));
        for step in 1..=60 {
            history.commit(state_with_exposure(step as f32 * 0.1), 5.0 + step as f64 * 0.01);
        }
        assert_eq!(history.states.len(), 2, "the drag should be one step");
        assert_eq!(history.undo().unwrap().exposure, 0.0);
        // Editing after an undo starts a fresh step instead of merging into the
        // one that was just stepped back to.
        history.commit(state_with_exposure(9.0), 5.7);
        assert_eq!(history.states.len(), 2);
        assert_eq!(history.undo().unwrap().exposure, 0.0);
    }

    // Regression: a control that sets the dirty flag without changing anything
    // used to spend an undo step, so undo looked broken -- press it and nothing
    // moves. A no-op must also leave the redo branch alone.
    #[test]
    fn an_unchanged_commit_is_not_an_undo_step() {
        let mut history = ImageHistory::default();
        history.reset(state_with_exposure(1.0));
        history.commit(state_with_exposure(1.0), 10.0);
        history.commit(state_with_exposure(1.0), 20.0);
        assert_eq!(history.states.len(), 1, "no step for an unchanged state");
        // Back-step, change, then re-commit the same value it started from.
        history.commit(state_with_exposure(2.0), 30.0);
        assert_eq!(history.states.len(), 2);
        history.commit(state_with_exposure(2.0), 40.0);
        assert_eq!(history.states.len(), 2);
        assert_eq!(history.undo().unwrap().exposure, 1.0);
        history.commit(state_with_exposure(2.0), 50.0);
        assert!(history.redo().is_none(), "the no-op ate the redo branch");
    }

    // Regression: two sources sharing a stem in one folder (a RAW and its JPEG)
    // used to be written to the same file, the second replacing the first.
    #[test]
    fn batch_targets_are_unique_even_for_repeated_stems() {
        let dir = Path::new("/tmp/exports");
        let sources = vec![
            PathBuf::from("/a/DSC0001.ARW"),
            PathBuf::from("/a/DSC0001.JPG"),
            PathBuf::from("/a/DSC0002.ARW"),
            PathBuf::from("/b/DSC0001.NEF"),
        ];
        let targets = export_targets(dir, &sources, "jpg");
        assert_eq!(targets[0], dir.join("DSC0001.jpg"));
        assert_eq!(targets[1], dir.join("DSC0001-2.jpg"));
        assert_eq!(targets[2], dir.join("DSC0002.jpg"));
        assert_eq!(targets[3], dir.join("DSC0001-3.jpg"));
        let unique: std::collections::HashSet<_> = targets.iter().collect();
        assert_eq!(unique.len(), sources.len());
    }

    #[test]
    fn a_new_edit_drops_the_redo_branch() {
        let mut history = ImageHistory::default();
        history.reset(state_with_exposure(0.0));
        history.commit(state_with_exposure(1.0), 10.0);
        history.commit(state_with_exposure(2.0), 11.0);
        assert_eq!(history.undo().unwrap().exposure, 1.0);
        history.commit(state_with_exposure(5.0), 12.0);
        assert!(history.redo().is_none(), "the old branch is gone");
        assert_eq!(history.undo().unwrap().exposure, 1.0);
    }

    #[test]
    fn the_history_is_capped_and_keeps_the_newest_steps() {
        let mut history = ImageHistory::default();
        history.reset(state_with_exposure(0.0));
        for step in 1..=HISTORY_STEPS * 3 {
            history.commit(state_with_exposure(step as f32), step as f64);
        }
        assert_eq!(history.states.len(), HISTORY_STEPS);
        assert_eq!(history.at, HISTORY_STEPS - 1);
        let mut steps = 0;
        while history.undo().is_some() {
            steps += 1;
        }
        assert_eq!(steps, HISTORY_STEPS - 1);
    }

    #[test]
    fn each_image_gets_its_own_history() {
        let mut histories = Histories::default();
        let a = Path::new("/tmp/a.jpg");
        let b = Path::new("/tmp/b.jpg");
        histories.reset(a, state_with_exposure(0.0));
        histories.commit(a, state_with_exposure(1.0), 10.0);
        histories.reset(b, state_with_exposure(7.0));
        assert_eq!(histories.entry(a).undo().unwrap().exposure, 0.0);
        assert!(histories.entry(b).undo().is_none());
        // Only a bounded number of images are remembered.
        for i in 0..HISTORY_IMAGES * 2 {
            let other = PathBuf::from(format!("/tmp/o{i}.jpg"));
            histories.entry(&other);
        }
        assert!(histories.images.len() <= HISTORY_IMAGES);
    }

    // The export size is the crop, then the size limit -- both of which the
    // dialog reports before anything is written.
    #[test]
    fn export_sizes_shrink_to_the_limit_and_keep_the_aspect() {
        assert_eq!(super::processor::fit_within(6000, 4000, 0), (6000, 4000));
        assert_eq!(super::processor::fit_within(6000, 4000, 2048), (2048, 1365));
        assert_eq!(super::processor::fit_within(4000, 6000, 3000), (2000, 3000));
        assert_eq!(super::processor::fit_within(800, 600, 2048), (800, 600));
        assert_eq!(super::processor::fit_within(3000, 2000, 1000), (1000, 667));
        // Never zero, however extreme the ratio.
        assert_eq!(super::processor::fit_within(100000, 1, 1), (1, 1));
    }

    #[test]
    fn export_format_keys_round_trip() {
        for format in [ExportFormat::Png16, ExportFormat::Jpeg] {
            assert_eq!(ExportFormat::from_key(format.key()), format);
        }
        assert_eq!(ExportFormat::from_key("nonsense"), ExportFormat::Png16);
    }

    #[test]
    fn bool_pref_roundtrip() {        let conn = rusqlite::Connection::open_in_memory().unwrap();
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
