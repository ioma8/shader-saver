# Culling Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the GPU editor into a real photo culler: ratings/flags/labels with hotkeys, filmstrip + arrow navigation, neighbor prefetching with fast RAW preview opens, browse filter/sort, and act-on-verdict file operations (trash/move rejects, copy picks).

**Architecture:** Culling metadata lives in a new `meta` SQLite table (separate from `edits` so "Reset All Edits" never touches culling decisions) mirrored into an in-memory `HashMap<PathBuf, CullMeta>` — no DB reads during paint. All pure logic (meta store, actions, filter/sort ordering) goes in a new `src/cull.rs` with unit tests. UI changes follow the existing `main.rs` patterns: split field borrows into the egui closure, deferred actions via `egui::Id` temp data (like `KEY_OPEN_PATH`) or values returned from the frame-scope tuple. Image loading gets a fast path (`load_edit_rgba`: embedded JPEG for RAW) plus a background full-demosaic swap, and a `src/prefetch.rs` worker decodes prev/next neighbors ahead of navigation.

**Tech Stack:** Rust, wgpu 22, egui 0.29 (verified: `Key::Num0..Num9`, `Key::P/X`, `Key::ArrowLeft/ArrowRight/Enter` exist), rusqlite (bundled — in-memory DBs for tests), `image`, `jpgfromraw`, new dep: `trash = "5"`.

**Conventions for every task:**
- Run tests with `cargo test` (binary crate — `#[cfg(test)]` modules work fine).
- Build with `cargo build` (debug is fine for iteration); manual GUI verification with `cargo run --release -- <folder>` against a folder of mixed JPEG/RAW photos.
- Commit after every task with the message given in the task.
- The app repaints continuously (`about_to_wait` → `request_redraw`), so background threads never need `ctx.request_repaint()` except where the existing thumbnail loader already does it.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/cull.rs` | **create** | `CullMeta` (rating/flag/label), `CullAction` + `apply_action`, `Filter`/`Sort` + `visible_indices`, SQLite `meta` table CRUD, edits/meta row ops for file moves. Pure + DB logic, fully unit-tested, no egui/wgpu deps. |
| `src/prefetch.rs` | **create** | `Prefetcher`: background decode worker + bounded cache of neighbor images. No egui/wgpu deps; testable. |
| `src/imgload.rs` | modify | Add `load_edit_rgba` (full-res fast path: embedded JPEG for RAW, flagged preview-quality). |
| `src/processor.rs` | modify | Split `load_image` into `upload_rgba(&RgbaImage, …)`; callers decode themselves. |
| `src/main.rs` | modify | New App fields (meta map, selection, filter/sort, prefetcher, demosaic-swap channel), keyboard shortcuts, badges, star row, filmstrip panel, filter/sort toolbar, cull-action buttons + confirm dialog, post-frame file ops. |
| `Cargo.toml` | modify | Add `trash = "5"`. |

---

### Task 1: Culling metadata module (`src/cull.rs`) — types, actions, SQLite

**Files:**
- Create: `src/cull.rs`
- Modify: `src/main.rs:1-2` (add `mod cull;`), `src/main.rs:23-37` (`open_db` calls `cull::init_meta_table`)

- [ ] **Step 1: Create `src/cull.rs` with types and failing tests**

Write the full module with types, action logic, and DB functions left as stubs that panic, plus the tests. (Types must exist for the tests to compile; the stubs make them fail.)

```rust
// Culling metadata (ratings / flags / labels), stored in its own SQLite table
// so "Reset All Edits" never touches culling decisions.
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum Flag {
    #[default]
    None,
    Pick,
    Reject,
}

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum Label {
    #[default]
    None,
    Red,
    Yellow,
    Green,
    Blue,
}

#[derive(Clone, Copy, PartialEq, Default, Debug)]
pub struct CullMeta {
    pub rating: u8, // 0..=5
    pub flag: Flag,
    pub label: Label,
}

impl CullMeta {
    pub fn is_default(self) -> bool {
        self == Self::default()
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum CullAction {
    Rating(u8),
    TogglePick,
    ToggleReject,
    ToggleLabel(Label),
}

pub fn apply_action(meta: &mut CullMeta, action: CullAction) {
    todo!()
}

pub fn init_meta_table(conn: &rusqlite::Connection) {
    todo!()
}

pub fn save_meta(conn: &rusqlite::Connection, path: &Path, meta: CullMeta) {
    todo!()
}

pub fn load_all_meta(conn: &rusqlite::Connection) -> HashMap<PathBuf, CullMeta> {
    todo!()
}

// Row maintenance for file operations (move / copy / delete). These touch BOTH
// the edits and meta tables so a moved file keeps its edits and its culling state.
pub fn delete_rows(conn: &rusqlite::Connection, path: &Path) {
    todo!()
}

pub fn rekey_rows(conn: &rusqlite::Connection, old: &Path, new: &Path) {
    todo!()
}

pub fn copy_rows(conn: &rusqlite::Connection, old: &Path, new: &Path) {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> rusqlite::Connection {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        init_meta_table(&conn);
        // mirror of the edits table created in main.rs::open_db
        conn.execute_batch(
            "CREATE TABLE edits (path TEXT PRIMARY KEY, params TEXT NOT NULL, updated INTEGER NOT NULL);",
        )
        .unwrap();
        conn
    }

    #[test]
    fn meta_roundtrip() {
        let conn = db();
        let m = CullMeta { rating: 4, flag: Flag::Pick, label: Label::Green };
        save_meta(&conn, Path::new("/a/img1.jpg"), m);
        let all = load_all_meta(&conn);
        assert_eq!(all.get(Path::new("/a/img1.jpg")), Some(&m));
    }

    #[test]
    fn default_meta_deletes_row() {
        let conn = db();
        save_meta(&conn, Path::new("/a/img1.jpg"), CullMeta { rating: 2, ..Default::default() });
        save_meta(&conn, Path::new("/a/img1.jpg"), CullMeta::default());
        assert!(load_all_meta(&conn).is_empty());
    }

    #[test]
    fn actions_toggle() {
        let mut m = CullMeta::default();
        apply_action(&mut m, CullAction::Rating(3));
        assert_eq!(m.rating, 3);
        apply_action(&mut m, CullAction::Rating(3)); // same rating again clears
        assert_eq!(m.rating, 0);
        apply_action(&mut m, CullAction::Rating(0)); // 0 always clears
        assert_eq!(m.rating, 0);
        apply_action(&mut m, CullAction::TogglePick);
        assert_eq!(m.flag, Flag::Pick);
        apply_action(&mut m, CullAction::ToggleReject); // reject replaces pick
        assert_eq!(m.flag, Flag::Reject);
        apply_action(&mut m, CullAction::ToggleReject); // toggle off
        assert_eq!(m.flag, Flag::None);
        apply_action(&mut m, CullAction::ToggleLabel(Label::Red));
        assert_eq!(m.label, Label::Red);
        apply_action(&mut m, CullAction::ToggleLabel(Label::Red));
        assert_eq!(m.label, Label::None);
    }

    #[test]
    fn rekey_copy_delete_rows() {
        let conn = db();
        conn.execute("INSERT INTO edits (path, params, updated) VALUES ('/a/x.jpg', '{}', 1)", [])
            .unwrap();
        save_meta(&conn, Path::new("/a/x.jpg"), CullMeta { rating: 5, ..Default::default() });

        copy_rows(&conn, Path::new("/a/x.jpg"), Path::new("/b/x.jpg"));
        assert_eq!(load_all_meta(&conn).len(), 2);
        let n: i64 = conn.query_row("SELECT COUNT(*) FROM edits", [], |r| r.get(0)).unwrap();
        assert_eq!(n, 2);

        rekey_rows(&conn, Path::new("/a/x.jpg"), Path::new("/c/x.jpg"));
        let all = load_all_meta(&conn);
        assert!(all.contains_key(Path::new("/c/x.jpg")));
        assert!(!all.contains_key(Path::new("/a/x.jpg")));

        delete_rows(&conn, Path::new("/c/x.jpg"));
        let n: i64 = conn.query_row("SELECT COUNT(*) FROM edits", [], |r| r.get(0)).unwrap();
        assert_eq!(n, 1); // only the copied /b/x.jpg row remains
        assert_eq!(load_all_meta(&conn).len(), 1);
    }
}
```

Add to `src/main.rs` line 1 (next to `mod imgload;`):

```rust
mod cull;
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test cull -- --nocapture`
Expected: FAIL — each test panics with `not yet implemented` (from `todo!()`).

- [ ] **Step 3: Implement the module body**

Replace the `todo!()` stubs:

```rust
pub fn apply_action(meta: &mut CullMeta, action: CullAction) {
    match action {
        CullAction::Rating(n) => {
            meta.rating = if meta.rating == n { 0 } else { n.min(5) };
        }
        CullAction::TogglePick => {
            meta.flag = if meta.flag == Flag::Pick { Flag::None } else { Flag::Pick };
        }
        CullAction::ToggleReject => {
            meta.flag = if meta.flag == Flag::Reject { Flag::None } else { Flag::Reject };
        }
        CullAction::ToggleLabel(l) => {
            meta.label = if meta.label == l { Label::None } else { l };
        }
    }
}

fn flag_to_i(f: Flag) -> i64 {
    match f { Flag::None => 0, Flag::Pick => 1, Flag::Reject => 2 }
}
fn flag_from_i(i: i64) -> Flag {
    match i { 1 => Flag::Pick, 2 => Flag::Reject, _ => Flag::None }
}
fn label_to_i(l: Label) -> i64 {
    match l { Label::None => 0, Label::Red => 1, Label::Yellow => 2, Label::Green => 3, Label::Blue => 4 }
}
fn label_from_i(i: i64) -> Label {
    match i { 1 => Label::Red, 2 => Label::Yellow, 3 => Label::Green, 4 => Label::Blue, _ => Label::None }
}

pub fn init_meta_table(conn: &rusqlite::Connection) {
    let _ = conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS meta (
            path    TEXT PRIMARY KEY,
            rating  INTEGER NOT NULL DEFAULT 0,
            flag    INTEGER NOT NULL DEFAULT 0,
            label   INTEGER NOT NULL DEFAULT 0,
            updated INTEGER NOT NULL
        );",
    );
}

pub fn save_meta(conn: &rusqlite::Connection, path: &Path, meta: CullMeta) {
    let p = path.to_string_lossy();
    if meta.is_default() {
        // Keep the table sparse: default state == no row
        let _ = conn.execute("DELETE FROM meta WHERE path = ?1", rusqlite::params![p]);
        return;
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let _ = conn.execute(
        "INSERT INTO meta (path, rating, flag, label, updated) VALUES (?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(path) DO UPDATE SET rating = ?2, flag = ?3, label = ?4, updated = ?5",
        rusqlite::params![p, meta.rating as i64, flag_to_i(meta.flag), label_to_i(meta.label), now],
    );
}

pub fn load_all_meta(conn: &rusqlite::Connection) -> HashMap<PathBuf, CullMeta> {
    let mut map = HashMap::new();
    let Ok(mut stmt) = conn.prepare("SELECT path, rating, flag, label FROM meta") else {
        return map;
    };
    let rows = stmt.query_map([], |r| {
        Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?, r.get::<_, i64>(3)?))
    });
    if let Ok(rows) = rows {
        for (path, rating, flag, label) in rows.flatten() {
            map.insert(
                PathBuf::from(path),
                CullMeta {
                    rating: rating.clamp(0, 5) as u8,
                    flag: flag_from_i(flag),
                    label: label_from_i(label),
                },
            );
        }
    }
    map
}

pub fn delete_rows(conn: &rusqlite::Connection, path: &Path) {
    let p = path.to_string_lossy();
    let _ = conn.execute("DELETE FROM edits WHERE path = ?1", rusqlite::params![p]);
    let _ = conn.execute("DELETE FROM meta  WHERE path = ?1", rusqlite::params![p]);
}

pub fn rekey_rows(conn: &rusqlite::Connection, old: &Path, new: &Path) {
    let (o, n) = (old.to_string_lossy(), new.to_string_lossy());
    let _ = conn.execute("UPDATE OR REPLACE edits SET path = ?2 WHERE path = ?1", rusqlite::params![o, n]);
    let _ = conn.execute("UPDATE OR REPLACE meta  SET path = ?2 WHERE path = ?1", rusqlite::params![o, n]);
}

pub fn copy_rows(conn: &rusqlite::Connection, old: &Path, new: &Path) {
    let (o, n) = (old.to_string_lossy(), new.to_string_lossy());
    let _ = conn.execute(
        "INSERT OR REPLACE INTO edits (path, params, updated)
         SELECT ?2, params, updated FROM edits WHERE path = ?1",
        rusqlite::params![o, n],
    );
    let _ = conn.execute(
        "INSERT OR REPLACE INTO meta (path, rating, flag, label, updated)
         SELECT ?2, rating, flag, label, updated FROM meta WHERE path = ?1",
        rusqlite::params![o, n],
    );
}
```

In `src/main.rs::open_db` (line 23), add after the `edits` `execute_batch`:

```rust
    cull::init_meta_table(&conn);
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test cull`
Expected: `test result: ok. 4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/cull.rs src/main.rs
git commit -m "feat: culling metadata module (ratings/flags/labels) with SQLite meta table"
```

---

### Task 2: Filter & sort logic (`cull::visible_indices`)

**Files:**
- Modify: `src/cull.rs` (append)

- [ ] **Step 1: Write the failing test**

Append to `src/cull.rs` (above the `tests` module):

```rust
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum FlagFilter {
    #[default]
    All,
    Picks,
    Rejects,
    Unflagged,
}

#[derive(Clone, Copy, PartialEq, Default, Debug)]
pub struct Filter {
    pub flags: FlagFilter,
    pub min_rating: u8,
}

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum Sort {
    #[default]
    Name,
    Date,
}

// Lightweight per-image view the App builds from its ThumbEntry list each frame.
pub struct Item<'a> {
    pub path: &'a Path,
    pub mtime: Option<std::time::SystemTime>,
}

pub fn passes(meta: CullMeta, f: Filter) -> bool {
    todo!()
}

// Indices into `items`, filtered by culling state and sorted. This single
// ordering drives the browse grid, the filmstrip, arrow navigation, and
// prefetch neighbor selection — they must never disagree.
pub fn visible_indices(
    items: &[Item],
    meta: &HashMap<PathBuf, CullMeta>,
    filter: Filter,
    sort: Sort,
) -> Vec<usize> {
    todo!()
}
```

And inside `mod tests`:

```rust
    #[test]
    fn filter_and_sort_visible() {
        use std::time::{Duration, UNIX_EPOCH};
        let paths = [PathBuf::from("/d/b.jpg"), PathBuf::from("/d/a.jpg"), PathBuf::from("/d/c.jpg")];
        let items: Vec<Item> = vec![
            Item { path: &paths[0], mtime: Some(UNIX_EPOCH + Duration::from_secs(30)) },
            Item { path: &paths[1], mtime: Some(UNIX_EPOCH + Duration::from_secs(10)) },
            Item { path: &paths[2], mtime: Some(UNIX_EPOCH + Duration::from_secs(20)) },
        ];
        let mut meta = HashMap::new();
        meta.insert(paths[0].clone(), CullMeta { flag: Flag::Pick, rating: 3, ..Default::default() });
        meta.insert(paths[1].clone(), CullMeta { flag: Flag::Reject, ..Default::default() });

        // no filter, name sort: a, b, c
        assert_eq!(visible_indices(&items, &meta, Filter::default(), Sort::Name), vec![1, 0, 2]);
        // date sort: a(10), c(20), b(30)
        assert_eq!(visible_indices(&items, &meta, Filter::default(), Sort::Date), vec![1, 2, 0]);
        // picks only → b.jpg
        let f = Filter { flags: FlagFilter::Picks, min_rating: 0 };
        assert_eq!(visible_indices(&items, &meta, f, Sort::Name), vec![0]);
        // min rating 1 → only b.jpg (rating 3)
        let f = Filter { flags: FlagFilter::All, min_rating: 1 };
        assert_eq!(visible_indices(&items, &meta, f, Sort::Name), vec![0]);
        // unflagged → only c.jpg
        let f = Filter { flags: FlagFilter::Unflagged, min_rating: 0 };
        assert_eq!(visible_indices(&items, &meta, f, Sort::Name), vec![2]);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test filter_and_sort_visible`
Expected: FAIL with `not yet implemented`.

- [ ] **Step 3: Implement**

```rust
pub fn passes(meta: CullMeta, f: Filter) -> bool {
    let flag_ok = match f.flags {
        FlagFilter::All => true,
        FlagFilter::Picks => meta.flag == Flag::Pick,
        FlagFilter::Rejects => meta.flag == Flag::Reject,
        FlagFilter::Unflagged => meta.flag == Flag::None,
    };
    flag_ok && meta.rating >= f.min_rating
}

pub fn visible_indices(
    items: &[Item],
    meta: &HashMap<PathBuf, CullMeta>,
    filter: Filter,
    sort: Sort,
) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..items.len())
        .filter(|&i| passes(meta.get(items[i].path).copied().unwrap_or_default(), filter))
        .collect();
    match sort {
        Sort::Name => idx.sort_by(|&a, &b| items[a].path.cmp(items[b].path)),
        Sort::Date => idx.sort_by(|&a, &b| {
            items[a].mtime.cmp(&items[b].mtime).then_with(|| items[a].path.cmp(items[b].path))
        }),
    }
    idx
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/cull.rs
git commit -m "feat: filter/sort ordering (visible_indices) for culling views"
```

---

### Task 3: Wire meta into App + keyboard shortcuts

**Files:**
- Modify: `src/main.rs` — App struct (~line 81), `App::new` (~line 110), `scan_folder` (~line 232), `register_image` (~line 264), egui closure (~line 348-370), post-frame section (~line 1072)

- [ ] **Step 1: Add App state**

In the `App` struct (after `db: Option<rusqlite::Connection>,` line 107):

```rust
    // Culling state: in-memory mirror of the SQLite meta table (write-through)
    meta: std::collections::HashMap<PathBuf, cull::CullMeta>,
    // Browse-grid selection (decoupled from current_path, which is the image
    // loaded in the editor — edits must never save under a merely-selected path)
    selected: Option<PathBuf>,
    filter: cull::Filter,
    sort: cull::Sort,
```

In `App::new()` (after `db: open_db(),`):

```rust
            meta: std::collections::HashMap::new(),
            selected: None,
            filter: cull::Filter::default(),
            sort: cull::Sort::default(),
```

- [ ] **Step 2: Load meta on folder scan, set selection on open**

In `scan_folder` (after `self.browse_dir = Some(dir.to_owned());`):

```rust
        self.meta = self.db.as_ref().map(cull::load_all_meta).unwrap_or_default();
```

In `register_image` (next to `self.current_path = Some(path.to_owned());`):

```rust
            self.selected = Some(path.to_owned());
```

- [ ] **Step 3: Collect cull actions in the frame, apply after it**

In the frame-scope `let` block (~line 348), add to the local bindings:

```rust
            let selected   = &mut self.selected;
            let meta_map   = &self.meta;
            let mut cull_actions: Vec<(PathBuf, cull::CullAction)> = Vec::new();
```

Inside the `egui_ctx.run` closure, right after the tabs `TopBottomPanel` block (before `if *view == View::Browse`), add the keyboard handler — it runs in BOTH views:

```rust
                // ---- Culling shortcuts: 1-5 stars, 0 clear, P pick, X reject, 6-9 labels ----
                if !ctx.wants_keyboard_input() {
                    let target: Option<PathBuf> = if *view == View::Edit {
                        current_path.clone()
                    } else {
                        selected.clone().or_else(|| current_path.clone())
                    };
                    if let Some(t) = target {
                        use cull::{CullAction as A, Label as L};
                        let keymap = [
                            (egui::Key::Num0, A::Rating(0)),
                            (egui::Key::Num1, A::Rating(1)),
                            (egui::Key::Num2, A::Rating(2)),
                            (egui::Key::Num3, A::Rating(3)),
                            (egui::Key::Num4, A::Rating(4)),
                            (egui::Key::Num5, A::Rating(5)),
                            (egui::Key::P, A::TogglePick),
                            (egui::Key::X, A::ToggleReject),
                            (egui::Key::Num6, A::ToggleLabel(L::Red)),
                            (egui::Key::Num7, A::ToggleLabel(L::Yellow)),
                            (egui::Key::Num8, A::ToggleLabel(L::Green)),
                            (egui::Key::Num9, A::ToggleLabel(L::Blue)),
                        ];
                        ctx.input(|i| {
                            for (k, a) in keymap {
                                if i.key_pressed(k) {
                                    cull_actions.push((t.clone(), a));
                                }
                            }
                        });
                    }
                }
```

Extend the tuple returned from the frame scope: change the tuple binding (~line 348) and the return expression (~line 1068) to also carry `cull_actions`:

```rust
        let (shapes, textures_delta, pixels_per_point, open_path, export_path, scan_dir, auto_req, mut needs_process, cull_actions) = {
```

```rust
            (full_output.shapes, full_output.textures_delta, full_output.pixels_per_point,
             open_path, export_path, scan_dir, auto_req.is_some(), needs_process, cull_actions)
```

After the frame scope, next to the `// File ops` section, apply and persist:

```rust
        // Culling actions: update the in-memory map, write through to SQLite.
        // Work on a copy to avoid holding a &mut into the map while mutating it.
        for (path, action) in cull_actions {
            let mut m = self.meta.get(&path).copied().unwrap_or_default();
            cull::apply_action(&mut m, action);
            if m.is_default() {
                self.meta.remove(&path);
            } else {
                self.meta.insert(path.clone(), m);
            }
            if let Some(db) = &self.db {
                cull::save_meta(db, &path, m); // save_meta deletes the row for default state
            }
        }
```

- [ ] **Step 4: Build and verify manually**

Run: `cargo build` — expect clean compile (warnings about unused `filter`/`sort`/`meta_map` are fine until Tasks 4-5).
Run: `cargo run --release -- ~/Pictures` (any photo folder). Open an image, press `3`, `P`, `7`. Then:

```bash
sqlite3 ~/.image-processor/edits.db "SELECT path, rating, flag, label FROM meta;"
```

Expected: one row with rating 3, flag 1, label 2. Press `0`, `P`, `7` again in the app; re-query — the row should be gone (default state deletes). Also confirm typing into the levels DragValue boxes does NOT trigger shortcuts (the `wants_keyboard_input` guard).

- [ ] **Step 5: Commit**

```bash
git add src/main.rs
git commit -m "feat: culling keyboard shortcuts with write-through meta persistence"
```

---

### Task 4: Culling GUI — grid badges, edit-panel star row, browse selection

**Files:**
- Modify: `src/main.rs` — browse grid cell painting (~line 415-498), side panel top (~line 509), free functions at file bottom (before `init_gpu`)

- [ ] **Step 1: Add badge-painting helpers**

At the bottom of `src/main.rs` (before `async fn init_gpu`):

```rust
fn label_color(l: cull::Label) -> Option<egui::Color32> {
    match l {
        cull::Label::None => None,
        cull::Label::Red => Some(egui::Color32::from_rgb(220, 70, 60)),
        cull::Label::Yellow => Some(egui::Color32::from_rgb(230, 200, 60)),
        cull::Label::Green => Some(egui::Color32::from_rgb(80, 190, 90)),
        cull::Label::Blue => Some(egui::Color32::from_rgb(80, 140, 240)),
    }
}

// Rating stars (bottom-left), pick/reject badge (top-left, reject also dims),
// label dot (top-right). Shared by the browse grid and the filmstrip.
fn paint_badges(p: &egui::Painter, rect: egui::Rect, meta: cull::CullMeta) {
    use egui::{Align2, Color32, FontId, pos2};
    match meta.flag {
        cull::Flag::Pick => {
            p.circle_filled(pos2(rect.left() + 12.0, rect.top() + 12.0), 7.0, Color32::from_rgb(60, 160, 70));
            p.text(pos2(rect.left() + 12.0, rect.top() + 12.0), Align2::CENTER_CENTER, "✓",
                   FontId::proportional(10.0), Color32::WHITE);
        }
        cull::Flag::Reject => {
            p.rect_filled(rect, 4.0, Color32::from_black_alpha(120));
            p.circle_filled(pos2(rect.left() + 12.0, rect.top() + 12.0), 7.0, Color32::from_rgb(200, 60, 50));
            p.text(pos2(rect.left() + 12.0, rect.top() + 12.0), Align2::CENTER_CENTER, "✕",
                   FontId::proportional(10.0), Color32::WHITE);
        }
        cull::Flag::None => {}
    }
    if meta.rating > 0 {
        p.text(pos2(rect.left() + 6.0, rect.bottom() - 4.0), Align2::LEFT_BOTTOM,
               "★".repeat(meta.rating as usize),
               FontId::proportional(11.0), Color32::from_rgb(255, 200, 70));
    }
    if let Some(c) = label_color(meta.label) {
        p.circle_filled(pos2(rect.right() - 11.0, rect.top() + 11.0), 5.5, c);
    }
}
```

- [ ] **Step 2: Paint badges + selection in the browse grid**

In the grid cell loop (~line 415), after the image (or "…") is painted and before the filename label, add:

```rust
                                        paint_badges(
                                            &p,
                                            img_area,
                                            meta_map.get(&entry.path).copied().unwrap_or_default(),
                                        );
```

Change the selection highlight (~line 474) to use grid selection instead of `current_path`:

```rust
                                        let is_sel =
                                            selected.as_deref() == Some(entry.path.as_path());
                                        if is_sel {
```

(keep the blue stroke / hover stroke bodies as they are, just rename the variable).

In the click handler (~line 490), also set the selection:

```rust
                                        if resp.clicked() {
                                            *selected = Some(entry.path.clone());
                                            ctx.data_mut(|d| {
                                                d.insert_temp(
                                                    egui::Id::new(KEY_OPEN_PATH),
                                                    entry.path.clone(),
                                                )
                                            });
                                        }
```

- [ ] **Step 3: Star/flag/label row at the top of the Edit side panel**

At the very top of the `SidePanel::right("controls")` closure (line ~510, before the histogram allocation):

```rust
                        if let Some(cp) = current_path {
                            let m = meta_map.get(cp).copied().unwrap_or_default();
                            ui.add_space(4.0);
                            ui.horizontal(|ui| {
                                for i in 1..=5u8 {
                                    let (star, col) = if m.rating >= i {
                                        ("★", egui::Color32::from_rgb(255, 200, 70))
                                    } else {
                                        ("☆", egui::Color32::from_gray(110))
                                    };
                                    let b = egui::Button::new(
                                        egui::RichText::new(star).size(15.0).color(col),
                                    )
                                    .frame(false);
                                    if ui.add(b).clicked() {
                                        cull_actions.push((cp.clone(), cull::CullAction::Rating(i)));
                                    }
                                }
                                ui.add_space(6.0);
                                if ui.selectable_label(m.flag == cull::Flag::Pick, "P")
                                    .on_hover_text("Pick (P)").clicked()
                                {
                                    cull_actions.push((cp.clone(), cull::CullAction::TogglePick));
                                }
                                if ui.selectable_label(m.flag == cull::Flag::Reject, "X")
                                    .on_hover_text("Reject (X)").clicked()
                                {
                                    cull_actions.push((cp.clone(), cull::CullAction::ToggleReject));
                                }
                                ui.add_space(6.0);
                                for l in [cull::Label::Red, cull::Label::Yellow, cull::Label::Green, cull::Label::Blue] {
                                    let c = label_color(l).unwrap();
                                    let dot = egui::RichText::new("●").size(13.0).color(c);
                                    if ui.selectable_label(m.label == l, dot).clicked() {
                                        cull_actions.push((cp.clone(), cull::CullAction::ToggleLabel(l)));
                                    }
                                }
                            });
                            ui.add_space(2.0);
                        }
```

- [ ] **Step 4: Build and verify manually**

Run: `cargo run --release -- ~/Pictures`
Checklist:
- Browse grid: rated/flagged images show stars, ✓/✕ badge, label dot; rejected thumbs are dimmed.
- Pressing keys in Browse updates the badge of the *selected* (blue-bordered) thumb instantly.
- Edit view: star row reflects current image; clicking stars/P/X/dots works and survives switching images (persistence from Task 3).
- "Reset All Edits" does NOT clear rating/flag/label.

- [ ] **Step 5: Commit**

```bash
git add src/main.rs
git commit -m "feat: culling badges in grid, star/flag/label row in editor"
```

---

### Task 5: Filter & sort toolbar in Browse

**Files:**
- Modify: `src/main.rs` — `ThumbEntry` (~line 69), `scan_folder` (~line 243), frame scope (~line 348), browse header row (~line 379), grid loop (~line 415)

- [ ] **Step 1: Add mtime to ThumbEntry**

```rust
struct ThumbEntry {
    path: PathBuf,
    tex: Option<egui::TextureHandle>,
    mtime: Option<std::time::SystemTime>,
}
```

In `scan_folder`:

```rust
        self.thumbs = paths
            .iter()
            .map(|p| ThumbEntry {
                path: p.clone(),
                tex: None,
                mtime: std::fs::metadata(p).and_then(|m| m.modified()).ok(),
            })
            .collect();
```

- [ ] **Step 2: Compute the shared visible ordering each frame**

In `render()`, immediately BEFORE the frame-scope `let` block (~line 347):

```rust
        // One ordering for grid, filmstrip, arrow nav and prefetch
        let visible: Vec<usize> = {
            let items: Vec<cull::Item> = self.thumbs.iter()
                .map(|t| cull::Item { path: &t.path, mtime: t.mtime })
                .collect();
            cull::visible_indices(&items, &self.meta, self.filter, self.sort)
        };
        let n_picks = self.thumbs.iter()
            .filter(|t| self.meta.get(&t.path).map_or(false, |m| m.flag == cull::Flag::Pick))
            .count();
        let n_rejects = self.thumbs.iter()
            .filter(|t| self.meta.get(&t.path).map_or(false, |m| m.flag == cull::Flag::Reject))
            .count();
```

Add `filter`/`sort` to the frame-scope bindings:

```rust
            let filter = &mut self.filter;
            let sort   = &mut self.sort;
```

(`visible`, `n_picks`, `n_rejects` are read-only locals; the closure captures them by reference automatically.)

- [ ] **Step 3: Toolbar row + filtered grid**

In the Browse header `ui.horizontal` (after the folder-path label, ~line 394), append:

```rust
                                ui.separator();
                                for (f, lbl) in [
                                    (cull::FlagFilter::All, "All"),
                                    (cull::FlagFilter::Picks, "Picks"),
                                    (cull::FlagFilter::Rejects, "Rejects"),
                                    (cull::FlagFilter::Unflagged, "Unflagged"),
                                ] {
                                    if ui.selectable_label(filter.flags == f, lbl).clicked() {
                                        filter.flags = f;
                                    }
                                }
                                ui.separator();
                                ui.label(egui::RichText::new("★ ≥").small());
                                ui.add(egui::DragValue::new(&mut filter.min_rating).range(0..=5));
                                ui.separator();
                                ui.label(egui::RichText::new("Sort").small());
                                for (s, lbl) in [(cull::Sort::Name, "Name"), (cull::Sort::Date, "Date")] {
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
```

Change the grid loop (~line 415) to iterate the filtered ordering:

```rust
                                    for &ti in &visible {
                                        let entry = &thumbs[ti];
```

(the rest of the cell body is unchanged — it already uses `entry`). Also update the empty-state: show "No images match the filter" when `thumbs` is non-empty but `visible` is empty:

```rust
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
```

- [ ] **Step 4: Build and verify manually**

Run: `cargo run --release -- ~/Pictures`
Checklist:
- Flag a few picks/rejects; the Picks/Rejects/Unflagged filters narrow the grid; counts row is correct.
- ★ ≥ filter hides lower-rated images; `★ ≥ 0` shows all.
- Sort by Date reorders by file modification time; Name restores alphabetical.

- [ ] **Step 5: Commit**

```bash
git add src/main.rs
git commit -m "feat: browse filter (flags, min rating) and sort (name/date) toolbar"
```

---

### Task 6: Filmstrip + Left/Right arrow navigation

**Files:**
- Modify: `src/main.rs` — App struct/new (scroll flags), keyboard block from Task 3, Edit branch of the closure (filmstrip panel goes between the `SidePanel` and the `CentralPanel`, ~line 938), `register_image`

- [ ] **Step 1: Add scroll-sync flags**

App struct:

```rust
    // One-shot "scroll the strip/grid to the current item" flags, set on
    // keyboard navigation / image load, consumed by the next painted frame
    strip_scroll: bool,
    grid_scroll: bool,
```

`App::new()`: `strip_scroll: false, grid_scroll: false,`
`register_image` (next to `self.view = View::Edit;`): `self.strip_scroll = true;`
Frame-scope bindings: `let strip_scroll = &mut self.strip_scroll;` and `let grid_scroll = &mut self.grid_scroll;`

- [ ] **Step 2: Arrow navigation (both views) in the keyboard block**

Append inside the `if !ctx.wants_keyboard_input() {` block from Task 3:

```rust
                    // ---- Arrow navigation through the visible ordering ----
                    if !visible.is_empty() {
                        let (left, right, enter) = ctx.input(|i| {
                            (i.key_pressed(egui::Key::ArrowLeft),
                             i.key_pressed(egui::Key::ArrowRight),
                             i.key_pressed(egui::Key::Enter))
                        });
                        let delta = right as i32 - left as i32;
                        if delta != 0 {
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
                            if let Some(p) = selected.clone() {
                                ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_OPEN_PATH), p));
                            }
                        }
                    }
```

- [ ] **Step 3: Grid scroll-to-selection**

In the grid cell loop, right after `allocate_exact_size` (BEFORE the `is_rect_visible` continue):

```rust
                                        if *grid_scroll
                                            && selected.as_deref() == Some(entry.path.as_path())
                                        {
                                            ui.scroll_to_rect(rect, Some(egui::Align::Center));
                                            *grid_scroll = false;
                                        }
```

- [ ] **Step 4: Filmstrip panel in Edit view**

In the Edit branch, AFTER the `SidePanel::right("controls").show(...)` call and BEFORE `egui::CentralPanel::default()` (~line 938):

```rust
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
                                    let is_cur =
                                        current_path.as_deref() == Some(entry.path.as_path());
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
                                            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                                            egui::Color32::WHITE,
                                        );
                                    } else {
                                        p.text(img_area.center(), egui::Align2::CENTER_CENTER, "…",
                                               egui::FontId::proportional(14.0), egui::Color32::from_gray(90));
                                    }
                                    paint_badges(
                                        &p,
                                        img_area,
                                        meta_map.get(&entry.path).copied().unwrap_or_default(),
                                    );
                                    if is_cur {
                                        p.rect_stroke(rect.shrink(2.0), 3.0,
                                            egui::Stroke::new(2.0, egui::Color32::from_rgb(90, 140, 255)));
                                    } else if resp.hovered() {
                                        p.rect_stroke(rect.shrink(2.0), 3.0,
                                            egui::Stroke::new(1.0, egui::Color32::from_gray(110)));
                                    }
                                    if resp.clicked() {
                                        ctx.data_mut(|d| {
                                            d.insert_temp(egui::Id::new(KEY_OPEN_PATH), entry.path.clone())
                                        });
                                    }
                                }
                            });
                        });
                    });
```

- [ ] **Step 5: Build and verify manually**

Run: `cargo run --release -- ~/Pictures/some-photo.jpg`
Checklist:
- Edit view shows a filmstrip of the folder; current image has a blue border and is scrolled into view.
- Left/Right arrows move to prev/next image; edits of each image restore (existing persistence); zoom resets to fit; the strip follows.
- Filter set in Browse (e.g. Picks) restricts what arrows/filmstrip traverse in Edit.
- In Browse: arrows move the blue selection (grid scrolls along), Enter opens the selected image.
- Curve/levels dragging is unaffected (arrow keys don't fire while a text field is focused).

- [ ] **Step 6: Commit**

```bash
git add src/main.rs
git commit -m "feat: filmstrip in editor and arrow-key navigation in both views"
```

---

### Task 7: Processor refactor — `upload_rgba` + texture rebinding helper

**Files:**
- Modify: `src/processor.rs:460-512` (`load_image`), `src/main.rs` (`register_image`, new helper)

- [ ] **Step 1: Split decode from upload in processor.rs**

Replace `load_image` with:

```rust
    pub fn load_image(&mut self, path: &Path, device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
        let Some(img) = crate::imgload::load_rgba(path, 0) else {
            return false;
        };
        self.upload_rgba(&img, device, queue);
        true
    }

    // Upload an already-decoded image and (re)create the pipeline textures.
    pub fn upload_rgba(&mut self, img: &image::RgbaImage, device: &wgpu::Device, queue: &wgpu::Queue) {
        let (width, height) = img.dimensions();
        // ... existing body of load_image from `let input_tex = ...`
        //     through `self.process(device, queue);`, unchanged ...
    }
```

(i.e. everything from `let input_tex = device.create_texture(...)` down to `self.process(device, queue);` moves verbatim into `upload_rgba`; `load_image` keeps only decode + delegation and the `true` return.)

- [ ] **Step 2: Extract texture rebinding in main.rs**

Add to `impl App` (near `register_image`):

```rust
    // Free and re-register the egui textures that point at the processor's
    // input/output — required whenever the underlying wgpu textures are recreated.
    fn rebind_image_textures(&mut self) {
        let (Some(gpu), Some(proc), Some(er)) =
            (self.gpu.as_ref(), self.processor.as_ref(), self.egui_renderer.as_mut())
        else {
            return;
        };
        if let Some(id) = self.image_tex_id.take() { er.free_texture(&id); }
        if let Some(id) = self.original_tex_id.take() { er.free_texture(&id); }
        let output_view = proc.output_view().unwrap();
        let input_view = proc.input_view().unwrap();
        self.image_tex_id = Some(er.register_native_texture(
            &gpu.device, &output_view, wgpu::FilterMode::Linear,
        ));
        self.original_tex_id = Some(er.register_native_texture(
            &gpu.device, &input_view, wgpu::FilterMode::Linear,
        ));
    }
```

Rewrite `register_image` to use it (same behavior as today, structured for Task 8/9):

```rust
    fn register_image(&mut self, path: &std::path::Path) {
        if self.gpu.is_none() || self.processor.is_none() || self.egui_renderer.is_none() {
            return;
        }

        let state = self
            .db
            .as_ref()
            .and_then(|db| load_edits(db, path))
            .unwrap_or_default();

        let loaded = {
            let gpu = self.gpu.as_ref().unwrap();
            let proc = self.processor.as_mut().unwrap();
            proc.apply_edit_state(&state);
            proc.load_image(path, &gpu.device, &gpu.queue)
        };
        if !loaded {
            return;
        }
        self.rebind_image_textures();

        if let Some(window) = &self.window {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("image-processor");
            window.set_title(name);
        }

        self.output_dirty = true;
        self.zoom_fit = true;
        self.view = View::Edit;
        self.strip_scroll = true;
        self.current_path = Some(path.to_owned());
        self.selected = Some(path.to_owned());

        if let Some(parent) = path.parent() {
            if self.browse_dir.as_deref() != Some(parent) {
                self.scan_folder(parent);
            }
        }
    }
```

- [ ] **Step 3: Build and verify**

Run: `cargo build` — clean.
Run: `cargo run --release -- ~/Pictures` — open a few images (JPEG and RAW), confirm edit view, original-preview (hold mouse), export, and arrow navigation all still work exactly as before.

- [ ] **Step 4: Commit**

```bash
git add src/processor.rs src/main.rs
git commit -m "refactor: split image decode from GPU upload; extract texture rebinding"
```

---

### Task 8: Fast RAW open (embedded JPEG) + background full-demosaic swap

**Files:**
- Modify: `src/imgload.rs` (new `load_edit_rgba`), `src/main.rs` (App fields, `register_image`, swap pump in `render()`)

- [ ] **Step 1: Add the fast full-resolution decode to imgload.rs**

```rust
// Full-resolution decode tuned for the culling loop: RAW files open via the
// embedded JPEG (no demosaic — typically 10-50x faster). Returns
// (image, preview_quality); preview_quality=true means the caller should
// schedule a real demosaic and swap it in when ready.
pub fn load_edit_rgba(path: &Path) -> Option<(image::RgbaImage, bool)> {
    if is_raw(path) {
        if let Some(img) = jpgfromraw_preview(path, 0) {
            return Some((img, true));
        }
    }
    load_rgba(path, 0).map(|img| (img, false))
}
```

- [ ] **Step 2: App plumbing for the demosaic swap**

App struct:

```rust
    // Full-quality RAW demosaic results stream in here and replace the
    // preview-quality embedded JPEG shown for instant opens
    full_tx: std::sync::mpsc::Sender<(PathBuf, image::RgbaImage)>,
    full_rx: std::sync::mpsc::Receiver<(PathBuf, image::RgbaImage)>,
    // Demosaic currently running (at most one at a time)
    full_pending: Option<PathBuf>,
    // Debounce: demosaic only starts after the image has stayed current ~0.6 s,
    // so arrowing through a RAW folder never stacks up decode threads
    wants_full: Option<(PathBuf, std::time::Instant)>,
```

`App::new()`:

```rust
        let (full_tx, full_rx) = std::sync::mpsc::channel();
        Self {
            // ... existing fields ...
            full_tx,
            full_rx,
            full_pending: None,
            wants_full: None,
        }
```

(adjust `new()` to build the channel before the struct literal.)

- [ ] **Step 3: register_image uses the fast path**

In `register_image` from Task 7, replace the `let loaded = { ... }` block with:

```rust
        let Some((img, preview_quality)) = imgload::load_edit_rgba(path) else {
            return;
        };
        {
            let gpu = self.gpu.as_ref().unwrap();
            let proc = self.processor.as_mut().unwrap();
            proc.apply_edit_state(&state);
            proc.upload_rgba(&img, &gpu.device, &gpu.queue);
        }
        self.rebind_image_textures();

        self.wants_full = preview_quality
            .then(|| (path.to_owned(), std::time::Instant::now()));
```

(`Processor::load_image` is now unused — delete it from processor.rs.)

- [ ] **Step 4: Debounced demosaic spawn + swap pump in render()**

In `render()`, next to the thumbnail pump (~line 318):

```rust
        // Start the full demosaic once the RAW has stayed current for a moment.
        // Clone out of wants_full so we can reassign it inside the branches.
        if let Some((p, t0)) = self.wants_full.clone() {
            if self.current_path.as_deref() != Some(p.as_path()) {
                self.wants_full = None; // navigated away before the debounce
            } else if self.full_pending.is_none() && t0.elapsed().as_secs_f32() > 0.6 {
                self.full_pending = Some(p.clone());
                self.wants_full = None;
                let tx = self.full_tx.clone();
                std::thread::spawn(move || {
                    if let Some(img) = imgload::load_rgba(&p, 0) {
                        let _ = tx.send((p, img));
                    }
                });
            }
        }

        // Swap in finished demosaics (ignore results for images we've left).
        // Drain into a Vec first: the loop body calls &mut self methods, which
        // would conflict with a receiver borrow held across a while-let body.
        let full_results: Vec<(PathBuf, image::RgbaImage)> =
            std::iter::from_fn(|| self.full_rx.try_recv().ok()).collect();
        for (p, img) in full_results {
            if self.full_pending.as_deref() == Some(p.as_path()) {
                self.full_pending = None;
            }
            if self.current_path.as_deref() == Some(p.as_path()) {
                {
                    let gpu = self.gpu.as_ref().unwrap();
                    let proc = self.processor.as_mut().unwrap();
                    proc.upload_rgba(&img, &gpu.device, &gpu.queue);
                }
                self.rebind_image_textures();
                self.output_dirty = true;
            }
        }
```

Placement: right after the existing component guard at the top of `render()` (the guard already ensures gpu/processor exist before the unwraps), next to the thumbnail pump.

- [ ] **Step 5: Build and verify manually**

Run: `cargo run --release -- /path/to/raw-folder`
Checklist:
- Opening a RAW shows the image near-instantly (embedded JPEG) instead of the previous multi-second demosaic wait.
- ~1 s after settling on a RAW, the image refines (subtle resolution/color shift) — that's the demosaic swap; sliders keep working before, during, after.
- Arrowing rapidly through 10 RAWs: CPU does NOT spike with parallel demosaics (debounce holds); only the image you stop on refines.
- JPEG behavior unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/imgload.rs src/main.rs src/processor.rs
git commit -m "feat: instant RAW opens via embedded JPEG with background demosaic swap"
```

---

### Task 9: Neighbor prefetching (`src/prefetch.rs`)

**Files:**
- Create: `src/prefetch.rs`
- Modify: `src/main.rs` (mod decl, App field, `register_image`, pump in `render()`, neighbor request helper)

- [ ] **Step 1: Create `src/prefetch.rs` with failing tests**

```rust
// Background decode of the culling loop's neighbor images so Left/Right
// navigation is instant. The cache is bounded by retain(): it only ever
// holds the currently wanted neighbor set (±1 → ≤2 full-size images).
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};

pub struct Prefetched {
    pub img: image::RgbaImage,
    pub preview_quality: bool, // RAW embedded JPEG — caller still wants a demosaic
}

pub struct Prefetcher {
    req_tx: Sender<Vec<PathBuf>>,
    res_rx: Receiver<(PathBuf, Prefetched)>,
    cache: HashMap<PathBuf, Prefetched>,
    in_flight: Vec<PathBuf>,
}

impl Prefetcher {
    pub fn new() -> Self {
        todo!()
    }

    // Ask the worker for any of `want` not already cached or being decoded.
    pub fn request(&mut self, want: Vec<PathBuf>) {
        todo!()
    }

    // Move finished decodes into the cache.
    pub fn pump(&mut self) {
        todo!()
    }

    pub fn take(&mut self, path: &Path) -> Option<Prefetched> {
        todo!()
    }

    // Drop cached images outside the wanted set (bounds memory).
    pub fn retain(&mut self, keep: &[PathBuf]) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retain_drops_unwanted() {
        let mut pf = Prefetcher::new();
        pf.cache.insert(
            PathBuf::from("/a"),
            Prefetched { img: image::RgbaImage::new(1, 1), preview_quality: false },
        );
        pf.cache.insert(
            PathBuf::from("/b"),
            Prefetched { img: image::RgbaImage::new(1, 1), preview_quality: false },
        );
        pf.retain(&[PathBuf::from("/a")]);
        assert!(pf.cache.contains_key(Path::new("/a")));
        assert_eq!(pf.cache.len(), 1);
    }

    #[test]
    fn prefetch_decodes_and_caches() {
        let p = std::env::temp_dir().join("prefetch_test_8x8.png");
        image::RgbaImage::new(8, 8).save(&p).unwrap();

        let mut pf = Prefetcher::new();
        pf.request(vec![p.clone()]);

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let mut got = None;
        while got.is_none() && std::time::Instant::now() < deadline {
            got = pf.take(&p);
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let got = got.expect("worker should decode and deliver the image");
        assert_eq!(got.img.dimensions(), (8, 8));
        assert!(!got.preview_quality);
        assert!(pf.take(&p).is_none()); // take() consumes
        let _ = std::fs::remove_file(&p);
    }
}
```

Add to `src/main.rs` line 2: `mod prefetch;`

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test prefetch`
Expected: FAIL with `not yet implemented`.

- [ ] **Step 3: Implement**

```rust
impl Prefetcher {
    pub fn new() -> Self {
        let (req_tx, req_rx) = channel::<Vec<PathBuf>>();
        let (res_tx, res_rx) = channel();
        std::thread::spawn(move || {
            while let Ok(want) = req_rx.recv() {
                for p in want {
                    let Some((img, preview_quality)) = crate::imgload::load_edit_rgba(&p) else {
                        continue;
                    };
                    if res_tx.send((p, Prefetched { img, preview_quality })).is_err() {
                        return; // app dropped the receiver — shut down
                    }
                }
            }
        });
        Self { req_tx, res_rx, cache: HashMap::new(), in_flight: Vec::new() }
    }

    pub fn request(&mut self, want: Vec<PathBuf>) {
        let missing: Vec<PathBuf> = want
            .into_iter()
            .filter(|p| !self.cache.contains_key(p) && !self.in_flight.contains(p))
            .collect();
        if missing.is_empty() {
            return;
        }
        self.in_flight.extend(missing.iter().cloned());
        let _ = self.req_tx.send(missing);
    }

    pub fn pump(&mut self) {
        while let Ok((p, res)) = self.res_rx.try_recv() {
            self.in_flight.retain(|q| q != &p);
            self.cache.insert(p, res);
        }
    }

    pub fn take(&mut self, path: &Path) -> Option<Prefetched> {
        self.pump();
        self.cache.remove(path)
    }

    pub fn retain(&mut self, keep: &[PathBuf]) {
        self.cache.retain(|p, _| keep.iter().any(|k| k == p));
    }
}
```

A failed decode leaves a stale `in_flight` entry — that only suppresses one re-request for an undecodable file, never a decodable one, so no cleanup pass is needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test`
Expected: `7 passed` (5 cull + 2 prefetch).

- [ ] **Step 5: Integrate into App**

App struct: `prefetch: prefetch::Prefetcher,` — `App::new()`: `prefetch: prefetch::Prefetcher::new(),`

In `register_image` (Task 8 version), change the decode line to consult the cache first:

```rust
        let fetched = self
            .prefetch
            .take(path)
            .map(|p| (p.img, p.preview_quality))
            .or_else(|| imgload::load_edit_rgba(path));
        let Some((img, preview_quality)) = fetched else {
            return;
        };
```

At the END of `register_image`, request neighbors:

```rust
        self.request_neighbor_prefetch();
```

Add the helper to `impl App`:

```rust
    // Queue prev/next (in the current visible ordering) for background decode
    // and drop cached images that are no longer neighbors.
    fn request_neighbor_prefetch(&mut self) {
        let Some(cur) = &self.current_path else { return };
        let items: Vec<cull::Item> = self.thumbs.iter()
            .map(|t| cull::Item { path: &t.path, mtime: t.mtime })
            .collect();
        let order = cull::visible_indices(&items, &self.meta, self.filter, self.sort);
        let Some(pos) = order.iter().position(|&i| self.thumbs[i].path == *cur) else { return };
        let mut want = Vec::new();
        if pos + 1 < order.len() {
            want.push(self.thumbs[order[pos + 1]].path.clone());
        }
        if pos > 0 {
            want.push(self.thumbs[order[pos - 1]].path.clone());
        }
        self.prefetch.retain(&want);
        self.prefetch.request(want);
    }
```

In `render()` next to the thumbnail pump: `self.prefetch.pump();`

- [ ] **Step 6: Build and verify manually**

Run: `cargo run --release -- /path/to/raw-folder`
Checklist:
- Open an image, wait a beat, press Right: the next image appears essentially instantly (cache hit — only a GPU upload).
- Arrow back and forth between two images repeatedly: instant both ways.
- Memory stays bounded while walking a large folder (cache holds ≤2 images; check Activity Monitor roughly stabilizes).

- [ ] **Step 7: Commit**

```bash
git add src/prefetch.rs src/main.rs
git commit -m "feat: background prefetch of neighbor images for instant navigation"
```

---

### Task 10: Cull actions — trash/move rejects, copy picks

**Files:**
- Modify: `Cargo.toml` (add `trash = "5"`), `src/main.rs` (KEY consts, App field, toolbar buttons, confirm window, post-frame handlers, helpers)

- [ ] **Step 1: Write the failing test for the move helper**

Add to `src/main.rs` bottom:

```rust
// Move a file, falling back to copy+delete for cross-device destinations.
fn move_file(from: &std::path::Path, to: &std::path::Path) -> std::io::Result<()> {
    todo!()
}

#[cfg(test)]
mod tests {
    #[test]
    fn move_file_moves() {
        let dir = std::env::temp_dir().join("ip_move_test");
        std::fs::create_dir_all(&dir).unwrap();
        let from = dir.join("a.txt");
        let to = dir.join("b.txt");
        std::fs::write(&from, b"hi").unwrap();
        super::move_file(&from, &to).unwrap();
        assert!(!from.exists());
        assert_eq!(std::fs::read(&to).unwrap(), b"hi");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test move_file_moves`
Expected: FAIL with `not yet implemented`.

- [ ] **Step 3: Implement the helper, run test**

```rust
fn move_file(from: &std::path::Path, to: &std::path::Path) -> std::io::Result<()> {
    match std::fs::rename(from, to) {
        Ok(()) => Ok(()),
        Err(_) => {
            std::fs::copy(from, to)?;
            std::fs::remove_file(from)
        }
    }
}
```

Run: `cargo test move_file_moves` — Expected: PASS.

- [ ] **Step 4: Dependency + state + toolbar UI**

`Cargo.toml` dependencies: add

```toml
trash = "5"
```

KEY consts (next to the existing ones, line ~16):

```rust
const KEY_TRASH_REJECTS: &str = "trash_rejects";
const KEY_MOVE_REJECTS: &str = "move_rejects";
const KEY_COPY_PICKS: &str = "copy_picks";
```

App struct: `confirm_trash: bool,` — `App::new()`: `confirm_trash: false,` — frame-scope binding: `let confirm_trash = &mut self.confirm_trash;`

In the Browse header row (after the counts label from Task 5):

```rust
                                ui.separator();
                                if ui
                                    .add_enabled(n_rejects > 0, egui::Button::new(format!("Trash Rejects ({n_rejects})")))
                                    .clicked()
                                {
                                    *confirm_trash = true;
                                }
                                if ui
                                    .add_enabled(n_rejects > 0, egui::Button::new("Move Rejects…"))
                                    .clicked()
                                {
                                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                                        ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_MOVE_REJECTS), dir));
                                    }
                                }
                                if ui
                                    .add_enabled(n_picks > 0, egui::Button::new(format!("Copy Picks ({n_picks})…")))
                                    .clicked()
                                {
                                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                                        ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_COPY_PICKS), dir));
                                    }
                                }
```

Confirm dialog — inside the closure, before the Browse/Edit branch:

```rust
                if *confirm_trash {
                    egui::Window::new("Trash rejected photos?")
                        .collapsible(false)
                        .resizable(false)
                        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                        .show(ctx, |ui| {
                            ui.label(format!("Move {n_rejects} rejected photo(s) to the system trash?"));
                            ui.add_space(8.0);
                            ui.horizontal(|ui| {
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
```

- [ ] **Step 5: Post-frame handlers**

Read the temp keys with the others (~line 1060):

```rust
            let trash_req: Option<bool> =
                self.egui_ctx.data_mut(|d| d.remove_temp(egui::Id::new(KEY_TRASH_REJECTS)));
            let move_rejects_dir: Option<PathBuf> =
                self.egui_ctx.data_mut(|d| d.remove_temp(egui::Id::new(KEY_MOVE_REJECTS)));
            let copy_picks_dir: Option<PathBuf> =
                self.egui_ctx.data_mut(|d| d.remove_temp(egui::Id::new(KEY_COPY_PICKS)));
```

…and add them to the frame-scope tuple (now: `…, cull_actions, trash_req.is_some(), move_rejects_dir, copy_picks_dir)`).

Helpers on `impl App`:

```rust
    fn flagged_paths(&self, flag: cull::Flag) -> Vec<PathBuf> {
        self.thumbs
            .iter()
            .filter(|t| self.meta.get(&t.path).map_or(false, |m| m.flag == flag))
            .map(|t| t.path.clone())
            .collect()
    }
```

Handlers in `render()` after the cull-actions application:

```rust
        if trash_req {
            let rejects = self.flagged_paths(cull::Flag::Reject);
            if trash::delete_all(&rejects).is_ok() {
                if let Some(db) = &self.db {
                    for p in &rejects {
                        cull::delete_rows(db, p);
                    }
                }
                self.after_files_removed(&rejects);
            }
        }
        if let Some(dest) = move_rejects_dir {
            let rejects = self.flagged_paths(cull::Flag::Reject);
            let mut moved = Vec::new();
            for p in &rejects {
                let Some(name) = p.file_name() else { continue };
                let np = dest.join(name);
                if move_file(p, &np).is_ok() {
                    if let Some(db) = &self.db {
                        cull::rekey_rows(db, p, &np);
                    }
                    moved.push(p.clone());
                }
            }
            self.after_files_removed(&moved);
        }
        if let Some(dest) = copy_picks_dir {
            let picks = self.flagged_paths(cull::Flag::Pick);
            for p in &picks {
                let Some(name) = p.file_name() else { continue };
                let np = dest.join(name);
                if std::fs::copy(p, &np).is_ok() {
                    if let Some(db) = &self.db {
                        cull::copy_rows(db, p, &np);
                    }
                }
            }
        }
```

And the cleanup helper on `impl App`:

```rust
    // After rejects were trashed/moved: drop dangling editor/selection state
    // and rescan so the grid reflects the disk.
    fn after_files_removed(&mut self, removed: &[PathBuf]) {
        if removed.is_empty() {
            return;
        }
        if self.current_path.as_ref().map_or(false, |c| removed.contains(c)) {
            self.current_path = None;
            self.view = View::Browse;
        }
        if self.selected.as_ref().map_or(false, |s| removed.contains(s)) {
            self.selected = None;
        }
        if let Some(dir) = self.browse_dir.clone() {
            self.scan_folder(&dir);
        }
    }
```

- [ ] **Step 6: Build and verify manually**

Use a **scratch folder with copies of photos** (these operations move real files):

```bash
mkdir -p /tmp/cull-test && cp ~/Pictures/*.jpg /tmp/cull-test/ 2>/dev/null
cargo run --release -- /tmp/cull-test
```

Checklist:
- Mark 2 rejects, 2 picks. "Trash Rejects (2)" → confirm dialog → files land in the macOS Trash, grid rescans without them, their DB rows are gone.
- Mark a reject, "Move Rejects…" to a subfolder → file moved; open it from the new location → its edits and meta survived (rekeyed rows).
- "Copy Picks…" to a folder → files copied; opening a copy shows the same edits as the original.
- Reject the currently edited image and trash it → app returns to Browse, no stale selection.

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml Cargo.lock src/main.rs
git commit -m "feat: cull actions — trash/move rejects, copy picks (with DB row migration)"
```

---

### Task 11: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the new features**

Add to the Features list:

```markdown
- **Culling** — 1–5 star ratings, pick/reject flags (P/X), color labels (6–9),
  saved in SQLite independently of edits; badges in the grid and filmstrip
- **Filmstrip & navigation** — Left/Right arrows move through the folder in
  Edit view (filmstrip at the bottom) and Browse (grid selection, Enter opens)
- **Fast culling loop** — neighbors are pre-decoded in the background; RAW
  opens instantly from the embedded JPEG, full demosaic swaps in moments later
- **Filter & sort** — show all/picks/rejects/unflagged, minimum star rating,
  sort by name or date
- **Cull actions** — trash or move rejected photos, copy picks to a folder
  (edits and culling state follow the files)
```

Add to the shortcuts section:

```markdown
- 1–5: rate · 0: clear rating · P: pick · X: reject · 6–9: color label
- Left/Right: previous/next image · Enter (Browse): open selected
```

- [ ] **Step 2: Run the full test suite one last time**

Run: `cargo test && cargo build --release`
Expected: `8 passed`, clean release build.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document culling features and shortcuts"
```

---

## Out of Scope (deliberate, for later)

- **XMP sidecar read/write** for Lightroom interop — separate plan.
- **EXIF capture-date sort** — Date sort uses file mtime (equals capture time for camera files); a `kamadak-exif` integration can replace it later.
- **Up/Down navigation in the wrapped grid** — column count varies with window width; Left/Right covers the loop.
- **Demosaic result caching** — re-visiting a RAW re-runs the demosaic after the debounce; a small LRU could keep the last few.
- **Async file ops** — trash/move/copy run synchronously on the UI thread; fine for typical reject counts, revisit if users trash thousands at once.

## Verification checklist for the whole feature (after Task 11)

1. `cargo test` — 8 tests pass.
2. Cull a real folder end-to-end: open folder → arrow through images in Edit (instant on prefetched neighbors, RAW opens fast) → rate/flag with keys → filter to Rejects in Browse → Trash Rejects → filter to Picks → Copy Picks.
3. Restart the app: ratings/flags/labels and edits all restored; "Reset All Edits" clears sliders but not culling state.
