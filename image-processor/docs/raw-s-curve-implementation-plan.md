# RAW Development: rawloader basic development + phone S-curve

Implementation plan. Target repo: `image-processor/` (Rust, wgpu + egui editor).

## Goal

Replace the current "Develop RAW" pipeline — a nearest-pair 3D LUT database
(`models/raw_developer.json`, 4.3 MB) plus a 512×384 neural correction field
(`models/neural_isp_pixel.onnx`, trained by `tools/train_neural_isp.py`) — with a
small, deterministic pipeline:

1. **rawloader basic development**: decode the CFA plane, crop, subtract black
   level, normalize by white level, apply as-shot white balance, demosaic,
   camera→sRGB matrix → **linear** RGB.
2. **Phone S-curve**: one global monotone curve fitted from the 46 DNG/JPEG
   pairs. It maps *exposure-normalized* linear luminance to the phone's display
   tone, so every DNG lands on the same tone placement as its developed JPEG.

No neural network, no per-image model. The entire "look" is one small JSON file
(a few KB), and each image is auto-exposed to the fitted anchor (median linear
luminance) before the curve is applied.

## Locked-in decisions

- **Delete the old pipeline entirely** — `neural_isp.rs`, `raw_developer.rs`,
  both `.onnx` models, `raw_developer.json`, `tools/train_neural_isp.py`, and
  the `--train-raw-developer` / `--export-raw-pairs` CLI flags are removed.
- **S-curve = global curve + per-image exposure normalize**: the exposure anchor
  is the 50th percentile (median) of linear luminance; fit and apply both
  normalize by it.
- **Develop action only**: opening a RAW still shows the current `imagepipe`
  render; "Develop RAW" re-decodes via rawloader-linear + S-curve and swaps the
  result into the editor input (same UX as today). `imagepipe` stays in
  `imgload.rs` as the default open render.
- **Demosaic: 3×3 same-color average** (identical algorithm to imagepipe's own
  demosaic — proven in this codebase, keeps the open/developed renders visually
  consistent). *Deviation from the originally proposed Malvar 5×5: the Malvar
  coefficient tables are not reliably available; the 3×3 is risk-free and
  matches "basic" development.* Easy to swap for Malvar later if sharpness is
  reviewed and flagged.
- **Tone curve applied on luminance, hue-preserving** (consistent with the
  app's shader convention): output RGB = input linear RGB × (display / luma),
  guarded at luma ≈ 0.
- **Fit uses ≤ 512 px renders** (tone statistics are global); **the develop
  action decodes at the editor's existing 2048 cap** for interactivity.
- Output is encoded to sRGB 8-bit RGBA and baked into the editor input texture,
  exactly like today (`raw_isp_enabled` SQLite persistence unchanged).

## Current state

Phase 1 is **implemented and tested** (`cargo test raw_develop::` → 4/4 pass):

- `Cargo.toml`: `rawloader = "0.37"` added as a direct dependency (already in
  the lock file via `imagepipe`).
- `src/raw_develop.rs`:
  - `LinearImage { width, height, data: Vec<f32> }` — linear sRGB RGB.
  - `develop_linear(path, max_dim) -> Option<LinearImage>` — rawloader decode,
    crop (`crops`), black-level subtract + white normalize, as-shot WB from
    `wb_coeffs` with `neutralwb()` fallback (NaN / flat-1.0 DNGs handled),
    3×3 same-color demosaic on the cropped CFA, camera→sRGB via
    `cam_to_xyz_normalized()` (row-normalized so neutral passes through),
    box-downsample to `max_dim` when needed.
  - `LumaHistogram` / `luminance_percentile` — histogram-based percentiles.
  - `srgb_encode`, `to_rgba`.
- Unit tests: flat-CFA demosaic fidelity (two patterns), known DNGs decode to
  sane linear images, percentile monotonicity.

Remaining: phases 2–5 below.

## Phase 2 — S-curve model + fit tool (`--fit-raw-scurve`)

**File: `src/raw_develop.rs` (extend).**

```rust
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SCurve {
    pub anchor_percentile: f64,        // 0.5 (median linear luminance)
    pub points: Vec<[f32; 2]>,         // (normalized linear luma x, display luma y), monotone
}
impl SCurve {
    pub fn load() -> Option<Self>;                       // include_str! models/raw_s_curve.json
    pub fn apply(&self, x: f32) -> f32;                  // monotone piecewise-linear, clamp beyond last
}
```

Fit algorithm (`fit_s_curve(folder: &Path, output: &Path) -> Result<usize, String>`):

1. `find_pairs` (move the existing `*.RAW-02.ORIGINAL.dng` + `*.RAW-01*.jpg`
   pairing logic into this module before deleting `raw_developer.rs`).
2. Per pair:
   - `develop_linear(dng, 512)`; build linear-luma histogram.
   - decode the JPEG via `image::open` (≤ 512), build encoded-luma histogram.
   - anchors `p ∈ {0, 0.01, 0.05, 0.1, 0.2, …, 0.9, 0.95, 0.99, 1.0}`:
     `x_kp = L_p / L_0.5` (exposure-normalized), `y_kp = J_p`.
3. Pool: for each anchor `p`, `x_p = median_k(x_kp)`, `y_p = median_k(y_kp)`.
4. Enforce monotone non-decreasing `x` (drop duplicates); prepend `(0, 0)` if
   the first `x > 0`.
5. Write `models/raw_s_curve.json` (small; embedded via `include_str!`).

**File: `src/main.rs`** — replace the two old CLI flags in `main()` with:

```
--fit-raw-scurve <folder> [output]   # default output models/raw_s_curve.json
```

**Tests:** fitted curve is monotone, `apply(0)=0`, `apply` clamps beyond the
last point; JSON round-trips.

**Verification:** run against `/Users/jakubkolcar/Downloads` (46 pairs) and
commit the real `models/raw_s_curve.json`.

## Phase 3 — Rewrite the Develop RAW action

**File: `src/raw_develop.rs` (extend).**

```rust
pub fn develop_raw(path: &Path, max_dim: u32, curve: &SCurve) -> Option<RgbaImage>
```

1. `develop_linear(path, max_dim)`.
2. `m = luminance_percentile(0.5)`; guard `m.max(1e-3)` (near-black frames).
3. Per pixel: `x = luma / m`, `display = curve.apply(x)`,
   `scale = display / luma` (0 at luma ≤ 0), `rgb_out = rgb * scale`,
   clamp [0,1], `srgb_encode` → RGBA.
4. Apply rawloader `orientation` (rotate/flip) so the result matches the
   pre-oriented JPEG.

**File: `src/main.rs`** — rewire the four call sites (all currently call
`raw_developer.as_ref()?.apply(&img)` on the imagepipe render; all must develop
from `current_path` instead):

| Site | Line ~ | Change |
|---|---|---|
| Field + init | 559, 663 | `raw_developer: Option<Arc<raw_developer::RawDeveloper>>` → `s_curve: Option<Arc<raw_develop::SCurve>>`; `RawDeveloper::load()` → `SCurve::load()` |
| `mod` decls | 13 | `mod raw_developer; mod neural_isp;` → `mod raw_develop;` |
| Look-transfer worker | 869–879 | if `is_raw`: `develop_raw(path, 768, curve)` instead of load-then-apply; skip the wasted imagepipe load; keep `raw_isp_enabled = true` |
| Open-time apply | 1023–1028 | `if raw_isp_enabled && is_raw`: `develop_raw(path, 2048, curve)` → `replace_input_rgba` |
| Edit-state restore | 1144–1154 | same replacement (drop the intermediate `load_edit_rgba`) |
| Button handler | 2903–2922 | same replacement; `processor.raw_isp_enabled = true; processor.raw_development = None;` unchanged |
| Button enablement | 1339, 2241–2248 | `raw_developer_available` → `s_curve.is_some()`; update hover text |

`raw_isp_enabled` persistence in SQLite stays as-is, so re-opened RAWs redevelop
on load.

## Phase 4 — Delete the old pipeline

- Delete `src/neural_isp.rs`, `src/raw_developer.rs` (after moving `find_pairs`
  and the profile-distance test pattern into `raw_develop.rs`).
- Delete `models/neural_isp_base.onnx`, `models/neural_isp_pixel.onnx`,
  `models/raw_developer.json`, `tools/train_neural_isp.py` (and `__pycache__`).
- Remove `--train-raw-developer` and `--export-raw-pairs` from `main()`.
- `rg` for `raw_developer`, `neural_isp`, `neural_isp_` must come back clean.
- `tract` stays (photo-LUT / look features use it; RAW no longer does).

## Phase 5 — Verify & document

- `cargo test` (all), `cargo build --release` clean, `cargo clippy` on changed
  files.
- Fit against the real 46-pair folder; regenerate `models/raw_s_curve.json`.
- Distance check on the two known pairs (port the old
  `developed_raw_moves_toward_its_phone_jpeg` test, env-var driven):
  `LookProfile` distance to the phone JPEG must **decrease** from the
  undeveloped render; dump developed vs. JPEG PNGs to `/tmp` for a visual pass
  (`RAW_DEVELOP_RAW` / `RAW_DEVELOP_JPEG` / `RAW_DEVELOP_OUTPUT`).
- Update `image-processor/README.md` + root `README.md`: describe
  rawloader-basic + phone S-curve development; drop neural-ISP mentions.

## Open items / notes

- `LinearImage` currently skips orientation; `develop_raw` must apply it (check
  what imagepipe does for the base render so both agree).
- Confirm the Pixel DNGs decode as Integer (u16) — the Float path is handled
  but untested.
- If the fitted curve's highlight end clips too aggressively in bright scenes,
  the anchor percentile (currently 0.5) or the clamp behavior can be tuned
  without touching the architecture.
