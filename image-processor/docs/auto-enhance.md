# Auto Enhance

The **Auto** button should turn any photo — RAW-developed or JPEG — into a
pleasing image by moving the *visible* editor sliders only. No LUT, no hidden
transform, no sidecar data: after Auto runs, every adjustment is a normal slider
the user can see and change.

## How it works today (and why it is being replaced)

`main.rs` has two paths. It first asks `enhance::Enhancer`, which runs the
embedded **Image-Adaptive-3DLUT** model (tract/ONNX) and applies three 33³ LUTs
through the GPU (`ai_lut_enabled`). Only if that fails does it fall back to
`Processor::auto_adjust`, a histogram heuristic that sets `levels_black/white`,
`brightness`, `shadows`, `highlights` and `contrast` and nothing else.

The LUT path is the one being dropped. It also cannot be edited afterwards, and
`docs/research/auto-adjustment-onnx-models.md` already notes that forcing its
output into sliders is a poor approximation. But its *quality* is fine — the
same note selected it over IAT after a differential test.

## The reframe

"Beautiful" is not a property of sliders; it is a **target**. Auto is really
*"pick a pleasing target, then fit the visible sliders to it."* That splits the
work cleanly:

- an **analytic seed** (Phase 1) that always runs and is the fallback, and
- a **learned residual** (Phase 2) that supplies the taste the seed cannot.

Both must finish well under the button's budget; the target is **instant
(< 50 ms)**, so the taste is a small forward pass, never a search.

## Phase 1 — analytic seed (`src/auto.rs`)

Everything tonal in the shader is luminance-only and applied back as a uniform
RGB scale, so a **256-bin luma histogram is a sufficient statistic**. The seed
therefore reads the photo's own statistics and sets, in one pass:

- **White balance** — gray-world over near-neutral mid-tone pixels, converted to
  `TEMP`/`TINT` through the shader's own gain model, then halved so the scene
  keeps some of its mood.
- **Levels** — `levels_black`/`levels_white` at the 0.3/99.7 percentiles, and
  `levels_gamma` chosen to put the median at 0.45.
- **Contrast** — from the post-levels inter-quartile range toward ~0.30.
- **Shadows / highlights** — recovery only when the tails are actually bunched.
- **Vibrance** — modest, larger when the image is already colour-poor.

It never touches `exposure` (the downstream levels would only re-normalise it
and risk clipping first), `blacks`/`whites` (levels already own the endpoints),
`sharpen`, or `vignette`.

Output is a full `EditState`, so Auto is a normal, persisted, editable edit.

## Phase 2 — learned residual

Train a small network that predicts a **correction on top of the seed**, so a
bad prediction degrades to Phase 1 rather than to nonsense.

1. **Data.** MIT-Adobe FiveK, expert C, 480 p. 5,000 input/retouched pairs; the
   standard "auto enhancement" target.
2. **Fit sliders per pair, offline.** For each pair, the same analytic fit used
   in Phase 1 is refined against the retouched image to get a target slider
   vector. The fit is cheap because it is mostly closed form on the histograms.
3. **Distil.** Train a tiny CNN (≈ MobileNet-scale, 256 px input) to predict the
   residual `target − seed`. Export to ONNX; ship it beside the tagger and run
   it with `tract` on the CPU (< 50 ms).
4. **Runtime.** `seed = auto::suggest_edit(img)`;
   `state = seed + clamp(model(img))`. If the model is missing, Auto is Phase 1.

### Data options (to decide)

| Source | Size | Notes |
|---|---|---|
| `HuiZeng/Image-Adaptive-3DLUT` 480 p (GDrive/OneDrive) | ~14 GB | Canonical 480 p FiveK (sRGB in + expert C). Same set the LUT was trained on. |
| `HoangPhuc7678/FiveK` on HF | 14.4 GB | `hf download`, paired `img`/`gt`, scriptable today. Provenance of `gt` to confirm. |
| `logasja/mit-adobe-fivek` on HF | 607 GB | Full resolution — too heavy. |
| `data.csail.mit.edu/graphics/fivek` archive | ~50 GB | Original. Also ships the **Lightroom slider values** for all five experts — useful as a sanity prior. |

## Phase 3 — validation

- `auto_probe`: render a contact sheet of many sample photos with Auto applied,
  for quick visual review (the repo's established way to judge develop output).
- A bench against FiveK expert C on the held-out split (MAE, plus a
  clipping/hue sanity check).
- Hard guardrails kept in the metric: neither seed nor model may raise the
  clipped fraction or push skin/neutrals out of gamut.

## Explicitly not doing

- **No LUT and no learned image output.** The user's constraint, and it keeps
  the result editable.
- **No search against an aesthetic model (NIMA).** `judge.rs` records that an
  earlier free search against a learned metric "scored well while visibly
  wrecking pictures". NIMA may only ever be a weak tie-breaker bounded by the
  guardrails above.

## Status

- [x] Phase 1 analytic seed (`src/auto.rs`, wired to the Auto button)
- [x] Phase 2 pipeline: FiveK download, pair extraction, shader-replica renderer,
  residual CNN, ONNX export (`tools/fivek/`)
- [x] Phase 2 runtime: seed + `clamp(residual)` in `src/auto/model.rs`
- [ ] Phase 3 probe + held-out bench, and visual tuning of the seed

Phase 2 closes the loop: `tools/fivek/train.py` fits the residual CNN against
expert retouches, exports `models/auto_residual.onnx`, and `src/auto/model.rs`
runs it with `tract` and applies `clamp(seed + residual * scale)` to the visible
sliders. If the model file is missing or fails, Auto degrades to the seed alone.

## The domain gap (important)

FiveK's inputs are flat, dark, old-camera images that need a heavy correction.
A model trained on them with generous residual caps learns a near-constant
"darken + saturate + crush blacks" move: it is a large win in distribution
(validation MAE 0.122 → 0.065) but **destructive out of distribution** — on an
already-good RAW develop it produced black blotches in foliage and dark edge
fringing, because the seed has already opened the shadows that FiveK's targets
crush.

The shipped model therefore keeps every residual cap tight and adds an L2
penalty (`--reg`) so it moves only where it clearly helps. It now behaves
sensibly: residuals are large on FiveK-like flat inputs and near zero on an
already-exposed photo. In-distribution MAE is 0.122 → 0.073; out of distribution
Auto is effectively the seed, which is the safe fallback.

The consequence is that on modern, already-well-exposed photos the learned
residual adds little. Clearing a "really beautiful" bar for those inputs needs a
taste source that matches them — expert retouches of modern photos, or the
user's own edits — not FiveK alone.
