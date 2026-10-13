# Retouch brush

A healing brush for spots, dust and small intrusions. The tool lives in the
right panel (`⚕ Retouch`): arm it, click the spot you want gone, and it picks a
patch of the same image to put in its place -- blended and tone-matched, so the
join does not show. It is deliberately close to Photoshop's *spot healing
brush*: you never pick the source yourself.

It is a spot tool, not an object remover: make the brush **bigger than the
blemish**, so the blemish sits inside the plateau below and the ring around it
is clean background. A blemish that reaches into the ring -- a long wire
crossing the whole frame, say -- has nothing clean for the search to match, and
a blemish larger than the brush cannot be taken out at all.

The whole layer is removable. Dabs are stored in the edit state next to the
sliders, not painted into the pixels, so later adjustments still apply to them
and they can be undone individually.

## Using it

| Action | Result |
|---|---|
| Click `⚕ Retouch` | Arm the tool. Dab markers and the brush cursor appear. |
| Click the photo | Heal that spot. The dab's ring becomes a marker. |
| `[` / `]` | Shrink / grow the brush. The cursor ring resizes under the pointer. |
| Right-click a marker | Delete that dab, restoring the original pixels. |
| `Clear N retouch(es)` | Delete every dab on this image (two clicks: it cannot be undone). |
| Click `⚕ Retouch` again | Put the tool away. Dabs stay; markers and the cursor go. |
| Reset All Edits | Clears the dabs along with every other edit. |

A dab is a layer, not a pixel edit: it re-renders from the current pixels, so
a later exposure or contrast change keeps it, and removing one restores the
render byte-for-byte (asserted by the GPU test below).

## Where the source patch comes from

When you click, the app reads back the pixels the shader will sample -- the
image with every edit applied but *before* the retouch layer -- cropped to the
region the dab can reach, and searches a ring of candidate patches around the
spot (`src/retouch.rs`). Reading the finished display instead would let the
search score candidates against pixels the render never uses.

Candidates sit on four distances (`2.2`, `3`, `4.2`, `6` brush radii) around the
target, at 32 angles, so the direction is free. Each is scored on two things:

- **Does its ring match?** Only the outer band is compared -- the plateau is
  replaced outright, so scoring it would reward a source that looks like the
  blemish. Point the brush at a dust spot and the whole-dab version pastes a
  second dust spot; matching the ring instead picks the clean sky around it.
  A mild pull towards nearby sources keeps the replacement related to its
  surroundings, and candidates that would overlap the target are rejected.
- **Would pasting it drop something the surroundings do not have?** The band is
  not scored, so the interior has to be checked separately -- whatever sits
  there is about to be copied onto the photo. A source whose interior holds
  something the surroundings do not is rejected, with the tolerance set by how
  much the surroundings themselves vary, so texture is not penalised but an
  anomaly on a smooth background is.

The winner gets an **affine colour correction** (`gain`, then `bias`) fitted
over its ring. Fitting a gain as well as an offset is the "healing" half: a
patch lifted from a darker area is scaled onto the target's tone, and because
the gain multiplies the source, a later exposure change scales both together
and the dab stays put -- a bare additive offset would be left behind. A source
patch flat enough that there is nothing to fit against falls back to the offset
alone.

The result is stored as an offset plus that correction, not as copied pixels.
Every later render repeats the gesture from the *current* pixels.

Dabs never sample each other: each one copies its source out of the image
underneath the layer, not out of another dab's result. Two dabs therefore
compose in any order, and overlapping them cannot bring a blemish back.

## How it renders

The dab list is a uniform the GPU reads in one extra compute pass
(`retouch_pass` in `src/shaders.wgsl`), placed after the whole edit chain and
before the 8-bit display copy:

```
contrast → tonal → sharpen → blur → retouch → display
```

Each dab copies its source with a bilinear tap (so a non-integer offset does
not jitter) and blends it in through a smooth radial weight, so there is no
hard edge. Up to
64 dabs per image (`MAX_RETOUCH_SPOTS`); the shader loops only over the live
count, and with no dabs the pass is a straight copy.

All of a dab's geometry is resolution-independent: the centre and source offset
are fractions of width/height, and the radius a fraction of the shorter side,
so the same dab lands in the same place whichever resolution the image is
reopened at.

## State and reversibility

`EditState::retouch_spots` (`src/processor.rs`) holds the list, so it travels
with the edit through SQLite like everything else -- old rows load with no
dabs. Because dabs are a layer rather than pixels:

- right-clicking or `Clear` removes them exactly, leaving the render byte-for-
  byte what it was before (asserted by the GPU test below);
- a loaded list is clamped and any unusable dab dropped, so a hand-edited or
  corrupt database cannot NaN the image;
- Reset All Edits drops them with the rest of the edit;
- nothing is destructive, so the tool can be used freely.

## Keyboard

The keys are matched on their **physical** position (`KeyCode::BracketLeft` /
`BracketRight`), not on the character they type. On the Czech layout `[` and `]`
sit under `ú` and `)`, so a character match would miss; the physical key is the
same on every layout.

A click that places nothing says so, briefly, in the panel -- a brush with no
room to resolve a source is clamped before it gets there, but a degenerate case
should never look like the tool is broken.

## Tests

- `src/retouch.rs` covers the search: it never sources from the dab it
  replaces, it prefers a same-tone neighbour, a dust spot is not healed from
  another dust spot, the affine correction lands the copied patch on the ring
  it has to blend into, a brush nearly as large as the frame still resolves a
  source, and a cropped view agrees with the whole image.
- `processor::tests::retouch_layer_applies_and_reverses_a_dab` renders real
  dabs on the GPU and checks the centre takes the corrected source at full
  strength out to the plateau, pixels outside are untouched, the rim eases off
  between the two, a second dab is not confused with the first, and clearing
  them restores the render exactly. It skips when no GPU adapter is available.
- `processor::tests::retouch_uniform_layout_matches_the_shader` pins the
  uniform packing (three `vec4`s per dab) against the buffer size the shader
  declares; `a_corrupt_dab_list_is_pulled_back_into_range` covers sanitising.
- `ui::marker_tests` checks that every dab is marked and that a right-click
  deletes the dab drawn on top; `ui::pointer_tests` covers the click-to-dab and
  right-click plumbing; `ui::tests` covers the layout-independent `[` / `]`
  handling and the brush size clamps.
