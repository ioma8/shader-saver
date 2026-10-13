# Retouch brush

A healing brush for spots, dust and small intrusions. The tool lives in the
right panel (`⚕ Retouch`): arm it, click the spot you want gone, and it picks a
patch of the same image to put in its place -- feathered and tone-matched, so
the join does not show. It is deliberately close to Photoshop's *spot healing
brush*: you never pick the source yourself.

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
| `Clear N retouch(es)` | Delete every dab on this image. |
| Click `⚕ Retouch` again | Put the tool away. Dabs stay; markers and the cursor go. |
| Reset All Edits | Clears the dabs along with every other edit. |

A dab is a layer, not a pixel edit: it re-renders from the current pixels, so
a later exposure or contrast change keeps it, and removing one restores the
render byte-for-byte (asserted by the GPU test below).

## Where the source patch comes from

When you click, the app reads the pixels **as rendered on screen** -- so the
colour the search sees is the colour you clicked, whatever sliders produced it
-- and searches a ring of candidate patches around the spot (`src/retouch.rs`):

- candidates sit on four distances (`2.2`, `3`, `4.2`, `6` brush radii) around
  the target, at 32 angles, so the direction is free;
- each is scored by the sum-of-squared-differences between the target patch and
  the candidate over a lattice of samples inside the dab, with a mild pull
  towards nearby sources so the result stays related to its surroundings;
- candidates that would overlap the target are rejected, or a dab would copy
  the very thing it is meant to remove;
- the winner's **mean colour difference** to the target is recorded as the
  dab's `bias`.

That bias is the "healing" half of a healing brush: a patch lifted from a
slightly darker or cooler area is shifted onto the target's tone before it is
blended, so a mismatched source still disappears. The search therefore prefers
a same-tone neighbour (small bias) but copes when one is not available.

The result is stored as an offset, not as copied pixels. Every later render
repeats the gesture from the *current* pixels, so a dab that heals an area
composes with a later exposure or contrast change instead of going stale.

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
- Reset All Edits drops them with the rest of the edit;
- nothing is destructive, so the tool can be used freely.

## Keyboard

The keys are matched on their **physical** position (`KeyCode::BracketLeft` /
`BracketRight`), not on the character they type. On the Czech layout `[` and `]`
sit under `ú` and `)`, so a character match would miss; the physical key is the
same on every layout.

## Tests

- `src/retouch.rs` covers the source search: it never sources from the dab it
  replaces, it prefers a same-tone neighbour, and the recorded bias lands the
  copied patch on the target's mean.
- `processor::tests::retouch_layer_applies_and_reverses_a_dab` renders a real
  dab on the GPU and checks the centre takes the corrected source, pixels
  outside are untouched, the rim is feathered between the two, and clearing the
  dab restores the render exactly. It skips when no GPU adapter is available.
- `processor::tests::retouch_uniform_layout_matches_the_shader` pins the
  uniform packing against the buffer size the shader declares.
