# Crop

A non-destructive crop: `⛶ Crop` in the right panel arms the tool, you drag the
corners or edges (or inside, to move the rectangle), lock a ratio from the
presets if you want one, and `Reset crop` puts it back to the whole frame. It is
stored in the edit state, so it is undoable, it persists, and it travels with
everything else.

While the tool is armed the panel shows the **whole frame** — that is what you
crop against, with everything outside the rectangle dimmed and a rule of thirds
to place it. Once the tool is put away the panel shows the crop alone, so the
image area takes on the cropped shape.

## Whole pixels, and why that is the whole design

The crop is stored as `(x, y, w, h)` fractions of the input image, but every
consumer goes through `crop_px` (`src/processor.rs`), which snaps it to a
rectangle of whole pixels. That single decision is what keeps three things
consistent with no work:

- **the display** points the painter's uv window at the image texture, so the
  panel shows exactly those pixels;
- **the export** copies exactly those pixels out of the readback (`crop_pixels`),
  so the file matches the panel and needs no resampling;
- **the histogram** measures exactly that window, so it describes the exported
  image rather than the whole sensor frame.

`crop_fractions`/`crop_from_px` are the inverses, and a test pins the round trip
so the stored fractions and the pixel rect can never drift apart.

Anything that places something *on* the image goes through `ImageView`
(`src/ui.rs`), which knows the crop. That is how the retouch markers and clicks
follow the pixels: dabs are stored against the whole image, so cropping must not
move them, and clicking must invert the mapping exactly. Both directions are
tested.

## The tool

- Drag a corner or an edge to resize; drag inside to move. The grab reach
  shrinks with the rectangle so a small crop can still be moved rather than
  being all handle.
- Aspect presets: Free, Original, 1:1, 4:5, 5:4, 3:2, 2:3, 16:9. Locking one
  reshapes the current crop about its centre straight away, so the setting is
  visible without touching a handle.
- The rectangle never inverts itself, never drops below 16 px on a side, and
  never leaves the image (`drag_crop`, `crop_inside`), all tested.
- Panning and double-click-to-zoom stand aside while the tool owns the pointer,
  and the retouch tool is disarmed so one tool has the pointer at a time.
- Changing the crop needs no re-render of the edit chain — only the histogram,
  which is why a crop drag is cheap.

## Not implemented: straighten

Rotating the frame is deliberately absent. An axis-aligned pixel rect is what
makes display, export and histogram identical with no resampling pass; a
rotation needs a warped draw, the inscribed-rectangle constraint (rotate and
shrink the crop to stay inside the image), and its own overlay. That is a
separate piece of work, not a slider bolted on.

## Corrupt rows

The crop travels through a JSON column, so `sanitize_crop` runs on load — the
same hazard retouch dabs are guarded against. A non-finite crop used to reach
`f32::clamp`'s assertion with its bounds the wrong way round and **abort the
process**, and a zero-sized image did the same; `crop_px` is now total, and a
wild row is pulled back to something renderable. Both are regression-tested,
including the case where a valid crop must come back untouched.

## Tests

`src/processor.rs`: the pixel round trip, `fit_within` behaviour, `crop_pixels`
taking exactly the rect, corrupt-input robustness, and an end-to-end GPU export
of a cropped region. `src/ui.rs`: the drag rules, a locked ratio staying locked,
handle picking, the view mapping both ways, and every aspect preset having a
ratio.
