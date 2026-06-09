# Image Processor

A fast GPU image editor in Rust — wgpu compute shaders + egui. Every adjustment
runs as a compute pass on the GPU, so editing stays real-time even on large
images.

![](docs/screenshot.png)

## Features

- **Browse / Edit tabs** — folder thumbnail browser (background-threaded
  loading) and a full editor; click a thumbnail to edit it
- **Formats** — PNG, JPEG, TIFF, BMP, WebP, GIF, and camera RAW
  (DNG, CR2, CR3, NEF, ARW, ORF, RW2, RAF, PEF, …) via rawloader/imagepipe
- **Adjustments** — white balance (temp/tint), exposure (stops),
  brightness (Capture One-style midtone bias), contrast (logistic S-curve),
  blacks/shadows/highlights/whites tonal zones, box blur, unsharp mask,
  vignette — all hue-preserving (computed on luminance)
- **Levels** — draggable black/gamma/white handles under the live
  GPU-computed luminance histogram
- **Curves** — Photoshop-style: click the histogram to add points,
  natural cubic spline, applied as a 256-entry LUT
- **Auto** — one-click auto levels, brightness, contrast,
  shadows/highlights from the histogram
- **Persistent edits** — every image's edit state is saved to SQLite
  (`~/.image-processor/edits.db`) and restored when reopened;
  Reset All Edits reverts to defaults
- **Export** — save the processed result as PNG

## Run

```bash
cargo run --release                  # empty start
cargo run --release -- photo.cr2     # open an image
cargo run --release -- ~/Pictures    # browse a folder
```

## Editor shortcuts

- Scroll: zoom · drag: pan · double-click: 100% / fit
- Hold mouse or Space: show original
- Double-click a slider: reset it
- Curves: click to add a point, right-click or drag out to remove,
  double-click to reset
