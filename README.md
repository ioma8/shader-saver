# shader-saver

GPU graphics experiments: two Metal projects in a Swift package, plus a Rust image editor.

## Image Processor

A real-time GPU image editor (Rust, wgpu compute shaders + egui): thumbnail
browser, RAW support, levels/curves/tonal adjustments, auto-correction, and
SQLite-persisted edits. See [image-processor/README.md](image-processor/README.md).

## STLViewer

A minimal macOS STL model viewer built on Metal.

**Features**
- Binary and ASCII STL support
- Diffuse lighting + rim lighting
- Mouse drag to rotate, scroll to zoom
- Auto-centers and scales any model to fit

**Install as default .stl handler**

```bash
./install-stlviewer.sh
```

Builds a release `.app`, installs to `/Applications`, and sets it as the default app for `.stl` files via `duti` (installed automatically via Homebrew if needed).

**Run manually**

```bash
swift run STLViewer /path/to/model.stl
# or without argument — shows an open file dialog
swift run STLViewer
```

## ShaderSaver

A fullscreen nebula screensaver rendered entirely in a Metal fragment shader using volumetric raymarching. No geometry — the entire 3D scene is a procedural math function evaluated per pixel.

```bash
swift run ShaderSaver
```

Press Escape to quit.

## Requirements

- macOS 13+
- Xcode / Swift toolchain
