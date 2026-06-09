# Rust STL Viewer — Design Spec

**Date:** 2026-06-09  
**Goal:** Rewrite the current Swift/Metal STL viewer as a standalone Rust desktop app with cross-platform GPU support (macOS, Linux, Windows) using wgpu + winit + WGSL.

---

## Overview

A single Rust binary named `stl-viewer`. Same CLI interface as the current app:

```
stl-viewer file.stl    # open directly
stl-viewer             # open native file picker dialog
```

Supports all GPU backends automatically via wgpu: Metal (macOS), Vulkan (Linux/Windows, Nvidia/AMD/Intel), DirectX 12 (Windows fallback). Shaders written once in WGSL.

---

## Project Structure

```
stl-viewer/
├── Cargo.toml
└── src/
    ├── main.rs          # entry point, CLI arg parsing, file picker fallback, winit event loop
    ├── parser.rs        # STL binary + ASCII parser → Vec<Vertex>
    ├── renderer.rs      # wgpu device, pipeline, buffers, draw
    ├── camera.rs        # rotation/zoom state, matrix computation (glam)
    └── shader.wgsl      # single WGSL shader file (vertex + fragment)
```

---

## Dependencies

| Crate | Purpose |
|-------|---------|
| `wgpu` | GPU abstraction — Metal, Vulkan, DX12 |
| `winit` | Cross-platform window + input events |
| `bytemuck` | Safe cast of Rust structs to GPU bytes (`Pod`/`Zeroable`) |
| `glam` | Math — `Vec3`, `Mat4`, replaces manual Swift matrix helpers |
| `rfd` | Native file picker dialog (macOS, Linux, Windows) |
| `pollster` | Block on async wgpu init in sync `main` |

---

## Data Structures

```rust
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal:   [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    model:      [[f32; 4]; 4],
    view:       [[f32; 4]; 4],
    proj:       [[f32; 4]; 4],
    camera_pos: [f32; 4],   // w unused, padded for alignment
}
```

---

## GPU Pipeline

`renderer.rs` owns:
- `wgpu::Device` + `wgpu::Queue`
- `wgpu::RenderPipeline` — compiled from `shader.wgsl` at startup
- `wgpu::Buffer` for vertices (uploaded once at load)
- `wgpu::Buffer` for uniforms (written every frame via `queue.write_buffer`)
- `wgpu::BindGroup` for uniforms
- Depth texture (`Depth32Float`) with `Less` compare, write enabled

Pipeline built idiomatically following wgpu patterns: explicit `VertexBufferLayout`, `BindGroupLayout`, and `PipelineLayout`.

---

## Shader (WGSL)

Single file `shader.wgsl` with vertex and fragment entry points. Lighting logic is a direct translation of the current Metal shader:

**Vertex stage:** Transform position and normal to world space, output clip-space position.

**Fragment stage:**
- Diffuse lighting: `max(0.0, dot(n, light_dir))` with fixed directional light `(1, 2, -1)`
- Ambient: constant `0.15` added to diffuse
- Back-fill: `max(0.0, dot(-n, light_dir)) * 0.2` — softens unlit faces
- Rim lighting: `pow(1.0 - max(0.0, dot(n, view_dir)), 2.5)` — view-dependent silhouette glow in blue `(0.2, 0.5, 1.0)`

---

## Camera & Input

`camera.rs` holds:
- `rotation_x: f32`, `rotation_y: f32` — mouse drag
- `camera_distance: f32` — scroll wheel, clamped `0.5..=20.0`
- Computes model/view/projection matrices each frame using `glam`
- Model pipeline: center → scale to unit size → Z-up to Y-up rotation → user rotation

`winit` event handling in `main.rs`:
- `MouseButton::Left` down/move → rotate
- `MouseWheel` → zoom
- `Key::Escape` → quit
- `WindowEvent::DroppedFile` → reload STL (drag-and-drop onto window)

---

## STL Parser

`parser.rs` handles both binary and ASCII STL formats, same logic as current Swift implementation. Returns `Vec<Vertex>`. Model is centered and scaled to fit a unit bounding box before upload.

---

## Error Handling

- Missing/invalid STL file: print error and exit with non-zero code
- No GPU available: print error and exit
- No wgpu adapter found: print error and exit

No panics in production paths — use `Result` propagation to `main`.

---

## Out of Scope

- No screensaver integration (that stays in Swift)
- No animation or auto-rotation
- No lighting controls UI
- No export functionality
