---
name: feedback_idiomatic_rust_wgpu
description: When porting Swift/Metal code to Rust/wgpu, prefer idiomatic Rust/wgpu patterns over replicating Swift app structure
metadata:
  type: feedback
---

Prefer idiomatic wgpu/Rust patterns over replicating Swift/Metal workflow when porting the STL viewer.

**Why:** User explicitly called this out during design — don't mirror Swift app structure, write it the way wgpu best practices dictate.

**How to apply:** When implementing the Rust STL viewer, follow wgpu examples and community patterns (e.g. wgpu's own examples repo) rather than translating Swift constructs 1:1.
