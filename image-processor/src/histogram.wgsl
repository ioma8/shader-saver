@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> bins: array<atomic<u32>>;
// Origin and size, in pixels, of the region to measure (the crop). The
// histogram has to describe what will be exported, not the whole sensor frame.
@group(0) @binding(2) var<uniform> window: vec4<u32>;

@compute @workgroup_size(16, 16)
fn histogram_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= window.z || gid.y >= window.w { return; }
    let at = vec2<i32>(window.xy + gid.xy);
    let c   = textureLoad(input_tex, at, 0);
    let lum = dot(c.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let bin = min(u32(lum * 256.0), 255u);
    atomicAdd(&bins[bin], 1u);
}
