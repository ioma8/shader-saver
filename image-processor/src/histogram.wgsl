@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> bins: array<atomic<u32>>;

@compute @workgroup_size(16, 16)
fn histogram_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let c   = textureLoad(input_tex, vec2<i32>(gid.xy), 0);
    let lum = dot(c.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let bin = min(u32(lum * 256.0), 255u);
    atomicAdd(&bins[bin], 1u);
}
