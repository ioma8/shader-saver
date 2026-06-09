@group(0) @binding(0) var input_tex:  texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba8unorm, write>;

struct Params { value: f32, value2: f32 }
@group(0) @binding(2) var<uniform> params: Params;

// Shared helper — clamp-to-edge load used by all passes
fn load_clamped(x: i32, y: i32, dims: vec2<u32>) -> vec4<f32> {
    let cx = clamp(x, 0, i32(dims.x) - 1);
    let cy = clamp(y, 0, i32(dims.y) - 1);
    return textureLoad(input_tex, vec2<i32>(cx, cy), 0);
}

@compute @workgroup_size(8, 8)
fn contrast_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let c = textureLoad(input_tex, vec2<i32>(gid.xy), 0);
    let rgb = clamp((c.rgb - 0.5) * params.value + 0.5, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(rgb, c.a));
}

// Separable box blur — horizontal pass (params.value = radius)
@compute @workgroup_size(8, 8)
fn blur_h_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let radius = i32(params.value);
    if radius == 0 {
        textureStore(output_tex, vec2<i32>(gid.xy), textureLoad(input_tex, vec2<i32>(gid.xy), 0));
        return;
    }
    var sum = vec4<f32>(0.0);
    for (var x = -radius; x <= radius; x++) {
        sum += load_clamped(i32(gid.x) + x, i32(gid.y), dims);
    }
    textureStore(output_tex, vec2<i32>(gid.xy), sum / f32(2 * radius + 1));
}

// Separable box blur — vertical pass (params.value = radius)
@compute @workgroup_size(8, 8)
fn blur_v_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let radius = i32(params.value);
    if radius == 0 {
        textureStore(output_tex, vec2<i32>(gid.xy), textureLoad(input_tex, vec2<i32>(gid.xy), 0));
        return;
    }
    var sum = vec4<f32>(0.0);
    for (var y = -radius; y <= radius; y++) {
        sum += load_clamped(i32(gid.x), i32(gid.y) + y, dims);
    }
    textureStore(output_tex, vec2<i32>(gid.xy), sum / f32(2 * radius + 1));
}

// Unsharp mask — params.value = strength, params.value2 = mask blur radius
// Receives the contrast-adjusted (pre-blur) image so the mask anchors off clean signal.
@compute @workgroup_size(8, 8)
fn sharpen_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let orig = textureLoad(input_tex, vec2<i32>(gid.xy), 0);
    if params.value == 0.0 {
        textureStore(output_tex, vec2<i32>(gid.xy), orig);
        return;
    }
    var blur = vec4<f32>(0.0);
    var count = 0.0;
    let r = i32(params.value2);
    for (var x = -r; x <= r; x++) {
        for (var y = -r; y <= r; y++) {
            blur  += load_clamped(i32(gid.x) + x, i32(gid.y) + y, dims);
            count += 1.0;
        }
    }
    blur /= count;
    textureStore(output_tex, vec2<i32>(gid.xy),
        clamp(orig + (orig - blur) * params.value, vec4<f32>(0.0), vec4<f32>(1.0)));
}
