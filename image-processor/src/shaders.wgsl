@group(0) @binding(0) var input_tex:  texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba8unorm, write>;

struct Params { v0: f32, v1: f32, v2: f32, v3: f32 }
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
    let rgb = clamp((c.rgb - 0.5) * params.v0 + 0.5, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(rgb, c.a));
}

// Separable box blur — horizontal pass (params.v0 = radius)
@compute @workgroup_size(8, 8)
fn blur_h_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let radius = i32(params.v0);
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

// Separable box blur — vertical pass (params.v0 = radius)
@compute @workgroup_size(8, 8)
fn blur_v_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let radius = i32(params.v0);
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

// Unsharp mask — params.v0 = strength, params.v1 = mask blur radius (box)
// Receives the contrast-adjusted (pre-blur) image so the mask anchors off clean signal.
@compute @workgroup_size(8, 8)
fn sharpen_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let orig = textureLoad(input_tex, vec2<i32>(gid.xy), 0);
    if params.v0 == 0.0 {
        textureStore(output_tex, vec2<i32>(gid.xy), orig);
        return;
    }
    var blur = vec4<f32>(0.0);
    var count = 0.0;
    let r = i32(params.v1);
    for (var x = -r; x <= r; x++) {
        for (var y = -r; y <= r; y++) {
            blur  += load_clamped(i32(gid.x) + x, i32(gid.y) + y, dims);
            count += 1.0;
        }
    }
    blur /= count;
    textureStore(output_tex, vec2<i32>(gid.xy),
        clamp(orig + (orig - blur) * params.v0, vec4<f32>(0.0), vec4<f32>(1.0)));
}

// Tonal adjustments — params: blacks, shadows, highlights, whites (each -100..100)
// Zone weights form a partition of unity: at any luminance their sum == 1.
// Adjustment is applied as a luminance-proportional scale to preserve hue.
@compute @workgroup_size(8, 8)
fn tonal_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let c = textureLoad(input_tex, vec2<i32>(gid.xy), 0);

    if params.v0 == 0.0 && params.v1 == 0.0 && params.v2 == 0.0 && params.v3 == 0.0 {
        textureStore(output_tex, vec2<i32>(gid.xy), c);
        return;
    }

    // Scale: ÷100 maps slider range to [-1,1], ×0.4 sets max luminance shift to ±0.4
    let blacks     = params.v0 * 0.004;
    let shadows    = params.v1 * 0.004;
    let highlights = params.v2 * 0.004;
    let whites     = params.v3 * 0.004;

    let lum = dot(c.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));

    // Four overlapping bands that partition luminance [0, 1] — weights sum to 1.0
    let w_blacks     = 1.0 - smoothstep(0.0, 0.33, lum);
    let w_shadows    = smoothstep(0.0, 0.33, lum) * (1.0 - smoothstep(0.33, 0.66, lum));
    let w_highlights = smoothstep(0.33, 0.66, lum) * (1.0 - smoothstep(0.66, 1.0, lum));
    let w_whites     = smoothstep(0.66, 1.0, lum);

    let delta   = w_blacks * blacks + w_shadows * shadows
                + w_highlights * highlights + w_whites * whites;
    let new_lum = clamp(lum + delta, 0.0, 1.0);

    // Proportional RGB scale preserves hue; fall back to 1.0 for near-black pixels
    let scale = select(new_lum / lum, 1.0, lum < 0.001);
    textureStore(output_tex, vec2<i32>(gid.xy),
        vec4<f32>(clamp(c.rgb * scale, vec3<f32>(0.0), vec3<f32>(1.0)), c.a));
}
