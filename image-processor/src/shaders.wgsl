@group(0) @binding(0) var input_tex:  texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba8unorm, write>;

struct Params { v0: f32, v1: f32, v2: f32, v3: f32 }
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> curve_lut: array<f32, 256>;

// Tone curve lookup with linear interpolation between LUT entries
fn curve_apply(v: f32) -> f32 {
    let x = clamp(v, 0.0, 1.0) * 255.0;
    let i = u32(floor(x));
    let j = min(i + 1u, 255u);
    return mix(curve_lut[i], curve_lut[j], x - f32(i));
}

// Shared helper — clamp-to-edge load used by all passes
fn load_clamped(x: i32, y: i32, dims: vec2<u32>) -> vec4<f32> {
    let cx = clamp(x, 0, i32(dims.x) - 1);
    let cy = clamp(y, 0, i32(dims.y) - 1);
    return textureLoad(input_tex, vec2<i32>(cx, cy), 0);
}

// params: v0=contrast, v1=levels_black (0-255), v2=levels_white (0-255), v3=levels_gamma
@compute @workgroup_size(8, 8)
fn contrast_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let c = textureLoad(input_tex, vec2<i32>(gid.xy), 0);

    // Levels: remap [black, white] → [0, 1] then apply midtone gamma
    let black = params.v1 / 255.0;
    let white = params.v2 / 255.0;
    let range = max(white - black, 0.001);
    var rgb = clamp((c.rgb - black) / range, vec3<f32>(0.0), vec3<f32>(1.0));
    rgb = pow(rgb, vec3<f32>(1.0 / max(params.v3, 0.01)));

    // Contrast S-curve around midpoint
    rgb = clamp((rgb - 0.5) * params.v0 + 0.5, vec3<f32>(0.0), vec3<f32>(1.0));

    // Tone curve (Photoshop-style curves; identity LUT when no points added)
    rgb = vec3<f32>(curve_apply(rgb.r), curve_apply(rgb.g), curve_apply(rgb.b));
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
// Lightroom-style zones: shadows/highlights are endpoint-anchored bumps,
// blacks/whites shift the endpoints themselves.
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

    // ÷100 maps slider range to [-1, 1]
    let blacks     = params.v0 * 0.01;
    let shadows    = params.v1 * 0.01;
    let highlights = params.v2 * 0.01;
    let whites     = params.v3 * 0.01;

    let lum = dot(c.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let inv = 1.0 - lum;

    // Shadows/highlights: polynomial bumps zero at BOTH endpoints, so the black
    // and white points stay anchored (no haze when lifting shadows). Peaks are
    // at lum 0.25 / 0.75 with broad falloff covering roughly half the range.
    // 9.4815 = 1 / (0.25 * 0.75^3) normalizes the peak to 1.0.
    let w_shadows    = 9.4815 * lum * inv * inv * inv;
    let w_highlights = 9.4815 * lum * lum * lum * inv;

    // Blacks/whites: tight falloff from the endpoints — these intentionally
    // move the black/white points themselves (clipping-style control).
    let w_blacks = inv * inv * inv * inv * inv * inv;
    let w_whites = lum * lum * lum * lum * lum * lum;

    let delta = blacks * 0.25 * w_blacks
              + shadows * 0.35 * w_shadows
              + highlights * 0.35 * w_highlights
              + whites * 0.25 * w_whites;
    let new_lum = clamp(lum + delta, 0.0, 1.0);

    // Proportional RGB scale preserves hue; fall back to 1.0 for near-black pixels
    let scale = select(new_lum / lum, 1.0, lum < 0.001);
    textureStore(output_tex, vec2<i32>(gid.xy),
        vec4<f32>(clamp(c.rgb * scale, vec3<f32>(0.0), vec3<f32>(1.0)), c.a));
}
