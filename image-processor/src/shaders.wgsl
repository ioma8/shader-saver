@group(0) @binding(0) var input_tex:  texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba8unorm, write>;

struct Params { value: f32 }
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn contrast_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let c = textureLoad(input_tex, vec2<i32>(gid.xy), 0);
    let rgb = clamp((c.rgb - 0.5) * params.value + 0.5, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(rgb, c.a));
}

@compute @workgroup_size(8, 8)
fn blur_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let radius = i32(params.value);
    if radius == 0 {
        textureStore(output_tex, vec2<i32>(gid.xy), textureLoad(input_tex, vec2<i32>(gid.xy), 0));
        return;
    }
    var sum = vec4<f32>(0.0);
    var count = 0.0;
    for (var x = -radius; x <= radius; x++) {
        for (var y = -radius; y <= radius; y++) {
            let coord = vec2<i32>(
                clamp(i32(gid.x) + x, 0, i32(dims.x) - 1),
                clamp(i32(gid.y) + y, 0, i32(dims.y) - 1),
            );
            sum += textureLoad(input_tex, coord, 0);
            count += 1.0;
        }
    }
    textureStore(output_tex, vec2<i32>(gid.xy), sum / count);
}

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
    for (var x = -2; x <= 2; x++) {
        for (var y = -2; y <= 2; y++) {
            let coord = vec2<i32>(
                clamp(i32(gid.x) + x, 0, i32(dims.x) - 1),
                clamp(i32(gid.y) + y, 0, i32(dims.y) - 1),
            );
            blur += textureLoad(input_tex, coord, 0);
            count += 1.0;
        }
    }
    blur /= count;
    let sharpened = clamp(orig + (orig - blur) * params.value, vec4<f32>(0.0), vec4<f32>(1.0));
    textureStore(output_tex, vec2<i32>(gid.xy), sharpened);
}
