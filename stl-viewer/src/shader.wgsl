struct Uniforms {
    model: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = uniforms.model * vec4<f32>(in.position, 1.0);
    out.clip_position = uniforms.proj * uniforms.view * world_pos;
    out.world_normal = (uniforms.model * vec4<f32>(in.normal, 0.0)).xyz;
    out.world_pos = world_pos.xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let light_dir = normalize(vec3<f32>(1.0, 2.0, -1.0));
    let diffuse = max(0.0, dot(n, light_dir));
    let back_fill = max(0.0, dot(-n, light_dir)) * 0.2;
    var color = vec3<f32>(0.8, 0.7, 0.6) * (diffuse + 0.15 + back_fill);

    let view_dir = normalize(uniforms.camera_pos.xyz - in.world_pos);
    var rim = 1.0 - max(0.0, dot(n, view_dir));
    rim = pow(rim, 2.5);
    color += vec3<f32>(0.2, 0.5, 1.0) * rim * 1.8;

    return vec4<f32>(color, 1.0);
}
