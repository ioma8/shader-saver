use glam::{Mat4, Vec3};
use std::path::Path;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

pub fn load_stl(path: &Path) -> Option<Vec<Vertex>> {
    let data = std::fs::read(path).ok()?;
    let is_ascii = data.len() > 5 && data[..5] == *b"solid";
    let mut vertices = if is_ascii {
        if let Some(text) = std::str::from_utf8(&data).ok() {
            parse_ascii(text)
        } else {
            parse_binary(&data)
        }
    } else {
        parse_binary(&data)
    };

    if vertices.is_empty() {
        return None;
    }

    normalize_vertices(&mut vertices);
    Some(vertices)
}

fn parse_binary(data: &[u8]) -> Vec<Vertex> {
    if data.len() < 84 {
        return vec![];
    }
    let triangle_count = u32::from_le_bytes(data[80..84].try_into().unwrap()) as usize;
    let mut vertices = Vec::with_capacity(triangle_count * 3);

    for i in 0..triangle_count {
        let base = 84 + i * 50;
        if base + 50 > data.len() {
            break;
        }
        let normal = read_vec3(data, base);
        for v in 0..3 {
            vertices.push(Vertex {
                position: read_vec3(data, base + 12 + v * 12),
                normal,
            });
        }
    }
    vertices
}

fn read_vec3(data: &[u8], offset: usize) -> [f32; 3] {
    [
        f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()),
        f32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap()),
        f32::from_le_bytes(data[offset + 8..offset + 12].try_into().unwrap()),
    ]
}

fn parse_ascii(text: &str) -> Vec<Vertex> {
    let mut vertices = Vec::new();
    let mut current_normal = [0f32; 3];
    let mut face_verts: Vec<[f32; 3]> = Vec::new();

    for line in text.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts.as_slice() {
            ["facet", "normal", nx, ny, nz] => {
                current_normal = [
                    nx.parse().unwrap_or(0.0),
                    ny.parse().unwrap_or(0.0),
                    nz.parse().unwrap_or(0.0),
                ];
                face_verts.clear();
            }
            ["vertex", x, y, z] => {
                face_verts.push([
                    x.parse().unwrap_or(0.0),
                    y.parse().unwrap_or(0.0),
                    z.parse().unwrap_or(0.0),
                ]);
            }
            ["endfacet"] => {
                for &pos in &face_verts {
                    vertices.push(Vertex { position: pos, normal: current_normal });
                }
            }
            _ => {}
        }
    }
    vertices
}

// Center at origin, scale to unit size, rotate Z-up (STL/CAD convention) → Y-up
fn normalize_vertices(vertices: &mut Vec<Vertex>) {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for v in vertices.iter() {
        let p = Vec3::from(v.position);
        min = min.min(p);
        max = max.max(p);
    }
    let center = (min + max) * 0.5;
    let size = (max - min).length();
    let scale = if size > 0.0 { 2.0 / size } else { 1.0 };

    // Z-up to Y-up: rotate -90° around X
    let transform = Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2)
        * Mat4::from_scale(Vec3::splat(scale))
        * Mat4::from_translation(-center);
    let normal_transform = transform.inverse().transpose();

    for v in vertices.iter_mut() {
        let p = transform.transform_point3(Vec3::from(v.position));
        let n = normal_transform.transform_vector3(Vec3::from(v.normal)).normalize_or_zero();
        v.position = p.into();
        v.normal = n.into();
    }
}
