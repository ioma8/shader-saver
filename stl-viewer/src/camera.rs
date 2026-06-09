use glam::{Mat4, Vec3};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub model: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
}

pub struct Camera {
    pub rotation_x: f32,
    pub rotation_y: f32,
    pub distance: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            rotation_x: 0.0,
            rotation_y: std::f32::consts::PI + 0.5,
            distance: 3.0,
        }
    }
}

impl Camera {
    pub fn uniforms(&self, aspect: f32) -> Uniforms {
        let model = Mat4::from_rotation_x(self.rotation_x) * Mat4::from_rotation_y(self.rotation_y);
        let eye = Vec3::new(0.0, 0.0, -self.distance);
        let view = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_3, aspect, 0.01, 100.0);
        Uniforms {
            model: model.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            camera_pos: [eye.x, eye.y, eye.z, 0.0],
        }
    }
}
