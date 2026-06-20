//! Cameras: orbit + fly controllers, input key state, and the GPU camera uniform.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use crate::CO_VERSION;

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum CameraMode {
    Orbit,
    Fly,
}

pub(crate) struct OrbitCamera {
    pub(crate) target: Vec3,
    pub(crate) distance: f32,
    pub(crate) yaw: f32,
    pub(crate) pitch: f32,
    pub(crate) fovy_radians: f32,
    pub(crate) znear: f32,
    pub(crate) zfar: f32,
}

impl OrbitCamera {
    pub(crate) fn eye(&self) -> Vec3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let offset = Vec3::new(cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw);
        self.target + offset * self.distance
    }

    pub(crate) fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye(), self.target, Vec3::Y);
        let proj = Mat4::perspective_rh(self.fovy_radians, aspect, self.znear, self.zfar);
        proj * view
    }

    pub(crate) fn orbit(&mut self, dx: f32, dy: f32) {
        const SENSITIVITY: f32 = 0.005;
        self.yaw -= dx * SENSITIVITY;
        self.pitch -= dy * SENSITIVITY;
        let limit = std::f32::consts::FRAC_PI_2 - 0.01;
        self.pitch = self.pitch.clamp(-limit, limit);
    }

    pub(crate) fn pan(&mut self, dx: f32, dy: f32) {
        let forward = (self.target - self.eye()).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward).normalize();
        let speed = self.distance * 0.0015;
        self.target += (-right * dx + up * dy) * speed;
    }

    pub(crate) fn zoom(&mut self, scroll: f32) {
        let factor = (1.0 - scroll * 0.1).clamp(0.5, 1.5);
        self.distance = (self.distance * factor).clamp(1.0, 80.0);
    }
}

pub(crate) struct FlyCamera {
    pub(crate) position: Vec3,
    pub(crate) yaw: f32,
    pub(crate) pitch: f32,
    pub(crate) fovy_radians: f32,
    pub(crate) znear: f32,
    pub(crate) zfar: f32,
    pub(crate) speed: f32,
}

impl FlyCamera {
    pub(crate) fn forward(&self) -> Vec3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        Vec3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw)
    }

    pub(crate) fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = Mat4::look_at_rh(self.position, self.position + self.forward(), Vec3::Y);
        let proj = Mat4::perspective_rh(self.fovy_radians, aspect, self.znear, self.zfar);
        proj * view
    }

    pub(crate) fn look(&mut self, dx: f32, dy: f32) {
        const SENSITIVITY: f32 = 0.004;
        self.yaw += dx * SENSITIVITY;
        self.pitch -= dy * SENSITIVITY;
        let limit = std::f32::consts::FRAC_PI_2 - 0.01;
        self.pitch = self.pitch.clamp(-limit, limit);
    }
}

#[derive(Default)]
pub(crate) struct Keys {
    pub(crate) forward: bool,
    pub(crate) back: bool,
    pub(crate) left: bool,
    pub(crate) right: bool,
    pub(crate) up: bool,
    pub(crate) down: bool,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct CameraUniform {
    pub(crate) view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub(crate) fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
        }
    }
}

pub(crate) fn title_for(mode: CameraMode) -> String {
    let name = match mode {
        CameraMode::Orbit => "Orbit",
        CameraMode::Fly => "Fly",
    };
    format!("CoEngine v{CO_VERSION}   [{name}]")
}
