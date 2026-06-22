//! Geometry: vertices, cubes, the grid, and ray/AABB picking math.

use bytemuck::{Pod, Zeroable};
use glam::{EulerRot, Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

pub(crate) const CUBE_HALF: f32 = 0.5;
pub(crate) const CUBE_BASE_COLOR: [f32; 3] = [0.85, 0.55, 0.25];

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct Vertex {
    pub(crate) position: [f32; 3],
    pub(crate) color: [f32; 3],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

    pub(crate) fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

/// The shape an entity renders as. More primitives (and loaded meshes / 2D
/// sprites) slot in here later without touching the model.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub(crate) enum ShapeKind {
    #[default]
    Cube,
    Sphere,
}

impl ShapeKind {
    /// Human label (also used to name new entities: "Cube 3", "Sphere 4").
    pub(crate) fn label(self) -> &'static str {
        match self {
            ShapeKind::Cube => "Cube",
            ShapeKind::Sphere => "Sphere",
        }
    }
}

fn vec3_one() -> Vec3 {
    Vec3::ONE
}

/// A scene object: a stable id + name + a full transform (position, rotation in
/// euler degrees, scale) + color + shape. Dimension-agnostic — a future 2D scene
/// reuses this same model (a sprite is just another `ShapeKind` with z fixed).
///
/// `pos`/`color` keep their old field names so pre-v0.0.16 `project.json` files
/// (which serialized cubes as `{pos, color}`) still load; the new fields default.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct Entity {
    #[serde(default)]
    pub(crate) id: u32,
    #[serde(default)]
    pub(crate) name: String,
    pub(crate) pos: Vec3,
    #[serde(default)]
    pub(crate) rotation: Vec3,
    #[serde(default = "vec3_one")]
    pub(crate) scale: Vec3,
    pub(crate) color: [f32; 3],
    #[serde(default)]
    pub(crate) kind: ShapeKind,
}

impl Entity {
    /// The world transform (T · R · S) used to place this entity's geometry.
    pub(crate) fn model_matrix(&self) -> Mat4 {
        let r = self.rotation * (std::f32::consts::PI / 180.0);
        Mat4::from_scale_rotation_translation(
            self.scale,
            Quat::from_euler(EulerRot::XYZ, r.x, r.y, r.z),
            self.pos,
        )
    }
}

/// Half-grid placement: the Nth cube's (x, z) on the ground grid.
pub(crate) fn grid_slot(n: usize) -> (f32, f32) {
    let cols = 7;
    let col = (n % cols) as f32 - 3.0;
    let row = (n / cols) as f32 - 3.0;
    (col * 1.5, row * 1.5)
}

/// Parse a color name or "#rrggbb" hex into RGB. Falls back to the default cube color.
pub(crate) fn parse_color(name: &str) -> [f32; 3] {
    let n = name.trim().to_lowercase();
    if let Some(hex) = n.strip_prefix('#') {
        if hex.len() == 6 {
            if let (Ok(r), Ok(g), Ok(b)) = (
                u8::from_str_radix(&hex[0..2], 16),
                u8::from_str_radix(&hex[2..4], 16),
                u8::from_str_radix(&hex[4..6], 16),
            ) {
                return [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0];
            }
        }
    }
    match n.as_str() {
        "red" => [0.85, 0.20, 0.20],
        "green" => [0.20, 0.70, 0.25],
        "blue" => [0.25, 0.45, 0.95],
        "cobalt" => [0.22, 0.45, 0.82],
        "yellow" => [0.92, 0.85, 0.25],
        "orange" => [0.90, 0.55, 0.20],
        "purple" | "violet" => [0.55, 0.30, 0.75],
        "magenta" | "pink" => [0.90, 0.35, 0.65],
        "cyan" | "teal" => [0.25, 0.80, 0.85],
        "white" => [0.92, 0.92, 0.92],
        "black" => [0.10, 0.10, 0.10],
        "gray" | "grey" => [0.50, 0.50, 0.50],
        "brown" => [0.50, 0.33, 0.18],
        _ => CUBE_BASE_COLOR,
    }
}

pub(crate) fn build_grid(half: i32, spacing: f32) -> Vec<Vertex> {
    let mut verts = Vec::new();
    let max = half as f32 * spacing;

    let gray = [0.18, 0.20, 0.24]; // quieter grid
    let x_axis = [0.90, 0.32, 0.28]; // clearer red X axis
    let z_axis = [0.30, 0.58, 1.00]; // clearer blue Z axis

    for i in -half..=half {
        let p = i as f32 * spacing;

        let c = if i == 0 { z_axis } else { gray };
        verts.push(Vertex { position: [p, 0.0, -max], color: c });
        verts.push(Vertex { position: [p, 0.0, max], color: c });

        let c = if i == 0 { x_axis } else { gray };
        verts.push(Vertex { position: [-max, 0.0, p], color: c });
        verts.push(Vertex { position: [max, 0.0, p], color: c });
    }

    verts
}

pub(crate) fn push_quad(out: &mut Vec<Vertex>, a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3], color: [f32; 3]) {
    for pos in [a, b, c, a, c, d] {
        out.push(Vertex { position: pos, color });
    }
}

pub(crate) fn unit_cube() -> Vec<Vertex> {
    let s = CUBE_HALF;

    // Per-face brightness (grayscale); multiplied by each cube's color at build time.
    let top = [1.00, 1.00, 1.00];
    let bottom = [0.48, 0.48, 0.48];
    let front = [0.86, 0.86, 0.86];
    let back = [0.64, 0.64, 0.64];
    let right = [0.76, 0.76, 0.76];
    let left = [0.70, 0.70, 0.70];

    let p000 = [-s, -s, -s];
    let p001 = [-s, -s, s];
    let p010 = [-s, s, -s];
    let p011 = [-s, s, s];
    let p100 = [s, -s, -s];
    let p101 = [s, -s, s];
    let p110 = [s, s, -s];
    let p111 = [s, s, s];

    let mut v = Vec::with_capacity(36);
    push_quad(&mut v, p010, p011, p111, p110, top);
    push_quad(&mut v, p000, p100, p101, p001, bottom);
    push_quad(&mut v, p001, p101, p111, p011, front);
    push_quad(&mut v, p100, p000, p010, p110, back);
    push_quad(&mut v, p101, p100, p110, p111, right);
    push_quad(&mut v, p000, p001, p011, p010, left);
    v
}

/// A unit sphere (radius `CUBE_HALF`, so its diameter matches the cube) as a
/// triangle list. Per-vertex grayscale shade lives in `color` (like `unit_cube`'s
/// per-face brightness) so `build_scene_vertices` can tint it by the entity color.
pub(crate) fn unit_sphere() -> Vec<Vertex> {
    let r = CUBE_HALF;
    let stacks = 14usize; // latitude bands
    let sectors = 20usize; // longitude segments
    let pi = std::f32::consts::PI;
    let light = Vec3::new(0.35, 1.0, 0.45).normalize();

    // Unit normal (also the position direction) at a stack/sector grid point.
    let dir = |st: usize, se: usize| -> Vec3 {
        let phi = pi * (st as f32 / stacks as f32); // 0..pi  (north→south)
        let theta = 2.0 * pi * (se as f32 / sectors as f32); // 0..2pi (around)
        Vec3::new(phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin())
    };
    // Soft top-down shade so the sphere reads as 3D (ambient floor + diffuse).
    let vert = |n: Vec3| -> Vertex {
        let s = 0.45 + 0.55 * n.dot(light).clamp(0.0, 1.0);
        Vertex {
            position: [n.x * r, n.y * r, n.z * r],
            color: [s, s, s],
        }
    };

    let mut v = Vec::with_capacity(stacks * sectors * 6);
    for st in 0..stacks {
        for se in 0..sectors {
            let a = dir(st, se);
            let b = dir(st + 1, se);
            let c = dir(st + 1, se + 1);
            let d = dir(st, se + 1);
            for n in [a, b, c, a, c, d] {
                v.push(vert(n));
            }
        }
    }
    v
}

/// Build the scene's triangle vertices: each entity's shape (cube or sphere)
/// transformed by its model matrix (position/rotation/scale) and tinted by its
/// color, with the selected entity glowing cobalt.
pub(crate) fn build_scene_vertices(entities: &[Entity], selected: Option<usize>) -> Vec<Vertex> {
    let cube = unit_cube();
    let sphere = unit_sphere();
    let mut out = Vec::new();

    for (i, e) in entities.iter().enumerate() {
        let highlight = Some(i) == selected;
        let model = e.model_matrix();
        let base = match e.kind {
            ShapeKind::Cube => &cube,
            ShapeKind::Sphere => &sphere,
        };
        for v in base {
            let shade = v.color[0]; // per-face grayscale brightness
            let color = if highlight {
                // The selected entity glows cobalt (the engine identity color).
                [shade * 0.25, shade * 0.55, shade * 1.00]
            } else {
                [shade * e.color[0], shade * e.color[1], shade * e.color[2]]
            };
            let p = model.transform_point3(Vec3::from(v.position));
            out.push(Vertex {
                position: [p.x, p.y, p.z],
                color,
            });
        }
    }
    out
}

pub(crate) fn ray_aabb(origin: Vec3, dir: Vec3, center: Vec3, half: f32) -> Option<f32> {
    let min = center - Vec3::splat(half);
    let max = center + Vec3::splat(half);
    let inv = Vec3::ONE / dir;
    let t0 = (min - origin) * inv;
    let t1 = (max - origin) * inv;
    let t_enter = t0.min(t1).max_element();
    let t_exit = t0.max(t1).min_element();
    if t_enter <= t_exit && t_exit >= 0.0 {
        Some(if t_enter >= 0.0 { t_enter } else { t_exit })
    } else {
        None
    }
}
