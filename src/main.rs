//! CoEngine — Milestone 1, Step 5: "Chat Panel + Version Watermark"
//!
//! Builds on Step 4 (select & remove). New in this step:
//!   * **egui** is integrated as an on-screen UI layer drawn over the 3D scene,
//!   * a **chat panel** docked on the right: a scrolling message history plus a
//!     text box + Send/Clear. For now it **echoes locally** (no network) — it is
//!     wired to Claude in Step 6,
//!   * the **bottom-left version watermark** ("CoEngine v0.0.5"), read from the
//!     crate version.
//!
//! egui captures input over its own panels, so clicking/typing in the chat does
//! not move the camera or spawn cubes.
//!
//! Controls (when not interacting with the chat):
//!   Orbit: left-drag orbit · right-drag pan · scroll zoom · click select · C add · Del remove · Tab Fly
//!   Fly:   WASD move · E/Q up/down · right-drag look · click select · C add · Del remove · Tab Orbit
//!   Esc quits.

use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{
    DeviceEvent, DeviceId, ElementState, MouseButton, MouseScrollDelta, WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const CUBE_HALF: f32 = 0.5;
const VERSION: &str = env!("CARGO_PKG_VERSION");

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

fn build_grid(half: i32, spacing: f32) -> Vec<Vertex> {
    let mut verts = Vec::new();
    let max = half as f32 * spacing;

    let gray = [0.32, 0.32, 0.34];
    let x_axis = [0.85, 0.25, 0.25];
    let z_axis = [0.25, 0.45, 0.90];

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

fn push_quad(out: &mut Vec<Vertex>, a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3], color: [f32; 3]) {
    for pos in [a, b, c, a, c, d] {
        out.push(Vertex { position: pos, color });
    }
}

fn unit_cube() -> Vec<Vertex> {
    let s = CUBE_HALF;

    let top = [0.92, 0.64, 0.32];
    let bottom = [0.42, 0.27, 0.13];
    let front = [0.84, 0.55, 0.27];
    let back = [0.64, 0.41, 0.20];
    let right = [0.76, 0.49, 0.24];
    let left = [0.70, 0.45, 0.22];

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

fn build_cube_vertices(positions: &[Vec3], selected: Option<usize>) -> Vec<Vertex> {
    let base = unit_cube();
    let mut out = Vec::with_capacity(positions.len() * base.len());

    for (i, p) in positions.iter().enumerate() {
        let highlight = Some(i) == selected;
        for v in &base {
            let color = if highlight {
                [
                    v.color[0] * 0.4 + 1.00 * 0.6,
                    v.color[1] * 0.4 + 0.95 * 0.6,
                    v.color[2] * 0.4 + 0.30 * 0.6,
                ]
            } else {
                v.color
            };
            out.push(Vertex {
                position: [v.position[0] + p.x, v.position[1] + p.y, v.position[2] + p.z],
                color,
            });
        }
    }
    out
}

fn ray_aabb(origin: Vec3, dir: Vec3, center: Vec3, half: f32) -> Option<f32> {
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

// ---------------------------------------------------------------------------
// Chat (UI state)
// ---------------------------------------------------------------------------

struct ChatMessage {
    role: String,
    text: String,
}

#[derive(Default)]
struct ChatUi {
    messages: Vec<ChatMessage>,
    input: String,
}

/// Local-echo send (Step 5). Step 6 replaces the stub reply with a real Claude call.
fn send_message(chat: &mut ChatUi) {
    let text = chat.input.trim().to_string();
    chat.input.clear();
    if text.is_empty() {
        return;
    }
    chat.messages.push(ChatMessage { role: "You".to_string(), text: text.clone() });
    chat.messages.push(ChatMessage {
        role: "Claude (stub)".to_string(),
        text: format!("(local echo) {text}"),
    });
}

/// Build the egui UI for this frame: bottom-left watermark + right-side chat panel.
fn build_chat_ui(ctx: &egui::Context, chat: &mut ChatUi) {
    // Version watermark, anchored to the bottom-left, non-interactive.
    egui::Area::new(egui::Id::new("version_watermark"))
        .anchor(egui::Align2::LEFT_BOTTOM, egui::vec2(10.0, -8.0))
        .interactable(false)
        .show(ctx, |ui| {
            ui.label(
                egui::RichText::new(format!("CoEngine v{VERSION}"))
                    .monospace()
                    .color(egui::Color32::from_white_alpha(150)),
            );
        });

    // Chat panel docked on the right edge.
    egui::SidePanel::right("chat_panel")
        .resizable(true)
        .default_width(320.0)
        .show(ctx, |ui| {
            egui::TopBottomPanel::top("chat_header").show_inside(ui, |ui| {
                ui.add_space(6.0);
                ui.heading("Chat");
                ui.label(
                    egui::RichText::new("Local echo — wired to Claude in Step 6")
                        .weak()
                        .small(),
                );
                ui.add_space(4.0);
            });

            egui::TopBottomPanel::bottom("chat_input").show_inside(ui, |ui| {
                ui.add_space(6.0);
                let resp = ui.add(
                    egui::TextEdit::singleline(&mut chat.input)
                        .hint_text("Type a message…")
                        .desired_width(f32::INFINITY),
                );
                let enter_pressed =
                    resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                ui.horizontal(|ui| {
                    let send_clicked = ui.button("Send").clicked();
                    if ui.button("Clear").clicked() {
                        chat.messages.clear();
                    }
                    if send_clicked || enter_pressed {
                        send_message(chat);
                        resp.request_focus();
                    }
                });
                ui.add_space(6.0);
            });

            egui::CentralPanel::default().show_inside(ui, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for m in &chat.messages {
                            ui.label(egui::RichText::new(format!("{}:", m.role)).strong());
                            ui.label(&m.text);
                            ui.add_space(6.0);
                        }
                    });
            });
        });
}

// ---------------------------------------------------------------------------
// Cameras
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum CameraMode {
    Orbit,
    Fly,
}

struct OrbitCamera {
    target: Vec3,
    distance: f32,
    yaw: f32,
    pitch: f32,
    fovy_radians: f32,
    znear: f32,
    zfar: f32,
}

impl OrbitCamera {
    fn eye(&self) -> Vec3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let offset = Vec3::new(cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw);
        self.target + offset * self.distance
    }

    fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye(), self.target, Vec3::Y);
        let proj = Mat4::perspective_rh(self.fovy_radians, aspect, self.znear, self.zfar);
        proj * view
    }

    fn orbit(&mut self, dx: f32, dy: f32) {
        const SENSITIVITY: f32 = 0.005;
        self.yaw -= dx * SENSITIVITY;
        self.pitch -= dy * SENSITIVITY;
        let limit = std::f32::consts::FRAC_PI_2 - 0.01;
        self.pitch = self.pitch.clamp(-limit, limit);
    }

    fn pan(&mut self, dx: f32, dy: f32) {
        let forward = (self.target - self.eye()).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward).normalize();
        let speed = self.distance * 0.0015;
        self.target += (-right * dx + up * dy) * speed;
    }

    fn zoom(&mut self, scroll: f32) {
        let factor = (1.0 - scroll * 0.1).clamp(0.5, 1.5);
        self.distance = (self.distance * factor).clamp(1.0, 80.0);
    }
}

struct FlyCamera {
    position: Vec3,
    yaw: f32,
    pitch: f32,
    fovy_radians: f32,
    znear: f32,
    zfar: f32,
    speed: f32,
}

impl FlyCamera {
    fn forward(&self) -> Vec3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        Vec3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw)
    }

    fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = Mat4::look_at_rh(self.position, self.position + self.forward(), Vec3::Y);
        let proj = Mat4::perspective_rh(self.fovy_radians, aspect, self.znear, self.zfar);
        proj * view
    }

    fn look(&mut self, dx: f32, dy: f32) {
        const SENSITIVITY: f32 = 0.004;
        self.yaw += dx * SENSITIVITY;
        self.pitch -= dy * SENSITIVITY;
        let limit = std::f32::consts::FRAC_PI_2 - 0.01;
        self.pitch = self.pitch.clamp(-limit, limit);
    }
}

#[derive(Default)]
struct Keys {
    forward: bool,
    back: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
        }
    }
}

fn title_for(mode: CameraMode) -> &'static str {
    match mode {
        CameraMode::Orbit => concat!(
            "CoEngine v",
            env!("CARGO_PKG_VERSION"),
            "   [Orbit]   drag orbit · R-drag pan · scroll zoom · click=select · C=add · Del=remove · Tab=Fly"
        ),
        CameraMode::Fly => concat!(
            "CoEngine v",
            env!("CARGO_PKG_VERSION"),
            "   [Fly]   WASD · E/Q up/down · R-drag look · click=select · C=add · Del=remove · Tab=Orbit"
        ),
    }
}

// ---------------------------------------------------------------------------
// Renderer state
// ---------------------------------------------------------------------------

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    grid_pipeline: wgpu::RenderPipeline,
    cube_pipeline: wgpu::RenderPipeline,
    depth_view: wgpu::TextureView,

    grid_buffer: wgpu::Buffer,
    grid_vertex_count: u32,

    cubes: Vec<Vec3>,
    cube_buffer: Option<wgpu::Buffer>,
    cube_vertex_count: u32,
    selected: Option<usize>,
    spawn_count: u32,

    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    orbit: OrbitCamera,
    fly: FlyCamera,
    mode: CameraMode,

    keys: Keys,
    mouse_left_down: bool,
    mouse_right_down: bool,
    cursor_pos: (f32, f32),
    left_drag_dist: f32,
    last_frame: Instant,

    // egui (UI layer).
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    chat: ChatUi,
}

impl State {
    fn new(window: Arc<Window>) -> State {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create a surface for the window");
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            },
        ))
        .expect("no compatible GPU adapter was found");
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("CoEngine Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .expect("failed to create the GPU device");
        let config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("this surface is not supported by the adapter");
        surface.configure(&device, &config);

        let depth_view = create_depth_view(&device, &config);

        let orbit = OrbitCamera {
            target: Vec3::ZERO,
            distance: 14.0,
            yaw: std::f32::consts::FRAC_PI_4,
            pitch: 0.5,
            fovy_radians: 45.0_f32.to_radians(),
            znear: 0.1,
            zfar: 100.0,
        };
        let fly = FlyCamera {
            position: Vec3::new(8.0, 6.0, 8.0),
            yaw: 0.0,
            pitch: 0.0,
            fovy_radians: 45.0_f32.to_radians(),
            znear: 0.1,
            zfar: 100.0,
            speed: 8.0,
        };
        let mode = CameraMode::Orbit;

        let aspect = config.width.max(1) as f32 / config.height.max(1) as f32;
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.view_proj = orbit.view_proj(aspect).to_cols_array_2d();

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Uniform Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Camera Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Grid/Cube Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("grid.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        let depth_stencil = wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };
        let color_target = wgpu::ColorTargetState {
            format: config.format,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        };

        let grid_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Grid Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(color_target.clone())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(depth_stencil.clone()),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let cube_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Cube Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(color_target)],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(depth_stencil),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let grid = build_grid(10, 1.0);
        let grid_vertex_count = grid.len() as u32;
        let grid_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Vertex Buffer"),
            contents: bytemuck::cast_slice(&grid),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // egui setup.
        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(&device, config.format, None, 1, false);

        State {
            window,
            surface,
            device,
            queue,
            config,
            grid_pipeline,
            cube_pipeline,
            depth_view,
            grid_buffer,
            grid_vertex_count,
            cubes: Vec::new(),
            cube_buffer: None,
            cube_vertex_count: 0,
            selected: None,
            spawn_count: 0,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            orbit,
            fly,
            mode,
            keys: Keys::default(),
            mouse_left_down: false,
            mouse_right_down: false,
            cursor_pos: (0.0, 0.0),
            left_drag_dist: 0.0,
            last_frame: Instant::now(),
            egui_ctx,
            egui_state,
            egui_renderer,
            chat: ChatUi::default(),
        }
    }

    fn aspect(&self) -> f32 {
        self.config.width.max(1) as f32 / self.config.height.max(1) as f32
    }

    fn current_view_proj(&self) -> Mat4 {
        match self.mode {
            CameraMode::Orbit => self.orbit.view_proj(self.aspect()),
            CameraMode::Fly => self.fly.view_proj(self.aspect()),
        }
    }

    fn update_camera(&mut self) {
        self.camera_uniform.view_proj = self.current_view_proj().to_cols_array_2d();
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }

    fn add_cube(&mut self) {
        let n = self.spawn_count;
        let cols = 7;
        let col = (n % cols) as f32 - 3.0;
        let row = (n / cols) as f32 - 3.0;
        self.cubes.push(Vec3::new(col * 1.5, CUBE_HALF, row * 1.5));
        self.spawn_count += 1;
        self.rebuild_cubes();
        println!("Cubes in scene: {}", self.cubes.len());
    }

    fn rebuild_cubes(&mut self) {
        if self.cubes.is_empty() {
            self.cube_buffer = None;
            self.cube_vertex_count = 0;
            return;
        }
        let verts = build_cube_vertices(&self.cubes, self.selected);
        self.cube_vertex_count = verts.len() as u32;
        self.cube_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Cube Vertex Buffer"),
                contents: bytemuck::cast_slice(&verts),
                usage: wgpu::BufferUsages::VERTEX,
            },
        ));
    }

    fn pick(&mut self, cursor: (f32, f32)) {
        let w = self.config.width.max(1) as f32;
        let h = self.config.height.max(1) as f32;

        let ndc_x = (cursor.0 / w) * 2.0 - 1.0;
        let ndc_y = 1.0 - (cursor.1 / h) * 2.0;

        let inv = self.current_view_proj().inverse();
        let near = inv * Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
        let far = inv * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
        let near = near.truncate() / near.w;
        let far = far.truncate() / far.w;

        let origin = near;
        let dir = (far - near).normalize();

        let mut best: Option<(usize, f32)> = None;
        for (i, c) in self.cubes.iter().enumerate() {
            if let Some(t) = ray_aabb(origin, dir, *c, CUBE_HALF) {
                if best.map_or(true, |(_, bt)| t < bt) {
                    best = Some((i, t));
                }
            }
        }

        self.selected = best.map(|(i, _)| i);
        self.rebuild_cubes();
        match self.selected {
            Some(i) => println!("Selected cube #{i}"),
            None => println!("Selection cleared"),
        }
    }

    fn remove_selected(&mut self) {
        if let Some(i) = self.selected {
            if i < self.cubes.len() {
                self.cubes.remove(i);
            }
            self.selected = None;
            self.rebuild_cubes();
            println!("Removed cube; {} remain", self.cubes.len());
        }
    }

    fn toggle_mode(&mut self) {
        match self.mode {
            CameraMode::Orbit => {
                let eye = self.orbit.eye();
                let dir = (self.orbit.target - eye).normalize_or_zero();
                self.fly.position = eye;
                self.fly.pitch = dir.y.clamp(-1.0, 1.0).asin();
                self.fly.yaw = dir.z.atan2(dir.x);
                self.mode = CameraMode::Fly;
            }
            CameraMode::Fly => {
                let f = self.fly.forward();
                self.orbit.target = self.fly.position + f * self.orbit.distance;
                let d = -f;
                self.orbit.pitch = d.y.clamp(-1.0, 1.0).asin();
                self.orbit.yaw = d.x.atan2(d.z);
                self.mode = CameraMode::Orbit;
            }
        }
        self.window.set_title(title_for(self.mode));
        self.update_camera();
    }

    fn update(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        if self.mode == CameraMode::Fly {
            let f = self.fly.forward();
            let right = f.cross(Vec3::Y).normalize_or_zero();

            let mut v = Vec3::ZERO;
            if self.keys.forward {
                v += f;
            }
            if self.keys.back {
                v -= f;
            }
            if self.keys.right {
                v += right;
            }
            if self.keys.left {
                v -= right;
            }
            if self.keys.up {
                v += Vec3::Y;
            }
            if self.keys.down {
                v -= Vec3::Y;
            }

            if v.length_squared() > 0.0 {
                self.fly.position += v.normalize() * self.fly.speed * dt;
                self.update_camera();
            }
        }
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_view = create_depth_view(&self.device, &self.config);
            self.update_camera();
        }
    }

    fn render(&mut self) {
        let frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                eprintln!("GPU out of memory — stopping render");
                return;
            }
            Err(wgpu::SurfaceError::Timeout) => return,
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("CoEngine Command Encoder"),
            });

        // --- Pass 1: the 3D scene (grid + cubes) ---
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Scene Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.10,
                            g: 0.20,
                            b: 0.30,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_bind_group(0, &self.camera_bind_group, &[]);

            pass.set_pipeline(&self.grid_pipeline);
            pass.set_vertex_buffer(0, self.grid_buffer.slice(..));
            pass.draw(0..self.grid_vertex_count, 0..1);

            if let Some(cube_buffer) = &self.cube_buffer {
                pass.set_pipeline(&self.cube_pipeline);
                pass.set_vertex_buffer(0, cube_buffer.slice(..));
                pass.draw(0..self.cube_vertex_count, 0..1);
            }
        }

        // --- egui: run the UI, then draw it over the scene (Pass 2) ---
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let ctx = self.egui_ctx.clone();
        let chat = &mut self.chat;
        let full_output = ctx.run(raw_input, |c| build_chat_ui(c, chat));
        self.egui_state
            .handle_platform_output(&self.window, full_output.platform_output);

        let paint_jobs = ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: full_output.pixels_per_point,
        };

        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, image_delta);
        }
        let user_cmd_bufs =
            self.egui_renderer
                .update_buffers(&self.device, &self.queue, &mut encoder, &paint_jobs, &screen);

        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // keep the scene underneath
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            // egui-wgpu wants a 'static render pass.
            let mut pass = pass.forget_lifetime();
            self.egui_renderer.render(&mut pass, &paint_jobs, &screen);
        }

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.queue
            .submit(user_cmd_bufs.into_iter().chain(std::iter::once(encoder.finish())));
        frame.present();
    }
}

fn create_depth_view(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth Texture"),
        size: wgpu::Extent3d {
            width: config.width.max(1),
            height: config.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

// ---------------------------------------------------------------------------
// Application / event loop
// ---------------------------------------------------------------------------

#[derive(Default)]
struct App {
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let attributes = Window::default_attributes()
            .with_title(title_for(CameraMode::Orbit))
            .with_inner_size(LogicalSize::new(1280.0, 720.0));

        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .expect("failed to create the window"),
        );

        self.state = Some(State::new(window));

        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        // Window-management events: always handled.
        match &event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
                return;
            }
            WindowEvent::Resized(new_size) => {
                state.resize(*new_size);
                return;
            }
            WindowEvent::RedrawRequested => {
                state.update();
                state.render();
                return;
            }
            _ => {}
        }

        // Tab toggles the camera mode. Handle it ourselves and swallow it so egui
        // does NOT use Tab to move keyboard focus into the chat box (which would
        // make WASD type into the chat instead of flying the camera).
        if let WindowEvent::KeyboardInput { event: ke, .. } = &event {
            if ke.physical_key == PhysicalKey::Code(KeyCode::Tab) {
                if ke.state == ElementState::Pressed && !ke.repeat {
                    state.toggle_mode();
                }
                return;
            }
        }

        // Let egui consume input over its panels (typing, clicking Send, etc.).
        let egui_consumed = state
            .egui_state
            .on_window_event(&state.window, &event)
            .consumed;
        if egui_consumed {
            return;
        }

        // Input that drives the 3D viewport.
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                state.cursor_pos = (position.x as f32, position.y as f32);
            }

            WindowEvent::KeyboardInput {
                event: key_event, ..
            } => {
                if let PhysicalKey::Code(code) = key_event.physical_key {
                    let pressed = key_event.state == ElementState::Pressed;
                    let first_press = pressed && !key_event.repeat;
                    match code {
                        KeyCode::KeyW => state.keys.forward = pressed,
                        KeyCode::KeyS => state.keys.back = pressed,
                        KeyCode::KeyA => state.keys.left = pressed,
                        KeyCode::KeyD => state.keys.right = pressed,
                        KeyCode::KeyE | KeyCode::Space => state.keys.up = pressed,
                        KeyCode::KeyQ => state.keys.down = pressed,
                        KeyCode::KeyC if first_press => state.add_cube(),
                        KeyCode::Delete | KeyCode::Backspace if first_press => {
                            state.remove_selected()
                        }
                        // Tab is handled earlier (before egui) — see window_event top.
                        KeyCode::Escape => event_loop.exit(),
                        _ => {}
                    }
                }
            }

            WindowEvent::MouseInput {
                state: btn_state,
                button,
                ..
            } => {
                let pressed = btn_state == ElementState::Pressed;
                match button {
                    MouseButton::Left => {
                        state.mouse_left_down = pressed;
                        if pressed {
                            state.left_drag_dist = 0.0;
                        } else if state.left_drag_dist < 5.0 {
                            let cursor = state.cursor_pos;
                            state.pick(cursor);
                        }
                    }
                    MouseButton::Right => state.mouse_right_down = pressed,
                    _ => {}
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                if state.mode == CameraMode::Orbit {
                    let scroll = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32 / 120.0,
                    };
                    state.orbit.zoom(scroll);
                    state.update_camera();
                }
            }

            WindowEvent::Focused(false) => state.keys = Keys::default(),

            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            let (dx, dy) = (dx as f32, dy as f32);

            if state.mouse_left_down {
                state.left_drag_dist += dx.abs() + dy.abs();
            }

            match state.mode {
                CameraMode::Orbit => {
                    if state.mouse_left_down {
                        state.orbit.orbit(dx, dy);
                        state.update_camera();
                    } else if state.mouse_right_down {
                        state.orbit.pan(dx, dy);
                        state.update_camera();
                    }
                }
                CameraMode::Fly => {
                    if state.mouse_right_down {
                        state.fly.look(dx, dy);
                        state.update_camera();
                    }
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("failed to create the event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop
        .run_app(&mut app)
        .expect("the event loop exited with an error");
}
