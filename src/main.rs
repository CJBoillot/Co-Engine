//! CoEngine — Milestone 1, Step 2b: "Free-fly Camera + Mode Toggle"
//!
//! Builds on Step 2a (orbit camera). New in this step:
//!   * a **free-fly camera** (FPS-style): WASD to move, E/Q (or Space) up/down,
//!     right-drag to look around,
//!   * **Tab** toggles between Orbit and Fly modes (the two are kept in sync so
//!     the view doesn't jump),
//!   * a real **per-frame update loop** with delta-time so movement is smooth
//!     while keys are held (the engine now renders continuously).
//!
//! Controls:
//!   Orbit mode: left-drag orbit · right-drag pan · scroll zoom · Tab -> Fly
//!   Fly mode:   WASD move · E/Q (or Space) up/down · right-drag look · Tab -> Orbit
//!   Esc quits.

use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
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
    let x_axis = [0.85, 0.25, 0.25]; // red
    let z_axis = [0.25, 0.45, 0.90]; // blue

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

// ---------------------------------------------------------------------------
// Cameras
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum CameraMode {
    Orbit,
    Fly,
}

/// Editor-style camera that orbits a focus point.
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

/// First-person "free-fly" camera: a position plus a look direction (yaw/pitch).
struct FlyCamera {
    position: Vec3,
    yaw: f32,
    pitch: f32,
    fovy_radians: f32,
    znear: f32,
    zfar: f32,
    speed: f32, // movement speed in world units per second
}

impl FlyCamera {
    /// Unit vector the camera is looking along.
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

    /// Right-drag look: rotate the view by mouse-motion pixels.
    fn look(&mut self, dx: f32, dy: f32) {
        const SENSITIVITY: f32 = 0.004;
        self.yaw += dx * SENSITIVITY;
        self.pitch -= dy * SENSITIVITY;
        let limit = std::f32::consts::FRAC_PI_2 - 0.01;
        self.pitch = self.pitch.clamp(-limit, limit);
    }
}

/// Which movement keys are currently held (for the fly camera).
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
        CameraMode::Orbit => {
            "CoEngine — Step 2b  [Orbit]   L-drag orbit · R-drag pan · scroll zoom · Tab=Fly · Esc=quit"
        }
        CameraMode::Fly => {
            "CoEngine — Step 2b  [Fly]   WASD move · E/Q up/down · R-drag look · Tab=Orbit · Esc=quit"
        }
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

    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
    depth_view: wgpu::TextureView,

    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    // Two cameras; `mode` picks which one is active.
    orbit: OrbitCamera,
    fly: FlyCamera,
    mode: CameraMode,

    // Input state.
    keys: Keys,
    mouse_left_down: bool,
    mouse_right_down: bool,

    // Timing for smooth, frame-rate-independent movement.
    last_frame: Instant,
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

        // Orbit camera: starting 3/4 view of the grid.
        let orbit = OrbitCamera {
            target: Vec3::ZERO,
            distance: 14.0,
            yaw: std::f32::consts::FRAC_PI_4,
            pitch: 0.5,
            fovy_radians: 45.0_f32.to_radians(),
            znear: 0.1,
            zfar: 100.0,
        };

        // Fly camera placeholder; the first Tab toggle syncs it to the orbit view.
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
            label: Some("Grid Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("grid.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Grid Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let grid = build_grid(10, 1.0);
        let vertex_count = grid.len() as u32;
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Vertex Buffer"),
            contents: bytemuck::cast_slice(&grid),
            usage: wgpu::BufferUsages::VERTEX,
        });

        State {
            window,
            surface,
            device,
            queue,
            config,
            pipeline,
            vertex_buffer,
            vertex_count,
            depth_view,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            orbit,
            fly,
            mode,
            keys: Keys::default(),
            mouse_left_down: false,
            mouse_right_down: false,
            last_frame: Instant::now(),
        }
    }

    fn aspect(&self) -> f32 {
        self.config.width.max(1) as f32 / self.config.height.max(1) as f32
    }

    /// View*projection matrix for whichever camera is active.
    fn current_view_proj(&self) -> Mat4 {
        match self.mode {
            CameraMode::Orbit => self.orbit.view_proj(self.aspect()),
            CameraMode::Fly => self.fly.view_proj(self.aspect()),
        }
    }

    /// Recompute and upload the active camera matrix.
    fn update_camera(&mut self) {
        self.camera_uniform.view_proj = self.current_view_proj().to_cols_array_2d();
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }

    /// Switch between Orbit and Fly, syncing position + look so the view is
    /// continuous across the toggle.
    fn toggle_mode(&mut self) {
        match self.mode {
            CameraMode::Orbit => {
                // Stand at the orbit eye, looking toward the orbit target.
                let eye = self.orbit.eye();
                let dir = (self.orbit.target - eye).normalize_or_zero();
                self.fly.position = eye;
                self.fly.pitch = dir.y.clamp(-1.0, 1.0).asin();
                self.fly.yaw = dir.z.atan2(dir.x);
                self.mode = CameraMode::Fly;
            }
            CameraMode::Fly => {
                // Orbit around a point in front of the fly camera, same distance.
                let f = self.fly.forward();
                self.orbit.target = self.fly.position + f * self.orbit.distance;
                let d = -f; // direction from target back to the eye
                self.orbit.pitch = d.y.clamp(-1.0, 1.0).asin();
                self.orbit.yaw = d.x.atan2(d.z);
                self.mode = CameraMode::Orbit;
            }
        }
        self.window.set_title(title_for(self.mode));
        self.update_camera();
    }

    /// Per-frame update: advance time and move the fly camera by held keys.
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

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Pass"),
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

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.draw(0..self.vertex_count, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
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

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(new_size) => state.resize(new_size),

            WindowEvent::RedrawRequested => {
                state.update();
                state.render();
            }

            WindowEvent::KeyboardInput {
                event: key_event, ..
            } => {
                if let PhysicalKey::Code(code) = key_event.physical_key {
                    let pressed = key_event.state == ElementState::Pressed;
                    match code {
                        KeyCode::KeyW => state.keys.forward = pressed,
                        KeyCode::KeyS => state.keys.back = pressed,
                        KeyCode::KeyA => state.keys.left = pressed,
                        KeyCode::KeyD => state.keys.right = pressed,
                        KeyCode::KeyE | KeyCode::Space => state.keys.up = pressed,
                        KeyCode::KeyQ => state.keys.down = pressed,
                        KeyCode::Tab => {
                            if pressed && !key_event.repeat {
                                state.toggle_mode();
                            }
                        }
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
                    MouseButton::Left => state.mouse_left_down = pressed,
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

            // Clear held keys when the window loses focus so nothing gets "stuck".
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

    /// With ControlFlow::Poll this runs every loop iteration; keep redrawing so
    /// movement animates smoothly even when there are no input events.
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("failed to create the event loop");
    // Continuous rendering: the loop runs every frame (vsync-capped on present).
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop
        .run_app(&mut app)
        .expect("the event loop exited with an error");
}
