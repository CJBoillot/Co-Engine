//! CoEngine — Milestone 1, Step 0: "First Window"
//!
//! Goal of this step: open a resizable window and clear it to a solid color.
//!
//! We use `winit` to create/manage the window and handle OS events, and `wgpu`
//! to talk to the GPU. On this Windows machine `wgpu` runs on **DirectX 12**
//! under the hood — so we're already on DX12, just without the heavy boilerplate.
//! Everything later (grid, cubes, camera, chat) builds on this foundation.

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

/// Everything needed to draw into the window: the window itself plus the wgpu
/// objects bound to it. Created once the window exists (see `App::resumed`).
struct State {
    /// The OS window. `Arc` lets both us and the GPU surface share ownership.
    window: Arc<Window>,
    /// The drawable area of the window, owned by the GPU.
    surface: wgpu::Surface<'static>,
    /// Our connection to the GPU.
    device: wgpu::Device,
    /// The queue we submit GPU work to.
    queue: wgpu::Queue,
    /// How the surface is configured (size, pixel format, vsync).
    config: wgpu::SurfaceConfiguration,
    /// Current window size in physical pixels.
    size: PhysicalSize<u32>,
}

impl State {
    /// Set up wgpu for `window`. Talking to the GPU is asynchronous, so we block
    /// on those calls with `pollster::block_on` to keep this function simple.
    fn new(window: Arc<Window>) -> State {
        let size = window.inner_size();

        // 1. Instance — the entry point to wgpu. It chooses a backend
        //    (DirectX 12 on this machine) and lets us create a surface.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());

        // 2. Surface — the region of the window we render into.
        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create a surface for the window");

        // 3. Adapter — a handle to a physical GPU (your RTX 4090).
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            },
        ))
        .expect("no compatible GPU adapter was found");

        // 4. Device + Queue — `device` creates GPU resources; `queue` runs commands.
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

        // 5. Configure the surface: a sensible default (format, size, vsync) for
        //    this adapter, then apply it.
        let config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("this surface is not supported by the adapter");
        surface.configure(&device, &config);

        State {
            window,
            surface,
            device,
            queue,
            config,
            size,
        }
    }

    /// Re-create the surface at the new size when the window is resized.
    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    /// Draw one frame: clear the whole window to a solid color and present it.
    fn render(&mut self) {
        // Grab the next image from the surface to draw into.
        let frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            // Common during resizes — just reconfigure and skip this frame.
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

        // A "view" is how the render pass refers to the texture.
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // An encoder records GPU commands; we finish and submit it below.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("CoEngine Command Encoder"),
            });

        // A render pass that does nothing but clear the screen to our color.
        // (The scope `{ }` ends the pass by dropping it before we submit.)
        {
            let _clear_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Clear Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // The solid background color (a dark blue-gray).
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.10,
                            g: 0.20,
                            b: 0.30,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }

        // Send the commands to the GPU and show the finished frame.
        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
    }
}

/// The application object winit drives. Holds the `State` once the window exists.
#[derive(Default)]
struct App {
    state: Option<State>,
}

impl ApplicationHandler for App {
    /// Called when the app can create its window. On desktop this fires once.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return; // Window already created.
        }

        let attributes = Window::default_attributes()
            .with_title("CoEngine — Step 0")
            .with_inner_size(LogicalSize::new(1280.0, 720.0));

        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .expect("failed to create the window"),
        );

        self.state = Some(State::new(window));

        // Ask for an initial paint so the clear color shows immediately.
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }

    /// Handle a single window event (close, resize, redraw).
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
            // User clicked the window's close button.
            WindowEvent::CloseRequested => event_loop.exit(),

            // Window was resized — match the surface to the new size and repaint.
            WindowEvent::Resized(new_size) => {
                state.resize(new_size);
                state.window.request_redraw();
            }

            // The OS (or our request) asked us to repaint.
            WindowEvent::RedrawRequested => state.render(),

            _ => {}
        }
    }
}

fn main() {
    // The event loop receives OS events and drives our `App`.
    let event_loop = EventLoop::new().expect("failed to create the event loop");

    // `Wait` sleeps until there's an event (low CPU). A static clear color doesn't
    // need to redraw continuously; later steps that animate will switch to `Poll`.
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::default();
    event_loop
        .run_app(&mut app)
        .expect("the event loop exited with an error");
}
