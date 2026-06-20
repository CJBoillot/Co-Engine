//! CoEngine — Milestone 1, Step 6: "Chat Wired to Claude"
//!
//! Builds on Step 5 (egui chat panel, local echo). New in this step:
//!   * the chat now talks to **Claude** (`claude-opus-4-8`) via the Anthropic
//!     Messages API, with the reply **streamed** in token-by-token,
//!   * the request runs on a **background thread** so the 3D app never freezes;
//!     streamed text is delivered to the UI over a channel and appended each frame,
//!   * the API key is read once from the **`ANTHROPIC_API_KEY`** environment
//!     variable — never stored in the repo. If it's unset, the chat says so.
//!
//! Versioning uses CoSemVer: `CO_VERSION` drives the on-screen version; a trailing
//! letter (e.g. 0.0.6b) marks a bug fix (Cargo.toml keeps the numeric base).
//!
//! Controls (when not typing in the chat):
//!   Orbit: left-drag orbit · right-drag pan · scroll zoom · click select · C add · Del remove · Tab Fly
//!   Fly:   WASD move · E/Q up/down · right-drag look · click select · C add · Del remove · Tab Orbit
//!   Esc quits.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
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
const CUBE_BASE_COLOR: [f32; 3] = [0.85, 0.55, 0.25];

/// CoSemVer display version (see memory: CoSemVer). A trailing letter marks a
/// bug fix for this version. Kept separate from Cargo's strict-SemVer `version`.
const CO_VERSION: &str = "0.0.9";

/// A Claude model the in-engine chat can use. `effort` marks models that accept
/// the `output_config.effort` speed control (Haiku 4.5 does not).
struct ModelChoice {
    name: &'static str,
    id: &'static str,
    effort: bool,
}

/// Models offered in the chat's Model dropdown, fastest/cheapest first.
const MODELS: &[ModelChoice] = &[
    ModelChoice { name: "Haiku 4.5 — fastest", id: "claude-haiku-4-5", effort: false },
    ModelChoice { name: "Sonnet 4.6 — balanced", id: "claude-sonnet-4-6", effort: true },
    ModelChoice { name: "Opus 4.8 — most capable", id: "claude-opus-4-8", effort: true },
    ModelChoice { name: "Fable 5 — most powerful", id: "claude-fable-5", effort: true },
];

/// Speed presets → the API `effort` value (effort-capable models only).
const EFFORTS: &[(&str, &str)] = &[
    ("Fast", "low"),
    ("Balanced", "medium"),
    ("High", "high"),
    ("Max", "max"),
];

const DEFAULT_MODEL_IDX: usize = 0; // Haiku 4.5 (cheapest/fastest, default)
const DEFAULT_EFFORT_IDX: usize = 1; // Balanced

// Forged CoEngine accent colors (from the icon): cobalt identity, gold accents.
const ACCENT_GOLD: egui::Color32 = egui::Color32::from_rgb(217, 138, 43);
const ACCENT_COBALT: egui::Color32 = egui::Color32::from_rgb(56, 116, 210);

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

/// A cube in the scene: a position plus a base color (shaded per-face at build).
#[derive(Clone, Copy)]
struct Cube {
    pos: Vec3,
    color: [f32; 3],
}

/// Half-grid placement: the Nth cube's (x, z) on the ground grid.
fn grid_slot(n: usize) -> (f32, f32) {
    let cols = 7;
    let col = (n % cols) as f32 - 3.0;
    let row = (n / cols) as f32 - 3.0;
    (col * 1.5, row * 1.5)
}

/// Parse a color name or "#rrggbb" hex into RGB. Falls back to the default cube color.
fn parse_color(name: &str) -> [f32; 3] {
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

fn build_grid(half: i32, spacing: f32) -> Vec<Vertex> {
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

fn push_quad(out: &mut Vec<Vertex>, a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3], color: [f32; 3]) {
    for pos in [a, b, c, a, c, d] {
        out.push(Vertex { position: pos, color });
    }
}

fn unit_cube() -> Vec<Vertex> {
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

fn build_cube_vertices(cubes: &[Cube], selected: Option<usize>) -> Vec<Vertex> {
    let base = unit_cube();
    let mut out = Vec::with_capacity(cubes.len() * base.len());

    for (i, cube) in cubes.iter().enumerate() {
        let highlight = Some(i) == selected;
        for v in &base {
            let shade = v.color[0]; // per-face grayscale brightness
            let color = if highlight {
                // Selected cubes glow cobalt (the engine identity color).
                [shade * 0.25, shade * 0.55, shade * 1.00]
            } else {
                [
                    shade * cube.color[0],
                    shade * cube.color[1],
                    shade * cube.color[2],
                ]
            };
            out.push(Vertex {
                position: [
                    v.position[0] + cube.pos.x,
                    v.position[1] + cube.pos.y,
                    v.position[2] + cube.pos.z,
                ],
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
// Chat (UI state + Claude streaming)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Role {
    User,
    Assistant,
}

impl Role {
    fn label(self) -> &'static str {
        match self {
            Role::User => "You",
            Role::Assistant => "CoE-AI",
        }
    }
    fn api(self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

struct ChatMessage {
    role: Role,
    text: String,
}

/// Messages sent from the streaming worker thread back to the UI.
enum StreamMsg {
    Delta(String),
    Command(SceneCommand),
    ClaudePrompt(String),
    Done,
    Error(String),
}

/// A mutation the AI agent wants applied to the scene (executed on the main thread).
enum SceneCommand {
    Add { x: f32, z: f32, color: [f32; 3] },
    SetColor { index: usize, color: [f32; 3] },
    Remove { index: usize },
    Select { index: usize },
    Clear,
}

/// Snapshot of the scene handed to the agent so it knows what exists.
struct SceneSnapshot {
    cubes: Vec<(Vec3, [f32; 3])>,
    selected: Option<usize>,
}

/// Everything the worker thread needs to run an agent turn.
struct AgentRequest {
    api_key: String,
    model_id: String,
    effort: Option<&'static str>,
    system: String,
    positions: Vec<(f32, f32)>,
    selected: Option<usize>,
    messages: Vec<serde_json::Value>,
}

#[derive(Default)]
struct ChatUi {
    messages: Vec<ChatMessage>,
    input: String,
    /// Receiver for the in-flight reply (None when idle).
    rx: Option<Receiver<StreamMsg>>,
    /// Index of the assistant message currently being streamed into.
    streaming_index: Option<usize>,
    /// A short status/error line shown under the header.
    status: String,
    /// Currently selected model + speed (indices into MODELS / EFFORTS).
    model_idx: usize,
    effort_idx: usize,
    /// A prompt CoE-AI prepared for the user to paste to Claude (Desktop).
    pending_claude_prompt: Option<String>,
}

/// Find the Anthropic API key: the ANTHROPIC_API_KEY environment variable first,
/// then a local `.env` file (a line like `ANTHROPIC_API_KEY=sk-ant-...`). The
/// `.env` file is gitignored, so the key is never committed.
fn load_api_key() -> Option<String> {
    if let Ok(k) = std::env::var("ANTHROPIC_API_KEY") {
        if !k.trim().is_empty() {
            return Some(k.trim().to_string());
        }
    }
    if let Ok(contents) = std::fs::read_to_string(".env") {
        for line in contents.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((key, val)) = line.split_once('=') {
                if key.trim() == "ANTHROPIC_API_KEY" {
                    let val = val.trim().trim_matches('"').trim_matches('\'').trim();
                    if !val.is_empty() {
                        return Some(val.to_string());
                    }
                }
            }
        }
    }
    None
}

/// Send the current input to Claude (real, streaming). Ignored if a reply is
/// already in flight or the input is empty.
fn send_message(chat: &mut ChatUi, scene: &SceneSnapshot) {
    if chat.rx.is_some() {
        return; // a reply is already streaming
    }
    let text = chat.input.trim().to_string();
    if text.is_empty() {
        return;
    }
    chat.input.clear();

    // Find the API key: env var first, then a local .env file. Never in the repo.
    let api_key = match load_api_key() {
        Some(k) => k,
        None => {
            chat.messages.push(ChatMessage { role: Role::User, text });
            chat.messages.push(ChatMessage {
                role: Role::Assistant,
                text: "No API key found. Paste your key into the .env file in the project folder (ANTHROPIC_API_KEY=sk-ant-...) and save, then send again — no restart needed. (Or set the ANTHROPIC_API_KEY environment variable.)"
                    .to_string(),
            });
            chat.status = "API key not set".to_string();
            return;
        }
    };

    // Record the user's message.
    chat.messages.push(ChatMessage { role: Role::User, text });

    // Resolve the chosen model + speed.
    let model = &MODELS[chat.model_idx.min(MODELS.len() - 1)];
    let model_id = model.id.to_string();
    let effort: Option<&'static str> = if model.effort {
        Some(EFFORTS[chat.effort_idx.min(EFFORTS.len() - 1)].1)
    } else {
        None
    };

    // Snapshot the conversation (the user's message was already pushed above).
    let api_messages: Vec<serde_json::Value> = chat
        .messages
        .iter()
        .map(|m| serde_json::json!({ "role": m.role.api(), "content": m.text }))
        .collect();

    // Create the empty assistant message the agent's final text fills in.
    chat.messages.push(ChatMessage { role: Role::Assistant, text: String::new() });
    chat.streaming_index = Some(chat.messages.len() - 1);
    chat.status = "CoE-AI is working…".to_string();

    let req = AgentRequest {
        api_key,
        model_id,
        effort,
        system: build_system_prompt(scene),
        positions: scene.cubes.iter().map(|(p, _)| (p.x, p.z)).collect(),
        selected: scene.selected,
        messages: api_messages,
    };

    let (tx, rx) = std::sync::mpsc::channel();
    chat.rx = Some(rx);
    std::thread::spawn(move || run_agent(req, tx));
}

/// Build the agent's system prompt from the current scene state.
fn build_system_prompt(scene: &SceneSnapshot) -> String {
    let mut s = String::from(
        "You are CoE-AI, the assistant built into CoEngine, a 3D engine the user is building. \
You can DIRECTLY change the 3D scene using the provided tools — when the user asks for a scene change \
(add, recolor, remove, or select cubes), DO IT with the tools rather than describing code. After acting, \
confirm in one short sentence.\n\n",
    );
    s.push_str(
        "About CoEngine (use this context when writing prompts for Claude): it is a desktop 3D engine \
written in Rust, using wgpu (DirectX 12) for rendering, winit for the window, and egui for the UI. The \
scene is a list of cubes, each with a position and an RGB color (a `Vec<Cube>` in src/main.rs). Versioning \
is \"CoSemVer\" (0.0.N). Cody builds the engine with Claude (the desktop assistant) and builds games with \
you. Development goes one small step at a time.\n\n",
    );
    if scene.cubes.is_empty() {
        s.push_str("The scene currently has no cubes.\n");
    } else {
        s.push_str(&format!("The scene has {} cube(s):\n", scene.cubes.len()));
        for (i, (pos, color)) in scene.cubes.iter().enumerate() {
            let marker = if scene.selected == Some(i) { "  [SELECTED]" } else { "" };
            s.push_str(&format!(
                "  #{i}: x={:.1}, z={:.1}, color=({:.2},{:.2},{:.2}){marker}\n",
                pos.x, pos.z, color[0], color[1], color[2]
            ));
        }
    }
    match scene.selected {
        Some(i) => s.push_str(&format!(
            "\n\"This cube\" / \"the selected cube\" refers to cube #{i}.\n"
        )),
        None => s.push_str(
            "\nNo cube is selected; if the user says \"this cube\" with nothing selected, ask which index.\n",
        ),
    }
    s.push_str(
        "Colors may be names (red, green, blue, orange, yellow, purple, cyan, pink, white, black, gray, brown) or hex like #33cc44.\n",
    );
    s.push_str(
        "\nIMPORTANT — your limits: you can ONLY use the tools provided (currently: add/recolor/remove/select/clear cubes). You CANNOT modify CoEngine itself, save or load projects, import arbitrary assets, or do file operations. If the user asks for anything beyond your tools, do NOT pretend and do NOT give code to paste elsewhere — call the request_engine_change tool. Make the prompt SELF-CONTAINED and CoEngine-specific: include the engine context above (Rust + wgpu + egui; the `Vec<Cube>` scene of position+color), the exact capability needed, sensible defaults (e.g. a file format like JSON, and where it integrates — a CoE-AI tool and/or an Esc-menu item), and mention any load/round-trip needs. Respect CoEngine's one-step-at-a-time style. Then tell the user in one sentence that you've prepared a prompt they can copy.\n",
    );
    s
}

/// Worker thread: POST to the Anthropic Messages API and stream the reply,
/// forwarding text deltas to the UI over `tx`.
/// The tools the agent can call to act on the 3D scene.
fn tools_json() -> serde_json::Value {
    serde_json::json!([
        {
            "name": "add_cube",
            "description": "Add a cube to the 3D scene. Optionally give a color (a name like \"green\" or hex like \"#33cc44\") and an x/z position on the ground; if x/z are omitted the engine auto-places it.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "color": { "type": "string" },
                    "x": { "type": "number" },
                    "z": { "type": "number" }
                }
            }
        },
        {
            "name": "set_cube_color",
            "description": "Change the color of an existing cube, identified by its index.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "index": { "type": "integer" },
                    "color": { "type": "string" }
                },
                "required": ["index", "color"]
            }
        },
        {
            "name": "remove_cube",
            "description": "Remove a cube from the scene by its index.",
            "input_schema": {
                "type": "object",
                "properties": { "index": { "type": "integer" } },
                "required": ["index"]
            }
        },
        {
            "name": "select_cube",
            "description": "Select (highlight) a cube by its index.",
            "input_schema": {
                "type": "object",
                "properties": { "index": { "type": "integer" } },
                "required": ["index"]
            }
        },
        {
            "name": "clear_scene",
            "description": "Remove every cube from the scene.",
            "input_schema": { "type": "object", "properties": {} }
        },
        {
            "name": "request_engine_change",
            "description": "Use this when the user asks for something you CANNOT do with your other tools (saving/loading projects, importing assets, file operations, new engine features or new tools). Provide a clear, specific prompt the user can paste to Claude (the desktop assistant that builds CoEngine) to add the capability. Do NOT give code to run elsewhere — call this tool instead.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "prompt": { "type": "string", "description": "A ready-to-paste prompt for Claude (Desktop) describing the engine change needed." },
                    "reason": { "type": "string", "description": "One short sentence on what the user wanted." }
                },
                "required": ["prompt"]
            }
        }
    ])
}

/// Execute one tool call against the worker's scene mirror, emit the matching
/// SceneCommand for the main thread, and return a short result for the model.
fn execute_tool(
    name: &str,
    input: &serde_json::Value,
    mirror: &mut Vec<(f32, f32)>,
    selected: &mut Option<usize>,
    tx: &Sender<StreamMsg>,
) -> String {
    match name {
        "add_cube" => {
            let color_name = input.get("color").and_then(|c| c.as_str()).unwrap_or("orange");
            let color = parse_color(color_name);
            let n = mirror.len();
            let (x, z) = match (
                input.get("x").and_then(|v| v.as_f64()),
                input.get("z").and_then(|v| v.as_f64()),
            ) {
                (Some(x), Some(z)) => (x as f32, z as f32),
                _ => grid_slot(n),
            };
            mirror.push((x, z));
            let _ = tx.send(StreamMsg::Command(SceneCommand::Add { x, z, color }));
            format!("Added a {color_name} cube as #{n} at x={x:.1}, z={z:.1}.")
        }
        "set_cube_color" => {
            let idx = input.get("index").and_then(|v| v.as_u64()).unwrap_or(u64::MAX) as usize;
            let color_name = input.get("color").and_then(|c| c.as_str()).unwrap_or("orange");
            if idx < mirror.len() {
                let color = parse_color(color_name);
                let _ = tx.send(StreamMsg::Command(SceneCommand::SetColor { index: idx, color }));
                format!("Cube #{idx} is now {color_name}.")
            } else {
                format!("There is no cube #{idx} (the scene has {} cubes).", mirror.len())
            }
        }
        "remove_cube" => {
            let idx = input.get("index").and_then(|v| v.as_u64()).unwrap_or(u64::MAX) as usize;
            if idx < mirror.len() {
                mirror.remove(idx);
                if *selected == Some(idx) {
                    *selected = None;
                }
                let _ = tx.send(StreamMsg::Command(SceneCommand::Remove { index: idx }));
                format!("Removed cube #{idx}.")
            } else {
                format!("There is no cube #{idx} (the scene has {} cubes).", mirror.len())
            }
        }
        "select_cube" => {
            let idx = input.get("index").and_then(|v| v.as_u64()).unwrap_or(u64::MAX) as usize;
            if idx < mirror.len() {
                *selected = Some(idx);
                let _ = tx.send(StreamMsg::Command(SceneCommand::Select { index: idx }));
                format!("Selected cube #{idx}.")
            } else {
                format!("There is no cube #{idx} (the scene has {} cubes).", mirror.len())
            }
        }
        "clear_scene" => {
            mirror.clear();
            *selected = None;
            let _ = tx.send(StreamMsg::Command(SceneCommand::Clear));
            "Cleared all cubes from the scene.".to_string()
        }
        "request_engine_change" => {
            let prompt = input.get("prompt").and_then(|v| v.as_str()).unwrap_or("").to_string();
            if prompt.trim().is_empty() {
                "No prompt was provided.".to_string()
            } else {
                let _ = tx.send(StreamMsg::ClaudePrompt(prompt));
                "Prepared a prompt for Claude and showed Cody a Copy button. Briefly tell Cody what it's for.".to_string()
            }
        }
        other => format!("Unknown tool: {other}"),
    }
}

/// Worker thread: run the agentic tool-use loop. Tool calls mutate the scene
/// (via SceneCommands) and their results feed back until the model gives a final
/// text answer. Non-streaming for simplicity; capped at a few iterations.
fn run_agent(req: AgentRequest, tx: Sender<StreamMsg>) {
    let AgentRequest {
        api_key,
        model_id,
        effort,
        system,
        positions,
        selected,
        mut messages,
    } = req;

    let tools = tools_json();
    let mut mirror = positions;
    let mut selected = selected;

    for _ in 0..8 {
        let mut body = serde_json::json!({
            "model": model_id,
            "max_tokens": 1024,
            "system": system,
            "tools": tools,
            "messages": messages,
        });
        if let Some(eff) = effort {
            body["output_config"] = serde_json::json!({ "effort": eff });
            body["thinking"] = serde_json::json!({ "type": "adaptive" });
        }
        let body_str = serde_json::to_string(&body).unwrap_or_default();

        let response = ureq::post("https://api.anthropic.com/v1/messages")
            .set("x-api-key", &api_key)
            .set("anthropic-version", "2023-06-01")
            .set("content-type", "application/json")
            .send_string(&body_str);

        let response = match response {
            Ok(r) => r,
            Err(ureq::Error::Status(code, r)) => {
                let detail = r.into_string().unwrap_or_default();
                let _ = tx.send(StreamMsg::Error(format!("HTTP {code}: {detail}")));
                return;
            }
            Err(e) => {
                let _ = tx.send(StreamMsg::Error(format!("request failed: {e}")));
                return;
            }
        };

        let raw = match response.into_string() {
            Ok(s) => s,
            Err(e) => {
                let _ = tx.send(StreamMsg::Error(format!("read failed: {e}")));
                return;
            }
        };
        let v: serde_json::Value = match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(e) => {
                let _ = tx.send(StreamMsg::Error(format!("bad response: {e}")));
                return;
            }
        };

        let content = v
            .get("content")
            .and_then(|c| c.as_array())
            .cloned()
            .unwrap_or_default();

        let mut text = String::new();
        let mut tool_uses: Vec<serde_json::Value> = Vec::new();
        for block in &content {
            match block.get("type").and_then(|t| t.as_str()) {
                Some("text") => {
                    if let Some(t) = block.get("text").and_then(|t| t.as_str()) {
                        text.push_str(t);
                    }
                }
                Some("tool_use") => tool_uses.push(block.clone()),
                _ => {}
            }
        }

        if tool_uses.is_empty() {
            if !text.trim().is_empty() {
                let _ = tx.send(StreamMsg::Delta(text));
            }
            let _ = tx.send(StreamMsg::Done);
            return;
        }

        if !text.trim().is_empty() {
            let _ = tx.send(StreamMsg::Delta(format!("{}\n", text.trim())));
        }

        let mut results: Vec<serde_json::Value> = Vec::new();
        for tu in &tool_uses {
            let id = tu.get("id").and_then(|x| x.as_str()).unwrap_or("").to_string();
            let name = tu.get("name").and_then(|x| x.as_str()).unwrap_or("");
            let input = tu.get("input").cloned().unwrap_or_else(|| serde_json::json!({}));
            let result = execute_tool(name, &input, &mut mirror, &mut selected, &tx);
            results.push(serde_json::json!({
                "type": "tool_result",
                "tool_use_id": id,
                "content": result,
            }));
        }

        messages.push(serde_json::json!({ "role": "assistant", "content": content }));
        messages.push(serde_json::json!({ "role": "user", "content": results }));
    }

    let _ = tx.send(StreamMsg::Delta("(Stopped after several tool steps.)".to_string()));
    let _ = tx.send(StreamMsg::Done);
}

/// Visual theme preset for the UI (the 3D viewport is unaffected by themes).
#[derive(Clone, Copy, PartialEq)]
enum Theme {
    DefaultSimple,
    Barbarian,
}

/// Which main editor tab is active.
#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Scene,
    Logic,
    Code,
}

/// UI/chrome state: theme, menu/modal visibility, active tab, controls overlay.
struct UiState {
    theme: Theme,
    dark_mode: bool,
    menu_open: bool,
    settings_open: bool,
    show_debug: bool,
    active_tab: Tab,
    should_quit: bool,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            theme: Theme::DefaultSimple,
            dark_mode: true,
            menu_open: false,
            settings_open: false,
            show_debug: true,
            active_tab: Tab::Scene,
            should_quit: false,
        }
    }
}

/// Pick the egui visuals for the chosen theme + light/dark mode.
fn theme_visuals(theme: Theme, dark: bool) -> egui::Visuals {
    match (theme, dark) {
        (Theme::DefaultSimple, true) => default_dark(),
        (Theme::DefaultSimple, false) => default_light(),
        (Theme::Barbarian, true) => barbarian_dark(),
        (Theme::Barbarian, false) => barbarian_light(),
    }
}

/// "Default Simple" dark: clean charcoal-iron base, cobalt identity, gold accents.
fn default_dark() -> egui::Visuals {
    use egui::{Color32, Rounding, Stroke};
    let round = Rounding::same(3.0);
    let charcoal = Color32::from_rgb(18, 20, 26);
    let iron = Color32::from_rgb(34, 38, 46);
    let iron_hover = Color32::from_rgb(50, 56, 66);
    let slate = Color32::from_rgb(24, 27, 34);
    let text = Color32::from_rgb(226, 221, 207);

    let mut v = egui::Visuals::dark();
    v.override_text_color = Some(text);
    v.panel_fill = charcoal;
    v.window_fill = slate;
    v.window_stroke = Stroke::new(1.0, Color32::from_rgb(80, 64, 36));
    v.window_rounding = Rounding::same(5.0);
    v.extreme_bg_color = Color32::from_rgb(12, 13, 17);
    v.faint_bg_color = Color32::from_rgb(28, 31, 38);
    v.selection.bg_fill = Color32::from_rgb(34, 64, 120);
    v.selection.stroke = Stroke::new(1.0, ACCENT_COBALT);
    v.hyperlink_color = ACCENT_GOLD;

    v.widgets.noninteractive.bg_fill = charcoal;
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, text);
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, Color32::from_rgb(42, 46, 54));

    v.widgets.inactive.bg_fill = iron;
    v.widgets.inactive.weak_bg_fill = iron;
    v.widgets.inactive.fg_stroke = Stroke::new(1.0, text);
    v.widgets.inactive.bg_stroke = Stroke::new(1.0, Color32::from_rgb(62, 68, 78));
    v.widgets.inactive.rounding = round;

    v.widgets.hovered.bg_fill = iron_hover;
    v.widgets.hovered.weak_bg_fill = iron_hover;
    v.widgets.hovered.fg_stroke = Stroke::new(1.5, Color32::WHITE);
    v.widgets.hovered.bg_stroke = Stroke::new(1.5, ACCENT_GOLD);
    v.widgets.hovered.rounding = round;

    v.widgets.active.bg_fill = Color32::from_rgb(38, 64, 116);
    v.widgets.active.weak_bg_fill = Color32::from_rgb(38, 64, 116);
    v.widgets.active.fg_stroke = Stroke::new(1.5, Color32::WHITE);
    v.widgets.active.bg_stroke = Stroke::new(1.5, ACCENT_COBALT);
    v.widgets.active.rounding = round;

    v.widgets.open.bg_fill = iron;
    v.widgets.open.fg_stroke = Stroke::new(1.0, text);
    v.widgets.open.rounding = round;
    v
}

/// "Default Simple" light: cool stone/parchment.
fn default_light() -> egui::Visuals {
    use egui::{Color32, Rounding, Stroke};
    let round = Rounding::same(3.0);
    let parchment = Color32::from_rgb(224, 218, 204);
    let stone = Color32::from_rgb(202, 196, 182);
    let ink = Color32::from_rgb(38, 34, 28);

    let mut v = egui::Visuals::light();
    v.override_text_color = Some(ink);
    v.panel_fill = parchment;
    v.window_fill = Color32::from_rgb(232, 227, 214);
    v.window_stroke = Stroke::new(1.0, Color32::from_rgb(150, 120, 70));
    v.faint_bg_color = stone;
    v.extreme_bg_color = Color32::from_rgb(238, 233, 222);
    v.selection.bg_fill = Color32::from_rgb(150, 178, 222);
    v.selection.stroke = Stroke::new(1.0, ACCENT_COBALT);
    v.hyperlink_color = Color32::from_rgb(150, 95, 25);
    v.widgets.inactive.bg_fill = stone;
    v.widgets.inactive.weak_bg_fill = stone;
    v.widgets.inactive.rounding = round;
    v.widgets.hovered.bg_stroke = Stroke::new(1.5, Color32::from_rgb(176, 110, 30));
    v.widgets.hovered.rounding = round;
    v.widgets.active.bg_stroke = Stroke::new(1.5, ACCENT_COBALT);
    v.widgets.active.rounding = round;
    v
}

/// "Barbarian" dark: heavier, warmer forged look — leather/iron tones, bold gold
/// borders, stronger bevels. (Image-based metal/rune textures are a future layer.)
fn barbarian_dark() -> egui::Visuals {
    use egui::{Color32, Rounding, Stroke};
    let r = Rounding::same(5.0);
    let mut v = default_dark();
    v.panel_fill = Color32::from_rgb(28, 23, 18);
    v.window_fill = Color32::from_rgb(36, 29, 22);
    v.window_stroke = Stroke::new(2.0, ACCENT_GOLD);
    v.window_rounding = Rounding::same(6.0);
    v.faint_bg_color = Color32::from_rgb(40, 33, 25);
    v.widgets.noninteractive.bg_fill = Color32::from_rgb(28, 23, 18);
    v.widgets.inactive.bg_fill = Color32::from_rgb(48, 40, 30);
    v.widgets.inactive.weak_bg_fill = Color32::from_rgb(48, 40, 30);
    v.widgets.inactive.bg_stroke = Stroke::new(1.5, Color32::from_rgb(120, 92, 44));
    v.widgets.inactive.rounding = r;
    v.widgets.hovered.bg_fill = Color32::from_rgb(64, 53, 38);
    v.widgets.hovered.weak_bg_fill = Color32::from_rgb(64, 53, 38);
    v.widgets.hovered.bg_stroke = Stroke::new(2.0, ACCENT_GOLD);
    v.widgets.hovered.rounding = r;
    v.widgets.active.bg_fill = Color32::from_rgb(56, 78, 130);
    v.widgets.active.weak_bg_fill = Color32::from_rgb(56, 78, 130);
    v.widgets.active.bg_stroke = Stroke::new(2.0, ACCENT_COBALT);
    v.widgets.active.rounding = r;
    v.selection.bg_fill = Color32::from_rgb(48, 82, 142);
    v
}

/// "Barbarian" light: tanned leather/parchment with heavy gold edges.
fn barbarian_light() -> egui::Visuals {
    use egui::{Color32, Stroke};
    let mut v = default_light();
    v.panel_fill = Color32::from_rgb(214, 200, 176);
    v.window_fill = Color32::from_rgb(224, 211, 188);
    v.window_stroke = Stroke::new(2.0, Color32::from_rgb(150, 110, 50));
    v.faint_bg_color = Color32::from_rgb(198, 184, 160);
    v.widgets.inactive.bg_fill = Color32::from_rgb(198, 182, 156);
    v.widgets.inactive.weak_bg_fill = Color32::from_rgb(198, 182, 156);
    v.widgets.inactive.bg_stroke = Stroke::new(1.5, Color32::from_rgb(150, 110, 50));
    v.widgets.hovered.bg_stroke = Stroke::new(2.0, Color32::from_rgb(176, 110, 30));
    v.widgets.active.bg_stroke = Stroke::new(2.0, ACCENT_COBALT);
    v
}

/// Build the whole egui UI for this frame: top bar (menu + identity + tabs), tool
/// row, tab content, chat panel, the bottom-left controls HUD, and menu/settings.
fn build_ui(
    ctx: &egui::Context,
    ui_state: &mut UiState,
    chat: &mut ChatUi,
    mode: CameraMode,
    logo: Option<&egui::TextureHandle>,
    scene: &SceneSnapshot,
) {
    // Apply the selected theme + mode (UI only — the 3D background is fixed).
    ctx.set_visuals(theme_visuals(ui_state.theme, ui_state.dark_mode));

    // Top bar: Menu + identity + tabs, all on one row.
    egui::TopBottomPanel::top("topbar").show(ctx, |ui| {
        ui.add_space(2.0);
        ui.horizontal(|ui| {
            if ui.button("Menu").clicked() {
                ui_state.menu_open = true;
            }
            ui.separator();
            ui.label(egui::RichText::new("CoEngine").strong().color(ACCENT_GOLD));
            ui.label(egui::RichText::new(format!("v{CO_VERSION}")));
            ui.separator();
            ui.selectable_value(&mut ui_state.active_tab, Tab::Scene, "3D Scene");
            ui.selectable_value(&mut ui_state.active_tab, Tab::Logic, "Logic");
            ui.selectable_value(&mut ui_state.active_tab, Tab::Code, "Code");
            // When the debug overlay is hidden, offer a way back (also bound to H).
            if !ui_state.show_debug {
                ui.separator();
                if ui
                    .button("Debug Info (H)")
                    .on_hover_text("Show the controls & version overlay (H)")
                    .clicked()
                {
                    ui_state.show_debug = true;
                }
            }
        });
        ui.add_space(2.0);
    });

    // Tool row (blank placeholder for now).
    egui::TopBottomPanel::top("tool_bar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.add_space(2.0);
            ui.label(egui::RichText::new("Tools").small());
        });
    });

    // Right-side chat panel.
    build_chat_panel(ctx, chat, scene);

    // Tab content. The 3D Scene tab leaves the central area empty so the wgpu
    // viewport shows through; other tabs cover it with a placeholder.
    match ui_state.active_tab {
        Tab::Scene => {}
        Tab::Logic => {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.centered_and_justified(|ui| {
                    ui.label(egui::RichText::new("Logic — coming soon").heading().weak());
                });
            });
        }
        Tab::Code => {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.centered_and_justified(|ui| {
                    ui.label(egui::RichText::new("Code — coming soon").heading().weak());
                });
            });
        }
    }

    // Bottom-left debug overlay: controls + version on a dark plate so the text is
    // readable over the 3D viewport. Hidden entirely with H (re-shown via the top bar).
    if ui_state.show_debug {
        egui::Area::new(egui::Id::new("hud_bottom_left"))
            .anchor(egui::Align2::LEFT_BOTTOM, egui::vec2(10.0, -10.0))
            .interactable(false)
            .show(ctx, |ui| {
                egui::Frame::none()
                    .fill(egui::Color32::from_rgba_unmultiplied(12, 14, 20, 235))
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(70, 56, 30)))
                    .rounding(egui::Rounding::same(4.0))
                    .inner_margin(egui::Margin::same(8.0))
                    .show(ui, |ui| {
                        let text = egui::Color32::from_rgb(228, 223, 209);
                        let dim = egui::Color32::from_rgb(168, 163, 148);
                        let cam = match mode {
                            CameraMode::Orbit => {
                                "Orbit:  drag = orbit · R-drag = pan · scroll = zoom"
                            }
                            CameraMode::Fly => "Fly:  WASD move · E/Q up/down · R-drag = look",
                        };
                        ui.label(egui::RichText::new(cam).color(text).small());
                        ui.label(
                            egui::RichText::new(
                                "click = select · C = add cube · Del = remove · Tab = orbit/fly",
                            )
                            .color(text)
                            .small(),
                        );
                        ui.label(
                            egui::RichText::new("Esc = menu · H = hide debug info")
                                .color(dim)
                                .small(),
                        );
                        ui.label(
                            egui::RichText::new(format!("CoEngine v{CO_VERSION}"))
                                .monospace()
                                .color(ACCENT_GOLD),
                        );
                    });
            });
    }

    // Menu window (opened by Esc or the Menu button).
    if ui_state.menu_open {
        egui::Window::new("menu")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .movable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                ui.set_min_width(240.0);
                ui.add_space(10.0);
                ui.vertical_centered(|ui| {
                    if let Some(tex) = logo {
                        ui.add(egui::Image::new(egui::load::SizedTexture::new(
                            tex.id(),
                            egui::vec2(56.0, 56.0),
                        )));
                    }
                    ui.add_space(4.0);
                    ui.label(egui::RichText::new("CoEngine").heading().color(ACCENT_GOLD));
                    ui.label(egui::RichText::new(format!("v{CO_VERSION}")).small());
                });
                ui.add_space(10.0);
                ui.separator();
                ui.add_space(8.0);

                let bw = ui.available_width();
                if ui
                    .add_sized([bw, 34.0], egui::Button::new("Settings"))
                    .clicked()
                {
                    ui_state.settings_open = true;
                    ui_state.menu_open = false;
                }
                ui.add_space(6.0);
                if ui
                    .add_sized(
                        [bw, 34.0],
                        egui::Button::new("Exit").fill(egui::Color32::from_rgb(122, 44, 38)),
                    )
                    .clicked()
                {
                    ui_state.should_quit = true;
                }
                ui.add_space(8.0);
                ui.separator();
                ui.add_space(6.0);
                if ui.add_sized([bw, 24.0], egui::Button::new("Close")).clicked() {
                    ui_state.menu_open = false;
                }
                ui.add_space(8.0);
            });
    }

    // Settings window (modal-ish).
    if ui_state.settings_open {
        egui::Window::new("Settings")
            .collapsible(false)
            .resizable(false)
            .movable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                ui.set_min_width(260.0);
                ui.label("Theme");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut ui_state.theme, Theme::DefaultSimple, "Default Simple");
                    ui.selectable_value(&mut ui_state.theme, Theme::Barbarian, "Barbarian");
                });
                ui.add_space(6.0);
                ui.label("Mode (UI only — does not affect the 3D view)");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut ui_state.dark_mode, true, "Dark");
                    ui.selectable_value(&mut ui_state.dark_mode, false, "Light");
                });
                ui.separator();
                if ui.button("Close").clicked() {
                    ui_state.settings_open = false;
                }
            });
    }
}

/// Build the right-side chat panel.
fn build_chat_panel(ctx: &egui::Context, chat: &mut ChatUi, scene: &SceneSnapshot) {
    egui::SidePanel::right("chat_panel")
        .resizable(true)
        .default_width(320.0)
        .show(ctx, |ui| {
            egui::TopBottomPanel::top("chat_header").show_inside(ui, |ui| {
                ui.add_space(6.0);
                ui.heading("Chat");
                if !chat.status.is_empty() {
                    ui.label(egui::RichText::new(&chat.status).small().italics());
                }
                ui.add_space(4.0);
            });

            // A prompt CoE-AI prepared for the user to paste to Claude (Desktop).
            if let Some(prompt) = chat.pending_claude_prompt.clone() {
                egui::TopBottomPanel::top("claude_prompt_panel").show_inside(ui, |ui| {
                    egui::Frame::none()
                        .fill(egui::Color32::from_rgb(40, 33, 18))
                        .stroke(egui::Stroke::new(1.0, ACCENT_GOLD))
                        .rounding(egui::Rounding::same(4.0))
                        .inner_margin(egui::Margin::same(8.0))
                        .show(ui, |ui| {
                            ui.label(
                                egui::RichText::new("Prompt for Claude (to evolve the engine):")
                                    .strong()
                                    .color(ACCENT_GOLD),
                            );
                            ui.add_space(4.0);
                            egui::ScrollArea::vertical()
                                .max_height(120.0)
                                .auto_shrink([false, true])
                                .show(ui, |ui| {
                                    ui.label(egui::RichText::new(&prompt).small());
                                });
                            ui.add_space(4.0);
                            ui.horizontal(|ui| {
                                if ui.button("Copy prompt").clicked() {
                                    ui.output_mut(|o| o.copied_text = prompt.clone());
                                }
                                if ui.button("Dismiss").clicked() {
                                    chat.pending_claude_prompt = None;
                                }
                            });
                        });
                });
            }

            egui::TopBottomPanel::bottom("chat_input").show_inside(ui, |ui| {
                ui.add_space(6.0);
                let streaming = chat.rx.is_some();
                // Always enabled so focus persists across sends (Send button locks instead).
                let resp = ui.add(
                    egui::TextEdit::singleline(&mut chat.input)
                        .hint_text("Message CoE-AI…")
                        .desired_width(f32::INFINITY),
                );
                let enter_pressed =
                    resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));

                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    // Model drop-up. Effort-capable models also show a Speed drop-up.
                    egui::ComboBox::from_id_salt("model_select")
                        .selected_text(MODELS[chat.model_idx].name)
                        .show_ui(ui, |ui| {
                            for (i, m) in MODELS.iter().enumerate() {
                                ui.selectable_value(&mut chat.model_idx, i, m.name);
                            }
                        });
                    if MODELS[chat.model_idx].effort {
                        egui::ComboBox::from_id_salt("speed_select")
                            .selected_text(EFFORTS[chat.effort_idx].0)
                            .show_ui(ui, |ui| {
                                for (i, e) in EFFORTS.iter().enumerate() {
                                    ui.selectable_value(&mut chat.effort_idx, i, e.0);
                                }
                            });
                    }

                    // Send / Clear pinned to the right of the same row.
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let send_clicked =
                            ui.add_enabled(!streaming, egui::Button::new("Send")).clicked();
                        if ui.button("Clear").clicked() {
                            chat.messages.clear();
                        }
                        if send_clicked || enter_pressed {
                            send_message(chat, scene);
                            resp.request_focus();
                        }
                    });
                });
                ui.add_space(6.0);
            });

            egui::CentralPanel::default().show_inside(ui, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for m in &chat.messages {
                            ui.label(egui::RichText::new(format!("{}:", m.role.label())).strong());
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

fn title_for(mode: CameraMode) -> String {
    let name = match mode {
        CameraMode::Orbit => "Orbit",
        CameraMode::Fly => "Fly",
    };
    format!("CoEngine v{CO_VERSION}   [{name}]")
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

    cubes: Vec<Cube>,
    cube_buffer: Option<wgpu::Buffer>,
    cube_vertex_count: u32,
    selected: Option<usize>,

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

    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    chat: ChatUi,
    ui: UiState,
    /// The CoE logo as an egui texture (None if assets/icon.png is missing).
    logo_texture: Option<egui::TextureHandle>,
}

impl State {
    fn new(window: Arc<Window>, logo: Option<(Vec<u8>, u32, u32)>) -> State {
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

        let logo_texture = logo.map(|(rgba, w, h)| {
            let image = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba);
            egui_ctx.load_texture("coengine_logo", image, egui::TextureOptions::LINEAR)
        });

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
            chat: ChatUi {
                model_idx: DEFAULT_MODEL_IDX,
                effort_idx: DEFAULT_EFFORT_IDX,
                ..Default::default()
            },
            ui: UiState::default(),
            logo_texture,
        }
    }

    /// Esc behavior: close the settings modal if open, else toggle the menu.
    fn toggle_menu(&mut self) {
        if self.ui.settings_open {
            self.ui.settings_open = false;
        } else {
            self.ui.menu_open = !self.ui.menu_open;
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
        let (x, z) = grid_slot(self.cubes.len());
        self.cubes.push(Cube {
            pos: Vec3::new(x, CUBE_HALF, z),
            color: CUBE_BASE_COLOR,
        });
        self.rebuild_cubes();
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
            if let Some(t) = ray_aabb(origin, dir, c.pos, CUBE_HALF) {
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
        self.window.set_title(&title_for(self.mode));
        self.update_camera();
    }

    fn update(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        // Fly-camera movement.
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

        // Drain agent output (text + scene commands) since last frame.
        let mut deltas: Vec<String> = Vec::new();
        let mut commands: Vec<SceneCommand> = Vec::new();
        let mut claude_prompt: Option<String> = None;
        let mut done = false;
        let mut error: Option<String> = None;
        if let Some(rx) = &self.chat.rx {
            loop {
                match rx.try_recv() {
                    Ok(StreamMsg::Delta(t)) => deltas.push(t),
                    Ok(StreamMsg::Command(c)) => commands.push(c),
                    Ok(StreamMsg::ClaudePrompt(p)) => claude_prompt = Some(p),
                    Ok(StreamMsg::Done) => {
                        done = true;
                        break;
                    }
                    Ok(StreamMsg::Error(e)) => {
                        error = Some(e);
                        done = true;
                        break;
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        done = true;
                        break;
                    }
                }
            }
        }

        // Apply scene commands the agent issued.
        if !commands.is_empty() {
            for cmd in commands {
                match cmd {
                    SceneCommand::Add { x, z, color } => self.cubes.push(Cube {
                        pos: Vec3::new(x, CUBE_HALF, z),
                        color,
                    }),
                    SceneCommand::SetColor { index, color } => {
                        if index < self.cubes.len() {
                            self.cubes[index].color = color;
                        }
                    }
                    SceneCommand::Remove { index } => {
                        if index < self.cubes.len() {
                            self.cubes.remove(index);
                            self.selected = match self.selected {
                                Some(s) if s == index => None,
                                Some(s) if s > index => Some(s - 1),
                                other => other,
                            };
                        }
                    }
                    SceneCommand::Select { index } => {
                        if index < self.cubes.len() {
                            self.selected = Some(index);
                        }
                    }
                    SceneCommand::Clear => {
                        self.cubes.clear();
                        self.selected = None;
                    }
                }
            }
            self.rebuild_cubes();
        }

        if let Some(p) = claude_prompt {
            self.chat.pending_claude_prompt = Some(p);
        }

        if !deltas.is_empty() || done {
            if let Some(idx) = self.chat.streaming_index {
                if let Some(msg) = self.chat.messages.get_mut(idx) {
                    for d in &deltas {
                        msg.text.push_str(d);
                    }
                    if let Some(e) = &error {
                        let note = format!("[error] {e}");
                        if msg.text.is_empty() {
                            msg.text = note;
                        } else {
                            msg.text.push('\n');
                            msg.text.push_str(&note);
                        }
                    }
                }
            }
            if done {
                self.chat.rx = None;
                self.chat.streaming_index = None;
                self.chat.status.clear();
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

        // Fixed cinematic 3D background — independent of the UI theme/mode so scene
        // creation stays visually consistent.
        let clear = wgpu::Color { r: 0.055, g: 0.070, b: 0.095, a: 1.0 };

        // --- Pass 1: the 3D scene ---
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Scene Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear),
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

        // --- egui UI over the scene ---
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let ctx = self.egui_ctx.clone();
        let mode = self.mode;
        let logo = self.logo_texture.as_ref();
        let scene = SceneSnapshot {
            cubes: self.cubes.iter().map(|c| (c.pos, c.color)).collect(),
            selected: self.selected,
        };
        let chat = &mut self.chat;
        let ui_state = &mut self.ui;
        let full_output =
            ctx.run(raw_input, |c| build_ui(c, ui_state, chat, mode, logo, &scene));
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
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
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

/// Load the engine logo from `assets/icon.png` as RGBA8 bytes + dimensions.
/// Returns None if the file is missing or can't be decoded (the app still runs).
fn load_logo_rgba() -> Option<(Vec<u8>, u32, u32)> {
    let bytes = std::fs::read("assets/icon.png").ok()?;
    let img = image::load_from_memory(&bytes).ok()?;
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    Some((rgba.into_raw(), w, h))
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

        // Load the logo (assets/icon.png) for the window icon + the in-UI overlay.
        let logo = load_logo_rgba();

        let mut attributes = Window::default_attributes()
            .with_title(title_for(CameraMode::Orbit))
            .with_inner_size(LogicalSize::new(1280.0, 720.0));

        if let Some((rgba, w, h)) = &logo {
            if let Ok(icon) = winit::window::Icon::from_rgba(rgba.clone(), *w, *h) {
                attributes = attributes.with_window_icon(Some(icon));
            }
        }

        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .expect("failed to create the window"),
        );

        self.state = Some(State::new(window, logo));

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
                if state.ui.should_quit {
                    event_loop.exit();
                }
                return;
            }
            _ => {}
        }

        // Tab toggles camera mode; swallow it so egui can't focus the chat with it.
        if let WindowEvent::KeyboardInput { event: ke, .. } = &event {
            if ke.physical_key == PhysicalKey::Code(KeyCode::Tab) {
                if ke.state == ElementState::Pressed && !ke.repeat {
                    state.toggle_mode();
                }
                return;
            }
        }

        let egui_consumed = state
            .egui_state
            .on_window_event(&state.window, &event)
            .consumed;
        if egui_consumed {
            return;
        }

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
                        KeyCode::KeyH if first_press => {
                            state.ui.show_debug = !state.ui.show_debug
                        }
                        KeyCode::Escape if first_press => state.toggle_menu(),
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
