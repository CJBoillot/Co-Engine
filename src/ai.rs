//! CoE-AI chat: UI state, the Claude streaming/agent worker, and the chat tab.

use std::sync::mpsc::{Receiver, Sender};

use glam::Vec3;

use crate::mesh::{grid_slot, parse_color};
use crate::theme::ACCENT_GOLD;

/// A Claude model the in-engine chat can use. `effort` marks models that accept
/// the `output_config.effort` speed control (Haiku 4.5 does not).
pub(crate) struct ModelChoice {
    pub(crate) name: &'static str,
    pub(crate) id: &'static str,
    pub(crate) effort: bool,
}

/// Models offered in the chat's Model dropdown, fastest/cheapest first.
pub(crate) const MODELS: &[ModelChoice] = &[
    ModelChoice { name: "Haiku 4.5 — fastest", id: "claude-haiku-4-5", effort: false },
    ModelChoice { name: "Sonnet 4.6 — balanced", id: "claude-sonnet-4-6", effort: true },
    ModelChoice { name: "Opus 4.8 — most capable", id: "claude-opus-4-8", effort: true },
    ModelChoice { name: "Fable 5 — most powerful", id: "claude-fable-5", effort: true },
];

/// Speed presets → the API `effort` value (effort-capable models only).
pub(crate) const EFFORTS: &[(&str, &str)] = &[
    ("Fast", "low"),
    ("Balanced", "medium"),
    ("High", "high"),
    ("Max", "max"),
];

pub(crate) const DEFAULT_MODEL_IDX: usize = 0; // Haiku 4.5 (cheapest/fastest, default)
pub(crate) const DEFAULT_EFFORT_IDX: usize = 1; // Balanced

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Role {
    User,
    Assistant,
}

impl Role {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Role::User => "You",
            Role::Assistant => "CoE-AI",
        }
    }
    pub(crate) fn api(self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

pub(crate) struct ChatMessage {
    pub(crate) role: Role,
    pub(crate) text: String,
}

/// Messages sent from the streaming worker thread back to the UI.
pub(crate) enum StreamMsg {
    Delta(String),
    Command(SceneCommand),
    ClaudePrompt(String),
    /// CoE-AI wants to run a terminal command; the main thread asks the user to
    /// approve, runs it, and sends the captured output back via `reply`.
    CommandRequest { command: String, reply: Sender<String> },
    Done,
    Error(String),
}

/// A terminal command CoE-AI proposed, awaiting the user's approve/deny.
pub(crate) struct PendingCommand {
    pub(crate) command: String,
    pub(crate) reply: Sender<String>,
}

/// A mutation the AI agent wants applied to the scene (executed on the main thread).
pub(crate) enum SceneCommand {
    Add { x: f32, z: f32, color: [f32; 3] },
    AddSphere { x: f32, z: f32, color: [f32; 3] },
    SetColor { index: usize, color: [f32; 3] },
    Remove { index: usize },
    Select { index: usize },
    Clear,
}

/// Snapshot of the scene handed to the agent so it knows what exists.
pub(crate) struct SceneSnapshot {
    pub(crate) cubes: Vec<(Vec3, [f32; 3])>,
    pub(crate) selected: Option<usize>,
}

/// Everything the worker thread needs to run an agent turn.
pub(crate) struct AgentRequest {
    pub(crate) api_key: String,
    pub(crate) model_id: String,
    pub(crate) effort: Option<&'static str>,
    pub(crate) system: String,
    pub(crate) positions: Vec<(f32, f32)>,
    pub(crate) selected: Option<usize>,
    pub(crate) messages: Vec<serde_json::Value>,
}

#[derive(Default)]
pub(crate) struct ChatUi {
    pub(crate) messages: Vec<ChatMessage>,
    pub(crate) input: String,
    /// Receiver for the in-flight reply (None when idle).
    pub(crate) rx: Option<Receiver<StreamMsg>>,
    /// Index of the assistant message currently being streamed into.
    pub(crate) streaming_index: Option<usize>,
    /// A short status/error line shown under the header.
    pub(crate) status: String,
    /// Currently selected model + speed (indices into MODELS / EFFORTS).
    pub(crate) model_idx: usize,
    pub(crate) effort_idx: usize,
    /// A prompt CoE-AI prepared for the user to paste to Claude (Desktop).
    pub(crate) pending_claude_prompt: Option<String>,
    /// Whether the AI help modal (models & speed reference) is open.
    pub(crate) help_open: bool,
}

/// Find the Anthropic API key: the ANTHROPIC_API_KEY environment variable first,
/// then a local `.env` file (a line like `ANTHROPIC_API_KEY=sk-ant-...`). The
/// `.env` file is gitignored, so the key is never committed.
pub(crate) fn load_api_key() -> Option<String> {
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
pub(crate) fn send_message(chat: &mut ChatUi, scene: &SceneSnapshot) {
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
pub(crate) fn build_system_prompt(scene: &SceneSnapshot) -> String {
    let mut s = String::from(
        "You are CoE-AI, the assistant built into CoEngine, a 3D engine the user is building. \
You can DIRECTLY change the 3D scene using the provided tools — when the user asks for a scene change \
(add or recolor cubes and spheres, remove, or select objects), DO IT with the tools rather than describing code. After acting, \
confirm in one short sentence. You also have a `run_command` tool that runs a shell command in the \
engine's terminal (the user must approve each one) and returns its output — use it for git, file listing, \
builds, and similar tasks.\n\n",
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
pub(crate) fn tools_json() -> serde_json::Value {
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
            "name": "add_sphere",
            "description": "Add a sphere to the 3D scene (same size as a cube). Optionally give a color (a name like \"green\" or hex like \"#33cc44\") and an x/z position on the ground; if x/z are omitted the engine auto-places it.",
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
            "name": "set_sphere_color",
            "description": "Change the color of an existing object (cube or sphere), identified by its index.",
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
            "name": "run_command",
            "description": "Run a shell command in the engine's terminal (PowerShell/cmd, in the project folder) and get its output back. Use for git, file listing, builds, etc. Each command must be approved by the user before it runs. Output is captured and returned to you.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "The exact command line to run." }
                },
                "required": ["command"]
            }
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
pub(crate) fn execute_tool(
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
        "add_sphere" => {
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
            let _ = tx.send(StreamMsg::Command(SceneCommand::AddSphere { x, z, color }));
            format!("Added a {color_name} sphere as #{n} at x={x:.1}, z={z:.1}.")
        }
        "set_sphere_color" => {
            let idx = input.get("index").and_then(|v| v.as_u64()).unwrap_or(u64::MAX) as usize;
            let color_name = input.get("color").and_then(|c| c.as_str()).unwrap_or("orange");
            if idx < mirror.len() {
                let color = parse_color(color_name);
                let _ = tx.send(StreamMsg::Command(SceneCommand::SetColor { index: idx, color }));
                format!("Object #{idx} is now {color_name}.")
            } else {
                format!("There is no object #{idx} (the scene has {} objects).", mirror.len())
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
        "run_command" => {
            let command = input
                .get("command")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if command.is_empty() {
                return "No command was provided.".to_string();
            }
            // Hand the command to the main thread for approval + execution, then
            // block until it sends back the captured output (or a denial).
            let (reply, rx) = std::sync::mpsc::channel();
            if tx
                .send(StreamMsg::CommandRequest { command, reply })
                .is_err()
            {
                return "The engine isn't accepting commands right now.".to_string();
            }
            rx.recv()
                .unwrap_or_else(|_| "The command was cancelled.".to_string())
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
pub(crate) fn run_agent(req: AgentRequest, tx: Sender<StreamMsg>) {
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

/// Render the AI Chat content into a dock tab's Ui.
/// The short model name for the compact button (drops the "— descriptor" tail).
fn short_model_name(name: &str) -> &str {
    name.split(" — ").next().unwrap_or(name)
}

/// The AI help modal: a reference page for each model + speed, with a relative
/// token-cost rating and best-use guidance. A hub for more help pages later.
fn ai_help_modal(ctx: &egui::Context, open: &mut bool) {
    if !*open {
        return;
    }
    // (short name, cost rating, best for, description) — parallel to MODELS.
    const MODEL_HELP: &[(&str, &str, &str, &str)] = &[
        (
            "Haiku 4.5",
            "$",
            "Quick edits, simple scene ops, fast iteration",
            "Fastest and cheapest. Great for small, well-defined tasks. No speed control — always runs fast.",
        ),
        (
            "Sonnet 4.6",
            "$$",
            "Everyday building and most CoE-AI work",
            "A strong balance of speed, quality, and cost. The sensible default for general use.",
        ),
        (
            "Opus 4.8",
            "$$$",
            "Complex, multi-step reasoning and tricky problems",
            "Most capable for hard tasks. Slower and pricier than Sonnet.",
        ),
        (
            "Fable 5",
            "$$$$",
            "The most demanding creative / agentic work",
            "Most powerful, with the highest cost and latency. Reach for it when nothing else suffices.",
        ),
    ];
    const SPEED_HELP: &[(&str, &str)] = &[
        ("Fast", "Least thinking, fewest tokens — quickest and cheapest."),
        ("Balanced", "Moderate thinking. The default."),
        ("High", "More thorough reasoning, more tokens."),
        ("Max", "Maximum reasoning depth — slowest and most tokens."),
    ];

    let mut close = false;
    egui::Window::new("ai_help")
        .title_bar(false)
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .default_width(560.0)
        .show(ctx, |ui| {
            ui.set_min_width(560.0);
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.heading("AI Help");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Close").clicked() {
                        close = true;
                    }
                });
            });
            ui.separator();
            egui::ScrollArea::vertical()
                .max_height(440.0)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.add_space(4.0);
                    ui.label(egui::RichText::new("Models").strong().color(ACCENT_GOLD));
                    for (name, cost, best, desc) in MODEL_HELP {
                        ui.add_space(8.0);
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(*name).strong());
                            ui.label(egui::RichText::new(format!("token cost {cost}")).small().weak());
                        });
                        ui.label(egui::RichText::new(format!("Best for: {best}")).small());
                        ui.label(egui::RichText::new(*desc).small().weak());
                    }
                    ui.add_space(12.0);
                    ui.separator();
                    ui.label(
                        egui::RichText::new("Speed (thinking effort)")
                            .strong()
                            .color(ACCENT_GOLD),
                    );
                    ui.label(
                        egui::RichText::new(
                            "Controls how long the model reasons before answering. \
                             Applies to Sonnet, Opus, and Fable; Haiku always runs fast.",
                        )
                        .small()
                        .weak(),
                    );
                    for (name, desc) in SPEED_HELP {
                        ui.add_space(6.0);
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(*name).strong());
                            ui.label(egui::RichText::new(*desc).small().weak());
                        });
                    }
                    ui.add_space(8.0);
                });
            ui.add_space(6.0);
            if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                close = true;
            }
        });
    if close {
        *open = false;
    }
}

/// One row in the model/speed picker menu: full-width, subtle hover, name on the
/// left, a gold checkmark on the right when selected (Claude-style — no big bar).
fn model_menu_row(ui: &mut egui::Ui, text: &str, selected: bool) -> bool {
    let w = ui.available_width().max(180.0);
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(w, 28.0), egui::Sense::click());
    if resp.hovered() {
        ui.painter().rect_filled(
            rect,
            crate::theme::RADIUS,
            ui.visuals().widgets.hovered.weak_bg_fill,
        );
    }
    ui.painter().text(
        egui::pos2(rect.left() + 10.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        text,
        egui::FontId::proportional(13.5),
        ui.visuals().text_color(),
    );
    if selected {
        ui.painter().text(
            egui::pos2(rect.right() - 10.0, rect.center().y),
            egui::Align2::RIGHT_CENTER,
            crate::theme::icon::CHECK,
            egui::FontId::proportional(15.0),
            crate::theme::ACCENT_GOLD,
        );
    }
    resp.clicked()
}

pub(crate) fn chat_tab(ui: &mut egui::Ui, chat: &mut ChatUi, scene: &SceneSnapshot) {
    ai_help_modal(ui.ctx(), &mut chat.help_open);
    egui::TopBottomPanel::top("chat_header").show_inside(ui, |ui| {
                ui.add_space(6.0);
                ui.heading("AI Chat");
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
                    // Compact model picker (Claude-style): a small button showing just
                    // the short model name; the menu lists models and, for effort-
                    // capable models, the speed options + a Help entry.
                    let label = format!(
                        "{}  {}",
                        short_model_name(MODELS[chat.model_idx].name),
                        crate::theme::icon::CHEVRON_DOWN
                    );
                    ui.menu_button(egui::RichText::new(label).small(), |ui| {
                        ui.set_min_width(236.0);
                        ui.add_space(2.0);
                        ui.label(egui::RichText::new("Model").small().weak());
                        for (i, m) in MODELS.iter().enumerate() {
                            if model_menu_row(ui, m.name, i == chat.model_idx) {
                                chat.model_idx = i;
                                // No second choice (speed) → pick and close immediately.
                                if !MODELS[i].effort {
                                    ui.close_menu();
                                }
                            }
                        }
                        if MODELS[chat.model_idx].effort {
                            ui.add_space(4.0);
                            ui.separator();
                            ui.label(egui::RichText::new("Speed").small().weak());
                            for (i, e) in EFFORTS.iter().enumerate() {
                                if model_menu_row(ui, e.0, i == chat.effort_idx) {
                                    chat.effort_idx = i;
                                    ui.close_menu();
                                }
                            }
                        }
                        ui.add_space(4.0);
                        ui.separator();
                        if model_menu_row(
                            ui,
                            &format!("{}  Help — models & speed", crate::theme::icon::INSPECTOR),
                            false,
                        ) {
                            chat.help_open = true;
                            ui.close_menu();
                        }
                        ui.add_space(2.0);
                    })
                    .response
                    .on_hover_text("Model & speed");

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
}
