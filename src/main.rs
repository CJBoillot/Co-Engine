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

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::TryRecvError;
use std::time::{Duration, Instant};

use glam::{EulerRot, Mat4, Quat, Vec3, Vec4};
use serde::{Deserialize, Serialize};
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition, PhysicalSize};
use winit::event::{
    DeviceEvent, DeviceId, ElementState, MouseButton, MouseScrollDelta, WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use egui_dock::{DockArea, DockState, NodeIndex, Style, TabViewer};

mod theme;
use theme::{ACCENT_GOLD, Theme, theme_visuals};

mod mesh;
use mesh::*;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// CoSemVer display version (see memory: CoSemVer). A trailing letter marks a
/// bug fix for this version. Kept separate from Cargo's strict-SemVer `version`.
pub(crate) const CO_VERSION: &str = "0.0.18";

mod ai;
use ai::*;

mod terminal;
use terminal::*;

mod git;
use git::*;

mod editor;
use editor::*;

mod camera;
use camera::*;

mod controls;
use controls::*;

mod project;
use project::*;

/// The dockable widgets/windows of the engine workspace. Not `Copy` because
/// `File` carries a path.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub(crate) enum DockTab {
    Scene,
    Logic,
    /// Inspector / properties panel for the selected object.
    Inspector,
    /// Project file tree (IDE-style). `alias = "Code"` keeps v0.0.12 layouts
    /// (which serialized this tab as "Code") loadable.
    #[serde(alias = "Code")]
    Explorer,
    AiChat,
    Log,
    /// In-engine terminal (a live shell in a PTY).
    Terminal,
    /// Git / Source Control panel.
    Git,
    /// Project-wide search panel.
    Search,
    /// A file viewer opened from the Explorer; the tab is titled by file name and
    /// shows the file's text/code, or the image if it's an image.
    File(PathBuf),
}




/// Which Settings category/submenu is showing.
#[derive(Clone, Copy, PartialEq)]
enum SettingsTab {
    Theme,
    Controls,
    Terminal,
}

/// UI/chrome state: theme, menu/modal visibility, active tab, controls overlay.
struct UiState {
    theme: Theme,
    dark_mode: bool,
    /// Which shell the in-engine terminal launches.
    shell: Shell,
    menu_open: bool,
    settings_open: bool,
    /// Which Settings submenu (Theme / Controls) is showing.
    settings_tab: SettingsTab,
    show_debug: bool,
    should_quit: bool,
    /// When Some, the next key press rebinds this action (Settings -> Controls).
    rebinding: Option<ControlAction>,
    /// One-shot requests raised by the Log tab's Undo / Redo buttons.
    undo_requested: bool,
    redo_requested: bool,
    /// One-shot project requests raised by the Menu (processed in `update`).
    save_requested: bool,
    save_as_requested: bool,
    open_requested: bool,
    new_requested: bool,
    /// Startup popup ("Open last used project?") + its one-shot button requests.
    show_startup_popup: bool,
    open_last_requested: bool,
    /// Display name of the last-used project (for the startup popup label).
    last_project_name: Option<String>,
    /// Transient status line shown in the Menu after a save/open (e.g. errors).
    project_status: Option<String>,
    /// One-shot requests from the scene outliner (Logic tab), serviced in `update`.
    outliner_select: Option<usize>,
    outliner_delete: Option<usize>,
    outliner_add: bool,
    /// Inspector edit requests: the edited entity (live) + a commit flag (undo).
    inspector_apply: Option<Entity>,
    inspector_commit: bool,
    /// Gizmo-mode switch requested from the on-screen toolbar.
    gizmo_mode_req: Option<GizmoMode>,
    /// Uniform scale requested from the on-screen scale slider (live), plus a
    /// commit flag (record one undo entry when the slider interaction ends).
    scale_req: Option<f32>,
    scale_commit: bool,
    /// Pending Explorer file op (new/rename/delete), shown as a modal until
    /// confirmed or canceled. `fs_prompt_error` shows a failed op's message.
    fs_prompt: Option<FsPrompt>,
    fs_prompt_error: Option<String>,
    /// Caret (line, col) of the focused file editor, for the status bar. Captured
    /// during the dock pass and shown on the next frame (one-frame lag).
    editor_cursor: Option<(usize, usize)>,
    /// In-file Find/Replace bar state + a one-shot "go to this match" range.
    find: FindState,
    find_goto: Option<(usize, usize)>,
    /// Project-wide Search tab state.
    search: SearchUi,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            theme: Theme::DefaultSimple,
            dark_mode: true,
            shell: Shell::default(),
            menu_open: false,
            settings_open: false,
            settings_tab: SettingsTab::Theme,
            show_debug: true,
            should_quit: false,
            rebinding: None,
            undo_requested: false,
            redo_requested: false,
            save_requested: false,
            save_as_requested: false,
            open_requested: false,
            new_requested: false,
            show_startup_popup: false,
            open_last_requested: false,
            last_project_name: None,
            project_status: None,
            outliner_select: None,
            outliner_delete: None,
            outliner_add: false,
            inspector_apply: None,
            inspector_commit: false,
            gizmo_mode_req: None,
            scale_req: None,
            scale_commit: false,
            fs_prompt: None,
            fs_prompt_error: None,
            editor_cursor: None,
            find: FindState::default(),
            find_goto: None,
            search: SearchUi::default(),
        }
    }
}

/// Short display label for a dock tab (used by the Focus-mode minimized list).
fn dock_tab_label(tab: &DockTab) -> String {
    match tab {
        DockTab::Scene => "3D Scene".to_string(),
        DockTab::Logic => "Objects Added".to_string(),
        DockTab::Inspector => "Inspector".to_string(),
        DockTab::Explorer => "Explorer".to_string(),
        DockTab::AiChat => "AI Chat".to_string(),
        DockTab::Log => "Log".to_string(),
        DockTab::Terminal => "Terminal".to_string(),
        DockTab::Git => "Git".to_string(),
        DockTab::Search => "Search".to_string(),
        DockTab::File(p) => p
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| p.display().to_string()),
    }
}

/// A small hand-drawn trash-can button (egui's default fonts lack a glyph for
/// it). Returns true on click.
fn trash_button(ui: &mut egui::Ui) -> bool {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(20.0, 20.0), egui::Sense::click());
    let hov = resp.hovered();
    let col = if hov {
        egui::Color32::from_rgb(228, 120, 110)
    } else {
        egui::Color32::from_gray(150)
    };
    let p = ui.painter();
    if hov {
        p.rect_filled(rect, 3.0, egui::Color32::from_black_alpha(60));
    }
    let c = rect.center();
    let (w, h) = (9.0_f32, 11.0_f32);
    let s = egui::Stroke::new(1.4, col);
    let lid_y = c.y - h * 0.42;
    // lid + handle
    p.line_segment([egui::pos2(c.x - w * 0.6, lid_y), egui::pos2(c.x + w * 0.6, lid_y)], s);
    p.line_segment(
        [egui::pos2(c.x - w * 0.22, lid_y - 2.0), egui::pos2(c.x + w * 0.22, lid_y - 2.0)],
        s,
    );
    p.line_segment([egui::pos2(c.x - w * 0.22, lid_y - 2.0), egui::pos2(c.x - w * 0.22, lid_y)], s);
    p.line_segment([egui::pos2(c.x + w * 0.22, lid_y - 2.0), egui::pos2(c.x + w * 0.22, lid_y)], s);
    // tapered body
    let (top_y, bot_y) = (lid_y + 1.5, c.y + h * 0.5);
    let (tlx, trx) = (c.x - w * 0.46, c.x + w * 0.46);
    let (blx, brx) = (c.x - w * 0.34, c.x + w * 0.34);
    p.line_segment([egui::pos2(tlx, top_y), egui::pos2(blx, bot_y)], s);
    p.line_segment([egui::pos2(trx, top_y), egui::pos2(brx, bot_y)], s);
    p.line_segment([egui::pos2(blx, bot_y), egui::pos2(brx, bot_y)], s);
    // ribs
    for fx in [-0.18, 0.0, 0.18] {
        let x = c.x + w * fx;
        p.line_segment(
            [egui::pos2(x, top_y + 1.5), egui::pos2(x, bot_y - 1.5)],
            egui::Stroke::new(1.0, col),
        );
    }
    resp.on_hover_text("Delete object").clicked()
}

/// Which icon a gizmo-toolbar button draws.
#[derive(Clone, Copy)]
enum ToolIcon {
    Move,
    Rotate,
    Scale,
    Delete,
}

/// A WoW-Housing-style icon button for the on-screen gizmo toolbar. Drawn by
/// hand (egui has no icon font). Highlights when active or hovered.
fn tool_button(ui: &mut egui::Ui, icon: ToolIcon, active: bool) -> bool {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(40.0, 40.0), egui::Sense::click());
    let hov = resp.hovered();
    let bg = if active {
        egui::Color32::from_rgb(70, 56, 30)
    } else if hov {
        egui::Color32::from_rgb(44, 46, 54)
    } else {
        egui::Color32::from_rgba_unmultiplied(30, 32, 40, 200)
    };
    let p = ui.painter();
    p.rect(
        rect,
        egui::Rounding::same(6.0),
        bg,
        egui::Stroke::new(
            1.0,
            if active {
                ACCENT_GOLD
            } else {
                egui::Color32::from_gray(70)
            },
        ),
    );
    let col = if active {
        ACCENT_GOLD
    } else {
        egui::Color32::from_gray(210)
    };
    let c = rect.center();
    let s = egui::Stroke::new(1.8, col);
    match icon {
        ToolIcon::Move => {
            for (dx, dy) in [(0.0, -1.0), (0.0, 1.0), (-1.0, 0.0), (1.0, 0.0)] {
                let dir = egui::vec2(dx, dy);
                let perp = egui::vec2(-dy, dx);
                let tip = c + dir * 9.0;
                let base = tip - dir * 4.0;
                p.line_segment([c, tip], s);
                p.line_segment([tip, base + perp * 3.0], s);
                p.line_segment([tip, base - perp * 3.0], s);
            }
        }
        ToolIcon::Rotate => {
            let r = 8.0;
            let mut prev: Option<egui::Pos2> = None;
            let (a0, a1) = (-2.2_f32, 2.6_f32);
            for i in 0..=22 {
                let a = a0 + (a1 - a0) * i as f32 / 22.0;
                let pt = c + egui::vec2(a.cos() * r, a.sin() * r);
                if let Some(pp) = prev {
                    p.line_segment([pp, pt], s);
                }
                prev = Some(pt);
            }
            // arrowhead at the arc end
            let tip = c + egui::vec2(a1.cos() * r, a1.sin() * r);
            let tang = egui::vec2(-a1.sin(), a1.cos());
            let radial = egui::vec2(a1.cos(), a1.sin());
            p.line_segment([tip, tip - tang * 4.0 + radial * 3.0], s);
            p.line_segment([tip, tip - tang * 4.0 - radial * 3.0], s);
        }
        ToolIcon::Scale => {
            // A small square with a diagonal expand arrow.
            let r = 6.5;
            p.rect_stroke(
                egui::Rect::from_center_size(c, egui::vec2(r * 2.0, r * 2.0)),
                1.0,
                s,
            );
            let tip = c + egui::vec2(r + 3.0, -(r + 3.0));
            p.line_segment([c, tip], s);
            p.line_segment([tip, tip + egui::vec2(-4.0, 0.0)], s);
            p.line_segment([tip, tip + egui::vec2(0.0, 4.0)], s);
        }
        ToolIcon::Delete => {
            let (w, h) = (9.0_f32, 11.0_f32);
            let lid_y = c.y - h * 0.42;
            p.line_segment([egui::pos2(c.x - w * 0.6, lid_y), egui::pos2(c.x + w * 0.6, lid_y)], s);
            let (top_y, bot_y) = (lid_y + 1.5, c.y + h * 0.5);
            let (tlx, trx) = (c.x - w * 0.46, c.x + w * 0.46);
            let (blx, brx) = (c.x - w * 0.34, c.x + w * 0.34);
            p.line_segment([egui::pos2(tlx, top_y), egui::pos2(blx, bot_y)], s);
            p.line_segment([egui::pos2(trx, top_y), egui::pos2(brx, bot_y)], s);
            p.line_segment([egui::pos2(blx, bot_y), egui::pos2(brx, bot_y)], s);
        }
    }
    resp.clicked()
}

/// A small triangle arrow button (◄ / ►) for the scale slider ends.
fn arrow_button(ui: &mut egui::Ui, right: bool) -> bool {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(16.0, 20.0), egui::Sense::click());
    let col = if resp.hovered() {
        ACCENT_GOLD
    } else {
        egui::Color32::from_gray(190)
    };
    let c = rect.center();
    let dx = if right { 4.0 } else { -4.0 };
    ui.painter().add(egui::Shape::convex_polygon(
        vec![
            egui::pos2(c.x + dx, c.y),
            egui::pos2(c.x - dx, c.y - 5.0),
            egui::pos2(c.x - dx, c.y + 5.0),
        ],
        col,
        egui::Stroke::NONE,
    ));
    resp.clicked()
}

/// WoW-Housing-style horizontal scale slider: a segmented track with a knob, end
/// arrows, and a % readout. Edits drive `req` (live) and `commit` (on release).
fn scale_slider(ui: &mut egui::Ui, value: f32, req: &mut Option<f32>, commit: &mut bool) {
    let frac = ((value - SCALE_MIN) / (SCALE_MAX - SCALE_MIN)).clamp(0.0, 1.0);
    ui.horizontal(|ui| {
        if arrow_button(ui, false) {
            *req = Some((value - 0.05).clamp(SCALE_MIN, SCALE_MAX));
            *commit = true;
        }
        let (rect, resp) =
            ui.allocate_exact_size(egui::vec2(210.0, 18.0), egui::Sense::click_and_drag());
        let p = ui.painter();
        let track = egui::Rect::from_center_size(rect.center(), egui::vec2(rect.width(), 8.0));
        p.rect_filled(track, 4.0, egui::Color32::from_rgb(40, 42, 50));
        // filled portion
        let fill = egui::Rect::from_min_size(
            track.min,
            egui::vec2(track.width() * frac, track.height()),
        );
        p.rect_filled(fill, 4.0, egui::Color32::from_rgb(150, 120, 60));
        // segment ticks
        for i in 1..10 {
            let x = track.left() + track.width() * i as f32 / 10.0;
            p.line_segment(
                [egui::pos2(x, track.top()), egui::pos2(x, track.bottom())],
                egui::Stroke::new(1.0, egui::Color32::from_black_alpha(70)),
            );
        }
        // knob (diamond)
        let kx = track.left() + track.width() * frac;
        let kc = egui::pos2(kx, track.center().y);
        p.add(egui::Shape::convex_polygon(
            vec![
                egui::pos2(kc.x, kc.y - 8.0),
                egui::pos2(kc.x + 7.0, kc.y),
                egui::pos2(kc.x, kc.y + 8.0),
                egui::pos2(kc.x - 7.0, kc.y),
            ],
            ACCENT_GOLD,
            egui::Stroke::new(1.0, egui::Color32::from_gray(30)),
        ));
        if resp.dragged() || resp.clicked() {
            if let Some(pos) = resp.interact_pointer_pos() {
                let f = ((pos.x - track.left()) / track.width()).clamp(0.0, 1.0);
                *req = Some(SCALE_MIN + f * (SCALE_MAX - SCALE_MIN));
            }
        }
        if resp.drag_stopped() {
            *commit = true;
        }
        if arrow_button(ui, true) {
            *req = Some((value + 0.05).clamp(SCALE_MIN, SCALE_MAX));
            *commit = true;
        }
    });
    ui.label(
        egui::RichText::new(format!("{:.0}%", value * 100.0))
            .strong()
            .color(ACCENT_GOLD),
    );
}

/// Scene outliner (the Logic tab): a flat list of the scene's objects with
/// select / delete, plus an add button. Requests flow back to `State` via the
/// out-params (serviced in `update`). Rename + properties come from the inspector.
fn outliner_ui(
    ui: &mut egui::Ui,
    names: &[String],
    selected: Option<usize>,
    select_req: &mut Option<usize>,
    delete_req: &mut Option<usize>,
    add_req: &mut bool,
) {
    egui::TopBottomPanel::top("outliner_header").show_inside(ui, |ui| {
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.heading("Scene");
            ui.label(egui::RichText::new(format!("· {} objects", names.len())).weak());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("+ Cube").on_hover_text("Add a cube (C)").clicked() {
                    *add_req = true;
                }
            });
        });
        ui.add_space(4.0);
    });
    egui::CentralPanel::default().show_inside(ui, |ui| {
        if names.is_empty() {
            ui.add_space(8.0);
            ui.weak("No objects yet — press C or “+ Cube”.");
            return;
        }
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for (i, name) in names.iter().enumerate() {
                    ui.horizontal(|ui| {
                        if ui
                            .selectable_label(Some(i) == selected, name)
                            .clicked()
                        {
                            *select_req = Some(i);
                        }
                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                if trash_button(ui) {
                                    *delete_req = Some(i);
                                }
                            },
                        );
                    });
                }
            });
    });
}

/// One labelled X/Y/Z row of drag values editing a `Vec3`. Returns true if any
/// component changed this frame; sets `*ended` when a drag/edit finishes.
fn vec3_row(ui: &mut egui::Ui, label: &str, v: &mut glam::Vec3, speed: f32, ended: &mut bool) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.add_sized([62.0, 18.0], egui::Label::new(label));
        for comp in [&mut v.x, &mut v.y, &mut v.z] {
            let r = ui.add(egui::DragValue::new(comp).speed(speed).fixed_decimals(2));
            changed |= r.changed();
            if r.drag_stopped() || r.lost_focus() {
                *ended = true;
            }
        }
    });
    changed
}

/// Inspector / properties panel: edit the selected entity's name + transform +
/// color. Edits flow back to `State` via `apply` (live) and `commit` (records an
/// undo entry once the edit finishes).
fn inspector_ui(
    ui: &mut egui::Ui,
    entity: Option<&Entity>,
    apply: &mut Option<Entity>,
    commit: &mut bool,
) {
    let Some(src) = entity else {
        ui.centered_and_justified(|ui| {
            ui.label(egui::RichText::new("Select an object to edit its properties.").weak());
        });
        return;
    };
    let mut e = src.clone();
    let mut changed = false;
    let mut ended = false;

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.add_space(6.0);
            ui.heading("Inspector");
            ui.separator();
            ui.horizontal(|ui| {
                ui.add_sized([62.0, 18.0], egui::Label::new("Name"));
                let r = ui.text_edit_singleline(&mut e.name);
                changed |= r.changed();
                if r.lost_focus() {
                    ended = true;
                }
            });
            ui.add_space(8.0);
            ui.label(egui::RichText::new("Transform").strong().color(ACCENT_GOLD));
            changed |= vec3_row(ui, "Position", &mut e.pos, 0.05, &mut ended);
            changed |= vec3_row(ui, "Rotation°", &mut e.rotation, 0.5, &mut ended);
            changed |= vec3_row(ui, "Scale", &mut e.scale, 0.02, &mut ended);
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.add_sized([62.0, 18.0], egui::Label::new("Color"));
                let r = ui.color_edit_button_rgb(&mut e.color);
                if r.changed() {
                    changed = true;
                    ended = true;
                }
            });
        });

    if changed {
        *apply = Some(e);
    }
    if ended {
        *commit = true;
    }
}

/// Project a world point into the viewport's screen space, or None if behind
/// the camera.
fn project_to_screen(w: glam::Vec3, view_proj: glam::Mat4, rect: egui::Rect) -> Option<egui::Pos2> {
    let clip = view_proj * w.extend(1.0);
    if clip.w <= 0.001 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    Some(egui::pos2(
        rect.left() + (ndc.x * 0.5 + 0.5) * rect.width(),
        rect.top() + (1.0 - ndc.y) * 0.5 * rect.height(),
    ))
}

/// World length of each gizmo axis arrow (also used by the engine's hit-testing).
pub(crate) const GIZMO_AXIS_LEN: f32 = 1.3;

/// The three gizmo axis colors (X red, Y green, Z blue).
const AXIS_COLORS: [egui::Color32; 3] = [
    egui::Color32::from_rgb(222, 74, 64),
    egui::Color32::from_rgb(96, 200, 96),
    egui::Color32::from_rgb(82, 132, 236),
];

/// Draw the translate gizmo: three world-axis arrows (with arrowheads) for the
/// object at `obj_pos`. While an axis is being dragged (`active = Some`), only
/// that axis is shown. Drawing only — the drag is handled by the engine's input.
fn draw_translate_gizmo(
    painter: &egui::Painter,
    rect: egui::Rect,
    view_proj: glam::Mat4,
    obj_pos: glam::Vec3,
    active: Option<usize>,
) {
    let Some(center) = project_to_screen(obj_pos, view_proj, rect) else {
        return;
    };
    for (ai, dir) in [glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z].iter().enumerate() {
        if active.is_some() && active != Some(ai) {
            continue; // hide the other axes while dragging one
        }
        let Some(tip) = project_to_screen(obj_pos + *dir * GIZMO_AXIS_LEN, view_proj, rect)
        else {
            continue;
        };
        let color = AXIS_COLORS[ai];
        let hot = active == Some(ai);
        painter.line_segment([center, tip], egui::Stroke::new(if hot { 4.0 } else { 3.0 }, color));
        // Arrowhead (filled triangle pointing outward along the axis).
        let d = tip - center;
        let dn = d / d.length().max(1.0);
        let perp = egui::vec2(-dn.y, dn.x);
        let size = if hot { 13.0 } else { 11.0 };
        let base = tip - dn * size;
        painter.add(egui::Shape::convex_polygon(
            vec![tip, base + perp * (size * 0.5), base - perp * (size * 0.5)],
            color,
            egui::Stroke::NONE,
        ));
    }
    painter.circle_filled(center, 4.0, egui::Color32::from_gray(235));
}

/// The two in-plane basis vectors of the rotation ring for `axis` (the plane
/// perpendicular to that world axis).
fn ring_basis(axis: usize) -> (glam::Vec3, glam::Vec3) {
    // Each (u, v) is right-handed with its axis (u × v = axis), so +rotation
    // increases the in-plane angle uniformly — without this the Y ring inverts.
    match axis {
        0 => (glam::Vec3::Y, glam::Vec3::Z), // Y × Z = X
        1 => (glam::Vec3::Z, glam::Vec3::X), // Z × X = Y
        _ => (glam::Vec3::X, glam::Vec3::Y), // X × Y = Z
    }
}

/// Local anchor direction of each axis's ball handle, in the object's frame.
/// The ball is rigidly attached here (so it moves with the whole object).
const ROT_ANCHOR: [glam::Vec3; 3] = [glam::Vec3::Y, glam::Vec3::Z, glam::Vec3::X];

/// The object's rotation as a quaternion (euler degrees → quat).
fn obj_quat(rot_deg: glam::Vec3) -> glam::Quat {
    glam::Quat::from_euler(
        glam::EulerRot::XYZ,
        rot_deg.x.to_radians(),
        rot_deg.y.to_radians(),
        rot_deg.z.to_radians(),
    )
}

/// World position of the ball handle for `axis` — its local anchor rigidly
/// transformed by the object's full rotation (so it sticks to the object).
fn rotate_ball_world(axis: usize, center: glam::Vec3, rot_deg: glam::Vec3) -> glam::Vec3 {
    center + obj_quat(rot_deg) * (ROT_ANCHOR[axis] * GIZMO_AXIS_LEN)
}

/// Project to screen + return clip-space `w` (a camera-distance proxy for depth).
fn project_depth(w: glam::Vec3, view_proj: glam::Mat4, rect: egui::Rect) -> Option<(egui::Pos2, f32)> {
    let clip = view_proj * w.extend(1.0);
    if clip.w <= 0.001 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    Some((
        egui::pos2(
            rect.left() + (ndc.x * 0.5 + 0.5) * rect.width(),
            rect.top() + (1.0 - ndc.y) * 0.5 * rect.height(),
        ),
        clip.w,
    ))
}

/// A color dimmed (more transparent) to read as "behind the object".
fn dim(c: egui::Color32, a: u8) -> egui::Color32 {
    let [r, g, b, _] = c.to_array();
    egui::Color32::from_rgba_unmultiplied(r, g, b, a)
}

/// Draw the rotate gizmo: three rings centered on the object and tilted with it
/// (object-local), each carrying a ball handle. The half of each ring behind the
/// object is dimmed so the object reads as occluding it. While dragging
/// (`active = Some`), only that axis is shown.
fn draw_rotate_gizmo(
    painter: &egui::Painter,
    rect: egui::Rect,
    view_proj: glam::Mat4,
    obj_pos: glam::Vec3,
    rot_deg: glam::Vec3,
    active: Option<usize>,
) {
    let q = obj_quat(rot_deg);
    let center_depth = project_depth(obj_pos, view_proj, rect).map_or(1.0, |(_, d)| d);
    for axis in 0..3 {
        if active.is_some() && active != Some(axis) {
            continue;
        }
        let color = AXIS_COLORS[axis];
        let hot = active == Some(axis);
        let (ul, vl) = ring_basis(axis);
        let (u, v) = (q * ul, q * vl);
        let mut prev: Option<(egui::Pos2, f32)> = None;
        for i in 0..=72 {
            let a = i as f32 / 72.0 * std::f32::consts::TAU;
            let p = obj_pos + (u * a.cos() + v * a.sin()) * GIZMO_AXIS_LEN;
            let cur = project_depth(p, view_proj, rect);
            if let (Some((pa, da)), Some((pb, db))) = (prev, cur) {
                let behind = (da + db) * 0.5 > center_depth + 0.02;
                let c = if behind { dim(color, 70) } else { color };
                painter.line_segment([pa, pb], egui::Stroke::new(if hot { 3.0 } else { 2.0 }, c));
            }
            prev = cur;
        }
        // Ball handle, stuck to the object.
        if let Some((b, bd)) = project_depth(obj_pos + q * (ROT_ANCHOR[axis] * GIZMO_AXIS_LEN), view_proj, rect) {
            let behind = bd > center_depth + 0.02;
            let r = if hot { 11.0 } else { 8.5 };
            painter.circle_filled(b, r, if behind { dim(color, 120) } else { color });
            painter.circle_filled(b, r * 0.4, egui::Color32::from_gray(245));
        }
    }
    if let Some(center) = project_to_screen(obj_pos, view_proj, rect) {
        painter.circle_filled(center, 4.0, egui::Color32::from_gray(235));
    }
}

/// Draw the scale gizmo: three world-axis stalks with square handles (uniform
/// scale). While dragging (`active = Some`), only that axis is shown.
fn draw_scale_gizmo(
    painter: &egui::Painter,
    rect: egui::Rect,
    view_proj: glam::Mat4,
    obj_pos: glam::Vec3,
    active: Option<usize>,
) {
    let Some(center) = project_to_screen(obj_pos, view_proj, rect) else {
        return;
    };
    for (ai, dir) in [glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z].iter().enumerate() {
        if active.is_some() && active != Some(ai) {
            continue;
        }
        let Some(tip) = project_to_screen(obj_pos + *dir * GIZMO_AXIS_LEN, view_proj, rect)
        else {
            continue;
        };
        let color = AXIS_COLORS[ai];
        let hot = active == Some(ai);
        painter.line_segment([center, tip], egui::Stroke::new(if hot { 4.0 } else { 3.0 }, color));
        let h = if hot { 9.0 } else { 7.0 };
        painter.rect_filled(egui::Rect::from_center_size(tip, egui::vec2(h, h)), 1.0, color);
    }
    painter.circle_filled(center, 4.0, egui::Color32::from_gray(235));
}

/// Draw an eyeball (almond sclera + iris + pupil) centered at `c`. Muted by
/// default so it doesn't overpower the tab title; brighter on hover; gold when
/// Focus is active.
fn draw_eye(painter: &egui::Painter, c: egui::Pos2, active: bool, hovered: bool) {
    let (rx, ry) = (6.6, 4.1);
    let n = 26;
    let pts: Vec<egui::Pos2> = (0..n)
        .map(|i| {
            let a = i as f32 / n as f32 * std::f32::consts::TAU;
            egui::pos2(c.x + rx * a.cos(), c.y + ry * a.sin())
        })
        .collect();
    let (sclera, iris, outline, highlight) = if active {
        (
            egui::Color32::from_rgb(74, 64, 34),
            ACCENT_GOLD,
            egui::Color32::from_rgb(120, 100, 40),
            true,
        )
    } else if hovered {
        (
            egui::Color32::from_rgb(205, 207, 214),
            egui::Color32::from_rgb(90, 150, 235),
            egui::Color32::from_gray(40),
            true,
        )
    } else {
        // Dim/muted so the eye reads as a subtle icon, not a bright eyeball.
        (
            egui::Color32::from_rgb(72, 76, 86),
            egui::Color32::from_rgb(104, 116, 138),
            egui::Color32::from_gray(58),
            false,
        )
    };
    painter.add(egui::Shape::convex_polygon(
        pts,
        sclera,
        egui::Stroke::new(1.0, outline),
    ));
    painter.circle_filled(c, 2.9, iris);
    painter.circle_filled(c, 1.4, egui::Color32::from_rgb(16, 16, 20));
    if highlight {
        painter.circle_filled(c + egui::vec2(-1.0, -1.0), 0.6, egui::Color32::from_white_alpha(210));
    }
}

/// Per-tab content router for the dockable workspace.
struct EngineTabs<'a> {
    chat: &'a mut ChatUi,
    scene: &'a SceneSnapshot,
    history: &'a [HistoryEntry],
    history_cursor: usize,
    undo_req: &'a mut bool,
    redo_req: &'a mut bool,
    scene_rect_out: &'a mut Option<egui::Rect>,
    /// Root of the Explorer tree (the open project folder; None = no project).
    project_root: Option<PathBuf>,
    /// Set when the user clicks a file in the Explorer (handled after the dock
    /// pass, since adding a tab needs `&mut DockState`).
    open_file_req: &'a mut Option<PathBuf>,
    /// Set when the user picks an Explorer right-click op (new/rename/delete);
    /// handled after the dock pass (it opens a modal / mutates the dock state).
    fs_req: &'a mut Option<FsPrompt>,
    /// Set when the user drags a tree item onto a folder: `(src, dest_dir)`.
    fs_move_req: &'a mut Option<(PathBuf, PathBuf)>,
    /// Decoded file contents for `File` viewer tabs, keyed by path (lazy-loaded).
    file_cache: &'a mut HashMap<PathBuf, FileView>,
    /// The in-engine terminal's shell session (lazily started).
    terminal: &'a mut TerminalState,
    /// The shell executable to launch for the terminal.
    terminal_shell: String,
    /// Git panel state.
    git: &'a mut GitUi,
    /// Set to a tab when its eye (Focus toggle) is clicked; handled after the pass.
    focus_req: &'a mut Option<DockTab>,
    /// Set to a tab when its custom close X is clicked; handled after the pass.
    close_req: &'a mut Option<DockTab>,
    /// Whether Focus mode is currently active (colors the eye, flips its action).
    in_focus: bool,
    /// Scene outliner (Logic tab): entity names + selection + request out-params.
    outliner: &'a [String],
    outliner_select: &'a mut Option<usize>,
    outliner_delete: &'a mut Option<usize>,
    outliner_add: &'a mut bool,
    /// Inspector: the selected entity (clone) + edit out-params.
    inspector: Option<Entity>,
    inspector_apply: &'a mut Option<Entity>,
    inspector_commit: &'a mut bool,
    /// Out-param: caret (line, col) of the focused file editor (for the status bar).
    cursor_out: &'a mut Option<(usize, usize)>,
    /// Project-wide Search tab state.
    search: &'a mut SearchUi,
    /// The active focused file (so only its editor consumes a Find "go to match").
    active_file: Option<PathBuf>,
    /// A Find match to scroll/select in the active editor (one-shot).
    find_goto: &'a mut Option<(usize, usize)>,
    /// The live Find query to highlight in the active editor (None = bar closed).
    find_query: Option<String>,
}

impl TabViewer for EngineTabs<'_> {
    type Tab = DockTab;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        // Trailing spaces reserve room for the Focus eye + close X that
        // `on_tab_button` draws over the right end of the tab.
        const PAD: &str = "          ";
        let dirty = matches!(tab, DockTab::File(p) if self.file_cache.get(p).is_some_and(FileView::is_dirty));
        let base = dock_tab_label(tab);
        // VS Code-style unsaved indicator: a dot before the name.
        if dirty {
            format!("● {base}{PAD}").into()
        } else {
            format!("{base}{PAD}").into()
        }
    }

    /// The 3D Scene tab is transparent so the wgpu viewport renders through it.
    fn clear_background(&self, tab: &Self::Tab) -> bool {
        !matches!(tab, DockTab::Scene)
    }

    /// We draw our own close X (next to the Focus eye), so suppress egui_dock's.
    fn closeable(&mut self, _tab: &mut Self::Tab) -> bool {
        false
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        match tab {
            DockTab::Scene => {
                // Capture the viewport rect; body stays empty so the 3D shows through.
                *self.scene_rect_out = Some(ui.max_rect());
            }
            DockTab::Logic => {
                outliner_ui(
                    ui,
                    self.outliner,
                    self.scene.selected,
                    self.outliner_select,
                    self.outliner_delete,
                    self.outliner_add,
                );
            }
            DockTab::Inspector => {
                inspector_ui(
                    ui,
                    self.inspector.as_ref(),
                    self.inspector_apply,
                    self.inspector_commit,
                );
            }
            DockTab::Explorer => match &self.project_root {
                Some(root) => {
                    file_tree_ui(ui, root, self.open_file_req, self.fs_req, self.fs_move_req)
                }
                None => {
                    ui.centered_and_justified(|ui| {
                        ui.label(
                            egui::RichText::new("No project open\nMenu → New / Open Project")
                                .weak(),
                        );
                    });
                }
            },
            DockTab::AiChat => chat_tab(ui, self.chat, self.scene),
            DockTab::Log => log_tab(
                ui,
                self.history,
                self.history_cursor,
                self.undo_req,
                self.redo_req,
            ),
            DockTab::Terminal => {
                terminal_tab_ui(
                    ui,
                    self.terminal,
                    self.project_root.as_deref(),
                    &self.terminal_shell,
                );
            }
            DockTab::Git => {
                git_tab_ui(ui, self.git, self.project_root.as_deref());
            }
            DockTab::Search => {
                search_tab(ui, self.search, self.project_root.as_deref(), self.open_file_req);
            }
            DockTab::File(path) => {
                let lang = language_for(path);
                // Only the active file consumes a pending Find "go to match" and
                // shows live match highlights.
                let is_active = self.active_file.as_deref() == Some(path.as_path());
                let mut none = None;
                let goto = if is_active { &mut *self.find_goto } else { &mut none };
                let find_query = if is_active {
                    self.find_query.as_deref().unwrap_or("")
                } else {
                    ""
                };
                let view = self
                    .file_cache
                    .entry(path.clone())
                    .or_insert_with(|| load_file_view(ui.ctx(), path.as_path()));
                file_view_ui(ui, view, lang, self.cursor_out, goto, find_query);
            }
        }
    }

    /// Stable per-tab id (independent of the title, which changes with the dirty
    /// dot), so egui_dock keeps tab identity when the title text changes.
    fn id(&mut self, tab: &mut Self::Tab) -> egui::Id {
        egui::Id::new(("dock_tab", format!("{tab:?}")))
    }

    /// Draw the Focus eye + close X on the tab itself, right of the title (the
    /// title reserves trailing space for them). `response` gives the tab's rect.
    fn on_tab_button(&mut self, tab: &mut Self::Tab, response: &egui::Response) {
        let ctx = response.ctx.clone();
        let rect = response.rect;
        let slot = 15.0;
        let pad = 4.0; // gap from the tab's right edge
        let gap = 2.0; // gap between the eye and the X
        let cy = rect.center().y - slot / 2.0;
        let close_rect = egui::Rect::from_min_size(
            egui::pos2(rect.right() - slot - pad, cy),
            egui::vec2(slot, slot),
        );
        let eye_rect = egui::Rect::from_min_size(
            egui::pos2(rect.right() - 2.0 * slot - pad - gap, cy),
            egui::vec2(slot, slot),
        );

        let pointer = ctx.input(|i| i.pointer.interact_pos());
        let clicked = ctx.input(|i| i.pointer.primary_clicked());
        let eye_hov = pointer.is_some_and(|p| eye_rect.contains(p));
        let close_hov = pointer.is_some_and(|p| close_rect.contains(p));

        let painter = ctx.layer_painter(response.layer_id);
        if eye_hov || self.in_focus {
            painter.rect_filled(eye_rect, 3.0, egui::Color32::from_black_alpha(110));
        }
        draw_eye(&painter, eye_rect.center(), self.in_focus, eye_hov);
        if close_hov {
            painter.rect_filled(close_rect, 3.0, egui::Color32::from_black_alpha(110));
        }
        let m = close_rect.shrink(slot * 0.32);
        let xcol = if close_hov {
            egui::Color32::from_rgb(228, 128, 118)
        } else {
            egui::Color32::from_gray(135) // ~25% dimmer than a normal ~gray-180 icon
        };
        let s = egui::Stroke::new(1.5, xcol);
        painter.line_segment([m.left_top(), m.right_bottom()], s);
        painter.line_segment([m.right_top(), m.left_bottom()], s);

        if clicked {
            if let Some(p) = pointer {
                if eye_rect.contains(p) {
                    *self.focus_req = Some(tab.clone());
                } else if close_rect.contains(p) {
                    *self.close_req = Some(tab.clone());
                }
            }
        }
    }
}

/// Build the per-frame egui chrome (top bar, tool row, menu/settings, HUD) and the
/// dockable workspace (3D Scene / Logic / Code / AI Chat / Log).
fn build_ui(
    ctx: &egui::Context,
    ui_state: &mut UiState,
    chat: &mut ChatUi,
    mode: CameraMode,
    logo: Option<&egui::TextureHandle>,
    splash: Option<&egui::TextureHandle>,
    loading: bool,
    view_proj: Mat4,
    gizmo_axis: Option<usize>,
    gizmo_mode: GizmoMode,
    scene: &SceneSnapshot,
    outliner: &[String],
    inspector: Option<&Entity>,
    controls: &Controls,
    history: &[HistoryEntry],
    history_cursor: usize,
    project_path: Option<&Path>,
    project_dirty: bool,
    file_cache: &mut HashMap<PathBuf, FileView>,
    terminal: &mut TerminalState,
    git: &mut GitUi,
    pending_command: &mut Option<PendingCommand>,
    focus_restore: &mut Option<DockState<DockTab>>,
    dock_state: &mut DockState<DockTab>,
    scene_rect_out: &mut Option<egui::Rect>,
    toolbar_rect_out: &mut Option<egui::Rect>,
) {
    // Apply the selected theme + mode (UI only — the 3D background is fixed).
    ctx.set_visuals(theme_visuals(ui_state.theme, ui_state.dark_mode));

    // Loading screen: the splash at full opacity, fit to the window at its own
    // aspect ratio (no stretching), centered on black, until the timer elapses.
    if loading {
        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(egui::Color32::BLACK))
            .show(ctx, |ui| {
                let area = ui.max_rect();
                if let Some(tex) = splash {
                    let img = tex.size_vec2();
                    let ar = img.x / img.y.max(1.0);
                    let area_ar = area.width() / area.height().max(1.0);
                    // "Contain": largest rect with the image's aspect that fits.
                    let size = if area_ar > ar {
                        egui::vec2(area.height() * ar, area.height())
                    } else {
                        egui::vec2(area.width(), area.width() / ar)
                    };
                    let draw = egui::Rect::from_center_size(area.center(), size);
                    let uv =
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
                    ui.painter().image(tex.id(), draw, uv, egui::Color32::WHITE);
                    ui.painter().text(
                        draw.center_bottom() - egui::vec2(0.0, 16.0),
                        egui::Align2::CENTER_BOTTOM,
                        "Loading…",
                        egui::FontId::proportional(18.0),
                        egui::Color32::from_rgb(228, 223, 209),
                    );
                }
            });
        return;
    }

    // Startup popup: offer to reopen the last-used project (or new / empty).
    if ui_state.show_startup_popup {
        let name = ui_state
            .last_project_name
            .clone()
            .unwrap_or_else(|| "last project".to_string());
        egui::Window::new("startup")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .movable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                ui.set_min_width(300.0);
                ui.add_space(10.0);
                ui.vertical_centered(|ui| {
                    if let Some(tex) = logo {
                        ui.add(egui::Image::new(egui::load::SizedTexture::new(
                            tex.id(),
                            egui::vec2(48.0, 48.0),
                        )));
                    }
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new("Welcome to CoEngine")
                            .heading()
                            .color(ACCENT_GOLD),
                    );
                });
                ui.add_space(10.0);
                ui.label("Open last used project?");
                ui.label(egui::RichText::new(&name).strong());
                ui.add_space(10.0);

                let bw = ui.available_width();
                if ui
                    .add_sized(
                        [bw, 32.0],
                        egui::Button::new(format!("Open  {name}"))
                            .fill(egui::Color32::from_rgb(60, 80, 120)),
                    )
                    .clicked()
                {
                    ui_state.open_last_requested = true;
                    ui_state.show_startup_popup = false;
                }
                ui.add_space(6.0);
                if ui
                    .add_sized([bw, 30.0], egui::Button::new("New Project…"))
                    .clicked()
                {
                    ui_state.new_requested = true;
                    ui_state.show_startup_popup = false;
                }
                ui.add_space(6.0);
                if ui
                    .add_sized([bw, 30.0], egui::Button::new("Start empty"))
                    .clicked()
                {
                    ui_state.show_startup_popup = false;
                }
                ui.add_space(8.0);
            });
    }

    // Guards the click-away close so the very click that opens a popup this frame
    // doesn't immediately close it again.
    let mut popup_just_opened = false;

    // Top bar: Menu + identity + tabs, all on one row.
    egui::TopBottomPanel::top("topbar").show(ctx, |ui| {
        ui.add_space(2.0);
        ui.horizontal(|ui| {
            if ui.button("Menu").clicked() {
                ui_state.menu_open = !ui_state.menu_open;
                popup_just_opened = true;
            }
            ui.separator();
            ui.label(egui::RichText::new("CoEngine").strong().color(ACCENT_GOLD));
            ui.label(egui::RichText::new(format!("v{CO_VERSION}")));

            // Reopen buttons for any closed dock widgets.
            let tabs = [
                (DockTab::Scene, "3D Scene"),
                (DockTab::Logic, "Objects Added"),
                (DockTab::Inspector, "Inspector"),
                (DockTab::Explorer, "Explorer"),
                (DockTab::AiChat, "AI Chat"),
                (DockTab::Log, "Log"),
                (DockTab::Terminal, "Terminal"),
                (DockTab::Git, "Git"),
                (DockTab::Search, "Search"),
            ];
            // (Reopen buttons are hidden during Focus mode — the minimized tabs
            // are shown by the Focus row below instead.)
            if focus_restore.is_none()
                && tabs.iter().any(|(t, _)| dock_state.find_tab(t).is_none())
            {
                ui.separator();
                ui.label(egui::RichText::new("Open:").small());
                for (tab, label) in tabs {
                    if dock_state.find_tab(&tab).is_none() && ui.button(label).clicked() {
                        dock_state.push_to_focused_leaf(tab);
                    }
                }
            }

            // Focus mode: the other tabs are "minimized" up here. Click one to
            // focus it instead; the focused tab's eye (in its corner) exits.
            if focus_restore.is_some() {
                ui.separator();
                ui.label(egui::RichText::new("Focus:").small().color(ACCENT_GOLD));
                let focused = dock_state.iter_all_tabs().next().map(|(_, t)| t.clone());
                if let Some(f) = &focused {
                    ui.label(egui::RichText::new(dock_tab_label(f)).small().strong());
                }
                let mut switch_to: Option<DockTab> = None;
                if let Some(saved) = focus_restore.as_ref() {
                    for (_, tab) in saved.iter_all_tabs() {
                        if focused.as_ref() != Some(tab)
                            && ui.small_button(dock_tab_label(tab)).clicked()
                        {
                            switch_to = Some(tab.clone());
                        }
                    }
                }
                if let Some(t) = switch_to {
                    *dock_state = DockState::new(vec![t]);
                }
            }

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

    // Tool row (scene tools will live here later).
    // Tool row: Save / Save All for the active file viewer (also Ctrl+S).
    let active_file: Option<PathBuf> = dock_state.find_active_focused().and_then(|(_, tab)| {
        match tab {
            DockTab::File(p) => Some(p.clone()),
            _ => None,
        }
    });
    let active_dirty = active_file
        .as_ref()
        .is_some_and(|p| file_cache.get(p).is_some_and(FileView::is_dirty));
    let any_dirty = file_cache.values().any(FileView::is_dirty);
    let mut do_save = ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::S));
    let mut do_save_all = false;
    let mut do_save_project = false;
    egui::TopBottomPanel::top("tool_bar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.add_space(2.0);
            // Save the whole project (scene + settings + layout). Always available;
            // a leading dot marks unsaved changes.
            let label = if project_dirty {
                "● Save Project"
            } else {
                "Save Project"
            };
            if ui
                .button(label)
                .on_hover_text("Save the project — scene, settings, and layout")
                .clicked()
            {
                do_save_project = true;
            }
            ui.separator();
            // File saves (the active editor file).
            if ui
                .add_enabled(active_dirty, egui::Button::new("Save File"))
                .on_hover_text("Save the active file (Ctrl+S)")
                .clicked()
            {
                do_save = true;
            }
            if ui
                .add_enabled(any_dirty, egui::Button::new("Save All"))
                .clicked()
            {
                do_save_all = true;
            }
            ui.separator();
            ui.label(egui::RichText::new("Tools").small());
        });
    });

    // Bottom status bar: git branch (left); caret position, active-file language,
    // and unsaved-file count (right). The caret is last frame's value (see the
    // dock pass below) — imperceptible lag. No error/warning count yet: that needs
    // the diagnostics/LSP layer, which is deferred until the scripting lang is set.
    let unsaved = file_cache.values().filter(|v| v.is_dirty()).count();
    let lang = active_file.as_ref().map(|p| language_for(p));
    let branch = project_path.and_then(current_branch);
    let cursor = ui_state.editor_cursor;
    egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.add_space(6.0);
            match &branch {
                Some(b) => {
                    ui.label("Branch:");
                    ui.label(egui::RichText::new(b).color(ACCENT_GOLD));
                }
                None => {
                    ui.label(egui::RichText::new("No repo").weak());
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(6.0);
                if let Some((line, col)) = cursor {
                    ui.label(format!("Ln {line}, Col {col}"));
                    ui.separator();
                }
                if let Some(l) = lang {
                    ui.label(l.to_uppercase());
                    ui.separator();
                }
                let dot = if unsaved > 0 { "● " } else { "" };
                ui.label(format!("{dot}{unsaved} unsaved"));
            });
        });
    });

    if do_save_project {
        ui_state.save_requested = true;
    }
    if do_save && active_dirty {
        if let Some(p) = active_file.as_ref() {
            match save_file(p, file_cache) {
                Ok(()) => println!("Saved {}", p.display()),
                Err(e) => eprintln!("Save failed: {e}"),
            }
        }
    }
    if do_save_all {
        let dirty: Vec<PathBuf> = file_cache
            .iter()
            .filter(|(_, v)| v.is_dirty())
            .map(|(k, _)| k.clone())
            .collect();
        for p in dirty {
            if let Err(e) = save_file(&p, file_cache) {
                eprintln!("Save failed for {}: {e}", p.display());
            }
        }
    }

    // In-file Find/Replace bar (Ctrl+F), for the active text file. Runs before the
    // dock pass so its edits land before the viewer borrows `file_cache`; sets
    // `find_goto`, which the active editor consumes during the dock pass.
    let active_text = active_file
        .as_ref()
        .filter(|p| matches!(file_cache.get(*p), Some(FileView::Text { .. })))
        .cloned();
    if active_text.is_some()
        && ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::F))
    {
        ui_state.find.open = true;
        ui_state.find.focus_query = true;
    }
    if ui_state.find.open {
        match active_text {
            Some(path) => find_bar(
                ctx,
                &mut ui_state.find,
                &mut ui_state.find_goto,
                &path,
                file_cache,
            ),
            None => ui_state.find.open = false,
        }
    }

    // The dockable workspace fills the central area. Each tab can be dragged into a
    // column or a full-screen tab and resized; the 3D Scene tab is the live viewport.
    let mut open_file_req: Option<PathBuf> = None;
    let mut fs_req: Option<FsPrompt> = None;
    let mut fs_move_req: Option<(PathBuf, PathBuf)> = None;
    let mut focus_req: Option<DockTab> = None;
    let mut close_req: Option<DockTab> = None;
    // Recomputed each frame from whichever file editor has focus.
    ui_state.editor_cursor = None;
    {
        let mut viewer = EngineTabs {
            chat,
            scene,
            history,
            history_cursor,
            undo_req: &mut ui_state.undo_requested,
            redo_req: &mut ui_state.redo_requested,
            scene_rect_out: &mut *scene_rect_out,
            project_root: project_path.map(|p| p.to_path_buf()),
            open_file_req: &mut open_file_req,
            fs_req: &mut fs_req,
            fs_move_req: &mut fs_move_req,
            file_cache,
            terminal: &mut *terminal,
            terminal_shell: ui_state.shell.command().to_string(),
            git: &mut *git,
            focus_req: &mut focus_req,
            close_req: &mut close_req,
            in_focus: focus_restore.is_some(),
            outliner,
            outliner_select: &mut ui_state.outliner_select,
            outliner_delete: &mut ui_state.outliner_delete,
            outliner_add: &mut ui_state.outliner_add,
            inspector: inspector.cloned(),
            inspector_apply: &mut ui_state.inspector_apply,
            inspector_commit: &mut ui_state.inspector_commit,
            cursor_out: &mut ui_state.editor_cursor,
            search: &mut ui_state.search,
            active_file: active_file.clone(),
            find_goto: &mut ui_state.find_goto,
            find_query: if ui_state.find.open {
                Some(ui_state.find.query.clone())
            } else {
                None
            },
        };
        DockArea::new(dock_state)
            .style(Style::from_egui(ctx.style().as_ref()))
            .show(ctx, &mut viewer);
    }
    // The 3D Scene tab's rect this frame (None when the Scene tab isn't visible).
    let scene_rect = *scene_rect_out;

    // Custom close X clicked: remove that tab from the layout.
    if let Some(tab) = close_req {
        if let Some(loc) = dock_state.find_tab(&tab) {
            dock_state.remove_tab(loc);
        }
    }

    // Focus eye toggled: enter focus (save layout, show only this tab) or, if
    // already focused, exit (restore the saved layout).
    if let Some(tab) = focus_req {
        if let Some(saved) = focus_restore.take() {
            *dock_state = saved;
        } else {
            *focus_restore = Some(std::mem::replace(dock_state, DockState::new(vec![tab])));
        }
    }
    // Safety: if the single focused tab got closed, leave Focus mode.
    if focus_restore.is_some() && dock_state.iter_all_tabs().next().is_none() {
        if let Some(saved) = focus_restore.take() {
            *dock_state = saved;
        }
    }

    // A file was clicked in the Explorer: open (or focus) a viewer tab for it,
    // docked in the same leaf as Logic (top-center) when Logic is present.
    if let Some(path) = open_file_req {
        open_file_tab(dock_state, path);
    }

    // An Explorer right-click op was chosen: stage it in the modal (fresh, so any
    // earlier error message clears). Then render the modal and run it on confirm.
    if let Some(req) = fs_req {
        ui_state.fs_prompt = Some(req);
        ui_state.fs_prompt_error = None;
    }
    // A tree item was dragged onto a folder: move it, syncing open tabs. A failed
    // move (name conflict, into-itself) surfaces as a standalone error popup.
    if let Some((from, dest)) = fs_move_req {
        match move_into(&from, &dest) {
            Ok(Some((from, to))) => {
                let was_file = to.is_file();
                close_file_tabs(dock_state, file_cache, |p| path_is_within(p, &from));
                if was_file {
                    open_file_tab(dock_state, to);
                }
            }
            Ok(None) => {}
            Err(e) => ui_state.fs_prompt_error = Some(e),
        }
    }
    service_fs_prompt(ctx, ui_state, file_cache, dock_state);

    // Bottom-left debug overlay: controls + version on a dark plate, confined to
    // the 3D Scene viewport and only shown when the Scene tab is visible. Hidden
    // entirely with H (re-shown via the top bar).
    if ui_state.show_debug {
        if let Some(rect) = scene_rect {
        egui::Area::new(egui::Id::new("hud_bottom_left"))
            .pivot(egui::Align2::LEFT_BOTTOM)
            .fixed_pos(egui::pos2(rect.left() + 10.0, rect.bottom() - 10.0))
            .constrain_to(rect)
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
                                "Orbit:  drag = orbit · R-drag = pan · scroll = zoom".to_string()
                            }
                            CameraMode::Fly => format!(
                                "Fly:  {}{}{}{} move · {}/{} up/down · R-drag = look",
                                key_label(controls.forward),
                                key_label(controls.left),
                                key_label(controls.back),
                                key_label(controls.right),
                                key_label(controls.up),
                                key_label(controls.down),
                            ),
                        };
                        ui.label(egui::RichText::new(cam).color(text).small());
                        ui.label(
                            egui::RichText::new(format!(
                                "click = select · {} = add cube · {} = remove · {} = orbit/fly",
                                key_label(controls.add_cube),
                                key_label(controls.remove),
                                key_label(controls.toggle_camera),
                            ))
                            .color(text)
                            .small(),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "{} = menu · {} = hide debug info",
                                key_label(controls.toggle_menu),
                                key_label(controls.toggle_debug),
                            ))
                            .color(dim)
                            .small(),
                        );
                        // Object nudge controls — only while something is selected.
                        if scene.selected.is_some() {
                            ui.add_space(2.0);
                            ui.label(
                                egui::RichText::new(
                                    "selected:  Arrows = move · PgUp/PgDn = height",
                                )
                                .color(text)
                                .small(),
                            );
                            ui.label(
                                egui::RichText::new(format!(
                                    ", / . = rotate · R = move/rotate gizmo · X = grid · {} = delete",
                                    key_label(controls.remove),
                                ))
                                .color(text)
                                .small(),
                            );
                        }
                        ui.label(
                            egui::RichText::new(format!("CoEngine v{CO_VERSION}"))
                                .monospace()
                                .color(ACCENT_GOLD),
                        );
                    });
            });
        }
    }

    // Translate gizmo over the selected object (draw-only; the drag is handled
    // by the engine's input). A foreground layer painter doesn't register an
    // interactive area, so it never blocks the camera.
    if let (Some(i), Some(rect)) = (scene.selected, scene_rect) {
        if let Some((obj_pos, _)) = scene.cubes.get(i).copied() {
            let mut painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("gizmo_draw"),
            ));
            painter.set_clip_rect(rect);
            match gizmo_mode {
                GizmoMode::Translate => {
                    draw_translate_gizmo(&painter, rect, view_proj, obj_pos, gizmo_axis);
                }
                GizmoMode::Rotate => {
                    let rot = inspector.map(|e| e.rotation).unwrap_or(glam::Vec3::ZERO);
                    draw_rotate_gizmo(&painter, rect, view_proj, obj_pos, rot, gizmo_axis);
                    // Live angle readout while dragging a ring.
                    if let (Some(ax), Some(e), Some(c)) = (
                        gizmo_axis,
                        inspector,
                        project_to_screen(obj_pos, view_proj, rect),
                    ) {
                        let deg = [e.rotation.x, e.rotation.y, e.rotation.z][ax].rem_euclid(360.0);
                        painter.text(
                            c + egui::vec2(16.0, -16.0),
                            egui::Align2::LEFT_BOTTOM,
                            format!("{deg:.0}°"),
                            egui::FontId::proportional(15.0),
                            egui::Color32::from_gray(240),
                        );
                    }
                }
                GizmoMode::Scale => {
                    draw_scale_gizmo(&painter, rect, view_proj, obj_pos, gizmo_axis);
                    // Live percentage readout while dragging.
                    if let (Some(_), Some(e), Some(c)) = (
                        gizmo_axis,
                        inspector,
                        project_to_screen(obj_pos, view_proj, rect),
                    ) {
                        painter.text(
                            c + egui::vec2(16.0, -16.0),
                            egui::Align2::LEFT_BOTTOM,
                            format!("{:.0}%", e.scale.x * 100.0),
                            egui::FontId::proportional(15.0),
                            egui::Color32::from_gray(240),
                        );
                    }
                }
            }
        }
    }

    // On-screen gizmo toolbar (WoW-Housing-style), at the bottom-center of the
    // scene when an object is selected. Its rect is recorded so clicks on it
    // don't fall through to the camera / picking.
    *toolbar_rect_out = None;
    if let (Some(sel_i), Some(rect)) = (scene.selected, scene_rect) {
        let area = egui::Area::new(egui::Id::new("gizmo_toolbar"))
            .pivot(egui::Align2::CENTER_BOTTOM)
            .fixed_pos(egui::pos2(rect.center().x, rect.bottom() - 14.0))
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                egui::Frame::none()
                    .fill(egui::Color32::from_rgba_unmultiplied(18, 20, 26, 235))
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(70, 56, 30)))
                    .rounding(egui::Rounding::same(9.0))
                    .inner_margin(egui::Margin::symmetric(8.0, 6.0))
                    .show(ui, |ui| {
                        ui.vertical_centered(|ui| {
                        // Scale slider (WoW-style) sits above the tool buttons.
                        if gizmo_mode == GizmoMode::Scale {
                            let cur = inspector.map(|e| e.scale.x).unwrap_or(1.0);
                            scale_slider(
                                ui,
                                cur,
                                &mut ui_state.scale_req,
                                &mut ui_state.scale_commit,
                            );
                            ui.add_space(4.0);
                        }
                        ui.horizontal(|ui| {
                            if tool_button(ui, ToolIcon::Move, gizmo_mode == GizmoMode::Translate) {
                                ui_state.gizmo_mode_req = Some(GizmoMode::Translate);
                            }
                            if tool_button(ui, ToolIcon::Rotate, gizmo_mode == GizmoMode::Rotate) {
                                ui_state.gizmo_mode_req = Some(GizmoMode::Rotate);
                            }
                            if tool_button(ui, ToolIcon::Scale, gizmo_mode == GizmoMode::Scale) {
                                ui_state.gizmo_mode_req = Some(GizmoMode::Scale);
                            }
                            ui.add_space(6.0);
                            if tool_button(ui, ToolIcon::Delete, false) {
                                ui_state.outliner_delete = Some(sel_i);
                            }
                        });
                        });
                    });
            });
        *toolbar_rect_out = Some(area.response.rect);
    }

    // Menu window (opened by Esc or the Menu button).
    if ui_state.menu_open {
        let resp = egui::Window::new("menu")
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

                // Project: New / Open / Save / Save As. These raise one-shot requests
                // that `State::update` services (the file I/O needs the live scene).
                if ui
                    .add_sized([bw, 30.0], egui::Button::new("New Project…"))
                    .clicked()
                {
                    ui_state.new_requested = true;
                    ui_state.menu_open = false;
                }
                ui.add_space(6.0);
                if ui
                    .add_sized([bw, 30.0], egui::Button::new("Open Project…"))
                    .clicked()
                {
                    ui_state.open_requested = true;
                    ui_state.menu_open = false;
                }
                ui.add_space(6.0);
                if ui
                    .add_sized([bw, 30.0], egui::Button::new("Save Project"))
                    .clicked()
                {
                    ui_state.save_requested = true;
                    ui_state.menu_open = false;
                }
                ui.add_space(6.0);
                if ui
                    .add_sized([bw, 30.0], egui::Button::new("Save Project As…"))
                    .clicked()
                {
                    ui_state.save_as_requested = true;
                    ui_state.menu_open = false;
                }
                if let Some(status) = &ui_state.project_status {
                    ui.add_space(4.0);
                    ui.label(egui::RichText::new(status).small().italics());
                }
                ui.add_space(8.0);
                ui.separator();
                ui.add_space(8.0);

                if ui
                    .add_sized([bw, 34.0], egui::Button::new("Settings"))
                    .clicked()
                {
                    ui_state.settings_open = true;
                    ui_state.menu_open = false;
                    popup_just_opened = true;
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
        if !popup_just_opened && clicked_outside(ctx, &resp) {
            ui_state.menu_open = false;
        }
    }

    // Settings window (modal-ish).
    if ui_state.settings_open {
        let resp = egui::Window::new("Settings")
            .collapsible(false)
            .resizable(false)
            .movable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .fixed_size(egui::vec2(580.0, 300.0))
            .show(ctx, |ui| {
                ui.horizontal_top(|ui| {
                    // Left category nav.
                    ui.vertical(|ui| {
                        ui.add_space(2.0);
                        ui.selectable_value(
                            &mut ui_state.settings_tab,
                            SettingsTab::Theme,
                            "Theme",
                        );
                        ui.selectable_value(
                            &mut ui_state.settings_tab,
                            SettingsTab::Controls,
                            "Controls",
                        );
                        ui.selectable_value(
                            &mut ui_state.settings_tab,
                            SettingsTab::Terminal,
                            "Terminal",
                        );
                    });
                    ui.add_space(14.0);
                    // Content for the selected category.
                    ui.vertical(|ui| {
                        match ui_state.settings_tab {
                            SettingsTab::Theme => {
                                ui.label("Theme");
                                ui.horizontal(|ui| {
                                    ui.selectable_value(
                                        &mut ui_state.theme,
                                        Theme::DefaultSimple,
                                        "Default Simple",
                                    );
                                    ui.selectable_value(
                                        &mut ui_state.theme,
                                        Theme::Barbarian,
                                        "Barbarian",
                                    );
                                });
                                ui.add_space(6.0);
                                ui.label("Mode (UI only — does not affect the 3D view)");
                                ui.horizontal(|ui| {
                                    ui.selectable_value(&mut ui_state.dark_mode, true, "Dark");
                                    ui.selectable_value(&mut ui_state.dark_mode, false, "Light");
                                });
                            }
                            SettingsTab::Controls => {
                                ui.label("Click a key to rebind");
                                egui::Grid::new("controls_grid")
                                    .num_columns(4)
                                    .spacing([12.0, 4.0])
                                    .show(ui, |ui| {
                                        // Two actions per row to keep the panel short.
                                        for pair in CONTROL_ACTIONS.chunks(2) {
                                            for &(action, label) in pair {
                                                ui.label(label);
                                                let active = ui_state.rebinding == Some(action);
                                                let txt = if active {
                                                    "press…".to_string()
                                                } else {
                                                    key_label(controls.key(action))
                                                };
                                                if ui
                                                    .add(
                                                        egui::Button::new(txt)
                                                            .min_size(egui::vec2(80.0, 0.0)),
                                                    )
                                                    .clicked()
                                                {
                                                    ui_state.rebinding =
                                                        if active { None } else { Some(action) };
                                                }
                                            }
                                            ui.end_row();
                                        }
                                    });
                                if ui_state.rebinding.is_some() {
                                    ui.label(
                                        egui::RichText::new("Press any key to bind · Esc to cancel")
                                            .small()
                                            .italics(),
                                    );
                                }
                            }
                            SettingsTab::Terminal => {
                                ui.label("Shell for the in-engine Terminal");
                                ui.add_space(4.0);
                                let before = ui_state.shell;
                                for sh in [Shell::PowerShell, Shell::Pwsh, Shell::Cmd] {
                                    ui.selectable_value(&mut ui_state.shell, sh, sh.label());
                                }
                                if ui_state.shell != before {
                                    // Restart the terminal with the newly chosen shell.
                                    *terminal = TerminalState::Off;
                                }
                                ui.add_space(6.0);
                                ui.label(
                                    egui::RichText::new(
                                        "Changing the shell restarts the Terminal.",
                                    )
                                    .small()
                                    .italics(),
                                );
                            }
                        }
                    });
                });
                // Pin Close to the bottom so the layout is identical for every submenu.
                let pad = (ui.available_height() - 30.0).max(0.0);
                ui.add_space(pad);
                ui.separator();
                if ui.button("Close").clicked() {
                    ui_state.settings_open = false;
                    ui_state.rebinding = None;
                }
            });
        // Don't click-away close while rebinding a key (the next click is the bind).
        if !popup_just_opened && ui_state.rebinding.is_none() && clicked_outside(ctx, &resp) {
            ui_state.settings_open = false;
        }
    }

    // CoE-AI command approval (confirm-each): the agent's worker thread is blocked
    // until the user approves (runs the command) or denies it.
    if pending_command.is_some() {
        let resp = egui::Window::new("cmd_confirm")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .movable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                ui.set_min_width(380.0);
                ui.add_space(8.0);
                ui.label(
                    egui::RichText::new("CoE-AI wants to run a terminal command")
                        .strong()
                        .color(ACCENT_GOLD),
                );
                ui.add_space(6.0);
                let cmd_text = pending_command
                    .as_ref()
                    .map(|p| p.command.clone())
                    .unwrap_or_default();
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    ui.add(
                        egui::Label::new(egui::RichText::new(&cmd_text).monospace())
                            .wrap_mode(egui::TextWrapMode::Extend),
                    );
                });
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui
                        .button(
                            egui::RichText::new("Approve & Run")
                                .color(egui::Color32::from_rgb(220, 230, 220)),
                        )
                        .clicked()
                    {
                        if let Some(pc) = pending_command.take() {
                            let shell = ui_state.shell;
                            let cwd = project_path.map(|p| p.to_path_buf());
                            std::thread::spawn(move || {
                                let out = run_captured(shell, &pc.command, cwd.as_deref());
                                let _ = pc.reply.send(out);
                            });
                        }
                    }
                    if ui.button("Deny").clicked() {
                        if let Some(pc) = pending_command.take() {
                            let _ = pc
                                .reply
                                .send("The user denied running this command.".to_string());
                        }
                    }
                });
                ui.add_space(8.0);
            });
        // Click-away denies (safe default: nothing runs).
        if clicked_outside(ctx, &resp) {
            if let Some(pc) = pending_command.take() {
                let _ = pc
                    .reply
                    .send("The user denied running this command.".to_string());
            }
        }
    }
}

/// True when the user presses the primary mouse button this frame outside the
/// given popup window — the shared "click-away closes it" rule for modals/popups.
fn clicked_outside<R>(ctx: &egui::Context, inner: &Option<egui::InnerResponse<R>>) -> bool {
    let Some(ir) = inner else { return false };
    if !ctx.input(|i| i.pointer.primary_pressed()) {
        return false;
    }
    ctx.input(|i| i.pointer.interact_pos())
        .is_some_and(|p| !ir.response.rect.contains(p))
}

/// Open (or focus) a viewer tab for `path`, docked in the same leaf as Logic
/// (top-center) when Logic is present. Shared by the Explorer click handler and
/// the "new file" op.
fn open_file_tab(dock_state: &mut DockState<DockTab>, path: PathBuf) {
    let tab = DockTab::File(path);
    if let Some(loc) = dock_state.find_tab(&tab) {
        dock_state.set_active_tab(loc);
    } else {
        if let Some((surface, node, _)) = dock_state.find_tab(&DockTab::Logic) {
            dock_state.set_focused_node_and_surface((surface, node));
        }
        dock_state.push_to_focused_leaf(tab);
    }
}

/// Close any open `File` viewer tabs whose path satisfies `pred` (used after a
/// rename/delete so stale tabs don't linger), and drop their cached contents.
fn close_file_tabs(
    dock_state: &mut DockState<DockTab>,
    file_cache: &mut HashMap<PathBuf, FileView>,
    pred: impl Fn(&Path) -> bool,
) {
    let affected: Vec<DockTab> = dock_state
        .iter_all_tabs()
        .filter_map(|(_, t)| match t {
            DockTab::File(p) if pred(p) => Some(DockTab::File(p.clone())),
            _ => None,
        })
        .collect();
    for tab in affected {
        if let DockTab::File(p) = &tab {
            file_cache.remove(p);
        }
        if let Some(loc) = dock_state.find_tab(&tab) {
            dock_state.remove_tab(loc);
        }
    }
}

/// True if `p` is `base` itself or lives underneath it (so a folder rename/delete
/// also catches the files open from inside it).
fn path_is_within(p: &Path, base: &Path) -> bool {
    p == base || p.starts_with(base)
}

/// Render the Explorer file-op modal (new file/folder, rename, delete) when one
/// is pending, and run the op on confirm — syncing open tabs and the file cache.
fn service_fs_prompt(
    ctx: &egui::Context,
    ui_state: &mut UiState,
    file_cache: &mut HashMap<PathBuf, FileView>,
    dock_state: &mut DockState<DockTab>,
) {
    if ui_state.fs_prompt.is_none() {
        // No naming modal open, but a drag-move may have failed — show it on its
        // own so the error isn't lost.
        if let Some(err) = ui_state.fs_prompt_error.clone() {
            let mut dismiss = false;
            let resp = egui::Window::new("fs_error")
                .title_bar(false)
                .collapsible(false)
                .resizable(false)
                .movable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .show(ctx, |ui| {
                    ui.set_min_width(300.0);
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new("Couldn't move that")
                            .strong()
                            .color(ACCENT_GOLD),
                    );
                    ui.add_space(6.0);
                    ui.label(
                        egui::RichText::new(err).color(egui::Color32::from_rgb(228, 128, 118)),
                    );
                    ui.add_space(10.0);
                    if ui.button("OK").clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                        dismiss = true;
                    }
                    ui.add_space(8.0);
                });
            if dismiss || clicked_outside(ctx, &resp) {
                ui_state.fs_prompt_error = None;
            }
        }
        return;
    }

    let mut do_run = false;
    let mut do_cancel = false;
    let resp = egui::Window::new("fs_prompt")
        .title_bar(false)
        .collapsible(false)
        .resizable(false)
        .movable(false)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ctx, |ui| {
            ui.set_min_width(340.0);
            ui.add_space(8.0);
            // Title + (for naming ops) an editable name field; Enter confirms.
            let prompt = ui_state.fs_prompt.as_mut().expect("prompt present");
            let (title, name): (&str, Option<&mut String>) = match prompt {
                FsPrompt::NewFile { name, .. } => ("New File", Some(name)),
                FsPrompt::NewFolder { name, .. } => ("New Folder", Some(name)),
                FsPrompt::Rename { name, .. } => ("Rename", Some(name)),
                FsPrompt::Delete { target } => {
                    let label = target
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| target.display().to_string());
                    ui.label(
                        egui::RichText::new(format!("Delete \"{label}\"?"))
                            .strong()
                            .color(ACCENT_GOLD),
                    );
                    if target.is_dir() {
                        ui.add_space(4.0);
                        ui.label(
                            egui::RichText::new("This folder and everything in it will be removed.")
                                .weak(),
                        );
                    }
                    ("Delete", None)
                }
            };
            if let Some(buf) = name {
                ui.label(egui::RichText::new(title).strong().color(ACCENT_GOLD));
                ui.add_space(6.0);
                let field = ui.add(
                    egui::TextEdit::singleline(buf)
                        .desired_width(f32::INFINITY)
                        .hint_text("name"),
                );
                field.request_focus();
                if field.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    do_run = true;
                }
            }
            if let Some(err) = &ui_state.fs_prompt_error {
                ui.add_space(6.0);
                ui.label(egui::RichText::new(err).color(egui::Color32::from_rgb(228, 128, 118)));
            }
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                let confirm = match ui_state.fs_prompt.as_ref().expect("prompt present") {
                    FsPrompt::Delete { .. } => "Delete",
                    FsPrompt::Rename { .. } => "Rename",
                    _ => "Create",
                };
                if ui.button(confirm).clicked() {
                    do_run = true;
                }
                if ui.button("Cancel").clicked() {
                    do_cancel = true;
                }
            });
            ui.add_space(8.0);
            if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                do_cancel = true;
            }
        });

    if do_cancel || clicked_outside(ctx, &resp) {
        ui_state.fs_prompt = None;
        ui_state.fs_prompt_error = None;
        return;
    }
    if !do_run {
        return;
    }

    let prompt = ui_state.fs_prompt.as_ref().expect("prompt present");
    match run_fs_prompt(prompt) {
        Ok(outcome) => {
            match outcome {
                FsOutcome::CreatedFile(p) => open_file_tab(dock_state, p),
                FsOutcome::CreatedFolder => {}
                FsOutcome::Renamed { from, to } => {
                    // Reopen a single renamed file at its new path; just close
                    // anything affected for a folder rename.
                    let was_file = to.is_file();
                    close_file_tabs(dock_state, file_cache, |p| path_is_within(p, &from));
                    if was_file {
                        open_file_tab(dock_state, to);
                    }
                }
                FsOutcome::Deleted(target) => {
                    close_file_tabs(dock_state, file_cache, |p| path_is_within(p, &target));
                }
            }
            ui_state.fs_prompt = None;
            ui_state.fs_prompt_error = None;
        }
        // Keep the modal open and show why it failed.
        Err(e) => ui_state.fs_prompt_error = Some(e),
    }
}

/// The in-file Find/Replace bar (a small floating window). Operates directly on
/// the active file's buffer; Prev/Next set `find_goto` (the editor scrolls/selects
/// the match), Replace/All edit the buffer in place.
fn find_bar(
    ctx: &egui::Context,
    find: &mut FindState,
    find_goto: &mut Option<(usize, usize)>,
    path: &Path,
    file_cache: &mut HashMap<PathBuf, FileView>,
) {
    let Some(FileView::Text { buf, dirty }) = file_cache.get_mut(path) else {
        find.open = false;
        return;
    };
    let matches = find_matches(buf, &find.query);
    let n = matches.len();
    if find.current >= n {
        find.current = 0;
    }
    let mut goto_idx: Option<usize> = None;
    let mut did_replace = false;
    let mut do_close = false;
    let resp = egui::Window::new("find_bar")
        .title_bar(false)
        .resizable(false)
        .collapsible(false)
        .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-14.0, 64.0))
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                let q = ui.add(
                    egui::TextEdit::singleline(&mut find.query)
                        .desired_width(170.0)
                        .hint_text("Find"),
                );
                if find.focus_query {
                    q.request_focus();
                    find.focus_query = false;
                }
                let enter = q.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                let count = if n > 0 {
                    format!("{}/{}", find.current + 1, n)
                } else if find.query.is_empty() {
                    String::new()
                } else {
                    "0".to_string()
                };
                ui.label(egui::RichText::new(count).weak());
                if ui.button("Prev").clicked() && n > 0 {
                    find.current = (find.current + n - 1) % n;
                    goto_idx = Some(find.current);
                }
                if (ui.button("Next").clicked() || enter) && n > 0 {
                    find.current = (find.current + 1) % n;
                    goto_idx = Some(find.current);
                }
                if ui.button("Close").clicked() {
                    do_close = true;
                }
            });
            ui.horizontal(|ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut find.replace)
                        .desired_width(170.0)
                        .hint_text("Replace"),
                );
                if ui.button("Replace").clicked() && n > 0 {
                    let (s, e) = matches[find.current.min(n - 1)];
                    replace_char_range(buf, s, e, &find.replace);
                    *dirty = true;
                    did_replace = true;
                }
                if ui.button("All").clicked() && n > 0 {
                    // Reverse order so earlier char indices stay valid as we splice.
                    for &(s, e) in matches.iter().rev() {
                        replace_char_range(buf, s, e, &find.replace);
                    }
                    *dirty = true;
                    did_replace = true;
                }
            });
            if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                do_close = true;
            }
        });
    if let Some(i) = goto_idx {
        *find_goto = Some(matches[i]);
    }
    if did_replace {
        find.current = 0;
    }
    if do_close || clicked_outside(ctx, &resp) {
        find.open = false;
    }
}

/// Render the Log content (action history + Undo / Redo) into a dock tab's Ui.
fn log_tab(
    ui: &mut egui::Ui,
    history: &[HistoryEntry],
    cursor: usize,
    undo_req: &mut bool,
    redo_req: &mut bool,
) {
    egui::TopBottomPanel::top("log_header").show_inside(ui, |ui| {
                ui.add_space(6.0);
                ui.heading("Log");
                ui.horizontal(|ui| {
                    let can_undo = cursor > 0;
                    let can_redo = cursor + 1 < history.len();
                    if ui
                        .add_enabled(can_undo, egui::Button::new("Undo"))
                        .clicked()
                    {
                        *undo_req = true;
                    }
                    if ui
                        .add_enabled(can_redo, egui::Button::new("Redo"))
                        .clicked()
                    {
                        *redo_req = true;
                    }
                });
                ui.add_space(4.0);
            });
            egui::CentralPanel::default().show_inside(ui, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for (i, entry) in history.iter().enumerate() {
                            let mut rt = egui::RichText::new(&entry.label).small();
                            if i == cursor {
                                rt = rt.strong().color(ACCENT_GOLD);
                            } else if i > cursor {
                                rt = rt.weak();
                            }
                            ui.label(rt);
                        }
                    });
            });
}


/// One entry in the session action log / undo history: a label plus the full
/// scene snapshot taken right after that action.
#[derive(Clone)]
struct HistoryEntry {
    label: String,
    entities: Vec<Entity>,
    selected: Option<usize>,
}

/// The default no-project layout, captured from a hand-arranged session
/// (assets/default_layout.json) and baked into the binary. Used on a fresh
/// launch and for "Start empty". Stale rects are recomputed by egui_dock from
/// the stored split fractions on the first layout pass.
const DEFAULT_LAYOUT_JSON: &str = include_str!("../assets/default_layout.json");

/// Build the default dock layout: deserialize the baked-in arranged layout,
/// falling back to the programmatic layout if it ever fails to parse.
fn build_dock_state() -> DockState<DockTab> {
    serde_json::from_str(DEFAULT_LAYOUT_JSON).unwrap_or_else(|e| {
        eprintln!("default_layout.json failed to parse ({e}); using fallback layout");
        fallback_dock_state()
    })
}

/// Programmatic fallback: 3D Scene + Logic as main tabs, with Code, AI Chat,
/// and Log peeled off as columns on the right.
fn fallback_dock_state() -> DockState<DockTab> {
    let mut state = DockState::new(vec![DockTab::Scene, DockTab::Logic]);
    let surface = state.main_surface_mut();
    let [main, aichat] = surface.split_right(NodeIndex::root(), 0.6, vec![DockTab::AiChat]);
    let [_aichat, _log] = surface.split_right(aichat, 0.5, vec![DockTab::Log]);
    let [_main, _explorer] = surface.split_right(main, 0.72, vec![DockTab::Explorer]);
    state
}

// ---------------------------------------------------------------------------
// Renderer state
// ---------------------------------------------------------------------------

/// Held keys for nudging the selected object: move on the ground (fwd/back/
/// left/right), height (up/down), and yaw rotation (rot_left/rot_right).
#[derive(Default)]
struct NudgeKeys {
    fwd: bool,
    back: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    rot_left: bool,
    rot_right: bool,
}

/// Which transform gizmo is shown over the selected object.
#[derive(Clone, Copy, PartialEq, Eq)]
enum GizmoMode {
    Translate,
    Rotate,
    Scale,
}

/// Uniform scale clamp for the scale gizmo (10% … 200%).
const SCALE_MIN: f32 = 0.1;
const SCALE_MAX: f32 = 2.0;

impl NudgeKeys {
    fn any(&self) -> bool {
        self.fwd
            || self.back
            || self.left
            || self.right
            || self.up
            || self.down
            || self.rot_left
            || self.rot_right
    }
}

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

    entities: Vec<Entity>,
    /// Next entity id to assign (monotonic; ids stay stable for the session).
    next_id: u32,
    /// True when the scene/settings have unsaved changes (drives the Save Project ●).
    project_dirty: bool,
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
    controls: Controls,
    /// Held keys for nudging the selected object (WoW-Housing-style).
    nudge: NudgeKeys,
    /// True while the object is being nudged (to record one undo entry on stop).
    nudging: bool,
    /// Whether the ground grid is drawn (toggled with X).
    show_grid: bool,
    /// Which gizmo (move / rotate) is active for the selected object.
    gizmo_mode: GizmoMode,
    /// Screen rect (egui points) of the on-screen gizmo toolbar, so clicks on it
    /// don't fall through to camera orbit / pick.
    toolbar_rect: Option<egui::Rect>,
    /// Which gizmo axis (0=X,1=Y,2=Z) is being dragged, if any.
    gizmo_drag: Option<usize>,
    /// Scale-drag start state: uniform scale + cursor distance from center at grab.
    gizmo_scale_start: f32,
    gizmo_scale_dist0: f32,
    /// Whether Shift is held (fast keyboard nudge).
    shift_down: bool,
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
    /// Session action log + undo/redo history (newest last; cursor = current state).
    history: Vec<HistoryEntry>,
    history_cursor: usize,
    /// Dockable workspace layout + the 3D Scene tab's rect (in egui points).
    dock_state: DockState<DockTab>,
    scene_rect: Option<egui::Rect>,
    /// The CoE logo as an egui texture (None if assets/icon.png is missing).
    logo_texture: Option<egui::TextureHandle>,
    /// The blueprint splash art (assets/splash.png) — loading screen + bg.
    splash_texture: Option<egui::TextureHandle>,
    /// While Some and still in the future, a full-screen loading splash is shown
    /// (set at startup and whenever a project loads). Cleared once it elapses.
    loading_until: Option<Instant>,
    /// True until the *startup* splash finishes — then the window resizes from
    /// the splash size to the working size. Project-load splashes don't resize.
    startup_splash: bool,
    /// The folder of the currently open project (`project.json` lives inside).
    /// `None` until the scene is saved or a project is opened.
    project_path: Option<PathBuf>,
    /// The last-used project folder from the global config (for "Open last").
    startup_last_project: Option<PathBuf>,
    /// Lazy cache of decoded file contents for open `File` viewer tabs.
    file_cache: HashMap<PathBuf, FileView>,
    /// The in-engine terminal's shell (lazily started when the tab is shown).
    terminal: TerminalState,
    /// Git (Source Control) panel state.
    git: GitUi,
    /// A terminal command CoE-AI proposed, awaiting the user's approve/deny.
    pending_command: Option<PendingCommand>,
    /// Focus mode: when Some, holds the dock layout to restore on exit (the live
    /// `dock_state` is the single focused tab filling the workspace).
    focus_restore: Option<DockState<DockTab>>,
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

        let splash_texture = load_splash_rgba().map(|(rgba, w, h)| {
            let image = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba);
            egui_ctx.load_texture("coengine_splash", image, egui::TextureOptions::LINEAR)
        });

        // Seed this session from the global "last settings", and remember the
        // last-used project for the startup popup (only if it still exists).
        let cfg = load_global_config();
        let controls = cfg.controls.unwrap_or_default();
        let mut ui = UiState::default();
        if let Some(t) = cfg.theme {
            ui.theme = t;
        }
        if let Some(d) = cfg.dark_mode {
            ui.dark_mode = d;
        }
        if let Some(s) = cfg.shell {
            ui.shell = s;
        }
        let startup_last_project = cfg
            .last_project
            .filter(|p| p.join("project.json").is_file());
        if let Some(p) = &startup_last_project {
            ui.show_startup_popup = true;
            ui.last_project_name = p
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .or_else(|| Some(p.display().to_string()));
        }

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
            entities: Vec::new(),
            next_id: 1,
            project_dirty: false,
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
            controls,
            nudge: NudgeKeys::default(),
            nudging: false,
            show_grid: true,
            gizmo_mode: GizmoMode::Translate,
            toolbar_rect: None,
            gizmo_drag: None,
            gizmo_scale_start: 1.0,
            gizmo_scale_dist0: 1.0,
            shift_down: false,
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
            ui,
            history: vec![HistoryEntry {
                label: "Session start".to_string(),
                entities: Vec::new(),
                selected: None,
            }],
            history_cursor: 0,
            dock_state: build_dock_state(),
            scene_rect: None,
            logo_texture,
            splash_texture,
            // Show the splash for a moment while the first frames warm up.
            loading_until: Some(Instant::now() + Duration::from_millis(1400)),
            startup_splash: true,
            project_path: None,
            startup_last_project,
            file_cache: HashMap::new(),
            terminal: TerminalState::Off,
            git: GitUi::default(),
            pending_command: None,
            focus_restore: None,
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

    /// The 3D viewport rect in physical pixels — the Scene tab's area, or the full
    /// window when the Scene tab isn't currently visible.
    fn viewport_px(&self) -> (f32, f32, f32, f32) {
        let fw = self.config.width.max(1) as f32;
        let fh = self.config.height.max(1) as f32;
        match self.scene_rect {
            Some(r) => {
                let ppp = self.window.scale_factor() as f32;
                let x = (r.min.x * ppp).max(0.0);
                let y = (r.min.y * ppp).max(0.0);
                let w = (r.width() * ppp).clamp(1.0, (fw - x).max(1.0));
                let h = (r.height() * ppp).clamp(1.0, (fh - y).max(1.0));
                (x, y, w, h)
            }
            None => (0.0, 0.0, fw, fh),
        }
    }

    fn aspect(&self) -> f32 {
        let (_, _, w, h) = self.viewport_px();
        w / h.max(1.0)
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

    /// Build a fresh entity with the next id and a default name/transform.
    fn new_entity(&mut self, pos: Vec3, color: [f32; 3]) -> Entity {
        let id = self.next_id;
        self.next_id += 1;
        Entity {
            id,
            name: format!("Object {id}"),
            pos,
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
            color,
            kind: ShapeKind::Cube,
        }
    }

    fn add_cube(&mut self) {
        let (x, z) = grid_slot(self.entities.len());
        let e = self.new_entity(Vec3::new(x, CUBE_HALF, z), CUBE_BASE_COLOR);
        self.entities.push(e);
        self.rebuild_cubes();
        self.record_history("Add cube");
    }

    fn rebuild_cubes(&mut self) {
        if self.entities.is_empty() {
            self.cube_buffer = None;
            self.cube_vertex_count = 0;
            return;
        }
        let verts = build_scene_vertices(&self.entities, self.selected);
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
        let (vx, vy, vw, vh) = self.viewport_px();
        // Ignore clicks outside the 3D viewport rect.
        if cursor.0 < vx || cursor.0 > vx + vw || cursor.1 < vy || cursor.1 > vy + vh {
            return;
        }

        let ndc_x = ((cursor.0 - vx) / vw) * 2.0 - 1.0;
        let ndc_y = 1.0 - ((cursor.1 - vy) / vh) * 2.0;

        let inv = self.current_view_proj().inverse();
        let near = inv * Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
        let far = inv * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
        let near = near.truncate() / near.w;
        let far = far.truncate() / far.w;

        let origin = near;
        let dir = (far - near).normalize();

        // Ray-test in each entity's local space (handles rotation + scale): the
        // ray parameter `t` is preserved by the affine inverse, so it stays
        // comparable across entities for picking the nearest.
        let mut best: Option<(usize, f32)> = None;
        for (i, e) in self.entities.iter().enumerate() {
            let inv = e.model_matrix().inverse();
            let lo = inv.transform_point3(origin);
            let ld = inv.transform_vector3(dir);
            if let Some(t) = ray_aabb(lo, ld, Vec3::ZERO, CUBE_HALF) {
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

    /// Project the selected object's gizmo center + 3 axis tips into viewport
    /// pixels (matching `cursor_pos`/`viewport_px` space). None if no selection.
    fn gizmo_screen_points(&self) -> Option<((f32, f32), [(f32, f32); 3])> {
        let i = self.selected?;
        let obj = self.entities.get(i)?.pos;
        let vp = self.current_view_proj();
        let (vx, vy, vw, vh) = self.viewport_px();
        let proj = |w: Vec3| -> Option<(f32, f32)> {
            let clip = vp * w.extend(1.0);
            if clip.w <= 0.001 {
                return None;
            }
            let ndc = clip.truncate() / clip.w;
            Some((
                vx + (ndc.x * 0.5 + 0.5) * vw,
                vy + (1.0 - ndc.y) * 0.5 * vh,
            ))
        };
        let center = proj(obj)?;
        let tips = [
            proj(obj + Vec3::X * GIZMO_AXIS_LEN)?,
            proj(obj + Vec3::Y * GIZMO_AXIS_LEN)?,
            proj(obj + Vec3::Z * GIZMO_AXIS_LEN)?,
        ];
        Some((center, tips))
    }

    /// Is the cursor over the on-screen gizmo toolbar? (toolbar rect is in egui
    /// points; the cursor is physical px.)
    fn pointer_on_toolbar(&self) -> bool {
        let Some(tr) = self.toolbar_rect else {
            return false;
        };
        let ppp = self.window.scale_factor() as f32;
        let (cx, cy) = self.cursor_pos;
        cx >= tr.left() * ppp
            && cx <= tr.right() * ppp
            && cy >= tr.top() * ppp
            && cy <= tr.bottom() * ppp
    }

    /// Project a single world point into viewport pixels (None if behind camera).
    fn project_world(&self, w: Vec3) -> Option<(f32, f32)> {
        let clip = self.current_view_proj() * w.extend(1.0);
        if clip.w <= 0.001 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        let (vx, vy, vw, vh) = self.viewport_px();
        Some((vx + (ndc.x * 0.5 + 0.5) * vw, vy + (1.0 - ndc.y) * 0.5 * vh))
    }

    /// Which gizmo handle (translate-axis tip or rotate ring) is under `cursor`.
    fn gizmo_handle_hit(&self, cursor: (f32, f32)) -> Option<usize> {
        let i = self.selected?;
        let obj = self.entities.get(i)?.pos;
        let ppp = self.window.scale_factor() as f32;
        let mut best: Option<(usize, f32)> = None;
        match self.gizmo_mode {
            GizmoMode::Translate | GizmoMode::Scale => {
                let radius = 16.0 * ppp;
                for (ai, dir) in [Vec3::X, Vec3::Y, Vec3::Z].iter().enumerate() {
                    if let Some(t) = self.project_world(obj + *dir * GIZMO_AXIS_LEN) {
                        let d = ((t.0 - cursor.0).powi(2) + (t.1 - cursor.1).powi(2)).sqrt();
                        if d <= radius && best.map_or(true, |(_, bd)| d < bd) {
                            best = Some((ai, d));
                        }
                    }
                }
            }
            GizmoMode::Rotate => {
                // Hit-test the ball handle on each ring.
                let rot = self.entities[i].rotation;
                let radius = 22.0 * ppp;
                for axis in 0..3 {
                    if let Some(b) = self.project_world(rotate_ball_world(axis, obj, rot)) {
                        let d = ((b.0 - cursor.0).powi(2) + (b.1 - cursor.1).powi(2)).sqrt();
                        if d <= radius && best.map_or(true, |(_, bd)| d < bd) {
                            best = Some((axis, d));
                        }
                    }
                }
            }
        }
        best.map(|(i, _)| i)
    }

    /// Apply a mouse motion of (dx, dy) physical px to the selected object via
    /// the active gizmo: translate along axis `ax`, or rotate about it.
    fn gizmo_drag_motion(&mut self, ax: usize, dx: f32, dy: f32) {
        let Some(i) = self.selected else { return };
        if i >= self.entities.len() {
            return;
        }
        match self.gizmo_mode {
            GizmoMode::Translate => {
                let Some((center, tips)) = self.gizmo_screen_points() else {
                    return;
                };
                let (sx, sy) = (tips[ax].0 - center.0, tips[ax].1 - center.1);
                let slen = (sx * sx + sy * sy).sqrt().max(1.0);
                let along = (dx * sx + dy * sy) / slen;
                let dir = [Vec3::X, Vec3::Y, Vec3::Z][ax];
                self.entities[i].pos += dir * (along * GIZMO_AXIS_LEN / slen);
            }
            GizmoMode::Rotate => {
                // Rotate about the object's LOCAL axis so its (rigidly-attached)
                // ball follows the cursor along its tilted ring; store back as euler.
                let q = obj_quat(self.entities[i].rotation);
                let (ul, vl) = ring_basis(ax);
                let (u, v) = (q * ul, q * vl);
                let ballvec = q * (ROT_ANCHOR[ax] * GIZMO_AXIS_LEN);
                let alpha = ballvec.dot(v).atan2(ballvec.dot(u));
                let Some(phi) = self.cursor_ring_angle(ax) else {
                    return;
                };
                let local_axis = [Vec3::X, Vec3::Y, Vec3::Z][ax];
                let qnew = q * Quat::from_axis_angle(local_axis, phi - alpha);
                let (ex, ey, ez) = qnew.to_euler(EulerRot::XYZ);
                self.entities[i].rotation =
                    Vec3::new(ex.to_degrees(), ey.to_degrees(), ez.to_degrees());
            }
            GizmoMode::Scale => {
                // Uniform scale from how far the cursor is from the object center
                // relative to the grab distance, clamped to 10%–200%.
                let Some((center, _)) = self.gizmo_screen_points() else {
                    return;
                };
                let (cx, cy) = self.cursor_pos;
                let dist = ((cx - center.0).powi(2) + (cy - center.1).powi(2)).sqrt().max(1.0);
                let factor = dist / self.gizmo_scale_dist0.max(1.0);
                let s = (self.gizmo_scale_start * factor).clamp(SCALE_MIN, SCALE_MAX);
                self.entities[i].scale = Vec3::splat(s);
            }
        }
        self.project_dirty = true;
        self.rebuild_cubes();
    }

    /// Angle (radians) of the cursor's ray-vs-ring-plane hit point, measured in
    /// the object-tilted ring basis — i.e. where on `axis`'s ring the cursor points.
    fn cursor_ring_angle(&self, axis: usize) -> Option<f32> {
        let i = self.selected?;
        let e = self.entities.get(i)?;
        let obj = e.pos;
        let q = obj_quat(e.rotation);
        let (ul, vl) = ring_basis(axis);
        let (u, v) = (q * ul, q * vl);
        let normal = q * [Vec3::X, Vec3::Y, Vec3::Z][axis];
        let (vx, vy, vw, vh) = self.viewport_px();
        let (cx, cy) = self.cursor_pos;
        let ndc_x = ((cx - vx) / vw) * 2.0 - 1.0;
        let ndc_y = 1.0 - ((cy - vy) / vh) * 2.0;
        let inv = self.current_view_proj().inverse();
        let near = inv * Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
        let far = inv * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
        let origin = near.truncate() / near.w;
        let dir = (far.truncate() / far.w - origin).normalize();
        let denom = dir.dot(normal);
        if denom.abs() < 1e-5 {
            return None;
        }
        let t = (obj - origin).dot(normal) / denom;
        if t < 0.0 {
            return None;
        }
        let local = (origin + dir * t) - obj;
        Some(local.dot(v).atan2(local.dot(u)))
    }

    fn remove_selected(&mut self) {
        if let Some(i) = self.selected {
            if i < self.entities.len() {
                self.entities.remove(i);
            }
            self.selected = None;
            self.rebuild_cubes();
            self.record_history("Remove object");
            println!("Removed object; {} remain", self.entities.len());
        }
    }

    /// Push a snapshot of the current scene onto the history (dropping any redo tail).
    fn record_history(&mut self, label: impl Into<String>) {
        self.project_dirty = true;
        self.history.truncate(self.history_cursor + 1);
        self.history.push(HistoryEntry {
            label: label.into(),
            entities: self.entities.clone(),
            selected: self.selected,
        });
        const MAX_HISTORY: usize = 100;
        if self.history.len() > MAX_HISTORY {
            let drop = self.history.len() - MAX_HISTORY;
            self.history.drain(0..drop);
        }
        self.history_cursor = self.history.len() - 1;
    }

    fn history_undo(&mut self) {
        if self.history_cursor > 0 {
            self.history_cursor -= 1;
            self.restore_history();
        }
    }

    fn history_redo(&mut self) {
        if self.history_cursor + 1 < self.history.len() {
            self.history_cursor += 1;
            self.restore_history();
        }
    }

    /// Restore the scene to the snapshot at the current history cursor.
    fn restore_history(&mut self) {
        let entry = self.history[self.history_cursor].clone();
        self.entities = entry.entities;
        self.selected = entry.selected;
        self.rebuild_cubes();
    }

    // ---- Project persistence (v0.0.12) -------------------------------------

    /// Snapshot the whole project (scene + per-project settings + layout +
    /// window geometry) into the on-disk manifest shape.
    fn gather_project(&self) -> Project {
        Project {
            format_version: PROJECT_FORMAT_VERSION,
            engine_version: CO_VERSION.to_string(),
            scene: self.entities.clone(),
            theme: self.ui.theme,
            dark_mode: self.ui.dark_mode,
            controls: self.controls,
            shell: self.ui.shell,
            // In Focus mode, save the real (un-focused) layout, not the single tab.
            dock_state: self
                .focus_restore
                .clone()
                .unwrap_or_else(|| self.dock_state.clone()),
            window_size: Some((self.config.width, self.config.height)),
            window_pos: self
                .window
                .outer_position()
                .ok()
                .map(|p| (p.x, p.y)),
        }
    }

    /// Replace the live engine state with a loaded project, rebuilding GPU
    /// buffers and resetting the undo history to the loaded scene as baseline.
    fn apply_project(&mut self, p: Project) {
        self.entities = p.scene;
        // Normalize ids/names so old projects (ids default to 0) and any
        // duplicates get unique, stable ids; reset the id counter past them.
        for (i, e) in self.entities.iter_mut().enumerate() {
            e.id = i as u32 + 1;
            if e.name.trim().is_empty() {
                e.name = format!("Object {}", e.id);
            }
        }
        self.next_id = self.entities.len() as u32 + 1;
        self.selected = None;
        self.project_dirty = false;
        self.rebuild_cubes();

        self.ui.theme = p.theme;
        self.ui.dark_mode = p.dark_mode;
        self.controls = p.controls;
        self.ui.shell = p.shell;
        self.dock_state = p.dock_state;
        // Loading a project leaves Focus mode (the manifest holds the full layout).
        self.focus_restore = None;
        // Restart the terminal so it uses the loaded project's shell.
        self.terminal = TerminalState::Off;

        // The loaded scene becomes the new history baseline (undo can't go past it).
        self.history = vec![HistoryEntry {
            label: "Opened project".to_string(),
            entities: self.entities.clone(),
            selected: None,
        }];
        self.history_cursor = 0;

        // Briefly show the loading splash while the loaded project settles in.
        self.loading_until = Some(Instant::now() + Duration::from_millis(700));

        // Restore window geometry. Resizing triggers the normal resize path which
        // reconfigures the surface; the camera aspect re-syncs each frame.
        if let Some((w, h)) = p.window_size {
            let _ = self.window.request_inner_size(PhysicalSize::new(w, h));
        }
        if let Some((x, y)) = p.window_pos {
            self.window.set_outer_position(PhysicalPosition::new(x, y));
        }
    }

    /// Write `project.json` into the given project folder.
    fn write_project(&self, dir: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self.gather_project())
            .map_err(std::io::Error::other)?;
        std::fs::write(dir.join("project.json"), json)
    }

    /// Save to the current project folder, or fall back to "Save As" if none yet.
    fn save_project(&mut self) {
        match self.project_path.clone() {
            Some(dir) => self.save_into(&dir),
            None => self.save_project_as(),
        }
    }

    /// Prompt for a folder, then save the project there and adopt it as current.
    fn save_project_as(&mut self) {
        if let Some(dir) = rfd::FileDialog::new()
            .set_title("Choose a folder for this CoEngine project")
            .pick_folder()
        {
            self.save_into(&dir);
        }
    }

    /// Shared save body: write the manifest, adopt the folder, report status.
    fn save_into(&mut self, dir: &Path) {
        match self.write_project(dir) {
            Ok(()) => {
                self.project_path = Some(dir.to_path_buf());
                self.project_dirty = false;
                self.ui.project_status = Some(format!("Saved to {}", dir.display()));
                println!("Saved project to {}", dir.display());
                self.update_global_config();
            }
            Err(e) => {
                self.ui.project_status = Some(format!("Save failed: {e}"));
                eprintln!("Save failed: {e}");
            }
        }
    }

    /// Prompt for a project folder, then open its `project.json`.
    fn open_project(&mut self) {
        if let Some(dir) = rfd::FileDialog::new()
            .set_title("Open a CoEngine project folder")
            .pick_folder()
        {
            self.open_project_at(&dir);
        }
    }

    /// Load the `project.json` in a known folder (used by Open and the startup
    /// "Open last" button — no file dialog).
    fn open_project_at(&mut self, dir: &Path) {
        let manifest = dir.join("project.json");
        let text = match std::fs::read_to_string(&manifest) {
            Ok(t) => t,
            Err(e) => {
                self.ui.project_status = Some(format!("No project.json here: {e}"));
                eprintln!("Open failed: {e}");
                return;
            }
        };
        match serde_json::from_str::<Project>(&text) {
            Ok(p) => {
                if p.format_version != PROJECT_FORMAT_VERSION {
                    self.ui.project_status = Some(format!(
                        "Unsupported project format v{} (this build expects v{PROJECT_FORMAT_VERSION})",
                        p.format_version
                    ));
                    return;
                }
                self.apply_project(p);
                self.project_path = Some(dir.to_path_buf());
                self.ui.project_status = Some(format!("Opened {}", dir.display()));
                println!("Opened project {}", dir.display());
                self.update_global_config();
            }
            Err(e) => {
                self.ui.project_status = Some(format!("Couldn't parse project.json: {e}"));
                eprintln!("Parse failed: {e}");
            }
        }
    }

    /// Start a new, empty project: pick a folder, reset to the default layout +
    /// empty scene (keeping the seeded settings), then save it there.
    fn new_project(&mut self) {
        let Some(dir) = rfd::FileDialog::new()
            .set_title("Choose a folder for the new CoEngine project")
            .pick_folder()
        else {
            return;
        };
        self.entities.clear();
        self.next_id = 1;
        self.selected = None;
        self.rebuild_cubes();
        self.dock_state = build_dock_state();
        self.history = vec![HistoryEntry {
            label: "New project".to_string(),
            entities: Vec::new(),
            selected: None,
        }];
        self.history_cursor = 0;
        self.loading_until = Some(Instant::now() + Duration::from_millis(700));
        self.save_into(&dir);
    }

    /// Persist the global "last settings" + last project to `%APPDATA%`. Keeps
    /// the prior last-project if none was opened/saved this session.
    fn update_global_config(&self) {
        save_global_config(&GlobalConfig {
            last_project: self
                .project_path
                .clone()
                .or_else(|| self.startup_last_project.clone()),
            theme: Some(self.ui.theme),
            dark_mode: Some(self.ui.dark_mode),
            controls: Some(self.controls),
            shell: Some(self.ui.shell),
        });
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
        // Undo / Redo requested from the Log panel during the previous frame.
        if std::mem::take(&mut self.ui.undo_requested) {
            self.history_undo();
        }
        if std::mem::take(&mut self.ui.redo_requested) {
            self.history_redo();
        }

        // Scene outliner (Logic tab) requests: select / delete / add.
        if let Some(i) = self.ui.outliner_select.take() {
            if i < self.entities.len() {
                self.selected = Some(i);
                self.rebuild_cubes();
            }
        }
        if let Some(i) = self.ui.outliner_delete.take() {
            if i < self.entities.len() {
                self.entities.remove(i);
                self.selected = match self.selected {
                    Some(s) if s == i => None,
                    Some(s) if s > i => Some(s - 1),
                    other => other,
                };
                self.rebuild_cubes();
                self.record_history("Delete object");
            }
        }
        if std::mem::take(&mut self.ui.outliner_add) {
            self.add_cube();
        }

        // Inspector edits: apply live to the selected entity; record one undo
        // entry when the edit finishes (drag released / field committed).
        if let Some(edited) = self.ui.inspector_apply.take() {
            if let Some(i) = self.selected {
                if i < self.entities.len() {
                    self.entities[i] = edited;
                    self.project_dirty = true;
                    self.rebuild_cubes();
                }
            }
        }
        if std::mem::take(&mut self.ui.inspector_commit) {
            self.record_history("Edit object");
        }
        if let Some(m) = self.ui.gizmo_mode_req.take() {
            self.gizmo_mode = m;
        }
        if let Some(s) = self.ui.scale_req.take() {
            if let Some(i) = self.selected {
                if i < self.entities.len() {
                    self.entities[i].scale = Vec3::splat(s.clamp(SCALE_MIN, SCALE_MAX));
                    self.project_dirty = true;
                    self.rebuild_cubes();
                }
            }
        }
        if std::mem::take(&mut self.ui.scale_commit) {
            self.record_history("Scale object");
        }


        // Collect the result of an in-flight async git operation, if it finished.
        let mut git_done: Option<String> = None;
        if let Some(rx) = &self.git.rx {
            if let Ok(out) = rx.try_recv() {
                git_done = Some(out);
            }
        }
        if let Some(out) = git_done {
            self.git.output = out;
            self.git.rx = None;
        }

        // Clear the loading splash once its timer elapses. The first (startup)
        // splash then grows the window from the splash size to the working size.
        if let Some(t) = self.loading_until {
            if Instant::now() >= t {
                self.loading_until = None;
                if self.startup_splash {
                    self.startup_splash = false;
                    let _ = self.window.request_inner_size(LogicalSize::new(1280.0, 720.0));
                }
            }
        }

        // Project Open / Save / Save As / New requested from the Menu or the
        // startup popup last frame.
        if std::mem::take(&mut self.ui.open_requested) {
            self.open_project();
        }
        if std::mem::take(&mut self.ui.open_last_requested) {
            if let Some(dir) = self.startup_last_project.clone() {
                self.open_project_at(&dir);
            }
        }
        if std::mem::take(&mut self.ui.new_requested) {
            self.new_project();
        }
        if std::mem::take(&mut self.ui.save_requested) {
            self.save_project();
        }
        if std::mem::take(&mut self.ui.save_as_requested) {
            self.save_project_as();
        }

        // Keep the camera projection's aspect locked to the current viewport rect
        // every frame, so resizing the window or dock never warps the 3D scene.
        self.update_camera();

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

        // Nudge the selected object with the held arrow/PageUp-Down/comma-period
        // keys (forward = -Z; right = +X; up = +Y; rotation about Y).
        if self.nudge.any() {
            if let Some(i) = self.selected {
                if i < self.entities.len() {
                    // Deliberately slow — the keyboard is for fine precision (the
                    // last 5%); the on-screen gizmos handle fast/bulk movement.
                    // Holding Shift moves much faster.
                    let fast = if self.shift_down { 16.0 } else { 1.0 };
                    let mv = 0.5 * fast * dt; // units / second
                    let rot = 8.0 * fast * dt; // degrees / second
                    let e = &mut self.entities[i];
                    if self.nudge.fwd {
                        e.pos.z -= mv;
                    }
                    if self.nudge.back {
                        e.pos.z += mv;
                    }
                    if self.nudge.left {
                        e.pos.x -= mv;
                    }
                    if self.nudge.right {
                        e.pos.x += mv;
                    }
                    if self.nudge.up {
                        e.pos.y += mv;
                    }
                    if self.nudge.down {
                        e.pos.y -= mv;
                    }
                    if self.nudge.rot_left {
                        e.rotation.y += rot;
                    }
                    if self.nudge.rot_right {
                        e.rotation.y -= rot;
                    }
                    self.rebuild_cubes();
                    self.project_dirty = true;
                    self.nudging = true;
                }
            }
        } else if self.nudging {
            // Nudging just stopped — record one undo entry for the whole move.
            self.nudging = false;
            self.record_history("Move object");
        }

        // Drain agent output (text + scene commands) since last frame.
        let mut deltas: Vec<String> = Vec::new();
        let mut commands: Vec<SceneCommand> = Vec::new();
        let mut claude_prompt: Option<String> = None;
        let mut command_request: Option<PendingCommand> = None;
        let mut done = false;
        let mut error: Option<String> = None;
        if let Some(rx) = &self.chat.rx {
            loop {
                match rx.try_recv() {
                    Ok(StreamMsg::Delta(t)) => deltas.push(t),
                    Ok(StreamMsg::Command(c)) => commands.push(c),
                    Ok(StreamMsg::ClaudePrompt(p)) => claude_prompt = Some(p),
                    Ok(StreamMsg::CommandRequest { command, reply }) => {
                        command_request = Some(PendingCommand { command, reply });
                        // Stop draining; wait for the user to approve/deny before
                        // more agent traffic (the worker is blocked on the reply).
                        break;
                    }
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

        // Apply scene commands the agent issued (each scene change is recorded for undo).
        if !commands.is_empty() {
            for cmd in commands {
                match cmd {
                    SceneCommand::Add { x, z, color } => {
                        let e = self.new_entity(Vec3::new(x, CUBE_HALF, z), color);
                        self.entities.push(e);
                        self.record_history("CoE-AI: add cube");
                    }
                    SceneCommand::SetColor { index, color } => {
                        if index < self.entities.len() {
                            self.entities[index].color = color;
                            self.record_history("CoE-AI: recolor cube");
                        }
                    }
                    SceneCommand::Remove { index } => {
                        if index < self.entities.len() {
                            self.entities.remove(index);
                            self.selected = match self.selected {
                                Some(s) if s == index => None,
                                Some(s) if s > index => Some(s - 1),
                                other => other,
                            };
                            self.record_history("CoE-AI: remove cube");
                        }
                    }
                    SceneCommand::Select { index } => {
                        if index < self.entities.len() {
                            self.selected = Some(index);
                        }
                    }
                    SceneCommand::Clear => {
                        self.entities.clear();
                        self.selected = None;
                        self.record_history("CoE-AI: clear scene");
                    }
                }
            }
            self.rebuild_cubes();
        }

        if let Some(p) = claude_prompt {
            self.chat.pending_claude_prompt = Some(p);
        }

        if command_request.is_some() {
            self.pending_command = command_request;
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
        // Only draw the 3D when the Scene tab is actually visible (Some rect).
        let scene_viewport = self.scene_rect.map(|_| self.viewport_px());

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

            if let Some((vx, vy, vw, vh)) = scene_viewport {
                pass.set_viewport(vx, vy, vw, vh, 0.0, 1.0);
                pass.set_scissor_rect(vx as u32, vy as u32, vw as u32, vh as u32);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);

                if self.show_grid {
                    pass.set_pipeline(&self.grid_pipeline);
                    pass.set_vertex_buffer(0, self.grid_buffer.slice(..));
                    pass.draw(0..self.grid_vertex_count, 0..1);
                }

                if let Some(cube_buffer) = &self.cube_buffer {
                    pass.set_pipeline(&self.cube_pipeline);
                    pass.set_vertex_buffer(0, cube_buffer.slice(..));
                    pass.draw(0..self.cube_vertex_count, 0..1);
                }
            }
        }

        // --- egui UI over the scene ---
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let ctx = self.egui_ctx.clone();
        let mode = self.mode;
        let logo = self.logo_texture.as_ref();
        let splash = self.splash_texture.as_ref();
        let loading = self.loading_until.is_some();
        let scene = SceneSnapshot {
            cubes: self.entities.iter().map(|e| (e.pos, e.color)).collect(),
            selected: self.selected,
        };
        let outliner: Vec<String> = self.entities.iter().map(|e| e.name.clone()).collect();
        let inspector = self.selected.and_then(|i| self.entities.get(i).cloned());
        let view_proj = self.current_view_proj();
        let gizmo_axis = self.gizmo_drag;
        let gizmo_mode = self.gizmo_mode;
        let project_path = self.project_path.clone();
        let project_dirty = self.project_dirty;
        let file_cache = &mut self.file_cache;
        let terminal = &mut self.terminal;
        let git = &mut self.git;
        let pending_command = &mut self.pending_command;
        let focus_restore = &mut self.focus_restore;
        let chat = &mut self.chat;
        let ui_state = &mut self.ui;
        let controls = &self.controls;
        let history = &self.history;
        let history_cursor = self.history_cursor;
        let dock_state = &mut self.dock_state;
        let mut captured_scene_rect: Option<egui::Rect> = None;
        let mut captured_toolbar_rect: Option<egui::Rect> = None;
        let full_output = ctx.run(raw_input, |c| {
            build_ui(
                c,
                ui_state,
                chat,
                mode,
                logo,
                splash,
                loading,
                view_proj,
                gizmo_axis,
                gizmo_mode,
                &scene,
                &outliner,
                inspector.as_ref(),
                controls,
                history,
                history_cursor,
                project_path.as_deref(),
                project_dirty,
                file_cache,
                terminal,
                git,
                pending_command,
                focus_restore,
                dock_state,
                &mut captured_scene_rect,
                &mut captured_toolbar_rect,
            )
        });
        self.scene_rect = captured_scene_rect;
        self.toolbar_rect = captured_toolbar_rect;
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
    load_png_rgba("assets/icon.png")
}

/// Load the blueprint splash art from `assets/splash.png` (loading screen +
/// 50%-opacity empty-area background). None if missing — the app still runs.
fn load_splash_rgba() -> Option<(Vec<u8>, u32, u32)> {
    load_png_rgba("assets/splash.png")
}

/// Decode a PNG on disk to RGBA8 bytes + dimensions.
fn load_png_rgba(path: &str) -> Option<(Vec<u8>, u32, u32)> {
    let bytes = std::fs::read(path).ok()?;
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

        // Open at the splash art's 4:3 size so the startup splash shows undistorted;
        // State resizes the window to the working size once loading finishes.
        let mut attributes = Window::default_attributes()
            .with_title(title_for(CameraMode::Orbit))
            .with_inner_size(LogicalSize::new(900.0, 675.0));

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

        // While rebinding a control, the next key press is captured for that action
        // (Esc cancels). Done before egui so any key — even Tab — can be bound.
        if let WindowEvent::KeyboardInput { event: ke, .. } = &event {
            if let Some(action) = state.ui.rebinding {
                if ke.state == ElementState::Pressed && !ke.repeat {
                    if let PhysicalKey::Code(code) = ke.physical_key {
                        if code == KeyCode::Escape {
                            state.ui.rebinding = None;
                        } else {
                            state.controls.set(action, code);
                            state.project_dirty = true;
                            state.ui.rebinding = None;
                        }
                    }
                }
                return;
            }
        }

        // Tab is swallowed before egui (so it can't focus-cycle into the chat box);
        // if Tab is the camera-toggle key, toggle the camera here.
        if let WindowEvent::KeyboardInput { event: ke, .. } = &event {
            if ke.physical_key == PhysicalKey::Code(KeyCode::Tab) {
                if ke.state == ElementState::Pressed
                    && !ke.repeat
                    && state.controls.toggle_camera == KeyCode::Tab
                {
                    state.toggle_mode();
                }
                return;
            }
        }

        let egui_consumed = state
            .egui_state
            .on_window_event(&state.window, &event)
            .consumed;

        // Always track the cursor — 3D picking and orbit need the latest position.
        if let WindowEvent::CursorMoved { position, .. } = &event {
            state.cursor_pos = (position.x as f32, position.y as f32);
        }

        // The 3D viewport only reacts when the cursor is inside its rect. Dock split
        // handles and tab bars sit outside this rect, so resizing/dragging them never
        // moves the camera; egui keeps the mouse everywhere else.
        let in_scene = match state.scene_rect {
            Some(_) => {
                let (vx, vy, vw, vh) = state.viewport_px();
                let (cx, cy) = state.cursor_pos;
                cx >= vx && cx <= vx + vw && cy >= vy && cy <= vy + vh
            }
            None => false,
        };

        match event {
            WindowEvent::KeyboardInput {
                event: key_event, ..
            } => {
                // Let egui keep keyboard input when it's focused (e.g. typing in chat).
                if egui_consumed {
                    return;
                }
                if let PhysicalKey::Code(code) = key_event.physical_key {
                    let pressed = key_event.state == ElementState::Pressed;
                    let first_press = pressed && !key_event.repeat;
                    let c = state.controls;
                    if code == c.forward {
                        state.keys.forward = pressed;
                    } else if code == c.back {
                        state.keys.back = pressed;
                    } else if code == c.left {
                        state.keys.left = pressed;
                    } else if code == c.right {
                        state.keys.right = pressed;
                    } else if code == c.up {
                        state.keys.up = pressed;
                    } else if code == c.down {
                        state.keys.down = pressed;
                    } else if first_press && code == c.add_cube {
                        state.add_cube();
                    } else if first_press && code == c.remove {
                        state.remove_selected();
                    } else if first_press && code == c.toggle_debug {
                        state.ui.show_debug = !state.ui.show_debug;
                    } else if first_press && code == c.toggle_menu {
                        state.toggle_menu();
                    } else if first_press && code == c.toggle_camera {
                        // Non-Tab camera key (Tab is handled before egui above).
                        state.toggle_mode();
                    }

                    // WoW-Housing-style nudge keys for the selected object, plus the
                    // grid toggle (these are fixed, not remappable for now).
                    match code {
                        KeyCode::ArrowUp => state.nudge.fwd = pressed,
                        KeyCode::ArrowDown => state.nudge.back = pressed,
                        KeyCode::ArrowLeft => state.nudge.left = pressed,
                        KeyCode::ArrowRight => state.nudge.right = pressed,
                        KeyCode::PageUp => state.nudge.up = pressed,
                        KeyCode::PageDown => state.nudge.down = pressed,
                        KeyCode::Comma => state.nudge.rot_left = pressed,
                        KeyCode::Period => state.nudge.rot_right = pressed,
                        KeyCode::KeyX if first_press => state.show_grid = !state.show_grid,
                        KeyCode::KeyR if first_press => {
                            state.gizmo_mode = match state.gizmo_mode {
                                GizmoMode::Translate => GizmoMode::Rotate,
                                GizmoMode::Rotate => GizmoMode::Scale,
                                GizmoMode::Scale => GizmoMode::Translate,
                            };
                        }
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
                        if pressed {
                            if in_scene && !state.pointer_on_toolbar() {
                                // A gizmo handle takes priority; otherwise orbit/select.
                                if let Some(ax) = state.gizmo_handle_hit(state.cursor_pos) {
                                    state.gizmo_drag = Some(ax);
                                    // Capture scale-grab state so scaling is relative.
                                    if state.gizmo_mode == GizmoMode::Scale {
                                        if let (Some(i), Some((center, _))) =
                                            (state.selected, state.gizmo_screen_points())
                                        {
                                            let (cx, cy) = state.cursor_pos;
                                            state.gizmo_scale_dist0 = ((cx - center.0).powi(2)
                                                + (cy - center.1).powi(2))
                                            .sqrt()
                                            .max(1.0);
                                            state.gizmo_scale_start = state.entities[i].scale.x;
                                        }
                                    }
                                } else {
                                    state.mouse_left_down = true;
                                    state.left_drag_dist = 0.0;
                                }
                            }
                        } else if state.gizmo_drag.take().is_some() {
                            // Finished a gizmo drag — record one undo entry.
                            state.record_history("Move object");
                        } else {
                            // Pick only if the drag started as a real scene click.
                            let was_down = std::mem::take(&mut state.mouse_left_down);
                            if was_down && state.left_drag_dist < 5.0 {
                                let cursor = state.cursor_pos;
                                state.pick(cursor);
                            }
                        }
                    }
                    MouseButton::Right => {
                        if pressed {
                            if in_scene {
                                state.mouse_right_down = true;
                            }
                        } else {
                            state.mouse_right_down = false;
                        }
                    }
                    _ => {}
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                if !in_scene {
                    return;
                }
                if state.mode == CameraMode::Orbit {
                    let scroll = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32 / 120.0,
                    };
                    state.orbit.zoom(scroll);
                    state.update_camera();
                }
            }

            WindowEvent::Focused(false) => {
                state.keys = Keys::default();
                state.nudge = NudgeKeys::default();
                state.shift_down = false;
            }

            WindowEvent::ModifiersChanged(mods) => {
                state.shift_down = mods.state().shift_key();
            }

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

            // Dragging a gizmo axis takes over the mouse (no camera movement).
            if let Some(ax) = state.gizmo_drag {
                state.gizmo_drag_motion(ax, dx, dy);
                return;
            }

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

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Persist the global "last settings" + last project on quit, so the next
        // launch can seed settings and offer to reopen the project.
        if let Some(state) = &self.state {
            state.update_global_config();
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
