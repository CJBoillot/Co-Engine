//! The in-engine terminal: shell selection, a one-shot captured runner, a live
//! PTY-backed session, and the Terminal tab renderer.

use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;

use portable_pty::{Child, CommandBuilder, MasterPty, PtySize};
use serde::{Deserialize, Serialize};

use crate::theme::ACCENT_GOLD;

/// Which shell the in-engine terminal launches (Settings → Terminal).
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub(crate) enum Shell {
    #[default]
    PowerShell,
    Pwsh,
    Cmd,
}

impl Shell {
    /// The executable to launch.
    pub(crate) fn command(self) -> &'static str {
        match self {
            Shell::PowerShell => "powershell.exe",
            Shell::Pwsh => "pwsh.exe",
            Shell::Cmd => "cmd.exe",
        }
    }

    /// Human-friendly name for the Settings picker.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Shell::PowerShell => "PowerShell",
            Shell::Pwsh => "PowerShell 7 (pwsh)",
            Shell::Cmd => "Command Prompt (cmd)",
        }
    }
}

/// Run a one-shot command through the chosen shell in `cwd`, capturing
/// stdout+stderr as a single string (truncated if very long). Used by CoE-AI's
/// `run_command` tool — separate from the interactive Terminal so output is
/// cleanly captured.
pub(crate) fn run_captured(shell: Shell, command: &str, cwd: Option<&Path>) -> String {
    let mut c = std::process::Command::new(shell.command());
    match shell {
        Shell::Cmd => {
            c.arg("/C").arg(command);
        }
        Shell::PowerShell | Shell::Pwsh => {
            c.arg("-NoProfile").arg("-Command").arg(command);
        }
    }
    if let Some(d) = cwd {
        c.current_dir(d);
    }
    match c.output() {
        Ok(out) => {
            let mut s = String::from_utf8_lossy(&out.stdout).into_owned();
            let err = String::from_utf8_lossy(&out.stderr);
            if !err.trim().is_empty() {
                if !s.is_empty() {
                    s.push('\n');
                }
                s.push_str(&err);
            }
            if s.trim().is_empty() {
                s = format!("(no output; exit code {})", out.status.code().unwrap_or(-1));
            }
            if s.len() > 8000 {
                s.truncate(8000);
                s.push_str("\n…(output truncated)");
            }
            s
        }
        Err(e) => format!("Failed to run command: {e}"),
    }
}

/// A live shell running in a pseudo-terminal (ConPTY on Windows). A background
/// thread feeds the shell's output into a `vt100` parser (the screen state);
/// `send` writes input to the shell; `resize` keeps the PTY + parser in sync.
pub(crate) struct TerminalSession {
    /// Shared screen state, updated by the reader thread, read by the UI.
    parser: Arc<Mutex<vt100::Parser>>,
    /// Input side of the PTY (keystrokes/commands go here).
    writer: Box<dyn Write + Send>,
    /// Kept alive for resize; dropping it closes the PTY.
    master: Box<dyn MasterPty + Send>,
    /// The shell process; killed on drop.
    child: Box<dyn Child + Send + Sync>,
    rows: u16,
    cols: u16,
}

impl TerminalSession {
    pub(crate) fn spawn(shell: &str, cwd: Option<&Path>) -> std::io::Result<Self> {
        let (rows, cols) = (24u16, 80u16);
        let pair = portable_pty::native_pty_system()
            .openpty(PtySize {
                rows,
                cols,
                pixel_width: 0,
                pixel_height: 0,
            })
            .map_err(|e| std::io::Error::other(e.to_string()))?;

        let mut cmd = CommandBuilder::new(shell);
        if let Some(dir) = cwd {
            cmd.cwd(dir);
        }
        let child = pair
            .slave
            .spawn_command(cmd)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        drop(pair.slave); // release the slave so EOF propagates when the shell exits

        let mut reader = pair
            .master
            .try_clone_reader()
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        let writer = pair
            .master
            .take_writer()
            .map_err(|e| std::io::Error::other(e.to_string()))?;

        let parser = Arc::new(Mutex::new(vt100::Parser::new(rows, cols, 2000)));
        let parser_rx = parser.clone();
        std::thread::spawn(move || {
            let mut buf = [0u8; 8192];
            loop {
                match reader.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        if let Ok(mut p) = parser_rx.lock() {
                            p.process(&buf[..n]);
                        }
                    }
                }
            }
        });

        Ok(Self {
            parser,
            writer,
            master: pair.master,
            child,
            rows,
            cols,
        })
    }

    /// Send bytes (typed input or a command) to the shell.
    fn send(&mut self, bytes: &[u8]) {
        let _ = self.writer.write_all(bytes);
        let _ = self.writer.flush();
    }

    /// Resize the PTY and the parser to a new grid size.
    fn resize(&mut self, rows: u16, cols: u16) {
        if (rows, cols) == (self.rows, self.cols) || rows == 0 || cols == 0 {
            return;
        }
        self.rows = rows;
        self.cols = cols;
        let _ = self.master.resize(PtySize {
            rows,
            cols,
            pixel_width: 0,
            pixel_height: 0,
        });
        if let Ok(mut p) = self.parser.lock() {
            p.set_size(rows, cols);
        }
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

/// Lifecycle of the Terminal tab's shell (lazily started the first time the tab
/// is shown, so we don't spawn a shell unless the terminal is actually used).
pub(crate) enum TerminalState {
    Off,
    Running(TerminalSession),
    Failed(String),
}

const TERM_BG: egui::Color32 = egui::Color32::from_rgb(16, 16, 20);
const TERM_FG: egui::Color32 = egui::Color32::from_rgb(222, 222, 222);

/// Map an ANSI 256-color index to an RGB color.
fn ansi_256(i: u8) -> egui::Color32 {
    const BASE: [(u8, u8, u8); 16] = [
        (0, 0, 0),
        (205, 49, 49),
        (13, 188, 121),
        (229, 229, 16),
        (36, 114, 200),
        (188, 63, 188),
        (17, 168, 205),
        (229, 229, 229),
        (102, 102, 102),
        (241, 76, 76),
        (35, 209, 139),
        (245, 245, 67),
        (59, 142, 234),
        (214, 112, 214),
        (41, 184, 219),
        (255, 255, 255),
    ];
    if (i as usize) < 16 {
        let (r, g, b) = BASE[i as usize];
        egui::Color32::from_rgb(r, g, b)
    } else if i >= 232 {
        let v = (8 + (i as u16 - 232) * 10).min(255) as u8;
        egui::Color32::from_rgb(v, v, v)
    } else {
        let i = i - 16;
        let conv = |n: u8| if n == 0 { 0 } else { 55 + n * 40 };
        egui::Color32::from_rgb(conv(i / 36), conv((i % 36) / 6), conv(i % 6))
    }
}

/// Resolve a vt100 cell color to an egui color (using `default` for Default).
fn vt_to_color(c: vt100::Color, default: egui::Color32) -> egui::Color32 {
    match c {
        vt100::Color::Default => default,
        vt100::Color::Idx(i) => ansi_256(i),
        vt100::Color::Rgb(r, g, b) => egui::Color32::from_rgb(r, g, b),
    }
}

/// Map Ctrl+letter to its control byte (Ctrl+A = 0x01 … Ctrl+Z = 0x1a).
fn ctrl_byte(key: egui::Key) -> Option<u8> {
    let name = key.name().as_bytes();
    (name.len() == 1 && name[0].is_ascii_alphabetic()).then(|| name[0].to_ascii_uppercase() & 0x1f)
}

/// Translate this frame's egui input events into bytes to write to the shell.
fn translate_terminal_input(events: &[egui::Event]) -> Vec<u8> {
    let mut out = Vec::new();
    for ev in events {
        match ev {
            egui::Event::Text(t) => out.extend_from_slice(t.as_bytes()),
            egui::Event::Key {
                key,
                pressed: true,
                modifiers,
                ..
            } => {
                if (modifiers.ctrl || modifiers.command) && !modifiers.alt {
                    if let Some(b) = ctrl_byte(*key) {
                        out.push(b);
                        continue;
                    }
                }
                match key {
                    egui::Key::Enter => out.push(b'\r'),
                    egui::Key::Backspace => out.push(0x7f),
                    egui::Key::Tab => out.push(b'\t'),
                    egui::Key::Escape => out.push(0x1b),
                    egui::Key::ArrowUp => out.extend_from_slice(b"\x1b[A"),
                    egui::Key::ArrowDown => out.extend_from_slice(b"\x1b[B"),
                    egui::Key::ArrowRight => out.extend_from_slice(b"\x1b[C"),
                    egui::Key::ArrowLeft => out.extend_from_slice(b"\x1b[D"),
                    egui::Key::Home => out.extend_from_slice(b"\x1b[H"),
                    egui::Key::End => out.extend_from_slice(b"\x1b[F"),
                    egui::Key::Delete => out.extend_from_slice(b"\x1b[3~"),
                    egui::Key::PageUp => out.extend_from_slice(b"\x1b[5~"),
                    egui::Key::PageDown => out.extend_from_slice(b"\x1b[6~"),
                    _ => {}
                }
            }
            _ => {}
        }
    }
    out
}

/// Render the Terminal tab: start the shell on first show; draw the live colored
/// grid + cursor; resize the PTY to the tab; route keystrokes to the shell when
/// the terminal has focus (click to focus).
pub(crate) fn terminal_tab_ui(ui: &mut egui::Ui, term: &mut TerminalState, cwd: Option<&Path>, shell: &str) {
    match term {
        TerminalState::Off => {
            *term = match TerminalSession::spawn(shell, cwd) {
                Ok(t) => TerminalState::Running(t),
                Err(e) => TerminalState::Failed(format!("Couldn't start terminal: {e}")),
            };
            ui.label("Starting terminal…");
            ui.ctx().request_repaint();
        }
        TerminalState::Running(t) => {
            let font_id = egui::FontId::monospace(13.0);
            let (char_w, row_h) = ui.fonts(|f| {
                (
                    f.glyph_width(&font_id, 'M').max(1.0),
                    f.row_height(&font_id).max(1.0),
                )
            });
            let avail = ui.available_size();
            let cols = ((avail.x / char_w).floor() as i32).clamp(1, 400) as u16;
            let rows = ((avail.y / row_h).floor() as i32).clamp(1, 200) as u16;
            t.resize(rows, cols);

            let (rect, resp) = ui.allocate_exact_size(avail, egui::Sense::click());
            if resp.clicked() {
                resp.request_focus();
            }
            let focused = resp.has_focus();

            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 0.0, TERM_BG);
            if let Ok(parser) = t.parser.lock() {
                let screen = parser.screen();
                for row in 0..rows {
                    for col in 0..cols {
                        let Some(cell) = screen.cell(row, col) else {
                            continue;
                        };
                        let x = rect.min.x + col as f32 * char_w;
                        let y = rect.min.y + row as f32 * row_h;
                        let fg0 = vt_to_color(cell.fgcolor(), TERM_FG);
                        let bg_opt = match cell.bgcolor() {
                            vt100::Color::Default => None,
                            c => Some(vt_to_color(c, TERM_FG)),
                        };
                        let (fg, bg) = if cell.inverse() {
                            (bg_opt.unwrap_or(TERM_BG), Some(fg0))
                        } else {
                            (fg0, bg_opt)
                        };
                        let cell_rect =
                            egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(char_w, row_h));
                        if let Some(bg) = bg {
                            painter.rect_filled(cell_rect, 0.0, bg);
                        }
                        let contents = cell.contents();
                        if !contents.is_empty() && contents != " " {
                            painter.text(
                                egui::pos2(x, y),
                                egui::Align2::LEFT_TOP,
                                contents,
                                font_id.clone(),
                                fg,
                            );
                        }
                    }
                }
                if !screen.hide_cursor() {
                    let (cr, cc) = screen.cursor_position();
                    let x = rect.min.x + cc as f32 * char_w;
                    let y = rect.min.y + cr as f32 * row_h;
                    let alpha = if focused { 160 } else { 70 };
                    painter.rect_filled(
                        egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(char_w, row_h)),
                        0.0,
                        egui::Color32::from_rgba_unmultiplied(210, 210, 210, alpha),
                    );
                }
            }

            if focused {
                painter.rect_stroke(rect, 0.0, egui::Stroke::new(1.0, ACCENT_GOLD));
                let bytes = ui.input(|i| translate_terminal_input(&i.events));
                if !bytes.is_empty() {
                    t.send(&bytes);
                }
            } else {
                painter.text(
                    rect.left_bottom() + egui::vec2(6.0, -6.0),
                    egui::Align2::LEFT_BOTTOM,
                    "click to type",
                    egui::FontId::proportional(11.0),
                    egui::Color32::from_gray(120),
                );
            }
        }
        TerminalState::Failed(e) => {
            ui.centered_and_justified(|ui| {
                ui.label(egui::RichText::new(e.as_str()).weak());
            });
        }
    }
}
