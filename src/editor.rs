//! The file editor/viewer + Explorer tree: decoding files, syntax language
//! detection, and the egui rendering for file tabs and the project tree.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::theme::ACCENT_GOLD;

/// Cached, decoded contents of a file shown in a `DockTab::File` viewer.
pub(crate) enum FileView {
    /// Editable text/code; `dirty` = edited since last load/save.
    Text { buf: String, dirty: bool },
    Image(egui::TextureHandle),
    Binary,
    Error(String),
}

impl FileView {
    /// True if this is text with unsaved edits.
    pub(crate) fn is_dirty(&self) -> bool {
        matches!(self, FileView::Text { dirty: true, .. })
    }
}

/// Write a cached text file back to disk and clear its dirty flag. No-op for
/// non-text views. Returns any I/O error.
pub(crate) fn save_file(path: &Path, cache: &mut HashMap<PathBuf, FileView>) -> std::io::Result<()> {
    if let Some(FileView::Text { buf, dirty }) = cache.get_mut(path) {
        std::fs::write(path, buf.as_bytes())?;
        *dirty = false;
    }
    Ok(())
}

/// Is this path an image we can decode (by extension)?
pub(crate) fn is_image_path(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .as_deref(),
        Some("png" | "jpg" | "jpeg" | "gif" | "bmp" | "webp")
    )
}

/// Load + decode a file for the viewer: an image becomes a texture, a UTF-8 file
/// becomes text, anything else is flagged binary. Errors are surfaced, not fatal.
pub(crate) fn load_file_view(ctx: &egui::Context, path: &Path) -> FileView {
    if is_image_path(path) {
        return match std::fs::read(path) {
            Ok(bytes) => match image::load_from_memory(&bytes) {
                Ok(img) => {
                    let rgba = img.to_rgba8();
                    let (w, h) = rgba.dimensions();
                    let ci = egui::ColorImage::from_rgba_unmultiplied(
                        [w as usize, h as usize],
                        &rgba,
                    );
                    let tex = ctx.load_texture(
                        format!("file:{}", path.display()),
                        ci,
                        egui::TextureOptions::LINEAR,
                    );
                    FileView::Image(tex)
                }
                Err(e) => FileView::Error(format!("Couldn't decode image: {e}")),
            },
            Err(e) => FileView::Error(format!("Couldn't read file: {e}")),
        };
    }
    match std::fs::read(path) {
        Ok(bytes) => match String::from_utf8(bytes) {
            Ok(s) => FileView::Text {
                buf: s,
                dirty: false,
            },
            Err(_) => FileView::Binary,
        },
        Err(e) => FileView::Error(format!("Couldn't read file: {e}")),
    }
}

/// Map a file path to a syntax-highlighting language token (egui_extras).
pub(crate) fn language_for(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("rs") => "rs",
        Some("toml") => "toml",
        Some("json") => "json",
        Some("md" | "markdown") => "md",
        Some("py") => "py",
        Some("js" | "mjs") => "js",
        Some("ts") => "ts",
        Some("html" | "htm") => "html",
        Some("css") => "css",
        Some("c" | "h") => "c",
        Some("cpp" | "cc" | "cxx" | "hpp") => "cpp",
        Some("sh" | "bash") => "sh",
        Some("yaml" | "yml") => "yaml",
        _ => "txt",
    }
}

/// State for the in-file Find/Replace bar (one active at a time, for the focused
/// file). Char-index ranges; `current` indexes into the live match list.
#[derive(Default)]
pub(crate) struct FindState {
    pub(crate) open: bool,
    pub(crate) query: String,
    pub(crate) replace: String,
    pub(crate) current: usize,
    /// Request keyboard focus into the query field (set when the bar opens).
    pub(crate) focus_query: bool,
}

/// All (case-insensitive, ASCII-folded) match ranges of `needle` in `hay`, as
/// non-overlapping char-index pairs `(start, end)`.
pub(crate) fn find_matches(hay: &str, needle: &str) -> Vec<(usize, usize)> {
    if needle.is_empty() {
        return Vec::new();
    }
    let h: Vec<char> = hay.chars().map(|c| c.to_ascii_lowercase()).collect();
    let n: Vec<char> = needle.chars().map(|c| c.to_ascii_lowercase()).collect();
    let mut out = Vec::new();
    if n.len() > h.len() {
        return out;
    }
    let mut i = 0;
    while i + n.len() <= h.len() {
        if h[i..i + n.len()] == n[..] {
            out.push((i, i + n.len()));
            i += n.len();
        } else {
            i += 1;
        }
    }
    out
}

/// All (case-insensitive, ASCII-folded) match ranges of `needle` in `hay`, as
/// non-overlapping BYTE ranges. ASCII lowercasing preserves byte length, so the
/// offsets line up with the original `hay` — used to highlight matches in the
/// editor's layout job.
pub(crate) fn find_match_bytes(hay: &str, needle: &str) -> Vec<(usize, usize)> {
    if needle.is_empty() {
        return Vec::new();
    }
    let hay_l = hay.to_ascii_lowercase();
    let needle_l = needle.to_ascii_lowercase();
    let mut out = Vec::new();
    let mut start = 0;
    while let Some(pos) = hay_l[start..].find(&needle_l) {
        let s = start + pos;
        let e = s + needle_l.len();
        out.push((s, e));
        start = e;
    }
    out
}

/// Overlay a background color on the given BYTE ranges of a layout job, splitting
/// existing (syntax-colored) sections at match boundaries so highlights compose
/// with the syntax coloring.
fn add_match_highlight(job: &mut egui::text::LayoutJob, matches: &[(usize, usize)], bg: egui::Color32) {
    if matches.is_empty() {
        return;
    }
    let mut sections = Vec::new();
    for sec in job.sections.drain(..) {
        let (start, end) = (sec.byte_range.start, sec.byte_range.end);
        // Cut points: section bounds + any match edges that fall inside it.
        let mut points = vec![start, end];
        for &(s, e) in matches {
            if e > start && s < end {
                points.push(s.clamp(start, end));
                points.push(e.clamp(start, end));
            }
        }
        points.sort_unstable();
        points.dedup();
        for w in points.windows(2) {
            let (a, b) = (w[0], w[1]);
            if a == b {
                continue;
            }
            let mut format = sec.format.clone();
            if matches.iter().any(|&(s, e)| s <= a && b <= e) {
                format.background = bg;
            }
            sections.push(egui::text::LayoutSection {
                leading_space: if a == start { sec.leading_space } else { 0.0 },
                byte_range: a..b,
                format,
            });
        }
    }
    job.sections = sections;
}

/// Byte offset of char index `idx` in `s` (== `s.len()` when at/after the end).
fn byte_of_char(s: &str, idx: usize) -> usize {
    s.char_indices().nth(idx).map(|(b, _)| b).unwrap_or(s.len())
}

/// Replace the char range `[start, end)` of `buf` with `with`, in place.
pub(crate) fn replace_char_range(buf: &mut String, start: usize, end: usize, with: &str) {
    let b0 = byte_of_char(buf, start);
    let b1 = byte_of_char(buf, end);
    buf.replace_range(b0..b1, with);
}

/// Convert a char index into 1-based (line, column) for the status bar.
pub(crate) fn line_col(text: &str, char_idx: usize) -> (usize, usize) {
    let mut line = 1usize;
    let mut col = 1usize;
    for ch in text.chars().take(char_idx) {
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

/// Render a loaded file: editable, syntax-highlighted code/text; an image scaled
/// to fit; or a placeholder for binary/error. When the text editor has focus, its
/// caret position is reported (1-based line/col) in `cursor_out` for the status bar.
pub(crate) fn file_view_ui(
    ui: &mut egui::Ui,
    view: &mut FileView,
    lang: &str,
    cursor_out: &mut Option<(usize, usize)>,
    goto: &mut Option<(usize, usize)>,
    find_query: &str,
) {
    match view {
        FileView::Text { buf, dirty } => {
            let theme = egui_extras::syntax_highlighting::CodeTheme::from_style(ui.style());
            let mut layouter = |ui: &egui::Ui, text: &str, wrap_width: f32| {
                let mut job = egui_extras::syntax_highlighting::highlight(
                    ui.ctx(),
                    ui.style(),
                    &theme,
                    text,
                    lang,
                );
                if !find_query.is_empty() {
                    let matches = find_match_bytes(text, find_query);
                    add_match_highlight(&mut job, &matches, egui::Color32::from_rgb(110, 90, 20));
                }
                job.wrap.max_width = wrap_width;
                ui.fonts(|f| f.layout_job(job))
            };
            egui::ScrollArea::both()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    // Editable, highlighted code/text; mark dirty on edit (saved via the toolbar).
                    let output = egui::TextEdit::multiline(buf)
                        .code_editor()
                        .desired_width(f32::INFINITY)
                        .layouter(&mut layouter)
                        .show(ui);
                    if output.response.changed() {
                        *dirty = true;
                    }
                    if output.response.has_focus() {
                        if let Some(range) = output.cursor_range {
                            *cursor_out = Some(line_col(buf, range.primary.ccursor.index));
                        }
                    }
                    // A Find match was requested: select it and scroll it into view.
                    if let Some((s, e)) = goto.take() {
                        let mut state = output.state.clone();
                        let range = egui::text::CCursorRange::two(
                            egui::text::CCursor::new(s),
                            egui::text::CCursor::new(e),
                        );
                        state.cursor.set_char_range(Some(range));
                        state.store(ui.ctx(), output.response.id);
                        let cursor = output.galley.from_ccursor(egui::text::CCursor::new(s));
                        let rect = output
                            .galley
                            .pos_from_cursor(&cursor)
                            .translate(output.galley_pos.to_vec2());
                        ui.scroll_to_rect(rect, Some(egui::Align::Center));
                        ui.ctx().request_repaint();
                    }
                });
        }
        FileView::Image(tex) => {
            egui::ScrollArea::both()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    let avail = ui.available_width();
                    ui.add(
                        egui::Image::new(egui::load::SizedTexture::new(tex.id(), tex.size_vec2()))
                            .max_width(avail),
                    );
                });
        }
        FileView::Binary => {
            ui.centered_and_justified(|ui| {
                ui.label(egui::RichText::new("Binary file — can't display as text").weak());
            });
        }
        FileView::Error(e) => {
            ui.centered_and_justified(|ui| {
                ui.label(egui::RichText::new(e.as_str()).weak());
            });
        }
    }
}

/// A pending Explorer file operation. Raised by the tree's right-click menu and
/// shown as a modal that collects a name (or a delete confirmation) before the
/// op runs. `name` buffers the user's text as they type in the modal.
pub(crate) enum FsPrompt {
    NewFile { parent: PathBuf, name: String },
    NewFolder { parent: PathBuf, name: String },
    Rename { target: PathBuf, name: String },
    Delete { target: PathBuf },
}

/// What a completed file op did, so the caller can sync open tabs / the cache.
pub(crate) enum FsOutcome {
    /// A file was created (the caller may open it in a viewer tab).
    CreatedFile(PathBuf),
    /// A folder was created (nothing to open).
    CreatedFolder,
    /// `from` was renamed/moved to `to`.
    Renamed { from: PathBuf, to: PathBuf },
    /// `target` (file or folder) was removed.
    Deleted(PathBuf),
}

/// Reject empty names and anything with path separators or parent refs — the
/// Explorer only creates/renames within the chosen folder, never across paths.
fn validate_name(name: &str) -> Result<&str, String> {
    let n = name.trim();
    if n.is_empty() {
        return Err("Name can't be empty".into());
    }
    if n.contains('/') || n.contains('\\') || n == "." || n == ".." {
        return Err("Name can't contain path separators".into());
    }
    Ok(n)
}

/// Execute a confirmed file op on disk. Errors are returned (surfaced in the
/// modal), never panicking — the tree re-reads disk each frame, so success
/// shows up automatically.
pub(crate) fn run_fs_prompt(prompt: &FsPrompt) -> Result<FsOutcome, String> {
    let map_io = |e: std::io::Error| e.to_string();
    match prompt {
        FsPrompt::NewFile { parent, name } => {
            let p = parent.join(validate_name(name)?);
            if p.exists() {
                return Err("A file or folder with that name already exists".into());
            }
            std::fs::File::create(&p).map_err(map_io)?;
            Ok(FsOutcome::CreatedFile(p))
        }
        FsPrompt::NewFolder { parent, name } => {
            let p = parent.join(validate_name(name)?);
            if p.exists() {
                return Err("A file or folder with that name already exists".into());
            }
            std::fs::create_dir(&p).map_err(map_io)?;
            Ok(FsOutcome::CreatedFolder)
        }
        FsPrompt::Rename { target, name } => {
            let valid = validate_name(name)?;
            let to = target
                .parent()
                .map(|p| p.join(valid))
                .ok_or_else(|| "Can't rename this item".to_string())?;
            if to == *target {
                return Err("That's already the name".into());
            }
            if to.exists() {
                return Err("A file or folder with that name already exists".into());
            }
            std::fs::rename(target, &to).map_err(map_io)?;
            Ok(FsOutcome::Renamed {
                from: target.clone(),
                to,
            })
        }
        FsPrompt::Delete { target } => {
            if target.is_dir() {
                std::fs::remove_dir_all(target).map_err(map_io)?;
            } else {
                std::fs::remove_file(target).map_err(map_io)?;
            }
            Ok(FsOutcome::Deleted(target.clone()))
        }
    }
}

/// Move `src` (file or folder) into directory `dest_dir`, keeping its name.
/// `Ok(None)` means "nothing to do" (dropped back into its own folder); `Ok(Some)`
/// returns `(from, to)`; `Err` is a real conflict (name taken, into-itself).
pub(crate) fn move_into(src: &Path, dest_dir: &Path) -> Result<Option<(PathBuf, PathBuf)>, String> {
    let name = src
        .file_name()
        .ok_or_else(|| "Can't move this item".to_string())?;
    // No-op: already directly inside the destination.
    if src.parent() == Some(dest_dir) {
        return Ok(None);
    }
    // Can't drop a folder onto itself or into its own subtree.
    if dest_dir == src || dest_dir.starts_with(src) {
        return Err("Can't move a folder into itself".into());
    }
    let to = dest_dir.join(name);
    if to.exists() {
        return Err("The destination already has an item with that name".into());
    }
    std::fs::rename(src, &to).map_err(|e| e.to_string())?;
    Ok(Some((src.to_path_buf(), to)))
}

/// Context-menu items for a folder (or the project root): create inside it, and
/// — unless it's the root — rename/delete it. Sets `fs_req` on click.
fn folder_menu(ui: &mut egui::Ui, dir: &Path, is_root: bool, fs_req: &mut Option<FsPrompt>) {
    if ui.button("New File").clicked() {
        *fs_req = Some(FsPrompt::NewFile {
            parent: dir.to_path_buf(),
            name: String::new(),
        });
        ui.close_menu();
    }
    if ui.button("New Folder").clicked() {
        *fs_req = Some(FsPrompt::NewFolder {
            parent: dir.to_path_buf(),
            name: String::new(),
        });
        ui.close_menu();
    }
    if !is_root {
        ui.separator();
        if ui.button("Rename").clicked() {
            *fs_req = Some(FsPrompt::Rename {
                target: dir.to_path_buf(),
                name: file_name_string(dir),
            });
            ui.close_menu();
        }
        if ui.button("Delete").clicked() {
            *fs_req = Some(FsPrompt::Delete {
                target: dir.to_path_buf(),
            });
            ui.close_menu();
        }
    }
}

/// Context-menu items for a file: create a sibling, rename, or delete it.
fn file_menu(ui: &mut egui::Ui, path: &Path, fs_req: &mut Option<FsPrompt>) {
    if let Some(parent) = path.parent() {
        if ui.button("New File").clicked() {
            *fs_req = Some(FsPrompt::NewFile {
                parent: parent.to_path_buf(),
                name: String::new(),
            });
            ui.close_menu();
        }
        if ui.button("New Folder").clicked() {
            *fs_req = Some(FsPrompt::NewFolder {
                parent: parent.to_path_buf(),
                name: String::new(),
            });
            ui.close_menu();
        }
        ui.separator();
    }
    if ui.button("Rename").clicked() {
        *fs_req = Some(FsPrompt::Rename {
            target: path.to_path_buf(),
            name: file_name_string(path),
        });
        ui.close_menu();
    }
    if ui.button("Delete").clicked() {
        *fs_req = Some(FsPrompt::Delete {
            target: path.to_path_buf(),
        });
        ui.close_menu();
    }
}

/// The final path component as an owned string (empty if none).
fn file_name_string(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// One project-search result line.
pub(crate) struct SearchHit {
    pub(crate) path: PathBuf,
    pub(crate) line: usize,
    pub(crate) text: String,
}

/// State for the project-wide Search tab.
#[derive(Default)]
pub(crate) struct SearchUi {
    pub(crate) query: String,
    pub(crate) results: Vec<SearchHit>,
    /// Whether a search has been run (to distinguish "no results" from "idle").
    pub(crate) ran: bool,
}

/// Walk every text file under `root` (skipping `.git`, `target`, `node_modules`,
/// and files >2 MB / non-UTF-8), returning lines that contain `query`
/// (case-insensitive). Capped at 1000 hits.
pub(crate) fn project_search(root: &Path, query: &str) -> Vec<SearchHit> {
    let mut hits = Vec::new();
    if query.is_empty() {
        return hits;
    }
    const MAX_HITS: usize = 1000;
    let needle = query.to_ascii_lowercase();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let rd = match std::fs::read_dir(&dir) {
            Ok(r) => r,
            Err(_) => continue,
        };
        for e in rd.flatten() {
            let path = e.path();
            if path.is_dir() {
                let name = e.file_name();
                if matches!(name.to_string_lossy().as_ref(), ".git" | "target" | "node_modules") {
                    continue;
                }
                stack.push(path);
            } else {
                if e.metadata().map(|m| m.len() > 2_000_000).unwrap_or(true) {
                    continue;
                }
                let Ok(content) = std::fs::read_to_string(&path) else {
                    continue;
                };
                for (i, line) in content.lines().enumerate() {
                    if line.to_ascii_lowercase().contains(&needle) {
                        hits.push(SearchHit {
                            path: path.clone(),
                            line: i + 1,
                            text: line.trim().chars().take(200).collect(),
                        });
                        if hits.len() >= MAX_HITS {
                            return hits;
                        }
                    }
                }
            }
        }
    }
    hits
}

/// Every file path under `root` (skipping `.git`, `target`, `node_modules`),
/// capped at 4000, for the command palette's quick-open.
pub(crate) fn list_project_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    const MAX: usize = 4000;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.flatten() {
            let path = e.path();
            if path.is_dir() {
                let name = e.file_name();
                if matches!(name.to_string_lossy().as_ref(), ".git" | "target" | "node_modules") {
                    continue;
                }
                stack.push(path);
            } else {
                out.push(path);
                if out.len() >= MAX {
                    return out;
                }
            }
        }
    }
    out
}

/// The project-wide Search dock tab: a query box + a clickable list of results.
/// Clicking a result records the file in `open` (the caller opens a viewer tab).
pub(crate) fn search_tab(
    ui: &mut egui::Ui,
    search: &mut SearchUi,
    root: Option<&Path>,
    open: &mut Option<PathBuf>,
) {
    egui::TopBottomPanel::top("search_header").show_inside(ui, |ui| {
        ui.add_space(6.0);
        ui.heading("Search");
        ui.horizontal(|ui| {
            let field = ui.add(
                egui::TextEdit::singleline(&mut search.query)
                    .desired_width(f32::INFINITY)
                    .hint_text("Search in project…"),
            );
            let submit =
                field.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            if (ui.button("Go").clicked() || submit) && root.is_some() {
                search.results = project_search(root.unwrap(), &search.query);
                search.ran = true;
            }
        });
        ui.add_space(4.0);
    });
    egui::CentralPanel::default().show_inside(ui, |ui| {
        if root.is_none() {
            ui.centered_and_justified(|ui| {
                ui.label(egui::RichText::new("No project open").weak());
            });
            return;
        }
        if search.ran && search.results.is_empty() {
            ui.label(egui::RichText::new("No results").weak());
            return;
        }
        if !search.results.is_empty() {
            ui.label(
                egui::RichText::new(format!("{} results", search.results.len())).weak(),
            );
        }
        let base = root.unwrap();
        egui::ScrollArea::both()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for hit in &search.results {
                    let rel = hit.path.strip_prefix(base).unwrap_or(&hit.path);
                    let label = format!("{}:{}", rel.display(), hit.line);
                    let resp = ui.add(
                        egui::Label::new(egui::RichText::new(label).color(ACCENT_GOLD).small())
                            .sense(egui::Sense::click()),
                    );
                    ui.add(
                        egui::Label::new(egui::RichText::new(&hit.text).monospace().small())
                            .truncate(),
                    );
                    if resp.clicked() {
                        *open = Some(hit.path.clone());
                    }
                    ui.add_space(2.0);
                }
            });
    });
}

/// Explorer: an IDE-style file tree rooted at the project folder. The project
/// folder is the top of the hierarchy — nothing above it is reachable. Clicking
/// a file records it in `open` (the caller opens a viewer tab for it); right-click
/// raises a file op in `fs_req` (the caller runs it via a modal).
pub(crate) fn file_tree_ui(
    ui: &mut egui::Ui,
    root: &Path,
    open: &mut Option<PathBuf>,
    fs_req: &mut Option<FsPrompt>,
    move_req: &mut Option<(PathBuf, PathBuf)>,
) {
    egui::ScrollArea::both()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let name = root
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| root.display().to_string());
            let header = ui.add(
                egui::Label::new(egui::RichText::new(name).strong().color(ACCENT_GOLD))
                    .sense(egui::Sense::click()),
            );
            // The root header is both a right-click menu and a drop target (move
            // a dragged item to the top level).
            drop_target(ui, &header, root, move_req);
            header.context_menu(|ui| folder_menu(ui, root, true, fs_req));
            ui.separator();
            dir_contents_ui(ui, root, open, fs_req, move_req);

            // Right-clicking the blank area below the tree targets the project
            // root, so you can create at the top level without finding the header;
            // dropping there moves the item to the root too.
            let avail = ui.available_size();
            if avail.y > 1.0 {
                let (_rect, resp) =
                    ui.allocate_exact_size(avail, egui::Sense::click_and_drag());
                drop_target(ui, &resp, root, move_req);
                resp.context_menu(|ui| folder_menu(ui, root, true, fs_req));
            }
        });
}

/// Mark `resp` as a drop target for a dragged tree item: highlight while a drag
/// hovers it, and on release request a move into `dest_dir`.
fn drop_target(
    ui: &egui::Ui,
    resp: &egui::Response,
    dest_dir: &Path,
    move_req: &mut Option<(PathBuf, PathBuf)>,
) {
    if resp.dnd_hover_payload::<PathBuf>().is_some() {
        ui.painter()
            .rect_stroke(resp.rect, 2.0, egui::Stroke::new(1.0, ACCENT_GOLD));
    }
    if let Some(src) = resp.dnd_release_payload::<PathBuf>() {
        *move_req = Some((src.as_ref().clone(), dest_dir.to_path_buf()));
    }
}

/// Recursively render one directory's children (folders first, then files,
/// case-insensitive). Folders are collapsing headers (and drop targets); files
/// are clickable and draggable. Every node has a right-click context menu.
pub(crate) fn dir_contents_ui(
    ui: &mut egui::Ui,
    dir: &Path,
    open: &mut Option<PathBuf>,
    fs_req: &mut Option<FsPrompt>,
    move_req: &mut Option<(PathBuf, PathBuf)>,
) {
    let mut entries: Vec<std::fs::DirEntry> = match std::fs::read_dir(dir) {
        Ok(rd) => rd.flatten().collect(),
        Err(_) => return,
    };
    entries.sort_by_key(|e| {
        let is_dir = e.path().is_dir();
        (!is_dir, e.file_name().to_string_lossy().to_lowercase())
    });
    for e in entries {
        let path = e.path();
        let name = e.file_name().to_string_lossy().into_owned();
        if path.is_dir() {
            let header = egui::CollapsingHeader::new(name)
                .id_salt(&path)
                .default_open(false)
                .show(ui, |ui| dir_contents_ui(ui, &path, open, fs_req, move_req));
            // The header is a drag source (move the folder), a drop target (move
            // items into it), and a right-click menu. Drag sense is added on top of
            // the header so a plain click still toggles expand/collapse.
            let hr = header
                .header_response
                .interact(egui::Sense::click_and_drag());
            hr.dnd_set_drag_payload(path.clone());
            drop_target(ui, &hr, &path, move_req);
            hr.context_menu(|ui| folder_menu(ui, &path, false, fs_req));
        } else {
            // A normal selectable label (senses clicks → open) with drag sense
            // added on top, so a quick click opens and a press-drag moves the file.
            let resp = ui
                .selectable_label(false, name)
                .interact(egui::Sense::click_and_drag());
            if resp.clicked() {
                *open = Some(path.clone());
            }
            resp.dnd_set_drag_payload(path.clone());
            resp.context_menu(|ui| file_menu(ui, &path, fs_req));
        }
    }
}
