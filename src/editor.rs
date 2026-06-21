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

/// Render a loaded file: editable, syntax-highlighted code/text; an image scaled
/// to fit; or a placeholder for binary/error.
pub(crate) fn file_view_ui(ui: &mut egui::Ui, view: &mut FileView, lang: &str) {
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
                job.wrap.max_width = wrap_width;
                ui.fonts(|f| f.layout_job(job))
            };
            egui::ScrollArea::both()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    // Editable, highlighted code/text; mark dirty on edit (saved via the toolbar).
                    let resp = ui.add(
                        egui::TextEdit::multiline(buf)
                            .code_editor()
                            .desired_width(f32::INFINITY)
                            .layouter(&mut layouter),
                    );
                    if resp.changed() {
                        *dirty = true;
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
            let hr = header.header_response;
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
