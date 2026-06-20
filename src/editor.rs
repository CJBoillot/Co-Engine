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

/// Explorer: an IDE-style file tree rooted at the project folder. The project
/// folder is the top of the hierarchy — nothing above it is reachable. Clicking
/// a file records it in `open` (the caller opens a viewer tab for it).
pub(crate) fn file_tree_ui(ui: &mut egui::Ui, root: &Path, open: &mut Option<PathBuf>) {
    egui::ScrollArea::both()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let name = root
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| root.display().to_string());
            ui.label(egui::RichText::new(name).strong().color(ACCENT_GOLD));
            ui.separator();
            dir_contents_ui(ui, root, open);
        });
}

/// Recursively render one directory's children (folders first, then files,
/// case-insensitive). Folders are collapsing headers; files are clickable.
pub(crate) fn dir_contents_ui(ui: &mut egui::Ui, dir: &Path, open: &mut Option<PathBuf>) {
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
            egui::CollapsingHeader::new(name)
                .id_salt(&path)
                .default_open(false)
                .show(ui, |ui| dir_contents_ui(ui, &path, open));
        } else if ui.selectable_label(false, name).clicked() {
            *open = Some(path);
        }
    }
}
