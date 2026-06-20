//! Project persistence: the on-disk `project.json` manifest and the global
//! cross-project config store.

use std::path::PathBuf;

use egui_dock::DockState;
use serde::{Deserialize, Serialize};

use crate::DockTab;
use crate::controls::Controls;
use crate::mesh::Cube;
use crate::terminal::Shell;
use crate::theme::Theme;

/// On-disk project manifest (`project.json`). One file per project folder; holds
/// the whole reproducible state of a project: the scene plus the settings that
/// belong to *this* project (see memory: v0.0.12 settings model). The same shape
/// also backs the global "last settings" store — fields a fresh project seeds
/// from are exactly the non-scene ones.
///
/// `format_version` lets later versions migrate old files instead of failing to
/// load. Window geometry is stored as plain numbers (not winit types) so the
/// JSON stays human-readable and decoupled from the windowing crate.
#[derive(Serialize, Deserialize)]
pub(crate) struct Project {
    /// Manifest schema version; bump when the format changes incompatibly.
    pub(crate) format_version: u32,
    /// The CoEngine (CoSemVer) build that last wrote this file — for diagnostics.
    pub(crate) engine_version: String,
    /// The 3D scene: every placed cube.
    pub(crate) scene: Vec<Cube>,
    /// UI theme preset.
    pub(crate) theme: Theme,
    /// Dark vs. light mode for the chosen theme.
    pub(crate) dark_mode: bool,
    /// Remappable control bindings (keymap).
    pub(crate) controls: Controls,
    /// Terminal shell choice. `default` keeps pre-v0.0.14 manifests loadable.
    #[serde(default)]
    pub(crate) shell: Shell,
    /// Dockable workspace layout.
    pub(crate) dock_state: DockState<DockTab>,
    /// Last window inner size in physical pixels, if known.
    pub(crate) window_size: Option<(u32, u32)>,
    /// Last window position (top-left, physical pixels), if known.
    pub(crate) window_pos: Option<(i32, i32)>,
}

/// Current `project.json` schema version.
pub(crate) const PROJECT_FORMAT_VERSION: u32 = 1;

/// Global, cross-project config (the "last settings" store, see v0.0.12 settings
/// model): remembers the most-recently-used project and the settings new/empty
/// sessions seed from. Lives at `%APPDATA%\CoEngine\config.json`.
#[derive(Serialize, Deserialize, Default)]
pub(crate) struct GlobalConfig {
    /// Folder of the most recently saved/opened project (for the startup popup).
    pub(crate) last_project: Option<PathBuf>,
    /// Last-used UI theme (seeds new/empty sessions).
    pub(crate) theme: Option<Theme>,
    /// Last-used dark/light mode.
    pub(crate) dark_mode: Option<bool>,
    /// Last-used control bindings.
    pub(crate) controls: Option<Controls>,
    /// Last-used terminal shell.
    pub(crate) shell: Option<Shell>,
}

/// Path to the global config file, creating `%APPDATA%\CoEngine\` if needed.
pub(crate) fn global_config_path() -> Option<PathBuf> {
    let appdata = std::env::var_os("APPDATA")?;
    let dir = PathBuf::from(appdata).join("CoEngine");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir.join("config.json"))
}

/// Load the global config (defaults if missing or unreadable).
pub(crate) fn load_global_config() -> GlobalConfig {
    global_config_path()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

/// Write the global config, ignoring errors (it's a convenience store).
pub(crate) fn save_global_config(cfg: &GlobalConfig) {
    if let Some(p) = global_config_path() {
        if let Ok(json) = serde_json::to_string_pretty(cfg) {
            let _ = std::fs::write(p, json);
        }
    }
}
