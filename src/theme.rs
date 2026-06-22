//! UI theme presets (egui visuals) + the design-system foundation: the bundled
//! icon font, named icon glyphs, spacing/radius tokens, and a refined global
//! style. The 3D viewport is unaffected by themes.

use serde::{Deserialize, Serialize};

// Forged CoEngine accent colors (from the icon): cobalt identity, gold accents.
pub(crate) const ACCENT_GOLD: egui::Color32 = egui::Color32::from_rgb(217, 138, 43);
pub(crate) const ACCENT_COBALT: egui::Color32 = egui::Color32::from_rgb(56, 116, 210);

// --- Design tokens (spacing / radius scale) -------------------------------
/// Corner radius for buttons, fields, tabs.
pub(crate) const RADIUS: f32 = 6.0;
/// Corner radius for windows / popups / cards.
pub(crate) const RADIUS_LG: f32 = 9.0;

/// Bundled Tabler icon font + glyph constants. Use like `ui.label(icon::SEARCH)`.
/// The font is appended as a fallback to the proportional/monospace families
/// (`install_fonts`), so these chars render anywhere text does.
#[allow(dead_code)] // the full set lands as the redesign consumes them step by step
pub(crate) mod icon {
    pub(crate) const FILES: &str = "\u{edef}";
    pub(crate) const FILE: &str = "\u{eaa2}";
    pub(crate) const FILE_PLUS: &str = "\u{eaa0}";
    pub(crate) const FOLDER: &str = "\u{eaad}";
    pub(crate) const FOLDER_OPEN: &str = "\u{faf7}";
    pub(crate) const FOLDER_PLUS: &str = "\u{eaab}";
    pub(crate) const SEARCH: &str = "\u{eb1c}";
    pub(crate) const GIT_BRANCH: &str = "\u{eab2}";
    pub(crate) const BOX: &str = "\u{ea45}";
    pub(crate) const CUBE: &str = "\u{fa97}";
    pub(crate) const TERMINAL: &str = "\u{ebef}";
    pub(crate) const SETTINGS: &str = "\u{eb20}";
    pub(crate) const PLAY: &str = "\u{ed46}";
    pub(crate) const X: &str = "\u{eb55}";
    pub(crate) const CHEVRON_DOWN: &str = "\u{ea5f}";
    pub(crate) const CHEVRON_RIGHT: &str = "\u{ea61}";
    pub(crate) const CHEVRON_UP: &str = "\u{ea62}";
    pub(crate) const POINT: &str = "\u{eb0c}";
    pub(crate) const CIRCLE_DOT: &str = "\u{efb1}";
    pub(crate) const SAVE: &str = "\u{eb62}";
    pub(crate) const PLUS: &str = "\u{eb0b}";
    pub(crate) const TRASH: &str = "\u{eb41}";
    pub(crate) const PENCIL: &str = "\u{eb04}";
    pub(crate) const COPY: &str = "\u{ea7a}";
    pub(crate) const MENU: &str = "\u{ec42}";
    pub(crate) const HEXAGON: &str = "\u{ec02}";
    pub(crate) const EYE: &str = "\u{ea9a}";
    pub(crate) const VIDEO: &str = "\u{ed22}";
    pub(crate) const ARROW_LEFT: &str = "\u{ea19}";
    pub(crate) const ARROW_RIGHT: &str = "\u{ea1f}";
    pub(crate) const DOTS: &str = "\u{ea94}";
    pub(crate) const SIDEBAR: &str = "\u{eada}";
    pub(crate) const SCENE: &str = "\u{ecd7}";
    pub(crate) const INSPECTOR: &str = "\u{ea03}";
    pub(crate) const CHAT: &str = "\u{eaec}";
    pub(crate) const LOG: &str = "\u{eb6b}";
    pub(crate) const PHOTO: &str = "\u{eb0a}";
    pub(crate) const ROBOT: &str = "\u{f00b}";
    pub(crate) const GAUGE: &str = "\u{eab1}";
    pub(crate) const CPU: &str = "\u{ef8e}";
    pub(crate) const CHECK: &str = "\u{ea5e}";
    pub(crate) const LIBRARY: &str = "\u{fd4c}";
    pub(crate) const MUSIC: &str = "\u{eafc}";
    pub(crate) const BRACES: &str = "\u{ebcc}";
    pub(crate) const UPLOAD: &str = "\u{eb47}";
    pub(crate) const MODEL3D: &str = "\u{f032}";
    pub(crate) const GRID: &str = "\u{edba}";
    pub(crate) const LIST: &str = "\u{ef40}";
    pub(crate) const PAUSE: &str = "\u{ed45}";
    pub(crate) const STOP: &str = "\u{ed4a}";
    pub(crate) const VOLUME: &str = "\u{eb51}";
    pub(crate) const VOLUME_MUTE: &str = "\u{eb50}";
    pub(crate) const BACK10: &str = "\u{faba}";
    pub(crate) const FWD10: &str = "\u{fac2}";
    pub(crate) const SKIP_START: &str = "\u{ed48}";
    pub(crate) const SKIP_END: &str = "\u{ed49}";
}

/// Register the bundled icon font, appended as a fallback to the default
/// proportional and monospace families so icon glyphs render inline with text,
/// plus a dedicated `icons` family for explicit icon-only widgets. Call once at
/// startup, before the first frame.
pub(crate) fn install_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "tabler".to_owned(),
        egui::FontData::from_static(include_bytes!("../assets/tabler-icons.ttf")),
    );
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .push("tabler".to_owned());
    }
    fonts.families.insert(
        egui::FontFamily::Name("icons".into()),
        vec!["tabler".to_owned()],
    );
    ctx.set_fonts(fonts);
}

/// Apply the global style tokens (spacing, button padding, margins). Visuals are
/// set separately per-frame via `theme_visuals`; this only touches `spacing`, so
/// the two don't fight. Call once at startup.
pub(crate) fn apply_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.spacing.button_padding = egui::vec2(10.0, 5.0);
    style.spacing.menu_margin = egui::Margin::same(8.0);
    style.spacing.window_margin = egui::Margin::same(10.0);
    style.spacing.interact_size.y = 26.0;
    ctx.set_style(style);
}

/// Visual theme preset for the UI (the 3D viewport is unaffected by themes).
#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) enum Theme {
    DefaultSimple,
    Barbarian,
}

/// Pick the egui visuals for the chosen theme + light/dark mode.
pub(crate) fn theme_visuals(theme: Theme, dark: bool) -> egui::Visuals {
    match (theme, dark) {
        (Theme::DefaultSimple, true) => default_dark(),
        (Theme::DefaultSimple, false) => default_light(),
        (Theme::Barbarian, true) => barbarian_dark(),
        (Theme::Barbarian, false) => barbarian_light(),
    }
}

/// "Default Simple" dark: clean charcoal-iron base, cobalt identity, gold accents.
pub(crate) fn default_dark() -> egui::Visuals {
    use egui::{Color32, Rounding, Stroke};
    let round = Rounding::same(RADIUS);
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
    v.window_rounding = Rounding::same(RADIUS_LG);
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
pub(crate) fn default_light() -> egui::Visuals {
    use egui::{Color32, Rounding, Stroke};
    let round = Rounding::same(RADIUS);
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
pub(crate) fn barbarian_dark() -> egui::Visuals {
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
pub(crate) fn barbarian_light() -> egui::Visuals {
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
