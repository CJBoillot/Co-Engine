//! UI theme presets (egui visuals). The 3D viewport is unaffected by themes.

use serde::{Deserialize, Serialize};

// Forged CoEngine accent colors (from the icon): cobalt identity, gold accents.
pub(crate) const ACCENT_GOLD: egui::Color32 = egui::Color32::from_rgb(217, 138, 43);
pub(crate) const ACCENT_COBALT: egui::Color32 = egui::Color32::from_rgb(56, 116, 210);

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
pub(crate) fn default_light() -> egui::Visuals {
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
