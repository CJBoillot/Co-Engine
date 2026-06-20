//! Remappable input controls: the action set, the keymap, and key labels.

use serde::{Deserialize, Serialize};
use winit::keyboard::KeyCode;

/// One remappable input action.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ControlAction {
    Forward,
    Back,
    Left,
    Right,
    Up,
    Down,
    AddCube,
    Remove,
    ToggleDebug,
    ToggleMenu,
    ToggleCamera,
}

/// All actions + display labels, in the order they appear in Settings.
pub(crate) const CONTROL_ACTIONS: &[(ControlAction, &str)] = &[
    (ControlAction::Forward, "Move forward"),
    (ControlAction::Back, "Move back"),
    (ControlAction::Left, "Move left"),
    (ControlAction::Right, "Move right"),
    (ControlAction::Up, "Move up"),
    (ControlAction::Down, "Move down"),
    (ControlAction::AddCube, "Add cube"),
    (ControlAction::Remove, "Remove selected"),
    (ControlAction::ToggleDebug, "Toggle debug info"),
    (ControlAction::ToggleMenu, "Open / close menu"),
    (ControlAction::ToggleCamera, "Toggle camera mode"),
];

/// The current key bound to each action. Remappable in Settings -> Controls.
#[derive(Clone, Copy, Serialize, Deserialize)]
pub(crate) struct Controls {
    pub(crate) forward: KeyCode,
    pub(crate) back: KeyCode,
    pub(crate) left: KeyCode,
    pub(crate) right: KeyCode,
    pub(crate) up: KeyCode,
    pub(crate) down: KeyCode,
    pub(crate) add_cube: KeyCode,
    pub(crate) remove: KeyCode,
    pub(crate) toggle_debug: KeyCode,
    pub(crate) toggle_menu: KeyCode,
    pub(crate) toggle_camera: KeyCode,
}

impl Default for Controls {
    fn default() -> Self {
        Self {
            forward: KeyCode::KeyW,
            back: KeyCode::KeyS,
            left: KeyCode::KeyA,
            right: KeyCode::KeyD,
            up: KeyCode::KeyE,
            down: KeyCode::KeyQ,
            add_cube: KeyCode::KeyC,
            remove: KeyCode::Delete,
            toggle_debug: KeyCode::KeyH,
            toggle_menu: KeyCode::Escape,
            toggle_camera: KeyCode::Tab,
        }
    }
}

impl Controls {
    pub(crate) fn key(&self, a: ControlAction) -> KeyCode {
        match a {
            ControlAction::Forward => self.forward,
            ControlAction::Back => self.back,
            ControlAction::Left => self.left,
            ControlAction::Right => self.right,
            ControlAction::Up => self.up,
            ControlAction::Down => self.down,
            ControlAction::AddCube => self.add_cube,
            ControlAction::Remove => self.remove,
            ControlAction::ToggleDebug => self.toggle_debug,
            ControlAction::ToggleMenu => self.toggle_menu,
            ControlAction::ToggleCamera => self.toggle_camera,
        }
    }

    pub(crate) fn set(&mut self, a: ControlAction, k: KeyCode) {
        match a {
            ControlAction::Forward => self.forward = k,
            ControlAction::Back => self.back = k,
            ControlAction::Left => self.left = k,
            ControlAction::Right => self.right = k,
            ControlAction::Up => self.up = k,
            ControlAction::Down => self.down = k,
            ControlAction::AddCube => self.add_cube = k,
            ControlAction::Remove => self.remove = k,
            ControlAction::ToggleDebug => self.toggle_debug = k,
            ControlAction::ToggleMenu => self.toggle_menu = k,
            ControlAction::ToggleCamera => self.toggle_camera = k,
        }
    }
}

/// Human-friendly label for a key (KeyW -> "W", Digit1 -> "1", else Debug name).
pub(crate) fn key_label(code: KeyCode) -> String {
    let raw = format!("{code:?}");
    if let Some(rest) = raw.strip_prefix("Key") {
        rest.to_string()
    } else if let Some(rest) = raw.strip_prefix("Digit") {
        rest.to_string()
    } else {
        raw
    }
}
