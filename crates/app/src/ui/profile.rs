//! Profiles (user request 2026-10-19): the viewer's own look and layout — where the dock tabs are,
//! which dock is open on which tab and how wide, the view and density-look settings, the units —
//! as a small text file of `key = value` lines. Display state only: nothing here is an `Action`
//! or reaches the worker, so a profile never changes the physics or a recording.
//!
//! The window keeps the state in force in `%LOCALAPPDATA%\atomic-sandbox\current.profile`
//! (written a moment after each change, read at launch: the app opens as it was left); named
//! profiles are saved to and loaded from the `profiles` folder (`scene::profiles_dir`) so they can
//! be shared; Reset puts back the built-in defaults. File ▸ Profiles… is the window.
use super::{theme, DockTab, EnergyUnits, TempUnits, UiState};
use crate::names::NameStyle;
use egui::{Color32, RichText, Stroke, Vec2};
use std::path::PathBuf;
use std::time::{Duration, Instant};

macro_rules! fields {
    ($m:ident) => {
        $m!(show_partial_core, show_bonds, show_labels, show_badge, show_ghost, show_angles, show_bond_lengths, follow_selected,
            shimmer, dots_by_orbital, orbital_phase, colour_additive, colour_physical, show_cell, show_floor, show_axes,
            show_place_grid, grid_in_angstrom, show_trails, hide_cloud, camera_view, dots_float, dots_glow, smooth_motion, dock_open, left_open, show_atom_card, graph_all;
            gamma, opacity, iso, phase_iso, spin_iso, trail_opacity, trail_fs, volume_mode, motion_blur, shutter, exposure_ev, camera_solidity, camera_size, fov_deg, label_opacity, dots_drift, dots_amount, dots_size, dots_brightness)
    };
}

fn tab_name(t: DockTab) -> String {
    format!("{t:?}")
}
fn tab_of(w: &str) -> Option<DockTab> {
    DockTab::ALL.iter().copied().find(|t| tab_name(*t) == w)
}
fn tabs_of(v: &str) -> Option<Vec<DockTab>> {
    v.split_whitespace().map(tab_of).collect()
}
fn pick<T: Copy + std::fmt::Debug>(all: &[T], w: &str) -> Option<T> {
    all.iter().copied().find(|t| format!("{t:?}") == w)
}
const NAME_STYLES: [NameStyle; 4] = [NameStyle::Common, NameStyle::Stoichiometric, NameStyle::Hydride, NameStyle::FormulaOnly];
const ENERGY_UNITS: [EnergyUnits; 3] = [EnergyUnits::Ha, EnergyUnits::Ev, EnergyUnits::Both];
const TEMP_UNITS: [TempUnits; 3] = [TempUnits::K, TempUnits::C, TempUnits::F];

/// The profile of `st`, as text.
pub fn capture(st: &UiState) -> String {
    let mut out = String::from("# Atomic Sandbox profile: the viewer's layout and look (display only)\n");
    macro_rules! put {
        ($($b:ident),*; $($n:ident),*) => {
            $(out += &format!("{} = {}\n", stringify!($b), st.$b);)*
            $(out += &format!("{} = {}\n", stringify!($n), st.$n);)*
        };
    }
    fields!(put);
    out += &format!("nucleus_scale = {}\n", st.nucleus_scale.map_or("auto".into(), |x| x.to_string()));
    out += &format!("background = {}\n", st.background.map_or("default".into(), |c| format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])));
    out += &format!("name_style = {:?}\n", st.name_style);
    out += &format!("energy_units = {:?}\n", st.energy_units);
    out += &format!("temp_units = {:?}\n", st.temp_units);
    out += &format!("left_tab = {}\nright_tab = {}\n", tab_name(st.left_tab), tab_name(st.dock_tab));
    let names = |v: &[DockTab]| v.iter().map(|t| tab_name(*t)).collect::<Vec<_>>().join(" ");
    out += &format!("left_tabs = {}\nright_tabs = {}\n", names(&st.left_tabs), names(&st.right_tabs));
    out += &format!("left_width = {:.0}\nright_width = {:.0}\n", st.dock_w[0], st.dock_w[1]);
    out
}

/// Applies a profile's text to `st`: unknown keys and unreadable values are skipped, a tab
/// layout that does not name every tab exactly once is ignored.
pub fn apply(st: &mut UiState, text: &str) {
    let kv: std::collections::HashMap<&str, &str> = text
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.trim(), v.trim()))
        .collect();
    macro_rules! get {
        ($($b:ident),*; $($n:ident),*) => {
            $(if let Some(v) = kv.get(stringify!($b)).and_then(|v| v.parse().ok()) { st.$b = v; })*
            $(if let Some(v) = kv.get(stringify!($n)).and_then(|v| v.parse().ok()) { st.$n = v; })*
        };
    }
    fields!(get);
    if let Some(v) = kv.get("nucleus_scale") {
        st.nucleus_scale = v.parse().ok();
    }
    if let Some(v) = kv.get("background") {
        st.background = v.strip_prefix('#').filter(|h| h.len() == 6).and_then(|h| {
            let b = |i: usize| u8::from_str_radix(&h[i..i + 2], 16).ok();
            Some([b(0)?, b(2)?, b(4)?])
        });
    }
    if let Some(v) = kv.get("name_style").and_then(|w| pick(&NAME_STYLES, w)) {
        st.name_style = v;
    }
    if let Some(v) = kv.get("energy_units").and_then(|w| pick(&ENERGY_UNITS, w)) {
        st.energy_units = v;
    }
    if let Some(v) = kv.get("temp_units").and_then(|w| pick(&TEMP_UNITS, w)) {
        st.temp_units = v;
    }
    if let (Some(l), Some(r)) = (kv.get("left_tabs").and_then(|v| tabs_of(v)), kv.get("right_tabs").and_then(|v| tabs_of(v))) {
        let mut all: Vec<u8> = l.iter().chain(&r).map(|t| *t as u8).collect();
        all.sort_unstable();
        all.dedup();
        if all.len() == DockTab::ALL.len() && l.len() + r.len() == DockTab::ALL.len() {
            (st.left_tabs, st.right_tabs) = (l, r);
        }
    }
    if let Some(t) = kv.get("left_tab").and_then(|w| tab_of(w)).filter(|t| st.left_tabs.contains(t)) {
        st.left_tab = t;
    }
    if let Some(t) = kv.get("right_tab").and_then(|w| tab_of(w)).filter(|t| st.right_tabs.contains(t)) {
        st.dock_tab = t;
    }
    if st.left_tabs.is_empty() {
        st.left_open = false;
    } else if !st.left_tabs.contains(&st.left_tab) {
        st.left_tab = st.left_tabs[0];
    }
    if st.right_tabs.is_empty() {
        st.dock_open = false;
    } else if !st.right_tabs.contains(&st.dock_tab) {
        st.dock_tab = st.right_tabs[0];
    }
    let w = |k: &str| kv.get(k).and_then(|v| v.parse::<f32>().ok()).map(|x| x.clamp(280.0, 720.0));
    if let (Some(l), Some(r)) = (w("left_width"), w("right_width")) {
        st.dock_w_set = Some([l, r]);
    }
}

/// The built-in defaults (a fresh state's profile).
pub fn defaults(st: &UiState) -> String {
    let fresh = UiState::new("h2", &crate::scene::Scene::preset("h2").expect("the h2 preset"));
    let mut text = capture(&fresh);
    // the docks' default width (a fresh state has not measured them)
    text = text.replace(&format!("left_width = {:.0}", fresh.dock_w[0]), "left_width = 336");
    text = text.replace(&format!("right_width = {:.0}", fresh.dock_w[1]), "right_width = 336");
    let _ = st;
    text
}

/// The window's profile bookkeeping: where the state in force lives, and when it was last written.
#[derive(Default)]
pub struct Profiles {
    /// `%LOCALAPPDATA%\atomic-sandbox\current.profile` in the window; `None` headless (never saved).
    pub current_file: Option<PathBuf>,
    written: String,
    changed_at: Option<Instant>,
    pub open: bool,
    name: String,
    list: Vec<(String, PathBuf)>,
    reset_armed: Option<Instant>,
    note: String,
}

/// Reads the state in force at launch (the window).
pub fn load_current(st: &mut UiState) {
    let Some(text) = st.profiles.current_file.as_ref().and_then(|p| std::fs::read_to_string(p).ok()) else { return };
    apply(st, &text);
    st.profiles.written = capture(st);
}

/// Writes the state in force now if it changed (the window's close).
pub fn flush(st: &mut UiState) {
    let Some(path) = st.profiles.current_file.clone() else { return };
    let now = capture(st);
    if now != st.profiles.written {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(&path, &now);
        st.profiles.written = now;
    }
}

/// Writes the state in force a moment after it last changed (once per frame: a cheap compare);
/// asks for the frame that writes it, so an idle window still saves.
pub fn autosave(ctx: &egui::Context, st: &mut UiState) {
    let Some(path) = st.profiles.current_file.clone() else { return };
    let now = capture(st);
    if now == st.profiles.written {
        st.profiles.changed_at = None;
        return;
    }
    let since = *st.profiles.changed_at.get_or_insert_with(Instant::now);
    ctx.request_repaint_after(Duration::from_millis(750));
    if since.elapsed() >= Duration::from_millis(700) {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(&path, &now);
        st.profiles.written = now;
        st.profiles.changed_at = None;
    }
}

fn refresh(st: &mut UiState) {
    let mut list = Vec::new();
    if let Some(dir) = crate::scene::profiles_dir() {
        if let Ok(rd) = std::fs::read_dir(dir) {
            for p in rd.flatten().map(|e| e.path()) {
                if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("profile")) {
                    list.push((p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(), p));
                }
            }
        }
    }
    list.sort();
    st.profiles.list = list;
}

/// Opens the Profiles window.
pub fn open(st: &mut UiState) {
    st.profiles.open = true;
    st.profiles.note.clear();
    refresh(st);
}

/// The Profiles window, while open.
pub fn show(ctx: &egui::Context, st: &mut UiState) {
    if !st.profiles.open {
        return;
    }
    let mut close = false;
    let modal = egui::Modal::new(egui::Id::new("profiles-window"))
        .frame(
            egui::Frame::new()
                .fill(theme::PANEL)
                .stroke(Stroke::new(1.0, theme::LINE))
                .corner_radius(10)
                .inner_margin(18)
                .shadow(egui::Shadow { offset: [0, 8], blur: 24, spread: 0, color: Color32::from_black_alpha(140) }),
        )
        .show(ctx, |ui| {
            ui.set_width(520.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new("Profiles").size(17.0).strong());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Close").clicked() {
                        close = true;
                    }
                });
            });
            theme::note(ui, "A profile is the app's layout and look: where the dock tabs are, which docks are open and how wide, the view and density settings, the units. The app keeps what you change and opens as you left it.");
            ui.add_space(8.0);
            let dir = crate::scene::profiles_dir().unwrap_or_else(|| PathBuf::from("profiles"));
            ui.label(RichText::new(dir.display().to_string()).monospace().size(10.5).color(theme::INK_FAINT));
            egui::ScrollArea::vertical().max_height(240.0).auto_shrink([false, true]).show(ui, |ui| {
                if st.profiles.list.is_empty() {
                    theme::note(ui, "No saved profiles yet: name the current one below and save it.");
                }
                let mut load = None;
                for (name, path) in &st.profiles.list {
                    egui::Frame::new().fill(theme::PANEL_2).corner_radius(6).inner_margin(egui::Margin::symmetric(10, 6)).show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(name).size(14.5).strong());
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                let b = egui::Button::new(RichText::new("Load").strong().color(Color32::WHITE)).fill(theme::GLOW).corner_radius(6).min_size(Vec2::new(90.0, 28.0));
                                if ui.add(b).on_hover_text("switch to this profile's layout and look").clicked() {
                                    load = Some((name.clone(), path.clone()));
                                }
                            });
                        });
                    });
                }
                if let Some((name, path)) = load {
                    match std::fs::read_to_string(&path) {
                        Ok(text) => {
                            apply(st, &text);
                            st.profiles.note = format!("loaded {name}");
                        }
                        Err(e) => st.profiles.note = format!("{name}: {e}"),
                    }
                }
            });
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut st.profiles.name).desired_width(220.0).hint_text("name"));
                let name = st.profiles.name.trim().to_string();
                let taken = st.profiles.list.iter().any(|(n, _)| n.eq_ignore_ascii_case(&name));
                let word = if taken { "Replace" } else { "Save current" };
                if ui.add_enabled(!name.is_empty(), egui::Button::new(word)).on_hover_text("a snapshot of the layout and look as they are now").clicked() {
                    let dir = crate::scene::profiles_dir().unwrap_or_else(|| PathBuf::from("profiles"));
                    let clean: String = name.chars().map(|c| if "\\/:*?\"<>|".contains(c) { '_' } else { c }).collect();
                    let _ = std::fs::create_dir_all(&dir);
                    match std::fs::write(dir.join(format!("{clean}.profile")), capture(st)) {
                        Ok(()) => st.profiles.note = format!("saved {clean}"),
                        Err(e) => st.profiles.note = format!("save {clean}: {e}"),
                    }
                    refresh(st);
                }
            });
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                let armed = st.profiles.reset_armed.is_some_and(|t| t.elapsed() < Duration::from_secs(4));
                let b = if armed {
                    egui::Button::new(RichText::new("Click again to reset").color(theme::DANGER_INK))
                } else {
                    egui::Button::new("Reset to the default profile")
                };
                if ui.add(b).on_hover_text("the built-in layout and look (two clicks)").clicked() {
                    if armed {
                        let d = defaults(st);
                        apply(st, &d);
                        st.profiles.reset_armed = None;
                        st.profiles.note = "reset to the default profile".into();
                    } else {
                        st.profiles.reset_armed = Some(Instant::now());
                    }
                }
                if !st.profiles.note.is_empty() {
                    ui.label(RichText::new(&st.profiles.note).size(12.5).color(theme::ACCENT_INK));
                }
            });
            // the completion times behind the ETAs: the user's into the repository's file
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                let (mine, shipped) = st.timings.counts();
                ui.label(RichText::new(format!("completion times: {mine} yours · {shipped} in the repo")).size(12.5).color(theme::INK_DIM));
                if ui.add_enabled(mine > 0, egui::Button::new("Save mine into timings/")).on_hover_text("merge your measured times (loading scenes, …) into timings/timings.txt — commit it to ship them as everyone's starting ETAs").clicked() {
                    st.profiles.note = match st.timings.publish() {
                        Ok(p) => format!("saved to {}", p.display()),
                        Err(e) => format!("timings: {e}"),
                    };
                }
            });
        });
    if close || modal.should_close() {
        st.profiles.open = false;
    }
}
