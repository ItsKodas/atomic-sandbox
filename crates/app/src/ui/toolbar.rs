//! The bottom toolbar: what the hands do. Groups (mockup §1): Time (Rewind, Play / Pause, Step),
//! Speed (steps per tick, the measured fs/s large, the watchable preset and this tick's rate),
//! dt, Tool (Orbit / Place / Select), the element chips (a pick switches to Place), the
//! selection's actions (Pin, Focus, Delete) and the Solver (CPU / GPU / Estimated, with the
//! trust light and who answered this step). The compact layout (mockup §4) drops the labels and
//! folds the element chips into one chip with a periodic-table flyout.
use super::icons::{self, Icon};
use super::menus::dt_field;
use super::search;
use super::{source_colour, source_label, theme, trust, watchable_speed, Action, PanelInput, ToolMode, UiState, SPARK_E};
use crate::sim::Backend;
use egui::{Button, Frame, Margin, RichText, Stroke, Ui, Vec2};
use render::elements;
use sim::dynamics::{DT_WARN_FS, MAX_SPEED};

const SPEED_TIP: &str = "time steps attempted per ≈16 ms tick; never changes dt. One DFT step usually fills a tick by itself, so the measured fs/s is what counts; with the estimated solver answering (~10 µs a step) a tick runs all its steps and then waits out its 16 ms, except at 1000 (as fast as possible). Above ~30 fs/s a 10 fs bond vibration is faster than the frame rate — you will see a shimmer, not a swing; use the watchable preset to see motion.";
const RATE_TIP: &str = "simulated femtoseconds per wall second: over the last 2 s of play (what you see), and over the last tick alone (a DFT step and a surrogate tick differ by 10⁴)";
const SAMPLING_TIP: &str = "a sampling pass is running: the dynamics stay paused until it ends (or Cancel it)";
const REWIND_TIP: &str = "reverse every velocity (v → −v): without thermostat or drag the trajectory retraces itself";
const STEP_TIP: &str = "pause and advance one time step";
const PIN_TIP: &str = "hold this nucleus still (velocity zeroed every step)";
const ORBIT_TIP: &str = "middle-drag or left-drag on empty space; right-drag pans; wheel zooms";
const PLACE_TIP: &str = "click to drop a nucleus at rest (also while playing)";
const SELECT_TIP: &str = "click a nucleus; drag it: paused = move it there, playing = pull it with a spring";
const CPU_TIP: &str = "qm::scf, f64 reference; a solve cannot be interrupted (one MD step = one SCF solve: slow)";
const GPU_TIP: &str = "wgpu compute, f32 grid; Pause and edits abandon the solve in flight";

fn estimated_tip(gpu: bool) -> String {
    let teacher = if gpu { "GPU" } else { "CPU" };
    format!("a surrogate model fitted to this scene's DFT (teacher: the {teacher} solver), with a Bayesian uncertainty: it answers a step in ~10 µs when it is sure and hands the step to DFT when it is not. Without a model a sampling pass runs first.")
}

pub fn show(root: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, compact: bool, acts: &mut Vec<Action>) {
    egui::Panel::bottom("toolbar")
        .exact_size(if compact { 50.0 } else { 84.0 })
        .resizable(false)
        .frame(Frame::new().fill(theme::PANEL).inner_margin(Margin::symmetric(6, 6)).stroke(Stroke::new(1.0, theme::LINE)))
        .show(root, |ui| {
            ui.horizontal_centered(|ui| {
                if compact {
                    compact_bar(ui, st, input, acts)
                } else {
                    full_bar(ui, st, input, acts)
                }
            });
        });
}

/// One labelled group: caps label (with an optional "i"), a 30-point row, and a sub-line, both
/// drawn by `add`; the group's rect is marked with the search `anchor`.
fn group(ui: &mut Ui, st: &mut UiState, anchor: &str, label: &str, tip: Option<&str>, add: impl FnOnce(&mut Ui, &mut UiState)) {
    let r = ui.vertical(|ui| {
        ui.spacing_mut().item_spacing.y = 3.0;
        ui.horizontal(|ui| {
            ui.label(theme::caps(label));
            if let Some(t) = tip {
                theme::info(ui, t);
            }
        });
        add(ui, st);
    });
    search::mark(ui, st, anchor, r.response.rect);
    ui.add(egui::Separator::default().vertical().spacing(8.0));
}

/// The 30-point main row of a group.
fn row<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
    ui.horizontal(|ui| {
        ui.set_min_height(30.0);
        add(ui)
    })
    .inner
}

fn sub(ui: &mut Ui, text: impl Into<String>) -> egui::Response {
    ui.label(RichText::new(text.into()).size(10.5).color(theme::INK_DIM))
}

fn kbd(ui: &mut Ui, key: &str) {
    Frame::new()
        .fill(theme::PANEL_2)
        .stroke(Stroke::new(1.0, theme::LINE))
        .corner_radius(4)
        .inner_margin(Margin::symmetric(4, 0))
        .show(ui, |ui| ui.label(RichText::new(key).monospace().size(10.0).color(theme::INK_DIM)));
}

fn full_bar(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    let sampling = input.sampling();
    let d = input.snapshot.map(|s| &s.dynamics);
    group(ui, st, "tb-time", "Time", None, |ui, _| {
        row(ui, |ui| time_buttons(ui, input, sampling, false, acts));
        ui.horizontal(|ui| {
            kbd(ui, "Space");
            sub(ui, "play / pause");
        });
    });
    group(ui, st, "tb-speed", "Speed · steps per tick", Some(SPEED_TIP), |ui, st| {
        row(ui, |ui| {
            let mut speed = st.speed;
            ui.spacing_mut().slider_width = 80.0;
            if ui.add(egui::Slider::new(&mut speed, 1..=MAX_SPEED).logarithmic(true).show_value(false)).on_hover_text(SPEED_TIP).changed()
                | ui.add(egui::DragValue::new(&mut speed).range(1..=MAX_SPEED)).on_hover_text(SPEED_TIP).changed()
            {
                acts.push(Action::SetSpeed(speed));
            }
            let rate = input.measured_fs_per_s.map_or("—".to_string(), fs_per_s);
            ui.label(RichText::new(rate).monospace().strong().size(15.0)).on_hover_text(RATE_TIP);
            ui.label(RichText::new("fs/s").color(theme::INK_DIM)).on_hover_text("simulated fs per wall second over the last 2 s of play");
        });
        ui.horizontal(|ui| match d {
            Some(d) => {
                let (n, rate) = watchable_speed(st.dt_fs, d.ms_per_step);
                if ui
                    .add(Button::new(RichText::new(format!("watchable (≈ {rate:.0} fs/s)")).size(10.5).color(theme::ACCENT_INK)).frame(false))
                    .on_hover_text("steps per tick for one ~10 fs H–X vibration in ≈ 0.5 s of wall time, at the current time step and measured cost per step")
                    .clicked()
                {
                    acts.push(Action::SetSpeed(n));
                }
                sub(ui, format!("· tick {:.2} fs/s", d.wall_ratio)).on_hover_text(RATE_TIP);
            }
            None => {
                sub(ui, "no solution yet");
            }
        });
    });
    group(ui, st, "tb-dt", "dt", None, |ui, st| {
        row(ui, |ui| dt_field(ui, st, acts));
        match d.and_then(|d| d.dt_warning.as_ref()) {
            Some(w) => sub(ui, "under-resolved").on_hover_text(w.as_str()),
            None => sub(ui, format!("amber > {DT_WARN_FS}")),
        };
    });
    group(ui, st, "tb-tool", "Tool", None, |ui, st| {
        row(ui, |ui| tool_segments(ui, st, false, acts));
        sub(ui, "right-drag pans · wheel zooms");
    });
    group(ui, st, "tb-elements", "Place element · Z 1–10", None, |ui, st| {
        row(ui, |ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            for z in 1..=10u8 {
                if element_chip(ui, z, st.element == z, 21.0).clicked() {
                    acts.push(Action::Element(z));
                }
            }
        });
        sub(ui, format!("{} (Z = {}) · a pick switches to Place", elements::symbol(st.element), st.element));
    });
    let sel = input.selection();
    let label = match sel {
        Some((i, n)) => format!("Selection · {} #{i}", elements::symbol(n.z)),
        None => "Selection".into(),
    };
    group(ui, st, "tb-selection", &label, None, |ui, _| {
        row(ui, |ui| selection_buttons(ui, input, acts));
        match sel {
            Some((_, n)) => ui.label(
                RichText::new(format!("({:.2}, {:.2}, {:.2}) Bohr", n.pos[0], n.pos[1], n.pos[2])).monospace().size(10.5).color(theme::INK_DIM),
            ),
            None => sub(ui, "no selection (Select tool, click a nucleus)"),
        };
    });
    let r = ui.vertical(|ui| {
        ui.spacing_mut().item_spacing.y = 3.0;
        ui.label(theme::caps("Solver"));
        row(ui, |ui| solver_segments(ui, st, false, acts));
        ui.horizontal(|ui| solver_sub(ui, input));
    });
    search::mark(ui, st, "tb-solver", r.response.rect);
}

fn compact_bar(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    let sampling = input.sampling();
    let r = ui.horizontal(|ui| time_buttons(ui, input, sampling, true, acts));
    search::mark(ui, st, "tb-time", r.response.rect);
    ui.separator();
    let r = ui.horizontal(|ui| {
        let mut speed = st.speed;
        if ui.add(egui::DragValue::new(&mut speed).range(1..=MAX_SPEED).prefix("×")).on_hover_text(SPEED_TIP).changed() {
            acts.push(Action::SetSpeed(speed));
        }
        let rate = input.measured_fs_per_s.map_or("—".to_string(), fs_per_s);
        ui.label(RichText::new(format!("{rate} fs/s")).monospace().color(theme::INK_DIM)).on_hover_text(RATE_TIP);
    });
    search::mark(ui, st, "tb-speed", r.response.rect);
    let r = ui.horizontal(|ui| dt_field(ui, st, acts));
    search::mark(ui, st, "tb-dt", r.response.rect);
    ui.separator();
    let r = ui.horizontal(|ui| tool_segments(ui, st, true, acts));
    search::mark(ui, st, "tb-tool", r.response.rect);
    let r = ui.horizontal(|ui| element_flyout(ui, st, acts));
    search::mark(ui, st, "tb-elements", r.response.rect);
    ui.separator();
    let r = ui.horizontal(|ui| {
        selection_buttons(ui, input, acts);
        if input.selection().is_none() {
            ui.label(RichText::new("no selection").color(theme::INK_FAINT));
        }
    });
    search::mark(ui, st, "tb-selection", r.response.rect);
    ui.separator();
    let r = ui.horizontal(|ui| {
        solver_segments(ui, st, true, acts);
        solver_sub(ui, input);
    });
    search::mark(ui, st, "tb-solver", r.response.rect);
}

/// "1 240" above 100 fs/s, "15.6" below.
fn fs_per_s(r: f64) -> String {
    if r >= 100.0 {
        format!("{r:.0}")
    } else {
        format!("{r:.1}")
    }
}

fn time_buttons(ui: &mut Ui, input: &PanelInput<'_>, sampling: bool, compact: bool, acts: &mut Vec<Action>) {
    if icons::button(ui, Icon::Rewind, 28.0, false, true).on_hover_text(REWIND_TIP).clicked() {
        acts.push(Action::Rewind);
    }
    let (icon, label) = if input.playing { (Icon::Pause, "Pause") } else { (Icon::Play, "Play") };
    let r = if compact {
        let r = ui.add_enabled(
            !sampling,
            Button::new("").min_size(Vec2::new(30.0, 28.0)).fill(theme::ACCENT).stroke(Stroke::new(1.0, theme::ACCENT)),
        );
        let c = egui::Color32::from_rgb(0x17, 0x11, 0x2a);
        icons::paint(ui.painter(), egui::Rect::from_center_size(r.rect.center(), Vec2::splat(15.0)), icon, c);
        r.on_hover_text(format!("{label} (Space)"))
    } else {
        icons::text_button(ui, icon, label, 84.0, true, !sampling).on_hover_text("Space")
    };
    if r.on_disabled_hover_text(SAMPLING_TIP).clicked() {
        acts.push(Action::TogglePlay);
    }
    if icons::button(ui, Icon::Step, 28.0, false, !sampling)
        .on_hover_text(STEP_TIP)
        .on_disabled_hover_text(SAMPLING_TIP)
        .clicked()
    {
        acts.push(Action::Step);
    }
}

fn tool_segments(ui: &mut Ui, st: &UiState, compact: bool, acts: &mut Vec<Action>) {
    Frame::new()
        .fill(theme::APP)
        .stroke(Stroke::new(1.0, theme::LINE))
        .corner_radius(7)
        .inner_margin(Margin::same(2))
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            for (mode, icon, text, tip) in [
                (ToolMode::Orbit, Icon::Orbit, "Orbit", ORBIT_TIP),
                (ToolMode::Place, Icon::Place, "Place", PLACE_TIP),
                (ToolMode::Select, Icon::Select, "Select", SELECT_TIP),
            ] {
                let text = if compact { "" } else { text };
                if icons::seg_button(ui, icon, text, st.mode == mode).on_hover_text(tip).clicked() {
                    acts.push(Action::Tool(mode));
                }
            }
        });
}

/// Display name of Z (the atom card's).
fn element_name(z: u8) -> &'static str {
    crate::atom_card::element_name(z)
}

/// An element chip: the symbol with Z small at the top left; the selected one glows.
pub fn element_chip(ui: &mut Ui, z: u8, selected: bool, width: f32) -> egui::Response {
    let (rect, r) = ui.allocate_exact_size(Vec2::new(width, 28.0), egui::Sense::click());
    let p = ui.painter();
    let (bg, stroke) = if selected {
        (theme::ACCENT_BG, Stroke::new(1.0, theme::ACCENT))
    } else if r.hovered() {
        (theme::PANEL_3, Stroke::new(1.0, theme::LINE))
    } else {
        (theme::PANEL_2, Stroke::new(1.0, theme::LINE))
    };
    if selected {
        p.rect_filled(rect.expand(2.0), 7.0, theme::GLOW.gamma_multiply(0.18));
    }
    p.rect(rect, 5.0, bg, stroke, egui::StrokeKind::Inside);
    p.text(rect.left_top() + Vec2::new(2.5, 1.0), egui::Align2::LEFT_TOP, z.to_string(), egui::FontId::monospace(7.0), theme::INK_FAINT);
    p.text(
        rect.center() + Vec2::new(0.0, 2.0),
        egui::Align2::CENTER_CENTER,
        elements::symbol(z),
        egui::FontId::monospace(12.0),
        if selected { egui::Color32::WHITE } else { theme::INK },
    );
    r.on_hover_text(format!("{} (Z = {z}): click to place it (switches the tool to Place)", element_name(z)))
}

/// Compact: one chip for the element, opening a periodic-table flyout (rows 1–2: H, He; Li–Ne).
fn element_flyout(ui: &mut Ui, st: &UiState, acts: &mut Vec<Action>) {
    let r = ui.add(
        Button::selectable(true, RichText::new(format!("{} ▾", elements::symbol(st.element))).monospace().strong())
            .min_size(Vec2::new(40.0, 26.0)),
    );
    let r = r.on_hover_text("element to place (Z 1–10)");
    egui::Popup::from_toggle_button_response(&r)
        .id(egui::Id::new("element-flyout"))
        .align(egui::RectAlign::TOP_START)
        .gap(6.0)
        .show(|ui| {
            egui::Grid::new("ptable").spacing(Vec2::splat(3.0)).show(ui, |ui| {
                for row in [[Some(1u8), None, None, None, None, None, None, Some(2)], [3, 4, 5, 6, 7, 8, 9, 10].map(Some)] {
                    for z in row {
                        match z {
                            Some(z) => {
                                if element_chip(ui, z, st.element == z, 26.0).clicked() {
                                    acts.push(Action::Element(z));
                                    egui::Popup::close_id(ui.ctx(), egui::Id::new("element-flyout"));
                                }
                            }
                            None => {
                                ui.label("");
                            }
                        }
                    }
                    ui.end_row();
                }
            });
            theme::note(ui, "Z 1–10 (Z > 10 needs pseudopotentials)");
        });
}

fn selection_buttons(ui: &mut Ui, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    let sel = input.selection().map(|(i, _)| i);
    let pinned = sel.and_then(|i| input.pinned(i));
    let mut p = pinned.unwrap_or(false);
    if ui
        .add_enabled(pinned.is_some(), egui::Checkbox::new(&mut p, "Pin"))
        .on_hover_text(PIN_TIP)
        .changed()
    {
        if let Some(i) = sel {
            acts.push(Action::Pin(i, p));
        }
    }
    if ui.add_enabled(sel.is_some(), Button::new("Focus")).on_hover_text("Focus selected: centre the camera on it").clicked() {
        acts.push(Action::Focus);
    }
    if ui
        .add_enabled(sel.is_some(), Button::new(RichText::new("Delete").color(theme::DANGER_INK)))
        .on_hover_text("Delete selected (the Delete key; also while playing)")
        .clicked()
    {
        if let Some(i) = sel {
            acts.push(Action::Delete(i));
        }
    }
}

fn solver_segments(ui: &mut Ui, st: &UiState, compact: bool, acts: &mut Vec<Action>) {
    Frame::new()
        .fill(theme::APP)
        .stroke(Stroke::new(1.0, theme::LINE))
        .corner_radius(7)
        .inner_margin(Margin::same(2))
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            if ui.add(Button::selectable(st.backend == Backend::Cpu, "CPU")).on_hover_text(CPU_TIP).clicked() {
                acts.push(Action::Backend(Backend::Cpu));
            }
            if ui
                .add_enabled(st.gpu_available, Button::selectable(st.backend == Backend::Gpu, "GPU"))
                .on_hover_text(GPU_TIP)
                .on_disabled_hover_text("no GPU adapter")
                .clicked()
            {
                acts.push(Action::Backend(Backend::Gpu));
            }
            let est = if compact { "Est." } else { "Estimated" };
            if ui
                .add(Button::selectable(st.backend == Backend::Estimated, est))
                .on_hover_text(estimated_tip(st.gpu_available))
                .clicked()
            {
                acts.push(Action::Backend(Backend::Estimated));
            }
        });
}

/// The solver's sub-line: the trust light, who answered this step, trust, fallbacks, epoch
/// (estimated), or the Kohn-Sham engine.
fn solver_sub(ui: &mut Ui, input: &PanelInput<'_>) {
    let Some(s) = input.snapshot else {
        theme::light(ui, theme::INK_FAINT);
        ui.label(RichText::new("waiting for the first solve").size(10.5).color(theme::INK_DIM));
        return;
    };
    let d = &s.dynamics;
    if s.backend != Backend::Estimated {
        theme::light(ui, SPARK_E);
        ui.label(RichText::new(format!("DFT · {}", s.engine.label())).size(10.5).color(SPARK_E));
        return;
    }
    let m = input.model();
    let (colour, text, tip) = trust(m);
    if let Some(p) = d.sampling.as_ref().filter(|_| input.sampling()) {
        theme::light(ui, theme::INK_FAINT);
        ui.label(RichText::new(format!("building the model · {} {}/{}", p.stage, p.done, p.total)).size(10.5).color(theme::INK_DIM));
        return;
    }
    let r = theme::light(ui, colour);
    if let Some(t) = tip {
        r.on_hover_text(t);
    }
    if m.is_none_or(|m| !m.fitted) {
        ui.label(RichText::new("no model yet").size(10.5).color(theme::INK_DIM));
        return;
    }
    let src = d.source;
    ui.label(RichText::new(source_label(src)).size(10.5).color(source_colour(src))).on_hover_text(super::SOURCE_TIP);
    let short = if text == "yes" {
        "trusted".to_string()
    } else if text.starts_with("yes") {
        "trusted, see Model".into()
    } else {
        "untrusted, see Model".into()
    };
    ui.label(
        RichText::new(format!(
            "· {short} · {:.0} % fallback · epoch {}",
            100.0 * d.fallback_fraction_100,
            m.map_or(0, |m| m.epoch)
        ))
        .size(10.5)
        .color(theme::INK_DIM),
    )
    .on_hover_text(format!(
        "trusted: {text}; fallbacks over the last 100 steps: {:.0} %; model epoch {} (Model tab)",
        100.0 * d.fallback_fraction_100,
        m.map_or(0, |m| m.epoch)
    ));
}
