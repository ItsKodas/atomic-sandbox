//! The bottom toolbar: what the hands do. Groups (mockup §1): Time (Rewind, Play / Pause, Step),
//! Speed (steps per tick, the measured fs/s large, the watchable preset and this tick's rate),
//! dt, Tool (Orbit / Place / Select), the element chips (a pick switches to Place), the
//! selection's actions (Pin, Focus, Delete) and the Solver (CPU / GPU / Estimated, with the
//! trust light and who answered this step). The compact layout (mockup §4) drops the labels and
//! folds the element chips into one chip with a periodic-table flyout.
//!
//! Fixed widths (user report 2026-10-14: "when the tick changes on speed too much the whole bottom
//! bar keeps growing and shrinking super fast"): every group, and every slot holding a live value,
//! is as wide as the widest text it can show, measured once per frame with the fonts in use
//! ([`Widths`]) and never from the live values. Live numbers are monospace (tabular digits) in a
//! bounded format ([`fs_per_s`]: at most five characters, right-aligned; percentages and epochs
//! padded), and the secondary lines elide at their slot's edge with the full text on hover. So a
//! rate going 1 → 79 365 fs/s, a fallback share 0 → 100 % or a new epoch moves nothing.
use super::icons::{self, Icon};
use super::menus::dt_field;
use super::search;
use super::{probe, probed, source_colour, source_label, theme, trust, watchable_speed, TrustKind, Action, PanelInput, ToolMode, UiState, SPARK_E};
use crate::sim::Backend;
use egui::text::{LayoutJob, TextFormat};
use egui::{Align, Button, Color32, Direction, FontId, Frame, Label, Layout, Margin, RichText, Stroke, TextStyle, Ui, Vec2, WidgetText};
use render::elements;
use sim::dynamics::{DT_WARN_FS, MAX_SPEED};

const SPEED_TIP: &str = "time steps attempted per ≈16 ms tick; never changes dt. One DFT step usually fills a tick by itself, so the measured fs/s is what counts; with the estimated solver answering (~10 µs a step) a tick runs all its steps and then waits out its 16 ms, except at 1000 (as fast as possible). Above ~30 fs/s a 10 fs bond vibration is faster than the frame rate — you will see a shimmer, not a swing; use the watchable preset to see motion.";
const RATE_TIP: &str = "simulated femtoseconds per wall second: over the last 2 s of play (what you see), and over the last tick alone (a DFT step and a surrogate tick differ by 10⁴)";
const SAMPLING_TIP: &str = "a sampling pass is running: the dynamics stay paused until it ends (or Cancel it)";
const RELAXING_TIP: &str = "relaxing: the dynamics stay paused until the relax ends (or Stop it)";
const REWIND_TIP: &str = "reverse every velocity (v → −v): without thermostat or drag the trajectory retraces itself";
const STEP_TIP: &str = "pause and advance one time step";
const PIN_TIP: &str = "hold this nucleus still (velocity zeroed every step)";
const ORBIT_TIP: &str = "middle-drag or left-drag on empty space; right-drag pans; wheel zooms (wheel while dragging an atom sets the pull)";
const PLACE_TIP: &str = "click to drop a nucleus at rest (also while playing)";
const SELECT_TIP: &str = "click a nucleus; drag it: paused = move it there, playing = pull it with a spring; wheel while dragging sets the pull";
const CPU_TIP: &str = "qm::scf, f64 reference; a solve cannot be interrupted (one MD step = one SCF solve: slow)";
const GPU_TIP: &str = "wgpu compute, f32 grid; Pause and edits abandon the solve in flight";

fn estimated_tip(gpu: bool) -> String {
    let teacher = if gpu { "GPU" } else { "CPU" };
    format!("a surrogate model fitted to this scene's DFT (teacher: the {teacher} solver), with a Bayesian uncertainty: it answers a step in ~10 µs when it is sure and hands the step to DFT when it is not. Without a model a sampling pass runs first.")
}

/// Draws the toolbar and returns its rect (the progress bar sits on its top edge).
pub fn show(root: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, compact: bool, acts: &mut Vec<Action>) -> egui::Rect {
    // The full layout's fixed widths need `Widths::full` points: on a window narrower than that
    // (but above the shared `COMPACT_WIDTH`) the toolbar takes its compact layout rather than
    // running its Solver group off the edge. The top bar keeps its own switch.
    let compact = compact || root.available_width() < Widths::measure(root).full;
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
        })
        .response
        .rect
}

/// The sub-lines' text size.
const SUB: f32 = 10.5;
/// Room beside a group's widest measured content.
const PAD: f32 = 2.0;
/// The width a separator between two groups takes (the line in its middle), besides the item
/// spacing on either side.
const SEP: f32 = 4.0;
/// The least room (points) the compact Solver line is drawn in: its light and a few letters.
const MIN_SUB: f32 = 48.0;
/// The selection hint. The Selection group is sized for a selection's coordinates (|x| < 10
/// Bohr), not for this: at 1400 pt that leaves the Solver line its room, and the hint elides its
/// last words (full text on hover).
const NO_SELECTION: &str = "no selection (Select tool, click a nucleus)";
/// The Relax / Stop button's least width (the Time group, full layout).
const RELAX_W: f32 = 48.0;
/// The element group's sub-line opener of the periodic table.
const PTABLE_OPENER: &str = "periodic table ›";

/// Fixed widths (points) of the toolbar's groups and of the slots inside them that hold live
/// values (module doc): each is the widest text that place can show, measured with the fonts in
/// use, plus the widgets' fixed sizes — a pure function of the style, never of the live values.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Widths {
    // the full layout's groups
    time: f32,
    speed: f32,
    dt: f32,
    tool: f32,
    elements: f32,
    selection: f32,
    /// The Solver group with its sub-line whole (it gives way to the window edge, never to a
    /// value: [`Self::last`]).
    solver: f32,
    /// The Solver group's segmented row: the least it is given.
    solver_row: f32,
    /// The whole full bar with the Solver group at its row: the window width it needs.
    full: f32,
    // the full layout's slots
    speed_field: f32,
    rate: f32,
    dt_field: f32,
    // the compact layout's groups
    c_time: f32,
    c_speed: f32,
    c_dt: f32,
    c_tool: f32,
    c_elements: f32,
    c_selection: f32,
    c_solver: f32,
    c_solver_row: f32,
    // the compact layout's slots
    c_speed_field: f32,
    c_rate: f32,
}

impl Widths {
    fn measure(ui: &Ui) -> Self {
        let s = ui.style().clone();
        let sp = s.spacing.item_spacing.x;
        let bp = s.spacing.button_padding.x;
        let interact = s.spacing.interact_size;
        let (body, button, mono) = (TextStyle::Body.resolve(&s), TextStyle::Button.resolve(&s), TextStyle::Monospace.resolve(&s));
        let drag = s.drag_value_text_style.resolve(&s);
        let (sub, sub_mono) = (FontId::proportional(SUB), FontId::monospace(SUB));
        let w = |t: &str, f: &FontId| tw(ui, t, f.clone());
        let max = |v: &[f32]| v.iter().copied().fold(0.0, f32::max);
        let btn = |t: &str| w(t, &button) + 2.0 * bp;
        let caps = |t: &str, tip: bool| w(&t.to_uppercase(), &FontId::monospace(9.5)) + if tip { sp + 14.0 } else { 0.0 };
        // a DragValue showing `t` (a little over its text: the slot centres it)
        let field = |t: &str| (w(t, &drag) + 2.0 * bp).max(interact.x) + 2.0;
        // a segmented frame (margin 2 + stroke 1 a side, 2 between segments)
        let segs = |v: &[f32]| v.iter().sum::<f32>() + 2.0 * (v.len() as f32 - 1.0) + 6.0;
        // the five-character fs/s slot (`fs_per_s`), in `f`
        let rate = |f: &FontId, unit: &str| max(&[w(&format!("00.0k{unit}"), f), w(&format!("    —{unit}"), f)]);
        let line_w = |parts: &[&str]| parts.iter().enumerate().map(|(k, p)| w(p, if k % 2 == 1 { &sub_mono } else { &sub })).sum::<f32>();
        let kbd = w("Space", &FontId::monospace(10.0)) + 10.0;
        let solver_sub = 10.0 + sp + w(source_label(sim::Source::Fallback), &sub) + sp + line_w(&["· trusted · ", "100", " % fallback · epoch ", "9999"]);

        let speed_field = field(&MAX_SPEED.to_string());
        let rate_full = rate(&FontId::monospace(15.0), "");
        let dt_field = field("0.000 fs");
        let play = btn("      Pause").max(84.0);
        let relax = btn("Relax").max(btn("Stop")).max(RELAX_W);
        let selection_row = s.spacing.icon_width + s.spacing.icon_spacing + w("Pin", &body) + sp + btn("Focus") + sp + btn("Delete");
        let solver_row = segs(&[btn("CPU"), btn("GPU"), btn("Estimated")]);
        let c_speed_field = field(&format!("×{MAX_SPEED}"));
        let c_rate = rate(&mono, " fs/s");
        let c_solver_row = segs(&[btn("CPU"), btn("GPU"), btn("Est.")]);
        let mut widths = Widths {
            time: max(&[caps("Time", false), 28.0 + sp + play + sp + 28.0 + sp + relax, kbd + sp + w("play / pause", &sub)]) + PAD,
            speed: max(&[
                caps("Speed · steps per tick", true),
                80.0 + sp + speed_field + sp + rate_full + sp + w("fs/s", &body),
                line_w(&["watchable (≈ ", "99.9", " fs/s)"]) + 2.0 * bp + sp + line_w(&["· tick ", "00.0k", " fs/s"]),
            ]) + PAD,
            dt: max(&[caps("dt", false), dt_field, w("under-resolved", &sub), w(&format!("amber > {DT_WARN_FS}"), &sub)]) + PAD,
            tool: max(&[
                caps("Tool", false),
                segs(&[btn("      Orbit"), btn("      Place"), btn("      Select")]),
                w("right-drag pans · wheel zooms", &sub),
                line_w(&["drag + wheel: pull ", "0.0000", " Ha/Bohr"]),
            ]) + PAD,
            elements: max(&[caps("Place element · Z 1–10", false), 10.0 * 21.0 + 9.0 * 2.0, w(PTABLE_OPENER, &sub) + 2.0 * bp + sp + w("²²Ne (Z = 10)", &sub)]) + PAD,
            selection: max(&[caps("Selection · He #99", false), selection_row, w("(-0.00, -0.00, -0.00) Bohr", &sub_mono)]) + PAD,
            solver: max(&[caps("Solver", false), solver_row, solver_sub]) + PAD,
            solver_row: solver_row + PAD,
            full: 0.0,
            speed_field,
            rate: rate_full,
            dt_field,
            c_time: 28.0 + sp + 30.0 + sp + 28.0,
            c_speed: c_speed_field + sp + c_rate + sp + w("watch", &sub) + 2.0 * bp,
            c_dt: dt_field,
            c_tool: segs(&[30.0; 3]),
            c_elements: (w("He ▾", &mono) + 2.0 * bp).max(40.0),
            c_selection: selection_row + sp + w("no selection", &body),
            c_solver: c_solver_row + sp + solver_sub,
            c_solver_row,
            c_speed_field,
            c_rate,
        };
        // the panel's margin and stroke on both sides, six separators (8 + item spacing a side)
        let g = &widths;
        widths.full = 2.0 * 7.0 + g.time + g.speed + g.dt + g.tool + g.elements + g.selection + 6.0 * (SEP + 2.0 * sp) + g.solver_row;
        widths
    }

    /// The last group's width: its whole content where the window has room, else what is left
    /// (it elides), never less than its static row. Depends on the window, not on any value.
    fn last(ui: &Ui, whole: f32, row: f32) -> f32 {
        whole.min(ui.available_width()).max(row)
    }
}

/// Width (points) of `text` in `font`, unwrapped.
fn tw(ui: &Ui, text: &str, font: FontId) -> f32 {
    ui.ctx().fonts_mut(|f| f.layout_no_wrap(text.to_owned(), font, Color32::WHITE).size().x)
}

/// A line whose live numbers are monospace (tabular digits): `parts` alternate text and numbers
/// (numbers at the odd indices).
fn line(parts: &[&str], size: f32, colour: Color32) -> LayoutJob {
    let mut job = LayoutJob::default();
    for (k, p) in parts.iter().enumerate() {
        let font_id = if k % 2 == 1 { FontId::monospace(size) } else { FontId::proportional(size) };
        job.append(p, 0.0, TextFormat { font_id, color: colour, ..Default::default() });
    }
    job
}

/// Lays `add` out in a child exactly `w` points wide and `h` high at most (`layout` inside it),
/// so what it shows never changes the space it takes; a truncating label inside elides at its
/// right edge.
fn fixed<R>(ui: &mut Ui, w: f32, h: f32, layout: Layout, add: impl FnOnce(&mut Ui) -> R) -> egui::InnerResponse<R> {
    ui.allocate_ui_with_layout(Vec2::new(w, h), layout, |ui| {
        ui.set_width(w);
        add(ui)
    })
}

/// A DragValue (or anything) stretched over a fixed `w`-wide slot: "1" and "1000" take the same room.
fn field(ui: &mut Ui, w: f32, add: impl FnOnce(&mut Ui) -> egui::Response) -> egui::Response {
    let h = ui.spacing().interact_size.y;
    fixed(ui, w, h, Layout::centered_and_justified(Direction::LeftToRight), add).inner
}

/// A compact group: a fixed-width row the bar's full height, contents centred vertically.
fn slot<R>(ui: &mut Ui, w: f32, add: impl FnOnce(&mut Ui) -> R) -> egui::InnerResponse<R> {
    let h = ui.available_height();
    fixed(ui, w, h, Layout::left_to_right(Align::Center), add)
}

#[cfg(test)]
thread_local! {
    /// Test hook: each toolbar group's rect in the last frame and the width it was given.
    pub(crate) static GROUPS: std::cell::RefCell<std::collections::BTreeMap<&'static str, (egui::Rect, f32)>> = Default::default();
}

/// Marks a group for the search (and, in tests, records its rect against its fixed width `w`).
fn mark(ui: &Ui, st: &mut UiState, anchor: &'static str, rect: egui::Rect, w: f32) {
    #[cfg(test)]
    GROUPS.with(|g| {
        g.borrow_mut().insert(anchor, (rect, w));
    });
    #[cfg(not(test))]
    let _ = w;
    search::mark(ui, st, anchor, rect);
}

/// One labelled group `w` points wide: caps label (with an optional "i"), a 30-point row, and a
/// sub-line, both drawn by `add`; the group's rect is marked with the search `anchor`.
#[allow(clippy::too_many_arguments)]
fn group(ui: &mut Ui, st: &mut UiState, anchor: &'static str, label: &str, tip: Option<&str>, w: f32, add: impl FnOnce(&mut Ui, &mut UiState)) {
    group_body(ui, st, anchor, label, tip, w, add);
    ui.add(egui::Separator::default().vertical().spacing(SEP));
}

fn group_body(ui: &mut Ui, st: &mut UiState, anchor: &'static str, label: &str, tip: Option<&str>, w: f32, add: impl FnOnce(&mut Ui, &mut UiState)) {
    let r = ui.vertical(|ui| {
        ui.set_width(w);
        ui.spacing_mut().item_spacing.y = 3.0;
        ui.horizontal(|ui| {
            ui.add(Label::new(theme::caps(label)).truncate());
            if let Some(t) = tip {
                theme::info(ui, t);
            }
        });
        add(ui, st);
    });
    mark(ui, st, anchor, r.response.rect, w);
}

/// The 30-point main row of a group.
fn row<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
    ui.horizontal(|ui| {
        ui.set_min_height(30.0);
        add(ui)
    })
    .inner
}

/// A sub-line that elides at its slot's edge (the full text on hover); plain text is drawn dim at
/// [`SUB`], styled text and layout jobs as given.
fn sub(ui: &mut Ui, text: impl Into<WidgetText>) -> egui::Response {
    let text = match text.into() {
        WidgetText::Text(t) => WidgetText::from(RichText::new(t).size(SUB).color(theme::INK_DIM)),
        styled => styled,
    };
    ui.add(Label::new(text).truncate())
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
    let w = Widths::measure(ui);
    let sampling = input.sampling();
    let d = input.snapshot.map(|s| &s.dynamics);
    group(ui, st, "tb-time", "Time", None, w.time, |ui, _| {
        row(ui, |ui| {
            time_buttons(ui, input, sampling, false, acts);
            super::relax_toggle(ui, input, Vec2::new(RELAX_W, 28.0), ("relax", "relax"), acts);
        });
        ui.horizontal(|ui| match d.and_then(|d| d.relax.as_ref()).filter(|r| r.converged.is_none()) {
            Some(r) => {
                let lines = super::relax_lines(r);
                sub(ui, line(&["relaxing · step ", &r.steps.to_string(), " · max|F| ", &format!("{:.1e}", r.max_force)], SUB, theme::ACCENT_INK)).on_hover_text(lines.join("\n"));
            }
            None => {
                kbd(ui, "Space");
                sub(ui, "play / pause");
            }
        });
    });
    group(ui, st, "tb-speed", "Speed · steps per tick", Some(SPEED_TIP), w.speed, |ui, st| {
        row(ui, |ui| {
            let mut speed = st.speed;
            ui.spacing_mut().slider_width = 80.0;
            let slid = ui.add(egui::Slider::new(&mut speed, 1..=MAX_SPEED).logarithmic(true).show_value(false)).on_hover_text(SPEED_TIP).changed();
            let typed = field(ui, w.speed_field, |ui| probed("speed", ui.add(egui::DragValue::new(&mut speed).range(1..=MAX_SPEED))))
                .on_hover_text(SPEED_TIP)
                .changed();
            if slid | typed {
                acts.push(Action::SetSpeed(speed));
            }
            // right-aligned in its five-character monospace slot
            let rate = input.measured_fs_per_s.map_or("—".to_string(), fs_per_s);
            fixed(ui, w.rate, 30.0, Layout::left_to_right(Align::Center), |ui| {
                ui.label(RichText::new(format!("{rate:>5}")).monospace().strong().size(15.0)).on_hover_text(RATE_TIP)
            });
            ui.label(RichText::new("fs/s").color(theme::INK_DIM)).on_hover_text("simulated fs per wall second over the last 2 s of play");
        });
        ui.horizontal(|ui| match d {
            Some(d) => {
                let (n, rate) = watchable_speed(st.dt_fs, d.ms_per_step);
                let text = line(&["watchable (≈ ", &fs_per_s(rate), " fs/s)"], SUB, theme::ACCENT_INK);
                if probed("watchable", ui.add(Button::new(text).frame(false)))
                    .on_hover_text("steps per tick for one ~10 fs H–X vibration in ≈ 0.5 s of wall time, at the current time step and measured cost per step")
                    .clicked()
                {
                    acts.push(Action::SetSpeed(n));
                }
                sub(ui, line(&["· tick ", &fs_per_s(d.wall_ratio), " fs/s"], SUB, theme::INK_DIM)).on_hover_text(RATE_TIP);
            }
            None => {
                sub(ui, "no solution yet");
            }
        });
    });
    group(ui, st, "tb-dt", "dt", None, w.dt, |ui, st| {
        row(ui, |ui| {
            let r = field(ui, w.dt_field, |ui| dt_field(ui, st, acts));
            probe("dt", &r)
        });
        match d.and_then(|d| d.dt_warning.as_ref()) {
            Some(w) => sub(ui, "under-resolved").on_hover_text(w.as_str()),
            None => sub(ui, format!("amber > {DT_WARN_FS}")),
        };
    });
    group(ui, st, "tb-tool", "Tool", None, w.tool, |ui, st| {
        row(ui, |ui| tool_segments(ui, st, false, acts));
        if st.mode == ToolMode::Select {
            // the pull the wheel sets while dragging (Physics ▸ Drag pull)
            sub(ui, line(&["drag + wheel: pull ", &super::sig3(st.drag_force), " Ha/Bohr"], SUB, theme::INK_DIM))
                .on_hover_text(format!("{} now. {}", super::pull_label(st.drag_force), super::menus::DRAG_TIP));
        } else {
            sub(ui, "right-drag pans · wheel zooms");
        }
    });
    group(ui, st, "tb-elements", "Place element · Z 1–10", None, w.elements, |ui, st| {
        row(ui, |ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            for z in 1..=10u8 {
                if probed(&format!("element-{z}"), element_chip(ui, z, st.element == z, 21.0)).clicked() {
                    acts.push(Action::Element(z));
                }
            }
        });
        ui.horizontal(|ui| {
            ui.spacing_mut().button_padding.y = 0.0;
            super::ptable::opener(ui, st, RichText::new(PTABLE_OPENER).size(SUB).color(theme::ACCENT_INK), Vec2::ZERO, acts);
            let a = st.isotope.or_else(|| units::most_abundant_a(st.element));
            let iso = a.map_or_else(|| elements::symbol(st.element).to_string(), |a| units::isotopes::isotope_label(st.element, a));
            sub(ui, format!("{iso} (Z = {})", st.element)).on_hover_text("the element and isotope the Place tool drops; a pick switches to Place");
        });
    });
    let sel = input.selection();
    let label = match sel {
        Some((i, n)) => format!("Selection · {} #{i}", elements::symbol(n.z)),
        None => "Selection".into(),
    };
    group(ui, st, "tb-selection", &label, None, w.selection, |ui, _| {
        row(ui, |ui| selection_buttons(ui, input, acts));
        match sel {
            Some((_, n)) => sub(
                ui,
                RichText::new(format!("({:.2}, {:.2}, {:.2}) Bohr", n.pos[0], n.pos[1], n.pos[2])).monospace().size(SUB).color(theme::INK_DIM),
            ),
            None => sub(ui, NO_SELECTION),
        };
    });
    let solver = Widths::last(ui, w.solver, w.solver_row);
    group_body(ui, st, "tb-solver", "Solver", None, solver, |ui, st| {
        row(ui, |ui| solver_segments(ui, st, false, acts));
        ui.horizontal(|ui| solver_sub(ui, st, input));
    });
}

fn compact_bar(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    let w = Widths::measure(ui);
    let sampling = input.sampling();
    let r = slot(ui, w.c_time, |ui| time_buttons(ui, input, sampling, true, acts));
    mark(ui, st, "tb-time", r.response.rect, w.c_time);
    ui.separator();
    let r = slot(ui, w.c_speed, |ui| {
        let mut speed = st.speed;
        if field(ui, w.c_speed_field, |ui| ui.add(egui::DragValue::new(&mut speed).range(1..=MAX_SPEED).prefix("×")))
            .on_hover_text(SPEED_TIP)
            .changed()
        {
            acts.push(Action::SetSpeed(speed));
        }
        let rate = input.measured_fs_per_s.map_or("—".to_string(), fs_per_s);
        let tick = input.snapshot.map_or(String::new(), |s| format!(" · tick {} fs/s", fs_per_s(s.dynamics.wall_ratio)));
        fixed(ui, w.c_rate, ui.available_height(), Layout::left_to_right(Align::Center), |ui| {
            ui.label(RichText::new(format!("{rate:>5} fs/s")).monospace().color(theme::INK_DIM))
                .on_hover_text(format!("{RATE_TIP}{tick}"))
        });
        // the watchable preset, compact: one ~10 fs vibration in ≈ 0.5 s
        if let Some(d) = input.snapshot.map(|s| &s.dynamics) {
            let (n, rate) = watchable_speed(st.dt_fs, d.ms_per_step);
            if ui
                .add(Button::new(RichText::new("watch").size(SUB).color(theme::ACCENT_INK)).frame(false))
                .on_hover_text(format!("watchable (≈ {} fs/s): steps per tick for one ~10 fs H–X vibration in ≈ 0.5 s of wall time, at the current time step and measured cost per step", fs_per_s(rate)))
                .clicked()
            {
                acts.push(Action::SetSpeed(n));
            }
        }
    });
    mark(ui, st, "tb-speed", r.response.rect, w.c_speed);
    let r = slot(ui, w.c_dt, |ui| field(ui, w.dt_field, |ui| dt_field(ui, st, acts)));
    mark(ui, st, "tb-dt", r.response.rect, w.c_dt);
    ui.separator();
    let r = slot(ui, w.c_tool, |ui| tool_segments(ui, st, true, acts));
    mark(ui, st, "tb-tool", r.response.rect, w.c_tool);
    let r = slot(ui, w.c_elements, |ui| element_flyout(ui, st, w.c_elements, acts));
    mark(ui, st, "tb-elements", r.response.rect, w.c_elements);
    ui.separator();
    let r = slot(ui, w.c_selection, |ui| {
        selection_buttons(ui, input, acts);
        if input.selection().is_none() {
            ui.label(RichText::new("no selection").color(theme::INK_FAINT));
        }
    });
    mark(ui, st, "tb-selection", r.response.rect, w.c_selection);
    ui.separator();
    let solver = Widths::last(ui, w.c_solver, w.c_solver_row);
    let r = slot(ui, solver, |ui| {
        solver_segments(ui, st, true, acts);
        // the light and a few letters at least, else (a very narrow window) no line at all; the
        // top bar's status chip still names the source
        if solver >= w.c_solver_row + MIN_SUB {
            solver_sub(ui, st, input);
        }
    });
    mark(ui, st, "tb-solver", r.response.rect, solver);
}

/// The fs/s readouts: three significant figures with an SI suffix, at most five characters
/// ("0.25", "15.6", "126", "1.24k", "79.4k", "999k", "1.00M"), so a `{:>5}` monospace slot
/// right-aligns every value in the same width ("126" and "79 365.08" used to differ by 5 digits).
pub(crate) fn fs_per_s(r: f64) -> String {
    if !(r.is_finite() && r >= 0.0) {
        return "—".into();
    }
    for (scale, suffix) in [(1.0, ""), (1e3, "k"), (1e6, "M"), (1e9, "G"), (1e12, "T")] {
        let v = r / scale;
        if v < 999.5 {
            let d = if v < 9.995 {
                2
            } else if v < 99.95 {
                1
            } else {
                0
            };
            return format!("{v:.d$}{suffix}");
        }
    }
    ">1P".into()
}

fn time_buttons(ui: &mut Ui, input: &PanelInput<'_>, sampling: bool, compact: bool, acts: &mut Vec<Action>) {
    // a relax is a tool mode like a pass: Play and Step wait for it (the driver refuses them)
    let relaxing = input.relaxing();
    let sampling = sampling || relaxing;
    if icons::button(ui, Icon::Rewind, 28.0, false, true).on_hover_text(REWIND_TIP).clicked() {
        acts.push(Action::Rewind);
    }
    let (icon, label) = if input.playing { (Icon::Pause, "Pause") } else { (Icon::Play, "Play") };
    // estimated with no model: nothing to play on (Pause stays available while playing)
    let no_model = !input.playing && input.snapshot.is_some_and(|s| s.backend == crate::sim::Backend::Estimated) && !input.model().is_some_and(|m| m.fitted) && !input.sampling();
    let play_ok = !sampling && !no_model;
    let r = if compact {
        let r = ui.add_enabled(
            play_ok,
            Button::new("").min_size(Vec2::new(30.0, 28.0)).fill(theme::ACCENT).stroke(Stroke::new(1.0, theme::ACCENT)),
        );
        let c = egui::Color32::from_rgb(0x17, 0x11, 0x2a);
        icons::paint(ui.painter(), egui::Rect::from_center_size(r.rect.center(), Vec2::splat(15.0)), icon, c);
        r.on_hover_text(format!("{label} (Space)"))
    } else {
        icons::text_button(ui, icon, label, 84.0, true, play_ok).on_hover_text("Space")
    };
    let busy_tip = if no_model && !sampling {
        "the estimated solver has no model to play on: load one (File ▸ Open) or press Start sampling in the Model tab"
    } else if relaxing {
        RELAXING_TIP
    } else {
        SAMPLING_TIP
    };
    if probed("play", r).on_disabled_hover_text(busy_tip).clicked() {
        acts.push(Action::TogglePlay);
    }
    if icons::button(ui, Icon::Step, 28.0, false, !sampling)
        .on_hover_text(STEP_TIP)
        .on_disabled_hover_text(busy_tip)
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

/// Display name of Z (`names::element_name`, the atom card's).
fn element_name(z: u8) -> &'static str {
    crate::names::element_name(z)
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

/// Compact: one chip for the element (`width` wide whatever the symbol), opening the periodic
/// table (`ptable`).
fn element_flyout(ui: &mut Ui, st: &mut UiState, width: f32, acts: &mut Vec<Action>) {
    let text = RichText::new(format!("{} ▾", elements::symbol(st.element))).monospace().strong();
    super::ptable::opener(ui, st, text, Vec2::new(width, 26.0), acts);
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
    if probed("delete", ui.add_enabled(sel.is_some(), Button::new(RichText::new("Delete").color(theme::DANGER_INK))))
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
            if probed("solver-cpu", ui.add(Button::selectable(st.backend == Backend::Cpu, "CPU"))).on_hover_text(CPU_TIP).clicked() {
                acts.push(Action::Backend(Backend::Cpu));
            }
            if probed("solver-gpu", ui.add_enabled(st.gpu_available, Button::selectable(st.backend == Backend::Gpu, "GPU")))
                .on_hover_text(GPU_TIP)
                .on_disabled_hover_text("no GPU adapter")
                .clicked()
            {
                acts.push(Action::Backend(Backend::Gpu));
            }
            let est = if compact { "Est." } else { "Estimated" };
            if probed("solver-estimated", ui.add(Button::selectable(st.backend == Backend::Estimated, est)))
                .on_hover_text(estimated_tip(st.gpu_available))
                .clicked()
            {
                acts.push(Action::Backend(Backend::Estimated));
            }
        });
}

/// The solver's sub-line: the trust light, who answered this step, trust, fallbacks, epoch
/// (estimated), or the Kohn-Sham engine. Every label elides at the slot's edge (full text on
/// hover); the fallback share is padded to three digits and the pass's count to its total's.
fn solver_sub(ui: &mut Ui, st: &UiState, input: &PanelInput<'_>) {
    let Some(s) = input.snapshot else {
        theme::light(ui, theme::INK_FAINT);
        sub(ui, "waiting for the first solve");
        return;
    };
    let d = &s.dynamics;
    if s.backend != Backend::Estimated {
        theme::light(ui, SPARK_E);
        sub(ui, RichText::new(format!("DFT · {}", s.engine.label())).size(SUB).color(SPARK_E));
        return;
    }
    let m = input.model();
    let t = trust(m, input.scene, st);
    if let Some(p) = d.sampling.as_ref().filter(|_| input.sampling()) {
        theme::light(ui, theme::INK_FAINT);
        let total = p.total.to_string();
        let count = format!("{:>w$}/{total}", p.done, w = total.len());
        sub(ui, line(&[&format!("building the model · {} ", crate::ui::activity::words::pass_stage(p)), &count], SUB, theme::INK_DIM));
        return;
    }
    let r = theme::light(ui, t.colour);
    if let Some(tip) = t.tip {
        r.on_hover_text(match &t.detail {
            Some(d) => format!("{tip}

the fit's own words: {d}"),
            None => tip.to_string(),
        });
    }
    if t.kind == TrustKind::NoModel {
        sub(ui, t.kind.short());
        return;
    }
    let src = d.source;
    sub(ui, RichText::new(source_label(src)).size(SUB).color(source_colour(src))).on_hover_text(super::SOURCE_TIP);
    let short = t.kind.short();
    let epoch = m.map_or(0, |m| m.epoch);
    let pct = format!("{:>3.0}", 100.0 * d.fallback_fraction_100);
    sub(ui, line(&[&format!("· {short} · "), &pct, " % fallback · epoch ", &epoch.to_string()], SUB, theme::INK_DIM)).on_hover_text(format!(
        "trusted: {}; fallbacks over the last 100 steps: {:.0} %; model epoch {} (Model tab)",
        t.text,
        100.0 * d.fallback_fraction_100,
        epoch
    ));
}
