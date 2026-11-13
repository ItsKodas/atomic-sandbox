//! The Readouts panel (restructured 2026-11-14 from the user's approved mockup; the old Graph
//! panel merged into it). Top to bottom: the three tiles (T, drift, rate with the step), the
//! **chart card** — two stacked plots on one time axis, T in the chosen temperature unit with its
//! real minimum and maximum, and ΔE = E total minus its value at the window's first point on a real
//! axis in µHa (mHa above 1000 µHa) with a dashed zero line and labelled ticks, coloured per step by
//! who answered it ([`delta_axis`]: the old Graph scaled each curve to its own range, which drew
//! 1e-6 Ha of noise as a full swing) —, the energies (KE, E_KS, E total, E conserved; Ha / eV by the
//! header's switch), one time-and-cost line with the health checks as dots (a check that is not ok
//! also prints its text), and the folds Work done, Energy breakdown and solver and Compare with
//! reference (PySCF) ([`Sections`], `UiState::readouts_sections`, the profile's `readouts_sections`).
//! Display only but the reference comparison's own actions (`super::compare`).
use super::super::pseudo;
use super::super::{
    box_warning, drift_colour, energy_cells, probe_rect, search, source_colour, source_label, theme, PanelInput, UiState, AMBER, BOX_MARGIN_WARN, GREEN, RED, SPARK_T,
    SPARK_WINDOW,
};
use crate::sim::Backend;
use egui::{pos2, vec2, Align2, Color32, FontId, Rect, RichText, Sense, Stroke, Ui};
use sim::dynamics::{DRIFT_MIN_POINTS, DT_WARN_FS, F_DRAG_HI, F_DRAG_LO, F_DRAG_MAX};
use sim::Source;

/// One of the panel's folds (`UiState::readouts_sections`). All closed by default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Section {
    Work,
    Breakdown,
    Compare,
}

impl Section {
    pub const ALL: [Section; 3] = [Section::Work, Section::Breakdown, Section::Compare];

    /// Its word in the profile's `readouts_sections` line (its header's probe: `readouts-fold-<id>`).
    pub fn id(self) -> &'static str {
        match self {
            Section::Work => "work",
            Section::Breakdown => "breakdown",
            Section::Compare => "compare",
        }
    }

    /// The search anchor its contents mark: a Ctrl K reveal unfolds it.
    pub fn anchor(self) -> &'static str {
        match self {
            Section::Work => "d-work",
            Section::Breakdown => "d-breakdown",
            Section::Compare => "d-compare",
        }
    }
}

/// Which folds are open (`UiState::readouts_sections`): none by default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sections([bool; 3]);

impl Sections {
    pub fn is_open(&self, s: Section) -> bool {
        self.0[s as usize]
    }

    pub fn set(&mut self, s: Section, open: bool) {
        self.0[s as usize] = open;
    }

    /// Every fold open (or none).
    #[cfg(test)]
    pub fn all(open: bool) -> Sections {
        Sections([open; 3])
    }

    /// The profile's value: the open folds' ids, comma separated.
    pub fn encode(&self) -> String {
        Section::ALL.into_iter().filter(|&s| self.is_open(s)).map(Section::id).collect::<Vec<_>>().join(",")
    }

    /// The profile's value read back (unknown words dropped).
    pub fn parse(v: &str) -> Sections {
        let mut s = Sections::default();
        for w in v.split(',').map(str::trim) {
            if let Some(x) = Section::ALL.into_iter().find(|x| x.id().eq_ignore_ascii_case(w)) {
                s.set(x, true);
            }
        }
        s
    }
}

fn tile(ui: &mut Ui, key: &str, value: RichText, unit: &str, tip: &str) {
    egui::Frame::new()
        .fill(theme::PANEL_2)
        .stroke(egui::Stroke::new(1.0, theme::LINE_SOFT))
        .corner_radius(7)
        .inner_margin(egui::Margin::symmetric(8, 5))
        .show(ui, |ui| {
            ui.set_width((ui.available_width()).max(70.0));
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 1.0;
                ui.label(RichText::new(key.to_uppercase()).size(10.0).color(theme::INK_FAINT));
                ui.label(value.monospace().strong().size(14.0));
                ui.add(egui::Label::new(RichText::new(unit).size(10.5).color(theme::INK_DIM)).truncate());
            });
        })
        .response
        .on_hover_text(tip);
}

/// Below this width (points) the energies' Ha and eV stack in one cell (side by side they need
/// about 400 pt with the row names).
const STACK_BELOW: f32 = 400.0;

/// The chart card's explanation (the old Graph tab's note), its ⓘ.
pub const CHART_TIP: &str = "E total = KE + E_KS: it moves when tools or the thermostat act. The drift readout uses E conserved. The lower plot is ΔE, E total minus its value at the first point shown, on a real axis (µHa, or mHa past 1000 µHa) — noise of 1e-6 Ha stays a ripple — coloured by who answered each step; the upper is T in the chosen unit, its real minimum and maximum labelled";

/// The ΔE plot's axis: the unit (`per_ha` units per Hartree: 1e6 µHa, 1e3 mHa), its range and its
/// labelled ticks, in that unit.
#[derive(Clone, Debug, PartialEq)]
pub struct DeltaAxis {
    pub per_ha: f64,
    pub unit: &'static str,
    pub lo: f64,
    pub hi: f64,
    pub ticks: Vec<f64>,
}

/// The ΔE axis never spans less than ±this (µHa): a flat E total stays flat (a 1e-7 Ha wiggle is
/// a fraction of a point at any height the plot is drawn), not blown up to the plot's height.
pub const DELTA_MIN_HALF_UHA: f64 = 10.0;

/// The axis for the ΔE values `deltas` (Ha): µHa, or mHa when they span more than 1000 µHa; the
/// range holds zero and at least ±[`DELTA_MIN_HALF_UHA`], padded 8 %; ticks at a 1-2-5 step.
pub fn delta_axis(deltas: &[f64]) -> DeltaAxis {
    let (mn, mx) = deltas.iter().fold((0.0f64, 0.0f64), |(lo, hi), &x| (lo.min(x), hi.max(x)));
    let (per_ha, unit) = if (mx - mn) * 1e6 > 1000.0 { (1e3, "mHa") } else { (1e6, "µHa") };
    let floor = DELTA_MIN_HALF_UHA * per_ha / 1e6;
    let (lo, hi) = ((mn * per_ha).min(-floor), (mx * per_ha).max(floor));
    let pad = 0.08 * (hi - lo);
    let (lo, hi) = (lo - pad, hi + pad);
    let raw = (hi - lo) / 4.0;
    let mag = 10f64.powf(raw.log10().floor());
    let step = [1.0, 2.0, 5.0, 10.0].into_iter().map(|k| k * mag).find(|s| *s >= raw).unwrap_or(10.0 * mag);
    let mut ticks = Vec::new();
    let mut t = (lo / step).ceil() * step;
    while t <= hi + 1e-12 {
        ticks.push(if t.abs() < step * 1e-9 { 0.0 } else { t });
        t += step;
    }
    DeltaAxis { per_ha, unit, lo, hi, ticks }
}

impl DeltaAxis {
    /// The screen y of `v` (in the axis's unit) in `plot`.
    pub fn y(&self, v: f64, plot: Rect) -> f32 {
        plot.bottom() - ((v - self.lo) / (self.hi - self.lo)) as f32 * plot.height()
    }
}

/// The plots' heights for a panel `view_h` points tall: they shrink with a short panel but keep
/// these minimums (T 48, ΔE 70).
pub fn plot_heights(view_h: f32) -> (f32, f32) {
    ((view_h * 0.12).clamp(48.0, 90.0), (view_h * 0.16).clamp(70.0, 130.0))
}

pub fn show(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<super::super::Action>) {
    // the visible part of the panel (the tests' width checks)
    probe_rect("readouts-panel", ui.clip_rect());
    // a Ctrl K reveal unfolds the fold holding its anchor (it stays open)
    if let Some(a) = st.search.unrevealed() {
        if let Some(s) = Section::ALL.into_iter().find(|s| s.anchor() == a) {
            st.readouts_sections.set(s, true);
        }
    }
    let Some(s) = input.snapshot else {
        ui.weak("no solution yet");
        return;
    };
    let d = &s.dynamics;
    let estimated = s.backend == Backend::Estimated;
    let tu = st.temp_units;
    let t_shown = super::super::averaged_temperature_text(d, &st.temp_avg, 1, tu);
    super::group(ui, st, "d-tiles", |ui, _| {
        ui.columns(3, |c| {
            tile(
                &mut c[0],
                "T",
                RichText::new(t_shown.clone()),
                "1 s average",
                "kinetic temperature by equipartition over 3N − 3 degrees of freedom (COM motion removed; rotation counted; a single nucleus: 3, lab frame), averaged over the last second of steps — a few nuclei's instantaneous T swings with every vibration",
            );
            let (v, unit) = match d.drift_per_ps {
                Some(x) => {
                    let col = drift_colour(x).unwrap_or(theme::INK);
                    let word = match drift_colour(x) {
                        Some(c) if c == AMBER => "Ha/ps · amber",
                        Some(_) => "Ha/ps · red",
                        None => "Ha/ps · ok",
                    };
                    (RichText::new(format!("{x:+.2e}")).color(col), word)
                }
                None => (RichText::new("—").color(Color32::GRAY), "Ha/ps"),
            };
            tile(
                &mut c[1],
                "drift",
                v,
                unit,
                &format!("least-squares slope of E conserved over the last ≤ 2000 steps (shown after {DRIFT_MIN_POINTS}); amber above 1e-4, red above 1e-3 Ha/ps"),
            );
            let rate = input.measured_fs_per_s.map_or("—".to_string(), |r| format!("{r:.1}"));
            tile(
                &mut c[2],
                "rate fs/s",
                RichText::new(rate),
                &format!("step {}", d.step),
                "simulated femtoseconds per wall second over the last 2 s of play (what you see); below it, the step count",
            );
        });
    });
    super::group(ui, st, "d-graph", |ui, st| chart_card(ui, st, s));
    super::group(ui, st, "d-energies", |ui, st| {
        ui.horizontal(|ui| {
            theme::section(ui, "Energies");
            theme::info(ui, "E total = KE + E_KS: conserved in NVE without tools. E conserved = E total + drag spring + field potential + container walls − the work of thermostat, pins, drag damping, COM correction and moving walls (Work done, below).");
        });
        let units = st.energy_units;
        let stacked = ui.available_width() < STACK_BELOW;
        super::wide_grid(ui, "energies", 3).striped(true).show(ui, |ui| {
            let mut row = |name: &str, x: f64, strong: bool, tip: &str| {
                let n = if strong { RichText::new(name).strong() } else { RichText::new(name) };
                ui.label(n).on_hover_text(tip);
                energy_cells(ui, x, units, strong, stacked);
                ui.end_row();
            };
            row("KE (nuclei)", d.kinetic, false, "Σ ½ m v² of the nuclei (lab frame)");
            let e_ks_label = pseudo::energy_label(&d.eval.pseudised);
            let e_ks_tip = if d.eval.pseudised.is_empty() {
                "Kohn-Sham energy at the current positions (electrons + nuclear repulsion)".to_string()
            } else {
                format!("Kohn-Sham energy at the current positions (valence electrons + the ions' repulsion). {}", pseudo::VALENCE_NOTE)
            };
            row(e_ks_label, d.energy_ks, false, &e_ks_tip);
            row("E total", d.total, true, "KE + E_KS: conserved in NVE without tools");
            row(
                "E conserved",
                d.energy_conserved,
                false,
                "E total + drag spring + field potential + container walls − the work of the thermostat, pins, drag damping, COM correction and moving walls: the invariant the drift is fitted on",
            );
        });
        // M7 (Task 6 review A.6): a partly filled degenerate shell is shared equally, the spherical-atom ensemble
        for line in pseudo::occupation_lines(&d.eval.fractional) {
            theme::note(ui, line);
        }
    });
    status_line(ui, st, input, s, acts);
    // the folds
    let model_work = if estimated { d.work_refit + d.work_mismatch } else { 0.0 };
    let tools = d.total - d.energy_conserved - model_work;
    let summary = if estimated {
        format!("walls {} · refit {} · mismatch {}", short_ha(d.walls_work), short_ha(d.work_refit), short_ha(d.work_mismatch))
    } else if d.walls.is_some() || d.walls_work != 0.0 {
        format!("tools {} · walls {}", short_ha(tools), short_ha(d.walls_work))
    } else {
        format!("tools {}", short_ha(tools))
    };
    if fold_header(ui, st, Section::Work, "Work done", summary) {
        super::group(ui, st, "d-work", |ui, st| ui.indent("readouts-work", |ui| work_rows(ui, st, s, tools)));
    }
    if fold_header(ui, st, Section::Breakdown, "Energy breakdown and solver", format!("{} SCF · {} ms", d.eval.scf_steps, d.eval.solve_ms)) {
        super::group(ui, st, "d-breakdown", |ui, st| ui.indent("readouts-breakdown", |ui| breakdown(ui, st, s)));
    }
    if std::mem::take(&mut st.compare.open_request) {
        st.readouts_sections.set(Section::Compare, true);
    }
    let (h, _) = theme::fold_header(ui, "Compare with reference (PySCF)", st.readouts_sections.is_open(Section::Compare), None, None, false);
    probe_rect("readouts-fold-compare", h.rect);
    if h.clicked() {
        let open = st.readouts_sections.is_open(Section::Compare);
        st.readouts_sections.set(Section::Compare, !open);
    }
    search::mark(ui, st, "d-compare", h.rect);
    if st.readouts_sections.is_open(Section::Compare) {
        st.compare.ensure_probe();
        ui.indent("readouts-compare", |ui| super::compare::body(ui, st, input, acts));
    }
}

/// A Hartree value as a fold's summary reads it: "0", else six places with a true minus sign.
fn short_ha(x: f64) -> String {
    if x == 0.0 { "0".into() } else { format!("{x:+.6}").replace('-', "−") }
}

/// A fold's header (`theme::fold_header`, probed `readouts-fold-<id>`); returns whether it is open.
fn fold_header(ui: &mut Ui, st: &mut UiState, s: Section, title: &str, summary: String) -> bool {
    let (r, _) = theme::fold_header(ui, title, st.readouts_sections.is_open(s), Some((summary, theme::INK_FAINT)), None, false);
    probe_rect(&format!("readouts-fold-{}", s.id()), r.rect);
    if r.clicked() {
        let open = st.readouts_sections.is_open(s);
        st.readouts_sections.set(s, !open);
    }
    st.readouts_sections.is_open(s)
}

/// The chart card: its header (the ⓘ, the window), the T plot over the ΔE plot on one time axis,
/// the step legend and the current E total.
fn chart_card(ui: &mut Ui, st: &mut UiState, s: &crate::sim::Snapshot) {
    let d = &s.dynamics;
    let r = egui::Frame::new().fill(theme::PANEL_2).stroke(Stroke::new(1.0, theme::LINE)).corner_radius(8).inner_margin(egui::Margin::symmetric(10, 8)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            ui.label(theme::caps("E total and T"));
            theme::info(ui, CHART_TIP);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let r = ui.scope(|ui| {
                    if let Some(a) = theme::segmented(ui, st.graph_all, &[(true, "all ≤ 2000", "every point kept (at most 2000)"), (false, "last 400", "the newest 400 points")]) {
                        st.graph_all = a;
                    }
                });
                probe_rect("chart-window", r.response.rect);
            });
        });
        let history = &d.history;
        if history.len() < 2 {
            theme::note(ui, "E total and T appear here once time runs");
            return;
        }
        let window = if st.graph_all { usize::MAX } else { SPARK_WINDOW };
        let start = history.len().saturating_sub(window);
        let pts = &history[start..];
        let src = |i: usize| d.history_source.get(start + i).copied().unwrap_or(Source::Dft);
        let guess = |i: usize| d.history_guess.get(start + i).copied().unwrap_or(false);
        let any_guess = (0..pts.len()).any(guess);
        let (t0, t1) = (pts[0].0, pts[pts.len() - 1].0);
        let (t_h, e_h) = plot_heights(ui.clip_rect().height());
        let gutter = 46.0;
        let w = ui.available_width();
        let x_of = |plot: Rect, t: f64| plot.left() + if t1 > t0 { ((t - t0) / (t1 - t0)) as f32 } else { 0.0 } * plot.width();
        let small = FontId::monospace(9.0);
        // T, in the chosen unit, its real minimum and maximum
        let (rect, resp) = ui.allocate_exact_size(vec2(w, t_h), Sense::hover());
        let plot = Rect::from_min_max(rect.min + vec2(gutter, 4.0), rect.max - vec2(2.0, 4.0));
        let p = ui.painter_at(rect.expand(1.0));
        p.rect_filled(plot, 3.0, Color32::from_black_alpha(70));
        let (tlo, thi) = pts.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), q| (lo.min(q.2), hi.max(q.2)));
        let (alo, ahi) = if thi - tlo < 1e-9 { (tlo - 1.0, thi + 1.0) } else { (tlo, thi) };
        let ty = |t: f64| plot.bottom() - 3.0 - ((t - alo) / (ahi - alo)) as f32 * (plot.height() - 6.0);
        let line: Vec<egui::Pos2> = pts.iter().map(|q| pos2(x_of(plot, q.0), ty(q.2))).collect();
        p.add(egui::Shape::line(line, Stroke::new(1.4, SPARK_T)));
        p.text(pos2(plot.left() - 4.0, ty(thi)), Align2::RIGHT_CENTER, st.temp_units.show(thi, 0), small.clone(), SPARK_T);
        p.text(pos2(plot.left() - 4.0, ty(tlo)), Align2::RIGHT_CENTER, st.temp_units.show(tlo, 0), small.clone(), SPARK_T);
        p.text(plot.left_top() + vec2(4.0, 2.0), Align2::LEFT_TOP, "T", FontId::proportional(10.0), theme::INK_DIM);
        probe_rect("chart-t", plot);
        resp.on_hover_text(format!("T over the window: {} … {} (its real range, in the chosen unit)", st.temp_units.show(tlo, 1), st.temp_units.show(thi, 1)));
        // ΔE on a real axis
        let deltas: Vec<f64> = pts.iter().map(|q| q.1 - pts[0].1).collect();
        let axis = delta_axis(&deltas);
        let (rect, resp) = ui.allocate_exact_size(vec2(w, e_h + 14.0), Sense::hover());
        let plot = Rect::from_min_max(rect.min + vec2(gutter, 4.0), pos2(rect.right() - 2.0, rect.top() + e_h));
        let p = ui.painter_at(rect.expand(1.0));
        p.rect_filled(plot, 3.0, Color32::from_black_alpha(70));
        for &tk in &axis.ticks {
            let y = axis.y(tk, plot);
            let label = if tk == 0.0 { "0".to_string() } else { format!("{tk:+}").replace('-', "−") };
            p.text(pos2(plot.left() - 4.0, y), Align2::RIGHT_CENTER, label, small.clone(), theme::INK_FAINT);
            if tk != 0.0 {
                p.line_segment([pos2(plot.left(), y), pos2(plot.right(), y)], Stroke::new(1.0, Color32::from_white_alpha(10)));
            }
        }
        let zero = axis.y(0.0, plot);
        p.extend(egui::Shape::dashed_line(&[pos2(plot.left(), zero), pos2(plot.right(), zero)], Stroke::new(1.0, theme::INK_FAINT), 4.0, 3.0));
        probe_rect("chart-zero", Rect::from_min_max(pos2(plot.left(), zero), pos2(plot.right(), zero)));
        p.text(plot.left_top() + vec2(4.0, 2.0), Align2::LEFT_TOP, format!("ΔE {}", axis.unit), FontId::proportional(10.0), theme::INK_DIM);
        let at = |i: usize| pos2(x_of(plot, pts[i].0), axis.y(deltas[i] * axis.per_ha, plot));
        for i in 1..pts.len() {
            let c = if guess(i) { RED } else { source_colour(src(i)) };
            p.line_segment([at(i - 1), at(i)], Stroke::new(1.5, c));
        }
        for i in 0..pts.len() {
            if src(i) == Source::Fallback {
                p.circle_filled(at(i), 2.2, AMBER);
            }
        }
        probe_rect("chart-delta", plot);
        // the shared time axis, its ends labelled
        let ty0 = plot.bottom() + 2.0;
        p.text(pos2(plot.left(), ty0), Align2::LEFT_TOP, format!("{:.1} fs", units::au_time_to_fs(t0)), small.clone(), theme::INK_FAINT);
        p.text(pos2(plot.right(), ty0), Align2::RIGHT_TOP, format!("{:.1} fs", units::au_time_to_fs(t1)), small.clone(), theme::INK_FAINT);
        resp.on_hover_text(format!("ΔE = E total − {:+.6} Ha (its value at the first point shown), in {}; each step coloured by who answered it", pts[0].1, axis.unit));
        // the legend, and the current E total right-aligned
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            let mut key = |c: Color32, word: &str| {
                let (sq, _) = ui.allocate_exact_size(vec2(8.0, 8.0), Sense::hover());
                ui.painter().rect_filled(sq, 1.0, c);
                ui.label(RichText::new(word).size(10.5).color(theme::INK_DIM));
            };
            for src in [Source::Surrogate, Source::Fallback, Source::Dft, Source::Reference] {
                key(source_colour(src), source_label(src));
            }
            if any_guess {
                key(RED, "draft guess");
            }
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
            // in the header's unit (Ha for "both")
            let (text, other) = match st.energy_units {
                super::super::EnergyUnits::Ev => (format!("E total {:+.3} eV", units::hartree_to_ev(d.total)), format!("{:+.6} Ha", d.total)),
                _ => (format!("E total {:+.6} Ha", d.total), format!("{:+.4} eV", units::hartree_to_ev(d.total))),
            };
            let r = ui.label(RichText::new(text).monospace().size(11.0)).on_hover_text(other);
            probe_rect("chart-e-total", r.rect);
        });
    });
    probe_rect("readouts-chart", r.response.rect);
}

/// The time-and-cost line ("t 180.25 fs · 3.3 ms/step · model, no SCF"; its detail on hover) with
/// the health checks as dots on the right; a check that is not ok also prints its text under it,
/// then the newest dynamics notes.
fn status_line(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, s: &crate::sim::Snapshot, acts: &mut Vec<super::super::Action>) {
    let d = &s.dynamics;
    let per_step = match d.source {
        Source::Surrogate => "model, no SCF".to_string(),
        Source::Reference => "atom references, no SCF".to_string(),
        _ => format!("{} SCF iterations", d.eval.scf_steps),
    };
    let line = format!("t {:.2} fs · {:.1} ms/step · {per_step}", d.sim_time_fs(), d.ms_per_step);
    let measured = input.measured_fs_per_s.map_or("—".to_string(), |r| format!("{r:.1}"));
    let mut tip = format!(
        "t = {:.2} fs · step {}\nfs / s: {measured} (last 2 s) · {:.2} (this tick) — simulated femtoseconds per wall second\nlast tick: {} step{} · {:.3} ms/step",
        d.sim_time_fs(),
        d.step,
        d.wall_ratio,
        d.steps_last_tick,
        if d.steps_last_tick == 1 { "" } else { "s" },
        d.ms_per_step
    );
    tip += &match d.source {
        Source::Surrogate => format!("\nper step: model, no SCF · {:.3} ms (moving mean over the model's own steps; DFT steps here — fallbacks; audits in the step with the CPU teacher — average {:.0} ms)", d.ms_per_step_model, d.ms_per_step_dft),
        Source::Reference => format!("\nper step: answered from the atom references, no SCF. {}", super::super::REFERENCE_TIP),
        _ => {
            let cost = match d.eval.readbacks {
                Some(r) => format!("{} SCF iterations, {r} readbacks", d.eval.scf_steps),
                None => format!("{} SCF iterations", d.eval.scf_steps),
            };
            format!("\nper step: {cost} · {} ms", d.eval.solve_ms)
        }
    };
    if let Some(a) = input.model().and_then(|m| m.last_audit.as_ref()).filter(|a| a.asynchronous) {
        tip += &format!("\naudit lag: {} steps (answered steps between the last audit's dispatch and its verdict, Model tab)", a.unaudited_steps);
    }
    // the health checks: (colour, text); green is ok
    let mut checks: Vec<(Color32, String)> = Vec::new();
    match box_warning(&s.scene) {
        Some((c, w, _)) => checks.push((c, w)),
        None => checks.push((GREEN, format!("all nuclei ≥ {BOX_MARGIN_WARN} Bohr from the box face"))),
    }
    match &d.dt_warning {
        Some(w) => checks.push((AMBER, w.clone())),
        None => checks.push((GREEN, format!("dt {:.3} fs (the warning starts above {DT_WARN_FS} fs)", d.dt_fs()))),
    }
    for w in &d.eval.warnings {
        checks.push((AMBER, w.clone()));
    }
    super::group(ui, st, "d-time", |ui, _| {
        ui.horizontal(|ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                dots(ui, &checks);
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    let r = ui.add(egui::Label::new(RichText::new(&line).monospace().size(11.0).color(theme::INK_DIM)).truncate()).on_hover_text(&tip);
                    probe_rect("readouts-status", r.rect);
                });
            });
        });
    });
    super::group(ui, st, "d-health", |ui, st| {
        for (k, (c, text)) in checks.iter().enumerate().filter(|(_, (c, _))| *c != GREEN) {
            let r = ui.horizontal_wrapped(|ui| {
                theme::light(ui, *c);
                ui.label(RichText::new(text).size(11.5).color(*c));
            });
            probe_rect(&format!("readouts-warning-{k}"), r.response.rect);
        }
        // the estimated solver's box when it is short (`ui::estimated_box`): a line and its regrid
        crate::ui::estimated_box::show_note(ui, st, input, acts);
        for n in d.notes.iter().rev().take(3) {
            ui.add(egui::Label::new(RichText::new(n).size(11.0).color(theme::INK_FAINT)).wrap());
        }
        if d.notes.len() > 3 {
            theme::note(ui, "older notes are in the Log tab");
        }
    });
}

/// The Work done fold: the work of the tools, the walls, and (estimated solver) the model's own
/// bookings — the three add up to E total − E conserved.
fn work_rows(ui: &mut Ui, st: &UiState, s: &crate::sim::Snapshot, tools: f64) {
    let d = &s.dynamics;
    let estimated = s.backend == Backend::Estimated;
    let units = st.energy_units;
    let stacked = ui.available_width() < STACK_BELOW;
    super::wide_grid(ui, "work", 3).striped(true).show(ui, |ui| {
        let mut row = |name: &str, x: f64, tip: &str| {
            ui.label(name).on_hover_text(tip);
            energy_cells(ui, x, units, false, stacked);
            ui.end_row();
        };
        row(
            "work done by tools",
            tools,
            &format!(
                "the energy the drag, field, thermostat, pins, COM correction and container walls have put into the nuclei and electrons (net of what the spring, the field and the walls still hold). \
                 The drag spring pulls with at most the pull ({F_DRAG_MAX} Ha/Bohr by default; the wheel while dragging sets it, {F_DRAG_LO}–{F_DRAG_HI}): drag slowly and the molecule follows with little work; a fast drag mostly heats it (and every tug shows up here). \
                 With the estimated solver the model's own bookings are on the rows below; the three add up to E total − E conserved."
            ),
        );
        if d.walls.is_some() || d.walls_work != 0.0 {
            row(
                "   of which walls (piston)",
                d.walls_work,
                "the work of switching and moving the container walls (ΔU of the wall at the atoms' positions per move): positive when shrinking the box squeezes them",
            );
        }
        if estimated {
            row(
                "model refit",
                d.work_refit,
                "the jumps of the model's energy surface under the atoms (a refit, a failed audit's switch to DFT), booked so E conserved stays continuous: a model learning a new bonding situation can book a large one. Not work anybody did — the surface the energy is measured on moved.",
            );
            row(
                "model force mismatch",
                d.work_mismatch,
                "work of (F_DFT − F_model) on fallback steps: the atoms move on DFT forces while the energy is the model's, so the difference is booked like an external force",
            );
        }
    });
    if estimated {
        theme::note(ui, "The three work rows add up to E total − E conserved.");
    }
}

/// The latest evaluation's energy breakdown and solver diagnostics.
fn breakdown(ui: &mut Ui, st: &mut UiState, s: &crate::sim::Snapshot) {
    let d = &s.dynamics;
    let e = &d.eval.energy;
    let units = st.energy_units;
    let stacked = ui.available_width() < STACK_BELOW;
    super::wide_grid(ui, "energy", 3).striped(true).show(ui, |ui| {
        let mut row = |name: &str, x: f64| {
            ui.label(name);
            energy_cells(ui, x, units, false, stacked);
            ui.end_row();
        };
        row(pseudo::energy_label(&d.eval.pseudised), e.total);
        row("kinetic (electrons)", e.kinetic);
        row("external", e.external);
        row("Hartree", e.hartree);
        row("xc", e.xc);
        row("nuclear", e.nuclear);
    });
    let sigma = s.scene.sigma.unwrap_or(s.scene.grid_h);
    if d.eval.pseudised.is_empty() {
        ui.weak(format!("smeared nuclei (σ = {sigma} Bohr): absolute energies depend on σ and h"));
    } else {
        ui.weak(format!("{} — all-electron species are smeared nuclei (σ = {sigma} Bohr)", pseudo::VALENCE_NOTE));
    }
    if d.source == Source::Surrogate {
        ui.weak("surrogate step: only the total is the model's (it has no electrons); the parts above are 0");
    } else if d.source == Source::Reference {
        ui.weak("lone atoms, answered from the atom references: only the total is theirs (Σ of the isolated-atom reference energies, no SCF ran); the parts above and the SCF numbers below are 0");
    } else if d.source == Source::Fallback {
        ui.weak("fallback step: the DFT teacher's breakdown (with a trusted model E_KS above is the model's: total − teacher gap)");
    }
    if s.engine == Backend::Gpu {
        ui.weak("GPU (f32 grid): energies scatter by up to ~1e-5 Ha between solve paths");
    }
    super::wide_grid(ui, "scf", 2).striped(true).show(ui, |ui| {
        let mut row = |name: &str, v: String| {
            ui.label(name);
            ui.monospace(v);
            ui.end_row();
        };
        row("SCF iterations", d.eval.scf_steps.to_string());
        // the worst LOBPCG residual among the solves the converged state rests on (the
        // GPU's deliberately loosened early SCF steps excluded)
        row("eigen residual (final)", format!("{:.1e}", d.eval.max_eigen_residual));
        row("Poisson residual", format!("{:.1e}", d.eval.poisson_residual));
        row(
            "solve",
            match d.eval.readbacks {
                Some(r) => format!("{} ms, {r} readbacks ({})", d.eval.solve_ms, s.engine.label()),
                None => format!("{} ms ({})", d.eval.solve_ms, s.engine.label()),
            },
        );
        row("grid", format!("{}³ × {} Bohr", s.scene.grid_n, s.scene.grid_h));
        if let Some(max) = d.display.as_ref().map(|x| match x {
            sim::Display::Cpu { max, .. } | sim::Display::Gpu { max, .. } => *max,
        }) {
            row("ρ max", format!("{max:.3} e/Bohr³"));
        }
    });
    for n in &d.eval.notes {
        ui.weak(n);
    }
}

/// The health checks as dots, right to left (in a right-to-left layout), each with its full text
/// on hover (probed `readouts-dot-<k>`).
fn dots(ui: &mut Ui, checks: &[(Color32, String)]) {
    for (k, (c, text)) in checks.iter().enumerate().rev() {
        let r = theme::light(ui, *c).on_hover_text(text);
        probe_rect(&format!("readouts-dot-{k}"), r.rect);
    }
}
