//! The job list (activity spec §3.3.1, §7): one row per running or waiting piece of work of the
//! live worker, each with its progress and the one control that exists today. It is the one
//! component the precision milestone's job queue and the user-data model manager add rows to
//! (`JobKey`): nobody builds a second list.
use super::headline::{iteration_text, sample_text, steps_text, sweep_edge, sweep_phase};

use crate::ui::{theme, Action, PanelInput, AMBER, GREEN};
use egui::{Color32, RichText, Ui};

/// What a row is about. The precision milestone adds `Queued(JobId)`, user data its long
/// operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobKey {
    LiveSolve,
    Pass,
    Refit,
    Audit,
    /// A density refresh or an isolated-atom density on the background teacher (spec A8).
    Background,
    BondReferences,
    /// The display eigensolve of unoccupied orbitals after a DFT step (display only).
    Orbitals,
    Relax,
    PassOffer,
}

/// The stall tag ("no 2× gain for 31 / 50") shows from this many iterations without a 2× gain of
/// the best Δρ: half the stall window, above the longest such run measured in a solve that still
/// converged (11 iterations, `validate/references.md` "SCF stall stop"), so a normal solve never
/// shows it.
pub const STALL_TAG_FROM: u32 = 25;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JobProgress {
    Fraction { done: usize, total: usize },
    /// An SCF solve: the iteration being solved and the last solve's count (no total before
    /// convergence).
    Steps { step: u32, last: Option<u32> },
    Indeterminate,
    Waiting,
}

#[derive(Clone, Debug)]
pub struct JobRow {
    pub key: JobKey,
    pub title: String,
    pub detail: String,
    pub progress: JobProgress,
    pub elapsed_s: Option<f64>,
    pub eta_s: Option<f64>,
    /// The one control that exists for this work: its label and the action it sends.
    pub control: Option<(String, Action)>,
    /// Why there is no control (shown on hover).
    pub no_control_why: Option<&'static str>,
    /// The running solve's iterations (the detail line, the sparkline and the stall tag).
    pub scf: Option<sim::ScfLive>,
}

/// The running solve's line ("Δρ 3.2e-5 → 1e-6 · ΔE −2.1e-4 Ha · 85 ms/iteration") and, once
/// [`STALL_TAG_FROM`] iterations passed without a 2× gain of the best Δρ, the stall tag
/// ("no 2× gain for 31 / 50": the stall stop ends the solve at 50/50, `qm::scf` "The stall stop").
pub fn scf_lines(l: &sim::ScfLive) -> (String, Option<String>) {
    let mut parts = vec![format!("Δρ {:.1e} → {:.0e}", l.delta_rho, l.tol)];
    if l.d_energy.is_finite() {
        let sign = if l.d_energy < 0.0 { '−' } else { '+' };
        parts.push(format!("ΔE {sign}{:.1e} Ha", l.d_energy.abs()));
    }
    if l.ms_per_iteration.is_finite() {
        parts.push(if l.ms_per_iteration >= 1000.0 { format!("{:.1} s/iteration", l.ms_per_iteration / 1e3) } else { format!("{:.0} ms/iteration", l.ms_per_iteration) });
    }
    // only while the stop applies (best Δρ at or above the stall level): below it a plateau is never stopped
    let tag = (l.stall_left().is_some() && l.since_gain >= STALL_TAG_FROM).then(|| format!("no 2× gain for {} / {}", l.since_gain, l.window));
    (parts.join(" · "), tag)
}

/// The rows for what the worker does now. `probe` is the live solve state (`PanelInput::probe`).
pub fn rows(input: &PanelInput<'_>, probe: Option<&sim::ProbeView>) -> Vec<JobRow> {
    let mut v = Vec::new();
    let snap = input.snapshot;
    let d = snap.map(|s| &s.dynamics);
    let m = input.model();
    if let Some(p) = d.and_then(|d| d.sampling.as_ref()).filter(|_| input.sampling()) {
        // a solver with nothing to report yet reads 0: say nothing rather than "iteration 0"
        let running = probe.map(|p| p.main).filter(|s| s.running && (s.scf_step > 0 || s.live.is_some_and(|l| l.iteration > 0)));
        let step = running.map(|s| format!(" · {}", iteration_text(s.live.as_ref(), s.scf_step))).unwrap_or_default();
        v.push(JobRow {
            key: JobKey::Pass,
            title: "Sampling pass".into(),
            detail: format!("{} · {}{step}", super::words::pass_stage_of(p, d.and_then(|d| d.ghost.as_ref())), sample_text(p.done, p.total).to_lowercase()),
            progress: JobProgress::Fraction { done: p.done, total: p.total },
            elapsed_s: None,
            eta_s: m.and_then(|m| m.pass_eta_s),
            control: Some(("Cancel".into(), Action::CancelSampling)),
            no_control_why: None,
            // kept between samples: the probe keeps the last solve's record until the next one
            // starts (user request 2026-10-17: the block hid between samples), so it stays on
            // screen and is replaced, not blanked, as each sample begins
            scf: probe.and_then(|p| p.main.live),
        });
    } else if let Some(pv) = probe.filter(|p| p.main.running) {
        let last = d.map(|d| d.eval.scf_steps as u32).filter(|&n| n > 0);
        let estimated = snap.is_some_and(|s| s.backend == crate::sim::Backend::Estimated);
        let cpu = snap.is_some_and(|s| s.engine == crate::sim::Backend::Cpu);
        v.push(JobRow {
            key: JobKey::LiveSolve,
            title: if estimated { "DFT fallback".into() } else { "Solving".into() },
            detail: match (pv.main.scf_step, pv.main.live.filter(|l| l.iteration > 0), cpu) {
                (0, None, true) => "CPU reference: no iteration count yet".into(),
                (0, None, false) => "starting".into(),
                (k, live, _) => steps_text(live.as_ref(), k, last),
            },
            progress: JobProgress::Steps { step: pv.main.scf_step, last },
            elapsed_s: Some(pv.main.elapsed_s),
            eta_s: None,
            control: input.playing.then(|| ("Pause".into(), Action::Pause)),
            no_control_why: (!input.playing).then_some("a paused step finishes; edits abandon it at the next SCF iteration (CPU: after the solve)"),
            scf: pv.main.live,
        });
    }
    if let Some((s, m)) = probe.and_then(|p| p.display) {
        v.push(JobRow {
            key: JobKey::Orbitals,
            title: "Computing orbitals".into(),
            detail: format!("{m} unoccupied orbital{} per spin for the Orbitals view, on the last DFT step's converged state · display only", if m == 1 { "" } else { "s" }),
            progress: JobProgress::Indeterminate,
            elapsed_s: Some(s),
            eta_s: None,
            control: None,
            no_control_why: Some("display only: it never feeds the solve, and any command (an edit, Play, a slider) interrupts it at once; 0 unoccupied orbitals in the Orbitals tab turns it off"),
            scf: None,
        });
    }
    if let Some((s, n)) = probe.and_then(|p| p.refit) {
        v.push(JobRow {
            key: JobKey::Refit,
            title: "Refitting".into(),
            detail: format!("{n} samples"),
            progress: JobProgress::Indeterminate,
            elapsed_s: Some(s),
            eta_s: None,
            control: None,
            no_control_why: Some("a refit takes milliseconds to a few seconds and runs between steps"),
            scf: None,
        });
    }
    if m.is_some_and(|m| m.audit_in_flight) {
        let detail = match probe.filter(|p| p.side.running) {
            Some(p) => format!("background teacher · {}", iteration_text(p.side.live.as_ref(), p.side.scf_step)),
            None => "background teacher".into(),
        };
        v.push(JobRow {
            key: JobKey::Audit,
            title: "Audit".into(),
            detail,
            progress: JobProgress::Indeterminate,
            elapsed_s: probe.map(|p| p.side.elapsed_s).filter(|&s| s > 0.0),
            eta_s: None,
            control: None,
            no_control_why: Some("an audit finishes its solve on the background teacher; the model keeps answering meanwhile"),
            scf: probe.filter(|p| p.side.running).and_then(|p| p.side.live),
        });
    } else if let Some(p) = probe.filter(|p| p.side.running && d.and_then(|d| d.bond_reference).is_none()) {
        // the auditor's other jobs (spec A8): a density refresh or an isolated-atom density
        v.push(JobRow {
            key: JobKey::Background,
            title: "Background teacher".into(),
            detail: format!("density for the cloud · {}", iteration_text(p.side.live.as_ref(), p.side.scf_step)),
            progress: JobProgress::Indeterminate,
            elapsed_s: Some(p.side.elapsed_s),
            eta_s: None,
            control: None,
            no_control_why: Some("a display-only solve on the background teacher; it teaches nothing and never delays a step"),
            scf: None,
        });
    }
    if let Some(r) = d.and_then(|d| d.bond_reference) {
        v.push(JobRow {
            key: JobKey::BondReferences,
            title: "Bond references".into(),
            detail: format!("isolated {} atom, {}/{}", units::isotopes::symbol(r.z), (r.done + 1).min(r.total), r.total),
            progress: JobProgress::Fraction { done: r.done, total: r.total },
            elapsed_s: Some(r.elapsed_s),
            eta_s: None,
            control: None,
            no_control_why: Some("isolated-atom solves for the bond detector, off the stepping path"),
            scf: None,
        });
    }
    if let Some(r) = d.and_then(|d| d.relax.as_ref()).filter(|r| r.converged.is_none()) {
        v.push(JobRow {
            key: JobKey::Relax,
            title: "Relaxing (FIRE)".into(),
            detail: crate::ui::relax_lines(r).first().cloned().unwrap_or_default(),
            progress: JobProgress::Indeterminate,
            elapsed_s: None,
            eta_s: None,
            control: Some(("Stop".into(), Action::StopRelax)),
            no_control_why: None,
            scf: None,
        });
    }
    if let Some(o) = m.and_then(|m| m.pass_offer.as_ref()).filter(|_| !input.sampling()) {
        v.push(JobRow {
            key: JobKey::PassOffer,
            title: "First sampling pass (waiting)".into(),
            detail: format!("{} solves, ≈ {}", o.solves, crate::ui::duration_label(o.eta_s)),
            progress: JobProgress::Waiting,
            elapsed_s: None,
            eta_s: Some(o.eta_s),
            control: Some(("Start anyway".into(), Action::StartOffered(o.plan.clone()))),
            no_control_why: None,
            scf: None,
        });
    }
    v
}

/// The height of [`scf_detail`]'s line and sparkline, reserved while a pass's next sample starts.
const SCF_BLOCK_HEIGHT: f32 = 22.0 + 2.0 * 16.0 + 6.0;

/// The running solve's line, its stall tag and the Δρ sparkline (log Δρ over the iterations so
/// far, the target as a dashed line).
fn scf_detail(ui: &mut Ui, l: &sim::ScfLive) {
    let (line, tag) = scf_lines(l);
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(line).size(11.0).monospace().color(theme::INK_DIM))
            .on_hover_text("this SCF iteration: Δρ = ∫|ρ_out − ρ_in| against the target, the change of the iteration's (Harris–Foulkes) energy since the last one, and the mean time per iteration. On the GPU a ΔE below ~1e-5 Ha is at the f32 reduction noise of the energy terms");
        if let Some(t) = tag {
            ui.label(RichText::new(t).size(11.0).strong().color(AMBER)).on_hover_text(format!(
                "the best Δρ ({:.1e}) has not halved for {} iterations and is still above the stall level {:.1e}; at {} the solve stops as stalled (the iteration cap {} is the backstop; below the level a plateau is never stopped)",
                l.best_delta_rho, l.since_gain, l.stall_level, l.window, l.max_iter
            ));
        }
    });
    let pts = l.spark();
    let finite: Vec<f32> = pts.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.len() < 2 {
        // keep the sparkline's place while a sample's first iteration arrives, so the tab does not
        // jump between samples (user request 2026-10-19)
        ui.allocate_exact_size(egui::vec2(ui.available_width(), 22.0), egui::Sense::hover());
        return;
    }
    let target = (l.tol > 0.0).then(|| l.tol.log10() as f32);
    let lo = finite.iter().copied().fold(f32::INFINITY, f32::min).min(target.unwrap_or(f32::INFINITY));
    let hi = finite.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let span = (hi - lo).max(0.5);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 22.0), egui::Sense::hover());
    let p = ui.painter();
    let x = |i: usize| rect.left() + rect.width() * i as f32 / (pts.len() - 1).max(1) as f32;
    let y = |v: f32| rect.bottom() - rect.height() * (v - lo) / span;
    if let Some(t) = target {
        p.add(egui::Shape::dashed_line(&[egui::pos2(rect.left(), y(t)), egui::pos2(rect.right(), y(t))], egui::Stroke::new(1.0, theme::INK_FAINT), 3.0, 3.0));
    }
    let line: Vec<egui::Pos2> = pts.iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(i, v)| egui::pos2(x(i), y(*v))).collect();
    let colour = if l.stall_left().is_some() && l.since_gain >= STALL_TAG_FROM { AMBER } else { theme::ACCENT };
    p.add(egui::Shape::line(line, egui::Stroke::new(1.2, colour)));
}

/// "1:40" for 100 s, "0:07" for 7 s.
fn mmss(s: f64) -> String {
    let s = s.max(0.0).round() as u64;
    format!("{}:{:02}", s / 60, s % 60)
}

fn colour_of(key: JobKey) -> Color32 {
    match key {
        JobKey::Audit | JobKey::Background => GREEN,
        JobKey::Orbitals => theme::INK_DIM,
        JobKey::PassOffer => AMBER,
        _ => theme::ACCENT,
    }
}

/// Draws the rows; a click on a control pushes its action.
pub fn show(ui: &mut Ui, rows: &[JobRow], acts: &mut Vec<Action>) {
    let t = ui.input(|i| i.time);
    let mut animated = false;
    for r in rows {
        let c = colour_of(r.key);
        ui.horizontal(|ui| {
            theme::light(ui, c);
            ui.label(RichText::new(&r.title).strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| match (&r.control, r.no_control_why) {
                (Some((label, action)), _) => {
                    if ui.button(label).clicked() {
                        acts.push(action.clone());
                    }
                }
                (None, Some(why)) => {
                    ui.label(RichText::new("can't be cancelled").size(10.5).color(theme::INK_FAINT)).on_hover_text(why);
                }
                (None, None) => {}
            });
        });
        ui.add(egui::Label::new(RichText::new(&r.detail).size(11.5).color(theme::INK_DIM)).wrap());
        match r.scf.as_ref().filter(|l| l.iteration > 0) {
            // a pass's block always takes the same height — filled, or empty while the next sample
            // sets up — so the rows below never jump between samples (user requests 2026-10-18/19)
            Some(l) if r.key == JobKey::Pass => {
                let w = ui.available_width();
                ui.allocate_ui_with_layout(egui::vec2(w, SCF_BLOCK_HEIGHT), egui::Layout::top_down(egui::Align::Min), |ui| {
                    ui.set_min_height(SCF_BLOCK_HEIGHT);
                    scf_detail(ui, l);
                });
            }
            Some(l) => scf_detail(ui, l),
            None if r.key == JobKey::Pass => {
                ui.allocate_exact_size(egui::vec2(ui.available_width(), SCF_BLOCK_HEIGHT), egui::Sense::hover());
            }
            None => {}
        }
        // the bar: filled for a fraction, a sweeping segment otherwise
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 6.0), egui::Sense::hover());
        let p = ui.painter();
        p.rect_filled(rect, 3.0, theme::PANEL_3);
        let at = |f: f32| rect.left() + f.clamp(0.0, 1.0) * rect.width();
        match r.progress {
            JobProgress::Fraction { done, total } => {
                let f = if total > 0 { done as f32 / total as f32 } else { 0.0 };
                p.rect_filled(egui::Rect::from_x_y_ranges(rect.left()..=at(f), rect.y_range()), 3.0, c);
            }
            JobProgress::Steps { .. } | JobProgress::Indeterminate => {
                let x = sweep_edge(sweep_phase(t));
                p.rect_filled(egui::Rect::from_x_y_ranges(at(x)..=at(x + 0.2), rect.y_range()), 3.0, c);
                animated = true;
            }
            JobProgress::Waiting => {}
        }
        let mut parts = Vec::new();
        if let JobProgress::Fraction { done, total } = r.progress {
            if total > 0 {
                parts.push(format!("{:.0} %", 100.0 * done as f64 / total as f64));
            }
        }
        if let Some(s) = r.elapsed_s {
            parts.push(format!("{} elapsed", mmss(s)));
        }
        if let Some(s) = r.eta_s {
            parts.push(format!("{} left", mmss(s)));
        }
        resp.on_hover_text(r.detail.as_str());
        if !parts.is_empty() {
            theme::note(ui, parts.join(" · "));
        }
        ui.add_space(2.0);
    }
    if animated {
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(33));
    }
}
