//! The job list (activity spec §3.3.1, §7): one row per running or waiting piece of work of the
//! live worker, each with its progress and the one control that exists today. It is the one
//! component the precision milestone's job queue and the user-data model manager add rows to
//! (`JobKey`): nobody builds a second list.
use super::headline::{steps_text, sweep_edge, sweep_phase};
use super::words::stage_word;
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
    Relax,
    PassOffer,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JobProgress {
    Fraction { done: usize, total: usize },
    /// An SCF solve: the step being solved and the last solve's count (no total before convergence).
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
}

/// The rows for what the worker does now. `probe` is the live solve state (`PanelInput::probe`).
pub fn rows(input: &PanelInput<'_>, probe: Option<&sim::ProbeView>) -> Vec<JobRow> {
    let mut v = Vec::new();
    let snap = input.snapshot;
    let d = snap.map(|s| &s.dynamics);
    let m = input.model();
    if let Some(p) = d.and_then(|d| d.sampling.as_ref()).filter(|_| input.sampling()) {
        let step = probe.filter(|p| p.main.running).map(|p| format!(" · SCF step {}", p.main.scf_step)).unwrap_or_default();
        v.push(JobRow {
            key: JobKey::Pass,
            title: "Sampling pass".into(),
            detail: format!("{} · {}/{}{step}", stage_word(p.stage), p.done, p.total),
            progress: JobProgress::Fraction { done: p.done, total: p.total },
            elapsed_s: None,
            eta_s: m.and_then(|m| m.pass_eta_s),
            control: Some(("Cancel".into(), Action::CancelSampling)),
            no_control_why: None,
        });
    } else if let Some(pv) = probe.filter(|p| p.main.running) {
        let last = d.map(|d| d.eval.scf_steps as u32).filter(|&n| n > 0);
        let estimated = snap.is_some_and(|s| s.backend == crate::sim::Backend::Estimated);
        v.push(JobRow {
            key: JobKey::LiveSolve,
            title: if estimated { "DFT fallback".into() } else { "Solving".into() },
            detail: if pv.main.scf_step == 0 { "CPU reference: no step count".into() } else { steps_text(pv.main.scf_step, last) },
            progress: JobProgress::Steps { step: pv.main.scf_step, last },
            elapsed_s: Some(pv.main.elapsed_s),
            eta_s: None,
            control: input.playing.then(|| ("Pause".into(), Action::Pause)),
            no_control_why: (!input.playing).then_some("a paused step finishes; edits abandon it at the next SCF step (CPU: after the solve)"),
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
        });
    }
    if m.is_some_and(|m| m.audit_in_flight) {
        let detail = match probe.filter(|p| p.side.running) {
            Some(p) => format!("background teacher · SCF step {}", p.side.scf_step),
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
        });
    } else if let Some(p) = probe.filter(|p| p.side.running && d.and_then(|d| d.bond_reference).is_none()) {
        // the auditor's other jobs (spec A8): a density refresh or an isolated-atom density
        v.push(JobRow {
            key: JobKey::Background,
            title: "Background teacher".into(),
            detail: format!("density for the cloud · SCF step {}", p.side.scf_step),
            progress: JobProgress::Indeterminate,
            elapsed_s: Some(p.side.elapsed_s),
            eta_s: None,
            control: None,
            no_control_why: Some("a display-only solve on the background teacher; it teaches nothing and never delays a step"),
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
        });
    }
    v
}

/// "1:40" for 100 s, "0:07" for 7 s.
fn mmss(s: f64) -> String {
    let s = s.max(0.0).round() as u64;
    format!("{}:{:02}", s / 60, s % 60)
}

fn colour_of(key: JobKey) -> Color32 {
    match key {
        JobKey::Audit | JobKey::Background => GREEN,
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
