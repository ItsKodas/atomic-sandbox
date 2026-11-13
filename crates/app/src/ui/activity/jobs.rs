//! The job list (activity spec §3.3.1, §7): one row per running or waiting piece of work of the
//! live worker, each with its progress and the one control that exists today. It is the one
//! component the precision milestone's job queue and the user-data model manager add rows to
//! (`JobKey`): nobody builds a second list. Drawn (2026-11-14) as the running job's card ([`card`]:
//! the convergence plot from [`Traces`], the iterations strip from [`StripBar`]s) and the other
//! jobs one line each ([`compact`]).
use super::headline::{iteration_text, sample_text, steps_text, sweep_edge, sweep_phase};

use crate::ui::{theme, Action, PanelInput, AMBER, GREEN, RED};
use std::time::{Duration, Instant};
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
    /// The row keeps the solve block's height while it has nothing to show (a sample or step
    /// setting up, its first iteration), so the rows below never jump.
    pub keeps_block: bool,
    /// The card title's second part ("Sampling pass · bond scan").
    pub stage: Option<String>,
    /// The card's count line ("sample 40 of 144"); the detail otherwise.
    pub count: Option<String>,
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
            keeps_block: true,
            stage: Some(super::words::pass_stage_of(p, d.and_then(|d| d.ghost.as_ref()))),
            count: Some(sample_text(p.done, p.total).to_lowercase()),
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
            keeps_block: !cpu,
            stage: None,
            count: None,
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
            keeps_block: false,
            stage: None,
            count: None,
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
            keeps_block: false,
            stage: None,
            count: None,
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
            keeps_block: false,
            stage: None,
            count: None,
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
            keeps_block: false,
            stage: None,
            count: None,
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
            keeps_block: false,
            stage: None,
            count: None,
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
            keeps_block: false,
            stage: None,
            count: None,
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
            keeps_block: false,
            stage: None,
            count: None,
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
        JobKey::Orbitals => theme::INK_DIM,
        JobKey::PassOffer => AMBER,
        _ => theme::ACCENT,
    }
}

// ---- the job card (the Activity redesign, 2026-11-14) ----

/// Solves younger than this are not animated: the plot shows them once they end (a fast solve's
/// line would only flicker).
pub const ANIMATE_AFTER: Duration = Duration::from_millis(500);
/// Earlier solves of the same job drawn faint behind the current one.
pub const GHOSTS: usize = 3;

/// One solve's Δρ trace: (iteration, log10 Δρ) points from the solver's record
/// (`sim::ScfLive::spark`, point k at iteration (k + 1) × stride), the latest iteration and Δρ.
#[derive(Clone, Debug, PartialEq)]
pub struct Trace {
    pub points: Vec<(f32, f32)>,
    pub iteration: u32,
    pub attempt: u32,
    pub delta_rho: f64,
    pub tol: f64,
    pub started: Instant,
}

impl Trace {
    fn of(l: &sim::ScfLive, now: Instant) -> Trace {
        let mut t = Trace { points: Vec::new(), iteration: 0, attempt: l.attempt, delta_rho: l.delta_rho, tol: l.tol, started: now };
        t.update(l);
        t
    }

    fn update(&mut self, l: &sim::ScfLive) {
        let stride = l.spark_stride.max(1) as f32;
        self.points = l.spark().iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(k, &v)| ((k as f32 + 1.0) * stride, v)).collect();
        if l.delta_rho > 0.0 && l.delta_rho.is_finite() && self.points.last().is_none_or(|p| p.0 < l.iteration as f32) {
            self.points.push((l.iteration as f32, l.delta_rho.log10() as f32));
        }
        self.iteration = l.iteration;
        self.attempt = l.attempt;
        self.delta_rho = l.delta_rho;
        self.tol = l.tol;
    }
}

/// The convergence plot's memory (`ActivityState::conv`): the job it follows, its fixed log10 Δρ
/// range (set by the job's first solve: from max(1, its first Δρ) to a decade below the stop
/// tolerance — never rescaled per solve), the solve running now and the [`GHOSTS`] before it (kept
/// when a new solve starts, so the plot never flashes blank).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Traces {
    pub key: Option<JobKey>,
    pub range: Option<(f32, f32)>,
    pub current: Option<Trace>,
    pub ghosts: std::collections::VecDeque<Trace>,
}

impl Traces {
    /// Takes the job's live solve record this frame (`None`: the job has no solve to show yet).
    pub fn observe(&mut self, key: JobKey, live: Option<&sim::ScfLive>, now: Instant) {
        if self.key != Some(key) {
            *self = Traces { key: Some(key), ..Traces::default() };
        }
        let Some(l) = live.filter(|l| l.iteration > 0) else { return };
        if self.range.is_none() {
            let first = l.spark().iter().copied().find(|v| v.is_finite()).unwrap_or(if l.delta_rho > 0.0 { l.delta_rho.log10() as f32 } else { 0.0 });
            let hi = first.max(0.0).ceil();
            let lo = if l.tol > 0.0 { l.tol.log10().floor() as f32 - 1.0 } else { hi - 7.0 };
            self.range = Some((lo.min(hi - 1.0), hi));
        }
        match &mut self.current {
            // the same solve: its iterations only grow
            Some(c) if c.attempt == l.attempt && l.iteration >= c.iteration => c.update(l),
            _ => {
                if let Some(old) = self.current.take() {
                    self.ghosts.push_back(old);
                    while self.ghosts.len() > GHOSTS {
                        self.ghosts.pop_front();
                    }
                }
                self.current = Some(Trace::of(l, now));
            }
        }
    }

    /// The trace drawn bright: the running solve once it is [`ANIMATE_AFTER`] old, else the last
    /// finished one; and the faint ones behind it.
    pub fn shown(&self, now: Instant) -> (Option<&Trace>, Vec<&Trace>) {
        let live = self.current.as_ref().filter(|c| now.saturating_duration_since(c.started) >= ANIMATE_AFTER);
        match live {
            Some(c) => (Some(c), self.ghosts.iter().collect()),
            None => {
                let n = self.ghosts.len();
                (self.ghosts.back(), self.ghosts.iter().take(n.saturating_sub(1)).collect())
            }
        }
    }
}

/// "1e-6", "1e-3", "1": a log10 tick's label.
fn tick_label(v: f32) -> String {
    match v.round() as i32 {
        0 => "1".into(),
        1 => "10".into(),
        k => format!("1e{k}"),
    }
}

/// The convergence plot: log Δρ on the fixed range of [`Traces`], the target as a dashed line,
/// the earlier solves faint, the current one bright with a dot at its latest point and its label.
fn conv_plot(ui: &mut Ui, tr: &Traces) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 74.0), egui::Sense::hover());
    crate::ui::probe_rect("activity-conv", rect);
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 4.0, Color32::from_black_alpha(60));
    let Some((lo, hi)) = tr.range else {
        p.text(rect.center(), egui::Align2::CENTER_CENTER, "Δρ per iteration appears as the solve runs", egui::FontId::proportional(11.0), theme::INK_FAINT);
        return;
    };
    let now = Instant::now();
    let (bright, faint) = tr.shown(now);
    let inner = egui::Rect::from_min_max(rect.min + egui::vec2(30.0, 6.0), rect.max - egui::vec2(6.0, 6.0));
    let x_max = bright.into_iter().chain(faint.iter().copied()).flat_map(|t| t.points.last().map(|p| p.0)).fold(10.0f32, f32::max);
    let x = |it: f32| inner.left() + inner.width() * (it / x_max).clamp(0.0, 1.0);
    let y = |v: f32| inner.bottom() - inner.height() * ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
    // ticks: every few decades, labelled
    let step = (((hi - lo) / 3.0).ceil()).max(1.0);
    let mut v = hi;
    while v >= lo - 1e-3 {
        p.line_segment([egui::pos2(inner.left(), y(v)), egui::pos2(inner.right(), y(v))], egui::Stroke::new(1.0, theme::LINE_SOFT));
        p.text(egui::pos2(inner.left() - 4.0, y(v)), egui::Align2::RIGHT_CENTER, tick_label(v), egui::FontId::monospace(9.0), theme::INK_FAINT);
        v -= step;
    }
    let tol = bright.or(faint.last().copied()).or(tr.current.as_ref()).map(|t| t.tol).filter(|t| *t > 0.0);
    if let Some(t) = tol {
        let ty = y(t.log10() as f32);
        p.add(egui::Shape::dashed_line(&[egui::pos2(inner.left(), ty), egui::pos2(inner.right(), ty)], egui::Stroke::new(1.0, GREEN.gamma_multiply(0.8)), 4.0, 3.0));
        p.text(egui::pos2(inner.right(), ty - 1.0), egui::Align2::RIGHT_BOTTOM, "target", egui::FontId::proportional(9.5), GREEN);
    }
    let line = |t: &Trace| t.points.iter().map(|&(i, v)| egui::pos2(x(i), y(v))).collect::<Vec<_>>();
    for t in &faint {
        p.add(egui::Shape::line(line(t), egui::Stroke::new(1.0, theme::INK_FAINT.gamma_multiply(0.7))));
    }
    if let Some(t) = bright {
        p.add(egui::Shape::line(line(t), egui::Stroke::new(1.6, theme::ACCENT)));
        if let Some(&(i, v)) = t.points.last() {
            let at = egui::pos2(x(i), y(v));
            p.circle_filled(at, 3.0, theme::ACCENT);
            let label = format!("it {} · {:.1e} → {:.0e}", t.iteration, t.delta_rho, t.tol);
            let align = if at.x > inner.center().x { egui::Align2::RIGHT_BOTTOM } else { egui::Align2::LEFT_BOTTOM };
            let off = if at.x > inner.center().x { egui::vec2(-5.0, -3.0) } else { egui::vec2(5.0, -3.0) };
            let at_y = at.y.max(inner.top() + 12.0);
            p.text(egui::pos2(at.x, at_y) + off, align, label, egui::FontId::monospace(9.5), theme::ACCENT_INK);
        }
    }
}

/// One bar of the iterations-per-sample strip ([`super::strip`]): a finished sample of the pass
/// running now (its SCF iterations; `danger`: failed, timed out, stalled or not kept — drawn full
/// height), or the sample being solved (`current`).
#[derive(Clone, Debug, PartialEq)]
pub struct StripBar {
    pub iterations: Option<usize>,
    pub danger: bool,
    pub current: bool,
    pub words: String,
}

/// The strip under the plot: one thin bar per sample, height = its SCF iterations.
fn strip(ui: &mut Ui, bars: &[StripBar]) {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 22.0), egui::Sense::hover());
    crate::ui::probe_rect("activity-strip", rect);
    let p = ui.painter_at(rect);
    // the newest that fit, 3 pt a bar
    let fit = ((rect.width() / 3.0) as usize).max(1);
    let shown = &bars[bars.len().saturating_sub(fit)..];
    let max = shown.iter().filter(|b| !b.danger).filter_map(|b| b.iterations).max().unwrap_or(1).max(1) as f32;
    let w = 3.0;
    let mut hit: Option<&StripBar> = None;
    for (k, b) in shown.iter().enumerate() {
        let x0 = rect.left() + k as f32 * w;
        let h = if b.danger { rect.height() } else { (rect.height() * b.iterations.unwrap_or(0) as f32 / max).clamp(1.5, rect.height()) };
        let c = if b.danger {
            RED
        } else if b.current {
            theme::ACCENT
        } else {
            theme::INK_FAINT
        };
        let r = egui::Rect::from_min_max(egui::pos2(x0, rect.bottom() - h), egui::pos2(x0 + w - 1.0, rect.bottom()));
        p.rect_filled(r, 0.0, c);
        if resp.hover_pos().is_some_and(|q| q.x >= x0 && q.x < x0 + w) {
            hit = Some(b);
        }
    }
    let tip = hit.map_or_else(|| "SCF iterations per sample of this pass (red: failed, timed out or not kept)".to_string(), |b| b.words.clone());
    resp.on_hover_text(tip);
}

/// The running job's card: "Sampling pass · bond scan" with its control, the bar, "sample 40 of 144"
/// and "27 % · 53 min left", the solve's small line ("iteration 6 · 2.2 s each", "Δρ 6.8e-1 →
/// 1e-6"), the convergence plot and (a pass) the iterations strip.
pub fn card(ui: &mut Ui, r: &JobRow, tr: &Traces, bars: &[StripBar], acts: &mut Vec<Action>) {
    let c = colour_of(r.key);
    let resp = egui::Frame::new().fill(theme::PANEL_2).stroke(egui::Stroke::new(1.0, theme::LINE)).corner_radius(8).inner_margin(egui::Margin::symmetric(10, 8)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        let title = match &r.stage {
            Some(s) => format!("{} · {s}", r.title),
            None => r.title.clone(),
        };
        ui.horizontal_top(|ui| {
            theme::light(ui, c);
            let cw = control_width(ui, r);
            let w = (ui.available_width() - cw).max(80.0);
            ui.allocate_ui_with_layout(egui::vec2(w, 0.0), egui::Layout::top_down(egui::Align::Min), |ui| {
                ui.set_width(w);
                crate::ui::probed("activity-job-title", ui.add(egui::Label::new(RichText::new(&title).strong().size(13.5)).wrap()));
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| control(ui, r, acts));
        });
        bar(ui, r, c);
        // "sample 40 of 144" · "27 % · 53 min left"
        let left = r.count.clone().unwrap_or_else(|| r.detail.clone());
        let right = progress_words(r);
        two_sides(ui, &left, &right, egui::FontId::proportional(11.5), theme::INK_DIM).on_hover_text(r.detail.as_str());
        if let Some(l) = r.scf.as_ref().filter(|l| l.iteration > 0) {
            let each = if l.ms_per_iteration.is_finite() {
                if l.ms_per_iteration >= 1000.0 { format!(" · {:.1} s each", l.ms_per_iteration / 1e3) } else { format!(" · {:.0} ms each", l.ms_per_iteration) }
            } else {
                String::new()
            };
            let retry = if l.attempt > 1 { " · cold retry" } else { "" };
            let (full, tag) = scf_lines(l);
            two_sides(ui, &format!("iteration {}{each}{retry}", l.iteration), &format!("Δρ {:.1e} → {:.0e}", l.delta_rho, l.tol), egui::FontId::monospace(10.5), theme::INK_DIM)
                .on_hover_text(format!("{full}\n\nΔρ = ∫|ρ_out − ρ_in| against the target; the time is the mean per iteration. On the GPU a ΔE below ~1e-5 Ha is at the f32 reduction noise of the energy terms"));
            if let Some(t) = tag {
                ui.label(RichText::new(t).size(11.0).strong().color(AMBER)).on_hover_text(format!(
                    "the best Δρ ({:.1e}) has not halved for {} iterations and is still above the stall level {:.1e}; at {} the solve stops as stalled (the iteration cap {} is the backstop; below the level a plateau is never stopped)",
                    l.best_delta_rho, l.since_gain, l.stall_level, l.window, l.max_iter
                ));
            }
        }
        if r.scf.is_none_or(|l| l.iteration == 0) && r.keeps_block {
            // the iteration line's place while the next sample or step sets up (the rows below
            // never jump, user requests 2026-10-18/19, 2026-11-11)
            let h = ui.ctx().fonts_mut(|f| f.row_height(&egui::FontId::monospace(10.5))) + 2.0;
            ui.allocate_exact_size(egui::vec2(ui.available_width(), h), egui::Sense::hover());
        }
        if r.scf.is_some() || r.keeps_block || tr.range.is_some() {
            conv_plot(ui, tr);
        }
        if r.key == JobKey::Pass {
            strip(ui, bars);
        }
    });
    crate::ui::probe_rect("activity-job-card", resp.response.rect);
}

/// The room [`control`] takes on the right, with the gap before it.
fn control_width(ui: &Ui, r: &JobRow) -> f32 {
    let w = |text: &str, size: f32| ui.painter().layout_no_wrap(text.to_string(), egui::FontId::proportional(size), theme::INK).size().x;
    let gap = ui.spacing().item_spacing.x;
    match (&r.control, r.no_control_why) {
        (Some((label, _)), _) => w(label, 13.0) + 2.0 * ui.spacing().button_padding.x + gap,
        (None, Some(_)) => w("can't be cancelled", 10.5) + gap,
        (None, None) => 0.0,
    }
}

/// A row's control on the right ("Cancel", "Pause", …), or "can't be cancelled" with why.
fn control(ui: &mut Ui, r: &JobRow, acts: &mut Vec<Action>) {
    match (&r.control, r.no_control_why) {
        (Some((label, action)), _) => {
            if crate::ui::probed(&format!("job-control-{}", label.to_lowercase().replace(' ', "-")), ui.button(label)).clicked() {
                acts.push(action.clone());
            }
        }
        (None, Some(why)) => {
            ui.label(RichText::new("can't be cancelled").size(10.5).color(theme::INK_FAINT)).on_hover_text(why);
        }
        (None, None) => {}
    }
}

/// "27 % · 53 min left", "0:07 elapsed".
fn progress_words(r: &JobRow) -> String {
    let mut parts = Vec::new();
    if let JobProgress::Fraction { done, total } = r.progress {
        if total > 0 {
            parts.push(format!("{:.0} %", 100.0 * done as f64 / total as f64));
        }
    }
    if let Some(s) = r.eta_s {
        parts.push(format!("{} left", crate::ui::duration_label(s)));
    } else if let Some(s) = r.elapsed_s {
        parts.push(format!("{} elapsed", mmss(s)));
    }
    parts.join(" · ")
}

/// A line with `left` at the start and `right` at the end, each cut with "…" when the width is
/// short (the right side keeps its own width first).
pub fn two_sides(ui: &mut Ui, left: &str, right: &str, font: egui::FontId, colour: Color32) -> egui::Response {
    let h = ui.ctx().fonts_mut(|f| f.row_height(&font)) + 2.0;
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), h), egui::Sense::hover());
    let p = ui.painter();
    let rg = p.layout_no_wrap(right.to_string(), font.clone(), colour);
    let rw = rg.size().x.min(rect.width() * 0.6);
    let mut job = egui::text::LayoutJob::single_section(right.to_string(), egui::TextFormat { font_id: font.clone(), color: colour, ..Default::default() });
    job.wrap = egui::text::TextWrapping { max_width: rw.max(1.0), max_rows: 1, break_anywhere: true, overflow_character: Some('…') };
    let rg = p.layout_job(job);
    p.galley(egui::pos2(rect.right() - rg.size().x, rect.top() + 1.0), rg.clone(), colour);
    let room = (rect.width() - rg.size().x - 10.0).max(1.0);
    let mut job = egui::text::LayoutJob::single_section(left.to_string(), egui::TextFormat { font_id: font, color: colour, ..Default::default() });
    job.wrap = egui::text::TextWrapping { max_width: room, max_rows: 1, break_anywhere: true, overflow_character: Some('…') };
    let lg = p.layout_job(job);
    p.galley(egui::pos2(rect.left(), rect.top() + 1.0), lg, colour);
    resp
}

/// The progress bar: filled for a fraction, a sweeping segment otherwise.
fn bar(ui: &mut Ui, r: &JobRow, c: Color32) {
    let t = ui.input(|i| i.time);
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
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(33));
        }
        JobProgress::Waiting => {}
    }
    resp.on_hover_text(r.detail.as_str());
}

/// A queued or side job in one line: its light, title and detail (cut), the progress words and
/// its control.
pub fn compact(ui: &mut Ui, r: &JobRow, acts: &mut Vec<Action>) {
    let c = colour_of(r.key);
    ui.horizontal(|ui| {
        theme::light(ui, c);
        let words = progress_words(r);
        let cw = if r.control.is_some() { control_width(ui, r) } else { 0.0 };
        let w = (ui.available_width() - cw).max(60.0);
        let line = if words.is_empty() { format!("{} · {}", r.title, r.detail) } else { format!("{} · {} · {words}", r.title, r.detail) };
        let mut job = egui::text::LayoutJob::single_section(line, egui::TextFormat { font_id: egui::FontId::proportional(11.5), color: theme::INK, ..Default::default() });
        job.wrap = egui::text::TextWrapping { max_width: w - 4.0, max_rows: 1, break_anywhere: true, overflow_character: Some('…') };
        let g = ui.painter().layout_job(job);
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(w, g.size().y.max(16.0)), egui::Sense::hover());
        ui.painter().galley(egui::pos2(rect.left(), rect.center().y - g.size().y / 2.0), g, theme::INK);
        resp.on_hover_text(format!("{}\n{}", r.title, r.detail));
        if r.control.is_some() {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| control(ui, r, acts));
        }
    });
}

/// The job list: the first job as the card ([`card`]), the others one line each ([`compact`]).
pub fn show(ui: &mut Ui, rows: &[JobRow], tr: &Traces, bars: &[StripBar], acts: &mut Vec<Action>) {
    let Some((first, rest)) = rows.split_first() else { return };
    card(ui, first, tr, bars, acts);
    for r in rest {
        compact(ui, r, acts);
    }
}
