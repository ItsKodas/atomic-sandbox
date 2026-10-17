//! The Activity dock tab (activity spec §3.3): what runs, what the teacher was asked and why,
//! and how the model is learning. State lives here; `jobs`, `log`, `trust`, `plots` and `headline`
//! draw, and `words` words. Everything is read from records already in memory (the snapshot's
//! [`ActivityDelta`](crate::sim::ActivityDelta) and the live probe): **no view causes a teacher
//! solve**.
pub mod headline;
pub mod jobs;
pub mod log;
pub mod plots;
pub mod trust;
pub mod words;

use super::{dock, search, theme, Action, PanelInput, UiState};
use crate::sim::{Backend, FitPoint, Snapshot};
use egui::Ui;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Solve-log rows kept (spec §3.3.3).
pub const LOG_ROWS: usize = 5000;
/// The burst-temperature histogram's bin (K; spec §3.3.5).
pub const BURST_BIN_K: f64 = 100.0;
/// The histogram's last bin (a stray temperature cannot grow it without bound).
const BURST_BINS_MAX: usize = 1000;
/// Fit-history points kept (one per refit; a session at 2 per second would take hours to pass it).
const FIT_POINTS: usize = 10_000;

pub struct LogRow {
    /// Session time of the event.
    pub at: Duration,
    /// The event as the worker shipped it, with the driver step of its evaluation (`step`).
    pub event: sim::TeachEvent,
    /// The named atom was within `words::d_box(h)` of a face in the snapshot it arrived with.
    pub near_face: bool,
}

/// One mark of the timeline (spec §3.3.6).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mark {
    Fallback(Duration),
    Audit(Duration, sim::AuditVerdict),
    /// A refit; whether the model was trusted after it.
    Refit(Duration, bool),
    /// Samples newly quarantined.
    Quarantine(Duration, usize),
    /// A sampling pass: start, end (None while running), cancelled.
    Pass(Duration, Option<Duration>, bool),
}

/// The log's chips.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogFilter {
    pub pass: bool,
    pub fallback: bool,
    pub audit: bool,
    /// The DFT check of a relaxed geometry.
    pub check: bool,
    pub kept: bool,
    pub not_kept: bool,
}

pub struct ActivityState {
    start: Instant,
    pub rows: VecDeque<LogRow>,
    pub session_solves: u64,
    /// Events absorbed over the session, kept or dropped from the ring (a replay's `solve-log`
    /// baseline: the ring's length stops growing at [`LOG_ROWS`]).
    pub rows_seen: u64,
    pub fit_points: Vec<FitPoint>,
    pub views: Option<Arc<sim::TrainingViews>>,
    /// When `views` arrived (the age line).
    pub views_at: Option<Instant>,
    pub timeline: Vec<Mark>,
    pub filter: LogFilter,
    /// The Training section is open: the worker builds the views only then (`views_sent` is what
    /// it was last told).
    pub training_open: bool,
    pub views_sent: bool,
    /// The parity plot shows forces instead of energies.
    pub parity_forces: bool,
    /// The model was taught at other settings than the scene's (rule 3: it answers nothing here):
    /// the pair-curve caption says so.
    pub provenance_mismatch: bool,
    /// (activity ms, teacher ms) of the current worker (spec §6 footer).
    pub overhead: (f64, f64),
    pub dropped: u64,
    /// Hot-burst geometries solved this session, by instantaneous temperature in bins of
    /// [`BURST_BIN_K`]: kept apart from the log ring so early bursts are not lost (plan review S6).
    pub burst_hist: Vec<u32>,
    /// Session times at which a new run began (`ActivityDelta::run_started`: the worker's start,
    /// a scene replacement, a panic recovery). The log draws a "new run — steps restart" divider
    /// between the rows either side of one: by event time, never by snapshot, because the old
    /// run's last events often arrive in the first snapshot after the replacement (Task 5 review
    /// M2).
    pub runs: Vec<Duration>,
    last_quarantined: Option<usize>,
    pass_running: bool,
    pub headline: headline::Throttled<Option<headline::Headline>>,
}

impl Default for ActivityState {
    fn default() -> Self {
        ActivityState::new()
    }
}

impl ActivityState {
    pub fn new() -> ActivityState {
        ActivityState {
            start: Instant::now(),
            rows: VecDeque::new(),
            session_solves: 0,
            rows_seen: 0,
            fit_points: Vec::new(),
            views: None,
            views_at: None,
            timeline: Vec::new(),
            filter: LogFilter { pass: true, fallback: true, audit: true, check: true, kept: true, not_kept: true },
            training_open: false,
            views_sent: false,
            parity_forces: false,
            provenance_mismatch: false,
            overhead: (0.0, 0.0),
            dropped: 0,
            burst_hist: Vec::new(),
            runs: Vec::new(),
            last_quarantined: None,
            pass_running: false,
            headline: headline::Throttled::new(None),
        }
    }

    fn since(&self, t: Instant) -> Duration {
        t.saturating_duration_since(self.start)
    }

    /// Takes one snapshot's activity: rows, counters, fit points, views, timeline marks.
    pub fn absorb(&mut self, s: &Snapshot) {
        let a = &s.activity;
        let scene = &s.dynamics.scene;
        let d_box = words::d_box(scene.cell.h);
        if let Some(t) = a.run_started {
            // divided by event time: rows with `at` before it belong to the run before (M2)
            self.runs.push(self.since(t));
        }
        for e in &a.events {
            if e.is_solve() {
                self.session_solves += 1;
            }
            if let Some(t) = e.burst_t.filter(|t| t.is_finite() && *t >= 0.0) {
                let bin = ((t / BURST_BIN_K) as usize).min(BURST_BINS_MAX);
                if self.burst_hist.len() <= bin {
                    self.burst_hist.resize(bin + 1, 0);
                }
                self.burst_hist[bin] += 1;
            }
            self.rows_seen += 1;
            let at = self.since(e.at);
            match e.origin {
                sim::activity::TeachOrigin::Fallback => self.timeline.push(Mark::Fallback(at)),
                sim::activity::TeachOrigin::Audit => {
                    if let Some(v) = e.audit {
                        self.timeline.push(Mark::Audit(at, v));
                    }
                }
                _ => {}
            }
            // an event of an earlier run arrives with the new run's scene: its atom index names another geometry
            let old_run = self.runs.last().is_some_and(|&r| at < r);
            let near_face = !old_run && e.atom.and_then(|i| scene.nuclei.get(i)).is_some_and(|n| words::face_distance(scene, n.pos) < d_box);
            self.rows.push_back(LogRow { at, event: e.clone(), near_face });
            if self.rows.len() > LOG_ROWS {
                self.rows.pop_front();
            }
        }
        for p in &a.fit_points {
            // a respawned worker that loads the same model reports its fit again: not a refit
            if self.fit_points.last().is_some_and(|l| (l.epoch, l.samples, l.rmse_e.to_bits(), l.rmse_f.to_bits()) == (p.epoch, p.samples, p.rmse_e.to_bits(), p.rmse_f.to_bits())) {
                continue;
            }
            self.timeline.push(Mark::Refit(self.since(p.at), p.trusted));
            self.fit_points.push(p.clone());
        }
        if self.fit_points.len() > FIT_POINTS {
            self.fit_points.drain(..self.fit_points.len() - FIT_POINTS);
        }
        // `views: None` = no new build; no fitted model = no views (Task 5 review I2)
        if s.model.as_ref().is_none_or(|m| !m.fitted) {
            self.views = None;
            self.views_at = None;
        } else if let Some(v) = &a.views {
            self.views = Some(v.clone());
            self.views_at = Some(Instant::now());
        }
        self.provenance_mismatch = s.model.as_ref().is_some_and(|m| m.provenance_mismatch.is_some());
        self.overhead = (a.activity_ms, a.teacher_ms);
        self.dropped = a.dropped;
        let now = self.since(Instant::now());
        if let Some(m) = &s.model {
            if let Some(q) = self.last_quarantined.filter(|&q| m.quarantined > q) {
                self.timeline.push(Mark::Quarantine(now, m.quarantined - q));
            }
            self.last_quarantined = Some(m.quarantined);
        }
        let running = crate::sim::sampling_active(&s.dynamics);
        if running && !self.pass_running {
            self.timeline.push(Mark::Pass(now, None, false));
        } else if !running && self.pass_running {
            let cancelled = s.dynamics.sampling.as_ref().is_some_and(|p| p.cancelled);
            if let Some(Mark::Pass(_, end, c)) = self.timeline.iter_mut().rev().find(|m| matches!(m, Mark::Pass(..))) {
                *end = Some(now);
                *c = cancelled;
            }
        }
        self.pass_running = running;
        // the timeline is a view: bounded like the log
        if self.timeline.len() > 4 * LOG_ROWS {
            self.timeline.drain(..self.timeline.len() - 4 * LOG_ROWS);
        }
    }

    /// A new worker is starting (`Viewer::start`): a pass the old one was running is closed as
    /// cancelled (the new worker's snapshots have no pass to say so), and the quarantine count is
    /// the new model's own, not a change (activity final review M2).
    pub fn new_worker(&mut self) {
        let now = self.since(Instant::now());
        if self.pass_running {
            if let Some(Mark::Pass(_, end, cancelled)) = self.timeline.iter_mut().rev().find(|m| matches!(m, Mark::Pass(..))) {
                *end = Some(now);
                *cancelled = true;
            }
        }
        self.pass_running = false;
        self.last_quarantined = None;
    }

    /// Session time now (the timeline's right edge).
    pub fn clock_now(&self) -> Duration {
        self.since(Instant::now())
    }

    /// The species of the newest *finished* pass reference solve of the pass now running (the
    /// headline's "reference (last done: H)"): a reference row is pushed when its solve completes,
    /// so it is not the species being solved, and rows from before this pass began (an earlier
    /// pass, another scene) are not this pass's (activity final review I2). `None` without a
    /// running pass or before its first reference lands.
    pub fn current_reference(&self) -> Option<u8> {
        let since = self.timeline.iter().rev().find_map(|m| match m {
            Mark::Pass(start, None, _) => Some(*start),
            _ => None,
        })?;
        self.rows.iter().rev().take_while(|r| r.at >= since).find_map(|r| match r.event.rule {
            sim::activity::TeachRule::Pass(sim::activity::PassJob::Reference { z, .. }) => Some(z),
            _ => None,
        })
    }
}

/// The tab: jobs, counters, the solve log, then (estimated solver) the trust checklist, the
/// training views and the timeline. Each group is marked with its search anchor.
pub fn show(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    let estimated = st.backend == Backend::Estimated;
    dock::group(ui, st, "d-jobs", |ui, _| {
        theme::section(ui, "Jobs");
        let rows = jobs::rows(input, input.probe.as_ref());
        if rows.is_empty() {
            theme::note(ui, "nothing running");
        } else {
            jobs::show(ui, &rows, acts);
        }
    });
    dock::group(ui, st, "d-counters", |ui, st| {
        theme::section(ui, "Counters");
        log::counters(ui, &st.activity, input.model());
    });
    dock::group(ui, st, "d-solve-log", |ui, st| {
        theme::section(ui, "Solve log");
        log::show_log(ui, &mut st.activity, estimated);
    });
    if !estimated {
        theme::note(ui, "The trust checklist, the training views and the timeline are for the estimated solver. A plain DFT step is not a teaching solve, so it adds no log row and does not tick the counter.");
        return;
    }
    dock::group(ui, st, "d-trust", |ui, st| {
        theme::section(ui, "Trust");
        trust::show(ui, st, input);
    });
    let target_t = st.plan.hot_burst_t;
    let draft = input.model().is_some_and(|m| m.draft);
    let open = st.activity.training_open || search::opens(st, "d-training").is_some();
    let r = egui::CollapsingHeader::new("Training").id_salt("activity-training").open(Some(open)).show(ui, |ui| plots::training(ui, &mut st.activity, Some(target_t), draft));
    if r.header_response.clicked() {
        st.activity.training_open = !open;
    } else if open {
        st.activity.training_open = true;
    }
    search::mark(ui, st, "d-training", r.header_response.rect);
    dock::group(ui, st, "d-timeline", |ui, st| {
        theme::section(ui, "Timeline");
        plots::timeline(ui, &st.activity);
    });
    overhead_footer(ui, &st.activity);
}

/// The measured cost of the bookkeeping (spec §6), dim, hidden while there is no teacher time.
fn overhead_footer(ui: &mut Ui, a: &ActivityState) {
    let (act, teacher) = a.overhead;
    if teacher > 0.0 {
        let views = a.views.as_ref().map_or(String::new(), |v| format!("; views {:.1} ms per build", v.build_ms));
        theme::note(ui, format!("activity bookkeeping {:.3} % of teacher time ({act:.1} ms of {teacher:.0} ms){views}", 100.0 * act / teacher));
    }
}

#[cfg(test)]
pub fn fixture_views() -> sim::TrainingViews {
    use sim::activity::{PairCurve, PairHistogram, ParityPoint, SampleKind, SampleState, TeachRule};
    let n = sim::activity::CURVE_POINTS;
    let r: Vec<f64> = (0..n).map(|k| 0.5 + 5.5 * k as f64 / (n - 1) as f64).collect();
    let morse = |r: f64| 0.17 * ((-2.0 * 1.0 * (r - 1.4)).exp() - 2.0 * (-(r - 1.4)).exp());
    let energy: Vec<f64> = r.iter().map(|&x| morse(x)).collect();
    let curve = PairCurve {
        z: (1, 1),
        sigma: r.iter().map(|&x| 2e-5 * (1.0 + (x - 1.4).abs())).collect(),
        force_ratio: r.iter().map(|&x| 0.2 + 0.1 * (x - 1.4).abs()).collect(),
        refused: (0..n).map(|k| (k >= n - 20).then_some(TeachRule::IsolatedAtom)).collect(),
        dots: vec![(1.4, -0.17, SampleState::Fitted), (2.0, -0.12, SampleState::Unfitted)],
        energy,
        r,
        r_cut: 6.0,
    };
    sim::TrainingViews {
        epoch: 3,
        model_gen: 4,
        samples: 273,
        parity_energy: vec![
            ParityPoint { dft: -0.17, model: -0.1702, kind: SampleKind::Pass, state: SampleState::Fitted, index: 0 },
            ParityPoint { dft: -0.1, model: -0.099, kind: SampleKind::Fallback, state: SampleState::Quarantined, index: 1 },
        ],
        parity_force: vec![[0.01, 0.0102], [-0.02, -0.0199], [0.0, 0.0001]],
        force_points_total: 3,
        parity_omitted: 1,
        curves: vec![curve],
        curves_skipped: 0,
        pairs: vec![PairHistogram { z: (1, 1), bin: 0.1, counts: (0..60).map(|k| (k % 7) as u32).collect() }],
        dimer_floor: 4e-5,
        energy_tier: 1e-3,
        build_ms: 2.3,
    }
}

/// The progress bar on the toolbar's top edge (spec §3.2, ruling A1): filled to done/total for a
/// pass and for bond references, a sweeping segment for single solves, refits and relaxes, an
/// amber tail while a background audit runs alongside, hidden when idle. Hovering shows the
/// headline; clicking opens this tab.
pub fn bar(root: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, toolbar: egui::Rect) {
    use egui::{Color32, Id, LayerId, Order, Rect};
    let Some(h) = st.activity.headline.peek().clone() else { return };
    let p = root.ctx().layer_painter(LayerId::new(Order::Foreground, Id::new("activity-bar")));
    let r = Rect::from_min_max(toolbar.left_top(), egui::pos2(toolbar.right(), toolbar.top() + 3.0));
    p.rect_filled(r, 0.0, theme::LINE);
    let fill = |a: f32, b: f32, c: Color32| {
        p.rect_filled(Rect::from_x_y_ranges(r.left() + a.clamp(0.0, 1.0) * r.width()..=r.left() + b.clamp(0.0, 1.0) * r.width(), r.y_range()), 0.0, c)
    };
    match h.bar {
        headline::BarMode::Fraction(f) => {
            fill(0.0, f, theme::ACCENT);
        }
        headline::BarMode::Sweep => {
            let x = headline::sweep_edge(headline::sweep_phase(root.input(|i| i.time)));
            fill(x, x + 0.2, theme::ACCENT);
        }
    }
    if h.audit.is_some() {
        fill(0.94, 1.0, super::AMBER);
    }
    let text = if h.text.is_empty() { input.snapshot.and_then(|s| s.dynamics.bond_reference).map_or(String::new(), |r| super::overlays::bond_reference_line(&r)) } else { h.text.clone() };
    let hit = root.interact(r.expand2(egui::vec2(0.0, 3.0)), Id::new("activity-bar-hit"), egui::Sense::click());
    super::probe("activity-bar", &hit);
    if hit.on_hover_text(format!("{text}\n(click: the Activity tab)")).clicked() {
        st.dock_open = true;
        st.dock_tab = super::DockTab::Activity;
    }
    root.ctx().request_repaint_after(Duration::from_millis(33));
}
