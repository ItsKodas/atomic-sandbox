//! The worker's side of the Activity view (activity spec §4.3 A7, §6): drains the driver's
//! teaching events every round (already stamped with the step whose evaluation produced each, by
//! `sim::Dynamics`: plan review M6 — the worker adds no stamp), keeps the fit history, builds the
//! training views only while the UI shows them, on their own schedule, and accounts its own cost.
//!
//! # The events
//! [`Worker::drain_teach_events`] runs once per round (every tick, quiet polling ticks included:
//! a round that drained events is not quiet and publishes) and once more in every publish (early
//! publishes too), so the driver's bounded buffer (`sim::activity::DRIVER_EVENTS`) only ever
//! holds one round's events. What is drained waits in [`ActivityWork`] until the next publish
//! hands it out, oldest first: the snapshots carry every event exactly once, in the order the
//! provider made them (`TeachEvent::at`), whatever the command traffic (a flood of commands is a
//! flood of early publishes, each of which drains).
//!
//! **A scene replacement does not clear pending events** (Task 1b carry): the driver collects
//! the provider's queue at the old step before it resets (`ReplaceScene`), and the worker
//! delivers them like any other. They are teaching solves that happened, cost teacher time and
//! are in the model's lifetime count (`SolveCounts`, which a scene replacement keeps); dropping
//! them would make the log's session count disagree with the model's. Their step stamps belong
//! to the run before the replacement — the driver's step count restarts at 0 with the new scene
//! — so the log is in arrival order (`at`), which is step order within a run.
//!
//! # Runs
//! A run is what one step count numbers: it starts with the worker (a backend switch is a new
//! worker), at every scene replacement the worker applies (a new scene, charge, multiplicity,
//! grid or σ) and at a panic recovery (time restarts). [`ActivityDelta::run_started`] carries the
//! instant of the latest start since the last publish, and the UI divides its log **by
//! `TeachEvent::at`**, not by snapshot: the old run's last events often arrive in the first
//! snapshot after the replacement (the driver collected them at the old step), so a divider
//! placed per snapshot would put them in the new run. A run start also drops
//! [`ModelStatus::last_audit_step`] (a step of the old run; the audit record itself is kept).
//!
//! # The training views' schedule (Task 4 physics review, cost ruling)
//! Built on the worker (not off-thread: ruling item 5), only while
//! [`Command::SetActivityViews`] is on and the model is fitted, only when what they show may have
//! changed ([`ViewKey`]: the model generation plus what moves between generations), and never
//! sooner than [`views_interval`] after the last build:
//! max([`VIEWS_MIN_INTERVAL`], [`VIEWS_DUTY_FACTOR`] × the last build's `build_ms`) — at most
//! 2 % of the worker in the long run. The provider caches the expensive curves per model
//! generation, so a rebuild at the same generation is cheap (S = 4: 3.1 s cold, 2.3 ms warm,
//! Task 4); the factor keeps a cold S = 4 build from recurring every 2 s. Turning the views on
//! again forgets the key (the next due publish rebuilds even when nothing changed) but not the
//! interval.
//!
//! **A deferred build wakes the worker** (Task 5 review I1): when the key changed but the
//! interval has not passed, the build's due time is kept ([`Worker::views_wake`]) and an idle
//! worker's loop waits for it instead of blocking for the next command; at the deadline it runs a
//! round with no command, whose publish builds the views. Without it the views stayed stale
//! until an unrelated command arrived (a pass's last refit within 2 s of the last build, then
//! idle).
//!
//! **`views: None` means "no new build", not "no views"**: the UI keeps the last views it got
//! and drops them itself when `ModelStatus::fitted` is false (a Forget, no model) and when it
//! starts a new worker (plan Task 6, `ActivityState::absorb` / `Viewer::start`).
use super::*;

/// The fewest seconds between two training-view builds (spec §6).
pub const VIEWS_MIN_INTERVAL: Duration = Duration::from_secs(2);

/// The backstop's factor on the last build's wall time (Task 4 review, cost ruling item 3): the
/// next build waits at least this many times as long as the last one took, so view builds take
/// at most 1/50 = 2 % of the worker.
pub const VIEWS_DUTY_FACTOR: f64 = 50.0;

/// The wait after a build of `build_ms` before the next one (ruling item 3):
/// max([`VIEWS_MIN_INTERVAL`], [`VIEWS_DUTY_FACTOR`] × `build_ms`). A non-finite or negative
/// time counts as 0, and the wait is capped at an hour.
pub fn views_interval(build_ms: f64) -> Duration {
    let ms = if build_ms.is_finite() { build_ms.max(0.0) } else { 0.0 };
    VIEWS_MIN_INTERVAL.max(Duration::from_secs_f64((VIEWS_DUTY_FACTOR * ms / 1e3).min(3600.0)))
}

/// One point of the fit history (spec §4.2 "rmse per epoch"): the model's last fit whenever it
/// changes. rmse E in Ha/atom, rmse F in Ha/Bohr per component.
#[derive(Clone, Debug, PartialEq)]
pub struct FitPoint {
    pub epoch: u64,
    pub samples: usize,
    pub rmse_e: f64,
    pub rmse_f: f64,
    pub trusted: bool,
    pub at: Instant,
}

/// What one snapshot adds to the Activity view (spec A7).
#[derive(Clone, Debug, Default)]
pub struct ActivityDelta {
    /// The teaching events since the last publish, oldest first, each with its driver step.
    pub events: Vec<sim::TeachEvent>,
    /// New fit-history points since the last publish (one per new fit).
    pub fit_points: Vec<FitPoint>,
    /// The training views, when rebuilt for this snapshot. `None` = no new build (keep the last
    /// ones) — **not** "no views": the UI drops its views when `ModelStatus::fitted` is false and
    /// when it starts a new worker (module doc).
    pub views: Option<Arc<sim::TrainingViews>>,
    /// This worker's activity bookkeeping and teacher time so far (ms; spec §6): the drains, the
    /// fit history and the view builds, against the events' `solve_ms`.
    pub activity_ms: f64,
    pub teacher_ms: f64,
    /// Events the driver's buffer dropped before a drain took them (this worker, so far; a
    /// panic recovery's rebuilt driver adds to it). The provider's own queue
    /// (`providers::surrogate`, 2048 events) counts its drops apart and is not reachable through
    /// `sim::Trainable`; it cannot overflow in practice — the driver empties it after every
    /// evaluation, and a pass tick runs a handful of solves (Task 5 review M5).
    pub dropped: u64,
    /// A new run started since the last snapshot, at this instant (module doc, "Runs"): the UI
    /// puts a run divider before the first event whose `at` is not earlier.
    pub run_started: Option<Instant>,
}

/// What the training views show depends on (the change half of the schedule; Task 4 review I2:
/// not (epoch, samples) alone — an untrusted refit leaves the epoch): the model generation
/// (`TrainableStatus::model_gen`: every fit, install, restore, forget and policy change), and
/// what moves between generations — the store (a kept fallback adds samples and dots, a fence
/// release moves states) and the gate (a failed audit untrusts the model without a refit).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ViewKey {
    model_gen: u64,
    samples: usize,
    trusted: bool,
    quarantined: usize,
    superseded: usize,
    box_excluded: usize,
    unfitted: usize,
}

impl ViewKey {
    fn of(m: &ModelStatus) -> ViewKey {
        ViewKey {
            model_gen: m.model_gen,
            samples: m.samples,
            trusted: m.trusted,
            quarantined: m.quarantined,
            superseded: m.superseded,
            box_excluded: m.box_excluded,
            unfitted: m.unfitted,
        }
    }
}

/// The worker's activity state (module doc).
#[derive(Default)]
pub(super) struct ActivityWork {
    /// The UI shows the training views ([`Command::SetActivityViews`]).
    pub(super) views_on: bool,
    /// The fit the last [`FitPoint`] was taken from: (epoch, the fit summary bitwise).
    last_fit: Option<(u64, (usize, usize, u64, u64))>,
    /// The last build: what it showed (`None` after the views were turned on again), when it
    /// ended and its `build_ms`.
    built_key: Option<ViewKey>,
    built_at: Option<(Instant, f64)>,
    /// A build deferred by the interval is due at this instant (module doc; [`Worker::views_wake`]).
    views_due_at: Option<Instant>,
    /// The latest run start not yet published, and the current run's start (module doc, "Runs").
    run_started: Option<Instant>,
    run_start: Option<Instant>,
    /// Drained events waiting for the next publish, oldest first.
    pending: Vec<sim::TeachEvent>,
    /// Drops counted by drivers this worker replaced (a panic recovery).
    dropped_before: u64,
    /// The driver step stamped on the last judged audit event ([`ModelStatus::last_audit_step`]).
    pub(super) last_audit_step: Option<u64>,
    activity_ms: f64,
    teacher_ms: f64,
}

impl ActivityWork {
    /// A worker's activity state; its first run starts now.
    pub(super) fn new() -> ActivityWork {
        let now = Instant::now();
        ActivityWork { run_started: Some(now), run_start: Some(now), ..ActivityWork::default() }
    }

    /// [`Command::SetActivityViews`]: turning the views on again forgets what the last build
    /// showed (the UI may have dropped it), not when it ran.
    pub(super) fn set_views(&mut self, on: bool) {
        if on && !self.views_on {
            self.built_key = None;
        }
        self.views_on = on;
        if !on {
            self.views_due_at = None;
        }
    }

    /// A new run starts (module doc, "Runs"): a scene replacement or a panic recovery.
    pub(super) fn new_run(&mut self) {
        let now = Instant::now();
        (self.run_started, self.run_start) = (Some(now), Some(now));
        self.last_audit_step = None;
    }
}

impl Worker {
    /// When an idle worker must wake for a deferred view build (module doc; the loop waits until
    /// then instead of blocking for a command).
    pub(super) fn views_wake(&self) -> Option<Instant> {
        self.activity.views_due_at.filter(|_| self.activity.views_on)
    }

    /// Moves the driver's teaching events into the pending list (module doc, "The events");
    /// whether there were any. Latches the step of the last judged audit (its driver stamp:
    /// [`ModelStatus::last_audit_step`]).
    pub(super) fn drain_teach_events(&mut self) -> bool {
        let t = Instant::now();
        let events = self.dy.take_teach_events();
        let any = !events.is_empty();
        let a = &mut self.activity;
        for e in &events {
            a.teacher_ms += e.solve_ms;
            // (an old run's audit, collected at its replacement, names an old-run step: skipped)
            let this_run = a.run_start.is_none_or(|t| e.at >= t);
            if e.origin == sim::activity::TeachOrigin::Audit && e.audit.is_some() && this_run {
                a.last_audit_step = e.step.or(a.last_audit_step);
            }
        }
        a.pending.extend(events);
        a.activity_ms += t.elapsed().as_secs_f64() * 1e3;
        any
    }

    /// A panic recovery is about to replace the driver: its undrained events and its drop count
    /// are kept (the events may be partial if it panicked mid-step; a failed drain is ignored),
    /// and a new run starts.
    pub(super) fn keep_activity_of_old_driver(&mut self) {
        let _ = catch_unwind(AssertUnwindSafe(|| self.drain_teach_events()));
        self.activity.dropped_before += self.dy.teach_events_dropped();
        self.activity.new_run();
    }

    /// The activity delta of one publish (called by `publish` after the model readouts, whose
    /// events were drained before them): the pending events, a fit point when the fit changed,
    /// the training views when due (module doc).
    pub(super) fn activity_delta(&mut self, model: Option<&ModelStatus>) -> ActivityDelta {
        let t = Instant::now();
        let mut d = ActivityDelta {
            events: std::mem::take(&mut self.activity.pending),
            run_started: self.activity.run_started.take(),
            ..ActivityDelta::default()
        };
        // set again below while a build waits for its interval
        self.activity.views_due_at = None;
        if let Some(m) = model.filter(|m| m.fitted) {
            if let Some((n, q, e, f)) = m.fit {
                let key = (m.epoch, (n, q, e.to_bits(), f.to_bits()));
                if self.activity.last_fit != Some(key) {
                    self.activity.last_fit = Some(key);
                    d.fit_points.push(FitPoint { epoch: m.epoch, samples: n, rmse_e: e, rmse_f: f, trusted: m.trusted, at: Instant::now() });
                }
            }
            let key = ViewKey::of(m);
            let a = &self.activity;
            let wanted = a.views_on && a.built_key.as_ref() != Some(&key);
            let due_at = a.built_at.map(|(at, ms)| at + views_interval(ms));
            if wanted && due_at.is_some_and(|t| Instant::now() < t) {
                // deferred: the loop wakes an idle worker then (module doc)
                self.activity.views_due_at = due_at;
            } else if wanted {
                let tb = Instant::now();
                let views = trainable(&mut self.dy).and_then(|s| s.training_views());
                let ms = views.as_ref().map_or(tb.elapsed().as_secs_f64() * 1e3, |v| v.build_ms);
                // a refused build (no model after all) also waits out its interval
                self.activity.built_at = Some((Instant::now(), ms));
                if let Some(v) = views {
                    self.activity.built_key = Some(key);
                    d.views = Some(Arc::new(v));
                }
            }
        }
        let a = &mut self.activity;
        a.activity_ms += t.elapsed().as_secs_f64() * 1e3;
        d.activity_ms = a.activity_ms;
        d.teacher_ms = a.teacher_ms;
        d.dropped = a.dropped_before + self.dy.teach_events_dropped();
        d
    }
}

#[cfg(test)]
mod schedule_tests {
    use super::*;

    /// Task 4 cost ruling item 3: max(2 s, 50 × build_ms) — a warm build (2.3 ms at S = 4) keeps
    /// the 2 s floor, a cold S = 4 build (3.1 s) waits 155 s (2 %), and a bad time never panics.
    #[test]
    fn the_views_interval_is_the_duty_backstop() {
        assert_eq!(views_interval(0.0), VIEWS_MIN_INTERVAL);
        assert_eq!(views_interval(2.3), VIEWS_MIN_INTERVAL);
        assert_eq!(views_interval(40.0), VIEWS_MIN_INTERVAL);
        assert_eq!(views_interval(60.0), Duration::from_secs(3));
        assert_eq!(views_interval(3100.0), Duration::from_secs(155));
        for bad in [f64::NAN, f64::INFINITY, -5.0] {
            assert_eq!(views_interval(bad), VIEWS_MIN_INTERVAL, "{bad}");
        }
        assert_eq!(views_interval(1e12), Duration::from_secs(3600));
        // the duty at the interval is at most 2 % once the build outlasts 40 ms
        for ms in [40.0, 100.0, 3100.0, 3800.0] {
            let duty = ms / 1e3 / views_interval(ms).as_secs_f64();
            assert!(duty <= 1.0 / VIEWS_DUTY_FACTOR + 1e-12, "{ms} ms: {duty}");
        }
    }
}
