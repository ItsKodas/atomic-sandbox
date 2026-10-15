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
//! — so the log is in arrival order (`at`), which is step order within a run; the UI marks the
//! run boundary where the snapshot's step goes back.
//!
//! # The training views' schedule (Task 4 physics review, cost ruling)
//! Built on the worker (not off-thread: ruling item 5), only while
//! [`Command::SetActivityViews`] is on and the model is fitted, only when what they show may have
//! changed ([`ViewKey`]), and never sooner than [`views_interval`] after the last build:
//! max([`VIEWS_MIN_INTERVAL`], [`VIEWS_DUTY_FACTOR`] × the last build's `build_ms`) — at most
//! 2 % of the worker in the long run. The provider caches the expensive curves per model
//! generation, so a rebuild at the same generation is cheap (S = 4: 3.1 s cold, 2.3 ms warm,
//! Task 4); the factor keeps a cold S = 4 build from recurring every 2 s. Turning the views on
//! again forgets the key (the next due publish rebuilds even when nothing changed) but not the
//! interval.
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
    #[cfg_attr(not(test), expect(dead_code, reason = "the Activity tab reads it (activity Tasks 6-9)"))]
    pub events: Vec<sim::TeachEvent>,
    /// New fit-history points since the last publish (one per new fit).
    pub fit_points: Vec<FitPoint>,
    /// The training views, when rebuilt for this snapshot (`None`: keep the last ones; the UI
    /// drops them when `ModelStatus::fitted` is false).
    pub views: Option<Arc<sim::TrainingViews>>,
    /// This worker's activity bookkeeping and teacher time so far (ms; spec §6): the drains, the
    /// fit history and the view builds, against the events' `solve_ms`.
    pub activity_ms: f64,
    pub teacher_ms: f64,
    /// Events the driver's buffer dropped before a drain took them (this worker, so far; a
    /// panic recovery's rebuilt driver adds to it).
    pub dropped: u64,
}

/// What the training views show depends on (the change half of the schedule; Task 4 review I2:
/// not (epoch, samples) alone — an untrusted refit leaves the epoch): the store (samples and
/// their states), the fit (its summary, bitwise), the gate (trust), the policy (cadence, draft)
/// and the refits the probe has counted (a refit that reproduces the same summary still makes a
/// new model generation).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ViewKey {
    epoch: u64,
    samples: usize,
    fit: Option<(usize, usize, u64, u64)>,
    trusted: bool,
    quarantined: usize,
    superseded: usize,
    box_excluded: usize,
    unfitted: usize,
    audit_every: usize,
    draft: bool,
    refits: u64,
}

impl ViewKey {
    fn of(m: &ModelStatus, refits: u64) -> ViewKey {
        ViewKey {
            epoch: m.epoch,
            samples: m.samples,
            fit: m.fit.map(|(n, q, e, f)| (n, q, e.to_bits(), f.to_bits())),
            trusted: m.trusted,
            quarantined: m.quarantined,
            superseded: m.superseded,
            box_excluded: m.box_excluded,
            unfitted: m.unfitted,
            audit_every: m.audit_every,
            draft: m.draft,
            refits,
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
    /// [`Command::SetActivityViews`]: turning the views on again forgets what the last build
    /// showed (the UI may have dropped it), not when it ran.
    pub(super) fn set_views(&mut self, on: bool) {
        if on && !self.views_on {
            self.built_key = None;
        }
        self.views_on = on;
    }
}

impl Worker {
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
            if e.origin == sim::activity::TeachOrigin::Audit && e.audit.is_some() {
                a.last_audit_step = e.step.or(a.last_audit_step);
            }
        }
        a.pending.extend(events);
        a.activity_ms += t.elapsed().as_secs_f64() * 1e3;
        any
    }

    /// A panic recovery is about to replace the driver: its undrained events and its drop count
    /// are kept (the events may be partial if it panicked mid-step; a failed drain is ignored).
    pub(super) fn keep_activity_of_old_driver(&mut self) {
        let _ = catch_unwind(AssertUnwindSafe(|| self.drain_teach_events()));
        self.activity.dropped_before += self.dy.teach_events_dropped();
    }

    /// The activity delta of one publish (called by `publish` after the model readouts, whose
    /// events were drained before them): the pending events, a fit point when the fit changed,
    /// the training views when due (module doc).
    pub(super) fn activity_delta(&mut self, model: Option<&ModelStatus>) -> ActivityDelta {
        let t = Instant::now();
        let mut d = ActivityDelta { events: std::mem::take(&mut self.activity.pending), ..ActivityDelta::default() };
        if let Some(m) = model.filter(|m| m.fitted) {
            if let Some((n, q, e, f)) = m.fit {
                let key = (m.epoch, (n, q, e.to_bits(), f.to_bits()));
                if self.activity.last_fit != Some(key) {
                    self.activity.last_fit = Some(key);
                    d.fit_points.push(FitPoint { epoch: m.epoch, samples: n, rmse_e: e, rmse_f: f, trusted: m.trusted, at: Instant::now() });
                }
            }
            let key = ViewKey::of(m, self.probe.view().refits);
            let a = &self.activity;
            let due = a.views_on
                && a.built_key.as_ref() != Some(&key)
                && a.built_at.is_none_or(|(at, ms)| at.elapsed() >= views_interval(ms));
            if due {
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
