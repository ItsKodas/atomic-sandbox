//! Waits that give up when the machine stops making progress, not when a wall clock says so.
//!
//! The tests and the input replayer wait for the worker (a sampling pass, an O bond reference,
//! a far-drag replay's solves). How long that takes depends on the load on the machine: a
//! reference that takes 96–130 s alone took longer than a fixed 120 s snapshot gap or a 600 s
//! wait limit when a build or another suite ran beside it, and the test failed although nothing
//! was wrong. What a hang looks like is different: nothing moves. So a wait here watches a
//! *progress token* — the solvers' own counters ([`sim::ActivityProbe`]: SCF steps and finished
//! solves on both slots, refits) plus whatever the caller adds (the simulation step, a pass's
//! done count) — and fails only when the token has not changed for [`STALL`], with a backstop
//! ([`BACKSTOP`]) that no honest run reaches. Waits that *are* the property under test (a latency
//! bound) do not use this: they keep their own limits.
use sim::ActivityProbe;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How long nothing may move before a wait fails. One GPU SCF step on the largest scene takes a few
/// seconds; a CPU solve has no step counter, so it counts as one unit of progress: H₂O 57³ takes
/// ~2 min alone (CLAUDE.md) and several times that under load, so this is 10 minutes. A real hang
/// costs that long to be reported; a loaded machine no longer fails a test.
pub const STALL: Duration = Duration::from_secs(600);

/// An absolute limit for a wait that keeps making progress for ever (a runaway loop).
pub const BACKSTOP: Duration = Duration::from_secs(3600);

/// The solvers' counters that change whenever one of them does work: the worker's identity (a new
/// worker has a new probe), SCF steps and finished solves of the main and side slots, and the
/// refits. Never a clock (`SlotView::elapsed_s` grows even in a hung solve).
pub fn probe_token(probe: &Arc<ActivityProbe>) -> [u64; 8] {
    let v = probe.view();
    [
        Arc::as_ptr(probe) as usize as u64,
        v.main.scf_step as u64,
        v.main.solves,
        v.side.scf_step as u64,
        v.side.solves,
        v.refits,
        v.refit.is_some() as u64,
        v.main.running as u64 | (v.side.running as u64) << 1,
    ]
}

/// Remembers when a token last changed.
#[derive(Debug)]
pub struct StallWatch<T> {
    last: Option<T>,
    since: Instant,
}

impl<T: PartialEq> StallWatch<T> {
    pub fn new(now: Instant) -> Self {
        StallWatch { last: None, since: now }
    }

    /// Records the token seen at `now`; returns how long it has been unchanged.
    pub fn observe(&mut self, token: T, now: Instant) -> Duration {
        if self.last.as_ref() != Some(&token) {
            self.last = Some(token);
            self.since = now;
        }
        now.saturating_duration_since(self.since)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A wait fails only after a whole `STALL` without a change, however long it has run, and
    /// any change restarts the clock — the property that makes the wait load-proof.
    #[test]
    fn the_watch_counts_time_since_the_last_change() {
        let t0 = Instant::now();
        let at = |s: u64| t0 + Duration::from_secs(s);
        let mut w = StallWatch::new(t0);
        assert_eq!(w.observe(1, at(0)), Duration::ZERO);
        assert_eq!(w.observe(1, at(100)), Duration::from_secs(100));
        // a long wait that keeps moving never stalls: 1000 s in, the last change was 10 s ago
        for s in (110..=1000u64).step_by(10) {
            assert!(w.observe(s, at(s)) < STALL);
        }
        // and one that stops does, measured from the last change, not from the start
        let s = STALL.as_secs();
        assert_eq!(w.observe(1000, at(1000 + s - 1)), Duration::from_secs(s - 1));
        assert!(w.observe(1000, at(1000 + s)) >= STALL);
        // a change restarts it
        assert_eq!(w.observe(1001, at(1000 + s + 100)), Duration::ZERO);
    }

    /// The probe's token moves with SCF steps and finished solves, and not with the clock.
    #[test]
    fn the_probe_token_moves_with_work_not_with_time() {
        let probe = ActivityProbe::new();
        let idle = probe_token(&probe);
        assert_eq!(probe_token(&probe), idle, "idle: unchanged");
        let busy = probe.busy(sim::Slot::Main);
        let started = probe_token(&probe);
        assert_ne!(started, idle, "a solve started");
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(probe_token(&probe), started, "a running solve with no step: the clock alone is not progress");
        probe.scf_step(sim::Slot::Main);
        let stepped = probe_token(&probe);
        assert_ne!(stepped, started, "an SCF step");
        drop(busy);
        assert_ne!(probe_token(&probe), stepped, "the solve finished");
        // another worker (a backend switch) is a change too
        assert_ne!(probe_token(&ActivityProbe::new()), idle);
    }
}
