//! The activity ledger (activity spec §4.3 A2–A3): one `TeachEvent` per teaching solve in a
//! bounded queue the worker drains (through `sim::Dynamics`, which stamps each with its step),
//! and the model's lifetime `SolveCounts`. Nothing here decides anything: the events are pushed
//! after the policy has decided, from what it decided.
//!
//! Who pushes: `teach` (every fallback, audit in the step and relax verification, and a teacher
//! error other than a cancel as `Failed`), `evaluate` (rule 0, the cell check: `NotSolved`, not
//! counted), the background auditor's judgement (`audit.rs`: judged, failed or discarded — a
//! cancelled auditor solve of an older generation pushes nothing and is not counted) and the
//! sampling pass (`crate::sampling`: every reference multiplicity, sample, failure and skip).

use super::*;
use std::collections::VecDeque;

/// Events held for the worker's next drain; older ones are dropped and counted.
pub const EVENT_QUEUE: usize = 2048;

#[derive(Debug, Default)]
pub(super) struct Ledger {
    pub(super) events: VecDeque<TeachEvent>,
    pub(super) dropped: u64,
    pub(super) counts: SolveCounts,
}

impl SurrogateKs {
    /// Records one teaching solve (or a cell-guard skip): counted (`SolveCounts::count` leaves out
    /// what is not a solve and rule-3 solves, spec A7), queued.
    pub(crate) fn push_event(&mut self, e: TeachEvent) {
        self.ledger.counts.count(&e);
        if self.ledger.events.len() == EVENT_QUEUE {
            self.ledger.events.pop_front();
            self.ledger.dropped += 1;
        }
        self.ledger.events.push_back(e);
    }

    /// The model's lifetime teaching solves (spec A2): since the provider was built, the last
    /// forget, or as loaded from a model file (`unrecorded_before` when it carried no record).
    pub fn solve_counts(&self) -> SolveCounts {
        self.ledger.counts
    }

    /// Events the queue dropped because the worker did not drain it in time.
    pub fn events_dropped(&self) -> u64 {
        self.ledger.dropped
    }

    pub(super) fn take_events(&mut self) -> Vec<TeachEvent> {
        self.ledger.events.drain(..).collect()
    }
}

/// Who asked the teacher for a step's solve (activity spec A3): an audit (in the step), the relax
/// verification, else a fallback. A relax verification whose gate refuses (rules 1–4) carries the
/// gate's rule, so it is logged as origin Fallback with that rule, not as a check, and the
/// `verification` count misses it (plan review N2): deliberate — the gate's reason is the more
/// specific one.
pub(super) fn origin_of(audit: bool, rule: TeachRule) -> TeachOrigin {
    if audit {
        TeachOrigin::Audit
    } else if rule == TeachRule::RelaxVerification {
        TeachOrigin::Verification
    } else {
        TeachOrigin::Fallback
    }
}
