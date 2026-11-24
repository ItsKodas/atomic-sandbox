//! What a running refit is doing now, stage by stage (the refit picture; spec
//! `docs/superpowers/specs/2026-11-22-refit-visualisation-design.md`, plan
//! `docs/superpowers/plans/2026-11-25-refit-visualisation.md`). Display only: the counters are
//! relaxed atomics written by the refit's threads ([`RefitMeter`], held by the
//! [`crate::ActivityProbe`]) and read once per frame as a [`RefitLive`]; nothing here reaches a
//! fit or a verdict.
//!
//! # The stages ([`LiveStage`])
//!
//! A refit's time is not spent per sample (measured on a channel collection, the plan's table):
//! - **Gathering** ([`LiveStage::Gather`]): the samples described into the fit's statistics, one
//!   block per sample (`surrogate::Accumulator::add_many`).
//! - **Learning the mix W** (a channel collection, when it is due): its own gathering (the
//!   low-order fit's samples) and its own search; [`RefitLive::mix`] is set while it runs.
//! - **Solving the model** ([`LiveStage::Solve`]): the hyperparameter search, round by round, each
//!   round's O(Q′³) decompositions of the Q′ × Q′ matrix counted as they finish (a library kernel
//!   is one step).
//! - **Checking each sample** ([`LiveStage::Check`]): the leave-one-out pass, one sample at a time
//!   on the worker threads, each with its ratio to its bound ([`ActivityProbe::refit_ratios`]).
//! - **The other checks** ([`LiveStage::Other`]): the passes without a per-sample counter (the
//!   training residuals, the active set, the null space).
//! - **Trust** ([`RefitLive::trust`]): in a trust recovery, its round and the samples it holds
//!   out; every round runs the stages above again.
//!
//! # The estimate ([`RefitLive::stage_left_s`])
//!
//! Only the running stage's, and only from its own counter: a counted stage (gathering, checking)
//! at a rate of done / seconds in the stage, once [`ESTIMATE_AFTER_S`] have passed and one item is
//! done. The search has no total (it stops when its levels settle: 3–6 rounds typically), so it
//! has no time left — its rounds, decompositions and the mean time of one are shown instead, and
//! the current decomposition's share of that mean fills its square ([`RefitLive::decomposition_fill`]).
//! A stage that has not started has no number.
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering::Relaxed};
use std::sync::Mutex;

/// A counted stage's time left shows from this many seconds into it (a rate from the first
/// block would swing).
pub const ESTIMATE_AFTER_S: f64 = 1.0;
/// The decomposition square's fill stops short of full until the decomposition reports.
pub const DECOMPOSITION_FILL_MAX: f64 = 0.95;
/// Slots of [`RefitLive::spent_s`]: starting, gathering, solving, checking, the other passes
/// ([`LiveStage::slot`]) and learning the mix W ([`MIX_SLOT`]).
pub const STAGES: usize = 6;
/// The slot of [`RefitLive::spent_s`] the mix W's learn books its gathering and search to.
pub const MIX_SLOT: usize = 5;

/// A pass without a per-sample counter ([`LiveStage::Other`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OtherPass {
    /// The training residuals of every fitted sample (one prediction each).
    Residuals,
    /// The active set (rule 5e): the Gram of the fitted atoms' features and its eigenvectors.
    ActiveSet,
    /// The null space of the fit (rule 5d).
    NullSpace,
    /// Counting the fitted samples and columns per environment key (the trust floors, the body-4
    /// degree's floor, the class pass's sums).
    Counts,
}

/// What the refit is doing now (module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LiveStage {
    /// Begun, no stage reported yet.
    #[default]
    Starting,
    Gather,
    Solve,
    Check,
    Other(OtherPass),
}

impl LiveStage {
    fn code(self) -> u8 {
        match self {
            LiveStage::Starting => 0,
            LiveStage::Gather => 1,
            LiveStage::Solve => 2,
            LiveStage::Check => 3,
            LiveStage::Other(OtherPass::Residuals) => 4,
            LiveStage::Other(OtherPass::ActiveSet) => 5,
            LiveStage::Other(OtherPass::NullSpace) => 6,
            LiveStage::Other(OtherPass::Counts) => 7,
        }
    }

    fn of(code: u8) -> LiveStage {
        match code {
            1 => LiveStage::Gather,
            2 => LiveStage::Solve,
            3 => LiveStage::Check,
            4 => LiveStage::Other(OtherPass::Residuals),
            5 => LiveStage::Other(OtherPass::ActiveSet),
            6 => LiveStage::Other(OtherPass::NullSpace),
            7 => LiveStage::Other(OtherPass::Counts),
            _ => LiveStage::Starting,
        }
    }

    /// Its slot in [`RefitLive::spent_s`]: starting, gathering, solving, checking, other.
    pub fn slot(self) -> usize {
        match self {
            LiveStage::Starting => 0,
            LiveStage::Gather => 1,
            LiveStage::Solve => 2,
            LiveStage::Check => 3,
            LiveStage::Other(_) => 4,
        }
    }
}

/// A trust recovery's round (module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrustRound {
    /// The quarantine round (1-based); 0 while the recovery tries its basis retries first.
    pub round: u32,
    /// Its rounds at most (`providers::surrogate::QUARANTINE_ROUNDS`).
    pub max: u32,
    /// Samples held out of the fit now.
    pub held: u32,
}

/// The running refit's stage counters, read once per frame (module doc).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RefitLive {
    pub stage: LiveStage,
    /// Seconds in the stage so far.
    pub stage_s: f64,
    /// A counted stage's items done of its total (gathering: samples described; checking: samples
    /// checked); 0 of 0 otherwise.
    pub done: u32,
    pub total: u32,
    /// The stage belongs to learning the mix W (a channel collection's low-order fit).
    pub mix: bool,
    /// The search's round (1-based; 0 before its first) and decompositions finished in this search.
    pub round: u32,
    pub decompositions: u32,
    /// The matrix's size Q′ (0 before a search).
    pub q: u32,
    /// The mean seconds of one decomposition in this search (NaN before the first), and the
    /// seconds since the last one finished (or the search started).
    pub decomposition_s: f64,
    pub since_decomposition_s: f64,
    /// Checked samples beyond their bound so far (ratio > 1, or undefined).
    pub beyond: u32,
    /// The trust recovery's round, once it entered it.
    pub trust: Option<TrustRound>,
    /// Seconds spent per slot ([`STAGES`]: [`LiveStage::slot`], the mix W's learn in
    /// [`MIX_SLOT`] instead of its own stages'), the running stage's included.
    pub spent_s: [f64; STAGES],
}

impl RefitLive {
    /// The running stage's time left (module doc): `None` for a stage without a total, before
    /// [`ESTIMATE_AFTER_S`] or before its first item.
    pub fn stage_left_s(&self) -> Option<f64> {
        if !matches!(self.stage, LiveStage::Gather | LiveStage::Check) || self.total == 0 {
            return None;
        }
        if self.done >= self.total {
            return Some(0.0);
        }
        if self.done == 0 || self.stage_s < ESTIMATE_AFTER_S {
            return None;
        }
        Some((self.total - self.done) as f64 * self.stage_s / self.done as f64)
    }

    /// The running stage's share done (a counted stage), else `None`.
    pub fn stage_fraction(&self) -> Option<f64> {
        (matches!(self.stage, LiveStage::Gather | LiveStage::Check) && self.total > 0).then(|| (self.done as f64 / self.total as f64).clamp(0.0, 1.0))
    }

    /// The current decomposition's square fill (module doc): the seconds since the last one over
    /// the mean of one, at most [`DECOMPOSITION_FILL_MAX`]; `None` before the first finished (no
    /// measure of one yet) or outside the search.
    pub fn decomposition_fill(&self) -> Option<f64> {
        (self.stage == LiveStage::Solve && self.decompositions > 0 && self.decomposition_s > 0.0).then(|| (self.since_decomposition_s / self.decomposition_s).clamp(0.0, DECOMPOSITION_FILL_MAX))
    }
}

/// The counters' store (module doc): relaxed atomics, and the checked samples' ratios behind a
/// lock taken once per sample by a worker and once per frame by the display.
#[derive(Debug, Default)]
pub struct RefitMeter {
    stage: AtomicU8,
    /// µs since the probe's origin when the stage started.
    stage_us: AtomicU64,
    done: AtomicU32,
    total: AtomicU32,
    mix: AtomicBool,
    round: AtomicU32,
    decompositions: AtomicU32,
    q: AtomicU32,
    /// When this search started, and its last decomposition finished (µs).
    search_us: AtomicU64,
    decomposition_us: AtomicU64,
    beyond: AtomicU32,
    trust: AtomicBool,
    trust_round: AtomicU32,
    trust_max: AtomicU32,
    trust_held: AtomicU32,
    /// µs spent per finished stage slot.
    spent_us: [AtomicU64; STAGES],
    /// The checked samples' ratios in the pass's order (NaN: not yet).
    ratios: Mutex<Vec<f32>>,
}

impl RefitMeter {
    /// A refit begins at `now_us`: every counter back to none.
    pub fn begin(&self, now_us: u64) {
        self.stage.store(LiveStage::Starting.code(), Relaxed);
        self.stage_us.store(now_us, Relaxed);
        for a in [&self.done, &self.total, &self.round, &self.decompositions, &self.q, &self.beyond, &self.trust_round, &self.trust_max, &self.trust_held] {
            a.store(0, Relaxed);
        }
        self.mix.store(false, Relaxed);
        self.trust.store(false, Relaxed);
        self.search_us.store(now_us, Relaxed);
        self.decomposition_us.store(now_us, Relaxed);
        for s in &self.spent_us {
            s.store(0, Relaxed);
        }
        self.ratios.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    /// The slot the running stage's time goes to.
    fn slot_now(&self) -> usize {
        if self.mix.load(Relaxed) {
            MIX_SLOT
        } else {
            LiveStage::of(self.stage.load(Relaxed)).slot()
        }
    }

    /// The running stage's time so far is booked to its slot, from `now_us` on afresh.
    fn book(&self, now_us: u64) {
        let since = self.stage_us.swap(now_us, Relaxed);
        self.spent_us[self.slot_now()].fetch_add(now_us.saturating_sub(since), Relaxed);
    }

    /// The stage becomes `stage` at `now_us` (the time of the one it leaves is booked).
    fn enter(&self, stage: LiveStage, now_us: u64) {
        self.book(now_us);
        self.stage.store(stage.code(), Relaxed);
    }

    /// `n` more samples of a gathering pass over `total` (a new pass when the stage, the total or
    /// a finished count says so).
    pub fn described(&self, n: usize, total: usize, now_us: u64) {
        let total = total.min(u32::MAX as usize) as u32;
        let fresh = LiveStage::of(self.stage.load(Relaxed)) != LiveStage::Gather || self.total.load(Relaxed) != total || self.done.load(Relaxed) >= total;
        if fresh {
            self.enter(LiveStage::Gather, now_us);
            self.total.store(total, Relaxed);
            self.done.store(0, Relaxed);
        }
        self.done.fetch_add(n.min(u32::MAX as usize) as u32, Relaxed);
    }

    /// Round `round` of a search on a `q` × `q` matrix starts (round 1: a new search).
    pub fn search_round(&self, round: usize, q: usize, now_us: u64) {
        if round <= 1 || LiveStage::of(self.stage.load(Relaxed)) != LiveStage::Solve {
            self.enter(LiveStage::Solve, now_us);
            self.decompositions.store(0, Relaxed);
            self.search_us.store(now_us, Relaxed);
            self.decomposition_us.store(now_us, Relaxed);
        }
        self.round.store(round.min(u32::MAX as usize) as u32, Relaxed);
        self.q.store(q.min(u32::MAX as usize) as u32, Relaxed);
    }

    /// One decomposition finished.
    pub fn decomposition(&self, q: usize, now_us: u64) {
        if LiveStage::of(self.stage.load(Relaxed)) != LiveStage::Solve {
            // a solve outside a search (pinned levels): a search of one round
            self.search_round(1, q, now_us);
        }
        self.decompositions.fetch_add(1, Relaxed);
        self.decomposition_us.store(now_us, Relaxed);
    }

    /// A leave-one-out pass over `total` samples starts.
    pub fn check_begin(&self, total: usize, now_us: u64) {
        self.enter(LiveStage::Check, now_us);
        let total = total.min(u32::MAX as usize);
        self.total.store(total as u32, Relaxed);
        self.done.store(0, Relaxed);
        self.beyond.store(0, Relaxed);
        let mut r = self.ratios.lock().unwrap_or_else(|e| e.into_inner());
        r.clear();
        r.resize(total, f32::NAN);
    }

    /// Sample `k` of the pass is checked: its ratio to its bound (∞ when undefined). Any thread.
    pub fn checked(&self, k: usize, ratio: f64) {
        if let Some(x) = self.ratios.lock().unwrap_or_else(|e| e.into_inner()).get_mut(k) {
            *x = ratio as f32;
        }
        if ratio.is_nan() || ratio > 1.0 {
            self.beyond.fetch_add(1, Relaxed);
        }
        self.done.fetch_add(1, Relaxed);
    }

    /// A pass without a counter starts.
    pub fn other(&self, pass: OtherPass, now_us: u64) {
        self.enter(LiveStage::Other(pass), now_us);
        self.total.store(0, Relaxed);
        self.done.store(0, Relaxed);
    }

    /// Learning the mix W starts (`true`) or ends, at `now_us` (the time before is booked to the
    /// slot it ran in).
    pub fn mix(&self, on: bool, now_us: u64) {
        self.book(now_us);
        self.mix.store(on, Relaxed);
    }

    /// The trust recovery is in `round` of up to `max`, holding `held` samples out.
    pub fn trust_round(&self, round: usize, max: usize, held: usize) {
        self.trust_round.store(round.min(u32::MAX as usize) as u32, Relaxed);
        self.trust_max.store(max.min(u32::MAX as usize) as u32, Relaxed);
        self.trust_held.store(held.min(u32::MAX as usize) as u32, Relaxed);
        self.trust.store(true, Relaxed);
    }

    /// The counters at `now_us`.
    pub fn view(&self, now_us: u64) -> RefitLive {
        let stage = LiveStage::of(self.stage.load(Relaxed));
        let stage_s = now_us.saturating_sub(self.stage_us.load(Relaxed)) as f64 * 1e-6;
        let decompositions = self.decompositions.load(Relaxed);
        let (search, last) = (self.search_us.load(Relaxed), self.decomposition_us.load(Relaxed));
        let mut spent_s = [0.0; STAGES];
        for (s, a) in spent_s.iter_mut().zip(&self.spent_us) {
            *s = a.load(Relaxed) as f64 * 1e-6;
        }
        spent_s[self.slot_now()] += stage_s;
        RefitLive {
            stage,
            stage_s,
            done: self.done.load(Relaxed),
            total: self.total.load(Relaxed),
            mix: self.mix.load(Relaxed),
            round: self.round.load(Relaxed),
            decompositions,
            q: self.q.load(Relaxed),
            decomposition_s: if decompositions > 0 { last.saturating_sub(search) as f64 * 1e-6 / decompositions as f64 } else { f64::NAN },
            since_decomposition_s: now_us.saturating_sub(last) as f64 * 1e-6,
            beyond: self.beyond.load(Relaxed),
            trust: self.trust.load(Relaxed).then(|| TrustRound { round: self.trust_round.load(Relaxed), max: self.trust_max.load(Relaxed), held: self.trust_held.load(Relaxed) }),
            spent_s,
        }
    }

    /// The checked samples' ratios of the running (or last) leave-one-out pass, in its order, into
    /// `out` (NaN: not checked yet).
    pub fn ratios_into(&self, out: &mut Vec<f32>) {
        out.clear();
        out.extend_from_slice(&self.ratios.lock().unwrap_or_else(|e| e.into_inner()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: u64 = 1_000_000;

    #[test]
    fn gathering_counts_a_pass_and_estimates_from_its_own_rate() {
        let m = RefitMeter::default();
        m.begin(0);
        assert_eq!(m.view(0).stage, LiveStage::Starting);
        m.described(100, 1000, S / 2);
        let v = m.view(S / 2);
        assert_eq!((v.stage, v.done, v.total), (LiveStage::Gather, 100, 1000));
        assert_eq!(v.stage_left_s(), None, "no estimate in the stage's first second");
        m.described(100, 1000, 2 * S);
        let v = m.view(2 * S + S / 2);
        // 200 of 1000 in 2 s of the stage: 800 more at 100/s
        assert_eq!(v.done, 200);
        assert!((v.stage_s - 2.0).abs() < 1e-9);
        assert!((v.stage_left_s().unwrap() - 8.0).abs() < 1e-9);
        assert_eq!(v.stage_fraction(), Some(0.2));
        // the starting half second was booked when gathering began
        assert!((v.spent_s[0] - 0.5).abs() < 1e-9 && (v.spent_s[1] - 2.0).abs() < 1e-9);
        // a pass over another total (the W learn's, then the fit's) starts again at 0
        m.described(5, 40, 3 * S);
        let v = m.view(3 * S);
        assert_eq!((v.done, v.total), (5, 40));
        // a finished pass followed by one of the same total starts again too
        m.described(35, 40, 4 * S);
        assert_eq!(m.view(4 * S).stage_left_s(), Some(0.0));
        m.described(1, 40, 5 * S);
        assert_eq!(m.view(5 * S).done, 1);
    }

    #[test]
    fn a_search_counts_rounds_and_decompositions_and_has_no_time_left() {
        let m = RefitMeter::default();
        m.begin(0);
        m.search_round(1, 4140, S);
        let v = m.view(S);
        assert_eq!((v.stage, v.round, v.q, v.decompositions), (LiveStage::Solve, 1, 4140, 0));
        assert!(v.decomposition_s.is_nan() && v.decomposition_fill().is_none(), "no measure before the first");
        m.decomposition(4140, 3 * S);
        m.decomposition(4140, 5 * S);
        m.search_round(2, 4140, 5 * S);
        let v = m.view(6 * S);
        assert_eq!((v.round, v.decompositions), (2, 2));
        assert!((v.decomposition_s - 2.0).abs() < 1e-9, "4 s for 2");
        assert!((v.decomposition_fill().unwrap() - 0.5).abs() < 1e-9, "1 s into a 2 s decomposition");
        assert_eq!(v.stage_left_s(), None, "the search has no total");
        assert_eq!(m.view(60 * S).decomposition_fill(), Some(DECOMPOSITION_FILL_MAX));
        // a new search (round 1) starts its counts again
        m.search_round(1, 4140, 7 * S);
        assert_eq!(m.view(7 * S).decompositions, 0);
    }

    #[test]
    fn checking_fills_the_ratios_from_any_thread_and_counts_those_beyond() {
        let m = std::sync::Arc::new(RefitMeter::default());
        m.begin(0);
        m.check_begin(6, S);
        let mut r = Vec::new();
        m.ratios_into(&mut r);
        assert_eq!(r.len(), 6);
        assert!(r.iter().all(|x| x.is_nan()));
        std::thread::scope(|s| {
            for (k, ratio) in [(0, 0.3), (5, 2.5), (2, f64::INFINITY)] {
                let m = m.clone();
                s.spawn(move || m.checked(k, ratio));
            }
        });
        m.ratios_into(&mut r);
        assert_eq!(r[0], 0.3);
        assert_eq!(r[5], 2.5);
        assert!(r[2].is_infinite() && r[1].is_nan());
        let v = m.view(2 * S);
        assert_eq!((v.stage, v.done, v.total, v.beyond), (LiveStage::Check, 3, 6, 2));
        assert_eq!(v.stage_fraction(), Some(0.5));
        // a pass of another stage keeps the last ratios for the picture
        m.other(OtherPass::ActiveSet, 3 * S);
        m.ratios_into(&mut r);
        assert_eq!(r.len(), 6);
        let v = m.view(3 * S);
        assert_eq!((v.stage, v.stage_left_s(), v.stage_fraction()), (LiveStage::Other(OtherPass::ActiveSet), None, None));
    }

    /// The mix W's learn books its gathering and search to its own slot, not to the fit's.
    #[test]
    fn the_mix_learn_is_timed_apart_from_the_fit_s_stages() {
        let m = RefitMeter::default();
        m.begin(0);
        m.mix(true, S);
        m.described(10, 10, S);
        m.search_round(1, 50, 2 * S);
        m.decomposition(50, 5 * S);
        m.mix(false, 6 * S);
        m.described(5, 20, 7 * S);
        let v = m.view(8 * S);
        assert!(!v.mix);
        assert!((v.spent_s[MIX_SLOT] - 5.0).abs() < 1e-9, "1 s → 6 s learning W: {:?}", v.spent_s);
        assert!((v.spent_s[LiveStage::Solve.slot()] - 1.0).abs() < 1e-9, "the 6 s → 7 s after W, before gathering: {:?}", v.spent_s);
        assert!((v.spent_s[LiveStage::Gather.slot()] - 1.0).abs() < 1e-9, "the fit's own gathering: {:?}", v.spent_s);
        assert!((v.spent_s[0] - 1.0).abs() < 1e-9);
        assert!((v.spent_s.iter().sum::<f64>() - 8.0).abs() < 1e-9, "every second booked once");
    }

    #[test]
    fn the_trust_round_and_the_mix_flag_are_reported_and_cleared_by_a_new_refit() {
        let m = RefitMeter::default();
        m.begin(0);
        assert_eq!(m.view(0).trust, None);
        m.mix(true, 0);
        m.trust_round(3, 5, 41);
        let v = m.view(0);
        assert!(v.mix);
        assert_eq!(v.trust, Some(TrustRound { round: 3, max: 5, held: 41 }));
        m.check_begin(2, 0);
        m.checked(0, 9.0);
        m.begin(10);
        let v = m.view(10);
        assert!(!v.mix && v.trust.is_none() && v.beyond == 0 && v.stage == LiveStage::Starting);
        let mut r = vec![1.0];
        m.ratios_into(&mut r);
        assert!(r.is_empty());
    }
}
