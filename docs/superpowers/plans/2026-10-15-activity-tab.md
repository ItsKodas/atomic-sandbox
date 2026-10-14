# Activity Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make processing visible and the model's learning watchable. The status chip and a toolbar progress bar say what runs and how far it is. A new **Activity** dock tab holds the job list, three live counters, a live solve log (one row per teacher solve), a plain-words trust checklist and the training views (rmse per epoch, parity, pair curves with a ±σ band, coverage, timeline). No view causes a teacher solve.

**Architecture:**
- `sim::activity` (new) holds plain records: the `ActivityProbe` (live solve state as atomics), `TeachEvent` with its `TeachRule` / `TeachOutcome` codes, `SolveCounts`, `TrustCheck`, `StuckInfo` and `TrainingViews`.
- `providers` records what the policy already decides. `GpuKs`/`CpuKs`/`SurrogateKs` take the probe; `SurrogateKs` emits one event per teaching solve, counts solves (persisted as an optional sidecar header key), exposes trust checks and stuck info, and builds training views on demand.
- `sim::Trainable` gains two defaulted methods. The worker drains events, stamps them with the driver step, keeps the fit history, gates view builds and ships an `ActivityDelta` in each snapshot.
- `app::ui::activity` (new) holds the state, the job list, the headline and bar, the plain words, and the tab with its plots.

**Tech Stack:** Rust, egui (hand-painted plots, as `ui::dock::graph`), std atomics. No new crates.

**Spec:** `docs/superpowers/specs/2026-10-15-activity-tab-design.md` (§ numbers below refer to it). Requirements: `docs/superpowers/notes/2026-10-13-m5-user-requests.md` ("Activity view…" and its follow-ups; "Three hydrogens will not reach trust").

**Scheduling:** on branch `activity-tab` (worktree `D:\Projects\Physics-activity`) off `main` `dc03aaa`. Before the user-data milestone. Tasks run in order: 1 → 2 → 3 → 4 (providers), 5 (worker), 6 → 7 → 8 → 9 (UI). Tasks 6–8 depend only on 1 and 5 for types, so 6 may start once 5 lands.

**Model tags** (CLAUDE.md "Model assignment"):
- **[Fable]**: Tasks 2, 3, 4. These touch the surrogate's decision points and numbers: rule codes, trust checks, the read-only "would it answer", the parity data and the ±σ band. Each ends with a **mandatory `physics-reviewer` pass** before it lands. If Fable is unavailable, Opus implements and the review stays mandatory.
- **[Opus]**: Tasks 1, 5, 6, 7, 8, 9 (records and probe plumbing, worker, UI, docs).

## Global Constraints

- **No policy change.** Every rule, threshold, thinning, quarantine and trust decision in `providers::surrogate` behaves exactly as at `dc03aaa`. The additions record decisions and never make them. Gate for Tasks 2–4: `cargo test -p providers --release` and `cargo test -p validate --release --test surrogate` pass with unchanged numbers (rule 6). Never loosen a tolerance; a failure is investigated.
- **No view causes a teacher solve.** `training_views`, `would_answer`, `trust_checks` and `stuck_info` never call the teacher or the auditor. Tests assert `MorseDimer::calls` is unchanged.
- **Rule 1.** Nothing per element. Element symbols come from `units::isotopes::symbol` (display only). `d_box(h)` is the stated H 1s-tail estimate from `box_advice`, unchanged.
- **Rule 3.** Records cross every boundary in Hartree, Bohr, kelvin, ms. mHa, mHa/Bohr and Å are converted in `app::ui::activity` only (Å through `units::bohr_to_angstrom`).
- **Rule 5.** `sim` holds plain data and never knows the solver. `ForceProvider` does not change. The probe is attached by whoever constructs the provider (the worker).
- **Rate limits (spec §6).** Headline, bar and job rows rebuild at most every **100 ms**. Training views build only while the Training section is open, only when (epoch, samples) changed, at most once per **2 s**.
- **Caps.** Provider event queue **2 048** (drops the oldest, counts `dropped`). UI log ring **5 000** rows. Force-parity points **≤ 20 000**. Pair-curve grid **96** points from 0.5 Bohr to r_cut.
- **Timings are reported, never asserted.** The < 1 % criterion of spec §6 is read from the printouts and recorded in the task report and README.
- **Release-only** for SCF/GPU tests (`#[cfg_attr(debug_assertions, ignore)]`); GPU tests skip without an adapter and take `gpu_lock()`.
- **Copy rules.** The plain sentences of spec §5 are pinned verbatim by tests, the user's example included: "atom 0 is outside what the model has learned and is touching the box edge — move it inward". The technical text is always on hover.
- **Commits** end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (a Fable task ends with its own model's line). Use `git add <paths>` then `git commit -o -m "<message>" -- <paths>` (the message before `--`: everything after it is a pathspec). Never pipe a git command. Never push.
- **README / CLAUDE.md** updated in Task 9: the Activity tab, `tab=activity`, the overhead numbers.

## Review Focus

1. **Counters drift from reality.** The user watches "samples in the model" and "teacher solves" tick. An event lost in the early-publish path (`apply_batch` publishes without a tick), double-counted after a panic recovery, or counted for a rule-3 solve makes the numbers disagree with the store. Pinned: `events_arrive_once_in_order_across_ticks_and_early_publishes` (Task 5) and `solve_counts_add_up_over_a_pass_fallbacks_and_audits` (Task 2, kept = samples added).
2. **A long session floods the log.** A box-affected fallback storm emits one event per DFT step for hours. Expected: bounded memory, a responsive tab, counters still exact. Pinned: `the_event_queue_drops_the_oldest_and_counts_them` (Task 2) and `the_log_keeps_5000_rows_and_the_counters_keep_counting` (Task 6).
3. **The chip says "solving" forever.** A solve that errors, is cancelled, or panics must clear the probe. Pinned: `the_probe_clears_on_error_and_cancel` (Task 1, RAII guard) and `a_stale_probe_is_not_a_headline` (Task 7: running with a 30 s-old start and no snapshot change still reads correctly, the elapsed time grows).
4. **"This is why" blames the wrong thing.** The checklist's first ✗, the Model tab line and the event's `Untrusted(key)` must name the condition the provider actually refused on; the box line goes first only under `box_blocks_trust`. Pinned: `the_first_unmet_trust_check_is_the_refusal` (Task 3, every branch) and `the_three_hydrogen_refusal_names_the_box_first` (Task 8, diag-3h fixture).
5. **Views stall the worker or go stale after a backend switch.** At S = 4 (Q′ 1232) a view build costs a variance per curve point. A respawned worker must get `SetActivityViews(true)` again if the section is open. Pinned: `views_are_built_only_while_open_and_at_most_every_2_s` (Task 5), `the_views_command_is_resent_after_a_worker_respawn` (Task 6), and the build-time printout for a 4-species store (`training_views_build_time_is_reported`, Task 4).

---

## File Structure

```
crates/sim/src/activity.rs                 # Task 1 (new): probe, events, codes, counts, checks, stuck info, views records
crates/sim/src/lib.rs                      # Task 1: pub mod activity; re-exports
crates/sim/src/provider.rs                 # Task 1: Trainable::{take_teach_events, training_views}; TrainableStatus::{solves, trust_checks, stuck_info}
crates/sim/src/test_providers.rs           # Task 1: SharedTrainable delegates the two new methods
crates/providers/src/ks.rs                 # Task 1: GpuKs/CpuKs::with_probe
crates/providers/src/surrogate/mod.rs      # Task 1: SurrogateKs::with_probe (refit phase); Task 2: Handoff.reason: Refusal, events in teach/evaluate; Task 3: status fields
crates/providers/src/surrogate/ledger.rs   # Task 2 (new): Ledger, push_event, counts, the queue
crates/providers/src/surrogate/policy.rs   # Task 2: Refusal; Gate/envelope_outside/VarianceVerdict carry codes
crates/providers/src/surrogate/quarantine.rs # Task 2: admit → Admission
crates/providers/src/surrogate/trust.rs    # Task 2: untrusted_check; class_untrusted → Refusal; Task 3: trust_checks, stuck_info
crates/providers/src/surrogate/audit.rs    # Task 2: judge_audit emits its event
crates/providers/src/surrogate/store.rs    # Task 2: keep_sample → bool; counts in save/install/forget
crates/providers/src/surrogate/views.rs    # Task 4 (new): would_answer, training_views
crates/providers/src/surrogate/tests.rs    # Tasks 1–4: tests appended
crates/providers/src/sampling/mod.rs       # Task 2: pass events (record, sample failure, reference, skip, burst T)
crates/surrogate/src/sidecar.rs            # Task 2: optional "teacher_solves" header key
crates/app/src/worker/mod.rs               # Task 5: probe, ActivityDelta in Snapshot, publish drains, SimWorker::probe
crates/app/src/worker/activity.rs          # Task 5 (new): stamping, fit history, view gating, overhead
crates/app/src/worker/commands.rs          # Task 5: Command::SetActivityViews
crates/app/src/worker/model.rs             # Task 5: ModelStatus::{solves, trust_checks, stuck_info}
crates/app/src/worker/tests.rs             # Task 5: tests
crates/app/src/ui/activity/mod.rs          # Task 6 (new): ActivityState, absorb, the tab
crates/app/src/ui/activity/jobs.rs         # Task 6 (new): JobRow, rows, show
crates/app/src/ui/activity/log.rs          # Task 6 (new): the solve log
crates/app/src/ui/activity/headline.rs     # Task 7 (new): headline, bar mode, Throttled
crates/app/src/ui/activity/words.rs        # Task 8 (new): plain wording, d_box
crates/app/src/ui/activity/trust.rs        # Task 8 (new): the checklist
crates/app/src/ui/activity/plots.rs        # Task 9 (new): rmse, parity, pair curves, histograms, timeline
crates/app/src/ui/{mod.rs, dock/mod.rs, icons.rs, overlays.rs, toolbar.rs, search.rs, dock/model.rs, tests.rs}  # Tasks 6–8
crates/app/src/{viewer.rs, screenshot.rs, replay.rs}  # Tasks 6, 9
crates/app/replays/surrogate_far_drag_recovers.replay # Task 9
README.md, CLAUDE.md, docs/superpowers/notes/2026-10-14-ui-redesign-checklist.md     # Task 9
```

---

### Task 1: `sim::activity` records, the probe, and the KS wiring **[Opus]**

**Files:**
- Create: `crates/sim/src/activity.rs`
- Modify: `crates/sim/src/lib.rs`, `crates/sim/src/provider.rs` (trait + `TrainableStatus`), `crates/sim/src/test_providers.rs` (`SharedTrainable`), `crates/providers/src/ks.rs`, `crates/providers/src/surrogate/mod.rs` (`with_probe`, refit phase, the three new status fields as defaults)
- Test: `crates/sim/src/activity.rs` (`#[cfg(test)] mod tests`), `crates/providers/src/ks.rs` tests, `crates/providers/src/surrogate/tests.rs`

**Interfaces:**
- Consumes: nothing new.
- Produces (every later task uses these names):
  - `sim::activity::{ActivityProbe, Slot, Busy, SlotView, ProbeView}`
  - `sim::activity::{TeachOrigin, PassJob, EnvClass, TrustKey, TeachRule, TeachOutcome, TeachEvent, SolveCounts, TrustCheck, StuckInfo}`
  - `sim::activity::{SampleKind, SampleState, ParityPoint, PairCurve, PairHistogram, TrainingViews}`
  - `Trainable::take_teach_events(&mut self) -> Vec<TeachEvent>` and `Trainable::training_views(&mut self) -> Option<TrainingViews>`, both defaulted
  - `TrainableStatus::{solves: SolveCounts, trust_checks: Vec<TrustCheck>, stuck_info: Option<StuckInfo>}`
  - `GpuKs::with_probe(self, Arc<ActivityProbe>, Slot) -> GpuKs`, `CpuKs::with_probe(...)`, `SurrogateKs::with_probe(self, Arc<ActivityProbe>) -> SurrogateKs`

- [ ] **Step 1: Write the failing tests** (`crates/sim/src/activity.rs`, bottom)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_probe_counts_scf_steps_and_clears_at_the_end() {
        let p = ActivityProbe::new();
        assert!(!p.view().main.running);
        {
            let _b = p.busy(Slot::Main);
            p.scf_step(Slot::Main);
            p.scf_step(Slot::Main);
            let v = p.view();
            assert!(v.main.running && !v.side.running);
            assert_eq!(v.main.scf_step, 2);
        }
        let v = p.view();
        assert!(!v.main.running);
        assert_eq!(v.main.solves, 1);
        assert_eq!(v.main.scf_step, 2, "the last solve's count stays readable");
    }

    #[test]
    fn a_new_solve_restarts_the_step_count() {
        let p = ActivityProbe::new();
        drop(p.busy(Slot::Side));
        p.scf_step(Slot::Side);
        let _b = p.busy(Slot::Side);
        assert_eq!(p.view().side.scf_step, 0);
    }

    #[test]
    fn the_refit_phase_is_reported_with_its_sample_count() {
        let p = ActivityProbe::new();
        assert_eq!(p.view().refit, None);
        p.begin_refit(250);
        assert!(matches!(p.view().refit, Some((t, 250)) if t >= 0.0));
        p.end_refit();
        assert_eq!(p.view().refit, None);
    }

    #[test]
    fn solve_counts_total_and_not_kept() {
        let c = SolveCounts { pass: 273, fallback: 70, audit: 18, verification: 1, kept: 343, unrecorded_before: false };
        assert_eq!(c.total(), 362);
        assert_eq!(c.not_kept(), 19);
    }

    #[test]
    fn kept_outcomes_and_counted_events() {
        assert!(TeachOutcome::KeptReleased { released: 1, superseded: 0 }.is_kept());
        assert!(!TeachOutcome::Thinned.is_kept() && !TeachOutcome::Reference.is_kept());
        let mut e = TeachEvent::new(TeachOrigin::Fallback, TeachRule::Grade, TeachOutcome::Kept);
        assert!(e.is_solve() && e.counts_for_model());
        e.rule = TeachRule::OtherSettings;
        assert!(e.is_solve() && !e.counts_for_model(), "spec A7: a rule-3 solve is not spent on the model");
        e.outcome = TeachOutcome::NotSolved;
        assert!(!e.is_solve());
    }
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p sim --release activity`
Expected: compile error, `activity` module missing.

- [ ] **Step 3: Implement `crates/sim/src/activity.rs`**

```rust
//! Plain records of what the providers are doing, for the app's Activity view (spec
//! `2026-10-15-activity-tab-design.md`): the live solve state ([`ActivityProbe`], atomics shared
//! between the solver threads and the render thread), one [`TeachEvent`] per teaching solve with
//! the rule that caused it and what happened to the answer, lifetime [`SolveCounts`], the trust
//! rule's conditions ([`TrustCheck`]), a stuck model's reason ([`StuckInfo`]) and the training
//! views' data ([`TrainingViews`]). Atomic units (Hartree, Bohr), kelvin, milliseconds. Nothing
//! here decides anything: the providers fill these from decisions they already made.
use crate::AuditVerdict;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::Arc;
use std::time::Instant;

/// Which solver a probe slot follows: the stepping teacher, or the background auditor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    Main,
    Side,
}

#[derive(Debug, Default)]
struct SolveSlot {
    running: AtomicBool,
    scf_step: AtomicU32,
    started_us: AtomicU64,
    solves: AtomicU64,
}

/// Live solve state (spec §4.3 A1). Written with relaxed atomics by the solver threads, read by
/// the render thread; the numbers are for display, so no ordering between fields is promised.
#[derive(Debug)]
pub struct ActivityProbe {
    origin: Instant,
    main: SolveSlot,
    side: SolveSlot,
    /// Start of the refit in flight (µs since `origin`), `u64::MAX` when none.
    refit_started_us: AtomicU64,
    refit_samples: AtomicU32,
}

/// Marks a slot running until dropped (an error, a cancel or a panic clears it too).
#[must_use]
pub struct Busy {
    probe: Arc<ActivityProbe>,
    slot: Slot,
}

impl Drop for Busy {
    fn drop(&mut self) {
        let s = self.probe.slot(self.slot);
        s.running.store(false, Relaxed);
        s.solves.fetch_add(1, Relaxed);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SlotView {
    pub running: bool,
    /// SCF step being solved (running) or the last solve's count (idle); 0 for the CPU solver.
    pub scf_step: u32,
    /// Seconds since the running solve started (0 when idle).
    pub elapsed_s: f64,
    pub solves: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ProbeView {
    pub main: SlotView,
    pub side: SlotView,
    /// (seconds since the refit started, samples it fits) while one runs.
    pub refit: Option<(f64, u32)>,
}

impl ActivityProbe {
    pub fn new() -> Arc<ActivityProbe> {
        Arc::new(ActivityProbe {
            origin: Instant::now(),
            main: SolveSlot::default(),
            side: SolveSlot::default(),
            refit_started_us: AtomicU64::new(u64::MAX),
            refit_samples: AtomicU32::new(0),
        })
    }

    fn slot(&self, s: Slot) -> &SolveSlot {
        match s {
            Slot::Main => &self.main,
            Slot::Side => &self.side,
        }
    }

    fn now_us(&self) -> u64 {
        self.origin.elapsed().as_micros() as u64
    }

    /// A solve starts on `slot`; it ends when the guard drops.
    pub fn busy(self: &Arc<Self>, slot: Slot) -> Busy {
        let s = self.slot(slot);
        s.scf_step.store(0, Relaxed);
        s.started_us.store(self.now_us(), Relaxed);
        s.running.store(true, Relaxed);
        Busy { probe: self.clone(), slot }
    }

    /// One SCF step starts (called from the solver's per-step abort poll).
    pub fn scf_step(&self, slot: Slot) {
        self.slot(slot).scf_step.fetch_add(1, Relaxed);
    }

    pub fn begin_refit(&self, samples: usize) {
        self.refit_samples.store(samples.min(u32::MAX as usize) as u32, Relaxed);
        self.refit_started_us.store(self.now_us(), Relaxed);
    }

    pub fn end_refit(&self) {
        self.refit_started_us.store(u64::MAX, Relaxed);
    }

    pub fn view(&self) -> ProbeView {
        let now = self.now_us();
        let slot = |s: &SolveSlot| {
            let running = s.running.load(Relaxed);
            SlotView {
                running,
                scf_step: s.scf_step.load(Relaxed),
                elapsed_s: if running { now.saturating_sub(s.started_us.load(Relaxed)) as f64 * 1e-6 } else { 0.0 },
                solves: s.solves.load(Relaxed),
            }
        };
        let r = self.refit_started_us.load(Relaxed);
        ProbeView {
            main: slot(&self.main),
            side: slot(&self.side),
            refit: (r != u64::MAX).then(|| (now.saturating_sub(r) as f64 * 1e-6, self.refit_samples.load(Relaxed))),
        }
    }
}

/// Who asked the teacher (spec §3.3.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TeachOrigin {
    Pass,
    Fallback,
    Audit,
    /// The relax verification (`Trainable::force_teacher_next`).
    Verification,
}

/// A sampling-pass job.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassJob {
    Reference { z: u8, multiplicity: u32 },
    Perturbation,
    Scan,
    HotBurst,
}

/// The class-wise trust rule's classes (`providers::surrogate::ENV_CLASSES` order).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnvClass {
    Pair,
    ManyBody,
}

/// The trust rule's conditions, in the order `untrusted_reason` tests them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrustKey {
    Samples,
    ResidualDofE,
    ResidualDofF,
    EnergyResidual,
    ForceResidual,
    ForceNoise,
    Audit,
}

/// Why the teacher answered (the fallback policy's decision list, `providers::surrogate`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TeachRule {
    /// 0: the cell check refused the geometry (a pass skip, or the run's refusal).
    Cell,
    NoModel,
    NewSpecies(u8),
    OtherSettings,
    Untrusted(TrustKey),
    Class(EnvClass),
    IsolatedAtom,
    Coordination,
    Spread,
    NoGrade,
    NewEnvironment,
    Grade,
    ForceUncertainty,
    EnergyUncertainty,
    NearQuarantine,
    Audit,
    RelaxVerification,
    Pass(PassJob),
}

/// What happened to the teacher's answer (`providers::surrogate::admit` and the pass).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TeachOutcome {
    Kept,
    /// Kept although box-affected (the model is still being built; it leaves the fit later).
    KeptBoxAffected,
    KeptByVote,
    KeptReleased { released: u32, superseded: u32 },
    Thinned,
    BoxNotKept,
    FenceDisagrees,
    WaitingForPass,
    NotOffered,
    StoreRefused,
    Reference,
    Failed,
    NotSolved,
}

impl TeachOutcome {
    pub fn is_kept(self) -> bool {
        matches!(self, TeachOutcome::Kept | TeachOutcome::KeptBoxAffected | TeachOutcome::KeptByVote | TeachOutcome::KeptReleased { .. })
    }
}

/// One teaching solve (spec §4.3 A3). Energies in Hartree, the face density in e/Bohr³.
#[derive(Clone, Debug, PartialEq)]
pub struct TeachEvent {
    pub at: Instant,
    pub origin: TeachOrigin,
    pub rule: TeachRule,
    /// The atom the rule names, when it names one.
    pub atom: Option<usize>,
    /// The provider's own note for this solve (the technical text, shown on hover).
    pub detail: String,
    pub outcome: TeachOutcome,
    pub face_density: Option<f64>,
    pub box_affected: bool,
    /// E_teacher − E_model where the model was consulted.
    pub gap: Option<f64>,
    pub n_atoms: usize,
    pub solve_ms: f64,
    pub scf_steps: usize,
    pub audit: Option<AuditVerdict>,
    /// A background audit: answered steps between its dispatch and its verdict.
    pub audit_lag: Option<usize>,
    /// A hot-burst geometry's instantaneous kinetic temperature (K).
    pub burst_t: Option<f64>,
}

impl TeachEvent {
    pub fn new(origin: TeachOrigin, rule: TeachRule, outcome: TeachOutcome) -> TeachEvent {
        TeachEvent {
            at: Instant::now(),
            origin,
            rule,
            atom: None,
            detail: String::new(),
            outcome,
            face_density: None,
            box_affected: false,
            gap: None,
            n_atoms: 0,
            solve_ms: 0.0,
            scf_steps: 0,
            audit: None,
            audit_lag: None,
            burst_t: None,
        }
    }

    /// A teacher evaluation happened (a cell-guard skip did not).
    pub fn is_solve(&self) -> bool {
        self.outcome != TeachOutcome::NotSolved
    }

    /// Counted in the model's lifetime solves (spec A7: not a rule-3 solve).
    pub fn counts_for_model(&self) -> bool {
        self.is_solve() && self.rule != TeachRule::OtherSettings
    }
}

/// Lifetime teaching solves of one model (spec §3.3.2, A2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SolveCounts {
    pub pass: u64,
    pub fallback: u64,
    pub audit: u64,
    pub verification: u64,
    pub kept: u64,
    /// The model came from a file without the record: solves before it are unrecorded.
    pub unrecorded_before: bool,
}

impl SolveCounts {
    pub fn total(&self) -> u64 {
        self.pass + self.fallback + self.audit + self.verification
    }

    pub fn not_kept(&self) -> u64 {
        self.total().saturating_sub(self.kept)
    }

    pub fn count(&mut self, e: &TeachEvent) {
        if !e.counts_for_model() {
            return;
        }
        match e.origin {
            TeachOrigin::Pass => self.pass += 1,
            TeachOrigin::Fallback => self.fallback += 1,
            TeachOrigin::Audit => self.audit += 1,
            TeachOrigin::Verification => self.verification += 1,
        }
        if e.outcome.is_kept() {
            self.kept += 1;
        }
    }
}

/// One condition of the trust rule (spec §3.3.4, A4).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrustCheck {
    pub key: TrustKey,
    pub met: bool,
    pub value: f64,
    pub limit: f64,
}

/// A trusted model that keeps falling back (spec A5; `SurrogateKs::stuck` in structured form).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StuckInfo {
    pub rule: TeachRule,
    pub atom: Option<usize>,
    pub steps: usize,
    pub box_streak: bool,
    pub gap: Option<f64>,
    pub face_density: f64,
}

/// A stored sample's origin by its store tag (audits are `Fallback` until sidecar v4, spec A6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleKind {
    Pass,
    Fallback,
    Live,
}

/// Whether a stored sample is in the current fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleState {
    Fitted,
    Unfitted,
    Quarantined,
    Superseded,
    BoxExcluded,
}

/// One sample of the energy parity plot: (E − Σ E_ref)/N for the teacher and the model (Ha/atom).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParityPoint {
    pub dft: f64,
    pub model: f64,
    pub kind: SampleKind,
    pub state: SampleState,
    pub index: usize,
}

/// A species pair's model curve (spec §3.3.5, §9). Hartree and Bohr.
#[derive(Clone, Debug, PartialEq)]
pub struct PairCurve {
    pub z: (u8, u8),
    pub r: Vec<f64>,
    /// E_model(dimer at r) − E_ref(a) − E_ref(b).
    pub energy: Vec<f64>,
    /// The epistemic energy std √(energy_std² − (2 s_E)²); empty without a posterior.
    pub sigma: Vec<f64>,
    /// Per r: `None` when the model would answer this dimer, else the rule that refuses it.
    pub refused: Vec<Option<TeachRule>>,
    /// Stored two-atom samples of this pair: (r, E − E_ref(a) − E_ref(b), state).
    pub dots: Vec<(f64, f64, SampleState)>,
    pub r_cut: f64,
}

/// Pair distances < r_cut over the fitted samples, bins of `bin` Bohr from 0.
#[derive(Clone, Debug, PartialEq)]
pub struct PairHistogram {
    pub z: (u8, u8),
    pub bin: f64,
    pub counts: Vec<u32>,
}

/// The training views' data (spec §3.3.5, A6).
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingViews {
    pub epoch: u64,
    pub samples: usize,
    pub parity_energy: Vec<ParityPoint>,
    /// (teacher, model) per force component of fitted samples (Ha/Bohr), thinned to ≤ 20 000.
    pub parity_force: Vec<[f32; 2]>,
    pub force_points_total: usize,
    pub curves: Vec<PairCurve>,
    pub pairs: Vec<PairHistogram>,
    /// The per-dimer aleatoric floor 2·s_E (Ha) and the rule-7 per-atom tier (Ha/atom).
    pub dimer_floor: f64,
    pub energy_tier: f64,
    pub build_ms: f64,
}

/// Max force-parity points (spec §3.3.5).
pub const FORCE_PARITY_CAP: usize = 20_000;
/// Points per pair curve and its inner end (Bohr).
pub const CURVE_POINTS: usize = 96;
pub const CURVE_R_MIN: f64 = 0.5;
```

In `crates/sim/src/lib.rs` add `pub mod activity;` and `pub use activity::{ActivityProbe, ProbeView, Slot, SolveCounts, TeachEvent, TrainingViews};`.

- [ ] **Step 4: Extend `Trainable` and `TrainableStatus`** (`crates/sim/src/provider.rs`)

In `pub trait Trainable`, after `take_notes`:

```rust
    /// The teaching solves since the last call, oldest first (`crate::activity::TeachEvent`; the
    /// Activity tab's solve log). A provider without a teacher has none.
    fn take_teach_events(&mut self) -> Vec<crate::activity::TeachEvent> {
        Vec::new()
    }
    /// The training views' data built from what the provider holds, without a teacher solve
    /// (`crate::activity::TrainingViews`); `None` without a fitted model.
    fn training_views(&mut self) -> Option<crate::activity::TrainingViews> {
        None
    }
```

In `pub struct TrainableStatus`, after `class_loo`:

```rust
    /// The model's lifetime teaching solves (`crate::activity::SolveCounts`).
    pub solves: crate::activity::SolveCounts,
    /// The trust rule's conditions in its order; empty before the first fit.
    pub trust_checks: Vec<crate::activity::TrustCheck>,
    /// `stuck` in structured form (the plain wording's input).
    pub stuck_info: Option<crate::activity::StuckInfo>,
```

In `crates/sim/src/test_providers.rs`, `impl Trainable for SharedTrainable<P>` delegates both:

```rust
    fn take_teach_events(&mut self) -> Vec<crate::activity::TeachEvent> {
        Trainable::take_teach_events(&mut *self.lock())
    }

    fn training_views(&mut self) -> Option<crate::activity::TrainingViews> {
        Trainable::training_views(&mut *self.lock())
    }
```

In `providers/src/surrogate/mod.rs` `status()`, fill the new fields with `solves: SolveCounts::default(), trust_checks: Vec::new(), stuck_info: None` (Tasks 2–3 replace them).

- [ ] **Step 5: Wire the probe into `GpuKs`, `CpuKs`, `SurrogateKs`**

`crates/providers/src/ks.rs`: add the field `probe: Option<(Arc<ActivityProbe>, Slot)>` to both structs (`None` in their constructors) and

```rust
    /// Report this solver's live state on `probe`'s `slot` (the Activity view, spec §4.3 A1).
    pub fn with_probe(mut self, probe: Arc<sim::ActivityProbe>, slot: sim::Slot) -> Self {
        self.probe = Some((probe, slot));
        self
    }
```

In `GpuKs::evaluate`, after `self.check_vram(scene)?;` and the first `should_abort()` check:

```rust
        // the Activity probe (spec A1): running for the whole attempt, cleared by the guard on
        // every exit; one step per poll of `solve_with` (it polls before every SCF step)
        let probe = self.probe.clone();
        let _busy = probe.as_ref().map(|(p, s)| p.busy(*s));
        let mut polled = || {
            if let Some((p, s)) = &probe {
                p.scf_step(*s);
            }
            should_abort()
        };
```

and pass `&mut polled` instead of `should_abort` to `solver.solve_with(...)` inside `retry_cold_on_stall`. In `CpuKs::evaluate`, after its `should_abort()` check: `let probe = self.probe.clone(); let _busy = probe.as_ref().map(|(p, s)| p.busy(*s));` (no step count).

`crates/providers/src/surrogate/mod.rs`: field `probe: Option<Arc<sim::ActivityProbe>>` (`None` in `new`), the builder `pub fn with_probe(mut self, p: Arc<sim::ActivityProbe>) -> SurrogateKs`, and in the inherent `refit()` (trust.rs) bracket the body:

```rust
        if let Some(p) = &self.probe {
            p.begin_refit(self.samples());
        }
        let out = self.refit_inner();
        if let Some(p) = &self.probe {
            p.end_refit();
        }
        out
```

(rename the current body `refit_inner`, private). `finish_pass` calls `refit()`, so the pass's own refit is reported too.

- [ ] **Step 6: Provider tests** (append to `crates/providers/src/ks.rs` tests and `surrogate/tests.rs`)

```rust
    /// Spec A1, Review Focus 3: the probe runs during a GPU solve, counts its SCF steps as the
    /// solver reports them, and clears on success, cancellation and error.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn the_probe_clears_on_error_and_cancel() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        let probe = sim::ActivityProbe::new();
        let mut p = GpuKs::new(ctx, 1e-6).with_probe(probe.clone(), sim::Slot::Main);
        let scene = h2_scene(25, 1.4);
        let r = p.evaluate(&scene, &mut || false).unwrap();
        let v = probe.view();
        assert!(!v.main.running);
        assert_eq!(v.main.scf_step as usize, r.stats.scf_steps, "one poll per SCF step");
        let mut n = 0;
        let mut abort_third = || { n += 1; n > 3 };
        assert!(matches!(p.evaluate(&scene, &mut abort_third), Err(ProviderError::Cancelled)));
        assert!(!probe.view().main.running);
        let mut bad = scene.clone();
        bad.nuclei[1].pos = bad.nuclei[0].pos; // coincident nuclei: an error
        assert!(p.evaluate(&bad, &mut || false).is_err());
        assert!(!probe.view().main.running);
        // the error may be raised before the solve starts (problem setup) or inside it
        assert!((2..=3).contains(&probe.view().main.solves));
    }
```

(`h2_scene(n, r)`: use the helper the existing ks tests build their 25³ H₂ with; if none is named so, add a three-line helper next to them.) If `scf_steps` counts differ by the pre-loop poll, assert `scf_step == scf_steps` or `scf_steps + 1` and document which in the doc comment. Do not loosen anything else.

In `surrogate/tests.rs`:

```rust
/// Spec A1: a refit marks the probe's refit phase for its duration, and clears it.
#[test]
fn a_refit_reports_its_phase_on_the_probe() {
    let probe = sim::ActivityProbe::new();
    let (s, _) = trained(1.2, 2.2, 0, FallbackPolicy::default());
    let mut s = s.with_probe(probe.clone());
    for i in 0..60 {
        let sc = h2(1.2 + i as f64 * 0.02);
        s.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
    }
    s.refit().unwrap();
    assert_eq!(probe.view().refit, None);
}
```

- [ ] **Step 7: Run**

Run: `cargo test -p sim --release activity` → PASS. `cargo test -p providers --release` → PASS (all existing tests unchanged). `cargo test -p validate --release --test gpu_vs_cpu` → PASS, numbers unchanged (the abort wrapper must not change a solve). `cargo build --workspace`.

- [ ] **Step 8: Commit**

```bash
git add crates/sim/src/activity.rs crates/sim/src/lib.rs crates/sim/src/provider.rs crates/sim/src/test_providers.rs crates/providers/src/ks.rs crates/providers/src/surrogate/mod.rs crates/providers/src/surrogate/trust.rs crates/providers/src/surrogate/tests.rs
git commit -o -m "feat(sim,providers): activity records and the live solve probe

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- crates/sim/src/activity.rs crates/sim/src/lib.rs crates/sim/src/provider.rs crates/sim/src/test_providers.rs crates/providers/src/ks.rs crates/providers/src/surrogate/mod.rs crates/providers/src/surrogate/trust.rs crates/providers/src/surrogate/tests.rs
```

---

### Task 2: rule and outcome codes, one event per teaching solve, lifetime counts **[Fable — physics-review gate]**

**Files:**
- Create: `crates/providers/src/surrogate/ledger.rs`
- Modify: `crates/providers/src/surrogate/{mod.rs, policy.rs, quarantine.rs, trust.rs, audit.rs, store.rs}`, `crates/providers/src/sampling/mod.rs`, `crates/surrogate/src/sidecar.rs`
- Test: `crates/providers/src/surrogate/tests.rs`, `crates/surrogate/src/sidecar.rs` tests

**Interfaces:**
- Consumes: Task 1's `TeachEvent`, `TeachRule`, `TeachOutcome`, `TeachOrigin`, `PassJob`, `EnvClass`, `TrustKey`, `SolveCounts`.
- Produces:
  - `pub(super) struct Refusal { text: String, rule: TeachRule, atom: Option<usize> }` (policy.rs). `Gate.reason`, `Handoff.reason`, `VarianceVerdict.reason`, `envelope_outside`'s second value, `class_untrusted`'s first value and `streak_reason` become `Refusal`. Every `text` is **byte-identical** to today's string.
  - `pub(super) struct Admission { kept: bool, outcome: TeachOutcome }` returned by `admit`
  - `SurrogateKs::untrusted_check(&self) -> Option<(TrustKey, String)>`; `untrusted_reason()` = `untrusted_check().map(|c| c.1)`
  - `SurrogateKs::push_event(&mut self, e: TeachEvent)` (`pub(crate)`, used by `crate::sampling`), `SurrogateKs::solve_counts(&self) -> SolveCounts`, `Trainable::take_teach_events` implemented
  - `pub const EVENT_QUEUE: usize = 2048;`
  - `surrogate::sidecar::{TeacherSolves, write_full}` and `Sidecar::teacher_solves: Option<TeacherSolves>`

- [ ] **Step 1: Write the failing tests** (append to `crates/providers/src/surrogate/tests.rs`)

```rust
// ---- activity events (spec §4.3 A3) ----

fn drain(s: &mut SurrogateKs) -> Vec<sim::TeachEvent> {
    Trainable::take_teach_events(s)
}

/// Each rule path the policy tests already drive emits one event with the rule's code, the atom
/// it names, the outcome and the origin; the detail is the note the step carries.
#[test]
fn every_teacher_step_emits_one_event_with_its_rule() {
    use sim::activity::{TeachOrigin as O, TeachOutcome as K, TeachRule as R, TrustKey};
    // rule 1: no model
    let (mut s, _) = trained(1.2, 2.2, 0, FallbackPolicy::default());
    let r = s.evaluate(&h2(1.6), &mut never()).unwrap();
    let e = drain(&mut s);
    assert_eq!(e.len(), 1);
    assert_eq!((e[0].origin, e[0].rule, e[0].outcome), (O::Fallback, R::NoModel, K::Kept));
    assert!(r.stats.notes.iter().any(|n| n.starts_with(&e[0].detail)), "{:?} vs {:?}", e[0].detail, r.stats.notes);
    // rule 4 on the count
    let (mut s, _) = trained(1.2, 2.2, 5, FallbackPolicy::default());
    s.evaluate(&h2(1.6), &mut never()).unwrap();
    let e = drain(&mut s);
    assert_eq!(e[0].rule, R::Untrusted(TrustKey::Samples));
    // rule 5 (outside the window): the atom is named, the outcome is a kept or thinned sample
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    assert!(s.trusted());
    s.evaluate(&in_big_cell(h2(4.0)), &mut never()).unwrap();
    let e = drain(&mut s);
    assert!(matches!(e[0].rule, R::IsolatedAtom | R::Coordination | R::Spread | R::Grade | R::NewEnvironment), "{:?}", e[0]);
    assert!(e[0].atom.is_some() && e[0].gap.is_some() && e[0].n_atoms == 2);
    // rule 6 (inflated posterior): the atom with the largest ratio
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    always_unsure(&mut s);
    s.evaluate(&h2(1.6), &mut never()).unwrap();
    let e = drain(&mut s);
    assert_eq!(e[0].rule, R::ForceUncertainty);
    assert!(e[0].atom.is_some());
    // a model answer emits nothing
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    let r = s.evaluate(&h2(1.6), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Surrogate);
    assert!(drain(&mut s).is_empty());
}

/// Rule 3 is logged, not kept, and not counted for the model (spec A7); species is rule 2.
#[test]
fn other_settings_and_new_species_have_their_codes() {
    use sim::activity::{TeachOutcome as K, TeachRule as R};
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    let before = s.solve_counts();
    let mut charged = h2(1.6);
    charged.charge = 1;
    s.evaluate(&charged, &mut never()).unwrap();
    let e = drain(&mut s);
    assert_eq!((e[0].rule, e[0].outcome), (R::OtherSettings, K::NotOffered));
    assert_eq!(s.solve_counts(), before);
    s.evaluate(&dimer(1.6, [1.0, 0.0, 0.0], [0.0; 3], 1, 2), &mut never()).unwrap();
    assert_eq!(drain(&mut s)[0].rule, R::NewSpecies(2));
}

/// Admission outcomes: a box-affected answer is used and not kept; thinning says so.
#[test]
fn outcomes_name_what_admit_decided() {
    use sim::activity::TeachOutcome as K;
    // the existing box test's setup: a trusted model, a teacher that reports a high face density
    let (mut s, h) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    h.lock().unwrap().calls = 0;
    always_unsure(&mut s);
    s.set_test_face_density(Some(1.0)); // see Step 3: a test-only hook on the shared Morse teacher's stats
    s.evaluate(&h2(1.6), &mut never()).unwrap();
    let e = drain(&mut s);
    assert_eq!(e[0].outcome, K::BoxNotKept);
    assert!(e[0].box_affected && e[0].face_density == Some(1.0));
}

/// `MorseDimer` answers only two nuclei; the pass's isolated-atom references need one. This
/// answers the lone atom with E_atom (as the worker tests' `ScriptedMorse` does).
struct AtomMorse(MorseDimer);

impl ForceProvider for AtomMorse {
    fn evaluate(&mut self, scene: &Scene, a: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        if scene.nuclei.len() == 1 {
            let mut stats = EvalStats::default();
            stats.energy.total = self.0.e_atom;
            return Ok(ForceResult { energy: self.0.e_atom, forces: vec![[0.0; 3]], stats, display: None });
        }
        self.0.evaluate(scene, a)
    }

    fn name(&self) -> &'static str {
        "morse dimer"
    }

    fn reset(&mut self) {}
}

/// Review Focus 1: kept counts equal the samples added; origins add up.
#[test]
fn solve_counts_add_up_over_a_pass_fallbacks_and_audits() {
    let teacher = AtomMorse(MorseDimer::h2());
    let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy { audit_every: 5, ..FallbackPolicy::default() }).with_teacher_tol(1e-6);
    let scene = h2(1.45);
    let mut refs = sim::AtomReferences::default();
    Trainable::start_pass(&mut s, &scene, &sim::SamplingPlan::default(), &mut refs).unwrap();
    while Trainable::pass_step(&mut s, &mut never()).unwrap().is_some() {}
    let pass_events = drain(&mut s);
    let c = s.solve_counts();
    assert_eq!(c.pass as usize, pass_events.iter().filter(|e| e.counts_for_model()).count());
    assert_eq!(c.kept as usize, s.samples(), "every kept pass answer is a sample");
    assert!(pass_events.iter().any(|e| matches!(e.rule, sim::activity::TeachRule::Pass(sim::activity::PassJob::Reference { z: 1, .. }))));
    assert!(pass_events.iter().filter_map(|e| e.burst_t).all(|t| t.is_finite() && t >= 0.0));
    assert!(s.trusted(), "the default plan trusts the Morse H₂ (else take the plan the worker's sizing test uses)");
    for i in 0..40 {
        s.evaluate(&h2(1.40 + 0.002 * i as f64), &mut never()).unwrap();
    }
    let c2 = s.solve_counts();
    assert!(c2.audit > 0, "audit_every 5 over 40 answered steps");
    assert_eq!(c2.kept as usize, s.samples());
    assert_eq!(c2.total(), c2.kept + c2.not_kept());
}

/// Review Focus 2: the queue keeps the newest 2 048 events and counts the dropped ones; the
/// counts are not capped.
#[test]
fn the_event_queue_drops_the_oldest_and_counts_them() {
    let (mut s, _) = trained(1.2, 2.2, 0, FallbackPolicy::default());
    for i in 0..(EVENT_QUEUE + 10) {
        let mut e = sim::TeachEvent::new(sim::activity::TeachOrigin::Fallback, sim::activity::TeachRule::NoModel, sim::activity::TeachOutcome::Thinned);
        e.n_atoms = i;
        s.push_event(e);
    }
    assert_eq!(s.events_dropped(), 10);
    let e = drain(&mut s);
    assert_eq!(e.len(), EVENT_QUEUE);
    assert_eq!(e[0].n_atoms, 10, "the oldest went first");
    assert_eq!(s.solve_counts().fallback, (EVENT_QUEUE + 10) as u64);
}

/// Spec A2: the counts survive a sidecar round trip; a file without the key reads as
/// "unrecorded before"; a forget resets them.
#[test]
fn solve_counts_survive_the_sidecar_and_forget_resets_them() {
    let dir = std::env::temp_dir().join(format!("as-activity-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("counts.model");
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    always_unsure(&mut s);
    for _ in 0..3 {
        s.evaluate(&h2(1.6), &mut never()).unwrap();
    }
    let c = s.solve_counts();
    assert_eq!(c.fallback, 3);
    s.save_sidecar(&path).unwrap();
    let (mut back, _) = trained(1.2, 2.2, 0, FallbackPolicy::default());
    back.install_sidecar(surrogate::sidecar::read(&path).unwrap()).unwrap();
    assert_eq!(back.solve_counts(), c);
    Trainable::forget(&mut back);
    assert_eq!(back.solve_counts(), sim::SolveCounts::default());
    // a file written without the key (the pre-activity writer)
    let sc = surrogate::sidecar::read(&path).unwrap();
    surrogate::sidecar::write_full(&path, &sc.model, &sc.accumulator, sc.report.as_ref(), &sc.samples, sc.reference_rule.as_deref(), &sc.quarantine, &sc.face_density, None).unwrap();
    back.install_sidecar(surrogate::sidecar::read(&path).unwrap()).unwrap();
    assert!(back.solve_counts().unrecorded_before && back.solve_counts().total() == 0);
    let _ = std::fs::remove_dir_all(&dir);
}
```

Sidecar test (`crates/surrogate/src/sidecar.rs` tests):

```rust
    /// The optional "teacher_solves" key (activity spec A2): written only when given, read back
    /// exactly, absent → None; a file without it is byte-identical to the old writer's.
    #[test]
    fn teacher_solves_is_an_optional_header_key() {
        let (orig, dir) = fixture_sidecar(); // the fixture the round-trip tests already build
        let a = dir.join("a.model");
        let b = dir.join("b.model");
        write_with_flags(&a, &orig.model, &orig.accumulator, orig.report.as_ref(), &orig.samples, None, &[], &[]).unwrap();
        write_full(&b, &orig.model, &orig.accumulator, orig.report.as_ref(), &orig.samples, None, &[], &[], None).unwrap();
        assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
        let t = TeacherSolves { pass: 273, fallback: 70, audit: 18, verification: 1, kept: 343 };
        write_full(&b, &orig.model, &orig.accumulator, orig.report.as_ref(), &orig.samples, None, &[], &[], Some(&t)).unwrap();
        assert_eq!(read(&b).unwrap().teacher_solves, Some(t));
        assert_eq!(read(&a).unwrap().teacher_solves, None);
    }
```

(Use whatever the existing round-trip tests name their fixture; if it is inline, factor it into `fixture_sidecar()` in the test module.)

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p providers --release activity` and `cargo test -p surrogate --release teacher_solves`
Expected: compile errors (`take_teach_events` not implemented, `Refusal`, `push_event`, `write_full` missing).

- [ ] **Step 3: Thread the codes through the decision points (no decision changes)**

`policy.rs`:

```rust
/// A refusal: the text the notes have always carried (byte-identical), the rule's code and the
/// atom it names (activity spec A3).
#[derive(Clone, Debug)]
pub(super) struct Refusal {
    pub(super) text: String,
    pub(super) rule: TeachRule,
    pub(super) atom: Option<usize>,
}

impl Refusal {
    pub(super) fn new(text: impl Into<String>, rule: TeachRule, atom: Option<usize>) -> Refusal {
        Refusal { text: text.into(), rule, atom }
    }
}
```

Mapping, site by site (each keeps its `format!` exactly; only the wrapper changes):

| site | rule | atom |
|---|---|---|
| `gate`: no model | `NoModel` | — |
| `gate`: provenance | `OtherSettings` | — |
| `gate`: species `zu` | `NewSpecies(zu)` | — |
| `gate`: untrusted | `Untrusted(key)` from `untrusted_check()` | — |
| `envelope_outside`: 5a | `IsolatedAtom` | `i` |
| 5b | `Coordination` | `worst_ratio.1` |
| 5c | `Spread` | `worst.1` |
| 5e void | `NoGrade` | — |
| 5e ∞ | `NewEnvironment` | `worst_grade.1` |
| 5e | `Grade` | `worst_grade.1` |
| `evaluate` 5d (novelty) | `NewEnvironment` | `at` |
| `variance_check` rule 6 | `ForceUncertainty` | `j` |
| rule 7 | `EnergyUncertainty` | — |
| `class_untrusted` (each branch) | `Class(if c == 0 { EnvClass::Pair } else { EnvClass::ManyBody })` | — |
| fence (`evaluate`, "within … of a quarantined geometry") | `NearQuarantine` | — |
| audit reason (`evaluate`) | `Audit` | — |
| relax verification | `RelaxVerification` | — |

`trust.rs`: rename the body of `untrusted_reason` into `pub fn untrusted_check(&self) -> Option<(TrustKey, String)>`, each branch returning its key (`Samples`, `ResidualDofE`, `ResidualDofF`, `EnergyResidual`, `ForceResidual`, `ForceNoise`, `Audit`) with today's string, and

```rust
    pub fn untrusted_reason(&self) -> Option<String> {
        self.untrusted_check().map(|(_, s)| s)
    }
```

`stuck()` reads `self.streak_reason.as_ref().map(|r| r.text.as_str())`; behaviour unchanged.

`quarantine.rs`: `admit` returns `Admission`:

```rust
pub(super) struct Admission {
    pub(super) kept: bool,
    pub(super) outcome: TeachOutcome,
}
```

`!keep` → `NotOffered`; the no-model cap → `WaitingForPass`; box → `BoxNotKept`; fence vote kept → `KeptByVote`; fence disagree → `FenceDisagrees`; fence agree → `KeptReleased { released, superseded }`; thinned → `Thinned`; the final `true` → `KeptBoxAffected` when `face > box_affected_face_density(h)` (an essential box answer), else `Kept`. `store.rs`: `keep_sample` returns `bool` (false on a store error, note unchanged); a `false` turns the outcome into `StoreRefused`.

Test hook for `outcomes_name_what_admit_decided`: a `#[cfg(test)] pub(super) fn set_test_face_density(&mut self, rho: Option<f64>)` storing an override that `teach` applies to `r.stats.face_density` before `admit` (test-only; production path untouched). If the existing `a_box_affected_answer_is_used_but_not_trained_on` test already injects a face density another way, reuse that and drop the hook.

- [ ] **Step 4: The ledger and the events**

`ledger.rs`:

```rust
//! The activity ledger (activity spec §4.3 A2–A3): one `TeachEvent` per teaching solve in a
//! bounded queue the worker drains, and the model's lifetime `SolveCounts`.
use super::*;
use sim::activity::{SolveCounts, TeachEvent};
use std::collections::VecDeque;

/// Events held for the worker's next drain; older ones are dropped and counted.
pub const EVENT_QUEUE: usize = 2048;

#[derive(Default)]
pub(super) struct Ledger {
    pub(super) events: VecDeque<TeachEvent>,
    pub(super) dropped: u64,
    pub(super) counts: SolveCounts,
}

impl SurrogateKs {
    /// Records one teaching solve (or a cell-guard skip): counted, queued.
    pub(crate) fn push_event(&mut self, e: TeachEvent) {
        self.ledger.counts.count(&e);
        if self.ledger.events.len() == EVENT_QUEUE {
            self.ledger.events.pop_front();
            self.ledger.dropped += 1;
        }
        self.ledger.events.push_back(e);
    }

    pub fn solve_counts(&self) -> SolveCounts {
        self.ledger.counts
    }

    pub fn events_dropped(&self) -> u64 {
        self.ledger.dropped
    }

    pub(super) fn take_events(&mut self) -> Vec<TeachEvent> {
        self.ledger.events.drain(..).collect()
    }
}
```

Add `mod ledger; pub use ledger::EVENT_QUEUE;` and the field `ledger: Ledger` to `SurrogateKs` (`Ledger::default()` in `new`). `forget()` resets `self.ledger.counts = SolveCounts::default()` (events stay queued: they happened). `Trainable::take_teach_events` → `self.take_events()`. `status()` → `solves: self.solve_counts()`.

`teach()`: capture `let teacher_ms = r.stats.solve_ms as f64; let scf = r.stats.scf_steps;` right after the teacher answers. A teacher error other than `Cancelled` pushes `Failed` before returning it:

```rust
        let r = match self.teacher.evaluate(scene, should_abort) {
            Ok(r) => r,
            Err(e) => {
                if !matches!(e, ProviderError::Cancelled) {
                    let mut ev = TeachEvent::new(origin_of(&h), h.reason.rule, TeachOutcome::Failed);
                    ev.detail = format!("{} — the teacher failed: {e}", h.reason.text);
                    ev.n_atoms = h.pos.len();
                    self.push_event(ev);
                }
                return Err(e);
            }
        };
```

with `fn origin_of(h: &Handoff) -> TeachOrigin { if h.audit { TeachOrigin::Audit } else if h.reason.rule == TeachRule::RelaxVerification { TeachOrigin::Verification } else { TeachOrigin::Fallback } }`. At the end of `teach`, before `Ok(...)`:

```rust
        let mut ev = TeachEvent::new(origin_of_parts(h_audit, rule), rule, outcome);
        ev.atom = atom;
        ev.detail = stats.notes.first().cloned().unwrap_or_default();
        ev.face_density = face;
        ev.box_affected = boxed.is_some();
        ev.gap = gap; // Some(r.energy − m.energy) when h.model was Some
        ev.n_atoms = n_atoms;
        ev.solve_ms = teacher_ms;
        ev.scf_steps = scf;
        ev.audit = if h_audit { self.last_audit.as_ref().map(|a| a.verdict) } else { None };
        self.push_event(ev);
```

(`h` is moved by the `match h.model`; copy `h.audit`, `h.reason.rule`, `h.reason.atom` and `h.pos.len()` into locals first.)

`evaluate()`: rule 0. Replace `scene.check_inside_cell()?;` by

```rust
        if let Err(e) = scene.check_inside_cell() {
            let mut ev = TeachEvent::new(TeachOrigin::Fallback, TeachRule::Cell, TeachOutcome::NotSolved);
            ev.detail = e.to_string();
            ev.n_atoms = scene.nuclei.len();
            self.push_event(ev);
            return Err(e);
        }
```

`audit.rs` `judge_audit`: after the verdict, push `TeachEvent { origin: Audit, rule: Audit, outcome: admission.outcome (StoreRefused if keep_sample failed), gap: Some(gap), face_density: face, box_affected: box_rho.is_some(), audit: Some(verdict), audit_lag: Some(unaudited), solve_ms: ms, detail: notes[0].clone(), n_atoms }`. An audit discarded as "the model changed settings or species" pushes nothing (no answer was judged).

`crate::sampling` (`sampling/mod.rs`): `record(…, tag, solve_ms, scf, burst_t)` pushes after `add_sample_with_face`:

```rust
        let job = match tag {
            SampleTag::Perturbation => PassJob::Perturbation,
            SampleTag::Scan => PassJob::Scan,
            _ => PassJob::HotBurst,
        };
        let boxed = face.is_some_and(|rho| rho > box_affected_face_density(scene.cell.h));
        let outcome = match &added {
            Ok(()) if boxed => TeachOutcome::KeptBoxAffected,
            Ok(()) => TeachOutcome::Kept,
            Err(_) => TeachOutcome::StoreRefused,
        };
        let mut ev = TeachEvent::new(TeachOrigin::Pass, TeachRule::Pass(job), outcome);
        ev.face_density = face;
        ev.box_affected = boxed;
        ev.n_atoms = scene.nuclei.len();
        ev.solve_ms = solve_ms;
        ev.scf_steps = scf;
        ev.burst_t = burst_t;
        ev.detail = format!("sampling: {} {}/{}", self.progress.stage, self.progress.done + 1, self.progress.total);
        surrogate.push_event(ev);
```

`burst_step` passes `Some(work.temperature())` (sim's `Scene::temperature`, the instantaneous kinetic T of the solved geometry, spec §9.5). `sample` failures push `Failed` with the stage's job. A reference multiplicity solve pushes `Pass(Reference { z, multiplicity: m })` with `Reference` or `Failed` (time it with `Instant::now()` around `scan.step`). `Job::Skip` and the burst leaving the cell push `TeachRule::Cell` with origin `Pass` and outcome `NotSolved`: one event per skipped geometry, and the burst's skipped tail as one event whose `detail` is "the hot burst reached the cell margin; its last k evaluations were skipped".

- [ ] **Step 5: Persist the counts (sidecar optional key)**

`crates/surrogate/src/sidecar.rs`:

```rust
/// Lifetime teaching solves of the model (header key "teacher_solves"; optional, activity spec
/// A2): pass, fallback, audit, verification and how many were kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TeacherSolves {
    pub pass: u64,
    pub fallback: u64,
    pub audit: u64,
    pub verification: u64,
    pub kept: u64,
}
```

`Sidecar` gains `pub teacher_solves: Option<TeacherSolves>`. `write_with_flags(...)` becomes a thin call to

```rust
#[allow(clippy::too_many_arguments)]
pub fn write_full(path: &Path, model: &Model, acc: &Accumulator, report: Option<&FitReport>, samples: &[Sample], reference_rule: Option<&str>, quarantine: &[u8], face_density: &[f64], solves: Option<&TeacherSolves>) -> io::Result<()>
```

with `solves: None`. `encode` builds the header pairs as a `Vec<(&str, String)>` and appends `("teacher_solves", obj(&[("pass", t.pass.to_string()), ("fallback", …), ("audit", …), ("verification", …), ("kept", …)]))` **only when `Some`**, so files without it are byte-identical (the test pins it). `decode`:

```rust
    let teacher_solves = match h.get("teacher_solves") {
        Err(_) | Ok(Json::Null) => None,
        Ok(t) => Some(TeacherSolves {
            pass: t.get("pass")?.usize()? as u64,
            fallback: t.get("fallback")?.usize()? as u64,
            audit: t.get("audit")?.usize()? as u64,
            verification: t.get("verification")?.usize()? as u64,
            kept: t.get("kept")?.usize()? as u64,
        }),
    };
```

Document the key in the module doc's format section ("Optional, every version (activity spec A2)"). `store.rs`: `save_sidecar` passes `Some(&TeacherSolves { … from self.ledger.counts })`; `install_sidecar` sets `self.ledger.counts` from the file (`unrecorded_before: false`) or, when `None`, `SolveCounts { unrecorded_before: true, ..Default::default() }`.

- [ ] **Step 6: Run the tests and the gate**

Run: `cargo test -p surrogate --release` → PASS. `cargo test -p providers --release` → PASS, every pre-existing test unchanged (the note strings are byte-identical: any test matching a note still passes). `cargo test -p validate --release --test surrogate` → PASS with the same printed numbers as before (paste both runs' key lines in the task report).

- [ ] **Step 7: Physics review**

Dispatch `physics-reviewer` on the diff with: "No decision changed? Is every `Refusal` text byte-identical? Is each rule code the rule that actually fired, given the priority envelope → novelty → class → variance → fence? Does `KeptBoxAffected` match the essential-box exemption exactly? Is `burst_t` the temperature of the solved geometry?" Fix findings before committing.

- [ ] **Step 8: Commit**

```bash
git add crates/providers/src/surrogate crates/providers/src/sampling/mod.rs crates/surrogate/src/sidecar.rs
git commit -o -m "feat(providers): one activity event per teaching solve with its rule and outcome; lifetime solve counts in the sidecar

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -- crates/providers/src/surrogate crates/providers/src/sampling/mod.rs crates/surrogate/src/sidecar.rs
```

---

### Task 3: the trust checklist's data and the structured stuck reason **[Fable — physics-review gate]**

**Files:**
- Modify: `crates/providers/src/surrogate/trust.rs`, `crates/providers/src/surrogate/mod.rs` (`status()`)
- Test: `crates/providers/src/surrogate/tests.rs`

**Interfaces:**
- Consumes: Task 2's `untrusted_check`, `Refusal` in `streak_reason`.
- Produces: `SurrogateKs::trust_checks(&self) -> Vec<TrustCheck>`, `SurrogateKs::stuck_info(&self) -> Option<StuckInfo>`; `TrainableStatus::{trust_checks, stuck_info}` filled.

- [ ] **Step 1: Write the failing tests**

```rust
/// Review Focus 4: on every branch of the trust rule the first unmet check is the condition the
/// refusal names, and a trusted model meets every check.
#[test]
fn the_first_unmet_trust_check_is_the_refusal() {
    let first_unmet = |s: &SurrogateKs| s.trust_checks().into_iter().find(|c| !c.met).map(|c| c.key);
    let key = |s: &SurrogateKs| s.untrusted_check().map(|c| c.0);
    // n < Q′
    let (s, _) = trained(1.2, 2.2, 5, FallbackPolicy::default());
    assert_eq!(first_unmet(&s), Some(sim::activity::TrustKey::Samples));
    assert_eq!(first_unmet(&s), key(&s));
    // residual dofs
    let (s, _) = trained(1.2, 2.2, 30, FallbackPolicy::default());
    assert_eq!(first_unmet(&s), key(&s));
    assert!(matches!(key(&s), Some(sim::activity::TrustKey::ResidualDofE | sim::activity::TrustKey::ResidualDofF)));
    // training residuals (the misspecified fit of `misspecified_fit_is_untrusted_by_its_training_residuals`)
    let (s, _) = trained_oscillatory(1.2, 2.2, 120, 0.02, 6.0);
    assert_eq!(first_unmet(&s), key(&s));
    // a failed audit
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    s.audit_failed = Some(0.5);
    assert_eq!(first_unmet(&s), Some(sim::activity::TrustKey::Audit));
    assert_eq!(first_unmet(&s), key(&s));
    // trusted
    let (s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    assert!(s.trusted() && s.trust_checks().iter().all(|c| c.met));
    // values are the report's and the policy's
    let r = s.report().unwrap().clone();
    let c = s.trust_checks();
    assert_eq!((c[0].value, c[0].limit), (r.n_samples as f64, r.q_prime as f64));
    assert_eq!((c[1].value, c[1].limit), (r.residual_dof_e, MIN_RESIDUAL_DOF));
}

/// Spec A5: `stuck_info` is `stuck` in structured form: same streak, rule of the last fallback,
/// its atom, the box streak.
#[test]
fn stuck_info_carries_the_rule_and_atom_of_the_streak() {
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    assert!(s.stuck_info().is_none());
    always_unsure(&mut s);
    for _ in 0..STUCK_STEPS {
        s.evaluate(&h2(1.6), &mut never()).unwrap();
    }
    let i = s.stuck_info().expect("stuck after STUCK_STEPS fallbacks");
    assert!(s.stuck().is_some());
    assert_eq!(i.rule, sim::activity::TeachRule::ForceUncertainty);
    assert!(i.atom.is_some() && i.steps >= STUCK_STEPS && !i.box_streak);
}
```

(`trained_oscillatory`'s signature is in tests.rs line 235; pick arguments that the existing misspecification test shows untrusted by residuals, and assert `key(&s)` is `EnergyResidual` or `ForceResidual`.)

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p providers --release trust_check stuck_info`
Expected: compile error, `trust_checks`/`stuck_info` missing.

- [ ] **Step 3: Implement** (`trust.rs`)

```rust
    /// The trust rule's conditions with their numbers, in `untrusted_check`'s order (activity spec
    /// A4); empty before the first fit. The first unmet one is the refusal's condition (pinned).
    pub fn trust_checks(&self) -> Vec<TrustCheck> {
        let Some(r) = self.report.as_ref() else { return Vec::new() };
        let p = &self.policy;
        let mut v = vec![
            TrustCheck { key: TrustKey::Samples, met: r.n_samples >= r.q_prime, value: r.n_samples as f64, limit: r.q_prime as f64 },
            TrustCheck { key: TrustKey::ResidualDofE, met: r.residual_dof_e >= MIN_RESIDUAL_DOF, value: r.residual_dof_e, limit: MIN_RESIDUAL_DOF },
            TrustCheck { key: TrustKey::ResidualDofF, met: r.residual_dof_f >= MIN_RESIDUAL_DOF, value: r.residual_dof_f, limit: MIN_RESIDUAL_DOF },
        ];
        if let Some(t) = &self.training_residuals {
            v.push(TrustCheck { key: TrustKey::EnergyResidual, met: t.energy_per_atom <= p.energy_std_per_atom, value: t.energy_per_atom, limit: p.energy_std_per_atom });
            v.push(TrustCheck { key: TrustKey::ForceResidual, met: t.force <= t.force_bound, value: t.force, limit: t.force_bound });
        }
        v.push(TrustCheck { key: TrustKey::ForceNoise, met: !self.noise_fails(), value: r.sigma_f, limit: p.force_std_abs });
        v.push(TrustCheck { key: TrustKey::Audit, met: self.audit_failed.is_none(), value: self.audit_failed.map_or(0.0, f64::abs), limit: p.energy_std_per_atom });
        v
    }

    /// [`stuck`](Self::stuck) in structured form (activity spec A5).
    pub fn stuck_info(&self) -> Option<StuckInfo> {
        (self.streak >= STUCK_STEPS && self.trusted()).then(|| {
            let r = self.streak_reason.as_ref();
            StuckInfo {
                rule: r.map_or(TeachRule::NearQuarantine, |r| r.rule),
                atom: r.and_then(|r| r.atom),
                steps: self.streak,
                box_streak: self.streak_box >= STUCK_STEPS,
                gap: self.streak_gap,
                face_density: self.streak_box_face,
            }
        })
    }
```

(Every comparison is the negation of `untrusted_check`'s: `<` ↔ `>=`, `>` ↔ `<=`. If any differs at equality, the test catches it; fix `trust_checks`, never the rule.) `status()` sets `trust_checks: self.trust_checks(), stuck_info: self.stuck_info()`.

- [ ] **Step 4: Run** `cargo test -p providers --release` → PASS.

- [ ] **Step 5: Physics review.** `physics-reviewer`: "Is each check the exact negation of its refusal branch, including at equality? Is the audit check's 'limit' honest (the per-atom tier, the value a total gap — say so in the UI or change the limit to N·tier)?" Apply the ruling.

- [ ] **Step 6: Commit**

```bash
git add crates/providers/src/surrogate/trust.rs crates/providers/src/surrogate/mod.rs crates/providers/src/surrogate/tests.rs
git commit -o -m "feat(providers): trust checks and structured stuck reason for the Activity tab

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -- crates/providers/src/surrogate/trust.rs crates/providers/src/surrogate/mod.rs crates/providers/src/surrogate/tests.rs
```

---

### Task 4: `would_answer` and the training views **[Fable — physics-review gate]**

**Files:**
- Create: `crates/providers/src/surrogate/views.rs`
- Modify: `crates/providers/src/surrogate/mod.rs` (`mod views;`, `Trainable::training_views`)
- Test: `crates/providers/src/surrogate/tests.rs`

**Interfaces:**
- Consumes: Task 2's `Refusal` codes; Task 1's view records.
- Produces: `SurrogateKs::would_answer(&mut self, pos: &[[f64; 3]], z: &[u8], prov: &Provenance) -> Result<(), (TeachRule, Option<usize>)>`; `SurrogateKs::training_views(&mut self) -> Option<TrainingViews>`; `Trainable::training_views` implemented.

- [ ] **Step 1: Write the failing tests**

```rust
/// Spec A6: `would_answer` is `evaluate`'s verdict without its side effects. With the variance
/// evaluated on every consultation and audits off, `evaluate` answers from the model exactly
/// where `would_answer` says Ok, with the same rule otherwise, over dimers and trimers.
#[test]
fn would_answer_agrees_with_evaluate() {
    let policy = FallbackPolicy { variance_cadence: 1, audit_every: 0, ..FallbackPolicy::default() };
    let (mut s, h) = trained(1.2, 2.2, 120, policy);
    let mut rng = StdRng::seed_from_u64(0x5EED);
    let mut scenes = Vec::new();
    for _ in 0..150 {
        scenes.push(h2(rng.gen_range(0.9..3.2)));
    }
    for _ in 0..50 {
        let mut sc = h2(rng.gen_range(1.2..2.0));
        sc.nuclei.push(Nucleus::new(1, [rng.gen_range(-2.5..2.5), rng.gen_range(-2.5..2.5), rng.gen_range(-2.5..2.5)]));
        scenes.push(sc);
    }
    let calls_before = calls(&h);
    let verdicts: Vec<_> = scenes
        .iter()
        .map(|sc| {
            let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
            let z: Vec<u8> = sc.nuclei.iter().map(|n| n.z).collect();
            let prov = s.provenance_for(sc);
            s.would_answer(&pos, &z, &prov)
        })
        .collect();
    assert_eq!(calls(&h), calls_before, "would_answer never calls the teacher");
    // the verdicts were all taken before any evaluate: no refit runs in between, so the model
    // they judged is the one evaluate consults (a kept fallback sample changes no verdict until a refit)
    let (mut agree, mut answered) = (0, 0);
    for (sc, v) in scenes.iter().zip(&verdicts) {
        Trainable::take_teach_events(&mut s);
        s.discontinuity();
        if sc.check_inside_cell().is_err() {
            continue;
        }
        // the Morse teacher answers only dimers: a trimer handed to it fails, and the failure's
        // event still carries the rule that handed it over (Task 2)
        let r = s.evaluate(sc, &mut never());
        let ev = Trainable::take_teach_events(&mut s);
        match v {
            Ok(()) => {
                assert_eq!(r.expect("a model answer needs no teacher").stats.source, Source::Surrogate, "{:?}", sc.nuclei);
                answered += 1;
            }
            Err((rule, _)) => assert_eq!(ev.first().map(|e| e.rule), Some(*rule), "{:?}", sc.nuclei),
        }
        agree += 1;
    }
    eprintln!("would_answer: {agree} geometries compared, {answered} answered by the model");
    assert!(answered > 20 && agree - answered > 20, "the sweep must cover both verdicts");
}

/// Spec §9: parity points are (E − Σ E_ref)/N of the teacher and of `predict` at each stored
/// sample; the band is the epistemic std, the same number rule 7 compares; no teacher call.
#[test]
fn training_views_read_the_store_and_the_posterior_only() {
    let (mut s, h) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    let c0 = calls(&h);
    let v = s.training_views().expect("a fitted model");
    assert_eq!(calls(&h), c0);
    assert_eq!(v.parity_energy.len(), s.samples());
    let e_ref = MorseDimer::h2().e_atom;
    let st = s.store().unwrap();
    let m = s.model().unwrap();
    for p in &v.parity_energy {
        let smp = &st.samples[p.index];
        let n = smp.pos.len() as f64;
        assert!((p.dft - (smp.energy - n * e_ref) / n).abs() < 1e-12);
        let pred = m.predict(&smp.pos, &smp.z, false).unwrap();
        assert!((p.model - (pred.energy - n * e_ref) / n).abs() < 1e-12);
        assert_eq!(p.state, sim::activity::SampleState::Fitted);
    }
    assert_eq!(v.curves.len(), 1);
    let c = &v.curves[0];
    assert_eq!(c.z, (1, 1));
    assert_eq!(c.r.len(), sim::activity::CURVE_POINTS);
    let k = c.r.iter().position(|&r| r >= 1.6).unwrap();
    let pos = [[-c.r[k] / 2.0, 0.0, 0.0], [c.r[k] / 2.0, 0.0, 0.0]];
    let pred = m.predict(&pos, &[1, 1], true).unwrap();
    let floor = 2.0 * m.sigma_e; // energy_per_atom models: N s_E
    let epi = (pred.energy_std.unwrap().powi(2) - floor * floor).max(0.0).sqrt();
    assert!((c.sigma[k] - epi).abs() < 1e-15);
    assert!((c.energy[k] - (pred.energy - 2.0 * e_ref)).abs() < 1e-12);
    assert_eq!(c.refused[k], None, "inside the window the model answers");
    assert!(c.refused.last().unwrap().is_some(), "near r_cut the dimer is refused (5a or the envelope)");
    assert_eq!(c.dots.len(), 120, "every stored sample is a two-atom H–H sample");
    assert_eq!(v.pairs[0].counts.iter().sum::<u32>(), 120);
    assert!((v.dimer_floor - floor).abs() < 1e-18 && v.energy_tier == 1e-3);
}

/// Review Focus 5: the build time at S = 4 is reported (never asserted).
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn training_views_build_time_is_reported() {
    // a 4-species store: H, He, Li, Be dimers on the shared Morse form (the species only label
    // channels here; the numbers are a cost probe, not physics)
    let (mut s, _) = four_species_trained(); // see Step 3: builds 400 dimers over the 10 pairs and refits
    let t = std::time::Instant::now();
    let v = s.training_views().unwrap();
    eprintln!("training views at S = 4 (Q′ {}): {:.1} ms (build_ms {:.1}), {} curves, {} parity points",
              s.report().unwrap().q_prime, t.elapsed().as_secs_f64() * 1e3, v.build_ms, v.curves.len(), v.parity_energy.len());
}
```

`four_species_trained()`: a helper next to `trained` that sets references for Z 1–4 to `MorseDimer::h2().e_atom` and adds 40 `morse_sample`s for each of the 10 pairs (`dimer(r, …, za, zb)`), then refits. If the existing new-species test has a cheaper multi-species builder, reuse it.

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p providers --release would_answer training_views`
Expected: compile errors.

- [ ] **Step 3: Implement `views.rs`**

```rust
//! The training views (activity spec §3.3.5, §9, A6): what the Activity tab plots, built from
//! the store and the current posterior — never a teacher call — and `would_answer`, the fallback
//! policy's verdict at a geometry without any of `evaluate`'s side effects.
use super::*;
use sim::activity::{
    PairCurve, PairHistogram, ParityPoint, SampleKind, SampleState, TeachRule, TrainingViews, CURVE_POINTS, CURVE_R_MIN, FORCE_PARITY_CAP,
};

impl SurrogateKs {
    /// Rules 1–7 and the fence at (`pos`, `z`) under `prov`, evaluated as `evaluate` evaluates them
    /// with a fresh variance (the cadence is not consulted, nothing is counted, no state moves
    /// but the descriptor cache): `Ok` where the model would answer, else the rule that refuses
    /// and its atom. Draft mode is ignored (this is the policy's verdict).
    pub fn would_answer(&mut self, pos: &[[f64; 3]], z: &[u8], prov: &Provenance) -> Result<(), (TeachRule, Option<usize>)> {
        if let Err(g) = self.gate(z, prov) {
            return Err((g.reason.rule, g.reason.atom));
        }
        let model = self.model.as_ref().expect("the gate passed");
        let rc2 = model.basis.r_cut * model.basis.r_cut;
        let counts = neighbour_counts(pos, rc2);
        let pred = model.predict(pos, z, true).map_err(|_| (TeachRule::NewEnvironment, None))?;
        let mut variance = match (pred.energy_std, pred.force_std_epistemic.as_ref()) {
            (Some(e), Some(f)) => Some(self.variance_check(model, &pred, e, f)),
            _ => None,
        };
        let novel = self.null_space.as_ref().and_then(|ns| ns.novelty(pos, z).ok()).filter(|(nu, _)| *nu > 1.0).map(|(_, at)| (TeachRule::NewEnvironment, at));
        if novel.is_some() {
            variance = None;
        }
        let (_, envelope) = self.envelope_outside(model, pos, z);
        if let Some(r) = envelope {
            return Err((r.rule, r.atom));
        }
        if let Some(n) = novel {
            return Err(n);
        }
        if let Some((r, _)) = self.class_untrusted(&counts) {
            return Err((r.rule, r.atom));
        }
        if let Some(r) = variance.and_then(|v| if v.accept { None } else { v.reason }) {
            return Err((r.rule, r.atom));
        }
        if !self.near_quarantined(pos, z).is_empty() {
            return Err((TeachRule::NearQuarantine, None));
        }
        Ok(())
    }
```

(Mirror `evaluate`'s order exactly as it stands after Task 2: envelope → novelty → class → variance → fence. If `evaluate` computes the novelty only when the variance is evaluated, that is the case here too, since the variance is always fresh. If the borrow checker objects to `model` borrowed across `&mut self` calls, collect `pred` and the verdicts first, then call `near_quarantined` last, as written.)

```rust
    /// The training views (spec A6). `None` without a fitted model.
    pub fn training_views(&mut self) -> Option<TrainingViews> {
        let t0 = Instant::now();
        let (model, store, report) = (self.model.as_ref()?, self.store.as_ref()?, self.report.as_ref()?);
        let refs = |zs: &[u8]| -> Option<f64> { zs.iter().map(|z| self.refs.get(z).copied()).sum() };
        let state = |i: usize| -> SampleState {
            if i >= self.accumulated {
                return SampleState::Unfitted;
            }
            match self.quarantine.get(i).copied().unwrap_or(Q_NONE) {
                Q_HELD => SampleState::Quarantined,
                Q_SUPERSEDED => SampleState::Superseded,
                Q_BOX => SampleState::BoxExcluded,
                _ => SampleState::Fitted,
            }
        };
        let kind = |t: SampleTag| match t {
            SampleTag::Fallback => SampleKind::Fallback,
            SampleTag::LiveStep => SampleKind::Live,
            _ => SampleKind::Pass,
        };
        let r_cut = model.basis.r_cut;
        let mut parity_energy = Vec::with_capacity(store.len());
        let mut forces: Vec<[f32; 2]> = Vec::new();
        let mut force_total = 0usize;
        let mut dots: BTreeMap<(u8, u8), Vec<(f64, f64, SampleState)>> = BTreeMap::new();
        let mut hist: BTreeMap<(u8, u8), Vec<u32>> = BTreeMap::new();
        const BIN: f64 = 0.1;
        for (i, smp) in store.samples.iter().enumerate() {
            let (Some(eref), Ok(pred)) = (refs(&smp.z), model.predict(&smp.pos, &smp.z, false)) else { continue };
            let n = smp.pos.len() as f64;
            let st = state(i);
            parity_energy.push(ParityPoint { dft: (smp.energy - eref) / n, model: (pred.energy - eref) / n, kind: kind(smp.tag), state: st, index: i });
            if st == SampleState::Fitted {
                for (ft, fm) in smp.forces.iter().zip(&pred.forces) {
                    for a in 0..3 {
                        force_total += 1;
                        forces.push([ft[a] as f32, fm[a] as f32]);
                    }
                }
            }
            for a in 0..smp.pos.len() {
                for b in a + 1..smp.pos.len() {
                    let d = dist(smp.pos[a], smp.pos[b]);
                    let key = (smp.z[a].min(smp.z[b]), smp.z[a].max(smp.z[b]));
                    if smp.pos.len() == 2 {
                        dots.entry(key).or_default().push((d, smp.energy - eref, st));
                    }
                    if st == SampleState::Fitted && d < r_cut {
                        let h = hist.entry(key).or_insert_with(|| vec![0; (r_cut / BIN).ceil() as usize]);
                        h[((d / BIN) as usize).min(h.len() - 1)] += 1;
                    }
                }
            }
        }
        // deterministic thinning to the cap (spec §3.3.5)
        if forces.len() > FORCE_PARITY_CAP {
            let stride = forces.len().div_ceil(FORCE_PARITY_CAP);
            forces = forces.into_iter().step_by(stride).collect();
        }
        let floor2 = if model.energy_per_atom { 2.0 * model.sigma_e } else { model.sigma_e };
        let species = model.basis.species.clone();
        let (epoch, samples, tier) = (self.epoch, store.len(), self.policy.energy_std_per_atom);
        let _ = report;
        // the curves: predictions first (immutable borrows), then the verdicts (&mut self)
        let mut curves = Vec::new();
        for (ia, &za) in species.iter().enumerate() {
            for &zb in &species[ia..] {
                let r: Vec<f64> = (0..CURVE_POINTS).map(|k| CURVE_R_MIN + (r_cut * 0.999 - CURVE_R_MIN) * k as f64 / (CURVE_POINTS - 1) as f64).collect();
                let e_ab = self.refs[&za] + self.refs[&zb];
                let model = self.model.as_ref().expect("fitted");
                let (mut energy, mut sigma) = (Vec::with_capacity(CURVE_POINTS), Vec::with_capacity(CURVE_POINTS));
                for &x in &r {
                    let p = model.predict(&[[-x / 2.0, 0.0, 0.0], [x / 2.0, 0.0, 0.0]], &[za, zb], true).ok()?;
                    energy.push(p.energy - e_ab);
                    if let Some(s) = p.energy_std {
                        sigma.push((s * s - floor2 * floor2).max(0.0).sqrt());
                    }
                }
                let prov = model.provenance.clone();
                let refused = r
                    .iter()
                    .map(|&x| self.would_answer(&[[-x / 2.0, 0.0, 0.0], [x / 2.0, 0.0, 0.0]], &[za, zb], &prov).err().map(|e| e.0))
                    .collect();
                curves.push(PairCurve { z: (za, zb), r, energy, sigma, refused, dots: dots.remove(&(za, zb)).unwrap_or_default(), r_cut });
            }
        }
        let pairs = hist.into_iter().map(|(z, counts)| PairHistogram { z, bin: BIN, counts }).collect();
        Some(TrainingViews {
            epoch,
            samples,
            parity_energy,
            parity_force: forces,
            force_points_total: force_total,
            curves,
            pairs,
            dimer_floor: floor2,
            energy_tier: tier,
            build_ms: t0.elapsed().as_secs_f64() * 1e3,
        })
    }
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}
```

(The first block borrows `self` immutably through `model`/`store`; end those borrows before the curve loop. If the compiler needs it, compute the parity/histogram part in a private `fn parity_and_coverage(&self) -> (...)`.) The provenance in `would_answer` for a dimer is the model's own: the curve asks "would this model answer this dimer at its own settings". `Trainable::training_views` calls the inherent method.

- [ ] **Step 4: Run** `cargo test -p providers --release` → PASS; the S = 4 printout is in the task report. `cargo test -p validate --release --test surrogate` → PASS, unchanged.

- [ ] **Step 5: Physics review.** `physics-reviewer` with spec §9: the parity axes, the band (epistemic, floor 2·s_E for per-atom models), the shading (`would_answer` agrees with `evaluate`; the sweep covers both verdicts), the dimer-slice caption, the histogram's fitted-only population. Apply the review before committing.

- [ ] **Step 6: Commit**

```bash
git add crates/providers/src/surrogate/views.rs crates/providers/src/surrogate/mod.rs crates/providers/src/surrogate/tests.rs
git commit -o -m "feat(providers): training views and a read-only would_answer for the Activity tab

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -- crates/providers/src/surrogate/views.rs crates/providers/src/surrogate/mod.rs crates/providers/src/surrogate/tests.rs
```

---

### Task 5: the worker ships activity in every snapshot **[Opus]**

**Files:**
- Create: `crates/app/src/worker/activity.rs`
- Modify: `crates/app/src/worker/mod.rs`, `crates/app/src/worker/commands.rs`, `crates/app/src/worker/model.rs`
- Test: `crates/app/src/worker/tests.rs`

**Interfaces:**
- Consumes: Tasks 1–4 (`Trainable::{take_teach_events, training_views}`, the status fields, `with_probe`).
- Produces:
  - `pub struct StampedEvent { pub event: sim::TeachEvent, pub step: Option<u64>, pub dispatch_step: Option<u64> }`
  - `pub struct FitPoint { pub epoch: u64, pub samples: usize, pub rmse_e: f64, pub rmse_f: f64, pub trusted: bool, pub at: Instant }`
  - `#[derive(Default)] pub struct ActivityDelta { pub events: Vec<StampedEvent>, pub fit_points: Vec<FitPoint>, pub views: Option<Arc<sim::TrainingViews>>, pub activity_ms: f64, pub teacher_ms: f64, pub dropped: u64 }`
  - `Snapshot::activity: ActivityDelta`; `Command::SetActivityViews(bool)`; `SimWorker::probe(&self) -> Arc<sim::ActivityProbe>`
  - `ModelStatus::{solves: sim::SolveCounts, trust_checks: Vec<sim::activity::TrustCheck>, stuck_info: Option<sim::activity::StuckInfo>}`
  - `pub const VIEWS_MIN_INTERVAL: Duration = Duration::from_secs(2);`

- [ ] **Step 1: Write the failing tests** (`worker/tests.rs`)

```rust
/// A SurrogateKs over the scripted Morse teacher (no GPU; it answers the lone atom of the pass's
/// references, which `MorseDimer` cannot): the activity tests' worker.
fn morse_estimated(scene: Scene) -> SimWorker {
    SimWorker::spawn_test(
        scene,
        Box::new(|| {
            let teacher = ScriptedMorse { morse: sim::MorseDimer::h2(), script: Arc::new(std::sync::Mutex::new(VecDeque::new())) };
            Box::new(SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_teacher_tol(SAMPLE_TOL))
        }),
    )
}

/// Snapshots until `f`, handing each one to `each` first (`snapshot_until` takes an `Fn`; the
/// activity tests collect deltas).
fn snapshots_until(w: &SimWorker, what: &str, mut each: impl FnMut(&Snapshot), f: impl Fn(&Snapshot) -> bool) -> Snapshot {
    let deadline = Instant::now() + WAIT;
    loop {
        assert!(Instant::now() < deadline, "no snapshot with {what} within {WAIT:?}");
        let s = next_snapshot(w);
        each(&s);
        if f(&s) {
            return s;
        }
    }
}

/// Review Focus 1: every teaching solve reaches the viewer once, in order, stamped with the
/// driver's step — through ticks, paused bursts and early publishes alike; the session count
/// equals the model's lifetime count for a fresh model.
#[test]
fn events_arrive_once_in_order_across_ticks_and_early_publishes() {
    let w = morse_estimated(h2_33());
    idle(&w);
    w.send(Command::EnsureModel { autoload: None, plan: SamplingPlan::default() });
    let mut events = Vec::new();
    snapshots_until(&w, "the pass to finish", |s| events.extend(s.activity.events.iter().cloned()), |s| {
        s.dynamics.sampling.as_ref().is_some_and(|p| p.finished) && !s.working
    });
    // play with a flood of speed commands (early publishes)
    w.send(Command::Dyn(DynCommand::Play));
    for k in 0..50 {
        w.send(Command::Dyn(DynCommand::SetSpeed(1 + k % 5)));
    }
    snapshots_until(&w, "200 steps", |s| events.extend(s.activity.events.iter().cloned()), |s| s.dynamics.step >= 200);
    w.send(Command::Dyn(DynCommand::Pause));
    let s = snapshots_until(&w, "idle", |s| events.extend(s.activity.events.iter().cloned()), |s| !s.working);
    let m = s.model.as_ref().unwrap();
    let counted = events.iter().filter(|e| e.event.counts_for_model()).count() as u64;
    assert_eq!(counted, m.solves.total(), "no event lost or duplicated");
    assert!(events.windows(2).all(|p| p[0].event.at <= p[1].event.at), "in order");
    assert!(events.iter().filter(|e| e.event.origin == sim::activity::TeachOrigin::Pass).all(|e| e.step.is_none()));
    assert!(events.iter().filter(|e| e.event.origin == sim::activity::TeachOrigin::Fallback).all(|e| e.step.is_some()));
}

/// The fit history gains one point per new (epoch, fit).
#[test]
fn the_fit_history_gains_a_point_per_refit() {
    let w = morse_estimated(h2_33());
    idle(&w);
    w.send(Command::EnsureModel { autoload: None, plan: SamplingPlan::default() });
    let mut points = Vec::new();
    let s = snapshots_until(&w, "a fitted model", |s| points.extend(s.activity.fit_points.iter().cloned()), |s| {
        s.model.as_ref().is_some_and(|m| m.fitted) && !s.working
    });
    assert_eq!(points.len(), 1);
    assert_eq!(points[0].epoch, s.model.as_ref().unwrap().epoch);
}

/// Review Focus 5: views are built only while enabled, only when (epoch, samples) changed, and
/// at most once per VIEWS_MIN_INTERVAL.
#[test]
fn views_are_built_only_while_open_and_at_most_every_2_s() {
    let w = morse_estimated(h2_33());
    idle(&w);
    w.send(Command::EnsureModel { autoload: None, plan: SamplingPlan::default() });
    snapshot_until(&w, "a fitted model", |s| s.model.as_ref().is_some_and(|m| m.fitted) && !s.working);
    w.send(Command::Dyn(DynCommand::Step));
    let s = idle(&w);
    assert!(s.activity.views.is_none(), "not built while closed");
    w.send(Command::SetActivityViews(true));
    let s = snapshot_until(&w, "views", |s| s.activity.views.is_some());
    let t_first = Instant::now();
    assert_eq!(s.activity.views.as_ref().unwrap().epoch, s.model.as_ref().unwrap().epoch);
    // more snapshots with nothing new: no rebuild
    w.send(Command::Dyn(DynCommand::SetSpeed(3)));
    let s = idle(&w);
    assert!(s.activity.views.is_none());
    // a far teleport (a fallback, a sample, a paused refit: a new epoch) inside the interval:
    // no snapshot before 2 s carries views
    w.send(Command::MoveNucleus { index: 1, pos: [0.0, 0.0, 2.6] });
    let mut early = false;
    snapshots_until(&w, "idle after the teleport", |s| early |= s.activity.views.is_some(), |s| !s.working);
    if t_first.elapsed() < VIEWS_MIN_INTERVAL {
        assert!(!early, "rebuilt inside the interval");
    }
    // the worker is idle: nudge it with a harmless command once the interval has passed
    std::thread::sleep((VIEWS_MIN_INTERVAL + Duration::from_millis(100)).saturating_sub(t_first.elapsed()));
    w.send(Command::Dyn(DynCommand::SetSpeed(2)));
    let s = snapshot_until(&w, "rebuilt views", |s| s.activity.views.is_some());
    assert_ne!(s.activity.views.as_ref().unwrap().epoch, 0);
    w.send(Command::SetActivityViews(false));
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p app --release worker::tests::events_arrive`
Expected: compile error (`activity` field, `SetActivityViews`).

- [ ] **Step 3: Implement**

`commands.rs`: `SetActivityViews(bool)` in `Command`; in `aborts`, `is_edit`, `reaches_driver` it is `false` (a view setting); `Worker::apply` sets `self.activity.views_on = on`.

`worker/activity.rs`:

```rust
//! The worker's side of the Activity view (activity spec §4.3 A7, §6): drains the provider's
//! teaching events once per publish and stamps them with the driver's step, keeps the fit history,
//! builds the training views only while the UI shows them, and accounts its own cost.
use super::*;
use sim::activity::TeachOrigin;

/// The fewest seconds between two training-view builds (spec §6).
pub const VIEWS_MIN_INTERVAL: Duration = Duration::from_secs(2);

#[derive(Clone, Debug)]
pub struct StampedEvent {
    pub event: sim::TeachEvent,
    /// The driver's step when the event reached the worker (`None` for pass solves).
    pub step: Option<u64>,
    /// A background audit's dispatch step (its step less its lag).
    pub dispatch_step: Option<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FitPoint {
    pub epoch: u64,
    pub samples: usize,
    pub rmse_e: f64,
    pub rmse_f: f64,
    pub trusted: bool,
    pub at: Instant,
}

#[derive(Clone, Debug, Default)]
pub struct ActivityDelta {
    pub events: Vec<StampedEvent>,
    pub fit_points: Vec<FitPoint>,
    pub views: Option<Arc<sim::TrainingViews>>,
    /// This worker's activity bookkeeping and teacher time so far (ms; spec §6).
    pub activity_ms: f64,
    pub teacher_ms: f64,
    pub dropped: u64,
}

#[derive(Default)]
pub(super) struct ActivityWork {
    pub(super) views_on: bool,
    last_fit: Option<(u64, Option<(usize, usize, f64, f64)>)>,
    built: Option<(u64, usize, Instant)>,
    activity_ms: f64,
    teacher_ms: f64,
}

impl Worker {
    /// The activity delta for `snap` (called by `publish`, before the model readouts).
    pub(super) fn activity_delta(&mut self, snap: &DynSnapshot, model: Option<&ModelStatus>) -> ActivityDelta {
        let t = Instant::now();
        let mut d = ActivityDelta::default();
        let (pass, step) = (sampling_active(snap), snap.step);
        if let Some(s) = trainable(&mut self.dy) {
            for e in s.take_teach_events() {
                self.activity.teacher_ms += e.solve_ms;
                let step_of = (e.origin != TeachOrigin::Pass && !pass).then_some(step);
                let dispatch = e.audit_lag.map(|lag| step.saturating_sub(lag as u64));
                d.events.push(StampedEvent { event: e, step: step_of, dispatch_step: dispatch });
            }
        }
        if let Some(m) = model.filter(|m| m.fitted) {
            let key = (m.epoch, m.fit);
            if self.activity.last_fit != Some(key) {
                self.activity.last_fit = Some(key);
                if let Some((n, _, e, f)) = m.fit {
                    d.fit_points.push(FitPoint { epoch: m.epoch, samples: n, rmse_e: e, rmse_f: f, trusted: m.trusted, at: Instant::now() });
                }
            }
            let due = self.activity.views_on
                && self.activity.built.is_none_or(|(e, n, at)| (e, n) != (m.epoch, m.samples) && at.elapsed() >= VIEWS_MIN_INTERVAL);
            if due {
                if let Some(v) = trainable(&mut self.dy).and_then(|s| s.training_views()) {
                    self.activity.built = Some((m.epoch, m.samples, Instant::now()));
                    d.views = Some(Arc::new(v));
                }
            }
        }
        self.activity.activity_ms += t.elapsed().as_secs_f64() * 1e3;
        d.activity_ms = self.activity.activity_ms;
        d.teacher_ms = self.activity.teacher_ms;
        d
    }
}
```

`worker/mod.rs`: `mod activity; pub use activity::*;`. `Worker` gains `activity: ActivityWork` and `probe: Arc<sim::ActivityProbe>`. `SimWorker` gains `probe: Arc<sim::ActivityProbe>` and

```rust
    /// The live solve state of this worker's solvers (activity spec A1).
    pub fn probe(&self) -> Arc<sim::ActivityProbe> {
        self.probe.clone()
    }
```

`spawn_with` and `spawn_test` create `let probe = sim::ActivityProbe::new();` and pass a clone into `worker(...)` (new last argument). In `worker`, the `make` closure captures `probe` and builds:

```rust
                let ks = kohn_sham(engine, &gpu, SAMPLE_TOL, 2, &probe, sim::Slot::Main);
                let s = SurrogateKs::new(ks, FallbackPolicy::default()).with_teacher_tol(SAMPLE_TOL).with_probe(probe.clone());
                Box::new(match gpu.as_ref().filter(|_| engine == Backend::Gpu) {
                    Some(ctx) => s.with_auditor(Box::new(audit_teacher(ctx).with_probe(probe.clone(), sim::Slot::Side))),
                    None => s,
                })
```

and `kohn_sham(…, probe, slot)` calls `.with_probe(probe.clone(), slot)` on both `GpuKs` and `CpuKs`. `publish`:

```rust
        let model = self.model_status(&snap.scene, &snap.forces, snap.step);
        let activity = self.activity_delta(&snap, model.as_ref());
```

and `Snapshot { …, activity }`. `model_status` copies `solves`, `trust_checks`, `stuck_info` from `TrainableStatus` into `ModelStatus` (new fields with doc comments). Update the `model()` fixture in `ui/tests.rs` with `solves: sim::SolveCounts::default(), trust_checks: Vec::new(), stuck_info: None`, and every `Snapshot { … }` literal outside the worker (`grep -rn "Snapshot {" crates/app/src`: the ui tests' `snapshot()` helper, viewer/replay/screenshot tests) with `activity: ActivityDelta::default()`. `kohn_sham` gains `probe: &Arc<sim::ActivityProbe>, slot: sim::Slot` and the plain DFT backends call it with `sim::Slot::Main` too, so their solves show on the chip.

- [ ] **Step 4: Run** `cargo test -p app --release worker::` → the three new tests PASS, every existing worker test PASS.

- [ ] **Step 5: The overhead printout** (spec §6; GPU, skips without an adapter)

```rust
/// Spec §6: the activity bookkeeping and the views against teacher time on the H₂ 49³ pass,
/// and the pass's median solve with the probe attached. Printed, never asserted.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn activity_overhead_is_reported() {
    let _g = gpu_lock();
    let Some(ctx) = gpu_ctx() else { return };
    let w = SimWorker::spawn_with(Scene::preset("h2").unwrap(), Backend::Estimated, Some(ctx), Vec::new(), None);
    idle(&w);
    w.send(Command::SetActivityViews(true));
    w.send(Command::EnsureModel { autoload: None, plan: SamplingPlan::default() });
    let mut solve_ms = Vec::new();
    let mut last = ActivityDelta::default();
    let mut builds = Vec::new();
    snapshots_until(
        &w,
        "the pass to finish",
        |s| {
            solve_ms.extend(s.activity.events.iter().filter(|e| e.event.is_solve()).map(|e| e.event.solve_ms));
            if let Some(v) = &s.activity.views {
                builds.push(v.build_ms);
            }
            last = s.activity.clone();
        },
        |s| s.dynamics.sampling.as_ref().is_some_and(|p| p.finished) && !s.working,
    );
    let med = crate::sim::median(&solve_ms).unwrap_or(f64::NAN);
    eprintln!(
        "activity overhead (H₂ 49³ pass, views on): bookkeeping {:.2} ms over {:.0} ms of teacher time = {:.4} %; median solve {med:.1} ms (probe attached); view builds {builds:?} ms",
        last.activity_ms, last.teacher_ms, 100.0 * last.activity_ms / last.teacher_ms
    );
}
```

(For the probe-on vs probe-off median, run `cargo test -p providers --release` with a short ignored diag `diag_probe_cost` in `ks.rs` that times 20 warm 49³ H₂ solves with and without `with_probe` and prints both medians.) Record the numbers in the task report.

- [ ] **Step 6: Commit**

```bash
git add crates/app/src/worker crates/app/src/ui/tests.rs
git commit -o -m "feat(app): the worker ships activity events, fit history and gated training views per snapshot

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- crates/app/src/worker crates/app/src/ui/tests.rs
```

---

### Task 6: the Activity tab: jobs, counters, solve log **[Opus]**

**Files:**
- Create: `crates/app/src/ui/activity/{mod.rs, jobs.rs, log.rs}`
- Modify: `crates/app/src/ui/mod.rs` (`DockTab::Activity`, `UiState::activity`, `PanelInput::probe`, `mod activity`, emit `SetActivityViews` on the edge), `crates/app/src/ui/dock/mod.rs` (icon, peek, body), `crates/app/src/ui/icons.rs` (`Icon::Activity`), `crates/app/src/ui/search.rs` (anchors), `crates/app/src/viewer.rs` (absorb, probe in `PanelInput`, reset on respawn), `crates/app/src/screenshot.rs` (`tab=activity`)
- Test: `crates/app/src/ui/tests.rs`

**Interfaces:**
- Consumes: Task 5's `Snapshot::activity`, `StampedEvent`, `SimWorker::probe`, `Command::SetActivityViews`.
- Produces:
  - `ui::activity::ActivityState` with `pub fn new() -> Self`, `pub fn absorb(&mut self, s: &Snapshot)`, `pub session_solves: u64`, `pub rows: VecDeque<LogRow>`, `pub fit_points: Vec<FitPoint>`, `pub views: Option<Arc<TrainingViews>>`, `pub training_open: bool`, `pub views_sent: bool`, `pub overhead: (f64, f64)`, `pub timeline: Vec<Mark>`
  - `pub struct LogRow { pub at: Duration, pub event: StampedEvent, pub near_face: bool }`, `pub const LOG_ROWS: usize = 5000;`
  - `jobs::{JobKey, JobProgress, JobRow, rows, show}` (spec §7, exact signatures there)
  - `PanelInput::probe: Option<sim::ProbeView>`
  - `pub enum Mark { Fallback(Duration), Audit(Duration, sim::AuditVerdict), Refit(Duration, bool), Quarantine(Duration, usize), Pass(Duration, Option<Duration>, bool) }`

- [ ] **Step 1: Write the failing tests** (`ui/tests.rs`)

```rust
fn event(origin: sim::activity::TeachOrigin, rule: sim::activity::TeachRule, outcome: sim::activity::TeachOutcome, step: Option<u64>) -> crate::sim::StampedEvent {
    let mut e = sim::TeachEvent::new(origin, rule, outcome);
    e.n_atoms = 2;
    e.solve_ms = 92.0;
    e.gap = Some(3.1e-3);
    crate::sim::StampedEvent { event: e, step, dispatch_step: None }
}

/// The log is newest first, counts solves (not skips), keeps 5 000 rows and keeps counting
/// (Review Focus 2).
#[test]
fn the_log_keeps_5000_rows_and_the_counters_keep_counting() {
    use sim::activity::{TeachOrigin as O, TeachOutcome as K, TeachRule as R};
    let scene = Scene::preset("h2").unwrap();
    let mut a = activity::ActivityState::new();
    let mut snap = snapshot(&scene, Backend::Estimated, Some(model()));
    snap.activity.events = (0..6000).map(|i| event(O::Fallback, R::Grade, K::Kept, Some(i))).collect();
    snap.activity.events.push(event(O::Pass, R::Cell, K::NotSolved, None));
    a.absorb(&snap);
    assert_eq!(a.rows.len(), activity::LOG_ROWS);
    assert_eq!(a.session_solves, 6000, "a skip is not a solve");
    assert_eq!(a.rows.back().unwrap().event.event.rule, R::Cell, "newest at the back, drawn first");
}

/// Solves this session add up over workers; a respawn does not reset them.
#[test]
fn session_solves_survive_a_respawn() {
    use sim::activity::{TeachOrigin as O, TeachOutcome as K, TeachRule as R};
    let scene = Scene::preset("h2").unwrap();
    let mut a = activity::ActivityState::new();
    let mut s1 = snapshot(&scene, Backend::Estimated, Some(model()));
    s1.activity.events = vec![event(O::Fallback, R::NoModel, K::Kept, Some(1))];
    a.absorb(&s1);
    let mut s2 = snapshot(&scene, Backend::Gpu, None);
    s2.activity.events = Vec::new();
    a.absorb(&s2);
    let mut s3 = snapshot(&scene, Backend::Estimated, Some(model()));
    s3.activity.events = vec![event(O::Audit, R::Audit, K::Kept, Some(9))];
    a.absorb(&s3);
    assert_eq!(a.session_solves, 2);
}

/// The job rows and their controls (spec §3.3.1): a pass row cancels, a playing solve pauses,
/// an audit has no control and says why.
#[test]
fn job_rows_carry_the_existing_controls() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let mut snap = snapshot(&scene, Backend::Estimated, Some(model()));
    snap.dynamics.sampling = Some(sim::SamplingProgress { done: 41, total: 273, stage: "scans", last_energy: -1.1, cancelled: false, skipped: 0, failed: 0, finished: false });
    let probe = sim::ProbeView { side: sim::activity::SlotView { running: true, scf_step: 4, elapsed_s: 0.2, solves: 3 }, ..Default::default() };
    let inp = PanelInput { probe: Some(probe), ..input(&scene, Some(&snap), &cam) };
    let rows = activity::jobs::rows(&inp, inp.probe.as_ref());
    let pass = rows.iter().find(|r| r.key == activity::jobs::JobKey::Pass).unwrap();
    assert!(matches!(pass.progress, activity::jobs::JobProgress::Fraction { done: 41, total: 273 }));
    assert!(matches!(pass.control, Some((_, Action::CancelSampling))));
    let audit = rows.iter().find(|r| r.key == activity::jobs::JobKey::Audit).unwrap();
    assert!(audit.control.is_none() && audit.no_control_why.is_some());
    assert!(audit.detail.contains("SCF step 4"));
}

/// Review Focus 5: opening Training sends SetActivityViews(true) once; a respawn (views_sent
/// reset by the viewer) sends it again; nothing is sent while the section stays closed.
#[test]
fn the_views_command_is_resent_after_a_worker_respawn() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let ctx = egui::Context::default();
    let mut st = UiState::new("h2", &scene);
    st.dock_open = true;
    st.dock_tab = DockTab::Activity;
    let inp = input(&scene, None, &cam);
    assert!(frame(&ctx, &mut st, &inp, 1400.0, vec![]).commands.is_empty());
    st.activity.training_open = true;
    let out = frame(&ctx, &mut st, &inp, 1400.0, vec![]);
    assert!(matches!(out.commands.as_slice(), [Command::SetActivityViews(true)]));
    assert!(frame(&ctx, &mut st, &inp, 1400.0, vec![]).commands.is_empty());
    st.activity.views_sent = false; // Viewer::start
    assert!(matches!(frame(&ctx, &mut st, &inp, 1400.0, vec![]).commands.as_slice(), [Command::SetActivityViews(true)]));
}
```

Extend `every_layout_draws_and_sends_nothing_without_input`: `DockTab::ALL` now includes `Activity` (the loop covers it); add a snapshot whose `activity` carries a few events of each origin and a `views` built from a fixture (`activity::fixture_views()` in the test module: two parity points, one curve of 96 points, one histogram) so the plots of Task 9 draw too. Add `ui::search` entries for `d-jobs`, `d-counters`, `d-solve-log`, `d-trust`, `d-training`, `d-timeline` (the anchor test fails until each is marked).

`screenshot.rs` parse test (with the existing `ShotUi` tests):

```rust
    #[test]
    fn tab_activity_is_a_ui_value() {
        let s = ShotUi::parse("tab=activity;wait=sampling").unwrap();
        assert_eq!(s.tab.as_deref(), Some("activity"));
    }
```

and `apply` maps `"activity" => DockTab::Activity`; the doc comment's list gains `activity`.

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p app --release ui::tests`
Expected: compile errors (`activity` module, `DockTab::Activity`, `PanelInput::probe`).

- [ ] **Step 3: Implement**

`ui/mod.rs`: `DockTab::Activity` (after `Model` in `ALL`; title "Activity"); `UiState::activity: activity::ActivityState` (`ActivityState::new()` in `UiState::new`); `PanelInput::probe: Option<sim::ProbeView>` (tests' `input()` sets `None`; the viewer's two sites set `self.sim.as_ref().map(|w| w.probe().view())`). At the end of `ui::show`:

```rust
    // the training views are built only while shown (activity spec §6)
    let want = st.dock_open && st.dock_tab == DockTab::Activity && st.activity.training_open;
    if want != st.activity.views_sent {
        st.activity.views_sent = want;
        out.commands.push(Command::SetActivityViews(want));
    }
```

`ActivityState::new()` starts with `views_sent: false`, so nothing is sent while closed. `viewer.rs`: in `on_event_with`, next to `self.log(&s)`, `self.ui.activity.absorb(&s);`; in `Viewer::start`, `self.ui.activity.views_sent = false;`.

`ui/activity/mod.rs` (the state):

```rust
//! The Activity dock tab (activity spec §3.3): what runs, what the teacher was asked and why,
//! and how the model is learning. State lives here; `jobs`, `log`, `trust`, `plots`, `words`
//! and `headline` draw.
pub mod headline;
pub mod jobs;
pub mod log;
pub mod plots;
pub mod trust;
pub mod words;

use crate::sim::{FitPoint, Snapshot, StampedEvent};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Solve-log rows kept (spec §3.3.3).
pub const LOG_ROWS: usize = 5000;

pub struct LogRow {
    pub at: Duration,
    pub event: StampedEvent,
    /// The named atom was within `words::d_box(h)` of a face in the snapshot it arrived with.
    pub near_face: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mark {
    Fallback(Duration),
    Audit(Duration, sim::AuditVerdict),
    Refit(Duration, bool),
    Quarantine(Duration, usize),
    /// start, end (None while running), cancelled
    Pass(Duration, Option<Duration>, bool),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogFilter {
    pub pass: bool,
    pub fallback: bool,
    pub audit: bool,
    pub kept: bool,
    pub not_kept: bool,
}

pub struct ActivityState {
    start: Instant,
    pub rows: VecDeque<LogRow>,
    pub session_solves: u64,
    pub fit_points: Vec<FitPoint>,
    pub views: Option<Arc<sim::TrainingViews>>,
    pub timeline: Vec<Mark>,
    pub filter: LogFilter,
    pub training_open: bool,
    pub views_sent: bool,
    /// (activity ms, teacher ms) of the current worker (spec §6 footer).
    pub overhead: (f64, f64),
    pub dropped: u64,
    last_quarantined: Option<usize>,
    pass_running: bool,
    pub headline: headline::Throttled<Option<headline::Headline>>,
}

impl ActivityState {
    pub fn new() -> ActivityState {
        ActivityState {
            start: Instant::now(),
            rows: VecDeque::new(),
            session_solves: 0,
            fit_points: Vec::new(),
            views: None,
            timeline: Vec::new(),
            filter: LogFilter { pass: true, fallback: true, audit: true, kept: true, not_kept: true },
            training_open: false,
            views_sent: false,
            overhead: (0.0, 0.0),
            dropped: 0,
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
        for e in &a.events {
            if e.event.is_solve() {
                self.session_solves += 1;
            }
            let at = self.since(e.event.at);
            match e.event.origin {
                sim::activity::TeachOrigin::Fallback => self.timeline.push(Mark::Fallback(at)),
                sim::activity::TeachOrigin::Audit => {
                    if let Some(v) = e.event.audit {
                        self.timeline.push(Mark::Audit(at, v));
                    }
                }
                _ => {}
            }
            let near_face = e.event.atom.and_then(|i| scene.nuclei.get(i)).is_some_and(|n| words::face_distance(scene, n.pos) < d_box);
            self.rows.push_back(LogRow { at, event: e.clone(), near_face });
            if self.rows.len() > LOG_ROWS {
                self.rows.pop_front();
            }
        }
        for p in &a.fit_points {
            self.timeline.push(Mark::Refit(self.since(p.at), p.trusted));
            self.fit_points.push(p.clone());
        }
        if let Some(v) = &a.views {
            self.views = Some(v.clone());
        }
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

    /// Session time of an instant (the log's clock).
    pub fn clock(&self, t: Instant) -> Duration {
        self.since(t)
    }

    /// When this state was created (the progress bar's sweep phase).
    pub fn origin(&self) -> Instant {
        self.start
    }

    /// The species of the newest pass reference solve, while the pass is in its references stage
    /// (the headline's "reference H").
    pub fn current_reference(&self) -> Option<u8> {
        self.rows.iter().rev().find_map(|r| match r.event.event.rule {
            sim::activity::TeachRule::Pass(sim::activity::PassJob::Reference { z, .. }) => Some(z),
            _ => None,
        })
    }
}
```

(`words::d_box` and `words::face_distance` are created in this task as two small functions in `words.rs` (Task 8 fills the rest): `pub fn d_box(h: f64) -> f64 { (1.0 / (std::f64::consts::PI * providers::box_affected_face_density(h))).ln() / 2.0 }`, moved out of `box_advice_with`, which now calls it; `pub fn face_distance(scene: &sim::Scene, p: [f64; 3]) -> f64` = half the cell length minus max |coordinate|. `headline::Throttled` is created in Task 7; for this task give `headline.rs` the `Throttled` type and a stub `Headline` struct so the state compiles, as listed in Task 7 Step 3.)

`ui/activity/jobs.rs` implements spec §7's types exactly, plus:

```rust
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
            detail: format!("{} · {}/{}{step}", super::words::stage_word(p.stage), p.done, p.total),
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
            detail: super::headline::steps_text(pv.main.scf_step, last),
            progress: JobProgress::Steps { step: pv.main.scf_step, last },
            elapsed_s: Some(pv.main.elapsed_s),
            eta_s: None,
            control: input.playing.then(|| ("Pause".into(), Action::Pause)),
            no_control_why: (!input.playing).then_some("a paused step finishes; edits abandon it at the next SCF step (CPU: after the solve)"),
        });
    }
    if let Some((s, n)) = probe.and_then(|p| p.refit) {
        v.push(JobRow { key: JobKey::Refit, title: "Refitting".into(), detail: format!("{n} samples"), progress: JobProgress::Indeterminate, elapsed_s: Some(s), eta_s: None, control: None, no_control_why: Some("a refit takes milliseconds to a few seconds and runs between steps") });
    }
    if m.is_some_and(|m| m.audit_in_flight) {
        let detail = match probe.filter(|p| p.side.running) {
            Some(p) => format!("background teacher · SCF step {}", p.side.scf_step),
            None => "background teacher".into(),
        };
        v.push(JobRow { key: JobKey::Audit, title: "Audit".into(), detail, progress: JobProgress::Indeterminate, elapsed_s: probe.map(|p| p.side.elapsed_s), eta_s: None, control: None, no_control_why: Some("an audit finishes its solve on the background teacher; the model keeps answering meanwhile") });
    } else if let Some(p) = probe.filter(|p| p.side.running && d.and_then(|d| d.bond_reference).is_none()) {
        // the auditor's other jobs (spec A8): a density refresh or an isolated-atom density
        v.push(JobRow { key: JobKey::Background, title: "Background teacher".into(), detail: format!("density for the cloud · SCF step {}", p.side.scf_step), progress: JobProgress::Indeterminate, elapsed_s: Some(p.side.elapsed_s), eta_s: None, control: None, no_control_why: Some("a display-only solve on the background teacher; it teaches nothing and never delays a step") });
    }
    if let Some(r) = d.and_then(|d| d.bond_reference) {
        v.push(JobRow { key: JobKey::BondReferences, title: "Bond references".into(), detail: crate::ui::overlays::bond_reference_line(&r), progress: JobProgress::Fraction { done: r.done, total: r.total }, elapsed_s: Some(r.elapsed_s), eta_s: None, control: None, no_control_why: Some("isolated-atom solves for the bond detector, off the stepping path") });
    }
    if let Some(r) = d.and_then(|d| d.relax.as_ref()).filter(|r| r.converged.is_none()) {
        v.push(JobRow { key: JobKey::Relax, title: "Relaxing (FIRE)".into(), detail: crate::ui::relax_lines(r).first().cloned().unwrap_or_default(), progress: JobProgress::Indeterminate, elapsed_s: None, eta_s: None, control: Some(("Stop".into(), Action::StopRelax)), no_control_why: None });
    }
    if let Some(o) = m.and_then(|m| m.pass_offer.as_ref()).filter(|_| !input.sampling()) {
        v.push(JobRow { key: JobKey::PassOffer, title: "First sampling pass (waiting)".into(), detail: format!("{} solves, ≈ {}", o.solves, crate::ui::duration_label(o.eta_s)), progress: JobProgress::Waiting, elapsed_s: None, eta_s: Some(o.eta_s), control: Some(("Start anyway".into(), Action::StartOffered(o.plan.clone()))), no_control_why: None });
    }
    v
}
```

(Fix visibility as needed: `overlays::bond_reference_line` is already `pub`; make `relax_lines`, `duration_label`, `PanelInput::{model, sampling}` `pub(crate)` if they are private.) `show` draws each row as: a status light (accent), title (strong), detail (dim), a 6 px progress bar (`Fraction`: filled; `Steps`/`Indeterminate`: the sweep of Task 7's `headline::sweep_x`), "elapsed · ETA", and the control button (or a hover-only "can't be cancelled" note with `no_control_why`). Clicks push the row's `Action`.

`ui/activity/log.rs`: the counters line and the log.

```rust
pub fn counters(ui: &mut egui::Ui, st: &ActivityState, m: Option<&crate::sim::ModelStatus>) {
    ui.horizontal_wrapped(|ui| {
        if let Some(m) = m {
            ui.label(egui::RichText::new(format!("samples in the model {}", m.samples)).strong())
                .on_hover_text("the samples the model holds (the store); the Model tab shows the same number");
            let c = &m.solves;
            let total = if c.unrecorded_before { format!("≥ {} (earlier solves unrecorded)", (c.total() as usize).max(m.samples)) } else { c.total().to_string() };
            ui.label(egui::RichText::new(format!("teacher solves for this model {total}")).strong()).on_hover_text(format!(
                "pass {} · fallback {} · audit {} · relax checks {} — kept {}, not kept {} (references, similar answers, box-affected, …)",
                c.pass, c.fallback, c.audit, c.verification, c.kept, c.not_kept()
            ));
        }
        ui.label(egui::RichText::new(format!("solves this session {}", st.session_solves)).strong())
            .on_hover_text("teacher solves offered to a model since the app started, over every solver switch");
    });
}
```

`show_log(ui, st, input)`: the filter chips (`selectable_label` × 5) and the table drawn with `egui::ScrollArea::vertical().max_height(260.0).show_rows(ui, row_h, n, |ui, range| …)` over the filtered rows in reverse order (newest first). Each row: `logbook::clock(row.at)`, the step cell (`pass`, `{step}`, or `{dispatch}←` with a hover "dispatched at step d, judged at s"), origin (colour: pass `theme::ACCENT`, fallback `AMBER`, audit `GREEN`, check `theme::INK_DIM`), rule (`words::rule_label`, hover `words::rule_sentence` + "\n\n" + `detail`), outcome (`words::outcome_label`, hover `words::outcome_sentence`), |ΔE| (`format!("{:.2} mHa", gap.abs() * 1e3)`, hover per atom and the signed Ha value), solve (`{ms:.0} ms` or `{s:.1} s` above 10 s, hover the SCF steps). Empty text: "no teacher solves yet — they appear here as they land (estimated solver)". `dropped > 0` adds "k older events were dropped before they reached the view (a fallback storm); the counters include them".

`dock/mod.rs`: icon `Icon::Activity` (a pulse polyline in `icons.rs`, drawn like the others in the 24 × 24 box: `M3 12h4l2-5 4 10 2-5h6`), peek "Activity\n{headline or idle}\n{session} solves this session", the body arm `DockTab::Activity => super::activity::show(ui, st, input, acts)`, and a small accent dot on the strip button while `input.probe` shows work or `input.sampling()`.

`activity::show` (in `mod.rs`): groups `d-jobs` (jobs::show of jobs::rows; "nothing running" when empty), `d-counters`, `d-solve-log`, then `trust::show` (`d-trust`, Task 8; a stub that draws nothing until then), a `CollapsingHeader::new("Training").open(Some(st.activity.training_open))` driven by the state, where a click on its header (`header_response.clicked()`) toggles `st.activity.training_open` (`d-training`, Task 9 fills it; the state, not egui's memory, is the source of truth, so the test's direct write and the views command agree), and `d-timeline` (Task 9). On a DFT backend the training groups are replaced by one note: "The training views are for the estimated solver."

- [ ] **Step 4: Run** `cargo test -p app --release ui::` → PASS (the smoke run covers the new tab; the anchor test passes once every group is marked).

- [ ] **Step 5: Screenshot check**

Run: `cargo run -p app --release -- --screenshot target/activity.png --estimated --sample --ui "tab=activity"` and open the PNG: the jobs group says "nothing running", the counters show the pass's solves, and the log lists the pass rows. Attach it to the task report.

- [ ] **Step 6: Commit**

```bash
git add crates/app/src/ui crates/app/src/viewer.rs crates/app/src/screenshot.rs
git commit -o -m "feat(app): Activity dock tab with the job list, live counters and the solve log

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- crates/app/src/ui crates/app/src/viewer.rs crates/app/src/screenshot.rs
```

---

### Task 7: the status-chip headline and the toolbar progress bar **[Opus]**

**Files:**
- Modify: `crates/app/src/ui/activity/headline.rs`, `crates/app/src/ui/overlays.rs` (`state`, `status_chip`), `crates/app/src/ui/toolbar.rs` (`show` returns its rect), `crates/app/src/ui/mod.rs` (paint the bar)
- Test: `crates/app/src/ui/tests.rs`

**Interfaces:**
- Consumes: Task 6's `ActivityState::headline`, `PanelInput::probe`.
- Produces: `headline::{Headline, BarMode, Throttled, headline(), steps_text(), sweep_x()}`, `pub const REFRESH: Duration = Duration::from_millis(100);`

- [ ] **Step 1: Write the failing tests**

```rust
/// Spec §3.1: one headline per state, first match wins.
#[test]
fn the_headline_names_what_runs() {
    use activity::headline::{headline, BarMode};
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let running = |step| sim::ProbeView { main: sim::activity::SlotView { running: true, scf_step: step, elapsed_s: 0.4, solves: 1 }, ..Default::default() };
    // a DFT step on the GPU backend
    let mut snap = snapshot(&scene, Backend::Gpu, None);
    snap.dynamics.eval.scf_steps = 11;
    let p = running(7);
    let inp = PanelInput { busy: true, probe: Some(p), ..input(&scene, Some(&snap), &cam) };
    let h = headline(&inp, inp.probe.as_ref()).unwrap();
    assert_eq!(h.text, "Solving · SCF step 7 (last solve 11)");
    assert_eq!(h.bar, BarMode::Sweep);
    // estimated: a fallback
    let snap = snapshot(&scene, Backend::Estimated, Some(model()));
    let inp = PanelInput { busy: true, playing: true, probe: Some(running(3)), ..input(&scene, Some(&snap), &cam) };
    assert!(headline(&inp, inp.probe.as_ref()).unwrap().text.starts_with("DFT fallback · SCF step 3"));
    // a pass with its ETA
    let mut snap = snapshot(&scene, Backend::Estimated, Some(crate::sim::ModelStatus { pass_eta_s: Some(100.0), ..model() }));
    snap.dynamics.sampling = Some(sim::SamplingProgress { done: 41, total: 273, stage: "scans", last_energy: -1.1, cancelled: false, skipped: 0, failed: 0, finished: false });
    let inp = PanelInput { busy: true, probe: Some(sim::ProbeView::default()), ..input(&scene, Some(&snap), &cam) };
    let h = headline(&inp, inp.probe.as_ref()).unwrap();
    assert_eq!(h.text, "Sampling 41/273 · bond scan · 1:40 left");
    assert!(matches!(h.bar, BarMode::Fraction(f) if (f - 41.0 / 273.0).abs() < 1e-6));
    // a refit
    let snap = snapshot(&scene, Backend::Estimated, Some(model()));
    let inp = PanelInput { busy: true, probe: Some(sim::ProbeView { refit: Some((0.1, 250)), ..Default::default() }), ..input(&scene, Some(&snap), &cam) };
    assert_eq!(headline(&inp, inp.probe.as_ref()).unwrap().text, "Refitting · 250 samples");
    // the CPU solver (no step count)
    let mut snap = snapshot(&scene, Backend::Cpu, None);
    snap.engine = Backend::Cpu;
    let p = sim::ProbeView { main: sim::activity::SlotView { running: true, scf_step: 0, elapsed_s: 12.3, solves: 0 }, ..Default::default() };
    let inp = PanelInput { busy: true, probe: Some(p), ..input(&scene, Some(&snap), &cam) };
    assert_eq!(headline(&inp, inp.probe.as_ref()).unwrap().text, "Solving (CPU reference) · 12 s");
    // a background audit beside a model-answered run: a second pill
    let snap = snapshot(&scene, Backend::Estimated, Some(model())); // audit_in_flight: true
    let p = sim::ProbeView { side: sim::activity::SlotView { running: true, scf_step: 4, elapsed_s: 0.1, solves: 2 }, ..Default::default() };
    let inp = PanelInput { busy: true, playing: true, probe: Some(p), ..input(&scene, Some(&snap), &cam) };
    let h = headline(&inp, inp.probe.as_ref()).unwrap();
    assert_eq!(h.audit.as_deref(), Some("Audit running · SCF step 4"));
    // idle: no headline (the old idle texts stay)
    let snap = snapshot(&scene, Backend::Gpu, None);
    let inp = PanelInput { probe: Some(sim::ProbeView::default()), ..input(&scene, Some(&snap), &cam) };
    assert!(headline(&inp, inp.probe.as_ref()).is_none());
    assert_eq!(overlays::state(&inp).1, "paused — converged");
}

/// Spec §6: the headline is rebuilt at most every 100 ms.
#[test]
fn the_headline_refreshes_at_most_every_100_ms() {
    use activity::headline::{Throttled, REFRESH};
    let t0 = std::time::Instant::now();
    let mut t = Throttled::new(0u32);
    assert_eq!(*t.get(t0, || 1), 1);
    assert_eq!(*t.get(t0 + REFRESH / 2, || 2), 1, "inside 100 ms: cached");
    assert_eq!(*t.get(t0 + REFRESH, || 3), 3);
}

/// Review Focus 3: a long solve keeps its headline, and its elapsed time moves.
#[test]
fn a_stale_probe_is_not_a_headline() {
    use activity::headline::headline;
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let mut snap = snapshot(&scene, Backend::Cpu, None);
    snap.engine = Backend::Cpu;
    let p = sim::ProbeView { main: sim::activity::SlotView { running: true, scf_step: 0, elapsed_s: 31.0, solves: 0 }, ..Default::default() };
    let inp = PanelInput { busy: true, probe: Some(p), ..input(&scene, Some(&snap), &cam) };
    assert_eq!(headline(&inp, inp.probe.as_ref()).unwrap().text, "Solving (CPU reference) · 31 s");
    let p = sim::ProbeView::default();
    let inp = PanelInput { busy: false, probe: Some(p), ..input(&scene, Some(&snap), &cam) };
    assert!(headline(&inp, inp.probe.as_ref()).is_none(), "a finished solve leaves no headline");
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p app --release ui::tests::the_headline`
Expected: FAIL (stub `headline` returns `None`).

- [ ] **Step 3: Implement `headline.rs`**

```rust
//! The activity headline (spec §3.1): the status chip's state text while work runs, the toolbar
//! bar's mode, and the 100 ms refresh (§6).
use super::super::PanelInput;
use crate::sim::Backend;
use std::time::{Duration, Instant};

pub const REFRESH: Duration = Duration::from_millis(100);
/// One sweep of the indeterminate bar.
pub const SWEEP: Duration = Duration::from_millis(1200);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BarMode {
    Fraction(f32),
    Sweep,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Headline {
    pub text: String,
    /// The second pill: a background audit alongside.
    pub audit: Option<String>,
    pub bar: BarMode,
}

/// A value rebuilt at most every [`REFRESH`].
pub struct Throttled<T> {
    at: Option<Instant>,
    value: T,
}

impl<T> Throttled<T> {
    pub fn new(value: T) -> Throttled<T> {
        Throttled { at: None, value }
    }

    pub fn get(&mut self, now: Instant, f: impl FnOnce() -> T) -> &T {
        if self.at.is_none_or(|a| now.saturating_duration_since(a) >= REFRESH) {
            self.value = f();
            self.at = Some(now);
        }
        &self.value
    }

    /// The cached value, without refreshing it.
    pub fn peek(&self) -> &T {
        &self.value
    }
}

/// "SCF step 7 (last solve 11)", or "SCF step 7" before any solve finished.
pub fn steps_text(step: u32, last: Option<u32>) -> String {
    match last {
        Some(n) => format!("SCF step {step} (last solve {n})"),
        None => format!("SCF step {step}"),
    }
}

/// Left edge (0..1) of the indeterminate bar's 20 % segment at `t`.
pub fn sweep_x(t: Instant, origin: Instant) -> f32 {
    let p = t.saturating_duration_since(origin).as_secs_f32() / SWEEP.as_secs_f32();
    (p.fract() * 1.2) - 0.2
}

fn clock(s: f64) -> String {
    let s = s.max(0.0).round() as u64;
    format!("{}:{:02}", s / 60, s % 60)
}

pub fn headline(input: &PanelInput<'_>, probe: Option<&sim::ProbeView>) -> Option<Headline> {
    let snap = input.snapshot?;
    let d = &snap.dynamics;
    let p = probe.copied().unwrap_or_default();
    let audit = input
        .model()
        .filter(|m| m.audit_in_flight && p.side.running)
        .map(|_| format!("Audit running · SCF step {}", p.side.scf_step));
    let last = (d.eval.scf_steps > 0).then_some(d.eval.scf_steps as u32);
    let mk = |text: String, bar| Some(Headline { text, audit: audit.clone(), bar });
    if let Some(s) = d.sampling.as_ref().filter(|_| input.sampling()) {
        let frac = BarMode::Fraction(if s.total > 0 { s.done as f32 / s.total as f32 } else { 0.0 });
        let stage = super::words::stage_word(s.stage);
        let step = if p.main.running && p.main.scf_step > 0 { format!(" · SCF step {}", p.main.scf_step) } else { String::new() };
        let eta = input.model().and_then(|m| m.pass_eta_s).map(|e| format!(" · {} left", clock(e))).unwrap_or_default();
        return mk(format!("Sampling {}/{} · {stage}{eta}{step}", s.done, s.total), frac);
    }
    if let Some((t, n)) = p.refit {
        let secs = if t >= 0.3 { format!(" · {t:.1} s") } else { String::new() };
        return mk(format!("Refitting · {n} samples{secs}"), BarMode::Sweep);
    }
    if p.main.running {
        if snap.engine == Backend::Cpu && p.main.scf_step == 0 {
            return mk(format!("Solving (CPU reference) · {:.0} s", p.main.elapsed_s), BarMode::Sweep);
        }
        let who = if snap.backend == Backend::Estimated { "DFT fallback" } else { "Solving" };
        return mk(format!("{who} · {}", steps_text(p.main.scf_step, last)), BarMode::Sweep);
    }
    if let Some(r) = d.relax.as_ref().filter(|r| r.converged.is_none()) {
        return mk(format!("Relaxing (FIRE) · step {}", r.steps), BarMode::Sweep);
    }
    audit.clone().map(|a| Headline { text: if input.playing { "playing".into() } else { a.clone() }, audit: input.playing.then_some(a), bar: BarMode::Sweep })
}
```

(Use the relax readout's actual step-count field name; `words::stage_word` maps `"references"` → "reference", `"perturbations"` → "random nudges", `"scans"` → "bond scan", `"hot burst"` → "hot MD burst", others verbatim; it lives in `words.rs`, created here if Task 8 has not landed.) When the reference stage runs, the pass text carries the species: `headline` stays a function of the input, and the caller (`overlays::show`, which owns `UiState`) rewrites " · reference" to " · reference H" with `ActivityState::current_reference()` (the newest `Pass(Reference { z, .. })` row) before caching, giving `Sampling 3/273 · reference H · SCF step 9`. Add a test for the rewrite (`the_reference_stage_names_its_species`: a state with one reference row for Z = 1 and a references-stage snapshot).

`overlays::state(input)` keeps its signature and every old branch (the row-2 tests call it). `overlays::show` (which has `&mut UiState`) computes the throttled headline once per frame, `let h = st.activity.headline.get(Instant::now(), || headline(input, input.probe.as_ref())).clone();`, and hands it to `status_chip`, which uses `h.text` as the state text when `Some` (busy, spinner) and `state(input)` otherwise. `status_chip` draws `h.audit` as a second `theme::pill(ui, a, GREEN)` and requests `ctx.request_repaint_after(REFRESH)` while a headline exists. Because `overlays::show` runs after `toolbar::show` in `ui::show`, the bar below reads the value cached by the previous frame (`peek`): one frame of lag, invisible.

`toolbar::show` returns the panel's `egui::Rect` (`.show(...).response.rect`). In `ui::show`, after `toolbar::show`:

```rust
    if let Some(h) = st.activity.headline.peek().clone() {
        let p = root.ctx().layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("activity-bar")));
        let r = egui::Rect::from_min_max(bar_rect.left_top(), egui::pos2(bar_rect.right(), bar_rect.top() + 3.0));
        p.rect_filled(r, 0.0, theme::LINE);
        let fill = |a: f32, b: f32, c| p.rect_filled(egui::Rect::from_x_y_ranges(r.left() + a.clamp(0.0, 1.0) * r.width()..=r.left() + b.clamp(0.0, 1.0) * r.width(), r.y_range()), 0.0, c);
        match h.bar {
            activity::headline::BarMode::Fraction(f) => fill(0.0, f, theme::ACCENT),
            activity::headline::BarMode::Sweep => {
                let x = activity::headline::sweep_x(Instant::now(), st.activity.origin());
                fill(x, x + 0.2, theme::ACCENT)
            }
        }
        if h.audit.is_some() {
            fill(0.94, 1.0, AMBER);
        }
        let hover = root.interact(r.expand2(egui::vec2(0.0, 3.0)), egui::Id::new("activity-bar-hit"), egui::Sense::click());
        if hover.on_hover_text(format!("{}\n(click: the Activity tab)", h.text)).clicked() {
            st.dock_open = true;
            st.dock_tab = DockTab::Activity;
        }
        root.ctx().request_repaint_after(std::time::Duration::from_millis(33));
    }
```

(`Throttled::peek(&self) -> &T` returns the cached value; `ActivityState::origin()` returns its `start`.) Ruling A1 placement: the toolbar's top edge.

- [ ] **Step 4: Run** `cargo test -p app --release ui::` → PASS; the row-2 test (`overlays::state` idle texts) still passes.

- [ ] **Step 5: Screenshot check.** `cargo run -p app --release -- --screenshot target/chip.png --estimated --ui "wait=sampling"`: the chip reads "Sampling n/N · …" and the bar is partly filled. Attach.

- [ ] **Step 6: Commit**

```bash
git add crates/app/src/ui
git commit -o -m "feat(app): activity headline in the status chip and a progress bar on the toolbar

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- crates/app/src/ui
```

---

### Task 8: plain wording and the trust checklist **[Opus]**

**Files:**
- Modify: `crates/app/src/ui/activity/words.rs`, `crates/app/src/ui/activity/trust.rs`, `crates/app/src/ui/mod.rs` (`trust()` uses plain lines, `TrustView::detail`), `crates/app/src/ui/dock/model.rs` (trusted line hover), `crates/app/src/ui/toolbar.rs` (trust tooltip)
- Test: `crates/app/src/ui/tests.rs`

**Interfaces:**
- Consumes: `ModelStatus::{trust_checks, stuck_info, class_loo}`, `box_blocks_trust`, `box_advice`, `LogRow::near_face`.
- Produces: `words::{rule_label, rule_sentence, outcome_label, outcome_sentence, trust_line, class_line, stuck_sentence, stage_word, d_box, face_distance}`, `trust::{Line, lines, show}`.

- [ ] **Step 1: Write the failing tests**

```rust
/// Spec §5.1 with the box clause: the user's sentence verbatim when the named atom is near a face
/// and the answer was box-affected; the other two cases.
#[test]
fn the_box_clause_names_the_atom_at_the_edge() {
    use activity::words::{rule_sentence, BoxCtx};
    use sim::activity::TeachRule as R;
    assert_eq!(
        rule_sentence(R::Spread, Some(0), BoxCtx { box_affected: true, near_face: true }, None),
        "atom 0 is outside what the model has learned and is touching the box edge — move it inward"
    );
    assert_eq!(
        rule_sentence(R::Grade, Some(2), BoxCtx { box_affected: true, near_face: false }, None),
        "atom 2 is outside what the model has learned — this answer also touches the box walls, so it is not learned here; a bigger cell would help"
    );
    assert_eq!(rule_sentence(R::Grade, Some(2), BoxCtx { box_affected: false, near_face: true }, None), "atom 2 is outside what the model has learned");
    assert_eq!(rule_sentence(R::NewSpecies(8), None, BoxCtx::default(), None), "the model has never seen O — it learns it at the next refit");
    assert_eq!(rule_sentence(R::Untrusted(sim::activity::TrustKey::Samples), None, BoxCtx::default(), Some("Enough samples — 12 of 16 needed (one per independent direction the model can bend)")),
               "the model isn't trusted yet: Enough samples — 12 of 16 needed (one per independent direction the model can bend)");
}

/// Short labels as the user wrote them (spec §3.3.3).
#[test]
fn rule_labels_are_the_codes_the_user_reads() {
    use activity::words::rule_label;
    use sim::activity::{PassJob, TeachRule as R};
    assert_eq!(rule_label(R::Grade, None), "5e grade");
    assert_eq!(rule_label(R::IsolatedAtom, None), "5a lone atom");
    assert_eq!(rule_label(R::Cell, None), "0 cell");
    assert_eq!(rule_label(R::Pass(PassJob::HotBurst), Some(1830.0)), "hot burst 1830 K");
    assert_eq!(rule_label(R::Pass(PassJob::Reference { z: 1, multiplicity: 2 }), None), "reference H m2");
}

/// Review Focus 4 on the diag-3h fixture: the box line is first and "this is why"; the fit's own
/// failed condition is ✗ below it with its number.
#[test]
fn the_three_hydrogen_refusal_names_the_box_first() {
    use sim::activity::{TrustCheck, TrustKey};
    let reason = "max training energy residual 2.2e-3 > 1.0e-3 Ha/atom";
    let m = crate::sim::ModelStatus {
        trusted: false,
        untrusted_reason: Some(reason.into()),
        samples: 145,
        quarantined: 0,
        box_affected: 130,
        box_excluded: 0,
        trust_floor: Some(59),
        trust_checks: vec![
            TrustCheck { key: TrustKey::Samples, met: true, value: 145.0, limit: 29.0 },
            TrustCheck { key: TrustKey::ResidualDofE, met: true, value: 120.0, limit: 30.0 },
            TrustCheck { key: TrustKey::ResidualDofF, met: true, value: 700.0, limit: 30.0 },
            TrustCheck { key: TrustKey::EnergyResidual, met: false, value: 2.2e-3, limit: 1e-3 },
            TrustCheck { key: TrustKey::ForceResidual, met: true, value: 1e-3, limit: 2e-3 },
            TrustCheck { key: TrustKey::ForceNoise, met: true, value: 4e-4, limit: 2e-3 },
            TrustCheck { key: TrustKey::Audit, met: true, value: 0.0, limit: 1e-3 },
        ],
        ..model()
    };
    let scene = Scene::preset("h2").unwrap();
    let st = UiState::new("h2", &scene);
    let lines = activity::trust::lines(&m, &scene, &st);
    assert!(lines[0].why && !lines[0].met && lines[0].text.starts_with("Samples clear of the box walls — 130 of 145 touch them"), "{:?}", lines[0]);
    let e = lines.iter().find(|l| l.text.starts_with("Reproduces its own training energies")).unwrap();
    assert!(!e.met && !e.why && e.text.contains("worst 2.2 mHa/atom, allowed 1 mHa/atom"));
    assert_eq!(lines.iter().filter(|l| l.why).count(), 1);
    // the Model tab's line uses the box sentence, the old reason on hover
    let t = trust(Some(&m), &scene, &st);
    assert!(t.text.starts_with("no — Samples clear of the box walls") && t.detail.as_deref().is_some_and(|d| d.contains(reason)));
}

/// Without the box, the first ✗ is "this is why" and the Model tab says it in plain words.
#[test]
fn the_first_unmet_check_is_this_is_why() {
    use sim::activity::{TrustCheck, TrustKey};
    let m = crate::sim::ModelStatus {
        trusted: false,
        untrusted_reason: Some("n_samples 12 < q_prime 16".into()),
        trust_checks: vec![TrustCheck { key: TrustKey::Samples, met: false, value: 12.0, limit: 16.0 }],
        ..model()
    };
    let scene = Scene::preset("h2").unwrap();
    let st = UiState::new("h2", &scene);
    let lines = activity::trust::lines(&m, &scene, &st);
    assert!(lines[0].why && lines[0].text == "Enough samples — 12 of 16 needed (one per independent direction the model can bend)");
    assert_eq!(trust(Some(&m), &scene, &st).text, "no — Enough samples — 12 of 16 needed (one per independent direction the model can bend)");
}

/// The stuck variant in plain words (spec §5.3).
#[test]
fn the_stuck_line_is_plain() {
    let info = sim::activity::StuckInfo { rule: sim::activity::TeachRule::Spread, atom: Some(0), steps: 50, box_streak: true, gap: Some(2e-3), face_density: 4e-5 };
    let mut scene = Scene::preset("h2").unwrap();
    scene.nuclei[0].pos = [0.0, 0.0, 5.0]; // 1 Bohr from the +z face of the preset's 49³ × 0.25 cell (d_box(0.25) = 4.84)
    assert_eq!(
        activity::words::stuck_sentence(&info, &scene.to_sim().unwrap()),
        "trusted, but not answering here: atom 0 is outside what the model has learned and is touching the box edge — move it inward (the last 50 steps were DFT)"
    );
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p app --release ui::tests::the_box_clause`
Expected: compile errors.

- [ ] **Step 3: Implement `words.rs`**

```rust
//! Plain wording (activity spec §5): codes in, sentences out; the provider's technical text stays
//! on hover. Element symbols from `units` (display only, rule 1); mHa and Å at this edge (rule 3).
use sim::activity::{EnvClass, PassJob, StuckInfo, TeachOutcome as K, TeachRule as R, TrustCheck, TrustKey};
use units::isotopes::symbol;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BoxCtx {
    pub box_affected: bool,
    pub near_face: bool,
}

/// `box_advice`'s distance: where a hydrogen-like tail e^(−2r)/π falls to the box rule's face
/// density (5.35 Bohr at h = 0.15) — the stated estimate, unchanged.
pub fn d_box(h: f64) -> f64 {
    (1.0 / (std::f64::consts::PI * providers::box_affected_face_density(h))).ln() / 2.0
}

/// Distance (Bohr) from `p` to the nearest face of `scene`'s cell.
pub fn face_distance(scene: &sim::Scene, p: [f64; 3]) -> f64 {
    let half = (scene.cell.n as f64 - 1.0) * scene.cell.h / 2.0;
    half - p.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

pub fn stage_word(stage: &str) -> &str {
    match stage {
        "references" => "reference",
        "perturbations" => "random nudges",
        "scans" => "bond scan",
        "hot burst" => "hot MD burst",
        s => s,
    }
}

pub fn rule_label(rule: R, burst_t: Option<f64>) -> String {
    match rule {
        R::Cell => "0 cell".into(),
        R::NoModel => "1 no model".into(),
        R::NewSpecies(z) => format!("2 new element {}", symbol(z)),
        R::OtherSettings => "3 other settings".into(),
        R::Untrusted(_) => "4 untrusted".into(),
        R::Class(_) => "4b class".into(),
        R::IsolatedAtom => "5a lone atom".into(),
        R::Coordination => "5b coordination".into(),
        R::Spread => "5c envelope".into(),
        R::NewEnvironment => "5d new environment".into(),
        R::NoGrade | R::Grade => "5e grade".into(),
        R::ForceUncertainty => "6 force σ".into(),
        R::EnergyUncertainty => "7 energy σ".into(),
        R::NearQuarantine => "fence".into(),
        R::Audit => "8 audit".into(),
        R::RelaxVerification => "relax check".into(),
        R::Pass(PassJob::Reference { z, multiplicity }) => format!("reference {} m{multiplicity}", symbol(z)),
        R::Pass(PassJob::Perturbation) => "nudge".into(),
        R::Pass(PassJob::Scan) => "bond scan".into(),
        R::Pass(PassJob::HotBurst) => match burst_t {
            Some(t) => format!("hot burst {t:.0} K"),
            None => "hot burst".into(),
        },
    }
}

const OUTSIDE: &str = "is outside what the model has learned";

/// Spec §5.1. `untrusted`: the plain line of the first unmet trust check, for rule 4.
pub fn rule_sentence(rule: R, atom: Option<usize>, bx: BoxCtx, untrusted: Option<&str>) -> String {
    let a = |i: Option<usize>| i.map_or("an atom".to_string(), |i| format!("atom {i}"));
    let with_box = |s: String| match (bx.box_affected, bx.near_face) {
        (true, true) => format!("{s} and is touching the box edge — move it inward"),
        (true, false) => format!("{s} — this answer also touches the box walls, so it is not learned here; a bigger cell would help"),
        _ => s,
    };
    match rule {
        R::Cell => format!("{} is too close to the edge of the simulation cell — move it inward", a(atom)),
        R::NoModel => "there is no model yet — this answer helps build one".into(),
        R::NewSpecies(z) => format!("the model has never seen {} — it learns it at the next refit", symbol(z)),
        R::OtherSettings => "this scene's settings (grid, smearing, charge or spin) differ from the model's — the answer cannot be used for it".into(),
        R::Untrusted(_) => format!("the model isn't trusted yet: {}", untrusted.unwrap_or("see the trust checklist")),
        R::Class(EnvClass::Pair) => "the model hasn't learned two-atom arrangements well enough yet".into(),
        R::Class(EnvClass::ManyBody) => "the model hasn't learned three-or-more-atom arrangements well enough yet".into(),
        R::IsolatedAtom => with_box(format!("{} has no neighbour within reach — the model can't describe a lone atom inside a molecule's spin state", a(atom))),
        R::Coordination | R::Spread | R::NoGrade | R::NewEnvironment | R::Grade => with_box(format!("{} {OUTSIDE}", a(atom))),
        R::ForceUncertainty | R::EnergyUncertainty => with_box(match atom {
            Some(i) => format!("the model is unsure here (atom {i}) — this answer teaches it"),
            None => "the model is unsure here — this answer teaches it".into(),
        }),
        R::NearQuarantine => "this geometry is next to one the model could not learn".into(),
        R::Audit => "a routine check of the model against DFT".into(),
        R::RelaxVerification => "checking the relaxed geometry with DFT".into(),
        R::Pass(PassJob::Reference { z, multiplicity }) => format!("sampling: the isolated {} atom (spin multiplicity {multiplicity}), for its reference energy", symbol(z)),
        R::Pass(PassJob::Perturbation) => "sampling: a random nudge of the scene".into(),
        R::Pass(PassJob::Scan) => "sampling: a bond-length scan".into(),
        R::Pass(PassJob::HotBurst) => "sampling: a hot MD burst".into(),
    }
}

pub fn outcome_label(o: K) -> String {
    match o {
        K::Kept => "kept".into(),
        K::KeptBoxAffected => "kept (box)".into(),
        K::KeptByVote => "kept by vote".into(),
        K::KeptReleased { released, .. } => format!("kept, released {released}"),
        K::Thinned => "skipped: similar".into(),
        K::BoxNotKept => "not kept: box".into(),
        K::FenceDisagrees => "not kept: fence".into(),
        K::WaitingForPass => "not kept: waiting for a pass".into(),
        K::NotOffered => "not used: other settings".into(),
        K::StoreRefused => "refused by store".into(),
        K::Reference => "reference".into(),
        K::Failed => "failed".into(),
        K::NotSolved => "not solved".into(),
    }
}

pub fn outcome_sentence(o: K) -> String {
    match o {
        K::Kept => "kept: it will be in the next fit".into(),
        K::KeptBoxAffected => "kept while the model is being built; it touches the box walls and leaves the fit once enough others exist".into(),
        K::KeptByVote => "kept: three fresh answers agree with each other but not with the model — the next refit decides".into(),
        K::KeptReleased { released, superseded } => format!("kept: it agrees with the model, so {released} quarantined samples were released ({superseded} superseded)"),
        K::Thinned => "skipped: too similar to a sample already held (one in ten such is kept)".into(),
        K::BoxNotKept => "not kept: it touches the box walls — the box's squeeze, not the molecule".into(),
        K::FenceDisagrees => "not kept: next to a geometry the model could not learn, and the teacher still disagrees with the model there".into(),
        K::WaitingForPass => "not kept: many answers are held without a model — run a sampling pass".into(),
        K::NotOffered => "not used: other settings".into(),
        K::StoreRefused => "the store refused it (shape or settings)".into(),
        K::Reference => "a reference energy, not a sample".into(),
        K::Failed => "the solve failed".into(),
        K::NotSolved => "not solved".into(),
    }
}

/// Hartree → mHa for the plain lines: 2 significant figures, a trailing ".0" dropped ("1", "2.2", "0.45").
fn mha(x: f64) -> String {
    let v = x * 1e3;
    let s = if v >= 10.0 { format!("{v:.0}") } else if v >= 1.0 { format!("{v:.1}") } else { format!("{v:.2}") };
    s.strip_suffix(".0").map_or(s.clone(), str::to_string)
}

/// Spec §5.3.
pub fn trust_line(c: &TrustCheck) -> String {
    match c.key {
        TrustKey::Samples => format!("Enough samples — {} of {} needed (one per independent direction the model can bend)", c.value as usize, c.limit as usize),
        TrustKey::ResidualDofE => format!("Enough data beyond the fit's own parameters to measure its noise — energies {:.0} of {:.0}", c.value, c.limit),
        TrustKey::ResidualDofF => format!("Enough data beyond the fit's own parameters to measure its noise — forces {:.0} of {:.0}", c.value, c.limit),
        TrustKey::EnergyResidual => format!("Reproduces its own training energies to the accuracy it claims — worst {} mHa/atom, allowed {} mHa/atom", mha(c.value), mha(c.limit)),
        TrustKey::ForceResidual => format!("Reproduces its own training forces — worst {}, allowed {} mHa/Bohr for that atom", mha(c.value), mha(c.limit)),
        TrustKey::ForceNoise => format!("Its force labels agree with each other — scatter {}, allowed {} mHa/Bohr", mha(c.value), mha(c.limit)),
        TrustKey::Audit if c.met => "Last check against DFT — passed".into(),
        TrustKey::Audit => format!("Last check against DFT — found the model off by {} mHa (refit to recover)", mha(c.value)),
    }
}

pub fn class_line(l: &sim::ClassLoo, q_prime: usize) -> String {
    let name = if l.class == "pair" { "Two-atom arrangements" } else { "Three-or-more-atom arrangements" };
    if l.trusted {
        format!("{name} — learned")
    } else if l.samples < q_prime {
        format!("{name} — not yet ({} < {q_prime} samples)", l.samples)
    } else {
        format!("{name} — not yet (leave-one-out force error {:.1}× the bound)", l.force_max)
    }
}

/// Spec §5.3, the stuck variant.
pub fn stuck_sentence(i: &StuckInfo, scene: &sim::Scene) -> String {
    let near = i.atom.and_then(|a| scene.nuclei.get(a)).is_some_and(|n| face_distance(scene, n.pos) < d_box(scene.cell.h));
    let s = rule_sentence(i.rule, i.atom, BoxCtx { box_affected: i.box_streak, near_face: near }, None);
    format!("trusted, but not answering here: {s} (the last {} steps were DFT)", i.steps)
}
```

(`stuck_sentence` takes the driver's `sim::Scene`, which the Model tab has as `snapshot.dynamics.scene`; the test converts the preset with `to_sim()`.)

`trust.rs`:

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct Line {
    pub met: bool,
    /// The refusal's condition ("this is why").
    pub why: bool,
    pub text: String,
    pub hover: String,
}

pub fn lines(m: &crate::sim::ModelStatus, scene: &crate::scene::Scene, st: &crate::ui::UiState) -> Vec<Line> {
    let mut v = Vec::new();
    let boxed = crate::sim::box_blocks_trust(m);
    if let Some((b, n)) = boxed {
        v.push(Line {
            met: false,
            why: true,
            text: format!("Samples clear of the box walls — {b} of {n} touch them. {}", crate::ui::box_advice(scene, st)),
            hover: format!("the fit's own check failed: {}", m.untrusted_reason.as_deref().unwrap_or("?")),
        });
    }
    let first = m.trust_checks.iter().position(|c| !c.met);
    for (k, c) in m.trust_checks.iter().enumerate() {
        v.push(Line { met: c.met, why: boxed.is_none() && Some(k) == first, text: super::words::trust_line(c), hover: format!("value {:.3e}, limit {:.3e}", c.value, c.limit) });
    }
    if boxed.is_none() && first.is_none() && m.fitted {
        v.push(Line { met: true, why: false, text: "Samples clear of the box walls".into(), hover: "box-affected samples are not keeping the model untrusted".into() });
    }
    let q = m.fit.map_or(0, |f| f.1);
    for l in &m.class_loo {
        v.push(Line { met: l.trusted, why: false, text: super::words::class_line(l, q), hover: format!("{} samples, LOO force RMS {:.2}× / max {:.2}× the bound, energy {:.2}× the tier", l.samples, l.force_rms, l.force_max, l.energy_max) });
    }
    v
}
```

`show(ui, st, input)` draws each line with ✓ (GREEN) / ✗ (AMBER), "this is why" as an amber pill on the `why` line, and the hover. `ui/mod.rs` `trust()`: `TrustView` gains `detail: Option<String>`. The `Untrusted` variants use `lines(…).into_iter().find(|l| l.why)` for their text, `format!("no — {}", line.text)`, with `detail = Some(old reason)`. The box variant keeps `BOX_TRUST_TIP` as `tip`. `Stuck` uses `stuck_sentence` when `stuck_info` is present (the old string as `detail`). `trust()` needs the scene and `UiState` for `box_advice`: change it to `trust(m, scene, st)` and update its callers (dock strip dot, toolbar, Model tab, and the M6 tests around `ui/tests.rs:1649`, which call `trust(Some(&…))`), passing `input.scene`/`st`. The Model tab and the toolbar show `detail` on hover after the existing tip.

- [ ] **Step 4: Run** `cargo test -p app --release ui::` → PASS, including the M6 box tests at `ui/tests.rs:1649` (update their expected prefix from "no — {b} of {n} samples touch the box walls" to the new sentence only where they assert the text; the `box_blocks_trust` predicate assertions stay unchanged).

- [ ] **Step 5: Commit**

```bash
git add crates/app/src/ui
git commit -o -m "feat(app): plain wording for refusals and a live trust checklist

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- crates/app/src/ui
```

---

### Task 9: the training plots, the timeline, replay, measurements, docs **[Opus]**

**Files:**
- Modify: `crates/app/src/ui/activity/plots.rs`, `crates/app/src/ui/activity/mod.rs` (Training and Timeline groups, overhead footer), `crates/app/src/replay.rs` (`Expect::{SolveLog, Counters}`), `crates/app/replays/surrogate_far_drag_recovers.replay`, `README.md`, `CLAUDE.md`, `docs/superpowers/notes/2026-10-14-ui-redesign-checklist.md`
- Test: `crates/app/src/ui/tests.rs`, `crates/app/src/replay.rs` tests

**Interfaces:**
- Consumes: `ActivityState::{fit_points, views, timeline}`, `TrainingViews`.
- Produces: `plots::{rmse, parity, pair_curve, histogram, timeline}` (each `fn(ui: &mut egui::Ui, …)`, pure painters), and the pure helpers `plots::{log_range, to_screen, band_polygon}` they share.

- [ ] **Step 1: Write the failing tests**

```rust
/// The plots' maths: log ranges, the band polygon (upper edge forward, lower edge back), the
/// coordinate map; and each plot draws with no data, one point and full data.
#[test]
fn the_training_plots_draw_and_their_maths_holds() {
    use activity::plots::{band_polygon, log_range, to_screen};
    let (lo, hi) = log_range(&[1e-6, 1e-3, 0.0, f64::NAN]).unwrap();
    assert!((lo + 6.0).abs() < 1e-12 && (hi + 3.0).abs() < 1e-12, "zero and NaN are skipped");
    assert_eq!(log_range(&[]), None);
    let r = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(100.0, 50.0));
    let p = to_screen(r, (0.0, 1.0), (0.0, 1.0), 0.5, 0.5);
    assert_eq!((p.x, p.y), (50.0, 25.0));
    let poly = band_polygon(r, (0.0, 2.0), (-1.0, 1.0), &[0.0, 1.0, 2.0], &[0.0, 0.0, 0.0], &[0.1, 0.2, 0.1]);
    assert_eq!(poly.len(), 6);
    assert!(poly[0].y < poly[5].y, "upper edge above the lower one");
    let ctx = egui::Context::default();
    for (views, forces) in [(None, false), (Some(Arc::new(fixture_views())), false), (Some(Arc::new(fixture_views())), true)] {
        let mut st = activity::ActivityState::new();
        st.views = views;
        st.training_open = true;
        st.parity_forces = forces;
        let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 900.0))), ..Default::default() };
        let mut full = ctx.run_ui(raw, |ui| {
            activity::plots::training(ui, &mut st, Some(2000.0));
            activity::plots::timeline(ui, &st);
        });
        full.textures_delta.clear();
    }
}

/// Spec §6: the Activity tab's frame cost with 5 000 rows and full views, printed.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn the_activity_tab_frame_cost_is_reported() {
    use sim::activity::{TeachOrigin as O, TeachOutcome as K, TeachRule as R};
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    st.backend = Backend::Estimated;
    st.dock_open = true;
    st.dock_tab = DockTab::Activity;
    st.activity.training_open = true;
    let mut snap = snapshot(&scene, Backend::Estimated, Some(model()));
    snap.activity.events = (0..5000).map(|i| event(O::Fallback, R::Grade, K::Kept, Some(i))).collect();
    snap.activity.views = Some(Arc::new(fixture_views()));
    st.activity.absorb(&snap);
    let inp = input(&scene, Some(&snap), &cam);
    frame(&ctx, &mut st, &inp, 1400.0, vec![]);
    let t = std::time::Instant::now();
    for _ in 0..60 {
        frame(&ctx, &mut st, &inp, 1400.0, vec![]);
    }
    eprintln!("Activity tab: {:.2} ms per frame (5 000 rows, full views, 1400 px)", t.elapsed().as_secs_f64() * 1e3 / 60.0);
}
```

Replay (`replay.rs` tests): `Expect::parse(&["solve-log", "1"])` → `Expect::SolveLog(1)`; `Expect::parse(&["counters"])` → `Expect::Counters`; `describe` round-trips both.

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p app --release ui::tests::the_training_plots`
Expected: compile errors (`plots::training`, helpers missing).

- [ ] **Step 3: Implement `plots.rs`**

The helpers (pure):

```rust
/// (log10 min, log10 max) over the positive finite values; `None` without one.
pub fn log_range(xs: &[f64]) -> Option<(f64, f64)> {
    let mut it = xs.iter().copied().filter(|x| x.is_finite() && *x > 0.0).map(f64::log10);
    let first = it.next()?;
    Some(it.fold((first, first), |(a, b), x| (a.min(x), b.max(x))))
}

/// Data (x, y) in ranges `xr`, `yr` to a point of `r` (y up).
pub fn to_screen(r: egui::Rect, xr: (f64, f64), yr: (f64, f64), x: f64, y: f64) -> egui::Pos2 {
    let fx = if xr.1 > xr.0 { (x - xr.0) / (xr.1 - xr.0) } else { 0.5 };
    let fy = if yr.1 > yr.0 { (y - yr.0) / (yr.1 - yr.0) } else { 0.5 };
    egui::pos2(r.left() + fx as f32 * r.width(), r.bottom() - fy as f32 * r.height())
}

/// The ±σ band as a closed polygon: y + σ forward, y − σ backward.
pub fn band_polygon(r: egui::Rect, xr: (f64, f64), yr: (f64, f64), x: &[f64], y: &[f64], s: &[f64]) -> Vec<egui::Pos2> {
    let up = x.iter().zip(y).zip(s).map(|((&x, &y), &s)| to_screen(r, xr, yr, x, y + s));
    let down: Vec<egui::Pos2> = x.iter().zip(y).zip(s).map(|((&x, &y), &s)| to_screen(r, xr, yr, x, y - s)).collect();
    up.chain(down.into_iter().rev()).collect()
}
```

`pub fn training(ui: &mut egui::Ui, st: &mut ActivityState, target_t: Option<f64>)` (`target_t`: the plan's `hot_burst_t`, passed by `activity::show` from `UiState::plan`) and `pub fn timeline(ui: &mut egui::Ui, st: &ActivityState)`. `ActivityState` gains `pub parity_forces: bool` (false) and `pub views_at: Option<Instant>` (set in `absorb` when views arrive). `training` draws, in order and each in a `ui.allocate_exact_size(vec2(w, 150), Sense::hover())` rect with `painter_at`:

- **`rmse`**: two series from `st.fit_points` on a log y (mHa/atom and mHa/Bohr, both ×1e3), x = index; filled circles when trusted, hollow when not; hover shows "epoch e · n samples · E x mHa/atom · F y mHa/Bohr". Note under it: "one point per refit this session; a loaded model starts with its file's fit".
- **`parity`**: a segmented toggle E | F (state `st.parity_forces: bool`, add it to `ActivityState`). E: points (dft, model) ×1e3 mHa/atom, the y = x line, the ±tier band (`views.energy_tier` ×1e3 around y = x as a translucent strip); colour by `SampleKind` (pass `theme::ACCENT`, fallback `AMBER`, live `SPARK_E`); filled when `Fitted`, hollow otherwise; hover the nearest point: "sample i · state · DFT x · model y mHa/atom". F: `parity_force` as 1.5 px dots (×1e3 mHa/Bohr) with "showing n of N components" when `force_points_total > parity_force.len()`. Caption: "model vs DFT for the training samples, (E − Σ E_ref)/N; audits are counted as fallbacks until the model library (sidecar v4) tags them".
- **`pair_curve`** (one per `views.curves`): the band polygon (`theme::ACCENT.gamma_multiply(0.25)`), the floor band ±`dimer_floor` (grey, thinner), the dashed ±2·`energy_tier` lines labelled "rule 7 refuses beyond this", the curve (accent, 1.5 px), the dots (filled/hollow by state), the shading under the x axis wherever `refused[k]` is `Some(rule)` (amber, 4 px strip), labelled on hover with `words::rule_label(rule, None)` and "the model won't answer this dimer here", and the rug (histogram bins with counts as 3 px ticks). y in mHa, x in Bohr (Å on hover via `units::bohr_to_angstrom`). Title "H–H" from `units::isotopes::symbol`. Caption (spec §9.4): "the model's answer for this pair alone; in a larger molecule the pair's energy also depends on its neighbours. Band: the model's own uncertainty (what rule 7 judges); grey: the teacher's scatter."
- **`histogram`**: per `views.pairs` the bars over 0..r_cut; then burst temperatures from `st.rows` (rows whose `event.burst_t` is `Some`, 100 K bins), `target_t` drawn as a dashed line. Caption: "instantaneous kinetic temperature of each hot-burst geometry solved this session; a loaded model's earlier bursts are unrecorded".
- An age line under the group: "built {s} s ago at epoch {e}, {build_ms:.1} ms" from `views` and `views_at`.

`timeline(ui, st)`: five lanes (fallbacks, audits, refits, quarantines, passes), x = session time over `[first mark, now]`; fallback ticks binned per pixel (one 1 px tick per occupied pixel column, height by count, capped); audit dots coloured by verdict; refit diamonds (hollow untrusted); quarantine squares with the count; pass bars (hatched when cancelled). Hover: the mark's text.

The overhead footer (spec §6) under the tab: `format!("activity bookkeeping {:.3} % of teacher time ({:.1} ms of {:.0} ms){}", 100.0 * a / t, a, t, views.map_or(String::new(), |v| format!("; views {:.1} ms per build", v.build_ms)))`, dim, hidden while `t == 0`.

- [ ] **Step 4: Replay expects**

`replay.rs`: `Expect::SolveLog(usize)` ("at least n solve-log rows since the mark, every one with a rule label and an outcome label"); `Expect::Counters` ("the samples counter equals `ModelStatus::samples`; solves this session ≥ the rows since the mark"). The mark records `viewer.ui.activity.rows.len()` and `session_solves`. Append to `crates/app/replays/surrogate_far_drag_recovers.replay` after its last `expect`:

```
# Activity tab (activity spec §8): the fallbacks of the drag reached the solve log and the counters
expect solve-log 1
expect counters
```

Run: `cargo test -p app --release replays_pass` → PASS.

- [ ] **Step 5: Measurements.** Run `cargo test -p app --release activity_overhead_is_reported -- --nocapture`, `cargo test -p app --release the_activity_tab_frame_cost_is_reported -- --nocapture`, `cargo test -p providers --release training_views_build_time_is_reported diag_probe_cost -- --nocapture --include-ignored`. Write the numbers into the task report and the README's performance paragraph. If the bookkeeping ratio is ≥ 1 % of teacher time, **stop and report**; do not ship (spec §6).

- [ ] **Step 6: Docs**

- `README.md`: an "Activity tab" section (what the chip, the bar and each group show; the plain wording; that nothing in it causes a solve; the overhead numbers), the dock-tab list, and `tab=activity` in the screenshot flags.
- `CLAUDE.md`: the viewer-layout line gains the Activity tab and `sim::activity`; the `--ui` keys line gains `tab=activity`; the tests line gains the activity tests and the `surrogate_far_drag_recovers` expects.
- `docs/superpowers/notes/2026-10-14-ui-redesign-checklist.md`: new rows 183–190 (status-chip headline, progress bar, Activity tab: jobs, counters, solve log, trust checklist, training views, timeline) with their `file:line` anchors; row 2's text gains "busy states worded by the activity headline (activity spec §3.1)"; a deviation note for A1 (bar on the toolbar's top edge).
- Screenshots: `--screenshot docs/… --estimated --sample --ui "tab=activity"` and `--ui "tab=activity;wait=sampling"`, added wherever the README lists its screenshots.

- [ ] **Step 7: Full verification**

Run: `cargo build --workspace`; `cargo test -p sim --release`; `cargo test -p providers --release`; `cargo test -p surrogate --release`; `cargo test -p app --release`; `cargo test -p validate --release --test surrogate`; `cargo test -p validate --release --test gpu_vs_cpu`. All PASS; the surrogate suite's numbers match Task 2's report. Then run the viewer checklist items for the dock and the status chip by hand (M2 plan Task 7 Step 5) and note the result in `docs/superpowers/notes/2026-10-13-m2-viewer-checklist.md`.

- [ ] **Step 8: Commit**

```bash
git add crates/app/src/ui crates/app/src/replay.rs crates/app/replays/surrogate_far_drag_recovers.replay README.md CLAUDE.md docs/superpowers/notes/2026-10-14-ui-redesign-checklist.md docs/superpowers/notes/2026-10-13-m2-viewer-checklist.md
git commit -o -m "feat(app): Activity training views and timeline; replay expects; README and checklist

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- crates/app/src/ui crates/app/src/replay.rs crates/app/replays/surrogate_far_drag_recovers.replay README.md CLAUDE.md docs/superpowers/notes/2026-10-14-ui-redesign-checklist.md docs/superpowers/notes/2026-10-13-m2-viewer-checklist.md
```
