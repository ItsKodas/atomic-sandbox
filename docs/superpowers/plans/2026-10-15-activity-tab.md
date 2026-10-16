# Activity Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make processing visible and the model's learning watchable. The status chip and a toolbar progress bar say what runs and how far it is. A new **Activity** dock tab holds the job list, three live counters, a live solve log (one row per teacher solve), a plain-words trust checklist and the training views (rmse per epoch, parity, pair curves with a ±σ band, coverage, timeline). No view causes a teacher solve.

**Architecture:**
- `sim::activity` (new) holds plain records: the `ActivityProbe` (live solve state as atomics), `TeachEvent` with its `TeachRule` / `TeachOutcome` codes, `SolveCounts`, `TrustCheck`, `StuckInfo` and `TrainingViews`.
- `providers` records what the policy already decides. `GpuKs`/`CpuKs`/`SurrogateKs` take the probe; `SurrogateKs` emits one event per teaching solve, counts solves (persisted as an optional sidecar header key), exposes trust checks and stuck info, and builds training views on demand.
- `sim::Trainable` gains two defaulted methods. `sim::Dynamics` drains the provider's events right after each `evaluate` and stamps each with the step whose evaluation produced it (Task 1b, plan review M6). The worker drains the driver's buffer, keeps the fit history, gates view builds and ships an `ActivityDelta` in each snapshot.
- `app::ui::activity` (new) holds the state, the job list, the headline and bar, the plain words, and the tab with its plots.

**Tech Stack:** Rust, egui (hand-painted plots, as `ui::dock::graph`), std atomics. No new crates.

**Spec:** `docs/superpowers/specs/2026-10-15-activity-tab-design.md` (§ numbers below refer to it). Requirements: `docs/superpowers/notes/2026-10-13-m5-user-requests.md` ("Activity view…" and its follow-ups; "Three hydrogens will not reach trust").

**Scheduling:** on branch `activity-tab` (worktree `D:\Projects\Physics-activity`) off `main` `dc03aaa`. Before the user-data milestone. Tasks run in order: 1 → 1b → 2 → 3 → 4 (sim and providers), 5 (worker), 6 → 7 → 8 → 9 (UI). Tasks 6–8 depend only on 1, 1b and 5 for types, so 6 may start once 5 lands. Task 1b holds the plan review's amendments to what Task 1 already built (`docs/superpowers/notes/2026-10-15-activity-plan-review.md`; spec §12).

**Model tags** (CLAUDE.md "Model assignment"):
- **[Fable]**: Tasks 2, 3, 4. These touch the surrogate's decision points and numbers: rule codes, trust checks, the read-only "would it answer", the parity data and the ±σ band. Each ends with a **mandatory `physics-reviewer` pass** before it lands. If Fable is unavailable, Opus implements and the review stays mandatory.
- **[Opus]**: Tasks 1, 1b, 5, 6, 7, 8, 9 (records, probe and driver plumbing, worker, UI, docs).

## Global Constraints

- **No policy change.** Every rule, threshold, thinning, quarantine and trust decision in `providers::surrogate` behaves exactly as at `dc03aaa`. The additions record decisions and never make them. Gate for Tasks 2–4: `cargo test -p providers --release` and `cargo test -p validate --release --test surrogate` pass with unchanged numbers (rule 6). Never loosen a tolerance; a failure is investigated.
- **No view causes a teacher solve.** `training_views`, `would_answer`, `trust_checks` and `stuck_info` never call the teacher or the auditor. Tests assert `MorseDimer::calls` is unchanged.
- **Rule 1.** Nothing per element. Element symbols come from `units::isotopes::symbol` (display only). `d_box(h)` is the stated H 1s-tail estimate from `box_advice`, unchanged.
- **Rule 3.** Records cross every boundary in Hartree, Bohr, kelvin, ms. mHa, mHa/Bohr and Å are converted in `app::ui::activity` only (Å through `units::bohr_to_angstrom`).
- **Rule 5.** `sim` holds plain data and never knows the solver. `ForceProvider` does not change. The probe is attached by whoever constructs the provider (the worker). The driver stamps teaching events with its step through `Trainable::take_teach_events` (Task 1b) and never interprets them.
- **One source per displayed verdict.** A check the UI shows next to a policy decision is read from the predicate the policy uses, never re-derived beside it (plan review M2, M3, M5): `trust_checks` and `untrusted_check` share one predicate per condition (Task 3); the class line words `ClassLoo::refusal` (Task 3); the pair curve's refusal styling comes from `would_answer` (Task 4), never from σ.
- **Rate limits (spec §6).** Headline, bar and job rows rebuild at most every **100 ms**. Training views build only while the Training section is open, only when (epoch, samples) changed, at most once per **2 s**.
- **Caps.** Provider event queue **2 048** and driver event buffer **2 048** (each drops the oldest, counts `dropped`). UI log ring **5 000** rows (the burst-temperature histogram and the counters are kept apart from it, so they cover the whole session). Force-parity points **≤ 20 000**. Pair-curve grid **96** points from 0.5 Bohr to r_cut.
- **Timings are reported, never asserted.** The < 1 % criterion of spec §6 is read from the printouts and recorded in the task report and README.
- **Release-only** for SCF/GPU tests (`#[cfg_attr(debug_assertions, ignore)]`); GPU tests skip without an adapter and take `gpu_lock()`.
- **Copy rules.** The plain sentences of spec §5 are pinned verbatim by tests, the user's example included: "atom 0 is outside what the model has learned and is touching the box edge — move it inward". The technical text is always on hover.
- **Commits** end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (a Fable task ends with its own model's line). Use `git add <paths>` then `git commit -o -m "<message>" -- <paths>` (the message before `--`: everything after it is a pathspec). Never pipe a git command. Never push.
- **README / CLAUDE.md** updated in Task 9: the Activity tab, `tab=activity`, the overhead numbers.

## Review Focus

1. **Counters drift from reality.** The user watches "samples in the model" and "teacher solves" tick. An event lost in the early-publish path (`apply_batch` publishes without a tick), double-counted after a panic recovery, or counted for a rule-3 solve makes the numbers disagree with the store. Pinned: `events_arrive_once_in_order_across_ticks_and_early_publishes` (Task 5) and `solve_counts_add_up_over_a_pass_fallbacks_and_audits` (Task 2, kept = samples added).
2. **A long session floods the log.** A box-affected fallback storm emits one event per DFT step for hours. Expected: bounded memory, a responsive tab, counters still exact. Pinned: `the_event_queue_drops_the_oldest_and_counts_them` (Task 2) and `the_log_keeps_5000_rows_and_the_counters_keep_counting` (Task 6).
3. **The chip says "solving" forever.** A solve that errors, is cancelled, or panics must clear the probe. Pinned: `the_probe_clears_on_error_and_cancel` (Task 1, RAII guard) and `a_stale_probe_is_not_a_headline` (Task 7: running with a 30 s-old start and no snapshot change still reads correctly, the elapsed time grows).
4. **"This is why" blames the wrong thing.** The checklist's first ✗, the Model tab line and the event's `Untrusted(key)` must name the condition the provider actually refused on; the box line goes first only under `box_blocks_trust`, and only when the first unmet check is a fit condition. Pinned: `the_first_unmet_trust_check_is_the_refusal` (Task 3, every branch), `the_trust_checks_and_the_refusal_agree_at_equality_and_nan` (Task 3: one predicate per condition), `the_three_hydrogen_refusal_names_the_box_first` (Task 8, diag-3h fixture) and `the_box_is_not_blamed_for_a_failed_audit_or_a_count` (Task 8).
5. **Views stall the worker, go stale after a backend switch, or move the policy.** At S = 4 (Q′ 1232) a view build costs a variance per curve point, on the worker thread between steps. A respawned worker must get `SetActivityViews(true)` again if the section is open. A view build must not change one decision. Pinned: `views_are_built_only_while_open_and_at_most_every_2_s` (Task 5), `the_views_command_is_resent_after_a_worker_respawn` (Task 6), `training_views_change_nothing_the_policy_reads` (Task 4; `would_answer` is `&self`), and the build-time printout for a 4-species store (`training_views_build_time_is_reported`, Task 4), read against the dynamics it throttles.
6. **A number that does not mean what its label says** (plan review M1, M4, M6). The audit check compares a per-atom gap with a per-atom tier and never says "passed" when no audit ran; the "earlier solves unrecorded" caveat survives the autosave; a log row's step is the step whose evaluation produced it, and an audit row says how many answered steps after its dispatch it was judged. Pinned: `the_audit_check_is_per_atom_and_names_forces` (Task 3), `the_audit_line_says_what_ran` (Task 8), `solve_counts_survive_the_sidecar_and_forget_resets_them` (Task 2: load, save, load), `the_driver_stamps_each_event_with_its_step_and_leaves_pass_events_unstamped` (Task 1b) and `fallback_events_carry_the_step_of_their_evaluation` (Task 2).

---

## File Structure

```
crates/sim/src/activity.rs                 # Task 1 (new): probe, events, codes, counts, checks, stuck info, views records; Task 1b: step stamp, TeachBuffer, review amendments
crates/sim/src/lib.rs                      # Task 1: pub mod activity; re-exports; Task 1b: TeachBuffer
crates/sim/src/dynamics.rs                 # Task 1b: the driver drains and stamps teaching events after each evaluate
crates/sim/src/provider.rs                 # Task 1: Trainable::{take_teach_events, training_views}; TrainableStatus::{solves, trust_checks, stuck_info}; Task 3: ClassKey, ClassLoo::refusal
crates/sim/src/test_providers.rs           # Task 1: SharedTrainable delegates the two new methods
crates/providers/src/ks.rs                 # Task 1: GpuKs/CpuKs::with_probe
crates/providers/src/surrogate/mod.rs      # Task 1: SurrogateKs::with_probe (refit phase); Task 2: Handoff.reason: Refusal, events in teach/evaluate; Task 3: audit_atoms, status fields
crates/providers/src/surrogate/ledger.rs   # Task 2 (new): Ledger, push_event, counts, the queue
crates/providers/src/surrogate/policy.rs   # Task 2: Refusal; Gate/envelope_outside/VarianceVerdict carry codes
crates/providers/src/surrogate/quarantine.rs # Task 2: admit → Admission; Task 4: quarantine_distances_cached, near_quarantined_cached (&self)
crates/providers/src/surrogate/trust.rs    # Task 2: untrusted_check; class_untrusted → Refusal; Task 3: shared predicates, trust_checks, untrusted_check from them, class_refusal, stuck_info
crates/providers/src/surrogate/audit.rs    # Task 2: judge_audit, the auditor's failure and discard emit events; Task 3: audit_atoms
crates/providers/src/surrogate/store.rs    # Task 2: keep_sample → bool; counts in save/install/forget
crates/providers/src/surrogate/views.rs    # Task 4 (new): would_answer, training_views
crates/providers/src/surrogate/tests.rs    # Tasks 1–4: tests appended
crates/providers/src/sampling/mod.rs       # Task 2: pass events (record, sample failure, reference, skip, burst T)
crates/surrogate/src/sidecar.rs            # Task 2: optional "teacher_solves" header key (with "complete"), read leniently
crates/app/src/worker/mod.rs               # Task 5: probe, ActivityDelta in Snapshot, publish drains, SimWorker::probe
crates/app/src/worker/activity.rs          # Task 5 (new): draining the driver's stamped events, fit history, view gating, overhead
crates/app/src/worker/commands.rs          # Task 5: Command::SetActivityViews
crates/app/src/worker/model.rs             # Task 5: ModelStatus::{solves, trust_checks, stuck_info}; Task 8: box_blocks_trust keys on the first unmet TrustKey
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

> Done at `fcdd9bd` (report: `.superpowers/sdd/2026-10-15-activity-tab/task-1-report.md`). The records below are as Task 1 built them; Task 1b amends them after the plan review, and later tasks use the amended names.

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
    /// The per-dimer floor 2·s_E (Ha), the fit's residual scatter (teacher noise plus what the basis cannot fit), and the rule-7 per-atom tier (Ha/atom).
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

### Task 1b: Task-1 follow-ups from the plan review **[Opus]**

Task 1 landed at `fcdd9bd` (report: `.superpowers/sdd/2026-10-15-activity-tab/task-1-report.md`), with its review fix `29f5a25` (the probe's refit counter, `ProbeView::refits`; nothing here touches the probe). The plan's physics review (`docs/superpowers/notes/2026-10-15-activity-plan-review.md`, verdict "ready after amendments") changes several records Task 1 built. This task makes those changes against the code as built; it does not redo Task 1. Nothing here decides anything, and no provider behaviour changes: `SurrogateKs` emits no events until Task 2.

| finding | change here | used by |
|---|---|---|
| M6 (log stamps) | `TeachEvent::step`; `TeachBuffer` (the driver's buffer, stamping, cap); `Dynamics` drains after each `evaluate` | Tasks 2, 5, 6 |
| M1 (audit check) | `TrustKey::AuditForces` | Tasks 3, 8 |
| M5 (pair curve) | `PairCurve::force_ratio` (rule 6's ratio per r) | Tasks 4, 9 |
| S7 (5e at γ = ∞) | `TeachRule::UnspannedAtom` | Tasks 2, 8 |
| S10 (`stuck_info` invents a rule) | `StuckInfo::rule: Option<TeachRule>` | Tasks 3, 8 |
| S11 (auditor solves leave no event) | `TeachOutcome::Discarded` | Tasks 2, 8 |
| S18 (one failed `predict` drops every view) | `TrainingViews::{parity_omitted, curves_skipped}` | Tasks 4, 9 |
| S6, S12 (doc only) | `TeachEvent::burst_t` is the post-Verlet, pre-CSVR temperature; `detail` holds every note of the solve | Task 2 |

`SolveCounts` and `ActivityProbe` need no change: M4 persists the "unrecorded" flag in the sidecar record (Task 2 Step 5), and `SolveCounts::unrecorded_before` already carries it in memory.

**Files:**
- Modify: `crates/sim/src/activity.rs`, `crates/sim/src/lib.rs`, `crates/sim/src/dynamics.rs`
- Test: `crates/sim/src/activity.rs` (`mod tests`)

**Interfaces:**
- Consumes: Task 1's records as built.
- Produces (later tasks use these names):
  - `TeachEvent::step: Option<u64>`: the driver step whose evaluation produced the event; `None` for pass events and before the driver drains it
  - `sim::activity::{TeachBuffer, DRIVER_EVENTS}`; `Dynamics::take_teach_events(&mut self) -> Vec<TeachEvent>`, `Dynamics::teach_events_dropped(&self) -> u64`
  - `TeachRule::UnspannedAtom`, `TeachOutcome::Discarded`, `TrustKey::AuditForces`
  - `StuckInfo::rule: Option<TeachRule>`
  - `PairCurve::force_ratio: Vec<f64>`, `TrainingViews::{parity_omitted: usize, curves_skipped: usize}`

- [ ] **Step 1: Write the failing tests** (append inside `mod tests` of `crates/sim/src/activity.rs`)

```rust
    /// Plan review M6: the driver stamps an event with the step whose evaluation produced it,
    /// leaves pass events unstamped (the log shows "pass"), and never overwrites a stamp.
    #[test]
    fn the_driver_stamps_each_event_with_its_step_and_leaves_pass_events_unstamped() {
        let ev = |o| TeachEvent::new(o, TeachRule::Grade, TeachOutcome::Kept);
        let mut b = TeachBuffer::default();
        b.absorb(vec![ev(TeachOrigin::Fallback)], 7);
        b.absorb(vec![ev(TeachOrigin::Pass), ev(TeachOrigin::Audit)], 8);
        let mut pre = ev(TeachOrigin::Fallback);
        pre.step = Some(3);
        b.absorb(vec![pre], 9);
        let steps: Vec<Option<u64>> = b.take().into_iter().map(|e| e.step).collect();
        assert_eq!(steps, vec![Some(7), None, Some(8), Some(3)]);
        assert!(b.take().is_empty(), "a take drains the buffer");
    }

    /// The driver's buffer is bounded like the provider's queue: the oldest go first and are
    /// counted.
    #[test]
    fn the_driver_buffer_drops_the_oldest_and_counts_them() {
        let mut b = TeachBuffer::default();
        let events = (0..DRIVER_EVENTS + 5)
            .map(|i| {
                let mut e = TeachEvent::new(TeachOrigin::Fallback, TeachRule::Grade, TeachOutcome::Kept);
                e.n_atoms = i;
                e
            })
            .collect();
        b.absorb(events, 1);
        assert_eq!(b.dropped(), 5);
        let e = b.take();
        assert_eq!(e.len(), DRIVER_EVENTS);
        assert_eq!(e[0].n_atoms, 5, "the oldest went first");
    }

    /// Plan review S11: a background audit discarded because the model changed is a solve spent
    /// on the model (counted), not a kept sample.
    #[test]
    fn a_discarded_audit_is_counted_and_not_kept() {
        let e = TeachEvent::new(TeachOrigin::Audit, TeachRule::Audit, TeachOutcome::Discarded);
        assert!(e.is_solve() && e.counts_for_model() && !e.outcome.is_kept());
        let mut c = SolveCounts::default();
        c.count(&e);
        assert_eq!((c.audit, c.kept, c.not_kept()), (1, 0, 1));
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p sim --release activity`
Expected: compile errors (`TeachBuffer`, `DRIVER_EVENTS`, `TeachEvent::step`, `TeachOutcome::Discarded` missing).

- [ ] **Step 3: Amend the records** (`crates/sim/src/activity.rs`)

`TrustKey` (replace the enum and its doc):

```rust
/// The trust rule's conditions, in the order `trust_checks` lists them and `untrusted_check`
/// tests them (`providers::surrogate`). The audit condition is `Audit` (the per-atom energy gap,
/// or no failed audit since the refit) or `AuditForces` (the failed audit's worst atom force
/// residual): plan review M1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrustKey {
    Samples,
    ResidualDofE,
    ResidualDofF,
    EnergyResidual,
    ForceResidual,
    ForceNoise,
    Audit,
    AuditForces,
}
```

`TeachRule`: after `NewEnvironment`, add

```rust
    /// 5e at γ = ∞: an atom's descriptor has a component along a direction no training row of
    /// its species spans (5d is the null-space novelty test; plan review S7).
    UnspannedAtom,
```

and give `NewEnvironment` the doc `/// 5d: the geometry reaches directions no training row spans (the null-space novelty).`

`TeachOutcome`: after `NotOffered`, add

```rust
    /// A background audit's answer arrived after the model changed settings or species: not
    /// judged, not kept (plan review S11).
    Discarded,
```

`TeachEvent`: change the `detail` and `burst_t` docs and add `step` after `at`:

```rust
    /// The driver step whose evaluation produced this event, stamped by `sim::Dynamics` right
    /// after that `evaluate` (plan review M6); `None` for a sampling-pass event. A background
    /// audit carries the step it was judged at; `audit_lag` says how many answered steps after
    /// its dispatch that was.
    pub step: Option<u64>,
```

```rust
    /// Every note the provider wrote for this solve, one per line (the technical text, shown on
    /// hover; plan review S12).
    pub detail: String,
```

```rust
    /// A hot-burst geometry's instantaneous kinetic temperature (K, `Scene::temperature`, 3N − 3
    /// dof): the Verlet velocities v(t + dt) at the solved geometry, before the CSVR thermostat
    /// rescales them (plan review S6).
    pub burst_t: Option<f64>,
```

and `step: None,` in `TeachEvent::new`.

`StuckInfo`:

```rust
    /// The rule of the streak's latest fallback; `None` when the provider stored no reason
    /// (plan review S10: never a guessed rule).
    pub rule: Option<TeachRule>,
```

`PairCurve`: after `sigma`, add

```rust
    /// Rule 6's ratio per r: max over the dimer's atoms of σ_F,epi/(√3 · bound), the quantity
    /// rule 6 compares with 1 (plan review M5); empty without a posterior.
    pub force_ratio: Vec<f64>,
```

`TrainingViews`: after `force_points_total`, add

```rust
    /// Stored samples the parity plot omits (a species without a model reference, or a
    /// failed prediction; plan review S18).
    pub parity_omitted: usize,
```

and after `curves`, add

```rust
    /// Species pairs whose curve was skipped because a prediction on its grid failed.
    pub curves_skipped: usize,
```

The driver's buffer (after `SolveCounts`):

```rust
/// Teaching events the driver holds between two drains by the worker (plan review M6).
pub const DRIVER_EVENTS: usize = 2048;

/// The driver's teaching events since the worker's last drain (plan review M6). `sim::Dynamics`
/// moves the provider's events here right after each `evaluate`, stamped with the step whose
/// evaluation produced them, so a tick of n steps gives n stamps; a pass event keeps `None`, and
/// a stamp an event already has is kept. Bounded: the oldest is dropped and counted.
#[derive(Debug, Default)]
pub struct TeachBuffer {
    events: std::collections::VecDeque<TeachEvent>,
    dropped: u64,
}

impl TeachBuffer {
    pub fn absorb(&mut self, events: Vec<TeachEvent>, step: u64) {
        for mut e in events {
            if e.origin != TeachOrigin::Pass && e.step.is_none() {
                e.step = Some(step);
            }
            if self.events.len() == DRIVER_EVENTS {
                self.events.pop_front();
                self.dropped += 1;
            }
            self.events.push_back(e);
        }
    }

    pub fn take(&mut self) -> Vec<TeachEvent> {
        self.events.drain(..).collect()
    }

    pub fn dropped(&self) -> u64 {
        self.dropped
    }
}
```

`crates/sim/src/lib.rs`: add `TeachBuffer` to the `pub use activity::{…}` list.

- [ ] **Step 4: The driver drains and stamps** (`crates/sim/src/dynamics.rs`)

Field `teach: crate::activity::TeachBuffer` on `Dynamics` (`Default::default()` in `new`), and in the internals:

```rust
    /// Moves the provider's teaching events into the driver's buffer, stamped with `step` (plan
    /// review M6). An associated function, so `step_once` can call it while it still borrows
    /// `self.forces` and `self.pins`.
    fn collect_teach_events(provider: &mut dyn ForceProvider, buf: &mut crate::activity::TeachBuffer, step: u64) {
        if let Some(t) = provider.as_trainable() {
            buf.absorb(t.take_teach_events(), step);
        }
    }
```

and the public pair (next to `snapshot`):

```rust
    /// The teaching events since the last call, oldest first, each stamped with the step whose
    /// evaluation produced it (`crate::activity::TeachBuffer`; the worker drains this once per
    /// publish). Events the provider queued outside an evaluation (a background audit judged
    /// between ticks, a pass solve) are collected now, at the current step.
    pub fn take_teach_events(&mut self) -> Vec<crate::activity::TeachEvent> {
        Self::collect_teach_events(self.provider.as_mut(), &mut self.teach, self.step);
        self.teach.take()
    }

    /// Events the driver's buffer dropped since the start (the worker shows the count).
    pub fn teach_events_dropped(&self) -> u64 {
        self.teach.dropped()
    }
```

Call sites, each **before** the result is unwrapped, so a failed or refused solve is stamped too:

- `fn evaluate`: `None => { let r = self.provider.evaluate(&self.scene, should_abort); Self::collect_teach_events(self.provider.as_mut(), &mut self.teach, self.step); r? }`.
- `step_once`: the integrator's evaluation is at the positions of step `self.step + 1`:

```rust
        let stepped = self.integrator.step(&mut self.scene, forces_old, self.provider.as_mut(), &mut augment, should_abort);
        // the step's evaluation is at step `self.step + 1`'s positions (plan review M6)
        Self::collect_teach_events(self.provider.as_mut(), &mut self.teach, self.step + 1);
        let (f_new, stats) = stepped?;
```

- the relax verification (`force_teacher_next` then `self.provider.evaluate(…)`) and `refresh_density`: bind the result first, `let r = self.provider.evaluate(&self.scene, should_abort); Self::collect_teach_events(self.provider.as_mut(), &mut self.teach, self.step);`, then `match r { … }` unchanged.

The end-to-end pin (a `SurrogateKs` driven by `Dynamics`, one stamp per stepped fallback) lands in Task 2 as `fallback_events_carry_the_step_of_their_evaluation`: no provider emits events before Task 2.

- [ ] **Step 5: Run**

Run: `cargo test -p sim --release activity` → PASS (the 5 Task-1 tests and the 3 new ones). `cargo test -p sim --release` → PASS (the dynamics tests unchanged: no provider emits yet). `cargo build --workspace` with zero warnings (a `match` on `TeachRule`, `TeachOutcome` or `TrustKey` added since Task 1 gains its arm; none exists at `fcdd9bd`).

- [ ] **Step 6: Commit**

```bash
git add crates/sim/src/activity.rs crates/sim/src/lib.rs crates/sim/src/dynamics.rs
git commit -o -m "feat(sim): activity follow-ups from the plan review — driver-stamped events, audit-forces key, curve force ratio, discarded audits

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- crates/sim/src/activity.rs crates/sim/src/lib.rs crates/sim/src/dynamics.rs
```

---

### Task 2: rule and outcome codes, one event per teaching solve, lifetime counts **[Fable — physics-review gate]**

**Files:**
- Create: `crates/providers/src/surrogate/ledger.rs`
- Modify: `crates/providers/src/surrogate/{mod.rs, policy.rs, quarantine.rs, trust.rs, audit.rs, store.rs}`, `crates/providers/src/sampling/mod.rs`, `crates/surrogate/src/sidecar.rs`
- Test: `crates/providers/src/surrogate/tests.rs`, `crates/surrogate/src/sidecar.rs` tests

**Interfaces:**
- Consumes: Task 1's `TeachEvent`, `TeachRule`, `TeachOutcome`, `TeachOrigin`, `PassJob`, `EnvClass`, `TrustKey`, `SolveCounts`; Task 1b's `TeachEvent::step`, `TeachRule::UnspannedAtom`, `TeachOutcome::Discarded`, `Dynamics::take_teach_events`.
- Produces:
  - `pub(super) struct Refusal { text: String, rule: TeachRule, atom: Option<usize> }` (policy.rs). `Gate.reason`, `Handoff.reason`, `VarianceVerdict.reason`, `envelope_outside`'s second value, `class_untrusted`'s first value and `streak_reason` become `Refusal`. Every `text` is **byte-identical** to today's string.
  - `pub(super) struct Admission { kept: bool, outcome: TeachOutcome }` returned by `admit`
  - `SurrogateKs::untrusted_check(&self) -> Option<(TrustKey, String)>`; `untrusted_reason()` = `untrusted_check().map(|c| c.1)` (Task 3 keeps the signature and derives the body from the trust-check list, plan review M2)
  - `SurrogateKs::push_event(&mut self, e: TeachEvent)` (`pub(crate)`, used by `crate::sampling`), `SurrogateKs::solve_counts(&self) -> SolveCounts`, `Trainable::take_teach_events` implemented
  - `pub const EVENT_QUEUE: usize = 2048;`
  - `surrogate::sidecar::{TeacherSolves, write_full}`, `TeacherSolves::complete: bool`, `Sidecar::teacher_solves: Option<TeacherSolves>` and `Sidecar::teacher_solves_note: Option<String>` (plan review M4, S17)

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
    // every note of the solve, one per line (plan review S12)
    assert!(!e[0].detail.is_empty() && e[0].detail.lines().all(|l| r.stats.notes.iter().any(|n| n == l)), "{:?} vs {:?}", e[0].detail, r.stats.notes);
    assert_eq!(e[0].step, None, "the provider does not stamp: the driver does (Task 1b)");
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
    assert!(matches!(e[0].rule, R::IsolatedAtom | R::Coordination | R::Spread | R::Grade | R::UnspannedAtom | R::NewEnvironment), "{:?}", e[0]);
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
/// "unrecorded before", and that flag survives the next save and load (plan review M4: the
/// session autosave round-trips every model through the file); a forget resets them.
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
    // load → one more solve → save (the autosave) → load: still "earlier solves unrecorded"
    always_unsure(&mut back);
    back.evaluate(&h2(1.6), &mut never()).unwrap();
    back.save_sidecar(&path).unwrap();
    let (mut again, _) = trained(1.2, 2.2, 0, FallbackPolicy::default());
    again.install_sidecar(surrogate::sidecar::read(&path).unwrap()).unwrap();
    assert_eq!(again.solve_counts(), back.solve_counts());
    assert!(again.solve_counts().unrecorded_before && again.solve_counts().fallback == 1);
    // an unreadable record never refuses the model: it reads as unrecorded, with a note
    let mut sc = surrogate::sidecar::read(&path).unwrap();
    sc.teacher_solves = None;
    sc.teacher_solves_note = Some("the record is malformed".into());
    again.install_sidecar(sc).unwrap();
    assert!(again.solve_counts().unrecorded_before && again.solve_counts().total() == 0);
    assert!(Trainable::take_notes(&mut again).iter().any(|n| n.contains("the record is malformed")));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Plan review M6: driven by `sim::Dynamics`, each fallback's event carries the step whose
/// evaluation produced it (stamped right after that evaluate, Task 1b), so one tick of several
/// steps gives distinct, increasing stamps; the provider itself stamps nothing.
#[test]
fn fallback_events_carry_the_step_of_their_evaluation() {
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    always_unsure(&mut s);
    let mut scene = h2(1.6);
    scene.nuclei[1].vel = [2e-4, 0.0, 0.0];
    let (p, _sh) = driven(s);
    let mut d = Dynamics::new(scene, p, 0.25, 1);
    let first = d.take_teach_events();
    assert!(first.iter().all(|e| e.step == Some(0)), "the evaluation at construction is step 0: {first:?}");
    d.apply(DynCommand::Step(5));
    d.tick(f64::INFINITY, &mut never());
    let steps: Vec<u64> = d
        .take_teach_events()
        .iter()
        .filter(|e| e.origin == sim::activity::TeachOrigin::Fallback)
        .map(|e| e.step.expect("stamped by the driver"))
        .collect();
    // a refit inside the tick may end the fallbacks early; the stamps stay distinct
    assert!(steps.len() >= 2, "{steps:?}");
    assert!(steps.windows(2).all(|w| w[0] < w[1]) && steps.iter().all(|s| (1..=5).contains(s)), "{steps:?}");
}

/// A background audit job for `s` at `scene` under `prov` (the auditor's own bookkeeping
/// fields; the tests below hand its answer to `handle_answer` directly).
fn audit_answer(s: &SurrogateKs, scene: &Scene, prov: Provenance, result: Result<(f64, Vec<[f64; 3]>, Option<f64>, Option<Display>), String>) -> AuditAnswer {
    let generation = s.auditor.as_ref().expect("an auditor").generation();
    let job = AuditJob { scene: scene.clone(), prov, generation, answered: s.answered, kind: JobKind::Audit, id: 0 };
    AuditAnswer { job, result, reference: None, ms: 7.0 }
}

/// Plan review S11: the background auditor's failed solve and its discarded answer are solves
/// spent on the model, so they leave an event (origin Audit) and are counted, never kept.
#[test]
fn a_failed_or_discarded_background_audit_emits_an_event() {
    use sim::activity::{TeachOrigin as O, TeachOutcome as K, TeachRule as R};
    let (mut s, _, _, _) = with_gated_auditor(5, 0.0);
    let scene = h2(1.5);
    let prov = s.provenance_for(&scene);
    let before = s.solve_counts();
    s.handle_answer(audit_answer(&s, &scene, prov, Err("the solve diverged".into())));
    let e = drain(&mut s);
    assert_eq!(e.len(), 1);
    assert_eq!((e[0].origin, e[0].rule, e[0].outcome), (O::Audit, R::Audit, K::Failed));
    assert!(e[0].detail.contains("the solve diverged"));
    // an answer for other settings than the model's: discarded, not judged
    let mut charged = scene.clone();
    charged.charge = 1;
    let other = s.provenance_for(&charged);
    let teacher = MorseDimer::h2().evaluate(&scene, &mut never()).unwrap();
    s.handle_answer(audit_answer(&s, &scene, other, Ok((teacher.energy, teacher.forces, None, None))));
    let e = drain(&mut s);
    assert_eq!((e[0].origin, e[0].outcome), (O::Audit, K::Discarded));
    let c = s.solve_counts();
    assert_eq!((c.audit - before.audit, c.kept - before.kept), (2, 0));
}
```

(`AuditAnswer`, `AuditJob` and `JobKind` are `audit.rs`'s `pub(super)` types; import them into the test module if `super::*` does not bring them.)

Sidecar test (`crates/surrogate/src/sidecar.rs` tests):

```rust
    /// The optional "teacher_solves" key (activity spec A2): written only when given, read back
    /// exactly with its "complete" flag (plan review M4), absent → None; a file without it is
    /// byte-identical to the old writer's.
    #[test]
    fn teacher_solves_is_an_optional_header_key() {
        let (orig, dir) = fixture_sidecar(); // the fixture the round-trip tests already build
        let a = dir.join("a.model");
        let b = dir.join("b.model");
        write_with_flags(&a, &orig.model, &orig.accumulator, orig.report.as_ref(), &orig.samples, None, &[], &[]).unwrap();
        write_full(&b, &orig.model, &orig.accumulator, orig.report.as_ref(), &orig.samples, None, &[], &[], None).unwrap();
        assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
        for complete in [true, false] {
            let t = TeacherSolves { pass: 273, fallback: 70, audit: 18, verification: 1, kept: 343, complete };
            write_full(&b, &orig.model, &orig.accumulator, orig.report.as_ref(), &orig.samples, None, &[], &[], Some(&t)).unwrap();
            let back = read(&b).unwrap();
            assert_eq!((back.teacher_solves, back.teacher_solves_note), (Some(t), None));
        }
        assert_eq!(read(&a).unwrap().teacher_solves, None);
    }

    /// Plan review S17: a malformed optional record is an error of the record only — `read`
    /// reports it as `teacher_solves_note` and never refuses the file through it.
    #[test]
    fn a_malformed_teacher_solves_record_is_reported_not_fatal() {
        let ok = Parser::parse(r#"{"pass": 1, "fallback": 2, "audit": 3, "verification": 0, "kept": 4, "complete": false}"#).unwrap();
        assert_eq!(decode_teacher_solves(&ok), Ok(TeacherSolves { pass: 1, fallback: 2, audit: 3, verification: 0, kept: 4, complete: false }));
        for bad in [r#"{"pass": 1}"#, r#"{"pass": -1, "fallback": 2, "audit": 3, "verification": 0, "kept": 4, "complete": true}"#, r#"[1, 2]"#] {
            assert!(decode_teacher_solves(&Parser::parse(bad).unwrap()).is_err(), "{bad}");
        }
    }
```

(Use whatever the existing round-trip tests name their fixture; if it is inline, factor it into `fixture_sidecar()` in the test module. `Parser::parse` is the module's header parser.)

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p providers --release -- event solve_counts outcomes other_settings` and `cargo test -p surrogate --release teacher_solves`
Expected: compile errors (`take_teach_events` not implemented, `Refusal`, `push_event`, `write_full`, `TeacherSolves::complete`, `decode_teacher_solves` missing).

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
| 5e ∞ (an unspanned atom) | `UnspannedAtom` (plan review S7) | `worst_grade.1` |
| 5e | `Grade` | `worst_grade.1` |
| `evaluate` 5d (novelty) | `NewEnvironment` | `at` |
| `variance_check` rule 6 | `ForceUncertainty` | `j` |
| rule 7 | `EnergyUncertainty` | — |
| `class_untrusted` (each branch) | `Class(if c == 0 { EnvClass::Pair } else { EnvClass::ManyBody })` | — |
| fence (`evaluate`, "within … of a quarantined geometry") | `NearQuarantine` | — |
| audit reason (`evaluate`) | `Audit` | — |
| relax verification | `RelaxVerification` | — |

`trust.rs`: rename the body of `untrusted_reason` into `pub fn untrusted_check(&self) -> Option<(TrustKey, String)>`, each branch returning its key (`Samples`, `ResidualDofE`, `ResidualDofF`, `EnergyResidual`, `ForceResidual`, `ForceNoise`, and for the audit `AuditForces` when `audit_force` is set, else `Audit`) with today's string. Task 3 replaces this body with the first unmet entry of the trust-check list (plan review M2); the signature and the strings stay. Then:

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

with `fn origin_of(h: &Handoff) -> TeachOrigin { if h.audit { TeachOrigin::Audit } else if h.reason.rule == TeachRule::RelaxVerification { TeachOrigin::Verification } else { TeachOrigin::Fallback } }`. Document on `origin_of` (plan review N2): a relax verification whose gate refuses (rules 1–4) carries the gate's rule, so it is logged as origin Fallback with that rule, not as a "check"; the `verification` count misses it. That is deliberate: the gate's reason is the more specific one. At the end of `teach`, before `Ok(...)`:

```rust
        let mut ev = TeachEvent::new(origin_of_parts(h_audit, rule), rule, outcome);
        ev.atom = atom;
        // every note of the step, the admission's numbers included (plan review S12)
        ev.detail = stats.notes.join("\n");
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

(Plan review N3: a drag held against the cell margin pushes one such row per refused `evaluate`. They are bounded by the queues and the log ring, and not counted; nothing more is needed.)

`audit.rs` `judge_audit`: after the verdict, push `TeachEvent { origin: Audit, rule: Audit, outcome: admission.outcome (StoreRefused if keep_sample failed), gap: Some(gap), face_density: face, box_affected: box_rho.is_some(), audit: Some(verdict), audit_lag: Some(unaudited), solve_ms: ms, detail: notes.join("\n"), n_atoms }`. The two auditor paths that judge nothing still spent a teacher solve on the model, so they push an event too (plan review S11; pinned by `a_failed_or_discarded_background_audit_emits_an_event`):

- `handle_answer`'s `(JobKind::Audit, Err(e))` arm: `TeachEvent::new(TeachOrigin::Audit, TeachRule::Audit, TeachOutcome::Failed)` with `detail` = the note it pushes, `n_atoms` = the job's nuclei, `solve_ms` = `ans.ms`;
- `judge_audit`'s discard ("the model changed settings or species since its dispatch"): outcome `Discarded`, `detail` = that note, `audit_lag: Some(unaudited)`, `solve_ms: ms`.

A cancelled auditor solve (a Forget or an installed model abandons the job; its answer comes back of an older generation) pushes nothing and is not counted: the counter's hover says so (Task 6).

`crate::sampling` (`sampling/mod.rs`): `record(&mut self, surrogate, scene, r: ForceResult, tag, burst_t: Option<f64>)` (the new last argument; `None` for perturbations and scans) reads `let (solve_ms, scf) = (r.stats.solve_ms as f64, r.stats.scf_steps);` before `r` is consumed, and pushes after `add_sample_with_face`:

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

`burst_step` passes the temperature of the solved geometry's Verlet velocities v(t + dt), taken **before** the CSVR thermostat rescales them (plan review S6; `record` runs after `csvr.apply` today, so take it in between):

```rust
            Ok(r) => {
                let f_new = r.forces.clone();
                if !fresh {
                    Integrator::half_kick(&mut work, &f_new, dt);
                }
                // the instantaneous kinetic T (3N − 3 dof) of the solved geometry, before CSVR
                let burst_t = work.temperature();
                if !fresh {
                    self.burst.csvr.apply(&mut work, dt);
                }
                self.record(surrogate, &work, r, SampleTag::HotBurst, Some(burst_t));
```

(sim's `Scene::temperature`, spec §9.5; at the burst's start, `fresh`, it is the drawn velocities' T). `sample` failures push `Failed` with the stage's job. A reference multiplicity solve pushes `Pass(Reference { z, multiplicity: m })` with `Reference` or `Failed` (time it with `Instant::now()` around `scan.step`). `Job::Skip` and the burst leaving the cell push `TeachRule::Cell` with origin `Pass` and outcome `NotSolved`: one event per skipped geometry, and the burst's skipped tail as one event whose `detail` is "the hot burst reached the cell margin; its last k evaluations were skipped".

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
    /// `false`: the model was once loaded from a file without this record, so the counts start
    /// there and earlier solves are unrecorded (plan review M4: the flag must survive the
    /// session autosave, which round-trips every model through the file).
    pub complete: bool,
}
```

`Sidecar` gains `pub teacher_solves: Option<TeacherSolves>` and `pub teacher_solves_note: Option<String>` (`Some` when the key is present but unreadable; plan review S17). `write_with_flags(...)` becomes a thin call to

```rust
#[allow(clippy::too_many_arguments)]
pub fn write_full(path: &Path, model: &Model, acc: &Accumulator, report: Option<&FitReport>, samples: &[Sample], reference_rule: Option<&str>, quarantine: &[u8], face_density: &[f64], solves: Option<&TeacherSolves>) -> io::Result<()>
```

with `solves: None`. `encode` builds the header pairs as a `Vec<(&str, String)>` and appends `("teacher_solves", obj(&[("pass", t.pass.to_string()), ("fallback", …), ("audit", …), ("verification", …), ("kept", …), ("complete", t.complete.to_string())]))` **only when `Some`**, so files without it are byte-identical (the test pins it). `decode` reads the optional key leniently: a malformed record is a note, never a refused file (plan review S17; older readers ignore the key, since look-ups are by name, and nothing gates on it):

```rust
    let (teacher_solves, teacher_solves_note) = match h.get("teacher_solves") {
        Err(_) | Ok(Json::Null) => (None, None),
        Ok(t) => match decode_teacher_solves(t) {
            Ok(s) => (Some(s), None),
            Err(e) => (None, Some(format!("the model file's \"teacher_solves\" record is unreadable ({e}); its earlier solves count as unrecorded"))),
        },
    };
```

```rust
/// The optional "teacher_solves" record (activity spec A2); every field is required inside it.
fn decode_teacher_solves(t: &Json) -> Result<TeacherSolves, String> {
    Ok(TeacherSolves {
        pass: t.get("pass")?.u64()?,
        fallback: t.get("fallback")?.u64()?,
        audit: t.get("audit")?.u64()?,
        verification: t.get("verification")?.u64()?,
        kept: t.get("kept")?.u64()?,
        complete: t.get("complete")?.bool()?,
    })
}
```

Document the key in the module doc's format section ("Optional, every version (activity spec A2); unreadable → ignored with a note"). `store.rs`: `save_sidecar` passes

```rust
        let c = self.ledger.counts;
        let solves = TeacherSolves { pass: c.pass, fallback: c.fallback, audit: c.audit, verification: c.verification, kept: c.kept, complete: !c.unrecorded_before };
```

as `Some(&solves)`. `install_sidecar` reads `sc.teacher_solves` and `sc.teacher_solves_note` into locals first and sets the counts **only on its success path**, after the last fallible step (plan review N11: a refused file leaves the counts alone):

```rust
        self.ledger.counts = match solves {
            Some(t) => SolveCounts { pass: t.pass, fallback: t.fallback, audit: t.audit, verification: t.verification, kept: t.kept, unrecorded_before: !t.complete },
            None => SolveCounts { unrecorded_before: true, ..SolveCounts::default() },
        };
        if let Some(n) = note {
            self.pending_notes.push(format!("surrogate: {n}"));
        }
```

- [ ] **Step 6: Run the tests and the gate**

Run: `cargo test -p surrogate --release` → PASS. `cargo test -p providers --release` → PASS, every pre-existing test unchanged (the note strings are byte-identical: any test matching a note still passes). `cargo test -p validate --release --test surrogate` → PASS with the same printed numbers as before (paste both runs' key lines in the task report).

- [ ] **Step 7: Physics review**

Dispatch `physics-reviewer` on the diff with: "No decision changed? Is every `Refusal` text byte-identical? Is each rule code the rule that actually fired, given the priority envelope → novelty → class → variance → fence? Does `KeptBoxAffected` match the essential-box exemption exactly? Is `burst_t` the pre-CSVR temperature of the solved geometry? Does every auditor solve that is not cancelled leave exactly one event? Does the `complete` flag survive load → save → load?" Fix findings before committing.

- [ ] **Step 8: Commit**

```bash
git add crates/providers/src/surrogate crates/providers/src/sampling/mod.rs crates/surrogate/src/sidecar.rs
git commit -o -m "feat(providers): one activity event per teaching solve with its rule and outcome; lifetime solve counts in the sidecar

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -- crates/providers/src/surrogate crates/providers/src/sampling/mod.rs crates/surrogate/src/sidecar.rs
```

---

### Task 3: the trust checklist's data, the class verdict and the structured stuck reason **[Fable — physics-review gate]**

The plan review (M1, M2, M3, S10) changed this task. The checklist is no longer a second copy of the trust rule. There is one predicate per condition. `trust_checks` builds the list from those predicates, and `untrusted_check` is the list's first unmet entry, worded with today's strings. `residuals_fail` and `noise_fails` call the same predicates. The audit check is per atom, or names forces, and it never reads "passed" when no audit ran. A class's verdict is computed once, in structured form, and both `ClassLoo::trusted` and `class_untrusted`'s wording read it.

**Files:**
- Modify: `crates/providers/src/surrogate/trust.rs`, `crates/providers/src/surrogate/mod.rs` (field `audit_atoms`, set at the in-step failed audit; `status()`), `crates/providers/src/surrogate/audit.rs` (`audit_atoms` in `judge_audit`), `crates/sim/src/provider.rs` (`ClassKey`, `ClassLoo::refusal`), the two app fixtures that build a `ClassLoo` literal (`crates/app/src/ui/dock/model.rs` tests, `crates/app/src/ui/tests.rs` `model()`)
- Test: `crates/providers/src/surrogate/tests.rs`

**Interfaces:**
- Consumes: Task 2's `untrusted_check` and `Refusal` (in `streak_reason` and `class_untrusted`); Task 1b's `TrustKey::AuditForces` and `StuckInfo::rule: Option<TeachRule>`.
- Produces:
  - `SurrogateKs::trust_checks(&self) -> Vec<TrustCheck>`, with `untrusted_check(&self) -> Option<(TrustKey, String)>` (Task 2's signature) now derived from it, and `SurrogateKs::stuck_info(&self) -> Option<StuckInfo>`
  - `sim::ClassKey { Samples, Force, Energy }` and `sim::ClassLoo::refusal: Option<(ClassKey, f64, f64)>`: (key, value, limit) of the class's first failing condition, `None` when trusted
  - `pub(super) fn class_refusal(samples: usize, q_prime: usize, force_rms: f64, force_max: f64, energy_max: f64) -> Option<(ClassKey, f64, f64)>` (trust.rs)
  - the private readout field `SurrogateKs::audit_atoms: usize`; `TrainableStatus::{trust_checks, stuck_info}` filled

- [ ] **Step 1: Write the failing tests** (append to `crates/providers/src/surrogate/tests.rs`)

```rust
/// Review Focus 4: on every branch of the trust rule the first unmet check is the condition the
/// refusal names, and a trusted model meets every check.
#[test]
fn the_first_unmet_trust_check_is_the_refusal() {
    use sim::activity::TrustKey;
    let first_unmet = |s: &SurrogateKs| s.trust_checks().into_iter().find(|c| !c.met).map(|c| c.key);
    let key = |s: &SurrogateKs| s.untrusted_check().map(|c| c.0);
    // n < Q′
    let (s, _) = trained(1.2, 2.2, 5, FallbackPolicy::default());
    assert_eq!(first_unmet(&s), Some(TrustKey::Samples));
    assert_eq!(first_unmet(&s), key(&s));
    // residual dofs
    let (s, _) = trained(1.2, 2.2, 30, FallbackPolicy::default());
    assert_eq!(first_unmet(&s), key(&s));
    assert!(matches!(key(&s), Some(TrustKey::ResidualDofE | TrustKey::ResidualDofF)));
    // training residuals (the misspecified fit of `misspecified_fit_is_untrusted_by_its_training_residuals`)
    let (s, _) = trained_oscillatory(1.2, 2.2, 120, 0.02, 6.0);
    assert_eq!(first_unmet(&s), key(&s));
    assert!(matches!(key(&s), Some(TrustKey::EnergyResidual | TrustKey::ForceResidual)));
    // a failed audit, on the energy and on the forces
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    s.audit_failed = Some(0.5);
    s.audit_atoms = 2;
    assert_eq!(first_unmet(&s), Some(TrustKey::Audit));
    assert_eq!(first_unmet(&s), key(&s));
    s.audit_force = Some((5e-3, 2e-3));
    assert_eq!(first_unmet(&s), Some(TrustKey::AuditForces));
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

/// Plan review M2: one predicate per condition serves the checklist and the refusal, so they
/// agree at equality and on NaN by construction. (A NaN residual dof passes the rule today —
/// `NaN < 30` is false — so the checklist must show it met, never a ✗ on a trusted model.)
#[test]
fn the_trust_checks_and_the_refusal_agree_at_equality_and_nan() {
    use sim::activity::TrustKey;
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    assert!(s.trusted());
    let tier = s.policy.energy_std_per_atom;
    for (dof, energy) in [(MIN_RESIDUAL_DOF, tier), (f64::NAN, f64::NAN)] {
        s.report.as_mut().unwrap().residual_dof_e = dof;
        s.training_residuals.as_mut().unwrap().energy_per_atom = energy;
        let checks = s.trust_checks();
        let met = |k: TrustKey| checks.iter().find(|c| c.key == k).unwrap().met;
        assert!(met(TrustKey::ResidualDofE) && met(TrustKey::EnergyResidual), "{dof} {energy}: {checks:?}");
        assert_eq!(s.untrusted_check(), None);
        assert!(s.trusted());
    }
    // just past a limit: unmet, and the refusal names the same key
    s.report.as_mut().unwrap().residual_dof_e = MIN_RESIDUAL_DOF - 1e-9;
    assert_eq!(s.trust_checks().into_iter().find(|c| !c.met).map(|c| c.key), Some(TrustKey::ResidualDofE));
    assert_eq!(s.untrusted_check().map(|c| c.0), Some(TrustKey::ResidualDofE));
}

/// The refusal strings stay byte-identical (the Log, the app's `COUNT_REFUSALS` until Task 8,
/// and every test that matches a note read them).
#[test]
fn the_refusal_strings_are_unchanged() {
    let (s, _) = trained(1.2, 2.2, 5, FallbackPolicy::default());
    let r = s.report().unwrap().clone();
    assert_eq!(s.untrusted_reason().unwrap(), format!("n_samples {} < q_prime {}", r.n_samples, r.q_prime));
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    s.audit_failed = Some(0.5);
    s.audit_atoms = 2;
    assert_eq!(s.untrusted_reason().unwrap(), "audit found teacher − model +5.0e-1 Ha; refit to recover");
    s.audit_force = Some((5e-3, 2e-3));
    assert_eq!(s.untrusted_reason().unwrap(), "audit found forces off by 5.0e-3 > 2.0e-3 Ha/Bohr (teacher − model +5.0e-1 Ha); refit to recover");
}

/// Plan review M1: the audit check compares the failed audit's |gap|/N with the per-atom tier,
/// or, when the forces failed it, the worst atom's residual with its bound. With no failed
/// audit since the refit it is met with value 0, whether or not an audit ever ran: the UI words
/// that case from `last_audit` (Task 8), never as "passed".
#[test]
fn the_audit_check_is_per_atom_and_names_forces() {
    use sim::activity::TrustKey;
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    let audit = |s: &SurrogateKs| *s.trust_checks().last().unwrap();
    assert!(s.last_audit().is_none());
    let c = audit(&s);
    assert_eq!((c.key, c.met, c.value, c.limit), (TrustKey::Audit, true, 0.0, s.policy.energy_std_per_atom));
    s.audit_failed = Some(-3e-3);
    s.audit_atoms = 2;
    let c = audit(&s);
    assert_eq!((c.key, c.met), (TrustKey::Audit, false));
    assert!((c.value - 1.5e-3).abs() < 1e-18 && c.limit == s.policy.energy_std_per_atom, "per atom: {c:?}");
    s.audit_force = Some((4e-3, 2e-3));
    let c = audit(&s);
    assert_eq!((c.key, c.met, c.value, c.limit), (TrustKey::AuditForces, false, 4e-3, 2e-3));
    s.refit().unwrap();
    let c = audit(&s);
    assert_eq!((c.key, c.met, c.value), (TrustKey::Audit, true, 0.0), "a refit clears the failure");
}

/// Plan review M3: one function decides a class's trust and names its failing condition; the
/// wording reads that, so an energy-only failure is never worded as a force error.
#[test]
fn the_class_verdict_names_its_failing_condition() {
    use sim::ClassKey;
    assert_eq!(class_refusal(10, 16, 0.1, 0.2, 0.3), Some((ClassKey::Samples, 10.0, 16.0)));
    assert_eq!(class_refusal(40, 16, 0.9, 2.5, 0.3), Some((ClassKey::Force, 2.5, 1.0)));
    assert_eq!(class_refusal(40, 16, 0.2, 0.4, 1100.0), Some((ClassKey::Energy, 1100.0, 1.0)));
    assert_eq!(class_refusal(40, 16, 0.2, 0.4, 0.3), None);
    assert!(class_refusal(40, 16, f64::NAN, 0.4, 0.3).is_some(), "NaN fails, as `trusted`'s `<=` did");
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    assert!(s.class_loo().iter().all(|l| l.trusted == l.refusal.is_none()));
    s.class_loo[0] = sim::ClassLoo { class: "pair", samples: 120, force_rms: 0.1, force_max: 0.4, energy_max: 1100.0, trusted: false, refusal: Some((ClassKey::Energy, 1100.0, 1.0)) };
    let (r, short) = s.class_untrusted(&[1, 1]).unwrap();
    assert!(!short && r.text.contains("LOO energy error 1100.0× the tier"), "{}", r.text);
    assert_eq!(r.rule, sim::activity::TeachRule::Class(sim::activity::EnvClass::Pair));
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
    assert_eq!(i.rule, Some(sim::activity::TeachRule::ForceUncertainty));
    assert!(i.atom.is_some() && i.steps >= STUCK_STEPS && !i.box_streak);
}
```

(`trained_oscillatory`'s signature is in tests.rs line 235; pick arguments that the existing misspecification test shows untrusted by residuals.) In the existing `a_failed_background_audit_untrusts_and_books_the_flip`, after `assert!(!sk.trusted() && sk.audit_failed().is_some());`, pin the real path (the 2-atom audit):

```rust
    let c = sk.trust_checks().into_iter().find(|c| !c.met).expect("the failed audit is unmet");
    match c.key {
        sim::activity::TrustKey::Audit => assert!((c.value - sk.audit_failed().unwrap().abs() / 2.0).abs() < 1e-15, "per atom: {c:?}"),
        sim::activity::TrustKey::AuditForces => assert!(c.value > c.limit),
        k => panic!("the audit is the refusal, not {k:?}"),
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p providers --release -- trust_check refusal audit_check class_verdict stuck_info`
Expected: compile errors (`trust_checks`, `audit_atoms`, `class_refusal`, `ClassKey`, `ClassLoo::refusal`, `stuck_info` missing).

- [ ] **Step 3: Implement**

`crates/sim/src/provider.rs`, beside `ClassLoo`:

```rust
/// The condition that keeps an environment class untrusted ([`ClassLoo::refusal`]), in the order
/// the class rule tests them (`providers::surrogate`, "Class-wise trust").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClassKey {
    /// Fewer samples of the class than Q′: (samples, Q′).
    Samples,
    /// The leave-one-out force error over the class's atoms: (the larger of max and RMS, as a
    /// ratio to each atom's bound; limit 1).
    Force,
    /// The largest leave-one-out per-atom energy error, as a ratio to the tier (limit 1).
    Energy,
}
```

and the field, after `trusted`:

```rust
    /// The class rule's verdict in structured form: (the first failing condition, its value, its
    /// limit), `None` when trusted. `trusted == refusal.is_none()` always (one function decides
    /// both; plan review M3).
    pub refusal: Option<(ClassKey, f64, f64)>,
```

Re-export `ClassKey` from `sim`'s root next to `ClassLoo`. In the app's two `ClassLoo` literals add the field: the trusted pair in `ui/tests.rs` `model()` gets `refusal: None`; the many-body one in `ui/dock/model.rs`'s test gets `refusal: Some((sim::ClassKey::Force, 2.69, 1.0))`.

`trust.rs`: one predicate per condition, beside `MIN_RESIDUAL_DOF`:

```rust
// The trust rule's conditions, one predicate each (plan review M2). `trust_checks` builds the
// checklist from them as `met = !fails`; `untrusted_check` is that list's first unmet entry;
// `residuals_fail`, `noise_fails` and `trust_needs_samples` call them. No condition is written
// twice, so the checklist and the refusal agree at equality and on NaN by construction.
fn samples_short(r: &FitReport) -> bool {
    r.n_samples < r.q_prime
}
fn dof_short(dof: f64) -> bool {
    dof < MIN_RESIDUAL_DOF
}
fn energy_residual_fails(t: &TrainingResiduals, p: &FallbackPolicy) -> bool {
    t.energy_per_atom > p.energy_std_per_atom
}
fn force_residual_fails(t: &TrainingResiduals) -> bool {
    t.force > t.force_bound
}
fn force_noise_fails(r: &FitReport, p: &FallbackPolicy) -> bool {
    r.sigma_f > p.force_std_abs
}

/// The class rule for one class (module doc, "Class-wise trust"): the first failing condition
/// (too few samples, the LOO force error, the LOO energy error) as (key, value, limit), `None`
/// when the class is trusted. `ClassLoo::trusted` and `class_untrusted`'s wording both read it
/// (plan review M3). The force and energy tests are written as `!(x <= 1)`, so a NaN ratio
/// fails, exactly as the old `trusted`'s `<=` made it fail.
pub(super) fn class_refusal(samples: usize, q_prime: usize, force_rms: f64, force_max: f64, energy_max: f64) -> Option<(ClassKey, f64, f64)> {
    if samples < q_prime {
        return Some((ClassKey::Samples, samples as f64, q_prime as f64));
    }
    if !(force_max <= 1.0 && force_rms <= 1.0) {
        return Some((ClassKey::Force, force_max.max(force_rms), 1.0));
    }
    if !(energy_max <= 1.0) {
        return Some((ClassKey::Energy, energy_max, 1.0));
    }
    None
}
```

In `impl SurrogateKs` (replacing the body of `untrusted_check` that Task 2 wrote; `untrusted_reason` stays `untrusted_check().map(|c| c.1)`):

```rust
    /// The trust rule's conditions with their numbers, in the order the rule tests them
    /// (activity spec A4); empty before the first fit. Each entry is `met = !(its predicate)`;
    /// the first unmet one is the refusal (`untrusted_check`), by construction. The audit entry
    /// is per atom: |gap|/N against `energy_std_per_atom`, or, when the forces failed the audit,
    /// the worst atom's residual against its bound (`AuditForces`); with no failed audit since
    /// the refit it is met with value 0 (plan review M1).
    pub fn trust_checks(&self) -> Vec<TrustCheck> {
        let Some(r) = self.report.as_ref() else { return Vec::new() };
        let p = &self.policy;
        let check = |key, fails: bool, value: f64, limit: f64| TrustCheck { key, met: !fails, value, limit };
        let mut v = vec![
            check(TrustKey::Samples, samples_short(r), r.n_samples as f64, r.q_prime as f64),
            check(TrustKey::ResidualDofE, dof_short(r.residual_dof_e), r.residual_dof_e, MIN_RESIDUAL_DOF),
            check(TrustKey::ResidualDofF, dof_short(r.residual_dof_f), r.residual_dof_f, MIN_RESIDUAL_DOF),
        ];
        if let Some(t) = &self.training_residuals {
            v.push(check(TrustKey::EnergyResidual, energy_residual_fails(t, p), t.energy_per_atom, p.energy_std_per_atom));
            v.push(check(TrustKey::ForceResidual, force_residual_fails(t), t.force, t.force_bound));
        }
        v.push(check(TrustKey::ForceNoise, force_noise_fails(r, p), r.sigma_f, p.force_std_abs));
        v.push(match (self.audit_failed, self.audit_force) {
            (Some(_), Some((d, b))) => check(TrustKey::AuditForces, true, d, b),
            (Some(gap), None) => check(TrustKey::Audit, true, gap.abs() / self.audit_atoms.max(1) as f64, p.energy_std_per_atom),
            (None, _) => check(TrustKey::Audit, false, 0.0, p.energy_std_per_atom),
        });
        v
    }

    /// The trust rule's first unmet condition and today's refusal text for it (byte-identical
    /// to the strings before the Activity tab; pinned by `the_refusal_strings_are_unchanged`).
    pub fn untrusted_check(&self) -> Option<(TrustKey, String)> {
        let r = self.report.as_ref()?;
        let c = self.trust_checks().into_iter().find(|c| !c.met)?;
        let gap = self.audit_failed.unwrap_or(0.0);
        // the count conditions: the app's `COUNT_REFUSALS` matched their wording ("n_samples ",
        // "residual_dof_") until Task 8 keyed on `TrustKey` — keep the prefixes
        let text = match c.key {
            TrustKey::Samples => format!("n_samples {} < q_prime {}", r.n_samples, r.q_prime),
            TrustKey::ResidualDofE => format!("residual_dof_e {:.1} < {MIN_RESIDUAL_DOF:.0}", r.residual_dof_e),
            TrustKey::ResidualDofF => format!("residual_dof_f {:.1} < {MIN_RESIDUAL_DOF:.0}", r.residual_dof_f),
            TrustKey::EnergyResidual => format!("max training energy residual {:.1e} > {:.1e} Ha/atom", c.value, c.limit),
            TrustKey::ForceResidual => format!("max training force residual {:.1e} > {:.1e} Ha/Bohr", c.value, c.limit),
            TrustKey::ForceNoise => format!(
                "force noise σ_f {:.1e} > {:.1e} Ha/Bohr — the fit's own force labels scatter beyond the force tier",
                c.value, c.limit
            ),
            TrustKey::AuditForces => format!("audit found forces off by {:.1e} > {:.1e} Ha/Bohr (teacher − model {gap:+.1e} Ha); refit to recover", c.value, c.limit),
            TrustKey::Audit => format!("audit found teacher − model {gap:+.1e} Ha; refit to recover"),
        };
        Some((c.key, text))
    }
```

and route the rest through the predicates:

```rust
    pub fn trusted(&self) -> bool {
        // the same verdict as `untrusted_reason().is_none()`, without formatting a string (an
        // empty list, no report yet, is all met, as `untrusted_reason`'s `None` was)
        self.model.is_some() && self.trust_checks().iter().all(|c| c.met)
    }

    pub(super) fn residuals_fail(&self) -> bool {
        self.training_residuals.as_ref().is_some_and(|t| energy_residual_fails(t, &self.policy) || force_residual_fails(t))
    }

    pub(super) fn noise_fails(&self) -> bool {
        self.report.as_ref().is_some_and(|r| force_noise_fails(r, &self.policy))
    }

    pub(super) fn trust_needs_samples(&self) -> bool {
        self.report.as_ref().is_none_or(|r| samples_short(r) || dof_short(r.residual_dof_e) || dof_short(r.residual_dof_f))
    }
```

`compute_class_loo` ends with

```rust
            .map(|(&(samples, ss, atoms, force_max, energy_max), class)| {
                let force_rms = if atoms > 0 { (ss / atoms as f64).sqrt() } else { 0.0 };
                let force_rms = if force_max.is_finite() { force_rms } else { f64::INFINITY };
                let refusal = class_refusal(samples, report.q_prime, force_rms, force_max, energy_max);
                ClassLoo { class, samples, force_rms, force_max, energy_max, trusted: refusal.is_none(), refusal }
            })
```

and `class_untrusted` (as Task 2 left it, returning `(Refusal, bool)`) picks its branch from `l.refusal` instead of re-testing the numbers. Each `format!` and its `Refusal::new(…, TeachRule::Class(…), None)` wrapper stay as Task 2 wrote them:

```rust
        Some(match l.refusal {
            Some((ClassKey::Samples, ..)) => (/* the "{} samples < Q′ {q}" refusal */, true),
            Some((ClassKey::Force, ..)) => (/* the "LOO force error {:.1}× the bound, RMS …" refusal */, false),
            Some((ClassKey::Energy, ..)) | None => (/* the "LOO energy error {:.1}× the tier" refusal */, false),
        })
```

(`None` cannot occur: `l.trusted` returned earlier.) The texts are identical for every finite ratio. A NaN force ratio, which the LOO cannot produce (an undefined LOO is ∞), would now be worded as a force error; before, it was worded as an energy error. The decision is the same.

`stuck_info` (plan review S10: the rule is `None` when no reason was stored, never a guessed one):

```rust
    /// [`stuck`](Self::stuck) in structured form (activity spec A5).
    pub fn stuck_info(&self) -> Option<StuckInfo> {
        (self.streak >= STUCK_STEPS && self.trusted()).then(|| {
            let r = self.streak_reason.as_ref();
            StuckInfo {
                rule: r.map(|r| r.rule),
                atom: r.and_then(|r| r.atom),
                steps: self.streak,
                box_streak: self.streak_box >= STUCK_STEPS,
                gap: self.streak_gap,
                face_density: self.streak_box_face,
            }
        })
    }
```

The audit's atom count (plan review M1): `mod.rs` gains the field

```rust
    /// Atoms of the geometry whose failed audit set `audit_failed`: the checklist's per-atom
    /// value |gap|/N (activity plan review M1). A readout only; it decides nothing.
    audit_atoms: usize,
```

(`0` in `new`). Set it next to every `self.audit_failed = Some(gap);`: `self.audit_atoms = n as usize;` in `teach`'s failed in-step audit (`mod.rs`) and in `judge_audit`'s `Failed` arm (`audit.rs`); both have `n = pos.len().max(1) as f64` in scope. `status()` sets `trust_checks: self.trust_checks(), stuck_info: self.stuck_info()`.

- [ ] **Step 4: Run** `cargo test -p providers --release` → PASS (every pre-existing test unchanged; the refusal strings are pinned). `cargo test -p app --release ui::` → PASS (the `ClassLoo` fixtures). `cargo test -p validate --release --test surrogate` → PASS with the numbers of Task 2's report.

- [ ] **Step 5: Physics review.** `physics-reviewer`: "Is every check `!(its refusal predicate)`, with the predicate shared (M2)? Is the audit check per atom against the per-atom tier, or the worst force residual against its bound when `audit_force` is set, and met at value 0 only when no audit failed since the refit (M1)? Does `class_refusal` reproduce the old `trusted` exactly, NaN included, and is the energy-only class worded as energy (M3)? Are the refusal strings byte-identical?" Apply the ruling.

- [ ] **Step 6: Commit**

```bash
git add crates/providers/src/surrogate/trust.rs crates/providers/src/surrogate/mod.rs crates/providers/src/surrogate/audit.rs crates/providers/src/surrogate/tests.rs crates/sim/src/provider.rs crates/sim/src/lib.rs crates/app/src/ui/dock/model.rs crates/app/src/ui/tests.rs
git commit -o -m "feat(providers): one trust predicate per condition, per-atom audit check, structured class verdict, stuck info

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -- crates/providers/src/surrogate/trust.rs crates/providers/src/surrogate/mod.rs crates/providers/src/surrogate/audit.rs crates/providers/src/surrogate/tests.rs crates/sim/src/provider.rs crates/sim/src/lib.rs crates/app/src/ui/dock/model.rs crates/app/src/ui/tests.rs
```

---

### Task 4: `would_answer` and the training views **[Fable — physics-review gate]**

The plan review changed this task in six places:
- `would_answer` is `&self`, so the compiler enforces "nothing moves" (S1). It is pinned by behaviour too (S1).
- It evaluates the refusals in priority order with early returns, and the curves pass in the band's prediction (S2).
- The views use the fit's own reference energies (S4).
- The force parity is thinned by whole atoms (S5).
- A failed prediction skips one curve or one sample and is counted (S18).
- The curve carries rule 6's ratio for the uncertainty lane (M5).

**Files:**
- Create: `crates/providers/src/surrogate/views.rs`
- Modify: `crates/providers/src/surrogate/mod.rs` (`mod views;`, `Trainable::training_views`), `crates/providers/src/surrogate/quarantine.rs` (the read-only fence)
- Test: `crates/providers/src/surrogate/tests.rs`

**Interfaces:**
- Consumes: Task 2's `Refusal` codes; Task 1's view records with Task 1b's `PairCurve::force_ratio` and `TrainingViews::{parity_omitted, curves_skipped}`.
- Produces:
  - `pub type Verdict = Result<(), (TeachRule, Option<usize>)>;` (views.rs)
  - `SurrogateKs::would_answer(&self, pos: &[[f64; 3]], z: &[u8], prov: &Provenance) -> Verdict` (the fence reads the descriptor cache: the caller fills it with `ensure_desc_cache` first)
  - `SurrogateKs::training_views(&mut self) -> Option<TrainingViews>` (fills the cache once, then reads only); `Trainable::training_views` implemented
  - `pub(super) fn thin_atoms(atoms: &[[[f32; 2]; 3]], cap: usize) -> Vec<[f32; 2]>`
  - quarantine.rs: `quarantine_distances_cached(&self, …)`, `near_quarantined_cached(&self, …)`

- [ ] **Step 1: Write the failing tests**

```rust
/// Spec A6: `would_answer` is `evaluate`'s verdict without its side effects. With the variance
/// evaluated on every consultation and audits off, `evaluate` answers from the model exactly
/// where `would_answer` says Ok, with the same rule and atom otherwise, over dimers and trimers.
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
    s.ensure_desc_cache(); // the fence reads the cache: `would_answer` is `&self`
    let verdicts: Vec<_> = scenes
        .iter()
        .map(|sc| {
            let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
            let z: Vec<u8> = sc.nuclei.iter().map(|n| n.z).collect();
            s.would_answer(&pos, &z, &s.provenance_for(sc))
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
            Err((rule, atom)) => assert_eq!(ev.first().map(|e| (e.rule, e.atom)), Some((*rule, *atom)), "{:?}", sc.nuclei),
        }
        agree += 1;
    }
    eprintln!("would_answer: {agree} geometries compared, {answered} answered by the model");
    assert!(answered > 20 && agree - answered > 20, "the sweep must cover both verdicts");
}

/// Plan review S1: the fence, the trust rule and draft mode. `would_answer` names the fence and
/// rule 4 as `evaluate` does, and ignores draft mode: it is the policy's verdict (the pair-curve
/// caption says so in draft mode, Task 9).
#[test]
fn would_answer_covers_the_fence_untrusted_and_draft() {
    use sim::activity::TeachRule as R;
    let verdict = |s: &mut SurrogateKs, sc: &Scene| {
        let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
        let z: Vec<u8> = sc.nuclei.iter().map(|n| n.z).collect();
        s.ensure_desc_cache();
        s.would_answer(&pos, &z, &s.provenance_for(sc))
    };
    let policy = FallbackPolicy { variance_cadence: 1, audit_every: 0, ..FallbackPolicy::default() };
    // the fence: a stored sample held in quarantine fences its own geometry
    let (mut s, _) = trained(1.2, 2.2, 120, policy.clone());
    let k = 60;
    let p = s.store().unwrap().samples[k].pos.clone();
    let r_k = ((p[0][0] - p[1][0]).powi(2) + (p[0][1] - p[1][1]).powi(2) + (p[0][2] - p[1][2]).powi(2)).sqrt();
    s.quarantine.resize(s.samples(), Q_NONE);
    s.quarantine[k] = Q_HELD;
    let sc = h2(r_k);
    assert_eq!(verdict(&mut s, &sc), Err((R::NearQuarantine, None)));
    s.evaluate(&sc, &mut never()).unwrap();
    assert_eq!(Trainable::take_teach_events(&mut s).first().map(|e| e.rule), Some(R::NearQuarantine));
    // untrusted (rule 4 on the count)
    let (mut s, _) = trained(1.2, 2.2, 5, policy.clone());
    let v = verdict(&mut s, &h2(1.6));
    assert!(matches!(v, Err((R::Untrusted(_), None))), "{v:?}");
    s.evaluate(&h2(1.6), &mut never()).unwrap();
    assert_eq!(Trainable::take_teach_events(&mut s).first().map(|e| e.rule), v.err().map(|e| e.0));
    // draft: evaluate answers from the model where the policy refuses; would_answer still refuses
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy { never_fallback: true, ..policy });
    let far = in_big_cell(h2(4.0));
    assert!(verdict(&mut s, &far).is_err(), "the policy's verdict, draft or not");
    assert_eq!(s.evaluate(&far, &mut never()).unwrap().stats.source, Source::Surrogate);
}

/// Plan review S1: building the views between steps changes nothing the policy reads. Two
/// identical providers, one building `training_views` before every evaluate, give bit-identical
/// sources, energies, forces, variance-check counts, stores and event streams over 200 steps.
#[test]
fn training_views_change_nothing_the_policy_reads() {
    let policy = FallbackPolicy { audit_every: 7, ..FallbackPolicy::default() };
    let (mut a, _) = trained(1.2, 2.2, 120, policy.clone());
    let (mut b, _) = trained(1.2, 2.2, 120, policy);
    let mut rng = StdRng::seed_from_u64(0xA11);
    let key = |e: &sim::TeachEvent| (e.origin, e.rule, e.atom, e.outcome, e.gap.map(f64::to_bits), e.detail.clone());
    for _ in 0..200 {
        let sc = h2(rng.gen_range(1.0..3.0));
        let _ = b.training_views();
        match (a.evaluate(&sc, &mut never()), b.evaluate(&sc, &mut never())) {
            (Ok(ra), Ok(rb)) => {
                assert_eq!(ra.stats.source, rb.stats.source);
                assert_eq!(ra.energy.to_bits(), rb.energy.to_bits());
                assert_eq!(ra.forces, rb.forces);
            }
            (Err(ea), Err(eb)) => assert_eq!(ea.to_string(), eb.to_string()),
            (ra, rb) => panic!("diverged: {:?} vs {:?}", ra.map(|r| r.stats.source), rb.map(|r| r.stats.source)),
        }
        let (ea, eb) = (Trainable::take_teach_events(&mut a), Trainable::take_teach_events(&mut b));
        assert_eq!(ea.iter().map(key).collect::<Vec<_>>(), eb.iter().map(key).collect::<Vec<_>>());
        assert_eq!((a.variance_checks(), a.samples()), (b.variance_checks(), b.samples()));
    }
}

/// Spec §9: parity points are (E − Σ E_ref)/N of the teacher and of `predict` at each stored
/// sample, with the fit's own references; the band is the epistemic std, the same number rule 7
/// compares; the force ratio is rule 6's; no teacher call.
#[test]
fn training_views_read_the_store_and_the_posterior_only() {
    let (mut s, h) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    let c0 = calls(&h);
    let v = s.training_views().expect("a fitted model");
    assert_eq!(calls(&h), c0);
    assert_eq!(v.parity_energy.len(), s.samples());
    assert_eq!((v.parity_omitted, v.curves_skipped), (0, 0));
    let st = s.store().unwrap();
    let m = s.model().unwrap();
    // the fit's own reference energy (plan review S4); here the Morse atom's
    let e_ref = m.ref_energy[m.basis.species_index(1).unwrap()];
    assert!((e_ref - MorseDimer::h2().e_atom).abs() < 1e-12);
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
    assert_eq!((c.r.len(), c.force_ratio.len()), (sim::activity::CURVE_POINTS, sim::activity::CURVE_POINTS));
    let k = c.r.iter().position(|&r| r >= 1.6).unwrap();
    let pos = [[-c.r[k] / 2.0, 0.0, 0.0], [c.r[k] / 2.0, 0.0, 0.0]];
    let pred = m.predict(&pos, &[1, 1], true).unwrap();
    let floor = 2.0 * m.sigma_e; // energy_per_atom models: N s_E
    let epi = (pred.energy_std.unwrap().powi(2) - floor * floor).max(0.0).sqrt();
    assert!((c.sigma[k] - epi).abs() < 1e-15);
    let ratio = s.variance_check(m, &pred, pred.energy_std.unwrap(), pred.force_std_epistemic.as_ref().unwrap()).atoms.iter().copied().fold(0.0, f64::max);
    assert!((c.force_ratio[k] - ratio).abs() < 1e-15, "rule 6's ratio");
    assert!((c.energy[k] - (pred.energy - 2.0 * e_ref)).abs() < 1e-12);
    assert_eq!(c.refused[k], None, "inside the window the model answers");
    assert!(c.refused.last().unwrap().is_some(), "near r_cut the dimer is refused (5a or the envelope)");
    assert_eq!(c.dots.len(), 120, "every stored sample is a two-atom H–H sample");
    assert_eq!(v.pairs[0].counts.iter().sum::<u32>(), 120);
    assert!((v.dimer_floor - floor).abs() < 1e-18 && v.energy_tier == 1e-3);
}

/// Plan review S5: the force parity is thinned by whole atoms, so x, y and z stay equally
/// represented (a component stride that is a multiple of 3 kept a single component), and the
/// cap holds.
#[test]
fn force_parity_thins_whole_atoms() {
    use sim::activity::FORCE_PARITY_CAP;
    let atoms: Vec<[[f32; 2]; 3]> = (0..7000).map(|i| [[i as f32, 0.0], [i as f32, 1.0], [i as f32, 2.0]]).collect();
    let out = thin_atoms(&atoms, FORCE_PARITY_CAP);
    assert!(out.len() <= FORCE_PARITY_CAP && out.len() % 3 == 0);
    for c in 0..3 {
        assert_eq!(out.iter().filter(|p| p[1] == c as f32).count(), out.len() / 3, "component {c}");
    }
    for n in [1, 6666, 6667, 13_334, 40_000] {
        let a: Vec<[[f32; 2]; 3]> = atoms.iter().cycle().take(n).copied().collect();
        let out = thin_atoms(&a, FORCE_PARITY_CAP);
        assert!(out.len() <= FORCE_PARITY_CAP, "{n} atoms → {} components", out.len());
    }
    assert_eq!(thin_atoms(&atoms[..10], FORCE_PARITY_CAP).len(), 30, "under the cap nothing is dropped");
}

/// Review Focus 5: the build time at S = 4 is reported (never asserted), with the share of the
/// worker's time it takes at one build per 2 s (plan review S2: the build runs between steps).
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn training_views_build_time_is_reported() {
    // a 4-species store: H, He, Li, Be dimers on the shared Morse form (the species only label
    // channels here; the numbers are a cost probe, not physics)
    let (mut s, _) = four_species_trained(); // see Step 3: builds 400 dimers over the 10 pairs and refits
    let t = std::time::Instant::now();
    let v = s.training_views().unwrap();
    let ms = t.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "training views at S = 4 (Q′ {}): {ms:.1} ms (build_ms {:.1}), {} curves, {} parity points; at one build per 2 s the worker spends {:.1} % of its time on views",
        s.report().unwrap().q_prime, v.build_ms, v.curves.len(), v.parity_energy.len(), 100.0 * ms / 2000.0
    );
}
```

`four_species_trained()`: a helper next to `trained` that sets references for Z 1–4 to `MorseDimer::h2().e_atom` and adds 40 `morse_sample`s for each of the 10 pairs (`dimer(r, …, za, zb)`), then refits. If the existing new-species test has a cheaper multi-species builder, reuse it.

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p providers --release -- would_answer training_views force_parity`
Expected: compile errors.

- [ ] **Step 3: Implement**

`quarantine.rs`: split the fence's distance into a filling wrapper and a read-only body (behaviour unchanged for every existing caller):

```rust
    /// (index, distance) of every held quarantined sample from the geometry (`near_quarantined`).
    pub(super) fn quarantine_distances(&mut self, pos: &[[f64; 3]], z: &[u8]) -> Vec<(usize, f64)> {
        if self.quarantined() == 0 {
            return Vec::new();
        }
        self.ensure_desc_cache();
        self.quarantine_distances_cached(pos, z)
    }

    /// [`quarantine_distances`](Self::quarantine_distances) on a descriptor cache the caller
    /// filled (`ensure_desc_cache`): read-only, for `would_answer` (activity plan review S1).
    pub(super) fn quarantine_distances_cached(&self, pos: &[[f64; 3]], z: &[u8]) -> Vec<(usize, f64)> {
        if self.quarantined() == 0 {
            return Vec::new();
        }
        debug_assert_eq!(self.desc_cache.len(), self.store.as_ref().map_or(0, |s| s.len()), "fill the descriptor cache first");
        // … the body after `self.ensure_desc_cache();`, unchanged …
    }

    /// [`near_quarantined`](Self::near_quarantined) on a filled descriptor cache (read-only).
    pub(super) fn near_quarantined_cached(&self, pos: &[[f64; 3]], z: &[u8]) -> Vec<usize> {
        let limit = QUARANTINE_FRACTION * self.policy.max_descriptor_distance;
        self.quarantine_distances_cached(pos, z).into_iter().filter(|&(_, d)| d <= limit).map(|(i, _)| i).collect()
    }
```

Make `ensure_desc_cache` `pub(super)` if it is not already (tests call it).

`views.rs`:

```rust
//! The training views (activity spec §3.3.5, §9, A6): what the Activity tab plots, built from
//! the store and the current posterior, never a teacher call, and `would_answer`, the fallback
//! policy's verdict at a geometry, read-only by type.
use super::*;
use sim::activity::{
    PairCurve, PairHistogram, ParityPoint, SampleKind, SampleState, TeachRule, TrainingViews, CURVE_POINTS, CURVE_R_MIN, FORCE_PARITY_CAP,
};

/// The fallback policy's verdict at one geometry: `Ok` where the model would answer, else the
/// rule that refuses and the atom it names.
pub type Verdict = Result<(), (TeachRule, Option<usize>)>;

impl SurrogateKs {
    /// Rules 1–7 and the fence at (`pos`, `z`) under `prov`, in `evaluate`'s priority order
    /// (gate → envelope → novelty → class → variance → fence) with a fresh variance. The first
    /// refusal returns, so the costly prediction is skipped wherever the envelope, the novelty or
    /// the class already refuses: the same verdict as `evaluate`'s "compute all, take the first"
    /// (plan review S2). Read-only by type (S1): no cadence, audit counter, streak, store,
    /// quarantine or cache moves. The fence reads the descriptor cache: call `ensure_desc_cache`
    /// first (`training_views` does). Not part of this verdict: rule 0 (the cell), draft mode,
    /// audits, the relax verification and the cadence's reuse of an accepting variance.
    pub fn would_answer(&self, pos: &[[f64; 3]], z: &[u8], prov: &Provenance) -> Verdict {
        self.gate(z, prov).map_err(|g| (g.reason.rule, g.reason.atom))?;
        self.verdict_past_gate(pos, z, None)
    }

    /// [`would_answer`](Self::would_answer) after the gate passed. `pred`: the caller's
    /// prediction with variance at (`pos`, `z`), when it already has one (the curve's band).
    fn verdict_past_gate(&self, pos: &[[f64; 3]], z: &[u8], pred: Option<&surrogate::Prediction>) -> Verdict {
        let model = self.model.as_ref().expect("the gate passed: a model exists");
        let code = |r: &Refusal| (r.rule, r.atom);
        if let (_, Some(r)) = self.envelope_outside(model, pos, z) {
            return Err(code(&r));
        }
        // 5d is evaluated beside the variance, so only with a posterior (as in `evaluate`)
        if model.sigma_inv_chol.is_some() {
            if let Some((_, at)) = self.null_space.as_ref().and_then(|ns| ns.novelty(pos, z).ok()).filter(|(nu, _)| *nu > 1.0) {
                return Err((TeachRule::NewEnvironment, at));
            }
        }
        let rc2 = model.basis.r_cut * model.basis.r_cut;
        if let Some((r, _)) = self.class_untrusted(&neighbour_counts(pos, rc2)) {
            return Err(code(&r));
        }
        let owned;
        let pred = match pred {
            Some(p) => p,
            None => {
                // a geometry the model cannot describe: `evaluate` returns an error there (no
                // rule, no teacher); never reached on the curve grid (r ≥ CURVE_R_MIN)
                owned = model.predict(pos, z, true).map_err(|_| (TeachRule::NewEnvironment, None))?;
                &owned
            }
        };
        if let (Some(e), Some(f)) = (pred.energy_std, pred.force_std_epistemic.as_ref()) {
            let v = self.variance_check(model, pred, e, f);
            if !v.accept {
                return Err(code(v.reason.as_ref().expect("a refusing verdict names its rule")));
            }
        }
        if !self.near_quarantined_cached(pos, z).is_empty() {
            return Err((TeachRule::NearQuarantine, None));
        }
        Ok(())
    }

    /// The training views (spec A6, §9); `None` without a fitted model. Fills the fence's
    /// descriptor cache once, then reads only.
    pub fn training_views(&mut self) -> Option<TrainingViews> {
        self.ensure_desc_cache();
        self.views_from_cache()
    }

    fn views_from_cache(&self) -> Option<TrainingViews> {
        let t0 = Instant::now();
        let (model, store) = (self.model.as_ref()?, self.store.as_ref()?);
        // the fit's own reference energies (plan review S4): what the fit subtracted, whatever
        // `self.refs` holds now (a pass may have re-solved one: `refs_dirty`)
        let e_ref = |zs: &[u8]| -> Option<f64> {
            let idx = model.basis.species_indices(zs).ok()?;
            Some(idx.iter().map(|&i| model.ref_energy[i]).sum())
        };
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
        let mut parity_omitted = 0usize;
        let mut force_atoms: Vec<[[f32; 2]; 3]> = Vec::new();
        let mut dots: BTreeMap<(u8, u8), Vec<(f64, f64, SampleState)>> = BTreeMap::new();
        let mut hist: BTreeMap<(u8, u8), Vec<u32>> = BTreeMap::new();
        const BIN: f64 = 0.1;
        for (i, smp) in store.samples.iter().enumerate() {
            // one failed prediction omits that sample, counted, never the views (plan review S18)
            let (Some(eref), Ok(pred)) = (e_ref(&smp.z), model.predict(&smp.pos, &smp.z, false)) else {
                parity_omitted += 1;
                continue;
            };
            let n = smp.pos.len() as f64;
            let st = state(i);
            parity_energy.push(ParityPoint { dft: (smp.energy - eref) / n, model: (pred.energy - eref) / n, kind: kind(smp.tag), state: st, index: i });
            if st == SampleState::Fitted {
                for (ft, fm) in smp.forces.iter().zip(&pred.forces) {
                    force_atoms.push([0, 1, 2].map(|a| [ft[a] as f32, fm[a] as f32]));
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
        let force_points_total = 3 * force_atoms.len();
        let parity_force = thin_atoms(&force_atoms, FORCE_PARITY_CAP);
        let floor2 = if model.energy_per_atom { 2.0 * model.sigma_e } else { model.sigma_e };
        // the curve asks "would this model answer this dimer at its own settings"
        let prov = model.provenance.clone();
        let species = &model.basis.species;
        let (mut curves, mut curves_skipped) = (Vec::new(), 0usize);
        for (ia, &za) in species.iter().enumerate() {
            'pair: for &zb in &species[ia..] {
                let zs = [za, zb];
                let e_ab = e_ref(&zs).expect("both species are the basis's");
                // rules 1–4 depend on the species and the settings only: one gate per pair
                let gate = self.gate(&zs, &prov).err().map(|g| g.reason.rule);
                let mut c = PairCurve {
                    z: (za, zb),
                    r: Vec::with_capacity(CURVE_POINTS),
                    energy: Vec::with_capacity(CURVE_POINTS),
                    sigma: Vec::with_capacity(CURVE_POINTS),
                    force_ratio: Vec::with_capacity(CURVE_POINTS),
                    refused: Vec::with_capacity(CURVE_POINTS),
                    dots: dots.remove(&(za, zb)).unwrap_or_default(),
                    r_cut,
                };
                for k in 0..CURVE_POINTS {
                    let x = CURVE_R_MIN + (r_cut * 0.999 - CURVE_R_MIN) * k as f64 / (CURVE_POINTS - 1) as f64;
                    let pos = [[-x / 2.0, 0.0, 0.0], [x / 2.0, 0.0, 0.0]];
                    // one prediction per point serves the band and the verdict (plan review S2);
                    // a failed one skips this curve, counted (S18)
                    let Ok(p) = model.predict(&pos, &zs, true) else {
                        curves_skipped += 1;
                        continue 'pair;
                    };
                    c.energy.push(p.energy - e_ab);
                    if let (Some(e), Some(f)) = (p.energy_std, p.force_std_epistemic.as_ref()) {
                        c.sigma.push((e * e - floor2 * floor2).max(0.0).sqrt());
                        // rule 6's ratio: max over the atoms of σ_F,epi/(√3 · bound) (plan review M5)
                        c.force_ratio.push(self.variance_check(model, &p, e, f).atoms.iter().copied().fold(0.0, f64::max));
                    }
                    c.refused.push(match gate {
                        Some(rule) => Some(rule),
                        None => self.verdict_past_gate(&pos, &zs, Some(&p)).err().map(|e| e.0),
                    });
                    c.r.push(x);
                }
                curves.push(c);
            }
        }
        let pairs = hist.into_iter().map(|(z, counts)| PairHistogram { z, bin: BIN, counts }).collect();
        Some(TrainingViews {
            epoch: self.epoch,
            samples: store.len(),
            parity_energy,
            parity_force,
            force_points_total,
            parity_omitted,
            curves,
            curves_skipped,
            pairs,
            dimer_floor: floor2,
            energy_tier: self.policy.energy_std_per_atom,
            build_ms: t0.elapsed().as_secs_f64() * 1e3,
        })
    }
}

/// Deterministic thinning of the force parity to at most `cap` points, by whole atoms (plan
/// review S5): a component stride that is a multiple of 3 would keep one Cartesian component
/// only. Keeps every `stride`-th atom with stride = ⌈atoms / ⌊cap/3⌋⌉, hence ≤ cap components.
pub(super) fn thin_atoms(atoms: &[[[f32; 2]; 3]], cap: usize) -> Vec<[f32; 2]> {
    let stride = atoms.len().div_ceil((cap / 3).max(1)).max(1);
    atoms.iter().step_by(stride).flat_map(|a| a.iter().copied()).collect()
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}
```

(`gate`, `envelope_outside`, `class_untrusted`, `variance_check` and `NullSpace::novelty` take `&self` today; if one does not, stop and report rather than make `would_answer` `&mut`.) `Trainable::training_views` calls the inherent method.

- [ ] **Step 4: Run** `cargo test -p providers --release` → PASS; the S = 4 printout (build time and the worker share) goes in the task report. `cargo test -p validate --release --test surrogate` → PASS, unchanged.

- [ ] **Step 5: Physics review.** `physics-reviewer` with spec §9: the parity axes and the fit's own references; the band (epistemic, floor 2·s_E for per-atom models) and the rule-6 ratio; `would_answer` (read-only by type, the priority order with early returns, its agreement with `evaluate` including atoms, the fence, rule 4 and draft); the pin that a view build moves nothing; thinning by whole atoms; the histogram's fitted-only population. Apply the review before committing.

- [ ] **Step 6: Commit**

```bash
git add crates/providers/src/surrogate/views.rs crates/providers/src/surrogate/mod.rs crates/providers/src/surrogate/quarantine.rs crates/providers/src/surrogate/tests.rs
git commit -o -m "feat(providers): training views and a read-only would_answer for the Activity tab

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -- crates/providers/src/surrogate/views.rs crates/providers/src/surrogate/mod.rs crates/providers/src/surrogate/quarantine.rs crates/providers/src/surrogate/tests.rs
```

---

### Task 5: the worker ships activity in every snapshot **[Opus]**

**Files:**
- Create: `crates/app/src/worker/activity.rs`
- Modify: `crates/app/src/worker/mod.rs`, `crates/app/src/worker/commands.rs`, `crates/app/src/worker/model.rs`
- Test: `crates/app/src/worker/tests.rs`

**Interfaces:**
- Consumes: Tasks 1–4 (`Trainable::training_views`, the status fields, `with_probe`) and Task 1b's `Dynamics::{take_teach_events, teach_events_dropped}` (the events arrive stamped with their step: the worker does not stamp, plan review M6).
- Produces:
  - `pub struct FitPoint { pub epoch: u64, pub samples: usize, pub rmse_e: f64, pub rmse_f: f64, pub trusted: bool, pub at: Instant }`
  - `#[derive(Default)] pub struct ActivityDelta { pub events: Vec<sim::TeachEvent>, pub fit_points: Vec<FitPoint>, pub views: Option<Arc<sim::TrainingViews>>, pub activity_ms: f64, pub teacher_ms: f64, pub dropped: u64 }`
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

/// Review Focus 1 and 6: every teaching solve reaches the viewer once, in order, stamped with
/// the step whose evaluation produced it (by the driver, Task 1b): through ticks, paused bursts
/// and early publishes alike; the session count equals the model's lifetime count for a fresh
/// model.
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
    // an event is never stamped later than the snapshot that carries it
    let mut late = Vec::new();
    let mut collect = |s: &Snapshot| {
        late.extend(s.activity.events.iter().filter_map(|e| e.step).filter(|&st| st > s.dynamics.step));
        events.extend(s.activity.events.iter().cloned());
    };
    snapshots_until(&w, "200 steps", &mut collect, |s| s.dynamics.step >= 200);
    w.send(Command::Dyn(DynCommand::Pause));
    let s = snapshots_until(&w, "idle", &mut collect, |s| !s.working);
    let m = s.model.as_ref().unwrap();
    let counted = events.iter().filter(|e| e.counts_for_model()).count() as u64;
    assert_eq!(counted, m.solves.total(), "no event lost or duplicated");
    assert!(events.windows(2).all(|p| p[0].at <= p[1].at), "in order");
    assert!(late.is_empty(), "stamped after the snapshot that carried them: {late:?}");
    let origin = |o| events.iter().filter(move |e| e.origin == o);
    assert!(origin(sim::activity::TeachOrigin::Pass).all(|e| e.step.is_none()));
    assert!(origin(sim::activity::TeachOrigin::Fallback).all(|e| e.step.is_some()));
    let steps: Vec<u64> = origin(sim::activity::TeachOrigin::Fallback).filter_map(|e| e.step).collect();
    assert!(steps.windows(2).all(|w| w[0] <= w[1]), "the driver's steps never go back: {steps:?}");
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
//! The worker's side of the Activity view (activity spec §4.3 A7, §6): drains the driver's
//! teaching events once per publish (already stamped with the step whose evaluation produced
//! each, by `sim::Dynamics`: plan review M6), keeps the fit history, builds the training views
//! only while the UI shows them, and accounts its own cost.
use super::*;

/// The fewest seconds between two training-view builds (spec §6).
pub const VIEWS_MIN_INTERVAL: Duration = Duration::from_secs(2);

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
    /// The teaching events since the last publish, oldest first, each with its driver step.
    pub events: Vec<sim::TeachEvent>,
    pub fit_points: Vec<FitPoint>,
    pub views: Option<Arc<sim::TrainingViews>>,
    /// This worker's activity bookkeeping and teacher time so far (ms; spec §6).
    pub activity_ms: f64,
    pub teacher_ms: f64,
    /// Events the driver's buffer dropped before a publish drained them (this worker, so far).
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
    /// The activity delta of one publish (called by `publish`, after the model readouts).
    pub(super) fn activity_delta(&mut self, model: Option<&ModelStatus>) -> ActivityDelta {
        let t = Instant::now();
        let mut d = ActivityDelta::default();
        // stamped by the driver right after each evaluate (Task 1b); nothing to add here
        d.events = self.dy.take_teach_events();
        d.dropped = self.dy.teach_events_dropped();
        self.activity.teacher_ms += d.events.iter().map(|e| e.solve_ms).sum::<f64>();
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
        let activity = self.activity_delta(model.as_ref());
```

(`ModelStatus::last_audit_step`, the M6 readout "verdict step less its lag", has the units problem of plan review M6(a) too: the lag counts answered steps only. The Activity log does not use it; it is left for the M6 owner and noted in the task report.)

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
            solve_ms.extend(s.activity.events.iter().filter(|e| e.is_solve()).map(|e| e.solve_ms));
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
- Consumes: Task 5's `Snapshot::activity` (events as `sim::TeachEvent`, each with its driver `step`; `run_started`, the instant a new run began: Task 5 review M2), `SimWorker::probe`, `Command::SetActivityViews`. Task 5 left `#[cfg_attr(not(test), expect(dead_code, …))]` on `Snapshot::activity`, `ActivityDelta::{events, run_started}`, `SimWorker::probe` (field and method) and `Command::SetActivityViews`: remove each one as this task (or Tasks 7–9) reads it — the build then warns until it is gone.
- **Contract (Task 5 review I2):** `ActivityDelta::views == None` means "no new build", not "no views". `absorb` drops `self.views` whenever the snapshot has no fitted model (`s.model` absent or `!fitted`: a Forget, a load refused, another backend), and `Viewer::start` drops it with the old worker.
- Produces:
  - `ui::activity::ActivityState` with `pub fn new() -> Self`, `pub fn absorb(&mut self, s: &Snapshot)`, `pub session_solves: u64`, `pub rows: VecDeque<LogRow>`, `pub fit_points: Vec<FitPoint>`, `pub views: Option<Arc<TrainingViews>>`, `pub training_open: bool`, `pub views_sent: bool`, `pub overhead: (f64, f64)`, `pub timeline: Vec<Mark>`, `pub burst_hist: Vec<u32>` (plan review S6), `pub runs: Vec<Duration>` (session times of run starts: Task 5 review M2)
  - `pub struct LogRow { pub at: Duration, pub event: sim::TeachEvent, pub near_face: bool }`, `pub const LOG_ROWS: usize = 5000;`, `pub const BURST_BIN_K: f64 = 100.0;`
  - `jobs::{JobKey, JobProgress, JobRow, rows, show}` (spec §7, exact signatures there)
  - `PanelInput::probe: Option<sim::ProbeView>`
  - `pub enum Mark { Fallback(Duration), Audit(Duration, sim::AuditVerdict), Refit(Duration, bool), Quarantine(Duration, usize), Pass(Duration, Option<Duration>, bool) }`

- [ ] **Step 1: Write the failing tests** (`ui/tests.rs`)

```rust
fn event(origin: sim::activity::TeachOrigin, rule: sim::activity::TeachRule, outcome: sim::activity::TeachOutcome, step: Option<u64>) -> sim::TeachEvent {
    let mut e = sim::TeachEvent::new(origin, rule, outcome);
    e.step = step;
    e.n_atoms = 2;
    e.solve_ms = 92.0;
    e.gap = Some(3.1e-3);
    e
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
    assert_eq!(a.rows.back().unwrap().event.rule, R::Cell, "newest at the back, drawn first");
}

/// Plan review S6: the burst-temperature histogram covers the whole session, apart from the
/// 5 000-row ring that drops early rows.
#[test]
fn burst_temperatures_are_kept_for_the_whole_session() {
    use sim::activity::{PassJob, TeachOrigin as O, TeachOutcome as K, TeachRule as R};
    let scene = Scene::preset("h2").unwrap();
    let mut a = activity::ActivityState::new();
    let mut snap = snapshot(&scene, Backend::Estimated, Some(model()));
    snap.activity.events = (0..10)
        .map(|i| {
            let mut e = event(O::Pass, R::Pass(PassJob::HotBurst), K::Kept, None);
            e.burst_t = Some(1850.0 + i as f64);
            e
        })
        .collect();
    a.absorb(&snap);
    snap.activity.events = (0..6000).map(|i| event(O::Fallback, R::Grade, K::Kept, Some(i))).collect();
    a.absorb(&snap);
    assert!(a.rows.iter().all(|r| r.event.burst_t.is_none()), "the ring dropped the bursts");
    assert_eq!(a.burst_hist.iter().sum::<u32>(), 10);
    assert_eq!(a.burst_hist[(1850.0 / activity::BURST_BIN_K) as usize], 10);
}

/// Plan review M6: the step cell is the driver step of the event's evaluation; an audit says how
/// many answered steps after its dispatch it was judged, and no dispatch step is invented.
#[test]
fn the_step_cell_says_what_it_means() {
    use activity::log::step_cell;
    use sim::activity::{PassJob, TeachOrigin as O, TeachOutcome as K, TeachRule as R};
    assert_eq!(step_cell(&event(O::Pass, R::Pass(PassJob::Scan), K::Kept, None)), ("pass".to_string(), None));
    assert_eq!(step_cell(&event(O::Fallback, R::Grade, K::Kept, Some(1607))), ("1607".to_string(), None));
    let mut a = event(O::Audit, R::Audit, K::Kept, Some(1640));
    a.audit_lag = Some(40);
    assert_eq!(step_cell(&a), ("1640".to_string(), Some("judged 40 answered steps after its dispatch".to_string())));
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

/// Task 5 review I2: the views are dropped when a snapshot has no fitted model (a Forget) and
/// kept across snapshots whose `views` is `None` while the model is fitted.
#[test]
fn views_are_dropped_without_a_fitted_model() {
    let scene = Scene::preset("h2").unwrap();
    let mut st = activity::ActivityState::new();
    let mut s = snapshot(&scene, Backend::Estimated, Some(model()));
    s.activity.views = Some(std::sync::Arc::new(activity::fixture_views()));
    st.absorb(&s);
    assert!(st.views.is_some());
    s.activity.views = None;
    st.absorb(&s);
    assert!(st.views.is_some(), "None = no new build");
    st.absorb(&snapshot(&scene, Backend::Estimated, Some(ModelStatus { fitted: false, ..model() })));
    assert!(st.views.is_none(), "forgotten");
    s.activity.views = Some(std::sync::Arc::new(activity::fixture_views()));
    st.absorb(&s);
    st.absorb(&snapshot(&scene, Backend::Gpu, None));
    assert!(st.views.is_none(), "no model at all");
}

/// Task 5 review M2: the run divider goes by event time — an old-run event carried by the
/// snapshot that announces the new run is placed before the divider, the new run's after it.
#[test]
fn the_run_divider_follows_event_time() {
    let scene = Scene::preset("h2").unwrap();
    let mut st = activity::ActivityState::new();
    let old = sim::TeachEvent { step: Some(40), ..sim::TeachEvent::new(TeachOrigin::Fallback, TeachRule::Spread, TeachOutcome::Kept) };
    let run = Instant::now();
    let new = sim::TeachEvent { step: Some(0), ..sim::TeachEvent::new(TeachOrigin::Fallback, TeachRule::Spread, TeachOutcome::Kept) };
    let mut s = snapshot(&scene, Backend::Estimated, Some(model()));
    s.activity.events = vec![old, new];
    s.activity.run_started = Some(run);
    st.absorb(&s);
    let r = *st.runs.last().unwrap();
    assert!(st.rows[0].at < r && st.rows[1].at >= r);
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

Extend `every_layout_draws_and_sends_nothing_without_input`: `DockTab::ALL` now includes `Activity` (the loop covers it); add a snapshot whose `activity` carries a few events of each origin and a `views` built from a fixture (`activity::fixture_views()` in the test module: two parity points, one curve of 96 points with `sigma` and `force_ratio` filled and `refused` set to `Some(TeachRule::IsolatedAtom)` on its last 20 points, one histogram, `parity_omitted: 1`, `curves_skipped: 0`) so the plots of Task 9 draw too. Add `ui::search` entries for `d-jobs`, `d-counters`, `d-solve-log`, `d-trust`, `d-training`, `d-timeline` (the anchor test fails until each is marked).

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

`ActivityState::new()` starts with `views_sent: false`, so nothing is sent while closed. `viewer.rs`: in `on_event_with`, next to `self.log(&s)`, `self.ui.activity.absorb(&s);`; in `Viewer::start`, `self.ui.activity.views_sent = false;` and `self.ui.activity.views = None;` (the old worker's model; Task 5 review I2 — the new worker's first snapshot also carries `run_started`, so the log divides there).

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

use crate::sim::{FitPoint, Snapshot};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Solve-log rows kept (spec §3.3.3).
pub const LOG_ROWS: usize = 5000;
/// The burst-temperature histogram's bin (K; spec §3.3.5).
pub const BURST_BIN_K: f64 = 100.0;

pub struct LogRow {
    pub at: Duration,
    /// The event as the worker shipped it, with the driver step of its evaluation (`step`).
    pub event: sim::TeachEvent,
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
    /// Hot-burst geometries solved this session, by instantaneous temperature in bins of
    /// [`BURST_BIN_K`]: kept apart from the log ring so early bursts are not lost (plan review S6).
    pub burst_hist: Vec<u32>,
    /// Session times at which a new run began (`ActivityDelta::run_started`: the worker's start,
    /// a scene replacement, a panic recovery). The log draws a "new run — steps restart" divider
    /// before the first row whose `at` is not earlier: by event time, never by snapshot, because
    /// the old run's last events often arrive in the first snapshot after the replacement (Task 5
    /// review M2).
    pub runs: Vec<Duration>,
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
                let bin = (t / BURST_BIN_K) as usize;
                if self.burst_hist.len() <= bin {
                    self.burst_hist.resize(bin + 1, 0);
                }
                self.burst_hist[bin] += 1;
            }
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
            let near_face = e.atom.and_then(|i| scene.nuclei.get(i)).is_some_and(|n| words::face_distance(scene, n.pos) < d_box);
            self.rows.push_back(LogRow { at, event: e.clone(), near_face });
            if self.rows.len() > LOG_ROWS {
                self.rows.pop_front();
            }
        }
        for p in &a.fit_points {
            self.timeline.push(Mark::Refit(self.since(p.at), p.trusted));
            self.fit_points.push(p.clone());
        }
        // `views: None` = no new build; no fitted model = no views (Task 5 review I2)
        if s.model.as_ref().is_none_or(|m| !m.fitted) {
            self.views = None;
        } else if let Some(v) = &a.views {
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
        self.rows.iter().rev().find_map(|r| match r.event.rule {
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
            // held vs fitted (plan review S9): the trust check counts the fit's samples
            let fitted = m.fit.map_or(String::new(), |f| format!("; {} in the current fit", f.0));
            ui.label(egui::RichText::new(format!("samples in the model {}", m.samples)).strong())
                .on_hover_text(format!("held{fitted} (the store; the Model tab shows the same number)"));
            let c = &m.solves;
            let total = if c.unrecorded_before { format!("≥ {} (earlier solves unrecorded)", (c.total() as usize).max(m.samples)) } else { c.total().to_string() };
            ui.label(egui::RichText::new(format!("teacher solves for this model {total}")).strong()).on_hover_text(format!(
                "pass {} · fallback {} · audit {} · relax checks {} — kept {}, not kept {} (references, similar answers, box-affected, failed, discarded audits, …). Completed or failed teaching solves; a cancelled solve is not counted",
                c.pass, c.fallback, c.audit, c.verification, c.kept, c.not_kept()
            ));
        }
        // teaching solves only: a plain DFT step is not one (plan review S16)
        ui.label(egui::RichText::new(format!("teaching solves this session {}", st.session_solves)).strong())
            .on_hover_text("teacher solves offered to a model since the app started, over every solver switch; plain DFT steps are not counted");
    });
}

/// The step cell of a log row and its hover (plan review M6): `pass` for a pass solve, else the
/// driver step whose evaluation produced the event; an audit's hover says how many answered
/// steps after its dispatch it was judged (fallback steps in between are not counted in that lag).
pub fn step_cell(e: &sim::TeachEvent) -> (String, Option<String>) {
    let text = match (e.origin, e.step) {
        (sim::activity::TeachOrigin::Pass, _) | (_, None) => "pass".to_string(),
        (_, Some(s)) => s.to_string(),
    };
    let hover = e.audit_lag.map(|k| format!("judged {k} answered steps after its dispatch"));
    (text, hover)
}
```

`show_log(ui, st, input)`: the filter chips (`selectable_label` × 5) and the table drawn with `egui::ScrollArea::vertical().max_height(260.0).show_rows(ui, row_h, n, |ui, range| …)` over the filtered rows in reverse order (newest first). Each row: `logbook::clock(row.at)`, the step cell (`step_cell`), origin (colour: pass `theme::ACCENT`, fallback `AMBER`, audit `GREEN`, check `theme::INK_DIM`), rule (`words::rule_label`, hover `words::rule_sentence` + "\n\n" + `detail`), outcome (`words::outcome_label`, hover `words::outcome_sentence`, plus "; the answer is box-affected (face density ρ e/Bohr³)" when `box_affected` and the outcome is neither `KeptBoxAffected` nor `BoxNotKept`, as a vote can keep a box-affected answer: plan review N1), |ΔE| (`format!("{:.2} mHa", gap.abs() * 1e3)`, hover per atom and the signed Ha value), solve (`{ms:.0} ms` or `{s:.1} s` above 10 s, hover the SCF steps). A divider row "new run — steps restart at 0" is drawn before the first (oldest-first) row whose `at` ≥ an entry of `st.runs` (none before the first row of the session: the worker's own start); the step cell numbers steps within its run (Task 5 review M2). Empty text: "no teacher solves yet — they appear here as they land (estimated solver)". `dropped > 0` adds "k older events were dropped before they reached the view (a fallback storm); the counters include them".

`dock/mod.rs`: icon `Icon::Activity` (a pulse polyline in `icons.rs`, drawn like the others in the 24 × 24 box: `M3 12h4l2-5 4 10 2-5h6`), peek "Activity\n{headline or idle}\n{session} teaching solves this session", the body arm `DockTab::Activity => super::activity::show(ui, st, input, acts)`, and a small accent dot on the strip button while `input.probe` shows work or `input.sampling()`.

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

Known and accepted (plan review N10): without a background auditor (the CPU teacher), an audit runs in the step, and while it runs the headline reads "DFT fallback · SCF step k". The probe cannot tell an audit from a fallback mid-solve. The log row names it `8 audit` once it lands.

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

The plan review changed the wording in several places:
- The audit line is per atom or names forces, and is worded from `last_audit` when met (M1).
- The class line words `ClassLoo::refusal` (M3).
- The `ForceNoise` line no longer blames the teacher (S8).
- "Samples" means samples in the fit (S9).
- Sentences that could be false are conditioned on the facts (S13): "teaches it" only when the answer was kept; the new-species promise; "not used"; the constants named in code.
- Rule 4 on a log row is worded by its key (S14).
- The box goes first only for a fit condition (S15).
- The box line says which samples it counts (N6). The classes are explained on hover (N8). `mha` keeps two significant figures (N7).

**Files:**
- Modify: `crates/app/src/ui/activity/words.rs`, `crates/app/src/ui/activity/trust.rs`, `crates/app/src/ui/activity/log.rs` (rule sentences with `RowCtx`), `crates/app/src/ui/mod.rs` (`trust()` uses plain lines, `TrustView::detail`), `crates/app/src/ui/dock/model.rs` (trusted line hover), `crates/app/src/ui/toolbar.rs` (trust tooltip), `crates/app/src/worker/model.rs` (`box_blocks_trust` keys on the first unmet `TrustKey`)
- Test: `crates/app/src/ui/tests.rs`

**Interfaces:**
- Consumes: `ModelStatus::{trust_checks, stuck_info, class_loo, last_audit, audit_every}`, `ClassLoo::refusal` (Task 3), `box_blocks_trust`, `box_advice`, `LogRow::near_face`.
- Produces: `words::{RowCtx, rule_label, rule_sentence, untrusted_title, outcome_label, outcome_sentence, trust_line, class_line, stuck_sentence, stage_word, mha, d_box, face_distance}`, `trust::{Line, lines, show}`.

- [ ] **Step 1: Write the failing tests**

```rust
/// The trust-check list of an untrusted fit whose first unmet condition is `key` (the box
/// fixtures carry the list `box_blocks_trust` reads; plan review S15).
fn checks_failing(key: sim::activity::TrustKey) -> Vec<sim::activity::TrustCheck> {
    use sim::activity::{TrustCheck, TrustKey as K};
    let audit = if key == K::AuditForces { K::AuditForces } else { K::Audit };
    [K::Samples, K::ResidualDofE, K::ResidualDofF, K::EnergyResidual, K::ForceResidual, K::ForceNoise, audit]
        .into_iter()
        .map(|k| TrustCheck { key: k, met: k != key, value: 0.0, limit: 1.0 })
        .collect()
}

/// Spec §5.1 with the box clause: the user's sentence verbatim when the named atom is near a face
/// and the answer was box-affected; the other cases; rules 6–7 say "teaches it" only for a kept
/// answer (plan review S13); a log row words rule 4 by its key (S14).
#[test]
fn the_box_clause_names_the_atom_at_the_edge() {
    use activity::words::{rule_sentence, RowCtx};
    use sim::activity::{TeachRule as R, TrustKey};
    let cx = |box_affected, near_face, kept| RowCtx { box_affected, near_face, kept };
    assert_eq!(
        rule_sentence(R::Spread, Some(0), cx(true, true, false)),
        "atom 0 is outside what the model has learned and is touching the box edge — move it inward"
    );
    assert_eq!(
        rule_sentence(R::Grade, Some(2), cx(true, false, false)),
        "atom 2 is outside what the model has learned — this answer also touches the box walls, so it is not learned here; a bigger cell would help"
    );
    assert_eq!(
        rule_sentence(R::Grade, Some(2), cx(true, false, true)),
        "atom 2 is outside what the model has learned — this answer also touches the box walls, so it stays in the fit only until enough others exist; a bigger cell would help"
    );
    assert_eq!(rule_sentence(R::Grade, Some(2), cx(false, true, true)), "atom 2 is outside what the model has learned");
    assert_eq!(rule_sentence(R::ForceUncertainty, Some(1), cx(false, false, true)), "the model is unsure here (atom 1) — this answer teaches it");
    assert_eq!(rule_sentence(R::ForceUncertainty, Some(1), cx(false, false, false)), "the model is unsure here (atom 1)");
    assert_eq!(
        rule_sentence(R::ForceUncertainty, Some(1), cx(true, true, false)),
        "the model is unsure here (atom 1), and atom 1 is touching the box edge — move it inward"
    );
    assert_eq!(
        rule_sentence(R::NewSpecies(8), None, RowCtx::default()),
        "the model has never seen O — answers like this one teach it; it can answer for O after a refit with O's reference energy and enough O samples"
    );
    assert_eq!(rule_sentence(R::Untrusted(TrustKey::Samples), None, RowCtx::default()), "the model was not trusted then: too few samples in the fit");
}

/// Short labels as the user wrote them (spec §3.3.3); 5d and 5e at γ = ∞ are different tests
/// (plan review S7).
#[test]
fn rule_labels_are_the_codes_the_user_reads() {
    use activity::words::rule_label;
    use sim::activity::{PassJob, TeachRule as R};
    assert_eq!(rule_label(R::Grade, None), "5e grade");
    assert_eq!(rule_label(R::UnspannedAtom, None), "5e new environment");
    assert_eq!(rule_label(R::NewEnvironment, None), "5d new environment");
    assert_eq!(rule_label(R::IsolatedAtom, None), "5a lone atom");
    assert_eq!(rule_label(R::Cell, None), "0 cell");
    assert_eq!(rule_label(R::Pass(PassJob::HotBurst), Some(1830.0)), "hot burst 1830 K");
    assert_eq!(rule_label(R::Pass(PassJob::Reference { z: 1, multiplicity: 2 }), None), "reference H m2");
}

/// Plan review S13: the outcome words follow the policy's constants and say what happened.
#[test]
fn outcome_words_follow_the_constants_and_the_facts() {
    use activity::words::{outcome_label, outcome_sentence};
    use providers::surrogate::{FENCE_VOTES, THIN_EVERY};
    use sim::activity::TeachOutcome as K;
    assert_eq!(outcome_sentence(K::Thinned), format!("skipped: too similar to a sample already held (one in {THIN_EVERY} such is kept)"));
    assert!(outcome_sentence(K::KeptByVote).starts_with(&format!("kept: {FENCE_VOTES} fresh answers agree with each other")));
    assert_eq!(outcome_label(K::NotOffered), "not kept: other settings");
    assert_eq!(outcome_sentence(K::NotOffered), "used for this step, not kept for this model: its settings differ");
    assert_eq!(outcome_label(K::Discarded), "discarded");
}

/// Plan review N7: mHa with two significant figures (σ_f 4.5e-5 Ha/Bohr is "0.045", not "0.04").
#[test]
fn mha_keeps_two_significant_figures() {
    use activity::words::mha;
    assert_eq!(mha(1e-3), "1");
    assert_eq!(mha(2.2e-3), "2.2");
    assert_eq!(mha(4.5e-5), "0.045");
    assert_eq!(mha(4.5e-4), "0.45");
    assert_eq!(mha(1.23e-2), "12");
    assert_eq!(mha(0.0), "0");
}

/// Plan review M1: the audit line is per atom or names forces, and when met it says what the
/// last audit was (or that none ran), never "passed" by default.
#[test]
fn the_audit_line_says_what_ran() {
    use activity::words::trust_line;
    use sim::activity::{TrustCheck, TrustKey as K};
    use sim::AuditVerdict as V;
    let met = TrustCheck { key: K::Audit, met: true, value: 0.0, limit: 1e-3 };
    let rec = |verdict| sim::AuditRecord { gap: 2e-4, verdict, face_density: None, unaudited_steps: 0, asynchronous: true, teacher_ms: 90.0 };
    let head = "Last check against DFT — no failed check since the last refit";
    assert_eq!(trust_line(&met, None, 20), format!("{head} (last check: none yet, one every 20 answered steps)"));
    assert_eq!(trust_line(&met, None, 0), format!("{head} (last check: none, audits are off)"));
    assert_eq!(trust_line(&met, Some(&rec(V::Passed)), 20), format!("{head} (last check: passed)"));
    assert_eq!(trust_line(&met, Some(&rec(V::BoxAffected)), 20), format!("{head} (last check: box-affected, not judged)"));
    assert_eq!(trust_line(&met, Some(&rec(V::Failed)), 20), format!("{head} (last check: failed before the refit)"));
    let energy = TrustCheck { key: K::Audit, met: false, value: 2.5e-3, limit: 1e-3 };
    assert_eq!(trust_line(&energy, Some(&rec(V::Failed)), 20), "Last check against DFT — found the energy off by 2.5 mHa/atom, allowed 1 mHa/atom (refit to recover)");
    let forces = TrustCheck { key: K::AuditForces, met: false, value: 4.2e-3, limit: 2e-3 };
    assert_eq!(trust_line(&forces, Some(&rec(V::Failed)), 20), "Last check against DFT — found forces off by 4.2, allowed 2 mHa/Bohr (refit to recover)");
}

/// Plan review M3: the class line words the class's own verdict, so an energy-only failure is
/// an energy error.
#[test]
fn the_class_line_words_the_failing_condition() {
    use activity::words::class_line;
    use sim::{ClassKey, ClassLoo};
    let l = |class, refusal: Option<(ClassKey, f64, f64)>| ClassLoo { class, samples: 120, force_rms: 0.1, force_max: 0.4, energy_max: 1100.0, trusted: refusal.is_none(), refusal };
    assert_eq!(class_line(&l("pair", None)), "Two-atom arrangements — learned");
    assert_eq!(class_line(&l("pair", Some((ClassKey::Energy, 1100.0, 1.0)))), "Two-atom arrangements — not yet (leave-one-out energy error 1100.0× the tier)");
    assert_eq!(class_line(&l("pair", Some((ClassKey::Force, 2.69, 1.0)))), "Two-atom arrangements — not yet (leave-one-out force error 2.7× the bound)");
    assert_eq!(class_line(&l("many-body", Some((ClassKey::Samples, 12.0, 40.0)))), "Three-or-more-atom arrangements — not yet (12 < 40 samples)");
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
    assert!(lines[0].why && !lines[0].met && lines[0].text.starts_with("Samples clear of the box walls — 130 in the fit touch them (145 held)"), "{:?}", lines[0]);
    let e = lines.iter().find(|l| l.text.starts_with("Reproduces its own training energies")).unwrap();
    assert!(!e.met && !e.why && e.text.contains("worst 2.2 mHa/atom, allowed 1 mHa/atom"));
    assert_eq!(lines.iter().filter(|l| l.why).count(), 1);
    // the Model tab's line uses the box sentence, the old reason on hover
    let t = trust(Some(&m), &scene, &st);
    assert!(t.text.starts_with("no — Samples clear of the box walls") && t.detail.as_deref().is_some_and(|d| d.contains(reason)));
}

/// Plan review S15: the box is blamed only when the first unmet check is a condition of the fit
/// it is in. Not a count (Sample more cures it), not a failed audit (judged against a fresh
/// teacher answer that is not box-affected).
#[test]
fn the_box_is_not_blamed_for_a_failed_audit_or_a_count() {
    use sim::activity::TrustKey as K;
    let m = crate::sim::ModelStatus { trusted: false, untrusted_reason: Some("…".into()), samples: 145, quarantined: 0, box_affected: 130, box_excluded: 0, trust_floor: Some(59), ..model() };
    for key in [K::EnergyResidual, K::ForceResidual, K::ForceNoise] {
        assert_eq!(crate::sim::box_blocks_trust(&crate::sim::ModelStatus { trust_checks: checks_failing(key), ..m.clone() }), Some((130, 145)), "{key:?}");
    }
    for key in [K::Samples, K::ResidualDofE, K::ResidualDofF, K::Audit, K::AuditForces] {
        assert_eq!(crate::sim::box_blocks_trust(&crate::sim::ModelStatus { trust_checks: checks_failing(key), ..m.clone() }), None, "{key:?}");
    }
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
    assert!(lines[0].why && lines[0].text == "Enough samples in the fit — 12 of 16 needed (one per independent direction the model can bend)");
    assert_eq!(trust(Some(&m), &scene, &st).text, "no — Enough samples in the fit — 12 of 16 needed (one per independent direction the model can bend)");
}

/// The stuck variant in plain words (spec §5.3); no stored reason, no invented rule (plan
/// review S10).
#[test]
fn the_stuck_line_is_plain() {
    let info = sim::activity::StuckInfo { rule: Some(sim::activity::TeachRule::Spread), atom: Some(0), steps: 50, box_streak: true, gap: Some(2e-3), face_density: 4e-5 };
    let mut scene = Scene::preset("h2").unwrap();
    scene.nuclei[0].pos = [0.0, 0.0, 5.0]; // 1 Bohr from the +z face of the preset's 49³ × 0.25 cell (d_box(0.25) = 4.84)
    let sim_scene = scene.to_sim().unwrap();
    assert_eq!(
        activity::words::stuck_sentence(&info, &sim_scene),
        "trusted, but not answering here: atom 0 is outside what the model has learned and is touching the box edge — move it inward (the last 50 steps were DFT)"
    );
    let unknown = sim::activity::StuckInfo { rule: None, ..info };
    assert_eq!(activity::words::stuck_sentence(&unknown, &sim_scene), "trusted, but not answering here (the last 50 steps were DFT)");
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p app --release -- ui::tests::the_box ui::tests::rule_labels ui::tests::outcome_words ui::tests::mha ui::tests::the_audit_line ui::tests::the_class_line ui::tests::the_three ui::tests::the_first_unmet ui::tests::the_stuck`
Expected: compile errors.

- [ ] **Step 3: Implement `words.rs`**

```rust
//! Plain wording (activity spec §5): codes in, sentences out; the provider's technical text stays
//! on hover. Element symbols from `units` (display only, rule 1); mHa and Å at this edge (rule 3).
use providers::surrogate::{FENCE_VOTES, THIN_EVERY};
use sim::activity::{EnvClass, PassJob, StuckInfo, TeachOutcome as K, TeachRule as R, TrustCheck, TrustKey};
use sim::{AuditRecord, AuditVerdict, ClassKey};
use units::isotopes::symbol;

/// What a log row (or a stuck streak) knew about its answer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RowCtx {
    /// The teacher's answer was box-affected.
    pub box_affected: bool,
    /// The named atom was within `d_box(h)` of a cell face.
    pub near_face: bool,
    /// The answer was kept as a sample (`TeachOutcome::is_kept`).
    pub kept: bool,
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
        R::UnspannedAtom => "5e new environment".into(),
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

/// A trust condition by its key alone: a log row is history, so it never shows today's numbers
/// (plan review S14; the numbers at the time are in the row's technical text).
pub fn untrusted_title(key: TrustKey) -> &'static str {
    match key {
        TrustKey::Samples => "too few samples in the fit",
        TrustKey::ResidualDofE | TrustKey::ResidualDofF => "too little data beyond the fit's own parameters",
        TrustKey::EnergyResidual => "it did not reproduce its own training energies",
        TrustKey::ForceResidual => "it did not reproduce its own training forces",
        TrustKey::ForceNoise => "its forces did not fit the data to within the force tier",
        TrustKey::Audit | TrustKey::AuditForces => "the last check against DFT had failed",
    }
}

const OUTSIDE: &str = "is outside what the model has learned";

/// Spec §5.1.
pub fn rule_sentence(rule: R, atom: Option<usize>, cx: RowCtx) -> String {
    let a = |i: Option<usize>| i.map_or("an atom".to_string(), |i| format!("atom {i}"));
    // the box clause; `subject`: the sentence's subject is the atom itself (the user's sentence)
    let with_box = |s: String, subject: bool| match (cx.box_affected, cx.near_face) {
        (true, true) if subject => format!("{s} and is touching the box edge — move it inward"),
        (true, true) => format!("{s}, and {} is touching the box edge — move it inward", a(atom)),
        (true, false) if cx.kept => format!("{s} — this answer also touches the box walls, so it stays in the fit only until enough others exist; a bigger cell would help"),
        (true, false) => format!("{s} — this answer also touches the box walls, so it is not learned here; a bigger cell would help"),
        _ => s,
    };
    match rule {
        R::Cell => format!("{} is too close to the edge of the simulation cell — move it inward", a(atom)),
        R::NoModel => "there is no model yet — this answer helps build one".into(),
        // the refit needs El's reference, and the larger basis is untrusted until El samples
        // accumulate (plan review S13)
        R::NewSpecies(z) => {
            let el = symbol(z);
            format!("the model has never seen {el} — answers like this one teach it; it can answer for {el} after a refit with {el}'s reference energy and enough {el} samples")
        }
        R::OtherSettings => "this scene's settings (grid, smearing, charge or spin) differ from the model's — the answer cannot be used for it".into(),
        R::Untrusted(k) => format!("the model was not trusted then: {}", untrusted_title(k)),
        R::Class(EnvClass::Pair) => "the model hasn't learned two-atom arrangements well enough yet".into(),
        R::Class(EnvClass::ManyBody) => "the model hasn't learned three-or-more-atom arrangements well enough yet".into(),
        R::IsolatedAtom => with_box(format!("{} has no neighbour within reach — the model can't describe a lone atom inside a molecule's spin state", a(atom)), true),
        R::Coordination | R::Spread | R::NoGrade | R::UnspannedAtom | R::NewEnvironment | R::Grade => with_box(format!("{} {OUTSIDE}", a(atom)), true),
        R::ForceUncertainty | R::EnergyUncertainty => {
            let base = match atom {
                Some(i) => format!("the model is unsure here (atom {i})"),
                None => "the model is unsure here".into(),
            };
            // "teaches it" only for a kept answer that no box clause explains (plan review S13)
            if cx.box_affected {
                with_box(base, false)
            } else if cx.kept {
                format!("{base} — this answer teaches it")
            } else {
                base
            }
        }
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
        K::NotOffered => "not kept: other settings".into(),
        K::Discarded => "discarded".into(),
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
        K::KeptByVote => format!("kept: {FENCE_VOTES} fresh answers agree with each other but not with the model — the next refit decides"),
        K::KeptReleased { released, superseded } => format!("kept: it agrees with the model, so {released} quarantined samples were released ({superseded} superseded)"),
        K::Thinned => format!("skipped: too similar to a sample already held (one in {THIN_EVERY} such is kept)"),
        K::BoxNotKept => "not kept: it touches the box walls — the box's squeeze, not the molecule".into(),
        K::FenceDisagrees => "not kept: next to a geometry the model could not learn, and the teacher still disagrees with the model there".into(),
        K::WaitingForPass => "not kept: many answers are held without a model — run a sampling pass".into(),
        K::NotOffered => "used for this step, not kept for this model: its settings differ".into(),
        K::Discarded => "not judged and not kept: the model changed settings or species before the background answer arrived".into(),
        K::StoreRefused => "the store refused it (shape or settings)".into(),
        K::Reference => "a reference energy, not a sample".into(),
        K::Failed => "the solve failed".into(),
        K::NotSolved => "not solved".into(),
    }
}

/// Hartree → mHa (or Ha/Bohr → mHa/Bohr) for the plain lines: two significant figures, trailing
/// zeros dropped ("1", "2.2", "0.45", "0.045", "12"; plan review N7).
pub fn mha(x: f64) -> String {
    let v = x * 1e3;
    if v == 0.0 || !v.is_finite() {
        return format!("{v}");
    }
    let decimals = (1 - v.abs().log10().floor() as i32).max(0) as usize;
    let s = format!("{v:.decimals$}");
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s }
}

/// Spec §5.3. `last` and `audit_every` word the met audit line (plan review M1): it says what the
/// last audit was, or that none ran, never "passed" by default.
pub fn trust_line(c: &TrustCheck, last: Option<&AuditRecord>, audit_every: usize) -> String {
    match c.key {
        TrustKey::Samples => format!("Enough samples in the fit — {} of {} needed (one per independent direction the model can bend)", c.value as usize, c.limit as usize),
        TrustKey::ResidualDofE => format!("Enough data beyond the fit's own parameters to measure its noise — energies {:.0} of {:.0}", c.value, c.limit),
        TrustKey::ResidualDofF => format!("Enough data beyond the fit's own parameters to measure its noise — forces {:.0} of {:.0}", c.value, c.limit),
        TrustKey::EnergyResidual => format!("Reproduces its own training energies to the accuracy it claims — worst {} mHa/atom, allowed {} mHa/atom", mha(c.value), mha(c.limit)),
        TrustKey::ForceResidual => format!("Reproduces its own training forces — worst {}, allowed {} mHa/Bohr for that atom", mha(c.value), mha(c.limit)),
        // σ_f is label noise and basis misfit together (plan review S8): never "the labels disagree"
        TrustKey::ForceNoise => format!("Its forces fit the data to within the force tier on average — scatter σ_f {}, allowed {} mHa/Bohr", mha(c.value), mha(c.limit)),
        TrustKey::AuditForces => format!("Last check against DFT — found forces off by {}, allowed {} mHa/Bohr (refit to recover)", mha(c.value), mha(c.limit)),
        TrustKey::Audit if !c.met => format!("Last check against DFT — found the energy off by {} mHa/atom, allowed {} mHa/atom (refit to recover)", mha(c.value), mha(c.limit)),
        TrustKey::Audit => {
            let what = match last.map(|a| a.verdict) {
                None if audit_every == 0 => "none, audits are off".to_string(),
                None => format!("none yet, one every {audit_every} answered steps"),
                Some(AuditVerdict::Passed) => "passed".into(),
                Some(AuditVerdict::BoxAffected) => "box-affected, not judged".into(),
                Some(AuditVerdict::Failed) => "failed before the refit".into(),
            };
            format!("Last check against DFT — no failed check since the last refit (last check: {what})")
        }
    }
}

/// The class line from the class's own verdict (`ClassLoo::refusal`, plan review M3).
pub fn class_line(l: &sim::ClassLoo) -> String {
    let name = if l.class == "pair" { "Two-atom arrangements" } else { "Three-or-more-atom arrangements" };
    match l.refusal {
        None => format!("{name} — learned"),
        Some((ClassKey::Samples, n, q)) => format!("{name} — not yet ({n:.0} < {q:.0} samples)"),
        Some((ClassKey::Force, x, _)) => format!("{name} — not yet (leave-one-out force error {x:.1}× the bound)"),
        Some((ClassKey::Energy, x, _)) => format!("{name} — not yet (leave-one-out energy error {x:.1}× the tier)"),
    }
}

/// Spec §5.3, the stuck variant.
pub fn stuck_sentence(i: &StuckInfo, scene: &sim::Scene) -> String {
    let Some(rule) = i.rule else {
        return format!("trusted, but not answering here (the last {} steps were DFT)", i.steps);
    };
    let near = i.atom.and_then(|a| scene.nuclei.get(a)).is_some_and(|n| face_distance(scene, n.pos) < d_box(scene.cell.h));
    let s = rule_sentence(rule, i.atom, RowCtx { box_affected: i.box_streak, near_face: near, kept: false });
    format!("trusted, but not answering here: {s} (the last {} steps were DFT)", i.steps)
}
```

(`stuck_sentence` takes the driver's `sim::Scene`, which the Model tab has as `snapshot.dynamics.scene`; the test converts the preset with `to_sim()`.) `log.rs` builds each row's `RowCtx { box_affected: e.box_affected, near_face: row.near_face, kept: e.outcome.is_kept() }` for its rule hover.

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

/// What the classes mean (plan review N8: two separate H₂ molecules are "two-atom").
const CLASS_HELP: &str = "two-atom: every atom has at most one neighbour within the model's reach (two separate H₂ molecules count too); three or more: some atom has two or more";

pub fn lines(m: &crate::sim::ModelStatus, scene: &crate::scene::Scene, st: &crate::ui::UiState) -> Vec<Line> {
    let mut v = Vec::new();
    let boxed = crate::sim::box_blocks_trust(m);
    if let Some((b, n)) = boxed {
        v.push(Line {
            met: false,
            why: true,
            // b counts the box-affected samples in the fit, n the samples held (plan review N6)
            text: format!("Samples clear of the box walls — {b} in the fit touch them ({n} held). {}", crate::ui::box_advice(scene, st)),
            hover: format!("the fit's own check failed: {}", m.untrusted_reason.as_deref().unwrap_or("?")),
        });
    }
    let first = m.trust_checks.iter().position(|c| !c.met);
    for (k, c) in m.trust_checks.iter().enumerate() {
        v.push(Line {
            met: c.met,
            why: boxed.is_none() && Some(k) == first,
            text: super::words::trust_line(c, m.last_audit.as_ref(), m.audit_every),
            hover: format!("value {:.3e}, limit {:.3e}", c.value, c.limit),
        });
    }
    if boxed.is_none() && first.is_none() && m.fitted {
        v.push(Line { met: true, why: false, text: "Samples clear of the box walls".into(), hover: "box-affected samples are not keeping the model untrusted".into() });
    }
    for l in &m.class_loo {
        v.push(Line {
            met: l.trusted,
            why: false,
            text: super::words::class_line(l),
            hover: format!("{} samples, LOO force RMS {:.2}× / max {:.2}× the bound, energy {:.2}× the tier; {CLASS_HELP}", l.samples, l.force_rms, l.force_max, l.energy_max),
        });
    }
    v
}
```

`show(ui, st, input)` draws each line with ✓ (GREEN) / ✗ (AMBER), "this is why" as an amber pill on the `why` line, and the hover. `ui/mod.rs` `trust()`: `TrustView` gains `detail: Option<String>`. The `Untrusted` variants use `lines(…).into_iter().find(|l| l.why)` for their text, `format!("no — {}", line.text)`, with `detail = Some(old reason)`. The box variant keeps `BOX_TRUST_TIP` as `tip`. `Stuck` uses `stuck_sentence` when `stuck_info` is present (the old string as `detail`). `trust()` needs the scene and `UiState` for `box_advice`: change it to `trust(m, scene, st)` and update its callers (dock strip dot, toolbar, Model tab, and the M6 tests around `ui/tests.rs:1649`, which call `trust(Some(&…))`), passing `input.scene`/`st`. The Model tab and the toolbar show `detail` on hover after the existing tip.

`crates/app/src/worker/model.rs`: `box_blocks_trust` keys on the first unmet check's key instead of the refusal string's prefix (plan review S15). It blames the box only for a condition of the fit the box samples are in. It does not blame the box for a count condition, which Sample more cures, or for a failed audit, which is judged against a fresh teacher answer that is not box-affected:

```rust
pub fn box_blocks_trust(m: &ModelStatus) -> Option<(usize, usize)> {
    use sim::activity::TrustKey;
    let in_fit = m.box_affected.saturating_sub(m.box_excluded);
    let floor = m.trust_floor?;
    let clean = m.samples.saturating_sub(m.box_affected + m.quarantined + m.superseded);
    let fit_condition = m
        .trust_checks
        .iter()
        .find(|c| !c.met)
        .is_some_and(|c| matches!(c.key, TrustKey::EnergyResidual | TrustKey::ForceResidual | TrustKey::ForceNoise));
    (m.fitted && !m.trusted && fit_condition && in_fit > 0 && clean < floor).then_some((in_fit, m.samples))
}
```

Rewrite its doc's "**Not on a count refusal**" paragraph to name the keys, and delete `COUNT_REFUSALS` (nothing else reads it; Task 3's comment in `trust.rs` already says it was replaced).

- [ ] **Step 4: Run** `cargo test -p app --release ui::` → PASS. Update the M6 box tests at `ui/tests.rs:1649` (`the_trust_readout_names_the_box_when_it_keeps_the_model_untrusted`) to the list `box_blocks_trust` now reads: the fixture gains `trust_checks: checks_failing(sim::activity::TrustKey::EnergyResidual)`; the count loop iterates `[("n_samples 40 < q_prime 59", K::Samples), ("residual_dof_e 20.0 < 30", K::ResidualDofE), ("residual_dof_f 12.5 < 30", K::ResidualDofF)]` with `trust_checks: checks_failing(key)` beside the reason. Its text assertions change only where they assert the old sentence: the box variant now starts with "no — Samples clear of the box walls", and a count refusal no longer equals `format!("no — {count}")` (its plain line is pinned by `the_first_unmet_check_is_this_is_why`). The `box_blocks_trust` predicate assertions and the `box_advice` assertions stay unchanged. `cargo test -p app --release worker::` → PASS (the worker tests read `box_blocks_trust` through real `SurrogateKs` statuses, which carry the checks since Task 3).

- [ ] **Step 5: Commit**

```bash
git add crates/app/src/ui crates/app/src/worker/model.rs
git commit -o -m "feat(app): plain wording for refusals and a live trust checklist

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- crates/app/src/ui crates/app/src/worker/model.rs
```

---

### Task 9: the training plots, the timeline, replay, measurements, docs **[Opus]**

**Files:**
- Modify: `crates/app/src/ui/activity/plots.rs`, `crates/app/src/ui/activity/mod.rs` (Training and Timeline groups, overhead footer), `crates/app/src/replay.rs` (`Expect::{SolveLog, Counters}`), `crates/app/replays/surrogate_far_drag_recovers.replay`, `README.md`, `CLAUDE.md`, `docs/superpowers/notes/2026-10-14-ui-redesign-checklist.md`
- Test: `crates/app/src/ui/tests.rs`, `crates/app/src/replay.rs` tests

**Interfaces:**
- Consumes: `ActivityState::{fit_points, views, timeline, burst_hist}`, `TrainingViews` (with Task 1b's `PairCurve::force_ratio`, `parity_omitted`, `curves_skipped`), `ModelStatus::draft`.
- Produces: `plots::{training, rmse, parity, pair_curve, histogram, timeline}` (each `fn(ui: &mut egui::Ui, …)`, pure painters; `training(ui, st, target_t, draft)`), and the pure helpers `plots::{log_range, to_screen, band_polygon, curve_y_range, refused_stretches, lane_ratios}` they share.

The plan review changed the pair curve (M5, S3). Where the model would refuse is drawn from `would_answer` and is the dominant encoding: the curve is dashed there, its band is dimmed, and the stretch is labelled with its rule. The 4 px strip is gone. The rule-7 comparison moves to a log-scale uncertainty lane under the curve, which shows σ_epi/N against the tier, rule 6's ratio against 1, and the teacher's floor. The dashed ±2·tier lines in a plot of E are gone (they compared an energy with a width). The y range comes from the answered points and the dots. The captions say what an isolated-pair curve is and is not, and what draft mode changes. The burst histogram reads the session accumulator (S6).

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
            // draft on in the force case, so both pair-curve captions draw
            activity::plots::training(ui, &mut st, Some(2000.0), forces);
            activity::plots::timeline(ui, &st);
        });
        full.textures_delta.clear();
    }
}

/// Plan review M5: the curve's refusal styling comes from the policy's verdict (`refused`, from
/// `would_answer`), stretch by stretch with its rule; the y range from the answered points and
/// the dots only, so a refused extrapolation toward 0.5 Bohr cannot set the scale; the lane
/// compares σ_epi/N with the tier (rule 7) as a ratio, beside rule 6's ratio.
#[test]
fn the_pair_curve_reads_the_policy_verdict() {
    use activity::plots::{curve_y_range, lane_ratios, refused_stretches};
    use sim::activity::TeachRule as R;
    let refused = [Some(R::IsolatedAtom), Some(R::IsolatedAtom), None, None, Some(R::Grade), Some(R::IsolatedAtom)];
    assert_eq!(refused_stretches(&refused), vec![(0, 2, R::IsolatedAtom), (4, 5, R::Grade), (5, 6, R::IsolatedAtom)]);
    assert!(refused_stretches(&[None, None]).is_empty());
    let mut c = fixture_views().curves[0].clone();
    let n = c.r.len();
    c.energy = (0..n).map(|k| if k < 10 { 5.0 } else { -0.17 + 1e-3 * k as f64 }).collect();
    c.refused = (0..n).map(|k| (k < 10).then_some(R::Coordination)).collect();
    c.dots = vec![(1.4, -0.2, sim::activity::SampleState::Fitted)];
    let (lo, hi) = curve_y_range(&c).unwrap();
    assert!(hi < 1.0, "the refused wall at short r does not set the scale: {hi}");
    assert!((lo + 0.2).abs() < 1e-12 && (hi - (-0.17 + 1e-3 * (n - 1) as f64)).abs() < 1e-12, "answered points and dots: {lo} {hi}");
    c.sigma = vec![2e-3; n];
    let (energy, floor) = lane_ratios(&c, 4e-5, 1e-3);
    assert!((energy[0] - 1.0).abs() < 1e-12, "σ_epi/N ÷ tier with N = 2");
    assert!((floor - 0.02).abs() < 1e-12, "s_E ÷ tier: the per-dimer floor 2 s_E over N = 2");
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

Run: `cargo test -p app --release -- ui::tests::the_training_plots ui::tests::the_pair_curve`
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

/// The runs of points where the policy would refuse the dimer: half-open index ranges
/// [start, end) of one rule each (the curve's dashed, labelled stretches; plan review M5).
pub fn refused_stretches(refused: &[Option<sim::activity::TeachRule>]) -> Vec<(usize, usize, sim::activity::TeachRule)> {
    let mut v: Vec<(usize, usize, sim::activity::TeachRule)> = Vec::new();
    for (k, r) in refused.iter().enumerate() {
        match (r, v.last_mut()) {
            (Some(rule), Some(last)) if last.1 == k && last.2 == *rule => last.1 = k + 1,
            (Some(rule), _) => v.push((k, k + 1, *rule)),
            (None, _) => {}
        }
    }
    v
}

/// The curve's y range (Ha): the answered points and the dots, so a refused extrapolation (r
/// toward 0.5 Bohr) does not set the scale (plan review M5); every point when none of those
/// exists; `None` for an empty curve.
pub fn curve_y_range(c: &sim::activity::PairCurve) -> Option<(f64, f64)> {
    let answered = c.energy.iter().zip(&c.refused).filter(|(_, r)| r.is_none()).map(|(e, _)| *e);
    let mut ys: Vec<f64> = answered.chain(c.dots.iter().map(|d| d.1)).filter(|y| y.is_finite()).collect();
    if ys.is_empty() {
        ys = c.energy.iter().copied().filter(|y| y.is_finite()).collect();
    }
    let first = *ys.first()?;
    Some(ys.iter().fold((first, first), |(a, b), &y| (a.min(y), b.max(y))))
}

/// The uncertainty lane's energy series and floor as ratios to the rule-7 tier, for a dimer
/// (N = 2): σ_epi/N ÷ `energy_std_per_atom` per r, and (`dimer_floor`/N) ÷ tier, the teacher's
/// own scatter s_E on the same scale. Rule 6's series is `c.force_ratio` as it is. Both rules
/// refuse above 1.
pub fn lane_ratios(c: &sim::activity::PairCurve, dimer_floor: f64, energy_tier: f64) -> (Vec<f64>, f64) {
    const N: f64 = 2.0;
    (c.sigma.iter().map(|s| s / N / energy_tier).collect(), dimer_floor / N / energy_tier)
}
```

`pub fn training(ui: &mut egui::Ui, st: &mut ActivityState, target_t: Option<f64>, draft: bool)` (`target_t`: the plan's `hot_burst_t`, passed by `activity::show` from `UiState::plan`; `draft`: `ModelStatus::draft`, for the pair-curve caption) and `pub fn timeline(ui: &mut egui::Ui, st: &ActivityState)`. `ActivityState` gains `pub parity_forces: bool` (false) and `pub views_at: Option<Instant>` (set in `absorb` when views arrive). `training` draws, in order and each in a `ui.allocate_exact_size(vec2(w, 150), Sense::hover())` rect with `painter_at`:

- **`rmse`**: two series from `st.fit_points` on a log y (mHa/atom and mHa/Bohr, both ×1e3), x = index; filled circles when trusted at that refit, hollow when not; hover shows "epoch e · n samples · E x mHa/atom · F y mHa/Bohr". Note under it: "one point per refit this session; filled: trusted at the refit (a later failed audit does not change it); a loaded model starts with its file's fit" (plan review N9).
- **`parity`**: a segmented toggle E | F (state `st.parity_forces: bool`, add it to `ActivityState`). E: points (dft, model) ×1e3 mHa/atom, the y = x line, the ±tier band (`views.energy_tier` ×1e3 around y = x as a translucent strip); colour by `SampleKind` (pass `theme::ACCENT`, fallback `AMBER`, live `SPARK_E`); filled when `Fitted`, hollow otherwise; hover the nearest point: "sample i · state · DFT x · model y mHa/atom". F: `parity_force` as 1.5 px dots (×1e3 mHa/Bohr) with "showing n of N components (whole atoms, thinned evenly)" when `force_points_total > parity_force.len()`. Caption: "model vs DFT for the training samples, (E − Σ E_ref)/N with the fit's own reference energies; audits are counted as fallbacks until the model library (sidecar v4) tags them", plus "; k samples omitted (no reference or no prediction)" when `parity_omitted > 0` (plan review S18).
- **`pair_curve`** (one per `views.curves`; a line "k species pairs skipped: a prediction on their grid failed" when `curves_skipped > 0`). The main panel (150 px):
  - the band polygon ±σ_epi (`theme::ACCENT.gamma_multiply(0.25)`);
  - the floor band ±`dimer_floor` (grey, thinner), on hover "the fit's residual scatter s_E (teacher noise plus what this basis cannot fit; more samples of the same kind do not remove it)";
  - the curve (accent, 1.5 px). It is **solid where `refused[k]` is `None` and dashed, with the band dimmed to a third, on every stretch of `refused_stretches`**. Each stretch is labelled at its middle with `words::rule_label(rule, None)`, and its hover reads "the model won't answer this dimer here: " followed by `words::rule_sentence(rule, None, RowCtx::default())`. This is the dominant encoding: where the policy would refuse comes from `would_answer`, never from the band (plan review M5);
  - the dots (filled/hollow by state) and the rug (histogram bins with counts as 3 px ticks);
  - y in mHa over `curve_y_range` padded by 10 %; x in Bohr (Å on hover via `units::bohr_to_angstrom`). Title "H–H" from `units::isotopes::symbol`.

  Under it is **the uncertainty lane** (60 px, log y, the same x). From `lane_ratios`, it draws σ_epi/N ÷ tier labelled "rule 7 (energy)" and `force_ratio` labelled "rule 6 (forces)", each as a polyline. The floor ratio is a grey dashed line labelled "fit scatter s_E" (hover: the sentence above). A line at 1 is labelled "refuses above 1". Hover gives σ_epi/N in mHa/atom and the force ratio at that r. Rule 6 usually fires before rule 7, and the lane shows it. Without a posterior (`sigma` empty), the band and the lane are omitted and the caption says so.

  Caption (spec §9.4, plan review S3): "The model's energy of an isolated A–B pair. It is not a pair potential: in a molecule the model has no per-pair energy. The band, the lane and the dashed stretches describe the isolated pair. Band: ±1σ model (coefficient) uncertainty, what rule 7 judges — not an error bar against DFT. Grey: the fit's residual scatter s_E (teacher noise plus what this basis cannot fit; more samples of the same kind do not remove it). Inside a molecule the model may answer this pair where the curve is dashed, and refuse it where it is solid; when the scene's settings differ from the model's it answers nothing here (rule 3); a non-finite answer counts among the rules. Dashed: where the model would not answer this dimer (rules 4–7 and the fence, with a fresh uncertainty check; rule 0, audits and the check's reuse between steps aside)." When `draft` is on, add: "Draft mode is on: the model answers wherever rules 1–3 pass; dashed shows where it would fall back outside draft mode."
- **`histogram`**: per `views.pairs` the bars over 0..r_cut, caption "pair distances in the current fit" (plan review N12); then burst temperatures from `st.burst_hist` (the session accumulator, `BURST_BIN_K` bins; plan review S6), `target_t` drawn as a dashed line. Caption: "instantaneous kinetic temperature (3N − 3 dof) of each hot-burst geometry solved this session, from its Verlet velocities before the thermostat. Consecutive points of one burst are correlated (CSVR τ = 5 fs at dt = 0.25 fs), and for two atoms the instantaneous T alone spreads by about 82 % (√(2/3)). A loaded model's earlier bursts are unrecorded."
- An age line under the group: "built {s} s ago at epoch {e}, {build_ms:.1} ms" from `views` and `views_at`.

`timeline(ui, st)`: five lanes (fallbacks, audits, refits, quarantines, passes), x = session time over `[first mark, now]`; fallback ticks binned per pixel (one 1 px tick per occupied pixel column, height by count, capped); audit dots coloured by verdict; refit diamonds (hollow untrusted); quarantine squares with the count; pass bars (hatched when cancelled). Hover: the mark's text.

The overhead footer (spec §6) under the tab: `format!("activity bookkeeping {:.3} % of teacher time ({:.1} ms of {:.0} ms){}", 100.0 * a / t, a, t, views.map_or(String::new(), |v| format!("; views {:.1} ms per build", v.build_ms)))`, dim, hidden while `t == 0`.

- [ ] **Step 4: Replay expects**

`replay.rs`: `Expect::SolveLog(usize)` ("at least n solve-log rows since the mark, every one with a rule label and an outcome label"); `Expect::Counters` ("the samples counter equals `ModelStatus::samples`; teaching solves this session ≥ the solve rows (`is_solve`) since the mark"). The mark records `viewer.ui.activity.rows.len()` and `session_solves`. Append to `crates/app/replays/surrogate_far_drag_recovers.replay` after its last `expect`:

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
