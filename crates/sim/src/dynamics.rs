//! The dynamics driver (spec §7): owns the scene, the [`ForceProvider`], the integrator and the
//! optional thermostat; takes commands ([`DynCommand`]) and advances time in budgeted ticks;
//! publishes [`DynSnapshot`]s. Atomic units inside (CLAUDE.md rule 3): `dt` and the thermostat's
//! τ arrive in femtoseconds and the target temperature in kelvin only at the command edge, and
//! the drift readout leaves in Ha/ps. The driver never knows what computes the forces (rule 5),
//! and the speed control is a number of steps per tick, never `dt` (rule 8).
//!
//! # One step (spec §7 order)
//! `Integrator::step` (half-kick → drift → `evaluate` → half-kick) with the tool forces added
//! by its `augment` closure before both kicks → pins → thermostat → pins → bookkeeping. The
//! commands are applied between ticks ([`Dynamics::apply`]), i.e. between steps.
//!
//! # Tool forces (controller rulings R1, R9)
//! `augment` turns the raw provider forces F into the forces the kicks use, in this order:
//! 1. **COM correction** (`SetComCorrection(true)`): F_i −= F̄ = Σ_j F_j / N. Only the *provider*
//!    forces are averaged — the correction exists for the grid's spurious net force (the egg-box
//!    ΣF ≈ 3e-4 Ha/Bohr at h = 0.25); averaging the drag spring too would hand −F_spring/N back
//!    to every nucleus and cancel a one-nucleus drag outright. The external field is untouched
//!    (gravity still moves the centre of mass).
//! 2. **Drag spring, critically damped** (R9), **force-capped** (R12): F_d += F_s − γ v_d on the
//!    dragged nucleus, F_s = k (target − r_d) clamped to |F_s| ≤ [`F_DRAG_MAX`] = 0.1 Ha/Bohr
//!    (direction kept; the damping term is not capped). The cap exists because an interactive
//!    drag moves the target ~1e13× faster than any physical motion: a cursor 10 Bohr away would
//!    otherwise pull with 5 Ha/Bohr and pour Hartrees into one nucleus. 0.1 Ha/Bohr is the
//!    restoring force of a strong bond stretched by a few tenths of a Bohr, so a drag can still
//!    break a bond, slowly; the power it injects is bounded by F_max·|v|. The capped spring is
//!    the gradient of the Huber-type potential U = ½k d² for d ≤ d_c = F_max/k = 0.2 Bohr and
//!    F_max (d − d_c/2) beyond, so the bookkeeping below stays exact for the force applied.
//!    k = [`K_DRAG`] = 0.5 Ha/Bohr², γ = 2√(k m_d) — the critical value, so a dragged
//!    nucleus settles on the target in a few ω⁻¹ (ω = √(k/m), ≈ 2.4 fs for ¹H) instead of
//!    oscillating about it. v_d is the velocity the kick uses (v(t) for the first kick,
//!    v(t + dt/2) for the second: explicit, stable while γ dt/m = 2ω dt < 2).
//! 3. **Pins** (not a force in the user's sense, a constraint): a pinned nucleus's entry is
//!    replaced by −m a_ext, so the kicks give it zero net acceleration; its velocity is zeroed
//!    when it is pinned and after every step, and its position is restored to the pre-step value
//!    (bitwise: (−m a)/m + a need not round to 0). Zeroing the velocity alone would let it creep
//!    by dt² F/2m per step (the first half-kick and the drift happen before any zeroing).
//!
//! The snapshot's `forces` are the RAW provider forces (what the hover readout should show).
//!
//! # Energy bookkeeping and the drift health number
//! `total` = K + E_KS (spec §7) is the NVE invariant only without tools: no field, no spring,
//! no thermostat, no pin edits, no COM correction. The driver therefore also keeps
//! **`energy_conserved`** = E_KS + K + U_spring + U_field − W, with U_spring the capped
//! spring's potential (½ k_drag |target − r_d|² within d_c, linear beyond — "Tool forces"), U_field = −Σ m_i a_ext·r_i, and W the energy the non-Hamiltonian
//! operations have put in:
//! - the thermostat rescale and the pins' velocity zeroing: W += ΔK (Bussi's "effective energy",
//!   J. Chem. Phys. 126, 014101, §IV — conserved by CSVR up to the integration error; the
//!   thermostat preserves the COM velocity (R10), so ΔK is exactly its internal ΔK);
//! - a Maxwell–Boltzmann draw on enabling the thermostat: W += ΔK;
//! - moving, starting or ending a drag, and changing the field: W += ΔU at fixed positions (the
//!   user's work);
//! - the drag damping's work per step, trapezoidal over the two kicks: W += ½(f₁ + f₂)·Δr_d with
//!   f_k = −γ v_d the damping force each kick used;
//! - the COM correction's work per step, trapezoidal: W += ½(c_old + c_new)·Σ_i Δr_i with
//!   c = −F̄ (uniform). Exact when the correction is linear in r (the analytic `Harmonic`
//!   provider: the corrected dynamics is then Hamiltonian with V' = V − (N k/2)|d̄|² and the
//!   trapezoid books exactly ΔV − ΔV'); otherwise O(dt³) per step on a small force.
//!
//! With fixed tools that is a constant of the Verlet flow (to its bounded O(dt²) fluctuation),
//! so the drift — the least-squares slope of `energy_conserved` vs t over the ≤ 2000-step
//! history tail, in Ha/ps, `None` below 100 points ([`health`]) — is the integration-error
//! health number with a fixed drag target, the field and NVT on. **While the drag target
//! moves** the booking of ΔU is exact for H but Verlet conserves its shadow H̃ (for the spring
//! H̃ − H = −(h²/4) U_spring, h = ω dt), so `energy_conserved` picks up ≈ −(h²/4) of the drag
//! work (0.7 % for ¹H at 0.25 fs) and the drift readout shows a slope proportional to the drag
//! power for as long as the user drags; the damping's trapezoidal booking adds an error of the
//! same order. `total` and the history's E_total stay the spec's K + E_KS for the sparkline.
//!
//! - a provider whose energy surface moved under the atoms (a surrogate refit, M5): the
//!   provider reports the change of its energy at the current positions once, as
//!   `EvalStats::refit_jump`, and W += that jump — the same booking as the thermostat's;
//! - the surrogate's force-mismatch work (M5): on a fallback step booked on the model surface
//!   the atoms move on the teacher's forces while `energy_ks` is the model's, so
//!   ΔF = F_teacher − F_model (`EvalStats::model_forces`) is an external force on the model
//!   Hamiltonian; W += ½(ΔF_old + ΔF_new)·Δr per step, trapezoidal like the drag damping, and
//!   `work_mismatch` publishes the running sum. Without it the drift readout would carry the
//!   model's force error (−Δ teacher_gap per fallback step) on top of the integrator's.
//!
//! The history (and W) restart at every discontinuity: a discontinuous edit, `ReplaceScene`,
//! `SetDt` (a new dt is a new shadow Hamiltonian, offset by O(dt²) from the old), and a change
//! of the energy surface an evaluation is on ([`EvalStats::energy_surface`]: teacher ↔ model)
//! that the provider did not announce with a `refit_jump`. `Rewind` does not restart it (K is
//! even in v) and `sim_time` keeps increasing through a rewind: it is the time simulated, not a
//! trajectory coordinate.
//!
//! # A provider whose surface moves (M5 surrogate)
//! [`ForceProvider::surface_epoch`] changes when the provider's surface changed for a reason
//! other than the positions (the worker refitted the surrogate between ticks through
//! [`Dynamics::provider_mut`]). The next `tick` then re-evaluates at the unchanged positions
//! before anything else — like stale forces after an edit, but without a history restart: the
//! evaluation's `refit_jump` is booked into W, so `E_conserved` is continuous to rounding, and
//! the step that follows already kicks with the new surface's forces (a step that mixed the
//! old forces in its first half-kick with the new in its second would inject ½ dt v·ΔF
//! unbooked). The snapshot carries the provider's `source`, `uncertainty` and `teacher_gap`,
//! the epoch, and the fraction of the last ≤ 100 steps the surrogate handed back to its
//! teacher (`fallback_fraction_100`; 0 under a Kohn-Sham provider).
//!
//! # Time control (ruling R2)
//! `SetSpeed(n)`: n steps attempted per tick, clamped to 1..=1000 (default 1). A playing
//! [`Dynamics::tick`] runs min(n, the steps that fit `budget_ms` by the running mean cost per
//! step), and at least one unless the tick already spent a solve re-evaluating stale forces;
//! `wall_ratio` is the measured fs simulated per wall second over the tick (0 when paused or
//! when nothing ran). `Step(n)` pauses and queues n steps that the following ticks run
//! (budgeted the same way, not capped by `speed`). `should_abort` is polled before every step
//! (an abort preempts even the first) and passed into the provider; it ends the tick without
//! pausing.
//!
//! # What should abort a running step (R11)
//! `should_abort` abandons the solve in flight (GPU: latency ≤ one SCF step), so the caller
//! must raise it only for commands that make the running step pointless: **Pause, the
//! discontinuous edits (Move, Add, Remove), ReplaceScene, a backend or grid change, and
//! shutdown.** It must **never** fire for DragStart/DragMove/DragEnd, SetSpeed, Pin, Rewind,
//! SetThermostat, SetExternalField, SetDt or SetComCorrection: those only change the state the
//! *next* step starts from, and a DragMove arrives every frame — aborting on it would cancel
//! every DFT step in flight and freeze the run while the user drags. Queue them and `apply`
//! them after the tick returns.
//!
//! # Failures (Review Focus 3)
//! `Err` from a step: the integrator left the scene as it was before the step; the driver
//! pauses, drops the queued steps, sets `last_error`, and keeps the last good forces, energies,
//! display and history. `ProviderError::Cancelled` is not an error: the tick ends, the run stays
//! playing, a queued step stays queued. `last_error` (provider failures only) is cleared by
//! `Play`, `Step` and an accepted edit (the "try again" actions). A rejected command sets
//! `last_rejection` instead, which the next accepted command clears.
//!
//! # Edits (ruling R8)
//! `AddNucleus` and `RemoveNucleus` are accepted at any time; `MoveNucleusDiscontinuous` and
//! `ReplaceScene` only while paused (else rejected). Move zeroes that nucleus's velocity; Add
//! puts the nucleus in at rest (spec §7, "place = nucleus at rest"); ReplaceScene keeps the new
//! scene's velocities (finite positions, velocities and field and positive masses required) and
//! restarts time, step count, pins and drag. Add, Remove and ReplaceScene call
//! `provider.reset()` (cold); Move calls `provider.discontinuity()` (R13: the warm state is a
//! good start for a teleported existing nucleus, only the extrapolation history goes). Each
//! marks the forces stale; the next `tick` (paused or playing) re-evaluates them before anything
//! else — `apply` never evaluates. Pins are a `Vec<bool>` beside the nuclei, so `RemoveNucleus` shifts
//! them with their nuclei; the drag index is shifted likewise, and removing the dragged nucleus
//! ends the drag. Pin toggles and drags do not reset the provider.
//!
//! # Sampling and the model commands (M5)
//! `StartSampling(plan)` plans a [`Sampler`] over the current scene (`crate::sampling`) and
//! pauses the dynamics; while it runs, Play and Step are refused (`last_rejection`), edits are
//! accepted (the pass keeps sampling the scene it started from), and each `tick` spends its
//! budget on the pass instead of steps — at least one teacher solve per tick (none if the tick
//! already re-evaluated stale forces), more while the running mean cost fits `budget_ms` — so
//! the UI stays live. `should_abort` interrupts the solve in flight; the pass resumes on the
//! next tick. When the pass finishes, or on `CancelSampling` (or a `ReplaceScene`), the
//! teacher is told `discontinuity()` (its warm state is a sample geometry's, not the live
//! one's) and the surrogate is refitted over everything it holds; the refit moves the surface
//! epoch when it changes the reported surface, and the next tick refreshes at the unchanged
//! positions with the jump booked (above). The snapshot's `sampling` carries the progress, and
//! the final state after the pass ends. The isolated-atom references the passes solve are
//! cached across passes (and seeded by a loaded model), keyed by the atom's solve settings.
//! `ForgetModel` cancels a running pass without a refit and drops the model, its samples and
//! references (`SurrogateKs::forget`; the next evaluation is on the teacher surface and the
//! conserved-energy history restarts). `SaveModel(path)` writes the surrogate's full sidecar;
//! `LoadModel(path)` reads one and installs it only if its provenance equals the current
//! scene's (grid, h, σ, tolerance, charge, multiplicity, teacher) — otherwise it is refused with
//! a note naming both, and nothing changes. A corrupt file is refused with the reader's error.
//! All five need a `SurrogateKs` provider and are refused otherwise; they reach it through
//! `ForceProvider::as_any_mut` in one helper, never on the stepping path. **The worker should
//! raise `should_abort` for CancelSampling, ForgetModel and LoadModel** (they make a running
//! sampling solve pointless) and, as for steps, not for anything else.
//! # Thermostat (rulings R7, R10)
//! `SetThermostat(Some((T, τ)))` requires finite T ≥ 0 and finite τ > 0 (else rejected). A
//! running thermostat is retuned in place (its random stream continues); a new one is seeded
//! from the driver's seed and the number of thermostats created so far (one seed reproduces a
//! whole command sequence). If K < 1 % of K̄(dof) — a system at rest — velocities are drawn
//! once from Maxwell–Boltzmann at T so no mode starts frozen (`thermostat` module doc, "Frozen
//! modes"), with a note in the snapshot; the same check runs after a discontinuous edit while
//! the thermostat is on (a replaced or rebuilt scene at rest). The thermostat rescales internal
//! velocities only and preserves the COM velocity (R10), so a drag or the field moves a
//! thermostatted system at the NVE pace. With pins, `Scene::dof` still counts the pinned
//! nuclei's degrees of freedom and the COM the thermostat preserves includes them, so the
//! readout and target are slightly inconsistent (the pinned velocities are re-zeroed and that
//! ΔK is booked; nothing leaks). `Rewind` under the thermostat, or with an active (damped) drag,
//! flips the velocities but does not retrace; it leaves a note saying so.
mod health;
#[cfg(test)]
mod tests;

pub use health::{DRIFT_MIN_POINTS, HISTORY_LEN};

use crate::integrator::Integrator;
use crate::provider::{Display, EvalStats, ForceProvider, ProviderError, Source};
use crate::sampling::{ReferenceCache, Sampler, SamplingPlan, SamplingProgress};
use crate::scene::{neumaier, Cell, Nucleus, Scene};
use crate::surrogate_ks::SurrogateKs;
use crate::thermostat::Csvr;
use health::{History, Point};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use surrogate::Provenance;
use std::time::Instant;

/// Drag spring constant (Hartree/Bohr²).
pub const K_DRAG: f64 = 0.5;
/// `SetDt` above this (fs) leaves a warning in the snapshot (spec §7: H–X stretches ~10 fs).
pub const DT_WARN_FS: f64 = 0.5;
/// `SetSpeed` range: steps attempted per tick.
pub const MAX_SPEED: u32 = 1000;
/// Cap on the drag spring's force (Hartree/Bohr, R12; module doc, "Tool forces").
pub const F_DRAG_MAX: f64 = 0.1;
/// Notes kept for the snapshot (newest last).
const NOTES_KEPT: usize = 8;
/// K below this fraction of K̄ counts as "at rest" when the thermostat is switched on (R7).
const AT_REST_FRACTION: f64 = 0.01;
/// Steps the fallback fraction is computed over.
const FALLBACK_WINDOW: usize = 100;

/// What the UI can ask of the driver. Units at this edge: fs (`SetDt`, the thermostat's τ), K
/// (the thermostat's target), Bohr (positions, targets), a.u. acceleration (the field). Which
/// commands should abort a running step: module doc, "What should abort a running step".
#[derive(Clone, Debug)]
pub enum DynCommand {
    Play,
    Pause,
    /// Pause and queue n steps.
    Step(usize),
    /// Steps attempted per tick, clamped to 1..=[`MAX_SPEED`] (ruling R2; the brief's `f64`
    /// "real-time scale" has no meaning when one step costs a DFT solve).
    SetSpeed(u32),
    /// The time step in fs: finite and > 0, warning above [`DT_WARN_FS`].
    SetDt(f64),
    /// v ↦ −v (reversible in NVE without a drag only).
    Rewind,
    /// `Some((T in K, τ in fs))` on (or retune), `None` off (NVE).
    SetThermostat(Option<(f64, f64)>),
    /// Uniform acceleration on every nucleus (Bohr/a.u.², the gravity hook; not an electric
    /// field — `Scene::external_field`).
    SetExternalField([f64; 3]),
    DragStart { index: usize, target: [f64; 3] },
    /// Moves the current drag's target; ignored without a drag.
    DragMove { target: [f64; 3] },
    DragEnd,
    Pin(usize, bool),
    /// Paused only: a new scene, time restarts.
    ReplaceScene(Scene),
    /// Paused only: teleport, velocity zeroed, provider reset.
    MoveNucleusDiscontinuous { index: usize, pos: [f64; 3] },
    /// Any time (R8): added at rest, unpinned, provider reset.
    AddNucleus(Nucleus),
    /// Any time (R8): pins and drag follow their nuclei, provider reset.
    RemoveNucleus(usize),
    SetComCorrection(bool),
    /// Start a sampling pass over the current scene (module doc, "Sampling"): pauses the
    /// dynamics; Play and Step are refused until it finishes or is cancelled. Surrogate only.
    StartSampling(SamplingPlan),
    /// End the running pass early (no-op without one); the samples so far are refitted.
    CancelSampling,
    /// Drop the surrogate's model, samples and references (a running pass is cancelled first).
    ForgetModel,
    /// Write the surrogate's `.model` sidecar (`surrogate::sidecar`).
    SaveModel(PathBuf),
    /// Read a sidecar and install it — refused (with a note) if its provenance is not the
    /// current scene's, or while a pass is running.
    LoadModel(PathBuf),
}

/// An immutable picture of the driver for the UI. Atomic units unless stated. `Default` is an
/// empty, paused placeholder for before the first snapshot (it describes no real scene).
#[derive(Clone, Debug)]
pub struct DynSnapshot {
    pub scene: Scene,
    /// RAW provider forces at `scene`'s positions (Hartree/Bohr); empty when `forces_stale`.
    pub forces: Vec<[f64; 3]>,
    /// True after a discontinuous edit until the next tick re-evaluates (then `energy_ks`,
    /// `eval` and `display` are the pre-edit values).
    pub forces_stale: bool,
    /// The provider's energy at the current positions (Hartree).
    pub energy_ks: f64,
    /// Σ ½ m v² (Hartree, lab frame).
    pub kinetic: f64,
    /// `kinetic + energy_ks` — spec §7's total energy.
    pub total: f64,
    /// The invariant the drift is fitted on (module doc): total + U_spring + U_field − W.
    pub energy_conserved: f64,
    /// Kelvin (`Scene::temperature`).
    pub temperature: f64,
    pub sim_time: f64,
    pub step: u64,
    /// The time step (atomic units); `sim_time_fs`/`dt_fs` convert.
    pub dt: f64,
    /// Slope of `energy_conserved` over the history tail, Ha/ps; `None` below 100 points.
    pub drift_per_ps: Option<f64>,
    pub steps_last_tick: usize,
    /// Steps still queued by `Step`.
    pub pending_steps: usize,
    /// fs simulated per wall second over the last tick (0 when paused or idle).
    pub wall_ratio: f64,
    pub paused: bool,
    pub speed: u32,
    /// Mean wall cost of a step (ms, exponential moving average; 0 before the first step).
    pub ms_per_step: f64,
    /// The last provider failure (pauses the run); cleared by Play, Step, an accepted edit.
    pub last_error: Option<String>,
    /// The last rejected command; cleared by the next accepted command.
    pub last_rejection: Option<String>,
    /// Present while dt > [`DT_WARN_FS`].
    pub dt_warning: Option<String>,
    /// Informational lines (thermostat start, irreversible rewind), newest last, ≤ 8.
    pub notes: Vec<String>,
    /// Notes ever written (monotone): a change means `notes` has a new last entry.
    pub notes_total: u64,
    /// Monotone state counter: bumped by every `apply`, step and evaluation.
    pub seq: u64,
    /// The latest evaluation's cost and diagnostics.
    pub eval: EvalStats,
    /// The latest evaluation's density payload (ruling R5), opaque to `sim`.
    pub display: Option<Display>,
    /// `(t, E_total, T)` per step, oldest first, ≤ 2000 (a.u., Hartree, K); shared, so cloning a
    /// snapshot is cheap.
    pub history: Arc<[(f64, f64, f64)]>,
    /// `(T in K, τ in fs)` when the thermostat is on.
    pub thermostat: Option<(f64, f64)>,
    pub com_correction: bool,
    /// One flag per nucleus.
    pub pins: Vec<bool>,
    /// `(index, target)` of the active drag.
    pub drag: Option<(usize, [f64; 3])>,
    pub provider: &'static str,
    /// Who produced the latest evaluation (`eval.source`; `Dft` for a Kohn-Sham provider).
    pub source: Source,
    /// The surrogate's per-component force uncertainty on the latest evaluation (Ha/Bohr).
    pub uncertainty: Option<f64>,
    /// E_teacher − E_model on a fallback step booked on the model surface (Ha).
    pub teacher_gap: Option<f64>,
    /// The provider's surface epoch (`ForceProvider::surface_epoch`; 0 for a fixed surface).
    pub model_epoch: u64,
    /// Fraction of the last ≤ 100 steps whose evaluation was a `Source::Fallback` (0 before
    /// the first step and under a provider that never falls back).
    pub fallback_fraction_100: f64,
    /// Σ of the surrogate force-mismatch work booked into W since the history (re)started
    /// (Hartree; module doc, "Energy bookkeeping"). 0 under a Kohn-Sham provider.
    pub work_mismatch: f64,
    /// The running sampling pass's progress, or the last pass's final state (`finished`,
    /// possibly `cancelled`) until the next pass or `ForgetModel`; `None` before any.
    pub sampling: Option<SamplingProgress>,
}

impl DynSnapshot {
    pub fn sim_time_fs(&self) -> f64 {
        units::au_time_to_fs(self.sim_time)
    }

    pub fn dt_fs(&self) -> f64 {
        units::au_time_to_fs(self.dt)
    }
}

impl Default for DynSnapshot {
    fn default() -> DynSnapshot {
        DynSnapshot {
            scene: Scene { nuclei: Vec::new(), cell: Cell { n: 0, h: 0.0 }, charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3] },
            forces: Vec::new(),
            forces_stale: true,
            energy_ks: 0.0,
            kinetic: 0.0,
            total: 0.0,
            energy_conserved: 0.0,
            temperature: 0.0,
            sim_time: 0.0,
            step: 0,
            dt: 0.0,
            drift_per_ps: None,
            steps_last_tick: 0,
            pending_steps: 0,
            wall_ratio: 0.0,
            paused: true,
            speed: 1,
            ms_per_step: 0.0,
            last_error: None,
            last_rejection: None,
            dt_warning: None,
            notes: Vec::new(),
            notes_total: 0,
            seq: 0,
            eval: EvalStats::default(),
            display: None,
            history: Arc::from(Vec::new()),
            thermostat: None,
            com_correction: false,
            pins: Vec::new(),
            drag: None,
            provider: "",
            source: Source::Dft,
            uncertainty: None,
            teacher_gap: None,
            model_epoch: 0,
            fallback_fraction_100: 0.0,
            work_mismatch: 0.0,
            sampling: None,
        }
    }
}

/// The driver (module doc). Single-threaded: the app's worker owns it and feeds it commands.
pub struct Dynamics {
    scene: Scene,
    provider: Box<dyn ForceProvider>,
    integrator: Integrator,
    thermostat: Option<Csvr>,
    /// `(T, τ fs)` as commanded (the `Csvr` holds τ in a.u.).
    thermostat_params: Option<(f64, f64)>,
    seed: u64,
    thermostats_created: u64,
    /// RAW provider forces at the current positions; `None` = stale (re-evaluate first).
    forces: Option<Vec<[f64; 3]>>,
    /// A stale re-evaluation failed: paused ticks do not retry it until Play, Step or an edit.
    stale_eval_failed: bool,
    /// The latest evaluation's energy, stats and display (kept through failures).
    energy_ks: f64,
    eval: EvalStats,
    display: Option<Display>,
    pins: Vec<bool>,
    drag: Option<(usize, [f64; 3])>,
    /// Drag damping as a fraction of critical (1.0; the tests set 0 for the undamped maths).
    drag_damping: f64,
    /// The spring's force cap (`F_DRAG_MAX`; the linear-spring tests set ∞).
    drag_force_max: f64,
    com_correction: bool,
    speed: u32,
    paused: bool,
    pending_steps: usize,
    sim_time: f64,
    step: u64,
    /// Energy put in by non-Hamiltonian operations (module doc).
    work: f64,
    history: History,
    /// `provider.surface_epoch()` at the last evaluation; a change makes the forces stale.
    last_epoch: u64,
    /// The pending stale evaluation is a surface refresh (module doc): book the jump instead of
    /// restarting the history.
    surface_refresh: bool,
    /// Whether each of the last ≤ 100 steps was a fallback.
    fallbacks: VecDeque<bool>,
    /// The part of W that is force-mismatch work (module doc).
    work_mismatch: f64,
    steps_last_tick: usize,
    wall_ratio: f64,
    ms_per_step: f64,
    last_error: Option<String>,
    last_rejection: Option<String>,
    notes: Vec<String>,
    notes_total: u64,
    seq: u64,
    /// The running sampling pass (module doc, "Sampling").
    sampler: Option<Sampler>,
    /// Its progress, kept after it ends (the snapshot's `sampling`).
    sampling: Option<SamplingProgress>,
    /// Isolated-atom energies solved by earlier passes or carried by a loaded model.
    ref_cache: ReferenceCache,
    /// Mean wall cost of one sampling solve (ms, moving average).
    sample_ms: f64,
}

/// The surrogate behind the provider, for the model commands only (module doc, "Sampling"):
/// the one place the driver looks behind the boundary, and never on the stepping path.
fn surrogate_mut(provider: &mut Box<dyn ForceProvider>) -> Option<&mut SurrogateKs> {
    provider.as_any_mut()?.downcast_mut::<SurrogateKs>()
}

/// A provenance for messages: everything a load compares.
fn describe_provenance(p: &Provenance) -> String {
    format!(
        "{}³ h {} σ {}, charge {}, multiplicity {}, teacher '{}' at tol {:e}",
        p.grid_n,
        p.h,
        p.sigma,
        p.charge,
        p.multiplicity.map_or("lowest".to_string(), |m| m.to_string()),
        p.teacher,
        p.tol
    )
}

impl Dynamics {
    /// A paused driver at `dt_fs` (finite, > 0 — panics otherwise; `SetDt` validates live
    /// edits instead). `seed` seeds every thermostat this driver creates. Evaluates the forces
    /// once (cold, uncancellable) so the first step has them; if that fails, `last_error` says
    /// why and the next Play or Step tries again.
    pub fn new(scene: Scene, provider: Box<dyn ForceProvider>, dt_fs: f64, seed: u64) -> Dynamics {
        assert!(dt_fs.is_finite() && dt_fs > 0.0, "Dynamics::new: dt = {dt_fs} fs");
        let n = scene.nuclei.len();
        let mut d = Dynamics {
            scene,
            provider,
            integrator: Integrator::new_fs(dt_fs),
            thermostat: None,
            thermostat_params: None,
            seed,
            thermostats_created: 0,
            forces: None,
            stale_eval_failed: false,
            energy_ks: 0.0,
            eval: EvalStats::default(),
            display: None,
            pins: vec![false; n],
            drag: None,
            drag_damping: 1.0,
            drag_force_max: F_DRAG_MAX,
            com_correction: false,
            speed: 1,
            paused: true,
            pending_steps: 0,
            sim_time: 0.0,
            step: 0,
            work: 0.0,
            history: History::default(),
            last_epoch: 0,
            surface_refresh: false,
            fallbacks: VecDeque::with_capacity(FALLBACK_WINDOW),
            work_mismatch: 0.0,
            steps_last_tick: 0,
            wall_ratio: 0.0,
            ms_per_step: 0.0,
            last_error: None,
            last_rejection: None,
            notes: Vec::new(),
            notes_total: 0,
            seq: 0,
            sampler: None,
            sampling: None,
            ref_cache: ReferenceCache::default(),
            sample_ms: 0.0,
        };
        d.provider.reset();
        d.last_epoch = d.provider.surface_epoch();
        if let Err(e) = d.evaluate(&mut || false) {
            d.stale_eval_failed = true;
            d.last_error = Some(format!("{}: {e}", d.provider.name()));
        }
        d
    }

    /// Applies one command between steps (module doc for each). Never evaluates and never
    /// panics on bad input: a rejected command sets `last_rejection` and changes nothing; an
    /// accepted one clears it.
    pub fn apply(&mut self, cmd: DynCommand) {
        self.seq += 1;
        self.last_rejection = None;
        match cmd {
            DynCommand::Play | DynCommand::Step(_) if self.sampler.is_some() => {
                return self.reject("a sampling pass is running: the dynamics stay paused until it finishes or is cancelled".into());
            }
            DynCommand::Play => {
                self.paused = false;
                self.pending_steps = 0;
                self.last_error = None;
                self.stale_eval_failed = false;
            }
            DynCommand::Pause => {
                self.paused = true;
                self.pending_steps = 0;
            }
            DynCommand::Step(n) => {
                self.paused = true;
                self.pending_steps = self.pending_steps.saturating_add(n);
                self.last_error = None;
                self.stale_eval_failed = false;
            }
            DynCommand::SetSpeed(n) => self.speed = n.clamp(1, MAX_SPEED),
            DynCommand::SetDt(fs) => {
                if !(fs.is_finite() && fs > 0.0) {
                    return self.reject(format!("time step {fs} fs: must be finite and > 0"));
                }
                self.integrator = Integrator::new_fs(fs);
                self.restart_history();
            }
            DynCommand::Rewind => {
                Integrator::flip_velocities(&mut self.scene);
                if let Some((t, _)) = self.thermostat_params {
                    self.note(format!("rewind under the thermostat ({t} K): velocities flipped, but NVT is not reversible — the trajectory will not retrace"));
                } else if self.drag.is_some() && self.drag_damping > 0.0 {
                    self.note("rewind during a drag: velocities flipped, but the damped drag spring is not reversible — the trajectory will not retrace".into());
                }
            }
            DynCommand::SetThermostat(None) => {
                self.thermostat = None;
                self.thermostat_params = None;
            }
            DynCommand::SetThermostat(Some((t, tau))) => self.set_thermostat(t, tau),
            DynCommand::SetExternalField(a) => {
                if !finite3(a) {
                    return self.reject(format!("external field {a:?}: must be finite"));
                }
                let u0 = self.field_energy();
                self.scene.external_field = a;
                self.work += self.field_energy() - u0;
            }
            DynCommand::DragStart { index, target } => {
                if index >= self.scene.nuclei.len() || !finite3(target) {
                    return self.reject(format!("drag nucleus {index} to {target:?}: no such nucleus or non-finite target"));
                }
                self.set_drag(Some((index, target)));
            }
            DynCommand::DragMove { target } => {
                if let Some((i, _)) = self.drag {
                    if !finite3(target) {
                        return self.reject(format!("drag target {target:?}: must be finite"));
                    }
                    self.set_drag(Some((i, target)));
                }
            }
            DynCommand::DragEnd => self.set_drag(None),
            DynCommand::Pin(i, on) => {
                if i >= self.pins.len() {
                    return self.reject(format!("pin nucleus {i}: there are {}", self.pins.len()));
                }
                self.pins[i] = on;
                let k0 = self.scene.kinetic_energy();
                self.zero_pinned_velocities();
                self.work += self.scene.kinetic_energy() - k0;
            }
            DynCommand::SetComCorrection(on) => self.com_correction = on,
            DynCommand::ReplaceScene(scene) => {
                if !self.paused {
                    return self.reject("pause before replacing the scene".into());
                }
                // a new scene ends a pass over the old one (what it sampled is kept and refitted)
                if let Some(mut sampler) = self.sampler.take() {
                    sampler.cancel();
                    self.finish_sampling(sampler);
                }
                let bad = scene.nuclei.iter().position(|n| !(finite3(n.pos) && finite3(n.vel) && n.mass.is_finite() && n.mass > 0.0));
                if bad.is_some() || !finite3(scene.external_field) {
                    return self.reject(format!("replace scene: nucleus {bad:?} or the field is non-finite, or a mass is not positive"));
                }
                self.pins = vec![false; scene.nuclei.len()];
                self.scene = scene;
                self.drag = None;
                self.sim_time = 0.0;
                self.step = 0;
                self.after_edit(false);
            }
            DynCommand::MoveNucleusDiscontinuous { index, pos } => {
                if !self.paused {
                    return self.reject("pause before moving a nucleus (or drag it while playing)".into());
                }
                if index >= self.scene.nuclei.len() || !finite3(pos) {
                    return self.reject(format!("move nucleus {index} to {pos:?}: no such nucleus or non-finite position"));
                }
                let n = &mut self.scene.nuclei[index];
                n.pos = pos;
                n.vel = [0.0; 3];
                self.after_edit(true);
            }
            DynCommand::AddNucleus(n) => {
                if !(finite3(n.pos) && n.mass.is_finite() && n.mass > 0.0) {
                    return self.reject(format!("add nucleus at {:?} with mass {}: non-finite or non-positive", n.pos, n.mass));
                }
                self.scene.nuclei.push(Nucleus { vel: [0.0; 3], ..n });
                self.pins.push(false);
                self.after_edit(false);
            }
            DynCommand::RemoveNucleus(i) => {
                if i >= self.scene.nuclei.len() {
                    return self.reject(format!("remove nucleus {i}: there are {}", self.scene.nuclei.len()));
                }
                self.scene.nuclei.remove(i);
                self.pins.remove(i);
                self.drag = match self.drag {
                    Some((d, _)) if d == i => None,
                    Some((d, t)) if d > i => Some((d - 1, t)),
                    other => other,
                };
                self.after_edit(false);
            }
            DynCommand::StartSampling(plan) => self.start_sampling(plan),
            DynCommand::CancelSampling => {
                if let Some(mut sampler) = self.sampler.take() {
                    sampler.cancel();
                    self.finish_sampling(sampler);
                }
            }
            DynCommand::ForgetModel => self.forget_model(),
            DynCommand::SaveModel(path) => self.save_model(path),
            DynCommand::LoadModel(path) => self.load_model(path),
        }
    }

    /// Advances time (module doc, "Time control"): re-evaluates stale forces first, then runs
    /// the queued `Step`s or, when playing, up to `speed` steps within `budget_ms` (at least
    /// one, unless the tick already re-evaluated stale forces). `should_abort` is polled before
    /// every step and handed to the provider; an abort ends the tick without pausing.
    ///
    /// **Raise `should_abort` only for commands that make the running step pointless** (R11):
    /// Pause, MoveNucleusDiscontinuous, AddNucleus, RemoveNucleus, ReplaceScene, a backend or
    /// grid change, shutdown. Never for DragStart/DragMove/DragEnd, SetSpeed, Pin, Rewind,
    /// SetThermostat, SetExternalField, SetDt or SetComCorrection — they change only the next
    /// step's starting state, and aborting on a per-frame DragMove would cancel every step.
    pub fn tick(&mut self, budget_ms: f64, should_abort: &mut dyn FnMut() -> bool) -> DynSnapshot {
        let t0 = Instant::now();
        self.steps_last_tick = 0;
        self.wall_ratio = 0.0;
        let want = if self.paused { self.pending_steps } else { self.speed as usize };
        // the provider's surface moved under the atoms (module doc): re-evaluate first, booking
        // the jump — unless an edit already made the forces stale (that path restarts)
        if self.provider.surface_epoch() != self.last_epoch && self.forces.is_some() {
            self.forces = None;
            self.stale_eval_failed = false;
            self.surface_refresh = true;
        }
        // stale forces: re-evaluate (also when paused with nothing queued — the edit's new
        // energy and density are wanted at once) unless that already failed and nothing has
        // asked for a retry since (Play, Step, an edit)
        let mut evaluated = false;
        if self.forces.is_none() && !self.stale_eval_failed {
            if should_abort() {
                return self.snapshot();
            }
            match self.evaluate(should_abort) {
                Ok(()) => evaluated = true,
                Err(ProviderError::Cancelled) => return self.snapshot(),
                Err(e) => {
                    self.stale_eval_failed = true;
                    self.fail(e);
                    return self.snapshot();
                }
            }
        }
        let mut done = 0usize;
        while done < want && self.forces.is_some() {
            if should_abort() {
                break;
            }
            // the "at least one step" guarantee does not apply after a stale re-evaluation:
            // that solve already spent the tick (a cold GPU solve costs ~1 s)
            if (done > 0 || evaluated) && t0.elapsed().as_secs_f64() * 1e3 + self.ms_per_step > budget_ms {
                break;
            }
            let ts = Instant::now();
            match self.step_once(should_abort) {
                Ok(()) => {
                    done += 1;
                    if self.paused {
                        self.pending_steps -= 1;
                    }
                    let ms = ts.elapsed().as_secs_f64() * 1e3;
                    self.ms_per_step = if self.ms_per_step == 0.0 { ms } else { 0.8 * self.ms_per_step + 0.2 * ms };
                }
                Err(ProviderError::Cancelled) => break,
                Err(e) => {
                    self.fail(e);
                    break;
                }
            }
        }
        self.steps_last_tick = done;
        if self.sampler.is_some() {
            self.advance_sampling(t0, budget_ms, evaluated, should_abort);
        }
        let wall = t0.elapsed().as_secs_f64();
        if !self.paused && done > 0 && wall > 0.0 {
            self.wall_ratio = done as f64 * self.integrator.dt_fs() / wall;
        }
        self.snapshot()
    }

    /// The current state (clones the scene and the display handle, builds the history tail).
    pub fn snapshot(&self) -> DynSnapshot {
        let kinetic = self.scene.kinetic_energy();
        let dt_fs = self.integrator.dt_fs();
        DynSnapshot {
            scene: self.scene.clone(),
            forces: self.forces.clone().unwrap_or_default(),
            forces_stale: self.forces.is_none(),
            energy_ks: self.energy_ks,
            kinetic,
            total: self.energy_ks + kinetic,
            energy_conserved: self.energy_conserved(),
            temperature: self.scene.temperature(),
            sim_time: self.sim_time,
            step: self.step,
            dt: self.integrator.dt,
            drift_per_ps: self.history.drift_per_ps(),
            steps_last_tick: self.steps_last_tick,
            pending_steps: self.pending_steps,
            wall_ratio: self.wall_ratio,
            paused: self.paused,
            speed: self.speed,
            ms_per_step: self.ms_per_step,
            last_error: self.last_error.clone(),
            last_rejection: self.last_rejection.clone(),
            // the relative margin keeps a round trip fs → a.u. → fs of exactly 0.5 quiet
            dt_warning: (dt_fs > DT_WARN_FS * (1.0 + 1e-12))
                .then(|| format!("dt = {dt_fs:.3} fs is above {DT_WARN_FS} fs: H–X stretches (~10 fs period) are under-resolved; watch the drift")),
            notes: self.notes.clone(),
            notes_total: self.notes_total,
            seq: self.seq,
            eval: self.eval.clone(),
            display: self.display.clone(),
            history: Arc::from(self.history.tail()),
            thermostat: self.thermostat_params,
            com_correction: self.com_correction,
            pins: self.pins.clone(),
            drag: self.drag,
            provider: self.provider.name(),
            source: self.eval.source,
            uncertainty: self.eval.uncertainty,
            teacher_gap: self.eval.teacher_gap,
            model_epoch: self.provider.surface_epoch(),
            fallback_fraction_100: if self.fallbacks.is_empty() {
                0.0
            } else {
                self.fallbacks.iter().filter(|&&f| f).count() as f64 / self.fallbacks.len() as f64
            },
            work_mismatch: self.work_mismatch,
            sampling: self.sampling.clone(),
        }
    }

    /// The force provider, for what the boundary does not express: the worker refits a
    /// surrogate between ticks through `as_any_mut` (module doc, "A provider whose surface
    /// moves"). Never call `evaluate` through it: the driver's forces would go stale unnoticed.
    pub fn provider_mut(&mut self) -> &mut dyn ForceProvider {
        self.provider.as_mut()
    }

    // ---- sampling and the model commands (module doc, "Sampling") ----

    fn no_surrogate(&mut self, what: &str) {
        let name = self.provider.name();
        self.reject(format!("{what}: the provider '{name}' has no surrogate model (choose the estimated solver)"));
    }

    fn start_sampling(&mut self, plan: SamplingPlan) {
        if surrogate_mut(&mut self.provider).is_none() {
            return self.no_surrogate("start sampling");
        }
        if self.sampler.is_some() {
            return self.reject("start sampling: a pass is already running (cancel it first)".into());
        }
        let sampler = match Sampler::new(&self.scene, &plan) {
            Ok(s) => s.with_references(std::mem::take(&mut self.ref_cache)),
            Err(e) => return self.reject(format!("sampling plan: {e}")),
        };
        self.paused = true;
        self.pending_steps = 0;
        let p = sampler.progress();
        self.note(format!(
            "sampling pass started: {} jobs ({} pair scans), the dynamics are paused until it ends",
            p.total,
            sampler.scans().len()
        ));
        self.sampling = Some(p);
        self.sampler = Some(sampler);
    }

    /// Spends the tick's remaining budget on the pass (module doc).
    fn advance_sampling(&mut self, t0: Instant, budget_ms: f64, evaluated: bool, should_abort: &mut dyn FnMut() -> bool) {
        let Some(mut sampler) = self.sampler.take() else { return };
        let mut solves = 0usize;
        let end: Option<Result<(), String>> = loop {
            if should_abort() {
                break None;
            }
            if (solves > 0 || evaluated) && t0.elapsed().as_secs_f64() * 1e3 + self.sample_ms > budget_ms {
                break None;
            }
            let Some(s) = surrogate_mut(&mut self.provider) else {
                break Some(Err("the provider is no longer a surrogate".into()));
            };
            let ts = Instant::now();
            match sampler.step_own_teacher(s, should_abort) {
                Ok(Some(p)) => {
                    solves += 1;
                    let ms = ts.elapsed().as_secs_f64() * 1e3;
                    self.sample_ms = if self.sample_ms == 0.0 { ms } else { 0.8 * self.sample_ms + 0.2 * ms };
                    self.sampling = Some(p);
                    self.seq += 1;
                }
                Ok(None) => break Some(Ok(())),
                Err(ProviderError::Cancelled) => break None,
                Err(e) => break Some(Err(e.to_string())),
            }
        };
        for n in sampler.take_notes() {
            self.note(n);
        }
        match end {
            None => self.sampler = Some(sampler),
            Some(Ok(())) => self.finish_sampling(sampler),
            Some(Err(e)) => {
                self.last_error = Some(format!("sampling: {e}"));
                self.finish_sampling(sampler);
            }
        }
    }

    /// After a pass (complete or cancelled): the cache back, the teacher told its state is a
    /// sample geometry's, the surrogate refitted over everything it holds.
    fn finish_sampling(&mut self, mut sampler: Sampler) {
        let progress = sampler.progress();
        self.ref_cache = sampler.take_references();
        for n in sampler.take_notes() {
            self.note(n);
        }
        self.sampling = Some(progress.clone());
        self.seq += 1;
        let added = sampler.samples_added();
        let how = if progress.cancelled { "cancelled" } else { "finished" };
        let msg = match surrogate_mut(&mut self.provider) {
            None => format!("sampling pass {how}: {added} samples added"),
            Some(s) => {
                s.teacher_mut().discontinuity();
                if s.samples() == 0 {
                    format!("sampling pass {how}: no samples to fit")
                } else {
                    match s.refit() {
                        Ok(r) => format!(
                            "sampling pass {how}: {added} samples added ({} skipped, {} failed); refit on {} samples, rmse {:.1e} Ha/atom, {:.1e} Ha/Bohr — {}",
                            progress.skipped,
                            progress.failed,
                            r.n_samples,
                            r.rmse_energy_per_atom,
                            r.rmse_force,
                            match s.untrusted_reason() {
                                None => "trusted".to_string(),
                                Some(why) => format!("untrusted ({why})"),
                            }
                        ),
                        Err(e) => format!("sampling pass {how}: {added} samples added; the refit failed ({e})"),
                    }
                }
            }
        };
        self.note(msg);
    }

    fn forget_model(&mut self) {
        let Some(s) = surrogate_mut(&mut self.provider) else { return self.no_surrogate("forget model") };
        let running = self.sampler.take();
        if running.is_some() {
            s.teacher_mut().discontinuity();
        }
        let n = s.samples();
        s.forget();
        if let Some(mut sampler) = running {
            self.ref_cache = sampler.take_references();
        }
        self.sampling = None;
        self.note(format!("model forgotten ({n} samples dropped): the teacher answers every step"));
    }

    fn save_model(&mut self, path: PathBuf) {
        let Some(s) = surrogate_mut(&mut self.provider) else { return self.no_surrogate("save model") };
        let (n, pending) = (s.samples(), s.pending_refit());
        match s.save_sidecar(&path) {
            Ok(()) => self.note(format!(
                "model saved to {} ({n} samples{})",
                path.display(),
                if pending { ", some taken since the last refit" } else { "" }
            )),
            Err(e) => self.reject(format!("save model to {}: {e}", path.display())),
        }
    }

    fn load_model(&mut self, path: PathBuf) {
        if surrogate_mut(&mut self.provider).is_none() {
            return self.no_surrogate("load model");
        }
        if self.sampler.is_some() {
            return self.reject("load model: cancel the running sampling pass first".into());
        }
        let sc = match surrogate::sidecar::read(&path) {
            Ok(sc) => sc,
            Err(e) => return self.reject(format!("load model: {e}")),
        };
        let s = surrogate_mut(&mut self.provider).expect("checked above");
        let want = s.provenance_for(&self.scene);
        if sc.model.provenance != want {
            let msg = format!(
                "model {} refused: it was trained at {} but the scene is solved at {} — another Kohn-Sham surface",
                path.display(),
                describe_provenance(&sc.model.provenance),
                describe_provenance(&want)
            );
            self.note(msg.clone());
            return self.reject(msg);
        }
        let atom = ReferenceCache::atom_provenance(&sc.model.provenance);
        let refs: Vec<(u8, f64)> = sc.model.basis.species.iter().copied().zip(sc.model.ref_energy.iter().copied()).collect();
        let n = sc.samples.len();
        match s.install_sidecar(sc) {
            Ok(()) => {
                let trusted = s.trusted();
                for (z, e) in refs {
                    self.ref_cache.insert(atom.clone(), z, e);
                }
                self.sampling = None;
                self.note(format!("model loaded from {} ({n} samples, {})", path.display(), if trusted { "trusted" } else { "untrusted" }));
            }
            Err(e) => self.reject(format!("load model {}: {e}", path.display())),
        }
    }

    // ---- internals ----

    /// Evaluates at the current positions (stale forces → fresh). After an edit the energy is
    /// a new level and the history restarts; after a surface refresh (module doc) the jump is
    /// booked instead and the history continues.
    fn evaluate(&mut self, should_abort: &mut dyn FnMut() -> bool) -> Result<(), ProviderError> {
        let r = self.provider.evaluate(&self.scene, should_abort)?;
        if r.forces.len() != self.scene.nuclei.len() {
            return Err(ProviderError::Invalid(format!("provider '{}' returned {} forces for {} nuclei", self.provider.name(), r.forces.len(), self.scene.nuclei.len())));
        }
        let refresh = std::mem::take(&mut self.surface_refresh);
        if refresh {
            self.book_surface_change(&r.stats);
        }
        self.forces = Some(r.forces);
        self.energy_ks = r.energy;
        self.eval = r.stats;
        self.display = r.display;
        self.last_epoch = self.provider.surface_epoch();
        self.seq += 1;
        if !refresh {
            self.restart_history();
        }
        Ok(())
    }

    /// The surface bookkeeping between the previous evaluation (`self.eval`) and `new` (module
    /// doc, "Energy bookkeeping"): a reported `refit_jump` goes into W; a change of surface
    /// without one restarts the history.
    fn book_surface_change(&mut self, new: &EvalStats) {
        if let Some(j) = new.refit_jump {
            self.work += j;
        } else if new.energy_surface() != self.eval.energy_surface() {
            self.restart_history();
            self.note(format!("energy surface changed ({:?} → {:?}) without a booked jump: the conserved-energy history restarts",
                              self.eval.energy_surface(), new.energy_surface()));
        }
    }

    /// One step: Verlet with the tool forces, pins, thermostat, pins, bookkeeping.
    fn step_once(&mut self, should_abort: &mut dyn FnMut() -> bool) -> Result<(), ProviderError> {
        let forces_old = self.forces.as_ref().expect("step_once with stale forces");
        let pos_before: Vec<[f64; 3]> = self.scene.nuclei.iter().map(|n| n.pos).collect();
        let tools = Tools { drag: self.drag, damping: self.drag_damping, force_max: self.drag_force_max, com: self.com_correction, pins: &self.pins };
        // the damping force each kick used, for the work booking (only on success)
        let mut damping_used: Vec<[f64; 3]> = Vec::with_capacity(2);
        let mut augment = |s: &Scene, f: &mut Vec<[f64; 3]>| {
            if let Some(fd) = tools.apply(s, f) {
                damping_used.push(fd);
            }
        };
        let (f_new, stats) = self.integrator.step(&mut self.scene, forces_old, self.provider.as_mut(), &mut augment, should_abort)?;
        // the COM correction's work, trapezoidal (module doc)
        if tools.com && !f_new.is_empty() {
            let (c_old, c_new) = (mean(forces_old), mean(&f_new));
            let mut w = 0.0;
            for (n, r0) in self.scene.nuclei.iter().zip(&pos_before) {
                for a in 0..3 {
                    w -= 0.5 * (c_old[a] + c_new[a]) * (n.pos[a] - r0[a]);
                }
            }
            self.work += w;
        }
        // the drag damping's work, trapezoidal over the two kicks (module doc)
        if let (Some((i, _)), [f1, f2]) = (tools.drag, damping_used.as_slice()) {
            let r1 = self.scene.nuclei[i].pos;
            let r0 = pos_before[i];
            self.work += (0..3).map(|a| 0.5 * (f1[a] + f2[a]) * (r1[a] - r0[a])).sum::<f64>();
        }
        // pins are a constraint: restore the (rounding-level) displacement exactly
        for ((n, r0), &p) in self.scene.nuclei.iter_mut().zip(&pos_before).zip(&self.pins) {
            if p {
                n.pos = *r0;
            }
        }
        // the surrogate's force-mismatch work, trapezoidal over the step (module doc)
        let w_mis = mismatch_work(forces_old, self.eval.model_forces.as_deref(), &f_new, stats.eval.model_forces.as_deref(), &self.scene, &pos_before);
        // pins → thermostat → pins; the kinetic energy they change is booked in W
        let k0 = self.scene.kinetic_energy();
        self.zero_pinned_velocities();
        if let Some(th) = &mut self.thermostat {
            th.apply(&mut self.scene, self.integrator.dt);
            self.zero_pinned_velocities();
        }
        self.work += self.scene.kinetic_energy() - k0;
        self.work += w_mis;
        self.work_mismatch += w_mis;
        self.book_surface_change(&stats.eval);
        if self.fallbacks.len() == FALLBACK_WINDOW {
            self.fallbacks.pop_front();
        }
        self.fallbacks.push_back(stats.eval.source == Source::Fallback);
        self.forces = Some(f_new);
        self.energy_ks = stats.energy_ks;
        self.eval = stats.eval;
        self.display = stats.display;
        self.last_epoch = self.provider.surface_epoch();
        self.sim_time += self.integrator.dt;
        self.step += 1;
        self.seq += 1;
        let kinetic = self.scene.kinetic_energy();
        self.history.push(Point {
            t: self.sim_time,
            e_total: self.energy_ks + kinetic,
            e_conserved: self.energy_conserved(),
            temperature: self.scene.temperature(),
        });
        Ok(())
    }

    fn set_thermostat(&mut self, t: f64, tau_fs: f64) {
        if !(t.is_finite() && t >= 0.0 && tau_fs.is_finite() && tau_fs > 0.0) {
            return self.reject(format!("thermostat T = {t} K, τ = {tau_fs} fs: need finite T ≥ 0 and τ > 0"));
        }
        match &mut self.thermostat {
            Some(th) => {
                th.target_t = t;
                th.tau = units::fs_to_au_time(tau_fs);
            }
            None => {
                let seed = self.seed.wrapping_add(self.thermostats_created.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                self.thermostats_created += 1;
                self.thermostat = Some(Csvr::new(t, tau_fs, seed));
            }
        }
        self.thermostat_params = Some((t, tau_fs));
        self.thermalise_if_at_rest();
    }

    /// R7: with the thermostat on and the *internal* K (`Scene::internal_kinetic_energy`, COM
    /// motion removed — the thermostat's and the readout's frame) below 1 % of K̄ (a system
    /// internally at rest, translating or not), draw Maxwell–Boltzmann velocities at the target
    /// about the existing centre-of-mass velocity (N ≥ 2: the draw has zero COM velocity and u is
    /// added back, so a moving molecule keeps moving; the change is booked in W, and it is the
    /// internal ΔK) and leave a note. A lab-frame test would let a drifting, internally cold
    /// molecule through undrawn, with its rotations frozen (final review B-1).
    fn thermalise_if_at_rest(&mut self) {
        let Some(th) = self.thermostat.as_mut() else { return };
        let k_int = self.scene.internal_kinetic_energy();
        if self.scene.nuclei.is_empty() || k_int >= AT_REST_FRACTION * th.target_kinetic(self.scene.dof()) {
            return;
        }
        let k0 = self.scene.kinetic_energy();
        let u = if self.scene.nuclei.len() >= 2 { self.scene.com_velocity() } else { [0.0; 3] };
        th.maxwell_boltzmann(&mut self.scene);
        for n in &mut self.scene.nuclei {
            for (v, du) in n.vel.iter_mut().zip(u) {
                *v += du;
            }
        }
        let t = th.target_t;
        self.zero_pinned_velocities();
        self.work += self.scene.kinetic_energy() - k0;
        self.note(format!("thermostat on: Maxwell–Boltzmann start at {t} K (the system was internally at rest)"));
    }

    fn set_drag(&mut self, drag: Option<(usize, [f64; 3])>) {
        let u0 = self.spring_energy();
        self.drag = drag;
        self.work += self.spring_energy() - u0;
    }

    /// The capped spring's potential (Hartree), 0 without a drag (module doc, "Tool forces").
    fn spring_energy(&self) -> f64 {
        match self.drag {
            Some((i, t)) => spring_potential(t, self.scene.nuclei[i].pos, self.drag_force_max),
            None => 0.0,
        }
    }

    /// −Σ m a·r (Hartree): the potential of the uniform field. Pinned nuclei included (their
    /// term is constant while pinned).
    fn field_energy(&self) -> f64 {
        let a = self.scene.external_field;
        -neumaier(self.scene.nuclei.iter().map(|n| n.mass * (a[0] * n.pos[0] + a[1] * n.pos[1] + a[2] * n.pos[2])))
    }

    fn energy_conserved(&self) -> f64 {
        self.energy_ks + self.scene.kinetic_energy() + self.spring_energy() + self.field_energy() - self.work
    }

    fn zero_pinned_velocities(&mut self) {
        for (n, &p) in self.scene.nuclei.iter_mut().zip(&self.pins) {
            if p {
                n.vel = [0.0; 3];
            }
        }
    }

    /// History and W restart: E_conserved is re-referenced to the current state.
    fn restart_history(&mut self) {
        self.history.clear();
        self.work = 0.0;
        self.work_mismatch = 0.0;
    }

    /// After an accepted discontinuous edit: the provider is told (R13: a teleport of existing
    /// nuclei, `keep_warm`, calls `ForceProvider::discontinuity` — warm start kept, no
    /// extrapolation; Add / Remove / ReplaceScene call `reset` — cold), stale forces, history
    /// restart, and (thermostat on) the R7 at-rest check on the edited scene.
    fn after_edit(&mut self, keep_warm: bool) {
        if keep_warm {
            self.provider.discontinuity();
        } else {
            self.provider.reset();
        }
        self.forces = None;
        self.stale_eval_failed = false;
        self.surface_refresh = false;
        self.restart_history();
        self.last_error = None;
        self.thermalise_if_at_rest();
    }

    fn fail(&mut self, e: ProviderError) {
        self.paused = true;
        self.pending_steps = 0;
        self.last_error = Some(format!("{}: {e}", self.provider.name()));
    }

    fn reject(&mut self, msg: String) {
        self.last_rejection = Some(msg);
    }

    fn note(&mut self, msg: String) {
        if self.notes.len() == NOTES_KEPT {
            self.notes.remove(0);
        }
        self.notes.push(format!("step {}: {msg}", self.step));
        self.notes_total += 1;
    }
}

/// The drag spring's force k (t − r), clamped to |F| ≤ `force_max` with its direction kept.
fn spring_force(t: [f64; 3], r: [f64; 3], force_max: f64) -> [f64; 3] {
    let d: [f64; 3] = std::array::from_fn(|a| t[a] - r[a]);
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let scale = if K_DRAG * len > force_max { force_max / len } else { K_DRAG };
    d.map(|x| scale * x)
}

/// The potential whose gradient is [`spring_force`]: ½ k d² for d ≤ d_c = F_max / k, and
/// F_max (d − d_c / 2) beyond (continuous with its derivative at d_c).
fn spring_potential(t: [f64; 3], r: [f64; 3], force_max: f64) -> f64 {
    let d = ((t[0] - r[0]).powi(2) + (t[1] - r[1]).powi(2) + (t[2] - r[2]).powi(2)).sqrt();
    let dc = force_max / K_DRAG;
    if d <= dc { 0.5 * K_DRAG * d * d } else { force_max * (d - 0.5 * dc) }
}

fn finite3(x: [f64; 3]) -> bool {
    x.iter().all(|v| v.is_finite())
}

/// Σ_i ½(ΔF_old,i + ΔF_new,i)·Δr_i with ΔF = raw provider force − model force (zero for an
/// evaluation without model forces: a surrogate answer has ΔF = 0, a teacher-surface step
/// books nothing). Module doc, "Energy bookkeeping".
fn mismatch_work(f_old: &[[f64; 3]], m_old: Option<&[[f64; 3]]>, f_new: &[[f64; 3]], m_new: Option<&[[f64; 3]]>, scene: &Scene, pos_before: &[[f64; 3]]) -> f64 {
    if m_old.is_none() && m_new.is_none() {
        return 0.0;
    }
    let mut w = 0.0;
    for (i, (n, r0)) in scene.nuclei.iter().zip(pos_before).enumerate() {
        for a in 0..3 {
            let d_old = m_old.map_or(0.0, |m| f_old[i][a] - m[i][a]);
            let d_new = m_new.map_or(0.0, |m| f_new[i][a] - m[i][a]);
            w += 0.5 * (d_old + d_new) * (n.pos[a] - r0[a]);
        }
    }
    w
}

/// Σ F / N (zero for no forces).
fn mean(f: &[[f64; 3]]) -> [f64; 3] {
    if f.is_empty() {
        return [0.0; 3];
    }
    std::array::from_fn(|a| neumaier(f.iter().map(|x| x[a])) / f.len() as f64)
}

/// The tool settings one step's `augment` uses (copied before the step: the scene the kicks see
/// is the integrator's working copy).
#[derive(Clone, Copy)]
struct Tools<'a> {
    drag: Option<(usize, [f64; 3])>,
    /// Fraction of critical damping.
    damping: f64,
    /// Cap on |k (target − r)| (Hartree/Bohr).
    force_max: f64,
    com: bool,
    pins: &'a [bool],
}

impl Tools<'_> {
    /// The `augment` of every step (module doc, "Tool forces"): COM correction on the provider
    /// forces, then the damped drag spring, then the pins' zero-net-acceleration override.
    /// Returns the damping force applied to the dragged nucleus (for the work booking).
    fn apply(&self, scene: &Scene, f: &mut [[f64; 3]]) -> Option<[f64; 3]> {
        if self.com {
            let m = mean(f);
            for x in f.iter_mut() {
                for a in 0..3 {
                    x[a] -= m[a];
                }
            }
        }
        let mut damping = None;
        if let Some((i, t)) = self.drag {
            let n = &scene.nuclei[i];
            let gamma = self.damping * 2.0 * (K_DRAG * n.mass).sqrt();
            let fd: [f64; 3] = std::array::from_fn(|a| -gamma * n.vel[a]);
            let fs = spring_force(t, n.pos, self.force_max);
            for a in 0..3 {
                f[i][a] += fs[a] + fd[a];
            }
            damping = Some(fd);
        }
        let g = scene.external_field;
        for ((x, n), &p) in f.iter_mut().zip(&scene.nuclei).zip(self.pins) {
            if p {
                *x = [-n.mass * g[0], -n.mass * g[1], -n.mass * g[2]];
            }
        }
        damping
    }
}
