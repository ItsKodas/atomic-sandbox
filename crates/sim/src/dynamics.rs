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
//! # Tool forces (controller ruling R1)
//! `augment` turns the raw provider forces F into the forces the kicks use, in this order:
//! 1. **COM correction** (`SetComCorrection(true)`): F_i −= F̄ = Σ_j F_j / N. Only the *provider*
//!    forces are averaged — the correction exists for the grid's spurious net force (the egg-box
//!    ΣF ≈ 3e-4 Ha/Bohr at h = 0.25); averaging the drag spring too would hand −F_spring/N back
//!    to every nucleus and cancel a one-nucleus drag outright. The external field is untouched
//!    (gravity still moves the centre of mass).
//! 2. **Drag spring**: F_d += k_drag (target − r_d), k_drag = [`K_DRAG`] = 0.5 Ha/Bohr², on the
//!    dragged nucleus. An undamped spring: a dragged nucleus oscillates about the target (ω =
//!    √(k_drag/m), period ≈ 37 steps for ¹H at 0.25 fs) until something else damps it.
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
//! **`energy_conserved`** = E_KS + K + U_spring + U_field − W, with U_spring = ½ k_drag
//! |target − r_d|², U_field = −Σ m_i a_ext·r_i, and W the energy the non-Hamiltonian
//! operations have put in:
//! - the thermostat rescale and the pins' velocity zeroing: W += ΔK (Bussi's "effective energy",
//!   J. Chem. Phys. 126, 014101, §IV — conserved by CSVR up to the integration error);
//! - a Maxwell–Boltzmann draw on enabling the thermostat: W += ΔK;
//! - moving, starting or ending a drag, and changing the field: W += ΔU at fixed positions (the
//!   user's work);
//! - the COM correction's work per step, trapezoidal: W += ½(c_old + c_new)·Σ_i Δr_i with
//!   c = −F̄ (uniform). Exact when the correction is linear in r (the analytic `Harmonic`
//!   provider: the corrected dynamics is then Hamiltonian with V' = V − (N k/2)|d̄|² and the
//!   trapezoid books exactly ΔV − ΔV'); otherwise O(dt³) per step on a small force.
//!
//! With fixed tools that is a constant of the Verlet flow (to its bounded O(dt²) fluctuation),
//! so the drift — the least-squares slope of `energy_conserved` vs t over the ≤ 2000-step
//! history tail, in Ha/ps, `None` below 100 points ([`health`]) — is the integration-error
//! health number in every mode, NVT and dragging included. `total` and the history's E_total
//! stay the spec's K + E_KS for the sparkline.
//!
//! The history (and W) restart at every discontinuity: a discontinuous edit, `ReplaceScene`,
//! `SetDt` (a new dt is a new shadow Hamiltonian, offset by O(dt²) from the old). `Rewind` does
//! not restart it (K is even in v) and `sim_time` keeps increasing through a rewind: it is the
//! time simulated, not a trajectory coordinate.
//!
//! # Time control (ruling R2)
//! `SetSpeed(n)`: n steps attempted per tick, clamped to 1..=1000 (default 1). A playing
//! [`Dynamics::tick`] runs min(n, the steps that fit `budget_ms` by the running mean cost per
//! step) and at least one; `wall_ratio` is the measured fs simulated per wall second over the
//! tick (0 when paused or when nothing ran). `Step(n)` pauses and queues n steps that the
//! following ticks run (budgeted the same way, not capped by `speed`). `should_abort` is polled
//! before every step (an abort preempts even the first) and passed into the provider; it ends
//! the tick without pausing.
//!
//! # Failures (Review Focus 3)
//! `Err` from a step: the integrator left the scene as it was before the step; the driver
//! pauses, drops the queued steps, sets `last_error`, and keeps the last good forces, energies,
//! display and history. `ProviderError::Cancelled` is not an error: the tick ends, the run stays
//! playing, a queued step stays queued. `last_error` is also set by a rejected command; it is
//! cleared by `Play`, `Step` and an accepted edit (the "try again" actions).
//!
//! # Edits
//! `MoveNucleusDiscontinuous`, `AddNucleus`, `RemoveNucleus` and `ReplaceScene` are accepted only
//! while paused (else rejected with `last_error`). Move zeroes that nucleus's velocity; Add puts
//! the nucleus in at rest (spec §7, "place = nucleus at rest"); ReplaceScene keeps the new
//! scene's velocities and restarts time, step count, pins and drag. Each calls
//! `provider.reset()` and marks the forces stale; the next `tick` (paused or not) re-evaluates
//! them before anything else — `apply` never evaluates. Pins are a `Vec<bool>` beside the nuclei,
//! so `RemoveNucleus` shifts them with their nuclei; the drag index is shifted likewise, and
//! removing the dragged nucleus ends the drag. Pin toggles and drags do not reset the provider.
//!
//! # Thermostat (ruling R7)
//! `SetThermostat(Some((T, τ)))` requires finite T ≥ 0 and finite τ > 0 (else rejected). A
//! running thermostat is retuned in place (its random stream continues); a new one is seeded
//! from the driver's seed and the number of thermostats created so far (one seed reproduces a
//! whole command sequence). If K < 1 % of K̄(dof) — a system at rest — velocities are drawn
//! once from Maxwell–Boltzmann at T so no mode starts frozen (`thermostat` module doc, "Frozen
//! modes"), with a note in the snapshot. `Rewind` under the thermostat flips the velocities but
//! does not retrace (the rescaling is stochastic); it leaves a note saying so.
mod health;
#[cfg(test)]
mod tests;

pub use health::{DRIFT_MIN_POINTS, HISTORY_LEN};

use crate::integrator::Integrator;
use crate::provider::{Display, EvalStats, ForceProvider, ProviderError};
use crate::scene::{neumaier, Nucleus, Scene};
use crate::thermostat::Csvr;
use health::{History, Point};
use std::time::Instant;

/// Drag spring constant (Hartree/Bohr²).
pub const K_DRAG: f64 = 0.5;
/// `SetDt` above this (fs) leaves a warning in the snapshot (spec §7: H–X stretches ~10 fs).
pub const DT_WARN_FS: f64 = 0.5;
/// `SetSpeed` range: steps attempted per tick.
pub const MAX_SPEED: u32 = 1000;
/// Notes kept for the snapshot (newest last).
const NOTES_KEPT: usize = 8;
/// K below this fraction of K̄ counts as "at rest" when the thermostat is switched on (R7).
const AT_REST_FRACTION: f64 = 0.01;

/// What the UI can ask of the driver. Units at this edge: fs (`SetDt`, the thermostat's τ), K
/// (the thermostat's target), Bohr (positions, targets), a.u. acceleration (the field).
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
    /// v ↦ −v (reversible in NVE only).
    Rewind,
    /// `Some((T in K, τ in fs))` on (or retune), `None` off (NVE).
    SetThermostat(Option<(f64, f64)>),
    /// Uniform acceleration on every nucleus (a.u.; `Scene::external_field`).
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
    /// Paused only: added at rest, unpinned, provider reset.
    AddNucleus(Nucleus),
    /// Paused only: pins and drag follow their nuclei, provider reset.
    RemoveNucleus(usize),
    SetComCorrection(bool),
}

/// An immutable picture of the driver for the UI. Atomic units unless stated.
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
    pub last_error: Option<String>,
    /// Present while dt > [`DT_WARN_FS`].
    pub dt_warning: Option<String>,
    /// Informational lines (thermostat start, NVT rewind), newest last, ≤ 8.
    pub notes: Vec<String>,
    /// The latest evaluation's cost and diagnostics.
    pub eval: EvalStats,
    /// The latest evaluation's density payload (ruling R5), opaque to `sim`.
    pub display: Option<Display>,
    /// `(t, E_total, T)` per step, oldest first, ≤ 2000 (a.u., Hartree, K).
    pub history: Vec<(f64, f64, f64)>,
    /// `(T in K, τ in fs)` when the thermostat is on.
    pub thermostat: Option<(f64, f64)>,
    pub com_correction: bool,
    /// One flag per nucleus.
    pub pins: Vec<bool>,
    /// `(index, target)` of the active drag.
    pub drag: Option<(usize, [f64; 3])>,
    pub provider: &'static str,
}

impl DynSnapshot {
    pub fn sim_time_fs(&self) -> f64 {
        units::au_time_to_fs(self.sim_time)
    }

    pub fn dt_fs(&self) -> f64 {
        units::au_time_to_fs(self.dt)
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
    com_correction: bool,
    speed: u32,
    paused: bool,
    pending_steps: usize,
    sim_time: f64,
    step: u64,
    /// Energy put in by non-Hamiltonian operations (module doc).
    work: f64,
    history: History,
    steps_last_tick: usize,
    wall_ratio: f64,
    ms_per_step: f64,
    last_error: Option<String>,
    notes: Vec<String>,
}

impl Dynamics {
    /// A paused driver at `dt_fs` (finite, > 0 — panics otherwise; `SetDt` validates live
    /// edits instead). `seed` seeds every thermostat this driver creates. Evaluates the forces
    /// once (cold, uncancellable) so the first step has them; if that fails, `last_error` says
    /// why and the next tick tries again.
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
            com_correction: false,
            speed: 1,
            paused: true,
            pending_steps: 0,
            sim_time: 0.0,
            step: 0,
            work: 0.0,
            history: History::default(),
            steps_last_tick: 0,
            wall_ratio: 0.0,
            ms_per_step: 0.0,
            last_error: None,
            notes: Vec::new(),
        };
        d.provider.reset();
        if let Err(e) = d.evaluate(&mut || false) {
            d.stale_eval_failed = true;
            d.last_error = Some(format!("{}: {e}", d.provider.name()));
        }
        d
    }

    /// Applies one command between steps (module doc for each). Never evaluates and never
    /// panics on bad input: a rejected command sets `last_error` and changes nothing.
    pub fn apply(&mut self, cmd: DynCommand) {
        match cmd {
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
                self.pending_steps += n;
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
                }
            }
            DynCommand::SetThermostat(None) => {
                self.thermostat = None;
                self.thermostat_params = None;
            }
            DynCommand::SetThermostat(Some((t, tau))) => self.set_thermostat(t, tau),
            DynCommand::SetExternalField(a) => {
                if !a.iter().all(|x| x.is_finite()) {
                    return self.reject(format!("external field {a:?}: must be finite"));
                }
                let u0 = self.field_energy();
                self.scene.external_field = a;
                self.work += self.field_energy() - u0;
            }
            DynCommand::DragStart { index, target } => {
                if index >= self.scene.nuclei.len() || !target.iter().all(|x| x.is_finite()) {
                    return self.reject(format!("drag nucleus {index} to {target:?}: no such nucleus or non-finite target"));
                }
                self.set_drag(Some((index, target)));
            }
            DynCommand::DragMove { target } => {
                if let Some((i, _)) = self.drag {
                    if !target.iter().all(|x| x.is_finite()) {
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
                self.pins = vec![false; scene.nuclei.len()];
                self.scene = scene;
                self.drag = None;
                self.sim_time = 0.0;
                self.step = 0;
                self.discontinuity();
            }
            DynCommand::MoveNucleusDiscontinuous { index, pos } => {
                if !self.paused {
                    return self.reject("pause before moving a nucleus (or drag it while playing)".into());
                }
                if index >= self.scene.nuclei.len() || !pos.iter().all(|x| x.is_finite()) {
                    return self.reject(format!("move nucleus {index} to {pos:?}: no such nucleus or non-finite position"));
                }
                let n = &mut self.scene.nuclei[index];
                n.pos = pos;
                n.vel = [0.0; 3];
                self.discontinuity();
            }
            DynCommand::AddNucleus(n) => {
                if !self.paused {
                    return self.reject("pause before adding a nucleus".into());
                }
                if !(n.pos.iter().all(|x| x.is_finite()) && n.mass.is_finite() && n.mass > 0.0) {
                    return self.reject(format!("add nucleus at {:?} with mass {}: non-finite or non-positive", n.pos, n.mass));
                }
                self.scene.nuclei.push(Nucleus { vel: [0.0; 3], ..n });
                self.pins.push(false);
                self.discontinuity();
            }
            DynCommand::RemoveNucleus(i) => {
                if !self.paused {
                    return self.reject("pause before removing a nucleus".into());
                }
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
                self.discontinuity();
            }
        }
    }

    /// Advances time (module doc, "Time control"): re-evaluates stale forces first, then runs
    /// the queued `Step`s or, when playing, up to `speed` steps within `budget_ms` (at least
    /// one). `should_abort` is polled before every step and handed to the provider.
    pub fn tick(&mut self, budget_ms: f64, should_abort: &mut dyn FnMut() -> bool) -> DynSnapshot {
        let t0 = Instant::now();
        self.steps_last_tick = 0;
        self.wall_ratio = 0.0;
        let want = if self.paused { self.pending_steps } else { self.speed as usize };
        // stale forces: re-evaluate (also when paused with nothing queued — the edit's new
        // energy and density are wanted at once) unless that already failed and nothing has
        // asked for a retry since (Play, Step, an edit)
        if self.forces.is_none() && !self.stale_eval_failed {
            if should_abort() {
                return self.snapshot();
            }
            match self.evaluate(should_abort) {
                Ok(()) => {}
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
            if done > 0 && t0.elapsed().as_secs_f64() * 1e3 + self.ms_per_step > budget_ms {
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
        let wall = t0.elapsed().as_secs_f64();
        if !self.paused && done > 0 && wall > 0.0 {
            self.wall_ratio = done as f64 * self.integrator.dt_fs() / wall;
        }
        self.snapshot()
    }

    /// The current state (cheap: clones the scene, the history tail and the display handle).
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
            // the relative margin keeps a round trip fs → a.u. → fs of exactly 0.5 quiet
            dt_warning: (dt_fs > DT_WARN_FS * (1.0 + 1e-12))
                .then(|| format!("dt = {dt_fs:.3} fs is above {DT_WARN_FS} fs: H–X stretches (~10 fs period) are under-resolved; watch the drift")),
            notes: self.notes.clone(),
            eval: self.eval.clone(),
            display: self.display.clone(),
            history: self.history.tail(),
            thermostat: self.thermostat_params,
            com_correction: self.com_correction,
            pins: self.pins.clone(),
            drag: self.drag,
            provider: self.provider.name(),
        }
    }

    // ---- internals ----

    /// Evaluates at the current positions (stale forces → fresh), restarting the history: the
    /// energy after an edit is a new level.
    fn evaluate(&mut self, should_abort: &mut dyn FnMut() -> bool) -> Result<(), ProviderError> {
        let r = self.provider.evaluate(&self.scene, should_abort)?;
        if r.forces.len() != self.scene.nuclei.len() {
            return Err(ProviderError::Invalid(format!("provider '{}' returned {} forces for {} nuclei", self.provider.name(), r.forces.len(), self.scene.nuclei.len())));
        }
        self.forces = Some(r.forces);
        self.energy_ks = r.energy;
        self.eval = r.stats;
        self.display = r.display;
        self.restart_history();
        Ok(())
    }

    /// One step: Verlet with the tool forces, pins, thermostat, pins, bookkeeping.
    fn step_once(&mut self, should_abort: &mut dyn FnMut() -> bool) -> Result<(), ProviderError> {
        let forces_old = self.forces.as_ref().expect("step_once with stale forces");
        let pos_before: Vec<[f64; 3]> = self.scene.nuclei.iter().map(|n| n.pos).collect();
        let (drag, com, pins) = (self.drag, self.com_correction, &self.pins);
        let mut augment = |s: &Scene, f: &mut Vec<[f64; 3]>| tool_forces(s, f, drag, com, pins);
        let (f_new, stats) = self.integrator.step(&mut self.scene, forces_old, self.provider.as_mut(), &mut augment, should_abort)?;
        // the COM correction's work, trapezoidal (module doc)
        if com && !f_new.is_empty() {
            let (c_old, c_new) = (mean(forces_old), mean(&f_new));
            let mut w = 0.0;
            for (n, r0) in self.scene.nuclei.iter().zip(&pos_before) {
                for a in 0..3 {
                    w -= 0.5 * (c_old[a] + c_new[a]) * (n.pos[a] - r0[a]);
                }
            }
            self.work += w;
        }
        // pins are a constraint: restore the (rounding-level) displacement exactly
        for ((n, r0), &p) in self.scene.nuclei.iter_mut().zip(&pos_before).zip(&self.pins) {
            if p {
                n.pos = *r0;
            }
        }
        // pins → thermostat → pins; the kinetic energy they change is booked in W
        let k0 = self.scene.kinetic_energy();
        self.zero_pinned_velocities();
        if let Some(th) = &mut self.thermostat {
            th.apply(&mut self.scene, self.integrator.dt);
            self.zero_pinned_velocities();
        }
        self.work += self.scene.kinetic_energy() - k0;
        self.forces = Some(f_new);
        self.energy_ks = stats.energy_ks;
        self.eval = stats.eval;
        self.display = stats.display;
        self.sim_time += self.integrator.dt;
        self.step += 1;
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
        let th = self.thermostat.as_mut().expect("just set");
        let k0 = self.scene.kinetic_energy();
        if !self.scene.nuclei.is_empty() && k0 < AT_REST_FRACTION * th.target_kinetic(self.scene.dof()) {
            th.maxwell_boltzmann(&mut self.scene);
            self.zero_pinned_velocities();
            self.work += self.scene.kinetic_energy() - k0;
            self.note(format!("thermostat on: Maxwell–Boltzmann start at {t} K (the system was at rest)"));
        }
    }

    fn set_drag(&mut self, drag: Option<(usize, [f64; 3])>) {
        let u0 = self.spring_energy();
        self.drag = drag;
        self.work += self.spring_energy() - u0;
    }

    /// ½ k_drag |target − r|² (Hartree), 0 without a drag.
    fn spring_energy(&self) -> f64 {
        match self.drag {
            Some((i, t)) => {
                let r = self.scene.nuclei[i].pos;
                0.5 * K_DRAG * ((t[0] - r[0]).powi(2) + (t[1] - r[1]).powi(2) + (t[2] - r[2]).powi(2))
            }
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
    }

    /// After an accepted discontinuous edit: cold provider, stale forces, history restart.
    fn discontinuity(&mut self) {
        self.provider.reset();
        self.forces = None;
        self.stale_eval_failed = false;
        self.restart_history();
        self.last_error = None;
    }

    fn fail(&mut self, e: ProviderError) {
        self.paused = true;
        self.pending_steps = 0;
        self.last_error = Some(format!("{}: {e}", self.provider.name()));
    }

    fn reject(&mut self, msg: String) {
        self.last_error = Some(msg);
    }

    fn note(&mut self, msg: String) {
        if self.notes.len() == NOTES_KEPT {
            self.notes.remove(0);
        }
        self.notes.push(format!("step {}: {msg}", self.step));
    }
}

/// Σ F / N (zero for no forces).
fn mean(f: &[[f64; 3]]) -> [f64; 3] {
    if f.is_empty() {
        return [0.0; 3];
    }
    std::array::from_fn(|a| neumaier(f.iter().map(|x| x[a])) / f.len() as f64)
}

/// The `augment` of every step (module doc, "Tool forces"): COM correction on the provider
/// forces, then the drag spring, then the pins' zero-net-acceleration override.
fn tool_forces(scene: &Scene, f: &mut [[f64; 3]], drag: Option<(usize, [f64; 3])>, com: bool, pins: &[bool]) {
    if com {
        let m = mean(f);
        for x in f.iter_mut() {
            for a in 0..3 {
                x[a] -= m[a];
            }
        }
    }
    if let Some((i, t)) = drag {
        let r = scene.nuclei[i].pos;
        for a in 0..3 {
            f[i][a] += K_DRAG * (t[a] - r[a]);
        }
    }
    let g = scene.external_field;
    for ((x, n), &p) in f.iter_mut().zip(&scene.nuclei).zip(pins) {
        if p {
            *x = [-n.mass * g[0], -n.mass * g[1], -n.mass * g[2]];
        }
    }
}
