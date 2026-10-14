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
//! **`energy_conserved`** = E_KS + K + U_spring + U_field + U_wall − W (U_wall: "Container walls"), with U_spring the capped
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
//! [`Dynamics::provider_mut`] and [`Trainable::refit`](crate::Trainable::refit)). The next `tick` then re-evaluates at the unchanged positions
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
//! SetThermostat, SetExternalField, SetDt, SetComCorrection or SetWalls: those only change the state the
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
//! `SetIsotope { index, a }` (M6 spec §5.3) is accepted at any time and is not an edit of the
//! surface: forces do not depend on mass, so the provider is neither reset nor re-evaluated and
//! the forces stay fresh; the nucleus takes the listed isotope's mass (`Nucleus::isotope`; an
//! unlisted A is rejected naming the listed ones) at fixed velocity, so K jumps by
//! ΔK = ½(m′ − m)|v|², booked as W += ΔK exactly like a thermostat rescale — E_conserved is
//! continuous and the history is kept. The thermostat's target K̄ is a function of T and dof
//! only, so it is unaffected; a pinned nucleus has v = 0, so its ΔK is 0. U_spring and U_wall
//! are mass-free; U_field = −Σ m_i a·r_i is not, and its change (m′ − m)(−a·r) is deliberately
//! NOT booked: with the gravity hook's a = 9.8 m/s² = 1.1e-22 a.u. an H → D swap at 10 Bohr
//! moves U_field by ≈ 2e-18 Ha, three orders below f64 rounding of E (≈ 1e-16 Ha at |E| ≈ 1),
//! so booking it would add a term that can never be seen.
//!
//! # Sampling and the model commands (M5)
//! `StartSampling(plan)` plans a sampling pass over the current scene (`providers::sampling`'s
//! `Sampler`, run inside the provider: [`Trainable::start_pass`](crate::Trainable::start_pass))
//! and pauses the dynamics; while it runs, Play and Step are refused (`last_rejection`), edits are
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
//! cached across passes by the provider (and seeded by a loaded model), keyed by the atom's
//! solve settings.
//! `ForgetModel` cancels a running pass without a refit and drops the model, its samples and
//! references (`Trainable::forget`; the next evaluation is on the teacher surface and the
//! conserved-energy history restarts). `SaveModel(path)` writes the surrogate's full sidecar;
//! `LoadModel(path)` reads one and installs it only if its provenance equals the current
//! scene's (grid, h, σ, tolerance, charge, multiplicity, teacher) — otherwise it is refused with
//! a note naming both, and nothing changes. A corrupt file is refused with the reader's error.
//! All five need a provider that can be taught ([`ForceProvider::as_trainable`], the surrogate)
//! and are refused otherwise; they reach it through that trait, never by downcasting and never
//! on the stepping path. The messages the provider produces for them (a pass started or
//! finished, the references, a model loaded, refused or forgotten) are its queued notes
//! ([`Trainable::take_notes`](crate::Trainable::take_notes)), logged at once. **The worker should
//! raise `should_abort` for CancelSampling, ForgetModel and LoadModel** (they make a running
//! sampling solve pointless) and, as for steps, not for anything else.
//! # Side densities (M5b, "Density in estimated mode")
//! A provider may solve densities off the stepping path ([`ForceProvider::side_jobs`] /
//! [`ForceProvider::take_side_display`]: the surrogate's background audits and display refreshes).
//! The driver stamps every new side job with the simulated time of the geometry it was dispatched
//! at — checked at the start of each tick (a refresh the worker asked for between ticks is at the
//! current positions) and after every evaluation (an audit is dispatched inside the evaluation of
//! the step's new positions, t + dt) — and polls for an arrived density at the end of each tick.
//! The newest one is the snapshot's `side_display` ([`SideDensity`]: the density, who solved it,
//! when, at which positions, and a sequence number that changes with each new one) until an edit
//! (the positions it was solved at no longer exist) clears it. Display only: nothing here enters
//! the energies.
//!
//! # Container walls (user request 2026-10-14)
//! `SetWalls(Some(Walls { half_width: L, k }))` puts a smooth repulsive wall on each of the six
//! faces of the cube [−L, L]³ (centred on the cell like the grid). Per nucleus and per axis, with
//! s = max(L − [`WALL_LAYER`], 0) the inner edge of the soft layer:
//! U = u(|x| − s) for |x| > s, else 0, F = −u'(|x| − s) sign(x) — into the box — with the depth
//! d = |x| − s and u'' = k min(d/a, 1): u = k d³/(6a) for d ≤ a = [`WALL_ONSET`], then
//! k (d²/2 − a d/2 + a²/6) (harmonic at stiffness k past the onset). The curvature ramps up instead
//! of jumping because a plain ½ k d² wall (C¹: u'' jumps from 0 to k at the layer's edge) has no
//! Verlet shadow Hamiltonian across the edge: measured, its kinetic energy random-walked by ≈ 1 %
//! per bounce (20 % after ~300 bounces of a free ¹H at 0.05 Ha), which would make the drift
//! readout meaningless for a gas in a box; with u'' continuous the error stays bounded — measured
//! max |K/E − 1| = 1.05e-3 over 306 bounces (test bound 2e-3,
//! `a_free_nucleus_bounces_off_the_wall_with_e_conserved_flat`; `validate/references.md`, "Viewer
//! extras (M5b)", has every wall constant and test bound). It is a
//! conservative potential of the positions, entered through the `augment` closure after the
//! drag spring (so the COM correction never averages it; a pinned nucleus's override still wins)
//! and **booked like the spring**: U_wall is in `energy_conserved`, and switching the walls on or
//! off or moving them (`SetWalls` while playing: the piston) books ΔU at fixed positions into W —
//! the wall's work on the atoms, Σ F_wall·Δwall for an instantaneous move, the same booking as a
//! moving drag target (with the same O(h²) shadow residual while it moves). Rewind stays exact:
//! the wall is reversible. `walls_work` publishes the running sum of those moves.
//!
//! **Stiffness.** k = [`WALL_K`] = 3.5 Ha/Bohr², onset a = 0.5 Bohr, layer D = [`WALL_LAYER`] =
//! 1 Bohr — fixed constants, not per element (rule 1). A nucleus whose kinetic energy along one
//! axis is E turns at the depth δ with u(δ) = E — δ = a/2 + √(2E/k − a²/12) once E ≥ k a²/6 —
//! so k is chosen for **u(D) = k (D²/2 − aD/2 + a²/6) = 1.02 Ha: a nucleus with E ≤ 1 Ha per axis
//! turns at or before the face L** (δ = 0.99 Bohr at 1 Ha; |v_a| ≤ √(2 Ha / m); 1 Ha ≈ 3.2e5 K,
//! far above any bond energy here). Verlet's discrete orbit overshoots the exact turning point by
//! O((ω dt)²) of the depth (the shadow ellipse of the harmonic part has amplitude × 1/√(1 −
//! (ω dt)²/4)), with ω = √(k/m) ≤ √(3.5/1837.15) = 0.044 a.u.⁻¹ for ¹H (the lightest nucleus):
//! ω dt = 0.45 at 0.25 fs, 0.90 at 0.5 fs — measured 0.006 and 0.022 Bohr past the exact turn at
//! 1 Ha per axis (`the_wall_stops_a_one_hartree_nucleus_inside_the_margin`), inside the h/2 left
//! between the face and the providers' margin. That is why L is capped at
//! [`Walls::max_half_width`] = `Cell::max_coord()` − h/2: the cell's margin guard (R12, "left the
//! simulation cell") then never fires for a wall-confined nucleus with E ≤ 1 Ha per axis, and it
//! still protects the solver from anything faster (above u(D + h/2) = 1.38 Ha at h = 0.25 the
//! guard pauses the run, as without walls). Verlet is stable for ω dt < 2; a stiffer wall would
//! cost resolution of the bounce at the default dt (and a larger O((ω dt)²) energy swing while
//! in contact), a softer one a thicker layer for the same stopping energy.
//! `ReplaceScene` into a smaller cell moves L in to the new maximum (with a note; the history
//! restarts there anyway). `SetWalls` never aborts a running step (it changes only the state the
//! next step starts from, like a drag move).
//!
//! **Degrees of freedom** (review M3): the convention is unchanged — `Scene::dof` = 3N − 3 and the
//! thermostat preserves the COM velocity (R10). With walls the total momentum is not conserved
//! (every bounce exchanges some), so the COM mode thermalises through the walls, not through the
//! thermostat; the temperature readout stays a valid estimator.
//! # Relax (M6 spec §6; `crate::relax` for FIRE itself)
//! `Relax` starts a relaxation, `StopRelax` (and `Pause`) ends it. A relax is a **tool mode**
//! like a sampling pass: the run pauses (its velocities are quenched, ΔK booked), each `tick`
//! runs as many FIRE steps as fit `budget_ms` — at least one, unless the tick already spent a
//! solve re-evaluating stale forces —, `Play`, `Step` and `StartSampling` are refused with
//! "relaxing: Stop relax first", edits are accepted (the stale forces are re-evaluated by the
//! next tick as usual, and FIRE restarts its Δt and α on the edited scene — `Fire::restart`,
//! likewise after a surface refresh), pins hold, and the drag spring, the COM correction and
//! the walls act through the same `Tools::apply` as in MD — without the drag damping: FIRE has
//! its own dissipation, and the stop criterion must be a function of the positions. The
//! thermostat is bypassed (and the R7 at-rest draw after an edit is skipped while relaxing).
//! **Bookkeeping (ruling P8, review MUST-7):** a FIRE step is not dynamics, so everything it
//! changes is tool work — per step W += `FireStep::delta_booked` (ΔE_KS + ΔK over the whole
//! step, backtrack included) + ΔU_spring + ΔU_field + ΔU_wall, so `energy_conserved` is
//! constant to rounding through the relax and W grows by exactly E_end − E_start (the energy
//! drained; W is "energy put in", so it is negative for a relax). A `refit_jump` or a surface
//! flip inside a FIRE step's evaluation is already inside ΔE_KS and is not booked again; the
//! force-mismatch work has nothing to book (there is no trajectory). `sim_time` and `step` do
//! not move (FIRE's Δt is its own, rule 8), no history point is pushed, and when the relax ends
//! — `Converged`, `Stalled`, `StepCap`, `StopRelax`/`Pause`, a provider failure, or
//! `ReplaceScene` — the velocities are zeroed (ΔK booked), the drift history is cleared once
//! (W is kept: it is the drained energy), `relax.converged` is `Some(outcome == Converged)`
//! with a note for anything else. On a provider that can be taught (the surrogate) a converged
//! relax whose last evaluation was the model's is checked by one teacher solve at the result
//! ([`Trainable::force_teacher_next`](crate::Trainable::force_teacher_next), then `evaluate`: the answer is a model-surface fallback
//! that is kept as a sample and refitted by the worker as usual) — `verified_max_force` is that
//! answer's max|F_i|; the positions did not move, so only a surface flip at fixed positions
//! (teacher − model energy, when the model could not answer) is booked, exactly like a refit
//! jump. If it exceeds F_TOL the note says so, and a second `Relax` continues from there.
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
use crate::provider::{Display, EvalStats, ForceProvider, ProviderError, SideKind, Source, Surface};
use crate::references::AtomReferences;
use crate::relax::{max_norm, Fire, FireOutcome, FireParams};
pub use crate::relax::RelaxProgress;
use crate::sampling::{SamplingPlan, SamplingProgress};
use crate::scene::{neumaier, Cell, Nucleus, Scene};
use crate::thermostat::Csvr;
use health::{History, Point};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
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
/// Container wall stiffness (Hartree/Bohr²; module doc, "Container walls").
pub const WALL_K: f64 = 3.5;
/// Thickness of the walls' soft layer inside the face (Bohr; module doc, "Container walls").
pub const WALL_LAYER: f64 = 1.0;
/// Depth over which the walls' curvature ramps from 0 to k (Bohr; module doc, "Container walls").
pub const WALL_ONSET: f64 = 0.5;

/// A density solved off the stepping path (module doc, "Side densities").
#[derive(Clone, Debug)]
pub struct SideDensity {
    pub display: Display,
    pub kind: SideKind,
    /// Simulated time (a.u.) of the geometry it was solved at.
    pub t: f64,
    /// The nuclear positions it was solved at (Bohr).
    pub positions: Vec<[f64; 3]>,
    /// Changes with every new side density.
    pub seq: u64,
}

impl SideDensity {
    pub fn t_fs(&self) -> f64 {
        units::au_time_to_fs(self.t)
    }
}

/// Side jobs whose dispatch time the driver remembers (one audit or refresh is in flight at a
/// time; the margin covers jobs dispatched within one tick).
const SIDE_TIMES_KEPT: usize = 8;

/// Container walls (module doc, "Container walls"): the cube [−half_width, half_width]³ (Bohr),
/// stiffness `k` (Hartree/Bohr², [`WALL_K`] by default).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Walls {
    pub half_width: f64,
    pub k: f64,
}

impl Walls {
    /// Walls at `half_width` with the default stiffness [`WALL_K`].
    pub fn new(half_width: f64) -> Walls {
        Walls { half_width, k: WALL_K }
    }

    /// The largest half-width the cell allows: its usable half-width (`Cell::max_coord`, the
    /// providers' 3h margin) less h/2 for Verlet's overshoot (module doc, "Container walls").
    pub fn max_half_width(cell: &Cell) -> f64 {
        cell.max_coord() - 0.5 * cell.h
    }

    /// Where the soft layer starts (|x| beyond this is pushed back).
    fn layer_start(&self) -> f64 {
        (self.half_width - WALL_LAYER).max(0.0)
    }

    /// U = Σ_axes u(|x| − s) beyond the layer start s (Hartree): u(d) = k d³/(6a) for d ≤ a,
    /// k (d²/2 − a d/2 + a²/6) beyond — u'' = k min(d/a, 1), a = [`WALL_ONSET`].
    pub fn potential(&self, pos: [f64; 3]) -> f64 {
        let s = self.layer_start();
        let a = WALL_ONSET;
        pos.iter()
            .map(|x| (x.abs() - s).max(0.0))
            .map(|d| if d <= a { self.k * d * d * d / (6.0 * a) } else { self.k * (0.5 * d * d - 0.5 * a * d + a * a / 6.0) })
            .sum()
    }

    /// −∇U: into the box (Hartree/Bohr).
    pub fn force(&self, pos: [f64; 3]) -> [f64; 3] {
        let s = self.layer_start();
        let a = WALL_ONSET;
        pos.map(|x| {
            let d = (x.abs() - s).max(0.0);
            let du = if d <= a { self.k * d * d / (2.0 * a) } else { self.k * (d - 0.5 * a) };
            -du * x.signum()
        })
    }
}

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
    /// Any time (spec §5.3): the nucleus takes isotope A's mass from the table (refused with the
    /// listed isotopes otherwise); K changes at fixed v and the ΔK is booked into W; the provider
    /// is not reset (module doc, "Edits").
    SetIsotope { index: usize, a: u16 },
    /// Any time (R8): pins and drag follow their nuclei, provider reset.
    RemoveNucleus(usize),
    SetComCorrection(bool),
    /// Container walls on (`Some`) or off (module doc, "Container walls"): finite k > 0 and
    /// 0 < half_width ≤ [`Walls::max_half_width`] of the cell, else rejected. Booked as ΔU at fixed
    /// positions (the wall's work), so it may be sent while playing (the piston).
    SetWalls(Option<Walls>),
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
    /// Start a relaxation (module doc, "Relax"): FIRE steps replace the dynamics until it
    /// converges, stalls, hits its cap or is stopped. Paused or playing (the run pauses first;
    /// the velocities are zeroed, ΔK booked); refused while a sampling pass or another relax
    /// runs.
    Relax,
    /// End the relaxation where it is (velocities zeroed, ΔK booked); `Pause` does the same.
    /// No-op without one.
    StopRelax,
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
    /// The invariant the drift is fitted on (module doc): total + U_spring + U_field + U_wall − W.
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
    /// The expected wall cost of the next step (ms): the moving average of the latest step's kind
    /// — [`ms_per_step_model`](Self::ms_per_step_model) after a surrogate answer, else
    /// [`ms_per_step_dft`](Self::ms_per_step_dft) — which is what the tick's budget check uses
    /// (Task 9: one mixed average made a DFT fallback stop the next model ticks after one step).
    /// 0 before the first step.
    pub ms_per_step: f64,
    /// Exponential moving averages of the wall cost of a step answered by the surrogate model
    /// and of one answered by DFT (a Kohn-Sham provider's step, or a surrogate's fallback), ms;
    /// 0 before the first step of that kind.
    pub ms_per_step_model: f64,
    pub ms_per_step_dft: f64,
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
    /// Each history point's evaluation source (`Source`), parallel to `history`.
    pub history_source: Arc<[Source]>,
    /// Whether each history point was a draft-mode guess (`EvalStats::extrapolation`: the model
    /// answered where it would otherwise have handed the step to its teacher), parallel to
    /// `history` — the sparkline marks those steps.
    pub history_guess: Arc<[bool]>,
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
    /// Σ of the surface jumps (`EvalStats::refit_jump`: refits, failed audits) booked into W
    /// since the history (re)started (Hartree): the model's part of "work done by tools", shown
    /// on its own row. 0 under a Kohn-Sham provider.
    pub work_refit: f64,
    /// The running sampling pass's progress, or the last pass's final state (`finished`,
    /// possibly `cancelled`) until the next pass or `ForgetModel`; `None` before any.
    pub sampling: Option<SamplingProgress>,
    /// The running relaxation's progress, or the last one's final state (`converged` is
    /// `Some`) until the next `Relax`; `None` before any (module doc, "Relax").
    pub relax: Option<RelaxProgress>,
    /// The newest density solved off the stepping path (module doc, "Side densities").
    pub side_display: Option<SideDensity>,
    /// The container walls (module doc, "Container walls").
    pub walls: Option<Walls>,
    /// Σ of the walls' moves booked into W since the history (re)started (Hartree): the piston's
    /// work on the atoms (positive when shrinking the box squeezes them).
    pub walls_work: f64,
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
            ms_per_step_model: 0.0,
            ms_per_step_dft: 0.0,
            last_error: None,
            last_rejection: None,
            dt_warning: None,
            notes: Vec::new(),
            notes_total: 0,
            seq: 0,
            eval: EvalStats::default(),
            display: None,
            history: Arc::from(Vec::new()),
            history_source: Arc::from(Vec::new()),
            history_guess: Arc::from(Vec::new()),
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
            work_refit: 0.0,
            sampling: None,
            relax: None,
            side_display: None,
            walls: None,
            walls_work: 0.0,
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
    walls: Option<Walls>,
    /// The part of W that is the walls' moves.
    walls_work: f64,
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
    /// The part of W that is surface jumps (`refit_jump`).
    work_refit: f64,
    steps_last_tick: usize,
    wall_ratio: f64,
    /// Step-cost moving averages (ms) of model and DFT steps, and the latest step's kind.
    ms_model: f64,
    ms_dft: f64,
    last_step_model: bool,
    last_error: Option<String>,
    last_rejection: Option<String>,
    notes: Vec<String>,
    notes_total: u64,
    seq: u64,
    /// The progress of the sampling pass (module doc, "Sampling"; the pass itself runs inside the
    /// provider), kept after it ends (the snapshot's `sampling`).
    sampling: Option<SamplingProgress>,
    /// Isolated-atom references handed to every pass (`Trainable::start_pass`; a stub until M6
    /// Task 3 makes the driver their owner — the provider keeps the cache meanwhile).
    refs: AtomReferences,
    /// Mean wall cost of one sampling solve (ms, moving average).
    sample_ms: f64,
    /// The running relaxation (module doc, "Relax").
    fire: Option<Fire>,
    /// Its progress, kept after it ends (the snapshot's `relax`).
    relax: Option<RelaxProgress>,
    /// Side jobs seen, their dispatch times (job, t a.u.), the newest side density and its
    /// sequence number (module doc, "Side densities").
    side_jobs_seen: u64,
    side_times: VecDeque<(u64, f64)>,
    side_display: Option<SideDensity>,
    side_seq: u64,
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
            walls: None,
            walls_work: 0.0,
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
            work_refit: 0.0,
            steps_last_tick: 0,
            wall_ratio: 0.0,
            ms_model: 0.0,
            ms_dft: 0.0,
            last_step_model: false,
            last_error: None,
            last_rejection: None,
            notes: Vec::new(),
            notes_total: 0,
            seq: 0,
            sampling: None,
            refs: AtomReferences::default(),
            sample_ms: 0.0,
            fire: None,
            relax: None,
            side_jobs_seen: 0,
            side_times: VecDeque::new(),
            side_display: None,
            side_seq: 0,
        };
        d.provider.reset();
        d.last_epoch = d.provider.surface_epoch();
        d.side_jobs_seen = d.provider.side_jobs();
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
            DynCommand::Play | DynCommand::Step(_) | DynCommand::StartSampling(_) if self.fire.is_some() => {
                return self.reject("relaxing: Stop relax first".into());
            }
            DynCommand::Play | DynCommand::Step(_) | DynCommand::Relax if self.pass_active() => {
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
                self.finish_relax(None, &mut || false);
            }
            DynCommand::Relax => self.start_relax(),
            DynCommand::StopRelax => self.finish_relax(None, &mut || false),
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
            DynCommand::SetWalls(w) => {
                if let Some(w) = w {
                    let max = Walls::max_half_width(&self.scene.cell);
                    if !(w.k.is_finite() && w.k > 0.0 && w.half_width.is_finite() && w.half_width > 0.0 && w.half_width <= max * (1.0 + 1e-12)) {
                        return self.reject(format!("walls at {} Bohr with k = {}: need finite k > 0 and 0 < half-width ≤ {max:.3} Bohr (the cell less its margin)", w.half_width, w.k));
                    }
                }
                let u0 = self.walls_energy();
                self.walls = w;
                let du = self.walls_energy() - u0;
                self.work += du;
                self.walls_work += du;
            }
            DynCommand::ReplaceScene(scene) => {
                if !self.paused {
                    return self.reject("pause before replacing the scene".into());
                }
                let bad = scene.nuclei.iter().position(|n| !(finite3(n.pos) && finite3(n.vel) && n.mass.is_finite() && n.mass > 0.0));
                if bad.is_some() || !finite3(scene.external_field) {
                    return self.reject(format!("replace scene: nucleus {bad:?} or the field is non-finite, or a mass is not positive"));
                }
                // an accepted new scene ends a relax of the old one (the new scene's velocities
                // are kept, so the old one's are zeroed and booked first) and a pass over it
                // (what it sampled is kept and refitted); a rejected one changes nothing
                self.finish_relax(None, &mut || false);
                self.cancel_sampling();
                self.pins = vec![false; scene.nuclei.len()];
                self.scene = scene;
                self.drag = None;
                let max = Walls::max_half_width(&self.scene.cell);
                if let Some(w) = self.walls.as_mut().filter(|w| w.half_width > max) {
                    w.half_width = max.max(f64::MIN_POSITIVE);
                    self.note(format!("container walls moved in to {max:.2} Bohr: the new cell is smaller"));
                }
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
            DynCommand::SetIsotope { index, a } => {
                let Some(n) = self.scene.nuclei.get(index) else {
                    return self.reject(format!("set isotope of nucleus {index}: there are {}", self.scene.nuclei.len()));
                };
                let iso = match Nucleus::isotope(n.z, a, n.pos) {
                    Ok(iso) => iso,
                    Err(e) => return self.reject(e),
                };
                // the mass changes at fixed velocity: ΔK = ½(m′ − m)|v|² is the user's work
                // (the ≈ 2e-18 Ha U_field term is documented as unbooked — module doc, "Edits")
                let k0 = self.scene.kinetic_energy();
                let n = &mut self.scene.nuclei[index];
                n.a = iso.a;
                n.mass = iso.mass;
                self.work += self.scene.kinetic_energy() - k0;
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
            DynCommand::CancelSampling => self.cancel_sampling(),
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
    /// SetThermostat, SetExternalField, SetDt, SetComCorrection or SetWalls — they change only the next
    /// step's starting state, and aborting on a per-frame DragMove would cancel every step.
    pub fn tick(&mut self, budget_ms: f64, should_abort: &mut dyn FnMut() -> bool) -> DynSnapshot {
        let t0 = Instant::now();
        self.steps_last_tick = 0;
        self.wall_ratio = 0.0;
        // a side job asked for between ticks is at the current positions (module doc)
        self.stamp_side_jobs(self.sim_time);
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
            // (a live evaluation between sample solves must not extrapolate the teacher's density
            // through the other's geometry, review m2: the provider that runs the pass sees to it)
            match self.evaluate(should_abort) {
                Ok(()) => evaluated = true,
                Err(ProviderError::Cancelled) => return self.snapshot(),
                Err(e) => {
                    self.stale_eval_failed = true;
                    self.fail(e);
                    return self.snapshot();
                }
            }
            // a relax continues on the edited scene (or the refreshed surface) from FIRE's
            // start values; the fresh energy is its new reference (module doc, "Relax")
            if let Some(fire) = self.fire.as_mut() {
                fire.restart(self.energy_ks);
            }
            if self.fire.is_some() {
                self.relax = Some(self.relax_progress());
            }
        }
        if self.fire.is_some() {
            self.advance_relax(t0, budget_ms, evaluated, should_abort);
            self.poll_side_display();
            return self.snapshot();
        }
        let mut done = 0usize;
        while done < want && self.forces.is_some() {
            if should_abort() {
                break;
            }
            // the "at least one step" guarantee does not apply after a stale re-evaluation:
            // that solve already spent the tick (a cold GPU solve costs ~1 s)
            if (done > 0 || evaluated) && t0.elapsed().as_secs_f64() * 1e3 + self.ms_next() > budget_ms {
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
                    let model = self.eval.source == Source::Surrogate;
                    let ema = if model { &mut self.ms_model } else { &mut self.ms_dft };
                    *ema = if *ema == 0.0 { ms } else { 0.8 * *ema + 0.2 * ms };
                    self.last_step_model = model;
                }
                Err(ProviderError::Cancelled) => break,
                Err(e) => {
                    self.fail(e);
                    break;
                }
            }
        }
        self.steps_last_tick = done;
        if self.pass_active() {
            self.advance_sampling(t0, budget_ms, evaluated, should_abort);
        }
        let wall = t0.elapsed().as_secs_f64();
        if !self.paused && done > 0 && wall > 0.0 {
            self.wall_ratio = done as f64 * self.integrator.dt_fs() / wall;
        }
        self.poll_side_display();
        self.snapshot()
    }

    /// Stamps the side jobs dispatched since the last check with `t` (module doc, "Side
    /// densities").
    fn stamp_side_jobs(&mut self, t: f64) {
        let jobs = self.provider.side_jobs();
        if jobs == self.side_jobs_seen {
            return;
        }
        for job in self.side_jobs_seen.saturating_add(1).max(jobs.saturating_sub(SIDE_TIMES_KEPT as u64 - 1))..=jobs {
            if self.side_times.len() == SIDE_TIMES_KEPT {
                self.side_times.pop_front();
            }
            self.side_times.push_back((job, t));
        }
        self.side_jobs_seen = jobs;
    }

    /// Takes an arrived side density (module doc, "Side densities").
    fn poll_side_display(&mut self) {
        let Some(sd) = self.provider.take_side_display() else { return };
        let t = self.side_times.iter().find(|(j, _)| *j == sd.job).map_or(self.sim_time, |&(_, t)| t);
        self.side_seq += 1;
        self.seq += 1;
        self.side_display = Some(SideDensity { display: sd.display, kind: sd.kind, t, positions: sd.positions, seq: self.side_seq });
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
            ms_per_step: self.ms_next(),
            ms_per_step_model: self.ms_model,
            ms_per_step_dft: self.ms_dft,
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
            history_source: Arc::from(self.history.sources()),
            history_guess: Arc::from(self.history.guesses()),
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
            work_refit: self.work_refit,
            sampling: self.sampling.clone(),
            relax: self.relax.clone(),
            side_display: self.side_display.clone(),
            walls: self.walls,
            walls_work: self.walls_work,
        }
    }

    /// A line for `DynSnapshot::notes` from the driver's owner (the app worker's refit outcomes,
    /// which happen between ticks, outside the driver).
    pub fn add_note(&mut self, msg: String) {
        self.note(msg);
        self.seq += 1;
    }

    /// The force provider, for what the stepping path does not do: the worker refits a surrogate
    /// between ticks and reads its status through `as_trainable` (module doc, "A provider whose
    /// surface moves"). Never call `evaluate` through it: the driver's forces would go stale
    /// unnoticed.
    pub fn provider_mut(&mut self) -> &mut dyn ForceProvider {
        self.provider.as_mut()
    }

    // ---- sampling and the model commands (module doc, "Sampling") ----

    fn no_surrogate(&mut self, what: &str) {
        let name = self.provider.name();
        self.reject(format!("{what}: the provider '{name}' has no surrogate model (choose the estimated solver)"));
    }

    /// A sampling pass is running in the provider.
    fn pass_active(&mut self) -> bool {
        self.provider.as_trainable().is_some_and(|t| t.pass_active())
    }

    /// The provider's queued messages into the log (`Trainable::take_notes`).
    fn take_provider_notes(&mut self) {
        let notes = self.provider.as_trainable().map(|t| t.take_notes()).unwrap_or_default();
        for n in notes {
            self.note(n);
        }
    }

    fn start_sampling(&mut self, plan: SamplingPlan) {
        let Some(t) = self.provider.as_trainable() else {
            return self.no_surrogate("start sampling");
        };
        if t.pass_active() {
            return self.reject("start sampling: a pass is already running (cancel it first)".into());
        }
        if let Err(e) = t.start_pass(&self.scene, &plan, &mut self.refs) {
            return self.reject(format!("sampling plan: {e}"));
        }
        let p = t.progress();
        self.paused = true;
        self.pending_steps = 0;
        self.take_provider_notes();
        self.sampling = p;
    }

    /// Spends the tick's remaining budget on the pass (module doc).
    fn advance_sampling(&mut self, t0: Instant, budget_ms: f64, evaluated: bool, should_abort: &mut dyn FnMut() -> bool) {
        let mut solves = 0usize;
        let end: Option<Result<(), String>> = loop {
            if should_abort() {
                break None;
            }
            if (solves > 0 || evaluated) && t0.elapsed().as_secs_f64() * 1e3 + self.sample_ms > budget_ms {
                break None;
            }
            let Some(t) = self.provider.as_trainable() else {
                break Some(Err("the provider is no longer a surrogate".into()));
            };
            let ts = Instant::now();
            match t.pass_step(should_abort) {
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
        if let Some(Err(e)) = &end {
            self.last_error = Some(format!("sampling: {e}"));
        }
        self.take_provider_notes();
        if end.is_some() {
            self.pass_ended();
        }
    }

    /// Ends a running pass early (`CancelSampling`, an accepted `ReplaceScene`): the provider
    /// keeps and refits what it sampled. No-op without one.
    fn cancel_sampling(&mut self) {
        let Some(t) = self.provider.as_trainable().filter(|t| t.pass_active()) else { return };
        t.cancel_pass();
        self.take_provider_notes();
        self.pass_ended();
    }

    /// After a pass (complete or cancelled; the provider told its teacher that its state is a
    /// sample geometry's and refitted over everything it holds): its final progress.
    fn pass_ended(&mut self) {
        self.sampling = self.provider.as_trainable().and_then(|t| t.progress());
        self.seq += 1;
    }

    fn forget_model(&mut self) {
        let Some(t) = self.provider.as_trainable() else { return self.no_surrogate("forget model") };
        t.forget();
        self.sampling = None;
        self.take_provider_notes();
    }

    fn save_model(&mut self, path: PathBuf) {
        let forces = self.forces.clone().unwrap_or_default();
        let Some(t) = self.provider.as_trainable() else { return self.no_surrogate("save model") };
        let st = t.status(&self.scene, &forces);
        let (n, pending) = (st.samples, st.pending_refit);
        match t.save(&path) {
            Ok(()) => self.note(format!(
                "model saved to {} ({n} samples{})",
                path.display(),
                if pending { ", some taken since the last refit" } else { "" }
            )),
            Err(e) => self.reject(format!("save model to {}: {e}", path.display())),
        }
    }

    fn load_model(&mut self, path: PathBuf) {
        let Some(t) = self.provider.as_trainable() else { return self.no_surrogate("load model") };
        if t.pass_active() {
            return self.reject("load model: cancel the running sampling pass first".into());
        }
        let loaded = t.load(&path, &self.scene);
        if loaded.is_ok() {
            self.sampling = None;
        }
        self.take_provider_notes();
        if let Err(e) = loaded {
            self.reject(e);
        }
    }

    // ---- the relax tool (module doc, "Relax") ----

    /// The tool potentials a FIRE step's booking covers beside ΔE_KS + ΔK.
    fn tool_potentials(&self) -> f64 {
        self.spring_energy() + self.field_energy() + self.walls_energy()
    }

    /// The readout of the running relax from the driver's state (before a step, or after a
    /// restart on an edited scene).
    fn relax_progress(&self) -> RelaxProgress {
        let fire = self.fire.as_ref().expect("relax_progress without a relax");
        let (max_force, max_force_augmented) = match &self.forces {
            Some(f) => {
                let tools = Tools { drag: self.drag, damping: 0.0, force_max: self.drag_force_max, com: self.com_correction, walls: self.walls, pins: &self.pins };
                let mut a = f.clone();
                tools.apply(&self.scene, &mut a);
                for (x, &p) in a.iter_mut().zip(&self.pins) {
                    if p {
                        *x = [0.0; 3];
                    }
                }
                (max_norm(f), max_norm(&a))
            }
            // stale: the previous readout's numbers until the next tick evaluates
            None => self.relax.as_ref().map_or((0.0, 0.0), |r| (r.max_force, r.max_force_augmented)),
        };
        RelaxProgress {
            steps: fire.steps(),
            max_force,
            max_force_augmented,
            energy: self.energy_ks,
            dt_fs: units::au_time_to_fs(fire.dt()),
            alpha: fire.alpha(),
            converged: None,
            verified_max_force: None,
            note: None,
        }
    }

    fn start_relax(&mut self) {
        if self.fire.is_some() {
            return self.reject("already relaxing: Stop relax to end it".into());
        }
        self.paused = true;
        self.pending_steps = 0;
        self.last_error = None;
        self.stale_eval_failed = false;
        // FIRE starts from rest: the run's velocities are quenched, the ΔK is tool work
        let k0 = self.scene.kinetic_energy();
        for n in &mut self.scene.nuclei {
            n.vel = [0.0; 3];
        }
        self.work += self.scene.kinetic_energy() - k0;
        let params = FireParams::for_dt(self.integrator.dt);
        self.fire = Some(Fire::new(params, self.energy_ks));
        self.relax = Some(self.relax_progress());
        self.note(format!(
            "relax started: FIRE from Δt {:.3} fs (≤ {:.3} fs), stop at max|F| < {:.0e} Ha/Bohr; the dynamics are paused until it ends",
            units::au_time_to_fs(params.dt_start),
            units::au_time_to_fs(params.dt_max),
            params.f_tol
        ));
    }

    /// Spends the tick on FIRE steps (module doc, "Relax"): at least one unless the tick
    /// already re-evaluated stale forces, more while the running mean cost fits the budget.
    fn advance_relax(&mut self, t0: Instant, budget_ms: f64, evaluated: bool, should_abort: &mut dyn FnMut() -> bool) {
        let Some(mut fire) = self.fire.take() else { return };
        let mut done = 0usize;
        let mut end: Option<FireOutcome> = None;
        let mut failure: Option<ProviderError> = None;
        while self.forces.is_some() {
            if should_abort() {
                break;
            }
            if (done > 0 || evaluated) && t0.elapsed().as_secs_f64() * 1e3 + self.ms_next() > budget_ms {
                break;
            }
            let ts = Instant::now();
            let u0 = self.tool_potentials();
            let tools = Tools { drag: self.drag, damping: 0.0, force_max: self.drag_force_max, com: self.com_correction, walls: self.walls, pins: &self.pins };
            let mut augment = |s: &Scene, f: &mut Vec<[f64; 3]>| {
                tools.apply(s, f);
            };
            let forces = self.forces.as_mut().expect("checked by the loop");
            match fire.step(&mut self.scene, forces, &self.pins, self.provider.as_mut(), &mut augment, should_abort) {
                Ok(fs) => {
                    done += 1;
                    // everything the step changed is tool work: E_conserved stays put
                    self.work += fs.delta_booked + (self.tool_potentials() - u0);
                    self.energy_ks = fs.energy;
                    self.eval = fs.eval;
                    self.display = fs.display;
                    self.last_epoch = self.provider.surface_epoch();
                    self.seq += 1;
                    let ms = ts.elapsed().as_secs_f64() * 1e3;
                    let model = self.eval.source == Source::Surrogate;
                    let ema = if model { &mut self.ms_model } else { &mut self.ms_dft };
                    *ema = if *ema == 0.0 { ms } else { 0.8 * *ema + 0.2 * ms };
                    self.last_step_model = model;
                    self.relax = Some(RelaxProgress {
                        steps: fire.steps(),
                        max_force: fs.max_force,
                        max_force_augmented: fs.max_force_augmented,
                        energy: fs.energy,
                        dt_fs: units::au_time_to_fs(fire.dt()),
                        alpha: fire.alpha(),
                        converged: None,
                        verified_max_force: None,
                        note: None,
                    });
                    if fs.outcome != FireOutcome::Running {
                        end = Some(fs.outcome);
                        break;
                    }
                }
                Err(ProviderError::Cancelled) => break,
                Err(e) => {
                    failure = Some(e);
                    break;
                }
            }
        }
        self.fire = Some(fire);
        if let Some(e) = failure {
            // the step was atomic: the scene and forces are the last good ones
            self.fail(e);
            self.finish_relax(None, should_abort);
        } else if let Some(o) = end {
            self.finish_relax(Some(o), should_abort);
        }
    }

    /// Ends the relax (module doc, "Relax"): velocities zeroed and booked, the surrogate's
    /// result checked by its teacher when it converged, the drift history cleared once, the
    /// final readout kept. No-op without a running relax. `None` = stopped before an outcome.
    fn finish_relax(&mut self, outcome: Option<FireOutcome>, should_abort: &mut dyn FnMut() -> bool) {
        let Some(fire) = self.fire.take() else { return };
        let k0 = self.scene.kinetic_energy();
        for n in &mut self.scene.nuclei {
            n.vel = [0.0; 3];
        }
        self.work += self.scene.kinetic_energy() - k0;
        let converged = outcome == Some(FireOutcome::Converged);
        let p = *fire.params();
        let mut progress = self.relax.take().unwrap_or_else(|| RelaxProgress {
            steps: fire.steps(),
            max_force: self.forces.as_deref().map_or(0.0, max_norm),
            max_force_augmented: 0.0,
            energy: self.energy_ks,
            dt_fs: units::au_time_to_fs(fire.dt()),
            alpha: fire.alpha(),
            converged: None,
            verified_max_force: None,
            note: None,
        });
        progress.converged = Some(converged);
        let (steps, f_aug) = (progress.steps, progress.max_force_augmented);
        progress.note = match outcome {
            Some(FireOutcome::Converged) => None,
            Some(FireOutcome::Stalled) => Some(format!(
                "stalled after {steps} steps: max|F| did not improve by {:.0} % over {} steps (a soft mode on the grid's egg-box ripple?); ended at max|F_aug| {f_aug:.1e} Ha/Bohr",
                100.0 * p.stall_improvement,
                p.stall_window
            )),
            Some(FireOutcome::StepCap) => Some(format!("stopped at the {}-step cap with max|F_aug| {f_aug:.1e} Ha/Bohr (> {:.0e}): Relax again to continue", p.max_steps, p.f_tol)),
            Some(FireOutcome::Running) | None => Some(format!("stopped after {steps} steps at max|F_aug| {f_aug:.1e} Ha/Bohr")),
        };
        // a surrogate's result is checked by its teacher (spec §6.2) — one solve, unless the
        // last answer was the teacher's already
        if converged && self.forces.is_some() && self.provider.as_trainable().is_some() {
            if self.eval.energy_surface() == Surface::Model {
                self.provider.as_trainable().expect("checked").force_teacher_next();
                match self.provider.evaluate(&self.scene, should_abort) {
                    Ok(r) if r.forces.len() == self.scene.nuclei.len() => {
                        let vmax = max_norm(&r.forces);
                        // the positions did not move: a change of the reported energy is a
                        // surface flip at fixed positions, booked like a refit jump
                        self.work += r.energy - self.energy_ks;
                        self.forces = Some(r.forces);
                        self.energy_ks = r.energy;
                        self.eval = r.stats;
                        self.display = r.display;
                        self.last_epoch = self.provider.surface_epoch();
                        progress.verified_max_force = Some(vmax);
                        if vmax >= p.f_tol {
                            progress.note = Some(format!(
                                "relaxed on the model; DFT max|F| = {vmax:.1e} Ha/Bohr (≥ {:.0e}): Relax again to continue on the refitted model",
                                p.f_tol
                            ));
                        }
                    }
                    Ok(r) => progress.note = Some(format!("verification: provider '{}' returned {} forces for {} nuclei", self.provider.name(), r.forces.len(), self.scene.nuclei.len())),
                    Err(ProviderError::Cancelled) => progress.note = Some("verification by the teacher cancelled: the result is the model's".into()),
                    Err(e) => progress.note = Some(format!("verification by the teacher failed ({e}): the result is the model's")),
                }
            } else {
                progress.verified_max_force = Some(progress.max_force);
            }
        }
        // the drift is for dynamics: its history restarts here (W stays — the energy drained)
        self.history.clear();
        self.seq += 1;
        let summary = match (&progress.note, progress.verified_max_force) {
            (Some(n), _) => format!("relax ended ({n})"),
            (None, Some(v)) => format!("relax converged after {steps} steps: max|F| {:.1e} Ha/Bohr, verified by the teacher at {v:.1e}", progress.max_force),
            (None, None) => format!("relax converged after {steps} steps: max|F| {:.1e} Ha/Bohr", progress.max_force),
        };
        self.note(summary);
        self.relax = Some(progress);
    }

    // ---- internals ----

    /// Evaluates at the current positions (stale forces → fresh). After an edit the energy is
    /// a new level and the history restarts; after a surface refresh (module doc) the jump is
    /// booked instead and the history continues.
    fn evaluate(&mut self, should_abort: &mut dyn FnMut() -> bool) -> Result<(), ProviderError> {
        let r = self.provider.evaluate(&self.scene, should_abort)?;
        self.stamp_side_jobs(self.sim_time);
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
            self.work_refit += j;
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
        let tools = Tools { drag: self.drag, damping: self.drag_damping, force_max: self.drag_force_max, com: self.com_correction, walls: self.walls, pins: &self.pins };
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
        // an audit dispatched in the step's evaluation is at its new positions (module doc)
        self.stamp_side_jobs(self.sim_time + self.integrator.dt);
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
            source: self.eval.source,
            guess: self.eval.extrapolation.is_some(),
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

    /// Σ U_wall over the nuclei (Hartree), 0 without walls (module doc, "Container walls").
    fn walls_energy(&self) -> f64 {
        match &self.walls {
            Some(w) => neumaier(self.scene.nuclei.iter().map(|n| w.potential(n.pos))),
            None => 0.0,
        }
    }

    fn energy_conserved(&self) -> f64 {
        self.energy_ks + self.scene.kinetic_energy() + self.spring_energy() + self.field_energy() + self.walls_energy() - self.work
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
        self.work_refit = 0.0;
        self.walls_work = 0.0;
    }

    /// The expected cost of the next step (ms): the moving average of the latest step's kind,
    /// or of the other kind before any step of this one (`DynSnapshot::ms_per_step`).
    fn ms_next(&self) -> f64 {
        let (this, other) = if self.last_step_model { (self.ms_model, self.ms_dft) } else { (self.ms_dft, self.ms_model) };
        if this > 0.0 { this } else { other }
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
        self.side_display = None;
        self.restart_history();
        self.last_error = None;
        // not while relaxing: FIRE bypasses the thermostat, and a draw would hand the relax
        // velocities it did not ask for (module doc, "Relax")
        if self.fire.is_none() {
            self.thermalise_if_at_rest();
        }
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
    walls: Option<Walls>,
    pins: &'a [bool],
}

impl Tools<'_> {
    /// The `augment` of every step (module doc, "Tool forces"): COM correction on the provider
    /// forces, then the damped drag spring, the container walls, then the pins'
    /// zero-net-acceleration override.
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
        if let Some(w) = &self.walls {
            for (x, n) in f.iter_mut().zip(&scene.nuclei) {
                let fw = w.force(n.pos);
                for a in 0..3 {
                    x[a] += fw[a];
                }
            }
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
