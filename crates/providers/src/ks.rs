//! The two Kohn-Sham adapters behind the force boundary (`sim::ForceProvider`, CLAUDE.md rule
//! 5), both on the cubic grid of `scene.cell` with σ = `scene.sigma` or h and the provider's SCF
//! tolerance (moved out of `sim::provider` in M6, Task 1, so `sim` no longer depends on `qm`):
//!
//! - [`CpuKs`]: `qm::scf` + `qm::forces` (f64, the reference). Warm-starts from the previous
//!   converged state on the same grid. `qm::scf` has no cancellation point (and `qm` is the
//!   untouched reference), so `should_abort` is polled **once, before the solve**; a CPU solve
//!   that has started always runs to completion.
//! - [`GpuKs`]: `qm_gpu::GpuSolver::solve_with` + `GpuSolver::forces`. `should_abort` is polled
//!   before the solve and then before every SCF step (latency ≤ one step); an abandoned solve
//!   parks its in-progress state as the next warm start (`solve_with` doc). Density
//!   extrapolation across warm starts (ρ_in = 2ρ_t − ρ_{t−1}, `set_density_extrapolation`) is
//!   on: that is the MD case, consecutive equal time steps. The solver (kernels, Poisson
//!   operator; ~0.3 s and up to 227 MB at 97³) is rebuilt only when the grid changes.
//!
//! [`ForceProvider::discontinuity`] (a teleport while paused, ruling R13) keeps the warm state —
//! a nearby geometry's orbitals are a good start, as the M2 viewer's warm drags showed — but
//! drops the GPU's density-extrapolation history (2ρ_t − ρ_{t−1} across a jump would double the
//! density at the new site). The default is `reset`.
//!
//! [`ForceProvider::reset`] makes the next evaluation cold: the warm state is dropped (CPU) or
//! not used (GPU: the next solve runs with `warm = false`, which discards the parked state, and
//! the extrapolation history ρ_{t−1} is dropped at once), so a discontinuous edit never
//! warm-starts from — or extrapolates through — the old geometry. A grid change does the same
//! by itself.
//!
//! # Display payload
//! Each successful evaluation can carry the converged electron density for the viewer
//! ([`Display`]; on by default, `set_display(false)` turns it off): the CPU adapter as f32
//! values, the GPU adapter as a 3D texture filled on the device (`copy_density_to_volume`, no
//! CPU round trip) from a pool that never overwrites a texture someone still holds
//! ([`TexturePool`]). `sim` carries the payload opaquely — nothing here interprets it — so the
//! renderer still never learns what computed the density.
//!
//! # Orbital display (orbital-viewer spec §4.2–4.4; plan Task 3)
//! With an orbital view on (`ForceProvider::set_orbital_view`), a successful evaluation also
//! carries an `OrbitalDisplay` (`ForceResult::orbitals`) when the rate bound allows
//! (`crate::orbitals::ViewState`): [`CpuKs`] runs `qm::orbitals` in f64 on its `KsState`
//! (`crate::orbitals::CpuView`, the field as f32 values), [`GpuKs`] a
//! `qm_gpu::orbitals::OrbitalView` on its solver's device (one scalar readback, the field as a
//! texture from a pool per product kind, never overwriting one someone holds). Both run **after**
//! the energy and forces are final, read the converged state only and write only their own
//! fields and textures: the dynamics are bitwise the same with the view on or off
//! (`orbital_view_does_not_change_the_dynamics`). The producer is built at the first payload and
//! dropped when the view is switched off, on `reset` and on a grid change (spec §5.4 item 5: the
//! tracker restarts; the ids the boundary sees never repeat — `crate::orbitals`, "Track ids
//! across rebuilds"); a `discontinuity` keeps it (matching decides). `orbital_display_now`
//! answers on the converged state without a solve under the same condition as
//! `sample_density` (the latest evaluation succeeded, and no `reset` or `discontinuity` since).
//! Unoccupied orbitals (plan Task 4) come from the producer's display eigensolve on that same
//! converged state, paused in practice (`crate::orbitals`, "Unoccupied orbitals"); its SCF wall
//! accounting is the evaluation's solve and forces. A solver with a probe (`with_probe`) reports
//! each such eigensolve to it (`ActivityProbe::display_solve`: the Activity tab's job line and
//! log row; display only, not a teaching solve).
//!
//! Each payload's rows may carry their Hirshfeld shares on the atoms (orbital-atom-shares spec;
//! `crate::orbitals`, "Orbital shares"), against the free-atom tables the driver sends
//! (`ForceProvider::set_share_tables`, kept across `reset`: they are the scene's data, checked
//! against each payload's evaluated species). [`CpuKs`] computes them from its f64 ψ̃ at the
//! evaluated positions; [`GpuKs`] runs `OrbitalView::shares` on its displayed ψ̃ (one GPU pass and
//! one readback of the finished partials, after the update; the tables are packed once per change,
//! `ShareTablesGpu`) and brings those numbers to the same check and set average
//! (`crate::orbitals::settle_shares`). Both run inside the payload, i.e. only when a refresh is due
//! (`ViewState`'s rate bound) and after the step's energy and forces are final; the GPU pass costs
//! 0.45–0.5 ms at H₂O 95³ × 0.15 with 10 rows × 3 nuclei, +0.5 ms on the payload
//! (`gpu_shares_cost_is_reported`).
//!
//! # Harris display (`ForceProvider::harris_display`; live-orbitals spec §4)
//! [`GpuKs`] answers a Harris display request now (`HarrisAnswer::Solved`): orbitals at the scene's
//! positions in the potential of the superposed isolated atoms — one diagonalisation, not
//! self-consistent, display only. The steps: `problem` (the scene's grid, electrons and occupation
//! rule, the floor-aware tolerance as the SCF would run it); the solver for that grid (the rule
//! `evaluate` uses; a rebuild drops the view, the Harris state and the converged density); the GPU
//! memory guard ([`gpu_footprint_bytes`] of the scene — its pseudo extras included — the view's and
//! [`crate::harris::harris_footprint_bytes`]; over the
//! budget with m unoccupied orbitals but within it without, they are refused with a note; over it
//! without, the payload is a note and nothing is solved); ρ_in from the `HarrisAtoms` profiles that
//! fit the scene's surface ([`crate::harris::superposition`]: each atom renormalised to Z, or Z_v
//! under a pseudopotential, the sum rescaled to the scene's N — a refusal, N = 0 included, is a
//! note-only payload); then `qm_gpu::GpuSolver::harris` into the provider's own `HarrisState`, the
//! probe's display slot marked `DisplayJob::Harris` while it runs (never a solve slot), and the
//! payload through the converged payload's own tail on the Harris state (`OrbitalKind::Harris`, its
//! rescale, a note when |rescale − 1| > 1e-9; the setup's notes — an ionic fallback or refusal —
//! the solve's, and the superposition's sanity-bound notes). A cancelled solve is
//! `ProviderError::Cancelled`. It never sets `density_current`, never touches `last` or the SCF's warm
//! state, so `sample_density`, `orbital_display_now` and the next `evaluate` are what they would have
//! been (`harris_is_not_a_teacher_solve`).
//!
//! **One view, both kinds.** The Harris payload goes through the same `OrbitalView` as the converged
//! ones, so the tracks continue between them. Its unoccupied rows are the Harris solve's own
//! (`OrbitalView::adopt_unoccupied`, only when m > 0; otherwise the view shows none, since a converged
//! state's belong to another potential), marked in `UnoccupiedState` as a Harris solve's: the next
//! converged payload clears them before its own plan, so it shows its own eigensolve's rows or none,
//! never these (`converged_payload_after_harris_never_shows_harris_unoccupied`). The Harris state is
//! dropped when the view is switched off, on a solver rebuild, on `reset`, after a failed readback
//! (`drop_view_after_fault`: its warm vectors may hold the zeros too) and when the memory guard
//! refuses the unoccupied orbitals (the m = 0 solve is then within the budget it was checked
//! against).
//!
//! # Density samples (`ForceProvider::sample_density`, M6 spec §3.3)
//! Both adapters answer `Some` iff their latest `evaluate` succeeded and neither `reset` nor
//! `discontinuity` was called since — the density is always the geometry last evaluated; a
//! failed or cancelled evaluation, or a teleport, leaves `None` until the next success (the GPU
//! solver keeps its state as a warm start across a `reset`, which is exactly what must not be
//! handed out). [`CpuKs`]: `qm::Grid::trilinear` in f64 on the state's ρ↑ + ρ↓ (summed once per
//! state, cached). [`GpuKs`]: `qm_gpu::GpuSolver::sample_density` on the solver's padded `rho`
//! (the converged ρ_out, not the display texture: it works with `set_display(false)`), one
//! dispatch and one readback of `points.len()` f32 per call, widened to f64. Both use the cell
//! and weights of `qm::Grid::trilinear_cell`; they differ by f32 rounding (≤ 1e-6·max(1, ρ),
//! `qm_gpu::sample` tests).
//!
//! # A warm solve that stalls is retried cold (ruling R15)
//! The warm, density-extrapolated SCF can stagnate on a fast-changing density (the `h2o` preset
//! at 57³ stalls at Δρ ≈ 1e-4 on its second step). Both adapters therefore retry a **warm**
//! solve that returns `QmError::ScfNotConverged` — or `QmError::ScfStalled`, the SCF's stall stop
//! (`qm::scf` "The stall stop") — exactly once **cold** ([`retry_cold_on_stall`]:
//! CPU without the previous state; GPU with `warm = false` and density extrapolation off for
//! that solve — after the failure the solver holds no state and no ρ_{t−1} anyway). A converged
//! retry is returned with a note in [`EvalStats::notes`] ("warm SCF did not converge …; retried
//! cold"), its `scf_steps`/`readbacks` including the failed attempt's: the step costs about
//! twice as much, and the answer is a converged one. A cold solve that fails, or a failed
//! retry, is the error as before; any other error is never retried.
//!
//! # Pseudopotentials (M7 Task 9; spec §4.7, §5.5)
//! `Scene::ionic` (`sim::IonicPolicy`) decides how the nuclei enter. `problem` asks `crate::pseudo::setup_for`
//! for the scene's `qm::IonicSetup` — each species' pseudopotential generated on first use by the recipe from
//! this program's own atom, cached for the process, filtered for the scene's h, and **refused with the `h_req`
//! named** when the grid is coarser than the table is certified for (`ProviderError::Invalid`, before any solve).
//! Nothing pseudised (policy `None`; `CoresOnly` with only H/He or recipe-refused species, which fall back to
//! all-electron with a note) is the all-electron problem, `System::new`. With a setup the electron count is the
//! valence count (Z_v), and `charge` and `multiplicity` are checked against it.
//!
//! # Occupations: Fermi–Dirac on every scene (spec 2026-11-13 §6.1; plan Task 9)
//! Every scene, pseudised or not, solves with `Occupations::FermiDirac { kt: qm::ELECTRON_KT }` (kT = 5 mHa,
//! T_el ≈ 1579 K): the energy is the Mermin free energy F = E − TS (`EvalStats::energy`: `internal` = E, `ts`, `total`
//! = F) and the forces are −dF/dR. The all-electron surface is no longer today's aufbau bit for bit: every channel now
//! solves its witness vectors (the levels above μ + 36 kT, `qm::scf` "Occupations"), so the all-electron bitwise pins
//! were re-pinned with that reason, and the all-electron tag moved with it (`crate::pseudo::all_electron_tag`). A
//! gapped state's TS falls as e^−(ε_LUMO − μ)/kT: ≤ 1e-12 Ha only once ε_LUMO − μ ≳ 25 kT (H₂ at R_e: 4e-20), while a
//! narrower gap leaves a small TS (LiH at 57³ × 0.25 ≈ 2.4e-6 Ha, H₃⁺ stretched to 1.6 r₀ ≈ 2.8e-8 Ha); F equals the
//! aufbau E to that TS plus the solver's scatter.
//!
//! Each result then carries `EvalStats::{pseudised, core_overlaps, partial}`: the species' generation records, the pairs
//! whose frozen cores overlap (spec §5.6, never trained on) and the partly occupied levels of the final step
//! (min(f, 1 − f) ≥ `qm::PARTIAL_OCCUPATION_MIN`, `sim::PartialLevel`). The `branch_jump` guard below replaced the
//! shared-shell occupation watch (plan Task 9): FD has no shared-shell pattern to switch, and a switch of stationary
//! state shows as F changing by more than the forces' work. The GPU adapter stops its SCF at `max(tol, 2·floor(N))`
//! under a setup (spec §7 risk 6: the f32 density floor grows with the valence electron count N; noted in
//! `EvalStats::notes`), and on a step whose occupations have a partly occupied level at max(that, `qm::GPU_PARTIAL_TOL`)
//! (spec 2026-11-13 §5.1, the solver's own rule). Every answer records the tolerance its converged step was tested
//! against, `EvalStats::scf_tol_used` = `qm::ScfStats::tol_used` (F9: never compared for provenance — the surrogate's
//! provenance carries the stop *rules* in its ionic tag, the nominal tolerance as before; the guard picks its noise
//! tier from it).
//!
//! Under a setup, `set_display_partial_core` adds ρ_pc to the
//! density shown — a display of the generator's core model, computed after the solve, never fed back.
//! [`gpu_footprint_bytes`] counts the valence electrons and [`pseudo_extra_bytes`] under a setup.
//!
//! # The branch_jump guard (spec 2026-11-13 §6.3; plan Task 8)
//! Each adapter holds a [`crate::branch_jump::BranchJumpGuard`] and judges every successful evaluation against *its own
//! previous one*: ΔF against the forces' trapezoid work, within 3σ of the surface's path scatter at the tolerance the
//! solve stopped at (`ScfStats::tol_used`, `branch_jump::noise_tier`) plus 3κ_max‖ΔR‖³. A flagged step carries
//! `EvalStats::branch_jump = Some(m)` and the warning `sim::branch_jump_warning(m)`; the surrogate never trains on it
//! (`EvalStats::untrainable`). Under the surrogate "previous" is the last teacher call (as it was for the occupation
//! watch the guard replaced), so only fallback streaks are judged (consecutive calls ≤ 0.1 Bohr apart): sampling-pass solves and sparse
//! audits are not covered (R-NOTE-b). The guard forgets on `reset` (every edit, so every scene replace) and on a grid
//! change, and restarts by itself on any change of species, atom count, cell or origin; `discontinuity` keeps it (a
//! teleport of more than 0.1 Bohr is not judged anyway). With N↑ = N↓ a judged step also reports
//! `EvalStats::spin_change`, |Δ∫|ρ↑ − ρ↓|| — the CPU from `qm::orbitals::spin_summary`, the GPU from
//! `GpuSolver::spin_abs_integral` (one extra reduction and readback per such evaluation). The guard's surface (pseudo
//! or all-electron tier) is `system.ionic.is_none()`, which is exactly `crate::pseudo::is_all_electron_tag` of the
//! scene's provenance tag (`setup_for` and the tag share one per-species rule; pinned by
//! `the_guard_surface_is_the_tag_surface`), read without taking the pseudopotential cache's lock.
//!
//! # Errors
//! - [`ProviderError::Cancelled`]: `should_abort` returned true (the GPU keeps the in-progress
//!   state as its warm start).
//! - [`ProviderError::Solver`]: the solver's own errors — coincident nuclei (raised before any
//!   work), no SCF convergence, and, GPU only, `QmError::Numerical` (non-finite numerics; the
//!   solver drops its state and the next evaluation is cold) and `QmError::Device` ([`SolverError::Device`],
//!   "GPU busy": a GPU → CPU readback the shared device did not deliver — see `gpu::readback`;
//!   [`GpuKs`] repeats the evaluation up to [`DEVICE_RETRIES`] times after a short pause, notes
//!   a retry that succeeded, and reports the error only when they run out; the worker shows it
//!   and pauses, and Play or any edit tries again). After an error the CPU provider
//!   holds no warm state either.
//! - [`ProviderError::Invalid`]: the scene cannot be posed — a charge above the proton count, a
//!   multiplicity impossible for the electron count (what `qm::System::new` would panic on), or
//!   a nucleus closer than `CELL_MARGIN_H`·h to a cell face or outside the cell (R12, `Scene::check_inside_cell`: the
//!   density is cut off there, so the forces would be artefacts of the box). No `QmError`
//!   variant fits these: they are input errors, not solver outcomes.
//!
//! An empty scene (no nuclei) is answered without a solve, by both adapters, through
//! `sim::provider::nothing_to_solve` (its module doc, "Nothing to solve").
//!
//! # GPU memory (main's fix round 1, review I2; moved here from `sim::provider` with the providers)
//! A wgpu allocation that does not fit panics through the device's uncaptured-error handler, so
//! [`GpuKs`] can be given a budget (`with_vram_budget`) and then refuses a scene whose estimated
//! footprint ([`gpu_footprint_bytes`]) exceeds it — `ProviderError::Invalid`, before the solver is
//! built — and one whose single field would exceed the device's storage-binding or buffer limit.
//! The estimate follows `qm_gpu::scf`'s "Fields and VRAM": per solve K = N orbitals and K kinetic
//! scratch fields, 3 densities + V_H + V_nuc, ~8 working fields, 24 DIIS fields (history 6), the
//! Poisson multigrid (≈ 60 field-equivalents incl. its coarse levels), `LobpcgWork` (8·k_max + 2),
//! plus the display texture pool and the extrapolation history (4); each field padded with
//! `qm_gpu::grid::GHOST` = 4 ghost layers per side, (n + 2·GHOST)³ f32;
//! × 1.2 for what the list misses (uniforms, readback buffers, allocator slack). wgpu reports no
//! total VRAM, so the caller picks the budget (the app: `quality::vram_budget` — half the card's
//! memory when the adapter reports it, `gpu::GpuContext::device_local_memory_bytes`) and may say
//! where it came from (`with_vram_budget_note`, printed with the budget in a refusal).
//! **The orbital view** (orbital-viewer plan Task 4, Task 3 review ruling 4) is not charged to the
//! scene — that would refuse scenes that fit without it. It is checked when a payload is made:
//! the solve's footprint plus [`orbital_view_footprint_bytes`] (the view's display fields — the
//! shown rows, the scratch, and with m unoccupied orbitals the kept vectors, a display solve's
//! new vectors (the kept ones stay intact until it completes, for an abort), occupied copies and
//! potentials — the solver's `LobpcgWork` grown by 8(m + 2) fields, the row
//! buffer, and the product textures ×3 for the pool, the colour accumulator included; ×1.2 as
//! above). Over the budget with m unoccupied orbitals but within it without them, the payload
//! shows the occupied rows and says why the unoccupied ones were refused; over it even without,
//! the view's producer is dropped and the payload is a note only. At 97³ (4.63 MB a field) H₂O
//! with m = 4 adds ≈ 0.59 GB to its 0.91 GB solve (`the_orbital_view_is_charged_to_the_vram_budget`). A Harris
//! display solve adds [`crate::harris::harris_footprint_bytes`] on top of both (module doc "Harris display"). The estimated mode's
//! producer lives on the auditor's teacher, whose budget is half the app's (`app::worker`).
//!
//! The budget carries over to [`ForceProvider::side_solver`]'s second `GpuKs`, which checks it
//! against its **own** scene only — one atom at the stepping scene's grid, so up to one more
//! scene-sized footprint (fewer orbitals: ≈ 100 field-equivalents, ~1 GB at Precise 121³) while a
//! reference job runs (M6 final review M8). The budget is meant to leave that headroom: the
//! app's default 12 GB is half the 24 GB card, and a budget set near the device's size with
//! `ATOMIC_SANDBOX_VRAM_GB` does not leave it. The driver drops the side solver once its
//! reference queue is empty (`sim::dynamics`), so the second set is held only while references
//! are being solved.
use crate::branch_jump::{noise_tier, BranchJumpGuard, Evaluation};
use crate::harris::{harris_footprint_bytes, superposition};
use crate::orbitals::{spin_in, CpuView, PickKind, Pool, TextureKind, UnoccupiedPlan, UnoccupiedState, ViewChange, ViewState, UNOCCUPIED_MAX};
use crate::surrogate::box_affected_face_density;
use gpu::{ColourVolume3D, GpuContext, Volume3D};
use qm::{Grid, ScfSettings, ScfStats, System};
use qm_gpu::orbitals::{OrbitalView, Product, ShareTablesGpu};
use qm_gpu::{HarrisInput, HarrisState};
use sim::{
    nothing_to_solve, ActivityProbe, AtomProfile, Cell, Display, DisplayInterrupt, DisplayJob, EnergyBreakdown, EvalStats, ForceProvider, ForceResult,
    HarrisAnswer, HarrisAtoms, OrbitalDisplay, OrbitalField, OrbitalKind, OrbitalViewRequest, ProviderError, RadialDensity, Scene, ShareTables, Slot,
    SolverError, Source, Weights,
};
use std::sync::Arc;
use std::time::Instant;

/// `qm`'s errors as the boundary's [`SolverError`] (variant for variant, the same messages).
fn solver_error(e: qm::QmError) -> ProviderError {
    ProviderError::Solver(match e {
        qm::QmError::CoincidentNuclei(i, j) => SolverError::CoincidentNuclei(i, j),
        qm::QmError::ScfNotConverged(n, d) => SolverError::ScfNotConverged(n, d),
        qm::QmError::ScfStalled { iterations, best_delta_rho, tol, window, level } => SolverError::ScfStalled { iterations, best_delta_rho, tol, window, level },
        qm::QmError::Numerical(m) => SolverError::Numerical(m),
        qm::QmError::Device(m) => SolverError::Device(m),
    })
}

/// A failed GPU readback latched on this thread since the last check, as the boundary's error
/// (`qm_gpu::grid::take_readback_fault`). `GpuSolver`'s display-side calls (`density_max`,
/// `sample_density`, the orbital view) return numbers rather than `Result`s; zeros stand in for
/// a failed readback and this is how the provider learns not to trust them.
fn device_fault() -> Result<(), ProviderError> {
    match qm_gpu::grid::take_readback_fault() {
        Some(f) => Err(ProviderError::Solver(SolverError::Device(f.to_string()))),
        None => Ok(()),
    }
}

/// Failed attempts of one evaluation that [`retry_on_device_fault`] repeats before giving up.
const DEVICE_RETRIES: usize = 2;
/// The pause before retry i (ms): contention clears in tens of milliseconds, a resize or another
/// process's job in a few hundred.
const DEVICE_BACKOFF_MS: [u64; DEVICE_RETRIES] = [100, 400];

/// The "GPU busy" policy: `attempt` runs; a [`SolverError::Device`] (a readback that the shared
/// device did not deliver) is retried after a pause, up to [`DEVICE_RETRIES`] times, `pause(ms)`
/// returning false when the evaluation was cancelled meanwhile. Every other outcome is the first
/// attempt's, untouched. Returns the value and the number of failed attempts before it; the
/// final error names the attempts.
fn retry_on_device_fault<T>(mut attempt: impl FnMut() -> Result<T, ProviderError>, mut pause: impl FnMut(u64) -> bool)
                            -> Result<(T, usize), ProviderError> {
    let mut failed = 0;
    loop {
        match attempt() {
            Ok(v) => return Ok((v, failed)),
            Err(ProviderError::Solver(SolverError::Device(m))) => {
                if failed == DEVICE_RETRIES {
                    return Err(ProviderError::Solver(SolverError::Device(format!("{m}; gave up after {} attempts", failed + 1))));
                }
                if !pause(DEVICE_BACKOFF_MS[failed]) {
                    return Err(ProviderError::Cancelled);
                }
                failed += 1;
            }
            Err(e) => return Err(e),
        }
    }
}

/// Sleeps `ms` in short slices so a cancellation is noticed within ~20 ms; false if cancelled.
fn pause_unless_aborted(ms: u64, should_abort: &mut dyn FnMut() -> bool) -> bool {
    let end = Instant::now() + std::time::Duration::from_millis(ms);
    while Instant::now() < end {
        if should_abort() {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    !should_abort()
}

/// The estimated GPU memory (bytes) of one [`GpuKs`] solve of `scene` (module doc, "GPU memory");
/// 0 for an empty scene (nothing is solved).
///
/// Under an ionic policy with something pseudised (M7 Task 9; the Task 8a review SHOULD-5 and the Task 8b ruling
/// (c)) the electron count is the valence count and the pseudo path's extras are added
/// ([`pseudo_extra_bytes`]); every other scene is counted exactly as before. It never generates a pseudopotential
/// and never waits for the cache (`crate::pseudo::footprint_inputs`; the viewer asks on its render thread): a species
/// whose table is not generated yet is counted at the bound (all Z electrons, a generic ion), so the answer is
/// conservative until the worker has generated it, then exact.
pub fn gpu_footprint_bytes(scene: &Scene) -> u64 {
    footprint_bytes_in(crate::pseudo::cache(), scene)
}

/// [`gpu_footprint_bytes`] on a given pseudopotential cache (tests use a private one).
pub fn footprint_bytes_in(cache: &std::sync::Mutex<crate::pseudo::PseudoCache>, scene: &Scene) -> u64 {
    if scene.nuclei.is_empty() {
        return 0;
    }
    // never generates a pseudopotential and never waits (`crate::pseudo::footprint_inputs`): the guard runs on the
    // render thread; a table not yet generated is counted at its bound
    let fp = crate::pseudo::footprint_in(cache, scene);
    let n_el = fp.n_valence.max(0) as u64;
    let unpaired = match scene.multiplicity {
        Some(m) => (m as u64).saturating_sub(1).min(n_el),
        None => n_el % 2,
    };
    let k_max = (n_el + unpaired).div_ceil(2);
    let base = solve_footprint_bytes(scene.cell.n, n_el, k_max);
    if fp.pseudised {
        base + pseudo_extra_bytes(scene.cell.n, scene.cell.h, k_max, &fp.ions)
    } else {
        base + witness_extra_bytes(scene.cell.n)
    }
}

/// The witness vectors the scene guard counts per spin: what the solver allocates at the start — the measured peak of
/// tested witness levels, 2, plus the buffer column (`qm_gpu::scf` "The cold-start window"; plan 2026-11-13 fix round 2,
/// Decision 1). Measured with the 9c cold start at the app's 14 Bohr all-electron H₂O box (167³ × 0.0848, the 177³
/// case's box): a peak block of n_σ + 3, no growth. A scene is refused unless its solve fits with this many; the
/// solver's cap is then raised toward `qm::scf::MAX_WITNESS_GROWTH` as far as the remaining budget allows
/// (`witness_cap_for`), and a solve that needs more than its cap ends with the cap error naming the budget — never an
/// out-of-memory. (Task 9b counted 16, a worst-case guess from the blob start's transient windows of 16–20; 9c measured
/// the start-window cause away. Fewer fields per vector — 12 now, 0.73 GB per vector at 177³ in estimated mode — is a
/// later optimisation.)
pub const WITNESS_GROWTH_COUNTED: usize = 3;

/// The witness growth's fields at `w` extra vectors per spin: the one `LobpcgWork` (sized for the larger spin) grows by
/// 8 fields a vector, and each spin stores that many more orbitals and kinetic scratch fields: (8 + 2 + 2)·w fields (a
/// bound — most solves grow a few).
fn witness_growth_fields(w: usize) -> u64 { 12 * w as u64 }

/// The witness growth at `w` extra vectors per spin on an n³ grid (`witness_growth_fields`, with the 1.2 slack of
/// [`solve_footprint_bytes`]).
pub fn witness_bytes(n: usize, w: usize) -> u64 {
    (witness_growth_fields(w) as f64 * field_bytes(n) as f64 * 1.2) as u64
}

/// The witness growth of an all-electron solve under `FermiDirac` at the counted `WITNESS_GROWTH_COUNTED` (36 fields):
/// the scene guard's term (plan 2026-11-13 Task 7 fix rounds 1–2; the count since fix round 2 of Task 9c).
pub fn witness_extra_bytes(n: usize) -> u64 { witness_bytes(n, WITNESS_GROWTH_COUNTED) }

/// The `FermiDirac` witness cap a VRAM `budget` leaves `scene` (plan 2026-11-13 Task 9b ruling: min(32, what fits)): the
/// largest w ≤ `qm::scf::MAX_WITNESS_GROWTH` for which the scene's solve with w extra vectors per spin **plus** the
/// orbital view and the Harris display at their largest request (`UNOCCUPIED_MAX` rows, the colour textures, the
/// Harris block at w) fit, so every display guard at any request passes whenever the cap is above the counted start;
/// never below `WITNESS_GROWTH_COUNTED` (the scene guard has accepted that much, and the display guards count it as
/// before). Growth past the cap is the solver's named cap error (`GpuSolver::set_witness_cap`), never an allocation.
pub fn witness_cap_for(scene: &Scene, budget: u64) -> usize {
    let n = scene.cell.n;
    let fp = crate::pseudo::footprint_inputs(scene);
    let n_el = fp.n_valence.max(0) as usize;
    let unpaired = match scene.multiplicity { Some(m) => (m as usize).saturating_sub(1).min(n_el), None => n_el % 2 };
    let n_up = (n_el + unpaired).div_ceil(2);
    let n_dn = n_el - n_up;
    let base = gpu_footprint_bytes(scene).saturating_sub(witness_extra_bytes(n));
    let m = UNOCCUPIED_MAX as usize;
    let fits = |w: usize| {
        base + witness_bytes(n, w) + orbital_view_footprint_bytes(n, n_up, n_dn, m, TextureKind::Colour)
            + harris_footprint_bytes(n, n_up, n_dn, m, fp.pseudised, w) <= budget
    };
    (WITNESS_GROWTH_COUNTED + 1..=qm::scf::MAX_WITNESS_GROWTH).take_while(|&w| fits(w)).last().unwrap_or(WITNESS_GROWTH_COUNTED)
}

/// The device-limit refusal of an n³ solve (`GpuKs::check_vram`), or `None` when it fits: its largest single storage
/// binding — one padded field, or the Poisson boundary table (`qm_gpu::poisson::ghost_table_bytes_for`, 4704·n² B, the
/// larger from n ≈ 169 at wgpu's default 128 MiB; task 9d review m5) — against the device's binding and buffer limits.
pub fn binding_refusal(n: usize, max_binding: u64, max_buffer: u64) -> Option<String> {
    let limit = max_binding.min(max_buffer);
    let (field, table) = (field_bytes(n), qm_gpu::poisson::ghost_table_bytes_for(n));
    let (what, bytes) = if table > field { ("its Poisson boundary table", table) } else { ("one field", field) };
    (bytes > limit).then(|| format!("{n}³: {what} is {:.0} MB, over this GPU's {:.0} MB buffer limit: choose a coarser quality (Scene ▸ Quality)",
                                    bytes as f64 / 1e6, limit as f64 / 1e6))
}

/// [`gpu_footprint_bytes`] from the counts: n³ grid, N electrons, k_max = max(n↑, n↓).
pub fn solve_footprint_bytes(n: usize, n_el: u64, k_max: u64) -> u64 {
    let fields = 2 * n_el + 5 + 8 + 24 + 60 + (8 * k_max + 2) + 4;
    (fields as f64 * field_bytes(n) as f64 * 1.2) as u64
}

/// The extra GPU memory (bytes) of a pseudo solve on an n³ grid of spacing h with k_max = max(n↑, n↓) occupied
/// orbitals, for the pseudised ions given as (r, rows) — the radius of the ion's projector/partial-core ball in
/// Bohr and its number of projector rows Σ(2l + 1) — with the 1.2 slack of [`solve_footprint_bytes`] applied:
/// - ρ_pc and the two XC input fields ρ_σ + ½ρ_pc (Task 8a review SHOULD-5): 3 fields;
/// - the witness vectors at the counted `WITNESS_GROWTH_COUNTED` = 3 per spin (the solver's start allocation; its cap
///   above it comes from the budget, `witness_cap_for`): the one `LobpcgWork` (sized for the larger spin) grows by 8
///   fields a vector, and each spin stores that many more orbitals and kinetic scratch fields: (8 + 2 + 2)·3 fields
///   (`witness_growth_fields`);
/// - the v̄ scratch of the force path (Task 8b ruling (c)): 6 fields;
/// - the display copy of ρ + ρ_pc (`GpuSolver::copy_density_with_core_to_volume`): 1 field;
/// - the force weights, Σ_rows n_pts × 20 B with n_pts = (4π/3)(r/h)³ points in the ball (Task 8b ruling (c));
/// - the force partials and their staging pair, 2·16 B × (wg·n_nuc + ch·rows·n_orb) with wg = 256 workgroups per
///   reduction, ch = 64 chunks per row and n_orb = 2·k_max orbitals (Task 8b ruling (c); generous: < 10 MB for
///   tens of ions).
/// The `qm_gpu::ionic` projector buffers are the force weights' sibling (the same point lists) and are inside
/// the weights term.
pub fn pseudo_extra_bytes(n: usize, h: f64, k_max: u64, ions: &[(f64, u64)]) -> u64 {
    let fields = 3 + witness_growth_fields(WITNESS_GROWTH_COUNTED) + 6 + 1;
    let mut bytes = fields * field_bytes(n);
    let rows: u64 = ions.iter().map(|i| i.1).sum();
    for &(r, rws) in ions {
        let n_pts = (4.0 / 3.0 * std::f64::consts::PI * (r / h).powi(3)).ceil() as u64 + 1;
        bytes += rws * n_pts * 20;
    }
    const WG: u64 = 256;
    const CH: u64 = 64;
    bytes += 2 * 16 * (WG * ions.len() as u64 + CH * rows * 2 * k_max.max(1));
    (bytes as f64 * 1.2) as u64
}

/// The orbital view's GPU memory (bytes) on top of a solve with n↑, n↓ occupied orbitals on an
/// n³ grid, with m unoccupied orbitals per channel and the product textures of `texture` (module
/// doc, "GPU memory"): `OrbitalView::display_fields` + `OrbitalView::work_growth_fields` + the row
/// buffer (one field) padded fields, plus three textures of the kind (the pool's settled size; an
/// r32float n³ volume, or an rgba16float one and the colour accumulator), × 1.2.
pub fn orbital_view_footprint_bytes(n: usize, n_up: usize, n_dn: usize, m: usize, texture: TextureKind) -> u64 {
    let fields = OrbitalView::display_fields(n_up + n_dn, n_up.max(n_dn), m) + OrbitalView::work_growth_fields(m) + 1;
    let voxels = (n as u64).pow(3);
    let textures = match texture {
        TextureKind::None => 0,
        TextureKind::Scalar => 3 * voxels * 4,
        TextureKind::Colour => 3 * voxels * 8 + voxels * 16,
    };
    ((fields as u64 * field_bytes(n) + textures) as f64 * 1.2) as u64
}

/// One padded f32 field of an n³ grid as `qm_gpu` allocates it: (n + 2·GHOST)³ × 4 bytes
/// (`qm_gpu::grid::GHOST` ghost layers per side for the 8th-order stencil).
pub fn field_bytes(n: usize) -> u64 {
    (n as u64 + 2 * qm_gpu::grid::GHOST as u64).pow(3) * 4
}

/// Pre-validates what `System::new` / `System::with_ionic` would panic on and builds the solve's inputs: the grid
/// of `scene.cell`, the system (the electrons the ions leave, the multiplicity) and the SCF settings
/// (`ScfSettings::for_grid` with `tol` and σ = `scene.sigma` or h).
///
/// Under an ionic policy (M7; module doc, "Pseudopotentials") the scene's species are first turned into a
/// `qm::IonicSetup` by [`crate::pseudo::setup_for`] — generation on first use, the `h_req` check, the fallback
/// notes (pushed to `notes`) — and the electron count is the valence count: Z_v for a pseudised species, Z for
/// the rest, minus the charge. When nothing is pseudised (policy `None`; `CoresOnly` with only H/He or refused
/// species) the setup is `None` and the problem is the all-electron one. Every scene's occupations are
/// `FermiDirac { kt: qm::ELECTRON_KT }` (spec 2026-11-13 §6.1, plan Task 9; module doc, "Occupations").
pub fn problem(scene: &Scene, tol: f64, notes: &mut Vec<String>) -> Result<(Grid, System, ScfSettings), ProviderError> {
    scene.check_inside_cell()?;
    let grid = Grid::cube(scene.cell.n, scene.cell.h);
    let nuclei: Vec<qm::Nucleus> = scene.nuclei.iter().map(|n| qm::Nucleus { pos: n.pos, z: n.z }).collect();
    let mut settings = ScfSettings::for_grid(&grid);
    settings.tol = tol;
    // every scene, whatever its ionic policy (spec 2026-11-13 §6.1, G2); `for_grid`'s default too, stated here because the
    // providers' surface (its provenance tag) is defined by it
    settings.occupations = qm::Occupations::FermiDirac { kt: qm::ELECTRON_KT };
    if let Some(s) = scene.sigma {
        settings.sigma = s;
    }
    let setup = crate::pseudo::setup_for(scene, settings.sigma, notes)?;
    let protons: i64 = nuclei.iter().map(|n| n.z as i64).sum();
    let valence = qm::valence_electrons(&nuclei, setup.as_deref(), 0);
    let total = valence - scene.charge as i64;
    if total < 0 {
        return Err(ProviderError::Invalid(if setup.is_some() {
            format!("charge {} exceeds the {valence} valence electrons the pseudopotentials leave ({protons} protons; the cores hold the rest)", scene.charge)
        } else {
            format!("charge {} exceeds the {protons} protons", scene.charge)
        }));
    }
    if let Some(m) = scene.multiplicity {
        let n = total as usize;
        if m < 1 || (m as usize - 1) > n || !(n - (m as usize - 1)).is_multiple_of(2) {
            return Err(ProviderError::Invalid(format!("multiplicity {m} impossible for {n} electrons")));
        }
    }
    let system = match setup {
        Some(setup) => System::with_ionic(nuclei, scene.charge, scene.multiplicity, setup),
        None => System::new(nuclei, scene.charge, scene.multiplicity),
    };
    Ok((grid, system, settings))
}

/// The GPU's floor-aware SCF stop (spec §7 risk 6; ionic setups only — the all-electron path is
/// untouched): the f32 storage rounding of ρ puts a floor under Δρ that grows with the electron
/// count, so `settings.tol` is raised to `crate::pseudo::effective_gpu_tol`. Returns the note when
/// it was raised.
fn raise_to_gpu_floor(tol: f64, system: &System, settings: &mut ScfSettings) -> Option<String> {
    system.ionic.as_ref()?;
    let n_valence = (system.n_up + system.n_dn) as i64;
    let tol_eff = crate::pseudo::effective_gpu_tol(tol, n_valence);
    (tol_eff > settings.tol).then(|| {
        let note = format!(
            "GPU SCF tolerance raised from {:.1e} to {tol_eff:.1e}: twice the f32 density floor of {n_valence} valence electrons ({:.1e})",
            settings.tol,
            crate::pseudo::gpu_tol_floor(n_valence)
        );
        settings.tol = tol_eff;
        note
    })
}

/// A warm solve that failed before the cold retry (R15): its SCF iterations and Δρ — the final Δρ
/// at the cap, the best Δρ when the stall stop ended it (`stalled`).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Stall {
    pub iterations: usize,
    pub delta_rho: f64,
    pub stalled: bool,
}

impl Stall {
    fn note(&self) -> String {
        if self.stalled {
            format!("warm SCF stalled (Δρ best {:.2e}, not improved 2× in the stall window; stopped after {} iterations); retried cold", self.delta_rho, self.iterations)
        } else {
            format!("warm SCF did not converge (Δρ {:.2e} after {} steps); retried cold", self.delta_rho, self.iterations)
        }
    }
}

/// The R15 policy (module doc), separated from the solvers so it is testable without one:
/// `solve(w)` runs one SCF solve, warm iff `w`. The first attempt runs with `warm`; if that was
/// a warm attempt and it returned `ScfNotConverged` or `ScfStalled` (the stall stop, `qm::scf`), `solve(false)` runs exactly once and its
/// outcome is returned, with the stall when it converged. Every other outcome — success, a
/// cold failure, a non-convergence error of another kind — is the first attempt's, untouched.
fn retry_cold_on_stall<T>(warm: bool, mut solve: impl FnMut(bool) -> Result<T, qm::QmError>)
                                    -> Result<(T, Option<Stall>), qm::QmError> {
    match solve(warm) {
        Err(qm::QmError::ScfNotConverged(iterations, delta_rho)) if warm => {
            solve(false).map(|v| (v, Some(Stall { iterations, delta_rho, stalled: false })))
        }
        Err(qm::QmError::ScfStalled { iterations, best_delta_rho, .. }) if warm => {
            solve(false).map(|v| (v, Some(Stall { iterations, delta_rho: best_delta_rho, stalled: true })))
        }
        other => other.map(|v| (v, None)),
    }
}

/// `qm`'s per-iteration record as the probe's (`sim` does not depend on `qm`).
fn iteration_of(p: &qm::ScfProgress) -> sim::ScfIteration {
    sim::ScfIteration {
        iteration: p.iteration, max_iter: p.max_iter, delta_rho: p.delta_rho, tol: p.tol, energy: p.energy, since_gain: p.since_gain, window: p.window,
        best_delta_rho: p.best_delta_rho, stall_level: p.stall_level,
    }
}

fn elapsed_ms(t0: Instant) -> u64 {
    t0.elapsed().as_millis() as u64
}

/// The `EvalStats` fields that come straight from `ScfStats` — the energy breakdown with E, TS and F, the partly
/// occupied levels, and the tolerance the converged step was tested against (`ScfStats::tol_used`, F9: the GPU's
/// partial-level stop shows as 1e-5) — plus the M7 ones: the pseudised species' summaries (from `setup`) and the
/// core overlaps — and the per-spin `frontier` levels of the converged state ([`frontier`]).
fn eval_stats(stats: ScfStats, readbacks: Option<usize>, t0: Instant, notes: Vec<String>, eig_tol_used: f64, system: &System, frontier: [sim::Frontier; 2]) -> EvalStats {
    let setup = system.ionic.as_deref();
    let ScfStats { iterations, energy, boundary_density, max_eigen_residual, poisson_residual, warnings, core_overlaps, partial, tol_used, mu, .. } = stats;
    // each channel's μ_σ beside its frontier levels (stray-atoms Task 7, controller ruling 10a)
    let mut frontier = frontier;
    for (f, m) in frontier.iter_mut().zip(mu) {
        f.mu = m;
    }
    EvalStats {
        source: Source::Dft,
        uncertainty: None,
        teacher_gap: None,
        refit_jump: None,
        model_forces: None,
        scf_steps: iterations,
        readbacks,
        solve_ms: elapsed_ms(t0),
        warnings,
        notes,
        face_density: Some(boundary_density),
        // n↑ ≥ n↓ by construction (`qm::scf::spin_split`): 2S + 1 of the state solved
        multiplicity: Some((system.n_up - system.n_dn + 1) as u32),
        eig_tol_used: Some(eig_tol_used),
        scf_tol_used: Some(tol_used),
        energy: EnergyBreakdown {
            kinetic: energy.kinetic,
            external: energy.external,
            hartree: energy.hartree,
            xc: energy.xc,
            nuclear: energy.nuclear,
            internal: energy.internal,
            ts: energy.ts,
            total: energy.total,
        },
        max_eigen_residual,
        poisson_residual,
        atom_uncertainty: None,
        extrapolation: None,
        pseudised: setup.map(crate::pseudo::summaries).unwrap_or_default(),
        core_overlaps,
        partial: partial.into_iter().map(|p| sim::PartialLevel { spin: p.spin, index: p.index, eps: p.eps, f: p.f }).collect(),
        frontier,
        branch_jump: None,
        spin_change: None,
    }
}

/// One spin channel's frontier levels (`EvalStats::frontier`) from the converged state's eigenvalues and their
/// occupations (one f per stored orbital, `qm::KsState` / `qm_gpu::GpuState`): the highest ε with f ≥ ½ and the lowest
/// with f < ½. Under `Aufbau` (exactly N_σ orbitals, all f = 1) `lumo` is `None`.
fn frontier(evals: &[f64], occ: &[f64]) -> sim::Frontier {
    let (mut homo, mut lumo): (Option<f64>, Option<f64>) = (None, None);
    for (&e, &f) in evals.iter().zip(occ) {
        if f >= 0.5 {
            homo = Some(homo.map_or(e, |h| h.max(e)));
        } else {
            lumo = Some(lumo.map_or(e, |l| l.min(e)));
        }
    }
    sim::Frontier { homo, lumo, mu: None }
}

/// The `branch_jump` guard on one finished evaluation (module doc, "The branch_jump guard"; spec 2026-11-13 §6.3):
/// judged against the same provider's previous evaluation on the tier of the surface and of the tolerance the solve
/// stopped at (`EvalStats::scf_tol_used` = `ScfStats::tol_used`); a flagged step gets `EvalStats::branch_jump` and the
/// spec §6.6 warning, a judged one at N↑ = N↓ the spin measure. `spin_abs`: ∫|ρ↑ − ρ↓| when N↑ = N↓, else `None`.
fn judge_branch(guard: &mut BranchJumpGuard, stats: &mut EvalStats, scene: &Scene, grid: &Grid, system: &System, forces: &[[f64; 3]], spin_abs: Option<f64>) {
    let tol_used = stats.scf_tol_used.expect("a Kohn-Sham evaluation records the tolerance it stopped at");
    let z: Vec<u8> = scene.nuclei.iter().map(|n| n.z).collect();
    let positions: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
    let j = guard.judge(&Evaluation {
        z: &z,
        positions: &positions,
        forces,
        energy: stats.energy.total,
        cell: scene.cell.clone(),
        origin: grid.origin,
        spin_abs,
        // the all-electron surface iff nothing is pseudised, i.e. iff `is_all_electron_tag(provenance_tag(scene))` (module
        // doc; `the_guard_surface_is_the_tag_surface`), without the cache lock
        tier: noise_tier(system.ionic.is_none(), tol_used, stats.energy.total),
    });
    stats.branch_jump = j.branch_jump();
    stats.spin_change = j.spin_change;
    if let Some(m) = stats.branch_jump {
        stats.warnings.push(sim::branch_jump_warning(m));
    }
}

/// The CPU Kohn-Sham reference behind the force boundary (module doc).
pub struct CpuKs {
    settings_tol: f64,
    /// The last converged state (taken by a solve: a failed solve leaves none).
    warm: Option<qm::KsState>,
    /// The grid of the last evaluation; a different one drops `warm`.
    grid: Option<qm::Grid>,
    /// Return `Display::Cpu` with each result.
    display: bool,
    /// `warm` is the density of the geometry last evaluated (module doc, "Density samples").
    density_current: bool,
    /// ρ↑ + ρ↓ of `warm`, summed on the first `sample_density` after a solve.
    rho: Option<Vec<f64>>,
    /// The Activity view's live probe and the slot this solver reports on (`with_probe`).
    probe: Option<(Arc<ActivityProbe>, Slot)>,
    /// The orbital view (module doc, "Orbital display"): its state, the producer, the cell,
    /// nuclei and settings of the latest successful evaluation (the geometry `warm` was solved
    /// at), and the unoccupied orbitals' bookkeeping.
    orb: ViewState,
    view: Option<CpuView>,
    last: Option<(Cell, Vec<qm::Nucleus>, ScfSettings)>,
    unocc: UnoccupiedState,
    /// The driver's display interrupt (`crate::orbitals`, "Cancellation").
    interrupt: Option<DisplayInterrupt>,
    /// The density shown includes ρ_pc (`set_display_partial_core`).
    show_core: bool,
    /// The `branch_jump` guard (module doc): the previous evaluation.
    guard: BranchJumpGuard,
    /// The orbital shares' free-atom tables (`set_share_tables`; module doc, "Orbital display").
    share_tables: Option<Arc<ShareTables>>,
}

impl CpuKs {
    /// `tol`: the SCF tolerance on Δρ = ∫|ρ_out − ρ_in|.
    pub fn new(tol: f64) -> CpuKs {
        CpuKs {
            settings_tol: tol, warm: None, grid: None, display: true, density_current: false, rho: None, probe: None, orb: ViewState::default(), view: None,
            last: None,
            unocc: UnoccupiedState::default(),
            interrupt: None,
            show_core: false,
            guard: BranchJumpGuard::new(),
            share_tables: None,
        }
    }

    /// Report this solver's live state on `probe`'s `slot` (the Activity view, spec §4.3 A1):
    /// running for each solve and its iterations (`qm::scf_observed`'s observer →
    /// `ActivityProbe::scf_iteration`); no abort-poll step count (`qm::scf` is not cancellable).
    pub fn with_probe(mut self, probe: Arc<ActivityProbe>, slot: Slot) -> Self {
        self.probe = Some((probe, slot));
        self
    }

    /// The orbital payload of `warm` (module doc, "Orbital display"); `None` unless it is the
    /// geometry last evaluated. `should_abort` interrupts the unoccupied eigensolve only.
    fn orbital_payload(&mut self, should_abort: &mut dyn FnMut() -> bool) -> Option<OrbitalDisplay> {
        if !self.density_current {
            return None;
        }
        let state = self.warm.as_ref()?;
        let (cell, nuclei, settings) = self.last.as_ref()?;
        let view = self.view.get_or_insert_with(|| CpuView::new(&state.grid));
        let positions: Vec<[f64; 3]> = nuclei.iter().map(|n| n.pos).collect();
        // unoccupied orbitals (`crate::orbitals`, "Unoccupied orbitals")
        // `requested` is the app's (the driver matches it); the solve is capped at UNOCCUPIED_MAX
        let requested = self.orb.request().unoccupied;
        let m = requested.min(UNOCCUPIED_MAX) as usize;
        let now = Instant::now();
        self.unocc.revalidate(&positions);
        match self.unocc.plan(m, now, self.orb.request().force_every_step) {
            UnoccupiedPlan::Solve => {
                // the Activity tab's job line and log row (display only)
                let busy = self.probe.as_ref().map(|(p, _)| p.display_solve(sim::DisplayJob::Unoccupied(m as u32)));
                let solved = view.unoccupied(state, nuclei, m, settings, should_abort);
                if let Some(b) = busy {
                    b.finish(solved.is_none());
                }
                match solved {
                    Some(_) => self.unocc.solved(m, positions.clone(), now, now.elapsed()),
                    None => self.unocc.interrupted(),
                }
            }
            UnoccupiedPlan::Keep => {}
            UnoccupiedPlan::Clear => {
                view.clear_unoccupied();
                self.unocc.dropped();
            }
        }
        let (out, spin, pick, field) = view.update(state, nuclei, self.settings_tol, &self.orb);
        let status = self.unocc.status(requested, view.unoccupied_info(), &out, Vec::new());
        // the CPU solver's nominal eigensolver tolerance (no effective one); not measured on a
        // grid's noise, but the 49³ CPU closed-shell data (`references.md`) sit far below the budget
        let noise = spin.noise_l1(settings.tol, settings.eig_tol);
        let mut payload = self.orb.display(cell.clone(), state.grid.origin, positions, &out, &spin, noise, pick, field, status);
        if state.ionic.is_some() && nuclei.len() == 1 {
            payload.notes.push(crate::orbitals::PSEUDO_LABELS_NOTE.into());
        }
        // the rows' shares at the evaluated nuclei, from the ψ̃ just shown (`crate::orbitals`, "Orbital shares")
        let z: Vec<u8> = nuclei.iter().map(|n| n.z).collect();
        crate::orbitals::attach_shares(&mut payload, self.share_tables.as_deref(), &z, |s, i| view.shown(s, i));
        Some(payload)
    }

    /// Whether results carry the density (`ForceResult::display`; default on).
    pub fn set_display(&mut self, on: bool) {
        self.display = on;
    }
}

impl ForceProvider for CpuKs {
    /// `should_abort` is polled once, before the solve (module doc: `qm::scf` has no
    /// cancellation point).
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        // whatever happens below, the held density is no longer this scene's until it succeeds
        self.density_current = false;
        self.rho = None;
        if let Some(r) = nothing_to_solve(scene) {
            return r;
        }
        let mut notes = Vec::new();
        let (grid, system, settings) = problem(scene, self.settings_tol, &mut notes)?;
        if should_abort() {
            return Err(ProviderError::Cancelled);
        }
        // the Activity probe (spec A1): running for the whole attempt, cleared by the guard on
        // every exit
        let probe = self.probe.clone();
        let _busy = probe.as_ref().map(|(p, s)| p.busy(*s));
        let t0 = Instant::now();
        if self.grid.as_ref() != Some(&grid) {
            self.warm = None;
            self.guard.reset();
            self.grid = Some(grid.clone());
            // the orbital producer is grid-sized: rebuilt (module doc, "Orbital display")
            self.view = None;
            self.orb.rebuilt();
            self.unocc.dropped();
        }
        let prev = self.warm.take();
        // R15: a warm stall is retried once without the previous state (module doc)
        // the probe's per-iteration record (Activity view), when there is a probe
        let mut observe = |p: &qm::ScfProgress| {
            if let Some((pr, slot)) = &probe {
                pr.scf_iteration(*slot, &iteration_of(p));
            }
        };
        let ((state, mut stats), stall) = retry_cold_on_stall(prev.is_some(), |w| {
            let observer: Option<&mut dyn FnMut(&qm::ScfProgress)> = if probe.is_some() { Some(&mut observe) } else { None };
            qm::scf_observed(&grid, &system, &settings, if w { prev.as_ref() } else { None }, observer)
        })
        .map_err(solver_error)?;
        drop(prev);
        let forces = qm::forces(&grid, &state, &system.nuclei, settings.sigma).map_err(solver_error)?;
        // the guard's spin measure (N↑ = N↓ only): the CPU spin reduction, compensated
        let spin_abs = (system.n_up == system.n_dn).then(|| qm::orbitals::spin_summary(&grid, &state.rho_up, &state.rho_dn, system.n_up, system.n_dn).abs_integral);
        let levels = [frontier(&state.evals_up, &state.occ_up), frontier(&state.evals_dn, &state.occ_dn)];
        self.unocc.new_state(t0.elapsed());
        let display = self.display.then(|| {
            // ρ (+ ρ_pc when asked: the generator's core model, a display aid, never solved)
            let mut rho: Vec<f64> = state.rho();
            if let (true, Some(setup)) = (self.show_core, system.ionic.as_deref()) {
                for (r, c) in rho.iter_mut().zip(qm::nuclear::partial_core_field(&grid, &system.nuclei, setup)) {
                    *r += c;
                }
            }
            let rho: Vec<f32> = rho.iter().map(|&x| x as f32).collect();
            let max = rho.iter().copied().fold(0.0f32, f32::max);
            Display::Cpu { cell: scene.cell.clone(), origin: grid.origin, rho: Arc::new(rho), max }
        });
        self.warm = Some(state);
        self.density_current = true;
        self.last = Some((scene.cell.clone(), system.nuclei.clone(), settings));
        // after the energy and forces: display only (module doc, "Orbital display"); its
        // eigensolve yields to the driver's display interrupt, never to `should_abort`
        let interrupt = self.interrupt.clone();
        let orbitals = if self.orb.due(Instant::now()) { self.orbital_payload(&mut || interrupt.as_ref().is_some_and(|f| f())) } else { None };
        if let Some(st) = stall {
            stats.iterations += st.iterations;
            notes.push(st.note());
        }
        if stats.max_eigen_residual >= settings.eig_tol {
            notes.push(format!("LOBPCG stopped at its iteration cap at least once (max residual {:.2e} ≥ tol {:.0e})",
                               stats.max_eigen_residual, settings.eig_tol));
        }
        let energy = stats.energy.total;
        let mut stats = eval_stats(stats, None, t0, notes, settings.eig_tol, &system, levels);
        judge_branch(&mut self.guard, &mut stats, scene, &grid, &system, &forces, spin_abs);
        Ok(ForceResult { energy, forces, stats, display, orbitals })
    }

    fn name(&self) -> &'static str {
        "Kohn-Sham (CPU)"
    }

    fn reset(&mut self) {
        self.warm = None;
        self.guard.reset();
        self.density_current = false;
        self.rho = None;
        self.view = None;
        self.orb.rebuilt();
        self.unocc.dropped();
    }

    /// Keeps the warm `KsState` (CPU has no extrapolation history to drop); its density is the
    /// old geometry's, so `sample_density` says `None` until the next evaluation.
    fn discontinuity(&mut self) {
        self.density_current = false;
    }

    /// `qm::Grid::trilinear` on the converged ρ↑ + ρ↓ (module doc, "Density samples").
    fn sample_density(&mut self, points: &[[f64; 3]]) -> Option<Vec<f64>> {
        if !self.density_current {
            return None;
        }
        let state = self.warm.as_ref()?;
        let rho = self.rho.get_or_insert_with(|| state.rho());
        Some(points.iter().map(|&p| state.grid.trilinear(rho, p)).collect())
    }

    fn scf_tol(&self) -> Option<f64> {
        Some(self.settings_tol)
    }

    /// A second `CpuKs` at the same tolerance, display off (the references' solver).
    fn side_solver(&self) -> Option<Box<dyn ForceProvider>> {
        let mut c = CpuKs::new(self.settings_tol);
        c.set_display(false);
        Some(Box::new(c))
    }

    fn set_display_partial_core(&mut self, on: bool) {
        self.show_core = on;
    }

    fn ionic_tag(&self, scene: &Scene) -> String {
        crate::pseudo::provenance_tag(scene)
    }

    fn valence_charges(&self, scene: &Scene) -> Option<Vec<f64>> {
        crate::pseudo::valence_charges(scene)
    }

    fn set_orbital_view(&mut self, request: OrbitalViewRequest) {
        if self.orb.set(request) == ViewChange::Off {
            self.view = None;
            self.unocc.dropped();
        }
    }

    /// `qm::orbitals` on the converged state, no solve (module doc, "Orbital display").
    fn orbital_display_now(&mut self, should_abort: &mut dyn FnMut() -> bool) -> Option<OrbitalDisplay> {
        if !self.orb.on() || !self.density_current || self.warm.is_none() {
            return None;
        }
        self.orb.now(Instant::now());
        self.orbital_payload(should_abort)
    }

    fn set_display_interrupt(&mut self, interrupt: DisplayInterrupt) {
        self.interrupt = Some(interrupt);
    }

    fn set_share_tables(&mut self, tables: Option<Arc<ShareTables>>) {
        self.share_tables = tables;
    }
}

/// What [`GpuKs::shown_payload`] shows: the cell and nuclei the state was solved at (and their
/// positions), the SCF tolerance the state was solved at — `settings.tol` after
/// `raise_to_gpu_floor`, the one value behind both `OrbitalView::update`'s τ = `qm::orbitals::tau`
/// and the spin-noise bound (a Harris solve's τ-sets use the same τ, so the view and the solve
/// group levels alike; review N1) — the eigensolver tolerance, the unoccupied count asked for and
/// the memory guard's notes.
struct Shown<'a> {
    cell: &'a Cell,
    nuclei: &'a [qm::Nucleus],
    positions: Vec<[f64; 3]>,
    scf_tol: f64,
    eig_tol: f64,
    requested: u8,
    guard: Vec<String>,
}

/// The GPU adapter's density textures (`crate::orbitals::Pool`: a texture leaves the pool inside
/// a [`Display`] as an `Arc` and is written again only when the pool holds the only reference, so
/// it is never overwritten while it can be on screen). With a viewer that keeps one result this
/// settles at two textures (ping-pong: the displayed one and the one being written); a grid
/// change drops the old size. The orbital products have pools of their own.
type TexturePool = Pool<Volume3D>;

/// The GPU Kohn-Sham solver behind the force boundary (module doc).
pub struct GpuKs {
    /// The solver for the current grid (`None` until the first evaluation).
    solver: Option<qm_gpu::GpuSolver>,
    ctx: GpuContext,
    tol: f64,
    /// Density extrapolation across warm starts (on: the MD case).
    extrapolate: bool,
    /// The next solve runs cold (`reset`).
    cold: bool,
    /// Return `Display::Gpu` with each result, from `textures`.
    display: bool,
    textures: TexturePool,
    /// The solver's state is the density of the geometry last evaluated (module doc, "Density
    /// samples").
    density_current: bool,
    /// Refuse scenes whose estimated footprint exceeds this many bytes (module doc, "GPU memory").
    vram_budget: Option<u64>,
    /// Where the budget came from, for the refusals (`with_vram_budget_note`).
    vram_note: Option<String>,
    /// The `FermiDirac` witness cap of the last scene checked (`check_vram`: `witness_cap_for` under a budget,
    /// `qm::scf::MAX_WITNESS_GROWTH` without one), handed to the solver before each solve.
    witness_cap: usize,
    /// The Activity view's live probe and the slot this solver reports on (`with_probe`).
    probe: Option<(Arc<ActivityProbe>, Slot)>,
    /// The orbital view (module doc, "Orbital display"): its state, the producer (for the
    /// solver's grid and device), the textures of its scalar and colour products, and the cell
    /// and nuclei of the latest successful evaluation.
    orb: ViewState,
    orbital: Option<OrbitalView>,
    orb_scalar: Pool<Volume3D>,
    orb_colour: Pool<ColourVolume3D>,
    last: Option<(Cell, Vec<qm::Nucleus>, ScfSettings)>,
    /// The unoccupied orbitals' bookkeeping (`crate::orbitals`, "Unoccupied orbitals").
    unocc: UnoccupiedState,
    /// The Harris display solve's own state (module doc, "Harris display"), for the solver's grid.
    harris: HarrisState,
    /// The driver's display interrupt (`crate::orbitals`, "Cancellation").
    interrupt: Option<DisplayInterrupt>,
    /// The density shown includes ρ_pc (`set_display_partial_core`).
    show_core: bool,
    /// The `branch_jump` guard (module doc): the previous evaluation.
    guard: BranchJumpGuard,
    /// The orbital shares' free-atom tables (`set_share_tables`; module doc, "Orbital display").
    share_tables: Option<Arc<ShareTables>>,
    /// Those tables on the device, for the cell they were packed for: re-uploaded only when the
    /// tables change (pointer and content) or the cell does (the lookups reach its diagonal).
    share_gpu: Option<(Arc<ShareTables>, Cell, ShareTablesGpu)>,
}

impl GpuKs {
    /// `ctx`: the device to solve on (a clone of the viewer's, so the display textures can be
    /// bound by its renderer); `tol`: the SCF tolerance on Δρ.
    pub fn new(ctx: GpuContext, tol: f64) -> GpuKs {
        GpuKs {
            solver: None,
            ctx,
            tol,
            extrapolate: true,
            cold: false,
            display: true,
            textures: TexturePool::default(),
            density_current: false,
            vram_budget: None,
            vram_note: None,
            witness_cap: qm::scf::MAX_WITNESS_GROWTH,
            probe: None,
            orb: ViewState::default(),
            orbital: None,
            orb_scalar: Pool::default(),
            orb_colour: Pool::default(),
            last: None,
            unocc: UnoccupiedState::default(),
            interrupt: None,
            show_core: false,
            guard: BranchJumpGuard::new(),
            share_tables: None,
            share_gpu: None,
            harris: HarrisState::new(),
        }
    }

    /// Report this solver's live state on `probe`'s `slot` (the Activity view, spec §4.3 A1):
    /// running for each solve, and the SCF step in progress (one count per poll of `solve_with`,
    /// which polls once before every SCF step). Not inherited by `side_solver`: whoever builds a
    /// solver attaches its probe.
    pub fn with_probe(mut self, probe: Arc<ActivityProbe>, slot: Slot) -> Self {
        self.probe = Some((probe, slot));
        self
    }

    /// The orbital payload of the solver's converged state (module doc, "Orbital display"):
    /// one `OrbitalView` update, its product chosen from its own rows. `None` unless the state is
    /// the geometry last evaluated. `should_abort` interrupts the unoccupied eigensolve only.
    fn orbital_payload(&mut self, should_abort: &mut dyn FnMut() -> bool) -> Option<OrbitalDisplay> {
        if !self.density_current {
            return None;
        }
        let GpuKs { solver, ctx, orb, orbital, orb_scalar, orb_colour, last, vram_budget, vram_note, unocc, probe, share_tables, share_gpu, .. } = self;
        let solver = solver.as_mut()?;
        let (cell, nuclei, settings) = last.as_ref()?;
        let positions: Vec<[f64; 3]> = nuclei.iter().map(|n| n.pos).collect();
        // the rows the view shows: the state's shown columns (under FermiDirac the f-bearing ones; its witness columns
        // are the solve's, counted in the scene's estimate — `witness_extra_bytes` / `pseudo_extra_bytes` — and are not
        // display rows, `qm::orbitals::shown_columns`, plan 2026-11-13 Task 9)
        let (n, n_up, n_dn, origin, all_electron) = {
            let st = solver.state()?;
            let fd = matches!(st.occupations, qm::Occupations::FermiDirac { .. });
            let shown = |len: usize, occ: &[f64]| qm::orbitals::shown_columns(len, occ, fd).len();
            (st.grid.n[0], shown(st.orbitals_up.len(), &st.occ_up), shown(st.orbitals_dn.len(), &st.occ_dn), st.grid.origin, st.ionic.is_none())
        };
        // `requested` is the app's (the driver matches it); the solve is capped at UNOCCUPIED_MAX
        let requested = orb.request().unoccupied;
        let mut m = requested.min(UNOCCUPIED_MAX) as usize;
        // the GPU memory guard (module doc, "GPU memory"): the view, not the scene, is refused
        let mut guard = Vec::new();
        if let Some(budget) = *vram_budget {
            // the scene guard's all-electron witness bound too (`witness_extra_bytes`, plan 2026-11-13 Task 7 fix round 1),
            // so a budget the scene passes is judged on the same base
            // (at the solver's witness cap, plan 2026-11-13 Task 9b: above the counted start (`WITNESS_GROWTH_COUNTED`) only when the budget has room
            // for the largest view, `witness_cap_for`)
            let base = solve_footprint_bytes(n, (n_up + n_dn) as u64, n_up.max(n_dn) as u64) + if all_electron { witness_bytes(n, solver.witness_cap()) } else { 0 };
            let need = |m: usize| base + orbital_view_footprint_bytes(n, n_up, n_dn, m, orb.texture());
            let gb = |b: u64| b as f64 / 1e9;
            if need(m) > budget {
                if m > 0 && need(0) <= budget {
                    guard.push(format!(
                        "{requested} unoccupied orbitals refused: with them this scene needs ≈ {:.2} GB of GPU memory at {n}³, over {} — fewer unoccupied orbitals or a coarser quality (Scene ▸ Quality) fits",
                        gb(need(m)), GpuKs::budget_words(budget, vram_note.as_deref())));
                    m = 0;
                } else {
                    if let Some(v) = orbital.take() {
                        v.release(solver);
                    }
                    orb_scalar.clear();
                    orb_colour.clear();
                    unocc.dropped();
                    return Some(orb.refused(cell.clone(), origin, positions, format!(
                        "the orbital view needs ≈ {:.2} GB of GPU memory at {n}³ with this scene, over {}: choose a coarser quality (Scene ▸ Quality)",
                        gb(need(0)), GpuKs::budget_words(budget, vram_note.as_deref()))));
                }
            }
        }
        let view = orbital.get_or_insert_with(|| {
            let mut v = OrbitalView::new(solver);
            v.set_face_limit(box_affected_face_density(solver.grid().h));
            v
        });
        // a Harris solve's unoccupied vectors (module doc, "Harris display") are never this state's
        if unocc.harris_held() {
            view.clear_unoccupied();
            unocc.dropped();
        }
        // unoccupied orbitals (`crate::orbitals`, "Unoccupied orbitals")
        let now = Instant::now();
        unocc.revalidate(&positions);
        match unocc.plan(m, now, orb.request().force_every_step) {
            UnoccupiedPlan::Solve => {
                // the Activity tab's job line and log row (display only)
                let busy = probe.as_ref().map(|(p, _)| p.display_solve(sim::DisplayJob::Unoccupied(m as u32)));
                let solved = view.unoccupied(solver, nuclei, m, settings, should_abort);
                if let Some(b) = busy {
                    b.finish(solved.is_none());
                }
                match solved {
                    Some(_) => unocc.solved(m, positions.clone(), now, now.elapsed()),
                    None => unocc.interrupted(),
                }
            }
            UnoccupiedPlan::Keep => {}
            UnoccupiedPlan::Clear => {
                if unocc.held() {
                    view.release_work(solver);
                }
                unocc.dropped();
            }
        }
        let solver = &*solver;
        let state = solver.state()?;
        let shown = Shown { cell, nuclei, positions, scf_tol: settings.tol, eig_tol: solver.profile().eig_tol_used, requested, guard };
        Some(GpuKs::shown_payload(ctx, orb, view, solver, state, (orb_scalar, orb_colour), (share_tables.as_ref(), share_gpu), unocc, shown))
    }

    /// The payload of one `OrbitalView` update on `state` — the converged state's (`orbital_payload`)
    /// or a Harris solve's (`harris_once`) — from the product textures to the shares: the product
    /// picked from the update's own rows, its texture from the pools, the rows and verdicts with the
    /// unoccupied status (`unocc`), the box marks, the lone pseudo-atom's labels note and the rows'
    /// shares at `shown.nuclei`. `orb`'s ids, selection fallback and rows carry over between the two
    /// kinds (one view: the tracks continue).
    #[allow(clippy::too_many_arguments)]
    fn shown_payload(ctx: &GpuContext, orb: &mut ViewState, view: &mut OrbitalView, solver: &qm_gpu::GpuSolver, state: &qm_gpu::GpuState,
                     (orb_scalar, orb_colour): (&mut Pool<Volume3D>, &mut Pool<ColourVolume3D>),
                     (share_tables, share_gpu): (Option<&Arc<ShareTables>>, &mut Option<(Arc<ShareTables>, Cell, ShareTablesGpu)>),
                     unocc: &UnoccupiedState, shown: Shown) -> OrbitalDisplay {
        let Shown { cell, nuclei, positions, scf_tol, eig_tol, requested, guard } = shown;
        let dims = solver.grid().n.map(|n| n as u32);
        let (scalar, colour) = match orb.texture() {
            TextureKind::Scalar => (Some(orb_scalar.take(ctx, dims)), None),
            TextureKind::Colour => (None, Some(orb_colour.take(ctx, dims))),
            TextureKind::None => (None, None),
        };
        let mut picked = None;
        let mut include: Vec<u32> = Vec::new();
        let u = {
            let (inc, picked, vs, sv, cv) = (&mut include, &mut picked, &*orb, scalar.as_deref(), colour.as_deref());
            view.update_with(solver, state, nuclei, scf_tol, move |out| {
                let p = vs.pick(out);
                let product = match (&p.kind, sv, cv) {
                    (&PickKind::Orbital { spin, track, squared }, Some(v), _) => Product::Orbital { spin, track, squared, out: v },
                    (PickKind::Spin, Some(v), _) => Product::Spin { out: v },
                    (PickKind::Colour { include, physical, look }, _, Some(c)) => {
                        inc.extend_from_slice(include);
                        let boundary = look.enclose.map(|pct| qm_gpu::orbitals::Boundary { share: pct as f32 / 100.0, shell: look.shell });
                        Product::Colour { include: inc.as_slice(), physical: *physical, boundary, out: c }
                    }
                    _ => Product::None,
                };
                *picked = Some(p);
                product
            })
        };
        let pick = picked.expect("update_with always picks");
        // `scale == 0`: nothing was written (`ViewUpdate` doc)
        let field = match &pick.kind {
            &PickKind::Orbital { squared, .. } => scalar.filter(|_| u.scale > 0.0).map(|volume| OrbitalField::Gpu { volume, signed: !squared, scale: u.scale }),
            PickKind::Spin => scalar.map(|volume| OrbitalField::Gpu { volume, signed: true, scale: u.scale }),
            PickKind::Colour { .. } => colour.map(|volume| OrbitalField::GpuColour { volume, scale: u.scale }),
            PickKind::None => None,
        };
        let status = unocc.status(requested, view.unoccupied_info(), &u.out, guard);
        let noise = u.spin.noise_l1(scf_tol, eig_tol);
        let mut payload = orb.display(cell.clone(), state.grid.origin, positions, &u.out, &u.spin, noise, pick, field, status);
        orb.mark_touching(&mut payload, &u.touching);
        if state.ionic.is_some() && nuclei.len() == 1 {
            payload.notes.push(crate::orbitals::PSEUDO_LABELS_NOTE.into());
        }
        // the rows' shares at the evaluated nuclei, from the ψ̃ just shown (`crate::orbitals`,
        // "Orbital shares"): the GPU pass, then the CPU path's own check and set average
        let z: Vec<u8> = nuclei.iter().map(|n| n.z).collect();
        let fit = share_tables.filter(|t| !z.is_empty() && crate::orbitals::tables_fit(Some(t), &z).is_some() && z.len() == payload.positions.len());
        if let Some(tables) = fit {
            let gpu_tables = GpuKs::share_tables_on_device(share_gpu, ctx, tables, cell, &payload.positions);
            let raw = view.shares(solver, gpu_tables, &payload.positions);
            let rows: Vec<Option<Vec<f64>>> = payload.rows.iter().map(|r| raw[spin_in(r.spin).index()].get(r.index).cloned()).collect();
            crate::orbitals::settle_shares(&mut payload, &z, rows);
        }
        payload
    }

    /// `tables` packed for the GPU share pass on `cell` (`ShareTablesGpu`; the lookups are
    /// `sim::Weights`' own), kept in `held` and re-uploaded only when the tables change — a new
    /// `Arc` with the same content is adopted without an upload — or the cell does.
    fn share_tables_on_device<'h>(held: &'h mut Option<(Arc<ShareTables>, Cell, ShareTablesGpu)>, ctx: &GpuContext, tables: &Arc<ShareTables>,
                                  cell: &Cell, positions: &[[f64; 3]]) -> &'h ShareTablesGpu {
        match held {
            Some((t, c, _)) if c == cell && (Arc::ptr_eq(t, tables) || **t == **tables) => *t = tables.clone(),
            _ => {
                let free: Vec<&RadialDensity> = tables.tables.iter().map(|a| a.as_ref()).collect();
                let w = Weights::new(&free, positions, cell);
                let (inv_dr, lookups, which) = w.lookups();
                *held = Some((tables.clone(), cell.clone(), ShareTablesGpu::new(ctx, cell.h, inv_dr, &lookups, which)));
            }
        }
        &held.as_ref().expect("set above").2
    }

    /// With a GPU memory budget (bytes): a scene over it is refused before anything is allocated
    /// (module doc, "GPU memory").
    pub fn with_vram_budget(mut self, bytes: u64) -> GpuKs {
        self.vram_budget = Some(bytes);
        self
    }

    /// Where the budget came from (the app: the card's memory, `ATOMIC_SANDBOX_VRAM_GB` or the
    /// default), said in brackets after the budget in a refusal.
    pub fn with_vram_budget_note(mut self, note: String) -> GpuKs {
        self.vram_note = Some(note);
        self
    }

    /// "the 12.9 GB budget (…note…)" for a refusal.
    fn budget_words(budget: u64, note: Option<&str>) -> String {
        match note {
            Some(n) => format!("the {:.1} GB budget ({n})", budget as f64 / 1e9),
            None => format!("the {:.1} GB budget", budget as f64 / 1e9),
        }
    }

    /// The budget check of `evaluate` (module doc, "GPU memory"); on success it records the scene's witness cap
    /// (`witness_cap_for`; `MAX_WITNESS_GROWTH` without a budget) for `apply_witness_cap`.
    fn check_vram(&mut self, scene: &Scene) -> Result<(), ProviderError> {
        let limits = self.ctx.device.limits();
        if let Some(m) = binding_refusal(scene.cell.n, limits.max_storage_buffer_binding_size as u64, limits.max_buffer_size) {
            return Err(ProviderError::Invalid(m));
        }
        if let Some(budget) = self.vram_budget {
            let need = gpu_footprint_bytes(scene);
            if need > budget {
                let fp = crate::pseudo::footprint_inputs(scene);
                return Err(ProviderError::Invalid(format!(
                    "this scene needs ≈ {:.1} GB of GPU memory at {}³ ({} {}electrons), over {}: choose a coarser quality (Scene ▸ Quality) or fewer atoms",
                    need as f64 / 1e9,
                    scene.cell.n,
                    fp.n_valence.max(0),
                    if fp.pseudised { "valence " } else { "" },
                    GpuKs::budget_words(budget, self.vram_note.as_deref())
                )));
            }
        }
        self.witness_cap = self.vram_budget.map_or(qm::scf::MAX_WITNESS_GROWTH, |b| witness_cap_for(scene, b));
        Ok(())
    }

    /// Hands the solver the witness cap `check_vram` derived, naming the budget it came from (plan 2026-11-13 Task 9b:
    /// the cap error then says the cap is what the budget fits).
    fn apply_witness_cap(&mut self) {
        let words = self.vram_budget.map(|b| GpuKs::budget_words(b, self.vram_note.as_deref()));
        if let Some(s) = self.solver.as_mut() { s.set_witness_cap(self.witness_cap, words); }
    }

    /// Drops the orbital producer and the growth its display eigensolves made in the solver's
    /// LOBPCG work space (`OrbitalView::release`).
    fn drop_orbital_view(&mut self) {
        if let Some(v) = self.orbital.take() {
            match self.solver.as_mut() {
                Some(s) => v.release(s),
                None => drop(v),
            }
        }
        self.unocc.dropped();
    }

    /// A readback failed while the orbital view updated. `OrbitalView::update` commits its
    /// persistent state from what it read — the tracker's ids, slots and phase signs (from the
    /// overlaps with the previous display), the copies of the previously shown orbitals, and the
    /// warm unoccupied vectors — so zeros from the failed readback would renumber the tracks,
    /// flip signs, change the user's selection and warm-start the next display eigensolve from
    /// garbage, long after the glitch. The view is dropped instead (the "producer rebuilt" path
    /// of `reset`: raw ids restart, the cadence is armed so the next refresh is due at once) and
    /// built again from the solver's state.
    fn drop_view_after_fault(&mut self) {
        self.drop_orbital_view();
        // the Harris state's warm vectors may hold the same zeros (Task 4 review M4)
        self.harris.clear();
        self.orb.rebuilt();
    }

    /// Off, then back on: drops the parked ρ_{t−1} so the next warm start is not extrapolated.
    fn drop_extrapolation(&mut self) {
        if let Some(s) = self.solver.as_mut() {
            s.set_density_extrapolation(false);
            s.set_density_extrapolation(self.extrapolate);
        }
    }

    /// Whether results carry the density texture (`ForceResult::display`; default on). Off: no
    /// copy, no texture, and the pool's own references are dropped.
    pub fn set_display(&mut self, on: bool) {
        self.display = on;
        if !on {
            self.textures.clear();
        }
    }
}

impl GpuKs {
    /// The solver for `grid`: the one held, or — none yet, or another grid — a new one, the old
    /// one's buffers freed first. A rebuild drops what belonged to the old solver: the orbital
    /// producer (module doc, "Orbital display"), the Harris state, and the converged density
    /// (`density_current`: the new solver holds none).
    fn solver_for(&mut self, grid: &Grid) {
        if self.solver.as_ref().is_none_or(|s| s.grid() != grid) {
            self.solver = None; // free the old grid's buffers before allocating the new ones
            // the orbital producer belongs to the old solver: rebuilt (module doc, "Orbital display")
            self.orbital = None;
            self.orb.rebuilt();
            self.unocc.dropped();
            self.harris.clear();
            self.guard.reset();
            self.density_current = false;
            let mut s = qm_gpu::GpuSolver::new(self.ctx.clone(), grid);
            s.set_density_extrapolation(self.extrapolate);
            self.solver = Some(s);
        }
    }

    /// [`ForceProvider::harris_display`]'s answer (module doc, "Harris display").
    fn harris_once(&mut self, scene: &Scene, atoms: &HarrisAtoms, should_abort: &mut dyn FnMut() -> bool) -> Result<OrbitalDisplay, ProviderError> {
        let t0 = Instant::now();
        // the setup's notes (an ionic fallback or refusal) travel with the payload (Task 4 review M3)
        let mut setup_notes = Vec::new();
        let (grid, system, mut settings) = problem(scene, self.tol, &mut setup_notes)?;
        // the τ-sets' tolerance as the SCF on this teacher would run it
        let _ = raise_to_gpu_floor(self.tol, &system, &mut settings);
        self.check_vram(scene)?;
        let positions: Vec<[f64; 3]> = system.nuclei.iter().map(|n| n.pos).collect();
        // a payload that only says why nothing was solved (still a Harris answer; nothing was rescaled)
        let note_only = |orb: &ViewState, note: String| {
            let mut p = orb.refused(scene.cell.clone(), grid.origin, positions.clone(), note);
            p.notes.extend(setup_notes.iter().cloned());
            p.kind = OrbitalKind::Harris { experience: None, rescale: 1.0, wall_s: t0.elapsed().as_secs_f64() };
            p
        };
        if !self.orb.on() {
            return Ok(note_only(&self.orb, "no orbital view is on: no orbital display solve was made".into()));
        }
        let (n, n_up, n_dn) = (scene.cell.n, system.n_up, system.n_dn);
        // `requested` is the app's; the solve is capped at UNOCCUPIED_MAX (as `orbital_payload`)
        let requested = self.orb.request().unoccupied;
        let mut m = requested.min(UNOCCUPIED_MAX) as usize;
        // the GPU memory guard (module doc, "GPU memory"): the scene's solve (its pseudo extras
        // included: Task 4 review M1), the view and the Harris state
        let mut guard = Vec::new();
        if let Some(budget) = self.vram_budget {
            // the solve and the Harris block at the cap `check_vram` derived (`witness_cap_for`: when it is above the counted
            // start every request here fits by construction; at the count this is the guard as before)
            let w = self.witness_cap;
            let base = gpu_footprint_bytes(scene) - witness_extra_bytes(n) + witness_bytes(n, w);
            let texture = self.orb.texture();
            let pseudised = system.ionic.is_some();
            let need = |m: usize| base + orbital_view_footprint_bytes(n, n_up, n_dn, m, texture) + harris_footprint_bytes(n, n_up, n_dn, m, pseudised, w);
            let gb = |b: u64| b as f64 / 1e9;
            if need(m) > budget {
                if m > 0 && need(0) <= budget {
                    guard.push(format!(
                        "{requested} unoccupied orbitals refused: with them the orbital display solve needs ≈ {:.2} GB of GPU memory at {n}³, over {} — fewer unoccupied orbitals or a coarser quality (Scene ▸ Quality) fits",
                        gb(need(m)), GpuKs::budget_words(budget, self.vram_note.as_deref())));
                    m = 0;
                    // the state's vectors of the larger block go now, not after the solve: the
                    // m = 0 solve is then within the budget it was checked against (review M5)
                    self.harris.clear();
                } else {
                    self.harris.clear();
                    return Ok(note_only(&self.orb, format!(
                        "the orbital display solve needs ≈ {:.2} GB of GPU memory at {n}³ with this scene, over {}: choose a coarser quality (Scene ▸ Quality)",
                        gb(need(0)), GpuKs::budget_words(budget, self.vram_note.as_deref()))));
                }
            }
        }
        // ρ_in (`crate::harris`): each atom renormalised to its electrons (Z, or Z_v under a
        // pseudopotential), the sum rescaled to the scene's N; only profiles of this surface
        let counted: Vec<f64> = system.nuclei.iter().map(|nu| system.ionic.as_deref().map_or(nu.z as f64, |s| s.valence_charge(nu.z))).collect();
        let fitting: Vec<AtomProfile> = atoms.profiles.iter().filter(|p| p.fits(n, scene.cell.h, scene.sigma, scene.ionic)).cloned().collect();
        let rho = match superposition(&grid, &system.nuclei, &fitting, &counted, n_up, n_dn) {
            Ok(r) => r,
            Err(why) => return Ok(note_only(&self.orb, why)),
        };
        if should_abort() {
            return Err(ProviderError::Cancelled);
        }
        self.solver_for(&grid);
        self.apply_witness_cap();
        // the probe's display slot (never its solve slots: this is not a teacher solve)
        let busy = self.probe.as_ref().map(|(p, _)| p.display_solve(DisplayJob::Harris));
        qm_gpu::grid::clear_readback_fault();
        let GpuKs { solver, ctx, orb, orbital, orb_scalar, orb_colour, unocc, harris, share_tables, share_gpu, .. } = self;
        let solver = solver.as_mut().expect("built above");
        let input = HarrisInput { system: &system, settings: &settings, rho_in: [&rho.up, &rho.dn], unoccupied: m };
        let solve = match solver.harris(harris, &input, should_abort) {
            Ok(Some(s)) => s,
            Ok(None) => {
                if let Some(b) = busy {
                    b.finish(true);
                }
                return Err(ProviderError::Cancelled);
            }
            Err(e) => return Err(solver_error(e)), // the guard's drop records it as stopped
        };
        let view = orbital.get_or_insert_with(|| {
            let mut v = OrbitalView::new(solver);
            v.set_face_limit(box_affected_face_density(solver.grid().h));
            v
        });
        // the Harris solve's unoccupied vectors, only when some were asked for; otherwise the view
        // shows none (a converged state's belong to another potential)
        match harris.unoccupied() {
            (vectors, Some(info)) if m > 0 => {
                view.adopt_unoccupied(solver, vectors, info.clone());
                unocc.harris_adopted(m, positions.clone(), solve.wall);
            }
            _ => {
                if unocc.held() || view.unoccupied_info().is_some() {
                    view.clear_unoccupied();
                }
                unocc.dropped();
            }
        }
        let solver = &*solver;
        let state = harris.state().expect("a completed Harris solve holds its state");
        let eig_tol = solver.effective_eig_tol(&settings);
        let shown = Shown { cell: &scene.cell, nuclei: &system.nuclei, positions, scf_tol: settings.tol, eig_tol, requested, guard };
        let mut payload = GpuKs::shown_payload(ctx, orb, view, solver, state, (orb_scalar, orb_colour), (share_tables.as_ref(), share_gpu), unocc, shown);
        payload.notes.extend(setup_notes.iter().cloned());
        payload.notes.extend(solve.notes.iter().cloned());
        // the sanity bound's (Task 4 review M2): a profile off the count this solve uses
        payload.notes.extend(rho.notes.iter().cloned());
        if (rho.rescale - 1.0).abs() > 1e-9 {
            let carried = counted.iter().sum::<f64>();
            payload.notes.push(format!(
                "the superposed atoms carry {carried} electrons: their density was rescaled to the scene's {} (× {:.4}, charge {:+})",
                n_up + n_dn, rho.rescale, scene.charge));
        }
        // the per-atom electrons before renormalisation: the app logs this note, never draws it
        payload.notes.push(crate::harris::per_atom_note(&system.nuclei, &rho.per_atom, &counted));
        // display only: a failed readback is an error, and the view that read the zeros goes
        if let Err(e) = device_fault() {
            self.drop_view_after_fault();
            return Err(e);
        }
        payload.kind = OrbitalKind::Harris { experience: None, rescale: rho.rescale, wall_s: t0.elapsed().as_secs_f64() };
        if let Some(b) = busy {
            b.finish(false);
        }
        Ok(payload)
    }

    /// One attempt of [`ForceProvider::evaluate`], with no "GPU busy" retry.
    fn evaluate_once(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        // whatever happens below, the held density is no longer this scene's until it succeeds
        self.density_current = false;
        if let Some(r) = nothing_to_solve(scene) {
            return r;
        }
        let mut notes = Vec::new();
        let (grid, system, mut settings) = problem(scene, self.tol, &mut notes)?;
        notes.extend(raise_to_gpu_floor(self.tol, &system, &mut settings));
        self.check_vram(scene)?;
        if should_abort() {
            return Err(ProviderError::Cancelled);
        }
        // the Activity probe (spec A1): running for the whole attempt, cleared by the guard on
        // every exit; one step per poll of `solve_with` (it polls before every SCF step)
        let probe = self.probe.clone();
        let _busy = probe.as_ref().map(|(p, s)| p.busy(*s));
        // the same abort for both of `solve_polled`'s polls: before every step (counted by the probe) and inside the
        // FD eigensolves (plan 2026-11-13 Task 9c: Pause/Cancel within one LOBPCG iteration; not counted)
        let abort = std::cell::RefCell::new(should_abort);
        let mut polled = || {
            if let Some((p, s)) = &probe {
                p.scf_step(*s);
            }
            (abort.borrow_mut())()
        };
        let mut inner_polled = || (abort.borrow_mut())();
        // the probe's per-iteration record (Activity view): rides on the SCF's own Δρ readback
        let mut observe = |p: &qm::ScfProgress| {
            if let Some((pr, slot)) = &probe {
                pr.scf_iteration(*slot, &iteration_of(p));
            }
        };
        self.solver_for(&grid);
        self.apply_witness_cap();
        let t0 = Instant::now(); // after the build: `solve_ms` excludes it (`EvalStats` doc)
        let solver = self.solver.as_mut().expect("created above");
        // consumed by this attempt even if it is abandoned: the state it parks belongs to the
        // new geometry (a valid warm start); an error leaves the solver without state anyway
        let warm = !std::mem::take(&mut self.cold);
        // R15 (module doc): a warm stall is retried once cold, density extrapolation off for
        // that solve (the failure already dropped the state and ρ_{t−1}; this makes the cold
        // solve independent of that detail) and back on afterwards
        let mut stalled_readbacks = 0;
        let result = retry_cold_on_stall(warm, |w| {
            if warm && !w {
                stalled_readbacks = solver.profile().readbacks;
                solver.set_density_extrapolation(false);
            }
            let observer: Option<&mut dyn FnMut(&qm::ScfProgress)> = if probe.is_some() { Some(&mut observe) } else { None };
            solver.solve_polled(&system, &settings, w, &mut polled, Some(&mut inner_polled), observer).map(|r| r.map(|(_, stats)| stats))
        });
        solver.set_density_extrapolation(self.extrapolate);
        // ...except the coincident-nuclei error, which keeps the old state as the warm start:
        // a pending reset must survive it (after the other errors the solver holds no state)
        if result.is_err() {
            self.cold |= !warm;
        }
        let (stats, stall) = result.map_err(solver_error)?;
        let Some(mut stats) = stats else {
            return Err(ProviderError::Cancelled);
        };
        let state = solver.state().expect("a converged solve exposes its state");
        let forces = solver.forces(state, &system.nuclei, settings.sigma).map_err(solver_error)?;
        // the guard's spin measure (N↑ = N↓ only): the orbital view's spin reduction, one readback (a failed one is
        // caught by the `device_fault` below and the evaluation repeated)
        let spin_abs = (system.n_up == system.n_dn).then(|| solver.spin_abs_integral(state));
        let levels = [frontier(&state.evals_up, &state.occ_up), frontier(&state.evals_dn, &state.occ_dn)];
        self.unocc.new_state(t0.elapsed());
        let display = self.display.then(|| {
            let volume = self.textures.take(&self.ctx, grid.n.map(|n| n as u32));
            // queue order makes the copy visible to any later render submission; the max
            // reduction below also waits for the GPU
            let max = if self.show_core && system.ionic.is_some() {
                solver.copy_density_with_core_to_volume(state, &volume)
            } else {
                solver.copy_density_to_volume(state, &volume);
                solver.density_max(state)
            };
            Display::Gpu { cell: scene.cell.clone(), origin: grid.origin, volume, max }
        });
        device_fault()?; // the density maximum is a readback: zeros would normalise the texture wrongly
        let profile = solver.profile();
        notes.extend(profile.notes.iter().cloned());
        if profile.extrapolated {
            notes.push("start density extrapolated (2ρ_t − ρ_{t−1})".into());
        }
        let (mut readbacks, eig_tol_used) = (profile.readbacks, profile.eig_tol_used);
        if let Some(st) = stall {
            stats.iterations += st.iterations;
            readbacks += stalled_readbacks;
            notes.push(st.note());
        }
        let energy = stats.energy.total;
        self.density_current = true;
        self.last = Some((scene.cell.clone(), system.nuclei.clone(), settings));
        // after the energy and forces: display only (module doc, "Orbital display"); its own
        // readback is not the solve's (`readbacks` above)
        let interrupt = self.interrupt.clone();
        let orbitals = if self.orb.due(Instant::now()) { self.orbital_payload(&mut || interrupt.as_ref().is_some_and(|f| f())) } else { None };
        // display only: a failed readback drops this refresh, never the solve — and the view that
        // read the zeros (`drop_view_after_fault`), whose next refresh is then due at once
        let orbitals = if device_fault().is_ok() {
            orbitals
        } else {
            self.drop_view_after_fault();
            None
        };
        let mut stats = eval_stats(stats, Some(readbacks), t0, notes, eig_tol_used, &system, levels);
        judge_branch(&mut self.guard, &mut stats, scene, &grid, &system, &forces, spin_abs);
        Ok(ForceResult { energy, forces, stats, display, orbitals })
    }
}

impl ForceProvider for GpuKs {
    /// `should_abort` is polled before the solve and before every SCF step. A readback the shared
    /// device does not deliver ([`SolverError::Device`]: the solve, the forces or the display
    /// maximum) is retried after a short pause ([`retry_on_device_fault`]); a retry that succeeds
    /// says so in the result's notes, and one that runs out is the error the worker reports
    /// ("GPU busy …") — the user's Play or any edit tries again.
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        // both closures below need the one `should_abort`, never at the same time
        let abort = std::cell::RefCell::new(should_abort);
        let this = &mut *self;
        let (mut result, failed) = retry_on_device_fault(
            || this.evaluate_once(scene, &mut || (*abort.borrow_mut())()),
            |ms| pause_unless_aborted(ms, &mut || (*abort.borrow_mut())()),
        )?;
        if failed > 0 {
            result.stats.notes.push(format!("GPU busy: a device readback failed {failed} time(s); the evaluation was repeated"));
        }
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "Kohn-Sham (GPU)"
    }

    /// The next solve is cold and does not extrapolate: the extrapolation history is dropped
    /// now (off, then back on) and `warm = false` discards the warm or parked state.
    fn reset(&mut self) {
        self.cold = true;
        self.guard.reset();
        self.density_current = false;
        self.drop_extrapolation();
        self.drop_orbital_view();
        // a reset is a new scene: the next Harris solve starts cold too (Task 4 review M4)
        self.harris.clear();
        self.orb.rebuilt();
    }

    /// Keeps the warm (or parked) state and drops only the extrapolation history ρ_{t−1}
    /// (`GpuSolver::set_density_extrapolation` doc: a discontinuous move must reset the chain).
    /// The state's density is the old geometry's: `sample_density` says `None` until the next
    /// evaluation.
    fn discontinuity(&mut self) {
        self.density_current = false;
        self.drop_extrapolation();
    }

    /// `GpuSolver::sample_density` on the converged state, f32 widened to f64 (module doc,
    /// "Density samples"): one dispatch, one readback.
    fn sample_density(&mut self, points: &[[f64; 3]]) -> Option<Vec<f64>> {
        if !self.density_current {
            return None;
        }
        let solver = self.solver.as_ref()?;
        let state = solver.state()?;
        qm_gpu::grid::clear_readback_fault();
        let samples = solver.sample_density(state, points);
        // a failed readback is zeros: no samples rather than a density that is zero (the caller asks again)
        device_fault().ok()?;
        Some(samples.into_iter().map(f64::from).collect())
    }

    fn scf_tol(&self) -> Option<f64> {
        Some(self.tol)
    }

    fn set_display_partial_core(&mut self, on: bool) {
        self.show_core = on;
    }

    fn ionic_tag(&self, scene: &Scene) -> String {
        crate::pseudo::provenance_tag(scene)
    }

    fn valence_charges(&self, scene: &Scene) -> Option<Vec<f64>> {
        crate::pseudo::valence_charges(scene)
    }

    /// A second `GpuKs` on the same context at the same tolerance, display off — its own
    /// `GpuSolver` and warm state, like the surrogate's background auditor (the references'
    /// solver).
    fn side_solver(&self) -> Option<Box<dyn ForceProvider>> {
        let mut g = GpuKs::new(self.ctx.clone(), self.tol);
        g.vram_budget = self.vram_budget;
        g.vram_note = self.vram_note.clone();
        g.set_display(false);
        Some(Box::new(g))
    }

    fn set_orbital_view(&mut self, request: OrbitalViewRequest) {
        if self.orb.set(request) == ViewChange::Off {
            self.drop_orbital_view();
            self.harris.clear();
            self.orb_scalar.clear();
            self.orb_colour.clear();
        }
    }

    /// One `OrbitalView` update on the converged state, no solve (module doc, "Orbital display").
    fn orbital_display_now(&mut self, should_abort: &mut dyn FnMut() -> bool) -> Option<OrbitalDisplay> {
        if !self.orb.on() || !self.density_current {
            return None;
        }
        self.orb.now(Instant::now());
        qm_gpu::grid::clear_readback_fault();
        let payload = self.orbital_payload(should_abort);
        if device_fault().is_ok() {
            payload
        } else {
            self.drop_view_after_fault();
            None
        }
    }

    fn set_display_interrupt(&mut self, interrupt: DisplayInterrupt) {
        self.interrupt = Some(interrupt);
    }

    fn set_share_tables(&mut self, tables: Option<Arc<ShareTables>>) {
        self.share_tables = tables;
    }

    /// A Harris display solve now (module doc, "Harris display"): always `Solved`.
    fn harris_display(&mut self, scene: &Scene, atoms: &Arc<HarrisAtoms>, should_abort: &mut dyn FnMut() -> bool) -> Option<HarrisAnswer> {
        Some(HarrisAnswer::Solved(self.harris_once(scene, atoms, should_abort)))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use sim::{Cell, Nucleus};
    use gpu::GpuContext;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    /// Serialises the GPU sections of concurrently running tests (poisoning is harmless here; `crate::fd_regression`
    /// takes it too).
    pub(crate) static GPU: Mutex<()> = Mutex::new(());

    fn gpu() -> Option<GpuContext> {
        let ctx = GpuContext::new_headless();
        if ctx.is_none() { eprintln!("providers: no GPU adapter — skipped"); }
        ctx
    }

    fn scene(n: usize, nuclei: Vec<Nucleus>) -> Scene {
        Scene { nuclei, cell: Cell { n, h: 0.25 }, charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3], ionic: sim::IonicPolicy::None, train_mask: None }
    }

    /// H₂ along x at bond length `r` (nucleus 0 at +r/2), grid n³ at h = 0.25.
    fn h2(n: usize, r: f64) -> Scene {
        scene(n, vec![Nucleus::new(1, [r / 2.0, 0.0, 0.0]), Nucleus::new(1, [-r / 2.0, 0.0, 0.0])])
    }

    /// H₂O at the `h2o` preset geometry (O–H 1.81 Bohr, 104.5°).
    fn h2o(n: usize) -> Scene {
        scene(n, vec![Nucleus::new(8, [0.0; 3]), Nucleus::new(1, [1.43, 1.11, 0.0]), Nucleus::new(1, [-1.43, 1.11, 0.0])])
    }

    fn never() -> impl FnMut() -> bool { || false }

    fn max_force_diff(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
        a.iter().zip(b).flat_map(|(x, y)| (0..3).map(move |k| (x[k] - y[k]).abs())).fold(0.0, f64::max)
    }

    /// R15's policy, without a solver: which attempts run, with which `warm`, and what comes
    /// back. Only a *warm* `ScfNotConverged` is retried, exactly once and cold; the retry's
    /// outcome (success with the stall recorded, or its own error) is returned.
    #[test]
    fn warm_stall_is_retried_once_cold() {
        use qm::QmError;
        type Script = Vec<Result<u32, QmError>>;
        fn run(warm: bool, mut script: Script) -> (Result<(u32, Option<Stall>), QmError>, Vec<bool>) {
            script.reverse();
            let mut calls = Vec::new();
            let r = retry_cold_on_stall(warm, |w| {
                calls.push(w);
                script.pop().expect("more attempts than scripted")
            });
            (r, calls)
        }
        // warm stall → cold retry converges: its value, the stall recorded
        let (r, calls) = run(true, vec![Err(QmError::ScfNotConverged(200, 1.2e-4)), Ok(7)]);
        assert_eq!(calls, [true, false]);
        let (v, stall) = r.expect("the retry converged");
        assert_eq!(v, 7);
        assert_eq!(stall, Some(Stall { iterations: 200, delta_rho: 1.2e-4, stalled: false }));
        let note = stall.unwrap().note();
        assert!(note.contains("retried cold") && note.contains("1.20e-4") && note.contains("200"), "{note}");
        // warm stall → cold retry stalls too: the retry's error, no third attempt
        let (r, calls) = run(true, vec![Err(QmError::ScfNotConverged(200, 1e-4)), Err(QmError::ScfNotConverged(150, 3e-5))]);
        assert_eq!(calls, [true, false]);
        assert!(matches!(r, Err(QmError::ScfNotConverged(150, d)) if d == 3e-5), "{r:?}");
        // a cold stall is not retried
        let (r, calls) = run(false, vec![Err(QmError::ScfNotConverged(200, 1e-4))]);
        assert_eq!(calls, [false]);
        assert!(matches!(r, Err(QmError::ScfNotConverged(200, _))), "{r:?}");
        // the stall stop is a non-convergence too: a warm stall is retried cold, a cold one is not
        let stalled = |n| QmError::ScfStalled { iterations: n, best_delta_rho: 3.1e-4, tol: 1e-6, window: 50, level: 2e-4 };
        let (r, calls) = run(true, vec![Err(stalled(61)), Ok(9)]);
        assert_eq!(calls, [true, false]);
        let (v, stall) = r.expect("the retry converged");
        assert_eq!((v, stall), (9, Some(Stall { iterations: 61, delta_rho: 3.1e-4, stalled: true })));
        let note = stall.unwrap().note();
        assert!(note.contains("stalled") && note.contains("3.10e-4") && note.contains("61") && note.contains("retried cold"), "{note}");
        let (r, calls) = run(false, vec![Err(stalled(55))]);
        assert_eq!(calls, [false]);
        assert!(matches!(r, Err(QmError::ScfStalled { iterations: 55, .. })), "{r:?}");
        // review S2: the boundary error names the floor-aware target with three digits and the stall level
        let e = solver_error(stalled(70)).to_string();
        assert!(e.contains("target 1.00e-6") && e.contains("still above the stall level 2.0e-4") && e.contains("stopped after 70 iterations"), "{e}");
        // a warm success, and a warm error of another kind, run once
        let (r, calls) = run(true, vec![Ok(3)]);
        assert_eq!(calls, [true]);
        assert_eq!(r.unwrap(), (3, None));
        let (r, calls) = run(true, vec![Err(QmError::Numerical("test".into()))]);
        assert_eq!(calls, [true]);
        assert!(matches!(r, Err(QmError::Numerical(_))), "{r:?}");
    }

    /// The "GPU busy" policy without a solver: only a `Device` error is retried, after the
    /// scripted pauses, at most `DEVICE_RETRIES` times; the final error names the attempts; a
    /// cancellation during a pause ends it; every other error is the first attempt's.
    #[test]
    fn a_device_fault_is_retried_a_bounded_number_of_times() {
        let dev = || ProviderError::Solver(SolverError::Device("callback dropped".into()));
        type Script = Vec<Result<u32, ProviderError>>;
        fn run(mut script: Script, alive: bool) -> (Result<(u32, usize), ProviderError>, usize, Vec<u64>) {
            script.reverse();
            let (mut calls, mut pauses) = (0, Vec::new());
            let r = retry_on_device_fault(|| { calls += 1; script.pop().expect("more attempts than scripted") }, |ms| { pauses.push(ms); alive });
            (r, calls, pauses)
        }
        let (r, calls, pauses) = run(vec![Err(dev()), Err(dev()), Ok(5)], true);
        assert_eq!((r.unwrap(), calls, pauses), ((5, 2), 3, DEVICE_BACKOFF_MS.to_vec()));
        let (r, calls, pauses) = run(vec![Err(dev()), Err(dev()), Err(dev())], true);
        assert_eq!((calls, pauses.len()), (DEVICE_RETRIES + 1, DEVICE_RETRIES));
        match r {
            Err(ProviderError::Solver(SolverError::Device(m))) => assert!(m.contains("callback dropped") && m.contains("gave up after 3 attempts"), "{m}"),
            other => panic!("{other:?}"),
        }
        let (r, calls, _) = run(vec![Err(dev())], false);
        assert!(matches!(r, Err(ProviderError::Cancelled)), "{r:?}");
        assert_eq!(calls, 1);
        let (r, calls, pauses) = run(vec![Err(ProviderError::Solver(SolverError::Numerical("x".into())))], true);
        assert!(matches!(r, Err(ProviderError::Solver(SolverError::Numerical(_)))), "{r:?}");
        assert_eq!((calls, pauses.len()), (1, 0));
        // the display message the worker shows
        let msg = ProviderError::Solver(SolverError::Device("callback dropped".into())).to_string();
        assert!(msg.starts_with("GPU busy"), "{msg}");
    }

    /// Through the real solver: one failed readback anywhere in an evaluation is retried and the
    /// result is the clean one (noted); a device that keeps failing ends in "GPU busy" after the
    /// bounded attempts and leaves the provider usable.
    #[test]
    fn a_failed_gpu_readback_is_retried_then_reported_as_gpu_busy() {
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let s = h2(49, 1.4);
        let before = qm_gpu::grid::readbacks_on_this_thread();
        let clean = GpuKs::new(ctx.clone(), 1e-6).evaluate(&s, &mut never()).expect("clean solve");
        let total = qm_gpu::grid::readbacks_on_this_thread() - before;
        assert!(total > 20, "test premise: an evaluation reads back many times ({total})");
        // the last two land after the SCF: the guard's spin measure (H₂ has N↑ = N↓; plan 2026-11-13 Task 8) and the
        // display maximum (the forces' readbacks just before them)
        for skip in [0, total / 2, total - 2, total - 1] {
            let mut g = GpuKs::new(ctx.clone(), 1e-6);
            qm_gpu::grid::clear_readback_fault();
            gpu::readback::fail_readbacks_after(skip, 1);
            let r = g.evaluate(&s, &mut never());
            gpu::readback::fail_readbacks_after(0, 0);
            let r = r.unwrap_or_else(|e| panic!("skip {skip}: {e}"));
            if skip < total - 2 {
                // inside the SCF: the failed solve left no state, the retry is a cold solve like
                // the clean one — the same bits (the `qm-gpu` injection test pins that)
                assert_eq!(r.energy.to_bits(), clean.energy.to_bits(), "skip {skip}: {} vs {}", r.energy, clean.energy);
            } else {
                // in the spin measure or the display maximum, after a converged SCF: the solver keeps
                // that state, so the retry is a warm solve from it, converged to the same Δρ tol
                // (1e-6) but not the same iterates: equal to the SCF's own accuracy, not bitwise
                assert!((r.energy - clean.energy).abs() < 1e-5, "skip {skip}: {} vs {}", r.energy, clean.energy);
            }
            assert!(r.stats.notes.iter().any(|n| n.contains("GPU busy")), "skip {skip}: {:?}", r.stats.notes);
        }
        let mut g = GpuKs::new(ctx, 1e-6);
        gpu::readback::fail_readbacks_after(0, 1000);
        let started = Instant::now();
        let e = g.evaluate(&s, &mut never()).err().expect("a device that never answers is an error");
        gpu::readback::fail_readbacks_after(0, 0);
        assert!(matches!(&e, ProviderError::Solver(SolverError::Device(m)) if m.contains("gave up after 3 attempts")), "{e:?}");
        assert!(e.to_string().starts_with("GPU busy"), "{e}");
        assert!(started.elapsed().as_millis() >= 500, "the backoff pauses ran");
        assert!(g.evaluate(&s, &mut never()).is_ok(), "the provider is usable once the device answers");
    }

    /// A readback that fails while the orbital view updates must not leave the view's tracker,
    /// previous-orbital copies and warm vectors committed from zeros: the evaluation keeps its
    /// solve (no "GPU busy", no retry) but sends no payload, drops the view, and the next
    /// refresh rebuilds it from the solver and shows the same two orbitals.
    #[test]
    fn a_display_fault_drops_the_orbital_view_and_the_next_refresh_rebuilds_it() {
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let s = h2(49, 1.4);
        let request = || OrbitalViewRequest::new(sim::OrbitalRequest::List);
        let reads = |g: &mut GpuKs| {
            let before = qm_gpu::grid::readbacks_on_this_thread();
            let r = g.evaluate(&s, &mut never()).expect("clean evaluation");
            (r, qm_gpu::grid::readbacks_on_this_thread() - before)
        };
        // the readbacks of an evaluation without the view: the view's start after them
        let (_, plain) = reads(&mut GpuKs::new(ctx.clone(), 1e-6));
        let mut clean = GpuKs::new(ctx.clone(), 1e-6);
        clean.set_orbital_view(request());
        let (r, with_view) = reads(&mut clean);
        let reference = r.orbitals.expect("a clean evaluation sends the payload");
        assert!(with_view > plain && clean.orbital.is_some(), "test premise: the view reads back ({plain} → {with_view})");

        let mut g = GpuKs::new(ctx, 1e-6);
        g.set_orbital_view(request());
        qm_gpu::grid::clear_readback_fault();
        gpu::readback::fail_readbacks_after(plain, 1); // the view's first readback
        let r = g.evaluate(&s, &mut never());
        gpu::readback::fail_readbacks_after(0, 0);
        let r = r.expect("a display fault is not the evaluation's failure");
        assert!(r.orbitals.is_none(), "no payload built from zeros");
        assert!(!r.stats.notes.iter().any(|n| n.contains("GPU busy")), "{:?}", r.stats.notes);
        assert!(g.orbital.is_none(), "the view that read the zeros is dropped");
        let d = g.orbital_display_now(&mut never()).expect("the next refresh rebuilds the view");
        assert!(g.orbital.is_some());
        assert_eq!(d.rows.len(), reference.rows.len());
        for (a, b) in d.rows.iter().zip(&reference.rows) {
            assert_eq!((a.spin, a.index, a.occupation), (b.spin, b.index, b.occupation));
            assert!((a.energy - b.energy).abs() < 1e-5, "{} vs {}", a.energy, b.energy);
        }
    }

    /// The empty scene (module doc, "Nothing to solve"): both adapters answer E = 0, no forces,
    /// no density, without a solve — even with the abort raised (there is nothing to cancel), and
    /// the GPU adapter builds no solver for it. A charge or a multiplicity ≠ 1 with no nuclei is
    /// an input error.
    #[test]
    fn an_empty_scene_has_nothing_to_solve() {
        let empty = scene(49, Vec::new());
        let check = |p: &mut dyn ForceProvider| {
            for abort in [false, true] {
                let r = p.evaluate(&empty, &mut || abort).unwrap_or_else(|e| panic!("{}: {e}", p.name()));
                assert_eq!(r.energy, 0.0);
                assert!(r.forces.is_empty() && r.display.is_none(), "{}", p.name());
                assert_eq!((r.stats.scf_steps, r.stats.source), (0, Source::Dft));
                assert_eq!(r.stats.energy.total, 0.0);
            }
            let mut singlet = empty.clone();
            singlet.multiplicity = Some(1);
            assert!(p.evaluate(&singlet, &mut never()).is_ok());
            for (charge, mult) in [(-1, None), (1, None), (0, Some(2))] {
                let mut bad = empty.clone();
                (bad.charge, bad.multiplicity) = (charge, mult);
                let e = p.evaluate(&bad, &mut never()).unwrap_err();
                assert!(matches!(&e, ProviderError::Invalid(m) if m.contains("no nuclei") || m.contains("0 electrons")), "{e:?}");
            }
        };
        check(&mut CpuKs::new(1e-6));
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let mut g = GpuKs::new(ctx, 1e-6);
        check(&mut g);
        assert!(g.solver.is_none(), "no solver built for an empty scene");
    }

    /// The GPU memory guard (module doc, "GPU memory"): the estimate grows with K and n³, and a
    /// `GpuKs` over its budget refuses the scene before building a solver.
    #[test]
    fn a_scene_over_the_vram_budget_is_refused_before_any_allocation() {
        assert_eq!(gpu_footprint_bytes(&scene(49, Vec::new())), 0);
        // the padding qm-gpu allocates: 4 ghost layers per side
        assert_eq!(qm_gpu::grid::GHOST, 4);
        assert_eq!(field_bytes(49), 57u64.pow(3) * 4);
        let small = gpu_footprint_bytes(&h2(49, 1.4));
        // same fields, so the ratio is exactly the padded volumes' (105/57)³
        let r = gpu_footprint_bytes(&h2(97, 1.4)) as f64 / small as f64;
        assert!((r - (105.0f64 / 57.0).powi(3)).abs() < 1e-6, "∝ (n + 2·GHOST)³: {r}");
        assert!(gpu_footprint_bytes(&h2o(49)) > small, "more orbitals");
        // H₂O at 97³ (K = 10): within 2× of `qm_gpu::scf`'s ~700 MB at 97³ with K = 8
        // re-pinned (plan 2026-11-13 Task 7 fix rounds 1–2): the estimate now adds the all-electron FD witness growth
        // (`witness_extra_bytes`: the counted start since Task 9c fix round 2, 36 fields); the solve part keeps its window
        let w = solve_footprint_bytes(97, 10, 5) as f64 / 1e6;
        assert!((600.0..1400.0).contains(&w), "{w} MB");
        assert_eq!(gpu_footprint_bytes(&h2o(97)), solve_footprint_bytes(97, 10, 5) + witness_extra_bytes(97));
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let mut g = GpuKs::new(ctx.clone(), 1e-6).with_vram_budget(small / 2);
        let e = g.evaluate(&h2(49, 1.4), &mut never()).unwrap_err();
        assert!(matches!(&e, ProviderError::Invalid(m) if m.contains("GPU memory") && m.contains("budget")), "{e}");
        assert!(g.solver.is_none(), "refused before the solver was built");
        assert!(GpuKs::new(ctx, 1e-6).with_vram_budget(small * 2).check_vram(&h2(49, 1.4)).is_ok());
    }

    /// Nuclei without electrons (H₂²⁺) still go to the solvers (module doc, "Nothing to solve"):
    /// both channels empty, the answer is the bare Coulomb repulsion, E = Z₁Z₂/R = 0.5 Ha at R = 2
    /// Bohr and F = ±1/R² = ±0.25 Ha/Bohr along the axis — finite, no NaN from the empty density.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn nuclei_without_electrons_repel_as_bare_charges() {
        let mut s = h2(33, 2.0);
        s.charge = 2;
        let check = |p: &mut dyn ForceProvider| {
            let r = p.evaluate(&s, &mut never()).unwrap_or_else(|e| panic!("{}: {e}", p.name()));
            println!("{}: H₂²⁺ R = 2: E = {:+.12} Ha, F = {:?}, {} SCF steps", p.name(), r.energy, r.forces, r.stats.scf_steps);
            assert!((r.energy - 0.5).abs() < 1e-12, "E = {}", r.energy);
            assert!((r.forces[0][0] - 0.25).abs() < 1e-9 && (r.forces[1][0] + 0.25).abs() < 1e-9, "{:?}", r.forces);
            assert!(r.forces.iter().flatten().all(|x| x.is_finite()));
        };
        check(&mut CpuKs::new(1e-6));
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        check(&mut GpuKs::new(ctx, 1e-6));
    }

    /// Pre-validation turns what `System::new` would panic on into an error (no solve runs).
    #[test]
    fn invalid_scene_is_an_error_not_a_panic() {
        let mut too_charged = h2(17, 1.4);
        too_charged.charge = 3;
        let mut bad_mult = h2(17, 1.4);
        bad_mult.multiplicity = Some(2); // 2 electrons cannot have one unpaired
        for s in [too_charged, bad_mult] {
            let err = CpuKs::new(1e-6).evaluate(&s, &mut never()).unwrap_err();
            assert!(matches!(err, ProviderError::Invalid(_)), "{err:?}");
            println!("rejected: {err}");
        }
    }

    /// The CPU adapter is `qm::scf` + `qm::forces` and nothing else: bitwise. Since plan 2026-11-13 Task 9 both sides run
    /// `for_grid`'s Fermi–Dirac occupations (the Task 7 Aufbau hold is gone), and the breakdown carries E and TS.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn cpu_provider_matches_qm_directly() {
        let s = h2(33, 1.4);
        let r = CpuKs::new(1e-6).evaluate(&s, &mut never()).expect("CpuKs");

        let grid = qm::Grid::cube(33, 0.25);
        let sys = qm::System::new(vec![qm::Nucleus { pos: [0.7, 0.0, 0.0], z: 1 }, qm::Nucleus { pos: [-0.7, 0.0, 0.0], z: 1 }], 0, None);
        let mut settings = qm::ScfSettings::for_grid(&grid);
        settings.tol = 1e-6;
        assert_eq!(settings.occupations, qm::Occupations::FermiDirac { kt: qm::ELECTRON_KT }, "the providers' rule is for_grid's");
        let (state, stats) = qm::scf(&grid, &sys, &settings, None).expect("qm::scf");
        let f = qm::forces(&grid, &state, &sys.nuclei, settings.sigma).expect("qm::forces");
        println!("H₂ 33³: E = {:+.12} Ha ({} steps), F = {:?}", r.energy, r.stats.scf_steps, r.forces);
        assert_eq!(r.energy.to_bits(), stats.energy.total.to_bits(), "{} vs {}", r.energy, stats.energy.total);
        assert_eq!(r.forces, f);
        assert_eq!(r.stats.scf_steps, stats.iterations);
        assert_eq!(r.stats.readbacks, None);
        assert_eq!(r.stats.warnings, stats.warnings);
        let (a, b) = (r.stats.energy, stats.energy);
        for (x, y) in [(a.kinetic, b.kinetic), (a.external, b.external), (a.hartree, b.hartree), (a.xc, b.xc), (a.nuclear, b.nuclear), (a.internal, b.internal), (a.ts, b.ts),
                       (a.total, b.total)] {
            assert_eq!(x.to_bits(), y.to_bits(), "energy breakdown {a:?} vs {b:?}");
        }
        assert_eq!(r.stats.scf_tol_used, Some(stats.tol_used), "F9: the stop the solve was tested against");
        assert!(r.stats.partial.is_empty() && a.ts <= 1e-12, "gapped H₂: no partial level, TS {:.1e}", a.ts);
        assert_eq!(r.stats.max_eigen_residual.to_bits(), stats.max_eigen_residual.to_bits());
        assert_eq!(r.stats.poisson_residual.to_bits(), stats.poisson_residual.to_bits());
        // the display payload is the converged density as f32, on the scene's cell
        let Some(Display::Cpu { cell, origin, rho, max }) = &r.display else { panic!("expected Display::Cpu, got {:?}", r.display) };
        assert_eq!(*cell, s.cell);
        assert_eq!(*origin, grid.origin);
        assert_eq!(rho.len(), 33 * 33 * 33);
        let expect: Vec<f32> = state.rho().iter().map(|&x| x as f32).collect();
        assert!(rho.as_slice() == expect.as_slice(), "display density differs from the state's");
        assert!(*max > 0.0 && *max == expect.iter().copied().fold(0.0f32, f32::max), "max {max}");
        println!("CPU display: max ρ = {max:.6} e/Bohr³, breakdown {a:?}, eig res {:.2e}, Poisson res {:.2e}",
                 r.stats.max_eigen_residual, r.stats.poisson_residual);
    }

    /// `frontier`: the highest level with f ≥ ½ and the lowest with f < ½, whatever the order; `None` for a channel
    /// without such a level (all occupied: the `Aufbau` block, or an empty channel).
    #[test]
    fn frontier_picks_the_half_occupation_split() {
        let f = frontier(&[-0.2, -0.5, 0.1, 0.05], &[0.5, 1.0, 1e-9, 0.49]);
        assert_eq!(f, sim::Frontier { homo: Some(-0.2), lumo: Some(0.05), mu: None });
        assert_eq!(frontier(&[-0.5, -0.3], &[1.0, 1.0]), sim::Frontier { homo: Some(-0.3), lumo: None, mu: None }, "no unoccupied level");
        assert_eq!(frontier(&[0.2], &[1e-12]), sim::Frontier { homo: None, lumo: Some(0.2), mu: None }, "no occupied level");
        assert_eq!(frontier(&[], &[]), sim::Frontier { homo: None, lumo: None, mu: None });
    }

    /// The frontier levels of H₂ at 1.4 Bohr (41³ × 0.25, automatic multiplicity): the closed shell's ↑ and ↓ HOMO agree,
    /// each below its channel's LUMO, and the HOMO is the solve's own lowest eigenvalue (σ_g, the one occupied level).
    fn assert_h2_frontier(stats: &EvalStats, lowest: f64) -> f64 {
        let [up, dn] = &stats.frontier;
        println!("H₂ 41³: ↑ {up:?}, ↓ {dn:?}, lowest ε {lowest:+.8}");
        let (hu, hd) = (up.homo.expect("↑ HOMO"), dn.homo.expect("↓ HOMO"));
        assert!((hu - hd).abs() < 1e-6, "closed shell: ↑ HOMO {hu} vs ↓ {hd}");
        for (s, f) in [("↑", up), ("↓", dn)] {
            let (h, l) = (f.homo.unwrap(), f.lumo.expect("a computed unoccupied level"));
            assert!(h < l, "{s}: HOMO {h} < LUMO {l}");
        }
        assert!((hu - lowest).abs() < 1e-6, "HOMO {hu} is the lowest eigenvalue {lowest}");
        hu
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn cpu_h2_reports_frontier_levels() {
        let mut c = CpuKs::new(1e-6);
        let r = c.evaluate(&h2(41, 1.4), &mut never()).expect("CpuKs");
        let state = c.warm.as_ref().expect("a converged state");
        let lowest = state.evals_up.iter().chain(&state.evals_dn).copied().fold(f64::INFINITY, f64::min);
        assert_h2_frontier(&r.stats, lowest);
    }

    /// As `cpu_h2_reports_frontier_levels` through the GPU provider, and its HOMO within 1e-3 Ha of the CPU's (f32 grid).
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn gpu_h2_reports_frontier_levels() {
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let s = h2(41, 1.4);
        let mut g = GpuKs::new(ctx, 1e-6);
        let r = g.evaluate(&s, &mut never()).expect("GpuKs");
        let state = g.solver.as_ref().and_then(|sv| sv.state()).expect("a converged state");
        let lowest = state.evals_up.iter().chain(&state.evals_dn).copied().fold(f64::INFINITY, f64::min);
        let gpu_homo = assert_h2_frontier(&r.stats, lowest);
        let cpu = CpuKs::new(1e-6).evaluate(&s, &mut never()).expect("CpuKs");
        let cpu_homo = cpu.stats.frontier[0].homo.expect("CPU ↑ HOMO");
        assert!((gpu_homo - cpu_homo).abs() < 1e-3, "GPU HOMO {gpu_homo} vs CPU {cpu_homo}");
    }

    /// GPU vs CPU on the same discrete problem (σ = h = 0.25, 33³, tol 1e-6) along a short
    /// equal-step stretch of H₂ — cold, warm, then warm + density extrapolation — against the
    /// validate suite's H₂ bounds (`gpu_vs_cpu.rs`: `E_BOUND_H2` 2e-6, `F_BOUND_H2` 5e-6).
    /// The 8-Bohr box is small for H₂ (both solvers warn: face density ~4e-5 > 1e-5); that is a
    /// property of the shared discrete problem, not a GPU/CPU difference, so the comparison
    /// stands.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn gpu_provider_matches_cpu_provider_at_fixed_sigma() {
        let Some(ctx) = gpu() else { return };
        let mut g = GpuKs::new(ctx, 1e-6);
        let mut c = CpuKs::new(1e-6);
        let mut cold_steps = 0;
        let mut held = Vec::new(); // every result's display is kept alive: no texture may be reused
        for (i, r) in [1.40, 1.42, 1.44].into_iter().enumerate() {
            let s = h2(33, r);
            c.reset(); // the CPU reference solves cold every time
            let rc = c.evaluate(&s, &mut never()).expect("CpuKs");
            let rg = {
                let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
                g.evaluate(&s, &mut never()).expect("GpuKs")
            };
            let de = (rg.energy - rc.energy).abs();
            let df = max_force_diff(&rg.forces, &rc.forces);
            println!("H₂ R = {r} 33³: E_gpu {:+.9} E_cpu {:+.9} |ΔE| {de:.2e}, max|ΔF| {df:.2e}; GPU {} steps / {:?} readbacks / eig tol {:?} / {} ms, CPU {} steps / {} ms; GPU notes {:?} warnings {:?}",
                     rg.energy, rc.energy, rg.stats.scf_steps, rg.stats.readbacks, rg.stats.eig_tol_used, rg.stats.solve_ms,
                     rc.stats.scf_steps, rc.stats.solve_ms, rg.stats.notes, rg.stats.warnings);
            if let Some(p) = g.solver.as_ref().map(|s| s.profile()) {
                println!("  GPU profile: witness growths {}, trims {}, buffer witnesses {}, buffer growths {}, DIIS resets {}, freeze trims {}, frozen steps {}, restores {}",
                         p.witness_growths, p.witness_trims, p.buffer_witnesses, p.buffer_growths, p.diis_resets, p.diis_freeze_trims, p.frozen_steps, p.eigen_restores);
            }
            assert!(de < 2e-6, "R = {r}: |ΔE| = {de:.3e}");
            assert!(df < 5e-6, "R = {r}: max |ΔF| = {df:.3e}");
            assert!(rg.stats.readbacks.is_some_and(|n| n > 0));
            assert!(rg.stats.eig_tol_used.is_some());
            match i {
                0 => cold_steps = rg.stats.scf_steps,
                _ => assert!(rg.stats.scf_steps <= cold_steps, "warm {} > cold {cold_steps} steps", rg.stats.scf_steps),
            }
            // extrapolation is on by default: the third solve follows two converged ones
            assert_eq!(rg.stats.notes.iter().any(|n| n.contains("extrapolated")), i == 2, "notes {:?}", rg.stats.notes);
            assert_eq!(rg.energy.to_bits(), rg.stats.energy.total.to_bits());
            assert!(rg.stats.max_eigen_residual.is_finite() && rg.stats.poisson_residual.is_finite());
            // display: GPU texture of the scene's cell; its max agrees with the CPU density's
            let Some(Display::Cpu { max: max_cpu, .. }) = rc.display else { panic!("expected Display::Cpu, got {:?}", rc.display) };
            let Some(Display::Gpu { cell, origin, volume, max }) = rg.display.clone() else { panic!("expected Display::Gpu, got {:?}", rg.display) };
            assert_eq!(cell, s.cell);
            assert_eq!(origin, qm::Grid::cube(33, 0.25).origin);
            assert_eq!(volume.dims, [33; 3]);
            let rel = ((max - max_cpu) / max_cpu).abs();
            println!("  display max ρ: GPU {max:.6}, CPU {max_cpu:.6} (rel {rel:.1e})");
            assert!(max > 0.0 && rel < 1e-3, "display max GPU {max} vs CPU {max_cpu}");
            assert!(held.iter().all(|v: &Arc<gpu::Volume3D>| !Arc::ptr_eq(v, &volume)), "a held texture was handed out again");
            held.push(volume);
        }
        assert_eq!(g.textures.len(), 3);
        // released by everyone: the pool reuses instead of allocating
        drop(held);
        let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
        g.evaluate(&h2(33, 1.46), &mut never()).expect("GpuKs");
        assert_eq!(g.textures.len(), 3, "a free texture was reused");
    }

    /// An abort is honoured at the next SCF step and leaves the provider usable.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn gpu_provider_cancels() {
        let Some(ctx) = gpu() else { return };
        let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let mut g = GpuKs::new(ctx, 1e-5);
        let s = h2o(97);
        // polls: 1 = before the solve (the provider), 2 = before SCF step 1, 3 = before step 2
        let mut polls = 0;
        let t = Instant::now();
        let err = g.evaluate(&s, &mut || { polls += 1; polls > 2 }).unwrap_err();
        let cancel_ms = t.elapsed().as_secs_f64() * 1e3;
        assert!(matches!(err, ProviderError::Cancelled), "{err:?}");
        assert_eq!(polls, 3, "abandoned at the first poll that asked for it");
        let t = Instant::now();
        let r = g.evaluate(&s, &mut never()).expect("the next evaluate succeeds");
        let solve_ms = t.elapsed().as_secs_f64() * 1e3;
        println!("H₂O 97³: cancelled after one SCF step in {cancel_ms:.0} ms (incl. solver build); next evaluate {solve_ms:.0} ms, {} steps, E {:+.6}, notes {:?}, warnings {:?}",
                 r.stats.scf_steps, r.energy, r.stats.notes, r.stats.warnings);
        assert!(r.energy.is_finite() && r.forces.len() == 3);
        assert!(r.forces.iter().flatten().all(|f| f.is_finite()));
    }

    /// `set_display(false)`: no payload, and on the GPU no texture at all.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn display_off_yields_none() {
        let mut c = CpuKs::new(1e-5);
        c.set_display(false);
        assert!(c.evaluate(&h2(17, 1.4), &mut never()).expect("CPU").display.is_none());

        let Some(ctx) = gpu() else { return };
        let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let mut g = GpuKs::new(ctx, 1e-5);
        let on = g.evaluate(&h2(17, 1.4), &mut never()).expect("GPU display on");
        assert!(matches!(on.display, Some(Display::Gpu { .. })));
        assert_eq!(g.textures.len(), 1);
        g.set_display(false);
        assert!(g.textures.is_empty(), "turning display off drops the pool's textures");
        let off = g.evaluate(&h2(17, 1.42), &mut never()).expect("GPU display off");
        assert!(off.display.is_none());
        assert!(g.textures.is_empty(), "no texture created with display off");
        drop(on);
    }

    /// A grid change (and `reset`) makes the next solve cold: as many steps as a fresh provider.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn grid_change_resets_warm_state() {
        // CPU: the 25³ solve after a 33³ one is bitwise the cold 25³ solve
        let mut c = CpuKs::new(1e-6);
        c.evaluate(&h2(33, 1.4), &mut never()).expect("CPU 33³");
        let after = c.evaluate(&h2(25, 1.4), &mut never()).expect("CPU 25³ after 33³");
        let cold = CpuKs::new(1e-6).evaluate(&h2(25, 1.4), &mut never()).expect("CPU 25³ cold");
        println!("CPU 25³: after a grid change {} steps, cold {} steps", after.stats.scf_steps, cold.stats.scf_steps);
        assert_eq!(after.energy.to_bits(), cold.energy.to_bits());
        assert_eq!(after.stats.scf_steps, cold.stats.scf_steps);

        let Some(ctx) = gpu() else { return };
        let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let mut g = GpuKs::new(ctx.clone(), 1e-6);
        g.evaluate(&h2(33, 1.4), &mut never()).expect("GPU 33³");
        let after = g.evaluate(&h2(25, 1.4), &mut never()).expect("GPU 25³ after 33³");
        let cold = GpuKs::new(ctx.clone(), 1e-6).evaluate(&h2(25, 1.4), &mut never()).expect("GPU 25³ cold");
        println!("GPU 25³: after a grid change {} steps / {:?} readbacks, cold {} steps / {:?} readbacks",
                 after.stats.scf_steps, after.stats.readbacks, cold.stats.scf_steps, cold.stats.readbacks);
        assert!(after.stats.scf_steps >= cold.stats.scf_steps);
        // a cold GPU solve is bit-reproducible: the post-change solve *is* the cold one
        assert_eq!(after.energy.to_bits(), cold.energy.to_bits());
        assert_eq!(after.stats.readbacks, cold.stats.readbacks);

        // reset on the same grid: warm → cold again
        let warm = g.evaluate(&h2(25, 1.42), &mut never()).expect("GPU warm");
        g.reset();
        let reset = g.evaluate(&h2(25, 1.42), &mut never()).expect("GPU after reset");
        let cold = GpuKs::new(ctx, 1e-6).evaluate(&h2(25, 1.42), &mut never()).expect("GPU cold");
        println!("GPU 25³ R 1.42: warm {} steps, after reset {} steps, cold {} steps", warm.stats.scf_steps, reset.stats.scf_steps, cold.stats.scf_steps);
        assert!(warm.stats.scf_steps < cold.stats.scf_steps, "test assumption: the warm start saves steps");
        assert!(reset.stats.scf_steps >= cold.stats.scf_steps);

        // a reset survives an evaluation that fails before solving (coincident nuclei keep the
        // solver's old state as its warm start)
        g.evaluate(&h2(25, 1.44), &mut never()).expect("GPU warm");
        g.reset();
        let err = g.evaluate(&h2(25, 0.0), &mut never()).unwrap_err();
        assert!(matches!(err, ProviderError::Solver(SolverError::CoincidentNuclei(0, 1))), "{err:?}");
        let after_err = g.evaluate(&h2(25, 1.42), &mut never()).expect("GPU after reset + error");
        println!("GPU 25³ R 1.42 after reset + coincident-nuclei error: {} steps (cold {})", after_err.stats.scf_steps, cold.stats.scf_steps);
        assert!(after_err.stats.scf_steps >= cold.stats.scf_steps);
    }

    /// The M = 33 points s_k = k/32 of the chord a → b (spec §3.1).
    fn chord(a: [f64; 3], b: [f64; 3]) -> impl Iterator<Item = [f64; 3]> {
        (0..33).map(move |k| { let s = k as f64 / 32.0; std::array::from_fn(|c| a[c] + s * (b[c] - a[c])) })
    }

    /// `sample_density` is `None` until an evaluation succeeds and again after `reset`,
    /// `discontinuity` or a failed evaluation; `Some` with one value per point after a converged
    /// one, on both adapters. H₂ at R = 1.6 (σ = h = 0.25, 33³, tol 1e-6): the midpoint density
    /// agrees between the adapters to 1e-5 and lies in [0.15, 0.30] e/Bohr³ (≈ 0.2 expected).
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn sample_density_is_none_before_a_solve_and_some_after_on_both_adapters() {
        let s = h2(33, 1.6);
        let (a, b) = (s.nuclei[0].pos, s.nuclei[1].pos);
        let mut points: Vec<[f64; 3]> = chord(a, b).collect();
        points.push([0.0; 3]); // the midpoint (also chord point 16)
        points.push([0.0, 0.0, 9.0]); // outside the 8 Bohr box → 0
        let mid = points.len() - 2;

        let mut c = CpuKs::new(1e-6);
        assert!(c.sample_density(&points).is_none(), "CPU: nothing solved yet");
        c.evaluate(&s, &mut never()).expect("CpuKs");
        let rc = c.sample_density(&points).expect("CPU: a converged density");
        assert_eq!(rc.len(), points.len());
        assert_eq!(rc, c.sample_density(&points).expect("CPU again"), "the cached ρ↑ + ρ↓ gives the same samples");
        // the CPU adapter samples the state's ρ↑ + ρ↓ with `Grid::trilinear`, nothing else
        let grid = qm::Grid::cube(33, 0.25);
        let state_rho = c.warm.as_ref().expect("warm state").rho();
        for (p, v) in points.iter().zip(&rc) { assert_eq!(v.to_bits(), grid.trilinear(&state_rho, *p).to_bits()); }
        c.reset();
        assert!(c.sample_density(&points).is_none(), "CPU: after reset");
        c.evaluate(&s, &mut never()).expect("CpuKs again");
        assert!(c.sample_density(&points).is_some());
        c.discontinuity();
        assert!(c.sample_density(&points).is_none(), "CPU: after a discontinuity the density is the old geometry's");
        c.evaluate(&s, &mut never()).expect("CpuKs after the teleport");
        let mut bad = h2(33, 1.6);
        bad.charge = 3;
        c.evaluate(&bad, &mut never()).unwrap_err();
        assert!(c.sample_density(&points).is_none(), "CPU: after a failed evaluation");

        let Some(ctx) = gpu() else { return };
        let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let mut g = GpuKs::new(ctx, 1e-6);
        assert!(g.sample_density(&points).is_none(), "GPU: nothing solved yet");
        g.evaluate(&s, &mut never()).expect("GpuKs");
        let t = Instant::now();
        let rg = g.sample_density(&points).expect("GPU: a converged density");
        let ms = t.elapsed().as_secs_f64() * 1e3;
        assert_eq!(rg.len(), points.len());
        let chord_diff = rg[..33].iter().zip(&rc[..33]).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        println!("H₂ R = 1.6 33³: midpoint ρ CPU {:.6} GPU {:.6} (|Δ| {:.2e}); chord max |Δ| {chord_diff:.2e}; GPU sample of {} points {ms:.2} ms",
                 rc[mid], rg[mid], (rg[mid] - rc[mid]).abs(), points.len());
        println!("  chord ρ (CPU): {}", rc[..33].iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(" "));
        assert!((rg[mid] - rc[mid]).abs() <= 1e-5, "midpoint GPU {} vs CPU {}", rg[mid], rc[mid]);
        assert!((0.15..=0.30).contains(&rc[mid]), "H₂ midpoint density {} e/Bohr³", rc[mid]);
        assert_eq!(rc[mid].to_bits(), rc[16].to_bits(), "the midpoint is chord point 16");
        assert_eq!((rc[mid + 1], rg[mid + 1]), (0.0, 0.0), "outside the box");
        g.reset();
        assert!(g.sample_density(&points).is_none(), "GPU: after reset (the solver still holds the state, but it must not be handed out)");
        g.evaluate(&s, &mut never()).expect("GpuKs after reset");
        assert!(g.sample_density(&points).is_some());
        g.discontinuity();
        assert!(g.sample_density(&points).is_none(), "GPU: after a discontinuity");
        g.evaluate(&s, &mut never()).expect("GpuKs after the teleport");
        assert!(g.evaluate(&s, &mut || true).is_err(), "cancelled");
        assert!(g.sample_density(&points).is_none(), "GPU: after a cancelled evaluation");
    }

    /// The detector's round 1 end to end through `GpuKs` on a 10-nucleus scene (five H₂ at
    /// R = 1.4 on a ring of radius 4 Bohr, 65³, h = 0.25): all 45 chords × 33 points in one call,
    /// equal to `Grid::trilinear` on the solver's own f32 density to 1e-6·max(1, ρ); the wall
    /// per call is printed (median of 20).
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn chord_sampling_of_a_ten_nucleus_scene_through_gpu_ks() {
        let Some(ctx) = gpu() else { return };
        let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let nuclei: Vec<Nucleus> = (0..5).flat_map(|m| {
            let phi = 2.0 * std::f64::consts::PI * m as f64 / 5.0;
            let c = [4.0 * phi.cos(), 4.0 * phi.sin(), 0.0];
            [Nucleus::new(1, [c[0], c[1], 0.7]), Nucleus::new(1, [c[0], c[1], -0.7])]
        }).collect();
        let s = scene(65, nuclei);
        let mut g = GpuKs::new(ctx, 1e-5);
        g.set_display(false);
        let t = Instant::now();
        let r = g.evaluate(&s, &mut never()).expect("GpuKs H₁₀");
        let solve_ms = t.elapsed().as_secs_f64() * 1e3;
        let pos: Vec<[f64; 3]> = s.nuclei.iter().map(|n| n.pos).collect();
        let points: Vec<[f64; 3]> = (0..10).flat_map(|i| ((i + 1)..10).map(move |j| (i, j)))
            .flat_map(|(i, j)| chord(pos[i], pos[j])).collect();
        assert_eq!(points.len(), 45 * 33);
        let mut walls = Vec::new();
        let mut last = Vec::new();
        for _ in 0..20 {
            let t = Instant::now();
            last = g.sample_density(&points).expect("converged");
            walls.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let first = walls[0];
        walls.sort_by(f64::total_cmp);
        let solver = g.solver.as_ref().expect("solver");
        let rho: Vec<f64> = solver.density_f32(solver.state().expect("state")).iter().map(|&v| v as f64).collect();
        let grid = qm::Grid::cube(65, 0.25);
        let mut worst = 0.0f64;
        for (p, v) in points.iter().zip(&last) {
            let c = grid.trilinear(&rho, *p);
            let bound = 1e-6 * c.abs().max(1.0);
            worst = worst.max((v - c).abs() / bound);
            assert!((v - c).abs() <= bound, "p = {p:?}: GPU {v} vs CPU trilinear {c}");
        }
        let bond_mid = last[16]; // chord (0, 1) is the first molecule's bond
        println!("H₁₀ 65³: solve {solve_ms:.0} ms ({} SCF steps); 45 chords × 33 = {} points per call: median {:.3} ms, min {:.3}, max {:.3} (first call {:.3}); worst |Δ| vs CPU trilinear {worst:.3} of the bound; ρ at a bond midpoint {bond_mid:.4}",
                 r.stats.scf_steps, points.len(), walls[10], walls[0], walls[19], first);
        assert!(bond_mid > 0.15, "H₂ bond midpoint {bond_mid}");
    }

    /// R12: a nucleus closer than 3h to a face, or outside the cell, is refused before any solve
    /// (33³ at h = 0.25: half-edge 4.0, max |coordinate| 3.25 Bohr), by both adapters.
    #[test]
    fn nuclei_near_or_outside_the_cell_face_are_refused() {
        let mut s = h2(33, 1.4);
        assert_eq!((s.cell.half_extent(), s.cell.max_coord()), (4.0, 3.25));
        s.nuclei[1].pos = [0.0, 3.25, 0.0];
        assert!(s.check_inside_cell().is_ok(), "exactly 3h from the face is allowed");
        s.nuclei[1].pos = [0.0, 3.3, 0.0];
        let near = CpuKs::new(1e-5).evaluate(&s, &mut never()).unwrap_err();
        println!("{near}");
        assert!(matches!(&near, ProviderError::Invalid(m) if m.contains("nucleus 1 (H) left the simulation cell") && m.contains("0.70 Bohr from its face")), "{near}");
        s.nuclei[1].pos = [0.0, 0.0, -4.5];
        let out = CpuKs::new(1e-5).evaluate(&s, &mut never()).unwrap_err();
        assert!(matches!(&out, ProviderError::Invalid(m) if m.contains("0.50 Bohr outside its face")), "{out}");
        if let Some(ctx) = gpu() {
            let e = GpuKs::new(ctx, 1e-5).evaluate(&s, &mut never()).unwrap_err();
            assert!(matches!(&e, ProviderError::Invalid(m) if m.contains("left the simulation cell")), "{e}");
        }
    }

    /// M6 Task 3 diagnostic (Task 2's physics flag; Task 7 pins the answer on the GPU): which
    /// route the bond detector takes for H₂ on this surface once the chord is bias-corrected. The
    /// H reference is solved by `sim::references::solve_atom` through `CpuKs` (33³, σ = h = 0.25,
    /// tol 1e-5), then H₂ at R = 1.6 (the app's start) and at R_e(σ) = 1.789: the raw and the
    /// corrected chords, q, and the `PairReport` are printed; asserted only that both are
    /// `Bonded`. Numbers of record: `validate/references.md`, "Tools (M6)".
    #[test]
    #[ignore = "diagnostic: cargo test -p providers --release diag_h2_bond_route -- --ignored --nocapture (three CPU solves at 33³)"]
    fn diag_h2_bond_route_on_the_cpu() {
        use sim::bonds::{Detector, Promolecule, Verdict};
        use sim::references::{solve_atom, AtomReferences, ReferenceKey};
        let mut c = CpuKs::new(1e-5);
        let cell = Cell { n: 33, h: 0.25 };
        let t0 = Instant::now();
        let h = solve_atom(&mut c, &cell, None, sim::IonicPolicy::None, 1, Nucleus::new(1, [0.0; 3]).mass, &mut never()).expect("H reference");
        println!(
            "H reference ({:.1} s): E = {:.6} Ha, multiplicity {:?}, {} shell entries to r = {:.3}, {:.4} electrons in the table; ρ̄(0) = {:.4}, ρ̄(0.8) = {:.4}, ρ̄(1.6) = {:.4}",
            t0.elapsed().as_secs_f64(), h.energy, h.multiplicity, h.radial.r.len(), h.radial.r.last().unwrap(), h.radial.electrons(), h.radial.at(0.0), h.radial.at(0.8), h.radial.at(1.6)
        );
        let name = c.name();
        let key = |z: u8| ReferenceKey::new(&cell, None, Some(1e-5), name, z, "none");
        let mut refs = AtomReferences::default();
        refs.insert(key(1), h);
        let fmt = |v: &[f64]| v.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(" ");
        for r in [1.6, 1.789] {
            let s = h2(33, r);
            c.reset();
            let t0 = Instant::now();
            let res = c.evaluate(&s, &mut never()).expect("H₂");
            let pos: Vec<[f64; 3]> = s.nuclei.iter().map(|n| n.pos).collect();
            let pro = Promolecule::from_refs(&refs, key, &cell, &pos, &[1, 1]).expect("the H table");
            let mut det = Detector::new(&pos);
            let chord = det.points().to_vec();
            let raw = c.sample_density(&chord).expect("a density after the solve");
            let corrected: Vec<f64> = chord.iter().zip(&raw).map(|(&p, &v)| pro.corrected(p, v)).collect();
            let q: Vec<f64> = chord.iter().zip(&raw).map(|(&p, &v)| pro.ratio(p, v)).collect();
            println!("H₂ R = {r} ({:.1} s, E = {:.6} Ha, {} SCF steps):", t0.elapsed().as_secs_f64(), res.energy, res.stats.scf_steps);
            println!("  raw ρ_tri   {}", fmt(&raw));
            println!("  corrected ρ̂ {}", fmt(&corrected));
            println!("  q = ρ/ρ_pro^tri {}", fmt(&q));
            let mut pts = chord;
            loop {
                let rho = c.sample_density(&pts).expect("a density");
                match det.feed(&rho, &pro, cell.h) {
                    Some(next) => pts = next.to_vec(),
                    None => break,
                }
            }
            let rep = &det.reports()[0];
            println!("  report: {rep:?}");
            assert!(matches!(rep.verdict, Verdict::Bonded { .. }), "{rep:?}");
        }
    }

    /// Spec A1, Review Focus 3: the probe runs during a GPU solve, counts its SCF steps as the
    /// solver reports them, and clears on success, cancellation and error. `solve_with` polls
    /// once before every SCF step and the converged step ends the loop, so the probe's count is
    /// exactly `scf_steps` (no extra pre-loop poll: `evaluate`'s own pre-solve abort check is not
    /// a step and is not counted).
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn the_probe_clears_on_error_and_cancel() {
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let probe = sim::ActivityProbe::new();
        let mut p = GpuKs::new(ctx, 1e-6).with_probe(probe.clone(), sim::Slot::Main);
        let scene = h2(25, 1.4);
        let r = p.evaluate(&scene, &mut || false).unwrap();
        let v = probe.view();
        assert!(!v.main.running);
        assert_eq!(v.main.scf_step as usize, r.stats.scf_steps, "one poll per SCF step");
        // cold, so the solve outlasts the third poll (a warm start at the converged geometry
        // can finish in one or two steps)
        p.reset();
        // abort at the poll before step 3: steps 1 and 2 ran. (Since plan 2026-11-13 Task 9c the same `should_abort` is
        // also polled inside the FD eigensolves, uncounted, so the step is read off the probe, not a call count.) That
        // poll is counted, so the idle count reads the step being solved when it stopped.
        let watch = probe.clone();
        let mut pre_solve = true; // `evaluate`'s own pre-solve check reads the last solve's count: never abort there
        let mut abort_third = || !std::mem::take(&mut pre_solve) && watch.view().main.scf_step >= 3;
        assert!(matches!(p.evaluate(&scene, &mut abort_third), Err(ProviderError::Cancelled)));
        let v = probe.view();
        assert!(!v.main.running);
        assert_eq!((v.main.scf_step, v.main.solves), (3, 2));
        let mut bad = scene.clone();
        bad.nuclei[1].pos = bad.nuclei[0].pos; // coincident nuclei: an error
        assert!(p.evaluate(&bad, &mut || false).is_err());
        let v = probe.view();
        assert!(!v.main.running);
        // exactly 3: the coincident-nuclei error is raised inside `solve_with`
        // (`nuclear_repulsion`, before its first poll), after the guard was taken — not by
        // `problem()`; if it ever moves there this count drops to 2 and must be looked at
        assert_eq!((v.main.solves, v.main.scf_step), (3, 0));
        println!("probe after success, cancel, error: {:?}", probe.view());
    }

    /// Attaching the probe does not change a solve (the wrapped poll returns `should_abort()`
    /// unchanged): a cold H₂ 25³ solve with and without it, bitwise. `validate --test
    /// gpu_vs_cpu` drives `GpuSolver` directly and cannot see the wrapper.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn the_probe_leaves_a_gpu_solve_unchanged() {
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let probe = sim::ActivityProbe::new();
        let s = h2(25, 1.4);
        let a = GpuKs::new(ctx.clone(), 1e-6).evaluate(&s, &mut never()).unwrap();
        let b = GpuKs::new(ctx, 1e-6).with_probe(probe.clone(), sim::Slot::Main).evaluate(&s, &mut never()).unwrap();
        assert_eq!(a.energy.to_bits(), b.energy.to_bits(), "{} vs {}", a.energy, b.energy);
        assert_eq!(a.forces, b.forces);
        assert_eq!(a.stats.scf_steps, b.stats.scf_steps);
        assert_eq!(probe.view().main.scf_step as usize, b.stats.scf_steps);
    }

    /// The CPU solver has no per-step poll: the probe marks it running for the solve and counts
    /// it, with `scf_step` 0 (`sim::activity::SlotView::scf_step` doc).
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn the_cpu_probe_marks_the_solve_without_steps() {
        let probe = sim::ActivityProbe::new();
        let mut p = CpuKs::new(1e-6).with_probe(probe.clone(), sim::Slot::Side);
        p.evaluate(&h2(17, 1.4), &mut never()).unwrap();
        let v = probe.view();
        assert!(!v.side.running && !v.main.running);
        assert_eq!((v.side.solves, v.side.scf_step, v.main.solves), (1, 0, 0));
    }

    /// Spec §6 (< 1 % overhead), reported never asserted: the probe's per-SCF-step cost (one
    /// relaxed `fetch_add` through the wrapped abort poll) against a measured GPU SCF step.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn probe_overhead_per_scf_step_is_reported() {
        let probe = sim::ActivityProbe::new();
        let calls = 10_000_000u32;
        let time = |with: Option<&(Arc<sim::ActivityProbe>, sim::Slot)>| {
            let mut abort = never();
            let mut polled = || {
                if let Some((p, s)) = with {
                    p.scf_step(*s);
                }
                abort()
            };
            let poll: &mut dyn FnMut() -> bool = &mut polled;
            let t = Instant::now();
            let mut any = false;
            for _ in 0..calls {
                any |= std::hint::black_box(&mut *poll)();
            }
            assert!(!any);
            t.elapsed().as_secs_f64() * 1e9 / calls as f64
        };
        let attached = (probe.clone(), sim::Slot::Main);
        let (bare, with) = (time(None), time(Some(&attached)));
        assert_eq!(probe.view().main.scf_step, calls);
        let per_step_ns = (with - bare).max(0.0);
        println!("probe poll: {bare:.2} ns bare, {with:.2} ns with the probe → {per_step_ns:.2} ns per SCF step");
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let mut g = GpuKs::new(ctx, 1e-6).with_probe(probe, sim::Slot::Main);
        let s = h2(49, 1.4);
        g.evaluate(&s, &mut never()).unwrap(); // warm the pipelines
        g.reset();
        let r = g.evaluate(&s, &mut never()).unwrap();
        let step_ms = r.stats.solve_ms as f64 / r.stats.scf_steps.max(1) as f64;
        println!("H₂ 49³ cold GPU solve: {} ms, {} SCF steps → {step_ms:.2} ms per step; probe overhead {:.1e} of a step",
                 r.stats.solve_ms, r.stats.scf_steps, per_step_ns * 1e-6 / step_ms.max(1e-9));
    }

    /// Activity Task 5 (spec §6): the median of 20 warm H₂ 49³ GPU solves (each a 0.01 Bohr
    /// stretch from the last) with and without `with_probe`, interleaved (and alternating which
    /// runs first) so drift and ordering affect both alike. A measurement driver: printed, never asserted; `#[ignore]`d.
    #[test]
    #[ignore = "diagnostic: cargo test -p providers --release diag_probe_cost -- --ignored --nocapture"]
    fn diag_probe_cost() {
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let probe = sim::ActivityProbe::new();
        let mut bare = GpuKs::new(ctx.clone(), 1e-6);
        let mut with = GpuKs::new(ctx, 1e-6).with_probe(probe.clone(), sim::Slot::Main);
        // the cold first solves (pipelines, no warm start) are not timed
        bare.evaluate(&h2(49, 1.4), &mut never()).unwrap();
        with.evaluate(&h2(49, 1.4), &mut never()).unwrap();
        let (mut tb, mut tw) = (Vec::new(), Vec::new());
        for k in 1..=20 {
            let s = h2(49, 1.4 + 0.01 * k as f64);
            // alternating which goes first: the second of a pair runs right behind the first
            let mut pair = [(&mut bare, &mut tb), (&mut with, &mut tw)];
            if k % 2 == 0 {
                pair.reverse();
            }
            for (p, out) in pair {
                let t = Instant::now();
                p.evaluate(&s, &mut never()).unwrap();
                out.push(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            0.5 * (v[9] + v[10])
        };
        let (mb, mw) = (med(&mut tb), med(&mut tw));
        println!(
            "H₂ 49³ warm GPU solves (20 each): median {mb:.2} ms without the probe, {mw:.2} ms with it ({:+.2} %); probe {:?}",
            100.0 * (mw / mb - 1.0),
            probe.view().main
        );
    }

    // ---- the orbital display (module doc, "Orbital display"; orbital-viewer plan Task 3) ----

    use sim::{OrbitalRequest, OrbitalViewRequest, Spin};

    fn view(what: OrbitalRequest) -> OrbitalViewRequest {
        OrbitalViewRequest { what, unoccupied: 0, force_every_step: true }
    }

    /// The track of the first row of `spin` (the lowest orbital of that channel).
    fn first_track(o: &OrbitalDisplay, spin: Spin) -> u32 {
        o.rows.iter().find(|r| r.spin == spin && r.index == 0).expect("a row of that channel").track
    }

    /// Spec §4.4 / Review Focus 1: 20 velocity-Verlet steps of H₂ (49³, from 1.6 Bohr, dt 0.25 fs,
    /// `GpuKs` at tol 1e-6) twice — the view off, and the view cycling List → Orbital(σ_g ↑) →
    /// AllColoured → SpinDensity at every step with `force_every_step` — give bitwise-equal
    /// energies and forces at every step; exactly 20 payloads arrive (review SHOULD 13: the rate
    /// bound cannot make the test vacuous), each with the mode's field.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn orbital_view_does_not_change_the_dynamics() {
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let start = h2(49, 1.6);
        let integrator = sim::Integrator::new_fs(0.25);
        let run = |on: bool| -> (Vec<(u64, Vec<[u64; 3]>)>, usize) {
            let mut p = GpuKs::new(ctx.clone(), 1e-6);
            let mut scene = start.clone();
            let first = p.evaluate(&scene, &mut never()).expect("the first solve");
            let bits = |e: f64, f: &[[f64; 3]]| (e.to_bits(), f.iter().map(|v| v.map(f64::to_bits)).collect::<Vec<_>>());
            let mut out = vec![bits(first.energy, &first.forces)];
            let mut f = first.forces;
            let (mut payloads, mut sigma_g) = (0, None);
            for i in 0..20 {
                if on {
                    p.set_orbital_view(view(match i % 4 {
                        0 => OrbitalRequest::List,
                        1 => OrbitalRequest::Orbital { spin: Spin::Up, track: sigma_g.expect("listed at the step before"), squared: false },
                        2 => OrbitalRequest::AllColoured { include: None, physical: false, look: Default::default() },
                        _ => OrbitalRequest::SpinDensity,
                    }));
                }
                let (f_new, stats) = integrator.step(&mut scene, &f, &mut p, &mut |_, _| {}, &mut never()).expect("a step");
                if let Some(o) = &stats.orbitals {
                    payloads += 1;
                    sigma_g = Some(first_track(o, Spin::Up));
                    let ok = match (i % 4, &o.field) {
                        (0, None) => true,
                        (1, Some(OrbitalField::Gpu { signed: true, scale, .. })) => *scale > 0.0,
                        (2, Some(OrbitalField::GpuColour { .. })) => true,
                        // the spin density of this closed-shell H₂ is solver noise: no field is sent
                        // (`SpinSummary::noise_l1`), and the payload says why
                        (3, None) => o.spin.no_net_spin() && o.notes.first().map(String::as_str) == Some(crate::orbitals::NO_NET_SPIN),
                        _ => false,
                    };
                    assert!(ok, "step {i}: field {:?} for {:?}", o.field, o.what);
                    assert_eq!(o.rows.len(), 2, "σ_g ↑ and ↓");
                }
                out.push(bits(stats.energy_ks, &f_new));
                f = f_new;
            }
            (out, payloads)
        };
        let (off, n_off) = run(false);
        let (on, n_on) = run(true);
        println!("H₂ 49³, 20 steps: payloads off {n_off}, on {n_on}; E[20] off {:.12} on {:.12}", f64::from_bits(off[20].0), f64::from_bits(on[20].0));
        assert_eq!((n_off, n_on), (0, 20));
        for (k, (a, b)) in off.iter().zip(&on).enumerate() {
            assert_eq!(a, b, "step {k}: energy or forces differ with the view on");
        }
    }

    /// Wraps a provider and counts the orbital payloads it hands out (evaluations and
    /// `orbital_display_now`).
    struct Counted<P> {
        inner: P,
        payloads: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl<P: ForceProvider> ForceProvider for Counted<P> {
        fn evaluate(&mut self, scene: &Scene, a: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
            let r = self.inner.evaluate(scene, a)?;
            if r.orbitals.is_some() {
                self.payloads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
            Ok(r)
        }
        fn name(&self) -> &'static str {
            self.inner.name()
        }
        fn reset(&mut self) {
            self.inner.reset()
        }
        fn discontinuity(&mut self) {
            self.inner.discontinuity()
        }
        fn sample_density(&mut self, points: &[[f64; 3]]) -> Option<Vec<f64>> {
            self.inner.sample_density(points)
        }
        fn scf_tol(&self) -> Option<f64> {
            self.inner.scf_tol()
        }
        fn set_orbital_view(&mut self, request: OrbitalViewRequest) {
            self.inner.set_orbital_view(request)
        }
        fn orbital_display_now(&mut self, a: &mut dyn FnMut() -> bool) -> Option<OrbitalDisplay> {
            let o = self.inner.orbital_display_now(a);
            if o.is_some() {
                self.payloads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
            o
        }
    }

    /// Spec §4.4, "Rate": `sim::Dynamics` on `GpuKs` (H₂ 33³, tol 1e-5) playing at full speed for
    /// 2 s of wall with the view on (the app's `force_every_step` off): at most 21 payloads
    /// (one per 100 ms, plus the first); each paused single step: exactly one.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn payload_rate_is_bounded_in_play() {
        use sim::{DynCommand, Dynamics};
        use std::sync::atomic::Ordering::SeqCst;
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let payloads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut g = GpuKs::new(ctx, 1e-5);
        g.set_display(false);
        let p = Counted { inner: g, payloads: payloads.clone() };
        let mut d = Dynamics::new(h2(33, 1.6), Box::new(p), 0.25, 1);
        d.apply(DynCommand::SetOrbitalView(OrbitalViewRequest::new(OrbitalRequest::List)));
        assert_eq!(payloads.load(SeqCst), 1, "switched on while paused: shown at once");
        d.apply(DynCommand::SetSpeed(sim::dynamics::MAX_SPEED));
        d.apply(DynCommand::Play);
        let (t0, before) = (Instant::now(), payloads.load(SeqCst));
        let mut steps = 0;
        while t0.elapsed().as_secs_f64() < 2.0 {
            steps += d.tick(16.0, &mut never()).steps_last_tick;
        }
        let wall = t0.elapsed().as_secs_f64();
        d.apply(DynCommand::Pause);
        let in_play = payloads.load(SeqCst) - before;
        println!("play {wall:.2} s: {steps} steps, {in_play} payloads (≤ 21 allowed)");
        assert!(in_play <= 21, "{in_play} payloads in {wall:.2} s of play");
        assert!(in_play >= 1, "the view kept updating in play");
        if steps > 25 {
            assert!(in_play < steps, "the bound skipped payloads ({steps} steps)");
        }
        for k in 0..3 {
            let b = payloads.load(SeqCst);
            d.apply(DynCommand::Step(1));
            let s = d.tick(f64::INFINITY, &mut never());
            assert_eq!(s.steps_last_tick, 1);
            assert_eq!(payloads.load(SeqCst) - b, 1, "paused single step {k}: exactly one payload");
        }
    }

    /// Spec §7, "View switched on while paused": after one evaluation the view needs no solve —
    /// the list, then a selected orbital, come from the converged state (the solver's profile,
    /// i.e. its last solve, is untouched); a reset or a teleport makes it `None` until the next
    /// evaluation.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn view_on_while_paused_needs_no_solve() {
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let mut g = GpuKs::new(ctx, 1e-6);
        assert!(g.orbital_display_now(&mut || false).is_none(), "nothing solved, view off");
        g.evaluate(&h2(33, 1.6), &mut never()).expect("GpuKs");
        let p0 = g.solver.as_ref().unwrap().profile().clone();
        assert!(g.orbital_display_now(&mut || false).is_none(), "view off");
        g.set_orbital_view(OrbitalViewRequest::new(OrbitalRequest::List));
        let list = g.orbital_display_now(&mut || false).expect("the list without a solve");
        assert!(list.field.is_none() && list.rows.len() == 2);
        let t = first_track(&list, Spin::Up);
        g.set_orbital_view(OrbitalViewRequest::new(OrbitalRequest::Orbital { spin: Spin::Up, track: t, squared: false }));
        let o = g.orbital_display_now(&mut || false).expect("the orbital without a solve");
        assert_eq!(o.what, OrbitalRequest::Orbital { spin: Spin::Up, track: t, squared: false });
        assert!(matches!(o.field, Some(OrbitalField::Gpu { signed: true, scale, .. }) if scale > 0.0), "{:?}", o.field);
        assert_eq!(first_track(&o, Spin::Up), t, "the same track (continuity)");
        let p1 = g.solver.as_ref().unwrap().profile();
        assert_eq!((p1.steps, p1.readbacks, p1.wall), (p0.steps, p0.readbacks, p0.wall), "no solve");
        assert_eq!(o.positions, vec![[0.8, 0.0, 0.0], [-0.8, 0.0, 0.0]]);
        g.discontinuity();
        assert!(g.orbital_display_now(&mut || false).is_none(), "after a teleport: the old geometry's state");
        g.evaluate(&h2(33, 1.62), &mut never()).expect("GpuKs");
        assert!(g.orbital_display_now(&mut || false).is_some());
        g.reset();
        assert!(g.orbital_display_now(&mut || false).is_none(), "after a reset");
    }

    /// The CPU and GPU producers on the same discrete problem (H₂ 49³, σ = h = 0.25, tol 1e-6):
    /// the same rows (count, channels, labels, occupations) with ε within 1e-4 Ha, and the fields
    /// they make for a selected orbital and the spin density agree in kind and scale.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn cpu_and_gpu_rows_agree() {
        let Some(ctx) = gpu() else { return };
        let s = h2(49, 1.6);
        let mut c = CpuKs::new(1e-6);
        c.set_orbital_view(view(OrbitalRequest::List));
        let rc = c.evaluate(&s, &mut never()).expect("CpuKs");
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let mut g = GpuKs::new(ctx, 1e-6);
        g.set_orbital_view(view(OrbitalRequest::List));
        let rg = g.evaluate(&s, &mut never()).expect("GpuKs");
        let (oc, og) = (rc.orbitals.expect("CPU payload"), rg.orbitals.expect("GPU payload"));
        let key = |o: &OrbitalDisplay| o.rows.iter().map(|r| (r.spin, r.index, r.label.to_string(), r.occupation)).collect::<Vec<_>>();
        println!("rows CPU {:?}\n     GPU {:?}", key(&oc), key(&og));
        assert_eq!(key(&oc), key(&og));
        assert_eq!(oc.lone_atom, og.lone_atom);
        for (a, b) in oc.rows.iter().zip(&og.rows) {
            println!("{} {:?}: ε CPU {:.7} GPU {:.7} (Δ {:.1e})", a.label, a.spin, a.energy, b.energy, (a.energy - b.energy).abs());
            assert!((a.energy - b.energy).abs() <= 1e-4, "{}: ε CPU {} GPU {}", a.label, a.energy, b.energy);
        }
        // the fields: σ_g ↑ (phase) and the spin density, without a solve on either side
        let scales = |p: &mut dyn ForceProvider, o: &OrbitalDisplay| -> (f32, f32) {
            p.set_orbital_view(OrbitalViewRequest::new(OrbitalRequest::Orbital { spin: Spin::Up, track: first_track(o, Spin::Up), squared: false }));
            let a = p.orbital_display_now(&mut || false).expect("orbital");
            p.set_orbital_view(OrbitalViewRequest::new(OrbitalRequest::SpinDensity));
            let b = p.orbital_display_now(&mut || false).expect("spin");
            let sc = |f: &Option<OrbitalField>| match f {
                Some(OrbitalField::Cpu { signed: true, scale, .. } | OrbitalField::Gpu { signed: true, scale, .. }) => *scale,
                other => panic!("{other:?}"),
            };
            // H₂'s spin density is noise: no field is sent, on either adapter; the summary's
            // display scale (the floor) is what the two are compared by
            assert!(b.field.is_none() && b.spin.no_net_spin(), "{:?} / {:?}", b.field.is_some(), b.spin);
            (sc(&a.field), b.spin.scale as f32)
        };
        let (pc, sc) = scales(&mut c, &oc);
        let (pg, sg) = scales(&mut g, &og);
        println!("max|σ_g|: CPU {pc:.6} GPU {pg:.6}; spin scale CPU {sc:.3e} GPU {sg:.3e}");
        assert!(((pc - pg) / pc).abs() < 1e-3, "orbital scale CPU {pc} GPU {pg}");
        assert!(((sc - sg) / sc).abs() < 1e-3, "spin scale (the floor, 1e-3 max ρ) CPU {sc} GPU {sg}");
    }

    /// Spec §7: zero electrons (H⁺) — no rows, "no electrons — nothing to show", n↑ = n↓ = 0,
    /// and no field for a selected orbital; an empty channel (the H atom's ↓) — the ↑ row only,
    /// "↓: none occupied", ∫m = 1 (± 1e-4). Both adapters.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn empty_scene_and_empty_channel() {
        let ctx = gpu();
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let mut providers: Vec<Box<dyn ForceProvider>> = vec![Box::new(CpuKs::new(1e-6))];
        if let Some(ctx) = ctx {
            providers.push(Box::new(GpuKs::new(ctx, 1e-6)));
        }
        for p in providers.iter_mut() {
            let mut bare = scene(33, vec![Nucleus::new(1, [0.0; 3])]);
            bare.charge = 1;
            p.set_orbital_view(view(OrbitalRequest::Orbital { spin: Spin::Up, track: 0, squared: true }));
            let o = p.evaluate(&bare, &mut never()).unwrap_or_else(|e| panic!("{}: {e}", p.name())).orbitals.expect("a payload");
            println!("{} H⁺: rows {}, notes {:?}, spin {:?}", p.name(), o.rows.len(), o.notes, o.spin);
            assert!(o.rows.is_empty() && o.field.is_none(), "{}", p.name());
            assert!(o.notes.iter().any(|n| n == "no electrons — nothing to show"), "{:?}", o.notes);
            assert_eq!((o.spin.n_up, o.spin.n_dn), (0, 0));
            let h = scene(33, vec![Nucleus::new(1, [0.0; 3])]);
            p.reset();
            p.set_orbital_view(view(OrbitalRequest::SpinDensity));
            let o = p.evaluate(&h, &mut never()).expect("H").orbitals.expect("a payload");
            println!("{} H: rows {:?}, notes {:?}, ∫m = {:.8}", p.name(), o.rows.iter().map(|r| (r.label.to_string(), r.spin)).collect::<Vec<_>>(), o.notes, o.spin.integral);
            assert_eq!(o.rows.len(), 1);
            assert_eq!((o.rows[0].spin, o.rows[0].label.to_string()), (Spin::Up, "1s".to_string()), "a lone atom's 1s");
            assert!(o.notes.iter().any(|n| n == "↓: none occupied"), "{:?}", o.notes);
            assert!((o.spin.integral - 1.0).abs() <= 1e-4, "∫m = {}", o.spin.integral);
            assert!(o.field.is_some(), "the spin density of one electron");
        }
    }

    /// Spec §7, "Orbital count changes": H₂ → an H added far away (the driver resets the provider:
    /// a cold re-solve). The tracks restart, in energy order, after every id handed out before
    /// (none is reused); a selection of a vanished track falls back to the same channel and index
    /// (↓ #1), with the note; removing the H again drops ↑ #2, and a selection of it falls back to
    /// the HOMO.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn count_change_resets_tracks_and_moves_the_selection() {
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let mut g = GpuKs::new(ctx, 1e-6);
        g.set_orbital_view(view(OrbitalRequest::List));
        let two = h2(49, 1.6);
        let a = g.evaluate(&two, &mut never()).expect("H₂").orbitals.expect("payload");
        let old_max = a.rows.iter().map(|r| r.track).max().unwrap();
        let sel = first_track(&a, Spin::Down);
        g.set_orbital_view(view(OrbitalRequest::Orbital { spin: Spin::Down, track: sel, squared: false }));
        let mut three = two.clone();
        three.nuclei.push(Nucleus::new(1, [0.0, 4.5, 0.0]));
        g.reset();
        let b = g.evaluate(&three, &mut never()).expect("H₃").orbitals.expect("payload");
        let tracks: Vec<(Spin, usize, u32)> = b.rows.iter().map(|r| (r.spin, r.index, r.track)).collect();
        println!("H₂ tracks {:?}; after the H was added {:?}; notes {:?}; shown {:?}", a.rows.iter().map(|r| r.track).collect::<Vec<_>>(), tracks, b.notes, b.what);
        assert_eq!(tracks.len(), 3, "↑ 2, ↓ 1");
        assert!(tracks.iter().all(|t| t.2 > old_max), "no id reused after the reset");
        let mut sorted = tracks.clone();
        sorted.sort_by_key(|t| (t.0 == Spin::Down, t.1));
        assert!(sorted.windows(2).all(|w| w[1].2 == w[0].2 + 1), "fresh tracks in energy order (↑ then ↓): {sorted:?}");
        let down = b.rows.iter().find(|r| r.spin == Spin::Down && r.index == 0).unwrap();
        assert_eq!(b.what, OrbitalRequest::Orbital { spin: Spin::Down, track: down.track, squared: false }, "same channel and index");
        assert!(b.notes.iter().any(|n| n.starts_with("selected orbital no longer exists — showing #1 ↓")), "{:?}", b.notes);
        assert!(matches!(b.field, Some(OrbitalField::Gpu { .. })));
        // select ↑ #2, then remove the H: the HOMO
        let up2 = b.rows.iter().find(|r| r.spin == Spin::Up && r.index == 1).unwrap().track;
        g.set_orbital_view(view(OrbitalRequest::Orbital { spin: Spin::Up, track: up2, squared: true }));
        g.reset();
        let c = g.evaluate(&two, &mut never()).expect("H₂ again").orbitals.expect("payload");
        let homo = c.rows.iter().filter(|r| r.occupation > 0.0).fold(None::<&sim::OrbitalInfo>, |b, r| match b {
            Some(b) if b.energy >= r.energy => Some(b),
            _ => Some(r),
        }).unwrap();
        println!("after the H was removed: {:?}; notes {:?}", c.what, c.notes);
        assert_eq!(c.what, OrbitalRequest::Orbital { spin: homo.spin, track: homo.track, squared: true }, "the HOMO");
        assert!(c.notes.iter().any(|n| n.starts_with("selected orbital no longer exists")), "{:?}", c.notes);
    }

    // ---- unoccupied orbitals (orbital-viewer plan Task 4; `crate::orbitals`, "Unoccupied orbitals") ----

    fn with_unoccupied(what: OrbitalRequest, m: u8, force_every_step: bool) -> OrbitalViewRequest {
        OrbitalViewRequest { what, unoccupied: m, force_every_step }
    }

    /// Unoccupied rows of a payload per channel.
    fn unocc_rows(o: &OrbitalDisplay) -> [usize; 2] {
        [Spin::Up, Spin::Down].map(|s| o.rows.iter().filter(|r| r.spin == s && r.occupation == 0.0).count())
    }

    /// Spec §4.4/§4.5 (review SHOULD 13): 20 velocity-Verlet steps of H₂ (49³, from 1.6 Bohr,
    /// dt 0.25 fs, `GpuKs` at tol 1e-6) with the view on, m = 4 unoccupied orbitals and the display
    /// eigensolve **forced at every step** (`force_every_step`: one solve per evaluation), against
    /// the same run with the view never on: energy and every force component bitwise equal at all
    /// 21 points. Exactly 21 eigensolves ran (`solves` 1…21, one per evaluation), each payload's
    /// unoccupied rows current and shown — so the test cannot be vacuous.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn unoccupied_solve_leaves_scf_state_untouched() {
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let start = h2(49, 1.6);
        let integrator = sim::Integrator::new_fs(0.25);
        let run = |on: bool| -> (Vec<(u64, Vec<[u64; 3]>)>, Vec<u64>) {
            let mut p = GpuKs::new(ctx.clone(), 1e-6);
            if on {
                p.set_orbital_view(with_unoccupied(OrbitalRequest::List, 4, true));
            }
            let mut scene = start.clone();
            let first = p.evaluate(&scene, &mut never()).expect("the first solve");
            let bits = |e: f64, f: &[[f64; 3]]| (e.to_bits(), f.iter().map(|v| v.map(f64::to_bits)).collect::<Vec<_>>());
            let mut out = vec![bits(first.energy, &first.forces)];
            let mut solves = Vec::new();
            let mut check = |o: Option<&OrbitalDisplay>| {
                if let Some(o) = o {
                    let u = o.unoccupied.as_ref().expect("m = 4 asked for");
                    assert!(u.current && u.notes.is_empty(), "{u:?}");
                    assert!(unocc_rows(o).iter().all(|&k| k >= 1), "unoccupied rows shown: {:?}", unocc_rows(o));
                    solves.push(u.solves);
                }
            };
            check(first.orbitals.as_ref());
            let mut f = first.forces;
            for _ in 0..20 {
                let (f_new, stats) = integrator.step(&mut scene, &f, &mut p, &mut |_, _| {}, &mut never()).expect("a step");
                check(stats.orbitals.as_ref());
                out.push(bits(stats.energy_ks, &f_new));
                f = f_new;
            }
            (out, solves)
        };
        let (off, s_off) = run(false);
        let t = Instant::now();
        let (on, s_on) = run(true);
        println!("H₂ 49³, 20 steps, m = 4 forced: eigensolves {:?} ({:.1} s with them); E[20] off {:.12} on {:.12}", s_on, t.elapsed().as_secs_f64(),
                 f64::from_bits(off[20].0), f64::from_bits(on[20].0));
        assert!(s_off.is_empty());
        assert_eq!(s_on, (1..=21).collect::<Vec<u64>>(), "one eigensolve per evaluation");
        for (k, (a, b)) in off.iter().zip(&on).enumerate() {
            assert_eq!(a, b, "step {k}: energy or forces differ with the unoccupied eigensolve on");
        }
    }

    /// One evaluation or display seen through [`Recorded`]: the result's `solve_ms` (0 for an
    /// `orbital_display_now`) and the payload's unoccupied status.
    type Seen = (f64, Option<sim::UnoccupiedStatus>);

    /// Wraps a provider and records every result's wall and unoccupied status.
    struct Recorded<P> {
        inner: P,
        seen: Arc<Mutex<Vec<Seen>>>,
    }

    impl<P: ForceProvider> ForceProvider for Recorded<P> {
        fn evaluate(&mut self, scene: &Scene, a: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
            let r = self.inner.evaluate(scene, a)?;
            self.seen.lock().unwrap().push((r.stats.solve_ms as f64, r.orbitals.as_ref().and_then(|o| o.unoccupied.clone())));
            Ok(r)
        }
        fn name(&self) -> &'static str {
            self.inner.name()
        }
        fn reset(&mut self) {
            self.inner.reset()
        }
        fn discontinuity(&mut self) {
            self.inner.discontinuity()
        }
        fn scf_tol(&self) -> Option<f64> {
            self.inner.scf_tol()
        }
        fn set_orbital_view(&mut self, request: OrbitalViewRequest) {
            self.inner.set_orbital_view(request)
        }
        fn orbital_display_now(&mut self, a: &mut dyn FnMut() -> bool) -> Option<OrbitalDisplay> {
            let o = self.inner.orbital_display_now(a);
            self.seen.lock().unwrap().push((0.0, o.as_ref().and_then(|o| o.unoccupied.clone())));
            o
        }
    }

    /// Spec §4.5 (review SHOULD 11), through `sim::Dynamics` on `GpuKs` (tol 1e-5, the display
    /// backends' `INTERACTIVE_TOL`):
    /// - H₂ 49³, m = 4. Switched on while paused: one eigensolve at once. Three paused single
    ///   steps: one each. Then 5 s of play at full speed: at most one per second, and their summed
    ///   wall ≤ `UNOCCUPIED_BUDGET` × the summed SCF wall + one solve (printed: Task 0 found
    ///   the eigensolve 4–50 warm SCFs, so the budget keeps it paused-only — none runs).
    /// - H₂O, m = 4: one paused eigensolve, its wall printed against the SCF's; three play
    ///   evaluations after it run none and the status says paused-only while the budget is short.
    ///   Re-pinned in plan 2026-11-13 Task 9 (was 97³ × 0.25): under FermiDirac the cold all-electron H₂O at 97³ × 0.25
    ///   (a 24 Bohr box) hits the witness cap on a transient step ("no gap witness within 16 extra levels", measured
    ///   2026-11-16 after 630 s; the Task 7 fix-round-1 diagnosis, plan Task 9b owns the cap and the cost), so the leg
    ///   runs on the H₂O preset's own Accurate cell, 95³ × 0.15 (14 Bohr; Task 8 measured it converging cold in 94 s).
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn unoccupied_budget_in_play() {
        use crate::orbitals::{UNOCCUPIED_BUDGET, UNOCCUPIED_MIN_INTERVAL};
        use sim::{DynCommand, Dynamics};
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let seen: Arc<Mutex<Vec<Seen>>> = Arc::default();
        let mut g = GpuKs::new(ctx.clone(), 1e-5);
        g.set_display(false);
        let mut d = Dynamics::new(h2(49, 1.6), Box::new(Recorded { inner: g, seen: seen.clone() }), 0.25, 1);
        let solves = |v: &[Seen]| v.iter().filter_map(|s| s.1.as_ref().map(|u| u.solves)).max().unwrap_or(0);
        d.apply(DynCommand::SetOrbitalView(with_unoccupied(OrbitalRequest::List, 4, false)));
        let first = seen.lock().unwrap().last().cloned().expect("shown at once").1.expect("m = 4");
        println!("H₂ 49³ m = 4, switched on paused: eigensolve {:.2} s, shown {:?}, paused only {}", first.wall_s, first.shown, first.paused_only);
        assert_eq!((first.solves, first.current), (1, true));
        for k in 0..3 {
            d.apply(DynCommand::Step(1));
            let _ = d.tick(f64::INFINITY, &mut never());
            assert_eq!(solves(&seen.lock().unwrap()), 2 + k, "paused single step {k}: one eigensolve");
        }
        let before = seen.lock().unwrap().len();
        d.apply(DynCommand::SetSpeed(sim::dynamics::MAX_SPEED));
        d.apply(DynCommand::Play);
        let t0 = Instant::now();
        let mut steps = 0;
        while t0.elapsed().as_secs_f64() < 5.0 {
            steps += d.tick(16.0, &mut never()).steps_last_tick;
        }
        let wall = t0.elapsed();
        d.apply(DynCommand::Pause);
        let play: Vec<Seen> = seen.lock().unwrap()[before..].to_vec();
        let mut last = 4;
        let (mut eig_s, mut runs, mut scf_s) = (0.0, 0, 0.0);
        for (ms, u) in &play {
            let mut s = ms / 1e3;
            if let Some(u) = u {
                if u.solves > last {
                    assert!(u.current, "a payload with a new eigensolve is this state's");
                    last = u.solves;
                    runs += 1;
                    eig_s += u.wall_s;
                    s -= u.wall_s;
                } else {
                    assert!(!u.current, "play: a payload without a new eigensolve shows an earlier geometry's, stamped not current");
                }
            }
            scf_s += s;
        }
        let paused_only = play.iter().rev().find_map(|s| s.1.as_ref()).map(|u| u.paused_only);
        println!("play {:.2} s: {steps} steps, {} payloads, {runs} eigensolves ({eig_s:.2} s) against {scf_s:.2} s of SCF (budget {UNOCCUPIED_BUDGET}); paused only {paused_only:?}",
                 wall.as_secs_f64(), play.iter().filter(|s| s.1.is_some()).count());
        assert!(runs as f64 <= (wall.as_secs_f64() / UNOCCUPIED_MIN_INTERVAL.as_secs_f64()).ceil(), "{runs} eigensolves in {wall:?}");
        assert!(eig_s <= UNOCCUPIED_BUDGET * scf_s + first.wall_s.max(eig_s / runs.max(1) as f64), "{eig_s} s of eigensolves against {scf_s} s of SCF");
        // H₂O at the preset's Accurate cell (95³ × 0.15; re-pinned from 97³ × 0.25, doc above): the measured cost,
        // paused-only by the budget
        let mut g = GpuKs::new(ctx, 1e-5);
        g.set_display(false);
        g.set_orbital_view(with_unoccupied(OrbitalRequest::List, 4, true));
        let mut s = Scene { cell: Cell { n: 95, h: 0.15 }, ..h2o(95) };
        let r = g.evaluate(&s, &mut never()).expect("H₂O 95³ × 0.15");
        let u = r.orbitals.as_ref().and_then(|o| o.unoccupied.clone()).expect("a paused payload with m = 4");
        let scf = r.stats.solve_ms as f64 / 1e3 - u.wall_s;
        println!("H₂O 95³ × 0.15 m = 4, paused: eigensolve {:.2} s vs the SCF's {scf:.2} s (cold) — a play solve needs {:.0} s of SCF wall first; shown {:?}, notes {:?}",
                 u.wall_s, u.wall_s / UNOCCUPIED_BUDGET, u.shown, u.notes);
        g.set_orbital_view(with_unoccupied(OrbitalRequest::List, 4, false));
        let mut scf_play = 0.0;
        for k in 1..=3 {
            s.nuclei[0].pos[0] += 0.005;
            let r = g.evaluate(&s, &mut never()).expect("a play step");
            scf_play += r.stats.solve_ms as f64 / 1e3;
            if let Some(o) = r.orbitals {
                let v = o.unoccupied.expect("m = 4");
                println!("  play step {k}: solves {}, current {}, paused only {}", v.solves, v.current, v.paused_only);
                if u.wall_s > UNOCCUPIED_BUDGET * scf_play {
                    assert_eq!((v.solves, v.current, v.paused_only), (1, false, true), "over budget: no solve in play");
                }
            }
        }
    }

    /// Task 3 review ruling 4, built in Task 4: the orbital view is charged to the GPU memory
    /// budget when a payload is made — not to the scene, which would refuse scenes that fit
    /// without it. The footprint grows with m by the kept vectors, the solve's copies and the
    /// `LobpcgWork` growth (field counts pinned). On the GPU (H₂ 49³, m = 4): a budget with room
    /// for the view shows the unoccupied rows; one with room only without them shows the occupied
    /// rows and says why; one without room even for that gives a note-only payload.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn the_orbital_view_is_charged_to_the_vram_budget() {
        let s = h2(49, 1.4);
        // + the all-electron FD witness vectors at the count (plan 2026-11-13 Task 7 fix rounds 1–2; 3 since Task 9c fix round 2)
        assert_eq!(solve_footprint_bytes(49, 2, 1) + witness_extra_bytes(49), gpu_footprint_bytes(&s), "one formula");
        let fb = field_bytes(49) as f64;
        let fields = |m: usize| (orbital_view_footprint_bytes(49, 1, 1, m, TextureKind::None) as f64 / (1.2 * fb)).round() as usize;
        // m = 0: K = 2 shown + scratch + row buffer; m = 4: + 2·6 shown + 2·6 kept + 2·6 new (the
        // previous ones stay intact for an abort) + k_max 1 + 5 potentials + 8·6 work
        assert_eq!((fields(0), fields(4)), (4, 4 + 12 + 12 + 12 + 1 + 5 + 48));
        let gb = |b: u64| b as f64 / 1e9;
        for n in [49usize, 57, 81, 97] {
            println!("orbital view VRAM at {n}³ (K = 10): m = 0 list {:.3} GB, m = 4 list {:.3} GB, m = 4 colour {:.3} GB (a solve: {:.3} GB)",
                     gb(orbital_view_footprint_bytes(n, 5, 5, 0, TextureKind::None)), gb(orbital_view_footprint_bytes(n, 5, 5, 4, TextureKind::None)),
                     gb(orbital_view_footprint_bytes(n, 5, 5, 4, TextureKind::Colour)), gb(solve_footprint_bytes(n, 10, 5)));
        }
        println!("orbital view at 97³ (H₂O, K = 10): m = 0 list {:.3} GB, m = 4 list {:.3} GB, m = 4 colour {:.3} GB; the H₂O solve {:.3} GB",
                 gb(orbital_view_footprint_bytes(97, 5, 5, 0, TextureKind::None)), gb(orbital_view_footprint_bytes(97, 5, 5, 4, TextureKind::None)),
                 gb(orbital_view_footprint_bytes(97, 5, 5, 4, TextureKind::Colour)), gb(gpu_footprint_bytes(&h2o(97))));
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let base = gpu_footprint_bytes(&s);
        let (v0, v4) = (orbital_view_footprint_bytes(49, 1, 1, 0, TextureKind::None), orbital_view_footprint_bytes(49, 1, 1, 4, TextureKind::None));
        let payload = |budget: u64| {
            let mut g = GpuKs::new(ctx.clone(), 1e-6).with_vram_budget(budget);
            g.set_orbital_view(with_unoccupied(OrbitalRequest::List, 4, true));
            let r = g.evaluate(&s, &mut never()).expect("the scene itself fits");
            (r.orbitals.expect("a payload"), g.orbital.is_some())
        };
        let (o, built) = payload(base + v4);
        assert!(built && unocc_rows(&o).iter().all(|&k| k >= 1), "room for the view: unoccupied rows {:?}", unocc_rows(&o));
        let (o, built) = payload(base + (v0 + v4) / 2);
        let u = o.unoccupied.as_ref().expect("m = 4 asked for");
        println!("room without the unoccupied orbitals: rows {}, notes {:?}", o.rows.len(), u.notes);
        assert!(built && o.rows.len() == 2 && unocc_rows(&o) == [0, 0], "the occupied rows only");
        assert!(u.notes.iter().any(|n| n.contains("4 unoccupied orbitals refused") && n.contains("budget")), "{:?}", u.notes);
        let (o, built) = payload(base + v0 / 2);
        println!("no room for the view: {:?}", o.notes);
        assert!(!built && o.rows.is_empty() && o.field.is_none(), "refused: no producer, a note");
        assert_eq!(o.unoccupied.as_ref().map(|u| (u.requested, u.shown)), Some((4, [0, 0])), "it still answers the request (m = 4)");
        assert!(o.notes.iter().any(|n| n.contains("orbital view needs") && n.contains("budget")), "{:?}", o.notes);
    }

    /// The CPU producer's unoccupied rows (plan Task 4; `qm::orbitals::unoccupied_cpu`): H₂ 33³,
    /// tol 1e-6, m = 2, paused. The payload has the occupied rows (no residual) and unoccupied rows
    /// with occupation 0, their residual and the not-converged flag consistent with the tolerance
    /// (eig_tol 1e-6 on the CPU), current; a re-display of the same state (another mode) reuses
    /// the vectors without a new eigensolve.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn cpu_unoccupied_rows_reach_the_payload() {
        let mut c = CpuKs::new(1e-6);
        c.set_orbital_view(with_unoccupied(OrbitalRequest::List, 2, true));
        let o = c.evaluate(&h2(33, 1.4), &mut never()).expect("CpuKs").orbitals.expect("a payload");
        let u = o.unoccupied.clone().expect("m = 2 asked for");
        for r in &o.rows {
            println!("{} {:?}: ε {:+.6} occ {} residual {:?}{}{}", r.label, r.spin, r.energy, r.occupation, r.residual,
                     if r.not_converged { " NOT converged" } else { "" }, if r.box_state { " box" } else { "" });
        }
        println!("status {u:?}");
        assert_eq!((u.requested, u.tol, u.solves, u.current), (2, 1e-6, 1, true));
        assert!(u.wall_s > 0.0 && u.shown.iter().all(|&k| k >= 1), "{u:?}");
        let homo = o.rows.iter().filter(|r| r.occupation > 0.0).map(|r| r.energy).fold(f64::NEG_INFINITY, f64::max);
        for r in &o.rows {
            if r.occupation > 0.0 {
                assert!(r.residual.is_none() && !r.not_converged && !r.box_state, "occupied rows carry no eigensolve data");
            } else {
                let res = r.residual.expect("an unoccupied row's residual");
                assert_eq!(r.not_converged, res > u.tol, "the tag is the residual against the tolerance");
                assert!(r.energy > homo, "above the HOMO");
            }
        }
        c.set_orbital_view(with_unoccupied(OrbitalRequest::SpinDensity, 2, true));
        let again = c.orbital_display_now(&mut || false).expect("a re-display").unoccupied.expect("m = 2");
        assert_eq!((again.solves, again.current), (1, true), "the same state: no new eigensolve");
    }

    // ---- orbital shares (orbital-atom-shares plan Task 2) ----

    /// Share tables for species `z`: one H-like free atom e^{−2r}/π shared by every nucleus (an
    /// analytic stand-in for the solved reference — a mirror-symmetric molecule's split is even for
    /// any one table, and the sum rule holds for any weights).
    fn h_like_tables(z: &[u8]) -> Arc<sim::ShareTables> {
        let t = Arc::new(sim::RadialDensity::from_fn(|r: f64| (-2.0 * r).exp() / std::f64::consts::PI, 12.0, 0.05));
        Arc::new(sim::ShareTables { tables: z.iter().map(|_| t.clone()).collect(), z: z.to_vec() })
    }

    /// Plan Task 2: `CpuKs` H₂ (33³, R = 1.4) with tables for both H → σ_g (↑ and ↓) splits
    /// 0.5/0.5 within 1e-6 and its shares sum to 1 within 1e-9 (h³ Σ ψ̃² = 1); no note added. A
    /// re-display without a solve carries the same shares.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn cpu_payload_carries_shares() {
        let mut c = CpuKs::new(1e-6);
        c.set_orbital_view(view(OrbitalRequest::List));
        c.set_share_tables(Some(h_like_tables(&[1, 1])));
        let o = c.evaluate(&h2(33, 1.4), &mut never()).expect("CpuKs").orbitals.expect("a payload");
        assert_eq!(o.rows.len(), 2, "σ_g ↑ and ↓");
        for r in &o.rows {
            let s = r.shares.as_ref().expect("shares on every row");
            let sum = sim::share_sum(s);
            println!("{} {:?}: shares {s:?}, Σ − 1 = {:+.2e}, s₀ − s₁ = {:+.2e}", r.label, r.spin, sum - 1.0, s[0] - s[1]);
            assert_eq!(s.len(), 2);
            assert!((sum - 1.0).abs() <= 1e-9, "Σ = {sum}");
            assert!(s.iter().all(|x| (x - 0.5).abs() <= 1e-6), "{s:?}");
        }
        assert!(!o.notes.iter().any(|n| n.contains("orbital shares")), "{:?}", o.notes);
        let again = c.orbital_display_now(&mut || false).expect("a re-display");
        assert_eq!(again.rows.iter().map(|r| r.shares.clone()).collect::<Vec<_>>(), o.rows.iter().map(|r| r.shares.clone()).collect::<Vec<_>>());
        // the tables withdrawn: as before them
        c.set_share_tables(None);
        let none = c.orbital_display_now(&mut || false).expect("a re-display");
        assert!(none.rows.iter().all(|r| r.shares.is_none()));
    }

    /// Plan Task 2 / Review Focus 1: tables for [1, 1] and a scene changed to three H before the
    /// payload → every row's `shares` is `None` (never shares indexed to the wrong atoms) and the
    /// payload carries no note about it; the tables for the new species list give shares again.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn stale_tables_give_no_shares() {
        let mut c = CpuKs::new(1e-6);
        c.set_orbital_view(view(OrbitalRequest::List));
        c.set_share_tables(Some(h_like_tables(&[1, 1])));
        let h3 = scene(33, vec![Nucleus::new(1, [0.7, 0.0, 0.0]), Nucleus::new(1, [-0.7, 0.0, 0.0]), Nucleus::new(1, [0.0, 1.6, 0.0])]);
        let o = c.evaluate(&h3, &mut never()).expect("CpuKs").orbitals.expect("a payload");
        assert!(!o.rows.is_empty());
        assert!(o.rows.iter().all(|r| r.shares.is_none()), "{:?}", o.rows.iter().map(|r| &r.shares).collect::<Vec<_>>());
        assert!(!o.notes.iter().any(|n| n.contains("orbital shares")), "{:?}", o.notes);
        c.set_share_tables(Some(h_like_tables(&[1, 1, 1])));
        let o = c.orbital_display_now(&mut || false).expect("a re-display");
        for r in &o.rows {
            let s = r.shares.as_ref().expect("the new tables fit");
            println!("H₃ {} {:?}: {s:?}", r.label, r.spin);
            assert!(s.len() == 3 && (sim::share_sum(s) - 1.0).abs() <= 1e-9, "{s:?}");
        }
    }

    /// Fix round 1 (M3): `share_eligible` through `CpuKs` with unoccupied rows. H₂⁺ (49³, R = 2.0;
    /// a cation's potential keeps bound virtuals, unlike LSDA H₂'s box states). The rows are checked
    /// in two states:
    /// - **Paused** (forced: the eigensolve runs, the rows are current): every non-box row,
    ///   occupied and unoccupied, carries shares with Σ = 1, and a box state carries none.
    /// - **In play, after a move** (the eigensolve's play budget keeps the held vectors: stamped
    ///   not current): the unoccupied rows carry none, the occupied rows keep theirs.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn cpu_unoccupied_rows_get_shares_only_when_current() {
        let mut c = CpuKs::new(1e-6);
        c.set_orbital_view(with_unoccupied(OrbitalRequest::List, 2, true));
        c.set_share_tables(Some(h_like_tables(&[1, 1])));
        let mut s = h2(49, 2.0);
        s.charge = 1;
        let o = c.evaluate(&s, &mut never()).expect("CpuKs").orbitals.expect("a payload");
        let u = o.unoccupied.clone().expect("m = 2 asked for");
        assert!(u.current, "paused: the eigensolve ran at this geometry");
        let mut unocc_with_shares = 0;
        for r in &o.rows {
            println!("current: {} {:?} ε {:+.4} occ {} box {} shares {:?}", r.label, r.spin, r.energy, r.occupation, r.box_state, r.shares);
            if r.box_state {
                assert!(r.shares.is_none(), "a box state gets no share");
            } else {
                let sh = r.shares.as_ref().expect("a current row's shares");
                assert!((sim::share_sum(sh) - 1.0).abs() <= 1e-9, "{sh:?}");
                if r.residual.is_some() {
                    unocc_with_shares += 1;
                }
            }
        }
        assert!(unocc_with_shares > 0, "the test needs a bound (non-box) unoccupied row");
        assert_eq!(o.share_z, Some(vec![1, 1]));
        // play: a moved geometry, the held unoccupied vectors stamped not current
        c.set_orbital_view(with_unoccupied(OrbitalRequest::List, 2, false));
        s.nuclei[0].pos[0] += 0.02;
        c.evaluate(&s, &mut never()).expect("a play step");
        let o = c.orbital_display_now(&mut || false).expect("a re-display");
        let u = o.unoccupied.clone().expect("m = 2");
        assert!(!u.current, "the play budget kept the earlier geometry's vectors: {u:?}");
        for r in &o.rows {
            println!("stale: {} {:?} occ {} residual {:?} shares {:?}", r.label, r.spin, r.occupation, r.residual, r.shares);
            if r.residual.is_some() {
                assert!(r.shares.is_none(), "an unoccupied row of an earlier geometry gets no share");
            } else {
                assert!(r.shares.as_ref().is_some_and(|sh| (sim::share_sum(sh) - 1.0).abs() <= 1e-9), "an occupied row keeps its shares");
            }
        }
    }

    // ---- orbital shares on the GPU (orbital-atom-shares plan Task 3) ----

    /// Share tables with nucleus a's own profile `profiles[a]` (one `Arc` each, so the packed
    /// buffer holds one table per nucleus), tabulated to `r_max` at dr = 0.05. `cut`: one more
    /// entry that rises, so the table is 0 beyond r_max + 0.05 (`RadialDensity::at`) and the nodes
    /// far from every nucleus go whole to the nearest one (the underflow rule).
    fn tables_with(z: &[u8], profiles: &[fn(f64) -> f64], r_max: f64, cut: bool) -> Arc<sim::ShareTables> {
        let tables = profiles.iter().map(|&f| {
            let mut t = sim::RadialDensity::from_fn(f, r_max, 0.05);
            if cut {
                let (r, l) = (*t.r.last().expect("entries"), *t.log_rho.last().expect("entries"));
                t.r.push(r + 0.05);
                t.log_rho.push(l + 1.0);
            }
            Arc::new(t)
        }).collect();
        Arc::new(sim::ShareTables { tables, z: z.to_vec() })
    }

    fn h_like(r: f64) -> f64 { (-2.0 * r).exp() / std::f64::consts::PI }
    fn he_like(r: f64) -> f64 { 3.07 * (-3.375 * r).exp() }
    fn broad(r: f64) -> f64 { 0.3 * (-1.2 * r).exp() }
    /// The bond detector's water fixture profiles (Gaussian tails: below f32's range inside a 95³
    /// cell, and below f64's beyond the table — the log storage and the sentinel both used).
    fn o_gauss(r: f64) -> f64 { 6.0 * (-r * r / 0.18).exp() + 0.5 * (-r * r / 1.62).exp() }
    fn h_gauss(r: f64) -> f64 { 0.3 * (-r * r / 0.5).exp() }

    /// The GPU pass (`OrbitalView::shares`, raw: before the check and the set average) on the ψ̃
    /// `g`'s view displayed last, and the CPU f64 reference (`sim::orbital_shares_many` with
    /// `Weights::new`'s weights) on the same fields read back (f32 → f64): (channel, row, cpu, gpu).
    fn shares_on_the_same_fields(g: &mut GpuKs, tables: &Arc<sim::ShareTables>, s: &Scene) -> Vec<(usize, usize, Vec<f64>, Vec<f64>)> {
        let positions: Vec<[f64; 3]> = s.nuclei.iter().map(|n| n.pos).collect();
        let GpuKs { solver, orbital, share_gpu, ctx, .. } = g;
        let (solver, view) = (solver.as_ref().expect("solved"), orbital.as_mut().expect("a view"));
        let on_device = GpuKs::share_tables_on_device(share_gpu, ctx, tables, &s.cell, &positions);
        let gpu = view.shares(solver, on_device, &positions);
        let free: Vec<&sim::RadialDensity> = tables.tables.iter().map(|a| a.as_ref()).collect();
        let w = sim::Weights::new(&free, &positions, &s.cell);
        let qg = qm_gpu::grid::GpuGrid::new(solver.grid());
        let mut out = Vec::new();
        for (ch, rows) in gpu.iter().enumerate() {
            for (i, sg) in rows.iter().enumerate() {
                let psi = view.shown_field(ch, i).expect("a shown row").read_interior(ctx, &qg);
                let cpu = sim::orbital_shares_many(&w, &s.cell, &[&psi]).remove(0);
                out.push((ch, i, cpu, sg.clone()));
            }
        }
        out
    }

    fn worst(pairs: impl Iterator<Item = (f64, f64)>) -> f64 {
        pairs.map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)
    }

    /// Plan Task 3: the GPU share kernel against the CPU reference, at 49³ × 0.25, two ways:
    /// - **the same ψ̃**: the raw GPU shares of the fields the view displayed vs
    ///   `sim::orbital_shares_many` on those fields read back — the kernel alone (weights, lookup,
    ///   underflow rule, reduction);
    /// - **the same state on both providers**: `GpuKs`'s payload shares vs `CpuKs`'s, rows matched
    ///   by (spin, index) — the kernel plus the f32 solve's difference from the f64 one.
    ///
    /// The same-ψ̃ comparison to |Δs| ≤ 5e-6 per row and nucleus (Task 3 review Minor 1: the
    /// derived bound, ≈ 1.3e-6 from the f32 sums plus ≤ 0.25·5e-6 from the f32 log lookup;
    /// measured ≤ 1.2e-8), the providers' to |Δs| ≤ 1e-4 (it includes the f32 solve), and Σ_A
    /// s_gpu within 1e-4 of 1. Cases: H₂ with one shared table (one packed table, two nuclei),
    /// with two different tables (nucleus 0's H-like, nucleus 1's broader: an uneven split), with
    /// both tables cut at 2 Bohr (most nodes go to the nearest nucleus by the underflow rule; the
    /// mid-plane nodes x = 0 are exact ties, which both sides give to nucleus 0), and HeH⁺
    /// (He-like and H-like tables). Then HeH⁺ with two unoccupied orbitals (Task 3 review Minor
    /// 2): the display eigensolve's vectors are rows n_σ.. of the view's shown fields, and the
    /// payload's unoccupied rows must carry the shares of exactly those fields — the same-ψ̃
    /// comparison over every shown row, each payload row of its own (no set average) holding the
    /// raw GPU shares of its (channel, index) bit for bit, and a box state, if the payload has any, holding none. A cation,
    /// because its lowest unoccupied orbitals are bound: H₂'s at 49³ are all box states (measured),
    /// which carry no shares, so they cannot show the mapping. That case has no provider
    /// comparison: the two providers' unoccupied vectors are separate eigensolves.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn gpu_shares_match_cpu() {
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let mut heh = scene(49, vec![Nucleus::new(2, [0.73, 0.0, 0.0]), Nucleus::new(1, [-0.73, 0.0, 0.0])]);
        heh.charge = 1;
        let cases: Vec<(&str, Scene, Arc<sim::ShareTables>, u8)> = vec![
            ("H₂, one shared H-like table", h2(49, 1.4), h_like_tables(&[1, 1]), 0),
            ("H₂, H-like and broad tables", h2(49, 1.4), tables_with(&[1, 1], &[h_like, broad], 12.0, false), 0),
            ("H₂, tables cut at 2 Bohr", h2(49, 1.4), tables_with(&[1, 1], &[h_like, h_like], 2.0, true), 0),
            ("HeH⁺, He-like and H-like tables", heh.clone(), tables_with(&[2, 1], &[he_like, h_like], 12.0, false), 0),
            ("HeH⁺, He-like and H-like tables, 2 unoccupied", heh.clone(), tables_with(&[2, 1], &[he_like, h_like], 12.0, false), 2),
        ];
        for (name, s, tables, m) in cases {
            let mut g = GpuKs::new(ctx.clone(), 1e-6);
            g.set_orbital_view(with_unoccupied(OrbitalRequest::List, m, true));
            g.set_share_tables(Some(tables.clone()));
            let pg = g.evaluate(&s, &mut never()).expect("GpuKs").orbitals.expect("a GPU payload");
            // the kernel alone, on the same fields
            let same = shares_on_the_same_fields(&mut g, &tables, &s);
            assert!(!same.is_empty());
            let d_same = worst(same.iter().flat_map(|(_, _, c, q)| c.iter().copied().zip(q.iter().copied())));
            let sum_same = worst(same.iter().map(|(_, _, _, q)| (sim::share_sum(q), 1.0)));
            assert!(d_same <= 5e-6 && sum_same <= 1e-4, "{name}: the kernel on the same ψ̃: {d_same:.2e}, Σ {sum_same:.2e}");
            if m > 0 {
                assert_eq!(pg.share_z, Some(s.nuclei.iter().map(|n| n.z).collect::<Vec<_>>()), "{name}: {:?}", pg.notes);
                let unocc: Vec<&sim::OrbitalInfo> = pg.rows.iter().filter(|r| r.occupation == 0.0).collect();
                assert!(!unocc.is_empty(), "{name}: unoccupied rows shown");
                for (ch, spin) in [(0, Spin::Up), (1, Spin::Down)] {
                    assert_eq!(same.iter().filter(|e| e.0 == ch).count(), pg.rows.iter().filter(|r| r.spin == spin).count(),
                               "{name} {spin:?}: one shown field per payload row");
                }
                let mut checked = 0;
                for r in &pg.rows {
                    let ch = if r.spin == Spin::Up { 0 } else { 1 };
                    let raw = same.iter().find(|e| e.0 == ch && e.1 == r.index).unwrap_or_else(|| panic!("{name}: no shown field for {} {:?} #{}", r.label, r.spin, r.index));
                    println!("{name}: {} {:?} #{} occ {} box {} set {:?}: payload {:.7?}, raw GPU {:.7?}, CPU on the same ψ̃ {:.7?}",
                             r.label, r.spin, r.index, r.occupation, r.box_state, r.cluster.map(|c| c.0), r.shares, raw.3, raw.2);
                    assert!(!(r.box_state && r.shares.is_some()), "{name}: the box state {} {:?} #{} carries shares", r.label, r.spin, r.index);
                    if let (Some(sh), true) = (&r.shares, r.cluster.is_none_or(|c| c.0 == 1)) {
                        assert_eq!(sh, &raw.3, "{name}: {} {:?} #{} carries another field's shares", r.label, r.spin, r.index);
                        checked += usize::from(r.occupation == 0.0);
                    }
                }
                assert!(checked > 0, "{name}: no unoccupied row of its own with shares to check the mapping on");
                println!("{name}: same ψ̃ max|Δs| = {d_same:.2e}, max|Σ−1| = {sum_same:.2e} over {} shown rows; {checked} unoccupied rows carry their own field's shares", same.len());
                continue;
            }
            // the providers, on the same scene
            let mut c = CpuKs::new(1e-6);
            c.set_orbital_view(view(OrbitalRequest::List));
            c.set_share_tables(Some(tables.clone()));
            let pc = c.evaluate(&s, &mut never()).expect("CpuKs").orbitals.expect("a CPU payload");
            assert_eq!(pg.share_z, Some(s.nuclei.iter().map(|n| n.z).collect::<Vec<_>>()), "{name}: {:?}", pg.notes);
            let mut d_prov = 0.0f64;
            let mut sum_prov = 0.0f64;
            for r in &pg.rows {
                let sg = r.shares.as_ref().unwrap_or_else(|| panic!("{name}: a GPU row without shares: {} {:?}", r.label, r.spin));
                let rc = pc.rows.iter().find(|q| q.spin == r.spin && q.index == r.index).expect("the same row on the CPU");
                let sc = rc.shares.as_ref().expect("CPU shares");
                println!("{name}: {} {:?} GPU {sg:.7?} CPU {sc:.7?}", r.label, r.spin);
                d_prov = d_prov.max(worst(sg.iter().copied().zip(sc.iter().copied())));
                sum_prov = sum_prov.max((sim::share_sum(sg) - 1.0).abs());
            }
            println!("{name}: same ψ̃ max|Δs| = {d_same:.2e}, max|Σ−1| = {sum_same:.2e}; GpuKs vs CpuKs max|Δs| = {d_prov:.2e}, max|Σ−1| = {sum_prov:.2e}");
            assert!(d_prov <= 1e-4 && sum_prov <= 1e-4, "{name}: GpuKs vs CpuKs: {d_prov:.2e}, Σ {sum_prov:.2e}");
        }
    }

    /// Plan Task 3 (report-only, asserted loosely): the GPU share pass at H₂O 95³ × 0.15 (the
    /// app's Accurate spacing; 5 ↑ + 5 ↓ rows × 3 nuclei, Gaussian tables) — the pass's wall time
    /// (dispatches, one submission, one map), min and median of 20, and the payload's wall time
    /// with and without tables (min of 10 each: the time the shares add to a payload, which is
    /// built inside `evaluate` when a refresh is due, after the energy and forces). Also the same-ψ̃
    /// agreement at this size, to the derived 5e-6 of `gpu_shares_match_cpu` (measured 2.3e-8).
    /// Asserts the pass's minimum ≤ 50 ms.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn gpu_shares_cost_is_reported() {
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let mut s = h2o(95);
        s.cell = Cell { n: 95, h: 0.15 };
        let (o, h) = (Arc::new(sim::RadialDensity::from_fn(o_gauss, 12.0, 0.05)), Arc::new(sim::RadialDensity::from_fn(h_gauss, 12.0, 0.05)));
        let tables = Arc::new(sim::ShareTables { tables: vec![o, h.clone(), h], z: vec![8, 1, 1] });
        let mut g = GpuKs::new(ctx, 1e-6);
        g.set_orbital_view(view(OrbitalRequest::List));
        g.set_share_tables(Some(tables.clone()));
        let t = Instant::now();
        let p = g.evaluate(&s, &mut never()).expect("GpuKs H₂O 95³").orbitals.expect("a payload");
        println!("H₂O 95³ × 0.15: solve + payload {:.1} s", t.elapsed().as_secs_f64());
        for r in &p.rows {
            println!("  {} {:?}: shares {:.4?}", r.label, r.spin, r.shares);
        }
        assert_eq!(p.share_z, Some(vec![8, 1, 1]), "{:?}", p.notes);
        let same = shares_on_the_same_fields(&mut g, &tables, &s);
        let d_same = worst(same.iter().flat_map(|(_, _, c, q)| c.iter().copied().zip(q.iter().copied())));
        let sum_same = worst(same.iter().map(|(_, _, _, q)| (sim::share_sum(q), 1.0)));
        // the pass alone
        let positions: Vec<[f64; 3]> = s.nuclei.iter().map(|n| n.pos).collect();
        let mut pass_ms: Vec<f64> = Vec::new();
        {
            let GpuKs { solver, orbital, share_gpu, ctx, .. } = &mut g;
            let (solver, view) = (solver.as_ref().expect("solved"), orbital.as_mut().expect("a view"));
            let on_device = GpuKs::share_tables_on_device(share_gpu, ctx, &tables, &s.cell, &positions);
            for _ in 0..20 {
                let t = Instant::now();
                let sh = view.shares(solver, on_device, &positions);
                pass_ms.push(t.elapsed().as_secs_f64() * 1e3);
                assert_eq!(sh[0].len() + sh[1].len(), same.len());
            }
        }
        pass_ms.sort_by(f64::total_cmp);
        // the payload with and without the tables
        let payload_ms = |g: &mut GpuKs| -> f64 {
            (0..10).map(|_| {
                let t = Instant::now();
                g.orbital_display_now(&mut never()).expect("a re-display");
                t.elapsed().as_secs_f64() * 1e3
            }).fold(f64::INFINITY, f64::min)
        };
        let with = payload_ms(&mut g);
        g.set_share_tables(None);
        let without = payload_ms(&mut g);
        println!("H₂O 95³: {} rows × 3 nuclei: share pass min {:.2} ms, median {:.2} ms, max {:.2} ms; payload min {with:.2} ms with tables, {without:.2} ms without ({:+.2} ms); same ψ̃ max|Δs| = {d_same:.2e}, max|Σ−1| = {sum_same:.2e}",
                 same.len(), pass_ms[0], pass_ms[pass_ms.len() / 2], pass_ms[pass_ms.len() - 1], with - without);
        assert!(d_same <= 5e-6 && sum_same <= 1e-4, "the kernel on the same ψ̃ at 95³: {d_same:.2e}, Σ {sum_same:.2e}");
        assert!(pass_ms[0] <= 50.0, "share pass {:.2} ms > 50 ms", pass_ms[0]);
    }

    /// Controller ruling 2: the packed tables are uploaded when the `Arc<ShareTables>` changes,
    /// not per payload — a second payload reuses the buffer, a new `Arc` with the same content is
    /// adopted without an upload, other content uploads anew. A `GpuKs` without tables behaves as
    /// before: no row carries shares, no `share_z`, no note, nothing packed.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn gpu_share_tables_upload_once_per_change() {
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let mut g = GpuKs::new(ctx, 1e-6);
        g.set_orbital_view(view(OrbitalRequest::List));
        let s = h2(33, 1.4);
        let bare = g.evaluate(&s, &mut never()).expect("GpuKs").orbitals.expect("a payload");
        assert!(bare.rows.iter().all(|r| r.shares.is_none()) && bare.share_z.is_none() && g.share_gpu.is_none());
        assert!(!bare.notes.iter().any(|n| n.contains("orbital shares")), "{:?}", bare.notes);
        let buffer = |g: &GpuKs| g.share_gpu.as_ref().expect("packed").2.buffer().clone();
        g.set_share_tables(Some(h_like_tables(&[1, 1])));
        let first = g.orbital_display_now(&mut never()).expect("a payload");
        assert!(first.rows.iter().all(|r| r.shares.as_ref().is_some_and(|s| (s[0] - 0.5).abs() <= 1e-4)), "{:?}", first.rows);
        let b0 = buffer(&g);
        g.orbital_display_now(&mut never()).expect("a payload");
        assert!(buffer(&g) == b0, "a second payload reuses the packed tables");
        g.set_share_tables(Some(h_like_tables(&[1, 1]))); // a new Arc, the same content
        g.orbital_display_now(&mut never()).expect("a payload");
        assert!(buffer(&g) == b0, "the same content is not uploaded again");
        g.set_share_tables(Some(tables_with(&[1, 1], &[h_like, broad], 12.0, false)));
        let other = g.orbital_display_now(&mut never()).expect("a payload");
        assert!(buffer(&g) != b0, "other content is uploaded");
        assert!(other.rows.iter().all(|r| r.shares.as_ref().is_some_and(|s| (s[0] - 0.5).abs() > 0.01)), "the new tables split unevenly: {:?}", other.rows);
    }

    // ---- pseudopotentials (M7 Task 9) ----

    /// A lone O at the origin (triplet) in a cell of 6 Bohr half-width at spacing h.
    fn o_scene(h: f64, policy: sim::IonicPolicy) -> Scene {
        let n = 2 * (6.0 / h).round() as usize + 1;
        Scene { nuclei: vec![Nucleus::new(8, [0.0; 3])], cell: Cell { n, h }, charge: 0, multiplicity: Some(3), sigma: None, external_field: [0.0; 3], ionic: policy, train_mask: None }
    }

    /// Spec §8 row 9 (§5.5, §4.8 T8): a scene coarser than a species' h_req is refused with
    /// ProviderError::Invalid naming h and h_req — before any solve.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only (generates O); cargo test -p providers --release")]
    fn scene_with_coarser_h_than_h_req_is_invalid() {
        let h_req = qm::radial::generate(8).unwrap().record.h_req.expect("O has h_req");
        for policy in [sim::IonicPolicy::CoresOnly, sim::IonicPolicy::All] {
            let e = CpuKs::new(1e-6).evaluate(&o_scene(h_req + 0.05, policy), &mut || false).unwrap_err();
            assert!(matches!(&e, ProviderError::Invalid(m) if m.contains("coarser") && m.contains("h_req")), "{e:?}");
        }
        // the same scene under None is posed (today's all-electron solver; not solved here)
        assert!(super::problem(&o_scene(h_req + 0.05, sim::IonicPolicy::None), 1e-6, &mut Vec::new()).is_ok());
    }

    /// Review Focus 1: charge and multiplicity are checked against the valence count under a pseudo
    /// policy (O: 6 valence electrons), never panicking in System::with_ionic.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only (generates O); cargo test -p providers --release")]
    fn charge_and_multiplicity_are_checked_against_valence_electrons() {
        let h = qm::radial::generate(8).unwrap().record.h_req.unwrap();
        let mut s = o_scene(h, sim::IonicPolicy::CoresOnly);
        s.charge = 7;
        s.multiplicity = None;
        let e = super::problem(&s, 1e-6, &mut Vec::new()).err().expect("refused");
        assert!(matches!(&e, ProviderError::Invalid(m) if m.contains("charge 7") && m.contains("6 valence electrons")), "{e:?}");
        let mut s = o_scene(h, sim::IonicPolicy::CoresOnly);
        s.multiplicity = Some(9);
        let e = super::problem(&s, 1e-6, &mut Vec::new()).err().expect("refused");
        assert!(matches!(&e, ProviderError::Invalid(m) if m.contains("multiplicity 9") && m.contains("6 electrons")), "{e:?}");
        // under None the same charge is posed (8 − 7 = 1 electron)
        let mut s = o_scene(h, sim::IonicPolicy::None);
        s.charge = 7;
        s.multiplicity = None;
        assert!(super::problem(&s, 1e-6, &mut Vec::new()).is_ok());
    }

    /// Review Focus 5: Li (two valence s states) falls back to all-electron under CoresOnly with a note,
    /// and is Invalid under All with the reason and the ONCV stage named.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only (radial Li atom); cargo test -p providers --release")]
    fn refused_species_falls_back_under_cores_only_and_is_invalid_under_all() {
        let li = |policy| Scene { nuclei: vec![Nucleus::new(3, [0.0; 3])], cell: Cell { n: 33, h: 0.25 }, charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3], ionic: policy, train_mask: None };
        let mut notes = Vec::new();
        let (_, sys, settings) = super::problem(&li(sim::IonicPolicy::CoresOnly), 1e-6, &mut notes).expect("posed");
        assert!(sys.ionic.is_none(), "nothing pseudised: Li runs on the all-electron path");
        // re-pinned in plan 2026-11-13 Task 9 (was Aufbau): every scene is FD; the all-electron tag moved with it
        assert_eq!(settings.occupations, qm::Occupations::FermiDirac { kt: qm::ELECTRON_KT }, "every scene's occupations are Fermi–Dirac");
        assert!(crate::pseudo::is_all_electron_tag(&crate::pseudo::provenance_tag(&li(sim::IonicPolicy::CoresOnly))), "the all-electron surface");
        assert!(notes.iter().any(|n| n.contains("Li") && n.contains("two valence s states") && n.contains("all-electron")), "{notes:?}");
        let e = super::problem(&li(sim::IonicPolicy::All), 1e-6, &mut Vec::new()).err().expect("refused");
        assert!(matches!(&e, ProviderError::Invalid(m) if m.contains("refused") && m.contains("ONCV")), "{e:?}");
    }

    /// The warning prefix `sim` carries is `qm`'s (the driver and the surrogate match on it).
    #[test]
    fn core_overlap_warning_text_is_qms() {
        assert_eq!(sim::CORE_OVERLAP_WARNING, qm::ionic::CORE_OVERLAP_WARNING);
    }

    /// Task 8a SHOULD-5 / Task 8b ruling (c): the memory guard counts what a pseudo solve adds — the valence
    /// electron count, rho_pc and the two XC inputs, the witness growth, the force scratch (6 fields), the
    /// weights and the partials — and a `None` scene is counted exactly as before. It never generates: asked of a
    /// cache without O's table it counts the bound. Everything runs on private caches, so no other test thread can
    /// warm them, generate into them or hold their lock, and every assertion always runs.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only (generates O); cargo test -p providers --release")]
    fn the_memory_guard_counts_the_pseudo_path_and_leaves_none_unchanged() {
        use crate::pseudo::{footprint_in, PseudoCache};
        let h = 0.20;
        let none = o_scene(h, sim::IonicPolicy::None);
        let pseudo = o_scene(h, sim::IonicPolicy::CoresOnly);
        let ledger = qm::radial::ledger::LEDGER;
        // a cache without O's table: the guard counts the bound and generates nothing
        let cold = std::sync::Mutex::new(PseudoCache::default());
        let bound = footprint_in(&cold, &pseudo);
        let reach = ledger.gamma_r * crate::pseudo::GENERIC_R_MAX + ledger.numerics.kspl_step_over_h * h;
        assert!(bound.pseudised && bound.n_valence == 8 && bound.ions == vec![(reach, 4)], "{bound:?}");
        assert!(cold.lock().unwrap().generated() == 0 && !cold.lock().unwrap().has_table(8), "asking the guard generated nothing");
        assert_eq!(footprint_bytes_in(&cold, &pseudo), solve_footprint_bytes(pseudo.cell.n, 8, 5) + pseudo_extra_bytes(pseudo.cell.n, h, 5, &bound.ions));
        // a warm one: the exact valence count and extent
        let warm = std::sync::Mutex::new(PseudoCache::default());
        warm.lock().unwrap().table(8).expect("O is accepted");
        // None: the all-electron formula, 8 electrons (triplet: k_max = 5), whatever the cache holds, plus the
        // all-electron FD witness vectors at the count (plan 2026-11-13 Task 7 fix rounds 1–2; 3 since Task 9c fix round 2)
        assert_eq!(footprint_bytes_in(&warm, &none), solve_footprint_bytes(none.cell.n, 8, 5) + witness_extra_bytes(none.cell.n));
        // pseudo: 6 valence electrons (k_max = 4) plus the extras
        let fp = footprint_in(&warm, &pseudo);
        assert!(fp.pseudised && fp.n_valence == 6, "{fp:?}");
        assert_eq!(fp.ions.len(), 1, "one pseudised ion");
        let extra = pseudo_extra_bytes(pseudo.cell.n, h, 4, &fp.ions);
        assert_eq!(footprint_bytes_in(&warm, &pseudo), solve_footprint_bytes(pseudo.cell.n, 6, 4) + extra);
        // the extras are at least the fields of the review (rho_pc + 2 XC inputs + 6 force fields + the witness growth), x 1.2
        let n = pseudo.cell.n;
        let min_fields = 3 + 6 + 8 * WITNESS_GROWTH_COUNTED as u64;
        assert!(extra as f64 >= 1.2 * (min_fields * field_bytes(n)) as f64, "extra {extra} B for {min_fields} fields of {} B", field_bytes(n));
        // the weights term grows as the grid refines: n_pts = (4 pi / 3)(r / h)^3
        assert!(pseudo_extra_bytes(n, 0.10, 4, &fp.ions) > pseudo_extra_bytes(n, 0.20, 4, &fp.ions));
        // the bound is conservative: the exact extent is inside it
        assert!(fp.ions[0].0 <= reach + 1e-9, "{:?} vs the bound {reach}", fp.ions);
        // H2 under CoresOnly pseudises nothing: counted as today
        let h2c = Scene { ionic: sim::IonicPolicy::CoresOnly, ..h2(33, 1.4) };
        assert_eq!(footprint_bytes_in(&warm, &h2c), footprint_bytes_in(&warm, &h2(33, 1.4)));
        // plan 2026-11-13 Task 7 fix round 2: with the witness term counted (16 then, 3 since Task 9c fix round 2), the Precise H₂O preset
        // (141³ × 0.10, the 14 Bohr box) fits the estimated mode's auditor share again — half of the 12.9 GB adapter
        // budget of the RX 7900 XTX, 6.45 GB — both all-electron and pseudised under `Accepted` (counted at the bound
        // by a cache that holds no table: the conservative estimate)
        let precise = |ionic: sim::IonicPolicy| Scene { cell: Cell { n: 141, h: 0.10 }, ionic, ..h2o(141) };
        let auditor = 6.45e9;
        for (what, policy) in [("all-electron", sim::IonicPolicy::None), ("Accepted (bound)", sim::IonicPolicy::Accepted)] {
            let need = footprint_bytes_in(&std::sync::Mutex::new(PseudoCache::default()), &precise(policy)) as f64;
            println!("Precise H₂O 141³ × 0.10, {what}: {:.2} GB of the auditor's {:.2} GB", need / 1e9, auditor / 1e9);
            assert!(need <= auditor, "{what}: {need:e} B over the auditor's half budget");
        }
    }

    /// Task 9d review m5: the device-limit check counts the Poisson boundary table, the largest binding from n ≈ 169:
    /// at wgpu's default 128 MiB binding a 177³ solve is refused naming the table (147 MB), 161³ (122 MB) passes, and at
    /// this project's adapter limit (2 GiB − 4 B) the largest grid (201³, 190 MB) passes.
    #[test]
    fn the_binding_check_counts_the_poisson_table() {
        let default = 128u64 << 20;
        let refused = binding_refusal(177, default, 256 << 20).expect("177³ is over the default binding");
        assert!(refused.contains("Poisson boundary table") && refused.contains("147 MB"), "{refused}");
        assert_eq!(binding_refusal(161, default, 256 << 20), None);
        assert_eq!(binding_refusal(201, (2u64 << 30) - 4, 2 << 30), None);
        assert_eq!(qm_gpu::poisson::ghost_table_bytes_for(177), 4704 * 177 * 177);
    }

    /// Plan 2026-11-13 Task 9b ruling: the witness cap is min(32, what fits). `witness_cap_for` raises it from the counted
    /// start (`WITNESS_GROWTH_COUNTED`) only as far as the solve at the cap plus the largest orbital view and Harris
    /// display fit the budget, so every display guard passes at any request whenever the cap is above the count; it never
    /// drops below the count (what the scene guard accepted). The app's 12.9 GB at the Accurate H₂O preset (95³ × 0.15)
    /// fits 32; the auditor's half of it at Precise (141³ × 0.10) fits a cap between the count and 32; a budget just over
    /// the counted scene keeps the count.
    #[test]
    fn the_witness_cap_is_what_the_budget_fits() {
        let accurate = Scene { cell: Cell { n: 95, h: 0.15 }, ..h2o(95) };
        assert_eq!(witness_cap_for(&accurate, 12_900_000_000), qm::scf::MAX_WITNESS_GROWTH);
        let precise = Scene { cell: Cell { n: 141, h: 0.10 }, ..h2o(141) };
        let auditor = witness_cap_for(&precise, 6_450_000_000);
        let tight = witness_cap_for(&accurate, gpu_footprint_bytes(&accurate) + 1);
        println!("witness cap: Accurate H₂O 12.9 GB {}, Precise H₂O 6.45 GB {auditor}, Accurate at its counted footprint + 1 B {tight}",
                 witness_cap_for(&accurate, 12_900_000_000));
        assert_eq!(tight, WITNESS_GROWTH_COUNTED);
        assert!((WITNESS_GROWTH_COUNTED..=qm::scf::MAX_WITNESS_GROWTH).contains(&auditor));
        // by construction: a cap above the count leaves room for the largest view and Harris display at that cap
        let m = UNOCCUPIED_MAX as usize;
        for (scene, budget) in [(&accurate, 12_900_000_000u64), (&precise, 6_450_000_000), (&precise, 12_900_000_000)] {
            let w = witness_cap_for(scene, budget);
            if w > WITNESS_GROWTH_COUNTED {
                let n = scene.cell.n;
                let need = gpu_footprint_bytes(scene) - witness_extra_bytes(n) + witness_bytes(n, w)
                    + orbital_view_footprint_bytes(n, 5, 5, m, TextureKind::Colour) + harris_footprint_bytes(n, 5, 5, m, false, w);
                assert!(need <= budget, "cap {w}: {need} B over {budget} B");
            }
        }
    }

    /// Spec §7 risk 6 and MUST-1 of the Task 9 physics review: under a pseudo scene the GPU solver's SCF actually
    /// stops at max(tol, 2 floor(N)) — pseudo-O (N = 6) at the requested 1e-6, O2 (N = 12) at 1.2e-6, with a note —
    /// and every answer records the tolerance it ran at (`EvalStats::scf_tol_used`); H2 under CoresOnly (nothing
    /// pseudised) and the CPU adapter run at exactly the requested one.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only (generates O); cargo test -p providers --release")]
    fn the_gpu_stop_is_floor_aware_for_pseudo_scenes_only() {
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let mut g = GpuKs::new(ctx, 1e-6);
        g.set_display(false);
        let raised = |r: &ForceResult| r.stats.notes.iter().any(|n| n.contains("GPU SCF tolerance raised"));
        let o6 = g.evaluate(&o_scene(0.2, sim::IonicPolicy::CoresOnly), &mut never()).expect("pseudo-O");
        // re-pinned in plan 2026-11-13 Task 9 (was 1e-6): the triplet's minority 2p is partly occupied under FD, so the GPU
        // stops at the partial-level rule max(tol, GPU_PARTIAL_TOL) and `scf_tol_used` now records that stop (F9); the
        // floor-aware part (N = 6: 2·floor = 6e-7 < 1e-6) still raises nothing
        assert_eq!(o6.stats.scf_tol_used, Some(qm::GPU_PARTIAL_TOL), "N = 6, a partial level");
        assert!(!raised(&o6));
        g.reset();
        let o2 = Scene { nuclei: vec![Nucleus::new(8, [-1.15, 0.0, 0.0]), Nucleus::new(8, [1.15, 0.0, 0.0])], cell: Cell { n: 61, h: 0.20 }, ionic: sim::IonicPolicy::CoresOnly, multiplicity: Some(3), ..o_scene(0.2, sim::IonicPolicy::None) };
        let r = g.evaluate(&o2, &mut never()).expect("pseudo-O2 converges at the raised stop");
        let t = r.stats.scf_tol_used.expect("recorded");
        assert!((t - 1.2e-6).abs() < 1e-15, "N = 12: {t}");
        assert!(raised(&r), "{:?}", r.stats.notes);
        g.reset();
        let h2c = g.evaluate(&Scene { ionic: sim::IonicPolicy::CoresOnly, ..h2(33, 1.4) }, &mut never()).expect("H2");
        assert_eq!(h2c.stats.scf_tol_used, Some(1e-6), "nothing pseudised: untouched");
        assert!(!raised(&h2c));
        let c = CpuKs::new(1e-6).evaluate(&h2(33, 1.4), &mut never()).expect("CPU H2");
        assert_eq!(c.stats.scf_tol_used, Some(1e-6), "the CPU has no f32 floor");
    }

    /// Ruling (a) of the stall-stop review: the stop is level-gated (`qm::SCF_STALL_LEVEL_PER_ELECTRON`). On the measured
    /// cases, cold GPU solves at the default settings (stop on): stretched spin-symmetric H₂ at 65³ × 0.25, R = 6.0 and
    /// 107³ × 0.15, R = 5.75 — under aufbau plateaus of 152 and 118 iterations at a best Δρ ≤ 9.6e-5, below L = 2e-3 —
    /// converge, the stop leaving them bitwise as with it off; water's O–H at 1.8 r₀ (95³ × 0.15, `CoresOnly`, the sampling
    /// pass's failing point, Δρ oscillating at 0.3–1.5 under the shared-shell rule, L = 8e-3) stalled early.
    ///
    /// **Re-pinned in plan 2026-11-13 Task 9** (the scenes are Fermi–Dirac now; all-electron now solves witness vectors):
    /// the stretched H₂ converged in **14** and **13** iterations (plan Task 9b: 17 and the count printed, compared with
    /// the stop off in the same run instead of pinned) (were 191 and 144 under aufbau; measured 2026-11-16 with
    /// the stop off too: the same counts and the same F bits), their σ_g/σ_u pair partly occupied (f ≈ 0.93 / 0.07 per
    /// channel, TS 5.2 / 4.8 mHa); and the water point no longer stalls — it **converges in 16** (FD resolves the
    /// oscillation the shared-shell rule's switching drove, spec 2026-11-13 §1). The stop's early end of a hopeless solve
    /// is still pinned in `qm`'s and `qm-gpu`'s stall-watch tests.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only (generates O); cargo test -p providers --release")]
    fn the_level_gated_stall_stop_on_the_measured_cases() {
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        // re-measured in plan Task 9b (witness columns at the guard tolerance, convergence-phase guess band): the stop-off
        // counts are solved here beside the stop-on ones instead of pinned (Task 9: 14 and 13; Task 9b: printed)
        for (n, h, r) in [(65usize, 0.25, 6.0), (107, 0.15, 5.75)] {
            // the measurement's order (`sampling` tests' `pair`): −r/2 first
            let scene = Scene { nuclei: vec![Nucleus::new(1, [-0.5 * r, 0.0, 0.0]), Nucleus::new(1, [0.5 * r, 0.0, 0.0])], cell: Cell { n, h }, ..h2(33, r) };
            let (grid, system, settings) = problem(&scene, 1e-6, &mut Vec::new()).expect("H2");
            assert_eq!(settings.stall_window, qm::SCF_STALL_WINDOW);
            let mut s = qm_gpu::GpuSolver::new(ctx.clone(), &grid);
            let st = s.solve(&system, &settings, false).unwrap_or_else(|e| panic!("H₂ {n}³ R {r}: {e}")).1;
            let off = ScfSettings { stall_window: 0, ..settings };
            let so = s.solve(&system, &off, false).unwrap_or_else(|e| panic!("H₂ {n}³ R {r}, stop off: {e}")).1;
            println!("H₂ {n}³ R {r} (stop on): converged in {} iterations, Δρ {:.2e}, F {:+.10}, TS {:.3e}; stop off {} iterations",
                     st.iterations, st.delta_rho, st.energy.total, st.energy.ts, so.iterations);
            assert_eq!((st.iterations, st.energy.total.to_bits()), (so.iterations, so.energy.total.to_bits()), "the stop must leave this converging solve untouched");
            assert!(st.iterations <= 30, "a converging solve ({} iterations; 191 / 144 under aufbau)", st.iterations);
        }
        let water = Scene {
            nuclei: vec![Nucleus::new(8, [0.0; 3]), Nucleus::new(1, [1.8 * 1.43, 1.8 * 1.11, 0.0]), Nucleus::new(1, [-1.43, 1.11, 0.0])],
            cell: Cell { n: 95, h: 0.15 },
            ionic: sim::IonicPolicy::CoresOnly,
            ..h2(33, 1.4)
        };
        let (grid, system, settings) = problem(&water, 1e-6, &mut Vec::new()).expect("water");
        let mut s = qm_gpu::GpuSolver::new(ctx, &grid);
        let t = Instant::now();
        let r = s.solve(&system, &settings, false).map(|(_, st)| st.iterations);
        println!("water O–H 1.8 r₀ (stop on): {r:?} after {:.1} s", t.elapsed().as_secs_f64());
        // re-pinned in plan 2026-11-13 Task 9 (was an early `ScfStalled`): under FD the point converges
        // re-pinned in plan 2026-11-13 Task 9c (the GPU's buffer column changes the path; 16 before), and in its fix round
        // 2 to 18: the all-electron H of this CoresOnly scene now starts from its free atom (review M4; 16 with the buffer
        // rule of review I1 alone)
        assert_eq!(r.ok(), Some(18), "the water point converges under Fermi–Dirac");
    }

    /// Physics review S3 (stall stop): the shell-release path. The OH radical at 3.5 Bohr (`CoresOnly`: pseudo-O, an
    /// all-electron H; M = 2) at 95³ × 0.15 releases its shared minority cluster at step 37 of 62
    /// (`scf_stall_measurement`), so the stall watch restarts mid-solve: the stop off and the stop on with an observer
    /// give the same bits, and the last Harris–Foulkes energy lands on the reported one. (At 2.5 Bohr nothing is
    /// released: 19 steps, two shared levels.)
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only (generates O); cargo test -p providers --release")]
    fn the_stall_stop_leaves_a_released_pseudo_solve_bitwise() {
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let scene = Scene {
            nuclei: vec![Nucleus::new(8, [0.0; 3]), Nucleus::new(1, [3.5, 0.0, 0.0])],
            cell: Cell { n: 95, h: 0.15 },
            multiplicity: Some(2),
            ionic: sim::IonicPolicy::CoresOnly,
            ..h2(33, 1.4)
        };
        let (grid, system, on) = problem(&scene, 1e-6, &mut Vec::new()).expect("OH");
        assert!(system.ionic.is_some());
        assert_eq!(on.occupations, qm::Occupations::FermiDirac { kt: qm::ELECTRON_KT });
        let off = ScfSettings { stall_window: 0, ..on };
        let mut s = qm_gpu::GpuSolver::new(ctx, &grid);
        let a = s.solve(&system, &off, false).expect("OH converges").1;
        let mut seen: Vec<qm::ScfProgress> = Vec::new();
        let c = s.solve_observed(&system, &on, false, &mut || false, Some(&mut |p: &qm::ScfProgress| seen.push(*p))).unwrap().unwrap().1;
        assert_eq!(a.energy.total.to_bits(), c.energy.total.to_bits());
        assert_eq!(a.delta_rho.to_bits(), c.delta_rho.to_bits());
        assert_eq!(a.iterations, c.iterations);
        let last = seen.last().unwrap();
        println!("pseudo OH 3.5 Bohr M = 2 95³: {} steps, F {:+.10}, F_HF(last) {:+.10} (Δ {:.2e}), TS {:.3e}, partial {:?}; no-gain counter {:?}",
                 a.iterations, a.energy.total, last.energy, last.energy - a.energy.total, a.energy.ts, a.partial, seen.iter().map(|p| p.since_gain).collect::<Vec<_>>());
        // re-pinned in plan 2026-11-13 Task 9 (was `!released.is_empty()`): the scene is FD now, which has no shell release
        // (the stall watch's restart went with the shared-shell rule's phase 2, deleted in Task 10 with `ScfStats.released`);
        // the stop on and off still give the same bits on this slow, partly occupied solve, and the last Harris–Foulkes F
        // lands on the reported F
        assert!(!a.partial.is_empty(), "FD: a partly occupied level: {:?}", a.partial);
        assert!((last.energy - a.energy.total).abs() < 1e-5, "Harris–Foulkes {} vs F {}", last.energy, a.energy.total);
    }

    /// Task 6 review A.6 and the orbital viewer (task-9 carry item 6), re-pinned for Fermi–Dirac in plan 2026-11-13 Task 9
    /// (was the shared-shell rule's cluster): a lone pseudo-O (triplet, 2p^4: three p electrons up full, one p
    /// electron down over the three p levels, exactly degenerate at a grid point so f = 1/3 each) — the CPU solve reports
    /// the three partly occupied ↓ levels, TS = 3 kT s(1/3), the pseudised species and its h_req, and the orbital rows
    /// carry the state's occupations (1, 1/3, ≈ 0 for the stored witnesses) with the electron counts n_up = 4, n_dn = 2.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only (CPU pseudo-O solve); cargo test -p providers --release")]
    fn a_lone_pseudo_o_reports_its_shared_shell_and_fractional_orbital_rows() {
        let mut s = o_scene(0.20, sim::IonicPolicy::CoresOnly);
        s.cell = Cell { n: 41, h: 0.20 };
        let mut c = CpuKs::new(1e-6);
        c.set_orbital_view(view(OrbitalRequest::List));
        let r = c.evaluate(&s, &mut never()).expect("pseudo-O solves");
        // the stats
        assert_eq!(r.stats.pseudised.len(), 1);
        let p = &r.stats.pseudised[0];
        assert_eq!((p.z, p.z_v), (8, 6.0));
        assert!(p.h_req.is_some_and(|h| h >= 0.20) && p.r_c_max > 0.5 && p.record.contains("recipe"), "{p:?}");
        assert_eq!(r.stats.partial.len(), 3, "the minority 2p: three partly occupied levels {:?}", r.stats.partial);
        // |f − 1/3| ≤ 1e-6 (was 1e-12 under the shared-shell rule, whose equal shares were exact; bcd1463, solver-half review
        // M4): under Fermi–Dirac the three ↓ p levels share by f = 1/(1 + e^{(ε − μ)/kT}), so a level spread δε moves each
        // f by |Δf| ≤ f(1 − f)/kT·δε = (2/9)/(5e-3 Ha)·δε ≈ 44·δε; the lattice-split, eigensolver-limited spread at a grid
        // point is δε ≲ 1e-9 Ha (Ritz values are second order in the residual), so |Δf| ≲ 4e-8 — the bound is 25× that
        assert!(r.stats.partial.iter().all(|p| p.spin == 1 && (p.f - 1.0 / 3.0).abs() < 1e-6), "{:?}", r.stats.partial);
        // the closed form of three exactly degenerate levels at f = 1/3: kT·3·s(1/3), s(f) = −f ln f − (1 − f) ln(1 − f)
        let s3 = 3.0 * (-(1.0f64 / 3.0) * (1.0f64 / 3.0).ln() - (2.0 / 3.0) * (2.0f64 / 3.0).ln());
        assert!((r.stats.energy.ts - qm::ELECTRON_KT * s3).abs() < 1e-7, "TS {} vs 3 kT s(1/3) = {}", r.stats.energy.ts, qm::ELECTRON_KT * s3);
        assert_eq!(r.stats.energy.total.to_bits(), (r.stats.energy.internal - r.stats.energy.ts).to_bits(), "F = E − TS");
        assert!(r.stats.branch_jump.is_none() && r.stats.core_overlaps.is_empty());
        assert!(r.stats.notes.iter().all(|n| !n.contains("tolerance raised")), "CPU: no floor-aware stop");
        // the orbital rows
        let o = r.orbitals.expect("a payload");
        for row in &o.rows {
            println!("{} {:?} eps {:+.5} occ {:.4}", row.label, row.spin, row.energy, row.occupation);
        }
        assert!(o.rows.iter().all(|r| matches!(r.label, sim::OrbitalLabel::Index(_))), "a pseudised atom's rows read #k (no core nodes: its 2s would read 1s)");
        assert!(o.notes.iter().any(|n| n.contains("pseudised atom")), "{:?}", o.notes);
        let total: f64 = o.rows.iter().map(|r| r.occupation).sum();
        assert!((total - 6.0).abs() < 1e-9, "sum f = the 6 valence electrons, got {total}");
        let dn_frac: Vec<f64> = o.rows.iter().filter(|r| r.spin == Spin::Down && r.occupation > 1e-6 && r.occupation < 1.0 - 1e-6).map(|r| r.occupation).collect();
        assert_eq!(dn_frac.len(), 3, "three down p levels share the electron");
        assert!(dn_frac.iter().all(|f| (f - 1.0 / 3.0).abs() < 1e-6), "{dn_frac:?}"); // the same |Δf| ≲ 4e-8 bound as above
        assert!(o.rows.iter().filter(|r| r.spin == Spin::Up).all(|r| r.occupation < 1e-6 || r.occupation > 1.0 - 1e-6), "up is closed");
        assert_eq!((o.spin.n_up, o.spin.n_dn), (4, 2), "electrons per channel (the stored gap witnesses are not electrons)");
    }

    /// Task 6 review A.6 (the atom-reference scan under the pseudo policies): every candidate multiplicity of a
    /// lone atom converges (under Fermi–Dirac since plan 2026-11-13 Task 9; the shared-shell rule before), and the scan
    /// keeps O at M = 3, N at M = 4 and C at M = 3 —
    /// Hund's rule, from the energies alone (`sim::references::solve_atom`, ruling S9). GPU, 61³ × 0.20.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only (generates C, N, O); cargo test -p providers --release")]
    fn the_pseudo_atom_reference_scan_keeps_hunds_multiplicity() {
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let cell = Cell { n: 61, h: 0.20 };
        for (z, want) in [(8u8, 3u32), (7, 4), (6, 3)] {
            let mut g = GpuKs::new(ctx.clone(), 1e-6);
            let mass = Nucleus::new(z, [0.0; 3]).mass;
            let r = sim::references::solve_atom(&mut g, &cell, None, sim::IonicPolicy::CoresOnly, z, mass, &mut never())
                .unwrap_or_else(|e| panic!("Z = {z}: every candidate converges and one wins: {e}"));
            println!("Z = {z}: M = {:?}, E = {:.6} Ha ({} electrons in the table)", r.multiplicity, r.energy, r.radial.electrons());
            assert_eq!(r.multiplicity, Some(want), "Z = {z}");
            assert!(r.energy > -40.0, "a valence energy, not the all-electron total: {}", r.energy);
        }
    }
    /// Plan 2026-11-13 Task 9 (F9; spec §5.1): `EvalStats::scf_tol_used` is the tolerance the converged step was tested
    /// against (`ScfStats::tol_used`). A lone pseudo-O at M = 3 (its minority 2p partly occupied) through the GPU provider
    /// at the sample tolerance (1e-6, `branch_jump::SAMPLE_TOL`) stopped at the partial-level rule, `GPU_PARTIAL_TOL`;
    /// gapped all-electron H₂ at the requested tolerance. The `branch_jump` guard judges each on that tier.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only (generates O); cargo test -p providers --release")]
    fn scf_tol_used_reports_the_partial_stop() {
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let tol = crate::branch_jump::SAMPLE_TOL;
        let mut g = GpuKs::new(ctx, tol);
        g.set_display(false);
        let o = g.evaluate(&o_scene(0.2, sim::IonicPolicy::CoresOnly), &mut never()).expect("pseudo-O M = 3");
        println!("pseudo-O M = 3 61³ × 0.20: scf_tol_used {:?}, {} steps, TS {:.6e}, partial {:?}", o.stats.scf_tol_used, o.stats.scf_steps, o.stats.energy.ts, o.stats.partial);
        assert!(!o.stats.partial.is_empty(), "the minority 2p is partly occupied");
        assert_eq!(o.stats.scf_tol_used, Some(qm::GPU_PARTIAL_TOL), "the partial-level stop, not the requested {tol:e}");
        g.reset();
        let h2 = g.evaluate(&h2(33, 1.4), &mut never()).expect("H₂");
        println!("H₂ 33³: scf_tol_used {:?}, TS {:.1e}", h2.stats.scf_tol_used, h2.stats.energy.ts);
        assert!(h2.stats.partial.is_empty(), "gapped");
        assert_eq!(h2.stats.scf_tol_used, Some(tol), "a gapped solve stops at the requested tolerance");
    }

    /// Plan 2026-11-13 Task 8 review M6 (Task 9): the `branch_jump` guard picks its surface as `system.ionic.is_none()`
    /// (no cache lock on the evaluation path); that is exactly `is_all_electron_tag` of the scene's provenance tag, since
    /// `setup_for` and the tag apply one per-species rule. Review M3 (fix round 1): H₂ under `Accepted` and `All` too,
    /// with the GPU dimer backend the tests use (the gated H: about 1 min once per process); without an adapter those
    /// two legs are skipped.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only (generates O, and the gated H on the GPU); cargo test -p providers --release")]
    fn the_guard_surface_is_the_tag_surface() {
        use sim::IonicPolicy::{All, Accepted, CoresOnly, None as AllElectron};
        let water = |ionic| Scene { ionic, cell: Cell { n: 61, h: 0.2 }, ..h2o(61) };
        let h2c = |ionic| Scene { ionic, ..h2(33, 1.4) };
        let cases = [h2c(AllElectron), h2c(CoresOnly), water(AllElectron), water(CoresOnly), o_scene(0.2, AllElectron), o_scene(0.2, CoresOnly),
                     o_scene(0.2, All), o_scene(0.2, Accepted)];
        let check = |scene: &Scene| {
            let (_, system, _) = problem(scene, 1e-6, &mut Vec::new()).expect("posed");
            let tag = crate::pseudo::provenance_tag(scene);
            assert_eq!(system.ionic.is_none(), crate::pseudo::is_all_electron_tag(&tag), "{:?} {:?}: tag {tag}", scene.ionic, scene.nuclei.iter().map(|n| n.z).collect::<Vec<_>>());
            tag
        };
        for scene in &cases {
            check(scene);
        }
        // H and He are dimer-gated: with the GPU backend `Accepted` and `All` pseudise H (the gate runs on the GPU)
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        crate::pseudo::set_dimer_backend(crate::pseudo::DimerBackend::Gpu(ctx));
        for policy in [Accepted, All] {
            let tag = check(&h2c(policy));
            assert!(!crate::pseudo::is_all_electron_tag(&tag), "{policy:?} H₂ with the GPU gate is pseudised: {tag}");
        }
    }

    /// Spec 2026-11-13 NOTE-6 / V8 (plan Task 9): the all-electron witness cap on an anion whose LSDA HOMO is unbound, in
    /// the largest **preset** box at the app's default quality — the 14 Bohr box of the h2o/h2s/hcl/ph3/sih4/lih presets
    /// (57³ × 0.25) regridded to Accurate, 95³ × 0.15. This pin covers that box only: the app's largest box (Scene ▸ Grid,
    /// `MAX_GRID_N` = 201, 30 Bohr at h = 0.15) is `all_electron_h_minus_witness_cap_at_the_largest_box`, a plan Task 9b
    /// RED test (review I1). All-electron H⁻ (2 electrons, singlet), cold, at the app's
    /// interactive tolerance 1e-5, through `problem` (FD at kT = 5 mHa) and the GPU solver. The measured outcome is pinned
    /// (2026-11-16, RX 7900 XTX): it **converges**, in 18 steps (74.5 s), F = −0.474954 Ha, TS = 1.9e-7 Ha (gapped); the
    /// HOMO is unbound in LSDA (ε = +0.089 Ha, held by the box) and the witness arms took 10 extra vectors ↑ and 11 ↓, under
    /// the cap of 16 (`MAX_WITNESS_GROWTH`) — 5–6 to spare. A larger box (Scene ▸ Grid allows up to 201³, 30 Bohr at
    /// h = 0.15) can still exceed it: N(E) ∝ L³ (spec §4.2), the cost and the cap are plan Task 9b's.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only (GPU, 95³ all-electron); cargo test -p providers --release")]
    fn all_electron_h_minus_witness_cap() {
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let scene = Scene { nuclei: vec![Nucleus::new(1, [0.0; 3])], cell: Cell { n: 95, h: 0.15 }, charge: -1, multiplicity: None, sigma: None,
                            external_field: [0.0; 3], ionic: sim::IonicPolicy::None, train_mask: None };
        let (grid, system, settings) = problem(&scene, 1e-5, &mut Vec::new()).expect("H⁻ is posed");
        assert!(system.ionic.is_none() && (system.n_up, system.n_dn) == (1, 1));
        let mut s = qm_gpu::GpuSolver::new(ctx, &grid);
        let t = Instant::now();
        let r = s.solve(&system, &settings, false);
        let wall = t.elapsed().as_secs_f64();
        let st = match r {
            Ok((_, st)) => st,
            Err(e) => panic!("H⁻ 95³ × 0.15: the pinned outcome is convergence, got after {wall:.1} s: {e}"),
        };
        let state = s.state().expect("converged");
        let extra = [state.orbitals_up.len() - system.n_up, state.orbitals_dn.len() - system.n_dn];
        println!("H⁻ 95³ × 0.15 (14 Bohr): converged in {} steps, {wall:.1} s; F {:+.8}, TS {:.3e}, tol_used {:e}; extra vectors ↑ {} ↓ {} (cap {}); ε↑ {:?}",
                 st.iterations, st.energy.total, st.energy.ts, st.tol_used, extra[0], extra[1], qm::scf::MAX_WITNESS_GROWTH, state.evals_up);
        assert!(st.iterations <= 30, "{} steps (measured 18; the V2 rule ⌈1.5 n⌉ to a multiple of 5)", st.iterations);
        assert!((st.energy.total - -0.474954).abs() <= 1e-5, "F {} vs the measured −0.474954", st.energy.total);
        assert!(st.energy.ts < 1e-6, "gapped: TS {:.2e}", st.energy.ts);
        assert!(extra.iter().all(|&e| e >= 1 && e <= qm::scf::MAX_WITNESS_GROWTH), "{extra:?}");
        let homo = state.evals_up.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(homo > 0.0, "LSDA's H⁻ HOMO is unbound (held by the box): ε = {homo}");
    }

    /// Spec 2026-11-13 NOTE-6 / V8 at the app's **largest** box (review I1), re-ruled by plan Task 9c fix round 2 (Decision
    /// 1): all-electron H⁻ at 201³ × 0.15 (`MAX_GRID_N`, 30 Bohr) under the app's budget (half the adapter's device-local
    /// memory, `app::quality::vram_budget`'s rule; 12.9 GB on the RX 7900 XTX). Weyl's estimate gives ≈ 79 box states
    /// below ≈ 0.21 Ha at L = 30 (10–11 witness vectors were measured in the 14 Bohr box): no cap a 24 GB card can hold
    /// reaches them. The guard now counts the start allocation (`WITNESS_GROWTH_COUNTED`), so the scene is accepted, with
    /// a cap of what the budget leaves (`witness_cap_for`) — well below 79, so its solve ends on the cap error naming the
    /// budget, never an allocation past it (that path is solved in `a_growth_over_the_budget_is_the_named_cap_error`; this
    /// 201³ solve is not run: minutes of grinding toward the same error). Guard-level only: no GPU work.
    #[test]
    fn all_electron_h_minus_witness_cap_at_the_largest_box() {
        let budget = 12_880_000_000u64; // half of the RX 7900 XTX's 25.76 GB (the app's rule)
        let scene = Scene { nuclei: vec![Nucleus::new(1, [0.0; 3])], cell: Cell { n: 201, h: 0.15 }, charge: -1, multiplicity: None, sigma: None,
                            external_field: [0.0; 3], ionic: sim::IonicPolicy::None, train_mask: None };
        let need = gpu_footprint_bytes(&scene);
        let cap = witness_cap_for(&scene, budget);
        println!("H⁻ 201³ × 0.15: needs {:.2} GB with the counted {WITNESS_GROWTH_COUNTED} witness vectors per spin, budget {:.2} GB; cap {cap}; one vector per spin {:.2} GB",
                 need as f64 / 1e9, budget as f64 / 1e9, witness_bytes(201, 1) as f64 / 1e9);
        assert!(need <= budget, "the counted start fits");
        assert!((WITNESS_GROWTH_COUNTED..79).contains(&cap), "a cap below Weyl's ≈ 79 box states: the solve ends on the cap error ({cap})");
        // the cap is what the budget leaves: one more vector per spin, with the largest displays, does not fit
        let (m, n) = (UNOCCUPIED_MAX as usize, 201);
        let at = |w: usize| need - witness_extra_bytes(n) + witness_bytes(n, w) + orbital_view_footprint_bytes(n, 1, 1, m, TextureKind::Colour)
            + harris_footprint_bytes(n, 1, 1, m, false, w);
        assert!(cap == qm::scf::MAX_WITNESS_GROWTH || at(cap + 1) > budget, "cap {cap}: one more vector still fits");
    }

    /// Plan 2026-11-13 Task 9c fix round 2 (Decision 1): a solve whose witness growth would pass what the budget leaves
    /// ends with the solver's cap error naming the budget — never an allocation past it. All-electron H⁻ at 95³ × 0.15
    /// (the 14 Bohr V8 box: 8 witness vectors per spin measured) under a budget that fits the counted start plus one
    /// vector per spin with the largest displays (cap = count + 1 = 4): the cold solve stops at the cap with
    /// `QmError::Numerical` "no gap witness within 4 extra levels … the most the … budget fits".
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only (GPU); cargo test -p providers --release")]
    fn a_growth_over_the_budget_is_the_named_cap_error() {
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let scene = Scene { nuclei: vec![Nucleus::new(1, [0.0; 3])], cell: Cell { n: 95, h: 0.15 }, charge: -1, multiplicity: None, sigma: None,
                            external_field: [0.0; 3], ionic: sim::IonicPolicy::None, train_mask: None };
        let (n, m, w) = (95, UNOCCUPIED_MAX as usize, WITNESS_GROWTH_COUNTED + 1);
        let budget = gpu_footprint_bytes(&scene) - witness_extra_bytes(n) + witness_bytes(n, w) + orbital_view_footprint_bytes(n, 1, 1, m, TextureKind::Colour)
            + harris_footprint_bytes(n, 1, 1, m, false, w);
        assert_eq!(witness_cap_for(&scene, budget), w, "the budget leaves one vector per spin past the count");
        let mut g = GpuKs::new(ctx, 1e-5).with_vram_budget(budget);
        let t = Instant::now();
        let r = g.evaluate(&scene, &mut never());
        println!("H⁻ 95³ × 0.15 under {:.3} GB (cap {w}): {:?} after {:.1} s", budget as f64 / 1e9, r.as_ref().map(|r| r.stats.scf_steps), t.elapsed().as_secs_f64());
        match r {
            Err(ProviderError::Solver(SolverError::Numerical(m))) => {
                assert!(m.contains(&format!("within {w} extra levels")) && m.contains("budget"), "the cap error names the cap and the budget: {m}");
            }
            Err(e) => panic!("expected the named cap error, got {e}"),
            Ok(r) => panic!("expected the named cap error, got a result ({} steps)", r.stats.scf_steps),
        }
        let peak = g.solver.as_ref().map(|s| s.profile().peak_block);
        assert!(peak.is_some_and(|p| p <= 1 + w), "the block never passed the cap: peak {peak:?}");
    }

    /// Review I4: the cold all-electron H₂O at 97³ × 0.25 (24 Bohr) — `unoccupied_budget_in_play`'s former leg and
    /// `gpu_vs_cpu::timing_table`'s held case — under FermiDirac through `problem`, at the app's 1e-5. Before plan Task
    /// 9b it ended after 630 s with "no gap witness within 16 extra levels of spin 0" on a transient step (Task 7 fix
    /// round 1's diagnosis: steps 3–4 need 16–19 extra vectors). With 9b's witness columns at the guard tolerance,
    /// the doubled growth and the cap of 32 (no budget here) it converges; measured in the 9b report.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only (GPU, 97³ all-electron); cargo test -p providers --release")]
    fn cold_all_electron_h2o_at_97_converges() {
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let (grid, system, settings) = problem(&h2o(97), 1e-5, &mut Vec::new()).expect("H₂O 97³ × 0.25 is posed");
        assert!(system.ionic.is_none());
        let mut s = qm_gpu::GpuSolver::new(ctx, &grid);
        let t = Instant::now();
        let r = s.solve(&system, &settings, false).map(|(_, st)| st);
        let wall = t.elapsed().as_secs_f64();
        let p = s.profile().clone();
        println!("H₂O 97³ × 0.25 all-electron, cold: {:?} after {wall:.1} s; peak block {}, witness growths {}, buffer witnesses {} / growths {}, residual growths {}, class raises {}, locked calls {}, frozen steps {}, DIIS resets {}, freeze trims {}, restores {}",
                 r.as_ref().map(|st| (st.iterations, st.energy.total, st.energy.ts, st.max_eigen_residual, st.max_guard_residual)),
                 p.peak_block, p.witness_growths, p.buffer_witnesses, p.buffer_growths, p.witness_residual_growths, p.class_raises, p.locked_calls, p.frozen_steps, p.diis_resets, p.diis_freeze_trims, p.eigen_restores);
        let st = r.unwrap_or_else(|e| panic!("H₂O 97³ × 0.25 cold: {e}"));
        let state = s.state().expect("converged");
        println!("  stored ↑ {} ↓ {} (n {} / {}), ε↑ {:?}", state.orbitals_up.len(), state.orbitals_dn.len(), system.n_up, system.n_dn, state.evals_up);
        assert!(st.energy.total.is_finite());
    }

    /// Plan 2026-11-13 Task 9c, the VRAM measurement: the cold all-electron H₂O of the app's
    /// `a_change_over_the_vram_budget_is_refused` (177³ × 0.08, 14.08 Bohr) under FermiDirac at the app's 1e-5 — the
    /// solve's peak witness allocation (`SolveProfile::peak_block` − n_σ extra vectors per spin), against which the
    /// footprint's count (`WITNESS_GROWTH_COUNTED`) is set. (Until fix round 2 it ran at 167³ × 0.0848, the same box:
    /// 177³'s Poisson boundary table was over wgpu's default 128 MiB storage binding.) A measurement:
    /// `cargo test -p providers --release --lib -- --ignored --nocapture cold_all_electron_h2o_at_177_peak`.
    #[test]
    #[ignore = "measurement (GPU, 177³ all-electron, about a minute)"]
    fn cold_all_electron_h2o_at_177_peak() {
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let mut sc = h2o(177);
        sc.cell.h = 0.08;
        let (grid, system, settings) = problem(&sc, 1e-5, &mut Vec::new()).expect("H₂O 177³ × 0.08 is posed");
        assert!(system.ionic.is_none());
        let mut s = qm_gpu::GpuSolver::new(ctx, &grid);
        let t = Instant::now();
        let r = s.solve(&system, &settings, false).map(|(_, st)| st);
        let wall = t.elapsed().as_secs_f64();
        let p = s.profile().clone();
        println!("H₂O 177³ × 0.08 (14.08 Bohr) all-electron, cold: {:?} after {wall:.1} s; peak block {} (n↑ {}, n↓ {}), witness growths {}, trims {}, class raises {}, locked calls {}, DIIS resets {}",
                 r.as_ref().map(|st| (st.iterations, st.energy.total, st.max_eigen_residual)), p.peak_block, system.n_up, system.n_dn,
                 p.witness_growths, p.witness_trims, p.class_raises, p.locked_calls, p.diis_resets);
        r.unwrap_or_else(|e| panic!("H₂O 177³ cold: {e}"));
    }

    // ---- the Harris display (module doc, "Harris display"; live-orbitals Task 4) ----

    /// The analytic H 1s profile e^{−2r}/π on `s`'s surface (grid, σ, ionic policy): a stand-in for
    /// the isolated-atom profile the app's auditor solves (`sim::profiles`' own test atom).
    fn h_atoms(s: &Scene) -> Arc<sim::HarrisAtoms> {
        let (n, h) = (s.cell.n, s.cell.h);
        let grid = Grid::cube(n, h);
        let rho: Vec<f32> = grid.sample(|p| (-2.0 * (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()).exp() / std::f64::consts::PI)
            .into_iter().map(|v| v as f32).collect();
        Arc::new(sim::HarrisAtoms { profiles: vec![sim::radial_profile(1, &rho, n, h, grid.origin, s.sigma, s.ionic)] })
    }

    /// `harris_display` with H profiles, unwrapped: a `GpuKs` always answers `Solved`.
    fn harris(p: &mut GpuKs, s: &Scene) -> Result<OrbitalDisplay, ProviderError> {
        match p.harris_display(s, &h_atoms(s), &mut never()) {
            Some(HarrisAnswer::Solved(r)) => r,
            other => panic!("GpuKs answers now: {other:?}"),
        }
    }

    fn homo_view() -> OrbitalViewRequest {
        view(OrbitalRequest::Orbital { spin: Spin::Up, track: sim::TRACK_HOMO, squared: false })
    }

    /// H₃⁺ (ruling R5: the validate suite's equilateral triangle, side 2.1 Bohr, charge +1) on 33³ ×
    /// 0.25 — built inline, providers cannot reach validate's common module.
    fn h3_plus() -> Scene {
        let r = 2.1 / 3f64.sqrt(); // the circumradius
        let at = |deg: f64| Nucleus::new(1, [r * deg.to_radians().cos(), r * deg.to_radians().sin(), 0.0]);
        let mut s = scene(33, vec![at(0.0), at(120.0), at(240.0)]);
        s.charge = 1;
        s
    }

    /// Spec §4.1 / Review Focus 4: H₃⁺ — three H atoms carry 3 electrons, the scene has 2 — is solved
    /// with the superposition rescaled by 2/3, and the payload says so (its kind and a note). One
    /// occupied level per spin, the HOMO's field drawn.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn charged_scene_is_rescaled_and_labelled() {
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let mut p = GpuKs::new(ctx, 1e-6);
        p.set_orbital_view(homo_view());
        let s = h3_plus();
        let o = harris(&mut p, &s).expect("a Harris payload");
        let OrbitalKind::Harris { experience, rescale, wall_s } = o.kind.clone() else { panic!("kind {:?}", o.kind) };
        let occ: Vec<(Spin, f64, f64)> = o.rows.iter().filter(|r| r.occupation > 0.0).map(|r| (r.spin, r.energy, r.occupation)).collect();
        println!("H₃⁺ Harris (33³ × 0.25): rescale {rescale:.15}, {wall_s:.3} s, occupied {occ:?}, notes {:?}", o.notes);
        assert!(experience.is_none(), "Phase A");
        assert!((rescale - 2.0 / 3.0).abs() < 1e-12, "{rescale}");
        assert!(o.notes.iter().any(|n| n.contains("rescaled") && n.contains("0.6667")), "{:?}", o.notes);
        assert_eq!(occ.iter().map(|r| (r.0, r.2)).collect::<Vec<_>>(), vec![(Spin::Up, 1.0), (Spin::Down, 1.0)]);
        assert!(o.field.is_some(), "the HOMO is drawn");
    }

    /// Review Focus 4: H⁺ alone has no electrons — a note-only Harris payload, no field, and the
    /// probe's display slot never started (nothing was solved).
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn no_electrons_is_a_note_not_a_solve() {
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let probe = ActivityProbe::new();
        let mut p = GpuKs::new(ctx, 1e-6).with_probe(probe.clone(), Slot::Side);
        p.set_orbital_view(homo_view());
        let mut s = scene(33, vec![Nucleus::new(1, [0.1, 0.0, 0.0])]);
        s.charge = 1;
        let o = harris(&mut p, &s).expect("a note, not an error");
        println!("H⁺: notes {:?}", o.notes);
        assert!(o.field.is_none() && o.rows.is_empty());
        assert!(o.notes.iter().any(|n| n.contains("no electrons")), "{:?}", o.notes);
        assert!(matches!(o.kind, OrbitalKind::Harris { .. }));
        let v = probe.view();
        assert_eq!((v.display, v.display_solves, v.side.solves), (None, 0, 0), "no display solve, no teacher solve");
        assert!(probe.take_display_solves().is_empty());
    }

    /// G4 / anti-goals: a Harris display solve on a GPU teacher (probe slot Side) after a converged
    /// evaluation is not a teacher solve — the side slot's count is unchanged, the display slot
    /// counts one `DisplayJob::Harris` — and it leaves the converged state alone: `sample_density`
    /// answers bitwise as before, and `orbital_display_now` still shows the converged state.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn harris_is_not_a_teacher_solve() {
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let probe = ActivityProbe::new();
        let mut p = GpuKs::new(ctx, 1e-6).with_probe(probe.clone(), Slot::Side);
        p.set_orbital_view(view(OrbitalRequest::List));
        p.evaluate(&h2(33, 1.4), &mut never()).expect("the SCF");
        let points: Vec<[f64; 3]> = chord([-1.5, 0.1, 0.0], [1.5, -0.1, 0.2]).collect();
        let bits = |v: Vec<f64>| v.into_iter().map(f64::to_bits).collect::<Vec<_>>();
        let before = bits(p.sample_density(&points).expect("a converged density"));
        let (side, shown) = (probe.view().side.solves, probe.view().display_solves);
        probe.take_display_solves();
        let o = harris(&mut p, &h2(33, 1.6)).expect("a Harris payload");
        assert!(matches!(o.kind, OrbitalKind::Harris { rescale, .. } if rescale == 1.0), "{:?}", o.kind);
        let v = probe.view();
        assert_eq!(v.side.solves, side, "not a teacher solve");
        assert!(!v.side.running && !v.main.running && v.display.is_none());
        assert_eq!(v.display_solves, shown + 1);
        let done = probe.take_display_solves();
        let last = done.last().expect("one display solve recorded");
        println!("H₂ Harris at R = 1.6 after the SCF at 1.4: {:.0} ms, rows {:?}", last.ms, o.rows.iter().map(|r| (r.spin, r.energy)).collect::<Vec<_>>());
        assert_eq!((done.len(), last.job, last.interrupted), (1, sim::DisplayJob::Harris, false));
        assert_eq!(bits(p.sample_density(&points).expect("still the converged density")), before, "the converged density is untouched");
        let now = p.orbital_display_now(&mut never()).expect("the converged state is still current");
        assert_eq!(now.kind, OrbitalKind::Converged);
        assert_eq!(now.positions, vec![[0.7, 0.0, 0.0], [-0.7, 0.0, 0.0]], "the evaluated geometry, not the Harris one");
    }

    /// Spec §4.2 "VRAM": a budget of the solve's and the view's footprints + 1 byte leaves no room
    /// for the Harris state — the payload is a note naming the GB, and nothing is solved.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn harris_over_the_vram_budget_is_a_note() {
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let s = h2(33, 1.4);
        // + the all-electron witness term `gpu_footprint_bytes` counts on the FD surface (smearing Task 7/9; scf-stall ← main merge)
        let budget = solve_footprint_bytes(33, 2, 1) + witness_extra_bytes(33) + orbital_view_footprint_bytes(33, 1, 1, 0, TextureKind::Scalar) + 1;
        assert!(gpu_footprint_bytes(&s) <= budget, "the scene itself fits: only the Harris state is over");
        let probe = ActivityProbe::new();
        let mut p = GpuKs::new(ctx, 1e-6).with_vram_budget(budget).with_probe(probe.clone(), Slot::Side);
        p.set_orbital_view(homo_view());
        let o = harris(&mut p, &s).expect("a note, not an error");
        println!("over the budget by the Harris term ({} B): {:?}", harris_footprint_bytes(33, 1, 1, 0, false, WITNESS_GROWTH_COUNTED), o.notes);
        assert!(o.field.is_none() && o.rows.is_empty());
        assert!(o.notes.iter().any(|n| n.contains("GB") && n.contains("orbital display solve")), "{:?}", o.notes);
        assert!(matches!(o.kind, OrbitalKind::Harris { .. }));
        assert_eq!((probe.view().display_solves, probe.view().side.solves), (0, 0), "no solve of any kind");
    }

    /// Task 4 review M3: `problem`'s setup notes (here CoresOnly's all-electron fallback for Li)
    /// reach the Harris payload — a note-only one too (no Li profile is given, so nothing is
    /// solved). M4: a solved Harris state is dropped by `reset` and by `drop_view_after_fault`.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only (radial Li atom); cargo test -p providers --release")]
    fn harris_carries_setup_notes_and_reset_drops_its_state() {
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let mut p = GpuKs::new(ctx, 1e-6);
        p.set_orbital_view(homo_view());
        let li = Scene { ionic: sim::IonicPolicy::CoresOnly, ..scene(33, vec![Nucleus::new(3, [0.1, 0.0, 0.0])]) };
        let o = harris(&mut p, &li).expect("a note, not an error");
        println!("Li under CoresOnly with only H profiles: {:?}", o.notes);
        assert!(o.rows.is_empty() && o.notes.iter().any(|n| n.contains("no isolated-atom profile for Li")), "{:?}", o.notes);
        assert!(o.notes.iter().any(|n| n.contains("Li") && n.contains("all-electron")), "the setup's fallback note: {:?}", o.notes);
        for drop in [GpuKs::reset as fn(&mut GpuKs), GpuKs::drop_view_after_fault] {
            p.set_orbital_view(homo_view());
            harris(&mut p, &h2(33, 1.4)).expect("a Harris payload");
            assert!(p.harris.state().is_some(), "solved");
            drop(&mut p);
            assert!(p.harris.state().is_none(), "dropped");
        }
    }

    /// Unoccupied rows of a payload per channel, as (spin, ε bits).
    fn unocc_eps(o: &OrbitalDisplay) -> Vec<(Spin, u64)> {
        o.rows.iter().filter(|r| r.occupation == 0.0).map(|r| (r.spin, r.energy.to_bits())).collect()
    }

    /// Spec §4.2 / Task 4: m = 2. A Harris payload shows its own unoccupied rows; the converged
    /// payloads after it (another geometry) show their own eigensolve's rows or none, never the
    /// Harris vectors (whose ε differ). In play (the plan alone would Keep the held vectors: same m,
    /// no budget measured) the converged payload shows none; paused, its own solve's.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn converged_payload_after_harris_never_shows_harris_unoccupied() {
        let Some(ctx) = gpu() else { return };
        let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let mut p = GpuKs::new(ctx, 1e-6);
        p.set_orbital_view(with_unoccupied(OrbitalRequest::List, 2, false));
        p.evaluate(&h2(33, 1.4), &mut never()).expect("the SCF");
        let h = harris(&mut p, &h2(33, 1.6)).expect("a Harris payload");
        let harris_eps = unocc_eps(&h);
        let u = h.unoccupied.as_ref().expect("m = 2 asked for");
        println!("Harris unoccupied rows {:?} (status current {}, shown {:?})", h.rows.iter().filter(|r| r.occupation == 0.0).map(|r| (r.spin, r.energy)).collect::<Vec<_>>(), u.current, u.shown);
        assert!(unocc_rows(&h).iter().all(|&k| k >= 1) && u.current, "the Harris payload shows its own: {:?}", unocc_rows(&h));
        // play: the converged state at another geometry, then a re-display
        let r = p.evaluate(&h2(33, 1.5), &mut never()).expect("the SCF");
        let now = p.orbital_display_now(&mut never()).expect("a converged payload");
        for o in r.orbitals.iter().chain([&now]) {
            assert_eq!(o.kind, OrbitalKind::Converged);
            let seen = unocc_eps(o);
            println!("converged (play) unoccupied rows: {:?}", seen.iter().map(|&(s, e)| (s, f64::from_bits(e))).collect::<Vec<_>>());
            assert!(seen.is_empty(), "play, no eigensolve of this state yet: none, never the Harris rows ({seen:?})");
        }
        // paused: its own eigensolve's rows, current, none of them the Harris ones
        p.set_orbital_view(with_unoccupied(OrbitalRequest::List, 2, true));
        let own = p.orbital_display_now(&mut never()).expect("a converged payload");
        let seen = unocc_eps(&own);
        println!("converged (paused) unoccupied rows: {:?}", own.rows.iter().filter(|r| r.occupation == 0.0).map(|r| (r.spin, r.energy)).collect::<Vec<_>>());
        assert!(!seen.is_empty() && own.unoccupied.as_ref().is_some_and(|u| u.current), "its own eigensolve ran");
        assert!(seen.iter().all(|e| !harris_eps.contains(e)), "never a Harris row: {seen:?} vs {harris_eps:?}");
    }
}
