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
//! accounting is the evaluation's solve and forces.
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
//! solve that returns `QmError::ScfNotConverged` exactly once **cold** ([`retry_cold_on_stall`]:
//! CPU without the previous state; GPU with `warm = false` and density extrapolation off for
//! that solve — after the failure the solver holds no state and no ρ_{t−1} anyway). A converged
//! retry is returned with a note in [`EvalStats::notes`] ("warm SCF did not converge …; retried
//! cold"), its `scf_steps`/`readbacks` including the failed attempt's: the step costs about
//! twice as much, and the answer is a converged one. A cold solve that fails, or a failed
//! retry, is the error as before; any other error is never retried.
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
//! total VRAM, so the caller picks the budget (the app: `quality::vram_budget_bytes`). The
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
//! with m = 4 adds ≈ 0.59 GB to its 0.91 GB solve (`the_orbital_view_is_charged_to_the_vram_budget`). The estimated mode's
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
use crate::orbitals::{CpuView, PickKind, Pool, TextureKind, UnoccupiedPlan, UnoccupiedState, ViewChange, ViewState, UNOCCUPIED_MAX};
use crate::surrogate::box_affected_face_density;
use gpu::{ColourVolume3D, GpuContext, Volume3D};
use qm::{Grid, ScfSettings, ScfStats, System};
use qm_gpu::orbitals::{OrbitalView, Product};
use sim::{
    nothing_to_solve, ActivityProbe, Cell, Display, DisplayInterrupt, EnergyBreakdown, EvalStats, ForceProvider, ForceResult, OrbitalDisplay,
    OrbitalField, OrbitalViewRequest, ProviderError, Scene, Slot, SolverError, Source,
};
use std::sync::Arc;
use std::time::Instant;

/// `qm`'s errors as the boundary's [`SolverError`] (variant for variant, the same messages).
fn solver_error(e: qm::QmError) -> ProviderError {
    ProviderError::Solver(match e {
        qm::QmError::CoincidentNuclei(i, j) => SolverError::CoincidentNuclei(i, j),
        qm::QmError::ScfNotConverged(n, d) => SolverError::ScfNotConverged(n, d),
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
pub fn gpu_footprint_bytes(scene: &Scene) -> u64 {
    if scene.nuclei.is_empty() {
        return 0;
    }
    let n_el = scene.electrons().max(0) as u64;
    let unpaired = match scene.multiplicity {
        Some(m) => (m as u64).saturating_sub(1).min(n_el),
        None => n_el % 2,
    };
    let k_max = (n_el + unpaired).div_ceil(2);
    solve_footprint_bytes(scene.cell.n, n_el, k_max)
}

/// [`gpu_footprint_bytes`] from the counts: n³ grid, N electrons, k_max = max(n↑, n↓).
pub fn solve_footprint_bytes(n: usize, n_el: u64, k_max: u64) -> u64 {
    let fields = 2 * n_el + 5 + 8 + 24 + 60 + (8 * k_max + 2) + 4;
    (fields as f64 * field_bytes(n) as f64 * 1.2) as u64
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

/// Pre-validates what `System::new` would panic on and builds the solve's inputs: the grid of
/// `scene.cell`, the system (ΣZ − charge electrons, the multiplicity) and the SCF settings
/// (`ScfSettings::for_grid` with `tol` and σ = `scene.sigma` or h).
fn problem(scene: &Scene, tol: f64) -> Result<(Grid, System, ScfSettings), ProviderError> {
    scene.check_inside_cell()?;
    let grid = Grid::cube(scene.cell.n, scene.cell.h);
    let nuclei: Vec<qm::Nucleus> = scene.nuclei.iter().map(|n| qm::Nucleus { pos: n.pos, z: n.z }).collect();
    let protons: i64 = nuclei.iter().map(|n| n.z as i64).sum();
    let total = protons - scene.charge as i64;
    if total < 0 {
        return Err(ProviderError::Invalid(format!("charge {} exceeds the {protons} protons", scene.charge)));
    }
    if let Some(m) = scene.multiplicity {
        let n = total as usize;
        if m < 1 || (m as usize - 1) > n || !(n - (m as usize - 1)).is_multiple_of(2) {
            return Err(ProviderError::Invalid(format!("multiplicity {m} impossible for {n} electrons")));
        }
    }
    let system = System::new(nuclei, scene.charge, scene.multiplicity);
    let mut settings = ScfSettings::for_grid(&grid);
    settings.tol = tol;
    if let Some(s) = scene.sigma {
        settings.sigma = s;
    }
    Ok((grid, system, settings))
}

/// A warm solve that stalled before the cold retry (R15): its SCF steps and final Δρ.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Stall {
    pub iterations: usize,
    pub delta_rho: f64,
}

impl Stall {
    fn note(&self) -> String {
        format!("warm SCF did not converge (Δρ {:.2e} after {} steps); retried cold", self.delta_rho, self.iterations)
    }
}

/// The R15 policy (module doc), separated from the solvers so it is testable without one:
/// `solve(w)` runs one SCF solve, warm iff `w`. The first attempt runs with `warm`; if that was
/// a warm attempt and it returned `ScfNotConverged`, `solve(false)` runs exactly once and its
/// outcome is returned, with the stall when it converged. Every other outcome — success, a
/// cold failure, a non-convergence error of another kind — is the first attempt's, untouched.
fn retry_cold_on_stall<T>(warm: bool, mut solve: impl FnMut(bool) -> Result<T, qm::QmError>)
                                    -> Result<(T, Option<Stall>), qm::QmError> {
    match solve(warm) {
        Err(qm::QmError::ScfNotConverged(iterations, delta_rho)) if warm => {
            solve(false).map(|v| (v, Some(Stall { iterations, delta_rho })))
        }
        other => other.map(|v| (v, None)),
    }
}

fn elapsed_ms(t0: Instant) -> u64 {
    t0.elapsed().as_millis() as u64
}

/// The `EvalStats` fields that come straight from `ScfStats`.
fn eval_stats(stats: ScfStats, readbacks: Option<usize>, t0: Instant, notes: Vec<String>, eig_tol_used: f64) -> EvalStats {
    let ScfStats { iterations, energy, boundary_density, max_eigen_residual, poisson_residual, warnings, .. } = stats;
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
        eig_tol_used: Some(eig_tol_used),
        energy: EnergyBreakdown { kinetic: energy.kinetic, external: energy.external, hartree: energy.hartree, xc: energy.xc, nuclear: energy.nuclear, total: energy.total },
        max_eigen_residual,
        poisson_residual,
        atom_uncertainty: None,
        extrapolation: None,
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
}

impl CpuKs {
    /// `tol`: the SCF tolerance on Δρ = ∫|ρ_out − ρ_in|.
    pub fn new(tol: f64) -> CpuKs {
        CpuKs {
            settings_tol: tol, warm: None, grid: None, display: true, density_current: false, rho: None, probe: None, orb: ViewState::default(), view: None,
            last: None,
            unocc: UnoccupiedState::default(),
            interrupt: None,
        }
    }

    /// Report this solver's live state on `probe`'s `slot` (the Activity view, spec §4.3 A1):
    /// running for each solve, no SCF step count (`qm::scf` has no per-step callback).
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
            UnoccupiedPlan::Solve => match view.unoccupied(state, nuclei, m, settings, should_abort) {
                Some(_) => self.unocc.solved(m, positions.clone(), now, now.elapsed()),
                None => self.unocc.interrupted(),
            },
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
        Some(self.orb.display(cell.clone(), state.grid.origin, positions, &out, &spin, noise, pick, field, status))
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
        let (grid, system, settings) = problem(scene, self.settings_tol)?;
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
            self.grid = Some(grid.clone());
            // the orbital producer is grid-sized: rebuilt (module doc, "Orbital display")
            self.view = None;
            self.orb.rebuilt();
            self.unocc.dropped();
        }
        let prev = self.warm.take();
        // R15: a warm stall is retried once without the previous state (module doc)
        let ((state, mut stats), stall) =
            retry_cold_on_stall(prev.is_some(), |w| qm::scf(&grid, &system, &settings, if w { prev.as_ref() } else { None })).map_err(solver_error)?;
        drop(prev);
        let forces = qm::forces(&grid, &state, &system.nuclei, settings.sigma).map_err(solver_error)?;
        self.unocc.new_state(t0.elapsed());
        let display = self.display.then(|| {
            let rho: Vec<f32> = state.rho().iter().map(|&x| x as f32).collect();
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
        let mut notes = Vec::new();
        if let Some(st) = stall {
            stats.iterations += st.iterations;
            notes.push(st.note());
        }
        if stats.max_eigen_residual >= settings.eig_tol {
            notes.push(format!("LOBPCG stopped at its iteration cap at least once (max residual {:.2e} ≥ tol {:.0e})",
                               stats.max_eigen_residual, settings.eig_tol));
        }
        let energy = stats.energy.total;
        Ok(ForceResult { energy, forces, stats: eval_stats(stats, None, t0, notes, settings.eig_tol), display, orbitals })
    }

    fn name(&self) -> &'static str {
        "Kohn-Sham (CPU)"
    }

    fn reset(&mut self) {
        self.warm = None;
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
    /// The driver's display interrupt (`crate::orbitals`, "Cancellation").
    interrupt: Option<DisplayInterrupt>,
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
            probe: None,
            orb: ViewState::default(),
            orbital: None,
            orb_scalar: Pool::default(),
            orb_colour: Pool::default(),
            last: None,
            unocc: UnoccupiedState::default(),
            interrupt: None,
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
        let GpuKs { solver, ctx, tol, orb, orbital, orb_scalar, orb_colour, last, vram_budget, unocc, .. } = self;
        let solver = solver.as_mut()?;
        let (cell, nuclei, settings) = last.as_ref()?;
        let positions: Vec<[f64; 3]> = nuclei.iter().map(|n| n.pos).collect();
        let (n, n_up, n_dn, origin) = {
            let st = solver.state()?;
            (st.grid.n[0], st.orbitals_up.len(), st.orbitals_dn.len(), st.grid.origin)
        };
        // `requested` is the app's (the driver matches it); the solve is capped at UNOCCUPIED_MAX
        let requested = orb.request().unoccupied;
        let mut m = requested.min(UNOCCUPIED_MAX) as usize;
        // the GPU memory guard (module doc, "GPU memory"): the view, not the scene, is refused
        let mut guard = Vec::new();
        if let Some(budget) = *vram_budget {
            let base = solve_footprint_bytes(n, (n_up + n_dn) as u64, n_up.max(n_dn) as u64);
            let need = |m: usize| base + orbital_view_footprint_bytes(n, n_up, n_dn, m, orb.texture());
            let gb = |b: u64| b as f64 / 1e9;
            if need(m) > budget {
                if m > 0 && need(0) <= budget {
                    guard.push(format!(
                        "{requested} unoccupied orbitals refused: with them this scene needs ≈ {:.2} GB of GPU memory at {n}³, over the {:.1} GB budget — fewer unoccupied orbitals or a coarser quality (Scene ▸ Quality) fits",
                        gb(need(m)), gb(budget)));
                    m = 0;
                } else {
                    if let Some(v) = orbital.take() {
                        v.release(solver);
                    }
                    orb_scalar.clear();
                    orb_colour.clear();
                    unocc.dropped();
                    return Some(orb.refused(cell.clone(), origin, positions, format!(
                        "the orbital view needs ≈ {:.2} GB of GPU memory at {n}³ with this scene, over the {:.1} GB budget: choose a coarser quality (Scene ▸ Quality)",
                        gb(need(0)), gb(budget))));
                }
            }
        }
        let view = orbital.get_or_insert_with(|| {
            let mut v = OrbitalView::new(solver);
            v.set_face_limit(box_affected_face_density(solver.grid().h));
            v
        });
        // unoccupied orbitals (`crate::orbitals`, "Unoccupied orbitals")
        let now = Instant::now();
        unocc.revalidate(&positions);
        match unocc.plan(m, now, orb.request().force_every_step) {
            UnoccupiedPlan::Solve => match view.unoccupied(solver, nuclei, m, settings, should_abort) {
                Some(_) => unocc.solved(m, positions.clone(), now, now.elapsed()),
                None => unocc.interrupted(),
            },
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
            view.update_with(solver, state, nuclei, *tol, move |out| {
                let p = vs.pick(out);
                let product = match (&p.kind, sv, cv) {
                    (&PickKind::Orbital { spin, track, squared }, Some(v), _) => Product::Orbital { spin, track, squared, out: v },
                    (PickKind::Spin, Some(v), _) => Product::Spin { out: v },
                    (PickKind::Colour { include, physical }, _, Some(c)) => {
                        inc.extend_from_slice(include);
                        Product::Colour { include: inc.as_slice(), physical: *physical, out: c }
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
        // the eigensolver tolerance the converged step ran at: the effective one (grid-aware)
        let noise = u.spin.noise_l1(settings.tol, solver.profile().eig_tol_used);
        Some(orb.display(cell.clone(), state.grid.origin, positions, &u.out, &u.spin, noise, pick, field, status))
    }

    /// With a GPU memory budget (bytes): a scene over it is refused before anything is allocated
    /// (module doc, "GPU memory").
    pub fn with_vram_budget(mut self, bytes: u64) -> GpuKs {
        self.vram_budget = Some(bytes);
        self
    }

    /// The budget check of `evaluate` (module doc, "GPU memory").
    fn check_vram(&self, scene: &Scene) -> Result<(), ProviderError> {
        let limits = self.ctx.device.limits();
        let field = field_bytes(scene.cell.n);
        let max_field = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
        if field > max_field {
            return Err(ProviderError::Invalid(format!(
                "one {}³ field is {:.0} MB, over this GPU's {:.0} MB buffer limit: choose a coarser quality (Scene ▸ Quality)",
                scene.cell.n,
                field as f64 / 1e6,
                max_field as f64 / 1e6
            )));
        }
        if let Some(budget) = self.vram_budget {
            let need = gpu_footprint_bytes(scene);
            if need > budget {
                return Err(ProviderError::Invalid(format!(
                    "this scene needs ≈ {:.1} GB of GPU memory at {}³ ({} electrons), over the {:.1} GB budget: choose a coarser quality (Scene ▸ Quality) or fewer atoms",
                    need as f64 / 1e9,
                    scene.cell.n,
                    scene.electrons().max(0),
                    budget as f64 / 1e9
                )));
            }
        }
        Ok(())
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
    /// One attempt of [`ForceProvider::evaluate`], with no "GPU busy" retry.
    fn evaluate_once(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        // whatever happens below, the held density is no longer this scene's until it succeeds
        self.density_current = false;
        if let Some(r) = nothing_to_solve(scene) {
            return r;
        }
        let (grid, system, settings) = problem(scene, self.tol)?;
        self.check_vram(scene)?;
        if should_abort() {
            return Err(ProviderError::Cancelled);
        }
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
        if self.solver.as_ref().is_none_or(|s| *s.grid() != grid) {
            self.solver = None; // free the old grid's buffers before allocating the new ones
            // the orbital producer belongs to the old solver: rebuilt (module doc, "Orbital display")
            self.orbital = None;
            self.orb.rebuilt();
            self.unocc.dropped();
            let mut s = qm_gpu::GpuSolver::new(self.ctx.clone(), &grid);
            s.set_density_extrapolation(self.extrapolate);
            self.solver = Some(s);
        }
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
            solver.solve_with(&system, &settings, w, &mut polled).map(|r| r.map(|(_, stats)| stats))
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
        self.unocc.new_state(t0.elapsed());
        let display = self.display.then(|| {
            let volume = self.textures.take(&self.ctx, grid.n.map(|n| n as u32));
            // queue order makes the copy visible to any later render submission; the max
            // reduction below also waits for the GPU
            solver.copy_density_to_volume(state, &volume);
            let max = solver.density_max(state);
            Display::Gpu { cell: scene.cell.clone(), origin: grid.origin, volume, max }
        });
        device_fault()?; // the density maximum is a readback: zeros would normalise the texture wrongly
        let profile = solver.profile();
        let mut notes = profile.notes.clone();
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
        Ok(ForceResult { energy, forces, stats: eval_stats(stats, Some(readbacks), t0, notes, eig_tol_used), display, orbitals })
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
        self.density_current = false;
        self.drop_extrapolation();
        self.drop_orbital_view();
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

    /// A second `GpuKs` on the same context at the same tolerance, display off — its own
    /// `GpuSolver` and warm state, like the surrogate's background auditor (the references'
    /// solver).
    fn side_solver(&self) -> Option<Box<dyn ForceProvider>> {
        let mut g = GpuKs::new(self.ctx.clone(), self.tol);
        g.vram_budget = self.vram_budget;
        g.set_display(false);
        Some(Box::new(g))
    }

    fn set_orbital_view(&mut self, request: OrbitalViewRequest) {
        if self.orb.set(request) == ViewChange::Off {
            self.drop_orbital_view();
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim::{Cell, Nucleus};
    use gpu::GpuContext;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    /// Serialises the GPU sections of concurrently running tests (poisoning is harmless here).
    static GPU: Mutex<()> = Mutex::new(());

    fn gpu() -> Option<GpuContext> {
        let ctx = GpuContext::new_headless();
        if ctx.is_none() { eprintln!("providers: no GPU adapter — skipped"); }
        ctx
    }

    fn scene(n: usize, nuclei: Vec<Nucleus>) -> Scene {
        Scene { nuclei, cell: Cell { n, h: 0.25 }, charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3] }
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
        assert_eq!(stall, Some(Stall { iterations: 200, delta_rho: 1.2e-4 }));
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
        // the last two land in the forces and the display maximum after the SCF
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
                // in the forces or the display maximum, after a converged SCF: the solver keeps
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
        let w = gpu_footprint_bytes(&h2o(97)) as f64 / 1e6;
        assert!((600.0..1400.0).contains(&w), "{w} MB");
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

    /// The CPU adapter is `qm::scf` + `qm::forces` and nothing else: bitwise.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn cpu_provider_matches_qm_directly() {
        let s = h2(33, 1.4);
        let r = CpuKs::new(1e-6).evaluate(&s, &mut never()).expect("CpuKs");

        let grid = qm::Grid::cube(33, 0.25);
        let sys = qm::System::new(vec![qm::Nucleus { pos: [0.7, 0.0, 0.0], z: 1 }, qm::Nucleus { pos: [-0.7, 0.0, 0.0], z: 1 }], 0, None);
        let mut settings = qm::ScfSettings::for_grid(&grid);
        settings.tol = 1e-6;
        let (state, stats) = qm::scf(&grid, &sys, &settings, None).expect("qm::scf");
        let f = qm::forces(&grid, &state, &sys.nuclei, settings.sigma).expect("qm::forces");
        println!("H₂ 33³: E = {:+.12} Ha ({} steps), F = {:?}", r.energy, r.stats.scf_steps, r.forces);
        assert_eq!(r.energy.to_bits(), stats.energy.total.to_bits(), "{} vs {}", r.energy, stats.energy.total);
        assert_eq!(r.forces, f);
        assert_eq!(r.stats.scf_steps, stats.iterations);
        assert_eq!(r.stats.readbacks, None);
        assert_eq!(r.stats.warnings, stats.warnings);
        let (a, b) = (r.stats.energy, stats.energy);
        for (x, y) in [(a.kinetic, b.kinetic), (a.external, b.external), (a.hartree, b.hartree), (a.xc, b.xc), (a.nuclear, b.nuclear), (a.total, b.total)] {
            assert_eq!(x.to_bits(), y.to_bits(), "energy breakdown {a:?} vs {b:?}");
        }
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
        let h = solve_atom(&mut c, &cell, None, 1, Nucleus::new(1, [0.0; 3]).mass, &mut never()).expect("H reference");
        println!(
            "H reference ({:.1} s): E = {:.6} Ha, multiplicity {:?}, {} shell entries to r = {:.3}, {:.4} electrons in the table; ρ̄(0) = {:.4}, ρ̄(0.8) = {:.4}, ρ̄(1.6) = {:.4}",
            t0.elapsed().as_secs_f64(), h.energy, h.multiplicity, h.radial.r.len(), h.radial.r.last().unwrap(), h.radial.electrons(), h.radial.at(0.0), h.radial.at(0.8), h.radial.at(1.6)
        );
        let name = c.name();
        let key = |z: u8| ReferenceKey::new(&cell, None, Some(1e-5), name, z);
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
        let mut n = 0;
        let mut abort_third = || {
            n += 1;
            n > 3
        };
        assert!(matches!(p.evaluate(&scene, &mut abort_third), Err(ProviderError::Cancelled)));
        let v = probe.view();
        assert!(!v.main.running);
        // polls 2–4 are inside `solve_with` (poll 1 is `evaluate`'s pre-solve check): steps 1
        // and 2 ran and the poll before step 3 aborted. That poll is counted, so the idle count
        // reads the step being solved when it stopped.
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
                        2 => OrbitalRequest::AllColoured { include: None, physical: false },
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
    /// - H₂O 97³, m = 4: one paused eigensolve, its wall printed against the SCF's; three play
    ///   evaluations after it run none and the status says paused-only while the budget is short.
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
        // H₂O at 97³: the measured cost, paused-only by the budget
        let mut g = GpuKs::new(ctx, 1e-5);
        g.set_display(false);
        g.set_orbital_view(with_unoccupied(OrbitalRequest::List, 4, true));
        let mut s = h2o(97);
        let r = g.evaluate(&s, &mut never()).expect("H₂O 97³");
        let u = r.orbitals.as_ref().and_then(|o| o.unoccupied.clone()).expect("a paused payload with m = 4");
        let scf = r.stats.solve_ms as f64 / 1e3 - u.wall_s;
        println!("H₂O 97³ m = 4, paused: eigensolve {:.2} s vs the SCF's {scf:.2} s (cold) — a play solve needs {:.0} s of SCF wall first; shown {:?}, notes {:?}",
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
        assert_eq!(solve_footprint_bytes(49, 2, 1), gpu_footprint_bytes(&s), "one formula");
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
}
