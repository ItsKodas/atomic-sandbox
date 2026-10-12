//! Spin-polarized Kohn-Sham SCF on the GPU — the port of `qm::scf` (read its module doc first:
//! the total-energy functional and why it is evaluated from the *output* density with a fresh
//! Poisson solve, the Pulay/DIIS derivation with the two spin channels stacked and the
//! charge-conservation argument, the electron-count/multiplicity rules, the warm-start rules
//! and the box-size warning). The algorithm is step for step the CPU's:
//!   V_eff,σ = V_nuc + V_H[ρ_in] + v_xc,σ[ρ_in↑, ρ_in↓]  →  lowest n_σ eigenpairs of H_σ per spin
//!   →  ρ_out,σ = Σ_i ψ_{iσ}²,  Δρ = ∫(|ρ_out↑ − ρ_in↑| + |ρ_out↓ − ρ_in↓|)  →  DIIS mix,
//! and the energy E = T_s + E_ext + E_H + E_xc + E_nn from ρ_out and the orbitals. This doc
//! covers only what the GPU changes.
//!
//! # What stays f64 (CLAUDE.md rule 4)
//! Every reduction finish (`Reducer`: f32 workgroup partials → Neumaier f64), the DIIS Gram
//! matrix and its bordered LU solve, the Rayleigh–Ritz inside LOBPCG, the CG scalars inside the
//! Poisson solver, the eigenvalues, E_nn (`qm::nuclear::nuclear_repulsion`) and the final sum of
//! the energy components. Everything on the grid — orbitals, densities, potentials, residuals,
//! the DIIS history — is f32 in padded `Field`s; the cold-start guess (`deposit_charge` at
//! σ = 1, `qm::scf::initial_orbitals`) is built in f64 on the CPU and uploaded once.
//!
//! # The three f32 floors and how the loop lives with them
//! - **Poisson** (`GpuPoisson`): the true residual floors at ~6e-6 (33³) … 2e-5 (97³) relative.
//!   Every solve is warm-started from the previous V_H and asked for `POISSON_TOL` = 1e-8, i.e.
//!   *below* the floor: the CG recurrence runs until it reaches 1e-8 or the floor signature is
//!   detected, and a stop at the floor (`Stop::Floor`/`Stagnated`) is "use V" — only `MaxIter`
//!   warns. Stopping the recurrence at 1e-6 instead (the first choice) leaves a *smooth* error
//!   of that relative size in V_H that differs from one warm restart to the next, which showed
//!   up as a Δρ stall at 2–5e-6 for H₂ (19 steps vs the CPU's 10); at 1e-8 the leftover is the
//!   floor's high-frequency rounding drift, which the density barely sees (10 steps, +15% CG
//!   iterations, equal wall time; measured in `diag_trajectory_and_knobs`). E_H of a unit
//!   charge at the floor is good to ~1e-6 Ha (Task 4).
//! - **LOBPCG** (`eigen`): the residual of an f32-stored vector floors at
//!   `residual_floor_estimate(g)` ≈ 1.8e-6 at h = 0.25. The eigensolver is called with
//!   `tol = max(settings.eig_tol, 3 × floor)`, so the default `eig_tol` = 1e-6 does not cost a
//!   12-iteration stall window per call (factor 1 was measured: H₂ 11 steps, Li 26 vs 17, and
//!   1.3–1.7× the wall time); `max_eigen_residual` reports the true residual, and a `Floor`
//!   exit above tol is *informational* (ruled): counted in `SolveProfile::eigen_floor_accepts`
//!   with a `notes` line, never a warning and never a failure. `ScfStats::warnings` carries only
//!   actionable conditions (box too small, Poisson or LOBPCG at its iteration cap). Eigenvalues
//!   and the density are second order in the residual.
//! - **The mixing step** is where f32 would hurt most: a plain `lincomb` Σ cᵢ(ρᵢ + α rᵢ) with
//!   coefficients of both signs rounds at |c|·ρ size and leaves ~3e-7·|c|·N of noise in Δρ, in
//!   competition with tol = 1e-6 for N ≥ 2. The `diis_mix` kernel evaluates the same
//!   combination about the latest input density, ρ_m + Σ_{i<m} cᵢ(ρᵢ − ρ_m) + αΣ cᵢ rᵢ
//!   (identical since Σ cᵢ = 1): the differences are exact (Sterbenz) and the small terms are
//!   summed before one ρ-sized rounding, which puts the floor at the storage rounding of ρ_in
//!   itself (~3e-8·N). The negative clamp is folded into the same dispatch.
//! - **Charge renormalisation.** The clamp changes ∫ρ_in by a tiny amount (as on the CPU, where
//!   it is left alone — the M1 deferred minor); here each channel is rescaled afterwards (one
//!   readback of two sums) to ∫ρ_out,σ of the step just taken, *not* to n_σ: the f32-normalised
//!   orbitals integrate to n_σ(1 + ε) with ε ≈ 1e-7 (measured −1.0e-7 for H₂), and a fixed point
//!   whose input charge is n_σ while its output charge is n_σ(1 + ε) has Δρ ≥ N|ε| — ~1e-6 for
//!   H₂O against tol 1e-6. Renormalising to ∫ρ_out keeps the fixed point charge-consistent and
//!   only repairs the clamp's loss.
//!
//! # Readbacks per SCF step (the sync points)
//! Poisson: 3 + 2·(CG iterations) + checks + 1 (warm from the previous V_H: a few iterations
//! after the first step); LOBPCG per non-empty channel: k + 2 for the guess MGS/λ and 2k + 1
//! per iteration; then exactly two for the SCF itself — Δρ together with the new DIIS Gram row
//! (one `Reducer::many_in`), and the two channel charges for the renormalisation. The energy
//! after convergence is one Poisson solve plus a single readback (T_s per orbital, E_ext, E_H,
//! E_xc, the face maximum). Everything between sync points is recorded into one `Batch`
//! (`Poisson` and `lobpcg` run their own batches; the SCF batch is flushed before each).
//!
//! # Determinism
//! Two identical cold solves give bit-identical energies: every reduction has a fixed
//! workgroup count, per-thread order and tree shape (`reduce`), every pointwise kernel writes
//! its own element, LOBPCG's re-seeding and the DIIS LU are deterministic, and no per-op state
//! depends on timing. `solve_is_deterministic` pins it (same solver twice, and a fresh solver).
//!
//! # Fields and VRAM
//! The solver owns a pool of padded fields that grows to the peak concurrent need and is
//! reused across solves (nothing is allocated per step). Per solve: K = n↑ + n↓ orbitals, 3
//! densities, V_H, V_nuc in the returned `GpuState`; ~8 working fields; 4 per DIIS history
//! entry (24 at the default history 6); K kinetic scratch fields for the energy. Plus one
//! `GpuPoisson` (54 fields, 227 MB at 97³) and one `LobpcgWork` (8k + 2 fields) per solver.
//! At 97³ with K = 8 that is ~700 MB. A warm solve reuses the previous state's fields in place.
use crate::batch::Batch;
use crate::eigen::{lobpcg, residual_floor_estimate, EigenStats, GpuHamiltonian, LobpcgWork, Stop as EigenStop};
use crate::grid::{Field, GpuGrid};
use crate::kernels::{encoder, Kernels};
use crate::physics::{nuclear_potential_in, xc_lsda_in};
use crate::pointwise::{diis_mix_in, fill_in, lincomb_in, mul_add_sq_in, scale_in, unpad_rec, unpad_row_stride, DiisTerm};
use crate::poisson::{GpuPoisson, PoissonStats, Stop as PoissonStop};
use crate::reduce::{neumaier, Reducer, Reduction};
use crate::stencil::kinetic_in;
use gpu::{GpuContext, Volume3D};
use nalgebra::{DMatrix, DVector};
use qm::nuclear::{deposit_charge, nuclear_repulsion};
use qm::scf::initial_orbitals;
use qm::{Energy, QmError, ScfSettings, ScfStats, System};
use std::time::{Duration, Instant};

/// Relative residual requested from every Poisson solve — below the f32 floor on purpose (module
/// doc), so each solve runs to the floor and the floor is accepted — and its iteration cap.
pub const POISSON_TOL: f64 = 1e-8;
const POISSON_MAX_ITER: usize = 5000;
/// LOBPCG Rayleigh–Ritz steps per call (as on the CPU).
const EIGEN_MAX_ITER: usize = 200;
/// LOBPCG tolerance = max(settings.eig_tol, this × `residual_floor_estimate`).
const EIG_TOL_FLOOR_FACTOR: f64 = 3.0;
/// Density on a box face above this (e/Bohr³) triggers the box-size warning (`qm::scf`).
const BOUNDARY_DENSITY_WARN: f64 = 1e-5;
/// Relative diagonal regularisation of the (scaled) residual Gram matrix (`qm::scf`).
const DIIS_REG: f64 = 1e-12;

/// Converged Kohn-Sham state on the GPU: orbitals, eigenvalues, output densities (`rho` =
/// `rho_up` + `rho_dn`) and the potentials (`v_h` is V_H of `rho`), all padded f32 fields.
/// `grid` and `sigma` record what it was converged on. Owned by the `GpuSolver` that made it
/// (it is the warm start of the next solve); borrow it through `solve`.
pub struct GpuState {
    pub grid: qm::Grid, pub sigma: f64,
    pub orbitals_up: Vec<Field>, pub orbitals_dn: Vec<Field>,
    pub evals_up: Vec<f64>, pub evals_dn: Vec<f64>,
    pub rho: Field, pub rho_up: Field, pub rho_dn: Field,
    pub v_h: Field, pub v_nuc: Field,
}

/// Cost accounting and informational diagnostics of the last `solve`: SCF steps, GPU→CPU
/// readbacks by origin, inner iteration totals, wall time (reported, not asserted —
/// hardware-dependent), and the LOBPCG floor bookkeeping that is *not* a warning: how many
/// eigensolves ended on the f32 floor above tol (`eigen_floor_accepts`, worst residual
/// `eigen_floor_worst`), the max residual of the final SCF step's eigensolves
/// (`eigen_last_step_residual` — `ScfStats::max_eigen_residual` is the worst over all steps and
/// cannot say whether the last one converged), and `notes`, informational lines.
#[derive(Debug, Clone, Default)]
pub struct SolveProfile {
    pub steps: usize,
    pub readbacks: usize,
    pub poisson_readbacks: usize,
    pub eigen_readbacks: usize,
    pub poisson_iterations: usize,
    pub eigen_iterations: usize,
    pub wall: Duration,
    pub eigen_floor_accepts: usize,
    pub eigen_floor_worst: f64,
    pub eigen_last_step_residual: f64,
    pub notes: Vec<String>,
}

/// A pool of padded fields reused across solves (never shrinks). Every consumer overwrites the
/// interior of what it takes; ghosts are zero by the crate invariant.
struct Pool { free: Vec<Field> }

impl Pool {
    fn take(&mut self, ctx: &GpuContext, g: &GpuGrid) -> Field {
        self.free.pop().unwrap_or_else(|| Field::new(ctx, g, "scf field"))
    }
    fn give(&mut self, f: Field) { self.free.push(f); }
    fn give_state(&mut self, s: GpuState) {
        let GpuState { orbitals_up, orbitals_dn, rho, rho_up, rho_dn, v_h, v_nuc, .. } = s;
        self.free.extend(orbitals_up.into_iter().chain(orbitals_dn).chain([rho, rho_up, rho_dn, v_h, v_nuc]));
    }
}

/// One Pulay history entry (input densities and residuals per spin) as fields.
struct DiisEntry { rho_up: Field, rho_dn: Field, r_up: Field, r_dn: Field }

/// The DIIS history with its Gram matrix B_ij = ⟨r_i↑, r_j↑⟩ + ⟨r_i↓, r_j↓⟩ kept on the CPU and
/// grown by one row per step (the dots come with the Δρ readback), shrunk when an entry drops.
struct Diis { entries: Vec<DiisEntry>, gram: Vec<Vec<f64>>, cap: usize }

impl Diis {
    fn new(cap: usize) -> Diis { Diis { entries: Vec::new(), gram: Vec::new(), cap: cap.max(1) } }

    /// Appends `e` whose Gram row `row` = [⟨r_e, r_0⟩, …, ⟨r_e, r_{m−1}⟩, ⟨r_e, r_e⟩]; drops the
    /// oldest entry when over capacity.
    fn push(&mut self, e: DiisEntry, row: Vec<f64>, pool: &mut Pool) {
        let m = self.entries.len();
        assert_eq!(row.len(), m + 1);
        for (i, r) in self.gram.iter_mut().enumerate() { r.push(row[i]); }
        self.gram.push(row);
        self.entries.push(e);
        if self.entries.len() > self.cap { self.drop_oldest(pool); }
    }

    fn drop_oldest(&mut self, pool: &mut Pool) {
        let e = self.entries.remove(0);
        pool.give(e.rho_up); pool.give(e.rho_dn); pool.give(e.r_up); pool.give(e.r_dn);
        self.gram.remove(0);
        for r in &mut self.gram { r.remove(0); }
    }

    fn clear(&mut self, pool: &mut Pool) { while !self.entries.is_empty() { self.drop_oldest(pool); } }

    /// Pulay coefficients (`qm::scf::diis_coefficients`): bordered system on the Gram matrix
    /// scaled by its largest diagonal, +1e-12 relative on the diagonal, LU. `None` on failure.
    fn coefficients(&self) -> Option<Vec<f64>> {
        let m = self.entries.len();
        let mut b = DMatrix::<f64>::zeros(m + 1, m + 1);
        for i in 0..m { for j in 0..m { b[(i, j)] = self.gram[i][j]; } }
        let scale = (0..m).map(|i| b[(i, i)]).fold(0.0f64, f64::max);
        if scale.is_nan() || scale <= 0.0 { return None; }
        for i in 0..m { for j in 0..m { b[(i, j)] /= scale; } b[(i, i)] += DIIS_REG; }
        for i in 0..m { b[(i, m)] = 1.0; b[(m, i)] = 1.0; }
        let mut rhs = DVector::<f64>::zeros(m + 1);
        rhs[m] = 1.0;
        let x = b.lu().solve(&rhs)?;
        let c: Vec<f64> = x.iter().take(m).copied().collect();
        if c.iter().all(|v| v.is_finite()) { Some(c) } else { None }
    }

    /// ρ_in,new,σ = Σ cᵢ (ρ_in,i,σ + α r_i,σ) clamped at 0, evaluated about the latest entry
    /// (module doc). Drops the oldest entries while the coefficient solve fails; one entry is
    /// linear mixing.
    #[allow(clippy::too_many_arguments)]
    fn mix(&mut self, b: &mut Batch, k: &Kernels, g: &GpuGrid, alpha: f64, out_up: &Field, out_dn: &Field, pool: &mut Pool) {
        let c = loop {
            if self.entries.len() == 1 { break vec![1.0]; }
            match self.coefficients() {
                Some(c) => break c,
                None => self.drop_oldest(pool),
            }
        };
        let m = self.entries.len();
        let base = &self.entries[m - 1];
        let terms = |rho: &dyn Fn(&DiisEntry) -> &Field, r: &dyn Fn(&DiisEntry) -> &Field| -> Vec<DiisTerm> {
            (0..m - 1).map(|i| (c[i] as f32, rho(&self.entries[i]), true))
                .chain((0..m).map(|i| ((alpha * c[i]) as f32, r(&self.entries[i]), false)))
                .collect()
        };
        diis_mix_in(b, k, g, out_up, &base.rho_up, &terms(&|e| &e.rho_up, &|e| &e.r_up), true);
        diis_mix_in(b, k, g, out_dn, &base.rho_dn, &terms(&|e| &e.rho_dn, &|e| &e.r_dn), true);
    }
}

/// NaN-propagating max (a non-finite residual must not hide behind `f64::max`).
fn nan_max(a: f64, b: f64) -> f64 { if a.is_nan() || b.is_nan() { f64::NAN } else { a.max(b) } }

/// Worst-case inner-solver diagnostics over a solve (`qm::scf::InnerStats` with the GPU's stop
/// semantics). `warnings` hold actionable conditions only: Poisson `MaxIter`, LOBPCG `MaxIter`
/// (summarised). A LOBPCG `Floor` exit above tol is informational (ruled) and goes to
/// `SolveProfile::{eigen_floor_accepts, eigen_floor_worst, notes}`, never to `warnings`.
struct Inner {
    eigen_residual: f64, poisson_residual: f64, poisson_iterations: usize, warnings: Vec<String>,
    eigen_calls: usize, eigen_max_iter: usize, eigen_max_iter_worst: f64, step_residual: f64, eig_tol: f64,
    profile: SolveProfile,
}

impl Inner {
    fn new(eig_tol: f64) -> Inner {
        Inner { eigen_residual: 0.0, poisson_residual: 0.0, poisson_iterations: 0, warnings: Vec::new(),
                eigen_calls: 0, eigen_max_iter: 0, eigen_max_iter_worst: 0.0, step_residual: 0.0, eig_tol, profile: SolveProfile::default() }
    }

    /// Call at the start of each SCF step: resets the per-step eigen residual.
    fn begin_step(&mut self) { self.step_residual = 0.0; }

    fn eigen(&mut self, s: EigenStats) {
        self.eigen_calls += 1;
        self.eigen_residual = nan_max(self.eigen_residual, s.max_residual);
        self.step_residual = nan_max(self.step_residual, s.max_residual);
        self.profile.eigen_last_step_residual = self.step_residual;
        self.profile.eigen_readbacks += s.readbacks;
        self.profile.eigen_iterations += s.iterations;
        match s.stop {
            EigenStop::Floor if s.max_residual > self.eig_tol => {
                self.profile.eigen_floor_accepts += 1;
                self.profile.eigen_floor_worst = self.profile.eigen_floor_worst.max(s.max_residual);
            }
            EigenStop::MaxIter => {
                self.eigen_max_iter += 1;
                self.eigen_max_iter_worst = self.eigen_max_iter_worst.max(s.max_residual);
            }
            _ => {}
        }
    }

    fn poisson(&mut self, s: PoissonStats, what: &str) {
        self.poisson_residual = nan_max(self.poisson_residual, s.residual);
        self.poisson_iterations = self.poisson_iterations.max(s.iterations);
        self.profile.poisson_readbacks += s.readbacks;
        self.profile.poisson_iterations += s.iterations;
        if s.stop == PoissonStop::MaxIter {
            self.warnings.push(format!("{what}: Poisson stopped after {} iterations at relative residual {:.2e} > {POISSON_TOL:.0e}", s.iterations, s.residual));
        }
    }

    fn finish(mut self) -> (Vec<String>, SolveProfile) {
        if self.profile.eigen_floor_accepts > 0 {
            self.profile.notes.push(format!("LOBPCG accepted at the f32 floor in {} of {} calls (worst residual {:.2e} > tol {:.2e})",
                                            self.profile.eigen_floor_accepts, self.eigen_calls, self.profile.eigen_floor_worst, self.eig_tol));
        }
        if self.eigen_max_iter > 0 {
            self.warnings.push(format!("LOBPCG stopped at its {EIGEN_MAX_ITER}-iteration cap in {} of {} calls (worst residual {:.2e} > tol {:.2e})",
                                       self.eigen_max_iter, self.eigen_calls, self.eigen_max_iter_worst, self.eig_tol));
        }
        (self.warnings, self.profile)
    }
}

/// Submits everything recorded so far (before handing the queue to a sub-solver with its own batch).
fn flush(b: &mut Batch) {
    let enc = b.take_encoder();
    b.ctx().queue.submit([enc.finish()]);
}

/// The Kohn-Sham solver for one grid: kernels, reducer, Poisson solver, LOBPCG work space, a
/// field pool and the last converged state (the warm start). `Send` (one per worker thread).
pub struct GpuSolver {
    ctx: GpuContext, k: Kernels, r: Reducer, g: GpuGrid, grid: qm::Grid,
    poisson: GpuPoisson,
    work: Option<LobpcgWork>, work_k: usize,
    pool: Pool,
    warm: Option<GpuState>,
    /// Row-padded staging buffer for `copy_density_to_volume` and its row stride in floats.
    rows: wgpu::Buffer, row_stride: u32,
    profile: SolveProfile,
    /// `POISSON_TOL` and `EIG_TOL_FLOOR_FACTOR` as instance knobs (tests probe them).
    poisson_tol: f64, eig_floor_factor: f64,
}

impl GpuSolver {
    /// Compiles the kernels and builds the Poisson solver (49 R_lm fields + ghost table) for `grid`.
    pub fn new(ctx: GpuContext, grid: &qm::Grid) -> GpuSolver {
        let g = GpuGrid::new(grid);
        let k = Kernels::new(&ctx);
        let r = Reducer::new(&ctx, &k, &g);
        let poisson = GpuPoisson::new(&ctx, &k, &g);
        let row_stride = unpad_row_stride(g.n);
        let rows = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("density rows"), size: row_stride as u64 * (g.n * g.n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
        });
        GpuSolver { ctx, k, r, g, grid: grid.clone(), poisson, work: None, work_k: 0, pool: Pool { free: Vec::new() },
                    warm: None, rows, row_stride, profile: SolveProfile::default(),
                    poisson_tol: POISSON_TOL, eig_floor_factor: EIG_TOL_FLOOR_FACTOR }
    }

    /// Diagnostic knobs: the Poisson tolerance and the LOBPCG tolerance floor factor.
    #[cfg(test)]
    fn set_knobs(&mut self, poisson_tol: f64, eig_floor_factor: f64) { self.poisson_tol = poisson_tol; self.eig_floor_factor = eig_floor_factor; }

    pub fn grid(&self) -> &qm::Grid { &self.grid }

    /// The last converged state, if any (what `warm = true` would start from).
    pub fn state(&self) -> Option<&GpuState> { self.warm.as_ref() }

    /// Cost accounting of the last `solve`.
    pub fn profile(&self) -> &SolveProfile { &self.profile }

    /// Same contract as `qm::scf::scf` on this solver's grid: `Ok((state, stats))` or
    /// `Err(QmError)` (coincident nuclei; no convergence within `settings.max_iter`, including a
    /// non-finite Δρ). `warm` reuses the previous converged state's orbitals, densities and V_H
    /// when its orbital counts match `system` (a different σ or geometry is fine); otherwise the
    /// cold path runs. After an error the solver holds no state (the next solve is cold).
    /// The returned `&GpuState` borrows the solver mutably (it lives inside it); to use it with
    /// `density_f32`/`copy_density_to_volume`/`density_max`, re-borrow it through `state()`.
    pub fn solve(&mut self, system: &System, settings: &ScfSettings, warm: bool) -> Result<(&GpuState, ScfStats), QmError> {
        let stats = self.run(system, settings, warm)?;
        Ok((self.warm.as_ref().expect("run stores the converged state"), stats))
    }

    fn run(&mut self, system: &System, settings: &ScfSettings, warm: bool) -> Result<ScfStats, QmError> {
        let t0 = Instant::now();
        let e_nn = nuclear_repulsion(&system.nuclei)?;
        let (n_up, n_dn) = (system.n_up, system.n_dn);
        let GpuSolver { ctx, k, r, g, grid, poisson, work, work_k, pool, warm: slot, profile, poisson_tol, eig_floor_factor, .. } = self;
        let (poisson_tol, eig_floor_factor) = (*poisson_tol, *eig_floor_factor);
        let (ctx, k, r, g): (&GpuContext, &Kernels, &Reducer, &GpuGrid) = (ctx, k, r, g);
        let kmax = n_up.max(n_dn);
        if kmax > 0 && (work.is_none() || *work_k < kmax) {
            *work = Some(LobpcgWork::new(ctx, g, kmax));
            *work_k = kmax;
        }
        let eig_tol = settings.eig_tol.max(eig_floor_factor * residual_floor_estimate(g));
        let mut inner = Inner::new(eig_tol);
        let mut batch = Batch::new(ctx, "scf");

        // --- starting point ---
        let prev = slot.take();
        let reuse = warm && prev.as_ref().is_some_and(|s| s.grid == *grid && s.orbitals_up.len() == n_up && s.orbitals_dn.len() == n_dn);
        let (mut orb_up, mut orb_dn, mut ev_up, mut ev_dn, mut rho_in_up, mut rho_in_dn, v_h, v_nuc) = match prev {
            Some(s) if reuse => {
                let GpuState { orbitals_up, orbitals_dn, evals_up, evals_dn, rho, rho_up, rho_dn, v_h, v_nuc, .. } = s;
                pool.give(rho);
                (orbitals_up, orbitals_dn, evals_up, evals_dn, rho_up, rho_dn, v_h, v_nuc)
            }
            other => {
                if let Some(s) = other { pool.give_state(s); }
                let total = (n_up + n_dn) as f64;
                let z_tot: f64 = system.nuclei.iter().map(|q| q.z as f64).sum();
                let guess = deposit_charge(grid, &system.nuclei, 1.0); // broad atomic-like blobs
                let scale = if z_tot > 0.0 { total / z_tot } else { 0.0 };
                let fu = if total > 0.0 { n_up as f64 / total } else { 0.0 };
                let fd = if total > 0.0 { n_dn as f64 / total } else { 0.0 };
                let upload = |pool: &mut Pool, data: &[f64]| { let f = pool.take(ctx, g); f.upload_interior(ctx, g, data); f };
                let orbitals = |pool: &mut Pool, count: usize, seed: u64| -> Vec<Field> {
                    initial_orbitals(grid, &system.nuclei, count, seed).iter().map(|d| upload(pool, d)).collect()
                };
                let orb_up = orbitals(pool, n_up, 1);
                let orb_dn = orbitals(pool, n_dn, 2);
                let rho_up = upload(pool, &guess.iter().map(|q| q * scale * fu).collect::<Vec<_>>());
                let rho_dn = upload(pool, &guess.iter().map(|q| q * scale * fd).collect::<Vec<_>>());
                let v_h = pool.take(ctx, g);
                fill_in(&mut batch, k, g, &v_h, 0.0); // cold CG start
                (orb_up, orb_dn, vec![-1.0; n_up], vec![-1.0; n_dn], rho_up, rho_dn, v_h, pool.take(ctx, g))
            }
        };
        nuclear_potential_in(&mut batch, k, g, &v_nuc, &system.nuclei, settings.sigma);
        let (rho_out_up, rho_out_dn, rho_tot) = (pool.take(ctx, g), pool.take(ctx, g), pool.take(ctx, g));
        let (v_up, v_dn, exc, xc_up, xc_dn) = (pool.take(ctx, g), pool.take(ctx, g), pool.take(ctx, g), pool.take(ctx, g), pool.take(ctx, g));
        let mut hist = Diis::new(settings.diis_history);

        // --- the loop ---
        let mut iterations = 0;
        let mut delta_rho = f64::INFINITY;
        while iterations < settings.max_iter {
            iterations += 1;
            inner.begin_step();
            lincomb_in(&mut batch, k, g, &rho_tot, &[(1.0, &rho_in_up), (1.0, &rho_in_dn)]);
            flush(&mut batch);
            let ps = poisson.solve(ctx, k, g, r, &rho_tot, &v_h, poisson_tol, POISSON_MAX_ITER);
            inner.poisson(ps, &format!("SCF step {iterations}"));
            xc_lsda_in(&mut batch, k, g, &rho_in_up, &rho_in_dn, &exc, &xc_up, &xc_dn);
            lincomb_in(&mut batch, k, g, &v_up, &[(1.0, &v_nuc), (1.0, &v_h), (1.0, &xc_up)]);
            lincomb_in(&mut batch, k, g, &v_dn, &[(1.0, &v_nuc), (1.0, &v_h), (1.0, &xc_dn)]);
            flush(&mut batch);
            if n_up > 0 {
                let h = GpuHamiltonian { v: &v_up };
                inner.eigen(lobpcg(ctx, k, g, r, &h, &mut orb_up, &mut ev_up, work.as_mut().expect("work sized above"), eig_tol, EIGEN_MAX_ITER));
            }
            if n_dn > 0 {
                let h = GpuHamiltonian { v: &v_dn };
                inner.eigen(lobpcg(ctx, k, g, r, &h, &mut orb_dn, &mut ev_dn, work.as_mut().expect("work sized above"), eig_tol, EIGEN_MAX_ITER));
            }
            fill_in(&mut batch, k, g, &rho_out_up, 0.0);
            for psi in &orb_up { mul_add_sq_in(&mut batch, k, g, &rho_out_up, psi); }
            fill_in(&mut batch, k, g, &rho_out_dn, 0.0);
            for psi in &orb_dn { mul_add_sq_in(&mut batch, k, g, &rho_out_dn, psi); }
            // residuals, Δρ, ∫ρ_out,σ and the new Gram row in one readback
            let (r_up, r_dn) = (pool.take(ctx, g), pool.take(ctx, g));
            lincomb_in(&mut batch, k, g, &r_up, &[(1.0, &rho_out_up), (-1.0, &rho_in_up)]);
            lincomb_in(&mut batch, k, g, &r_dn, &[(1.0, &rho_out_dn), (-1.0, &rho_in_dn)]);
            let (row, q_out) = {
                let mut ops = vec![Reduction::AbsDiffSum(&rho_out_up, &rho_in_up), Reduction::AbsDiffSum(&rho_out_dn, &rho_in_dn),
                                   Reduction::Sum(&rho_out_up), Reduction::Sum(&rho_out_dn)];
                for e in &hist.entries { ops.push(Reduction::Dot(&r_up, &e.r_up)); ops.push(Reduction::Dot(&r_dn, &e.r_dn)); }
                ops.push(Reduction::Dot(&r_up, &r_up));
                ops.push(Reduction::Dot(&r_dn, &r_dn));
                let vals = r.many_in(&mut batch, k, g, &ops);
                inner.profile.readbacks += 1;
                delta_rho = vals[0] + vals[1];
                (vals[4..].chunks(2).map(|c| c[0] + c[1]).collect::<Vec<f64>>(), [vals[2], vals[3]])
            };
            // converged, or a non-finite Δρ (the error path below)
            if delta_rho.is_nan() || delta_rho < settings.tol { pool.give(r_up); pool.give(r_dn); break; }
            hist.push(DiisEntry { rho_up: rho_in_up, rho_dn: rho_in_dn, r_up, r_dn }, row, pool);
            rho_in_up = pool.take(ctx, g);
            rho_in_dn = pool.take(ctx, g);
            hist.mix(&mut batch, k, g, settings.mixing, &rho_in_up, &rho_in_dn, pool);
            // charge renormalisation per channel after the clamp: to ∫ρ_out,σ, not to n_σ — the
            // f32-normalised orbitals give ∫ρ_out = n_σ(1 + ε), ε ~ 1e-7, and a fixed point whose
            // input and output charges differ by n_σ ε would floor Δρ at N|ε| (module doc).
            let q_in = r.many_in(&mut batch, k, g, &[Reduction::Sum(&rho_in_up), Reduction::Sum(&rho_in_dn)]);
            inner.profile.readbacks += 1;
            for (count, target, sum, f) in [(n_up, q_out[0], q_in[0], &rho_in_up), (n_dn, q_out[1], q_in[1], &rho_in_dn)] {
                if count > 0 && sum > 0.0 && target > 0.0 { scale_in(&mut batch, k, g, f, (target / sum) as f32); }
            }
        }
        hist.clear(pool);
        // a NaN Δρ is an error, not a convergence
        if delta_rho.is_nan() || delta_rho >= settings.tol { return Err(QmError::ScfNotConverged(iterations, delta_rho)); }

        // --- energy from the output density and orbitals (qm::scf module doc) ---
        lincomb_in(&mut batch, k, g, &rho_tot, &[(1.0, &rho_out_up), (1.0, &rho_out_dn)]);
        flush(&mut batch);
        let ps = poisson.solve(ctx, k, g, r, &rho_tot, &v_h, poisson_tol, POISSON_MAX_ITER);
        inner.poisson(ps, "final energy");
        xc_lsda_in(&mut batch, k, g, &rho_out_up, &rho_out_dn, &exc, &xc_up, &xc_dn);
        let kin: Vec<Field> = (0..n_up + n_dn).map(|_| pool.take(ctx, g)).collect();
        for (psi, t) in orb_up.iter().chain(&orb_dn).zip(&kin) { kinetic_in(&mut batch, k, g, psi, t); }
        let energy = {
            let mut ops = vec![Reduction::Dot(&rho_tot, &v_nuc), Reduction::Dot(&rho_tot, &v_h), Reduction::Sum(&exc), Reduction::FaceMax(&rho_tot)];
            ops.extend(orb_up.iter().chain(&orb_dn).zip(&kin).map(|(psi, t)| Reduction::Dot(psi, t)));
            let vals = r.many_in(&mut batch, k, g, &ops);
            inner.profile.readbacks += 1;
            let kinetic = neumaier(vals[4..].iter().copied());
            let (external, hartree, xc, boundary) = (vals[0], 0.5 * vals[1], vals[2], vals[3]);
            (Energy { kinetic, external, hartree, xc, nuclear: e_nn, total: kinetic + external + hartree + xc + e_nn }, boundary)
        };
        let (energy, boundary_density) = energy;
        drop(batch); // empty after the last readback
        for f in [rho_in_up, rho_in_dn, v_up, v_dn, exc, xc_up, xc_dn].into_iter().chain(kin) { pool.give(f); }

        let mut warnings = Vec::new();
        if boundary_density > BOUNDARY_DENSITY_WARN { warnings.push(format!("density on box faces is {boundary_density:.2e}: box too small")); }
        let (max_eigen_residual, poisson_residual, poisson_iterations) = (inner.eigen_residual, inner.poisson_residual, inner.poisson_iterations);
        let (mut inner_warnings, mut prof) = inner.finish();
        warnings.append(&mut inner_warnings);
        prof.steps = iterations;
        prof.readbacks += prof.poisson_readbacks + prof.eigen_readbacks;
        prof.wall = t0.elapsed();
        *profile = prof;
        *slot = Some(GpuState { grid: grid.clone(), sigma: settings.sigma, orbitals_up: orb_up, orbitals_dn: orb_dn,
                                evals_up: ev_up, evals_dn: ev_dn, rho: rho_tot, rho_up: rho_out_up, rho_dn: rho_out_dn, v_h, v_nuc });
        Ok(ScfStats { iterations, delta_rho, energy, boundary_density, max_eigen_residual, poisson_residual, poisson_iterations, warnings })
    }

    fn check_state(&self, state: &GpuState) {
        assert!(state.grid == self.grid, "GpuSolver: state from another grid ({:?} vs {:?})", state.grid.n, self.grid.n);
    }

    /// Interior total density as f32 in `qm::Grid` layout (one readback).
    pub fn density_f32(&self, state: &GpuState) -> Vec<f32> {
        self.check_state(state);
        state.rho.read_interior_f32(&self.ctx, &self.g)
    }

    /// Copies the interior density into `volume` (an n³ `Volume3D`) on the GPU: `unpad` into
    /// the row-padded staging buffer, then `copy_buffer_to_texture` with `bytes_per_row` =
    /// 4·stride (stride = n rounded up to a multiple of 64 floats, wgpu's 256-byte row
    /// alignment) and `rows_per_image` = n. No CPU round trip; does not wait.
    pub fn copy_density_to_volume(&self, state: &GpuState, volume: &Volume3D) {
        self.check_state(state);
        let n = self.g.n as u32;
        assert_eq!(volume.dims, [n; 3], "copy_density_to_volume: volume dims {:?} != grid n = {n}", volume.dims);
        let mut enc = encoder(&self.ctx, "density to volume");
        unpad_rec(&mut enc, &self.ctx, &self.k, &self.g, &state.rho, &self.rows, self.row_stride);
        enc.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo { buffer: &self.rows, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * self.row_stride), rows_per_image: Some(n) } },
            wgpu::TexelCopyTextureInfo { texture: &volume.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::Extent3d { width: n, height: n, depth_or_array_layers: n },
        );
        self.ctx.queue.submit([enc.finish()]);
    }

    /// max ρ over the interior (two-level max reduction; one readback).
    pub fn density_max(&self, state: &GpuState) -> f32 {
        self.check_state(state);
        self.r.many(&self.ctx, &self.k, &self.g, &[Reduction::Max(&state.rho)])[0] as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qm::{Grid, Nucleus};
    use std::time::Instant;

    fn hydrogen(pos: [f64; 3]) -> System { System::new(vec![Nucleus { pos, z: 1 }], 0, None) }
    fn h2(shift: f64) -> System {
        System::new(vec![Nucleus { pos: [0.7 + shift, 0.0, 0.0], z: 1 }, Nucleus { pos: [-0.7 + shift, 0.0, 0.0], z: 1 }], 0, None)
    }
    fn lithium() -> System { System::new(vec![Nucleus { pos: [0.0; 3], z: 3 }], 0, None) }

    /// A solver on an n³, h = 0.25 cube (σ = h = 0.25, tol 1e-6 from `ScfSettings::for_grid`).
    fn solver(n: usize) -> Option<(GpuSolver, Grid, ScfSettings)> {
        let ctx = GpuContext::new_headless()?;
        let grid = Grid::cube(n, 0.25);
        let settings = ScfSettings::for_grid(&grid);
        Some((GpuSolver::new(ctx, &grid), grid, settings))
    }

    fn cpu(grid: &Grid, system: &System, settings: &ScfSettings) -> (qm::KsState, ScfStats, f64) {
        let t = Instant::now();
        let (state, stats) = qm::scf(grid, system, settings, None).expect("cpu reference converges");
        (state, stats, t.elapsed().as_secs_f64())
    }

    /// Per-component comparison of a GPU energy with the CPU's; prints the table and asserts
    /// the plan bounds: |ΔE_total| < `total_bound`, each component < `comp_bound`.
    fn compare(name: &str, gpu: &ScfStats, cpu: &ScfStats, total_bound: f64, comp_bound: f64) -> f64 {
        let (a, b) = (gpu.energy, cpu.energy);
        let rows = [("kinetic", a.kinetic, b.kinetic), ("external", a.external, b.external), ("hartree", a.hartree, b.hartree),
                    ("xc", a.xc, b.xc), ("nuclear", a.nuclear, b.nuclear), ("total", a.total, b.total)];
        println!("{name}: GPU {} steps (Δρ {:.2e}, eig res {:.2e}, Poisson res {:.2e} / {} it, warnings {:?}) vs CPU {} steps (Δρ {:.2e})",
                 gpu.iterations, gpu.delta_rho, gpu.max_eigen_residual, gpu.poisson_residual, gpu.poisson_iterations, gpu.warnings, cpu.iterations, cpu.delta_rho);
        for (what, g, c) in rows { println!("  {what:>8}: gpu {g:+.8} cpu {c:+.8} Δ {:+.2e}", g - c); }
        for (what, g, c) in &rows[..5] { assert!((g - c).abs() < comp_bound, "{name} {what}: gpu {g} cpu {c}"); }
        let d = (a.total - b.total).abs();
        assert!(d < total_bound, "{name} total: gpu {} cpu {} (|Δ| = {d:.3e})", a.total, b.total);
        d
    }

    fn charge(s: &GpuSolver, state: &GpuState) -> f64 {
        s.density_f32(state).iter().map(|v| *v as f64).sum::<f64>() * s.grid().dv()
    }

    #[test]
    fn hydrogen_matches_cpu_at_fixed_sigma() {
        let Some((mut s, grid, settings)) = solver(33) else { return };
        let sys = hydrogen([0.0; 3]);
        let (_, cpu_stats, cpu_s) = cpu(&grid, &sys, &settings);
        let t = Instant::now();
        let stats = s.solve(&sys, &settings, false).unwrap().1;
        let gpu_s = t.elapsed().as_secs_f64();
        let state = s.state().unwrap();
        let q = charge(&s, state);
        println!("H 33³ σ = 0.25: GPU {:.2} s ({:?}), CPU {cpu_s:.2} s; ∫ρ = {q:.9}", gpu_s, s.profile());
        let d = compare("H 33³", &stats, &cpu_stats, 1e-5, 2e-5); // brief 1e-4 / 2e-4; achieved 9.6e-8 / ≤ 3.1e-7
        println!("H 33³: |ΔE| = {d:.2e}");
        assert!((q - 1.0).abs() < 1e-5, "∫ρ = {q}");
        assert_eq!(state.orbitals_dn.len(), 0);
        assert!(stats.delta_rho < settings.tol);
    }

    #[test]
    fn h2_matches_cpu_at_fixed_sigma() {
        let Some((mut s, grid, settings)) = solver(33) else { return };
        let sys = h2(0.0);
        let (_, cpu_stats, cpu_s) = cpu(&grid, &sys, &settings);
        let t = Instant::now();
        let stats = s.solve(&sys, &settings, false).unwrap().1;
        let gpu_s = t.elapsed().as_secs_f64();
        let state = s.state().unwrap();
        let q = charge(&s, state);
        println!("H₂ 33³ R = 1.4: GPU {:.2} s ({:?}), CPU {cpu_s:.2} s; ∫ρ = {q:.9}", gpu_s, s.profile());
        let d = compare("H₂ 33³", &stats, &cpu_stats, 1e-5, 2e-5); // brief 1e-4 / 2e-4; achieved 9.1e-8 / ≤ 2.4e-6
        println!("H₂ 33³: |ΔE| = {d:.2e}");
        assert!((q - 2.0).abs() < 1e-5, "∫ρ = {q}");
        assert!((stats.iterations as i64 - cpu_stats.iterations as i64).abs() <= 3, "DIIS steps: gpu {} cpu {}", stats.iterations, cpu_stats.iterations);
        println!("H₂ 33³: Δρ_final = {:.2e} (tol {:.0e}), final-step eigen residual {:.2e}", stats.delta_rho, settings.tol, s.profile().eigen_last_step_residual);
        // No Δρ floor planted by the charge bookkeeping: ρ_in is renormalised to ∫ρ_out, not to N,
        // so a fixed point with ∫ρ_out = N(1 + ε), ε ~ 1e-7, does not force Δρ ≥ N|ε| ≈ 2e-7.
        // A cold solve at tol 1e-7 (below that would-be floor) must converge, in about the CPU's
        // step count. (A warm continuation is not a test: with the same geometry LOBPCG takes 0
        // iterations, ρ_out ≡ ρ_in and Δρ = 0 exactly.)
        let mut tight = ScfSettings::for_grid(&grid);
        tight.tol = 1e-7;
        let (_, cpu_tight, cpu_tight_s) = cpu(&grid, &sys, &tight);
        let t = Instant::now();
        let tight_stats = s.solve(&sys, &tight, false).unwrap_or_else(|e| panic!("H₂ at tol 1e-7 did not converge (a Δρ floor): {e}")).1;
        println!("H₂ 33³ cold at tol 1e-7: GPU {} steps, {:.0} ms, Δρ_final {:.2e}, E {:+.9}; CPU {} steps, {cpu_tight_s:.2} s, Δρ_final {:.2e}, E {:+.9} (|ΔE| {:.2e})",
                 tight_stats.iterations, t.elapsed().as_secs_f64() * 1e3, tight_stats.delta_rho, tight_stats.energy.total,
                 cpu_tight.iterations, cpu_tight.delta_rho, cpu_tight.energy.total, (tight_stats.energy.total - cpu_tight.energy.total).abs());
        assert!(tight_stats.delta_rho < 1e-7);
        assert!((tight_stats.iterations as i64 - cpu_tight.iterations as i64).abs() <= 3, "tol 1e-7 steps: gpu {} cpu {}", tight_stats.iterations, cpu_tight.iterations);
        assert!((tight_stats.energy.total - cpu_tight.energy.total).abs() < 1e-5);
    }

    /// Two cold solves on the same solver, and one on a fresh solver, give bit-identical
    /// energies and iteration counts (Review Focus 1: fixed reduction order, no timing-dependent
    /// state).
    #[test]
    fn solve_is_deterministic() {
        let Some((mut s, grid, settings)) = solver(33) else { return };
        let sys = h2(0.0);
        let e1 = { let (_, st) = s.solve(&sys, &settings, false).unwrap(); st };
        let e2 = { let (_, st) = s.solve(&sys, &settings, false).unwrap(); st };
        let mut fresh = GpuSolver::new(s.ctx.clone(), &grid);
        let e3 = { let (_, st) = fresh.solve(&sys, &settings, false).unwrap(); st };
        println!("deterministic: {:+.12} / {:+.12} / {:+.12} ({} / {} / {} steps)", e1.energy.total, e2.energy.total, e3.energy.total, e1.iterations, e2.iterations, e3.iterations);
        for (name, a, b) in [("same solver", &e1, &e2), ("fresh solver", &e1, &e3)] {
            assert_eq!(a.energy.total.to_bits(), b.energy.total.to_bits(), "{name}: {} vs {}", a.energy.total, b.energy.total);
            assert_eq!(a.energy.kinetic.to_bits(), b.energy.kinetic.to_bits(), "{name} kinetic");
            assert_eq!(a.energy.hartree.to_bits(), b.energy.hartree.to_bits(), "{name} hartree");
            assert_eq!(a.delta_rho.to_bits(), b.delta_rho.to_bits(), "{name} Δρ");
            assert_eq!(a.iterations, b.iterations, "{name} steps");
        }
    }

    /// As the CPU test: a warm start on the displaced geometry is cheaper than a cold solve of
    /// the same geometry and lands on the same energy.
    #[test]
    fn warm_start_is_cheaper_and_agrees() {
        let Some((mut s, _, settings)) = solver(33) else { return };
        let (_, cold) = s.solve(&hydrogen([0.0; 3]), &settings, false).unwrap();
        let cold_prof = s.profile().clone();
        let (_, warm) = s.solve(&hydrogen([0.02, 0.0, 0.0]), &settings, true).unwrap();
        let warm_prof = s.profile().clone();
        let (_, cold_disp) = s.solve(&hydrogen([0.02, 0.0, 0.0]), &settings, false).unwrap();
        let disp_prof = s.profile().clone();
        println!("warm start 33³: cold {} steps {:.0} ms, warm {} steps {:.0} ms (E {:+.8}), cold displaced {} steps {:.0} ms (E {:+.8}); readbacks {} / {} / {}",
                 cold.iterations, cold_prof.wall.as_secs_f64() * 1e3, warm.iterations, warm_prof.wall.as_secs_f64() * 1e3, warm.energy.total,
                 cold_disp.iterations, disp_prof.wall.as_secs_f64() * 1e3, cold_disp.energy.total, cold_prof.readbacks, warm_prof.readbacks, disp_prof.readbacks);
        assert!(warm.iterations < cold_disp.iterations, "cold {} warm {}", cold_disp.iterations, warm.iterations);
        assert!(warm_prof.readbacks < disp_prof.readbacks, "warm {} vs cold {} readbacks", warm_prof.readbacks, disp_prof.readbacks);
        assert!((warm.energy.total - cold_disp.energy.total).abs() < 1e-5, "warm {} vs cold {}", warm.energy.total, cold_disp.energy.total);
        // a warm request with mismatched counts falls back to the cold path
        let (_, h2_after_h) = s.solve(&h2(0.0), &settings, true).unwrap();
        let (_, h2_cold) = s.solve(&h2(0.0), &settings, false).unwrap();
        assert_eq!(h2_after_h.energy.total.to_bits(), h2_cold.energy.total.to_bits(), "mismatched warm state was not ignored");
    }

    /// H (n↓ = 0: the down channel is skipped) and Li (n↑ = 2, n↓ = 1) converge; Li's energy
    /// matches the CPU on the 49³ grid of `qm::scf`'s Li test.
    #[test]
    fn empty_channel_and_lithium_run() {
        let Some((mut s, grid, settings)) = solver(49) else { return };
        let h = s.solve(&hydrogen([0.0; 3]), &settings, false).unwrap().1;
        let state = s.state().unwrap();
        assert!(h.delta_rho < settings.tol);
        assert!(state.orbitals_dn.is_empty() && state.evals_dn.is_empty());
        let dn: Vec<f32> = state.rho_dn.read_interior_f32(&s.ctx, &s.g);
        assert!(dn.iter().all(|v| *v == 0.0), "empty channel density must be exactly 0");
        println!("H 49³: {} steps, E {:+.8}, ∫ρ {:.9}, {:?}", h.iterations, h.energy.total, charge(&s, state), s.profile());
        let li = lithium();
        let (_, cpu_stats, cpu_s) = cpu(&grid, &li, &settings);
        let t = Instant::now();
        let stats = s.solve(&li, &settings, false).unwrap().1;
        let gpu_s = t.elapsed().as_secs_f64();
        let state = s.state().unwrap();
        let q = charge(&s, state);
        println!("Li 49³: GPU {:.2} s ({:?}), CPU {cpu_s:.2} s; ∫ρ = {q:.9}; evals up {:?} dn {:?}", gpu_s, s.profile(), state.evals_up, state.evals_dn);
        assert_eq!((state.orbitals_up.len(), state.orbitals_dn.len()), (2, 1));
        assert!(stats.delta_rho < settings.tol);
        assert!((q - 3.0).abs() < 1e-5, "∫ρ = {q}");
        let d = compare("Li 49³", &stats, &cpu_stats, 5e-5, 1e-4); // brief 5e-4; achieved 3.6e-7 / ≤ 3.6e-6
        let p = s.profile().clone();
        println!("Li 49³: |ΔE| = {d:.2e}; Δρ_final {:.2e}; floor accepts {} (worst {:.2e}), last-step eigen residual {:.2e}, notes {:?}",
                 stats.delta_rho, p.eigen_floor_accepts, p.eigen_floor_worst, p.eigen_last_step_residual, p.notes);
        // floor accepts are informational: in the profile (with a note), never in `warnings`
        assert!(p.eigen_floor_accepts >= 1, "Li's 2s/2p near-degeneracy ends some eigensolves on the floor (measured 4 of 34)");
        assert!(p.eigen_floor_worst > 0.0 && p.notes.iter().any(|n| n.contains("floor")), "{p:?}");
        assert!(stats.warnings.iter().all(|w| !w.to_lowercase().contains("floor")), "floor accept leaked into warnings: {:?}", stats.warnings);
        assert!(p.eigen_last_step_residual > 0.0 && p.eigen_last_step_residual <= stats.max_eigen_residual);
    }

    /// `copy_density_to_volume` into an n³ `Volume3D` equals `density_f32` voxel for voxel
    /// (no padding leaked, x fastest) and the density of a displaced H peaks at the right
    /// (i, j, k) (Review Focus 2).
    #[test]
    fn density_copy_to_volume_is_unpadded_and_centred() {
        let Some((mut s, grid, settings)) = solver(33) else { return };
        // nucleus at x = +1 Bohr = 4 cells off centre: the peak must land at (20, 16, 16), so a
        // transposed or shifted copy (or a ghost cell read as data) fails, not only a non-centred one
        s.solve(&hydrogen([1.0, 0.0, 0.0]), &settings, false).unwrap();
        let state = s.state().unwrap();
        let vol = Volume3D::new(&s.ctx, [33; 3]);
        s.copy_density_to_volume(state, &vol);
        let from_texture = vol.read_back(&s.ctx);
        let direct = s.density_f32(state);
        assert_eq!(from_texture.len(), direct.len());
        assert_eq!(from_texture, direct, "texture copy differs from the direct readback");
        let (argmax, &peak) = direct.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
        assert_eq!(argmax, grid.idx(20, 16, 16), "density peak at {:?}, expected [1, 0, 0]", grid.point(argmax));
        assert_eq!(grid.point(argmax), [1.0, 0.0, 0.0]);
        let dmax = s.density_max(state);
        println!("volume copy 33³: peak {peak} at {:?}; density_max {dmax}", grid.point(argmax));
        assert_eq!(dmax, peak);
        assert!(peak > 0.1 && peak < 0.5, "H peak density {peak} (σ-smeared 1/π ≈ 0.32)");
        // a corner voxel is tiny: the interior, not a ghost cell or a shifted copy
        assert!(direct[0] < 1e-4 && direct[0] >= 0.0, "corner voxel {}", direct[0]);
    }

    /// Diagnostic (ignored; `cargo test -p qm-gpu --release -- --ignored --nocapture diag_`):
    /// Δρ per SCF step for H₂ at 33³, GPU vs CPU, obtained by capping `max_iter` at 1, 2, … and
    /// reading Δρ from the `ScfNotConverged` error; then the step count, energy and cost of H₂,
    /// H and Li as a function of the Poisson tolerance and the LOBPCG tolerance floor factor
    /// (the measurements behind `POISSON_TOL` and `EIG_TOL_FLOOR_FACTOR`).
    #[test]
    #[ignore]
    fn diag_trajectory_and_knobs() {
        let Some((mut s, grid, _)) = solver(33) else { return };
        let sys = h2(0.0);
        let traj = |run: &mut dyn FnMut(usize) -> Result<f64, f64>| -> Vec<f64> {
            let mut out = Vec::new();
            for n in 1..=30 { match run(n) { Ok(d) => { out.push(d); break; } Err(d) => out.push(d) } }
            out
        };
        let cpu_traj = traj(&mut |n| {
            let mut st = ScfSettings::for_grid(&grid); st.max_iter = n;
            match qm::scf(&grid, &sys, &st, None) { Ok((_, x)) => Ok(x.delta_rho), Err(QmError::ScfNotConverged(_, d)) => Err(d), Err(e) => panic!("{e}") }
        });
        let gpu_traj = traj(&mut |n| {
            let mut st = ScfSettings::for_grid(&grid); st.max_iter = n;
            match s.solve(&sys, &st, false) { Ok((_, x)) => Ok(x.delta_rho), Err(QmError::ScfNotConverged(_, d)) => Err(d), Err(e) => panic!("{e}") }
        });
        println!("step   CPU Δρ      GPU Δρ");
        for i in 0..cpu_traj.len().max(gpu_traj.len()) {
            println!("{:>4}   {:>9}   {:>9}", i + 1, cpu_traj.get(i).map_or("-".into(), |d| format!("{d:.2e}")), gpu_traj.get(i).map_or("-".into(), |d| format!("{d:.2e}")));
        }
        for (name, n, sys) in [("H₂", 33, h2(0.0)), ("H", 33, hydrogen([0.0; 3])), ("Li", 49, lithium())] {
            let grid = Grid::cube(n, 0.25);
            let settings = ScfSettings::for_grid(&grid);
            let mut s = GpuSolver::new(s.ctx.clone(), &grid);
            for (ptol, factor) in [(1e-6, 3.0), (1e-8, 3.0), (1e-8, 1.0)] {
                s.set_knobs(ptol, factor);
                let t = Instant::now();
                let stats = s.solve(&sys, &settings, false).unwrap().1;
                let p = s.profile().clone();
                println!("{name} {n}³ Poisson tol {ptol:.0e}, eig floor factor {factor}: {} steps in {:.0} ms, E {:+.9}, Δρ {:.2e}, eig res {:.2e}, Poisson res {:.2e}, Poisson it {} ({:.1}/solve), LOBPCG it {}, readbacks {}",
                         stats.iterations, t.elapsed().as_secs_f64() * 1e3, stats.energy.total, stats.delta_rho, stats.max_eigen_residual, stats.poisson_residual,
                         p.poisson_iterations, p.poisson_iterations as f64 / (p.steps + 1) as f64, p.eigen_iterations, p.readbacks);
            }
        }
    }

    /// Timing for the report: `cargo test -p qm-gpu --release -- --ignored --nocapture bench_scf`.
    /// H₂ cold and warm (displaced by 0.02 Bohr) at 33³ and 49³: ms per step and per solve,
    /// readbacks per step by origin.
    #[test]
    #[ignore]
    fn bench_scf_33_49() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        for n in [33usize, 49] {
            let grid = Grid::cube(n, 0.25);
            let settings = ScfSettings::for_grid(&grid);
            let mut s = GpuSolver::new(ctx.clone(), &grid);
            let _ = s.solve(&h2(0.0), &settings, false).unwrap(); // warm the pipelines/pool
            let report = |label: &str, stats: &ScfStats, p: SolveProfile| {
                let ms = p.wall.as_secs_f64() * 1e3;
                println!("{n}³ H₂ {label}: {} steps, {ms:.1} ms ({:.1} ms/step), E {:+.8}, Δρ {:.2e}; readbacks {} ({:.1}/step: Poisson {:.1}, LOBPCG {:.1}, SCF {:.1}); Poisson {} it, LOBPCG {} it; eig res {:.2e}, Poisson res {:.2e}",
                         stats.iterations, ms / stats.iterations as f64, stats.energy.total, stats.delta_rho, p.readbacks, p.readbacks as f64 / p.steps as f64,
                         p.poisson_readbacks as f64 / p.steps as f64, p.eigen_readbacks as f64 / p.steps as f64,
                         (p.readbacks - p.poisson_readbacks - p.eigen_readbacks) as f64 / p.steps as f64,
                         p.poisson_iterations, p.eigen_iterations, stats.max_eigen_residual, stats.poisson_residual);
            };
            let (_, cold) = s.solve(&h2(0.0), &settings, false).unwrap();
            report("cold", &cold, s.profile().clone());
            let (_, warm) = s.solve(&h2(0.02), &settings, true).unwrap();
            report("warm (shifted 0.02)", &warm, s.profile().clone());
            let (_, cold2) = s.solve(&h2(0.02), &settings, false).unwrap();
            report("cold (shifted 0.02)", &cold2, s.profile().clone());
        }
    }
}
