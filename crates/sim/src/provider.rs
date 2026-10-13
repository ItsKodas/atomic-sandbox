//! The force boundary (CLAUDE.md rule 5): [`ForceProvider`] turns a [`Scene`]'s nuclear
//! positions into the Born-Oppenheimer energy and the forces on the nuclei; the integrator never
//! knows what computes them. Two Kohn-Sham adapters, both on the cubic grid of `scene.cell` with
//! σ = `scene.sigma` or h and the provider's SCF tolerance:
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
//! # Errors
//! - [`ProviderError::Cancelled`]: `should_abort` returned true (the GPU keeps the in-progress
//!   state as its warm start).
//! - [`ProviderError::Solver`]: the solver's own errors — coincident nuclei (raised before any
//!   work), no SCF convergence, and, GPU only, `QmError::Numerical` (non-finite numerics; the
//!   solver drops its state and the next evaluation is cold). After an error the CPU provider
//!   holds no warm state either.
//! - [`ProviderError::Invalid`]: the scene cannot be posed — a charge above the proton count or
//!   a multiplicity impossible for the electron count (what `qm::System::new` would panic on).
//!   No `QmError` variant fits these: they are input errors, not solver outcomes.
use crate::scene::{Cell, Scene};
use gpu::{GpuContext, Volume3D};
use qm::{Grid, ScfSettings, ScfStats, System};
use std::sync::Arc;
use std::time::Instant;

/// One evaluation: E_KS (Hartree, nuclear repulsion included; the same number as
/// `stats.energy.total`) and the force on each nucleus (Hartree/Bohr, in `scene.nuclei` order),
/// plus the density for the viewer when the provider makes one (module doc, "Display payload").
#[derive(Clone, Debug)]
pub struct ForceResult {
    pub energy: f64,
    pub forces: Vec<[f64; 3]>,
    pub stats: EvalStats,
    pub display: Option<Display>,
}

/// The converged total density ρ↑ + ρ↓ (e/Bohr³) on the evaluation's grid, for display only.
/// `cell` is the scene's; `origin` is the position of grid point (0, 0, 0) (`qm::Grid::origin`);
/// values are in `qm::Grid` layout (x fastest); `max` is the largest value.
#[derive(Clone)]
pub enum Display {
    /// n³ f32 values.
    Cpu { cell: Cell, origin: [f64; 3], rho: Arc<Vec<f32>>, max: f32 },
    /// An n³ r32float texture on the provider's device (the `GpuContext` it was built with).
    Gpu { cell: Cell, origin: [f64; 3], volume: Arc<Volume3D>, max: f32 },
}

impl std::fmt::Debug for Display {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Display::Cpu { cell, origin, rho, max } => f.debug_struct("Display::Cpu").field("cell", cell).field("origin", origin)
                .field("len", &rho.len()).field("max", max).finish(),
            Display::Gpu { cell, origin, volume, max } => f.debug_struct("Display::Gpu").field("cell", cell).field("origin", origin)
                .field("dims", &volume.dims).field("max", max).finish(),
        }
    }
}

/// Cost and diagnostics of one evaluation.
/// - `scf_steps`: SCF iterations; `readbacks`: GPU→CPU readbacks (GPU only — the GPU's cost
///   measure, `SolveProfile::readbacks`).
/// - `solve_ms`: wall of the evaluation — solve, forces and the display copy — **excluding**
///   the `GpuSolver` build on a grid change (one-off setup, ~0.3 s at 97³, kept out so a
///   per-step cost estimate is not spiked by it); reported, never a pass/fail measure.
/// - `warnings`: the solver's warnings (`ScfStats::warnings`: box too small, Poisson or
///   final-step LOBPCG not converged); `notes`: informational lines that are not warnings (GPU:
///   the `SolveProfile` notes and whether the start density was extrapolated; CPU: LOBPCG
///   stopping at its iteration cap).
/// - `eig_tol_used`: the LOBPCG tolerance of the final SCF step (GPU: the Δρ-tied tol,
///   `SolveProfile::eig_tol_used`; CPU: the fixed `ScfSettings::eig_tol`).
/// - `energy`: the breakdown (kinetic, external, Hartree, xc, nuclear, total — Hartree);
///   `max_eigen_residual`, `poisson_residual`: as `ScfStats` defines them (the worst LOBPCG
///   exit residual of the tight-tol steps; the worst relative Poisson residual).
#[derive(Clone, Debug)]
pub struct EvalStats {
    pub scf_steps: usize,
    pub readbacks: Option<usize>,
    pub solve_ms: u64,
    pub warnings: Vec<String>,
    pub notes: Vec<String>,
    pub eig_tol_used: Option<f64>,
    pub energy: qm::Energy,
    pub max_eigen_residual: f64,
    pub poisson_residual: f64,
}

/// All zeros (`qm::Energy` has no `Default`; an analytic provider fills `energy.total` and
/// leaves the rest).
impl Default for EvalStats {
    fn default() -> EvalStats {
        EvalStats {
            scf_steps: 0,
            readbacks: None,
            solve_ms: 0,
            warnings: Vec::new(),
            notes: Vec::new(),
            eig_tol_used: None,
            energy: qm::Energy { kinetic: 0.0, external: 0.0, hartree: 0.0, xc: 0.0, nuclear: 0.0, total: 0.0 },
            max_eigen_residual: 0.0,
            poisson_residual: 0.0,
        }
    }
}

/// Why an evaluation produced no forces (module doc, "Errors").
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("cancelled")]
    Cancelled,
    #[error("{0}")]
    Solver(#[from] qm::QmError),
    /// The scene cannot be posed as a Kohn-Sham problem (charge or multiplicity).
    #[error("{0}")]
    Invalid(String),
}

pub trait ForceProvider: Send {
    /// Energy and forces for the scene's current nuclear positions. `should_abort` is polled at solver-step granularity.
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError>;
    fn name(&self) -> &'static str;
    /// Forget warm state (grid change, discontinuous edit).
    fn reset(&mut self);
}

/// Pre-validates what `System::new` would panic on and builds the solve's inputs: the grid of
/// `scene.cell`, the system (ΣZ − charge electrons, the multiplicity) and the SCF settings
/// (`ScfSettings::for_grid` with `tol` and σ = `scene.sigma` or h).
fn problem(scene: &Scene, tol: f64) -> Result<(Grid, System, ScfSettings), ProviderError> {
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

fn elapsed_ms(t0: Instant) -> u64 {
    t0.elapsed().as_millis() as u64
}

/// The `EvalStats` fields that come straight from `ScfStats`.
fn eval_stats(stats: ScfStats, readbacks: Option<usize>, t0: Instant, notes: Vec<String>, eig_tol_used: f64) -> EvalStats {
    let ScfStats { iterations, energy, max_eigen_residual, poisson_residual, warnings, .. } = stats;
    EvalStats {
        scf_steps: iterations,
        readbacks,
        solve_ms: elapsed_ms(t0),
        warnings,
        notes,
        eig_tol_used: Some(eig_tol_used),
        energy,
        max_eigen_residual,
        poisson_residual,
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
}

impl CpuKs {
    /// `tol`: the SCF tolerance on Δρ = ∫|ρ_out − ρ_in|.
    pub fn new(tol: f64) -> CpuKs {
        CpuKs { settings_tol: tol, warm: None, grid: None, display: true }
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
        let (grid, system, settings) = problem(scene, self.settings_tol)?;
        if should_abort() {
            return Err(ProviderError::Cancelled);
        }
        let t0 = Instant::now();
        if self.grid.as_ref() != Some(&grid) {
            self.warm = None;
            self.grid = Some(grid.clone());
        }
        let prev = self.warm.take();
        let (state, stats) = qm::scf(&grid, &system, &settings, prev.as_ref())?;
        let forces = qm::forces(&grid, &state, &system.nuclei, settings.sigma)?;
        let display = self.display.then(|| {
            let rho: Vec<f32> = state.rho().iter().map(|&x| x as f32).collect();
            let max = rho.iter().copied().fold(0.0f32, f32::max);
            Display::Cpu { cell: scene.cell.clone(), origin: grid.origin, rho: Arc::new(rho), max }
        });
        self.warm = Some(state);
        let mut notes = Vec::new();
        if stats.max_eigen_residual >= settings.eig_tol {
            notes.push(format!("LOBPCG stopped at its iteration cap at least once (max residual {:.2e} ≥ tol {:.0e})",
                               stats.max_eigen_residual, settings.eig_tol));
        }
        let energy = stats.energy.total;
        Ok(ForceResult { energy, forces, stats: eval_stats(stats, None, t0, notes, settings.eig_tol), display })
    }

    fn name(&self) -> &'static str {
        "Kohn-Sham (CPU)"
    }

    fn reset(&mut self) {
        self.warm = None;
    }
}

/// The GPU adapter's density textures. A texture leaves the pool inside a [`Display`] as an
/// `Arc` and is written again only when the pool holds the only reference: no result, snapshot
/// or viewer (displayed or bound) still points at it, so a texture is never overwritten while
/// it can be on screen. The check is race-free: other holders can only drop references, never
/// create one from nothing. With a viewer that keeps one result this settles at two textures
/// (ping-pong: the displayed one and the one being written); a grid change drops the old size.
/// (Moved from `app::sim`, whose copy goes when the app switches to `sim`.)
#[derive(Default)]
struct TexturePool {
    all: Vec<Arc<Volume3D>>,
}

impl TexturePool {
    fn take(&mut self, ctx: &GpuContext, dims: [u32; 3]) -> Arc<Volume3D> {
        self.all.retain(|t| t.dims == dims);
        if let Some(t) = self.all.iter().find(|t| Arc::strong_count(t) == 1) {
            return t.clone();
        }
        let t = Arc::new(Volume3D::new(ctx, dims));
        self.all.push(t.clone());
        t
    }
}

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
}

impl GpuKs {
    /// `ctx`: the device to solve on (a clone of the viewer's, so the display textures can be
    /// bound by its renderer); `tol`: the SCF tolerance on Δρ.
    pub fn new(ctx: GpuContext, tol: f64) -> GpuKs {
        GpuKs { solver: None, ctx, tol, extrapolate: true, cold: false, display: true, textures: TexturePool::default() }
    }

    /// Whether results carry the density texture (`ForceResult::display`; default on). Off: no
    /// copy, no texture, and the pool's own references are dropped.
    pub fn set_display(&mut self, on: bool) {
        self.display = on;
        if !on {
            self.textures.all.clear();
        }
    }
}

impl ForceProvider for GpuKs {
    /// `should_abort` is polled before the solve and before every SCF step.
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        let (grid, system, settings) = problem(scene, self.tol)?;
        if should_abort() {
            return Err(ProviderError::Cancelled);
        }
        if self.solver.as_ref().is_none_or(|s| *s.grid() != grid) {
            self.solver = None; // free the old grid's buffers before allocating the new ones
            let mut s = qm_gpu::GpuSolver::new(self.ctx.clone(), &grid);
            s.set_density_extrapolation(self.extrapolate);
            self.solver = Some(s);
        }
        let t0 = Instant::now(); // after the build: `solve_ms` excludes it (`EvalStats` doc)
        let solver = self.solver.as_mut().expect("created above");
        // consumed by this attempt even if it is abandoned: the state it parks belongs to the
        // new geometry (a valid warm start); an error leaves the solver without state anyway
        let warm = !std::mem::take(&mut self.cold);
        let result = solver.solve_with(&system, &settings, warm, should_abort);
        // ...except the coincident-nuclei error, which keeps the old state as the warm start:
        // a pending reset must survive it (after the other errors the solver holds no state)
        if result.is_err() {
            self.cold |= !warm;
        }
        let Some((_, stats)) = result? else {
            return Err(ProviderError::Cancelled);
        };
        let state = solver.state().expect("a converged solve exposes its state");
        let forces = solver.forces(state, &system.nuclei, settings.sigma)?;
        let display = self.display.then(|| {
            let volume = self.textures.take(&self.ctx, grid.n.map(|n| n as u32));
            // queue order makes the copy visible to any later render submission; the max
            // reduction below also waits for the GPU
            solver.copy_density_to_volume(state, &volume);
            let max = solver.density_max(state);
            Display::Gpu { cell: scene.cell.clone(), origin: grid.origin, volume, max }
        });
        let profile = solver.profile();
        let mut notes = profile.notes.clone();
        if profile.extrapolated {
            notes.push("start density extrapolated (2ρ_t − ρ_{t−1})".into());
        }
        let energy = stats.energy.total;
        Ok(ForceResult { energy, forces, stats: eval_stats(stats, Some(profile.readbacks), t0, notes, profile.eig_tol_used), display })
    }

    fn name(&self) -> &'static str {
        "Kohn-Sham (GPU)"
    }

    /// The next solve is cold and does not extrapolate: the extrapolation history is dropped
    /// now (off, then back on) and `warm = false` discards the warm or parked state.
    fn reset(&mut self) {
        self.cold = true;
        if let Some(s) = self.solver.as_mut() {
            s.set_density_extrapolation(false);
            s.set_density_extrapolation(self.extrapolate);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{Cell, Nucleus};
    use gpu::GpuContext;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    /// Serialises the GPU sections of concurrently running tests (poisoning is harmless here).
    static GPU: Mutex<()> = Mutex::new(());

    fn gpu() -> Option<GpuContext> {
        let ctx = GpuContext::new_headless();
        if ctx.is_none() { eprintln!("sim: no GPU adapter — skipped"); }
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
        assert_eq!(g.textures.all.len(), 3);
        // released by everyone: the pool reuses instead of allocating
        drop(held);
        let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
        g.evaluate(&h2(33, 1.46), &mut never()).expect("GpuKs");
        assert_eq!(g.textures.all.len(), 3, "a free texture was reused");
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
        assert_eq!(g.textures.all.len(), 1);
        g.set_display(false);
        assert!(g.textures.all.is_empty(), "turning display off drops the pool's textures");
        let off = g.evaluate(&h2(17, 1.42), &mut never()).expect("GPU display off");
        assert!(off.display.is_none());
        assert!(g.textures.all.is_empty(), "no texture created with display off");
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
        assert!(matches!(err, ProviderError::Solver(qm::QmError::CoincidentNuclei(0, 1))), "{err:?}");
        let after_err = g.evaluate(&h2(25, 1.42), &mut never()).expect("GPU after reset + error");
        println!("GPU 25³ R 1.42 after reset + coincident-nuclei error: {} steps (cold {})", after_err.stats.scf_steps, cold.stats.scf_steps);
        assert!(after_err.stats.scf_steps >= cold.stats.scf_steps);
    }
}
