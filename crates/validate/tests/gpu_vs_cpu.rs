//! GPU solver (`qm-gpu`) against the CPU reference (`qm`) **at fixed σ** (M3 Task 8; numbers
//! and the run log in `references.md`, section "GPU vs CPU at fixed σ"). Run with
//! `cargo test -p validate --release --test gpu_vs_cpu` (release-only; skips cleanly without an adapter).
//!
//! # Why fixed σ and not the σ → 0 oracles
//! The physics oracles in `oracles.rs` extrapolate E(σ) to σ → 0 through a four-point fit whose
//! intercept weights sum to |w|₁ ≈ 88 (M1 handoff §3; `validate` module doc), so every 1e-6 Ha
//! of noise in an E(σ) becomes ~1e-4 in the extrapolated energy, and the He oracle passes with
//! 2.7 mHa of margin. The GPU keeps every grid quantity in f32: its total energy carries a
//! path-dependent scatter of ~6e-7 Ha for H₂ at 49³ (Task 7, cold vs warm-from-neighbour) and
//! ~5e-6 Ha for H₂O (Task 8, two cold trajectories), i.e. the extrapolated oracles would amplify
//! f32 noise to the mHa level and could fail a correct solver. The GPU is therefore validated
//! the other way round: the CPU f64 solver, which *does* pass the oracles, is the reference, and
//! the GPU must reproduce it on the **same discrete problem** — same grid, same σ = h = 0.25,
//! same tolerances — to f32-limited bounds. Everything here compares two solutions of one
//! finite-σ, finite-h functional; nothing is extrapolated. The extrapolated oracles stay the
//! CPU-reference gate until pseudopotentials land.
//!
//! # What is compared
//! - `energies_match_at_fixed_sigma`: H, He, H₂ (R = 1.4), LiH (R = 3.0, the `lih` preset) and
//!   H₂O (the `h2o` preset) at 49³, tol 1e-6; |E_gpu − E_cpu| per system against bounds that
//!   started at the brief's 1e-4 (H/He/H₂) and 5e-4 (LiH/H₂O) and were tightened to the
//!   achieved values (constants below, with the measurements).
//! - `forces_match_at_fixed_sigma`: Hellmann-Feynman forces per component for H₂ at
//!   R = 1.0 / 1.4 / 2.5 and H₂O at 49³ (brief 1e-3 Ha/Bohr, tightened).
//! - `h2_curve_shape_matches`: E(R) for R ∈ {1.0, 1.4, 1.8, 2.4, 3.2}: a constant offset
//!   between the two solvers is allowed, the shape must match (|ΔE − mean ΔE| ≤ `SHAPE_BOUND` = 5e-6; the brief's was 5e-5), for the
//!   GPU solved cold at every R and warm along R (the MD path).
//! - `timing_table`: cold/warm solves of H₂ at 33³/49³/65³/97³ and H₂O at 57³/97³ with the
//!   per-phase split from `SolveProfile` (Poisson / LOBPCG / cold-start guess / the rest);
//!   printed and written to `last-run-gpu.md` (gitignored). Asserts only that it finished:
//!   the plan's targets (49³ H₂ cold ≤ 0.5 s, warm ≤ 0.15 s; 97³ H₂O cold ≤ 5 s) are reported,
//!   and the Task 10 (multigrid) trigger — Poisson ≥ 40 % of a 97³ step, or the 97³ H₂O cold
//!   solve > 5 s — is evaluated and printed.
//!
//! The GPU work of the concurrently running tests is serialised through `GPU` so the timing
//! rows are uncontended; the CPU reference solves (the long part) run in parallel. Both
//! solvers are deterministic (the GPU bit-for-bit, `qm_gpu::scf` doc), so the achieved numbers
//! are reproducible on one driver.

use gpu::GpuContext;
use qm::{Energy, Grid, Nucleus, ScfSettings, ScfStats, System};
use qm_gpu::{GpuSolver, SolveProfile};
use std::fmt::Write as _;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Serialises the GPU sections of concurrently running tests (poisoning is harmless here).
static GPU: Mutex<()> = Mutex::new(());

fn gpu() -> Option<GpuContext> {
    let ctx = GpuContext::new_headless();
    if ctx.is_none() { eprintln!("gpu_vs_cpu: no GPU adapter — skipped"); }
    ctx
}

fn atom(z: u8) -> System { System::new(vec![Nucleus { pos: [0.0; 3], z }], 0, None) }

/// H₂ along x at bond length `r`, centred at `shift` (nucleus 0 at +r/2).
fn h2_at(r: f64, shift: f64) -> System {
    System::new(vec![Nucleus { pos: [r / 2.0 + shift, 0.0, 0.0], z: 1 }, Nucleus { pos: [-r / 2.0 + shift, 0.0, 0.0], z: 1 }], 0, None)
}
fn h2(r: f64) -> System { h2_at(r, 0.0) }

/// LiH at the `lih` preset geometry: Li at −0.8, H at +2.2 (R = 3.0 Bohr), 4 electrons (2, 2).
fn lih() -> System { System::new(vec![Nucleus { pos: [-0.8, 0.0, 0.0], z: 3 }, Nucleus { pos: [2.2, 0.0, 0.0], z: 1 }], 0, None) }

/// H₂O at the `h2o` preset geometry: O at the origin, H at (±1.43, 1.11, 0) — O–H 1.81 Bohr,
/// H–O–H 104.5° — shifted by `shift` along x; 10 electrons (5, 5).
fn h2o_at(shift: f64) -> System {
    System::new(vec![Nucleus { pos: [shift, 0.0, 0.0], z: 8 }, Nucleus { pos: [1.43 + shift, 1.11, 0.0], z: 1 }, Nucleus { pos: [-1.43 + shift, 1.11, 0.0], z: 1 }], 0, None)
}
fn h2o() -> System { h2o_at(0.0) }

fn ms(d: Duration) -> f64 { d.as_secs_f64() * 1e3 }

/// One CPU reference solve (cold), panicking on failure: a reference that does not converge is
/// a test failure, not a skip.
fn cpu_solve(grid: &Grid, sys: &System, settings: &ScfSettings, what: &str) -> (qm::KsState, ScfStats, Duration) {
    let t = Instant::now();
    let (state, stats) = qm::scf(grid, sys, settings, None).unwrap_or_else(|e| panic!("CPU reference {what} did not converge: {e}"));
    (state, stats, t.elapsed())
}

/// One GPU solve under the GPU lock; returns the stats, the profile and the wall.
fn gpu_solve(s: &mut GpuSolver, sys: &System, settings: &ScfSettings, warm: bool, what: &str) -> (ScfStats, SolveProfile, Duration) {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let t = Instant::now();
    let stats = s.solve(sys, settings, warm).unwrap_or_else(|e| panic!("GPU {what} did not converge: {e}")).1;
    (stats, s.profile().clone(), t.elapsed())
}

fn components(e: &Energy) -> [(&'static str, f64); 6] {
    [("kinetic", e.kinetic), ("external", e.external), ("hartree", e.hartree), ("xc", e.xc), ("nuclear", e.nuclear), ("total", e.total)]
}

/// The largest |Δ| over the five energy components (the total is the sixth).
fn worst_component(a: &Energy, b: &Energy) -> (&'static str, f64) {
    components(a).iter().zip(components(b).iter()).take(5)
        .map(|((n, x), (_, y))| (*n, (x - y).abs())).fold(("", 0.0), |m, c| if c.1 > m.1 { c } else { m })
}

/// Lines any test may attach to the report: warnings and notes of both solvers.
fn diagnostics(name: &str, gpu: &ScfStats, prof: &SolveProfile, cpu: &ScfStats) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "  {name}: GPU Δρ {:.2e}, eig res {:.2e} (last step {:.2e}), Poisson res {:.2e}, floor accepts {}, notes {:?}, warnings {:?}",
                     gpu.delta_rho, gpu.max_eigen_residual, prof.eigen_last_step_residual, gpu.poisson_residual, prof.eigen_floor_accepts, prof.notes, gpu.warnings);
    let _ = writeln!(s, "  {name}: CPU Δρ {:.2e}, eig res {:.2e}, Poisson res {:.2e}, warnings {:?}", cpu.delta_rho, cpu.max_eigen_residual, cpu.poisson_residual, cpu.warnings);
    s
}

/// |ΔE_total| bounds (Ha) per system, 49³, σ = h = 0.25, tol 1e-6. Brief: 1e-4 for H/He/H₂,
/// 5e-4 for LiH/H₂O. Tightened (plan rule: ~3× the achieved) after the 2026-10-13 run
/// (references.md): Task 8 achieved |ΔE| H 1.5e-7, He 1.0e-6, H₂ 5.9e-7, LiH 3.1e-7, H₂O 1.27e-5.
/// The bound is 3× the larger of the achieved total and the f32 *path scatter* of the GPU
/// energy for that system class (two correct GPU trajectories differ by 6e-7 for H₂ at 49³,
/// Task 7, and 5.4e-6 for H₂O, Task 8) rounded up to one digit — a total that happens to
/// cancel better than its components (LiH: 3.1e-7 vs 2.1e-6 in E_ext) is not pinned at
/// its lucky value. Never loosened from here (CLAUDE.md rule 6).
///
/// Task 8's H₂O |ΔE| of 1.27e-5 was at the f32 level but its mechanism was not isolated; the
/// hedge named the A&S erf in `nuclear.wgsl` as the leading candidate. Task 10 settled it the
/// other way: the multigrid Poisson (commit B) alone took it to 1.84e-6 — the old Jacobi-CG
/// ran 60–100 iterations per solve to an f32 recurrence floor whose true residual was ~1e-5
/// relative, and that V_H error biased the fixed point (the components now cancel to second
/// order: external −4.4e-5, hartree +2.9e-5, kinetic +2.1e-5); the erf upgrade (commit C)
/// took the total from +1.84e-6 to +2.35e-7. Achieved after Task 10: H 4.7e-9, He 4.2e-8,
/// H₂ 1.4e-8, LiH 1.1e-7, H₂O 2.35e-7. The bounds are 3× the larger of the achieved value and the measured path
/// scatter (6e-7 H₂-class, 5.4e-6 H₂O — Task 8's two-trajectory figure, not re-measured), so
/// the scatter now sets every bound: H₂O 4e-5 → 2e-5, He 3e-6 → 2e-6.
/// 3× max(2.1e-8 achieved, 6e-7 H₂-class path scatter) = 1.8e-6, pinned below at 1e-6 (H has no second nucleus, less scatter)
const E_BOUND_H: f64 = 1e-6;
/// 3× max(6.4e-8 achieved, 6e-7 scatter) = 1.8e-6 → 2e-6 (Task 8: 3e-6 from the 1.0e-6 then achieved)
const E_BOUND_HE: f64 = 2e-6;
/// 3× max(1.3e-7 achieved, 6e-7 scatter) = 1.8e-6 → 2e-6
const E_BOUND_H2: f64 = 2e-6;
/// 3× 1e-6 (the component-level scatter of a Z = 3 system; the 2.9e-7 total is a cancellation)
const E_BOUND_LIH: f64 = 3e-6;
/// 3× max(1.8e-6 achieved, 5.4e-6 path scatter) = 1.6e-5 → 2e-5 (Task 8: 4e-5 from the 1.27e-5 then achieved)
const E_BOUND_H2O: f64 = 2e-5;

/// Step-count guard for H₂O (GPU steps ≤ CPU steps + this): the regression test for the
/// ρ_in renormalisation target (`qm_gpu::scf` doc). Renormalising to n_σ instead of ∫ρ_out,σ
/// plants a Δρ floor N|ε| that costs H₂O (N = 10) ten DIIS steps — measured 27 vs 17 at
/// tol 1e-6 with the CPU at 17 (references.md) — while its energy stays inside `E_BOUND_H2O`
/// (−7.3e-6 vs the CPU), so only the step count catches it. H₂ cannot: its N|ε| ≈ 1e-7 sits
/// at the tolerance. Not applied to the other systems (cold H₂ at R = 3.2 takes 14 steps vs
/// the CPU's 9 in `h2_curve_shape_matches` — a trajectory difference, not a floor).
const H2O_STEP_MARGIN: usize = 5;

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test gpu_vs_cpu")]
fn energies_match_at_fixed_sigma() {
    let Some(ctx) = gpu() else { return };
    let grid = Grid::cube(49, 0.25);
    let settings = ScfSettings::for_grid(&grid);
    // (name, system, |ΔE| bound, step-count margin if guarded)
    let cases: Vec<(&str, System, f64, Option<usize>)> = vec![
        ("H", atom(1), E_BOUND_H, None), ("He", atom(2), E_BOUND_HE, None), ("H₂ R=1.4", h2(1.4), E_BOUND_H2, None),
        ("LiH R=3.0", lih(), E_BOUND_LIH, None), ("H₂O", h2o(), E_BOUND_H2O, Some(H2O_STEP_MARGIN)),
    ];
    let mut s = GpuSolver::new(ctx, &grid);
    let mut table = String::from("| system | E_cpu | E_gpu | ΔE (gpu − cpu) | steps cpu / gpu | ms cpu / gpu | worst component | bound |\n|---|---|---|---|---|---|---|---|\n");
    let mut diag = String::new();
    let mut misses = Vec::new();
    for (name, sys, bound, step_margin) in &cases {
        let (_, cpu, cpu_t) = cpu_solve(&grid, sys, &settings, name);
        let (gpu, prof, gpu_t) = gpu_solve(&mut s, sys, &settings, false, name);
        let d = gpu.energy.total - cpu.energy.total;
        let (wc, wd) = worst_component(&gpu.energy, &cpu.energy);
        let _ = writeln!(table, "| {name} | {:+.9} | {:+.9} | {d:+.2e} | {} / {} | {:.0} / {:.0} | {wc} {wd:.1e} | {bound:.0e} |",
                         cpu.energy.total, gpu.energy.total, cpu.iterations, gpu.iterations, ms(cpu_t), ms(gpu_t));
        diag += &diagnostics(name, &gpu, &prof, &cpu);
        for ((n, g), (_, c)) in components(&gpu.energy).iter().zip(components(&cpu.energy).iter()) {
            let _ = writeln!(diag, "    {n:>8}: gpu {g:+.9} cpu {c:+.9} Δ {:+.2e}", g - c);
        }
        assert!(gpu.warnings.iter().all(|w| !w.contains("Poisson")), "{name}: GPU Poisson at its cap: {:?}", gpu.warnings);
        let ok = d.abs() <= *bound; // NaN is a miss
        if !ok { misses.push(format!("{name}: |ΔE| = {:.3e} > {bound:.0e}", d.abs())); }
        if let Some(margin) = step_margin {
            if gpu.iterations > cpu.iterations + margin {
                misses.push(format!("{name}: GPU took {} DIIS steps vs the CPU's {} (+{margin} allowed): a Δρ floor (renormalisation target?)", gpu.iterations, cpu.iterations));
            }
        }
    }
    println!("GPU vs CPU energies, 49³, σ = h = 0.25, tol 1e-6 (Ha; ms = wall of each solve, GPU uncontended)\n{table}{diag}");
    assert!(misses.is_empty(), "energy bounds missed: {misses:?}");
}

/// |ΔF| bound per component (Ha/Bohr), 49³, both states at tol 1e-6 (first order in their
/// Δρ ≈ 5e-7 times ∫|∇V_I|, which grows with Z). Brief 1e-3; achieved (references.md,
/// 2026-10-13) H₂ 1.35e-6 / 1.4e-8 / 1.66e-6 at R = 1.0 / 1.4 / 2.5, H₂O 1.73e-5 (F_y on
/// O, Z = 8) → bounds 3× the achieved, rounded up. After Task 10 (MG-PCG) the H₂ numbers are
/// 2.8e-6 / 3.0e-6 / 1.8e-6 against `F_BOUND_H2` = 5e-6: a 1.7× margin, the thinnest in the
/// suite and on a first-order quantity, so a driver-level trip reads as expected sensitivity,
/// not a regression. After M4 Task 1 (Δρ-tied eigen tol): 3.72e-6 / 2.63e-6 / 1.17e-6, a 1.34×
/// margin — a re-draw of the frozen-final-orbital force scatter (2e-7…3e-6 under either eigen
/// rule, `qm_gpu` `diag_force_vs_eig_rule`). A future trip is that scatter; the remedy is a lower
/// LOBPCG floor (Task 10 remedy 2), never a larger bound.
const F_BOUND_H2: f64 = 5e-6;
const F_BOUND_H2O: f64 = 6e-5;

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test gpu_vs_cpu")]
fn forces_match_at_fixed_sigma() {
    let Some(ctx) = gpu() else { return };
    let grid = Grid::cube(49, 0.25);
    let settings = ScfSettings::for_grid(&grid);
    let cases: Vec<(&str, System, f64)> = vec![
        ("H₂ R=1.0", h2(1.0), F_BOUND_H2), ("H₂ R=1.4", h2(1.4), F_BOUND_H2), ("H₂ R=2.5", h2(2.5), F_BOUND_H2), ("H₂O", h2o(), F_BOUND_H2O),
    ];
    let mut s = GpuSolver::new(ctx, &grid);
    let mut out = String::from("GPU vs CPU Hellmann-Feynman forces, 49³, σ = h = 0.25, tol 1e-6 (Ha/Bohr)\n");
    let mut misses = Vec::new();
    for (name, sys, bound) in &cases {
        let (cpu_state, cpu, _) = cpu_solve(&grid, sys, &settings, name);
        let f_cpu = qm::forces(&grid, &cpu_state, &sys.nuclei, settings.sigma).unwrap();
        let (gpu, prof, _) = gpu_solve(&mut s, sys, &settings, false, name);
        let f_gpu = {
            let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
            s.forces(s.state().unwrap(), &sys.nuclei, settings.sigma).unwrap()
        };
        let mut worst = 0.0f64;
        let _ = writeln!(out, "{name}: E_gpu {:+.9} E_cpu {:+.9} (ΔE {:+.2e}); steps {} / {}", gpu.energy.total, cpu.energy.total, gpu.energy.total - cpu.energy.total, cpu.iterations, gpu.iterations);
        for (i, nuc) in sys.nuclei.iter().enumerate() {
            let d: Vec<f64> = (0..3).map(|a| (f_gpu[i][a] - f_cpu[i][a]).abs()).collect();
            worst = worst.max(d.iter().cloned().fold(0.0, f64::max));
            let _ = writeln!(out, "  nucleus {i} (Z = {}): gpu [{:+.8}, {:+.8}, {:+.8}]  cpu [{:+.8}, {:+.8}, {:+.8}]  |Δ| [{:.1e}, {:.1e}, {:.1e}]",
                             nuc.z, f_gpu[i][0], f_gpu[i][1], f_gpu[i][2], f_cpu[i][0], f_cpu[i][1], f_cpu[i][2], d[0], d[1], d[2]);
        }
        let sum_gpu: Vec<f64> = (0..3).map(|a| f_gpu.iter().map(|f| f[a]).sum()).collect();
        let sum_cpu: Vec<f64> = (0..3).map(|a| f_cpu.iter().map(|f| f[a]).sum()).collect();
        let _ = writeln!(out, "  max |ΔF| = {worst:.2e} (bound {bound:.0e}); Σ_I F_I gpu [{:+.1e}, {:+.1e}, {:+.1e}] cpu [{:+.1e}, {:+.1e}, {:+.1e}] (egg-box, not zero on a grid)",
                         sum_gpu[0], sum_gpu[1], sum_gpu[2], sum_cpu[0], sum_cpu[1], sum_cpu[2]);
        out += &diagnostics(name, &gpu, &prof, &cpu);
        let ok = worst <= *bound; // NaN is a miss
        if !ok { misses.push(format!("{name}: max |ΔF| = {worst:.3e} > {bound:.0e}")); }
    }
    println!("{out}");
    assert!(misses.is_empty(), "force bounds missed: {misses:?}");
}

/// Shape bound: |ΔE(R) − mean ΔE| (brief 5e-5 Ha). Achieved 9.7e-7 cold / 8.4e-7 warm
/// (references.md, 2026-10-13) — the per-point f32 path scatter, so the bound is 5× rather
/// than 3× the achieved: 5e-6.
const SHAPE_BOUND: f64 = 5e-6;

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test gpu_vs_cpu")]
fn h2_curve_shape_matches() {
    let Some(ctx) = gpu() else { return };
    let grid = Grid::cube(49, 0.25);
    let settings = ScfSettings::for_grid(&grid);
    let rs = [1.0, 1.4, 1.8, 2.4, 3.2];
    let mut s = GpuSolver::new(ctx, &grid);
    let mut e_cpu = Vec::new();
    let mut e_cold = Vec::new();
    let mut e_warm = Vec::new();
    let mut steps = Vec::new();
    for &r in &rs {
        let sys = h2(r);
        let (_, cpu, _) = cpu_solve(&grid, &sys, &settings, &format!("H₂ R = {r}"));
        let (cold, _, _) = gpu_solve(&mut s, &sys, &settings, false, &format!("H₂ R = {r} cold"));
        e_cpu.push(cpu.energy.total);
        e_cold.push(cold.energy.total);
        steps.push((cpu.iterations, cold.iterations));
    }
    // the MD path: warm along R from the R = 1.0 state (the solver holds the last cold state, R = 3.2 — start again)
    let mut warm_steps = Vec::new();
    for (i, &r) in rs.iter().enumerate() {
        let (st, _, _) = gpu_solve(&mut s, &h2(r), &settings, i > 0, &format!("H₂ R = {r} warm"));
        e_warm.push(st.energy.total);
        warm_steps.push(st.iterations);
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let d_cold: Vec<f64> = e_cold.iter().zip(&e_cpu).map(|(g, c)| g - c).collect();
    let d_warm: Vec<f64> = e_warm.iter().zip(&e_cpu).map(|(g, c)| g - c).collect();
    let (m_cold, m_warm) = (mean(&d_cold), mean(&d_warm));
    let mut out = String::from("H₂ E(R), 49³, σ = h = 0.25, tol 1e-6 (Ha)\n| R | E_cpu | E_gpu cold | ΔE cold | ΔE − mean | E_gpu warm along R | ΔE warm | ΔE − mean | steps cpu / cold / warm |\n|---|---|---|---|---|---|---|---|---|\n");
    let (mut worst_cold, mut worst_warm) = (0.0f64, 0.0f64);
    for i in 0..rs.len() {
        let (sc, sw) = (d_cold[i] - m_cold, d_warm[i] - m_warm);
        worst_cold = worst_cold.max(sc.abs());
        worst_warm = worst_warm.max(sw.abs());
        let _ = writeln!(out, "| {} | {:+.9} | {:+.9} | {:+.2e} | {sc:+.2e} | {:+.9} | {:+.2e} | {sw:+.2e} | {} / {} / {} |",
                         rs[i], e_cpu[i], e_cold[i], d_cold[i], e_warm[i], d_warm[i], steps[i].0, steps[i].1, warm_steps[i]);
    }
    let _ = writeln!(out, "mean ΔE: cold {m_cold:+.2e}, warm {m_warm:+.2e}; max |ΔE − mean|: cold {worst_cold:.2e}, warm {worst_warm:.2e} (bound {SHAPE_BOUND:.0e})");
    // a binding curve on every path: E(1.4) < E(1.0) (repulsive wall) and E(3.2) > E(1.8) (the
    // σ-shifted minimum is near R = 1.79 at σ = 0.25, so E(1.8) is the lowest sample)
    println!("{out}");
    assert!(worst_cold <= SHAPE_BOUND, "cold shape mismatch {worst_cold:.3e}");
    assert!(worst_warm <= SHAPE_BOUND, "warm-along-R shape mismatch {worst_warm:.3e}");
    for e in [&e_cpu, &e_cold, &e_warm] { assert!(e[1] < e[0] && e[4] > e[2], "not a binding curve: {e:?}"); }
}

/// One timing row.
struct TimingRow { system: &'static str, n: usize, label: &'static str, stats: ScfStats, prof: SolveProfile, wall: Duration }

impl TimingRow {
    fn line(&self) -> String {
        let p = &self.prof;
        let steps = self.stats.iterations as f64;
        let total = ms(self.wall);
        let (po, ei, st) = (ms(p.poisson_wall), ms(p.eigen_wall), ms(p.start_wall));
        let rest = total - po - ei - st;
        format!("| {} | {}³ | {} | {} | {total:.0} | {:.1} | {po:.0} ({:.0} %) | {ei:.0} ({:.0} %) | {st:.0} | {rest:.0} | {} ({:.0}/step) | {} | {} | {:+.8} |",
                self.system, self.n, self.label, self.stats.iterations, total / steps, 100.0 * po / total, 100.0 * ei / total,
                p.readbacks, p.readbacks as f64 / steps, p.poisson_iterations, p.eigen_iterations, self.stats.energy.total)
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test gpu_vs_cpu")]
fn timing_table() {
    let Some(ctx) = gpu() else { return };
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner()); // the whole table is uncontended
    let adapter = ctx.adapter.get_info();
    let mut rows: Vec<TimingRow> = Vec::new();
    let mut setup = String::new();
    let mut run = |system: &'static str, n: usize, make: &dyn Fn(f64) -> System| {
        let grid = Grid::cube(n, 0.25);
        let settings = ScfSettings::for_grid(&grid);
        let t = Instant::now();
        let mut s = GpuSolver::new(ctx.clone(), &grid);
        let _ = writeln!(setup, "- {system} {n}³: `GpuSolver::new` {:.0} ms (kernels, Poisson ghost table)", ms(t.elapsed()));
        let solve = |s: &mut GpuSolver, sys: &System, warm: bool, label: &'static str| -> TimingRow {
            let t = Instant::now();
            let stats = s.solve(sys, &settings, warm).unwrap_or_else(|e| panic!("{system} {n}³ {label}: {e}")).1;
            TimingRow { system, n, label, stats, prof: s.profile().clone(), wall: t.elapsed() }
        };
        rows.push(solve(&mut s, &make(0.0), false, "first (cold, incl. warm-up)"));
        rows.push(solve(&mut s, &make(0.0), false, "cold"));
        rows.push(solve(&mut s, &make(0.02), true, "warm (shifted 0.02 Bohr)"));
    };
    for n in [33, 49, 65, 97] { run("H₂ R=1.4", n, &|shift| h2_at(1.4, shift)); }
    for n in [57, 97] { run("H₂O", n, &h2o_at); }
    drop(run);

    let mut md = String::new();
    let _ = writeln!(md, "# GPU timing table (`validate::gpu_vs_cpu::timing_table`, {} UTC)\n", chrono_free_date());
    let _ = writeln!(md, "Adapter: {} ({:?}, {:?}); release build; σ = h = 0.25, tol 1e-6 (`ScfSettings::for_grid`).", adapter.name, adapter.backend, adapter.device_type);
    let _ = writeln!(md, "**Contention:** the test cannot tell how it was invoked. Run as part of the suite (`--test gpu_vs_cpu`) the GPU is uncontended (the `GPU` lock) but three CPU reference solves run alongside, and the rows are 5–20 % slower than a solo run (`--test gpu_vs_cpu timing_table`), which is the mode of record in `references.md`. Treat this file as \"in-suite (contended)\" unless you ran the test alone.");
    let _ = writeln!(md, "\"first\" is the first solve on a fresh `GpuSolver` (pipeline warm-up, pool growth); \"cold\" the next cold solve of the same geometry; \"warm\" a warm start from it on the molecule shifted by 0.02 Bohr (the MD step). Phase walls from `SolveProfile`: Poisson = inside `GpuPoisson::solve` (every SCF step + the energy), LOBPCG = inside `lobpcg`, start = building/uploading the cold-start guess (CPU f64), rest = pointwise work, DIIS, the SCF readbacks and the energy.\n");
    md += &setup;
    let _ = writeln!(md, "\n| system | grid | solve | steps | wall ms | ms/step | Poisson ms (share) | LOBPCG ms (share) | start ms | rest ms | readbacks (per step) | CG it | LOBPCG it | E (Ha) |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for r in &rows { md += &r.line(); md.push('\n'); }

    // plan targets and the Task 10 trigger
    let find = |system: &str, n: usize, label: &str| rows.iter().find(|r| r.system == system && r.n == n && r.label == label).expect("row");
    let h2_cold = find("H₂ R=1.4", 49, "cold");
    let h2_warm = find("H₂ R=1.4", 49, "warm (shifted 0.02 Bohr)");
    let h2o_cold = find("H₂O", 97, "cold");
    let h2_97 = find("H₂ R=1.4", 97, "cold");
    let share = |r: &TimingRow| ms(r.prof.poisson_wall) / ms(r.wall);
    let _ = writeln!(md, "\n## Plan targets (reported, not asserted)\n");
    let _ = writeln!(md, "- 49³ H₂ cold: {:.0} ms (target ≤ 500) — {}", ms(h2_cold.wall), if ms(h2_cold.wall) <= 500.0 { "met" } else { "missed" });
    let _ = writeln!(md, "- 49³ H₂ warm: {:.0} ms (target ≤ 150) — {}", ms(h2_warm.wall), if ms(h2_warm.wall) <= 150.0 { "met" } else { "missed" });
    let _ = writeln!(md, "- 97³ H₂O cold: {:.0} ms (target ≤ 5000) — {}", ms(h2o_cold.wall), if ms(h2o_cold.wall) <= 5000.0 { "met" } else { "missed" });
    let _ = writeln!(md, "\n## Task 10 (multigrid Poisson) trigger\n");
    let _ = writeln!(md, "- Poisson share of a 97³ step: H₂ {:.0} %, H₂O {:.0} % (trigger ≥ 40 %)", 100.0 * share(h2_97), 100.0 * share(h2o_cold));
    let _ = writeln!(md, "- 97³ H₂O cold solve: {:.2} s (trigger > 5 s)", h2o_cold.wall.as_secs_f64());
    let triggered = share(h2_97) >= 0.4 || share(h2o_cold) >= 0.4 || h2o_cold.wall.as_secs_f64() > 5.0;
    let _ = writeln!(md, "- **Task 10 trigger (historical; targets now met): {}**", if triggered { "TRIGGERED" } else { "not triggered" });
    let _ = writeln!(md, "- LOBPCG share of the same steps: H₂ {:.0} %, H₂O {:.0} % — what a Poisson-only change cannot touch",
                     100.0 * ms(h2_97.prof.eigen_wall) / ms(h2_97.wall), 100.0 * ms(h2o_cold.prof.eigen_wall) / ms(h2o_cold.wall));

    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/last-run-gpu.md");
    std::fs::write(path, &md).unwrap_or_else(|e| panic!("writing {path}: {e}"));
    println!("{md}\n(written to {path})");
}

/// Today's date from the system clock without a date crate: days since the Unix epoch →
/// civil date (Howard Hinnant's algorithm), UTC.
fn chrono_free_date() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let z = (secs / 86_400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}
