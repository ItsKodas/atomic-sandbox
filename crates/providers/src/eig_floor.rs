//! Measurement (report-only, `#[ignore]`d): the cold Fermi–Dirac solve of a sampling pass's water sample at the app's
//! Estimated-mode box — the H₂O preset geometry (O at the origin, H at (±1.43, 1.11, 0)) and two pass-like
//! perturbations of it, on 119³ × 0.15 (17.7 Bohr), `Accepted` (pseudised O and H through the GPU dimer gate), kT 5 mHa,
//! the pass's teacher tolerance 1e-6 (`branch_jump::SAMPLE_TOL`, raised by `effective_gpu_tol` under the setup: 1e-6
//! for 8 valence electrons) — what a "Quick look" pass solves 158 times. Each geometry is solved cold on a fresh
//! solver, then re-solved warm at the same geometry (the f32 noise level of the answer: a warm/cold pair converges to
//! the same fixed point from two directions). Prints per solve: the wall, the SCF steps, the eigen wall and iterations,
//! the floor accepts and cap hits, the witness growths and peak block, the per-step eigen tols and Δρ, F and the forces.
//! Run with `SCF_TRACE=1` for the per-call LOBPCG lines (`qm_gpu::scf::Channel::solve_fd`).
//!
//! `cargo test -p providers --release eig_floor -- --ignored --nocapture`
use crate::ks::problem;
use sim::{Cell, IonicPolicy, Nucleus, Scene};
use std::time::Instant;

/// The app's Estimated-mode box at the Accurate spacing (the user's scene: 119³ × 0.15).
const CELL: Cell = Cell { n: 119, h: 0.15 };

fn water(d: [[f64; 3]; 3]) -> Scene {
    let base = [[0.0, 0.0, 0.0], [1.43, 1.11, 0.0], [-1.43, 1.11, 0.0]];
    let z = [8u8, 1, 1];
    Scene {
        nuclei: (0..3).map(|i| Nucleus::new(z[i], std::array::from_fn(|k| base[i][k] + d[i][k]))).collect(),
        cell: CELL,
        charge: 0,
        multiplicity: None,
        sigma: None,
        external_field: [0.0; 3],
        ionic: IonicPolicy::Accepted,
        train_mask: None,
    }
}

struct Out {
    f: f64,
    forces: Vec<[f64; 3]>,
}

fn solve_once(s: &mut qm_gpu::GpuSolver, scene: &Scene, warm: bool, label: &str) -> Out {
    let mut notes = Vec::new();
    let (_, system, mut settings) = problem(scene, crate::branch_jump::SAMPLE_TOL, &mut notes).expect("a valid scene");
    assert!(system.ionic.is_some(), "Accepted pseudises O and H on the GPU path");
    settings.tol = crate::pseudo::effective_gpu_tol(settings.tol, (system.n_up + system.n_dn) as i64);
    let t = Instant::now();
    let (_, st) = s.solve(&system, &settings, warm).unwrap_or_else(|e| panic!("{label}: {e}"));
    let wall = t.elapsed().as_secs_f64();
    let state = s.state().expect("converged");
    let forces = s.forces(state, &system.nuclei, settings.sigma).expect("forces");
    let p = s.profile();
    println!(
        "{label}: {wall:.1} s, {} SCF steps (tol {:.1e}, tol_used {:.1e}), F {:+.8} Ha, eigen {:.1} s / {} it / {} readbacks, poisson {:.1} s, \
         floor accepts {} (worst {:.2e}), cap hits {} (worst {:.2e}), witness growths {}, residual growths {}, lower witnesses {}, buffer growths {}, \
         peak block {}, locked calls {}, class raises {}, max eigen residual {:.2e}, guard {:.2e}, vectors ↑ {} ↓ {}",
        st.iterations, settings.tol, st.tol_used, st.energy.total, p.eigen_wall.as_secs_f64(), p.eigen_iterations, p.eigen_readbacks,
        p.poisson_wall.as_secs_f64(), p.eigen_floor_accepts, p.eigen_floor_worst, p.eigen_cap_hits, p.eigen_cap_worst, p.witness_growths,
        p.witness_residual_growths, p.lower_witnesses, p.buffer_growths, p.peak_block, p.locked_calls, p.class_raises, st.max_eigen_residual, st.max_guard_residual,
        state.orbitals_up.len(), state.orbitals_dn.len()
    );
    println!("    eig tol per step {:?}", p.eig_tol_steps.iter().map(|d| format!("{d:.1e}")).collect::<Vec<_>>());
    println!("    Δρ per step      {:?}", p.delta_rho_steps.iter().map(|d| format!("{d:.1e}")).collect::<Vec<_>>());
    for n in &p.notes {
        println!("    note: {n}");
    }
    for w in &st.warnings {
        println!("    warning: {w}");
    }
    println!("    forces {:?}", forces.iter().map(|f| format!("({:+.6e}, {:+.6e}, {:+.6e})", f[0], f[1], f[2])).collect::<Vec<_>>());
    Out { f: st.energy.total, forces }
}

fn max_force_diff(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (0..3).map(|k| (x[k] - y[k]).powi(2)).sum::<f64>().sqrt()).fold(0.0, f64::max)
}

/// The whole "Quick look" pass (30 perturbations σ 0.10, the three pair scans × 8 points over 0.7–1.8 r₀, a 100-step
/// 800 K burst; seed 1) on the H₂O preset at 119³ × 0.15, `Accepted`, with the app's estimated-mode teacher (`GpuKs` at
/// `SAMPLE_TOL`, the teacher's half of a 60 % share of a 24 GB card) inside a `SurrogateKs`, driven step by step as the
/// worker does (`Sampler::step_own_teacher`). Prints each step's wall and stage; a `=== pass step` marker goes to stderr
/// before each step so the `SCF_TRACE` lines can be attributed. Report-only.
#[test]
#[ignore = "report-only measurement (GPU, 119³ × 0.15, ~20+ min): cargo test -p providers --release quick_look_pass -- --ignored --nocapture"]
fn quick_look_pass_wall_per_sample() {
    let Some(ctx) = gpu::GpuContext::new_headless() else {
        eprintln!("eig_floor: no GPU adapter — skipped");
        return;
    };
    crate::pseudo::set_dimer_backend(crate::pseudo::DimerBackend::Gpu(ctx.clone()));
    let _g = crate::ks::tests::GPU.lock().unwrap_or_else(|e| e.into_inner());
    let budget = (24e9 * 0.6 / 2.0) as u64;
    let teacher = crate::GpuKs::new(ctx.clone(), crate::branch_jump::SAMPLE_TOL).with_vram_budget(budget);
    let mut sur = crate::SurrogateKs::new(Box::new(teacher), crate::FallbackPolicy::default()).with_teacher_tol(crate::branch_jump::SAMPLE_TOL);
    let plan = sim::SamplingPlan { perturbations: 30, perturbation_sigma: 0.10, hot_burst_steps: 100, hot_burst_t: 800.0, scan_points_per_pair: 8, seed: 1, ..sim::SamplingPlan::default() };
    let scene = water([[0.0; 3]; 3]);
    let mut s = crate::Sampler::with_cutoff(&scene, &plan, sur.cutoff()).expect("a valid plan");
    let t_pass = Instant::now();
    let mut walls: Vec<(usize, &'static str, f64)> = Vec::new();
    for i in 0.. {
        eprintln!("=== pass step {i}");
        let t = Instant::now();
        match s.step_own_teacher(&mut sur, &mut || false) {
            Ok(Some(p)) => {
                let w = t.elapsed().as_secs_f64();
                println!("step {i:>3}: {w:>7.1} s  {} {}/{} (failed {}, skipped {}) E {:.6}", p.stage, p.done, p.total, p.failed, p.skipped, p.last_energy);
                walls.push((i, p.stage, w));
            }
            Ok(None) => break,
            Err(e) => {
                println!("step {i}: error {e}");
                break;
            }
        }
    }
    let total = t_pass.elapsed().as_secs_f64();
    let mut sorted: Vec<f64> = walls.iter().map(|w| w.2).collect();
    sorted.sort_by(f64::total_cmp);
    let median = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
    println!("pass: {} steps, {total:.0} s, median step {median:.1} s, slowest {:?}", walls.len(),
             { let mut v = walls.clone(); v.sort_by(|a, b| b.2.total_cmp(&a.2)); v.truncate(10); v });
    for n in s.take_notes() {
        println!("note: {n}");
    }
}

/// Six pass-like geometries: the preset, then five with every coordinate displaced by an independent Gaussian of
/// width 0.10 Bohr (the "Quick look" σ), from a fixed LCG seed.
fn chain_geometries() -> Vec<[[f64; 3]; 3]> {
    let mut s = 0x5EED_u64;
    let mut uni = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut gauss = || (-2.0 * uni().ln()).sqrt() * (2.0 * std::f64::consts::PI * uni()).cos();
    let mut out = vec![[[0.0; 3]; 3]];
    for _ in 0..5 {
        out.push(std::array::from_fn(|_| std::array::from_fn(|_| 0.10 * gauss())));
    }
    out
}

/// The pass's warm path (`Sampler` module doc: `discontinuity` before each perturbation — warm orbitals kept, no
/// density extrapolation): the preset cold, then each perturbed geometry warm from the previous one, on one solver; and
/// every geometry cold on a fresh solver as the reference. Prints the walls and, per geometry, the chain's answer
/// against the cold one (|ΔF| in Ha, max |ΔF_atom| in Ha/Bohr): two converged answers from different starts, i.e. the
/// noise level of the answer at this tolerance. Report-only (`SCF_TRACE=1` for the per-call lines).
#[test]
#[ignore = "report-only measurement (GPU, 119³ × 0.15, ~5 min): cargo test -p providers --release warm_chain -- --ignored --nocapture"]
fn h2o_estimated_box_warm_chain() {
    let Some(ctx) = gpu::GpuContext::new_headless() else {
        eprintln!("eig_floor: no GPU adapter — skipped");
        return;
    };
    crate::pseudo::set_dimer_backend(crate::pseudo::DimerBackend::Gpu(ctx.clone()));
    let _g = crate::ks::tests::GPU.lock().unwrap_or_else(|e| e.into_inner());
    let grid = qm::Grid::cube(CELL.n, CELL.h);
    let geoms = chain_geometries();
    eprintln!("=== chain");
    let mut chain = Vec::new();
    let mut s = qm_gpu::GpuSolver::new(ctx.clone(), &grid);
    let t_chain = Instant::now();
    for (i, d) in geoms.iter().enumerate() {
        eprintln!("=== chain geometry {i}");
        chain.push(solve_once(&mut s, &water(*d), i > 0, &format!("chain {i} {}", if i > 0 { "warm" } else { "cold" })));
    }
    let chain_wall = t_chain.elapsed().as_secs_f64();
    eprintln!("=== cold references");
    let mut worst = (0.0f64, 0.0f64);
    for (i, d) in geoms.iter().enumerate() {
        eprintln!("=== cold geometry {i}");
        let mut s = qm_gpu::GpuSolver::new(ctx.clone(), &grid);
        let cold = solve_once(&mut s, &water(*d), false, &format!("cold {i}"));
        let (df, dfa) = ((chain[i].f - cold.f).abs(), max_force_diff(&chain[i].forces, &cold.forces));
        worst = (worst.0.max(df), worst.1.max(dfa));
        println!("geometry {i}: chain − cold |ΔF| {df:.2e} Ha, max |ΔF_atom| {dfa:.2e} Ha/Bohr\n");
    }
    println!("chain wall {chain_wall:.1} s; worst chain − cold |ΔF| {:.2e} Ha, max |ΔF_atom| {:.2e} Ha/Bohr", worst.0, worst.1);
    for (i, c) in chain.iter().enumerate() {
        println!("chain {i}: F {:+.9} forces {:?}", c.f, c.forces.iter().map(|f| format!("({:+.7e}, {:+.7e}, {:+.7e})", f[0], f[1], f[2])).collect::<Vec<_>>());
    }
}

/// Module doc. Prints; asserts only that every solve converges.
#[test]
#[ignore = "report-only measurement (GPU, 119³ × 0.15, minutes): cargo test -p providers --release eig_floor -- --ignored --nocapture"]
fn h2o_estimated_box_cold_solve_profile() {
    let Some(ctx) = gpu::GpuContext::new_headless() else {
        eprintln!("eig_floor: no GPU adapter — skipped");
        return;
    };
    crate::pseudo::set_dimer_backend(crate::pseudo::DimerBackend::Gpu(ctx.clone()));
    let _g = crate::ks::tests::GPU.lock().unwrap_or_else(|e| e.into_inner());
    let geometries: [(&str, [[f64; 3]; 3]); 3] = [
        ("preset", [[0.0; 3]; 3]),
        ("perturbed a", [[0.03, -0.02, 0.01], [-0.06, 0.04, 0.05], [0.05, -0.03, -0.04]]),
        ("perturbed b", [[-0.02, 0.03, -0.02], [0.08, -0.05, 0.02], [-0.04, 0.07, 0.06]]),
    ];
    let grid = qm::Grid::cube(CELL.n, CELL.h);
    for (name, d) in geometries {
        let scene = water(d);
        let mut s = qm_gpu::GpuSolver::new(ctx.clone(), &grid);
        let cold = solve_once(&mut s, &scene, false, &format!("{name} cold"));
        let warm = solve_once(&mut s, &scene, true, &format!("{name} warm re-solve"));
        println!(
            "{name}: warm − cold |ΔF| {:.2e} Ha, max |ΔF_atom| {:.2e} Ha/Bohr (the f32 noise of the answer)\n",
            (warm.f - cold.f).abs(),
            max_force_diff(&warm.forces, &cold.forces)
        );
    }
}
