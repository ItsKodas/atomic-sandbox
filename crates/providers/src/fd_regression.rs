//! V2 (spec 2026-11-13 §9 "V2 cases", amended by plan Task 3 step 0 at kT = 5 mHa): the SCF-stall regression, the
//! diagnosis harness reduced and committed (plan Task 9). The prototype's diagnosis harness (on branch
//! `smearing-prototype`, never merged) found loose hand-built H clusters that stalled under integer / shared-shell
//! occupations; Fermi–Dirac at kT = 5 mHa converges every one of them. Each case here goes the app's way: through
//! `providers::ks::problem` (FD on every scene), cold, at 81³ × 0.15 (a 12 Bohr box, Accurate), the app's interactive
//! tolerance 1e-5 raised by `effective_gpu_tol` under an ionic setup (as `GpuKs`; 1e-5 for every case here, N ≤ 26), the
//! stall stop on at its default window, the 200-step cap.
//!
//! - **Must converge** (no stall stop, no step cap, no witness cap: any of them is an `Err`) within the case's budget,
//!   ⌈1.5 × the 5 mHa count⌉ rounded up to a multiple of 5.
//! - **F and TS** within 1e-5 Ha of the amended table (printed to six decimals; TS = E − F of the printed values, so
//!   ±1e-6 of rounding). A mismatch is investigated, never re-pinned blindly.
//! - Geometries: D§2.1's, centred on their centroid; the seeded clusters are the prototype's generator
//!   ([`random_cluster`]), the ring its vertex phase (vertex 0 on +x).
//! - "default" is the app's `Accepted` policy with the GPU dimer backend (pseudised H: the dimer gate runs once per
//!   process, ≈ 1 min); "`ionic none`" is all-electron.
//! - All-electron H₂O + 2 H (434 s of GPU at 5 mHa in A1) is its own test function, run as its own GPU batch, and logs
//!   the extra (witness) vectors each channel grew.
//!
//! GPU, release-only; the tests skip without an adapter. Each takes the ks tests' GPU lock (one solve at a time).
use crate::ks::problem;
use sim::{Cell, IonicPolicy, Nucleus, Scene};
use std::time::Instant;

/// The V2 grid: 81³ × 0.15 (12 Bohr), the Accurate presets' box.
const CELL: Cell = Cell { n: 81, h: 0.15 };

/// The app's interactive tolerance (`effective_gpu_tol` leaves it at 1e-5 for every case: N ≤ 26 valence electrons).
const APP_TOL: f64 = 1e-5;

/// The pins' tolerance on F and on TS (spec §9 "V2 cases").
const PIN_TOL: f64 = 1e-5;

/// One V2 case: its name, its atoms (Z, position before centring), the policy, the budget, and the 5 mHa pins (F, TS).
struct Case {
    name: &'static str,
    atoms: Vec<(u8, [f64; 3])>,
    policy: IonicPolicy,
    budget: usize,
    f: f64,
    ts: f64,
}

/// The diagnosis harness's "loosely bound" random H cluster (its `random_cluster` on branch `smearing-prototype`,
/// verbatim): `n` atoms, each attached to a random earlier one at a distance in [dmin, dmax], kept when no atom is
/// closer than dmin. A 64-bit LCG with the harness's constants, so the seeds give the harness's geometries.
fn random_cluster(n: usize, seed: u64, dmin: f64, dmax: f64) -> Vec<(u8, [f64; 3])> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut rnd = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut p: Vec<[f64; 3]> = vec![[0.0; 3]];
    while p.len() < n {
        let a = p[(rnd() * p.len() as f64) as usize % p.len()];
        let (ct, ph) = (2.0 * rnd() - 1.0, 2.0 * std::f64::consts::PI * rnd());
        let st = (1.0 - ct * ct).sqrt();
        let d = dmin + (dmax - dmin) * rnd();
        let q = [a[0] + d * st * ph.cos(), a[1] + d * st * ph.sin(), a[2] + d * ct];
        if p.iter().all(|x| ((x[0] - q[0]).powi(2) + (x[1] - q[1]).powi(2) + (x[2] - q[2]).powi(2)).sqrt() >= dmin) {
            p.push(q);
        }
    }
    p.into_iter().map(|x| (1u8, x)).collect()
}

/// The harness's seeded cluster (dmin 1.5, dmax 2.5 Bohr).
fn seeded(n: usize, seed: u64) -> Vec<(u8, [f64; 3])> {
    random_cluster(n, seed, 1.5, 2.5)
}

/// A regular `n`-gon of H with side `side` in the xy plane, vertex 0 on +x (radius side / (2 sin(π/n))).
fn ring(n: usize, side: f64) -> Vec<(u8, [f64; 3])> {
    let r = side / (2.0 * (std::f64::consts::PI / n as f64).sin());
    (0..n).map(|i| {
        let t = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
        (1u8, [r * t.cos(), r * t.sin(), 0.0])
    }).collect()
}

fn h(p: [f64; 3]) -> (u8, [f64; 3]) {
    (1, p)
}

fn rhombus() -> Vec<(u8, [f64; 3])> {
    vec![h([0.0, 0.0, 0.0]), h([1.8, 0.0, 0.0]), h([2.7, 1.5588, 0.0]), h([0.9, 1.5588, 0.0])]
}

fn lopsided_tetra() -> Vec<(u8, [f64; 3])> {
    vec![h([0.0, 0.0, 0.0]), h([1.9, 0.0, 0.0]), h([0.7, 1.7, 0.0]), h([0.9, 0.6, 1.8])]
}

fn water() -> Vec<(u8, [f64; 3])> {
    vec![(8, [0.0, 0.0, 0.0]), h([1.43, 1.11, 0.0]), h([-1.43, 1.11, 0.0])]
}

/// The harness's scene: the atoms centred on their centroid, neutral, automatic multiplicity, σ = h, no field.
fn scene(atoms: &[(u8, [f64; 3])], policy: IonicPolicy) -> Scene {
    let n = atoms.len() as f64;
    let c: [f64; 3] = std::array::from_fn(|k| atoms.iter().map(|a| a.1[k]).sum::<f64>() / n);
    Scene {
        nuclei: atoms.iter().map(|&(z, p)| Nucleus::new(z, [p[0] - c[0], p[1] - c[1], p[2] - c[2]])).collect(),
        cell: CELL,
        charge: 0,
        multiplicity: None,
        sigma: None,
        external_field: [0.0; 3],
        ionic: policy,
        train_mask: None,
    }
}

/// What one cold solve gave: steps, F, TS, the vectors each channel solved (occupied + witnesses) and the electrons.
struct Solved {
    iterations: usize,
    f: f64,
    ts: f64,
    tol_used: f64,
    vectors: [usize; 2],
    electrons: [usize; 2],
    wall: f64,
}

/// One case the app's way (module doc): `problem`, the GPU stop raised by `effective_gpu_tol` under a setup, a cold
/// solve on a fresh solver. `Err` carries the solver's error text (a stall stop, the cap or the witness cap).
fn solve(ctx: &gpu::GpuContext, atoms: &[(u8, [f64; 3])], policy: IonicPolicy) -> Result<Solved, String> {
    let sc = scene(atoms, policy);
    let mut notes = Vec::new();
    let (grid, system, mut settings) = problem(&sc, APP_TOL, &mut notes).map_err(|e| e.to_string())?;
    assert_eq!(settings.occupations, qm::Occupations::FermiDirac { kt: qm::ELECTRON_KT });
    assert_eq!(settings.stall_window, qm::SCF_STALL_WINDOW, "the stall stop is on");
    if system.ionic.is_some() {
        settings.tol = crate::pseudo::effective_gpu_tol(settings.tol, (system.n_up + system.n_dn) as i64);
    }
    assert_eq!(settings.tol, APP_TOL, "effective_gpu_tol leaves 1e-5 for every V2 case");
    let mut s = qm_gpu::GpuSolver::new(ctx.clone(), &grid);
    let t = Instant::now();
    let (_, st) = s.solve(&system, &settings, false).map_err(|e| e.to_string())?;
    let wall = t.elapsed().as_secs_f64();
    let state = s.state().expect("a converged state");
    Ok(Solved {
        iterations: st.iterations,
        f: st.energy.total,
        ts: st.energy.ts,
        tol_used: st.tol_used,
        vectors: [state.orbitals_up.len(), state.orbitals_dn.len()],
        electrons: [system.n_up, system.n_dn],
        wall,
    })
}

/// The GPU and the dimer backend for the `Accepted` cases (H pseudised through the GPU dimer gate), or `None` without an
/// adapter.
fn gpu_for(policy_needs_gate: bool) -> Option<gpu::GpuContext> {
    let Some(ctx) = gpu::GpuContext::new_headless() else {
        eprintln!("fd_regression: no GPU adapter — skipped");
        return None;
    };
    if policy_needs_gate {
        crate::pseudo::set_dimer_backend(crate::pseudo::DimerBackend::Gpu(ctx.clone()));
    }
    Some(ctx)
}

/// Runs `cases` one at a time and checks each against the module doc's rules; every failure is collected and reported
/// together, so one run shows them all.
fn run(ctx: &gpu::GpuContext, cases: &[Case]) {
    let mut misses = Vec::new();
    for c in cases {
        match solve(ctx, &c.atoms, c.policy) {
            Ok(s) => {
                let (df, dts) = (s.f - c.f, s.ts - c.ts);
                println!(
                    "V2 {:<26} {:?}: {} steps (budget {}), F {:+.6} (pin {:+.6}, Δ {df:+.1e}), TS {:.6} (pin {:.6}, Δ {dts:+.1e}), tol_used {:.0e}, vectors ↑ {}+{} ↓ {}+{}, {:.1} s",
                    c.name, c.policy, s.iterations, c.budget, s.f, c.f, s.ts, c.ts, s.tol_used, s.electrons[0], s.vectors[0] - s.electrons[0], s.electrons[1],
                    s.vectors[1] - s.electrons[1], s.wall
                );
                if s.iterations > c.budget {
                    misses.push(format!("{}: {} steps > budget {}", c.name, s.iterations, c.budget));
                }
                if !(df.abs() <= PIN_TOL) {
                    misses.push(format!("{}: F {:.7} vs pin {:.6} (Δ {df:+.2e})", c.name, s.f, c.f));
                }
                if !(dts.abs() <= PIN_TOL) {
                    misses.push(format!("{}: TS {:.7} vs pin {:.6} (Δ {dts:+.2e})", c.name, s.ts, c.ts));
                }
            }
            Err(e) => {
                println!("V2 {:<26} {:?}: FAILED: {e}", c.name, c.policy);
                misses.push(format!("{}: {e}", c.name));
            }
        }
    }
    assert!(misses.is_empty(), "V2 misses:\n  {}", misses.join("\n  "));
}

/// V2, the default policy (`Accepted`, pseudised H): the five clusters that stalled before FD (or, seeds 1, 3 and 8,
/// that carry a partial level at 5 mHa) converge within their budgets on the amended pins.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only (GPU, 81³ × 0.15); cargo test -p providers --release fd_regression")]
fn fd_converges_default_policy_stallers() {
    let Some(ctx) = gpu_for(true) else { return };
    let _g = crate::ks::tests::GPU.lock().unwrap_or_else(|e| e.into_inner());
    let a = IonicPolicy::Accepted;
    run(&ctx, &[
        Case { name: "H₄ rhombus", atoms: rhombus(), policy: a, budget: 20, f: -2.079322, ts: 0.000027 },
        Case { name: "H₄ lopsided tetra", atoms: lopsided_tetra(), policy: a, budget: 25, f: -1.943830, ts: 0.013889 },
        Case { name: "H₄ seed 1", atoms: seeded(4, 1), policy: a, budget: 20, f: -2.078602, ts: 0.002076 },
        Case { name: "H₅ seed 3", atoms: seeded(5, 3), policy: a, budget: 25, f: -2.564249, ts: 0.003006 },
        Case { name: "H₈ seed 8", atoms: seeded(8, 8), policy: a, budget: 25, f: -4.220394, ts: 0.000163 },
    ]);
}

/// V2, all-electron (`ionic none`): the H₅ ring (radius 1.531, its pair exactly degenerate: TS = 2 kT ln 2), H₆ seed 5
/// and the lopsided tetrahedron converge within their budgets on the amended pins.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only (GPU, 81³ × 0.15); cargo test -p providers --release fd_regression")]
fn fd_converges_all_electron_failures() {
    let Some(ctx) = gpu_for(false) else { return };
    let _g = crate::ks::tests::GPU.lock().unwrap_or_else(|e| e.into_inner());
    let r = ring(5, 1.8);
    let radius = (r[0].1[0].powi(2) + r[0].1[1].powi(2)).sqrt();
    assert!((radius - 1.531).abs() < 5e-4, "the H₅ ring's radius {radius}");
    let none = IonicPolicy::None;
    run(&ctx, &[
        Case { name: "H₅ ring", atoms: r, policy: none, budget: 20, f: -2.470682, ts: 0.006932 },
        Case { name: "H₆ seed 5", atoms: seeded(6, 5), policy: none, budget: 25, f: -2.973255, ts: 0.009715 },
        Case { name: "H₄ lopsided tetra", atoms: lopsided_tetra(), policy: none, budget: 25, f: -1.824952, ts: 0.013978 },
    ]);
}

/// V2, all-electron H₂O + 2 H, its own GPU batch (spec §9 "V2 cases": 434 s at 5 mHa in A1): converged within 50
/// steps on the amended pins; the extra (witness) vectors each channel grew are printed with the solve.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only (GPU, 81³ × 0.15, ≈ 7 min); cargo test -p providers --release fd_converges_all_electron_h2o_plus_2h")]
fn fd_converges_all_electron_h2o_plus_2h() {
    let Some(ctx) = gpu_for(false) else { return };
    let _g = crate::ks::tests::GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut atoms = water();
    atoms.extend([h([0.5, -2.2, 1.0]), h([-0.9, -2.0, -1.6])]);
    run(&ctx, &[Case { name: "H₂O + 2 H", atoms, policy: IonicPolicy::None, budget: 50, f: -47.350019, ts: 0.013163 }]);
}

/// V2's controls [SHOULD-6]: gapped H₂ and H₂O under the default policy keep the D§6 energies within the suite's rule
/// 3 × max(achieved, path scatter): H₂ 2e-6 Ha, H₂O 6e-6 Ha (FD changes the block, so the bits, not the digits).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only (GPU, 81³ × 0.15); cargo test -p providers --release fd_regression")]
fn fd_controls_keep_their_energies() {
    let Some(ctx) = gpu_for(true) else { return };
    let _g = crate::ks::tests::GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut misses = Vec::new();
    for (name, atoms, pin, bound) in [("H₂", vec![h([0.0; 3]), h([1.434, 0.0, 0.0])], -1.136119, 2e-6), ("H₂O", water(), -17.476023, 6e-6)] {
        let s = solve(&ctx, &atoms, IonicPolicy::Accepted).unwrap_or_else(|e| panic!("{name}: {e}"));
        let d = s.f - pin;
        println!("V2 control {name}: {} steps, F {:+.7} (pin {pin:+.6}, Δ {d:+.1e}, bound {bound:.0e}), TS {:.1e}, {:.1} s", s.iterations, s.f, s.ts, s.wall);
        if !(d.abs() <= bound) {
            misses.push(format!("{name}: F {:.7} vs {pin:.6} (Δ {d:+.2e} > {bound:.0e})", s.f));
        }
        if !(s.ts <= 1e-9) {
            misses.push(format!("{name}: TS {:.2e} on a gapped control", s.ts));
        }
    }
    assert!(misses.is_empty(), "{misses:?}");
}

/// The generator is the harness's: the clusters keep every atom at least dmin = 1.5 Bohr from the others, and the seeds
/// are deterministic (no GPU).
#[test]
fn the_seeded_clusters_are_the_harness_geometries() {
    for (n, seed) in [(4, 1), (5, 3), (6, 5), (8, 8)] {
        let c = seeded(n, seed);
        assert_eq!(c.len(), n);
        assert_eq!(c, seeded(n, seed), "deterministic");
        for i in 0..n {
            for j in 0..i {
                let d = (0..3).map(|k| (c[i].1[k] - c[j].1[k]).powi(2)).sum::<f64>().sqrt();
                assert!(d >= 1.5, "H{n} seed {seed}: atoms {i}, {j} at {d}");
            }
        }
    }
    let r = ring(5, 1.8);
    let side = (0..3).map(|k| (r[1].1[k] - r[0].1[k]).powi(2)).sum::<f64>().sqrt();
    assert!((side - 1.8).abs() < 1e-12 && r[0].1 == [r[0].1[0], 0.0, 0.0]);
}
