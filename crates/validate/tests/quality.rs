//! The app's quality presets (user request 2026-10-14; `crates/app/src/quality.rs`): what each
//! grid spacing buys on H₂ and what it costs. **Report-only** (`#[ignore]`, no assertion on the
//! numbers): the measured R_e(σ = h) and the solve times go to `references.md`, "Quality presets",
//! where the app's accuracy hints cite them.
//!
//! Method: the presets' 12 Bohr box at h = 0.30 / 0.25 / 0.15 / 0.10 (41³ / 49³ / 81³ / 121³),
//! σ = h (the app's default: σ follows h), `GpuKs` at the sample tolerance 1e-6. R_e(σ = h) by
//! bisection of the Hellmann-Feynman F_x on nucleus 0 in [1.40, 2.20] Bohr to 1e-4 (warm solves
//! along the bisection), the root by linear interpolation in the last bracket. Then the cost at
//! R_e: one cold solve (the solver already built, its state reset) and one warm re-solve after
//! a 0.01 Bohr nudge, wall time each (after one untimed warm-up solve that compiles the pipelines).
//!
//! Run: `cargo test -p validate --release --test quality -- --ignored --nocapture`.
mod common;
use common::{gpu, never, GPU};
use providers::GpuKs;
use sim::{Cell, ForceProvider, Nucleus, Scene};
use std::time::Instant;

const BOX: f64 = 12.0;
const TOL: f64 = 1e-6;
const BRACKET: (f64, f64) = (1.40, 2.20);
const R_TOL: f64 = 1e-4;

fn h2(n: usize, h: f64, r: f64) -> Scene {
    Scene {
        nuclei: vec![Nucleus::new(1, [r / 2.0, 0.0, 0.0]), Nucleus::new(1, [-r / 2.0, 0.0, 0.0])],
        cell: Cell { n, h },
        charge: 0,
        multiplicity: None,
        sigma: None,
        external_field: [0.0; 3],
    }
}

/// The smallest odd n with (n − 1)·h ≥ BOX (`app::quality::grid_n`).
fn grid_n(h: f64) -> usize {
    let k = (BOX / h * (1.0 - 1e-9)).ceil() as usize;
    if k % 2 == 1 { k + 2 } else { k + 1 }
}

#[test]
#[ignore = "report-only: measures H₂'s R_e and solve time per quality level (~minutes on the GPU)"]
fn h2_bond_length_per_quality() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let Some(ctx) = gpu("quality") else { return };
    // warm-up: the device's first solve compiles the pipelines (≈ 0.5 s) — not a level's cost
    GpuKs::new(ctx.clone(), TOL).evaluate(&h2(33, 0.25, 1.6), &mut never()).unwrap();
    let mut rows = Vec::new();
    for (name, h) in [("fast", 0.30), ("balanced", 0.25), ("accurate", 0.15), ("precise", 0.10)] {
        let n = grid_n(h);
        let mut p = GpuKs::new(ctx.clone(), TOL);
        let t0 = Instant::now();
        let mut f_at = |r: f64| p.evaluate(&h2(n, h, r), &mut never()).unwrap_or_else(|e| panic!("{name} R = {r}: {e}")).forces[0][0];
        let (mut lo, mut hi) = BRACKET;
        let (mut f_lo, mut f_hi) = (f_at(lo), f_at(hi));
        assert!(f_lo > 0.0 && f_hi < 0.0, "{name}: R_e not bracketed: {f_lo:+.3e}, {f_hi:+.3e}");
        let mut solves = 2;
        while hi - lo > R_TOL {
            let mid = 0.5 * (lo + hi);
            let f = f_at(mid);
            solves += 1;
            if f > 0.0 { lo = mid; f_lo = f; } else { hi = mid; f_hi = f; }
        }
        let r_e = lo + (hi - lo) * f_lo / (f_lo - f_hi);
        let scan = t0.elapsed().as_secs_f64();
        // the cost at R_e: cold (solver built, state reset), then warm after a small nudge
        p.reset();
        let t = Instant::now();
        let cold = p.evaluate(&h2(n, h, r_e), &mut never()).unwrap();
        let cold_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let warm = p.evaluate(&h2(n, h, r_e + 0.01), &mut never()).unwrap();
        let warm_s = t.elapsed().as_secs_f64();
        let ang = units::bohr_to_angstrom(r_e);
        let row = format!(
            "| {name} | {h:.2} | {n}³ | {r_e:.4} Bohr = {ang:.4} Å | {:+.4} Å | {cold_s:.2} s ({} SCF steps) | {warm_s:.2} s ({} SCF steps) | {solves} solves, {scan:.1} s |",
            ang - 0.7414,
            cold.stats.scf_steps,
            warm.stats.scf_steps
        );
        eprintln!("{row}  E(R_e) = {:+.6} Ha", cold.energy);
        rows.push(row);
    }
    eprintln!("\n| level | h = σ (Bohr) | grid | R_e(σ = h) | vs 0.7414 Å | cold solve at R_e | warm re-solve (+0.01 Bohr) | scan |\n|---|---|---|---|---|---|---|---|");
    for r in rows {
        eprintln!("{r}");
    }
}
