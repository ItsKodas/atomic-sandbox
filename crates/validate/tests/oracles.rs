//! Physics oracles (design: `validate` module doc; references: `references.md`). Each test
//! prints its raw table (`--nocapture`) so a miss can be attributed; the assertion bounds are
//! the brief's and are never loosened here. Run with `cargo test -p validate --release`.

use qm::{forces, Grid, KsState, Nucleus, System};
use std::time::Instant;
use validate::{converged, energy_ladder, extrapolate_sigma, first_order_shift, interpolated_root, sigma_ladder};

fn atom(z: u8) -> System { System::new(vec![Nucleus { pos: [0.0; 3], z }], 0, None) }

fn h2(r: f64) -> System {
    System::new(vec![Nucleus { pos: [r / 2.0, 0.0, 0.0], z: 1 }, Nucleus { pos: [-r / 2.0, 0.0, 0.0], z: 1 }], 0, None)
}

/// Runs the atom ladder, prints the E(σ) table with the fit and the residual against the
/// first-order curve, and returns (a, warnings seen).
fn atom_oracle(name: &str, z: u8, n: usize, h: f64, rho0_cusp: f64) -> (f64, Vec<String>) {
    let g = Grid::cube(n, h);
    let sig = sigma_ladder(h);
    let t0 = Instant::now();
    let (e, stats) = energy_ladder(&g, &atom(z), &sig, 1e-7, 1e-7);
    let wall = t0.elapsed().as_secs_f64();
    let a = extrapolate_sigma(&sig, &e);
    eprintln!("{name}: {n}³, h = {h}, box {:.2} Bohr, σ ladder {sig:?}, wall {wall:.1} s", (n as f64 - 1.0) * h);
    for k in 0..4 {
        let d1 = first_order_shift(sig[k], z as f64, rho0_cusp);
        eprintln!("  σ {:.3}  E {:+.6}  E−a {:+.5}  1st-order {:+.5}  ratio {:.3}  iters {}  Δρ {:.1e}  face {:.1e}  {:?}",
            sig[k], e[k], e[k] - a, d1, (e[k] - a) / d1, stats[k].iterations, stats[k].delta_rho, stats[k].boundary_density, stats[k].energy);
    }
    eprintln!("  extrapolated a = {a:+.6}");
    (a, stats.iter().flat_map(|s| s.warnings.iter().cloned()).collect())
}

#[test]
fn h_atom_lsda() {
    // NIST LSD H: −0.478671 Ha. 81³ at h = 0.15 (12 Bohr), σ = 0.15..0.24 (module doc: fit bias ≲ 0.6 mHa).
    let (e, warnings) = atom_oracle("H", 1, 81, 0.15, 1.0 / std::f64::consts::PI);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!((e - (-0.478671)).abs() < 4e-3, "H LSD: got {e}, reference −0.478671");
}

// The He and Li oracles (NIST LSD −2.834836 / −7.335195, bounds 1.5e-2 / 4e-2) are not in this
// file yet: with the controller's h = 0.15 designs the four-point fit misses by +46 mHa (He) and
// +331 mHa (Li) — the pre-asymptotic smearing series at Z·σ_max = 0.48 / 0.72, as the module doc
// predicts, not a solver error. The runnable tests, tables and the h = 0.10 He variant that
// passes (+12.3 mHa) are in `oracles_pending.rs` (untracked) and the Task 11 report, awaiting a
// ruling on their grids.

/// F_x on nucleus 0 (at +R/2) for H₂ at bond length `r`, with warm start.
fn h2_force(g: &Grid, r: f64, sigma: f64, tol: f64, eig_tol: f64, warm: Option<&KsState>) -> (f64, KsState, usize) {
    let sys = h2(r);
    let (state, st) = converged(g, &sys, sigma, tol, eig_tol, warm);
    assert!(st.warnings.is_empty(), "R = {r}, σ = {sigma}: {:?}", st.warnings);
    let f = forces(g, &state, &sys.nuclei, sigma).unwrap();
    assert!((f[0][0] + f[1][0]).abs() < 1e-6 && f[0][1].abs() < 1e-6 && f[0][2].abs() < 1e-6, "R = {r}: {f:?}");
    (f[0][0], state, st.iterations)
}

#[test]
fn h2_bond_length() {
    // LSDA R_e = 1.446 Bohr (NIST CCCBDB). At each σ the root of F_x(R; σ) is located on the
    // scan R ∈ {1.45, 1.60, 1.75, 1.90} (cubic interpolant, sign-change interval), then R_e(σ)
    // is extrapolated to σ → 0 (module doc). 81³, h = 0.15, tol 1e-8, eig_tol 1e-8.
    // The cubic through all four scan points is essential, not cosmetic: the σ → 0 intercept
    // weights sum to |w|₁ ≈ 88, so any error in R_e(σ) that is not smooth in σ is amplified
    // ~90×. The 3-point parabola printed below as a check agrees with the cubic to ≤ 1.4e-3 Bohr
    // per σ, but its bracketing triple switches between σ = 0.18 and 0.21 and the resulting
    // alternating error extrapolates to R_e = 1.539 (Task 11 measurement) — a fail — while the
    // fixed-node cubic gives 1.451.
    let g = Grid::cube(81, 0.15);
    let sig = sigma_ladder(0.15);
    let rs = [1.45, 1.60, 1.75, 1.90];
    let t0 = Instant::now();
    let mut f = [[0.0; 4]; 4]; // f[k][i]: σ_k, R_i
    let mut iters = [[0usize; 4]; 4];
    for (i, &r) in rs.iter().enumerate() {
        let mut prev: Option<KsState> = None;
        for (k, &s) in sig.iter().enumerate() {
            let (fx, state, it) = h2_force(&g, r, s, 1e-8, 1e-8, prev.as_ref());
            f[k][i] = fx; iters[k][i] = it;
            prev = Some(state);
        }
    }
    let wall = t0.elapsed().as_secs_f64();
    eprintln!("H2 bond scan: 81³, h = 0.15, R = {rs:?}, σ = {sig:?}, wall {wall:.1} s");
    let mut re = [0.0; 4];
    for k in 0..4 {
        eprintln!("  σ {:.3}: F_x0(R) = {:?}  (SCF steps {:?})", sig[k], f[k], iters[k]);
        let i = (0..3).find(|&i| f[k][i] > 0.0 && f[k][i + 1] <= 0.0)
            .unwrap_or_else(|| panic!("σ = {}: no sign change of F_x0 on the scan: {:?}", sig[k], f[k]));
        re[k] = interpolated_root(&rs, &f[k], rs[i], rs[i + 1]);
        // 3-point parabola through the bracketing triple, as an interpolation-error check
        let j = if i == 0 || (i < 2 && (re[k] - rs[i]) > (rs[i + 1] - re[k])) { i } else { i - 1 };
        let p = interpolated_root(&rs[j..j + 3], &f[k][j..j + 3], rs[i], rs[i + 1]);
        eprintln!("    R_e(σ) = {:.5} (cubic)   {:.5} (parabola through {:?})", re[k], p, &rs[j..j + 3]);
    }
    let r_e = extrapolate_sigma(&sig, &re);
    eprintln!("  R_e(σ) = {re:?}  →  extrapolated R_e = {r_e:.4} Bohr (reference 1.446)");
    assert!((r_e - 1.446).abs() < 0.05, "H2 R_e = {r_e} Bohr, reference 1.446 (R_e(σ) = {re:?}, F = {f:?})");
}

#[test]
fn h2_force_vanishes_at_equilibrium() {
    // F_x on nucleus 0 at R = 1.446 for the σ ladder (tol 1e-9, eig_tol 1e-8), extrapolated to σ → 0.
    let g = Grid::cube(81, 0.15);
    let sig = sigma_ladder(0.15);
    let t0 = Instant::now();
    let mut fx = [0.0; 4];
    let mut prev: Option<KsState> = None;
    for (k, &s) in sig.iter().enumerate() {
        let (f, state, it) = h2_force(&g, 1.446, s, 1e-9, 1e-8, prev.as_ref());
        fx[k] = f;
        eprintln!("  σ {s:.3}: F_x0(1.446) = {f:+.6}  (SCF steps {it})");
        prev = Some(state);
    }
    let f0 = extrapolate_sigma(&sig, &fx);
    eprintln!("H2 force at R_e: 81³, h = 0.15, F(σ) = {fx:?} → extrapolated {f0:+.6} Ha/Bohr, wall {:.1} s", t0.elapsed().as_secs_f64());
    assert!(f0.abs() < 1e-2, "residual force at R_e after σ → 0: {f0} (F(σ) = {fx:?})");
}
