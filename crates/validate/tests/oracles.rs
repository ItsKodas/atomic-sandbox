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
#[cfg_attr(debug_assertions, ignore = "release-only oracle; run: cargo test -p validate --release")]
fn h_atom_lsda() {
    // NIST LSD H: −0.478671 Ha. 81³ at h = 0.15 (12 Bohr), σ = 0.15..0.24 (module doc: fit bias ≲ 0.6 mHa).
    let (e, warnings) = atom_oracle("H", 1, 81, 0.15, 1.0 / std::f64::consts::PI);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!((e - (-0.478671)).abs() < 4e-3, "H LSD: got {e}, reference −0.478671");
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only oracle; run: cargo test -p validate --release")]
fn he_atom_lsda() {
    // NIST LSD He: −2.834836 Ha. 101³ at h = 0.10 (10 Bohr; face density ~1e-7), σ = 0.10..0.16
    // (controller ruling 2026-10-13). At h = 0.15 the four-point fit misses by +45.9 mHa because
    // the smearing series is pre-asymptotic at Z·σ_max = 0.48 (module doc); at h = 0.10 it is
    // +12.3 mHa — inside the 15 mHa bound with 2.7 mHa margin, the grid error between the two
    // spacings being < 0.5 mHa at equal σ. Z·h, not h, is what this oracle is sensitive to.
    let (e, warnings) = atom_oracle("He", 2, 101, 0.10, 3.6);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!((e - (-2.834836)).abs() < 1.5e-2, "He LSD: got {e}, reference −2.834836");
}

/// Li LSD total energy oracle — **deferred as a documented limitation** (controller ruling
/// 2026-10-13), not a solver defect. Reference NIST LSD −7.343957 Ha (corrected 2026-10-15, M7 Task 1 review:
/// −7.335195 is NIST's LDA column), bound 4e-2 (untouched).
///
/// Measured with this design (107³, h = 0.15, 15.9 Bohr box — no face warning, 5e-7 —
/// σ = 0.15/0.18/0.21/0.24, tol 1e-7): E(σ) = −5.730663 / −5.419456 / −5.138109 / −4.883871,
/// four-point σ → 0 fit a = **−7.004270, +340 mHa** (8.5× the bound). SCF is clean: 11–13 DIIS
/// steps, Δρ ≤ 8e-8, (n↑, n↓) = (2, 1), ∫ρ = 3.
///
/// Why: the smearing shift of a hydrogenic cusp is (4πρ(0)/Z)·F(Zσ), and the four-point
/// σ²/σ³/σ⁴ fit drops the (Zσ)⁵ term, so its bias scales as ρ(0)·Z⁴·σ_max⁵ (`validate` module
/// doc). With ρ(0) ≈ 13.8 and Z = 3 the first-order model predicts +207 mHa at h = 0.15
/// (measured +340 once density relaxation is included; the same model predicted +20 / +4 mHa
/// for He at h = 0.15 / 0.10 against measured +46 / +12). At Z·σ_max = 0.72 the series is far
/// from asymptotic — the fitted σ² coefficient (111) is less than half the physical 2πZρ(0) ≈ 260.
/// Reaching 40 mHa needs Z·h ≲ 0.2, i.e. h ≲ 0.07 in a 16 Bohr box (≥ 230³ ≈ 12 M points,
/// hours per σ on the CPU reference), and h = 0.10 (161³) would still miss (+50 mHa predicted).
/// This is the known cost of the Gaussian-smeared all-electron rung for Z ≥ 3, not a defect
/// of the SCF, Poisson, eigensolver or XC code — Li's SCF convergence and spin bookkeeping
/// remain covered by `qm::scf::tests::lithium_runs_with_unequal_spin_channels`.
///
/// Planned remedies: self-generated pseudopotentials (design spec §11 item 1), which remove
/// the 1s cusp and with it the Z⁴ scaling; or h ≲ 0.07 once the GPU solver makes it affordable.
/// Until one lands this test is `#[ignore]`; run it with
/// `cargo test -p validate --release --test oracles -- --ignored --nocapture li_atom_lsda`.
#[test]
#[ignore]
fn li_atom_lsda() {
    let (e, warnings) = atom_oracle("Li", 3, 107, 0.15, 13.8);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!((e - (-7.343957)).abs() < 4e-2, "Li LSD: got {e}, reference −7.343957");
}

/// F_x on nucleus 0 (at +R/2) for H₂ at bond length `r`, with warm start.
fn h2_force(g: &Grid, r: f64, sigma: f64, tol: f64, eig_tol: f64, warm: Option<&KsState>) -> (f64, KsState, usize) {
    let sys = h2(r);
    let (state, st) = converged(g, &sys, sigma, tol, eig_tol, warm);
    assert!(st.warnings.is_empty(), "R = {r}, σ = {sigma}: {:?}", st.warnings);
    let f = forces(g, &state, &sys.nuclei, sigma).unwrap();
    assert!((f[0][0] + f[1][0]).abs() < 1e-6 && f[0][1].abs() < 1e-6 && f[0][2].abs() < 1e-6, "R = {r}: {f:?}");
    (f[0][0], state, st.iterations)
}

/// The bond-length scan at 81³, h = 0.15: F_x0(R; σ) on the fixed scan `rs` for the σ ladder
/// (each R cold at the first σ, then warm along σ), R_e(σ) as the root of the Lagrange
/// polynomial through all scan points in the sign-change interval, then the σ → 0 fit.
/// Returns the extrapolated R_e; prints everything.
fn bond_length_scan(rs: &[f64], tol: f64, eig_tol: f64) -> f64 {
    let g = Grid::cube(81, 0.15);
    let sig = sigma_ladder(0.15);
    let t0 = Instant::now();
    let mut f = vec![vec![0.0; rs.len()]; 4]; // f[k][i]: σ_k, R_i
    let mut iters = vec![vec![0usize; rs.len()]; 4];
    for (i, &r) in rs.iter().enumerate() {
        let mut prev: Option<KsState> = None;
        for (k, &s) in sig.iter().enumerate() {
            let (fx, state, it) = h2_force(&g, r, s, tol, eig_tol, prev.as_ref());
            f[k][i] = fx; iters[k][i] = it;
            prev = Some(state);
        }
    }
    let wall = t0.elapsed().as_secs_f64();
    eprintln!("H2 bond scan: 81³, h = 0.15, tol {tol:.0e}, R = {rs:?}, σ = {sig:?}, wall {wall:.1} s");
    let mut re = [0.0; 4];
    for k in 0..4 {
        eprintln!("  σ {:.3}: F_x0(R) = {:?}  (SCF steps {:?})", sig[k], f[k], iters[k]);
        let i = (0..rs.len() - 1).find(|&i| f[k][i] > 0.0 && f[k][i + 1] <= 0.0)
            .unwrap_or_else(|| panic!("σ = {}: no sign change of F_x0 on the scan: {:?}", sig[k], f[k]));
        re[k] = interpolated_root(rs, &f[k], rs[i], rs[i + 1]);
        eprintln!("    R_e(σ) = {:.5}", re[k]);
    }
    let r_e = extrapolate_sigma(&sig, &re);
    eprintln!("  R_e(σ) = {re:?}  →  extrapolated R_e = {r_e:.4} Bohr (reference 1.446)");
    r_e
}

/// The scan nodes. Four points at 0.15 spacing bracket the σ-shifted minimum (R_e(σ) = 1.60–1.77)
/// with ≥ 0.13 Bohr margin on both sides. Three nodes are not enough: with the σ → 0 intercept
/// weights summing to |w|₁ ≈ 88, the ~1 mBohr interpolation error of a parabola over a 0.30 Bohr
/// span — which varies with σ as the root crosses the span — moves the extrapolated R_e by
/// 0.01–0.02 Bohr (measured on this data: fixed triples {1.60, 1.75, 1.90} / {1.45, 1.60, 1.75}
/// give 1.427 / 1.440 vs the cubic's 1.451), and a triple that switches with σ gives 1.539.
const BOND_SCAN: [f64; 4] = [1.45, 1.60, 1.75, 1.90];

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only oracle; run: cargo test -p validate --release")]
fn h2_bond_length() {
    // LSDA R_e = 1.446 Bohr (NIST CCCBDB). 81³, h = 0.15, σ = 0.15..0.24, eig_tol 1e-8;
    // SCF tol per `BOND_SCAN_TOL` (module doc of `validate` and `references.md` for the design).
    let r_e = bond_length_scan(&BOND_SCAN, BOND_SCAN_TOL, 1e-8);
    assert!((r_e - 1.446).abs() < 0.05, "H2 R_e = {r_e} Bohr, reference 1.446");
}

/// SCF tolerance of the bond scan (runtime ruling 2026-10-13: cuts allowed where the
/// extrapolated R_e moves < 1e-3 Bohr). Measured: tol 1e-8 → R_e = 1.4510614, 217 DIIS steps
/// over the 16 SCFs; tol 1e-7 → 1.4510602, 150 steps (−31 %), Δ = −1.2e-6 Bohr. The forces at
/// σ = 0.15 are bit-identical between the two: once LOBPCG's residual is below eig_tol = 1e-8
/// the orbitals — hence ρ_out and F — are frozen, and the extra steps from Δρ 1e-7 → 1e-8 only
/// let ρ_in catch up to a fixed ρ_out. eig_tol, not tol, bounds the force here.
const BOND_SCAN_TOL: f64 = 1e-7;

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only oracle; run: cargo test -p validate --release")]
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
