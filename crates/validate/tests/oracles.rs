//! Physics oracles (design: `validate` module doc; references: `references.md`). Each test
//! prints its raw table (`--nocapture`) so a miss can be attributed; the assertion bounds are
//! the brief's and are never loosened here. Run with `cargo test -p validate --release`.

mod common;

use qm::{forces, scf, Grid, KsState, Nucleus, Occupations, ScfSettings, System, ELECTRON_KT};
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

// ---- V3: the FD-consistency force oracle (Fermi–Dirac smearing, spec 2026-11-13 §9 "V3 force oracle") ----
//
// Derivation (atomic units; a force is −d/dx; F, E, TS in Ha; x in Bohr).
//
// Mermin: F[{ψ}, {f}] = E_KS[ρ] − kT·Σ_iσ s(f_iσ), s(f) = −f ln f − (1 − f) ln(1 − f), ρ_σ = Σ_i f_iσ|ψ_iσ|², with
// Σ_i f_iσ = N_σ per channel (multiplier μ_σ). ∂E_KS/∂f_i = ε_i, so ∂/∂f_i [F − μ_σ(Σf − N_σ)] = ε_i − μ_σ
// − kT ln((1 − f_i)/f_i) = 0 gives f_i = 1/(1 + e^{(ε_i − μ_σ)/kT}): Fermi–Dirac occupations make F stationary in
// f, and the Kohn–Sham equations make it stationary in ψ (orthonormality). So dF/dx = ∂F/∂x at fixed {ψ, f}
// = ∫ρ ∂V_ext/∂x + ∂E_nn/∂x, i.e. the Hellmann–Feynman force is HF_x = −dF/dx. E alone is not stationary in f:
// dE/dx = dF/dx + d(TS)/dx, hence HF_x + dE/dx = d(TS)/dx, nonzero exactly where a partial level's f moves.
//
// Stencil: FD_X(d) = −[X(x + d) − X(x − d)]/(2d) = −X′ − X‴d²/6 − X⁽⁵⁾d⁴/120 + O(d⁶), X ∈ {F (`total`), E (`internal`)}.
// Richardson R_X(a, 2a) = (4·FD_X(a) − FD_X(2a))/3 = −X′ + X⁽⁵⁾a⁴/30 + O(a⁶). Hence
//   R(δ/2, δ) − R(δ, 2δ) = −X⁽⁵⁾δ⁴/32: 15× the remainder of R(δ/2, δ) (X⁽⁵⁾δ⁴/480), and 3.75× FD(δ)'s own δ⁴ term
//   (X⁽⁵⁾δ⁴/120), the spec's "3.75×"; either way ≤ 2.5e-5 certifies R(δ/2, δ) to ≤ 1.7e-6.
// The δ-scaling check: with e = FD_F(δ) − R_F(δ/2, δ) ≈ −F‴δ²/6 and b = HF − R_F(δ/2, δ), FD_F(2δ) − R_F ≈ 4e, so
//   (FD_F(2δ) − HF)/(FD_F(δ) − HF) = (4e − b)/(e − b) ∈ [3, 5]  ⇔  b/e ∈ [−½, ¼]: −e/2 ≤ b ≤ e/4 for e > b (the
//   spec's form), e/4 ≤ b ≤ −e/2 for e < b (this case: F‴ > 0 makes e < 0).
// Spec §9 V3 pre-registration: if the ratio is asserted and b lies outside that interval, stop and report (a spec
// decision, never a re-pin).

/// δ of the V3 stencil (Bohr): central differences at δ/2, δ and 2δ (spec §9 V3 "Method").
const V3_DELTA: f64 = 0.01;
/// SCF tolerance of the V3 solves (spec §9 V3: CPU f64, tol 1e-9; `qm` then asks its Poisson solves for 1e-11).
const V3_TOL: f64 = 1e-9;
/// Eigensolver tolerance of the V3 solves (Task 6 ruling, test-local). Near the pinned pair an eigen residual r leaves the
/// pair's mutual rotation θ ≈ r/Δ unresolved, which moves ρ_out by ≈ (f₁ − f₂)·θ ≈ 1e-8 at r = 1e-9 (Δ = 12.9 mHa,
/// f ≈ 0.79/0.21): measured, a warm re-solve at eig_tol 1e-9 sat at Δρ 1e-8 … 2e-7 for 200 steps, at 1e-10 it converged
/// in 19.
const V3_EIG_TOL: f64 = 1e-10;

/// One V3 stencil point: displacement x, F, E, TS, the frontier split Δ, SCF steps and the final Δρ.
struct V3Point { x: f64, f: f64, e: f64, ts: f64, split: f64, iterations: usize, delta_rho: f64, face: f64 }

/// The V3 stencil: HF at x = 0, the seven points x ∈ {0, ±δ/2, ±δ, ±2δ} in ascending x (the centre cold, the six
/// displaced solves warm from it), and the centre re-solved warm from the +δ/2 state (its F − the cold F, E likewise:
/// the CPU path scatter).
struct V3Stencil { hf: f64, points: Vec<V3Point>, scatter: (f64, f64), wall: f64 }

impl V3Stencil {
    fn at(&self, x: f64) -> &V3Point { self.points.iter().find(|p| p.x == x).unwrap_or_else(|| panic!("no stencil point at {x}")) }
    /// FD_X(d) = −[X(+d) − X(−d)]/(2d), a force.
    fn fd(&self, x: fn(&V3Point) -> f64, d: f64) -> f64 { -(x(self.at(d)) - x(self.at(-d))) / (2.0 * d) }
    /// R_X(a, 2a) = (4·FD_X(a) − FD_X(2a))/3.
    fn rich(&self, x: fn(&V3Point) -> f64, a: f64) -> f64 { (4.0 * self.fd(x, a) - self.fd(x, 2.0 * a)) / 3.0 }
}

/// The frontier split of a state (Ha): ε at ascending rank N_σ minus ε at N_σ − 1 (0-based), averaged over the two
/// channels (both occupied here).
fn frontier_split(st: &KsState, sys: &System) -> f64 {
    let split = |evals: &[f64], n: usize| {
        let mut e = evals.to_vec();
        e.sort_by(f64::total_cmp);
        assert!(n >= 1 && e.len() > n, "a witness above rank {n}: {e:?}");
        e[n] - e[n - 1]
    };
    0.5 * (split(&st.evals_up, sys.n_up) + split(&st.evals_dn, sys.n_dn))
}

/// `f` over `items` on at most 4 threads (`qm` is single-threaded; the machine is shared), results in input order.
fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out: Vec<std::sync::Mutex<Option<R>>> = items.iter().map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|sc| {
        for _ in 0..4.min(items.len()) {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if i >= items.len() { break; }
                *out[i].lock().unwrap() = Some(f(&items[i]));
            });
        }
    });
    out.into_iter().map(|m| m.into_inner().unwrap().expect("every item ran")).collect()
}

/// Solves the V3 stencil on `common::v3_h4()` (CPU f64, σ = h, tol `V3_TOL`, eig_tol `V3_EIG_TOL`, `FermiDirac { kt:
/// ELECTRON_KT }`) and prints its table. Every solve must converge without a warning.
fn v3_stencil() -> V3Stencil {
    let (g, sys, atom, axis) = common::v3_h4();
    let mut s = ScfSettings::for_grid(&g);
    (s.tol, s.eig_tol, s.occupations) = (V3_TOL, V3_EIG_TOL, Occupations::FermiDirac { kt: ELECTRON_KT });
    let moved = |x: f64| {
        let mut nuclei = sys.nuclei.clone();
        nuclei[atom].pos[axis] += x;
        System { nuclei, n_up: sys.n_up, n_dn: sys.n_dn, ionic: None }
    };
    let solve = |x: f64, warm: Option<&KsState>| {
        let (st, stats) = scf(&g, &moved(x), &s, warm).unwrap_or_else(|e| panic!("V3 x = {x:+}: {e}"));
        // the box warning alone is allowed: its face density (≈ 1.7e-5, the partly occupied level's slow tail) is the
        // same Dirichlet box at every x, part of the discrete problem whose F the stencil differentiates
        let is_box = |w: &String| w.starts_with("density on box faces is ") && w.ends_with(": box too small");
        assert!(stats.warnings.iter().all(is_box) && stats.warnings.len() <= 1, "V3 x = {x:+}: {:?}", stats.warnings);
        let p = V3Point { x, f: stats.energy.total, e: stats.energy.internal, ts: stats.energy.ts, split: frontier_split(&st, &sys),
                          iterations: stats.iterations, delta_rho: stats.delta_rho, face: stats.boundary_density };
        (st, p)
    };
    let t0 = Instant::now();
    let (centre, c0) = solve(0.0, None);
    let hf = forces(&g, &centre, &sys.nuclei, s.sigma).unwrap()[atom][axis];
    let xs = [-2.0 * V3_DELTA, -V3_DELTA, -0.5 * V3_DELTA, 0.5 * V3_DELTA, V3_DELTA, 2.0 * V3_DELTA];
    let mut out = par_map(&xs, |&x| solve(x, Some(&centre)));
    let (_, again) = solve(0.0, Some(&out[3].0));
    let scatter = (again.f - c0.f, again.e - c0.e);
    let mut points: Vec<V3Point> = out.drain(..).map(|(_, p)| p).collect();
    points.insert(3, c0);
    let wall = t0.elapsed().as_secs_f64();
    eprintln!("V3 H₄ 1.8 × 1.95, atom {atom} along axis {axis}: 41³ × 0.30, σ = h, CPU f64, tol {V3_TOL:.0e}, eig_tol {V3_EIG_TOL:.0e}, FD kT {ELECTRON_KT}; wall {wall:.0} s");
    for p in &points {
        eprintln!("  x {:+.3}  F {:+.12}  E {:+.12}  TS {:.9e}  Δ {:.6e} (u = {:.3})  {} steps  Δρ {:.1e}  face {:.2e}",
                  p.x, p.f, p.e, p.ts, p.split, p.split / (4.0 * ELECTRON_KT), p.iterations, p.delta_rho, p.face);
    }
    eprintln!("  centre warm from +δ/2 − cold: F {:+.2e}, E {:+.2e} ({} steps)", scatter.0, scatter.1, again.iterations);
    for w in points.windows(2) {
        let dx = w[1].x - w[0].x;
        eprintln!("  [{:+.3}, {:+.3}]: c = dΔ/dx {:.5}  d(TS)/dx {:+.5e}  (d(TS)/dx)/c {:+.3}", w[0].x, w[1].x,
                  (w[1].split - w[0].split) / dx, (w[1].ts - w[0].ts) / dx, (w[1].ts - w[0].ts) / (w[1].split - w[0].split));
    }
    V3Stencil { hf, points, scatter, wall }
}

/// V3 [MUST-4] (spec 2026-11-13 §9 "V3 force oracle"; plan Task 6; the derivation above): near a pinned pair the
/// Hellmann–Feynman force is −dF/dx, F = E − TS, and not −dE/dx. `common::v3_h4()` (H₄ 1.8 × 1.95, atom 0 along x;
/// frontier split ≈ 2.6 kT), CPU f64, 41³ × 0.30, σ = h, tol 1e-9, eig_tol 1e-10, `FermiDirac { kt: ELECTRON_KT }`; eight
/// solves (the centre cold, the six displaced ones warm from it, the centre again warm for the path scatter). The
/// bounds are the spec's, verbatim: |R_F(δ/2, δ) − R_F(δ, 2δ)| ≤ 2.5e-5 Ha/Bohr (certifies the δ⁴ remainder); the
/// δ-scaling ratio (FD_F(2δ) − HF)/(FD_F(δ) − HF) ∈ [3, 5] when |FD_F(δ) − HF| ≥ 1e-5 (pre-registered stop: b outside
/// [−e/2, e/4] is a spec decision); |HF − R_F(δ/2, δ)| ≤ 1e-4 (the "M7 forces" Richardson bound); |HF − R_E(δ/2, δ)|
/// ≥ 5e-4 (tells F's derivative from E's). Measured numbers: `references.md` "Fermi–Dirac migration", V3.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only oracle; run: cargo test -p validate --release")]
fn fd_forces_are_minus_df_dx_near_a_pinned_pair() {
    let v = v3_stencil();
    let (f, e): (fn(&V3Point) -> f64, fn(&V3Point) -> f64) = (|p| p.f, |p| p.e);
    let d = V3_DELTA;
    let hf = v.hf;
    let (fd_half, fd1, fd2) = (v.fd(f, 0.5 * d), v.fd(f, d), v.fd(f, 2.0 * d));
    let (r_fine, r_coarse) = (v.rich(f, 0.5 * d), v.rich(f, d));
    let (r_e_fine, r_e_coarse) = (v.rich(e, 0.5 * d), v.rich(e, d));
    let trunc = fd1 - r_fine; // e
    let offset = hf - r_fine; // b
    let ratio = (fd2 - hf) / (fd1 - hf);
    let ratio_asserted = (fd1 - hf).abs() >= 1e-5;
    eprintln!("V3: HF {hf:+.10}  FD_F(δ/2, δ, 2δ) {fd_half:+.10} {fd1:+.10} {fd2:+.10}  plain |FD(δ) − FD(2δ)| {:.3e}", (fd1 - fd2).abs());
    eprintln!("    R_F(δ/2, δ) {r_fine:+.10}  R_F(δ, 2δ) {r_coarse:+.10}  |difference| {:.3e} (≤ 2.5e-5)", (r_fine - r_coarse).abs());
    eprintln!("    R_E(δ/2, δ) {r_e_fine:+.10}  R_E(δ, 2δ) {r_e_coarse:+.10}  FD_E(δ/2, δ, 2δ) {:+.10} {:+.10} {:+.10}",
              v.fd(e, 0.5 * d), v.fd(e, d), v.fd(e, 2.0 * d));
    eprintln!("    HF − R_F(δ/2, δ) = b {offset:+.3e} (|b| ≤ 1e-4);  HF − R_E(δ/2, δ) {:+.4e} (≥ 5e-4)", hf - r_e_fine);
    eprintln!("    e = FD_F(δ) − R_F(δ/2, δ) {trunc:+.3e} (implied F‴ = −6e/δ² {:+.3}); FD_F(δ) − HF {:+.3e}; ratio (FD(2δ) − HF)/(FD(δ) − HF) {ratio:.4} \
               ({}); b/e {:+.4} (in [−½, ¼] ⇔ ratio in [3, 5])",
              -6.0 * trunc / (d * d), fd1 - hf, if ratio_asserted { "asserted" } else { "not asserted: |FD(δ) − HF| < 1e-5" }, offset / trunc);
    eprintln!("    path scatter (centre warm − cold): F {:+.2e}, E {:+.2e}; wall {:.0} s", v.scatter.0, v.scatter.1, v.wall);
    assert!((r_fine - r_coarse).abs() <= 2.5e-5, "R_F(δ/2, δ) {r_fine} vs R_F(δ, 2δ) {r_coarse}: the δ⁴ remainder is not certified");
    if ratio_asserted {
        assert!((3.0..=5.0).contains(&ratio),
                "δ-scaling ratio {ratio} ∉ [3, 5] (e = {trunc:e}, b = {offset:e}, b/e = {}): STOP and report: spec 2026-11-13 §9 V3 \
                 pre-registration, a spec decision for the user, never a re-pin", offset / trunc);
    }
    assert!((hf - r_fine).abs() <= 1e-4, "HF {hf} vs R_F(δ/2, δ) {r_fine}: HF is not −dF/dx within the M7 Richardson bound");
    assert!((hf - r_e_fine).abs() >= 5e-4, "HF {hf} vs R_E(δ/2, δ) {r_e_fine}: the test does not tell dF/dx from dE/dx here");
}
