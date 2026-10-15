//! M7 Task 6: the self-generated pseudopotentials in the 3-D CPU solver (spec §5.1–§5.3, §6.3, §6.6
//! attribution; plan Task 6). CPU, f64, atomic units (Hartree, Bohr, Ha/Bohr). Release-only: run
//! `cargo test -p validate --release --test pseudo`.
//!
//! Protocol:
//! - The tables come from `qm::radial` (`generate`, cached per binary; `filtered` at the scene's h).
//! - Setups are built by `IonicSetup::build` with the real `has_core`/`filtered` closures, so every potential is
//!   certified at its h (T8, `ionic` module doc), except where a test says it compares an uncertified table on
//!   purpose (Ne at 0.15: no h_req).
//! - ScfSettings: `for_grid` (tol 1e-6) with eig_tol 1e-7, unless stated; an unconverged solve is an error,
//!   never a pass.
//! - Boxes are ≈ 12 Bohr cubes with a grid point at the origin (`n_for`).
//! - Forces on pseudised states are Task 7's. Until then `qm::forces` refuses them, and the single-atom floor
//!   below takes the Hellmann–Feynman force as −dE/dR of the *frozen* energy terms that depend on R:
//!   ∫ρ V_loc(R), E_NL(R) at fixed orbitals, E_xc[ρ_σ + ½ρ_pc(R)] at fixed spin densities, and E_ii.
//!   This uses only Task 6's own functions (`frozen_force`). At self-consistency that is dE/dR (the
//!   grid does not move: no Pulay term, `forces.rs`), so those solves run at tol = eig_tol = 1e-9 (plan review S6).
use qm::ionic::{Ionic, IonicError, IonicPolicy, IonicSetup};
use qm::nonlocal::NonlocalOp;
use qm::nuclear::{has_partial_core, ionic_local_potential, ionic_repulsion, partial_core_field};
use qm::radial::{filter::filtered, generate, has_core, PseudoTable};
use qm::{scf, Grid, KsState, Nucleus, ScfSettings, System};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

fn table(z: u8) -> Arc<PseudoTable> {
    static T: OnceLock<Mutex<HashMap<u8, Arc<PseudoTable>>>> = OnceLock::new();
    if let Some(t) = T.get_or_init(Default::default).lock().unwrap().get(&z) { return t.clone(); }
    let t = Arc::new(generate(z).expect("generates"));
    T.get_or_init(Default::default).lock().unwrap().entry(z).or_insert(t).clone()
}

pub fn setup(policy: IonicPolicy, h: f64, zs: &[u8]) -> Arc<IonicSetup> {
    Arc::new(IonicSetup::build(policy, h, h, zs, &mut |z| has_core(z), &mut |z| Ok(Arc::new(filtered(&table(z), h)))).expect("setup builds"))
}

/// The odd n giving a box of ≈ `box_bohr` at spacing h (a grid point at the origin).
fn n_for(box_bohr: f64, h: f64) -> usize { 2 * (box_bohr / (2.0 * h)).round() as usize + 1 }

fn solve_with(h: f64, box_bohr: f64, setup: Arc<IonicSetup>, nuclei: Vec<Nucleus>, mult: Option<u32>, tol: Option<f64>) -> (f64, KsState, qm::ScfStats) {
    let g = Grid::cube(n_for(box_bohr, h), h);
    let sys = System::with_ionic(nuclei, 0, mult, setup);
    let mut s = ScfSettings::for_grid(&g);
    s.eig_tol = 1e-7;
    if let Some(t) = tol { (s.tol, s.eig_tol) = (t, t); }
    let (st, stats) = scf(&g, &sys, &s, None).expect("SCF converges");
    (stats.energy.total, st, stats)
}

fn solve(h: f64, box_bohr: f64, policy: IonicPolicy, nuclei: Vec<Nucleus>, mult: Option<u32>) -> (f64, KsState, qm::ScfStats) {
    let zs: Vec<u8> = nuclei.iter().map(|n| n.z).collect();
    solve_with(h, box_bohr, setup(policy, h, &zs), nuclei, mult, None)
}

/// The R-dependent part of the energy at a frozen state (module doc): ∫ρ V_loc + E_NL + E_xc[ρ_σ + ½ρ_pc] + E_ii.
fn frozen_energy(g: &Grid, st: &KsState, setup: &IonicSetup, nuclei: &[Nucleus]) -> f64 {
    let rho = st.rho();
    let mut e = g.dot(&rho, &ionic_local_potential(g, nuclei, setup));
    let nl = NonlocalOp::build(g, nuclei, setup);
    e += nl.energy(&st.orbitals_up) + nl.energy(&st.orbitals_dn);
    if has_partial_core(nuclei, setup) {
        let pc = partial_core_field(g, nuclei, setup);
        let u: Vec<f64> = st.rho_up.iter().zip(&pc).map(|(a, c)| a + 0.5 * c).collect();
        let d: Vec<f64> = st.rho_dn.iter().zip(&pc).map(|(a, c)| a + 0.5 * c).collect();
        e += g.integrate(&qm::xc::lsda_pw92(&u, &d).energy_density);
    }
    e + ionic_repulsion(nuclei, setup).unwrap()
}

/// −∂E_frozen/∂R_I (module doc), Richardson-extrapolated central differences at δ = 1e-4 and 2e-4 Bohr (Task 7's
/// frozen-state protocol, plan review S2), and the δ = 2e-3 central difference as a cross-check (it straddles
/// ≥ 2 nodes of the uniform tables, Δ = h/256 ≤ 1.4e-3).
fn frozen_force(g: &Grid, st: &KsState, setup: &IonicSetup, nuclei: &[Nucleus], i: usize) -> ([f64; 3], [f64; 3]) {
    let (mut f, mut wide) = ([0.0; 3], [0.0; 3]);
    for a in 0..3 {
        let d = |delta: f64| {
            let (mut p, mut m) = (nuclei.to_vec(), nuclei.to_vec());
            p[i].pos[a] += delta;
            m[i].pos[a] -= delta;
            -(frozen_energy(g, st, setup, &p) - frozen_energy(g, st, setup, &m)) / (2.0 * delta)
        };
        f[a] = (4.0 * d(1e-4) - d(2e-4)) / 3.0;
        wide[a] = d(2e-3);
    }
    (f, wide)
}

fn norm(v: [f64; 3]) -> f64 { (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() }

/// Spec §6.3: the pseudo-Ne atom (closed shell, spherical) on the 3-D grid at h = 0.15 reproduces the
/// radial KB-PP atom (filtered at the same h) to < 2 mHa.
///
/// Deviation (Task 6): `IonicSetup::build` refuses Ne at 0.15. Ne has no h_req at `m7-tm-kb-nlcc-3`: T8(a)
/// misses by 0.35 mHa at the finest rung, an outcome pinned in the records (Task 5 ruling item 4). This test
/// asserts that refusal, then compares the same filtered table through `IonicSetup::from_tables`. 3-D vs radial
/// on one table is a solver-integration test; it does not depend on the table's T8 certificate. Reported: the
/// finer-than-ladder path (`check()` at h = 0.125, certified by its own T8 row).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo")]
fn pseudo_ne_3d_matches_radial() {
    let h = 0.15;
    let ft = Arc::new(filtered(&table(10), h));
    match IonicSetup::build(IonicPolicy::CoresOnly, h, h, &[10], &mut |z| has_core(z), &mut |_| Ok(ft.clone())) {
        Err(IonicError::TooCoarse { z: 10, h_req: None, .. }) => {}
        other => panic!("Ne at h = 0.15 must be TooCoarse with no h_req: {other:?}"),
    }
    let fine = 0.125;
    let ft_fine = Arc::new(filtered(&table(10), fine));
    let verdict = IonicSetup::build(IonicPolicy::CoresOnly, fine, fine, &[10], &mut |z| has_core(z), &mut |_| Ok(ft_fine.clone()));
    eprintln!("Ne at h = {fine} (finer than the ladder): T8 row ΔE {:+.3} mHa, Δε {:.3} mHa, T3 {:.3} mHa, clamped {}, F_table {:.2e} → {}",
              ft_fine.check.de * 1e3, ft_fine.check.max_eig * 1e3, ft_fine.check.t3_max * 1e3, ft_fine.check.clamped, ft_fine.check.f_table.iter().sum::<f64>(),
              match &verdict { Ok(_) => "certified".to_string(), Err(e) => e.to_string() });
    assert!(!matches!(verdict, Err(IonicError::TooCoarse { .. })), "a scene finer than the ladder is certified at its own h, never TooCoarse");

    let radial = qm::radial::kb::solve_kb_atom(&ft.grid, &ft.input(), &qm::radial::kb::reference_config(&table(10).partition), qm::radial::SpinMode::Restricted).unwrap();
    let s = Arc::new(IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, BTreeMap::from([(10, Ionic::Pseudo(ft.clone()))])));
    let t0 = Instant::now();
    let (e3d, st, stats) = solve_with(h, 12.0, s, vec![Nucleus { pos: [0.0; 3], z: 10 }], Some(1), None);
    eprintln!("pseudo-Ne: E_3D {e3d:.6} vs radial {:.6} (Δ {:+.2e}), {} SCF steps, {:.1} s; ε_3D ↑ {:?} vs radial {:?}; face density {:.1e}, warnings {:?}",
              radial.energy, e3d - radial.energy, stats.iterations, t0.elapsed().as_secs_f64(), st.evals_up, radial.eps, stats.boundary_density, stats.warnings);
    assert!(stats.warnings.is_empty(), "{:?}", stats.warnings);
    assert!((e3d - radial.energy).abs() < 2e-3);
}

/// Spec §6.3 / §6.4: with pseudised O, Hund's rule emerges — the triplet lies > 20 mHa below the
/// singlet — with *converged* open-shell solves (M6 Task 3 found the all-electron ones unconverged at
/// the 200-step cap; an unconverged solve here is an error, not a pass).
/// BLOCKED (M7 Task 6 report, "Open-shell O"): neither solve converges under integer aufbau occupations.
/// The singlet's p hole flips orientation every step (no aufbau-consistent fixed point; Δρ ≈ 2 after 60 steps).
/// The triplet's minority hole orientation is a near-zero mode (linear mixing: Δρ floor 7.3e-5 at 41³,
/// contracting 2.7e-4 per step; DIIS destabilises on it).
#[test]
#[ignore = "BLOCKED: open-shell O SCF does not converge (M7 Task 6 report); needs a ruling"]
fn pseudo_o_triplet_below_singlet() {
    let h = table(8).record.h_req.expect("O has h_req").min(0.20);
    let o = || vec![Nucleus { pos: [0.0; 3], z: 8 }];
    let t0 = Instant::now();
    let (e1, _, s1) = solve(h, 12.0, IonicPolicy::CoresOnly, o(), Some(1));
    let t1 = t0.elapsed().as_secs_f64();
    let (e3, _, s3) = solve(h, 12.0, IonicPolicy::CoresOnly, o(), Some(3));
    let t3 = t0.elapsed().as_secs_f64() - t1;
    eprintln!("pseudo-O at h = {h}: singlet {e1:.6} ({} steps, {t1:.1} s), triplet {e3:.6} ({} steps, {t3:.1} s), splitting {:.2} mHa; radial Hund multiplet (record T7/G-): see references.md",
              s1.iterations, s3.iterations, (e1 - e3) * 1e3);
    assert!(e3 < e1 - 0.020, "triplet {e3:.6} vs singlet {e1:.6}");
}

/// Spec §6.6 attribution (Task 5 physics review S2). The 3-D egg-box of one pseudised atom at its h_req (capped at 0.25):
/// - max |F| over the 8 offsets {±h/4}³, the gate's own probe (one atom, so |ΣF| = |F|);
/// - the energy spread over {0, h/2}³ (plan review M4);
/// - the record's T8(d) F_atom at that h, printed beside it.
/// The valence floor is ≥ |F| − F_atom. Reported and recorded in references.md; Task 7 reads it before the gate.
/// O (CoresOnly, triplet) and H (All) here, on the CPU. C and N, and h = 0.15 for all four, are measured on the GPU
/// in Task 10.
/// Deviation (Task 6): F is `frozen_force` (module doc), not `qm::forces` (Task 7), from solves at tol = eig_tol
/// = 1e-9; the δ = 2e-3 cross-check is printed beside it.
fn eggbox_floor(z: u8, mult: Option<u32>, policy: IonicPolicy) {
    let t = table(z);
    let h = t.record.h_req.expect("an M7 species has h_req").min(0.25);
    let f_atom: f64 = t.record.filter.iter().find(|r| (r.h - h).abs() < 1e-12).expect("ladder row").f_table.iter().sum();
    let s = setup(policy, h, &[z]);
    let g = Grid::cube(n_for(12.0, h), h);
    let (q, half) = (h / 4.0, h / 2.0);
    let t0 = Instant::now();
    let (mut fmax, mut cross, mut steps) = (0.0f64, 0.0f64, Vec::new());
    for sx in [-q, q] { for sy in [-q, q] { for sz in [-q, q] {
        let nuc = vec![Nucleus { pos: [sx, sy, sz], z }];
        let (_, st, stats) = solve_with(h, 12.0, s.clone(), nuc.clone(), mult, Some(1e-9));
        let (f, wide) = frozen_force(&g, &st, &s, &nuc, 0);
        eprintln!("  Z = {z}, R = ({sx:+.4}, {sy:+.4}, {sz:+.4}): F = ({:+.3e}, {:+.3e}, {:+.3e}) |F| {:.3e} (δ = 2e-3: |F| {:.3e}), {} SCF steps",
                  f[0], f[1], f[2], norm(f), norm(wide), stats.iterations);
        fmax = fmax.max(norm(f));
        cross = cross.max(norm([f[0] - wide[0], f[1] - wide[1], f[2] - wide[2]]));
        steps.push(stats.iterations);
    }}}
    let t_force = t0.elapsed().as_secs_f64();
    let mut es = Vec::new();
    for sx in [0.0, half] { for sy in [0.0, half] { for sz in [0.0, half] {
        es.push(solve_with(h, 12.0, s.clone(), vec![Nucleus { pos: [sx, sy, sz], z }], mult, None).0);
    }}}
    let spread = es.iter().cloned().fold(f64::MIN, f64::max) - es.iter().cloned().fold(f64::MAX, f64::min);
    eprintln!("Z = {z} at h = {h}: max |F| {fmax:.2e} Ha/Bohr over ±h/4 (tables F_atom {f_atom:.2e}; floor ≥ {:.2e}), E spread {spread:.2e} Ha over {{0, h/2}}³ (E: {es:.8?}); FD cross-check max |F(1e-4 Rich.) − F(2e-3)| {cross:.1e}; SCF steps {steps:?}; {t_force:.0} s for the 8 force solves",
              (fmax - f_atom).max(0.0));
    assert!(fmax.is_finite() && spread.is_finite());
}

/// The floor of H under `All` (`eggbox_floor`).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo")]
fn single_atom_eggbox_floor() {
    eggbox_floor(1, Some(2), IonicPolicy::All);
}

/// The floor of O (CoresOnly, triplet; `eggbox_floor`). BLOCKED (M7 Task 6 report, "Open-shell O"): the triplet does
/// not converge at tol 1e-9 (nor at 1e-6) under integer aufbau occupations.
#[test]
#[ignore = "BLOCKED: open-shell O SCF does not converge (M7 Task 6 report); needs a ruling"]
fn single_atom_eggbox_floor_oxygen() {
    eggbox_floor(8, Some(3), IonicPolicy::CoresOnly);
}

/// Report (not a gate; `--ignored`): cost of the pseudised solver against the all-electron one at equal h, and the
/// nonlocal share of H application (spec §5.3). The H application: O (p-local, one s projector) at its h_req 0.20 on
/// 61³. The whole solve: closed-shell Ne at h = 0.20 (the open-shell O does not converge, Task 6 report), pseudo
/// (`from_tables`: Ne has no h_req; 4 orbitals per spin, one s projector) against all-electron σ = h (5 per spin),
/// tol 1e-6 / eig 1e-7, 12 Bohr box.
#[test]
#[ignore = "report: cargo test -p validate --release --test pseudo -- --ignored --nocapture pseudo_cost_report"]
fn pseudo_cost_report() {
    use qm::eigen::{LocalHamiltonian, Operator};
    use qm::nonlocal::PseudoHamiltonian;
    let h = 0.20;
    let o = || vec![Nucleus { pos: [0.0; 3], z: 8 }];
    let s = setup(IonicPolicy::CoresOnly, h, &[8]);
    let g = Grid::cube(n_for(12.0, h), h);
    let t0 = Instant::now();
    let nl = NonlocalOp::build(&g, &o(), &s);
    let t_build = t0.elapsed().as_secs_f64();
    let pts: usize = nl.spheres.iter().map(|sp| sp.points.len()).sum();
    let v = ionic_local_potential(&g, &o(), &s);
    let x: Vec<f64> = (0..g.len()).map(|i| ((i * 7919) % 1000) as f64 * 1e-3).collect();
    let mut y = vec![0.0; g.len()];
    let reps = 20;
    let t0 = Instant::now();
    for _ in 0..reps { LocalHamiltonian { grid: &g, v: &v }.apply(&x, &mut y); }
    let t_loc = t0.elapsed().as_secs_f64() / reps as f64;
    let t0 = Instant::now();
    for _ in 0..reps { PseudoHamiltonian { grid: &g, v: &v, nl: &nl }.apply(&x, &mut y); }
    let t_ps = t0.elapsed().as_secs_f64() / reps as f64;
    eprintln!("O at h = {h}, {}³: {} sphere(s), {pts} points (radius {:.3}); build {:.1} ms; H·ψ local {:.2} ms, pseudo {:.2} ms → nonlocal share {:.1} %",
              g.n[0], nl.spheres.len(), match s.ionic(8) { Ionic::Pseudo(t) => t.projectors[0].radius, _ => 0.0 }, t_build * 1e3, t_loc * 1e3, t_ps * 1e3, 100.0 * (t_ps - t_loc) / t_ps);
    let ne = || vec![Nucleus { pos: [0.0; 3], z: 10 }];
    let ps = Arc::new(IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, BTreeMap::from([(10, Ionic::Pseudo(Arc::new(filtered(&table(10), h))))])));
    let t0 = Instant::now();
    let (ep, _, sp) = solve_with(h, 12.0, ps, ne(), Some(1), None);
    let tp = t0.elapsed().as_secs_f64();
    let mut sae = ScfSettings::for_grid(&g);
    sae.eig_tol = 1e-7;
    let t0 = Instant::now();
    let (_, ae_stats) = scf(&g, &System::new(ne(), 0, Some(1)), &sae, None).expect("AE Ne converges");
    let ta = t0.elapsed().as_secs_f64();
    eprintln!("Ne at h = {h}, {}³, tol 1e-6 / eig 1e-7: pseudo {tp:.1} s ({} steps, {:.2} s/step, E_valence {ep:.6}); all-electron σ = h {ta:.1} s ({} steps, {:.2} s/step, E {:.6}); ratio {:.2}",
              g.n[0], sp.iterations, tp / sp.iterations as f64, ae_stats.iterations, ta / ae_stats.iterations as f64, ae_stats.energy.total, tp / ta);
}

/// Report (not a gate; `--ignored`): is the frozen-energy Hellmann–Feynman force of `single_atom_eggbox_floor`
/// the derivative of the SCF energy? Pseudo-H (All) at h = 0.25, R = (h/4, h/4, h/4): F_x from `frozen_force` against
/// −dE/dX by central differences of tight SCF energies (tol = eig_tol = 1e-9) at δ = 0.01 and 0.02 and their
/// Richardson combination. `forces.rs` bounds the HF − dE/dR difference on this grid at ≲ 1e-5 (the Hartree map's
/// boundary asymmetry) plus the FD truncation.
#[test]
#[ignore = "report: cargo test -p validate --release --test pseudo -- --ignored --nocapture h_floor_hf_matches_energy_differences"]
fn h_floor_hf_matches_energy_differences() {
    let h = 0.25;
    let q = h / 4.0;
    let s = setup(IonicPolicy::All, h, &[1]);
    let g = Grid::cube(n_for(12.0, h), h);
    let at = |dx: f64| vec![Nucleus { pos: [q + dx, q, q], z: 1 }];
    let e = |dx: f64| solve_with(h, 12.0, s.clone(), at(dx), Some(2), Some(1e-9)).0;
    let (_, st, _) = solve_with(h, 12.0, s.clone(), at(0.0), Some(2), Some(1e-9));
    let (hf, _) = frozen_force(&g, &st, &s, &at(0.0), 0);
    let d = |delta: f64| -(e(delta) - e(-delta)) / (2.0 * delta);
    let (d1, d2) = (d(0.01), d(0.02));
    eprintln!("pseudo-H at R = (h/4)·(1,1,1), h = {h}: HF F_x {:+.4e}; −dE/dX FD(0.01) {d1:+.4e}, FD(0.02) {d2:+.4e}, Richardson {:+.4e}; HF − Richardson {:+.2e}",
              hf[0], (4.0 * d1 - d2) / 3.0, hf[0] - (4.0 * d1 - d2) / 3.0);
}
