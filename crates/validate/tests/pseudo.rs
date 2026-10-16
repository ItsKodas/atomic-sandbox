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
//! - Boxes are ≈ 12 Bohr cubes with a grid point at the origin (`n_for`), except where a test says otherwise (the O
//!   Hund gate: 14 Bohr, spec §6.3).
//! - Occupations: `DegenerateShells` (τ = `DEGENERACY_TAU`), the providers' choice under the pseudo policies (spec
//!   §5.7; Task 6 review ruling A); closed shells come out aufbau's.
//! - Every rigid-shift egg-box probe has the Dirichlet-wall term removed (spec §6.6; Task 6 review ruling C, `Wall`).
//! - Forces on pseudised states are Task 7's. Until then `qm::forces` refuses them, and the single-atom floor
//!   below takes the Hellmann–Feynman force as −dE/dR of the *frozen* energy terms that depend on R:
//!   ∫ρ V_loc(R), E_NL(R) at fixed orbitals, E_xc[ρ_σ + ½ρ_pc(R)] at fixed spin densities, and E_ii.
//!   This uses only Task 6's own functions (`frozen_force`). At self-consistency that is dE/dR (the
//!   grid does not move: no Pulay term, `forces.rs`), so those solves run at tol = eig_tol = 1e-9 (plan review S6).
//! - The heavy studies (the O₂ egg-box gate, the O per-term attribution) also run on the GPU since M7 Task 8a
//!   (`o2_eggbox_gate_gpu`, `o_floor_attribution_gpu`; `GPU_REPORT_TOL`): GPU solves, the same f64 frozen-state probes on
//!   the CPU from the downloaded state (`GpuSolver::ks_state`). The CPU reports stay as the f64 answer key.
use qm::ionic::{Ionic, IonicError, IonicPolicy, IonicSetup};
use qm::nonlocal::NonlocalOp;
use qm::nuclear::{centres, has_partial_core, ionic_local_potential, ionic_repulsion, partial_core_field, smeared_potential};
use qm::radial::{filter::filtered, generate, has_core, PseudoTable};
use qm::radial::{ground_state, transfer_configs, SpinMode};
use qm::{scf, Grid, KsState, Nucleus, Occupations, ScfSettings, System, DEGENERACY_TAU};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;
use qm_gpu::GpuSolver;

mod common;

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
    s.occupations = Occupations::DegenerateShells { tau: DEGENERACY_TAU };
    if let Some(t) = tol { (s.tol, s.eig_tol) = (t, t); }
    let (st, stats) = scf(&g, &sys, &s, None).expect("SCF converges");
    (stats.energy.total, st, stats)
}

/// At most this many SCF solves run at once in one test (each holds ≈ 170 MB at 61³: LOBPCG's 3k-block workspace and
/// the DIIS history; the machine is shared).
const MAX_PARALLEL_SOLVES: usize = 4;

/// `f` over `items` on at most `MAX_PARALLEL_SOLVES` threads, results in input order.
fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out: Vec<Mutex<Option<R>>> = items.iter().map(|_| Mutex::new(None)).collect();
    std::thread::scope(|sc| {
        for _ in 0..MAX_PARALLEL_SOLVES.min(items.len()) {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if i >= items.len() { break; }
                *out[i].lock().unwrap() = Some(f(&items[i]));
            });
        }
    });
    out.into_iter().map(|m| m.into_inner().unwrap().expect("every item ran")).collect()
}

/// The highest occupied level of a state (either spin, f > 0), for the wall's decay κ = √(−2ε).
fn highest_occupied(st: &KsState) -> f64 {
    st.evals_up.iter().zip(&st.occ_up).chain(st.evals_dn.iter().zip(&st.occ_dn)).filter(|(_, &f)| f > 0.0).map(|(&e, _)| e).fold(f64::MIN, f64::max)
}

/// The box-wall term of a rigid-shift probe (Task 6 review, ruling C; spec §6.6). The egg-box is periodic in each
/// shift component with period h, so the lattice translates ±h·ê_b see the unshifted geometry's egg-box and differ
/// from it only through the Dirichlet faces. Each face's term depends on the displacement normal to it and decays as
/// e^{−2κd}, κ = √(−2 ε_max) (highest occupied level), so per axis
///   E_w,b(s) = A₊(e^{2κs} − 1) + A₋(e^{−2κs} − 1),  F_w,b(s) = −2κ(A₊e^{2κs} − A₋e^{−2κs}),
/// with A± solved from E(±h·ê_b) − E(0). Review measurement (pseudo-H, h = 0.25, 12 Bohr): removes 1.651e-5 of
/// 1.675e-5 Ha/Bohr per component at h/4; the remainder, 2.4e-7, agrees with the periodic energy (2.3e-7).
pub struct Wall { pub kappa: f64, pub a: [[f64; 2]; 3] }

impl Wall {
    pub fn fit(h: f64, kappa: f64, e0: f64, e_plus: [f64; 3], e_minus: [f64; 3]) -> Wall {
        let (p, m) = ((2.0 * kappa * h).exp() - 1.0, (-2.0 * kappa * h).exp() - 1.0);
        let det = p * p - m * m;
        Wall { kappa, a: [0, 1, 2].map(|b| { let (dp, dm) = (e_plus[b] - e0, e_minus[b] - e0); [(p * dp - m * dm) / det, (p * dm - m * dp) / det] }) }
    }
    pub fn energy(&self, s: [f64; 3]) -> f64 {
        (0..3).map(|b| { let x = 2.0 * self.kappa * s[b]; self.a[b][0] * (x.exp() - 1.0) + self.a[b][1] * ((-x).exp() - 1.0) }).sum()
    }
    pub fn force(&self, s: [f64; 3]) -> [f64; 3] {
        [0, 1, 2].map(|b| { let x = 2.0 * self.kappa * s[b]; -2.0 * self.kappa * (self.a[b][0] * x.exp() - self.a[b][1] * (-x).exp()) })
    }
}

/// The R-dependent part of the energy at a frozen state (module doc): ∫ρ V_loc + E_NL + E_xc[ρ_σ + ½ρ_pc] + E_ii.
fn frozen_energy(g: &Grid, st: &KsState, setup: &IonicSetup, nuclei: &[Nucleus]) -> f64 {
    let rho = st.rho();
    let mut e = g.dot(&rho, &ionic_local_potential(g, nuclei, setup));
    let nl = NonlocalOp::build(g, nuclei, setup);
    e += nl.energy_weighted(&st.orbitals_up, &st.occ_up) + nl.energy_weighted(&st.orbitals_dn, &st.occ_dn); // f_iσ (spec §5.7)
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
/// `IonicSetup::build` at 0.15 must agree with Ne's record: certified iff h_req ≥ 0.15 (at `m7-tm-kb-nlcc-3` Ne had no
/// h_req — T8(a) missed by 0.35 mHa at the finest rung — and the build refused; at -4, c = ¼, Ne's h_req is 0.15). The
/// comparison itself uses the same filtered table through `IonicSetup::from_tables`: 3-D vs radial on one table is a
/// solver-integration test and does not depend on the T8 certificate. Reported: the finer-than-ladder path (`check()` at
/// h = 0.125, certified by its own T8 row).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo")]
fn pseudo_ne_3d_matches_radial() {
    let h = 0.15;
    let ft = Arc::new(filtered(&table(10), h));
    let ne_h_req = table(10).record.h_req;
    match (IonicSetup::build(IonicPolicy::CoresOnly, h, h, &[10], &mut |z| has_core(z), &mut |_| Ok(ft.clone())), ne_h_req) {
        (Ok(_), Some(hr)) if hr >= h - 1e-12 => eprintln!("Ne at h = {h}: certified (h_req {hr})"),
        (Err(IonicError::TooCoarse { z: 10, .. }), None) => eprintln!("Ne at h = {h}: TooCoarse (no h_req)"),
        (other, hr) => panic!("Ne at h = 0.15: the build {other:?} disagrees with the record's h_req {hr:?}"),
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

/// Spec §6.3 / §6.4 (Task 6 review ruling A): pseudo-O (open shell, spec §5.7). Both multiplicities are the spherical
/// ensemble, so the 3-D atom *is* the radial one, as for Ne. CPU, `CoresOnly`, h = h_req,O = 0.20, a 14 Bohr box (the
/// wall bias of a −0.27 Ha level is 0.3 mHa at 12 Bohr), the atom off a grid point, tol 1e-6 / eig 1e-7, cold; an
/// unconverged solve is an error.
/// - Gated: each of M = 1 and M = 3 within 2 mHa of the radial KB atom on the same filtered table (M = 1: the
///   spin-restricted reference configuration; M = 3: T3's Hund configuration), and E(3) < E(1) − 20 mHa.
/// - Asserted: every shared cluster's spread < τ/10.
/// - Reported: the splitting against the radial one and the all-electron spherical 54.14 mHa (references.md "M7 radial
///   atom": E(M=1) − E(M=3) = −74.470692 + 74.524830). LSDA's integer lowering is `pseudo_o_integer_triplet_report`.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo")]
fn pseudo_o_triplet_below_singlet() {
    let h = table(8).record.h_req.expect("O has h_req").min(0.20);
    let s = setup(IonicPolicy::CoresOnly, h, &[8]);
    let Ionic::Pseudo(ft) = s.ionic(8) else { panic!("O is pseudised under CoresOnly") };
    let t = table(8);
    let hund_ae = ground_state(8, SpinMode::Polarised).expect("the AE O ground state");
    assert_eq!(hund_ae.multiplicity, 3, "the radial AE O ground state is the triplet");
    let hund = transfer_configs(&t, &hund_ae).into_iter().find(|c| c.name == "Hund").expect("T3's Hund configuration");
    let r1 = qm::radial::kb::solve_kb_atom(&ft.grid, &ft.input(), &qm::radial::kb::reference_config(&t.partition), SpinMode::Restricted).unwrap().energy;
    let r3 = qm::radial::kb::solve_kb_atom(&ft.grid, &ft.input(), &hund.pp, SpinMode::Polarised).unwrap().energy;
    let o = || vec![Nucleus { pos: [0.013, -0.0091, 0.0052], z: 8 }];
    let mut both = par_map(&[1u32, 3], |&m| { let t0 = Instant::now(); let (e, st, stats) = solve_with(h, 14.0, s.clone(), o(), Some(m), None); (e, st, stats, t0.elapsed().as_secs_f64()) });
    let (e3, st3, s3, t3) = both.pop().unwrap();
    let (e1, st1, s1, t1) = both.pop().unwrap();
    for (m, st, stats) in [(1, &st1, &s1), (3, &st3, &s3)] {
        eprintln!("pseudo-O M = {m}: occ ↑ {:?} ε ↑ {:.6?}; occ ↓ {:?} ε ↓ {:.6?}; shared {:?}; max eigen residual {:.1e}; face {:.1e}; warnings {:?}",
                  st.occ_up, st.evals_up, st.occ_dn, st.evals_dn, stats.fractional, stats.max_eigen_residual, stats.boundary_density, stats.warnings);
        for c in &stats.fractional { assert!(c.spread < DEGENERACY_TAU / 10.0, "M = {m}: cluster {c:?}"); }
    }
    eprintln!("pseudo-O at h = {h}, 14 Bohr ({}³): M = 1 {e1:.6} vs radial {r1:.6} (Δ {:+.2} mHa; {} steps, {t1:.0} s); M = 3 {e3:.6} vs radial {r3:.6} (Δ {:+.2} mHa; {} steps, {t3:.0} s); \
               splitting E(1) − E(3) = {:.3} mHa (radial {:.3}, all-electron spherical 54.138)",
              n_for(14.0, h), (e1 - r1) * 1e3, s1.iterations, (e3 - r3) * 1e3, s3.iterations, (e1 - e3) * 1e3, (r1 - r3) * 1e3);
    assert!((e1 - r1).abs() < 2e-3, "M = 1: {e1:.6} vs radial {r1:.6}");
    assert!((e3 - r3).abs() < 2e-3, "M = 3: {e3:.6} vs radial {r3:.6}");
    assert!(e3 < e1 - 0.020, "triplet {e3:.6} vs singlet {e1:.6}");
}

/// Report (not a gate; `--ignored`): LSDA's non-spherical lowering E_int − E_sph of pseudo-O's triplet (spec §6.3,
/// rule 7), the "few mHa" the integer atom lies below the spherical ensemble. The same grid, box and position as the
/// Hund gate. E_int: integer aufbau (`Occupations::Aufbau`), linear mixing α = 0.3 driven one SCF step at a time
/// (each `scf` call converges trivially at tol = ∞ and returns E[ψ_out]; the next input is ρ_in + α(ρ_out − ρ_in)),
/// stopped when |E_k − E_{k−20}| < 1e-7 Ha (E is stationary, so its error is second order in the residual), recorded
/// with its Δρ. E_sph: `DegenerateShells`, as in the gate.
#[test]
#[ignore = "report: cargo test -p validate --release --test pseudo -- --ignored --nocapture pseudo_o_integer_triplet_report"]
fn pseudo_o_integer_triplet_report() {
    let h = 0.20;
    let s = setup(IonicPolicy::CoresOnly, h, &[8]);
    let o = || vec![Nucleus { pos: [0.013, -0.0091, 0.0052], z: 8 }];
    let (e_sph, _, _) = solve_with(h, 14.0, s.clone(), o(), Some(3), None);
    let g = Grid::cube(n_for(14.0, h), h);
    let sys = System::with_ionic(o(), 0, Some(3), s);
    let mut set = ScfSettings::for_grid(&g);
    (set.tol, set.max_iter, set.eig_tol) = (f64::INFINITY, 1, 1e-7);
    let (mut st, mut stats) = scf(&g, &sys, &set, None).unwrap();
    let (mut rho_up, mut rho_dn) = (st.rho_up.clone(), st.rho_dn.clone());
    let mut es = vec![stats.energy.total];
    let alpha = 0.3;
    while es.len() < 600 && !(es.len() > 20 && (es[es.len() - 1] - es[es.len() - 21]).abs() < 1e-7) {
        let warm = KsState { rho_up: rho_up.clone(), rho_dn: rho_dn.clone(), ..st };
        (st, stats) = scf(&g, &sys, &set, Some(&warm)).unwrap();
        for (r, o) in rho_up.iter_mut().zip(&st.rho_up).chain(rho_dn.iter_mut().zip(&st.rho_dn)) { *r = (*r + alpha * (o - *r)).max(0.0); }
        es.push(stats.energy.total);
    }
    let e_int = *es.last().unwrap();
    eprintln!("pseudo-O triplet at h = {h}, 14 Bohr: integer aufbau E_int {e_int:.7} after {} steps (last Δρ {:.2e}, |E_k − E_(k−20)| {:.1e}); spherical ensemble E_sph {e_sph:.7}; E_int − E_sph = {:+.3} mHa; ε ↓ {:.6?}",
              es.len(), stats.delta_rho, (es[es.len() - 1] - es[es.len().saturating_sub(21)]).abs(), (e_int - e_sph) * 1e3, st.evals_dn);
}

/// Spec §5.7 / review A.6: a lone pseudised O or N converges at every allowed multiplicity (the spherical ensemble), and
/// the lowest is Hund's: O M = 3, N M = 4. At each species' h_req, 12 Bohr, CoresOnly.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo")]
fn lone_o_and_n_scan_picks_hund() {
    for (z, ms, hund) in [(8u8, vec![1u32, 3, 5], 3u32), (7, vec![2, 4], 4)] {
        let h = table(z).record.h_req.expect("h_req");
        let s = setup(IonicPolicy::CoresOnly, h, &[z]);
        let es: Vec<(u32, f64, usize, usize)> = par_map(&ms, |&m| {
            let (e, _, stats) = solve_with(h, 12.0, s.clone(), vec![Nucleus { pos: [0.011, -0.007, 0.004], z }], Some(m), None);
            (m, e, stats.iterations, stats.fractional.len())
        });
        eprintln!("Z = {z} at h = {h}: (M, E, SCF steps, shared clusters) {es:.6?}");
        let best = es.iter().min_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
        assert_eq!(best, hund, "Z = {z}: the scan picks M = {best}");
    }
}

/// Spec §6.6 attribution (Task 5 physics review S2; Task 6 review ruling C). The 3-D egg-box of one pseudised atom at its
/// h_req (capped at 0.25), 12 Bohr box centred on a grid point:
/// - |F| over the 8 offsets {±h/4}³, the gate's own probe (one atom, so |ΣF| = |F|), raw, the Dirichlet wall's
///   (`Wall`, fitted from E(0) and E(h·ê_b); E(−h·ê_b) = E(h·ê_b) by the centred box's inversion symmetry) and
///   wall-subtracted, the egg-box;
/// - the energy spread over {0, h/2}³ (plan review M4), raw and after subtracting the wall energy;
/// - the record's T8(d) F_atom at that h, printed beside it.
/// The valence floor is ≥ |F − F_wall| − F_atom. Reported and recorded in references.md; Task 7 reads it before the
/// gate. Precondition (spec §6.6): max |F_wall| ≤ 3e-4 Ha/Bohr, else the box is too small for the model.
/// F is `frozen_force` (module doc; `qm::forces` is Task 7's) from solves at tol = eig_tol = 1e-9, all of which run in
/// parallel threads; the δ = 2e-3 cross-check is printed beside it.
fn eggbox_floor(z: u8, mult: Option<u32>, policy: IonicPolicy) {
    let t = table(z);
    let h = t.record.h_req.expect("an M7 species has h_req").min(0.25);
    let f_atom: f64 = t.record.filter.iter().find(|r| (r.h - h).abs() < 1e-12).expect("ladder row").f_table.iter().sum();
    let s = setup(policy, h, &[z]);
    let g = Grid::cube(n_for(12.0, h), h);
    let (q, half) = (h / 4.0, h / 2.0);
    let probes: Vec<[f64; 3]> = [-q, q].iter().flat_map(|&x| [-q, q].map(move |y| [x, y])).flat_map(|[x, y]| [-q, q].map(move |zz| [x, y, zz])).collect();
    let ripple: Vec<[f64; 3]> = [0.0, half].iter().flat_map(|&x| [0.0, half].map(move |y| [x, y])).flat_map(|[x, y]| [0.0, half].map(move |zz| [x, y, zz])).collect();
    let translates = [[h, 0.0, 0.0], [0.0, h, 0.0], [0.0, 0.0, h]];
    let t0 = Instant::now();
    // 8 force probes, 8 ripple shifts, 3 lattice translates, MAX_PARALLEL_SOLVES at a time (each solve is single-threaded);
    // a job returns only its numbers — the state (≈ 30 MB at 61³) is dropped inside it
    // (force, δ = 2e-3 force, SCF steps, every partly filled set shared: a fractional cluster and no release)
    let forces: Vec<([f64; 3], [f64; 3], usize, bool)> = par_map(&probes, |&p| {
        let nuc = vec![Nucleus { pos: p, z }];
        let (_, st, stats) = solve_with(h, 12.0, s.clone(), nuc.clone(), mult, Some(1e-9));
        let (f, wide) = frozen_force(&g, &st, &s, &nuc, 0);
        (f, wide, stats.iterations, stats.released.is_empty())
    });
    let shifts: Vec<[f64; 3]> = ripple.iter().chain(&translates).copied().collect();
    // each shift's occupation pattern is printed: spec §5.7's rule decides shared vs integer per site (Task 6 round 1)
    let energies: Vec<(f64, f64, String, bool)> = par_map(&shifts, |&p| {
        let (e, st, stats) = solve_with(h, 12.0, s.clone(), vec![Nucleus { pos: p, z }], mult, Some(1e-9));
        (e, highest_occupied(&st), format!("shared {:?}; released {}; f↓ {:?}", stats.fractional.iter().map(|c| (c.spin, c.count, c.f, c.spread)).collect::<Vec<_>>(),
                                           stats.released.len(), st.occ_dn), stats.released.is_empty())
    });
    for (p, (e, _, pattern, _)) in shifts.iter().zip(&energies) { eprintln!("  Z = {z}, shift ({:.3}, {:.3}, {:.3}): E {e:.9}; {pattern}", p[0], p[1], p[2]); }
    let all_shared = forces.iter().all(|f| f.3) && energies.iter().all(|e| e.3);
    let e0 = energies[0].0;
    let kappa = (-2.0 * energies[0].1).sqrt();
    let e_plus = [energies[8].0, energies[9].0, energies[10].0];
    let wall = Wall::fit(h, kappa, e0, e_plus, e_plus);
    let (mut raw, mut wall_max, mut egg, mut cross) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (p, &(f, wide, steps, _)) in probes.iter().zip(&forces) {
        let fw = wall.force(*p);
        let fe = [f[0] - fw[0], f[1] - fw[1], f[2] - fw[2]];
        eprintln!("  Z = {z}, R = ({:+.4}, {:+.4}, {:+.4}): F = ({:+.3e}, {:+.3e}, {:+.3e}), wall ({:+.3e}, {:+.3e}, {:+.3e}), egg-box ({:+.3e}, {:+.3e}, {:+.3e}) |F − F_wall| {:.3e}; δ = 2e-3: |F| {:.3e}; {steps} SCF steps",
                  p[0], p[1], p[2], f[0], f[1], f[2], fw[0], fw[1], fw[2], fe[0], fe[1], fe[2], norm(fe), norm(wide));
        raw = raw.max(norm(f));
        wall_max = wall_max.max(norm(fw));
        egg = egg.max(norm(fe));
        cross = cross.max(norm([f[0] - wide[0], f[1] - wide[1], f[2] - wide[2]]));
    }
    let es: Vec<f64> = energies[..8].iter().map(|e| e.0).collect();
    let corrected: Vec<f64> = es.iter().zip(&ripple).map(|(e, p)| e - wall.energy(*p)).collect();
    let spread = |v: &[f64]| v.iter().cloned().fold(f64::MIN, f64::max) - v.iter().cloned().fold(f64::MAX, f64::min);
    eprintln!("Z = {z} at h = {h}: max |F| raw {raw:.2e}, wall {wall_max:.2e}, egg-box (wall-subtracted) {egg:.2e} Ha/Bohr over ±h/4 (tables F_atom {f_atom:.2e}; valence floor ≥ {:.2e}); \
               κ = {kappa:.4} Bohr⁻¹, E(h·ê_b) − E(0) = [{:.3e}, {:.3e}, {:.3e}] Ha; E spread over {{0, h/2}}³ raw {:.2e}, wall-subtracted {:.2e} Ha (E: {es:.9?}); FD cross-check {cross:.1e}; {:.0} s",
              (egg - f_atom).max(0.0), e_plus[0] - e0, e_plus[1] - e0, e_plus[2] - e0, spread(&es), spread(&corrected), t0.elapsed().as_secs_f64());
    assert!(wall_max <= 3e-4, "the wall term {wall_max:.2e} exceeds 10× the gate: enlarge the box (spec §6.6)");
    assert!(egg.is_finite() && spread(&corrected).is_finite());
    // Task 6 re-review R5: no partly filled set released at any of the 19 sites, and the wall-subtracted ripple is the
    // egg-box (expected ~3e-6 Ha for O), not occupation switching (round 1's rule gave 4.11e-3)
    assert!(all_shared, "a partly filled set was released at some site (spec §5.7)");
    assert!(spread(&corrected) < 1e-5, "wall-subtracted {{0, h/2}}³ spread {:.3e} Ha", spread(&corrected));
}

/// The floor of H under `All` (`eggbox_floor`). Review expectation at 12 Bohr: raw 2.90e-5, wall 2.86e-5, egg-box ≈ 4e-7.
#[test]
#[ignore = "slow CPU report (19 SCF solves; O ≈ 47 min), run with --ignored; rerun on the GPU after M7 Task 8"]
fn single_atom_eggbox_floor() {
    eggbox_floor(1, Some(2), IonicPolicy::All);
}

/// The floor of O (CoresOnly, triplet, the shared spherical state of spec §5.7; `eggbox_floor`). Task 7's precondition.
#[test]
#[ignore = "slow CPU report (19 SCF solves; O ≈ 47 min), run with --ignored; rerun on the GPU after M7 Task 8"]
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


/// Task 6 re-review R5 (spec §5.7, τ's lower bound): at low-symmetry lattice sites every partly filled set of C, N, O
/// and F is shared in its converged state with spread ≤ τ/10. Each species at its h_req (capped where none), a 12 Bohr
/// box, sites (0, h/2, h/2), (h/2, 0, 0), (h/2, h/2, h/2), (h/4, h/4, h/4); cases O M = 1, 3, C M = 3, N M = 2 and F M = 2,
/// each at its record h_req. Reported: the step-1 spread (a one-step solve at tol = ∞),
/// which round 1's one-shot rule let decide. The box must not confine (no box-size warning is asserted): the review's 8 Bohr
/// box gave C (diffuse 2p, h = 0.30) a face density of 1.4e-4 and a spread of 1.21e-4 at (0, h/2, h/2), against 3.6e-5 at
/// 12 Bohr and 4.2e-5 at 16 Bohr (`c_spread_box_report`): the box's anisotropy, not the lattice's (Task 6 round 2).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo")]
fn degenerate_spread_at_low_symmetry_sites() {
    // each species at its h_req (Task 6: O 0.20, C 0.30, N 0.30 at -3; N 0.20 at -4, c = ¼; F 0.15)
    let at = |z: u8| table(z).record.h_req.expect("an M7 species has h_req");
    let cases: [(u8, u32, f64); 5] = [(8, 1, at(8)), (8, 3, at(8)), (6, 3, at(6)), (7, 2, at(7)), (9, 2, at(9))];
    let jobs: Vec<(u8, u32, f64, [f64; 3])> = cases.iter().flat_map(|&(z, m, h)| {
        [[0.0, h / 2.0, h / 2.0], [h / 2.0, 0.0, 0.0], [h / 2.0, h / 2.0, h / 2.0], [h / 4.0, h / 4.0, h / 4.0]].map(move |p| (z, m, h, p))
    }).collect();
    let setups: HashMap<(u8, u64), Arc<IonicSetup>> = cases.iter().map(|&(z, _, h)| ((z, h.to_bits()), setup(IonicPolicy::CoresOnly, h, &[z]))).collect();
    let rows = par_map(&jobs, |&(z, m, h, p)| {
        let s = setups[&(z, h.to_bits())].clone();
        let g = Grid::cube(n_for(12.0, h), h);
        let nuc = vec![Nucleus { pos: p, z }];
        let sys = System::with_ionic(nuc.clone(), 0, Some(m), s.clone());
        let mut one = ScfSettings::for_grid(&g);
        (one.tol, one.max_iter, one.eig_tol) = (f64::INFINITY, 1, 1e-7);
        one.occupations = Occupations::DegenerateShells { tau: DEGENERACY_TAU };
        let (_, first) = scf(&g, &sys, &one, None).unwrap();
        let (_, _, stats) = solve_with(h, 12.0, s, nuc, Some(m), None);
        (z, m, h, p, first.fractional.iter().map(|c| c.spread).fold(0.0f64, f64::max), stats)
    });
    let mut bad = Vec::new();
    for (z, m, h, p, step1, stats) in &rows {
        let worst = stats.fractional.iter().map(|c| c.spread).fold(0.0f64, f64::max);
        eprintln!("Z = {z} M = {m} h = {h} at ({:.3}, {:.3}, {:.3}): step-1 spread {step1:.2e}; converged shared {:?}, released {:?}; worst spread {worst:.2e} ({} steps)",
                  p[0], p[1], p[2], stats.fractional.iter().map(|c| (c.spin, c.count, c.f)).collect::<Vec<_>>(), stats.released, stats.iterations);
        let confined = stats.warnings.iter().any(|w| w.contains("box too small"));
        if stats.fractional.is_empty() || !stats.released.is_empty() || worst > DEGENERACY_TAU / 10.0 || confined { bad.push((*z, *m, *p)); }
    }
    assert!(bad.is_empty(), "not shared with spread ≤ τ/10 at: {bad:?} (Task 6 re-review R1-M2: back to physics review, never a larger τ)");
}

/// One frozen energy term's −dE/dR_I (Richardson δ = 1e-4, 2e-4), for the per-term attribution.
fn term_force(nuclei: &[Nucleus], i: usize, e: &dyn Fn(&[Nucleus]) -> f64) -> [f64; 3] {
    [0, 1, 2].map(|a| {
        let d = |delta: f64| { let (mut p, mut m) = (nuclei.to_vec(), nuclei.to_vec()); p[i].pos[a] += delta; m[i].pos[a] -= delta; -(e(&p) - e(&m)) / (2.0 * delta) };
        (4.0 * d(1e-4) - d(2e-4)) / 3.0
    })
}

/// Report (not a gate; `--ignored`; Task 6 re-review R1-M3, before Task 7's O₂ gate): the per-term split of pseudo-O's
/// egg-box force. The O triplet (shared, spec §5.7) at R = (h/4)·(1, 1, 1) in a 16 Bohr box (the wall ≲ 2e-6 there,
/// not subtracted), h = 0.20 and 0.15, tol = eig_tol = 1e-9; the frozen Hellmann–Feynman terms F_LR = −∂/∂R ∫ρV_LR,
/// F_SR = −∂/∂R ∫ρV_SR, F_NL = −∂/∂R Σ f E_KB c², F_pc = −∂/∂R E_xc[ρ_σ + ½ρ_pc(R)]. The remedy follows the dominant
/// term (review ruling 2): XC/NLCC → XC on a refined grid; V_SR → a smoother model core or softer f; F_NL above its T8(d)
/// share → Ono–Hirose. The GPU twin is `o_floor_attribution_gpu`.
#[test]
#[ignore = "slow CPU report (hours), run with --ignored; the GPU twin is o_floor_attribution_gpu: cargo test -p validate --release --test pseudo -- --ignored --nocapture o_floor_attribution_report"]
fn o_floor_attribution_report() {
    for h in [0.20, 0.15] {
        let s = setup(IonicPolicy::CoresOnly, h, &[8]);
        let g = Grid::cube(n_for(16.0, h), h);
        let nuc = vec![Nucleus { pos: [h / 4.0; 3], z: 8 }];
        let t0 = Instant::now();
        let (e, st, stats) = solve_with(h, 16.0, s.clone(), nuc.clone(), Some(3), Some(1e-9));
        eprintln!("pseudo-O triplet at (h/4)³, h = {h}, 16 Bohr ({}³), CPU tol 1e-9: {}; E {e:.9}, {} steps, shared {:?}, released {}, face {:.1e}; {:.0} s",
                  g.n[0], attribution_line(&g, &st, &s, &nuc), stats.iterations, stats.fractional.iter().map(|c| (c.spin, c.count, c.spread)).collect::<Vec<_>>(),
                  stats.released.len(), stats.boundary_density, t0.elapsed().as_secs_f64());
    }
}

/// The per-term frozen forces of nucleus 0 (F_LR, F_SR, F_NL, F_pc; Richardson, `term_force`) and the whole `frozen_force`.
fn attribution_terms(g: &Grid, st: &KsState, s: &IonicSetup, nuc: &[Nucleus]) -> ([[f64; 3]; 4], [f64; 3]) {
    let rho = st.rho();
    let lr = |n: &[Nucleus]| g.dot(&rho, &smeared_potential(g, &centres(n, s)));
    let sr = |n: &[Nucleus]| { let v = ionic_local_potential(g, n, s); let l = smeared_potential(g, &centres(n, s));
                                g.dot(&rho, &v.iter().zip(&l).map(|(a, b)| a - b).collect::<Vec<_>>()) };
    let nl = |n: &[Nucleus]| { let op = NonlocalOp::build(g, n, s); op.energy_weighted(&st.orbitals_up, &st.occ_up) + op.energy_weighted(&st.orbitals_dn, &st.occ_dn) };
    let pc = |n: &[Nucleus]| { let c = partial_core_field(g, n, s);
        let u: Vec<f64> = st.rho_up.iter().zip(&c).map(|(a, x)| a + 0.5 * x).collect();
        let d: Vec<f64> = st.rho_dn.iter().zip(&c).map(|(a, x)| a + 0.5 * x).collect();
        g.integrate(&qm::xc::lsda_pw92(&u, &d).energy_density) };
    let terms: [&dyn Fn(&[Nucleus]) -> f64; 4] = [&lr, &sr, &nl, &pc];
    (terms.map(|e| term_force(nuc, 0, e)), frozen_force(g, st, s, nuc, 0).0)
}

/// The per-term frozen forces of one state (`o_floor_attribution_report`'s line): F_LR, F_SR, F_NL, F_pc, their sum and
/// the whole `frozen_force`.
fn attribution_line(g: &Grid, st: &KsState, s: &IonicSetup, nuc: &[Nucleus]) -> String {
    let rho = st.rho();
    let lr = |n: &[Nucleus]| g.dot(&rho, &smeared_potential(g, &centres(n, s)));
    let sr = |n: &[Nucleus]| { let v = ionic_local_potential(g, n, s); let l = smeared_potential(g, &centres(n, s));
                                g.dot(&rho, &v.iter().zip(&l).map(|(a, b)| a - b).collect::<Vec<_>>()) };
    let nl = |n: &[Nucleus]| { let op = NonlocalOp::build(g, n, s); op.energy_weighted(&st.orbitals_up, &st.occ_up) + op.energy_weighted(&st.orbitals_dn, &st.occ_dn) };
    let pc = |n: &[Nucleus]| { let c = partial_core_field(g, n, s);
        let u: Vec<f64> = st.rho_up.iter().zip(&c).map(|(a, x)| a + 0.5 * x).collect();
        let d: Vec<f64> = st.rho_dn.iter().zip(&c).map(|(a, x)| a + 0.5 * x).collect();
        g.integrate(&qm::xc::lsda_pw92(&u, &d).energy_density) };
    let terms: [(&str, &dyn Fn(&[Nucleus]) -> f64); 4] = [("LR", &lr), ("SR", &sr), ("NL", &nl), ("pc", &pc)];
    let mut total = [0.0; 3];
    let mut line = String::new();
    for (name, e) in terms {
        let f = term_force(nuc, 0, e);
        for a in 0..3 { total[a] += f[a]; }
        line += &format!("F_{name} ({:+.3e}, {:+.3e}, {:+.3e}) |{:.2e}|; ", f[0], f[1], f[2], norm(f));
    }
    let (whole, _) = frozen_force(g, st, s, nuc, 0);
    format!("{line}sum ({:+.3e}, {:+.3e}, {:+.3e}) |{:.2e}| vs frozen_force |{:.2e}|", total[0], total[1], total[2], norm(total), norm(whole))
}

/// Comma-separated scientific notation.
fn sci(v: &[f64]) -> String { v.iter().map(|x| format!("{x:.2e}")).collect::<Vec<_>>().join(", ") }

/// −∂E_frozen/∂T for a rigid translation T of every nucleus: Σ_I F_I (Richardson δ = 1e-4, 2e-4).
fn frozen_sum_force(g: &Grid, st: &KsState, setup: &IonicSetup, nuclei: &[Nucleus]) -> [f64; 3] {
    [0, 1, 2].map(|a| {
        let d = |delta: f64| {
            let shift = |sgn: f64| nuclei.iter().map(|n| { let mut m = *n; m.pos[a] += sgn * delta; m }).collect::<Vec<_>>();
            -(frozen_energy(g, st, setup, &shift(1.0)) - frozen_energy(g, st, setup, &shift(-1.0))) / (2.0 * delta)
        };
        (4.0 * d(1e-4) - d(2e-4)) / 3.0
    })
}

/// O₂ of the early gate run, rigidly shifted by `s`: the plan's geometry (R = 2.28, offset (0.07, 0.03)).
fn o2_at(s: [f64; 3]) -> Vec<Nucleus> {
    vec![Nucleus { pos: [-1.14 + s[0], 0.07 + s[1], 0.03 + s[2]], z: 8 }, Nucleus { pos: [1.14 + s[0], 0.07 + s[1], 0.03 + s[2]], z: 8 }]
}

/// The O₂ probe's shifts at h: the 8 force probes {±h/4}³, then the 14 energy shifts — the ripple {0, h/2}³ and the
/// 6 lattice translates ±h·ê_b.
fn o2_shifts(h: f64) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let (q, half) = (h / 4.0, h / 2.0);
    let probes: Vec<[f64; 3]> = [-q, q].iter().flat_map(|&x| [-q, q].map(move |y| [x, y])).flat_map(|[x, y]| [-q, q].map(move |zz| [x, y, zz])).collect();
    let ripple: Vec<[f64; 3]> = [0.0, half].iter().flat_map(|&x| [0.0, half].map(move |y| [x, y])).flat_map(|[x, y]| [0.0, half].map(move |zz| [x, y, zz])).collect();
    let translates = [[h, 0.0, 0.0], [0.0, h, 0.0], [0.0, 0.0, h], [-h, 0.0, 0.0], [0.0, -h, 0.0], [0.0, 0.0, -h]];
    (probes, ripple.iter().chain(&translates).copied().collect())
}

/// The O₂ probe's analysis (`o2_eggbox_gate_early_run`'s doc), shared by the CPU and GPU runs: `forces[i]` = (ΣF, SCF
/// steps, shared + released clusters) at `probes[i]`, `energies[j]` = (E, ε_max) at `o2_shifts`'s j-th energy shift.
/// Prints the per-shift lines and the verdict line; returns the wall-subtracted max |ΣF| and the wall maximum.
fn o2_summary(label: &str, h: f64, forces: &[([f64; 3], usize, usize)], energies: &[(f64, f64)], secs: f64) -> (f64, f64) {
    let (probes, shifts) = o2_shifts(h);
    let e0 = energies[0].0;
    let kappa = (-2.0 * energies[0].1).sqrt();
    let e_plus = [energies[8].0, energies[9].0, energies[10].0];
    let e_minus = [energies[11].0, energies[12].0, energies[13].0];
    let wall = Wall::fit(h, kappa, e0, e_plus, e_minus);
    let (mut raw, mut wall_max, mut corrected) = (0.0f64, 0.0f64, 0.0f64);
    let mut fy = Vec::new();
    for (p, &(f, steps, shared)) in probes.iter().zip(forces) {
        let fw = wall.force(*p);
        let fc = [f[0] - fw[0], f[1] - fw[1], f[2] - fw[2]];
        eprintln!("  O₂ [{label}] shift ({:+.3}, {:+.3}, {:+.3}): ΣF ({:+.3e}, {:+.3e}, {:+.3e}), wall ({:+.3e}, {:+.3e}, {:+.3e}), corrected ({:+.3e}, {:+.3e}, {:+.3e}) |{:.3e}|; {steps} steps; shared/released clusters {shared}",
                  p[0], p[1], p[2], f[0], f[1], f[2], fw[0], fw[1], fw[2], fc[0], fc[1], fc[2], norm(fc));
        raw = raw.max(norm(f));
        wall_max = wall_max.max(norm(fw));
        corrected = corrected.max(norm(fc));
        fy.push(fc[1]);
    }
    let es: Vec<f64> = energies[..8].iter().map(|e| e.0).collect();
    let corr_e: Vec<f64> = es.iter().zip(&shifts[..8]).map(|(e, p)| e - wall.energy(*p)).collect();
    let spread = |v: &[f64]| v.iter().cloned().fold(f64::MIN, f64::max) - v.iter().cloned().fold(f64::MAX, f64::min);
    eprintln!("O₂ (CoresOnly) [{label}] at h = {h}, 14 Bohr: max |ΣF| raw {raw:.2e}, wall {wall_max:.2e}, wall-subtracted {corrected:.2e} Ha/Bohr → {} the 3e-5 gate; ΣF_y (corrected) [{}]; \
               κ {kappa:.4}; E(±h·ê_b) − E(0) = [{}] / [{}]; ripple {{0, h/2}}³ raw {:.2e}, wall-subtracted {:.2e} Ha; {secs:.0} s",
              if corrected < 3e-5 { "PASSES" } else { "MISSES" }, sci(&fy), sci(&e_plus.map(|e| e - e0)), sci(&e_minus.map(|e| e - e0)), spread(&es), spread(&corr_e));
    (corrected, wall_max)
}

/// Report, run early (Task 6 re-review ruling 2: Task 7's precondition has fired, so the O₂ gate is run and recorded;
/// `--ignored`). Task 7 owns the asserting gate (`o2_max_shift_sum_of_forces_below_3e_5`, with `qm::forces`); this is
/// the same probe with the frozen Hellmann–Feynman ΣF. O₂ (`CoresOnly`, triplet: π*↑ full, the §5.7 rule inert), the
/// plan's geometry (R = 2.28, offset (0.07, 0.03)), h = h_req,O = 0.20, 14 Bohr, tol = eig_tol = 1e-9:
/// - max over the 8 rigid shifts {±h/4}³ of |ΣF − F_wall|, the wall fitted from the 6 lattice translates ±h·ê_b (the
///   molecule is off the box centre: no inversion shortcut), κ = √(−2ε_max) of the unshifted state;
/// - the raw and wall maxima, ΣF_y per shift (the like-for-like component), the {0, h/2}³ ripple wall-subtracted;
/// - the verdict against 3e-5 (printed; the gate is not moved — review ruling 2 predicts ≈ 4e-5).
/// The GPU twin is `o2_eggbox_gate_gpu`.
#[test]
#[ignore = "slow CPU report (hours), run with --ignored; the GPU twin is o2_eggbox_gate_gpu: cargo test -p validate --release --test pseudo -- --ignored --nocapture o2_eggbox_gate_early_run"]
fn o2_eggbox_gate_early_run() {
    let h = 0.25f64.min(table(8).record.h_req.expect("O has h_req"));
    let s = setup(IonicPolicy::CoresOnly, h, &[8]);
    let g = Grid::cube(n_for(14.0, h), h);
    let (probes, shifts) = o2_shifts(h);
    let t0 = Instant::now();
    let forces: Vec<([f64; 3], usize, usize)> = par_map(&probes, |&p| {
        let nuc = o2_at(p);
        let (_, st, stats) = solve_with(h, 14.0, s.clone(), nuc.clone(), Some(3), Some(1e-9));
        (frozen_sum_force(&g, &st, &s, &nuc), stats.iterations, stats.fractional.len() + stats.released.len())
    });
    let energies: Vec<(f64, f64)> = par_map(&shifts, |&p| { let (e, st, _) = solve_with(h, 14.0, s.clone(), o2_at(p), Some(3), Some(1e-9)); (e, highest_occupied(&st)) });
    let (_, wall_max) = o2_summary("CPU, tol 1e-9", h, &forces, &energies, t0.elapsed().as_secs_f64());
    assert!(wall_max <= 3e-4, "the wall term {wall_max:.2e} exceeds 10× the gate: enlarge the box (spec §6.6)");
}

// ---- M7 Task 8a: the heavy reports on the GPU (CPU = the f64 answer key on small cases only; progress.md ruling) ----

/// The SCF tolerance of the GPU reports (Δρ, e). The CPU reports' 1e-9 (plan review S6: the frozen Hellmann–Feynman force
/// is dE/dR only at self-consistency) is far below the f32 Δρ floor (~3e-8·N, `qm_gpu::scf` "The mixing step"), so the
/// GPU solves stop at this tol (the GPU suite's standard), above the floor for every report system (O₂, N = 12, floors at
/// Δρ ≈ 5.5e-7: 3e-7 does not converge); the frozen force then carries a first-order error ∫δρ ∇V. How large that is,
/// is measured, not assumed: the GPU attribution and O₂ numbers are printed beside the CPU runs' (references.md, Task 6
/// round 2) and the difference is the GPU protocol's error (Task 8a also ran the attribution at 3e-7: the same terms
/// to ≤ 1e-8 Ha/Bohr; references.md). Measured: the lone atom's terms agree with the CPU to ≤ 2 % (no soft dipolar mode),
/// but O₂'s ΣF carries 0.9–2.3e-5 Ha/Bohr (cold) and up to 6.7e-5 (warm from a neighbour) against the same state
/// polished to 1e-9 on the CPU (`o2_gpu_protocol_report`). So a molecular GPU frozen-HF ΣF is printed only with that
/// band and is never the gate's number: the gate is measured from GPU energies (`o2_eggbox_gate_gpu`, spec §6.6 "Instrument").
const GPU_REPORT_TOL: f64 = 1e-6;

/// The device's live allocations (MB; wgpu's allocator report, Vulkan/DX12), NaN where the backend has none.
fn vram_mb(ctx: &gpu::GpuContext) -> f64 { ctx.device.generate_allocator_report().map_or(f64::NAN, |r| r.total_allocated_bytes as f64 / 1e6) }

/// `solve_with` on the GPU: the same settings (`for_grid`, eig_tol 1e-7 — the GPU raises it to 3× its f32 floor —,
/// `DegenerateShells`), tol `GPU_REPORT_TOL`, cold; the state downloaded as a `KsState` for the f64 frozen probes.
fn gpu_solve_with(solver: &mut GpuSolver, setup: Arc<IonicSetup>, nuclei: Vec<Nucleus>, mult: Option<u32>) -> (f64, KsState, qm::ScfStats) {
    gpu_solve_from(solver, setup, nuclei, mult, false)
}

/// `gpu_solve_with`, cold or warm from the solver's last state (`GpuSolver::solve`'s `warm`).
fn gpu_solve_from(solver: &mut GpuSolver, setup: Arc<IonicSetup>, nuclei: Vec<Nucleus>, mult: Option<u32>, warm: bool) -> (f64, KsState, qm::ScfStats) {
    let g = solver.grid().clone();
    let sys = System::with_ionic(nuclei, 0, mult, setup);
    let mut s = ScfSettings::for_grid(&g);
    s.eig_tol = 1e-7;
    s.tol = GPU_REPORT_TOL;
    s.occupations = Occupations::DegenerateShells { tau: DEGENERACY_TAU };
    let (_, stats) = solver.solve(&sys, &s, warm).unwrap_or_else(|e| panic!("GPU SCF: {e}"));
    let st = solver.ks_state(solver.state().expect("a converged state"));
    (stats.energy.total, st, stats)
}

/// `o_floor_attribution_report` on the GPU (M7 Task 8a): the same O triplet, positions, boxes and h (0.20, 0.15); the
/// state solved on the GPU at `GPU_REPORT_TOL`, the terms taken in f64 on the CPU from the downloaded state. Compare with
/// the CPU table (references.md "Round 2" / Task 6 report: h = 0.20 F_LR +2.91e-5, F_SR +3.26e-5, F_NL −5.49e-5,
/// F_pc −2.12e-5 per component, net |2.50e-5|; h = 0.15 F_pc −4.79e-5, net |7.07e-5|).
#[test]
#[ignore = "GPU report: cargo test -p validate --release --test pseudo -- --ignored --nocapture o_floor_attribution_gpu"]
fn o_floor_attribution_gpu() {
    let Some(ctx) = common::gpu("pseudo") else { return };
    for h in [0.20, 0.15] {
        let s = setup(IonicPolicy::CoresOnly, h, &[8]);
        let g = Grid::cube(n_for(16.0, h), h);
        let nuc = vec![Nucleus { pos: [h / 4.0; 3], z: 8 }];
        let mut solver = GpuSolver::new(ctx.clone(), &g);
        let t0 = Instant::now();
        let (e, st, stats) = gpu_solve_with(&mut solver, s.clone(), nuc.clone(), Some(3));
        let t_solve = t0.elapsed().as_secs_f64();
        let line = attribution_line(&g, &st, &s, &nuc);
        eprintln!("pseudo-O triplet at (h/4)³, h = {h}, 16 Bohr ({}³), GPU tol {GPU_REPORT_TOL:.0e} (Δρ {:.2e}): {line}; E {e:.9}, {} steps, shared {:?}, released {}, face {:.1e}; GPU solve {t_solve:.1} s, total {:.0} s; VRAM allocated {:.0} MB",
                  g.n[0], stats.delta_rho, stats.iterations, stats.fractional.iter().map(|c| (c.spin, c.count, c.spread)).collect::<Vec<_>>(),
                  stats.released.len(), stats.boundary_density, t0.elapsed().as_secs_f64(), vram_mb(&ctx));
    }
}

/// The 3-D confirmation of the NLCC crossover (NLCC ruling, 2026-10-16, Ruling 5 step 2; `pseudo_gen` `CONFIRMED_3D`):
/// the lone O triplet (CoresOnly, shared, spec §5.7) at R = (h/4)(1, 1, 1), 16 Bohr, h = 0.20 and 0.15, solved on the GPU at
/// `GPU_REPORT_TOL` with the f64 per-term attribution (qualified to ≤ 2 %, Task 8a), at the recipe's c.
/// - Check a, per component: |F_pc,3D| ≤ 2·|F_pc,radial| + 0.02·max(|F_LR|, |F_SR|, |F_NL|), F_pc,radial = T8(e)'s
///   `pc_alias` of the filtered radial KB atom in the Hund configuration at (h/4)(1, 1, 1) (the estimator check's 2×, plus
///   the GPU attribution's qualified 2 % of the largest term).
/// - Check b: |frozen_force| at 0.15 ≤ 3.5e-5 Ha/Bohr, half the -3 value 7.07e-5 (CPU, Task 6 round 2). The ruling's
///   prediction ≈ 1.2e-5 (the c = 1 sum of the non-pc terms) + 0.71e-5 per component.
/// A failure is the ruling's automatic fallback to c = 1 (`m7-tm-kb-nlcc-4` with c = 1).
#[test]
#[ignore = "GPU (≈ 3 min): cargo test -p validate --release --test pseudo -- --ignored --nocapture --exact lone_o_attribution_confirms_the_crossover_gpu"]
fn lone_o_attribution_confirms_the_crossover_gpu() {
    let Some(ctx) = common::gpu("pseudo") else { return };
    let t = table(8);
    let hund_ae = ground_state(8, SpinMode::Polarised).unwrap();
    let hund = transfer_configs(&t, &hund_ae).into_iter().find(|c| c.name == "Hund").expect("Hund");
    let mut fails = Vec::new();
    for h in [0.20, 0.15] {
        let s = setup(IonicPolicy::CoresOnly, h, &[8]);
        let Ionic::Pseudo(ft) = s.ionic(8) else { panic!("O is pseudised") };
        let atom = qm::radial::kb::solve_kb_atom(&ft.grid, &ft.input(), &hund.pp, hund.spin).unwrap();
        let (_, radial) = qm::radial::filter::pc_alias(ft, &atom);
        let g = Grid::cube(n_for(16.0, h), h);
        let nuc = vec![Nucleus { pos: [h / 4.0; 3], z: 8 }];
        let mut solver = GpuSolver::new(ctx.clone(), &g);
        let t0 = Instant::now();
        let (_, st, stats) = gpu_solve_with(&mut solver, s.clone(), nuc.clone(), Some(3));
        let (terms, whole) = attribution_terms(&g, &st, &s, &nuc);
        let [lr, sr, nl, pc] = terms;
        for a in 0..3 {
            let bound = 2.0 * radial[a].abs() + 0.02 * lr[a].abs().max(sr[a].abs()).max(nl[a].abs());
            if pc[a].abs() > bound { fails.push(format!("h = {h}, component {a}: |F_pc,3D| {:.3e} > {bound:.3e}", pc[a].abs())); }
        }
        eprintln!("lone O (CoresOnly triplet, c = {}) at (h/4)(1,1,1), h = {h}, 16 Bohr ({}³), GPU tol {GPU_REPORT_TOL:.0e} ({} steps): F_LR ({}), F_SR ({}), F_NL ({}), F_pc ({});                    radial T8(e) F_pc (Hund) ({}); frozen_force ({}) |{:.3e}|; {:.0} s",
                  qm::radial::LEDGER.nlcc_crossover, g.n[0], stats.iterations, sci(&lr), sci(&sr), sci(&nl), sci(&pc), sci(&radial), sci(&whole), norm(whole), t0.elapsed().as_secs_f64());
        if h == 0.15 && norm(whole) > 3.5e-5 { fails.push(format!("h = 0.15: |frozen_force| {:.3e} > 3.5e-5", norm(whole))); }
    }
    assert!(fails.is_empty(), "the crossover's 3-D confirmation fails (NLCC ruling: fall back to c = 1): {}", fails.join("; "));
}

/// The CPU tol-1e-9 per-probe ΣF of the O₂ gate on record (raw, Ha/Bohr): each the frozen ΣF of the cold GPU state
/// polished by the CPU SCF to tol = eig_tol = 1e-9 (the CPU report's protocol). Probe signs σ: the shift (h/4)σ. The
/// probes (∓1)(1,1,1) are Task 8a's `o2_gpu_protocol_report` (c); the other six `o2_cpu_probe_polish_report` (2026-10-16:
/// 15–56 CPU steps, 345–622 s each; the instrument's E_f64 of the GPU state equals the polished state's to ≤ 1.1e-9 Ha).
const O2_CPU_PROBES: &[([i32; 3], [f64; 3])] = &[
    ([-1, -1, -1], [-3.550e-5, -2.953e-5, 2.953e-5]), ([1, 1, 1], [3.424e-5, 2.381e-5, -3.369e-5]),
    ([-1, -1, 1], [-3.4765e-5, -3.0736e-5, -3.2479e-5]), ([-1, 1, -1], [-3.4744e-5, 2.2600e-5, 3.0738e-5]),
    ([-1, 1, 1], [-3.4246e-5, 2.3805e-5, -3.3687e-5]), ([1, -1, -1], [3.5497e-5, -2.9529e-5, 2.9529e-5]),
    ([1, -1, 1], [3.4765e-5, -3.0738e-5, -3.2478e-5]), ([1, 1, -1], [3.4743e-5, 2.2599e-5, 3.0738e-5]),
];
/// The CPU run's wall inputs on record (Task 6 round 2, references.md): κ and E(±h·ê_b) − E(0) (Ha).
const O2_CPU_WALL: (f64, [f64; 3], [f64; 3]) = (0.7143, [3.82e-6, 1.71e-6, 1.30e-6], [3.82e-6, 2.94e-7, 6.93e-7]);
/// The CPU run's per-shift wall-subtracted magnitudes on record (x, then y and z; Ha/Bohr): every probe's |ΣF_a − F_wall,a|
/// lies in these ranges (references.md, Task 6 round 2).
const O2_CPU_RANGES: [(f64, f64); 3] = [(4.37e-5, 4.50e-5), (2.85e-5, 2.98e-5), (2.85e-5, 2.98e-5)];
/// The estimator qualifies when u_I ≤ this (spec §6.6, "Instrument").
const O2_QUALIFY: f64 = 5e-6;

/// The qualification of an estimator's eight probe vectors against the CPU values on record (spec §6.6, "Instrument"):
/// u_I = the largest per-component difference — exactly at every recorded probe (wall-subtracted with the CPU run's own
/// wall `O2_CPU_WALL`), and as the distance outside the recorded per-component ranges (`O2_CPU_RANGES`) at all eight.
/// Prints one line per probe; returns (u_I, G = max |ΣF|).
fn o2_qualify(label: &str, h: f64, probes: &[([i32; 3], [f64; 3])]) -> (f64, f64) {
    let q = h / 4.0;
    let cpu_wall = Wall::fit(h, O2_CPU_WALL.0, 0.0, O2_CPU_WALL.1, O2_CPU_WALL.2);
    let (mut big_g, mut u) = (0.0f64, 0.0f64);
    for (p, f) in probes {
        big_g = big_g.max(norm(*f));
        let range_dist = (0..3).map(|a| { let (lo, hi) = O2_CPU_RANGES[a]; (lo - f[a].abs()).max(f[a].abs() - hi).max(0.0) }).fold(0.0, f64::max);
        u = u.max(range_dist);
        let mut line = format!("  O₂ [{label}] probe ({:+.3}, {:+.3}, {:+.3}): ΣF ({:+.3e}, {:+.3e}, {:+.3e}) |{:.3e}|; outside the CPU ranges by {range_dist:.1e}",
                               p[0] as f64 * q, p[1] as f64 * q, p[2] as f64 * q, f[0], f[1], f[2], norm(*f));
        if let Some((_, raw)) = O2_CPU_PROBES.iter().find(|(sg, _)| sg == p) {
            let fw = cpu_wall.force(p.map(|c| c as f64 * q));
            let cpu = [raw[0] - fw[0], raw[1] - fw[1], raw[2] - fw[2]];
            let d = (0..3).map(|a| (f[a] - cpu[a]).abs()).fold(0.0, f64::max);
            u = u.max(d);
            line += &format!("; CPU 1e-9 (wall-subtracted) ({:+.3e}, {:+.3e}, {:+.3e}) |{:.3e}|, max component |Δ| {d:.2e}", cpu[0], cpu[1], cpu[2], norm(cpu));
        }
        eprintln!("{line}");
    }
    (u, big_g)
}

/// Spec §6.6's verdict from (G, u_I).
fn o2_verdict(big_g: f64, u: f64) -> &'static str {
    if u > O2_QUALIFY { "NOT QUALIFIED — every probe goes to the CPU f64 polish (spec §6.6)" }
    else if big_g + 2.0 * u < 3e-5 { "PASSES" }
    else if big_g - 2.0 * u >= 3e-5 { "MISSES" }
    else { "inside the ±2u_I band — the decisive probes go to the CPU f64 polish (spec §6.6)" }
}

/// The O₂ egg-box gate on the GPU, the spec §6.6 "Instrument" (Task 8a review, item 1): the gate's number comes from
/// energies, never from a GPU frozen-HF ΣF of a molecule (first order in an SCF residual the f32 stop cannot see;
/// `o2_gpu_protocol_report`). On the fixed grid Σ_I F_I(s) = −∇_s E(s) exactly and E is variational, so per probe
/// p ∈ {±h/4}³ and axis a
///   ΣF_a(p) = −(π/h)·[Ẽ(p + (h/4)ê_a) − Ẽ(p − (h/4)ê_a)],   Ẽ = E − E_wall,
/// exact for the egg-box fundamental and the cosine second harmonic. 36 cold GPU solves (coordinate a ∈ {0, ±h/2}, the
/// others ±h/4) plus E(0) and the six wall translates ±h·ê_b (`Wall`), at `GPU_REPORT_TOL`.
///
/// Which energy (the O₂ egg-box study, 2026-10-16; `o2_gate_run`): the GPU's own E carries f32 *evaluation* noise —
/// measured here as the scatter of E_GPU − E_f64 over the 43 states — and with it the estimator missed qualification by
/// 11× (u_I 5.5e-5, a sign flip in ΣF_y at (−h/4)(1,1,1)). The gate's number is therefore the estimator on E_f64, the
/// Kohn–Sham energy of each downloaded GPU state re-evaluated in f64 on the CPU (`ks_energy_f64`); the GPU-energy
/// estimator is printed beside it. Both are qualified against the CPU tol-1e-9 per-probe values (`o2_qualify`): u_I ≤ 5e-6,
/// then PASS iff G + 2u_I < 3e-5, MISS iff G − 2u_I ≥ 3e-5, otherwise the decisive probes go to the CPU f64 polish.
/// The study's remedy screens (`study_prims`) are printed after the verdict.
#[test]
#[ignore = "GPU report (≈ 10 min): cargo test -p validate --release --test pseudo -- --ignored --nocapture --exact o2_eggbox_gate_gpu"]
fn o2_eggbox_gate_gpu() {
    let Some(ctx) = common::gpu("pseudo") else { return };
    let h = 0.25f64.min(table(8).record.h_req.expect("O has h_req"));
    let s = setup(IonicPolicy::CoresOnly, h, &[8]);
    let (prims, variants) = study_prims(h);
    let run = o2_gate_run(&ctx, h, s, &prims, 14.0);
    let (pf, wall) = run.probes(&[]);
    let (u, big_g) = o2_qualify("f64 energies", h, &pf);
    let (pg, _) = run.probes_gpu();
    let (u_gpu, g_gpu) = o2_qualify("GPU energies", h, &pg);
    let wall_max = wall_max(&wall, h);
    run.print_summary("baseline", &variants);
    eprintln!("O₂ (CoresOnly) gate [f64 energies of the GPU states, tol {GPU_REPORT_TOL:.0e}] at h = {h}, 14 Bohr: G = max |ΣF − F_wall| {big_g:.2e} Ha/Bohr; u_I {u:.2e} ({}); verdict: {}; \
               [GPU energies: G {g_gpu:.2e}, u_I {u_gpu:.2e}: {}]; max |F_wall| at the probes {wall_max:.2e}",
              if u <= O2_QUALIFY { "qualified" } else { "not qualified" }, o2_verdict(big_g, u), o2_verdict(g_gpu, u_gpu));
    assert!(wall_max <= 3e-4, "the wall term {wall_max:.2e} exceeds 10× the gate: enlarge the box (spec §6.6)");
}


/// Report (M7 Task 8a; `--ignored`): how much of the GPU O₂ ΣF is the GPU protocol's tol (`GPU_REPORT_TOL`, against
/// the CPU's 1e-9)? At two of the gate's shifts, (−h/4)(1,1,1) and (+h/4)(1,1,1): the frozen ΣF of (a) the GPU state
/// (cold), (b) the GPU state of the same geometry reached warm from the other shift, and (c) that GPU state polished by
/// the CPU SCF to tol = eig_tol = 1e-9 (`qm::scf` warm from the downloaded state: same setup, same occupation scheme).
/// (c) is the CPU report's protocol; a − c is the GPU protocol's error at that site. Measured 2026-10-16 (1828 s, the CPU
/// polishes 1356 s and 415 s): (−h/4)(1,1,1) |a − c| 2.27e-5, |b − c| 6.66e-5; (+h/4)(1,1,1) |a − c| 8.75e-6,
/// |b − c| 2.44e-5. Cold, the error lies along the bond (x); warm, mostly across it (y, z up to 4.8e-5): the converged
/// f32 state's polarisation depends on the path. A lone atom shows none of it (the attribution agrees to ≤ 2 %). The
/// energies differ by 1.3–1.4e-6 (second order).
#[test]
#[ignore = "GPU + CPU report (≈ 20 min): cargo test -p validate --release --test pseudo -- --ignored --nocapture o2_gpu_protocol_report"]
fn o2_gpu_protocol_report() {
    let Some(ctx) = common::gpu("pseudo") else { return };
    let h = 0.25f64.min(table(8).record.h_req.expect("O has h_req"));
    let s = setup(IonicPolicy::CoresOnly, h, &[8]);
    let g = Grid::cube(n_for(14.0, h), h);
    let q = h / 4.0;
    let mut solver = GpuSolver::new(ctx, &g);
    for (p, other) in [([-q; 3], [q; 3]), ([q; 3], [-q; 3])] {
        let nuc = o2_at(p);
        let (_, st_a, stats_a) = gpu_solve_with(&mut solver, s.clone(), nuc.clone(), Some(3));
        let f_a = frozen_sum_force(&g, &st_a, &s, &nuc);
        // (b): the other shift cold, then this one warm from it
        let _ = gpu_solve_with(&mut solver, s.clone(), o2_at(other), Some(3));
        let sys = System::with_ionic(nuc.clone(), 0, Some(3), s.clone());
        let mut set = ScfSettings::for_grid(&g);
        (set.eig_tol, set.tol) = (1e-7, GPU_REPORT_TOL);
        set.occupations = Occupations::DegenerateShells { tau: DEGENERACY_TAU };
        let (_, stats_b) = solver.solve(&sys, &set, true).expect("GPU warm converges");
        let st_b = solver.ks_state(solver.state().unwrap());
        let f_b = frozen_sum_force(&g, &st_b, &s, &nuc);
        // (c): the CPU polish of (a) at the CPU report's tol
        let t = Instant::now();
        (set.tol, set.eig_tol) = (1e-9, 1e-9);
        let (st_c, stats_c) = scf(&g, &sys, &set, Some(&st_a)).expect("the CPU polish converges");
        let f_c = frozen_sum_force(&g, &st_c, &s, &nuc);
        eprintln!("O₂ shift ({:+.3}, {:+.3}, {:+.3}): ΣF (a) GPU cold ({:+.3e}, {:+.3e}, {:+.3e}) [{} steps, Δρ {:.1e}]; (b) GPU warm ({:+.3e}, {:+.3e}, {:+.3e}) [{} steps, Δρ {:.1e}]; \
                   (c) CPU polish to 1e-9 ({:+.3e}, {:+.3e}, {:+.3e}) [{} steps, {:.0} s]; |a − c| {:.2e}, |b − c| {:.2e}; E a {:.9} c {:.9}",
                  p[0], p[1], p[2], f_a[0], f_a[1], f_a[2], stats_a.iterations, stats_a.delta_rho, f_b[0], f_b[1], f_b[2], stats_b.iterations, stats_b.delta_rho,
                  f_c[0], f_c[1], f_c[2], stats_c.iterations, t.elapsed().as_secs_f64(),
                  norm([f_a[0] - f_c[0], f_a[1] - f_c[1], f_a[2] - f_c[2]]), norm([f_b[0] - f_c[0], f_b[1] - f_c[1], f_b[2] - f_c[2]]), stats_a.energy.total, stats_c.energy.total);
    }
}

/// Report (not a gate; `--ignored`; Task 6 round 2): is carbon's converged p-shell spread at the tetragonal site
/// (1.21e-4 in `degenerate_spread_at_low_symmetry_sites`' 8 Bohr box, over τ/10) the lattice or the box? The same solve
/// (C M = 3, h = 0.30, (0, h/2, h/2)) in 8, 12 and 16 Bohr boxes; a box term falls with the box, the lattice's does not.
#[test]
#[ignore = "report: cargo test -p validate --release --test pseudo -- --ignored --nocapture c_spread_box_report"]
fn c_spread_box_report() {
    let h = 0.30;
    let s = setup(IonicPolicy::CoresOnly, h, &[6]);
    for b in [8.0, 12.0, 16.0] {
        let (e, _, stats) = solve_with(h, b, s.clone(), vec![Nucleus { pos: [0.0, h / 2.0, h / 2.0], z: 6 }], Some(3), None);
        eprintln!("C M = 3 at (0, h/2, h/2), h = {h}, {b} Bohr: E {e:.9}; shared {:?}; released {}; face {:.1e}",
                  stats.fractional.iter().map(|c| (c.spin, c.count, c.spread)).collect::<Vec<_>>(), stats.released.len(), stats.boundary_density);
    }
}

// ---- The O₂ egg-box study (2026-10-16): the f64 energy instrument, the h dependence, the remedy screens ----
//
// Atomic units throughout (Hartree, Bohr, Ha/Bohr). The study measures; it changes no recipe constant and no solver.
//
// The instrument (`ks_energy_f64`). Spec §6.6's estimator differences energies 2·(h/4) apart; at h = 0.20 a 3e-5 Ha/Bohr
// force is a 1.9e-6 Ha energy difference, so the energies need σ_E ≪ 1e-7 Ha (noise (π√2/h)σ_E). The GPU's own E is
// assembled from f32 fields and carries ~1e-6 Ha of evaluation noise (measured: the E_GPU − E_f64 scatter printed by
// `print_summary`). Each GPU state is therefore downloaded and its Kohn–Sham energy re-evaluated in f64 on the CPU with
// the CPU `scf`'s final-energy expression (`qm::scf`, "energy from the output density and orbitals"):
//   E = Σ_iσ f_iσ⟨ψ_iσ|−½∇²_h|ψ_iσ⟩ + ∫ρV_loc + E_NL + ½∫ρV_H + E_xc[ρ_↑ + ½ρ_pc, ρ_↓ + ½ρ_pc] + E_ii,
// with the occupied orbitals (f > 0) Löwdin-orthonormalised per spin in f64 first (their f32 norms are off by ~1e-7,
// which would enter E at first order, ~ε_i·1e-7 per orbital), ρ_σ = Σ_i f_iσ|ψ_iσ|², V_H from the CPU Poisson solver
// (the same multipole boundary) at relative residual 1e-11, started from the GPU's V_H. E is variational in ψ at fixed
// occupations, so the f32 rounding of ψ (~6e-8), LOBPCG's f32 residual (~1e-5) and the SCF residual enter at second
// order. The estimator on E_f64 is qualified against the CPU tol-1e-9 per-probe ΣF exactly as spec §6.6 qualifies the
// GPU-energy one (`o2_qualify`).
//
// The remedy screens (`Prim`). A candidate remedy changes the discrete energy functional, E → E' = E + W. Evaluated on
// the baseline's self-consistent states, E'[ψ(s)] differs from E'[ψ'(s)] (the remedy's own SCF) by ½δψᵀHδψ with
// δψ = −H⁻¹∂W/∂ψ: second order in W. Its s-dependence — the error of the screened ΣF — is the lattice-periodic part of
// that second-order energy: small when W is small against the electronic energy scale. Screens are therefore a ranking,
// and the leading candidates are re-measured self-consistently (`o2_remedy_scf_gpu`). Every screen is a universal
// numerical rule (rule 1): nothing depends on Z.
// - DgSr/DgNl/DgPc: Ono & Hirose's double grid (PRL 82, 5016 (1999)) with refinement m and interpolation kernel K
//   (order 1: linear; 3: 4-point Lagrange, the 1-D weight of a node at ξ spacings). A coarse field φ_c is prolonged as
//   (Pφ)(r_f) = Σ_c K_c(r_f)φ_c, K_c(r) = Π_a K((r_a − r_c,a)/h), onto the fine lattice of spacing h/m.
//   · V_SR (linear in ρ): ∫ρV_SR → (h/m)³Σ_f (Pρ)(r_f)V_SR(r_f − R) = h³Σ_c ρ_c Ṽ_c, Ṽ_c = m⁻³Σ_f K_c(r_f)V_SR(r_f − R).
//   · β (Ono–Hirose's own case): ⟨β|ψ⟩ → (h/m)³Σ_f β(r_f − R)(Pψ)(r_f) = h³Σ_c β̃_c ψ_c, β̃_c = m⁻³Σ_f K_c(r_f)β(r_f − R);
//     V_NL keeps its separable form with β̃.
//   · ρ_pc (E_xc is nonlinear): ΔE_pc = E_xc[ρ_σ + ½ρ_pc] − E_xc[ρ_σ] → (h/m)³Σ_f [e_xc(Pρ_↑ + ½ρ_pc, Pρ_↓ + ½ρ_pc) −
//     e_xc(Pρ_↑, Pρ_↓)](r_f); the valence-only E_xc stays on the coarse grid (no seam: the bracket is 0 where ρ_pc = 0).
//   With m = 1 every image is the coarse term itself (K(integer) = δ): `study_prims` carries those as identity checks.
// - PcFrom: ρ_pc taken from the same species' table filtered for a coarser spacing h' (band edge π/h' < π/h): a softer
//   partial core on the same grid, every other table unchanged.

/// Neumaier-compensated sum (CLAUDE.md rule 4).
fn nsum(it: impl Iterator<Item = f64>) -> f64 {
    let (mut s, mut c) = (0.0f64, 0.0f64);
    for x in it {
        let t = s + x;
        c += if s.abs() >= x.abs() { (s - t) + x } else { (x - t) + s };
        s = t;
    }
    s + c
}

/// S^{−1/2} of a symmetric positive-definite k×k matrix near the identity: the coupled Newton–Schulz iteration (Higham,
/// *Functions of Matrices*, SIAM 2008, eq. 6.35): Y₀ = S, Z₀ = I, T = ½(3I − ZY), Y ← YT, Z ← TZ; Z → S^{−1/2}
/// quadratically for ‖I − S‖ < 1 (here ≲ 1e-6).
fn inv_sqrt_spd(s: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let k = s.len();
    let mul = |a: &[Vec<f64>], b: &[Vec<f64>]| -> Vec<Vec<f64>> { (0..k).map(|i| (0..k).map(|j| (0..k).map(|l| a[i][l] * b[l][j]).sum()).collect()).collect() };
    let eye: Vec<Vec<f64>> = (0..k).map(|i| (0..k).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
    let (mut y, mut z) = (s.to_vec(), eye.clone());
    for _ in 0..20 {
        let zy = mul(&z, &y);
        let t: Vec<Vec<f64>> = (0..k).map(|i| (0..k).map(|j| 1.5 * eye[i][j] - 0.5 * zy[i][j]).collect()).collect();
        y = mul(&y, &t);
        z = mul(&t, &z);
    }
    z
}

/// The occupied orbitals (f > 0) of a state, Löwdin-orthonormalised per spin in f64 (ψ'_a = Σ_b ψ_b (S^{−1/2})_ba), their
/// occupations and densities; `dev` = the largest |S_ab − δ_ab| before (the f32 orthonormality error).
struct Occupied { orbs: [Vec<Vec<f64>>; 2], f: [Vec<f64>; 2], rho: [Vec<f64>; 2], dev: f64 }

fn occupied(g: &Grid, st: &KsState) -> Occupied {
    let n = g.len();
    let mut out = Occupied { orbs: [Vec::new(), Vec::new()], f: [Vec::new(), Vec::new()], rho: [vec![0.0; n], vec![0.0; n]], dev: 0.0 };
    for (sp, (orbs, f)) in [(&st.orbitals_up, &st.occ_up), (&st.orbitals_dn, &st.occ_dn)].into_iter().enumerate() {
        let idx: Vec<usize> = (0..orbs.len()).filter(|&i| f[i] > 0.0).collect();
        let s: Vec<Vec<f64>> = idx.iter().map(|&i| idx.iter().map(|&j| g.dot(&orbs[i], &orbs[j])).collect()).collect();
        for (a, row) in s.iter().enumerate() { for (b, &x) in row.iter().enumerate() { out.dev = out.dev.max((x - if a == b { 1.0 } else { 0.0 }).abs()); } }
        let z = inv_sqrt_spd(&s);
        for a in 0..idx.len() {
            let mut v = vec![0.0; n];
            for (b, &j) in idx.iter().enumerate() { let c = z[b][a]; for (vp, op) in v.iter_mut().zip(&orbs[j]) { *vp += c * op; } }
            let fa = f[idx[a]];
            for (r, x) in out.rho[sp].iter_mut().zip(&v) { *r += fa * x * x; }
            out.orbs[sp].push(v);
            out.f[sp].push(fa);
        }
    }
    out
}

/// The f64 Kohn–Sham energy of a state (study notes, "The instrument"), Hartree.
#[derive(Clone, Copy, Debug)]
struct EnergyF64 { total: f64, kinetic: f64, local: f64, nl: f64, hartree: f64, xc: f64, e_ii: f64 }

fn xc_with_core(g: &Grid, rho: &[Vec<f64>; 2], pc: &[f64]) -> f64 {
    let u: Vec<f64> = rho[0].iter().zip(pc).map(|(a, c)| a + 0.5 * c).collect();
    let d: Vec<f64> = rho[1].iter().zip(pc).map(|(a, c)| a + 0.5 * c).collect();
    g.integrate(&qm::xc::lsda_pw92(&u, &d).energy_density)
}

fn ks_energy_f64(g: &Grid, occ: &Occupied, v_h_guess: &[f64], s: &IonicSetup, nuc: &[Nucleus], poisson: &mut qm::poisson::PoissonSolver) -> (EnergyF64, qm::poisson::PoissonStats) {
    let n = g.len();
    let rho: Vec<f64> = (0..n).map(|i| occ.rho[0][i] + occ.rho[1][i]).collect();
    let mut t = vec![0.0; n];
    let mut kin = Vec::new();
    for sp in 0..2 { for (psi, &f) in occ.orbs[sp].iter().zip(&occ.f[sp]) { qm::stencil::kinetic(g, psi, &mut t); kin.push(f * g.dot(psi, &t)); } }
    let kinetic = nsum(kin.into_iter());
    let local = g.dot(&rho, &ionic_local_potential(g, nuc, s));
    let op = NonlocalOp::build(g, nuc, s);
    let nl = op.energy_weighted(&occ.orbs[0], &occ.f[0]) + op.energy_weighted(&occ.orbs[1], &occ.f[1]);
    let mut v = v_h_guess.to_vec();
    let ps = poisson.solve(g, &rho, &mut v, 1e-11, 20000);
    let hartree = qm::poisson::hartree_energy(g, &rho, &v);
    let xc = xc_with_core(g, &occ.rho, &partial_core_field(g, nuc, s));
    let e_ii = ionic_repulsion(nuc, s).unwrap();
    (EnergyF64 { total: nsum([kinetic, local, nl, hartree, xc, e_ii].into_iter()), kinetic, local, nl, hartree, xc, e_ii }, ps)
}

/// The double grid's 1-D kernel (study notes): the weight of a node at ξ spacings; order 1 linear, 3 the 4-point Lagrange.
fn dg_kernel(order: u8, xi: f64) -> f64 {
    let a = xi.abs();
    match order {
        1 => (1.0 - a).max(0.0),
        3 => if a < 1.0 { (a + 1.0) * (a - 1.0) * (a - 2.0) / 2.0 } else if a < 2.0 { let t = a - 1.0; -t * (t - 1.0) * (t - 2.0) / 6.0 } else { 0.0 },
        _ => panic!("kernel order {order}"),
    }
}

/// A box of coarse nodes lo..=hi per axis (x fastest in every local array).
#[derive(Clone, Copy, Debug)]
struct CBox { lo: [i64; 3], hi: [i64; 3] }

impl CBox {
    /// The nodes within the ball (c, r) widened by the kernel's half-support (order 1: 1, order 3: 2 spacings), clipped
    /// to the grid (nodes beyond the Dirichlet faces do not exist).
    fn around(g: &Grid, c: [f64; 3], r: f64, order: u8) -> CBox {
        let s = if order == 1 { 1 } else { 2 };
        CBox { lo: [0, 1, 2].map(|a| (((c[a] - r - g.origin[a]) / g.h).floor() as i64 - s).max(0)),
               hi: [0, 1, 2].map(|a| (((c[a] + r - g.origin[a]) / g.h).ceil() as i64 + s).min(g.n[a] as i64 - 1)) }
    }
    fn union(self, o: CBox) -> CBox { CBox { lo: [0, 1, 2].map(|a| self.lo[a].min(o.lo[a])), hi: [0, 1, 2].map(|a| self.hi[a].max(o.hi[a])) } }
    fn dims(&self) -> [usize; 3] { [0, 1, 2].map(|a| (self.hi[a] - self.lo[a] + 1) as usize) }
    fn fine_dims(&self, m: usize) -> [usize; 3] { [0, 1, 2].map(|a| ((self.hi[a] - self.lo[a]) as usize) * m + 1) }
    /// The grid index of every node of the box, x fastest.
    fn nodes(&self, g: &Grid) -> Vec<usize> {
        let mut v = Vec::new();
        for k in self.lo[2]..=self.hi[2] { for j in self.lo[1]..=self.hi[1] { for i in self.lo[0]..=self.hi[0] { v.push(g.idx(i as usize, j as usize, k as usize)); } } }
        v
    }
    /// The position of fine node (i, j, k) (spacing h/m, fine node 0 = coarse node lo).
    fn fine_point(&self, g: &Grid, m: usize, ijk: [usize; 3]) -> [f64; 3] {
        [0, 1, 2].map(|a| g.origin[a] + (self.lo[a] as f64 + ijk[a] as f64 / m as f64) * g.h)
    }
}

/// The kernel matrix of one axis: rows the fine nodes lo·m..=hi·m, columns the coarse nodes lo..=hi.
fn kernel_matrix(lo: i64, hi: i64, m: usize, order: u8) -> Vec<Vec<f64>> {
    let mi = m as i64;
    (lo * mi..=hi * mi).map(|j| (lo..=hi).map(|c| dg_kernel(order, (j - c * mi) as f64 / m as f64)).collect()).collect()
}

fn transpose(w: &[Vec<f64>]) -> Vec<Vec<f64>> { (0..w[0].len()).map(|c| w.iter().map(|row| row[c]).collect()).collect() }

/// The mode-`axis` product: a's length dims[axis] (= mat's column count) becomes mat's row count. x fastest.
fn mode_product(a: &[f64], dims: [usize; 3], axis: usize, mat: &[Vec<f64>]) -> (Vec<f64>, [usize; 3]) {
    let mut od = dims;
    od[axis] = mat.len();
    let mut out = vec![0.0; od[0] * od[1] * od[2]];
    let (si, so) = ([1, dims[0], dims[0] * dims[1]], [1, od[0], od[0] * od[1]]);
    let ranges: Vec<(usize, usize)> = mat.iter().map(|row| (row.iter().position(|&x| x != 0.0).unwrap_or(0), row.iter().rposition(|&x| x != 0.0).map_or(0, |l| l + 1))).collect();
    let (b1, b2) = match axis { 0 => (1, 2), 1 => (0, 2), _ => (0, 1) };
    for i2 in 0..dims[b2] { for i1 in 0..dims[b1] {
        let (bi, bo) = (i1 * si[b1] + i2 * si[b2], i1 * so[b1] + i2 * so[b2]);
        for (r, row) in mat.iter().enumerate() {
            let (c0, c1) = ranges[r];
            let mut acc = 0.0;
            for c in c0..c1 { acc += row[c] * a[bi + c * si[axis]]; }
            out[bo + r * so[axis]] = acc;
        }
    }}
    (out, od)
}

/// f̃_c = m⁻³Σ_f K_c(r_f)f(r_f) at every node of the box (dense, x fastest; study notes).
fn dg_smooth(g: &Grid, b: &CBox, m: usize, order: u8, f: &dyn Fn([f64; 3]) -> f64) -> Vec<f64> {
    let fd = b.fine_dims(m);
    let mut a = Vec::with_capacity(fd[0] * fd[1] * fd[2]);
    for k in 0..fd[2] { for j in 0..fd[1] { for i in 0..fd[0] { a.push(f(b.fine_point(g, m, [i, j, k]))); } } }
    let mut dims = fd;
    for axis in 0..3 { (a, dims) = mode_product(&a, dims, axis, &transpose(&kernel_matrix(b.lo[axis], b.hi[axis], m, order))); }
    let scale = 1.0 / (m * m * m) as f64;
    a.iter().map(|x| x * scale).collect()
}

/// (Pφ)(r_f) at every fine node of the box (x fastest), from φ's values at the box's coarse nodes.
fn dg_prolong(g: &Grid, b: &CBox, m: usize, order: u8, field: &[f64]) -> (Vec<f64>, [usize; 3]) {
    let mut a: Vec<f64> = b.nodes(g).iter().map(|&p| field[p]).collect();
    let mut dims = b.dims();
    for axis in 0..3 { (a, dims) = mode_product(&a, dims, axis, &kernel_matrix(b.lo[axis], b.hi[axis], m, order)); }
    (a, dims)
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }

/// The pseudised ions of a geometry with their tables.
fn pseudo_ions<'a>(s: &'a IonicSetup, nuc: &[Nucleus]) -> Vec<(usize, &'a qm::radial::FilteredTable, [f64; 3])> {
    nuc.iter().enumerate().filter_map(|(i, n)| match s.ionic(n.z) { Ionic::Pseudo(t) => Some((i, &**t, n.pos)), _ => None }).collect()
}

/// h³Σ_p ρ_p V_SR(r_p − R) over the pseudised ions: coarse (m = 0) or its double-grid image.
fn sr_energy(g: &Grid, rho: &[f64], s: &IonicSetup, nuc: &[Nucleus], dg: Option<(usize, u8)>) -> f64 {
    let mut terms = Vec::new();
    for (_, t, pos) in pseudo_ions(s, nuc) {
        match dg {
            None => for p in qm::nuclear::points_within(g, pos, t.sr_radius()) { terms.push(rho[p] * t.v_sr(norm(sub3(g.point(p), pos)))); },
            Some((m, order)) => {
                let b = CBox::around(g, pos, t.sr_radius(), order);
                let v = dg_smooth(g, &b, m, order, &|r| t.v_sr(norm(sub3(r, pos))));
                for (p, x) in b.nodes(g).into_iter().zip(v) { terms.push(rho[p] * x); }
            }
        }
    }
    nsum(terms.into_iter()) * g.dv()
}

/// V_NL with every β_lm replaced by its double-grid image β̃ (study notes); the same spheres' structure as
/// `NonlocalOp::build`, the point list = the box nodes where some β̃_lm ≠ 0.
fn nl_op_dg(g: &Grid, nuc: &[Nucleus], s: &IonicSetup, m: usize, order: u8) -> NonlocalOp {
    use qm::poisson::{lm_index, solid_harmonics, N_MOMENTS};
    let mut spheres = Vec::new();
    for (ion, t, pos) in pseudo_ions(s, nuc) {
        for (k, p) in t.projectors.iter().enumerate() {
            let b = CBox::around(g, pos, p.radius, order);
            let nrm = ((2 * p.l + 1) as f64 / (4.0 * std::f64::consts::PI)).sqrt();
            let li = p.l as i32;
            let comps: Vec<Vec<f64>> = (-li..=li).map(|mm| dg_smooth(g, &b, m, order, &|r| {
                let d = sub3(r, pos);
                let mut sh = [0.0; N_MOMENTS];
                solid_harmonics(d, &mut sh);
                t.beta_radial(k, norm(d)) * nrm * sh[lm_index(p.l as usize, mm)]
            })).collect();
            let nodes = b.nodes(g);
            let keep: Vec<usize> = (0..nodes.len()).filter(|&i| comps.iter().any(|c| c[i] != 0.0)).collect();
            spheres.push(qm::nonlocal::Sphere { ion, z: nuc[ion].z, l: p.l, points: keep.iter().map(|&i| nodes[i]).collect(),
                                                beta: comps.iter().map(|c| keep.iter().map(|&i| c[i]).collect()).collect(), e_kb: p.e_kb });
        }
    }
    NonlocalOp { spheres, dv: g.dv() }
}

/// ΔE_pc = E_xc[ρ_σ + ½ρ_pc] − E_xc[ρ_σ]: on the coarse grid (dg None), or on the fine lattice with ρ_σ prolonged.
fn pc_energy(g: &Grid, occ: &Occupied, s: &IonicSetup, nuc: &[Nucleus], dg: Option<(usize, u8)>) -> f64 {
    let cores: Vec<(usize, &qm::radial::FilteredTable, [f64; 3])> = pseudo_ions(s, nuc).into_iter().filter(|c| c.1.pc_radius > 0.0).collect();
    if cores.is_empty() { return 0.0; }
    let ex = |u: f64, d: f64, pc: f64| qm::xc::exc_point(u + 0.5 * pc, d + 0.5 * pc).0 - qm::xc::exc_point(u, d).0;
    match dg {
        None => {
            let pc = partial_core_field(g, nuc, s);
            nsum((0..g.len()).filter(|&p| pc[p] != 0.0).map(|p| ex(occ.rho[0][p], occ.rho[1][p], pc[p]))) * g.dv()
        }
        Some((m, order)) => {
            let b = cores.iter().map(|(_, t, pos)| CBox::around(g, *pos, t.pc_radius, order)).reduce(CBox::union).unwrap();
            let (u, dims) = dg_prolong(g, &b, m, order, &occ.rho[0]);
            let (d, _) = dg_prolong(g, &b, m, order, &occ.rho[1]);
            let mut terms = Vec::new();
            for k in 0..dims[2] { for j in 0..dims[1] { for i in 0..dims[0] {
                let r = b.fine_point(g, m, [i, j, k]);
                let pc: f64 = cores.iter().map(|(_, t, pos)| t.rho_pc_at(norm(sub3(r, *pos)))).sum();
                if pc == 0.0 { continue; }
                let q = i + dims[0] * (j + dims[1] * k);
                terms.push(ex(u[q], d[q], pc));
            }}}
            nsum(terms.into_iter()) * (g.h / m as f64).powi(3)
        }
    }
}

/// A primitive screen: δ = E_variant − E at a frozen state (study notes, "The remedy screens").
#[derive(Clone)]
enum Prim { DgSr(usize, u8), DgNl(usize, u8), DgPc(usize, u8), PcFrom(String, Arc<IonicSetup>) }

impl Prim {
    fn name(&self) -> String {
        match self {
            Prim::DgSr(m, o) => format!("DG V_SR m{m} k{o}"), Prim::DgNl(m, o) => format!("DG β m{m} k{o}"),
            Prim::DgPc(m, o) => format!("DG ρ_pc-XC m{m} k{o}"), Prim::PcFrom(l, _) => format!("ρ_pc {l}"),
        }
    }
    fn delta(&self, g: &Grid, occ: &Occupied, s: &IonicSetup, nuc: &[Nucleus]) -> f64 {
        let rho = || -> Vec<f64> { occ.rho[0].iter().zip(&occ.rho[1]).map(|(a, b)| a + b).collect() };
        match self {
            Prim::DgSr(m, o) => { let r = rho(); sr_energy(g, &r, s, nuc, Some((*m, *o))) - sr_energy(g, &r, s, nuc, None) }
            Prim::DgNl(m, o) => {
                let (a, b) = (nl_op_dg(g, nuc, s, *m, *o), NonlocalOp::build(g, nuc, s));
                let e = |op: &NonlocalOp| op.energy_weighted(&occ.orbs[0], &occ.f[0]) + op.energy_weighted(&occ.orbs[1], &occ.f[1]);
                e(&a) - e(&b)
            }
            Prim::DgPc(m, o) => pc_energy(g, occ, s, nuc, Some((*m, *o))) - pc_energy(g, occ, s, nuc, None),
            Prim::PcFrom(_, other) => pc_energy(g, occ, other, nuc, None) - pc_energy(g, occ, s, nuc, None),
        }
    }
}

/// O's tables filtered at h, cached per process (`filtered` re-runs the radial T8 at that h: seconds).
fn filtered_o(h: f64) -> Arc<qm::radial::FilteredTable> {
    static C: OnceLock<Mutex<HashMap<u64, Arc<qm::radial::FilteredTable>>>> = OnceLock::new();
    if let Some(t) = C.get_or_init(Default::default).lock().unwrap().get(&h.to_bits()) { return t.clone(); }
    let t = Arc::new(filtered(&table(8), h));
    C.get_or_init(Default::default).lock().unwrap().entry(h.to_bits()).or_insert(t).clone()
}

/// The O setup at h: certified by `IonicSetup::build` when it passes, else the same table through `from_tables` with the
/// refusal printed (a measurement at an uncertified h is labelled, never silently admitted).
fn study_setup(h: f64) -> Arc<IonicSetup> {
    let t = filtered_o(h);
    match IonicSetup::build(IonicPolicy::CoresOnly, h, h, &[8], &mut |z| has_core(z), &mut |_| Ok(t.clone())) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            eprintln!("  NOTE: O at h = {h} is not certified ({e}); measured through from_tables");
            Arc::new(IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, BTreeMap::from([(8, Ionic::Pseudo(t))])))
        }
    }
}

/// O at h with ρ_pc taken from the table filtered at h' (`Prim::PcFrom`): ρ_pc's log-grid values and support from h', its
/// uniform table resampled at h's spacing Δ = h/256 from h''s exact band-limited interpolant (4-point Lagrange error
/// ≤ 5.3e-10 of max ρ_pc, spec §5.1 "Sampling"); every other table h's own. The T8 row is h's (stale for ρ_pc: the
/// radial check of the variant is `pc_band_radial_report`).
fn pc_band_table(h: f64, h_pc: f64) -> qm::radial::FilteredTable {
    let mut ft = (*filtered_o(h)).clone();
    let fp = filtered_o(h_pc);
    ft.rho_pc = fp.rho_pc.clone();
    ft.pc_radius = fp.pc_radius;
    ft.rho_pc_u = (0..(fp.pc_radius / ft.du).ceil() as usize + 4).map(|k| fp.rho_pc_at(k as f64 * ft.du)).collect();
    ft
}

fn pc_band_setup(h: f64, h_pc: f64) -> Arc<IonicSetup> {
    Arc::new(IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, BTreeMap::from([(8, Ionic::Pseudo(Arc::new(pc_band_table(h, h_pc))))])))
}

/// The study's screens at h and the variants printed from them (name, prim indices). Indices 0–2 are the m = 1 identity
/// checks (δ must vanish to rounding). Env `O2_SCREENS`: "full" (default at h = 0.20), "light" (the scans' default: the
/// m = 3 cubic images and ρ_pc at h/0.75) or "none".
fn study_prims(h: f64) -> (Vec<Prim>, Vec<(String, Vec<usize>)>) {
    let level = std::env::var("O2_SCREENS").unwrap_or_else(|_| if (h - 0.20).abs() < 1e-12 { "full".into() } else { "light".into() });
    let mut prims = vec![Prim::DgSr(1, 3), Prim::DgNl(1, 3), Prim::DgPc(1, 3)];
    if level == "none" { return (prims, Vec::new()); }
    prims.extend([Prim::DgSr(3, 3), Prim::DgNl(3, 3), Prim::DgPc(3, 3), Prim::PcFrom(format!("filtered at h/0.75 = {:.4}", h / 0.75), pc_band_setup(h, h / 0.75))]);
    if level == "full" {
        prims.extend([Prim::DgNl(3, 1), Prim::DgPc(2, 3), Prim::PcFrom(format!("filtered at h/0.5 = {:.4}", h / 0.5), pc_band_setup(h, h / 0.5))]);
    }
    let mut variants: Vec<(String, Vec<usize>)> = (3..prims.len()).map(|i| (prims[i].name(), vec![i])).collect();
    variants.push(("DG all three m3 k3".into(), vec![3, 4, 5]));
    variants.push(("DG V_SR + β m3 k3".into(), vec![3, 4]));
    (prims, variants)
}

/// The box edge of the gate runs (Bohr): env `O2_BOX`, else spec §6.6's 14.
fn box_bohr() -> f64 { std::env::var("O2_BOX").map(|v| v.parse().expect("O2_BOX: Bohr")).unwrap_or(14.0) }

/// The 43 shifts of spec §6.6's energy estimator in units of q = h/4: E(0), the translates ±4ê_b, and the 36 points
/// (coordinate a ∈ {0, ±2}, the others ±1).
fn gate_keys() -> Vec<[i32; 3]> {
    let mut keys: Vec<[i32; 3]> = vec![[0, 0, 0]];
    for b in 0..3 { for sgn in [4, -4] { let mut k = [0; 3]; k[b] = sgn; keys.push(k); } }
    for a in 0..3 { for ca in [-2, 0, 2] { for s1 in [-1, 1] { for s2 in [-1, 1] {
        let mut k = [0; 3];
        k[a] = ca;
        let others: Vec<usize> = (0..3).filter(|&b| b != a).collect();
        (k[others[0]], k[others[1]]) = (s1, s2);
        if !keys.contains(&k) { keys.push(k); }
    }}}}
    keys
}

/// Spec §6.6's estimator from energies at `gate_keys` (Ẽ = E − E_wall, the wall fitted from the translates):
/// ΣF_a(p) = −(π/h)[Ẽ(p + ê_a) − Ẽ(p − ê_a)] at the eight probes p ∈ {±1}³ (units of h/4).
fn estimator(h: f64, kappa: f64, e: &dyn Fn([i32; 3]) -> f64) -> (Vec<([i32; 3], [f64; 3])>, Wall) {
    let q = h / 4.0;
    let e0 = e([0, 0, 0]);
    let wall = Wall::fit(h, kappa, e0, [e([4, 0, 0]), e([0, 4, 0]), e([0, 0, 4])], [e([-4, 0, 0]), e([0, -4, 0]), e([0, 0, -4])]);
    let et = |k: [i32; 3]| e(k) - wall.energy(k.map(|c| c as f64 * q));
    let mut out = Vec::new();
    for sx in [-1, 1] { for sy in [-1, 1] { for sz in [-1, 1] {
        let p = [sx, sy, sz];
        out.push((p, [0, 1, 2].map(|a| { let (mut up, mut dn) = (p, p); up[a] += 1; dn[a] -= 1; -(std::f64::consts::PI / h) * (et(up) - et(dn)) })));
    }}}
    (out, wall)
}

fn wall_max(w: &Wall, h: f64) -> f64 {
    let q = h / 4.0;
    let mut m = 0.0f64;
    for x in [-q, q] { for y in [-q, q] { for z in [-q, q] { m = m.max(norm(w.force([x, y, z]))); } } }
    m
}

fn gmax(p: &[([i32; 3], [f64; 3])]) -> f64 { p.iter().map(|(_, f)| norm(*f)).fold(0.0, f64::max) }

/// One gate state: the GPU's E, the f64 energy, the screens' δ, ε_max, SCF steps, Δρ, the f32 orthonormality error and
/// the f64 Poisson solve (iterations, residual).
struct GateState { e_gpu: f64, e: EnergyF64, deltas: Vec<f64>, eps_max: f64, steps: usize, delta_rho: f64, dev: f64, poisson: (usize, f64) }

struct GateRun { h: f64, n: usize, states: HashMap<[i32; 3], GateState>, prims: Vec<Prim>, secs: f64, vram: f64 }

/// The O₂ gate's 43 cold GPU solves at h (`o2_at`, `CoresOnly` triplet, `GPU_REPORT_TOL`), each state downloaded and
/// re-evaluated in f64 (`ks_energy_f64`) with every screen's δ.
fn o2_gate_run(ctx: &gpu::GpuContext, h: f64, s: Arc<IonicSetup>, prims: &[Prim], box_bohr: f64) -> GateRun {
    gate_run(ctx, h, s, prims, box_bohr, &o2_at, Some(3))
}

/// `o2_gate_run` for any molecule: `at(shift)` its nuclei rigidly shifted, `mult` its multiplicity.
fn gate_run(ctx: &gpu::GpuContext, h: f64, s: Arc<IonicSetup>, prims: &[Prim], box_bohr: f64, at: &dyn Fn([f64; 3]) -> Vec<Nucleus>, mult: Option<u32>) -> GateRun {
    let g = Grid::cube(n_for(box_bohr, h), h);
    let mut solver = GpuSolver::new(ctx.clone(), &g);
    let mut poisson = qm::poisson::PoissonSolver::new(&g);
    let q = h / 4.0;
    let t0 = Instant::now();
    let mut states = HashMap::new();
    for k in gate_keys() {
        let nuc = at(k.map(|c| c as f64 * q));
        let (e_gpu, st, stats) = gpu_solve_with(&mut solver, s.clone(), nuc.clone(), mult);
        let occ = occupied(&g, &st);
        let (e, ps) = ks_energy_f64(&g, &occ, &st.v_h, &s, &nuc, &mut poisson);
        let deltas = prims.iter().map(|p| p.delta(&g, &occ, &s, &nuc)).collect();
        states.insert(k, GateState { e_gpu, e, deltas, eps_max: highest_occupied(&st), steps: stats.iterations, delta_rho: stats.delta_rho, dev: occ.dev, poisson: (ps.iterations, ps.residual) });
    }
    GateRun { h, n: g.n[0], states, prims: prims.to_vec(), secs: t0.elapsed().as_secs_f64(), vram: vram_mb(ctx) }
}

impl GateRun {
    fn kappa(&self) -> f64 { (-2.0 * self.states[&[0, 0, 0]].eps_max).sqrt() }
    /// The estimator on E_f64 plus the selected screens' δ.
    fn probes(&self, sel: &[usize]) -> (Vec<([i32; 3], [f64; 3])>, Wall) {
        estimator(self.h, self.kappa(), &|k| { let st = &self.states[&k]; st.e.total + sel.iter().map(|&i| st.deltas[i]).sum::<f64>() })
    }
    fn probes_gpu(&self) -> (Vec<([i32; 3], [f64; 3])>, Wall) { estimator(self.h, self.kappa(), &|k| self.states[&k].e_gpu) }

    /// The run's lines: the instrument (GPU-energy noise, orthonormality, Poisson), the base probes, every variant's G with
    /// its per-axis maxima and the δ spread. Returns the base G.
    fn print_summary(&self, label: &str, variants: &[(String, Vec<usize>)]) -> f64 {
        let h = self.h;
        let d: Vec<f64> = self.states.values().map(|s| s.e_gpu - s.e.total).collect();
        let mean = d.iter().sum::<f64>() / d.len() as f64;
        let sd = (d.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (d.len() - 1) as f64).sqrt();
        let spread = |v: &[f64]| v.iter().cloned().fold(f64::MIN, f64::max) - v.iter().cloned().fold(f64::MAX, f64::min);
        let steps: Vec<usize> = self.states.values().map(|s| s.steps).collect();
        eprintln!("O₂ [{label}] h = {h}, {}³: instrument: E_GPU − E_f64 mean {mean:+.3e}, sd {sd:.2e}, spread {:.2e} Ha (→ GPU-energy estimator noise (π√2/h)·sd = {:.1e} Ha/Bohr); \
                   f32 orthonormality ≤ {:.1e}; f64 Poisson ≤ {} iterations, residual ≤ {:.1e}; SCF steps {}–{}, Δρ ≤ {:.1e}; E(0) f64 {:.9} (T {:.6}, loc {:.6}, NL {:.6}, H {:.6}, xc {:.6}, ii {:.6}); {:.0} s, VRAM {:.0} MB",
                  self.n, spread(&d), std::f64::consts::PI * 2f64.sqrt() / h * sd,
                  self.states.values().map(|s| s.dev).fold(0.0, f64::max), self.states.values().map(|s| s.poisson.0).max().unwrap(),
                  self.states.values().map(|s| s.poisson.1).fold(0.0, f64::max), steps.iter().min().unwrap(), steps.iter().max().unwrap(),
                  self.states.values().map(|s| s.delta_rho).fold(0.0, f64::max),
                  { let e = &self.states[&[0, 0, 0]].e; e.total }, self.states[&[0, 0, 0]].e.kinetic, self.states[&[0, 0, 0]].e.local, self.states[&[0, 0, 0]].e.nl,
                  self.states[&[0, 0, 0]].e.hartree, self.states[&[0, 0, 0]].e.xc, self.states[&[0, 0, 0]].e.e_ii, self.secs, self.vram);
        for i in 0..3.min(self.prims.len()) {
            let worst = self.states.values().map(|s| s.deltas[i].abs()).fold(0.0, f64::max);
            eprintln!("  O₂ [{label}] identity check {}: max |δ| over the states {worst:.1e} Ha", self.prims[i].name());
        }
        let (base, wall) = self.probes(&[]);
        let line = |name: &str, p: &[([i32; 3], [f64; 3])]| {
            let ax = [0, 1, 2].map(|a| p.iter().map(|(_, f)| f[a].abs()).fold(0.0, f64::max));
            format!("G {:.2e} (max |ΣF_x|, |ΣF_y|, |ΣF_z| = {:.2e}, {:.2e}, {:.2e}) → {} [{name}]", gmax(p), ax[0], ax[1], ax[2], if gmax(p) < 3e-5 { "under 3e-5" } else { "over 3e-5" })
        };
        eprintln!("O₂ [{label}] h = {h}: baseline (f64 energies) {}; wall max {:.2e}", line("baseline", &base), wall_max(&wall, h));
        for (p, f) in &base { eprintln!("    probe {p:?}: ΣF ({:+.3e}, {:+.3e}, {:+.3e})", f[0], f[1], f[2]); }
        for (name, sel) in variants {
            let (p, _) = self.probes(sel);
            let dmean: Vec<f64> = sel.iter().map(|&i| { let v: Vec<f64> = self.states.values().map(|s| s.deltas[i]).collect(); v.iter().sum::<f64>() / v.len() as f64 }).collect();
            eprintln!("  O₂ [{label}] h = {h}: screen {} (mean δ {})", line(name, &p), sci(&dmean));
        }
        gmax(&base)
    }
}

/// Report (the O₂ egg-box study; `--ignored`): the CPU tol-1e-9 per-probe ΣF the estimator is qualified against (spec
/// §6.6). For each probe (env `O2_POLISH` = "all" or a list like "-1-1+1,+1-1-1"; default the six not on record): the
/// cold GPU state (a) and (c) = (a) polished by the CPU SCF to tol = eig_tol = 1e-9 (`o2_gpu_protocol_report`'s protocol);
/// printed: the frozen ΣF of (c) (raw: the number recorded in `O2_CPU_PROBES`), of (a), E_GPU(a), E_f64(a), the CPU's
/// E(c) and E_f64(c) (the instrument's self-check: E_f64(c) must equal the CPU's own E(c) to its Poisson tolerance, and
/// E_f64(a) − E(c) is the instrument's error at that geometry). One probe at a time; 415–1356 s of CPU each.
#[test]
#[ignore = "GPU + CPU report (≈ 2 h): cargo test -p validate --release --test pseudo -- --ignored --nocapture --exact o2_cpu_probe_polish_report"]
fn o2_cpu_probe_polish_report() {
    let Some(ctx) = common::gpu("pseudo") else { return };
    let h = 0.25f64.min(table(8).record.h_req.expect("O has h_req"));
    let s = setup(IonicPolicy::CoresOnly, h, &[8]);
    let g = Grid::cube(n_for(14.0, h), h);
    let q = h / 4.0;
    let mut all: Vec<[i32; 3]> = Vec::new();
    for x in [-1, 1] { for y in [-1, 1] { for z in [-1, 1] { all.push([x, y, z]); } } }
    let parse = |t: &str| -> Vec<[i32; 3]> {
        t.split(',').map(|p| { let v: Vec<i32> = p.trim().as_bytes().chunks(2).map(|c| if c[0] == b'-' { -1 } else { 1 }).collect(); [v[0], v[1], v[2]] }).collect()
    };
    let which: Vec<[i32; 3]> = match std::env::var("O2_POLISH").ok().as_deref() {
        Some("all") => all.clone(),
        Some(t) => parse(t),
        None => all.iter().copied().filter(|p| !O2_CPU_PROBES.iter().any(|(k, _)| k == p)).collect(),
    };
    let mut solver = GpuSolver::new(ctx, &g);
    let mut poisson = qm::poisson::PoissonSolver::new(&g);
    for p in which {
        let t0 = Instant::now();
        let nuc = o2_at(p.map(|c| c as f64 * q));
        let (e_gpu, st_a, stats_a) = gpu_solve_with(&mut solver, s.clone(), nuc.clone(), Some(3));
        let (ef_a, _) = ks_energy_f64(&g, &occupied(&g, &st_a), &st_a.v_h, &s, &nuc, &mut poisson);
        let f_a = frozen_sum_force(&g, &st_a, &s, &nuc);
        let sys = System::with_ionic(nuc.clone(), 0, Some(3), s.clone());
        let mut set = ScfSettings::for_grid(&g);
        (set.tol, set.eig_tol) = (1e-9, 1e-9);
        set.occupations = Occupations::DegenerateShells { tau: DEGENERACY_TAU };
        let (st_c, stats_c) = scf(&g, &sys, &set, Some(&st_a)).expect("the CPU polish converges");
        let f_c = frozen_sum_force(&g, &st_c, &s, &nuc);
        let (ef_c, _) = ks_energy_f64(&g, &occupied(&g, &st_c), &st_c.v_h, &s, &nuc, &mut poisson);
        eprintln!("O₂ polish probe {p:?}: (c) CPU 1e-9 raw ΣF ({:+.4e}, {:+.4e}, {:+.4e}) [{} steps]; (a) GPU cold ΣF ({:+.3e}, {:+.3e}, {:+.3e}) [{} steps, Δρ {:.1e}]; \
                   E: GPU(a) {e_gpu:.10}, f64(a) {:.10}, CPU(c) {:.10}, f64(c) {:.10}; f64(a) − CPU(c) {:+.2e}, f64(c) − CPU(c) {:+.2e}, GPU(a) − CPU(c) {:+.2e}; {:.0} s",
                  f_c[0], f_c[1], f_c[2], stats_c.iterations, f_a[0], f_a[1], f_a[2], stats_a.iterations, stats_a.delta_rho,
                  ef_a.total, stats_c.energy.total, ef_c.total, ef_a.total - stats_c.energy.total, ef_c.total - stats_c.energy.total, e_gpu - stats_c.energy.total,
                  t0.elapsed().as_secs_f64());
    }
}

/// The h list of the scans: env `O2_H` (comma-separated, Bohr), else the default.
fn h_list(default: &[f64]) -> Vec<f64> {
    std::env::var("O2_H").ok().map(|v| v.split(',').map(|x| x.trim().parse().expect("O2_H: comma-separated spacings")).collect()).unwrap_or_else(|| default.to_vec())
}

/// Report (the O₂ egg-box study; `--ignored`): the energy-based O₂ gate (`o2_gate_run`, f64 energies; the instrument
/// qualified at h = 0.20 by `o2_eggbox_gate_gpu`) at the spacings of `O2_H` (default 0.175, 0.15, 0.125) in the 14 Bohr
/// box, with every screen of `study_prims`. Separates grid resolution from recipe hardness (h_req,O = 0.20 by the
/// energy criterion; the app's default Accurate is 0.15).
#[test]
#[ignore = "GPU report (≈ 20–60 min per h): O2_H=0.175,0.15 cargo test -p validate --release --test pseudo -- --ignored --nocapture --exact o2_eggbox_h_scan_gpu"]
fn o2_eggbox_h_scan_gpu() {
    let Some(ctx) = common::gpu("pseudo") else { return };
    let mut rows = Vec::new();
    for h in h_list(&[0.175, 0.15, 0.125]) {
        let s = study_setup(h);
        let (prims, variants) = study_prims(h);
        let run = o2_gate_run(&ctx, h, s, &prims, box_bohr());
        let g = run.print_summary(&format!("h scan, {} Bohr", box_bohr()), &variants);
        rows.push((h, g));
        eprintln!("O₂ h scan so far (h, G): {rows:.4?}");
    }
}

/// The f64-energy instrument carried to another h or molecule (spec §6.6 "Instrument"; O₂ egg-box study ruling,
/// 2026-10-16, Ruling 1): u_I = (π√2/h)σ_E + (4π/h)|b₂| + (8π/h)|a₃|.
/// - Writing the wall-subtracted energy along axis a as Ẽ(s_a) = Σ_n a_n cos(2πns_a/h) + b_n sin(2πns_a/h), the
///   estimator at p_a = ±h/4 (from Ẽ(0) and Ẽ(±h/2)) is exact for a₁, b₁, a₂, b₃; it misses b₂ ((4π/h)|b₂|) and returns
///   +(2π/h)a₃ for the true −(6π/h)a₃ ((8π/h)|a₃|).
/// - The decisive probe p* is the one with the largest |ΣF − F_wall|, a* its largest component. The scan takes Ẽ at
///   s_a* = k·h/8, k = −3..=4, the other coordinates at p*'s: k = 0 and k = 4 are gate states (a* ∈ {0, +h/2} in the 36),
///   the six others cold solves (the ruling counted 4 extra; the probe lines s_a* = ±h/4 are not among the 36, so 6). The
///   8-point DFT gives a₀…a₄, b₁…b₃ exactly; the gate's s_a* = −h/2 state against k = 4 is printed as the check of the
///   wall model and of periodicity.
/// - σ_E: at the gate state p* + (h/4)ê_a*, the f64 energy of a warm solve started from a cold state 0.02 Bohr away along
///   a*, minus the cold gate state's.
struct Transfer { u: f64, sigma_e: f64, b2: f64, a3: f64, probe: [i32; 3], axis: usize, coeffs: Vec<(f64, f64)>, periodicity: f64 }

fn transfer_check(ctx: &gpu::GpuContext, run: &GateRun, s: Arc<IonicSetup>, box_bohr: f64, at: &dyn Fn([f64; 3]) -> Vec<Nucleus>, mult: Option<u32>) -> Transfer {
    let h = run.h;
    let q = h / 4.0;
    let (probes, wall) = run.probes(&[]);
    let (probe, f) = probes.iter().copied().max_by(|a, b| norm(a.1).total_cmp(&norm(b.1))).expect("eight probes");
    let axis = (0..3).max_by(|&a, &b| f[a].abs().total_cmp(&f[b].abs())).unwrap();
    let g = Grid::cube(n_for(box_bohr, h), h);
    let mut solver = GpuSolver::new(ctx.clone(), &g);
    let mut poisson = qm::poisson::PoissonSolver::new(&g);
    let mut e_f64 = |solver: &mut GpuSolver, shift: [f64; 3], warm: bool| -> f64 {
        let nuc = at(shift);
        let (_, st, _) = gpu_solve_from(solver, s.clone(), nuc.clone(), mult, warm);
        ks_energy_f64(&g, &occupied(&g, &st), &st.v_h, &s, &nuc, &mut poisson).0.total
    };
    let base = probe.map(|c| c as f64 * q);
    let along = |sa: f64| { let mut p = base; p[axis] = sa; p };
    let gate_e = |ka: i32| { let mut k = probe; k[axis] = ka; run.states[&k].e.total };
    let mut et = Vec::new();
    for k in -3i32..=4 {
        let p = along(k as f64 * h / 8.0);
        let e = match k { 0 => gate_e(0), 4 => gate_e(2), _ => e_f64(&mut solver, p, false) };
        et.push(e - wall.energy(p));
    }
    let periodicity = (gate_e(-2) - wall.energy(along(-h / 2.0))) - et[7];
    let coeffs: Vec<(f64, f64)> = (0..=4).map(|n| {
        let (c, sn) = (-3i32..=4).zip(&et).fold((0.0, 0.0), |(c, sn), (k, e)| {
            let x = 2.0 * std::f64::consts::PI * (n * k) as f64 / 8.0;
            (c + e * x.cos(), sn + e * x.sin())
        });
        let w = if n == 0 || n == 4 { 1.0 / 8.0 } else { 2.0 / 8.0 };
        (w * c, w * sn)
    }).collect();
    // σ_E at the gate state p* + (h/4)ê_a*: cold (the gate's) against warm from 0.02 Bohr away
    let mut ke = probe;
    ke[axis] += 1;
    let pe = ke.map(|c| c as f64 * q);
    let mut near = pe;
    near[axis] += 0.02;
    e_f64(&mut solver, near, false);
    let sigma_e = (e_f64(&mut solver, pe, true) - run.states[&ke].e.total).abs();
    let (b2, a3) = (coeffs[2].1, coeffs[3].0);
    let pi = std::f64::consts::PI;
    let u = pi * 2f64.sqrt() / h * sigma_e + 4.0 * pi / h * b2.abs() + 8.0 * pi / h * a3.abs();
    Transfer { u, sigma_e, b2, a3, probe, axis, coeffs, periodicity }
}

/// The CoresOnly H₂O of the probe: the `h2o` preset geometry (O at the origin, H at (±1.43, 1.11, 0): O–H 1.81 Bohr,
/// 104.5°) offset by (0, 0.07, 0.03) like O₂, rigidly shifted by `s`.
fn h2o_at(s: [f64; 3]) -> Vec<Nucleus> {
    let o = [0.0 + s[0], 0.07 + s[1], 0.03 + s[2]];
    vec![Nucleus { pos: o, z: 8 }, Nucleus { pos: [o[0] + 1.43, o[1] + 1.11, o[2]], z: 1 }, Nucleus { pos: [o[0] - 1.43, o[1] + 1.11, o[2]], z: 1 }]
}

/// O (pseudised) and H (all-electron, σ = h) under `CoresOnly` at h: certified by `IonicSetup::build` when O's table passes
/// at h, else through `from_tables` with the refusal printed (`study_setup`'s rule).
fn h2o_setup(h: f64) -> Arc<IonicSetup> {
    let t = filtered_o(h);
    match IonicSetup::build(IonicPolicy::CoresOnly, h, h, &[8, 1], &mut |z| has_core(z), &mut |_| Ok(t.clone())) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            eprintln!("  NOTE: O at h = {h} is not certified ({e}); measured through from_tables");
            Arc::new(IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, BTreeMap::from([(8, Ionic::Pseudo(t)), (1, Ionic::AllElectron { sigma: h })])))
        }
    }
}

/// The §6.6 records at the current recipe version (`m7-tm-kb-nlcc-4`, c = ¼; Task 7c round 2, 2026-10-16, RX 7900 XTX,
/// `eggbox_probe_with_transfer_gpu`): (molecule, h, G_rec, u_I,rec, PASS achieved). `CoresOnly`, 14 Bohr. A version that
/// lowers G re-records G_rec (spec §6.6 "Gate and fallback").
const EGGBOX_RECORD: &[(&str, f64, f64, f64, bool)] = &[
    ("o2", 0.20, 4.262e-5, 1.67e-7, false), ("o2", 0.175, 9.168e-6, 9.33e-8, true), ("o2", 0.15, 6.746e-6, 1.12e-7, true),
    ("h2o", 0.20, 3.033e-5, 1.53e-8, false), ("h2o", 0.15, 4.139e-6, 6.32e-7, true),
];

/// Spec §6.6 with the f64-energy instrument and its transfer check (`transfer_check`), gate and fallback (NLCC ruling,
/// 2026-10-16; §7 risk 1), for the gated molecules with a partial core under `CoresOnly`: O₂ (`o2_at`, triplet) and H₂O
/// (`h2o_at`, singlet; two smeared H). `--ignored` (GPU ≈ 5–9 min per molecule and h). Asserted per (molecule, h):
/// - the instrument: u_I ≤ 5e-6 and the wall maximum ≤ 3e-4;
/// - where `EGGBOX_RECORD` has PASS: G + 2u_I < 3e-5 (the hard gate);
/// - where it has MISS: non-regression, G ≤ G_rec + 2(u_I + u_I,rec), each u_I ≥ 1.15e-7; MISS is printed with the achieved
///   factor against the all-electron figure of record, 3.4e-4/G (vector) and 3.4e-4/max|ΣF_y| (component) — raw,
///   σ = h = 0.25, 12 Bohr: indicative, not like-for-like.
/// Env `EGG_MOLS` (default "o2,h2o"), `O2_H` (default: the recorded spacings of each molecule), `O2_BOX` (default 14).
#[test]
#[ignore = "GPU (≈ 40 min): cargo test -p validate --release --test pseudo -- --ignored --nocapture --exact eggbox_probe_with_transfer_gpu"]
fn eggbox_probe_with_transfer_gpu() {
    let Some(ctx) = common::gpu("pseudo") else { return };
    let mols = std::env::var("EGG_MOLS").unwrap_or_else(|_| "o2,h2o".into());
    for mol in mols.split(',').map(str::trim) {
        let recorded: Vec<f64> = EGGBOX_RECORD.iter().filter(|r| r.0 == mol).map(|r| r.1).collect();
        for h in h_list(&recorded) {
            let (at, mult, s): (&dyn Fn([f64; 3]) -> Vec<Nucleus>, Option<u32>, Arc<IonicSetup>) = match mol {
                "o2" => (&o2_at, Some(3), study_setup(h)),
                "h2o" => (&h2o_at, Some(1), h2o_setup(h)),
                other => panic!("EGG_MOLS: unknown molecule {other}"),
            };
            let t0 = Instant::now();
            let run = gate_run(&ctx, h, s.clone(), &[], box_bohr(), at, mult);
            let big_g = run.print_summary(&format!("{mol} CoresOnly, {} Bohr", box_bohr()), &[]);
            let tr = transfer_check(&ctx, &run, s, box_bohr(), at, mult);
            let pi = std::f64::consts::PI;
            eprintln!("{mol} (CoresOnly) at h = {h}, {} Bohr: G = {big_g:.3e} Ha/Bohr; transfer: decisive probe {:?} axis {}; σ_E {:.2e} Ha → {:.2e}, b₂ {:+.2e} Ha → {:.2e}, a₃ {:+.2e} Ha → {:.2e}; u_I {:.2e} Ha/Bohr ({}); \
                       §6.6 verdict: {}; scan (a_n, b_n), n = 0…4: [{}]; periodicity Ẽ(−h/2) − Ẽ(h/2) {:+.2e} Ha; {:.0} s",
                      box_bohr(), tr.probe, tr.axis, tr.sigma_e, pi * 2f64.sqrt() / h * tr.sigma_e, tr.b2, 4.0 * pi / h * tr.b2.abs(), tr.a3, 8.0 * pi / h * tr.a3.abs(), tr.u,
                      if tr.u <= O2_QUALIFY { "transfers" } else { "does not transfer" }, o2_verdict(big_g, tr.u), tr.coeffs.iter().map(|(a, b)| format!("({a:+.3e}, {b:+.3e})")).collect::<Vec<_>>().join(", "), tr.periodicity, t0.elapsed().as_secs_f64());
            let (probes, wall) = run.probes(&[]);
            let (wmax, fy) = (wall_max(&wall, h), probes.iter().map(|(_, f)| f[1].abs()).fold(0.0f64, f64::max));
            assert!(tr.u <= O2_QUALIFY && wmax <= 3e-4, "{mol} at h = {h}: the instrument fails (u_I {:.2e}, wall {wmax:.2e})", tr.u);
            match EGGBOX_RECORD.iter().find(|r| r.0 == mol && (r.1 - h).abs() < 1e-12) {
                Some(&(_, _, _, _, true)) => assert!(big_g + 2.0 * tr.u < 3e-5, "{mol} at h = {h}: G {big_g:.3e} + 2u_I misses the 3e-5 gate it passed at record"),
                Some(&(_, _, g_rec, u_rec, false)) => {
                    let bound = g_rec + 2.0 * (tr.u.max(1.15e-7) + u_rec.max(1.15e-7));
                    eprintln!("{mol} at h = {h}: MISS (recorded, spec §6.6 / §7 risk 1); achieved factor against the all-electron 3.4e-4: {:.1}× (vector), {:.1}× (max |ΣF_y| {fy:.2e}); ratchet G ≤ {bound:.3e}",
                              3.4e-4 / big_g, 3.4e-4 / fy);
                    assert!(big_g <= bound, "{mol} at h = {h}: G {big_g:.3e} regressed past the record {g_rec:.3e} (ratchet {bound:.3e})");
                }
                None => eprintln!("{mol} at h = {h}: no record at this spacing (report only)"),
            }
        }
    }
}

/// Report (the O₂ egg-box study; `--ignored`): a remedy measured self-consistently on the GPU (the screens' check), the
/// energy-based O₂ gate with every GPU solve under the variant's tables. Env `O2_REMEDY`: "pc:<h'>" (ρ_pc filtered at
/// h', `pc_band_setup`); without it the baseline tables (used for a code-level variant compiled in, e.g. a double-grid
/// patch of `NonlocalOp::build`). Spacings `O2_H` (default 0.20).
#[test]
#[ignore = "GPU report: O2_REMEDY=pc:0.2667 O2_H=0.20 cargo test -p validate --release --test pseudo -- --ignored --nocapture --exact o2_remedy_scf_gpu"]
fn o2_remedy_scf_gpu() {
    let Some(ctx) = common::gpu("pseudo") else { return };
    let remedy = std::env::var("O2_REMEDY").unwrap_or_default();
    for h in h_list(&[0.20]) {
        let s = match remedy.strip_prefix("pc:") {
            Some(hp) => pc_band_setup(h, hp.parse::<f64>().expect("pc:<h'>")),
            None => study_setup(h),
        };
        let (prims, variants) = study_prims(h);
        let run = o2_gate_run(&ctx, h, s, &prims, box_bohr());
        run.print_summary(&format!("SCF remedy '{remedy}', {} Bohr, env M7_STUDY_* {:?}", box_bohr(), std::env::vars().filter(|(k, _)| k.starts_with("M7_STUDY")).collect::<Vec<_>>()), &variants);
    }
}

/// F̂(q) = 4π∫f(r)j₀(qr)r²dr of a log-grid function.
fn radial_ft(grid: &qm::radial::LogGrid, f: &[f64], q: f64) -> f64 {
    4.0 * std::f64::consts::PI * grid.integrate(&grid.r.iter().zip(f).map(|(&r, &x)| x * qm::radial::sph_bessel(0, q * r) * r * r).collect::<Vec<_>>())
}

/// The partial core as the grid sees it at h: ρ_pc^filt(0), the unfiltered ρ_pc(0), the table radius, F̂ at q = 0, q_c and
/// π/h (unfiltered beside it), and the band fraction ∫_{q_c}^{π/h}|F̂|²q²dq / ∫_0^{π/h}|F̂|²q²dq of the filtered ρ_pc.
fn pc_diagnostics(ft: &qm::radial::FilteredTable) -> String {
    let (h, un) = (ft.h, &table(8).partial_core.rho_pc);
    let (qc, qm_) = (0.7 * std::f64::consts::PI / h, std::f64::consts::PI / h);
    let n = 400;
    let (mut lo, mut band) = (0.0, 0.0);
    for k in 0..n {
        let q = (k as f64 + 0.5) * qm_ / n as f64;
        let w = radial_ft(&ft.grid, &ft.rho_pc, q).powi(2) * q * q;
        if q < qc { lo += w } else { band += w }
    }
    format!("ρ_pc^filt(0) {:.4} (unfiltered {:.2}), table radius {:.3}; F̂(0) {:.4}, F̂(q_c = {qc:.2}) {:+.3e} (unfiltered {:+.3e}), F̂(π/h) {:+.3e} (unfiltered {:+.3e}); band fraction {:.2e}",
            ft.rho_pc_at(0.0), un[0], ft.pc_radius, radial_ft(&ft.grid, &ft.rho_pc, 0.0), radial_ft(&ft.grid, &ft.rho_pc, qc), radial_ft(&table(8).grid, un, qc),
            radial_ft(&ft.grid, &ft.rho_pc, qm_), radial_ft(&table(8).grid, un, qm_), band / (lo + band))
}

/// Report (the O₂ egg-box study; `--ignored`): the lone-O per-term attribution (`o_floor_attribution_gpu`'s protocol:
/// GPU tol 1e-6, f64 frozen terms; qualified to ≤ 2 %) at the spacings of `O2_H` (default 0.20, 0.175, 0.15, 0.125, 0.10),
/// with ρ_pc as the grid sees it (`pc_diagnostics`) and the T8 row at that h; and, with `O2_PC_FROM` = h' set, the same
/// with ρ_pc filtered at h' (`pc_band_setup`). The h = 0.15 puzzle: is F_pc's growth the partial core's band (q_c ∝ 1/h
/// admits more of the unresolved 1s tail) or its support?
#[test]
#[ignore = "GPU report: cargo test -p validate --release --test pseudo -- --ignored --nocapture --exact o_floor_attribution_h_scan_gpu"]
fn o_floor_attribution_h_scan_gpu() {
    let Some(ctx) = common::gpu("pseudo") else { return };
    let pc_from: Option<f64> = std::env::var("O2_PC_FROM").ok().map(|v| v.parse().expect("O2_PC_FROM: h'"));
    for h in h_list(&[0.20, 0.175, 0.15, 0.125, 0.10]) {
        let (s, ft) = match pc_from {
            Some(hp) => { let t = pc_band_table(h, hp); (pc_band_setup(h, hp), Arc::new(t)) }
            None => (study_setup(h), filtered_o(h)),
        };
        let g = Grid::cube(n_for(16.0, h), h);
        let nuc = vec![Nucleus { pos: [h / 4.0; 3], z: 8 }];
        let mut solver = GpuSolver::new(ctx.clone(), &g);
        let t0 = Instant::now();
        let (e, st, stats) = gpu_solve_with(&mut solver, s.clone(), nuc.clone(), Some(3));
        let line = attribution_line(&g, &st, &s, &nuc);
        let c = &ft.check;
        eprintln!("pseudo-O triplet at (h/4)³, h = {h}{}, 16 Bohr ({}³), GPU tol {GPU_REPORT_TOL:.0e}: {line}; E {e:.9}, {} steps, Δρ {:.1e}, shared {:?}; {}; T8 row: ΔE {:+.3} mHa, Δε {:.3} mHa, T3 {:.3} mHa, clamp {}, F_table [{}]; {:.0} s, VRAM {:.0} MB",
                  pc_from.map_or(String::new(), |hp| format!(" (ρ_pc filtered at {hp})")), g.n[0], stats.iterations, stats.delta_rho,
                  stats.fractional.iter().map(|x| (x.spin, x.count, x.spread)).collect::<Vec<_>>(), pc_diagnostics(&ft),
                  c.de * 1e3, c.max_eig * 1e3, c.t3_max * 1e3, c.clamped, sci(&c.f_table), t0.elapsed().as_secs_f64(), vram_mb(&ctx));
    }
}

/// Report (the O₂ egg-box study; `--ignored`; CPU, seconds): the radial cost of a softer partial core (`pc_band_table`):
/// at h, ρ_pc filtered at h' = h/κ, the filtered radial KB atom against the unfiltered one (T8(a): ΔE and the occupied
/// eigenvalues, 1 mHa) and every gated T3 configuration against the all-electron ΔE (T8(b), 5 mHa), beside the
/// unmodified table's row; plus T8(c)'s clamp count. κ ∈ {0.75, 0.5}, h ∈ `O2_H` (default 0.20, 0.15).
#[test]
#[ignore = "CPU report: cargo test -p validate --release --test pseudo -- --ignored --nocapture --exact pc_band_radial_report"]
fn pc_band_radial_report() {
    use qm::radial::kb::{reference_config, solve_kb_atom};
    let t = table(8);
    let un = solve_kb_atom(&t.grid, &t.kb.input(), &reference_config(&t.partition), SpinMode::Restricted).unwrap();
    let hund_ae = ground_state(8, SpinMode::Polarised).unwrap();
    let cfgs: Vec<(qm::radial::TransferConfig, f64)> = transfer_configs(&t, &hund_ae).into_iter()
        .filter_map(|c| t.gates.t3.iter().find(|r| r.name == c.name).map(|r| (c, r.de_ae))).collect();
    let check = |ft: &qm::radial::FilteredTable| -> (f64, f64, f64, usize) {
        let input = ft.input();
        let fa = solve_kb_atom(&ft.grid, &input, &reference_config(&t.partition), SpinMode::Restricted).unwrap();
        let max_eig = t.partition.channels.iter().filter(|c| c.occupied).map(|c| (fa.eps[c.l as usize][0] - un.eps[c.l as usize][0]).abs()).fold(0.0, f64::max);
        let mut clamped = qm::radial::filter::xc_domain_violations(&fa.rho, &ft.rho_pc);
        let mut t3 = 0.0f64;
        for (cfg, de_ae) in &cfgs {
            let pp = solve_kb_atom(&ft.grid, &input, &cfg.pp, cfg.spin).unwrap();
            clamped += qm::radial::filter::xc_domain_violations(&pp.rho, &ft.rho_pc);
            t3 = t3.max(((pp.energy - fa.energy) - de_ae).abs());
        }
        (fa.energy - un.energy, max_eig, t3, clamped)
    };
    for h in h_list(&[0.20, 0.15]) {
        let base = filtered_o(h);
        let (de, me, t3, cl) = check(&base);
        eprintln!("O at h = {h}: as built: ΔE {:+.3} mHa, Δε {:.3} mHa, T3 {:.3} mHa, clamp {cl} (record row: ΔE {:+.3}, Δε {:.3}, T3 {:.3})",
                  de * 1e3, me * 1e3, t3 * 1e3, base.check.de * 1e3, base.check.max_eig * 1e3, base.check.t3_max * 1e3);
        for kappa in [0.75, 0.5] {
            let ft = pc_band_table(h, h / kappa);
            let (de, me, t3, cl) = check(&ft);
            eprintln!("O at h = {h}, ρ_pc filtered at h/{kappa} = {:.4}: ΔE {:+.3} mHa, Δε {:.3} mHa, T3 {:.3} mHa, clamp {cl} → T8(a–c) {}; {}",
                      h / kappa, de * 1e3, me * 1e3, t3 * 1e3, if de.abs() < 1e-3 && me < 1e-3 && t3 < 5e-3 && cl == 0 { "pass" } else { "FAIL" }, pc_diagnostics(&ft));
        }
    }
}

/// Report (the O₂ egg-box study; `--ignored`; CPU, ≈ 1 min): O's generation record as this build generates it — f, the
/// local channel, r_c, r_nlc, ρ_pc's support, T3, T5, h_req and the T8 ladder — for comparing a recipe variant (e.g. a
/// larger r_nlc, the study's patched build with `M7_STUDY_NLCC_CROSSOVER`) with the recipe as built.
#[test]
#[ignore = "CPU report: cargo test -p validate --release --test pseudo -- --ignored --nocapture --exact o_record_report"]
fn o_record_report() {
    let t = table(8);
    let r = &t.record;
    eprintln!("O record (env M7_STUDY_* {:?}): f {}, local {}, r_c {:?}, r_nlc {:?}, r_cmp {:.3}, r_core {:.3}, E_KB {:?}; T1 {:.2e}, T3 {:?}; T5 {:?}; h_req {:?}",
              std::env::vars().filter(|(k, _)| k.starts_with("M7_STUDY")).collect::<Vec<_>>(), r.f, r.local, r.r_c, r.r_nlc, r.r_cmp, r.r_core, r.e_kb,
              r.gates.t1, r.gates.t3.iter().map(|x| (x.name.clone(), (x.de_pp - x.de_ae) * 1e3)).collect::<Vec<_>>(), r.gates.t5, r.h_req);
    for row in &r.filter {
        eprintln!("  T8 h = {}: ΔE {:+.3} mHa, Δε {:.3} mHa, T3 {:.3} mHa, clamp {}, F_table [{}] → {}", row.h, row.de * 1e3, row.max_eig * 1e3, row.t3_max * 1e3, row.clamped,
                  sci(&row.f_table), if row.passes() { "pass" } else { "fail" });
    }
    for h in [0.20, 0.15] { eprintln!("  at h = {h}: {}", pc_diagnostics(&filtered_o(h))); }
}
