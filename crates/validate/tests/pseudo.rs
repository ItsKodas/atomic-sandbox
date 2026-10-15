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
use qm::ionic::{Ionic, IonicError, IonicPolicy, IonicSetup};
use qm::nonlocal::NonlocalOp;
use qm::nuclear::{centres, has_partial_core, ionic_local_potential, ionic_repulsion, partial_core_field, smeared_potential};
use qm::radial::{filter::filtered, generate, has_core, PseudoTable};
use qm::radial::{ground_state, transfer_configs, SpinMode};
use qm::{scf, Grid, KsState, Nucleus, Occupations, ScfSettings, System, DEGENERACY_TAU};
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
/// box, sites (0, h/2, h/2), (h/2, 0, 0), (h/2, h/2, h/2), (h/4, h/4, h/4); cases O M = 1, 3 at
/// 0.20, C M = 3 and N M = 2 at 0.30, F M = 2 at 0.15. Reported: the step-1 spread (a one-step solve at tol = ∞),
/// which round 1's one-shot rule let decide. The box must not confine (no box-size warning is asserted): the review's 8 Bohr
/// box gave C (diffuse 2p, h = 0.30) a face density of 1.4e-4 and a spread of 1.21e-4 at (0, h/2, h/2), against 3.6e-5 at
/// 12 Bohr and 4.2e-5 at 16 Bohr (`c_spread_box_report`): the box's anisotropy, not the lattice's (Task 6 round 2).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo")]
fn degenerate_spread_at_low_symmetry_sites() {
    let cases: [(u8, u32, f64); 5] = [(8, 1, 0.20), (8, 3, 0.20), (6, 3, 0.30), (7, 2, 0.30), (9, 2, 0.15)];
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
/// share → Ono–Hirose.
#[test]
#[ignore = "slow CPU report (hours), run with --ignored; rerun on the GPU after M7 Task 8: cargo test -p validate --release --test pseudo -- --ignored --nocapture o_floor_attribution_report"]
fn o_floor_attribution_report() {
    for h in [0.20, 0.15] {
        let s = setup(IonicPolicy::CoresOnly, h, &[8]);
        let g = Grid::cube(n_for(16.0, h), h);
        let nuc = vec![Nucleus { pos: [h / 4.0; 3], z: 8 }];
        let t0 = Instant::now();
        let (e, st, stats) = solve_with(h, 16.0, s.clone(), nuc.clone(), Some(3), Some(1e-9));
        let rho = st.rho();
        let lr = |n: &[Nucleus]| g.dot(&rho, &smeared_potential(&g, &centres(n, &s)));
        let sr = |n: &[Nucleus]| { let v = ionic_local_potential(&g, n, &s); let l = smeared_potential(&g, &centres(n, &s));
                                    g.dot(&rho, &v.iter().zip(&l).map(|(a, b)| a - b).collect::<Vec<_>>()) };
        let nl = |n: &[Nucleus]| { let op = NonlocalOp::build(&g, n, &s); op.energy_weighted(&st.orbitals_up, &st.occ_up) + op.energy_weighted(&st.orbitals_dn, &st.occ_dn) };
        let pc = |n: &[Nucleus]| { let c = partial_core_field(&g, n, &s);
            let u: Vec<f64> = st.rho_up.iter().zip(&c).map(|(a, x)| a + 0.5 * x).collect();
            let d: Vec<f64> = st.rho_dn.iter().zip(&c).map(|(a, x)| a + 0.5 * x).collect();
            g.integrate(&qm::xc::lsda_pw92(&u, &d).energy_density) };
        let terms: [(&str, &dyn Fn(&[Nucleus]) -> f64); 4] = [("LR", &lr), ("SR", &sr), ("NL", &nl), ("pc", &pc)];
        let mut total = [0.0; 3];
        let mut line = String::new();
        for (name, e) in terms {
            let f = term_force(&nuc, 0, e);
            for a in 0..3 { total[a] += f[a]; }
            line += &format!("F_{name} ({:+.3e}, {:+.3e}, {:+.3e}) |{:.2e}|; ", f[0], f[1], f[2], norm(f));
        }
        let (whole, _) = frozen_force(&g, &st, &s, &nuc, 0);
        eprintln!("pseudo-O triplet at (h/4)³, h = {h}, 16 Bohr ({}³): {line}sum ({:+.3e}, {:+.3e}, {:+.3e}) |{:.2e}| vs frozen_force |{:.2e}|; E {e:.9}, {} steps, shared {:?}, released {}, face {:.1e}; {:.0} s",
                  g.n[0], total[0], total[1], total[2], norm(total), norm(whole), stats.iterations, stats.fractional.iter().map(|c| (c.spin, c.count, c.spread)).collect::<Vec<_>>(),
                  stats.released.len(), stats.boundary_density, t0.elapsed().as_secs_f64());
    }
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

/// Report, run early (Task 6 re-review ruling 2: Task 7's precondition has fired, so the O₂ gate is run and recorded;
/// `--ignored`). Task 7 owns the asserting gate (`o2_max_shift_sum_of_forces_below_3e_5`, with `qm::forces`); this is
/// the same probe with the frozen Hellmann–Feynman ΣF. O₂ (`CoresOnly`, triplet: π*↑ full, the §5.7 rule inert), the
/// plan's geometry (R = 2.28, offset (0.07, 0.03)), h = h_req,O = 0.20, 14 Bohr, tol = eig_tol = 1e-9:
/// - max over the 8 rigid shifts {±h/4}³ of |ΣF − F_wall|, the wall fitted from the 6 lattice translates ±h·ê_b (the
///   molecule is off the box centre: no inversion shortcut), κ = √(−2ε_max) of the unshifted state;
/// - the raw and wall maxima, ΣF_y per shift (the like-for-like component), the {0, h/2}³ ripple wall-subtracted;
/// - the verdict against 3e-5 (printed; the gate is not moved — review ruling 2 predicts ≈ 4e-5).
#[test]
#[ignore = "slow CPU report (hours), run with --ignored; rerun on the GPU after M7 Task 8: cargo test -p validate --release --test pseudo -- --ignored --nocapture o2_eggbox_gate_early_run"]
fn o2_eggbox_gate_early_run() {
    let h = 0.25f64.min(table(8).record.h_req.expect("O has h_req"));
    let o2 = |s: [f64; 3]| vec![Nucleus { pos: [-1.14 + s[0], 0.07 + s[1], 0.03 + s[2]], z: 8 }, Nucleus { pos: [1.14 + s[0], 0.07 + s[1], 0.03 + s[2]], z: 8 }];
    let s = setup(IonicPolicy::CoresOnly, h, &[8]);
    let g = Grid::cube(n_for(14.0, h), h);
    let (q, half) = (h / 4.0, h / 2.0);
    let probes: Vec<[f64; 3]> = [-q, q].iter().flat_map(|&x| [-q, q].map(move |y| [x, y])).flat_map(|[x, y]| [-q, q].map(move |zz| [x, y, zz])).collect();
    let ripple: Vec<[f64; 3]> = [0.0, half].iter().flat_map(|&x| [0.0, half].map(move |y| [x, y])).flat_map(|[x, y]| [0.0, half].map(move |zz| [x, y, zz])).collect();
    let translates = [[h, 0.0, 0.0], [0.0, h, 0.0], [0.0, 0.0, h], [-h, 0.0, 0.0], [0.0, -h, 0.0], [0.0, 0.0, -h]];
    let t0 = Instant::now();
    let forces: Vec<([f64; 3], usize, usize)> = par_map(&probes, |&p| {
        let nuc = o2(p);
        let (_, st, stats) = solve_with(h, 14.0, s.clone(), nuc.clone(), Some(3), Some(1e-9));
        (frozen_sum_force(&g, &st, &s, &nuc), stats.iterations, stats.fractional.len() + stats.released.len())
    });
    let shifts: Vec<[f64; 3]> = ripple.iter().chain(&translates).copied().collect();
    let energies: Vec<(f64, f64)> = par_map(&shifts, |&p| { let (e, st, _) = solve_with(h, 14.0, s.clone(), o2(p), Some(3), Some(1e-9)); (e, highest_occupied(&st)) });
    let e0 = energies[0].0;
    let kappa = (-2.0 * energies[0].1).sqrt();
    let e_plus = [energies[8].0, energies[9].0, energies[10].0];
    let e_minus = [energies[11].0, energies[12].0, energies[13].0];
    let wall = Wall::fit(h, kappa, e0, e_plus, e_minus);
    let (mut raw, mut wall_max, mut corrected) = (0.0f64, 0.0f64, 0.0f64);
    let mut fy = Vec::new();
    for (p, &(f, steps, shared)) in probes.iter().zip(&forces) {
        let fw = wall.force(*p);
        let fc = [f[0] - fw[0], f[1] - fw[1], f[2] - fw[2]];
        eprintln!("  O₂ shift ({:+.3}, {:+.3}, {:+.3}): ΣF ({:+.3e}, {:+.3e}, {:+.3e}), wall ({:+.3e}, {:+.3e}, {:+.3e}), corrected ({:+.3e}, {:+.3e}, {:+.3e}) |{:.3e}|; {steps} steps; shared/released clusters {shared}",
                  p[0], p[1], p[2], f[0], f[1], f[2], fw[0], fw[1], fw[2], fc[0], fc[1], fc[2], norm(fc));
        raw = raw.max(norm(f));
        wall_max = wall_max.max(norm(fw));
        corrected = corrected.max(norm(fc));
        fy.push(fc[1]);
    }
    let es: Vec<f64> = energies[..8].iter().map(|e| e.0).collect();
    let corr_e: Vec<f64> = es.iter().zip(&ripple).map(|(e, p)| e - wall.energy(*p)).collect();
    let spread = |v: &[f64]| v.iter().cloned().fold(f64::MIN, f64::max) - v.iter().cloned().fold(f64::MAX, f64::min);
    eprintln!("O₂ (CoresOnly) at h = {h}, 14 Bohr: max |ΣF| raw {raw:.2e}, wall {wall_max:.2e}, wall-subtracted {corrected:.2e} Ha/Bohr → {} the 3e-5 gate; ΣF_y (corrected) [{}]; \
               κ {kappa:.4}; E(±h·ê_b) − E(0) = [{}] / [{}]; ripple {{0, h/2}}³ raw {:.2e}, wall-subtracted {:.2e} Ha; {:.0} s",
              if corrected < 3e-5 { "PASSES" } else { "MISSES" }, sci(&fy), sci(&e_plus.map(|e| e - e0)), sci(&e_minus.map(|e| e - e0)), spread(&es), spread(&corr_e), t0.elapsed().as_secs_f64());
    assert!(wall_max <= 3e-4, "the wall term {wall_max:.2e} exceeds 10× the gate: enlarge the box (spec §6.6)");
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
