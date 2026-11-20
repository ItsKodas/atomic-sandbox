//! Stray atoms (spec 2026-11-18, plan Task 2; `stray` module doc): the split into a bonded part and
//! stray atoms, the signature, the static floor, the composition check's verdict and the cleared
//! region. Pure logic: no teacher, no provider.
use super::super::stray::*;
use super::*;
use sim::{Frontier, PartialLevel};

const R_CUT: f64 = 6.0;

/// H₂ at 1.4 Bohr along x about the origin plus one H per entry of `strays`.
fn h2_plus(strays: &[[f64; 3]]) -> Scene {
    let mut sc = h2(1.4);
    sc.nuclei.extend(strays.iter().map(|&p| Nucleus::new(1, p)));
    sc
}

fn positions(sc: &Scene) -> Vec<[f64; 3]> {
    sc.nuclei.iter().map(|n| n.pos).collect()
}

/// H (m = 2, the only multiplicity it has) and all-electron O on Accurate (its gapped M = 1 winner, which shows
/// partial levels at 5 mHa: `lone` module doc); anything else unknown.
fn m_ref(z: u8) -> Option<(u32, Option<bool>)> {
    match z {
        1 => Some((2, None)),
        8 => Some((1, Some(true))),
        _ => None,
    }
}

fn floor(sc: &Scene, faces: f64, fence: bool) -> Option<String> {
    let split = split(&positions(sc), R_CUT).expect("a bonded part and a stray atom");
    floor_refusal(sc, &split, &m_ref, &|_, _| fence, &|_| faces)
}

fn prov() -> Provenance {
    Provenance {
        grid_n: 49,
        h: 0.25,
        sigma: 0.25,
        tol: 1e-6,
        charge: 0,
        multiplicity: None,
        teacher: "cpu-ks".into(),
        masses: vec![(1, 1)],
        ionic: "none/fd5e-3/pin1e-5".into(),
    }
}

#[test]
fn split_finds_bonded_and_stray() {
    let sc = h2_plus(&[[0.0, 8.0, 0.0]]);
    let s = split(&positions(&sc), R_CUT).expect("H₂ + a far H");
    assert_eq!(s, Split { bonded: vec![0, 1], stray: vec![2] });
    // every atom alone (rule 0b's), every atom bonded, one atom: no split
    assert_eq!(split(&[[0.0; 3], [10.0, 0.0, 0.0]], R_CUT), None);
    assert_eq!(split(&positions(&h2(1.4)), R_CUT), None);
    assert_eq!(split(&[[0.0; 3]], R_CUT), None);
    // strictly inside r_cut is a neighbour (f_c > 0), at r_cut is not
    assert_eq!(split(&[[0.0; 3], [1.4, 0.0, 0.0], [1.4 + R_CUT, 0.0, 0.0]], R_CUT), Some(Split { bonded: vec![0, 1], stray: vec![2] }));
    // the signature: sorted Z lists, the charge, the reference settings of the surface
    let mut w = h2(1.4);
    w.nuclei = vec![Nucleus::new(8, [0.0; 3]), Nucleus::new(1, [0.0, 9.0, 0.0]), Nucleus::new(1, [1.8, 0.0, 0.0]), Nucleus::new(1, [-0.45, 1.75, 0.0])];
    let ws = split(&positions(&w), R_CUT).unwrap();
    assert_eq!(ws, Split { bonded: vec![0, 2, 3], stray: vec![1] });
    let z: Vec<u8> = w.nuclei.iter().map(|n| n.z).collect();
    let sig = Signature::new(&z, &ws, 0, &prov());
    assert_eq!((sig.bonded_z.clone(), sig.stray_z.clone(), sig.charge), (vec![1, 1, 8], vec![1], 0));
    // charge and multiplicity are not the surface's (the floor judges them); every reference setting is
    let mut other = prov();
    other.charge = 1;
    other.multiplicity = Some(2);
    other.masses = vec![(1, 2)];
    assert_eq!(Signature::new(&z, &ws, 0, &other), sig);
    other.h = 0.15;
    assert_ne!(Signature::new(&z, &ws, 0, &other).surface, sig.surface);
    let mut other = prov();
    other.ionic = "accepted/abc".into();
    assert_ne!(Signature::new(&z, &ws, 0, &other).surface, sig.surface);
}

#[test]
fn floor_refuses_charge_forced_m_and_spin() {
    let far = [0.0, 8.0, 0.0];
    let mut sc = h2_plus(&[far]);
    assert_eq!(floor(&sc, 100.0, false), None, "H₂ + H at automatic: M = 2 = 1 + (2 − 1)");
    sc.charge = 1;
    let why = floor(&sc, 100.0, false).expect("a charged scene");
    assert!(why.contains("charge"), "{why}");
    sc.charge = 0;
    sc.multiplicity = Some(2);
    let why = floor(&sc, 100.0, false).expect("a forced multiplicity, even the aligned one");
    assert!(why.contains("forced"), "{why}");
    // two stray H at automatic: 4 electrons → M = 1, the aligned product needs 3
    let two = h2_plus(&[far, [0.0, -8.0, 0.0]]);
    let why = floor(&two, 100.0, false).expect("H₂ + 2 H at automatic");
    assert!(why.contains("spin"), "{why}");
    // a stray species whose reference multiplicity is not known here
    let mut li = h2_plus(&[far]);
    li.nuclei[2].z = 3;
    let why = floor(&li, 100.0, false).expect("Li's reference spin is unknown in this fixture");
    assert!(why.contains("spin") && why.contains("Li"), "{why}");
}

#[test]
fn floor_refuses_mixed_stray_species() {
    // H₂O bonded, a stray H and a stray O: 19 electrons → M = 2 = 1 + (2 − 1) + (1 − 1), so the spin rule passes and
    // the mixed-species rule decides
    let mut sc = h2(1.4);
    sc.nuclei = vec![
        Nucleus::new(8, [0.0; 3]),
        Nucleus::new(1, [1.8, 0.0, 0.0]),
        Nucleus::new(1, [-0.45, 1.75, 0.0]),
        Nucleus::new(1, [0.0, 9.0, 0.0]),
        Nucleus::new(8, [0.0, -9.0, 0.0]),
    ];
    let why = floor(&sc, 100.0, false).expect("mixed stray species");
    assert!(why.contains("mixed"), "{why}");
    // two stray O: one species, but its reference has partial levels (not known to end at a gap)
    sc.nuclei[3].z = 8;
    // 10 + 16 = 26 electrons → M = 1 = 1 + 0 + 0: the spin rule passes, the gap rule refuses
    let why = floor(&sc, 100.0, false).expect("two open-shell stray O");
    assert!(why.contains("2 stray O"), "{why}");
}

#[test]
fn floor_refuses_a_stray_near_a_face() {
    let sc = h2_plus(&[[0.0, 8.0, 0.0]]);
    let why = floor(&sc, 3.9, false).expect("3.9 Bohr from a face");
    assert!(why.contains("face"), "{why}");
    assert_eq!(floor(&sc, 4.1, false), None);
    // the faces of the stray atoms only: a bonded atom near a face is the ordinary rules' business
    let split = split(&positions(&sc), R_CUT).unwrap();
    let near_bonded = |r: [f64; 3]| if r[1] == 8.0 { 10.0 } else { 1.0 };
    assert_eq!(floor_refusal(&sc, &split, &m_ref, &|_, _| false, &near_bonded), None);
    // a shell-check fence on (stray H, another H)
    let why = floor(&sc, 100.0, true).expect("a fenced H–H pair");
    assert!(why.contains("fence"), "{why}");
}

/// H₂ (fragment) + one far H, split exactly: E_scene = E_frag + E_atom, forces and levels the union. LSDA-like
/// levels (Ha): H₂ σg −0.37 both spins, σu +0.02; H 1s↑ −0.27, 1s↓ −0.05 (empty), 2s↑ +0.05.
fn exact() -> (PartAnswer, PartAnswer, Vec<PartAnswer>, Split) {
    let frag = PartAnswer {
        energy: -1.137,
        forces: vec![[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]],
        frontier: [Frontier { homo: Some(-0.37), lumo: Some(0.02), mu: None }, Frontier { homo: Some(-0.37), lumo: Some(0.02), mu: None }],
        partial: Vec::new(),
        clean: true,
        boxed: false,
    };
    let atom = PartAnswer {
        energy: -0.479,
        forces: vec![[0.0, 1e-4, 0.0]],
        frontier: [Frontier { homo: Some(-0.27), lumo: Some(0.05), mu: None }, Frontier { homo: None, lumo: Some(-0.05), mu: None }],
        partial: Vec::new(),
        clean: true,
        boxed: false,
    };
    let scene = PartAnswer {
        energy: frag.energy + atom.energy,
        forces: vec![frag.forces[0], frag.forces[1], atom.forces[0]],
        frontier: [Frontier { homo: Some(-0.27), lumo: Some(0.02), mu: None }, Frontier { homo: Some(-0.37), lumo: Some(-0.05), mu: None }],
        partial: Vec::new(),
        clean: true,
        boxed: false,
    };
    (scene, frag, vec![atom], Split { bonded: vec![0, 1], stray: vec![2] })
}

#[test]
fn judge_passes_an_exact_split() {
    let p = FallbackPolicy::default();
    let (scene, frag, atoms, split) = exact();
    match judge(&p, &scene, &frag, &atoms, &split) {
        CheckVerdict::Pass { residual, margin } => {
            assert!(residual < 1e-12, "{residual}");
            // the smallest of: σu↑ − 1s↑ = 0.29, 2s↑ − σg↑ = 0.42, 1s↓ − σg↓ = 0.32 (↓ has no stray HOMO)
            assert!((margin - 0.29).abs() < 1e-12, "{margin}");
        }
        v => panic!("{v:?}"),
    }
    assert!((frontier_margin_min() - 0.138).abs() < 1e-12, "27.6 kT at 5 mHa");
}

#[test]
fn judge_fails_an_energy_offset() {
    let p = FallbackPolicy::default();
    let (mut scene, frag, atoms, split) = exact();
    scene.energy += 2.0 * p.energy_std_per_atom;
    match judge(&p, &scene, &frag, &atoms, &split) {
        CheckVerdict::Fail(why) => assert!(why.contains("energy"), "{why}"),
        v => panic!("{v:?}"),
    }
    // just inside the tolerance (n_stray × energy_std_per_atom) passes
    let (mut scene, ..) = exact();
    scene.energy += 0.9 * p.energy_std_per_atom;
    assert!(matches!(judge(&p, &scene, &frag, &atoms, &split), CheckVerdict::Pass { .. }));
    // a bonded force that is not the fragment's, and a stray force that is not its own solve's
    let (mut scene, ..) = exact();
    scene.forces[1][0] += 0.01;
    match judge(&p, &scene, &frag, &atoms, &split) {
        CheckVerdict::Fail(why) => assert!(why.contains("force") && why.contains("bonded"), "{why}"),
        v => panic!("{v:?}"),
    }
    let (mut scene, ..) = exact();
    scene.forces[2][2] += 0.01;
    match judge(&p, &scene, &frag, &atoms, &split) {
        CheckVerdict::Fail(why) => assert!(why.contains("force") && why.contains("stray"), "{why}"),
        v => panic!("{v:?}"),
    }
}

#[test]
fn judge_fails_a_narrow_margin() {
    let p = FallbackPolicy::default();
    let (mut scene, frag, mut atoms, split) = exact();
    // the stray HOMO 0.1 Ha below the fragment's LUMO (0.1 < 0.138), the scene's levels consistent with it
    atoms[0].frontier[0].homo = Some(frag.frontier[0].lumo.unwrap() - 0.1);
    scene.frontier[0].homo = atoms[0].frontier[0].homo;
    match judge(&p, &scene, &frag, &atoms, &split) {
        CheckVerdict::Fail(why) => assert!(why.contains("margin"), "{why}"),
        v => panic!("{v:?}"),
    }
    // the reverse direction: the stray LUMO 0.1 Ha above the fragment's HOMO in ↓
    let (mut scene, frag, mut atoms, split) = exact();
    atoms[0].frontier[1].lumo = Some(frag.frontier[1].homo.unwrap() + 0.1);
    scene.frontier[1].lumo = atoms[0].frontier[1].lumo;
    match judge(&p, &scene, &frag, &atoms, &split) {
        CheckVerdict::Fail(why) => assert!(why.contains("margin"), "{why}"),
        v => panic!("{v:?}"),
    }
}

/// `exact` with the fragment open in ↑: its partial levels `levels` (ε, f), its ↑ HOMO the highest with f ≥ ½ (they lie
/// above σg), the scene's ↑ partial levels `scene_levels` with its ↑ HOMO likewise.
fn open_frag(levels: &[(f64, f64)], scene_levels: &[(f64, f64)]) -> (PartAnswer, PartAnswer, Vec<PartAnswer>, Split) {
    let (mut scene, mut frag, atoms, split) = exact();
    let to = |l: &[(f64, f64)], first: usize| -> Vec<PartialLevel> {
        l.iter().enumerate().map(|(k, &(eps, f))| PartialLevel { spin: 0, index: first + k, eps, f }).collect()
    };
    let homo = |l: &[(f64, f64)], other: f64| l.iter().filter(|x| x.1 >= 0.5).map(|x| x.0).fold(other, f64::max);
    frag.partial = to(levels, 1);
    frag.frontier[0].homo = Some(homo(levels, -0.37));
    scene.partial = to(scene_levels, 2);
    scene.frontier[0].homo = Some(homo(scene_levels, -0.27));
    (scene, frag, atoms, split)
}

#[test]
fn judge_fails_mismatched_partial_levels() {
    let p = FallbackPolicy::default();
    let fail_words = |v: CheckVerdict| match v {
        CheckVerdict::Fail(why) => why,
        v => panic!("{v:?}"),
    };
    // the fragment's partial level (f ≥ ½: its ↑ HOMO) reappears in the scene, renumbered, within the box splitting
    let (scene, frag, atoms, split) = open_frag(&[(-0.1, 0.6)], &[(-0.1 + 1e-3, 0.62)]);
    assert!(matches!(judge(&p, &scene, &frag, &atoms, &split), CheckVerdict::Pass { .. }), "{:?}", judge(&p, &scene, &frag, &atoms, &split));
    // a level the parts do not have (charge pinned at μ in the scene; f ≥ ½ there, so the scene's HOMO)
    let (scene, frag, atoms, split) = open_frag(&[], &[(-0.2, 0.6)]);
    let why = fail_words(judge(&p, &scene, &frag, &atoms, &split));
    assert!(why.contains("levels") && why.contains("partly occupied"), "{why}");
    // a level moved by more than the box splitting
    let (scene, frag, atoms, split) = open_frag(&[(-0.1, 0.6)], &[(-0.1 + 5e-3, 0.6)]);
    assert!(fail_words(judge(&p, &scene, &frag, &atoms, &split)).contains("levels"));
    // Σf not conserved (each f within its bound, 0.1 < 0.18; the sum off by 0.2)
    let (scene, frag, atoms, split) = open_frag(&[(-0.1, 0.7), (-0.09, 0.6)], &[(-0.1, 0.6), (-0.09, 0.5)]);
    let why = fail_words(judge(&p, &scene, &frag, &atoms, &split));
    assert!(why.contains("Σf"), "{why}");
    // one f beyond its bound (0.2 > 0.18) at levels 10 mHa apart: unambiguous, a failure
    let (scene, frag, atoms, split) = open_frag(&[(-0.1, 0.7), (-0.09, 0.6)], &[(-0.1, 0.9), (-0.09, 0.5)]);
    let why = fail_words(judge(&p, &scene, &frag, &atoms, &split));
    assert!(why.contains("has f"), "{why}");
    // the same at near-degenerate levels (0.5 mHa apart): the ε order need not pair the same orbitals — void
    let (scene, frag, atoms, split) = open_frag(&[(-0.1, 0.9), (-0.0995, 0.6)], &[(-0.1, 0.6), (-0.0995, 0.9)]);
    match judge(&p, &scene, &frag, &atoms, &split) {
        CheckVerdict::Void(why) => assert!(why.contains("near-degenerate"), "{why}"),
        v => panic!("{v:?}"),
    }
    // a frontier level that is not the union's
    let (mut scene, frag, atoms, split) = exact();
    scene.frontier[0].homo = Some(-0.27 + 5e-3);
    assert!(fail_words(judge(&p, &scene, &frag, &atoms, &split)).contains("levels"));
}

/// Review I1: a partial level donates and accepts, whatever side of ½ its f is.
#[test]
fn judge_fails_a_stray_whose_partial_level_is_its_homo() {
    let p = FallbackPolicy::default();
    // H₂ + a stray open-shell O-like atom at its partial ensemble: 2p at −0.33 Ha, f ≈ 2/3 in both spins (so its HOMO),
    // 3s LUMO at +0.1. HOMO/LUMO alone: σu − 2p = 0.35 and 3s − σg = 0.47 pass; the real acceptor is the 2p itself,
    // 40 mHa above σg
    let (mut scene, frag, mut atoms, split) = exact();
    let o2p = |spin: u8| PartialLevel { spin, index: 2, eps: -0.33, f: 2.0 / 3.0 };
    atoms[0].frontier = [Frontier { homo: Some(-0.33), lumo: Some(0.1), mu: None }; 2];
    atoms[0].partial = vec![o2p(0), o2p(1)];
    scene.frontier = [Frontier { homo: Some(-0.33), lumo: Some(0.02), mu: None }; 2];
    scene.partial = vec![PartialLevel { index: 3, ..o2p(0) }, PartialLevel { index: 3, ..o2p(1) }];
    match judge(&p, &scene, &frag, &atoms, &split) {
        CheckVerdict::Fail(why) => assert!(why.contains("margin") && why.contains("stray acceptor"), "{why}"),
        v => panic!("{v:?}"),
    }
    // a partial level with f < ½ and no HOMO in its channel still donates (the stray H's ↓ partly filled)
    let (mut scene, frag, mut atoms, split) = exact();
    atoms[0].partial = vec![PartialLevel { spin: 1, index: 0, eps: -0.05, f: 0.3 }];
    scene.partial = vec![PartialLevel { spin: 1, index: 1, eps: -0.05, f: 0.3 }];
    match judge(&p, &scene, &frag, &atoms, &split) {
        CheckVerdict::Fail(why) => assert!(why.contains("margin") && why.contains("fragment acceptor"), "{why}"),
        v => panic!("{v:?}"),
    }
    // two parts both open in one channel fail one direction, however far apart their levels: one μ_σ is shared
    let (mut scene, mut frag, mut atoms, split) = exact();
    frag.frontier[0] = Frontier { homo: Some(-0.5), lumo: Some(0.3), mu: None };
    frag.partial = vec![PartialLevel { spin: 0, index: 0, eps: -0.5, f: 0.6 }];
    atoms[0].frontier[0] = Frontier { homo: Some(0.0), lumo: Some(0.4), mu: None };
    atoms[0].partial = vec![PartialLevel { spin: 0, index: 0, eps: 0.0, f: 0.6 }];
    scene.frontier[0] = Frontier { homo: Some(0.0), lumo: Some(0.3), mu: None };
    scene.partial = vec![frag.partial[0].clone(), atoms[0].partial[0].clone()];
    match judge(&p, &scene, &frag, &atoms, &split) {
        CheckVerdict::Fail(why) => assert!(why.contains("margin"), "{why}"),
        v => panic!("{v:?}"),
    }
}

#[test]
fn judge_voids_an_unclean_solve_and_a_missing_lumo() {
    let p = FallbackPolicy::default();
    for which in 0..3 {
        let (mut scene, mut frag, mut atoms, split) = exact();
        match which {
            0 => scene.clean = false,
            1 => frag.clean = false,
            _ => atoms[0].clean = false,
        }
        // even with an energy that would fail: the unclean solve is not a measurement
        scene.energy += 1.0;
        assert!(matches!(judge(&p, &scene, &frag, &atoms, &split), CheckVerdict::Void(_)), "solve {which}");
    }
    // the fragment's LUMO, which the margin against the stray HOMO needs
    let (scene, mut frag, atoms, split) = exact();
    frag.frontier[0].lumo = None;
    assert_eq!(judge(&p, &scene, &frag, &atoms, &split), CheckVerdict::Void("frontier level not computed".into()));
    // the scene's LUMO, which the levels test needs
    let (mut scene, frag, atoms, split) = exact();
    scene.frontier[1].lumo = None;
    assert_eq!(judge(&p, &scene, &frag, &atoms, &split), CheckVerdict::Void("frontier level not computed".into()));
    // a scene that reports no frontier level at all while its parts do (a wiring slip): void, never a refusal
    let (mut scene, frag, atoms, split) = exact();
    scene.frontier = [Frontier::default(); 2];
    assert_eq!(judge(&p, &scene, &frag, &atoms, &split), CheckVerdict::Void("frontier level not computed".into()));
    // an absent HOMO (the stray H's empty ↓ channel) needs no margin: `exact` passes with it
}

/// H₂ descriptors at bond length `r` under `basis`, and their species indices.
fn h2_desc(basis: &Basis, r: f64) -> (Vec<Vec<f64>>, Vec<usize>) {
    let pos = [[-0.5 * r, 0.0, 0.0], [0.5 * r, 0.0, 0.0]];
    (describe_values(basis, &pos, &[1, 1]).unwrap(), basis.species_indices(&[1, 1]).unwrap())
}

/// A training envelope (the model's, whose spreads are the store's thinning units) of H₂ at 1.3–1.6 Bohr.
fn units(basis: &Basis) -> Envelope {
    let mut env = Envelope::new(basis);
    for i in 0..21 {
        let (d, s) = h2_desc(basis, 1.3 + 0.3 * i as f64 / 20.0);
        for (si, b) in s.iter().zip(&d) {
            env.add_atom(*si, b);
        }
    }
    env
}

fn sig() -> Signature {
    Signature { bonded_z: vec![1, 1], stray_z: vec![1], charge: 0, surface: "test".into() }
}

const PASS: CheckVerdict = CheckVerdict::Pass { residual: 1e-4, margin: 0.29 };

/// Where the pure tests' checks put the stray atom (Bohr): 8 from the bonded part, 5 from a face.
const PLACE: StrayPlace = StrayPlace { reach: 8.0, face: 5.0 };

fn thin() -> f64 {
    THIN_FRACTION * FallbackPolicy::default().max_descriptor_distance
}

#[test]
fn region_clears_after_three_distinct_passes() {
    let basis = Basis::new(&[1, 8]);
    let env = units(&basis);
    let mut book = StrayBook::default();
    for (k, r) in [1.35, 1.45, 1.55].into_iter().enumerate() {
        let (d, s) = h2_desc(&basis, r);
        let v = if k == 1 { CheckVerdict::Pass { residual: 3e-4, margin: 0.2 } } else { PASS };
        book.record(sig(), v, &d, &s, PLACE, &basis, &env, thin());
        match (k, book.state(&sig())) {
            (0 | 1, Some(StrayState::Checking { passes, .. })) => assert_eq!(*passes, k + 1),
            (2, Some(StrayState::Cleared { residual, margin, .. })) => {
                assert_eq!((*residual, *margin), (3e-4, 0.2), "the worst residual and the smallest margin of the passes");
            }
            (k, st) => panic!("after pass {k}: {st:?}"),
        }
    }
    assert_eq!(STRAY_CHECKS, 3);
    // a pass outside the region grows it, and the worst numbers follow
    let (d, s) = h2_desc(&basis, 1.25);
    let Some(StrayState::Cleared { region, .. }) = book.state(&sig()) else { unreachable!() };
    assert!(!region.contains(&d, &s));
    book.record(sig(), CheckVerdict::Pass { residual: 5e-4, margin: 0.25 }, &d, &s, PLACE, &basis, &env, thin());
    let Some(StrayState::Cleared { region, residual, margin, .. }) = book.state(&sig()) else { panic!() };
    assert!(region.contains(&d, &s));
    assert_eq!((*residual, *margin), (5e-4, 0.2));
    // the book is session state: `clear` forgets it
    book.clear();
    assert!(book.state(&sig()).is_none());
}

#[test]
fn passes_at_one_geometry_do_not_clear() {
    let basis = Basis::new(&[1, 8]);
    let env = units(&basis);
    let mut book = StrayBook::default();
    // the same bonded geometry three times (a check differing only in stray distance has the same bonded descriptors)
    let (d, s) = h2_desc(&basis, 1.45);
    for _ in 0..3 {
        book.record(sig(), PASS, &d, &s, PLACE, &basis, &env, thin());
    }
    assert!(matches!(book.state(&sig()), Some(StrayState::Checking { passes: 1, .. })), "{:?}", book.state(&sig()));
    // nor within the thinning radius of it
    let (near, _) = h2_desc(&basis, 1.451);
    book.record(sig(), PASS, &near, &s, PLACE, &basis, &env, thin());
    assert!(matches!(book.state(&sig()), Some(StrayState::Checking { passes: 1, .. })));
    // and a species the training never saw cannot be measured for distinctness: never counted
    let empty = Envelope::new(&basis);
    let (far, _) = h2_desc(&basis, 1.9);
    book.record(sig(), PASS, &far, &s, PLACE, &basis, &empty, thin());
    assert!(matches!(book.state(&sig()), Some(StrayState::Checking { passes: 1, .. })));
    // a void changes nothing
    let (d2, _) = h2_desc(&basis, 1.55);
    book.record(sig(), CheckVerdict::Void("x".into()), &d2, &s, PLACE, &basis, &env, thin());
    assert!(matches!(book.state(&sig()), Some(StrayState::Checking { passes: 1, .. })));
}

#[test]
fn region_contains_its_points_and_not_beyond() {
    let basis = Basis::new(&[1, 8]);
    let env = units(&basis);
    let mut region = Region::new(&basis);
    let (d0, s) = h2_desc(&basis, 1.45);
    assert!(!region.contains(&d0, &s), "an empty region contains nothing");
    for r in [1.35, 1.45, 1.55] {
        let (d, s) = h2_desc(&basis, r);
        region.add(&d, &s, &env, thin());
    }
    for r in [1.35, 1.45, 1.55] {
        let (d, s) = h2_desc(&basis, r);
        assert!(region.contains(&d, &s), "{r}");
    }
    for r in [1.2, 1.7, 2.5] {
        let (d, s) = h2_desc(&basis, r);
        assert!(!region.contains(&d, &s), "{r}");
    }
    // a species absent from the region: the O index with an H descriptor
    assert!(!region.contains(&d0, &[1, 1]));
    // the distinctness of a point from itself is 0
    assert!(!region.distinct(&d0, &s, &env));
    let (far, _) = h2_desc(&basis, 1.9);
    assert!(region.distinct(&far, &s, &env));
}

#[test]
fn one_fail_refuses_for_good() {
    let basis = Basis::new(&[1, 8]);
    let env = units(&basis);
    let mut book = StrayBook::default();
    let geoms: Vec<_> = [1.35, 1.45, 1.55, 1.25].iter().map(|&r| h2_desc(&basis, r)).collect();
    book.record(sig(), PASS, &geoms[0].0, &geoms[0].1, PLACE, &basis, &env, thin());
    book.record(sig(), CheckVerdict::Fail("energy: 2.00 mHa".into()), &geoms[1].0, &geoms[1].1, PLACE, &basis, &env, thin());
    assert_eq!(book.state(&sig()).map(|s| matches!(s, StrayState::Refused(w) if w.contains("energy"))), Some(true));
    for (d, s) in &geoms {
        book.record(sig(), PASS, d, s, PLACE, &basis, &env, thin());
    }
    assert!(matches!(book.state(&sig()), Some(StrayState::Refused(_))), "passes after a fail do not count");
    // a cleared signature is refused by one fail too
    let mut other = sig();
    other.stray_z = vec![1, 1];
    for (d, s) in &geoms[..3] {
        book.record(other.clone(), PASS, d, s, PLACE, &basis, &env, thin());
    }
    assert!(matches!(book.state(&other), Some(StrayState::Cleared { .. })));
    book.record(other.clone(), CheckVerdict::Fail("margin".into()), &geoms[3].0, &geoms[3].1, PLACE, &basis, &env, thin());
    assert!(matches!(book.state(&other), Some(StrayState::Refused(_))));
    // a refusal by hand (the floor, a failed audit), on a signature never checked
    let mut third = sig();
    third.charge = 1;
    book.refuse(&third, "charge".into());
    assert!(matches!(book.state(&third), Some(StrayState::Refused(w)) if w == "charge"));
    // the first signature is untouched by the others
    assert!(matches!(book.state(&sig()), Some(StrayState::Refused(w)) if w.contains("energy")));
}

// ---- plan Task 3: the checks on the auditor, the book's lifetime, the sample hold-out ----

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};

/// Where the auditor tests put H₂ (its centre) and the stray H: 8 Bohr apart, the H 5 Bohr from the faces of the
/// 81³ cell (half-extent 10 Bohr at h = 0.25), so the floor's 4-Bohr face margin holds.
const H2_AT: [f64; 3] = [0.0, -3.0, 0.0];
const STRAY_AT: [f64; 3] = [0.0, 5.0, 0.0];

fn big(mut sc: Scene) -> Scene {
    sc.cell = Cell { n: 81, h: 0.25 };
    sc
}

/// H₂ at `r` along x about [`H2_AT`] plus one H per entry of `strays`, in the 81³ cell.
fn h2_with(r: f64, strays: &[[f64; 3]]) -> Scene {
    let mut sc = big(dimer(r, [1.0, 0.0, 0.0], H2_AT, 1, 1));
    sc.nuclei.extend(strays.iter().map(|&p| Nucleus::new(1, p)));
    sc
}

/// The per-spin frontier levels a composing Kohn-Sham teacher reports (Ha; a stand-in, not a measurement): per cluster
/// of atoms joined by neighbours within `r_cut`, an even-electron one has its HOMO at −0.37 in both spins and its LUMO
/// at +0.05, an odd-electron one its 1s↑ at −0.27 (LUMO↑ 0, LUMO↓ −0.05, no ↓ HOMO); the scene's are the clusters'
/// union (HOMO the highest, LUMO the lowest).
fn cluster_levels(pos: &[[f64; 3]], z: &[u8], r_cut: f64) -> [Frontier; 2] {
    let n = pos.len();
    let mut label: Vec<usize> = (0..n).collect();
    loop {
        let mut moved = false;
        for i in 0..n {
            for j in 0..n {
                let d2: f64 = (0..3).map(|k| (pos[i][k] - pos[j][k]).powi(2)).sum();
                if i != j && d2 < r_cut * r_cut && label[j] < label[i] {
                    label[i] = label[j];
                    moved = true;
                }
            }
        }
        if !moved {
            break;
        }
    }
    let mut out = [Frontier { homo: None, lumo: None, mu: None }; 2];
    let mut roots: Vec<usize> = label.clone();
    roots.sort_unstable();
    roots.dedup();
    for c in roots {
        let electrons: u32 = (0..n).filter(|&i| label[i] == c).map(|i| z[i] as u32).sum();
        let levels = if electrons % 2 == 0 { [(Some(-0.37), Some(0.05)), (Some(-0.37), Some(0.05))] } else { [(Some(-0.27), Some(0.0)), (None, Some(-0.05))] };
        for (f, (homo, lumo)) in out.iter_mut().zip(levels) {
            f.homo = match (f.homo, homo) {
                (Some(a), Some(b)) => Some(f64::max(a, b)),
                (a, b) => a.or(b),
            };
            f.lumo = match (f.lumo, lumo) {
                (Some(a), Some(b)) => Some(f64::min(a, b)),
                (a, b) => a.or(b),
            };
        }
    }
    out
}

/// A composing teacher for the checks: [`SpeciesMorse`] (cut off smoothly at r_cut, so the energy splits exactly into
/// the parts') with [`cluster_levels`], face density 0 and the automatic multiplicity (or the scene's). `offset` (Ha)
/// is added whenever the scene has a stray atom within `offset_near` (Bohr) of a bonded atom — a composite error the
/// parts alone never show (the spin or charge-transfer case the check exists for; an induction-like term that grows as
/// the stray atom comes nearer, final review I1, when `offset_near` is finite). `boxed_within` (Bohr): every solve with
/// an atom nearer than this to a face reports a face density 10× the box rule (box-affected; final review I3), 0 never.
/// `open` gates the answer (the auditor's instance), `calls` counts solves.
#[derive(Clone)]
struct Composing {
    offset: f64,
    offset_near: f64,
    boxed_within: f64,
    open: Arc<AtomicBool>,
    calls: Arc<AtomicUsize>,
}

impl Composing {
    fn new(offset: f64) -> Composing {
        Composing { offset, offset_near: f64::INFINITY, boxed_within: 0.0, open: Arc::new(AtomicBool::new(true)), calls: Arc::new(AtomicUsize::new(0)) }
    }
}

impl ForceProvider for Composing {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        while !self.open.load(SeqCst) {
            if should_abort() {
                return Err(ProviderError::Cancelled);
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        self.calls.fetch_add(1, SeqCst);
        let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        let z: Vec<u8> = scene.nuclei.iter().map(|n| n.z).collect();
        let r_cut = Basis::new(&[1]).r_cut;
        let (mut energy, forces) = SpeciesMorse::energy_and_forces(&pos, &z);
        if let Some(sp) = split(&pos, r_cut) {
            let d = |a: [f64; 3], b: [f64; 3]| (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt();
            let reach = sp.stray.iter().flat_map(|&i| sp.bonded.iter().map(move |&j| (i, j))).map(|(i, j)| d(pos[i], pos[j])).fold(f64::INFINITY, f64::min);
            if reach < self.offset_near {
                energy += self.offset;
            }
        }
        let electrons = z.iter().map(|&x| x as i64).sum::<i64>() - scene.charge as i64;
        let mut stats = EvalStats::default();
        stats.energy.total = energy;
        let half = scene.cell.half_extent();
        let face = pos.iter().map(|p| half - p.iter().fold(0.0f64, |m, x| m.max(x.abs()))).fold(f64::INFINITY, f64::min);
        stats.face_density = Some(if face < self.boxed_within { 10.0 * box_affected_face_density(scene.cell.h) } else { 0.0 });
        stats.multiplicity = Some(scene.multiplicity.unwrap_or((electrons.rem_euclid(2) + 1) as u32));
        stats.frontier = cluster_levels(&pos, &z, r_cut);
        Ok(ForceResult { energy, forces, stats, display: None, orbitals: None })
    }

    fn name(&self) -> &'static str {
        "composing morse"
    }

    fn reset(&mut self) {}
}

/// The teacher's exact answer at `sc` as a scan sample.
fn exact_sample(sc: &Scene) -> Sample {
    let pos = positions(sc);
    let z: Vec<u8> = sc.nuclei.iter().map(|n| n.z).collect();
    let (energy, forces) = SpeciesMorse::energy_and_forces(&pos, &z);
    Sample { pos, z, energy, forces, tag: SampleTag::Scan, weight: 1.0, force_mask: None, use_energy: true }
}

/// A trusted strict-mode H₂ student of the [`Composing`] teacher (`offset`): `n` H₂ samples at r ∈ [lo, hi] in the 81³
/// cell, refitted; with a background auditor (a second instance, gated by the returned flag; its solve count) when
/// `auditor`. No audits (`audit_every` 0): every step of these tests is a fallback.
fn student(offset: f64, ranges: &[(f64, f64)], n: usize, auditor: bool) -> (SurrogateKs, Arc<AtomicBool>, Arc<AtomicUsize>) {
    student_of(&Composing::new(offset), ranges, n, auditor)
}

/// [`student`] of the teacher `c` (its offset, `offset_near` and `boxed_within`; the teacher and the auditor alike).
fn student_of(c: &Composing, ranges: &[(f64, f64)], n: usize, auditor: bool) -> (SurrogateKs, Arc<AtomicBool>, Arc<AtomicUsize>) {
    let mut s = SurrogateKs::new(Box::new(c.clone()), FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() }).with_teacher_tol(1e-6);
    s.set_reference_energy(1, SpeciesMorse::atom(1));
    let mut rng = StdRng::seed_from_u64(0x57a);
    for &(lo, hi) in ranges {
    for i in 0..n {
        let r = lo + (hi - lo) * i as f64 / (n - 1) as f64;
        let origin: [f64; 3] = std::array::from_fn(|k| H2_AT[k] + rng.gen_range(-0.3..0.3));
        let sc = big(dimer(r, unit_vector(&mut rng), origin, 1, 1));
        s.add_sample(exact_sample(&sc), &sc).unwrap();
    }
    }
    s.refit().expect("a fit forms");
    assert!(s.trusted(), "the premise: a trusted H₂ model ({:?})", s.untrusted_reason());
    let gate = Arc::new(AtomicBool::new(true));
    let calls = Arc::new(AtomicUsize::new(0));
    if auditor {
        s = s.with_auditor(Box::new(Composing { open: gate.clone(), calls: calls.clone(), ..c.clone() }));
    }
    (s, gate, calls)
}

/// Polls the auditor until no composition check is in flight and it is idle (≤ 5 s).
fn settle(s: &mut SurrogateKs) {
    let t0 = Instant::now();
    while s.stray_pending.is_some() || !s.side_queue.is_empty() || !s.auditor_idle() {
        assert!(t0.elapsed().as_secs_f64() < 5.0, "the check did not finish within 5 s");
        std::thread::sleep(std::time::Duration::from_millis(1));
        s.poll_side_jobs();
    }
}

fn zs(sc: &Scene) -> Vec<u8> {
    sc.nuclei.iter().map(|n| n.z).collect()
}

/// `sc`'s signature at the student's cutoff (the floor passes: asserted).
fn sig_of(s: &SurrogateKs, sc: &Scene) -> Signature {
    let (_, sig, floor) = s.stray_split(&positions(sc), &zs(sc), &s.provenance_for(sc)).expect("a split");
    assert!(floor.is_none(), "the premise: the floor passes ({floor:?})");
    sig
}

/// The check geometries (H–H, Bohr) and the student's training window: H₂ trained at 1.0–1.25 Bohr, so the bonded
/// part's spreads are small and 1.35, 1.45 and 1.55 are each beyond the store's thinning radius of the training samples
/// and of each other (controller ruling A: asserted in [`assert_distinct`], so no test passes vacuously).
const CHECKS: [f64; 3] = [1.35, 1.45, 1.55];
const TRAINED: [(f64, f64); 1] = [(1.0, 1.25)];

/// Ruling A: the bonded geometries `rs` are pairwise distinct in the store's thinning metric (the model's training
/// spreads, `THIN_FRACTION` × `max_descriptor_distance`), and none is within it of a training sample.
fn assert_distinct(s: &mut SurrogateKs, rs: &[f64]) {
    let basis = s.model().unwrap().basis.clone();
    let units = s.envelope().unwrap().clone();
    let thin = THIN_FRACTION * s.policy().max_descriptor_distance;
    for (a, &ra) in rs.iter().enumerate() {
        let mut region = Region::new(&basis);
        let (d, sp) = h2_desc(&basis, ra);
        region.add(&d, &sp, &units, thin);
        for &rb in &rs[a + 1..] {
            let (d, sp) = h2_desc(&basis, rb);
            assert!(region.distinct(&d, &sp, &units), "the premise: H–H {ra} and {rb} are distinct geometries");
        }
        let h2 = big(dimer(ra, [1.0, 0.0, 0.0], H2_AT, 1, 1));
        let d = s.nearest_stored(&positions(&h2), &zs(&h2));
        assert!(d > thin, "the premise: H–H {ra} is {d:.2} > {thin:.2} spreads from every training sample");
    }
}

/// The last event's outcome among those drained.
fn outcomes(s: &mut SurrogateKs) -> Vec<TeachOutcome> {
    s.take_events().iter().map(|e| e.outcome).collect()
}

/// Ruling D: `Q_STRAY` is no other in-memory code and no code a collection status record (or a file) can hold.
#[test]
fn stray_code_is_no_other_code() {
    let others = [Q_NONE, Q_HELD, Q_SUPERSEDED, Q_BOX, Q_DRAFT, Q_REF_WAIT, Q_LONE, Q_FORGOTTEN, Q_FORGOTTEN_UNFENCED, surrogate::quarantine::Q_RESOLVED];
    assert!(!others.contains(&Q_STRAY), "{Q_STRAY} is taken");
    for c in 0..=u8::MAX {
        if surrogate::quarantine::is_collection_code(c) || surrogate::quarantine::is_persisted_code(c) {
            assert_ne!(c, Q_STRAY, "a collection or a file can hold {c}");
        }
    }
    assert_eq!(Q_STRAY, 10, "the next free value after the migration's 9");
}

/// Plan Task 3 step 1: one strict step at H₂ + a stray H 8 Bohr away falls back (rule 5a), keeps its sample held out
/// (`Q_STRAY`) and queues the bonded part alone and the stray atom alone behind the auditor's job in flight, on the
/// scene's own surface (its provenance, cell, σ and ionic policy; review M5): the fragment neutral at automatic
/// multiplicity, the atom at its reference multiplicity (2). A second fallback while the check is in flight queues
/// nothing; the check's pass releases its own sample, and the other stays held (its geometry was not measured).
#[test]
fn a_stray_fallback_queues_fragment_and_atom_jobs() {
    let (mut s, gate, calls) = student(0.0, &TRAINED, 40, true);
    let sc = h2_with(CHECKS[0], &[STRAY_AT]);
    // the auditor busy with a display, so both parts wait in its queue
    gate.store(false, SeqCst);
    assert!(s.request_display(&big(h2(1.1))));
    let r = s.evaluate(&sc, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback, "{:?}", r.stats.notes);
    let ev = s.take_events();
    assert_eq!(ev.last().map(|e| (e.rule, e.outcome)), Some((TeachRule::IsolatedAtom, TeachOutcome::KeptStrayHeld)), "{ev:?}");
    assert!(r.stats.notes.iter().any(|n| n.contains("stray atoms (H2 + H) — a composition check") && n.contains("queued")), "{:?}", r.stats.notes);
    assert!(r.stats.notes.iter().any(|n| n.contains("held out of the fit — stray atoms (H2 + H)")), "{:?}", r.stats.notes);
    let i = s.samples() - 1;
    assert_eq!((s.store().unwrap().samples[i].pos.len(), s.quarantine[i], s.stray_held_out()), (3, Q_STRAY, 1));
    // the jobs
    let prov = s.provenance_for(&sc);
    let kinds: Vec<JobKind> = s.side_queue.iter().map(|j| j.2).collect();
    assert_eq!(kinds, [JobKind::Fragment, JobKind::AtomAt(1)]);
    let (frag, fprov, _, fid) = &s.side_queue[0];
    assert_eq!(positions(frag), positions(&sc)[..2].to_vec());
    assert_eq!((frag.charge, frag.multiplicity, frag.cell.clone(), frag.sigma, frag.ionic), (0, None, sc.cell.clone(), sc.sigma, sc.ionic));
    assert_eq!(fprov, &prov, "the scene's surface, never the fragment's own tag");
    let (atom, aprov, _, aid) = &s.side_queue[1];
    assert_eq!(positions(atom), vec![STRAY_AT]);
    assert_eq!((atom.charge, atom.multiplicity, atom.cell.clone(), atom.ionic), (0, Some(2), sc.cell.clone(), sc.ionic));
    assert_eq!((aprov, aid), (&prov, fid));
    let pending = s.stray_pending.as_ref().expect("a check in flight");
    assert_eq!((pending.frag.is_none(), pending.atoms.len(), pending.sample), (true, 1, Some(i)));
    // a second fallback while it is in flight queues nothing
    let other = h2_with(CHECKS[1], &[STRAY_AT]);
    assert_eq!(s.evaluate(&other, &mut never()).unwrap().stats.source, Source::Fallback);
    let j = s.samples() - 1;
    assert_eq!((j, s.quarantine[j]), (i + 1, Q_STRAY), "kept, held out");
    assert_eq!(s.side_queue.len(), 2);
    assert_eq!(s.stray_pending.as_ref().map(|p| p.sample), Some(Some(i)), "the first check stays");
    // the check runs: display, fragment, atom
    gate.store(true, SeqCst);
    settle(&mut s);
    assert_eq!(calls.load(SeqCst), 3);
    let ev = s.take_events();
    let check: Vec<&sim::TeachEvent> = ev.iter().filter(|e| e.origin == TeachOrigin::Audit).collect();
    assert_eq!(check.iter().map(|e| (e.rule, e.outcome)).collect::<Vec<_>>(), [(TeachRule::IsolatedAtom, TeachOutcome::StrayChecking { n: 1, of: 3 })], "{ev:?}");
    assert!(check[0].detail.contains("composition check passed") && check[0].detail.contains("1 held sample released"), "{}", check[0].detail);
    let sig = sig_of(&s, &sc);
    assert!(matches!(s.stray_book.state(&sig), Some(StrayState::Checking { passes: 1, .. })));
    assert_eq!((s.quarantine[i], s.quarantine[j]), (Q_NONE, Q_STRAY), "the measured sample is released, the other stays held");
}

/// Plan Task 3 step 1: three fallbacks at distinct bonded geometries (ruling A, asserted) clear the signature, each
/// check releasing its own sample; a held sample whose own check never ran is released when the region that opens
/// covers it; the refit then fits them all.
#[test]
fn three_distinct_passes_clear_and_release() {
    let (mut s, gate, _) = student(0.0, &TRAINED, 40, true);
    assert_distinct(&mut s, &CHECKS);
    let sig = sig_of(&s, &h2_with(CHECKS[0], &[STRAY_AT]));
    let mut kept = Vec::new();
    for (k, &r) in CHECKS.iter().enumerate() {
        if k == 2 {
            gate.store(false, SeqCst);
        }
        assert_eq!(s.evaluate(&h2_with(r, &[STRAY_AT]), &mut never()).unwrap().stats.source, Source::Fallback);
        let i = s.samples() - 1;
        assert_eq!(s.quarantine[i], Q_STRAY, "check {k}: held out until measured");
        kept.push(i);
        if k < 2 {
            settle(&mut s);
            assert!(matches!(s.stray_book.state(&sig), Some(StrayState::Checking { passes, .. }) if *passes == k + 1), "{:?}", s.stray_book.state(&sig));
            assert_eq!(s.quarantine[i], Q_NONE, "check {k}: its own pass releases it");
        }
    }
    // while the third check is in flight: a fallback between the checked geometries, kept and held (no check queued)
    let between = h2_with(1.5, &[STRAY_AT]);
    assert_eq!(s.evaluate(&between, &mut never()).unwrap().stats.source, Source::Fallback);
    let b = s.samples() - 1;
    assert_eq!(s.quarantine[b], Q_STRAY);
    gate.store(true, SeqCst);
    settle(&mut s);
    let Some(StrayState::Cleared { region, residual, margin, .. }) = s.stray_book.state(&sig) else { panic!("cleared: {:?}", s.stray_book.state(&sig)) };
    println!("cleared: worst residual {residual:.2e} Ha, smallest margin {margin:.3} Ha");
    assert!(*residual < 1e-12 && (*margin - 0.32).abs() < 1e-9, "an exact split, the stand-in levels' margin");
    let (d, sp) = s.bonded_descriptors(&positions(&between), &zs(&between), &Split { bonded: vec![0, 1], stray: vec![2] }).unwrap();
    assert!(region.contains(&d, &sp), "the premise: the region covers H–H 1.5");
    let outs = outcomes(&mut s);
    let checks: Vec<TeachOutcome> = outs.into_iter().filter(|o| matches!(o, TeachOutcome::StrayChecking { .. })).collect();
    assert_eq!(checks, [1, 2, 3].map(|n| TeachOutcome::StrayChecking { n, of: 3 }));
    assert!(kept.iter().chain([&b]).all(|&i| s.quarantine[i] == Q_NONE), "{:?}", &s.quarantine[40..]);
    assert_eq!(s.stray_held_out(), 0);
    assert!(s.pending_refit && s.species_dirty, "the refit is armed");
    // a fallback inside the cleared region (H–H 1.4: beyond the thinning radius of every stored sample in
    // the training spreads, so it is kept)
    // is kept in the fit at once, and queues no check
    let inside = h2_with(1.4, &[STRAY_AT]);
    let (d, sp) = s.bonded_descriptors(&positions(&inside), &zs(&inside), &Split { bonded: vec![0, 1], stray: vec![2] }).unwrap();
    assert!(matches!(s.stray_book.state(&sig), Some(StrayState::Cleared { region, .. }) if region.contains(&d, &sp)), "the premise: the region covers H–H 1.4");
    let n = s.samples();
    assert_eq!(s.evaluate(&inside, &mut never()).unwrap().stats.source, Source::Fallback, "answering is plan Task 4's");
    assert_eq!(s.samples(), n + 1, "kept, not thinned");
    let i = s.samples() - 1;
    assert_eq!((s.store().unwrap().samples[i].pos.clone(), s.excluded(i)), (positions(&inside), false), "covered: in the fit at once");
    assert!(s.stray_pending.is_none() && s.side_queue.is_empty(), "no check inside a cleared region");
    let rep = s.refit().unwrap();
    assert_eq!(rep.n_samples, 45, "the released samples and the covered one are fitted");
}

/// Plan Task 3 step 1: a composite offset (5 mHa whenever a stray atom is present, which the parts alone never show)
/// fails the energy test: the signature is refused for the session and its samples stay held out for good — no
/// later check, no release, never fitted.
#[test]
fn a_composite_offset_refuses_and_holds_out_for_good() {
    let (mut s, _, calls) = student(5e-3, &TRAINED, 40, true);
    let sc = h2_with(CHECKS[0], &[STRAY_AT]);
    assert_eq!(s.evaluate(&sc, &mut never()).unwrap().stats.source, Source::Fallback);
    let i = s.samples() - 1;
    settle(&mut s);
    let sig = sig_of(&s, &sc);
    assert!(matches!(s.stray_book.state(&sig), Some(StrayState::Refused(why)) if why.starts_with("energy: |E_scene − E_frag − Σ E_atom| = 5.000 mHa")), "{:?}", s.stray_book.state(&sig));
    let ev = s.take_events();
    let refused: Vec<&sim::TeachEvent> = ev.iter().filter(|e| e.outcome == TeachOutcome::StrayRefused).collect();
    assert_eq!(refused.len(), 1, "{ev:?}");
    assert!(refused[0].detail.contains("composition fails: energy") && refused[0].rule == TeachRule::IsolatedAtom, "{}", refused[0].detail);
    assert_eq!(s.quarantine[i], Q_STRAY);
    // refused for good: the next fallbacks queue nothing and stay held
    let solved = calls.load(SeqCst);
    for &r in &CHECKS[1..] {
        assert_eq!(s.evaluate(&h2_with(r, &[STRAY_AT]), &mut never()).unwrap().stats.source, Source::Fallback);
        assert!(s.stray_pending.is_none() && s.side_queue.is_empty());
        assert_eq!(s.quarantine[s.samples() - 1], Q_STRAY);
    }
    settle(&mut s);
    assert_eq!(calls.load(SeqCst), solved, "no check after a refusal");
    assert_eq!(s.stray_held_out(), 3);
    s.species_dirty = true;
    assert_eq!(s.refit().unwrap().n_samples, 40, "never fitted");
    assert_eq!(s.stray_held_out(), 3);
}

/// Plan Task 3 step 1: two stray H at automatic multiplicity (1, but the parts' spins aligned are 3) are refused by
/// the floor: the step falls back, the answer is not kept (`StrayNotKept`) and no check is queued.
#[test]
fn floor_refused_samples_are_never_kept() {
    let (mut s, _, calls) = student(0.0, &TRAINED, 40, true);
    let sc = h2_with(CHECKS[0], &[STRAY_AT, [-5.5, 3.0, 5.0]]);
    let (split, _, floor) = s.stray_split(&positions(&sc), &zs(&sc), &s.provenance_for(&sc)).expect("a split");
    assert_eq!(split.stray, vec![2, 3], "the premise: two stray atoms");
    assert!(floor.as_deref().is_some_and(|w| w.starts_with("spin: 2 stray H, H at automatic")), "{floor:?}");
    let n = s.samples();
    let r = s.evaluate(&sc, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback);
    let ev = s.take_events();
    assert_eq!(ev.last().map(|e| (e.rule, e.outcome)), Some((TeachRule::IsolatedAtom, TeachOutcome::StrayNotKept)), "{ev:?}");
    assert!(r.stats.notes.iter().any(|n| n.contains("sample not kept (stray atoms — spin: 2 stray H, H at automatic")), "{:?}", r.stats.notes);
    assert_eq!(s.samples(), n, "not kept");
    assert!(s.stray_pending.is_none() && s.side_queue.is_empty());
    settle(&mut s);
    assert_eq!(calls.load(SeqCst), 0, "no check");
    // the keep path refuses it too (a caller that keeps directly: the teach-around pass)
    let mut notes = Vec::new();
    let (pos, z) = (positions(&sc), zs(&sc));
    let (e, f) = SpeciesMorse::energy_and_forces(&pos, &z);
    let prov = s.provenance_for(&sc);
    assert!(!s.keep_sample(&pos, &z, e, &f, Some(0.0), Some(1), &prov, SampleTag::Targeted, &mut notes));
    assert_eq!(s.samples(), n);
}

/// Review Focus 4: the scene is edited while a check is in flight — its answer is dropped silently (no verdict, no
/// event, the sample still held), and a fallback of the edited scene checks afresh; a check of a scene edited since
/// is replaced by the next fallback's, and the old one's late part is told apart by its job id.
#[test]
fn a_check_answer_after_an_edit_is_dropped() {
    let (mut s, gate, calls) = student(0.0, &TRAINED, 40, true);
    let first = h2_with(CHECKS[0], &[STRAY_AT]);
    let sig = sig_of(&s, &first);
    gate.store(false, SeqCst);
    assert_eq!(s.evaluate(&first, &mut never()).unwrap().stats.source, Source::Fallback);
    let i = s.samples() - 1;
    assert!(s.stray_pending.is_some());
    // the edit (a teleport keeps the warm start: `discontinuity`), then the late answers
    s.discontinuity();
    gate.store(true, SeqCst);
    settle(&mut s);
    assert!(s.stray_book.state(&sig).is_none(), "no verdict: {:?}", s.stray_book.state(&sig));
    assert!(s.take_events().iter().all(|e| e.origin != TeachOrigin::Audit), "no check event");
    assert!(calls.load(SeqCst) <= 2);
    assert_eq!(s.quarantine[i], Q_STRAY, "still held");
    // the edited scene's fallback checks afresh
    assert_eq!(s.evaluate(&first, &mut never()).unwrap().stats.source, Source::Fallback);
    settle(&mut s);
    assert!(matches!(s.stray_book.state(&sig), Some(StrayState::Checking { passes: 1, .. })));
    // an edit with a check in flight, then a fallback before its answer: the new check replaces the old
    gate.store(false, SeqCst);
    assert_eq!(s.evaluate(&h2_with(CHECKS[1], &[STRAY_AT]), &mut never()).unwrap().stats.source, Source::Fallback);
    let old = s.stray_pending.as_ref().map(|p| p.id).expect("in flight");
    s.reset();
    let second = h2_with(CHECKS[2], &[STRAY_AT]);
    assert_eq!(s.evaluate(&second, &mut never()).unwrap().stats.source, Source::Fallback);
    let j = s.samples() - 1;
    let new = s.stray_pending.as_ref().map(|p| p.id).expect("replaced");
    assert!(new > old);
    assert!(s.side_queue.iter().all(|q| q.3 == new), "the old check's queued part is dropped");
    gate.store(true, SeqCst);
    settle(&mut s);
    assert!(matches!(s.stray_book.state(&sig), Some(StrayState::Checking { passes: 2, .. })), "{:?}", s.stray_book.state(&sig));
    assert_eq!(s.quarantine[j], Q_NONE, "the new check's own sample");
    assert_eq!(s.quarantine[j - 1], Q_STRAY, "the replaced check's sample stays held");
}

/// Plan Global Constraints, controller ruling C: without a background auditor (the CPU teacher's synchronous audits)
/// no check runs — the step falls back as before, with the reason in its notes — and the sample is still held out.
#[test]
fn no_auditor_no_check_but_held_out() {
    let (mut s, _, _) = student(0.0, &TRAINED, 40, false);
    let r = s.evaluate(&h2_with(CHECKS[0], &[STRAY_AT]), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback);
    assert!(r.stats.notes.iter().any(|n| n.contains("no background teacher, so no composition check runs")), "{:?}", r.stats.notes);
    assert!(s.stray_pending.is_none() && s.side_queue.is_empty());
    let i = s.samples() - 1;
    assert_eq!((s.quarantine[i], s.stray_held_out()), (Q_STRAY, 1));
    // Task 6: the note once per signature, not at every step; the held sample's event says it is held, not "kept"
    let ev = s.take_events();
    assert_eq!(ev.last().map(|e| e.outcome), Some(TeachOutcome::KeptStrayHeld), "{ev:?}");
    let again = s.evaluate(&h2_with(CHECKS[1], &[STRAY_AT]), &mut never()).unwrap();
    assert_eq!(again.stats.source, Source::Fallback);
    assert!(!again.stats.notes.iter().any(|n| n.contains("no background teacher")), "{:?}", again.stats.notes);
    s.species_dirty = true;
    assert_eq!(s.refit().unwrap().n_samples, 40, "held out of the fit");
}

/// Controller ruling B: the book lives in one model's descriptor space — a refit with the same basis keeps it, one with
/// another basis (a new species) clears it, releasing nothing and holding nothing anew; a Forget and an installed model
/// clear it too, and an installed file's samples with stray atoms are held out again (its fit rebuilt without them).
#[test]
fn the_book_is_cleared_with_the_models_basis() {
    let (mut s, _, _) = student(0.0, &TRAINED, 40, true);
    let sig = sig_of(&s, &h2_with(CHECKS[0], &[STRAY_AT]));
    for &r in &CHECKS {
        s.evaluate(&h2_with(r, &[STRAY_AT]), &mut never()).unwrap();
        settle(&mut s);
    }
    assert!(matches!(s.stray_book.state(&sig), Some(StrayState::Cleared { .. })));
    // the same basis: kept
    s.refit().unwrap();
    assert!(!s.stray_book_current() && s.stray_book.state(&sig).is_some(), "a refit with the same basis keeps the book");
    // the file of this model, for the install below (its stray samples were released, code 0)
    let path = std::env::temp_dir().join(format!("stray-book-{}.model", std::process::id()));
    s.save_sidecar(&path).unwrap();
    // final review minor 2 (ruling 13): a refusal is a fact of the teacher, not of the basis — it survives the basis
    // change below (only the regions, in the old basis's feature space, are dropped)
    let mut other = sig.clone();
    other.stray_z = vec![1, 1];
    s.stray_book.refuse(&other, "test".into());
    // another basis: a He sample makes the next refit's species H, He
    s.set_reference_energy(2, SpeciesMorse::atom(2));
    let he = {
        let mut sc = big(dimer(2.6, [1.0, 0.0, 0.0], H2_AT, 1, 2));
        sc.nuclei[1].z = 2;
        sc
    };
    s.add_sample(exact_sample(&he), &he).unwrap();
    s.refit().unwrap();
    assert_eq!(s.model().unwrap().basis.species, vec![1, 2]);
    let codes = s.quarantine.clone();
    assert!(s.stray_book_current(), "another basis clears the book");
    assert!(s.stray_book.state(&sig).is_none());
    assert!(matches!(s.stray_book.state(&other), Some(StrayState::Refused(w)) if w == "test"), "a refusal survives a basis change: {:?}", s.stray_book.state(&other));
    assert_eq!(s.quarantine, codes, "nothing released or held anew");
    // a Forget clears it
    s.stray_book.refuse(&sig, "test".into());
    s.forget();
    assert!(s.stray_book.state(&sig).is_none());
    // an installed model clears it, and holds its stray samples out again
    let (mut t, _, _) = student(0.0, &TRAINED, 40, true);
    t.stray_book.refuse(&sig, "test".into());
    t.install_sidecar(surrogate::sidecar::read(&path).unwrap()).unwrap();
    let _ = std::fs::remove_file(&path);
    assert!(t.stray_book.state(&sig).is_none(), "an installed model clears the book");
    assert_eq!(t.stray_held_out(), 3, "its samples with stray atoms are held out again: {:?}", t.user_notes);
    assert_eq!(t.report().unwrap().n_samples, 40, "and its fit rebuilt without them");
    assert!(t.user_notes.iter().any(|n| n.contains("its fit used 3 of the samples with stray atoms — rebuilt")), "{:?}", t.user_notes);
}

/// One collection sample of H₂ (at H–H `r`) and a stray H 8 Bohr away, in the 81³ box, as an older build kept it.
fn stray_record(r: f64) -> surrogate::CollectionSample {
    let sc = h2_with(r, &[STRAY_AT]);
    let mut sample = exact_sample(&sc);
    sample.tag = SampleTag::Fallback;
    surrogate::CollectionSample { sample, grid_n: 81, multiplicity_forced: false, multiplicity: Some(2), scene: "H3".into() }
}

/// Review Focus 5: a collection holding samples with stray atoms (appended by an older build) attaches with them held
/// out (`Q_STRAY`) and says so once; the fit and a rebuild never see them; nothing is written for them — `samples.bin`
/// hashes identical before and after, and the collection's codes are the stored ones (a forgotten one keeps its own).
#[test]
fn loaded_stray_samples_are_held_out_not_rewritten() {
    let dir = collection_root("stray-attach");
    let scene = h2(1.4);
    let key = {
        let (mut s, _) = pair_student();
        s.attach_collection(&dir, &scene).unwrap();
        let mut rng = StdRng::seed_from_u64(31);
        s.set_reference_solved(1, SpeciesMorse::atom(1), 33, Some(0.0));
        for i in 0..40 {
            s.evaluate(&h2_at(1.1 + 1.2 * i as f64 / 39.0, &mut rng), &mut never()).unwrap();
        }
        s.refit().unwrap();
        assert!(s.trusted(), "the premise: {:?}", s.untrusted_reason());
        s.collection().unwrap().key.clone()
    };
    {
        let (mut c, _) = surrogate::Collection::open(&dir, &key).unwrap();
        assert_eq!(c.samples.len(), 40);
        for r in [1.3, 1.5, 1.7, 1.9] {
            c.append(stray_record(r)).unwrap();
        }
        c.set_status(41, Q_FORGOTTEN).unwrap();
    }
    let bin = dir.join(key.slug()).join("samples.bin");
    let hash = |b: &[u8]| b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x0000_0100_0000_01b3));
    let stored = std::fs::read(&bin).unwrap();
    let before = hash(&stored);
    let mut want = vec![Q_NONE; 44];
    want[41] = Q_FORGOTTEN;

    let (mut s, _) = pair_student();
    s.attach_collection(&dir, &scene).unwrap();
    assert_eq!((s.samples(), s.stray_held_out()), (44, 3));
    assert!([40, 42, 43].iter().all(|&i| s.quarantine[i] == Q_STRAY && s.excluded(i)), "{:?}", &s.quarantine[40..]);
    assert_eq!(s.quarantine[41], Q_FORGOTTEN, "an excluded sample keeps its own code");
    let note = format!("shared collection {}: 3 samples with stray atoms (no neighbour within 6 Bohr) held out of the fit until a composition check measures them (kept on disk, unchanged)", key.slug());
    assert_eq!(s.user_notes.iter().filter(|n| **n == note).count(), 1, "said once: {:?}", s.user_notes);
    assert_eq!(s.report().expect("fitted at attach").n_samples, 40, "the stray samples are not fitted");
    s.species_dirty = true;
    assert_eq!(s.refit().unwrap().n_samples, 40);
    assert!([40, 42, 43].iter().all(|&i| s.quarantine[i] == Q_STRAY));
    // never written: the collection's codes stand, the file is byte-identical after a flush
    assert_eq!(s.collection().unwrap().quarantine, want);
    s.flush_collection();
    drop(s);
    let after = std::fs::read(&bin).unwrap();
    assert_eq!((hash(&after), after.len()), (before, stored.len()), "samples.bin unchanged: no record for a stray sample");
    let (again, _) = surrogate::Collection::open(&dir, &key).unwrap();
    assert_eq!(again.quarantine, want);
}

// ---- plan Task 3 fix round 1 ----

/// Clears `sig`'s region with three passes at [`CHECKS`] (the pure book, at the student's model and spreads) — the
/// state three distinct composition checks leave.
fn clear_by_hand(s: &mut SurrogateKs, sig: &Signature) {
    s.stray_book_current();
    let basis = s.model().unwrap().basis.clone();
    let units = s.envelope().unwrap().clone();
    let thin = THIN_FRACTION * s.policy().max_descriptor_distance;
    for &r in &CHECKS {
        let (d, sp) = h2_desc(&basis, r);
        let sc = h2_with(r, &[STRAY_AT]);
        let place = StrayPlace::of(&positions(&sc), &Split { bonded: vec![0, 1], stray: vec![2] }, sc.cell.half_extent());
        s.stray_book.record(sig.clone(), CheckVerdict::Pass { residual: 0.0, margin: 0.32 }, &d, &sp, place, &basis, &units, thin);
    }
    assert!(matches!(s.stray_book.state(sig), Some(StrayState::Cleared { .. })));
}

/// Review Important 1 (controller ruling 5): the sampling pass's keep path (`add_pass_sample`) follows the same rules —
/// a pass sample of H₂ + a stray H is held out (`Q_STRAY`) and not fitted; one with two stray H at automatic (the floor's
/// spin rule) is not kept; an H₂-only one is kept in the fit as before.
#[test]
fn pass_samples_with_stray_atoms_are_held_or_not_kept() {
    let (mut s, _, _) = student(0.0, &TRAINED, 40, false);
    let n = s.samples();
    let one = h2_with(CHECKS[0], &[STRAY_AT]);
    assert_eq!(s.add_pass_sample(exact_sample(&one), &one, Some(0.0), Some(2)), Ok(PassKeep::Kept { held: true }));
    assert_eq!((s.samples(), s.quarantine[n], s.stray_held_out()), (n + 1, Q_STRAY, 1));
    let two = h2_with(CHECKS[0], &[STRAY_AT, [-5.5, 3.0, 5.0]]);
    match s.add_pass_sample(exact_sample(&two), &two, Some(0.0), Some(1)) {
        Ok(PassKeep::StrayNotKept(why)) => assert!(why.starts_with("spin: 2 stray H, H at automatic"), "{why}"),
        other => panic!("not kept: {other:?}"),
    }
    assert_eq!(s.samples(), n + 1, "the floor-refused sample is not kept");
    let plain = big(dimer(1.2, [1.0, 0.0, 0.0], H2_AT, 1, 1));
    assert_eq!(s.add_pass_sample(exact_sample(&plain), &plain, Some(0.0), Some(1)), Ok(PassKeep::Kept { held: false }));
    assert!(!s.excluded(n + 1));
    s.species_dirty = true;
    assert_eq!(s.refit().unwrap().n_samples, 41, "the held one is not fitted, the H₂-only one is");
    // under a cleared region covering it, a pass sample is kept in the fit
    let sig = sig_of(&s, &one);
    clear_by_hand(&mut s, &sig);
    let again = h2_with(CHECKS[1], &[STRAY_AT]);
    assert_eq!(s.add_pass_sample(exact_sample(&again), &again, Some(0.0), Some(2)), Ok(PassKeep::Kept { held: false }));
}

/// Review M2: a draft guess's audit with stray atoms is held as a stray sample (`Q_STRAY`, its tag saying it is a
/// draft guess), so `judge_held_out` — which promotes `Q_DRAFT` alone — never takes an unmeasured one into the fit; a
/// clearance covering it releases it to the draft hold-out `Q_DRAFT`, not straight into the fit.
#[test]
fn a_stray_draft_guess_is_released_to_the_draft_hold_out() {
    let (mut s, _, _) = student(0.0, &TRAINED, 40, true);
    let sc = h2_with(CHECKS[1], &[STRAY_AT]);
    let (pos, z) = (positions(&sc), zs(&sc));
    let (e, f) = SpeciesMorse::energy_and_forces(&pos, &z);
    let prov = s.provenance_for(&sc);
    let mut notes = Vec::new();
    assert!(s.keep_sample(&pos, &z, e, &f, Some(0.0), Some(2), &prov, SampleTag::DraftAudit, &mut notes));
    let i = s.samples() - 1;
    assert_eq!((s.quarantine[i], s.store().unwrap().samples[i].tag), (Q_STRAY, SampleTag::DraftAudit), "{notes:?}");
    assert_eq!(s.judge_held_out(), 0);
    assert_eq!(s.quarantine[i], Q_STRAY, "never promoted while unmeasured");
    let sig = sig_of(&s, &sc);
    clear_by_hand(&mut s, &sig);
    assert_eq!(s.release_stray_samples(&sig), 1);
    assert_eq!(s.quarantine[i], Q_DRAFT, "released to the draft hold-out");
}

/// Review M3: a sample loaded (here: re-held at an install of samples into a live provider) while its signature is
/// already cleared and its region covers it stays in the fit; one outside the region is held.
#[test]
fn a_hold_out_after_a_clearance_spares_the_covered_samples() {
    let (mut s, _, _) = student(0.0, &TRAINED, 40, true);
    let inside = h2_with(1.5, &[STRAY_AT]);
    let outside = h2_with(1.9, &[STRAY_AT]);
    let sig = sig_of(&s, &inside);
    clear_by_hand(&mut s, &sig);
    // as a restore brings them in: stored at code 0, then the hold-out
    s.add_sample(exact_sample(&inside), &inside).unwrap();
    s.add_sample(exact_sample(&outside), &outside).unwrap();
    let (a, b) = (s.samples() - 2, s.samples() - 1);
    assert_eq!((s.quarantine.get(a).copied().unwrap_or(Q_NONE), s.quarantine.get(b).copied().unwrap_or(Q_NONE)), (Q_NONE, Q_NONE));
    let r_cut = s.cutoff();
    assert_eq!(s.hold_out_stray(r_cut), 1);
    assert_eq!((s.quarantine[a], s.quarantine[b]), (Q_NONE, Q_STRAY), "the covered one stays in the fit");
}

/// Review M4: an atom answer that matches none of the check's stray atoms drops the check with a note (never a check
/// wedged in flight, blocking every later one), and the next fallback checks again.
#[test]
fn an_unmatched_atom_answer_drops_the_check() {
    let (mut s, gate, _) = student(0.0, &TRAINED, 40, true);
    let sc = h2_with(CHECKS[0], &[STRAY_AT]);
    let sig = sig_of(&s, &sc);
    gate.store(false, SeqCst);
    s.evaluate(&sc, &mut never()).unwrap();
    let id = s.stray_pending.as_ref().map(|p| p.id).expect("in flight");
    let mut stray = sc.clone();
    stray.nuclei = vec![Nucleus::new(1, [0.0, 4.0, 0.0])];
    let prov = s.provenance_for(&sc);
    let mut ans = audit_answer(&s, &stray, prov, Ok((SpeciesMorse::atom(1), vec![[0.0; 3]], Some(0.0), None)));
    (ans.job.kind, ans.job.id) = (JobKind::AtomAt(1), id);
    s.take_stray_answer(ans);
    assert!(s.stray_pending.is_none() && !s.side_queue.iter().any(|j| matches!(j.2, JobKind::AtomAt(_))), "dropped");
    assert!(s.pending_notes.iter().any(|n| n.contains("matched none of its stray atoms")), "{:?}", s.pending_notes);
    gate.store(true, SeqCst);
    settle(&mut s);
    assert!(s.stray_book.state(&sig).is_none(), "no verdict from the dropped check");
    s.discontinuity();
    s.evaluate(&sc, &mut never()).unwrap();
    settle(&mut s);
    assert!(matches!(s.stray_book.state(&sig), Some(StrayState::Checking { passes: 1, .. })), "the next fallback checks again");
}

/// Review M10: with the stray H first in the scene (index 0; the bonded part 1 and 2), the scene's forces and the
/// fragment's are paired by the split, not by position in the list: the check passes and releases its sample.
#[test]
fn a_stray_atom_first_in_the_scene_is_checked() {
    let (mut s, _, _) = student(0.0, &TRAINED, 40, true);
    let mut sc = h2_with(CHECKS[0], &[]);
    sc.nuclei.insert(0, Nucleus::new(1, STRAY_AT));
    let (split, sig, floor) = s.stray_split(&positions(&sc), &zs(&sc), &s.provenance_for(&sc)).unwrap();
    assert_eq!((split.bonded.clone(), split.stray.clone(), floor), (vec![1, 2], vec![0], None));
    // a scene whose forces are not zero on the bonded atoms (H–H 1.35 is compressed), so a mispairing would show
    let (_, f) = SpeciesMorse::energy_and_forces(&positions(&sc), &zs(&sc));
    assert!(f[1][0].abs() > 1e-3 && f[0] == [0.0; 3], "{f:?}");
    s.evaluate(&sc, &mut never()).unwrap();
    let i = s.samples() - 1;
    settle(&mut s);
    let ev = s.take_events();
    assert!(ev.iter().any(|e| e.outcome == TeachOutcome::StrayChecking { n: 1, of: 3 }), "{ev:?}");
    assert!(matches!(s.stray_book.state(&sig), Some(StrayState::Checking { passes: 1, .. })));
    assert_eq!(s.quarantine[i], Q_NONE);
}

/// Review M10, collection mode with a nonzero box offset: the scene's answer enters the check raw (never the canonical
/// energy, which differs here by 3 × 1 mHa), so the residual stays ≈ 0 and the check passes; the held sample waits
/// outside the collection (`Link::Waiting`) and is appended once released; and a held sample of a box without its
/// reference is released over the reference wait (`Q_REF_WAIT`), not into the fit.
#[test]
fn in_collection_mode_a_check_reads_raw_energies_and_a_release_appends() {
    use super::super::shared::Link;
    let dir = collection_root("stray-collection");
    let template = h2_with(1.1, &[]);
    let mut s = SurrogateKs::new(Box::new(Composing::new(0.0)), FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() }).with_teacher_tol(1e-6);
    s.attach_collection(&dir, &template).unwrap();
    // the 81³ box's reference, and a larger box's 1 mHa lower: the canonical one, so the 81³ box's offset is nonzero
    s.set_reference_solved(1, SpeciesMorse::atom(1), 81, Some(0.0));
    s.set_reference_solved(1, SpeciesMorse::atom(1) - 1e-3, 97, Some(0.0));
    let mut rng = StdRng::seed_from_u64(0x57b);
    for i in 0..40 {
        let r = TRAINED[0].0 + (TRAINED[0].1 - TRAINED[0].0) * i as f64 / 39.0;
        let origin: [f64; 3] = std::array::from_fn(|k| H2_AT[k] + rng.gen_range(-0.3..0.3));
        s.evaluate(&big(dimer(r, unit_vector(&mut rng), origin, 1, 1)), &mut never()).unwrap();
    }
    s.refit().unwrap();
    assert!(s.trusted(), "the premise: {:?}", s.untrusted_reason());
    let mut s = s.with_auditor(Box::new(Composing::new(0.0)));
    let sc = h2_with(CHECKS[0], &[STRAY_AT]);
    let (pos, z) = (positions(&sc), zs(&sc));
    let raw = SpeciesMorse::energy_and_forces(&pos, &z).0;
    let offset = raw - s.to_canonical(&z, 81, raw);
    assert!((offset - 3e-3).abs() < 1e-9, "the premise: a 3 mHa box offset ({offset:e})");
    let in_coll = s.collection().unwrap().samples.len();
    s.evaluate(&sc, &mut never()).unwrap();
    let i = s.samples() - 1;
    assert_eq!(s.quarantine[i], Q_STRAY);
    assert!(matches!(s.shared.as_ref().unwrap().link[i], Link::Waiting { .. }), "it waits outside the collection");
    assert_eq!(s.collection().unwrap().samples.len(), in_coll);
    settle(&mut s);
    let ev = s.take_events();
    let check = ev.iter().find(|e| e.origin == TeachOrigin::Audit && e.rule == TeachRule::IsolatedAtom).expect("a check event");
    assert_eq!(check.outcome, TeachOutcome::StrayChecking { n: 1, of: 3 }, "{}", check.detail);
    let sig = sig_of(&s, &sc);
    let Some(StrayState::Checking { residual, .. }) = s.stray_book.state(&sig) else { panic!("{:?}", s.stray_book.state(&sig)) };
    assert!(*residual < 1e-9, "raw energies throughout: residual {residual:e}");
    assert_eq!(s.quarantine[i], Q_NONE, "released");
    assert!(matches!(s.shared.as_ref().unwrap().link[i], Link::In(k) if k == in_coll), "appended once released");
    let stored = &s.collection().unwrap().samples[in_coll];
    assert_eq!(stored.sample.energy.to_bits(), raw.to_bits(), "the collection holds the teacher's raw answer");
    // a box without its reference (85³): kept held over the reference wait, released into the wait
    let mut far = h2_with(CHECKS[1], &[STRAY_AT]);
    far.cell = Cell { n: 85, h: 0.25 };
    let (pos, z) = (positions(&far), zs(&far));
    let (e, f) = SpeciesMorse::energy_and_forces(&pos, &z);
    let prov = s.provenance_for(&far);
    let mut notes = Vec::new();
    assert!(s.keep_sample(&pos, &z, e, &f, Some(0.0), Some(2), &prov, SampleTag::Fallback, &mut notes), "{notes:?}");
    let j = s.samples() - 1;
    assert_eq!(s.quarantine[j], Q_STRAY, "{notes:?}");
    let sig85 = sig_of(&s, &far);
    assert_ne!(sig85, sig, "another box: another signature");
    clear_by_hand(&mut s, &sig85);
    assert_eq!(s.release_stray_samples(&sig85), 1);
    assert_eq!(s.quarantine[j], Q_REF_WAIT, "released into the reference wait, not the fit");
}


// ---- plan Task 4: answering a cleared, in-region scene ----

/// The answering tests' geometries (H–H, Bohr): the region three checks at [`CHECKS`] open (ruling A's fixture,
/// [`TRAINED`]) covers 1.4 ([`INSIDE`], never checked) and not 1.9 ([`OUTSIDE`]); both asserted in
/// [`cleared_student`]. The region is a per-feature box of the checks' descriptors (`Region::contains`), so it covers a
/// geometry between checks only where every feature is monotone between them — true at this spacing, not at 0.3 Bohr
/// (measured: checks at 1.2 / 1.5 / 1.8 cover none of 1.25–1.75 but themselves).
const INSIDE: f64 = 1.4;
const OUTSIDE: f64 = 1.9;

/// Whether `sig`'s region's per-feature box contains the bonded H₂ at H–H `r` (ruling 12 adds each check's thinning
/// radius to the region: `Region::covers`).
fn covers(s: &SurrogateKs, sig: &Signature, r: f64) -> bool {
    let sc = h2_with(r, &[STRAY_AT]);
    let (d, sp) = s.bonded_descriptors(&positions(&sc), &zs(&sc), &Split { bonded: vec![0, 1], stray: vec![2] }).unwrap();
    matches!(s.stray_book.state(sig), Some(StrayState::Cleared { region, .. }) if region.contains(&d, &sp))
}

/// A trusted H₂ student ([`TRAINED`]) whose H₂ + H signature three real composition checks at [`CHECKS`] (asserted
/// distinct) cleared, refitted with the samples the checks released (so the bonded part at [`INSIDE`] lies inside the
/// training data: it must pass rules 5b–5e on its own), its events drained; the book survives the refit (same basis),
/// and its region covers [`INSIDE`] and not [`OUTSIDE`] (asserted).
fn cleared_student() -> (SurrogateKs, Signature) {
    cleared_student_of(&Composing::new(0.0))
}

/// [`cleared_student`] of the teacher `c` (its checks at [`STRAY_AT`] must pass: no offset there, not box-affected).
fn cleared_student_of(c: &Composing) -> (SurrogateKs, Signature) {
    let (mut s, _, _) = student_of(c, &TRAINED, 40, true);
    assert_distinct(&mut s, &CHECKS);
    let sig = sig_of(&s, &h2_with(CHECKS[0], &[STRAY_AT]));
    for &r in &CHECKS {
        let res = s.evaluate(&h2_with(r, &[STRAY_AT]), &mut never()).unwrap();
        assert_eq!(res.stats.source, Source::Fallback, "a check geometry before clearance: {:?}", res.stats.notes);
        settle(&mut s);
    }
    assert!(matches!(s.stray_book.state(&sig), Some(StrayState::Cleared { .. })), "{:?}", s.stray_book.state(&sig));
    assert_eq!(s.refit().unwrap().n_samples, 43, "the checks' samples are released into the fit");
    assert!(!s.stray_book_current() && s.trusted(), "the same basis keeps the book; trusted: {:?}", s.untrusted_reason());
    assert!(covers(&s, &sig, INSIDE) && !covers(&s, &sig, OUTSIDE), "the premise: the region covers {INSIDE}, not {OUTSIDE}");
    s.take_events();
    (s, sig)
}

/// The step note of a cleared answer (spec §4.4, review M6), with the book's numbers for `sig`.
fn cleared_note(s: &SurrogateKs, sig: &Signature) -> String {
    let Some(StrayState::Cleared { residual, margin, passes, .. }) = s.stray_book.state(sig) else { panic!("not cleared") };
    format!(
        "stray atoms: model for the bonded part + free atoms (composition measured, {passes} checks; worst residual {:.2} mHa, frontier margin {:.0} mHa)",
        residual * 1e3,
        margin * 1e3
    )
}

/// One strict step at `sc`, asserting first that the read-only verdicts agree with it (plan Task 4,
/// `would_answer_agrees_with_evaluate`): `would_answer`, `atom_verdicts` and the readiness readout's `first_refusal`
/// answer exactly when `evaluate` does, and a refusal names the rule and atom the step's Activity row names.
fn judged_step(s: &mut SurrogateKs, sc: &Scene) -> ForceResult {
    let (pos, z) = (positions(sc), zs(sc));
    let prov = s.provenance_for(sc);
    let would = s.would_answer(&pos, &z, &prov);
    let atoms = s.atom_verdicts(sc).expect("atoms");
    let first = s.first_refusal(&pos, &z, &prov).map(|r| (r.rule, r.atom));
    s.take_events();
    let r = s.evaluate(sc, &mut never()).unwrap();
    let answered = r.stats.source == Source::Surrogate;
    assert_eq!(would.is_ok(), answered, "would_answer {would:?} against the step's {:?}: {:?}", r.stats.source, r.stats.notes);
    assert_eq!(atoms.answers(), answered, "atom_verdicts {atoms:?} against the step's {:?}", r.stats.source);
    assert_eq!(first, would.err(), "the readiness readout's refusal is would_answer's");
    if let Err(code) = would {
        let ev = s.take_events();
        let row = ev.iter().find(|e| e.origin == TeachOrigin::Fallback).expect("a fallback row");
        assert_eq!((row.rule, row.atom), code, "{}", row.detail);
    }
    r
}

fn ulp(x: f64) -> f64 {
    let a = x.abs();
    f64::from_bits(a.to_bits() + 1) - a
}

/// Plan Task 4, review M1: after three checks clear H₂ + H, a fourth bonded geometry inside the region (never checked)
/// is the model's (`Source::Surrogate`). Its answer is the model's own `predict(scene)`, which is spec §2's sum: the
/// energy within 4 ulp of `predict(H₂ alone) + E_ref(H)`, the bonded forces bit for bit the bonded part's, the stray
/// force exactly 0; the note says what answered and on what measurement. The step queues no check.
#[test]
fn a_cleared_scene_is_the_models() {
    let (mut s, sig) = cleared_student();
    let sc = h2_with(INSIDE, &[STRAY_AT]);
    let r = judged_step(&mut s, &sc);
    assert_eq!(r.stats.source, Source::Surrogate, "{:?}", r.stats.notes);
    let m = s.model().unwrap();
    let frag = big(dimer(INSIDE, [1.0, 0.0, 0.0], H2_AT, 1, 1));
    assert_eq!(positions(&frag), positions(&sc)[..2].to_vec(), "the premise: the bonded part alone");
    let pf = m.predict(&positions(&frag), &[1, 1], false).unwrap();
    let e_ref = m.ref_energy[m.basis.species_index(1).unwrap()];
    let want = pf.energy + e_ref;
    println!("E_scene {:.15} vs E_frag + E_ref {want:.15}: {:.1} ulp", r.energy, (r.energy - want).abs() / ulp(want));
    assert!((r.energy - want).abs() <= 4.0 * ulp(want), "E {} vs {want}", r.energy);
    for k in 0..2 {
        assert_eq!(r.forces[k].map(f64::to_bits), pf.forces[k].map(f64::to_bits), "bonded atom {k}: bit for bit");
    }
    assert_eq!(r.forces[2], [0.0; 3], "the stray atom's force is exactly 0");
    assert!(pf.forces[0][0].abs() > 1e-3, "the premise: nonzero bonded forces ({:?})", pf.forces);
    assert!(r.stats.notes.contains(&cleared_note(&s, &sig)), "{:?}", r.stats.notes);
    assert!(s.stray_pending.is_none() && s.side_queue.is_empty(), "no check inside the region");
}

/// Plan Task 4: H–H 1.9 Bohr lies outside the cleared region — rule 5a refuses (the stray atom), the teacher answers,
/// and the fallback is one more check, whose pass grows the region (`StrayCheckingNew`); the next step there is the
/// model's.
#[test]
fn outside_the_region_falls_back_and_checks() {
    let (mut s, sig) = cleared_student();
    // in the region first (the model's), then out of it: the exit switch is booked (fix round 1 M3)
    let r = judged_step(&mut s, &h2_with(INSIDE, &[STRAY_AT]));
    assert_eq!(r.stats.source, Source::Surrogate, "{:?}", r.stats.notes);
    let sc = h2_with(OUTSIDE, &[STRAY_AT]);
    let r = judged_step(&mut s, &sc);
    assert_eq!(r.stats.source, Source::Fallback);
    assert!(r.stats.refit_jump.is_some() && s.last.as_ref().is_some_and(|l| !l.on_model), "the switch out to the teacher's surface is booked");
    assert!(r.stats.notes.iter().any(|n| n.contains("isolated atom — atom 2")), "{:?}", r.stats.notes);
    assert!(r.stats.notes.iter().any(|n| n.contains("stray atoms (H2 + H) — a composition check") && n.contains("queued")), "{:?}", r.stats.notes);
    settle(&mut s);
    let ev = s.take_events();
    assert!(ev.iter().any(|e| e.outcome == TeachOutcome::StrayCheckingNew), "{ev:?}");
    assert!(covers(&s, &sig, OUTSIDE), "the region grows");
    // with its released sample fitted, the next step there is the model's
    s.refit().unwrap();
    let r = judged_step(&mut s, &sc);
    assert_eq!(r.stats.source, Source::Surrogate, "{:?}", r.stats.notes);
    assert!(r.stats.refit_jump.is_some(), "the switch back to the model's surface is booked");
}

/// Plan Task 4, review I4: rule 7 normalises per **bonded** atom. With the energy tier at 0.5 mHa per atom (and the
/// force tier at 1e-2 Ha/Bohr, so rule 6 does not decide first), the model's posterior factor is scaled so that at the
/// cleared geometry the epistemic energy std √(e² − (2 s_E)²) is 1.25 mHa: 0.42 mHa per atom over the scene's 3 atoms
/// (it would pass), 0.62 mHa over the bonded part's 2 (it fails). The step falls back on rule 7, and the read-only
/// verdicts agree; rule 6 passes and the stray atom's force std is exactly 0 (asserted).
///
/// Task 5 fix round 1 (controller ruling 9a): `Model::predict`'s floor is (N_e s_E)², N_e the energy-row atoms, so
/// the scene's energy std is its bonded part's **bit for bit**, and the general rule 7 (per N_e) is Task 4's
/// per-bonded-atom special case exactly — which is therefore gone. The verdict no longer depends on the exemption: near
/// a face (rule 5a refuses, the stray atom not exempt) the variance refuses on rule 7 too.
#[test]
fn rule7_uses_n_bonded() {
    let (mut s, sig) = cleared_student();
    let p = *s.policy();
    s.set_policy(FallbackPolicy { energy_std_per_atom: 5e-4, force_std_abs: 1e-2, ..p });
    // a policy change drops the region (final review minor 5): its three checks booked again under the new tiers
    clear_by_hand(&mut s, &sig);
    assert!(s.trusted(), "the premise: {:?}", s.untrusted_reason());
    let sc = h2_with(INSIDE, &[STRAY_AT]);
    let (pos, z) = (positions(&sc), zs(&sc));
    let epistemic = |s: &SurrogateKs| {
        let m = s.model().unwrap();
        let p = m.predict(&pos, &z, true).unwrap();
        (p.energy_std.unwrap().powi(2) - (2.0 * m.sigma_e).powi(2)).sqrt()
    };
    let c = 1.25e-3 / epistemic(&s);
    let model = s.model.as_mut().unwrap();
    for x in model.sigma_inv_chol.as_mut().expect("a posterior") {
        *x /= c;
    }
    s.last_variance = None;
    assert!((epistemic(&s) - 1.25e-3).abs() < 1e-7, "{:e}", epistemic(&s));
    let m = s.model().unwrap().clone();
    let pred = m.predict(&pos, &z, true).unwrap();
    let (e, f) = (pred.energy_std.unwrap(), pred.force_std_epistemic.clone().unwrap());
    let infl = s.atom_inflation(&m, &pos, &z);
    assert!(infl.iter().all(|k| k.sum == 1.0 && k.tail == 1.0 && k.energy == 1.0), "the premise: no class inflation");
    // bit for bit: the scene's energy std is the bonded part's alone (the floor (N_e s_E)², N_e = 2; B_s adds zeros)
    let frag = m.predict(&pos[..2], &z[..2], true).unwrap();
    assert_eq!(e.to_bits(), frag.energy_std.unwrap().to_bits(), "the energy std of H₂ + H is H₂'s");
    let tier = 5e-4;
    let per_scene_atom = (e * e - (3.0 * m.sigma_e).powi(2)).max(0.0).sqrt() / 3.0;
    assert!(per_scene_atom <= tier, "the premise: per scene atom it would pass ({per_scene_atom:e})");
    assert!(s.energy_variance(&m, &pos, e, &infl).1.is_some(), "per energy-row atom it fails");
    assert_eq!(
        format!("{:?}", s.energy_variance(&m, &pos, e, &infl).1),
        format!("{:?}", s.energy_variance(&m, &pos[..2], frag.energy_std.unwrap(), &infl[..2]).1),
        "rule 7 at the scene is rule 7 on its bonded part"
    );
    let v = s.variance_check(&m, &pred, e, &f, &pos, &z);
    assert!(!v.accept && v.atoms[2] == 0.0, "rule 6 passes (the stray atom's std is 0), rule 7 refuses: {:?}", v.atoms);
    assert_eq!(s.would_answer(&pos, &z, &s.provenance_for(&sc)), Err((TeachRule::EnergyUncertainty, None)));
    let r = judged_step(&mut s, &sc);
    assert_eq!(r.stats.source, Source::Fallback);
    assert!(r.stats.notes.iter().any(|n| n.contains("epistemic energy std 6.25e-4 > 5.00e-4 Ha/atom")), "{:?}", r.stats.notes);
    // the variance cadence: a verdict reached while the stray atom was not exempt (here near a face: rule 5a refuses)
    // is the same rule 7 per energy-row atom, and the next, cleared step is judged afresh (the marker)
    let near_face = h2_with(INSIDE, &[[0.0, 6.5, 0.0]]);
    assert_eq!(judged_step(&mut s, &near_face).stats.source, Source::Fallback);
    assert!(s.last_variance.as_ref().is_some_and(|v| !v.accept), "rule 7 per energy-row atom refuses there too");
    // (back where the checks measured it: 5 Bohr from the face, 8.05 from the nearer H₂ atom — ruling 13's place)
    let back = h2_with(INSIDE, &[[0.0, 5.0, 0.5]]);
    let r = judged_step(&mut s, &back);
    assert!(r.stats.notes.iter().any(|n| n.contains("epistemic energy std 6.25e-4")), "judged afresh per bonded atom: {:?}", r.stats.notes);
}

/// The teacher, shifted by `shift` Ha everywhere after the model was fitted: every model ↔ teacher switch then moves
/// the reported energy by about `shift`, so an unbooked switch shows in the energy ledger.
struct Shifted {
    inner: Composing,
    shift: f64,
}

impl ForceProvider for Shifted {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        let mut r = self.inner.evaluate(scene, should_abort)?;
        r.energy += self.shift;
        r.stats.energy.total = r.energy;
        Ok(r)
    }

    fn name(&self) -> &'static str {
        // the model's teacher by name: the same surface (the provenance names it)
        "composing morse"
    }

    fn reset(&mut self) {}
}

/// Review Focus 1, spec §4.4 re-entry (review M3): the stray H on a path 8 → 5 → 8 Bohr from H₂'s centre in 1/8-Bohr
/// steps, the teacher shifted 10 mHa off the model. At ≥ 6 Bohr (stray, cleared) the model answers; inside r_cut the
/// ordinary rules refuse — rule 5b at the first step in (spec M3: the returning atom's coordination is far below any
/// training atom's; the fitted stray samples' stray atoms are no training atoms, fix round 1 C1) — and the teacher
/// answers on its own surface. Every model ↔ teacher switch carries a booking
/// (`refit_jump`, from `transition_jump`), and E_conserved has no unbooked step: per step, ΔE_reported − J +
/// ½ (F₀ + F₁)·Δr — what the driver's E_conserved moves by beyond the bounded shadow term — stays at the trapezoid's
/// error, far below the 10 mHa an unbooked switch would leave. No sample flood: the way back out keeps no copy of the
/// way in's fallbacks.
#[test]
fn reentry_and_exit_book_every_switch() {
    let (mut s, sig) = cleared_student();
    // no calibration ticks (trust tightening Task 7; as `tests.rs`'s excursion and session-golden tests): a tick falls
    // due at the 25th in-envelope step, on the way out, and its audit's teacher answer is kept like any audit's — one
    // sample the count below would read as a flood. The run measures the stray atom's re-entry and its bookings; the
    // ticks are pinned by their own tests
    s.tick_every = 0;
    // since ruling 13 the region reaches the stray atom no nearer than its passes: a check at the path's nearest stray
    // point (6 Bohr from H₂'s centre, 6.04 from its atoms) extends it there before the run
    let edge = h2_with(INSIDE, &[[H2_AT[0], H2_AT[1] + R_CUT, H2_AT[2]]]);
    let r = judged_step(&mut s, &edge);
    assert!(r.stats.source == Source::Fallback && queued(&r), "{:?}", r.stats.notes);
    settle(&mut s);
    assert!(matches!(s.stray_book.state(&sig), Some(StrayState::Cleared { region, .. }) if (region.place().reach - (0.49f64 + 36.0).sqrt()).abs() < 1e-9));
    s.refit().unwrap();
    s.teacher = Box::new(Shifted { inner: Composing::new(0.0), shift: 1e-2 });
    let mut path: Vec<f64> = (0..=24).map(|k| 8.0 - 0.125 * k as f64).collect();
    path.extend((1..=24).map(|k| 5.0 + 0.125 * k as f64));
    let n0 = s.samples();
    let mut at_turn = n0;
    let mut prev: Option<(f64, Vec<[f64; 3]>, Vec<[f64; 3]>, bool)> = None;
    let (mut switches, mut rules) = (0, Vec::new());
    for (k, &d) in path.iter().enumerate() {
        if k == 25 {
            at_turn = s.samples();
        }
        let stray = [H2_AT[0], H2_AT[1] + d, H2_AT[2]];
        let sc = h2_with(INSIDE, &[stray]);
        if d < R_CUT {
            rules.push(s.would_answer(&positions(&sc), &zs(&sc), &s.provenance_for(&sc)).err().map(|e| e.0));
        }
        let r = judged_step(&mut s, &sc);
        let on_model = s.last.as_ref().unwrap().on_model;
        assert_eq!(r.stats.source == Source::Surrogate, d >= R_CUT, "d = {d}: {:?}", r.stats.notes);
        assert_eq!(on_model, d >= R_CUT, "d = {d}: the re-entry steps are reported on the teacher's surface");
        let pos = positions(&sc);
        if let Some((e0, p0, f0, m0)) = prev.take() {
            let j = r.stats.refit_jump;
            if on_model != m0 {
                switches += 1;
                assert!(j.is_some(), "d = {d}: an unbooked switch ({} → {})", if m0 { "model" } else { "teacher" }, if on_model { "model" } else { "teacher" });
            }
            let mut work = 0.0;
            for ((a, b), (fa, fb)) in p0.iter().zip(&pos).zip(f0.iter().zip(&r.forces)) {
                for k in 0..3 {
                    work += 0.5 * (fa[k] + fb[k]) * (b[k] - a[k]);
                }
            }
            let unbooked = r.energy - e0 - j.unwrap_or(0.0) + work;
            assert!(unbooked.abs() < 1e-5, "d = {d}: E_conserved moves by {unbooked:.3e} Ha (jump {j:?})");
        }
        prev = Some((r.energy, pos, r.forces.clone(), on_model));
    }
    println!("re-entry refusals: {rules:?}; {} samples kept on the way in, {} on the way out", at_turn - n0, s.samples() - at_turn);
    assert_eq!(switches, 2, "in once, out once");
    assert!(rules.iter().all(|r| r.is_some_and(|r| r != TeachRule::IsolatedAtom)), "inside r_cut the ordinary rules refuse: {rules:?}");
    assert_eq!(rules[0], Some(TeachRule::Coordination), "rule 5b refuses at the first step inside r_cut (spec M3): {rules:?}");
    assert!(s.store().unwrap().samples[..n0].iter().any(|x| x.force_mask.is_some()), "the premise: released stray samples are fitted");
    assert_eq!(s.samples(), at_turn, "no sample flood: the way out revisits the way in's geometries, and keeps none of them");
}

/// Review Focus 3: a stray atom drifting toward a face during a cleared run falls back as soon as it is within the
/// floor's 4 Bohr of a face — the floor is re-checked at every step, the reason shown in the step's notes — and the
/// model answers again once it is back. (Since ruling 13 the region itself reaches no nearer a face than its passes:
/// a check at 4.1 Bohr from the face extends it there first.)
#[test]
fn a_stray_near_a_face_falls_back_inside_a_cleared_region() {
    let (mut s, sig) = cleared_student();
    // the 81³ cell's half-extent is 10 Bohr: at y = 5.9 the stray H is 4.1 Bohr from a face, at 6.5 3.5 Bohr
    let r = judged_step(&mut s, &h2_with(INSIDE, &[[0.0, 5.9, 0.0]]));
    assert!(r.stats.source == Source::Fallback && queued(&r), "nearer a face than the checks: a check first: {:?}", r.stats.notes);
    settle(&mut s);
    assert!(matches!(s.stray_book.state(&sig), Some(StrayState::Cleared { region, .. }) if (region.place().face - 4.1).abs() < 1e-9));
    // every step is a switch of surface (the last step was a fallback), each booked (fix round 1 M3)
    for (y, model) in [(5.9, true), (6.5, false), (5.9, true)] {
        let sc = h2_with(INSIDE, &[[0.0, y, 0.0]]);
        let r = judged_step(&mut s, &sc);
        assert_eq!(r.stats.source == Source::Surrogate, model, "y = {y}: {:?}", r.stats.notes);
        assert_eq!(s.last.as_ref().map(|l| l.on_model), Some(model), "y = {y}: the fallback is reported on the teacher's surface");
        assert!(r.stats.refit_jump.is_some(), "y = {y}: the switch is booked");
        if !model {
            assert!(r.stats.notes.iter().any(|n| n.contains("box: the stray H is 3.50 Bohr from a face")), "{:?}", r.stats.notes);
            assert!(s.stray_pending.is_none(), "no check of a floor-refused scene");
        }
    }
}

/// Plan Task 4: the read-only verdicts agree with `evaluate` on the answering tests' scenes (each step of those tests
/// goes through [`judged_step`]); here they are walked once more side by side — cleared, outside the region, a stray
/// near a face, a stray back inside r_cut, cleared again — and two more facts: a stray species missing from the basis
/// has no exemption (rule 2), and two stray H are another signature (rule 5a's).
#[test]
fn would_answer_agrees_with_evaluate() {
    let (mut s, _) = cleared_student();
    for (sc, model) in [
        (h2_with(INSIDE, &[STRAY_AT]), true),
        (h2_with(OUTSIDE, &[STRAY_AT]), false),
        (h2_with(INSIDE, &[[0.0, 6.5, 0.0]]), false),
        (h2_with(INSIDE, &[[0.0, 2.0, 0.0]]), false),
        (h2_with(INSIDE, &[STRAY_AT]), true),
    ] {
        assert_eq!(judged_step(&mut s, &sc).stats.source == Source::Surrogate, model);
        settle(&mut s);
    }
    // a He stray atom: not in the basis (rule 2), no exemption
    let mut he = h2_with(INSIDE, &[STRAY_AT]);
    he.nuclei[2].z = 2;
    let prov = s.provenance_for(&he);
    assert!(s.stray_cleared(&positions(&he), &zs(&he), &prov).is_none());
    assert!(matches!(s.would_answer(&positions(&he), &zs(&he), &prov), Err((TeachRule::NewSpecies(2), _))));
    // two stray H: another signature (and the floor's spin rule), rule 5a's
    let two = h2_with(INSIDE, &[STRAY_AT, [-5.5, 3.0, 5.0]]);
    assert_eq!(judged_step(&mut s, &two).stats.source, Source::Fallback);
}

/// Task 3 review M2, kept with the exemption: `judge_held_out` reads `geometry_refusal`, which exempts stray atoms
/// only inside a cleared region of the signature (`stray_cleared`). A draft guess with a stray atom inside the region
/// is kept at the draft hold-out and promoted; one outside stays held (`Q_STRAY`), and even forced to `Q_DRAFT` it is
/// never promoted — rule 5a refuses it there.
#[test]
fn a_stray_draft_guess_is_promoted_only_inside_a_cleared_region() {
    let (mut s, _) = cleared_student();
    let keep = |s: &mut SurrogateKs, r: f64| {
        let sc = h2_with(r, &[STRAY_AT]);
        let (pos, z) = (positions(&sc), zs(&sc));
        let (e, f) = SpeciesMorse::energy_and_forces(&pos, &z);
        let prov = s.provenance_for(&sc);
        let mut notes = Vec::new();
        assert!(s.keep_sample(&pos, &z, e, &f, Some(0.0), Some(2), &prov, SampleTag::DraftAudit, &mut notes), "{notes:?}");
        (s.samples() - 1, pos, z, prov)
    };
    let (a, pos_a, z_a, prov_a) = keep(&mut s, INSIDE);
    let (b, pos_b, z_b, prov_b) = keep(&mut s, OUTSIDE);
    let (c, _, _, _) = keep(&mut s, OUTSIDE + 0.05);
    assert_eq!((s.quarantine[a], s.quarantine[b], s.quarantine[c]), (Q_DRAFT, Q_STRAY, Q_STRAY));
    assert!(s.geometry_refusal(&pos_a, &z_a, &prov_a).is_none(), "inside the region: strict mode's domain");
    assert_eq!(s.geometry_refusal(&pos_b, &z_b, &prov_b).map(|r| r.rule), Some(TeachRule::IsolatedAtom));
    s.quarantine[c] = Q_DRAFT;
    assert_eq!(s.judge_held_out(), 1);
    assert_eq!((s.quarantine[a], s.quarantine[b], s.quarantine[c]), (Q_NONE, Q_STRAY, Q_DRAFT), "only the covered guess is promoted");
}


/// Ruling (the fences judge the bonded part alone, spec §4.4 "the bonded part must pass every rule on its own"): a
/// forgotten H₂ sample at the cleared geometry fences H₂ there, and so it fences H₂ + a stray H. Judged on the whole
/// scene the quarantine fence (the worst atom's distance to the fenced geometry) would let the stray H — no atom of
/// the fenced H₂ is its counterpart — carry the geometry outside (asserted: the premise).
#[test]
fn a_fence_on_the_bonded_part_fences_the_cleared_scene() {
    let (mut s, _) = cleared_student();
    let sc = h2_with(INSIDE, &[STRAY_AT]);
    let h2 = big(dimer(INSIDE, [1.0, 0.0, 0.0], H2_AT, 1, 1));
    s.add_sample(exact_sample(&h2), &h2).unwrap();
    let i = s.samples() - 1;
    s.quarantine.resize(s.samples(), Q_NONE);
    s.quarantine[i] = Q_FORGOTTEN;
    assert!(!s.near_quarantined_cached(&positions(&h2), &zs(&h2)).is_empty(), "the premise: H₂ alone is fenced");
    assert!(s.near_quarantined_cached(&positions(&sc), &zs(&sc)).is_empty(), "the premise: the whole scene escapes it");
    let r = judged_step(&mut s, &sc);
    assert_eq!(r.stats.source, Source::Fallback, "{:?}", r.stats.notes);
    assert_eq!(s.would_answer(&positions(&sc), &zs(&sc), &s.provenance_for(&sc)), Err((TeachRule::NearQuarantine, None)));
}

// ---- plan Task 4 fix round 1 ----

/// The envelope's bits (min, max, Σ, Σ², counts).
fn envelope_bits(e: &Envelope) -> Vec<u64> {
    [&e.min, &e.max, &e.sum, &e.sumsq].iter().flat_map(|v| v.iter().map(|x| x.to_bits())).chain(e.count.iter().map(|&c| c as u64)).collect()
}

/// The strict verdicts (`would_answer`'s rule, or none) on a stretched, non-stray H₂ scan r ∈ [4.5, 6) Bohr.
fn stretched_verdicts(s: &SurrogateKs) -> Vec<(f64, Option<TeachRule>)> {
    (0..15)
        .map(|k| {
            let r = 4.5 + 0.1 * k as f64;
            let sc = big(dimer(r, [1.0, 0.0, 0.0], H2_AT, 1, 1));
            (r, s.would_answer(&positions(&sc), &zs(&sc), &s.provenance_for(&sc)).err().map(|e| e.0))
        })
        .collect()
}

/// Fix round 1 C1 (controller ruling 7) and (g): a fitted sample with a stray atom leaves the training envelope bit
/// for bit what the same sample without its stray atom leaves — the stray H (B = 0) is no training atom — so the
/// coordination minimum stays the bonded atoms' and rule 5b judges a stretched, non-stray H₂ near r_cut exactly as
/// before: the verdicts of a scan r ∈ [4.5, 6) Bohr are unchanged by fitting it (admitted, the stray H would set the
/// minimum to 0, 5b would skip the channel and the model would answer the stretched pair by its free-atom limit,
/// the M6 Task 11 failure).
#[test]
fn a_fitted_stray_atom_leaves_the_envelope_and_the_stretched_pair_alone() {
    let (mut base, _, _) = student(0.0, &TRAINED, 40, false);
    let before = stretched_verdicts(&base);
    println!("stretched H₂ verdicts before: {before:?}");
    assert!(before.iter().all(|(_, v)| v.is_some()), "the premise: strict mode refuses the stretched pair");
    let (mut plain, _, _) = student(0.0, &TRAINED, 40, false);
    let bonded = big(dimer(CHECKS[0], [1.0, 0.0, 0.0], H2_AT, 1, 1));
    plain.add_sample(exact_sample(&bonded), &bonded).unwrap();
    let with = h2_with(CHECKS[0], &[STRAY_AT]);
    base.add_sample(exact_sample(&with), &with).unwrap();
    let i = base.samples() - 1;
    assert!(!base.excluded(i), "the premise: the stray sample is fitted");
    base.species_dirty = true;
    plain.species_dirty = true;
    assert_eq!(base.refit().unwrap().n_samples, 41);
    assert_eq!(plain.refit().unwrap().n_samples, 41);
    assert_eq!(envelope_bits(base.envelope().unwrap()), envelope_bits(plain.envelope().unwrap()), "the stray H adds nothing to the envelope");
    let k = base.model().unwrap().basis.body2_index(0, 0);
    assert!(base.envelope().unwrap().min[k] > 0.5, "the coordination minimum stays the bonded atoms' ({:e})", base.envelope().unwrap().min[k]);
    assert_eq!(stretched_verdicts(&base), before, "the stretched pair's verdicts are unchanged by the fitted stray sample");
}

/// Fix round 1 I2 (controller ruling 8): a sample with stray atoms is stored with them masked out of its force rows
/// (`force_mask`; its energy row kept), so a released one adds 3 × n_bonded force rows, not 3 × n — no zero rows
/// enter the force count, Σ F² or the residual degrees of freedom (they would bias σ_f down). A loaded one is masked
/// in memory at its hold-out; a capped segment's mask is narrowed, never widened.
#[test]
fn a_released_stray_sample_adds_no_stray_force_rows() {
    let (s, _) = cleared_student();
    let st = s.store().unwrap();
    let rows: usize = (0..s.samples()).filter(|&i| !s.excluded(i)).map(|i| 3 * st.samples[i].trained_atoms()).sum();
    let acc = s.accumulator().unwrap();
    assert_eq!(acc.n_samples, 43);
    assert_eq!((acc.n_f, rows), (3 * 2 * 43, 3 * 2 * 43), "40 H₂ samples and 3 released H₂ + H: 2 taught atoms each");
    for i in 40..43 {
        let smp = &st.samples[i];
        assert_eq!((smp.pos.len(), smp.force_mask.clone(), smp.use_energy), (3, Some(vec![true, true, false]), true), "sample {i}");
    }
    // the helper alone: a segment's mask narrowed, a sample without stray atoms untouched
    let mut seg = exact_sample(&h2_with(CHECKS[0], &[STRAY_AT]));
    seg.force_mask = Some(vec![false, true, true]);
    seg.use_energy = false;
    mask_stray_atoms(&mut seg, R_CUT);
    assert_eq!((seg.force_mask, seg.use_energy), (Some(vec![false, true, false]), false));
    let mut plain = exact_sample(&big(dimer(1.2, [1.0, 0.0, 0.0], H2_AT, 1, 1)));
    mask_stray_atoms(&mut plain, R_CUT);
    assert_eq!(plain.force_mask, None);
}

/// Fix round 1 C1(d): a model file written by an older build, whose fit admitted its released stray samples' stray
/// atoms into the envelope (simulated: an isolated H added to the stored envelope), installs with those samples held
/// out (`Q_STRAY`, the session's clearances start empty) — and every sample with an isolated atom is a lone or stray
/// sample, so the fit is rebuilt from the stored samples without them: the installed envelope holds no isolated atom.
#[test]
fn an_installed_file_with_isolated_atoms_in_its_envelope_is_rebuilt() {
    let (s, _) = cleared_student();
    let path = std::env::temp_dir().join(format!("stray-envelope-{}.model", std::process::id()));
    s.save_sidecar(&path).unwrap();
    let mut sc = surrogate::sidecar::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let p = sc.accumulator.basis.n_features();
    sc.accumulator.envelope.add_atom(0, &vec![0.0; p]);
    let k = sc.accumulator.basis.body2_index(0, 0);
    assert_eq!(sc.accumulator.envelope.min[k], 0.0, "the premise: the stored envelope admits an isolated atom");
    let (mut t, _, _) = student(0.0, &TRAINED, 40, false);
    t.install_sidecar(sc).unwrap();
    assert_eq!(t.stray_held_out(), 3);
    assert!(t.user_notes.iter().any(|n| n.contains("rebuilt from its stored samples without them")), "{:?}", t.user_notes);
    assert!(t.envelope().unwrap().min[k] > 0.5, "the rebuilt envelope holds no isolated atom ({:e})", t.envelope().unwrap().min[k]);
}

/// Fix round 1 M6: the per-step cost of the cleared decision on a stray scene (the split, the basis comparison, the
/// surface key, the book, the floor and the bonded part's descriptors), against a strict model step there. Reported,
/// not judged (a timing): printed for the task report.
#[test]
fn diag_stray_cleared_cost() {
    let (mut s, _) = cleared_student();
    let sc = h2_with(INSIDE, &[STRAY_AT]);
    let (pos, z, prov) = (positions(&sc), zs(&sc), s.provenance_for(&sc));
    let n = 20_000;
    let t0 = Instant::now();
    for _ in 0..n {
        std::hint::black_box(s.stray_cleared(&pos, &z, &prov));
    }
    let alone = t0.elapsed().as_secs_f64() / n as f64;
    // as `evaluate` asks: with the scene's descriptors it computes for the envelope anyway
    let desc = describe_values(&s.model().unwrap().basis, &pos, &z).unwrap();
    let t0 = Instant::now();
    for _ in 0..n {
        std::hint::black_box(s.stray_cleared_with(&pos, &z, &prov, Some(&desc)));
    }
    let cleared = t0.elapsed().as_secs_f64() / n as f64;
    let t0 = Instant::now();
    for _ in 0..n {
        let m = s.model().unwrap();
        std::hint::black_box((m.basis == s.stray_key.as_ref().unwrap().0, super::super::lone::reference_settings_key(&m.provenance)));
    }
    let key = t0.elapsed().as_secs_f64() / n as f64;
    let steps = 2_000;
    let t0 = Instant::now();
    for _ in 0..steps {
        assert_eq!(s.evaluate(&sc, &mut never()).unwrap().stats.source, Source::Surrogate);
    }
    let step = t0.elapsed().as_secs_f64() / steps as f64;
    println!(
        "stray_cleared {:.2} µs alone, {:.2} µs with the scene's descriptors (basis == + surface key {:.2} µs); a strict model step {:.2} µs: {:.1} %",
        alone * 1e6,
        cleared * 1e6,
        key * 1e6,
        step * 1e6,
        100.0 * cleared / step
    );
}


// ---- plan Task 5: audits of cleared answers and their attribution (spec §4.5, review I6) ----

/// Plan Task 5, A: the per-atom audit count is n_bonded at a scene with stray atoms and exactly the scene's n
/// otherwise — H₂, a three-atom chain, a single atom, two lone atoms (rule 0b's: no bonded part) — so a scene without a
/// stray atom is judged bit for bit as before (every site divides by `audit_n`, which returns `pos.len()` there).
#[test]
fn audit_n_is_the_scene_n_without_stray_atoms() {
    let (s, _) = cleared_student();
    let mut chain = big(dimer(1.4, [1.0, 0.0, 0.0], H2_AT, 1, 1));
    chain.nuclei.push(Nucleus::new(1, [2.2, -3.0, 0.0]));
    let mut one = h2_with(1.4, &[]);
    one.nuclei = vec![Nucleus::new(1, H2_AT)];
    let mut lone = h2_with(1.4, &[]);
    lone.nuclei = vec![Nucleus::new(1, H2_AT), Nucleus::new(1, STRAY_AT)];
    for sc in [big(dimer(1.4, [1.0, 0.0, 0.0], H2_AT, 1, 1)), chain, one, lone] {
        let pos = positions(&sc);
        assert!(split(&pos, s.cutoff()).is_none(), "the premise: no stray atom in {pos:?}");
        assert_eq!(s.audit_n(&pos), pos.len(), "{pos:?}");
        assert_eq!(surrogate::energy_row_atoms(&Basis::new(&[1]), &pos), pos.len());
    }
    assert_eq!(s.audit_n(&positions(&h2_with(INSIDE, &[STRAY_AT]))), 2, "H₂ + a stray H: per bonded atom");
    assert_eq!(s.audit_n(&positions(&h2_with(INSIDE, &[STRAY_AT, [-5.5, 3.0, 5.0]]))), 2);
    assert_eq!(s.audit_n(&[]), 1, "never below 1");
}

/// The composite offset of plan Task 5's audits (Ha): within the energy tier per scene atom over H₂ + H's 3 atoms
/// (0.83 mHa), beyond it per bonded atom over 2 (1.25 mHa) — so an audit that fails at it fails only per bonded atom
/// (asserted in [`audited_student`]).
const AUDIT_OFFSET: f64 = 2.5e-3;

/// The cleared H₂ + H student of [`cleared_student`] with `auditor` as its background teacher and every answered step
/// audited (`audit_every` 1): one model answer at [`INSIDE`], its audit judged (the answer is held until the next
/// poll, as until the next step's evaluate). The student, its signature and the store index the audit's answer was
/// kept at.
fn audited_student(auditor: Box<dyn ForceProvider>) -> (SurrogateKs, Signature, usize) {
    audited_student_gated(auditor, None)
}

/// [`audited_student`] with the auditor's `gate` closed once the audit has answered and before it is judged: the
/// attribution check's parts then wait on the auditor (Task 5 fix round 1: what happens while it is in flight).
fn audited_student_gated(auditor: Box<dyn ForceProvider>, gate: Option<&Arc<AtomicBool>>) -> (SurrogateKs, Signature, usize) {
    let (s, sig) = cleared_student();
    let tier = s.policy().energy_std_per_atom;
    assert!(AUDIT_OFFSET / 3.0 <= tier && AUDIT_OFFSET / 2.0 > tier, "the premise: the offset fails per bonded atom only");
    let mut s = s.with_auditor(auditor);
    let p = *s.policy();
    s.set_policy(FallbackPolicy { audit_every: 1, ..p });
    assert!(s.trusted(), "the premise: {:?}", s.untrusted_reason());
    let n = s.samples();
    let r = s.evaluate(&h2_with(INSIDE, &[STRAY_AT]), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Surrogate, "{:?}", r.stats.notes);
    assert!(s.audit_in_flight(), "the premise: the answer is audited on the background teacher");
    settle(&mut s);
    s.take_events();
    if let Some(g) = gate {
        g.store(false, SeqCst);
    }
    s.poll_audit();
    assert_eq!(s.samples(), n + 1, "the premise: the audit's answer is kept");
    (s, sig, n)
}

/// A teacher whose every answer with a bonded pair (some atom has a neighbour within r_cut) is `shift` Ha off the
/// Morse surface: the bonded part is perturbed alone — the composition stays exact (the scene and the fragment move
/// together, a lone atom does not) and the model, fitted before, is wrong on the fragment.
struct BondedShift {
    inner: Composing,
    shift: f64,
}

impl ForceProvider for BondedShift {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        let mut r = self.inner.evaluate(scene, should_abort)?;
        if !no_neighbours(&positions(scene), Basis::new(&[1]).r_cut) {
            r.energy += self.shift;
            r.stats.energy.total = r.energy;
        }
        Ok(r)
    }

    fn name(&self) -> &'static str {
        "composing morse"
    }

    fn reset(&mut self) {}
}

/// Plan Task 5 step 1 (spec §4.5, review I6): after clearance, the background teacher shows a composite offset the
/// parts alone never show ([`AUDIT_OFFSET`] whenever a stray atom is present). The audit of a cleared answer fails
/// **per bonded atom** (it would pass per scene atom), and in this order: the signature is refused at once, with the
/// audit's numbers; its samples (the three the checks released and the audit's own) go back to `Q_STRAY`; and an
/// attribution check is queued at the audited geometry with the audit's raw answer as its scene. The model's trust is
/// untouched meanwhile and after: on the bonded part alone the model is right, so the failure is the composition's
/// (`StrayAuditAttributed { model: false }`), nothing enters the audit record, and the next step there falls back on
/// rule 5a with no new check.
#[test]
fn a_failed_stray_audit_refuses_and_blames_by_a_check() {
    let (mut s, sig, audited) = audited_student(Box::new(Composing::new(AUDIT_OFFSET)));
    let history = s.audit_history().len();
    // 1. refused, held, the refit armed
    assert!(
        matches!(s.stray_book.state(&sig), Some(StrayState::Refused(why)) if why.starts_with("audit: teacher − model +2.50e-3 Ha, 1.2e-3 > 1.0e-3 Ha per bonded atom")),
        "{:?}",
        s.stray_book.state(&sig)
    );
    assert_eq!((s.quarantine[audited], s.stray_held_out()), (Q_STRAY, 4), "the released samples and the audit's own are held: {:?}", &s.quarantine[40..]);
    assert!(s.pending_refit && s.species_dirty);
    // 2. the attribution check at the audited geometry, the model untouched meanwhile
    let p = s.stray_pending.as_ref().expect("an attribution check in flight");
    assert!(p.attribution.is_some() && p.pos == positions(&h2_with(INSIDE, &[STRAY_AT])));
    assert!((p.scene.energy - SpeciesMorse::energy_and_forces(&p.pos, &p.z).0 - AUDIT_OFFSET).abs() < 1e-12, "the audit's own raw answer is the check's scene");
    assert!(s.trusted() && s.failed_audits.is_empty() && s.audit_failed().is_none(), "{:?}", s.untrusted_reason());
    assert!(!s.audit_flip, "no flip to book: the model was not untrusted");
    // 3. attributed: the composition's
    settle(&mut s);
    let ev = s.take_events();
    let att: Vec<&sim::TeachEvent> = ev.iter().filter(|e| matches!(e.outcome, TeachOutcome::StrayAuditAttributed { .. })).collect();
    assert_eq!(
        att.iter().map(|e| (e.origin, e.rule, e.outcome)).collect::<Vec<_>>(),
        [(TeachOrigin::Audit, TeachRule::Audit, TeachOutcome::StrayAuditAttributed { model: false })],
        "{ev:?}"
    );
    assert!(att[0].detail.contains("is not the model's (mostly the composition)") && att[0].detail.contains("the composition E_scene − E_frag − Σ E_atom +2.500 mHa"), "{}", att[0].detail);
    // ruling 9b: no history point for a failure that is not the model's, a `composition` count instead
    assert_eq!((s.audit_history().len(), s.audit_history().unrecorded.composition), (history, 1), "no point, one count");
    assert!(att[0].gap.is_some_and(|g| g.abs() < 1e-6), "the model on the bonded part alone: {:?}", att[0].gap);
    assert!(s.trusted() && s.failed_audits.is_empty() && s.audit_failed().is_none(), "the model untouched: {:?}", s.untrusted_reason());
    assert!(matches!(s.stray_book.state(&sig), Some(StrayState::Refused(_))), "the attribution books nothing");
    // the next step there is DFT's and checks nothing (refused)
    let r = judged_step(&mut s, &h2_with(INSIDE, &[STRAY_AT]));
    assert_eq!(r.stats.source, Source::Fallback);
    assert!(r.stats.notes.iter().any(|n| n.contains("isolated atom — atom 2")), "{:?}", r.stats.notes);
    settle(&mut s);
    assert!(s.stray_pending.is_none());
    assert_eq!(s.quarantine[s.samples() - 1], Q_STRAY);
    assert_eq!(s.refit().unwrap().n_samples, 40, "the signature's samples are out of the fit for good");
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
}

/// Plan Task 5 step 1: the background teacher's bonded part is perturbed ([`BondedShift`], [`AUDIT_OFFSET`] on every
/// answer with a bonded pair): the composition is exact, the model is wrong on the fragment. The audit of the cleared
/// answer fails per bonded atom; the attribution check finds the model off on the bonded part alone by the same gap,
/// so the failure enters the model's audit record (`StrayAuditAttributed { model: true }`: untrusted) — recorded at
/// the scene's geometry with the label the model failed to meet there (the fragment's energy + E_ref(H), the
/// fragment's forces, 0 on the stray atom). A refit re-judges it per bonded atom: still failed (per scene atom it
/// would pass, 0.83 mHa), so the model stays untrusted; the signature is refused either way.
#[test]
fn a_failed_audit_with_a_wrong_fragment_counts_against_the_model() {
    let (mut s, sig, _) = audited_student(Box::new(BondedShift { inner: Composing::new(0.0), shift: AUDIT_OFFSET }));
    assert!(matches!(s.stray_book.state(&sig), Some(StrayState::Refused(_))));
    assert!(s.trusted(), "not yet attributed");
    settle(&mut s);
    let ev = s.take_events();
    let att = ev.iter().find(|e| matches!(e.outcome, TeachOutcome::StrayAuditAttributed { .. })).expect("an attribution event");
    assert_eq!(att.outcome, TeachOutcome::StrayAuditAttributed { model: true }, "{}", att.detail);
    assert!(att.detail.contains("is the model's") && att.detail.contains("the model on the bonded part +2.500 mHa, the composition E_scene − E_frag − Σ E_atom +0.000 mHa"), "{}", att.detail);
    assert_eq!(s.failed_audits.len(), 1);
    let case = s.failed_audits[0].clone();
    assert_eq!(case.origin, AuditOrigin::StrayAttributed, "the synthetic label is marked (M7)");
    // ruling 9b: the history's point is the fragment's — its 2 atoms, the error on the bonded part
    let p = s.audit_history().iter().last().expect("a point").clone();
    assert_eq!((p.n_atoms, p.kind), (2, surrogate::history::PointKind::Audit));
    assert!((p.err_energy - AUDIT_OFFSET).abs() < 1e-6, "err_energy = gap_frag: {:e}", p.err_energy);
    assert_eq!(s.audit_history().unrecorded.composition, 0);
    let sc = h2_with(INSIDE, &[STRAY_AT]);
    let pos = positions(&sc);
    let (fe, ff) = SpeciesMorse::energy_and_forces(&pos[..2], &[1, 1]);
    let m = s.model().unwrap();
    let e_ref = m.ref_energy[m.basis.species_index(1).unwrap()];
    assert_eq!((case.pos.clone(), case.index), (pos.clone(), None));
    assert!((case.energy - (fe + AUDIT_OFFSET + e_ref)).abs() < 1e-12, "the fragment's label + E_ref: {} vs {}", case.energy, fe + AUDIT_OFFSET + e_ref);
    assert_eq!((case.forces[0], case.forces[1], case.forces[2]), (ff[0], ff[1], [0.0; 3]));
    assert!(!s.trusted() && s.audit_failed().is_some_and(|g| (g - AUDIT_OFFSET).abs() < 1e-6), "{:?} {:?}", s.untrusted_reason(), s.audit_failed());
    assert!(s.audit_flip, "the flip is booked at the next evaluation (the last step was the model's)");
    assert_eq!(s.audit_atoms, 2, "the checklist's per-atom numbers per bonded atom");
    // re-judged at the refit, per bonded atom: still failed
    s.refit().unwrap();
    assert_eq!(s.failed_audits.len(), 1, "{:?}", s.user_notes);
    assert!(!s.trusted(), "per bonded atom the gap is beyond the tier: {:?}", s.untrusted_reason());
    assert!(s.user_notes.iter().any(|n| n.contains("still untrusted")), "{:?}", s.user_notes);
}

/// The conservative side (module doc, "Audits of a cleared answer"): without a background teacher no attribution check
/// can run — the in-step audit of a cleared answer fails per bonded atom (the teacher's composite offset), the
/// signature is refused and its samples held, and the failure counts against the model at once with the scene's own
/// labels, as any failed audit.
#[test]
fn an_unattributable_stray_audit_counts_against_the_model() {
    let (mut s, sig) = cleared_student();
    s.auditor = None;
    s.teacher = Box::new(Composing::new(AUDIT_OFFSET));
    let p = *s.policy();
    s.set_policy(FallbackPolicy { audit_every: 1, ..p });
    assert!(s.trusted());
    let sc = h2_with(INSIDE, &[STRAY_AT]);
    let r = s.evaluate(&sc, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback, "the in-step audit's answer");
    assert!(r.stats.notes.iter().any(|n| n.contains("no check can attribute the failure (no background teacher), so it counts against the model")), "{:?}", r.stats.notes);
    assert!(matches!(s.stray_book.state(&sig), Some(StrayState::Refused(_))));
    assert_eq!(s.stray_held_out(), 4);
    assert_eq!(s.failed_audits.len(), 1);
    let (e, _) = SpeciesMorse::energy_and_forces(&positions(&sc), &zs(&sc));
    assert!((s.failed_audits[0].energy - (e + AUDIT_OFFSET)).abs() < 1e-12, "the scene's own label");
    assert!(!s.trusted() && s.audit_atoms == 2, "{:?}", s.untrusted_reason());
}

/// Plan Task 5, A (`teach`'s over-tier report): a fallback at a cleared scene whose gap is beyond the energy tier per
/// bonded atom but within it per scene atom is reported on the teacher's surface. The fixture of `rule7_uses_n_bonded`
/// (tier 0.5 mHa; rule 7 refuses per bonded atom — a fallback its rule does not report on the teacher's surface), the
/// teacher shifted 1.25 mHa: 0.42 mHa per scene atom, 0.62 mHa per bonded atom.
#[test]
fn over_tier_uses_n_bonded() {
    let (mut s, sig) = cleared_student();
    let p = *s.policy();
    s.set_policy(FallbackPolicy { energy_std_per_atom: 5e-4, force_std_abs: 1e-2, ..p });
    // a policy change drops the region (final review minor 5): its three checks booked again under the new tiers
    clear_by_hand(&mut s, &sig);
    let sc = h2_with(INSIDE, &[STRAY_AT]);
    let (pos, z) = (positions(&sc), zs(&sc));
    let m = s.model().unwrap();
    let pr = m.predict(&pos, &z, true).unwrap();
    let c = 1.25e-3 / (pr.energy_std.unwrap().powi(2) - (2.0 * m.sigma_e).powi(2)).sqrt();
    for x in s.model.as_mut().unwrap().sigma_inv_chol.as_mut().expect("a posterior") {
        *x /= c;
    }
    s.last_variance = None;
    let shift = 1.25e-3;
    assert!(shift / 3.0 <= 5e-4 && shift / 2.0 > 5e-4, "the premise");
    s.teacher = Box::new(Shifted { inner: Composing::new(0.0), shift });
    let r = s.evaluate(&sc, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback);
    assert!(r.stats.notes.iter().any(|n| n.contains("epistemic energy std")), "rule 7: {:?}", r.stats.notes);
    assert!(r.stats.notes.iter().any(|n| n.contains("reported on the teacher's surface")), "over the tier per bonded atom: {:?}", r.stats.notes);
    assert!(s.last.as_ref().is_some_and(|l| !l.on_model));
}

/// Plan Task 5, B (Task 4 review): the stray force mask is never written. A released stray sample appended to a
/// collection is recorded without it (its energy row and all three atoms' forces as the teacher solved them), while
/// the store's copy is masked; a fresh attach reads the record unmasked and masks its store copy in memory. A model
/// file likewise: written unmasked, masked again at the install.
#[test]
fn a_stray_record_carries_no_mask_and_a_load_masks_it() {
    let dir = collection_root("stray-record");
    let template = h2_with(1.1, &[]);
    let policy = FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() };
    let mut s = SurrogateKs::new(Box::new(Composing::new(0.0)), policy).with_teacher_tol(1e-6);
    s.attach_collection(&dir, &template).unwrap();
    s.set_reference_solved(1, SpeciesMorse::atom(1), 81, Some(0.0));
    let mut rng = StdRng::seed_from_u64(0x57c);
    for i in 0..40 {
        let r = TRAINED[0].0 + (TRAINED[0].1 - TRAINED[0].0) * i as f64 / 39.0;
        let origin: [f64; 3] = std::array::from_fn(|k| H2_AT[k] + rng.gen_range(-0.3..0.3));
        s.evaluate(&big(dimer(r, unit_vector(&mut rng), origin, 1, 1)), &mut never()).unwrap();
    }
    s.refit().unwrap();
    assert!(s.trusted(), "the premise: {:?}", s.untrusted_reason());
    let mut s = s.with_auditor(Box::new(Composing::new(0.0)));
    let in_coll = s.collection().unwrap().samples.len();
    s.evaluate(&h2_with(CHECKS[0], &[STRAY_AT]), &mut never()).unwrap();
    let i = s.samples() - 1;
    settle(&mut s);
    assert_eq!(s.quarantine[i], Q_NONE, "the premise: released and appended");
    assert_eq!(s.store().unwrap().samples[i].force_mask, Some(vec![true, true, false]), "masked in memory");
    let rec = &s.collection().unwrap().samples[in_coll].sample;
    assert_eq!((rec.pos.len(), rec.force_mask.clone(), rec.use_energy), (3, None, true), "the record carries no stray mask");
    drop(s);
    // a fresh attach: the record as written, the store's copy masked in memory (and held: the clearance was the session's)
    let mut t = SurrogateKs::new(Box::new(Composing::new(0.0)), policy).with_teacher_tol(1e-6);
    t.attach_collection(&dir, &template).unwrap();
    let rec = &t.collection().unwrap().samples[in_coll].sample;
    assert_eq!((rec.pos.len(), rec.force_mask.clone()), (3, None), "read back unmasked");
    let k = (0..t.samples()).find(|&k| t.store().unwrap().samples[k].pos.len() == 3).expect("the stray sample is in the store");
    assert_eq!((t.store().unwrap().samples[k].force_mask.clone(), t.quarantine[k]), (Some(vec![true, true, false]), Q_STRAY));
    // a model file: written unmasked, masked again at the install
    let (c, _) = cleared_student();
    let path = std::env::temp_dir().join(format!("stray-record-{}.model", std::process::id()));
    c.save_sidecar(&path).unwrap();
    let sc = surrogate::sidecar::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert!(sc.samples[40..43].iter().all(|x| x.pos.len() == 3 && x.force_mask.is_none()), "the file holds no stray mask");
    let (mut u, _, _) = student(0.0, &TRAINED, 40, false);
    u.install_sidecar(sc).unwrap();
    assert!((40..43).all(|k| u.store().unwrap().samples[k].force_mask == Some(vec![true, true, false])), "masked again at the install");
    // the helper: a segment's own mask keeps its other entries
    let mut seg = exact_sample(&h2_with(CHECKS[0], &[STRAY_AT]));
    seg.force_mask = Some(vec![false, true, true]);
    seg.use_energy = false;
    mask_stray_atoms(&mut seg, R_CUT);
    assert_eq!(unmasked_record(&seg, R_CUT).force_mask, Some(vec![false, true, true]));
}

/// A standard normal deviate (Box–Muller).
fn gauss(rng: &mut StdRng) -> f64 {
    let (u, v): (f64, f64) = (rng.gen_range(1e-300..1.0), rng.gen());
    (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
}

/// One labelled geometry of [`diag_stray_sigma_bias`]: the H₂ scene, its energy noise (Ha) and its force noise.
type Noisy = (Scene, f64, Vec<[f64; 3]>);

/// Plan Task 5, C (Task 4 review; measured once, not a gate): the fit's noise levels σ_E and σ_F with and without a
/// set of released stray samples. A masked stray sample's energy row is still (E − Σ E_ref)/n over the scene's n = 3,
/// so its label noise enters at 1/3 where the bonded part's own row has 1/2. Three fits per case: the 40 H₂ samples
/// alone, with `k` released H₂ + H samples, and with the same `k` bonded parts alone (H₂: the same geometries and
/// noise draws — the unbiased counterpart). Exact Morse labels, then labels with Gaussian noise per sample (energy)
/// and per force component. Printed for the task report.
///
/// Task 5 fix round 1 (ruling 9a): the stray row is now per energy-row atom (2), so (b) and (c) should agree with no
/// box term; the stray rows then also carry an ε_pos term (a stray atom's confinement and egg-box term against its
/// box reference, drawn N(0, s_pos²) per sample, s_pos 0, 2e-4 and 1e-3 Ha — the box rule bounds it near 1.8 mHa),
/// which the fragment rows do not.
#[test]
fn diag_stray_sigma_bias() {
    for (label, s_e, s_f, s_pos) in [
        ("exact labels", 0.0, 0.0, 0.0),
        ("noisy labels (s_E 2e-4 Ha per sample, s_F 2e-4 Ha/Bohr), ε_pos 0", 2e-4, 2e-4, 0.0),
        ("noisy labels, ε_pos ~ N(0, (2e-4)²)", 2e-4, 2e-4, 2e-4),
        ("noisy labels, ε_pos ~ N(0, (1e-3)²)", 2e-4, 2e-4, 1e-3),
    ] {
        for k in [10usize, 40] {
            let mut rng = StdRng::seed_from_u64(0x5ca1e);
            let mut rng_pos = StdRng::seed_from_u64(0x5ca1f);
            let eps: Vec<f64> = (0..k).map(|_| s_pos * gauss(&mut rng_pos)).collect();
            let draw = |r: f64, rng: &mut StdRng| -> Noisy {
                let origin: [f64; 3] = std::array::from_fn(|kk| H2_AT[kk] + rng.gen_range(-0.3..0.3));
                let sc = big(dimer(r, unit_vector(rng), origin, 1, 1));
                let de = s_e * gauss(rng);
                let df: Vec<[f64; 3]> = (0..2).map(|_| std::array::from_fn(|_| s_f * gauss(rng))).collect();
                (sc, de, df)
            };
            let base: Vec<Noisy> = (0..40).map(|i| draw(TRAINED[0].0 + (TRAINED[0].1 - TRAINED[0].0) * i as f64 / 39.0, &mut rng)).collect();
            let extra: Vec<Noisy> = (0..k)
                .map(|_| {
                    let r = rng.gen_range(TRAINED[0].0..TRAINED[0].1);
                    draw(r, &mut rng)
                })
                .collect();
            let fit = |extra: &[Noisy], stray: bool| {
                let mut s = SurrogateKs::new(Box::new(Composing::new(0.0)), FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() }).with_teacher_tol(1e-6);
                s.set_reference_energy(1, SpeciesMorse::atom(1));
                for (j, (sc, de, df)) in base.iter().chain(extra).enumerate() {
                    let mut sc = sc.clone();
                    let mut df = df.clone();
                    let mut box_term = 0.0;
                    if stray && j >= base.len() {
                        sc.nuclei.push(Nucleus::new(1, STRAY_AT));
                        df.push([0.0; 3]);
                        box_term = eps[j - base.len()];
                    }
                    let mut smp = exact_sample(&sc);
                    smp.energy += de + box_term;
                    for (f, d) in smp.forces.iter_mut().zip(&df) {
                        for a in 0..3 {
                            f[a] += d[a];
                        }
                    }
                    s.add_sample(smp, &sc).unwrap();
                }
                // released: in the fit, as a clearance would leave them (the mask is applied at the store's entry)
                for q in s.quarantine.iter_mut() {
                    *q = Q_NONE;
                }
                let rep = s.refit().expect("a fit");
                if stray {
                    assert!(s.store().unwrap().samples[base.len()..].iter().all(|x| x.force_mask == Some(vec![true, true, false])), "the premise: masked");
                }
                (rep.n_samples, rep.sigma_e, rep.sigma_f)
            };
            let a = fit(&[], false);
            let b = fit(&extra, true);
            let c = fit(&extra, false);
            println!(
                "{label}, k = {k}: H₂ alone ({}) σ_E {:.4e} σ_F {:.4e} | + {k} released H₂ + H ({}) σ_E {:.4e} σ_F {:.4e} | + the same {k} as H₂ ({}) σ_E {:.4e} σ_F {:.4e} | stray / fragment: σ_E × {:.4}, σ_F × {:.4}; stray / H₂ alone: σ_E × {:.4}",
                a.0,
                a.1,
                a.2,
                b.0,
                b.1,
                b.2,
                c.0,
                c.1,
                c.2,
                b.1 / c.1,
                b.2 / c.2,
                b.1 / a.1
            );
        }
    }
}

// ---- plan Task 5 fix round 1 ----

/// A composing teacher whose answers for a lone atom are `shift` Ha high (a stray atom's confinement and egg-box term
/// against its box reference, ε_pos, which the model's E_ref does not carry); every other answer is `inner`'s.
struct LoneShift {
    inner: Composing,
    shift: f64,
}

impl ForceProvider for LoneShift {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        let mut r = self.inner.evaluate(scene, should_abort)?;
        if scene.nuclei.len() == 1 {
            r.energy += self.shift;
            r.stats.energy.total = r.energy;
        }
        Ok(r)
    }

    fn name(&self) -> &'static str {
        "composing morse"
    }

    fn reset(&mut self) {}
}

/// Minor 1 (rule 7 honesty): the attribution names all three terms of the audit's gap — gap_frag, the composition and
/// the stray atoms' box term ε_pos — and they add up to it (no refit between). The background teacher has a composite
/// offset of 2.5 mHa and a lone atom 2 mHa high: composition +0.5 mHa, ε_pos +2.0 mHa, the model 0 — "mostly the stray
/// atoms' place in the box", not a composition that "does not split". Every ordinary check logs ε_pos too.
#[test]
fn the_attribution_names_the_three_terms_of_the_gap() {
    let delta = 2e-3;
    let (mut s, _, _) = audited_student(Box::new(LoneShift { inner: Composing::new(AUDIT_OFFSET), shift: delta }));
    settle(&mut s);
    let ev = s.take_events();
    let att = ev.iter().find(|e| matches!(e.outcome, TeachOutcome::StrayAuditAttributed { .. })).expect("an attribution");
    assert_eq!(att.outcome, TeachOutcome::StrayAuditAttributed { model: false }, "{}", att.detail);
    assert!(att.detail.contains("mostly the stray atoms' place in the box"), "{}", att.detail);
    assert!(att.detail.contains("of teacher − model +2.500 mHa: the model on the bonded part -0.000 mHa, the composition E_scene − E_frag − Σ E_atom +0.500 mHa, the stray atoms' box term Σ (E_atom − E_box) +2.000 mHa"), "{}", att.detail);
    let gap_frag = att.gap.expect("the bonded part's gap");
    assert!((gap_frag + (AUDIT_OFFSET - delta) + delta - AUDIT_OFFSET).abs() < 1e-7, "gap_audit = gap_frag + comp + ε_pos ({gap_frag:e})");
    // an ordinary check logs ε_pos: the checks of `cleared_student` (exact teacher: 0)
    let (mut c, _, _) = student(0.0, &TRAINED, 40, true);
    c.evaluate(&h2_with(CHECKS[0], &[STRAY_AT]), &mut never()).unwrap();
    settle(&mut c);
    let ev = c.take_events();
    assert!(ev.iter().any(|e| e.detail.contains("composition check passed") && e.detail.contains("[the stray atoms' box term Σ (E_atom − E_box) = +0.000 mHa]")), "{ev:?}");
}

/// Important 1: a Forget while an attribution check is in flight discards it without counting anything — no failed
/// audit, no audit gap left behind (`reset_audits`: nothing is booked), no "counts against the model" note — and its
/// late parts are dropped.
#[test]
fn a_forget_during_an_attribution_counts_nothing() {
    let auditor = Composing::new(AUDIT_OFFSET);
    let gate = auditor.open.clone();
    let (mut s, _, _) = audited_student_gated(Box::new(auditor), Some(&gate));
    assert!(s.stray_pending.as_ref().is_some_and(|p| p.attribution.is_some()), "the premise: in flight");
    s.pending_notes.clear();
    s.forget();
    assert!(s.stray_pending.is_none() && s.stray_attributions.is_empty());
    assert!(s.failed_audits.is_empty() && s.audit_failed().is_none(), "nothing counted: {:?}", s.audit_failed());
    assert!(!s.pending_notes.iter().chain(&s.user_notes).any(|n| n.contains("counts against the model")), "{:?}", s.pending_notes);
    gate.store(true, SeqCst);
    let t0 = Instant::now();
    while !s.auditor_idle() && t0.elapsed().as_secs_f64() < 5.0 {
        std::thread::sleep(std::time::Duration::from_millis(1));
        s.poll_side_jobs();
    }
    assert!(s.failed_audits.is_empty() && !s.take_events().iter().any(|e| matches!(e.outcome, TeachOutcome::StrayAuditAttributed { .. })));
}

/// Important 2: a basis change while an attribution check is in flight (here a refit with a new species) clears the
/// book but keeps the check — its parts are teacher solves at a fixed geometry — and it is judged with the model current
/// when they answer: the composition's failure, the model untouched (it would otherwise have counted with the
/// composite label, never to be re-earned).
#[test]
fn an_attribution_survives_a_basis_change() {
    let auditor = Composing::new(AUDIT_OFFSET);
    let gate = auditor.open.clone();
    let (mut s, _, _) = audited_student_gated(Box::new(auditor), Some(&gate));
    s.set_reference_energy(2, SpeciesMorse::atom(2));
    let he = {
        let mut sc = big(dimer(2.6, [1.0, 0.0, 0.0], H2_AT, 1, 2));
        sc.nuclei[1].z = 2;
        sc
    };
    s.add_sample(exact_sample(&he), &he).unwrap();
    s.refit().unwrap();
    assert_eq!(s.model().unwrap().basis.species, vec![1, 2], "the premise: another basis");
    assert!(s.stray_book_current() || s.stray_book.state(&sig_of(&s, &h2_with(INSIDE, &[STRAY_AT]))).is_none(), "the book is cleared");
    assert!(s.stray_pending.as_ref().is_some_and(|p| p.attribution.is_some()), "the attribution check is kept");
    assert!(s.side_queue.iter().any(|j| matches!(j.2, JobKind::AtomAt(_))) || !s.auditor_idle(), "its parts too");
    gate.store(true, SeqCst);
    settle(&mut s);
    let ev = s.take_events();
    assert!(ev.iter().any(|e| e.outcome == TeachOutcome::StrayAuditAttributed { model: false }), "{ev:?}");
    assert!(s.failed_audits.is_empty(), "the model untouched");
}

/// Minor 5: an attribution that lands after a refit and finds the model right audits the next `AUDIT_BURST` answered
/// steps, as a failed audit a refit re-earns does.
#[test]
fn an_attribution_after_a_refit_audits_the_next_steps() {
    let auditor = Composing::new(AUDIT_OFFSET);
    let gate = auditor.open.clone();
    let (mut s, _, _) = audited_student_gated(Box::new(auditor), Some(&gate));
    s.audit_burst = 0;
    s.refit().unwrap();
    gate.store(true, SeqCst);
    settle(&mut s);
    assert!(s.take_events().iter().any(|e| e.outcome == TeachOutcome::StrayAuditAttributed { model: false }));
    assert_eq!(s.audit_burst, AUDIT_BURST);
}

/// Minor 4: a second failed stray audit while an attribution check is in flight waits behind it (not counted at once
/// with the composite label); both are attributed in turn.
#[test]
fn failed_stray_audits_queue_for_their_attribution() {
    let auditor = Composing::new(AUDIT_OFFSET);
    let gate = auditor.open.clone();
    let (mut s, _, _) = audited_student_gated(Box::new(auditor), Some(&gate));
    let sc = h2_with(1.45, &[STRAY_AT]);
    let (pos, z) = (positions(&sc), zs(&sc));
    let (e, f) = SpeciesMorse::energy_and_forces(&pos, &z);
    let case = FailedAudit { pos: pos.clone(), z: z.clone(), energy: e + AUDIT_OFFSET, forces: f.clone(), index: None, ionic: s.provenance_for(&sc).ionic.clone(), rejudges: 0, origin: AuditOrigin::Audit, pair: None };
    match s.stray_audit_failed(&sc, e + AUDIT_OFFSET, &f, &case, AUDIT_OFFSET, "test".into(), Default::default()) {
        StrayAudit::Attributing(w) => assert!(w.contains("queued behind 1 other"), "{w}"),
        other => panic!("queued: {other:?}"),
    }
    assert_eq!(s.stray_attributions.len(), 1);
    gate.store(true, SeqCst);
    settle(&mut s);
    let n = s.take_events().iter().filter(|e| e.outcome == TeachOutcome::StrayAuditAttributed { model: false }).count();
    assert_eq!((n, s.stray_attributions.len()), (2, 0), "both attributed, in turn");
    assert!(s.failed_audits.is_empty());
}

/// Important 3: a passing audit of a cleared answer is recorded over the bonded part — n_atoms 2, the bonded part's
/// energy std (bit for bit the scene's: the floor per energy-row atom) — so calibration's energy evidence is per bonded
/// atom.
#[test]
fn a_cleared_scenes_history_point_is_over_its_bonded_part() {
    let (s, _, _) = audited_student(Box::new(Composing::new(0.0)));
    let p = s.audit_history().iter().last().expect("the audit's point").clone();
    assert_eq!((p.n_atoms, p.kind), (2, surrogate::history::PointKind::Audit));
    let sc = h2_with(INSIDE, &[STRAY_AT]);
    let frag = s.model().unwrap().predict(&positions(&sc)[..2], &[1, 1], true).unwrap();
    assert_eq!(p.pred_energy_std.to_bits(), frag.energy_std.unwrap().to_bits());
    assert!(p.classes.iter().all(|&c| c != 0), "no isolated atom recorded");
}

/// Minor 2: two fence votes of different scenes with the same per-atom count (H₂, and H₂ + a stray H: n_bonded 2
/// each) do not agree — the atoms must be the same, or the per-atom pairing mispairs.
#[test]
fn fence_votes_of_different_scenes_never_agree() {
    use super::super::quarantine::{votes_agree, FenceVote};
    let p = FallbackPolicy::default();
    let a = FenceVote { near: vec![0], n: 2, energy: -1.0, forces: vec![[0.0; 3]; 2], residual: None, gen: 0 };
    let b = FenceVote { forces: vec![[0.0; 3]; 3], ..a.clone() };
    assert!(votes_agree(&p, &a, &a.clone()), "the premise: equal votes agree");
    assert!(!votes_agree(&p, &a, &b) && !votes_agree(&p, &b, &a));
}

/// Minor 7: a stray-attributed failed audit whose geometry no longer splits at the model's cutoff (a longer reach)
/// turns into an audit fence at the next re-judgement; a whole sample's stray mask is derived afresh at each entry, so
/// a mask of a geometry no longer stray is dropped.
#[test]
fn a_stray_label_is_valid_only_while_the_atom_is_stray() {
    let (mut s, _) = cleared_student();
    let sc = big(dimer(INSIDE, [1.0, 0.0, 0.0], H2_AT, 1, 1));
    let (e, f) = SpeciesMorse::energy_and_forces(&positions(&sc), &zs(&sc));
    s.failed_audits.push(FailedAudit { pos: positions(&sc), z: zs(&sc), energy: e + 0.1, forces: f, index: None, ionic: s.provenance_for(&sc).ionic.clone(), rejudges: 0, origin: AuditOrigin::StrayAttributed, pair: None });
    s.rejudge_failed_audits(true);
    assert!(s.failed_audits.is_empty() && s.audit_fences.iter().any(|c| c.origin == AuditOrigin::StrayAttributed), "fenced");
    assert!(s.user_notes.iter().any(|n| n.contains("no longer such a scene at the model's cutoff")), "{:?}", s.user_notes);
    let mut smp = exact_sample(&sc);
    smp.force_mask = Some(vec![true, false]);
    mask_stray_atoms(&mut smp, R_CUT);
    assert_eq!(smp.force_mask, None, "a stale stray mask is dropped");
}

/// Ruling 9b: the `composition` count survives a model file (an optional header key; a file without it reads 0).
#[test]
fn the_composition_count_is_saved_with_the_model() {
    let (mut s, _, _) = audited_student(Box::new(Composing::new(AUDIT_OFFSET)));
    settle(&mut s);
    assert_eq!(s.audit_history().unrecorded.composition, 1);
    let path = std::env::temp_dir().join(format!("stray-composition-{}.model", std::process::id()));
    s.save_sidecar(&path).unwrap();
    let sc = surrogate::sidecar::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(sc.audit_history.unrecorded.composition, 1);
}

/// Ruling 9a, watch-list item 5: a stray sample in the misfit set M is named as one in the trust words (its energy row
/// carries the composition and box terms), and the words without one are unchanged.
#[test]
fn a_stray_misfit_is_named_in_the_words() {
    let (mut s, _) = cleared_student();
    let before = s.misfit_scenes();
    assert!(!before.contains("stray"), "{before}");
    let i = (0..s.samples()).find(|&i| s.store().unwrap().samples[i].pos.len() == 3).expect("a released stray sample");
    s.misfit.as_mut().unwrap().samples.push(i);
    assert!(s.misfit_scenes().ends_with("; 1 with stray atoms, whose energy rows per bonded atom also carry the composition and the stray atoms' box term"), "{}", s.misfit_scenes());
}

/// Task 6 review I1: a signature's label writes the stray atoms as separate atoms — two stray H are "2 H", never the
/// formula "H2" (which reads as a hydrogen molecule).
#[test]
fn a_signature_label_names_stray_atoms_as_atoms() {
    let sig = |bonded: &[u8], stray: &[u8]| Signature { bonded_z: bonded.to_vec(), stray_z: stray.to_vec(), charge: 0, surface: String::new() };
    assert_eq!(signature_label(&sig(&[1, 1], &[1])), "H2 + H");
    assert_eq!(signature_label(&sig(&[1, 1], &[1, 1])), "H2 + 2 H");
    assert_eq!(signature_label(&sig(&[1, 1, 8], &[8])), "H2O + O");
    assert_eq!(signature_label(&sig(&[1, 1], &[1, 8])), "H2 + H + O");
}

// ---- controller ruling 10 (Task 7 on the real teacher): the frontier window and the empty channel ----

/// A failed verdict's words (panics on any other verdict).
fn failed(v: CheckVerdict) -> String {
    match v {
        CheckVerdict::Fail(why) => why,
        v => panic!("not a failure: {v:?}"),
    }
}

/// Rulings 10a and 11: the frontier levels are compared only within 13.8 kT (69 mHa) of the scene's μ_σ. The real teacher's H₂ + H at
/// 1.434 Bohr (81³ × 0.20, `references.md` "Stray atoms"): the ↑ LUMOs are box states — scene +25.5, H₂ σu +28.4,
/// H +36.5 mHa, a 2.9 mHa mismatch — about 145 mHa above μ↑; they move with the box, not the composition.
#[test]
fn judge_compares_frontier_levels_only_within_the_window() {
    let p = FallbackPolicy::default();
    let box_states = || {
        let (mut scene, mut frag, mut atoms, split) = exact();
        frag.frontier[0].lumo = Some(0.0284);
        frag.frontier[1].lumo = Some(0.0284);
        atoms[0].frontier[0].lumo = Some(0.0365);
        scene.frontier[0].lumo = Some(0.0255);
        (scene, frag, atoms, split)
    };
    // without a μ (an aufbau teacher): every level compared, as before the window — the box states fail
    let (scene, frag, atoms, split) = box_states();
    let why = failed(judge(&p, &scene, &frag, &atoms, &split));
    assert!(why.contains("LUMO ↑"), "{why}");
    // the scene's μ↑ at −0.12 Ha: +25.5 mHa lies 145.5 mHa above it, outside 69 — not compared; μ↓ at −0.21 keeps the
    // ↓ HOMO (−0.37) and LUMO (−0.05) 160 mHa away, outside too
    let (mut scene, frag, atoms, split) = box_states();
    scene.frontier[0].mu = Some(-0.12);
    scene.frontier[1].mu = Some(-0.21);
    assert!(matches!(judge(&p, &scene, &frag, &atoms, &split), CheckVerdict::Pass { .. }), "{:?}", judge(&p, &scene, &frag, &atoms, &split));
    // a mismatch inside the window still fails: μ↑ at −0.03 puts +25.5 mHa 55.5 mHa above it
    scene.frontier[0].mu = Some(-0.03);
    let why = failed(judge(&p, &scene, &frag, &atoms, &split));
    assert!(why.contains("LUMO ↑") && why.contains("μ"), "{why}");
    // an in-window HOMO mismatch fails; one far below μ is not compared
    let (mut scene, frag, atoms, split) = box_states();
    scene.frontier[0].mu = Some(-0.12);
    scene.frontier[1].mu = Some(-0.21);
    scene.frontier[1].homo = Some(-0.37 - 5e-3);
    assert!(matches!(judge(&p, &scene, &frag, &atoms, &split), CheckVerdict::Pass { .. }), "a HOMO 165 mHa below μ↓ is outside");
    // μ↑ at −0.20: the scene's ↑ HOMO 5 mHa above the parts' (−0.265 against −0.27) lies 65 mHa below μ↑, inside
    scene.frontier[0].mu = Some(-0.20);
    scene.frontier[0].homo = Some(-0.27 + 5e-3);
    assert!(failed(judge(&p, &scene, &frag, &atoms, &split)).contains("HOMO ↑"));
}

/// Ruling 10a: an occupied level on one side only fails wherever it lies (the channel's electrons moved), and the
/// partial levels are compared unfiltered (each lies within 13.8 kT of its own μ_σ: inside the window by construction).
#[test]
fn judge_window_keeps_one_sided_and_partial_levels() {
    let p = FallbackPolicy::default();
    let (mut scene, mut frag, atoms, split) = exact();
    scene.frontier[0].mu = Some(-0.12);
    scene.frontier[1].mu = Some(-0.21);
    frag.frontier[1].homo = None;
    let why = failed(judge(&p, &scene, &frag, &atoms, &split));
    assert!(why.contains("HOMO ↓"), "{why}");
    let (mut scene, frag, atoms, split) = exact();
    scene.frontier[0].mu = Some(-0.12);
    scene.frontier[1].mu = Some(-0.21);
    scene.partial = vec![PartialLevel { spin: 0, index: 1, eps: -0.12, f: 0.5 }];
    let why = failed(judge(&p, &scene, &frag, &atoms, &split));
    assert!(why.contains("partly occupied"), "{why}");
}

/// A part's orbital payload with one unoccupied row per channel (what [`part_view`] asks the teacher for): ↑ at
/// `up`, ↓ at `down` (Ha), occupation 0.
fn unoccupied_payload(up: f64, down: f64) -> sim::OrbitalDisplay {
    let row = |spin: sim::Spin, energy: f64| sim::OrbitalInfo {
        spin, index: 0, track: 0, slot: 0, energy, eps_range: (energy, energy), occupation: 0.0, label: sim::OrbitalLabel::Index(1),
        cluster: None, rotated: false, grid_split: None, box_state: false, smeared_1s: false, residual: None, not_converged: false, l: None, touches_box: false, shares: None,
    };
    sim::OrbitalDisplay {
        cell: Cell { n: 81, h: 0.25 }, origin: [0.0; 3], positions: Vec::new(), rows: vec![row(sim::Spin::Up, up), row(sim::Spin::Down, down)],
        field: None, spin: sim::SpinSummary::default(), unoccupied: None, notes: Vec::new(), what: sim::OrbitalRequest::List,
        lone_atom: sim::LoneAtomLabels::NotLoneAtom, share_z: None, kind: sim::OrbitalKind::Converged,
    }
}

/// Ruling 10b: a stray H alone stores no ↓ orbital, so its ↓ channel has no level and the check is void for good; the
/// part's requested unoccupied level ([`with_unoccupied`]) completes it and the check is judged — a real acceptor
/// (the empty 1s↓) in the margin, never "above every computed level".
#[test]
fn an_empty_channel_is_judged_with_its_requested_unoccupied_level() {
    assert_eq!(part_view(), sim::OrbitalViewRequest { what: sim::OrbitalRequest::List, unoccupied: 1, force_every_step: true });
    let p = FallbackPolicy::default();
    let (scene, frag, mut atoms, split) = exact();
    atoms[0].frontier[1] = Frontier::default();
    match judge(&p, &scene, &frag, &atoms, &split) {
        CheckVerdict::Void(why) => assert!(why.contains("not computed"), "{why}"),
        v => panic!("an empty channel without its level is void: {v:?}"),
    }
    // the payload's ↑ row lies above the SCF block's LUMO (kept), its ↓ row is the channel's only level
    atoms[0].frontier = with_unoccupied(atoms[0].frontier, Some(&unoccupied_payload(0.06, -0.05)));
    assert_eq!((atoms[0].frontier[0].lumo, atoms[0].frontier[1].lumo, atoms[0].frontier[1].homo), (Some(0.05), Some(-0.05), None));
    assert!(matches!(judge(&p, &scene, &frag, &atoms, &split), CheckVerdict::Pass { .. }), "{:?}", judge(&p, &scene, &frag, &atoms, &split));
    // the completed level is an acceptor in the margin: a stray 1s↓ within 138 mHa of the fragment's σg fails it
    let (mut scene, frag, mut atoms, split) = exact();
    atoms[0].frontier[1] = Frontier::default();
    atoms[0].frontier = with_unoccupied(atoms[0].frontier, Some(&unoccupied_payload(0.06, -0.30)));
    scene.frontier[1].lumo = Some(-0.30);
    let why = failed(judge(&p, &scene, &frag, &atoms, &split));
    assert!(why.contains("margin") && why.contains("stray acceptor"), "{why}");
    // a lower payload level replaces the SCF block's; no payload leaves the frontier as it was
    let f = [Frontier { homo: Some(-0.3), lumo: Some(0.2), mu: Some(-0.1) }, Frontier::default()];
    assert_eq!(with_unoccupied(f, Some(&unoccupied_payload(0.1, 0.3)))[0].lumo, Some(0.1));
    assert_eq!(with_unoccupied(f, None), f);
}

/// A [`Composing`] auditor that reports what the real teacher does (Task 7): a one-electron solve (a stray H alone)
/// stores no ↓ orbital, so its ↓ channel has no level — unless its orbital view asks for unoccupied orbitals, when the
/// payload carries the empty 1s↓ (−0.05) and the ↑ 2s (0.0). Every view it is given is recorded.
struct EmptyChannel {
    inner: Composing,
    view: sim::OrbitalViewRequest,
    views: Arc<Mutex<Vec<sim::OrbitalViewRequest>>>,
}

impl ForceProvider for EmptyChannel {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        let mut r = self.inner.evaluate(scene, should_abort)?;
        let electrons: u32 = scene.nuclei.iter().map(|n| n.z as u32).sum();
        if electrons == 1 {
            r.stats.frontier[1] = Frontier::default();
            if self.view.unoccupied >= 1 {
                r.orbitals = Some(unoccupied_payload(0.0, -0.05));
            }
        }
        Ok(r)
    }

    fn name(&self) -> &'static str {
        "composing morse"
    }

    fn reset(&mut self) {}

    fn set_orbital_view(&mut self, request: sim::OrbitalViewRequest) {
        self.views.lock().unwrap().push(request.clone());
        self.view = request;
    }
}

/// Ruling 10b through the provider: the auditor's part jobs carry [`part_view`], so an auditor whose lone H has no ↓
/// level unless asked is judged (a pass), not void; the view is set for the part jobs only.
#[test]
fn part_solves_ask_for_an_unoccupied_level_per_channel() {
    let (s, _, _) = student(0.0, &TRAINED, 40, false);
    let views = Arc::new(Mutex::new(Vec::new()));
    let mut s = s.with_auditor(Box::new(EmptyChannel { inner: Composing::new(0.0), view: sim::OrbitalViewRequest::default(), views: views.clone() }));
    let sc = h2_with(CHECKS[0], &[STRAY_AT]);
    assert_eq!(s.evaluate(&sc, &mut never()).unwrap().stats.source, Source::Fallback);
    settle(&mut s);
    let ev = s.take_events();
    let check: Vec<&sim::TeachEvent> = ev.iter().filter(|e| e.origin == TeachOrigin::Audit).collect();
    assert_eq!(check.iter().map(|e| e.outcome).collect::<Vec<_>>(), [TeachOutcome::StrayChecking { n: 1, of: 3 }], "{check:?}");
    assert_eq!(views.lock().unwrap().as_slice(), [part_view()], "set once for the parts, kept across both");
}

/// Ruling 11: the real teacher's stray H's empty 1s↓, 4.3 mHa lower in the scene than alone (−95.4 against −91.1 mHa at
/// r = 1.75 Bohr, 81³ × 0.20) and 126.3 mHa ≈ 25 kT above the scene's μ↓ (−221.7): outside the 13.8 kT window, so not
/// compared — the check passes; the same mismatch within 13.8 kT of μ↓ fails.
#[test]
fn a_far_level_mismatch_passes_and_an_in_window_one_fails() {
    let p = FallbackPolicy::default();
    assert!((level_window() - 13.8155 * qm::ELECTRON_KT).abs() < 1e-4 * qm::ELECTRON_KT, "ln(1e6) kT");
    let (mut scene, frag, mut atoms, split) = exact();
    atoms[0].frontier[1].lumo = Some(-0.0911);
    scene.frontier[1].lumo = Some(-0.0954);
    scene.frontier[0].mu = Some(-0.1357);
    scene.frontier[1].mu = Some(-0.2217);
    assert!((-0.0954f64 - -0.2217) / qm::ELECTRON_KT > 25.0, "the premise: ≈ 25 kT out");
    assert!(matches!(judge(&p, &scene, &frag, &atoms, &split), CheckVerdict::Pass { .. }), "{:?}", judge(&p, &scene, &frag, &atoms, &split));
    // μ↓ at −0.15: the 1s↓ lies 54.6 mHa ≈ 10.9 kT above it, inside the window — the 4.3 mHa mismatch fails
    scene.frontier[1].mu = Some(-0.15);
    let why = failed(judge(&p, &scene, &frag, &atoms, &split));
    assert!(why.contains("LUMO ↓"), "{why}");
}

/// Ruling 12: the region covers a geometry inside the per-feature box **or** within the thinning radius of a passed
/// check, in the distinctness metric — so every geometry of a measurable species is covered or distinct (checkable).
/// The real teacher's checks (H–H 1.434, 1.15, 1.75 Bohr) under the default H basis: 1.435 lies outside the box (1.434
/// sits at b2[n=6]'s maximum) yet within the radius of the 1.434 check — covered, not distinct.
///
/// Final review I2(a) (ruling 13): the covered bond interval about each check, printed and asserted, with the
/// stretched check at 2.5 Bohr added (the GPU clears test's fourth check). The frontier margin σu − 1s falls from
/// 181 mHa at 2.5 to 136.9 at 3.0, under the 138 mHa floor (`references.md`, "Stray atoms"): the 2.5 check's reach must
/// stop short of 3.0. Units: an H₂ training envelope over 1.0–2.5 Bohr, frozen at each check.
#[test]
fn the_region_covers_the_thinning_radius_of_its_checks() {
    let basis = Basis::new(&[1]);
    let mut env = Envelope::new(&basis);
    for i in 0..41 {
        let (d, s) = h2_desc(&basis, 1.0 + 1.5 * i as f64 / 40.0);
        for (si, b) in s.iter().zip(&d) {
            env.add_atom(*si, b);
        }
    }
    let mut region = Region::new(&basis);
    for r in [1.434, 1.15, 1.75] {
        let (d, s) = h2_desc(&basis, r);
        region.add(&d, &s, &env, thin());
    }
    let (d, s) = h2_desc(&basis, 1.435);
    assert!(!region.contains(&d, &s), "the premise: 1.435 leaves the box (the dead zone)");
    assert!(region.near(&d, &s) && region.covers_bonded(&d, &s), "within the radius: covered");
    assert!(!region.distinct(&d, &s, &env), "…and not checkable");
    // covered ∪ distinct is everything, and the two never overlap outside the box
    let mut beyond = None;
    for k in 0..=200 {
        let r = 0.9 + 0.01 * k as f64;
        let (d, s) = h2_desc(&basis, r);
        let (covered, distinct) = (region.covers_bonded(&d, &s), region.distinct(&d, &s, &env));
        assert!(covered || distinct, "r = {r}: neither covered nor checkable");
        assert!(!(distinct && region.near(&d, &s)), "r = {r}");
        if !covered && beyond.is_none() && r > 1.75 {
            beyond = Some(r);
        }
    }
    // a geometry beyond both is not covered and is distinct: it falls back and queues a check
    let r = beyond.expect("a stretched bond beyond the box and the radius");
    let (d, s) = h2_desc(&basis, r);
    assert!(!region.covers_bonded(&d, &s) && region.distinct(&d, &s, &env), "r = {r}");
    // a species the units never saw is never measurable (not distinct), and nothing near it
    let empty = Envelope::new(&basis);
    assert!(!region.distinct(&d, &s, &empty));
    // the covered interval about each check (steps of 1e-3 Bohr outward until a bond is not covered; the box and the
    // radius together): the three checks, then with the stretched check
    let intervals = |region: &Region, checks: &[f64]| {
        let covered = |r: f64| {
            let (d, s) = h2_desc(&basis, r);
            region.covers_bonded(&d, &s)
        };
        let edge = |c: f64, dir: f64| (1..=3000).map(|k| c + dir * 1e-3 * k as f64).take_while(|&r| covered(r)).last().unwrap_or(c);
        let mut lines = Vec::new();
        for &c in checks {
            let (lo, hi) = (edge(c, -1.0), edge(c, 1.0));
            lines.push(format!("check {c}: covered {lo:.3}–{hi:.3} Bohr (−{:.3} / +{:.3})", c - lo, hi - c));
            assert!(lo <= c && c <= hi && hi - lo < 1.0, "{}", lines.last().unwrap());
        }
        lines.join("; ")
    };
    println!("the covered bond intervals, 3 checks (default H basis, units H₂ 1.0–2.5): {}", intervals(&region, &[1.15, 1.434, 1.75]));
    let (d, s) = h2_desc(&basis, 2.5);
    region.add(&d, &s, &env, thin());
    println!("the covered bond intervals, with the 2.5 check: {}", intervals(&region, &[1.15, 1.434, 1.75, 2.5]));
    let covered = |r: f64| {
        let (d, s) = h2_desc(&basis, r);
        region.covers_bonded(&d, &s)
    };
    let edge = |c: f64, dir: f64| (1..=3000).map(|k| c + dir * 1e-3 * k as f64).take_while(|&r| covered(r)).last().unwrap_or(c);
    let stretched_hi = edge(2.5, 1.0);
    assert!(stretched_hi < 3.0, "the 2.5 Bohr check's reach stops short of 3.0, where the margin crosses the floor: {stretched_hi}");
    assert!(!covered(3.0), "3.0 Bohr is not covered");
}

/// Ruling 13 (final review I1, I3): a region covers a geometry only with its stray atoms where the passes measured
/// them — no nearer the bonded part than the nearest pass, no nearer a face — the bonded geometry covered or not, with
/// no slack; a pass at a nearer place extends it in the Checking state too (it counts nothing towards K), and an empty
/// region covers nothing.
#[test]
fn a_region_covers_the_stray_atoms_only_where_its_passes_put_them() {
    let basis = Basis::new(&[1, 8]);
    let env = units(&basis);
    let mut book = StrayBook::default();
    let places = [StrayPlace { reach: 8.5, face: 6.0 }, StrayPlace { reach: 8.0, face: 5.5 }, StrayPlace { reach: 9.0, face: 5.2 }];
    for (r, p) in [1.35, 1.45, 1.55].into_iter().zip(places) {
        let (d, s) = h2_desc(&basis, r);
        book.record(sig(), PASS, &d, &s, p, &basis, &env, thin());
    }
    let Some(StrayState::Cleared { region, .. }) = book.state(&sig()) else { panic!("{:?}", book.state(&sig())) };
    assert_eq!(region.place(), StrayPlace { reach: 8.0, face: 5.2 }, "the smallest of the passes'");
    let (d, s) = h2_desc(&basis, 1.45);
    for (p, want) in [
        (StrayPlace { reach: 8.0, face: 5.2 }, true),
        (StrayPlace { reach: 20.0, face: 9.0 }, true),
        (StrayPlace { reach: 7.999, face: 6.0 }, false),
        (StrayPlace { reach: 9.0, face: 5.199 }, false),
        (StrayPlace { reach: f64::NAN, face: 6.0 }, false),
    ] {
        assert_eq!(region.covers(&d, &s, p), want, "{p:?}");
    }
    // a pass differing only in where the stray atom sits extends the place (an open region)
    book.record(sig(), PASS, &d, &s, StrayPlace { reach: 7.0, face: 5.2 }, &basis, &env, thin());
    let Some(StrayState::Cleared { region, .. }) = book.state(&sig()) else { panic!() };
    assert!(region.covers(&d, &s, StrayPlace { reach: 7.0, face: 5.2 }));
    // in the Checking state too, while counting nothing towards K
    let mut other = sig();
    other.charge = -1;
    let (d0, s0) = h2_desc(&basis, 1.35);
    book.record(other.clone(), PASS, &d0, &s0, PLACE, &basis, &env, thin());
    book.record(other.clone(), PASS, &d0, &s0, StrayPlace { reach: 6.5, face: 4.5 }, &basis, &env, thin());
    let Some(StrayState::Checking { passes: 1, region, .. }) = book.state(&other) else { panic!("{:?}", book.state(&other)) };
    assert_eq!(region.place(), StrayPlace { reach: 6.5, face: 4.5 });
    // an empty region covers nothing, wherever the stray atoms sit
    let empty = Region::new(&basis);
    assert!(!empty.covers(&d, &s, StrayPlace { reach: f64::INFINITY, face: f64::INFINITY }));
    // the place of a scene: the nearest stray–bonded pair and the stray atom nearest a face (half-extent 10)
    let pos = [[0.0, -3.0, 0.0], [1.4, -3.0, 0.0], [0.0, 5.0, 0.0], [-6.0, 2.0, -8.5]];
    let p = StrayPlace::of(&pos, &Split { bonded: vec![0, 1], stray: vec![2, 3] }, 10.0);
    assert!((p.reach - 8.0).abs() < 1e-12 && (p.face - 1.5).abs() < 1e-12, "{p:?}");
}

/// Ruling 12 through the provider: after the three checks clear H₂ + H, a step at a bond within the thinning radius of
/// a check but outside the box is the model's (no check queued), and one beyond both falls back and queues a check.
#[test]
fn a_step_near_a_check_is_answered_and_one_beyond_queues_a_check() {
    let (mut s, sig) = cleared_student();
    let Some(StrayState::Cleared { region, .. }) = s.stray_book.state(&sig).cloned() else { panic!("cleared: {:?}", s.stray_book.state(&sig)) };
    let split = Split { bonded: vec![0, 1], stray: vec![2] };
    let desc = |s: &SurrogateKs, r: f64| {
        let sc = h2_with(r, &[STRAY_AT]);
        s.bonded_descriptors(&positions(&sc), &zs(&sc), &split).unwrap()
    };
    // the first bond off the outer check (steps of 1e-4 Bohr) outside the box but within the radius, and the first beyond
    // both
    let near = (1..=2000).map(|k| CHECKS[2] + 1e-4 * k as f64).find(|&r| {
        let (d, sp) = desc(&s, r);
        !region.contains(&d, &sp) && region.near(&d, &sp)
    });
    let beyond = (1..=2000).map(|k| CHECKS[2] + 1e-3 * k as f64).find(|&r| {
        let (d, sp) = desc(&s, r);
        !region.covers_bonded(&d, &sp)
    });
    let (near, beyond) = (near.expect("the premise: a dead-zone bond near the 1.55 check"), beyond.expect("a bond beyond both"));
    let _ = s.take_events();
    s.discontinuity();
    let r = s.evaluate(&h2_with(near, &[STRAY_AT]), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Surrogate, "H–H {near}: {:?}", r.stats.notes);
    assert!(s.stray_pending.is_none() && s.side_queue.is_empty(), "no check");
    s.discontinuity();
    let r = s.evaluate(&h2_with(beyond, &[STRAY_AT]), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback, "H–H {beyond}");
    assert!(r.stats.notes.iter().any(|n| n.contains("composition check") && n.contains("queued")), "H–H {beyond}: {:?}", r.stats.notes);
}

// ---- the final review's fix round (controller ruling 13): where the stray atoms sit, the frozen radius, minors ----

/// Whether the step at `sc` would be answered as a cleared stray-atom scene (`stray_cleared`: the book, the floor, the
/// region with the stray atoms' place).
fn cleared_at(s: &SurrogateKs, sc: &Scene) -> bool {
    s.stray_cleared(&positions(sc), &zs(sc), &s.provenance_for(sc)).is_some()
}

/// The step queued a composition check (its note; a void's words "no check is queued again" are not one).
fn queued(r: &ForceResult) -> bool {
    r.stats.notes.iter().any(|n| n.contains("a composition check (the bonded part alone, each stray atom alone) is queued"))
}

/// Final review I1 (ruling 13): the region covers the stray atoms only as near the bonded part as a passed check put
/// them. The checks' stray H ([`STRAY_AT`]) is 8.03 Bohr from the nearer H₂ atom; at y = 4 it is 7.03 Bohr away (and
/// 6 Bohr from a face, farther than the checks' 5): the bonded geometry is covered, the stray atom is not — the step
/// falls back on rule 5a, its sample is held, and a check is queued even though the bonded geometry is inside the
/// region. Its pass lowers the region's reach to 7.03: the step there is then the model's, one nearer still is not.
#[test]
fn a_stray_atom_nearer_than_every_check_falls_back_and_checks() {
    let (mut s, sig) = cleared_student();
    let near = h2_with(INSIDE, &[[0.0, 4.0, 0.0]]);
    assert!(cleared_at(&s, &h2_with(INSIDE, &[STRAY_AT])), "the premise: the bonded geometry is covered");
    assert!(!cleared_at(&s, &near), "a stray atom nearer than every check is not covered");
    // its sample is held out of the fit (the same rule)
    let (pos, z) = (positions(&near), zs(&near));
    let (e, f) = SpeciesMorse::energy_and_forces(&pos, &z);
    let mut notes = Vec::new();
    let prov = s.provenance_for(&near);
    assert!(s.keep_sample(&pos, &z, e, &f, Some(0.0), Some(2), &prov, SampleTag::Fallback, &mut notes), "{notes:?}");
    let held = s.samples() - 1;
    assert_eq!(s.quarantine[held], Q_STRAY, "held: {notes:?}");
    let r = judged_step(&mut s, &near);
    assert_eq!(r.stats.source, Source::Fallback, "{:?}", r.stats.notes);
    assert!(r.stats.notes.iter().any(|n| n.contains("isolated atom — atom 2")), "{:?}", r.stats.notes);
    assert!(queued(&r), "a check is queued although the bonded geometry is covered: {:?}", r.stats.notes);
    settle(&mut s);
    let ev = s.take_events();
    assert!(ev.iter().any(|e| e.outcome == TeachOutcome::StrayCheckingNew), "{ev:?}");
    let Some(StrayState::Cleared { region, .. }) = s.stray_book.state(&sig) else { panic!("{:?}", s.stray_book.state(&sig)) };
    let reach = region.place().reach;
    println!("the region's reach after the nearer pass: {reach:.4} Bohr (the face distance {:.2})", region.place().face);
    assert!((reach - (0.49f64 + 49.0).sqrt()).abs() < 1e-9, "the nearest stray–bonded distance of the passes: {reach}");
    assert_eq!(s.quarantine[held], Q_NONE, "the held sample is released by the region that now reaches it");
    assert_eq!(judged_step(&mut s, &near).stats.source, Source::Surrogate);
    let nearer = h2_with(INSIDE, &[[0.0, 3.9, 0.0]]);
    assert_eq!(judged_step(&mut s, &nearer).stats.source, Source::Fallback, "nearer still: not measured");
}

/// Final review I3 (ruling 13): the region covers a stray atom only as near a face as a passed check put one, and a
/// check voided by a box-affected solve is not queued again until the stray atom is farther from the faces than at
/// that void. The teacher here is box-affected within 4.5 Bohr of a face; the checks' stray H sits 5 Bohr from one.
#[test]
fn a_stray_atom_nearer_a_face_than_every_check_falls_back_and_a_boxed_void_waits() {
    let mut c = Composing::new(0.0);
    c.boxed_within = 4.5;
    let (mut s, sig) = cleared_student_of(&c);
    let at = |y: f64| h2_with(INSIDE, &[[0.0, y, 0.0]]);
    // 4.4 Bohr from a face: the floor (4) passes, no check measured it
    assert!(!cleared_at(&s, &at(5.6)), "nearer a face than every check: not covered");
    let r = judged_step(&mut s, &at(5.6));
    assert_eq!(r.stats.source, Source::Fallback, "{:?}", r.stats.notes);
    assert!(queued(&r), "{:?}", r.stats.notes);
    settle(&mut s);
    let ev = s.take_events();
    let void = ev.iter().find(|e| e.outcome == TeachOutcome::Discarded).unwrap_or_else(|| panic!("a void check: {ev:?}"));
    assert!(void.detail.contains("composition check void") && void.detail.contains("4.40 Bohr"), "{}", void.detail);
    // the same place and nearer the face: no check (one void per signature until the stray atom moves away)
    for y in [5.6, 5.7] {
        let r = judged_step(&mut s, &at(y));
        assert_eq!(r.stats.source, Source::Fallback);
        assert!(!queued(&r) && s.stray_pending.is_none() && s.side_queue.is_empty(), "y = {y}: {:?}", r.stats.notes);
    }
    // farther from the face than the void: checked again — 4.5 Bohr from a face is clean here, so it passes
    let r = judged_step(&mut s, &at(5.5));
    assert!(queued(&r), "{:?}", r.stats.notes);
    settle(&mut s);
    assert!(s.take_events().iter().any(|e| e.outcome == TeachOutcome::StrayCheckingNew));
    let Some(StrayState::Cleared { region, .. }) = s.stray_book.state(&sig) else { panic!() };
    assert!((region.place().face - 4.5).abs() < 1e-9, "{:?}", region.place());
    assert_eq!(judged_step(&mut s, &at(5.5)).stats.source, Source::Surrogate);
}

/// Final review minor 1 (ruling 13): a failed check takes back every sample of its signature in the fit — the ones its
/// region released included — as a failed audit does (spec §4.3: after a refusal no region covers them). The teacher's
/// composite offset (5 mHa) applies only with a stray atom within 7.5 Bohr of the bonded part, so the checks at 8.03
/// clear and a check with the stray H 7.03 Bohr away fails.
#[test]
fn a_failed_check_takes_back_what_its_region_released() {
    let mut c = Composing::new(5e-3);
    c.offset_near = 7.5;
    let (mut s, sig) = cleared_student_of(&c);
    let released: Vec<usize> = (0..s.samples()).filter(|&i| s.store().unwrap().samples[i].pos.len() == 3).collect();
    assert_eq!(released.len(), 3, "the premise: the checks' samples");
    assert!(released.iter().all(|&i| s.quarantine[i] == Q_NONE), "the premise: in the fit");
    let r = judged_step(&mut s, &h2_with(OUTSIDE, &[[0.0, 4.0, 0.0]]));
    assert!(queued(&r), "{:?}", r.stats.notes);
    settle(&mut s);
    assert!(matches!(s.stray_book.state(&sig), Some(StrayState::Refused(w)) if w.starts_with("energy")), "{:?}", s.stray_book.state(&sig));
    let ev = s.take_events();
    let refused = ev.iter().find(|e| e.outcome == TeachOutcome::StrayRefused).expect("a refusal");
    assert!(refused.detail.contains("taken back out"), "{}", refused.detail);
    assert!(released.iter().all(|&i| s.quarantine[i] == Q_STRAY), "{:?}", released.iter().map(|&i| s.quarantine[i]).collect::<Vec<_>>());
    assert!(s.pending_refit, "the refit is armed");
}

/// Final review minor 5 (ruling 13): a region was cleared under the policy's tolerances (the check's energy and force
/// tests, the thinning radius); a change of any of them drops every region (the signatures are checked again), never a
/// refusal. A change that judges no check (the audit cadence, draft mode) keeps them.
#[test]
fn a_policy_change_drops_the_regions_not_the_refusals() {
    let (mut s, sig) = cleared_student();
    let mut other = sig.clone();
    other.stray_z = vec![1, 1];
    s.stray_book.refuse(&other, "test".into());
    let p = *s.policy();
    s.set_policy(FallbackPolicy { audit_every: 7, ..p });
    s.set_policy(FallbackPolicy { never_fallback: true, ..p });
    s.set_policy(p);
    assert!(matches!(s.stray_book.state(&sig), Some(StrayState::Cleared { .. })), "the cadence and draft mode keep the region");
    s.set_policy(FallbackPolicy { energy_std_per_atom: 5e-4, ..p });
    assert!(s.stray_book.state(&sig).is_none(), "a tighter energy tier drops the region: {:?}", s.stray_book.state(&sig));
    assert!(matches!(s.stray_book.state(&other), Some(StrayState::Refused(_))), "a refusal stays");
    for q in [FallbackPolicy { force_std_abs: 1e-3, ..p }, FallbackPolicy { force_std_rel: 0.05, ..p }, FallbackPolicy { max_descriptor_distance: 2.0, ..p }] {
        let (mut s, sig) = cleared_student();
        s.set_policy(q);
        assert!(s.stray_book.state(&sig).is_none(), "{q:?}");
    }
}

/// Final review minor 8 (ruling 13): the answer identity E = E_model(bonded) + Σ E_ref needs every per-species offset 0
/// (mean 0); a model with a nonzero mean answers no stray-atom scene — in a release build too (a guard, not a
/// `debug_assert`): the step falls back.
#[test]
fn a_model_with_a_nonzero_mean_answers_no_stray_scene() {
    let (mut s, _) = cleared_student();
    let sc = h2_with(INSIDE, &[STRAY_AT]);
    assert!(cleared_at(&s, &sc), "the premise");
    for m in s.model.as_mut().unwrap().mean.iter_mut() {
        *m = 1e-3;
    }
    assert!(s.model().unwrap().effective_coefficients().1.iter().any(|&o| o != 0.0), "the premise: a per-species offset");
    assert!(!cleared_at(&s, &sc), "no cleared answer");
    s.discontinuity();
    assert_eq!(s.evaluate(&sc, &mut never()).unwrap().stats.source, Source::Fallback);
}

/// Final review I2(c) (ruling 13): the thinning radius of a check is frozen in the descriptor units of its pass, so a
/// later change of the model's training spreads (a refit) moves no coverage. Here the model's envelope is widened (H₂
/// 0.8–3.0 Bohr): a bond just beyond the 1.55 check's radius stays uncovered (stepped and held alike), and one within
/// it stays covered. (Narrower spreads cannot be shown on this fixture: the spread floor, 0.1 × the feature's RMS, already
/// holds the training units at their smallest.)
#[test]
fn the_thinning_radius_is_frozen_at_the_checks() {
    let (mut s, sig) = cleared_student();
    let basis = s.model().unwrap().basis.clone();
    let units = s.envelope().unwrap().clone();
    let thin = THIN_FRACTION * s.policy().max_descriptor_distance;
    let gap = |env: &Envelope, a: f64, b: f64| {
        let (da, sa) = h2_desc(&basis, a);
        let (db, _) = h2_desc(&basis, b);
        da.iter().zip(&db).zip(&sa).map(|((x, y), &k)| super::super::quarantine::spread_distance(env, k, x, y)).fold(0.0f64, f64::max)
    };
    let covered = |s: &SurrogateKs, r: f64| cleared_at(s, &h2_with(r, &[STRAY_AT]));
    let near = (1..=2000).map(|k| CHECKS[2] + 1e-4 * k as f64).find(|&r| !covers(&s, &sig, r) && gap(&units, r, CHECKS[2]) <= thin).expect("a bond in the radius");
    let beyond = (1..=2000).map(|k| CHECKS[2] + 1e-3 * k as f64).find(|&r| !covered(&s, r)).expect("a bond beyond");
    assert!(covered(&s, near) && !covered(&s, beyond), "the premise: {near} covered, {beyond} not");
    let mut wide = units.clone();
    for k in 0..=22 {
        let (d, sp) = h2_desc(&basis, 0.8 + 0.1 * k as f64);
        for (si, b) in sp.iter().zip(&d) {
            wide.add_atom(*si, b);
        }
    }
    assert!(gap(&wide, beyond, CHECKS[2]) <= thin, "the premise: in the wider spreads {beyond} would be within the radius");
    s.envelope = Some(wide);
    assert!(!covered(&s, beyond), "a wider envelope does not widen the region");
    assert!(covered(&s, near), "…nor moves what it covers");
    // the hold-out reads the same frozen radius: a held sample at `beyond` stays held under the wider spreads
    let sc = h2_with(beyond, &[STRAY_AT]);
    let (pos, z) = (positions(&sc), zs(&sc));
    let (e, f) = SpeciesMorse::energy_and_forces(&pos, &z);
    let prov = s.provenance_for(&sc);
    let mut notes = Vec::new();
    assert!(s.keep_sample(&pos, &z, e, &f, Some(0.0), Some(2), &prov, SampleTag::Fallback, &mut notes));
    assert_eq!(s.quarantine[s.samples() - 1], Q_STRAY, "{notes:?}");
    assert_eq!(s.release_stray_samples(&sig), 0);
}
