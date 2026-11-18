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
        frontier: [Frontier { homo: Some(-0.37), lumo: Some(0.02) }, Frontier { homo: Some(-0.37), lumo: Some(0.02) }],
        partial: Vec::new(),
        clean: true,
    };
    let atom = PartAnswer {
        energy: -0.479,
        forces: vec![[0.0, 1e-4, 0.0]],
        frontier: [Frontier { homo: Some(-0.27), lumo: Some(0.05) }, Frontier { homo: None, lumo: Some(-0.05) }],
        partial: Vec::new(),
        clean: true,
    };
    let scene = PartAnswer {
        energy: frag.energy + atom.energy,
        forces: vec![frag.forces[0], frag.forces[1], atom.forces[0]],
        frontier: [Frontier { homo: Some(-0.27), lumo: Some(0.02) }, Frontier { homo: Some(-0.37), lumo: Some(-0.05) }],
        partial: Vec::new(),
        clean: true,
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
    atoms[0].frontier = [Frontier { homo: Some(-0.33), lumo: Some(0.1) }; 2];
    atoms[0].partial = vec![o2p(0), o2p(1)];
    scene.frontier = [Frontier { homo: Some(-0.33), lumo: Some(0.02) }; 2];
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
    frag.frontier[0] = Frontier { homo: Some(-0.5), lumo: Some(0.3) };
    frag.partial = vec![PartialLevel { spin: 0, index: 0, eps: -0.5, f: 0.6 }];
    atoms[0].frontier[0] = Frontier { homo: Some(0.0), lumo: Some(0.4) };
    atoms[0].partial = vec![PartialLevel { spin: 0, index: 0, eps: 0.0, f: 0.6 }];
    scene.frontier[0] = Frontier { homo: Some(0.0), lumo: Some(0.3) };
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
        book.record(sig(), v, &d, &s, &basis, &env, thin());
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
    book.record(sig(), CheckVerdict::Pass { residual: 5e-4, margin: 0.25 }, &d, &s, &basis, &env, thin());
    let Some(StrayState::Cleared { region, residual, margin }) = book.state(&sig()) else { panic!() };
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
        book.record(sig(), PASS, &d, &s, &basis, &env, thin());
    }
    assert!(matches!(book.state(&sig()), Some(StrayState::Checking { passes: 1, .. })), "{:?}", book.state(&sig()));
    // nor within the thinning radius of it
    let (near, _) = h2_desc(&basis, 1.451);
    book.record(sig(), PASS, &near, &s, &basis, &env, thin());
    assert!(matches!(book.state(&sig()), Some(StrayState::Checking { passes: 1, .. })));
    // and a species the training never saw cannot be measured for distinctness: never counted
    let empty = Envelope::new(&basis);
    let (far, _) = h2_desc(&basis, 1.9);
    book.record(sig(), PASS, &far, &s, &basis, &empty, thin());
    assert!(matches!(book.state(&sig()), Some(StrayState::Checking { passes: 1, .. })));
    // a void changes nothing
    let (d2, _) = h2_desc(&basis, 1.55);
    book.record(sig(), CheckVerdict::Void("x".into()), &d2, &s, &basis, &env, thin());
    assert!(matches!(book.state(&sig()), Some(StrayState::Checking { passes: 1, .. })));
}

#[test]
fn region_contains_its_points_and_not_beyond() {
    let basis = Basis::new(&[1, 8]);
    let mut region = Region::new(&basis);
    let (d0, s) = h2_desc(&basis, 1.45);
    assert!(!region.contains(&d0, &s), "an empty region contains nothing");
    for r in [1.35, 1.45, 1.55] {
        let (d, s) = h2_desc(&basis, r);
        region.add(&d, &s);
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
    let env = units(&basis);
    assert!(!region.distinct(&d0, &s, &env, thin()));
    let (far, _) = h2_desc(&basis, 1.9);
    assert!(region.distinct(&far, &s, &env, thin()));
}

#[test]
fn one_fail_refuses_for_good() {
    let basis = Basis::new(&[1, 8]);
    let env = units(&basis);
    let mut book = StrayBook::default();
    let geoms: Vec<_> = [1.35, 1.45, 1.55, 1.25].iter().map(|&r| h2_desc(&basis, r)).collect();
    book.record(sig(), PASS, &geoms[0].0, &geoms[0].1, &basis, &env, thin());
    book.record(sig(), CheckVerdict::Fail("energy: 2.00 mHa".into()), &geoms[1].0, &geoms[1].1, &basis, &env, thin());
    assert_eq!(book.state(&sig()).map(|s| matches!(s, StrayState::Refused(w) if w.contains("energy"))), Some(true));
    for (d, s) in &geoms {
        book.record(sig(), PASS, d, s, &basis, &env, thin());
    }
    assert!(matches!(book.state(&sig()), Some(StrayState::Refused(_))), "passes after a fail do not count");
    // a cleared signature is refused by one fail too
    let mut other = sig();
    other.stray_z = vec![1, 1];
    for (d, s) in &geoms[..3] {
        book.record(other.clone(), PASS, d, s, &basis, &env, thin());
    }
    assert!(matches!(book.state(&other), Some(StrayState::Cleared { .. })));
    book.record(other.clone(), CheckVerdict::Fail("margin".into()), &geoms[3].0, &geoms[3].1, &basis, &env, thin());
    assert!(matches!(book.state(&other), Some(StrayState::Refused(_))));
    // a refusal by hand (the floor, a failed audit), on a signature never checked
    let mut third = sig();
    third.charge = 1;
    book.refuse(&third, "charge".into());
    assert!(matches!(book.state(&third), Some(StrayState::Refused(w)) if w == "charge"));
    // the first signature is untouched by the others
    assert!(matches!(book.state(&sig()), Some(StrayState::Refused(w)) if w.contains("energy")));
}
