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
    let mut out = [Frontier { homo: None, lumo: None }; 2];
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
/// is added whenever the scene has a stray atom — a composite error the parts alone never show (the spin or
/// charge-transfer case the check exists for). `open` gates the answer (the auditor's instance), `calls` counts solves.
#[derive(Clone)]
struct Composing {
    offset: f64,
    open: Arc<AtomicBool>,
    calls: Arc<AtomicUsize>,
}

impl Composing {
    fn new(offset: f64) -> Composing {
        Composing { offset, open: Arc::new(AtomicBool::new(true)), calls: Arc::new(AtomicUsize::new(0)) }
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
        if split(&pos, r_cut).is_some() {
            energy += self.offset;
        }
        let electrons = z.iter().map(|&x| x as i64).sum::<i64>() - scene.charge as i64;
        let mut stats = EvalStats::default();
        stats.energy.total = energy;
        stats.face_density = Some(0.0);
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
    let mut s = SurrogateKs::new(Box::new(Composing::new(offset)), FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() }).with_teacher_tol(1e-6);
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
        s = s.with_auditor(Box::new(Composing { offset, open: gate.clone(), calls: calls.clone() }));
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
        region.add(&d, &sp);
        for &rb in &rs[a + 1..] {
            let (d, sp) = h2_desc(&basis, rb);
            assert!(region.distinct(&d, &sp, &units, thin), "the premise: H–H {ra} and {rb} are distinct geometries");
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
    assert_eq!(ev.last().map(|e| (e.rule, e.outcome)), Some((TeachRule::IsolatedAtom, TeachOutcome::Kept)), "{ev:?}");
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
    let Some(StrayState::Cleared { region, residual, margin }) = s.stray_book.state(&sig) else { panic!("cleared: {:?}", s.stray_book.state(&sig)) };
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
        s.stray_book.record(sig.clone(), CheckVerdict::Pass { residual: 0.0, margin: 0.32 }, &d, &sp, &basis, &units, thin);
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

/// Whether `sig`'s region covers the bonded H₂ at H–H `r`.
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
    let (mut s, _, _) = student(0.0, &TRAINED, 40, true);
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
    let Some(StrayState::Cleared { residual, margin, .. }) = s.stray_book.state(sig) else { panic!("not cleared") };
    format!(
        "stray atoms: model for the bonded part + free atoms (composition measured, 3 checks; worst residual {:.2} mHa, frontier margin {:.0} mHa)",
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
    let sc = h2_with(OUTSIDE, &[STRAY_AT]);
    let r = judged_step(&mut s, &sc);
    assert_eq!(r.stats.source, Source::Fallback);
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
}

/// Plan Task 4, review I4: rule 7 normalises per **bonded** atom. With the energy tier at 0.5 mHa per atom (and the
/// force tier at 1e-2 Ha/Bohr, so rule 6 does not decide first), the model's posterior factor is scaled so that at the
/// cleared geometry the epistemic energy std √(e² − (3 s_E)²) is 1.25 mHa: 0.42 mHa per atom over the scene's 3 atoms
/// (it would pass), 0.62 mHa over the bonded part's 2 (it fails). The step falls back on rule 7, and the read-only
/// verdicts agree; rule 6 passes and the stray atom's force std is exactly 0 (asserted).
#[test]
fn rule7_uses_n_bonded() {
    let (mut s, _) = cleared_student();
    let p = *s.policy();
    s.set_policy(FallbackPolicy { energy_std_per_atom: 5e-4, force_std_abs: 1e-2, ..p });
    assert!(s.trusted(), "the premise: {:?}", s.untrusted_reason());
    let sc = h2_with(INSIDE, &[STRAY_AT]);
    let (pos, z) = (positions(&sc), zs(&sc));
    let epistemic = |s: &SurrogateKs| {
        let m = s.model().unwrap();
        let p = m.predict(&pos, &z, true).unwrap();
        (p.energy_std.unwrap().powi(2) - (3.0 * m.sigma_e).powi(2)).sqrt()
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
    assert!(s.energy_variance_refusal(&m, 3, 3, e, &infl).is_none(), "the premise: it passes per scene atom");
    assert!(s.energy_variance_refusal(&m, 3, 2, e, &infl).is_some(), "the premise: it fails per bonded atom");
    let v = s.variance_check_n(&m, &pred, e, &f, &pos, &z, 3);
    assert!(v.accept && v.atoms[2] == 0.0, "the premise: rules 6–7 per scene atom pass, the stray atom's std is 0: {:?}", v.atoms);
    assert_eq!(s.would_answer(&pos, &z, &s.provenance_for(&sc)), Err((TeachRule::EnergyUncertainty, None)));
    let r = judged_step(&mut s, &sc);
    assert_eq!(r.stats.source, Source::Fallback);
    assert!(r.stats.notes.iter().any(|n| n.contains("epistemic energy std 6.25e-4 > 5.00e-4 Ha/atom")), "{:?}", r.stats.notes);
    // the variance cadence: a verdict reached while the stray atom was not exempt (here near a face: rule 5a refuses,
    // and the variance, per scene atom, accepts) is not reused for the next, cleared step at the same neighbour counts
    let near_face = h2_with(INSIDE, &[[0.0, 6.5, 0.0]]);
    assert_eq!(judged_step(&mut s, &near_face).stats.source, Source::Fallback);
    assert!(s.last_variance.as_ref().is_some_and(|v| v.accept), "the premise: an accepting verdict per scene atom");
    let back = h2_with(INSIDE, &[[0.0, 5.9, 0.0]]);
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
/// ordinary rules refuse and the teacher answers on its own surface. Every model ↔ teacher switch carries a booking
/// (`refit_jump`, from `transition_jump`), and E_conserved has no unbooked step: per step, ΔE_reported − J +
/// ½ (F₀ + F₁)·Δr — what the driver's E_conserved moves by beyond the bounded shadow term — stays at the trapezoid's
/// error, far below the 10 mHa an unbooked switch would leave. No sample flood: the way back out keeps no copy of the
/// way in's fallbacks.
#[test]
fn reentry_and_exit_book_every_switch() {
    let (mut s, _) = cleared_student();
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
    assert_eq!(s.samples(), at_turn, "no sample flood: the way out revisits the way in's geometries, and keeps none of them");
}

/// Review Focus 3: a stray atom drifting toward a face during a cleared run falls back as soon as it is within the
/// floor's 4 Bohr of a face — the floor is re-checked at every step, the reason shown in the step's notes — and the
/// model answers again once it is back.
#[test]
fn a_stray_near_a_face_falls_back_inside_a_cleared_region() {
    let (mut s, _) = cleared_student();
    // the 81³ cell's half-extent is 10 Bohr: at y = 5.9 the stray H is 4.1 Bohr from a face, at 6.5 3.5 Bohr
    for (y, model) in [(5.9, true), (6.5, false), (5.9, true)] {
        let sc = h2_with(INSIDE, &[[0.0, y, 0.0]]);
        let r = judged_step(&mut s, &sc);
        assert_eq!(r.stats.source == Source::Surrogate, model, "y = {y}: {:?}", r.stats.notes);
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
