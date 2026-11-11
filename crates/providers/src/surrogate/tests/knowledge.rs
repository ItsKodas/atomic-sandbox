//! M8a Task 9: the knowledge map's builder (`surrogate::knowledge`) on the pair-additive Morse
//! fixtures of `tests.rs` (species-blind: rule 1 is the model's, not the fixture's).
use super::*;
use sim::{atom_coverage, Confidence, KnowledgeMap};

/// A CH₄-like C + 4 H: C–H 1.95–2.15 Bohr along the tetrahedron's directions, each tilted by up
/// to ~0.1 rad, in the 65³ cell's middle.
fn random_ch4(rng: &mut StdRng) -> Scene {
    let mut sc = in_cell_65(dimer(2.0, [1.0, 0.0, 0.0], [0.0; 3], 6, 1));
    sc.nuclei.clear();
    sc.nuclei.push(Nucleus::new(6, [0.0; 3]));
    let t = 1.0 / 3f64.sqrt();
    for d in [[t, t, t], [t, -t, -t], [-t, t, -t], [-t, -t, t]] {
        let r = rng.gen_range(1.95..2.15);
        let tilt = unit_vector(rng);
        let v: [f64; 3] = std::array::from_fn(|a| d[a] + 0.1 * tilt[a]);
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        sc.nuclei.push(Nucleus::new(1, std::array::from_fn(|a| r * v[a] / n)));
    }
    sc
}

/// The pair's confidence read the independent way: the model's trust rule and every per-key
/// verdict the Model tab lists (`class_loo().species`) whose central species sees the other.
fn confidence_from_class_loo(s: &SurrogateKs, m: &KnowledgeMap, a: u8, b: u8) -> Confidence {
    let p = m.pair(a, b).unwrap();
    if p.samples == 0 {
        return Confidence::Unseen;
    }
    let keys: Vec<&sim::SpeciesVerdict> = s
        .class_loo()
        .iter()
        .flat_map(|c| c.species.iter())
        .filter(|v| (v.z == a && v.neighbours.contains(&b)) || (v.z == b && v.neighbours.contains(&a)))
        .collect();
    assert!(!keys.is_empty(), "a seen pair is met in a key");
    if s.trusted() && keys.iter().all(|v| v.trusted) {
        Confidence::Trusted
    } else {
        Confidence::Thin
    }
}

fn h2_and_ch4(n_h2: usize, n_ch4: usize) -> SurrogateKs {
    let mut s = SurrogateKs::new(Box::new(PairMorse(MorseDimer::h2())), FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() }).with_teacher_tol(1e-6);
    for z in [1, 6] {
        s.set_reference_energy(z, MorseDimer::h2().e_atom);
    }
    let mut rng = StdRng::seed_from_u64(0x9c4);
    for i in 0..n_h2 {
        let sc = in_cell_65(dimer(1.2 + 2.6 * i as f64 / (n_h2 - 1) as f64, unit_vector(&mut rng), [0.0; 3], 1, 1));
        s.add_sample(pair_morse_sample(&sc), &sc).unwrap();
    }
    for _ in 0..n_ch4 {
        let sc = random_ch4(&mut rng);
        s.add_sample(pair_morse_sample(&sc), &sc).unwrap();
    }
    s.refit().unwrap();
    s
}

/// The plan's first test: a collection of H₂ and CH₄ samples. H–H and C–H are seen, with their
/// sample counts (every CH₄ holds H–H at ≈ 3.4 Bohr < r_cut); C–C is `Unseen`; C's coordination
/// within r_cut is 4 in every sample; its shell angles (C–H's first peak) sit at the tetrahedral
/// 109.5°; the confidences are the per-key verdicts' (read independently from `class_loo`); and
/// held-out samples (any quarantine code, the provider-local ones included) are not counted.
#[test]
fn knowledge_counts_pairs_and_marks_unseen() {
    let (n_h2, n_ch4) = (120, 200);
    let mut s = h2_and_ch4(n_h2, n_ch4);
    let m = s.knowledge_map().expect("a fitted model");
    println!(
        "trusted {} ({:?}); pairs {:?}; keys {:?}",
        s.trusted(),
        s.untrusted_reason(),
        m.pairs.iter().map(|p| (p.z, p.samples, p.confidence)).collect::<Vec<_>>(),
        key_counts(&s)
    );
    assert_eq!(m.r_cut, s.cutoff());
    assert_eq!((m.samples, m.slug.as_str()), (n_h2 + n_ch4, ""), "a per-scene model has no slug");
    assert_eq!(m.elements.iter().map(|e| (e.z, e.atoms_seen)).collect::<Vec<_>>(), vec![(1, 2 * n_h2 + 4 * n_ch4), (6, n_ch4)]);
    assert_eq!(m.pairs.iter().map(|p| (p.z, p.samples)).collect::<Vec<_>>(), vec![((1, 1), n_h2 + n_ch4), ((1, 6), n_ch4), ((6, 6), 0)]);
    let cc = m.pair(6, 6).unwrap();
    assert_eq!(cc.confidence, Confidence::Unseen);
    assert!(cc.hist.counts.iter().all(|&c| c == 0) && cc.hist.counts.len() == (m.r_cut / 0.1).ceil() as usize);
    let ch = m.pair(6, 1).unwrap();
    assert_eq!(ch.hist.counts.iter().sum::<u32>() as usize, 4 * n_ch4, "four C–H per CH₄");
    for (a, b) in [(1, 1), (1, 6), (6, 6)] {
        assert_eq!(m.pair(a, b).unwrap().confidence, confidence_from_class_loo(&s, &m, a, b), "({a}, {b})");
    }
    let c = m.element(6).unwrap();
    assert_eq!(c.coordination.iter().enumerate().max_by_key(|(_, &n)| n).map(|(k, _)| k), Some(4));
    assert_eq!(c.coordination[4] as usize, n_ch4, "every C has its four H within r_cut");
    let peak = c.angles.iter().enumerate().max_by_key(|(_, &n)| n).unwrap().0;
    println!("C angles (5° bins): {:?}", c.angles);
    assert!((20..=22).contains(&peak), "the tetrahedral angle's bin: {peak}");
    assert_eq!(c.angles.iter().sum::<u32>() as usize, 6 * n_ch4, "six H–C–H angles per CH₄, all in the C–H shell");
    assert!(c.angles[..18].iter().chain(&c.angles[27..]).all(|&n| n == 0), "nothing below 90° or above 135°");

    // three CH₄ held out (a held quarantine, a lone-atom and a reference-wait code): not knowledge
    let n = s.samples();
    s.quarantine.resize(n, Q_NONE);
    let last = s.accumulated - 1;
    for (k, q) in [Q_HELD, Q_LONE, Q_REF_WAIT].into_iter().enumerate() {
        s.quarantine[last - k] = q;
    }
    let m2 = s.knowledge_map().unwrap();
    assert_eq!(m2.samples, n_h2 + n_ch4 - 3);
    assert_eq!(m2.element(6).unwrap().atoms_seen, n_ch4 - 3);
    assert_eq!(m2.pair(1, 6).unwrap().samples, n_ch4 - 3);
}

/// The plan's second test, on a built map: an atom with a neighbour of a pair the collection never
/// saw (ethane's C–C on the H₂ + CH₄ map) reads 0; a CH₄ like the training ones reads well above
/// it; an element the collection lacks reads 0.
#[test]
fn atom_coverage_is_low_for_an_unseen_pair() {
    let s = h2_and_ch4(120, 200);
    let m = s.knowledge_map().unwrap();
    let ch4 = random_ch4(&mut StdRng::seed_from_u64(5));
    let (z, pos): (Vec<u8>, Vec<[f64; 3]>) = ch4.nuclei.iter().map(|n| (n.z, n.pos)).unzip();
    let known = atom_coverage(&m, &z, &pos);
    // ethane: the second carbon 2.9 Bohr along x with its own H's mirrored
    let (mut z2, mut pos2) = (z.clone(), pos.clone());
    for (zi, p) in z.iter().zip(&pos) {
        z2.push(*zi);
        pos2.push([2.9 - p[0], p[1], p[2]]);
    }
    let ethane = atom_coverage(&m, &z2, &pos2);
    let water = atom_coverage(&m, &[8, 1, 1], &[[0.0; 3], [1.43, 1.1, 0.0], [-1.43, 1.1, 0.0]]);
    println!("CH₄ {known:?}; C₂H₆ {ethane:?}; H₂O {water:?}");
    let want_known = match m.pair(1, 6).unwrap().confidence.min_with(m.pair(1, 1).unwrap().confidence) {
        Confidence::Trusted => 0.9,
        _ => 0.45,
    };
    assert!(known.iter().all(|&c| c > want_known), "a training-like CH₄: {known:?}");
    assert_eq!((ethane[0], ethane[5]), (0.0, 0.0), "the carbons see a C–C neighbour: {ethane:?}");
    assert!(ethane.iter().all(|&c| c < known.iter().copied().fold(1.0, f32::min)), "every ethane atom is less known than methane's least-known");
    assert_eq!(water[0], 0.0, "O is unknown");
}

trait MinWith {
    fn min_with(self, o: Confidence) -> Confidence;
}

impl MinWith for Confidence {
    /// The weaker of two confidences.
    fn min_with(self, o: Confidence) -> Confidence {
        let rank = |c: Confidence| match c {
            Confidence::Trusted => 2,
            Confidence::Thin => 1,
            Confidence::Unseen => 0,
        };
        if rank(self) <= rank(o) { self } else { o }
    }
}

/// The controller's ruling on Task 9: a pair met only in a short key reads `Thin`, not `Trusted`.
/// H₂ dimers, H₃ trimers and N₂ dimers (120 each) and three NH₃ — NH₃'s keys (N{H}, H{H, N}) are
/// short on their own sample floor (`a_key_counts_only_the_samples_holding_an_atom_of_it`), so
/// N–H is `Thin` in a model that is trusted and answers N–N.
#[test]
fn a_pair_whose_key_is_short_reads_thin() {
    let mut s = SurrogateKs::new(Box::new(PairMorse(MorseDimer::h2())), FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() }).with_teacher_tol(1e-6);
    for z in [1, 7] {
        s.set_reference_energy(z, MorseDimer::h2().e_atom);
    }
    let mut rng = StdRng::seed_from_u64(0x4e3);
    let add = |s: &mut SurrogateKs, sc: Scene| s.add_sample(pair_morse_sample(&sc), &sc).unwrap();
    for (z0, z1) in [(1, 1), (7, 7)] {
        for i in 0..120 {
            add(&mut s, in_cell_65(dimer(1.2 + 2.6 * i as f64 / 119.0, unit_vector(&mut rng), [0.0; 3], z0, z1)));
        }
    }
    for _ in 0..120 {
        let sc = random_h3(&mut rng);
        add(&mut s, sc);
    }
    for _ in 0..3 {
        let sc = random_nh3(&mut rng);
        add(&mut s, sc);
    }
    s.refit().unwrap();
    let m = s.knowledge_map().unwrap();
    println!(
        "trusted {} ({:?}); pairs {:?}; keys {:?}",
        s.trusted(),
        s.untrusted_reason(),
        m.pairs.iter().map(|p| (p.z, p.samples, p.confidence)).collect::<Vec<_>>(),
        key_counts(&s)
    );
    assert!(s.trusted(), "premise: the model answers: {:?}", s.untrusted_reason());
    let nh = m.pair(1, 7).unwrap();
    assert_eq!((nh.samples, nh.confidence), (3, Confidence::Thin), "three NH₃: short keys");
    assert_eq!(m.pair(7, 7).unwrap().confidence, Confidence::Trusted, "N₂'s own key is learned");
    // H–H is met in NH₃'s short H{H, N} too: not known everywhere it was met (module doc)
    assert_eq!(m.pair(1, 1).unwrap().confidence, Confidence::Thin);
    for (a, b) in [(1, 1), (1, 7), (7, 7)] {
        assert_eq!(m.pair(a, b).unwrap().confidence, confidence_from_class_loo(&s, &m, a, b), "({a}, {b})");
    }
}
