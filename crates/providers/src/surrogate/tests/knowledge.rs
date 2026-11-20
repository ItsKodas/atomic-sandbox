//! M8a Task 9: the knowledge map's builder (`surrogate::knowledge`) on the pair-additive Morse
//! fixtures of `tests.rs` (species-blind: rule 1 is the model's, not the fixture's).
use super::*;
use sim::knowledge::Composition;
use sim::{atom_coverage, Confidence, KnowledgeMap};

/// A CH₄-like C + 4 H: C–H 1.95–2.15 Bohr along the tetrahedron's directions, each tilted by up
/// to ~0.1 rad, in the 65³ cell's middle.
pub(super) fn random_ch4(rng: &mut StdRng) -> Scene {
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

pub(super) fn h2_and_ch4(n_h2: usize, n_ch4: usize) -> SurrogateKs {
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

/// 2026-11-16, the live knowledge graph (`sim::knowledge` module doc, "New samples, waiting for
/// the refit"): samples kept since the last refit are pending — counted per pair, in the fitted
/// histogram's bins, by the same geometry rule, and per composition — and change nothing fitted:
/// the sample counts, the histograms (so every bin's taught/thin class), the confidences and the
/// elements stay the fitted samples'. A forgotten, a quarantined, a box-excluded and a
/// box-affected sample (its face density over the box rule, not yet coded `Q_BOX`: that is done
/// inside the next fit) is never pending, and reads as held; the refit takes the pending ones in.
#[test]
fn pending_samples_are_counted_apart_and_change_no_class() {
    let mut s = h2_and_ch4(120, 200);
    let m0 = s.knowledge_map().unwrap();
    assert_eq!((m0.pending, s.pending_samples()), (0, 0));
    assert!(m0.pairs.iter().all(|p| p.pending_samples == 0 && p.pending.len() == p.hist.counts.len() && p.pending.iter().all(|&n| n == 0)));

    // six H₂ at 1.45 Bohr and four CH₄, kept, not fitted
    let mut rng = StdRng::seed_from_u64(0x11e);
    for _ in 0..6 {
        let sc = in_cell_65(dimer(1.45, unit_vector(&mut rng), [0.0; 3], 1, 1));
        s.add_sample(pair_morse_sample(&sc), &sc).unwrap();
    }
    for _ in 0..4 {
        let sc = random_ch4(&mut rng);
        s.add_sample(pair_morse_sample(&sc), &sc).unwrap();
    }
    assert_eq!(s.unfitted(), 10, "premise: kept since the last refit");
    let m = s.knowledge_map().unwrap();
    assert_eq!((m.pending, s.pending_samples()), (10, 10));
    assert_eq!((m.samples, &m.elements), (m0.samples, &m0.elements), "nothing fitted moved");
    for (p, p0) in m.pairs.iter().zip(&m0.pairs) {
        assert_eq!((p.z, p.samples, &p.hist, p.confidence), (p0.z, p0.samples, &p0.hist, p0.confidence), "the pair's fitted counts and class");
    }
    let hh = m.pair(1, 1).unwrap();
    assert_eq!(hh.pending_samples, 10, "six H₂ and four CH₄ hold H–H within r_cut");
    assert_eq!(hh.pending.iter().sum::<u32>(), 6 + 4 * 6, "one H–H per H₂, six per CH₄");
    assert_eq!(hh.pending[14], 6, "the H₂ at 1.45 Bohr: bin 14");
    let ch = m.pair(1, 6).unwrap();
    assert_eq!((ch.pending_samples, ch.pending.iter().sum::<u32>()), (4, 16), "four C–H per CH₄");
    assert_eq!(m.pair(6, 6).unwrap().pending_samples, 0);
    let h2 = m.molecule(&Composition::of(&[1, 1])).unwrap();
    assert_eq!((h2.fitted, h2.waiting, h2.pending), (120, 6, 6), "the molecules count them as new");
    let ch4 = m.molecule(&Composition::of(&[6, 1, 1, 1, 1])).unwrap();
    assert_eq!((ch4.fitted, ch4.pending), (200, 4));

    // forgotten, quarantined, box-excluded and box-affected: never pending (the last four are CH₄)
    let n = s.samples();
    s.quarantine.resize(n, Q_NONE);
    for (k, q) in [Q_FORGOTTEN, Q_HELD, Q_BOX].into_iter().enumerate() {
        s.quarantine[n - 1 - k] = q;
    }
    s.face.resize(n, f64::NAN);
    s.face[n - 4] = 1.0;
    assert!(s.is_box(n - 4) && !s.excluded(n - 4), "premise: box-affected, not yet coded");
    let m2 = s.knowledge_map().unwrap();
    assert_eq!((m2.pending, s.pending_samples()), (6, 6));
    let ch = m2.pair(1, 6).unwrap();
    assert_eq!((ch.pending_samples, ch.pending.iter().sum::<u32>()), (0, 0));
    assert_eq!(m2.pair(1, 1).unwrap().pending_samples, 6);
    let ch4 = m2.molecule(&Composition::of(&[6, 1, 1, 1, 1])).unwrap();
    assert_eq!((ch4.pending, ch4.waiting, ch4.held, ch4.forgotten), (0, 0, 3, 1), "the box-affected one reads held");

    // the refit: no longer pending, now fitted
    s.refit().unwrap();
    let m3 = s.knowledge_map().unwrap();
    assert_eq!((m3.pending, s.pending_samples()), (0, 0));
    assert!(m3.pairs.iter().all(|p| p.pending_samples == 0 && p.pending.iter().all(|&n| n == 0)));
    assert!(m3.samples > m0.samples, "the refit fitted them: {} vs {}", m3.samples, m0.samples);
}

/// The live Teach next (2026-11-16): the pending samples are counted per environment key and per
/// class the way the per-key verdict counts its fitted samples — so the refit that takes them in
/// raises every key's and class's fitted count by exactly its pending count — and a forgotten or
/// box-affected sample counts toward no key.
#[test]
fn pending_samples_are_counted_per_key_like_the_fitted_ones() {
    let mut s = h2_and_ch4(120, 200);
    let m0 = s.knowledge_map().unwrap();
    assert!(m0.pending_keys.is_empty() && m0.pending_classes.is_empty());
    let before: BTreeMap<(&str, u8, Vec<u8>), usize> = s.class_loo().iter().flat_map(|c| c.species.iter().map(move |v| ((c.class, v.z, v.neighbours.clone()), v.samples))).collect();
    let class_before: BTreeMap<&str, usize> = s.class_loo().iter().map(|c| (c.class, c.samples)).collect();
    // six H₂ and four CH₄ kept, then a fifth CH₄ forgotten and a sixth box-affected
    let mut rng = StdRng::seed_from_u64(0x7ea);
    for _ in 0..6 {
        let sc = in_cell_65(dimer(1.45, unit_vector(&mut rng), [0.0; 3], 1, 1));
        s.add_sample(pair_morse_sample(&sc), &sc).unwrap();
    }
    for _ in 0..6 {
        let sc = random_ch4(&mut rng);
        s.add_sample(pair_morse_sample(&sc), &sc).unwrap();
    }
    let n = s.samples();
    s.quarantine.resize(n, Q_NONE);
    s.quarantine[n - 1] = Q_FORGOTTEN;
    s.face.resize(n, f64::NAN);
    s.face[n - 2] = 1.0;
    let m = s.knowledge_map().unwrap();
    println!("pending keys {:?}; classes {:?}", m.pending_keys, m.pending_classes);
    assert_eq!(m.pending, 10);
    assert_eq!(m.pending_key("many-body", 6, &[1]), 4, "C with H: the four kept CH₄");
    assert_eq!(m.pending_key("many-body", 1, &[1, 6]), 4, "H with H, C: their hydrogens");
    assert_eq!(m.pending_keys.iter().map(|k| k.samples).sum::<usize>(), 6 + 4 + 4, "each sample once per key it holds");
    assert_eq!(m.pending_key("pair", 1, &[1]), 6, "H with H: the six H₂");
    assert_eq!((m.pending_class("many-body"), m.pending_class("pair")), (4, 6));
    // the refit takes them in: every fitted count rises by its pending count (the box-affected
    // sample is forgotten too here, so the fit's own box rule, which waits for the trust floor,
    // cannot take it in)
    (s.face[n - 2], s.quarantine[n - 2]) = (f64::NAN, Q_FORGOTTEN);
    s.refit().unwrap();
    for c in s.class_loo() {
        assert_eq!(c.samples, class_before.get(c.class).copied().unwrap_or(0) + m.pending_class(c.class), "class {}", c.class);
        for v in &c.species {
            let was = before.get(&(c.class, v.z, v.neighbours.clone())).copied().unwrap_or(0);
            assert_eq!(v.samples, was + m.pending_key(c.class, v.z, &v.neighbours), "{} {}", c.class, v.words());
        }
    }
}
