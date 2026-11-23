//! Chemical channels Tasks 5–6 (`represent` module doc; spec
//! `docs/superpowers/specs/2026-11-22-chemical-channels-design.md`, "Physics review 2, 2026-11-23"
//! item 10; review `docs/superpowers/research/2026-11-23-chemical-channels-task4-task5-review.md`
//! T1–T6): refuse rather than widen. The teacher is the pair-Morse [`SpeciesMorse`] (synthetic
//! samples where DFT would be needed), the store an H/Li/O/Cl (and S) channel collection shaped like
//! the day-1 gate's: Li and Cl only in diatomics (unrepresented), H and O in water (represented).
use super::*;
use surrogate::channels::LearntMix;
use surrogate::EnvKey;

const BOX: usize = 41;
const H: u8 = 1;
const LI: u8 = 3;
const O: u8 = 8;
const S: u8 = 16;
const CL: u8 = 17;

/// `atoms` centred on their centroid, each coordinate jittered by ±0.06 Bohr, in the 41³ cell.
fn mol(atoms: &[(u8, [f64; 3])], rng: &mut StdRng) -> Scene {
    let n = atoms.len() as f64;
    let c: [f64; 3] = std::array::from_fn(|a| atoms.iter().map(|(_, p)| p[a]).sum::<f64>() / n);
    let mut sc = in_cell(dimer(1.0, [1.0, 0.0, 0.0], [0.0; 3], H, H), BOX);
    sc.nuclei = atoms.iter().map(|&(z, p)| Nucleus::new(z, std::array::from_fn(|a| p[a] - c[a] + rng.gen_range(-0.06..0.06)))).collect();
    sc
}

fn bent(a: u8, b: u8, c: u8, r1: f64, r2: f64, deg: f64) -> Vec<(u8, [f64; 3])> {
    let t = deg.to_radians();
    vec![(a, [0.0; 3]), (b, [r1, 0.0, 0.0]), (c, [r2 * t.cos(), r2 * t.sin(), 0.0])]
}

fn water(rng: &mut StdRng) -> Scene {
    let (r1, r2, deg) = (rng.gen_range(1.72..1.92), rng.gen_range(1.72..1.92), rng.gen_range(98.0..112.0));
    mol(&bent(O, H, H, r1, r2, deg), rng)
}

fn diatomic(a: u8, b: u8, r: f64, rng: &mut StdRng) -> Scene {
    mol(&[(a, [0.0; 3]), (b, [r + rng.gen_range(-0.12..0.12), 0.0, 0.0])], rng)
}

/// H–O–Cl (O has the neighbours H and Cl).
fn hocl(rng: &mut StdRng) -> Scene {
    mol(&bent(O, H, CL, rng.gen_range(1.75..1.9), rng.gen_range(3.1..3.3), rng.gen_range(100.0..106.0)), rng)
}

/// Water with an HCl whose Cl sits 3.2 Bohr from one water H (that H has the neighbours O and Cl).
fn water_hcl(rng: &mut StdRng) -> Scene {
    let mut a = bent(O, H, H, 1.8, 1.8, 105.0);
    a.push((CL, [1.8 + rng.gen_range(3.1..3.3), 0.0, 0.0]));
    a.push((H, [1.8 + 3.2 + rng.gen_range(2.3..2.5), 0.4, 0.0]));
    mol(&a, rng)
}

/// Li–O–H, near linear (O has the neighbours Li and H).
fn lioh(rng: &mut StdRng) -> Scene {
    mol(&bent(O, H, LI, rng.gen_range(1.75..1.9), rng.gen_range(3.1..3.3), rng.gen_range(170.0..180.0)), rng)
}

fn h2s(rng: &mut StdRng) -> Scene {
    mol(&bent(S, H, H, rng.gen_range(2.45..2.6), rng.gen_range(2.45..2.6), rng.gen_range(89.0..95.0)), rng)
}

/// H–O–S–H: O with the neighbours H and S, a combination of represented species no sample holds.
fn hosh(rng: &mut StdRng) -> Scene {
    let mut a = bent(O, H, S, 1.8, 3.1, 105.0);
    let t = 105f64.to_radians();
    a.push((H, [3.1 * t.cos() + 2.5, 3.1 * t.sin() + 0.3, 0.0]));
    mol(&a, rng)
}

fn geometry(sc: &Scene) -> (Vec<[f64; 3]>, Vec<u8>) {
    (sc.nuclei.iter().map(|n| n.pos).collect(), sc.nuclei.iter().map(|n| n.z).collect())
}

/// The teacher's answer at `sc`, stored without the policy (no thinning, no no-model cap), with a
/// small label scatter that is a function of the geometry alone (a noiseless label set sends the
/// evidence search after σ → 0 for thousands of solves; the same geometry keeps the same label, so
/// the bit-identity comparisons hold).
fn teach(s: &mut SurrogateKs, sc: &Scene) {
    let (pos, z) = geometry(sc);
    let (mut e, mut f) = SpeciesMorse::energy_and_forces(&pos, &z);
    let seed: f64 = pos.iter().flatten().enumerate().map(|(i, x)| x * (i as f64 + 1.0)).sum();
    let jitter = |k: f64| (seed * 7919.0 + k * 104.729).sin();
    e += 2e-5 * jitter(0.0);
    for (j, fj) in f.iter_mut().enumerate() {
        for (a, c) in fj.iter_mut().enumerate() {
            *c += 2e-4 * jitter(1.0 + (3 * j + a) as f64);
        }
    }
    s.add_sample_answer(Sample::new(pos, z, e, f, SampleTag::Scan, 1.0), sc, Some(0.0), None).unwrap();
}

/// A channel collection of K = 3 asked (body 4 off and no trust recovery: fast fits) at `dir`, its
/// 41³ references solved;
/// `learn`: the provider learns its own W (else a test's mix stands).
fn channel_student(dir: &std::path::Path, learn: bool, rng: &mut StdRng) -> SurrogateKs {
    let (mut s, _) = pair_student();
    s.body4_off = true;
    // the trust recovery's fits are not what these tests pin (and dominate their time)
    s.skip_recovery = true;
    s.mix_learn_off = !learn;
    s.attach_refit_off = true;
    Trainable::set_model_basis(&mut s, sim::ModelBasis::Channels(3)).unwrap();
    s.attach_collection(dir, &water(rng)).unwrap();
    for z in [H, LI, O, S, CL] {
        s.set_reference_solved(z, SpeciesMorse::atom(z), BOX, Some(0.0));
    }
    s
}

/// A mix over H, Li, O, Cl at K_eff 2: the identity on H and O, Li and Cl zero rows (the day-1
/// gate's W). `flags`: Li and Cl flagged represented anyway — Task 4's semantics, where a zero row
/// was a value and nothing refused — or not (Task 5's).
fn gate_mix(li_cl_flagged: bool) -> LearntMix {
    LearntMix {
        k: 2,
        k_requested: 3,
        species: vec![H, LI, O, CL],
        w: vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        sigma_w: Vec::new(),
        singular_values: vec![2.0, 1.0, 0.0, 0.0],
        truncation_residual: vec![0.0, 1.0, 0.0, 1.0],
        nonlinearity: 0.0,
        weights: vec![1.0; 4],
        rank: 2,
        low_order: None,
        represented: vec![true, li_cl_flagged, true, li_cl_flagged],
        n_w_met: vec![true, li_cl_flagged, true, li_cl_flagged],
        many_body: vec![1_000_000; 4],
        history_from: 0,
    }
}

/// Water and the diatomics HCl, LiH, ClO, LiO (seeded).
fn teach_base(s: &mut SurrogateKs) {
    let mut rng = StdRng::seed_from_u64(91);
    for i in 0..60 {
        teach(s, &water(&mut rng));
        if i % 3 == 0 {
            teach(s, &diatomic(H, CL, 2.4, &mut rng));
            teach(s, &diatomic(LI, H, 3.0, &mut rng));
            teach(s, &diatomic(O, CL, 3.0, &mut rng));
            teach(s, &diatomic(LI, O, 3.2, &mut rng));
        }
    }
}

/// HOCl, H₂O·HCl and LiOH (seeded), after the base.
fn teach_many_body(s: &mut SurrogateKs) {
    let mut rng = StdRng::seed_from_u64(92);
    for _ in 0..10 {
        teach(s, &hocl(&mut rng));
        teach(s, &water_hcl(&mut rng));
        teach(s, &lioh(&mut rng));
    }
}

fn coeff_bits(s: &SurrogateKs) -> Vec<u64> {
    s.model().unwrap().coeffs.iter().flatten().map(|x| x.to_bits()).collect()
}

/// Review T1 and T3: HOCl, H₂O·HCl and LiOH are refused by rule 3b (an atom of two or more
/// neighbours with Cl or Li among them), where 5d and 5e alone would not have fired — under Task 4's
/// semantics (Li and Cl zero rows but nothing refusing) at a geometry the fit holds the novelty is
/// ν ≤ 1 and the grade γ ≤ γ_max, the posterior's own blind spot; and those samples, kept, leave the
/// channel fit (`Q_UNREP`) so the represented keys — water's — are fitted bit for bit as without
/// them (coefficients, report, per-key counts and floors, a water prediction).
#[test]
fn a_many_body_atom_with_an_unrepresented_neighbour_is_refused() {
    let root = collection_root("channels-t1");
    let mut rng = StdRng::seed_from_u64(90);
    let probes = {
        let mut r = StdRng::seed_from_u64(92);
        // the first of each kind `teach_many_body` stores: a geometry the fit holds
        (hocl(&mut r), water_hcl(&mut r), lioh(&mut r))
    };
    let probes = [(probes.0, CL), (probes.1, CL), (probes.2, LI)];

    // Task 4's semantics: the same zero rows, flagged represented — the HOCl samples are fitted
    let mut old = channel_student(&root.join("old"), false, &mut rng);
    old.set_collection_mix(gate_mix(true)).unwrap();
    teach_base(&mut old);
    teach_many_body(&mut old);
    old.refit().expect("Task 4's fit");
    assert!(old.quarantine.iter().all(|&q| q != Q_UNREP));
    for (sc, _) in &probes {
        let (pos, z) = geometry(sc);
        let (nu, _) = old.novelty_judged(&pos, &z, None).expect("a null space");
        let gamma = old.extrapolation_grade(&pos, &z).expect("a grade");
        assert!(nu <= 1.0 && gamma <= GRADE_MAX, "5d/5e would pass a geometry the fit holds: ν {nu:.3e}, γ {gamma:.3}");
    }

    // Task 5: Li and Cl unrepresented
    let mut new = channel_student(&root.join("new"), false, &mut rng);
    new.set_collection_mix(gate_mix(false)).unwrap();
    teach_base(&mut new);
    let base = new.samples();
    teach_many_body(&mut new);
    new.refit().expect("Task 5's fit");
    assert!(new.quarantine[..base].iter().all(|&q| q != Q_UNREP), "the base is not excluded");
    assert!(new.quarantine[base..].iter().all(|&q| q == Q_UNREP), "the many-body Cl and Li samples are kept, out of the channel fit");
    for (sc, zu) in &probes {
        let (pos, z) = geometry(sc);
        let prov = new.provenance_for(sc);
        match new.would_answer(&pos, &z, &prov) {
            Err((TeachRule::UnrepresentedNeighbour(got), Some(_))) => assert_eq!(got, *zu),
            v => panic!("rule 3b must refuse: {v:?}"),
        }
    }
    // the words name the species and its N_W progress (no census: learning off, so the plain form)
    let (pos, z) = geometry(&probes[0].0);
    let g = new.gate(&pos, &z, &new.provenance_for(&probes[0].0)).unwrap_err();
    assert!(g.reason.text.contains("one is Cl, which the chemical channels do not represent yet") && g.keep_sample && !g.untrusted, "{}", g.reason.text);

    // T3: the same fit as a store without them (water's coefficients bit for bit)
    let mut plain = channel_student(&root.join("plain"), false, &mut rng);
    plain.set_collection_mix(gate_mix(false)).unwrap();
    teach_base(&mut plain);
    plain.refit().expect("the base fit");
    let codes = |s: &SurrogateKs| (0..base).map(|i| s.quarantine.get(i).copied().unwrap_or(Q_NONE)).collect::<Vec<u8>>();
    assert_eq!(codes(&plain), codes(&new), "the same codes on the base");
    assert_eq!(coeff_bits(&plain), coeff_bits(&new), "excluded samples move no coefficient");
    let (rp, rn) = (plain.report().unwrap(), new.report().unwrap());
    assert_eq!((rp.n_samples, rp.lambda.to_bits(), rp.sigma_f.to_bits(), rp.q_prime), (rn.n_samples, rn.lambda.to_bits(), rn.sigma_f.to_bits(), rn.q_prime));
    let (jp, jn) = (plain.class_judge.as_ref().unwrap(), new.class_judge.as_ref().unwrap());
    assert_eq!((&jp.key_q, jp.fit_columns), (&jn.key_q, jn.fit_columns), "n_k and q_k unchanged");
    let w = water(&mut rng);
    let (pos, z) = geometry(&w);
    let (a, b) = (plain.model().unwrap().predict(&pos, &z, true).unwrap(), new.model().unwrap().predict(&pos, &z, true).unwrap());
    assert_eq!((a.energy.to_bits(), a.energy_std.map(f64::to_bits)), (b.energy.to_bits(), b.energy_std.map(f64::to_bits)), "water's prediction unchanged");
}

/// Review T2: a pair atom with an unrepresented neighbour is judged as before — HCl and LiH
/// predictions and verdicts are bit-identical to Task 4's model on the same store (no sample
/// excluded: none has a many-body atom next to Li or Cl).
#[test]
fn pair_atoms_with_an_unrepresented_neighbour_still_answer() {
    let root = collection_root("channels-t2");
    let mut rng = StdRng::seed_from_u64(93);
    let mut fitted = Vec::new();
    for flagged in [true, false] {
        let mut s = channel_student(&root.join(if flagged { "old" } else { "new" }), false, &mut rng);
        s.set_collection_mix(gate_mix(flagged)).unwrap();
        teach_base(&mut s);
        s.refit().unwrap();
        assert!(s.quarantine.iter().all(|&q| q != Q_UNREP), "no many-body atom next to Li or Cl");
        fitted.push(s);
    }
    assert_eq!(fitted[0].quarantine, fitted[1].quarantine);
    assert_eq!(coeff_bits(&fitted[0]), coeff_bits(&fitted[1]));
    let mut r = StdRng::seed_from_u64(94);
    for sc in [diatomic(H, CL, 2.4, &mut r), diatomic(LI, H, 3.0, &mut r), diatomic(O, CL, 3.0, &mut r)] {
        let (pos, z) = geometry(&sc);
        let prov = fitted[0].provenance_for(&sc);
        let p: Vec<_> = fitted.iter().map(|s| s.model().unwrap().predict(&pos, &z, true).unwrap()).collect();
        let bits = |q: &surrogate::Prediction| (q.energy.to_bits(), q.forces.iter().flatten().map(|x| x.to_bits()).collect::<Vec<_>>(), q.energy_std.map(f64::to_bits));
        assert_eq!(bits(&p[0]), bits(&p[1]), "{z:?}");
        let v: Vec<_> = fitted.iter().map(|s| s.would_answer(&pos, &z, &prov)).collect();
        assert_eq!(v[0], v[1], "{z:?}");
        assert!(!matches!(v[1], Err((TeachRule::UnrepresentedNeighbour(_), _))), "{z:?}: {:?}", v[1]);
    }
}

/// Review T4 and T6, the lifecycle: S thinned to a few samples is unrepresented (N_W unmet) after
/// the first learn — W from H and O only, S's row zero — and its many-body scenes go to the teacher
/// with their samples kept; once H₂S samples bring a many-body key with S among the neighbours to
/// its low-order floor, the next refit relearns W with S represented, clears the class memory and
/// moves the evidence floor past the earlier W's points. Then an unseen combination of represented
/// species (O with the neighbours H and S) is refused by its key's n_k = 0 floor.
#[test]
fn a_rare_element_is_unrepresented_until_n_w_then_relearns() {
    let root = collection_root("channels-t4");
    let mut rng = StdRng::seed_from_u64(95);
    let mut s = channel_student(&root.join("life"), true, &mut rng);
    let mut r = StdRng::seed_from_u64(96);
    for _ in 0..90 {
        teach(&mut s, &water(&mut r));
    }
    for _ in 0..3 {
        teach(&mut s, &h2s(&mut r));
    }
    s.refit().expect("the first fit, with the first learn");
    let mix = s.collection().unwrap().read_mix().unwrap().expect("W learnt and written");
    assert_eq!(mix.species, vec![H, O, S]);
    assert_eq!((mix.represented.clone(), mix.n_w_met.clone()), (vec![true, true, false], vec![true, true, false]), "S: N_W unmet");
    assert_eq!(mix.w[2 * mix.k..].iter().map(|x| x.to_bits()).collect::<Vec<_>>(), vec![0u64; mix.k], "S's row is exactly 0");
    assert!(s.user_notes.iter().any(|n| n.contains("chemical channels: W first learnt: H, O met N_W")), "{:?}", s.user_notes);
    let first_from = mix.history_from;
    assert_eq!(s.history_floor(), first_from);
    assert!(s.quarantine[90..93].iter().all(|&q| q == Q_UNREP), "H₂S: out of the channel fit");
    // an H₂S scene: rule 3b, the teacher's answer kept, its N_W progress said
    let _ = Trainable::take_teach_events(&mut s);
    let sc = h2s(&mut r);
    let out = s.evaluate(&sc, &mut never()).unwrap();
    assert_eq!(out.stats.source, Source::Fallback);
    assert!(out.stats.notes.iter().any(|n| n.contains("one is S, which the chemical channels do not represent yet (its best many-body environment has")), "{:?}", out.stats.notes);
    let ev = Trainable::take_teach_events(&mut s);
    let last = ev.last().expect("the teacher's row");
    assert_eq!(last.rule, TeachRule::UnrepresentedNeighbour(S));
    // kept, unless a stored H₂S is as close as rule 5's thinning distance (the same keep rule)
    assert!(matches!(last.outcome, sim::activity::TeachOutcome::Kept | sim::activity::TeachOutcome::Thinned), "{:?}", last.outcome);
    // enough H₂S for a many-body key with S among its neighbours to meet its low-order floor
    let low = surrogate::channels::LowOrder::REVIEW_I4.basis(&[H, O, S], surrogate::COLLECTION_R_CUT);
    let (pos, z) = geometry(&sc);
    let probe = Sample::new(pos, z, 0.0, vec![[0.0; 3]; 3], SampleTag::Scan, 1.0);
    let floor = super::super::trust::key_column_counts(&low, [&probe]).per_key.iter().filter(|(k, _)| k.class == 2 && k.neighbours.contains(&S)).map(|(_, &(_, q))| q).min().unwrap();
    for _ in 0..floor + 4 {
        teach(&mut s, &h2s(&mut r));
    }
    // the relearn clears the class memory (a planted entry stands for the standing verdicts)
    let planted = EnvKey { class: 2, z: O, neighbours: vec![H] };
    s.class_memory.insert(planted.clone(), (true, Some(1)));
    s.channel_lifecycle();
    assert!(s.class_memory.is_empty(), "every key re-enters at ENTER");
    let gen = s.model_gen;
    s.refit().expect("the refit on the relearnt W");
    let mix = s.collection().unwrap().read_mix().unwrap().unwrap();
    assert_eq!(mix.represented, vec![true, true, true], "S represented: {:?}", s.user_notes);
    assert_eq!((mix.k, mix.w.clone()), (3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]), "K covers H, O, S: the identity");
    assert!(s.user_notes.iter().any(|n| n.contains("relearnt: S first met N_W")), "{:?}", s.user_notes);
    assert_eq!(mix.history_from, gen + 1);
    assert!(mix.history_from > first_from && s.history_floor() == mix.history_from);
    assert!(s.quarantine.iter().all(|&q| q != Q_UNREP), "the H₂S samples are back in the channel fit");
    let sc = h2s(&mut r);
    let (pos, z) = geometry(&sc);
    assert!(!matches!(s.would_answer(&pos, &z, &s.provenance_for(&sc)), Err((TeachRule::UnrepresentedNeighbour(_), _))));
    // T6: O{H, S} — both species represented, the combination never fitted: the n_k = 0 floor
    let sc = hosh(&mut r);
    let (pos, z) = geometry(&sc);
    assert!(s.unrepresented_neighbour(&pos, &z).is_none());
    let key = EnvKey { class: 2, z: O, neighbours: vec![H, S] };
    assert!(!s.class_judge.as_ref().unwrap().key_q.contains_key(&key), "the premise: no fitted atom holds O{{H, S}}");
    let (refusal, samples_floor) = s.class_untrusted(&[Some(key)]).expect("refused");
    assert!(samples_floor, "on the sample floor: {}", refusal.text);
}

/// Review C2 (spec "Physics review 2" item 6): with a dense W every atom touches its centre's whole
/// channel block, but a key's floor is the smaller of that and its species-basis count — exactly
/// today's floor for a key over at most K species, never above it; the species basis is
/// `key_column_counts` bit for bit.
#[test]
fn c2_floors_keep_todays_floor_up_to_k_species() {
    use super::super::trust::{key_column_counts, key_column_counts_c2};
    let mut rng = StdRng::seed_from_u64(97);
    let samples: Vec<Sample> = (0..6)
        .flat_map(|_| [water(&mut rng), h2s(&mut rng), hosh(&mut rng)])
        .map(|sc| {
            let (pos, z) = geometry(&sc);
            Sample::new(pos, z.clone(), 0.0, vec![[0.0; 3]; z.len()], SampleTag::Scan, 1.0)
        })
        .collect();
    let species = Basis::new(&[H, O, S]);
    assert_eq!(key_column_counts_c2(&species, &samples), key_column_counts(&species, &samples));
    let dense = species.clone().with_channels(surrogate::ChannelMix { k: 2, w: vec![0.8, 0.6, -0.6, 0.8, 0.5, 0.5] });
    let (ch, sp, c2) = (key_column_counts(&dense, &samples), key_column_counts(&species, &samples), key_column_counts_c2(&dense, &samples));
    let water_o = EnvKey { class: 2, z: O, neighbours: vec![H] };
    let (n, q) = c2.per_key[&water_o];
    assert_eq!((n, q), sp.per_key[&water_o], "one neighbour species ≤ K: today's floor");
    assert!(ch.per_key[&water_o].1 > q, "the premise: a dense W touches more channel columns ({} > {q})", ch.per_key[&water_o].1);
    for (k, &(n, q)) in &c2.per_key {
        assert_eq!(n, sp.per_key[k].0);
        assert_eq!(q, ch.per_key[k].1.min(sp.per_key[k].1), "{k:?}");
    }
    assert_eq!(c2.union, ch.union.min(sp.union));
}
