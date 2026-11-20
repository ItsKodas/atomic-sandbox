//! Isolated atoms are no training atoms (`surrogate::envelope` module doc, "Isolated atoms are not
//! training atoms"): a fitted sample with a stray atom (no neighbour within r_cut, in a scene whose
//! other atoms are bonded) must not put its zero descriptor into the fit's envelope — admitted, it
//! sets every coordination minimum of its species to 0, `coordination_ratio` skips the channel and
//! rule 5b is off for that species everywhere (the M6 Task 11 failure: a stretched pair just below
//! r_cut answered at ≈ Σ E_ref). An installed file whose fit used such samples holds them out and
//! rebuilds its fit without them (`tests/stray.rs`,
//! `an_installed_file_with_isolated_atoms_in_its_envelope_is_rebuilt`).
use super::*;

/// 60 H₂ Morse scan samples r ∈ [1.1, 2.9] Bohr and the stretched tail sample at r = 5.1 (f_c
/// 0.054; `one_tail_sample_no_longer_lets_a_free_atom_in`'s store), in the 65³ cell, audits off,
/// refitted. A stretched pair near r_cut is refused there by rule 5b.
fn tail_student() -> SurrogateKs {
    let mut s = SurrogateKs::new(Box::new(PairMorse(MorseDimer::h2())), FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() }).with_teacher_tol(1e-6);
    s.set_reference_energy(1, MorseDimer::h2().e_atom);
    let mut rng = StdRng::seed_from_u64(4);
    for i in 0..61 {
        let r = if i < 60 { 1.1 + 1.8 * i as f64 / 59.0 } else { 5.1 };
        let sc = in_cell_65(dimer(r, unit_vector(&mut rng), [0.0; 3], 1, 1));
        s.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
    }
    s.refit().unwrap();
    s
}

/// H₂ at r = 1.6 Bohr along x at the origin of the 65³ cell, optionally with a stray H at
/// y = 7 Bohr (7.05 Bohr from both atoms: no neighbour within r_cut = 6).
fn h2_maybe_stray(stray: bool) -> Scene {
    let mut sc = in_cell_65(h2(1.6));
    if stray {
        sc.nuclei.push(Nucleus::new(1, [0.0, 7.0, 0.0]));
    }
    sc
}

/// The teacher's answer at `sc` split into its bonded part and free atoms (the stray-atom spec's
/// premise, ruling 9a): the pair-additive Morse on the H₂ alone, plus E_atom and zero force for
/// each atom beyond y = 6 Bohr. The Morse's own answer is not split — its tail at 7.05 Bohr puts
/// ≈ −2.4 mHa into the label that no model within r_cut can see, which per N_e = 2 atoms (the
/// stray-atom energy row) is a 1.2 mHa/atom residual above `energy_std_per_atom`: the refit's
/// misfit recovery holds it out (`Q_HELD`, the r = 5.1 tail sample with it), so it would not be
/// fitted (per N = 3, before ruling 9a, it was 0.8 mHa/atom and stayed in).
fn pair_sample(sc: &Scene) -> Sample {
    let (bonded, stray): (Vec<&Nucleus>, Vec<&Nucleus>) = sc.nuclei.iter().partition(|n| n.pos[1] < 6.0);
    let part = Scene { nuclei: bonded.iter().map(|&n| n.clone()).collect(), ..sc.clone() };
    let r = PairMorse(MorseDimer::h2()).evaluate(&part, &mut never()).unwrap();
    let mut forces = r.forces;
    forces.extend(stray.iter().map(|_| [0.0; 3]));
    let energy = r.energy + stray.len() as f64 * MorseDimer::h2().e_atom;
    let nuclei: Vec<&Nucleus> = bonded.iter().chain(&stray).copied().collect();
    Sample { pos: nuclei.iter().map(|n| n.pos).collect(), z: nuclei.iter().map(|n| n.z).collect(), energy, forces, tag: SampleTag::Fallback, weight: 1.0, force_mask: None, use_energy: true }
}

/// The envelope's bits (min, max, Σ, Σ², counts).
fn envelope_bits(e: &Envelope) -> Vec<u64> {
    [&e.min, &e.max, &e.sum, &e.sumsq].iter().flat_map(|v| v.iter().map(|x| x.to_bits())).chain(e.count.iter().map(|&c| c as u64)).collect()
}

/// The strict verdicts (`would_answer`'s rule, or none) on a stretched, non-stray H₂ scan
/// r ∈ [4.5, 6) Bohr.
fn stretched_verdicts(s: &mut SurrogateKs) -> Vec<(f64, Option<TeachRule>)> {
    s.ensure_desc_cache();
    (0..15)
        .map(|k| {
            let r = 4.5 + 0.1 * k as f64;
            let sc = in_cell_65(h2(r));
            let (pos, z) = pos_z(&sc);
            (r, s.would_answer(&pos, &z, &s.provenance_for(&sc)).err().map(|e| e.0))
        })
        .collect()
}

/// A fitted sample with a stray atom leaves the training envelope bit for bit what the same sample
/// without its stray atom leaves, so the coordination minimum stays the bonded atoms' (the tail
/// sample's f_c(5.1)) and rule 5b judges a stretched, non-stray H₂ near r_cut exactly as before:
/// the verdicts of a scan r ∈ [4.5, 6) Bohr are unchanged by fitting it and 5b refuses where it
/// did before the sample.
#[test]
fn a_fitted_stray_atom_leaves_the_envelope_and_the_stretched_pair_alone() {
    let mut base = tail_student();
    let before = stretched_verdicts(&mut base);
    println!("stretched H₂ verdicts before: {before:?}");
    let coordination: Vec<f64> = before.iter().filter(|(_, v)| *v == Some(TeachRule::Coordination)).map(|(r, _)| *r).collect();
    assert!(!coordination.is_empty(), "the premise: rule 5b refuses the stretched pair near r_cut: {before:?}");
    assert!(before.iter().all(|(_, v)| v.is_some()), "the premise: strict mode refuses the stretched pair: {before:?}");
    let min_before = base.envelope().unwrap().min[base.model().unwrap().basis.body2_index(0, 0)];
    // the same student taught the bonded part alone, for the envelope's bits
    let mut plain = tail_student();
    let bonded = h2_maybe_stray(false);
    plain.add_sample(pair_sample(&bonded), &bonded).unwrap();
    let with = h2_maybe_stray(true);
    base.add_sample(pair_sample(&with), &with).unwrap();
    let i = base.samples() - 1;
    assert!(!base.excluded(i), "the premise: the stray sample is fitted");
    base.refit().unwrap();
    plain.refit().unwrap();
    assert_eq!((base.accumulated, plain.accumulated), (62, 62), "both fits hold the new sample");
    assert!(!base.excluded(i) && !base.excluded(i - 1), "the stray sample and the tail sample stay fitted (no misfit)");
    assert_eq!(envelope_bits(base.envelope().unwrap()), envelope_bits(plain.envelope().unwrap()), "the stray H adds nothing to the envelope");
    let k = base.model().unwrap().basis.body2_index(0, 0);
    assert_eq!(base.envelope().unwrap().min[k], min_before, "the coordination minimum stays the tail sample's");
    let after = stretched_verdicts(&mut base);
    println!("stretched H₂ verdicts after fitting the stray sample: {after:?}");
    assert_eq!(after, before, "the stretched pair's verdicts are unchanged by the fitted stray sample");
}
