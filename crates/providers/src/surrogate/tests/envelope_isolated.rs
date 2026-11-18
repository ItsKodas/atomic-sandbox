//! Isolated atoms are no training atoms (`surrogate::envelope` module doc, "Isolated atoms are not
//! training atoms"): a kept rule-5a fallback with a stray atom (no neighbour within r_cut, in a
//! scene whose other atoms are bonded) must not put its zero descriptor into the fit's envelope —
//! admitted, it sets every coordination minimum of its species to 0, `coordination_ratio` skips
//! the channel and rule 5b is off for that species everywhere (the M6 Task 11 failure: a stretched
//! pair just below r_cut answered at ≈ Σ E_ref).
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

/// The teacher's exact answer at `sc` (the pair-additive Morse: the stray H adds E_atom and no
/// force at 7 Bohr only up to the Morse tail, which is what a kept fallback holds).
fn pair_sample(sc: &Scene) -> Sample {
    let r = PairMorse(MorseDimer::h2()).evaluate(sc, &mut never()).unwrap();
    Sample { pos: sc.nuclei.iter().map(|n| n.pos).collect(), z: sc.nuclei.iter().map(|n| n.z).collect(), energy: r.energy, forces: r.forces, tag: SampleTag::Fallback, weight: 1.0, force_mask: None, use_energy: true }
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
    assert_eq!(envelope_bits(base.envelope().unwrap()), envelope_bits(plain.envelope().unwrap()), "the stray H adds nothing to the envelope");
    let k = base.model().unwrap().basis.body2_index(0, 0);
    assert_eq!(base.envelope().unwrap().min[k], min_before, "the coordination minimum stays the tail sample's");
    let after = stretched_verdicts(&mut base);
    println!("stretched H₂ verdicts after fitting the stray sample: {after:?}");
    assert_eq!(after, before, "the stretched pair's verdicts are unchanged by the fitted stray sample");
}

/// A model file written before isolated atoms were left out of the envelope (simulated: an isolated
/// H added to the stored envelope, as that build's fit of a kept stray sample did) installs with
/// its envelope rebuilt from the fitted samples' descriptors (no refit, no DFT): bit for bit the
/// envelope of a fit of the same samples today, in the provider and in the accumulator a save
/// writes, with a note. A file whose fit holds no isolated atom keeps its stored envelope.
#[test]
fn an_installed_file_with_isolated_atoms_in_its_envelope_is_rebuilt() {
    let mut s = tail_student();
    let with = h2_maybe_stray(true);
    s.add_sample(pair_sample(&with), &with).unwrap();
    s.refit().unwrap();
    let want = envelope_bits(s.envelope().unwrap());
    let path = std::env::temp_dir().join(format!("envelope-isolated-{}.model", std::process::id()));
    s.save_sidecar(&path).unwrap();
    let clean = surrogate::sidecar::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    // an older build's stored envelope: the stray sample's isolated H in it
    let mut old = clean.clone();
    let p = old.accumulator.basis.n_features();
    old.accumulator.envelope.add_atom(0, &vec![0.0; p]);
    let k = old.accumulator.basis.body2_index(0, 0);
    assert_eq!(old.accumulator.envelope.min[k], 0.0, "the premise: the stored envelope admits an isolated atom");
    let (mut t, _) = trained(1.2, 2.2, 0, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    t.install_sidecar(old).unwrap();
    assert!(t.model().is_some() && t.samples() == 62, "installed");
    assert_eq!(envelope_bits(t.envelope().unwrap()), want, "the installed envelope is today's fit's: no isolated atom");
    assert_eq!(envelope_bits(&t.accumulator.as_ref().unwrap().envelope), want, "and the accumulator's (what a save writes, what a refit extends)");
    assert!(t.user_notes.iter().any(|n| n.contains("isolated atom") && n.contains("envelope was rebuilt")), "{:?}", t.user_notes);
    // a clean file: its stored envelope stands, no note
    let (mut u, _) = trained(1.2, 2.2, 0, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    u.install_sidecar(clean).unwrap();
    assert_eq!(envelope_bits(u.envelope().unwrap()), want);
    assert!(!u.user_notes.iter().any(|n| n.contains("envelope was rebuilt")), "{:?}", u.user_notes);
}

/// The rebuilt envelope is the accumulator's too, so the fit grows from it: after installing an
/// older file (its stored envelope admitting an isolated atom) one more sample and an incremental
/// refit leave the envelope bit for bit that of a fresh accumulator over every member, in order.
#[test]
fn an_incremental_refit_after_the_rebuild_matches_a_fresh_fit() {
    let mut s = tail_student();
    let with = h2_maybe_stray(true);
    s.add_sample(pair_sample(&with), &with).unwrap();
    s.refit().unwrap();
    let path = std::env::temp_dir().join(format!("envelope-isolated-grow-{}.model", std::process::id()));
    s.save_sidecar(&path).unwrap();
    let mut old = surrogate::sidecar::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let p = old.accumulator.basis.n_features();
    old.accumulator.envelope.add_atom(0, &vec![0.0; p]);
    let (mut t, _) = trained(1.2, 2.2, 0, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    t.install_sidecar(old).unwrap();
    assert_eq!(t.accumulated, 62);
    // one more sample, fitted on the incremental path (no species or reference change)
    let more = in_cell_65(h2(1.45));
    t.add_sample(pair_sample(&more), &more).unwrap();
    assert!(!t.species_dirty && !t.refs_dirty, "the premise: the refit adds to the installed accumulator");
    t.refit().unwrap();
    assert_eq!((t.accumulated, t.accumulator.as_ref().unwrap().n_samples), (63, 63));
    // a fresh accumulator over the same members, in store order
    let model = t.model().unwrap();
    let basis = model.basis.clone();
    let refs = model.ref_energy.clone();
    let st = t.store().unwrap();
    let members: Vec<&Sample> = (0..t.accumulated).filter(|&i| !t.excluded(i)).map(|i| &st.samples[i]).collect();
    assert_eq!(members.len(), 63);
    let mut fresh = surrogate::Accumulator::new(&basis);
    fresh.add_many(&members, &basis, &refs, &FitOptions::default()).unwrap();
    assert_eq!(envelope_bits(t.envelope().unwrap()), envelope_bits(&fresh.envelope), "the provider's envelope after the incremental refit");
    assert_eq!(envelope_bits(&t.accumulator.as_ref().unwrap().envelope), envelope_bits(&fresh.envelope), "and its accumulator's");
    let k = basis.body2_index(0, 0);
    assert!(fresh.envelope.min[k] > 0.0, "no isolated atom in it ({:e})", fresh.envelope.min[k]);
}
