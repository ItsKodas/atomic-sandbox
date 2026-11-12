//! Rule 0b, lone atoms (user request 2026-11-12; `lone` module doc): in strict mode a scene in
//! which no atom has a neighbour within r_cut is answered from the isolated-atom references —
//! E = Σ E_ref, every force 0, `Source::Reference`, no teacher call — whatever the model's trust,
//! and stands aside (today's path) when the references cannot answer: a charged scene, a missing
//! reference, a multiplicity other than the atoms' reference states aligned, mixed species.
use super::*;
use sim::AtomReference;

/// One H at a deliberately off-centre spot of a 33³ cell (|x| ≤ 3.25 inside the 3h margin).
fn lone_h() -> Scene {
    let mut sc = h2(1.4);
    sc.nuclei = vec![Nucleus::new(1, [1.3, -0.7, 2.1])];
    sc
}

/// Two H `d` apart along x at y = z = 0, atom 0 at x = −3, with multiplicity `m`.
fn two_h(d: f64, m: Option<u32>) -> Scene {
    let mut sc = h2(1.4);
    sc.nuclei = vec![Nucleus::new(1, [-3.0, 0.0, 0.0]), Nucleus::new(1, [-3.0 + d, 0.0, 0.0])];
    sc.multiplicity = m;
    sc
}

/// The species-Morse teacher's exact answer as a scan sample.
fn species_sample(sc: &Scene) -> Sample {
    let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
    let z: Vec<u8> = sc.nuclei.iter().map(|n| n.z).collect();
    let (energy, forces) = SpeciesMorse::energy_and_forces(&pos, &z);
    Sample { pos, z, energy, forces, tag: SampleTag::Scan, weight: 1.0, force_mask: None, use_energy: true }
}

/// A per-scene student of the species Morse with the H reference and `n` H₂ scan samples
/// (1.1–2.3 Bohr, random orientations), not refitted.
fn h_student(n: usize) -> (SurrogateKs, Arc<Mutex<SpeciesMorse>>) {
    let (mut s, t) = pair_student();
    s.set_reference_energy(1, SpeciesMorse::atom(1));
    let mut rng = StdRng::seed_from_u64(0x10e);
    for i in 0..n {
        let sc = h2_at(1.1 + 1.2 * i as f64 / (n.max(2) - 1) as f64, &mut rng);
        s.add_sample(species_sample(&sc), &sc).unwrap();
    }
    (s, t)
}

fn species_calls(t: &Arc<Mutex<SpeciesMorse>>) -> usize {
    t.lock().unwrap().calls
}

/// Asserts `r` is the rule's answer: the source, E bit for bit, F = 0, the note, the model surface.
fn assert_reference(r: &ForceResult, energy: f64, what: &str) {
    assert_eq!(r.stats.source, Source::Reference, "{what}: {:?}", r.stats.notes);
    assert_eq!(r.energy.to_bits(), energy.to_bits(), "{what}: E = {} against Σ E_ref = {energy}", r.energy);
    assert_eq!(r.stats.energy.total.to_bits(), energy.to_bits(), "{what}");
    assert!(r.forces.iter().all(|f| *f == [0.0; 3]), "{what}: {:?}", r.forces);
    assert!(r.display.is_none(), "{what}: no density");
    assert!(r.stats.notes.first().is_some_and(|n| n.contains("lone atoms: answered from the atom references")), "{what}: {:?}", r.stats.notes);
    assert_eq!(r.stats.energy_surface(), Surface::Model, "{what}");
}

/// The brief's first test: a lone H at a non-centre position, with no model (a store and its H
/// reference, no fit) and then with an untrusted one — E = E_ref(H) exactly, F = 0,
/// `Source::Reference`, no teacher call, no teaching event, counted as its own kind (not a solve).
#[test]
fn a_lone_h_is_answered_from_the_reference_with_no_model_or_an_untrusted_one() {
    let (mut s, t) = h_student(3);
    assert!(s.model().is_none());
    let before = s.solve_counts();
    let r = s.evaluate(&lone_h(), &mut never()).unwrap();
    assert_reference(&r, SpeciesMorse::atom(1), "no model");
    assert_eq!(species_calls(&t), 0, "no teacher call");
    assert!(s.take_events().is_empty(), "no teaching event");
    let after = s.solve_counts();
    assert_eq!((after.total(), after.reference), (before.total(), 1), "its own kind, not a solve");
    assert_eq!(s.samples(), 3, "nothing kept");

    // an untrusted fit: enough samples for a model, too few for trust
    let (mut s, t) = h_student(12);
    s.refit().expect("a fit forms");
    assert!(s.model().is_some() && !s.trusted(), "untrusted: {:?}", s.untrusted_reason());
    let r = s.evaluate(&lone_h(), &mut never()).unwrap();
    assert_reference(&r, SpeciesMorse::atom(1), "untrusted model");
    assert_eq!(species_calls(&t), 0);
    // the model's own answer at a lone geometry, bit for bit (`surrogate::fit`, "Isolated-atom limit")
    let pos = [[1.3, -0.7, 2.1]];
    let p = s.model().unwrap().predict(&pos, &[1], false).unwrap();
    assert_eq!(p.energy.to_bits(), r.energy.to_bits());
    assert_eq!(p.forces, r.forces);
    // the readout says strict mode answers here, though the model is untrusted
    let ready = s.readiness(&lone_h()).unwrap();
    assert!(ready.answers && ready.text.contains("atom references"), "{ready:?}");
    println!("lone H: E = {:.9} Ha (E_ref), untrusted: {:?}", r.energy, s.untrusted_reason());
}

/// A trusted model: lone atoms are still the rule's (the model is not consulted, the MorseDimer
/// teacher — which cannot solve one atom — never called); H₂ at 1.4 Bohr takes the normal strict
/// path, its answer the model's prediction bit for bit with no lone-atom note.
#[test]
fn a_trusted_model_leaves_lone_atoms_to_the_references_and_h2_unchanged() {
    let (mut s, h) = trained(1.2, 2.2, 40, FallbackPolicy::default());
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    let n0 = calls(&h);
    let r = s.evaluate(&lone_h(), &mut never()).unwrap();
    assert_reference(&r, MorseDimer::h2().e_atom, "trusted model");
    assert_eq!(calls(&h), n0, "no teacher call");
    let p = s.model().unwrap().predict(&[[1.3, -0.7, 2.1]], &[1], false).unwrap();
    assert_eq!(p.energy.to_bits(), r.energy.to_bits(), "the model's own answer");

    s.discontinuity();
    let sc = h2(1.4);
    let r = s.evaluate(&sc, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Surrogate, "{:?}", r.stats.notes);
    let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
    let p = s.model().unwrap().predict(&pos, &[1, 1], false).unwrap();
    assert_eq!(r.energy.to_bits(), p.energy.to_bits(), "H₂: the model's energy, bit for bit");
    assert_eq!(r.forces, p.forces, "H₂: the model's forces, bit for bit");
    assert!(!r.stats.notes.iter().any(|n| n.contains("lone")), "{:?}", r.stats.notes);
    assert_eq!(calls(&h), n0);
    assert_eq!(s.solve_counts().reference, 1);
}

/// Two H beyond r_cut (6.2 Bohr): answered at the parallel-spin product M = 3 (E = 2 E_ref);
/// under the automatic multiplicity (1 for two electrons) or a forced singlet the teacher answers —
/// LSDA's singlet there is the restricted state, 4.6e-2 Ha above 2 E_ref at 6.4 Bohr.
#[test]
fn two_lone_h_are_answered_only_at_the_parallel_spin_multiplicity() {
    let (mut s, t) = h_student(12);
    s.refit().expect("a fit forms");
    let r_cut = s.cutoff();
    assert!(6.2 > r_cut, "r_cut {r_cut}");
    let e2 = 2.0 * SpeciesMorse::atom(1);
    let r = s.evaluate(&two_h(6.2, Some(3)), &mut never()).unwrap();
    assert_reference(&r, e2, "two H, triplet");
    assert_eq!(species_calls(&t), 0);
    for (m, words) in [(None, "multiplicity 1 (automatic)"), (Some(1), "multiplicity 1 is not")] {
        let r = s.evaluate(&two_h(6.2, m), &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback, "M {m:?}: {:?}", r.stats.notes);
        assert!(r.stats.notes.iter().any(|n| n.contains("lone atoms, not answered from the atom references") && n.contains(words)), "M {m:?}: {:?}", r.stats.notes);
    }
    assert_eq!(species_calls(&t), 2, "both refusals went to the teacher");
    // inside r_cut the rule does not apply (and says nothing)
    let r = s.evaluate(&two_h(5.0, Some(3)), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback);
    assert!(!r.stats.notes.iter().any(|n| n.contains("lone")), "{:?}", r.stats.notes);
}

/// Charged lone atoms (H⁻, H⁺), a species without a reference, and a provider with no surface
/// pinned at all fall back with the reason noted.
#[test]
fn charged_lone_atoms_and_missing_references_fall_back() {
    let (mut s, t) = h_student(3);
    for (q, words) in [(-1, "charge is -1"), (1, "charge is +1")] {
        let mut sc = lone_h();
        sc.charge = q;
        let r = s.evaluate(&sc, &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback, "charge {q}: {:?}", r.stats.notes);
        assert!(r.stats.notes.iter().any(|n| n.contains(words)), "charge {q}: {:?}", r.stats.notes);
    }
    let mut he = lone_h();
    he.nuclei[0].z = 2;
    let r = s.evaluate(&he, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback, "{:?}", r.stats.notes);
    assert!(r.stats.notes.iter().any(|n| n.contains("no isolated-atom reference of He")), "{:?}", r.stats.notes);
    // mixed species beyond r_cut stand aside even with both references
    s.set_reference_energy(2, SpeciesMorse::atom(2));
    let mut hhe = two_h(6.2, Some(2));
    hhe.nuclei[1].z = 2;
    let r = s.evaluate(&hhe, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback, "{:?}", r.stats.notes);
    assert!(r.stats.notes.iter().any(|n| n.contains("more than one species")), "{:?}", r.stats.notes);
    assert_eq!(species_calls(&t), 4);
    assert_eq!(s.samples(), 3, "lone answers are never kept");
    // nothing learned or loaded: no references on any surface
    let (mut fresh, t) = pair_student();
    let r = fresh.evaluate(&lone_h(), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback);
    assert!(r.stats.notes.iter().any(|n| n.contains("no references on this surface yet")), "{:?}", r.stats.notes);
    assert_eq!(species_calls(&t), 1);
    // another grid than the store's: its references are another surface's
    let (mut s, _t) = h_student(3);
    let r = s.evaluate(&in_cell(lone_h(), 41), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback);
    assert!(r.stats.notes.iter().any(|n| n.contains("references are of another surface")), "{:?}", r.stats.notes);
}

/// Collection mode, no model: a box without its H reference falls back (rule 3's own condition);
/// once it has one, lone H is answered at the **canonical** reference (the largest box's) in
/// every box — what a teacher answer there is reported as, minus the wobble — and a new canonical
/// reference arriving under a reference report is booked into the next evaluation's jump.
#[test]
fn in_collection_mode_lone_atoms_take_the_canonical_reference() {
    let dir = collection_root("lone");
    let (mut s, t) = pair_student();
    s.attach_collection(&dir, &h2(1.4)).unwrap();
    let e = SpeciesMorse::atom(1);
    s.set_reference_solved(1, e, 33, Some(0.0));
    let big = in_cell(lone_h(), 49);
    let r = s.evaluate(&big, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback, "{:?}", r.stats.notes);
    assert!(r.stats.notes.iter().any(|n| n.contains("49³ box is not solved yet")), "{:?}", r.stats.notes);
    assert_eq!(species_calls(&t), 1);
    // the 49³ reference (1 mHa apart, as a box changes it) becomes the canonical one
    let e49 = e + 1e-3;
    s.set_reference_solved(1, e49, 49, Some(0.0));
    assert!(s.model().is_none(), "no model: the rule needs none");
    let r = s.evaluate(&big, &mut never()).unwrap();
    assert_reference(&r, e49, "49³");
    let r = s.evaluate(&lone_h(), &mut never()).unwrap();
    assert_reference(&r, e49, "33³ on the canonical surface");
    assert_eq!(species_calls(&t), 1, "no teacher call after the first");
    // a forced multiplicity is not the collection's surface
    let mut forced = lone_h();
    forced.multiplicity = Some(2);
    let r = s.evaluate(&forced, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback, "{:?}", r.stats.notes);
    // a larger box's reference arrives under a reference report: booked as Σ Δref_canonical
    s.discontinuity();
    let r = s.evaluate(&lone_h(), &mut never()).unwrap();
    assert_reference(&r, e49, "after an edit");
    assert_eq!(r.stats.refit_jump, None, "after an edit nothing is booked");
    let e57 = e + 2e-3;
    s.set_reference_solved(1, e57, 57, Some(0.0));
    let r = s.evaluate(&lone_h(), &mut never()).unwrap();
    assert_reference(&r, e57, "the new canonical reference");
    let j = r.stats.refit_jump.expect("the arrival is booked");
    assert!((j - (e57 - e49)).abs() < 1e-15, "booked {j:.3e}, moved {:.3e}", e57 - e49);
}

/// The surface switches between teacher answers and reference answers are booked along the
/// forces (`transition_jump`): J = E₁ − E₀ + ½(m̃₀ + m̃₁)·(r₁ − r₀), m̃ the reported forces (0 for a
/// reference answer).
#[test]
fn the_switch_between_teacher_and_reference_answers_is_booked() {
    let (mut s, _t) = h_student(12);
    s.refit().expect("a fit forms");
    let near = two_h(5.0, Some(3));
    let far = two_h(6.2, Some(3));
    let r0 = s.evaluate(&near, &mut never()).unwrap();
    assert_eq!(r0.stats.source, Source::Fallback);
    assert_eq!(r0.stats.energy_surface(), Surface::Teacher);
    let r1 = s.evaluate(&far, &mut never()).unwrap();
    assert_reference(&r1, 2.0 * SpeciesMorse::atom(1), "far");
    let work = |f: &[[f64; 3]], a: &Scene, b: &Scene| -> f64 {
        f.iter().zip(a.nuclei.iter().zip(&b.nuclei)).map(|(f, (p, q))| (0..3).map(|k| 0.5 * f[k] * (q.pos[k] - p.pos[k])).sum::<f64>()).sum()
    };
    let want = r1.energy - r0.energy + work(&r0.forces, &near, &far);
    let j = r1.stats.refit_jump.expect("teacher → reference is booked");
    assert!((j - want).abs() < 1e-14, "J {j:.6e} against {want:.6e}");
    let r2 = s.evaluate(&near, &mut never()).unwrap();
    assert_eq!(r2.stats.source, Source::Fallback);
    let want = r2.energy - r1.energy + work(&r2.forces, &far, &near);
    let j = r2.stats.refit_jump.expect("reference → teacher is booked");
    assert!((j - want).abs() < 1e-14, "J {j:.6e} against {want:.6e}");
    println!("teacher → reference J = {:.6e}, back {:.6e} Ha", r1.stats.refit_jump.unwrap(), j);
}

/// Audits are never scheduled at lone atoms: strict mode answers them from the references, and a
/// draft-mode model answer there (draft keeps its behaviour) dispatches none either; H₂ still does.
#[test]
fn no_audit_is_dispatched_at_lone_atoms() {
    let (mut s, _h, open, auditor_calls) = with_gated_auditor(1, 0.0);
    let jobs = s.side_jobs();
    let r = s.evaluate(&lone_h(), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Reference);
    assert_eq!(s.side_jobs(), jobs, "strict: no audit");
    Trainable::set_draft(&mut s, true);
    let r = s.evaluate(&lone_h(), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Surrogate, "draft mode keeps its behaviour: {:?}", r.stats.notes);
    assert_eq!(s.side_jobs(), jobs, "draft: no audit at lone atoms");
    assert!(!r.stats.notes.iter().any(|n| n.contains("audit dispatched")), "{:?}", r.stats.notes);
    // the mechanism is live: H₂ at 1.4 Bohr is audited (every answered step)
    let r = s.evaluate(&h2(1.4), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Surrogate);
    assert_eq!(s.side_jobs(), jobs + 1, "H₂ is audited: {:?}", r.stats.notes);
    open.store(true, std::sync::atomic::Ordering::SeqCst);
    assert_eq!(auditor_calls.load(std::sync::atomic::Ordering::SeqCst), 0, "nothing was solved for the lone atom");
}

/// A per-scene species-Morse student of `z`₂ dimers (r ∈ [lo, hi], random orientations) at
/// `charge`, with the `z` reference, refitted.
fn dimer_student(z: u8, lo: f64, hi: f64, n: usize, charge: i32) -> (SurrogateKs, Arc<Mutex<SpeciesMorse>>) {
    let (mut s, t) = pair_student();
    s.set_reference_energy(z, SpeciesMorse::atom(z));
    let mut rng = StdRng::seed_from_u64(0x10f);
    for i in 0..n {
        let r = lo + (hi - lo) * i as f64 / (n - 1) as f64;
        let origin = [rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3)];
        let mut sc = dimer(r, unit_vector(&mut rng), origin, z, z);
        sc.charge = charge;
        s.add_sample(species_sample(&sc), &sc).unwrap();
    }
    s.refit().expect("a fit forms");
    (s, t)
}

/// One atom of `z` at the off-centre spot of [`lone_h`].
fn lone(z: u8) -> Scene {
    let mut sc = lone_h();
    sc.nuclei[0].z = z;
    sc
}

/// Records a scanned reference of `z` with multiplicity `m` (and the fractional flag) in the
/// provider's reference cache at `scene`'s settings, as a sampling pass's scan does.
fn record_scan(s: &mut SurrogateKs, scene: &Scene, z: u8, m: u32, fractional: bool) {
    let key = atom_key(&s.provenance_for(scene), z);
    s.ref_cache.insert(key, AtomReference { energy: SpeciesMorse::atom(z), multiplicity: Some(m), fractional: Some(fractional), tol: 1e-6, provider: "pair morse".into(), radial: Default::default(), face_density: Some(0.0) });
}

/// Review I1: a lone C at automatic multiplicity (1) whose reference is the triplet, under a
/// **trusted** model — rule 0b declines (the scene is not solved at the reference's spin), and the
/// teacher answers: rule 5a's refusal for one atom too, never the model's E_ref (before fix round 1
/// 5a's one-atom exemption let the model answer). The step's own notes and its Activity row say why
/// (M5); readiness refuses the same way. At M = 3 the same atom is the rule's.
#[test]
fn a_lone_atom_the_references_decline_is_the_teachers_not_the_models() {
    let (mut s, t) = dimer_student(6, 2.4, 3.8, 40, 0);
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    // the spread rule (5c) refuses an empty neighbourhood of this dimer-only fit on its own; with
    // it off nothing else stood between the one-atom scene and the model (5a's exemption)
    let policy = *s.policy();
    s.set_policy(FallbackPolicy { max_descriptor_distance: f64::INFINITY, ..policy });
    let c = lone(6);
    record_scan(&mut s, &c, 6, 3, false);
    let _ = s.take_events();
    let n0 = species_calls(&t);
    let r = s.evaluate(&c, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback, "{:?}", r.stats.notes);
    assert_eq!(species_calls(&t), n0 + 1, "the teacher answered");
    assert_eq!(r.stats.energy_surface(), Surface::Teacher, "reported on the teacher's surface");
    assert!(r.stats.notes.iter().any(|n| n.contains("lone atoms the atom references cannot answer")), "{:?}", r.stats.notes);
    assert!(r.stats.notes.iter().any(|n| n.contains("not answered from the atom references") && n.contains("multiplicity 1 (automatic)")), "{:?}", r.stats.notes);
    let e = s.take_events();
    let ev = e.last().expect("the fallback's event");
    assert_eq!(ev.rule, sim::activity::TeachRule::IsolatedAtom);
    assert!(ev.detail.contains("multiplicity 1 (automatic)"), "the row says why (M5): {}", ev.detail);
    assert_eq!(s.samples(), 40, "never kept");
    let ready = s.readiness(&c).unwrap();
    assert!(!ready.answers && ready.rule == Some(sim::activity::TeachRule::IsolatedAtom), "{ready:?}");
    // a fresh readiness (no recorded verdict at this geometry) says the same
    let mut moved = c.clone();
    moved.nuclei[0].pos[0] += 0.3;
    let ready = s.readiness(&moved).unwrap();
    assert!(!ready.answers && ready.rule == Some(sim::activity::TeachRule::IsolatedAtom), "{ready:?}");
    // at the reference's own spin state the references answer (another provenance than the
    // model's: the provider's references)
    let mut triplet = c.clone();
    triplet.multiplicity = Some(3);
    let r = s.evaluate(&triplet, &mut never()).unwrap();
    assert_reference(&r, SpeciesMorse::atom(6), "C at M = 3");
    // the notes of a declined step do not leak into the next (M5)
    assert!(!r.stats.notes.iter().any(|n| n.contains("not answered")), "{:?}", r.stats.notes);
}

/// Review I1: a lone H at charge −1 under a trusted store trained at charge −1 (the same surface, so
/// rules 1–4 pass) — 0b declines on the charge, the teacher answers, not the model's E_ref(H⁰).
#[test]
fn a_charged_lone_atom_under_a_charged_model_is_the_teachers() {
    let (mut s, t) = dimer_student(1, 1.1, 2.3, 40, -1);
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    // as above: without the spread rule nothing else stood between the scene and the model
    let policy = *s.policy();
    s.set_policy(FallbackPolicy { max_descriptor_distance: f64::INFINITY, ..policy });
    let mut h = lone_h();
    h.charge = -1;
    let n0 = species_calls(&t);
    let r = s.evaluate(&h, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback, "{:?}", r.stats.notes);
    assert_eq!(species_calls(&t), n0 + 1);
    assert!(r.stats.notes.iter().any(|n| n.contains("charge is -1")), "{:?}", r.stats.notes);
    assert!(r.stats.notes.iter().any(|n| n.contains("lone atoms the atom references cannot answer")), "{:?}", r.stats.notes);
    let ready = s.readiness(&h).unwrap();
    assert!(!ready.answers, "{ready:?}");
}

/// Review I2: a shell check measured H–H still interacting near the cutoff (its fence) — two lone H
/// at the parallel multiplicity, which the rule would answer, go to the teacher instead.
#[test]
fn a_shell_check_fence_keeps_lone_pairs_of_its_species_with_the_teacher() {
    let (mut s, t) = h_student(12);
    s.refit().expect("a fit forms");
    let r = s.evaluate(&two_h(6.2, Some(3)), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Reference, "the premise: answered without the fence");
    let check = two_h(5.2, None);
    s.audit_fences.push(FailedAudit { pos: check.nuclei.iter().map(|n| n.pos).collect(), z: vec![1, 1], energy: 0.0, forces: vec![[0.0; 3]; 2], index: None, ionic: "none".into(), rejudges: 0, origin: AuditOrigin::ShellCheck, pair: Some((1, 1)) });
    let n0 = species_calls(&t);
    let r = s.evaluate(&two_h(6.2, Some(3)), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback, "{:?}", r.stats.notes);
    assert_eq!(species_calls(&t), n0 + 1);
    assert!(r.stats.notes.iter().any(|n| n.contains("shell check measured H–H")), "{:?}", r.stats.notes);
    // a lone H alone holds no fenced pair
    assert_eq!(s.evaluate(&lone_h(), &mut never()).unwrap().stats.source, Source::Reference);
}

/// Review I4 (controller ruling): two or more lone atoms are answered only when their reference is
/// known to end at a gap — Z = 1; on a pseudised surface a scan that shared no shell. Two lone He
/// at M* on the all-electron surface (integer aufbau: an open shell would not show) decline.
#[test]
fn copies_of_an_atom_not_known_to_end_at_a_gap_are_the_teachers() {
    assert!(super::super::lone::closed_spin_channels(1, true, None), "one electron");
    assert!(super::super::lone::closed_spin_channels(2, false, Some(false)), "pseudised surface, no shared shell");
    assert!(!super::super::lone::closed_spin_channels(8, false, Some(true)), "a shared shell");
    assert!(!super::super::lone::closed_spin_channels(2, false, None), "not recorded");
    assert!(!super::super::lone::closed_spin_channels(2, true, Some(false)), "integer aufbau hides an open shell");
    let (mut s, t) = h_student(3);
    s.set_reference_energy(2, SpeciesMorse::atom(2));
    let mut he2 = two_h(6.2, None);
    for n in &mut he2.nuclei {
        n.z = 2;
    }
    record_scan(&mut s, &he2, 2, 1, false);
    // one He alone is its own reference
    assert_reference(&s.evaluate(&lone(2), &mut never()).unwrap(), SpeciesMorse::atom(2), "one He");
    let n0 = species_calls(&t);
    let r = s.evaluate(&he2, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback, "{:?}", r.stats.notes);
    assert_eq!(species_calls(&t), n0 + 1);
    assert!(r.stats.notes.iter().any(|n| n.contains("2 lone He atoms") && n.contains("integer occupations")), "{:?}", r.stats.notes);
}

/// Review M1: with a model the answer sums the model's own references, so a reference changed
/// before the refit moves neither the model's answers nor the reference answers (no unbooked step
/// at a model ↔ reference switch); the refit then moves both, and its jump is booked at the next
/// reference answer — for an untrusted refit too.
#[test]
fn the_answer_follows_the_models_references_until_a_refit_books_the_change() {
    let (mut s, _h) = trained(1.2, 2.2, 40, FallbackPolicy::default());
    let e = MorseDimer::h2().e_atom;
    let r = s.evaluate(&lone_h(), &mut never()).unwrap();
    assert_reference(&r, e, "before");
    s.set_reference_energy(1, e + 1e-3);
    let r = s.evaluate(&lone_h(), &mut never()).unwrap();
    assert_reference(&r, e, "the model's reference until the refit");
    assert_eq!(r.stats.refit_jump, None);
    let p = s.model().unwrap().predict(&[[1.3, -0.7, 2.1]], &[1], false).unwrap();
    assert_eq!(p.energy.to_bits(), r.energy.to_bits(), "the model's own answer");
    s.refit().expect("the refit");
    let r = s.evaluate(&lone_h(), &mut never()).unwrap();
    assert_reference(&r, e + 1e-3, "after the refit");
    let j = r.stats.refit_jump.expect("the refit's jump is booked");
    assert!((j - 1e-3).abs() < 1e-12, "booked {j:.3e} (trusted {})", s.trusted());
}

/// Review M3: a relax that ends on a reference answer is converged and **not** re-checked by the
/// teacher — nothing was consulted, and DFT's force there is the grid box's confinement.
#[test]
fn a_relax_on_lone_atoms_is_not_verified_by_the_teacher() {
    let (s, t) = h_student(3);
    let (p, _sh) = driven(s);
    let mut d = Dynamics::new(lone_h(), p, 0.25, 1);
    assert_eq!(d.snapshot().source, Source::Reference);
    d.apply(DynCommand::Relax);
    let r = loop {
        let snap = d.tick(-1.0, &mut never());
        let r = snap.relax.clone().expect("relaxing");
        if r.converged.is_some() {
            break r;
        }
        assert!(r.steps < 100);
    };
    assert_eq!(r.converged, Some(true));
    assert_eq!(r.verified_max_force, None, "nothing to verify");
    assert_eq!(r.note, None);
    assert_eq!(species_calls(&t), 0, "no teacher solve");
}

/// Measurement (not a gate; the brief's "DFT − reference for a lone H at 3 offsets"): the GPU
/// teacher's lone H off the cell centre against the same atom at the centre (where the sampling
/// pass solves the reference) — what rule 0b leaves out. In the app's default box (81³ at h = σ =
/// 0.15, L = 12 Bohr) and the 49³ × 0.25 scene-file box (the same 12 Bohr), all-electron (the app's
/// `Accepted` pseudises H only with its GPU dimer gate set, which a test binary does not: a run
/// with `Accepted` here is the all-electron one bit for bit); SCF tol 1e-6 (the sample tolerance).
/// Offsets: a sub-grid shift (the egg-box ripple), (2, 1, 0.5) Bohr (4 Bohr from the nearest face:
/// the box rule's edge), and 2 Bohr from a face. Measured 2026-11-12 (RX 7900 XTX), E − E_ref:
/// 81³ +8.7e-7 / +1.82e-3 / +3.47e-2 Ha (F_x −2.1e-5 / −2.6e-3 / −5.2e-2 Ha/Bohr, toward the
/// centre; face density 2.5e-7 / 7.3e-6 / 3.0e-4 against the box rule's 7.2e-6), 49³ +2.7e-6 /
/// +1.95e-3 / +3.29e-2 Ha (face density 8.1e-7 / 2.0e-5 / 6.9e-4 against 2e-5). The grid's own
/// ripple is µHa; the rest is the walls' confinement, which the reference (the free atom) leaves
/// out — DFT in the box is the less physical of the two there.
#[test]
#[ignore = "measurement: cargo test -p providers --release diag_lone_h_wobble -- --ignored --nocapture"]
fn diag_lone_h_wobble() {
    let Some(ctx) = gpu::GpuContext::new_headless() else {
        println!("no GPU adapter — skipped");
        return;
    };
    for (n, h, ionic) in [(81usize, 0.15, sim::IonicPolicy::None), (49, 0.25, sim::IonicPolicy::None)] {
        let mut t = crate::ks::GpuKs::new(ctx.clone(), 1e-6);
        t.set_display(false);
        let half = 0.5 * (n - 1) as f64 * h;
        let at = |p: [f64; 3]| Scene {
            nuclei: vec![Nucleus::new(1, p)],
            cell: Cell { n, h },
            charge: 0,
            multiplicity: None,
            sigma: None,
            external_field: [0.0; 3],
            ionic,
            train_mask: None,
        };
        let mut solve = |p: [f64; 3]| {
            t.reset();
            let r = t.evaluate(&at(p), &mut never()).unwrap_or_else(|e| panic!("{n}³ {ionic:?} at {p:?}: {e}"));
            (r.energy, r.forces[0], r.stats.face_density)
        };
        let (e0, _, f0) = solve([0.0; 3]);
        println!("{n}³ h {h} {ionic:?}: centre E_ref = {e0:.9} Ha (face density {f0:?})");
        for p in [[0.37 * h, 0.21 * h, 0.13 * h], [2.0, 1.0, 0.5], [half - 2.0, 0.0, 0.0]] {
            let (e, f, face) = solve(p);
            println!("    at {p:?}: E − E_ref = {:+.3e} Ha, F = ({:+.1e}, {:+.1e}, {:+.1e}) Ha/Bohr, face density {face:?}", e - e0, f[0], f[1], f[2]);
        }
    }
}
