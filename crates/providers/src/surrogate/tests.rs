use super::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sim::{Cell, DynCommand, DynSnapshot, Dynamics, MorseDimer, Nucleus, Shared, SharedTrainable, Surface};
use std::sync::{Arc, Mutex, MutexGuard};

fn never() -> impl FnMut() -> bool {
    || false
}

fn cell() -> Cell {
    Cell { n: 33, h: 0.25 }
}

/// Two nuclei (Z `z0`, `z1`) `r` apart along `dir`, centred on `origin` (so a dimer sits
/// near the middle of the 8 Bohr cell: the GPU test's energies must not see the faces).
fn dimer(r: f64, dir: [f64; 3], origin: [f64; 3], z0: u8, z1: u8) -> Scene {
    let p0: [f64; 3] = std::array::from_fn(|a| origin[a] - 0.5 * r * dir[a]);
    let p1: [f64; 3] = std::array::from_fn(|a| origin[a] + 0.5 * r * dir[a]);
    Scene {
        nuclei: vec![Nucleus::new(z0, p0), Nucleus::new(z1, p1)],
        cell: cell(),
        charge: 0,
        multiplicity: None,
        sigma: None,
        external_field: [0.0; 3],
    }
}

fn h2(r: f64) -> Scene {
    dimer(r, [1.0, 0.0, 0.0], [0.0; 3], 1, 1)
}

fn unit_vector(rng: &mut StdRng) -> [f64; 3] {
    loop {
        let v: [f64; 3] = [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)];
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if (1e-3..1.0).contains(&n) {
            return v.map(|x| x / n);
        }
    }
}

/// A Morse-teacher sample of the scene (the teacher's exact answer).
fn morse_sample(teacher: &MorseDimer, scene: &Scene) -> Sample {
    let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
    let (energy, forces) = teacher.energy_and_forces(&pos).unwrap();
    Sample { pos, z: scene.nuclei.iter().map(|n| n.z).collect(), energy, forces, tag: SampleTag::Scan, weight: 1.0 }
}

/// A surrogate over a shared Morse teacher with `n` scan samples on r ∈ [lo, hi] (random
/// orientations and origins), the H reference set, refitted when `n > 0`.
fn trained(lo: f64, hi: f64, n: usize, policy: FallbackPolicy) -> (SurrogateKs, Arc<Mutex<MorseDimer>>) {
    let teacher = Shared::new(MorseDimer::h2());
    let handle = teacher.handle();
    let mut s = SurrogateKs::new(Box::new(teacher), policy).with_teacher_tol(1e-6);
    s.set_reference_energy(1, MorseDimer::h2().e_atom);
    let mut rng = StdRng::seed_from_u64(4);
    for i in 0..n {
        let r = lo + (hi - lo) * i as f64 / (n.max(2) - 1) as f64;
        let origin = [rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3)];
        let sc = dimer(r, unit_vector(&mut rng), origin, 1, 1);
        s.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
    }
    if n > 0 {
        let rep = s.refit().expect("refit");
        println!("trained on {n} samples r ∈ [{lo}, {hi}]: rmse E/atom {:.1e}, F {:.1e}; s_E {:.1e} s_F {:.1e} λ {:.3e}; dof_E {:.1} dof_F {:.1} q′ {}; converged {}",
                 rep.rmse_energy_per_atom, rep.rmse_force, rep.sigma_e, rep.sigma_f, rep.lambda, rep.residual_dof_e, rep.residual_dof_f, rep.q_prime, rep.converged);
    }
    (s, handle)
}

fn calls(h: &Arc<Mutex<MorseDimer>>) -> usize {
    h.lock().unwrap().calls
}

fn max_force_diff(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter().zip(b).flat_map(|(x, y)| (0..3).map(move |k| (x[k] - y[k]).abs())).fold(0.0, f64::max)
}

/// The Verlet swing to compare against: E_conserved's spread and largest per-step change
/// over `steps` of a plain run on `teacher` from `scene` (same dt, no surrogate).
fn verlet_reference(scene: &Scene, teacher: MorseDimer, steps: usize) -> (f64, f64) {
    let mut d = Dynamics::new(scene.clone(), Box::new(teacher), 0.25, 1);
    let mut prev = d.snapshot().energy_conserved;
    let (mut lo, mut hi, mut max_step) = (prev, prev, 0.0f64);
    for _ in 0..steps {
        d.apply(DynCommand::Step(1));
        let e = d.tick(f64::INFINITY, &mut never()).energy_conserved;
        lo = lo.min(e);
        hi = hi.max(e);
        max_step = max_step.max((e - prev).abs());
        prev = e;
    }
    (hi - lo, max_step)
}

/// A surrogate boxed for a `Dynamics` (through `SharedTrainable`, which the driver reaches with
/// `as_trainable` like the surrogate itself), and the handle the test reads it by.
fn driven(s: SurrogateKs) -> (Box<SharedTrainable<SurrogateKs>>, Arc<Mutex<SurrogateKs>>) {
    let p = SharedTrainable::new(s);
    let h = p.handle();
    (Box::new(p), h)
}

/// The surrogate inside a `Dynamics` built with [`driven`], under its lock — never held across
/// a call into that `Dynamics` (the lock is not re-entrant).
fn surrogate_of(h: &Arc<Mutex<SurrogateKs>>) -> MutexGuard<'_, SurrogateKs> {
    h.lock().unwrap_or_else(|e| e.into_inner())
}

/// Rule 1: nothing fitted → the teacher answers, the sample is kept, the result is on the
/// teacher surface.
#[test]
fn no_model_means_fallback_and_a_sample() {
    let (mut s, h) = trained(0.0, 0.0, 0, FallbackPolicy::default());
    assert_eq!(s.name(), "surrogate(morse dimer)");
    assert!(!s.trusted() && s.model().is_none() && s.basis().is_none());
    let scene = h2(1.5);
    let r = s.evaluate(&scene, &mut never()).unwrap();
    assert_eq!(calls(&h), 1);
    assert_eq!(s.samples(), 1);
    assert!(s.pending_refit());
    assert_eq!(r.stats.source, Source::Fallback);
    assert_eq!(r.stats.energy_surface(), Surface::Teacher);
    assert_eq!(r.stats.teacher_gap, None);
    assert_eq!(r.stats.refit_jump, None);
    assert_eq!(r.stats.uncertainty, None);
    let (e, f) = MorseDimer::h2().energy_and_forces(&[[-0.75, 0.0, 0.0], [0.75, 0.0, 0.0]]).unwrap();
    assert_eq!(r.energy, e);
    assert_eq!(r.forces, f);
    assert!(r.stats.notes.iter().any(|n| n.contains("no model yet")), "{:?}", r.stats.notes);
    assert_eq!(s.epoch(), 0);
}

/// Rules 6–8 and the learning loop: a window-trained dimer answers inside the window to
/// the teacher's accuracy, falls back outside it (keeping the sample), and answers there
/// after the refit.
#[test]
fn window_trained_model_answers_inside_and_falls_back_outside() {
    let (mut s, h) = trained(1.2, 2.2, 40, FallbackPolicy::default());
    assert!(s.trusted(), "{:?}", s.report());
    assert_eq!(s.basis().unwrap().species, vec![1]);
    let teacher = MorseDimer::h2();

    let inside = h2(1.6);
    let r = s.evaluate(&inside, &mut never()).unwrap();
    let (e, f) = teacher.energy_and_forces(&[[-0.8, 0.0, 0.0], [0.8, 0.0, 0.0]]).unwrap();
    println!("r = 1.6: {:?}, |ΔE| {:.1e}, max|ΔF| {:.1e}, uncertainty {:?}", r.stats.source, (r.energy - e).abs(), max_force_diff(&r.forces, &f), r.stats.uncertainty);
    assert_eq!(r.stats.source, Source::Surrogate, "{:?}", r.stats.notes);
    assert_eq!(calls(&h), 0, "the teacher was not called");
    assert!((r.energy - e).abs() < 1e-5);
    assert!(max_force_diff(&r.forces, &f) < 1e-4);
    assert!(r.stats.uncertainty.is_some_and(|u| u < 2e-3));
    assert_eq!(r.stats.teacher_gap, None);
    assert!(r.display.is_none());
    assert_eq!(r.stats.scf_steps, 0);
    assert_eq!(r.stats.energy.total, r.energy);

    // outside the window: the teacher answers on the model surface and the sample is kept
    let near = h2(3.0);
    let r = s.evaluate(&near, &mut never()).unwrap();
    println!("r = 3.0: {:?}, notes {:?}", r.stats.source, r.stats.notes);
    assert_eq!(r.stats.source, Source::Fallback);
    assert_eq!(calls(&h), 1);
    assert_eq!(s.samples(), 41);
    assert!(s.pending_refit());
    let (e3, f3) = teacher.energy_and_forces(&[[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]]).unwrap();
    assert_eq!(r.forces, f3, "the teacher's forces");
    let gap = r.stats.teacher_gap.expect("on the model surface");
    assert!((r.energy + gap - e3).abs() < 1e-12, "energy is the model's: {} + {gap} vs {e3}", r.energy);
    assert_eq!(r.stats.energy_surface(), Surface::Model);
    assert!(r.stats.energy.total == e3, "the teacher's breakdown is kept");
    assert!(r.stats.model_forces.is_some(), "the model forces travel with a model-surface fallback");

    let rep = s.refit().unwrap();
    let tr = s.training_residuals().unwrap();
    println!("after the refit with the r = 3.0 sample: rmse E/atom {:.1e}, F {:.1e}; s_E {:.1e} s_F {:.1e}; training residuals E {:.1e}/atom, F {:.1e} (bound {:.1e})",
             rep.rmse_energy_per_atom, rep.rmse_force, rep.sigma_e, rep.sigma_f, tr.energy_per_atom, tr.force, tr.force_bound);
    assert_eq!(rep.n_samples, 41);
    assert!(!s.pending_refit());
    assert!(s.trusted(), "{tr:?}");
    assert_eq!(s.epoch(), 2, "the surface moved twice (first fit, this refit)");
    let r = s.evaluate(&near, &mut never()).unwrap();
    println!("r = 3.0 after refit: {:?}, |ΔE| {:.1e}, max|ΔF| {:.1e}, uncertainty {:?}, notes {:?}", r.stats.source, (r.energy - e3).abs(), max_force_diff(&r.forces, &f3), r.stats.uncertainty, r.stats.notes);
    assert_eq!(r.stats.source, Source::Surrogate, "{:?}", r.stats.notes);
    assert_eq!(calls(&h), 1);
    assert!((r.energy - e3).abs() < 1e-4);
    assert!(max_force_diff(&r.forces, &f3) < 1e-4);
    assert!(r.stats.refit_jump.is_some(), "the first evaluate after a refit reports the jump");
    assert_eq!(s.evaluate(&near, &mut never()).unwrap().stats.refit_jump, None, "once");

}

/// A surrogate over a shared Morse teacher trained on `n` window samples of the scripted
/// teacher Morse(r) + a·sin(k r), k = 2π·`periods`/(hi − lo) — the sinusoid rides on the
/// Morse energy with its consistent force −a k cos(k r) along the bond. With a = 0 this is
/// `trained`.
fn trained_oscillatory(lo: f64, hi: f64, n: usize, a: f64, periods: f64) -> (SurrogateKs, Arc<Mutex<MorseDimer>>) {
    let teacher = Shared::new(MorseDimer::h2());
    let handle = teacher.handle();
    let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_teacher_tol(1e-6);
    s.set_reference_energy(1, MorseDimer::h2().e_atom);
    let k = 2.0 * std::f64::consts::PI * periods / (hi - lo);
    let mut rng = StdRng::seed_from_u64(4);
    for i in 0..n {
        let r = lo + (hi - lo) * i as f64 / (n.max(2) - 1) as f64;
        let origin = [rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3)];
        let dir = unit_vector(&mut rng);
        let sc = dimer(r, dir, origin, 1, 1);
        let mut smp = morse_sample(&MorseDimer::h2(), &sc);
        smp.energy += a * (k * r).sin();
        let de = a * k * (k * r).cos(); // dE/dr; nucleus 1 sits at +r/2·dir
        for c in 0..3 {
            smp.forces[1][c] -= de * dir[c];
            smp.forces[0][c] += de * dir[c];
        }
        s.add_sample(smp, &sc).unwrap();
    }
    s.refit().expect("refit");
    (s, handle)
}

/// Ruling S8 (review C): basis misspecification is invisible to the posterior and caught by
/// the training residuals. The fixture must be something NO basis the provider can build
/// represents at the training points. The original one (40 Morse points on [1.2, 2.2] plus
/// one at r = 4.0, whose −3.4e-3 Ha tail the f_c-damped body-2 n_max = 8 basis could not
/// carry: 1e-2 Ha offset at a training point) was a *body-2* misspecification: the body-3
/// default basis adds the radial shapes T_{n₁}T_{n₂}f_c²(r/r_cut)^{2l} and fits window + r =
/// 4.0 (and + r = 5.0) to ~1e-8 Ha/atom, correctly earning trust (Task 6 fix round). The
/// fixture now is Morse plus a sinusoid of 6 periods across the window, amplitude 1e-2 Ha:
/// every feature of the default basis is a polynomial in r of degree ≤ 7 (body-2) or ≤ 4
/// (body-3 at D = 4) times the smooth f_c, so no fit can follow 12 extrema on 1 Bohr — the
/// two-noise evidence settles on "the data are noisy" (s_E ≈ 4e-3) while the max training
/// energy residual per atom sits at ≈ a/2 ≥ 3 × `energy_std_per_atom`; the refit must leave
/// the model untrusted with the note naming that maximum, and the fallback teacher answers.
/// The same construction with a = 0 is the Morse window and is trusted (the control).
#[test]
fn misspecified_fit_is_untrusted_by_its_training_residuals() {
    let (mut s, h) = trained_oscillatory(1.2, 2.2, 40, 1e-2, 6.0);
    let rep = s.report().unwrap().clone();
    let tr = s.training_residuals().unwrap();
    println!("Morse + 1e-2·sin(6 periods): rmse E/atom {:.1e}, F {:.1e}; s_E {:.1e} s_F {:.1e}; γ_E {:.2} γ_F {:.2}; training residuals E {:.1e}/atom, F {:.1e} (bound {:.1e}); trusted {}; reason {:?}",
             rep.rmse_energy_per_atom, rep.rmse_force, rep.sigma_e, rep.sigma_f, rep.gamma_e, rep.gamma_f, tr.energy_per_atom, tr.force, tr.force_bound, s.trusted(), s.untrusted_reason());
    assert!(rep.n_samples >= rep.q_prime && rep.residual_dof_e >= MIN_RESIDUAL_DOF && rep.residual_dof_f >= MIN_RESIDUAL_DOF, "{rep:?}: the sample-count conditions pass; only the residuals can refuse");
    assert!(!s.trusted(), "the misspecified fit must not be trusted");
    assert!(tr.energy_per_atom > 3e-3, "{tr:?}: max training energy residual ≥ 3× energy_std_per_atom");
    assert!(s.untrusted_reason().unwrap().starts_with("max training energy residual"), "{:?}", s.untrusted_reason());
    let r = s.evaluate(&h2(1.6), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback);
    assert_eq!(r.stats.energy_surface(), Surface::Teacher);
    assert!(r.stats.notes.iter().any(|n| n.contains("surrogate untrusted: max training energy residual")), "{:?}", r.stats.notes);
    assert_eq!(calls(&h), 1);
    // the epistemic energy criterion alone (the floor N·s_E excluded, as the provider does)
    // would not have caught it at an in-window geometry: the point of the rule
    let model = s.model().unwrap();
    let pred = model.predict(&[[-0.8, 0.0, 0.0], [0.8, 0.0, 0.0]], &[1, 1], true).unwrap();
    let std_e = pred.energy_std.unwrap();
    let epistemic_per_atom = (std_e * std_e - (2.0 * model.sigma_e).powi(2)).max(0.0).sqrt() / 2.0;
    println!("at r = 1.6 the model's epistemic energy std is {epistemic_per_atom:.1e} Ha/atom (total {std_e:.1e}, floor 2·s_E {:.1e}) against a {:.1e} Ha/atom training residual", 2.0 * model.sigma_e, tr.energy_per_atom);
    assert!(epistemic_per_atom < tr.energy_per_atom / 3.0, "the training residual is what refuses, not the posterior");

    // control: the same window without the sinusoid is the Morse fit and is trusted
    let (s0, _) = trained_oscillatory(1.2, 2.2, 40, 0.0, 6.0);
    let tr0 = s0.training_residuals().unwrap();
    println!("a = 0 control: training residuals E {:.1e}/atom, F {:.1e}; trusted {}", tr0.energy_per_atom, tr0.force, s0.trusted());
    assert!(s0.trusted(), "{:?}", s0.untrusted_reason());
    assert!(tr0.energy_per_atom < 1e-6);
}

/// Rule 4: a fit with too few samples is never consulted, and the note says why.
#[test]
fn untrusted_model_falls_back_with_the_reason() {
    let (mut s, h) = trained(1.2, 2.2, 5, FallbackPolicy::default());
    assert!(!s.trusted());
    let rep = s.report().unwrap().clone();
    assert!(rep.n_samples < rep.q_prime, "{rep:?}");
    let r = s.evaluate(&h2(1.6), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback);
    assert_eq!(r.stats.energy_surface(), Surface::Teacher);
    assert_eq!(calls(&h), 1);
    assert_eq!(s.samples(), 6);
    let want = format!("surrogate untrusted: n_samples {} < q_prime {}", rep.n_samples, rep.q_prime);
    assert!(r.stats.notes.iter().any(|n| n.contains(&want)), "{:?}", r.stats.notes);

    // enough samples for n_samples ≥ q′ (29 with the body-3 default basis, S = 1) but not
    // for 30 residual energy degrees of freedom (n_E − γ_E < 30 at 30 samples)
    let (mut s, _) = trained(1.2, 2.2, 30, FallbackPolicy::default());
    let rep = s.report().unwrap().clone();
    assert!(rep.n_samples >= rep.q_prime && rep.residual_dof_e < MIN_RESIDUAL_DOF, "{rep:?}");
    assert!(!s.trusted());
    let r = s.evaluate(&h2(1.6), &mut never()).unwrap();
    assert!(r.stats.notes.iter().any(|n| n.contains("untrusted: residual_dof_e")), "{:?}", r.stats.notes);
}

/// Rule 3: a scene at other settings than the model's is never answered, even where the
/// variance would pass; the teacher's answer is not kept.
#[test]
fn unseen_species_or_charge_falls_back_unconditionally() {
    let (mut s, h) = trained(1.2, 2.2, 40, FallbackPolicy::default());
    let mut cation = h2(1.6);
    cation.charge = 1;
    let r = s.evaluate(&cation, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback);
    assert_eq!(calls(&h), 1);
    assert_eq!(s.samples(), 40, "a sample from another surface is not kept");
    assert_eq!(r.stats.teacher_gap, None);
    assert_eq!(r.stats.uncertainty, None);
    assert_eq!(r.stats.energy_surface(), Surface::Teacher);
    assert!(r.stats.notes.iter().any(|n| n.contains("charge 1") && n.contains("differs from the model's")), "{:?}", r.stats.notes);
    // a different σ, grid or multiplicity likewise
    let mut other = h2(1.6);
    other.sigma = Some(0.3);
    assert_eq!(s.evaluate(&other, &mut never()).unwrap().stats.source, Source::Fallback);
    let mut other = h2(1.6);
    other.cell = Cell { n: 49, h: 0.25 };
    assert_eq!(s.evaluate(&other, &mut never()).unwrap().stats.source, Source::Fallback);
    let mut other = h2(1.6);
    other.multiplicity = Some(3);
    assert_eq!(s.evaluate(&other, &mut never()).unwrap().stats.source, Source::Fallback);
    assert_eq!(s.samples(), 40);
    // an unknown species at a foreign provenance: one note (rule 3 precedes rule 2), the
    // sample not kept, no rebuild scheduled
    let mut h_he_cation = dimer(1.6, [0.0, 1.0, 0.0], [0.0; 3], 1, 2);
    h_he_cation.charge = 1;
    let r = s.evaluate(&h_he_cation, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback);
    let surrogate_notes: Vec<&String> = r.stats.notes.iter().filter(|n| n.starts_with("surrogate")).collect();
    assert_eq!(surrogate_notes.len(), 1, "{:?}", r.stats.notes);
    assert!(surrogate_notes[0].contains("differs from the model's") && !surrogate_notes[0].contains("species"), "{:?}", r.stats.notes);
    assert_eq!(s.samples(), 40);
    assert!(!s.species_dirty);
    // the matching scene is still answered
    assert_eq!(s.evaluate(&h2(1.6), &mut never()).unwrap().stats.source, Source::Surrogate);
}

/// Rule 2 and the rebuild: a species the model does not know falls back unconditionally and
/// is learned at the next refit — once its reference energy exists.
#[test]
fn new_species_falls_back_and_is_learned_at_the_next_refit() {
    // 180 samples: the rebuilt two-species body-3 basis has Q′ = 176 (ruling S11/S12), and
    // the trust rule needs n_samples ≥ Q′ after the rebuild
    let (mut s, h) = trained(1.2, 2.2, 180, FallbackPolicy::default());
    let h_he = dimer(1.6, [0.0, 1.0, 0.0], [0.1, -0.1, 0.0], 1, 2);
    let r = s.evaluate(&h_he, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback);
    assert_eq!(r.stats.energy_surface(), Surface::Teacher);
    assert_eq!(calls(&h), 1);
    assert_eq!(s.samples(), 181, "kept: the rebuild needs it");
    assert!(r.stats.notes.iter().any(|n| n.contains("species He is not in the model (H)")), "{:?}", r.stats.notes);
    // no reference for He yet: the refit fails and the old model stands
    let old_epoch = s.epoch();
    assert_eq!(s.refit(), Err(FitError::MissingReference(2)));
    assert_eq!(s.basis().unwrap().species, vec![1]);
    assert!(s.trusted() && s.epoch() == old_epoch);
    assert_eq!(s.evaluate(&h2(1.6), &mut never()).unwrap().stats.source, Source::Surrogate, "still answers H₂");
    // with the reference the basis is rebuilt with both species
    s.set_reference_energy(2, -2.83);
    let rep = s.refit().unwrap();
    assert_eq!(s.basis().unwrap().species, vec![1, 2]);
    assert_eq!(rep.q_prime, 8 * 3 + 2 * 76, "n_max·S(S+1)/2 body-2 classes plus S·P₃ body-3 columns (D = 4) with S = 2");
    assert_eq!(rep.n_samples, 181);
    let r = s.evaluate(&h_he, &mut never()).unwrap();
    assert!(!r.stats.notes.iter().any(|n| n.contains("species")), "{:?}", r.stats.notes);
    println!("H–He after the rebuild: {:?} {:?}", r.stats.source, r.stats.notes);
    assert_eq!(s.evaluate(&h2(1.6), &mut never()).unwrap().stats.source, Source::Surrogate, "H₂ still answered after the rebuild");
}

/// Rule 5: near r_cut every body-2 feature vanishes, the variance is at its floor and would
/// pass, and only the envelope knows the model has never seen an empty neighbourhood.
#[test]
fn envelope_catches_the_vanishing_feature_direction_the_variance_cannot() {
    let policy = FallbackPolicy::default();
    let (mut s, h) = trained(1.2, 2.2, 40, policy);
    // Beyond r_cut = 6.0 every body-2 feature AND its gradient is exactly zero: the leverage
    // vanishes, the stds sit on their floors, and the model would answer "Σ ref, no force"
    // where the Morse still binds by 2e-3 Ha with a 2e-3 Ha/Bohr force. (Just inside r_cut,
    // r = 5.9, the force variance still fires: the Chebyshev gradients do not vanish there.)
    let far = h2(6.5);
    let model = s.model().unwrap();
    let pred = model.predict(&[[-3.25, 0.0, 0.0], [3.25, 0.0, 0.0]], &[1, 1], true).unwrap();
    let fs = pred.force_std.as_ref().unwrap().iter().fold(0.0f64, |m, &x| m.max(x)) / SQRT_3;
    let floor = 2.0 * model.sigma_e;
    let e_epi = (pred.energy_std.unwrap().powi(2) - floor * floor).max(0.0).sqrt() / 2.0;
    let (e_t, f_t) = MorseDimer::h2().energy_and_forces(&[[-3.25, 0.0, 0.0], [3.25, 0.0, 0.0]]).unwrap();
    println!("r = 6.5: per-component force std {fs:.1e} (bound {:.0e}), epistemic energy std/atom {e_epi:.1e} (bound {:.0e}); model E − teacher {:+.1e} Ha, model F {:?} vs teacher {:+.1e}",
             policy.force_std_abs, policy.energy_std_per_atom, pred.energy - e_t, pred.forces[0], f_t[0][0]);
    assert_eq!(pred.forces, vec![[0.0; 3]; 2], "beyond r_cut the model has no force");
    assert!(fs < 0.1 * policy.force_std_abs && e_epi < 0.1 * policy.energy_std_per_atom, "test premise: the variance passes");
    let r = s.evaluate(&far, &mut never()).unwrap();
    println!("notes {:?}", r.stats.notes);
    assert_eq!(r.stats.source, Source::Fallback);
    assert_eq!(calls(&h), 1);
    assert!(r.stats.notes.iter().any(|n| n.contains("outside the descriptor envelope")), "{:?}", r.stats.notes);
    assert!(r.stats.uncertainty.is_some_and(|u| u < policy.force_std_abs), "the variance is still reported");
    assert_eq!(r.stats.energy_surface(), Surface::Model);
    // inside the window the envelope is quiet
    let r = s.evaluate(&h2(1.7), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Surrogate);
    // the envelope's limit is the policy's
    s.set_policy(FallbackPolicy { max_descriptor_distance: f64::INFINITY, ..policy });
    let r = s.evaluate(&far, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Surrogate, "{:?}", r.stats.notes);
}

/// One-surface accounting through the driver: with the policy forcing fallbacks, the atoms
/// move on the teacher's forces while `energy_ks` is the model's and `teacher_gap` the known
/// difference between the two surfaces.
#[test]
fn fallback_step_uses_dft_forces_but_books_the_model_energy() {
    // trained on D = 0.17; the live teacher then has D′ = 1.25 D (energies and forces differ).
    // Fallbacks are forced through the envelope (a negative limit rejects every geometry):
    // zero force thresholds would also fail the training-residual trust condition, which
    // uses the same numbers, and an untrusted model is a teacher-surface fallback.
    let always_fallback = FallbackPolicy { max_descriptor_distance: -1.0, audit_every: 0, ..FallbackPolicy::default() };
    let (s, h) = trained(1.2, 2.2, 40, always_fallback);
    let model = s.model().unwrap().clone();
    let mut live = MorseDimer::h2();
    live.d *= 1.25;
    h.lock().unwrap().d = live.d;

    let v0 = [3e-4, -1e-4, 2e-4];
    let mut scene = h2(1.55);
    scene.nuclei[1].vel = v0;
    let (m0, m1) = (scene.nuclei[0].mass, scene.nuclei[1].mass);
    let dt = units::fs_to_au_time(0.25);
    let mut d = Dynamics::new(scene.clone(), Box::new(s), 0.25, 1);
    let snap0 = d.snapshot();
    assert_eq!(snap0.source, Source::Fallback);
    let p0: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
    let (e_t0, f_t0) = live.energy_and_forces(&p0).unwrap();
    let e_m0 = model.predict(&p0, &[1, 1], false).unwrap().energy;
    assert_eq!(snap0.energy_ks, e_m0, "the model's energy");
    assert_eq!(snap0.forces, f_t0, "the teacher's forces");
    assert!((snap0.teacher_gap.unwrap() - (e_t0 - e_m0)).abs() < 1e-14);

    d.apply(DynCommand::Step(1));
    let snap1 = d.tick(f64::INFINITY, &mut never());
    assert_eq!(snap1.step, 1);
    // Verlet drift with the teacher's forces
    let r1: Vec<[f64; 3]> = [(p0[0], [0.0; 3], m0), (p0[1], v0, m1)]
        .iter()
        .zip(&f_t0)
        .map(|((r, v, m), f)| std::array::from_fn(|a| r[a] + dt * v[a] + 0.5 * dt * dt * f[a] / m))
        .collect();
    for (n, r) in snap1.scene.nuclei.iter().zip(&r1) {
        for a in 0..3 {
            assert!((n.pos[a] - r[a]).abs() < 1e-15, "{:?} vs {r:?}", n.pos);
        }
    }
    let p1: Vec<[f64; 3]> = snap1.scene.nuclei.iter().map(|n| n.pos).collect();
    let (e_t1, f_t1) = live.energy_and_forces(&p1).unwrap();
    let e_m1 = model.predict(&p1, &[1, 1], false).unwrap().energy;
    assert_eq!(snap1.source, Source::Fallback);
    assert_eq!(snap1.energy_ks, e_m1);
    assert_eq!(snap1.forces, f_t1);
    let gap = snap1.teacher_gap.unwrap();
    assert!((gap - (e_t1 - e_m1)).abs() < 1e-14, "gap {gap} vs {}", e_t1 - e_m1);
    // the known offset between the surfaces: 0.25 D [(1 − x)² − 1] at r₁
    let r = ((p1[1][0] - p1[0][0]).powi(2) + (p1[1][1] - p1[0][1]).powi(2) + (p1[1][2] - p1[0][2]).powi(2)).sqrt();
    let x = (-(r - 1.4)).exp();
    let offset = 0.25 * 0.17 * ((1.0 - x) * (1.0 - x) - 1.0);
    println!("gap {gap:+.6e} Ha, analytic offset {offset:+.6e} Ha, model error {:+.1e}", gap - offset);
    assert!((gap - offset).abs() < 1e-5, "the gap is the teacher change plus the model's fit error");
    assert_eq!(snap1.eval.energy.total, e_t1, "the teacher's breakdown");
    assert_eq!(snap1.fallback_fraction_100, 1.0);
    assert_eq!(calls(&h), 2);
}

/// The accounting across a refit: `E_conserved` is continuous to rounding when the model
/// first becomes trusted (teacher surface → model surface), across the re-evaluation the
/// driver does when the epoch changes, and `teacher_gap` is `None` on Surrogate steps.
#[test]
fn energy_conserved_is_continuous_across_a_forced_refit() {
    let (s, _h) = trained(0.0, 0.0, 0, FallbackPolicy::default());
    let mut scene = h2(1.55);
    scene.nuclei[1].vel = [4e-4, 1e-4, 0.0];
    let (_, ref_step) = verlet_reference(&scene, MorseDimer::h2(), 100);
    let (p, sh_d) = driven(s);
    let mut d = Dynamics::new(scene.clone(), p, 0.25, 1);
    let mut e_c = Vec::new();
    let mut sources = Vec::new();
    for _ in 0..50 {
        d.apply(DynCommand::Step(1));
        let snap = d.tick(f64::INFINITY, &mut never());
        e_c.push(snap.energy_conserved);
        sources.push(snap.source);
    }
    assert!(sources.iter().all(|&s| s == Source::Fallback));
    let before = d.snapshot();
    assert_eq!(before.model_epoch, 0);
    assert_eq!(before.eval.energy_surface(), Surface::Teacher);
    {
        let mut s = surrogate_of(&sh_d);
        assert_eq!(s.samples(), 51);
        // 20 scan samples from a slightly different well (D 0.5 % deeper): the fit then
        // compromises between them and the 51 live samples, so the model surface differs
        // visibly from the teacher's at r_50 and the jump is a real number, not rounding —
        // while the training residuals (~2e-4/atom) stay inside the trust condition
        let mut deeper = MorseDimer::h2();
        deeper.d *= 1.005;
        let mut rng = StdRng::seed_from_u64(9);
        for i in 0..20 {
            let r = 1.2 + i as f64 / 19.0;
            let sc = dimer(r, unit_vector(&mut rng), [0.0; 3], 1, 1);
            s.add_sample(morse_sample(&deeper, &sc), &sc).unwrap();
        }
        let rep = s.refit().unwrap();
        println!("mid-run refit: {} samples, dof_E {:.1}, dof_F {:.1}, s_F {:.1e}, training residuals {:?}", rep.n_samples, rep.residual_dof_e, rep.residual_dof_f, rep.sigma_f, s.training_residuals());
        assert!(s.trusted(), "{:?}", s.training_residuals());
        assert_eq!(s.epoch(), 1);
    }
    // a paused tick with nothing queued re-evaluates at the unchanged positions (the epoch
    // changed) and books the jump: nothing else moves, so E_conserved must not move either
    let refreshed = d.tick(f64::INFINITY, &mut never());
    assert_eq!(refreshed.step, 50);
    assert_eq!(refreshed.model_epoch, 1);
    assert_eq!(refreshed.eval.energy_surface(), Surface::Model);
    assert_eq!(refreshed.scene, before.scene, "nothing moved");
    let jump = refreshed.eval.refit_jump.expect("the first evaluate after the refit reports the jump");
    let delta = refreshed.energy_conserved - before.energy_conserved;
    println!("E_c before {:.12}, after the refresh {:.12} (Δ {:+.2e}); refit jump {jump:+.3e} Ha (E_ks {:.9} → {:.9}); source {:?}",
             before.energy_conserved, refreshed.energy_conserved, delta, before.energy_ks, refreshed.energy_ks, refreshed.source);
    assert!((refreshed.energy_ks - before.energy_ks - jump).abs() < 1e-12, "the jump is the change of the reported energy");
    assert!(jump.abs() > 1e-5, "test premise: the surfaces differ visibly");
    assert!(delta.abs() < 1e-12, "E_conserved moved by {delta:+.3e} across the refit");
    assert!(refreshed.history.len() == before.history.len(), "no history restart");
    for _ in 0..50 {
        d.apply(DynCommand::Step(1));
        let snap = d.tick(f64::INFINITY, &mut never());
        if snap.source == Source::Surrogate {
            assert_eq!(snap.teacher_gap, None);
        }
        e_c.push(snap.energy_conserved);
        sources.push(snap.source);
    }
    let n_sur = sources.iter().filter(|&&s| s == Source::Surrogate).count();
    let n_fb = sources.iter().filter(|&&s| s == Source::Fallback).count();
    let fluct = e_c.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f64::max);
    println!("100 steps: {n_sur} surrogate, {n_fb} fallback; max |ΔE_c| per step {fluct:.2e} Ha; fallback fraction (last 100) {:.2}", d.snapshot().fallback_fraction_100);
    assert!(n_sur > 0, "the trained model answers somewhere on the path");
    // no step jumps: the fallback steps' force-mismatch work is booked, so only the Verlet
    // swing of this (0.15 Bohr off r_e) trajectory remains, the same as on the teacher alone
    println!("force-mismatch work booked over the run: {:+.3e} Ha; plain Verlet max per-step change {ref_step:.2e}", d.snapshot().work_mismatch);
    assert!(fluct < 1.5 * ref_step, "a jump: {fluct:.2e} vs the Verlet swing {ref_step:.2e}");
    assert!((d.snapshot().fallback_fraction_100 - n_fb as f64 / 100.0).abs() < 1e-12);
    // the history records each step's source (the viewer's sparkline colours by it): its
    // last 100 entries are exactly the 100 steps' sources, in order
    let snap = d.snapshot();
    assert_eq!(snap.history_source.len(), snap.history.len());
    assert_eq!(&snap.history_source[snap.history_source.len() - 100..], &sources[sources.len() - 100..]);
}

/// Review (d), ruling S8: on model-surface fallback steps the teacher's force moves the atoms
/// while the invariant lives on the model surface; ΔF·Δr is booked into W, so E_conserved is
/// flat to the Verlet swing, while the unbooked quantity (E_c + work_mismatch) oscillates with
/// the difference of the two surfaces along the path.
#[test]
fn fallback_force_mismatch_work_is_booked() {
    // forced through the envelope (see `fallback_step_uses_dft_forces_but_books_the_model_energy`)
    let always_fallback = FallbackPolicy { max_descriptor_distance: -1.0, audit_every: 0, ..FallbackPolicy::default() };
    let (s, h) = trained(1.2, 2.2, 40, always_fallback);
    h.lock().unwrap().d *= 1.25; // the live teacher's well is 25 % deeper than the model's
    let mut scene = h2(1.55);
    scene.nuclei[0].vel = [-3e-4, 0.0, 0.0];
    scene.nuclei[1].vel = [3e-4, 0.0, 0.0];
    // the same trajectory on the teacher alone: what Verlet itself does to E_conserved
    // (the dimer starts 0.15 Bohr off r_e, 4e-3 Ha of oscillation energy, so the swing is
    // tens of µHa — far above the 1e-7 of a near-equilibrium start)
    let mut live = MorseDimer::h2();
    live.d *= 1.25;
    let (ref_spread, _) = verlet_reference(&scene, live, 200);
    let mut d = Dynamics::new(scene, Box::new(s), 0.25, 1);
    let e0 = d.snapshot().energy_conserved;
    let (mut lo_b, mut hi_b, mut lo_u, mut hi_u) = (e0, e0, e0, e0);
    for _ in 0..200 {
        d.apply(DynCommand::Step(1));
        let snap = d.tick(f64::INFINITY, &mut never());
        assert_eq!(snap.source, Source::Fallback);
        assert!(snap.eval.model_forces.is_some());
        let b = snap.energy_conserved;
        let u = snap.energy_conserved + snap.work_mismatch;
        lo_b = lo_b.min(b);
        hi_b = hi_b.max(b);
        lo_u = lo_u.min(u);
        hi_u = hi_u.max(u);
    }
    let snap = d.snapshot();
    println!("200 forced-fallback steps: E_conserved spread {:.2e} Ha booked, {:.2e} Ha unbooked; plain Verlet on the teacher {:.2e}; Σ mismatch work {:+.3e} Ha; last teacher − model gap {:+.2e}",
             hi_b - lo_b, hi_u - lo_u, ref_spread, snap.work_mismatch, snap.teacher_gap.unwrap());
    // booked: the Verlet swing and nothing else (a wrong sign would double the unbooked spread)
    assert!(hi_b - lo_b < 1.5 * ref_spread, "booked {:.2e} vs the Verlet swing {ref_spread:.2e}", hi_b - lo_b);
    assert!(hi_u - lo_u > 10.0 * (hi_b - lo_b), "unbooked must visibly move: {:.2e} vs {:.2e}", hi_u - lo_u, hi_b - lo_b);
    assert!(snap.work_mismatch.abs() > 1e-6);
    assert_eq!(snap.fallback_fraction_100, 1.0);
}

/// Ruling S7: every `audit_every`-th answered step calls the teacher anyway; a passing audit
/// is an ordinary model-surface fallback, a failed one (the teacher drifted away) marks the
/// model untrusted and flips to the teacher surface with the gap booked as `refit_jump`, so
/// E_conserved stays continuous and the next refit clears the flag.
#[test]
fn audit_calls_the_teacher_and_a_failed_audit_flips_the_surface_continuously() {
    let policy = FallbackPolicy { audit_every: 5, ..FallbackPolicy::default() };
    let (s, h) = trained(1.2, 2.2, 40, policy);
    let mut scene = h2(1.5);
    scene.nuclei[1].vel = [2e-4, 0.0, 0.0];
    let (p, sh_d) = driven(s);
    let mut d = Dynamics::new(scene, p, 0.25, 1); // answered step 1
    assert_eq!(d.snapshot().source, Source::Surrogate);
    let mut e_c = vec![d.snapshot().energy_conserved];
    for i in 0..4 {
        d.apply(DynCommand::Step(1));
        let snap = d.tick(f64::INFINITY, &mut never());
        e_c.push(snap.energy_conserved);
        if i == 3 {
            // the 5th answered step is audited: teacher called, gap tiny, still trusted
            assert_eq!(snap.source, Source::Fallback, "{:?}", snap.eval.notes);
            assert!(snap.eval.notes.iter().any(|n| n.contains("surrogate audit (every 5 answered steps)")), "{:?}", snap.eval.notes);
            let gap = snap.teacher_gap.expect("a passing audit is a model-surface fallback");
            println!("passing audit: teacher − model {gap:+.2e} Ha");
            assert!(gap.abs() < 1e-5);
            assert_eq!(snap.eval.refit_jump, None);
        } else {
            assert_eq!(snap.source, Source::Surrogate, "{:?}", snap.eval.notes);
        }
    }
    assert_eq!(calls(&h), 1);
    assert!(surrogate_of(&sh_d).trusted());
    assert_eq!(surrogate_of(&sh_d).samples(), 41);
    // the teacher drifts away from the model
    h.lock().unwrap().d *= 1.1;
    for i in 0..5 {
        d.apply(DynCommand::Step(1));
        let snap = d.tick(f64::INFINITY, &mut never());
        e_c.push(snap.energy_conserved);
        if i == 4 {
            let jump = snap.eval.refit_jump.expect("the failed audit books the flip");
            let delta = e_c[e_c.len() - 1] - e_c[e_c.len() - 2];
            println!("failed audit: jump {jump:+.3e} Ha, ΔE_c across the flip {delta:+.2e} Ha, notes {:?}", snap.eval.notes);
            assert_eq!(snap.source, Source::Fallback);
            assert!(snap.eval.notes.iter().any(|n| n.contains("surrogate audit failed")), "{:?}", snap.eval.notes);
            assert_eq!(snap.teacher_gap, None);
            assert_eq!(snap.eval.energy_surface(), Surface::Teacher);
            assert!(jump.abs() / 2.0 > 1e-3, "test premise: the drift exceeds the per-atom bound");
            // The jump itself is booked exactly (same positions). What remains is the
            // mixed-force half-kick of this one step: its first half-kick used the model's
            // forces at r_prev (a Surrogate step), its second the teacher's at r_new, and
            // relative to the teacher surface the first was an external force
            // F_model − F_teacher whose work ½ dt v·ΔF (≈ 5·10·2e-4·5e-3 ≈ 5e-6 here,
            // measured 1e-6) cannot be booked without F_teacher(r_prev). A refit avoids
            // this by refreshing at unchanged positions; an audit flips mid-trajectory,
            // once. Bound it at 1e-5 against the 1e-2 Ha jump it replaced.
            assert!(delta.abs() < 1e-5, "E_conserved must not see the flip beyond the half-kick remainder: {delta:+.2e}");
            assert!(snap.notes.iter().all(|n| !n.contains("history restarts")), "{:?}", snap.notes);
        } else {
            assert_eq!(snap.source, Source::Surrogate, "{:?}", snap.eval.notes);
        }
    }
    assert_eq!(calls(&h), 2);
    assert!(!surrogate_of(&sh_d).trusted());
    assert!(surrogate_of(&sh_d).audit_failed().is_some());
    // every step until a refit is a teacher-surface fallback with the reason
    d.apply(DynCommand::Step(1));
    let snap = d.tick(f64::INFINITY, &mut never());
    assert_eq!(snap.source, Source::Fallback);
    assert!(snap.eval.notes.iter().any(|n| n.contains("surrogate untrusted: audit found teacher − model")), "{:?}", snap.eval.notes);
    assert_eq!(snap.eval.energy_surface(), Surface::Teacher);
    let fluct = e_c.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f64::max);
    println!("max |ΔE_c| per step over the run incl. the flip: {fluct:.2e} Ha");
    assert!(fluct < 1e-5);
    // a refit clears the audit flag (whatever the residuals then say)
    let mut s = surrogate_of(&sh_d);
    s.refit().unwrap();
    assert!(s.audit_failed().is_none());
    println!("after the refit over the drifted samples: trusted {} ({:?})", s.trusted(), s.training_residuals());
}

/// Task 5 carry-over (Task 4 re-review): a failed audit on the refresh evaluation right after
/// a refit must book the refit's pending jump AND the audit's gap — the reported energy at the
/// unchanged positions moves from the old model's to the teacher's, which is both.
#[test]
fn a_failed_audit_on_the_refresh_after_a_refit_books_the_pending_jump_too() {
    let policy = FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() };
    let (s, h) = trained(1.2, 2.2, 40, policy);
    let mut scene = h2(1.5);
    scene.nuclei[1].vel = [2e-4, 0.0, 0.0];
    let (p, sh_d) = driven(s);
    let mut d = Dynamics::new(scene, p, 0.25, 1);
    for _ in 0..5 {
        d.apply(DynCommand::Step(1));
        assert_eq!(d.tick(f64::INFINITY, &mut never()).source, Source::Surrogate);
    }
    let before = d.snapshot();
    {
        let mut s = surrogate_of(&sh_d);
        // a refit that moves the model surface a little (scan samples from a 0.5 % deeper well)
        let mut deeper = MorseDimer::h2();
        deeper.d *= 1.005;
        let mut rng = StdRng::seed_from_u64(9);
        for i in 0..20 {
            let r = 1.2 + i as f64 / 19.0;
            let sc = dimer(r, unit_vector(&mut rng), [0.0; 3], 1, 1);
            s.add_sample(morse_sample(&deeper, &sc), &sc).unwrap();
        }
        s.refit().unwrap();
        assert!(s.trusted(), "{:?}", s.untrusted_reason());
        let pos: Vec<[f64; 3]> = before.scene.nuclei.iter().map(|n| n.pos).collect();
        let pending = s.model().unwrap().predict(&pos, &[1, 1], false).unwrap().energy - before.energy_ks;
        println!("the refit's pending jump at r_last: {pending:+.3e} Ha");
        assert!(pending.abs() > 1e-6, "test premise: the refit moved the surface visibly");
        // every answered step is audited from now on, and the teacher has drifted 10 %:
        // the refresh evaluation is a failing audit
        s.set_policy(FallbackPolicy { audit_every: 1, ..FallbackPolicy::default() });
    }
    h.lock().unwrap().d *= 1.1;
    let refreshed = d.tick(f64::INFINITY, &mut never());
    assert_eq!(refreshed.scene, before.scene, "nothing moved");
    assert!(refreshed.eval.notes.iter().any(|n| n.contains("surrogate audit failed")), "{:?}", refreshed.eval.notes);
    assert_eq!(refreshed.eval.energy_surface(), Surface::Teacher);
    let jump = refreshed.eval.refit_jump.expect("the flip is booked");
    let delta = refreshed.energy_conserved - before.energy_conserved;
    let reported = refreshed.energy_ks - before.energy_ks;
    println!("refresh = failed audit: booked jump {jump:+.6e} Ha, reported energy change {reported:+.6e} Ha, ΔE_c {delta:+.2e}");
    assert!((jump - reported).abs() < 1e-12, "the booked jump is the whole change of the reported energy");
    assert!(delta.abs() < 1e-12, "E_conserved moved by {delta:+.3e} across the refresh");
    assert!(refreshed.notes.iter().all(|n| !n.contains("history restarts")), "{:?}", refreshed.notes);
}

/// Ruling S15, fallback thinning (module doc, "Trust recovery"): fallbacks at one geometry
/// while the model is untrusted for its residuals (not for want of data) keep one sample in
/// [`THIN_EVERY`]; a geometry far from every stored one is kept at once; while the trust rule
/// still needs samples (n_samples < Q′) every answer is kept.
#[test]
fn thinning_keeps_one_in_ten_near_identical_fallbacks_and_every_far_one() {
    let (mut s, _) = trained(1.2, 2.2, 40, FallbackPolicy::default());
    assert!(s.trusted());
    // untrusted by its residuals (an impossible energy threshold), not by its sample count
    s.set_policy(FallbackPolicy { energy_std_per_atom: 1e-12, ..FallbackPolicy::default() });
    assert!(s.untrusted_reason().unwrap().starts_with("max training energy residual"));
    let n0 = s.samples();
    let mut thinned_notes = 0;
    for _ in 0..25 {
        let r = s.evaluate(&h2(1.6), &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback);
        thinned_notes += r.stats.notes.iter().filter(|n| n.contains("sample not kept")).count();
    }
    println!("25 fallbacks at r = 1.6: {} kept, {thinned_notes} thinned", s.samples() - n0);
    assert_eq!(s.samples() - n0, 3, "the 1st, 11th and 21st are kept");
    assert_eq!(thinned_notes, 22);
    // far from every stored geometry: kept right away (4 thinned since the last kept one)
    let far = s.nearest_stored(&h2(4.0).nuclei.iter().map(|n| n.pos).collect::<Vec<_>>(), &[1, 1]);
    s.evaluate(&h2(4.0), &mut never()).unwrap();
    println!("r = 4.0 is {far:.1} training spreads from the nearest stored sample: kept");
    assert!(far > THIN_FRACTION * FallbackPolicy::default().max_descriptor_distance);
    assert_eq!(s.samples() - n0, 4);
    // a model that still needs data keeps every answer
    let (mut few, _) = trained(1.2, 2.2, 5, FallbackPolicy::default());
    assert!(few.untrusted_reason().unwrap().starts_with("n_samples"));
    let m0 = few.samples();
    for _ in 0..5 {
        few.evaluate(&h2(1.6), &mut never()).unwrap();
    }
    assert_eq!(few.samples() - m0, 5);
}

/// Ruling S15, quarantine and supersession: one stray sample (the teacher's energy at an
/// in-window geometry 0.1 Ha off, as a glitched solve would give) untrusted the whole model
/// before fix round 3, for good. The refit quarantines it: trusted again, the sample held
/// (store, sidecar) but excluded, the window around it still the model's. A fresh teacher
/// answer at that geometry agrees with the model, so the stray sample is superseded
/// (contradicted, still held and excluded, no longer fencing) and the fresh answer kept; the
/// geometry is the model's again. The codes survive the sidecar; Forget clears them.
#[test]
fn a_stray_sample_is_quarantined_then_superseded_and_trust_returns() {
    let (mut s, h) = trained(1.2, 2.2, 40, FallbackPolicy::default());
    let stray = dimer(1.63, [0.0, 1.0, 0.0], [0.1, 0.0, 0.0], 1, 1);
    let mut bad = morse_sample(&MorseDimer::h2(), &stray);
    bad.energy += 0.1;
    s.add_sample(bad, &stray).unwrap();
    let rep = s.refit().unwrap();
    println!("after the stray sample: {} samples, rmse E/atom {:.1e}; trusted {}; quarantined {}; note {:?}",
             rep.n_samples, rep.rmse_energy_per_atom, s.trusted(), s.quarantined(), s.recovery_note());
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    assert_eq!(s.quarantined(), 1);
    assert!(s.is_quarantined(40), "the stray sample (index 40) is the one held back");
    assert!(s.recovery_note().unwrap().contains("1 samples quarantined"));
    assert_eq!(s.samples(), 41, "never silently dropped");
    assert_eq!(rep.n_samples, 40, "excluded from the fit");
    let near = |s: &mut SurrogateKs, r: f64| {
        s.quarantine_distances(&h2(r).nuclei.iter().map(|n| n.pos).collect::<Vec<_>>(), &[1, 1]).into_iter().map(|x| x.1).fold(f64::INFINITY, f64::min)
    };
    let (d18, d20) = (near(&mut s, 1.8), near(&mut s, 2.0));
    let r18 = s.evaluate(&h2(1.8), &mut never()).unwrap();
    println!("from the quarantined geometry: r = 1.8 at {d18:.2}, r = 2.0 at {d20:.2} training spreads; r = 1.8 answered by {:?}", r18.stats.source);
    assert_eq!(r18.stats.source, Source::Surrogate, "the window around the stray sample is still the model's");
    // the code in the sidecar: a reloaded model fences the geometry too
    let path = std::env::temp_dir().join(format!("quarantine-{}.model", std::process::id()));
    s.save_sidecar(&path).unwrap();
    let mut back = SurrogateKs::new(Box::new(MorseDimer::h2()), FallbackPolicy::default()).with_teacher_tol(1e-6);
    back.install_sidecar(surrogate::sidecar::read(&path).unwrap()).unwrap();
    assert_eq!((back.quarantined(), back.is_quarantined(40), back.trusted()), (1, true, true));
    // the geometry falls back; the teacher (an exact Morse) agrees with the model there, so
    // the stray sample is superseded and the fresh answer kept
    let calls0 = calls(&h);
    let r = s.evaluate(&stray, &mut never()).unwrap();
    println!("at the stray geometry: {:?}, notes {:?}", r.stats.source, r.stats.notes);
    assert_eq!(r.stats.source, Source::Fallback);
    assert!(r.stats.notes.iter().any(|n| n.contains("quarantined geometry")), "{:?}", r.stats.notes);
    assert!(r.stats.notes.iter().any(|n| n.contains("0 released, 1 superseded")), "{:?}", r.stats.notes);
    assert_eq!((s.samples(), calls(&h), s.quarantined(), s.superseded()), (42, calls0 + 1, 0, 1));
    s.refit().unwrap();
    assert!(s.trusted());
    assert_eq!(s.evaluate(&stray, &mut never()).unwrap().stats.source, Source::Surrogate, "no longer fenced");
    let _ = std::fs::remove_file(&path);
    // Forget clears it
    s.forget();
    assert_eq!((s.quarantined(), s.superseded()), (0, 0));
}

/// A long-range Morse teacher (a = 0.3: at R = 8 the pair still binds by 4.4e-2 Ha, while
/// every feature of the 6 Bohr cutoff model has vanished and it can only say Σ E_ref) — a
/// geometry genuinely unrepresentable. Task 8 fix round 4 (the re-review's Important): the
/// envelope rule fired first there, the near-quarantine check was skipped, every 10th
/// answer was kept and quarantined in turn until the 10 % cap untrusted the model for good.
/// Now 600 teacher steps beyond the cutoff (R 7–10), with a refit every 25 as the worker
/// does, keep the trust, and R = 1.6 is the model's. Since Task 9 fix round 1 (fence voting,
/// I3) the store grows by exactly one sample: the first three mutually consistent answers
/// that disagree with the model are a vote, the next refit cannot reproduce the voted sample,
/// quarantines it and closes the region to further votes — the chain stays bounded.
#[test]
fn a_molecule_parked_beyond_the_cutoff_keeps_one_vote_and_never_untrusts_the_model() {
    let long = MorseDimer { a: 0.3, ..MorseDimer::h2() };
    let teacher = Shared::new(MorseDimer { a: 0.3, ..MorseDimer::h2() });
    let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_teacher_tol(1e-6);
    s.set_reference_energy(1, long.e_atom);
    let mut rng = StdRng::seed_from_u64(9);
    for i in 0..40 {
        let r = 1.2 + 1.0 * i as f64 / 39.0;
        let sc = dimer(r, unit_vector(&mut rng), [0.0; 3], 1, 1);
        s.add_sample(morse_sample(&long, &sc), &sc).unwrap();
    }
    s.refit().unwrap();
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    // the first answer beyond the cutoff: kept (far from everything), then quarantined
    let r = s.evaluate(&h2(8.0), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback);
    s.refit().unwrap();
    let (n1, q1) = (s.samples(), s.quarantined());
    println!("after R = 8: {n1} samples, {q1} quarantined, trusted {}, note {:?}", s.trusted(), s.recovery_note());
    assert!(s.trusted() && q1 == 1, "{:?}", s.untrusted_reason());
    let mut rejected_vote = false;
    for k in 0..600 {
        let r = 7.0 + 3.0 * ((k as f64) * 0.37).sin().abs();
        let ev = s.evaluate(&h2(r), &mut never()).unwrap();
        assert_eq!(ev.stats.source, Source::Fallback, "R = {r} beyond the cutoff");
        if k % 25 == 24 && s.pending_refit() {
            s.refit().unwrap();
            rejected_vote |= s.recovery_note().is_some_and(|n| n.contains("1 voted samples quarantined"));
        }
    }
    println!("after 600 steps beyond the cutoff: {} samples, {} quarantined, trusted {}", s.samples(), s.quarantined(), s.trusted());
    assert!(rejected_vote, "the refit judged the vote");
    assert_eq!((s.samples(), s.quarantined()), (n1 + 1, q1 + 1), "one vote, quarantined; nothing more kept");
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    assert_eq!(s.evaluate(&h2(1.6), &mut never()).unwrap().stats.source, Source::Surrogate);
}

/// Release at a refit (module doc, "Release"): a sample 5e-3 Ha off (2.5e-3 Ha/atom) is
/// quarantined under the default 1e-3 Ha/atom threshold; once the policy accepts 1e-2 the
/// next refit finds the model reproducing it, releases it into the fit and stays trusted.
/// (A Morse tail beyond what the cutoff can carry was tried first and is never reproduced:
/// that is what quarantine is for.)
#[test]
fn a_quarantined_sample_is_released_when_the_model_reproduces_it() {
    let (mut s, _) = trained(1.2, 2.2, 40, FallbackPolicy::default());
    let off = dimer(1.7, [0.0, 0.0, 1.0], [0.0; 3], 1, 1);
    let mut sample = morse_sample(&MorseDimer::h2(), &off);
    sample.energy += 5e-3;
    s.add_sample(sample, &off).unwrap();
    s.refit().unwrap();
    println!("5e-3 Ha off: trusted {}, quarantined {}, note {:?}", s.trusted(), s.quarantined(), s.recovery_note());
    assert!(s.trusted() && s.is_quarantined(40), "{:?}", s.untrusted_reason());
    s.set_policy(FallbackPolicy { energy_std_per_atom: 1e-2, ..FallbackPolicy::default() });
    let rep = s.refit().unwrap();
    println!("threshold 1e-2: trusted {}, quarantined {}, fitted {} samples, note {:?}", s.trusted(), s.quarantined(), rep.n_samples, s.recovery_note());
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    assert_eq!((s.quarantined(), rep.n_samples), (0, 41));
    assert!(s.recovery_note().unwrap().contains("1 quarantined samples released"), "{:?}", s.recovery_note());
}

/// A Morse teacher that reports a face density of the test's choosing (the number a grid
/// solver puts in `EvalStats::face_density`).
struct FaceDensity {
    inner: Arc<Mutex<MorseDimer>>,
    rho: Arc<Mutex<f64>>,
}

impl ForceProvider for FaceDensity {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        let mut r = self.inner.lock().unwrap().evaluate(scene, should_abort)?;
        r.stats.face_density = Some(*self.rho.lock().unwrap());
        Ok(r)
    }
    fn name(&self) -> &'static str {
        "morse dimer"
    }
    fn reset(&mut self) {}
}

/// Box-affected answers (fix round 4; Task 9: the solver's number, `EvalStats::face_density`,
/// not its warning text): a fallback answer whose teacher saw more than
/// `BOX_AFFECTED_FACE_DENSITY` on the faces is used for the step and not kept; one below it is
/// kept. The geometry (R = 2.6 against training on [1.1, 2.0]) is out of the envelope and far
/// from every stored sample, so nothing else would drop it.
#[test]
fn a_box_affected_answer_is_used_but_not_trained_on() {
    let rho = Arc::new(Mutex::new(0.0));
    let teacher = FaceDensity { inner: Arc::new(Mutex::new(MorseDimer::h2())), rho: rho.clone() };
    let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_teacher_tol(1e-6);
    s.set_reference_energy(1, MorseDimer::h2().e_atom);
    let mut rng = StdRng::seed_from_u64(4);
    for i in 0..40 {
        let r = 1.1 + 0.9 * i as f64 / 39.0;
        let origin = [rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3)];
        let sc = dimer(r, unit_vector(&mut rng), origin, 1, 1);
        s.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
    }
    s.refit().unwrap();
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    let n0 = s.samples();
    *rho.lock().unwrap() = 2.0 * BOX_AFFECTED_FACE_DENSITY;
    let r = s.evaluate(&h2(2.6), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback);
    assert_eq!(r.stats.face_density, Some(2.0 * BOX_AFFECTED_FACE_DENSITY));
    assert!(r.stats.notes.iter().any(|n| n.contains("box-affected")), "{:?}", r.stats.notes);
    assert_eq!(s.samples(), n0, "a box-affected answer was kept");
    *rho.lock().unwrap() = 0.5 * BOX_AFFECTED_FACE_DENSITY;
    let r = s.evaluate(&h2(2.6), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback);
    assert_eq!(s.samples(), n0 + 1, "an answer inside the box rule was not kept: {:?}", r.stats.notes);
}

/// Final review B2 (module doc, "No model yet"): with no model (a pass offer waiting, a refused
/// autoload) every DFT step used to keep a sample with no refit and no bound. Up to
/// NO_MODEL_SAMPLE_FACTOR · Q′ (4 × 29 for H at D = 4) they are kept; beyond, one in
/// THIN_EVERY, with a note saying how many are held.
#[test]
fn without_a_model_the_kept_answers_are_capped() {
    let mut s = SurrogateKs::new(Box::new(MorseDimer::h2()), FallbackPolicy::default()).with_teacher_tol(1e-6);
    let cap = NO_MODEL_SAMPLE_FACTOR * Basis::new(&[1]).column_classes().len();
    assert_eq!(cap, 116);
    let mut last = None;
    for i in 0..cap + 34 {
        last = Some(s.evaluate(&h2(1.2 + 0.004 * i as f64), &mut never()).unwrap());
    }
    // 116 kept, then of 34 more: 3 × (9 dropped, 1 kept) + 4 dropped
    assert_eq!(s.samples(), cap + 3);
    let notes = last.unwrap().stats.notes;
    assert!(notes.iter().any(|n| n.contains("waiting for a sampling pass — 119 samples held")), "{notes:?}");
}

/// The auditor's teacher for the asynchronous-audit tests: the same (shared, drift-able) Morse
/// as the main teacher, answering only once `open` is set (or abandoning the solve when the
/// surrogate asks), and reporting a face density of the test's choosing.
struct Gated {
    inner: Arc<Mutex<MorseDimer>>,
    open: Arc<std::sync::atomic::AtomicBool>,
    rho: f64,
    calls: Arc<std::sync::atomic::AtomicUsize>,
}

impl ForceProvider for Gated {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        use std::sync::atomic::Ordering::SeqCst;
        while !self.open.load(SeqCst) {
            if should_abort() {
                return Err(ProviderError::Cancelled);
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        self.calls.fetch_add(1, SeqCst);
        // a stand-in density (side densities: what the app shows), tagged with the nuclei
        let display = Some(Display::Cpu { cell: scene.cell.clone(), origin: [0.0; 3], rho: Arc::new(vec![scene.nuclei.len() as f32]), max: 1.0 });
        if scene.nuclei.len() == 1 {
            // an isolated atom (the Morse needs two): its reference energy, no force
            let e = self.inner.lock().unwrap().e_atom;
            return Ok(ForceResult { energy: e, forces: vec![[0.0; 3]], stats: EvalStats::default(), display });
        }
        let mut r = self.inner.lock().unwrap().evaluate(scene, should_abort)?;
        r.stats.face_density = Some(self.rho);
        r.display = display;
        Ok(r)
    }
    fn name(&self) -> &'static str {
        "morse dimer"
    }
    fn reset(&mut self) {}
}

/// A trained H₂ surrogate (audit every `every`) with a gated background auditor on the same
/// Morse; returns it with the teacher handle, the gate and the auditor's call counter.
#[allow(clippy::type_complexity)]
fn with_gated_auditor(every: usize, rho: f64) -> (SurrogateKs, Arc<Mutex<MorseDimer>>, Arc<std::sync::atomic::AtomicBool>, Arc<std::sync::atomic::AtomicUsize>) {
    let (s, h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: every, ..FallbackPolicy::default() });
    let open = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let auditor = Gated { inner: h.clone(), open: open.clone(), rho, calls: calls.clone() };
    (s.with_auditor(Box::new(auditor)), h, open, calls)
}

/// Steps `d` one step at a time until `f` holds (≤ 2 s; the background answer arrives on its
/// own thread), returning the snapshots seen.
fn step_until(d: &mut Dynamics, what: &str, f: impl Fn(&mut Dynamics) -> bool) -> Vec<DynSnapshot> {
    let t0 = Instant::now();
    let mut seen = Vec::new();
    while !f(d) {
        assert!(t0.elapsed().as_secs_f64() < 2.0, "{what}: not within 2 s");
        d.apply(DynCommand::Step(1));
        seen.push(d.tick(f64::INFINITY, &mut never()));
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    seen
}

/// Ruling S14: with a background auditor the audit step is the model's own answer — the
/// teacher solve runs on the auditor's thread while the model keeps stepping (here: held
/// back by a gate for 20 steps), and the verdict is recorded when it arrives, judged by the
/// current model, with the sample kept and the steps in between counted. A Forget abandons an
/// audit in flight: its answer is discarded.
#[test]
fn a_background_audit_runs_while_the_model_answers() {
    let (s, h, open, audit_calls) = with_gated_auditor(5, 0.0);
    assert!(s.audits_asynchronous());
    let mut scene = h2(1.5);
    scene.nuclei[1].vel = [2e-4, 0.0, 0.0];
    let (p, sh_d) = driven(s);
    let mut d = Dynamics::new(scene, p, 0.25, 1); // answered step 1
    let teacher_calls = calls(&h);
    let mut dispatched = None;
    for i in 0..24 {
        d.apply(DynCommand::Step(1));
        let snap = d.tick(f64::INFINITY, &mut never());
        assert_eq!(snap.source, Source::Surrogate, "step {i}: {:?}", snap.eval.notes);
        if snap.eval.notes.iter().any(|n| n.contains("surrogate audit dispatched")) {
            dispatched.get_or_insert(i);
        }
    }
    assert_eq!(dispatched, Some(3), "the 5th answered step dispatches the audit");
    let sk = surrogate_of(&sh_d);
    assert!(sk.audit_in_flight() && sk.last_audit().is_none());
    assert_eq!(calls(&h), teacher_calls, "the main teacher was never called");
    let n0 = sk.samples();
    drop(sk);
    // the gate opens: the verdict arrives with a later step
    open.store(true, std::sync::atomic::Ordering::SeqCst);
    let seen = step_until(&mut d, "the background verdict", |_| surrogate_of(&sh_d).last_audit().is_some());
    let note = seen.last().unwrap().eval.notes.iter().find(|n| n.contains("surrogate audit passed")).cloned();
    let sk = surrogate_of(&sh_d);
    let a = sk.last_audit().unwrap().clone();
    println!("background audit: {a:?}; note {note:?}");
    assert_eq!((a.verdict, a.asynchronous), (AuditVerdict::Passed, true));
    assert!(a.gap.abs() < 1e-5 && a.unaudited_steps >= 20, "{a:?}");
    assert!(note.is_some(), "{:?}", seen.last().unwrap().eval.notes);
    assert_eq!(sk.samples(), n0 + 1, "an audit's answer is a sample");
    assert!(sk.trusted() && sk.pending_refit());
    assert!(seen.iter().all(|s| s.source == Source::Surrogate));
    drop(sk);
    // a Forget abandons the audit in flight (gate shut again; the next one is dispatched)
    open.store(false, std::sync::atomic::Ordering::SeqCst);
    step_until(&mut d, "the next dispatch", |_| surrogate_of(&sh_d).audit_in_flight());
    let before = audit_calls.load(std::sync::atomic::Ordering::SeqCst);
    surrogate_of(&sh_d).forget();
    open.store(true, std::sync::atomic::Ordering::SeqCst);
    std::thread::sleep(std::time::Duration::from_millis(50));
    d.apply(DynCommand::Step(1));
    d.tick(f64::INFINITY, &mut never());
    let sk = surrogate_of(&sh_d);
    assert!(sk.last_audit().is_none() && !sk.audit_in_flight(), "{:?}", sk.last_audit());
    assert_eq!(audit_calls.load(std::sync::atomic::Ordering::SeqCst), before, "the abandoned job never reached the teacher");
}

/// Draft mode (module doc): an out-of-envelope geometry is answered by the model — the
/// teacher never called — with the extrapolation flagged (atoms outside, the rule that would
/// have fallen back) and the per-atom uncertainty ratios; the same geometry falls back with
/// draft mode off; an untrusted fit answers too, flagged with the trust rule's reason; a
/// provenance or species mismatch still goes to the teacher (nothing to guess with).
#[test]
fn draft_mode_answers_beyond_the_envelope_and_flags_it() {
    let draft = FallbackPolicy { audit_every: 0, never_fallback: true, ..FallbackPolicy::default() };
    let (mut s, h) = trained(1.2, 2.2, 40, draft);
    assert!(s.trusted());
    let c0 = calls(&h);
    let r = s.evaluate(&h2(4.0), &mut never()).unwrap();
    let x = r.stats.extrapolation.clone().expect("flagged");
    println!("draft at R = 4: {:?}; per-atom uncertainty {:?}", x, r.stats.atom_uncertainty);
    assert_eq!(r.stats.source, Source::Surrogate);
    assert!(x.atoms_outside == 2 && x.untrusted.is_none() && x.atom_envelope.iter().all(|&d| d > 1.0), "{x:?}");
    assert!(x.reason.as_deref().is_some_and(|m| m.contains("outside the descriptor envelope")), "{x:?}");
    assert_eq!(r.stats.atom_uncertainty.as_ref().map(Vec::len), Some(2));
    assert_eq!(calls(&h), c0, "the teacher was never called");
    // inside the envelope: an ordinary answer, nothing flagged, the ratios below 1
    let r = s.evaluate(&h2(1.6), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Surrogate);
    assert!(r.stats.extrapolation.is_none());
    assert!(r.stats.atom_uncertainty.as_ref().unwrap().iter().all(|&u| u < 1.0), "{:?}", r.stats.atom_uncertainty);
    // draft off: R = 4 is the teacher's
    s.set_policy(FallbackPolicy { never_fallback: false, ..*s.policy() });
    assert_eq!(s.evaluate(&h2(4.0), &mut never()).unwrap().stats.source, Source::Fallback);
    // mismatches still refuse in draft mode: another grid (rule 3), another species (rule 2)
    s.set_policy(FallbackPolicy { never_fallback: true, ..*s.policy() });
    let mut other = h2(1.6);
    other.cell = Cell { n: 41, h: 0.25 };
    let r = s.evaluate(&other, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback, "{:?}", r.stats.notes);
    assert!(r.stats.extrapolation.is_none());
    let r = s.evaluate(&dimer(1.6, [1.0, 0.0, 0.0], [0.0; 3], 1, 2), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback, "{:?}", r.stats.notes);
    // an untrusted fit (10 samples < Q′) answers, flagged with the trust rule's reason
    let (mut u, _) = trained(1.2, 2.2, 10, draft);
    assert!(u.model().is_some() && !u.trusted());
    let r = u.evaluate(&h2(1.6), &mut never()).unwrap();
    let x = r.stats.extrapolation.clone().expect("flagged");
    println!("draft, untrusted: {x:?}");
    assert_eq!(r.stats.source, Source::Surrogate);
    assert!(x.untrusted.as_deref().is_some_and(|m| m.contains("untrusted")), "{x:?}");
}

/// Review M7 (fix round 1): paused with an audit in flight, a side request is not stuck until
/// Play — the audit's answer is taken outside `evaluate` and held unjudged, the queued atom job
/// goes out and arrives, and the audit's verdict is taken at the next evaluate.
#[test]
fn a_side_request_made_while_paused_behind_an_audit_still_runs() {
    let (s, _h, open, _) = with_gated_auditor(5, 0.0);
    let mut scene = h2(1.5);
    scene.nuclei[1].vel = [2e-4, 0.0, 0.0];
    let (p, sh_d) = driven(s);
    let mut d = Dynamics::new(scene, p, 0.25, 1);
    step_until(&mut d, "an audit in flight", |_| surrogate_of(&sh_d).audit_in_flight());
    let sc = d.snapshot().scene;
    assert!(surrogate_of(&sh_d).request_atom_density(1, &sc), "queued behind the audit");
    open.store(true, std::sync::atomic::Ordering::SeqCst);
    let t0 = std::time::Instant::now();
    let atoms = loop {
        let a = surrogate_of(&sh_d).take_atom_densities(); // no tick, no evaluate
        if !a.is_empty() {
            break a;
        }
        assert!(t0.elapsed().as_secs_f64() < 2.0, "the atom job never went out while paused");
        std::thread::sleep(std::time::Duration::from_millis(1));
    };
    assert_eq!(atoms[0].0, 1);
    let sk = surrogate_of(&sh_d);
    assert!(sk.last_audit().is_none() && sk.audit_in_flight(), "the audit's answer is held, not judged");
    drop(sk);
    d.apply(DynCommand::Step(1));
    d.tick(f64::INFINITY, &mut never());
    assert!(surrogate_of(&sh_d).last_audit().is_some(), "judged at the next evaluate");
}

/// Review I1 (fix round 1): in draft mode a refit while the model is untrusted still moves the
/// surface the atoms are on, so its jump is booked like a trusted refit's — the driver's
/// refresh at the unchanged positions books `refit_jump` into W and E_conserved is continuous
/// to rounding (pinned: |ΔE_c| ≤ 1e-12 Ha against a jump of ≥ 1e-4 Ha); without the fix the
/// jump went unbooked (Model → Model, no restart, no note).
#[test]
fn draft_mode_books_the_jump_of_an_untrusted_refit() {
    let draft = FallbackPolicy { audit_every: 0, never_fallback: true, ..FallbackPolicy::default() };
    let (s, _h) = trained(1.2, 2.2, 10, draft);
    assert!(s.model().is_some() && !s.trusted(), "the premise: an untrusted fit");
    let mut scene = h2(1.6);
    scene.nuclei[1].vel = [2e-4, 0.0, 0.0];
    let (p, sh_d) = driven(s);
    let mut d = Dynamics::new(scene, p, 0.25, 1);
    d.apply(DynCommand::Step(20));
    let before = d.tick(f64::INFINITY, &mut never());
    assert_eq!(before.source, Source::Surrogate, "draft answers the untrusted model");
    // five labels from a deeper well: the refit moves the surface, still short of Q′ (untrusted)
    let deeper = MorseDimer { d: 0.2, ..MorseDimer::h2() };
    let mut sk = surrogate_of(&sh_d);
    for k in 0..5 {
        let sc = h2(1.3 + 0.15 * k as f64);
        sk.add_sample(morse_sample(&deeper, &sc), &sc).unwrap();
    }
    sk.refit().expect("refit");
    assert!(!sk.trusted(), "still untrusted after the refit");
    drop(sk);
    let after = d.tick(f64::INFINITY, &mut never()); // paused: the refresh at the same positions
    let jump = after.eval.refit_jump;
    println!("untrusted draft refit: jump {jump:?}, ΔE_conserved {:.2e} Ha, positions unchanged {}",
             after.energy_conserved - before.energy_conserved, after.scene == before.scene);
    assert_eq!(after.scene, before.scene);
    assert_eq!(after.source, Source::Surrogate);
    assert!(jump.is_some_and(|j| j.abs() >= 1e-4), "the jump is booked: {jump:?}");
    assert!((after.energy_conserved - before.energy_conserved).abs() <= 1e-12, "{:.2e}", after.energy_conserved - before.energy_conserved);
    assert!(after.work_refit != before.work_refit, "booked into W as a surface jump");
}

/// Draft mode keeps learning (module doc): the background audits run while the model answers
/// every step — out of the envelope here — their answers are kept as samples and a refit
/// runs on them; a failed audit (the teacher drifted) untrusts the model, which goes on
/// answering; switching draft mode off then hands the step to the teacher with the flip
/// booked (`refit_jump`), so E_conserved does not jump by the gap.
#[test]
fn draft_mode_audits_keep_teaching_and_never_block() {
    let (mut s, h, open, audit_calls) = with_gated_auditor(5, 0.0);
    s.set_policy(FallbackPolicy { never_fallback: true, ..*s.policy() });
    open.store(true, std::sync::atomic::Ordering::SeqCst);
    let mut scene = h2(3.2); // outside the trained 1.2–2.2
    scene.nuclei[1].vel = [1e-4, 0.0, 0.0];
    let (p, sh_d) = driven(s);
    let mut d = Dynamics::new(scene, p, 0.25, 1);
    let n0 = surrogate_of(&sh_d).samples();
    let seen = step_until(&mut d, "a draft-mode audit verdict", |_| surrogate_of(&sh_d).last_audit().is_some());
    assert!(seen.iter().all(|x| x.source == Source::Surrogate), "every step is the model's");
    assert!(seen.iter().any(|x| x.eval.extrapolation.as_ref().is_some_and(|e| e.atoms_outside > 0)));
    // the history marks the guessed steps (review M2: the sparkline draws them red)
    let last = seen.last().unwrap();
    assert_eq!(last.history_guess.len(), last.history.len());
    assert!(last.history_guess.iter().any(|&g| g));
    assert!(audit_calls.load(std::sync::atomic::Ordering::SeqCst) >= 1);
    let mut sk = surrogate_of(&sh_d);
    assert!(sk.samples() > n0 && sk.pending_refit(), "the audit's answer is a sample");
    sk.refit().expect("a refit on it");
    drop(sk);
    // the teacher drifts: the next audit fails, the model is untrusted and still answers
    h.lock().unwrap().d *= 1.1;
    step_until(&mut d, "a failed audit", |_| surrogate_of(&sh_d).last_audit().is_some_and(|a| a.verdict == AuditVerdict::Failed));
    assert!(!surrogate_of(&sh_d).trusted());
    d.apply(DynCommand::Step(3));
    let t = d.tick(f64::INFINITY, &mut never());
    assert_eq!(t.source, Source::Surrogate, "untrusted, still answering in draft mode");
    // draft off: the teacher's step, the flip booked at its positions
    let e0 = t.energy_conserved;
    let p = FallbackPolicy { never_fallback: false, ..*surrogate_of(&sh_d).policy() };
    surrogate_of(&sh_d).set_policy(p);
    d.apply(DynCommand::Step(1));
    let t = d.tick(f64::INFINITY, &mut never());
    println!("draft off after a failed audit: {:?}, refit_jump {:?}, ΔE_conserved {:.2e} Ha", t.source, t.eval.refit_jump, t.energy_conserved - e0);
    assert_eq!(t.source, Source::Fallback);
    assert!(t.eval.refit_jump.is_some(), "the flip is booked");
    // booked: E_conserved moves by the step's mixed forces only (first half-kick on the
    // model's, second on the teacher's: O(ΔF·Δr), the 10 % drift of a 0.05 Ha/Bohr force over
    // the step's ~0.05 Bohr), not by the 8e-3 Ha flip itself
    let jump = t.eval.refit_jump.unwrap();
    assert!((t.energy_conserved - e0).abs() < 0.1 * jump.abs(), "{:.2e} vs the flip {jump:.2e}", t.energy_conserved - e0);
}

/// Side densities (module doc): a background audit's density reaches the driver's snapshot
/// as `SideKind::Audit`, stamped with the simulated time of the geometry it was dispatched at
/// (the positions it carries are that snapshot's); a refresh asked for between ticks is at
/// the current positions and time, display-only (no sample, no verdict); an isolated-atom
/// density asked for while the auditor is busy is queued and arrives through
/// `take_atom_densities` after it. An edit clears the side density.
#[test]
fn side_densities_carry_their_kind_time_and_positions() {
    let (s, _h, open, _) = with_gated_auditor(5, 0.0);
    let mut scene = h2(1.5);
    scene.nuclei[1].vel = [2e-4, 0.0, 0.0];
    let (p, sh_d) = driven(s);
    let mut d = Dynamics::new(scene, p, 0.25, 1);
    open.store(true, std::sync::atomic::Ordering::SeqCst);
    let seen = step_until(&mut d, "an audit density", |d| d.snapshot().side_display.is_some());
    let side = d.snapshot().side_display.unwrap();
    assert_eq!(side.kind, SideKind::Audit);
    let at = seen.iter().find(|s| s.scene.nuclei.iter().map(|n| n.pos).collect::<Vec<_>>() == side.positions)
        .expect("the audited positions are a snapshot's");
    println!("audit density: solved at t = {:.3} fs (step {}), arrived at step {}", side.t_fs(), at.step, seen.last().unwrap().step);
    assert_eq!(side.t, at.sim_time, "stamped with the dispatch geometry's time");
    assert!(matches!(&side.display, Display::Cpu { rho, .. } if rho[0] == 2.0));
    // a refresh between ticks: the current positions and time, display-only
    let n0 = surrogate_of(&sh_d).samples();
    let (gap0, seq0) = (surrogate_of(&sh_d).last_audit().cloned(), side.seq);
    step_until(&mut d, "an idle auditor", |_| surrogate_of(&sh_d).auditor_idle());
    let now = d.snapshot();
    assert!(surrogate_of(&sh_d).request_display(&now.scene));
    assert!(!surrogate_of(&sh_d).auditor_idle(), "sent");
    // queued behind it (taken after the refresh, below)
    assert!(surrogate_of(&sh_d).request_atom_density(1, &now.scene));
    let t0 = std::time::Instant::now();
    let refresh = loop {
        let snap = d.tick(f64::INFINITY, &mut never()); // paused: no step, the poll only
        if let Some(sd) = snap.side_display.filter(|sd| sd.seq != seq0) {
            break sd;
        }
        assert!(t0.elapsed().as_secs_f64() < 2.0, "no refresh within 2 s");
        std::thread::sleep(std::time::Duration::from_millis(1));
    };
    assert_eq!(refresh.kind, SideKind::Refresh);
    assert_eq!(refresh.t, now.sim_time);
    assert_eq!(refresh.positions, now.scene.nuclei.iter().map(|n| n.pos).collect::<Vec<_>>());
    let sk = surrogate_of(&sh_d);
    assert_eq!(sk.samples(), n0, "a refresh is not a sample");
    assert_eq!(sk.last_audit().cloned(), gap0, "nor an audit");
    drop(sk);
    // the queued isolated atom: on the scene's cell, through its own queue
    let t0 = std::time::Instant::now();
    let atoms = loop {
        let a = surrogate_of(&sh_d).take_atom_densities();
        if !a.is_empty() {
            break a;
        }
        assert!(t0.elapsed().as_secs_f64() < 2.0, "no atom within 2 s");
        std::thread::sleep(std::time::Duration::from_millis(1));
    };
    assert!(matches!(atoms.as_slice(), [(1, Display::Cpu { cell, rho, .. })] if *cell == now.scene.cell && rho[0] == 1.0), "{atoms:?}");
    // an edit clears it
    d.apply(DynCommand::RemoveNucleus(1));
    assert!(d.snapshot().side_display.is_none());
}

/// Ruling S14 + S7: a failed background audit untrusts the model when its verdict arrives;
/// the next evaluation (a teacher step, rule 4) books the Model → Teacher flip at its own
/// positions as `refit_jump`, so E_conserved stays continuous, and the note says how many
/// answered steps went unaudited.
#[test]
fn a_failed_background_audit_untrusts_and_books_the_flip() {
    let (s, h, open, _) = with_gated_auditor(5, 0.0);
    // at r_e, where a 10 % deeper well changes the energy by 1.7e-2 Ha but the forces hardly:
    // the one mixed-force half-kick of the flip step (see the synchronous test) stays small
    let mut scene = h2(1.4);
    scene.nuclei[1].vel = [2e-4, 0.0, 0.0];
    let (p, sh_d) = driven(s);
    let mut d = Dynamics::new(scene, p, 0.25, 1);
    h.lock().unwrap().d *= 1.1; // the teacher drifts away from the model
    open.store(true, std::sync::atomic::Ordering::SeqCst);
    let mut e_c = vec![d.snapshot().energy_conserved];
    let mut flip = None;
    let t0 = Instant::now();
    while flip.is_none() {
        assert!(t0.elapsed().as_secs_f64() < 2.0, "no flip within 2 s");
        d.apply(DynCommand::Step(1));
        let snap = d.tick(f64::INFINITY, &mut never());
        e_c.push(snap.energy_conserved);
        if snap.source == Source::Fallback {
            flip = Some(snap);
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let snap = flip.unwrap();
    let jump = snap.eval.refit_jump.expect("the flip is booked");
    let delta = e_c[e_c.len() - 1] - e_c[e_c.len() - 2];
    println!("failed background audit: jump {jump:+.3e} Ha, ΔE_c across the flip {delta:+.2e} Ha, notes {:?}", snap.eval.notes);
    assert!(snap.eval.notes.iter().any(|n| n.contains("surrogate audit failed (background teacher)") && n.contains("unaudited")), "{:?}", snap.eval.notes);
    assert_eq!(snap.eval.energy_surface(), Surface::Teacher);
    assert!(jump.abs() / 2.0 > 1e-3, "test premise: the drift exceeds the per-atom bound");
    // the mixed-force half-kick remainder only, as for the synchronous flip
    assert!(delta.abs() < 1e-5, "E_conserved saw the flip: {delta:+.2e}");
    assert!(snap.notes.iter().all(|n| !n.contains("history restarts")), "{:?}", snap.notes);
    let mut sk = surrogate_of(&sh_d);
    assert!(!sk.trusted() && sk.audit_failed().is_some());
    assert_eq!(sk.last_audit().map(|a| a.verdict), Some(AuditVerdict::Failed));
    // a refit clears it
    sk.refit().unwrap();
    assert!(sk.audit_failed().is_none());
}

/// Task 9 item 1: an audit whose teacher answer is box-affected is recorded but not judged
/// — its gap is the box's confinement, which a translation-invariant model cannot learn —
/// and its answer is not kept: the model stays trusted and answers on, nothing is
/// quarantined. (Before: the audit failed on the confinement, the refit quarantined the
/// confined sample, and the quarantine fenced the equilibrium bond everywhere in the cell —
/// the user's "trusted, 100 % fallback, stable gap".) Both audit paths.
#[test]
fn a_box_affected_audit_is_not_judged_and_not_kept() {
    for background in [false, true] {
        let rho = 2.0 * BOX_AFFECTED_FACE_DENSITY;
        let (mut d, h, sh_d) = if background {
            let (s, h, open, _) = with_gated_auditor(5, rho);
            open.store(true, std::sync::atomic::Ordering::SeqCst);
            let (p, sh_d) = driven(s);
            (Dynamics::new(h2(1.5), p, 0.25, 1), h, sh_d)
        } else {
            let (s, h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 5, ..FallbackPolicy::default() });
            // the synchronous path: the main teacher reports the face density
            let teacher = FaceDensity { inner: h.clone(), rho: Arc::new(Mutex::new(rho)) };
            let mut s2 = SurrogateKs::new(Box::new(teacher), FallbackPolicy { audit_every: 5, ..FallbackPolicy::default() }).with_teacher_tol(1e-6);
            s2.set_reference_energy(1, MorseDimer::h2().e_atom);
            for smp in s.store().unwrap().samples.clone() {
                let sc = Scene { nuclei: smp.pos.iter().zip(&smp.z).map(|(p, &z)| Nucleus::new(z, *p)).collect(), ..h2(1.5) };
                s2.add_sample(smp, &sc).unwrap();
            }
            s2.refit().unwrap();
            let (p, sh_d) = driven(s2);
            (Dynamics::new(h2(1.5), p, 0.25, 1), h, sh_d)
        };
        h.lock().unwrap().d *= 1.05; // the "box" moves the teacher 5 % off the model
        let n0 = surrogate_of(&sh_d).samples();
        let seen = step_until(&mut d, "the audit verdict", |_| surrogate_of(&sh_d).last_audit().is_some());
        for _ in 0..10 {
            d.apply(DynCommand::Step(1));
            d.tick(f64::INFINITY, &mut never());
        }
        let mut sk = surrogate_of(&sh_d);
        let a = sk.last_audit().unwrap().clone();
        println!("background {background}: {a:?}");
        assert_eq!(a.verdict, AuditVerdict::BoxAffected);
        assert!(a.gap.abs() / 2.0 > 1e-3, "test premise: the gap alone would fail the audit");
        assert!(sk.trusted() && sk.audit_failed().is_none(), "{:?}", sk.untrusted_reason());
        assert_eq!(sk.samples(), n0, "a box-affected answer is not trained on");
        sk.refit().ok();
        assert!(sk.trusted() && sk.quarantined() == 0);
        let notes: Vec<String> = seen.iter().flat_map(|s| s.eval.notes.clone()).collect();
        assert!(notes.iter().any(|n| n.contains("surrogate audit not judged")), "{notes:?}");
        drop(sk);
        d.apply(DynCommand::Step(1));
        assert_eq!(d.tick(f64::INFINITY, &mut never()).source, Source::Surrogate);
    }
}

/// Ruling (Task 9 item 1): a trusted model that falls back on every consultation says so —
/// after `STUCK_STEPS` consecutive fallbacks `stuck()` names the rule and the gap — and the
/// next answer clears it.
#[test]
fn a_trusted_model_that_keeps_falling_back_says_so() {
    let (mut s, _h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    // R = 2.8 is outside the envelope: every consultation there falls back
    for i in 0..STUCK_STEPS {
        assert!(s.stuck().is_none(), "after {i} fallbacks");
        let r = s.evaluate(&h2(2.8), &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback);
    }
    let why = s.stuck().expect("stuck after STUCK_STEPS fallbacks");
    println!("{why}");
    assert!(s.trusted());
    assert!(why.contains("descriptor envelope") && why.contains("teacher − model"), "{why}");
    assert_eq!(s.evaluate(&h2(1.6), &mut never()).unwrap().stats.source, Source::Surrogate);
    assert!(s.stuck().is_none());
}

/// Task 9 item 5, the cadence invariant: with `variance_cadence` = k the variance is evaluated
/// on the first consultation after an edit (`discontinuity` / `reset`), a refit or a policy
/// change, never more than k consultations apart otherwise, and afresh after every rejection
/// (a rejecting verdict is never reused).
#[test]
fn the_variance_is_checked_at_least_every_k_consultations_and_after_every_edit() {
    let k = 4;
    let (mut s, _h) = trained(1.2, 2.2, 40, FallbackPolicy { variance_cadence: k, audit_every: 0, ..FallbackPolicy::default() });
    let scene = h2(1.6);
    let checked = |s: &mut SurrogateKs| {
        let before = s.variance_checks();
        let r = s.evaluate(&scene, &mut never()).unwrap();
        (s.variance_checks() > before, r.stats.source)
    };
    // the first consultation after the refit checks; then at most k apart
    let pattern: Vec<bool> = (0..13).map(|_| checked(&mut s).0).collect();
    println!("checks over 13 consultations at k = {k}: {pattern:?}");
    assert!(pattern[0], "the first consultation after a refit checks");
    let mut gap = 0;
    for &c in &pattern {
        gap = if c { 0 } else { gap + 1 };
        assert!(gap < k, "{k} or more consultations without a check: {pattern:?}");
    }
    assert_eq!(pattern.iter().filter(|&&c| c).count(), 4, "k = {k}: checks at 0, 4, 8, 12 (no more than needed)");
    // an edit, a refit, a policy change: the next consultation checks
    checked(&mut s);
    s.discontinuity();
    assert!(checked(&mut s).0, "after discontinuity()");
    checked(&mut s);
    s.reset();
    assert!(checked(&mut s).0, "after reset()");
    checked(&mut s);
    s.refit().unwrap();
    assert!(checked(&mut s).0, "after a refit");
    checked(&mut s);
    s.set_policy(*s.policy());
    assert!(checked(&mut s).0, "after a policy change");
    // a rejecting verdict is never reused: every consultation checks. A force bound (1e-5
    // Ha/Bohr) above the model's worst training residual — trust holds — but below its
    // uncertainty in the stretched tail (envelope off, so the variance rule is what refuses)
    let residual = s.training_residuals().unwrap().force;
    s.set_policy(FallbackPolicy { max_descriptor_distance: 1e9, force_std_abs: 1e-5, force_std_rel: 0.0, variance_cadence: k, audit_every: 0, ..FallbackPolicy::default() });
    assert!(s.trusted() && residual < 1e-5, "{:?} ({residual:.1e})", s.untrusted_reason());
    let r_far = [5.5, 5.0, 4.5]
        .into_iter()
        .find(|&r| {
            s.discontinuity();
            s.evaluate(&h2(r), &mut never()).unwrap().stats.notes.iter().any(|n| n.contains("force std"))
        })
        .expect("a geometry the variance rule refuses");
    println!("the variance rule refuses R = {r_far} at a 1e-5 Ha/Bohr bound (worst training residual {residual:.1e})");
    s.discontinuity();
    let stretched = h2(r_far);
    for i in 0..6 {
        let before = s.variance_checks();
        let src = s.evaluate(&stretched, &mut never()).unwrap().stats.source;
        let c = s.variance_checks() > before;

        assert!(c && src == Source::Fallback, "consultation {i} after a rejection: checked {c}, {src:?}");
    }
}

/// Ruling S17 (direction corrected in fix round 2): the reported face density is the
/// outermost grid plane, one h inside the wall, where the confined orbital is pinched ∝ h²,
/// so the confinement per unit of it grows as 1/h² and the tolerated density scales as h² —
/// a quarter at h = 0.125, four times at h = 0.5.
#[test]
fn the_box_rule_scales_with_the_grid_spacing() {
    assert_eq!(box_affected_face_density(0.25), BOX_AFFECTED_FACE_DENSITY);
    assert!((box_affected_face_density(0.125) - 0.25 * BOX_AFFECTED_FACE_DENSITY).abs() < 1e-18);
    assert!((box_affected_face_density(0.5) - 4.0 * BOX_AFFECTED_FACE_DENSITY).abs() < 1e-18);
}

/// A Morse teacher whose answers carry a confinement offset of the test's choosing (energy
/// +`bias`) and report a face density `rho` — a box-affected teacher.
struct Confined {
    inner: MorseDimer,
    rho: Arc<Mutex<(f64, f64)>>,
}

impl ForceProvider for Confined {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        let mut r = self.inner.evaluate(scene, should_abort)?;
        let (rho, bias) = *self.rho.lock().unwrap();
        r.energy += bias;
        r.stats.face_density = Some(rho);
        Ok(r)
    }
    fn name(&self) -> &'static str {
        "morse dimer"
    }
    fn reset(&mut self) {}
}

/// Task 9 fix round 1, S1: box-affected answers taken while the model was still being built
/// (no model yet: the "essential" exemption) carry their face density; they are never
/// quarantine candidates — so their descriptor-space fence cannot recreate the user's
/// "trusted, 100 % fallback" — and they leave the fit once the trust floor (Q′ + 30) is met
/// without them. The flags and densities survive the sidecar.
#[test]
fn box_affected_samples_never_fence_and_leave_the_fit_at_the_trust_floor() {
    let state = Arc::new(Mutex::new((2.0 * BOX_AFFECTED_FACE_DENSITY, 0.05)));
    let mut s = SurrogateKs::new(Box::new(Confined { inner: MorseDimer::h2(), rho: state.clone() }), FallbackPolicy::default()).with_teacher_tol(1e-6);
    s.set_reference_energy(1, MorseDimer::h2().e_atom);
    // no model yet: three confined answers (+0.05 Ha) are kept, as a pass on this box would
    for r in [1.5, 1.6, 1.7] {
        s.evaluate(&h2(r), &mut never()).unwrap();
    }
    assert_eq!(s.samples(), 3);
    let mut rng = StdRng::seed_from_u64(4);
    let mut add = |s: &mut SurrogateKs, n: usize| {
        for i in 0..n {
            let r = 1.2 + 1.0 * i as f64 / (n - 1) as f64;
            let sc = dimer(r, unit_vector(&mut rng), [0.0; 3], 1, 1);
            s.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
        }
    };
    // 40 clean samples: below the floor (29 + 30), the confined ones stay in the fit; the
    // residuals fail but none of them may be quarantined
    add(&mut s, 40);
    s.refit().unwrap();
    println!("below the floor: trusted {}, {:?}, quarantined {}, box-excluded {}", s.trusted(), s.untrusted_reason(), s.quarantined(), s.box_excluded());
    assert!(!s.trusted() && s.quarantined() == 0 && s.box_excluded() == 0, "{:?}", s.untrusted_reason());
    // 30 more: the floor is met without them — they leave the fit, never quarantined
    add(&mut s, 30);
    let rep = s.refit().unwrap();
    println!("at the floor: trusted {}, fitted {}, quarantined {}, box-excluded {}", s.trusted(), rep.n_samples, s.quarantined(), s.box_excluded());
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    assert_eq!((s.quarantined(), s.box_excluded(), rep.n_samples), (0, 3, 70));
    // the model answers at the confined samples' geometries: no fence there
    assert_eq!(s.evaluate(&h2(1.6), &mut never()).unwrap().stats.source, Source::Surrogate);
    // the sidecar keeps the code and the densities
    let path = std::env::temp_dir().join(format!("box-flags-{}.model", std::process::id()));
    s.save_sidecar(&path).unwrap();
    let mut back = SurrogateKs::new(Box::new(MorseDimer::h2()), FallbackPolicy::default()).with_teacher_tol(1e-6);
    back.install_sidecar(surrogate::sidecar::read(&path).unwrap()).unwrap();
    assert_eq!((back.box_excluded(), back.quarantined(), back.trusted()), (3, 0, true));
    let _ = std::fs::remove_file(&path);
}

/// A Morse teacher with a Gaussian bump of `amp` Ha at r0 (width w): the first fit, trained on
/// the plain Morse, misses it.
struct Bumped {
    inner: MorseDimer,
    amp: f64,
    r0: f64,
    w: f64,
}

impl ForceProvider for Bumped {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        let mut r = self.inner.evaluate(scene, should_abort)?;
        let (p0, p1) = (scene.nuclei[0].pos, scene.nuclei[1].pos);
        let d: [f64; 3] = std::array::from_fn(|a| p1[a] - p0[a]);
        let rr = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let g = self.amp * (-((rr - self.r0) / self.w).powi(2)).exp();
        let de = -2.0 * (rr - self.r0) / (self.w * self.w) * g;
        r.energy += g;
        for a in 0..3 {
            // the Morse convention: F₀ = +dE/dr û, F₁ = −dE/dr û (û from 0 to 1)
            r.forces[0][a] += de * d[a] / rr;
            r.forces[1][a] -= de * d[a] / rr;
        }
        r.stats.energy.total = r.energy;
        Ok(r)
    }
    fn name(&self) -> &'static str {
        "morse dimer"
    }
    fn reset(&mut self) {}
}

/// Task 9 fix round 1, I3 (fence voting): a quarantined *glitch* (a bad label, not
/// box-affected) inside a region where the model is genuinely wrong by more than the tier
/// used to leave no learning path — every fresh answer disagreed with the model, none was
/// kept, the fence never released: DFT speed under a trusted badge. The teacher here carries
/// a Gaussian bump (2e-2 Ha at r = 2.1, width 0.3) the first fit saw only the foot of (its
/// samples stop at r = 1.6), so at r ≈ 1.9 the model is off by ≈ 4e-3 Ha (2e-3 Ha/atom).
/// Now the third mutually consistent fresh answer that disagrees with the model is kept, the
/// refit learns the region from it, the next fresh answer agrees and supersedes the glitch,
/// and the model answers.
#[test]
fn consistent_answers_inside_a_fence_teach_the_model_the_region() {
    let teacher = || Bumped { inner: MorseDimer::h2(), amp: 2e-2, r0: 2.1, w: 0.3 };
    let mut s = SurrogateKs::new(Box::new(teacher()), FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() }).with_teacher_tol(1e-6);
    s.set_reference_energy(1, MorseDimer::h2().e_atom);
    let mut rng = StdRng::seed_from_u64(4);
    let mut t = teacher();
    for i in 0..60 {
        let r = 1.0 + 0.6 * i as f64 / 59.0;
        let sc = dimer(r, unit_vector(&mut rng), [0.0; 3], 1, 1);
        let res = t.evaluate(&sc, &mut never()).unwrap();
        let smp = Sample { pos: sc.nuclei.iter().map(|n| n.pos).collect(), z: vec![1, 1], energy: res.energy, forces: res.forces, tag: SampleTag::Scan, weight: 1.0 };
        s.add_sample(smp, &sc).unwrap();
    }
    // the glitch: 0.1 Ha off at r = 1.9 — quarantined, fencing r ≈ 1.9
    let stray = h2(1.9);
    let mut bad = morse_sample(&MorseDimer::h2(), &stray);
    bad.energy += 0.1;
    s.add_sample(bad, &stray).unwrap();
    s.refit().unwrap();
    println!("{} samples; trusted {} {:?}; quarantined {}", s.samples(), s.trusted(), s.untrusted_reason(), s.quarantined());
    assert!(s.trusted() && s.is_quarantined(60) && s.quarantined() == 1, "{:?} {}", s.untrusted_reason(), s.quarantined());
    let model_gap = |s: &SurrogateKs, r: f64| {
        let pos: Vec<[f64; 3]> = h2(r).nuclei.iter().map(|n| n.pos).collect();
        teacher().evaluate(&h2(r), &mut never()).unwrap().energy - s.model().unwrap().predict(&pos, &[1, 1], false).unwrap().energy
    };
    let before = model_gap(&s, 1.9);
    println!("teacher − model at r = 1.9 before: {before:+.2e} Ha");
    assert!(before.abs() / 2.0 > 1e-3, "test premise: the model misses the region by more than the tier");
    // a molecule moving slowly through the fenced region; the worker refits when a sample
    // is pending
    let (mut answered, mut refits, mut voted) = (None, 0, None);
    for i in 0..30 {
        let r = 1.9 + 0.002 * (i % 5) as f64;
        let res = s.evaluate(&h2(r), &mut never()).unwrap();
        if res.stats.source == Source::Surrogate {
            answered = Some(i);
            break;
        }
        if voted.is_none() && res.stats.notes.iter().any(|n| n.contains("mutually consistent fresh teacher answers")) {
            voted = Some(i);
        }
        if s.pending_refit() {
            s.refit().unwrap();
            refits += 1;
        }
    }
    let after = model_gap(&s, 1.9);
    println!("vote at evaluation {voted:?}, model answered at {answered:?} after {refits} refits; teacher − model now {after:+.2e} Ha; quarantined {}, superseded {}, trusted {}",
             s.quarantined(), s.superseded(), s.trusted());
    assert_eq!(voted, Some(2), "the third disagreeing answer is the vote");
    let i = answered.expect("the model never learned the fenced region");
    assert!(i <= 5 && refits <= 2, "{i} evaluations, {refits} refits");
    assert!(s.trusted() && s.quarantined() == 0 && s.superseded() == 1, "the glitch is superseded, not released");
    assert!(after.abs() / 2.0 <= 1e-3, "the region is learned: {after:+.2e}");
}

/// The driver's safety net: a change of energy surface between consecutive evaluations that
/// carries no `refit_jump` restarts the conserved-energy history with a note.
#[test]
fn a_surface_change_without_a_jump_restarts_the_history() {
    /// A Morse teacher that starts labelling its answers `Source::Surrogate` after `flip_at`
    /// evaluations (the energy stays the teacher's: a bare class flip, no jump reported).
    struct Flipper {
        inner: MorseDimer,
        flip_at: usize,
    }
    impl ForceProvider for Flipper {
        fn evaluate(&mut self, scene: &Scene, a: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
            let mut r = self.inner.evaluate(scene, a)?;
            if self.inner.calls > self.flip_at {
                r.stats.source = Source::Surrogate;
            }
            Ok(r)
        }
        fn name(&self) -> &'static str {
            "flipper"
        }
        fn reset(&mut self) {}
    }
    let mut scene = h2(1.5);
    scene.nuclei[1].vel = [2e-4, 0.0, 0.0];
    // evaluate 1 = Dynamics::new; evaluates 2–4 = steps 1–3 (Teacher); evaluate 5 = step 4 (Model)
    let mut d = Dynamics::new(scene, Box::new(Flipper { inner: MorseDimer::h2(), flip_at: 4 }), 0.25, 1);
    for _ in 0..3 {
        d.apply(DynCommand::Step(1));
        let snap = d.tick(f64::INFINITY, &mut never());
        assert_eq!(snap.eval.energy_surface(), Surface::Teacher);
    }
    assert_eq!(d.snapshot().history.len(), 3);
    d.apply(DynCommand::Step(1));
    let snap = d.tick(f64::INFINITY, &mut never());
    assert_eq!(snap.source, Source::Surrogate);
    assert_eq!(snap.eval.energy_surface(), Surface::Model);
    assert_eq!(snap.history.len(), 1, "the history restarted at this step");
    assert!(snap.notes.last().is_some_and(|n| n.contains("energy surface changed (Teacher → Model) without a booked jump")), "{:?}", snap.notes);
    assert_eq!(snap.work_mismatch, 0.0);
}

/// GPU H₂ round trip: the teacher's own samples (40 centred scan geometries at random
/// orientations, r ∈ [1.1, 2.0]) train a model that reproduces the teacher at R_e (body-2
/// only; Task 7 pins the tiered targets). The brief's bounds, 1e-4 Ha / 2e-3 Ha/Bohr, are
/// pinned at 41³ (10 Bohr box), where the teacher's own orientation scatter is at the
/// egg-box level (s_E 1.3e-5/atom, s_F 1.4e-4). At 33³ (8 Bohr) the dimer's far atom is
/// 2.8 Bohr from a face and the energy depends on the orientation at 7e-5/atom (the H atom
/// itself is 8 mHa higher than at 41³: confinement, the provider's own "face density"
/// warning) — a bound below the teacher's reproducibility would measure the box, so the 33³
/// run asserts the force bound and a calibration bound (|ΔE| within 3× the fit's total
/// energy scatter) and records the numbers.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn gpu_teacher_h2_round_trip() {
    let Some(ctx) = gpu::GpuContext::new_headless() else {
        eprintln!("sim: no GPU adapter — skipped");
        return;
    };
    let (de, df, s_e) = gpu_round_trip(ctx.clone(), 41);
    assert!(de < 1e-4, "41³: |ΔE| = {de:.2e}");
    assert!(df < 2e-3, "41³: max |ΔF| = {df:.2e}");
    assert!(s_e < 3e-5, "41³: the teacher's energy scatter is at the egg-box level, s_E = {s_e:.1e}");
    let (de, df, s_e) = gpu_round_trip(ctx, 33);
    assert!(df < 2e-3, "33³: max |ΔF| = {df:.2e}");
    assert!(de < 3.0 * 2.0 * s_e, "33³: |ΔE| = {de:.2e} against the fit's total energy scatter 2·s_E = {:.2e}", 2.0 * s_e);
}

/// Diagnostic (Task 9): the box's confinement energy against the face density — H₂ at R
/// translated toward a face of the 41³ box (and stretched at the centre), E(x) − E(0) next
/// to the teacher's face density. Sets `BOX_AFFECTED_FACE_DENSITY`.
#[test]
#[ignore = "diagnostic: cargo test -p sim --release diag_confinement -- --ignored --nocapture"]
fn diag_confinement_vs_face_density() {
    let Some(ctx) = gpu::GpuContext::new_headless() else { return };
    for n_grid in [41usize, 49] {
        let cell = Cell { n: n_grid, h: 0.25 };
        let mut t = crate::ks::GpuKs::new(ctx.clone(), 1e-6);
        t.set_display(false);
        let half = 0.5 * (n_grid - 1) as f64 * 0.25;
        for r in [1.45, 2.5, 3.5, 4.5] {
            let mut e0 = None;
            for x in [0.0, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0] {
                let far = x + 0.5 * r;
                if far > half - 0.8 {
                    continue;
                }
                let mut sc = dimer(r, [1.0, 0.0, 0.0], [x, 0.0, 0.0], 1, 1);
                sc.cell = cell.clone();
                let res = t.evaluate(&sc, &mut never()).unwrap();
                let e = res.energy;
                let base = *e0.get_or_insert(e);
                println!("{n_grid}³ R {r:.2} centre x {x:.1} (far atom {:.2} Bohr from the face): E {e:+.6}  E − E(centre) {:+.2e}  face density {:.2e}",
                         half - far, e - base, res.stats.face_density.unwrap_or(f64::NAN));
            }
        }
    }
}

/// One round trip on an n³ grid: (|ΔE| at R_e, max |ΔF| at R_e, the fit's s_E per atom).
fn gpu_round_trip(ctx: gpu::GpuContext, n_grid: usize) -> (f64, f64, f64) {
    let cell = Cell { n: n_grid, h: 0.25 };
    let with_cell = |mut sc: Scene| {
        sc.cell = cell.clone();
        sc
    };
    let t0 = Instant::now();
    let mut teacher = crate::ks::GpuKs::new(ctx.clone(), 1e-6);
    teacher.set_display(false);
    let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_teacher_tol(1e-6);
    // the H reference from the teacher itself (an isolated atom at the same settings)
    let atom = Scene { nuclei: vec![Nucleus::new(1, [0.0; 3])], cell: cell.clone(), charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3] };
    let e_h = s.evaluate(&atom, &mut never()).unwrap().energy;
    s.set_reference_energy(1, e_h);
    // 40 scan geometries on r ∈ [1.1, 2.0]: every fallback keeps its sample
    let mut rng = StdRng::seed_from_u64(7);
    let n = 40;
    for i in 0..n {
        let r = 1.1 + 0.9 * i as f64 / (n - 1) as f64;
        let origin = [rng.gen_range(-0.2..0.2), rng.gen_range(-0.2..0.2), rng.gen_range(-0.2..0.2)];
        let sc = with_cell(dimer(r, unit_vector(&mut rng), origin, 1, 1));
        // unrelated geometries: keep the warm state, drop the density extrapolation (R13)
        s.discontinuity();
        let r = s.evaluate(&sc, &mut never()).expect("GPU teacher");
        assert_eq!(r.stats.source, Source::Fallback);
    }
    assert_eq!(s.samples(), n + 1);
    let t_scan = t0.elapsed().as_secs_f64();
    let rep = s.refit().expect("refit");
    println!("GPU H₂ {n_grid}³: E_H = {e_h:+.6} Ha; {} samples in {t_scan:.1} s; fit rmse E/atom {:.2e} Ha, F {:.2e} Ha/Bohr; s_E {:.1e} s_F {:.1e} λ {:.3e}; dof_E {:.1} dof_F {:.1}; trusted {}",
             rep.n_samples, rep.rmse_energy_per_atom, rep.rmse_force, rep.sigma_e, rep.sigma_f, rep.lambda, rep.residual_dof_e, rep.residual_dof_f, s.trusted());
    assert!(s.trusted(), "{rep:?}");
    let re = with_cell(h2(1.45));
    s.discontinuity();
    let t1 = Instant::now();
    let r = s.evaluate(&re, &mut never()).unwrap();
    let t_sur = t1.elapsed().as_secs_f64() * 1e3;
    println!("surrogate at R = 1.45: {:?} E {:+.6} Ha, F_x(0) {:+.4e}, uncertainty {:?}, {t_sur:.2} ms, notes {:?}", r.stats.source, r.energy, r.forces[0][0], r.stats.uncertainty, r.stats.notes);
    assert_eq!(r.stats.source, Source::Surrogate, "{:?}", r.stats.notes);
    let mut direct = crate::ks::GpuKs::new(ctx, 1e-6);
    direct.set_display(false);
    let d = direct.evaluate(&re, &mut never()).unwrap();
    let de = (r.energy - d.energy).abs();
    let df = max_force_diff(&r.forces, &d.forces);
    println!("direct GPU: E {:+.6} Ha, F_x(0) {:+.4e}; |ΔE| {de:.2e} Ha, max|ΔF| {df:.2e} Ha/Bohr", d.energy, d.forces[0][0]);
    (de, df, rep.sigma_e)
}

/// Spec §6.2: a relax on the surrogate runs on the model; at the end one teacher solve at the
/// result is kept as a sample and reported as `verified_max_force`.
#[test]
fn a_relax_on_a_trainable_is_verified_by_its_teacher() {
    // (moved from `sim::dynamics::tests` with the providers, M6 Task 1)
    const DT_FS: f64 = 0.25;
    fn dimer(r: f64, dir: [f64; 3], origin: [f64; 3]) -> Scene {
        let p0: [f64; 3] = std::array::from_fn(|a| origin[a] - 0.5 * r * dir[a]);
        let p1: [f64; 3] = std::array::from_fn(|a| origin[a] + 0.5 * r * dir[a]);
        Scene { nuclei: vec![Nucleus::new(1, p0), Nucleus::new(1, p1)], cell: cell(), charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3] }
    }
    fn bond(s: &Scene) -> f64 {
        let (a, b) = (s.nuclei[0].pos, s.nuclei[1].pos);
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }
    let teacher = Shared::new(MorseDimer::h2());
    let handle = teacher.handle();
    let calls = || handle.lock().unwrap().calls;
    // audits off: every teacher call is then a fallback or the verification
    let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() }).with_teacher_tol(1e-6);
    s.set_reference_energy(1, MorseDimer::h2().e_atom);
    let mut rng = StdRng::seed_from_u64(4);
    let exact = MorseDimer::h2();
    for i in 0..40 {
        let r = 1.2 + (2.2 - 1.2) * i as f64 / 39.0;
        let dir = loop {
            let v: [f64; 3] = [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)];
            let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if (1e-3..1.0).contains(&n) {
                break v.map(|x| x / n);
            }
        };
        let origin = [rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3)];
        let sc = dimer(r, dir, origin);
        let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
        let (energy, forces) = exact.energy_and_forces(&pos).unwrap();
        s.add_sample(Sample { pos, z: vec![1, 1], energy, forces, tag: SampleTag::Scan, weight: 1.0 }, &sc).unwrap();
    }
    let rep = s.refit().expect("refit");
    println!("model: 40 Morse samples r ∈ [1.2, 2.2], rmse E/atom {:.1e}, F {:.1e}; trusted {}", rep.rmse_energy_per_atom, rep.rmse_force, s.trusted());
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    let samples0 = s.samples();

    let (p, sh_d) = driven(s);
    let mut d = Dynamics::new(dimer(1.7, [1.0, 0.0, 0.0], [0.0; 3]), p, DT_FS, 1);
    let first = d.snapshot();
    assert_eq!(first.source, Source::Surrogate, "the model answers inside its training range: {:?}", first.eval.notes);
    let calls0 = calls();
    let e_c0 = first.energy_conserved;
    d.apply(DynCommand::Relax);
    let (last, r) = loop {
        let snap = d.tick(-1.0, &mut never());
        let r = snap.relax.clone().expect("relaxing");
        assert!((snap.energy_conserved - e_c0).abs() <= 1e-10, "step {}: E_conserved moved by {:.2e}", r.steps, snap.energy_conserved - e_c0);
        if r.converged.is_some() {
            break (snap, r);
        }
        assert_eq!(snap.source, Source::Surrogate, "step {}: the model fell back ({:?})", r.steps, snap.eval.notes);
        assert_eq!(calls(), calls0, "no teacher call before the verification");
        assert!(r.steps < 2000);
    };
    assert_eq!(r.converged, Some(true));
    // exactly one teacher solve more than the fallbacks (none): the verification
    assert_eq!(calls(), calls0 + 1);
    let pos: Vec<[f64; 3]> = last.scene.nuclei.iter().map(|n| n.pos).collect();
    let (e_t, f_t) = exact.energy_and_forces(&pos).unwrap();
    let teacher_max = f_t.iter().map(|f| (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt()).fold(0.0, f64::max);
    println!(
        "relax on the model: {} steps to model max|F| {:.2e}; R = {:.5} (Morse r_e 1.4); verified (teacher) max|F| {:.2e}; teacher − model energy {:+.2e} Ha",
        r.steps, r.max_force, bond(&last.scene), teacher_max, e_t - last.energy_ks
    );
    assert_eq!(r.verified_max_force, Some(teacher_max));
    assert_eq!(last.forces, f_t, "the driver's forces are the teacher's answer");
    assert_eq!(last.source, Source::Fallback);
    assert!(last.teacher_gap.is_some(), "a model-surface fallback: the energy stays the model's");
    assert!((last.energy_conserved - e_c0).abs() <= 1e-10);
    // the relaxed geometry is the teacher's minimum to within the stop criterion's slack
    // plus the model's force error: |ΔR| ≤ (F_TOL + |F_teacher − F_model|) / k, k = 0.34
    assert!((bond(&last.scene) - 1.4).abs() <= (1e-3 + teacher_max) / 0.34);
    assert_eq!(r.note, if teacher_max >= 1e-3 { Some(format!("relaxed on the model; DFT max|F| = {teacher_max:.1e} Ha/Bohr (≥ 1e-3): Relax again to continue on the refitted model")) } else { None });
    // the answer was kept as a sample, pending the worker's refit
    let s = surrogate_of(&sh_d);
    assert_eq!(s.samples(), samples0 + 1);
    assert!(s.pending_refit());
}

/// M6 Task 1: `Trainable::status` carries the provider's own readouts — what the worker's
/// `ModelStatus` took from `SurrogateKs` directly before the downcast was removed — and the
/// policy setters reach the policy.
#[test]
fn trainable_status_matches_the_provider_fields() {
    let (mut s, _h) = trained(1.2, 2.2, 40, FallbackPolicy::default());
    let scene = h2(1.5);
    let forces = [[0.3, 0.0, 0.0], [-0.02, 0.0, 0.0]];
    let st = Trainable::status(&s, &scene, &forces);
    println!("{st:?}");
    assert_eq!(st.samples, 40);
    assert_eq!(st.samples, s.samples());
    assert!(st.fitted);
    assert_eq!(st.epoch, s.epoch());
    assert_eq!(st.trusted, s.trusted());
    assert!(st.trusted && st.untrusted_reason.is_none());
    assert_eq!(st.teacher, "morse dimer");
    assert!(!st.pending_refit);
    let r = s.report().unwrap();
    assert_eq!(st.fit, Some((r.n_samples, r.q_prime, r.rmse_energy_per_atom, r.rmse_force)));
    let t = s.training_residuals().unwrap();
    assert_eq!(st.residuals, Some((t.energy_per_atom, t.force, t.force_bound)));
    assert_eq!(st.force_bound, s.policy().force_bound(0.02), "the bound at the smallest force");
    assert_eq!(Trainable::status(&s, &scene, &[]).force_bound, s.policy().force_bound(0.0));
    assert_eq!(st.provenance_mismatch, None);
    let other = Scene { cell: Cell { n: 41, h: 0.25 }, ..scene.clone() };
    let m = Trainable::status(&s, &other, &forces).provenance_mismatch.expect("another grid");
    assert!(m.contains("33³") && m.contains("41³"), "{m}");
    assert_eq!((st.audit_every, st.draft), (2000, false));
    assert_eq!((st.quarantined, st.recovery.clone(), st.stuck.clone()), (0, None, None));
    assert_eq!((st.audits_background, st.audit_in_flight, st.last_audit.clone()), (false, false, None));
    // the setters reach the policy (and a value already set changes nothing)
    let epoch = s.epoch();
    Trainable::set_audit_every(&mut s, 7);
    Trainable::set_draft(&mut s, true);
    assert_eq!((s.policy().audit_every, s.policy().never_fallback), (7, true));
    let st = Trainable::status(&s, &scene, &forces);
    assert_eq!((st.audit_every, st.draft, st.epoch), (7, true, epoch));
    // a refit through the trait: the summary line, the same report
    let line = Trainable::refit(&mut s).unwrap();
    let r = s.report().unwrap();
    assert_eq!(line, format!("{} samples, rmse_E {:.1e} Ha/atom, rmse_F {:.1e} Ha/Bohr", r.n_samples, r.rmse_energy_per_atom, r.rmse_force));
    // a forget through the trait: noted, nothing left
    Trainable::forget(&mut s);
    assert_eq!(Trainable::take_notes(&mut s), vec!["model forgotten (40 samples dropped): the teacher answers every step".to_string()]);
    let st = Trainable::status(&s, &scene, &forces);
    assert!(!st.fitted && st.samples == 0 && st.fit.is_none());
}
