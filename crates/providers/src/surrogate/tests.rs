use super::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sim::{Cell, DynCommand, DynSnapshot, Dynamics, MorseDimer, Nucleus, Shared, SharedTrainable, Surface};
use std::sync::{Arc, Mutex, MutexGuard};

fn never() -> impl FnMut() -> bool {
    || false
}

/// Only a busy device, on a pass that can go on, and only a few times in a row, is kept for a retry.
#[test]
fn a_busy_device_does_not_end_a_pass_until_it_persists() {
    let busy = || ProviderError::Solver(sim::SolverError::Device("callback dropped".into()));
    assert!(pass_step_is_transient(&busy(), false, 0));
    assert!(pass_step_is_transient(&busy(), false, PASS_BUSY_RETRIES - 1));
    assert!(!pass_step_is_transient(&busy(), false, PASS_BUSY_RETRIES), "persistent: the pass ends");
    assert!(!pass_step_is_transient(&busy(), true, 0), "a finished pass has nothing to retry");
    for other in [ProviderError::Invalid("x".into()), ProviderError::Solver(sim::SolverError::Numerical("x".into()))] {
        assert!(!pass_step_is_transient(&other, false, 0), "{other:?}");
    }
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
            ionic: sim::IonicPolicy::None,
    }
}

fn h2(r: f64) -> Scene {
    dimer(r, [1.0, 0.0, 0.0], [0.0; 3], 1, 1)
}

/// `scene` in a 49³ cell (12 Bohr): room for a dimer beyond the cutoff, which the 33³ cell's
/// 3h margin (|x| ≤ 3.25 Bohr) refuses since M6 Task 11 (the surrogate applies the cell check).
fn in_big_cell(mut scene: Scene) -> Scene {
    scene.cell = Cell { n: 49, h: 0.25 };
    scene
}

/// Forces every consultation through the variance rules (6–7) while trust holds: the posterior
/// covariance is inflated 10¹⁸ × (the Cholesky factor of Σ⁻¹ scaled by 10⁻⁹), so the epistemic
/// stds are 10⁹ × the fit's (≈ 5e-9 → 5 Ha/Bohr in the window); the mean, the training residuals and σ_f are untouched. (Since M6
/// Task 11 an envelope fallback — rule 5 — is reported on the teacher's surface, so the
/// model-surface bookkeeping tests force the variance instead of the envelope.)
fn always_unsure(s: &mut SurrogateKs) {
    let m = s.model.as_mut().expect("a fitted model");
    for x in m.sigma_inv_chol.as_mut().expect("a posterior").iter_mut() {
        *x *= 1e-9;
    }
    s.last_variance = None;
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
fn verlet_reference(scene: &Scene, teacher: impl ForceProvider + 'static, steps: usize) -> (f64, f64) {
    verlet_reference_dt(scene, teacher, steps, 0.25)
}

/// [`verlet_reference`] at time step `dt_fs`.
fn verlet_reference_dt(scene: &Scene, teacher: impl ForceProvider + 'static, steps: usize, dt_fs: f64) -> (f64, f64) {
    let mut d = Dynamics::new(scene.clone(), Box::new(teacher), dt_fs, 1);
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

    // outside the window (rule 5, the envelope): the teacher answers, the sample is kept, and
    // since M6 Task 11 the step is reported on the teacher's surface — the model's energy there
    // is an extrapolation — and the transition from the model's surface (the r = 1.6 answer) is
    // booked along the forces, E₁ − E₀ + ½(m̃₀ + m̃₁)·Δr, never through the model at r = 3
    let near = h2(3.0);
    let p3 = [[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]];
    let p16 = [[-0.8, 0.0, 0.0], [0.8, 0.0, 0.0]];
    let m16 = s.model().unwrap().predict(&p16, &[1, 1], false).unwrap();
    let r = s.evaluate(&near, &mut never()).unwrap();
    println!("r = 3.0: {:?}, notes {:?}", r.stats.source, r.stats.notes);
    assert_eq!(r.stats.source, Source::Fallback);
    assert_eq!(calls(&h), 1);
    assert_eq!(s.samples(), 41);
    assert!(s.pending_refit());
    let (e3, f3) = teacher.energy_and_forces(&p3).unwrap();
    assert_eq!(r.forces, f3, "the teacher's forces");
    assert_eq!(r.energy, e3, "the teacher's energy");
    assert_eq!(r.stats.teacher_gap, None);
    assert_eq!(r.stats.energy_surface(), Surface::Teacher);
    let jump = r.stats.refit_jump.expect("the transition from the model's surface is booked");
    let work: f64 = (0..2).map(|j| (0..3).map(|a| 0.5 * (m16.forces[j][a] + f3[j][a]) * (p3[j][a] - p16[j][a])).sum::<f64>()).sum();
    let want = e3 - m16.energy + work;
    assert!((jump - want).abs() < 1e-12, "jump {jump} vs E₁ − E₀ + ½(m̃₀ + m̃₁)·Δr {want}");
    assert!(r.stats.energy.total == e3, "the teacher's breakdown is kept");
    assert_eq!(r.stats.model_forces, None, "a teacher-surface step books no mismatch");
    assert!(r.stats.notes.iter().any(|n| n.contains("reported on the teacher's surface")), "{:?}", r.stats.notes);

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
/// is learned at the next refit — once its reference energy exists. Since round 6 the class rule
/// sees the one H–He sample: no other row carries an H–He feature, its leave-one-out energy is
/// 1.1e3× the tier, and its class (pair, with the 180 dimers) is untrusted — H₂ falls back too —
/// until 40 H–He dimers make it predictable; then both are answered.
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
    // class-wise trust (round 6): the one H–He sample is in the pair class with the 180 dimers,
    // and the fit without it cannot predict it (no other row carries an H–He feature) — so the
    // pair class is untrusted, H₂ included, until H–He is learned
    let pair = s.class_loo()[0].clone();
    println!("pair class with one H–He sample: {pair:?}");
    assert!(!pair.trusted && pair.samples == 181 && pair.energy_max > 1.0, "{pair:?}");
    let r = s.evaluate(&h2(1.6), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback);
    assert!(r.stats.notes.iter().any(|n| n.contains("this kind of environment isn't learned well enough yet (pair: LOO")), "{:?}", r.stats.notes);
    assert_event_rule(drain(&mut s).last().unwrap(), sim::activity::TeachRule::Class(sim::activity::EnvClass::Pair));
    // H–He dimers over the window: the class earns its trust back and both are answered
    let mut rng = StdRng::seed_from_u64(9);
    for i in 0..40 {
        let sc = dimer(1.2 + i as f64 / 39.0, unit_vector(&mut rng), [0.0; 3], 1, 2);
        s.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
    }
    s.refit().unwrap();
    let pair = s.class_loo()[0].clone();
    println!("pair class with 41 H–He samples: {pair:?}");
    assert!(pair.trusted, "{pair:?}");
    for sc in [h2(1.6), h_he.clone()] {
        s.discontinuity();
        let r = s.evaluate(&sc, &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Surrogate, "{:?}", r.stats.notes);
    }
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
    // beyond r_cut both atoms are isolated (M6 Task 11, rule 5a: never answered in a scene of
    // more than one atom, whatever the envelope's limit)
    assert!(r.stats.notes.iter().any(|n| n.contains("isolated atom")), "{:?}", r.stats.notes);
    assert!(r.stats.uncertainty.is_some_and(|u| u < policy.force_std_abs), "the variance is still reported");
    assert_eq!(r.stats.energy_surface(), Surface::Teacher, "rule 5 is reported on the teacher's surface");
    // inside the window the envelope is quiet
    let r = s.evaluate(&h2(1.7), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Surrogate);
    // the spread rule (5c) catches the same direction inside r_cut: the first stretched
    // geometry it refuses while the coordination ratio (5b) and the variance would answer
    let (r_out, notes) = [2.6, 2.8, 3.0, 3.2, 3.4]
        .into_iter()
        .map(|r| (r, s.evaluate(&h2(r), &mut never()).unwrap().stats.notes))
        .find(|(_, n)| n.iter().any(|x| x.contains("training spreads outside the descriptor envelope")))
        .expect("a stretched geometry the spread rule refuses");
    println!("R = {r_out}: {notes:?}");
    // the envelope's limit is the policy's: with it off the spread rule is silent there (what
    // refuses R_out then is rule 5d — a dimer beyond the 1.2–2.2 window reaches directions the
    // window's rows span only to rounding, M6 Task 11 round 2); the isolated-atom rule is not
    // the envelope's
    s.set_policy(FallbackPolicy { max_descriptor_distance: f64::INFINITY, ..policy });
    let r = s.evaluate(&h2(r_out), &mut never()).unwrap();
    assert!(r.stats.notes.iter().all(|n| !n.contains("training spreads outside")), "{:?}", r.stats.notes);
    println!("R = {r_out} with the spread limit off: {:?} {:?}", r.stats.source, r.stats.notes.first());
    let r = s.evaluate(&far, &mut never()).unwrap();
    assert!(r.stats.notes.iter().any(|n| n.contains("isolated atom")), "{:?}", r.stats.notes);
}

/// One-surface accounting through the driver: with the policy forcing fallbacks, the atoms
/// move on the teacher's forces while `energy_ks` is the model's and `teacher_gap` the known
/// difference between the two surfaces.
#[test]
fn fallback_step_uses_dft_forces_but_books_the_model_energy() {
    // trained on D = 0.17; the live teacher then has D′ = 1.01 D (energies and forces differ,
    // the gap within the energy tier: a fallback beyond it is reported on the teacher's surface
    // since M6 Task 11 — 1.25 D before).
    // Fallbacks are forced through the variance (`always_unsure`): zero force thresholds would
    // also fail the training-residual trust condition, which uses the same numbers, an
    // untrusted model is a teacher-surface fallback, and so (M6 Task 11) is the envelope's.
    let (mut s, h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    always_unsure(&mut s);
    let model = s.model().unwrap().clone();
    let mut live = MorseDimer::h2();
    live.d *= 1.01;
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
    // the first evaluation of a history (M6 Task 11): a fallback starts it on the teacher's
    // surface, never on the model's guess
    assert_eq!(snap0.energy_ks, e_t0, "the teacher's energy on the first evaluation");
    assert_eq!(snap0.eval.energy_surface(), Surface::Teacher);
    assert_eq!(snap0.forces, f_t0, "the teacher's forces");
    assert_eq!(snap0.teacher_gap, None);

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
    // the transition onto the model's surface is booked along the forces (M6 Task 11):
    // E_model(r1) − E_teacher(r0) + ½(F_teacher(r0) + F_model(r1))·Δr
    let back = snap1.eval.refit_jump.expect("the transition back is booked");
    let f_m1 = model.predict(&p1, &[1, 1], false).unwrap().forces;
    let work: f64 = (0..2).map(|j| (0..3).map(|a| 0.5 * (f_t0[j][a] + f_m1[j][a]) * (p1[j][a] - p0[j][a])).sum::<f64>()).sum();
    let want = e_m1 - e_t0 + work;
    assert!((back - want).abs() < 1e-14, "jump {back} vs {want}");
    let gap = snap1.teacher_gap.unwrap();
    assert!((gap - (e_t1 - e_m1)).abs() < 1e-14, "gap {gap} vs {}", e_t1 - e_m1);
    // the known offset between the surfaces: 0.01 D [(1 − x)² − 1] at r₁
    let r = ((p1[1][0] - p1[0][0]).powi(2) + (p1[1][1] - p1[0][1]).powi(2) + (p1[1][2] - p1[0][2]).powi(2)).sqrt();
    let x = (-(r - 1.4)).exp();
    let offset = 0.01 * 0.17 * ((1.0 - x) * (1.0 - x) - 1.0);
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
    // forced through the variance (see `fallback_step_uses_dft_forces_but_books_the_model_energy`);
    // the first evaluation is on the teacher's surface and the first step books the flip back
    let (mut s, _h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    always_unsure(&mut s);
    // the live teacher carries a ripple the model never saw: 5e-4 Ha (so every gap stays inside
    // the energy tier and the steps stay on the model's surface — M6 Task 11; before, a 25 %
    // deeper well, whose 2e-2 Ha/atom gaps are now teacher-surface steps) with a 0.25 Bohr
    // period, a force difference of up to 1.3e-2 Ha/Bohr
    let live = || Rippled { inner: MorseDimer::h2(), amp: 5e-4, k: 2.0 * std::f64::consts::PI / 0.25 };
    s.teacher = Box::new(live());
    let mut scene = h2(1.55);
    scene.nuclei[0].vel = [-3e-4, 0.0, 0.0];
    scene.nuclei[1].vel = [3e-4, 0.0, 0.0];
    // the same trajectory on the teacher alone: what Verlet itself does to E_conserved
    // (the dimer starts 0.15 Bohr off r_e, 4e-3 Ha of oscillation energy, so the swing is
    // tens of µHa — far above the 1e-7 of a near-equilibrium start)
    let (ref_spread, _) = verlet_reference(&scene, live(), 200);
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

/// A Morse teacher with a ripple A·sin(k r) along the bond (consistent forces).
struct Rippled {
    inner: MorseDimer,
    amp: f64,
    k: f64,
}

impl ForceProvider for Rippled {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        let mut r = self.inner.evaluate(scene, should_abort)?;
        let (p0, p1) = (scene.nuclei[0].pos, scene.nuclei[1].pos);
        let d: [f64; 3] = std::array::from_fn(|a| p1[a] - p0[a]);
        let rr = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let de = self.amp * self.k * (self.k * rr).cos();
        r.energy += self.amp * (self.k * rr).sin();
        for a in 0..3 {
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

/// Ruling S15, fallback thinning (module doc, "Trust recovery"), with round 1 of the 2026-10-18
/// trust review (MUST-2, `is_label_copy`): fallbacks at one geometry while the model is untrusted
/// for its residuals keep the first answer (a new label) and none of its repetitions — copies of a
/// stored sample, labels within the fit's noise (before: one in [`THIN_EVERY`] was kept — rows that
/// shrink the fitted noise); a geometry far from every stored one is kept at once; while the trust
/// rule still needs samples a new answer is kept and its repetitions are not.
#[test]
fn thinning_keeps_one_in_ten_near_identical_fallbacks_and_every_far_one() {
    let (mut s, _) = trained(1.2, 2.2, 40, FallbackPolicy::default());
    assert!(s.trusted());
    // untrusted by its residuals (an impossible energy threshold), not by its sample count
    s.set_policy(FallbackPolicy { energy_std_per_atom: 1e-12, ..FallbackPolicy::default() });
    assert!(s.untrusted_reason().unwrap().starts_with("max training energy residual"));
    let n0 = s.samples();
    let mut copies = 0;
    for _ in 0..25 {
        let r = s.evaluate(&h2(1.6), &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback);
        copies += r.stats.notes.iter().filter(|n| n.contains("sample not kept — a copy")).count();
    }
    println!("25 fallbacks at r = 1.6: {} kept, {copies} copies", s.samples() - n0);
    assert_eq!((s.samples() - n0, copies), (1, 24), "the first is a new label, the rest are its copies");
    // far from every stored geometry: kept right away
    let far = s.nearest_stored(&h2(4.0).nuclei.iter().map(|n| n.pos).collect::<Vec<_>>(), &[1, 1]);
    s.evaluate(&h2(4.0), &mut never()).unwrap();
    println!("r = 4.0 is {far:.1} training spreads from the nearest stored sample: kept");
    assert!(far > THIN_FRACTION * FallbackPolicy::default().max_descriptor_distance);
    assert_eq!(s.samples() - n0, 2);
    // a model that still needs data keeps a new answer, and not its repetitions
    let (mut few, _) = trained(1.2, 2.2, 5, FallbackPolicy::default());
    assert!(few.untrusted_reason().unwrap().starts_with("n_samples"));
    let m0 = few.samples();
    for _ in 0..5 {
        few.evaluate(&h2(1.6), &mut never()).unwrap();
    }
    assert_eq!(few.samples() - m0, 1);
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
        let sc = in_big_cell(dimer(r, unit_vector(&mut rng), [0.0; 3], 1, 1));
        s.add_sample(morse_sample(&long, &sc), &sc).unwrap();
    }
    s.refit().unwrap();
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    // the first answer beyond the cutoff: kept (far from everything), then quarantined (a 49³
    // cell: the 33³ one's margin refuses |x| > 3.25 since the surrogate checks the cell)
    let r = s.evaluate(&in_big_cell(h2(8.0)), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback);
    s.refit().unwrap();
    let (n1, q1) = (s.samples(), s.quarantined());
    println!("after R = 8: {n1} samples, {q1} quarantined, trusted {}, note {:?}", s.trusted(), s.recovery_note());
    assert!(s.trusted() && q1 == 1, "{:?}", s.untrusted_reason());
    let mut rejected_vote = false;
    for k in 0..600 {
        let r = 7.0 + 3.0 * ((k as f64) * 0.37).sin().abs();
        let ev = s.evaluate(&in_big_cell(h2(r)), &mut never()).unwrap();
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
    assert_eq!(s.evaluate(&in_big_cell(h2(1.6)), &mut never()).unwrap().stats.source, Source::Surrogate);
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
            return Ok(ForceResult { energy: e, forces: vec![[0.0; 3]], stats: EvalStats::default(), display, orbitals: None });
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
    // at r = 3.2 strict mode refuses: the audit judged the guess, not the strict model (fix (b),
    // 2026-10-18) — recorded, held out of the fit, trust unchanged
    assert!(surrogate_of(&sh_d).trusted() && surrogate_of(&sh_d).held_out() >= 1);
    d.apply(DynCommand::Step(3));
    let t = d.tick(f64::INFINITY, &mut never());
    assert_eq!(t.source, Source::Surrogate, "still answering in draft mode");
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
    assert!(sk.last_audit_this_fit(), "the background audit judged the current fit");
    let c = sk.trust_checks().into_iter().find(|c| !c.met).expect("the failed audit is unmet");
    match c.key {
        sim::activity::TrustKey::Audit => assert!((c.value - sk.audit_failed().unwrap().abs() / 2.0).abs() < 1e-15, "per atom: {c:?}"),
        sim::activity::TrustKey::AuditForces => assert!(c.value > c.limit),
        k => panic!("the audit is the refusal, not {k:?}"),
    }
    assert_eq!(sk.last_audit().map(|a| a.verdict), Some(AuditVerdict::Failed));
    // a refit clears it
    sk.refit().unwrap();
    assert!(sk.audit_failed().is_none());
    assert!(!sk.last_audit_this_fit(), "the record is the earlier fit's");
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
        // M8a Task 2 (the Task 1 physics review's I4): a box-affected audit judged nothing, so it does
        // not satisfy the cadence — every answered step since was audited again (the teacher's
        // answer is box-affected every time here) and the checklist counts them; the steps stay on
        // the model's surface (the background path keeps answering from the model meanwhile; the
        // synchronous path's audits are teacher steps)
        let status = Trainable::status(&*sk, &h2(1.5), &[]);
        assert!(status.audits_box_unjudged >= 2, "{} box-affected audits counted since the last judged one", status.audits_box_unjudged);
        assert_eq!(status.audits_box_unjudged, sk.audit_history().unrecorded.box_affected as usize, "every one of them left a count, no point");
        drop(sk);
        d.apply(DynCommand::Step(1));
        let snap = d.tick(f64::INFINITY, &mut never());
        let sk = surrogate_of(&sh_d);
        if background {
            assert_eq!(snap.source, Source::Surrogate, "the model answers while the re-audit solves");
            assert!(sk.audit_in_flight() || sk.last_audit().is_some_and(|a| a.verdict == AuditVerdict::BoxAffected), "a re-audit was dispatched");
        } else {
            assert_eq!((snap.source, snap.eval.energy_surface()), (Source::Fallback, Surface::Model), "the next answered step is audited in the step, on the model's surface");
            assert!(snap.eval.notes.iter().any(|n| n.contains("surrogate audit not judged")), "{:?}", snap.eval.notes);
        }
        assert!(sk.trusted(), "the box-affected audits never untrust: {:?}", sk.untrusted_reason());
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
    // Ha/Bohr) above the model's worst training residual and its σ_f — trust holds — but below
    // its epistemic uncertainty in the stretched tail (the spread envelope off, so the variance
    // rule is what refuses; the tail stays within the coordination ratio, f_c(R) ≥ f_c(2.2)/3,
    // R ≤ 4.08 Bohr — the rule M6 Task 11 added, which the limit does not switch off)
    let residual = s.training_residuals().unwrap().force;
    s.set_policy(FallbackPolicy { max_descriptor_distance: 1e9, max_grade: f64::INFINITY, force_std_abs: 1e-5, force_std_rel: 0.0, variance_cadence: k, audit_every: 0, ..FallbackPolicy::default() });
    assert!(s.trusted() && residual < 1e-5, "{:?} ({residual:.1e})", s.untrusted_reason());
    let r_far = [4.0, 3.6, 3.2]
        .into_iter()
        .find(|&r| {
            s.discontinuity();
            // the verdict's rejection: rule 6, or rule 5d, which is evaluated with it (M6 Task 11)
            s.evaluate(&h2(r), &mut never()).unwrap().stats.notes.iter().any(|n| n.contains("force std") || n.contains("new environment"))
        })
        .expect("a geometry the variance verdict refuses");
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
    let mut outcomes = Vec::new();
    for i in 0..30 {
        let r = 1.9 + 0.002 * (i % 5) as f64;
        let res = s.evaluate(&h2(r), &mut never()).unwrap();
        outcomes.extend(drain(&mut s).into_iter().map(|e| (e.rule, e.outcome)));
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
    {
        // the fence decides the admission whichever rule refused (ruling S15): here 5c fired
        // first, so the rows say Spread with the fence's outcomes (`NearQuarantine` is the code
        // only when nothing else refused)
        use sim::activity::{TeachOutcome as K, TeachRule as R};
        assert_eq!(outcomes[..3], [(R::Spread, K::FenceDisagrees), (R::Spread, K::FenceDisagrees), (R::Spread, K::KeptByVote)], "{outcomes:?}");
    }
    let i = answered.expect("the model never learned the fenced region");
    assert!(i <= 5 && refits <= 2, "{i} evaluations, {refits} refits");
    assert!(s.trusted() && s.quarantined() == 0 && s.superseded() == 1, "the glitch is superseded, not released");
    assert!(after.abs() / 2.0 <= 1e-3, "the region is learned: {after:+.2e}");
}

/// Fence votes along a trajectory (M8a Task 5, the user's compressed-H₂ log: "fallback · fence ·
/// not kept: fence" every vibration period, residuals 1.39–1.66 mHa, never a vote): a molecule
/// that re-enters the fence at slightly different bond lengths and turned between visits gives
/// raw answers that differ by more than the tiers — the vote compared those — while its
/// residuals teacher − model agree. The vote compares residuals (rotation-invariantly), so the
/// third such answer is kept.
#[test]
fn a_vibrating_turning_molecule_in_a_fence_still_votes() {
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
    let stray = h2(1.9);
    let mut bad = morse_sample(&MorseDimer::h2(), &stray);
    bad.energy += 0.1;
    s.add_sample(bad, &stray).unwrap();
    s.refit().unwrap();
    assert!(s.trusted() && s.is_quarantined(60), "{:?}", s.untrusted_reason());
    // three visits: bond lengths 1.900, 1.906, 1.903 Bohr, the axis turned by 40° each time
    let visit = |k: usize| {
        let a = (40.0f64 * k as f64).to_radians();
        dimer([1.900, 1.906, 1.903][k], [a.cos(), a.sin(), 0.0], [0.0; 3], 1, 1)
    };
    // the premise: the raw answers disagree pairwise (the old vote), the model misses them all
    let raw: Vec<ForceResult> = (0..3).map(|k| teacher().evaluate(&visit(k), &mut never()).unwrap()).collect();
    let raw_agree = SurrogateKs::agree(s.policy(), 2, raw[0].energy, &raw[0].forces, raw[1].energy, &raw[1].forces);
    assert!(!raw_agree, "test premise: the raw answers of two visits differ by more than the tiers");
    let mut outcomes = Vec::new();
    let mut voted = None;
    for k in 0..3 {
        let res = s.evaluate(&visit(k), &mut never()).unwrap();
        assert_ne!(res.stats.source, Source::Surrogate, "visit {k} falls back (fenced)");
        outcomes.extend(drain(&mut s).into_iter().map(|e| e.outcome));
        if voted.is_none() && res.stats.notes.iter().any(|n| n.contains("mutually consistent fresh teacher answers")) {
            voted = Some(k);
        }
    }
    use sim::activity::TeachOutcome as K;
    println!("outcomes {outcomes:?}");
    assert_eq!(voted, Some(2), "the third visit is the vote: {outcomes:?}");
    assert_eq!(outcomes[..3], [K::FenceDisagrees, K::FenceDisagrees, K::KeptByVote]);
}

/// Diagnosis (M8a Task 5, the user's κ = 136.65): the calibration readout κ_c of a collection's
/// audit history against the decision numbers — the measured error W_c and the variance
/// inflation λ_c — and how much of κ_c's mean of z² its largest points carry. Reads the file
/// named by `COLLECTION_HISTORY` (a collection's `history.bin`; a copy, never the live one).
#[test]
#[ignore = "diagnosis: COLLECTION_HISTORY=<copy of history.bin> cargo test -p providers --release kappa_tail -- --ignored --nocapture"]
fn kappa_tail_of_a_collection_history() {
    let path = std::env::var("COLLECTION_HISTORY").expect("COLLECTION_HISTORY names a history.bin");
    let h = surrogate::AuditHistory::decode(&std::fs::read(path).unwrap()).unwrap();
    let pol = FallbackPolicy::default();
    let cal = super::calibrate::calibrate(&h, &pol);
    use sim::activity::EnvClass;
    let class_codes = [super::history_rule_code(TeachRule::Class(EnvClass::Pair)), super::history_rule_code(TeachRule::Class(EnvClass::ManyBody))];
    for (c, cc) in cal.per_class.iter().enumerate() {
        println!(
            "class {}: κ {:.2} (ẑ {:.2}, {} points: {} audits, {} fallbacks, evidence {}), measured W {:.3}, energy {:.3}, worst atom {:.2}, inflation λ {:.3} (raw {:.3}, {} points)",
            ENV_CLASSES[c], cc.factor, cc.z_rms, cc.points, cc.audits, cc.fallbacks, cc.evidence, cc.measured, cc.measured_energy, cc.worst_atom, cc.inflation, cc.inflation_raw, cc.inflation_points
        );
        // the verdict jurisdiction's points (all generations): z̄² per point, the share of the largest
        let mut z2: Vec<(f64, f64, f64, f64)> = Vec::new(); // (z̄², ‖ΔF‖ max, std at that atom, w̄)
        for p in h.iter() {
            let in_j = match p.kind {
                surrogate::history::PointKind::Audit => true,
                surrogate::history::PointKind::Fallback { rule } => class_codes.contains(&rule),
            };
            if !in_j {
                continue;
            }
            let (mut s, mut n, mut worst, mut wsum) = (0.0, 0usize, (0.0f64, 0.0f64, 0.0f64), 0.0);
            for j in 0..p.n_atoms {
                if p.classes[j] != c as u8 + 1 {
                    continue;
                }
                let z = p.err_force[j] / p.pred_force_std[j];
                s += z * z;
                n += 1;
                if z > worst.0 {
                    worst = (z, p.err_force[j], p.pred_force_std[j]);
                }
                wsum += (p.err_force[j] / 3f64.sqrt() / pol.force_bound(p.teacher_force[j])).powi(2);
            }
            if n > 0 {
                z2.push((s / n as f64, worst.1, worst.2, (wsum / n as f64).sqrt()));
            }
        }
        z2.sort_by(|a, b| b.0.total_cmp(&a.0));
        let total: f64 = z2.iter().map(|x| x.0).sum();
        let share = |k: usize| z2.iter().take(k).map(|x| x.0).sum::<f64>() / total;
        let without = |k: usize| (z2.iter().skip(k).map(|x| x.0).sum::<f64>() / (z2.len() - k).max(1) as f64).sqrt();
        println!(
            "  all-generation jurisdiction: {} points, ẑ {:.2}; the largest 1 / 5 / 10 points carry {:.1} % / {:.1} % / {:.1} % of Σz̄²; ẑ without them {:.2} / {:.2} / {:.2}; median z̄ {:.2}",
            z2.len(), (total / z2.len().max(1) as f64).sqrt(), 100.0 * share(1), 100.0 * share(5), 100.0 * share(10), without(1), without(5), without(10),
            z2.get(z2.len() / 2).map_or(f64::NAN, |x| x.0.sqrt())
        );
        for (k, x) in z2.iter().take(5).enumerate() {
            println!("    top {k}: z̄ {:.1} (worst atom ‖ΔF‖ {:.2e} Ha/Bohr over a std of {:.2e}); its error in bound units {:.2}", x.0.sqrt(), x.1, x.2, x.3);
        }
    }
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
#[ignore = "diagnostic: cargo test -p providers --release diag_confinement -- --ignored --nocapture"]
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
    let atom = Scene { nuclei: vec![Nucleus::new(1, [0.0; 3])], cell: cell.clone(), charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3], ionic: sim::IonicPolicy::None };
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
        Scene { nuclei: vec![Nucleus::new(1, p0), Nucleus::new(1, p1)], cell: cell(), charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3], ionic: sim::IonicPolicy::None }
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
    // the cheap accessors agree with the readouts
    assert_eq!((Trainable::fitted(&s), Trainable::samples(&s), Trainable::pending_refit(&s)), (st.fitted, st.samples, st.pending_refit));
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

/// Review M5: when the relax's last evaluation was a fallback (the forces already the
/// teacher's), the result is reported as verified without another teacher solve.
#[test]
fn a_relax_that_ended_on_a_fallback_is_verified_without_another_solve() {
    const DT_FS: f64 = 0.25;
    // every evaluation goes through the teacher (`always_unsure`: the variance refuses), booked
    // on the model surface — after the first, which starts the history on the teacher's (M6
    // Task 11)
    let (mut s, h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    always_unsure(&mut s);
    let samples0 = s.samples();
    let (p, sh_d) = driven(s);
    let mut d = Dynamics::new(dimer(1.7, [1.0, 0.0, 0.0], [0.0; 3], 1, 1), p, DT_FS, 1);
    let first = d.snapshot();
    assert_eq!((first.source, first.eval.energy_surface()), (Source::Fallback, Surface::Teacher), "{:?}", first.eval.notes);
    let e_c0 = first.energy_conserved;
    d.apply(DynCommand::Relax);
    let (last, r, calls_before_final) = loop {
        let c = calls(&h);
        let snap = d.tick(-1.0, &mut never());
        let r = snap.relax.clone().expect("relaxing");
        assert!((snap.energy_conserved - e_c0).abs() <= 1e-10, "step {}: E_conserved moved by {:.2e}", r.steps, snap.energy_conserved - e_c0);
        assert_eq!(snap.source, Source::Fallback);
        if r.converged.is_some() {
            break (snap, r, c);
        }
        assert!(r.steps < 2000);
    };
    assert_eq!(r.converged, Some(true));
    // the final tick made the step's evaluations (one, or two with a backtrack) and nothing more
    let final_calls = calls(&h) - calls_before_final;
    assert!((1..=2).contains(&final_calls), "{final_calls} teacher calls in the final tick: a verification solve was added");
    assert!(!last.eval.notes.iter().any(|n| n.contains("relax verification")), "{:?}", last.eval.notes);
    // the forces are the teacher's, so the verified number is the step's own raw maximum
    let pos: Vec<[f64; 3]> = last.scene.nuclei.iter().map(|n| n.pos).collect();
    let (_, f_t) = MorseDimer::h2().energy_and_forces(&pos).unwrap();
    assert_eq!(last.forces, f_t);
    assert_eq!(r.verified_max_force, Some(r.max_force));
    assert!(r.max_force < 1e-3);
    println!("relax on forced fallbacks: {} steps, verified = raw max|F| {:.2e}, {} samples kept", r.steps, r.max_force, surrogate_of(&sh_d).samples() - samples0);
}

/// A scripted teacher that holds a density (M6 Task 2): the Morse dimer's energies, and after
/// every evaluation a 9³ (h 0.5) density whose constant value names the geometry — 10·N plus the
/// first two nuclei's distance — **kept across `reset`/`discontinuity`** (the worst case: the
/// surrogate's own flag, not the teacher's manners, must keep a stale density from the caller).
struct DensityTeacher {
    morse: MorseDimer,
    rho: Option<f64>,
}

impl DensityTeacher {
    fn tag(scene: &Scene) -> f64 {
        let p = &scene.nuclei;
        10.0 * p.len() as f64 + if p.len() > 1 { (0..3).map(|a| (p[0].pos[a] - p[1].pos[a]).powi(2)).sum::<f64>().sqrt() } else { 0.0 }
    }
}

impl ForceProvider for DensityTeacher {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        let r = if scene.nuclei.len() == 1 {
            ForceResult { energy: self.morse.e_atom, forces: vec![[0.0; 3]], stats: EvalStats::default(), display: None, orbitals: None }
        } else {
            self.morse.evaluate(scene, should_abort)?
        };
        self.rho = Some(Self::tag(scene));
        Ok(r)
    }
    fn name(&self) -> &'static str {
        "morse dimer"
    }
    fn reset(&mut self) {}
    fn sample_density(&mut self, points: &[[f64; 3]]) -> Option<Vec<f64>> {
        let v = self.rho?;
        let grid = qm::Grid::cube(9, 0.5);
        let field = vec![v; grid.len()];
        Some(points.iter().map(|&p| grid.trilinear(&field, p)).collect())
    }
}

/// M6 Task 2 (spec §3.3, review S4): the surrogate hands out its stepping teacher's density only
/// when the latest step was the teacher's. No model → every step is a Fallback → `Some` (that
/// geometry's density); a trusted model's `Surrogate` answer → `None` although the teacher still
/// holds the previous fallback's density; a later far fallback → `Some` again (its geometry);
/// after a fallback, `teacher_mut().evaluate(atom)` (the driver's reference solve) → `None`, and
/// still `None` after the `reset()` that follows; `discontinuity()` → `None`.
#[test]
fn the_surrogate_samples_only_after_a_fallback() {
    let points = [[0.0; 3], [0.3, -0.2, 0.1], [9.0, 0.0, 0.0]];
    let teacher = DensityTeacher { morse: MorseDimer::h2(), rho: None };
    let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_teacher_tol(1e-6);
    s.set_reference_energy(1, MorseDimer::h2().e_atom);
    assert_eq!(s.sample_density(&points), None, "nothing evaluated yet");

    // no model: the teacher answers, and its density is this geometry's
    let r = s.evaluate(&h2(1.4), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback);
    assert_eq!(s.sample_density(&points), Some(vec![21.4, 21.4, 0.0]));

    // a trusted model answers: no density, though the teacher still holds R = 1.4's
    let mut rng = StdRng::seed_from_u64(4);
    for i in 0..40 {
        let r = 1.1 + 0.9 * i as f64 / 39.0;
        let origin = [rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3)];
        let sc = dimer(r, unit_vector(&mut rng), origin, 1, 1);
        s.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
    }
    s.refit().unwrap();
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    let r = s.evaluate(&h2(1.5), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Surrogate, "{:?}", r.stats.notes);
    assert_eq!(s.sample_density(&points), None, "a surrogate answer has no density");

    // a far geometry falls back: the teacher's density, of THIS geometry
    let r = s.evaluate(&h2(2.6), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback, "{:?}", r.stats.notes);
    assert_eq!(s.sample_density(&points), Some(vec![22.6, 22.6, 0.0]));

    // S4: the driver's reference-atom solve goes through `teacher_mut`, then `reset`; the teacher
    // now holds the atom's density, which must never be handed out for the molecule
    let mut atom = h2(2.6);
    atom.nuclei.truncate(1);
    s.teacher_mut().evaluate(&atom, &mut never()).unwrap();
    assert_eq!(s.sample_density(&points), None, "after a teacher_mut solve");
    s.reset();
    assert_eq!(s.sample_density(&points), None, "after the reset that follows it (S4)");

    // a fallback, then a teleport
    s.evaluate(&h2(2.7), &mut never()).unwrap();
    assert!(s.sample_density(&points).is_some());
    s.discontinuity();
    assert_eq!(s.sample_density(&points), None, "after a discontinuity");
}

/// Task 2 review I1: the test wrappers forward `sample_density` (the trait default is `None`, so
/// a missing forward would read as "no density"): `Shared`, `SharedTrainable` and the teacher
/// proxy `TeacherOf` its `teacher_mut` lends.
#[test]
fn the_forwarding_wrappers_pass_sample_density_through() {
    let points = [[0.0; 3]];
    let mut shared = Shared::new(DensityTeacher { morse: MorseDimer::h2(), rho: None });
    assert_eq!(shared.sample_density(&points), None);
    shared.evaluate(&h2(1.4), &mut never()).unwrap();
    assert_eq!(shared.sample_density(&points), Some(vec![21.4]));

    let s = SurrogateKs::new(Box::new(DensityTeacher { morse: MorseDimer::h2(), rho: None }), FallbackPolicy::default()).with_teacher_tol(1e-6);
    let mut wrapped = SharedTrainable::new(s);
    wrapped.evaluate(&h2(1.4), &mut never()).unwrap(); // no model: a fallback
    assert_eq!(wrapped.sample_density(&points), Some(vec![21.4]), "SharedTrainable");
    let teacher = wrapped.as_trainable().expect("trainable").teacher_mut();
    assert_eq!(teacher.sample_density(&points), Some(vec![21.4]), "TeacherOf");
}

/// The stepping teacher of [`the_references_are_the_background_auditors`], counted: a Gaussians
/// stand-in atom on every nucleus plus a bond bump (bonded H₂), named like its auditor.
struct WatchedGaussians {
    inner: sim::Gaussians,
    evals: Arc<std::sync::atomic::AtomicUsize>,
    resets: Arc<std::sync::atomic::AtomicUsize>,
}

impl ForceProvider for WatchedGaussians {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        self.evals.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.inner.evaluate(scene, should_abort)
    }
    fn name(&self) -> &'static str {
        self.inner.name()
    }
    fn reset(&mut self) {
        self.resets.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.inner.reset()
    }
    fn sample_density(&mut self, points: &[[f64; 3]]) -> Option<Vec<f64>> {
        self.inner.sample_density(points)
    }
}

/// M6 Task 3 review I2: with a background auditor the bond detector's isolated-atom references
/// are solved in its job slot (`request_atom_reference`), never on the stepping teacher. Through
/// a `Dynamics` over a model-less surrogate (every step a fallback, so the stepping teacher's
/// density is the step's): the constructor's density queues H; while the steps run, the
/// stepping teacher sees one evaluation per step and no reset — the atom was solved elsewhere;
/// the entry lands (multiplicity 2, a shell table) and the graph is published from the stepping
/// teacher's density (one bond) — which also pins that the driver no longer calls `teacher_mut`
/// per detector run (that call told the surrogate its teacher's state was gone, so
/// `sample_density` answered `None` after every fallback and no graph was ever published under
/// the surrogate).
#[test]
fn the_references_are_the_background_auditors() {
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    let bonded = || sim::Gaussians::per_atom(0.3, 0.5).with_pair_bump(0.15, 0.5, 3.0);
    let (evals, resets) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let teacher = WatchedGaussians { inner: bonded(), evals: evals.clone(), resets: resets.clone() };
    let s = SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_auditor(Box::new(bonded()));
    let (p, _sh) = driven(s);
    let mut d = Dynamics::new(h2(1.6), p, 0.25, 1);
    let s0 = d.snapshot();
    assert_eq!(s0.bonds_pending, vec![1], "the constructor's density queues H");
    let (e0, r0) = (evals.load(SeqCst), resets.load(SeqCst));
    let t0 = Instant::now();
    let t = loop {
        d.apply(DynCommand::Step(1));
        let t = d.tick(f64::INFINITY, &mut never());
        if t.bonds_pending.is_empty() && t.bonds.is_some() {
            break t;
        }
        assert!(t0.elapsed().as_secs_f64() < 2.0, "no graph within 2 s: pending {:?}, notes {:?}", t.bonds_pending, t.notes);
        std::thread::sleep(std::time::Duration::from_millis(1));
    };
    println!("after {} steps: {:?}", t.step, t.notes.last());
    assert_eq!(evals.load(SeqCst) - e0, t.step as usize, "the stepping teacher evaluated only the steps");
    assert_eq!(resets.load(SeqCst) - r0, 0, "the stepping teacher was never reset for the reference");
    assert!(t.notes.iter().any(|n| n.contains("atom reference H (Z = 1): multiplicity 2") && n.contains("off the stepping path")), "{:?}", t.notes);
    let g = t.bonds.clone().unwrap();
    assert_eq!(((g.bonds.len(), g.n_nuclei)), (1, 2), "{:?}", t.pair_reports);
    assert!(!t.bonds_stale);
}

/// M6 Task 3 review I2: every accepted reference request is answered exactly once — a repeat
/// while it is in flight is the same request; a Forget abandons it and it comes back
/// `Cancelled` (not dropped, so the driver is never left waiting); the next one is solved by the
/// auditor's teacher (here the Morse stand-in: E = E_atom, multiplicity 2, no density, so an
/// empty table).
#[test]
fn a_reference_request_is_answered_once_even_when_abandoned() {
    let (mut s, _h, open, _) = with_gated_auditor(0, 0.0);
    let sc = h2(1.5);
    assert!(s.request_atom_reference(1, 1837.15, &sc));
    assert!(s.request_atom_reference(1, 1837.15, &sc), "a repeat is the same request");
    Trainable::forget(&mut s);
    let wait = |s: &mut SurrogateKs| {
        let t0 = Instant::now();
        loop {
            let a = s.take_atom_references();
            if !a.is_empty() {
                return a;
            }
            assert!(t0.elapsed().as_secs_f64() < 2.0, "no answer within 2 s");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    };
    let a = wait(&mut s);
    assert_eq!(a.len(), 1);
    assert!(matches!(a[0], (1, Err(ProviderError::Cancelled))), "{:?}", a[0].1.as_ref().err());
    open.store(true, std::sync::atomic::Ordering::SeqCst);
    assert!(s.request_atom_reference(1, 1837.15, &sc));
    let a = wait(&mut s);
    let (z, r) = &a[0];
    let r = r.as_ref().expect("solved");
    assert_eq!((*z, r.multiplicity, r.energy), (1, Some(2), MorseDimer::h2().e_atom));
    assert!(r.radial.is_empty(), "the stand-in has no density");
    std::thread::sleep(std::time::Duration::from_millis(20));
    assert!(s.take_atom_references().is_empty(), "answered once");
}


// ---- M6 Task 11: the user's stuck-then-escape report (`docs/superpowers/notes/
// 2026-10-15-stuck-escape-diagnosis.md`) ----

/// Requirement 1 (diagnosis R1): the surrogate applies the teacher's inside-cell check (R12)
/// before any rule — a model answer is a function of the positions alone and would carry atoms
/// out of the cell. Same error, same message as `Scene::check_inside_cell` (what `KsProvider`
/// returns), nothing evaluated, nothing kept.
#[test]
fn the_surrogate_refuses_a_nucleus_outside_the_cell_like_its_teacher() {
    let (mut s, h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    // a well-trained geometry, translated until nucleus 1 is 0.25 Bohr past the 3h margin
    let max = cell().max_coord();
    let out = dimer(1.6, [1.0, 0.0, 0.0], [max + 0.25 - 0.8, 0.0, 0.0], 1, 1);
    let want = out.check_inside_cell().expect_err("test premise: outside the margin");
    let n0 = s.samples();
    let got = s.evaluate(&out, &mut never()).expect_err("the model must not answer outside the cell");
    println!("{got}");
    assert!(matches!((&got, &want), (ProviderError::Invalid(a), ProviderError::Invalid(b)) if a == b), "the teacher's error and message: {got:?} vs {want:?}");
    assert_eq!((calls(&h), s.samples()), (0, n0), "nothing solved, nothing kept");
    // the same geometry inside the cell is the model's
    let inside = dimer(1.6, [1.0, 0.0, 0.0], [max - 0.8, 0.0, 0.0], 1, 1);
    assert_eq!(s.evaluate(&inside, &mut never()).unwrap().stats.source, Source::Surrogate);
    // and in draft mode, which answers past rules 4–7, the cell still refuses
    s.set_policy(FallbackPolicy { never_fallback: true, audit_every: 0, ..FallbackPolicy::default() });
    let got = s.evaluate(&out, &mut never()).expect_err("draft mode too");
    assert_eq!(got.to_string(), want.to_string());
}

/// Requirement 1 through the driver: a model-driven dimer heated until it dissociates (CSVR at
/// 30 000 K, D = 0.17 Ha ≈ 1.8 k_B T) pauses at the cell's margin with the teacher's message —
/// the user's escape (|x| = 287 Bohr) cannot happen — and with the container walls on it never
/// reaches the margin. Every published scene is inside the margin.
#[test]
fn a_surrogate_driven_dimer_heated_apart_pauses_at_the_face_and_walls_keep_it_in() {
    let c = cell();
    for walls in [false, true] {
        let (s, _h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
        let mut d = Dynamics::new(h2(1.5), Box::new(s), 0.25, 7);
        if walls {
            d.apply(DynCommand::SetWalls(Some(sim::Walls::new(sim::Walls::max_half_width(&c)))));
        }
        d.apply(DynCommand::SetThermostat(Some((30_000.0, 10.0))));
        let (mut max_r, mut answered, mut stopped) = (0.0f64, 0usize, None);
        for i in 0..8000 {
            d.apply(DynCommand::Step(1));
            let snap = d.tick(f64::INFINITY, &mut never());
            let p: Vec<[f64; 3]> = snap.scene.nuclei.iter().map(|n| n.pos).collect();
            max_r = max_r.max((0..3).map(|a| (p[1][a] - p[0][a]).powi(2)).sum::<f64>().sqrt());
            answered += (snap.source == Source::Surrogate) as usize;
            let worst = p.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
            assert!(worst <= c.max_coord(), "step {i}: a published scene is outside the margin: |x| = {worst}");
            if let Some(e) = &snap.last_error {
                stopped = Some((i, e.clone()));
                break;
            }
        }
        println!("walls {walls}: max separation {max_r:.2} Bohr, {answered} model answers, stopped {stopped:?}");
        assert!(answered > 0, "the model drove the dimer");
        assert!(max_r > 6.0, "test premise: the dimer dissociated (beyond r_cut)");
        if walls {
            assert_eq!(stopped, None, "the walls keep both atoms off the margin");
        } else {
            let (_, e) = stopped.expect("the run pauses at the margin");
            assert!(e.contains("left the simulation cell"), "{e}");
        }
    }
}

/// Requirement 2 (diagnosis R2): one kept sample of a stretched pair (r = 5.1 Bohr, f_c 0.054)
/// drags the coordination feature's training minimum down while its spread unit stays the
/// bulk's, so an atom with NO neighbour was "inside" the envelope (half a spread out) and was
/// answered with Σ E_ref and zero force. Now: an atom with no neighbour within r_cut in a scene
/// of more than one atom is refused (5a), and a coordination more than 3× below the training
/// minimum is outside (5b); a one-atom scene is still the model's (its answer is the reference
/// itself).
#[test]
fn one_tail_sample_no_longer_lets_a_free_atom_in() {
    let teacher = Shared::new(MorseDimer::h2());
    let handle = teacher.handle();
    let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() }).with_teacher_tol(1e-6);
    s.set_reference_energy(1, MorseDimer::h2().e_atom);
    let mut rng = StdRng::seed_from_u64(4);
    let mut tail_r = Vec::new();
    for i in 0..61 {
        // the pass's scan range 1.1–2.9 (the H₂ pass: 0.7–1.8 × 1.6) and one tail sample at 5.1
        let r = if i < 60 { 1.1 + 1.8 * i as f64 / 59.0 } else { 5.1 };
        let sc = in_big_cell(dimer(r, unit_vector(&mut rng), [0.0; 3], 1, 1));
        s.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
        tail_r.push(r);
    }
    s.refit().unwrap();
    assert!(s.trusted() && s.quarantined() == 0, "the tail sample is a represented training sample: {:?}", s.untrusted_reason());
    // the premise: in spread units an empty neighbourhood is inside the envelope
    let model = s.model().unwrap().clone();
    let env = s.envelope.clone().unwrap();
    let free = describe_values(&model.basis, &[[0.0; 3], [7.0, 0.0, 0.0]], &[1, 1]).unwrap();
    let (spreads, k) = env.distance(0, &free[0]);
    println!("an isolated H: {spreads:.2} training spreads outside (feature {}), coordination minimum {:.4}", model.basis.feature_names()[k], env.min[0]);
    assert!(spreads < FallbackPolicy::default().max_descriptor_distance, "premise: the spread rule alone would answer");
    let calls0 = handle.lock().unwrap().calls;
    // 5a: r = 7 > r_cut — both atoms isolated
    let r = s.evaluate(&in_big_cell(h2(7.0)), &mut never()).unwrap();
    println!("r = 7.0: {:?} {:?}", r.stats.source, r.stats.notes);
    assert_eq!(r.stats.source, Source::Fallback);
    assert!(r.stats.notes.iter().any(|n| n.contains("isolated atom")), "{:?}", r.stats.notes);
    assert_eq!(r.stats.energy_surface(), Surface::Teacher);
    let e = drain(&mut s);
    assert_event_rule(e.last().unwrap(), sim::activity::TeachRule::IsolatedAtom);
    assert_eq!(e.last().unwrap().atom, Some(0));
    // 5b: r = 5.9, a neighbour inside r_cut whose f_c (6.9e-4) is 80× below the minimum
    let r = s.evaluate(&in_big_cell(h2(5.9)), &mut never()).unwrap();
    println!("r = 5.9: {:?} {:?}", r.stats.source, r.stats.notes);
    assert_eq!(r.stats.source, Source::Fallback);
    assert!(r.stats.notes.iter().any(|n| n.contains("below the smallest the fit has seen")), "{:?}", r.stats.notes);
    let e = drain(&mut s);
    assert_event_rule(e.last().unwrap(), sim::activity::TeachRule::Coordination);
    assert!(e.last().unwrap().atom.is_some());
    assert_eq!(handle.lock().unwrap().calls, calls0 + 2);
    // inside the window: since round 6 the class rule refuses it as well — the fit without the
    // lone tail sample predicts it 1.05× its force bound off, so the pair class is not learned
    // well enough (the GPU repro's tail sample: 14.9×, `the_stuck_escape_repro_is_answered_honestly`)
    let pair = s.class_loo()[0].clone();
    println!("pair class with the tail sample: {pair:?}");
    assert!(!pair.trusted && pair.samples == 61 && pair.force_max > 1.0, "{pair:?}");
    let r = s.evaluate(&in_big_cell(h2(1.6)), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback);
    assert!(r.stats.notes.iter().any(|n| n.contains("this kind of environment isn't learned well enough yet (pair: LOO force error")), "{:?}", r.stats.notes);
    assert_event_rule(drain(&mut s).last().unwrap(), sim::activity::TeachRule::Class(sim::activity::EnvClass::Pair));
    // a one-atom scene: the model's answer is E_ref exactly (no neighbour is its training state;
    // the isolated class is exempt from the class rule)
    let mut one = in_big_cell(h2(1.6));
    one.nuclei.truncate(1);
    let r = s.evaluate(&one, &mut never()).unwrap();
    assert_eq!((r.stats.source, r.energy), (Source::Surrogate, MorseDimer::h2().e_atom), "{:?}", r.stats.notes);
}

/// One excursion run of the Requirement-3 tests: a window-trained dimer (1.2–2.2) swinging from
/// r = 2.0 with 0.092 Ha of relative kinetic energy out to r ≈ 3 and back, for 150 fs at `dt`,
/// on `live` (the teacher the fallbacks call). Asserts on every step: a fallback is on the
/// teacher's surface and reports exactly `live`'s energy, a surface change carries a booked jump,
/// the history never restarts. Returns (E_conserved spread, plain-Verlet spread on `live` alone,
/// largest per-step change, Verlet's, switches).
fn excursion(dt: f64, live: &dyn Fn() -> Box<dyn ForceProvider>) -> (f64, f64, f64, f64, usize, f64, f64) {
    let (mut s, _h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    s.teacher = live();
    let mut probe = live();
    let mut scene = h2(2.0);
    // 0.092 Ha of relative kinetic energy: the turning point near r = 3 (outside the window)
    scene.nuclei[0].vel = [-0.0071, 0.0, 0.0];
    scene.nuclei[1].vel = [0.0071, 0.0, 0.0];
    let steps = (150.0 / dt) as usize;
    // Q = E_conserved − dt²Σ|F|²/8m: the booking's own |F|² term removed, so Q moves only by the
    // trapezoid's cubic remainders — the curvature part of the shadow (I1 test)
    let dta = units::fs_to_au_time(dt);
    let q_of = |snap: &DynSnapshot| snap.energy_conserved - dta * dta * snap.forces.iter().zip(&snap.scene.nuclei).map(|(f, n)| (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]) / (8.0 * n.mass)).sum::<f64>();
    let (ref_spread, ref_step, ref_q) = {
        let mut d = Dynamics::new(scene.clone(), live(), dt, 1);
        let s0 = d.snapshot();
        let mut prev = s0.energy_conserved;
        let q0 = q_of(&s0);
        let (mut lo, mut hi, mut max_step, mut q1) = (prev, prev, 0.0f64, q0);
        for _ in 0..steps {
            d.apply(DynCommand::Step(1));
            let snap = d.tick(f64::INFINITY, &mut never());
            let e = snap.energy_conserved;
            lo = lo.min(e);
            hi = hi.max(e);
            max_step = max_step.max((e - prev).abs());
            prev = e;
            q1 = q_of(&snap);
        }
        (hi - lo, max_step, q1 - q0)
    };
    let mut d = Dynamics::new(scene, Box::new(s), dt, 1);
    let e0 = d.snapshot().energy_conserved;
    let q0 = q_of(&d.snapshot());
    let mut q1 = q0;
    let (mut lo, mut hi, mut prev, mut max_step) = (e0, e0, e0, 0.0f64);
    let (mut on_teacher, mut answered, mut switches) = (0, 0, 0);
    let mut last_surface = d.snapshot().eval.energy_surface();
    for i in 0..steps {
        d.apply(DynCommand::Step(1));
        let snap = d.tick(f64::INFINITY, &mut never());
        let surface = snap.eval.energy_surface();
        match snap.source {
            Source::Surrogate => answered += 1,
            Source::Fallback => {
                assert_eq!(surface, Surface::Teacher, "step {i}: an envelope fallback on the model's surface: {:?}", snap.eval.notes);
                let e_t = probe.evaluate(&snap.scene, &mut never()).unwrap().energy;
                assert_eq!(snap.energy_ks, e_t, "step {i}: the reported energy is the teacher's");
                on_teacher += 1;
            }
            Source::Dft => unreachable!(),
        }
        if surface != last_surface {
            assert!(snap.eval.refit_jump.is_some(), "step {i}: a surface change without a booked jump");
            switches += 1;
        }
        last_surface = surface;
        let e = snap.energy_conserved;
        lo = lo.min(e);
        hi = hi.max(e);
        max_step = max_step.max((e - prev).abs());
        prev = e;
        q1 = q_of(&snap);
        assert!(snap.notes.iter().all(|n| !n.contains("history restarts")), "step {i}: {:?}", snap.notes);
    }
    assert!(answered > 0 && on_teacher > 0 && switches >= 4, "test premise: several excursions ({answered} answered, {on_teacher} on the teacher, {switches} switches)");
    (hi - lo, ref_spread, max_step, ref_step, switches, q1 - q0, ref_q)
}

/// Requirement 3 (diagnosis R4), the matched case: the live teacher is the Morse the model was
/// trained on (the model reproduces it to ~1e-7). Every rule-5 step reports exactly the
/// teacher's energy, every switch is booked, and the booking leaves exactly Verlet's own
/// ΔK − ½(F₀ + F₁)·Δr = dt²(|F₁|² − |F₀|²)/8m per step: the largest per-step change of
/// E_conserved is plain Verlet's on the teacher alone (within 10 %).
#[test]
fn out_of_envelope_steps_are_reported_on_the_teacher_surface_with_both_flips_booked() {
    for dt in [0.25, 0.0625] {
        let (spread, ref_spread, max_step, ref_step, switches, _, _) = excursion(dt, &|| Box::new(MorseDimer::h2()));
        println!("matched, dt {dt} fs: {switches} switches; E_conserved spread {spread:.2e} (plain Verlet {ref_spread:.2e}), max step {max_step:.2e} (Verlet {ref_step:.2e})");
        assert!(max_step < 1.1 * ref_step, "largest per-step change {max_step:.2e} vs Verlet's {ref_step:.2e}");
    }
}

/// Requirement 3 with a surface mismatch at the switches (physics review of round 2, I1). The
/// booking is exact per step — a switch step moves E_conserved by exactly Verlet's
/// dt²Σ(|F₁|² − |F₀|²)/8m — but what Verlet conserves is the shadow of the surface in use,
/// H̃ = H + dt²(pᵀV″p/12m² − |F|²/24m) + O(dt⁴), and a switch from surface A to B changes it by
/// δ ≈ (dt²/12)[pᵀ(V″_B − V″_A)p/m² + (|F_B|² − |F_A|²)/m] that no energy booking carries. So
/// E_conserved keeps, per switch, an **absolute residual O(dt²) whose coefficient is the force /
/// curvature mismatch of the two surfaces at the switch** — of the same order as Verlet's own
/// swing, so the spread ratio to plain Verlet plateaus at 1 + O(mismatch) instead of tending to 1.
///
/// The observable: Q = E_conserved − dt²Σ|F|²/8m (the |F|² part removed exactly, step by step),
/// whose change over the run is the curvature part: the single-surface endpoint terms plus the
/// switches' δ. Fixture: the live teacher is the Morse plus a ripple ε sin(kr), ε = 1e-3 Ha,
/// k = 10 Bohr⁻¹ (|ΔF| ≤ εk = 1e-2 Ha/Bohr, |ΔV″| ≤ εk² = 0.1 Ha/Bohr²); along the bond
/// pᵀV″p/m² = v_rel² V″ and Σ|F|²/m = |F|²/μ (μ the reduced mass). The derived bound, at the
/// fixture's maxima (v_rel ≤ 0.0142, |F| ≤ 0.25 Ha/Bohr, Morse V″ ≤ 1.0 Ha/Bohr² on r ≥ 1):
///   |ΔQ| ≤ (dt²/12)[N_switch (v² εk² + (2|F|εk + ε²k²)/μ) + 2 v² V″_max + |F|²_max/μ]
/// (the last two: the endpoint terms of Q itself, Q = H̃ − dt²(pᵀV″p/12m² + |F|²/12m) up to
/// O(dt⁴) — the review's round-5 correction added the |F|² one).
/// Asserted at dt = 0.25 … 0.015625 fs. A ratio test excess(dt)/excess(dt/2) ≈ 4 is not
/// measurable on this fixture: a switch lands at the first step past the envelope's edge, at a
/// dt-dependent point within that step, so each δ's size and sign change with dt — measured
/// spread excesses 1.3e-3, 1.6e-4, 1.9e-5, 9.9e-7, −1.6e-8 Ha and ΔQ/dt² from −4e-2 to −4e-5
/// Ha/fs² (reported in the log): bounded by the O(dt²) coefficient, not proportional to it.
#[test]
fn a_surface_mismatch_at_the_switches_leaves_an_order_dt2_residual() {
    let (eps, k) = (1e-3, 10.0);
    let live = || -> Box<dyn ForceProvider> { Box::new(Rippled { inner: MorseDimer::h2(), amp: 1e-3, k: 10.0 }) };
    let mu = 0.5 * Nucleus::new(1, [0.0; 3]).mass;
    let (v2, f_max, vpp_max) = (0.0142f64.powi(2), 0.25, 1.0);
    for dt in [0.25, 0.125, 0.0625, 0.03125, 0.015625] {
        let (spread, ref_spread, _, _, switches, dq, dq_ref) = excursion(dt, &live);
        let dta = units::fs_to_au_time(dt);
        let bound = dta * dta / 12.0 * (switches as f64 * (v2 * eps * k * k + (2.0 * f_max * eps * k + eps * eps * k * k) / mu) + 2.0 * v2 * vpp_max + f_max * f_max / mu);
        println!("ripple, dt {dt} fs: {switches} switches; spread {spread:.3e} vs plain Verlet {ref_spread:.3e} (excess {:.3e}, ratio {:.3}); ΔQ {dq:+.3e} (plain Verlet {dq_ref:+.3e}), ΔQ/dt² {:+.3e} Ha/fs²; bound {bound:.3e} ({:.2} of it)",
                 spread - ref_spread, spread / ref_spread, dq / (dt * dt), dq.abs() / bound);
        assert!(dq.abs() <= bound, "dt {dt}: |ΔQ| {:.3e} beyond the O(dt²) mismatch bound {bound:.3e}", dq.abs());
    }
}

/// Requirement 4 (diagnosis R3): a trusted model falling back where every teacher answer is
/// box-affected learns nothing (those answers are never kept); after `STUCK_STEPS` such
/// fallbacks the stuck note says so and what to do, instead of only naming the rule. One answer
/// inside the box rule ends that run (the plain note returns).
#[test]
fn a_stuck_model_says_when_every_answer_touches_the_box_wall() {
    let rho = Arc::new(Mutex::new(2.0 * BOX_AFFECTED_FACE_DENSITY));
    let teacher = FaceDensity { inner: Arc::new(Mutex::new(MorseDimer::h2())), rho: rho.clone() };
    let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() }).with_teacher_tol(1e-6);
    s.set_reference_energy(1, MorseDimer::h2().e_atom);
    let mut rng = StdRng::seed_from_u64(4);
    for i in 0..40 {
        let sc = dimer(1.2 + i as f64 / 39.0, unit_vector(&mut rng), [0.0; 3], 1, 1);
        s.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
    }
    s.refit().unwrap();
    let n0 = s.samples();
    for _ in 0..STUCK_STEPS {
        assert_eq!(s.evaluate(&h2(2.8), &mut never()).unwrap().stats.source, Source::Fallback);
    }
    let why = s.stuck().expect("stuck");
    println!("{why}");
    assert!(why.contains("every answer here touches the box wall — enlarge the grid (or turn container walls on)"), "{why}");
    assert_eq!(s.samples(), n0, "nothing was learned");
    *rho.lock().unwrap() = 0.5 * BOX_AFFECTED_FACE_DENSITY;
    s.evaluate(&h2(2.8), &mut never()).unwrap();
    let why = s.stuck().expect("still stuck");
    assert!(!why.contains("box wall") && why.contains("descriptor envelope"), "{why}");
}

/// Requirement 4: the panel's "fit vs held" — the store count equals the fit's count plus the
/// quarantined, superseded and box-excluded samples and those kept since the last refit,
/// exactly (the user's "230 vs 250": 20 box-excluded pass samples the panel never mentioned).
#[test]
fn the_status_reconciles_the_fit_count_with_the_store() {
    let state = Arc::new(Mutex::new((2.0 * BOX_AFFECTED_FACE_DENSITY, 0.05)));
    let mut s = SurrogateKs::new(Box::new(Confined { inner: MorseDimer::h2(), rho: state.clone() }), FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() }).with_teacher_tol(1e-6);
    s.set_reference_energy(1, MorseDimer::h2().e_atom);
    for r in [1.5, 1.6, 1.7] {
        s.evaluate(&h2(r), &mut never()).unwrap();
    }
    let mut rng = StdRng::seed_from_u64(4);
    for i in 0..70 {
        let sc = dimer(1.2 + i as f64 / 69.0, unit_vector(&mut rng), [0.0; 3], 1, 1);
        s.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
    }
    s.refit().unwrap();
    let check = |s: &SurrogateKs, what: &str| {
        let st = Trainable::status(s, &h2(1.6), &[[0.0; 3]; 2]);
        let fit = st.fit.expect("a fit").0;
        println!("{what}: held {} = fit {fit} + quarantined {} + superseded {} + box-excluded {} + unfitted {}", st.samples, st.quarantined, st.superseded, st.box_excluded, st.unfitted);
        assert_eq!(st.samples, fit + st.quarantined + st.superseded + st.box_excluded + st.unfitted, "{what}");
        st
    };
    let st = check(&s, "after the refit");
    assert_eq!((st.samples, st.box_excluded, st.unfitted), (73, 3, 0));
    // a kept fallback (rule 5, far from the stored ones, inside the box rule) is newer than the fit
    *state.lock().unwrap() = (0.0, 0.0);
    s.evaluate(&h2(2.9), &mut never()).unwrap();
    let st = check(&s, "after one kept fallback");
    assert_eq!(st.unfitted, 1);
}

/// Requirement 5 (diagnosis R6), the split: rule 6 compares the model's epistemic force std
/// only. The fit's σ_f is the label noise — force_std² = 3σ_f² + epistemic², exactly — and a
/// fallback on it teaches nothing (rule 7 already leaves the energy floor out). With the model's
/// σ_f set to 3e-3 (above the 2e-3 bound) and its posterior unchanged, the old rule fell back
/// at equilibrium on every step — the user's cycle; now the model answers. Whether such a fit
/// may answer at all is the trust rule's question: with the fit's own σ_f at 3e-3 it is
/// untrusted ("force noise").
#[test]
fn the_force_rule_leaves_the_label_noise_out_and_trust_judges_it() {
    let (mut s, _h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    let eq = h2(1.4);
    s.model.as_mut().unwrap().sigma_f = 3e-3;
    s.last_variance = None;
    let pred = s.model().unwrap().predict(&[[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], &[1, 1], true).unwrap();
    let (total, epi) = (pred.force_std.as_ref().unwrap()[0], pred.force_std_epistemic.as_ref().unwrap()[0]);
    println!("at r_e: force_std/√3 {:.2e} (σ_f 3e-3 included), epistemic/√3 {:.2e}", total / SQRT_3, epi / SQRT_3);
    assert!((total * total - (27e-6 + epi * epi)).abs() < 1e-15, "the exact split");
    let r = s.evaluate(&eq, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Surrogate, "{:?}", r.stats.notes);
    assert!(r.stats.uncertainty.is_some_and(|u| u < 1e-4), "the readout is the epistemic std: {:?}", r.stats.uncertainty);
    // the trust rule reads the fit's own σ_f
    s.report.as_mut().unwrap().sigma_f = 3e-3;
    let why = s.untrusted_reason().expect("untrusted");
    println!("{why}");
    assert!(why.contains("force noise σ_f 3.0e-3 > 2.0e-3"), "{why}");
    let r = s.evaluate(&eq, &mut never()).unwrap();
    assert_eq!((r.stats.source, r.stats.energy_surface()), (Source::Fallback, Surface::Teacher));
}

/// Requirement 5, end to end: a fit to deliberately noisy force labels — 120 compressed
/// geometries (r 0.85–1.1, |F| 0.19–0.40, every bound capped at 1e-2) with ±8e-3 Ha/Bohr on
/// every component, ‖δF‖/√3 = 8e-3: inside the training-residual bound, so the S8 rule alone
/// trusts the fit — plus 40 clean window samples. σ_f comes out ≈ 7e-3: before, the model was
/// trusted and fell back at equilibrium on every consultation (force_std/√3 ≥ σ_f > 2e-3 — the
/// user's cycle). Now it is untrusted ("force noise") and the refit's recovery cannot quarantine
/// 120 of 160 (the 10 % cap): every step is a plain teacher step — no cycle. A sampling pass's
/// worth of clean samples (1100) brings the noisy ones under the cap; the next refit
/// quarantines them, σ_f drops to the clean fit's and trust returns; the model then answers at
/// equilibrium every time.
#[test]
fn a_noisy_fit_loses_trust_recovers_and_a_clean_one_never_cycles() {
    let teacher = Shared::new(MorseDimer::h2());
    let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() }).with_teacher_tol(1e-6);
    let m = MorseDimer::h2();
    s.set_reference_energy(1, m.e_atom);
    let mut rng = StdRng::seed_from_u64(11);
    let add = |s: &mut SurrogateKs, rng: &mut StdRng, r: f64, noise: f64| {
        let sc = dimer(r, unit_vector(rng), [0.0; 3], 1, 1);
        let mut smp = morse_sample(&m, &sc);
        for f in smp.forces.iter_mut() {
            for x in f.iter_mut() {
                *x += if rng.gen::<bool>() { noise } else { -noise };
            }
        }
        s.add_sample(smp, &sc).unwrap();
    };
    for i in 0..40 {
        add(&mut s, &mut rng, 1.2 + i as f64 / 39.0, 0.0);
    }
    for i in 0..120 {
        add(&mut s, &mut rng, 0.85 + 0.25 * i as f64 / 119.0, 8e-3);
    }
    let rep = s.refit().unwrap();
    println!("noisy fit: n {}, σ_f {:.2e}, training residuals {:?}; trusted {}: {:?}; quarantined {}", rep.n_samples, rep.sigma_f, s.training_residuals(), s.trusted(), s.untrusted_reason(), s.quarantined());
    assert!(rep.sigma_f > 2e-3, "test premise: σ_f {:.2e}", rep.sigma_f);
    let t = s.training_residuals().unwrap();
    assert!(t.force <= t.force_bound && t.energy_per_atom <= 1e-3, "test premise: the S8 rule alone trusts it: {t:?}");
    assert!(!s.trusted());
    assert!(s.untrusted_reason().unwrap().contains("force noise"), "{:?}", s.untrusted_reason());
    for k in 0..20 {
        let r = s.evaluate(&h2(1.35 + 0.005 * k as f64), &mut never()).unwrap();
        assert_eq!((r.stats.source, r.stats.energy_surface()), (Source::Fallback, Surface::Teacher), "untrusted: plain teacher steps");
    }
    // recovery: clean samples (what a sampling pass adds), then the refit quarantines the noisy
    for i in 0..1100 {
        add(&mut s, &mut rng, 0.85 + 1.35 * i as f64 / 1099.0, 0.0);
    }
    let rep = s.refit().unwrap();
    println!("after 1100 clean samples: n {}, σ_f {:.2e}, quarantined {}, trusted {}: {:?}; recovery {:?}", rep.n_samples, rep.sigma_f, s.quarantined(), s.trusted(), s.untrusted_reason(), s.recovery_note());
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    assert!(rep.sigma_f <= 2e-3 && s.quarantined() >= 100, "σ_f {:.2e}, {} quarantined", rep.sigma_f, s.quarantined());
    // a clean model near equilibrium: answered on every consultation, no cycle
    s.discontinuity();
    let mut answered = 0;
    for k in 0..200 {
        let r = 1.4 + 0.08 * (k as f64 * 0.3).sin();
        answered += (s.evaluate(&h2(r), &mut never()).unwrap().stats.source == Source::Surrogate) as usize;
    }
    assert_eq!(answered, 200);
}

// ---- M6 Task 11 round 2 (controller ruling on concern 1): unspanned directions, the
// coordination-change cadence, audits on forces ----

/// A pair-additive Morse teacher for any number of hydrogens (the dimer fixture's pair): what a
/// three-atom fallback needs, since `MorseDimer` takes exactly two.
struct PairMorse(MorseDimer);

impl ForceProvider for PairMorse {
    fn evaluate(&mut self, scene: &Scene, _should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        let m = &self.0;
        let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        let n = pos.len();
        let mut energy = n as f64 * m.e_atom;
        let mut forces = vec![[0.0; 3]; n];
        for i in 0..n {
            for j in i + 1..n {
                let d: [f64; 3] = std::array::from_fn(|a| pos[j][a] - pos[i][a]);
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                let (e, de) = m.energy_and_derivative(r);
                energy += e - 2.0 * m.e_atom;
                for a in 0..3 {
                    forces[i][a] += de * d[a] / r;
                    forces[j][a] -= de * d[a] / r;
                }
            }
        }
        let mut stats = EvalStats::default();
        stats.energy.total = energy;
        Ok(ForceResult { energy, forces, stats, display: None, orbitals: None })
    }
    fn name(&self) -> &'static str {
        "morse dimer"
    }
    fn reset(&mut self) {}
}

/// `scene` in a 65³ cell (16 Bohr: |x| ≤ 7.25), room for a third atom beyond r_cut of a dimer.
fn in_cell_65(mut scene: Scene) -> Scene {
    scene.cell = Cell { n: 65, h: 0.25 };
    scene
}

/// A dimer-only model (window 1.0–5.1 Bohr, like the user's store with its stretched tail) with
/// a pair-additive three-atom teacher, in the 65³ cell.
fn dimer_model_three_atom_teacher() -> SurrogateKs {
    let mut s = SurrogateKs::new(Box::new(PairMorse(MorseDimer::h2())), FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() }).with_teacher_tol(1e-6);
    s.set_reference_energy(1, MorseDimer::h2().e_atom);
    let mut rng = StdRng::seed_from_u64(21);
    for i in 0..80 {
        let sc = in_cell_65(dimer(1.0 + 4.1 * i as f64 / 79.0, unit_vector(&mut rng), [0.0; 3], 1, 1));
        s.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
    }
    s.refit().unwrap();
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    s
}

/// H₂ with a third H `d` Bohr from its centre, perpendicular to the bond.
fn h2_plus_h(d: f64) -> Scene {
    let mut sc = in_cell_65(h2(1.4));
    sc.nuclei.push(Nucleus::new(1, [0.0, d, 0.0]));
    sc
}

/// Rule 5d (M6 Task 11 round 2): an H₂-only model asked about H₂ + H. No feature column was zero
/// in training — every body-3 feature of one neighbour is a function of its distance — but the
/// dimer data leave directions no training row spans, and a third atom reaches them: with the
/// spread envelope off, each falls back with the novelty reason, on the teacher's surface. (On
/// this noise-free fixture the posterior alone refuses some of them as well — reported; on the
/// GPU teacher's H₂ store it answered them with 4–22× force errors, the Task 11 repro.) A dimer
/// (every geometry on the training manifold) is still answered.
#[test]
fn an_h2_only_model_falls_back_on_h2_plus_h_as_a_new_environment() {
    let mut s = dimer_model_three_atom_teacher();
    s.set_policy(FallbackPolicy { max_descriptor_distance: f64::INFINITY, audit_every: 0, ..FallbackPolicy::default() });
    let ns = s.null_space.clone().expect("a null space after the fit");
    println!("unspanned directions: {} of {}", ns.dimension(), s.basis().unwrap().n_independent_columns());
    for d in [2.0, 3.0, 5.0] {
        let sc = h2_plus_h(d);
        let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
        let model = s.model().unwrap().clone();
        let pred = model.predict(&pos, &[1, 1, 1], true).unwrap();
        let premise = s.variance_check(&model, &pred, pred.energy_std.unwrap(), pred.force_std_epistemic.as_ref().unwrap(), &pos);
        let (nu, _) = ns.novelty(&pos, &[1, 1, 1]).unwrap();
        s.discontinuity();
        let r = s.evaluate(&sc, &mut never()).unwrap();
        println!("H₂ + H at {d} Bohr: the posterior alone {} ({:?}); ν {nu:.1e}; {:?} {:?}", if premise.accept { "answers" } else { "refuses" }, premise.reason, r.stats.source, r.stats.notes.first());
        assert_eq!(r.stats.source, Source::Fallback);
        assert!(r.stats.notes.iter().any(|n| n.contains("new environment (feature never seen in training)")), "{:?}", r.stats.notes);
        assert_eq!(r.stats.energy_surface(), Surface::Teacher);
        // the null-space novelty (5d) or, where a component leaves every training row's span
        // per atom, 5e at γ = ∞: either is "new environment", each with its own code
        let e = drain(&mut s);
        let last = e.last().unwrap();
        assert!(matches!(last.rule, sim::activity::TeachRule::NewEnvironment | sim::activity::TeachRule::UnspannedAtom), "{last:?}");
        assert_event_rule(last, last.rule);
    }
    for r in [1.1, 2.3, 4.7] {
        s.discontinuity();
        let ev = s.evaluate(&in_cell_65(h2(r)), &mut never()).unwrap();
        assert_eq!(ev.stats.source, Source::Surrogate, "dimer at {r}: {:?}", ev.stats.notes);
    }
}

/// The variance cadence (M6 Task 11 round 2): an accepted verdict is reused for at most k − 1
/// consultations — but never across a change of any atom's neighbour count within r_cut (a
/// neighbour arriving is a new environment for the verdict).
#[test]
fn a_coordination_change_reevaluates_the_verdict() {
    let mut s = dimer_model_three_atom_teacher();
    s.set_policy(FallbackPolicy { variance_cadence: 4, audit_every: 0, ..FallbackPolicy::default() });
    let checks = |s: &mut SurrogateKs, sc: &Scene| {
        let before = s.variance_checks();
        s.evaluate(sc, &mut never()).unwrap();
        s.variance_checks() - before
    };
    // the third atom beyond r_cut of both (neighbour counts [1, 1, 0]): the first consultation
    // checks, the next one reuses
    let far = |y: f64| h2_plus_h(y);
    assert_eq!(checks(&mut s, &far(6.3)), 1);
    assert_eq!(checks(&mut s, &far(6.31)), 0, "the cadence reuses an accepted verdict");
    // the third atom moves inside r_cut of both: counts [2, 2, 2] — checked afresh
    assert_eq!(checks(&mut s, &far(5.8)), 1, "a coordination change must re-evaluate the verdict");
}

/// Audits judge forces too (M6 Task 11 round 2): a teacher whose energy differs from the
/// model's by less than the energy tier (a 5e-4 Ha ripple) but whose forces differ by more than
/// the force bound (A·k = 1.3e-2 Ha/Bohr along the bond at r = 1.5, where cos kr = 1) fails the
/// audit on forces and takes the failed-audit path: teacher surface, the jump booked,
/// untrusted with the reason, cleared by a refit.
#[test]
fn a_force_only_audit_failure_untrusts_as_designed() {
    let (mut s, _h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 1, ..FallbackPolicy::default() });
    s.teacher = Box::new(Rippled { inner: MorseDimer::h2(), amp: 5e-4, k: 2.0 * std::f64::consts::PI / 0.25 });
    let first = s.evaluate(&h2(1.5), &mut never()).unwrap(); // after the fit: the first answer
    let r = if first.stats.notes.iter().any(|n| n.contains("audit")) { first } else { s.evaluate(&h2(1.5), &mut never()).unwrap() };
    println!("{:?}; untrusted: {:?}", r.stats.notes, s.untrusted_reason());
    assert!(r.stats.notes.iter().any(|n| n.contains("surrogate audit failed on forces")), "{:?}", r.stats.notes);
    assert_eq!(r.stats.energy_surface(), Surface::Teacher);
    assert!(s.audit_failed().is_some() && !s.trusted());
    assert!(s.untrusted_reason().unwrap().contains("audit found forces off by"), "{:?}", s.untrusted_reason());
    s.refit().unwrap();
    assert!(s.audit_failed().is_none());
}

// ---- M6 Task 11 round 4 (controller ruling): the extrapolation grade, rule 5e ----

/// A random H₃ geometry: an H₂ (bond 1.2–1.8) and a third atom 1.6–4 Bohr from its centre,
/// random orientation, inside the 65³ cell.
fn random_h3(rng: &mut StdRng) -> Scene {
    let r = rng.gen_range(1.2..1.8);
    let d = rng.gen_range(1.6..4.0);
    let u = unit_vector(rng);
    let mut w = unit_vector(rng);
    let dot = u[0] * w[0] + u[1] * w[1] + u[2] * w[2];
    for a in 0..3 {
        w[a] -= dot * u[a];
    }
    let n = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
    let w: [f64; 3] = std::array::from_fn(|a| w[a] / n);
    let mut sc = in_cell_65(dimer(r, u, [0.0; 3], 1, 1));
    let t: f64 = rng.gen_range(-0.6..0.6);
    sc.nuclei.push(Nucleus::new(1, std::array::from_fn(|a| d * (w[a] * (1.0 - t * t).sqrt() + u[a] * t))));
    sc
}

/// The PairMorse sample of a scene (its exact answer).
fn pair_morse_sample(sc: &Scene) -> Sample {
    let r = PairMorse(MorseDimer::h2()).evaluate(sc, &mut never()).unwrap();
    Sample { pos: sc.nuclei.iter().map(|n| n.pos).collect(), z: sc.nuclei.iter().map(|n| n.z).collect(), energy: r.energy, forces: r.forces, tag: SampleTag::Scan, weight: 1.0 }
}

/// Rule 5e on the dimer-only model plus k learned three-atom samples (pair-additive teacher),
/// judged on 40 held-out H₂ + H geometries (judged with `envelope_outside` and `predict`, so the
/// queries never become samples). The danger the grade exists for: with 1–3 H₃ samples the
/// model's forces on the held-out set are 33–65× their bound (`all` below) while the posterior
/// is confident; every one of them falls back. Measured (pinned): no held-out geometry answered
/// for k ≤ 10 (the grade or another envelope rule refuses), the first answers at k = 20 (26 of
/// 40, worst force ratio 0.02), all 40 at k = 80; at every k no answered geometry is beyond 3× its
/// bound. And every training row's grade is ≤ 1 + `MAXVOL_TOL` (the MaxVol property).
#[test]
fn h2_plus_h_falls_back_on_the_grade_until_enough_is_learned() {
    let mut s = dimer_model_three_atom_teacher();
    let mut rng = StdRng::seed_from_u64(31);
    let queries: Vec<Scene> = (0..40).map(|_| random_h3(&mut rng)).collect();
    let mut train_rng = StdRng::seed_from_u64(32);
    let mut added = 0;
    let mut answered_at = Vec::new();
    // the refusals' codes (Task 2 review M1), each checked against its text
    let mut codes = std::collections::BTreeMap::new();
    for k in [0usize, 1, 2, 3, 5, 10, 20, 40, 80] {
        while added < k {
            let sc = random_h3(&mut train_rng);
            s.add_sample(pair_morse_sample(&sc), &sc).unwrap();
            added += 1;
        }
        if k > 0 {
            s.refit().unwrap();
        }
        assert!(s.trusted(), "{:?}", s.untrusted_reason());
        let model = s.model().unwrap().clone();
        let (mut answered, mut worst_answered, mut worst_all) = (0, 0.0f64, 0.0f64);
        for q in &queries {
            let pos: Vec<[f64; 3]> = q.nuclei.iter().map(|n| n.pos).collect();
            let (_, reason) = s.envelope_outside(&model, &pos, &[1, 1, 1]);
            if let Some(r) = &reason {
                assert_refusal(r);
                *codes.entry(format!("{:?}", r.rule)).or_insert(0usize) += 1;
            }
            let t = pair_morse_sample(q);
            let pr = model.predict(&pos, &[1, 1, 1], false).unwrap();
            let f = pr.forces.iter().zip(&t.forces).map(|(a, b)| { let (d, bd) = s.policy().force_residual(a, b); d / bd }).fold(0.0, f64::max);
            worst_all = worst_all.max(f);
            if reason.is_none() {
                answered += 1;
                worst_answered = worst_answered.max(f);
            }
        }
        println!("{k} H₃ samples: {answered}/40 answered (worst force ratio {worst_answered:.2}); all 40: worst {worst_all:.2}");
        assert!(worst_answered <= 3.0, "k = {k}: an answered geometry {worst_answered:.1}× its force bound");
        answered_at.push((k, answered));
        // the MaxVol property on every training row
        s.ensure_desc_cache();
        let act = s.active_set.clone().unwrap();
        let worst_train = (0..s.accumulated.min(s.desc_cache.len()))
            .filter(|&i| !s.excluded(i))
            .flat_map(|i| s.desc_cache[i].iter().map(|(sp, b)| act.grade(*sp, b)).collect::<Vec<_>>())
            .fold(0.0f64, f64::max);
        assert!(worst_train <= 1.0 + surrogate::grade::MAXVOL_TOL + 1e-9, "k = {k}: a training row's grade {worst_train}");
    }
    assert_eq!(answered_at, vec![(0, 0), (1, 0), (2, 0), (3, 0), (5, 0), (10, 0), (20, 26), (40, 37), (80, 40)], "the pinned transition");
    println!("refusal codes over the sweep: {codes:?}");
    assert!(["Grade", "Spread", "UnspannedAtom"].iter().all(|c| codes.contains_key(*c)), "5c, 5e and 5e at γ = ∞ are each met: {codes:?}");
}

/// Rule 5e when MaxVol did not converge (physics review, round 5): the γ ≤ 1 + δ guarantee is
/// void, so every query falls back with the reason, and a converged refit restores the grade.
#[test]
fn a_non_converged_active_set_falls_back_on_every_query() {
    let (mut s, _h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    s.ensure_desc_cache();
    let model = s.model().unwrap().clone();
    let atoms: Vec<(usize, Vec<f64>)> = (0..s.accumulated).flat_map(|i| s.desc_cache[i].clone()).collect();
    let capped = ActiveSet::build_capped(&model.basis, &model.scale, atoms.iter().map(|(sp, b)| (*sp, &b[..])), 0);
    assert!(!capped.converged(), "test premise: the greedy start alone is not a MaxVol set on this store");
    s.active_set = Some(capped);
    let r = s.evaluate(&h2(1.6), &mut never()).unwrap();
    println!("{:?}", r.stats.notes);
    assert_eq!(r.stats.source, Source::Fallback);
    assert!(r.stats.notes.iter().any(|n| n.contains("MaxVol did not converge")), "{:?}", r.stats.notes);
    let e = drain(&mut s);
    assert_event_rule(e.last().unwrap(), sim::activity::TeachRule::NoGrade);
    assert_eq!(e.last().unwrap().atom, None);
    s.refit().unwrap();
    assert!(s.active_set.as_ref().unwrap().converged());
    assert_eq!(s.evaluate(&h2(1.6), &mut never()).unwrap().stats.source, Source::Surrogate);
}

/// Rule 5e: a geometry inside the trained H₂ window answers (its grade within the active set),
/// and the grade is invariant under rotations and translations (the descriptors are).
#[test]
fn the_grade_answers_inside_the_window_and_is_rotation_invariant() {
    let (mut s, _h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    let model = s.model().unwrap().clone();
    let act = s.active_set.clone().expect("an active set after the fit");
    let grade_of = |pos: &[[f64; 3]]| describe_values(&model.basis, pos, &vec![1; pos.len()]).unwrap().iter().map(|d| act.grade(0, d)).fold(0.0f64, f64::max);
    let inside = grade_of(&[[-0.8, 0.0, 0.0], [0.8, 0.0, 0.0]]);
    println!("H₂ at 1.6: grade {inside:.3}");
    assert!(inside <= 1.0 + surrogate::grade::MAXVOL_TOL);
    assert_eq!(s.evaluate(&h2(1.6), &mut never()).unwrap().stats.source, Source::Surrogate);
    // rotation + translation of a three-atom geometry (its grade is whatever it is — ∞ here — and
    // the same) and of a dimer just outside the window (finite)
    let mut rng = StdRng::seed_from_u64(77);
    for pos in [vec![[0.0, 0.0, 0.0], [2.4, 0.0, 0.0]], vec![[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.7, 2.5, 0.3]]] {
        let g0 = grade_of(&pos);
        for _ in 0..5 {
            let (a, b) = (unit_vector(&mut rng), rng.gen_range(0.0..std::f64::consts::TAU));
            // Rodrigues rotation by angle b about axis a, then a shift
            let rot = |p: [f64; 3]| -> [f64; 3] {
                let (c, sn) = (b.cos(), b.sin());
                let d = a[0] * p[0] + a[1] * p[1] + a[2] * p[2];
                let cr = [a[1] * p[2] - a[2] * p[1], a[2] * p[0] - a[0] * p[2], a[0] * p[1] - a[1] * p[0]];
                std::array::from_fn(|k| p[k] * c + cr[k] * sn + a[k] * d * (1.0 - c) + 0.3 * (k as f64 + 1.0))
            };
            let g = grade_of(&pos.iter().map(|&p| rot(p)).collect::<Vec<_>>());
            println!("grade {g0:.6} rotated {g:.6}");
            assert!(g == g0 || (g - g0).abs() <= 1e-9 * g0.abs().max(1.0), "{g} vs {g0}");
        }
    }
}

/// The class rule (round 6, module doc "Class-wise trust"): per atom the neighbour count within
/// r_cut, per geometry the largest — 0 isolated (exempt), 1 pair, ≥ 2 many-body — the same for
/// any species (nothing per element), with r_cut the only length.
#[test]
fn environment_classes_follow_the_largest_coordination() {
    assert_eq!(env_class(&[]), None);
    assert_eq!(env_class(&[0]), None);
    assert_eq!(env_class(&[0, 0, 0]), None);
    assert_eq!(env_class(&[1, 1]), Some(0));
    assert_eq!(env_class(&[1, 1, 0]), Some(0));
    assert_eq!(env_class(&[2, 1, 1]), Some(1));
    assert_eq!(env_class(&[5, 1, 1, 1, 1, 1]), Some(1));
    assert_eq!(ENV_CLASSES, ["pair", "many-body"]);
    let r_cut = 6.0;
    let rc2 = r_cut * r_cut;
    for z in [[1u8, 1, 1], [1, 2, 8], [6, 6, 1]] {
        // a pair and a third atom just outside / just inside r_cut of the nearer one
        for (d, want) in [(r_cut + 0.7 + 1e-9, Some(0)), (r_cut + 0.7 - 1e-9, Some(1))] {
            let pos = [[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0], [d, 0.0, 0.0]];
            assert_eq!(env_class(&neighbour_counts(&pos, rc2)), want, "z {z:?}, third atom at {d}");
        }
    }
}

/// The class trust on the dimer-only model plus k learned three-atom samples (pair-additive
/// teacher; the 40 held-out H₂ + H geometries of the grade test): the many-body class is
/// untrusted while it holds fewer than Q′ samples or any of them is predicted beyond its bound by
/// the fit without it, and a held-out geometry is answered only when both the class rule and rule
/// 5 let it. The transition is pinned (measured); every answered geometry within 3× its bound,
/// and the pair class (the 80 dimers) stays trusted throughout.
#[test]
fn the_many_body_class_earns_trust_as_samples_arrive() {
    let mut s = dimer_model_three_atom_teacher();
    let q = s.report().unwrap().q_prime;
    let rc2 = s.basis().unwrap().r_cut.powi(2);
    let mut rng = StdRng::seed_from_u64(31);
    let queries: Vec<Scene> = (0..40).map(|_| random_h3(&mut rng)).collect();
    let mut train_rng = StdRng::seed_from_u64(32);
    let mut added = 0;
    let mut trace = Vec::new();
    for k in [0usize, 10, 20, 28, 29, 30, 40, 60, 80, 120] {
        while added < k {
            let sc = random_h3(&mut train_rng);
            s.add_sample(pair_morse_sample(&sc), &sc).unwrap();
            added += 1;
        }
        if k > 0 {
            s.refit().unwrap();
        }
        assert!(s.trusted(), "{:?}", s.untrusted_reason());
        let pair = s.class_loo()[0].clone();
        let many = s.class_loo()[1].clone();
        assert!(pair.trusted, "k = {k}: the dimers' class {pair:?}");
        assert_eq!(many.samples, k);
        let model = s.model().unwrap().clone();
        let (mut answered, mut worst) = (0, 0.0f64);
        for qs in &queries {
            let pos: Vec<[f64; 3]> = qs.nuclei.iter().map(|n| n.pos).collect();
            let counts = neighbour_counts(&pos, rc2);
            assert_eq!(env_class(&counts), Some(1));
            let class = s.class_untrusted(&counts);
            assert_eq!(class.is_none(), many.trusted);
            if let Some((reason, short)) = &class {
                assert_eq!(*short, k < q, "{reason}");
                assert_refusal(reason);
                assert_eq!(reason.rule, sim::activity::TeachRule::Class(sim::activity::EnvClass::ManyBody));
            }
            let (_, envelope) = s.envelope_outside(&model, &pos, &[1, 1, 1]);
            if let Some(r) = &envelope {
                assert_refusal(r);
            }
            if class.is_none() && envelope.is_none() {
                answered += 1;
                let t = pair_morse_sample(qs);
                let pr = model.predict(&pos, &[1, 1, 1], false).unwrap();
                worst = worst.max(pr.forces.iter().zip(&t.forces).map(|(a, b)| { let (d, bd) = s.policy().force_residual(a, b); d / bd }).fold(0.0, f64::max));
            }
        }
        println!("{k} H₃ samples (Q′ {q}): many-body LOO F rms {:.3}× max {:.3}×, E {:.3}× → {}; pair LOO F max {:.3}×; {answered}/40 answered (worst force ratio {worst:.2})",
                 many.force_rms, many.force_max, many.energy_max, if many.trusted { "trusted" } else { "untrusted" }, pair.force_max);
        assert!(worst <= 3.0, "k = {k}: an answered geometry {worst:.1}× its force bound");
        trace.push((k, many.trusted, answered));
    }
    println!("{trace:?}");
    assert_eq!(
        trace,
        vec![(0, false, 0), (10, false, 0), (20, false, 0), (28, false, 0), (29, true, 40), (30, true, 40), (40, true, 39), (60, true, 40), (80, true, 40), (120, true, 40)],
        "the pinned transition (Q′ = 29)"
    );
}

/// The class verdicts are a property of the geometries, not of the frame: the same store rotated
/// rigidly (positions and forces) gives the same class sizes and verdicts, and the same per-class
/// leave-one-out numbers up to the hyperparameter search — the evidence is flat in λ, and the two
/// searches stop at λ 23.21 vs 23.27 (5 vs 8 rounds; printed): the LOO ratios move by up to
/// 1.4e-4 relative (measured), asserted within the ridge's own relative change |Δλ|/λ = 2.6e-3.
/// At fixed hyperparameters the LOO is rotation-covariant to rounding
/// (`surrogate::loo::tests::loo_is_rotation_covariant`, 1.7e-12).
#[test]
fn class_loo_is_rotation_invariant() {
    let mut s = dimer_model_three_atom_teacher();
    let mut rng = StdRng::seed_from_u64(32);
    for _ in 0..40 {
        let sc = random_h3(&mut rng);
        s.add_sample(pair_morse_sample(&sc), &sc).unwrap();
    }
    s.refit().unwrap();
    // R: 0.9 rad about (1, 2, 2)/3 (Rodrigues)
    let (k, th) = ([1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0], 0.9f64);
    let (c, sn) = (th.cos(), th.sin());
    let r: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| {
        let kx = [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]];
        (if i == j { c } else { 0.0 }) + sn * kx[i][j] + (1.0 - c) * k[i] * k[j]
    }));
    let rot = |v: &[f64; 3]| -> [f64; 3] { std::array::from_fn(|i| (0..3).map(|j| r[i][j] * v[j]).sum()) };
    let mut t = SurrogateKs::new(Box::new(PairMorse(MorseDimer::h2())), FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() }).with_teacher_tol(1e-6);
    t.set_reference_energy(1, MorseDimer::h2().e_atom);
    for smp in s.store.as_ref().unwrap().samples.clone() {
        let rs = Sample { pos: smp.pos.iter().map(rot).collect(), forces: smp.forces.iter().map(rot).collect(), ..smp };
        let mut sc = in_cell_65(dimer(1.4, [1.0, 0.0, 0.0], [0.0; 3], 1, 1));
        sc.nuclei = rs.pos.iter().zip(&rs.z).map(|(p, &z)| Nucleus::new(z, *p)).collect();
        t.add_sample(rs, &sc).unwrap();
    }
    t.refit().unwrap();
    let (ra, rb) = (s.report().unwrap(), t.report().unwrap());
    println!("λ {:e} vs {:e}, σ_e {:e} vs {:e}, σ_f {:e} vs {:e}, κ {:e}, hyper {} {}", ra.lambda, rb.lambda, ra.sigma_e, rb.sigma_e, ra.sigma_f, rb.sigma_f, ra.condition_estimate, ra.hyper_iterations, rb.hyper_iterations);
    for (a, b) in s.class_loo().iter().zip(t.class_loo()) {
        println!("{a:?}\n{b:?}");
        assert_eq!((a.class, a.samples, a.trusted), (b.class, b.samples, b.trusted));
        for (x, y) in [(a.force_rms, b.force_rms), (a.force_max, b.force_max), (a.energy_max, b.energy_max)] {
            let dl = (ra.lambda - rb.lambda).abs() / ra.lambda;
            println!("  relative difference {:.1e} (|Δλ|/λ {dl:.1e})", (x - y).abs() / x);
            assert!((x - y).abs() <= dl * x.abs(), "{x} vs {y} (|Δλ|/λ {dl:.1e})");
        }
    }
}

/// Spec A1: a refit marks the probe's refit phase for its duration and clears it on every exit,
/// a failed refit included; `refits` counts each bracket that closed (the wiring is observed).
#[test]
fn a_refit_reports_its_phase_on_the_probe() {
    let probe = sim::ActivityProbe::new();
    let (s, _) = trained(1.2, 2.2, 0, FallbackPolicy::default());
    let mut s = s.with_probe(probe.clone());
    let before = probe.view();
    assert_eq!((before.refit, before.refits), (None, 0));
    // an empty store: the fit refuses, the phase still closes
    assert!(s.refit().is_err());
    let v = probe.view();
    assert_eq!((v.refit, v.refits), (None, 1), "a failed refit was bracketed and closed");
    for i in 0..60 {
        let sc = h2(1.2 + i as f64 * 0.02);
        s.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
    }
    s.refit().unwrap();
    let v = probe.view();
    assert_eq!((v.refit, v.refits), (None, 2), "the refit was bracketed and closed");
}

// ---- the decision golden (Activity Task 2: the codes, outcomes and events change no decision) ----

/// `MorseDimer` answers only two nuclei; the pass's isolated-atom references need one. This
/// answers the lone atom with E_atom (as the worker tests' `ScriptedMorse` does).
struct AtomMorse(MorseDimer);

impl ForceProvider for AtomMorse {
    fn evaluate(&mut self, scene: &Scene, a: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        if scene.nuclei.len() == 1 {
            let mut stats = EvalStats::default();
            stats.energy.total = self.0.e_atom;
            return Ok(ForceResult { energy: self.0.e_atom, forces: vec![[0.0; 3]], stats, display: None, orbitals: None });
        }
        self.0.evaluate(scene, a)
    }

    fn name(&self) -> &'static str {
        "morse dimer"
    }

    fn reset(&mut self) {}
}

/// The scripted session's teacher: [`AtomMorse`] for one and two nuclei, the pair-additive
/// [`PairMorse`] for three (so a three-atom step reaches the rule that sent it — 4b, 5 or 5d —
/// rather than a teacher error), and an error for more (the teacher-failure path).
struct SessionMorse;

impl ForceProvider for SessionMorse {
    fn evaluate(&mut self, scene: &Scene, a: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        match scene.nuclei.len() {
            3 => PairMorse(MorseDimer::h2()).evaluate(scene, a),
            _ => AtomMorse(MorseDimer::h2()).evaluate(scene, a),
        }
    }

    fn name(&self) -> &'static str {
        "morse dimer"
    }

    fn reset(&mut self) {}
}

/// One line of the decision trace: everything an evaluation decided and reported (f64 `{:?}` is
/// the shortest round-trip form, so the line is bit-exact; the wall time is zeroed) and the
/// provider state it left behind.
fn trace_eval(t: &mut String, label: &str, s: &SurrogateKs, r: &Result<ForceResult, ProviderError>) {
    use std::fmt::Write;
    match r {
        Ok(r) => {
            let mut st = r.stats.clone();
            st.solve_ms = 0;
            let _ = write!(t, "{label}: E {:?} F {:?} display {} {st:?}", r.energy, r.forces, r.display.is_some());
        }
        Err(e) => {
            let _ = write!(t, "{label}: Err {e}");
        }
    }
    let audit = s.last_audit().map(|a| (a.gap, a.verdict, a.face_density, a.unaudited_steps, a.asynchronous));
    let _ = writeln!(
        t,
        " | samples {} vchecks {} epoch {} quarantined {} superseded {} box {} trusted {} pending {} untrusted {:?} stuck {:?} audit {audit:?} failed {:?}",
        s.samples(),
        s.variance_checks(),
        s.epoch(),
        s.quarantined(),
        s.superseded(),
        s.box_affected(),
        s.trusted(),
        s.pending_refit(),
        s.untrusted_reason(),
        s.stuck(),
        s.audit_failed()
    );
}

/// FNV-1a, 64 bit: the trace's digest.
fn fnv1a(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3))
}

/// The digest of [`scripted_session`]'s trace: a change to it is a changed decision, answer,
/// note, sample or RNG draw. Recorded from the code before Activity Task 2 (BASE 9a9a15c; the
/// session as of Task 2 review round 1 was recorded there too, and at 3bb81bd, before the round's
/// fixes).
///
/// **Re-pinning.** The trace prints `{:?}` of `EvalStats`, the audit record and the fit report,
/// so a field added to one of them changes the digest with no decision changed; so does a change
/// to the session itself. Then: export the parent commit (`git archive <parent> | tar -x -C
/// <dir>`), replace the golden section of its `tests.rs` (from `struct AtomMorse` through this
/// test) with this one, build it with its own target dir and run this test there with
/// `GOLDEN_TRACE=<a>`; run it here with `GOLDEN_TRACE=<b>`; `cmp <a> <b>` must report the
/// traces identical except for the new field's own text — only then re-pin the digest.
///
/// **Re-pinned for M7 Task 9** (0x5b8e_8f79_b7d5_94d1 → below; again in fix round 1): `EvalStats` gained `pseudised`,
/// `core_overlaps`, `fractional`, `occupation_changed` and `scf_tol_used` (341 occurrences of their `{:?}` text) and
/// `Provenance::describe` the suffix ", ionic none" (2 occurrences). Removing exactly those strings from the new trace
/// reproduces the old digest 0x5b8e_8f79_b7d5_94d1 — no decision, answer, note, sample or draw moved.
///
/// **Re-pinned for the 2026-10-18 trust fixes** (0x8eb6_57b6_4455_94f9 → below; the 16-line diff of
/// the two traces, checked by hand): at the injected failed audit (+1e-2 Ha, a glitch) the rule-4
/// fallback's answer is now kept instead of thinned (a failed audit is re-judged by samples near it:
/// never thinned), the audit's wording names the new rule ("untrusted until a refit predicts that
/// geometry without the audited sample"), and the refit quarantines the glitch and hands the failure
/// to the fence exactly as before — every later line differs only by that one extra sample (sample
/// counts +1, refit numbers in the ninth digit). No other decision moved.
/// **Re-pinned for round 1 of that review** (→ below; the trace diff checked by hand): MUST-2's
/// copies (`is_label_copy`) — up to step 91 only notes differ; from step 92 the many-body class's
/// short-of-samples fallbacks that repeat a stored H₃ answer within the fit's noise (the session
/// re-evaluates the same geometry) are not kept, one sample fewer from there on (counts and the later
/// refits differ in consequence). No other decision moved.
/// **Re-pinned for M8a Task 2** (0xe1f0_ad02_992d_2504 → below; the two traces exported from the
/// parent b35166e and this tree differ in exactly one line, `audit 2`): the box-affected audit's
/// note gained the suffix "; the next answered step is audited again" (the Task 1 physics
/// review's I4 ruling). No decision, answer, sample or draw moved — the session's audits never
/// gave the pair class a calibrated verdict different from the max rule's, and its auditor runs
/// with audits off, so the re-audit a box-affected answer now asks for is never dispatched there.
/// **Re-pinned for M8a Task 2 fix round 3** (0x8da9_fc7a_ba0d_fcc4 → below; the traces of d19dc83
/// and this tree compared by script with the std readouts masked: 0 lines differ in a decision):
/// rules 6/7 now multiply the epistemic stds by the class's variance inflation λ, so the reported
/// `uncertainty` / `atom_uncertainty` are the inflated numbers (the session's pair class measured
/// λ = 23.6 from its rule-6 fallbacks), and the 10 steps rule 6 already refused say so in their
/// note ("(the model's std × 23.62, its measured calibration for this environment class)"). No
/// source, sample, refit or verdict moved: the session's answered geometries have inflated stds
/// (2.4e-6 Ha/Bohr at step 40) three decades below their bounds.
/// **Re-pinned for M8a Task 4a** (0x5dbf_9e0f_50a2_02b1 → below): `EvalStats` gained `multiplicity`
/// (the analytic teacher reports none: 341 occurrences of ", multiplicity: None" in the 907-line
/// trace, the same in the views trace). Removing exactly those strings from the new traces
/// reproduces the old digest 0x5dbf_9e0f_50a2_02b1 — no decision, answer, note, sample or draw moved
/// (the session attaches no collection).
const SCRIPTED_SESSION_DIGEST: u64 = 0x229d_8fa5_2535_8822;

/// A fixed session through every decision path the providers' tests reach without a thread:
/// rule 1 before a model, a full sampling pass (its seeded perturbations, the burst's
/// Maxwell–Boltzmann draw and CSVR), then 200 scripted steps — model answers, rules 0, 2, 3, 5
/// (with thinning), 6 (an inflated posterior), the in-step audit every 7 answered steps, a relax
/// verification, draft mode, a stuck streak at a geometry the model cannot answer, refits — and
/// the background auditor's judgement of a passing, a failing, a box-affected, a failed and a
/// discarded answer. Returns the trace (one line per evaluation, pass step, note and sample).
fn scripted_session() -> String {
    scripted_session_with(false)
}

/// [`scripted_session`]; with `views`, the training views are built and `would_answer` is asked
/// (at the geometry about to be evaluated) before every evaluate, pass step and refit — activity
/// Task 4's pin on the golden (plan review S1). Neither writes to the trace.
fn scripted_session_with(views: bool) -> String {
    use std::fmt::Write;
    let peek = |s: &mut SurrogateKs, scene: Option<&Scene>| {
        if views {
            std::hint::black_box(s.training_views());
            if let Some(sc) = scene {
                let (pos, z) = pos_z(sc);
                let _ = std::hint::black_box(s.would_answer(&pos, &z, &s.provenance_for(sc)));
            }
        }
    };
    let mut t = String::new();
    let policy = FallbackPolicy { audit_every: 7, ..FallbackPolicy::default() };
    let mut s = SurrogateKs::new(Box::new(SessionMorse), policy).with_teacher_tol(1e-6);
    for (i, r) in [1.5, 1.55, 1.6].into_iter().enumerate() {
        peek(&mut s, Some(&h2(r)));
        let out = s.evaluate(&h2(r), &mut never());
        trace_eval(&mut t, &format!("no model {i}"), &s, &out);
    }
    let mut refs = sim::AtomReferences::default();
    Trainable::start_pass(&mut s, &h2(1.45), &SamplingPlan::default(), &mut refs).unwrap();
    loop {
        peek(&mut s, None);
        let Some(p) = Trainable::pass_step(&mut s, &mut never()).unwrap() else { break };
        let _ = writeln!(t, "pass {} {}/{} skipped {} failed {} E {:?}", p.stage, p.done, p.total, p.skipped, p.failed, p.last_energy);
    }
    for n in Trainable::take_notes(&mut s) {
        let _ = writeln!(t, "note {n}");
    }
    for smp in &s.store().expect("the pass stored samples").samples {
        let _ = writeln!(t, "sample {:?} {:?} {:?} {:?} {:?} {:?}", smp.z, smp.pos, smp.energy, smp.forces, smp.tag, smp.weight);
    }
    let mut rng = StdRng::seed_from_u64(77);
    for k in 0..200 {
        if k == 70 {
            always_unsure(&mut s); // rule 6 until the next refit
        }
        if k == 100 {
            s.set_policy(FallbackPolicy { never_fallback: true, ..policy });
        }
        if k == 115 {
            s.set_policy(policy);
        }
        let scene = match k {
            20 => {
                let mut c = h2(1.6);
                c.charge = 1;
                c
            }
            199 => dimer(1.6, [1.0, 0.0, 0.0], [0.0; 3], 1, 2), // rule 2 (last: no He reference)
            30..=44 => h2(3.0 + 0.001 * (k - 30) as f64),
            50 => {
                s.force_teacher_next();
                h2(1.5)
            }
            60 => h2(7.0), // outside the 33³ cell's margin: rule 0
            61 => h2(6.3), // beyond the cutoff: rule 5a
            62..=64 | 92..=98 => {
                // three atoms (the pair-additive teacher): the envelope or 5d before any is
                // learned, the many-body class (4b) once a few are held
                let j = if k < 90 { k - 62 } else { k - 92 };
                let mut c = h2(1.5 + 0.1 * (j % 3) as f64);
                c.nuclei.push(Nucleus::new(1, [0.0, 1.4 + 0.05 * j as f64, 0.0]));
                c
            }
            65 => {
                // four atoms: the session teacher refuses (a teacher failure)
                let mut c = h2(1.5);
                c.nuclei.push(Nucleus::new(1, [0.0, 1.4, 0.0]));
                c.nuclei.push(Nucleus::new(1, [0.0, -1.4, 0.0]));
                c
            }
            86..=88 => dimer(1.63, [0.0, 1.0, 0.0], [0.1, 0.0, 0.0], 1, 1), // the stray's fence
            102..=106 => h2(4.5),
            125..=190 => h2(4.5 + 0.0005 * (k - 125) as f64),
            _ => dimer(1.3 + 0.9 * rng.gen::<f64>(), unit_vector(&mut rng), [0.0; 3], 1, 1),
        };
        if k == 85 {
            // a glitched label: the refit quarantines it, its fence then takes fresh answers
            let stray = dimer(1.63, [0.0, 1.0, 0.0], [0.1, 0.0, 0.0], 1, 1);
            let mut bad = morse_sample(&MorseDimer::h2(), &stray);
            bad.energy += 0.1;
            s.add_sample(bad, &stray).unwrap();
            peek(&mut s, None);
            let r = s.refit();
            let _ = writeln!(t, "refit {r:?} recovery {:?}", s.recovery_note());
        }
        peek(&mut s, Some(&scene));
        let out = s.evaluate(&scene, &mut never());
        trace_eval(&mut t, &format!("step {k}"), &s, &out);
        // the worker's refit cadence, paused over the stuck streak
        if k % 20 == 19 && !(120..=190).contains(&k) && s.pending_refit() {
            peek(&mut s, None);
            let r = s.refit();
            let _ = writeln!(t, "refit {r:?} recovery {:?}", s.recovery_note());
        }
    }
    for n in Trainable::take_notes(&mut s) {
        let _ = writeln!(t, "note {n}");
    }
    // the background auditor's judgement, driven directly (no thread timing in the trace)
    let (mut s, h, _open, _calls) = with_gated_auditor(0, 0.0);
    let answer = |s: &SurrogateKs, scene: &Scene, prov: Provenance, result: Result<(f64, Vec<[f64; 3]>, Option<f64>, Option<Display>), String>| {
        let generation = s.auditor.as_ref().expect("an auditor").generation.load(Ordering::Relaxed);
        let job = AuditJob { scene: scene.clone(), prov, generation, answered: s.answered, kind: JobKind::Audit, id: 0, view: Default::default(), interrupt: None, core: false, draft_guess: false };
        AuditAnswer { job, result: solved(result), reference: None, ms: 7.0, scf_steps: None }
    };
    let mut teacher = MorseDimer::h2();
    for (i, (r, de, face, fail, charged)) in
        [(1.5, 0.0, None, false, false), (1.7, 1e-2, None, false, false), (1.6, 0.0, Some(1.0), false, false), (1.65, 0.0, None, true, false), (1.55, 0.0, None, false, true), (1.62, 0.0, None, false, false)]
            .into_iter()
            .enumerate()
    {
        let scene = h2(r);
        let mut other = scene.clone();
        if charged {
            other.charge = 1;
        }
        let prov = s.provenance_for(&other);
        let a = teacher.evaluate(&scene, &mut never()).unwrap();
        let result = if fail { Err("the solve diverged".to_string()) } else { Ok((a.energy + de, a.forces, face, None)) };
        peek(&mut s, Some(&scene));
        s.handle_answer(answer(&s, &scene, prov, result));
        peek(&mut s, Some(&h2(1.58)));
        let out = s.evaluate(&h2(1.58), &mut never());
        trace_eval(&mut t, &format!("audit {i}"), &s, &out);
        if s.pending_refit() {
            peek(&mut s, None);
            let r = s.refit();
            let _ = writeln!(t, "refit {r:?}");
        }
    }
    let _ = writeln!(t, "teacher calls {}", calls(&h));
    // no model: the cap on kept answers (one in THIN_EVERY beyond it)
    let mut c = SurrogateKs::new(Box::new(MorseDimer::h2()), policy).with_teacher_tol(1e-6);
    for i in 0..130 {
        peek(&mut c, Some(&h2(1.2 + 0.004 * i as f64)));
        let out = c.evaluate(&h2(1.2 + 0.004 * i as f64), &mut never());
        trace_eval(&mut t, &format!("cap {i}"), &c, &out);
    }
    // box-affected teacher answers: an essential one (no model) kept, a later one not
    let rho = Arc::new(Mutex::new(2.0 * BOX_AFFECTED_FACE_DENSITY));
    let boxed = FaceDensity { inner: Arc::new(Mutex::new(MorseDimer::h2())), rho: rho.clone() };
    let mut b = SurrogateKs::new(Box::new(boxed), policy).with_teacher_tol(1e-6);
    b.set_reference_energy(1, MorseDimer::h2().e_atom);
    peek(&mut b, Some(&h2(1.5)));
    let out = b.evaluate(&h2(1.5), &mut never());
    trace_eval(&mut t, "box essential", &b, &out);
    let mut rng = StdRng::seed_from_u64(4);
    for i in 0..40 {
        let r = 1.1 + 0.9 * i as f64 / 39.0;
        let origin = [rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3)];
        let sc = dimer(r, unit_vector(&mut rng), origin, 1, 1);
        b.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
    }
    peek(&mut b, None);
    let r = b.refit();
    let _ = writeln!(t, "refit {r:?}");
    for (i, f) in [2.0, 0.5, 2.0].into_iter().enumerate() {
        *rho.lock().unwrap() = f * BOX_AFFECTED_FACE_DENSITY;
        peek(&mut b, Some(&h2(2.6)));
        let out = b.evaluate(&h2(2.6), &mut never());
        trace_eval(&mut t, &format!("box {i}"), &b, &out);
    }
    t
}

/// Activity Task 2's guarantee, pinned: the session decides, answers, notes, keeps and draws
/// exactly as the code before the task did (the trace's digest). `GOLDEN_TRACE=<path>` writes
/// the trace for a diff.
#[test]
fn a_scripted_session_decides_exactly_as_before() {
    let t = scripted_session();
    if let Ok(path) = std::env::var("GOLDEN_TRACE") {
        std::fs::write(&path, &t).unwrap();
    }
    let digest = fnv1a(&t);
    println!("scripted session: {} lines, digest {digest:#018x}", t.lines().count());
    assert!(t.contains("Fallback") && t.contains("Surrogate") && t.contains("stuck Some"), "the session reaches its paths");
    assert_eq!(digest, SCRIPTED_SESSION_DIGEST, "a decision changed: diff the trace (GOLDEN_TRACE) against the recorded one");
}

// ---- activity events (spec §4.3 A3) ----

fn drain(s: &mut SurrogateKs) -> Vec<sim::TeachEvent> {
    Trainable::take_teach_events(s)
}

/// Each rule path the policy tests already drive emits one event with the rule's code, the atom
/// it names, the outcome and the origin; the detail is the note the step carries.
#[test]
fn every_teacher_step_emits_one_event_with_its_rule() {
    use sim::activity::{TeachOrigin as O, TeachOutcome as K, TeachRule as R, TrustKey};
    // rule 1: no model
    let (mut s, _) = trained(1.2, 2.2, 0, FallbackPolicy::default());
    let r = s.evaluate(&h2(1.6), &mut never()).unwrap();
    let e = drain(&mut s);
    assert_eq!(e.len(), 1);
    assert_eq!((e[0].origin, e[0].rule, e[0].outcome), (O::Fallback, R::NoModel, K::Kept));
    // every note of the solve, one per line (plan review S12)
    assert!(!e[0].detail.is_empty() && e[0].detail.lines().all(|l| r.stats.notes.iter().any(|n| n == l)), "{:?} vs {:?}", e[0].detail, r.stats.notes);
    assert_eq!(e[0].step, None, "the provider does not stamp: the driver does (Task 1b)");
    // rule 4 on the count
    let (mut s, _) = trained(1.2, 2.2, 5, FallbackPolicy::default());
    s.evaluate(&h2(1.6), &mut never()).unwrap();
    let e = drain(&mut s);
    assert_eq!(e[0].rule, R::Untrusted(TrustKey::Samples));
    assert_eq!(e[0].scf_steps, Some(0), "the analytic teacher's own (true) count, not a stand-in");
    // rule 5 (outside the window): 5c, atom 0 (the first atom 20.6 training spreads out on a
    // body-3 feature: pinned for this fixture, review M2). (In the model's own 33³ cell:
    // `in_big_cell` would change the provenance — rule 3.)
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    assert!(s.trusted());
    s.evaluate(&h2(4.0), &mut never()).unwrap();
    let e = drain(&mut s);
    assert_event_rule(&e[0], R::Spread);
    assert_eq!(e[0].atom, Some(0));
    assert!(e[0].gap.is_some() && e[0].n_atoms == 2, "{:?}", e[0]);
    // rule 6 (inflated posterior): the atom with the largest ratio
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    always_unsure(&mut s);
    s.evaluate(&h2(1.6), &mut never()).unwrap();
    let e = drain(&mut s);
    assert_event_rule(&e[0], R::ForceUncertainty);
    assert!(e[0].atom.is_some());
    // a model answer emits nothing
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    let r = s.evaluate(&h2(1.6), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Surrogate);
    assert!(drain(&mut s).is_empty());
}

/// Rule 3 is logged, not kept, and not counted for the model (spec A7); species is rule 2.
#[test]
fn other_settings_and_new_species_have_their_codes() {
    use sim::activity::{TeachOutcome as K, TeachRule as R};
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    let before = s.solve_counts();
    let mut charged = h2(1.6);
    charged.charge = 1;
    s.evaluate(&charged, &mut never()).unwrap();
    let e = drain(&mut s);
    assert_eq!((e[0].rule, e[0].outcome), (R::OtherSettings, K::NotOffered));
    assert_eq!(s.solve_counts(), before);
    s.evaluate(&dimer(1.6, [1.0, 0.0, 0.0], [0.0; 3], 1, 2), &mut never()).unwrap();
    assert_eq!(drain(&mut s)[0].rule, R::NewSpecies(2));
}

/// Rule 0 (the cell check): one event, nothing solved, nothing counted (plan review N3); the
/// error is the one the run always got.
#[test]
fn a_geometry_outside_the_cell_leaves_a_row_and_no_count() {
    use sim::activity::{TeachOrigin as O, TeachOutcome as K, TeachRule as R};
    let (mut s, h) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    let before = (s.solve_counts(), calls(&h));
    let err = s.evaluate(&h2(7.0), &mut never()).unwrap_err();
    let e = drain(&mut s);
    assert_eq!(e.len(), 1);
    assert_eq!((e[0].origin, e[0].rule, e[0].outcome), (O::Fallback, R::Cell, K::NotSolved));
    assert_eq!(e[0].detail, err.to_string());
    assert_eq!((s.solve_counts(), calls(&h)), before);
}

/// Admission outcomes: a box-affected answer is used and not kept, an essential one (no model
/// yet) is kept as box-affected; thinning says so. (The face density comes from the box test's
/// `FaceDensity` teacher, so the production path needs no test hook.)
#[test]
fn outcomes_name_what_admit_decided() {
    use sim::activity::{TeachOutcome as K, TeachRule as R};
    let rho = Arc::new(Mutex::new(1.0));
    let teacher = FaceDensity { inner: Arc::new(Mutex::new(MorseDimer::h2())), rho: rho.clone() };
    let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_teacher_tol(1e-6);
    s.set_reference_energy(1, MorseDimer::h2().e_atom);
    // no model: an essential answer, kept although box-affected
    s.evaluate(&h2(1.5), &mut never()).unwrap();
    let e = drain(&mut s);
    assert_eq!((e[0].rule, e[0].outcome), (R::NoModel, K::KeptBoxAffected));
    assert!(e[0].box_affected && e[0].face_density == Some(1.0));
    let mut rng = StdRng::seed_from_u64(4);
    for i in 0..40 {
        let r = 1.1 + 0.9 * i as f64 / 39.0;
        let origin = [rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3)];
        let sc = dimer(r, unit_vector(&mut rng), origin, 1, 1);
        s.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
    }
    s.refit().unwrap();
    // a consulted model that fell back: the box answer is used, not kept
    always_unsure(&mut s);
    s.evaluate(&h2(1.6), &mut never()).unwrap();
    let e = drain(&mut s);
    assert_eq!((e[0].rule, e[0].outcome), (R::ForceUncertainty, K::BoxNotKept));
    assert!(e[0].box_affected && e[0].face_density == Some(1.0));
    // thinning (the thinning test's setup: untrusted by residuals): the first answer kept, its
    // repetitions copies (round 1, MUST-2) — thinned; a far one kept
    let (mut s, _) = trained(1.2, 2.2, 40, FallbackPolicy::default());
    s.set_policy(FallbackPolicy { energy_std_per_atom: 1e-12, ..FallbackPolicy::default() });
    for _ in 0..11 {
        s.evaluate(&h2(1.6), &mut never()).unwrap();
    }
    s.evaluate(&h2(4.0), &mut never()).unwrap();
    let outcomes: Vec<K> = drain(&mut s).iter().map(|e| e.outcome).collect();
    let mut want = vec![K::Kept];
    want.extend([K::Thinned; 10]);
    want.push(K::Kept);
    assert_eq!(outcomes, want);
}

/// One event per teacher solve: over a scripted run every teacher call (and nothing else that is
/// not a rule-0 refusal) leaves exactly one solve event, with no stamp from the provider.
#[test]
fn one_event_per_teacher_solve_over_a_session() {
    let (mut s, h) = trained(1.2, 2.2, 60, FallbackPolicy { audit_every: 5, ..FallbackPolicy::default() });
    let mut rng = StdRng::seed_from_u64(12);
    let c0 = calls(&h);
    let mut fallbacks = 0;
    let mut events = Vec::new();
    for k in 0..120 {
        let scene = match k {
            10 => {
                let mut c = h2(1.6);
                c.charge = 1;
                c
            }
            20..=30 => h2(3.2),
            40 => h2(7.0),
            _ => dimer(1.3 + 0.8 * rng.gen::<f64>(), unit_vector(&mut rng), [0.0; 3], 1, 1),
        };
        if k == 70 {
            always_unsure(&mut s);
        }
        if let Ok(r) = s.evaluate(&scene, &mut never()) {
            fallbacks += usize::from(r.stats.source == Source::Fallback);
        }
        if k % 25 == 24 && s.pending_refit() {
            s.refit().unwrap();
        }
        events.extend(drain(&mut s));
    }
    let solves = events.iter().filter(|e| e.is_solve()).count();
    assert_eq!(solves, calls(&h) - c0, "one event per teacher call");
    assert_eq!(solves, fallbacks, "and every teacher answer is a fallback step");
    assert_eq!(events.iter().filter(|e| !e.is_solve()).count(), 1, "the one rule-0 refusal");
    assert!(events.iter().all(|e| e.step.is_none()));
    assert!(events.iter().any(|e| e.origin == sim::activity::TeachOrigin::Audit), "audits every 5 answered steps");
    let c = s.solve_counts();
    assert_eq!(c.total() as usize, events.iter().filter(|e| e.counts_for_model()).count());
    assert_eq!(c.total() as usize, solves - 1, "the rule-3 solve is not counted");
}

/// Review Focus 1: kept counts equal the samples added; origins add up.
#[test]
fn solve_counts_add_up_over_a_pass_fallbacks_and_audits() {
    let teacher = AtomMorse(MorseDimer::h2());
    let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy { audit_every: 5, ..FallbackPolicy::default() }).with_teacher_tol(1e-6);
    let scene = h2(1.45);
    let mut refs = sim::AtomReferences::default();
    Trainable::start_pass(&mut s, &scene, &sim::SamplingPlan::default(), &mut refs).unwrap();
    while Trainable::pass_step(&mut s, &mut never()).unwrap().is_some() {}
    let pass_events = drain(&mut s);
    let c = s.solve_counts();
    assert_eq!(c.pass as usize, pass_events.iter().filter(|e| e.counts_for_model()).count());
    assert_eq!(c.kept as usize, s.samples(), "every kept pass answer is a sample");
    assert!(pass_events.iter().any(|e| matches!(e.rule, sim::activity::TeachRule::Pass(sim::activity::PassJob::Reference { z: 1, .. }))));
    assert!(pass_events.iter().filter_map(|e| e.burst_t).all(|t| t.is_finite() && t >= 0.0));
    assert!(s.trusted(), "the default plan trusts the Morse H₂ (else take the plan the worker's sizing test uses)");
    for i in 0..40 {
        s.evaluate(&h2(1.40 + 0.002 * i as f64), &mut never()).unwrap();
    }
    let c2 = s.solve_counts();
    assert!(c2.audit > 0, "audit_every 5 over 40 answered steps");
    assert_eq!(c2.kept as usize, s.samples());
    assert_eq!(c2.total(), c2.kept + c2.not_kept());
}

/// Plan review S6: a hot-burst event's temperature is the solved geometry's, before the CSVR
/// thermostat rescales its velocities — the start geometry's is the Maxwell–Boltzmann draw's.
#[test]
fn burst_events_carry_the_pre_thermostat_temperature() {
    use sim::activity::{PassJob, TeachRule as R};
    let plan = SamplingPlan { perturbations: 0, scan_points_per_pair: 0, hot_burst_steps: 5, ..SamplingPlan::default() };
    let mut s = SurrogateKs::new(Box::new(AtomMorse(MorseDimer::h2())), FallbackPolicy::default()).with_teacher_tol(1e-6);
    let mut refs = sim::AtomReferences::default();
    let scene = h2(1.45);
    Trainable::start_pass(&mut s, &scene, &plan, &mut refs).unwrap();
    while Trainable::pass_step(&mut s, &mut never()).unwrap().is_some() {}
    let burst: Vec<f64> = drain(&mut s).iter().filter(|e| e.rule == R::Pass(PassJob::HotBurst)).map(|e| e.burst_t.expect("a burst temperature")).collect();
    assert_eq!(burst.len(), 5);
    // the same trajectory by hand: Verlet on the teacher, T read between the kick and CSVR
    let dt = units::fs_to_au_time(crate::sampling::HOT_BURST_DT_FS);
    let mut csvr = sim::thermostat::Csvr::new(plan.hot_burst_t, crate::sampling::HOT_BURST_TAU_FS, plan.seed ^ 0x9E37_79B9_7F4A_7C15);
    let mut work = scene.clone();
    csvr.maxwell_boltzmann(&mut work);
    let mut teacher = MorseDimer::h2();
    let mut f = teacher.evaluate(&work, &mut never()).unwrap().forces;
    let mut want = vec![work.temperature()];
    for _ in 1..5 {
        sim::integrator::Integrator::half_kick(&mut work, &f, dt);
        sim::integrator::Integrator::drift(&mut work, dt);
        f = teacher.evaluate(&work, &mut never()).unwrap().forces;
        sim::integrator::Integrator::half_kick(&mut work, &f, dt);
        want.push(work.temperature());
        csvr.apply(&mut work, dt);
    }
    assert_eq!(burst, want);
}

/// Review Focus 2: the queue keeps the newest 2 048 events and counts the dropped ones; the
/// counts are not capped.
#[test]
fn the_event_queue_drops_the_oldest_and_counts_them() {
    let (mut s, _) = trained(1.2, 2.2, 0, FallbackPolicy::default());
    for i in 0..(EVENT_QUEUE + 10) {
        let mut e = sim::TeachEvent::new(sim::activity::TeachOrigin::Fallback, sim::activity::TeachRule::NoModel, sim::activity::TeachOutcome::Thinned);
        e.n_atoms = i;
        s.push_event(e);
    }
    assert_eq!(s.events_dropped(), 10);
    let e = drain(&mut s);
    assert_eq!(e.len(), EVENT_QUEUE);
    assert_eq!(e[0].n_atoms, 10, "the oldest went first");
    assert_eq!(s.solve_counts().fallback, (EVENT_QUEUE + 10) as u64);
}

/// Spec A2: the counts survive a sidecar round trip; a file without the key reads as
/// "unrecorded before", and that flag survives the next save and load (plan review M4: the
/// session autosave round-trips every model through the file); a forget resets them.
#[test]
fn solve_counts_survive_the_sidecar_and_forget_resets_them() {
    let dir = std::env::temp_dir().join(format!("as-activity-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("counts.model");
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    always_unsure(&mut s);
    for _ in 0..3 {
        s.evaluate(&h2(1.6), &mut never()).unwrap();
    }
    let c = s.solve_counts();
    assert_eq!(c.fallback, 3);
    s.save_sidecar(&path).unwrap();
    let (mut back, _) = trained(1.2, 2.2, 0, FallbackPolicy::default());
    back.install_sidecar(surrogate::sidecar::read(&path).unwrap()).unwrap();
    assert_eq!(back.solve_counts(), c);
    Trainable::forget(&mut back);
    assert_eq!(back.solve_counts(), sim::SolveCounts::default());
    // a file written without the key (the pre-activity writer)
    let sc = surrogate::sidecar::read(&path).unwrap();
    surrogate::sidecar::write_full(&path, &sc.model, &sc.accumulator, sc.report.as_ref(), &sc.samples, sc.reference_rule.as_deref(), &sc.quarantine, &sc.face_density, None).unwrap();
    back.install_sidecar(surrogate::sidecar::read(&path).unwrap()).unwrap();
    assert!(back.solve_counts().unrecorded_before && back.solve_counts().total() == 0);
    // load → one more solve → save (the autosave) → load: still "earlier solves unrecorded"
    always_unsure(&mut back);
    back.evaluate(&h2(1.6), &mut never()).unwrap();
    back.save_sidecar(&path).unwrap();
    let (mut again, _) = trained(1.2, 2.2, 0, FallbackPolicy::default());
    again.install_sidecar(surrogate::sidecar::read(&path).unwrap()).unwrap();
    assert_eq!(again.solve_counts(), back.solve_counts());
    assert!(again.solve_counts().unrecorded_before && again.solve_counts().fallback == 1);
    // an unreadable record never refuses the model: it reads as unrecorded, with a note
    let mut sc = surrogate::sidecar::read(&path).unwrap();
    sc.teacher_solves = None;
    sc.teacher_solves_note = Some("the record is malformed".into());
    again.install_sidecar(sc).unwrap();
    assert!(again.solve_counts().unrecorded_before && again.solve_counts().total() == 0);
    assert!(Trainable::take_notes(&mut again).iter().any(|n| n.contains("the record is malformed")));
    // a refused file leaves the counts alone (plan review N11)
    let before = again.solve_counts();
    let mut sc = surrogate::sidecar::read(&path).unwrap();
    sc.report = None;
    sc.teacher_solves = Some(surrogate::sidecar::TeacherSolves { pass: 9, fallback: 9, audit: 9, verification: 9, kept: 9, complete: true });
    assert!(again.install_sidecar(sc).is_err());
    assert_eq!(again.solve_counts(), before);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Plan review M6: driven by `sim::Dynamics`, each fallback's event carries the step whose
/// evaluation produced it (stamped right after that evaluate, Task 1b), so one tick of several
/// steps gives distinct, increasing stamps; the provider itself stamps nothing.
#[test]
fn fallback_events_carry_the_step_of_their_evaluation() {
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    always_unsure(&mut s);
    let mut scene = h2(1.6);
    scene.nuclei[1].vel = [2e-4, 0.0, 0.0];
    let (p, _sh) = driven(s);
    let mut d = Dynamics::new(scene, p, 0.25, 1);
    let first = d.take_teach_events();
    assert!(first.iter().all(|e| e.step == Some(0)), "the evaluation at construction is step 0: {first:?}");
    d.apply(DynCommand::Step(5));
    d.tick(f64::INFINITY, &mut never());
    let steps: Vec<u64> = d
        .take_teach_events()
        .iter()
        .filter(|e| e.origin == sim::activity::TeachOrigin::Fallback)
        .map(|e| e.step.expect("stamped by the driver"))
        .collect();
    // a refit inside the tick may end the fallbacks early; the stamps stay distinct
    assert!(steps.len() >= 2, "{steps:?}");
    assert!(steps.windows(2).all(|w| w[0] < w[1]) && steps.iter().all(|s| (1..=5).contains(s)), "{steps:?}");
}

/// A teacher's (energy, forces, face density, display) as the auditor's answer carries it (no
/// orbital payload: these audits run with the view off).
fn solved(r: Result<(f64, Vec<[f64; 3]>, Option<f64>, Option<Display>), String>) -> Result<Solved, String> {
    r.map(|(energy, forces, face, display)| Solved { energy, forces, face, multiplicity: None, display, orbitals: None, unfit: None })
}

/// A background audit job for `s` at `scene` under `prov` (the auditor's own bookkeeping
/// fields; the tests below hand its answer to `handle_answer` directly).
fn audit_answer(s: &SurrogateKs, scene: &Scene, prov: Provenance, result: Result<(f64, Vec<[f64; 3]>, Option<f64>, Option<Display>), String>) -> AuditAnswer {
    let generation = s.auditor.as_ref().expect("an auditor").generation();
    let job = AuditJob { scene: scene.clone(), prov, generation, answered: s.answered, kind: JobKind::Audit, id: 0, view: Default::default(), interrupt: None, core: false, draft_guess: false };
    let scf_steps = result.is_ok().then_some(5);
    AuditAnswer { job, result: solved(result), reference: None, ms: 7.0, scf_steps }
}

/// Plan review S11: the background auditor's failed solve and its discarded answer are solves
/// spent on the model, so they leave an event (origin Audit) and are counted, never kept; a
/// judged one carries its verdict, gap and lag.
#[test]
fn a_failed_or_discarded_background_audit_emits_an_event() {
    use sim::activity::{TeachOrigin as O, TeachOutcome as K, TeachRule as R};
    let (mut s, _, _, _) = with_gated_auditor(5, 0.0);
    let scene = h2(1.5);
    let prov = s.provenance_for(&scene);
    let before = s.solve_counts();
    s.handle_answer(audit_answer(&s, &scene, prov.clone(), Err("the solve diverged".into())));
    let e = drain(&mut s);
    assert_eq!(e.len(), 1);
    assert_eq!((e[0].origin, e[0].rule, e[0].outcome), (O::Audit, R::Audit, K::Failed));
    assert!(e[0].detail.contains("the solve diverged"));
    assert_eq!((e[0].scf_steps, e[0].solve_ms), (None, 7.0), "a failed solve reports no SCF steps, never a stand-in 0");
    // an answer for other settings than the model's: discarded, not judged
    let mut charged = scene.clone();
    charged.charge = 1;
    let other = s.provenance_for(&charged);
    let teacher = MorseDimer::h2().evaluate(&scene, &mut never()).unwrap();
    s.handle_answer(audit_answer(&s, &scene, other, Ok((teacher.energy, teacher.forces.clone(), None, None))));
    let e = drain(&mut s);
    assert_eq!((e[0].origin, e[0].outcome), (O::Audit, K::Discarded));
    let c = s.solve_counts();
    assert_eq!((c.audit - before.audit, c.kept - before.kept), (2, 0));
    // a judged answer: the verdict, the gap and the lag
    s.handle_answer(audit_answer(&s, &scene, prov, Ok((teacher.energy, teacher.forces, None, None))));
    let e = drain(&mut s);
    assert_eq!(e.len(), 1);
    assert_eq!((e[0].origin, e[0].rule, e[0].audit), (O::Audit, R::Audit, Some(AuditVerdict::Passed)));
    assert!(e[0].gap.is_some_and(|g| g.abs() < 1e-3) && e[0].audit_lag == Some(0) && e[0].solve_ms == 7.0, "{:?}", e[0]);
    assert_eq!(e[0].scf_steps, Some(5), "the auditor's own SCF count is carried (review I1)");
}

/// Review M3: a fallback row's detail is its own solve's notes; a background audit judged in the
/// same `evaluate` keeps its notes on its own row (they still reach the step's `stats.notes`).
#[test]
fn a_fallback_detail_does_not_carry_an_audit_verdict() {
    let (mut s, _, _, _) = with_gated_auditor(0, 0.0);
    let scene = h2(1.5);
    let prov = s.provenance_for(&scene);
    let teacher = MorseDimer::h2().evaluate(&scene, &mut never()).unwrap();
    s.held_audit = Some(audit_answer(&s, &scene, prov, Ok((teacher.energy, teacher.forces, None, None))));
    always_unsure(&mut s);
    let r = s.evaluate(&h2(1.6), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback);
    assert!(r.stats.notes.iter().any(|n| n.contains("surrogate audit passed")), "{:?}", r.stats.notes);
    let e = drain(&mut s);
    assert_eq!(e.len(), 2, "{e:?}");
    assert_eq!((e[0].origin, e[1].origin), (sim::activity::TeachOrigin::Audit, sim::activity::TeachOrigin::Fallback));
    assert!(e[0].detail.contains("surrogate audit passed"));
    assert!(!e[1].detail.contains("surrogate audit"), "{:?}", e[1].detail);
    assert!(e[1].detail.lines().next().is_some_and(|l| l.contains("epistemic force std")), "{:?}", e[1].detail);
}

/// The code a refusal text belongs to (Task 2 review M1: the codes of the paths the golden
/// session does not reach are pinned by the tests that drive them): `None` for a text that is
/// not a refusal of rules 4b–7.
fn rule_of_text(t: &str) -> Option<sim::activity::TeachRule> {
    use sim::activity::{EnvClass, TeachRule as R};
    let table: [(&str, R); 12] = [
        ("the model's answer is not finite", R::NonFiniteAnswer),
        ("isolated atom — atom", R::IsolatedAtom),
        ("below the smallest the fit has seen", R::Coordination),
        ("training spreads outside the descriptor envelope", R::Spread),
        ("MaxVol did not converge", R::NoGrade),
        ("(grade γ = ∞)", R::UnspannedAtom),
        ("extrapolating: grade γ", R::Grade),
        ("farther into directions no training row spans", R::NewEnvironment),
        ("epistemic force std", R::ForceUncertainty),
        ("epistemic energy std", R::EnergyUncertainty),
        ("learned well enough yet (pair", R::Class(EnvClass::Pair)),
        ("learned well enough yet (many-body", R::Class(EnvClass::ManyBody)),
    ];
    table.iter().find(|(p, _)| t.contains(p)).map(|(_, r)| *r)
}

/// A refusal carries the code of the rule its text is, and names the atom its text names.
fn assert_refusal(r: &Refusal) {
    assert_eq!(Some(r.rule), rule_of_text(&r.text), "{}", r.text);
    if let Some(a) = r.atom {
        assert!(r.text.contains(&format!("atom {a}")), "atom {a} in {}", r.text);
    }
}

/// [`assert_refusal`] on an event: its rule is its detail's first line's, and that is `want`.
fn assert_event_rule(e: &sim::TeachEvent, want: sim::activity::TeachRule) {
    let first = e.detail.lines().next().unwrap_or("");
    assert_eq!((e.rule, rule_of_text(first)), (want, Some(want)), "{first}");
    if let Some(a) = e.atom {
        assert!(first.contains(&format!("atom {a}")), "atom {a} in {first}");
    }
}

/// Rules 6 and 7 at the verdict itself (rule 7 needs an energy std beyond the tier with every
/// force std inside it, which no fixture reaches through `evaluate`: the posterior inflation
/// trips rule 6 first).
#[test]
fn the_variance_verdict_carries_rule_6_or_7() {
    use sim::activity::TeachRule as R;
    let (s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    let model = s.model().unwrap().clone();
    let pos: Vec<[f64; 3]> = h2(1.6).nuclei.iter().map(|n| n.pos).collect();
    let pred = model.predict(&pos, &[1, 1], true).unwrap();
    let v = s.variance_check(&model, &pred, 1.0, &[0.0, 0.0], &pos);
    let r = v.reason.expect("an energy std of 1 Ha refuses");
    assert_eq!((r.rule, r.atom), (R::EnergyUncertainty, None));
    assert_refusal(&r);
    let v = s.variance_check(&model, &pred, 0.0, &[0.0, 1.0], &pos);
    let r = v.reason.expect("a force std of 1 Ha/Bohr on atom 1 refuses");
    assert_eq!((r.rule, r.atom), (R::ForceUncertainty, Some(1)));
    assert_refusal(&r);
}

// ---- Activity Task 3: the trust checklist, the class verdict, the stuck info ----

/// Review Focus 4: on every branch of the trust rule the first unmet check is the condition the
/// refusal names, and a trusted model meets every check.
#[test]
fn the_first_unmet_trust_check_is_the_refusal() {
    use sim::activity::TrustKey;
    let first_unmet = |s: &SurrogateKs| s.trust_checks().into_iter().find(|c| !c.met).map(|c| c.key);
    let key = |s: &SurrogateKs| s.untrusted_check().map(|c| c.0);
    // n < Q′
    let (s, _) = trained(1.2, 2.2, 5, FallbackPolicy::default());
    assert_eq!(first_unmet(&s), Some(TrustKey::Samples));
    assert_eq!(first_unmet(&s), key(&s));
    // residual dofs
    let (s, _) = trained(1.2, 2.2, 30, FallbackPolicy::default());
    assert_eq!(first_unmet(&s), key(&s));
    assert!(matches!(key(&s), Some(TrustKey::ResidualDofE | TrustKey::ResidualDofF)), "{:?}", s.untrusted_reason());
    // training residuals (the misspecified fit of `misspecified_fit_is_untrusted_by_its_training_residuals`)
    let (s, _) = trained_oscillatory(1.2, 2.2, 120, 0.02, 6.0);
    assert_eq!(first_unmet(&s), key(&s));
    assert!(matches!(key(&s), Some(TrustKey::EnergyResidual | TrustKey::ForceResidual)), "{:?}", s.untrusted_reason());
    // a failed audit, on the energy and on the forces
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    s.audit_failed = Some(0.5);
    s.audit_atoms = 2;
    assert_eq!(first_unmet(&s), Some(TrustKey::Audit));
    assert_eq!(first_unmet(&s), key(&s));
    s.audit_force = Some((5e-3, 2e-3));
    assert_eq!(first_unmet(&s), Some(TrustKey::AuditForces));
    assert_eq!(first_unmet(&s), key(&s));
    // trusted
    let (s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    assert!(s.trusted() && s.trust_checks().iter().all(|c| c.met));
    // values are the report's and the policy's
    let r = s.report().unwrap().clone();
    let c = s.trust_checks();
    assert_eq!((c[0].value, c[0].limit), (r.n_samples as f64, r.q_prime as f64));
    assert_eq!((c[1].value, c[1].limit), (r.residual_dof_e, MIN_RESIDUAL_DOF));
}

/// Plan review M2: one predicate per condition serves the checklist and the refusal, so they
/// agree at equality and on NaN by construction. At the limit a condition is met; a NaN fails it
/// (Task 3b, the physics review's ruling (b): a number the rule cannot compare is never a pass).
#[test]
fn the_trust_checks_and_the_refusal_agree_at_equality_and_nan() {
    use sim::activity::TrustKey;
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    assert!(s.trusted());
    let tier = s.policy.energy_std_per_atom;
    s.report.as_mut().unwrap().residual_dof_e = MIN_RESIDUAL_DOF;
    s.training_residuals.as_mut().unwrap().energy_per_atom = tier;
    let checks = s.trust_checks();
    let met = |k: TrustKey| checks.iter().find(|c| c.key == k).unwrap().met;
    assert!(met(TrustKey::ResidualDofE) && met(TrustKey::EnergyResidual), "at the limit: {checks:?}");
    assert_eq!(s.untrusted_check(), None);
    assert!(s.trusted());
    // NaN: unmet, and the refusal names the same key
    s.report.as_mut().unwrap().residual_dof_e = f64::NAN;
    assert_eq!(s.trust_checks().into_iter().find(|c| !c.met).map(|c| c.key), Some(TrustKey::ResidualDofE));
    assert_eq!(s.untrusted_check().map(|c| c.0), Some(TrustKey::ResidualDofE));
    assert!(!s.trusted());
    // just past a limit: unmet, and the refusal names the same key
    s.report.as_mut().unwrap().residual_dof_e = MIN_RESIDUAL_DOF - 1e-9;
    assert_eq!(s.trust_checks().into_iter().find(|c| !c.met).map(|c| c.key), Some(TrustKey::ResidualDofE));
    assert_eq!(s.untrusted_check().map(|c| c.0), Some(TrustKey::ResidualDofE));
}

/// The refusal strings stay byte-identical (the Log, the app's `COUNT_REFUSALS` until Task 8,
/// and every test that matches a note read them).
#[test]
fn the_refusal_strings_are_unchanged() {
    let (s, _) = trained(1.2, 2.2, 5, FallbackPolicy::default());
    let r = s.report().unwrap().clone();
    assert_eq!(s.untrusted_reason().unwrap(), format!("n_samples {} < q_prime {}", r.n_samples, r.q_prime));
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    s.audit_failed = Some(0.5);
    s.audit_atoms = 2;
    assert_eq!(s.untrusted_reason().unwrap(), "audit found teacher − model +5.0e-1 Ha; untrusted until a refit predicts that geometry without the audited sample");
    s.audit_force = Some((5e-3, 2e-3));
    assert_eq!(s.untrusted_reason().unwrap(), "audit found forces off by 5.0e-3 > 2.0e-3 Ha/Bohr (teacher − model +5.0e-1 Ha); untrusted until a refit predicts that geometry without the audited sample");
}

/// Plan review M1: the audit check compares the failed audit's |gap|/N with the per-atom tier,
/// or, when the forces failed it, the worst atom's residual with its bound. With no failed
/// audit since the refit it is met with value 0, whether or not an audit ever ran: the UI words
/// that case from `last_audit` (Task 8), never as "passed".
#[test]
fn the_audit_check_is_per_atom_and_names_forces() {
    use sim::activity::TrustKey;
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    let audit = |s: &SurrogateKs| *s.trust_checks().last().unwrap();
    assert!(s.last_audit().is_none());
    let c = audit(&s);
    assert_eq!((c.key, c.met, c.value, c.limit), (TrustKey::Audit, true, 0.0, s.policy.energy_std_per_atom));
    s.audit_failed = Some(-3e-3);
    s.audit_atoms = 2;
    let c = audit(&s);
    assert_eq!((c.key, c.met), (TrustKey::Audit, false));
    assert!((c.value - 1.5e-3).abs() < 1e-18 && c.limit == s.policy.energy_std_per_atom, "per atom: {c:?}");
    s.audit_force = Some((4e-3, 2e-3));
    let c = audit(&s);
    assert_eq!((c.key, c.met, c.value, c.limit), (TrustKey::AuditForces, false, 4e-3, 2e-3));
    s.refit().unwrap();
    let c = audit(&s);
    assert_eq!((c.key, c.met, c.value), (TrustKey::Audit, true, 0.0), "a refit clears the failure");
}

/// Plan review M3: one function decides a class's trust and names its failing condition; the
/// wording reads that, so an energy-only failure is never worded as a force error.
#[test]
fn the_class_verdict_names_its_failing_condition() {
    use super::trust::class_refusal;
    use sim::ClassKey;
    assert_eq!(class_refusal(10, 16, 0.1, 0.2, 0.3), Some((ClassKey::Samples, 10.0, 16.0)));
    assert_eq!(class_refusal(40, 16, 0.9, 2.5, 0.3), Some((ClassKey::Force, 2.5, 1.0)));
    assert_eq!(class_refusal(40, 16, 0.2, 0.4, 1100.0), Some((ClassKey::Energy, 1100.0, 1.0)));
    assert_eq!(class_refusal(40, 16, 0.2, 0.4, 0.3), None);
    assert!(class_refusal(40, 16, f64::NAN, 0.4, 0.3).is_some(), "NaN fails, as `trusted`'s `<=` did");
    assert!(class_refusal(40, 16, f64::NAN, 0.4, 0.3).unwrap().1.is_nan(), "a failing check never shows a value inside its limit");
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    assert!(s.class_loo().iter().all(|l| l.trusted == l.refusal.is_none()));
    s.class_loo[0] = sim::ClassLoo { class: "pair", samples: 120, force_rms: 0.1, force_max: 0.4, energy_rms: 110.0, energy_max: 1100.0, trusted: false, refusal: Some((ClassKey::Energy, 1100.0, 1.0)), calibration: 1.0, measured: None, measured_energy: None, worst_atom: None, evidence: 0, trusted_since: None, inflation: 1.0 };
    let (r, short) = s.class_untrusted(&[1, 1]).unwrap();
    assert!(!short && r.text.contains("LOO energy error 1100.0× the tier"), "{}", r.text);
    assert_eq!(r.rule, sim::activity::TeachRule::Class(sim::activity::EnvClass::Pair));
}

/// Spec A5: `stuck_info` is `stuck` in structured form: same streak, rule of the last fallback,
/// its atom, the box streak.
#[test]
fn stuck_info_carries_the_rule_and_atom_of_the_streak() {
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    assert!(s.stuck_info().is_none());
    always_unsure(&mut s);
    for _ in 0..STUCK_STEPS {
        s.evaluate(&h2(1.6), &mut never()).unwrap();
    }
    let i = s.stuck_info().expect("stuck after STUCK_STEPS fallbacks");
    assert!(s.stuck().is_some());
    assert_eq!(i.rule, Some(sim::activity::TeachRule::ForceUncertainty));
    assert!(i.atom.is_some() && i.steps >= STUCK_STEPS && !i.box_streak);
    // the structured form and the string read the same state
    assert_eq!(i.steps, s.streak);
    assert_eq!(i.gap, s.streak_gap);
    assert!(s.stuck().unwrap().contains(&format!("the last {} steps", i.steps)));
    // `status()` carries both new readouts
    let st = Trainable::status(&s, &h2(1.6), &[]);
    assert_eq!(st.stuck_info, Some(i));
    assert_eq!(st.trust_checks, s.trust_checks());
}

/// The trust rule as `untrusted_check` wrote it before Activity Task 3 (e53854e, copied
/// verbatim), with one change, Task 3b's NaN ruling: each comparison is written so a NaN fails
/// (`!(x >= limit)` for `x < limit`, `!(x <= limit)` for `x > limit`). The oracle the shared-
/// predicate rewrite must reproduce — key, string and verdict — on every state
/// (`the_trust_checks_reproduce_the_pre_task3_rule_on_a_sweep`); on every non-NaN state it is the
/// verbatim pre-Task-3 rule.
fn pre_task3_untrusted_check(s: &SurrogateKs) -> Option<(sim::activity::TrustKey, String)> {
    use sim::activity::TrustKey;
    let r = s.report.as_ref()?;
    if r.n_samples < r.q_prime {
        return Some((TrustKey::Samples, format!("n_samples {} < q_prime {}", r.n_samples, r.q_prime)));
    }
    if !(r.residual_dof_e >= MIN_RESIDUAL_DOF) {
        return Some((TrustKey::ResidualDofE, format!("residual_dof_e {:.1} < {MIN_RESIDUAL_DOF:.0}", r.residual_dof_e)));
    }
    if !(r.residual_dof_f >= MIN_RESIDUAL_DOF) {
        return Some((TrustKey::ResidualDofF, format!("residual_dof_f {:.1} < {MIN_RESIDUAL_DOF:.0}", r.residual_dof_f)));
    }
    if let Some(t) = &s.training_residuals {
        if !(t.energy_per_atom <= s.policy.energy_std_per_atom) {
            return Some((
                TrustKey::EnergyResidual,
                format!("max training energy residual {:.1e} > {:.1e} Ha/atom", t.energy_per_atom, s.policy.energy_std_per_atom),
            ));
        }
        if !(t.force <= t.force_bound) {
            return Some((TrustKey::ForceResidual, format!("max training force residual {:.1e} > {:.1e} Ha/Bohr", t.force, t.force_bound)));
        }
    }
    if !(r.sigma_f <= s.policy.force_std_abs) {
        return Some((
            TrustKey::ForceNoise,
            format!(
                "force noise σ_f {:.1e} > {:.1e} Ha/Bohr — the fit's own force labels scatter beyond the force tier",
                r.sigma_f, s.policy.force_std_abs
            ),
        ));
    }
    if let Some(gap) = s.audit_failed {
        return Some(match s.audit_force {
            Some((d, b)) => (TrustKey::AuditForces, format!("audit found forces off by {d:.1e} > {b:.1e} Ha/Bohr (teacher − model {gap:+.1e} Ha); untrusted until a refit predicts that geometry without the audited sample")),
            None => (TrustKey::Audit, format!("audit found teacher − model {gap:+.1e} Ha; untrusted until a refit predicts that geometry without the audited sample")),
        });
    }
    None
}

/// A value for a condition with limit `limit`: the limit itself, its neighbours one ulp away,
/// zero, the non-finite values, or a random multiple — the boundaries where a re-derived copy
/// of a condition would drift.
fn around(rng: &mut StdRng, limit: f64) -> f64 {
    match rng.gen_range(0..10) {
        0 => limit,
        1 => limit.next_up(),
        2 => limit.next_down(),
        3 => 0.0,
        4 => f64::NAN,
        5 => f64::INFINITY,
        6 => f64::NEG_INFINITY,
        _ => limit * rng.gen_range(0.0..2.0),
    }
}

/// Plan review M2 and "no decision changed": over many model states — every untrusted reason,
/// trusted ones, the limits to the ulp, NaN and ±∞ — the checklist's first unmet entry, the
/// refusal (key and string), `trusted` and the predicates' other callers (`residuals_fail`,
/// `noise_fails`, `trust_needs_samples`) equal the pre-Task-3 rule's (with Task 3b's NaN ruling;
/// a NaN anywhere the rule reads refuses trust). Every key is reached as
/// the refusal, and trusted states occur, or the sweep fails as vacuous.
#[test]
fn the_trust_checks_reproduce_the_pre_task3_rule_on_a_sweep() {
    use sim::activity::TrustKey;
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    let (report, residuals) = (s.report.clone().unwrap(), s.training_residuals.unwrap());
    let q = report.q_prime;
    let mut rng = StdRng::seed_from_u64(0x7a5c3);
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    for i in 0..40_000 {
        // a fresh state each round: report or none, residuals or none, an audit or none
        s.report = (rng.gen_range(0..40) != 0).then(|| {
            let mut r = report.clone();
            r.n_samples = [q.saturating_sub(1), q, q + 1, 0, q + 100][rng.gen_range(0..5)];
            r.residual_dof_e = if rng.gen_bool(0.7) { report.residual_dof_e } else { around(&mut rng, MIN_RESIDUAL_DOF) };
            r.residual_dof_f = if rng.gen_bool(0.7) { report.residual_dof_f } else { around(&mut rng, MIN_RESIDUAL_DOF) };
            r.sigma_f = if rng.gen_bool(0.7) { report.sigma_f } else { around(&mut rng, s.policy.force_std_abs) };
            r
        });
        s.training_residuals = (rng.gen_range(0..10) != 0).then(|| {
            let mut t = residuals;
            if rng.gen_bool(0.4) {
                t.energy_per_atom = around(&mut rng, s.policy.energy_std_per_atom);
            }
            if rng.gen_bool(0.4) {
                t.force_bound = [2e-3, 1e-2, f64::INFINITY, f64::NAN][rng.gen_range(0..4)];
                t.force = around(&mut rng, t.force_bound);
            }
            t
        });
        s.audit_failed = rng.gen_bool(0.3).then(|| around(&mut rng, s.policy.energy_std_per_atom) * if rng.gen_bool(0.5) { -1.0 } else { 3.0 });
        s.audit_force = rng.gen_bool(0.4).then(|| (around(&mut rng, 2e-3), 2e-3));
        s.audit_atoms = rng.gen_range(0..4);

        let old = pre_task3_untrusted_check(&s);
        let checks = s.trust_checks();
        assert_eq!(s.untrusted_check(), old, "round {i}: {checks:?}");
        assert_eq!(checks.iter().find(|c| !c.met).map(|c| c.key), old.as_ref().map(|c| c.0), "round {i}: {checks:?}");
        assert_eq!(s.trusted(), old.is_none(), "round {i}: the model exists, so trusted ⇔ no refusal");
        assert_eq!(checks.is_empty(), s.report.is_none(), "round {i}: the list is empty before a fit only");
        // the predicates' other callers, against their pre-Task-3 formulas
        let p = s.policy;
        assert_eq!(s.residuals_fail(), s.training_residuals.is_some_and(|t| !(t.energy_per_atom <= p.energy_std_per_atom) || !(t.force <= t.force_bound)), "round {i}");
        assert_eq!(s.noise_fails(), s.report.as_ref().is_some_and(|r| !(r.sigma_f <= p.force_std_abs)), "round {i}");
        assert_eq!(
            s.trust_needs_samples(),
            s.report.as_ref().is_none_or(|r| r.n_samples < r.q_prime || !(r.residual_dof_e >= MIN_RESIDUAL_DOF) || !(r.residual_dof_f >= MIN_RESIDUAL_DOF)),
            "round {i}"
        );
        // Task 3b: a NaN in any number the rule reads refuses trust
        let nan_read = s.report.as_ref().is_some_and(|r| {
            r.residual_dof_e.is_nan()
                || r.residual_dof_f.is_nan()
                || r.sigma_f.is_nan()
                || s.training_residuals.is_some_and(|t| t.energy_per_atom.is_nan() || t.force.is_nan() || t.force_bound.is_nan())
        });
        if nan_read {
            assert!(!s.trusted(), "round {i}: a NaN the rule reads must refuse trust: {checks:?}");
        }
        // the audit entry is the last, unmet exactly when an audit failed since the refit
        if let Some(a) = checks.last() {
            assert!(matches!(a.key, TrustKey::Audit | TrustKey::AuditForces) && a.met == s.audit_failed.is_none(), "round {i}: {a:?}");
        }
        *seen.entry(old.map_or("trusted".to_string(), |c| format!("{:?}", c.0))).or_default() += 1;
    }
    println!("trust-rule sweep, refusals by key over 40 000 states: {seen:?}");
    for k in ["trusted", "Samples", "ResidualDofE", "ResidualDofF", "EnergyResidual", "ForceResidual", "ForceNoise", "Audit", "AuditForces"] {
        assert!(seen.get(k).is_some_and(|&n| n >= 20), "the sweep reached {k} too rarely: {seen:?}");
    }
}

/// The class rule's wording as `class_untrusted` chose it before Activity Task 3 (e53854e):
/// (short of samples, the text) from the numbers, `None` when the old `trusted` held.
fn pre_task3_class_untrusted(l: &sim::ClassLoo, q: usize) -> Option<(bool, String)> {
    let trusted = l.samples >= q && l.force_rms <= 1.0 && l.force_max <= 1.0 && l.energy_max <= 1.0;
    if trusted {
        return None;
    }
    Some(if l.samples < q {
        (true, format!("surrogate: this kind of environment isn't learned well enough yet ({}: {} samples < Q′ {q})", l.class, l.samples))
    } else if l.force_max > 1.0 || l.force_rms > 1.0 {
        let energy = if l.energy_max > 1.0 { format!(", energy {:.1}× the tier", l.energy_max) } else { String::new() };
        (false, format!(
            "surrogate: this kind of environment isn't learned well enough yet ({}: LOO force error {:.1}× the bound, RMS {:.2}×{energy}, {} samples)",
            l.class, l.force_max, l.force_rms, l.samples
        ))
    } else {
        (false, format!(
            "surrogate: this kind of environment isn't learned well enough yet ({}: LOO energy error {:.1}× the tier, {} samples)",
            l.class, l.energy_max, l.samples
        ))
    })
}

/// Plan review M3 and "no decision changed", for the class rule: over a sweep of class numbers
/// (counts around Q′, ratios around 1 to the ulp, NaN, ∞) `class_refusal` decides exactly as the
/// old `trusted` formula, the short-of-samples flag (which exempts a fallback from thinning) is
/// the old one, and the wording is byte-identical — except a NaN force ratio with no other
/// force failure, which the old branch worded as an energy error and is now worded as a force
/// error. That case is reachable: `compute_class_loo`'s `.max` keeps `force_max` finite, but
/// one NaN LOO force reaches `force_rms` through the sum of squares (Task 3 review m1). The real
/// fit's classes agree too.
#[test]
fn the_class_verdict_reproduces_the_pre_task3_rule_on_a_sweep() {
    use super::trust::class_refusal;
    use sim::ClassKey;
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    let q = s.report().unwrap().q_prime;
    for l in s.class_loo() {
        assert_eq!(l.trusted, pre_task3_class_untrusted(l, q).is_none(), "{l:?}");
        assert_eq!(l.refusal, class_refusal(l.samples, q, l.force_rms, l.force_max, l.energy_max), "{l:?}");
    }
    let mut rng = StdRng::seed_from_u64(0xc1a55);
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    for i in 0..40_000 {
        let samples = [q.saturating_sub(1), q, q + 1, 0][rng.gen_range(0..4)];
        let (force_rms, force_max, energy_max) = (around(&mut rng, 1.0), around(&mut rng, 1.0), around(&mut rng, 1.0));
        let refusal = class_refusal(samples, q, force_rms, force_max, energy_max);
        let l = sim::ClassLoo { class: ENV_CLASSES[0], samples, force_rms, force_max, energy_rms: energy_max, energy_max, trusted: refusal.is_none(), refusal, calibration: 1.0, measured: None, measured_energy: None, worst_atom: None, evidence: 0, trusted_since: None, inflation: 1.0 };
        let old = pre_task3_class_untrusted(&l, q);
        assert_eq!(l.trusted, old.is_none(), "round {i}: {l:?}");
        if let Some((key, value, limit)) = refusal {
            match key {
                ClassKey::Samples => assert_eq!((value, limit), (samples as f64, q as f64)),
                // a failing check never shows a value inside its limit: NaN when a ratio is NaN
                ClassKey::Force => {
                    let nan = force_max.is_nan() || force_rms.is_nan();
                    assert!(limit == 1.0 && value.is_nan() == nan && (nan || (value > 1.0 && value == force_max.max(force_rms))), "round {i}: {l:?}");
                }
                ClassKey::Energy => assert!(limit == 1.0 && !(value <= 1.0) && value.to_bits() == energy_max.to_bits(), "round {i}: {l:?}"),
                ClassKey::Calibrated | ClassKey::LooFloor | ClassKey::MeasuredEnergy | ClassKey::WorstAtom => unreachable!("the uncalibrated rule never names the calibrated verdict"),
            }
        }
        s.class_loo[0] = l.clone();
        let new = s.class_untrusted(&[1, 1]).map(|(r, short)| (short, r.text));
        assert_eq!(new.is_some(), old.is_some(), "round {i}: {l:?}");
        if let (Some(new), Some(old)) = (&new, &old) {
            assert_eq!(new.0, old.0, "round {i}: the short-of-samples flag (thinning) is a decision: {l:?}");
            let nan_force_only = l.samples >= q && (force_rms.is_nan() || force_max.is_nan()) && !(force_max > 1.0 || force_rms > 1.0);
            if nan_force_only {
                assert!(new.1.contains("LOO force error") && old.1.contains("LOO energy error"), "round {i}: {new:?} {old:?}");
            } else {
                assert_eq!(new.1, old.1, "round {i}: {l:?}");
            }
        }
        *seen.entry(refusal.map_or("trusted".to_string(), |r| format!("{:?}", r.0))).or_default() += 1;
    }
    println!("class-rule sweep over 40 000 states: {seen:?}");
    for k in ["trusted", "Samples", "Force", "Energy"] {
        assert!(seen.get(k).is_some_and(|&n| n >= 100), "the sweep reached {k} too rarely: {seen:?}");
    }
}

/// Task 3 review I1: a refit leaves `last_audit` in place (the Log's record) but it no longer
/// describes the current fit, so "passed" must not be read as a check of the new coefficients.
/// Three states: never audited (`None`), audited before the refit (`Some`, not this fit), and
/// audited this fit. The audit row is met at value 0 in the first two either way: it says only
/// that no audit failed since the refit.
#[test]
fn a_refit_leaves_the_last_audit_to_an_earlier_fit() {
    use sim::activity::TrustKey;
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy { audit_every: 2, ..FallbackPolicy::default() });
    assert!(s.last_audit().is_none() && !s.last_audit_this_fit(), "never audited");
    let audit_now = |s: &mut SurrogateKs| {
        let before = s.last_audit().cloned();
        for _ in 0..10 {
            s.evaluate(&h2(1.6), &mut never()).unwrap();
            if s.last_audit().cloned() != before && s.last_audit().is_some() {
                return;
            }
        }
        panic!("no audit in 10 answered steps at audit_every 2");
    };
    audit_now(&mut s);
    assert_eq!(s.last_audit().map(|a| a.verdict), Some(AuditVerdict::Passed));
    assert!(s.last_audit_this_fit() && Trainable::status(&s, &h2(1.6), &[]).last_audit_this_fit);
    s.refit().unwrap();
    assert_eq!(s.last_audit().map(|a| a.verdict), Some(AuditVerdict::Passed), "the record stays, for the Log");
    assert!(!s.last_audit_this_fit(), "no audit judged this fit yet");
    assert!(!Trainable::status(&s, &h2(1.6), &[]).last_audit_this_fit);
    let c = *s.trust_checks().last().unwrap();
    assert_eq!((c.key, c.met, c.value), (TrustKey::Audit, true, 0.0));
    audit_now(&mut s);
    assert!(s.last_audit_this_fit(), "the next audit judges this fit");
}

// ---- Activity Task 3b: NaN refuses trust (the physics review's ruling (b)) ----

/// Every number the trust rule reads, made NaN alone on a trusted model, refuses trust with that
/// condition's key, and `trust_needs_samples` counts a NaN dof as short (every answer is then
/// kept, never thinned).
#[test]
fn a_nan_in_any_trust_input_refuses_trust() {
    use sim::activity::TrustKey;
    let (base, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    assert!(base.trusted());
    type Poke = fn(&mut SurrogateKs);
    let cases: [(&str, Poke, TrustKey); 6] = [
        ("residual_dof_e", |s| s.report.as_mut().unwrap().residual_dof_e = f64::NAN, TrustKey::ResidualDofE),
        ("residual_dof_f", |s| s.report.as_mut().unwrap().residual_dof_f = f64::NAN, TrustKey::ResidualDofF),
        ("energy residual", |s| s.training_residuals.as_mut().unwrap().energy_per_atom = f64::NAN, TrustKey::EnergyResidual),
        ("force residual", |s| s.training_residuals.as_mut().unwrap().force = f64::NAN, TrustKey::ForceResidual),
        ("force bound", |s| s.training_residuals.as_mut().unwrap().force_bound = f64::NAN, TrustKey::ForceResidual),
        ("σ_f", |s| s.report.as_mut().unwrap().sigma_f = f64::NAN, TrustKey::ForceNoise),
    ];
    for (name, poke, key) in cases {
        let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
        poke(&mut s);
        assert!(!s.trusted(), "a NaN {name} must refuse trust");
        assert_eq!(s.trust_checks().into_iter().find(|c| !c.met).map(|c| c.key), Some(key), "{name}");
        assert_eq!(s.untrusted_check().map(|c| c.0), Some(key), "{name}");
        if matches!(key, TrustKey::ResidualDofE | TrustKey::ResidualDofF) {
            assert!(s.trust_needs_samples(), "{name}: a NaN dof is short of samples");
        }
        if matches!(key, TrustKey::EnergyResidual | TrustKey::ForceResidual) {
            assert!(s.residuals_fail(), "{name}: the recovery sees it");
        }
        if key == TrustKey::ForceNoise {
            assert!(s.noise_fails(), "{name}: the recovery sees it");
        }
    }
}

/// Make every prediction of `s`'s model NaN (a coefficient the fit would never install since
/// Task 3b; injected here to drive the paths downstream of the fit).
fn poison_the_model(s: &mut SurrogateKs) {
    let m = s.model.as_mut().expect("a fitted model");
    for c in m.coeffs.iter_mut().flatten() {
        *c = f64::NAN;
    }
}

/// The accumulators propagate a NaN instead of dropping it (`f64::max` and `x > worst` both skip
/// a NaN): a model whose predictions are NaN reports NaN training residuals, not zero, and its
/// classes are not trusted by the leave-one-out errors.
#[test]
fn a_nan_prediction_reaches_the_training_residuals_and_the_class_rule() {
    use sim::ClassKey;
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    assert!(s.trusted() && s.class_loo()[0].trusted);
    poison_the_model(&mut s);
    let t = s.compute_training_residuals().expect("a model and a store");
    assert!(t.energy_per_atom.is_nan() && t.force.is_nan(), "NaN predictions must show as NaN residuals: {t:?}");
    s.training_residuals = Some(t);
    assert!(!s.trusted());
    let classes = s.compute_class_loo_full().0;
    let pair = &classes[0];
    assert!(!pair.trusted, "{pair:?}");
    assert!(matches!(pair.refusal, Some((ClassKey::Force | ClassKey::Energy, ..))), "the LOO errors refuse it, not the count: {pair:?}");
}

/// A teacher that answers NaN energies and forces while `nan` is set.
struct NanTeacher {
    inner: Arc<Mutex<MorseDimer>>,
    energy: Arc<std::sync::atomic::AtomicBool>,
    forces: Arc<std::sync::atomic::AtomicBool>,
}

impl ForceProvider for NanTeacher {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        use std::sync::atomic::Ordering::SeqCst;
        let mut r = self.inner.lock().unwrap().evaluate(scene, should_abort)?;
        if self.energy.load(SeqCst) {
            r.energy = f64::NAN;
        }
        if self.forces.load(SeqCst) {
            r.forces[0][0] = f64::NAN;
        }
        Ok(r)
    }
    fn name(&self) -> &'static str {
        "morse dimer"
    }
    fn reset(&mut self) {}
}

/// An audit whose gap or worst force residual is not a number fails (before: `NaN > tier` was
/// false, so it passed): in the step and on the background auditor.
#[test]
fn an_audit_with_a_nan_gap_or_force_fails() {
    use std::sync::atomic::{AtomicBool, Ordering::SeqCst};
    // in the step: a NaN teacher energy, then (after a refit) a NaN teacher force
    for nan_energy in [true, false] {
        let (mut s, h) = trained(1.2, 2.2, 120, FallbackPolicy { audit_every: 2, ..FallbackPolicy::default() });
        let (energy, forces) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
        s.teacher = Box::new(NanTeacher { inner: h.clone(), energy: energy.clone(), forces: forces.clone() });
        if nan_energy { energy.store(true, SeqCst) } else { forces.store(true, SeqCst) }
        for _ in 0..4 {
            s.evaluate(&h2(1.6), &mut never()).unwrap();
            if s.last_audit().is_some() {
                break;
            }
        }
        let a = s.last_audit().expect("an audit at audit_every 2");
        assert_eq!(a.verdict, AuditVerdict::Failed, "nan energy {nan_energy}: {a:?}");
        assert!(!s.trusted() && s.audit_failed().is_some());
        let c = s.trust_checks().into_iter().find(|c| !c.met).unwrap();
        assert!(matches!(c.key, sim::activity::TrustKey::Audit | sim::activity::TrustKey::AuditForces), "{c:?}");
    }
    // the background auditor's verdict: a NaN energy, then NaN forces
    for nan_energy in [true, false] {
        let (mut s, _, _, _) = with_gated_auditor(5, 0.0);
        let scene = h2(1.5);
        let prov = s.provenance_for(&scene);
        let mut teacher = MorseDimer::h2().evaluate(&scene, &mut never()).unwrap();
        if nan_energy {
            teacher.energy = f64::NAN;
        } else {
            teacher.forces[1][2] = f64::NAN;
        }
        assert!(s.trusted());
        s.handle_answer(audit_answer(&s, &scene, prov, Ok((teacher.energy, teacher.forces.clone(), None, None))));
        assert_eq!(s.last_audit().map(|a| a.verdict), Some(AuditVerdict::Failed), "nan energy {nan_energy}");
        assert!(!s.trusted() && s.audit_failed().is_some());
    }
}

/// A prediction that is not finite is never an answer, draft mode included: the teacher answers,
/// on its own surface, with the rule `NonFiniteAnswer`, and E_conserved stays finite.
#[test]
fn a_non_finite_answer_is_refused_even_in_draft_mode() {
    use sim::activity::TeachRule;
    for draft in [false, true] {
        let (mut s, h) = trained(1.2, 2.2, 120, FallbackPolicy { never_fallback: draft, ..FallbackPolicy::default() });
        let r = s.evaluate(&h2(1.6), &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Surrogate, "the control answers");
        poison_the_model(&mut s);
        let before = calls(&h);
        let r = s.evaluate(&h2(1.62), &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback, "draft {draft}: a NaN prediction must not answer");
        assert_eq!(r.stats.energy_surface(), Surface::Teacher);
        assert!(r.energy.is_finite() && r.forces.iter().flatten().all(|x| x.is_finite()));
        assert!(r.stats.refit_jump.is_none_or(|j| j.is_finite()), "the switch is booked finitely: {:?}", r.stats.refit_jump);
        assert_eq!(calls(&h), before + 1);
        assert!(r.stats.notes.iter().any(|n| n.contains("the model's answer is not finite")), "{:?}", r.stats.notes);
        let e = drain(&mut s);
        assert_eq!(e.last().map(|e| e.rule), Some(TeachRule::NonFiniteAnswer));
        assert_event_rule(e.last().unwrap(), TeachRule::NonFiniteAnswer);
    }
}

/// Rules 6 and 7 refuse a NaN std (before: `NaN > bound` was false and the energy's
/// `max(0.0)` turned a NaN variance into 0).
#[test]
fn the_variance_rules_refuse_a_nan_std() {
    use sim::activity::TeachRule as R;
    let (s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    let model = s.model().unwrap().clone();
    let pos: Vec<[f64; 3]> = h2(1.6).nuclei.iter().map(|n| n.pos).collect();
    let pred = model.predict(&pos, &[1, 1], true).unwrap();
    let v = s.variance_check(&model, &pred, f64::NAN, &[0.0, 0.0], &pos);
    assert!(!v.accept);
    assert_eq!(v.reason.as_ref().map(|r| r.rule), Some(R::EnergyUncertainty));
    let v = s.variance_check(&model, &pred, 0.0, &[0.0, f64::NAN], &pos);
    assert!(!v.accept);
    assert_eq!(v.reason.as_ref().map(|r| (r.rule, r.atom)), Some((R::ForceUncertainty, Some(1))));
    assert!(v.uncertainty.is_nan(), "the readout shows the NaN, not the other atom's std");
}

/// A fit whose numbers are not finite is refused at fit time; the previous model, its report and
/// its trust stay (the error path of `refit`).
#[test]
fn a_non_finite_fit_is_refused_and_the_previous_model_stays() {
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    let before = (s.model().unwrap().coeffs.clone(), s.report().unwrap().clone(), s.trusted());
    let sc = h2(1.7);
    let mut bad = morse_sample(&MorseDimer::h2(), &sc);
    bad.energy = f64::NAN;
    s.add_sample(bad, &sc).unwrap();
    let e = s.refit().expect_err("a NaN label cannot make a finite fit");
    assert!(matches!(e, FitError::NonFinite(_)), "{e:?}");
    assert_eq!((s.model().unwrap().coeffs.clone(), s.report().unwrap().clone(), s.trusted()), before);
}

// ---- would_answer and the training views (activity Task 4: spec A6, §9) ----

/// The positions and charges of a scene (`would_answer`'s arguments).
fn pos_z(sc: &Scene) -> (Vec<[f64; 3]>, Vec<u8>) {
    (sc.nuclei.iter().map(|n| n.pos).collect(), sc.nuclei.iter().map(|n| n.z).collect())
}

/// Spec A6: `would_answer` is `evaluate`'s verdict without its side effects. With the variance
/// evaluated on every consultation and audits off, `evaluate` answers from the model exactly
/// where `would_answer` says Ok, with the same rule and atom otherwise, over dimers and trimers.
#[test]
fn would_answer_agrees_with_evaluate() {
    let policy = FallbackPolicy { variance_cadence: 1, audit_every: 0, ..FallbackPolicy::default() };
    let (mut s, h) = trained(1.2, 2.2, 120, policy);
    let mut rng = StdRng::seed_from_u64(0x5EED);
    let mut scenes = Vec::new();
    for _ in 0..150 {
        scenes.push(h2(rng.gen_range(0.9..3.2)));
    }
    for _ in 0..50 {
        let mut sc = h2(rng.gen_range(1.2..2.0));
        sc.nuclei.push(Nucleus::new(1, [rng.gen_range(-2.5..2.5), rng.gen_range(-2.5..2.5), rng.gen_range(-2.5..2.5)]));
        scenes.push(sc);
    }
    let calls_before = calls(&h);
    s.ensure_desc_cache(); // the fence reads the cache: `would_answer` is `&self`
    let verdicts: Vec<_> = scenes
        .iter()
        .map(|sc| {
            let (pos, z) = pos_z(sc);
            s.would_answer(&pos, &z, &s.provenance_for(sc))
        })
        .collect();
    assert_eq!(calls(&h), calls_before, "would_answer never calls the teacher");
    // the verdicts were all taken before any evaluate: no refit runs in between, so the model
    // they judged is the one evaluate consults (a kept fallback sample changes no verdict until a refit)
    let (mut agree, mut answered) = (0, 0);
    let mut rules: BTreeMap<String, usize> = BTreeMap::new();
    for (sc, v) in scenes.iter().zip(&verdicts) {
        Trainable::take_teach_events(&mut s);
        s.discontinuity();
        if sc.check_inside_cell().is_err() {
            continue;
        }
        // the Morse teacher answers only dimers: a trimer handed to it fails, and the failure's
        // event still carries the rule that handed it over (Task 2)
        let r = s.evaluate(sc, &mut never());
        let ev = Trainable::take_teach_events(&mut s);
        match v {
            Ok(()) => {
                assert_eq!(r.expect("a model answer needs no teacher").stats.source, Source::Surrogate, "{:?}", sc.nuclei);
                answered += 1;
            }
            Err((rule, atom)) => {
                assert_eq!(ev.first().map(|e| (e.rule, e.atom)), Some((*rule, *atom)), "{:?}", sc.nuclei);
                *rules.entry(format!("{rule:?}")).or_default() += 1;
            }
        }
        agree += 1;
    }
    eprintln!("would_answer: {agree} geometries compared, {answered} answered by the model, refusals {rules:?}");
    assert!(answered > 20 && agree - answered > 20, "the sweep must cover both verdicts");
}

/// Plan review S1: the fence, the trust rule and draft mode — and the variance rule with its
/// atom, and the non-finite answer ahead of the envelope (Task 3b put it before every other
/// rule in `evaluate`). `would_answer` names each as `evaluate` does, and ignores draft mode: it
/// is the policy's verdict (the pair-curve caption says so in draft mode, Task 9).
#[test]
fn would_answer_covers_the_fence_untrusted_and_draft() {
    use sim::activity::TeachRule as R;
    let verdict = |s: &mut SurrogateKs, sc: &Scene| {
        let (pos, z) = pos_z(sc);
        s.ensure_desc_cache();
        s.would_answer(&pos, &z, &s.provenance_for(sc))
    };
    let first = |s: &mut SurrogateKs| Trainable::take_teach_events(s).first().map(|e| (e.rule, e.atom));
    let policy = FallbackPolicy { variance_cadence: 1, audit_every: 0, ..FallbackPolicy::default() };
    // the fence: a stored sample held in quarantine fences its own geometry
    let (mut s, _) = trained(1.2, 2.2, 120, policy);
    let k = 60;
    let p = s.store().unwrap().samples[k].pos.clone();
    let r_k = ((p[0][0] - p[1][0]).powi(2) + (p[0][1] - p[1][1]).powi(2) + (p[0][2] - p[1][2]).powi(2)).sqrt();
    s.quarantine.resize(s.samples(), Q_NONE);
    s.quarantine[k] = Q_HELD;
    let sc = h2(r_k);
    assert_eq!(verdict(&mut s, &sc), Err((R::NearQuarantine, None)));
    // the fence without the cache filled: the same verdict (the held sample is described on the fly)
    s.desc_cache.clear();
    s.desc_basis = None;
    let (pos, z) = pos_z(&sc);
    assert_eq!(s.would_answer(&pos, &z, &s.provenance_for(&sc)), Err((R::NearQuarantine, None)));
    s.evaluate(&sc, &mut never()).unwrap();
    assert_eq!(first(&mut s), Some((R::NearQuarantine, None)));
    // untrusted (rule 4 on the count)
    let (mut s, _) = trained(1.2, 2.2, 5, policy);
    let v = verdict(&mut s, &h2(1.6));
    assert!(matches!(v, Err((R::Untrusted(_), None))), "{v:?}");
    s.evaluate(&h2(1.6), &mut never()).unwrap();
    assert_eq!(first(&mut s), v.err());
    // rule 6 with its atom (an inflated posterior): every in-window dimer
    let (mut s, _) = trained(1.2, 2.2, 120, policy);
    always_unsure(&mut s);
    let mut rng = StdRng::seed_from_u64(61);
    for _ in 0..20 {
        let sc = dimer(rng.gen_range(1.3..2.1), unit_vector(&mut rng), [0.0; 3], 1, 1);
        let v = verdict(&mut s, &sc);
        assert!(matches!(v, Err((R::ForceUncertainty, Some(_)))), "{v:?}");
        s.discontinuity();
        s.evaluate(&sc, &mut never()).unwrap();
        assert_eq!(first(&mut s), v.err());
    }
    // a prediction that is not finite: refused ahead of the envelope, as `evaluate` does
    let (mut s, _) = trained(1.2, 2.2, 120, policy);
    let far = h2(4.0);
    assert_eq!(verdict(&mut s, &far), Err((R::Spread, Some(0))), "the control: 5c at 4.0 Bohr");
    poison_the_model(&mut s);
    assert_eq!(verdict(&mut s, &far), Err((R::NonFiniteAnswer, None)));
    s.evaluate(&far, &mut never()).unwrap();
    assert_eq!(first(&mut s), Some((R::NonFiniteAnswer, None)));
    // draft: evaluate answers from the model where the policy refuses; would_answer still
    // refuses. (In the model's own 33³ cell: `in_big_cell` changes the grid, so the gate would
    // refuse under rule 3, which draft mode never answers past.)
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy { never_fallback: true, ..policy });
    assert_eq!(verdict(&mut s, &far), Err((R::Spread, Some(0))), "the policy's verdict, draft or not");
    assert_eq!(s.evaluate(&far, &mut never()).unwrap().stats.source, Source::Surrogate);
}

/// Plan review S1: building the views between steps changes nothing the policy reads. Two
/// identical providers, one building `training_views` (and asking `would_answer`) before every
/// evaluate, give bit-identical sources, energies, forces, variance-check counts, stores and
/// event streams over 200 steps.
#[test]
fn training_views_change_nothing_the_policy_reads() {
    let policy = FallbackPolicy { audit_every: 7, ..FallbackPolicy::default() };
    let (mut a, _) = trained(1.2, 2.2, 120, policy);
    let (mut b, _) = trained(1.2, 2.2, 120, policy);
    let mut rng = StdRng::seed_from_u64(0xA11);
    let key = |e: &sim::TeachEvent| (e.origin, e.rule, e.atom, e.outcome, e.gap.map(f64::to_bits), e.detail.clone());
    for k in 0..200 {
        let sc = h2(rng.gen_range(1.0..3.0));
        let _ = b.training_views();
        let (pos, z) = pos_z(&sc);
        let _ = b.would_answer(&pos, &z, &b.provenance_for(&sc));
        match (a.evaluate(&sc, &mut never()), b.evaluate(&sc, &mut never())) {
            (Ok(ra), Ok(rb)) => {
                assert_eq!(ra.stats.source, rb.stats.source);
                assert_eq!(ra.energy.to_bits(), rb.energy.to_bits());
                assert_eq!(ra.forces, rb.forces);
            }
            (Err(ea), Err(eb)) => assert_eq!(ea.to_string(), eb.to_string()),
            (ra, rb) => panic!("diverged: {:?} vs {:?}", ra.map(|r| r.stats.source), rb.map(|r| r.stats.source)),
        }
        let (ea, eb) = (Trainable::take_teach_events(&mut a), Trainable::take_teach_events(&mut b));
        assert_eq!(ea.iter().map(key).collect::<Vec<_>>(), eb.iter().map(key).collect::<Vec<_>>());
        assert_eq!((a.variance_checks(), a.samples()), (b.variance_checks(), b.samples()));
        if k % 50 == 49 && a.pending_refit() {
            assert_eq!(a.refit().map(|r| r.n_samples), b.refit().map(|r| r.n_samples));
        }
    }
    assert_eq!(a.store().unwrap().samples, b.store().unwrap().samples);
}

/// Spec §9: parity points are (E − Σ E_ref)/N of the teacher and of `predict` at each stored
/// sample, with the fit's own references; the band is the epistemic std, the same number rule 7
/// compares; the force ratio is rule 6's; each curve point's verdict is `would_answer`'s at that
/// dimer; no teacher call.
#[test]
fn training_views_read_the_store_and_the_posterior_only() {
    let (mut s, h) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    let c0 = calls(&h);
    let v = s.training_views().expect("a fitted model");
    eprintln!("training views at S = 1 (Q′ {}): build_ms {:.2}, {} parity points", s.report().unwrap().q_prime, v.build_ms, v.parity_energy.len());
    assert_eq!(calls(&h), c0);
    assert_eq!(v.parity_energy.len(), s.samples());
    assert_eq!((v.parity_omitted, v.curves_skipped), (0, 0));
    let st = s.store().unwrap();
    let m = s.model().unwrap();
    // the fit's own reference energy (plan review S4); here the Morse atom's
    let e_ref = m.ref_energy[m.basis.species_index(1).unwrap()];
    assert!((e_ref - MorseDimer::h2().e_atom).abs() < 1e-12);
    for p in &v.parity_energy {
        let smp = &st.samples[p.index];
        let n = smp.pos.len() as f64;
        assert!((p.dft - (smp.energy - n * e_ref) / n).abs() < 1e-12);
        let pred = m.predict(&smp.pos, &smp.z, false).unwrap();
        assert!((p.model - (pred.energy - n * e_ref) / n).abs() < 1e-12);
        assert_eq!(p.state, sim::activity::SampleState::Fitted);
        assert_eq!(p.kind, sim::activity::SampleKind::Pass, "scan samples are the pass's");
    }
    assert_eq!(v.curves.len(), 1);
    let c = &v.curves[0];
    assert_eq!(c.z, (1, 1));
    assert_eq!((c.r.len(), c.force_ratio.len()), (sim::activity::CURVE_POINTS, sim::activity::CURVE_POINTS));
    assert_eq!((c.energy.len(), c.sigma.len(), c.refused.len()), (sim::activity::CURVE_POINTS, sim::activity::CURVE_POINTS, sim::activity::CURVE_POINTS));
    assert_eq!(c.r[0], sim::activity::CURVE_R_MIN);
    assert!(c.r.windows(2).all(|w| w[0] < w[1]) && *c.r.last().unwrap() < m.basis.r_cut);
    let k = c.r.iter().position(|&r| r >= 1.6).unwrap();
    let pos = [[-c.r[k] / 2.0, 0.0, 0.0], [c.r[k] / 2.0, 0.0, 0.0]];
    let pred = m.predict(&pos, &[1, 1], true).unwrap();
    let floor = 2.0 * m.sigma_e; // energy_per_atom models: N s_E
    let epi = (pred.energy_std.unwrap().powi(2) - floor * floor).max(0.0).sqrt();
    assert!((c.sigma[k] - epi).abs() < 1e-15);
    let ratio = s.variance_check(m, &pred, pred.energy_std.unwrap(), pred.force_std_epistemic.as_ref().unwrap(), &pos).atoms.iter().copied().fold(0.0, f64::max);
    assert!((c.force_ratio[k] - ratio).abs() < 1e-15, "rule 6's ratio");
    assert!((c.energy[k] - (pred.energy - 2.0 * e_ref)).abs() < 1e-12);
    assert_eq!(c.refused[k], None, "inside the window the model answers");
    assert!(c.refused.last().unwrap().is_some(), "near r_cut the dimer is refused (5a or the envelope)");
    // one prediction per point serves the band and the verdict: the verdict is `would_answer`'s
    for (j, &r) in c.r.iter().enumerate() {
        let pos = [[-r / 2.0, 0.0, 0.0], [r / 2.0, 0.0, 0.0]];
        assert_eq!(c.refused[j], s.would_answer(&pos, &[1, 1], &m.provenance).err().map(|e| e.0), "r = {r}");
    }
    eprintln!(
        "H–H curve: answered on r ∈ [{:.3}, {:.3}] Bohr; refusals {:?}",
        c.r.iter().zip(&c.refused).find(|(_, v)| v.is_none()).map_or(f64::NAN, |x| *x.0),
        c.r.iter().zip(&c.refused).filter(|(_, v)| v.is_none()).last().map_or(f64::NAN, |x| *x.0),
        c.refused.iter().flatten().fold(BTreeMap::<String, usize>::new(), |mut m, r| {
            *m.entry(format!("{r:?}")).or_default() += 1;
            m
        })
    );
    assert_eq!(c.dots.len(), 120, "every stored sample is a two-atom H–H sample");
    assert_eq!(v.pairs[0].counts.iter().sum::<u32>(), 120);
    assert!((v.dimer_floor - floor).abs() < 1e-18 && v.energy_tier == 1e-3);
    assert_eq!((v.epoch, v.samples), (s.epoch(), s.samples()));
    assert_eq!((v.force_points_total, v.parity_force.len()), (120 * 2 * 3, 120 * 2 * 3));
}

/// The sample states and what each view takes from them (spec §3.3.5, plan review N12, S18):
/// parity shows every stored sample with its state; the force parity and the histogram take the
/// fitted ones only; the dots every two-atom sample of the pair; a sample of a species the basis
/// lacks is omitted and counted.
#[test]
fn the_views_follow_the_sample_states() {
    use sim::activity::SampleState as St;
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    s.quarantine.resize(s.samples(), Q_NONE);
    s.quarantine[5] = Q_HELD;
    s.quarantine[6] = Q_SUPERSEDED;
    s.quarantine[7] = Q_BOX;
    let sc = h2(1.7);
    s.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap(); // index 120: not in the fit
    let he = dimer(1.6, [1.0, 0.0, 0.0], [0.0; 3], 1, 2);
    s.add_sample(morse_sample(&MorseDimer::h2(), &he), &he).unwrap(); // index 121: no He in the basis
    let mut nan = morse_sample(&MorseDimer::h2(), &sc);
    nan.energy = f64::NAN;
    s.add_sample(nan, &sc).unwrap(); // index 122: a label that is not finite (review m5)
    let v = s.training_views().unwrap();
    assert_eq!((v.samples, v.parity_energy.len(), v.parity_omitted), (123, 121, 2));
    assert!(v.parity_energy.iter().all(|p| p.dft.is_finite() && p.model.is_finite()));
    let state = |i: usize| v.parity_energy.iter().find(|p| p.index == i).map(|p| p.state);
    assert_eq!((state(5), state(6), state(7), state(120), state(121), state(0)), (Some(St::Quarantined), Some(St::Superseded), Some(St::BoxExcluded), Some(St::Unfitted), None, Some(St::Fitted)));
    let fitted = 117;
    assert_eq!((v.force_points_total, v.parity_force.len()), (fitted * 6, fitted * 6));
    assert_eq!(v.pairs.len(), 1, "the He sample is not in the fit");
    assert_eq!(v.pairs[0].counts.iter().sum::<u32>(), fitted as u32);
    assert_eq!(v.pairs[0].bin, 0.1);
    assert_eq!(v.curves.len(), 1, "the basis knows H only");
    assert_eq!(v.curves[0].dots.len(), 121, "every two-atom H–H sample, any state");
    assert_eq!(v.curves[0].dots.iter().filter(|d| d.2 == St::Fitted).count(), fitted);
    // the force parity of a fitted sample: (teacher, model) per component, in sample order
    let m = s.model().unwrap();
    let smp = &s.store().unwrap().samples[0];
    let pred = m.predict(&smp.pos, &smp.z, false).unwrap();
    assert_eq!(v.parity_force[0], [smp.forces[0][0] as f32, pred.forces[0][0] as f32]);
    assert_eq!(v.parity_force[5], [smp.forces[1][2] as f32, pred.forces[1][2] as f32]);
    // no model: no views
    let (mut empty, _) = trained(1.2, 2.2, 0, FallbackPolicy::default());
    assert!(empty.training_views().is_none() && Trainable::training_views(&mut empty).is_none());
    assert!(Trainable::training_views(&mut s).is_some(), "the Trainable face builds them too");
}

/// Plan review S5: the force parity is thinned by whole atoms, so x, y and z stay equally
/// represented (a component stride that is a multiple of 3 kept a single component), and the
/// cap holds.
#[test]
fn force_parity_thins_whole_atoms() {
    use super::views::thin_atoms;
    use sim::activity::FORCE_PARITY_CAP;
    let atoms: Vec<[[f32; 2]; 3]> = (0..7000).map(|i| [[i as f32, 0.0], [i as f32, 1.0], [i as f32, 2.0]]).collect();
    let out = thin_atoms(&atoms, FORCE_PARITY_CAP);
    assert!(out.len() <= FORCE_PARITY_CAP && out.len() % 3 == 0);
    for c in 0..3 {
        assert_eq!(out.iter().filter(|p| p[1] == c as f32).count(), out.len() / 3, "component {c}");
    }
    for n in [1, 6666, 6667, 13_334, 40_000] {
        let a: Vec<[[f32; 2]; 3]> = atoms.iter().cycle().take(n).copied().collect();
        let out = thin_atoms(&a, FORCE_PARITY_CAP);
        assert!(out.len() <= FORCE_PARITY_CAP, "{n} atoms → {} components", out.len());
    }
    assert_eq!(thin_atoms(&atoms[..10], FORCE_PARITY_CAP).len(), 30, "under the cap nothing is dropped");
    assert!(thin_atoms(&[], FORCE_PARITY_CAP).is_empty());
}

/// A store over four species (H, He, Li, Be) on the shared Morse form — the species only label
/// channels here; the numbers are a cost probe, not physics: 40 dimers for each of the 10 pairs,
/// every reference the Morse atom's. The fit is installed as `fit_store` installs it (model,
/// report, envelope, null space, training residuals, class LOO, active set) but with the
/// hyperparameters fixed: the evidence search at Q′ 1232 is the S = 4 refit `references.md`
/// records at 260 s (with Q′ + 60 samples; this store, below Q′, ran past 14 min before it was
/// stopped), and the views' cost does not depend on how λ, s_E and s_F were chosen.
fn four_species_trained() -> (SurrogateKs, Arc<Mutex<MorseDimer>>) {
    let teacher = Shared::new(MorseDimer::h2());
    let handle = teacher.handle();
    let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_teacher_tol(1e-6);
    for z in 1..=4 {
        s.set_reference_energy(z, MorseDimer::h2().e_atom);
    }
    let mut rng = StdRng::seed_from_u64(44);
    for za in 1..=4u8 {
        for zb in za..=4u8 {
            for i in 0..40 {
                let r = 1.2 + i as f64 / 39.0;
                let origin = [rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3)];
                let sc = dimer(r, unit_vector(&mut rng), origin, za, zb);
                s.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
            }
        }
    }
    let t = std::time::Instant::now();
    let basis = Basis::new(&[1, 2, 3, 4]);
    let refs = s.reference_vector(&basis).unwrap();
    let opts = FitOptions { lambda: Some(1e-3), sigma_e: Some(1e-5), sigma_f: Some(1e-4), ..FitOptions::default() };
    let mut acc = Accumulator::new(&basis);
    for smp in &s.store().unwrap().samples {
        acc.add(smp, &basis, &refs, &opts).unwrap();
    }
    let (model, report) = acc.solve(&opts, &s.store().unwrap().provenance, &refs).unwrap();
    let t_solve = t.elapsed().as_secs_f64();
    s.envelope = Some(acc.envelope.clone());
    s.null_space = Some(NullSpace::of(&acc, &model.scale));
    let t_null = t.elapsed().as_secs_f64() - t_solve;
    s.accumulated = s.samples();
    // what `fit_store` records beside an installed fit (the views' `Fitted` and cache key)
    s.fit_members = (s.samples(), Vec::new());
    s.model_gen += 1;
    s.accumulator = Some(acc);
    s.model = Some(model);
    s.report = Some(report);
    s.training_residuals = s.compute_training_residuals();
    s.class_loo = s.compute_class_loo_full().0;
    s.rebuild_active_set();
    eprintln!(
        "four-species fit (fixed hyperparameters): solve {t_solve:.1} s, null space {t_null:.1} s, all {:.1} s; Q′ {}, {} samples, trusted {} ({:?})",
        t.elapsed().as_secs_f64(),
        s.report().unwrap().q_prime,
        s.samples(),
        s.trusted(),
        s.untrusted_reason()
    );
    (s, handle)
}

/// Review Focus 5: the build time at S = 4 is reported (never asserted), with the share of the
/// worker's time it takes at one build per 2 s (plan review S2: the build runs between steps).
/// This store is short of Q′, so the gate refuses every curve point and the verdict costs
/// nothing there; the full verdict past the gate (envelope, novelty, class, variance, fence) on
/// the same 960 points is timed separately — the add-on for a trusted S = 4 model. Then S = 2
/// with a real refit. Release only (≈ 20 s: the S = 4 null space and class LOO, then the views).
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn training_views_build_time_is_reported() {
    let (mut s, h) = four_species_trained();
    let c0 = calls(&h);
    let t = std::time::Instant::now();
    let v = s.training_views().unwrap();
    let ms = t.elapsed().as_secs_f64() * 1e3;
    // a second build at the same model generation: the curve cache is warm
    let t = std::time::Instant::now();
    let v2 = s.training_views().unwrap();
    let ms2 = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(calls(&h), c0, "no teacher call");
    assert_eq!((v.curves.len(), v2.curves.len()), (10, 10));
    // the verdict past the gate on every curve point, from the band's prediction: the
    // predictions and the verdicts timed apart
    let m = s.model().unwrap().clone();
    let points: Vec<([[f64; 3]; 2], [u8; 2])> =
        v.curves.iter().flat_map(|c| c.r.iter().map(move |&r| ([[-r / 2.0, 0.0, 0.0], [r / 2.0, 0.0, 0.0]], [c.z.0, c.z.1]))).collect();
    let t = std::time::Instant::now();
    let preds: Vec<_> = points.iter().map(|(pos, z)| m.predict(pos, z, true).unwrap()).collect();
    let ms_pred = t.elapsed().as_secs_f64() * 1e3;
    let t = std::time::Instant::now();
    let mut refusals: BTreeMap<String, usize> = BTreeMap::new();
    for ((pos, z), p) in points.iter().zip(&preds) {
        let verdict = s.verdict_past_gate(&m, pos, z, p);
        *refusals.entry(format!("{:?}", verdict.err().map(|e| e.0))).or_default() += 1;
    }
    let ms_verdict = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(v.curves, v2.curves, "the warm build is the cold one");
    eprintln!(
        "training views at S = 4 (Q′ {}, {} samples): cold {ms:.0} ms (build_ms {:.0}), warm {ms2:.1} ms (build_ms {:.1}), {} curves × {} points, {} parity points, {} force components; the gate refuses every curve point ({:?}); cold, at one build per 2 s the worker would spend {:.0} % of its time on views",
        s.report().unwrap().q_prime,
        s.samples(),
        v.build_ms,
        v2.build_ms,
        v.curves.len(),
        sim::activity::CURVE_POINTS,
        v.parity_energy.len(),
        v.parity_force.len(),
        v.curves[0].refused[0],
        100.0 * ms / 2000.0
    );
    eprintln!(
        "the cold build caches every point's verdict past the gate, so it is the trusted model's cost too; its parts, timed apart: the {} curve predictions with variance {ms_pred:.0} ms ({:.2} ms each), the verdicts past the gate {ms_verdict:.0} ms ({:.2} ms per point; verdicts {refusals:?}); warm, at one build per 2 s: {:.2} % of the worker",
        points.len(),
        ms_pred / points.len() as f64,
        ms_verdict / points.len() as f64,
        100.0 * ms2 / 2000.0
    );
    // S = 2 (H, He), a real refit: 60 dimers of each of the 3 pairs (Q′ 176)
    let (mut s2, h2h) = trained(1.2, 2.2, 60, FallbackPolicy::default());
    s2.set_reference_energy(2, MorseDimer::h2().e_atom);
    let mut rng = StdRng::seed_from_u64(45);
    for (za, zb) in [(1, 2), (2, 2)] {
        for i in 0..60 {
            let sc = dimer(1.2 + i as f64 / 59.0, unit_vector(&mut rng), [0.0; 3], za, zb);
            s2.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
        }
    }
    let t = std::time::Instant::now();
    let rep = s2.refit().expect("refit");
    let refit_s = t.elapsed().as_secs_f64();
    let c2 = calls(&h2h);
    let t = std::time::Instant::now();
    let v = s2.training_views().unwrap();
    let ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(calls(&h2h), c2, "no teacher call");
    let answered: usize = v.curves.iter().map(|c| c.refused.iter().filter(|r| r.is_none()).count()).sum();
    eprintln!(
        "training views at S = 2 (Q′ {}, {} samples, refit {refit_s:.1} s, trusted {}): {ms:.1} ms, {} curves, {answered} of {} curve points answered; {:.2} % of the worker at one build per 2 s",
        rep.q_prime,
        s2.samples(),
        s2.trusted(),
        v.curves.len(),
        v.curves.len() * sim::activity::CURVE_POINTS,
        100.0 * ms / 2000.0
    );
}

/// The plan review's pin, on the golden session: building the views and asking `would_answer`
/// before every evaluate, pass step and refit of [`scripted_session`] leaves its trace — every
/// decision, answer, note, sample and RNG draw — byte-identical (the same digest).
#[test]
fn views_and_verdicts_between_the_steps_change_no_decision() {
    let t = scripted_session_with(true);
    if let Ok(path) = std::env::var("GOLDEN_TRACE_VIEWS") {
        std::fs::write(&path, &t).unwrap();
    }
    let digest = fnv1a(&t);
    println!("scripted session with views between the steps: {} lines, digest {digest:#018x}", t.lines().count());
    assert_eq!(digest, SCRIPTED_SESSION_DIGEST, "a view build or a verdict changed a decision");
}

// ---- Task 4 review round 1: the curve cache, the model generation, the fit members, the sidecar ----

/// The curve cache (review, the cost ruling): between refits a warm build recomputes the gate and
/// the fence and equals a cold build — a failed audit's rule 4, a quarantined sample's fence; a
/// refit or a policy change moves the model generation and rebuilds. Every point stays
/// `would_answer`'s verdict.
#[test]
fn the_curve_cache_recomputes_the_gate_and_the_fence() {
    use sim::activity::TeachRule as R;
    let (mut s, h) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    let c0 = calls(&h);
    let cold = |s: &mut SurrogateKs| {
        s.curve_cache = None;
        s.training_views().unwrap()
    };
    let per_point = |s: &SurrogateKs, v: &sim::TrainingViews| {
        let prov = s.model().unwrap().provenance.clone();
        for c in &v.curves {
            for (j, &r) in c.r.iter().enumerate() {
                let pos = [[-r / 2.0, 0.0, 0.0], [r / 2.0, 0.0, 0.0]];
                assert_eq!(c.refused[j], s.would_answer(&pos, &[c.z.0, c.z.1], &prov).err().map(|e| e.0), "r = {r}");
            }
        }
    };
    let v0 = s.training_views().unwrap();
    let g = s.model_generation();
    assert_eq!(v0.model_gen, g);
    // a failed audit untrusts without a refit: the warm build sees rule 4
    s.audit_failed = Some(5e-3);
    s.audit_atoms = 2;
    let warm = s.training_views().unwrap();
    assert_eq!(s.model_generation(), g, "no refit: the cache stands");
    assert!(warm.curves[0].refused.iter().all(|r| matches!(r, Some(R::Untrusted(_)))), "{:?}", warm.curves[0].refused);
    assert_eq!(warm.curves, cold(&mut s).curves);
    per_point(&s, &warm);
    s.audit_failed = None;
    // a held sample fences its geometry between refits: the warm build sees the fence
    let k = 60;
    s.quarantine.resize(s.samples(), Q_NONE);
    s.quarantine[k] = Q_HELD;
    let warm = s.training_views().unwrap();
    assert!(warm.curves[0].refused.contains(&Some(R::NearQuarantine)), "{:?}", warm.curves[0].refused);
    assert_eq!(warm.curves, cold(&mut s).curves);
    per_point(&s, &warm);
    s.quarantine[k] = Q_NONE;
    let back = s.training_views().unwrap();
    assert_eq!(back.curves, v0.curves, "the same model, the same state: the same curves");
    // a refit: a new generation, rebuilt
    let sc = h2(1.75);
    s.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
    s.refit().unwrap();
    assert!(s.model_generation() > g);
    let warm = s.training_views().unwrap();
    assert_eq!(warm.model_gen, s.model_generation());
    assert_ne!(warm.curves[0].energy, v0.curves[0].energy, "the new model's curve");
    assert_eq!(warm.curves, cold(&mut s).curves);
    per_point(&s, &warm);
    // a policy change is a new generation too (the verdicts read the policy)
    let g = s.model_generation();
    s.set_policy(FallbackPolicy { max_grade: 1.0, ..FallbackPolicy::default() });
    assert!(s.model_generation() > g);
    let warm = s.training_views().unwrap();
    per_point(&s, &warm);
    assert_eq!(calls(&h), c0, "no teacher call");
}

/// Review I2: a refit that leaves the model untrusted moves no surface epoch, but it is a new
/// model: the generation moves and the curves are the new model's.
#[test]
fn an_untrusted_refit_moves_the_model_generation_not_the_epoch() {
    let (mut s, _) = trained(1.2, 2.2, 5, FallbackPolicy::default());
    assert!(!s.trusted());
    let v = s.training_views().unwrap();
    let sc = h2(1.75);
    s.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
    s.refit().unwrap();
    assert!(!s.trusted());
    let w = s.training_views().unwrap();
    assert_eq!(w.epoch, v.epoch, "the surface did not move");
    assert!(w.model_gen > v.model_gen);
    assert_ne!(w.curves[0].energy, v.curves[0].energy);
}

/// Review ruling on the released-sample gap: `Fitted` is "in the fit that stands". A sample the
/// recovery quarantined and a fresh answer then released (between refits) reads `Unfitted`, not
/// `Fitted`, until a refit takes it in; a restored state restores the members with the model.
#[test]
fn a_released_sample_reads_unfitted_until_the_next_refit() {
    use sim::activity::SampleState as St;
    let (mut s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    let stray = dimer(1.63, [0.0, 1.0, 0.0], [0.1, 0.0, 0.0], 1, 1);
    let mut bad = morse_sample(&MorseDimer::h2(), &stray);
    bad.energy += 0.1;
    s.add_sample(bad, &stray).unwrap();
    s.refit().unwrap();
    let k = 120;
    assert!(s.is_quarantined(k), "the recovery quarantines the glitch: {:?}", s.recovery_note());
    let state = |v: &sim::TrainingViews, i: usize| v.parity_energy.iter().find(|p| p.index == i).map(|p| p.state);
    let v = s.training_views().unwrap();
    assert_eq!((state(&v, k), state(&v, 0)), (Some(St::Quarantined), Some(St::Fitted)));
    // the release `admit` does between refits
    s.quarantine[k] = Q_NONE;
    let v = s.training_views().unwrap();
    assert_eq!((state(&v, k), state(&v, 0)), (Some(St::Unfitted), Some(St::Fitted)), "not in the fit that stands");
    // the recovery's failed attempt restores the members with the model
    let saved = s.save_state();
    s.fit_members = (0, Vec::new());
    s.restore_state(saved);
    assert_eq!(state(&s.training_views().unwrap(), 0), Some(St::Fitted));
}

/// Review I4: a sidecar without the posterior covariance is refused (rules 5d, 6 and 7 cannot
/// judge its answers; `evaluate` panicked at its first consultation), by `install_sidecar` and by
/// the Trainable load, and the provider is left as it was.
#[test]
fn a_sidecar_without_the_posterior_covariance_is_refused() {
    let dir = std::env::temp_dir().join(format!("as-nocov-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("nocov.model");
    let (s, _) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    s.save_sidecar(&path).unwrap();
    let mut sc = surrogate::sidecar::read(&path).unwrap();
    sc.model.sigma_inv_chol = None;
    surrogate::sidecar::write_full(&path, &sc.model, &sc.accumulator, sc.report.as_ref(), &sc.samples, sc.reference_rule.as_deref(), &sc.quarantine, &sc.face_density, None).unwrap();
    let file = surrogate::sidecar::read(&path).expect("the reader takes has_covariance false");
    assert!(file.model.sigma_inv_chol.is_none());
    let (mut back, _) = trained(1.2, 2.2, 0, FallbackPolicy::default());
    let e = back.install_sidecar(file).unwrap_err();
    assert!(e.contains("no posterior covariance"), "{e}");
    assert!(back.model().is_none() && back.samples() == 0, "nothing installed");
    let e = Trainable::load(&mut back, &path, &h2(1.6)).unwrap_err();
    assert!(e.contains("no posterior covariance"), "{e}");
    assert!(back.model().is_none() && back.samples() == 0);
    // the same file with its covariance loads and answers
    s.save_sidecar(&path).unwrap();
    Trainable::load(&mut back, &path, &h2(1.6)).unwrap();
    assert_eq!(back.evaluate(&h2(1.6), &mut never()).unwrap().stats.source, Source::Surrogate);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Review m3: the order 5d → 4b → 6/7 pinned against `evaluate`, not only by reading — the
/// novelty on an H₂-only model asked about H₂ + H (the spread envelope off), and the class rule
/// on a model with one H–He sample.
#[test]
fn would_answer_agrees_with_evaluate_on_the_novelty_and_the_class() {
    use sim::activity::{EnvClass, TeachRule as R};
    let first = |s: &mut SurrogateKs| Trainable::take_teach_events(s).first().map(|e| (e.rule, e.atom));
    let mut s = dimer_model_three_atom_teacher();
    s.set_policy(FallbackPolicy { max_descriptor_distance: f64::INFINITY, audit_every: 0, variance_cadence: 1, ..FallbackPolicy::default() });
    // without the active set 5e never fires (γ = 0), so 5d is the first rule a third atom inside
    // r_cut meets (the module doc: ν ≈ 1e5 for H₂ + H); both paths read the same field
    s.active_set = None;
    let mut seen = std::collections::BTreeSet::new();
    for i in 0..15 {
        let d = 2.0 + 0.25 * i as f64;
        let sc = h2_plus_h(d);
        let (pos, z) = pos_z(&sc);
        let v = s.would_answer(&pos, &z, &s.provenance_for(&sc));
        Trainable::take_teach_events(&mut s);
        s.discontinuity();
        let r = s.evaluate(&sc, &mut never()).unwrap();
        match v {
            Ok(()) => assert_eq!(r.stats.source, Source::Surrogate, "d = {d}"),
            Err(e) => {
                assert_eq!(first(&mut s), Some(e), "d = {d}");
                seen.insert(format!("{:?}", e.0));
            }
        }
    }
    eprintln!("H₂ + H, spread envelope off: {seen:?}");
    assert!(seen.contains("NewEnvironment"), "5d reached: {seen:?}");
    // 4b: one H–He sample untrusts the pair class (`new_species_falls_back_and_is_learned_…`)
    let (mut s, _) = trained(1.2, 2.2, 180, FallbackPolicy { audit_every: 0, variance_cadence: 1, ..FallbackPolicy::default() });
    s.evaluate(&dimer(1.6, [0.0, 1.0, 0.0], [0.1, -0.1, 0.0], 1, 2), &mut never()).unwrap();
    s.set_reference_energy(2, -2.83);
    s.refit().unwrap();
    let sc = h2(1.6);
    let (pos, z) = pos_z(&sc);
    let v = s.would_answer(&pos, &z, &s.provenance_for(&sc));
    assert_eq!(v, Err((R::Class(EnvClass::Pair), None)));
    Trainable::take_teach_events(&mut s);
    s.discontinuity();
    s.evaluate(&sc, &mut never()).unwrap();
    assert_eq!(first(&mut s), v.err());
}

// ---- the orbital display: one producer in estimated mode (orbital-viewer plan Task 3) ----

/// What a [`Producer`] saw: the orbital requests and its resets.
#[derive(Default)]
struct ProducerLog {
    requests: Vec<sim::OrbitalViewRequest>,
    resets: usize,
    /// What the installed display interrupt said at each evaluation with the view on (the
    /// stand-in's "display work"; Task 4 re-review N1).
    polled: Vec<bool>,
}

/// A Morse teacher with a stand-in orbital producer: while a view is on, each evaluation's payload
/// has one row whose track is the producer's own continuity counter (its "tracker": `reset`
/// restarts it, as a Kohn-Sham adapter's reset does; `discontinuity` keeps it), and a stand-in
/// density (a side job needs one).
struct Producer {
    inner: Arc<Mutex<MorseDimer>>,
    view: sim::OrbitalViewRequest,
    tracker: u32,
    log: Arc<Mutex<ProducerLog>>,
    interrupt: Option<sim::DisplayInterrupt>,
}

impl Producer {
    fn new(inner: Arc<Mutex<MorseDimer>>) -> (Producer, Arc<Mutex<ProducerLog>>) {
        let log = Arc::new(Mutex::new(ProducerLog::default()));
        (Producer { inner, view: Default::default(), tracker: 0, log: log.clone(), interrupt: None }, log)
    }
}

impl ForceProvider for Producer {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        if let (false, Some(i)) = (self.view.what.is_off(), &self.interrupt) {
            self.log.lock().unwrap().polled.push(i());
        }
        let mut r = self.inner.lock().unwrap().evaluate(scene, should_abort)?;
        r.display = Some(Display::Cpu { cell: scene.cell.clone(), origin: [0.0; 3], rho: Arc::new(vec![1.0]), max: 1.0 });
        if !self.view.what.is_off() {
            self.tracker += 1;
            let row = sim::OrbitalInfo {
                spin: sim::Spin::Up, index: 0, track: self.tracker, slot: 0, energy: -0.6, eps_range: (-0.6, -0.6), occupation: 1.0, label: sim::OrbitalLabel::Index(1),
                cluster: None, rotated: false, grid_split: None, box_state: false, smeared_1s: false, residual: None, not_converged: false, l: None, touches_box: false,
            };
            r.orbitals = Some(sim::OrbitalDisplay {
                cell: scene.cell.clone(), origin: [0.0; 3], positions: scene.nuclei.iter().map(|n| n.pos).collect(), rows: vec![row],
                field: None, spin: sim::SpinSummary::default(), unoccupied: None, notes: Vec::new(), what: self.view.what.clone(),
                lone_atom: sim::LoneAtomLabels::NotLoneAtom,
            });
        }
        Ok(r)
    }
    fn name(&self) -> &'static str {
        "morse dimer"
    }
    fn reset(&mut self) {
        self.tracker = 0;
        self.log.lock().unwrap().resets += 1;
    }
    /// Keeps the tracker (a Kohn-Sham adapter's discontinuity does: matching decides).
    fn discontinuity(&mut self) {}
    fn set_orbital_view(&mut self, request: sim::OrbitalViewRequest) {
        self.log.lock().unwrap().requests.push(request.clone());
        self.view = request;
    }
    fn set_display_interrupt(&mut self, interrupt: sim::DisplayInterrupt) {
        self.interrupt = Some(interrupt);
    }
}

/// Task 4 re-review N1: the driver's display interrupt reaches the background auditor's teacher —
/// the one orbital producer — with every job, so a paused refresh's unoccupied eigensolve on the
/// auditor's thread yields its job slot to the user's next command. The stepping teacher never
/// shows orbitals and is not given one.
#[test]
fn the_auditor_teacher_polls_the_drivers_display_interrupt() {
    use sim::{OrbitalRequest, OrbitalViewRequest};
    use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
    let morse = Arc::new(Mutex::new(MorseDimer::h2()));
    let (stepping, step_log) = Producer::new(morse.clone());
    let (side, side_log) = Producer::new(morse.clone());
    let mut s = SurrogateKs::new(Box::new(stepping), FallbackPolicy::default()).with_teacher_tol(1e-6).with_auditor(Box::new(side));
    s.set_orbital_view(OrbitalViewRequest { force_every_step: true, ..OrbitalViewRequest::new(OrbitalRequest::List) });
    let pending = Arc::new(AtomicBool::new(true));
    let p = pending.clone();
    s.set_display_interrupt(Arc::new(move || p.load(AtomicOrdering::SeqCst)));
    let refresh = |s: &mut SurrogateKs| {
        assert!(s.request_display(&h2(1.6)));
        let t0 = std::time::Instant::now();
        loop {
            if let Some(d) = s.take_side_display() {
                assert!(d.orbitals.is_some(), "the refresh carries the orbitals");
                return;
            }
            assert!(t0.elapsed().as_secs_f64() < 2.0, "no refresh within 2 s");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    };
    refresh(&mut s);
    assert_eq!(side_log.lock().unwrap().polled, [true], "a command waiting: the producer's display work is told to yield");
    pending.store(false, AtomicOrdering::SeqCst);
    refresh(&mut s);
    assert_eq!(side_log.lock().unwrap().polled, [true, false], "the same interrupt, read live");
    assert!(step_log.lock().unwrap().polled.is_empty(), "the stepping teacher shows no orbitals");
}

/// Review SHOULD 12 (spec §4.3): estimated mode has exactly one orbital producer, the background
/// auditor's teacher. The stepping teacher is never given the view, so neither a model step nor a
/// fallback step carries orbitals; an audit's and a display refresh's side displays do, from the
/// one tracker — its tracks continue from one to the next and it is never reset between them.
/// `orbital_display_now` is `None` (the caller asks for a refresh). An audit is sent with the
/// rate flag clear, a paused refresh with it set (Task 4 review I1).
#[test]
fn estimated_mode_has_one_orbital_producer() {
    use sim::{OrbitalRequest, OrbitalViewRequest};
    let morse = Arc::new(Mutex::new(MorseDimer::h2()));
    let (stepping, step_log) = Producer::new(morse.clone());
    let (side, side_log) = Producer::new(morse.clone());
    let mut s = SurrogateKs::new(Box::new(stepping), FallbackPolicy { audit_every: 5, ..FallbackPolicy::default() }).with_teacher_tol(1e-6);
    s.set_reference_energy(1, MorseDimer::h2().e_atom);
    let mut rng = StdRng::seed_from_u64(4);
    for i in 0..40 {
        let sc = dimer(1.2 + i as f64 / 39.0, unit_vector(&mut rng), [0.0; 3], 1, 1);
        s.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
    }
    s.refit().expect("refit");
    assert!(s.trusted());
    let mut s = s.with_auditor(Box::new(side));
    // straight on the provider: a model step and a fallback step carry none, even with the view on
    s.set_orbital_view(OrbitalViewRequest { force_every_step: true, ..OrbitalViewRequest::new(OrbitalRequest::List) });
    let r = s.evaluate(&h2(1.6), &mut never()).unwrap();
    assert_eq!((r.stats.source, r.orbitals.is_none()), (Source::Surrogate, true), "a model step has no electrons");
    s.force_teacher_next();
    let r = s.evaluate(&h2(1.6), &mut never()).unwrap();
    assert_eq!((r.stats.source, r.orbitals.is_none()), (Source::Fallback, true), "a fallback step carries none");
    assert!(s.orbital_display_now(&mut || false).is_none());
    assert!(step_log.lock().unwrap().requests.is_empty(), "the stepping teacher never sees the view");
    // through the driver: the audits' side displays carry the orbitals
    s.set_orbital_view(OrbitalViewRequest::default());
    let mut scene = h2(1.5);
    scene.nuclei[1].vel = [2e-4, 0.0, 0.0];
    let (p, sh_d) = driven(s);
    let mut d = Dynamics::new(scene, p, 0.25, 1);
    d.apply(DynCommand::SetOrbitalView(OrbitalViewRequest::new(OrbitalRequest::List)));
    assert!(d.snapshot().orbitals.is_none(), "no orbital_display_now in estimated mode");
    let seen = step_until(&mut d, "an audit's orbitals", |d| d.snapshot().orbitals.is_some());
    assert!(seen.iter().all(|s| s.source == Source::Surrogate), "every step was the model's");
    let a = d.snapshot();
    assert_eq!(a.orbitals_side, Some(SideKind::Audit));
    let t_audit = a.orbitals.as_ref().unwrap().rows[0].track;
    assert!(t_audit >= 1);
    // no more audits; then a refresh between ticks: its payload continues the same tracker
    surrogate_of(&sh_d).set_audit_every(0);
    step_until(&mut d, "an idle auditor", |_| surrogate_of(&sh_d).auditor_idle());
    let latest = d.snapshot().orbitals.as_ref().unwrap().rows[0].track;
    let now = d.snapshot();
    assert!(surrogate_of(&sh_d).request_display(&now.scene));
    let t0 = std::time::Instant::now();
    let refresh = loop {
        let snap = d.tick(f64::INFINITY, &mut never());
        if snap.orbitals_side == Some(SideKind::Refresh) {
            break snap;
        }
        assert!(t0.elapsed().as_secs_f64() < 2.0, "no refresh within 2 s");
        std::thread::sleep(std::time::Duration::from_millis(1));
    };
    let t_refresh = refresh.orbitals.as_ref().unwrap().rows[0].track;
    println!("one producer: audit track {t_audit}, latest {latest}, refresh track {t_refresh}");
    assert_eq!(t_refresh, latest + 1, "the same tracker, continued");
    let side = side_log.lock().unwrap();
    assert_eq!(side.resets, 0, "never reset between side solves");
    // the rate flag (Task 4 review I1): an audit is sent unforced (the auditor teacher's own
    // cadence budgets its unoccupied eigensolve), a refresh forced only because the driver is
    // paused (its forwarded request is forced)
    assert!(side.requests.iter().all(|r| r.what == OrbitalRequest::List), "{:?}", side.requests);
    assert_eq!(side.requests.iter().map(|r| r.force_every_step).collect::<Vec<_>>(), [false, true], "audit unforced, paused refresh forced: {:?}", side.requests);
    assert!(step_log.lock().unwrap().requests.is_empty(), "the stepping teacher never sees the view");
    // a fallback step through the driver leaves the payload as it was
    let seq = refresh.orbitals_seq;
    surrogate_of(&sh_d).force_teacher_next();
    d.apply(DynCommand::Step(1));
    let f = d.tick(f64::INFINITY, &mut never());
    assert_eq!(f.source, Source::Fallback);
    assert_eq!(f.orbitals_seq, seq, "no payload from a fallback step");
}

// ---- pseudopotentials (M7 Task 9) ----

/// Spec §8 row 9 first failing test (spec §5.5): a model taught under one ionic policy is never
/// consulted under another — the surfaces differ by about 38 Ha per O and a bond angle — and the
/// teacher's answer from the other policy is not kept. The tag follows the physics, not the menu:
/// H has no core, so H2 under CoresOnly *is* today's surface (tag "none") and is answered.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only (generates H for the All tag); cargo test -p providers --release")]
fn provenance_refuses_model_across_policies() {
    // the All tag needs the dimer-gated H (M7 Task 11): ≈ 1 min on the GPU, ≈ 40 min on the CPU fallback — skip without one
    let Some(ctx) = gpu::GpuContext::new_headless() else { eprintln!("provenance_refuses_model_across_policies: no GPU adapter for the H dimer gate — skipped"); return };
    crate::pseudo::set_dimer_backend(crate::pseudo::DimerBackend::Gpu(ctx));
    let (mut s, h) = trained(1.2, 2.2, 40, FallbackPolicy::default()); // taught under IonicPolicy::None
    let mut cores = h2(1.6);
    cores.ionic = sim::IonicPolicy::CoresOnly;
    assert_eq!(s.evaluate(&cores, &mut never()).unwrap().stats.source, Source::Surrogate, "CoresOnly H2 = the all-electron surface");
    let mut all = h2(1.6);
    all.ionic = sim::IonicPolicy::All;
    let r = s.evaluate(&all, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback);
    assert!(r.stats.notes.iter().any(|n| n.contains("differs from the model's") && n.contains("ionic")), "{:?}", r.stats.notes);
    assert_eq!(calls(&h), 1);
    assert_eq!(s.samples(), 40, "no sample from another policy is kept");
    assert_eq!(crate::pseudo::provenance_tag(&cores), "none");
    assert!(crate::pseudo::provenance_tag(&all).starts_with("all/"));
}

/// A Morse teacher that reports what a pseudised one would: cores in contact, or a changed shared-shell pattern.
struct Reporting {
    inner: Arc<Mutex<MorseDimer>>,
    overlap: bool,
    pattern_changed: bool,
}

impl ForceProvider for Reporting {
    fn evaluate(&mut self, s: &Scene, a: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        let mut r = self.inner.lock().unwrap().evaluate(s, a)?;
        if self.overlap {
            r.stats.core_overlaps = vec![(0, 1, 0.5)];
            r.stats.warnings.push(format!("{}: 0–1 at R = 0.500 Bohr", sim::CORE_OVERLAP_WARNING));
        }
        r.stats.occupation_changed = self.pattern_changed;
        Ok(r)
    }

    fn name(&self) -> &'static str {
        "reporting Morse"
    }

    fn reset(&mut self) {}
}

/// Spec §5.5: a teacher answer with a core-overlap warning is used, never trained on; so is one whose
/// shared-shell pattern changed since the previous step (Task 6 review A.6). An ordinary answer is kept.
#[test]
fn core_overlap_answers_are_never_kept() {
    for (overlap, pattern_changed, why) in [(true, false, "cores overlap"), (false, true, "occupation pattern changed"), (false, false, "")] {
        let teacher = Arc::new(Mutex::new(MorseDimer::h2()));
        let mut s = SurrogateKs::new(Box::new(Reporting { inner: teacher, overlap, pattern_changed }), FallbackPolicy::default());
        let r = s.evaluate(&h2(1.6), &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback);
        if why.is_empty() {
            assert_eq!(s.samples(), 1, "an ordinary answer is kept");
            continue;
        }
        assert_eq!(s.samples(), 0, "{why}: never stored");
        assert!(r.stats.notes.iter().any(|n| n.contains(why) && n.contains("sample not kept")), "{:?}", r.stats.notes);
    }
}

/// The activity row of an unfit answer says so (the Activity tab's plain wording is `TeachOutcome::UnfitNotKept`).
#[test]
fn an_unfit_answer_leaves_its_own_activity_row() {
    let teacher = Arc::new(Mutex::new(MorseDimer::h2()));
    let mut s = SurrogateKs::new(Box::new(Reporting { inner: teacher, overlap: true, pattern_changed: false }), FallbackPolicy::default());
    s.evaluate(&h2(1.6), &mut never()).unwrap();
    let ev = s.take_events().into_iter().last().expect("an activity row");
    assert_eq!(ev.outcome, TeachOutcome::UnfitNotKept);
    assert!(!ev.outcome.is_kept());
}

/// Task 8b ruling (b) and Task 8a: the labels of a pseudopotential surface carry the GPU path's scatter
/// (about 5e-5 Ha/Bohr per component), so a model fitted on them reports at least that force noise — however clean the
/// samples look — while the all-electron surface keeps its evidence-estimated noise (bitwise as before).
#[test]
fn a_pseudo_surface_floors_the_force_noise_prior() {
    let fit_on = |ionic: &str| {
        let teacher = Shared::new(MorseDimer::h2());
        let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_teacher_tol(1e-6);
        s.set_reference_energy(1, MorseDimer::h2().e_atom);
        let mut rng = StdRng::seed_from_u64(4);
        for i in 0..40 {
            let r = 1.2 + 1.0 * i as f64 / 39.0;
            let sc = dimer(r, unit_vector(&mut rng), [0.0; 3], 1, 1);
            let mut prov = s.provenance_for(&sc);
            prov.ionic = ionic.to_string();
            s.add_sample_with_provenance(morse_sample(&MorseDimer::h2(), &sc), &prov).unwrap();
        }
        s.refit().expect("refit").sigma_f
    };
    let clean = fit_on("none");
    let pseudo = fit_on("cores/0123456789abcdef/shared1e-3");
    println!("sigma_F: all-electron {clean:.2e}, pseudo {pseudo:.2e}");
    assert!(clean < PSEUDO_FORCE_NOISE / 10.0, "exact Morse labels: the evidence estimate is tiny, got {clean:e}");
    assert!((pseudo - PSEUDO_FORCE_NOISE).abs() <= 1e-9, "the floor binds on the pseudo surface: {pseudo:e}");
    assert!(PSEUDO_FORCE_NOISE >= 4.7e-5, "the measured GPU path scatter (task-8b review ruling (b))");
}

/// Task 8b ruling (b): a force audit of a pseudo-scene model is never judged against a tier tighter than
/// 3 sigma_F of the label noise; the default tier (2e-3) is far above it, so the floor is inert there and binds
/// only on a tightened policy. The all-electron surface has no floor.
#[test]
fn force_audits_do_not_judge_a_pseudo_model_below_its_label_noise() {
    let tight = FallbackPolicy { force_std_abs: 1e-5, force_std_rel: 0.0, ..FallbackPolicy::default() };
    let teacher = [[0.0, 0.0, 0.0]];
    let model = [[1.0e-4, 0.0, 0.0]]; // a 1e-4 Ha/Bohr disagreement: label noise on the pseudo path
    let none = worst_force_residual(&tight, &model, &teacher, audit_force_floor("none")).unwrap();
    assert!(none.0 > none.1, "the tight tier fails the all-electron model: {none:?}");
    let pseudo = worst_force_residual(&tight, &model, &teacher, audit_force_floor("cores/0123456789abcdef/shared1e-3")).unwrap();
    assert!(pseudo.0 <= pseudo.1, "within 3 sigma_F of the noise prior: {pseudo:?}");
    assert_eq!(pseudo.1, 3.0 * PSEUDO_FORCE_NOISE);
    // the default policy is already far above the noise
    let d = FallbackPolicy::default();
    assert!(d.force_std_abs >= 40.0 * PSEUDO_FORCE_NOISE && audit_force_floor("cores/x") < d.force_std_abs);
}

/// The surrogate answers for its teacher's ionic surface and forwards the partial-core display to it.
#[test]
fn the_surrogate_forwards_the_ionic_tag_and_the_core_display_to_its_teacher() {
    struct Flags(Arc<Mutex<bool>>);
    impl ForceProvider for Flags {
        fn evaluate(&mut self, _: &Scene, _: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
            Err(ProviderError::Cancelled)
        }
        fn name(&self) -> &'static str {
            "flags"
        }
        fn reset(&mut self) {}
        fn set_display_partial_core(&mut self, on: bool) {
            *self.0.lock().unwrap() = on;
        }
        fn ionic_tag(&self, _: &Scene) -> String {
            "teacher's surface".into()
        }
    }
    let seen = Arc::new(Mutex::new(false));
    let mut s = SurrogateKs::new(Box::new(Flags(seen.clone())), FallbackPolicy::default());
    assert_eq!(s.ionic_tag(&h2(1.4)), "teacher's surface");
    s.set_display_partial_core(true);
    assert!(*seen.lock().unwrap());
    assert!(s.display_core, "kept for the background jobs");
}

/// MUST-1 of the Task 9 physics review: the compared provenance carries the GPU stop *rule*, never its N-dependent
/// value — two pseudo scenes of 12 (O₂) and 16 (CO₂) valence electrons get equal provenance (an Add or Remove must
/// not split a model's surface), with the nominal tolerance, and the rule in the ionic tag; the effective stops
/// themselves differ (1.2e-6 and 1.6e-6) and are recorded per answer, not compared.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only (generates C and O); cargo test -p providers --release")]
fn the_provenance_carries_the_stop_rule_not_its_value() {
    let s = SurrogateKs::new(Box::new(MorseDimer::h2()), FallbackPolicy::default()).with_teacher_tol(1e-6);
    let at = |z: &[u8]| Scene {
        nuclei: z.iter().enumerate().map(|(i, &zi)| Nucleus::new(zi, [2.3 * i as f64 - 1.15 * (z.len() as f64 - 1.0), 0.0, 0.0])).collect(),
        cell: Cell { n: 61, h: 0.20 },
        charge: 0,
        multiplicity: None,
        sigma: None,
        external_field: [0.0; 3],
        ionic: sim::IonicPolicy::CoresOnly,
    };
    let (o2, co2) = (at(&[8, 8]), at(&[8, 6, 8]));
    let (a, b) = (s.provenance_for(&o2), s.provenance_for(&co2));
    assert_eq!(a, b, "{} vs {}", a.describe(), b.describe());
    assert_eq!((a.tol, b.tol), (1e-6, 1e-6), "the nominal tolerance");
    assert!(a.ionic.starts_with("cores/") && a.ionic.ends_with("/stop2x5e-8"), "{}", a.ionic);
    assert!(a.ionic.contains("/shared1e-3/1e-1/"), "τ and τ_c are in the tag: {}", a.ionic);
    let (t12, t16) = (crate::pseudo::effective_gpu_tol(1e-6, 12), crate::pseudo::effective_gpu_tol(1e-6, 16));
    assert!((t12 - 1.2e-6).abs() < 1e-15 && (t16 - 1.6e-6).abs() < 1e-15, "{t12} {t16}");
}

/// A teacher that is never called: only its name (the provenance's) matters (`inspect_saved_model`).
struct NamedStub(&'static str);

impl ForceProvider for NamedStub {
    fn evaluate(&mut self, _scene: &Scene, _should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        Err(ProviderError::Invalid("inspection only: the stub teacher never solves".into()))
    }
    fn name(&self) -> &'static str {
        self.0
    }
    fn reset(&mut self) {}
}

/// E0 of the 2026-10-18 water investigation: the inspection report of a saved `.model` file —
/// `SURROGATE_INSPECT=<path> cargo test -p providers --release inspect_saved_model -- --ignored
/// --nocapture`. Installs the sidecar under a stub teacher of the file's teacher name (no solve),
/// then prints `SurrogateKs::inspect` with probes at each composition's lowest-energy stored
/// sample and, for an O/H model, the preset water at the file's grid. Report-only.
#[test]
#[ignore = "report: SURROGATE_INSPECT=<file.model> cargo test -p providers --release inspect_saved_model -- --ignored --nocapture"]
fn inspect_saved_model() {
    let Ok(path) = std::env::var("SURROGATE_INSPECT") else {
        println!("SURROGATE_INSPECT not set: nothing to inspect");
        return;
    };
    let sc = surrogate::sidecar::read(std::path::Path::new(&path)).unwrap_or_else(|e| panic!("{path}: {e}"));
    let prov = sc.model.provenance.clone();
    let name: &'static str = Box::leak(prov.teacher.clone().into_boxed_str());
    let mut s = SurrogateKs::new(Box::new(NamedStub(name)), FallbackPolicy::default()).with_teacher_tol(prov.tol);
    s.install_sidecar(sc).unwrap_or_else(|e| panic!("install {path}: {e}"));
    let ionic = if prov.ionic.starts_with("cores/") {
        sim::IonicPolicy::CoresOnly
    } else if prov.ionic.starts_with("all/") {
        sim::IonicPolicy::All
    } else {
        sim::IonicPolicy::None
    };
    let scene_of = |pos: &[[f64; 3]], z: &[u8]| Scene {
        nuclei: pos.iter().zip(z).map(|(p, &zi)| Nucleus::new(zi, *p)).collect(),
        cell: Cell { n: prov.grid_n, h: prov.h },
        charge: prov.charge,
        multiplicity: prov.multiplicity,
        sigma: Some(prov.sigma),
        external_field: [0.0; 3],
        ionic,
    };
    let mut probes: Vec<(String, Scene)> = Vec::new();
    let mut lowest: BTreeMap<Vec<u8>, usize> = BTreeMap::new();
    let samples = &s.store().expect("a store").samples;
    for (i, smp) in samples.iter().enumerate() {
        let mut key = smp.z.clone();
        key.sort_unstable();
        let e = lowest.entry(key).or_insert(i);
        if smp.energy < samples[*e].energy {
            *e = i;
        }
    }
    for (key, i) in &lowest {
        probes.push((format!("lowest-energy stored sample #{i} ({key:?})"), scene_of(&samples[*i].pos, &samples[*i].z)));
    }
    // each pass's start geometry: its hot burst's first evaluation is the scene's own positions
    let mut prev_burst = false;
    for (i, smp) in samples.iter().enumerate() {
        let burst = smp.tag == SampleTag::HotBurst;
        if burst && !prev_burst {
            probes.push((format!("a pass's start geometry (first hot-burst sample #{i}, {} atoms)", smp.z.len()), scene_of(&smp.pos, &smp.z)));
        }
        prev_burst = burst;
    }
    if s.model().is_some_and(|m| m.basis.species == vec![1, 8]) {
        probes.push(("the preset water (O–H 1.81 Bohr, 104.5°)".into(), scene_of(&[[0.0; 3], [1.43, 1.11, 0.0], [-1.43, 1.11, 0.0]], &[8, 1, 1])));
    }
    let refs: Vec<(&str, &Scene)> = probes.iter().map(|(w, sc)| (w.as_str(), sc)).collect();
    println!("{path}:\n{}", s.inspect(&refs));
}

// ---- the 2026-10-18 trust fixes (a)–(c): readiness, draft-guess audits, a failed audit's persistence ----

/// A surrogate trained on `n` Morse dimers r ∈ [lo, hi] with an open background auditor (the
/// `Gated` stand-in, face density 0), for the audit tests below.
fn trained_with_auditor(lo: f64, hi: f64, n: usize, policy: FallbackPolicy) -> (SurrogateKs, Arc<Mutex<MorseDimer>>) {
    let (s, h) = trained(lo, hi, n, policy);
    let open = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let auditor = Gated { inner: h.clone(), open, rho: 0.0, calls };
    (s.with_auditor(Box::new(auditor)), h)
}

/// Fix (a): "trusted" must say whether strict mode answers *here*. A dimer model with 10 learned
/// H₃ samples is trusted by the global rule while its many-body class is short of samples (rule
/// 4b): at a stored H₃ geometry the readiness refuses with the class rule — the same rule the step's
/// event carries — and names the class's worst leave-one-out sample; at a dimer it answers, as the
/// step does. The verdict `evaluate` recorded and a fresh one agree, and `status` carries it.
#[test]
fn readiness_names_the_strict_verdict_where_trust_alone_says_yes() {
    use sim::activity::{EnvClass, TeachRule as R};
    let mut s = dimer_model_three_atom_teacher();
    let mut rng = StdRng::seed_from_u64(32);
    let h3: Vec<Scene> = (0..10).map(|_| random_h3(&mut rng)).collect();
    for sc in &h3 {
        s.add_sample(pair_morse_sample(sc), sc).unwrap();
    }
    s.refit().unwrap();
    assert!(s.trusted(), "the premise: the global rule passes ({:?})", s.untrusted_reason());
    assert!(!s.class_loo()[1].trusted, "the premise: the many-body class is short of samples");
    // a stored H₃ geometry where the class rule is what refuses (not the envelope)
    let sc = h3
        .iter()
        .find(|sc| {
            let (pos, z) = pos_z(sc);
            s.first_refusal(&pos, &z, &s.provenance_for(sc)).is_some_and(|r| r.rule == R::Class(EnvClass::ManyBody))
        })
        .expect("a stored H₃ geometry refused by the class rule")
        .clone();
    let fresh = s.readiness(&sc).expect("a model");
    println!("readiness at a stored H₃ geometry: {fresh:?}");
    assert!(!fresh.answers && fresh.rule == Some(R::Class(EnvClass::ManyBody)), "{fresh:?}");
    assert!(fresh.text.starts_with("three-or-more-atom (many-body) environments untrusted (everywhere) — worst sample #") && fresh.text.contains("a sampling-pass pair scan point"), "the class and its worst sample are named: {}", fresh.text);
    let st = Trainable::status(&s, &sc, &[[0.0; 3]; 3]);
    assert!(st.trusted && st.readiness.as_ref() == Some(&fresh), "{:?}", st.readiness);
    // the step: a fallback under the same rule, and the recorded verdict is the fresh one
    s.discontinuity();
    let r = s.evaluate(&sc, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback);
    let ev = drain(&mut s);
    assert_eq!(ev.last().map(|e| e.rule), Some(R::Class(EnvClass::ManyBody)), "{ev:?}");
    assert_eq!(s.readiness(&sc).map(|x| x.rule), Some(fresh.rule), "recorded = fresh");
    // a dimer: answered, and the readiness says so
    let d = in_cell_65(h2(1.4));
    let r = s.evaluate(&d, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Surrogate, "{:?}", r.stats.notes);
    let here = s.readiness(&d).unwrap();
    assert!(here.answers && here.rule.is_none(), "{here:?}");
    // draft mode does not change the strict verdict: it is what strict mode would do
    s.set_draft(true);
    let r = s.evaluate(&sc, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Surrogate, "draft mode answers");
    assert_eq!(s.readiness(&sc).map(|x| x.rule), Some(fresh.rule), "the strict verdict under draft mode");
}

/// Fix (b): a draft-mode audit at a geometry strict mode refuses (a guess's excursion) is kept as a
/// `DraftAudit` sample held out of the strict fit (`Q_DRAFT`: not fitted, not in the class
/// leave-one-out), and a failed one is recorded without untrusting the strict model; a draft-mode
/// audit where strict mode answers is an ordinary `Audit` sample and is fitted. The held-out code
/// survives the sidecar through its tag.
#[test]
fn a_draft_guess_audit_is_held_out_and_never_untrusts_the_strict_model() {
    let (mut s, h) = trained_with_auditor(1.2, 2.2, 40, FallbackPolicy { audit_every: 1, never_fallback: true, ..FallbackPolicy::default() });
    let loo0 = s.class_loo().to_vec();
    // a guess: r = 3.2, outside the trained window — strict mode refuses there
    let far = h2(3.2);
    assert!(s.readiness(&far).is_some_and(|r| !r.answers), "the premise: strict mode refuses at 3.2");
    let r = s.evaluate(&far, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Surrogate, "draft mode answers");
    let t0 = Instant::now();
    while s.last_audit().is_none() {
        assert!(t0.elapsed().as_secs_f64() < 2.0, "no audit verdict within 2 s");
        std::thread::sleep(std::time::Duration::from_millis(2));
        let _ = s.evaluate(&far, &mut never()).unwrap();
    }
    let last = s.store().unwrap().samples.last().unwrap().tag;
    println!("draft guess audit: {:?}, tag {last:?}, held out {}", s.last_audit(), s.held_out());
    assert_eq!((last, s.held_out()), (SampleTag::DraftAudit, 1));
    s.refit().unwrap();
    assert_eq!(s.report().unwrap().n_samples, 40, "held out of the fit");
    assert_eq!(s.class_loo(), &loo0[..], "and of the class leave-one-out");
    // the teacher drifts: a failed guess audit is recorded, the strict model stays trusted
    h.lock().unwrap().d *= 1.1;
    let n_audits = s.solve_counts().audit;
    let t0 = Instant::now();
    while s.solve_counts().audit == n_audits || s.last_audit().is_some_and(|a| a.verdict != AuditVerdict::Failed) {
        assert!(t0.elapsed().as_secs_f64() < 2.0, "no failed audit within 2 s");
        std::thread::sleep(std::time::Duration::from_millis(2));
        let _ = s.evaluate(&far, &mut never()).unwrap();
    }
    assert_eq!(s.last_audit().map(|a| a.verdict), Some(AuditVerdict::Failed));
    assert!(s.trusted() && s.audit_failed().is_none(), "a guess's failure is not a verdict on the strict model: {:?}", s.untrusted_reason());
    h.lock().unwrap().d /= 1.1;
    // inside the window strict mode answers: a draft-mode audit there is an ordinary audit, fitted
    let near = h2(1.6);
    let (near_pos, _) = pos_z(&near);
    let t0 = Instant::now();
    // (an audit still in flight from the far geometry may land first: wait for one at `near`)
    while s.store().unwrap().samples.last().unwrap().pos != near_pos {
        assert!(t0.elapsed().as_secs_f64() < 2.0, "no audit sample at 1.6 within 2 s");
        std::thread::sleep(std::time::Duration::from_millis(2));
        let _ = s.evaluate(&near, &mut never()).unwrap();
    }
    assert_eq!(s.store().unwrap().samples.last().unwrap().tag, SampleTag::Audit);
    let held = s.held_out();
    s.refit().unwrap();
    assert_eq!(s.report().unwrap().n_samples, s.samples() - held, "every sample but the held-out guesses is fitted");
    // the sidecar: the tag carries the code
    let path = std::env::temp_dir().join(format!("trust-fix-held-out-{}.model", std::process::id()));
    s.save_sidecar(&path).unwrap();
    let mut back = SurrogateKs::new(Box::new(MorseDimer::h2()), *s.policy()).with_teacher_tol(1e-6);
    back.install_sidecar(surrogate::sidecar::read(&path).unwrap()).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!((back.held_out(), back.report().unwrap().n_samples, back.trusted()), (held, s.report().unwrap().n_samples, true));
}

/// Fix (c): a failed audit is no longer erased by the refit its own kept sample triggers. A model
/// trained on r ∈ [1.2, 1.6] is audited at a stretched bond it extrapolates to wrongly; the refit
/// with that one sample reproduces it, but its leave-one-out prediction there is the old
/// extrapolation, so the model stays untrusted with the reason; once samples around it arrive the
/// leave-one-out prediction passes the audit's own criteria and trust returns.
#[test]
fn a_failed_audit_stays_failed_until_a_refit_predicts_it_without_its_sample() {
    let (mut s, h) = trained_with_auditor(1.2, 1.6, 40, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    assert!(s.trusted());
    let teacher = |sc: &Scene| h.lock().unwrap().clone().evaluate(sc, &mut never()).unwrap();
    // the first stretched bond whose audit fails (the premise, printed)
    let (r_a, sc) = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2]
        .iter()
        .map(|&r| (r, dimer(r, [0.6, 0.8, 0.0], [0.0; 3], 1, 1)))
        .find(|(_, sc)| {
            let (pos, z) = pos_z(sc);
            let p = s.model().unwrap().predict(&pos, &z, false).unwrap();
            (teacher(sc).energy - p.energy).abs() / 2.0 > 1e-3
        })
        .expect("an extrapolation beyond the energy tier");
    let t = teacher(&sc);
    let prov = s.provenance_for(&sc);
    s.handle_answer(audit_answer(&s, &sc, prov, Ok((t.energy, t.forces.clone(), Some(0.0), None))));
    println!("audit at r = {r_a}: {:?}", s.last_audit());
    assert_eq!(s.last_audit().map(|a| a.verdict), Some(AuditVerdict::Failed));
    assert!(!s.trusted() && s.pending_refit());
    // the refit with the audited sample: still untrusted — by its leave-one-out prediction
    s.refit().unwrap();
    let notes = Trainable::take_notes(&mut s);
    println!("after the refit with the audited sample alone: trusted {}, {:?}, quarantined {}, notes {notes:?}", s.trusted(), s.untrusted_reason(), s.quarantined());
    assert_eq!(s.quarantined(), 0, "the premise: the fit reproduces the audited sample");
    assert!(!s.trusted() && s.audit_failed().is_some(), "the failed audit is not erased by its own sample's refit");
    assert!(s.untrusted_reason().unwrap().contains("untrusted until a refit predicts that geometry without the audited sample"), "{:?}", s.untrusted_reason());
    assert!(notes.iter().any(|n| n.contains("still untrusted") && n.contains("leave-one-out")), "{notes:?}");
    // neighbours arrive (fallback-like samples around the audited bond): trust returns
    let mut rng = StdRng::seed_from_u64(77);
    for k in 0..16 {
        let r = r_a - 0.4 + 0.8 * k as f64 / 15.0;
        let n = dimer(r, unit_vector(&mut rng), [0.0; 3], 1, 1);
        s.add_sample(morse_sample(&MorseDimer::h2(), &n), &n).unwrap();
    }
    s.refit().unwrap();
    let notes = Trainable::take_notes(&mut s);
    println!("after the neighbours: trusted {}, {:?}, notes {notes:?}", s.trusted(), s.untrusted_reason());
    assert!(s.trusted() && s.audit_failed().is_none(), "{:?}", s.untrusted_reason());
    assert!(notes.iter().any(|n| n.contains("the failed audit is passed by the refit's model without the audited sample (leave-one-out)")), "{notes:?}");
}

/// Fix (c), the other exit: a failed audit whose answer is a glitch (an energy 10 mHa off the
/// teacher's surface at a well-learned geometry) cannot be reproduced; the trust recovery
/// quarantines that sample, and the failure is handed to the fence — the model is trusted again
/// with that one geometry fenced, as for any sample it cannot represent.
#[test]
fn a_failed_audit_on_a_glitch_is_handed_to_the_fence() {
    let (mut s, _h) = trained_with_auditor(1.2, 2.2, 40, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    let sc = dimer(1.7, [0.0, 0.6, 0.8], [0.0; 3], 1, 1);
    let t = MorseDimer::h2().evaluate(&sc, &mut never()).unwrap();
    let prov = s.provenance_for(&sc);
    s.handle_answer(audit_answer(&s, &sc, prov, Ok((t.energy + 1e-2, t.forces.clone(), Some(0.0), None))));
    assert!(!s.trusted() && s.audit_failed().is_some());
    s.refit().unwrap();
    let notes = Trainable::take_notes(&mut s);
    println!("glitch audit after the refit: trusted {}, quarantined {}, notes {notes:?}", s.trusted(), s.quarantined());
    assert!(s.trusted() && s.audit_failed().is_none() && s.is_quarantined(40), "{:?}", s.untrusted_reason());
    assert!(notes.iter().any(|n| n.contains("the failed audit's sample was quarantined")), "{notes:?}");
}

/// An `n`-dimer H store (pair-additive Morse teacher) plus one tight H₃ triangle.
fn dimers_and_a_tight_triangle(n: usize) -> SurrogateKs {
    let mut s = SurrogateKs::new(Box::new(PairMorse(MorseDimer::h2())), FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() }).with_teacher_tol(1e-6);
    s.set_reference_energy(1, MorseDimer::h2().e_atom);
    let mut rng = StdRng::seed_from_u64(21);
    for i in 0..n {
        let sc = in_cell_65(dimer(1.0 + 4.1 * i as f64 / (n - 1) as f64, unit_vector(&mut rng), [0.0; 3], 1, 1));
        s.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
    }
    let mut sc = in_cell_65(h2(1.4));
    sc.nuclei.push(Nucleus::new(1, [0.7, 0.7, 0.0]));
    s.add_sample(pair_morse_sample(&sc), &sc).unwrap();
    s
}

/// The trust recovery's degree-cap retry (ruling S15, step 1) is kept only when the trust rule's
/// counts still hold (2026-10-18, the water investigation's E1(e): one H 0.9 Bohr from O, the
/// retry kept at n 333 < Q′ 344, every step DFT). Here: 50 samples, one tight H₃ triangle the
/// cap-4 fit cannot reproduce. The premise, on a twin store: at cap 6 the residuals pass but the
/// model is short of samples (50 < Q′ 51; one dimer fewer and the cap-6 residuals fail too, one more and
/// it is trusted) — the old rule kept exactly that. Now the retry is not kept:
/// the cap stays 4 and the model reports the residual that actually fails (the quarantine cannot
/// take the triangle out within its 10 % cap on this small store — untrusted either way, but for
/// the reason that is true). On the 80-dimer store, where cap 6 leaves enough samples, the retry
/// is kept and the model trusted.
#[test]
fn a_degree_cap_retry_that_leaves_too_few_samples_is_not_a_recovery() {
    const N_DIMERS: usize = 49;
    let mut twin = dimers_and_a_tight_triangle(N_DIMERS);
    twin.refit().unwrap();
    twin.degree_cap = Some(RECOVERY_DEGREE_CAP);
    twin.species_dirty = true;
    twin.fit_store().unwrap();
    println!("twin at cap 6: Q′ {}, residuals fail {}, short of samples {} ({:?})", twin.report().unwrap().q_prime, twin.recoverable_fail(), twin.trust_needs_samples(), twin.untrusted_reason());
    assert!(!twin.recoverable_fail() && twin.trust_needs_samples(), "the premise: the old rule would keep cap 6 with too few samples");
    let mut s = dimers_and_a_tight_triangle(N_DIMERS);
    s.refit().unwrap();
    println!("after the refit: cap {:?}, Q′ {}, trusted {} {:?}, quarantined {}", s.degree_cap, s.report().unwrap().q_prime, s.trusted(), s.untrusted_reason(), s.quarantined());
    assert_eq!(s.degree_cap, None, "the cap-6 retry is not kept");
    assert!(s.untrusted_reason().is_some_and(|r| r.contains("training energy residual")), "the reason that is true: {:?}", s.untrusted_reason());
    // with enough samples for cap 6 the retry is kept and trusted
    let mut big = dimer_model_three_atom_teacher();
    let mut sc = in_cell_65(h2(1.4));
    sc.nuclei.push(Nucleus::new(1, [0.7, 0.7, 0.0]));
    big.add_sample(pair_morse_sample(&sc), &sc).unwrap();
    big.refit().unwrap();
    assert_eq!((big.degree_cap, big.trusted()), (Some(RECOVERY_DEGREE_CAP), true), "{:?} {:?}", big.recovery_note(), big.untrusted_reason());
}

/// A `SurrogateKs` holding a saved model file's contents, over a stub teacher of its teacher's
/// name (no solve), and the provenance's ionic policy.
fn installed_from(path: &std::path::Path) -> (SurrogateKs, sim::IonicPolicy, Provenance) {
    let sc = surrogate::sidecar::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let prov = sc.model.provenance.clone();
    let name: &'static str = Box::leak(prov.teacher.clone().into_boxed_str());
    let mut s = SurrogateKs::new(Box::new(NamedStub(name)), FallbackPolicy::default()).with_teacher_tol(prov.tol);
    s.install_sidecar(sc).unwrap_or_else(|e| panic!("install {}: {e}", path.display()));
    let ionic = if prov.ionic.starts_with("cores/") {
        sim::IonicPolicy::CoresOnly
    } else if prov.ionic.starts_with("all/") {
        sim::IonicPolicy::All
    } else {
        sim::IonicPolicy::None
    };
    (s, ionic, prov)
}

/// The coordinator's requirement (2026-10-18): a model file of the current format (ASMODEL5 since
/// the shell fence's pair, final review I2, ASMODEL7 since the audit history; no new sample codes: a store of plain pass and fallback
/// samples, as every file written before the trust
/// fixes) is re-judged from its stored samples when it is installed — the per-atom classes, the
/// fenced outliers and the class verdicts recomputed from the geometries and labels, no teacher
/// call, the sample count unchanged. The store: H dimers plus H₃ triangles each with a far H₂
/// (beyond r_cut of the triangle): per geometry (the old rule) those are many-body only; per atom
/// their far H₂ atoms are pair environments, so after the install the pair class counts them.
#[test]
fn a_saved_model_is_rejudged_from_its_samples_on_load() {
    let mut s = dimer_model_three_atom_teacher();
    let mut rng = StdRng::seed_from_u64(32);
    for _ in 0..30 {
        let mut sc = random_h3(&mut rng);
        sc.nuclei.push(Nucleus::new(1, [0.0, 0.0, 10.5]));
        sc.nuclei.push(Nucleus::new(1, [0.0, 1.4, 10.5]));
        s.add_sample(pair_morse_sample(&sc), &sc).unwrap();
    }
    s.refit().unwrap();
    let rc2 = s.basis().unwrap().r_cut.powi(2);
    let mixed = s.store().unwrap().samples.iter().filter(|x| x.z.len() == 5).count();
    let per_geometry_pairs = s.store().unwrap().samples.iter().filter(|x| env_class(&neighbour_counts(&x.pos, rc2)) == Some(0)).count();
    assert_eq!((mixed, per_geometry_pairs), (30, 80), "the premise: per geometry the mixed samples are many-body only");
    let path = std::env::temp_dir().join(format!("trust-fix-rejudge-{}.model", std::process::id()));
    s.save_sidecar(&path).unwrap();
    let magic = std::fs::read(&path).unwrap();
    assert_eq!(&magic[..8], surrogate::sidecar::MAGIC_V7, "the current format (no new sample codes; version 7 since the audit history, M8a Task 1)");
    let (back, _, _) = installed_from(&path);
    let _ = std::fs::remove_file(&path);
    println!("re-judged on load: {:?}", back.class_loo());
    assert_eq!(back.samples(), s.samples(), "no sample added or dropped");
    assert_eq!(back.class_loo()[0].samples, 80 + 30, "per atom: the far H₂ of every mixed sample is pair evidence");
    // the same verdicts and numbers, `trusted_since` included: the hysteresis memory is saved with
    // the model since M8a Task 2 fix round 2 (the sidecar's `class_memory`), so the installed
    // verdicts carry the writer's generations, all behind the installing provider's own
    assert!(back.class_loo().iter().chain(s.class_loo()).all(|c| c.measured.is_none() && c.evidence == 0), "no evidence on either side");
    assert_eq!(back.class_loo(), s.class_loo(), "the same verdicts as the fit that wrote the file, since when included");
    assert!(back.class_loo().iter().all(|c| c.trusted_since.is_none_or(|g| g < back.model_generation())), "the writer's generations precede the install's");
    assert!(back.trusted());
}

// ---- round 1 of the 2026-10-18 trust review ----

/// Waits (≤ 2 s) until the background auditor has delivered `n` more audit verdicts than `before`.
fn await_audits(s: &mut SurrogateKs, sc: &Scene, before: u64, n: u64) {
    let t0 = Instant::now();
    while s.solve_counts().audit < before + n {
        assert!(t0.elapsed().as_secs_f64() < 2.0, "no audit verdict within 2 s");
        std::thread::sleep(std::time::Duration::from_millis(2));
        let _ = s.evaluate(sc, &mut never()).unwrap();
    }
}

/// MUST-1: (a) a draft-mode audit refused only by the global trust rule (4) is an ordinary audit,
/// fitted; (b) a held-out guess (refused for a reason of its geometry at dispatch) is a test set:
/// once strict mode's domain reaches it (here: samples around it widen the envelope) the refit
/// judges it by the plain prediction of the fit that never saw it, promotes it, and — the teacher's
/// label there contradicting the model — records a failed audit, so strict mode does not answer
/// there until the region is re-earned.
#[test]
fn a_held_out_guess_is_judged_when_strict_mode_reaches_it() {
    let (mut s, h) = trained_with_auditor(1.2, 2.2, 40, FallbackPolicy { audit_every: 1, never_fallback: true, ..FallbackPolicy::default() });
    // (a) untrusted by a failed audit (a glitch injected at 1.7, before its refit): an in-window
    // draft audit is refused by rule 4 only — an ordinary audit
    let g = dimer(1.7, [0.0, 0.6, 0.8], [0.0; 3], 1, 1);
    let t = MorseDimer::h2().evaluate(&g, &mut never()).unwrap();
    let prov = s.provenance_for(&g);
    s.handle_answer(audit_answer(&s, &g, prov, Ok((t.energy + 1e-2, t.forces.clone(), Some(0.0), None))));
    assert!(!s.trusted());
    let inside = h2(1.5);
    let before = s.solve_counts().audit;
    await_audits(&mut s, &inside, before, 1);
    let tag = s.store().unwrap().samples.last().unwrap().tag;
    assert_eq!((tag, s.held_out()), (SampleTag::Audit, 0), "refused only by the trust rule: an ordinary audit");
    s.refit().unwrap();
    assert!(s.trusted(), "the glitch is quarantined, trust returns: {:?}", s.untrusted_reason());
    // (b) the teacher drifts; a guess at r = 3.0 (outside the window: a geometry refusal) is held out
    h.lock().unwrap().d *= 1.1;
    let far = h2(3.0);
    assert!(s.readiness(&far).is_some_and(|r| !r.answers && r.rule.is_some_and(geometry_specific)), "the premise: a geometry refusal at 3.0");
    let before = s.solve_counts().audit;
    await_audits(&mut s, &far, before, 1);
    // (an audit still in flight lands too; wait until one at 3.0 is stored)
    let (far_pos, _) = pos_z(&far);
    let t0 = Instant::now();
    while !s.store().unwrap().samples.iter().any(|x| x.pos == far_pos) {
        assert!(t0.elapsed().as_secs_f64() < 2.0, "no sample at 3.0");
        std::thread::sleep(std::time::Duration::from_millis(2));
        let _ = s.evaluate(&far, &mut never()).unwrap();
    }
    let held_index = s.store().unwrap().samples.iter().position(|x| x.pos == far_pos).unwrap();
    assert_eq!(s.store().unwrap().samples[held_index].tag, SampleTag::DraftAudit);
    assert!(s.trusted() && s.held_out() >= 1);
    // no further audits; the one in flight lands (with the drifted label) before the teacher is
    // restored — else a truthful late answer would supersede the contradicting one (a race)
    s.set_audit_every(0);
    let t0 = Instant::now();
    while s.audit_in_flight() || !s.auditor_idle() {
        assert!(t0.elapsed().as_secs_f64() < 2.0, "the audit in flight did not land");
        std::thread::sleep(std::time::Duration::from_millis(2));
        let _ = s.evaluate(&far, &mut never()).unwrap();
    }
    h.lock().unwrap().d /= 1.1;
    s.set_draft(false);
    // strict mode's domain reaches 3.0: true-Morse samples around it
    let mut rng = StdRng::seed_from_u64(5);
    for k in 0..30 {
        let r = 2.2 + 1.2 * k as f64 / 29.0;
        let sc = dimer(r, unit_vector(&mut rng), [0.0; 3], 1, 1);
        s.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
    }
    s.refit().unwrap();
    let notes = Trainable::take_notes(&mut s);
    println!("after the domain grew: trusted {} {:?}, held out {}, quarantined {}, notes {notes:?}", s.trusted(), s.untrusted_reason(), s.held_out(), s.quarantined());
    assert_eq!(s.store().unwrap().samples[held_index].tag, SampleTag::Audit, "promoted");
    assert!(notes.iter().any(|n| n.contains("held-out draft-mode audit") && n.contains("a failed audit")), "{notes:?}");
    // the contradicting label is a failed audit; fitted, the recovery quarantines it (the model
    // cannot represent both labels) and its fence takes over — either way strict mode must not
    // answer there
    assert!(!s.trusted() || s.is_quarantined(held_index), "neither untrusted nor fenced");
    s.discontinuity();
    let r = s.evaluate(&far, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback, "{:?}", r.stats.notes);
}

/// MUST-2: a molecule parked under a class refusal (4b, the many-body class short of samples:
/// every one of its fallbacks is exempt from thinning) adds no copies of itself. 1000 teacher steps
/// at a stored H₃ geometry and 1000 at a 0.02 Bohr displacement of it: at most one sample kept,
/// and after a refit the fitted force noise s_F and rule 6's epistemic force std at a reference
/// dimer stay within 5 % (the bounds asserted; the numbers printed). The counterfactual (100 copies
/// added directly, the old behaviour at one in ten over 1000 steps) is printed: it is what the
/// copies did to the evidence.
#[test]
fn a_parked_molecule_neither_floods_the_store_nor_shrinks_the_noise() {
    let mut s = dimer_model_three_atom_teacher();
    let mut rng = StdRng::seed_from_u64(32);
    let h3: Vec<Scene> = (0..10).map(|_| random_h3(&mut rng)).collect();
    for sc in &h3 {
        s.add_sample(pair_morse_sample(sc), sc).unwrap();
    }
    s.refit().unwrap();
    let parked = h3
        .iter()
        .find(|sc| {
            let (pos, z) = pos_z(sc);
            s.first_refusal(&pos, &z, &s.provenance_for(sc)).is_some_and(|r| r.rule == sim::activity::TeachRule::Class(sim::activity::EnvClass::ManyBody))
        })
        .expect("a stored H₃ geometry refused by the class rule")
        .clone();
    let reference = in_cell_65(h2(1.4));
    let epi = |s: &SurrogateKs| {
        let (pos, z) = pos_z(&reference);
        let p = s.model().unwrap().predict(&pos, &z, true).unwrap();
        p.force_std_epistemic.unwrap().iter().fold(0.0f64, |m, x| m.max(*x))
    };
    let (n0, sf0, epi0) = (s.samples(), s.report().unwrap().sigma_f, epi(&s));
    let mut moved = parked.clone();
    moved.nuclei[2].pos[0] += 0.02;
    for sc in [&parked, &moved] {
        for i in 0..1000 {
            s.discontinuity();
            let r = s.evaluate(sc, &mut never()).unwrap();
            assert_eq!(r.stats.source, Source::Fallback, "step {i}: parked under 4b");
            if i % 25 == 24 && s.pending_refit() {
                s.refit().unwrap();
            }
        }
    }
    if s.pending_refit() {
        s.refit().unwrap();
    }
    let (n1, sf1, epi1) = (s.samples(), s.report().unwrap().sigma_f, epi(&s));
    println!("2000 parked teacher steps: samples {n0} → {n1}; s_F {sf0:.3e} → {sf1:.3e} (×{:.3}); rule-6 epistemic std at the reference {epi0:.3e} → {epi1:.3e} (×{:.3})", sf1 / sf0, epi1 / epi0);
    assert!(n1 <= n0 + 1, "a parked molecule floods the store: {n0} → {n1}");
    assert!((sf1 / sf0 - 1.0).abs() <= 0.05, "s_F moved ×{:.3}", sf1 / sf0);
    assert!((epi1 / epi0 - 1.0).abs() <= 0.05, "rule 6's std moved ×{:.3}", epi1 / epi0);
    // the counterfactual: 100 copies as samples (what one-in-ten thinning kept over 1000 steps)
    let mut cf = dimer_model_three_atom_teacher();
    for sc in &h3 {
        cf.add_sample(pair_morse_sample(sc), sc).unwrap();
    }
    cf.refit().unwrap();
    let (cf_sf0, cf_epi0) = (cf.report().unwrap().sigma_f, epi(&cf));
    for _ in 0..100 {
        cf.add_sample(pair_morse_sample(&parked), &parked).unwrap();
    }
    cf.refit().unwrap();
    println!("counterfactual, 100 copies kept: s_F ×{:.3}, rule-6 std at the reference ×{:.3}", cf.report().unwrap().sigma_f / cf_sf0, epi(&cf) / cf_epi0);
}

/// SHOULD-1: a cleared failed audit is followed by `AUDIT_BURST` audits of the next answered steps
/// (in the step here: no auditor), then the normal cadence.
#[test]
fn a_cleared_failed_audit_is_followed_by_audits_of_the_next_steps() {
    let (mut s, _h) = trained(1.2, 1.6, 40, FallbackPolicy { audit_every: 1000, ..FallbackPolicy::default() });
    let sc = dimer(2.6, [0.6, 0.8, 0.0], [0.0; 3], 1, 1);
    let t = MorseDimer::h2().evaluate(&sc, &mut never()).unwrap();
    let (pos, z) = pos_z(&sc);
    s.failed_audits.push(FailedAudit { pos, z, energy: t.energy, forces: t.forces.clone(), index: None, ionic: "none".into(), rejudges: 0, origin: AuditOrigin::Audit, pair: None });
    s.sync_audit_failed();
    assert!(!s.trusted(), "the premise: the model is off at 2.6 ({:?})", s.audit_failed());
    let mut rng = StdRng::seed_from_u64(77);
    for k in 0..16 {
        let n = dimer(2.2 + 0.8 * k as f64 / 15.0, unit_vector(&mut rng), [0.0; 3], 1, 1);
        s.add_sample(morse_sample(&MorseDimer::h2(), &n), &n).unwrap();
    }
    s.refit().unwrap();
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    drain(&mut s);
    let mut sources = Vec::new();
    for _ in 0..AUDIT_BURST + 1 {
        s.discontinuity();
        let r = s.evaluate(&h2(1.4), &mut never()).unwrap();
        // the first evaluation after an edit is the one the audit takes (a teacher step); none other refuses
        sources.push(r.stats.source);
    }
    let rules: Vec<sim::activity::TeachRule> = drain(&mut s).iter().map(|e| e.rule).collect();
    println!("after the clear: {sources:?}, {rules:?}");
    assert_eq!(rules, vec![sim::activity::TeachRule::Audit; AUDIT_BURST]);
    assert_eq!(*sources.last().unwrap(), Source::Surrogate);
}

/// SHOULD-2: a model saved while a failed audit untrusts it loads untrusted (the case is in the
/// file and re-judged on load, not counted as a refit); one saved after the case cleared loads
/// trusted.
#[test]
fn a_failed_audit_survives_the_model_file() {
    let (mut s, h) = trained_with_auditor(1.2, 1.6, 40, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    let sc = dimer(2.6, [0.6, 0.8, 0.0], [0.0; 3], 1, 1);
    let t = h.lock().unwrap().clone().evaluate(&sc, &mut never()).unwrap();
    let prov = s.provenance_for(&sc);
    s.handle_answer(audit_answer(&s, &sc, prov, Ok((t.energy, t.forces.clone(), Some(0.0), None))));
    s.refit().unwrap();
    assert!(!s.trusted() && s.failed_audits.len() == 1 && s.failed_audits[0].rejudges == 1);
    let path = std::env::temp_dir().join(format!("trust-fix-failed-audit-{}.model", std::process::id()));
    s.save_sidecar(&path).unwrap();
    let sc_file = surrogate::sidecar::read(&path).unwrap();
    assert_eq!(sc_file.failed_audits.len(), 1, "the case is in the file");
    let mut back = SurrogateKs::new(Box::new(MorseDimer::h2()), *s.policy()).with_teacher_tol(1e-6);
    back.install_sidecar(sc_file).unwrap();
    let _ = std::fs::remove_file(&path);
    println!("loaded: trusted {} {:?}", back.trusted(), back.untrusted_reason());
    assert!(!back.trusted() && back.failed_audits.len() == 1 && back.failed_audits[0].rejudges == 1, "loaded untrusted, the count unchanged");
    assert_eq!(back.audit_failed().map(|g| (g * 1e9).round()), s.audit_failed().map(|g| (g * 1e9).round()));
}

/// SHOULD-3: a failed audit not re-earned in `AUDIT_REJUDGE_LIMIT` refits stops untrusting the
/// whole model and fences its own neighbourhood (the audited geometry falls back, the window is
/// answered); the fence lifts when samples near it let a re-judge pass. A failed audit of a
/// composition the scene does not hold is fenced at once.
#[test]
fn a_failed_audit_that_is_not_re_earned_becomes_a_local_fence() {
    let (mut s, _h) = trained(1.2, 1.6, 40, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    let sc = dimer(2.6, [0.6, 0.8, 0.0], [0.0; 3], 1, 1);
    let t = MorseDimer::h2().evaluate(&sc, &mut never()).unwrap();
    let (pos, z) = pos_z(&sc);
    s.failed_audits.push(FailedAudit { pos: pos.clone(), z: z.clone(), energy: t.energy, forces: t.forces.clone(), index: None, ionic: "none".into(), rejudges: 0, origin: AuditOrigin::Audit, pair: None });
    s.sync_audit_failed();
    for k in 0..AUDIT_REJUDGE_LIMIT {
        assert!(!s.trusted(), "refit {k}: still untrusted");
        s.refit().unwrap();
    }
    let notes = Trainable::take_notes(&mut s);
    assert!(s.trusted() && s.failed_audits.is_empty() && s.audit_fences.len() == 1, "{notes:?}");
    assert!(notes.iter().any(|n| n.contains("an audit fence")), "{notes:?}");
    assert_eq!(s.audit_fence_refusal(&pos, &z).map(|r| r.rule), Some(sim::activity::TeachRule::NearQuarantine));
    assert_eq!(s.evaluate(&h2(1.4), &mut never()).unwrap().stats.source, Source::Surrogate, "the window is answered");
    // samples near it: the fence lifts at the next refit
    let mut rng = StdRng::seed_from_u64(77);
    for k in 0..16 {
        let n = dimer(2.2 + 0.8 * k as f64 / 15.0, unit_vector(&mut rng), [0.0; 3], 1, 1);
        s.add_sample(morse_sample(&MorseDimer::h2(), &n), &n).unwrap();
    }
    s.refit().unwrap();
    let notes = Trainable::take_notes(&mut s);
    assert!(s.audit_fences.is_empty() && notes.iter().any(|n| n.contains("an audit fence is lifted")), "{notes:?}");
    // another composition: fenced at the first evaluation of a scene without it
    let mut h3 = in_big_cell(h2(1.4));
    h3.nuclei.push(Nucleus::new(1, [0.0, 2.0, 0.0]));
    let (p3, z3) = pos_z(&h3);
    s.failed_audits.push(FailedAudit { pos: p3, z: z3, energy: -1.5, forces: vec![[0.0; 3]; 3], index: None, ionic: "none".into(), rejudges: 0, origin: AuditOrigin::Audit, pair: None });
    s.sync_audit_failed();
    assert!(!s.trusted());
    s.discontinuity();
    let r = s.evaluate(&h2(1.4), &mut never()).unwrap();
    assert!(s.trusted() && s.audit_fences.len() == 1, "{:?}", r.stats.notes);
    assert_eq!(r.stats.source, Source::Surrogate);
    // R1-4: the composition returns — the fence (made by the composition, not the limit) is an
    // active failed audit again (the check `evaluate` runs first; the Morse teacher solves dimers
    // only, so it is called directly here)
    let z3: Vec<u8> = h3.nuclei.iter().map(|n| n.z).collect();
    assert!(s.fence_failed_audits_off_composition(&z3));
    assert!(!s.trusted() && s.failed_audits.len() == 1 && s.audit_fences.is_empty());
    assert!(s.pending_notes.iter().any(|n| n.contains("untrust the model again")), "{:?}", s.pending_notes);
}

/// R1-2 (round 1 re-review): a molecule parked inside an audit fence adds no copies of itself.
#[test]
fn a_molecule_parked_in_an_audit_fence_adds_no_copies() {
    let (mut s, _h) = trained(1.2, 1.6, 40, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    // a fenced case at an in-window geometry (fenced directly: the lifecycle is the test above)
    let inside = h2(1.4);
    let t = MorseDimer::h2().evaluate(&inside, &mut never()).unwrap();
    let (pi, zi) = pos_z(&inside);
    s.audit_fences.push(FailedAudit { pos: pi, z: zi, energy: t.energy + 1e-2, forces: t.forces.clone(), index: None, ionic: "none".into(), rejudges: AUDIT_REJUDGE_LIMIT, origin: AuditOrigin::Audit, pair: None });
    let n0 = s.samples();
    for _ in 0..50 {
        s.discontinuity();
        let r = s.evaluate(&inside, &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback, "{:?}", r.stats.notes);
    }
    println!("50 steps parked in an audit fence: {} kept", s.samples() - n0);
    assert!(s.samples() - n0 <= 1, "copies kept inside an audit fence: {}", s.samples() - n0);
}

/// R1-1 (round 1 re-review): a molecule parked at a held-out draft guess still teaches strict mode.
/// A draft-mode audit at r = 3.0 (outside the window: held out), draft off, the molecule parked
/// there: the first rule-5 fallback is not a copy of the held-out sample (that is not the fit's
/// data) — it is kept, the envelope grows to 3.0, and the next refit promotes the held-out sample.
#[test]
fn a_parked_molecule_at_a_held_out_guess_still_teaches() {
    let (mut s, _h) = trained_with_auditor(1.2, 2.2, 40, FallbackPolicy { audit_every: 1, never_fallback: true, ..FallbackPolicy::default() });
    let far = h2(3.0);
    let (far_pos, far_z) = pos_z(&far);
    let t0 = Instant::now();
    while !s.store().unwrap().samples.iter().any(|x| x.pos == far_pos && x.tag == SampleTag::DraftAudit) {
        assert!(t0.elapsed().as_secs_f64() < 2.0, "no held-out guess at 3.0");
        std::thread::sleep(std::time::Duration::from_millis(2));
        let _ = s.evaluate(&far, &mut never()).unwrap();
    }
    s.set_audit_every(0);
    let t0 = Instant::now();
    while s.audit_in_flight() || !s.auditor_idle() {
        assert!(t0.elapsed().as_secs_f64() < 2.0, "the audit in flight did not land");
        std::thread::sleep(std::time::Duration::from_millis(2));
        let _ = s.evaluate(&far, &mut never()).unwrap();
    }
    s.set_draft(false);
    let env_before = {
        let m = s.model().unwrap().clone();
        s.envelope_outside(&m, &far_pos, &far_z).0.iter().fold(0.0f64, |w, x| w.max(*x))
    };
    let (n0, held0) = (s.samples(), s.held_out());
    assert!(held0 >= 1 && env_before > 1.0, "the premise: held out, outside the envelope ({env_before:.2})");
    for _ in 0..10 {
        s.discontinuity();
        let r = s.evaluate(&far, &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback);
    }
    let kept = s.samples() - n0;
    s.refit().unwrap();
    let env_after = {
        let m = s.model().unwrap().clone();
        s.envelope_outside(&m, &far_pos, &far_z).0.iter().fold(0.0f64, |w, x| w.max(*x))
    };
    let notes = Trainable::take_notes(&mut s);
    println!("parked at the held-out guess: {kept} kept of 10; envelope at 3.0 {env_before:.2} → {env_after:.2}; held out {held0} → {}; notes {notes:?}", s.held_out());
    assert_eq!(kept, 1, "the first fallback is kept (the held-out guess is not the fit's data), its repetitions are copies");
    assert!(env_after < env_before && env_after <= 1.0, "the envelope grew to 3.0");
    assert!(s.held_out() < held0, "the held-out guess is promoted once strict mode's domain reaches it");
}

/// SHOULD-4: the install checks the reconstructed fit membership against the file's report; a
/// release saved between a fit and the save (Q_HELD → Q_NONE without a refit) shifts the prefix,
/// and the fit is rebuilt from the stored samples instead of trusting the shifted membership.
#[test]
fn a_mismatched_fit_membership_is_rebuilt_on_load() {
    let (mut s, _h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    // a glitch, quarantined by the recovery
    let g = dimer(1.7, [0.0, 0.6, 0.8], [0.0; 3], 1, 1);
    let mut smp = morse_sample(&MorseDimer::h2(), &g);
    smp.energy += 1e-2;
    s.add_sample(smp, &g).unwrap();
    for k in 0..3 {
        let n = dimer(1.25 + 0.2 * k as f64, [1.0, 0.0, 0.0], [0.0; 3], 1, 1);
        s.add_sample(morse_sample(&MorseDimer::h2(), &n), &n).unwrap();
    }
    s.refit().unwrap();
    assert!(s.is_quarantined(40), "the premise: the glitch is quarantined");
    let fitted = s.report().unwrap().n_samples;
    // a release between the fit and the save
    s.quarantine[40] = Q_NONE;
    let path = std::env::temp_dir().join(format!("trust-fix-membership-{}.model", std::process::id()));
    s.save_sidecar(&path).unwrap();
    let mut back = SurrogateKs::new(Box::new(MorseDimer::h2()), *s.policy()).with_teacher_tol(1e-6);
    back.install_sidecar(surrogate::sidecar::read(&path).unwrap()).unwrap();
    let _ = std::fs::remove_file(&path);
    let notes = Trainable::take_notes(&mut back);
    println!("{notes:?}; report n {} (the file's fit {fitted})", back.report().unwrap().n_samples);
    assert!(notes.iter().any(|n| n.contains("did not match its report") && n.contains("rebuilt")), "{notes:?}");
    let members = (0..back.accumulated).filter(|&i| !back.excluded(i)).count();
    assert_eq!(back.report().unwrap().n_samples, members, "the rebuilt fit's membership is the store's");
}

/// SHOULD-6: the readiness recorded by `evaluate` is not reused after a quarantine flip at the same
/// fit (no model-generation bump): a fresh verdict names the fence.
#[test]
fn readiness_is_recomputed_after_a_quarantine_flip() {
    let (mut s, _h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    let sc = h2(1.5);
    s.evaluate(&sc, &mut never()).unwrap();
    assert!(s.readiness(&sc).unwrap().answers);
    // quarantine the stored sample nearest to the query, without a refit
    let (pos, z) = pos_z(&sc);
    s.ensure_desc_cache();
    let nearest = (0..s.samples())
        .map(|i| {
            let smp = &s.store().unwrap().samples[i];
            let d = s.model().unwrap().basis.clone();
            let a = describe_sample(&d, smp);
            let q = describe_values(&d, &pos, &z).unwrap();
            let env = s.envelope().unwrap();
            let dist = q.iter().fold(0.0f64, |w, qd| w.max(a.iter().map(|(sp, b)| spread_distance(env, *sp, qd, b)).fold(f64::INFINITY, f64::min)));
            (i, dist)
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    s.quarantine.resize(s.samples(), Q_NONE);
    s.quarantine[nearest.0] = Q_HELD;
    let r = s.readiness(&sc).unwrap();
    println!("after the flip (nearest #{} at {:.3} spreads): {r:?}", nearest.0, nearest.1);
    if nearest.1 <= QUARANTINE_FRACTION * s.policy().max_descriptor_distance {
        assert_eq!(r.rule, Some(sim::activity::TeachRule::NearQuarantine), "{r:?}");
    } else {
        assert!(r.answers);
    }
}

/// The `one_tail_sample_no_longer_lets_a_free_atom_in` store: 60 H₂ dimers on 1.1–2.9 Bohr plus
/// one tail sample at 5.1 (store index 60) in the 49³ cell; the fit represents the tail sample
/// (in-sample) but the fit without it does not (the pair class's leave-one-out, 1.05×).
fn tail_blocked() -> (SurrogateKs, Arc<Mutex<MorseDimer>>) {
    gap_blocked(2.9, 5.1)
}

/// 60 dimers on 1.1–`hi` Bohr plus one sample at `r_tail` (store index 60), 49³ cell. With
/// (2.0, 3.6) the lone sample beyond the gap is a pure leverage blocker: in-sample 2.8e-7 of its
/// bound, the pair class's leave-one-out 5.2× (measured).
fn gap_blocked(hi: f64, r_tail: f64) -> (SurrogateKs, Arc<Mutex<MorseDimer>>) {
    let teacher = Shared::new(MorseDimer::h2());
    let handle = teacher.handle();
    let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() }).with_teacher_tol(1e-6);
    s.set_reference_energy(1, MorseDimer::h2().e_atom);
    let mut rng = StdRng::seed_from_u64(4);
    for i in 0..61 {
        let r = if i < 60 { 1.1 + (hi - 1.1) * i as f64 / 59.0 } else { r_tail };
        let sc = in_big_cell(dimer(r, unit_vector(&mut rng), [0.0; 3], 1, 1));
        s.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
    }
    s.refit().unwrap();
    (s, handle)
}

fn bond(sc: &Scene) -> f64 {
    let (a, b) = (sc.nuclei[0].pos, sc.nuclei[1].pos);
    (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt()
}

/// Kabsch: a rigid rotation of a point set is recovered (to rounding), and it is proper.
#[test]
fn kabsch_recovers_a_rotation() {
    let mut rng = StdRng::seed_from_u64(9);
    let p: Vec<[f64; 3]> = (0..5).map(|_| std::array::from_fn(|_| rng.gen_range(-2.0..2.0))).collect();
    let c = teach::tests_api::centroid(&p);
    let p: Vec<[f64; 3]> = p.iter().map(|x| std::array::from_fn(|a| x[a] - c[a])).collect();
    let axis = nalgebra::Unit::new_normalize(nalgebra::Vector3::new(0.3, -0.5, 0.8));
    let rot = nalgebra::Rotation3::from_axis_angle(&axis, 1.1).into_inner();
    let q: Vec<[f64; 3]> = p
        .iter()
        .map(|x| {
            let v = rot * nalgebra::Vector3::new(x[0], x[1], x[2]);
            [v[0], v[1], v[2]]
        })
        .collect();
    let r = teach::tests_api::kabsch(&p, &q);
    let err = (r - rot).abs().max();
    println!("Kabsch: |R - R_true|_max = {err:.1e}, det {:.15}", r.determinant());
    assert!(err < 1e-12, "{err:e}");
    assert!((r.determinant() - 1.0).abs() < 1e-12);
}

/// The diagnosis (ruling 5): the tail sample is a leverage blocker — in-sample within its bound,
/// leave-one-out beyond — so it is offered (10 solves); the Morse + sinusoid fit the basis cannot
/// follow has its blockers refused as structural ("the model cannot represent it"), and nothing
/// is started for them.
#[test]
fn teach_diagnosis_separates_leverage_from_structure() {
    let (s, _h) = tail_blocked();
    let d = s.teach_diagnosis();
    println!("tail store: {d:?}");
    let tail = d.iter().find(|b| b.index == 60).expect("the tail sample blocks the pair class");
    assert_eq!(tail.class, 0);
    assert!(tail.loo > 1.0 && tail.in_sample <= 1.0 && tail.teachable(), "{tail:?}");
    let o = s.teach_offer().expect("an offer");
    assert!(o.blockers.iter().any(|b| b.0 == 60), "{o:?}");
    assert!(o.blockers.len() <= TEACH_MAX_BLOCKERS);
    assert_eq!(o.solves, o.blockers.len() * (TEACH_PERTURBATIONS + TEACH_PATH_POINTS));
    for (i, _, ins, _) in &o.refused {
        assert!(d.iter().any(|b| b.index == *i && b.in_sample > 1.0), "#{i} refused at in-sample {ins}");
    }
    // structure: the basis cannot follow 12 extrema on 1 Bohr
    let (mut m, _h) = trained_oscillatory(1.2, 2.2, 40, 1e-2, 6.0);
    let d = m.teach_diagnosis();
    let structural = d.iter().filter(|b| !b.teachable()).count();
    println!("Morse + sinusoid: {} blockers, {structural} structural; worst {:?}", d.len(), d.first());
    assert!(structural > 0, "{d:?}");
    let o = m.teach_offer().expect("an offer naming the refused blockers");
    for (i, _, ins, why) in &o.refused {
        if *ins > 1.0 {
            assert!(why.contains("cannot represent"), "#{i}: {why}");
        }
    }
    for b in &o.blockers {
        assert!(b.3 <= 1.0, "an offered blocker is a leverage one: {b:?}");
    }
    if o.blockers.is_empty() {
        let e = m.start_teach(&h2(1.6)).unwrap_err();
        assert!(e.starts_with("nothing to teach"), "{e}");
    }
    // hand-made verdicts
    assert!(teach::Blocker { index: 0, class: 0, loo: 1.5, in_sample: 0.9 }.teachable());
    assert!(!teach::Blocker { index: 0, class: 0, loo: 1.5, in_sample: 1.1 }.teachable());
    assert!(!teach::Blocker { index: 0, class: 0, loo: 0.9, in_sample: 0.2 }.teachable(), "not a blocker");
}

/// The job generator: per blocker at most 6 nudges + 4 path points, every geometry inside the
/// cell, farther than the thinning radius from the blocker and from each other, and no farther from
/// the rest of the data than the blocker is (review SHOULD-T2); two held-out checks; seeded. The
/// tail at 5.1 Bohr.
#[test]
fn teach_jobs_are_spread_inward_inside_the_cell_and_hold_two_out() {
    let (mut s, _h) = tail_blocked();
    let template = in_big_cell(h2(1.6));
    let jobs = s.teach_jobs(60, &template, 0x7EAC);
    let radius = THIN_FRACTION * s.policy().max_descriptor_distance;
    let blocker = s.store().unwrap().samples[60].clone();
    let d_blocker = s.data_distance(&blocker.pos, &blocker.z, 60);
    let pz = |sc: &Scene| -> (Vec<[f64; 3]>, Vec<u8>) { (sc.nuclei.iter().map(|n| n.pos).collect(), sc.nuclei.iter().map(|n| n.z).collect()) };
    let lens: Vec<f64> = jobs.iter().map(|j| bond(&j.scene)).collect();
    println!("{} jobs (radius {radius:.3}, blocker {d_blocker:.2} spreads from the rest): bonds {lens:.3?}, holdouts {:?}", jobs.len(), jobs.iter().map(|j| j.holdout).collect::<Vec<_>>());
    assert!(jobs.len() <= TEACH_PERTURBATIONS + TEACH_PATH_POINTS);
    assert!(jobs.len() >= 3, "too few geometries survive the filters: {}", jobs.len());
    assert_eq!(jobs.iter().filter(|j| j.holdout).count(), 2);
    for (a, j) in jobs.iter().enumerate() {
        assert_eq!(j.blocker, 60);
        assert!(j.scene.check_inside_cell().is_ok());
        let (p, z) = pz(&j.scene);
        let d0 = s.geometry_distance((&p, &z), (&blocker.pos, &blocker.z));
        assert!(d0 > radius, "job {a}: {d0:.3} spreads from the blocker");
        let dd = s.data_distance(&p, &z, 60);
        assert!(dd <= d_blocker, "job {a}: {dd:.2} spreads from the rest, farther than the blocker's {d_blocker:.2}");
        for k in &jobs[..a] {
            let (q, _) = pz(&k.scene);
            assert!(s.geometry_distance((&p, &z), (&q, &z)) > radius, "job {a} too close to an earlier one");
        }
    }
    // in the model's own metric the far tail is compressed (every feature → 0 at r_cut), so a nudge
    // to 5.6 Bohr is no farther from the data than the blocker at 5.1: the inward rule is a statement
    // in descriptor space, not in bond length (measured; the user's #664 — 11.3 spreads against its
    // blocker's 4.5 — is the case it removes)
    let again = s.teach_jobs(60, &template, 0x7EAC);
    assert_eq!(again.len(), jobs.len());
    assert!(again.iter().zip(&jobs).all(|(a, b)| a.scene.nuclei.iter().zip(&b.scene.nuclei).all(|(x, y)| x.pos == y.pos)), "seeded");
}

/// SHOULD-T1: the path's partner is matched atom to atom before the straight path is built — a
/// rotated copy of a molecule with its identical atoms listed in another order is aligned back onto
/// the original exactly (every atom on its own counterpart), so no path point drags two hydrogens
/// through each other; the closest pair of a matched path never drops below the endpoints'.
#[test]
fn path_partners_are_matched_atom_to_atom() {
    let p = vec![[0.0, 0.0, 0.0], [1.43, 1.11, 0.0], [-1.43, 1.11, 0.0], [0.2, -5.9, 0.3]];
    let z = [8u8, 1, 1, 1];
    let rot = nalgebra::Rotation3::from_axis_angle(&nalgebra::Vector3::z_axis(), 0.7).into_inner();
    let turned: Vec<[f64; 3]> = p.iter().map(|x| { let v = rot * nalgebra::Vector3::new(x[0], x[1], x[2]); [v[0] + 3.0, v[1] - 1.0, v[2]] }).collect();
    // the hydrogens listed in another order
    let q = vec![turned[0], turned[3], turned[1], turned[2]];
    let aligned = teach::tests_api::matched_partner(&p, &z, &q);
    let err = aligned.iter().zip(&p).map(|(a, b)| (0..3).map(|k| (a[k] - b[k]).abs()).fold(0.0, f64::max)).fold(0.0, f64::max);
    println!("matched and aligned: max |Δ| {err:.1e}");
    assert!(err < 1e-9, "{aligned:?}");
    // the unmatched order: the straight path's midpoint brings two atoms closer than either end
    let mid: Vec<[f64; 3]> = p.iter().zip(&q).map(|(a, b)| std::array::from_fn(|k| 0.5 * (a[k] + b[k]))).collect();
    let floor = teach::tests_api::closest_pair(&p);
    println!("closest pair: endpoints {floor:.3}, an unmatched midpoint {:.3}", teach::tests_api::closest_pair(&mid));
}

/// MUST-T1: the acceptance verdict. Accepted only when every class that holds samples is trusted;
/// a class short of samples whose leave-one-out passes is "needs samples" (not accepted), and a
/// short class whose leave-one-out still fails reads "not accepted" — the case the old rule called
/// accepted.
#[test]
fn a_short_class_is_never_an_acceptance() {
    use teach::tests_api::{teach_verdict, TeachVerdict};
    let class = |samples: usize, q: usize, f: f64, e: f64| {
        let refusal = super::trust::class_refusal(samples, q, f.min(1.0), f, e);
        sim::ClassLoo { class: "pair", samples, force_rms: f.min(1.0), force_max: f, energy_rms: e, energy_max: e, trusted: refusal.is_none(), refusal, calibration: 1.0, measured: None, measured_energy: None, worst_atom: None, evidence: 0, trusted_since: None, inflation: 1.0 }
    };
    let none = sim::ClassLoo { class: "many-body", samples: 0, force_rms: 0.0, force_max: 0.0, energy_rms: 0.0, energy_max: 0.0, trusted: false, refusal: Some((sim::ClassKey::Samples, 0.0, 344.0)), calibration: 1.0, measured: None, measured_energy: None, worst_atom: None, evidence: 0, trusted_since: None, inflation: 1.0 };
    assert_eq!(teach_verdict(&[class(400, 344, 0.8, 0.5), none.clone()], false, false, 0, 0, false), TeachVerdict::Accepted, "a class without samples is not involved");
    assert_eq!(teach_verdict(&[class(201, 344, 0.78, 0.48), none.clone()], false, false, 0, 0, false), TeachVerdict::NeedsSamples);
    assert_eq!(teach_verdict(&[class(201, 344, 1.6, 0.48), none.clone()], false, false, 0, 0, false), TeachVerdict::NotAccepted, "short AND leave-one-out beyond 1×");
    assert_eq!(teach_verdict(&[class(400, 344, 0.8, 0.5)], false, false, 1, 0, false), TeachVerdict::NotAccepted, "a failed check");
    assert_eq!(teach_verdict(&[class(400, 344, 0.8, 0.5)], false, false, 0, 1, false), TeachVerdict::NotAccepted, "a blocker with no check judged near it (R3-2)");
    assert_eq!(teach_verdict(&[class(400, 344, 0.8, 0.5)], true, false, 0, 0, false), TeachVerdict::NotAccepted, "a target still blocks");
    assert_eq!(teach_verdict(&[class(400, 344, 0.8, 0.5)], false, true, 0, 0, false), TeachVerdict::NotAccepted, "a target left the fit");
    assert_eq!(teach_verdict(&[class(400, 344, 0.8, 0.5)], false, false, 0, 0, true), TeachVerdict::Cancelled);
    // M8a Task 2: a class the calibrated verdict refuses (its leave-one-out passes, its audits do not) is not accepted
    let refused = sim::ClassLoo { trusted: false, refusal: Some((sim::ClassKey::Calibrated, 1.4, 1.0)), calibration: 3.5, evidence: 20, ..class(400, 344, 0.8, 0.5) };
    assert_eq!(teach_verdict(&[refused], false, false, 0, 0, false), TeachVerdict::NotAccepted, "refused by the calibrated verdict");
}

/// MUST-T1 end to end: a round that leaves its class short of samples is not accepted, and the
/// outcome says what is missing. The gap fixture's pair class with Q′ raised past its count
/// (`class_refusal` reads the report's Q′): the targets are predicted, the class is short.
#[test]
fn a_round_that_leaves_its_class_short_is_not_accepted() {
    let (mut s, _h) = gap_blocked(2.0, 3.6);
    s.start_teach(&in_big_cell(h2(1.6))).unwrap();
    while Trainable::pass_step(&mut s, &mut never()).unwrap().is_some() {}
    assert!(s.user_notes.iter().any(|n| n.starts_with("teaching around #60: accepted")), "premise: the round is accepted at Q′ {}", s.report().unwrap().q_prime);
    // the same state judged at a higher floor: the class is short, its leave-one-out still passes
    let n = s.class_loo()[0].samples;
    s.report.as_mut().unwrap().q_prime = n + 10;
    s.refresh_class_loo();
    assert!(matches!(s.class_loo()[0].refusal, Some((sim::ClassKey::Samples, ..))), "{:?}", s.class_loo()[0]);
    let progress = sim::SamplingProgress { done: 9, total: 9, stage: "done", last_energy: 0.0, cancelled: false, skipped: 0, failed: 0, finished: true, target: None };
    let held = s.store().unwrap().samples.last().unwrap().clone();
    let scene = Scene { nuclei: held.pos.iter().map(|p| Nucleus::new(1, *p)).collect(), ..in_big_cell(h2(1.6)) };
    let holdout = crate::sampling::Holdout { blocker: 60, scene, energy: held.energy, forces: held.forces.clone(), face: None, multiplicity: None, unfit: false };
    s.finish_teach(&[60], vec![holdout], &progress);
    let outcome = s.user_notes.last().unwrap().clone();
    println!("{outcome}");
    assert!(outcome.starts_with("teaching around #60: not accepted yet: the blockers are predicted"), "{outcome}");
    assert!(outcome.contains(&format!("they need {} samples and have {n}", n + 10)), "{outcome}");
}

/// The targeted pass on the Morse teacher (ruling 5): a lone sample beyond a gap in the data
/// (`gap_blocked(2.0, 3.6)`) blocks the pair class at 5.2× its leave-one-out; a
/// teach-around pass solves its ≈ 10 geometries (the held-out two are judged after the refit, then
/// kept for the next one), refits,
/// and the unchanged class rule — the tail sample still in the fit — passes, the checks pass, and
/// the outcome says "accepted"; afterwards nothing blocks and a second pass is refused.
#[test]
fn a_teach_around_pass_restores_the_pair_class_with_the_blocker_kept() {
    let (mut s, h) = gap_blocked(2.0, 3.6);
    assert!(!s.class_loo()[0].trusted, "premise: {:?}", s.class_loo()[0]);
    let before = s.samples();
    let calls0 = calls(&h);
    s.start_teach(&in_big_cell(h2(1.6))).unwrap();
    assert!(s.start_teach(&in_big_cell(h2(1.6))).is_err(), "one pass at a time");
    let mut stages = Vec::new();
    while let Some(p) = Trainable::pass_step(&mut s, &mut never()).unwrap() {
        stages.push((p.stage, p.target));
    }
    let outcome = s.user_notes.iter().rev().find(|n| n.starts_with("teaching around #60")).expect("an outcome note").clone();
    let solves = calls(&h) - calls0;
    println!("{solves} solves, {} stored; stages {stages:?}\n{outcome}\npair {:?}", s.samples() - before, s.class_loo()[0]);
    assert!(stages.iter().all(|&(st, t)| st == "teaching" && t == Some(60)), "{stages:?}");
    assert!(solves <= TEACH_PERTURBATIONS + TEACH_PATH_POINTS);
    assert_eq!(s.samples() - before, solves, "the held-out checks are judged, then kept (SHOULD-T4)");
    assert!(s.pending_refit(), "the kept checks wait for the next refit");
    assert!(s.store().unwrap().samples[before..].iter().all(|x| x.tag == SampleTag::Targeted));
    assert!(60 < s.accumulated && !s.excluded(60), "the blocker stays in the fit");
    assert!(outcome.contains(": accepted"), "{outcome}");
    assert!(s.class_loo()[0].trusted, "{:?}", s.class_loo()[0]);
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    assert!(s.teach_offer().is_none(), "nothing blocks any more");
    assert!(s.start_teach(&in_big_cell(h2(1.6))).is_err());
}

/// A cancelled teach-around round is not a round spent, and is not accepted.
#[test]
fn a_cancelled_teach_round_is_not_spent() {
    let (mut s, _h) = tail_blocked();
    s.start_teach(&in_big_cell(h2(1.6))).unwrap();
    Trainable::pass_step(&mut s, &mut never()).unwrap();
    Trainable::cancel_pass(&mut s);
    let outcome = s.user_notes.iter().rev().find(|n| n.starts_with("teaching around #60")).expect("an outcome").clone();
    println!("{outcome}");
    assert!(outcome.contains("cancelled"), "{outcome}");
    assert_eq!(s.teach_rounds.get(&0).copied().unwrap_or(0), 0, "the pair class's round is returned");
}

/// The honest negative (ruling 5): the tail sample at 5.1 Bohr looks like a leverage blocker
/// (in-sample 2.4e-7 of its bound, leave-one-out 1.05×), but it sits near the cutoff
/// (r_cut = 6 Bohr), where the Morse teacher still binds (V(6) ≈ −3.4e-3 Ha) while every
/// f_c-damped feature vanishes. One sample can be fitted by leverage; once its neighbourhood is
/// sampled (out to 5.4 Bohr) the fit matches the forces and shifts every energy by the tail value
/// (all residuals +1.9× the energy tier per atom, measured), so the pass is not accepted, both
/// held-out checks fail (failed audits of origin "teach check", kept with their samples' indices),
/// the outcome and the reason say so, and the next offer
/// refuses #60 as structural ("the model cannot represent it") instead of a second round.
#[test]
fn a_blocker_at_the_cutoff_turns_out_structural_and_is_reported() {
    let (mut s, _h) = tail_blocked();
    s.start_teach(&in_big_cell(h2(1.6))).unwrap();
    while Trainable::pass_step(&mut s, &mut never()).unwrap().is_some() {}
    let outcome = s.user_notes.iter().rev().find(|n| n.starts_with("teaching around #60")).expect("an outcome").clone();
    println!("{outcome}");
    assert!(outcome.contains(": not accepted"), "{outcome}");
    assert!(outcome.contains("cannot represent it"), "{outcome}");
    assert!(outcome.contains("2 off, so the model is untrusted until it predicts those geometries"), "{outcome}");
    assert!(!s.trusted());
    assert!(s.failed_audits.iter().all(|c| c.origin == AuditOrigin::TeachCheck && c.index.is_some()), "failed checks keep their samples' indices");
    assert!(outcome.contains("– the model as a whole: untrusted — max training energy residual"), "the first unmet trust condition, named: {outcome}");
    let o = s.teach_offer().expect("the offer names the refused blocker");
    println!("{} refused (listed {}): {:?}", o.refused_total, o.refused.len(), o.refused);
    assert!(o.blockers.is_empty(), "no second round on a structural class: {o:?}");
    assert!(o.refused.len() <= TEACH_MAX_BLOCKERS && o.refused_total >= o.refused.len());
    assert!(o.refused.iter().all(|r| r.2 > 1.0 && r.3.contains("cannot represent")), "{o:?}");
    assert!(s.start_teach(&in_big_cell(h2(1.6))).unwrap_err().starts_with("nothing to teach"));
}

/// `DynCommand::StartTeaching` through the driver (the app's "Teach around N blockers"): the
/// dynamics pause, the progress reads stage "teaching" with its blocker, Play is refused while it
/// runs, the pass runs to its end over ticks and its outcome reaches the snapshot's notes; a second
/// request (nothing blocks) and a provider without a surrogate are rejected in words.
#[test]
fn teaching_commands_flow_through_dynamics() {
    let (s, _h) = gap_blocked(2.0, 3.6);
    let (p, hs) = driven(s);
    let mut d = Dynamics::new(in_big_cell(h2(1.6)), p, 0.25, 1);
    d.apply(DynCommand::StartTeaching);
    let snap = d.snapshot();
    assert_eq!(snap.last_rejection, None);
    assert!(snap.paused);
    let p = snap.sampling.clone().expect("progress");
    assert_eq!((p.stage, p.target, p.done), ("teaching", Some(60), 0), "{p:?}");
    d.apply(DynCommand::Play);
    assert!(d.snapshot().last_rejection.is_some(), "Play is refused while teaching");
    let mut notes = Vec::new();
    for _ in 0..100 {
        let snap = d.tick(f64::INFINITY, &mut never());
        notes.extend(snap.notes.iter().cloned());
        if snap.sampling.as_ref().is_some_and(|p| p.finished) {
            break;
        }
    }
    let snap = d.snapshot();
    notes.extend(snap.notes.iter().cloned());
    assert!(snap.sampling.as_ref().is_some_and(|p| p.finished), "{:?}", snap.sampling);
    println!("{:?}", notes.iter().filter(|n| n.contains("teaching")).collect::<Vec<_>>());
    assert!(notes.iter().any(|n| n.contains("teaching around #60: accepted")), "{notes:?}");
    assert!(notes.iter().any(|n| n.contains("teach-around pass finished: ")), "{notes:?}");
    assert!(hs.lock().unwrap().class_loo()[0].trusted);
    d.apply(DynCommand::StartTeaching);
    let rej = d.snapshot().last_rejection.expect("nothing to teach");
    assert!(rej.starts_with("teach around the blockers: "), "{rej}");
    let mut plain = Dynamics::new(in_big_cell(h2(1.6)), Box::new(MorseDimer::h2()), 0.25, 1);
    plain.apply(DynCommand::StartTeaching);
    assert!(plain.snapshot().last_rejection.is_some_and(|r| r.contains("has no surrogate model")));
}

/// At most `TEACH_ROUNDS` rounds per environment class, whatever blocks it next: after two rounds
/// on the pair class a new blocker of that class is refused ("two teach-around rounds on this class
/// did not restore it") and nothing is started.
#[test]
fn teach_rounds_are_capped_per_class() {
    let (mut s, _h) = gap_blocked(2.0, 3.6);
    s.teach_rounds.insert(0, TEACH_ROUNDS);
    let o = s.teach_offer().expect("an offer naming the refused blocker");
    println!("{o:?}");
    assert!(o.blockers.is_empty(), "{o:?}");
    assert!(o.refused.iter().any(|r| r.0 == 60 && r.3.contains("rounds on this class")), "{o:?}");
    assert!(s.start_teach(&in_big_cell(h2(1.6))).unwrap_err().starts_with("nothing to teach"));
    s.teach_rounds.insert(0, TEACH_ROUNDS - 1);
    s.start_teach(&in_big_cell(h2(1.6))).unwrap();
    assert_eq!(s.teach_rounds.get(&0).copied(), Some(TEACH_ROUNDS), "the round is counted against the class");
}

/// SHOULD-T6: the per-class round count restarts after a regular sampling pass and once the class
/// is trusted.
#[test]
fn teach_rounds_restart_after_a_regular_pass_or_trust() {
    let (mut s, _h) = gap_blocked(2.0, 3.6);
    s.teach_rounds.insert(0, TEACH_ROUNDS);
    let plan = SamplingPlan { perturbations: 2, hot_burst_steps: 0, scan_points_per_pair: 0, ..SamplingPlan::default() };
    let mut refs = sim::AtomReferences::default();
    let scene = in_big_cell(h2(1.6));
    let prov = s.provenance_for(&scene);
    refs.seed_energy(crate::sampling::atom_key(&prov, 1), MorseDimer::h2().e_atom, None, prov.tol, &prov.teacher);
    Trainable::start_pass(&mut s, &scene, &plan, &mut refs).unwrap();
    while Trainable::pass_step(&mut s, &mut never()).unwrap().is_some() {}
    assert!(s.teach_rounds.is_empty(), "a regular pass restarts the count: {:?}", s.teach_rounds);
    let (mut t, _h) = trained(1.2, 2.2, 40, FallbackPolicy::default());
    assert!(t.class_loo()[0].trusted);
    t.teach_rounds.insert(0, 1);
    t.refit().unwrap();
    assert!(!t.teach_rounds.contains_key(&0), "a trusted class restarts its count");
}

/// SHOULD-T3/T5: the offer's words — the cost with its stall caveat and the risk sentence, and a
/// pair blocker's pair against r_cut.
#[test]
fn the_offer_states_its_cost_risk_and_the_pair_against_the_cutoff() {
    let (s, _h) = gap_blocked(2.0, 3.6);
    let o = s.teach_offer().unwrap();
    println!("{:#?}", o.notes);
    assert!(o.notes.iter().any(|n| n.contains("may stall") && n.contains("untrusted until it predicts that geometry")), "{:?}", o.notes);
    assert!(o.notes.iter().any(|n| n.starts_with("#60: H 3.60 Bohr from H (the model sees out to r_cut = ")), "{:?}", o.notes);
}

/// SHOULD-T4/T5: a failed teaching check is a failed audit with its sample's index and origin; the
/// reason names it ("a teaching check failed …") and a saved file keeps the origin.
#[test]
fn a_failed_teaching_check_is_named_and_persisted() {
    let (mut s, _h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    let sc = h2(1.7);
    let t = MorseDimer::h2().evaluate(&sc, &mut never()).unwrap();
    let (pos, z) = pos_z(&sc);
    let mut f = t.forces.clone();
    f[0][0] += 0.1;
    f[1][0] -= 0.1;
    s.failed_audits.push(FailedAudit { pos, z, energy: t.energy, forces: f, index: None, ionic: "none".into(), rejudges: 0, origin: AuditOrigin::TeachCheck, pair: None });
    s.sync_audit_failed();
    let why = s.untrusted_reason().unwrap();
    println!("{why}");
    assert!(why.starts_with("a teaching check failed (forces "), "{why}");
    let path = std::env::temp_dir().join(format!("teach-check-origin-{}.model", std::process::id()));
    s.save_sidecar(&path).unwrap();
    let (back, _, _) = installed_from(&path);
    let _ = std::fs::remove_file(&path);
    assert_eq!(back.failed_audits.first().map(|c| c.origin), Some(AuditOrigin::TeachCheck));
    assert!(back.untrusted_reason().unwrap().starts_with("a teaching check failed"), "{:?}", back.untrusted_reason());
}

/// The live case's prompt (2026-10-18): class-rule fallbacks far from the class's worst blocker
/// cannot cure it, so after `TEACH_PROMPT_STREAK` of them the offer carries a prompt; near the
/// blocker, or after a model answer, it does not.
#[test]
fn consecutive_class_fallbacks_far_from_the_blocker_prompt_the_teach_offer() {
    let (mut s, _h) = gap_blocked(2.0, 3.6);
    assert!(!s.class_loo()[0].trusted, "premise");
    assert!(s.teach_offer().unwrap().prompt.is_none());
    let here = in_big_cell(h2(1.5));
    for k in 1..=teach::TEACH_PROMPT_STREAK {
        s.discontinuity();
        let r = s.evaluate(&here, &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback);
        let p = s.teach_offer().unwrap().prompt;
        assert_eq!(p.is_some(), k >= teach::TEACH_PROMPT_STREAK, "after {k}: {p:?}");
    }
    let p = s.teach_offer().unwrap().prompt.unwrap();
    println!("{p}");
    assert!(p.contains("two-atom environments are untrusted everywhere — the worst is #60"), "{p}");
    // near the blocker: the same streak prompts nothing (fallbacks there teach its region)
    let (mut t, _h) = gap_blocked(2.0, 3.6);
    let near = in_big_cell(h2(3.6));
    for _ in 0..teach::TEACH_PROMPT_STREAK {
        t.discontinuity();
        let _ = t.evaluate(&near, &mut never()).unwrap();
    }
    let streak = t.class_streak;
    println!("near the blocker: streak {streak}, offer {:?}", t.teach_offer().map(|o| o.prompt));
    assert!(t.teach_offer().is_none_or(|o| o.prompt.is_none()));
}

/// R3-1: a pass's shell check judged after its refit. The Morse window model (1.2–2.2 Bohr) asked
/// at 5.2 Bohr: the teacher still binds there (V ≈ −7e-3 Ha) and the cutoff-damped model does not,
/// so the check fails — a named limitation and a failed audit of origin "shell check" (no sample),
/// whose reason says so; a check the model reproduces says "the model's cutoff holds".
#[test]
fn a_shell_check_names_the_truncation_or_says_the_cutoff_holds() {
    let (mut s, _h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    let n0 = s.samples();
    let far = in_big_cell(h2(5.2));
    let t = MorseDimer::h2().evaluate(&far, &mut never()).unwrap();
    let check = |energy: f64, forces: Vec<[f64; 3]>| crate::sampling::ShellCheck { pair: (1, 1), r: 5.2, r_cut: 6.0, scene: far.clone(), energy, forces, face: None, unfit: false };
    s.judge_shell_checks(vec![check(t.energy, t.forces.clone())]);
    let note = s.user_notes.last().unwrap().clone();
    println!("{note}");
    assert!(note.starts_with("sampling: H–H still interacts at r = 5.20 Bohr"), "{note}");
    assert_eq!(s.samples(), n0, "never a sample");
    // Ruling (shell → fence): fenced at once, never a global untrust
    assert!(s.failed_audits.is_empty(), "no global failed audit");
    assert_eq!(s.audit_fences.last().map(|c| (c.origin, c.index)), Some((AuditOrigin::ShellCheck, None)));
    assert!(s.trusted(), "the model stays trusted: {:?}", s.untrusted_reason());
    assert_eq!(AuditOrigin::from_word(AuditOrigin::ShellCheck.word()), AuditOrigin::ShellCheck);
    // inside 5/6 of r_cut: answered as before; in the band: refused, the limitation named
    let inside = h2(1.7); // the fixture's own 33³ cell (its provenance)
    let r = s.evaluate(&inside, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Surrogate, "{:?}", r.stats.notes);
    let (pos, z) = pos_z(&in_big_cell(h2(5.5)));
    let why = s.shell_fence_refusal(&pos, &z).expect("the pair in the shell band is refused");
    println!("{}", why.text);
    assert!(why.text.contains("H–H at 5.50 Bohr is in the cutoff shell (5.00–6.00 Bohr)"), "{}", why.text);
    let (pos, z) = pos_z(&in_big_cell(h2(4.8)));
    assert!(s.shell_fence_refusal(&pos, &z).is_none(), "below the band: not the shell fence");
    // a saved file restores the fence
    let path = std::env::temp_dir().join(format!("shell-fence-{}.model", std::process::id()));
    s.save_sidecar(&path).unwrap();
    let (back, _, _) = installed_from(&path);
    let _ = std::fs::remove_file(&path);
    let (pos, z) = pos_z(&in_big_cell(h2(5.5)));
    assert!(back.shell_fence_refusal(&pos, &z).is_some() && back.trusted());
    // a check the model reproduces: the cutoff holds
    let (mut u, _h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    let (pos, z) = pos_z(&far);
    let p = u.model().unwrap().predict(&pos, &z, false).unwrap();
    u.judge_shell_checks(vec![check(p.energy, p.forces.clone())]);
    let note = u.user_notes.last().unwrap().clone();
    println!("{note}");
    assert!(note.starts_with("sampling: the model's cutoff holds for H–H: at r = 5.20 Bohr"), "{note}");
    assert!(u.failed_audits.is_empty() && u.trusted());
}

/// Final review C1: a shell fence is structural — never lifted by a refit's or an install's
/// one-point re-judgement. A shell fence whose labels the model reproduces exactly (so the
/// re-judgement at its geometry passes) stays through an install-style re-judgement, a refit with
/// new samples and a save/load, and the band is still refused; an ordinary audit fence with the
/// same passing labels is lifted by the same re-judgement (the exemption is the shell's alone).
#[test]
fn a_shell_fence_is_never_lifted_by_a_refit_or_an_install() {
    let (mut s, _h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    let (pos, z) = pos_z(&in_big_cell(h2(5.2)));
    let p = s.model().unwrap().predict(&pos, &z, false).unwrap();
    let case = |origin, pair| FailedAudit { pos: pos.clone(), z: z.clone(), energy: p.energy, forces: p.forces.clone(), index: None, ionic: "none".into(), rejudges: AUDIT_REJUDGE_LIMIT, origin, pair };
    s.audit_fences.push(case(AuditOrigin::ShellCheck, Some((1, 1))));
    s.audit_fences.push(case(AuditOrigin::Audit, None));
    let shell_fences = |s: &SurrogateKs| s.audit_fences.iter().filter(|c| c.origin == AuditOrigin::ShellCheck).count();
    let band = pos_z(&in_big_cell(h2(5.5)));
    assert!(s.shell_fence_refusal(&band.0, &band.1).is_some(), "premise: the band is refused");
    // an install's re-judgement (not counted): the audit fence passes and is lifted, the shell fence stands
    s.rejudge_failed_audits(false);
    assert_eq!((shell_fences(&s), s.audit_fences.len()), (1, 1), "{:?}", s.user_notes);
    assert!(s.user_notes.iter().any(|n| n.contains("an audit fence is lifted")), "{:?}", s.user_notes);
    assert!(s.shell_fence_refusal(&band.0, &band.1).is_some());
    // a refit with new samples (its counted re-judgement)
    let mut rng = StdRng::seed_from_u64(9);
    for r in [1.5, 1.7, 1.9, 2.1] {
        let sc = dimer(r, unit_vector(&mut rng), [0.0; 3], 1, 1);
        s.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
    }
    s.refit().unwrap();
    assert_eq!(shell_fences(&s), 1, "{:?}", s.user_notes);
    assert!(s.shell_fence_refusal(&band.0, &band.1).is_some(), "the band is refused after the refit");
    assert!(s.trusted() && s.failed_audits.is_empty(), "the fence never untrusts: {:?}", s.untrusted_reason());
    // a save/load (the install's re-judgement again, on the file's fit)
    let path = std::env::temp_dir().join(format!("shell-fence-stands-{}.model", std::process::id()));
    s.save_sidecar(&path).unwrap();
    let magic = std::fs::read(&path).unwrap();
    assert_eq!(&magic[..8], surrogate::sidecar::MAGIC_V7, "the fence's pair needs format 5 or later (7 since the audit history)");
    let (back, _, _) = installed_from(&path);
    let _ = std::fs::remove_file(&path);
    assert_eq!(shell_fences(&back), 1);
    assert_eq!(back.audit_fences.iter().find(|c| c.origin == AuditOrigin::ShellCheck).and_then(|c| c.pair), Some((1, 1)));
    assert!(back.shell_fence_refusal(&band.0, &band.1).is_some() && back.trusted());
}

/// Final review I1: a strict run stepping through a fenced band is answered by the teacher at
/// every step — the answer drives the step and **none is fitted**: no row is added, the class
/// trust verdicts are unchanged and the model stays trusted; the note says why. The band's ends
/// (M4): 5/6·r_cut is in, r_cut is out.
#[test]
fn a_shell_fence_fallback_drives_the_step_and_is_never_fitted() {
    let (mut s, h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    let far = in_big_cell(h2(5.2));
    let t = MorseDimer::h2().evaluate(&far, &mut never()).unwrap();
    s.judge_shell_checks(vec![crate::sampling::ShellCheck { pair: (1, 1), r: 5.2, r_cut: 6.0, scene: far, energy: t.energy, forces: t.forces, face: None, unfit: false }]);
    assert!(s.audit_fences.iter().any(|c| c.origin == AuditOrigin::ShellCheck), "premise: fenced");
    let (n0, loo0, calls0) = (s.samples(), s.class_loo().to_vec(), calls(&h));
    // the fixture's own 33³ cell: a dimer of r ≤ 6 Bohr centred on the origin is inside its margin
    for r in [5.0, 5.2, 5.4, 5.6, 5.8, 5.99] {
        let res = s.evaluate(&h2(r), &mut never()).unwrap();
        assert_eq!(res.stats.source, Source::Fallback, "r = {r}: {:?}", res.stats.notes);
        assert!(res.stats.notes.iter().any(|n| n.contains("shell fence") && n.contains("not fitted")), "r = {r}: {:?}", res.stats.notes);
        assert_eq!(s.samples(), n0, "no row at r = {r}");
    }
    assert_eq!(calls(&h) - calls0, 6, "the teacher answered every step");
    assert!(!s.pending_refit, "nothing to refit");
    assert_eq!(s.class_loo(), &loo0[..], "class trust unchanged");
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    let ev: Vec<_> = s.ledger.events.iter().rev().take(6).map(|e| e.outcome).collect();
    assert!(ev.iter().all(|o| !o.is_kept()), "{ev:?}");
    // the band's ends, as the fence computes them
    let lo = crate::sampling::SCAN_CUTOFF_FRACTION * 6.0;
    for (r, inside) in [(lo, true), (lo - 1e-9, false), (6.0 - 1e-9, true), (6.0, false)] {
        let (pos, z) = pos_z(&in_big_cell(h2(r)));
        assert_eq!(s.shell_fence_refusal(&pos, &z).is_some(), inside, "r = {r}");
    }
}

/// Final review M1 (the M7 leftovers): the training views' pair curve — and `would_answer`, which
/// it composes — refuse a fenced shell band as `evaluate` does, and see an audit fence with no
/// sample quarantined. A model trained on 1.2–5.0 Bohr is trusted and answers just inside
/// [5/6 r_cut, r_cut) before the fence (the premise; the LiH case of the review); after a failing H–H shell check every curve point in the band
/// is refused, by the shell fence where nothing before it in the order refused, the points below
/// the band are unchanged, and `evaluate` hands the same band point to the teacher. Then an
/// ordinary audit fence at 1.7 Bohr (no quarantined sample: the old views skipped every fence)
/// refuses its neighbourhood on the curve.
#[test]
fn the_training_curve_refuses_a_fenced_shell_band_as_evaluate_does() {
    use sim::activity::TeachRule as R;
    let (mut s, _h) = trained(1.2, 5.0, 80, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    let rc = s.model().unwrap().basis.r_cut;
    let lo = crate::sampling::SCAN_CUTOFF_FRACTION * rc;
    let in_band = |r: f64| (lo..rc).contains(&r);
    let v0 = s.training_views().unwrap();
    let c0 = v0.curves[0].clone();
    let answered: Vec<usize> = (0..c0.r.len()).filter(|&j| in_band(c0.r[j]) && c0.refused[j].is_none()).collect();
    eprintln!("band [{lo:.2}, {rc:.2}): {} curve points, {} answered before the fence", c0.r.iter().filter(|&&r| in_band(r)).count(), answered.len());
    assert!(!answered.is_empty(), "premise: the model answers in the band before the fence: {:?}", c0.refused);
    assert_eq!(s.quarantined(), 0, "premise: no quarantined sample");
    // a failing shell check fences H–H (the teacher 0.1 Ha off the model at 5.2 Bohr)
    let far = in_big_cell(h2(5.2));
    let (pos, z) = pos_z(&far);
    let p = s.model().unwrap().predict(&pos, &z, false).unwrap();
    s.judge_shell_checks(vec![crate::sampling::ShellCheck { pair: (1, 1), r: 5.2, r_cut: rc, scene: far, energy: p.energy + 0.1, forces: p.forces, face: None, unfit: false }]);
    assert!(s.audit_fences.iter().any(|c| c.origin == AuditOrigin::ShellCheck), "premise: fenced");
    let v = s.training_views().unwrap();
    let c = &v.curves[0];
    let prov = s.model().unwrap().provenance.clone();
    for (j, &r) in c.r.iter().enumerate() {
        if in_band(r) {
            assert!(c.refused[j].is_some(), "r = {r}: the band is refused");
        } else if r < lo - 0.3 {
            // well below the band (outside the fence geometry's own quarantine radius): unchanged
            assert_eq!(c.refused[j], c0.refused[j], "r = {r}");
        }
        assert_eq!(c.refused[j], s.would_answer(&dimer_at_x(r), &[1, 1], &prov).err().map(|e| e.0), "r = {r}: the curve is would_answer's");
    }
    for &j in &answered {
        assert_eq!(c.refused[j], Some(R::NearQuarantine), "r = {}: refused by the fence", c.r[j]);
        // the shell fence names the pair's second atom; the audit fence of the same case names none (physics review Minor 1)
        assert_eq!(s.would_answer(&dimer_at_x(c.r[j]), &[1, 1], &prov), Err((R::NearQuarantine, Some(1))), "r = {}: the shell fence, not only the case's quarantine radius", c.r[j]);
    }
    // evaluate agrees at an answered band point: the teacher answers, the shell fence named
    let r = c.r[answered[0]];
    let res = s.evaluate(&h2(r), &mut never()).unwrap();
    assert_eq!(res.stats.source, Source::Fallback, "r = {r}: {:?}", res.stats.notes);
    assert!(res.stats.notes.iter().any(|n| n.contains("shell fence")), "{:?}", res.stats.notes);
    // an ordinary audit fence with nothing quarantined is on the curve too
    let (mut u, _h) = trained(1.2, 5.0, 80, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    let w0 = u.training_views().unwrap().curves[0].clone();
    let k = w0.r.iter().position(|&r| r >= 1.7).unwrap();
    assert_eq!(w0.refused[k], None, "premise: answered at 1.7 Bohr");
    let (pos, z) = pos_z(&in_big_cell(h2(w0.r[k])));
    let p = u.model().unwrap().predict(&pos, &z, false).unwrap();
    u.audit_fences.push(FailedAudit { pos, z, energy: p.energy, forces: p.forces, index: None, ionic: "none".into(), rejudges: AUDIT_REJUDGE_LIMIT, origin: AuditOrigin::Audit, pair: None });
    assert_eq!(u.quarantined(), 0);
    let w = u.training_views().unwrap();
    assert_eq!(w.curves[0].refused[k], Some(R::NearQuarantine), "the audit fence's own geometry is refused on the curve");
    let prov = u.model().unwrap().provenance.clone();
    for (j, &r) in w.curves[0].r.iter().enumerate() {
        assert_eq!(w.curves[0].refused[j], u.would_answer(&dimer_at_x(r), &[1, 1], &prov).err().map(|e| e.0), "r = {r}");
    }
}

/// Final review M2 (the M7 leftovers): a pass's shell checks are judged against that pass's own
/// refit only, and a check that is not judged is said, never dropped. (a) A pass too small to fit
/// (below Q′, no model at all): its shell check was solved but is reported "could not be judged
/// (the pass's refit did not run …) — … stays unmeasured", and nothing is fenced. (b) A pass
/// cancelled before its shell check (a model standing, the refit runs on the kept samples): "the
/// pass ended before its shell check of H–H …", nothing fenced and no "holds" verdict.
#[test]
fn shell_checks_are_judged_against_the_passs_own_refit_or_said_unmeasured() {
    let plan = SamplingPlan { perturbations: 0, hot_burst_steps: 0, scan_points_per_pair: 12, ..SamplingPlan::default() };
    // (a) no model, below Q′
    let mut s = SurrogateKs::new(Box::new(AtomMorse(MorseDimer::h2())), FallbackPolicy::default()).with_teacher_tol(1e-6);
    let mut refs = sim::AtomReferences::default();
    Trainable::start_pass(&mut s, &off_centre_h2(3.3), &plan, &mut refs).unwrap();
    assert_eq!(s.pass.as_ref().unwrap().pending_shell_checks().len(), 1, "premise: the 3.3 Bohr scan crosses 5/6 r_cut");
    while Trainable::pass_step(&mut s, &mut never()).unwrap().is_some() {}
    let notes = Trainable::take_notes(&mut s);
    println!("{}", notes.join("\n"));
    let (fittable, q_prime) = s.fittable_and_q_prime();
    assert!(s.model().is_none() && fittable < q_prime, "premise: below Q′ ({fittable} < {q_prime}), no fit");
    assert!(
        notes.iter().any(|n| n.starts_with("sampling: the shell check of H–H at r = ") && n.contains("could not be judged (the pass's refit did not run") && n.ends_with("stays unmeasured")),
        "{notes:?}"
    );
    assert!(!notes.iter().any(|n| n.contains("cutoff holds") || n.contains("still interacts")), "no verdict without the pass's fit");
    assert!(s.audit_fences.is_empty(), "nothing fenced");
    // (b) cancelled before the check: the refit runs, the unrun check is said
    let (mut t, _h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    let _ = Trainable::take_notes(&mut t);
    Trainable::start_pass(&mut t, &off_centre_h2(3.3), &plan, &mut refs).unwrap();
    let r = t.pass.as_ref().unwrap().pending_shell_checks()[0].1;
    Trainable::cancel_pass(&mut t);
    let notes = Trainable::take_notes(&mut t);
    println!("{}", notes.join("\n"));
    assert!(notes.iter().any(|n| n.contains("refit on")), "premise: the cancelled pass refits: {notes:?}");
    let want = format!("sampling: the pass ended before its shell check of H–H at r = {r:.2} Bohr — the cutoff shell of this pair stays unmeasured");
    assert!(notes.contains(&want), "{notes:?}");
    assert!(!notes.iter().any(|n| n.contains("cutoff holds") || n.contains("still interacts")));
    assert!(t.audit_fences.is_empty());
    // (c) judged against a fit with a caveat (untrusted, or without this pass's samples): fenced as before, the note says
    // the verdict may be the fit's (physics review Minor 2)
    let far = in_big_cell(h2(5.2));
    let (pos, z) = pos_z(&far);
    let p = t.model().unwrap().predict(&pos, &z, false).unwrap();
    let check = crate::sampling::ShellCheck { pair: (1, 1), r: 5.2, r_cut: 6.0, scene: far, energy: p.energy + 0.1, forces: p.forces, face: None, unfit: false };
    t.judge_shell_checks_with(vec![check], Some("judged against a fit that is not trusted"));
    let note = t.user_notes.last().unwrap();
    assert!(note.contains("still interacts") && note.ends_with("(judged against a fit that is not trusted: possibly the fit's error, not the cutoff's — the next pass's shell check measures it again)"), "{note}");
    assert!(t.audit_fences.iter().any(|c| c.origin == AuditOrigin::ShellCheck), "the fence is kept: the safe side");
}

/// An H₂ at `r` along x whose fixed atom sits at −1.5 − r/2 in the fixture's 33³ cell, so the scan's
/// moved atom (index 1, +x) stays inside the cell's margin out past 5/6 r_cut (the shell check).
fn off_centre_h2(r: f64) -> Scene {
    dimer(r, [1.0, 0.0, 0.0], [-1.5, 0.0, 0.0], 1, 1)
}

/// The training curve's dimer at distance `r` (Bohr), as `views` builds it.
fn dimer_at_x(r: f64) -> [[f64; 3]; 2] {
    [[-r / 2.0, 0.0, 0.0], [r / 2.0, 0.0, 0.0]]
}

/// The coordinator's ruling on I1 (2026-10-19): never a fitted band row — the relax verification
/// and an audit included. A relax verification at a fenced-band geometry is answered by the
/// teacher (the verification has its answer), not fitted (`ShellNotKept`) and leaves no failed
/// audit; a background audit whose geometry lies in the band is neither judged nor kept — no
/// failed audit, the model's trust untouched — even when its answer is far off the model.
#[test]
fn nothing_inside_a_fenced_band_is_fitted_or_judged() {
    let (mut s, h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    let far = in_big_cell(h2(5.2));
    let t = MorseDimer::h2().evaluate(&far, &mut never()).unwrap();
    s.judge_shell_checks(vec![crate::sampling::ShellCheck { pair: (1, 1), r: 5.2, r_cut: 6.0, scene: far, energy: t.energy, forces: t.forces.clone(), face: None, unfit: false }]);
    assert!(s.audit_fences.iter().any(|c| c.origin == AuditOrigin::ShellCheck), "premise: fenced");
    let n0 = s.samples();
    // the relax verification at a band geometry (the fixture's own cell: its provenance)
    s.force_teacher_next();
    let calls0 = calls(&h);
    let res = s.evaluate(&h2(5.5), &mut never()).unwrap();
    println!("{:?}", res.stats.notes);
    assert_eq!((res.stats.source, calls(&h)), (Source::Fallback, calls0 + 1), "the teacher answered the verification");
    assert!(res.stats.notes.iter().any(|n| n.contains("relax verification")), "{:?}", res.stats.notes);
    assert!(res.stats.notes.iter().any(|n| n.contains("shell fence") && n.contains("not fitted")), "{:?}", res.stats.notes);
    assert_eq!(s.ledger.events.back().map(|e| e.outcome), Some(TeachOutcome::ShellNotKept));
    assert_eq!(s.samples(), n0, "not fitted");
    assert!(s.failed_audits.is_empty() && s.audit_failed.is_none() && s.trusted(), "{:?}", s.untrusted_reason());
    // a background audit landing in the band, its answer 0.5 Ha off the model
    let scene = h2(5.5);
    let prov = s.provenance_for(&scene);
    let job = AuditJob { scene, prov, generation: 0, answered: 0, kind: JobKind::Audit, id: 0, view: OrbitalViewRequest::default(), interrupt: None, core: false, draft_guess: false };
    s.judge_audit(job, t.energy - 0.5, t.forces.clone(), None, None, None, 1.0, Some(10));
    let note = s.pending_notes.last().cloned().unwrap_or_default();
    println!("{note}");
    assert!(note.starts_with("surrogate audit not judged: H–H at 5.50 Bohr is in the cutoff shell") && note.ends_with("the answer is not fitted"), "{note}");
    assert_eq!(s.ledger.events.back().map(|e| e.outcome), Some(TeachOutcome::ShellNotKept));
    assert_eq!(s.samples(), n0, "not kept");
    assert!(s.failed_audits.is_empty() && s.audit_failed.is_none() && s.trusted(), "an audit in the band is no judgement: {:?}", s.untrusted_reason());
}

/// M8a Task 1: every judged audit leaves an `AuditPoint` in the history (`surrogate::history`) —
/// the background one (`judge_audit`, which now predicts **with** the variance) and the in-step one
/// — with the model's per-atom predictive force std (the total and its epistemic part) against
/// ‖F_teacher − F_model‖, the total-energy std against the signed gap, the atoms' class codes
/// (0 isolated, 1 pair, 2 many-body) and the model generation judged. A box-affected answer leaves
/// no point and is counted; the history is written with the sidecar (version 7) and installed
/// from it; a forget clears it.
#[test]
fn judged_audits_are_recorded_in_the_history() {
    let (mut s, _h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    assert!(s.audit_history().is_empty());
    let scene = h2(1.5);
    let t = MorseDimer::h2().evaluate(&scene, &mut never()).unwrap();
    let (pos, z) = pos_z(&scene);
    let pred = s.model().unwrap().predict(&pos, &z, true).unwrap();
    let prov = s.provenance_for(&scene);
    let job = || AuditJob { scene: scene.clone(), prov: prov.clone(), generation: 0, answered: 0, kind: JobKind::Audit, id: 0, view: OrbitalViewRequest::default(), interrupt: None, core: false, draft_guess: false };
    s.judge_audit(job(), t.energy, t.forces.clone(), None, None, None, 1.0, Some(10));
    assert_eq!(s.audit_history().len(), 1);
    let p = s.audit_history().iter().next().unwrap().clone();
    assert_eq!((p.kind, p.n_atoms, p.classes.clone(), p.model_gen), (surrogate::history::PointKind::Audit, 2, vec![1, 1], s.model_gen));
    assert_eq!(p.pred_force_std, pred.force_std.clone().unwrap(), "the total predictive force std, per atom");
    assert_eq!(p.pred_force_std_epistemic, pred.force_std_epistemic.clone().unwrap());
    assert_eq!(p.pred_energy_std, pred.energy_std.unwrap());
    assert_eq!(p.err_energy, t.energy - pred.energy, "signed: teacher − model");
    for (j, e) in p.err_force.iter().enumerate() {
        let d = (0..3).map(|k| (t.forces[j][k] - pred.forces[j][k]).powi(2)).sum::<f64>().sqrt();
        assert_eq!(*e, d, "‖F_teacher − F_model‖ of atom {j}");
    }
    // a box-affected answer (face density above the rule): judged "not judged", no point, counted
    s.judge_audit(job(), t.energy, t.forces.clone(), Some(1.0), None, None, 1.0, Some(10));
    assert_eq!((s.audit_history().len(), s.audit_history().unrecorded.box_affected), (1, 1));
    // a draft-mode guess at a geometry strict mode refuses: no point, counted
    s.judge_audit(AuditJob { draft_guess: true, ..job() }, t.energy, t.forces.clone(), None, None, None, 1.0, Some(10));
    assert_eq!((s.audit_history().len(), s.audit_history().unrecorded.draft), (1, 1));
    // the in-step audit (no auditor attached, audit every 2 answered steps): the second answered
    // step is audited inside `evaluate`; the cadence had reused the first verdict, so the point's
    // stds come from a variance evaluation made for the audit
    let (mut s2, _h2) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 2, ..FallbackPolicy::default() });
    let a = s2.evaluate(&h2(1.6), &mut never()).unwrap();
    assert_eq!((a.stats.source, s2.audit_history().len()), (Source::Surrogate, 0));
    let b = s2.evaluate(&h2(1.62), &mut never()).unwrap();
    assert_eq!(b.stats.source, Source::Fallback, "{:?}", b.stats.notes);
    assert!(b.stats.notes.iter().any(|n| n.contains("audit")), "{:?}", b.stats.notes);
    assert_eq!(s2.audit_history().len(), 1, "the in-step audit is recorded");
    let q = s2.audit_history().iter().next().unwrap();
    assert_eq!((q.n_atoms, q.classes.clone()), (2, vec![1, 1]));
    assert!(q.pred_force_std.iter().all(|x| x.is_finite() && *x > 0.0) && q.pred_energy_std > 0.0, "{q:?}");
    assert!(q.pred_force_std.iter().zip(&q.pred_force_std_epistemic).all(|(t, e)| t > e), "the total exceeds its epistemic part by the σ_f floor: {q:?}");
    let (pos_b, z_b) = pos_z(&h2(1.62));
    let pred_b = s2.model().unwrap().predict(&pos_b, &z_b, true).unwrap();
    assert_eq!(q.pred_force_std, pred_b.force_std.unwrap(), "the same model's stds at the audited geometry");
    // the history rides with the sidecar (version 7) and is installed from it
    let path = std::env::temp_dir().join(format!("audit-history-{}.model", std::process::id()));
    s.save_sidecar(&path).unwrap();
    let file = surrogate::sidecar::read(&path).unwrap();
    assert_eq!(&file.audit_history, s.audit_history());
    let (mut back, _hb) = trained(1.2, 2.2, 0, FallbackPolicy::default());
    back.install_sidecar(file).unwrap();
    assert_eq!(back.audit_history(), s.audit_history());
    let _ = std::fs::remove_file(&path);
    // a forget starts a new model's lifetime
    s.forget();
    assert_eq!(s.audit_history(), &surrogate::AuditHistory::default());
}

/// M8a Task 1 fix round 1 (the physics review's I3): a step the teacher answers because a rule
/// refused the model's answer leaves a **fallback** point too — flagged with the refusing rule's
/// code — with the model's stds at that geometry (evaluated for the point when the cadence had
/// reused a verdict) against the teacher's errors; a gate refusal before the model is consulted
/// (no model) leaves nothing; an unfit teacher answer is counted, not recorded. The sidecar carries
/// the kinds.
#[test]
fn fallback_steps_are_recorded_in_the_history_with_their_rule() {
    use surrogate::history::PointKind;
    // rule 1: no model → no prediction → nothing
    let (mut s0, _h0) = trained(1.2, 2.2, 0, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    let r = s0.evaluate(&h2(1.5), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback);
    assert_eq!(s0.audit_history(), &surrogate::AuditHistory::default(), "a gate refusal leaves no point and no count");
    // the envelope (rule 5): a dimer far outside the trained range
    let (mut s, _h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    let near = s.evaluate(&h2(1.6), &mut never()).unwrap();
    assert_eq!((near.stats.source, s.audit_history().len()), (Source::Surrogate, 0), "an answered step is no comparison");
    let far = h2(3.4);
    let (pos, z) = pos_z(&far);
    let pred = s.model().unwrap().predict(&pos, &z, true).unwrap();
    let t = MorseDimer::h2().evaluate(&far, &mut never()).unwrap();
    let r = s.evaluate(&far, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback, "{:?}", r.stats.notes);
    assert_eq!(s.audit_history().len(), 1, "the fallback is recorded");
    let p = s.audit_history().iter().next().unwrap();
    let rule = match p.kind {
        PointKind::Fallback { rule } => rule,
        PointKind::Audit => panic!("a fallback point"),
    };
    assert_eq!(history_rule_name(rule), "rule 5 (envelope)", "code {rule}: {:?}", r.stats.notes);
    assert_eq!(history_rule_code(TeachRule::Spread), rule);
    assert_eq!(p.pred_force_std, pred.force_std.clone().unwrap(), "the model's stds at the refused geometry");
    assert_eq!(p.pred_energy_std, pred.energy_std.unwrap());
    assert_eq!(p.err_energy, t.energy - pred.energy);
    for (j, e) in p.err_force.iter().enumerate() {
        let d = (0..3).map(|k| (t.forces[j][k] - pred.forces[j][k]).powi(2)).sum::<f64>().sqrt();
        assert_eq!(*e, d, "atom {j}");
    }
    // every code is distinct and named
    let rules = [TeachRule::Cell, TeachRule::NoModel, TeachRule::NewSpecies(1), TeachRule::OtherSettings, TeachRule::Untrusted(sim::activity::TrustKey::Samples), TeachRule::Class(sim::activity::EnvClass::Pair), TeachRule::Class(sim::activity::EnvClass::ManyBody), TeachRule::IsolatedAtom, TeachRule::Coordination, TeachRule::Spread, TeachRule::NoGrade, TeachRule::NewEnvironment, TeachRule::UnspannedAtom, TeachRule::Grade, TeachRule::ForceUncertainty, TeachRule::EnergyUncertainty, TeachRule::NonFiniteAnswer, TeachRule::NearQuarantine, TeachRule::Audit, TeachRule::RelaxVerification, TeachRule::Pass(sim::activity::PassJob::Scan)];
    let codes: std::collections::BTreeSet<u8> = rules.iter().map(|&r| history_rule_code(r)).collect();
    assert_eq!(codes.len(), rules.len(), "distinct codes");
    assert!(codes.iter().all(|&c| history_rule_name(c) != "?"));
    assert!(!codes.contains(&HISTORY_RULE_CLASS_AND_VARIANCE) && history_rule_name(HISTORY_RULE_CLASS_AND_VARIANCE) != "?", "code 21 is the provider's own, beside the rules'");
    assert_eq!(history_rule_name(22), "?");
    // the sidecar keeps the kind
    let path = std::env::temp_dir().join(format!("fallback-history-{}.model", std::process::id()));
    s.save_sidecar(&path).unwrap();
    let file = surrogate::sidecar::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(&file.audit_history, s.audit_history());
    assert_eq!(file.audit_history.iter().next().unwrap().kind, PointKind::Fallback { rule });
}

/// Final review I2: the fenced pair is the check's own. (a) A check whose point lies at or beyond
/// r_cut (a short user plan's first point past 5/6·r_cut) still fences the band for its pair —
/// the case geometry has no pair in the band, the fence's pair does. (b) A bystander pair of the
/// check's geometry that lies in the band is never fenced: a check of the pair H–He whose geometry
/// is an H–H dimer at 5.5 Bohr refuses H–He in the band and leaves H–H answered.
#[test]
fn the_shell_fence_names_the_checks_pair_not_the_case_geometry() {
    let policy = FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() };
    // (a) a check at 6.59 ≥ r_cut
    let (mut s, _h) = trained(1.2, 2.2, 40, policy.clone());
    let beyond = in_big_cell(h2(6.59));
    let (pos, z) = pos_z(&beyond);
    let p = s.model().unwrap().predict(&pos, &z, false).unwrap();
    s.judge_shell_checks(vec![crate::sampling::ShellCheck { pair: (1, 1), r: 6.59, r_cut: 6.0, scene: beyond, energy: p.energy - 0.05, forces: p.forces, face: None, unfit: false }]);
    let note = s.user_notes.last().unwrap().clone();
    println!("{note}");
    assert!(note.starts_with("sampling: H–H still interacts at r = 6.59 Bohr"), "{note}");
    assert_eq!(s.audit_fences.last().map(|c| (c.origin, c.pair)), Some((AuditOrigin::ShellCheck, Some((1, 1)))));
    let (pos, z) = pos_z(&in_big_cell(h2(5.5)));
    let why = s.shell_fence_refusal(&pos, &z).expect("the band is fenced although the check lay beyond r_cut");
    assert!(why.text.contains("H–H at 5.50 Bohr is in the cutoff shell"), "{}", why.text);
    let (pos, z) = pos_z(&in_big_cell(h2(6.59)));
    assert!(s.shell_fence_refusal(&pos, &z).is_none(), "beyond r_cut nothing is in the band");
    // (b) a bystander pair
    let (mut u, _h) = trained(1.2, 2.2, 40, policy);
    let band = in_big_cell(h2(5.5));
    let (pos, z) = pos_z(&band);
    let p = u.model().unwrap().predict(&pos, &z, false).unwrap();
    u.judge_shell_checks(vec![crate::sampling::ShellCheck { pair: (2, 1), r: 5.5, r_cut: 6.0, scene: band, energy: p.energy - 0.05, forces: p.forces, face: None, unfit: false }]);
    assert_eq!(u.audit_fences.last().map(|c| (c.origin, c.pair)), Some((AuditOrigin::ShellCheck, Some((1, 2)))), "sorted");
    assert!(u.shell_fence_refusal(&pos, &z).is_none(), "the case's own H–H pair in the band is a bystander: not fenced");
    let why = u.shell_fence_refusal(&pos, &[1, 2]).expect("the check's pair is fenced");
    assert!(why.text.contains("H–He at 5.50 Bohr is in the cutoff shell"), "{}", why.text);
    assert!(u.shell_fence_refusal(&pos, &[2, 2]).is_none(), "He–He is not fenced");
}

/// R3-1 end to end: a pass through the `Trainable` face on a dimer at r₀ = 3.3 Bohr (its scan
/// crosses 5/6 of r_cut) solves one shell check, never stores it, and reports its verdict after the
/// pass's refit.
#[test]
fn a_pass_reports_its_shell_check_after_the_refit() {
    let teacher = Shared::new(MorseDimer::h2());
    let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() }).with_teacher_tol(1e-6);
    let scene = in_big_cell(h2(3.3));
    let prov = s.provenance_for(&scene);
    let mut refs = sim::AtomReferences::default();
    refs.seed_energy(crate::sampling::atom_key(&prov, 1), MorseDimer::h2().e_atom, None, prov.tol, &prov.teacher);
    let plan = SamplingPlan { perturbations: 60, hot_burst_steps: 0, scan_points_per_pair: 12, seed: 3, ..SamplingPlan::default() };
    Trainable::start_pass(&mut s, &scene, &plan, &mut refs).unwrap();
    while Trainable::pass_step(&mut s, &mut never()).unwrap().is_some() {}
    let notes = Trainable::take_notes(&mut s);
    let shell: Vec<&String> = notes.iter().filter(|n| n.contains("still interacts at r =") || n.contains("cutoff holds for")).collect();
    println!("{shell:?}");
    assert_eq!(shell.len(), 1, "{notes:?}");
    assert!(s.store().unwrap().samples.iter().all(|x| x.pos.len() != 2 || {
        let r = (0..3).map(|k| (x.pos[0][k] - x.pos[1][k]).powi(2)).sum::<f64>().sqrt();
        r <= crate::sampling::SCAN_CUTOFF_FRACTION * 6.0 + 1.0 // nudges may stray a little; no scan point beyond
    }));
}

/// R3-2: a round over two blockers where one had no check judged near it is not accepted, and the
/// outcome names it.
#[test]
fn every_blocker_needs_a_judged_check() {
    let (mut s, _h) = gap_blocked(2.0, 3.6);
    s.start_teach(&in_big_cell(h2(1.6))).unwrap();
    while Trainable::pass_step(&mut s, &mut never()).unwrap().is_some() {}
    assert!(s.user_notes.iter().any(|n| n.starts_with("teaching around #60: accepted")), "premise");
    let progress = sim::SamplingProgress { done: 9, total: 9, stage: "done", last_energy: 0.0, cancelled: false, skipped: 0, failed: 0, finished: true, target: None };
    let held = s.store().unwrap().samples[30].clone();
    let scene = Scene { nuclei: held.pos.iter().map(|p| Nucleus::new(1, *p)).collect(), ..in_big_cell(h2(1.6)) };
    let near_60 = crate::sampling::Holdout { blocker: 60, scene, energy: held.energy, forces: held.forces.clone(), face: None, multiplicity: None, unfit: false };
    s.finish_teach(&[59, 60], vec![near_60], &progress);
    let outcome = s.user_notes.last().unwrap().clone();
    println!("{outcome}");
    assert!(outcome.starts_with("teaching around #59, #60: not accepted"), "{outcome}");
    assert!(outcome.contains("no check near #59 could be judged"), "{outcome}");
    assert!(outcome.contains("1 of 1 check geometries judged"), "{outcome}");
}

// ---- M8a Task 2: the calibrated, hysteretic class verdict (`calibrate`) ----

/// A history point of `n` atoms, every one of class `code` (1 pair, 2 many-body), whose teacher
/// error is `z` × the model's predicted force std of 1e-3 Ha/Bohr, at a teacher force of 0 (so
/// every atom's bound is the absolute 2e-3 Ha/Bohr: the measured ratio ‖ΔF‖/√3/bound is z × 0.2887)
/// and no energy error.
fn history_point(kind: surrogate::history::PointKind, n: usize, code: u8, z: f64) -> surrogate::AuditPoint {
    history_point_full(kind, n, code, z * 1e-3, 0.0)
}

/// [`history_point`] with the per-atom force error and the signed energy error given directly.
fn history_point_full(kind: surrogate::history::PointKind, n: usize, code: u8, err_force: f64, err_energy: f64) -> surrogate::AuditPoint {
    surrogate::AuditPoint {
        kind,
        classes: vec![code; n],
        pred_force_std: vec![1e-3; n],
        pred_force_std_epistemic: vec![5e-4; n],
        pred_energy_std: 1e-5,
        err_force: vec![err_force; n],
        teacher_force: vec![0.0; n],
        err_energy,
        n_atoms: n,
        model_gen: 1,
    }
}

/// The measured force ratio of a [`history_point`] at `z`: ‖ΔF‖/√3 over the absolute bound.
fn w_of(z: f64) -> f64 {
    z * 1e-3 / 3f64.sqrt() / 2e-3
}


/// Step 1 (the brief): a history whose measured errors are 3× the predicted stds gives a
/// calibration factor ≥ 2.5 (3 × (1 + 1/√40) = 3.47 from 20 audit points) and makes a previously
/// trusted class untrusted — here through the error the teacher measured in bound units (W = 0.866
/// × 1.158 = 1.003 > LEAVE); the same class under audits at z = 1 (W 0.33) stays trusted and would
/// enter. The other class, with no evidence, is judged by the rule of before.
#[test]
fn calibration_widens_when_audits_disagree() {
    use calibrate::{calibrate, ClassStats, ClassVerdict, LEAVE};
    use sim::ClassKey;
    use surrogate::history::PointKind;
    let pol = FallbackPolicy::default();
    let class = ClassStats { samples: 100, force_rms: 0.4, force_max: 0.9, energy_rms: 0.2, energy_max: 0.6 };
    let mut h = surrogate::AuditHistory::default();
    for _ in 0..20 {
        h.push(history_point(PointKind::Audit, 2, 1, 3.0));
    }
    let cal = calibrate(&h, &pol);
    let c = cal.per_class[0];
    println!("{c:?}");
    assert!(c.factor >= 2.5, "{c:?}");
    assert_eq!((c.audits, c.fallbacks, c.evidence, c.points), (20, 0, 20, 20));
    let infl = 1.0 + 1.0 / 40f64.sqrt();
    assert!((c.z_rms - 3.0).abs() < 1e-12 && (c.factor - 3.0 * infl).abs() < 1e-12, "{c:?}");
    assert!((c.measured - w_of(3.0) * infl).abs() < 1e-12 && c.measured > LEAVE, "{c:?}");
    match cal.class_verdict(0, &class, 29, true) {
        ClassVerdict::Untrusted { key, value, limit, reason } => {
            assert_eq!((key, limit), (ClassKey::Calibrated, LEAVE));
            assert!((value - c.measured).abs() < 1e-12, "{value}");
            assert!(reason.contains("DFT-measured force error 1.00× the bound over 20 DFT comparisons") && reason.contains("uncertainty was 3.5× too small"), "{reason}");
        }
        v => panic!("{v:?}"),
    }
    assert!(matches!(cal.class_verdict(0, &class, 29, false), ClassVerdict::Untrusted { key: ClassKey::Calibrated, .. }));
    // calibrated audits: W 0.33, the LOO floor 0.4 decides — trusted whether it was or not
    let mut ok = surrogate::AuditHistory::default();
    for _ in 0..20 {
        ok.push(history_point(PointKind::Audit, 2, 1, 1.0));
    }
    let cal = calibrate(&ok, &pol);
    assert!((cal.per_class[0].measured - w_of(1.0) * infl).abs() < 1e-12);
    assert_eq!(cal.class_verdict(0, &class, 29, true), ClassVerdict::Trusted);
    assert_eq!(cal.class_verdict(0, &class, 29, false), ClassVerdict::Trusted);
    // the LOO floor: a class whose own leave-one-out RMS is beyond ENTER does not enter on good audits
    assert!(matches!(cal.class_verdict(0, &ClassStats { force_rms: 0.8, ..class }, 29, false), ClassVerdict::Untrusted { key: ClassKey::LooFloor, value, .. } if value == 0.8));
    // the many-body class saw no evidence: the rule of before (max 0.9 ≤ 1 → trusted; max 1.5 → not)
    assert_eq!((cal.per_class[1].evidence, cal.per_class[1].factor), (0, 1.0));
    assert!(cal.per_class[1].measured.is_nan());
    assert_eq!(cal.class_verdict(1, &class, 29, false), ClassVerdict::Trusted);
    assert!(matches!(cal.class_verdict(1, &ClassStats { force_max: 1.5, ..class }, 29, true), ClassVerdict::Untrusted { key: ClassKey::Force, .. }));
    // short of samples first, whatever the evidence
    assert!(matches!(cal.class_verdict(0, &ClassStats { samples: 28, ..class }, 29, true), ClassVerdict::Untrusted { key: ClassKey::Samples, .. }));
}

/// The verdict's hysteresis and its evidence rules (module doc of `calibrate`): a measured error
/// between ENTER = 1/√2 and LEAVE = 1 keeps the standing verdict either way; the energy is judged
/// by its measured error the same way; evidence is counted per point (a trimer's three atoms are
/// one draw), a class's rule-4b fallback points weigh at most as much as the audits (never fewer
/// than three) while the inflation counts the points, fallbacks by other rules are not evidence, a
/// comparison that is not a number is infinite evidence against, and with fewer than three points
/// the rule of before stands — except that one atom measured beyond 3× its bound refuses (I4).
#[test]
fn the_verdict_has_hysteresis_and_counts_evidence_per_point_and_per_rule() {
    use calibrate::{calibrate, ClassStats, ClassVerdict, ENTER, LEAVE, MIN_EVIDENCE, WORST_ATOM};
    use sim::ClassKey;
    use surrogate::history::PointKind;
    let pol = FallbackPolicy::default();
    let code_4b = history_rule_code(TeachRule::Class(sim::activity::EnvClass::Pair));
    let code_5 = history_rule_code(TeachRule::Spread);
    // 400 audits at z: W = z × 0.2887 × (1 + 1/√800)
    let infl = 1.0 + 1.0 / 800f64.sqrt();
    let audits = |z: f64| {
        let mut h = surrogate::AuditHistory::default();
        for _ in 0..400 {
            h.push(history_point(PointKind::Audit, 2, 1, z));
        }
        calibrate(&h, &pol)
    };
    let z_for = |w: f64| w / (w_of(1.0) * infl); // z such that W_inflated = w
    let st = ClassStats { samples: 100, force_rms: 0.1, force_max: 3.0, energy_rms: 0.1, energy_max: 5.0 };
    let inside = audits(z_for(0.85));
    assert!((inside.per_class[0].measured - 0.85).abs() < 1e-9, "{:?}", inside.per_class[0]);
    assert_eq!(inside.class_verdict(0, &st, 29, true), ClassVerdict::Trusted, "stays");
    assert!(matches!(inside.class_verdict(0, &st, 29, false), ClassVerdict::Untrusted { key: ClassKey::Calibrated, limit, .. } if limit == ENTER), "does not enter");
    assert_eq!(audits(z_for(0.6)).class_verdict(0, &st, 29, false), ClassVerdict::Trusted, "enters below ENTER");
    assert!(matches!(audits(z_for(1.05)).class_verdict(0, &st, 29, true), ClassVerdict::Untrusted { key: ClassKey::Calibrated, limit, .. } if limit == LEAVE), "leaves beyond LEAVE");
    // the max no longer decides once calibrated (force_max 3.0 above); the measured energy does
    let mut he = surrogate::AuditHistory::default();
    for _ in 0..400 {
        he.push(history_point_full(PointKind::Audit, 2, 1, 0.5e-3, 0.9 * 2.0 * pol.energy_std_per_atom / infl));
    }
    let cal = calibrate(&he, &pol);
    assert!((cal.per_class[0].measured_energy - 0.9).abs() < 1e-9, "{:?}", cal.per_class[0]);
    assert!(matches!(cal.class_verdict(0, &st, 29, false), ClassVerdict::Untrusted { key: ClassKey::MeasuredEnergy, value, limit, .. } if (value - 0.9).abs() < 1e-9 && limit == ENTER));
    assert_eq!(cal.class_verdict(0, &st, 29, true), ClassVerdict::Trusted);
    // per point: one trimer with z = 3 on every atom and three dimer audits at z = 1 → ẑ² = (9 + 3)/4
    let mut h = surrogate::AuditHistory::default();
    h.push(history_point(PointKind::Audit, 3, 1, 3.0));
    for _ in 0..3 {
        h.push(history_point(PointKind::Audit, 2, 1, 1.0));
    }
    let c = calibrate(&h, &pol).per_class[0];
    assert!((c.z_rms - 3f64.sqrt()).abs() < 1e-12 && c.evidence == 4 && c.points == 4, "{c:?} (per atom it would be √(33/9))");
    // per rule: 3000 rule-4b fallbacks at z = 5 beside 10 audits at z = 1 weigh as 10 points:
    // ẑ² = (10 + 10·25)/20 = 13; the same 3000 points under rule 5 are not evidence at all
    for other in [false, true] {
        let mut h = surrogate::AuditHistory::default();
        for _ in 0..10 {
            h.push(history_point(PointKind::Audit, 2, 1, 1.0));
        }
        for _ in 0..3000 {
            h.push(history_point(PointKind::Fallback { rule: if other { code_5 } else { code_4b } }, 2, 1, 5.0));
        }
        let c = calibrate(&h, &pol).per_class[0];
        if other {
            assert_eq!((c.audits, c.fallbacks, c.evidence, c.points, c.z_rms), (10, 0, 10, 10, 1.0), "{c:?}");
        } else {
            assert_eq!((c.audits, c.fallbacks, c.evidence, c.points), (10, 3000, 20, 3010), "{c:?}");
            assert!((c.z_rms - 13f64.sqrt()).abs() < 1e-9, "{c:?}");
            assert!((c.measured - 13f64.sqrt() * w_of(1.0) * (1.0 + 1.0 / 6020f64.sqrt())).abs() < 1e-9, "the inflation counts the 3010 points: {c:?}");
        }
    }
    // no audits: a refused class's own fallbacks calibrate it, worth MIN_EVIDENCE points at most,
    // the inflation by their own number (M1)
    let mut h = surrogate::AuditHistory::default();
    for _ in 0..50 {
        h.push(history_point(PointKind::Fallback { rule: code_4b }, 2, 1, 0.8));
    }
    let c = calibrate(&h, &pol).per_class[0];
    assert_eq!((c.audits, c.fallbacks, c.evidence, c.points), (0, 50, MIN_EVIDENCE, 50));
    assert!(c.calibrated() && (c.factor - (1.0 + 1.0 / 100f64.sqrt())).abs() < 1e-12, "floored at 1, inflated for 50 points: {c:?}");
    // two of them are not enough: the rule of before (force_max 3.0 → refused)
    let mut h2 = surrogate::AuditHistory::default();
    for _ in 0..2 {
        h2.push(history_point(PointKind::Fallback { rule: code_4b }, 2, 1, 0.8));
    }
    let cal2 = calibrate(&h2, &pol);
    assert!(!cal2.per_class[0].calibrated());
    assert!(matches!(cal2.class_verdict(0, &st, 29, false), ClassVerdict::Untrusted { key: ClassKey::Force, .. }));
    // … unless a point already shows an atom beyond 3× its bound (I4): refused, entering or staying
    let mut h3 = surrogate::AuditHistory::default();
    h3.push(history_point(PointKind::Audit, 2, 1, 12.0)); // ‖ΔF‖/√3/bound = 3.46
    let cal3 = calibrate(&h3, &pol);
    let fine = ClassStats { force_max: 0.5, energy_max: 0.5, ..st };
    assert!((cal3.per_class[0].worst_atom - w_of(12.0)).abs() < 1e-12 && cal3.per_class[0].worst_atom > WORST_ATOM);
    for was in [false, true] {
        assert!(matches!(cal3.class_verdict(0, &fine, 29, was), ClassVerdict::Untrusted { key: ClassKey::WorstAtom, value, limit, .. } if limit == WORST_ATOM && (value - w_of(12.0)).abs() < 1e-12), "{was}");
    }
    let mut h4 = surrogate::AuditHistory::default();
    h4.push(history_point(PointKind::Audit, 2, 1, 9.0)); // 2.6: inside the guard
    assert_eq!(calibrate(&h4, &pol).class_verdict(0, &fine, 29, false), ClassVerdict::Trusted);
    // a comparison that is not a number: infinite factor and measured error, untrusted
    let mut h = surrogate::AuditHistory::default();
    for _ in 0..3 {
        h.push(history_point(PointKind::Audit, 2, 1, 1.0));
    }
    let mut bad = history_point(PointKind::Audit, 2, 1, 1.0);
    bad.err_force[0] = f64::NAN;
    h.push(bad);
    let cal = calibrate(&h, &pol);
    assert_eq!((cal.per_class[0].factor, cal.per_class[0].measured), (f64::INFINITY, f64::INFINITY));
    assert!(matches!(cal.class_verdict(0, &ClassStats { force_rms: 1e-6, ..st }, 29, true), ClassVerdict::Untrusted { key: ClassKey::Calibrated, .. }));
    // isolated atoms (code 0) and the other class's atoms are not this class's evidence
    let mut h = surrogate::AuditHistory::default();
    let mut p = history_point(PointKind::Audit, 3, 2, 4.0);
    p.classes = vec![0, 1, 2];
    h.push(p);
    let cal = calibrate(&h, &pol);
    assert_eq!((cal.per_class[0].evidence, cal.per_class[1].evidence), (1, 1));
    // the per-atom bound follows the teacher's force: at |F| = 0.05 the bound is 5e-3, the same error weighs 2.5× less
    let mut hb = surrogate::AuditHistory::default();
    let mut q = history_point(PointKind::Audit, 2, 1, 3.0);
    q.teacher_force = vec![0.05; 2];
    hb.push(q);
    assert!((calibrate(&hb, &pol).per_class[0].worst_atom - w_of(3.0) * 2e-3 / 5e-3).abs() < 1e-12);
}

/// Spec §4.1, the knife-edge the verdict owns: a pair class the max rule refuses everywhere for one
/// outlier (the gap fixture's sample at 3.6 Bohr beyond a window learned to 2.2: leave-one-out
/// 5×) earns the calibrated verdict from its own evidence — the rule-4b fallback steps the refusal
/// produces compare the model with the teacher inside the window, where the model is right — and
/// then holds: over 200 steps of a dimer vibrating in its window, refitting every 10 steps while
/// samples are pending, the class verdict changes at most once (untrusted → trusted), the class is
/// trusted at the end with the outlier still in the fit, and the model answers. Before this task the
/// class stayed refused for the whole run (0 model answers) — the brief's `class_verdict_does_not_flip`
/// scenario moved from the cutoff to the outlier: a pair crossing r_cut is the shell check's
/// structural case (`sampling`, R3-1), not the class verdict's.
#[test]
fn a_class_refused_for_one_outlier_earns_the_calibrated_verdict_and_holds() {
    use surrogate::history::PointKind;
    let teacher = Shared::new(MorseDimer::h2());
    let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() }).with_teacher_tol(1e-6);
    s.set_reference_energy(1, MorseDimer::h2().e_atom);
    let mut rng = StdRng::seed_from_u64(4);
    for i in 0..201 {
        let r = if i < 200 { 1.2 + 1.0 * i as f64 / 199.0 } else { 3.6 };
        let sc = in_big_cell(dimer(r, unit_vector(&mut rng), [0.0; 3], 1, 1));
        s.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
    }
    s.refit().unwrap();
    let pair = s.class_loo()[0].clone();
    println!("after the fit with the outlier: {pair:?}");
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    assert!(!pair.trusted && pair.force_max > 1.0 && pair.force_rms < calibrate::ENTER, "premise: refused by the max alone: {pair:?}");
    let mut verdicts = Vec::new();
    let mut answered = 0;
    for k in 0..200 {
        let r = 1.7 + 0.3 * (k as f64 / 10.0).sin();
        let sc = in_big_cell(dimer(r, unit_vector(&mut rng), [0.0; 3], 1, 1));
        let out = s.evaluate(&sc, &mut never()).unwrap();
        answered += (out.stats.source == Source::Surrogate) as usize;
        if k % 10 == 9 && s.pending_refit() {
            s.refit().unwrap();
        }
        verdicts.push(s.class_loo()[0].trusted);
    }
    let flips = verdicts.windows(2).filter(|w| w[0] != w[1]).count();
    let first = verdicts.iter().position(|&t| t);
    let l = &s.class_loo()[0];
    let evidence = s.audit_history().iter().filter(|p| matches!(p.kind, PointKind::Fallback { .. })).count();
    println!("verdict flips {flips}, first trusted after step {first:?}, {answered}/200 answered, {evidence} fallback points; {l:?}");
    assert!(flips <= 1, "the verdict flipped {flips} times: {verdicts:?}");
    assert!(l.trusted, "the class did not earn trust: {l:?}");
    assert!(l.calibration >= 1.0 && l.evidence >= calibrate::MIN_EVIDENCE && l.trusted_since.is_some(), "{l:?}");
    assert!(l.force_max > 1.0, "the outlier is still in the fit: {l:?}");
    assert!(answered >= 100, "{answered} answered");
}

/// The hysteresis memory through the provider (M8a Task 2, `class_memory`): the verdict a refit
/// reads is the one that stood when the refit began, committed once the refit is done — a class
/// whose calibrated error lies in the band (ENTER, LEAVE] stays trusted if it was and stays
/// untrusted if it was not, through `refit()` itself; a forget clears the memory.
#[test]
fn the_hysteresis_memory_is_the_verdict_that_stood_when_the_refit_began() {
    use calibrate::{ENTER, LEAVE};
    use surrogate::history::PointKind;
    let (mut s, _h) = trained(1.2, 2.2, 120, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    let r = s.class_loo()[0].force_rms;
    assert!(s.class_loo()[0].trusted && r > 0.0 && r < ENTER, "premise: a trusted pair class with LOO force rms {r}");
    // three audit points whose measured error puts the verdict's force value at `v`
    // (W = z × 0.2887 × (1 + 1/√6); the LOO floor r is far below)
    let evidence = |s: &mut SurrogateKs, v: f64| {
        let z = v / (w_of(1.0) * (1.0 + 1.0 / 6f64.sqrt()));
        assert!(w_of(z) <= 1.0, "premise: no atom beyond the 3× guard ({})", w_of(z));
        s.audit_history = surrogate::AuditHistory::default();
        for _ in 0..3 {
            s.audit_history.push(history_point(PointKind::Audit, 2, 1, z));
        }
    };
    // in the band: the memory decides
    evidence(&mut s, 0.85);
    s.class_memory = vec![(true, Some(7)), (false, None)];
    s.refresh_class_loo();
    let l = &s.class_loo()[0];
    assert!(l.trusted && l.trusted_since == Some(7) && (l.measured.unwrap() - 0.85).abs() < 1e-9, "stays: {l:?}");
    s.class_memory = vec![(false, None), (false, None)];
    s.refresh_class_loo();
    let l = s.class_loo()[0].clone();
    assert!(!l.trusted && l.trusted_since.is_none(), "does not enter: {l:?}");
    assert!(matches!(l.refusal, Some((sim::ClassKey::Calibrated, v, lim)) if (v - 0.85).abs() < 1e-9 && lim == ENTER), "{l:?}");
    let (r2, _) = s.class_untrusted(&[1, 1]).expect("refused");
    assert!(r2.text.contains("DFT-measured force error 0.85× the bound over 3 DFT comparisons, limit 0.71"), "{}", r2.text);
    // through refit(): the memory is read at the start and committed at the end
    s.refit().unwrap();
    assert!(!s.class_loo()[0].trusted, "untrusted memory, band value: still untrusted after the refit");
    assert_eq!(s.class_memory[0], (false, None), "committed");
    evidence(&mut s, 0.6);
    s.refit().unwrap();
    let gen = s.model_generation();
    let l = s.class_loo()[0].clone();
    assert!(l.trusted && l.trusted_since == Some(gen), "enters below ENTER at the refit's generation: {l:?}");
    assert_eq!(s.class_memory[0], (true, Some(gen)));
    evidence(&mut s, 0.85);
    s.refit().unwrap();
    assert!(s.class_loo()[0].trusted && s.class_loo()[0].trusted_since == Some(gen), "stays trusted in the band, since the same generation: {:?}", s.class_loo()[0]);
    evidence(&mut s, 1.05);
    s.refit().unwrap();
    let l = s.class_loo()[0].clone();
    assert!(!l.trusted && matches!(l.refusal, Some((sim::ClassKey::Calibrated, _, lim)) if lim == LEAVE), "leaves beyond LEAVE: {l:?}");
    // a forget clears the memory
    s.forget();
    assert!(s.class_memory.is_empty() && s.class_loo().is_empty());
}

/// The verdict's jurisdiction is exact (M8a Task 2): a class-refused step that rule 6 would have
/// refused as well is coded `HISTORY_RULE_CLASS_AND_VARIANCE` (the class's note is still the one
/// shown) and is **not** the class verdict's evidence; a class-refused step the variance accepted is
/// a rule-4b point and is. The relax verification keeps the rule that fired before its override
/// (the Task 1 re-review's Minor 1).
#[test]
fn a_class_refusal_the_variance_would_also_refuse_is_not_the_verdicts_evidence() {
    use surrogate::history::PointKind;
    let (mut s, _h) = trained(1.2, 2.2, 120, FallbackPolicy { audit_every: 0, variance_cadence: 1, ..FallbackPolicy::default() });
    let untrust_pair = |s: &mut SurrogateKs| {
        s.class_loo[0].trusted = false;
        s.class_loo[0].refusal = Some((sim::ClassKey::Force, 2.0, 1.0));
    };
    // the variance accepts: a rule-4b point, evidence
    untrust_pair(&mut s);
    let r = s.evaluate(&h2(1.6), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback);
    assert!(r.stats.notes[0].contains("isn't learned well enough yet"), "{:?}", r.stats.notes);
    let p = s.audit_history().iter().last().unwrap();
    assert_eq!(p.kind, PointKind::Fallback { rule: history_rule_code(TeachRule::Class(sim::activity::EnvClass::Pair)) });
    assert_eq!(calibrate::calibrate(s.audit_history(), s.policy()).per_class[0].fallbacks, 1);
    // the variance would refuse too (the posterior inflated): coded apart, not evidence
    always_unsure(&mut s);
    let r = s.evaluate(&h2(1.62), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback);
    assert!(r.stats.notes[0].contains("isn't learned well enough yet"), "the class's note is shown: {:?}", r.stats.notes);
    let p = s.audit_history().iter().last().unwrap();
    assert_eq!(p.kind, PointKind::Fallback { rule: HISTORY_RULE_CLASS_AND_VARIANCE });
    assert_eq!(history_rule_name(HISTORY_RULE_CLASS_AND_VARIANCE), "rule 4b with rule 6/7 failing too");
    let c = calibrate::calibrate(s.audit_history(), s.policy()).per_class[0];
    assert_eq!((c.fallbacks, c.evidence), (1, 1), "the second point is not evidence: {c:?}");
    // the relax verification over a class refusal keeps the class's code
    s.force_teacher_next();
    let r = s.evaluate(&h2(1.58), &mut never()).unwrap();
    assert!(r.stats.notes[0].contains("relax verification"), "{:?}", r.stats.notes);
    let p = s.audit_history().iter().last().unwrap();
    assert_eq!(p.kind, PointKind::Fallback { rule: HISTORY_RULE_CLASS_AND_VARIANCE }, "the rule that fired, not the override");
    // and over an answered geometry it is the verification's own code
    let (mut t, _h2) = trained(1.2, 2.2, 120, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    t.force_teacher_next();
    let r = t.evaluate(&h2(1.6), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback);
    assert_eq!(t.audit_history().iter().last().unwrap().kind, PointKind::Fallback { rule: history_rule_code(TeachRule::RelaxVerification) });
}

// ---- M8a Task 2 fix round 2: the exact evidence count, the evidence window, the entry guard,
// the persisted hysteresis memory, the #344 repro ----

/// The task review's I1: the evidence count n_a + min(n_f, cap) is formed in integers. As
/// n_a + (min(n_f, cap)/n_f)·n_f it was 2.9999999999999996 at n_f = 47 (and 94, 147, 173, 188, …),
/// below `MIN_EVIDENCE`: the class silently fell back to the max rule at those counts.
#[test]
fn the_evidence_count_is_exact_at_47_fallbacks() {
    use calibrate::{calibrate, MIN_EVIDENCE};
    use surrogate::history::PointKind;
    let pol = FallbackPolicy::default();
    let code_4b = history_rule_code(TeachRule::Class(sim::activity::EnvClass::Pair));
    let float_form = |n_f: usize| (n_f as f64).min(3.0) / n_f as f64 * n_f as f64;
    assert!(float_form(47) < 3.0, "the premise: the float form rounds below three at 47 ({:e})", float_form(47));
    for n_f in [3usize, 47, 94, 147, 173, 188, 4096] {
        let mut h = surrogate::AuditHistory::default();
        for _ in 0..n_f {
            h.push(history_point(PointKind::Fallback { rule: code_4b }, 2, 1, 0.8));
        }
        let c = calibrate(&h, &pol).per_class[0];
        assert_eq!((c.evidence, c.points), (MIN_EVIDENCE, n_f), "{n_f}: {c:?}");
        assert!(c.calibrated(), "{n_f}: {c:?}");
    }
}

/// Fix round 2, the controller's rulings on the lockout (the task review's I4) and on the entry
/// question: evidence ages by model generation, both ways. A class refused on 8 audits at 1.46×
/// its bound (the cadence-10 loss) stays refused while the model that produced them stands — 40
/// good fallbacks of the same generation are capped at the audits' weight — and while the newer
/// model has fewer than `WINDOW_POINTS` (300) comparisons; with 300 the window holds the newer
/// model alone and the class enters. The worst-atom guard: one atom at 3.46× its bound among 100 good
/// comparisons keeps an untrusted class out although the rms would enter, leaves a trusted class
/// trusted (its rms judges it), and ages with its generation.
#[test]
fn evidence_ages_by_model_generation_both_ways() {
    use calibrate::{calibrate, ClassStats, ClassVerdict, ENTER, LEAVE, WINDOW_POINTS, WORST_ATOM};
    use sim::ClassKey;
    use surrogate::history::PointKind;
    let pol = FallbackPolicy::default();
    let code_4b = history_rule_code(TeachRule::Class(sim::activity::EnvClass::Pair));
    let at = |p: surrogate::AuditPoint, gen: u64| surrogate::AuditPoint { model_gen: gen, ..p };
    let st = ClassStats { samples: 100, force_rms: 0.1, force_max: 3.0, energy_rms: 0.1, energy_max: 0.5 };
    let z_bad = 1.46 / w_of(1.0);
    let mut h = surrogate::AuditHistory::default();
    for _ in 0..8 {
        h.push(at(history_point(PointKind::Audit, 2, 1, z_bad), 1));
    }
    assert!(matches!(calibrate(&h, &pol).class_verdict(0, &st, 29, true), ClassVerdict::Untrusted { key: ClassKey::Calibrated, limit, .. } if limit == LEAVE), "leaves on its audits");
    // blocked while the refusing model stands: its own later fallbacks are capped at the audits
    for _ in 0..40 {
        h.push(at(history_point(PointKind::Fallback { rule: code_4b }, 2, 1, 0.2), 1));
    }
    let c = calibrate(&h, &pol).per_class[0];
    println!("generation 1 only: {c:?}");
    assert_eq!((c.audits, c.fallbacks, c.evidence, c.aged, c.window_from), (8, 40, 16, 0, None), "fewer than WINDOW_POINTS held: the whole ring");
    assert!(matches!(calibrate(&h, &pol).class_verdict(0, &st, 29, false), ClassVerdict::Untrusted { key: ClassKey::Calibrated, value, .. } if value > LEAVE), "blocked: {c:?}");
    // the refit (generation 2): its comparisons supersede the old ones only once they fill a window
    for k in 1..=WINDOW_POINTS {
        h.push(at(history_point(PointKind::Fallback { rule: code_4b }, 2, 1, 0.2), 2));
        let cal = calibrate(&h, &pol);
        let c = cal.per_class[0];
        let v = cal.class_verdict(0, &st, 29, false);
        if k < WINDOW_POINTS {
            assert!(c.aged == 0 && c.window_from != Some(2), "{k}: {c:?}");
            assert!(matches!(v, ClassVerdict::Untrusted { key: ClassKey::Calibrated, .. }), "{k}: still blocked: {v:?}");
        } else {
            println!("generation 2 fills the window: {c:?}");
            assert_eq!((c.window_from, c.aged, c.audits, c.fallbacks, c.evidence), (Some(2), 48, 0, WINDOW_POINTS, 3), "{c:?}");
            assert!(c.measured < ENTER, "{c:?}");
            assert_eq!(v, ClassVerdict::Trusted, "released by the newer model's own evidence");
        }
    }
    // the worst-atom guard on entry, at every evidence count, aged by generation
    let fine = ClassStats { force_max: 0.5, ..st };
    let mut g = surrogate::AuditHistory::default();
    g.push(at(history_point(PointKind::Audit, 2, 1, 12.0), 1)); // 3.46× its bound
    for _ in 0..100 {
        g.push(at(history_point(PointKind::Audit, 2, 1, 0.5), 1));
    }
    let cal = calibrate(&g, &pol);
    let c = cal.per_class[0];
    assert!(c.calibrated() && c.measured < ENTER && c.worst_atom > WORST_ATOM, "the premise: the rms alone would enter: {c:?}");
    assert!(
        matches!(cal.class_verdict(0, &fine, 29, false), ClassVerdict::Untrusted { key: ClassKey::WorstAtom, value, limit, .. } if limit == WORST_ATOM && (value - w_of(12.0)).abs() < 1e-12),
        "does not enter"
    );
    assert_eq!(cal.class_verdict(0, &fine, 29, true), ClassVerdict::Trusted, "a trusted class with evidence is judged by its rms");
    for _ in 0..WINDOW_POINTS {
        g.push(at(history_point(PointKind::Audit, 2, 1, 0.5), 2));
    }
    let cal = calibrate(&g, &pol);
    assert!(cal.per_class[0].worst_atom < 1.0 && cal.per_class[0].aged == 101, "{:?}", cal.per_class[0]);
    assert_eq!(cal.class_verdict(0, &fine, 29, false), ClassVerdict::Trusted, "the guard aged with its generation");
}

/// The lockout through the provider (fix round 2): a pair class its audits refused is refused by
/// the refit that follows them even with 40 good fallbacks of the same model beside them, stays
/// refused while the refitted model has fewer than `WINDOW_POINTS` (300) comparisons of its own,
/// and enters at the next refit once its own rule-4b fallbacks — the teacher steps the refusal
/// makes — show it within the bound. Under fix round 1's whole-ring evidence it stayed refused
/// until 4096 later points had evicted the audits.
#[test]
fn a_class_refused_on_its_evidence_reenters_after_a_teaching_refit() {
    use calibrate::{ENTER, LEAVE, WINDOW_POINTS};
    use surrogate::history::PointKind;
    let (mut s, _h) = trained(1.2, 2.2, 120, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    assert!(s.class_loo()[0].trusted, "premise: {:?}", s.class_loo());
    let code_4b = history_rule_code(TeachRule::Class(sim::activity::EnvClass::Pair));
    let g0 = s.model_generation();
    // the standing model's audits measured 1.46× the bound; 40 good fallbacks of the same model
    for _ in 0..8 {
        s.audit_history.push(surrogate::AuditPoint { model_gen: g0, ..history_point(PointKind::Audit, 2, 1, 1.46 / w_of(1.0)) });
    }
    for _ in 0..40 {
        s.audit_history.push(surrogate::AuditPoint { model_gen: g0, ..history_point(PointKind::Fallback { rule: code_4b }, 2, 1, 0.2) });
    }
    s.refit().unwrap();
    let l = s.class_loo()[0].clone();
    println!("the refit after the failing audits: {l:?}");
    assert!(!l.trusted && matches!(l.refusal, Some((sim::ClassKey::Calibrated, v, lim)) if v > LEAVE && lim == LEAVE), "leaves, blocked by its own generation: {l:?}");
    // the refitted model answers nothing of the class: its steps are rule-4b points
    let mut rng = StdRng::seed_from_u64(7);
    let mut step = |s: &mut SurrogateKs, k: usize| {
        let r = 1.65 + 0.25 * (k as f64 / 5.0).sin();
        let out = s.evaluate(&dimer(r, unit_vector(&mut rng), [0.0; 3], 1, 1), &mut never()).unwrap();
        assert_eq!(out.stats.source, Source::Fallback, "{:?}", out.stats.notes);
        assert_eq!(s.audit_history().iter().last().unwrap().kind, PointKind::Fallback { rule: code_4b });
    };
    for k in 0..WINDOW_POINTS - 1 {
        step(&mut s, k);
    }
    // a verdict while the new model has fewer than WINDOW_POINTS comparisons: the old generation still counts
    s.refresh_class_loo();
    assert!(!s.class_loo()[0].trusted, "blocked until the refitted model's evidence fills a window: {:?}", s.class_loo()[0]);
    for k in WINDOW_POINTS - 1..WINDOW_POINTS + 20 {
        step(&mut s, k);
    }
    s.refit().unwrap();
    let l = s.class_loo()[0].clone();
    println!("the next refit, on the refitted model's {} comparisons: {l:?}", WINDOW_POINTS + 20);
    assert!(l.trusted && l.measured.is_some_and(|m| m < ENTER) && l.evidence >= calibrate::MIN_EVIDENCE, "re-entered: {l:?}");
    assert!(l.worst_atom.is_some_and(|w| w < 1.0), "{l:?}");
}

/// The task review's M4, the controller's ruling: the hysteresis memory is saved with the model
/// (the sidecar's `class_memory`), so a reload — and the worker's re-spawn with its carried model
/// and a resumed training, which load the same file — keep a class verdict that sat inside the band
/// (ENTER, LEAVE]; a file without the memory re-enters (must reach ENTER again).
#[test]
fn a_band_verdict_survives_a_save_and_reload() {
    use calibrate::{ENTER, LEAVE};
    use surrogate::history::PointKind;
    let (mut s, _h) = trained(1.2, 2.2, 120, FallbackPolicy::default());
    let z = 0.85 / (w_of(1.0) * (1.0 + 1.0 / 6f64.sqrt()));
    s.audit_history = surrogate::AuditHistory::default();
    for _ in 0..3 {
        s.audit_history.push(history_point(PointKind::Audit, 2, 1, z));
    }
    let since = s.model_generation() - 1;
    s.class_memory = vec![(true, Some(since)), (false, None)];
    s.refresh_class_loo();
    s.commit_class_memory();
    let l = s.class_loo()[0].clone();
    assert!(l.trusted && l.trusted_since == Some(since) && (l.measured.unwrap() - 0.85).abs() < 1e-9 && 0.85 > ENTER && 0.85 <= LEAVE, "premise, in the band: {l:?}");
    let path = std::env::temp_dir().join(format!("band-verdict-{}.model", std::process::id()));
    s.save_sidecar(&path).unwrap();
    let (back, _, _) = installed_from(&path);
    let b = back.class_loo()[0].clone();
    println!("reloaded: {b:?}");
    assert!(b.trusted && b.trusted_since == Some(since), "kept through the reload: {b:?}");
    assert_eq!(back.class_memory, s.class_memory);
    assert!(back.model_generation() > since && back.model_generation() > back.audit_history().iter().map(|p| p.model_gen).max().unwrap(), "generations stay monotone across the install");
    // a file without the memory (as written before fix round 2): the band verdict re-enters
    s.class_memory = Vec::new();
    s.save_sidecar(&path).unwrap();
    let (fresh, _, _) = installed_from(&path);
    let _ = std::fs::remove_file(&path);
    let f = fresh.class_loo()[0].clone();
    assert!(!f.trusted && matches!(f.refusal, Some((sim::ClassKey::Calibrated, _, lim)) if lim == ENTER), "without a memory: must reach ENTER: {f:?}");
}

/// An H₃ triangle with sides `a` (B–C), `b` (A–C), `c` (A–B), centred, in a random orientation.
fn triangle(a: f64, b: f64, c: f64, rng: &mut StdRng) -> Scene {
    let x = (b * b + c * c - a * a) / (2.0 * c);
    let y = (b * b - x * x).max(0.0).sqrt();
    let flat = [[0.0, 0.0, 0.0], [c, 0.0, 0.0], [x, y, 0.0]];
    let centre: [f64; 3] = std::array::from_fn(|k| flat.iter().map(|p| p[k]).sum::<f64>() / 3.0);
    // a random orthonormal frame (Gram–Schmidt of two random directions)
    let u = unit_vector(rng);
    let w = unit_vector(rng);
    let d = u[0] * w[0] + u[1] * w[1] + u[2] * w[2];
    let v0: [f64; 3] = std::array::from_fn(|k| w[k] - d * u[k]);
    let nv = (v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2]).sqrt();
    let v = v0.map(|x| x / nv);
    let nuclei = flat.iter().map(|p| Nucleus::new(1, std::array::from_fn(|k| (p[0] - centre[0]) * u[k] + (p[1] - centre[1]) * v[k]))).collect();
    in_big_cell(Scene { nuclei, ..h2(1.4) })
}

/// The #344 fixture: training triangles, their shortest taught side and the compressed sample's
/// shortest H–H (Bohr). Measured on this fixture (seed 344): with the compressed sample at 1.2 /
/// 1.0 / 0.9 Bohr the fit's leave-one-out predicts it within the bound (max 0.08 / 0.42 / 0.84×),
/// at 0.8 Bohr no longer (max 1.57×, RMS 0.10×, energy max 1.31×): the knife-edge premise.
const TRIMER_SAMPLES: usize = 150;
const TRIMER_MIN: f64 = 1.6;
const TRIMER_COMPRESSED: f64 = 0.8;

/// Spec §4.1's second repro case, the user's #344 (compressed H–H in H₃⁺) on an analytic teacher
/// (the pair-additive Morse, so an H₃ triangle is many-body by the class rule: every atom has two
/// neighbours within r_cut): triangles taught with sides 1.6–2.1 Bohr plus one sample whose
/// shortest H–H is compressed below the taught minimum. Under the max rule that one sample's
/// leave-one-out refuses the many-body class everywhere (the premise); over 200 steps of a
/// triangle breathing inside the taught window, refitting every 10 steps while samples are pending,
/// the verdict changes at most once (untrusted → trusted), and it is judged by its evidence — the
/// rule-4b comparisons of those steps — with the compressed sample still in the fit.
#[test]
fn a_compressed_trimer_does_not_flip_the_many_body_verdict() {
    use surrogate::history::PointKind;
    let mut s = SurrogateKs::new(Box::new(PairMorse(MorseDimer::h2())), FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() }).with_teacher_tol(1e-6);
    s.set_reference_energy(1, MorseDimer::h2().e_atom);
    let mut rng = StdRng::seed_from_u64(344);
    for _ in 0..TRIMER_SAMPLES {
        let sc = triangle(rng.gen_range(TRIMER_MIN..2.1), rng.gen_range(TRIMER_MIN..2.1), rng.gen_range(TRIMER_MIN..2.1), &mut rng);
        s.add_sample(pair_morse_sample(&sc), &sc).unwrap();
    }
    let sc = triangle(TRIMER_COMPRESSED, 1.8, 1.8, &mut rng);
    s.add_sample(pair_morse_sample(&sc), &sc).unwrap();
    s.refit().unwrap();
    let mb = s.class_loo()[1].clone();
    println!("after the fit with the compressed trimer: {mb:?}");
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    assert!(!mb.trusted && mb.force_max > 1.0 && mb.force_rms < calibrate::ENTER, "premise: refused by the max alone: {mb:?}");
    let mut verdicts = Vec::new();
    let mut answered = 0;
    for k in 0..200 {
        let t = k as f64 / 10.0;
        let sc = triangle(1.8 + 0.15 * t.sin(), 1.8 + 0.15 * (1.3 * t).cos(), 1.8 + 0.12 * (0.7 * t).sin(), &mut rng);
        let out = s.evaluate(&sc, &mut never()).unwrap();
        answered += (out.stats.source == Source::Surrogate) as usize;
        if k % 10 == 9 && s.pending_refit() {
            s.refit().unwrap();
        }
        verdicts.push(s.class_loo()[1].trusted);
    }
    let flips = verdicts.windows(2).filter(|w| w[0] != w[1]).count();
    let l = s.class_loo()[1].clone();
    let rule_4b = s.audit_history().iter().filter(|p| p.kind == PointKind::Fallback { rule: history_rule_code(TeachRule::Class(sim::activity::EnvClass::ManyBody)) }).count();
    println!("verdict flips {flips}, first trusted after step {:?}, {answered}/200 answered, {rule_4b} rule-4b many-body points; {l:?}", verdicts.iter().position(|&t| t));
    assert!(flips <= 1, "the verdict flipped {flips} times: {verdicts:?}");
    assert!(l.trusted && l.evidence >= calibrate::MIN_EVIDENCE && l.measured.is_some(), "judged by its evidence: {l:?}");
    assert!(l.force_max > 1.0, "the compressed sample is still in the fit: {l:?}");
    assert!(answered >= 100, "{answered} answered");
}


// ---- M8a Task 2 fix round 3: the variance inflation of rules 6/7 ----

/// A history point whose per-atom epistemic force std is `epi` and error `err` (Ha/Bohr), at a
/// teacher force of 0 (every bound the absolute 2e-3).
fn posterior_point(kind: surrogate::history::PointKind, err: f64, epi: f64) -> surrogate::AuditPoint {
    surrogate::AuditPoint { pred_force_std_epistemic: vec![epi; 2], pred_force_std: vec![epi.max(1e-12); 2], ..history_point_full(kind, 2, 1, err, 0.0) }
}

/// The variance inflation λ (`calibrate`, "The variance inflation"): the measured error over the
/// epistemic std, both in bound units, as a **ratio of sums** over the posterior's jurisdiction —
/// 20 points at error 3× their std give λ = 3 × (1 + 1/√40); a calibrated class is floored at 1
/// (nothing loosened) and carries only the scatter factor; fewer than three points give exactly
/// 1; one confident point (std 1e-6) with a modest error (z = 62, the physics review's I2 case)
/// among 266 calibrated points barely moves it (a mean of z² would give 4.0); fallbacks refused
/// by rules 6/7 (codes 14, 15) and by the class with them (21) are its evidence, rule 5's are not;
/// a comparison that is not a number makes it infinite.
#[test]
fn the_variance_inflation_is_a_ratio_of_sums_over_the_posteriors_jurisdiction() {
    use calibrate::{calibrate, MIN_EVIDENCE};
    use surrogate::history::PointKind;
    let pol = FallbackPolicy::default();
    let infl = |n: usize| 1.0 + 1.0 / (2.0 * n as f64).sqrt();
    let mut h = surrogate::AuditHistory::default();
    for _ in 0..20 {
        h.push(posterior_point(PointKind::Audit, 1.5e-3, 5e-4));
    }
    let c = calibrate(&h, &pol).per_class[0];
    assert!((c.inflation_raw - 3.0).abs() < 1e-12 && (c.inflation - 3.0 * infl(20)).abs() < 1e-12 && c.inflation_points == 20, "{c:?}");
    // calibrated: floored at 1, the scatter factor only
    let mut h = surrogate::AuditHistory::default();
    for _ in 0..20 {
        h.push(posterior_point(PointKind::Audit, 2.5e-4, 5e-4));
    }
    let c = calibrate(&h, &pol).per_class[0];
    assert!((c.inflation_raw - 0.5).abs() < 1e-12 && (c.inflation - infl(20)).abs() < 1e-12, "{c:?}");
    // thin: exactly 1
    let mut h = surrogate::AuditHistory::default();
    for _ in 0..MIN_EVIDENCE - 1 {
        h.push(posterior_point(PointKind::Audit, 5e-2, 5e-4));
    }
    assert_eq!(calibrate(&h, &pol).per_class[0].inflation, 1.0);
    // I2: one confident point with a modest error does not set the factor
    let mut h = surrogate::AuditHistory::default();
    for _ in 0..266 {
        h.push(posterior_point(PointKind::Audit, 5e-4, 5e-4));
    }
    h.push(posterior_point(PointKind::Audit, 6.2e-5, 1e-6));
    let c = calibrate(&h, &pol).per_class[0];
    let mean_of_ratios = ((266.0 + 62.0f64.powi(2)) / 267.0).sqrt();
    println!("I2 case: λ raw {:.4} (a mean of z² would give {mean_of_ratios:.2}); {c:?}", c.inflation_raw);
    assert!(c.inflation_raw < 1.001 && mean_of_ratios > 3.9, "{c:?}");
    // the posterior's jurisdiction: rules 6/7 alone and with the class count, rule 5 does not
    for (rule, counted) in [
        (history_rule_code(TeachRule::ForceUncertainty), true),
        (history_rule_code(TeachRule::EnergyUncertainty), true),
        (HISTORY_RULE_CLASS_AND_VARIANCE, true),
        (history_rule_code(TeachRule::Spread), false),
        (history_rule_code(TeachRule::Grade), false),
    ] {
        let mut h = surrogate::AuditHistory::default();
        for _ in 0..5 {
            h.push(posterior_point(PointKind::Fallback { rule }, 1.5e-3, 5e-4));
        }
        let c = calibrate(&h, &pol).per_class[0];
        assert_eq!(c.inflation_points, if counted { 5 } else { 0 }, "rule {rule}: {c:?}");
        // the class verdict's own evidence is unchanged by them (not its jurisdiction)
        assert_eq!(c.points, 0, "rule {rule}");
    }
    // not a number: infinite
    let mut h = surrogate::AuditHistory::default();
    for _ in 0..3 {
        h.push(posterior_point(PointKind::Audit, 1e-3, 5e-4));
    }
    h.push(posterior_point(PointKind::Audit, f64::NAN, 5e-4));
    assert_eq!(calibrate(&h, &pol).per_class[0].inflation, f64::INFINITY);
}

/// Rule 6 applies the class's inflation (fix round 3): a trained pair model answers an in-window
/// H₂; the same model, once three rule-6 comparisons of the pair class have measured its error at
/// R × its epistemic std (so λ = R(1 + 1/√6) puts the inflated std at 4× the bound there),
/// refuses the same geometry by rule 6 — the note naming the calibration — and the step is a
/// rule-6 point of the history (the factor's own evidence). The class verdict is untouched
/// (rule-6 points are not its jurisdiction): the class stays trusted.
#[test]
fn rule_6_refuses_where_the_class_calibration_inflates_the_std() {
    use surrogate::history::PointKind;
    let (mut s, _h) = trained(1.2, 2.2, 120, FallbackPolicy { audit_every: 0, variance_cadence: 1, ..FallbackPolicy::default() });
    let sc = h2(1.6);
    let r = s.evaluate(&sc, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Surrogate, "{:?}", r.stats.notes);
    let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
    let pred = s.model().unwrap().predict(&pos, &[1, 1], true).unwrap();
    let epi = pred.force_std_epistemic.as_ref().unwrap().iter().copied().fold(0.0, f64::max);
    let fmag = pred.forces.iter().map(|f| (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt()).fold(f64::INFINITY, f64::min);
    let bound = s.policy().force_bound(fmag);
    // λ·epi/√3 = 4 bound with λ = R (1 + 1/√6) and R = err / 5e-4 (the points' epistemic std)
    let lambda = 4.0 * bound * 3f64.sqrt() / epi;
    let err = lambda / (1.0 + 1.0 / 6f64.sqrt()) * 5e-4;
    let code_6 = history_rule_code(TeachRule::ForceUncertainty);
    for _ in 0..3 {
        s.audit_history.push(posterior_point(PointKind::Fallback { rule: code_6 }, err, 5e-4));
    }
    s.refresh_class_loo();
    let l = s.class_loo()[0].clone();
    println!("epistemic std {epi:.3e} at a bound {bound:.1e}: λ {:.4e}; {l:?}", l.inflation);
    assert!((l.inflation - lambda).abs() < 1e-9 * lambda && l.trusted, "{l:?}");
    let r = s.evaluate(&h2(1.61), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback, "{:?}", r.stats.notes);
    assert!(r.stats.notes[0].contains("epistemic force std") && r.stats.notes[0].contains("its measured calibration for this environment class"), "{:?}", r.stats.notes);
    assert_eq!(s.audit_history().iter().last().unwrap().kind, PointKind::Fallback { rule: code_6 });
    // without the evidence (a forget of the history), the factor is 1 and the model answers again
    s.audit_history = surrogate::AuditHistory::default();
    s.refresh_class_loo();
    assert_eq!(s.class_loo()[0].inflation, 1.0);
    assert_eq!(s.evaluate(&h2(1.62), &mut never()).unwrap().stats.source, Source::Surrogate);
}

// ---- the shared collection (M8a Task 4a; `shared` module doc) ----

/// A pairwise Morse teacher over any species, smoothly cut off at the model's r_cut (so the
/// surface is a function the descriptors can represent), reporting a face density (`face`;
/// `None` = a teacher that reports none) and the multiplicity of the lowest-spin state of its
/// electrons (Σ Z − charge: paired, one unpaired electron for an odd count — `qm::System`'s
/// automatic choice), as a Kohn-Sham teacher does.
#[derive(Clone, Debug)]
struct SpeciesMorse {
    face: Option<f64>,
    calls: usize,
}

impl SpeciesMorse {
    fn new() -> SpeciesMorse {
        SpeciesMorse { face: Some(0.0), calls: 0 }
    }

    /// The isolated-atom energy of `z` (Hartree).
    fn atom(z: u8) -> f64 {
        match z {
            1 => -0.478,
            2 => -2.83,
            _ => -1.0 * z as f64,
        }
    }

    /// (D, a, r_e) of the pair (H–H the H₂-like well; H–He and He–He shallow, longer).
    fn pair(a: u8, b: u8) -> (f64, f64, f64) {
        match (a.min(b), a.max(b)) {
            (1, 1) => (0.17, 1.0, 1.4),
            (1, 2) => (0.012, 1.2, 2.6),
            _ => (0.006, 1.3, 3.0),
        }
    }

    fn energy_and_forces(pos: &[[f64; 3]], z: &[u8]) -> (f64, Vec<[f64; 3]>) {
        let r_cut = Basis::new(&[1]).r_cut;
        let mut e: f64 = z.iter().map(|&zi| SpeciesMorse::atom(zi)).sum();
        let mut f = vec![[0.0; 3]; pos.len()];
        for i in 0..pos.len() {
            for j in i + 1..pos.len() {
                let d: [f64; 3] = std::array::from_fn(|k| pos[j][k] - pos[i][k]);
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                if r >= r_cut {
                    continue;
                }
                let (dd, a, re) = SpeciesMorse::pair(z[i], z[j]);
                let x = (-a * (r - re)).exp();
                let v = dd * ((1.0 - x) * (1.0 - x) - 1.0);
                let dv = 2.0 * dd * a * (1.0 - x) * x;
                let c = 0.5 * (1.0 + (std::f64::consts::PI * r / r_cut).cos());
                let dc = -0.5 * std::f64::consts::PI / r_cut * (std::f64::consts::PI * r / r_cut).sin();
                e += v * c;
                let de = dv * c + v * dc;
                for k in 0..3 {
                    f[i][k] += de * d[k] / r;
                    f[j][k] -= de * d[k] / r;
                }
            }
        }
        (e, f)
    }
}

impl ForceProvider for SpeciesMorse {
    fn evaluate(&mut self, scene: &Scene, _should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        self.calls += 1;
        let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        let z: Vec<u8> = scene.nuclei.iter().map(|n| n.z).collect();
        let (energy, forces) = SpeciesMorse::energy_and_forces(&pos, &z);
        let electrons = z.iter().map(|&x| x as i64).sum::<i64>() - scene.charge as i64;
        let mut stats = EvalStats::default();
        stats.energy.total = energy;
        stats.face_density = self.face;
        stats.multiplicity = Some(scene.multiplicity.unwrap_or((electrons.rem_euclid(2) + 1) as u32));
        Ok(ForceResult { energy, forces, stats, display: None, orbitals: None })
    }

    fn name(&self) -> &'static str {
        "pair morse"
    }

    fn reset(&mut self) {}
}

/// A fresh, empty folder of a test's own, removed when it goes out of scope (declare it first:
/// the providers holding collections in it drop — and release their locks — before it).
struct CollectionRoot(std::path::PathBuf);

impl std::ops::Deref for CollectionRoot {
    type Target = std::path::Path;
    fn deref(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for CollectionRoot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn collection_root(name: &str) -> CollectionRoot {
    let dir = std::env::temp_dir().join(format!("as-shared-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    CollectionRoot(dir)
}

/// A surrogate over a shared [`SpeciesMorse`] teacher (default policy, tol 1e-6).
fn pair_student() -> (SurrogateKs, Arc<Mutex<SpeciesMorse>>) {
    let teacher = Shared::new(SpeciesMorse::new());
    let handle = teacher.handle();
    (SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_teacher_tol(1e-6), handle)
}

/// `scene` in an n³ cell at h = 0.25.
fn in_cell(mut scene: Scene, n: usize) -> Scene {
    scene.cell = Cell { n, h: 0.25 };
    scene
}

/// H₂ at r along a random direction, near the centre of a 33³ cell.
fn h2_at(r: f64, rng: &mut StdRng) -> Scene {
    let origin = [rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3)];
    dimer(r, unit_vector(rng), origin, 1, 1)
}

/// H₂ (r) with a He atom `d` from its centre, perpendicular, in a 41³ cell (another box than the
/// H₂ scene's 33³: the same surface).
fn h2_he_at(r: f64, d: f64, rng: &mut StdRng) -> Scene {
    let mut sc = in_cell(dimer(r, [1.0, 0.0, 0.0], [0.0; 3], 1, 1), 41);
    let tilt = rng.gen_range(-0.2..0.2);
    sc.nuclei.push(Nucleus::new(2, [tilt, d, 0.0]));
    sc
}

/// Teaches `s` (attached to a collection) H₂ at 40 teacher steps, refits; then 30 steps of H₂ + He
/// in a larger box, the He reference solved there, and refits again: (the H₂ steps, all steps).
fn teach_h2_then_h2_he(s: &mut SurrogateKs) -> (usize, usize) {
    let mut rng = StdRng::seed_from_u64(31);
    s.set_reference_solved(1, SpeciesMorse::atom(1), 33, Some(0.0));
    for i in 0..40 {
        let sc = h2_at(1.1 + 1.2 * i as f64 / 39.0, &mut rng);
        let r = s.evaluate(&sc, &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback, "no model yet: the teacher answers");
        assert!(!r.stats.notes.iter().any(|n| n.contains("session only")), "{:?}", r.stats.notes);
    }
    s.refit().expect("the H₂ fit");
    assert_eq!(s.basis().unwrap().species, vec![1]);
    // references are per box (fix round 2): the H reference of the 41³ box too
    s.set_reference_solved(1, SpeciesMorse::atom(1), 41, Some(0.0));
    for i in 0..30 {
        let sc = h2_he_at(1.2 + 0.6 * (i % 6) as f64 / 5.0, 2.4 + 0.15 * (i / 6) as f64, &mut rng);
        let r = s.evaluate(&sc, &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback);
        assert!(r.stats.notes.iter().any(|n| n.contains("species He is not in the model")), "{:?}", r.stats.notes);
    }
    s.set_reference_solved(2, SpeciesMorse::atom(2), 41, Some(0.0));
    (40, 70)
}

/// Plan Task 4 Step 1 (providers): with a collection attached, a second scene's answers append to
/// the same collection — another box, a new species — the basis grows to {H, He}, the refit
/// succeeds, and the earlier H₂ samples are still in the fit; each sample is filed with its box,
/// its solved multiplicity and its formula.
#[test]
fn collection_mode_learns_across_scenes() {
    let dir = collection_root("learn");
    let (mut s, teacher) = pair_student();
    s.attach_collection(&dir, &h2(1.4)).unwrap();
    let slug = s.provenance_for(&h2(1.4)).surface_key().slug();
    assert_eq!(s.collection().unwrap().folder(), dir.join(&slug));
    // a second handle of the same collection (read-only: this process holds the lock) never
    // replaces the writable one (fix round 1, review M6)
    let (second, r) = surrogate::Collection::open(&dir, &s.provenance_for(&h2(1.4)).surface_key()).unwrap();
    assert!(r.locked_by_other && second.read_only);
    s.attach(second).unwrap();
    assert!(!s.collection().unwrap().read_only, "the writable handle stays");
    let (h2_steps, all) = teach_h2_then_h2_he(&mut s);
    assert_eq!(teacher.lock().unwrap().calls, all);
    let c = s.collection().unwrap();
    assert_eq!(c.samples.len(), all, "every kept teacher answer was appended");
    assert_eq!(s.samples(), all);
    assert!((0..all).all(|i| s.collection_index(i) == Some(i)));
    assert!(c.samples[..h2_steps].iter().all(|x| x.grid_n == 33 && x.multiplicity == Some(1) && x.scene == "H2" && !x.multiplicity_forced));
    assert!(c.samples[h2_steps..].iter().all(|x| x.grid_n == 41 && x.multiplicity == Some(1) && x.scene == "H2He" && x.sample.z == vec![1, 1, 2]));
    assert_eq!(c.references.get(&(1, 33)).map(|r| (r.grid_n, r.face_density)), Some((Some(33), Some(0.0))));
    assert_eq!(c.references.keys().copied().collect::<Vec<_>>(), vec![(1, 33), (1, 41), (2, 41)], "per (species, box)");
    let rep = s.refit().expect("the refit with the new species succeeds");
    assert_eq!(s.basis().unwrap().species, vec![1, 2], "the basis grew");
    assert_eq!(rep.n_samples, all, "the H₂ samples are still used");
    println!("collection fit: {} samples, q′ {}, rmse E/atom {:.1e}, F {:.1e}; trusted {} ({:?})", rep.n_samples, rep.q_prime, rep.rmse_energy_per_atom, rep.rmse_force, s.trusted(), s.untrusted_reason());
    let st = Trainable::status(&s, &h2(1.4), &[[0.0; 3]; 2]);
    assert_eq!(st.model_source, sim::ModelSource::Collection { slug: slug.clone(), samples: all, read_only: false });
    assert_eq!(st.provenance_mismatch, None, "a 33³ scene shares the collection");
    assert!(st.provenance.as_deref().is_some_and(|p| p.contains("any box")), "{:?}", st.provenance);
    // the README says what it knows
    drop(s);
    let readme = std::fs::read_to_string(dir.join(&slug).join("README.txt")).unwrap();
    for want in ["70 samples", "He (Z 2)", "H–He", "H2He: 30 samples, grid 41³", "H2: 40 samples, grid 33³"] {
        assert!(readme.contains(want), "README lacks '{want}':\n{readme}");
    }
}

/// A collection reopened by a new provider rebuilds the same fit from what it stored: samples,
/// statuses and references — bit for bit the same predictions.
#[test]
fn a_reopened_collection_reproduces_the_same_fit() {
    let dir = collection_root("reopen");
    let probes = [h2(1.45), in_cell(h2(1.9), 41)];
    let mut rng = StdRng::seed_from_u64(5);
    let probes_he = [h2_he_at(1.4, 2.6, &mut rng), h2_he_at(1.6, 2.9, &mut rng)];
    let (first, report) = {
        let (mut s, _) = pair_student();
        s.attach_collection(&dir, &h2(1.4)).unwrap();
        teach_h2_then_h2_he(&mut s);
        let rep = s.refit().unwrap();
        let m = s.model().unwrap();
        let p: Vec<_> = probes.iter().chain(&probes_he).map(|sc| {
            let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
            let z: Vec<u8> = sc.nuclei.iter().map(|n| n.z).collect();
            m.predict(&pos, &z, true).unwrap()
        }).collect();
        (p, rep)
    };
    // a new provider (the first one dropped: the lock is free) attaches; with 70 samples below
    // Q′ = 176 it does not fit at once (as a samples-only file's install), the refit does
    let (mut s, teacher) = pair_student();
    s.attach_collection(&dir, &in_cell(h2(1.4), 41)).unwrap();
    assert_eq!(s.samples(), 70);
    assert!(s.model().is_none() && s.no_model().is_some_and(|n| n.missing.is_empty()), "every reference came with the collection");
    // the per-box references round-trip (fix round 2)
    let refs: Vec<((u8, usize), f64)> = s.collection().unwrap().references.iter().map(|(&k, r)| (k, r.energy)).collect();
    assert_eq!(refs, vec![((1, 33), SpeciesMorse::atom(1)), ((1, 41), SpeciesMorse::atom(1)), ((2, 41), SpeciesMorse::atom(2))]);
    assert_eq!(s.waiting_for_reference(), 0);
    let rep = s.refit().unwrap();
    assert_eq!(teacher.lock().unwrap().calls, 0, "no teacher solve: everything came from the collection");
    assert_eq!((rep.n_samples, rep.q_prime), (report.n_samples, report.q_prime));
    assert_eq!(rep.lambda.to_bits(), report.lambda.to_bits(), "{} vs {}", rep.lambda, report.lambda);
    assert_eq!(s.reference_energy(2), Some(SpeciesMorse::atom(2)));
    let m = s.model().unwrap();
    for (sc, a) in probes.iter().chain(&probes_he).zip(&first) {
        let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
        let z: Vec<u8> = sc.nuclei.iter().map(|n| n.z).collect();
        let b = m.predict(&pos, &z, true).unwrap();
        assert_eq!(a.energy.to_bits(), b.energy.to_bits(), "{} vs {}", a.energy, b.energy);
        assert_eq!(a.forces, b.forces);
        assert_eq!(a.energy_std.map(f64::to_bits), b.energy_std.map(f64::to_bits));
    }
}

/// Plan Task 4 Step 1 (providers): a charged scene keeps a per-scene store — attaching a
/// collection to it is refused with the reason and the provider learns it per scene; and a
/// provider learning into a collection hands a charged scene's step to the teacher without
/// keeping it, nothing appended, the note naming the reason.
#[test]
fn a_charged_scene_keeps_a_per_scene_store() {
    let dir = collection_root("charged");
    let mut rng = StdRng::seed_from_u64(7);
    let charged = |r: f64, rng: &mut StdRng| {
        let mut sc = h2_at(r, rng);
        sc.charge = 1;
        sc
    };
    // (a) the charged scene's own provider: no collection, a per-scene store
    let (mut p, _) = pair_student();
    let e = p.attach_collection(&dir, &charged(1.4, &mut rng)).unwrap_err();
    assert!(e.contains("charge 1: collections are neutral until M9"), "{e}");
    assert!(p.collection().is_none());
    for i in 0..5 {
        let r = p.evaluate(&charged(1.2 + 0.1 * i as f64, &mut rng), &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback);
    }
    assert_eq!(p.samples(), 5, "kept in the per-scene store");
    assert_eq!(p.store().unwrap().provenance.charge, 1);
    assert_eq!(Trainable::status(&p, &charged(1.4, &mut rng), &[[0.0; 3]; 2]).model_source, sim::ModelSource::None);
    assert!(std::fs::read_dir(&*dir).unwrap().next().is_none(), "no collection was created");
    // (b) a provider learning into the collection meets a charged scene
    let (mut q, _) = pair_student();
    q.attach_collection(&dir, &h2(1.4)).unwrap();
    for i in 0..3 {
        q.evaluate(&h2_at(1.3 + 0.1 * i as f64, &mut rng), &mut never()).unwrap();
    }
    assert_eq!(q.collection().unwrap().samples.len(), 3);
    let r = q.evaluate(&charged(1.4, &mut rng), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback, "the teacher answers the step");
    assert!(r.stats.notes.iter().any(|n| n.contains("charge 1: collections are neutral until M9") && n.contains("not kept")), "{:?}", r.stats.notes);
    assert_eq!(q.collection().unwrap().samples.len(), 3, "nothing appended");
    assert_eq!(q.samples(), 3, "and nothing kept in the shared store");
    // once a model stands, rule 3 refuses the charged scene the same way
    q.set_reference_solved(1, SpeciesMorse::atom(1), 33, Some(0.0));
    for i in 0..30 {
        q.evaluate(&h2_at(1.1 + 0.04 * i as f64, &mut rng), &mut never()).unwrap();
    }
    q.refit().unwrap();
    let n = q.collection().unwrap().samples.len();
    let r = q.evaluate(&charged(1.4, &mut rng), &mut never()).unwrap();
    assert!(r.stats.notes.iter().any(|n| n.contains("does not take this scene (charge 1: collections are neutral until M9)")), "{:?}", r.stats.notes);
    assert_eq!(q.collection().unwrap().samples.len(), n);
    let st = Trainable::status(&q, &charged(1.4, &mut rng), &[[0.0; 3]; 2]);
    assert!(st.provenance_mismatch.as_deref().is_some_and(|m| m.contains("charge 1")), "{:?}", st.provenance_mismatch);
    let plan = SamplingPlan::default();
    let e = Trainable::start_pass(&mut q, &charged(1.4, &mut rng), &plan, &mut sim::references::AtomReferences::default()).unwrap_err();
    assert!(e.contains("charge 1"), "{e}");
}

/// An answer the collection refuses — box-affected (its face density above the box rule even
/// while the provider keeps it at code 0), or of unknown box status — stays in the provider's
/// store for the session and is never written; a draft guess's audit waits outside (worded as
/// waiting, not refused) and is appended when a refit's `judge_held_out` promotes it; status
/// changes are appended as records.
#[test]
fn refused_answers_stay_per_scene_and_statuses_are_appended() {
    let dir = collection_root("refused");
    let (mut s, teacher) = pair_student();
    s.attach_collection(&dir, &h2(1.4)).unwrap();
    let mut rng = StdRng::seed_from_u64(11);
    s.evaluate(&h2_at(1.4, &mut rng), &mut never()).unwrap();
    teacher.lock().unwrap().face = Some(1.0);
    let r = s.evaluate(&h2_at(1.5, &mut rng), &mut never()).unwrap();
    assert!(r.stats.notes.iter().any(|n| n.contains("kept for this session only") && n.contains("box-affected sample")), "{:?}", r.stats.notes);
    teacher.lock().unwrap().face = None;
    let r = s.evaluate(&h2_at(1.6, &mut rng), &mut never()).unwrap();
    assert!(r.stats.notes.iter().any(|n| n.contains("kept for this session only") && n.contains("box status unknown")), "{:?}", r.stats.notes);
    assert_eq!(s.samples(), 3, "both kept in the session's store");
    assert_eq!((s.collection_index(0), s.collection_index(1), s.collection_index(2)), (Some(0), None, None));
    assert_eq!(s.collection().unwrap().samples.len(), 1);
    // a pass sample's refusal is summed for the pass, never said by a later live step
    let sc = h2_at(1.65, &mut rng);
    let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
    let (e, f) = SpeciesMorse::energy_and_forces(&pos, &[1, 1]);
    s.add_sample_answer(Sample { pos, z: vec![1, 1], energy: e, forces: f, tag: SampleTag::Scan, weight: 1.0 }, &sc, Some(1.0), Some(1)).unwrap();
    teacher.lock().unwrap().face = Some(0.0);
    let r = s.evaluate(&h2_at(1.55, &mut rng), &mut never()).unwrap();
    assert!(!r.stats.notes.iter().any(|n| n.contains("session only")), "{:?}", r.stats.notes);
    assert_eq!((s.samples(), s.collection().unwrap().samples.len()), (5, 2));
    let summary = s.take_collection_summary(0).unwrap();
    assert!(summary.contains("3 kept for this session only") && summary.contains("2 box-affected sample"), "{summary}");
    // a model over the window (40 pass-like samples, the reference with its box)
    s.set_reference_solved(1, SpeciesMorse::atom(1), 33, Some(0.0));
    for i in 0..40 {
        let sc = h2_at(1.1 + 1.2 * i as f64 / 39.0, &mut rng);
        let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
        let (e, f) = SpeciesMorse::energy_and_forces(&pos, &[1, 1]);
        s.add_sample_answer(Sample { pos, z: vec![1, 1], energy: e, forces: f, tag: SampleTag::Scan, weight: 1.0 }, &sc, Some(0.0), Some(1)).unwrap();
    }
    s.refit().unwrap();
    assert_eq!(s.collection().unwrap().samples.len(), 42);
    // a draft guess's audit inside the window: it waits, worded as waiting (review M2) ...
    let prov = s.provenance_for(&h2(1.4));
    let sc = h2_at(1.62, &mut rng);
    let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
    let (e, f) = SpeciesMorse::energy_and_forces(&pos, &[1, 1]);
    let mut notes = Vec::new();
    assert!(s.keep_sample(&pos, &[1, 1], e, &f, Some(0.0), Some(1), &prov, SampleTag::DraftAudit, &mut notes));
    let draft = s.samples() - 1;
    assert_eq!(s.collection_index(draft), None, "a draft guess is held out of the shared store too");
    assert!(notes.iter().any(|n| n.contains("held out of the strict fit") && n.contains("added if a refit promotes it")), "{notes:?}");
    assert!(!notes.iter().any(|n| n.contains("session only")), "{notes:?}");
    let summary = s.take_collection_summary(0).unwrap();
    assert!(summary.contains("1 draft-mode guesses") && !summary.contains("kept for this session only"), "{summary}");
    // ... and the next refit promotes it (strict mode's domain reaches it: `judge_held_out`), which
    // appends it (review M4: through the refit, not by hand)
    s.refit().unwrap();
    assert_eq!(s.quarantine.get(draft).copied().unwrap_or(Q_NONE), Q_NONE, "promoted by the refit");
    assert_eq!(s.collection_index(draft), Some(42), "promoted: appended");
    assert_eq!(s.collection().unwrap().samples[42].multiplicity, Some(1));
    assert_eq!(s.collection().unwrap().samples[42].sample.tag, SampleTag::Audit);
    // a quarantine code change of a collection sample is appended as a status record
    s.quarantine.resize(s.samples(), Q_NONE);
    s.quarantine[0] = Q_HELD;
    s.sync_collection_status();
    drop(s);
    let key = prov.surface_key();
    let (c, _) = surrogate::Collection::open(&dir, &key).unwrap();
    assert_eq!(c.samples.len(), 43);
    assert_eq!(c.quarantine[0], Q_HELD, "the hold survived the reopen");
    assert!(c.quarantine[1..].iter().all(|&q| q == Q_NONE));
}

/// Fix round 1 (review I1): an answer bit-identical to one the collection holds is not kept again —
/// in the shared store a twin doubles the geometry's weight and gives it a leave-one-out twin.
#[test]
fn a_duplicate_answer_is_not_kept_twice() {
    let dir = collection_root("duplicate");
    let (mut s, teacher) = pair_student();
    s.attach_collection(&dir, &h2(1.4)).unwrap();
    let sc = h2(1.45);
    s.evaluate(&sc, &mut never()).unwrap();
    assert_eq!((s.samples(), s.collection().unwrap().samples.len()), (1, 1));
    // the analytic teacher answers the same geometry bit for bit
    s.discontinuity();
    let r = s.evaluate(&sc, &mut never()).unwrap();
    assert_eq!(teacher.lock().unwrap().calls, 2, "the step itself is the teacher's");
    assert!(r.stats.notes.iter().any(|n| n.contains("sample not kept") && n.contains("identical teacher answer")), "{:?}", r.stats.notes);
    assert_eq!((s.samples(), s.collection().unwrap().samples.len()), (1, 1), "neither the session nor the collection holds a twin");
    // the pass's path refuses it too
    let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
    let (e, f) = SpeciesMorse::energy_and_forces(&pos, &[1, 1]);
    let again = Sample { pos, z: vec![1, 1], energy: e, forces: f, tag: SampleTag::Scan, weight: 1.0 };
    assert_eq!(s.add_sample_answer(again, &sc, Some(0.0), Some(1)), Err(StoreError::Duplicate));
    assert_eq!(s.samples(), 1);
}

/// Fix round 1 (review I1): a pass re-run on the same scene replays its seeded perturbations and its
/// scan points bit for bit; a geometry the collection holds is not solved again, and the pass's
/// summary counts the skips.
#[test]
fn a_rerun_pass_skips_the_geometries_the_collection_holds() {
    let dir = collection_root("rerun");
    let (mut s, teacher) = pair_student();
    let scene = h2(1.4);
    s.attach_collection(&dir, &scene).unwrap();
    let plan = SamplingPlan { perturbations: 6, hot_burst_steps: 0, scan_points_per_pair: 4, seed: 0, ..SamplingPlan::default() };
    let run = |s: &mut SurrogateKs| {
        Trainable::start_pass(s, &scene, &plan, &mut sim::references::AtomReferences::default()).unwrap();
        loop {
            match Trainable::pass_step(s, &mut never()) {
                Ok(Some(_)) => {}
                Ok(None) => break,
                Err(e) => panic!("pass: {e}"),
            }
        }
        Trainable::take_notes(s)
    };
    run(&mut s);
    let (calls, held) = (teacher.lock().unwrap().calls, s.collection().unwrap().samples.len());
    assert_eq!(held, 10, "6 perturbations and 4 scan points");
    let notes = run(&mut s);
    println!("{notes:#?}");
    assert_eq!(teacher.lock().unwrap().calls, calls, "the second pass solved nothing: references cached, every geometry held");
    assert_eq!(s.collection().unwrap().samples.len(), held, "nothing appended twice");
    assert_eq!(s.samples(), held, "nor kept twice in the session");
    assert!(notes.iter().any(|n| n.contains("10 geometries already in it, not solved again")), "{notes:?}");
    // a third pass on a moved scene solves again (other positions, other geometries)
    let moved = h2(1.5);
    Trainable::start_pass(&mut s, &moved, &plan, &mut sim::references::AtomReferences::default()).unwrap();
    while let Ok(Some(_)) = Trainable::pass_step(&mut s, &mut never()) {}
    assert!(s.collection().unwrap().samples.len() > held);
}

/// References per box (fix round 2, controller ruling): stored per (species, box) by the box rule
/// and odd n; an even-n or box-affected one is used for its box in this session only; a sample whose
/// box lacks a reference waits out of the fit until it arrives; the display reference is the largest
/// box's.
#[test]
fn references_are_kept_per_box_and_samples_wait_for_theirs() {
    let dir = collection_root("refs");
    let (mut s, _) = pair_student();
    s.attach_collection(&dir, &h2(1.4)).unwrap();
    let mut rng = StdRng::seed_from_u64(21);
    // an answer before any reference: kept, waiting
    let r = s.evaluate(&h2_at(1.4, &mut rng), &mut never()).unwrap();
    assert!(r.stats.notes.iter().any(|n| n.contains("waiting out of the fit") && n.contains("33³")), "{:?}", r.stats.notes);
    assert_eq!((s.samples(), s.waiting_for_reference()), (1, 1));
    s.set_reference_solved(1, -0.47, 48, Some(0.0));
    assert_eq!(s.collection().unwrap().references.len(), 0, "an even box is not stored");
    s.set_reference_solved(1, -0.4781, 41, Some(9.0));
    assert_eq!(s.collection().unwrap().references.len(), 0, "nor a box-affected one");
    assert_eq!(s.waiting_for_reference(), 1, "no reference of the 33³ box yet");
    s.set_reference_solved(1, -0.4782, 33, Some(1e-9));
    assert_eq!(s.waiting_for_reference(), 0, "its box's reference arrived");
    s.set_reference_solved(1, -0.4783, 33, Some(1e-9));
    assert_eq!(s.collection().unwrap().references.get(&(1, 33)).map(|r| r.energy), Some(-0.4782), "the same box keeps its reference");
    s.set_reference_solved(1, -0.4784, 57, Some(1e-12));
    assert_eq!(s.display_reference(1), Some(-0.4784), "the display reference is the largest box's");
    assert_eq!(s.collection().unwrap().references.len(), 2);
    let notes = Trainable::take_notes(&mut s);
    assert!(notes.iter().any(|n| n.contains("even grid n = 48") && n.contains("this session only")), "{notes:?}");
}

/// A teacher whose isolated atom depends on the box (the measured confinement of series (3) of
/// `references.md`, "M8a collection references"): every atom's energy is raised by
/// `BOX_SHIFT`·exp(−L/2), L = (n − 1)h the box length; the pairwise surface is [`SpeciesMorse`]'s.
#[derive(Clone, Debug)]
struct BoxMorse;

const BOX_SHIFT: f64 = 0.1;

impl BoxMorse {
    fn atom_shift(cell: &Cell) -> f64 {
        BOX_SHIFT * (-0.5 * (cell.n - 1) as f64 * cell.h).exp()
    }
}

impl ForceProvider for BoxMorse {
    fn evaluate(&mut self, scene: &Scene, _should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        let z: Vec<u8> = scene.nuclei.iter().map(|n| n.z).collect();
        let (e, forces) = SpeciesMorse::energy_and_forces(&pos, &z);
        let energy = e + z.len() as f64 * BoxMorse::atom_shift(&scene.cell);
        let mut stats = EvalStats::default();
        stats.energy.total = energy;
        stats.face_density = Some(0.0);
        stats.multiplicity = Some(1 + (z.iter().map(|&x| x as u32).sum::<u32>() % 2));
        Ok(ForceResult { energy, forces, stats, display: None, orbitals: None })
    }

    fn name(&self) -> &'static str {
        "pair morse"
    }

    fn reset(&mut self) {}
}

/// Fix round 2 (controller ruling): the same H₂ geometries taught in a 33³ (8 Bohr) and a 49³
/// (12 Bohr) box, whose isolated atoms differ by the box shift. With each sample fitted against its
/// own box's references the binding energies of the twin geometries agree and the fit is exact;
/// with one reference shared by both boxes (the RED case: the large box's reference used for the
/// small box's samples too) the twins disagree by 2·Δ and the fit carries that as noise.
#[test]
fn per_box_references_make_the_binding_energies_agree_across_boxes() {
    let r: Vec<f64> = (0..36).map(|i| 1.1 + 1.2 * i as f64 / 35.0).collect();
    let boxes = [33usize, 49];
    let shift = |n: usize| BoxMorse::atom_shift(&Cell { n, h: 0.25 });
    let delta = shift(33) - shift(49);
    let run = |shared_reference: bool| {
        let dir = collection_root(if shared_reference { "box-shared" } else { "box-own" });
        let mut s = SurrogateKs::new(Box::new(BoxMorse), FallbackPolicy::default()).with_teacher_tol(1e-6);
        s.attach_collection(&dir, &h2(1.4)).unwrap();
        for &n in &boxes {
            // the box's own isolated atom — or, shared, the 12 Bohr box's for both
            let e_ref = SpeciesMorse::atom(1) + if shared_reference { shift(49) } else { shift(n) };
            s.set_reference_solved(1, e_ref, n, Some(0.0));
        }
        let mut rng = StdRng::seed_from_u64(3);
        let mut twins = Vec::new();
        for &ri in &r {
            let dir_v = unit_vector(&mut rng);
            for &n in &boxes {
                let sc = in_cell(dimer(ri, dir_v, [0.0; 3], 1, 1), n);
                s.evaluate(&sc, &mut never()).unwrap();
                s.discontinuity();
            }
            let st = s.store().unwrap();
            twins.push((st.samples[st.len() - 2].energy, st.samples[st.len() - 1].energy));
        }
        let rep = s.refit().unwrap();
        let worst = twins.iter().map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        (worst, rep.rmse_energy_per_atom, rep.n_samples)
    };
    let (own_worst, own_rmse, n_own) = run(false);
    let (shared_worst, shared_rmse, n_shared) = run(true);
    println!(
        "per-box references: twin energies differ by at most {own_worst:.1e} Ha, fit rmse {own_rmse:.1e} Ha/atom ({n_own} samples); one shared reference: {shared_worst:.2e} Ha (2Δ = {:.2e}), rmse {shared_rmse:.1e} Ha/atom ({n_shared} samples)",
        2.0 * delta
    );
    assert!(delta > 1e-3, "the premise: the boxes' atoms differ by {delta:.1e} Ha");
    assert!(own_worst <= 1e-12, "own box: the twins' fit energies agree ({own_worst:.1e})");
    assert!((shared_worst - 2.0 * delta).abs() <= 1e-12, "shared reference: the twins differ by the two atoms' box shift ({shared_worst:.2e})");
    assert!(own_rmse < 0.1 * shared_rmse, "the shared reference's disagreement is fitted as noise: {own_rmse:.1e} vs {shared_rmse:.1e}");
}

/// A model file switches the provider to it (a specialised, per-scene model): the collection is let
/// go (its lock with it) and the source says File; saving is refused while a collection is attached;
/// an import goes through the provider's own handle and joins the store.
#[test]
fn a_model_file_switches_off_the_collection_and_imports_join_it() {
    let dir = collection_root("switch");
    let mut rng = StdRng::seed_from_u64(13);
    // a per-scene model of H₂ (no collection), saved
    let file = dir.join("h2.model");
    {
        let (mut p, _) = pair_student();
        p.set_reference_energy(1, SpeciesMorse::atom(1));
        for i in 0..40 {
            p.evaluate(&h2_at(1.1 + 0.03 * i as f64, &mut rng), &mut never()).unwrap();
        }
        p.refit().unwrap();
        p.save_sidecar(&file).unwrap();
    }
    let (mut s, _) = pair_student();
    s.attach_collection(&dir, &h2(1.4)).unwrap();
    for i in 0..3 {
        s.evaluate(&h2_at(1.2 + 0.1 * i as f64, &mut rng), &mut never()).unwrap();
    }
    let e = s.save_sidecar(&dir.join("x.model")).unwrap_err();
    assert_eq!(e.kind(), std::io::ErrorKind::Unsupported, "{e}");
    // the import: through the provider's handle, into its store
    let summary = Trainable::import_model(&mut s, &file).unwrap();
    println!("import: {summary:?}");
    assert_eq!(summary.imported, 40, "{summary:?}");
    assert_eq!(s.samples(), 43);
    assert_eq!(s.collection().unwrap().samples.len(), 43);
    assert!(summary.notes.iter().any(|n| n.contains("not taken")), "the file's reference has no box: {summary:?}");
    // the load switches to the file
    Trainable::load(&mut s, &file, &h2(1.4)).unwrap();
    assert!(s.collection().is_none());
    assert_eq!(Trainable::status(&s, &h2(1.4), &[[0.0; 3]; 2]).model_source, sim::ModelSource::File(file.clone()));
    assert_eq!(s.samples(), 40);
    let key = s.provenance_for(&h2(1.4)).surface_key();
    let (c, r) = surrogate::Collection::open(&dir, &key).unwrap();
    assert!(!r.locked_by_other && !c.read_only, "the lock was released");
    assert_eq!(c.samples.len(), 43);
}

/// Measurement (M8a Task 4a, the controller's ask; reported, not asserted): the cost of the class
/// rule's leave-one-out pass (`compute_class_loo_full`, run by every refit) and of the refit itself
/// at collection sizes, two species (H, He: Q′ 176) on the [`SpeciesMorse`] teacher; `LOO_S4=1` adds
/// four species (Q′ 1232) at one size. `cargo test -p providers --release class_loo_cost -- --ignored --nocapture`.
#[test]
#[ignore = "measurement: cargo test -p providers --release class_loo_cost -- --ignored --nocapture"]
fn class_loo_cost_at_collection_sizes() {
    let run = |species: &[u8], n: usize| {
        let (mut s, _) = pair_student();
        for &z in species {
            s.set_reference_energy(z, SpeciesMorse::atom(z));
        }
        let mut rng = StdRng::seed_from_u64(17);
        let t0 = Instant::now();
        for i in 0..n {
            // 2–4 atoms of the species within a 3 Bohr ball, no pair closer than 1.0 Bohr
            let k = 2 + i % 3;
            let mut pos: Vec<[f64; 3]> = Vec::new();
            while pos.len() < k {
                let p: [f64; 3] = std::array::from_fn(|_| rng.gen_range(-1.5..1.5));
                if pos.iter().all(|q| (0..3).map(|a| (p[a] - q[a]).powi(2)).sum::<f64>() > 1.0) {
                    pos.push(p);
                }
            }
            let z: Vec<u8> = (0..k).map(|j| species[(i + j) % species.len()]).collect();
            let (energy, forces) = SpeciesMorse::energy_and_forces(&pos, &z);
            let sc = Scene { nuclei: pos.iter().zip(&z).map(|(&p, &zi)| Nucleus::new(zi, p)).collect(), ..h2(1.4) };
            s.add_sample_with_face(Sample { pos, z, energy, forces, tag: SampleTag::Perturbation, weight: 1.0 }, &sc, Some(0.0)).unwrap();
        }
        let t_add = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let rep = s.refit().expect("refit");
        let t_refit = t1.elapsed().as_secs_f64();
        let t2 = Instant::now();
        let (loo, _, _) = s.compute_class_loo_full();
        let t_loo = t2.elapsed().as_secs_f64();
        println!(
            "S = {} (Q′ {}), {n} samples: store {t_add:.2} s, refit {t_refit:.2} s (its LOO pass included), class LOO alone {t_loo:.2} s = {:.2} ms/sample; classes {:?}",
            species.len(),
            rep.q_prime,
            1e3 * t_loo / n as f64,
            loo.iter().map(|c| (c.class, c.samples)).collect::<Vec<_>>()
        );
    };
    for n in [500, 1000, 2000, 4000] {
        run(&[1, 2], n);
    }
    if std::env::var("LOO_S4").is_ok() {
        run(&[1, 2, 3, 4], 1500);
    }
}

/// Measurement (M8a Task 4a, the Task 3 review's physics (c) (iv), and fix round 1's I2; reported,
/// not asserted; `validate/references.md` "M8a collection references", series (1)–(3)): an isolated
/// atom's energy (1) at the cell centre on a grid node (odd n) and at a cube centre (even n, the same
/// box ± h); (2) at the four high-symmetry sites of one odd box — the lattice phase alone; (3) on the
/// node in boxes of growing length — the box alone. GPU, tol 1e-6 (the sample tolerance the references are solved at), a 12 Bohr box.
/// `cargo test -p providers --release reference_lattice_phase -- --ignored --nocapture`.
#[test]
#[ignore = "measurement (GPU): cargo test -p providers --release reference_lattice_phase -- --ignored --nocapture"]
fn reference_lattice_phase_measurement() {
    let Some(ctx) = gpu::GpuContext::new_headless() else {
        eprintln!("no GPU adapter — skipped");
        return;
    };
    let cases = [(1u8, 2u32, sim::IonicPolicy::None), (8, 3, sim::IonicPolicy::None), (8, 3, sim::IonicPolicy::CoresOnly)];
    for (h, n_odd) in [(0.25, 49usize), (0.20, 61), (0.15, 81)] {
        for &(z, m, ionic) in &cases {
            let mut out = Vec::new();
            for n in [n_odd, n_odd - 1] {
                let mut t = crate::ks::GpuKs::new(ctx.clone(), 1e-6);
                t.set_display(false);
                let sc = Scene { nuclei: vec![Nucleus::new(z, [0.0; 3])], cell: Cell { n, h }, charge: 0, multiplicity: Some(m), sigma: None, external_field: [0.0; 3], ionic };
                let t0 = Instant::now();
                match t.evaluate(&sc, &mut never()) {
                    Ok(r) => out.push(Some((r.energy, r.stats.face_density.unwrap_or(f64::NAN), r.stats.scf_steps, t0.elapsed().as_secs_f64()))),
                    Err(e) => {
                        println!("Z {z} M {m} {ionic:?} h {h} n {n}: failed ({e})");
                        out.push(None);
                    }
                }
            }
            if let [Some(a), Some(b)] = &out[..] {
                println!(
                    "Z {z} M {m} {ionic:?} h {h}: node (n {n_odd}) {:.8} Ha, face {:.1e}, {} SCF, {:.1} s | cube centre (n {}) {:.8} Ha, face {:.1e}, {} SCF, {:.1} s | E_cube − E_node {:+.2e} Ha",
                    a.0, a.1, a.2, a.3, n_odd - 1, b.0, b.1, b.2, b.3, b.0 - a.0
                );
            }
            // fix round 1 (review I2): every high-symmetry site in the same odd box — node, edge
            // centre (h/2, 0, 0), face centre (h/2, h/2, 0), cube centre (h/2, h/2, h/2)
            let half = 0.5 * h;
            let mut sites = Vec::new();
            for (name, p) in [("node", [0.0; 3]), ("edge", [half, 0.0, 0.0]), ("face", [half, half, 0.0]), ("cube", [half, half, half])] {
                let mut t = crate::ks::GpuKs::new(ctx.clone(), 1e-6);
                t.set_display(false);
                let sc = Scene { nuclei: vec![Nucleus::new(z, p)], cell: Cell { n: n_odd, h }, charge: 0, multiplicity: Some(m), sigma: None, external_field: [0.0; 3], ionic };
                match t.evaluate(&sc, &mut never()) {
                    Ok(r) => sites.push((name, r.energy)),
                    Err(e) => {
                        println!("Z {z} M {m} {ionic:?} h {h} n {n_odd} {name}: failed ({e})");
                        break;
                    }
                }
            }
            if sites.len() == 4 {
                let e0 = sites[0].1;
                let words: Vec<String> = sites.iter().map(|(n, e)| format!("{n} {:+.2e}", e - e0)).collect();
                let spread = sites.iter().map(|s| s.1).fold(f64::NEG_INFINITY, f64::max) - sites.iter().map(|s| s.1).fold(f64::INFINITY, f64::min);
                println!("Z {z} M {m} {ionic:?} h {h} n {n_odd}: sites vs the node (Ha): {}; spread {spread:.2e}", words.join(", "));
            }
        }
    }
    // fix round 1: the box length at a fixed site (the node) and h — the part of an isolated atom's
    // energy that depends on the cell beyond the box rule's face density
    for (z, m, ionic, h, ns) in [(1u8, 2u32, sim::IonicPolicy::None, 0.25, vec![33usize, 41, 49, 57, 65, 81]), (8, 3, sim::IonicPolicy::CoresOnly, 0.20, vec![41, 51, 61, 71, 81])] {
        let mut rows = Vec::new();
        for n in ns {
            let mut t = crate::ks::GpuKs::new(ctx.clone(), 1e-6);
            t.set_display(false);
            let sc = Scene { nuclei: vec![Nucleus::new(z, [0.0; 3])], cell: Cell { n, h }, charge: 0, multiplicity: Some(m), sigma: None, external_field: [0.0; 3], ionic };
            match t.evaluate(&sc, &mut never()) {
                Ok(r) => rows.push((n, (n - 1) as f64 * h, r.energy, r.stats.face_density.unwrap_or(f64::NAN))),
                Err(e) => println!("Z {z} {ionic:?} h {h} n {n}: failed ({e})"),
            }
        }
        let last = rows.last().map_or(f64::NAN, |r| r.2);
        for (n, l, e, rho) in rows {
            println!("box series Z {z} {ionic:?} h {h}: n {n} (L {l:.2} Bohr) E {e:.8} Ha, E − E(largest) {:+.2e} Ha, face {rho:.1e}", e - last);
        }
    }
}

// ---- fix round 3 (re-review 2): references only from their own surface, the driver's delivered,
// a reference's arrival booked ----

/// Waits (≤ 5 s) for the auditor's reference answers handed on to the driver.
fn await_references(s: &mut SurrogateKs, want: usize) -> Vec<(u8, Result<AtomReference, ProviderError>)> {
    let t0 = Instant::now();
    let mut got = Vec::new();
    while got.len() < want {
        got.extend(s.take_atom_references());
        assert!(t0.elapsed().as_secs_f64() < 5.0, "no reference answer within 5 s ({} of {want})", got.len());
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    got
}

/// Re-review 2, item 1: a reference solved for a scene of another surface (another h) is handed to
/// the driver that asked, and never becomes one of the collection's; the same request from a scene
/// of the collection's surface is stored for its box; a box reference the provider asks for itself
/// is stored and not handed on.
#[test]
fn a_reference_from_another_surface_never_enters_the_collection() {
    let dir = collection_root("other-surface");
    let mut s = SurrogateKs::new(Box::new(SpeciesMorse::new()), FallbackPolicy::default()).with_teacher_tol(1e-6).with_auditor(Box::new(SpeciesMorse::new()));
    s.attach_collection(&dir, &h2(1.4)).unwrap();
    let mass = Nucleus::new(1, [0.0; 3]).mass;
    // another surface: h 0.2 (the collection's is 0.25)
    let mut other = h2(1.4);
    other.cell = Cell { n: 41, h: 0.2 };
    let r = s.evaluate(&other, &mut never()).unwrap();
    assert!(r.stats.notes.iter().any(|n| n.contains("does not take this scene")), "{:?}", r.stats.notes);
    assert!(s.request_atom_reference(1, mass, &other));
    let got = await_references(&mut s, 1);
    assert!(matches!(got[0], (1, Ok(_))), "the driver gets its answer");
    assert!(s.collection().unwrap().references.is_empty(), "another surface's atom never becomes a collection reference");
    assert!(s.box_refs_missing(&[1], 41).contains(&1), "nor a reference of the 41³ box in use");
    // the collection's own surface: the driver's request is stored too
    assert!(s.request_atom_reference(1, mass, &in_cell(h2(1.4), 41)));
    await_references(&mut s, 1);
    assert_eq!(s.collection().unwrap().references.keys().copied().collect::<Vec<_>>(), vec![(1, 41)]);
    // a box the provider asks for itself (its own scene lacks it): stored, not handed on
    s.evaluate(&in_cell(h2(1.4), 49), &mut never()).unwrap();
    let t0 = Instant::now();
    while !s.collection().unwrap().references.contains_key(&(1, 49)) {
        s.poll_side_jobs();
        assert!(t0.elapsed().as_secs_f64() < 5.0, "the provider's own request was not answered");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(s.take_atom_references().is_empty(), "the provider's own answer is not handed to the driver");
}

/// Re-review 2, item 2: in collection mode the driver's bond reference — asked after the provider
/// queued the same (Z, box) for itself — is delivered and the driver's reference job completes
/// (before the fix it was swallowed and the driver waited for ever).
#[test]
fn the_drivers_reference_is_answered_in_collection_mode() {
    let dir = collection_root("driver-ref");
    let bonded = || sim::Gaussians::per_atom(0.3, 0.5).with_pair_bump(0.15, 0.5, 3.0);
    let mut s = SurrogateKs::new(Box::new(bonded()), FallbackPolicy::default()).with_auditor(Box::new(bonded()));
    s.attach_collection(&dir, &h2(1.6)).unwrap();
    let (p, _sh) = driven(s);
    let mut d = Dynamics::new(h2(1.6), p, 0.25, 1);
    assert_eq!(d.snapshot().bonds_pending, vec![1], "the constructor's density queues H");
    let t0 = Instant::now();
    let t = loop {
        d.apply(DynCommand::Step(1));
        let t = d.tick(f64::INFINITY, &mut never());
        if t.bonds_pending.is_empty() && t.bonds.is_some() {
            break t;
        }
        assert!(t0.elapsed().as_secs_f64() < 5.0, "the driver's reference never arrived: pending {:?}, notes {:?}", t.bonds_pending, t.notes);
        std::thread::sleep(std::time::Duration::from_millis(1));
    };
    assert_eq!(t.bonds.as_ref().map(|g| g.bonds.len()), Some(1));
    // one job per (Z, box): a request for the same Z in another box is its own job, answered too
    let s2 = _sh;
    let mut g = surrogate_of(&s2);
    let mass = Nucleus::new(1, [0.0; 3]).mass;
    assert!(g.request_atom_reference(1, mass, &in_cell(h2(1.6), 41)));
    assert!(g.request_atom_reference(1, mass, &in_cell(h2(1.6), 49)));
    let got = await_references(&mut g, 2);
    assert_eq!(got.len(), 2);
}

/// Re-review 2, item 3: a box reference arriving mid-run moves the canonical surface under the
/// teacher-surface reports; the change is booked, so `E_conserved` stays continuous (the analytic
/// box-dependent teacher: the reported energy shifts by 2·Δ ≈ 3.2e-3 Ha at the arrival).
#[test]
fn a_reference_arriving_mid_run_keeps_e_conserved_continuous() {
    let dir = collection_root("arrival");
    let scene = in_cell(h2(1.6), 33);
    let mut s = SurrogateKs::new(Box::new(BoxMorse), FallbackPolicy::default()).with_teacher_tol(1e-6);
    s.attach_collection(&dir, &scene).unwrap();
    // the canonical H reference: a larger box's
    s.set_reference_solved(1, SpeciesMorse::atom(1) + BoxMorse::atom_shift(&Cell { n: 49, h: 0.25 }), 49, Some(0.0));
    let (p, h) = driven(s);
    let mut d = Dynamics::new(scene.clone(), p, 0.25, 1);
    let mut e = vec![d.snapshot().energy_conserved];
    for k in 0..40 {
        if k == 20 {
            // the 33³ box's reference arrives (a pass's, or the auditor's)
            surrogate_of(&h).set_reference_solved(1, SpeciesMorse::atom(1) + BoxMorse::atom_shift(&scene.cell), 33, Some(0.0));
        }
        d.apply(DynCommand::Step(1));
        e.push(d.tick(f64::INFINITY, &mut never()).energy_conserved);
    }
    let shift = 2.0 * (BoxMorse::atom_shift(&scene.cell) - BoxMorse::atom_shift(&Cell { n: 49, h: 0.25 }));
    let (verlet_spread, verlet_step) = verlet_reference(&scene, BoxMorse, 40);
    let step = e.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f64, f64::max);
    let spread = e.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - e.iter().cloned().fold(f64::INFINITY, f64::min);
    println!("reference arrival: the reported surface moved by {shift:.2e} Ha; E_conserved max step {step:.2e}, spread {spread:.2e} (plain Verlet: step {verlet_step:.2e}, spread {verlet_spread:.2e})");
    assert!(shift > 1e-3, "the premise");
    assert!(step <= 2.0 * verlet_step + 1e-12 && spread <= 2.0 * verlet_spread + 1e-12, "E_conserved continuous across the arrival: step {step:.2e}, spread {spread:.2e}");
}

/// A stub teacher of the GPU Kohn-Sham's name (a collection opens for its own teacher's name);
/// never asked to solve.
struct StubKsGpu;

impl ForceProvider for StubKsGpu {
    fn evaluate(&mut self, _scene: &Scene, _should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        Err(ProviderError::Invalid("a stub teacher: never asked".into()))
    }
    fn name(&self) -> &'static str {
        "Kohn-Sham (GPU)"
    }
    fn reset(&mut self) {}
}

/// The M8a user-test regression (`references.md` "M8a stuck collection"): the user's H-only
/// collection (`testdata/stuck-h-collection`, 578 GPU teacher answers: H, H₂ at 81³ and 101³,
/// H₃) as the live session held it — the collection's 8 held samples and the `degree_cap` 6 an
/// earlier recovery had set. Under the cap one sample (H₂ at 5.585 Bohr in the 101³ box, its
/// restricted-spin label 40 mHa above two atoms) drives the energy noise to 2e-2 Ha/atom and the
/// misfit spreads over every H₂ sample (−25 mHa/atom each, the sample itself −6 mHa/atom): the
/// worst offenders are 380 samples, past the quarantine's 10 % cap, so the old recovery did
/// nothing — untrusted at every refit, and (every fallback a copy of a stored sample) no refit
/// changed it. The recovery now re-judges the cap against the default basis (steps 1b/3): the
/// refit ends trusted, every held sample is written to the collection as a status record, and a
/// reopened collection holds the same codes.
#[test]
fn a_recovery_cap_that_spreads_the_misfit_does_not_leave_the_collection_stuck() {
    const SLUG: &str = "lsda_h0.15_s0.15_tol1e-6_kohn-sham-gpu_all-54eedc40ddc5";
    let dir = collection_root("stuck");
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/stuck-h-collection").join(SLUG);
    std::fs::create_dir_all(dir.join(SLUG)).unwrap();
    for f in ["key.txt", "references.txt", "samples.bin", "history.bin"] {
        std::fs::copy(src.join(f), dir.join(SLUG).join(f)).unwrap();
    }
    let key = surrogate::SurfaceKey {
        functional: "lsda".into(),
        h: 0.15,
        sigma: 0.15,
        tol: 1e-6,
        teacher: "Kohn-Sham (GPU)".into(),
        ionic: "all/30a74d9bda495668/shared1e-3/1e-1/stop2x5e-8".into(),
    };
    assert_eq!(key.slug(), SLUG);
    let (c, _) = surrogate::Collection::open(&dir, &key).unwrap();
    assert!(!c.read_only);
    let live_codes = c.quarantine.clone();
    assert_eq!((c.samples.len(), live_codes.iter().filter(|&&q| q == Q_HELD).count()), (578, 8), "the user's collection");
    let mut s = SurrogateKs::new(Box::new(StubKsGpu), FallbackPolicy::default()).with_teacher_tol(1e-6);
    s.attach(c).unwrap();
    // the live session's state: the collection's codes, the cap an earlier recovery set
    s.quarantine = live_codes.clone();
    s.degree_cap = Some(RECOVERY_DEGREE_CAP);
    s.species_dirty = true;
    s.fit_store().unwrap();
    let r = s.report().unwrap();
    println!(
        "the live state: Q′ {}, n {}, rmse E/atom {:.1e}, F {:.1e}; {:?}",
        r.q_prime,
        r.n_samples,
        r.rmse_energy_per_atom,
        r.rmse_force,
        s.untrusted_reason()
    );
    assert!(!s.trusted() && s.untrusted_reason().is_some_and(|w| w.contains("training energy residual")), "the premise: {:?}", s.untrusted_reason());
    assert!(s.worst_offenders().len() as f64 > QUARANTINE_MAX_FRACTION * 578.0, "the premise: the capped fit names no offender within the cap");
    // the next refit recovers
    let rep = s.refit().unwrap();
    let t = s.training_residuals().unwrap();
    println!(
        "after the refit: cap {:?}, Q′ {}, n {}, worst E/atom {:.1e} (allowed {:.1e}), trusted {} {:?}; {:?}",
        s.degree_cap,
        rep.q_prime,
        rep.n_samples,
        t.energy_per_atom,
        s.policy.energy_std_per_atom,
        s.trusted(),
        s.untrusted_reason(),
        s.recovery_note()
    );
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    assert!(t.energy_per_atom <= s.policy.energy_std_per_atom);
    assert!(s.quarantined() as f64 <= QUARANTINE_MAX_FRACTION * 578.0);
    assert!(s.is_quarantined(575), "the stretched H₂ label is held");
    // the codes are the collection's: written as status records, read back on reopening
    let held: Vec<usize> = (0..578).filter(|&i| s.is_quarantined(i)).collect();
    println!("held: {held:?}");
    assert!(held.iter().all(|&i| s.collection().unwrap().quarantine[i] == Q_HELD));
    drop(s);
    let (again, _) = surrogate::Collection::open(&dir, &key).unwrap();
    assert_eq!((0..578).filter(|&i| again.quarantine[i] == Q_HELD).collect::<Vec<_>>(), held);
}
