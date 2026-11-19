//! The predictive variance on the composition's support (spec `2026-11-20-variance-support-design.md`,
//! review I3a): a provider session on a two-species (H, He) model, where H-only scenes take the
//! support path (S({H}) = 29 of Q′ = 176 ≤ Q′/2), decides exactly as with the support path forced
//! off (`surrogate::model::force_full_variance_path`). The trace holds every decision and answer
//! bit for bit; only the reported stds are rounded (to 9 significant digits), since the support
//! path agrees with the full one to rounding, not bit for bit.
use super::*;

/// The session: a trained H₂ / HeH store, then trained and untrained H₂, H₃, HeH and inflated
/// posterior steps. Returns the trace and how many H-only steps found a support factor held.
fn session() -> (String, usize) {
    let mut t = String::new();
    let policy = FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() };
    let mut s = SurrogateKs::new(Box::new(SpeciesMorse::new()), policy).with_teacher_tol(1e-6);
    s.set_reference_energy(1, SpeciesMorse::atom(1));
    s.set_reference_energy(2, SpeciesMorse::atom(2));
    let add = |s: &mut SurrogateKs, sc: &Scene| {
        let (pos, z) = pos_z(sc);
        let (e, f) = SpeciesMorse::energy_and_forces(&pos, &z);
        s.add_sample(Sample { pos, z, energy: e, forces: f, tag: SampleTag::Scan, weight: 1.0, force_mask: None, use_energy: true }, sc).unwrap();
    };
    let mut rng = StdRng::seed_from_u64(0x5_0991);
    for i in 0..100 {
        add(&mut s, &h2_at(1.1 + 1.2 * i as f64 / 99.0, &mut rng));
    }
    for i in 0..30 {
        let origin = [rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3)];
        add(&mut s, &dimer(2.0 + 1.4 * i as f64 / 29.0, unit_vector(&mut rng), origin, 1, 2));
    }
    s.refit().unwrap();
    let mut on_support = 0;
    for k in 0..120 {
        if k == 60 {
            always_unsure(&mut s); // rule 6 until the next refit
        }
        let scene = match k % 6 {
            // trained H₂, and stretched past the training range
            0 | 1 => h2_at(1.15 + 1.1 * rng.gen::<f64>(), &mut rng),
            2 => h2_at(2.3 + 0.8 * rng.gen::<f64>(), &mut rng),
            // H₃ (no three-atom sample yet: the envelope, 5d or the class rule until learned)
            3 => {
                let mut c = h2(1.4 + 0.2 * rng.gen::<f64>());
                c.nuclei.push(Nucleus::new(1, [0.0, 1.5 + 0.3 * rng.gen::<f64>(), 0.0]));
                c
            }
            // HeH (the composition {H, He}: the full path)
            4 => dimer(2.2 + 1.0 * rng.gen::<f64>(), unit_vector(&mut rng), [0.0; 3], 1, 2),
            _ => h2_at(1.3 + 0.6 * rng.gen::<f64>(), &mut rng),
        };
        let mut out = s.evaluate(&scene, &mut never());
        if let Ok(r) = out.as_mut() {
            let round = |x: f64| format!("{x:.8e}").parse::<f64>().unwrap();
            r.stats.uncertainty = r.stats.uncertainty.map(round);
            r.stats.atom_uncertainty = r.stats.atom_uncertainty.as_ref().map(|v| v.iter().map(|&x| round(x)).collect());
        }
        trace_eval(&mut t, &format!("step {k}"), &s, &out);
        if scene.nuclei.iter().all(|n| n.z == 1) && s.model.as_ref().is_some_and(|m| m.variance_support.support_for(&[0]).is_some()) {
            on_support += 1;
        }
    }
    use std::fmt::Write;
    let _ = writeln!(t, "variance checks {}", s.variance_checks());
    (t, on_support)
}

#[test]
fn a_two_species_session_decides_the_same_on_the_support_and_the_full_path() {
    /// Turns the process-wide hook back off however the forced pass ends (other tests in this
    /// binary would otherwise keep the full path, which decides the same but is slower).
    struct Forced;
    impl Drop for Forced {
        fn drop(&mut self) {
            surrogate::model::force_full_variance_path(false);
        }
    }
    let (support, on_support) = session();
    let (full, forced) = {
        let _off = Forced;
        surrogate::model::force_full_variance_path(true);
        session()
    };
    println!(
        "{on_support} H-only steps on the support path; {} model answers, {} fallbacks; {}",
        support.matches("source: Surrogate").count(),
        support.matches("source: Fallback").count(),
        support.lines().last().unwrap_or("")
    );
    assert!(on_support > 20, "the support path was taken ({on_support} H-only steps)");
    assert_eq!(forced, 0, "forced off: no support factor is ever built");
    assert!(support.contains("Fallback") && support.contains("Surrogate"), "both answers occur");
    if support != full {
        for (a, b) in support.lines().zip(full.lines()) {
            assert_eq!(a, b, "the first differing line");
        }
    }
    assert_eq!(support, full);
}
