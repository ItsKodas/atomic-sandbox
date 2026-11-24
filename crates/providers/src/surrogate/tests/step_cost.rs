//! The estimated step's cost on a collection (2026-11-17, the variance-step speed-up): H₂ near
//! 1.4 Bohr answered by the model (draft on, audits off — the Model-only path), per step and per
//! variance step (every `variance_cadence`-th consultation: `predict(…, true)` and rule 5d's
//! novelty), and the pieces. A measurement, reported in `crates/validate/references.md`
//! ("Status cache and the variance step"); nothing is asserted but that the steps are answered.
//!
//! `COLLECTION_DIR=<copy of a collection's surface dir> PSEUDO_DIR=<the pseudo disk cache> cargo test -p providers --release --lib step_cost_on_a_collection -- --ignored --nocapture`
use super::*;

#[test]
#[ignore = "a measurement on a collection copy (COLLECTION_DIR)"]
fn step_cost_on_a_collection() {
    // the pseudopotential gate's disk cache (the app's `cache\pseudo`, or a copy): without it the
    // first `All`-policy provenance generates the H pseudopotential on the CPU (tens of minutes)
    crate::pseudo::set_disk_cache(std::env::var_os("PSEUDO_DIR").map(std::path::PathBuf::from));
    let dir = collection_dir_copy();
    let key = surrogate::SurfaceKey::from_text(&std::fs::read_to_string(dir.join("key.txt")).unwrap()).unwrap();
    let (mut c, _) = surrogate::Collection::open(dir.parent().unwrap(), &key).unwrap();
    c.read_only = true;
    let mut s = SurrogateKs::new(Box::new(StubKsGpu), FallbackPolicy::default()).with_teacher_tol(1e-6);
    s.attach(c).unwrap();
    if s.report().is_none() {
        SurrogateKs::refit(&mut s).unwrap();
    }
    let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
    let mk = |r: f64| Scene {
        nuclei: vec![Nucleus::new(1, [-0.5 * r, 0.0, 0.0]), Nucleus::new(1, [0.5 * r, 0.0, 0.0])],
        cell: Cell { n: 129, h: 0.15 },
        charge: 0,
        multiplicity: None,
        sigma: Some(0.15),
        external_field: [0.0; 3],
        ionic: sim::IonicPolicy::All,
        train_mask: None,
    };
    {
        let r = s.report().unwrap();
        println!("samples {}, Q′ {}, null-space dimension {:?}, trusted {}", s.samples(), r.q_prime, s.null_space.as_ref().map(|n| n.dimension()), s.trusted());
    }
    Trainable::set_draft(&mut s, true);
    Trainable::set_audit_every(&mut s, 0);
    // warm-up (first-call caches: the representatives, the descriptor cache)
    for k in 0..8 {
        let _ = s.evaluate(&mk(1.4 + 0.01 * k as f64), &mut never());
    }
    let steps = 400;
    let (mut all, mut var, mut plain) = (Vec::new(), Vec::new(), Vec::new());
    let mut answered = 0;
    for k in 0..steps {
        let sc = mk(1.4 + 0.05 * (k as f64 * 0.3).sin());
        let before = s.variance_checks();
        let t = Instant::now();
        let res = s.evaluate(&sc, &mut never());
        let dt = ms(t);
        all.push(dt);
        if s.variance_checks() > before { var.push(dt) } else { plain.push(dt) }
        if res.as_ref().is_ok_and(|f| f.stats.source == Source::Surrogate) {
            answered += 1;
        }
    }
    assert!(answered > 0, "the model answered no step");
    let stats = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        let n = v.len().max(1);
        (v.iter().sum::<f64>() / n as f64, v.get(n / 2).copied().unwrap_or(f64::NAN), v.last().copied().unwrap_or(f64::NAN), v.len())
    };
    let (a, v, p) = (stats(&mut all), stats(&mut var), stats(&mut plain));
    println!("answered {answered}/{steps}");
    println!("evaluate, every step:   mean {:.3} ms, median {:.3} ms, max {:.3} ms ({} steps)", a.0, a.1, a.2, a.3);
    println!("evaluate, variance step: mean {:.3} ms, median {:.3} ms, max {:.3} ms ({} steps)", v.0, v.1, v.2, v.3);
    println!("evaluate, other steps:   mean {:.3} ms, median {:.3} ms, max {:.3} ms ({} steps)", p.0, p.1, p.2, p.3);
    // the variance step's pieces at 1.41 Bohr
    let sc = mk(1.41);
    let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
    let z = vec![1u8, 1];
    let model = s.model.clone().unwrap();
    let reps = 50;
    let time = |name: &str, f: &dyn Fn()| {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        println!("  {name:<22} {:.3} ms", ms(t) / reps as f64);
    };
    time("predict (no variance)", &|| {
        std::hint::black_box(model.predict(&pos, &z, false).unwrap());
    });
    time("predict (variance)", &|| {
        std::hint::black_box(model.predict(&pos, &z, true).unwrap());
    });
    time("novelty", &|| {
        std::hint::black_box(s.null_space.as_ref().map(|ns| ns.novelty(&pos, &z).unwrap()));
    });
    let pred = model.predict(&pos, &z, true).unwrap();
    time("variance_check", &|| {
        std::hint::black_box(s.variance_check(&model, &pred, pred.energy_std.unwrap(), pred.force_std_epistemic.as_ref().unwrap(), &pos, &z));
    });
    println!("  ν at 1.41 Bohr {:?}", s.null_space.as_ref().map(|ns| ns.novelty(&pos, &z).unwrap()));
}
