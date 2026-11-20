//! What a policy change costs on a collection (quiet refits round 2): ⚠ Model only toggles two
//! policy fields (`Trainable::set_draft`, `set_audit_every`), each a `set_policy` on the worker
//! thread — the training residuals and the class-wise leave-one-out re-judged against the new
//! thresholds. A measurement; nothing is asserted.
//!
//! `COLLECTION_DIR=<copy of a collection's surface dir> PSEUDO_DIR=<the pseudo disk cache> cargo test -p providers --release --lib policy_cost_on_a_collection -- --ignored --nocapture`
use super::*;

#[test]
#[ignore = "a measurement on a collection copy (COLLECTION_DIR)"]
fn policy_cost_on_a_collection() {
    crate::pseudo::set_disk_cache(std::env::var_os("PSEUDO_DIR").map(std::path::PathBuf::from));
    let dir = std::path::PathBuf::from(std::env::var("COLLECTION_DIR").expect("COLLECTION_DIR"));
    let key = surrogate::SurfaceKey::from_text(&std::fs::read_to_string(dir.join("key.txt")).unwrap()).unwrap();
    let (mut c, _) = surrogate::Collection::open(dir.parent().unwrap(), &key).unwrap();
    c.read_only = true;
    let mut s = SurrogateKs::new(Box::new(StubKsGpu), FallbackPolicy::default()).with_teacher_tol(1e-6);
    s.attach(c).unwrap();
    if s.report().is_none() {
        SurrogateKs::refit(&mut s).unwrap();
    }
    println!("samples {}, Q′ {}, trusted {}", s.samples(), s.report().unwrap().q_prime, s.trusted());
    let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
    for round in 0..3 {
        // Model only on: draft on, audits off — two policy changes
        let t = Instant::now();
        Trainable::set_draft(&mut s, true);
        let a = ms(t);
        let t = Instant::now();
        Trainable::set_audit_every(&mut s, 0);
        let b = ms(t);
        // and off again
        let t = Instant::now();
        Trainable::set_draft(&mut s, false);
        let c = ms(t);
        let t = Instant::now();
        Trainable::set_audit_every(&mut s, 500);
        let d = ms(t);
        // the pieces of one set_policy
        let t = Instant::now();
        let _ = s.compute_residuals();
        let res = ms(t);
        let t = Instant::now();
        s.refresh_class_loo();
        let loo = ms(t);
        println!("round {round}: Model only on {a:.0} + {b:.0} ms, off {c:.0} + {d:.0} ms; compute_residuals {res:.0} ms, refresh_class_loo {loo:.0} ms");
    }
}
