//! The refit speed-ups' equivalence golden (branch `refit-speed`, 2026-11-17): on a **copy** of a
//! shared collection (`COLLECTION_DIR`, opened read-only), the cold refit (attach) and the
//! append-20 refit (the collection less its last 20 samples attached, those 20 pushed, refitted)
//! are recorded — the hyperparameters, the model's predictions and variances at fixed geometries,
//! every sample's quarantine code, the class verdicts, the misfit set M, the leave-one-out ratios,
//! the active set and the null space — and a later build is compared against the record:
//! hyperparameters and predictions within 1e-9 relative, the LOO numbers within 1e-8, codes,
//! verdicts and M identical. `#[ignore]`d (minutes; needs the collection copy):
//!
//! `COLLECTION_DIR=<copy> REFIT_GOLDEN=record:<file>|compare:<file> [REFIT_GOLDEN_CASES=cold,append,recovery] cargo test -p providers --release --lib refit_golden -- --ignored --nocapture --test-threads 1`
//!
//! `recovery` is the misfit-driven case: the cold attach with the energy tier tightened to a third
//! (`FallbackPolicy::energy_std_per_atom`), so the training residuals fail and the trust recovery
//! runs its cap retries and quarantine rounds.
use super::*;
use surrogate::Collection;

fn open_copy() -> Collection {
    let dir = std::path::PathBuf::from(std::env::var("COLLECTION_DIR").expect("COLLECTION_DIR names a collection folder copy"));
    let key = surrogate::SurfaceKey::from_text(&std::fs::read_to_string(dir.join("key.txt")).unwrap()).unwrap();
    let (mut c, _) = surrogate::Collection::open(dir.parent().unwrap(), &key).unwrap();
    c.read_only = true;
    c
}

fn provider() -> SurrogateKs {
    SurrogateKs::new(Box::new(StubKsGpu), FallbackPolicy::default()).with_teacher_tol(1e-6)
}

fn attach(s: &mut SurrogateKs, c: Collection) {
    s.attach(c).unwrap();
    if s.report().is_none() {
        SurrogateKs::refit(s).unwrap();
    }
}

/// The fixed probe geometries: every 8th sample of the full collection copy, as stored and
/// displaced by a deterministic ±0.04 Bohr per coordinate (xorshift).
fn probes() -> Vec<(Vec<[f64; 3]>, Vec<u8>)> {
    let c = open_copy();
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let mut out = Vec::new();
    for cs in c.samples.iter().step_by(8) {
        let s = &cs.sample;
        out.push((s.pos.clone(), s.z.clone()));
        out.push((s.pos.iter().map(|p| [p[0] + 0.08 * rnd(), p[1] + 0.08 * rnd(), p[2] + 0.08 * rnd()]).collect(), s.z.clone()));
    }
    out
}

fn nums(v: &[f64]) -> String {
    v.iter().map(|x| format!("{x:?}")).collect::<Vec<_>>().join(" ")
}

/// The record of one refit's outcome, one item per line: `<category> <key> : <values>`.
fn record(s: &mut SurrogateKs, probes: &[(Vec<[f64; 3]>, Vec<u8>)]) -> Vec<String> {
    let mut o = Vec::new();
    let r = s.report().unwrap().clone();
    o.push(format!("hyper report : {}", nums(&[r.lambda, r.sigma_e, r.sigma_f, r.log_evidence, r.gamma_e, r.gamma_f, r.residual_dof_e, r.residual_dof_f, r.rmse_energy_per_atom, r.rmse_force])));
    o.push(format!("exact shape : q {} n {} rows {} iters {} conv {}", r.q_prime, r.n_samples, r.n_rows, r.hyper_iterations, r.converged));
    o.push(format!("exact trust : {} {:?}", s.trusted(), s.untrusted_reason()));
    o.push(format!("exact recovery : {:?}", s.recovery_note()));
    o.push(format!("exact codes : {:?}", s.quarantine));
    let m = s.misfit().map(|m| (m.samples.clone(), m.atoms.clone()));
    o.push(format!("exact misfit : {m:?}"));
    for c in &s.class_loo {
        o.push(format!("exact class {} : samples {} trusted {} refusal-key {:?}", c.class, c.samples, c.trusted, c.refusal.as_ref().map(|r| r.0)));
        o.push(format!("loo class {} : {}", c.class, nums(&[c.force_rms, c.force_max, c.energy_rms, c.energy_max])));
        if let Some(rf) = &c.refusal {
            o.push(format!("loo class-refusal {} : {}", c.class, nums(&[rf.1, rf.2])));
        }
    }
    if let Some(j) = s.class_judge.as_ref() {
        for (k, st) in &j.key_stats {
            o.push(format!("exact key {k:?} : samples {} misfits {}", st.samples, st.energy_misfits));
            o.push(format!("loo key {k:?} : {}", nums(&[st.force_rms, st.force_max, st.energy_rms, st.energy_max])));
        }
        o.push(format!("exact keyq : {:?} fit_columns {}", j.key_q, j.fit_columns));
    }
    o.push(format!("exact class_worst-idx : {:?}", s.class_worst.iter().map(|w| w.map(|x| x.0)).collect::<Vec<_>>()));
    o.push(format!("loo class_worst : {}", nums(&s.class_worst.iter().map(|w| w.map_or(f64::NAN, |x| x.1)).collect::<Vec<_>>())));
    let lr = s.ensure_loo_ratios().cloned().expect("loo ratios");
    o.push(format!("exact loo-rows : {:?} channels {:?}", lr.rows.iter().map(|r| r.0).collect::<Vec<_>>(), lr.tier_channel));
    o.push(format!("loo ratios : {}", nums(&lr.rows.iter().map(|r| r.1).collect::<Vec<_>>())));
    o.push(format!("loo tier : {}", nums(&lr.tier)));
    if let Some(a) = s.active_set.as_ref() {
        for (k, sp) in a.species.iter().enumerate() {
            match sp {
                Some(sp) => o.push(format!("exact active {k} : r {} converged {} active {:?}", sp.u.len(), sp.converged, sp.active)),
                None => o.push(format!("exact active {k} : none")),
            }
        }
    }
    if let Some(ns) = s.null_space.as_ref() {
        o.push(format!("exact null : dim {}", ns.dimension()));
        o.push(format!("pred null-tau : {:?}", ns.tau));
    }
    let model = s.model.as_ref().unwrap().clone();
    for (i, (pos, z)) in probes.iter().enumerate() {
        let p = model.predict(pos, z, true).unwrap();
        let mut v = vec![p.energy, p.energy_std.unwrap()];
        v.extend(p.forces.iter().flatten());
        v.extend(p.force_std.unwrap());
        v.extend(p.force_std_epistemic.unwrap());
        o.push(format!("pred probe{i} : {}", nums(&v)));
        o.push(format!("exact verdict{i} : {:?}", s.would_answer(pos, z, &model.provenance)));
        let g = s.extrapolation_grade(pos, z).unwrap_or(f64::NAN);
        o.push(format!("grade probe{i} : {}", nums(&[g])));
        let nu = s.novelty(pos, z);
        o.push(format!("exact novelty-at{i} : {:?} over1 {:?}", nu.map(|x| x.1), nu.map(|x| x.0 > 1.0)));
        o.push(format!("novelty probe{i} : {}", nums(&[nu.map_or(f64::NAN, |x| x.0)])));
    }
    o
}

fn run_cases(cases: &str) -> Vec<String> {
    let probes = probes();
    let mut out = Vec::new();
    for case in cases.split(',').map(str::trim) {
        let t = std::time::Instant::now();
        let mut s = provider();
        match case {
            "cold" => attach(&mut s, open_copy()),
            "append" => {
                let mut c = open_copy();
                let n = c.samples.len() - 20;
                let tail: Vec<_> = c.samples.split_off(n);
                c.quarantine.truncate(n);
                attach(&mut s, c);
                let prov0 = s.shared.as_ref().unwrap().prov.clone();
                for cs in tail {
                    let prov = Provenance { grid_n: cs.grid_n, ..prov0.clone() };
                    s.push_shared(cs.sample, &prov, f64::NAN, cs.multiplicity).unwrap();
                }
                s.pending_refit = true;
                SurrogateKs::refit(&mut s).unwrap();
            }
            "recovery" => {
                let mut pol = FallbackPolicy::default();
                pol.energy_std_per_atom /= 3.0;
                s = SurrogateKs::new(Box::new(StubKsGpu), pol).with_teacher_tol(1e-6);
                attach(&mut s, open_copy());
            }
            other => panic!("unknown case {other}"),
        }
        println!("case {case}: {:.1} s; κ₂(A) {:.2e}; recovery {:?}; last solve {:?}", t.elapsed().as_secs_f64(), s.report().map_or(f64::NAN, |r| r.condition_estimate), s.recovery_note(), s.accumulator.as_ref().map(|a| a.last_refit()));
        out.extend(record(&mut s, &probes).into_iter().map(|l| format!("{case} {l}")));
    }
    out
}

/// Relative difference, NaN = NaN and ±∞ = ±∞ count as equal.
fn rel(a: f64, b: f64) -> f64 {
    if a == b || (a.is_nan() && b.is_nan()) {
        return 0.0;
    }
    (a - b).abs() / a.abs().max(b.abs()).max(f64::MIN_POSITIVE)
}

#[test]
#[ignore = "equivalence golden: COLLECTION_DIR=<copy> REFIT_GOLDEN=record:<file>|compare:<file> cargo test -p providers --release --lib refit_golden -- --ignored --nocapture --test-threads 1"]
fn refit_golden() {
    let mode = std::env::var("REFIT_GOLDEN").expect("REFIT_GOLDEN=record:<file> or compare:<file>");
    let cases = std::env::var("REFIT_GOLDEN_CASES").unwrap_or_else(|_| "cold,append".into());
    let (what, file) = mode.split_once(':').expect("record:<file> or compare:<file>");
    let got = run_cases(&cases);
    if what == "record" {
        std::fs::write(file, got.join("\n") + "\n").unwrap();
        println!("recorded {} lines to {file}", got.len());
        return;
    }
    assert_eq!(what, "compare");
    let want: Vec<String> = std::fs::read_to_string(file).unwrap().lines().map(str::to_string).collect();
    // per (case, category): the worst relative difference, and the exact lines that differ
    let mut worst: std::collections::BTreeMap<String, (f64, String)> = Default::default();
    let mut exact_fail: Vec<String> = Vec::new();
    let want_cases: std::collections::BTreeSet<&str> = cases.split(',').map(str::trim).collect();
    let want: Vec<&String> = want.iter().filter(|l| want_cases.contains(l.split(' ').next().unwrap())).collect();
    assert_eq!(want.len(), got.len(), "the records have different line counts");
    for (w, g) in want.iter().zip(&got) {
        let (wk, wv) = w.split_once(" : ").unwrap();
        let (gk, gv) = g.split_once(" : ").unwrap();
        assert_eq!(wk, gk, "the records' keys differ");
        let mut it = wk.split(' ');
        let (case, cat) = (it.next().unwrap(), it.next().unwrap());
        if cat == "exact" {
            if wv != gv {
                exact_fail.push(format!("{wk}\n  want {wv}\n  got  {gv}"));
            }
            continue;
        }
        let (a, b): (Vec<f64>, Vec<f64>) = (wv.split(' ').map(|x| x.parse().unwrap()).collect(), gv.split(' ').map(|x| x.parse().unwrap()).collect());
        assert_eq!(a.len(), b.len(), "{wk}: value counts differ");
        let label = if cat == "hyper" { wk.to_string() } else { format!("{case} {cat}") };
        for (x, y) in a.iter().zip(&b) {
            let d = rel(*x, *y);
            let e = worst.entry(label.clone()).or_insert((0.0, String::new()));
            if d > e.0 {
                *e = (d, format!("{wk}: {x:?} vs {y:?}"));
            }
        }
    }
    println!("\n| case category | worst relative difference | where |\n|---|---|---|");
    for (k, (d, at)) in &worst {
        println!("| {k} | {d:.2e} | {at} |");
    }
    println!("\nexact items differing: {}", exact_fail.len());
    for f in &exact_fail {
        println!("{f}");
    }
    let tol = |label: &str| if label.contains(" loo") || label.contains(" grade") || label.contains(" novelty") { 1e-8 } else { 1e-9 };
    let over: Vec<_> = worst.iter().filter(|(k, (d, _))| *d > tol(k)).collect();
    assert!(exact_fail.is_empty() && over.is_empty(), "not equivalent: {} exact items, {} categories over tolerance", exact_fail.len(), over.len());
}
