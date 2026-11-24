//! The refit speed-ups' equivalence golden (branch `refit-speed`, 2026-11-17): on a **copy** of a
//! shared collection (`COLLECTION_DIR`, opened read-only), the cold refit (attach) and the
//! append-20 refit (the collection less its last 20 samples attached, those 20 pushed, refitted)
//! are recorded — the hyperparameters, the model's predictions and variances at fixed geometries,
//! every sample's quarantine code, the class verdicts, the misfit set M, the leave-one-out ratios,
//! the active set and the null space — and a later build is compared against the record:
//! every verdict identical (codes, trust and recovery, M, the class and key verdicts, every
//! `would_answer` verdict and novelty refusal), and the predictions' absolute energy, force and std
//! differences within 1 % of the policy tiers (`energy_std_per_atom`, `force_std_abs`; the
//! controller's ruling 2026-11-17, replacing a 1e-9 relative bar no reordering of O(Q′³)
//! arithmetic meets at κ₂(A) ≈ 1e13). Relative differences and the other exact items are reported. `#[ignore]`d (minutes; needs the collection copy):
//!
//! `COLLECTION_DIR=<copy> REFIT_GOLDEN=record:<file>|compare:<file> [REFIT_GOLDEN_CASES=cold,append,recovery] cargo test -p providers --release --lib refit_golden -- --ignored --nocapture --test-threads 1`
//!
//! `recovery` is the misfit-driven case: the cold attach with the energy tier tightened to a third
//! (`FallbackPolicy::energy_std_per_atom`), so the training residuals fail and the trust recovery
//! runs its cap retries and quarantine rounds.
use super::*;
use surrogate::Collection;

fn open_copy() -> Collection {
    let dir = collection_dir_copy();
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
        // the case's fits, as solved (`fit_log`: a search's rounds, and whether it started from a
        // recovery trial's start)
        println!("case {case}: fits {:?}", s.fit_log);
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
    // per case, the absolute prediction differences (the controller's ruling 2026-11-17, review
    // I1): (|ΔE|/N Ha/atom, max |ΔF_component| Ha/Bohr, |Δσ_E|/N Ha/atom, max |Δσ_F| over the force
    // std and its epistemic part, Ha/Bohr), each with where it was
    let mut abs: std::collections::BTreeMap<&str, [(f64, String); 4]> = Default::default();
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
        if cat == "pred" && it.next().is_some_and(|k| k.starts_with("probe")) {
            // [E, σ_E, F (3N), σ_F (N), σ_F epistemic (N)]
            let n = (a.len() - 2) / 5;
            let d = |i: usize| (a[i] - b[i]).abs();
            let e = abs.entry(case).or_insert_with(|| std::array::from_fn(|_| (0.0, String::new())));
            let mut put = |slot: usize, v: f64| {
                if v > e[slot].0 {
                    e[slot] = (v, wk.to_string());
                }
            };
            put(0, d(0) / n as f64);
            put(1, (2..2 + 3 * n).map(d).fold(0.0, f64::max));
            put(2, d(1) / n as f64);
            put(3, (2 + 3 * n..2 + 5 * n).map(d).fold(0.0, f64::max));
        }
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
    // the tiers each case was judged by (`recovery`: the energy tier at a third, as `run_cases`)
    let p = FallbackPolicy::default();
    let tiers = |case: &str| (if case == "recovery" { p.energy_std_per_atom / 3.0 } else { p.energy_std_per_atom }, p.force_std_abs);
    let mut over_tier = Vec::new();
    println!("\n| case | max |ΔE|/N (Ha/atom) | ÷ energy tier | max |ΔF| (Ha/Bohr) | ÷ force tier | max |Δσ_E|/N | ÷ energy tier | max |Δσ_F| | ÷ force tier |\n|---|---|---|---|---|---|---|---|---|");
    for (case, e) in &abs {
        let (te, tf) = tiers(case);
        let frac = [e[0].0 / te, e[1].0 / tf, e[2].0 / te, e[3].0 / tf];
        println!("| {case} (tiers {te:.2e}, {tf:.2e}) | {:.2e} | {:.2e} | {:.2e} | {:.2e} | {:.2e} | {:.2e} | {:.2e} | {:.2e} |", e[0].0, frac[0], e[1].0, frac[1], e[2].0, frac[2], e[3].0, frac[3]);
        for (k, f) in frac.iter().enumerate() {
            if *f > 0.01 {
                over_tier.push(format!("{case}: quantity {k} at {:.2e} of its tier ({})", f, e[k].1));
            }
        }
        println!("  where: E {} · F {} · σ_E {} · σ_F {}", e[0].1, e[1].1, e[2].1, e[3].1);
    }
    // every verdict identical (the ruling): the codes, the trust verdict and the recovery, M, the
    // class and key verdicts and counts, the would_answer verdicts and the novelty refusals; the
    // other exact items (the search's round count, the active rows, the LOO row membership, the
    // worst-sample indices, the null-space dimension) are reported, not judged
    let verdict = |item: &str| {
        let key = item.split(['\n', ' ']).nth(2).unwrap_or("");
        ["trust", "recovery", "codes", "misfit", "class", "key", "keyq"].contains(&key) || key.starts_with("verdict") || key.starts_with("novelty-at")
    };
    let verdict_fail: Vec<&String> = exact_fail.iter().filter(|f| verdict(f)).collect();
    println!("\nexact items differing: {} ({} of them verdicts)", exact_fail.len(), verdict_fail.len());
    for f in &exact_fail {
        println!("{f}");
    }
    for o in &over_tier {
        println!("OVER 1 % OF A TIER: {o}");
    }
    assert!(verdict_fail.is_empty() && over_tier.is_empty(), "not equivalent: {} verdict items differ, {} predictions over 1 % of a tier", verdict_fail.len(), over_tier.len());
}

/// Where the active set's kept span ends against τ on the collection copy (`surrogate::grade`
/// module doc, "Eigenvalues within rounding of τ"): per species, τ, the band δ = √P·ε·max|λ|, the
/// kept counts of Jacobi (λ > τ) and of faer with the band rule (λ > τ + δ), the nearest
/// eigenvalues on either side of τ, and the largest Jacobi–faer eigenvalue gap. Reported.
#[test]
#[ignore = "measurement: COLLECTION_DIR=<copy> cargo test -p providers --release --lib active_span_margins -- --ignored --nocapture"]
fn active_span_margins() {
    let mut s = provider();
    attach(&mut s, open_copy());
    s.ensure_desc_cache();
    let model = s.model.clone().unwrap();
    let basis = s.desc_basis.clone().unwrap();
    let n = s.accumulated.min(s.desc_cache.len());
    let store = s.store.as_ref().unwrap();
    let atoms: Vec<(usize, Vec<f64>)> = (0..n)
        .filter(|&i| !s.excluded(i))
        .flat_map(|i| s.desc_cache[i].iter().enumerate().filter(move |(j, _)| store.samples[i].trains_atom(*j)).map(|(_, (sp, b))| (*sp, b.clone())))
        .collect();
    let spectra = surrogate::ActiveSet::species_spectra(&basis, &model.scale, atoms.iter().map(|(sp, b)| (*sp, &b[..])));
    println!("\n| species | P | τ | δ/τ | kept Jacobi | kept faer (λ > τ) | kept faer+band | nearest above τ (×τ) | nearest below τ (×τ) | max |λ_J − λ_f|/τ in (0.1τ, 10τ) |\n|---|---|---|---|---|---|---|---|---|---|");
    for (k, sp) in spectra.iter().enumerate() {
        let Some((ej, ef, tau)) = sp else { continue };
        let p = ej.len();
        let delta = (p as f64).sqrt() * f64::EPSILON * ef.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let kept_j = ej.iter().filter(|&&v| v > *tau).count();
        let kept_f = ef.iter().filter(|&&v| v > tau + delta).count();
        let kept_fb = ef.iter().filter(|&&v| v > *tau).count();
        // the eigenvalues nearest τ from each solver, in units of τ
        let near = |ev: &[f64]| { let mut x: Vec<f64> = ev.iter().map(|v| v / tau).filter(|r| (0.1..10.0).contains(r)).collect(); x.sort_by(f64::total_cmp); x.iter().map(|r| format!("{r:.3}")).collect::<Vec<_>>().join(" ") };
        println!("species {k}: Jacobi λ/τ in (0.1, 10): {}\n           faer   λ/τ in (0.1, 10): {}", near(ej), near(ef));
        let above = ef.iter().filter(|&&v| v > *tau).fold(f64::INFINITY, |m, &v| m.min(v)) / tau;
        let below = ef.iter().filter(|&&v| v <= *tau).fold(f64::NEG_INFINITY, |m, &v| m.max(v)) / tau;
        let gap = ej.iter().zip(ef).filter(|(a, _)| (0.1 * tau..10.0 * tau).contains(*a)).fold(0.0f64, |m, (a, b)| m.max((a - b).abs())) / tau;
        println!("| {k} | {p} | {tau:.3e} | {:.3} | {kept_j} | {kept_fb} | {kept_f} | {above:.3e} | {below:.3e} | {gap:.3e} |", delta / tau);
    }
}
