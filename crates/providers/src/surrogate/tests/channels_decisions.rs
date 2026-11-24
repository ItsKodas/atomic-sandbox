//! T7 of the chemical-channels plan (`docs/superpowers/plans/2026-11-22-chemical-channels.md`, the
//! Task 5–6 blocker; spec `docs/superpowers/specs/2026-11-22-chemical-channels-design.md`, "Physics
//! review 2" item 10's last test and I7; reviews `docs/superpowers/research/2026-11-23-…-task4-task5-review.md`
//! T7 and `2026-11-24-…-task56-review.md` I3): **the day-1 gate rerun through the full decision list**.
//! A measurement: it reports, it asserts nothing about the outcome.
//!
//! The day-1 gate (`channels_gate`) compared the models' design-row predictions. This one asks what
//! the app would do with each held-out geometry:
//!
//! 1. **The same split** as the day-1 gate (`channels_gate::{load, groups, split}`: 80/20 by scene
//!    group, blocks of 20, seed 2026), and the same rare-element thinning (the rarest element with
//!    ≥ 20 training samples and ≥ 20 held-out atoms, its training samples thinned to 5 % / 20 % with
//!    seed 2026 ^ 0x5eed).
//! 2. **A real provider per model.** A fresh read-only copy of the collection is opened, its basis set
//!    to the species basis or `BasisKind::Channels { k }` (the key's K: the provider learns its own W
//!    through `channel_lifecycle`, with the census's N_W mask as `MixSetup::eligible`, stages it, fits,
//!    commits), the held-out (and thinned-out) samples **removed from the collection** before the
//!    attach — the provider never sees them, as if they had never been collected — and the provider's
//!    own `refit` run, timed: for the channel model that is the W learn end to end (census, low-order
//!    fit, W, the channel fit and every trust pass), for the species model the cold fit. The audit
//!    history on file stands as the app would read it: the species fit keeps its evidence; the first
//!    W learn moves `history_from`, so the channel model starts with none (the app's lifecycle).
//! 3. **The decision list per held-out geometry**: `SurrogateKs::would_answer` at the sample's own box
//!    (rules 1–4 with 3b, the non-finite check, 5a–5c and 5e, 5d, the class rule 4b with the per-key
//!    C2 floors, rules 6/7 with the calibrated inflations, the fences), the first refusal's rule
//!    tallied per sample, per element and per environment key. An atom is **admitted** when its
//!    scene is answered.
//! 4. **Admitted-but-wrong**: an admitted atom whose model force misses the teacher's by more than 1×
//!    / 3× (`calibrate::WORST_ATOM`) the per-atom bound (`FallbackPolicy::force_residual`, the bound
//!    the audits use); an admitted energy row beyond the tier (`energy_std_per_atom`, 1e-3 Ha/atom)
//!    and 3× it.
//! 5. **Unrepresented many-body neighbours**: every admitted atom with two or more neighbours within
//!    r_cut and a neighbour species the committed mix does not represent, or whose W row in the
//!    standing model is zero (spec item 10: must be zero for the channel model).
//! 6. **Calibration**: z = ‖ΔF‖ / force_std (the posterior's total force std) and z_epi with the
//!    epistemic part alone, as rms over the held-out atoms; spec item 9 asks the channel model's z
//!    within 1 ± 1/√(2n) of the species model's.
//!
//! `COLLECTION_DIR=<copy> cargo test -p providers --release --lib channels_decisions -- --ignored --nocapture --test-threads 1`
//! `T7_MODELS` = comma list of `k3,species,k3_t5,species_t5,k3_t20,species_t20` (default all, in that
//! order; `k<K>` for another K).
use super::channels_gate::{groups, load, split, symbol, Data};
use super::refit_profile::{open_copy, provider};
use super::*;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write as _;
use std::time::Instant;
use surrogate::BasisKind;

fn flush() {
    let _ = std::io::stdout().flush();
}

/// One model of the run: its basis (`None`: species) and the rare element's thinning.
#[derive(Clone, Debug)]
struct Config {
    label: String,
    k: Option<usize>,
    thin: Option<f64>,
}

fn parse_config(s: &str) -> Config {
    let (base, thin) = match s.split_once("_t") {
        Some((b, t)) => (b, Some(t.parse::<f64>().expect("_t<percent>") / 100.0)),
        None => (s, None),
    };
    let k = if base == "species" { None } else { Some(base.strip_prefix('k').and_then(|x| x.parse().ok()).expect("species or k<K>")) };
    Config { label: s.to_string(), k, thin }
}

/// Atoms by outcome.
#[derive(Clone, Debug, Default)]
struct Tally {
    atoms: usize,
    answered: usize,
    /// admitted atoms beyond 1× / 3× the force bound
    wrong1: usize,
    wrong3: usize,
    /// the first refusal's rule → atoms
    refused: BTreeMap<String, usize>,
}

impl Tally {
    fn add(&mut self, verdict: &Result<(), String>, ratio: Option<f64>) {
        self.atoms += 1;
        match verdict {
            Ok(()) => {
                self.answered += 1;
                let r = ratio.unwrap_or(f64::INFINITY);
                self.wrong1 += usize::from(r > 1.0);
                self.wrong3 += usize::from(r > crate::surrogate::calibrate::WORST_ATOM);
            }
            Err(rule) => *self.refused.entry(rule.clone()).or_default() += 1,
        }
    }

    fn refused_words(&self) -> String {
        let mut v: Vec<(&String, &usize)> = self.refused.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        v.iter().map(|(r, n)| format!("{r} {n}")).collect::<Vec<_>>().join(", ")
    }
}

/// rms accumulator
#[derive(Clone, Copy, Debug, Default)]
struct Rms {
    sum2: f64,
    n: usize,
}

impl Rms {
    fn push(&mut self, x: f64) {
        if x.is_finite() {
            self.sum2 += x * x;
            self.n += 1;
        }
    }
    fn get(&self) -> f64 {
        if self.n == 0 { f64::NAN } else { (self.sum2 / self.n as f64).sqrt() }
    }
}

struct Outcome {
    cfg: Config,
    refit_s: f64,
    verdict_s: f64,
    q_prime: usize,
    trusted: bool,
    represented: String,
    samples: usize,
    answered_samples: usize,
    refused_samples: BTreeMap<String, usize>,
    e_rows: usize,
    e_answered: usize,
    e_wrong1: usize,
    e_wrong3: usize,
    all: Tally,
    by_z: BTreeMap<u8, Tally>,
    by_key: BTreeMap<String, Tally>,
    /// the standing verdict of each key met in the held-out set
    key_verdicts: BTreeMap<String, String>,
    /// admitted many-body atoms with an unrepresented neighbour (spec item 10: must be 0)
    unrep_admitted: usize,
    /// many-body atoms with an unrepresented neighbour, and of them refused by rule 3b
    unrep_atoms: usize,
    unrep_3b: usize,
    z_all: Rms,
    z_adm: Rms,
    zepi_all: Rms,
    zepi_adm: Rms,
    z_by_z: BTreeMap<u8, Rms>,
}

fn rule_name(r: &TeachRule) -> String {
    match r {
        TeachRule::UnrepresentedNeighbour(z) => format!("3b unrep {}", symbol(*z)),
        TeachRule::NewSpecies(z) => format!("2 new species {}", symbol(*z)),
        TeachRule::OtherSettings => "3 other settings".into(),
        TeachRule::NoModel => "1 no model".into(),
        TeachRule::Untrusted(k) => format!("4 untrusted {k:?}"),
        TeachRule::Class(c) => format!("4b class {c:?}"),
        TeachRule::IsolatedAtom => "5a isolated".into(),
        TeachRule::Coordination => "5b coordination".into(),
        TeachRule::Spread => "5c spread".into(),
        TeachRule::NewEnvironment => "5d new environment".into(),
        TeachRule::UnspannedAtom => "5e unspanned".into(),
        TeachRule::Grade => "5e grade".into(),
        TeachRule::NoGrade => "5e no grade".into(),
        TeachRule::ForceUncertainty => "6 force σ".into(),
        TeachRule::EnergyUncertainty => "7 energy σ".into(),
        TeachRule::NonFiniteAnswer => "non-finite".into(),
        TeachRule::NearQuarantine => "fence quarantine".into(),
        TeachRule::NearMisfit => "fence misfit".into(),
        other => format!("{other:?}"),
    }
}

fn key_name(k: &surrogate::EnvKey) -> String {
    let nb: Vec<&str> = k.neighbours.iter().map(|&t| symbol(t)).collect();
    format!("{} {}{{{}}}", if k.class == 1 { "pair" } else { "many" }, symbol(k.z), nb.join(","))
}

fn run(cfg: &Config, d: &Data, held: &[usize], drop: &BTreeSet<usize>) -> Outcome {
    println!("\n## {}: basis {}, {} samples removed from the collection ({} held out, {} thinned)", cfg.label, cfg.k.map_or("species".to_string(), |k| format!("channels K = {k}")), drop.len(), held.len(), drop.len() - held.len());
    flush();
    let mut c = open_copy();
    if let Some(k) = cfg.k {
        c.basis = BasisKind::Channels { k };
    }
    let n0 = c.samples.len();
    assert_eq!(c.quarantine.len(), n0, "a status per sample");
    let samples = std::mem::take(&mut c.samples);
    let codes = std::mem::take(&mut c.quarantine);
    for (i, (x, q)) in samples.into_iter().zip(codes).enumerate() {
        if !drop.contains(&i) {
            c.samples.push(x);
            c.quarantine.push(q);
        }
    }
    let mut s = provider();
    s.attach_refit_off = true;
    s.attach(c).unwrap();
    let t = Instant::now();
    SurrogateKs::refit(&mut s).unwrap();
    let refit_s = t.elapsed().as_secs_f64();
    for n in s.user_notes.drain(..) {
        println!("  note: {n}");
    }
    let r = s.report().unwrap().clone();
    let store_n = s.store.as_ref().map_or(0, |x| x.len());
    let fitted = (0..s.accumulated.min(store_n)).filter(|&i| !s.excluded(i)).count();
    let mut held_codes: BTreeMap<u8, usize> = BTreeMap::new();
    for i in 0..store_n {
        if s.excluded(i) {
            *held_codes.entry(s.quarantine[i]).or_default() += 1;
        }
    }
    println!(
        "  refit (end to end through the provider): {refit_s:.1} s; stored {store_n}, fitted {fitted}, held by code {held_codes:?}; Q′ {}, λ {:.3e}, σ_E {:.3e}, σ_F {:.3e}, train RMSE E {:.3e} F {:.3e}; trusted {} ({:?})",
        r.q_prime,
        r.lambda,
        r.sigma_e,
        r.sigma_f,
        r.rmse_energy_per_atom,
        r.rmse_force,
        s.trusted(),
        s.untrusted_reason()
    );
    let b = &s.model.as_ref().unwrap().basis;
    println!("  basis: species {:?}, n_max {} l_max {} degree_cap {} body4 {} D₄ {} r_cut {}, channels {:?}", b.species, b.n_max, b.l_max, b.degree_cap, b.body4, b.body4_degree, b.r_cut, b.channels().map(|m| m.k));
    let mut represented = "every species (species basis)".to_string();
    if let Some(sh) = s.shared.as_ref() {
        if let Some(cn) = sh.census.as_ref() {
            println!(
                "  census: {:?}",
                cn.species.iter().enumerate().map(|(t, &z)| format!("{} N_W {} many-body {} progress {:?}", symbol(z), cn.n_w[t], cn.many_body[t], cn.progress[t])).collect::<Vec<_>>()
            );
        }
        if let Some(m) = sh.mix.as_ref() {
            represented = crate::surrogate::represent::represented_words(m);
            println!("  committed mix: K {} (asked {}), {represented}; ρ {:?}; singular values {:?}; η {:.3}", m.k, m.k_requested, m.truncation_residual, m.singular_values, m.nonlinearity);
            for (t, &z) in m.species.iter().enumerate() {
                println!("    W[{}] = {:?}", symbol(z), (0..m.k).map(|c| format!("{:+.4}", m.w[t * m.k + c])).collect::<Vec<_>>());
            }
        } else if cfg.k.is_some() {
            represented = "no W committed".into();
            println!("  no mix committed");
        }
    }
    // the per-key verdicts that stand (the class rule's input, with the C2 floors)
    println!("  standing key verdicts:");
    for l in &s.class_loo {
        for v in &l.species {
            let k = surrogate::EnvKey { class: if l.class.starts_with("pair") { 1 } else { 2 }, z: v.z, neighbours: v.neighbours.clone() };
            println!(
                "    {} ({}): n_k {} q_k {}, trusted {}, refusal {:?}, evidence {}, λ {:.2}",
                key_name(&k),
                l.class,
                v.samples,
                v.columns,
                v.trusted,
                v.refusal,
                v.evidence,
                v.inflation
            );
        }
    }
    flush();
    let model = s.model.as_ref().unwrap();
    // the canonical references the held-out energies are rebased to (load's) must be this model's
    let mut dref = 0.0f64;
    for (t, &z) in model.basis.species.iter().enumerate() {
        if let Some(u) = d.species.iter().position(|&x| x == z) {
            dref = dref.max((model.ref_energy[t] - d.refs[u]).abs());
        }
    }
    println!("  max |ΔE_ref| between the model's references and the held-out samples' rebase: {dref:.3e} Ha");
    let rc2 = model.basis.r_cut * model.basis.r_cut;
    let zero_row = |t: u8| -> bool {
        model.basis.channels().is_some_and(|m| model.basis.species_index(t).is_none_or(|ti| m.w[ti * m.k..(ti + 1) * m.k].iter().all(|&x| x == 0.0)))
    };
    let pol = s.policy;
    let mut o = Outcome {
        cfg: cfg.clone(),
        refit_s,
        verdict_s: 0.0,
        q_prime: r.q_prime,
        trusted: s.trusted(),
        represented,
        samples: 0,
        answered_samples: 0,
        refused_samples: BTreeMap::new(),
        e_rows: 0,
        e_answered: 0,
        e_wrong1: 0,
        e_wrong3: 0,
        all: Tally::default(),
        by_z: BTreeMap::new(),
        by_key: BTreeMap::new(),
        key_verdicts: BTreeMap::new(),
        unrep_admitted: 0,
        unrep_atoms: 0,
        unrep_3b: 0,
        z_all: Rms::default(),
        z_adm: Rms::default(),
        zepi_all: Rms::default(),
        zepi_adm: Rms::default(),
        z_by_z: BTreeMap::new(),
    };
    let prov0 = s.store.as_ref().unwrap().provenance.clone();
    let t = Instant::now();
    let mut worst: Vec<(f64, String)> = Vec::new();
    for &i in held {
        let smp = &d.samples[i];
        let mut prov = prov0.clone();
        prov.grid_n = d.meta[i].3;
        let v = s.would_answer(&smp.pos, &smp.z, &prov);
        let verdict: Result<(), String> = v.map_err(|(rule, _)| rule_name(&rule));
        o.samples += 1;
        match &verdict {
            Ok(()) => o.answered_samples += 1,
            Err(r) => *o.refused_samples.entry(r.clone()).or_default() += 1,
        }
        let pred = model.predict(&smp.pos, &smp.z, true).ok();
        let keys = env_keys(&smp.pos, &smp.z, rc2);
        let nc = neighbour_counts(&smp.pos, rc2);
        let nbs = neighbour_species(&smp.pos, &smp.z, rc2);
        let formula = &d.meta[i].1;
        if smp.use_energy {
            o.e_rows += 1;
            if verdict.is_ok() {
                o.e_answered += 1;
                let p = pred.as_ref().expect("an answered scene has a prediction");
                let ne = surrogate::energy_row_atoms(&model.basis, &smp.pos).max(1) as f64;
                let err = (p.energy - smp.energy).abs() / ne;
                o.e_wrong1 += usize::from(err > pol.energy_std_per_atom);
                o.e_wrong3 += usize::from(err > 3.0 * pol.energy_std_per_atom);
                if err > pol.energy_std_per_atom {
                    worst.push((err / pol.energy_std_per_atom, format!("energy {formula} #{i}: {err:.3e} Ha/atom")));
                }
            }
        }
        for j in 0..smp.pos.len() {
            if !smp.trains_atom(j) {
                continue;
            }
            let (ratio, z_tot, z_epi) = match pred.as_ref() {
                Some(p) => {
                    let (dd, bound) = pol.force_residual(&p.forces[j], &smp.forces[j]);
                    let df: f64 = (0..3).map(|a| (p.forces[j][a] - smp.forces[j][a]).powi(2)).sum::<f64>().sqrt();
                    let zt = p.force_std.as_ref().map_or(f64::NAN, |f| df / f[j]);
                    let ze = p.force_std_epistemic.as_ref().map_or(f64::NAN, |f| df / f[j]);
                    (Some(dd / bound), zt, ze)
                }
                None => (None, f64::NAN, f64::NAN),
            };
            o.z_all.push(z_tot);
            o.zepi_all.push(z_epi);
            o.z_by_z.entry(smp.z[j]).or_default().push(z_tot);
            if verdict.is_ok() {
                o.z_adm.push(z_tot);
                o.zepi_adm.push(z_epi);
                if let Some(r) = ratio.filter(|&r| r > 1.0) {
                    worst.push((r, format!("force {} in {formula} #{i} (key {})", symbol(smp.z[j]), keys[j].as_ref().map_or("isolated".into(), key_name))));
                }
            }
            o.all.add(&verdict, ratio);
            o.by_z.entry(smp.z[j]).or_default().add(&verdict, ratio);
            let kn = keys[j].as_ref().map_or("isolated".to_string(), key_name);
            o.by_key.entry(kn.clone()).or_default().add(&verdict, ratio);
            if let Some(k) = keys[j].as_ref() {
                o.key_verdicts.entry(kn).or_insert_with(|| match s.class_untrusted(&[Some(k.clone())]) {
                    None => "trusted".into(),
                    Some((rf, short)) => format!("refused{}: {}", if short { " (short of samples)" } else { "" }, rf.text.trim_start_matches("surrogate: ")),
                });
            }
            // spec item 10: a many-body atom with a neighbour the mix does not represent
            if nc[j] >= 2 && nbs[j].1.iter().any(|&t| !s.represents(t) || zero_row(t)) {
                o.unrep_atoms += 1;
                if verdict.is_ok() {
                    o.unrep_admitted += 1;
                    println!("  !! admitted many-body atom with an unrepresented neighbour: {} in {formula} #{i}", symbol(smp.z[j]));
                }
                if verdict.as_ref().err().is_some_and(|r| r.starts_with("3b")) {
                    o.unrep_3b += 1;
                }
            }
        }
    }
    o.verdict_s = t.elapsed().as_secs_f64();
    worst.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!("  verdicts: {:.1} s for {} held-out samples", o.verdict_s, o.samples);
    print_outcome(&o);
    println!("  admitted-but-wrong (worst first, err/bound): {:?}", worst.iter().take(12).map(|(r, w)| format!("{r:.2} {w}")).collect::<Vec<_>>());
    flush();
    o
}

fn pct(a: usize, b: usize) -> String {
    format!("{:.1} %", 100.0 * a as f64 / b.max(1) as f64)
}

fn print_outcome(o: &Outcome) {
    println!(
        "  [{}] answered {} / {} samples ({}), {} / {} atoms ({}); admitted atoms > 1× bound {}, > 3× {}; energy rows answered {} / {}, > tier {}, > 3× tier {}; admitted many-body atoms with an unrepresented neighbour {} (of {} such atoms, {} refused by 3b)",
        o.cfg.label,
        o.answered_samples,
        o.samples,
        pct(o.answered_samples, o.samples),
        o.all.answered,
        o.all.atoms,
        pct(o.all.answered, o.all.atoms),
        o.all.wrong1,
        o.all.wrong3,
        o.e_answered,
        o.e_rows,
        o.e_wrong1,
        o.e_wrong3,
        o.unrep_admitted,
        o.unrep_atoms,
        o.unrep_3b
    );
    println!("    refused samples by the first rule: {:?}", o.refused_samples);
    println!("    z (‖ΔF‖/force_std) rms: all {:.3} (n {}), admitted {:.3} (n {}); z_epi rms all {:.3}, admitted {:.3}", o.z_all.get(), o.z_all.n, o.z_adm.get(), o.z_adm.n, o.zepi_all.get(), o.zepi_adm.get());
    for (z, t) in &o.by_z {
        println!(
            "    {:>2}: {} atoms, answered {} ({}), > 1× {}, > 3× {}, z rms {:.3}; refused: {}",
            symbol(*z),
            t.atoms,
            t.answered,
            pct(t.answered, t.atoms),
            t.wrong1,
            t.wrong3,
            o.z_by_z.get(z).map_or(f64::NAN, |r| r.get()),
            t.refused_words()
        );
    }
    for (k, t) in &o.by_key {
        println!(
            "    {k:>16}: {} atoms, answered {} ({}), > 1× {}, > 3× {}; refused: {}; standing: {}",
            t.atoms,
            t.answered,
            pct(t.answered, t.atoms),
            t.wrong1,
            t.wrong3,
            t.refused_words(),
            o.key_verdicts.get(k).map_or("-", |s| s.as_str())
        );
    }
}

#[test]
#[ignore = "measurement: COLLECTION_DIR=<copy> cargo test -p providers --release --lib channels_decisions -- --ignored --nocapture --test-threads 1"]
fn channels_decisions() {
    let seed: u64 = 2026;
    let d = load();
    let g = groups(&d, 20);
    let held_mask = split(&d, &g, seed);
    let train_idx: Vec<usize> = (0..d.samples.len()).filter(|&i| !held_mask[i]).collect();
    let held_idx: Vec<usize> = (0..d.samples.len()).filter(|&i| held_mask[i]).collect();
    println!("worker threads {}; split: train {} / held out {} (the day-1 gate's)", surrogate::threads::worker_threads(), train_idx.len(), held_idx.len());
    // the rare element (the day-1 gate's rule)
    let mut count: BTreeMap<u8, usize> = BTreeMap::new();
    for &i in &train_idx {
        let mut zs = d.samples[i].z.clone();
        zs.sort_unstable();
        zs.dedup();
        for z in zs {
            *count.entry(z).or_default() += 1;
        }
    }
    let mut held_atoms: BTreeMap<u8, usize> = BTreeMap::new();
    for &i in &held_idx {
        for &z in &d.samples[i].z {
            *held_atoms.entry(z).or_default() += 1;
        }
    }
    let rare = count.iter().filter(|(z, &n)| n >= 20 && held_atoms.get(z).copied().unwrap_or(0) >= 20).min_by_key(|(_, &n)| n).map(|(&z, _)| z).unwrap();
    println!("training samples per element {count:?}; held-out atoms {held_atoms:?}; the probe element {}", symbol(rare));
    let held_store: BTreeSet<usize> = held_idx.iter().map(|&i| d.store_idx[i]).collect();
    let thinned = |frac: f64| -> BTreeSet<usize> {
        let with: Vec<usize> = train_idx.iter().copied().filter(|&i| d.samples[i].z.contains(&rare)).collect();
        let mut order = with.clone();
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5eed);
        for k in (1..order.len()).rev() {
            let j = rng.gen_range(0..=k);
            order.swap(k, j);
        }
        let keep_n = ((frac * with.len() as f64).ceil() as usize).max(1);
        let keep: BTreeSet<usize> = order[..keep_n].iter().copied().collect();
        println!("  {} thinned to {keep_n} of {} training samples ({:.0} %)", symbol(rare), with.len(), 100.0 * frac);
        with.iter().filter(|i| !keep.contains(i)).map(|&i| d.store_idx[i]).collect()
    };
    let models = std::env::var("T7_MODELS").unwrap_or_else(|_| "k3,species,k3_t5,species_t5,k3_t20,species_t20".into());
    let mut out: Vec<Outcome> = Vec::new();
    for m in models.split(',').map(str::trim).filter(|m| !m.is_empty()) {
        let cfg = parse_config(m);
        let mut drop = held_store.clone();
        if let Some(f) = cfg.thin {
            drop.extend(thinned(f));
        }
        out.push(run(&cfg, &d, &held_idx, &drop));
        print_tables(&out, rare);
    }
}

fn print_tables(out: &[Outcome], rare: u8) {
    println!("\n## T7 tables so far ({} models; force bound = the audits' per-atom bound, energy tier 1e-3 Ha/atom)\n", out.len());
    println!("| model | refit wall s | verdicts s | Q′ | trusted | represented | answered samples | answered atoms | admitted > 1× | admitted > 3× | E answered | E > tier | E > 3× tier | unrep. many-body admitted | z rms all / admitted | z_epi rms all / admitted |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for o in out {
        println!(
            "| {} | {:.0} | {:.0} | {} | {} | {} | {} / {} | {} / {} | {} | {} | {} / {} | {} | {} | {} | {:.3} / {:.3} | {:.2} / {:.2} |",
            o.cfg.label,
            o.refit_s,
            o.verdict_s,
            o.q_prime,
            o.trusted,
            o.represented,
            o.answered_samples,
            o.samples,
            o.all.answered,
            o.all.atoms,
            o.all.wrong1,
            o.all.wrong3,
            o.e_answered,
            o.e_rows,
            o.e_wrong1,
            o.e_wrong3,
            o.unrep_admitted,
            o.z_all.get(),
            o.z_adm.get(),
            o.zepi_all.get(),
            o.zepi_adm.get()
        );
    }
    println!("\n| model | refused samples by the first rule |\n|---|---|");
    for o in out {
        let mut v: Vec<(&String, &usize)> = o.refused_samples.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        println!("| {} | {} |", o.cfg.label, v.iter().map(|(r, n)| format!("{r} {n}")).collect::<Vec<_>>().join(", "));
    }
    println!("\n| model | element | atoms | answered | > 1× | > 3× | refused (first rule, atoms) |\n|---|---|---|---|---|---|---|");
    for o in out {
        for (z, t) in &o.by_z {
            println!("| {} | {} | {} | {} | {} | {} | {} |", o.cfg.label, symbol(*z), t.atoms, t.answered, t.wrong1, t.wrong3, t.refused_words());
        }
    }
    println!("\n| model | key | atoms | answered | > 1× | > 3× | refused (first rule, atoms) | standing verdict |\n|---|---|---|---|---|---|---|---|");
    for o in out {
        for (k, t) in &o.by_key {
            println!("| {} | {k} | {} | {} | {} | {} | {} | {} |", o.cfg.label, t.atoms, t.answered, t.wrong1, t.wrong3, t.refused_words(), o.key_verdicts.get(k).map_or("-", |s| s.as_str()));
        }
    }
    // the gate: each channel model against the species model of the same training set
    println!("\n## T7 verdicts (admitted-but-wrong = admitted atoms beyond WORST_ATOM = 3× the bound; the rare element {})", symbol(rare));
    for c in out.iter().filter(|o| o.cfg.k.is_some()) {
        let Some(sp) = out.iter().find(|o| o.cfg.k.is_none() && o.cfg.thin == c.cfg.thin) else { continue };
        let tz = |o: &Outcome, z: u8| o.by_z.get(&z).cloned().unwrap_or_default();
        let (cr, sr) = (tz(c, rare), tz(sp, rare));
        let ok_all = c.all.wrong3 <= sp.all.wrong3;
        let ok_rare = cr.wrong3 <= sr.wrong3;
        let ok_unrep = c.unrep_admitted == 0;
        let n = c.z_all.n.min(sp.z_all.n).max(1) as f64;
        let zr = c.z_all.get() / sp.z_all.get();
        let zb = 1.0 / (2.0 * n).sqrt();
        println!(
            "  {} vs {}: > 3× overall {} vs {} ({}); {} > 3× {} vs {} ({}); > 1× overall {} vs {} ({}), {} > 1× {} vs {}; energy > tier {} vs {}; unrepresented many-body admitted {} ({}); z ratio {zr:.3} (1 ± {zb:.3}: {}) → {}",
            c.cfg.label,
            sp.cfg.label,
            c.all.wrong3,
            sp.all.wrong3,
            if ok_all { "≤" } else { ">" },
            symbol(rare),
            cr.wrong3,
            sr.wrong3,
            if ok_rare { "≤" } else { ">" },
            c.all.wrong1,
            sp.all.wrong1,
            if c.all.wrong1 <= sp.all.wrong1 { "≤" } else { ">" },
            symbol(rare),
            cr.wrong1,
            sr.wrong1,
            c.e_wrong1,
            sp.e_wrong1,
            c.unrep_admitted,
            if ok_unrep { "= 0" } else { "≠ 0" },
            if (zr - 1.0).abs() <= zb { "within" } else { "outside" },
            if ok_all && ok_rare && ok_unrep { "PASS" } else { "FAIL" }
        );
    }
    flush();
}
