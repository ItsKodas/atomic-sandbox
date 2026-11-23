//! The chemical-channels day-1 go / no-go gate (plan `docs/superpowers/plans/2026-11-22-chemical-channels.md`
//! Task 2; spec `docs/superpowers/specs/2026-11-22-chemical-channels-design.md` "Gates" and review I7). A
//! measurement: it reports, it asserts only its own self-checks (the row predictions against the fit).
//!
//! Reads a **copy** of a collection folder (`COLLECTION_DIR`, opened read-only as `refit_profile` does) and
//! attaches it without a fit, so the fitted set is exactly the provider's: the canonical (per-box rebased)
//! energies, the quarantine, lone and stray hold-outs. Then:
//!
//! 1. **The split (I7a): held out by scene group.** A group is a maximal run of consecutive stored samples with
//!    the same scene name, formula, origin tag and box (one burst, one scan, one stretch of fallbacks), cut into
//!    blocks of at most `GATE_BLOCK` (20) samples so a long burst does not decide a formula's split on its own.
//!    Consecutive samples of one run are near-duplicates (an MD burst moves a few mBohr per step); a random
//!    per-sample split would put each held-out geometry's twin in the training set. Per formula, the groups are
//!    shuffled (seed `GATE_SEED`, 2026) and a group is taken into the held-out set when that brings the
//!    formula's held-out count closer to 20 % of its samples; a formula with fewer than two groups stays in
//!    training (it cannot be held out and still be represented in training).
//! 2. **The models.** The species model on the collection's basis (degree cap 6, D₄ = 3, the collection's
//!    reach; the user's standing fit) and channel models at K = 3 and 4 with W learnt on the training samples
//!    only (`channels::learn_mix`, review I2/I4 recipe, energy weighting on). **The K = S control is the species
//!    model itself:** `learn_mix` returns W = I when K_eff = S (review I2f), and W = I reproduces the species
//!    features (Task 1's identity test; `GATE=tiny` checks the fit end to end on a small basis).
//! 3. **Predictions from the design rows**, never through `model::energy_and_forces` (whose body-3/4 adjoint is
//!    not channel-aware until Task 4): the standardised rows the fit itself uses (`loo::design_rows`) dotted with
//!    the fitted coefficients, and σ_epi² = ‖L⁻¹φ‖² from the fit's posterior factor (no Σ_W term: Task 5).
//! 4. **Rare-element probe (I7f):** K = 4 with the rarest element that has ≥ 20 training samples and ≥ 20
//!    held-out atoms (enough to measure) thinned to 20 % (and 5 %) of its training samples, W relearnt on the
//!    thinned set.
//!
//! `COLLECTION_DIR=<copy> cargo test -p providers --release --lib channels_gate -- --ignored --nocapture --test-threads 1`
//! `GATE` = `full` (default) | `tiny` (the self-check on a small basis) | `dump` (the groups only);
//! `GATE_MODELS` = comma list of `species,k3,k4,thin20,thin5` (default all).
use super::refit_profile::{open_copy, provider};
use super::*;
use std::collections::BTreeMap;
use std::io::Write as _;
use std::time::Instant;
use surrogate::channels::{learn_mix, LearntMix, MixSetup};
use surrogate::fit::{Accumulator, FitOptions, FitReport, Sample};
use surrogate::loo::{design_rows, forward_many};
use surrogate::{Basis, ChannelMix, Model, Provenance};

/// The provider's fitted set and what a fit of it needs.
struct Data {
    samples: Vec<Sample>,
    /// (scene, formula, origin tag, box) per sample, in collection order.
    meta: Vec<(String, String, String, usize)>,
    species: Vec<u8>,
    refs: Vec<f64>,
    opts: FitOptions,
    prov: Provenance,
    r_cut: f64,
}

fn load() -> Data {
    let mut s = provider();
    s.attach_refit_off = true;
    s.attach(open_copy()).unwrap();
    s.exclude_box_samples();
    let store = s.store.as_ref().unwrap();
    let coll = s.collection().unwrap();
    let mut species: Vec<u8> = store.samples.iter().flat_map(|x| x.z.iter().copied()).collect();
    species.sort_unstable();
    species.dedup();
    let basis = s.default_basis(&species);
    let refs = s.reference_vector(&basis).unwrap();
    let prov = store.provenance.clone();
    let opts = FitOptions { sigma_f_min: crate::surrogate::policy::force_noise_prior(&prov.ionic), ..FitOptions::default() };
    let mut samples = Vec::new();
    let mut meta = Vec::new();
    let mut held = BTreeMap::<u8, usize>::new();
    for i in 0..store.len() {
        if s.excluded(i) {
            *held.entry(s.quarantine[i]).or_default() += 1;
            continue;
        }
        let cs = &coll.samples[i];
        samples.push(store.samples[i].clone());
        meta.push((cs.scene.clone(), crate::surrogate::shared::formula(&cs.sample.z), format!("{:?}", cs.sample.tag), cs.grid_n));
    }
    println!(
        "collection: {} stored, {} fitted, held by code {held:?}; species {species:?}; refs {refs:?}; reach {} Bohr; sigma_f_min {}",
        store.len(),
        samples.len(),
        basis.r_cut,
        opts.sigma_f_min
    );
    Data { samples, meta, species, refs, opts, prov, r_cut: basis.r_cut }
}

/// The groups (module doc, step 1): group id per sample.
fn groups(d: &Data, block: usize) -> Vec<usize> {
    let mut g = Vec::with_capacity(d.samples.len());
    let mut id = 0;
    let mut run_len = 0;
    for i in 0..d.samples.len() {
        let same = i > 0 && d.meta[i] == d.meta[i - 1];
        if !same || run_len >= block {
            if i > 0 {
                id += 1;
            }
            run_len = 0;
        }
        run_len += 1;
        g.push(id);
    }
    g
}

/// The 80/20 split by group, stratified by formula (module doc, step 1): `true` = held out.
fn split(d: &Data, g: &[usize], seed: u64) -> Vec<bool> {
    let mut by_formula: BTreeMap<&str, BTreeMap<usize, usize>> = BTreeMap::new();
    for (i, m) in d.meta.iter().enumerate() {
        *by_formula.entry(&m.1).or_default().entry(g[i]).or_default() += 1;
    }
    let mut out_group = BTreeMap::<usize, bool>::new();
    let mut rng = StdRng::seed_from_u64(seed);
    for (_, gs) in by_formula {
        let total: usize = gs.values().sum();
        let mut ids: Vec<(usize, usize)> = gs.into_iter().collect();
        // Fisher–Yates with the seeded generator
        for k in (1..ids.len()).rev() {
            let j = rng.gen_range(0..=k);
            ids.swap(k, j);
        }
        let mut held = 0;
        let n_groups = ids.len();
        for (gid, n) in ids {
            // a group is taken when it brings the held-out count closer to 20 % (so one large block
            // cannot hold out nearly all of a small formula)
            let take = n_groups >= 2 && held as f64 + 0.5 * n as f64 <= 0.2 * total as f64;
            if take {
                held += n;
            }
            out_group.insert(gid, take);
        }
    }
    g.iter().map(|gid| out_group[gid]).collect()
}

/// The species basis of the collection (module doc, step 2).
fn species_basis(d: &Data) -> Basis {
    let mut b = Basis::new(&d.species);
    b.body4 = true;
    b.r_cut = d.r_cut;
    b.degree_cap = crate::surrogate::quarantine::RECOVERY_DEGREE_CAP;
    b.body4_degree = surrogate::BODY4_DEGREE_CAP;
    b
}

struct Fitted {
    model: Model,
    report: FitReport,
    accumulate_s: f64,
    solve_s: f64,
}

fn fit(basis: &Basis, train: &[&Sample], d: &Data) -> Fitted {
    let t = Instant::now();
    let mut acc = Accumulator::new(basis);
    acc.add_many(train, basis, &d.refs, &d.opts).unwrap();
    let accumulate_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let (model, report) = acc.refit(&d.opts, &d.prov, &d.refs).unwrap();
    let solve_s = t.elapsed().as_secs_f64();
    println!("  fit: {:?}, Q′ {}, accumulate {accumulate_s:.1} s, solve {solve_s:.1} s", acc.last_refit(), report.q_prime);
    drop(acc);
    Fitted { model, report, accumulate_s, solve_s }
}

/// θ: the standardised reduced coefficients, one per column class. The fit splits a class's θ equally
/// over its members (`fit` "assemble": coeffs[c] = θ·d_rep/|class|·scale_c, d_rep = 1/scale_rep), so the
/// representative's stored coefficient is θ/|class| and the row (`loo::design_rows`, the representative's
/// value) carries the class once.
fn theta(model: &Model) -> Vec<f64> {
    let p = model.basis.n_features();
    model.basis.column_classes().iter().map(|c| model.coeffs[c[0] / p][c[0] % p] * c.len() as f64).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

struct AtomRes {
    z: u8,
    /// teacher − model, Hartree/Bohr
    df: [f64; 3],
    fref: [f64; 3],
    pred: [f64; 3],
    /// Σ_a σ_epi² of its three rows (Hartree/Bohr)²
    var_epi: f64,
}

struct SampleRes {
    sample: usize,
    /// (teacher − model) per N_e, Hartree/atom; with σ_epi² of the energy row
    energy: Option<(f64, f64)>,
    atoms: Vec<AtomRes>,
}

/// Row predictions of `idx` (module doc, step 3), with σ_epi when `variance`, threaded over samples.
fn evaluate(model: &Model, samples: &[Sample], idx: &[usize], refs: &[f64], variance: bool) -> Vec<SampleRes> {
    let th = theta(model);
    let q = th.len();
    let l = model.sigma_inv_chol.as_ref().expect("a fitted model");
    assert_eq!(l.len(), q * q);
    let threads = surrogate::threads::worker_threads();
    let chunk = idx.len().div_ceil(threads).max(1);
    let mut out: Vec<SampleRes> = std::thread::scope(|sc| {
        let hs: Vec<_> = idx
            .chunks(chunk)
            .map(|part| {
                let th = &th;
                sc.spawn(move || {
                    surrogate::threads::lower_priority();
                    part.iter()
                        .map(|&i| {
                            let s = &samples[i];
                            let (e_row, f_rows) = design_rows(model, &s.pos, &s.z).unwrap();
                            let n = s.pos.len();
                            let taught: Vec<usize> = (0..n).filter(|&j| s.trains_atom(j)).collect();
                            let mut rows: Vec<&[f64]> = Vec::new();
                            if s.use_energy {
                                rows.push(&e_row);
                            }
                            for &j in &taught {
                                for a in 0..3 {
                                    rows.push(&f_rows[3 * j + a]);
                                }
                            }
                            let var: Vec<f64> = if variance { forward_many(l, q, &rows).iter().map(|y| dot(y, y)).collect() } else { vec![0.0; rows.len()] };
                            let sp = model.basis.species_indices(&s.z).unwrap();
                            let ne = surrogate::energy_row_atoms(&model.basis, &s.pos) as f64;
                            let mut k = 0;
                            let energy = s.use_energy.then(|| {
                                let target = (s.energy - sp.iter().map(|&t| refs[t]).sum::<f64>()) / ne;
                                k = 1;
                                (target - dot(&e_row, th), var[0])
                            });
                            let atoms = taught
                                .iter()
                                .enumerate()
                                .map(|(t, &j)| {
                                    let pred: [f64; 3] = std::array::from_fn(|a| dot(&f_rows[3 * j + a], th));
                                    let fref = s.forces[j];
                                    AtomRes {
                                        z: s.z[j],
                                        df: std::array::from_fn(|a| fref[a] - pred[a]),
                                        fref,
                                        pred,
                                        var_epi: var[k + 3 * t] + var[k + 3 * t + 1] + var[k + 3 * t + 2],
                                    }
                                })
                                .collect();
                            SampleRes { sample: i, energy, atoms }
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    out.sort_by_key(|r| r.sample);
    out
}

fn rms(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
}

fn quantile(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let x = p * (s.len() - 1) as f64;
    let (lo, hi) = (x.floor() as usize, x.ceil() as usize);
    s[lo] + (s[hi] - s[lo]) * (x - lo as f64)
}

fn symbol(z: u8) -> &'static str {
    match z {
        1 => "H",
        3 => "Li",
        8 => "O",
        16 => "S",
        17 => "Cl",
        _ => "?",
    }
}

/// The summary of one model's held-out residuals.
#[derive(Clone, Debug, Default)]
struct Summary {
    e_rmse: f64,
    f_rmse: f64,
    n_e: usize,
    n_atoms: usize,
    f_by_z: BTreeMap<u8, (f64, usize)>,
    f_by_formula: BTreeMap<String, (f64, usize)>,
    worst_ratio: f64,
    worst_at: String,
    frac1: f64,
    frac3: f64,
    z_epi: (f64, f64),
    z_tot: (f64, f64),
    z_e: (f64, f64),
    z_epi_by_z: BTreeMap<u8, (f64, f64)>,
}

fn summarise(res: &[SampleRes], d: &Data, model: &Model) -> Summary {
    let pol = FallbackPolicy::default();
    let mut e = Vec::new();
    let mut ze = Vec::new();
    let mut comps = Vec::new();
    let mut by_z: BTreeMap<u8, Vec<f64>> = BTreeMap::new();
    let mut by_f: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut zepi = Vec::new();
    let mut ztot = Vec::new();
    let mut zepi_by_z: BTreeMap<u8, Vec<f64>> = BTreeMap::new();
    let (mut worst, mut worst_at, mut n1, mut n3, mut na) = (0.0f64, String::new(), 0, 0, 0);
    for r in res {
        if let Some((de, ve)) = r.energy {
            e.push(de);
            ze.push(de.abs() / ve.sqrt());
        }
        for a in &r.atoms {
            comps.extend_from_slice(&a.df);
            by_z.entry(a.z).or_default().extend_from_slice(&a.df);
            by_f.entry(d.meta[r.sample].1.clone()).or_default().extend_from_slice(&a.df);
            let dn = (a.df[0] * a.df[0] + a.df[1] * a.df[1] + a.df[2] * a.df[2]).sqrt();
            zepi.push(dn / a.var_epi.sqrt());
            ztot.push(dn / (3.0 * model.sigma_f * model.sigma_f + a.var_epi).sqrt());
            zepi_by_z.entry(a.z).or_default().push(dn / a.var_epi.sqrt());
            let (dd, bound) = pol.force_residual(&a.pred, &a.fref);
            let ratio = dd / bound;
            na += 1;
            n1 += usize::from(ratio > 1.0);
            n3 += usize::from(ratio > 3.0);
            if ratio > worst {
                worst = ratio;
                let fm = (a.fref[0] * a.fref[0] + a.fref[1] * a.fref[1] + a.fref[2] * a.fref[2]).sqrt();
                worst_at = format!("{} in {} #{} (|F| {:.4}, err/comp {:.4}, bound {:.4})", symbol(a.z), d.meta[r.sample].1, r.sample, fm, dd, bound);
            }
        }
    }
    Summary {
        e_rmse: rms(&e),
        f_rmse: rms(&comps),
        n_e: e.len(),
        n_atoms: na,
        f_by_z: by_z.iter().map(|(&z, v)| (z, (rms(v), v.len() / 3))).collect(),
        f_by_formula: by_f.iter().map(|(k, v)| (k.clone(), (rms(v), v.len() / 3))).collect(),
        worst_ratio: worst,
        worst_at,
        frac1: n1 as f64 / na.max(1) as f64,
        frac3: n3 as f64 / na.max(1) as f64,
        z_epi: (quantile(&zepi, 0.5), quantile(&zepi, 0.9)),
        z_tot: (quantile(&ztot, 0.5), quantile(&ztot, 0.9)),
        z_e: (quantile(&ze, 0.5), quantile(&ze, 0.9)),
        z_epi_by_z: zepi_by_z.iter().map(|(&z, v)| (z, (quantile(v, 0.5), quantile(v, 0.9)))).collect(),
    }
}

fn print_summary(label: &str, s: &Summary) {
    println!("  [{label}] held-out: energy RMSE {:.3e} Ha/atom over {} energy rows; force RMSE {:.3e} Ha/Bohr/comp over {} atoms", s.e_rmse, s.n_e, s.f_rmse, s.n_atoms);
    for (z, (r, n)) in &s.f_by_z {
        let zz = s.z_epi_by_z[z];
        println!("    centre {:>2}: force RMSE {r:.3e} ({n} atoms), z_epi median {:.2} p90 {:.2}", symbol(*z), zz.0, zz.1);
    }
    for (f, (r, n)) in &s.f_by_formula {
        println!("    formula {f:>4}: force RMSE {r:.3e} ({n} atoms)");
    }
    println!(
        "    worst err/bound {:.2} at {}; beyond 1× bound {:.2} %, beyond 3× {:.2} %",
        s.worst_ratio,
        s.worst_at,
        100.0 * s.frac1,
        100.0 * s.frac3
    );
    println!("    z_epi (force, per atom) median {:.2} p90 {:.2}; z_tot median {:.2} p90 {:.2}; z_epi energy median {:.2} p90 {:.2}", s.z_epi.0, s.z_epi.1, s.z_tot.0, s.z_tot.1, s.z_e.0, s.z_e.1);
}

fn print_mix(m: &LearntMix, learn_s: f64) {
    println!(
        "  learn_mix: K {} (asked {}), rank {}, {learn_s:.1} s; low-order Q′ {:?} λ {:?}",
        m.k,
        m.k_requested,
        m.rank,
        m.low_order.as_ref().map(|r| r.q_prime),
        m.low_order.as_ref().map(|r| r.lambda)
    );
    println!("    singular values {:?}; gap σ_K/σ_K+1 {:?}; η {:.3}", m.singular_values.iter().map(|v| format!("{v:.4e}")).collect::<Vec<_>>(), m.gap(), m.nonlinearity);
    println!(
        "    truncation residual ρ_t {:?}; weights a_t {:?}",
        m.species.iter().zip(&m.truncation_residual).map(|(z, r)| format!("{} {r:.3e}", symbol(*z))).collect::<Vec<_>>(),
        m.weights.iter().map(|v| format!("{v:.3e}")).collect::<Vec<_>>()
    );
    for t in 0..m.species.len() {
        println!("    W[{}] = {:?}", symbol(m.species[t]), (0..m.k).map(|c| format!("{:+.4}", m.w[t * m.k + c])).collect::<Vec<_>>());
    }
}

/// The training-set force RMSE from the rows (a self-check: the fit's own report computes it from the
/// statistics).
fn train_rmse(model: &Model, d: &Data, train: &[usize]) -> (f64, f64) {
    let res = evaluate(model, &d.samples, train, &d.refs, false);
    let e: Vec<f64> = res.iter().filter_map(|r| r.energy.map(|x| x.0)).collect();
    let f: Vec<f64> = res.iter().flat_map(|r| r.atoms.iter().flat_map(|a| a.df)).collect();
    (rms(&e), rms(&f))
}

fn flush() {
    let _ = std::io::stdout().flush();
}

/// One model of the table.
struct Row {
    label: String,
    q: usize,
    fit_s: f64,
    learn_s: f64,
    report: FitReport,
    s: Summary,
    train: (f64, f64),
}

fn run_model(label: &str, basis: &Basis, train_idx: &[usize], held_idx: &[usize], d: &Data, learn_s: f64) -> Row {
    println!("\n## {label}");
    flush();
    let train: Vec<&Sample> = train_idx.iter().map(|&i| &d.samples[i]).collect();
    let f = fit(basis, &train, d);
    let r = &f.report;
    println!(
        "  report: λ {:.3e}, σ_E {:.3e}, σ_F {:.3e}, log ev {:.6e}, γ {:.1}, train RMSE E {:.3e} F {:.3e}, rows {}, samples {}, converged {}",
        r.lambda, r.sigma_e, r.sigma_f, r.log_evidence, r.dof_eff, r.rmse_energy_per_atom, r.rmse_force, r.n_rows, r.n_samples, r.converged
    );
    let t = Instant::now();
    let tr = train_rmse(&f.model, d, train_idx);
    println!("  train RMSE from the rows: E {:.3e} F {:.3e} ({:.1} s)", tr.0, tr.1, t.elapsed().as_secs_f64());
    let t = Instant::now();
    let res = evaluate(&f.model, &d.samples, held_idx, &d.refs, true);
    println!("  held-out rows + σ_epi: {:.1} s", t.elapsed().as_secs_f64());
    let s = summarise(&res, d, &f.model);
    print_summary(label, &s);
    flush();
    Row { label: label.into(), q: r.q_prime, fit_s: f.accumulate_s + f.solve_s, learn_s, report: f.report.clone(), s, train: tr }
}

fn mix_setup(d: &Data) -> MixSetup {
    let mut m = MixSetup::new(&d.species, &d.refs, d.r_cut);
    m.fit = d.opts.clone();
    m
}

fn channel_model(label: &str, k: usize, train_idx: &[usize], held_idx: &[usize], d: &Data) -> Row {
    println!("\n## {label}: learning W (K = {k}) on {} training samples", train_idx.len());
    flush();
    let train: Vec<&Sample> = train_idx.iter().map(|&i| &d.samples[i]).collect();
    let t = Instant::now();
    let mix = learn_mix(&train, &mix_setup(d), k).unwrap();
    let learn_s = t.elapsed().as_secs_f64();
    print_mix(&mix, learn_s);
    let basis = species_basis(d).with_channels(ChannelMix { k: mix.k, w: mix.w.clone() });
    run_model(label, &basis, train_idx, held_idx, d, learn_s)
}

#[test]
#[ignore = "measurement: COLLECTION_DIR=<copy> cargo test -p providers --release --lib channels_gate -- --ignored --nocapture --test-threads 1"]
fn channels_gate() {
    let mode = std::env::var("GATE").unwrap_or_else(|_| "full".into());
    let block: usize = std::env::var("GATE_BLOCK").map_or(20, |v| v.parse().unwrap());
    let seed: u64 = std::env::var("GATE_SEED").map_or(2026, |v| v.parse().unwrap());
    println!("worker threads {}; mode {mode}; block {block}; seed {seed}", surrogate::threads::worker_threads());
    let d = load();
    let g = groups(&d, block);
    let held = split(&d, &g, seed);
    let train_idx: Vec<usize> = (0..d.samples.len()).filter(|&i| !held[i]).collect();
    let held_idx: Vec<usize> = (0..d.samples.len()).filter(|&i| held[i]).collect();
    // the split, per formula
    let mut per: BTreeMap<&str, (usize, usize, BTreeMap<usize, ()>, BTreeMap<usize, ()>)> = BTreeMap::new();
    for i in 0..d.samples.len() {
        let e = per.entry(&d.meta[i].1).or_default();
        if held[i] {
            e.1 += 1;
            e.3.insert(g[i], ());
        } else {
            e.0 += 1;
            e.2.insert(g[i], ());
        }
    }
    println!("\nsplit: {} groups; train {} / held out {}", g.last().map_or(0, |x| x + 1), train_idx.len(), held_idx.len());
    for (f, (a, b, ga, gb)) in &per {
        println!("  {f:>4}: train {a} samples in {} groups, held out {b} in {} groups", ga.len(), gb.len());
    }
    let atoms = |idx: &[usize]| {
        let mut m: BTreeMap<u8, usize> = BTreeMap::new();
        for &i in idx {
            for &z in &d.samples[i].z {
                *m.entry(z).or_default() += 1;
            }
        }
        m.into_iter().map(|(z, n)| format!("{} {n}", symbol(z))).collect::<Vec<_>>()
    };
    println!("  atoms: train {:?}, held out {:?}", atoms(&train_idx), atoms(&held_idx));
    if mode == "dump" {
        let mut runs: Vec<(usize, &(String, String, String, usize))> = Vec::new();
        for (i, m) in d.meta.iter().enumerate() {
            match runs.last_mut() {
                Some((n, last)) if *last == m => *n += 1,
                _ => runs.push((1, m)),
            }
            let _ = i;
        }
        println!("\n{} runs (before blocking):", runs.len());
        for (n, m) in &runs {
            println!("  {n:>5} × {m:?}");
        }
        return;
    }
    if mode == "tiny" {
        tiny(&d, &train_idx, &held_idx);
        return;
    }
    let models = std::env::var("GATE_MODELS").unwrap_or_else(|_| "species,k3,k4,thin20,thin5".into());
    let want = |m: &str| models.split(',').any(|x| x.trim() == m);
    let mut rows: Vec<Row> = Vec::new();
    if want("species") {
        let b = species_basis(&d);
        println!("\nspecies basis: n_max {} l_max {} degree_cap {} body4 {} D₄ {} r_cut {} Q′ {}", b.n_max, b.l_max, b.degree_cap, b.body4, b.body4_degree, b.r_cut, b.n_independent_columns());
        let r = run_model("species (= K = S control)", &b, &train_idx, &held_idx, &d, 0.0);
        rows.push(r);
    }
    for k in [3usize, 4] {
        if want(&format!("k{k}")) {
            rows.push(channel_model(&format!("channels K = {k}"), k, &train_idx, &held_idx, &d));
        }
    }
    // the rare-element probe (I7f): the rarest element with ≥ 20 training samples
    let mut count: BTreeMap<u8, usize> = BTreeMap::new();
    for &i in &train_idx {
        let mut zs = d.samples[i].z.clone();
        zs.sort_unstable();
        zs.dedup();
        for z in zs {
            *count.entry(z).or_default() += 1;
        }
    }
    println!("\ntraining samples containing each element: {:?}", count.iter().map(|(z, n)| format!("{} {n}", symbol(*z))).collect::<Vec<_>>());
    // held-out atoms per element: the probe needs enough of them to measure an RMSE
    let mut held_atoms: BTreeMap<u8, usize> = BTreeMap::new();
    for &i in &held_idx {
        for &z in &d.samples[i].z {
            *held_atoms.entry(z).or_default() += 1;
        }
    }
    let rare = count
        .iter()
        .filter(|(z, &n)| n >= 20 && held_atoms.get(z).copied().unwrap_or(0) >= 20)
        .min_by_key(|(_, &n)| n)
        .map(|(&z, _)| z)
        .unwrap();
    println!("held-out atoms per element: {held_atoms:?}; the probe element (rarest with ≥ 20 training samples and ≥ 20 held-out atoms): {}", symbol(rare));
    for (tag, frac) in [("thin20", 0.20), ("thin5", 0.05)] {
        if !want(tag) {
            continue;
        }
        let with: Vec<usize> = train_idx.iter().copied().filter(|&i| d.samples[i].z.contains(&rare)).collect();
        let mut order = with.clone();
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5eed);
        for k in (1..order.len()).rev() {
            let j = rng.gen_range(0..=k);
            order.swap(k, j);
        }
        let keep_n = ((frac * with.len() as f64).ceil() as usize).max(1);
        let keep: std::collections::BTreeSet<usize> = order[..keep_n].iter().copied().collect();
        let thin: Vec<usize> = train_idx.iter().copied().filter(|i| !with.contains(i) || keep.contains(i)).collect();
        println!("\nrare-element probe: {} thinned to {keep_n} of {} training samples ({:.0} %)", symbol(rare), with.len(), 100.0 * frac);
        rows.push(channel_model(&format!("channels K = 4, {} at {:.0} %", symbol(rare), 100.0 * frac), 4, &thin, &held_idx, &d));
    }
    // the table
    println!("\n## Table (held out by scene group; force RMSE per component, Ha/Bohr; energy RMSE Ha/atom)\n");
    println!("| model | Q′ | fit wall s (W learn s) | E RMSE | F RMSE | F RMSE by centre | worst err/bound | > 1× | > 3× | z_epi med / p90 | z_tot med / p90 | λ | σ_F |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for r in &rows {
        let byz: Vec<String> = r.s.f_by_z.iter().map(|(z, (v, n))| format!("{} {v:.2e} ({n})", symbol(*z))).collect();
        println!(
            "| {} | {} | {:.0} ({:.0}) | {:.3e} | {:.3e} | {} | {:.2} | {:.1} % | {:.1} % | {:.2} / {:.2} | {:.2} / {:.2} | {:.2e} | {:.2e} |",
            r.label,
            r.q,
            r.fit_s,
            r.learn_s,
            r.s.e_rmse,
            r.s.f_rmse,
            byz.join(", "),
            r.s.worst_ratio,
            100.0 * r.s.frac1,
            100.0 * r.s.frac3,
            r.s.z_epi.0,
            r.s.z_epi.1,
            r.s.z_tot.0,
            r.s.z_tot.1,
            r.report.lambda,
            r.report.sigma_f
        );
    }
    if let Some(sp) = rows.iter().find(|r| r.label.starts_with("species")) {
        println!("\n## Gate against the species model");
        for r in rows.iter().filter(|r| r.label.starts_with("channels K = 3") || r.label == "channels K = 4") {
            let fr = r.s.f_rmse / sp.s.f_rmse;
            let er = r.s.e_rmse / sp.s.e_rmse;
            let wr = r.fit_s / sp.fit_s;
            let wr2 = (r.fit_s + r.learn_s) / sp.fit_s;
            println!(
                "  {}: force RMSE ratio {fr:.3} ({}), energy RMSE ratio {er:.3} ({}), fit wall ratio {wr:.3} ({}; with the W relearn {wr2:.3}) → {}",
                r.label,
                if fr <= 1.10 { "≤ 1.10" } else { "> 1.10" },
                if er <= 1.10 { "≤ 1.10" } else { "> 1.10" },
                if wr <= 0.25 { "≤ 0.25" } else { "> 0.25" },
                if fr <= 1.10 && er <= 1.10 && wr <= 0.25 { "GO" } else { "NO-GO" }
            );
        }
    }
    for r in &rows {
        println!("  train RMSE (rows) {}: E {:.3e} F {:.3e}; report E {:.3e} F {:.3e}", r.label, r.train.0, r.train.1, r.report.rmse_energy_per_atom, r.report.rmse_force);
    }
    flush();
}

/// `GATE=tiny`: the harness's own checks on a small basis (n_max 4, l_max 2, D₄ 2) and 400 training samples —
/// (1) the row prediction and σ_epi equal the species model's hot path and its `force_std_epistemic`;
/// (2) the channel basis at K = S with W = I fits to the species model's predictions (the K = S control);
/// (3) a K = 3 channel fit's training RMSE from the rows matches its report (the rows are the fit's).
fn tiny(d: &Data, train_idx: &[usize], held_idx: &[usize]) {
    let mut b = species_basis(d);
    b.n_max = 4;
    b.l_max = 2;
    b.degree_cap = 4;
    b.body4_degree = surrogate::BODY4_DEGREE_START;
    let train: Vec<usize> = train_idx.iter().copied().step_by(train_idx.len() / 400).collect();
    let held: Vec<usize> = held_idx.iter().copied().step_by(held_idx.len() / 60).collect();
    let samples: Vec<&Sample> = train.iter().map(|&i| &d.samples[i]).collect();
    println!("\ntiny: {} train, {} held out, Q′ {}", train.len(), held.len(), b.n_independent_columns());
    let sp = fit(&b, &samples, d);
    // (1)
    let res = evaluate(&sp.model, &d.samples, &held, &d.refs, true);
    let (mut de, mut df, mut dv) = (0.0f64, 0.0f64, 0.0f64);
    for r in &res {
        let s = &d.samples[r.sample];
        let p = sp.model.predict(&s.pos, &s.z, true).unwrap();
        let ne = surrogate::energy_row_atoms(&sp.model.basis, &s.pos) as f64;
        let sidx = sp.model.basis.species_indices(&s.z).unwrap();
        if let Some((e, _)) = r.energy {
            let target = (s.energy - sidx.iter().map(|&t| d.refs[t]).sum::<f64>()) / ne;
            let hot = (p.energy - sidx.iter().map(|&t| sp.model.ref_energy[t]).sum::<f64>()) / ne;
            de = de.max(((target - e) - hot).abs());
        }
        let taught: Vec<usize> = (0..s.pos.len()).filter(|&j| s.trains_atom(j)).collect();
        for (a, &j) in r.atoms.iter().zip(&taught) {
            for c in 0..3 {
                df = df.max((a.pred[c] - p.forces[j][c]).abs());
            }
            let hot = p.force_std_epistemic.as_ref().unwrap()[j];
            dv = dv.max((a.var_epi.sqrt() - hot).abs() / hot.max(1e-300));
        }
    }
    println!("  (1) rows vs hot path (species): max |ΔE| {de:.3e} Ha/atom, max |ΔF| {df:.3e} Ha/Bohr, max rel Δσ_epi {dv:.3e}");
    assert!(de < 1e-9 && df < 1e-9 && dv < 1e-6, "the row prediction is not the model's");
    // (2)
    let s_n = d.species.len();
    let eye: Vec<f64> = (0..s_n * s_n).map(|q| if q / s_n == q % s_n { 1.0 } else { 0.0 }).collect();
    let bi = b.clone().with_channels(ChannelMix { k: s_n, w: eye });
    let ci = fit(&bi, &samples, d);
    let rc = evaluate(&ci.model, &d.samples, &held, &d.refs, true);
    let (mut me, mut mf, mut mv) = (0.0f64, 0.0f64, 0.0f64);
    for (a, b2) in res.iter().zip(&rc) {
        if let (Some(x), Some(y)) = (a.energy, b2.energy) {
            me = me.max((x.0 - y.0).abs());
        }
        for (u, v) in a.atoms.iter().zip(&b2.atoms) {
            for c in 0..3 {
                mf = mf.max((u.pred[c] - v.pred[c]).abs());
            }
            mv = mv.max((u.var_epi.sqrt() - v.var_epi.sqrt()).abs() / u.var_epi.sqrt());
        }
    }
    println!(
        "  (2) K = S, W = I vs species: Q′ {} vs {}, λ {:.6e} vs {:.6e}; max |ΔE| {me:.3e}, max |ΔF| {mf:.3e}, max rel Δσ_epi {mv:.3e}",
        ci.report.q_prime, sp.report.q_prime, ci.report.lambda, sp.report.lambda
    );
    // (3)
    let mix = learn_mix(&samples, &mix_setup(d), 3).unwrap();
    print_mix(&mix, 0.0);
    let bc = b.clone().with_channels(ChannelMix { k: mix.k, w: mix.w.clone() });
    let c3 = fit(&bc, &samples, d);
    let tr = train_rmse(&c3.model, d, &train);
    println!(
        "  (3) K = 3: Q′ {}; train RMSE from the rows E {:.6e} F {:.6e}; the report's E {:.6e} F {:.6e}",
        c3.report.q_prime, tr.0, tr.1, c3.report.rmse_energy_per_atom, c3.report.rmse_force
    );
    let s3 = summarise(&evaluate(&c3.model, &d.samples, &held, &d.refs, true), d, &c3.model);
    print_summary("tiny K = 3", &s3);
    print_summary("tiny species", &summarise(&res, d, &sp.model));
}
