//! Diagnosis (2026-11-12, the user's "why does the collection fail residuals every refit?"):
//! the training-residual condition on a copy of a collection, sample by sample. Reported, never
//! asserted; nothing here changes what a refit does. Findings: `references.md` "Why the collection
//! fails its residuals at every refit".
//!
//! `COLLECTION_DIR=<copy> cargo test -p providers --release residual_diag -- --ignored --nocapture --test-threads 1`
//! (a **copy**: the attach's refit writes its quarantine codes to it). Runs the attach's refit,
//! then fits without recovery from the codes on disk and from the attach's. `DIAG_TESTS=1` adds
//! the hold-out fits (H₃O held, by multiplicity, H₃O alone, each on the default and the cap-6
//! basis); `DIAG_SPIKE=1` runs the body-4 measurement's five fits instead; `DIAG_RANGES=a-b,c-d` dumps those samples in store order (distances, per-atom forces).
//!
//! `acceptance_on_the_collection` (spec 2026-11-13 §5.1–5.2, Task 8) is the one test here that
//! asserts: the open-shell model's attach, its refits' fixed point and the D₄ = 3 against D₄ = 2
//! comparison on a fresh copy of the user's collection (its doc).
use super::*;
use std::collections::BTreeMap;
use surrogate::SampleTag;

struct Meta {
    scene: String,
    grid: usize,
    tag: SampleTag,
    n: usize,
    q0: u8,
    mult: Option<u32>,
}

fn collection_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("COLLECTION_DIR").expect("COLLECTION_DIR names a collection folder copy"))
}

fn open_at(dir: &std::path::Path) -> surrogate::Collection {
    let key = surrogate::SurfaceKey::from_text(&std::fs::read_to_string(dir.join("key.txt")).unwrap()).unwrap();
    surrogate::Collection::open(dir.parent().unwrap(), &key).unwrap().0
}

fn open_copy() -> surrogate::Collection {
    open_at(&collection_dir())
}

/// A fresh copy of the collection folder `dir` (its files, the lock left behind) at
/// `<dir's parent>_<tag>/<dir's name>`, made before anything writes to `dir`: a second provider's
/// own copy. Refuses a target that exists (every run copies afresh).
fn sibling_copy(dir: &std::path::Path, tag: &str) -> std::path::PathBuf {
    let parent = dir.parent().expect("the collection folder has a parent");
    let root = parent.with_file_name(format!("{}_{tag}", parent.file_name().unwrap().to_string_lossy()));
    assert!(!root.exists(), "{} exists: copy the collection afresh for every run", root.display());
    let to = root.join(dir.file_name().unwrap());
    std::fs::create_dir_all(&to).unwrap();
    for e in std::fs::read_dir(dir).unwrap() {
        let e = e.unwrap();
        if e.file_type().unwrap().is_file() && e.file_name() != ".lock" {
            std::fs::copy(e.path(), to.join(e.file_name())).unwrap();
        }
    }
    to
}

fn metas(c: &surrogate::Collection) -> Vec<Meta> {
    c.samples
        .iter()
        .zip(c.quarantine.iter().copied().chain(std::iter::repeat(0)))
        .map(|(x, q0)| Meta { scene: x.scene.clone(), grid: x.grid_n, tag: x.sample.tag, n: x.sample.pos.len(), q0, mult: x.multiplicity })
        .collect()
}

/// Per stored sample under the current model: (energy ratio, worst force ratio, signed energy
/// residual per atom model − teacher, worst force residual, its bound, |F| there).
fn ratios(s: &SurrogateKs) -> Vec<Option<(f64, f64, f64, f64, f64, f64)>> {
    let (model, store) = (s.model.as_ref().unwrap(), s.store.as_ref().unwrap());
    let p = &s.policy;
    store
        .samples
        .iter()
        .map(|x| {
            let pr = model.predict(&x.pos, &x.z, false).ok()?;
            let n = x.pos.len().max(1) as f64;
            let de = (pr.energy - x.energy) / n;
            let er = if x.use_energy { de.abs() / p.energy_std_per_atom } else { 0.0 };
            let (mut fr, mut fd, mut fb, mut fm) = (0.0f64, 0.0, 0.0, 0.0);
            for (j, (a, b)) in pr.forces.iter().zip(&x.forces).enumerate() {
                if !x.trains_atom(j) {
                    continue;
                }
                let (d, bound) = p.force_residual(a, b);
                if d / bound > fr {
                    fr = d / bound;
                    fd = d;
                    fb = bound;
                    fm = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
                }
            }
            Some((er, fr, de, fd, fb, fm))
        })
        .collect()
}

fn report(s: &SurrogateKs, meta: &[Meta], label: &str) {
    let r = s.report().unwrap();
    println!("\n=== {label}");
    println!(
        "trusted {} — {:?}; quarantined {}; degree_cap {:?}; fits {}; recovery {:?}",
        s.trusted(),
        s.untrusted_reason(),
        s.quarantined(),
        s.degree_cap,
        s.fits,
        s.recovery_note()
    );
    println!(
        "report: n {} Q′ {} σ_E {:.2e} σ_F {:.2e} rmse_E/atom {:.2e} rmse_F {:.2e} dof_e {:.1} dof_f {:.1}; training residuals {:?}",
        r.n_samples, r.q_prime, r.sigma_e, r.sigma_f, r.rmse_energy_per_atom, r.rmse_force, r.residual_dof_e, r.residual_dof_f, s.training_residuals()
    );
    let mut held: BTreeMap<String, usize> = BTreeMap::new();
    for (i, m) in meta.iter().enumerate() {
        if s.quarantine.get(i).copied().unwrap_or(0) != 0 {
            *held.entry(format!("{} q{}", m.scene, s.quarantine[i])).or_default() += 1;
        }
    }
    println!("held by scene: {held:?}");
    let rs = ratios(s);
    let fitted = |i: usize| i < s.accumulated && !s.excluded(i);
    let mut order: Vec<usize> = (0..rs.len()).filter(|&i| rs[i].is_some()).collect();
    order.sort_by(|&a, &b| {
        let (ra, rb) = (rs[a].unwrap(), rs[b].unwrap());
        rb.0.max(rb.1).partial_cmp(&ra.0.max(ra.1)).unwrap()
    });
    println!("worst 25 (all stored samples; F = fitted now, q0 = code on disk, q = code now):");
    println!("| i | F | q0 | q | scene | grid | tag | n | E ratio | dE/atom | F ratio | |ΔF|/√3 | bound | |F| |");
    for &i in order.iter().take(25) {
        let (er, fr, de, fd, fb, fm) = rs[i].unwrap();
        let m = &meta[i];
        let q = s.quarantine.get(i).copied().unwrap_or(0);
        println!(
            "| {i} | {} | {} | {q} | {} | {} | {:?} | {} | {er:.2} | {de:+.2e} | {fr:.2} | {fd:.2e} | {fb:.2e} | {fm:.2e} |",
            if fitted(i) { "F" } else { "-" },
            m.q0,
            m.scene,
            m.grid,
            m.tag,
            m.n
        );
    }
    println!("worst 15 fitted:");
    for &i in order.iter().filter(|&&i| fitted(i)).take(15) {
        let (er, fr, de, fd, fb, fm) = rs[i].unwrap();
        let m = &meta[i];
        println!("| {i} | {} | {} | {:?} | {} | {er:.2} | {de:+.2e} | {fr:.2} | {fd:.2e} | {fb:.2e} | {fm:.2e} | mult {:?}", m.scene, m.grid, m.tag, m.n, m.mult);
    }
    // fitted samples over 1 by scene / grid / tag, energy vs force
    let mut by: BTreeMap<String, (usize, usize, usize, f64, f64)> = BTreeMap::new();
    for i in (0..rs.len()).filter(|&i| fitted(i)) {
        let Some((er, fr, de, _, _, _)) = rs[i] else { continue };
        let m = &meta[i];
        for k in [format!("scene {}", m.scene), format!("grid {}", m.grid), format!("tag {:?}", m.tag), format!("scene {} grid {}", m.scene, m.grid)] {
            let e = by.entry(k).or_default();
            e.0 += 1;
            e.1 += (er > 1.0) as usize;
            e.2 += (fr > 1.0) as usize;
            e.3 += de;
            e.4 += de * de;
        }
    }
    let mut fs: BTreeMap<String, (usize, f64, f64, Vec<f64>)> = BTreeMap::new();
    {
        let (model, store) = (s.model.as_ref().unwrap(), s.store.as_ref().unwrap());
        for i in (0..store.len()).filter(|&i| fitted(i)) {
            let x = &store.samples[i];
            let Ok(pr) = model.predict(&x.pos, &x.z, false) else { continue };
            let e = fs.entry(meta[i].scene.clone()).or_insert((0, 0.0, 0.0, Vec::new()));
            for j in (0..x.pos.len()).filter(|&j| x.trains_atom(j)) {
                let (d, b) = s.policy.force_residual(&pr.forces[j], &x.forces[j]);
                e.0 += 1;
                e.1 += d * d;
                e.2 = e.2.max(d / b);
                e.3.push(d);
            }
        }
    }
    println!("force residual per scene: | scene | atoms | rms |ΔF|/√3 | p99 | p99.9 | max ratio |");
    for (k, (n, sq, mr, mut v)) in fs {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |f: f64| v[((v.len() as f64 * f) as usize).min(v.len() - 1)];
        println!("| {k} | {n} | {:.2e} | {:.2e} | {:.2e} | {mr:.2} |", (sq / n as f64).sqrt(), q(0.99), q(0.999));
    }
    println!("fitted samples: | group | n | E ratio > 1 | F ratio > 1 | mean dE/atom | rms dE/atom |");
    for (k, (n, e1, f1, sum, sq)) in by {
        println!("| {k} | {n} | {e1} | {f1} | {:+.2e} | {:.2e} |", sum / n as f64, (sq / n as f64).sqrt());
    }
}

/// Index-ordered dump of `range`: the residuals, the distances, the multiplicity, per-atom forces.
fn dump(s: &SurrogateKs, meta: &[Meta], range: std::ops::Range<usize>) {
    let (model, store) = (s.model.as_ref().unwrap(), s.store.as_ref().unwrap());
    let rs = ratios(s);
    println!("
--- dump {range:?}: i | fitted | q | scene grid tag mult | E (Ha) | dE/atom | F ratio | distances (Bohr) | per atom: z |F_teacher| |F_model−F|/√3");
    for i in range {
        let x = &store.samples[i];
        let m = &meta[i];
        let Some((_er, fr, de, ..)) = rs[i] else { continue };
        let pr = model.predict(&x.pos, &x.z, false).unwrap();
        let mut d = Vec::new();
        for a in 0..x.pos.len() {
            for b in a + 1..x.pos.len() {
                let r = ((0..3).map(|k| (x.pos[a][k] - x.pos[b][k]).powi(2)).sum::<f64>()).sqrt();
                d.push(format!("{}{}{}{}:{r:.3}", x.z[a], a, x.z[b], b));
            }
        }
        let atoms: Vec<String> = (0..x.pos.len())
            .map(|j| {
                let f = x.forces[j];
                let fm = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
                let dd = ((0..3).map(|k| (pr.forces[j][k] - f[k]).powi(2)).sum::<f64>()).sqrt() / 3f64.sqrt();
                format!("{}:{fm:.2e}/{dd:.2e}", x.z[j])
            })
            .collect();
        let fitted = i < s.accumulated && !s.excluded(i);
        println!(
            "{i} | {} | {} | {} {} {:?} {:?} | {:.6} | {de:+.2e} | {fr:.2} | {} | {}",
            if fitted { "F" } else { "-" },
            s.quarantine.get(i).copied().unwrap_or(0),
            m.scene,
            m.grid,
            m.tag,
            m.mult,
            x.energy,
            d.join(" "),
            atoms.join(" ")
        );
    }
}

#[test]
#[ignore]
fn residual_failures_on_a_collection() {
    let c = open_copy();
    let meta = metas(&c);
    let q0: Vec<u8> = c.quarantine.clone();
    println!("collection: {} samples, {} held on disk", meta.len(), q0.iter().filter(|&&q| q != 0).count());
    let mut s = SurrogateKs::new(Box::new(StubKsGpu), FallbackPolicy::default()).with_teacher_tol(1e-6);
    s.attach(c).unwrap();
    if s.report().is_none() {
        SurrogateKs::refit(&mut s).unwrap();
    }
    report(&s, &meta, "after the attach's refit (recovery included)");

    // the entry fit: the disk's codes, the default basis, no recovery
    let after_attach_q = s.quarantine.clone();
    s.quarantine = q0.clone();
    s.degree_cap = None;
    s.species_dirty = true;
    s.fit_store().unwrap();
    report(&s, &meta, "entry fit: the codes on disk, default basis, no recovery");

    // the same with the attach's quarantine, no recovery: what the next refit starts from
    s.quarantine = after_attach_q;
    s.species_dirty = true;
    s.fit_store().unwrap();
    report(&s, &meta, "the attach's codes, default basis, no recovery");
    if std::env::var("DIAG_SPIKE").is_ok() {
        // The body-4 measurement (references.md "Raising the body-4 degree"): the same fits, timed, at the build's D₄ — reproduced with `body4_cap` returning the cap under test.
        let cap6 = Some(super::super::quarantine::RECOVERY_DEGREE_CAP);
        let with = |keep: &dyn Fn(usize, &Meta) -> bool| {
            let mut q = q0.clone();
            q.resize(meta.len(), 0);
            for (i, m) in meta.iter().enumerate() {
                if !keep(i, m) {
                    q[i] = surrogate::quarantine::Q_HELD;
                } else if q[i] == surrogate::quarantine::Q_HELD {
                    q[i] = 0;
                }
            }
            q
        };
        let unheld = |q: &[u8]| {
            let mut q = q.to_vec();
            q.resize(meta.len(), 0);
            for c in q.iter_mut() {
                if *c == surrogate::quarantine::Q_HELD {
                    *c = 0;
                }
            }
            q
        };
        let fits: Vec<(&str, Option<usize>, Vec<u8>)> = vec![
            ("S0: the disk's codes, default body-3 (cap 4), the build's D4", None, q0.clone()),
            ("S1: the disk's codes, cap 6, the build's D4", cap6, q0.clone()),
            ("S2: no residual holds at all (every held sample back), cap 6, the build's D4", cap6, unheld(&q0)),
            ("S3: every H3O held, everything else back, cap 6, the build's D4", cap6, with(&|_, m: &Meta| m.scene != "H3O")),
            ("S4: H3O only, cap 6, the build's D4", cap6, with(&|_, m: &Meta| m.scene == "H3O")),
        ];
        for (label, cap, q) in fits {
            s.quarantine = q;
            s.degree_cap = cap;
            s.species_dirty = true;
            let t = std::time::Instant::now();
            match s.fit_store() {
                Ok(_) => {
                    println!("\n[{label}] fit took {:.1} s", t.elapsed().as_secs_f64());
                    report(&s, &meta, label)
                }
                Err(e) => println!("\n=== {label}: fit failed {e}"),
            }
        }
        s.degree_cap = None;
        return;
    }
    if std::env::var("DIAG_TESTS").is_ok() {
        // T1: every H3O sample held, the rest as on disk
        let mut q = q0.clone();
        q.resize(meta.len(), 0);
        for (i, m) in meta.iter().enumerate() {
            if m.scene == "H3O" {
                q[i] = 1;
            }
        }
        s.quarantine = q;
        s.degree_cap = None;
        s.species_dirty = true;
        s.fit_store().unwrap();
        report(&s, &meta, "T1: the disk's codes + every H3O sample held, default basis, no recovery");
        // T3: the disk's codes on the degree-cap-6 basis
        s.quarantine = q0.clone();
        s.degree_cap = Some(super::super::quarantine::RECOVERY_DEGREE_CAP);
        s.species_dirty = true;
        s.fit_store().unwrap();
        report(&s, &meta, "T3: the disk's codes, degree_cap 6, no recovery");
        // T4/T5/T6: by multiplicity (the disk's codes kept), and T1 on the cap-6 basis
        let held = |keep: &dyn Fn(&Meta) -> bool| {
            let mut q = q0.clone();
            q.resize(meta.len(), 0);
            for (i, m) in meta.iter().enumerate() {
                if !keep(m) {
                    q[i] = 1;
                }
            }
            q
        };
        for (label, cap, keep) in [
            ("T6: the disk's codes + every H3O held, degree_cap 6", Some(super::super::quarantine::RECOVERY_DEGREE_CAP), &(|m: &Meta| m.scene != "H3O") as &dyn Fn(&Meta) -> bool),
            ("T4: doublets only (H3, H3O, HO), default basis", None, &|m: &Meta| m.mult == Some(2)),
            ("T4b: doublets only, degree_cap 6", Some(super::super::quarantine::RECOVERY_DEGREE_CAP), &|m: &Meta| m.mult == Some(2)),
            ("T5: singlets only (H2, H2O, N2), default basis", None, &|m: &Meta| m.mult == Some(1)),
            ("T7: H3O only, default basis", None, &|m: &Meta| m.scene == "H3O"),
        ] {
            s.quarantine = held(keep);
            s.degree_cap = cap;
            s.species_dirty = true;
            match s.fit_store() {
                Ok(_) => report(&s, &meta, label),
                Err(e) => println!("
=== {label}: fit failed {e}"),
            }
        }
        s.degree_cap = None;
        return;
    }
    if let Ok(r) = std::env::var("DIAG_RANGES") {
        for part in r.split(',') {
            let (a, b) = part.split_once('-').unwrap();
            dump(&s, &meta, a.parse().unwrap()..b.parse::<usize>().unwrap().min(meta.len()));
        }
    }
    let mut mults: BTreeMap<String, usize> = BTreeMap::new();
    for m in &meta {
        *mults.entry(format!("{} {:?}", m.scene, m.mult)).or_default() += 1;
    }
    println!("multiplicities by scene: {mults:?}");
}

/// Per scene over the fitted samples (the accumulated prefix, excluded samples skipped — M's
/// samples are fitted and counted): the in-sample force rms and the leave-one-out force RMS, both
/// per Cartesian component over the trained atoms (Ha/Bohr; `loo_rms`'s definition, by scene), with
/// the samples and atoms counted and the samples whose LOO is undefined.
#[derive(Clone, Copy, Debug, Default)]
struct SceneRms {
    samples: usize,
    atoms: usize,
    rms: f64,
    loo: f64,
    loo_undefined: usize,
}

fn scene_rms(s: &SurrogateKs, meta: &[Meta]) -> BTreeMap<String, SceneRms> {
    let (model, store) = (s.model.as_ref().unwrap(), s.store.as_ref().unwrap());
    let fitted: Vec<usize> = (0..s.accumulated.min(store.len())).filter(|&i| !s.excluded(i)).collect();
    let refs: Vec<&surrogate::fit::Sample> = fitted.iter().map(|&i| &store.samples[i]).collect();
    let loo = surrogate::loo::leave_one_out_many(model, &refs);
    // (samples, atoms, Σ in-sample², in-sample components, Σ LOO², LOO components, undefined)
    let mut acc: BTreeMap<String, (usize, usize, f64, usize, f64, usize, usize)> = BTreeMap::new();
    for (&i, l) in fitted.iter().zip(&loo) {
        let x = &store.samples[i];
        let pr = model.predict(&x.pos, &x.z, false).expect("a fitted sample predicts");
        let e = acc.entry(meta[i].scene.clone()).or_default();
        e.0 += 1;
        for j in (0..x.pos.len()).filter(|&j| x.trains_atom(j)) {
            e.1 += 1;
            e.2 += (0..3).map(|a| (pr.forces[j][a] - x.forces[j][a]).powi(2)).sum::<f64>();
            e.3 += 3;
        }
        match l {
            None => e.6 += 1,
            Some(l) => {
                for r in l.forces.iter().flatten() {
                    e.4 += r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
                    e.5 += 3;
                }
            }
        }
    }
    acc.into_iter()
        .map(|(k, (n, atoms, sq, nc, lsq, lnc, und))| {
            (k, SceneRms { samples: n, atoms, rms: (sq / nc.max(1) as f64).sqrt(), loo: (lsq / lnc.max(1) as f64).sqrt(), loo_undefined: und })
        })
        .collect()
}

fn by_scene(meta: &[Meta], idx: impl IntoIterator<Item = usize>) -> BTreeMap<String, usize> {
    let mut m: BTreeMap<String, usize> = BTreeMap::new();
    for i in idx {
        *m.entry(meta[i].scene.clone()).or_default() += 1;
    }
    m
}

fn held(q: &[u8]) -> usize {
    q.iter().filter(|&&c| c == surrogate::quarantine::Q_HELD).count()
}

/// Samples `from` held that `to` no longer holds, and samples `to` holds that `from` did not.
fn released_and_new(from: &[u8], to: &[u8]) -> (usize, usize) {
    use surrogate::quarantine::{Q_HELD, Q_NONE};
    let at = |q: &[u8], i: usize| q.get(i).copied().unwrap_or(Q_NONE);
    let n = from.len().max(to.len());
    let rel = (0..n).filter(|&i| at(from, i) == Q_HELD && at(to, i) != Q_HELD).count();
    let new = (0..n).filter(|&i| at(from, i) != Q_HELD && at(to, i) == Q_HELD).count();
    (rel, new)
}

fn misfit_samples(s: &SurrogateKs) -> Vec<usize> {
    s.misfit().map(|m| m.samples.clone()).unwrap_or_default()
}

/// A class key for the tables: the centre, its neighbours, its class (display symbols only).
fn key_name(k: &surrogate::EnvKey) -> String {
    let sym = |z: u8| match z {
        1 => "H".to_string(),
        6 => "C".to_string(),
        7 => "N".to_string(),
        8 => "O".to_string(),
        z => format!("Z{z}"),
    };
    format!("{}{{{}}} c{}", sym(k.z), k.neighbours.iter().map(|&z| sym(z)).collect::<Vec<_>>().join(","), k.class)
}

/// The standing class-rule verdict of environment key `k` (`class_loo`'s per-key verdicts):
/// `Some(None)` trusted, `Some(Some(refusal))` refused, `None` when the key is not judged (no
/// sample of its own).
fn key_verdict(s: &SurrogateKs, k: &surrogate::EnvKey) -> Option<Option<(sim::ClassKey, f64, f64)>> {
    let c = s.class_loo().get((k.class - 1) as usize)?;
    c.species.iter().find(|v| v.z == k.z && v.neighbours == k.neighbours).map(|v| v.refusal)
}

/// The environment keys of the trained atoms of `scene`'s fitted samples.
fn scene_keys(s: &SurrogateKs, meta: &[Meta], scene: &str) -> std::collections::BTreeSet<surrogate::EnvKey> {
    let (model, store) = (s.model.as_ref().unwrap(), s.store.as_ref().unwrap());
    let rc2 = model.basis.r_cut * model.basis.r_cut;
    (0..s.accumulated.min(store.len()))
        .filter(|&i| !s.excluded(i) && meta[i].scene == scene)
        .flat_map(|i| {
            let x = &store.samples[i];
            super::super::trust::trained_keys(x, super::super::policy::env_keys(&x.pos, &x.z, rc2))
        })
        .flatten()
        .collect()
}

/// Task 7 review C1 as built (22553fd, e0f15ac; measurement only): per class key the provider's
/// own leave-one-out statistics (`compute_class_loo_full`'s key statistics: M in the RMS terms,
/// out of n_k and the max terms; `energy_misfits`, M's energy samples in the energy sum) and the
/// **standing verdict** of the class rule (`class_loo`), with C1's guard and the energy half
/// (`ClassKey::EnergyRms`) in it. Beside them, recomputed here for comparison: the RMS with M left
/// out (the rule before C1) and with M in (the cross-check of the provider's numbers), f and r_M
/// (M's share of the key's atoms and their LOO force ratio RMS), f_E and r_E (M's share of the
/// key's energy samples and their LOO energy ratio RMS).
fn c1_class_keys(s: &SurrogateKs, label: &str) {
    use std::collections::BTreeSet;
    #[derive(Default)]
    struct K {
        n: [usize; 2],
        f_ss: [f64; 2],
        f_at: [usize; 2],
        e_ss: [f64; 2],
        e_n: [usize; 2],
        undefined: [bool; 2],
        m_atoms: usize,
        m_ss: f64,
        m_e: usize,
        m_e_ss: f64,
    }
    let (model, store) = (s.model.as_ref().unwrap(), s.store.as_ref().unwrap());
    let rc2 = model.basis.r_cut * model.basis.r_cut;
    let fitted: Vec<usize> = (0..s.accumulated.min(store.len())).filter(|&i| !s.excluded(i)).collect();
    let refs: Vec<&surrogate::fit::Sample> = fitted.iter().map(|&i| &store.samples[i]).collect();
    let loos = surrogate::loo::leave_one_out_many(model, &refs);
    let mut acc: BTreeMap<surrogate::EnvKey, K> = BTreeMap::new();
    for (&i, l) in fitted.iter().zip(&loos) {
        let x = &store.samples[i];
        let in_m = s.in_misfit(i);
        let keys = if x.z.len() == x.pos.len() { super::super::trust::trained_keys(x, super::super::policy::env_keys(&x.pos, &x.z, rc2)) } else { Vec::new() };
        let n = x.pos.len().max(1) as f64;
        let e_ratio: Option<f64> = x
            .use_energy
            .then(|| l.as_ref().and_then(|l| l.energy).map_or(f64::INFINITY, |e| (if model.energy_per_atom { e.abs() } else { e.abs() / n }) / s.policy.energy_std_per_atom));
        let present: BTreeSet<&surrogate::EnvKey> = keys.iter().flatten().collect();
        for k in present {
            let a = acc.entry(k.clone()).or_default();
            if in_m {
                if let Some(e) = e_ratio {
                    a.m_e += 1;
                    a.m_e_ss += e * e;
                }
            }
            // w = 0: M left out (the rule before C1); w = 1: M kept in (the RMS terms as built)
            for w in [0usize, 1] {
                if w == 0 && in_m {
                    continue;
                }
                a.n[w] += 1;
                if let Some(e) = e_ratio {
                    a.e_ss[w] += e * e;
                    a.e_n[w] += 1;
                    if e == f64::INFINITY {
                        a.undefined[w] = true;
                    }
                }
                match l.as_ref() {
                    None => a.undefined[w] = true,
                    Some(l) => {
                        for ((r, f), kj) in l.forces.iter().zip(&x.forces).zip(&keys) {
                            let Some(r) = r.filter(|_| kj.as_ref() == Some(k)) else { continue };
                            let (d, b) = s.policy.force_residual(&[f[0] - r[0], f[1] - r[1], f[2] - r[2]], f);
                            let ratio = d / b;
                            a.f_ss[w] += ratio * ratio;
                            a.f_at[w] += 1;
                            if w == 1 && in_m {
                                a.m_atoms += 1;
                                a.m_ss += ratio * ratio;
                            }
                        }
                    }
                }
            }
        }
    }
    let (judged, _, _, _) = s.compute_class_loo_full();
    let provider = judged.map(|(_, j)| j.key_stats).unwrap_or_default();
    let m_n = s.misfit().map_or(0, |m| m.samples.len());
    println!("\nC1 [{label}] (|M| = {m_n}): per class key, the provider's statistics and standing verdict (C1 + the energy guard, as built)");
    println!("| key | n_k | provider R_c / R_E,c / energy M | standing verdict | recomputed R_c, R_E M in | M out (before C1) R_c, R_E | f | r_M | f_E | r_E |");
    let rms = |ss: f64, n: usize, und: bool| if und { f64::INFINITY } else if n > 0 { (ss / n as f64).sqrt() } else { 0.0 };
    for (k, a) in &acc {
        let rf = [rms(a.f_ss[0], a.f_at[0], a.undefined[0]), rms(a.f_ss[1], a.f_at[1], a.undefined[1])];
        let re = [rms(a.e_ss[0], a.e_n[0], a.undefined[0]), rms(a.e_ss[1], a.e_n[1], a.undefined[1])];
        let f = if a.f_at[1] > 0 { a.m_atoms as f64 / a.f_at[1] as f64 } else { 0.0 };
        let r_m = if a.m_atoms > 0 { (a.m_ss / a.m_atoms as f64).sqrt() } else { 0.0 };
        let f_e = if a.e_n[1] > 0 { a.m_e as f64 / a.e_n[1] as f64 } else { 0.0 };
        let r_e = if a.m_e > 0 { (a.m_e_ss / a.m_e as f64).sqrt() } else { 0.0 };
        let p = provider.get(k).map_or("— (not judged)".to_string(), |p| format!("{:.3} / {:.3} / {} (n {})", p.force_rms, p.energy_rms, p.energy_misfits, p.samples));
        let v = match key_verdict(s, k) {
            None => "not judged".to_string(),
            Some(None) => "trusted".to_string(),
            Some(Some((key, value, limit))) => format!("REFUSED {key:?} {value:.3} > {limit:.3}"),
        };
        println!("| {} | {} | {p} | {v} | {:.3}, {:.3} | {:.3}, {:.3} | {:.4} | {:.2} | {:.4} | {:.2} |", key_name(k), a.n[0], rf[1], re[1], rf[0], re[0], f, r_m, f_e, r_e);
    }
    for c in s.class_loo() {
        println!("class {}: {} samples, force rms {:.3}, energy rms {:.3}; {}", c.class, c.samples, c.force_rms, c.energy_rms, match c.refusal {
            None => "trusted".to_string(),
            Some((key, value, limit)) => format!("REFUSED {key:?} {value:.3} > {limit:.3}"),
        });
    }
}

/// Task 7 review C2 / Q3 (measurement only): the share of the fitted samples outside M, and of
/// their atoms, that the misfit fence would flag queried at the samples' own geometries, at the
/// shipped near distance (`QUARANTINE_FRACTION × max_descriptor_distance`) and at
/// `THIN_FRACTION × max_descriptor_distance`, by scene. Each atom's L∞ spread distance to the
/// nearest misfit atom of its species (`spread_distance`, as `near_misfit_atoms` takes it) is
/// computed here and thresholded at both radii — the product constant untouched; the shipped
/// radius's sample count is cross-checked against `near_misfit_atoms` itself.
fn c2_near_misfit(s: &mut SurrogateKs, meta: &[Meta], label: &str) {
    use super::super::quarantine::{spread_distance, QUARANTINE_FRACTION, THIN_FRACTION};
    s.ensure_desc_cache();
    let near = QUARANTINE_FRACTION * s.policy.max_descriptor_distance;
    let wide = THIN_FRACTION * s.policy.max_descriptor_distance;
    let m = s.misfit().cloned();
    let fence = s.misfit_fence().is_some();
    println!(
        "\nC2 [{label}]: |M| {} samples, {} atoms; the fence stands: {fence}; radii {near:.2} and {wide:.2} spreads",
        m.as_ref().map_or(0, |m| m.samples.len()),
        m.as_ref().map_or(0, |m| m.atoms.len())
    );
    let store = s.store.as_ref().unwrap();
    let env = s.envelope.as_ref().unwrap();
    let targets: Vec<(usize, &[f64])> = m
        .as_ref()
        .map(|m| m.atoms.iter().filter_map(|&(i, j)| s.desc_cache.get(i).and_then(|a| a.get(j)).map(|(sp, d)| (*sp, &d[..]))).collect())
        .unwrap_or_default();
    // per scene: samples, flagged at near, at wide, atoms, flagged at near, at wide, near_misfit_atoms's samples
    let mut by: BTreeMap<String, [usize; 7]> = BTreeMap::new();
    let fitted: Vec<usize> = (0..s.accumulated.min(store.len())).filter(|&i| !s.excluded(i) && !s.in_misfit(i)).collect();
    for &i in &fitted {
        let e = by.entry(meta[i].scene.clone()).or_default();
        e[0] += 1;
        let (mut sn, mut sw) = (false, false);
        for (sp, d) in &s.desc_cache[i] {
            let dmin = targets.iter().filter(|(t, _)| t == sp).map(|(_, td)| spread_distance(env, *sp, d, td)).fold(f64::INFINITY, f64::min);
            e[3] += 1;
            if dmin <= near {
                e[4] += 1;
                sn = true;
            }
            if dmin <= wide {
                e[5] += 1;
                sw = true;
            }
        }
        e[1] += sn as usize;
        e[2] += sw as usize;
    }
    if fence && !targets.is_empty() {
        for &i in &fitted {
            let x = &store.samples[i];
            if s.near_misfit_atoms(&x.pos, &x.z).iter().any(|&b| b) {
                by.get_mut(&meta[i].scene).unwrap()[6] += 1;
            }
        }
    }
    println!("| scene | samples | flagged at {near:.2} | at {wide:.2} | atoms | flagged at {near:.2} | at {wide:.2} | `near_misfit_atoms` samples (cross-check) |");
    let pct = |a: usize, b: usize| if b > 0 { 100.0 * a as f64 / b as f64 } else { 0.0 };
    let mut tot = [0usize; 7];
    for (k, e) in &by {
        for (t, v) in tot.iter_mut().zip(e) {
            *t += v;
        }
        println!("| {k} | {} | {} ({:.1} %) | {} ({:.1} %) | {} | {} ({:.1} %) | {} ({:.1} %) | {} |", e[0], e[1], pct(e[1], e[0]), e[2], pct(e[2], e[0]), e[3], e[4], pct(e[4], e[3]), e[5], pct(e[5], e[3]), e[6]);
    }
    let e = tot;
    println!("| all | {} | {} ({:.1} %) | {} ({:.1} %) | {} | {} ({:.1} %) | {} ({:.1} %) | {} |", e[0], e[1], pct(e[1], e[0]), e[2], pct(e[2], e[0]), e[3], e[4], pct(e[4], e[3]), e[5], pct(e[5], e[3]), e[6]);
}

/// Spec 2026-11-13 §5.1–5.2 (Task 8): the open-shell model on a copy of the user's collection (the
/// 181-hold copy of 2026-11-12: 1 805 samples, 180 held, 1 superseded).
///
/// `COLLECTION_DIR=<fresh copy> cargo test -p providers --release --lib acceptance_on_the_collection -- --ignored --nocapture --test-threads 1`
/// — a **fresh copy** every run: the refits write their codes to it, and the test copies it once
/// more (before anything writes) to `<copy's parent>_d4_2/` for the D₄ = 2 provider, refusing a
/// target that exists.
///
/// §5.1: the attach (the real growth rule) leaves the model trusted, holding no more than it
/// started with, with at least the 68 releases of 2026-11-12; M is printed by scene. Refits with
/// nothing new then converge (ruling 11: each better fit reproduces more held samples, and one
/// release pass runs per refit): the held count never rises, and two successive refits leave the
/// same codes and the same M. The user's decision 2026-11-13: no refit-count bound; 20 refits are
/// only the runtime bound (the measured fixed point is refit 8). §5.2: per scene, the force rms
/// and the leave-one-out force RMS at the converged D₄ = 3 fit are ≤ those of a D₄ = 2 provider's
/// attach fit on its own fresh copy, or both below 1e-4 Ha/Bohr — except HO's LOO, accepted by the
/// user (one sample whose keys the class rule refuses at both degrees; checked, not assumed). The
/// §5.2 is asserted twice: on each degree's own fitted set, and on the same samples (D₄ = 2
/// fitted on the D₄ = 3 codes, no recovery). The convergence and §5.2 failures are asserted
/// together at the end. Printed, not asserted: the Task 7 review's C1 (the class rule as built)
/// and C2 (rejected; kept as the record) on the attach and on a variant whose M is populated.
#[test]
#[ignore]
fn acceptance_on_the_collection() {
    use std::time::Instant;
    use surrogate::quarantine::{Q_HELD, Q_SUPERSEDED};
    const FLOOR: f64 = 1e-4;
    let dir = collection_dir();
    let dir2 = sibling_copy(&dir, "d4_2");
    println!("D₄ = 3 copy {}; D₄ = 2 copy {}", dir.display(), dir2.display());

    // ---- §5.1: the attach at the real growth rule
    let c = open_at(&dir);
    let meta = metas(&c);
    let q_disk = c.quarantine.clone();
    let held_before = held(&q_disk);
    println!(
        "collection: {} samples, {held_before} held, {} superseded on disk; held by scene {:?}",
        meta.len(),
        q_disk.iter().filter(|&&q| q == Q_SUPERSEDED).count(),
        by_scene(&meta, (0..q_disk.len()).filter(|&i| q_disk[i] == Q_HELD))
    );
    let mut s = SurrogateKs::new(Box::new(StubKsGpu), FallbackPolicy::default()).with_teacher_tol(1e-6);
    let t = Instant::now();
    s.attach(c).unwrap();
    if s.report().is_none() {
        SurrogateKs::refit(&mut s).unwrap();
    }
    let attach_wall = t.elapsed().as_secs_f64();
    report(&s, &meta, "D₄ growth rule: after the attach's refit (recovery included)");
    let (released, newly) = released_and_new(&q_disk, &s.quarantine);
    let held_after = s.quarantined();
    let m0 = misfit_samples(&s);
    println!(
        "\nATTACH: body4_degree {:?}, Q′ {}, {} fit_store calls, {attach_wall:.2} s; trusted {} ({:?}); held {held_before} → {held_after} ({released} released, {newly} newly held); recovery {:?}",
        s.body4_degree(),
        s.report().unwrap().q_prime,
        s.fits,
        s.trusted(),
        s.untrusted_reason(),
        s.recovery_note()
    );
    println!(
        "ATTACH M: {} samples, {} atoms, {} fitted counted; by scene {:?}; outside {:?}",
        m0.len(),
        s.misfit().map_or(0, |m| m.atoms.len()),
        s.misfit().map_or(0, |m| m.fitted),
        by_scene(&meta, m0.iter().copied()),
        s.misfit().map(|m| m.outside)
    );
    println!("ATTACH held by scene: {:?}", by_scene(&meta, (0..s.quarantine.len()).filter(|&i| s.quarantine[i] == Q_HELD)));
    assert_eq!(s.body4_degree(), Some(3), "the growth rule takes the collection to D₄ = 3");
    assert!(s.trusted(), "§5.1: the attach leaves the model trusted: {:?}", s.untrusted_reason());
    assert!(held_after <= held_before, "§5.1: the attach holds no more than it started with: {held_before} → {held_after}");
    assert!(released >= 68, "§5.1: at least the 68 releases of 2026-11-12 recur: {released}");
    assert!(s.fits > 0, "the attach fitted: {} fit_store calls", s.fits);
    // the Task 7 review's C1 / C2 measurements on the attach's refit (printed, not asserted)
    c1_class_keys(&s, "D₄ = 3, the attach's refit");
    c2_near_misfit(&mut s, &meta, "D₄ = 3, the attach's refit");

    // ---- §5.1 (ruling 11, the user's decision 2026-11-13): refits with nothing new reach a fixed
    // point — no refit-count bound; `MAX_REFITS` only keeps the test from looping forever. The
    // convergence failures are asserted at the end, after §5.2 has been measured too
    const MAX_REFITS: usize = 20;
    let mut failures: Vec<String> = Vec::new();
    println!("\n| refit | held | released | newly held | |M| | M by scene | trusted | fit_store calls | wall (s) |");
    println!("| attach | {held_after} | {released} | {newly} | {} | {:?} | {} | {} | {attach_wall:.2} |", m0.len(), by_scene(&meta, m0.iter().copied()), s.trusted(), s.fits);
    let (mut prev_q, mut prev_m) = (s.quarantine.clone(), m0);
    let mut fixed_at = None;
    for k in 1..=MAX_REFITS {
        let fits = s.fits;
        s.pending_refit = true;
        let t = Instant::now();
        SurrogateKs::refit(&mut s).unwrap();
        let wall = t.elapsed().as_secs_f64();
        let (rel, new) = released_and_new(&prev_q, &s.quarantine);
        let m = misfit_samples(&s);
        println!(
            "| {k} | {} | {rel} | {new} | {} | {:?} | {} | {} | {wall:.2} |",
            s.quarantined(),
            m.len(),
            by_scene(&meta, m.iter().copied()),
            s.trusted(),
            s.fits - fits
        );
        println!("   refit {k}: {:?}; recovery {:?}", s.untrusted_reason(), s.recovery_note());
        // a refit that did not fit would repeat the codes and M vacuously
        if s.fits <= fits {
            failures.push(format!("refit {k} ran no fit_store ({} → {}): the fixed point would be vacuous", fits, s.fits));
        }
        if s.quarantined() > held(&prev_q) {
            failures.push(format!("ruling 11: the held count rose at refit {k}: {} → {}", held(&prev_q), s.quarantined()));
        }
        let same = s.quarantine == prev_q && m == prev_m;
        prev_q = s.quarantine.clone();
        prev_m = m;
        if same {
            fixed_at = Some(k);
            break;
        }
    }
    match fixed_at {
        Some(k) => println!("FIXED POINT: refit {k} repeats refit {} (codes and M)", k - 1),
        None => println!("NO FIXED POINT within {MAX_REFITS} refits"),
    }
    if fixed_at.is_none() {
        failures.push(format!("ruling 11: no fixed point (the same codes and M at two successive refits) within {MAX_REFITS} refits (the runtime bound)"));
    }
    report(&s, &meta, &format!("D₄ = 3 at the last refit (fixed point {fixed_at:?})"));
    let rms3 = scene_rms(&s, &meta);

    // ---- §5.2: a D₄ = 2 provider on its own fresh copy
    let c2 = open_at(&dir2);
    let mut s2 = SurrogateKs::new(Box::new(StubKsGpu), FallbackPolicy::default()).with_teacher_tol(1e-6);
    s2.body4_force = Some(2);
    let t = Instant::now();
    s2.attach(c2).unwrap();
    if s2.report().is_none() {
        SurrogateKs::refit(&mut s2).unwrap();
    }
    let wall2 = t.elapsed().as_secs_f64();
    report(&s2, &meta, "D₄ = 2 (forced): after the attach's refit");
    let m2 = misfit_samples(&s2);
    println!(
        "D₄ = 2 ATTACH: body4_degree {:?}, Q′ {}, {} fit_store calls, {wall2:.2} s; trusted {} ({:?}); held {held_before} → {}; |M| {} by scene {:?}; recovery {:?}",
        s2.body4_degree(),
        s2.report().unwrap().q_prime,
        s2.fits,
        s2.trusted(),
        s2.untrusted_reason(),
        s2.quarantined(),
        m2.len(),
        by_scene(&meta, m2.iter().copied()),
        s2.recovery_note()
    );
    let rms2 = scene_rms(&s2, &meta);
    // The one accepted exception (the user's decision 2026-11-13; references.md "Acceptance
    // (2026-11-14)"): HO's LOO force RMS. HO is one sample (#1804, O–H 1.311 Bohr, the store's only
    // pair-class O–H geometry), so its LOO is the pair term's extrapolation to a distance no other
    // sample covers. The exception holds only while that stays true: one fitted sample at both
    // degrees, and every key of its atoms with n_k = 1 (`key_stats`) and refused by the class rule,
    // at both. Its in-sample force rms is still asserted; every other scene is asserted in full
    let lone_and_refused = |p: &SurrogateKs| {
        let keys = scene_keys(p, &meta, "HO");
        let stats = p.compute_class_loo_full().0.map(|(_, j)| j.key_stats).unwrap_or_default();
        !keys.is_empty() && keys.iter().all(|key| stats.get(key).is_some_and(|st| st.samples == 1) && matches!(key_verdict(p, key), Some(Some(_))))
    };
    let ho_exception = rms3.get("HO").is_some_and(|r| r.samples == 1) && rms2.get("HO").is_some_and(|r| r.samples == 1) && lone_and_refused(&s) && lone_and_refused(&s2);
    let loo_exception = |k: &str| k == "HO" && ho_exception;
    if rms3.contains_key("HO") {
        let v = |p: &SurrogateKs| {
            let stats = p.compute_class_loo_full().0.map(|(_, j)| j.key_stats).unwrap_or_default();
            scene_keys(p, &meta, "HO").iter().map(|key| format!("{} n_k {:?} {:?}", key_name(key), stats.get(key).map(|st| st.samples), key_verdict(p, key))).collect::<Vec<_>>()
        };
        println!("\nHO keys: D₄ = 3 {:?}; D₄ = 2 {:?}; the LOO exception applies: {ho_exception}", v(&s), v(&s2));
    }
    println!("\n§5.2 | scene | samples D₄3/D₄2 | atoms D₄3/D₄2 | rms D₄=3 | rms D₄=2 | LOO D₄=3 | LOO D₄=2 | LOO undefined 3/2 | verdict |");
    let ok = |a: f64, b: f64| a <= b || (a < FLOOR && b < FLOOR);
    for (k, r3) in &rms3 {
        let r2 = rms2.get(k).unwrap_or_else(|| panic!("scene {k} has no fitted sample at D₄ = 2"));
        let (ok_rms, ok_loo) = (ok(r3.rms, r2.rms), ok(r3.loo, r2.loo));
        println!(
            "| {k} | {}/{} | {}/{} | {:.3e} | {:.3e} | {:.3e} | {:.3e} | {}/{} | {} |",
            r3.samples,
            r2.samples,
            r3.atoms,
            r2.atoms,
            r3.rms,
            r2.rms,
            r3.loo,
            r2.loo,
            r3.loo_undefined,
            r2.loo_undefined,
            if ok_rms && ok_loo {
                "ok"
            } else if ok_rms && loo_exception(k) {
                "LOO ROSE: EXCEPTION (accepted by the user 2026-11-13: one sample, its keys refused at both degrees)"
            } else {
                "ROSE"
            }
        );
        if !ok_rms {
            failures.push(format!("{k}: force rms {:.3e} at D₄ = 3 > {:.3e} at D₄ = 2", r3.rms, r2.rms));
        }
        if !ok_loo && !loo_exception(k) {
            failures.push(format!("{k}: LOO force RMS {:.3e} at D₄ = 3 > {:.3e} at D₄ = 2", r3.loo, r2.loo));
        }
    }
    for k in rms2.keys().filter(|k| !rms3.contains_key(*k)) {
        failures.push(format!("{k}: fitted at D₄ = 2, no fitted sample at D₄ = 3"));
    }

    // §5.2 on the same samples (spec §5 item 2, "same-sample comparison"): D₄ = 2 fitted on the
    // D₄ = 3 fixed point's codes, no recovery
    s2.quarantine = s.quarantine.clone();
    s2.species_dirty = true;
    let t = Instant::now();
    s2.fit_store().unwrap();
    println!("\nD₄ = 2 on the D₄ = 3 codes (fit_store, no recovery, {:.2} s): training residuals {:?}", t.elapsed().as_secs_f64(), s2.training_residuals());
    let same = scene_rms(&s2, &meta);
    println!("same samples | scene | samples 3/2 | rms D₄=3 | rms D₄=2 | LOO D₄=3 | LOO D₄=2 | verdict |");
    // Asserted with the same condition and the same HO exception as the own-set comparison above.
    // A weak comparator: this D₄ = 2 fit is untrusted and unrecovered (it fits the samples D₄ = 2's
    // own recovery holds; max force residual 6.4e-3 Ha/Bohr, measured 2026-11-14), so passing it
    // says less than the own-set assertion does
    for (k, r3) in &rms3 {
        let Some(r2) = same.get(k) else {
            failures.push(format!("same samples, {k}: no fitted sample at D₄ = 2 on the D₄ = 3 codes"));
            continue;
        };
        let (ok_rms, ok_loo) = (ok(r3.rms, r2.rms), ok(r3.loo, r2.loo));
        let v = if ok_rms && ok_loo {
            "ok"
        } else if ok_rms && loo_exception(k) {
            "LOO ROSE: EXCEPTION (as above)"
        } else {
            "ROSE"
        };
        println!("| {k} | {}/{} | {:.3e} | {:.3e} | {:.3e} | {:.3e} | {v} |", r3.samples, r2.samples, r3.rms, r2.rms, r3.loo, r2.loo);
        if !ok_rms {
            failures.push(format!("same samples, {k}: force rms {:.3e} at D₄ = 3 > {:.3e} at D₄ = 2", r3.rms, r2.rms));
        }
        if !ok_loo && !loo_exception(k) {
            failures.push(format!("same samples, {k}: LOO force RMS {:.3e} at D₄ = 3 > {:.3e} at D₄ = 2", r3.loo, r2.loo));
        }
    }
    // C1 / C2 again on a test-only variant whose M is populated (the attach's M is empty on
    // this collection): D₄ = 3, every held sample back in the fit, no recovery (`fit_store`)
    for c in s.quarantine.iter_mut() {
        if *c == Q_HELD {
            *c = surrogate::quarantine::Q_NONE;
        }
    }
    s.species_dirty = true;
    let t = Instant::now();
    s.fit_store().unwrap();
    let mv = misfit_samples(&s);
    println!(
        "\nVARIANT (every hold cleared, no recovery, fit {:.2} s): body4_degree {:?}; trusted {} ({:?}); held {}; |M| {} samples, {} atoms, limit base {}; M by scene {:?}; training residuals outside M {:?}",
        t.elapsed().as_secs_f64(),
        s.body4_degree(),
        s.trusted(),
        s.untrusted_reason(),
        s.quarantined(),
        mv.len(),
        s.misfit().map_or(0, |m| m.atoms.len()),
        s.misfit().map_or(0, |m| m.fitted),
        by_scene(&meta, mv.iter().copied()),
        s.misfit().map(|m| m.outside)
    );
    c1_class_keys(&s, "variant: every hold cleared, no recovery");
    c2_near_misfit(&mut s, &meta, "variant: every hold cleared, no recovery");
    assert!(failures.is_empty(), "§5.1 convergence / §5.2: {failures:#?}");
}
