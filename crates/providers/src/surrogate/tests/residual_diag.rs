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

fn open_copy() -> surrogate::Collection {
    let dir = std::path::PathBuf::from(std::env::var("COLLECTION_DIR").expect("COLLECTION_DIR names a collection folder copy"));
    let key = surrogate::SurfaceKey::from_text(&std::fs::read_to_string(dir.join("key.txt")).unwrap()).unwrap();
    surrogate::Collection::open(dir.parent().unwrap(), &key).unwrap().0
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
    let meta: Vec<Meta> = c
        .samples
        .iter()
        .zip(c.quarantine.iter().copied().chain(std::iter::repeat(0)))
        .map(|(x, q0)| Meta { scene: x.scene.clone(), grid: x.grid_n, tag: x.sample.tag, n: x.sample.pos.len(), q0, mult: x.multiplicity })
        .collect();
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
