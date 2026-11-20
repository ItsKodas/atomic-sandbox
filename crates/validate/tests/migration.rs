//! The Fermi–Dirac collection migration on real data (spec `docs/superpowers/specs/2026-11-13-fermi-smearing-design.md`
//! §9 V11, §7.3–7.6; plan 2026-11-13 Task 17). A **measurement**, not a gate: `#[ignore]`d, GPU, run alone and never
//! on the original collection. **Slow:** 37 s per sample on average at the Accurate grid (V11, 2026-11-20), about 17 h
//! for the user's 1 657-sample collection — the run of record was stopped at 480 by the user's decision (the collection
//! is not kept), so its review, refit and trust comparison (steps 1, 4 below) have not been measured. Numbers of record: `crates/validate/references.md`, "Fermi–Dirac
//! migration" (V11). Atomic units throughout (Hartree, Bohr, Ha/Bohr).
//!
//! ```text
//! FD_DRY_RUN_COPY=<a copy of a pre-Fermi–Dirac collection folder> \
//!   cargo test -p validate --release --test migration dry_run -- --ignored --nocapture
//! ```
//!
//! # `dry_run_on_collection_copy`
//!
//! `FD_DRY_RUN_COPY` names a **copy** of a collection folder taught by the previous solver (the integer / shared
//! occupation rules; its `key.txt` carries the legacy ionic tag), e.g. the user's BEST collection copied without its
//! `.lock`. The test refuses a folder that sits in a `models/collections` folder (a live collections root: copy it
//! out first) and a root (the copy's parent) that already holds another folder, unless `FD_DRY_RUN_RESUME=1` (it then
//! resumes the pass from its `migration.txt`, as the app would).
//!
//! 1. **Baseline (the pre-migration verdict, this build's trust rule).** The old labels, their codes and the old
//!    references are written into a collection of the new key under `<root>-baseline`, attached by a fresh provider
//!    and refitted: the trust verdict, the checklist and the per-class verdicts this build gives the collection as it
//!    was. Like the migrated folder it has no audit history (`history.bin` is not carried, spec §7.5), so the two
//!    verdicts differ by the labels only. (The old build's own verdict also read its 434-point audit history; that one
//!    is not reproducible here and is quoted from the folder's `README.txt` instead.)
//! 2. **The migration**, through the provider exactly as the app runs it: `attach_collection` finds the predecessor
//!    (`predecessor_of`, the exact key under the legacy tag), `start_migration`, then `pass_step` until the pass ends;
//!    its last step refits, and that refit runs the code-9 review (spec §7.6). The teacher is the GPU Kohn–Sham at the
//!    sample tolerance 1e-6 behind a recorder that keeps every solve (F, E, TS, forces, the stop tolerance, wall time).
//! 3. **Re-derived per sample** from the recorder: the class by the engine's own `classify` with `migration_eps`, which
//!    must reproduce the log's counts; the code each sample was appended with, which must be
//!    `migrated_code(old code, class)` (U also bit for bit the old record) — **no sample changes code without a
//!    logged reason** (asserted: a class the log counts, or a not-migrated line with its reason); class U's largest
//!    |ΔE| and |ΔF| (the log's, re-derived); the time per sample; the **false-S estimate** (spec §7.4 [SHOULD-7]):
//!    the fraction of class S whose only failing test is the energy, by at most 3 ε_E — every force within ε_F and
//!    min(|F − E_old|, |E − E_old|) ∈ (ε_E, 3 ε_E].
//! 4. **The review and the verdict after:** the review log (released / kept / waiting), the code changes the refit
//!    made after the appends (each named with its source: the review log or the trust recovery's notes), the trust
//!    verdict and checklist against the baseline's. Reported, not gated: a migrated collection that is not trusted as
//!    before is investigated (spec §9 V11).
//! 5. **The source copy is byte-identical afterwards** (asserted: every file, before and after).
//!
//! A per-sample table is written beside the root (`<root>-v11-samples.csv`) for the record's analysis.
//!
//! # No `remake_shipped_models`
//!
//! Spec §7.9 and the plan's Task 17 had a second test here that re-made the four shipped `.model` files with the same
//! engine. The user's ruling of 2026-11-16 (plan ledger, "Task 17 drops the re-make") removed those models from the
//! repository instead ("Delete all old models that won't be compatible, I will retrain them all myself"), so no
//! re-make exists and none is needed: a pre-Fermi–Dirac `.model` is refused on load with the reason
//! (`providers::surrogate::legacy_model_refusal`).
use providers::sampling::migrate::{classify, migrated_code, migration_eps, MigrationClass};
use providers::{FallbackPolicy, SurrogateKs};
use sim::{ForceProvider, ForceResult, IonicPolicy, Nucleus, ProviderError, Scene, Trainable};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use surrogate::quarantine::Q_NONE;
use surrogate::{Collection, CollectionSample};

mod common;
use common::{ms, Locked};

/// The teacher's sample tolerance: the key's (`tol 1e-6`) and the app's `SAMPLE_TOL`.
const SAMPLE_TOL: f64 = 1e-6;
/// Consecutive transient "busy" answers (`pass_step`'s `Cancelled`) tolerated before the run gives up.
const BUSY_RETRIES: usize = 50;

/// One teacher solve, as the recorder kept it.
#[derive(Clone)]
struct Rec {
    z: Vec<u8>,
    pos: Vec<[f64; 3]>,
    n: usize,
    free: f64,
    internal: f64,
    ts: f64,
    forces: Vec<[f64; 3]>,
    tol_used: Option<f64>,
    scf_steps: usize,
    wall: Duration,
    partial: usize,
    warnings: Vec<String>,
}

/// The locked GPU teacher behind a recorder: every successful solve is kept in order.
struct Recording {
    inner: Locked,
    recs: Arc<Mutex<Vec<Rec>>>,
}

impl ForceProvider for Recording {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        let t = Instant::now();
        let r = self.inner.evaluate(scene, should_abort);
        if let Ok(r) = &r {
            self.recs.lock().unwrap().push(Rec {
                z: scene.nuclei.iter().map(|n| n.z).collect(),
                pos: scene.nuclei.iter().map(|n| n.pos).collect(),
                n: scene.cell.n,
                free: r.energy,
                internal: r.stats.energy.internal,
                ts: r.stats.energy.ts,
                forces: r.forces.clone(),
                tol_used: r.stats.scf_tol_used,
                scf_steps: r.stats.scf_steps,
                wall: t.elapsed(),
                partial: r.stats.partial.len(),
                warnings: r.stats.warnings.clone(),
            });
        }
        r
    }
    fn name(&self) -> &'static str { self.inner.name() }
    fn reset(&mut self) { self.inner.reset() }
    fn discontinuity(&mut self) { self.inner.discontinuity() }
    fn sample_density(&mut self, points: &[[f64; 3]]) -> Option<Vec<f64>> { self.inner.sample_density(points) }
    fn scf_tol(&self) -> Option<f64> { self.inner.scf_tol() }
    fn set_orbital_view(&mut self, request: sim::OrbitalViewRequest) { self.inner.set_orbital_view(request) }
}

/// Every file of `dir` (not recursive), name and bytes, sorted by name.
fn files(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut v: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .map(|e| (e.file_name().to_string_lossy().into_owned(), std::fs::read(e.path()).unwrap()))
        .collect();
    v.sort();
    v
}

/// FNV-1a 64 of `b`: a short print of a file for the log (the comparison itself is on the bytes).
fn fnv(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x0000_0100_0000_01b3))
}

/// The scene of a stored sample on the key's grid spacing, under `policy`, multiplicity auto (as the migration builds it).
fn scene_of(cs: &CollectionSample, h: f64, policy: IonicPolicy) -> Scene {
    Scene {
        nuclei: cs.sample.z.iter().zip(&cs.sample.pos).map(|(&z, &p)| Nucleus::new(z, p)).collect(),
        cell: sim::Cell { n: cs.grid_n, h },
        charge: 0,
        multiplicity: None,
        sigma: None,
        external_field: [0.0; 3],
        ionic: policy,
        train_mask: None,
    }
}

fn max_force_diff(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    if a.len() != b.len() {
        return f64::NAN;
    }
    a.iter().zip(b).flat_map(|(p, q)| (0..3).map(move |k| (p[k] - q[k]).abs())).fold(0.0f64, f64::max)
}

fn quantiles(mut v: Vec<f64>) -> String {
    if v.is_empty() {
        return "—".into();
    }
    v.sort_by(f64::total_cmp);
    let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
    format!("min {:.3e}, median {:.3e}, p90 {:.3e}, max {:.3e} (n {})", q(0.0), q(0.5), q(0.9), q(1.0), v.len())
}

/// The trust verdict and its checklist, one line each, and the per-class verdicts.
fn verdict_lines(tag: &str, s: &SurrogateKs) -> String {
    let mut out = format!("{tag}: trusted {}{}\n", s.trusted(), s.untrusted_reason().map(|r| format!(" — {r}")).unwrap_or_default());
    for c in s.trust_checks() {
        let _ = writeln!(out, "  {:?}: {} (value {:.4e}, limit {:.4e})", c.key, if c.met { "met" } else { "NOT met" }, c.value, c.limit);
    }
    let _ = writeln!(out, "  classes: {}", common::surrogate::class_summary(s));
    let _ = writeln!(out, "  residuals: {}", common::surrogate::residuals(s.training_residuals()));
    out
}

/// The provider's notes, printed with a prefix.
fn drain_notes(tag: &str, s: &mut SurrogateKs, sink: &mut Vec<String>) {
    for n in Trainable::take_notes(s) {
        println!("  [{tag}] {n}");
        sink.push(n);
    }
}

/// Spec §9 V11 (module doc): the migration of a copy of a pre-Fermi–Dirac collection, measured.
#[test]
#[ignore = "V11: a measurement on a copy of a real collection, GPU, hours (37 s/sample at Accurate); FD_DRY_RUN_COPY=<copy> (module doc)"]
fn dry_run_on_collection_copy() {
    let Some(copy) = std::env::var_os("FD_DRY_RUN_COPY").map(PathBuf::from) else {
        panic!("FD_DRY_RUN_COPY is not set: name a copy of a pre-Fermi–Dirac collection folder (module doc)");
    };
    let copy = copy.canonicalize().unwrap_or_else(|e| panic!("{}: {e}", copy.display()));
    let root = copy.parent().expect("the copy sits in a folder").to_path_buf();
    // never the original: a folder inside a live collections root is refused
    let name = |p: Option<&Path>| p.and_then(|p| p.file_name()).map(|f| f.to_string_lossy().to_lowercase());
    assert!(
        !(name(Some(&root)).as_deref() == Some("collections") && name(root.parent()).as_deref() == Some("models")),
        "{} is in a models/collections folder: V11 runs on a copy, never on the original (module doc)",
        copy.display()
    );
    assert!(!copy.join(".lock").exists(), "{}: the copy holds a .lock — copy the folder without it", copy.display());
    let resume = std::env::var("FD_DRY_RUN_RESUME").is_ok_and(|v| v == "1");
    let others: Vec<String> = std::fs::read_dir(&root).unwrap().filter_map(|e| e.ok()).filter(|e| e.path().is_dir() && e.path() != copy).map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    assert!(resume || others.is_empty(), "{} holds other folders {others:?} (a previous run?): remove them, or set FD_DRY_RUN_RESUME=1", root.display());
    let before = files(&copy);
    println!("V11 source copy {}: {} files", copy.display(), before.len());
    for (f, b) in &before {
        println!("  {f}: {} bytes, fnv {:016x}", b.len(), fnv(b));
    }
    let Some(ctx) = common::gpu("migration") else { return };
    let t0 = Instant::now();
    let old = Collection::open_read_only(&copy).unwrap_or_else(|e| panic!("{}: {e}", copy.display()));
    let (h, n_old) = (old.key.h, old.samples.len());
    println!("V11 source key: {}", old.key.to_text().replace('\n', " | "));
    let mut by_code: BTreeMap<u8, usize> = BTreeMap::new();
    for &c in &old.codes {
        *by_code.entry(c).or_default() += 1;
    }
    let mut by_scene: BTreeMap<String, usize> = BTreeMap::new();
    for s in &old.samples {
        *by_scene.entry(s.scene.clone()).or_default() += 1;
    }
    println!("V11 source: {n_old} samples, codes {by_code:?}, scenes {by_scene:?}, {} references, damaged tail {:?}", old.references.len(), old.damaged_tail);
    // the surface of the copy: the legacy tag of the app's default policy (Accepted) — checked by the predecessor match
    let policy = IonicPolicy::Accepted;
    assert_eq!(old.key.ionic, providers::pseudo::legacy_tag_of(policy), "the copy is the previous solver's Accepted surface");
    let attach_scene = scene_of(&old.samples[0], h, policy);
    let mut notes = Vec::new();

    // ---- 1. the baseline: the old labels under this build's trust rule ----
    let new_key = {
        let (teacher, _) = Locked::new(ctx.clone(), SAMPLE_TOL);
        let probe = SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_teacher_tol(SAMPLE_TOL);
        probe.provenance_for(&attach_scene).surface_key()
    };
    assert_ne!(new_key.ionic, old.key.ionic, "the new surface's tag differs from the legacy one");
    let base_root = root.with_file_name(format!("{}-baseline", root.file_name().unwrap().to_string_lossy()));
    let _ = std::fs::remove_dir_all(&base_root);
    std::fs::create_dir_all(&base_root).unwrap();
    {
        let (mut c, _) = Collection::open(&base_root, &new_key).unwrap();
        for (k, (cs, &code)) in old.samples.iter().zip(&old.codes).enumerate() {
            c.append(cs.clone()).unwrap_or_else(|e| panic!("baseline: sample {k}: {e}"));
            if code != Q_NONE {
                c.set_status(k, code).unwrap_or_else(|e| panic!("baseline: sample {k}'s code {code}: {e}"));
            }
        }
        for (z, _, r) in &old.references {
            let out = c.offer_reference(*z, r.clone(), f64::INFINITY).unwrap();
            println!("  baseline reference Z {z} {:?}³: {out:?}", r.grid_n);
        }
    }
    let tb = Instant::now();
    let base_verdict = {
        let (teacher, _) = Locked::new(ctx.clone(), SAMPLE_TOL);
        let mut b = SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_teacher_tol(SAMPLE_TOL);
        b.attach_collection(&base_root, &attach_scene).unwrap();
        drain_notes("baseline", &mut b, &mut Vec::new());
        match b.refit() {
            Ok(r) => println!("V11 baseline refit: {} samples, rmse {:.3e} Ha/atom, {:.3e} Ha/Bohr ({:.0} s)", r.n_samples, r.rmse_energy_per_atom, r.rmse_force, tb.elapsed().as_secs_f64()),
            Err(e) => println!("V11 baseline refit failed: {e}"),
        }
        drain_notes("baseline", &mut b, &mut Vec::new());
        let v = verdict_lines("V11 baseline (old labels, this build, no audit history)", &b);
        print!("{v}");
        (b.trusted(), v)
    };

    // ---- 2. the migration, as the app runs it ----
    let recs = Arc::new(Mutex::new(Vec::<Rec>::new()));
    let (inner, _log) = Locked::new(ctx.clone(), SAMPLE_TOL);
    let teacher = Recording { inner, recs: recs.clone() };
    let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_teacher_tol(SAMPLE_TOL);
    s.attach_collection(&root, &attach_scene).unwrap();
    drain_notes("attach", &mut s, &mut notes);
    let new_dir = s.collection().unwrap().folder().to_path_buf();
    assert_eq!(s.collection().unwrap().key, new_key);
    let offer = s.migration().expect("the copy is the attached surface's predecessor");
    println!("V11 offer: {:?} of {} samples, estimate {:?} s, species {:?}", offer.state, offer.samples, offer.eta_s, offer.species);
    assert!(resume || offer.state == sim::MigrationPhase::Offered, "{offer:?}");
    assert_eq!(offer.samples, n_old);
    s.start_migration().unwrap();
    drain_notes("start", &mut s, &mut notes);
    // per step: the cursor's move, the solves it made, and the code a sample was appended with
    let mut appended: BTreeMap<usize, (usize, u8)> = BTreeMap::new(); // source index → (new index, code at append)
    let mut solved: BTreeMap<usize, Rec> = BTreeMap::new(); // source index → its cold solve
    let (mut steps, mut busy) = (0usize, 0usize);
    let tp = Instant::now();
    loop {
        let c0 = s.migration().map_or(0, |m| m.done);
        let (r0, len0) = (recs.lock().unwrap().len(), s.collection().map_or(0, |c| c.samples.len()));
        match Trainable::pass_step(&mut s, &mut common::never()) {
            Ok(Some(_)) => busy = 0,
            Ok(None) => break,
            Err(ProviderError::Cancelled) => {
                busy += 1;
                drain_notes("pass", &mut s, &mut notes);
                assert!(busy <= BUSY_RETRIES, "the GPU stayed busy for {busy} steps");
                std::thread::sleep(Duration::from_millis(500));
                continue;
            }
            Err(e) => panic!("re-solve step {steps}: {e}"),
        }
        steps += 1;
        drain_notes("pass", &mut s, &mut notes);
        let c1 = s.migration().map_or(c0, |m| m.done);
        let r = recs.lock().unwrap();
        if c1 == c0 + 1 {
            let new: Vec<&Rec> = r[r0..].iter().collect();
            assert!(new.len() <= 1, "sample {c0}: {} solves in one step", new.len());
            if let Some(rec) = new.first() {
                let o = &old.samples[c0].sample;
                assert!(rec.z == o.z && rec.pos.iter().zip(&o.pos).all(|(a, b)| (0..3).all(|k| a[k].to_bits() == b[k].to_bits())), "sample {c0}: the solve is of another geometry");
                solved.insert(c0, (*rec).clone());
            }
            let c = s.collection().unwrap();
            if c.samples.len() == len0 + 1 {
                appended.insert(c0, (len0, c.quarantine[len0]));
            }
        } else {
            assert!(c1 == c0, "the cursor moved {c0} → {c1} in one step");
        }
        if c1 > c0 && c1 % 100 == 0 {
            let m = s.migration().unwrap();
            println!("V11 progress: {c1} of {n_old}, counts U/T/S/not {:?}, {:.0} s elapsed, eta {:?} s", m.counts, tp.elapsed().as_secs_f64(), m.eta_s.map(|x| x.round()));
        }
    }
    drain_notes("finish", &mut s, &mut notes);
    let pass_wall = tp.elapsed();
    let mlog = surrogate::read_migration(&new_dir).unwrap().expect("migration.txt");
    println!("V11 pass: {steps} steps in {:.1} min; state {:?}; counts U/T/S/not {:?}; class-U max |ΔE| {:.3e} Ha, |ΔF| {:.3e} Ha/Bohr", pass_wall.as_secs_f64() / 60.0, mlog.state, mlog.counts, mlog.class_u_max.0, mlog.class_u_max.1);
    for (z, n, o, nw) in &mlog.references {
        println!("  reference Z {z} {n}³: old {o:?} new {nw:?} (Δ {:?})", o.zip(*nw).map(|(a, b)| b - a));
    }
    for (i, code, why) in &mlog.not_migrated {
        println!("  not migrated: source {i} ({}): {code:?} — {why}", old.samples[*i].scene);
    }
    assert_eq!(mlog.cursor, n_old, "every source sample was handled");

    // ---- 3. re-derived per sample ----
    let all_electron = providers::pseudo::is_all_electron_tag(&new_key.ionic);
    let not: BTreeMap<usize, &String> = mlog.not_migrated.iter().map(|(i, _, w)| (*i, w)).collect();
    let mut counts = [0usize; 4];
    let (mut false_s, mut s_force_ok, mut s_force_fail) = (0usize, 0usize, 0usize);
    let mut d_u = Vec::new();
    let (mut d_t, mut d_s, mut ts_all, mut walls) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut walls_by_n: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    let mut by_scene_class: BTreeMap<String, [usize; 4]> = BTreeMap::new();
    let (mut fd_labelled_u, mut partial_u) = (0usize, 0usize);
    let mut unexplained = Vec::new();
    let mut csv = String::from("source,scene,grid_n,old_code,class,new_index,new_code,e_old,f_new,e_new,ts,d_free,d_internal,max_df,eps_e,eps_f,tol_used,partial,scf_steps,wall_ms\n");
    for (i, cs) in old.samples.iter().enumerate() {
        let code_old = old.codes[i];
        let sc = by_scene_class.entry(cs.scene.clone()).or_default();
        if let Some(why) = not.get(&i) {
            counts[3] += 1;
            sc[3] += 1;
            assert!(!why.is_empty(), "source {i}: not migrated without a reason");
            assert!(!appended.contains_key(&i), "source {i}: not migrated, yet appended");
            let _ = writeln!(csv, "{i},{},{},{code_old},X,,,{:.12},,,,,,,,,,,,", cs.scene, cs.grid_n, cs.sample.energy);
            continue;
        }
        let Some(rec) = solved.get(&i) else {
            unexplained.push(format!("source {i}: migrated without a recorded solve (a resume?)"));
            continue;
        };
        let eps = migration_eps(all_electron, rec.tol_used.unwrap_or(SAMPLE_TOL), cs.sample.energy);
        let class = classify(cs.sample.energy, &cs.sample.forces, rec.free, rec.internal, &rec.forces, eps);
        let (d_free, d_int, dff) = (rec.free - cs.sample.energy, rec.internal - cs.sample.energy, max_force_diff(&cs.sample.forces, &rec.forces));
        counts[class.index()] += 1;
        sc[class.index()] += 1;
        walls.push(rec.wall.as_secs_f64());
        walls_by_n.entry(rec.n).or_default().push(rec.wall.as_secs_f64());
        ts_all.push(rec.ts);
        match class {
            MigrationClass::Unchanged => {
                d_u.push(d_free.abs());
                if rec.ts > eps.e {
                    fd_labelled_u += 1; // an old label that already was F (a Fermi–Dirac preview's answer)
                }
                if rec.partial > 0 {
                    partial_u += 1;
                }
            }
            MigrationClass::EntropyOnly => d_t.push(d_free),
            MigrationClass::StateChanged => {
                d_s.push(d_free);
                if dff <= eps.f {
                    s_force_ok += 1;
                    let d = d_free.abs().min(d_int.abs());
                    if d > eps.e && d <= 3.0 * eps.e {
                        false_s += 1;
                    }
                } else {
                    s_force_fail += 1;
                }
            }
        }
        let want = migrated_code(code_old, class);
        let (k, code_new) = match appended.get(&i) {
            Some(&(k, c)) => (Some(k), Some(c)),
            None => (None, None),
        };
        match code_new {
            Some(c) if c == want => {}
            Some(c) => unexplained.push(format!("source {i}: appended at code {c}, its class {class:?} and old code {code_old} give {want}")),
            None => unexplained.push(format!("source {i}: re-solved ({class:?}) but not appended (a duplicate?)")),
        }
        if let (MigrationClass::Unchanged, Some(k)) = (class, k) {
            let n = &s.collection().unwrap().samples[k].sample;
            if n.energy.to_bits() != cs.sample.energy.to_bits() {
                unexplained.push(format!("source {i}: class U, but the appended energy is not the old record's"));
            }
        }
        let _ = writeln!(
            csv,
            "{i},{},{},{code_old},{},{},{},{:.12},{:.12},{:.12},{:.6e},{:.6e},{:.6e},{:.6e},{:.3e},{:.3e},{},{},{},{:.0}",
            cs.scene,
            cs.grid_n,
            ["U", "T", "S"][class.index()],
            k.map_or(String::new(), |k| k.to_string()),
            code_new.map_or(String::new(), |c| c.to_string()),
            cs.sample.energy,
            rec.free,
            rec.internal,
            rec.ts,
            d_free,
            d_int,
            dff,
            eps.e,
            eps.f,
            rec.tol_used.map_or(String::new(), |t| format!("{t:e}")),
            rec.partial,
            rec.scf_steps,
            ms(rec.wall)
        );
    }
    let csv_path = root.with_file_name(format!("{}-v11-samples.csv", root.file_name().unwrap().to_string_lossy()));
    std::fs::write(&csv_path, csv).unwrap();
    println!("V11 per-sample table: {}", csv_path.display());
    println!("V11 re-derived counts U/T/S/not {counts:?} (log {:?}); by scene {by_scene_class:?}", mlog.counts);
    println!("V11 class U: |F − E_old| {}; {fd_labelled_u} with TS > ε_E (an old label that already was F), {partial_u} with a partly occupied level", quantiles(d_u));
    println!("V11 class T: F − E_old {}", quantiles(d_t));
    println!("V11 class S: F − E_old {}; forces within ε_F {s_force_ok}, beyond {s_force_fail}", quantiles(d_s.clone()));
    let n_s = counts[2];
    println!("V11 false-S estimate: {false_s} of {n_s} S samples ({:.1} %) fail only the energy, by ≤ 3 ε_E", if n_s > 0 { 100.0 * false_s as f64 / n_s as f64 } else { 0.0 });
    println!("V11 TS of the new solves: {}", quantiles(ts_all));
    let warned: Vec<String> = solved.iter().filter(|(_, r)| !r.warnings.is_empty()).map(|(i, r)| format!("{i}: {:?}", r.warnings)).collect();
    println!("V11 sample solves with a solver warning: {} {:?}", warned.len(), warned.iter().take(10).collect::<Vec<_>>());
    println!("V11 time per sample solve (s): {}", quantiles(walls.clone()));
    for (n, w) in &walls_by_n {
        let mut w = w.clone();
        w.sort_by(f64::total_cmp);
        println!("  {n}³: {} solves, median {:.2} s, max {:.2} s", w.len(), w[w.len() / 2], w[w.len() - 1]);
    }
    let refs_wall: f64 = recs.lock().unwrap().iter().map(|r| r.wall.as_secs_f64()).sum::<f64>() - walls.iter().sum::<f64>();
    println!("V11 GPU time: sample solves {:.1} min, reference scans {:.1} min; the pass {:.1} min wall", walls.iter().sum::<f64>() / 60.0, refs_wall / 60.0, pass_wall.as_secs_f64() / 60.0);

    // ---- 4. the review and the verdict after ----
    let c = s.collection().unwrap();
    let mut changed: BTreeMap<(u8, u8), usize> = BTreeMap::new();
    let reviewed: BTreeMap<usize, (f64, bool)> = mlog.review.iter().map(|&(k, r, rel)| (k, (r, rel))).collect();
    let mut changed_unreviewed = Vec::new();
    for (&i, &(k, at_append)) in &appended {
        let now = c.quarantine[k];
        if now != at_append {
            *changed.entry((at_append, now)).or_default() += 1;
            if !reviewed.contains_key(&k) {
                changed_unreviewed.push(format!("source {i} → new {k}: {at_append} → {now}"));
            }
        }
    }
    let released = mlog.review.iter().filter(|e| e.2).count();
    let ratios: Vec<f64> = mlog.review.iter().map(|e| e.1).collect();
    println!("V11 review: {} verdicts, {released} released, {} kept, {} waiting (stray atoms); ratios {}", mlog.review.len(), mlog.review.len() - released, mlog.waiting.len(), quantiles(ratios));
    for (k, r, _) in mlog.review.iter().filter(|e| !e.2) {
        let src = appended.iter().find(|(_, v)| v.0 == *k).map(|(i, _)| *i);
        println!("  kept: new {k} (source {src:?}, {}), ratio {r:.3}", src.map_or("—".to_string(), |i| old.samples[i].scene.clone()));
    }
    println!("V11 codes changed after the appends (by the refit): {changed:?}; not in the review log: {}", changed_unreviewed.len());
    for x in changed_unreviewed.iter().take(40) {
        println!("  {x}");
    }
    println!("V11 state {:?}; summary:\n{}", mlog.state, surrogate::summary_text(&mlog).unwrap_or_default());
    let after = verdict_lines("V11 after the migration (new labels, reviewed)", &s);
    print!("{after}");
    println!("V11 trust verdict: baseline {} → migrated {}", base_verdict.0, s.trusted());
    let notes_path = root.with_file_name(format!("{}-v11-notes.txt", root.file_name().unwrap().to_string_lossy()));
    std::fs::write(&notes_path, notes.join("\n")).unwrap();
    println!("V11 provider notes ({}): {}", notes.len(), notes_path.display());
    drop(s);

    // ---- 5. the musts ----
    assert!(unexplained.is_empty(), "samples changed code without a logged reason:\n{}", unexplained.join("\n"));
    assert_eq!(counts, mlog.counts, "the log's counts are the classes re-derived from the solves");
    assert_eq!(files(&copy), before, "the source copy is byte-identical");
    println!("V11 source copy byte-identical: {} files; total {:.1} min", before.len(), t0.elapsed().as_secs_f64() / 60.0);
}
