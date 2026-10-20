use super::*;
use crate::surrogate::FallbackPolicy;
use sim::{AtomReferences, Cell, DynCommand, Dynamics, EvalStats, MorseDimer, Nucleus, Shared, SharedTrainable, Source, Trainable};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

fn never() -> impl FnMut() -> bool {
    || false
}

/// A scripted teacher: a sum of pair Morse wells (any N; one nucleus → `e_atom`, zero force),
/// which records every scene it is asked about, counts solves in a shared counter (readable
/// from a `should_abort` closure without the provider's lock), polls `should_abort` once
/// before solving (like `CpuKs`), and refuses a geometry outside the cell margin (so a guard
/// failure would show as a failed sample, not pass silently).
struct PairMorse {
    d: f64,
    a: f64,
    r_e: f64,
    e_atom: f64,
    seen: Vec<Scene>,
    solves: Arc<AtomicUsize>,
    resets: usize,
    discontinuities: usize,
    /// (Z, multiplicity, energy) overrides for a single atom (ruling S9 tests).
    atom_energy: Vec<(u8, u32, f64)>,
    /// A scripted box (final review I1): `(x_max, rho, bias)` — a geometry with a nucleus more
    /// than `x_max` off-centre along any axis is reported with face density `rho` and its energy
    /// raised by `bias` (the box's confinement); every other one with a face density of 1e-7.
    confine: Option<(f64, f64, f64)>,
    /// Lone-atom solves that fail before one succeeds (the 2026-10-20 PH₃ case: an isolated atom
    /// whose reference does not solve) — an SCF stall, or with `atom_device` a lost device readback;
    /// `usize::MAX` never succeeds.
    atom_failures: usize,
    atom_device: bool,
}

impl PairMorse {
    fn h2() -> PairMorse {
        let m = MorseDimer::h2();
        PairMorse { d: m.d, a: m.a, r_e: m.r_e, e_atom: m.e_atom, seen: Vec::new(), solves: Arc::new(AtomicUsize::new(0)), resets: 0, discontinuities: 0, atom_energy: Vec::new(), confine: None, atom_failures: 0, atom_device: false }
    }

    /// Whether the scripted box confines this geometry.
    fn confined(&self, pos: &[[f64; 3]]) -> bool {
        self.confine.is_some_and(|(x_max, _, _)| pos.iter().flatten().any(|x| x.abs() > x_max))
    }

    fn energy_forces(&self, pos: &[[f64; 3]]) -> (f64, Vec<[f64; 3]>) {
        let n = pos.len();
        let mut e = n as f64 * self.e_atom;
        let mut f = vec![[0.0; 3]; n];
        for i in 0..n {
            for j in i + 1..n {
                let d: [f64; 3] = std::array::from_fn(|a| pos[j][a] - pos[i][a]);
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                let x = (-self.a * (r - self.r_e)).exp();
                e += self.d * ((1.0 - x) * (1.0 - x) - 1.0);
                let de = 2.0 * self.d * self.a * (1.0 - x) * x;
                for a in 0..3 {
                    f[i][a] += de * d[a] / r;
                    f[j][a] -= de * d[a] / r;
                }
            }
        }
        (e, f)
    }
}

impl ForceProvider for PairMorse {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        if should_abort() {
            return Err(ProviderError::Cancelled);
        }
        if !inside_cell(scene) {
            return Err(ProviderError::Invalid("pair morse: nucleus outside the cell margin".into()));
        }
        self.seen.push(scene.clone());
        self.solves.fetch_add(1, Ordering::SeqCst);
        if scene.nuclei.len() == 1 && self.atom_failures > 0 {
            self.atom_failures -= 1;
            return Err(ProviderError::Solver(if self.atom_device {
                sim::SolverError::Device("callback dropped".into())
            } else {
                sim::SolverError::ScfStalled { iterations: 63, best_delta_rho: 0.2, tol: 1e-6, window: 50, level: 5e-3 }
            }));
        }
        let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        let (mut energy, forces) = self.energy_forces(&pos);
        if let ([n], Some(m)) = (scene.nuclei.as_slice(), scene.multiplicity) {
            if let Some(&(_, _, e)) = self.atom_energy.iter().find(|(z, mm, _)| *z == n.z && *mm == m) {
                energy = e;
            }
        }
        let mut stats = EvalStats::default();
        if let Some((_, rho, bias)) = self.confine {
            let boxed = self.confined(&pos);
            stats.face_density = Some(if boxed { rho } else { 1e-7 });
            if boxed {
                energy += bias;
            }
        }
        stats.energy.total = energy;
        Ok(ForceResult { energy, forces, stats, display: None, orbitals: None })
    }

    fn name(&self) -> &'static str {
        "pair morse"
    }

    fn reset(&mut self) {
        self.resets += 1;
    }

    fn discontinuity(&mut self) {
        self.discontinuities += 1;
    }
}

fn cell(n: usize) -> Cell {
    Cell { n, h: 0.25 }
}

/// Two nuclei `r` apart on the x axis, centred on the origin (±r/2: exactly 2× apart for 2r).
fn pair(z0: u8, z1: u8, r: f64, c: Cell) -> Scene {
    Scene {
        nuclei: vec![Nucleus::new(z0, [-0.5 * r, 0.0, 0.0]), Nucleus::new(z1, [0.5 * r, 0.0, 0.0])],
        cell: c,
        charge: 0,
        multiplicity: None,
        sigma: None,
        external_field: [0.0; 3],
            ionic: sim::IonicPolicy::None,
    }
}

fn dist(s: &Scene, i: usize, j: usize) -> f64 {
    let d: [f64; 3] = std::array::from_fn(|a| s.nuclei[j].pos[a] - s.nuclei[i].pos[a]);
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// A surrogate whose own teacher is the shared `PairMorse` (the test reads its record).
fn surrogate_over(teacher: &Shared<PairMorse>) -> SurrogateKs {
    SurrogateKs::new(Box::new(Shared(teacher.handle())), FallbackPolicy::default()).with_teacher_tol(1e-6)
}

fn tag_counts(s: &SurrogateKs) -> [usize; 6] {
    let mut c = [0; 6];
    for smp in &s.store().expect("samples").samples {
        let k = match smp.tag {
            SampleTag::LiveStep => 0,
            SampleTag::Perturbation => 1,
            SampleTag::HotBurst => 2,
            SampleTag::Scan => 3,
            SampleTag::Fallback => 4,
            SampleTag::Reference => 5,
            SampleTag::Audit => 6,
            SampleTag::DraftAudit => 7,
            SampleTag::Targeted => 8,
        };
        c[k] += 1;
    }
    c
}

/// Brief test 2: the plan's counts, the tags, the reference from an isolated-atom call at the
/// cell centre, the teacher-state hints, and the reference cache on a second pass.
#[test]
fn sampling_pass_yields_the_requested_counts_and_tags() {
    let mut teacher = Shared::new(PairMorse::h2());
    let h = teacher.handle();
    let mut s = surrogate_over(&teacher);
    let scene = pair(1, 1, 1.4, cell(41));
    let plan = SamplingPlan { perturbations: 10, hot_burst_steps: 20, scan_points_per_pair: 6, seed: 3, ..SamplingPlan::default() };
    let mut sampler = Sampler::new(&scene, &plan).unwrap();
    assert_eq!(sampler.progress().total, 1 + 10 + 6 + 20);
    let mut reports = Vec::new();
    while let Some(p) = sampler.step(&mut teacher, &mut s, &mut never()).unwrap() {
        reports.push(p);
    }
    let fin = sampler.progress();
    println!("final progress {fin:?}; notes {:?}", sampler.take_notes());
    assert_eq!(reports.len(), 37, "one progress report per solve");
    assert!(reports.windows(2).all(|w| w[1].done == w[0].done + 1));
    assert_eq!((fin.done, fin.total, fin.skipped, fin.failed), (37, 37, 0, 0));
    assert!(fin.finished && !fin.cancelled);
    assert_eq!(fin.stage, "done");
    assert_eq!(sampler.samples_added(), 36);
    assert_eq!(s.samples(), 10 + 20 + 6);
    // tags: LiveStep, Perturbation, HotBurst, Scan, Fallback, Reference
    assert_eq!(tag_counts(&s), [0, 10, 20, 6, 0, 0]);
    let t = h.lock().unwrap();
    assert_eq!(t.seen.len(), 37);
    // the first solve: one H at the cell centre of the scene's own grid, neutral, at its only
    // allowed multiplicity (one electron: 2)
    let atom = &t.seen[0];
    assert_eq!(atom.nuclei.len(), 1);
    assert_eq!((atom.nuclei[0].z, atom.nuclei[0].pos), (1, [0.0; 3]));
    assert_eq!((atom.cell.clone(), atom.charge, atom.multiplicity, atom.sigma), (scene.cell.clone(), 0, Some(2), None));
    assert_eq!(s.reference_energy(1), Some(t.e_atom), "the teacher's isolated-atom answer, not a table");
    // stages in order: perturbations, scans (fixed nucleus 0 stays, nucleus 1 moves: a tie →
    // the higher index), the burst (starts at the scene's positions and moves)
    let order: Vec<SampleTag> = s.store().unwrap().samples.iter().map(|x| x.tag).collect();
    assert!(order[..10].iter().all(|&g| g == SampleTag::Perturbation));
    assert!(order[10..16].iter().all(|&g| g == SampleTag::Scan));
    assert!(order[16..].iter().all(|&g| g == SampleTag::HotBurst));
    let store = &s.store().unwrap().samples;
    let disp: Vec<f64> = store[..10]
        .iter()
        .flat_map(|x| x.pos.iter().zip(&scene.nuclei).flat_map(|(p, n)| (0..3).map(move |a| p[a] - n.pos[a])))
        .collect();
    let rms = (disp.iter().map(|x| x * x).sum::<f64>() / disp.len() as f64).sqrt();
    println!("perturbation rms displacement {rms:.3} Bohr (σ = 0.15) over {} coordinates", disp.len());
    assert!((0.10..0.20).contains(&rms), "rms {rms}");
    for (k, smp) in store[10..16].iter().enumerate() {
        assert_eq!(smp.pos[0], scene.nuclei[0].pos);
        let want = 1.4 * (0.7 + 1.1 * k as f64 / 5.0);
        let r = (0..3).map(|a| (smp.pos[1][a] - smp.pos[0][a]).powi(2)).sum::<f64>().sqrt();
        assert!((r - want).abs() < 1e-12, "scan point {k}: r {r} vs {want}");
    }
    let burst = &store[16..];
    assert_eq!(burst[0].pos, vec![scene.nuclei[0].pos, scene.nuclei[1].pos], "the burst starts at the scene's geometry");
    let max_move = burst.windows(2).map(|w| (0..3).map(|a| (w[1].pos[1][a] - w[0].pos[1][a]).abs()).fold(0.0, f64::max)).fold(0.0, f64::max);
    println!("hot burst: max per-step displacement {max_move:.2e} Bohr");
    assert!(max_move > 1e-4 && max_move < 0.1, "a 2000 K trajectory at 0.25 fs: {max_move}");
    // every sample is the teacher's own answer at its geometry
    for smp in store {
        let (e, f) = t.energy_forces(&smp.pos);
        assert_eq!((smp.energy, &smp.forces), (e, &f));
    }
    // teacher state: reset on the nucleus-count changes (→ atom, → molecule), discontinuity
    // before each later perturbation, the scan start and the burst start
    assert_eq!((t.resets, t.discontinuities), (2, 9 + 1 + 1));
    drop(t);

    // a second pass at the same settings takes the reference from the cache: no atom solve
    let cache = sampler.take_references();
    assert_eq!(cache.len(), 1);
    let mut again = Sampler::new(&scene, &plan).unwrap().with_references(cache);
    while again.step(&mut teacher, &mut s, &mut never()).unwrap().is_some() {}
    let t = h.lock().unwrap();
    assert_eq!(t.seen.len(), 37 + 36, "no reference solve the second time");
    assert!(t.seen[37..].iter().all(|sc| sc.nuclei.len() == 2));
    assert_eq!(again.progress().done, 37);
    assert_eq!(s.samples(), 72);
    // the pass's samples fit: an exact pair potential, 72 geometries
    drop(t);
    let rep = s.refit().expect("refit");
    println!("refit on 72 samples: rmse E/atom {:.1e}, F {:.1e}; trusted {}", rep.rmse_energy_per_atom, rep.rmse_force, s.trusted());
}

/// Brief test 3: the scan's separations are fractions of the scene's own separation — two
/// scenes 1.4 and 2.8 Bohr apart give ranges exactly 2× apart — and do not depend on which
/// elements the pair is (rule 1: no radius table); the lighter nucleus is the one moved.
#[test]
fn scan_ranges_come_from_the_scene_not_a_table() {
    let plan = SamplingPlan { perturbations: 0, hot_burst_steps: 0, scan_points_per_pair: 12, ..SamplingPlan::default() };
    let short = Sampler::new(&pair(1, 1, 1.4, cell(49)), &plan).unwrap();
    let long = Sampler::new(&pair(1, 1, 2.8, cell(49)), &plan).unwrap();
    let (a, b) = (&short.scans()[0], &long.scans()[0]);
    println!("r₀ 1.4: r ∈ [{:.4}, {:.4}]; r₀ 2.8: r ∈ [{:.4}, {:.4}]", a.r[0], a.r[11], b.r[0], b.r[11]);
    assert_eq!((a.r0, b.r0), (1.4, 2.8));
    assert_eq!(a.r.len(), 12);
    for (x, y) in a.r.iter().zip(&b.r) {
        assert_eq!(*y, 2.0 * x, "exactly 2× (a power-of-two scale is exact in binary)");
    }
    assert_eq!((a.r[0], a.r[11]), (1.4 * 0.7, 1.4 * 1.8));
    // another species pair at the same separation: the same separations, the lighter moves
    let hli = Sampler::new(&pair(3, 1, 1.4, cell(49)), &plan).unwrap();
    let c = &hli.scans()[0];
    assert_eq!(c.r, a.r, "nothing per element enters the range");
    assert_eq!((a.moved, c.moved), (1, 1), "H–H: the higher index on a tie; Li–H: the H");
    let lih = Sampler::new(&pair(1, 3, 1.4, cell(49)), &plan).unwrap();
    assert_eq!(lih.scans()[0].moved, 0, "H–Li: the H (index 0)");
    // and the geometries the teacher sees realise those separations
    // (r₀ 2.8: its last point, 1.8 r₀ = 5.04 Bohr, is beyond 5/6 of r_cut and not solved)
    for (sc, sampler_scene, want) in [(a, pair(1, 1, 1.4, cell(49)), 12), (b, pair(1, 1, 2.8, cell(49)), 11)] {
        let mut teacher = Shared::new(PairMorse::h2());
        let h = teacher.handle();
        let mut s = surrogate_over(&teacher);
        let n = run_sampling(&sampler_scene, &plan, &mut teacher, &mut s, &mut |_| {}, &mut never()).unwrap();
        assert_eq!(n, want);
        let t = h.lock().unwrap();
        let seen: Vec<f64> = t.seen.iter().filter(|x| x.nuclei.len() == 2).map(|x| dist(x, 0, 1)).collect();
        for (got, want) in seen.iter().zip(&sc.r) {
            assert!((got - want).abs() < 1e-12 * want, "{got} vs {want}");
        }
    }
    // a pair beyond the model's cutoff is not scanned (no neighbour inside r_cut to learn)
    let far = Sampler::new(&pair(1, 1, 6.5, cell(49)), &plan).unwrap();
    assert!(far.scans().is_empty());
}

/// Ruling: a generated geometry closer than 3h to a face is counted and never sent to the
/// teacher (which here would refuse it and count a failure).
#[test]
fn geometries_outside_the_cell_margin_are_skipped_never_solved() {
    // 33³, h 0.25: |x| ≤ 4 − 0.75 = 3.25. Fixed nucleus at −1.4, the moved one at −1.4 + r:
    // of r = 2.8·{0.7, 0.92, 1.14, 1.36, 1.58, 1.8} only the last (3.64) is outside.
    let plan = SamplingPlan { perturbations: 0, hot_burst_steps: 0, scan_points_per_pair: 6, ..SamplingPlan::default() };
    let scene = pair(1, 1, 2.8, cell(33));
    let mut teacher = Shared::new(PairMorse::h2());
    let h = teacher.handle();
    let mut s = surrogate_over(&teacher);
    let mut last = None;
    let n = run_sampling(&scene, &plan, &mut teacher, &mut s, &mut |p| last = Some(p), &mut never()).unwrap();
    let p = last.unwrap();
    println!("{p:?}");
    assert_eq!(n, 5);
    assert_eq!((p.done, p.total, p.skipped, p.failed), (7, 7, 1, 0));
    let t = h.lock().unwrap();
    assert_eq!(t.seen.len(), 1 + 5);
    assert!(t.seen.iter().all(inside_cell));
    // a burst hot enough to leave the cell ends there: every later step is skipped, never solved
    let plan = SamplingPlan { perturbations: 0, scan_points_per_pair: 0, hot_burst_steps: 400, hot_burst_t: 3.0e6, ..SamplingPlan::default() };
    drop(t);
    let mut last = None;
    let n = run_sampling(&scene, &plan, &mut teacher, &mut s, &mut |p| last = Some(p), &mut never()).unwrap();
    let p = last.unwrap();
    println!("hot burst at 3e6 K: {n} samples, {p:?}");
    assert!(p.skipped > 0 && n + p.skipped == 400 && p.failed == 0, "{p:?}");
    assert!(h.lock().unwrap().seen.iter().all(inside_cell));
}

/// Final review I1: a pass's samples carry their teacher's face density, so the box rule applies
/// to them. On a small box (33³: faces at ±4 Bohr) the default plan's scan to 1.8 r₀ and its
/// 2000 K burst reach geometries whose labels carry the box's confinement — scripted here as
/// +0.02 Ha and a face density 5× the rule whenever a nucleus is more than 1.3 Bohr off-centre.
/// Those samples are box-affected: kept, never quarantined, and left out of the fit at the trust
/// floor, so the model is trusted on the clean ones and answers at the scene. The control: the
/// same samples handed over without their densities (what the pass did before) are taken for
/// glitches — quarantine candidates that fence their descriptor neighbourhood.
#[test]
fn a_pass_on_a_small_box_marks_its_confined_samples_box_affected() {
    let rho = 5.0 * crate::surrogate::BOX_AFFECTED_FACE_DENSITY;
    let mut morse = PairMorse::h2();
    morse.confine = Some((1.3, rho, 0.02));
    let mut teacher = Shared::new(morse);
    let h = teacher.handle();
    let mut s = surrogate_over(&teacher);
    let scene = pair(1, 1, 1.4, cell(33));
    let mut sampler = Sampler::new(&scene, &SamplingPlan::default()).unwrap();
    while sampler.step(&mut teacher, &mut s, &mut never()).unwrap().is_some() {}
    let notes = sampler.take_notes();
    let store = s.store().unwrap().samples.clone();
    let confined: Vec<usize> = (0..store.len()).filter(|&i| h.lock().unwrap().confined(&store[i].pos)).collect();
    println!("33³ pass: {} samples, {} confined by the scripted box; notes {notes:?}", store.len(), confined.len());
    assert!(confined.len() >= 4, "test premise: the scan's long end and the burst reach the box ({})", confined.len());
    assert_eq!(s.box_affected(), confined.len(), "every confined pass sample is stored as box-affected");
    assert!(notes.iter().any(|n| n.contains(&format!("{} of {} samples are box-affected", confined.len(), store.len()))), "{notes:?}");
    let rep = s.refit().expect("refit");
    println!(
        "refit: {} fitted, trusted {} {:?}, quarantined {}, box-excluded {}",
        rep.n_samples,
        s.trusted(),
        s.untrusted_reason(),
        s.quarantined(),
        s.box_excluded()
    );
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    assert_eq!((s.quarantined(), s.box_excluded(), rep.n_samples), (0, confined.len(), store.len() - confined.len()));
    assert!(confined.iter().all(|&i| !s.is_quarantined(i)));
    let r = s.evaluate(&scene, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Surrogate, "{:?}", r.stats.notes);

    // the control: the same labels without their densities
    let mut blind = surrogate_over(&Shared::new(PairMorse::h2()));
    blind.set_reference_energy(1, s.reference_energy(1).unwrap());
    for smp in &store {
        blind.add_sample(smp.clone(), &scene).unwrap();
    }
    blind.refit().expect("refit");
    let held = confined.iter().filter(|&&i| blind.is_quarantined(i)).count();
    println!("without densities: trusted {}, quarantined {} ({held} of them confined), box-excluded {}", blind.trusted(), blind.quarantined(), blind.box_excluded());
    assert!(held > 0 || !blind.trusted(), "the control should mistake the confined labels for glitches");
    assert_eq!(blind.box_excluded(), 0);
}

/// Brief test 4: an abort after 7 sample solves leaves those 7 samples (and the reference),
/// a cancelled progress, and a set that refits.
#[test]
fn cancelling_a_pass_leaves_a_usable_partial_set() {
    let mut teacher = Shared::new(PairMorse::h2());
    let solves = teacher.handle().lock().unwrap().solves.clone();
    let mut s = surrogate_over(&teacher);
    let scene = pair(1, 1, 1.4, cell(41));
    let mut reports = Vec::new();
    // 1 reference + 7 samples, then the abort (polled between solves and by the teacher)
    let mut abort = || solves.load(Ordering::SeqCst) >= 8;
    let n = run_sampling(&scene, &SamplingPlan::default(), &mut teacher, &mut s, &mut |p| reports.push(p), &mut abort).unwrap();
    let fin = reports.last().unwrap().clone();
    println!("cancelled: {n} samples, {fin:?}");
    assert_eq!(n, 7);
    assert_eq!(s.samples(), 7);
    assert!(fin.cancelled && fin.finished);
    assert_eq!(fin.stage, "cancelled");
    assert_eq!(fin.done, 8);
    assert_eq!(fin.total, 1 + 60 + 12 + 200);
    assert!(s.reference_energy(1).is_some());
    let rep = s.refit().expect("a partial set refits");
    println!("refit on the partial set: n_samples {}, q′ {}, trusted {}", rep.n_samples, rep.q_prime, s.trusted());
    assert_eq!(rep.n_samples, 7);
    assert!(s.model().is_some());
    // an abort raised inside a solve leaves that job queued: the next step retries it
    let mut sampler = Sampler::new(&scene, &SamplingPlan { perturbations: 2, hot_burst_steps: 0, scan_points_per_pair: 0, ..SamplingPlan::default() }).unwrap();
    assert!(matches!(sampler.step(&mut teacher, &mut s, &mut || true), Err(ProviderError::Cancelled)));
    assert_eq!(sampler.progress().done, 0);
    let mut k = 0;
    while sampler.step(&mut teacher, &mut s, &mut never()).unwrap().is_some() {
        k += 1;
    }
    assert_eq!((k, sampler.progress().done), (3, 3));
}

/// A surrogate boxed for a `Dynamics` (through `SharedTrainable`, which the driver reaches with
/// `as_trainable` like the surrogate itself), and the handle the test reads it by.
fn driven(s: SurrogateKs) -> (Box<SharedTrainable<SurrogateKs>>, Arc<Mutex<SurrogateKs>>) {
    let p = SharedTrainable::new(s);
    let h = p.handle();
    (Box::new(p), h)
}

/// The surrogate inside a `Dynamics` built with [`driven`], under its lock — never held across
/// a call into that `Dynamics` (the lock is not re-entrant).
fn surrogate_of(h: &Arc<Mutex<SurrogateKs>>) -> MutexGuard<'_, SurrogateKs> {
    h.lock().unwrap_or_else(|e| e.into_inner())
}

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("sim-sampling-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

/// Brief test 5: StartSampling → progress in the snapshot, the dynamics paused and Play/Step
/// refused, one solve per tick at a zero budget; CancelSampling → cancelled and the partial
/// set refitted; a full pass → a model, the reference cache reused by the next pass;
/// ForgetModel → no model, no progress.
#[test]
fn sampling_commands_flow_through_dynamics() {
    let teacher = Shared::new(PairMorse::h2());
    let h = teacher.handle();
    let mut scene = pair(1, 1, 1.45, cell(41));
    scene.nuclei[1].vel = [3e-4, 0.0, 0.0];
    let (p, sh_d) = driven(surrogate_over(&teacher));
    let mut d = Dynamics::new(scene, p, 0.25, 1);
    assert_eq!(d.snapshot().sampling, None);
    d.apply(DynCommand::Play);
    d.tick(f64::INFINITY, &mut never());
    let step0 = d.snapshot().step;
    assert!(step0 >= 1);
    let plan = SamplingPlan { perturbations: 30, hot_burst_steps: 20, scan_points_per_pair: 12, seed: 5, ..SamplingPlan::default() };
    d.apply(DynCommand::StartSampling(plan.clone()));
    let snap = d.snapshot();
    assert!(snap.paused, "sampling pauses the dynamics");
    assert_eq!(snap.last_rejection, None);
    let p = snap.sampling.clone().expect("progress while sampling");
    assert_eq!((p.done, p.total, p.finished), (0, 1 + 30 + 12 + 20, false));
    d.apply(DynCommand::Play);
    let rej = d.snapshot().last_rejection.expect("Play is refused while sampling");
    assert!(rej.contains("sampling"), "{rej}");
    d.apply(DynCommand::Step(3));
    assert!(d.snapshot().last_rejection.is_some(), "so is Step");
    d.apply(DynCommand::StartSampling(plan.clone()));
    assert!(d.snapshot().last_rejection.is_some(), "one pass at a time");
    let solves0 = h.lock().unwrap().seen.len();
    for k in 1..=3 {
        let snap = d.tick(0.0, &mut never());
        assert_eq!(snap.sampling.as_ref().unwrap().done, k, "a zero budget still advances one solve per tick");
        assert_eq!(snap.step, step0, "no dynamics while sampling");
    }
    assert_eq!(h.lock().unwrap().seen.len(), solves0 + 3);
    // an abort raised by the worker interrupts, it does not cancel
    let snap = d.tick(f64::INFINITY, &mut || true);
    assert_eq!(snap.sampling.as_ref().unwrap().done, 3);
    assert!(!snap.sampling.as_ref().unwrap().cancelled);
    let disc0 = h.lock().unwrap().discontinuities;
    d.apply(DynCommand::CancelSampling);
    let snap = d.snapshot();
    let p = snap.sampling.clone().unwrap();
    println!("cancelled: {p:?}; notes {:?}", snap.notes);
    assert!(p.cancelled && p.finished);
    assert!(snap.notes.iter().any(|n| n.contains("samples kept — not enough to fit (Q′ = ")), "{:?}", snap.notes);
    assert_eq!(p.stage, "cancelled");
    assert!(h.lock().unwrap().discontinuities > disc0, "the live run must not extrapolate from the last sample geometry");
    {
        let s = surrogate_of(&sh_d);
        // the live evaluations so far (fallbacks) + 2 perturbations, kept — below Q′ not
        // refitted (M6 Task 3 fix round 2: the model could not earn trust, and the fit cost
        // seconds inside the Cancel)
        assert!(s.model().is_none(), "below Q′ the partial set is kept, not refitted");
        assert!(s.store().unwrap().samples.iter().filter(|x| x.tag == SampleTag::Perturbation).count() == 2);
    }
    d.apply(DynCommand::Play);
    assert_eq!(d.snapshot().last_rejection, None, "Play is accepted again");
    d.apply(DynCommand::Pause);

    // a full pass: the reference comes from the cache now (no atom solve)
    let seen0 = h.lock().unwrap().seen.len();
    d.apply(DynCommand::StartSampling(plan.clone()));
    let mut ticks = 0;
    while !d.tick(f64::INFINITY, &mut never()).sampling.unwrap().finished {
        ticks += 1;
        assert!(ticks < 1000);
    }
    let snap = d.snapshot();
    let p = snap.sampling.clone().unwrap();
    println!("full pass: {p:?} in {} ticks; notes {:?}", ticks + 1, snap.notes);
    assert!(!p.cancelled && p.finished && p.done == p.total);
    assert_eq!(h.lock().unwrap().seen.len() - seen0, 30 + 12 + 20, "the cached reference cost no solve");
    let (trusted, n) = {
        let s = surrogate_of(&sh_d);
        (s.trusted(), s.report().unwrap().n_samples)
    };
    println!("after the pass: {n} samples fitted, trusted {trusted}");
    assert_eq!(n, surrogate_of(&sh_d).samples(), "the pass ends with a refit over everything");
    assert!(trusted, "an exact pair potential sampled 60+ times is learned");
    // the refit moved the surface: the next tick refreshes at the unchanged positions
    let refreshed = d.tick(0.0, &mut never());
    assert_eq!(refreshed.source, Source::Surrogate, "{:?}", refreshed.eval.notes);

    d.apply(DynCommand::ForgetModel);
    let snap = d.tick(0.0, &mut never());
    assert_eq!(snap.sampling, None);
    assert!(surrogate_of(&sh_d).model().is_none());
    assert_eq!(surrogate_of(&sh_d).samples(), 1, "only the refresh evaluation's fallback sample");
    assert_eq!(snap.source, Source::Fallback);

    // the commands are refused on a provider without a model
    let mut plain = Dynamics::new(pair(1, 1, 1.4, cell(41)), Box::new(MorseDimer::h2()), 0.25, 1);
    for cmd in [DynCommand::StartSampling(plan), DynCommand::ForgetModel, DynCommand::SaveModel(tmp("x.model")), DynCommand::LoadModel(tmp("x.model"))] {
        plain.apply(cmd);
        let r = plain.snapshot().last_rejection.expect("refused");
        assert!(r.contains("no surrogate"), "{r}");
    }
}

/// The sidecar through the driver: a trained model saved, reloaded into a fresh provider on the
/// same settings (bitwise the same model, the same trust verdict, answering at once), refused
/// with a note on another grid, a corrupt file refused, nothing to save refused.
#[test]
fn a_saved_model_reloads_through_dynamics_and_a_foreign_provenance_is_refused() {
    let teacher = Shared::new(PairMorse::h2());
    let scene = pair(1, 1, 1.45, cell(41));
    let (p, sh_d) = driven(surrogate_over(&teacher));
    let mut d = Dynamics::new(scene.clone(), p, 0.25, 1);
    let path = tmp("h2.model");
    // nothing fitted yet: the construction's rule-1 answer is kept as a sample, and a save keeps it
    // (a samples-only file since 2026-10-20 — before, an unfitted state could not be saved)
    d.apply(DynCommand::SaveModel(path.clone()));
    assert_eq!(d.snapshot().last_rejection, None);
    assert_eq!(&std::fs::read(&path).unwrap()[..8], surrogate::sidecar::MAGIC_V6);
    d.apply(DynCommand::StartSampling(SamplingPlan { perturbations: 30, hot_burst_steps: 20, scan_points_per_pair: 12, ..SamplingPlan::default() }));
    while !d.tick(f64::INFINITY, &mut never()).sampling.unwrap().finished {}
    d.apply(DynCommand::SaveModel(path.clone()));
    let snap = d.snapshot();
    assert_eq!(snap.last_rejection, None);
    assert!(snap.notes.last().unwrap().contains("saved"), "{:?}", snap.notes);
    let (model, report, n, trusted) = {
        let s = surrogate_of(&sh_d);
        (s.model().unwrap().clone(), s.report().unwrap().clone(), s.samples(), s.trusted())
    };
    assert!(trusted);

    // same settings, fresh provider: the model is applied and answers at once
    let teacher2 = Shared::new(PairMorse::h2());
    let h2 = teacher2.handle();
    let (p, sh_d2) = driven(surrogate_over(&teacher2));
    let mut d2 = Dynamics::new(scene.clone(), p, 0.25, 1);
    d2.apply(DynCommand::LoadModel(path.clone()));
    let snap = d2.snapshot();
    assert_eq!(snap.last_rejection, None, "{:?}", snap.notes);
    println!("loaded: {:?}", snap.notes.last());
    {
        let s = surrogate_of(&sh_d2);
        assert_eq!(s.model().unwrap(), &model, "bitwise the saved model");
        assert_eq!(s.report().unwrap(), &report);
        assert_eq!(s.samples(), n);
        assert_eq!(s.trusted(), trusted);
        assert!(s.reference_energy(1).is_some());
    }
    let calls = h2.lock().unwrap().seen.len();
    let snap = d2.tick(0.0, &mut never());
    assert_eq!(snap.source, Source::Surrogate, "{:?}", snap.eval.notes);
    assert_eq!(h2.lock().unwrap().seen.len(), calls, "the loaded model answered, no teacher call");
    // the loaded reference seeds the cache: a pass on the loaded model solves no atom
    d2.apply(DynCommand::StartSampling(SamplingPlan { perturbations: 2, hot_burst_steps: 0, scan_points_per_pair: 0, ..SamplingPlan::default() }));
    while !d2.tick(f64::INFINITY, &mut never()).sampling.unwrap().finished {}
    assert_eq!(h2.lock().unwrap().seen.len(), calls + 2);

    // another grid: refused with a note, the provider untouched
    let mut other = pair(1, 1, 1.45, cell(49));
    other.sigma = Some(0.25);
    let (p, sh_d3) = driven(surrogate_over(&Shared::new(PairMorse::h2())));
    let mut d3 = Dynamics::new(other, p, 0.25, 1);
    d3.apply(DynCommand::LoadModel(path.clone()));
    let snap = d3.snapshot();
    let r = snap.last_rejection.expect("a model from another grid is refused");
    println!("refused: {r}");
    assert!(r.contains("49³") && r.contains("41³"), "{r}");
    assert!(snap.notes.last().unwrap().contains("refused"), "{:?}", snap.notes);
    assert!(surrogate_of(&sh_d3).model().is_none());

    // ruling S18: a model taught at the interactive tol 1e-5 (a file saved before the sample
    // tolerance) is refused by a teacher at 1e-6 — both tolerances named, never mixed
    let loose = tmp("h2-tol1e-5.model");
    let mut sc = surrogate::sidecar::read(&path).unwrap();
    sc.model.provenance.tol = 1e-5;
    surrogate::sidecar::write(&loose, &sc.model, &sc.accumulator, sc.report.as_ref(), &sc.samples, sc.reference_rule.as_deref()).unwrap();
    let (p, sh_d4) = driven(surrogate_over(&Shared::new(PairMorse::h2())));
    let mut d4 = Dynamics::new(scene.clone(), p, 0.25, 1);
    let held = surrogate_of(&sh_d4).samples(); // the construction's own rule-1 answer
    d4.apply(DynCommand::LoadModel(loose.clone()));
    let r = d4.snapshot().last_rejection.expect("a model taught at another tolerance is refused");
    println!("refused: {r}");
    assert!(r.contains("at tol 1e-5") && r.contains("at tol 1e-6"), "{r}");
    assert!(surrogate_of(&sh_d4).model().is_none() && surrogate_of(&sh_d4).samples() == held, "nothing of the 1e-5 file was installed");
    let _ = std::fs::remove_file(loose);

    // a corrupt file: an error message, never a panic
    let bad = tmp("corrupt.model");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.truncate(bytes.len() / 2);
    std::fs::write(&bad, &bytes).unwrap();
    d3.apply(DynCommand::LoadModel(bad.clone()));
    let r = d3.snapshot().last_rejection.expect("refused");
    println!("corrupt: {r}");
    assert!(surrogate_of(&sh_d3).model().is_none());
    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_file(bad);
}

/// The 2026-10-20 PH₃ data loss, on the analytic teacher. A pass whose isolated-atom reference
/// stalls (deterministic) ends without a retry, no model, the reason in plain words and no Retry
/// offer; the samples taught without a model (live DFT steps) are saved as a samples-only file and
/// loaded back unfitted; there the reference is "not solved yet" and the Retry re-solves it
/// (references-only pass) and the model forms. A reference lost to the device is retried in the
/// pass itself, cold (an Activity event "(retry, cold)"), and the model forms. A file that already
/// knew every reference fits at load.
#[test]
fn an_unfitted_sample_set_survives_save_load_and_the_retry_fits_it() {
    let teacher = Shared::new(PairMorse::h2());
    let h = teacher.handle();
    h.lock().unwrap().atom_failures = usize::MAX;
    let scene = pair(1, 1, 1.45, cell(41));
    let (p, sh) = driven(surrogate_over(&teacher));
    let mut d = Dynamics::new(scene.clone(), p, 0.25, 1);
    let run_pass = |d: &mut Dynamics, plan: SamplingPlan| {
        d.apply(DynCommand::StartSampling(plan));
        for _ in 0..1000 {
            if d.tick(f64::INFINITY, &mut never()).sampling.is_none_or(|p| p.finished) {
                break;
            }
        }
    };
    let plan = SamplingPlan { perturbations: 30, hot_burst_steps: 20, scan_points_per_pair: 12, ..SamplingPlan::default() };
    // the pass: H's one multiplicity stalls — deterministic, not retried — and the pass ends
    run_pass(&mut d, plan.clone());
    let lone = |h: &Arc<Mutex<PairMorse>>| -> Vec<u32> { h.lock().unwrap().seen.iter().filter(|s| s.nuclei.len() == 1).map(|s| s.multiplicity.unwrap()).collect() };
    assert_eq!(lone(&h), vec![2], "the H scan (multiplicity 2), no retry of a stall");
    // play: rule 1 keeps the DFT answers as samples (no model to answer)
    d.apply(DynCommand::Play);
    for _ in 0..40 {
        d.tick(f64::INFINITY, &mut never());
    }
    d.apply(DynCommand::Pause);
    let n = surrogate_of(&sh).samples();
    assert!(n >= 30, "{n} samples kept without a model");
    let status = surrogate_of(&sh).status(&scene, &[[0.0; 3]; 2]);
    let words = status.no_model.clone().expect("no model: the reason is given");
    println!("{words}");
    assert!(words.contains("the H atom reference did not solve") && words.contains("SCF stalled") && words.contains("deterministic"), "{words}");
    assert!(!status.retry_fit && !status.fitted, "a stall is reproduced exactly: no Retry offered");
    d.apply(DynCommand::RetryFit);
    assert!(d.snapshot().last_rejection.expect("refused").contains("deterministic"));

    // Save works without a fit: a samples-only file; it loads back unfitted
    let path = tmp("unfitted.model");
    d.apply(DynCommand::SaveModel(path.clone()));
    assert_eq!(d.snapshot().last_rejection, None);
    assert!(d.snapshot().notes.iter().any(|m| m.contains("no model yet")), "{:?}", d.snapshot().notes);
    let teacher2 = Shared::new(PairMorse::h2());
    let h2 = teacher2.handle();
    let (p2, sh2) = driven(surrogate_over(&teacher2));
    let mut d2 = Dynamics::new(scene.clone(), p2, 0.25, 1);
    d2.apply(DynCommand::LoadModel(path.clone()));
    assert_eq!(d2.snapshot().last_rejection, None, "{:?}", d2.snapshot().notes);
    {
        let s = surrogate_of(&sh2);
        assert_eq!(s.samples(), n, "every sample came back");
        assert!(s.model().is_none() && s.reference_energy(1).is_none());
        let st = s.status(&scene, &[[0.0; 3]; 2]);
        assert!(st.no_model.as_deref().is_some_and(|w| w.contains("the H atom reference is not solved yet")), "{:?}", st.no_model);
        assert!(st.retry_fit);
    }
    // another grid refuses the file
    let (p3, sh3) = driven(surrogate_over(&Shared::new(PairMorse::h2())));
    let mut d3 = Dynamics::new(pair(1, 1, 1.45, cell(49)), p3, 0.25, 1);
    d3.apply(DynCommand::LoadModel(path.clone()));
    assert!(d3.snapshot().last_rejection.expect("another surface").contains("refused"));
    assert!(surrogate_of(&sh3).samples() <= 1);

    // the Retry: a references-only pass (just the atom: one solve), then the fit
    let before = h2.lock().unwrap().seen.len();
    d2.apply(DynCommand::RetryFit);
    assert_eq!(d2.snapshot().last_rejection, None, "{:?}", d2.snapshot().notes);
    for _ in 0..10 {
        if d2.tick(f64::INFINITY, &mut never()).sampling.is_none_or(|p| p.finished) {
            break;
        }
    }
    let solved: Vec<usize> = h2.lock().unwrap().seen[before..].iter().map(|s| s.nuclei.len()).collect();
    assert_eq!(&solved[..1], &[1], "the retry solved the lone atom first");
    {
        let s = surrogate_of(&sh2);
        assert!(s.model().is_some(), "the retry fitted the model");
        assert_eq!(s.reference_energy(1), Some(MorseDimer::h2().e_atom));
        assert!(s.status(&scene, &[[0.0; 3]; 2]).no_model.is_none());
    }
    assert!(d2.snapshot().notes.iter().any(|m| m.contains("reference retry")), "{:?}", d2.snapshot().notes);

    // a samples-only file that knew every reference fits at load; one of another rule does not
    d2.apply(DynCommand::ForgetModel);
    let surrogate::sidecar::Contents::Samples(mut set) = surrogate::sidecar::read_any(&path).unwrap() else { panic!("a samples-only file") };
    set.references = vec![(1, MorseDimer::h2().e_atom)];
    set.reference_rule = None;
    surrogate::sidecar::write_samples(&path, &set).unwrap();
    d2.apply(DynCommand::LoadModel(path.clone()));
    assert!(surrogate_of(&sh2).model().is_none() && surrogate_of(&sh2).reference_energy(1).is_none(), "pre-S9 references are not fitted on");
    set.reference_rule = Some(surrogate::sidecar::REFERENCE_RULE.into());
    surrogate::sidecar::write_samples(&path, &set).unwrap();
    d2.apply(DynCommand::LoadModel(path.clone()));
    assert!(surrogate_of(&sh2).model().is_some(), "every reference known: fitted at load");
    let _ = std::fs::remove_file(path);

    // a reference lost to the device: retried once, cold, in the pass itself; the model forms
    let teacher4 = Shared::new(PairMorse::h2());
    let h4 = teacher4.handle();
    {
        let mut t = h4.lock().unwrap();
        t.atom_failures = 1;
        t.atom_device = true;
    }
    let (p4, sh4) = driven(surrogate_over(&teacher4));
    let mut d4 = Dynamics::new(scene.clone(), p4, 0.25, 1);
    run_pass(&mut d4, plan);
    assert_eq!(lone(&h4), vec![2, 2], "the lost solve and its one cold retry");
    let mut s = surrogate_of(&sh4);
    assert!(s.model().is_some());
    let events = Trainable::take_teach_events(&mut *s);
    let refs: Vec<&TeachEvent> = events.iter().filter(|e| matches!(e.rule, TeachRule::Pass(PassJob::Reference { .. }))).collect();
    assert_eq!(refs.len(), 2, "both attempts are Activity events");
    assert_eq!((refs[0].outcome, refs[1].outcome), (TeachOutcome::Failed, TeachOutcome::Reference));
    assert!(!refs[0].detail.contains("retry") && refs[1].detail.contains("(retry, cold)"), "{:?}", refs.iter().map(|e| &e.detail).collect::<Vec<_>>());
}

/// Optional GPU smoke test (release; skips without an adapter): a tiny real pass on H₂ at 33³.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn gpu_sampling_smoke_h2_33() {
    let Some(ctx) = gpu::GpuContext::new_headless() else {
        eprintln!("sim: no GPU adapter — skipped");
        return;
    };
    // review I1: the surrogate's OWN teacher, so samples carry its true identity
    let mut teacher = crate::ks::GpuKs::new(ctx, 1e-6);
    teacher.set_display(false);
    let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_teacher_tol(1e-6);
    let scene = pair(1, 1, 1.4, cell(33));
    let plan = SamplingPlan { perturbations: 3, hot_burst_steps: 4, scan_points_per_pair: 3, ..SamplingPlan::default() };
    let t0 = std::time::Instant::now();
    let mut sampler = Sampler::new(&scene, &plan).unwrap();
    while sampler.step_own_teacher(&mut s, &mut never()).unwrap().is_some() {}
    let (n, p) = (sampler.samples_added(), sampler.progress());
    assert_eq!(s.store().unwrap().provenance.teacher, "Kohn-Sham (GPU)");
    let e_h = s.reference_energy(1).unwrap();
    println!("GPU H₂ 33³: {n} samples + 1 reference in {:.2} s; E_H(33³) = {e_h:+.6} Ha; {p:?}", t0.elapsed().as_secs_f64());
    assert_eq!((n, p.failed, p.skipped), (10, 0, 0));
    // ruling S9 on the real teacher, the trivial case: one electron allows only the doublet
    let key = atom_key(&s.provenance_for(&scene), 1);
    assert_eq!(sampler.take_references().get(&key).map(|r| r.multiplicity), Some(Some(2)));
    // the 33³ box's H atom (Task 4 measured −0.4143): confinement-raised above LSDA's −0.479
    assert!((-0.48..-0.40).contains(&e_h), "{e_h}");
    s.refit().expect("refit");
}

/// Ruling S9: the reference is the lowest-energy multiplicity of the same scan for every Z.
/// A scripted O whose triplet lies lowest (−1.0 vs singlet −0.8, quintet −0.5) gets the
/// triplet; H has only its doublet; every multiplicity is solved cold; an abort mid-scan
/// resumes where it stopped; the cache keeps the winner and its multiplicity.
#[test]
fn the_reference_is_the_lowest_energy_multiplicity() {
    assert_eq!(allowed_multiplicities(1), vec![2]);
    assert_eq!(allowed_multiplicities(2), vec![1, 3]);
    assert_eq!(allowed_multiplicities(3), vec![2, 4]);
    assert_eq!(allowed_multiplicities(6), vec![1, 3, 5]);
    assert_eq!(allowed_multiplicities(8), vec![1, 3, 5]);
    let mut pm = PairMorse::h2();
    pm.atom_energy = vec![(8, 1, -0.8), (8, 3, -1.0), (8, 5, -0.5)];
    let mut teacher = Shared::new(pm);
    let h = teacher.handle();
    let mut s = surrogate_over(&teacher);
    let scene = pair(1, 8, 1.8, cell(41));
    let plan = SamplingPlan { perturbations: 2, hot_burst_steps: 0, scan_points_per_pair: 0, ..SamplingPlan::default() };
    let mut sampler = Sampler::new(&scene, &plan).unwrap();
    assert_eq!(sampler.progress().total, 1 + 3 + 2, "one job per (Z, multiplicity)");
    // H, then O at multiplicity 1; abort inside O's multiplicity-3 solve
    for _ in 0..2 {
        sampler.step(&mut teacher, &mut s, &mut never()).unwrap().unwrap();
    }
    assert!(matches!(sampler.step(&mut teacher, &mut s, &mut || true), Err(ProviderError::Cancelled)));
    while sampler.step(&mut teacher, &mut s, &mut never()).unwrap().is_some() {}
    let notes = sampler.take_notes();
    println!("{notes:#?}");
    assert_eq!(s.reference_energy(8), Some(-1.0), "the triplet is lowest and wins");
    assert_eq!(s.reference_energy(1), Some(-0.478));
    assert!(notes.iter().any(|n| n.contains("reference O (Z = 8): multiplicity 3 (E = -1.000000 Ha) chosen over 1 (-0.800000), 5 (-0.500000)")), "{notes:?}");
    assert!(notes.iter().any(|n| n.contains("reference H (Z = 1): multiplicity 2") && n.contains("the only one solved")), "{notes:?}");
    let t = h.lock().unwrap();
    let atoms: Vec<(u8, Option<u32>)> = t.seen.iter().filter(|x| x.nuclei.len() == 1).map(|x| (x.nuclei[0].z, x.multiplicity)).collect();
    assert_eq!(atoms, vec![(1, Some(2)), (8, Some(1)), (8, Some(3)), (8, Some(5))], "each multiplicity once, the aborted one resumed");
    assert!(t.seen.iter().all(|x| x.charge == 0 || x.nuclei.len() > 1));
    // a reset before every atom solve attempt (cold; the aborted one included), one more when
    // the molecule follows
    assert_eq!(t.resets, 4 + 1 + 1);
    drop(t);
    let cache = sampler.take_references();
    let prov = s.provenance_for(&scene);
    let entry = |z: u8| cache.get(&atom_key(&prov, z)).map(|r| (r.energy, r.multiplicity, r.tol, r.provider.clone(), r.radial.is_empty()));
    // an analytic teacher has no density: the table stays empty; tol and the teacher are the key's
    assert_eq!(entry(8), Some((-1.0, Some(3), prov.tol, prov.teacher.clone(), true)));
    assert_eq!(entry(1), Some((-0.478, Some(2), prov.tol, prov.teacher.clone(), true)));
    let mut again = Sampler::new(&scene, &plan).unwrap().with_references(cache);
    while again.step(&mut teacher, &mut s, &mut never()).unwrap().is_some() {}
    assert!(again.take_notes().iter().any(|n| n.contains("reference O (Z = 8) from the cache: E = -1.000000 Ha, multiplicity 3")));
    assert_eq!(again.progress().done, again.progress().total);
    assert_eq!(h.lock().unwrap().seen.iter().filter(|x| x.nuclei.len() == 1).count(), 4, "no atom solve the second time");
}

/// Review I1: a teacher other than the surrogate's own is refused (its answers would be filed
/// under the surrogate's teacher's provenance).
#[test]
fn a_teacher_that_is_not_the_surrogates_own_is_refused() {
    let own = Shared::new(PairMorse::h2());
    let mut s = surrogate_over(&own);
    let scene = pair(1, 1, 1.4, cell(41));
    let mut other = MorseDimer::h2();
    let err = run_sampling(&scene, &SamplingPlan::default(), &mut other, &mut s, &mut |_| {}, &mut never()).unwrap_err();
    println!("{err}");
    assert!(matches!(err, ProviderError::Invalid(ref m) if m.contains("'morse dimer' is not the surrogate's teacher 'pair morse'")), "{err}");
    assert_eq!(s.samples(), 0);
    assert_eq!(other.calls, 0, "refused before any solve");
    assert!(own.handle().lock().unwrap().seen.is_empty());
}

/// Review m1 and the commands that must not disturb a running pass: a rejected ReplaceScene
/// leaves the pass running with no refit; LoadModel is refused while sampling; an accepted
/// ReplaceScene cancels the pass (and refits what it gathered).
#[test]
fn a_rejected_replace_scene_or_a_load_leaves_the_pass_running() {
    let teacher = Shared::new(PairMorse::h2());
    let (p, sh_d) = driven(surrogate_over(&teacher));
    let mut d = Dynamics::new(pair(1, 1, 1.45, cell(41)), p, 0.25, 1);
    d.apply(DynCommand::StartSampling(SamplingPlan { perturbations: 20, hot_burst_steps: 0, scan_points_per_pair: 0, ..SamplingPlan::default() }));
    d.tick(0.0, &mut never());
    d.tick(0.0, &mut never());
    let mut bad = pair(1, 1, 1.45, cell(41));
    bad.nuclei[0].pos[0] = f64::NAN;
    d.apply(DynCommand::ReplaceScene(bad));
    let snap = d.snapshot();
    assert!(snap.last_rejection.is_some());
    let p = snap.sampling.unwrap();
    assert!(!p.finished && !p.cancelled && p.done == 2, "{p:?}");
    assert!(surrogate_of(&sh_d).model().is_none(), "no refit from a rejected command");
    d.apply(DynCommand::LoadModel(tmp("never-read.model")));
    let r = d.snapshot().last_rejection.expect("LoadModel is refused while sampling");
    assert!(r.contains("cancel the running sampling pass"), "{r}");
    assert!(!d.snapshot().sampling.unwrap().finished);
    d.tick(0.0, &mut never());
    assert_eq!(d.snapshot().sampling.unwrap().done, 3, "the pass goes on");
    d.apply(DynCommand::ReplaceScene(pair(1, 1, 1.5, cell(41))));
    let snap = d.snapshot();
    assert_eq!(snap.last_rejection, None);
    let p = snap.sampling.unwrap();
    assert!(p.finished && p.cancelled, "an accepted new scene ends the pass: {p:?}");
    assert!(surrogate_of(&sh_d).model().is_none(), "what it gathered is kept, not refitted below Q′ (fix round 2)");
    assert!(surrogate_of(&sh_d).samples() > 0);
}

/// Ruling S9 and old files: a sidecar without the reference rule loads (its fit is consistent
/// with its references) but its references are not cached — the next pass re-solves them —
/// and the model is saved with the rule only once every reference is S9.
#[test]
fn references_from_a_file_without_the_reference_rule_are_re_solved() {
    let teacher = Shared::new(PairMorse::h2());
    let scene = pair(1, 1, 1.45, cell(41));
    let (p, sh_d) = driven(surrogate_over(&teacher));
    let mut d = Dynamics::new(scene.clone(), p, 0.25, 1);
    d.apply(DynCommand::StartSampling(SamplingPlan { perturbations: 30, hot_burst_steps: 20, scan_points_per_pair: 12, ..SamplingPlan::default() }));
    while !d.tick(f64::INFINITY, &mut never()).sampling.unwrap().finished {}
    // a pre-S9 file: the same contents, no reference rule
    let old = tmp("pre-s9.model");
    {
        let s = surrogate_of(&sh_d);
        surrogate::sidecar::write(&old, s.model().unwrap(), s.accumulator().unwrap(), s.report(), &s.store().unwrap().samples, None).unwrap();
        let marked = tmp("s9.model");
        s.save_sidecar(&marked).unwrap();
        assert_eq!(surrogate::sidecar::read(&marked).unwrap().reference_rule.as_deref(), Some(surrogate::sidecar::REFERENCE_RULE));
    }
    let teacher2 = Shared::new(PairMorse::h2());
    let h2 = teacher2.handle();
    let (p, sh_d2) = driven(surrogate_over(&teacher2));
    let mut d2 = Dynamics::new(scene.clone(), p, 0.25, 1);
    d2.apply(DynCommand::LoadModel(old.clone()));
    let snap = d2.snapshot();
    assert_eq!(snap.last_rejection, None);
    assert!(snap.notes.iter().any(|n| n.contains("predate ruling S9")), "{:?}", snap.notes);
    assert_eq!(surrogate_of(&sh_d2).legacy_references(), vec![1]);
    // saving now would not vouch for the references
    let resaved = tmp("resaved.model");
    d2.apply(DynCommand::SaveModel(resaved.clone()));
    assert_eq!(surrogate::sidecar::read(&resaved).unwrap().reference_rule, None);
    // the next pass solves the atom again (nothing cached from the file)
    d2.apply(DynCommand::StartSampling(SamplingPlan { perturbations: 1, hot_burst_steps: 0, scan_points_per_pair: 0, ..SamplingPlan::default() }));
    while !d2.tick(f64::INFINITY, &mut never()).sampling.unwrap().finished {}
    assert_eq!(h2.lock().unwrap().seen.iter().filter(|x| x.nuclei.len() == 1).count(), 1, "the reference was re-solved");
    assert!(surrogate_of(&sh_d2).legacy_references().is_empty());
    d2.apply(DynCommand::SaveModel(resaved.clone()));
    assert_eq!(surrogate::sidecar::read(&resaved).unwrap().reference_rule.as_deref(), Some(surrogate::sidecar::REFERENCE_RULE));
}

/// The teacher of `gpu_oxygen_reference_multiplicity`, recording every evaluation it answers (multiplicity, energy or
/// error, the solve's stats) and forwarding everything else, the ionic tag included.
struct Recording {
    inner: crate::ks::GpuKs,
    seen: Arc<Mutex<Vec<(Option<u32>, Result<(f64, EvalStats), String>)>>>,
}

impl ForceProvider for Recording {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<sim::ForceResult, ProviderError> {
        let r = self.inner.evaluate(scene, should_abort);
        let entry = match &r {
            Ok(r) => Ok((r.energy, r.stats.clone())),
            Err(e) => Err(e.to_string()),
        };
        self.seen.lock().unwrap().push((scene.multiplicity, entry));
        r
    }
    fn name(&self) -> &'static str {
        self.inner.name()
    }
    fn reset(&mut self) {
        self.inner.reset()
    }
    fn discontinuity(&mut self) {
        self.inner.discontinuity()
    }
    fn sample_density(&mut self, points: &[[f64; 3]]) -> Option<Vec<f64>> {
        self.inner.sample_density(points)
    }
    fn scf_tol(&self) -> Option<f64> {
        self.inner.scf_tol()
    }
    fn ionic_tag(&self, scene: &Scene) -> String {
        self.inner.ionic_tag(scene)
    }
}

/// Ruling S9 on the real teacher, under the pseudopotentials (M7 Task 10; spec §6.4's O row, Task 6 review A.7): an
/// isolated O atom solved by the sampling pass's reference scan at multiplicities 1, 3, 5 under `CoresOnly` (pseudo-O,
/// `DegenerateShells`: both multiplicities are the spherical ensemble), h = O's h_req (0.20), a 14 Bohr box (71³; the
/// 12 Bohr wall bias of a −0.27 Ha level is 0.3 mHa, Task 6 review V1), the atom at the cell centre (the scan's
/// placement), GPU, tol 1e-6, each multiplicity cold. Asserted (A.7): M = 1 and M = 3 converged (an unconverged solve is
/// the scan's failure), every shared cluster's spread < τ/10, each within 2 mHa of the radial KB atom on the same filtered
/// table in the same configuration (M = 1: the spin-restricted reference configuration; M = 3: T3's Hund configuration),
/// E(3) < E(1) − 20 mHa, and the scan's reference is the triplet — Hund's rule emerges (rule 1: the same scan for every
/// Z). The same A.7 form is asserted for the atom off a grid point (A.7's placement; Task 10 fix round), solved directly
/// on a fresh teacher at M = 1, 3, 5 cold. Reported: the splitting against the radial value and the all-electron
/// spherical 54.14 mHa (references.md "M7 radial atom"), M = 5, and the on- vs off-grid energy differences. The 56.07
/// against 54.14 mHa is the c = ¼ table's documented T3 Hund error (−1.931 mHa, NLCC ruling "Cost"), not a
/// non-spherical atom: these solves are the spherical ensemble (physics review SHOULD-3).
///
/// **What it replaces (M6, all-electron, `IonicPolicy::None`, h = σ = 0.25; the test was ignored with these):** at 33³
/// (26 s for the three cold solves) the singlet was lowest, E = −36.5650 Ha, the triplet −36.3380 (+0.227 Ha), the
/// quintet −35.5761; at 57³ the singlet was −36.5787 Ha and the triplet and quintet did not converge cold in 200 SCF
/// iterations (Δρ 1.4e-4 / 1.8e-3) — the smeared open-shell atom with integer occupations, a solver question that M7's
/// pseudopotential and spherical occupations answer.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only (GPU, 71³): cargo test -p providers --release gpu_oxygen_reference_multiplicity")]
fn gpu_oxygen_reference_multiplicity() {
    let Some(ctx) = gpu::GpuContext::new_headless() else {
        eprintln!("sim: no GPU adapter — skipped");
        return;
    };
    let table = crate::pseudo::table_for(8).expect("O is accepted");
    let h = table.record.h_req.expect("O has h_req").min(0.25);
    let n = 2 * (14.0 / (2.0 * h)).round() as usize + 1;
    let mut inner = crate::ks::GpuKs::new(ctx.clone(), 1e-6);
    inner.set_display(false);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let teacher = Recording { inner, seen: seen.clone() };
    let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_teacher_tol(1e-6);
    let scene = Scene { nuclei: vec![Nucleus::new(8, [0.0; 3])], cell: Cell { n, h }, charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3], ionic: sim::IonicPolicy::CoresOnly };
    let plan = SamplingPlan { perturbations: 0, hot_burst_steps: 0, scan_points_per_pair: 0, ..SamplingPlan::default() };
    let t0 = std::time::Instant::now();
    let mut sampler = Sampler::new(&scene, &plan).unwrap();
    while sampler.step_own_teacher(&mut s, &mut never()).unwrap().is_some() {}
    let notes = sampler.take_notes();
    println!("O atom (CoresOnly) at {n}³ × {h}, 3 multiplicities in {:.2} s: {notes:#?}", t0.elapsed().as_secs_f64());
    let key = atom_key(&s.provenance_for(&scene), 8);
    let refs = sampler.take_references();
    let r = refs.get(&key).expect("a reference");
    // the radial references on the table the solves used, filtered at h (qm::radial::kb, as `validate::pseudo`'s CPU gate)
    let ft = qm::radial::filtered(&table, h);
    let hund_ae = qm::radial::ground_state(8, qm::radial::SpinMode::Polarised).expect("the AE O ground state");
    let hund = qm::radial::transfer_configs(&table, &hund_ae).into_iter().find(|c| c.name == "Hund").expect("T3's Hund configuration");
    let r1 = qm::radial::kb::solve_kb_atom(&ft.grid, &ft.input(), &qm::radial::kb::reference_config(&table.partition), qm::radial::SpinMode::Restricted).unwrap().energy;
    let r3 = qm::radial::kb::solve_kb_atom(&ft.grid, &ft.input(), &hund.pp, qm::radial::SpinMode::Polarised).unwrap().energy;
    // A.7's form for one placement: (multiplicity, energy or error, stats) per solve
    let a7 = |label: &str, seen: &[(Option<u32>, Result<(f64, EvalStats), String>)]| -> (f64, f64) {
        let solved = |m: u32| -> Option<(f64, &EvalStats)> { seen.iter().find(|(mm, _)| *mm == Some(m)).and_then(|(_, r)| r.as_ref().ok()).map(|(e, st)| (*e, st)) };
        for (m, r) in seen.iter() {
            match r {
                Ok((e, st)) => println!("  {label} M = {m:?}: E = {e:.6} Ha, {} SCF steps, tol used {:?}, shared {:?}, warnings {:?}", st.scf_steps, st.scf_tol_used, st.fractional, st.warnings),
                Err(e) => println!("  {label} M = {m:?}: failed: {e}"),
            }
        }
        let (e1, st1) = solved(1).unwrap_or_else(|| panic!("{label}: the singlet did not converge"));
        let (e3, st3) = solved(3).unwrap_or_else(|| panic!("{label}: the triplet did not converge"));
        println!("{label}: M = 1 {e1:.6} vs radial {r1:.6} (Δ {:+.2} mHa); M = 3 {e3:.6} vs radial {r3:.6} (Δ {:+.2} mHa); splitting E(1) − E(3) = {:.3} mHa (radial {:.3}, all-electron spherical 54.14); M = 5: {:?}",
                 (e1 - r1) * 1e3, (e3 - r3) * 1e3, (e1 - e3) * 1e3, (r1 - r3) * 1e3, solved(5).map(|x| x.0));
        for (m, st) in [(1, st1), (3, st3)] {
            for c in &st.fractional {
                assert!(c.spread < qm::DEGENERACY_TAU / 10.0, "{label} M = {m}: shared cluster {c:?}");
            }
        }
        assert!((e1 - r1).abs() < 2e-3, "{label} M = 1: {e1:.6} vs radial {r1:.6}");
        assert!((e3 - r3).abs() < 2e-3, "{label} M = 3: {e3:.6} vs radial {r3:.6}");
        assert!(e3 < e1 - 0.020, "{label}: triplet {e3:.6} vs singlet {e1:.6}: Hund's splitting > 20 mHa");
        (e1, e3)
    };
    let (on1, on3) = a7("on-grid (the scan)", &seen.lock().unwrap()[..]);
    println!("O reference: multiplicity {:?}, E = {:.6} Ha", r.multiplicity, r.energy);
    assert_eq!(r.multiplicity, Some(3), "Hund's rule emerges for O: {notes:?}");
    // A.7 asks for the atom off a grid point (Task 10 fix round): the same solves, cold, at a low-symmetry offset
    // (`validate::pseudo`'s CPU gate position), straight on a fresh teacher
    let mut off = crate::ks::GpuKs::new(ctx.clone(), 1e-6);
    off.set_display(false);
    let mut off_seen = Vec::new();
    for m in [1u32, 3, 5] {
        off.reset();
        let sc = Scene { nuclei: vec![Nucleus::new(8, [0.013, -0.0091, 0.0052])], multiplicity: Some(m), ..scene.clone() };
        let res = off.evaluate(&sc, &mut never());
        off_seen.push((Some(m), res.map(|r| (r.energy, r.stats)).map_err(|e| e.to_string())));
    }
    let (off1, off3) = a7("off-grid (0.013, −0.0091, 0.0052)", &off_seen);
    println!("on- vs off-grid: ΔE(M = 1) {:+.2e} Ha, ΔE(M = 3) {:+.2e} Ha (the egg-box energy of one atom)", off1 - on1, off3 - on3);
}

/// Task 9 item 4: `scaled_to` shrinks the plan by one factor to fit a solve budget, never below
/// the samples the first fit needs to be able to earn trust, and leaves a plan that fits alone.
#[test]
fn a_plan_is_scaled_to_a_solve_budget_but_never_below_what_trust_needs() {
    let h2 = pair(1, 1, 1.6, cell(41));
    let plan = SamplingPlan::default();
    let full = plan.solves(&h2).unwrap();
    let floor = SamplingPlan::min_samples_for_trust(&h2);
    let refs = SamplingPlan { perturbations: 0, scan_points_per_pair: 0, hot_burst_steps: 0, ..plan.clone() }.solves(&h2).unwrap();
    println!("H₂: default plan {full} solves ({refs} references); trust needs ≥ {floor} samples");
    assert_eq!(full, 1 + 60 + 12 + 200);
    assert_eq!(plan.scaled_to(&h2, full), plan, "a plan that fits is unchanged");
    let half = plan.scaled_to(&h2, 140);
    let n = half.solves(&h2).unwrap();
    println!("scaled to ≤ 140: {n} solves ({} perturbations, {} scan points, {} burst steps)", half.perturbations, half.scan_points_per_pair, half.hot_burst_steps);
    assert!(n <= 140 && n >= 130, "{n}");
    assert_eq!((half.perturbation_sigma, half.hot_burst_t, half.scan_range, half.seed), (plan.perturbation_sigma, plan.hot_burst_t, plan.scan_range, plan.seed));
    // a budget below what trust needs: the plan that just reaches it
    let tiny = plan.scaled_to(&h2, 10);
    let n = tiny.solves(&h2).unwrap();
    println!("scaled to ≤ 10: {n} solves (floor {floor} + {refs} references)");
    assert!(n - refs >= floor && n - refs <= floor + 5, "{n}");
}

/// M6 Task 1: the pass runs inside the provider, through the `Trainable` trait — what
/// `Dynamics` now does. A full pass ends with `Ok(None)`, finished, every non-reference solve a
/// sample and the model refitted (noted); a second pass takes the reference from the provider's
/// cache; `refs` seeds a fresh provider's cache; a cancelled pass is no longer active and its
/// progress says cancelled; one pass at a time.
#[test]
fn a_pass_runs_through_the_trainable_trait() {
    let teacher = Shared::new(PairMorse::h2());
    let h = teacher.handle();
    let mut s = surrogate_over(&teacher);
    let scene = pair(1, 1, 1.45, cell(41));
    let plan = SamplingPlan { perturbations: 30, hot_burst_steps: 20, scan_points_per_pair: 12, seed: 3, ..SamplingPlan::default() };
    let solves = plan.solves(&scene).unwrap();
    let references = 1; // H: the doublet only
    assert_eq!(solves, references + 30 + 12 + 20);
    let t: &mut dyn Trainable = &mut s;
    assert!(!t.pass_active() && t.progress().is_none());
    t.start_pass(&scene, &plan, &mut AtomReferences::default()).unwrap();
    assert!(t.pass_active());
    assert_eq!(t.progress().map(|p| (p.done, p.total)), Some((0, solves)));
    assert!(t.start_pass(&scene, &plan, &mut AtomReferences::default()).is_err(), "one pass at a time");
    let mut steps = 0;
    while let Some(p) = t.pass_step(&mut never()).unwrap() {
        steps += 1;
        assert_eq!(p.done, steps);
    }
    let p = t.progress().expect("the finished pass's progress");
    assert!(p.finished && !p.cancelled && p.done == p.total, "{p:?}");
    assert!(!t.pass_active());
    let notes = t.take_notes();
    println!("{notes:#?}");
    assert!(notes.first().is_some_and(|n| n.starts_with(&format!("sampling pass started: {solves} jobs (1 pair scans)"))), "{notes:?}");
    assert!(notes.last().is_some_and(|n| n.starts_with(&format!("sampling pass finished: {} samples added", solves - references))), "{notes:?}");
    assert!(t.take_notes().is_empty(), "drained");
    let st = t.status(&scene, &[]);
    assert_eq!(st.samples, solves - references);
    assert!(st.fitted && st.trusted, "{st:?}");
    // a second pass: the reference from the provider's cache, no atom solve
    let atoms = || h.lock().unwrap().seen.iter().filter(|x| x.nuclei.len() == 1).count();
    assert_eq!(atoms(), 1);
    let short = SamplingPlan { perturbations: 2, hot_burst_steps: 0, scan_points_per_pair: 0, ..plan.clone() };
    t.start_pass(&scene, &short, &mut AtomReferences::default()).unwrap();
    while t.pass_step(&mut never()).unwrap().is_some() {}
    assert_eq!(atoms(), 1, "the cached reference cost no solve");
    // `refs` seeds a fresh provider's cache
    let mut seeded = AtomReferences::default();
    let key = atom_key(&surrogate_over(&teacher).provenance_for(&scene), 1);
    seeded.seed_energy(key.clone(), -0.478, Some(2), key.tol, &key.teacher);
    let mut fresh = surrogate_over(&teacher);
    fresh.start_pass(&scene, &short, &mut seeded).unwrap();
    while Trainable::pass_step(&mut fresh, &mut never()).unwrap().is_some() {}
    assert_eq!(atoms(), 1, "the seeded reference cost no solve");
    assert_eq!(fresh.reference_energy(1), Some(-0.478));
    // cancelled mid-way: no longer active, the progress says so, the samples so far refitted
    // (the first step takes the cached reference and solves the first perturbation)
    let t: &mut dyn Trainable = &mut s;
    t.start_pass(&scene, &plan, &mut AtomReferences::default()).unwrap();
    t.take_notes();
    t.pass_step(&mut never()).unwrap();
    t.pass_step(&mut never()).unwrap();
    t.cancel_pass();
    assert!(!t.pass_active());
    let p = t.progress().unwrap();
    assert!(p.cancelled && p.finished && p.done == 3, "{p:?}");
    let notes = t.take_notes();
    assert!(notes.last().is_some_and(|n| n.starts_with("sampling pass cancelled: 2 samples added")), "{notes:?}");
    t.cancel_pass(); // no-op without a pass
    assert!(t.take_notes().is_empty());
}

/// M6 Task 9: `Sampler::rescale` shrinks a running pass by dropping jobs not yet run — the
/// references are kept, nothing already solved is undone, and the pass never shrinks below the
/// samples the first fit needs to earn trust (`min_samples_for_trust`). Rescaling H₂'s default
/// pass to 3 solves (as the plan `scaled_to(.., 3)` and as a raw 1 + 1 + 1 plan) leaves exactly
/// the trust floor + the reference; a rescale after the reference and two perturbations keeps
/// those, the pass then runs to exactly its new total, and a rescale to a larger plan grows
/// nothing.
#[test]
fn rescale_never_drops_references_or_goes_below_the_trust_minimum() {
    let h2 = pair(1, 1, 1.6, cell(41));
    let plan = SamplingPlan::default();
    let floor = SamplingPlan::min_samples_for_trust(&h2);
    let refs = 1; // H: the doublet only
    let full = plan.solves(&h2).unwrap();
    assert!(floor > 3 && floor + refs < full, "test assumption: floor {floor}, full {full}");
    let raw = SamplingPlan { perturbations: 1, scan_points_per_pair: 1, hot_burst_steps: 1, ..plan.clone() };
    for (what, small) in [("scaled_to(3)", plan.scaled_to(&h2, 3)), ("1 + 1 + 1", raw)] {
        let mut s = Sampler::new(&h2, &plan).unwrap();
        s.rescale(&small);
        let p = s.progress();
        println!("{what}: {full} → {} jobs (floor {floor} + {refs} reference)", p.total);
        assert_eq!(p.total, 3usize.max(floor) + refs, "{what}");
        assert_eq!((p.done, p.finished), (0, false));
        assert!(matches!(s.queue.front(), Some(Job::Reference(scan)) if scan.z() == 1), "the reference stays first");
    }

    // mid-pass: the reference and two perturbations solved, then the rescale; the pass runs to
    // exactly its new total with every kept job solved once and the run ones kept
    let teacher = Shared::new(PairMorse::h2());
    let h = teacher.handle();
    let mut sur = surrogate_over(&teacher);
    let mut s = Sampler::new(&h2, &plan).unwrap();
    for _ in 0..3 {
        s.step_own_teacher(&mut sur, &mut never()).unwrap();
    }
    assert_eq!(s.progress().done, 3);
    let budget = 120;
    let scaled = plan.scaled_to(&h2, budget);
    s.rescale(&scaled);
    let p = s.progress();
    println!("after 3 solves, rescaled to ≤ {budget}: {} jobs ({} perturbations, {} scan points, {} burst steps)", p.total, scaled.perturbations, scaled.scan_points_per_pair, scaled.hot_burst_steps);
    assert_eq!(p.total, scaled.solves(&h2).unwrap(), "the plan's own count: nothing run was beyond it");
    assert!(p.total <= budget && p.total - refs >= floor);
    let mut steps = 0;
    while let Some(q) = s.step_own_teacher(&mut sur, &mut never()).unwrap() {
        steps += 1;
        assert_eq!(q.total, p.total);
    }
    let fin = s.progress();
    assert!(fin.finished && fin.done == fin.total && fin.total == p.total, "{fin:?}");
    assert_eq!(steps + 3, p.total);
    assert_eq!(h.lock().unwrap().seen.len(), p.total, "one solve per kept job");
    let c = tag_counts(&sur);
    assert_eq!((c[1], c[3], c[2]), (scaled.perturbations, scaled.scan_points_per_pair, scaled.hot_burst_steps), "{c:?}");
    // the kept scan points span the planned range: both ends of the 0.7 … 1.8 r₀ scan
    let scan_r: Vec<f64> = sur.store().unwrap().samples.iter().filter(|x| x.tag == SampleTag::Scan).map(|x| {
        (0..3).map(|a| (x.pos[1][a] - x.pos[0][a]).powi(2)).sum::<f64>().sqrt()
    }).collect();
    println!("kept scan separations {scan_r:.3?}");
    assert!((scan_r[0] - 0.7 * 1.6).abs() < 1e-12 && (scan_r[scan_r.len() - 1] - 1.8 * 1.6).abs() < 1e-12, "{scan_r:?}");

    // a rescale never grows a pass
    let mut s = Sampler::new(&h2, &plan).unwrap();
    s.rescale(&SamplingPlan { perturbations: 1000, ..plan.clone() });
    assert_eq!(s.progress().total, full);
}

/// Review M4: the exact trim to the trust floor takes the burst's last steps, then the last
/// queued perturbations — never a scan end while either is left. A plan with no burst whose
/// common factor overshoots the floor (60 perturbations + 60 scan points: the counts rise two
/// at a time, the floor 59 is odd) is trimmed by one perturbation, and both ends of the 0.7 …
/// 1.8 r₀ scan are still solved.
#[test]
fn the_trim_to_the_trust_floor_keeps_both_scan_ends() {
    let h2 = pair(1, 1, 1.6, cell(41));
    let floor = SamplingPlan::min_samples_for_trust(&h2);
    let plan = SamplingPlan { perturbations: 60, scan_points_per_pair: 60, hot_burst_steps: 0, ..SamplingPlan::default() };
    let teacher = Shared::new(PairMorse::h2());
    let mut sur = surrogate_over(&teacher);
    let mut s = Sampler::new(&h2, &plan).unwrap();
    s.rescale(&SamplingPlan { perturbations: 1, scan_points_per_pair: 1, hot_burst_steps: 0, ..plan.clone() });
    assert_eq!(s.progress().total, floor + 1, "the floor + the reference, exactly");
    while s.step_own_teacher(&mut sur, &mut never()).unwrap().is_some() {}
    let c = tag_counts(&sur);
    let scan_r: Vec<f64> = sur.store().unwrap().samples.iter().filter(|x| x.tag == SampleTag::Scan).map(|x| {
        (0..3).map(|a| (x.pos[1][a] - x.pos[0][a]).powi(2)).sum::<f64>().sqrt()
    }).collect();
    println!("floor {floor}: {} perturbations, {} scan points, ends {:.3} … {:.3}", c[1], c[3], scan_r[0], scan_r[scan_r.len() - 1]);
    assert_eq!(c[1] + c[3], floor);
    assert!((scan_r[0] - 0.7 * 1.6).abs() < 1e-12 && (scan_r[scan_r.len() - 1] - 1.8 * 1.6).abs() < 1e-12, "{scan_r:?}");
}

/// Final review M3: a rescale knows the shell check. (i) Shrinking a crossing pair's scan (a plan
/// above the trust floor, so the scan is really thinned) keeps its one pending shell check —
/// neither thinned as a scan point nor counted as run — and the pair's new count is the kept
/// points plus the check. (ii) The exact trim to the floor never pops a pending shell check from
/// the queue's tail: three H atoms whose last pair (2.85 Bohr) crosses 5/6·r_cut only at its last
/// point, so that pair's shell check is the queue's last job; a scans-only plan re-sized under the
/// floor (59) is raised by the common factor to 3 × 20 = 60 and must be trimmed by one — the
/// trim stops at the shell check (the pass stays one over the floor) and the pass solves it.
#[test]
fn a_rescale_keeps_the_pending_shell_check() {
    let r_cut = 6.0;
    let shells = |s: &Sampler| s.queue.iter().filter(|j| matches!(j, Job::Shell(..))).count();
    let scan_jobs = |s: &Sampler, pair: usize| s.queue.iter().filter(|j| matches!(j, Job::Scan(_, p, _) | Job::Skip("scans", Some((p, _)), _) if *p == pair)).count();
    // (i) thinning
    let dimer = pair(1, 1, 3.3, cell(65));
    let floor = SamplingPlan::min_samples_for_trust(&dimer);
    let plan = SamplingPlan { perturbations: floor, hot_burst_steps: 0, scan_points_per_pair: 12, ..SamplingPlan::default() };
    let mut s = Sampler::with_cutoff(&dimer, &plan, r_cut).unwrap();
    assert_eq!((shells(&s), s.scan_planned[0], scan_jobs(&s, 0)), (1, 12, 11), "premise: one shell check among the 12 planned points");
    s.rescale(&SamplingPlan { scan_points_per_pair: 4, ..plan.clone() });
    println!("(i) scan jobs 11 → {}, shell checks {}, planned {}", scan_jobs(&s, 0), shells(&s), s.scan_planned[0]);
    assert_eq!(shells(&s), 1, "the shell check survives the thinning");
    assert_eq!((scan_jobs(&s, 0), s.scan_planned[0], s.plan.scan_points_per_pair), (3, 4, 4), "the kept points plus the check");
    assert_eq!(s.progress().total, 1 + floor + 4, "the reference, the perturbations, 3 points + the check");
    // (ii) the tail trim
    let three = Scene {
        nuclei: vec![Nucleus::new(1, [0.0, 0.0, 0.0]), Nucleus::new(1, [1.5, 0.0, 0.0]), Nucleus::new(1, [1.5, 2.85, 0.0])],
        ..pair(1, 1, 1.0, cell(65))
    };
    assert_eq!(SamplingPlan::min_samples_for_trust(&three), 59, "the premise's floor");
    let plan = SamplingPlan { perturbations: 0, hot_burst_steps: 0, scan_points_per_pair: 24, ..SamplingPlan::default() };
    let mut t = Sampler::with_cutoff(&three, &plan, r_cut).unwrap();
    assert_eq!(t.scans.len(), 3);
    assert!(matches!(t.queue.back(), Some(Job::Shell(_, 2, 23))), "premise: the last pair's shell check is its last point and the queue's tail");
    let shells_before = shells(&t);
    t.rescale(&SamplingPlan { scan_points_per_pair: 1, ..plan.clone() });
    println!("(ii) shell checks {shells_before} → {}, planned per pair {:?}, total {}", shells(&t), t.scan_planned, t.progress().total);
    assert_eq!(t.scan_planned, vec![20, 20, 20], "raised to the floor by the common factor");
    assert_eq!(shells(&t), shells_before, "the trim never pops a pending shell check");
    assert!(matches!(t.queue.back(), Some(Job::Shell(_, 2, 23))), "the shell check is still the tail");
    assert_eq!(t.progress().total, 1 + 60, "one over the floor: the shell check is not trimmed");
    let teacher = Shared::new(PairMorse::h2());
    let mut sur = surrogate_over(&teacher);
    while t.step_own_teacher(&mut sur, &mut never()).unwrap().is_some() {}
    let checks = t.take_shell_checks();
    assert_eq!(checks.len(), shells_before, "the pass solved its shell checks");
    let last = checks.last().unwrap();
    assert!((last.r - 1.8 * 2.85).abs() < 1e-9 && last.pair == (1, 1), "{:?}", (last.r, last.pair));
}

/// Review I2 and M1: the re-sizing goes through the driver (`Dynamics::rescale_sampling`) and the
/// `SharedTrainable` wrapper forwards `rescale_pass`. The published total is the new one at once
/// and stays so through ticks that run no pass step — an aborted tick and an edit's
/// re-evaluation at a zero budget — before the pass resumes at the new total.
#[test]
fn a_rescale_through_the_driver_publishes_the_new_total_at_once_and_keeps_it() {
    let teacher = Shared::new(PairMorse::h2());
    let scene = pair(1, 1, 1.45, cell(41));
    let (p, _h) = driven(surrogate_over(&teacher));
    let mut d = Dynamics::new(scene, p, 0.25, 1);
    let plan = SamplingPlan::default();
    d.apply(DynCommand::StartSampling(plan.clone()));
    for _ in 0..3 {
        d.tick(0.0, &mut never());
    }
    let before = d.snapshot().sampling.unwrap();
    assert_eq!((before.done, before.total), (3, 273));
    let small = plan.scaled_to(&d.snapshot().scene, 120);
    d.rescale_sampling(&small);
    let total = d.snapshot().sampling.unwrap().total;
    assert_eq!(total, small.solves(&d.snapshot().scene).unwrap(), "through SharedTrainable, at once");
    assert!(total < 273);
    // an aborted tick: no pass step
    let snap = d.tick(f64::INFINITY, &mut || true);
    assert_eq!(snap.sampling.as_ref().map(|p| (p.done, p.total)), Some((3, total)));
    // an edit: the tick re-evaluates and, at a zero budget, runs no pass step
    let mut pos = d.snapshot().scene.nuclei[0].pos;
    pos[1] += 0.05;
    d.apply(DynCommand::MoveNucleusDiscontinuous { index: 0, pos });
    let snap = d.tick(0.0, &mut never());
    assert!(!snap.forces_stale);
    assert_eq!(snap.sampling.as_ref().map(|p| (p.done, p.total)), Some((3, total)), "the edit's tick ran no pass step and kept the new total");
    let snap = d.tick(0.0, &mut never());
    assert_eq!(snap.sampling.as_ref().map(|p| (p.done, p.total)), Some((4, total)));
    // a driver without a pass: no-op
    d.apply(DynCommand::CancelSampling);
    let fin = d.snapshot().sampling;
    d.rescale_sampling(&small);
    assert_eq!(d.snapshot().sampling, fin);
}

/// An empty scene has no species: a pass over it is refused with the message the app shows.
#[test]
fn a_pass_over_an_empty_scene_is_refused() {
    let empty = Scene { nuclei: Vec::new(), cell: Cell { n: 33, h: 0.25 }, charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3], ionic: sim::IonicPolicy::None };
    let plan = SamplingPlan::default();
    assert_eq!(Sampler::new(&empty, &plan).err().as_deref(), Some(EMPTY_SCENE));
    assert_eq!(plan.solves(&empty), Err(EMPTY_SCENE.to_string()));
}

/// The estimated solver on an empty scene (`sim::dynamics` module doc, "The empty scene"; moved
/// here from `sim`'s driver tests with the surrogate, M6 Task 1): no model is needed and none is
/// asked — a sampling pass is refused ("place atoms to sample"), Play runs nothing, and the
/// teacher is never called (no fallback, no sample, no audit).
#[test]
fn the_surrogate_on_an_empty_scene_neither_samples_nor_falls_back() {
    let teacher = Shared::new(PairMorse::h2());
    let h = teacher.handle();
    let empty = Scene { nuclei: Vec::new(), cell: cell(33), charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3], ionic: sim::IonicPolicy::None };
    let (p, sh) = driven(surrogate_over(&teacher));
    let mut d = Dynamics::new(empty.clone(), p, 0.25, 1);
    let s0 = d.snapshot();
    assert!(s0.last_error.is_none() && !s0.forces_stale && s0.forces.is_empty(), "{:?}", s0.last_error);
    d.apply(DynCommand::StartSampling(SamplingPlan::default()));
    let r = d.snapshot();
    assert_eq!(r.last_rejection.as_deref(), Some(EMPTY_SCENE));
    assert!(r.sampling.is_none());
    d.apply(DynCommand::Play);
    for _ in 0..3 {
        let s = d.tick(f64::INFINITY, &mut never());
        assert!(s.last_error.is_none() && s.forces.is_empty() && s.display.is_none());
        assert_eq!((s.step, s.sim_time, s.energy_ks, s.temperature), (0, 0.0, 0.0, 0.0));
        assert_eq!(s.fallback_fraction_100, 0.0);
    }
    assert_eq!(h.lock().unwrap().solves.load(Ordering::SeqCst), 0, "the teacher is never asked about an empty scene");
    let mut sur = surrogate_of(&sh);
    assert_eq!(sur.samples(), 0, "nothing was kept as a sample");
    assert!(!sur.request_display(&empty), "no display refresh of an empty scene");
}

/// A teacher whose answers at pair separations below `r_overlap` carry a core-overlap report (what a pseudised
/// teacher would say with the cores in contact), and — when `pattern_change_after` is Some(k) — flags the first
/// non-overlapping pair answer after its k-th answer as an occupation-pattern change (once).
struct Overlapping {
    inner: PairMorse,
    r_overlap: f64,
    answers: usize,
    pattern_change_after: Option<usize>,
    flagged: bool,
}

impl ForceProvider for Overlapping {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        let mut r = self.inner.evaluate(scene, should_abort)?;
        self.answers += 1;
        let pair = scene.nuclei.len() == 2;
        if pair && dist(scene, 0, 1) < self.r_overlap {
            r.stats.core_overlaps = vec![(0, 1, dist(scene, 0, 1))];
            r.stats.warnings.push(format!("{}: 0–1 at R = {:.3} Bohr", sim::CORE_OVERLAP_WARNING, dist(scene, 0, 1)));
        } else if pair && !self.flagged && self.pattern_change_after.is_some_and(|k| self.answers > k) {
            self.flagged = true;
            r.stats.occupation_changed = true;
        }
        Ok(r)
    }

    fn name(&self) -> &'static str {
        "pair morse" // the surrogate's own teacher by name (the pass refuses another)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// M7 Task 9 (spec §5.6; Task 6 review A.6): a sampling pass never stores a geometry whose cores overlap or whose
/// shared-shell pattern changed — they are counted as skipped, noted, and leave an activity row of their own;
/// every other geometry of the pass is stored as before.
#[test]
fn a_pass_never_stores_core_overlapping_or_pattern_changing_answers() {
    let mut teacher = Overlapping { inner: PairMorse::h2(), r_overlap: 1.2, answers: 0, pattern_change_after: Some(5), flagged: false };
    let mut s = SurrogateKs::new(Box::new(Shared(Shared::new(PairMorse::h2()).handle())), FallbackPolicy::default()).with_teacher_tol(1e-6);
    let scene = pair(1, 1, 1.4, cell(41));
    let plan = SamplingPlan { perturbations: 10, hot_burst_steps: 0, scan_points_per_pair: 6, seed: 3, ..SamplingPlan::default() };
    let mut sampler = Sampler::new(&scene, &plan).unwrap();
    while sampler.step(&mut teacher, &mut s, &mut never()).unwrap().is_some() {}
    let fin = sampler.progress();
    let notes = sampler.take_notes();
    println!("progress {fin:?}; notes {notes:?}");
    let overlapping = notes.iter().filter(|n| n.contains("cores overlap")).count();
    let changed = notes.iter().filter(|n| n.contains("occupation pattern changed")).count();
    assert!(overlapping >= 1, "a perturbation or scan geometry closer than 1.2 Bohr: {notes:?}");
    assert_eq!(changed, 1, "{notes:?}");
    assert!(fin.skipped >= overlapping + changed, "{fin:?}");
    assert_eq!(s.samples(), sampler.samples_added());
    // every geometry that was stored is neither an overlapping one (< 1.2 Bohr) nor the answer flagged as a pattern change
    assert!(s.store().unwrap().samples.iter().all(|x| dist_of(&x.pos) >= 1.2), "an overlapping geometry was stored");
    let events = Trainable::take_teach_events(&mut s);
    assert_eq!(events.iter().filter(|e| e.outcome == sim::activity::TeachOutcome::UnfitNotKept).count(), overlapping + changed);
}

fn dist_of(pos: &[[f64; 3]]) -> f64 {
    (0..3).map(|a| (pos[1][a] - pos[0][a]).powi(2)).sum::<f64>().sqrt()
}

/// A teacher that fails to converge on the two-nucleus geometries a predicate picks (the pair
/// separation), with a scripted SCF error; everything else is `PairMorse`.
struct FailsAt {
    inner: PairMorse,
    fails: fn(f64) -> bool,
    error: sim::SolverError,
}

impl ForceProvider for FailsAt {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        if scene.nuclei.len() == 2 && (self.fails)(dist(scene, 0, 1)) {
            self.inner.seen.push(scene.clone());
            return Err(ProviderError::Solver(self.error.clone()));
        }
        self.inner.evaluate(scene, should_abort)
    }
    fn name(&self) -> &'static str {
        "pair morse"
    }
    fn reset(&mut self) {}
    fn discontinuity(&mut self) {}
}

/// A failed scan point on the stretched side ends its scan there (module doc, "A failed scan point
/// …"): H₂ at r₀ = 1.4, 12 points 0.7 … 1.8 r₀, every point beyond 1.45 r₀ fails. Point 9 (1.50 r₀)
/// is solved and fails (stalled); points 10–12 are never sent to the teacher, each a skip row with
/// the note; the earlier points are samples. Then: the cap's error names "cap"; a failure on the
/// compressed side skips nothing; a numerical failure skips nothing.
#[test]
fn a_failed_scan_point_skips_the_rest_of_its_stretched_tail() {
    use sim::activity::{ScfEnd, TeachOutcome};
    use sim::SolverError;
    let stalled = SolverError::ScfStalled { iterations: 61, best_delta_rho: 3.1e-4, tol: 1e-6, window: 50, level: 2e-4 };
    let scene = pair(1, 1, 1.4, cell(49));
    let plan = SamplingPlan { perturbations: 0, hot_burst_steps: 0, scan_points_per_pair: 12, ..SamplingPlan::default() };
    let run = |fails: fn(f64) -> bool, error: SolverError| {
        let mut teacher = FailsAt { inner: PairMorse::h2(), fails, error };
        let mut s = SurrogateKs::new(Box::new(Shared(Shared::new(PairMorse::h2()).handle())), FallbackPolicy::default()).with_teacher_tol(1e-6);
        let mut sampler = Sampler::new(&scene, &plan).unwrap();
        while sampler.step(&mut teacher, &mut s, &mut never()).unwrap().is_some() {}
        let seen: Vec<f64> = teacher.inner.seen.iter().filter(|x| x.nuclei.len() == 2).map(|x| dist(x, 0, 1) / 1.4).collect();
        (sampler.progress(), sampler.take_notes(), Trainable::take_teach_events(&mut s), sampler.samples_added(), seen)
    };
    // the stall at 1.50 r₀
    let (p, notes, events, added, seen) = run(|r| r > 1.45 * 1.4, stalled.clone());
    println!("stalled: {p:?}\n  notes {notes:?}\n  solved r/r₀ {seen:?}");
    assert_eq!((p.done, p.total, p.failed, p.skipped), (13, 13, 1, 3));
    assert!(p.finished);
    assert_eq!(added, 8, "points 1–8 (0.7 … 1.4 r₀) are samples");
    assert_eq!(seen.len(), 9, "points 10–12 are never solved");
    assert!(seen.iter().all(|&f| f < 1.55), "{seen:?}");
    let note = "scan 0–1: point 9 of 12 (r = 1.50 r₀) failed (stalled); further-out points skipped";
    assert!(notes.iter().any(|n| n.contains(note) && n.contains("(3 not solved)")), "{notes:?}");
    assert!(notes.iter().any(|n| n.contains("3 scan points beyond a failed one were not solved")), "{notes:?}");
    assert!(!notes.iter().any(|n| n.contains("cell face")), "no cell-guard note: {notes:?}");
    let failed: Vec<_> = events.iter().filter(|e| e.outcome == TeachOutcome::Failed).collect();
    assert_eq!(failed.len(), 1);
    assert_eq!((failed[0].scf_end, failed[0].scf_steps), (Some(ScfEnd::Stalled), Some(61)));
    assert!(failed[0].detail.contains("SCF stalled: Δρ best 3.1e-4 not improved 2× in 50 iterations"), "{}", failed[0].detail);
    let skips: Vec<_> = events.iter().filter(|e| e.outcome == TeachOutcome::NotSolved).collect();
    assert_eq!(skips.len(), 3);
    assert!(skips.iter().all(|e| e.detail.contains(note) && e.rule == sim::activity::TeachRule::Pass(PassJob::Scan)), "{skips:?}");
    // the iteration cap: the same skip, named "cap"
    let (p, notes, events, _, _) = run(|r| r > 1.45 * 1.4, SolverError::ScfNotConverged(200, 2e-4));
    assert_eq!((p.failed, p.skipped), (1, 3));
    assert!(notes.iter().any(|n| n.contains("point 9 of 12 (r = 1.50 r₀) failed (cap)")), "{notes:?}");
    assert!(events.iter().any(|e| e.scf_end == Some(ScfEnd::Cap) && e.scf_steps == Some(200)));
    // the compressed side: points 1–2 (0.7, 0.8 r₀) fail, nothing is skipped, the rest are samples
    let (p, notes, _, added, seen) = run(|r| r < 0.85 * 1.4, stalled.clone());
    assert_eq!((p.failed, p.skipped, added, seen.len()), (2, 0, 10, 12), "{p:?} {notes:?}");
    // another kind of failure skips nothing — a numerical failure, and a non-finite Δρ (review S5)
    let (p, _, _, added, seen) = run(|r| r > 1.45 * 1.4, SolverError::Numerical("test".into()));
    assert_eq!((p.failed, p.skipped, added, seen.len()), (4, 0, 8, 12));
    let (p, _, events, added, seen) = run(|r| r > 1.45 * 1.4, SolverError::ScfNotConverged(7, f64::NAN));
    assert_eq!((p.failed, p.skipped, added, seen.len()), (4, 0, 8, 12));
    assert!(events.iter().any(|e| e.scf_end == Some(ScfEnd::NonFinite) && e.scf_steps == Some(7)));
    // converged samples carry their end in the log (an analytic teacher has no SCF: none)
    let (_, _, events, _, _) = run(|_| false, stalled);
    assert!(events.iter().filter(|e| e.outcome.is_kept()).all(|e| e.scf_end.is_none() && e.scf_steps == Some(0)));
}

/// Report-only measurement behind `qm::SCF_STALL_WINDOW` (`qm::scf` "The stall stop";
/// `validate/references.md` "SCF stall stop"): the GPU SCF's Δρ per iteration with the stall stop
/// off (`stall_window` 0, the cap 200 alone) on what the sampling pass of the app's water solves —
/// the H₂O preset at the Accurate grid (95³ × 0.15) under `CoresOnly` (pseudo-O, all-electron H),
/// at the pass's teacher tolerance (1e-6, floor-aware) — and on the all-electron presets: cold
/// solves of converging cases (the preset, five pass perturbations, the pseudo-O atom at M = 1
/// and 3, O–H scan points), all-electron H₂ (49³) and H₂O (57³) at h = 0.25, then one O–H scan run
/// warm point after point as the pass runs it (density extrapolation on, reset at the scan's first
/// point; a warm failure retried cold, R15). For every solve: the longest run of iterations without
/// a 2× gain of the best Δρ — the stall rule's own counter (`ScfProgress::since_gain`, restarted at
/// a shell release) — and, per window W, the smallest gain b(i−W)/b(i) of the best Δρ over any W
/// iterations. Prints; asserts nothing.
#[test]
#[ignore = "report-only measurement (GPU, minutes): cargo test -p providers --release scf_stall_measurement -- --ignored --nocapture"]
fn scf_stall_measurement() {
    let Some(ctx) = gpu::GpuContext::new_headless() else {
        eprintln!("scf_stall_measurement: no GPU adapter — skipped");
        return;
    };
    struct Run {
        label: String,
        converged: bool,
        hist: Vec<f64>,
        /// the stall watch's counter per iteration (restarted at a shell release)
        since: Vec<usize>,
        outcome: String,
        ms: f64,
    }
    // the smallest best-Δρ gain over any W consecutive iterations (None: shorter than W)
    fn min_gain(h: &[f64], win: usize) -> Option<f64> {
        let best: Vec<f64> = h
            .iter()
            .scan(f64::INFINITY, |b, &d| {
                *b = b.min(d);
                Some(*b)
            })
            .collect();
        (win < best.len()).then(|| (win..best.len()).map(|i| best[i - win] / best[i]).fold(f64::INFINITY, f64::min))
    }
    fn fmt_hist(h: &[f64]) -> String {
        h.iter().map(|d| format!("{d:.1e}")).collect::<Vec<_>>().join(" ")
    }
    fn longest(r: &Run) -> usize {
        r.since.iter().copied().max().unwrap_or(0)
    }
    let mut runs: Vec<Run> = Vec::new();
    let mut solve = |solver: &mut qm_gpu::GpuSolver, label: String, scene: &Scene, warm: bool| -> bool {
        let mut notes = Vec::new();
        let (_, system, mut settings) = crate::ks::problem(scene, 1e-6, &mut notes).expect("a valid scene");
        if system.ionic.is_some() {
            settings.tol = crate::pseudo::effective_gpu_tol(settings.tol, (system.n_up + system.n_dn) as i64);
        }
        settings.stall_window = 0; // measure: the cap alone
        let t = Instant::now();
        let mut seen: Vec<qm::ScfProgress> = Vec::new();
        let r = solver
            .solve_observed(&system, &settings, warm, &mut || false, Some(&mut |p: &qm::ScfProgress| seen.push(*p)))
            .map(|o| o.expect("never aborted").1);
        let released = r.as_ref().map(|st| st.released.clone()).unwrap_or_default();
        let r = r.map(|st| st.iterations);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let hist = solver.profile().delta_rho_steps.clone();
        let since: Vec<usize> = seen.iter().map(|p| p.since_gain).collect();
        let (converged, outcome) = match &r {
            Ok(n) => (true, format!("converged in {n}")),
            Err(e) => (false, e.to_string()),
        };
        let run = Run { label, converged, hist, since, outcome, ms };
        let g = |w: usize| min_gain(&run.hist, w).map_or("—".to_string(), |g| format!("{g:.2}"));
        println!(
            "{:<46} {:>4} it {:>7.0} ms  {}  longest no-2×-gain run {:>3}  min gain over W = 10: {:>6}, 20: {:>6}, 30: {:>6}, 50: {:>6}",
            run.label,
            run.hist.len(),
            run.ms,
            if run.converged { "conv" } else { "FAIL" },
            longest(&run),
            g(10),
            g(20),
            g(30),
            g(50),
        );
        if !run.converged {
            println!("    {}", run.outcome);
        }
        if !released.is_empty() {
            println!("    released (DegenerateShells phase 2): {released:?}");
        }
        runs.push(run);
        converged
    };
    let nuc = |z: u8, p: [f64; 3]| Nucleus::new(z, p);
    let water = |c: Cell, ionic: sim::IonicPolicy| Scene {
        nuclei: vec![nuc(8, [0.0, 0.0, 0.0]), nuc(1, [1.43, 1.11, 0.0]), nuc(1, [-1.43, 1.11, 0.0])],
        cell: c,
        charge: 0,
        multiplicity: None,
        sigma: None,
        external_field: [0.0; 3],
        ionic,
    };

    // 1. the all-electron presets at h = 0.25 (cold)
    let mut s = qm_gpu::GpuSolver::new(ctx.clone(), &qm::Grid::cube(49, 0.25));
    solve(&mut s, "AE H2 49^3 R 1.45 cold".into(), &pair(1, 1, 1.45, Cell { n: 49, h: 0.25 }), false);
    drop(s);
    let mut s = qm_gpu::GpuSolver::new(ctx.clone(), &qm::Grid::cube(57, 0.25));
    solve(&mut s, "AE H2O 57^3 preset cold".into(), &water(Cell { n: 57, h: 0.25 }, sim::IonicPolicy::None), false);
    drop(s);

    // 2. the app's pseudo water at 95³ × 0.15 (CoresOnly), cold
    let scene = water(Cell { n: 95, h: 0.15 }, sim::IonicPolicy::CoresOnly);
    let mut s = qm_gpu::GpuSolver::new(ctx.clone(), &qm::Grid::cube(95, 0.15));
    solve(&mut s, "pseudo H2O 95^3 preset cold".into(), &scene, false);
    for m in [1, 3] {
        let atom = Scene { nuclei: vec![nuc(8, [0.0; 3])], multiplicity: Some(m), ..scene.clone() };
        solve(&mut s, format!("pseudo O atom 95^3 M = {m} cold"), &atom, false);
    }
    let sampler = Sampler::new(&scene, &SamplingPlan::default()).unwrap();
    let perts: Vec<Scene> = sampler
        .queue
        .iter()
        .filter_map(|j| if let Job::Perturbation(sc) = j { Some(sc.clone()) } else { None })
        .take(5)
        .collect();
    for (i, sc) in perts.iter().enumerate() {
        solve(&mut s, format!("pseudo H2O perturbation {} cold", i + 1), sc, false);
    }
    let oh = sampler.scans().iter().position(|p| scene.nuclei[p.i].z == 8 || scene.nuclei[p.j].z == 8).expect("an O–H pair");
    let points: Vec<(usize, Scene)> = sampler
        .queue
        .iter()
        .filter_map(|j| match j {
            Job::Scan(sc, p, k) if *p == oh => Some((*k, sc.clone())),
            _ => None,
        })
        .collect();
    let ps = sampler.scans()[oh].clone();
    println!("O–H scan: nuclei {}–{}, r0 {:.3} Bohr, {} points", ps.i, ps.j, ps.r0, ps.r.len());
    for (k, sc) in points.iter().filter(|(k, _)| [0, 3, 8, 11].contains(k)) {
        solve(&mut s, format!("pseudo O-H scan {:.2} r0 cold", ps.r[*k] / ps.r0), sc, false);
    }

    // 3. the same scan warm, point after point, as the pass runs it (after the preset)
    solve(&mut s, "pseudo H2O 95^3 preset (start of the warm scan)".into(), &scene, false);
    s.set_density_extrapolation(true);
    for (k, sc) in &points {
        if *k == 0 {
            // `discontinuity` at the scan's first point: no extrapolation across the jump
            s.set_density_extrapolation(false);
            s.set_density_extrapolation(true);
        }
        let f = ps.r[*k] / ps.r0;
        if !solve(&mut s, format!("pseudo O-H scan {f:.2} r0 warm"), sc, true) {
            // R15: retried once cold, extrapolation off for that solve
            s.set_density_extrapolation(false);
            solve(&mut s, format!("pseudo O-H scan {f:.2} r0 cold retry"), sc, false);
            s.set_density_extrapolation(true);
        }
    }

    // 4. review S4: the regimes the first measurement did not cover — the GPU LiH, He and H₃⁺ (equilateral, 1.65 Bohr,
    // charge +1) all-electron presets at 49³ × 0.25; pseudo O–H at 2.5 / 3.5 / 5.0 Bohr (doublet, CoresOnly, 95³ × 0.15:
    // candidates for a DegenerateShells release — `released` is printed); and, after the GPU runs, the CPU at the
    // σ → 0 oracles' tight tolerance (H₂ 33³, tol 1e-9, eig_tol 1e-8)
    let mut s = qm_gpu::GpuSolver::new(ctx.clone(), &qm::Grid::cube(49, 0.25));
    let c49 = Cell { n: 49, h: 0.25 };
    let lih = Scene { nuclei: vec![nuc(3, [-0.8, 0.0, 0.0]), nuc(1, [2.2, 0.0, 0.0])], ..pair(1, 1, 1.4, c49.clone()) };
    solve(&mut s, "AE LiH 49^3 preset cold".into(), &lih, false);
    let he = Scene { nuclei: vec![nuc(2, [0.0; 3])], ..pair(1, 1, 1.4, c49.clone()) };
    solve(&mut s, "AE He 49^3 cold".into(), &he, false);
    let a = 1.65;
    let h3 = Scene {
        nuclei: vec![nuc(1, [a / 2.0, -a / (2.0 * 3f64.sqrt()), 0.0]), nuc(1, [-a / 2.0, -a / (2.0 * 3f64.sqrt()), 0.0]), nuc(1, [0.0, a / 3f64.sqrt(), 0.0])],
        charge: 1,
        ..pair(1, 1, 1.4, c49.clone())
    };
    solve(&mut s, "AE H3+ 49^3 cold".into(), &h3, false);
    drop(s);
    let mut s = qm_gpu::GpuSolver::new(ctx.clone(), &qm::Grid::cube(95, 0.15));
    for d in [2.5, 3.5, 5.0] {
        let oh = Scene { nuclei: vec![nuc(8, [0.0; 3]), nuc(1, [d, 0.0, 0.0])], multiplicity: Some(2), ..scene.clone() };
        solve(&mut s, format!("pseudo O-H {d} Bohr M = 2 cold"), &oh, false);
    }
    drop(s);
    {
        let g = qm::Grid::cube(33, 0.25);
        let mut st = qm::ScfSettings::for_grid(&g);
        (st.tol, st.eig_tol, st.stall_window) = (1e-9, 1e-8, 0);
        let sys = qm::System::new(vec![qm::Nucleus { pos: [0.7, 0.0, 0.0], z: 1 }, qm::Nucleus { pos: [-0.7, 0.0, 0.0], z: 1 }], 0, None);
        let mut seen: Vec<qm::ScfProgress> = Vec::new();
        let t = Instant::now();
        let r = qm::scf_observed(&g, &sys, &st, None, Some(&mut |p: &qm::ScfProgress| seen.push(*p)));
        let since: Vec<usize> = seen.iter().map(|p| p.since_gain).collect();
        let hist: Vec<f64> = seen.iter().map(|p| p.delta_rho).collect();
        println!(
            "{:<46} {:>4} it {:>7.0} ms  {}  longest no-2×-gain run {:>3}; Δρ: {}",
            "CPU H2 33^3 tol 1e-9 eig_tol 1e-8 cold",
            hist.len(),
            t.elapsed().as_secs_f64() * 1e3,
            if r.is_ok() { "conv" } else { "FAIL" },
            since.iter().copied().max().unwrap_or(0),
            fmt_hist(&hist)
        );
        runs.push(Run { label: "CPU H2 33^3 tol 1e-9".into(), converged: r.is_ok(), hist, since, outcome: String::new(), ms: t.elapsed().as_secs_f64() * 1e3 });
    }

    // the summary
    let conv: Vec<&Run> = runs.iter().filter(|r| r.converged).collect();
    let slowest = conv.iter().max_by_key(|r| r.hist.len()).expect("something converged");
    let plateau = conv.iter().max_by_key(|r| longest(r)).expect("something converged");
    println!();
    println!("slowest converged: {} — {} iterations; Δρ: {}", slowest.label, slowest.hist.len(), fmt_hist(&slowest.hist));
    println!("  its no-gain counter: {:?}", slowest.since);
    println!("longest no-2×-gain run in a converged solve: {} iterations ({}); Δρ: {}", longest(plateau), plateau.label, fmt_hist(&plateau.hist));
    println!("  its no-gain counter: {:?}", plateau.since);
    for win in [10, 20, 30, 50] {
        let worst = conv.iter().filter_map(|r| min_gain(&r.hist, win).map(|g| (g, r.label.clone()))).min_by(|a, b| a.0.total_cmp(&b.0));
        println!(
            "smallest best-Δρ gain over {win} iterations in a converged solve: {}",
            worst.map_or("— (none that long)".to_string(), |(g, l)| format!("{g:.2} ({l})"))
        );
    }
    for r in runs.iter().filter(|r| !r.converged) {
        let stop = r.since.iter().position(|&k| k >= qm::SCF_STALL_WINDOW).map(|i| i + 1);
        let best = stop.map(|n| r.hist[..n].iter().copied().fold(f64::INFINITY, f64::min));
        println!();
        println!(
            "failed: {} — {} ({:.0} ms); the rule at W = {} stops it at iteration {:?} (best Δρ then {:?}); Δρ: {}",
            r.label,
            r.outcome,
            r.ms,
            qm::SCF_STALL_WINDOW,
            stop,
            best,
            fmt_hist(&r.hist)
        );
        println!("  its no-gain counter: {:?}", r.since);
    }
}

/// Report-only (review M1 follow-up): stretched H₂ (`IonicPolicy::None`, multiplicity None — the spin-symmetric
/// solution, as the app and `validate`'s tools suite pose it), cold GPU solves with the stall stop off, at the tools
/// suite's grids: 65³ × 0.25 (R 1.8, 2.5, 3.5, 5.0, 6.0) and 107³ × 0.15 (R 1.45, 2.0 … 6.0 by 0.25). Prints each
/// solve's iterations and its longest run without a 2× gain of the best Δρ (the stall rule's counter), and the Δρ
/// trace of the longest.
#[test]
#[ignore = "report-only measurement (GPU, minutes): cargo test -p providers --release scf_stall_stretched_h2 -- --ignored --nocapture"]
fn scf_stall_stretched_h2() {
    let Some(ctx) = gpu::GpuContext::new_headless() else {
        eprintln!("no GPU adapter — skipped");
        return;
    };
    let mut worst: (usize, String, Vec<f64>, Vec<usize>) = (0, String::new(), Vec::new(), Vec::new());
    for (n, h, rs) in [(65usize, 0.25, vec![1.8, 2.5, 3.5, 5.0, 6.0]), (107, 0.15, std::iter::once(1.45).chain((0..17).map(|k| 2.0 + 0.25 * k as f64)).collect::<Vec<f64>>())] {
        let mut s = qm_gpu::GpuSolver::new(ctx.clone(), &qm::Grid::cube(n, h));
        for r in rs {
            let scene = pair(1, 1, r, Cell { n, h });
            let (_, system, mut settings) = crate::ks::problem(&scene, 1e-6, &mut Vec::new()).expect("H2");
            settings.stall_window = 0;
            let mut seen: Vec<qm::ScfProgress> = Vec::new();
            let t = Instant::now();
            let res = s.solve_observed(&system, &settings, false, &mut || false, Some(&mut |p: &qm::ScfProgress| seen.push(*p))).map(|o| o.expect("never aborted").1.iterations);
            let since: Vec<usize> = seen.iter().map(|p| p.since_gain).collect();
            let longest = since.iter().copied().max().unwrap_or(0);
            println!("H2 {n}^3 h {h} R {r:.2}: {:?} in {:.0} ms; longest no-2×-gain run {longest}", res.as_ref().map_err(|e| e.to_string()), t.elapsed().as_secs_f64() * 1e3);
            if longest > worst.0 && res.is_ok() {
                worst = (longest, format!("{n}^3 R {r:.2}"), seen.iter().map(|p| p.delta_rho).collect(), since);
            }
        }
    }
    println!("longest converged plateau: {} iterations ({}); Δρ: {}", worst.0, worst.1, worst.2.iter().map(|d| format!("{d:.1e}")).collect::<Vec<_>>().join(" "));
    println!("  its no-gain counter: {:?}", worst.3);
}

// ---- the ghost: what the pass is solving now (display only; `Sampler::ghost`, `Trainable::pass_ghost`) ----

fn positions(g: &PassGhost) -> Vec<[f64; 3]> {
    g.atoms.iter().map(|a| a.pos).collect()
}

fn stalled_error() -> sim::SolverError {
    sim::SolverError::ScfStalled { iterations: 61, best_delta_rho: 3.1e-4, tol: 1e-6, window: 50, level: 8e-3 }
}

/// At every step the ghost's geometry is exactly the one the teacher is asked about next — a
/// reference atom, a nudge, a scan point, a burst evaluation — with the stage's labels: the
/// reference's Z, the nudge's index, the scan's pair, point, r/r₀ and r, the burst's step and trail;
/// the base positions are the pass's; the last finished job's serial advances by one per job.
#[test]
fn the_ghost_is_the_geometry_the_teacher_is_asked_about_next() {
    let mut teacher = Shared::new(PairMorse::h2());
    let h = teacher.handle();
    let mut s = surrogate_over(&teacher);
    let scene = pair(1, 1, 1.4, cell(41));
    let plan = SamplingPlan { perturbations: 3, hot_burst_steps: 4, scan_points_per_pair: 5, seed: 3, ..SamplingPlan::default() };
    let mut sampler = Sampler::new(&scene, &plan).unwrap();
    let base: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
    let mut serial = 0;
    let mut stages: Vec<&str> = Vec::new();
    while let Some(g) = sampler.ghost(&s) {
        assert_eq!(g.finished.as_ref().map_or(0, |f| f.serial), serial, "one finished job per solve so far");
        if let Some(f) = &g.finished {
            assert_eq!(f.outcome, GhostOutcome::Kept, "an exact pair potential: every job is kept");
        }
        let n_before = h.lock().unwrap().seen.len();
        assert!(sampler.step(&mut teacher, &mut s, &mut never()).unwrap().is_some());
        serial += 1;
        let t = h.lock().unwrap();
        assert_eq!(t.seen.len(), n_before + 1);
        let asked: Vec<[f64; 3]> = t.seen.last().unwrap().nuclei.iter().map(|n| n.pos).collect();
        assert_eq!(positions(&g), asked, "the ghost of {:?} is what the teacher solved", g.job);
        let done_in_stage = stages.iter().filter(|x| **x == g.stage).count();
        match g.job {
            GhostJob::Reference { z } => {
                assert_eq!((g.stage, z), ("references", 1));
                assert_eq!(g.atoms, vec![GhostAtom { z: 1, pos: [0.0; 3] }], "the lone atom at the cell centre, as solved");
                assert!(g.base.is_empty() && g.trail.is_empty());
            }
            GhostJob::Perturbation { index, of } => {
                assert_eq!((g.stage, of, index), ("perturbations", 3, done_in_stage + 1));
                assert_eq!(g.base, base, "the unperturbed positions are the pass base");
                assert_ne!(positions(&g), base);
            }
            GhostJob::Scan { i, j, k, of, ratio, r } => {
                assert_eq!((g.stage, i, j, of, k), ("scans", 0, 1, 5, done_in_stage + 1));
                let want = 0.7 + 1.1 * (k - 1) as f64 / 4.0;
                assert!((ratio - want).abs() < 1e-12 && (r - 1.4 * want).abs() < 1e-12, "k {k}: ratio {ratio} r {r}");
                assert!((dist_of(&positions(&g)) - r).abs() < 1e-12);
                assert_eq!(g.base, base);
            }
            GhostJob::Teach { .. } => unreachable!("a default pass teaches nothing around"),
            GhostJob::ShellCheck { .. } => unreachable!("a 1.4 Bohr scan never crosses 5/6 r_cut"),
            GhostJob::HotBurst { step, of } => {
                assert_eq!((g.stage, of, step), ("hot burst", 4, done_in_stage + 1));
                assert_eq!(g.trail.len(), (step - 1).min(GHOST_TRAIL), "the solved evaluations before this one");
                if step == 1 {
                    assert_eq!(positions(&g), base, "the burst starts at the scene's geometry");
                }
            }
        }
        stages.push(g.stage);
    }
    assert_eq!(stages.len(), 1 + 3 + 5 + 4, "one ghost per solve");
    assert!(sampler.step(&mut teacher, &mut s, &mut never()).unwrap().is_none());
    assert!(sampler.ghost(&s).is_none(), "finished: nothing is being solved");
}

/// A trail never holds more than `GHOST_TRAIL` evaluations.
#[test]
fn the_burst_trail_is_capped() {
    let mut teacher = Shared::new(PairMorse::h2());
    let mut s = surrogate_over(&teacher);
    let scene = pair(1, 1, 1.4, cell(41));
    let plan = SamplingPlan { perturbations: 0, hot_burst_steps: GHOST_TRAIL + 4, scan_points_per_pair: 0, ..SamplingPlan::default() };
    let mut sampler = Sampler::new(&scene, &plan).unwrap();
    let mut longest = 0;
    while sampler.step(&mut teacher, &mut s, &mut never()).unwrap().is_some() {
        if let Some(g) = sampler.ghost(&s) {
            longest = longest.max(g.trail.len());
        }
    }
    assert_eq!(longest, GHOST_TRAIL);
}

/// A cached reference costs no solve, so it is never the ghost: a second pass at the same settings
/// starts at its first nudge.
#[test]
fn a_cached_reference_is_not_the_ghost() {
    let mut teacher = Shared::new(PairMorse::h2());
    let mut s = surrogate_over(&teacher);
    let scene = pair(1, 1, 1.4, cell(41));
    let plan = SamplingPlan { perturbations: 2, hot_burst_steps: 0, scan_points_per_pair: 0, ..SamplingPlan::default() };
    let mut first = Sampler::new(&scene, &plan).unwrap();
    assert!(matches!(first.ghost(&s).unwrap().job, GhostJob::Reference { z: 1 }));
    while first.step(&mut teacher, &mut s, &mut never()).unwrap().is_some() {}
    let again = Sampler::new(&scene, &plan).unwrap().with_references(first.take_references());
    let g = again.ghost(&s).unwrap();
    assert!(matches!(g.job, GhostJob::Perturbation { index: 1, of: 2 }), "{:?}", g.job);
    assert!(g.finished.is_none());
}

/// The outcome of the job just finished, in the solve log's words: a scan point that stalls is
/// `Failed(Stalled)` ("stalled"), its skipped tail "skipped", the points before it kept; a numerical
/// failure is plain "failed".
#[test]
fn the_ghost_reports_how_the_last_job_ended() {
    use sim::activity::ScfEnd;
    let scene = pair(1, 1, 1.4, cell(49));
    let plan = SamplingPlan { perturbations: 0, hot_burst_steps: 0, scan_points_per_pair: 12, ..SamplingPlan::default() };
    let run = |error: sim::SolverError| {
        let mut teacher = FailsAt { inner: PairMorse::h2(), fails: |r| r > 1.45 * 1.4, error };
        let mut s = SurrogateKs::new(Box::new(Shared(Shared::new(PairMorse::h2()).handle())), FallbackPolicy::default()).with_teacher_tol(1e-6);
        let mut sampler = Sampler::new(&scene, &plan).unwrap();
        let mut marks = Vec::new();
        let mut ghost_after_failure = None;
        while sampler.step(&mut teacher, &mut s, &mut never()).unwrap().is_some() {
            let m = sampler.finished.clone().unwrap();
            if matches!(m.outcome, GhostOutcome::Failed(_)) && ghost_after_failure.is_none() {
                ghost_after_failure = Some(sampler.ghost(&s));
            }
            marks.push(m.outcome);
        }
        marks.push(sampler.finished.clone().unwrap().outcome); // the skipped tail, accounted for by the last call
        (marks, ghost_after_failure)
    };
    let (marks, after) = run(stalled_error());
    // the reference, scan points 1-8 kept, point 9 stalls, then the three skip rows
    assert_eq!(marks.len(), 1 + 8 + 1 + 1, "{marks:?}");
    assert!(marks[..9].iter().all(|m| *m == GhostOutcome::Kept));
    assert_eq!(marks[9], GhostOutcome::Failed(Some(ScfEnd::Stalled)));
    assert_eq!(marks[10], GhostOutcome::Skipped);
    assert!(after.unwrap().is_none(), "only skip rows are left after the failed point: nothing is being solved");
    let (marks, _) = run(sim::SolverError::Numerical("test".into()));
    assert!(marks.contains(&GhostOutcome::Failed(None)), "{marks:?}");
    assert_eq!(GhostOutcome::Failed(Some(ScfEnd::Stalled)).word(), "stalled");
    assert_eq!(GhostOutcome::Failed(Some(ScfEnd::Cap)).word(), "cap");
    assert_eq!(GhostOutcome::Failed(None).word(), "failed");
    assert_eq!(GhostOutcome::Kept.word(), "kept");
    assert_eq!(GhostOutcome::Skipped.word(), "skipped");
}

/// Geometries whose cores overlap are solved and not kept: the ghost marks them skipped.
#[test]
fn an_unfit_geometry_is_marked_skipped() {
    let scene = pair(1, 1, 1.4, cell(49));
    let plan = SamplingPlan { perturbations: 0, hot_burst_steps: 0, scan_points_per_pair: 12, ..SamplingPlan::default() };
    let mut teacher = Overlapping { inner: PairMorse::h2(), r_overlap: 1.2, answers: 0, pattern_change_after: None, flagged: false };
    let mut s = SurrogateKs::new(Box::new(Shared(Shared::new(PairMorse::h2()).handle())), FallbackPolicy::default()).with_teacher_tol(1e-6);
    let mut sampler = Sampler::new(&scene, &plan).unwrap();
    let mut marks = Vec::new();
    while sampler.step(&mut teacher, &mut s, &mut never()).unwrap().is_some() {
        marks.push(sampler.finished.clone().unwrap().outcome);
    }
    assert!(marks.contains(&GhostOutcome::Skipped) && marks.contains(&GhostOutcome::Kept), "{marks:?}");
}

/// The ghost through the `Trainable` boundary and the driver: present while a pass runs, and gone
/// — the viewer shows the scene again — when it is cancelled, when it ends, and when it fails.
#[test]
fn the_ghost_ends_with_the_pass_cancelled_finished_or_failed() {
    let teacher = Shared::new(PairMorse::h2());
    let (p, _h) = driven(surrogate_over(&teacher));
    let scene = pair(1, 1, 1.45, cell(41));
    let mut d = Dynamics::new(scene.clone(), p, 0.25, 1);
    let plan = SamplingPlan { perturbations: 4, hot_burst_steps: 3, scan_points_per_pair: 3, seed: 5, ..SamplingPlan::default() };
    assert!(d.snapshot().ghost.is_none(), "no pass, no ghost");
    d.apply(DynCommand::StartSampling(plan.clone()));
    let g = d.snapshot().ghost.expect("the pass's first job at once");
    assert_eq!(g.stage, "references");
    // one solve per tick: the ghost follows
    let snap = d.tick(0.0, &mut never());
    assert_eq!(snap.ghost.as_ref().map(|g| g.stage), Some("perturbations"));
    assert_eq!(snap.ghost.as_ref().and_then(|g| g.finished.as_ref()).map(|f| f.serial), Some(1));
    // cancelled
    d.apply(DynCommand::CancelSampling);
    assert!(d.snapshot().ghost.is_none(), "cancelled");
    assert!(d.snapshot().sampling.unwrap().cancelled);
    // run to the end
    d.apply(DynCommand::StartSampling(plan.clone()));
    let (mut ticks, mut with_ghost) = (0, 0);
    loop {
        let snap = d.tick(0.0, &mut never());
        with_ghost += usize::from(snap.ghost.is_some());
        if snap.sampling.as_ref().unwrap().finished {
            assert!(snap.ghost.is_none(), "finished");
            break;
        }
        ticks += 1;
        assert!(ticks < 200);
    }
    assert!(with_ghost >= 8, "a ghost on the way: {with_ghost}");
    // failed: the reference solve fails, the pass ends with the error
    struct NoAtoms(PairMorse);
    impl ForceProvider for NoAtoms {
        fn evaluate(&mut self, scene: &Scene, a: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
            if scene.nuclei.len() == 1 {
                return Err(ProviderError::Invalid("no isolated atoms here".into()));
            }
            self.0.evaluate(scene, a)
        }
        fn name(&self) -> &'static str {
            "pair morse"
        }
        fn reset(&mut self) {}
    }
    let s = SurrogateKs::new(Box::new(NoAtoms(PairMorse::h2())), FallbackPolicy::default()).with_teacher_tol(1e-6);
    let (p, _h) = driven(s);
    let mut d = Dynamics::new(scene, p, 0.25, 1);
    d.apply(DynCommand::StartSampling(plan));
    assert!(d.snapshot().ghost.is_some());
    let snap = d.tick(f64::INFINITY, &mut never());
    assert!(snap.last_error.as_deref().is_some_and(|e| e.contains("sampling")), "{:?}", snap.last_error);
    assert!(snap.ghost.is_none(), "failed: the scene returns");
    // a provider without a model has none
    let plain = Dynamics::new(pair(1, 1, 1.4, cell(41)), Box::new(MorseDimer::h2()), 0.25, 1);
    assert!(plain.snapshot().ghost.is_none());
}

/// The scan stops short of r_cut (module doc, 3.; `SCAN_CUTOFF_FRACTION`; round 3 review R3-1): a
/// pair whose 0.7 … 1.8 r₀ scan crosses 5/6 of the model's cutoff (r₀ 3.3 Bohr: 2.31 … 5.94) has
/// its first point beyond the limit solved as a **shell check** — never fitted, handed over by
/// `take_shell_checks` — and its further points planned as skips (never solved, counted in
/// `skipped`, each with its note and the pass's tally); the total is unchanged. The same scan with
/// a larger cutoff solves every point and has no shell check.
#[test]
fn pair_scans_stop_short_of_the_cutoff() {
    let plan = SamplingPlan { perturbations: 0, hot_burst_steps: 0, scan_points_per_pair: 12, ..SamplingPlan::default() };
    let scene = pair(1, 1, 3.3, cell(65));
    let r_cut = 6.0;
    let limit = SCAN_CUTOFF_FRACTION * r_cut;
    let sampler = Sampler::with_cutoff(&scene, &plan, r_cut).unwrap();
    let want = sampler.scans()[0].r.iter().filter(|&&r| r > limit).count();
    println!("r = {:.3?}; beyond {limit:.2}: {want}", sampler.scans()[0].r);
    assert!(want > 0 && want < 12, "premise: the scan crosses the limit");
    let limited = sampler.queue.iter().filter(|j| matches!(j, Job::Skip("scans", Some(_), Some(n)) if n.contains(SCAN_LIMIT_MARK))).count();
    let shells = sampler.queue.iter().filter(|j| matches!(j, Job::Shell(..))).count();
    assert_eq!((shells, limited), (1, want - 1), "one shell check, the rest skipped");
    let teacher = Shared::new(PairMorse::h2());
    let h = teacher.handle();
    let mut s = surrogate_over(&teacher);
    let mut sm = Sampler::with_cutoff(&scene, &plan, r_cut).unwrap();
    while sm.step_own_teacher(&mut s, &mut never()).unwrap().is_some() {}
    let p = sm.progress();
    let notes = sm.take_notes();
    println!("{p:?}\n{notes:?}");
    assert_eq!(sm.samples_added(), 12 - want, "the shell check is never a sample");
    assert_eq!((p.total, p.skipped), (1 + 12, want - 1), "the total keeps the skipped points");
    let beyond: Vec<f64> = h.lock().unwrap().seen.iter().filter(|x| x.nuclei.len() == 2).map(|x| dist(x, 0, 1)).filter(|&r| r > limit).collect();
    let first = sm.scans()[0].r.iter().copied().find(|&r| r > limit).unwrap();
    assert_eq!(beyond.len(), 1, "only the shell check is solved beyond the limit: {beyond:?}");
    assert!((beyond[0] - first).abs() < 1e-12);
    let checks = sm.take_shell_checks();
    assert_eq!(checks.len(), 1);
    assert_eq!((checks[0].pair, checks[0].r, checks[0].r_cut), ((1, 1), first, r_cut));
    assert!(notes.iter().any(|n| n.starts_with(&format!("sampling: {} scan points beyond 5/6 of the model's cutoff were not solved", want - 1))), "{notes:?}");
    // a model with a larger cutoff scans the whole range
    let wide = Sampler::with_cutoff(&scene, &plan, 8.0).unwrap();
    assert!(wide.queue.iter().all(|j| !matches!(j, Job::Skip(..))));
}

/// The ghost skips the scan points planned as cutoff skips like any other skip (they cost no
/// solve): at every step it is the geometry the teacher is asked about. The one point beyond the
/// limit it shows is the pair's shell check (solved, never fitted), labelled as a check — never
/// as a fitted scan point (final review M9).
#[test]
fn the_ghost_skips_the_cutoff_skipped_scan_points() {
    let plan = SamplingPlan { perturbations: 0, hot_burst_steps: 0, scan_points_per_pair: 12, ..SamplingPlan::default() };
    let scene = pair(1, 1, 3.3, cell(65));
    let r_cut = 6.0;
    let limit = SCAN_CUTOFF_FRACTION * r_cut;
    let mut teacher = Shared::new(PairMorse::h2());
    let h = teacher.handle();
    let mut s = surrogate_over(&teacher);
    let mut sampler = Sampler::with_cutoff(&scene, &plan, r_cut).unwrap();
    let (mut scans, mut shell) = (0, 0);
    while let Some(g) = sampler.ghost(&s) {
        match g.job {
            GhostJob::Scan { r, .. } => {
                assert!(r <= limit, "a fitted scan point beyond the limit: {r}");
                scans += 1;
            }
            GhostJob::ShellCheck { i, j, r } => {
                assert!(r > limit && (i, j) == (0, 1), "{:?}", g.job);
                shell += 1;
            }
            _ => {}
        }
        assert!(sampler.step(&mut teacher, &mut s, &mut never()).unwrap().is_some());
        let t = h.lock().unwrap();
        let asked: Vec<[f64; 3]> = t.seen.last().unwrap().nuclei.iter().map(|n| n.pos).collect();
        assert_eq!(positions(&g), asked);
    }
    assert!(scans > 0 && scans < 12, "{scans}");
    assert_eq!(shell, 1, "beyond the limit the ghost shows only the shell check");
}

/// A teach-around pass has a ghost too: stage "teaching", the blocker and the solve in hand.
#[test]
fn the_ghost_describes_a_teach_around_pass() {
    let teacher = Shared::new(PairMorse::h2());
    let s = surrogate_over(&teacher);
    let jobs = vec![
        TargetJob { scene: pair(1, 1, 1.5, cell(41)), blocker: 7, holdout: false },
        TargetJob { scene: pair(1, 1, 1.6, cell(41)), blocker: 7, holdout: true },
    ];
    let sampler = Sampler::targeted(&pair(1, 1, 1.4, cell(41)), jobs);
    let g = sampler.ghost(&s).expect("the first teach solve");
    assert_eq!(g.stage, "teaching");
    assert_eq!(g.job, GhostJob::Teach { blocker: 7, index: 1, of: 2, holdout: false });
    assert_eq!(g.atoms.len(), 2);
}

/// A teacher that fails the two-nucleus solves whose 1-based count `fails` picks (a capped SCF),
/// counts its resets, and solves everything else as `PairMorse`.
struct FailsOnCall {
    inner: PairMorse,
    calls: usize,
    fails: fn(usize) -> bool,
    resets: usize,
}

impl ForceProvider for FailsOnCall {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        if scene.nuclei.len() == 2 {
            self.calls += 1;
            if (self.fails)(self.calls) {
                self.inner.seen.push(scene.clone());
                return Err(ProviderError::Solver(sim::SolverError::ScfNotConverged(200, 3e-4)));
            }
        }
        self.inner.evaluate(scene, should_abort)
    }
    fn name(&self) -> &'static str {
        "pair morse"
    }
    fn reset(&mut self) {
        self.resets += 1;
    }
    fn discontinuity(&mut self) {}
}

/// User report 2026-10-20 (module doc, 4.): a failed hot-burst step restarts the burst from the last
/// geometry it solved with fresh Maxwell–Boltzmann velocities (the teacher reset) instead of ending
/// it, so the pass makes its planned evaluations; three failures in a row, or a failed start
/// geometry, end it as before.
#[test]
fn a_failed_burst_step_restarts_the_burst_from_the_last_good_geometry() {
    use sim::activity::{ScfEnd, TeachOutcome};
    let scene = pair(1, 1, 1.4, cell(49));
    let plan = SamplingPlan { perturbations: 0, hot_burst_steps: 20, scan_points_per_pair: 0, seed: 7, ..SamplingPlan::default() };
    let run = |fails: fn(usize) -> bool| {
        let mut teacher = FailsOnCall { inner: PairMorse::h2(), calls: 0, fails, resets: 0 };
        let mut s = SurrogateKs::new(Box::new(Shared(Shared::new(PairMorse::h2()).handle())), FallbackPolicy::default()).with_teacher_tol(1e-6);
        let mut sampler = Sampler::new(&scene, &plan).unwrap();
        while sampler.step(&mut teacher, &mut s, &mut never()).unwrap().is_some() {}
        let kept: Vec<Vec<[f64; 3]>> = s.store().map_or(Vec::new(), |st| st.samples.iter().filter(|x| x.tag == SampleTag::HotBurst).map(|x| x.pos.clone()).collect());
        (sampler.progress(), sampler.take_notes(), Trainable::take_teach_events(&mut s), kept, teacher)
    };

    // the 6th evaluation fails: the burst goes on from the 5th with fresh velocities
    let (p, notes, events, kept, teacher) = run(|n| n == 6);
    println!("one failure: {p:?}\n  {notes:?}");
    assert_eq!((p.done, p.total, p.failed, p.skipped), (21, 21, 1, 0), "the failed step is one of the 20 evaluations; nothing skipped");
    assert!(p.finished);
    assert_eq!(kept.len(), 19);
    assert!(notes.iter().any(|n| n.contains("the burst restarts from the last geometry it solved with fresh Maxwell–Boltzmann velocities at 2000 K (restart 1; 14 evaluations left)")), "{notes:?}");
    let failed: Vec<_> = events.iter().filter(|e| e.outcome == TeachOutcome::Failed).collect();
    assert_eq!(failed.len(), 1);
    assert_eq!((failed[0].scf_end, failed[0].scf_steps), (Some(ScfEnd::Cap), Some(200)));
    // the step after the restart is one dt from the 5th solved geometry, not the failed one again
    let failed_pos: Vec<[f64; 3]> = teacher.inner.seen.iter().filter(|x| x.nuclei.len() == 2).nth(5).expect("the failed geometry").nuclei.iter().map(|n| n.pos).collect();
    let step = |a: &[[f64; 3]], b: &[[f64; 3]]| (0..a.len()).flat_map(|i| (0..3).map(move |k| (a[i][k] - b[i][k]).abs())).fold(0.0, f64::max);
    let typical = step(&kept[3], &kept[4]);
    assert!(step(&kept[4], &kept[5]) < 3.0 * typical, "one dt from the 5th: {} vs {typical}", step(&kept[4], &kept[5]));
    assert!(step(&failed_pos, &kept[5]) > 1e-9, "not the failed geometry again");
    // the restart's temperature is the burst's: K exactly the mean at the draw, one Verlet step later
    let t_after = events.iter().filter(|e| e.outcome.is_kept()).nth(5).and_then(|e| e.burst_t).expect("a burst temperature");
    assert!((t_after / 2000.0 - 1.0).abs() < 0.2, "T after the restart {t_after:.0} K");

    // the teacher is cold after a failure (one reset more than without it)
    let (_, _, _, _, t1) = run(|_| false);
    let (_, _, _, _, t2) = run(|n| n == 6);
    assert_eq!(t2.resets, t1.resets + 1, "the solve after the failure is cold");

    // three failures in a row end the burst: the 6th, 7th and 8th; 12 evaluations skipped
    let (p, notes, _, kept, _) = run(|n| (6..=8).contains(&n));
    assert_eq!((p.done, p.total, p.failed, p.skipped, kept.len()), (21, 21, 3, 12, 5), "{p:?}");
    assert!(notes.iter().any(|n| n.contains("the burst ends (3 failures in a row), 12 evaluations skipped")), "{notes:?}");
    // two in a row restart twice, then it goes on
    let (p, _, _, kept, _) = run(|n| (6..=7).contains(&n));
    assert_eq!((p.failed, p.skipped, kept.len()), (2, 0, 18));
    // a failed start geometry has nothing to go back to: the burst ends there
    let (p, notes, _, kept, _) = run(|n| n == 1);
    assert_eq!((p.failed, p.skipped, kept.len()), (1, 19, 0));
    assert!(notes.iter().any(|n| n.contains("the burst ends (its start geometry: no solved geometry to go back to), 19 evaluations skipped")), "{notes:?}");
    // the last evaluation failing ends nothing
    let (p, _, _, kept, _) = run(|n| n == 20);
    assert_eq!((p.failed, p.skipped, kept.len()), (1, 0, 19));
}

/// A teacher whose two-nucleus solves at separations `slow` picks take `secs` unless aborted first
/// (polling `should_abort` every millisecond, like the GPU teacher every SCF iteration).
struct Slow {
    inner: PairMorse,
    slow: fn(f64) -> bool,
    secs: f64,
    resets: usize,
}

impl ForceProvider for Slow {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        if scene.nuclei.len() == 2 && (self.slow)(dist(scene, 0, 1)) {
            let t0 = Instant::now();
            while t0.elapsed().as_secs_f64() < self.secs {
                if should_abort() {
                    return Err(ProviderError::Cancelled);
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        self.inner.evaluate(scene, should_abort)
    }
    fn name(&self) -> &'static str {
        "pair morse"
    }
    fn reset(&mut self) {
        self.resets += 1;
    }
    fn discontinuity(&mut self) {}
}

/// User report 2026-10-20 (module doc, "The time limit"): after three successful sample solves, one
/// that runs past 5× their median (never below the floor — 0.05 s here, for a quick test) is given
/// up: failed, `ScfEnd::TimeLimit`, never kept, and the stretched scan's further-out points are
/// skipped as after a stall. Before three solves there is no limit; a user abort is still a cancel.
#[test]
fn a_sample_solve_past_the_time_limit_is_given_up_like_a_stall() {
    use sim::activity::{ScfEnd, TeachOutcome};
    let scene = pair(1, 1, 1.4, cell(49));
    let plan = SamplingPlan { perturbations: 0, hot_burst_steps: 0, scan_points_per_pair: 12, ..SamplingPlan::default() };
    let run = |slow: fn(f64) -> bool, secs: f64| {
        let mut teacher = Slow { inner: PairMorse::h2(), slow, secs, resets: 0 };
        let mut s = SurrogateKs::new(Box::new(Shared(Shared::new(PairMorse::h2()).handle())), FallbackPolicy::default()).with_teacher_tol(1e-6);
        let mut sampler = Sampler::new(&scene, &plan).unwrap().with_time_limit_floor(0.05);
        let t0 = Instant::now();
        while sampler.step(&mut teacher, &mut s, &mut never()).unwrap().is_some() {}
        let wall = t0.elapsed().as_secs_f64();
        (sampler.progress(), sampler.take_notes(), Trainable::take_teach_events(&mut s), sampler.samples_added(), sampler.time_limit_s(), wall, teacher.resets)
    };
    // point 9 (1.50 r₀) would take 2 s: given up at the 0.05 s floor; points 10–12 skipped
    let (p, notes, events, added, limit, wall, resets) = run(|r| r > 1.45 * 1.4 && r < 1.55 * 1.4, 2.0);
    println!("timed out: {p:?} limit {limit:?} wall {wall:.2} s\n  {notes:?}");
    assert_eq!(limit, Some(0.1), "5× a microsecond median is under the floor, doubled by the time-out (the last solve)");
    assert_eq!(resets, 2, "the reference and the first scan point: nothing is solved after the time-out");
    assert!(wall < 1.5, "given up, not waited out ({wall:.2} s; wall clock, loose for a loaded machine)");
    assert_eq!((p.done, p.total, p.failed, p.skipped, added), (13, 13, 1, 3, 8));
    let failed: Vec<_> = events.iter().filter(|e| e.outcome == TeachOutcome::Failed).collect();
    assert_eq!(failed.len(), 1);
    assert_eq!((failed[0].scf_end, failed[0].scf_steps), (Some(ScfEnd::TimeLimit), None));
    assert!(failed[0].detail.contains("the pass's time limit (5× its median solve of"), "{}", failed[0].detail);
    assert!(notes.iter().any(|n| n.contains("point 9 of 12 (r = 1.50 r₀) failed (time limit); further-out points skipped")), "{notes:?}");
    // a time-out at r₀ (point 4) skips nothing, and the solve after it is cold (review MUST-1): the
    // reference, the first scan point and point 5 reset the teacher; a success clears the doubling
    let (p, _, events, added, limit, _, resets) = run(|r| (r / 1.4 - 1.0).abs() < 0.01, 2.0);
    assert_eq!((p.failed, p.skipped, added, resets), (1, 0, 11, 3), "{p:?}");
    assert_eq!(limit, Some(0.05));
    assert!(events.iter().any(|e| e.scf_end == Some(ScfEnd::TimeLimit)));
    // a time-out says nothing about the bond further out: H–H at r₀ = 3 Bohr reaches past 5/6 r_cut,
    // and its shell check (1.7 r₀ = 5.1 Bohr) is still solved after a time-out at 1.2 r₀ (review SHOULD-2)
    let wide = pair(1, 1, 3.0, cell(49));
    let mut teacher = Slow { inner: PairMorse::h2(), slow: |r| (r / 3.0 - 1.2).abs() < 0.01, secs: 2.0, resets: 0 };
    let mut s = SurrogateKs::new(Box::new(Shared(Shared::new(PairMorse::h2()).handle())), FallbackPolicy::default()).with_teacher_tol(1e-6);
    let mut sampler = Sampler::new(&wide, &plan).unwrap().with_time_limit_floor(0.05);
    while sampler.step(&mut teacher, &mut s, &mut never()).unwrap().is_some() {}
    let checks = sampler.take_shell_checks();
    assert_eq!(checks.len(), 1, "the shell check is kept and solved");
    assert!((checks[0].r - 5.1).abs() < 1e-9, "{}", checks[0].r);
    assert_eq!(sampler.progress().failed, 1);
    // the first scan point (0.7 r₀) is slow too, but no limit stands before three solves: it is waited out
    let (p, _, _, added, _, wall, _) = run(|r| r < 0.75 * 1.4, 0.3);
    assert_eq!((p.failed, p.skipped, added), (0, 0, 12), "{p:?}");
    assert!(wall >= 0.3, "{wall:.2} s");
    // a user abort during a timed solve stays a cancel: the job is kept for the next step
    let mut teacher = Slow { inner: PairMorse::h2(), slow: |r| r > 1.45 * 1.4, secs: 2.0, resets: 0 };
    let mut s = SurrogateKs::new(Box::new(Shared(Shared::new(PairMorse::h2()).handle())), FallbackPolicy::default()).with_teacher_tol(1e-6);
    let mut sampler = Sampler::new(&scene, &plan).unwrap();
    for _ in 0..9 {
        sampler.step(&mut teacher, &mut s, &mut never()).unwrap();
    }
    let before = sampler.progress().done;
    assert!(matches!(sampler.step(&mut teacher, &mut s, &mut || true), Err(ProviderError::Cancelled)));
    assert_eq!((sampler.progress().done, sampler.progress().failed), (before, 0));
}

/// The geometries a teacher was asked about, as flat coordinate lists.
fn seen_positions(t: &PairMorse) -> Vec<Vec<f64>> {
    t.seen.iter().map(|s| s.nuclei.iter().flat_map(|n| n.pos).collect()).collect()
}

/// User report 2026-10-20 (module doc, "Checkpoints"): a pass checkpointed after any solve and
/// resumed from the text asks the teacher about exactly the geometries the uninterrupted pass would
/// have — every queued perturbation, scan point and shell check bit for bit, and the burst's next
/// evaluation (the thermostat's later kicks come from a fresh stream) — with the same counters; the
/// text of a resumed pass is the text it was resumed from.
#[test]
fn a_checkpointed_pass_resumes_with_the_same_remaining_jobs() {
    // three nuclei (three scans, one of them reaching the shell), a burst, a cell-guard skip or two
    let scene = Scene {
        nuclei: vec![Nucleus::new(1, [-1.5, 0.0, 0.0]), Nucleus::new(1, [1.5, 0.0, 0.0]), Nucleus::new(1, [0.0, 1.3, 0.0])],
        cell: cell(49),
        charge: 0,
        multiplicity: None,
        sigma: None,
        external_field: [0.0; 3],
        ionic: sim::IonicPolicy::None,
    };
    let plan = SamplingPlan { perturbations: 5, hot_burst_steps: 8, scan_points_per_pair: 6, seed: 11, ..SamplingPlan::default() };
    let new_surrogate = || SurrogateKs::new(Box::new(Shared(Shared::new(PairMorse::h2()).handle())), FallbackPolicy::default()).with_teacher_tol(1e-6);
    // the uninterrupted pass
    let mut whole = PairMorse::h2();
    let mut s = new_surrogate();
    let mut sampler = Sampler::new(&scene, &plan).unwrap();
    let mut texts = Vec::new();
    loop {
        texts.push((whole.seen.len(), sampler.checkpoint("prov", 0, 0).expect("a running regular pass checkpoints"), sampler.progress()));
        if sampler.step(&mut whole, &mut s, &mut never()).unwrap().is_none() {
            break;
        }
    }
    let end = sampler.progress();
    assert!(sampler.checkpoint("prov", 0, 0).is_none(), "a finished pass has nothing to resume");
    let all = seen_positions(&whole);
    let burst_start = all.len() - plan.hot_burst_steps;
    println!("{} solves, {} checkpoints, burst from solve {burst_start}", all.len(), texts.len());
    for (k, (solved, text, progress)) in texts.iter().enumerate() {
        let mut resumed = Sampler::resume(text).unwrap_or_else(|e| panic!("checkpoint {k}: {e}"));
        assert_eq!(resumed.checkpoint("prov", 0, 0).as_deref(), Some(text.as_str()), "checkpoint {k} round-trips");
        let p = resumed.progress();
        let restarted_refs = progress.stage == "references" && progress.done > 0;
        if !restarted_refs {
            // (the stage: the resumed pass names the next job, the running one the job it last ran)
            assert_eq!((p.done, p.total, p.skipped, p.failed), (progress.done, progress.total, progress.skipped, progress.failed), "checkpoint {k}");
        }
        let mut t = PairMorse::h2();
        let mut s2 = new_surrogate();
        while resumed.step(&mut t, &mut s2, &mut never()).unwrap().is_some() {}
        let rest = seen_positions(&t);
        let want: Vec<Vec<f64>> = all[*solved..].iter().filter(|g| g.len() > 3).cloned().collect();
        let got: Vec<Vec<f64>> = rest.iter().filter(|g| g.len() > 3).cloned().collect();
        assert_eq!(got.len(), want.len(), "checkpoint {k}: the same number of molecule solves left");
        // every non-burst job bit for bit, and the burst's next evaluation
        let non_burst = all[(*solved).min(burst_start)..burst_start].iter().filter(|g| g.len() > 3).count();
        let exact = (non_burst + 1).min(want.len());
        assert_eq!(got[..exact], want[..exact], "checkpoint {k}: the remaining jobs");
        let q = resumed.progress();
        assert_eq!((q.done, q.total, q.skipped, q.failed, q.finished), (end.done, end.total, end.skipped, end.failed, true), "checkpoint {k}");
        assert_eq!(resumed.samples_added(), sampler.samples_added(), "checkpoint {k}");
    }
    // a damaged or truncated file is refused, never half-read
    let text = &texts[texts.len() / 2].1;
    assert!(Sampler::resume(&text[..text.len() / 2]).is_err());
    assert!(Sampler::resume(&text.replacen("plan ", "plan x ", 1)).is_err());
    assert!(checkpoint_info("hello").is_err());
}

/// The surrogate's side (`Trainable::pass_checkpoint` / `resume_pass`): the samples file and the
/// checkpoint written together resume the pass in a new surrogate, which ends with the samples the
/// uninterrupted pass would have; a checkpoint is refused on another scene, with another sample
/// count, or while a pass runs.
#[test]
fn a_pass_resumes_in_a_new_surrogate_from_its_samples_and_checkpoint() {
    let scene = pair(1, 1, 1.4, cell(41));
    let plan = SamplingPlan { perturbations: 12, hot_burst_steps: 10, scan_points_per_pair: 8, seed: 2, ..SamplingPlan::default() };
    let mut a = surrogate_over(&Shared::new(PairMorse::h2()));
    let mut refs = AtomReferences::default();
    Trainable::start_pass(&mut a, &scene, &plan, &mut refs).unwrap();
    for _ in 0..15 {
        Trainable::pass_step(&mut a, &mut never()).unwrap();
    }
    let text = Trainable::pass_checkpoint(&a).expect("a running pass");
    let path = tmp("resume.model");
    Trainable::save(&a, &path).unwrap();
    let held = a.samples();
    // the uninterrupted pass, for comparison
    while Trainable::pass_step(&mut a, &mut never()).unwrap().is_some() {}
    let info = checkpoint_info(&text).unwrap();
    assert_eq!((info.samples, info.pass_start, info.done, info.total), (held, 0, 15, Trainable::progress(&a).unwrap().total));

    let mut b = surrogate_over(&Shared::new(PairMorse::h2()));
    Trainable::load(&mut b, &path, &scene).unwrap();
    // another scene, another count, a running pass: refused, nothing changed
    let mut moved = scene.clone();
    moved.nuclei[1].pos[0] += 0.1;
    assert!(Trainable::resume_pass(&mut b, &moved, &text, &mut refs).unwrap_err().contains("not the one"));
    let mut c = surrogate_over(&Shared::new(PairMorse::h2()));
    assert!(Trainable::resume_pass(&mut c, &scene, &text, &mut refs).unwrap_err().contains("not saved together"));
    assert!(!Trainable::pass_active(&b));
    Trainable::resume_pass(&mut b, &scene, &text, &mut refs).unwrap();
    assert!(Trainable::resume_pass(&mut b, &scene, &text, &mut refs).unwrap_err().contains("already running"));
    assert!(Trainable::take_notes(&mut b).iter().any(|n| n.contains("sampling pass resumed at sample 16 of")));
    while Trainable::pass_step(&mut b, &mut never()).unwrap().is_some() {}
    assert_eq!(b.samples(), a.samples(), "the resumed pass ends with the uninterrupted one's samples");
    assert_eq!(Trainable::fitted(&b), Trainable::fitted(&a));
    let tags = |s: &SurrogateKs| tag_counts(s);
    assert_eq!(tags(&b), tags(&a));
    let _ = std::fs::remove_file(&path);
}
