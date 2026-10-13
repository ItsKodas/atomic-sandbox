use super::*;
use crate::dynamics::{DynCommand, Dynamics};
use crate::provider::{EvalStats, Source};
use crate::scene::Cell;
use crate::surrogate_ks::FallbackPolicy;
use crate::test_providers::{MorseDimer, Shared};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

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
}

impl PairMorse {
    fn h2() -> PairMorse {
        let m = MorseDimer::h2();
        PairMorse { d: m.d, a: m.a, r_e: m.r_e, e_atom: m.e_atom, seen: Vec::new(), solves: Arc::new(AtomicUsize::new(0)), resets: 0, discontinuities: 0 }
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
        let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        let (energy, forces) = self.energy_forces(&pos);
        let mut stats = EvalStats::default();
        stats.energy.total = energy;
        Ok(ForceResult { energy, forces, stats, display: None })
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
    // the first solve: one H at the cell centre of the scene's own grid, neutral, lowest spin
    let atom = &t.seen[0];
    assert_eq!(atom.nuclei.len(), 1);
    assert_eq!((atom.nuclei[0].z, atom.nuclei[0].pos), (1, [0.0; 3]));
    assert_eq!((atom.cell.clone(), atom.charge, atom.multiplicity, atom.sigma), (scene.cell.clone(), 0, None, None));
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
    for (sc, sampler_scene) in [(a, pair(1, 1, 1.4, cell(49))), (b, pair(1, 1, 2.8, cell(49)))] {
        let mut teacher = Shared::new(PairMorse::h2());
        let h = teacher.handle();
        let mut s = surrogate_over(&teacher);
        let n = run_sampling(&sampler_scene, &plan, &mut teacher, &mut s, &mut |_| {}, &mut never()).unwrap();
        assert_eq!(n, 12);
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

fn surrogate_of(d: &mut Dynamics) -> &mut SurrogateKs {
    d.provider_mut().as_any_mut().and_then(|a| a.downcast_mut::<SurrogateKs>()).expect("a SurrogateKs")
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
    let mut d = Dynamics::new(scene, Box::new(surrogate_over(&teacher)), 0.25, 1);
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
    assert_eq!(p.stage, "cancelled");
    assert!(h.lock().unwrap().discontinuities > disc0, "the live run must not extrapolate from the last sample geometry");
    {
        let s = surrogate_of(&mut d);
        // the live evaluations so far (fallbacks) + 2 perturbations, refitted
        assert!(s.model().is_some(), "the partial set was refitted");
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
        let s = surrogate_of(&mut d);
        (s.trusted(), s.report().unwrap().n_samples)
    };
    println!("after the pass: {n} samples fitted, trusted {trusted}");
    assert_eq!(n, surrogate_of(&mut d).samples(), "the pass ends with a refit over everything");
    assert!(trusted, "an exact pair potential sampled 60+ times is learned");
    // the refit moved the surface: the next tick refreshes at the unchanged positions
    let refreshed = d.tick(0.0, &mut never());
    assert_eq!(refreshed.source, Source::Surrogate, "{:?}", refreshed.eval.notes);

    d.apply(DynCommand::ForgetModel);
    let snap = d.tick(0.0, &mut never());
    assert_eq!(snap.sampling, None);
    assert!(surrogate_of(&mut d).model().is_none());
    assert_eq!(surrogate_of(&mut d).samples(), 1, "only the refresh evaluation's fallback sample");
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
    let mut d = Dynamics::new(scene.clone(), Box::new(surrogate_over(&teacher)), 0.25, 1);
    let path = tmp("h2.model");
    d.apply(DynCommand::SaveModel(path.clone()));
    assert!(d.snapshot().last_rejection.expect("nothing fitted yet").contains("no fitted model"));
    d.apply(DynCommand::StartSampling(SamplingPlan { perturbations: 30, hot_burst_steps: 20, scan_points_per_pair: 12, ..SamplingPlan::default() }));
    while !d.tick(f64::INFINITY, &mut never()).sampling.unwrap().finished {}
    d.apply(DynCommand::SaveModel(path.clone()));
    let snap = d.snapshot();
    assert_eq!(snap.last_rejection, None);
    assert!(snap.notes.last().unwrap().contains("saved"), "{:?}", snap.notes);
    let (model, report, n, trusted) = {
        let s = surrogate_of(&mut d);
        (s.model().unwrap().clone(), s.report().unwrap().clone(), s.samples(), s.trusted())
    };
    assert!(trusted);

    // same settings, fresh provider: the model is applied and answers at once
    let teacher2 = Shared::new(PairMorse::h2());
    let h2 = teacher2.handle();
    let mut d2 = Dynamics::new(scene.clone(), Box::new(surrogate_over(&teacher2)), 0.25, 1);
    d2.apply(DynCommand::LoadModel(path.clone()));
    let snap = d2.snapshot();
    assert_eq!(snap.last_rejection, None, "{:?}", snap.notes);
    println!("loaded: {:?}", snap.notes.last());
    {
        let s = surrogate_of(&mut d2);
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
    let mut d3 = Dynamics::new(other, Box::new(surrogate_over(&Shared::new(PairMorse::h2()))), 0.25, 1);
    d3.apply(DynCommand::LoadModel(path.clone()));
    let snap = d3.snapshot();
    let r = snap.last_rejection.expect("a model from another grid is refused");
    println!("refused: {r}");
    assert!(r.contains("49³") && r.contains("41³"), "{r}");
    assert!(snap.notes.last().unwrap().contains("refused"), "{:?}", snap.notes);
    assert!(surrogate_of(&mut d3).model().is_none());

    // a corrupt file: an error message, never a panic
    let bad = tmp("corrupt.model");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.truncate(bytes.len() / 2);
    std::fs::write(&bad, &bytes).unwrap();
    d3.apply(DynCommand::LoadModel(bad.clone()));
    let r = d3.snapshot().last_rejection.expect("refused");
    println!("corrupt: {r}");
    assert!(surrogate_of(&mut d3).model().is_none());
    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_file(bad);
}

/// Optional GPU smoke test (release; skips without an adapter): a tiny real pass on H₂ at 33³.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn gpu_sampling_smoke_h2_33() {
    let Some(ctx) = gpu::GpuContext::new_headless() else {
        eprintln!("sim: no GPU adapter — skipped");
        return;
    };
    let mut teacher = crate::provider::GpuKs::new(ctx, 1e-6);
    teacher.set_display(false);
    let mut s = SurrogateKs::new(Box::new(crate::test_providers::Zero), FallbackPolicy::default()).with_teacher_tol(1e-6);
    let scene = pair(1, 1, 1.4, cell(33));
    let plan = SamplingPlan { perturbations: 3, hot_burst_steps: 4, scan_points_per_pair: 3, ..SamplingPlan::default() };
    let t0 = std::time::Instant::now();
    let mut last = None;
    let n = run_sampling(&scene, &plan, &mut teacher, &mut s, &mut |p| last = Some(p), &mut never()).unwrap();
    let p = last.unwrap();
    let e_h = s.reference_energy(1).unwrap();
    println!("GPU H₂ 33³: {n} samples + 1 reference in {:.2} s; E_H(33³) = {e_h:+.6} Ha; {p:?}", t0.elapsed().as_secs_f64());
    assert_eq!((n, p.failed, p.skipped), (10, 0, 0));
    // the 33³ box's H atom (Task 4 measured −0.4143): confinement-raised above LSDA's −0.479
    assert!((-0.48..-0.40).contains(&e_h), "{e_h}");
    s.refit().expect("refit");
}
