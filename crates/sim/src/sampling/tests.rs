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
    /// (Z, multiplicity, energy) overrides for a single atom (ruling S9 tests).
    atom_energy: Vec<(u8, u32, f64)>,
    /// A scripted box (final review I1): `(x_max, rho, bias)` — a geometry with a nucleus more
    /// than `x_max` off-centre along any axis is reported with face density `rho` and its energy
    /// raised by `bias` (the box's confinement); every other one with a face density of 1e-7.
    confine: Option<(f64, f64, f64)>,
}

impl PairMorse {
    fn h2() -> PairMorse {
        let m = MorseDimer::h2();
        PairMorse { d: m.d, a: m.a, r_e: m.r_e, e_atom: m.e_atom, seen: Vec::new(), solves: Arc::new(AtomicUsize::new(0)), resets: 0, discontinuities: 0, atom_energy: Vec::new(), confine: None }
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
    let rho = 5.0 * crate::surrogate_ks::BOX_AFFECTED_FACE_DENSITY;
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

    // ruling S18: a model taught at the interactive tol 1e-5 (a file saved before the sample
    // tolerance) is refused by a teacher at 1e-6 — both tolerances named, never mixed
    let loose = tmp("h2-tol1e-5.model");
    let mut sc = surrogate::sidecar::read(&path).unwrap();
    sc.model.provenance.tol = 1e-5;
    surrogate::sidecar::write(&loose, &sc.model, &sc.accumulator, sc.report.as_ref(), &sc.samples, sc.reference_rule.as_deref()).unwrap();
    let mut d4 = Dynamics::new(scene.clone(), Box::new(surrogate_over(&Shared::new(PairMorse::h2()))), 0.25, 1);
    let held = surrogate_of(&mut d4).samples(); // the construction's own rule-1 answer
    d4.apply(DynCommand::LoadModel(loose.clone()));
    let r = d4.snapshot().last_rejection.expect("a model taught at another tolerance is refused");
    println!("refused: {r}");
    assert!(r.contains("at tol 1e-5") && r.contains("at tol 1e-6"), "{r}");
    assert!(surrogate_of(&mut d4).model().is_none() && surrogate_of(&mut d4).samples() == held, "nothing of the 1e-5 file was installed");
    let _ = std::fs::remove_file(loose);

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
    // review I1: the surrogate's OWN teacher, so samples carry its true identity
    let mut teacher = crate::provider::GpuKs::new(ctx, 1e-6);
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
    let key = ReferenceCache::atom_provenance(&s.provenance_for(&scene));
    assert_eq!(sampler.take_references().get(&key, 1).map(|r| r.1), Some(Some(2)));
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
    let key = ReferenceCache::atom_provenance(&s.provenance_for(&scene));
    assert_eq!(cache.get(&key, 8), Some((-1.0, Some(3))));
    assert_eq!(cache.get(&key, 1), Some((-0.478, Some(2))));
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
    let mut d = Dynamics::new(pair(1, 1, 1.45, cell(41)), Box::new(surrogate_over(&teacher)), 0.25, 1);
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
    assert!(surrogate_of(&mut d).model().is_none(), "no refit from a rejected command");
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
    assert!(surrogate_of(&mut d).model().is_some(), "and what it gathered is refitted");
}

/// Ruling S9 and old files: a sidecar without the reference rule loads (its fit is consistent
/// with its references) but its references are not cached — the next pass re-solves them —
/// and the model is saved with the rule only once every reference is S9.
#[test]
fn references_from_a_file_without_the_reference_rule_are_re_solved() {
    let teacher = Shared::new(PairMorse::h2());
    let scene = pair(1, 1, 1.45, cell(41));
    let mut d = Dynamics::new(scene.clone(), Box::new(surrogate_over(&teacher)), 0.25, 1);
    d.apply(DynCommand::StartSampling(SamplingPlan { perturbations: 30, hot_burst_steps: 20, scan_points_per_pair: 12, ..SamplingPlan::default() }));
    while !d.tick(f64::INFINITY, &mut never()).sampling.unwrap().finished {}
    // a pre-S9 file: the same contents, no reference rule
    let old = tmp("pre-s9.model");
    {
        let s = surrogate_of(&mut d);
        surrogate::sidecar::write(&old, s.model().unwrap(), s.accumulator().unwrap(), s.report(), &s.store().unwrap().samples, None).unwrap();
        let marked = tmp("s9.model");
        s.save_sidecar(&marked).unwrap();
        assert_eq!(surrogate::sidecar::read(&marked).unwrap().reference_rule.as_deref(), Some(surrogate::sidecar::REFERENCE_RULE));
    }
    let teacher2 = Shared::new(PairMorse::h2());
    let h2 = teacher2.handle();
    let mut d2 = Dynamics::new(scene.clone(), Box::new(surrogate_over(&teacher2)), 0.25, 1);
    d2.apply(DynCommand::LoadModel(old.clone()));
    let snap = d2.snapshot();
    assert_eq!(snap.last_rejection, None);
    assert!(snap.notes.iter().any(|n| n.contains("predate ruling S9")), "{:?}", snap.notes);
    assert_eq!(surrogate_of(&mut d2).legacy_references(), vec![1]);
    // saving now would not vouch for the references
    let resaved = tmp("resaved.model");
    d2.apply(DynCommand::SaveModel(resaved.clone()));
    assert_eq!(surrogate::sidecar::read(&resaved).unwrap().reference_rule, None);
    // the next pass solves the atom again (nothing cached from the file)
    d2.apply(DynCommand::StartSampling(SamplingPlan { perturbations: 1, hot_burst_steps: 0, scan_points_per_pair: 0, ..SamplingPlan::default() }));
    while !d2.tick(f64::INFINITY, &mut never()).sampling.unwrap().finished {}
    assert_eq!(h2.lock().unwrap().seen.iter().filter(|x| x.nuclei.len() == 1).count(), 1, "the reference was re-solved");
    assert!(surrogate_of(&mut d2).legacy_references().is_empty());
    d2.apply(DynCommand::SaveModel(resaved.clone()));
    assert_eq!(surrogate::sidecar::read(&resaved).unwrap().reference_rule.as_deref(), Some(surrogate::sidecar::REFERENCE_RULE));
}

/// Ruling S9 on the real teacher: an isolated O atom solved at multiplicities 1, 3, 5 — Hund's
/// rule should emerge as the triplet. **Ignored: documented finding (fix round 1).** With the
/// project's h = 0.25, σ = h (the `h2o` preset's settings) it does not: at 33³ (26 s for the
/// three cold solves) the singlet is lowest, E = −36.5650 Ha, the triplet −36.3380 (+0.227 Ha),
/// the quintet −35.5761; at 57³ the singlet is −36.5787 Ha and the triplet and quintet do not
/// converge cold in 200 SCF iterations (Δρ 1.4e-4 / 1.8e-3). LSDA puts the O triplet below the
/// singlet by tens of mHa, and E ≈ −36.6 Ha is half the resolved LSDA atom (≈ −74.5): the
/// smeared O core is far from resolved, and an open-shell atom with integer occupations over the
/// degenerate 2p levels is a solver question (Fable), not a sampling one. The S9 scan itself does
/// what it should — it picks the lowest-energy multiplicity the teacher converges — so on this
/// surface a scene with O gets the singlet reference, consistently with the teacher's own
/// molecular energies. Run with `--ignored` after a solver change.
#[test]
#[ignore = "documented finding: at h = 0.25 the O atom's singlet is lowest / the triplet does not converge (see doc)"]
fn gpu_oxygen_reference_multiplicity() {
    let Some(ctx) = gpu::GpuContext::new_headless() else {
        eprintln!("sim: no GPU adapter — skipped");
        return;
    };
    let mut teacher = crate::provider::GpuKs::new(ctx, 1e-6);
    teacher.set_display(false);
    let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_teacher_tol(1e-6);
    let scene = Scene { nuclei: vec![Nucleus::new(8, [0.0; 3])], cell: cell(33), charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3] };
    let plan = SamplingPlan { perturbations: 0, hot_burst_steps: 0, scan_points_per_pair: 0, ..SamplingPlan::default() };
    let t0 = std::time::Instant::now();
    let mut sampler = Sampler::new(&scene, &plan).unwrap();
    while sampler.step_own_teacher(&mut s, &mut never()).unwrap().is_some() {}
    let notes = sampler.take_notes();
    println!("O atom at 33³, 3 multiplicities in {:.2} s: {notes:#?}", t0.elapsed().as_secs_f64());
    let key = ReferenceCache::atom_provenance(&s.provenance_for(&scene));
    let (e, m) = sampler.take_references().get(&key, 8).expect("a reference");
    println!("O reference: multiplicity {m:?}, E = {e:.6} Ha");
    assert_eq!(sampler.progress().failed, 0, "{notes:?}");
    assert_eq!(m, Some(3), "Hund's rule emerges for O: {notes:?}");
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

/// An empty scene has no species: a pass over it is refused with the message the app shows.
#[test]
fn a_pass_over_an_empty_scene_is_refused() {
    let empty = Scene { nuclei: Vec::new(), cell: Cell { n: 33, h: 0.25 }, charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3] };
    let plan = SamplingPlan::default();
    assert_eq!(Sampler::new(&empty, &plan).err().as_deref(), Some(EMPTY_SCENE));
    assert_eq!(plan.solves(&empty), Err(EMPTY_SCENE.to_string()));
}
