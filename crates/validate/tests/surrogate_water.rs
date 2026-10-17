//! The 2026-10-18 pseudo-O water experiments (E1, E2 of `.superpowers/surrogate-water-investigation.md`
//! §5): report-only, `#[ignore]`d, GPU. They reproduce the user's session — water under the default
//! `CoresOnly` policy at the Accurate grid (95³ × 0.15 Bohr, the h2o preset's 14 Bohr box), the GPU
//! teacher at the sampling tolerance 1e-6 — and print what the surrogate decided and why
//! (`SurrogateKs::inspect` / `explain`, the fallback rules' histogram from the activity events).
//! Nothing is asserted beyond "the run did not error out of the harness": they are evidence for the
//! trust fixes, not gates (rule 6 gates stay in `surrogate.rs`).
//!
//! - `water_pseudo_accurate_strict_and_draft` (E1, ≈ 30 min): the default pass (seed 7) on water;
//!   inspect; (b) strict: 400 steps CSVR 300 K (τ 20 fs) then NVE, logging T, E_total, E_cons, the
//!   source, the refusal-rule histogram, γ_max and ν every 50 steps (teacher solves capped,
//!   `E1_STRICT_CAP`); (c) draft from the same 300 K state, NVE, every 40th frame solved by a second
//!   teacher (gap/atom, worst force residual / bound, γ), the model's E/F finite-difference check at
//!   20 frames, background audits every 200 answered steps (draft audits are kept as samples — the
//!   investigation's D3); refit, inspect, and strict again from the 300 K state; (e) the log's two
//!   pathological frames (an H at 0.9 Bohr from O, an O–H at 4.63 Bohr) taught as kept samples, refit,
//!   inspect, strict again. Saves the pass's model to `$CARGO_TARGET_TMPDIR/water_e1.model` for E2.
//! - `water_plus_two_h_box_rule` (E2, ≈ 40 min): from that model (trained afresh if missing), water +
//!   2 H (two H 3 Bohr from the O and 3.2 Bohr apart) centred (A) and shifted 1.5 Bohr toward a face
//!   (B); per variant the user's 214-job pass (34 perturbations, 8 points per pair scan, 100-step
//!   2000 K burst, seed 7); the pass's outcome counts (kept, box-affected, unfit, failed, skipped);
//!   inspect (the store by composition × state: are the 5-atom samples in the fit?); 200 strict steps
//!   (teacher solves capped, `E2_STRICT_CAP`) and 200 draft steps from the pass centre with the model
//!   checked against a fresh teacher solve at steps 0, 5, 10, 20, 50, 100, 200.
//!
//! Run: `cargo test -p validate --release --test surrogate_water -- --ignored --nocapture --test-threads 1`.
use gpu::GpuContext;
use providers::{FallbackPolicy, SurrogateKs};
use sim::activity::TeachOrigin;
use sim::{Cell, DynCommand, Dynamics, ForceProvider, IonicPolicy, Nucleus, SamplingPlan, Scene, SharedTrainable, Source, Trainable};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::{Arc, Mutex};
use std::time::Instant;

mod common;
use common::{fit_line, never, solve, Locked, DT_FS, TOL_H2};

/// The Accurate grid of the h2o preset (the app's `quality::preset("h2o", 0.15)`: 14 Bohr kept).
const N: usize = 95;
const H_GRID: f64 = 0.15;
const SEED: u64 = 7;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn cell() -> Cell {
    Cell { n: N, h: H_GRID }
}

fn scene_of(nuclei: Vec<Nucleus>) -> Scene {
    Scene { nuclei, cell: cell(), charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3], ionic: IonicPolicy::CoresOnly }
}

/// The h2o preset's geometry (O–H 1.81 Bohr, 104.5°), O at the origin.
fn water() -> Scene {
    scene_of(vec![Nucleus::new(8, [0.0; 3]), Nucleus::new(1, [1.43, 1.11, 0.0]), Nucleus::new(1, [-1.43, 1.11, 0.0])])
}

/// Water + 2 H: the two new H 3.0 Bohr from the O and 3.2 Bohr apart (on the far side of the O from
/// the water's H), the cluster's y-extent centred on the origin and then shifted by `dy`.
fn water_plus_two_h(dy: f64) -> Scene {
    let raw = [(8u8, [0.0, 0.0, 0.0]), (1, [1.43, 1.11, 0.0]), (1, [-1.43, 1.11, 0.0]), (1, [1.6, -2.54, 0.0]), (1, [-1.6, -2.54, 0.0])];
    let shift = -(1.11 + -2.54) / 2.0 + dy;
    scene_of(raw.iter().map(|(z, p)| Nucleus::new(*z, [p[0], p[1] + shift, p[2]])).collect())
}

fn gpu() -> Option<GpuContext> {
    common::gpu("surrogate_water")
}

/// The estimated solver as the app builds it: the GPU teacher at the sampling tolerance, the
/// default policy, a background auditor (a second teacher) when `audited`.
fn student(ctx: &GpuContext, audited: bool) -> SurrogateKs {
    let (p, _log) = Locked::new(ctx.clone(), TOL_H2);
    let s = SurrogateKs::new(Box::new(p), FallbackPolicy::default()).with_teacher_tol(TOL_H2);
    if audited { s.with_auditor(Box::new(Locked::new(ctx.clone(), TOL_H2).0)) } else { s }
}

fn pair_min(sc: &Scene) -> f64 {
    let p: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
    let mut m = f64::INFINITY;
    for i in 0..p.len() {
        for j in i + 1..p.len() {
            m = m.min((0..3).map(|a| (p[i][a] - p[j][a]).powi(2)).sum::<f64>().sqrt());
        }
    }
    m
}

fn fmax(f: &[[f64; 3]]) -> f64 {
    f.iter().map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()).fold(0.0f64, f64::max)
}

/// A sampling pass through the `Trainable` face, as the worker drives it (`start_pass`, then
/// `pass_step` until it ends; the pass's own refit and its notes), with the pass's activity
/// outcomes counted. Returns the wall seconds.
fn run_pass(s: &mut SurrogateKs, sc: &Scene, plan: &SamplingPlan, out: &mut String) -> f64 {
    let t = Instant::now();
    let mut refs = sim::AtomReferences::default();
    s.start_pass(sc, plan, &mut refs).unwrap_or_else(|e| panic!("start_pass: {e}"));
    let mut outcomes: BTreeMap<String, usize> = BTreeMap::new();
    loop {
        match s.pass_step(&mut never()) {
            Ok(Some(_)) => {}
            Ok(None) => break,
            Err(sim::ProviderError::Cancelled) => continue, // a busy device: retried
            Err(e) => {
                let _ = writeln!(out, "  the pass ended on an error: {e}");
                break;
            }
        }
        for e in s.take_teach_events() {
            *outcomes.entry(format!("{:?}", e.outcome)).or_default() += 1;
        }
    }
    for e in s.take_teach_events() {
        *outcomes.entry(format!("{:?}", e.outcome)).or_default() += 1;
    }
    let p = s.progress().expect("a pass ran");
    let wall = t.elapsed().as_secs_f64();
    let _ = writeln!(out, "  pass ({} perturbations, {} points/pair, {}-step burst at {} K, seed {}): {} jobs, {} done, {} skipped, {} failed in {wall:.0} s; outcomes {outcomes:?}",
                     plan.perturbations, plan.scan_points_per_pair, plan.hot_burst_steps, plan.hot_burst_t, plan.seed, p.total, p.done, p.skipped, p.failed);
    for n in s.take_notes() {
        let _ = writeln!(out, "    note: {n}");
    }
    wall
}

/// What one MD phase did.
#[derive(Default)]
struct Phase {
    steps: usize,
    answered: usize,
    fallbacks: usize,
    rules: BTreeMap<String, usize>,
    outcomes: BTreeMap<String, usize>,
    stopped: Option<String>,
    /// (t in ps, E_conserved) of the current run of model-answered steps, and the slopes of those
    /// of ≥ 200 steps (Ha/ps).
    stretch: (Vec<f64>, Vec<f64>),
    slopes: Vec<f64>,
}

impl Phase {
    fn close_stretch(&mut self) {
        if self.stretch.0.len() >= 200 {
            self.slopes.push(fit_line(&self.stretch.0, &self.stretch.1).0);
        }
        self.stretch = (Vec::new(), Vec::new());
    }

    fn summary(&self) -> String {
        let worst = self.slopes.iter().fold(0.0f64, |m, s| m.max(s.abs()));
        format!(
            "{} steps: {} model answers, {} teacher steps; refusal rules {:?}; teacher outcomes {:?}; E_cons drift on {} model-answered stretches (≥ 200 steps): worst |slope| {worst:.2e} Ha/ps{}",
            self.steps,
            self.answered,
            self.fallbacks,
            self.rules,
            self.outcomes,
            self.slopes.len(),
            self.stopped.as_ref().map_or(String::new(), |s| format!("; STOPPED: {s}"))
        )
    }
}

/// `steps` driver steps on `d` (thermostat as set by the caller), logging every `log_every`; the
/// worker's refits every 25 steps; stops once the teacher has answered `cap` steps, or on an error.
/// `check(i, snapshot)`: a per-step hook (the draft run's teacher checks).
#[allow(clippy::too_many_arguments)]
fn run_phase(d: &mut Dynamics, hs: &Arc<Mutex<SurrogateKs>>, what: &str, steps: usize, cap: usize, log_every: usize, out: &mut String, check: &mut dyn FnMut(usize, &sim::DynSnapshot, &mut String)) -> Phase {
    let mut ph = Phase::default();
    let t0 = Instant::now();
    for i in 0..steps {
        d.apply(DynCommand::Step(1));
        let snap = d.tick(f64::INFINITY, &mut never());
        for e in d.take_teach_events() {
            if matches!(e.origin, TeachOrigin::Fallback | TeachOrigin::Audit) {
                *ph.rules.entry(format!("{:?}/{:?}", e.origin, e.rule)).or_default() += 1;
                *ph.outcomes.entry(format!("{:?}", e.outcome)).or_default() += 1;
            }
        }
        if let Some(err) = &snap.last_error {
            ph.stopped = Some(format!("step {i}: {err}"));
            break;
        }
        ph.steps += 1;
        match snap.source {
            Source::Surrogate => {
                ph.answered += 1;
                ph.stretch.0.push(snap.sim_time / common::au_per_ps());
                ph.stretch.1.push(snap.energy_conserved);
            }
            _ => {
                ph.fallbacks += 1;
                ph.close_stretch();
            }
        }
        check(i, &snap, out);
        if i % log_every == 0 || i + 1 == steps {
            let s = hs.lock().unwrap_or_else(|e| e.into_inner());
            let pos: Vec<[f64; 3]> = snap.scene.nuclei.iter().map(|n| n.pos).collect();
            let z: Vec<u8> = snap.scene.nuclei.iter().map(|n| n.z).collect();
            let _ = writeln!(
                out,
                "    {what} {i:5}: t {:8.2} fs, T {:9.1} K, E_total {:+.6}, E_cons {:+.6}, {:?}, |F|max {:.3}, d_min {:.3}, γ {:.2}, ν {:.1e}, samples {} epoch {} trusted {}{}",
                snap.sim_time_fs(),
                snap.temperature,
                snap.total,
                snap.energy_conserved,
                snap.source,
                fmax(&snap.forces),
                pair_min(&snap.scene),
                s.extrapolation_grade(&pos, &z).unwrap_or(f64::NAN),
                s.novelty(&pos, &z).map_or(f64::NAN, |x| x.0),
                s.samples(),
                s.epoch(),
                s.trusted(),
                s.stuck().map_or(String::new(), |x| format!(" — stuck: {x}"))
            );
        }
        if i % 25 == 24 {
            let mut s = hs.lock().unwrap_or_else(|e| e.into_inner());
            if s.pending_refit() {
                let _ = s.refit();
            }
        }
        if ph.fallbacks >= cap {
            ph.stopped = Some(format!("teacher cap {cap} reached at step {i}"));
            break;
        }
        if snap.temperature > 1e6 {
            ph.stopped = Some(format!("T {:.0} K at step {i}", snap.temperature));
            break;
        }
    }
    ph.close_stretch();
    let _ = writeln!(out, "  {what}: {} ({:.0} s)", ph.summary(), t0.elapsed().as_secs_f64());
    ph
}

/// The model's 5-point finite-difference check at one geometry: worst |F − (−dE/dr)|/max(|F|, 1).
fn fd_check(s: &SurrogateKs, sc: &Scene) -> f64 {
    let m = s.model().expect("a model");
    let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
    let z: Vec<u8> = sc.nuclei.iter().map(|n| n.z).collect();
    let f = m.predict(&pos, &z, false).expect("predict").forces;
    let e = |q: &[[f64; 3]]| m.predict(q, &z, false).expect("predict").energy;
    let h = 1e-3;
    let mut worst = 0.0f64;
    for j in 0..pos.len() {
        for a in 0..3 {
            let at = |x: f64| {
                let mut q = pos.clone();
                q[j][a] = x;
                e(&q)
            };
            let x = pos[j][a];
            let fd = -(-at(x + 2.0 * h) + 8.0 * at(x + h) - 8.0 * at(x - h) + at(x - 2.0 * h)) / (12.0 * h);
            worst = worst.max((fd - f[j][a]).abs() / f[j][a].abs().max(1.0));
        }
    }
    worst
}

/// The model against a fresh teacher solve at `snap`'s geometry: (|ΔE|/N, worst force residual /
/// bound, the teacher's face density).
fn judge(checker: &mut dyn ForceProvider, snap: &sim::DynSnapshot) -> (f64, f64, Option<f64>) {
    let t = solve(checker, &snap.scene, "teacher check");
    let n = snap.scene.nuclei.len() as f64;
    let pol = FallbackPolicy::default();
    let ratio = snap.forces.iter().zip(&t.forces).map(|(fm, ft)| {
        let (d, b) = pol.force_residual(fm, ft);
        d / b
    }).fold(0.0f64, f64::max);
    ((snap.energy_ks - t.energy).abs() / n, ratio, t.stats.face_density)
}

fn model_path() -> std::path::PathBuf {
    std::env::var("SURROGATE_WATER_MODEL").map(Into::into).unwrap_or_else(|_| std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("water_e1.model"))
}

/// E1 (module doc).
#[test]
#[ignore = "experiment E1 (≈ 30 min GPU): cargo test -p validate --release --test surrogate_water water_pseudo -- --ignored --nocapture"]
fn water_pseudo_accurate_strict_and_draft() {
    let Some(ctx) = gpu() else { return };
    let t_test = Instant::now();
    let mut out = String::new();
    let start = water();
    let _ = writeln!(out, "E1: water, CoresOnly, {N}³ × {H_GRID}, GPU teacher tol {TOL_H2}, default policy + background auditor");
    let mut s = student(&ctx, true);
    let plan = SamplingPlan { seed: SEED, ..SamplingPlan::default() };
    run_pass(&mut s, &start, &plan, &mut out);
    let path = model_path();
    match s.save_sidecar(&path) {
        Ok(()) => { let _ = writeln!(out, "  model saved to {}", path.display()); }
        Err(e) => { let _ = writeln!(out, "  model NOT saved ({e})"); }
    }
    let _ = writeln!(out, "(a) after the pass:\n{}", s.inspect(&[("the preset water (the start)", &start)]));
    println!("{out}");
    out.clear();

    let p = SharedTrainable::new(s);
    let hs = p.handle();
    let strict_cap = env_usize("E1_STRICT_CAP", 300);
    // (b) strict: 300 K CSVR, then NVE
    let mut d = Dynamics::new(start.clone(), Box::new(p), DT_FS, SEED);
    d.apply(DynCommand::SetThermostat(Some((300.0, 20.0))));
    let _ = writeln!(out, "(b) strict: {} steps CSVR 300 K τ 20 fs, then {} NVE (teacher cap {strict_cap} per phase)", env_usize("E1_CSVR_STEPS", 400), env_usize("E1_NVE_STEPS", 4000));
    let ph = run_phase(&mut d, &hs, "strict CSVR", env_usize("E1_CSVR_STEPS", 400), strict_cap, 50, &mut out, &mut |_, _, _| {});
    let warm = d.snapshot().scene.clone();
    let _ = writeln!(out, "  the 300 K state (start of (c) and of the strict re-runs): T {:.0} K, d_min {:.3}", warm.temperature(), pair_min(&warm));
    if ph.stopped.is_none() {
        d.apply(DynCommand::SetThermostat(None));
        run_phase(&mut d, &hs, "strict NVE", env_usize("E1_NVE_STEPS", 4000), strict_cap, 50, &mut out, &mut |_, _, _| {});
    }
    let _ = writeln!(out, "  after (b):\n{}", hs.lock().unwrap().inspect(&[("the 300 K state", &warm)]));
    println!("{out}");
    out.clear();

    // (c) draft from the 300 K state, NVE, background audits every 200 answered steps
    {
        let mut s = hs.lock().unwrap();
        s.set_draft(true);
        s.set_audit_every(env_usize("E1_DRAFT_AUDIT_EVERY", 200));
    }
    let mut d = Dynamics::new(warm.clone(), Box::new(SharedTrainable::from_handle(hs.clone())), DT_FS, SEED);
    let mut checker = Locked::new(ctx.clone(), TOL_H2).0;
    let (mut worst_e, mut worst_f, mut worst_fd, mut judged) = (0.0f64, 0.0f64, 0.0f64, 0usize);
    let draft_steps = env_usize("E1_DRAFT_STEPS", 8000);
    let fd_every = (draft_steps / 20).max(1);
    let hs2 = hs.clone();
    let _ = writeln!(out, "(c) draft: {draft_steps} NVE steps from the 300 K state; every 40th frame against a fresh teacher solve; FD every {fd_every}");
    run_phase(&mut d, &hs, "draft NVE", draft_steps, usize::MAX, 200, &mut out, &mut |i, snap, out| {
        if i % 40 == 0 && snap.source == Source::Surrogate {
            let (de, fr, face) = judge(&mut checker, snap);
            let pos: Vec<[f64; 3]> = snap.scene.nuclei.iter().map(|n| n.pos).collect();
            let z: Vec<u8> = snap.scene.nuclei.iter().map(|n| n.z).collect();
            let g = hs2.lock().unwrap().extrapolation_grade(&pos, &z).unwrap_or(f64::NAN);
            worst_e = worst_e.max(de);
            worst_f = worst_f.max(fr);
            judged += 1;
            let _ = writeln!(out, "      check {i:5}: |ΔE|/N {de:.2e} Ha, force residual {fr:.2}× bound, γ {g:.2}, face {face:?}, T {:.0} K, d_min {:.3}", snap.temperature, pair_min(&snap.scene));
        }
        if i % fd_every == 0 {
            let w = fd_check(&hs2.lock().unwrap(), &snap.scene);
            worst_fd = worst_fd.max(w);
        }
    });
    let _ = writeln!(out, "  draft checks: {judged} judged, worst |ΔE|/N {worst_e:.2e} Ha, worst force residual {worst_f:.2}× bound; model E/F FD worst {worst_fd:.2e}");
    {
        let mut s = hs.lock().unwrap();
        // the draft audits' samples go into the fit (what the worker's refit does)
        let held = s.samples();
        let _ = s.refit();
        s.set_draft(false);
        s.set_audit_every(FallbackPolicy::default().audit_every);
        let _ = writeln!(out, "  after the draft run (samples {held}, refit):\n{}", s.inspect(&[("the 300 K state", &warm)]));
    }
    let mut d = Dynamics::new(warm.clone(), Box::new(SharedTrainable::from_handle(hs.clone())), DT_FS, SEED);
    run_phase(&mut d, &hs, "strict NVE after draft", env_usize("E1_RERUN_STEPS", 400), env_usize("E1_RERUN_CAP", 100), 50, &mut out, &mut |_, _, _| {});
    println!("{out}");
    out.clear();

    // (e) the log's pathologies taught as kept samples
    {
        let mut s = hs.lock().unwrap();
        let close = scene_of(vec![Nucleus::new(8, [0.0; 3]), Nucleus::new(1, [0.9, 0.0, 0.0]), Nucleus::new(1, [-0.45, 1.75, 0.0])]);
        let far = scene_of(vec![Nucleus::new(8, [0.0; 3]), Nucleus::new(1, [4.63, 0.0, 0.0]), Nucleus::new(1, [-0.47, 1.75, 0.0])]);
        for (what, sc) in [("H at 0.9 Bohr from O", &close), ("O–H 4.63 Bohr", &far)] {
            let n0 = s.samples();
            s.discontinuity();
            s.force_teacher_next();
            match s.evaluate(sc, &mut never()) {
                Ok(r) => { let _ = writeln!(out, "(e) {what}: teacher E {:.6}, face {:?}, kept {}; notes {:?}", r.energy, r.stats.face_density, s.samples() > n0, r.stats.notes); }
                Err(e) => { let _ = writeln!(out, "(e) {what}: teacher failed: {e}"); }
            }
        }
        let _ = s.refit();
        let _ = writeln!(out, "  after (e):\n{}", s.inspect(&[("the 300 K state", &warm)]));
    }
    let mut d = Dynamics::new(warm.clone(), Box::new(SharedTrainable::from_handle(hs.clone())), DT_FS, SEED);
    run_phase(&mut d, &hs, "strict NVE after (e)", env_usize("E1_RERUN_STEPS", 400), env_usize("E1_RERUN_CAP", 100), 50, &mut out, &mut |_, _, _| {});
    let _ = writeln!(out, "E1 wall {:.0} s", t_test.elapsed().as_secs_f64());
    println!("{out}");
}

/// E2 (module doc).
#[test]
#[ignore = "experiment E2 (≈ 40 min GPU): cargo test -p validate --release --test surrogate_water water_plus_two_h -- --ignored --nocapture"]
fn water_plus_two_h_box_rule() {
    let Some(ctx) = gpu() else { return };
    let t_test = Instant::now();
    let path = model_path();
    if !path.exists() {
        let mut out = String::new();
        let mut s = student(&ctx, false);
        run_pass(&mut s, &water(), &SamplingPlan { seed: SEED, ..SamplingPlan::default() }, &mut out);
        s.save_sidecar(&path).unwrap_or_else(|e| panic!("save {}: {e}", path.display()));
        println!("E2: no E1 model — trained water afresh:\n{out}");
    }
    let plan = SamplingPlan { perturbations: 34, scan_points_per_pair: 8, hot_burst_steps: 100, seed: SEED, ..SamplingPlan::default() };
    for (label, dy) in [("A (centred)", 0.0), ("B (1.5 Bohr toward the −y face)", -1.5)] {
        let mut out = String::new();
        let centre = water_plus_two_h(dy);
        let gap = centre.cell.max_coord() - centre.nuclei.iter().flat_map(|n| n.pos).fold(0.0f64, |m, x| m.max(x.abs()));
        let _ = writeln!(out, "E2 variant {label}: water + 2 H, nearest face {gap:.2} Bohr (the default walls' half-width {:.2} Bohr allow 1.5 Bohr)", centre.cell.max_coord() - 1.5);
        let mut s = student(&ctx, false);
        s.load(&path, &centre).unwrap_or_else(|e| panic!("load {}: {e}", path.display()));
        for n in s.take_notes() {
            let _ = writeln!(out, "  note: {n}");
        }
        let _ = writeln!(out, "  before the pass, at the centre:\n{}", s.explain(&centre));
        run_pass(&mut s, &centre, &plan, &mut out);
        let _ = writeln!(out, "  after the pass:\n{}", s.inspect(&[("the pass centre", &centre), ("the preset water", &water())]));
        println!("{out}");
        out.clear();
        let p = SharedTrainable::new(s);
        let hs = p.handle();
        let mut d = Dynamics::new(centre.clone(), Box::new(p), DT_FS, SEED);
        run_phase(&mut d, &hs, "strict", 200, env_usize("E2_STRICT_CAP", 60), 20, &mut out, &mut |_, _, _| {});
        hs.lock().unwrap().set_draft(true);
        let mut d = Dynamics::new(centre.clone(), Box::new(SharedTrainable::from_handle(hs.clone())), DT_FS, SEED);
        let mut checker = Locked::new(ctx.clone(), TOL_H2).0;
        run_phase(&mut d, &hs, "draft", 200, usize::MAX, 20, &mut out, &mut |i, snap, out| {
            if [0usize, 5, 10, 20, 50, 100, 199].contains(&i) {
                let (de, fr, face) = judge(&mut checker, snap);
                let _ = writeln!(out, "      check {i:3}: {:?}, |ΔE|/N {de:.2e} Ha, force residual {fr:.2}× bound, face {face:?}, T {:.0} K, d_min {:.3}", snap.source, snap.temperature, pair_min(&snap.scene));
            }
        });
        let _ = writeln!(out, "E2 variant {label}: wall so far {:.0} s", t_test.elapsed().as_secs_f64());
        println!("{out}");
    }
}


/// E1(e) again on the fixed code (the 2026-10-18 trust fixes): E1's pass model (from
/// `water_e1.model`), the H-at-0.9-Bohr frame taught as a kept sample, a refit, the inspection and
/// the readiness at the 300 K geometry, then 100 steps of 300 K CSVR and 200 of NVE in strict mode
/// (teacher cap 100). Before the fixes the degree-cap-6 retry was kept (Q′ 176 → 344 > n) and the
/// many-body class was then untrusted by that one sample: every step DFT under a "trusted" badge.
#[test]
#[ignore = "experiment E1(e) after the fixes (GPU): cargo test -p validate --release --test surrogate_water water_pathology_after -- --ignored --nocapture"]
fn water_pathology_after_the_fixes() {
    let Some(ctx) = gpu() else { return };
    let path = model_path();
    let start = water();
    let mut out = String::new();
    let mut s = student(&ctx, false);
    s.load(&path, &start).unwrap_or_else(|e| panic!("load {} (run E1 first): {e}", path.display()));
    let _ = s.take_notes();
    let close = scene_of(vec![Nucleus::new(8, [0.0; 3]), Nucleus::new(1, [0.9, 0.0, 0.0]), Nucleus::new(1, [-0.45, 1.75, 0.0])]);
    s.discontinuity();
    s.force_teacher_next();
    let r = s.evaluate(&close, &mut never()).unwrap_or_else(|e| panic!("the 0.9 Bohr frame: {e}"));
    let _ = writeln!(out, "E1(e) after the fixes: H at 0.9 Bohr from O, teacher E {:.6}, kept as sample #{}; notes {:?}", r.energy, s.samples() - 1, r.stats.notes);
    let _ = s.refit();
    for n in s.take_notes() {
        let _ = writeln!(out, "  note: {n}");
    }
    let _ = writeln!(out, "{}", s.inspect(&[("the preset water", &start)]));
    let _ = writeln!(out, "  readiness at the preset water: {:?}", s.readiness(&start));
    let p = SharedTrainable::new(s);
    let hs = p.handle();
    let mut d = Dynamics::new(start.clone(), Box::new(p), DT_FS, SEED);
    d.apply(DynCommand::SetThermostat(Some((300.0, 20.0))));
    run_phase(&mut d, &hs, "strict CSVR after the fixes", 100, 100, 25, &mut out, &mut |_, _, _| {});
    d.apply(DynCommand::SetThermostat(None));
    run_phase(&mut d, &hs, "strict NVE after the fixes", 200, 100, 25, &mut out, &mut |_, _, _| {});
    let snap = d.snapshot();
    let _ = writeln!(out, "  readiness at the end: {:?}", hs.lock().unwrap().readiness(&snap.scene));
    println!("{out}");
}
