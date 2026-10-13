//! Driver tests on the analytic providers (no grid, no GPU): time control, rewind, provider
//! failure, drag spring, pins, COM correction, energy health, thermostat commands, edits.
use super::*;
use crate::provider::{EvalStats, ForceResult};
use crate::scene::{Cell, Nucleus};
use crate::test_providers::{FailKind, Failing, Harmonic, Zero};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const DT_FS: f64 = 0.25;

fn scene(nuclei: Vec<Nucleus>) -> Scene {
    Scene { nuclei, cell: Cell { n: 33, h: 0.25 }, charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3] }
}

fn never() -> impl FnMut() -> bool {
    || false
}

fn dt_au() -> f64 {
    units::fs_to_au_time(DT_FS)
}

/// `Step(n)` and one unbounded tick: exactly n steps (or fewer on a failure).
fn step_n(d: &mut Dynamics, n: usize) -> DynSnapshot {
    d.apply(DynCommand::Step(n));
    d.tick(f64::INFINITY, &mut never())
}

/// Plays n ≤ 1000 steps in one unbounded tick, then pauses.
fn play_n(d: &mut Dynamics, n: u32) -> DynSnapshot {
    d.apply(DynCommand::SetSpeed(n));
    d.apply(DynCommand::Play);
    let s = d.tick(f64::INFINITY, &mut never());
    d.apply(DynCommand::Pause);
    s
}

/// Three nuclei displaced from their wells and moving: a generic NVE trajectory.
fn harmonic_three() -> (Scene, Harmonic) {
    let s = scene(vec![
        Nucleus::new(1, [0.05, -0.02, 0.01]).with_vel([1e-4, 0.0, -5e-5]),
        Nucleus::new(8, [1.5, 0.3, -0.2]).with_vel([0.0, 2e-5, 1e-5]),
        Nucleus::new(1, [-1.4, 1.0, 0.04]).with_vel([-3e-5, 1e-4, 0.0]),
    ]);
    let h = Harmonic { k: 0.4, centres: vec![[0.0; 3], [1.45, 0.35, -0.2], [-1.43, 1.11, 0.0]] };
    (s, h)
}

/// Uniform force F on every nucleus, from V = −F·Σ r_i (conservative, ΣF = N F ≠ 0).
struct Uniform {
    f: [f64; 3],
}

impl ForceProvider for Uniform {
    fn evaluate(&mut self, scene: &Scene, _a: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        let energy = -scene.nuclei.iter().map(|n| (0..3).map(|a| self.f[a] * n.pos[a]).sum::<f64>()).sum::<f64>();
        Ok(ForceResult { energy, forces: vec![self.f; scene.nuclei.len()], stats: EvalStats::default(), display: None })
    }
    fn name(&self) -> &'static str {
        "uniform"
    }
    fn reset(&mut self) {}
}

/// Zero forces, energy rising by `per_call` Ha per evaluation: a pure energy ramp.
struct Ramp {
    per_call: f64,
    calls: usize,
}

impl ForceProvider for Ramp {
    fn evaluate(&mut self, scene: &Scene, _a: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        self.calls += 1;
        Ok(ForceResult { energy: self.per_call * self.calls as f64, forces: vec![[0.0; 3]; scene.nuclei.len()], stats: EvalStats::default(), display: None })
    }
    fn name(&self) -> &'static str {
        "ramp"
    }
    fn reset(&mut self) {}
}

/// Counts evaluations and resets of the wrapped provider (shared counters: the driver owns it).
struct Counting {
    inner: Box<dyn ForceProvider>,
    evals: Arc<AtomicUsize>,
    resets: Arc<AtomicUsize>,
}

impl ForceProvider for Counting {
    fn evaluate(&mut self, scene: &Scene, a: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        self.evals.fetch_add(1, Ordering::SeqCst);
        self.inner.evaluate(scene, a)
    }
    fn name(&self) -> &'static str {
        "counting"
    }
    fn reset(&mut self) {
        self.resets.fetch_add(1, Ordering::SeqCst);
        self.inner.reset();
    }
}

#[test]
fn pause_does_not_advance_time_and_step_advances_exactly_dt() {
    let (s, h) = harmonic_three();
    let mut d = Dynamics::new(s.clone(), Box::new(h), DT_FS, 1);
    let s0 = d.snapshot();
    assert!(s0.paused, "starts paused");
    assert_eq!((s0.sim_time, s0.step), (0.0, 0));
    assert!(!s0.forces_stale && s0.forces.len() == 3, "new() evaluates once");
    // paused ticks change nothing
    for _ in 0..3 {
        let t = d.tick(f64::INFINITY, &mut never());
        assert_eq!((t.sim_time, t.step, t.steps_last_tick, t.wall_ratio), (0.0, 0, 0, 0.0));
        assert!(t.scene == s);
    }
    // Step(3): exactly three steps of dt, still paused
    let t = step_n(&mut d, 3);
    assert_eq!(t.step, 3);
    assert_eq!(t.steps_last_tick, 3);
    assert!((t.sim_time - 3.0 * dt_au()).abs() <= 1e-12 * t.sim_time, "{} vs {}", t.sim_time, 3.0 * dt_au());
    assert!(t.paused);
    assert_eq!(t.pending_steps, 0);
    assert!(t.scene != s);
    // and another paused tick does nothing
    let t2 = d.tick(f64::INFINITY, &mut never());
    assert_eq!((t2.step, t2.sim_time), (3, t.sim_time));
    // playing: `speed` steps per unbounded tick, a measured wall ratio
    d.apply(DynCommand::SetSpeed(7));
    d.apply(DynCommand::Play);
    let p = d.tick(f64::INFINITY, &mut never());
    assert!(!p.paused);
    assert_eq!(p.step, 10);
    assert_eq!(p.steps_last_tick, 7);
    assert!(p.wall_ratio > 0.0, "fs per wall second measured: {}", p.wall_ratio);
    // a zero budget still runs one step when playing
    let z = d.tick(0.0, &mut never());
    assert_eq!(z.steps_last_tick, 1);
    // should_abort preempts before the first step
    let a = d.tick(f64::INFINITY, &mut || true);
    assert_eq!(a.steps_last_tick, 0);
    assert!(!a.paused, "an abort is not a pause");
    d.apply(DynCommand::Pause);
    let q = d.tick(f64::INFINITY, &mut never());
    assert_eq!((q.step, q.steps_last_tick, q.wall_ratio), (11, 0, 0.0));
}

#[test]
fn rewind_retraces_the_trajectory() {
    let (s, h) = harmonic_three();
    let mut d = Dynamics::new(s.clone(), Box::new(h), DT_FS, 1);
    let mid = play_n(&mut d, 200);
    let moved = mid.scene.nuclei.iter().zip(&s.nuclei).map(|(a, b)| dist(a.pos, b.pos)).fold(0.0, f64::max);
    assert!(moved > 1e-2, "the trajectory went somewhere ({moved} Bohr)");
    d.apply(DynCommand::Rewind);
    let r = d.snapshot();
    assert!(r.notes.is_empty() || !r.notes.iter().any(|n| n.contains("not reversible")), "NVE rewind gives no warning: {:?}", r.notes);
    let back = play_n(&mut d, 200);
    let mut worst = 0.0f64;
    for (a, b) in back.scene.nuclei.iter().zip(&s.nuclei) {
        worst = worst.max(dist(a.pos, b.pos));
        for k in 0..3 {
            assert!((a.vel[k] + b.vel[k]).abs() < 1e-12, "velocity not reversed: {:?} vs {:?}", a.vel, b.vel);
        }
    }
    println!("rewind: max |r − r₀| after 200 + 200 steps = {worst:.3e} Bohr");
    assert!(worst < 1e-8, "{worst}");
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

#[test]
fn provider_error_pauses_and_keeps_last_good_state() {
    for kind in [FailKind::Invalid, FailKind::Numerical] {
        let (s, h) = harmonic_three();
        // call 1 is new()'s evaluation, calls 2–5 steps 1–4: step 5 (call 6) fails
        let mut d = Dynamics::new(s.clone(), Box::new(Failing::new(h.clone(), 6).with_kind(kind)), DT_FS, 1);
        d.apply(DynCommand::SetSpeed(10));
        d.apply(DynCommand::Play);
        let t = d.tick(f64::INFINITY, &mut never());
        let mut reference = Dynamics::new(s.clone(), Box::new(h), DT_FS, 1);
        let r = step_n(&mut reference, 4);
        println!("{kind:?}: last_error = {:?}", t.last_error);
        assert!(t.paused, "an error pauses");
        assert!(t.last_error.as_deref().is_some_and(|e| e.contains("injected failure")), "{:?}", t.last_error);
        assert_eq!(t.steps_last_tick, 4);
        assert_eq!(t.step, 4);
        assert!(t.scene == r.scene, "scene must be the state after step 4");
        assert_eq!(t.forces, r.forces, "forces must be the forces after step 4");
        assert_eq!(t.energy_ks, r.energy_ks);
        assert_eq!(t.sim_time, r.sim_time);
        // Play clears the error and resumes from the kept state
        d.apply(DynCommand::Play);
        assert!(d.snapshot().last_error.is_none());
        let t2 = d.tick(f64::INFINITY, &mut never());
        let r2 = step_n(&mut reference, 10);
        assert!(t2.scene == r2.scene && t2.step == 14);
    }
}

#[test]
fn cancelled_evaluation_ends_the_tick_without_error() {
    let (s, h) = harmonic_three();
    // call 4 = step 3 is cancelled
    let mut d = Dynamics::new(s.clone(), Box::new(Failing::new(h.clone(), 4).with_kind(FailKind::Cancelled)), DT_FS, 1);
    d.apply(DynCommand::SetSpeed(10));
    d.apply(DynCommand::Play);
    let t = d.tick(f64::INFINITY, &mut never());
    assert_eq!(t.steps_last_tick, 2, "the tick ends at the cancelled step");
    assert!(!t.paused, "a cancellation does not pause");
    assert!(t.last_error.is_none(), "a cancellation is not an error: {:?}", t.last_error);
    let mut reference = Dynamics::new(s.clone(), Box::new(h), DT_FS, 1);
    let r = step_n(&mut reference, 2);
    assert!(t.scene == r.scene && t.forces == r.forces);
    // the next tick carries on as if nothing happened
    let t2 = d.tick(f64::INFINITY, &mut never());
    let r2 = step_n(&mut reference, 10);
    assert_eq!(t2.step, 12);
    assert!(t2.scene == r2.scene);
    // a cancelled Step keeps its pending steps for the next tick
    let mut e = Dynamics::new(s.clone(), Box::new(Failing::new(Zero, 3).with_kind(FailKind::Cancelled)), DT_FS, 1);
    e.apply(DynCommand::Step(4));
    let c = e.tick(f64::INFINITY, &mut never());
    assert_eq!((c.steps_last_tick, c.pending_steps), (1, 3));
    let c2 = e.tick(f64::INFINITY, &mut never());
    assert_eq!((c2.steps_last_tick, c2.pending_steps, c2.step), (3, 0, 4));
}

/// The spring k_drag (target − r) on a free ¹H from rest: Verlet's positions are exactly
/// x_n = x_t + (x₀ − x_t) cos(n ω̃ dt) with cos(ω̃ dt) = 1 − (ω dt)²/2, ω² = k_drag/m (the
/// linear recurrence x_{n+1} − 2x_n + x_{n−1} = −(ω dt)²(x_n − x_t) with the first step
/// x₁ − x_t = (x₀ − x_t)(1 − (ω dt)²/2)). An undamped spring oscillates about the target
/// (period ≈ 37 steps here); the pull is checked against that closed form rather than a
/// phase-dependent "within 0.5 Bohr after 500 steps".
#[test]
fn drag_spring_pulls_a_free_nucleus_to_the_target_and_release_stops_the_pull() {
    let s = scene(vec![Nucleus::new(1, [0.0; 3])]);
    let m = s.nuclei[0].mass;
    let mut d = Dynamics::new(s, Box::new(Zero), DT_FS, 1);
    let target = [2.0, 0.0, 0.0];
    d.apply(DynCommand::DragStart { index: 0, target });
    let snap = d.snapshot();
    let e0 = snap.energy_conserved;
    assert_eq!(snap.drag, Some((0, target)));
    let w = (K_DRAG / m).sqrt() * dt_au();
    let wt = (1.0 - 0.5 * w * w).acos();
    let (mut emin, mut emax, mut cmax, mut xmax) = (f64::MAX, f64::MIN, 0.0f64, 0.0f64);
    for n in 1..=500 {
        let t = step_n(&mut d, 1);
        let x = t.scene.nuclei[0].pos;
        let exact = 2.0 - 2.0 * (n as f64 * wt).cos();
        assert!((x[0] - exact).abs() < 1e-9, "step {n}: x = {} vs {exact}", x[0]);
        assert_eq!((x[1], x[2]), (0.0, 0.0));
        emin = emin.min(t.total);
        emax = emax.max(t.total);
        cmax = cmax.max((t.energy_conserved - e0).abs());
        xmax = xmax.max(x[0]);
    }
    // it reaches the target and beyond (an undamped spring), the mean is the target
    assert!(xmax > 3.9 && xmax <= 4.0 + 1e-12, "{xmax}");
    // E_total = K swings by the spring energy ½ k 2² = 1 Ha; E_conserved = K + ½k|r − t|² holds
    // to Verlet's one-sided h²/4 of the mode energy (R6: E ∈ [E₀(1 − h²/4), E₀])
    let mode = 0.5 * K_DRAG * 4.0;
    println!("drag: E_total ∈ [{emin:.4}, {emax:.4}] Ha; max |ΔE_conserved| = {cmax:.3e} Ha (h²/4·E = {:.3e})", w * w / 4.0 * mode);
    assert!(emax - emin > 0.9 * mode);
    assert!(cmax <= 1.01 * w * w / 4.0 * mode, "{cmax}");
    let before = d.snapshot();
    assert!(before.drift_per_ps.is_some());
    // release: the spring's energy leaves the books with it, the velocity persists
    d.apply(DynCommand::DragEnd);
    let rel = d.snapshot();
    assert_eq!(rel.drag, None);
    assert!((rel.energy_conserved - before.energy_conserved).abs() < 1e-12, "release is continuous in E_conserved");
    let v = rel.scene.nuclei[0].vel;
    let x = rel.scene.nuclei[0].pos;
    let after = step_n(&mut d, 50);
    assert_eq!(after.scene.nuclei[0].vel, v, "no spring after DragEnd");
    let expect = x[0] + 50.0 * dt_au() * v[0];
    assert!((after.scene.nuclei[0].pos[0] - expect).abs() < 1e-12 * expect.abs().max(1.0));
    assert!((after.energy_conserved - rel.energy_conserved).abs() < 1e-12);
}

#[test]
fn pins_zero_velocity_each_step() {
    let (mut s, h) = harmonic_three();
    s.external_field = [0.0, 0.0, -1e-7]; // a pin holds against the field too
    let mut d = Dynamics::new(s.clone(), Box::new(h), DT_FS, 1);
    let free = d.snapshot();
    d.apply(DynCommand::Pin(0, true));
    let p = d.snapshot();
    // the kinetic energy the pin removes is booked: E_total drops, E_conserved does not
    let k0 = 0.5 * s.nuclei[0].mass * s.nuclei[0].vel.iter().map(|v| v * v).sum::<f64>();
    assert!((free.total - p.total - k0).abs() < 1e-15, "E_total drops by the pinned nucleus's K");
    assert!((p.energy_conserved - free.energy_conserved).abs() < 1e-15);
    assert_eq!(p.pins, vec![true, false, false]);
    assert_eq!(p.scene.nuclei[0].vel, [0.0; 3], "pinning zeroes the velocity at once");
    let e0 = p.energy_conserved;
    let mut cmax = 0.0f64;
    for _ in 0..30 {
        let t = play_n(&mut d, 10);
        assert_eq!(t.scene.nuclei[0].pos, s.nuclei[0].pos, "a pinned nucleus never moves (its well pulls it: r ≠ centre)");
        assert_eq!(t.scene.nuclei[0].vel, [0.0; 3]);
        cmax = cmax.max((t.energy_conserved - e0).abs());
    }
    let t = d.snapshot();
    assert!(dist(t.scene.nuclei[1].pos, s.nuclei[1].pos) > 1e-3, "the others move");
    println!("pins: max |ΔE_conserved| = {cmax:.3e} Ha over 300 steps");
    // unpinned: its well pulls it back
    d.apply(DynCommand::Pin(0, false));
    let u = play_n(&mut d, 50);
    assert!(dist(u.scene.nuclei[0].pos, s.nuclei[0].pos) > 1e-3, "an unpinned nucleus moves");
    // pin index out of range: rejected, no panic
    d.apply(DynCommand::Pin(7, true));
    assert!(d.snapshot().last_error.is_some());
}

#[test]
fn com_correction_removes_net_force() {
    let f = [1e-3, -2e-4, 0.0];
    let nuclei = vec![Nucleus::new(1, [0.0; 3]), Nucleus::new(8, [1.5, 0.0, 0.0]), Nucleus::new(1, [-1.5, 0.5, 0.0])];
    // off: Σ m v grows as N F t
    let mut off = Dynamics::new(scene(nuclei.clone()), Box::new(Uniform { f }), DT_FS, 1);
    let o = play_n(&mut off, 100);
    let mass = o.scene.total_mass();
    let u = o.scene.com_velocity();
    let t = 100.0 * dt_au();
    for a in 0..3 {
        let exact = 3.0 * f[a] * t / mass;
        assert!((u[a] - exact).abs() <= 1e-12 * exact.abs().max(1e-12), "COM velocity {u:?} vs N F t / M");
    }
    assert!(u[0] > 0.0);
    // on: the mean force is removed — no COM acceleration at all
    let mut on = Dynamics::new(scene(nuclei.clone()), Box::new(Uniform { f }), DT_FS, 1);
    on.apply(DynCommand::SetComCorrection(true));
    let c = play_n(&mut on, 100);
    assert!(c.com_correction);
    let uc = c.scene.com_velocity();
    println!("COM velocity after 100 steps: off {u:?}, on {uc:?}");
    for a in 0..3 {
        assert!(uc[a].abs() < 1e-15, "{uc:?}");
    }

    // with the correction on a harmonic system: E_total is not conserved (the correction is a
    // non-Hamiltonian force on V), E_conserved (its trapezoidal work booked) is — exactly here,
    // the correction being linear in r
    let (s, h) = harmonic_three();
    let mut d = Dynamics::new(s, Box::new(h), DT_FS, 1);
    d.apply(DynCommand::SetComCorrection(true));
    let e0 = d.snapshot();
    let (mut tmax, mut cmax) = (0.0f64, 0.0f64);
    for _ in 0..20 {
        let t = play_n(&mut d, 50);
        tmax = tmax.max((t.total - e0.total).abs());
        cmax = cmax.max((t.energy_conserved - e0.energy_conserved).abs());
    }
    println!("COM correction on Harmonic: max |ΔE_total| = {tmax:.3e}, max |ΔE_conserved| = {cmax:.3e} Ha");
    assert!(tmax > 100.0 * cmax, "the correction's work must show in E_total but not in E_conserved");
}

/// NVE health: the drift fit sees no slope on a conserved oscillator and the exact slope of an
/// energy ramp. The oscillator's Verlet energy wobble is A ≈ (ω dt)²/4 · E_mode at 2ω; the
/// least-squares slope of a bounded oscillation over a window T is ≤ 12 A / (2ω T²) — with
/// 0.05 Bohr displacements (E_mode ≈ 1.5e-3 Ha, ω dt = 0.15, A ≈ 8e-6 Ha) and T = 2000 steps
/// that is ≲ 1.5e-7 Ha/ps, so 1e-6 has ≥ 6× margin.
#[test]
fn drift_estimate_is_zero_for_a_conserved_oscillator_and_detects_a_ramp() {
    let s = scene(vec![
        Nucleus::new(1, [0.05, 0.0, 0.0]),
        Nucleus::new(1, [2.0, 0.05, 0.0]),
        Nucleus::new(1, [-2.0, 0.0, -0.05]),
    ]);
    let h = Harmonic { k: 0.4, centres: vec![[0.0; 3], [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]] };
    let mut d = Dynamics::new(s, Box::new(h), DT_FS, 1);
    assert_eq!(d.snapshot().drift_per_ps, None);
    let early = play_n(&mut d, 99);
    assert_eq!(early.drift_per_ps, None, "None until ≥ 100 history points");
    assert_eq!(play_n(&mut d, 1).history.len(), 100);
    assert!(d.snapshot().drift_per_ps.is_some());
    for _ in 0..2 {
        play_n(&mut d, 1000);
    }
    let t = d.snapshot();
    assert_eq!(t.history.len(), HISTORY_LEN);
    let drift = t.drift_per_ps.unwrap();
    println!("harmonic: drift = {drift:.3e} Ha/ps over {} steps", t.history.len());
    assert!(drift.abs() < 1e-6, "{drift}");

    // a ramp of 1e-3 Ha/ps (per evaluation: 1e-3 Ha/ps × dt in ps)
    let per_call = 1e-3 * DT_FS * 1e-3;
    let mut r = Dynamics::new(scene(vec![Nucleus::new(1, [0.0; 3])]), Box::new(Ramp { per_call, calls: 0 }), DT_FS, 1);
    let rs = play_n(&mut r, 300);
    let slope = rs.drift_per_ps.unwrap();
    println!("ramp: drift = {slope:.6e} Ha/ps (expected 1e-3)");
    assert!((slope - 1e-3).abs() < 0.1e-3, "{slope}");
}

/// A uniform field is a conservative force with U = −Σ m a·r: E_total (= K + E_KS) grows as the
/// system falls, E_conserved does not; changing the field is booked so E_conserved is continuous.
#[test]
fn energy_conserved_books_the_field_potential() {
    let mut s = scene(vec![Nucleus::new(1, [0.0; 3]), Nucleus::new(8, [1.0, 0.0, 0.0])]);
    s.external_field = [0.0, 0.0, -1e-6];
    let mut d = Dynamics::new(s, Box::new(Zero), DT_FS, 1);
    let e0 = d.snapshot();
    let t = play_n(&mut d, 500);
    println!("field: ΔE_total = {:.3e}, ΔE_conserved = {:.3e} Ha", t.total - e0.total, t.energy_conserved - e0.energy_conserved);
    assert!(t.total - e0.total > 1e-3, "it falls: K grows");
    assert!((t.energy_conserved - e0.energy_conserved).abs() < 1e-12 * (t.total - e0.total), "Verlet is exact under a uniform field");
    assert!(t.drift_per_ps.unwrap().abs() < 1e-9);
    d.apply(DynCommand::SetExternalField([1e-6, 0.0, 0.0]));
    let f = d.snapshot();
    assert_eq!(f.scene.external_field, [1e-6, 0.0, 0.0]);
    assert!((f.energy_conserved - t.energy_conserved).abs() < 1e-12);
    let g = play_n(&mut d, 200);
    assert!((g.energy_conserved - t.energy_conserved).abs() < 1e-9);
    d.apply(DynCommand::SetExternalField([f64::NAN, 0.0, 0.0]));
    assert!(d.snapshot().last_error.is_some());
    assert_eq!(d.snapshot().scene.external_field, [1e-6, 0.0, 0.0]);
}

/// Sixteen independent wells (N_f = 45: Var T / T² = 2/45, std 21 %).
fn sixteen(displaced: bool) -> (Scene, Harmonic) {
    let mut nuclei = Vec::new();
    let mut centres = Vec::new();
    for i in 0..16 {
        let c = [(i % 4) as f64 * 2.0, (i / 4) as f64 * 2.0, 0.0];
        let off = if displaced { [0.03 * ((i * 7 % 5) as f64 - 2.0), 0.02 * ((i * 3 % 4) as f64 - 1.5), 0.01] } else { [0.0; 3] };
        nuclei.push(Nucleus::new(1, [c[0] + off[0], c[1] + off[1], c[2] + off[2]]));
        centres.push(c);
    }
    (scene(nuclei), Harmonic { k: 0.4, centres })
}

#[test]
fn thermostat_commands_toggle_and_rewind_warns() {
    // at rest: switching the thermostat on draws Maxwell–Boltzmann velocities at exactly T
    let (s, h) = sixteen(false);
    let mut d = Dynamics::new(s, Box::new(h), DT_FS, 42);
    d.apply(DynCommand::SetThermostat(Some((300.0, 20.0))));
    let on = d.snapshot();
    println!("thermostat on at rest: T = {:.3} K, notes {:?}", on.temperature, on.notes);
    assert_eq!(on.thermostat, Some((300.0, 20.0)));
    assert!((on.temperature - 300.0).abs() < 1e-9 * 300.0);
    assert!(on.notes.iter().any(|n| n.contains("Maxwell–Boltzmann")), "{:?}", on.notes);
    assert!((on.energy_conserved - d.snapshot().energy_ks).abs() < 1e-12, "the draw is booked: E_conserved unchanged");

    // invalid parameters: rejected, no panic, the thermostat unchanged
    for bad in [(-1.0, 20.0), (300.0, 0.0), (300.0, -5.0), (f64::NAN, 20.0), (300.0, f64::INFINITY)] {
        d.apply(DynCommand::SetThermostat(Some(bad)));
        let b = d.snapshot();
        assert!(b.last_error.is_some(), "{bad:?} accepted");
        assert_eq!(b.thermostat, Some((300.0, 20.0)));
    }

    // rewinding under NVT flips the velocities but warns that it does not retrace
    d.apply(DynCommand::Rewind);
    let r = d.snapshot();
    assert!(r.notes.iter().any(|n| n.contains("not reversible")), "{:?}", r.notes);

    // off → NVE again
    d.apply(DynCommand::SetThermostat(None));
    assert_eq!(d.snapshot().thermostat, None);

    // not at rest (K = 10 % of K̄ ≥ 1 %): no draw; T relaxes from ~30 K to the target over τ
    let (s, h) = sixteen(true);
    let mut d = Dynamics::new(s, Box::new(h), DT_FS, 7);
    // scale the potential energy into kinetic: start from rest at displaced positions, step
    // until K is ~ 30 K worth, then switch on
    let mut t0 = d.snapshot();
    while t0.temperature < 30.0 {
        t0 = step_n(&mut d, 1);
    }
    d.apply(DynCommand::SetThermostat(Some((300.0, 20.0))));
    let s1 = d.snapshot();
    assert!(!s1.notes.iter().any(|n| n.contains("Maxwell–Boltzmann")), "no draw when moving: {:?}", s1.notes);
    assert!(s1.temperature < 100.0);
    let mut temps = Vec::new();
    let mut conserved = Vec::new();
    for _ in 0..16 {
        let t = play_n(&mut d, 500);
        temps.extend(t.history[t.history.len() - 500..].iter().map(|p| p.2));
        conserved.push(t.energy_conserved);
    }
    let early: f64 = temps[..40].iter().sum::<f64>() / 40.0;
    let late: f64 = temps[2000..].iter().sum::<f64>() / (temps.len() - 2000) as f64;
    let fin = d.snapshot();
    println!("thermostat: ⟨T⟩ first 40 steps {early:.1} K, steps 2000–8000 {late:.1} K; E_conserved {:?}; drift {:?} Ha/ps; E_total moved {:.3e}",
             conserved, fin.drift_per_ps, fin.total - s1.total);
    assert!(early < 150.0, "{early}");
    // T fluctuates by √(2/N_f) = 21 % per sample; with τ = 20 fs (80 steps) and the 2ω
    // oscillation averaging out, 6000 steps hold ≳ 70 independent samples → std err ≲ 2.5 %
    // (7.5 K): the 30 K bound is 4σ
    assert!((late - 300.0).abs() < 30.0, "{late}");
    // the bath's work is booked: E_total takes up the heat (N_f/2 k_B ΔT kinetic plus as much
    // potential), E_conserved only Verlet's bounded wobble, ≤ (ω dt)²/4 of the mode energy
    // (= E_total here: the wells' minimum is 0) — 3× that is the bound
    let heat = 0.5 * 45.0 * units::kelvin_to_hartree(270.0);
    assert!(fin.total - s1.total > 0.3 * heat);
    let spread = conserved.iter().fold(0.0f64, |m, c| m.max((c - s1.energy_conserved).abs()));
    let h = (0.4f64 / fin.scene.nuclei[0].mass).sqrt() * dt_au();
    let wobble = h * h / 4.0 * fin.total;
    println!("thermostat: max |ΔE_conserved| = {spread:.3e} Ha (Verlet wobble {wobble:.3e}) against {heat:.3e} Ha of heat");
    assert!(spread < 3.0 * wobble, "E_conserved moved {spread:.3e} Ha vs a wobble of {wobble:.3e} Ha ({heat:.3e} Ha of heat)");
}

#[test]
fn discontinuous_edits_only_while_paused_and_reset_the_provider() {
    let evals = Arc::new(AtomicUsize::new(0));
    let resets = Arc::new(AtomicUsize::new(0));
    let provider = Counting { inner: Box::new(Zero), evals: evals.clone(), resets: resets.clone() };
    let s = scene(vec![
        Nucleus::new(1, [0.0; 3]).with_vel([1e-4, 0.0, 0.0]),
        Nucleus::new(1, [1.4, 0.0, 0.0]).with_vel([0.0, 1e-4, 0.0]),
        Nucleus::new(8, [0.0, 2.0, 0.0]).with_vel([0.0, 0.0, 1e-5]),
    ]);
    let mut d = Dynamics::new(s, Box::new(provider), DT_FS, 1);
    assert_eq!(evals.load(Ordering::SeqCst), 1);
    assert_eq!(resets.load(Ordering::SeqCst), 1, "new() makes its one evaluation cold");
    resets.store(0, Ordering::SeqCst);

    // while playing every discontinuous edit is rejected and changes nothing
    d.apply(DynCommand::Play);
    let before = d.snapshot();
    for cmd in [
        DynCommand::MoveNucleusDiscontinuous { index: 0, pos: [3.0, 0.0, 0.0] },
        DynCommand::AddNucleus(Nucleus::new(1, [5.0, 0.0, 0.0])),
        DynCommand::RemoveNucleus(1),
        DynCommand::ReplaceScene(scene(vec![Nucleus::new(2, [0.0; 3])])),
    ] {
        d.apply(cmd);
        let a = d.snapshot();
        assert!(a.last_error.as_deref().is_some_and(|e| e.contains("pause")), "{:?}", a.last_error);
        assert!(a.scene == before.scene);
    }
    assert_eq!(resets.load(Ordering::SeqCst), 0);
    d.apply(DynCommand::Pause);

    // paused: Move teleports, zeroes that velocity, resets the provider; no evaluation until tick
    d.apply(DynCommand::Pin(2, true));
    d.apply(DynCommand::DragStart { index: 1, target: [1.4, 1.0, 0.0] });
    assert_eq!(resets.load(Ordering::SeqCst), 0, "pins and drags do not reset");
    d.apply(DynCommand::MoveNucleusDiscontinuous { index: 0, pos: [-1.0, 0.0, 0.0] });
    let m = d.snapshot();
    assert!(m.last_error.is_none(), "accepted edits clear the error: {:?}", m.last_error);
    assert_eq!(m.scene.nuclei[0].pos, [-1.0, 0.0, 0.0]);
    assert_eq!(m.scene.nuclei[0].vel, [0.0; 3]);
    assert_eq!(m.scene.nuclei[1].vel, [0.0, 1e-4, 0.0], "other velocities kept");
    assert!(m.forces_stale && m.forces.is_empty());
    assert_eq!(resets.load(Ordering::SeqCst), 1);
    assert_eq!(evals.load(Ordering::SeqCst), 1, "apply never evaluates");
    let e = d.tick(f64::INFINITY, &mut never());
    assert_eq!(evals.load(Ordering::SeqCst), 2, "a paused tick re-evaluates stale forces");
    assert!(!e.forces_stale && e.forces.len() == 3);
    assert_eq!((e.step, e.sim_time), (0, 0.0), "re-evaluating is not a step");
    d.tick(f64::INFINITY, &mut never());
    assert_eq!(evals.load(Ordering::SeqCst), 2, "fresh forces are not re-evaluated");

    // Add: at rest, unpinned; Remove shifts the pins and the drag index
    d.apply(DynCommand::AddNucleus(Nucleus::new(1, [4.0, 0.0, 0.0]).with_vel([1.0, 1.0, 1.0])));
    let a = d.snapshot();
    assert_eq!(a.scene.nuclei.len(), 4);
    assert_eq!(a.scene.nuclei[3].vel, [0.0; 3]);
    assert_eq!(a.pins, vec![false, false, true, false]);
    assert_eq!(resets.load(Ordering::SeqCst), 2);
    d.apply(DynCommand::RemoveNucleus(0));
    let r = d.snapshot();
    assert_eq!(r.scene.nuclei.len(), 3);
    assert_eq!(r.pins, vec![false, true, false]);
    assert_eq!(r.drag, Some((0, [1.4, 1.0, 0.0])), "drag index follows its nucleus");
    d.apply(DynCommand::RemoveNucleus(0));
    assert_eq!(d.snapshot().drag, None, "removing the dragged nucleus ends the drag");
    d.apply(DynCommand::RemoveNucleus(9));
    assert!(d.snapshot().last_error.is_some());
    let t = step_n(&mut d, 2);
    assert_eq!(t.step, 2);
    assert_eq!(t.scene.nuclei[0].pos, r.scene.nuclei[1].pos, "the pin moved with its nucleus");

    // ReplaceScene: a fresh start (time, step, pins, drag, history), velocities as given
    let new = scene(vec![Nucleus::new(2, [0.0; 3]).with_vel([0.0, 0.0, 2e-4])]);
    d.apply(DynCommand::ReplaceScene(new.clone()));
    let n = d.snapshot();
    assert!(n.scene == new);
    assert_eq!((n.step, n.sim_time, n.history.len()), (0, 0.0, 0));
    assert_eq!(n.pins, vec![false]);
    assert!(n.forces_stale);
    assert_eq!(step_n(&mut d, 1).step, 1);
}

#[test]
fn set_dt_validates_and_warns_and_speed_clamps() {
    let (s, h) = harmonic_three();
    let mut d = Dynamics::new(s, Box::new(h), DT_FS, 1);
    assert!(d.snapshot().dt_warning.is_none());
    d.apply(DynCommand::SetDt(0.6));
    let w = d.snapshot();
    assert!(w.dt_warning.as_deref().is_some_and(|m| m.contains("0.5")), "{:?}", w.dt_warning);
    assert!((w.dt - units::fs_to_au_time(0.6)).abs() < 1e-12);
    assert!(step_n(&mut d, 1).dt_warning.is_some(), "the warning persists");
    d.apply(DynCommand::SetDt(0.25));
    assert!(d.snapshot().dt_warning.is_none());
    for bad in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        d.apply(DynCommand::SetDt(bad));
        let b = d.snapshot();
        assert!(b.last_error.is_some(), "dt {bad} accepted");
        assert!((b.dt - dt_au()).abs() < 1e-12);
    }
    d.apply(DynCommand::SetSpeed(0));
    assert_eq!(d.snapshot().speed, 1);
    d.apply(DynCommand::SetSpeed(5000));
    assert_eq!(d.snapshot().speed, 1000);
}

/// Zero forces, but each evaluation busy-waits `ms` (a stand-in for a DFT solve's cost).
struct Slow {
    ms: f64,
}

impl ForceProvider for Slow {
    fn evaluate(&mut self, scene: &Scene, _a: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        let t0 = std::time::Instant::now();
        while t0.elapsed().as_secs_f64() * 1e3 < self.ms {
            std::hint::spin_loop();
        }
        Ok(ForceResult { energy: 0.0, forces: vec![[0.0; 3]; scene.nuclei.len()], stats: EvalStats::default(), display: None })
    }
    fn name(&self) -> &'static str {
        "slow"
    }
    fn reset(&mut self) {}
}

/// R2: a playing tick runs min(speed, the steps that fit the budget at the running cost per
/// step), at least one; the wall ratio is the fs simulated per wall second.
#[test]
fn tick_runs_the_steps_that_fit_the_budget() {
    let s = scene(vec![Nucleus::new(1, [0.0; 3]).with_vel([1e-4, 0.0, 0.0])]);
    let mut d = Dynamics::new(s, Box::new(Slow { ms: 2.0 }), DT_FS, 1);
    d.apply(DynCommand::SetSpeed(1000));
    d.apply(DynCommand::Play);
    let first = d.tick(0.0, &mut never());
    assert_eq!(first.steps_last_tick, 1, "at least one step, even with no budget");
    assert!(first.ms_per_step >= 2.0, "{}", first.ms_per_step);
    let t = d.tick(20.0, &mut never());
    // each step costs ≥ 2 ms and a step starts only if the mean cost still fits: ≤ 20/2 steps
    println!("budget 20 ms at ≥ 2 ms/step: {} steps, {:.2} ms/step, {:.3} fs/s", t.steps_last_tick, t.ms_per_step, t.wall_ratio);
    assert!((1..=10).contains(&t.steps_last_tick), "{}", t.steps_last_tick);
    // fs per wall second ≤ dt / 2 ms
    assert!(t.wall_ratio > 0.0 && t.wall_ratio <= DT_FS / 2e-3 * 1.0001, "{}", t.wall_ratio);
    // speed caps below the budget
    d.apply(DynCommand::SetSpeed(2));
    assert_eq!(d.tick(1e6, &mut never()).steps_last_tick, 2);
}
