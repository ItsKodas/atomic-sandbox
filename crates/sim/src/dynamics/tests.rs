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
    /// `discontinuity` calls (R13), counted apart from `reset`.
    discontinuities: Arc<AtomicUsize>,
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
    fn discontinuity(&mut self) {
        self.discontinuities.fetch_add(1, Ordering::SeqCst);
        self.inner.discontinuity();
    }
    fn sample_density(&mut self, points: &[[f64; 3]]) -> Option<Vec<f64>> {
        self.inner.sample_density(points)
    }
    fn scf_tol(&self) -> Option<f64> {
        self.inner.scf_tol()
    }
    fn side_solver(&self) -> Option<Box<dyn ForceProvider>> {
        self.inner.side_solver()
    }
}

/// Ticks (paused or playing, unbounded) until no reference is pending, at most 10 s.
fn settle_references(d: &mut Dynamics) -> DynSnapshot {
    let t0 = std::time::Instant::now();
    loop {
        let s = d.tick(f64::INFINITY, &mut never());
        if s.bonds_pending.is_empty() {
            return s;
        }
        assert!(t0.elapsed() < std::time::Duration::from_secs(10), "references still pending: {:?}", s.bonds_pending);
        std::thread::sleep(std::time::Duration::from_millis(1));
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

/// The UNDAMPED spring (`drag_damping = 0`; the default is critical, R9) k_drag (target − r) on
/// a free ¹H from rest: Verlet's positions are exactly
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
    d.drag_damping = 0.0;
    d.drag_force_max = f64::INFINITY; // the linear spring's closed form (the cap: `drag_force_is_capped…`)
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
    // independent wells: each mode's Verlet swing is ≤ h_i²/4 of its energy (R6), h_i ≤ h of ¹H
    let h = (0.4f64 / t.scene.nuclei[0].mass).sqrt() * dt_au();
    println!("pins: max |ΔE_conserved| = {cmax:.3e} Ha over 300 steps (bound h²/4·E = {:.3e})", h * h / 4.0 * free.total);
    assert!(cmax <= h * h / 4.0 * free.total, "{cmax}");
    // unpinned: its well pulls it back
    d.apply(DynCommand::Pin(0, false));
    let u = play_n(&mut d, 50);
    assert!(dist(u.scene.nuclei[0].pos, s.nuclei[0].pos) > 1e-3, "an unpinned nucleus moves");
    // pin index out of range: rejected, no panic
    d.apply(DynCommand::Pin(7, true));
    assert!(d.snapshot().last_rejection.is_some());
    assert!(d.snapshot().last_error.is_none(), "a rejection is not a provider error");
    d.apply(DynCommand::Pin(1, false));
    assert!(d.snapshot().last_rejection.is_none(), "the next accepted command clears it");
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
    // the corrected (Hamiltonian, V' + K ≤ E₀) dynamics' Verlet swing: ≤ h²/4 · E₀, h of ¹H
    let h = (0.4f64 / e0.scene.nuclei[0].mass).sqrt() * dt_au();
    println!("COM correction on Harmonic: max |ΔE_total| = {tmax:.3e}, max |ΔE_conserved| = {cmax:.3e} Ha (bound {:.3e})", h * h / 4.0 * e0.total);
    assert!(tmax > 100.0 * cmax, "the correction's work must show in E_total but not in E_conserved");
    assert!(cmax <= h * h / 4.0 * e0.total, "{cmax}");
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
    assert!(d.snapshot().last_rejection.is_some());
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
        assert!(b.last_rejection.is_some(), "{bad:?} accepted");
        assert_eq!(b.thermostat, Some((300.0, 20.0)));
    }

    // rewinding under NVT flips the velocities but warns that it does not retrace
    d.apply(DynCommand::Rewind);
    let r = d.snapshot();
    assert!(r.notes.iter().any(|n| n.contains("not reversible")), "{:?}", r.notes);
    assert_eq!(r.notes_total, 2, "two notes so far: the draw and the rewind");

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
    let discontinuities = Arc::new(AtomicUsize::new(0));
    let provider = Counting { inner: Box::new(Zero), evals: evals.clone(), resets: resets.clone(), discontinuities: discontinuities.clone() };
    let s = scene(vec![
        Nucleus::new(1, [0.0; 3]).with_vel([1e-4, 0.0, 0.0]),
        Nucleus::new(1, [1.4, 0.0, 0.0]).with_vel([0.0, 1e-4, 0.0]),
        Nucleus::new(8, [0.0, 2.0, 0.0]).with_vel([0.0, 0.0, 1e-5]),
    ]);
    let mut d = Dynamics::new(s, Box::new(provider), DT_FS, 1);
    assert_eq!(evals.load(Ordering::SeqCst), 1);
    assert_eq!(resets.load(Ordering::SeqCst), 1, "new() makes its one evaluation cold");
    resets.store(0, Ordering::SeqCst);

    // while playing, Move and ReplaceScene are rejected and change nothing (Add/Remove: R8,
    // `add_and_remove_while_playing`)
    d.apply(DynCommand::Play);
    let before = d.snapshot();
    for cmd in [
        DynCommand::MoveNucleusDiscontinuous { index: 0, pos: [3.0, 0.0, 0.0] },
        DynCommand::ReplaceScene(scene(vec![Nucleus::new(2, [0.0; 3])])),
    ] {
        d.apply(cmd);
        let a = d.snapshot();
        assert!(a.last_rejection.as_deref().is_some_and(|e| e.contains("pause")), "{:?}", a.last_rejection);
        assert!(a.scene == before.scene);
        assert!(a.seq > before.seq);
    }
    assert_eq!(resets.load(Ordering::SeqCst), 0);
    d.apply(DynCommand::Pause);

    // paused: Move teleports, zeroes that velocity, tells the provider (R13: `discontinuity`,
    // warm start kept — not `reset`); no evaluation until tick
    d.apply(DynCommand::Pin(2, true));
    d.apply(DynCommand::DragStart { index: 1, target: [1.4, 1.0, 0.0] });
    assert_eq!(resets.load(Ordering::SeqCst), 0, "pins and drags do not reset");
    d.apply(DynCommand::MoveNucleusDiscontinuous { index: 0, pos: [-1.0, 0.0, 0.0] });
    let m = d.snapshot();
    assert!(m.last_rejection.is_none(), "accepted commands clear the rejection: {:?}", m.last_rejection);
    assert_eq!(m.scene.nuclei[0].pos, [-1.0, 0.0, 0.0]);
    assert_eq!(m.scene.nuclei[0].vel, [0.0; 3]);
    assert_eq!(m.scene.nuclei[1].vel, [0.0, 1e-4, 0.0], "other velocities kept");
    assert!(m.forces_stale && m.forces.is_empty());
    assert_eq!(resets.load(Ordering::SeqCst), 0, "a teleport keeps the warm state (R13)");
    assert_eq!(discontinuities.load(Ordering::SeqCst), 1);
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
    assert_eq!(resets.load(Ordering::SeqCst), 1, "an addition is cold");
    assert_eq!(discontinuities.load(Ordering::SeqCst), 1);
    d.apply(DynCommand::RemoveNucleus(0));
    let r = d.snapshot();
    assert_eq!(r.scene.nuclei.len(), 3);
    assert_eq!(r.pins, vec![false, true, false]);
    assert_eq!(r.drag, Some((0, [1.4, 1.0, 0.0])), "drag index follows its nucleus");
    d.apply(DynCommand::RemoveNucleus(0));
    assert_eq!(d.snapshot().drag, None, "removing the dragged nucleus ends the drag");
    d.apply(DynCommand::RemoveNucleus(9));
    assert!(d.snapshot().last_rejection.is_some());
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
    // a replaced scene must be finite with positive masses
    let mut bad = new.clone();
    bad.nuclei[0].mass = 0.0;
    d.apply(DynCommand::ReplaceScene(bad));
    assert!(d.snapshot().last_rejection.is_some());
    let mut bad = new.clone();
    bad.nuclei[0].vel[1] = f64::INFINITY;
    d.apply(DynCommand::ReplaceScene(bad));
    assert!(d.snapshot().last_rejection.is_some());
    assert_eq!(d.snapshot().step, 1);
}

/// Spec §5.3: `SetIsotope` is accepted at any time — the mass changes at fixed velocity, so K
/// jumps by ΔK = ½(m′ − m)|v|², booked into W like a thermostat rescale (E_conserved flat to
/// rounding); an unlisted A is refused naming the listed ones; the provider is neither reset nor
/// re-evaluated (forces do not depend on mass).
#[test]
fn set_isotope_changes_the_mass_books_delta_k_and_keeps_e_conserved() {
    let evals = Arc::new(AtomicUsize::new(0));
    let resets = Arc::new(AtomicUsize::new(0));
    let discontinuities = Arc::new(AtomicUsize::new(0));
    let s = scene(vec![
        Nucleus::new(1, [0.75, 0.0, 0.0]).with_vel([3e-4, -1e-4, 2e-4]),
        Nucleus::new(1, [-0.75, 0.0, 0.0]).with_vel([-2e-4, 1e-4, 0.0]),
    ]);
    let h = Harmonic { k: 0.3, centres: vec![[0.7, 0.0, 0.0], [-0.7, 0.0, 0.0]] };
    let provider = Counting { inner: Box::new(h), evals: evals.clone(), resets: resets.clone(), discontinuities: discontinuities.clone() };
    let mut d = Dynamics::new(s, Box::new(provider), DT_FS, 1);
    let m_h = d.snapshot().scene.nuclei[0].mass;
    let m_d = units::amu_to_au_mass(units::isotope_mass_amu_a(1, 2).unwrap());

    // paused: the mass changes, the velocity does not, K and E_total jump by ΔK, E_conserved not
    let before = d.snapshot();
    let v = before.scene.nuclei[0].vel;
    let dk = 0.5 * (m_d - m_h) * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    assert!(dk > 0.0);
    d.apply(DynCommand::SetIsotope { index: 0, a: 2 });
    let after = d.snapshot();
    assert_eq!(after.last_rejection, None);
    assert_eq!((after.scene.nuclei[0].z, after.scene.nuclei[0].a), (1, 2));
    assert_eq!(after.scene.nuclei[0].mass, m_d);
    assert_eq!(after.scene.nuclei[0].vel, v);
    assert_eq!(after.scene.nuclei[0].pos, before.scene.nuclei[0].pos);
    assert_eq!(after.scene.nuclei[1], before.scene.nuclei[1]);
    assert!((after.kinetic - before.kinetic - dk).abs() < 1e-12, "ΔK {} vs {dk}", after.kinetic - before.kinetic);
    assert!((after.total - before.total - dk).abs() < 1e-12, "E_total jump {} vs {dk}", after.total - before.total);
    assert!((after.energy_conserved - before.energy_conserved).abs() < 1e-12, "E_conserved moved by {}", after.energy_conserved - before.energy_conserved);
    assert!(!after.forces_stale, "forces do not depend on mass");
    assert_eq!(after.energy_ks, before.energy_ks);
    assert_eq!((evals.load(Ordering::SeqCst), resets.load(Ordering::SeqCst)), (1, 1), "no evaluation, no reset (new() made the first)");

    // an unlisted isotope is refused with the listed ones named, and changes nothing
    d.apply(DynCommand::SetIsotope { index: 0, a: 5 });
    let r = d.snapshot();
    assert_eq!(r.last_rejection.as_deref(), Some("no mass for ⁵H; listed: ¹H ²H ³H"));
    assert_eq!(r.scene, after.scene);
    d.apply(DynCommand::SetIsotope { index: 7, a: 2 });
    assert!(d.snapshot().last_rejection.as_deref().is_some_and(|e| e.contains("nucleus 7")));
    assert_eq!(d.snapshot().scene, after.scene);

    // playing: the command between ticks costs no evaluation and no reset; the E_conserved
    // history stays continuous across the swap back to ¹H
    let n_before = evals.load(Ordering::SeqCst);
    let s1 = play_n(&mut d, 50);
    assert_eq!(evals.load(Ordering::SeqCst) - n_before, 50);
    let e1 = s1.energy_conserved;
    d.apply(DynCommand::SetIsotope { index: 0, a: 1 });
    let s2 = d.snapshot();
    assert_eq!(s2.scene.nuclei[0].mass, m_h);
    assert_eq!(evals.load(Ordering::SeqCst) - n_before, 50, "the command evaluated nothing");
    assert_eq!(resets.load(Ordering::SeqCst), 1, "the provider was not reset");
    assert_eq!(discontinuities.load(Ordering::SeqCst), 0);
    assert!((s2.energy_conserved - e1).abs() < 1e-12, "E_conserved moved by {} on the swap back", s2.energy_conserved - e1);
    let s3 = play_n(&mut d, 50);
    assert_eq!(evals.load(Ordering::SeqCst) - n_before, 100);
    // the harmonic trajectory conserves E to the Verlet swing; the swap did not add a jump
    assert!((s3.energy_conserved - e1).abs() < 1e-6, "E_conserved drifted by {}", s3.energy_conserved - e1);
    // SetIsotope clears a previous rejection like any accepted command
    assert_eq!(s3.last_rejection, None);
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
        assert!(b.last_rejection.is_some(), "dt {bad} accepted");
        assert!((b.dt - dt_au()).abs() < 1e-12);
    }
    // a new dt restarts the history (a new shadow Hamiltonian)
    assert!(step_n(&mut d, 5).history.len() >= 5);
    d.apply(DynCommand::SetDt(0.2));
    assert_eq!(d.snapshot().history.len(), 0);
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

/// R8: Add and Remove are accepted while playing; time keeps advancing, the forces are
/// re-evaluated (cold) at the next tick, and pins and the drag follow their nuclei.
#[test]
fn add_and_remove_while_playing() {
    let evals = Arc::new(AtomicUsize::new(0));
    let resets = Arc::new(AtomicUsize::new(0));
    let discontinuities = Arc::new(AtomicUsize::new(0));
    let provider = Counting { inner: Box::new(Zero), evals: evals.clone(), resets: resets.clone(), discontinuities: discontinuities.clone() };
    let s = scene(vec![
        Nucleus::new(1, [0.0; 3]).with_vel([1e-4, 0.0, 0.0]),
        Nucleus::new(8, [2.0, 0.0, 0.0]).with_vel([0.0, 1e-5, 0.0]),
        Nucleus::new(1, [0.0, 2.0, 0.0]).with_vel([0.0, 0.0, 1e-4]),
    ]);
    let mut d = Dynamics::new(s, Box::new(provider), DT_FS, 1);
    d.apply(DynCommand::Pin(2, true));
    d.apply(DynCommand::DragStart { index: 1, target: [2.0, 1.0, 0.0] });
    d.apply(DynCommand::SetSpeed(5));
    d.apply(DynCommand::Play);
    let p = d.tick(f64::INFINITY, &mut never());
    assert_eq!(p.step, 5);
    let (e0, r0) = (evals.load(Ordering::SeqCst), resets.load(Ordering::SeqCst));

    d.apply(DynCommand::AddNucleus(Nucleus::new(1, [5.0, 0.0, 0.0]).with_vel([1.0, 0.0, 0.0])));
    let a = d.snapshot();
    assert!(a.last_rejection.is_none(), "{:?}", a.last_rejection);
    assert!(!a.paused && a.forces_stale);
    assert_eq!(a.scene.nuclei[3].vel, [0.0; 3], "placed at rest");
    assert_eq!(a.pins, vec![false, false, true, false]);
    assert_eq!(resets.load(Ordering::SeqCst), r0 + 1);
    let t = d.tick(f64::INFINITY, &mut never());
    assert_eq!(evals.load(Ordering::SeqCst), e0 + 1 + 5, "one cold re-evaluation, then the steps");
    assert_eq!(t.step, 10, "time keeps advancing");
    assert!(!t.forces_stale && t.forces.len() == 4);

    d.apply(DynCommand::RemoveNucleus(0));
    let r = d.snapshot();
    assert!(r.last_rejection.is_none() && !r.paused && r.forces_stale);
    assert_eq!(r.pins, vec![false, true, false]);
    assert_eq!(r.drag, Some((0, [2.0, 1.0, 0.0])));
    let pinned = r.scene.nuclei[1].pos;
    let t = d.tick(f64::INFINITY, &mut never());
    assert_eq!(t.step, 15);
    assert_eq!(t.scene.nuclei[1].pos, pinned, "the pin moved with its nucleus");
    d.apply(DynCommand::RemoveNucleus(0));
    let r = d.snapshot();
    assert_eq!(r.drag, None, "removing the dragged nucleus ends the drag");
    assert_eq!(r.pins, vec![true, false]);
    assert_eq!(d.tick(f64::INFINITY, &mut never()).step, 20);
}

/// R9: the critically damped drag settles a free ¹H on a target 2 Bohr away (continuous:
/// r − t = −2 (1 + ωt) e^{−ωt}, ω t = 68 after 400 steps), and the damping's booked work keeps
/// E_conserved flat while E_total takes and loses the spring energy.
#[test]
fn damped_drag_settles_on_the_target() {
    let s = scene(vec![Nucleus::new(1, [0.0; 3])]);
    let m = s.nuclei[0].mass;
    let mut d = Dynamics::new(s, Box::new(Zero), DT_FS, 1);
    d.drag_force_max = f64::INFINITY; // the linear critically damped spring's maths
    let target = [2.0, 0.0, 0.0];
    d.apply(DynCommand::DragStart { index: 0, target });
    let e0 = d.snapshot().energy_conserved;
    let (mut cmax, mut xmax) = (0.0f64, 0.0f64);
    for _ in 0..40 {
        let t = play_n(&mut d, 10);
        cmax = cmax.max((t.energy_conserved - e0).abs());
        xmax = xmax.max(t.scene.nuclei[0].pos[0]);
    }
    let t = d.snapshot();
    let off = dist(t.scene.nuclei[0].pos, target);
    let v = t.scene.nuclei[0].vel.iter().map(|x| x * x).sum::<f64>().sqrt();
    let w = (K_DRAG / m).sqrt() * dt_au();
    println!("damped drag after 400 steps: |r − t| = {off:.3e} Bohr, |v| = {v:.3e}, max x = {xmax:.4}, max |ΔE_conserved| = {cmax:.3e} Ha (spring energy 1 Ha, h²/4 = {:.3e})", w * w / 4.0);
    assert!(off < 0.05, "{off}");
    assert!(v < 1e-6, "{v}");
    assert!(xmax < 2.0 + 0.05, "critical damping does not overshoot (beyond Verlet's O(h)): {xmax}");
    // all of the 1 Ha the spring held was dissipated and booked
    assert!(t.total < 1e-6, "E_total = {}", t.total);
    assert!(cmax <= 2.0 * w * w / 4.0 * 1.0, "E_conserved moved {cmax:.3e} Ha of 1 Ha dissipated");
    assert!(t.drift_per_ps.is_some());
    // a damped drag makes a rewind irreversible: noted
    d.apply(DynCommand::Rewind);
    assert!(d.snapshot().notes.iter().any(|n| n.contains("not reversible")));
}

/// Review M2: a moving target (DragMove every step) books its ΔU, so E_conserved stays within
/// Verlet's swing plus the O(h²) share of the drag work (module doc) while E_total takes the
/// work; and it is continuous across each individual move.
#[test]
fn drag_move_books_the_user_work() {
    let s = scene(vec![Nucleus::new(1, [0.0; 3])]);
    let m = s.nuclei[0].mass;
    let mut d = Dynamics::new(s, Box::new(Zero), DT_FS, 1);
    d.apply(DynCommand::DragStart { index: 0, target: [0.0; 3] });
    let e0 = d.snapshot();
    let mut cmax = 0.0f64;
    // 3 Bohr at 0.01 Bohr per step, then hold for 300 steps to settle
    for i in 1..=600 {
        if i <= 300 {
            let before = d.snapshot().energy_conserved;
            d.apply(DynCommand::DragMove { target: [0.01 * i as f64, 0.0, 0.0] });
            let after = d.snapshot();
            assert!((after.energy_conserved - before).abs() < 1e-14, "a move is booked at fixed r");
        }
        let t = step_n(&mut d, 1);
        cmax = cmax.max((t.energy_conserved - e0.energy_conserved).abs());
    }
    let t = d.snapshot();
    // the drag work: everything that went in was dissipated by the damping (E_total ≈ 0 again)
    // or is still kinetic; the work done is ≈ the damping loss, of order γ v² T
    let w = (K_DRAG / m).sqrt() * dt_au();
    let gamma = 2.0 * (K_DRAG * m).sqrt();
    let v = 0.01 / dt_au();
    let work = gamma * v * v * 300.0 * dt_au(); // dissipated at steady drag speed
    println!("moving drag: target 3 Bohr in 300 steps; r = {:.4}; max |ΔE_conserved| = {cmax:.3e} Ha; drag work ≈ {work:.3e} Ha (h²/4 = {:.3e})",
             t.scene.nuclei[0].pos[0], w * w / 4.0);
    assert!((t.scene.nuclei[0].pos[0] - 3.0).abs() < 0.05);
    // the module doc's estimate of the moving-target residual, (h²/4)·W, is the bound (the
    // nucleus trails the target closely, so U_spring and with it the residual stay far below)
    assert!(cmax <= w * w / 4.0 * work, "{cmax}");
}

/// Review M2: `new()` failing, a cancelled stale re-evaluation, and the snapshot placeholder.
#[test]
fn constructor_failure_and_cancelled_reevaluation() {
    let (s, h) = harmonic_three();
    // new()'s evaluation is call 1: it fails, the driver is paused with the error and stale
    let mut d = Dynamics::new(s.clone(), Box::new(Failing::new(h.clone(), 1)), DT_FS, 1);
    let f = d.snapshot();
    assert!(f.paused && f.forces_stale && f.last_error.is_some());
    // paused ticks do not retry; Play does (call 2 succeeds), then steps
    assert!(d.tick(f64::INFINITY, &mut never()).forces_stale);
    d.apply(DynCommand::Play);
    let p = d.tick(f64::INFINITY, &mut never());
    assert!(!p.forces_stale && p.last_error.is_none());
    assert_eq!(p.step, 1, "budget ∞: the re-evaluation and then a step");
    // review M7: after a stale re-evaluation an exhausted budget takes no step
    d.apply(DynCommand::Pause);
    d.apply(DynCommand::MoveNucleusDiscontinuous { index: 0, pos: [0.0; 3] });
    d.apply(DynCommand::Play);
    let q = d.tick(0.0, &mut never());
    assert!(!q.forces_stale);
    assert_eq!(q.steps_last_tick, 0, "the solve spent the tick");
    assert_eq!(d.tick(0.0, &mut never()).steps_last_tick, 1, "the next tick steps");

    // a cancelled stale re-evaluation: no error, still stale, retried next tick
    let mut c = Dynamics::new(s.clone(), Box::new(Failing::new(h, 2).with_kind(FailKind::Cancelled)), DT_FS, 1);
    c.apply(DynCommand::MoveNucleusDiscontinuous { index: 0, pos: [0.1, 0.0, 0.0] });
    let x = c.tick(f64::INFINITY, &mut never());
    assert!(x.forces_stale && x.last_error.is_none());
    let y = c.tick(f64::INFINITY, &mut never());
    assert!(!y.forces_stale);

    let z = DynSnapshot::default();
    assert!(z.paused && z.forces_stale && z.scene.nuclei.is_empty() && z.history.is_empty());
}

/// R10: the thermostat preserves the COM velocity, so a thermostatted system falls under the
/// field at the NVE pace (u = a t), while its internal temperature is controlled.
#[test]
fn thermostatted_system_falls_at_the_nve_pace() {
    let mut s = scene(vec![Nucleus::new(1, [0.0; 3]), Nucleus::new(1, [1.4, 0.0, 0.0]), Nucleus::new(8, [0.0, 1.5, 0.0])]);
    let a = [0.0, 0.0, -1e-6];
    s.external_field = a;
    let mut d = Dynamics::new(s, Box::new(Zero), DT_FS, 3);
    d.apply(DynCommand::SetThermostat(Some((300.0, 10.0))));
    assert!(d.snapshot().scene.com_velocity().iter().all(|u| u.abs() < 1e-18), "the draw starts the COM at rest");
    let e0 = d.snapshot().energy_conserved;
    let t = play_n(&mut d, 500);
    let u = t.scene.com_velocity();
    let expect = a[2] * t.sim_time;
    println!("thermostatted fall: u_z = {:.6e} vs a t = {expect:.6e}; T = {:.1} K; ΔE_conserved = {:.3e}", u[2], t.temperature, t.energy_conserved - e0);
    assert!((u[2] - expect).abs() < 1e-9 * expect.abs(), "{u:?} vs {expect}");
    assert!(u[0].abs() < 1e-15 && u[1].abs() < 1e-15);
    assert!(t.temperature > 30.0 && t.temperature < 3000.0, "{}", t.temperature);
    // free particles: no Verlet swing at all, so the booking is exact to rounding
    assert!((t.energy_conserved - e0).abs() < 1e-12, "{}", t.energy_conserved - e0);
}

/// B-1 (final review): the R7 "at rest" test is on the internal K. A molecule translating at a
/// uniform u (lab K = ½ M u² ≫ 1 % of K̄) with no internal motion still gets the
/// Maxwell–Boltzmann draw; the draw is about u, so the COM velocity is preserved, the internal
/// temperature is the target exactly, and the booked work is the internal ΔK (E_conserved
/// unchanged).
#[test]
fn translating_internally_cold_molecule_gets_the_draw() {
    let u = [3e-4, -1e-4, 2e-4]; // ½ M |u|² ≈ 2.3e-3 Ha for H₂O: ≈ 80× the at-rest threshold, 1 % of K̄(6 dof, 300 K)
    let s = scene(vec![
        Nucleus::new(8, [0.0; 3]).with_vel(u),
        Nucleus::new(1, [1.43, 1.11, 0.0]).with_vel(u),
        Nucleus::new(1, [-1.43, 1.11, 0.0]).with_vel(u),
    ]);
    let k_lab = s.kinetic_energy();
    let mut d = Dynamics::new(s, Box::new(Zero), DT_FS, 11);
    let before = d.snapshot();
    assert!(before.temperature.abs() < 1e-9, "internally at rest: {}", before.temperature);
    d.apply(DynCommand::SetThermostat(Some((300.0, 20.0))));
    let on = d.snapshot();
    let k_bar = 0.5 * 6.0 * units::kelvin_to_hartree(300.0);
    println!("translating H₂O: lab K {k_lab:.3e} Ha (1 % of K̄ = {:.3e}); after: T = {:.6} K, u = {:?}, notes {:?}",
             0.01 * k_bar, on.temperature, on.scene.com_velocity(), on.notes);
    assert!(k_lab > 0.01 * k_bar, "the scene must fail a lab-frame at-rest test");
    assert!(on.notes.iter().any(|n| n.contains("Maxwell–Boltzmann")), "{:?}", on.notes);
    assert!((on.temperature - 300.0).abs() < 1e-9 * 300.0, "{}", on.temperature);
    assert!((on.scene.internal_kinetic_energy() - k_bar).abs() < 1e-12 * k_bar);
    let com = on.scene.com_velocity();
    for a in 0..3 {
        assert!((com[a] - u[a]).abs() < 1e-15, "COM velocity {com:?} vs {u:?}");
    }
    assert!((on.energy_conserved - before.energy_conserved).abs() < 1e-14, "the draw is booked: {} vs {}",
            on.energy_conserved, before.energy_conserved);
}

/// R12: the drag spring's force is capped at F_DRAG_MAX with its direction kept, and the
/// bookkeeping stays exact for the force applied. An undamped capped spring pulls a free ¹H from
/// rest toward a target 2 Bohr away (d ≫ d_c = 0.2 Bohr): the first step is the constant force
/// F_max exactly (x₁ = ½ (F_max/m) dt², v₁ = (F_max/m) dt); E_total = K peaks at the capped
/// potential U(2) = F_max (2 − d_c/2) = 0.19 Ha (not the uncapped 1 Ha), and E_conserved stays
/// within Verlet's swing of that mode.
#[test]
fn drag_force_is_capped_and_its_work_booked() {
    let s = scene(vec![Nucleus::new(1, [0.0; 3])]);
    let m = s.nuclei[0].mass;
    let mut d = Dynamics::new(s, Box::new(Zero), DT_FS, 1);
    d.drag_damping = 0.0;
    let target = [0.0, 2.0, 0.0];
    d.apply(DynCommand::DragStart { index: 0, target });
    let e0 = d.snapshot().energy_conserved;
    let one = step_n(&mut d, 1);
    let (a, dt) = (F_DRAG_MAX / m, dt_au());
    let y = one.scene.nuclei[0].pos[1];
    assert!((y - 0.5 * a * dt * dt).abs() < 1e-15, "x₁ = {y}");
    assert!((one.scene.nuclei[0].vel[1] - a * dt).abs() < 1e-18);
    assert_eq!((one.scene.nuclei[0].pos[0], one.scene.nuclei[0].pos[2]), (0.0, 0.0), "direction kept");
    let (mut kmax, mut cmax) = (0.0f64, 0.0f64);
    for _ in 0..2000 {
        let t = step_n(&mut d, 1);
        kmax = kmax.max(t.total);
        cmax = cmax.max((t.energy_conserved - e0).abs());
    }
    let u2 = F_DRAG_MAX * (2.0 - 0.5 * F_DRAG_MAX / K_DRAG);
    println!("capped drag: max E_total {kmax:.5} Ha (U(2) = {u2:.5}), max |ΔE_conserved| {cmax:.3e} Ha");
    assert!((kmax - u2).abs() < 0.01 * u2, "the kinetic peak is the capped potential: {kmax} vs {u2}");
    // Verlet's swing on the stiff quadratic core (ω dt as in the linear test) times U(2)
    let w = (K_DRAG / m).sqrt() * dt;
    assert!(cmax <= w * w / 4.0 * u2 * 1.5, "E_conserved moved {cmax:.3e}");
}

/// A provider that answers like a surrogate (instantly, `Source::Surrogate`) except on every
/// `every`-th evaluation, which is a slow `Source::Fallback` (`ms` of wall): the model/DFT mix
/// of an estimated run. Free motion (zero forces).
struct Mixed {
    every: usize,
    ms: u64,
    calls: usize,
}

impl ForceProvider for Mixed {
    fn evaluate(&mut self, scene: &Scene, _should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        self.calls += 1;
        let mut stats = EvalStats::default();
        if self.calls % self.every == 0 {
            std::thread::sleep(std::time::Duration::from_millis(self.ms));
            stats.source = Source::Fallback;
        } else {
            stats.source = Source::Surrogate;
        }
        Ok(ForceResult { energy: 0.0, forces: vec![[0.0; 3]; scene.nuclei.len()], stats, display: None })
    }
    fn name(&self) -> &'static str {
        "mixed"
    }
    fn reset(&mut self) {}
}

/// Task 9 item 4: the step-cost averages are kept per kind (model / DFT), and the budget check
/// predicts the next step by the latest one's kind — so one slow DFT step (here 100 ms, a
/// fallback) no longer makes the next ticks stop after one fast model step each while a mixed
/// average decays (0.2 × 100 = 20 ms > the 16 ms budget: it used to take two such ticks).
#[test]
fn a_slow_dft_step_does_not_throttle_the_model_steps_after_it() {
    let s = scene(vec![Nucleus::new(1, [0.0; 3]).with_vel([1e-4, 0.0, 0.0])]);
    let mut d = Dynamics::new(s, Box::new(Mixed { every: 400, ms: 100, calls: 0 }), DT_FS, 1);
    d.apply(DynCommand::SetSpeed(1000));
    d.apply(DynCommand::Play);
    let mut ticks = Vec::new();
    for _ in 0..12 {
        ticks.push(d.tick(16.0, &mut never()));
    }
    let after: Vec<usize> = ticks
        .windows(2)
        .filter(|w| w[0].source == Source::Fallback || w[0].history_source.iter().rev().take(w[0].steps_last_tick).any(|&x| x == Source::Fallback))
        .map(|w| w[1].steps_last_tick)
        .collect();
    let last = ticks.last().unwrap();
    println!("steps per tick {:?}; after a fallback tick {after:?}; EMA model {:.4} ms, DFT {:.1} ms, next {:.4} ms",
             ticks.iter().map(|t| t.steps_last_tick).collect::<Vec<_>>(), last.ms_per_step_model, last.ms_per_step_dft, last.ms_per_step);
    assert!(!after.is_empty(), "test premise: a fallback happened");
    assert!(after.iter().all(|&n| n > 100), "a model tick after a DFT step ran only {after:?} steps");
    assert!(last.ms_per_step_dft >= 50.0 && last.ms_per_step_model < 1.0, "{} / {}", last.ms_per_step_dft, last.ms_per_step_model);
}

// ---- container walls (module doc, "Container walls") ----

/// The providers' margin guard (R12) around free motion: errors like `CpuKs`/`GpuKs` when a
/// nucleus is closer than 3h to a face ("left the simulation cell").
struct Guarded;

impl ForceProvider for Guarded {
    fn evaluate(&mut self, scene: &Scene, _a: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        scene.check_inside_cell()?;
        Ok(ForceResult { energy: 0.0, forces: vec![[0.0; 3]; scene.nuclei.len()], stats: EvalStats::default(), display: None })
    }
    fn name(&self) -> &'static str {
        "guarded"
    }
    fn reset(&mut self) {}
}

/// ω dt of a nucleus of mass m on the wall's layer (its harmonic part, the stiffest).
fn wall_omega_dt(m: f64, dt_fs: f64) -> f64 {
    (WALL_K / m).sqrt() * units::fs_to_au_time(dt_fs)
}

/// The exact turning depth into the layer at kinetic energy e along the axis: u(δ) = e.
fn wall_depth(e: f64) -> f64 {
    let (k, a) = (WALL_K, WALL_ONSET);
    if e <= k * a * a / 6.0 { (6.0 * a * e / k).cbrt() } else { 0.5 * a + (2.0 * e / k - a * a / 12.0).sqrt() }
}

/// A free ¹H at 0.05 Ha heads for the +x wall: it turns inside the layer at the exact depth (to
/// Verlet's overshoot), comes back with its speed, bounces off −x too; E_conserved = K + U_wall
/// holds to Verlet's swing (h²/4)·E on the layer — and stays bounded over ~300 bounces (module
/// doc: the C¹ ½ k d² wall random-walked to 20 % here; the C² wall stays at ~1e-3) — and
/// switching the walls on with the nucleus inside the layer is booked (no jump).
#[test]
fn a_free_nucleus_bounces_off_the_wall_with_e_conserved_flat() {
    let m = Nucleus::new(1, [0.0; 3]).mass;
    let e = 0.05;
    let v = (2.0 * e / m).sqrt();
    let s = scene(vec![Nucleus::new(1, [0.0; 3]).with_vel([v, 0.0, 0.0])]);
    let mut d = Dynamics::new(s, Box::new(Zero), DT_FS, 1);
    let walls = Walls::new(3.0);
    d.apply(DynCommand::SetWalls(Some(walls)));
    assert_eq!(d.snapshot().walls, Some(walls));
    let e0 = d.snapshot().energy_conserved;
    assert!((e0 - e).abs() < 1e-15, "{e0}");
    let (mut xmax, mut xmin, mut cmax, mut kdev) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (mut turned, mut was_in, mut bounces) = (false, false, 0usize);
    for _ in 0..20_000 {
        let t = step_n(&mut d, 1);
        let n = &t.scene.nuclei[0];
        xmax = xmax.max(n.pos[0]);
        xmin = xmin.min(n.pos[0]);
        turned |= n.vel[0] < 0.0;
        assert_eq!((n.pos[1], n.pos[2]), (0.0, 0.0));
        cmax = cmax.max((t.energy_conserved - e0).abs());
        // after each bounce, out of the layer: K alone is the energy
        let inside = n.pos[0].abs() > 2.0;
        if was_in && !inside {
            bounces += 1;
            kdev = kdev.max((t.kinetic / e - 1.0).abs());
        }
        was_in = inside;
    }
    let t = d.snapshot();
    let w = wall_omega_dt(m, DT_FS);
    let depth = wall_depth(e);
    println!("wall bounce: x ∈ [{xmin:.4}, {xmax:.4}] (layer from ±2, exact depth {depth:.4}); |v| {:.6e} → {:.6e}; max |ΔE_conserved| {cmax:.3e} Ha (h²/4·E = {:.3e}); {bounces} bounces, max |K/E − 1| after one {kdev:.2e}",
             v, t.scene.nuclei[0].vel[0].abs(), w * w / 4.0 * e);
    assert!(turned && xmin < -2.0 && bounces > 250, "it bounced off both walls: x ∈ [{xmin}, {xmax}], {bounces} bounces");
    assert!(xmax < 2.0 + depth * 1.02 && xmax > 2.0 + depth * 0.95, "turning point {xmax} vs 2 + {depth}");
    assert!(cmax <= 1.05 * w * w / 4.0 * e, "E_conserved moved {cmax:.3e}");
    assert!(kdev < 2e-3, "the energy after a bounce wanders: {kdev:.2e}");
    assert!((t.scene.nuclei[0].vel[0].abs() - v).abs() < 1e-3 * v, "the speed comes back");
    // switching on and off inside the layer is booked: no jump in E_conserved
    let mut d = Dynamics::new(scene(vec![Nucleus::new(1, [2.5, -2.4, 0.0])]), Box::new(Zero), DT_FS, 1);
    let before = d.snapshot().energy_conserved;
    d.apply(DynCommand::SetWalls(Some(walls)));
    let on = d.snapshot();
    assert!(on.walls_work > 0.0 && (on.energy_conserved - before).abs() < 1e-15, "{} {}", on.walls_work, on.energy_conserved);
    d.apply(DynCommand::SetWalls(None));
    assert!((d.snapshot().energy_conserved - before).abs() < 1e-15);
    // beyond the cell's maximum (33 × 0.25: 3.25 − 0.125) it is refused, nothing changes
    d.apply(DynCommand::SetWalls(Some(Walls::new(3.2))));
    assert!(d.snapshot().last_rejection.is_some() && d.snapshot().walls.is_none());
    d.apply(DynCommand::SetWalls(Some(Walls::new(Walls::max_half_width(&d.snapshot().scene.cell)))));
    assert!(d.snapshot().last_rejection.is_none() && d.snapshot().walls.is_some());
}

/// The piston: shrinking the box while playing squeezes a nucleus at rest; the walls' moves book
/// positive work (W), the nucleus gains kinetic energy, and E_conserved stays flat to the moving
/// wall's shadow residual (≈ (h²/4) of the work, as for a moving drag target).
#[test]
fn shrinking_the_box_while_playing_books_positive_work() {
    let m = Nucleus::new(1, [0.0; 3]).mass;
    let s = scene(vec![Nucleus::new(1, [1.5, 0.3, -0.2])]);
    let mut d = Dynamics::new(s, Box::new(Zero), DT_FS, 1);
    d.apply(DynCommand::SetWalls(Some(Walls::new(3.0))));
    let e0 = d.snapshot().energy_conserved;
    let mut cmax = 0.0f64;
    for i in 1..=500 {
        if i <= 300 {
            let before = d.snapshot().energy_conserved;
            d.apply(DynCommand::SetWalls(Some(Walls::new(3.0 - 0.005 * i as f64))));
            assert!((d.snapshot().energy_conserved - before).abs() < 1e-15, "a move is booked at fixed r");
        }
        let t = step_n(&mut d, 1);
        cmax = cmax.max((t.energy_conserved - e0).abs());
    }
    let t = d.snapshot();
    let w = wall_omega_dt(m, DT_FS);
    println!("piston: walls 3 → 1.5 Bohr; W_walls = {:.4e} Ha, K = {:.4e} Ha, max |ΔE_conserved| = {cmax:.3e} Ha (h²/4·W = {:.3e}); x = {:.3}",
             t.walls_work, t.kinetic, w * w / 4.0 * t.walls_work, t.scene.nuclei[0].pos[0]);
    // the nucleus started at rest: everything it has is the piston's work (≈ 8.6e-4 Ha here,
    // all kinetic once it is out of the layer)
    assert!(t.walls_work > 1e-4, "the piston did work: {}", t.walls_work);
    assert!((t.kinetic - t.walls_work).abs() <= w * w / 4.0 * t.walls_work, "K {} from W {}", t.kinetic, t.walls_work);
    assert!(cmax <= w * w / 4.0 * t.walls_work, "E_conserved moved {cmax:.3e}");
    // a replaced, smaller cell moves the walls in (with a note)
    d.apply(DynCommand::Pause);
    let mut small = d.snapshot().scene;
    small.cell = Cell { n: 25, h: 0.25 }; // max half-width 3 − 0.75 − 0.125 = 2.125
    small.nuclei[0].pos = [0.0; 3];
    d.apply(DynCommand::ReplaceScene(small));
    let r = d.snapshot();
    assert_eq!(r.walls.map(|w| w.half_width), Some(1.5), "1.5 already fits");
    d.apply(DynCommand::SetWalls(Some(Walls::new(2.125))));
    let mut smaller = d.snapshot().scene;
    smaller.cell = Cell { n: 21, h: 0.25 }; // max 2.5 − 0.875 = 1.625
    d.apply(DynCommand::ReplaceScene(smaller));
    let r = d.snapshot();
    assert_eq!(r.walls.map(|w| w.half_width), Some(1.625));
    assert!(r.notes.last().is_some_and(|n| n.contains("moved in")), "{:?}", r.notes);
}

/// The stopping bound (module doc): a ¹H with 1 Ha of kinetic energy along every axis (the
/// corner, the worst case) at the largest walls the cell allows never reaches the providers'
/// margin, at dt 0.25 and 0.5 fs — so the "left the simulation cell" pause cannot fire — while
/// 1.5 Ha per axis (beyond u(D + h/2) = 1.38 Ha) does reach it: the guard still protects the
/// solver.
#[test]
fn the_wall_stops_a_one_hartree_nucleus_inside_the_margin() {
    for dt_fs in [0.25, 0.5] {
        for (e, stops) in [(1.0, true), (1.5, false)] {
            let m = Nucleus::new(1, [0.0; 3]).mass;
            let v = (2.0 * e / m).sqrt();
            let s = scene(vec![Nucleus::new(1, [0.0; 3]).with_vel([v, -v, v])]);
            let cell = s.cell.clone();
            let mut d = Dynamics::new(s, Box::new(Guarded), dt_fs, 1);
            let l = Walls::max_half_width(&cell);
            d.apply(DynCommand::SetWalls(Some(Walls::new(l))));
            let mut worst = 0.0f64;
            for _ in 0..400 {
                let t = step_n(&mut d, 1);
                if t.last_error.is_some() {
                    break;
                }
                worst = worst.max(t.scene.nuclei[0].pos.iter().fold(0.0f64, |a, x| a.max(x.abs())));
            }
            let t = d.snapshot();
            let w = wall_omega_dt(m, dt_fs);
            println!("dt {dt_fs} fs, {e} Ha per axis: deepest |x| {worst:.4} (face {l:.3}, margin {:.3}; exact depth {:.4}, ×1/√(1 − (ω dt)²/4) = {:.4}); error: {}",
                     cell.max_coord(), wall_depth(e), wall_depth(e) / (1.0 - w * w / 4.0).sqrt(), t.last_error.is_some());
            if stops {
                assert!(t.last_error.is_none(), "dt {dt_fs}: {:?}", t.last_error);
                assert!(worst < cell.max_coord(), "dt {dt_fs}: reached {worst}");
                assert_eq!(t.step, 400);
            } else {
                assert!(t.last_error.as_deref().is_some_and(|e| e.contains("left the simulation cell")), "dt {dt_fs}: {:?}", t.last_error);
            }
        }
    }
}

// ---- the relax tool (module doc, "Relax"; M6 spec §6) ----

/// Review Focus 5 (MUST-7 / S2): every FIRE step books ΔE_KS + ΔK, so E_conserved is constant
/// to rounding through the whole relax — backtracks included — and W is exactly the energy
/// drained (E_end − E_start: W is "energy put in", so it is negative here).
#[test]
fn fire_books_every_step_and_keeps_e_conserved_constant() {
    // the brief's fixture (harmonic H₂ at ±0.7, k = 0.5, compressed to R = 0.9) converges in
    // 20 steps with ONE backtrack (measured: the midpoint backtrack lands the second descent
    // inside F_TOL), so the fixture is the H/O/H cluster: three independent wells with a 4×
    // frequency spread, displaced ~0.5 Bohr and moving (so the start quench is booked too) —
    // the modes cross their minima at different times, and P = ΣF·v changes sign several
    // times (measured: 4 backtracks in 58 steps; `harmonic_three`'s smaller displacements: 2)
    let s = scene(vec![
        Nucleus::new(1, [0.4, -0.3, 0.2]).with_vel([1e-4, 0.0, -5e-5]),
        Nucleus::new(8, [1.9, 0.0, -0.5]).with_vel([0.0, 2e-5, 1e-5]),
        Nucleus::new(1, [-1.0, 1.4, 0.3]).with_vel([-3e-5, 1e-4, 0.0]),
    ]);
    let h = Harmonic { k: 0.4, centres: vec![[0.0; 3], [1.45, 0.35, -0.2], [-1.43, 1.11, 0.0]] };
    let mut d = Dynamics::new(s, Box::new(h), DT_FS, 1);
    let start = d.snapshot();
    assert_eq!(start.relax, None);
    let (e_c0, e_start) = (start.energy_conserved, start.total);
    d.apply(DynCommand::Relax);
    let s0 = d.snapshot();
    assert_eq!(s0.last_rejection, None);
    assert!(s0.paused);
    let r0 = s0.relax.clone().expect("relaxing");
    assert_eq!((r0.steps, r0.converged, r0.alpha), (0, None, 0.1));
    assert!((r0.dt_fs - 0.25).abs() < 1e-12);
    assert_eq!(r0.max_force, start.forces.iter().map(|f| (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt()).fold(0.0, f64::max));
    assert_eq!(r0.max_force, r0.max_force_augmented);
    // the start quench: K → 0, booked — E_conserved did not move
    assert_eq!(s0.kinetic, 0.0);
    assert!((s0.energy_conserved - e_c0).abs() <= 1e-10);
    let (mut ticks, mut backtracks, mut worst) = (0usize, 0usize, 0.0f64);
    let last = loop {
        // budget < 0: exactly one FIRE step per tick
        let snap = d.tick(-1.0, &mut never());
        ticks += 1;
        let r = snap.relax.clone().expect("relaxing");
        assert_eq!(r.steps, ticks);
        let de = (snap.energy_conserved - e_c0).abs();
        worst = worst.max(de);
        assert!(de <= 1e-10, "tick {ticks}: E_conserved moved by {de:.2e}");
        assert_eq!((snap.sim_time, snap.step, snap.steps_last_tick), (0.0, 0, 0));
        assert!(snap.paused && !snap.forces_stale);
        if r.converged.is_none() && snap.scene.nuclei.iter().all(|n| n.vel == [0.0; 3]) {
            backtracks += 1;
        }
        if r.converged.is_some() {
            break snap;
        }
        assert!(ticks < 2000, "no outcome");
    };
    let r = last.relax.clone().unwrap();
    let e_end = last.total;
    println!(
        "FIRE on the H/O/H cluster: {} steps, {backtracks} backtracks; max |ΔE_conserved| {worst:.2e} Ha; W {:+.6e} vs E_end − E_start {:+.6e} (diff {:.1e}); max|F| {:.2e}",
        r.steps, d.work, e_end - e_start, (d.work - (e_end - e_start)).abs(), r.max_force
    );
    assert_eq!(r.converged, Some(true));
    assert!(backtracks >= 3, "{backtracks} P ≤ 0 events: the test is not exercising the backtrack booking");
    assert!(last.scene.nuclei.iter().all(|n| n.vel == [0.0; 3]));
    assert_eq!(last.kinetic, 0.0);
    assert_eq!(last.drift_per_ps, None, "the history restarted");
    assert!(last.history.is_empty());
    assert!(r.max_force_augmented < 1e-3 && r.max_force == r.max_force_augmented);
    assert!((d.work - (e_end - e_start)).abs() <= 1e-10, "W {} vs E_end − E_start {}", d.work, e_end - e_start);
    assert!(e_end < e_start, "the relax drained energy");
    // every nucleus within F_TOL / k of its well
    for (n, c) in last.scene.nuclei.iter().zip([[0.0; 3], [1.45, 0.35, -0.2], [-1.43, 1.11, 0.0]]) {
        assert!(dist(n.pos, c) <= 1e-3 / 0.4, "{:?} vs {c:?}", n.pos);
    }
    assert_eq!(r.verified_max_force, None, "not a surrogate");
    assert_eq!(r.note, None);
    assert!(last.notes.last().is_some_and(|n| n.contains("relax converged")));
    // afterwards: ordinary dynamics from rest, E_conserved continuous (W kept), the history fresh
    let after = step_n(&mut d, 1);
    assert_eq!((after.step, after.relax.clone().unwrap().converged), (1, Some(true)));
    assert!((after.energy_conserved - e_c0).abs() < 1e-8, "E_conserved after the first MD step moved by {:.2e}", after.energy_conserved - e_c0);
    assert_eq!(after.history.len(), 1);
}

/// Independent wells at the origin for any number of nuclei: F = −k r, E = ½ k Σ|r|² (an added
/// nucleus is welcome, unlike `Harmonic` with its fixed centre list).
struct OriginWells {
    k: f64,
}

impl ForceProvider for OriginWells {
    fn evaluate(&mut self, scene: &Scene, _a: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        let forces: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos.map(|x| -self.k * x)).collect();
        let energy = 0.5 * self.k * scene.nuclei.iter().map(|n| n.pos.iter().map(|x| x * x).sum::<f64>()).sum::<f64>();
        Ok(ForceResult { energy, forces, stats: EvalStats::default(), display: None })
    }
    fn name(&self) -> &'static str {
        "origin wells"
    }
    fn reset(&mut self) {}
}

#[test]
fn relax_refuses_play_and_accepts_edits_and_pause_ends_it() {
    let s = scene(vec![Nucleus::new(1, [0.3, 0.0, 0.0]).with_vel([1e-4, 0.0, 0.0]), Nucleus::new(1, [-0.2, 0.1, 0.0])]);
    let mut d = Dynamics::new(s, Box::new(OriginWells { k: 0.3 }), DT_FS, 1);
    let before = d.snapshot();
    d.apply(DynCommand::Relax);
    let s0 = d.snapshot();
    // the run's velocities are quenched at the start, the ΔK booked
    assert!(s0.scene.nuclei.iter().all(|n| n.vel == [0.0; 3]));
    assert!((s0.energy_conserved - before.energy_conserved).abs() < 1e-12);
    for _ in 0..8 {
        d.tick(-1.0, &mut never());
    }
    let s8 = d.snapshot();
    let r8 = s8.relax.clone().unwrap();
    assert_eq!((r8.steps, r8.converged), (8, None));
    assert!(r8.dt_fs > 0.25, "dt grew after the 6th downhill step: {}", r8.dt_fs);
    assert!(r8.alpha < 0.1);

    d.apply(DynCommand::Play);
    let p = d.snapshot();
    assert_eq!(p.last_rejection.as_deref(), Some("relaxing: Stop relax first"));
    assert!(p.paused);
    d.apply(DynCommand::Step(3));
    assert_eq!(d.snapshot().last_rejection.as_deref(), Some("relaxing: Stop relax first"));
    assert_eq!(d.snapshot().pending_steps, 0);
    d.apply(DynCommand::StartSampling(SamplingPlan::default()));
    assert_eq!(d.snapshot().last_rejection.as_deref(), Some("relaxing: Stop relax first"));
    d.apply(DynCommand::Relax);
    assert!(d.snapshot().last_rejection.as_deref().is_some_and(|m| m.starts_with("already relaxing")));

    // an edit is accepted: the forces go stale, the next tick re-evaluates (spending the tick)
    // and FIRE restarts at its start values on the edited scene
    d.apply(DynCommand::AddNucleus(Nucleus::new(1, [0.0, 0.5, 0.0])));
    let sa = d.snapshot();
    assert_eq!(sa.last_rejection, None);
    assert!(sa.forces_stale);
    assert_eq!(sa.scene.nuclei.len(), 3);
    let st = d.tick(-1.0, &mut never());
    assert!(!st.forces_stale);
    assert_eq!(st.forces.len(), 3);
    let rt = st.relax.clone().unwrap();
    assert!((rt.dt_fs - 0.25).abs() < 1e-12, "dt back at the start: {}", rt.dt_fs);
    assert_eq!((rt.alpha, rt.steps, rt.converged), (0.1, 8, None));
    assert!((rt.max_force - 0.15).abs() < 1e-12, "the new nucleus's force k·0.5 is the maximum: {}", rt.max_force);
    let e_c = st.energy_conserved;
    let st2 = d.tick(-1.0, &mut never());
    assert_eq!(st2.relax.clone().unwrap().steps, 9);
    assert!((st2.energy_conserved - e_c).abs() <= 1e-10);
    assert_eq!((st2.sim_time, st2.step), (0.0, 0));

    // Pause ends it where it is
    d.apply(DynCommand::Pause);
    let sp = d.snapshot();
    let rp = sp.relax.clone().unwrap();
    assert_eq!((rp.converged, rp.steps), (Some(false), 9));
    assert!(rp.note.as_deref().is_some_and(|n| n.starts_with("stopped after 9 steps")), "{:?}", rp.note);
    assert!(sp.paused);
    assert!(sp.scene.nuclei.iter().all(|n| n.vel == [0.0; 3]));
    assert!((sp.energy_conserved - e_c).abs() <= 1e-10);
    assert_eq!(sp.drift_per_ps, None);
    // and the dynamics are free again
    d.apply(DynCommand::Play);
    assert_eq!(d.snapshot().last_rejection, None);
    assert!(!d.snapshot().paused);
    let played = d.tick(f64::INFINITY, &mut never());
    assert_eq!(played.step, 1);
    d.apply(DynCommand::Pause);
    // StopRelax without a relax is a no-op; with one it ends it like Pause
    d.apply(DynCommand::StopRelax);
    assert_eq!(d.snapshot().last_rejection, None);
    assert_eq!(d.snapshot().relax.clone().unwrap().steps, 9);
    d.apply(DynCommand::Relax);
    d.tick(-1.0, &mut never());
    d.apply(DynCommand::StopRelax);
    let r = d.snapshot().relax.clone().unwrap();
    assert_eq!((r.steps, r.converged), (1, Some(false)));
    // a pinned nucleus holds through a relax
    d.apply(DynCommand::Pin(2, true));
    let r2 = d.snapshot().scene.nuclei[2].pos;
    d.apply(DynCommand::Relax);
    let fin = loop {
        let snap = d.tick(f64::INFINITY, &mut never());
        if snap.relax.as_ref().unwrap().converged.is_some() {
            break snap;
        }
    };
    assert_eq!(fin.scene.nuclei[2].pos, r2);
    assert_eq!(fin.relax.clone().unwrap().converged, Some(true));
    assert!(fin.scene.nuclei[..2].iter().all(|n| n.pos.iter().all(|x| x.abs() < 1e-3 / 0.3)));
}

/// M6 Task 1: the model commands reach the provider only through `as_trainable`; a provider
/// that cannot be taught refuses them with the M5 text (`no_surrogate`), and nothing changes.
#[test]
fn the_driver_refuses_model_commands_on_a_provider_that_is_not_trainable() {
    let sc = scene(vec![Nucleus::new(1, [0.7, 0.0, 0.0]), Nucleus::new(1, [-0.7, 0.0, 0.0])]);
    let mut d = Dynamics::new(sc.clone(), Box::new(Harmonic::at_rest_positions(&sc, 0.5)), DT_FS, 1);
    assert!(d.provider_mut().as_trainable().is_none());
    let path = std::env::temp_dir().join("sim-not-trainable.model");
    for (cmd, what) in [
        (DynCommand::StartSampling(SamplingPlan::default()), "start sampling"),
        (DynCommand::SaveModel(path.clone()), "save model"),
        (DynCommand::LoadModel(path.clone()), "load model"),
        (DynCommand::ForgetModel, "forget model"),
    ] {
        let notes = d.snapshot().notes_total;
        d.apply(cmd);
        let snap = d.snapshot();
        assert_eq!(
            snap.last_rejection,
            Some(format!("{what}: the provider 'harmonic' has no surrogate model (choose the estimated solver)"))
        );
        assert_eq!(snap.notes_total, notes, "{what}: nothing logged");
        assert!(snap.sampling.is_none() && snap.paused);
    }
    assert!(!path.exists(), "nothing was written");
    // CancelSampling without a pass is a no-op, not a refusal (M5)
    d.apply(DynCommand::CancelSampling);
    assert_eq!(d.snapshot().last_rejection, None);
    // the dynamics are unaffected: Play is accepted
    d.apply(DynCommand::Play);
    assert_eq!(d.snapshot().last_rejection, None);
}

/// Wells at the origin that refuse a third nucleus (`ProviderError::Invalid`): the stale
/// re-evaluation after an `AddNucleus` fails.
struct TwoAtMost {
    inner: OriginWells,
}

impl ForceProvider for TwoAtMost {
    fn evaluate(&mut self, scene: &Scene, a: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        if scene.nuclei.len() > 2 {
            return Err(ProviderError::Invalid("two nuclei at most".into()));
        }
        self.inner.evaluate(scene, a)
    }
    fn name(&self) -> &'static str {
        "two at most"
    }
    fn reset(&mut self) {}
}

/// Review round 1 (M1–M3): the thermostat's at-rest draw is deferred to the end of the relax,
/// `Rewind` is refused while relaxing, and a failed re-evaluation after an edit ends the relax.
#[test]
fn thermostat_rewind_and_a_failed_reevaluation_during_a_relax() {
    let s = scene(vec![Nucleus::new(1, [0.3, 0.0, 0.0]), Nucleus::new(1, [-0.2, 0.1, 0.0])]);
    let mut d = Dynamics::new(s, Box::new(OriginWells { k: 0.3 }), DT_FS, 1);
    let e_c0 = d.snapshot().energy_conserved;
    d.apply(DynCommand::Relax);
    d.tick(-1.0, &mut never());
    // M1: SetThermostat mid-relax is accepted but draws nothing — FIRE's velocities stay its own
    let v_before: Vec<[f64; 3]> = d.snapshot().scene.nuclei.iter().map(|n| n.vel).collect();
    let notes = d.snapshot().notes_total;
    d.apply(DynCommand::SetThermostat(Some((300.0, 20.0))));
    let st = d.snapshot();
    assert_eq!(st.last_rejection, None);
    assert_eq!(st.thermostat, Some((300.0, 20.0)));
    assert_eq!(st.scene.nuclei.iter().map(|n| n.vel).collect::<Vec<_>>(), v_before, "no Maxwell–Boltzmann draw into FIRE");
    assert_eq!(st.notes_total, notes, "no draw note");
    assert!((st.energy_conserved - e_c0).abs() <= 1e-10);
    // M2: Rewind is refused like Play
    d.apply(DynCommand::Rewind);
    let sr = d.snapshot();
    assert_eq!(sr.last_rejection.as_deref(), Some("relaxing: Stop relax first"));
    assert_eq!(sr.scene.nuclei.iter().map(|n| n.vel).collect::<Vec<_>>(), v_before, "velocities not flipped");
    // the relax runs to convergence with the thermostat bypassed (E_conserved flat, no draw)
    let fin = loop {
        let snap = d.tick(-1.0, &mut never());
        assert!((snap.energy_conserved - e_c0).abs() <= 1e-10, "E_conserved moved by {:.2e}", snap.energy_conserved - e_c0);
        if snap.relax.as_ref().unwrap().converged.is_some() {
            break snap;
        }
        assert!(!snap.notes.iter().any(|n| n.contains("Maxwell")), "a draw during the relax: {:?}", snap.notes);
    };
    assert_eq!(fin.relax.clone().unwrap().converged, Some(true));
    // M1, the deferred draw: at the end the resting, thermostatted system gets its thermal
    // velocities (booked: E_conserved still flat), so Play is not frozen at K = 0
    assert!(fin.kinetic > 0.0, "the deferred Maxwell–Boltzmann draw");
    println!("deferred draw at 300 K on 3 dof (seed 1): T = {:.0} K", fin.temperature);
    assert!(fin.notes.iter().any(|n| n.contains("Maxwell–Boltzmann start")), "{:?}", fin.notes);
    assert!((fin.energy_conserved - e_c0).abs() <= 1e-10);
    d.apply(DynCommand::SetThermostat(None));
    assert_eq!(d.snapshot().last_rejection, None);

    // M3: a failed re-evaluation after an edit ends the relax (the error is shown, Play is free)
    let s = scene(vec![Nucleus::new(1, [0.3, 0.0, 0.0]), Nucleus::new(1, [-0.2, 0.1, 0.0])]);
    let mut d = Dynamics::new(s, Box::new(TwoAtMost { inner: OriginWells { k: 0.3 } }), DT_FS, 1);
    d.apply(DynCommand::Relax);
    for _ in 0..3 {
        d.tick(-1.0, &mut never());
    }
    d.apply(DynCommand::AddNucleus(Nucleus::new(1, [0.0, 0.5, 0.0])));
    let sf = d.tick(-1.0, &mut never());
    assert!(sf.last_error.as_deref().is_some_and(|e| e.contains("two nuclei at most")), "{:?}", sf.last_error);
    assert!(sf.forces_stale);
    let rf = sf.relax.clone().unwrap();
    assert_eq!((rf.converged, rf.steps), (Some(false), 3));
    assert!(rf.note.as_deref().is_some_and(|n| n.starts_with("stopped after 3 steps")), "{:?}", rf.note);
    assert!(sf.scene.nuclei.iter().all(|n| n.vel == [0.0; 3]));
    d.apply(DynCommand::Play);
    assert_eq!(d.snapshot().last_rejection, None, "the relax is over: Play is not refused on top of the error");
    d.apply(DynCommand::Pause);
    // removing the third nucleus makes the scene solvable again; a new relax runs
    d.apply(DynCommand::RemoveNucleus(2));
    d.apply(DynCommand::Relax);
    let ok = d.tick(-1.0, &mut never());
    assert_eq!(ok.last_error, None);
    assert_eq!(ok.relax.clone().unwrap().converged, None);
    assert!(!ok.forces_stale);
}

/// M6 Task 3 (spec §3.2, ruling P6; review I2): a species without a reference is solved on the
/// provider's side solver, on its own thread, while the dynamics run on: the stepping provider
/// sees no reset and no extra evaluation (its warm state is untouched), no step is skipped and
/// nothing pauses; when the entry lands the graph is published at once from the stepping
/// provider's density, even while paused. A `ReplaceScene` with a new species queues only that
/// species (the H table is cached) and paused ticks publish the graph without a Play.
#[test]
fn references_are_solved_off_the_stepping_path_without_pausing() {
    use crate::bonds::Verdict;
    use crate::test_providers::Gaussians;
    let s = scene(vec![Nucleus::new(1, [-0.8, 0.0, 0.0]), Nucleus::new(1, [0.8, 0.0, 0.0])]);
    // a stand-in atom on every nucleus and a bump between every pair within 3 Bohr: bonded
    let g = Gaussians::per_atom(0.3, 0.5).with_pair_bump(0.15, 0.5, 3.0);
    let (evals, resets, disc) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let p = Counting { inner: Box::new(g), evals: evals.clone(), resets: resets.clone(), discontinuities: disc };
    let mut d = Dynamics::new(s, Box::new(p), DT_FS, 1);
    let s0 = d.snapshot();
    assert_eq!(s0.bonds_pending, vec![1]);
    assert!(s0.bonds.is_none() && !s0.forces_stale && s0.pair_reports.is_empty());
    d.apply(DynCommand::Play);
    let (r0, e0, n0) = (resets.load(Ordering::SeqCst), evals.load(Ordering::SeqCst), s0.notes_total);
    let t1 = d.tick(f64::INFINITY, &mut never());
    assert!(!t1.paused, "the dynamics do not pause for a reference");
    assert_eq!((t1.step, t1.steps_last_tick), (1, 1), "the reference skips no step");
    let t = settle_references(&mut d);
    assert!(!t.paused && t.step >= 1);
    assert_eq!(resets.load(Ordering::SeqCst) - r0, 0, "the stepping provider is never reset for a reference");
    assert_eq!(evals.load(Ordering::SeqCst) - e0, t.step as usize, "every evaluation of the stepping provider is a step's: the atom was solved elsewhere");
    assert_eq!(t.notes_total, n0 + 1);
    let note = t.notes.last().unwrap().clone();
    println!("{note}");
    assert!(note.contains("atom reference H (Z = 1)") && note.contains("off the stepping path"), "{note}");
    assert!(!t.forces_stale && !t.bonds_stale);
    let g = t.bonds.clone().expect("the graph");
    assert_eq!((g.bonds.len(), g.step, g.n_nuclei), (1, t.step, 2));
    assert_eq!((g.bonds[0].i, g.bonds[0].j), (0, 1));
    assert_eq!(g.components, vec![vec![0, 1]]);
    assert_eq!(t.pair_reports.len(), 1);
    assert_eq!(t.pair_reports[0].verdict, Verdict::Bonded { multi_centre: false });
    println!("H₂ report: {:?}", t.pair_reports[0]);
    // a new species through ReplaceScene (paused): only O is queued; paused ticks publish
    d.apply(DynCommand::Pause);
    let water = scene(vec![Nucleus::new(8, [0.0; 3]), Nucleus::new(1, [1.43, 1.11, 0.0]), Nucleus::new(1, [-1.43, 1.11, 0.0])]);
    d.apply(DynCommand::ReplaceScene(water));
    let s = d.snapshot();
    assert_eq!(s.bonds_pending, vec![8]);
    assert!(s.bonds.is_none() && s.pair_reports.is_empty());
    let (r1, e1) = (resets.load(Ordering::SeqCst), evals.load(Ordering::SeqCst));
    let t = settle_references(&mut d);
    assert!(t.paused && t.step == 0);
    assert_eq!(resets.load(Ordering::SeqCst) - r1, 0);
    assert_eq!(evals.load(Ordering::SeqCst) - e1, 1, "the stale re-evaluation only: O's three multiplicities were solved on the side");
    let g = t.bonds.clone().expect("the graph while paused");
    assert_eq!((g.n_nuclei, g.step), (3, 0));
    assert_eq!(t.pair_reports.len(), 3);
    for r in &t.pair_reports {
        println!("water-like report: {r:?}");
    }
    // the cached H reference and the fresh O one are both held
    let t = d.tick(f64::INFINITY, &mut never());
    assert!(t.bonds_pending.is_empty() && t.bonds.is_some());
}

/// M6 Task 3 (spec §3.4): a step whose evaluation carries no density (a surrogate-answered step)
/// keeps the graph and marks it stale with its step unchanged; a nucleus count change drops the
/// graph until the next evaluation with a density; a published graph never has another nucleus
/// count than the scene.
/// Task 3 fix round 1: a reference that lands after a Pause abandoned a step. The forces are the
/// step's start (fresh) but the provider's last solve was the abandoned trial geometry's, so it
/// holds no density; paused, the driver re-solves the density once at the current positions
/// (`refresh_density`) and publishes the graph — without replacing the kept forces and energy
/// (nothing unbooked in E_conserved) and without a reset. Playing, the next step does it.
#[test]
fn a_reference_landing_after_an_abandoned_step_re_solves_the_density_once() {
    use crate::test_providers::Gaussians;
    use std::sync::atomic::AtomicBool;
    /// Gaussians whose next evaluation is abandoned (`cancel_next`): `Cancelled`, no density left
    /// — what a GPU solve abandoned by a Pause leaves (`GpuKs::density_current` false).
    struct Abandoning {
        inner: Gaussians,
        cancel_next: Arc<AtomicBool>,
        evals: Arc<AtomicUsize>,
        resets: Arc<AtomicUsize>,
    }
    impl ForceProvider for Abandoning {
        fn evaluate(&mut self, scene: &Scene, a: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
            self.evals.fetch_add(1, Ordering::SeqCst);
            if self.cancel_next.swap(false, Ordering::SeqCst) {
                self.inner.reset();
                return Err(ProviderError::Cancelled);
            }
            self.inner.evaluate(scene, a)
        }
        fn name(&self) -> &'static str {
            "gaussians"
        }
        fn reset(&mut self) {
            self.resets.fetch_add(1, Ordering::SeqCst);
            self.inner.reset()
        }
        fn sample_density(&mut self, points: &[[f64; 3]]) -> Option<Vec<f64>> {
            self.inner.sample_density(points)
        }
        fn side_solver(&self) -> Option<Box<dyn ForceProvider>> {
            self.inner.side_solver()
        }
    }
    let s = scene(vec![Nucleus::new(1, [-0.8, 0.0, 0.0]), Nucleus::new(1, [0.8, 0.0, 0.0])]);
    let (cancel, evals, resets) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let p = Abandoning { inner: Gaussians::per_atom(0.3, 0.5).with_pair_bump(0.15, 0.5, 3.0), cancel_next: cancel.clone(), evals: evals.clone(), resets: resets.clone() };
    let mut d = Dynamics::new(s, Box::new(p), DT_FS, 1);
    let s0 = d.snapshot();
    assert_eq!(s0.bonds_pending, vec![1]);
    let (e0, r0) = (evals.load(Ordering::SeqCst), resets.load(Ordering::SeqCst));
    // playing, the first step is abandoned; then the Pause
    d.apply(DynCommand::Play);
    cancel.store(true, Ordering::SeqCst);
    let t = d.tick(f64::INFINITY, &mut never());
    assert_eq!((t.step, t.forces_stale), (0, false), "the abandoned step left the start's forces");
    d.apply(DynCommand::Pause);
    let t = settle_references(&mut d);
    let g = t.bonds.clone().expect("the graph, from the re-solved density");
    assert_eq!((g.bonds.len(), g.step, t.step), (1, 0, 0));
    assert!(!t.bonds_stale && t.paused && !t.forces_stale);
    assert_eq!(evals.load(Ordering::SeqCst) - e0, 2, "the abandoned step and one density re-solve");
    assert_eq!(resets.load(Ordering::SeqCst) - r0, 0, "no reset");
    assert_eq!((t.energy_ks, &t.forces), (s0.energy_ks, &s0.forces), "the kept forces and energy");
    // nothing more while paused
    let t = d.tick(f64::INFINITY, &mut never());
    assert_eq!(evals.load(Ordering::SeqCst) - e0, 2);
    assert!(t.bonds.is_some());
}

#[test]
fn bonds_go_stale_on_surrogate_steps_and_drop_on_count_change() {
    use crate::test_providers::Gaussians;
    use std::sync::atomic::AtomicBool;
    /// Exact Gaussian density; molecular scenes (> 1 nucleus) answer `sample_density` only while
    /// `some` is set (the reference solves always do).
    struct Toggled {
        inner: Gaussians,
        some: Arc<AtomicBool>,
        last_n: usize,
    }
    impl ForceProvider for Toggled {
        fn evaluate(&mut self, scene: &Scene, a: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
            self.last_n = scene.nuclei.len();
            self.inner.evaluate(scene, a)
        }
        fn name(&self) -> &'static str {
            "toggled"
        }
        fn reset(&mut self) {
            self.inner.reset();
        }
        fn sample_density(&mut self, points: &[[f64; 3]]) -> Option<Vec<f64>> {
            if self.last_n <= 1 || self.some.load(Ordering::SeqCst) { self.inner.sample_density(points) } else { None }
        }
        fn side_solver(&self) -> Option<Box<dyn ForceProvider>> {
            self.inner.side_solver()
        }
    }
    let some = Arc::new(AtomicBool::new(true));
    let s = scene(vec![Nucleus::new(1, [-0.8, 0.0, 0.0]), Nucleus::new(1, [0.8, 0.0, 0.0])]);
    let p = Toggled { inner: Gaussians::per_atom(0.3, 0.5).with_pair_bump(0.15, 0.5, 3.0), some: some.clone(), last_n: 0 };
    let mut d = Dynamics::new(s, Box::new(p), DT_FS, 1);
    let check = |s: &DynSnapshot| {
        if let Some(g) = &s.bonds {
            assert_eq!(g.n_nuclei, s.scene.nuclei.len(), "a graph with another nucleus count: {g:?}");
        }
    };
    let t = settle_references(&mut d); // the reference on the side, then the first graph
    check(&t);
    let g0 = t.bonds.clone().expect("graph");
    assert_eq!((g0.step, g0.n_nuclei, g0.bonds.len(), t.bonds_stale), (0, 2, 1, false));
    // a step without a density: stale, the graph unchanged
    some.store(false, Ordering::SeqCst);
    let t = step_n(&mut d, 1);
    check(&t);
    assert_eq!(t.step, 1);
    assert!(t.bonds_stale);
    assert_eq!(t.bonds.as_ref().map(|g| g.step), Some(0));
    // a step with one: fresh
    some.store(true, Ordering::SeqCst);
    let t = step_n(&mut d, 1);
    check(&t);
    assert!(!t.bonds_stale);
    assert_eq!(t.bonds.as_ref().map(|g| g.step), Some(2));
    // a nucleus added: no graph until the next evaluation with a density
    d.apply(DynCommand::AddNucleus(Nucleus::new(1, [0.0, 3.0, 0.0])));
    let s = d.snapshot();
    check(&s);
    assert!(s.bonds.is_none() && !s.bonds_stale && s.pair_reports.is_empty() && s.bonds_pending.is_empty());
    let t = d.tick(f64::INFINITY, &mut never()); // paused: the stale re-evaluation only
    check(&t);
    let g = t.bonds.clone().expect("graph after the re-evaluation");
    assert_eq!((g.n_nuclei, g.step, t.step), (3, 2, 2));
    assert_eq!(t.pair_reports.len(), 3);
    // removed, and the re-evaluation carries no density: still no graph (never a stale one
    // with the old indices)
    d.apply(DynCommand::RemoveNucleus(2));
    check(&d.snapshot());
    assert!(d.snapshot().bonds.is_none());
    some.store(false, Ordering::SeqCst);
    let t = d.tick(f64::INFINITY, &mut never());
    check(&t);
    assert!(t.bonds.is_none() && !t.bonds_stale && !t.forces_stale);
    some.store(true, Ordering::SeqCst);
    let t = step_n(&mut d, 1);
    check(&t);
    let g = t.bonds.clone().expect("graph");
    assert_eq!((g.n_nuclei, g.step, g.bonds.len()), (2, 3, 1));
}
