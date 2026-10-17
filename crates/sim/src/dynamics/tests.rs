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
        Ok(ForceResult { energy, forces: vec![self.f; scene.nuclei.len()], stats: EvalStats::default(), display: None, orbitals: None })
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
        Ok(ForceResult { energy: self.per_call * self.calls as f64, forces: vec![[0.0; 3]; scene.nuclei.len()], stats: EvalStats::default(), display: None, orbitals: None })
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
        Ok(ForceResult { energy: 0.0, forces: vec![[0.0; 3]; scene.nuclei.len()], stats: EvalStats::default(), display: None, orbitals: None })
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

/// The wheel's pull (`SetDragForce`, module doc "Tool forces"): the new cap is the force applied
/// beyond d_c — exactly, on the next step, raised and lowered mid-drag — and it is booked at fixed
/// positions (E_conserved continuous to rounding); it persists through DragEnd into the next drag;
/// it neither stales the forces nor restarts the history (the driver's side of "never aborts").
#[test]
fn drag_force_setting_applies_the_new_cap_and_books_the_change() {
    let s = scene(vec![Nucleus::new(1, [0.0; 3])]);
    let m = s.nuclei[0].mass;
    let mut d = Dynamics::new(s, Box::new(Zero), DT_FS, 1);
    d.drag_damping = 0.0; // the force alone: x₁ = ½ (F/m) dt², Δv = (F/m) dt
    let dt = dt_au();
    d.apply(DynCommand::DragStart { index: 0, target: [0.0, 2.0, 0.0] });
    let hist = step_n(&mut d, 3).history.len();
    // raise: 0.1 → 0.5 (d ≈ 2 > d_c = 1 Bohr)
    let (before, v0) = (d.snapshot(), d.snapshot().scene.nuclei[0].vel[1]);
    d.apply(DynCommand::SetDragForce { f_max: 0.5 });
    let after = d.snapshot();
    assert!(after.last_rejection.is_none() && after.drag_force_max == 0.5);
    assert!((after.energy_conserved - before.energy_conserved).abs() < 1e-14, "booked at fixed r: {} vs {}", after.energy_conserved, before.energy_conserved);
    assert!(!after.forces_stale && after.history.len() == hist, "no re-evaluation, no history restart");
    let one = step_n(&mut d, 1);
    let dv = one.scene.nuclei[0].vel[1] - v0;
    println!("raised to 0.5 Ha/Bohr: Δv = {dv:.6e} vs (F/m) dt = {:.6e}", 0.5 / m * dt);
    assert!((dv - 0.5 / m * dt).abs() < 1e-17, "Δv = {dv:e}");
    // lower: 0.5 → 0.02 (d_c = 0.04 Bohr)
    let (before, v0) = (d.snapshot(), d.snapshot().scene.nuclei[0].vel[1]);
    d.apply(DynCommand::SetDragForce { f_max: 0.02 });
    assert!((d.snapshot().energy_conserved - before.energy_conserved).abs() < 1e-14);
    let dv = step_n(&mut d, 1).scene.nuclei[0].vel[1] - v0;
    assert!((dv - 0.02 / m * dt).abs() < 1e-18, "Δv = {dv:e} vs {:e}", 0.02 / m * dt);
    // the setting outlives the drag: a fresh drag from rest pulls with 0.02 from its first step
    d.apply(DynCommand::DragEnd);
    d.apply(DynCommand::Pause);
    d.apply(DynCommand::MoveNucleusDiscontinuous { index: 0, pos: [0.0; 3] });
    d.tick(f64::INFINITY, &mut never());
    assert_eq!(d.snapshot().drag_force_max, 0.02);
    d.apply(DynCommand::DragStart { index: 0, target: [0.0, 0.0, -3.0] });
    let z = step_n(&mut d, 1).scene.nuclei[0].pos[2];
    assert!((z + 0.5 * 0.02 / m * dt * dt).abs() < 1e-16, "z₁ = {z:e}");
    // without a drag a change books nothing (there is no spring potential)
    d.apply(DynCommand::DragEnd);
    let e = d.snapshot().energy_conserved;
    d.apply(DynCommand::SetDragForce { f_max: F_DRAG_MAX });
    assert_eq!(d.snapshot().energy_conserved, e);
}

/// Review M4: the booking's three regimes at fixed r against the closed forms (module doc):
/// linear → quadratic (a crossing) books (k d − F_old)²/(2k), quadratic → quadratic books 0, back
/// across books the negative, and linear → linear books (F_n − F_o)(d − (F_n + F_o)/(2k)).
#[test]
fn drag_force_booking_matches_the_closed_forms_across_the_corner() {
    let mut d = Dynamics::new(scene(vec![Nucleus::new(1, [0.0; 3])]), Box::new(Zero), DT_FS, 1);
    let dd = 0.5;
    d.apply(DynCommand::DragStart { index: 0, target: [0.0, dd, 0.0] }); // 0.1: d_c = 0.2 < d (linear)
    let mut w = d.work;
    let cross = (K_DRAG * dd - F_DRAG_MAX).powi(2) / (2.0 * K_DRAG);
    d.apply(DynCommand::SetDragForce { f_max: 0.5 }); // d_c = 1 > d: quadratic
    assert!((d.work - w - cross).abs() < 1e-15, "crossing: {} vs {cross}", d.work - w);
    w = d.work;
    d.apply(DynCommand::SetDragForce { f_max: 0.8 }); // quadratic → quadratic
    assert_eq!(d.work, w);
    d.apply(DynCommand::SetDragForce { f_max: F_DRAG_MAX });
    assert!((d.work - w + cross).abs() < 1e-15, "back across: {}", d.work - w);
    d.apply(DynCommand::DragMove { target: [0.0, 3.0, 0.0] });
    w = d.work;
    d.apply(DynCommand::SetDragForce { f_max: 0.6 });
    let lin = (0.6 - F_DRAG_MAX) * (3.0 - (0.6 + F_DRAG_MAX) / (2.0 * K_DRAG));
    assert!((d.work - w - lin).abs() < 1e-15, "linear: {} vs {lin}", d.work - w);
}

/// Out-of-range or non-finite pulls are refused (`last_rejection`) and change nothing; the ends
/// of the range are accepted.
#[test]
fn drag_force_setting_outside_the_range_is_refused() {
    let mut d = Dynamics::new(scene(vec![Nucleus::new(1, [0.0; 3])]), Box::new(Zero), DT_FS, 1);
    d.apply(DynCommand::DragStart { index: 0, target: [1.0, 0.0, 0.0] });
    let e = d.snapshot().energy_conserved;
    for f in [0.0, -0.1, 0.0099, 1.01, f64::NAN, f64::INFINITY] {
        d.apply(DynCommand::SetDragForce { f_max: f });
        let s = d.snapshot();
        assert!(s.last_rejection.as_deref().is_some_and(|r| r.contains("drag force")), "{f}: {:?}", s.last_rejection);
        assert_eq!(s.drag_force_max, F_DRAG_MAX, "{f}");
        assert_eq!(s.energy_conserved, e);
    }
    for f in [F_DRAG_LO, F_DRAG_HI] {
        d.apply(DynCommand::SetDragForce { f_max: f });
        let s = d.snapshot();
        assert!(s.last_rejection.is_none() && s.drag_force_max == f, "{f}");
    }
}

/// The damping residual's per-transient coefficient (module doc, "Tool forces"): in the capped
/// spring's linear regime (a fixed pull direction), a gap e₀ = v_t d̂ − v to the terminal velocity
/// closes as e_n = e₀ (1 − a)^{2n}, a = ω dt, and every step leaves R = −½ m a³(2 − a)|e_n|² in
/// E_conserved (the explicit damping kicks against the trapezoidal booking of their work); the
/// whole transient leaves −c · ½ m |e₀|², c = a³(2 − a)/(1 − (1 − a)⁴) ≈ (a²/2)(1 + a).
fn damping_residual_coefficient(m: f64) -> f64 {
    let a = (K_DRAG / m).sqrt() * dt_au();
    a.powi(3) * (2.0 - a) / (1.0 - (1.0 - a).powi(4))
}

/// Review I1 (fix round 1): the residual a hard pull leaves in E_conserved is the closed form, to
/// 1e-9 relative — for a start from rest and for a full-speed stop (the cap dropped to the floor),
/// at the default pull and at the ceiling; it is one-signed (negative). The arrival at the target
/// adds a phase-dependent term from the Huber corner at d_c (measured over a sweep of distances,
/// bound below), and one hard start keeps the drift readout red over its 2000-step window.
#[test]
fn a_pull_transient_leaves_the_closed_form_damping_residual() {
    let m = Nucleus::new(1, [0.0; 3]).mass;
    let c = damping_residual_coefficient(m);
    let gamma = 2.0 * (K_DRAG * m).sqrt();
    for f in [F_DRAG_MAX, F_DRAG_HI] {
        let mut d = Dynamics::new(scene(vec![Nucleus::new(1, [0.0; 3])]), Box::new(Zero), DT_FS, 1);
        d.apply(DynCommand::SetDragForce { f_max: f });
        // the target far enough that the whole run is in the linear regime (d ≫ d_c)
        d.apply(DynCommand::DragStart { index: 0, target: [0.0, 60.0, 0.0] });
        let e0 = d.snapshot().energy_conserved;
        let start = step_n(&mut d, 150).energy_conserved - e0;
        let k_t = 0.5 * m * (f / gamma).powi(2);
        let want = -c * k_t;
        // the stop: the cap dropped to the floor at (numerically) terminal speed
        let v = d.snapshot().scene.nuclei[0].vel[1];
        d.apply(DynCommand::SetDragForce { f_max: F_DRAG_LO });
        let e1 = d.snapshot().energy_conserved;
        let stop = step_n(&mut d, 150).energy_conserved - e1;
        let want_stop = -c * 0.5 * m * (F_DRAG_LO / gamma - v).powi(2);
        println!("pull {f}: start from rest ΔE_c = {start:.6e} Ha (closed form {want:.6e}, K_t = {k_t:.4e}, c = {c:.5e}); stop to {F_DRAG_LO}: {stop:.6e} (closed form {want_stop:.6e})");
        assert!((start - want).abs() <= 1e-9 * want.abs(), "start: {start:e} vs {want:e}");
        assert!((stop - want_stop).abs() <= 1e-9 * want_stop.abs(), "stop: {stop:e} vs {want_stop:e}");
    }
    // the arrival: from rest at the ceiling to targets D Bohr away, settled; what the Huber corner
    // adds beyond the start's closed form, over D = 3 … 12 Bohr in 0.25 steps
    let k_t = 0.5 * m * (F_DRAG_HI / gamma).powi(2);
    let mut worst = 0.0f64;
    let mut dd = 3.0;
    while dd <= 12.0 {
        let mut d = Dynamics::new(scene(vec![Nucleus::new(1, [0.0; 3])]), Box::new(Zero), DT_FS, 1);
        d.apply(DynCommand::SetDragForce { f_max: F_DRAG_HI });
        d.apply(DynCommand::DragStart { index: 0, target: [0.0, dd, 0.0] });
        let e0 = d.snapshot().energy_conserved;
        let t = step_n(&mut d, 600);
        assert!(t.kinetic < 1e-20, "settled at D = {dd}");
        let arrival = t.energy_conserved - e0 + c * k_t;
        worst = if arrival.abs() > worst.abs() { arrival } else { worst };
        dd += 0.25;
    }
    println!("arrival (Huber corner) term at the ceiling, D = 3…12 Bohr: worst {worst:+.4e} Ha = {:.3} × the start's {:.4e}", worst.abs() / (c * k_t), c * k_t);
    assert!(worst.abs() <= 0.3 * c * k_t, "arrival term {worst:e}");
    // the drift readout over one hard start: 1000 steps at rest, then a drag to 6 Bohr
    for f in [F_DRAG_MAX, F_DRAG_HI] {
        let mut d = Dynamics::new(scene(vec![Nucleus::new(1, [0.0; 3])]), Box::new(Zero), DT_FS, 1);
        d.apply(DynCommand::SetDragForce { f_max: f });
        step_n(&mut d, 1000);
        d.apply(DynCommand::DragStart { index: 0, target: [0.0, 6.0, 0.0] });
        let t = step_n(&mut d, 1000);
        let drift = t.drift_per_ps.unwrap();
        println!("drift readout, 2000-step window with one start at its middle, pull {f}: {drift:+.3e} Ha/ps");
        if f == F_DRAG_HI {
            assert!(drift < -1e-3, "the documented red readout after a hard start: {drift:e}");
        }
    }
}

/// Review I2 (fix round 1): across raises, a mid-flight drop to the floor and a wheel flick
/// (0.01 ↔ 1 every 4 steps, six cycles), every change is booked at fixed r to 1e-14 and
/// E_conserved stays within the sum of the closed-form residuals of the transients the changes
/// start (`damping_residual_coefficient`; a transient cut short by the next change leaves a
/// partial sum, so this is an exact upper bound in the linear regime, asserted with a 1e-9
/// relative margin for rounding). The target is far, so no arrival happens (the arrival term:
/// `a_pull_transient_leaves_the_closed_form_damping_residual`).
#[test]
fn drag_force_changes_stay_within_the_damping_residual_bound() {
    let s = scene(vec![Nucleus::new(1, [0.0; 3])]);
    let m = s.nuclei[0].mass;
    let (c, gamma) = (damping_residual_coefficient(m), 2.0 * (K_DRAG * m).sqrt());
    let mut d = Dynamics::new(s, Box::new(Zero), DT_FS, 1);
    d.apply(DynCommand::DragStart { index: 0, target: [0.0, 60.0, 0.0] });
    let e0 = d.snapshot().energy_conserved;
    // the start from rest at the default pull
    let mut bound = c * 0.5 * m * (F_DRAG_MAX / gamma).powi(2);
    let mut cmax = 0.0f64;
    for i in 1..=300 {
        let f = match i {
            30 => Some(F_DRAG_HI),
            60 => Some(F_DRAG_LO),
            100..=147 if i % 4 == 0 => Some(if (i / 4) % 2 == 1 { F_DRAG_HI } else { F_DRAG_LO }),
            148 => Some(0.3),
            _ => None,
        };
        if let Some(f) = f {
            let before = d.snapshot();
            let v = before.scene.nuclei[0].vel;
            d.apply(DynCommand::SetDragForce { f_max: f });
            assert!((d.snapshot().energy_conserved - before.energy_conserved).abs() < 1e-14, "step {i}: booked at fixed r");
            bound += c * 0.5 * m * ((f / gamma - v[1]).powi(2) + v[0] * v[0] + v[2] * v[2]);
        }
        let t = step_n(&mut d, 1);
        let (j, tgt) = t.drag.unwrap();
        assert!(dist(tgt, t.scene.nuclei[j].pos) > F_DRAG_HI / K_DRAG, "linear regime throughout");
        cmax = cmax.max((t.energy_conserved - e0).abs());
    }
    println!("pull changes (raise, drop, 6-cycle flick): max |ΔE_conserved| = {cmax:.4e} Ha, closed-form bound {bound:.4e} (ratio {:.3})", cmax / bound);
    assert!(cmax <= bound * (1.0 + 1e-9), "E_conserved moved {cmax:.4e} > {bound:.4e}");
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
        Ok(ForceResult { energy: 0.0, forces: vec![[0.0; 3]; scene.nuclei.len()], stats, display: None, orbitals: None })
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
        Ok(ForceResult { energy: 0.0, forces: vec![[0.0; 3]; scene.nuclei.len()], stats: EvalStats::default(), display: None, orbitals: None })
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
        Ok(ForceResult { energy, forces, stats: EvalStats::default(), display: None, orbitals: None })
    }
    fn name(&self) -> &'static str {
        "origin wells"
    }
    fn reset(&mut self) {}
}

/// The external field is part of the force FIRE relaxes (final review M3): one ¹H in a well
/// k = 0.3 under a = 1e-4 a.u. along −y relaxes to the shifted minimum y = −m a/k (≈ −0.61 Bohr),
/// not to the well's centre where the field keeps pushing; and an isotope swap in that field
/// books ΔU_field, so E_conserved does not move.
#[test]
fn relax_includes_the_external_field_and_set_isotope_books_its_energy() {
    let s = scene(vec![Nucleus::new(1, [0.0, 0.0, 0.0])]);
    let mut d = Dynamics::new(s, Box::new(OriginWells { k: 0.3 }), DT_FS, 1);
    let a = 1e-4;
    d.apply(DynCommand::SetExternalField([0.0, -a, 0.0]));
    d.apply(DynCommand::Relax);
    let mut last = d.snapshot();
    for _ in 0..2000 {
        last = d.tick(-1.0, &mut never());
        if last.relax.as_ref().is_some_and(|r| r.converged.is_some()) {
            break;
        }
    }
    let r = last.relax.clone().unwrap();
    assert_eq!(r.converged, Some(true), "{r:?}");
    let m = last.scene.nuclei[0].mass;
    let y_min = -m * a / 0.3;
    assert!(y_min < -0.6, "{y_min}");
    let y = last.scene.nuclei[0].pos[1];
    assert!((y - y_min).abs() <= 1e-3 / 0.3, "relaxed to y = {y}, the field's minimum is {y_min}");
    // an isotope swap there: U_field = −m a_ext·r changes by (m′ − m)·a·y ≈ −0.11 Ha, booked
    let before = d.snapshot();
    d.apply(DynCommand::SetIsotope { index: 0, a: 2 });
    let after = d.snapshot();
    assert_eq!(after.last_rejection, None);
    let du = (after.scene.nuclei[0].mass - m) * a * y;
    assert!(du < -0.1, "an unbooked ΔU_field would move E_conserved by {du}");
    assert!((after.energy_conserved - before.energy_conserved).abs() < 1e-12, "E_conserved moved by {}", after.energy_conserved - before.energy_conserved);
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

/// Merge of main's drag pull (the wheel) with the relax tool: `SetDragForce` mid-relax is
/// accepted, booked at fixed positions, and restarts FIRE (Δt, α, stall window) with a drag on;
/// every FIRE step books the spring at the current pull, so E_conserved stays constant to
/// rounding; the relax converges where the well balances the spring capped at the new pull.
#[test]
fn a_drag_pull_change_during_a_relax_is_applied_booked_and_restarts_fire() {
    // one nucleus in a k = 0.3 well at the origin, dragged toward x = 3: at either pull the
    // balance x = F/k (1/3 at 0.1, 5/3 at 0.5) is beyond d_c = F/K_DRAG of the target, so the
    // spring is a constant force F there and the relax must end at x = F/k
    let k = 0.3;
    // (it starts at x = −2, far from either balance, so FIRE has grown its Δt before the change)
    let mut d = Dynamics::new(scene(vec![Nucleus::new(1, [-2.0, 0.0, 0.0])]), Box::new(OriginWells { k }), DT_FS, 1);
    d.apply(DynCommand::DragStart { index: 0, target: [3.0, 0.0, 0.0] });
    d.apply(DynCommand::Relax);
    let e_c0 = d.snapshot().energy_conserved;
    let r = loop {
        let s = d.tick(-1.0, &mut never());
        assert!((s.energy_conserved - e_c0).abs() <= 1e-10, "E_conserved moved by {:.2e}", s.energy_conserved - e_c0);
        let r = s.relax.clone().unwrap();
        assert!(r.converged.is_none(), "converged before Δt grew: {r:?}");
        if r.dt_fs > 0.25 {
            break r;
        }
    };
    assert!(r.alpha < 0.1, "{r:?}");
    let steps = r.steps;
    let (w0, u0) = (d.work, d.spring_energy());
    d.apply(DynCommand::SetDragForce { f_max: 0.5 });
    let s = d.snapshot();
    assert_eq!(s.last_rejection, None, "accepted mid-relax");
    assert_eq!(s.drag_force_max, 0.5);
    assert!((d.work - w0 - (d.spring_energy() - u0)).abs() <= 1e-15, "booked at fixed positions");
    assert!((s.energy_conserved - e_c0).abs() <= 1e-10);
    let r = s.relax.clone().unwrap();
    assert_eq!((r.steps, r.converged, r.alpha), (steps, None, 0.1), "FIRE restarted");
    assert!((r.dt_fs - 0.25).abs() < 1e-12, "Δt back at the start: {}", r.dt_fs);
    // the augmented force now holds the new cap: 0.5 − k x
    let x = s.scene.nuclei[0].pos[0];
    assert!((r.max_force_augmented - (0.5 - k * x).abs()).abs() < 1e-12, "{} vs {}", r.max_force_augmented, 0.5 - k * x);
    let mut worst = (s.energy_conserved - e_c0).abs();
    let fin = loop {
        let s = d.tick(-1.0, &mut never());
        worst = worst.max((s.energy_conserved - e_c0).abs());
        assert!((s.energy_conserved - e_c0).abs() <= 1e-10, "E_conserved moved by {:.2e}", s.energy_conserved - e_c0);
        if s.relax.as_ref().unwrap().converged.is_some() {
            break s;
        }
        assert!(s.relax.as_ref().unwrap().steps < 2000);
    };
    assert_eq!(fin.relax.clone().unwrap().converged, Some(true));
    let x = fin.scene.nuclei[0].pos;
    println!(
        "pull 0.1 → 0.5 after {steps} FIRE steps: converged after {} steps at x = {:.6} (F/k = {:.6}); max |ΔE_conserved| {worst:.1e} Ha",
        fin.relax.as_ref().unwrap().steps,
        x[0],
        0.5 / k
    );
    assert!((x[0] - 0.5 / k).abs() <= 1e-3 / k && x[1].abs() < 1e-9 && x[2].abs() < 1e-9, "{x:?}");
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

/// Counting wrapper with fresh counters: (provider, evaluations, resets).
fn counting(inner: Box<dyn ForceProvider>) -> (Box<dyn ForceProvider>, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let (evals, resets) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let p = Counting { inner, evals: evals.clone(), resets: resets.clone(), discontinuities: Arc::new(AtomicUsize::new(0)) };
    (Box::new(p), evals, resets)
}

/// Every readout of an empty-scene snapshot is defined and finite (module doc, "The empty
/// scene"): E = K = 0, T = 0, no forces, no density, no drift, nothing stale.
fn assert_empty_readouts(s: &DynSnapshot) {
    assert!(s.scene.nuclei.is_empty());
    assert!(!s.forces_stale && s.forces.is_empty() && s.display.is_none(), "{:?}", s.last_error);
    assert_eq!((s.energy_ks, s.kinetic, s.total, s.temperature), (0.0, 0.0, 0.0, 0.0));
    assert!(s.energy_conserved.is_finite() && s.wall_ratio == 0.0 && s.steps_last_tick == 0);
    assert!(s.history.iter().all(|&(t, e, temp)| t.is_finite() && e.is_finite() && temp.is_finite()));
    assert!(s.drift_per_ps.is_none_or(f64::is_finite));
    assert_eq!(s.eval.scf_steps, 0);
}

/// The empty scene (module doc): the driver never asks the provider about it; Play is accepted
/// but time stands still (no step, no history, queued steps dropped); adding the first atom is
/// one evaluation, removing it again is none; a single atom then steps as before.
#[test]
fn an_empty_scene_never_asks_the_provider_and_time_stands_still() {
    let (p, evals, resets) = counting(Box::new(Zero));
    let mut d = Dynamics::new(scene(Vec::new()), p, DT_FS, 1);
    assert_eq!(evals.load(Ordering::SeqCst), 0, "no evaluation of an empty scene");
    let s0 = d.snapshot();
    assert_empty_readouts(&s0);
    assert!(s0.last_error.is_none());

    // Play is accepted; ticks run nothing and time does not advance
    d.apply(DynCommand::SetSpeed(50));
    d.apply(DynCommand::Play);
    assert!(d.snapshot().last_rejection.is_none());
    for _ in 0..3 {
        let s = d.tick(f64::INFINITY, &mut never());
        assert!(!s.paused, "Play stays on");
        assert_eq!((s.step, s.sim_time, s.history.len()), (0, 0.0, 0));
        assert_empty_readouts(&s);
    }
    // the thermostat and the tools have nothing to act on
    d.apply(DynCommand::SetThermostat(Some((300.0, 10.0))));
    d.apply(DynCommand::SetComCorrection(true));
    assert_empty_readouts(&d.tick(f64::INFINITY, &mut never()));
    // queued steps are dropped
    let s = step_n(&mut d, 5);
    assert!(s.paused);
    assert_eq!((s.step, s.pending_steps), (0, 0));
    assert_empty_readouts(&s);
    assert_eq!(evals.load(Ordering::SeqCst), 0);
    // a relax is refused (final review M2: FIRE's first step used to ask the provider)
    d.apply(DynCommand::Relax);
    let s = d.tick(f64::INFINITY, &mut never());
    assert_eq!(s.last_rejection.as_deref(), Some(crate::relax::EMPTY_SCENE));
    assert!(s.relax.is_none(), "{:?}", s.relax);
    assert_eq!(evals.load(Ordering::SeqCst), 0);
    d.apply(DynCommand::SetThermostat(None));

    // the first atom: the usual edit path, one evaluation
    d.apply(DynCommand::AddNucleus(Nucleus::new(1, [0.3, 0.0, 0.0])));
    let s = d.tick(f64::INFINITY, &mut never());
    assert_eq!(evals.load(Ordering::SeqCst), 1, "one solve once there is something to solve");
    assert_eq!((s.scene.nuclei.len(), s.forces.len(), s.forces_stale), (1, 1, false));
    // a single atom steps as before
    let s = step_n(&mut d, 2);
    assert_eq!(s.step, 2);
    assert!(s.sim_time > 0.0);
    let (evals_before, t_before) = (evals.load(Ordering::SeqCst), s.sim_time);

    // removing the last atom: back to empty, no evaluation, and time stands still again
    let resets_before = resets.load(Ordering::SeqCst);
    d.apply(DynCommand::RemoveNucleus(0));
    assert_eq!(resets.load(Ordering::SeqCst), resets_before + 1, "the edit still resets the provider");
    let s = d.tick(f64::INFINITY, &mut never());
    assert_empty_readouts(&s);
    d.apply(DynCommand::Play);
    let s = d.tick(f64::INFINITY, &mut never());
    assert_eq!((s.step, s.sim_time), (2, t_before));
    assert_eq!(evals.load(Ordering::SeqCst), evals_before, "no evaluation after removing the last atom");
    assert!(s.last_error.is_none());

    // a replaced scene (paused): empty → H₂ is one evaluation, H₂ → empty none
    d.apply(DynCommand::Pause);
    d.apply(DynCommand::ReplaceScene(scene(vec![Nucleus::new(1, [0.7, 0.0, 0.0]), Nucleus::new(1, [-0.7, 0.0, 0.0])])));
    d.tick(f64::INFINITY, &mut never());
    assert_eq!(evals.load(Ordering::SeqCst), evals_before + 1);
    d.apply(DynCommand::ReplaceScene(scene(Vec::new())));
    let s = d.tick(f64::INFINITY, &mut never());
    assert_empty_readouts(&s);
    assert_eq!((s.step, s.sim_time), (0, 0.0), "a replaced scene restarts time");
    assert_eq!(evals.load(Ordering::SeqCst), evals_before + 1);
}

/// An empty scene with a charge cannot be posed: the evaluation fails like any other (paused,
/// `last_error`), still without asking the provider.
#[test]
fn a_charged_empty_scene_is_an_error_without_a_solve() {
    let (p, evals, _) = counting(Box::new(Zero));
    let mut s = scene(Vec::new());
    s.charge = -1;
    let mut d = Dynamics::new(s, p, DT_FS, 1);
    let snap = d.snapshot();
    assert!(snap.last_error.as_deref().is_some_and(|e| e.contains("no nuclei")), "{:?}", snap.last_error);
    assert!(snap.forces_stale);
    d.apply(DynCommand::Play);
    let snap = d.tick(f64::INFINITY, &mut never());
    assert!(snap.paused && snap.last_error.is_some());
    assert_eq!(evals.load(Ordering::SeqCst), 0);
}

/// A trainable stand-in for the surrogate: a [`Harmonic`] whose every evaluation queues one
/// fallback event (tagged with the call number in `n_atoms`), and whose queue the test can add to
/// directly (events the provider queues outside an evaluation). The evaluation with call number
/// `fail_at` queues its event and then fails. Only what the driver calls is implemented.
struct Teaching {
    inner: Harmonic,
    queue: Arc<std::sync::Mutex<Vec<crate::activity::TeachEvent>>>,
    calls: usize,
    fail_at: Option<usize>,
}

impl ForceProvider for Teaching {
    fn evaluate(&mut self, scene: &Scene, a: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        use crate::activity::{TeachEvent, TeachOrigin, TeachOutcome, TeachRule};
        let call = self.calls;
        self.calls += 1;
        let failed = self.fail_at == Some(call);
        let outcome = if failed { TeachOutcome::Failed } else { TeachOutcome::Kept };
        let mut e = TeachEvent::new(TeachOrigin::Fallback, TeachRule::Grade, outcome);
        e.n_atoms = call;
        self.queue.lock().unwrap().push(e);
        if failed {
            return Err(ProviderError::Invalid("teaching: the planned failure".into()));
        }
        self.inner.evaluate(scene, a)
    }
    fn name(&self) -> &'static str {
        "teaching"
    }
    fn reset(&mut self) {}
    fn as_trainable(&mut self) -> Option<&mut dyn crate::provider::Trainable> {
        Some(self)
    }
}

impl crate::provider::Trainable for Teaching {
    fn start_pass(&mut self, _s: &Scene, _p: &SamplingPlan, _r: &mut AtomReferences) -> Result<(), String> {
        Err("teaching: no passes".into())
    }
    fn pass_step(&mut self, _a: &mut dyn FnMut() -> bool) -> Result<Option<crate::sampling::SamplingProgress>, ProviderError> {
        Ok(None)
    }
    fn cancel_pass(&mut self) {}
    fn pass_active(&self) -> bool {
        false
    }
    fn progress(&self) -> Option<crate::sampling::SamplingProgress> {
        None
    }
    fn refit(&mut self) -> Result<String, String> {
        Err("teaching: no model".into())
    }
    fn forget(&mut self) {}
    fn save(&self, _p: &std::path::Path) -> std::io::Result<()> {
        Err(std::io::ErrorKind::InvalidInput.into())
    }
    fn load(&mut self, _p: &std::path::Path, _s: &Scene) -> Result<(), String> {
        Err("teaching: no model".into())
    }
    fn teacher_mut(&mut self) -> &mut dyn ForceProvider {
        &mut self.inner
    }
    fn status(&self, _s: &Scene, _f: &[[f64; 3]]) -> crate::provider::TrainableStatus {
        unimplemented!("the driver does not read the status")
    }
    fn fitted(&self) -> bool {
        false
    }
    fn samples(&self) -> usize {
        0
    }
    fn pending_refit(&self) -> bool {
        false
    }
    fn set_audit_every(&mut self, _n: usize) {}
    fn force_teacher_next(&mut self) {}
    fn take_teach_events(&mut self) -> Vec<crate::activity::TeachEvent> {
        std::mem::take(&mut *self.queue.lock().unwrap())
    }
}

/// Activity Task 1b (plan review M6): the driver stamps each teaching event with the step whose
/// evaluation produced it: the constructor's at step 0, an integrator step's at the step it
/// lands on, a failed evaluation's too; an event queued outside an evaluation is stamped at the
/// current step when the worker drains, and a pass event stays unstamped.
#[test]
fn the_driver_stamps_teach_events_with_the_step_of_their_evaluation() {
    use crate::activity::{TeachEvent, TeachOrigin, TeachOutcome, TeachRule};
    let (sc, h) = harmonic_three();
    let queue = Arc::new(std::sync::Mutex::new(Vec::new()));
    let p = Teaching { inner: h, queue: queue.clone(), calls: 0, fail_at: Some(4) };
    let mut d = Dynamics::new(sc, Box::new(p), DT_FS, 1);
    step_n(&mut d, 3);
    let got: Vec<(usize, Option<u64>)> = d.take_teach_events().into_iter().map(|e| (e.n_atoms, e.step)).collect();
    assert_eq!(got, vec![(0, Some(0)), (1, Some(1)), (2, Some(2)), (3, Some(3))]);
    assert!(d.take_teach_events().is_empty(), "a take drains the driver's buffer");

    // queued between ticks: a pass solve keeps None, anything else gets the current step
    queue.lock().unwrap().push(TeachEvent::new(TeachOrigin::Pass, TeachRule::NoModel, TeachOutcome::Kept));
    queue.lock().unwrap().push(TeachEvent::new(TeachOrigin::Audit, TeachRule::Audit, TeachOutcome::Kept));
    let got: Vec<(TeachOrigin, Option<u64>)> = d.take_teach_events().into_iter().map(|e| (e.origin, e.step)).collect();
    assert_eq!(got, vec![(TeachOrigin::Pass, None), (TeachOrigin::Audit, Some(3))]);

    // the fourth evaluation fails: its event is stamped with the step it was evaluated for
    let snap = step_n(&mut d, 1);
    assert_eq!(snap.step, 3, "the failed step did not land");
    let got: Vec<(usize, Option<u64>, TeachOutcome)> = d.take_teach_events().into_iter().map(|e| (e.n_atoms, e.step, e.outcome)).collect();
    assert_eq!(got, vec![(4, Some(4), TeachOutcome::Failed)]);
    assert_eq!(d.teach_events_dropped(), 0);
}

/// Activity Task 1b review round 1: a FIRE step's evaluations are collected right after the
/// step, stamped with the step the relax runs at (a relax does not advance it), so none is left
/// in the provider's queue for a later MD step to mis-stamp.
#[test]
fn relax_step_teach_events_are_collected_and_stamped_with_the_current_step() {
    let (sc, h) = harmonic_three();
    let queue = Arc::new(std::sync::Mutex::new(Vec::new()));
    let p = Teaching { inner: h, queue: queue.clone(), calls: 0, fail_at: None };
    let mut d = Dynamics::new(sc, Box::new(p), DT_FS, 1);
    step_n(&mut d, 2);
    assert_eq!(d.take_teach_events().len(), 3, "the constructor's and two steps'");
    d.apply(DynCommand::Relax);
    for _ in 0..3 {
        d.tick(-1.0, &mut never());
        assert!(queue.lock().unwrap().is_empty(), "the relax step's events stayed in the provider's queue");
    }
    d.apply(DynCommand::StopRelax);
    // no drain between the relax and the next MD step: each keeps its own stamp
    step_n(&mut d, 1);
    let got: Vec<(usize, Option<u64>)> = d.take_teach_events().into_iter().map(|e| (e.n_atoms, e.step)).collect();
    let (relax, md) = got.split_at(got.len() - 1);
    assert!(relax.len() >= 3, "one evaluation per FIRE step at least: {got:?}");
    assert!(relax.iter().enumerate().all(|(i, &(call, step))| call == 3 + i && step == Some(2)), "{got:?}");
    assert_eq!(md, &[(3 + relax.len(), Some(3))]);
}

/// Task 1b review round 1 (minor 1): `ReplaceScene` resets the step to 0, so an event still in
/// the provider's queue is collected first, at the old scene's step, not stamped 0 later.
#[test]
fn replace_scene_collects_pending_teach_events_at_the_old_step() {
    use crate::activity::{TeachEvent, TeachOrigin, TeachOutcome, TeachRule};
    let (sc, h) = harmonic_three();
    let queue = Arc::new(std::sync::Mutex::new(Vec::new()));
    let p = Teaching { inner: h, queue: queue.clone(), calls: 0, fail_at: None };
    let mut d = Dynamics::new(sc.clone(), Box::new(p), DT_FS, 1);
    step_n(&mut d, 5);
    queue.lock().unwrap().push(TeachEvent::new(TeachOrigin::Audit, TeachRule::Audit, TeachOutcome::Kept));
    d.apply(DynCommand::ReplaceScene(sc));
    assert_eq!(d.snapshot().last_rejection, None);
    let audit: Vec<Option<u64>> = d.take_teach_events().into_iter().filter(|e| e.origin == TeachOrigin::Audit).map(|e| e.step).collect();
    assert_eq!(audit, vec![Some(5)]);
}

// ---- the orbital display (module doc, "Orbital display"; orbital-viewer plan Task 3) ----

/// What [`Orbiting`] saw and made: every request forwarded to it, its payloads (by evaluation and
/// by `orbital_display_now`) and the `orbital_display_now` calls.
#[derive(Default)]
struct OrbitLog {
    requests: Vec<crate::provider::OrbitalViewRequest>,
    payloads: usize,
    now_calls: usize,
    /// Rate mode: an evaluation makes a payload only when the forwarded request is forced (the
    /// provider's rate bound skipping every play step — the worst case for Task 3 review M1).
    rate: bool,
    /// Side displays to hand out (`take_side_display`, oldest first) and the jobs dispatched.
    side: std::collections::VecDeque<crate::provider::SideDisplay>,
    jobs: u64,
    /// What each `orbital_display_now`'s abort said when polled, and whether the driver installed
    /// a display interrupt (Task 4 review I2).
    aborts_seen: Vec<bool>,
    interrupt_installed: bool,
    /// The next this many payloads with unoccupied orbitals say their eigensolve was interrupted.
    interrupt_next: usize,
    /// The next this many payloads with unoccupied orbitals say their rows are an earlier
    /// geometry's (`current: false`, not interrupted): the provider's play cap at work.
    stale_next: usize,
}

/// A harmonic provider with a stand-in orbital producer: while a view is on, every evaluation and
/// every `orbital_display_now` makes a payload whose one row's track is the payload count (the
/// rate rule is the provider's, `providers::ks`; this one makes a payload whenever asked, so the
/// driver's forwarding is what is observed).
struct Orbiting {
    inner: Harmonic,
    view: crate::provider::OrbitalViewRequest,
    log: Arc<std::sync::Mutex<OrbitLog>>,
    /// The positions of the last evaluation (the converged state `orbital_display_now` re-reads).
    last: Vec<[f64; 3]>,
}

impl Orbiting {
    fn payload(&mut self, positions: Vec<[f64; 3]>, cell: Cell) -> Option<crate::provider::OrbitalDisplay> {
        use crate::provider::*;
        if self.view.what.is_off() {
            return None;
        }
        let mut log = self.log.lock().unwrap();
        log.payloads += 1;
        let row = OrbitalInfo {
            spin: Spin::Up, index: 0, track: log.payloads as u32, slot: 0, energy: -0.5, eps_range: (-0.5, -0.5), occupation: 1.0, label: OrbitalLabel::Index(1),
            cluster: None, rotated: false, grid_split: None, box_state: false, smeared_1s: false, residual: None, not_converged: false,
        };
        // a view with unoccupied orbitals: a status, saying "interrupted" for the next
        // `interrupt_next` payloads (Task 4 re-review N2)
        let m = self.view.unoccupied;
        let unoccupied = (m > 0).then(|| {
            let interrupted = log.interrupt_next > 0;
            log.interrupt_next = log.interrupt_next.saturating_sub(1);
            let stale = log.stale_next > 0;
            log.stale_next = log.stale_next.saturating_sub(1);
            UnoccupiedStatus { requested: m, interrupted, held: m, current: !interrupted && !stale, ..Default::default() }
        });
        Some(OrbitalDisplay {
            cell, origin: [0.0; 3], positions, rows: vec![row], field: None, spin: SpinSummary::default(),
            unoccupied, notes: Vec::new(), what: self.view.what.clone(), lone_atom: LoneAtomLabels::NotLoneAtom,
        })
    }
}

impl ForceProvider for Orbiting {
    fn evaluate(&mut self, scene: &Scene, a: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        let mut r = self.inner.evaluate(scene, a)?;
        self.last = scene.nuclei.iter().map(|n| n.pos).collect();
        let skip = self.log.lock().unwrap().rate && !self.view.force_every_step;
        if !skip {
            r.orbitals = self.payload(self.last.clone(), scene.cell.clone());
        }
        Ok(r)
    }
    fn name(&self) -> &'static str {
        "orbiting"
    }
    fn side_jobs(&self) -> u64 {
        self.log.lock().unwrap().jobs
    }
    fn take_side_display(&mut self) -> Option<crate::provider::SideDisplay> {
        self.log.lock().unwrap().side.pop_front()
    }
    fn set_display_interrupt(&mut self, _interrupt: crate::provider::DisplayInterrupt) {
        self.log.lock().unwrap().interrupt_installed = true;
    }
    fn reset(&mut self) {}
    fn set_orbital_view(&mut self, request: crate::provider::OrbitalViewRequest) {
        self.log.lock().unwrap().requests.push(request.clone());
        self.view = request;
    }
    fn orbital_display_now(&mut self, should_abort: &mut dyn FnMut() -> bool) -> Option<crate::provider::OrbitalDisplay> {
        let abort = should_abort();
        self.log.lock().unwrap().aborts_seen.push(abort);
        self.log.lock().unwrap().now_calls += 1;
        self.payload(self.last.clone(), Cell { n: 33, h: 0.25 })
    }
}

fn orbiting() -> (Dynamics, Arc<std::sync::Mutex<OrbitLog>>) {
    let (s, h) = harmonic_three();
    let log = Arc::new(std::sync::Mutex::new(OrbitLog::default()));
    let p = Orbiting { inner: h, view: Default::default(), log: log.clone(), last: Vec::new() };
    (Dynamics::new(s, Box::new(p), DT_FS, 1), log)
}

/// The driver forwards the app's orbital view to the provider with `force_every_step` set while
/// paused or single-stepping and cleared while playing (the rate bound is the provider's: plan
/// Task 3); a changed view while paused is shown at once through `orbital_display_now` (no
/// solve); every payload is published with its geometry's time and a new sequence number; the
/// same view again (or only the app's rate flag different) asks for nothing; `Off` drops the
/// payload.
#[test]
fn the_orbital_view_reaches_the_provider_with_the_rate_flag_and_payloads_are_published() {
    use crate::provider::{OrbitalRequest, OrbitalViewRequest};
    let (mut d, log) = orbiting();
    let s0 = d.snapshot();
    assert!(s0.orbitals.is_none() && s0.orbital_view.what.is_off());
    assert!(log.lock().unwrap().requests.is_empty(), "nothing forwarded while the view is off");
    // switched on while paused: forwarded with the rate flag, shown at once without a solve
    let list = OrbitalViewRequest::new(OrbitalRequest::List);
    d.apply(DynCommand::SetOrbitalView(list.clone()));
    let s1 = d.snapshot();
    {
        let l = log.lock().unwrap();
        assert_eq!(l.requests.last().map(|r| (r.what.clone(), r.force_every_step)), Some((OrbitalRequest::List, true)), "paused: every evaluation carries one");
        assert_eq!((l.now_calls, l.payloads), (1, 1));
    }
    assert_eq!(s1.orbital_view, list, "the snapshot keeps the app's request (rate flag as asked)");
    assert!(s1.orbitals.is_some() && s1.orbitals_seq > s0.orbitals_seq);
    assert_eq!((s1.orbitals_t, s1.orbitals_side), (s1.sim_time, None));
    // a paused single step: its evaluation's payload, stamped with the new time
    let s2 = step_n(&mut d, 1);
    assert_eq!(log.lock().unwrap().payloads, 2);
    assert!(s2.orbitals_seq > s1.orbitals_seq);
    assert_eq!(s2.orbitals_t, s2.sim_time);
    assert_eq!(s2.orbitals.as_ref().unwrap().positions, s2.scene.nuclei.iter().map(|n| n.pos).collect::<Vec<_>>());
    // the same view again: nothing asked, nothing re-forwarded
    let before = log.lock().unwrap().requests.len();
    d.apply(DynCommand::SetOrbitalView(list.clone()));
    assert_eq!(log.lock().unwrap().requests.len(), before);
    assert_eq!(log.lock().unwrap().now_calls, 1);
    // playing: the provider's rate bound applies (flag cleared at the next tick)
    let s3 = play_n(&mut d, 3);
    assert_eq!(s3.steps_last_tick, 3);
    let flags: Vec<bool> = log.lock().unwrap().requests[before..].iter().map(|r| r.force_every_step).collect();
    assert_eq!(flags, [false], "cleared once for the play tick");
    // paused again: set at the next tick
    let _ = d.tick(f64::INFINITY, &mut never());
    assert!(log.lock().unwrap().requests.last().unwrap().force_every_step);
    // the app's own force_every_step holds while playing
    let forced = OrbitalViewRequest { force_every_step: true, ..list.clone() };
    d.apply(DynCommand::SetOrbitalView(forced));
    assert_eq!(log.lock().unwrap().now_calls, 1, "only the rate flag changed: no re-display");
    let _ = play_n(&mut d, 2);
    assert!(log.lock().unwrap().requests.last().unwrap().force_every_step);
    // a new mode while paused: shown at once
    let spin = OrbitalViewRequest::new(OrbitalRequest::SpinDensity);
    d.apply(DynCommand::SetOrbitalView(spin));
    assert_eq!(log.lock().unwrap().now_calls, 2);
    assert_eq!(d.snapshot().orbitals.unwrap().what, OrbitalRequest::SpinDensity);
    // off: forwarded, the payload dropped, and no more payloads
    let s4 = d.snapshot();
    d.apply(DynCommand::SetOrbitalView(OrbitalViewRequest::default()));
    let s5 = d.snapshot();
    assert!(s5.orbitals.is_none() && s5.orbitals_seq > s4.orbitals_seq);
    assert!(log.lock().unwrap().requests.last().unwrap().what.is_off());
    let p = log.lock().unwrap().payloads;
    let _ = step_n(&mut d, 2);
    assert_eq!(log.lock().unwrap().payloads, p);
    assert!(d.snapshot().orbitals.is_none());
}

/// Stale forces (an edit) are not a converged state at the current positions: a view switched on
/// then waits for the re-evaluation instead of asking `orbital_display_now`; an empty scene has
/// no orbitals (the last payload is dropped, not shown over nothing).
#[test]
fn the_orbital_view_waits_for_fresh_forces_and_an_empty_scene_drops_the_payload() {
    use crate::provider::{OrbitalRequest, OrbitalViewRequest};
    let (mut d, log) = orbiting();
    d.apply(DynCommand::MoveNucleusDiscontinuous { index: 2, pos: [-1.2, 1.0, 0.1] });
    d.apply(DynCommand::SetOrbitalView(OrbitalViewRequest::new(OrbitalRequest::List)));
    assert_eq!(log.lock().unwrap().now_calls, 0, "stale forces: no display without a solve");
    assert!(d.snapshot().orbitals.is_none());
    let s = d.tick(f64::INFINITY, &mut never());
    assert!(s.orbitals.as_ref().is_some_and(|o| o.positions[2] == [-1.2, 1.0, 0.1]), "the re-evaluation's payload, at the new positions");
    let mut empty = s.scene.clone();
    empty.nuclei.clear();
    d.apply(DynCommand::ReplaceScene(empty));
    let s = d.tick(f64::INFINITY, &mut never());
    assert!(s.scene.nuclei.is_empty());
    assert!(s.orbitals.is_none(), "no electrons: nothing to show");
}

/// Task 3 review M1: during play the provider's rate bound can skip every step's payload, so the
/// newest one is an earlier geometry's; pausing (the forced flag set again, fresh forces) makes the
/// driver re-read the converged state once through `orbital_display_now` — the snapshot then shows
/// the current positions' orbitals. A payload already at the current positions asks for nothing,
/// and neither does a pause with a step queued (the step brings its own payload).
#[test]
fn pausing_shows_the_orbitals_of_the_current_positions() {
    use crate::provider::{OrbitalRequest, OrbitalViewRequest};
    let (mut d, log) = orbiting();
    log.lock().unwrap().rate = true;
    d.apply(DynCommand::SetOrbitalView(OrbitalViewRequest::new(OrbitalRequest::List)));
    assert_eq!(log.lock().unwrap().now_calls, 1, "switched on while paused");
    let s = play_n(&mut d, 3);
    let here = |s: &DynSnapshot| s.scene.nuclei.iter().map(|n| n.pos).collect::<Vec<_>>();
    assert_eq!(s.steps_last_tick, 3);
    assert_ne!(s.orbitals.as_ref().unwrap().positions, here(&s), "play: every payload skipped, the newest is the start's");
    assert_eq!(log.lock().unwrap().payloads, 1);
    let s = d.tick(f64::INFINITY, &mut never());
    assert!(s.paused && s.steps_last_tick == 0);
    assert_eq!(log.lock().unwrap().now_calls, 2, "paused into a stale payload: re-read once");
    assert_eq!(s.orbitals.as_ref().unwrap().positions, here(&s));
    assert_eq!(s.orbitals_t, s.sim_time);
    let _ = d.tick(f64::INFINITY, &mut never());
    assert_eq!(log.lock().unwrap().now_calls, 2, "nothing more while it stays paused");
    // a paused single step carries its own (forced) payload; play then skips again, and pausing
    // re-reads again
    let _ = step_n(&mut d, 1);
    assert_eq!(log.lock().unwrap().now_calls, 2);
    let _ = play_n(&mut d, 2);
    let s = d.tick(f64::INFINITY, &mut never());
    assert_eq!(log.lock().unwrap().now_calls, 3);
    assert_eq!(s.orbitals.as_ref().unwrap().positions, here(&s));
    // play, then Pause and a Step queued together: the step's own payload is the current one —
    // no re-read before it (one payload for one paused step)
    let _ = play_n(&mut d, 2);
    let payloads = log.lock().unwrap().payloads;
    let s = step_n(&mut d, 1);
    let l = log.lock().unwrap();
    assert_eq!((l.now_calls, l.payloads - payloads), (3, 1), "the step's payload only");
    drop(l);
    assert_eq!(s.orbitals.as_ref().unwrap().positions, here(&s));
}

/// Final review I2: play at a grid whose unoccupied eigensolve is paused-only carries, at every
/// step, a payload at the current positions whose unoccupied rows are an earlier geometry's. The
/// pause then re-reads the converged state (`orbital_display_now`) though the payload is at the
/// current positions — the provider's forced plan solves the rows now — and not again while it
/// stays paused. (Without it the tab said "updated when paused" while the rows stayed stale.)
#[test]
fn pausing_re_reads_unoccupied_rows_that_are_an_earlier_geometrys() {
    use crate::provider::{OrbitalRequest, OrbitalViewRequest};
    let (mut d, log) = orbiting();
    d.apply(DynCommand::SetOrbitalView(OrbitalViewRequest { what: OrbitalRequest::List, unoccupied: 4, force_every_step: false }));
    let calls = || log.lock().unwrap().now_calls;
    assert_eq!(calls(), 1, "switched on while paused");
    log.lock().unwrap().stale_next = usize::MAX; // every play payload: held rows, not current
    let s = play_n(&mut d, 3);
    let here = |s: &DynSnapshot| s.scene.nuclei.iter().map(|n| n.pos).collect::<Vec<_>>();
    let o = s.orbitals.as_ref().unwrap();
    assert_eq!(o.positions, here(&s), "every play step carried a payload at the current positions");
    assert!(!o.unoccupied.as_ref().unwrap().current, "…with stale unoccupied rows");
    log.lock().unwrap().stale_next = 0; // the forced plan solves
    let s = d.tick(f64::INFINITY, &mut never());
    assert!(s.paused);
    assert_eq!(calls(), 2, "paused into stale rows at the current positions: re-read once");
    assert!(s.orbitals.as_ref().unwrap().unoccupied.as_ref().unwrap().current, "the rows are current now");
    let _ = d.tick(f64::INFINITY, &mut never());
    assert_eq!(calls(), 2, "nothing more while it stays paused");
}

/// Task 3 review M2: a side job's orbital payload (estimated mode's producer) replaces the shown
/// one only if it answers the current view and is not from an older job — an audit in flight
/// across a view change, or a held audit judged after a newer refresh, never snaps the view back.
/// A selected orbital may differ only by the selection fallback (the requested track absent).
#[test]
fn a_stale_side_payload_never_replaces_the_shown_one() {
    use crate::provider::*;
    let (mut d, log) = orbiting();
    let side = |job: u64, what: OrbitalRequest, tracks: &[u32], unocc: u8| -> SideDisplay {
        let rows = tracks.iter().enumerate().map(|(i, &t)| OrbitalInfo {
            spin: Spin::Up, index: i, track: t, slot: 0, energy: -0.5, eps_range: (-0.5, -0.5), occupation: 1.0, label: OrbitalLabel::Index(i + 1),
            cluster: None, rotated: false, grid_split: None, box_state: false, smeared_1s: false, residual: None, not_converged: false,
        }).collect();
        let unoccupied = (unocc > 0).then(|| UnoccupiedStatus { requested: unocc, ..Default::default() });
        let o = OrbitalDisplay {
            cell: Cell { n: 33, h: 0.25 }, origin: [0.0; 3], positions: Vec::new(), rows, field: None, spin: SpinSummary::default(),
            unoccupied, notes: Vec::new(), what, lone_atom: LoneAtomLabels::NotLoneAtom,
        };
        let display = Display::Cpu { cell: Cell { n: 33, h: 0.25 }, origin: [0.0; 3], rho: Arc::new(Vec::new()), max: 0.0 };
        SideDisplay { display, orbitals: Some(o), kind: SideKind::Audit, job, positions: Vec::new() }
    };
    let shown = |d: &Dynamics| d.snapshot().orbitals.map(|o| o.rows.iter().map(|r| r.track).collect::<Vec<_>>());
    let feed = |d: &mut Dynamics, sd: SideDisplay| {
        {
            let mut l = log.lock().unwrap();
            l.jobs = l.jobs.max(sd.job);
            l.side.push_back(sd);
        }
        let _ = d.tick(f64::INFINITY, &mut never());
    };
    d.apply(DynCommand::SetOrbitalView(OrbitalViewRequest::new(OrbitalRequest::SpinDensity)));
    feed(&mut d, side(2, OrbitalRequest::SpinDensity, &[102], 0));
    assert_eq!(shown(&d), Some(vec![102]), "the current view's");
    feed(&mut d, side(1, OrbitalRequest::SpinDensity, &[101], 0));
    assert_eq!(shown(&d), Some(vec![102]), "an older job's never replaces it");
    feed(&mut d, side(3, OrbitalRequest::List, &[103], 0));
    assert_eq!(shown(&d), Some(vec![102]), "another view's (an audit in flight across the change)");
    feed(&mut d, side(4, OrbitalRequest::SpinDensity, &[104], 2));
    assert_eq!(shown(&d), Some(vec![102]), "another unoccupied count");
    feed(&mut d, side(5, OrbitalRequest::SpinDensity, &[105], 0));
    assert_eq!((shown(&d), d.snapshot().orbitals_side), (Some(vec![105]), Some(SideKind::Audit)));
    let sel = |track: u32| OrbitalRequest::Orbital { spin: Spin::Up, track, squared: false };
    d.apply(DynCommand::SetOrbitalView(OrbitalViewRequest::new(sel(50))));
    feed(&mut d, side(6, sel(51), &[50, 51], 0));
    assert_ne!(shown(&d), Some(vec![50, 51]), "another selection while the requested track exists: stale");
    feed(&mut d, side(7, sel(51), &[51, 52], 0));
    assert_eq!(shown(&d), Some(vec![51, 52]), "the selection fallback (track 50 gone)");
    feed(&mut d, side(8, OrbitalRequest::Orbital { spin: Spin::Up, track: 50, squared: true }, &[50], 0));
    assert_eq!(shown(&d), Some(vec![51, 52]), "|ψ|² is not the phase view");
}

/// Task 4 review I2: the driver hands its display interrupt to the provider (for display work
/// inside evaluations) and passes it as `orbital_display_now`'s abort — both on a view change and
/// on the re-read when pausing (M1). Without one, the abort says false.
#[test]
fn the_display_interrupt_reaches_the_provider_and_orbital_display_now() {
    use crate::provider::{OrbitalRequest, OrbitalViewRequest};
    use std::sync::atomic::AtomicBool;
    let (mut d, log) = orbiting();
    d.apply(DynCommand::SetOrbitalView(OrbitalViewRequest::new(OrbitalRequest::List)));
    assert_eq!(log.lock().unwrap().aborts_seen, [false], "no interrupt installed: never aborted");
    let pending = Arc::new(AtomicBool::new(true));
    let p = pending.clone();
    d.set_display_interrupt(Arc::new(move || p.load(Ordering::SeqCst)));
    assert!(log.lock().unwrap().interrupt_installed, "installed on the provider");
    d.apply(DynCommand::SetOrbitalView(OrbitalViewRequest::new(OrbitalRequest::SpinDensity)));
    assert_eq!(log.lock().unwrap().aborts_seen, [false, true], "a command waiting: the view change's display is told to yield");
    pending.store(false, Ordering::SeqCst);
    log.lock().unwrap().rate = true;
    let _ = play_n(&mut d, 2);
    let _ = d.tick(f64::INFINITY, &mut never());
    assert_eq!(log.lock().unwrap().aborts_seen, [false, true, false], "the re-read on pausing gets it too");
}

/// Task 4 re-review N2: an unoccupied eigensolve that yielded to a newer command leaves the rows
/// stamped `interrupted`; once that command is applied and the worker is idle (paused, nothing
/// queued, the interrupt not raised) the next tick re-reads the converged state once. While a
/// command is still waiting it does not (the retry would yield at its first poll), a payload that
/// is not interrupted asks for nothing more, and neither does a view without unoccupied orbitals.
#[test]
fn an_interrupted_unoccupied_solve_is_retried_when_idle() {
    use crate::provider::{OrbitalRequest, OrbitalViewRequest};
    use std::sync::atomic::AtomicBool;
    let (mut d, log) = orbiting();
    let pending = Arc::new(AtomicBool::new(true));
    let p = pending.clone();
    d.set_display_interrupt(Arc::new(move || p.load(Ordering::SeqCst)));
    log.lock().unwrap().interrupt_next = 1;
    let view = OrbitalViewRequest { unoccupied: 2, ..OrbitalViewRequest::new(OrbitalRequest::List) };
    d.apply(DynCommand::SetOrbitalView(view));
    let s = d.snapshot();
    assert!(s.orbitals.as_ref().unwrap().unoccupied.as_ref().unwrap().interrupted, "the view change's solve yielded");
    assert_eq!(log.lock().unwrap().now_calls, 1);
    // the command that interrupted it is still waiting: no retry yet
    let _ = d.tick(f64::INFINITY, &mut never());
    assert_eq!(log.lock().unwrap().now_calls, 1, "a command waiting: not retried");
    // applied, idle: retried once, and the rows are current
    pending.store(false, Ordering::SeqCst);
    let s = d.tick(f64::INFINITY, &mut never());
    assert_eq!(log.lock().unwrap().now_calls, 2, "idle: retried");
    let u = s.orbitals.as_ref().unwrap().unoccupied.clone().unwrap();
    assert!(!u.interrupted && u.current);
    assert_eq!(s.orbitals_t, s.sim_time);
    let _ = d.tick(f64::INFINITY, &mut never());
    assert_eq!(log.lock().unwrap().now_calls, 2, "not interrupted: nothing more");
    // a paused step whose solve yields to a waiting command: retried at the first idle tick after it
    log.lock().unwrap().interrupt_next = 1;
    pending.store(true, Ordering::SeqCst);
    let s = step_n(&mut d, 1);
    assert!(s.orbitals.as_ref().unwrap().unoccupied.as_ref().unwrap().interrupted, "the step's solve yielded");
    assert_eq!(log.lock().unwrap().now_calls, 2, "the command still waiting: not retried in the step's tick");
    pending.store(false, Ordering::SeqCst);
    let s = d.tick(f64::INFINITY, &mut never());
    assert!(!s.orbitals.as_ref().unwrap().unoccupied.as_ref().unwrap().interrupted, "retried after the step");
    assert_eq!(log.lock().unwrap().now_calls, 3);
    // playing: never retried by the driver (the provider's cadence decides in play)
    log.lock().unwrap().interrupt_next = usize::MAX;
    let _ = play_n(&mut d, 2);
    let calls = log.lock().unwrap().now_calls;
    d.apply(DynCommand::Play);
    let _ = d.tick(f64::INFINITY, &mut never());
    assert_eq!(log.lock().unwrap().now_calls, calls, "in play: no retry");
}
