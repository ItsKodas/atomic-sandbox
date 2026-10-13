//! Velocity Verlet for the classical nuclei: kick–drift–kick, fixed `dt`, atomic units
//! (CLAUDE.md rule 3: `dt` in ħ/E_h = 0.024188843 fs, converted once from fs by
//! [`Integrator::new_fs`]); all nuclear state f64 on the CPU (rule 4). The time *speed* of the
//! simulation is the number of steps per frame and never touches `dt` (rule 8) — this module
//! does not know about frames at all.
//!
//! # The scheme
//! One step of length dt, with a(r) = F(r)/m + a_ext the acceleration of a nucleus:
//!
//! ```text
//! v(t + dt/2) = v(t)        + (dt/2) a(r(t))          half-kick with the OLD forces
//! r(t + dt)   = r(t)        +  dt    v(t + dt/2)      drift
//!               F(r(t + dt))                          one force evaluation per step
//! v(t + dt)   = v(t + dt/2) + (dt/2) a(r(t + dt))     half-kick with the NEW forces
//! ```
//!
//! Positions and velocities are synchronised at the end of every step, so the kinetic and the
//! potential energy are available at the same time without extra work; the one evaluation per
//! step is the DFT solve, the entire cost. The external field `scene.external_field` is a
//! uniform acceleration added inside every kick (it is not a force on the electrons; the
//! providers ignore it). Everything else that is not the Born-Oppenheimer force — drag spring,
//! centre-of-mass net-force correction — enters through the caller's `augment` closure, which
//! [`Integrator::step`] applies to the raw provider forces before *each* kick, so a tool force
//! is treated exactly like a physical one and the step stays pure and testable.
//!
//! # Why this integrator
//! Write K(τ) for the kick v ↦ v + τ a(r) and D(τ) for the drift r ↦ r + τ v; one step is
//! Φ_dt = K(dt/2) ∘ D(dt) ∘ K(dt/2).
//!
//! - **Symplectic.** K(τ) is the exact flow of the potential part H_V = V(r) (+ the field's
//!   −m a_ext·r) for time τ, and D(τ) the exact flow of the kinetic part H_T = p²/2m. Exact
//!   Hamiltonian flows are symplectic and compositions of symplectic maps are symplectic, so Φ_dt
//!   is. Consequence (backward error analysis): Φ_dt is the exact flow of a shadow Hamiltonian
//!   H̃ = H + dt² H₂ + O(dt⁴), hence the energy error is a *bounded oscillation* of size O(dt²)
//!   with no secular drift over exponentially long times — the property the sandbox's energy
//!   readout relies on. For a harmonic mode the shadow energy is exact and closed-form,
//!   Ẽ = ½ m v² + ½ k (1 − h²/4) |r − r₀|² with h = ω dt (derived below); the tests conserve it
//!   to ~1e-13 over 10⁴ steps, which is a far sharper pin on the implementation than a bound on
//!   the true energy.
//! - **Time-reversible.** With R: v ↦ −v, R D(τ) R = D(−τ) and R K(τ) R = K(−τ) (a(r) does not
//!   depend on v), so R Φ_dt R = K(−dt/2) D(−dt) K(−dt/2) = Φ_{−dt} = Φ_dt⁻¹ because the
//!   composition is symmetric (palindromic). Hence Φ_dt R Φ_dt R = id: flipping the velocities,
//!   stepping, and flipping again undoes a step, and n steps → flip → n steps → flip returns to
//!   the start up to rounding — that is [`Integrator::flip_velocities`], the rewind. It needs
//!   forces that depend on positions only; a thermostat or a moving drag target breaks it
//!   (documented by the driver).
//! - **Order.** The symmetric splitting cancels the O(dt²) term of the Baker–Campbell–Hausdorff
//!   expansion, leaving a local error O(dt³) in r and v and a global error O(dt²) over a fixed
//!   time — the same positions as Störmer–Verlet / leapfrog, with velocities on the grid.
//!
//! # Stability and accuracy on a harmonic mode (h = ω dt)
//! For ẍ = −ω² x one step is the linear map (x, v) ↦ M (x, v) with
//! M = [[1 − h²/2, dt], [−ω² dt (1 − h²/4), 1 − h²/2]], det M = 1, tr M = 2 − h². The step is
//! stable (|eigenvalues| = 1, |tr M| ≤ 2) iff **h < 2**; above it the mode blows up. Inside, the
//! eigenvalues are e^{±i ω̃ dt} with cos(ω̃ dt) = 1 − h²/2, so the numerical frequency is
//! ω̃ = ω (1 + h²/24 + O(h⁴)): the oscillator runs slightly fast, with a relative period error
//! of h²/24 (1.0e-4 at h = 0.05, 1.7e-3 at h = 0.2, 7e-3 at h = 0.4). The quadratic form
//! Ẽ = ½ v² + ½ ω² (1 − h²/4) x² is exactly invariant under M (check: Mᵀ diag(ω²(1 − h²/4), 1) M
//! reproduces the matrix), so the true energy E = Ẽ + (h²/8) ω² x² oscillates between Ẽ (at the
//! crossings) and Ẽ/(1 − h²/4) (at the turning points): a peak-to-peak swing of **h²/4** of the
//! mode's energy, ±h²/8 about its mean, and nothing more — no drift. Numbers: h = 0.05 →
//! 6.3e-4; h = 0.2 → 1.0e-2; h = 0.4 → 4.3e-2 (relative to the *mode's* energy, which is
//! ~1e-3 Ha for a molecular vibration, so ~1e-5 Ha in absolute terms at h = 0.2). The practical
//! rule is therefore h ≲ 0.2: frequencies right to ~2e-3, energy fluctuations ~1 % of the
//! vibrational energy and bounded, a 10× margin to the instability. (A "1e-3 energy accuracy"
//! needs h ≈ 0.06 if it means the fluctuation of the mode energy; h = 0.2 gives 1e-3 for the
//! *frequency*. The driver's drift diagnostic, not this fluctuation, is the quantity that
//! reveals a broken force–energy pair in BO-MD.)
//!
//! # Which mode sets dt: the H–X stretch
//! The fastest nuclear motion in any molecule is a hydrogen stretch, ~4400 cm⁻¹ for H₂
//! (4401 cm⁻¹ experimental; O–H ~3700, C–H ~3000 cm⁻¹). Converting, with
//! c = 2.99792458e10 cm s⁻¹:
//!
//! ```text
//! ν  = c ν̃ = 2.99792458e10 cm/s × 4400 cm⁻¹ = 1.3191e14 s⁻¹
//! T  = 1/ν  = 7.581e-15 s = 7.58 fs
//! ω  = 2π ν = 8.288e14 s⁻¹ = 0.8288 fs⁻¹   (= 0.8288 × 0.0241888 = 0.02005 per a.u. of time)
//! h = ω dt = 0.2  ⇒  dt = 0.2 / 0.8288 fs⁻¹ = 0.241 fs  (= 9.98 a.u.)
//! ```
//!
//! The spec's default **dt = 0.25 fs** gives h = 0.207 for a 4400 cm⁻¹ mode — right at the
//! edge of the h ≲ 0.2 rule (period error 1.8e-3, energy swing 1.1 % of the mode energy, 30 steps
//! per period), which is the usual MD compromise: every heavier mode is slower and better
//! resolved, and halving dt doubles the number of DFT solves per femtosecond. At **0.5 fs**,
//! h = 0.41: period error 0.7 %, energy swing 4 % of the mode energy, 15 steps per period, still
//! 5× below the h = 2 blow-up at dt = 2.41 fs — usable for heavy atoms, visibly rough for
//! hydrogen. That is where the UI warns (the driver's `SetDt` > 0.5 fs warning string, spec
//! §8): the warning marks the loss of accuracy for hydrogen, not yet the loss of stability.
//! There is no per-element dt — a single dt must resolve the fastest mode present.
//!
//! # Precision and failure
//! f64 throughout; the kinetic energy and temperature come from [`Scene`] (Neumaier-compensated
//! sums). `step` works on a **clone** of the scene and swaps it in only after the second kick:
//! if the provider fails (cancelled, no convergence, invalid scene) the caller's scene is
//! bit-for-bit what it was, with no restore path to get wrong; a clone of N nuclei (~64 bytes
//! each) is nothing next to the DFT solve the step just paid for.
use crate::provider::{Display, EvalStats, ForceProvider, ProviderError};
use crate::scene::Scene;

/// The fixed time step in atomic units (ħ/E_h ≈ 0.0242 fs). Build it from femtoseconds with
/// [`Integrator::new_fs`]; the UI's speed control changes steps per frame, never this.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Integrator {
    pub dt: f64,
}

/// The energy bookkeeping of one step, evaluated at the new, synchronised (r, v):
/// `energy_ks` the provider's energy (E_KS incl. nuclear repulsion for the DFT providers),
/// `kinetic` = `Scene::kinetic_energy` (lab frame), `total` = their sum (the conserved quantity
/// of NVE), `temperature` = `Scene::temperature` (K, COM removed, equipartition over
/// `Scene::dof`), `eval` the evaluation's cost/diagnostics, `display` its density payload for
/// the viewer (carried through so the integrator does not drop it on the floor).
#[derive(Clone, Debug)]
pub struct StepStats {
    pub energy_ks: f64,
    pub kinetic: f64,
    pub total: f64,
    pub temperature: f64,
    pub eval: EvalStats,
    pub display: Option<Display>,
}

impl Integrator {
    /// `dt` in atomic units of time.
    pub fn new(dt: f64) -> Integrator {
        Integrator { dt }
    }

    /// `dt_fs` in femtoseconds, converted once via `units::fs_to_au_time` (÷ 0.024188843).
    pub fn new_fs(dt_fs: f64) -> Integrator {
        Integrator { dt: units::fs_to_au_time(dt_fs) }
    }

    /// The step in femtoseconds (display only).
    pub fn dt_fs(&self) -> f64 {
        units::au_time_to_fs(self.dt)
    }

    /// v += (dt/2)·(F/m + a_ext) for every nucleus, a_ext = `scene.external_field`.
    /// `forces` is in `scene.nuclei` order (Hartree/Bohr); its length must match.
    pub fn half_kick(scene: &mut Scene, forces: &[[f64; 3]], dt: f64) {
        assert_eq!(forces.len(), scene.nuclei.len(), "half_kick: {} forces for {} nuclei", forces.len(), scene.nuclei.len());
        let half = 0.5 * dt;
        let a_ext = scene.external_field;
        for (n, f) in scene.nuclei.iter_mut().zip(forces) {
            for a in 0..3 {
                n.vel[a] += half * (f[a] / n.mass + a_ext[a]);
            }
        }
    }

    /// r += dt·v for every nucleus.
    pub fn drift(scene: &mut Scene, dt: f64) {
        for n in &mut scene.nuclei {
            for a in 0..3 {
                n.pos[a] += dt * n.vel[a];
            }
        }
    }

    /// One velocity-Verlet step from the forces at the current positions (module doc):
    /// `augment(scene, F_old) → half-kick → drift → provider.evaluate → augment(scene, F_new)
    /// → half-kick`. `forces_old` are the RAW provider forces at the current positions (one per
    /// nucleus — panics otherwise: after an edit that changes the nucleus count the driver must
    /// re-evaluate before stepping); `augment` sees the scene *at the positions the forces belong
    /// to* (old for the first kick,
    /// new for the second) and may add anything (drag spring, COM correction, pins' zeroing);
    /// the external field is not its job (it is applied inside the kicks). Returns the RAW
    /// provider forces at the new positions — the next step's `forces_old` — and the stats.
    ///
    /// On `Err` (cancelled, solver failure, invalid scene, or a provider returning the wrong
    /// number of forces → `ProviderError::Invalid`) the scene is untouched: the step runs on a
    /// clone that is swapped in only on success.
    pub fn step(
        &self,
        scene: &mut Scene,
        forces_old: &[[f64; 3]],
        provider: &mut dyn ForceProvider,
        augment: &mut dyn FnMut(&Scene, &mut Vec<[f64; 3]>),
        should_abort: &mut dyn FnMut() -> bool,
    ) -> Result<(Vec<[f64; 3]>, StepStats), ProviderError> {
        let n = scene.nuclei.len();
        assert_eq!(forces_old.len(), n, "step: {} forces for {n} nuclei", forces_old.len());
        // the step runs on a clone; `scene` is written only after everything succeeded
        let mut work = scene.clone();
        let mut f = forces_old.to_vec();
        augment(&work, &mut f);
        Self::half_kick(&mut work, &f, self.dt);
        Self::drift(&mut work, self.dt);
        let r = provider.evaluate(&work, should_abort)?;
        if r.forces.len() != n {
            return Err(ProviderError::Invalid(format!("provider '{}' returned {} forces for {n} nuclei", provider.name(), r.forces.len())));
        }
        let mut f_new = r.forces.clone();
        augment(&work, &mut f_new);
        Self::half_kick(&mut work, &f_new, self.dt);
        let kinetic = work.kinetic_energy();
        let stats = StepStats {
            energy_ks: r.energy,
            kinetic,
            total: r.energy + kinetic,
            temperature: work.temperature(),
            eval: r.stats,
            display: r.display,
        };
        *scene = work;
        Ok((r.forces, stats))
    }

    /// v ↦ −v for every nucleus: the rewind. Verlet is time-reversible (module doc), so the
    /// next steps retrace the trajectory with the same `forces_old` (forces depend on positions
    /// only, which did not change).
    pub fn flip_velocities(scene: &mut Scene) {
        for n in &mut scene.nuclei {
            for v in &mut n.vel {
                *v = -*v;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{Cell, Nucleus};
    use crate::test_providers::{FailKind, Failing, Harmonic, Zero};
    use std::f64::consts::PI;

    fn scene(nuclei: Vec<Nucleus>) -> Scene {
        Scene { nuclei, cell: Cell { n: 33, h: 0.25 }, charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3] }
    }

    fn never() -> impl FnMut() -> bool { || false }
    fn no_augment() -> impl FnMut(&Scene, &mut Vec<[f64; 3]>) { |_, _| {} }

    fn forces_of(p: &mut dyn ForceProvider, s: &Scene) -> Vec<[f64; 3]> {
        p.evaluate(s, &mut never()).expect("initial forces").forces
    }

    fn max_abs_diff(a: &Scene, b: &Scene) -> (f64, f64) {
        let mut dr: f64 = 0.0;
        let mut dv: f64 = 0.0;
        for (x, y) in a.nuclei.iter().zip(&b.nuclei) {
            for k in 0..3 {
                dr = dr.max((x.pos[k] - y.pos[k]).abs());
                dv = dv.max((x.vel[k] - y.vel[k]).abs());
            }
        }
        (dr, dv)
    }

    #[test]
    fn new_fs_converts_through_units() {
        let i = Integrator::new_fs(0.25);
        // 1 a.u. of time = 0.024188843265857 fs ⇒ 0.25 fs = 10.3353 a.u.
        assert!((i.dt - 10.33536).abs() < 1e-4, "dt = {} a.u.", i.dt);
        assert!((i.dt - 0.25 / units::AU_TIME_IN_FS).abs() < 1e-15);
        assert!((i.dt_fs() - 0.25).abs() < 1e-15);
        assert_eq!(Integrator::new(2.0).dt, 2.0);
    }

    #[test]
    fn half_kick_and_drift_are_the_formulas() {
        let mut s = scene(vec![Nucleus::new(1, [0.0; 3]).with_vel([1e-3, 0.0, 0.0]), Nucleus::new(8, [1.0, 0.0, 0.0])]);
        s.external_field = [0.0, 0.0, -2e-6];
        let (m1, m8) = (s.nuclei[0].mass, s.nuclei[1].mass);
        let f = [[0.01, -0.02, 0.03], [-0.01, 0.02, -0.03]];
        let dt = 4.0;
        Integrator::half_kick(&mut s, &f, dt);
        // v += (dt/2)(F/m + a_ext), per nucleus with its own mass
        assert_eq!(s.nuclei[0].vel, [1e-3 + 2.0 * 0.01 / m1, 2.0 * -0.02 / m1, 2.0 * (0.03 / m1 - 2e-6)]);
        assert_eq!(s.nuclei[1].vel, [2.0 * -0.01 / m8, 2.0 * 0.02 / m8, 2.0 * (-0.03 / m8 - 2e-6)]);
        let v0 = s.nuclei[0].vel;
        let v1 = s.nuclei[1].vel;
        Integrator::drift(&mut s, dt);
        assert_eq!(s.nuclei[0].pos, [dt * v0[0], dt * v0[1], dt * v0[2]]);
        assert_eq!(s.nuclei[1].pos, [1.0 + dt * v1[0], dt * v1[1], dt * v1[2]]);
        Integrator::flip_velocities(&mut s);
        assert_eq!(s.nuclei[0].vel, [-v0[0], -v0[1], -v0[2]]);
        assert_eq!(s.nuclei[1].vel, [-v1[0], -v1[1], -v1[2]]);
    }

    /// One H in a k = 0.5 well, released from rest at x₀ = 0.5 Bohr, h = ω dt = 0.05, 10⁴ steps
    /// (≈ 80 periods). Pins (module doc, "Stability and accuracy"): the shadow energy
    /// Ẽ = ½mv² + ½k(1 − h²/4)x² is conserved to rounding; the true energy's deviation from E₀
    /// is exactly h²/4 (turning-point start; the brief's "< 2e-4" is unattainable at this h,
    /// see the task report); the period from the zero crossings is 2π/ω̃ with
    /// cos(ω̃dt) = 1 − h²/2 (within 0.1 % of 2π/ω as the brief asks, and within 1e-6 of ω̃);
    /// and the period in fs through `new_fs` is 2π/ω × 0.0241888 fs (the dt-units pin).
    #[test]
    fn harmonic_energy_is_bounded_and_period_correct() {
        let k = 0.5;
        let x0 = 0.5;
        let mut s = scene(vec![Nucleus::new(1, [x0, 0.0, 0.0])]);
        let m = s.nuclei[0].mass;
        let omega = (k / m).sqrt();
        let h = 0.05;
        let dt = h / omega;
        let integ = Integrator::new_fs(units::au_time_to_fs(dt));
        assert!((integ.dt - dt).abs() <= 1e-14 * dt, "fs round trip: {} vs {dt}", integ.dt);
        let mut p = Harmonic { k, centres: vec![[0.0; 3]] };
        let mut f = forces_of(&mut p, &s);
        let e0 = 0.5 * k * x0 * x0;
        let shadow = |s: &Scene| s.kinetic_energy() + 0.5 * k * (1.0 - h * h / 4.0) * s.nuclei[0].pos[0].powi(2);
        let shadow0 = shadow(&s);
        let n = 10_000;
        let (mut max_de, mut max_dshadow) = (0.0f64, 0.0f64);
        let mut crossings: Vec<f64> = Vec::new(); // times (a.u.) of x = 0 by linear interpolation
        let mut prev_x = x0;
        for i in 1..=n {
            let (f_new, st) = integ.step(&mut s, &f, &mut p, &mut no_augment(), &mut never()).expect("harmonic never fails");
            f = f_new;
            let x = s.nuclei[0].pos[0];
            // stats are the scene's own numbers
            assert_eq!(st.kinetic, s.kinetic_energy());
            assert_eq!(st.total, st.energy_ks + st.kinetic);
            assert_eq!(st.temperature, s.temperature());
            assert_eq!(st.energy_ks, 0.5 * k * x * x);
            max_de = max_de.max((st.total - e0).abs() / e0);
            max_dshadow = max_dshadow.max((shadow(&s) - shadow0).abs() / e0);
            if (prev_x > 0.0) != (x > 0.0) {
                let t_prev = (i - 1) as f64 * dt;
                crossings.push(t_prev + dt * prev_x / (prev_x - x));
            }
            prev_x = x;
        }
        let period = 2.0 * (crossings.last().unwrap() - crossings[0]) / (crossings.len() - 1) as f64;
        let period_exact = 2.0 * PI / omega;
        let omega_verlet = (1.0 - h * h / 2.0).acos() / dt;
        let period_verlet = 2.0 * PI / omega_verlet;
        let period_fs = period / dt * integ.dt_fs();
        println!("H in k = 0.5: ω = {omega:.6e} a.u., dt = {dt:.4} a.u. = {:.5} fs, {} crossings", integ.dt_fs(), crossings.len());
        println!("  max |ΔE|/E₀ = {max_de:.4e} (h²/4 = {:.4e}); max |ΔẼ|/E₀ = {max_dshadow:.3e}", h * h / 4.0);
        println!("  period: measured {period:.6} a.u., 2π/ω = {period_exact:.6}, 2π/ω̃ = {period_verlet:.6}; rel to exact {:+.3e}, to ω̃ {:+.3e}",
                 period / period_exact - 1.0, period / period_verlet - 1.0);
        println!("  period in fs: {period_fs:.5} vs 2π/ω × 0.0241888 = {:.5}", period_exact * 0.024188843265857);
        // the exact Verlet swing: E ∈ [Ẽ, Ẽ/(1 − h²/4)] with E₀ at the top ⇒ max deviation h²/4
        // (6.25e-4 here; measured 6.2500e-4). The brief's "< 2e-4" cannot hold at h = 0.05 — it
        // would need h < 0.028 — so the closed form is pinned instead (task-3 report).
        let swing = h * h / 4.0;
        assert!((max_de / swing - 1.0).abs() < 1e-2, "max |ΔE|/E₀ = {max_de:.4e} vs h²/4 = {swing:.4e}");
        assert!(max_de < 7e-4);
        assert!(max_dshadow < 1e-10, "shadow energy not conserved: {max_dshadow:.3e}");
        assert!((period / period_exact - 1.0).abs() < 1e-3, "period {period} vs {period_exact}");
        assert!((period / period_verlet - 1.0).abs() < 1e-6, "period {period} vs Verlet's {period_verlet}");
        assert!((period_fs / (period_exact * 0.02419) - 1.0).abs() < 1e-3, "period {period_fs} fs vs {} fs", period_exact * 0.02419);
    }

    /// Two H nuclei in offset wells with mixed velocities: 100 steps, flip, 100 steps, flip →
    /// start, to rounding (module doc: Φ R Φ R = id).
    #[test]
    fn verlet_is_time_reversible() {
        let k = 0.5;
        let start = scene(vec![Nucleus::new(1, [0.3, -0.2, 0.1]).with_vel([2e-3, 1e-3, -5e-4]),
                               Nucleus::new(1, [-0.4, 0.5, -0.1]).with_vel([-1e-3, 3e-3, 2e-3])]);
        let omega = (k / start.nuclei[0].mass).sqrt();
        let integ = Integrator::new(0.1 / omega);
        let mut p = Harmonic { k, centres: vec![[0.0; 3], [0.1, 0.1, 0.1]] };
        let mut s = start.clone();
        let mut f = forces_of(&mut p, &s);
        for _ in 0..100 {
            f = integ.step(&mut s, &f, &mut p, &mut no_augment(), &mut never()).unwrap().0;
        }
        let (dr_far, _) = max_abs_diff(&s, &start);
        assert!(dr_far > 0.1, "test assumption: the system moved ({dr_far})");
        Integrator::flip_velocities(&mut s);
        for _ in 0..100 {
            f = integ.step(&mut s, &f, &mut p, &mut no_augment(), &mut never()).unwrap().0;
        }
        Integrator::flip_velocities(&mut s);
        let (dr, dv) = max_abs_diff(&s, &start);
        println!("reversibility after 100 + 100 steps: max |Δr| = {dr:.3e} Bohr, max |Δv| = {dv:.3e}");
        assert!(dr < 1e-10, "positions off by {dr:.3e}");
        assert!(dv < 1e-10, "velocities off by {dv:.3e}");
    }

    /// Zero forces, a_ext = (0, −g, 0): Verlet is exact for constant acceleration, so after n
    /// steps y = y₀ − ½ g (n dt)², v_y = −g n dt, and x drifts at v₀ — to rounding.
    #[test]
    fn external_field_accelerates_uniformly() {
        let g = 1e-4;
        let v0 = 1e-3;
        let mut s = scene(vec![Nucleus::new(1, [0.0, 2.0, 0.0]).with_vel([v0, 0.0, 0.0]), Nucleus::new(8, [1.0, 0.0, 0.0])]);
        s.external_field = [0.0, -g, 0.0];
        let integ = Integrator::new(1.0);
        let mut p = Zero;
        let mut f = forces_of(&mut p, &s);
        let n = 1000;
        for _ in 0..n {
            let (f_new, st) = integ.step(&mut s, &f, &mut p, &mut no_augment(), &mut never()).unwrap();
            f = f_new;
            assert_eq!(st.energy_ks, 0.0);
            assert!(f.iter().all(|f| *f == [0.0; 3]), "raw forces are the provider's (zero)");
        }
        let t = n as f64 * integ.dt;
        let (y_exact, vy_exact) = (2.0 - 0.5 * g * t * t, -g * t);
        println!("free fall {n} steps: y = {} (exact {y_exact}), v_y = {} (exact {vy_exact}), x = {} (exact {})",
                 s.nuclei[0].pos[1], s.nuclei[0].vel[1], s.nuclei[0].pos[0], v0 * t);
        for nuc in &s.nuclei {
            assert!((nuc.pos[1] - (if nuc.z == 1 { y_exact } else { -0.5 * g * t * t })).abs() < 1e-12, "y = {}", nuc.pos[1]);
            // 2000 kick additions onto a value of 0.1 (ulp 1.4e-17): rounding ~3e-15, measured
            assert!((nuc.vel[1] - vy_exact).abs() < 1e-13, "v_y = {}", nuc.vel[1]);
            assert_eq!(nuc.pos[2], 0.0);
        }
        assert!((s.nuclei[0].pos[0] - v0 * t).abs() < 1e-12);
        assert_eq!(s.nuclei[0].vel[0], v0);
        // the field is the same acceleration for every mass: H and O fell together
        assert_eq!(s.nuclei[0].vel[1], s.nuclei[1].vel[1]);
    }

    /// A two-body oscillator (pair spring through `augment`, momentum-conserving) stepped with
    /// and without a common COM velocity: `StepStats::temperature` = 2 K_int / (3 k_B) (DOF
    /// 3N − 3 = 3) in both runs, identical to rounding, while the kinetic energy carries the
    /// extra ½ M u²; one nucleus: DOF 3 with nothing removed; three: DOF 6.
    #[test]
    fn temperature_excludes_com_and_counts_dof() {
        let (k, d0) = (0.4, 1.4);
        let pair_spring = move |s: &Scene, f: &mut Vec<[f64; 3]>| {
            let d: [f64; 3] = std::array::from_fn(|a| s.nuclei[0].pos[a] - s.nuclei[1].pos[a]);
            let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            for a in 0..3 {
                let fa = -k * (l - d0) * d[a] / l;
                f[0][a] += fa;
                f[1][a] -= fa;
            }
        };
        let base = scene(vec![Nucleus::new(1, [-0.8, 0.0, 0.0]), Nucleus::new(1, [0.8, 0.0, 0.0])]);
        let m = base.nuclei[0].mass;
        let omega = (k / (0.5 * m)).sqrt(); // reduced mass m/2
        let integ = Integrator::new(0.1 / omega);
        let u = [2e-4, -3e-4, 5e-4];
        let mut moving = base.clone();
        for n in &mut moving.nuclei { n.vel = u; }
        let mut runs = [(base, Vec::new()), (moving, Vec::new())];
        for (s, log) in &mut runs {
            let mut p = Zero;
            let mut f = forces_of(&mut p, s);
            for _ in 0..50 {
                let (f_new, st) = integ.step(s, &f, &mut p, &mut pair_spring.clone(), &mut never()).unwrap();
                f = f_new;
                log.push((st.kinetic, st.temperature));
            }
        }
        let [(rest, log_rest), (moving, log_moving)] = runs;
        assert_eq!(rest.dof(), 3);
        let m_tot = rest.total_mass();
        let ke_com = 0.5 * m_tot * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        let mut max_dt_rel: f64 = 0.0;
        for ((ke_r, t_r), (ke_m, t_m)) in log_rest.iter().zip(&log_moving) {
            assert!(*t_r > 0.0);
            max_dt_rel = max_dt_rel.max((t_m / t_r - 1.0).abs());
            assert!((ke_m - ke_r - ke_com).abs() <= 1e-9 * ke_m, "KE {ke_m} vs {ke_r} + {ke_com}");
        }
        println!("pair oscillator: T_last = {:.3} K at rest vs {:.3} K moving (max rel diff {max_dt_rel:.2e}); KE {:.4e} vs {:.4e} Ha",
                 log_rest.last().unwrap().1, log_moving.last().unwrap().1, log_rest.last().unwrap().0, log_moving.last().unwrap().0);
        assert!(max_dt_rel < 1e-9, "T changed with the COM velocity: {max_dt_rel:.3e}");
        // T is 2 K_int / (3 k_B) where K_int is the kinetic energy in the COM frame
        let t_expect = units::hartree_to_kelvin(2.0 * log_rest.last().unwrap().0 / 3.0);
        assert!((log_rest.last().unwrap().1 / t_expect - 1.0).abs() < 1e-12);
        let com = moving.com_velocity();
        for a in 0..3 { assert!((com[a] - u[a]).abs() < 1e-15, "momentum not conserved: {com:?} vs {u:?}"); }

        // one nucleus: its translation is the only motion and counts (DOF 3, nothing removed)
        let mut one = scene(vec![Nucleus::new(1, [0.0; 3]).with_vel(u)]);
        let (_, st) = integ.step(&mut one, &[[0.0; 3]], &mut Zero, &mut no_augment(), &mut never()).unwrap();
        assert_eq!(st.temperature, units::hartree_to_kelvin(2.0 * st.kinetic / 3.0));
        assert!(st.kinetic > 0.0);
        // three nuclei: DOF 6
        let mut three = scene(vec![Nucleus::new(8, [0.0; 3]), Nucleus::new(1, [1.43, 1.11, 0.0]).with_vel([0.0, 1e-3, 0.0]), Nucleus::new(1, [-1.43, 1.11, 0.0]).with_vel([0.0, -1e-3, 0.0])]);
        let mut wells = Harmonic::at_rest_positions(&three, 0.3);
        let f0 = forces_of(&mut wells, &three);
        let (_, st) = integ.step(&mut three, &f0, &mut wells, &mut no_augment(), &mut never()).unwrap();
        let mut internal = three.clone();
        internal.remove_com_velocity();
        assert_eq!(three.dof(), 6);
        assert!((st.temperature - units::hartree_to_kelvin(2.0 * internal.kinetic_energy() / 6.0)).abs() <= 1e-9 * st.temperature);
    }

    /// The provider fails on its 3rd evaluation (the initial forces are call 1, so the second
    /// `step`): `Err` comes back and the scene is bit-for-bit the pre-step copy; the next step
    /// with the same `forces_old` succeeds and matches an unbroken run. All three error kinds.
    #[test]
    fn step_propagates_provider_error_without_mutating_scene() {
        for kind in [FailKind::Invalid, FailKind::Cancelled, FailKind::Numerical] {
            let k = 0.5;
            let start = scene(vec![Nucleus::new(1, [0.3, -0.2, 0.1]).with_vel([2e-3, 1e-3, -5e-4]), Nucleus::new(8, [-0.4, 0.5, -0.1])]);
            let integ = Integrator::new(5.0);
            let harmonic = Harmonic { k, centres: vec![[0.0; 3], [0.1, 0.1, 0.1]] };
            let mut failing = Failing::new(harmonic.clone(), 3).with_kind(kind);
            let mut s = start.clone();
            let mut f = forces_of(&mut failing, &s);
            f = integ.step(&mut s, &f, &mut failing, &mut no_augment(), &mut never()).expect("call 2 succeeds").0;
            let before = s.clone();
            let f_before = f.clone();
            let err = integ.step(&mut s, &f, &mut failing, &mut no_augment(), &mut never()).unwrap_err();
            println!("call 3 ({kind:?}): {err:?}");
            match kind {
                FailKind::Cancelled => assert!(matches!(err, ProviderError::Cancelled), "{err:?}"),
                FailKind::Invalid => assert!(matches!(err, ProviderError::Invalid(_)), "{err:?}"),
                FailKind::Numerical => assert!(matches!(err, ProviderError::Solver(qm::QmError::Numerical(_))), "{err:?}"),
            }
            assert_eq!(failing.calls, 3);
            assert!(s == before, "scene mutated by a failed step:\n{s:?}\nvs\n{before:?}");
            assert_eq!(f, f_before);
            // recovery: the same forces_old drive the next step, which equals the unbroken run
            let (f_after, st) = integ.step(&mut s, &f, &mut failing, &mut no_augment(), &mut never()).expect("call 4 succeeds");
            let mut reference = start.clone();
            let mut p = harmonic.clone();
            let mut fr = forces_of(&mut p, &reference);
            for _ in 0..2 {
                fr = integ.step(&mut reference, &fr, &mut p, &mut no_augment(), &mut never()).unwrap().0;
            }
            assert!(s == reference);
            assert_eq!(f_after, fr);
            assert!(st.total.is_finite());
        }
        // a provider returning the wrong number of forces is an error, not a panic, and the
        // scene is still untouched
        let mut s = scene(vec![Nucleus::new(1, [0.3, -0.2, 0.1]), Nucleus::new(1, [-0.3, 0.2, 0.1])]);
        let before = s.clone();
        let mut short = Harmonic { k: 0.5, centres: vec![[0.0; 3]] };
        let err = Integrator::new(1.0).step(&mut s, &[[0.0; 3]; 2], &mut short, &mut no_augment(), &mut never()).unwrap_err();
        assert!(matches!(err, ProviderError::Invalid(_)), "{err:?}");
        assert!(s == before);
    }

    /// `augment` enters BOTH kicks: a −k(r − r₀) spring added by the closure on top of the
    /// `Zero` provider reproduces the `Harmonic` provider's trajectory bit for bit (0 + x = x),
    /// while the returned raw forces stay the provider's zeros. Had the first kick skipped
    /// `augment`, the first half-step would differ and so would everything after it.
    #[test]
    fn augment_force_enters_both_kicks() {
        let k = 0.5;
        let centres = vec![[0.0; 3], [0.1, 0.1, 0.1]];
        let start = scene(vec![Nucleus::new(1, [0.3, -0.2, 0.1]).with_vel([2e-3, 1e-3, -5e-4]), Nucleus::new(8, [-0.4, 0.5, -0.1])]);
        let omega = (k / start.nuclei[0].mass).sqrt();
        let integ = Integrator::new(0.1 / omega);

        let mut a = start.clone();
        let mut harmonic = Harmonic { k, centres: centres.clone() };
        let mut fa = forces_of(&mut harmonic, &a);

        let mut b = start.clone();
        let c = centres.clone();
        let mut spring = move |s: &Scene, f: &mut Vec<[f64; 3]>| {
            for ((fi, n), c) in f.iter_mut().zip(&s.nuclei).zip(&c) {
                for x in 0..3 { fi[x] += -k * (n.pos[x] - c[x]); }
            }
        };
        let mut fb = forces_of(&mut Zero, &b);
        let mut energies_differ = false;
        for _ in 0..50 {
            let (fa_new, sta) = integ.step(&mut a, &fa, &mut harmonic, &mut no_augment(), &mut never()).unwrap();
            let (fb_new, stb) = integ.step(&mut b, &fb, &mut Zero, &mut spring, &mut never()).unwrap();
            fa = fa_new;
            fb = fb_new;
            assert!(a == b, "trajectories diverged:\n{a:?}\nvs\n{b:?}");
            assert!(fb.iter().all(|f| *f == [0.0; 3]), "raw forces must be the provider's");
            assert_eq!(sta.kinetic, stb.kinetic);
            energies_differ |= sta.energy_ks != stb.energy_ks; // Zero reports E = 0: the stats are the provider's
        }
        assert!(energies_differ);
        let (dr, _) = max_abs_diff(&a, &start);
        assert!(dr > 0.05, "test assumption: the system moved ({dr})");
    }
}
