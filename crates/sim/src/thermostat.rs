//! Bussi–Donadio–Parrinello canonical sampling through velocity rescaling (CSVR,
//! J. Chem. Phys. 126, 014101 (2007)): the one thermostat the spec admits (§7; Berendsen is
//! rejected for its wrong fluctuations, see below). Atomic units throughout (CLAUDE.md rule 3):
//! the target temperature in kelvin enters once, through `units::KB_HARTREE_PER_K`, in
//! [`Csvr::target_kinetic`]; `tau` and `dt` are atomic time units (converted from fs once, in
//! [`Csvr::new`]). All f64 on the CPU (rule 4). The driver calls [`Csvr::apply`] once per step
//! *after* the Verlet step (spec §7 order), never inside it; the integrator itself stays pure.
//!
//! # What the thermostat must sample
//! With N_f momentum components p_i (mass-scaled, p_i = √m v) the canonical density
//! e^{−β H} factorises, and each p_i is Gaussian with ⟨p_i²⟩ = k_B T. Hence
//! 2K/(k_B T) = Σ p_i²/(k_B T) ~ χ²_{N_f}, i.e. the kinetic energy is Gamma-distributed,
//!
//! ```text
//! P(K) ∝ K^{N_f/2 − 1} e^{−K/(k_B T)},     ⟨K⟩ = K̄ = N_f k_B T / 2,     Var K = N_f (k_B T)²/2,
//! Var K / ⟨K⟩² = 2 / N_f.
//! ```
//!
//! That last ratio is the fingerprint the tests check: 2/189 = 1.06 % for the 64-atom harmonic
//! system, 2/3 for a single atom. A thermostat that only drives ⟨K⟩ to K̄ is not sampling the
//! canonical ensemble; it must also reproduce these fluctuations.
//!
//! # Why Berendsen does not
//! Berendsen rescales by α² = 1 + (dt/τ)(K̄/K − 1): a deterministic drift of K toward K̄ with no
//! noise. Its fluctuations are τ-dependent and never canonical — Morishita (J. Chem. Phys. 113,
//! 2976 (2000)) shows they interpolate from microcanonical (τ → ∞) to isokinetic, Var K → 0
//! (τ → 0) — and the configurational distribution is wrong with them. Worse, uniform
//! deterministic rescaling pumps the energy the fast modes lose into the modes with no
//! dissipation channel of their own, the zero-frequency translations and rotations: the "flying
//! ice cube" (Harvey, Tan & Cheatham, J. Comput. Chem. 19, 726 (1998)) — a cold molecule drifting
//! as a block at the "right" temperature. The spec therefore rejects it; the sandbox's readouts
//! (T, Var K) would make the defect visible.
//!
//! # The CSVR process and its exact one-step solution
//! Bussi's construction: let every mass-scaled momentum component follow an Ornstein–Uhlenbeck
//! process relaxing to the Maxwell–Boltzmann variance,
//!
//! ```text
//! dp_i = −p_i dt/(2τ) + √(k_B T / τ) dW_i .
//! ```
//!
//! By Itô, K = ½ Σ p_i² obeys dK = Σ p_i dp_i + ½ Σ (dp_i)², and since Σ p_i dW_i has variance
//! Σ p_i² dt = 2K dt it is √(2K) dW for a single Wiener increment dW:
//!
//! ```text
//! dK = (K̄ − K) dt/τ + √(2 K k_B T / τ) dW = (K̄ − K) dt/τ + 2 √(K K̄ / (N_f τ)) dW,
//! ```
//!
//! which is Bussi's eq. (9) (k_B T = 2K̄/N_f). Its stationary density follows from the
//! Fokker–Planck equation with drift a = (K̄ − K)/τ and diffusion D = 2 K K̄/(N_f τ):
//! P ∝ D⁻¹ exp ∫ a/D dK, and a/D = (N_f/2)(1/K − 1/K̄) integrates to (N_f/2)(ln K − K/K̄), so
//! P ∝ K^{N_f/2 − 1} e^{−N_f K/(2K̄)} = K^{N_f/2 − 1} e^{−K/(k_B T)} — exactly the canonical
//! Gamma above. The OU process is linear, so it integrates *exactly* over any dt:
//!
//! ```text
//! p_i(t + dt) = √c p_i(t) + √((1 − c) k_B T) R_i,     c = e^{−dt/τ},  R_i ~ N(0, 1) i.i.d.
//! ```
//!
//! (stationary variance check: c k_B T + (1 − c) k_B T = k_B T). The R_i are isotropic, so rotate
//! them to put axis 1 along the current momentum p̂ = p/|p|, |p| = √(2K): the new kinetic energy
//! is K′ = ½[(√c √(2K) + √((1 − c) k_B T) R₁)² + (1 − c) k_B T Σ_{i≥2} R_i²]. The thermostat
//! realises exactly this K′ by rescaling every velocity by α = ±√(K′/K), which with
//! k_B T/(2K) = K̄/(N_f K) is
//!
//! ```text
//! α² = ( √c + √((1 − c) K̄/(N_f K)) R₁ )² + (1 − c) (K̄/(N_f K)) Σ_{i=2}^{N_f} R_i²
//!    = c + (1 − c) (K̄/(N_f K)) (R₁² + Σ_{i=2}^{N_f} R_i²) + 2 R₁ √(c (1 − c) K̄/(N_f K)),
//! ```
//!
//! Bussi's eq. (A7). Σ_{i=2}^{N_f} R_i² is a χ² with N_f − 1 degrees of freedom. The two forms are
//! algebraically identical; the code evaluates the first because it is manifestly ≥ 0 (no
//! cancellation when α changes sign) and because it collapses *exactly*: when dt/τ is so small
//! that c rounds to 1.0, (1 − c) is exactly 0, the stochastic terms are exactly 0, α² is exactly
//! 1.0 and α exactly 1.0 — `v *= 1.0` is a bitwise no-op, which is what τ → ∞ (NVE) must be.
//! (`1 − c` rather than `−expm1(−dt/τ)` for that reason; the relative precision it costs in the
//! noise amplitude at dt/τ ≪ 1 is irrelevant for a coupling constant.)
//!
//! **The sign of α.** The new component of p along p̂ is √c |p| + √((1 − c) k_B T) R₁, which is
//! negative when R₁ < −√(c N_f K/((1 − c) K̄)): the exact OU trajectory crossed the origin along
//! p̂. Rescaling by α = sgn(√c + √((1 − c) K̄/(N_f K)) R₁)·√(α²) reproduces that trajectory; taking
//! |α| would reflect instead. Both leave the Gamma distribution invariant (K′ is the same), so
//! both are canonical, but only the signed one is the exact solution of the process, and it is
//! what Bussi specifies — kept. In practice it is a many-sigma event (c = 0.975, K ≈ K̄,
//! N_f = 189 → R₁ < −86); it matters only as τ → 0.
//!
//! **Limits.** τ → ∞: c → 1, α = 1, identity (NVE). τ → 0: c → 0, α² = (K̄/(N_f K)) χ²_{N_f},
//! i.e. K′ = (k_B T/2) χ²_{N_f}: a fresh draw of the canonical kinetic energy every step,
//! independent of K, with the velocity directions kept (and a random overall sign). dt/τ in
//! between relaxes K to K̄ with time constant τ while keeping the Gamma distribution stationary
//! at every step — not only on average.
//!
//! **Why the composition with Verlet samples the canonical ensemble.** The rescale maps shells of
//! constant K to shells and keeps the direction of p, and the Maxwell–Boltzmann density is
//! uniform on each shell, so preserving the K-marginal (shown above) preserves the full momentum
//! distribution for any potential; the Verlet step preserves e^{−βH̃} of its shadow Hamiltonian.
//! Their product therefore has e^{−βH̃} as a stationary density (ergodicity assumed, as always).
//! For a harmonic mode H̃ is exactly ½mv² + ½k(1 − h²/4)x² (`integrator` module doc) — still
//! quadratic in v — so the kinetic-energy statistics the tests measure are those of the exact
//! canonical ensemble with no O(dt²) bias: the pins below are sharp.
//!
//! # Centre of mass
//! `apply` first removes the centre-of-mass velocity (N ≥ 2; for N = 1 the translation *is* the
//! motion, matching `Scene::{dof, temperature}`), then computes K and rescales in that frame. The
//! rescale is a common factor, so Σ m v stays zero to rounding afterwards. This is the
//! convention of `Scene::temperature` — the thermostat controls the temperature the panel shows —
//! and it is also what removes the flying-ice-cube channel for good: a COM drift can never be
//! fed by the thermostat. Consequence worth knowing: under an external field the COM velocity the
//! field builds during a step is removed at the end of it, so a thermostatted system does not
//! fall as a whole (it is in the thermostat's frame, the natural reading of "a bath").
//!
//! # Random numbers
//! Gaussians by Marsaglia's polar method on `rand::rngs::StdRng` (CLAUDE.md rule 2: `rand` is
//! plumbing; no distribution crate). The χ²_{N_f − 1} is the literal sum of N_f − 1 squared
//! Gaussians: exact, ~N_f draws per step, microseconds next to the DFT solve that preceded them
//! (a Marsaglia–Tsang gamma draw would save that only in the thousands-of-atoms regime this
//! sandbox cannot reach). A fixed seed gives a bitwise-reproducible trajectory.
//!
//! # Degenerate inputs
//! K = 0 (every velocity zero): α is undefined (0 × ∞) and the direction of the new momentum is
//! not determined by a rescale, so `apply` does nothing; a rescaling thermostat cannot create
//! motion from nothing — the forces do that on the next Verlet step (a scene placed by hand is
//! never at an exact minimum), after which the thermostat heats it. T_target = 0: K̄ = 0,
//! α² = c, an exponential quench. τ = 0: c = e^{−∞} = 0, the full-resample limit.
use crate::scene::Scene;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// The CSVR thermostat state: target temperature (K), coupling time `tau` (atomic time units)
/// and the seeded generator. Construct with [`Csvr::new`]; the public fields may be edited
/// live (the driver's `SetThermostat`).
#[derive(Clone, Debug)]
pub struct Csvr {
    /// Target temperature in kelvin (≥ 0).
    pub target_t: f64,
    /// Coupling time in atomic time units (≥ 0; 0 is the full-resample limit).
    pub tau: f64,
    rng: StdRng,
}

impl Csvr {
    /// `tau_fs` in femtoseconds (converted once via `units::fs_to_au_time`), `seed` for the
    /// generator. Panics on a negative temperature or coupling time.
    pub fn new(target_t: f64, tau_fs: f64, seed: u64) -> Csvr {
        assert!(target_t >= 0.0, "Csvr: target temperature {target_t} K < 0");
        assert!(tau_fs >= 0.0, "Csvr: coupling time {tau_fs} fs < 0");
        Csvr { target_t, tau: units::fs_to_au_time(tau_fs), rng: StdRng::seed_from_u64(seed) }
    }

    /// K̄ = N_f k_B T / 2 (Hartree) for `dof` degrees of freedom — the only place kelvin enters.
    pub fn target_kinetic(&self, dof: usize) -> f64 {
        0.5 * dof as f64 * units::kelvin_to_hartree(self.target_t.max(0.0))
    }

    /// One CSVR step of length `dt` (atomic time units) on the scene's velocities: removes the
    /// centre-of-mass velocity (N ≥ 2), then rescales every velocity by the signed α of the
    /// module doc, drawn with this thermostat's generator. Afterwards Σ m v = 0 (to rounding) and
    /// K is the canonical kinetic energy for `Scene::dof` degrees of freedom. A no-op when K = 0.
    pub fn apply(&mut self, scene: &mut Scene, dt: f64) {
        debug_assert!(dt >= 0.0, "Csvr::apply: dt = {dt} < 0");
        if scene.nuclei.is_empty() {
            return;
        }
        // the thermostat's frame: the one `Scene::temperature` reports (N = 1 keeps its motion)
        if scene.nuclei.len() >= 2 {
            scene.remove_com_velocity();
        }
        let k = scene.kinetic_energy();
        if k <= 0.0 {
            return; // nothing to rescale: the direction of the new momentum is undefined
        }
        let dof = scene.dof();
        // c = e^{−dt/τ}; `1 − c` (not −expm1) so that c == 1.0 gives exactly 0 (module doc)
        let c = (-dt / self.tau).exp();
        let omc = 1.0 - c;
        // K̄/(N_f K) = k_B T/(2K): the variance of one new mass-scaled component over |p|²
        let ratio = self.target_kinetic(dof) / (dof as f64 * k);
        let r1 = self.gaussian();
        let rest = self.chi_squared(dof - 1);
        // eq. A7 in its square-completed form: the new component along p̂ (in units of |p|),
        // squared, plus the perpendicular χ²_{N_f − 1} share
        let along = c.sqrt() + (omc * ratio).sqrt() * r1;
        let alpha2 = along * along + omc * ratio * rest;
        let alpha = alpha2.sqrt().copysign(along);
        for n in &mut scene.nuclei {
            for v in &mut n.vel {
                *v *= alpha;
            }
        }
    }

    /// Σ_{i=1}^{n} R_i² of n independent standard normals (χ² with n degrees of freedom).
    fn chi_squared(&mut self, n: usize) -> f64 {
        crate::scene::neumaier((0..n).map(|_| {
            let r = self.gaussian();
            r * r
        }))
    }

    /// One standard normal by Marsaglia's polar method: (u, v) uniform in the unit disc
    /// (rejection from the square), s = u² + v², return u √(−2 ln s / s); the partner value
    /// v √(−2 ln s / s) is discarded for simplicity (an extra ~0.27 draws per Gaussian).
    fn gaussian(&mut self) -> f64 {
        loop {
            let u = 2.0 * self.rng.gen::<f64>() - 1.0;
            let v = 2.0 * self.rng.gen::<f64>() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                return u * (-2.0 * s.ln() / s).sqrt();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrator::Integrator;
    use crate::scene::{Cell, Nucleus};
    use crate::test_providers::Harmonic;

    fn scene(nuclei: Vec<Nucleus>) -> Scene {
        Scene { nuclei, cell: Cell { n: 33, h: 0.25 }, charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3] }
    }

    /// 64 H nuclei on a 4×4×4 lattice (spacing 4 Bohr), each displaced 0.1 Bohr from its own
    /// well centre in a direction that cycles with the lattice index — "at rest" (K = 0) but not
    /// at the minimum, so the first Verlet step sets them moving (½k d² = 2.5e-3 Ha per nucleus
    /// against 3k_BT/2 = 1.4e-3 at 300 K: the system starts hot-ish and the thermostat cools it).
    fn lattice_64() -> (Scene, Harmonic) {
        let dirs: [[f64; 3]; 6] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
        let mut nuclei = Vec::new();
        let mut centres = Vec::new();
        for ix in 0..4 {
            for iy in 0..4 {
                for iz in 0..4 {
                    let i = ix * 16 + iy * 4 + iz;
                    let c = [4.0 * (ix as f64 - 1.5), 4.0 * (iy as f64 - 1.5), 4.0 * (iz as f64 - 1.5)];
                    let d = dirs[i % 6];
                    centres.push(c);
                    nuclei.push(Nucleus::new(1, [c[0] + 0.1 * d[0], c[1] + 0.1 * d[1], c[2] + 0.1 * d[2]]));
                }
            }
        }
        (scene(nuclei), Harmonic { k: 0.5, centres })
    }

    /// Verlet + CSVR for `steps` steps; returns (K, T) after each `apply` from step `discard` on,
    /// K = `Scene::kinetic_energy` (COM-free after `apply`), T = `Scene::temperature`.
    fn run(s: &mut Scene, p: &mut Harmonic, csvr: &mut Csvr, integ: &Integrator, steps: usize, discard: usize) -> Vec<(f64, f64)> {
        let mut f = p.evaluate(s, &mut || false).expect("harmonic").forces;
        let mut log = Vec::with_capacity(steps - discard);
        for i in 0..steps {
            f = integ.step(s, &f, p, &mut |_, _| {}, &mut || false).expect("harmonic never fails").0;
            csvr.apply(s, integ.dt);
            if i >= discard {
                log.push((s.kinetic_energy(), s.temperature()));
            }
        }
        log
    }

    fn mean(x: &[f64]) -> f64 {
        crate::scene::neumaier(x.iter().copied()) / x.len() as f64
    }

    fn variance(x: &[f64]) -> f64 {
        let m = mean(x);
        crate::scene::neumaier(x.iter().map(|v| (v - m) * (v - m))) / (x.len() - 1) as f64
    }

    /// Standard error of the mean of a correlated series by batch means: blocks of `block`
    /// consecutive samples (≫ the correlation time), the sample standard deviation of the block
    /// means over √(number of blocks).
    fn block_std_err(x: &[f64], block: usize) -> f64 {
        let means: Vec<f64> = x.chunks_exact(block).map(mean).collect();
        (variance(&means) / means.len() as f64).sqrt()
    }

    use crate::provider::ForceProvider;

    /// The brief's run: 64 H in k = 0.5 wells, T = 300 K, dt = 0.25 fs, τ = 10 fs, 20 000 steps,
    /// the first 2000 discarded. Mean T within 3 % of 300 K; mean K within 3 % of
    /// K̄ = N_f k_B T/2 with N_f = 189.
    ///
    /// Statistical margin (the module doc's process): the thermostat relaxes K toward K̄ with
    /// time constant τ, but for an oscillator the energy E = K + V, of which K is on average
    /// half, so E — and with it the slow part of K — relaxes with time constant 2τ = 20 fs =
    /// 80 steps. Var K = Var(E/2) + Var(fast part): for independent oscillators E ~ Gamma(N_f),
    /// so the slow half carries Var/K̄² = 1/N_f and the other 1/N_f decorrelates within half an
    /// oscillation period (4.6 fs ≈ 18 steps). The variance of the mean of n correlated samples
    /// is Σ σ_part² · 2τ_part/(n dt): (1/189)(160 + ~6)/18 000 = 4.9e-5, a relative standard
    /// error of ≈ 0.7 % (an all-slow estimate would say 1.0 %), so the 3 % bound is ≈ 4σ. The
    /// test measures the standard error by batch means (blocks of 500 steps ≫ 80: 0.59 % with
    /// the seed below, 5.1σ) and prints the margin in σ so the number is checked, not assumed.
    #[test]
    #[cfg_attr(debug_assertions, ignore)] // statistics over 10⁴–10⁵ steps: release only, like the oracles
    fn reaches_and_holds_target_temperature_on_harmonic_system() {
        let (mut s, mut p) = lattice_64();
        let integ = Integrator::new_fs(0.25);
        let mut csvr = Csvr::new(300.0, 10.0, 7);
        assert_eq!(s.dof(), 189);
        let kbar = csvr.target_kinetic(189);
        assert!((kbar - 0.5 * 189.0 * 300.0 * units::KB_HARTREE_PER_K).abs() < 1e-18);
        let log = run(&mut s, &mut p, &mut csvr, &integ, 20_000, 2_000);
        assert_eq!(log.len(), 18_000);
        let ks: Vec<f64> = log.iter().map(|l| l.0).collect();
        let ts: Vec<f64> = log.iter().map(|l| l.1).collect();
        let (mk, mt) = (mean(&ks), mean(&ts));
        let se_t = block_std_err(&ts, 500);
        println!("64 H, τ = 10 fs: ⟨T⟩ = {mt:.2} K (target 300, dev {:+.2} %, std err {:.2} K = {:.2} %, deviation {:.2}σ, 3 % bound = {:.1}σ)",
                 100.0 * (mt / 300.0 - 1.0), se_t, 100.0 * se_t / 300.0, (mt - 300.0).abs() / se_t, 0.03 * 300.0 / se_t);
        println!("  ⟨K⟩ = {mk:.6e} Ha vs K̄ = {kbar:.6e} (dev {:+.2} %)", 100.0 * (mk / kbar - 1.0));
        assert!((mt / 300.0 - 1.0).abs() < 0.03, "⟨T⟩ = {mt} K");
        assert!((mk / kbar - 1.0).abs() < 0.03, "⟨K⟩ = {mk} vs {kbar}");
        // T and K are the same statistic (T = 2K/(N_f k_B) with the COM removed by `apply`)
        assert!((mt - units::hartree_to_kelvin(2.0 * mk / 189.0)).abs() < 1e-9 * mt);
        // the 3 % bound is a real margin: at least 2.5× the measured standard error
        assert!(0.03 * 300.0 > 2.5 * se_t, "3 % bound is only {:.1}σ", 0.03 * 300.0 / se_t);
        // and the system did move: the COM frame is where the thermostat works
        let u = s.com_velocity();
        assert!(u.iter().all(|c| c.abs() < 1e-18), "COM velocity after apply: {u:?}");
    }

    /// The same run: Var K / ⟨K⟩² = 2/N_f = 2/189 within 25 % — the fingerprint Berendsen lacks
    /// (its ratio would be far below 2/N_f at τ = 10 fs and vanish as τ → 0).
    #[test]
    #[cfg_attr(debug_assertions, ignore)] // statistics over 10⁴–10⁵ steps: release only, like the oracles
    fn kinetic_energy_fluctuations_are_canonical() {
        let (mut s, mut p) = lattice_64();
        let integ = Integrator::new_fs(0.25);
        let mut csvr = Csvr::new(300.0, 10.0, 7);
        let log = run(&mut s, &mut p, &mut csvr, &integ, 20_000, 2_000);
        let ks: Vec<f64> = log.iter().map(|l| l.0).collect();
        let ratio = variance(&ks) / mean(&ks).powi(2);
        let expect = 2.0 / 189.0;
        println!("64 H, τ = 10 fs: Var K/⟨K⟩² = {ratio:.5} vs 2/N_f = {expect:.5} (ratio {:.3})", ratio / expect);
        assert!((ratio / expect - 1.0).abs() < 0.25, "Var K/⟨K⟩² = {ratio} vs {expect}");
    }

    /// τ = 1e30: c rounds to 1.0, α is exactly 1.0, the velocities are bitwise unchanged — on a
    /// scene whose COM velocity is exactly zero. On a scene with a COM velocity the result is
    /// bitwise the COM-removed scene (the thermostat works in that frame by design).
    #[test]
    fn tau_infinite_is_identity() {
        let v = [3e-4, -2e-4, 5e-4];
        let still = scene(vec![Nucleus::new(1, [0.7, 0.0, 0.0]).with_vel(v),
                               Nucleus::new(1, [-0.7, 0.0, 0.0]).with_vel([-v[0], -v[1], -v[2]])]);
        assert_eq!(still.com_velocity(), [0.0; 3]);
        for (tau_fs, set_au) in [(1e30, false), (0.0, true)] {
            let mut csvr = Csvr::new(300.0, tau_fs, 1);
            if set_au { csvr.tau = 1e30; }
            let mut s = still.clone();
            for _ in 0..10 { csvr.apply(&mut s, 10.0); }
            assert!(s == still, "τ = ∞ changed the scene:\n{s:?}\nvs\n{still:?}");
        }
        // with a COM velocity: identical to the COM-removed scene, bitwise
        let mut moving = scene(vec![Nucleus::new(8, [0.0; 3]).with_vel([1e-4, 2e-4, -3e-4]),
                                    Nucleus::new(1, [1.43, 1.11, 0.0]).with_vel([3e-4, -1e-3, 2e-4]),
                                    Nucleus::new(1, [-1.43, 1.11, 0.0]).with_vel([-2e-4, 5e-4, 7e-4])]);
        let mut removed = moving.clone();
        removed.remove_com_velocity();
        assert!(moving != removed);
        Csvr::new(300.0, 1e30, 1).apply(&mut moving, 10.0);
        assert!(moving == removed, "τ = ∞ with a COM velocity:\n{moving:?}\nvs\n{removed:?}");
        // a single nucleus: nothing removed, nothing changed
        let one = scene(vec![Nucleus::new(1, [0.0; 3]).with_vel(v)]);
        let mut s = one.clone();
        Csvr::new(300.0, 1e30, 1).apply(&mut s, 10.0);
        assert!(s == one);
    }

    /// Two thermostats with the same seed drive identical trajectories, bitwise; a different
    /// seed does not.
    #[test]
    fn deterministic_with_seed() {
        let integ = Integrator::new_fs(0.25);
        let mut runs = Vec::new();
        for seed in [42u64, 42, 43] {
            let (mut s, mut p) = lattice_64();
            let mut csvr = Csvr::new(300.0, 10.0, seed);
            let log = run(&mut s, &mut p, &mut csvr, &integ, 500, 0);
            runs.push((s, log));
        }
        assert!(runs[0].0 == runs[1].0, "same seed, different scenes");
        assert_eq!(runs[0].1, runs[1].1);
        assert!(runs[0].0 != runs[2].0, "different seeds gave the same trajectory");
        assert!(runs[0].1.iter().all(|(k, _)| *k > 0.0));
    }

    /// N = 1: N_f = 3, the χ² has 2 degrees of freedom, nothing panics, no COM is removed, and T
    /// converges. Fluctuations are large (Var K/⟨K⟩² = 2/3), so the run is longer: 200 000 steps,
    /// τ_c ≈ 2τ = 80 steps → relative standard error of ⟨T⟩ ≈ √(0.667 · 160 / 198 000) ≈ 2.3 %
    /// (less with the fast half, as in the 64-H test); the 10 % bound is ≥ 4σ. The variance ratio
    /// is checked loosely (its error is ≈ 7 %, from the Gamma(3/2) excess kurtosis of 4).
    ///
    /// The start is a *generic* displacement. A displacement along one axis alone would leave
    /// v_y = v_z = 0 exactly forever — zero force, and a rescale of 0 is 0 — a measure-zero
    /// invariant set the thermostat cannot leave: CSVR controls the total K and relies on the
    /// dynamics for equipartition between modes. Second half of the test: that frozen case is
    /// pinned to its own derived value. Phase-averaging the K-process over one active 1D mode
    /// with N_f = 3 in the formula (dE = (K̄ − E/2) dt/τ + √(2 E K̄/(N_f τ)) dW, K ≈ E sin²φ on
    /// average over φ) gives the stationary E ~ Gamma(N_f, k_B T), so ⟨K⟩ = ⟨E⟩/2 = K̄ — the
    /// temperature readout is still right — while Var K/⟨K⟩² = ⟨E²⟩⟨sin⁴φ⟩/K̄² − 1 =
    /// (1 + 1/N_f)(2K̄)²(3/8)/K̄² − 1 = (4/3)(4)(3/8) − 1 = 1, not 2/3: the fingerprint exposes
    /// the frozen modes even though T does not (checked within 25 %; measured 1.006).
    #[test]
    #[cfg_attr(debug_assertions, ignore)] // statistics over 10⁴–10⁵ steps: release only, like the oracles
    fn single_particle_dof_edge() {
        let integ = Integrator::new_fs(0.25);
        for (start, expect, label) in [([0.1, 0.07, -0.05], 2.0 / 3.0, "generic start"), ([0.1, 0.0, 0.0], 1.0, "frozen y, z")] {
            let mut s = scene(vec![Nucleus::new(1, start)]);
            let mut p = Harmonic { k: 0.5, centres: vec![[0.0; 3]] };
            let mut csvr = Csvr::new(300.0, 10.0, 11);
            assert_eq!(s.dof(), 3);
            let kbar = csvr.target_kinetic(3);
            let log = run(&mut s, &mut p, &mut csvr, &integ, 200_000, 2_000);
            let ks: Vec<f64> = log.iter().map(|l| l.0).collect();
            let ts: Vec<f64> = log.iter().map(|l| l.1).collect();
            let (mk, mt) = (mean(&ks), mean(&ts));
            let se_t = block_std_err(&ts, 1000);
            let ratio = variance(&ks) / mk.powi(2);
            println!("1 H, τ = 10 fs, {label}: ⟨T⟩ = {mt:.2} K (std err {se_t:.2} K, deviation {:.2}σ, 10 % = {:.1}σ), ⟨K⟩/K̄ = {:.4}, Var K/⟨K⟩² = {ratio:.4} vs {expect:.4}",
                     (mt - 300.0).abs() / se_t, 30.0 / se_t, mk / kbar);
            assert!(ks.iter().all(|k| k.is_finite()));
            assert!((mt / 300.0 - 1.0).abs() < 0.10, "{label}: ⟨T⟩ = {mt} K");
            assert!((mk / kbar - 1.0).abs() < 0.10);
            assert!(0.10 * 300.0 > 3.0 * se_t, "{label}: 10 % bound is only {:.1}σ", 30.0 / se_t);
            assert!((ratio / expect - 1.0).abs() < 0.25, "{label}: Var K/⟨K⟩² = {ratio} vs {expect}");
            // the lab-frame velocity is the motion: it was rescaled, not removed
            assert!(s.nuclei[0].vel[0] != 0.0);
            if expect == 1.0 {
                assert_eq!(s.nuclei[0].vel[1], 0.0, "a frozen mode stays exactly frozen");
                assert_eq!(s.nuclei[0].vel[2], 0.0);
            } else {
                assert!(s.nuclei[0].vel[1] != 0.0 && s.nuclei[0].vel[2] != 0.0);
            }
        }
    }

    /// Velocities are rescaled about the centre of mass: a zero COM velocity stays zero (to
    /// rounding) through many applies, and a COM velocity present before `apply` is removed —
    /// the internal velocities then equal those of the COM-removed scene rescaled with the same
    /// draws (to rounding, since a second `remove_com_velocity` is not an exact no-op).
    #[test]
    fn rescales_about_centre_of_mass() {
        let (mut s, _) = lattice_64();
        // thermal-ish velocities with a deliberate pattern, then an added COM velocity
        for (i, n) in s.nuclei.iter_mut().enumerate() {
            let f = i as f64;
            n.vel = [7e-4 * (0.3 * f).sin(), 7e-4 * (0.7 * f).cos(), 7e-4 * (0.5 * f + 1.0).sin()];
        }
        let mut internal = s.clone();
        internal.remove_com_velocity();
        let u = [2e-4, -3e-4, 5e-4];
        let mut moving = internal.clone();
        for n in &mut moving.nuclei { for (v, du) in n.vel.iter_mut().zip(u) { *v += du; } }
        let com = moving.com_velocity();
        assert!((0..3).all(|a| (com[a] - u[a]).abs() < 1e-18), "{com:?}");

        let mut a = Csvr::new(300.0, 1.0, 5);
        let mut b = a.clone();
        a.apply(&mut internal, 10.0);
        b.apply(&mut moving, 10.0);
        // Measurement floor: `com_velocity` sums 64 terms of magnitude m|v| ≈ 1.3 that cancel
        // to ~0, so its own rounding is up to ~N ε m v_th / M = N ε v_th ≈ 64 × 2.2e-16 × 7e-4
        // ≈ 1e-17 (measured residuals 5e-20 and 8e-19); the bound is that floor.
        let v_th = 7e-4;
        let floor = 64.0 * f64::EPSILON * v_th;
        let com_i = internal.com_velocity();
        let com_m = moving.com_velocity();
        println!("COM after apply: internal {com_i:?}, moving {com_m:?} (floor {floor:.1e})");
        for a in 0..3 {
            assert!(com_i[a].abs() < floor, "COM crept: {com_i:?}");
            assert!(com_m[a].abs() < floor, "COM not removed: {com_m:?}");
        }
        for (x, y) in internal.nuclei.iter().zip(&moving.nuclei) {
            for k in 0..3 {
                assert!((x.vel[k] - y.vel[k]).abs() < 1e-14 * v_th, "{:?} vs {:?}", x.vel, y.vel);
            }
        }
        // the rescale changed the kinetic energy (τ = 1 fs, dt = 10 a.u. = 0.24 fs is a strong kick)
        assert!(internal.kinetic_energy() > 0.0);
        // many applies: the COM stays put
        let mut c = Csvr::new(300.0, 1.0, 9);
        for _ in 0..1000 { c.apply(&mut internal, 10.0); }
        let com = internal.com_velocity();
        assert!(com.iter().all(|x| x.abs() < floor), "COM drifted over 1000 applies: {com:?}");
    }

    /// The rescaling alone, with no dynamics, is a Markov chain on K whose stationary
    /// distribution must be Gamma(N_f/2, k_B T) for every c (module doc). Static 64-H scene,
    /// 200 000 applies: at τ = 0 (c = 0) the draws are i.i.d. canonical, so ⟨K⟩ is pinned to
    /// 0.1 % and Var K/⟨K⟩² to ~1 %; at dt/τ = 0.1 the chain's correlation time is 10 applies,
    /// ⟨K⟩ to ~0.3 % and the variance ratio to ~3 %. Bounds 1 % and 10 %. This is the sharpest
    /// test of eq. A7 itself, independent of the integrator.
    #[test]
    #[cfg_attr(debug_assertions, ignore)] // statistics over 10⁴–10⁵ steps: release only, like the oracles
    fn static_rescaling_chain_is_stationary_gamma() {
        for (tau_fs, dt, label) in [(0.0, 10.0, "c = 0"), (10.0, units::fs_to_au_time(1.0), "dt/τ = 0.1")] {
            let (mut s, _) = lattice_64();
            for (i, n) in s.nuclei.iter_mut().enumerate() {
                let f = i as f64;
                n.vel = [7e-4 * (0.3 * f).sin(), 7e-4 * (0.7 * f).cos(), 7e-4 * (0.5 * f + 1.0).sin()];
            }
            // COM removed up front so the sign probe below sees only α's sign (the first
            // `apply` would otherwise turn nucleus 0's exact v_x = sin 0 = 0 into −u_x)
            s.remove_com_velocity();
            assert!(s.nuclei[0].vel[0] != 0.0);
            let mut csvr = Csvr::new(300.0, tau_fs, 3);
            let kbar = csvr.target_kinetic(189);
            let n = 200_000;
            let mut ks = Vec::with_capacity(n);
            let mut flips = 0usize;
            for _ in 0..n {
                let before = s.nuclei[0].vel[0];
                csvr.apply(&mut s, dt);
                if (s.nuclei[0].vel[0] < 0.0) != (before < 0.0) { flips += 1; }
                ks.push(s.kinetic_energy());
            }
            let m = mean(&ks);
            let ratio = variance(&ks) / (m * m);
            println!("static chain {label}: ⟨K⟩/K̄ = {:.5}, Var K/⟨K⟩² = {ratio:.5} vs 2/189 = {:.5} (ratio {:.4}), sign flips {flips}",
                     m / kbar, 2.0 / 189.0, ratio / (2.0 / 189.0));
            assert!((m / kbar - 1.0).abs() < 0.01, "{label}: ⟨K⟩/K̄ = {}", m / kbar);
            assert!((ratio / (2.0 / 189.0) - 1.0).abs() < 0.10, "{label}: Var K/⟨K⟩² = {ratio}");
            // c = 0: α = sgn(R₁)·√(…), so the overall sign flips half the time; dt/τ = 0.1: never
            if tau_fs == 0.0 {
                assert!((flips as f64 / n as f64 - 0.5).abs() < 0.01, "flip fraction {}", flips as f64 / n as f64);
            } else {
                assert_eq!(flips, 0);
            }
        }
    }

    /// The in-crate Gaussian: moments of 400 000 draws (mean, variance, skewness, kurtosis) and
    /// the χ² sum (mean n, variance 2n for n = 188 over 20 000 draws).
    #[test]
    #[cfg_attr(debug_assertions, ignore)] // statistics over 10⁴–10⁵ steps: release only, like the oracles
    fn gaussian_and_chi_squared_moments() {
        let mut c = Csvr::new(300.0, 10.0, 2024);
        let n = 400_000;
        let z: Vec<f64> = (0..n).map(|_| c.gaussian()).collect();
        let m1 = mean(&z);
        let m2 = mean(&z.iter().map(|x| x * x).collect::<Vec<_>>());
        let m3 = mean(&z.iter().map(|x| x * x * x).collect::<Vec<_>>());
        let m4 = mean(&z.iter().map(|x| x.powi(4)).collect::<Vec<_>>());
        println!("gaussian: mean {m1:+.4}, ⟨z²⟩ {m2:.4}, ⟨z³⟩ {m3:+.4}, ⟨z⁴⟩ {m4:.4} (expect 0, 1, 0, 3; std errs 0.0016, 0.0022, 0.006, 0.015)");
        assert!(m1.abs() < 0.008, "mean {m1}");
        assert!((m2 - 1.0).abs() < 0.01, "variance {m2}");
        assert!(m3.abs() < 0.03, "skewness {m3}");
        assert!((m4 - 3.0).abs() < 0.08, "kurtosis {m4}");
        assert!(z.iter().all(|x| x.is_finite()));
        let nf = 188;
        let chi: Vec<f64> = (0..20_000).map(|_| c.chi_squared(nf)).collect();
        let (cm, cv) = (mean(&chi), variance(&chi));
        println!("χ²_{nf}: mean {cm:.3} (expect {nf}), var {cv:.2} (expect {})", 2 * nf);
        assert!((cm / nf as f64 - 1.0).abs() < 0.005, "χ² mean {cm}");
        assert!((cv / (2.0 * nf as f64) - 1.0).abs() < 0.05, "χ² var {cv}");
        assert_eq!(c.chi_squared(0), 0.0);
        assert!(c.chi_squared(1) >= 0.0);
    }

    /// Degenerate inputs: K = 0 is a no-op; T = 0 quenches with α² = c exactly; τ = 0 does not
    /// panic; negative inputs are rejected by `new`.
    #[test]
    fn degenerate_inputs() {
        let rest = scene(vec![Nucleus::new(1, [0.3, 0.0, 0.0]), Nucleus::new(1, [-0.3, 0.0, 0.0])]);
        let mut s = rest.clone();
        Csvr::new(300.0, 10.0, 1).apply(&mut s, 10.0);
        assert!(s == rest, "K = 0 must be a no-op");
        // T = 0: α² = c = e^{−dt/τ}, deterministic
        let v = [3e-4, -2e-4, 5e-4];
        let mut s = scene(vec![Nucleus::new(1, [0.7, 0.0, 0.0]).with_vel(v), Nucleus::new(1, [-0.7, 0.0, 0.0]).with_vel([-v[0], -v[1], -v[2]])]);
        let k0 = s.kinetic_energy();
        let mut cold = Csvr::new(0.0, 10.0, 1);
        let dt = 5.0;
        cold.apply(&mut s, dt);
        let c = (-dt / cold.tau).exp();
        assert!((s.kinetic_energy() / k0 - c).abs() < 1e-14, "quench: K/K₀ = {} vs c = {c}", s.kinetic_energy() / k0);
        assert_eq!(cold.target_kinetic(189), 0.0);
        // τ = 0: a full resample, finite
        let mut hot = Csvr::new(300.0, 0.0, 1);
        hot.apply(&mut s, dt);
        assert!(s.kinetic_energy().is_finite() && s.kinetic_energy() > 0.0);
        // an empty scene
        let mut empty = scene(vec![]);
        hot.apply(&mut empty, dt);
        assert!(std::panic::catch_unwind(|| Csvr::new(-1.0, 10.0, 1)).is_err());
        assert!(std::panic::catch_unwind(|| Csvr::new(300.0, -1.0, 1)).is_err());
    }
}
