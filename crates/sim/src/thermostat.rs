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
//! # Centre of mass (controller ruling R10)
//! `apply` acts on the N_f internal degrees of freedom only, and the centre of mass is not one of
//! them (`Scene::dof` = 3N − 3): with u the centre-of-mass velocity (N ≥ 2; for N = 1 the
//! translation *is* the motion, u = 0, matching `Scene::{dof, temperature}`) it computes K from
//! the peculiar velocities v − u and sets v ← u + α (v − u). Σ m (v − u) = 0, so the COM
//! velocity is preserved (to rounding) and K changes by exactly the internal ΔK. This is the
//! convention of `Scene::temperature` — the thermostat controls the temperature the panel shows —
//! and it still closes the flying-ice-cube channel: α never sees, and so never feeds, the COM
//! kinetic energy. Consequence: COM motion driven from outside — the external field, the
//! driver's drag spring — is left to the dynamics, so a thermostatted system falls under gravity
//! and follows a drag at the NVE pace. (An earlier version removed u on every `apply`; that is
//! a dt-dependent friction on any driven COM motion — terminal speed dt·F/2M — and was dropped.)
//!
//! # Frozen modes, and why a system started from rest needs [`Csvr::maxwell_boltzmann`]
//! CSVR controls the *total* K and relies on the dynamics for equipartition between modes: a
//! global rescale multiplies every mode's velocity by the same α, so a mode with exactly zero
//! velocity amplitude stays exactly zero (0 · α = 0) unless a force excites it. This is not a
//! corner case — it is the default start of the sandbox. A molecule placed at rest has angular
//! momentum L = 0 exactly; any rotationally invariant force field (central pair forces, the
//! analytic `Harmonic` provider) is torque-free and conserves L, velocity Verlet conserves it to
//! rounding, and the rescale maps L to αL — so L stays 0 forever and the molecule never rotates.
//! `Scene::dof` = 3N − 3 counts the three rotations as thermal (a Task 2 convention), so the
//! thermostat drives the full K̄ = (3N − 3) k_B T/2 into the 3N − 6 vibrations (3N − 5 for a
//! linear molecule): vibrations hotter by **(3N − 3)/(3N − 6) — 3× for H₂, 2× for H₂O** — while
//! the T readout sits at the target (⟨K⟩ = K̄ regardless) and only the Var K/⟨K⟩² fingerprint
//! shows the frozen modes (the derivation and the pin are in `single_particle_dof_edge`;
//! the reviewer reproduced 309 K readout / ⟨K_vib⟩ = 3.09 · k_B T/2 on a diatomic from rest).
//!
//! Two halves of the resolution (controller ruling R7):
//! - **Remedy: start the velocities thermal, not at rest.** When the thermostat is switched on
//!   for a system at rest (K < 1 % of K̄) the driver calls [`Csvr::maxwell_boltzmann`], which
//!   draws every component from N(0, k_B T/m_i), removes the COM velocity and rescales to exactly
//!   K̄: the rotations (and every other mode) then start with their canonical share, L ≠ 0, and
//!   nothing is frozen. It also resolves the K = 0 no-op below for a single atom, whose DFT force
//!   at a grid-symmetric position is exactly zero.
//! - **DOF stays 3N − 3.** On the real-space grid the Kohn–Sham forces are *not* rotationally
//!   invariant (the egg-box anisotropy, ΣF ≈ 3e-4 Ha/Bohr at h = 0.25 per the M3 handoff), so L
//!   is not conserved by the DFT forces — only by the analytic providers — and rotation is a
//!   genuinely thermal mode of the simulated system, exchanging energy with the vibrations through
//!   the grid. Counting it is the honest convention; removing it (3N − 6) would mis-state the
//!   ensemble the grid actually samples. With Maxwell–Boltzmann starts the question is moot in
//!   practice; without them the analytic test providers show the (3N − 3)/(3N − 6) overheating.
//!
//! # Random numbers
//! Gaussians by Marsaglia's polar method on `rand::rngs::StdRng` (CLAUDE.md rule 2: `rand` is
//! plumbing; no distribution crate). The χ²_{N_f − 1} is the literal sum of N_f − 1 squared
//! Gaussians: exact, ~N_f draws per step, microseconds next to the DFT solve that preceded them
//! (a Marsaglia–Tsang gamma draw would save that only in the thousands-of-atoms regime this
//! sandbox cannot reach). A fixed seed gives a bitwise-reproducible trajectory — within one
//! `rand` version: `StdRng`'s algorithm is explicitly not stable across `rand` releases, so if
//! M5's input-replay harness must replay across upgrades it should name `rand_chacha::ChaCha12Rng`
//! (today's `StdRng`) or record the `rand` version with the replay.
//!
//! # Degenerate inputs
//! Internal K = 0 (every velocity equal to the COM velocity): α is undefined (0 × ∞) and the direction of the new momentum is
//! not determined by a rescale, so `apply` does nothing; a rescaling thermostat cannot create
//! motion from nothing — [`Csvr::maxwell_boltzmann`] does (the driver calls it on enabling the
//! thermostat at rest, R7 above); otherwise the forces do on the next Verlet step, after which
//! the thermostat heats whatever they excited (see "Frozen modes"). T_target = 0: K̄ = 0,
//! α² = c, an exponential quench. τ = 0: c = e^{−∞} = 0, the full-resample limit. Invalid
//! live-edited parameters (τ < 0, T < 0, NaN, or dt = 0 with τ = 0, which makes dt/τ = NaN) and
//! a non-finite K leave the scene bitwise unchanged (and `debug_assert!`): no path writes NaN
//! into a velocity.
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
    /// `target_t` is validated by [`Csvr::new`] (≥ 0); a live edit to a negative value makes
    /// this negative and [`Csvr::apply`] refuse to act (no clamping here, so the two agree).
    pub fn target_kinetic(&self, dof: usize) -> f64 {
        0.5 * dof as f64 * units::kelvin_to_hartree(self.target_t)
    }

    /// The parameters `apply` can act on: dt/τ a non-negative number (τ ≥ 0, dt ≥ 0, not both
    /// zero, neither NaN; dt/τ = +∞ is the τ = 0 limit and allowed) and T ≥ 0. Debug-asserted
    /// and, in release, the reason `apply` returns with the scene untouched.
    fn valid(&self, dt: f64) -> bool {
        let x = dt / self.tau;
        let ok = x >= 0.0 && self.target_t >= 0.0;
        debug_assert!(ok, "Csvr::apply: invalid parameters dt = {dt}, tau = {}, target_t = {}", self.tau, self.target_t);
        ok
    }

    /// One CSVR step of length `dt` (atomic time units) on the scene's internal velocities:
    /// v ← u + α (v − u) with u the centre-of-mass velocity (N ≥ 2; 0 for N = 1) and α the
    /// signed factor of the module doc, drawn with this thermostat's generator from the internal
    /// K = Σ ½ m |v − u|². Afterwards the COM velocity is u (to rounding) and the internal K is
    /// the canonical kinetic energy for `Scene::dof` degrees of freedom. A no-op when the
    /// internal K = 0 or α is exactly 1 (bitwise: τ → ∞ is the identity), and bitwise untouched
    /// for invalid parameters or a non-finite K (module doc, "Degenerate inputs").
    pub fn apply(&mut self, scene: &mut Scene, dt: f64) {
        if scene.nuclei.is_empty() || !self.valid(dt) {
            return;
        }
        // the thermostat's frame: the one `Scene::temperature` reports (N = 1 keeps its motion)
        let u = if scene.nuclei.len() >= 2 { scene.com_velocity() } else { [0.0; 3] };
        let k = crate::scene::neumaier(scene.nuclei.iter().map(|n| {
            let d: [f64; 3] = std::array::from_fn(|a| n.vel[a] - u[a]);
            0.5 * n.mass * (d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        }));
        if !k.is_finite() || k <= 0.0 {
            return; // nothing to rescale (direction undefined), or garbage in: never NaN out
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
        if alpha == 1.0 {
            return; // exactly the identity: (v − u) + u need not round back to v
        }
        if u == [0.0; 3] {
            // N = 1 (or an exactly resting COM): a plain rescale, which keeps signed zeros
            for n in &mut scene.nuclei {
                for v in &mut n.vel {
                    *v *= alpha;
                }
            }
        } else {
            for n in &mut scene.nuclei {
                for (v, ua) in n.vel.iter_mut().zip(u) {
                    *v = ua + alpha * (*v - ua);
                }
            }
        }
    }

    /// Maxwell–Boltzmann velocities at `target_t` (ruling R7, module doc "Frozen modes"): every
    /// component v_{i,a} ~ N(0, k_B T/m_i) from this thermostat's generator, the centre-of-mass
    /// velocity removed (N ≥ 2: a fresh start, the COM at rest — unlike `apply`, which preserves
    /// it), then one common rescale so that K is *exactly*
    /// K̄ = N_f k_B T/2 for `Scene::dof` — the readout starts at the target rather than at a
    /// random canonical draw (the ensemble is restored by the first few `apply`s). For equal
    /// masses the resulting per-component variance is k_B T/m_i · (1 − 1/N): the COM projection
    /// removes 3 of the 3N components' worth and the rescale fixes the total to the 3N − 3
    /// internal share (unequal masses: k_B T/m_i · (1 − m_i/M) before the common rescale). The driver
    /// calls this when the thermostat is switched on with the system at rest (K < 1 % of K̄); it
    /// is a method on `Csvr` so one seed reproduces the whole thermostatted run, draw included.
    /// T = 0 gives every velocity exactly 0; an empty scene is untouched.
    pub fn maxwell_boltzmann(&mut self, scene: &mut Scene) {
        // NaN counts as invalid, hence the explicit `is_nan()` rather than `!(t >= 0)`
        if scene.nuclei.is_empty() || self.target_t.is_nan() || self.target_t < 0.0 {
            debug_assert!(scene.nuclei.is_empty(), "Csvr::maxwell_boltzmann: target_t = {} K", self.target_t);
            return;
        }
        let kbt = units::kelvin_to_hartree(self.target_t);
        for n in &mut scene.nuclei {
            let sigma = (kbt / n.mass).sqrt();
            for v in &mut n.vel {
                *v = sigma * self.gaussian();
            }
        }
        if scene.nuclei.len() >= 2 {
            scene.remove_com_velocity();
        }
        let k = scene.kinetic_energy();
        if k > 0.0 {
            let alpha = (self.target_kinetic(scene.dof()) / k).sqrt();
            for n in &mut scene.nuclei {
                for v in &mut n.vel {
                    *v *= alpha;
                }
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
        standard_normal(&mut self.rng)
    }
}

/// A standard normal by Marsaglia's polar method (the thermostat's draws and the sampling
/// pass's perturbations share it).
pub(crate) fn standard_normal(rng: &mut StdRng) -> f64 {
    loop {
        let u = 2.0 * rng.gen::<f64>() - 1.0;
        let v = 2.0 * rng.gen::<f64>() - 1.0;
        let s = u * u + v * v;
        if s > 0.0 && s < 1.0 {
            return u * (-2.0 * s.ln() / s).sqrt();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrator::Integrator;
    use crate::provider::ForceProvider;
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
    /// K = the internal kinetic energy Σ ½ m |v − u|² (u the COM velocity for N ≥ 2, which
    /// `apply` preserves and the wells move), T = `Scene::temperature`.
    fn run(s: &mut Scene, p: &mut Harmonic, csvr: &mut Csvr, integ: &Integrator, steps: usize, discard: usize) -> Vec<(f64, f64)> {
        let mut f = p.evaluate(s, &mut || false).expect("harmonic").forces;
        let mut log = Vec::with_capacity(steps - discard);
        for i in 0..steps {
            f = integ.step(s, &f, p, &mut |_, _| {}, &mut || false).expect("harmonic never fails").0;
            csvr.apply(s, integ.dt);
            if i >= discard {
                log.push((internal_kinetic(s), s.temperature()));
            }
        }
        log
    }

    /// Σ ½ m |v − u|², u = the COM velocity for N ≥ 2 (the thermostat's K).
    fn internal_kinetic(s: &Scene) -> f64 {
        let u = if s.nuclei.len() >= 2 { s.com_velocity() } else { [0.0; 3] };
        crate::scene::neumaier(s.nuclei.iter().map(|n| (0..3).map(|a| 0.5 * n.mass * (n.vel[a] - u[a]).powi(2)).sum::<f64>()))
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
    /// means over √(number of blocks). With blocks of ~6 correlation times the estimate is biased
    /// low by ~10 % (the callers' margins are quoted with that in mind).
    fn block_std_err(x: &[f64], block: usize) -> f64 {
        let means: Vec<f64> = x.chunks_exact(block).map(mean).collect();
        (variance(&means) / means.len() as f64).sqrt()
    }

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
        // T and K are the same statistic (T = 2K/(N_f k_B), K the internal kinetic energy)
        assert!((mt - units::hartree_to_kelvin(2.0 * mk / 189.0)).abs() < 1e-9 * mt);
        // the 3 % bound is a real margin: at least 2.5× the measured standard error
        assert!(0.03 * 300.0 > 2.5 * se_t, "3 % bound is only {:.1}σ", 0.03 * 300.0 / se_t);
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
    /// scene whose COM velocity is exactly zero and on one with a COM velocity (R10: the COM is
    /// preserved, and α = 1 touches nothing).
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
        // with a COM velocity: bitwise unchanged too
        let start = scene(vec![Nucleus::new(8, [0.0; 3]).with_vel([1e-4, 2e-4, -3e-4]),
                               Nucleus::new(1, [1.43, 1.11, 0.0]).with_vel([3e-4, -1e-3, 2e-4]),
                               Nucleus::new(1, [-1.43, 1.11, 0.0]).with_vel([-2e-4, 5e-4, 7e-4])]);
        assert!(start.com_velocity() != [0.0; 3]);
        let mut moving = start.clone();
        let mut c = Csvr::new(300.0, 1e30, 1);
        for _ in 0..10 { c.apply(&mut moving, 10.0); }
        assert!(moving == start, "τ = ∞ with a COM velocity:\n{moving:?}\nvs\n{start:?}");
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
    /// The start is a *generic* displacement. A displacement along one axis alone leaves
    /// v_y = v_z = 0 exactly forever — zero force, and a rescale of 0 is 0 — an invariant set the
    /// thermostat cannot leave: CSVR controls the total K and relies on the dynamics for
    /// equipartition between modes. The same mechanism freezes the rigid rotation (L = 0) of
    /// every molecule started from rest under torque-free forces, which is why the driver starts
    /// thermostatted runs with `maxwell_boltzmann` (module doc, "Frozen modes"). Second half of
    /// the test: the frozen case is pinned to its own derived value. Phase-averaging the
    /// K-process over one active 1D mode with N_f = 3 in the formula
    /// (dE = (K̄ − E/2) dt/τ + √(2 E K̄/(N_f τ)) dW, K ≈ E sin²φ on average over φ) gives the
    /// stationary E ~ Gamma(N_f, k_B T), so ⟨K⟩ = ⟨E⟩/2 = K̄ — the temperature readout is still
    /// right — while Var K/⟨K⟩² = ⟨E²⟩⟨sin⁴φ⟩/K̄² − 1 = (1 + 1/N_f)(2K̄)²(3/8)/K̄² − 1 =
    /// (4/3)(4)(3/8) − 1 = 1, not 2/3: the fingerprint exposes the frozen modes even though T
    /// does not. The averaging assumes ωτ ≫ 1; here ωτ ≈ 6.8 (period 9.2 fs vs τ = 10 fs), so
    /// 1.000 is the leading order with O((ωτ)⁻²) ≈ 2 % corrections — invisible at the 25 % bound
    /// (measured 1.006).
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

    /// Velocities are rescaled about the centre of mass (R10): a zero COM velocity stays zero (to
    /// rounding) through many applies, and a COM velocity u present before `apply` survives it
    /// unchanged — the velocities are then u plus those of the COM-free scene rescaled with the
    /// same draws (to rounding) — while the internal temperature is controlled.
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
            assert!((com_m[a] - u[a]).abs() < floor, "COM velocity not preserved: {com_m:?} vs {u:?}");
        }
        for (x, y) in internal.nuclei.iter().zip(&moving.nuclei) {
            for ((xv, yv), uk) in x.vel.iter().zip(&y.vel).zip(u) {
                assert!((xv + uk - yv).abs() < 1e-14 * v_th, "{:?} + u vs {:?}", x.vel, y.vel);
            }
        }
        assert!((internal.temperature() - moving.temperature()).abs() < 1e-10 * internal.temperature());
        // the rescale changed the kinetic energy (τ = 1 fs, dt = 10 a.u. = 0.24 fs is a strong kick)
        assert!(internal.kinetic_energy() > 0.0);
        // many applies: the COM stays put
        let mut c = Csvr::new(300.0, 1.0, 9);
        for _ in 0..1000 { c.apply(&mut internal, 10.0); }
        let com = internal.com_velocity();
        assert!(com.iter().all(|x| x.abs() < floor), "COM drifted over 1000 applies: {com:?}");
        // a uniform COM velocity survives 1000 applies while T is held at the target
        let mut c = Csvr::new(300.0, 1.0, 9);
        let mut temps = Vec::new();
        for _ in 0..1000 {
            c.apply(&mut moving, 10.0);
            temps.push(moving.temperature());
        }
        let com = moving.com_velocity();
        println!("after 1000 applies: COM {com:?} (u {u:?}), ⟨T⟩ {:.1} K", mean(&temps));
        assert!((0..3).all(|a| (com[a] - u[a]).abs() < 10.0 * floor), "COM velocity changed: {com:?} vs {u:?}");
        // a static chain at dt/τ = 0.41: correlation ≈ 2.5 applies, Var T/T² = 2/189 → std err
        // of ⟨T⟩ over 1000 applies ≈ √(0.0106 · 5 / 1000) = 0.23 %; the 3 % bound is > 10σ
        assert!((mean(&temps) / 300.0 - 1.0).abs() < 0.03, "⟨T⟩ = {}", mean(&temps));
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
        hot.maxwell_boltzmann(&mut empty);
        assert!(std::panic::catch_unwind(|| Csvr::new(-1.0, 10.0, 1)).is_err());
        assert!(std::panic::catch_unwind(|| Csvr::new(300.0, -1.0, 1)).is_err());
        // dt = 0 with τ > 0 is a valid identity step (α = 1 exactly; the COM velocity is kept)
        let mut still = s.clone();
        let before = still.clone();
        Csvr::new(300.0, 10.0, 1).apply(&mut still, 0.0);
        assert!(still == before, "dt = 0 must be identity");
    }

    /// No path writes NaN into a velocity (F2): live-edited τ < 0, T < 0, NaN τ, dt = 0 with
    /// τ = 0 (dt/τ = NaN) and dt < 0 all leave the scene bitwise untouched (release; the
    /// `debug_assert!` fires in debug, so this part runs in release only), and a NaN K is not
    /// propagated. τ = +∞ is the identity, dt = +∞ with finite τ the full resample.
    #[test]
    fn invalid_parameters_leave_the_scene_untouched() {
        // equal masses, exactly opposite velocities: Σ m v is exactly 0, so a repeated COM
        // removal is an exact no-op and "untouched" can be asserted bitwise
        let v = [3e-4, -2e-4, 5e-4];
        let mut s = scene(vec![Nucleus::new(1, [0.7, 0.0, 0.0]).with_vel(v), Nucleus::new(1, [-0.7, 0.0, 0.0]).with_vel([-v[0], -v[1], -v[2]])]);
        assert_eq!(s.com_velocity(), [0.0; 3]);
        let before = s.clone();
        if !cfg!(debug_assertions) {
            for (tau, t, dt) in [(-1.0, 300.0, 10.0), (10.0, -1.0, 10.0), (f64::NAN, 300.0, 10.0), (0.0, 300.0, 0.0), (10.0, 300.0, -1.0), (10.0, f64::NAN, 10.0)] {
                let mut c = Csvr::new(300.0, 10.0, 1);
                c.tau = tau;
                c.target_t = t;
                c.apply(&mut s, dt);
                assert!(s == before, "τ = {tau}, T = {t}, dt = {dt} changed the scene: {s:?}");
            }
            // T < 0 after construction: target_kinetic is negative (no clamp) and apply refuses
            let mut c = Csvr::new(300.0, 10.0, 1);
            c.target_t = -5.0;
            assert!(c.target_kinetic(3) < 0.0);
            c.apply(&mut s, 10.0);
            assert!(s == before);
        }
        // τ = +∞: identity; dt = +∞: c = 0, a finite full resample
        let mut c = Csvr::new(300.0, f64::INFINITY, 1);
        c.apply(&mut s, 10.0);
        assert!(s == before);
        let mut c = Csvr::new(300.0, 10.0, 1);
        c.apply(&mut s, f64::INFINITY);
        assert!(s.nuclei.iter().all(|n| n.vel.iter().all(|v| v.is_finite())));
        assert!(s != before);
        // a NaN velocity component in: K = NaN, nothing is written — the scene is bitwise as it
        // came in (the NaN is not spread to the other nuclei)
        let mut bad = before.clone();
        bad.nuclei[0].vel[0] = f64::NAN;
        let bits = |s: &Scene| s.nuclei.iter().flat_map(|n| n.vel.map(f64::to_bits)).collect::<Vec<u64>>();
        let want = bits(&bad);
        let mut c = Csvr::new(300.0, 10.0, 1);
        c.apply(&mut bad, 10.0);
        assert_eq!(bits(&bad), want);
    }

    /// `maxwell_boltzmann` (R7): on 64 H at 300 K the readout is exactly the target (K = K̄ to
    /// rounding → T to 1e-9 relative), Σ m v = 0 (|u| < 1e-15), and over 1000 draws the 192 000
    /// components are Gaussian: mean 0, variance k_B T/m · (1 − 1/N) (the COM projection removes
    /// 3 of 3N components' worth and the rescale fixes the total to the 3N − 3 share — 1.6 % below
    /// k_B T/m for N = 64, asserted to 1 %, std err 0.3 %), ⟨z³⟩ = 0, ⟨z⁴⟩ = 3 (to 0.1; a point on a
    /// 189-sphere is Gaussian to O(1/N_f)). Same seed → same draw; N = 1 → K = 3k_BT/2 in the lab
    /// frame; T = 0 → all zero.
    #[test]
    fn maxwell_boltzmann_starts_at_the_target() {
        let (template, _) = lattice_64();
        let mut c = Csvr::new(300.0, 10.0, 77);
        let m = template.nuclei[0].mass;
        let sigma = (units::kelvin_to_hartree(300.0) / m).sqrt();
        let mut z: Vec<f64> = Vec::with_capacity(192_000);
        let mut worst_t: f64 = 0.0;
        let mut worst_u: f64 = 0.0;
        for _ in 0..1000 {
            let mut s = template.clone();
            c.maxwell_boltzmann(&mut s);
            worst_t = worst_t.max((s.temperature() / 300.0 - 1.0).abs());
            let u = s.com_velocity();
            worst_u = worst_u.max(u.iter().fold(0.0f64, |a, x| a.max(x.abs())));
            z.extend(s.nuclei.iter().flat_map(|n| n.vel.iter().map(|v| v / sigma)));
        }
        let m1 = mean(&z);
        let m2 = mean(&z.iter().map(|x| x * x).collect::<Vec<_>>());
        let m3 = mean(&z.iter().map(|x| x * x * x).collect::<Vec<_>>());
        let m4 = mean(&z.iter().map(|x| x.powi(4)).collect::<Vec<_>>());
        let var_expect = 1.0 - 1.0 / 64.0;
        println!("Maxwell–Boltzmann 64 H × 1000: worst |T/300 − 1| = {worst_t:.1e}, worst |u| = {worst_u:.1e}; z moments {m1:+.4}, {m2:.4} (expect {var_expect:.4}), {m3:+.4}, {m4:.3}");
        assert!(worst_t < 1e-9, "T off target by {worst_t:.1e}");
        assert!(worst_u < 1e-15, "COM velocity {worst_u:.1e}");
        assert!(m1.abs() < 0.01, "mean {m1}");
        assert!((m2 / var_expect - 1.0).abs() < 0.01, "variance {m2} vs {var_expect}");
        assert!(m3.abs() < 0.03, "skewness {m3}");
        assert!((m4 / (3.0 * var_expect * var_expect) - 1.0).abs() < 0.03, "kurtosis {m4}");
        // determinism: same seed, same draw
        let (mut a, mut b) = (template.clone(), template.clone());
        Csvr::new(300.0, 10.0, 5).maxwell_boltzmann(&mut a);
        Csvr::new(300.0, 10.0, 5).maxwell_boltzmann(&mut b);
        assert!(a == b);
        // N = 1: lab frame, K = 3 k_B T/2 exactly to rounding
        let mut one = scene(vec![Nucleus::new(8, [0.0; 3])]);
        c.maxwell_boltzmann(&mut one);
        assert!((one.kinetic_energy() / c.target_kinetic(3) - 1.0).abs() < 1e-12);
        assert!((one.temperature() / 300.0 - 1.0).abs() < 1e-9);
        // T = 0: at rest
        let mut cold = Csvr::new(0.0, 10.0, 1);
        let mut s = template.clone();
        cold.maxwell_boltzmann(&mut s);
        assert!(s.nuclei.iter().all(|n| n.vel == [0.0; 3]));
        // the thermostatted run then starts at the target instead of climbing from 0 K. Wells
        // at the nuclei (the remedy's case: a hand-placed scene near its minimum, V₀ = 0, F = 0):
        // the first step's first half-kick is empty, the drift makes F = −k dt v, and the second
        // half-kick changes K by exactly −h² = −(ω dt)² = −2.9 %; the apply adds ~√(1−c)·√(2/N_f)
        // ≈ 1.6 % of noise — so T after step 1 is 291 ± 5 K and the 10 % bound is ≈ 4σ. K then
        // swings toward E/2 = K̄/2 within a quarter period (the positions start cold) before τ
        // pulls it back — the band is printed, not pinned
        let (mut s, _) = lattice_64();
        let mut p = Harmonic::at_rest_positions(&s, 0.5);
        let mut c = Csvr::new(300.0, 10.0, 7);
        c.maxwell_boltzmann(&mut s);
        let h2 = 0.5 / s.nuclei[0].mass * Integrator::new_fs(0.25).dt.powi(2);
        let log = run(&mut s, &mut p, &mut c, &Integrator::new_fs(0.25), 100, 0);
        let (lo, hi) = log.iter().fold((f64::MAX, 0.0f64), |(lo, hi), (_, t)| (lo.min(*t), hi.max(*t)));
        println!("after the draw: T = {:.1} K after step 1 (expect 300 (1 − h²) = {:.1} ± 5); first 100 steps T ∈ [{lo:.1}, {hi:.1}] K", log[0].1, 300.0 * (1.0 - h2));
        assert!((log[0].1 / 300.0 - 1.0).abs() < 0.10, "T after one step = {} K", log[0].1);
        assert!((log[0].1 / (300.0 * (1.0 - h2)) - 1.0).abs() < 0.06, "T after one step = {} K vs {}", log[0].1, 300.0 * (1.0 - h2));
    }
}
