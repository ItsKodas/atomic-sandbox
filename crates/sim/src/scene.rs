//! The dynamical state: classical nuclei (position, velocity, Z, A, mass) on a cubic grid cell.
//! Atomic units throughout (CLAUDE.md rule 3): positions in Bohr, velocities in Bohr per atomic
//! time unit (ħ/E_h ≈ 0.0242 fs), masses in electron masses mₑ, energies in Hartree; kelvin
//! appears only as the output of [`Scene::temperature`], converted through `units`.

use crate::provider::ProviderError;

/// One classical nucleus. `mass` is the isotope mass in mₑ (the only per-element datum besides
/// Z, CLAUDE.md rule 1); `a` is the mass number it was taken from — bookkeeping for the badge,
/// the scene file and the nuclear-events milestone (M6 spec §5.1), never read by the physics.
#[derive(Clone, Debug, PartialEq)]
pub struct Nucleus {
    pub pos: [f64; 3],
    pub vel: [f64; 3],
    pub z: u8,
    pub a: u16,
    pub mass: f64,
}

impl Nucleus {
    /// A nucleus at rest with the most abundant isotope's mass (`units::isotope_mass_amu`,
    /// AME2020) in mₑ: the neutral atom's mass, electrons included — ¹H is 1837.15 mₑ, not the
    /// bare proton's 1836.15. The ~Z mₑ (≤ 0.05 % for H, less for heavier elements) is the
    /// convention of the tabulated masses; Born-Oppenheimer MD moves the nuclei with their
    /// electrons, so the atomic mass is the usual choice. Panics for a Z `units::isotopes` does not
    /// list (`units::most_abundant_a` is `None`).
    pub fn new(z: u8, pos: [f64; 3]) -> Nucleus {
        let a = units::most_abundant_a(z).expect("a Z in the isotope table");
        Nucleus::isotope(z, a, pos).expect("the most abundant isotope is listed")
    }

    /// A nucleus at rest of isotope (Z, A) with the table's mass (`units::isotope_mass_amu_a`,
    /// the same atomic-mass convention as [`new`](Self::new)); an unlisted A is refused with the
    /// listed ones named: `Err("no mass for ⁵H; listed: ¹H ²H ³H")`.
    pub fn isotope(z: u8, a: u16, pos: [f64; 3]) -> Result<Nucleus, String> {
        let amu = units::isotope_mass_amu_a(z, a).ok_or_else(|| units::isotopes::unlisted_isotope_message(z, a))?;
        Ok(Nucleus { pos, vel: [0.0; 3], z, a, mass: units::amu_to_au_mass(amu) })
    }

    /// N = A − Z (display data for the badge; 0 if A < Z, which the table never gives).
    pub fn neutrons(&self) -> u16 {
        self.a.saturating_sub(self.z as u16)
    }

    pub fn with_vel(self, vel: [f64; 3]) -> Nucleus {
        Nucleus { vel, ..self }
    }
}

/// The real-space grid the electrons live on: n³ points spaced `h` (Bohr), centred on the
/// origin (`qm::Grid::cube`), box edge (n − 1)·h.
#[derive(Clone, Debug, PartialEq)]
pub struct Cell {
    pub n: usize,
    pub h: f64,
}

/// Nuclei must stay at least this many grid steps from every cell face (ruling R12): the grid —
/// and with it the electron density — ends at the face, so closer in the nucleus's density is
/// cut off and its forces are artefacts of the box (an H 1s density 3h = 0.75 Bohr out is
/// e^{-1.5} ≈ 22 % of its peak at h = 0.25). The force providers refuse such a scene
/// (`ProviderError::Invalid`), which pauses the dynamics with the last good state.
pub const CELL_MARGIN_H: f64 = 3.0;

impl Cell {
    /// Half the box edge, (n − 1)·h / 2: the grid spans [−half, half]³ (`qm::Grid::cube`).
    pub fn half_extent(&self) -> f64 {
        (self.n as f64 - 1.0) * self.h / 2.0
    }

    /// The largest |coordinate| a nucleus may have: `half_extent − CELL_MARGIN_H·h`.
    pub fn max_coord(&self) -> f64 {
        self.half_extent() - CELL_MARGIN_H * self.h
    }
}

/// Everything a force evaluation and a time step need. `multiplicity` = 2S + 1 (`None`: lowest,
/// `qm::System::new`); `sigma`: nuclear smearing width in Bohr (`None`: h). `external_field` is
/// an **acceleration**, not an electric field: a uniform a (Bohr per atomic time unit², the same
/// for every nucleus whatever its mass or charge) added by the integrator — the gravity hook
/// (g = 9.8 m/s² is ≈ 1.1e-22 a.u.). Force providers ignore it: it acts on neither the electrons
/// nor through the charges, so a uniform *electric* field (q_I E/m_I per nucleus plus a term in
/// the Kohn-Sham potential) cannot be expressed with it. The name is kept (final review A-1).
#[derive(Clone, Debug, PartialEq)]
pub struct Scene {
    pub nuclei: Vec<Nucleus>,
    pub cell: Cell,
    pub charge: i32,
    pub multiplicity: Option<u32>,
    pub sigma: Option<f64>,
    pub external_field: [f64; 3],
}

impl Scene {
    /// N = ΣZ − charge (negative for an impossible charge; the providers reject it).
    pub fn electrons(&self) -> i64 {
        self.nuclei.iter().map(|n| n.z as i64).sum::<i64>() - self.charge as i64
    }

    /// Σ m (mₑ).
    pub fn total_mass(&self) -> f64 {
        self.nuclei.iter().map(|n| n.mass).sum()
    }

    /// Σ m v / Σ m; zero for an empty scene.
    pub fn com_velocity(&self) -> [f64; 3] {
        let m = self.total_mass();
        if m <= 0.0 { return [0.0; 3]; }
        std::array::from_fn(|a| self.nuclei.iter().map(|n| n.mass * n.vel[a]).sum::<f64>() / m)
    }

    /// Subtracts the centre-of-mass velocity from every nucleus (total momentum → 0).
    pub fn remove_com_velocity(&mut self) {
        let u = self.com_velocity();
        for n in &mut self.nuclei {
            for (v, du) in n.vel.iter_mut().zip(u) { *v -= du; }
        }
    }

    /// Σ ½ m |v|² (Hartree), centre-of-mass motion included — the kinetic term of the conserved
    /// total energy. Compensated sum (CLAUDE.md rule 4): the energy-drift diagnostic sits on
    /// differences of this number, so its rounding must not grow with N.
    pub fn kinetic_energy(&self) -> f64 {
        neumaier(self.nuclei.iter().map(|n| 0.5 * n.mass * (n.vel[0] * n.vel[0] + n.vel[1] * n.vel[1] + n.vel[2] * n.vel[2])))
    }

    /// Degrees of freedom of the internal motion: 3N − 3 (centre of mass removed) for N ≥ 2;
    /// 3 for a single nucleus, whose translation is all the motion there is — and 3 for an empty
    /// scene too, so that T = 0 / 3 = 0 stays finite there (nothing moves; the app shows "—").
    pub fn dof(&self) -> usize {
        if self.nuclei.len() >= 2 { 3 * self.nuclei.len() - 3 } else { 3 }
    }

    /// Instantaneous kinetic temperature (K) by equipartition, T = 2 K / (dof · k_B), with K the
    /// kinetic energy in the centre-of-mass frame for N ≥ 2 (the lab-frame value for N ≤ 1,
    /// matching `dof`). Rotation is not removed (an isolated molecule's 3N − 6 vs 3N − 3 is a
    /// convention; this one counts rotation as thermal).
    pub fn temperature(&self) -> f64 {
        units::hartree_to_kelvin(2.0 * self.internal_kinetic_energy() / self.dof() as f64)
    }

    /// The kinetic energy of the internal motion (Hartree), Σ ½ m |v − u|² with u the
    /// centre-of-mass velocity, for N ≥ 2; the lab-frame [`kinetic_energy`](Self::kinetic_energy)
    /// for N ≤ 1 (matching [`dof`](Self::dof)). The K of [`temperature`](Self::temperature), of
    /// the thermostat's frame and of the driver's "at rest" test (R7).
    pub fn internal_kinetic_energy(&self) -> f64 {
        let u = if self.nuclei.len() >= 2 { self.com_velocity() } else { [0.0; 3] };
        // Σ ½ m |v − u|² directly (not K − ½ M |u|², which cancels when the drift dominates)
        neumaier(self.nuclei.iter().map(|n| {
            let d: [f64; 3] = std::array::from_fn(|a| n.vel[a] - u[a]);
            0.5 * n.mass * (d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        }))
    }

    /// R12: every nucleus at least [`CELL_MARGIN_H`]·h inside every face, finite
    /// (`crate::provider` module doc, "Errors"): the Kohn-Sham providers refuse a scene that is not
    /// with this `ProviderError::Invalid`, before any solve.
    pub fn check_inside_cell(&self) -> Result<(), ProviderError> {
        let max = self.cell.max_coord();
        for (i, n) in self.nuclei.iter().enumerate() {
            let worst = n.pos.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            // (`f64::max` skips NaN, so non-finite coordinates are tested explicitly)
            if worst > max || n.pos.iter().any(|x| !x.is_finite()) {
                let half = self.cell.half_extent();
                let place = if worst > half {
                    format!("{:.2} Bohr outside its face", worst - half)
                } else {
                    format!("{:.2} Bohr from its face", half - worst)
                };
                return Err(ProviderError::Invalid(format!(
                    "nucleus {i} ({}) left the simulation cell: {place}, closer than {CELL_MARGIN_H}h = {:.2} Bohr — the density is cut off there, so its forces would be artefacts of the box (the scene is the last good step: move the nucleus back inside or rewind; Play alone retries the same step)",
                    units::isotopes::symbol(n.z),
                    CELL_MARGIN_H * self.cell.h
                )));
            }
        }
        Ok(())
    }
}

/// Neumaier-compensated summation (the same algorithm as `qm::grid`'s private helper): the
/// error is O(ε) of the result instead of O(N ε) of the partial sums. Used for every energy
/// that `sim` sums over nuclei (kinetic, internal kinetic, the analytic test potentials).
pub(crate) fn neumaier(iter: impl Iterator<Item = f64>) -> f64 {
    let (mut sum, mut comp) = (0.0f64, 0.0f64);
    for x in iter {
        let t = sum + x;
        if sum.abs() >= x.abs() { comp += (sum - t) + x; } else { comp += (x - t) + sum; }
        sum = t;
    }
    sum + comp
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene(nuclei: Vec<Nucleus>) -> Scene {
        Scene { nuclei, cell: Cell { n: 33, h: 0.25 }, charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3] }
    }

    #[test]
    fn nucleus_mass_is_isotope_mass_in_au() {
        // ¹H: the atom's isotope mass 1.00782503223 amu × 1822.888486 = 1837.15 mₑ (not the bare
        // proton, 1836.15); ¹⁶O: 15.99491461957 amu → 29156.9 mₑ
        let h = Nucleus::new(1, [0.0; 3]);
        let o = Nucleus::new(8, [0.0; 3]);
        println!("m(¹H) = {:.4} mₑ, m(¹⁶O) = {:.3} mₑ", h.mass, o.mass);
        assert!((h.mass - 1837.15).abs() < 0.05, "m(¹H) = {}", h.mass);
        assert!((o.mass - 29156.9).abs() < 0.5, "m(¹⁶O) = {}", o.mass);
        assert_eq!(h.vel, [0.0; 3]);
        assert_eq!(h.clone().with_vel([1.0, 2.0, 3.0]).vel, [1.0, 2.0, 3.0]);
        assert_eq!(h.clone().with_vel([1.0, 2.0, 3.0]).mass, h.mass);
    }

    /// `Nucleus::isotope` takes the mass of the listed isotope (²H: 2.014101777844 u, AME2020),
    /// records A and N = A − Z, and refuses an unlisted A naming the listed ones; `new` is the
    /// most abundant isotope (¹⁶O for oxygen).
    #[test]
    fn isotope_constructor_takes_the_table_mass_and_refuses_unlisted_a() {
        let d = Nucleus::isotope(1, 2, [0.0; 3]).unwrap();
        assert_eq!((d.z, d.a, d.neutrons()), (1, 2, 1));
        assert!((d.mass - units::amu_to_au_mass(2.014101777844)).abs() < 1e-6, "m(²H) = {}", d.mass);
        assert_eq!(d.vel, [0.0; 3]);
        assert_eq!(Nucleus::new(8, [0.0; 3]).a, 16);
        assert_eq!(Nucleus::new(8, [0.0; 3]).neutrons(), 8);
        assert_eq!(Nucleus::new(1, [0.0; 3]), Nucleus::isotope(1, 1, [0.0; 3]).unwrap());
        assert_eq!(Nucleus::isotope(1, 5, [0.0; 3]).unwrap_err(), "no mass for ⁵H; listed: ¹H ²H ³H");
        assert!(Nucleus::isotope(11, 23, [0.0; 3]).unwrap_err().starts_with("no mass for"));
        let moved = d.clone().with_vel([1.0, 0.0, 0.0]);
        assert_eq!((moved.a, moved.mass), (2, d.mass));
    }

    #[test]
    fn temperature_from_equipartition_with_com_removed() {
        let v = 1e-3;
        let mut s = scene(vec![Nucleus::new(1, [0.7, 0.0, 0.0]).with_vel([v, 0.0, 0.0]),
                               Nucleus::new(1, [-0.7, 0.0, 0.0]).with_vel([-v, 0.0, 0.0])]);
        let m = s.nuclei[0].mass;
        let ke = 2.0 * 0.5 * m * v * v;
        assert_eq!(s.dof(), 3);
        assert!((s.kinetic_energy() - ke).abs() <= 1e-15 * ke);
        let t = 2.0 * ke / (3.0 * units::KB_HARTREE_PER_K);
        assert!((s.temperature() - t).abs() <= 1e-12 * t, "T = {} vs {t}", s.temperature());
        assert_eq!(s.com_velocity(), [0.0; 3]);
        assert!((s.total_mass() - 2.0 * m).abs() <= 1e-12 * m);
        assert_eq!(s.electrons(), 2);

        // a common COM velocity: KE grows, T (COM removed) does not
        let u = [2e-4, -3e-4, 5e-4];
        for n in &mut s.nuclei { for (v, du) in n.vel.iter_mut().zip(u) { *v += du; } }
        let com = s.com_velocity();
        for a in 0..3 { assert!((com[a] - u[a]).abs() < 1e-18, "COM {com:?} vs {u:?}"); }
        assert!(s.kinetic_energy() > ke);
        assert!((s.temperature() - t).abs() <= 1e-9 * t, "T with COM = {} vs {t}", s.temperature());
        s.remove_com_velocity();
        for a in 0..3 { assert!(s.com_velocity()[a].abs() < 1e-18); }
        assert!((s.kinetic_energy() - ke).abs() <= 1e-9 * ke);

        // one nucleus: DOF 3, its motion is not removed (it is all there is)
        let one = scene(vec![Nucleus::new(1, [0.0; 3]).with_vel([v, 0.0, 0.0])]);
        assert_eq!(one.dof(), 3);
        let t1 = 2.0 * (0.5 * m * v * v) / (3.0 * units::KB_HARTREE_PER_K);
        assert!((one.temperature() - t1).abs() <= 1e-12 * t1);

        // no nuclei (the `empty` preset): every quantity is defined and zero
        let empty = scene(Vec::new());
        assert_eq!((empty.dof(), empty.electrons(), empty.total_mass()), (3, 0, 0.0));
        assert_eq!((empty.kinetic_energy(), empty.internal_kinetic_energy(), empty.temperature()), (0.0, 0.0, 0.0));
        assert_eq!(empty.com_velocity(), [0.0; 3]);

        // N = 3: DOF 6; charge counts against the electron number
        let mut three = scene(vec![Nucleus::new(8, [0.0; 3]), Nucleus::new(1, [1.43, 1.11, 0.0]), Nucleus::new(1, [-1.43, 1.11, 0.0])]);
        assert_eq!(three.dof(), 6);
        three.charge = 1;
        assert_eq!(three.electrons(), 9);
    }
}
