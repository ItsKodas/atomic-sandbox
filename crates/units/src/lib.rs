//! Atomic units are the internal unit system (ħ = mₑ = e = 1).
//! Constants: CODATA 2018. Isotope masses: AME2020, most abundant isotope.

pub const BOHR_IN_ANGSTROM: f64 = 0.529177210903;
pub const HARTREE_IN_EV: f64 = 27.211386245988;
pub const AU_TIME_IN_FS: f64 = 0.024188843265857;
pub const KB_HARTREE_PER_K: f64 = 3.166811563e-6;
pub const AMU_IN_ME: f64 = 1822.888486209;

pub fn bohr_to_angstrom(x: f64) -> f64 { x * BOHR_IN_ANGSTROM }
pub fn angstrom_to_bohr(x: f64) -> f64 { x / BOHR_IN_ANGSTROM }
pub fn hartree_to_ev(e: f64) -> f64 { e * HARTREE_IN_EV }
pub fn ev_to_hartree(e: f64) -> f64 { e / HARTREE_IN_EV }
pub fn au_time_to_fs(t: f64) -> f64 { t * AU_TIME_IN_FS }
pub fn fs_to_au_time(t: f64) -> f64 { t / AU_TIME_IN_FS }
pub fn kelvin_to_hartree(t: f64) -> f64 { t * KB_HARTREE_PER_K }
pub fn hartree_to_kelvin(e: f64) -> f64 { e / KB_HARTREE_PER_K }
pub fn amu_to_au_mass(m: f64) -> f64 { m * AMU_IN_ME }

/// Most-abundant-isotope mass in amu (AME2020). Z = 1..=10 for M1; extend as elements are validated.
pub fn isotope_mass_amu(z: u8) -> Option<f64> {
    Some(match z {
        1 => 1.00782503223,
        2 => 4.00260325413,
        3 => 7.0160034366,
        4 => 9.012183065,
        5 => 11.00930536,
        6 => 12.0,
        7 => 14.00307400443,
        8 => 15.99491461957,
        9 => 18.99840316273,
        10 => 19.9924401762,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn bohr_angstrom_roundtrip() {
        assert_relative_eq!(bohr_to_angstrom(1.0), 0.529177210903, epsilon = 1e-12);
        assert_relative_eq!(angstrom_to_bohr(bohr_to_angstrom(2.5)), 2.5, epsilon = 1e-12);
    }

    #[test]
    fn hartree_ev() {
        assert_relative_eq!(hartree_to_ev(1.0), 27.211386245988, epsilon = 1e-9);
        assert_relative_eq!(ev_to_hartree(13.605693122994), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn time_and_temperature() {
        assert_relative_eq!(au_time_to_fs(1.0), 0.024188843265857, epsilon = 1e-15);
        assert_relative_eq!(fs_to_au_time(0.25), 10.335, epsilon = 1e-3);
        assert_relative_eq!(kelvin_to_hartree(300.0), 9.5004e-4, epsilon = 1e-7);
    }

    #[test]
    fn masses() {
        assert_relative_eq!(isotope_mass_amu(1).unwrap(), 1.00782503223, epsilon = 1e-10);
        assert_relative_eq!(isotope_mass_amu(8).unwrap(), 15.99491461957, epsilon = 1e-10);
        assert!(isotope_mass_amu(0).is_none());
        assert_relative_eq!(amu_to_au_mass(1.0), 1822.888486209, epsilon = 1e-6);
    }
}
