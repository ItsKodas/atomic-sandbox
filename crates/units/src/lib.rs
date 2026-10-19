//! Atomic units are the internal unit system (ħ = mₑ = e = 1).
//! Constants: CODATA 2018. Isotope masses: AME2020 atomic masses (`isotopes` — the module doc
//! states the convention), the most abundant isotope by default.

pub mod elements;
pub mod isotopes;
pub use isotopes::{isotope_mass_amu, isotope_mass_amu_a, isotopes_of, most_abundant_a};

pub const BOHR_IN_ANGSTROM: f64 = 0.529177210903;
pub const HARTREE_IN_EV: f64 = 27.211386245988;
pub const AU_TIME_IN_FS: f64 = 0.024188843265857;
pub const KB_HARTREE_PER_K: f64 = 3.166811563e-6;
pub const AMU_IN_ME: f64 = 1822.888486209;
/// One atomic unit of electric dipole (e·a₀ = 8.4783536255e-30 C·m) in debye (1 D = 1e-21/c C·m,
/// exact): 2.541746473.
pub const AU_DIPOLE_IN_DEBYE: f64 = 2.541746473;

pub fn bohr_to_angstrom(x: f64) -> f64 { x * BOHR_IN_ANGSTROM }
pub fn angstrom_to_bohr(x: f64) -> f64 { x / BOHR_IN_ANGSTROM }
pub fn hartree_to_ev(e: f64) -> f64 { e * HARTREE_IN_EV }
pub fn ev_to_hartree(e: f64) -> f64 { e / HARTREE_IN_EV }
pub fn au_time_to_fs(t: f64) -> f64 { t * AU_TIME_IN_FS }
pub fn fs_to_au_time(t: f64) -> f64 { t / AU_TIME_IN_FS }
pub fn kelvin_to_hartree(t: f64) -> f64 { t * KB_HARTREE_PER_K }
pub fn hartree_to_kelvin(e: f64) -> f64 { e / KB_HARTREE_PER_K }
pub fn amu_to_au_mass(m: f64) -> f64 { m * AMU_IN_ME }
pub fn au_dipole_to_debye(d: f64) -> f64 { d * AU_DIPOLE_IN_DEBYE }

/// 0 °C in kelvin (exact, SI definition).
pub const ZERO_CELSIUS_IN_K: f64 = 273.15;
/// Display-edge temperature scales (rule 3: the simulation works in kelvin; these convert only
/// for the UI). °F = °C × 9/5 + 32, exact.
pub fn kelvin_to_celsius(t: f64) -> f64 { t - ZERO_CELSIUS_IN_K }
pub fn celsius_to_kelvin(t: f64) -> f64 { t + ZERO_CELSIUS_IN_K }
pub fn kelvin_to_fahrenheit(t: f64) -> f64 { (t - ZERO_CELSIUS_IN_K) * 9.0 / 5.0 + 32.0 }
pub fn fahrenheit_to_kelvin(t: f64) -> f64 { (t - 32.0) * 5.0 / 9.0 + ZERO_CELSIUS_IN_K }

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// The display scales: absolute zero is −273.15 °C = −459.67 °F; water boils (1 atm) at
    /// 373.15 K = 100 °C = 212 °F; −40 is the same in both; the round trips are exact to rounding.
    #[test]
    fn celsius_and_fahrenheit() {
        assert_relative_eq!(kelvin_to_celsius(0.0), -273.15);
        assert_relative_eq!(kelvin_to_fahrenheit(0.0), -459.67, epsilon = 1e-10);
        assert_relative_eq!(kelvin_to_celsius(373.15), 100.0, epsilon = 1e-12);
        assert_relative_eq!(kelvin_to_fahrenheit(373.15), 212.0, epsilon = 1e-10);
        assert_relative_eq!(kelvin_to_fahrenheit(celsius_to_kelvin(-40.0)), -40.0, epsilon = 1e-10);
        assert_relative_eq!(fahrenheit_to_kelvin(-459.67), 0.0, epsilon = 1e-10);
        for t in [0.0, 1.0, 300.0, 5000.0] {
            assert_relative_eq!(celsius_to_kelvin(kelvin_to_celsius(t)), t, epsilon = 1e-10);
            assert_relative_eq!(fahrenheit_to_kelvin(kelvin_to_fahrenheit(t)), t, epsilon = 1e-10);
        }
    }

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

    /// The most-abundant lookup is the AME2020 table (`isotopes` pins every entry); ¹H and ¹⁶O
    /// here at the table's own digits (the earlier AME2016 pins 1.00782503223 and 15.99491461957
    /// are 3.3e-10 u and 3.1e-10 u away).
    #[test]
    fn masses() {
        assert_relative_eq!(isotope_mass_amu(1).unwrap(), 1.007825031898, epsilon = 1e-12);
        assert_relative_eq!(isotope_mass_amu(8).unwrap(), 15.99491461926, epsilon = 1e-12);
        assert!(isotope_mass_amu(0).is_none());
        assert_relative_eq!(amu_to_au_mass(1.0), 1822.888486209, epsilon = 1e-6);
    }
}
