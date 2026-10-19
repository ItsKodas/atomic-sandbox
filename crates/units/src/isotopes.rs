//! The isotope mass table: AME2020 — Wang, Huang, Kondev, Audi & Naimi, "The AME 2020 atomic
//! mass evaluation (II). Tables, graphs and references", Chinese Phys. C 45, 030003 (2021),
//! Table I, the **atomic-mass column** (μu), transcribed from the published machine-readable
//! table `mass_1.mas20` (AMDC, dated 3 Mar 2021) — not the mass excess.
//!
//! **Atomic, not nuclear, masses.** The tabulated atomic mass is the mass of the neutral atom,
//! electrons and their binding included: ¹H is 1.007825 u = 1837.15 mₑ, the bare proton
//! 1836.15 mₑ. Born-Oppenheimer MD moves each nucleus together with its electrons, so the atomic
//! mass is the usual choice and the one `sim::Nucleus::new` has always used (its doc says so);
//! the difference is ≤ Z mₑ, i.e. ≤ 5.4e-4 relative for H and smaller for heavier elements, and
//! in a cation one spare electron mass is carried (1.8e-4 relative for H₃⁺) — below every bound
//! in `validate` (`references.md`). Masses are the only per-element datum besides Z
//! (CLAUDE.md rule 1): a *measured* quantity, never a fitted one.
//!
//! The table lists, for Z = 1..=18, the stable isotopes plus ³H and ¹⁴C (the long-lived ones a
//! later nuclear-events milestone wants): ¹H ²H ³H; ³He ⁴He; ⁶Li ⁷Li; ⁹Be; ¹⁰B ¹¹B; ¹²C ¹³C ¹⁴C;
//! ¹⁴N ¹⁵N; ¹⁶O ¹⁷O ¹⁸O; ¹⁹F; ²⁰Ne ²¹Ne ²²Ne; and, added by M7 Task 12 (the elements the
//! self-generated pseudopotentials enable), ²³Na; ²⁴Mg ²⁵Mg ²⁶Mg; ²⁷Al; ²⁸Si ²⁹Si ³⁰Si; ³¹P;
//! ³²S ³³S ³⁴S ³⁶S; ³⁵Cl ³⁷Cl; ³⁶Ar ³⁸Ar ⁴⁰Ar. "Most abundant" is the terrestrial isotopic
//! composition (¹H, ⁴He, ⁷Li, ⁹Be, ¹¹B, ¹²C, ¹⁴N, ¹⁶O, ¹⁹F, ²⁰Ne, ²³Na, ²⁴Mg, ²⁷Al, ²⁸Si, ³¹P,
//! ³²S, ³⁵Cl, ⁴⁰Ar) — a label for the default isotope, not a physical input. Every entry carries
//! all the digits the table prints (uncertainties range from 1.4e-11 u for ¹H to 2.0e-7 u for
//! ³⁶S); the `units` tests pin each one at 1e-9 u against a second transcription.
//!
//! Provenance of the Z = 11..=18 digits (`mass_1.mas20`, the atomic-mass column, μu → u): read
//! from two independent copies of the AME2020 file (the PtolemyGUI repository's `mass20.txt`, FSU,
//! and the `bhamnuclear/ckin` repository's `mass_table_2020.txt`, whose header is the AMDC's "mass.mas20,
//! 3 Mar 2021") — the two agreed digit for digit — and cross-checked against each row's printed
//! mass excess (m = A + ME/931494.10242 keV·u⁻¹ reproduces every atomic mass to within the
//! mass-excess rounding, ≤ 1e-9 u). The AMDC's own host refused automated fetches at the time
//! (HTTP 402), so the values were not read from it directly.
//!
//! Earlier builds carried AME2016 digits under an AME2020 label (¹H 1.00782503223,
//! ⁷Li 7.0160034366, ⁹Be 9.012183065, ¹¹B 11.00930536, …). The retabulation moves masses by at
//! most 1.9e-7 u (¹¹B: 11.00930536 was AME2016's 11.009305355 rounded; 1.8e-8 relative) and the
//! hydrogen mass by 3.3e-10 u (3.3e-10 relative) — no pinned number moves.

/// One row of the table: (A, atomic mass in u).
type Row = (u16, f64);

/// Table I's atomic-mass column, u, by Z (index 0 is the empty row for Z = 0). Ascending A within
/// each element.
const TABLE: [&[Row]; 19] = [
    &[],
    // H
    &[(1, 1.007825031898), (2, 2.014101777844), (3, 3.01604928132)],
    // He
    &[(3, 3.01602932197), (4, 4.00260325413)],
    // Li
    &[(6, 6.01512288742), (7, 7.01600343426)],
    // Be
    &[(9, 9.012183062)],
    // B
    &[(10, 10.012936862), (11, 11.009305166)],
    // C
    &[(12, 12.0), (13, 13.00335483534), (14, 14.00324198862)],
    // N
    &[(14, 14.00307400425), (15, 15.00010889827)],
    // O
    &[(16, 15.99491461926), (17, 16.99913175595), (18, 17.99915961214)],
    // F
    &[(19, 18.99840316207)],
    // Ne
    &[(20, 19.99244017525), (21, 20.993846685), (22, 21.991385113)],
    // Na
    &[(23, 22.98976928195)],
    // Mg
    &[(24, 23.985041689), (25, 24.985836966), (26, 25.982592972)],
    // Al
    &[(27, 26.981538408)],
    // Si
    &[(28, 27.97692653442), (29, 28.97649466434), (30, 29.973770137)],
    // P
    &[(31, 30.97376199768)],
    // S
    &[(32, 31.97207117354), (33, 32.97145890862), (34, 33.967867011), (36, 35.967080692)],
    // Cl
    &[(35, 34.968852694), (37, 36.965902573)],
    // Ar
    &[(36, 35.967545106), (38, 37.962732102), (40, 39.96238312204)],
];

/// The mass numbers of `TABLE`, by Z, for the `&'static [u16]` lookup (the tests check the two
/// agree row for row).
const MASS_NUMBERS: [&[u16]; 19] = [
    &[],
    &[1, 2, 3],
    &[3, 4],
    &[6, 7],
    &[9],
    &[10, 11],
    &[12, 13, 14],
    &[14, 15],
    &[16, 17, 18],
    &[19],
    &[20, 21, 22],
    &[23],
    &[24, 25, 26],
    &[27],
    &[28, 29, 30],
    &[31],
    &[32, 33, 34, 36],
    &[35, 37],
    &[36, 38, 40],
];

/// The most abundant isotope's A per Z (terrestrial composition; index 0 unused).
const MOST_ABUNDANT: [u16; 19] = [0, 1, 4, 7, 9, 11, 12, 14, 16, 19, 20, 23, 24, 27, 28, 31, 32, 35, 40];

/// The listed mass numbers of element `z`, ascending; empty beyond the table (Z > 18).
pub fn isotopes_of(z: u8) -> &'static [u16] {
    MASS_NUMBERS.get(z as usize).copied().unwrap_or(&[])
}

/// The most abundant isotope's mass number: 1, 4, 7, 9, 11, 12, 14, 16, 19, 20, 23, 24, 27, 28, 31, 32, 35, 40 for Z = 1..=18;
/// `None` beyond the table.
pub fn most_abundant_a(z: u8) -> Option<u16> {
    match MOST_ABUNDANT.get(z as usize) {
        Some(&a) if a > 0 => Some(a),
        _ => None,
    }
}

/// The AME2020 atomic mass of isotope (Z, A) in u; `None` for an unlisted one.
pub fn isotope_mass_amu_a(z: u8, a: u16) -> Option<f64> {
    isotopes_rows(z).iter().find(|r| r.0 == a).map(|r| r.1)
}

/// The most abundant isotope's atomic mass in u — `isotope_mass_amu_a(z, most_abundant_a(z)?)`.
pub fn isotope_mass_amu(z: u8) -> Option<f64> {
    isotope_mass_amu_a(z, most_abundant_a(z)?)
}

/// The element symbol (`crate::elements`, the workspace's one symbol table; Z = 1..=118).
pub use crate::elements::symbol;

/// "²H", "¹⁶O": the isotope label with a superscript mass number.
pub fn isotope_label(z: u8, a: u16) -> String {
    format!("{}{}", superscript(a), symbol(z))
}

/// The isotopes listed for Z as labels, "¹H ²H ³H" (empty beyond the table).
pub fn listed_isotopes(z: u8) -> String {
    isotopes_of(z).iter().map(|&a| isotope_label(z, a)).collect::<Vec<_>>().join(" ")
}

/// The message a request for an unlisted isotope gets: "no mass for ⁵H; listed: ¹H ²H ³H"
/// (beyond the table, where there is no symbol: "no mass for Z = 19, A = 39; listed: none
/// (Z = 19 is outside the table)").
pub fn unlisted_isotope_message(z: u8, a: u16) -> String {
    if isotopes_of(z).is_empty() {
        format!("no mass for Z = {z}, A = {a}; listed: none (Z = {z} is outside the table)")
    } else {
        format!("no mass for {}; listed: {}", isotope_label(z, a), listed_isotopes(z))
    }
}

fn isotopes_rows(z: u8) -> &'static [Row] {
    TABLE.get(z as usize).copied().unwrap_or(&[])
}

/// Superscript digits for a mass number.
fn superscript(n: u16) -> String {
    const DIGITS: [char; 10] = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
    n.to_string().chars().map(|c| DIGITS[c.to_digit(10).expect("a digit") as usize]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Review S7: every entry pinned at 1e-9 u against a second, independent transcription of
    /// Table I's atomic-mass column (μu → u), written here by hand from `mass_1.mas20` and not
    /// copied from the table above. The planner's digits (task brief) differed from the paper
    /// for ¹¹B (…167 vs …166), ¹⁷O (…757 vs …75595) and ²²Ne (…114 vs …113) — those three were
    /// AME2016 roundings; the paper's digits are the ones pinned.
    #[test]
    fn the_table_is_ame2020_and_the_most_abundant_lookup_is_consistent() {
        const PINS: &[(u8, u16, f64)] = &[
            (1, 1, 1.007825031898),
            (1, 2, 2.014101777844),
            (1, 3, 3.016049281320),
            (2, 3, 3.016029321970),
            (2, 4, 4.002603254130),
            (3, 6, 6.015122887420),
            (3, 7, 7.016003434260),
            (4, 9, 9.012183062),
            (5, 10, 10.012936862),
            (5, 11, 11.009305166),
            (6, 12, 12.0),
            (6, 13, 13.003354835340),
            (6, 14, 14.003241988620),
            (7, 14, 14.003074004250),
            (7, 15, 15.000108898270),
            (8, 16, 15.994914619260),
            (8, 17, 16.999131755950),
            (8, 18, 17.999159612140),
            (9, 19, 18.998403162070),
            (10, 20, 19.992440175250),
            (10, 21, 20.993846685),
            (10, 22, 21.991385113),
            // M7 Task 12: Z = 11..=18, a second transcription of Table I's atomic-mass column
            // (μu → u) from the printed rows, digit for digit
            (11, 23, 22.98976928195),
            (12, 24, 23.985041689),
            (12, 25, 24.985836966),
            (12, 26, 25.982592972),
            (13, 27, 26.981538408),
            (14, 28, 27.97692653442),
            (14, 29, 28.97649466434),
            (14, 30, 29.973770137),
            (15, 31, 30.97376199768),
            (16, 32, 31.97207117354),
            (16, 33, 32.97145890862),
            (16, 34, 33.967867011),
            (16, 36, 35.967080692),
            (17, 35, 34.968852694),
            (17, 37, 36.965902573),
            (18, 36, 35.967545106),
            (18, 38, 37.962732102),
            (18, 40, 39.96238312204),
        ];
        for &(z, a, m) in PINS {
            let got = isotope_mass_amu_a(z, a).unwrap_or_else(|| panic!("({z}, {a}) unlisted"));
            assert_relative_eq!(got, m, epsilon = 1e-9);
            assert!(isotopes_of(z).contains(&a));
        }
        assert_eq!(PINS.len(), (1..=18).map(|z| isotopes_of(z).len()).sum::<usize>());
        for z in 1..=18u8 {
            assert_eq!(isotope_mass_amu(z), isotope_mass_amu_a(z, most_abundant_a(z).unwrap()));
            assert!(isotopes_of(z).windows(2).all(|w| w[0] < w[1]), "Z = {z} not ascending");
            assert!(isotopes_of(z).contains(&most_abundant_a(z).unwrap()));
        }
        assert_eq!((1..=18).map(|z| most_abundant_a(z).unwrap()).collect::<Vec<_>>(), [1, 4, 7, 9, 11, 12, 14, 16, 19, 20, 23, 24, 27, 28, 31, 32, 35, 40]);
        for z in 0..=18u8 {
            assert_eq!(isotopes_of(z), isotopes_rows(z).iter().map(|r| r.0).collect::<Vec<_>>(), "Z = {z}: the two tables agree");
        }
        assert!(isotope_mass_amu_a(1, 5).is_none());
        assert!(isotopes_of(19).is_empty());
        assert!(most_abundant_a(19).is_none() && isotope_mass_amu(19).is_none() && isotope_mass_amu(0).is_none());
    }

    /// The numbers the dynamics oracle uses (spec §5.5): m_D/m_H and its square root, plus a
    /// transcription sanity check on every row — |m − A| < 0.1 u and m > Z × m_p (1.00727 u).
    #[test]
    fn mass_ratios_are_what_the_oracle_uses() {
        let r = isotope_mass_amu_a(1, 2).unwrap() / isotope_mass_amu_a(1, 1).unwrap();
        println!("m_D/m_H = {r:.9}, sqrt = {:.9}", r.sqrt());
        assert_relative_eq!(r, 1.9984637, epsilon = 1e-6);
        assert_relative_eq!(r.sqrt(), 1.4136703, epsilon = 1e-6);
        for z in 1..=18u8 {
            for &a in isotopes_of(z) {
                let m = isotope_mass_amu_a(z, a).unwrap();
                assert!((m - a as f64).abs() < 0.1, "({z}, {a}): {m}");
                assert!(m > z as f64 * 1.00727, "({z}, {a}): {m} lighter than its protons");
            }
        }
    }

    /// Spec §8 row 12 first failing test: the mass table covers Z ≤ 18 — every Z has at least one
    /// isotope, a most-abundant default, and a mass within 0.5 u of A (atomic masses: binding < 0.1 u
    /// per nucleon-mass unit at these A) — the only datum added (rule 1).
    #[test]
    fn isotope_mass_table_covers_z_up_to_18() {
        for z in 1u8..=18 {
            let isos = isotopes_of(z);
            assert!(!isos.is_empty(), "Z = {z} has no isotopes");
            let a = most_abundant_a(z).unwrap_or_else(|| panic!("Z = {z} has no default isotope"));
            assert!(isos.contains(&a));
            for &a in isos {
                let m = isotope_mass_amu_a(z, a).unwrap();
                assert!((m - a as f64).abs() < 0.5, "Z = {z}, A = {a}: {m}");
            }
        }
        assert!(most_abundant_a(19).is_none(), "Z > 18 is not enabled");
    }

    #[test]
    fn labels_and_the_unlisted_message() {
        assert_eq!(isotope_label(1, 2), "²H");
        assert_eq!(isotope_label(8, 16), "¹⁶O");
        assert_eq!(listed_isotopes(1), "¹H ²H ³H");
        assert_eq!(unlisted_isotope_message(1, 5), "no mass for ⁵H; listed: ¹H ²H ³H");
        assert_eq!(unlisted_isotope_message(19, 39), "no mass for Z = 19, A = 39; listed: none (Z = 19 is outside the table)");
        assert_eq!(unlisted_isotope_message(14, 27), "no mass for ²⁷Si; listed: ²⁸Si ²⁹Si ³⁰Si");
        assert_eq!(symbol(10), "Ne");
    }
}
