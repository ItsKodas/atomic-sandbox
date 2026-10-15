//! The composition badge shown at the cursor when hovering a nucleus (spec §7): isotope symbol, proton
//! and neutron counts (N = A − Z from the nucleus's own mass number) and the magnitude of the raw
//! provider force on it. Display only (rule 1); Ha/Bohr is the display unit.
use crate::names::superscript_digits;
use render::elements::symbol;

/// "¹⁶O · 8 p · 8 n · |F| 0.012 Ha/Bohr" (force omitted when `None`). |F| is printed with three decimals,
/// or as `4.2e-5` below 1e-3 so a nearly relaxed nucleus does not read as exactly zero.
pub fn badge_text(z: u8, a: u16, force: Option<[f64; 3]>) -> String {
    let n = (a as i32 - z as i32).max(0);
    let mut s = format!("{}{} · {z} p · {n} n", superscript_digits(a as u32), symbol(z));
    if let Some([fx, fy, fz]) = force {
        let f = (fx * fx + fy * fy + fz * fz).sqrt();
        let shown = if f == 0.0 || f >= 1e-3 { format!("{f:.3}") } else { format!("{f:.1e}") };
        s.push_str(&format!(" · |F| {shown} Ha/Bohr"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn badge_text_counts_neutrons_from_the_mass_number() {
        assert_eq!(badge_text(8, 16, None), "¹⁶O · 8 p · 8 n");
        assert_eq!(badge_text(1, 2, Some([0.0, 0.012, 0.0])), "²H · 1 p · 1 n · |F| 0.012 Ha/Bohr");
        assert_eq!(badge_text(1, 3, None), "³H · 1 p · 2 n");
    }

    #[test]
    fn the_force_is_the_euclidean_norm_and_small_forces_keep_their_digits() {
        assert_eq!(badge_text(1, 1, Some([0.003, 0.004, 0.0])), "¹H · 1 p · 0 n · |F| 0.005 Ha/Bohr");
        assert_eq!(badge_text(1, 1, Some([0.0, 0.0, 0.0])), "¹H · 1 p · 0 n · |F| 0.000 Ha/Bohr");
        assert_eq!(badge_text(1, 1, Some([4.2e-5, 0.0, 0.0])), "¹H · 1 p · 0 n · |F| 4.2e-5 Ha/Bohr");
    }
}
