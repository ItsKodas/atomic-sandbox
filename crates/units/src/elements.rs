//! Element symbols, Z = 1..=118 (IUPAC) — the **one** symbol table of the workspace (M6 Task 10:
//! `render::elements::symbol` and `units::isotopes::symbol` both read it). Labels, not physics
//! (CLAUDE.md rule 1 forbids per-element *behaviour*, not names): nothing in the solver or the
//! dynamics reads them. It lives here, in the lowest crate, because the isotope labels (`isotopes`,
//! the surrogate's provenance line) need it and `units` depends on nothing; the element *names*
//! (display only) stay in `app::names`.

/// Symbols by Z (index 0 is "?").
#[rustfmt::skip]
pub const SYMBOLS: [&str; 119] = [
    "?",
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
];

/// English element names for Z = 1..=54 (index 0 is "?"), IUPAC spellings — display labels only.
/// Beyond xenon the UI shows the symbol and Z (those elements are greyed until the pseudopotential
/// milestone enables them; their names arrive with that milestone's data).
#[rustfmt::skip]
pub const NAMES: [&str; 55] = [
    "?",
    "hydrogen", "helium",
    "lithium", "beryllium", "boron", "carbon", "nitrogen", "oxygen", "fluorine", "neon",
    "sodium", "magnesium", "aluminium", "silicon", "phosphorus", "sulfur", "chlorine", "argon",
    "potassium", "calcium", "scandium", "titanium", "vanadium", "chromium", "manganese", "iron",
    "cobalt", "nickel", "copper", "zinc", "gallium", "germanium", "arsenic", "selenium", "bromine", "krypton",
    "rubidium", "strontium", "yttrium", "zirconium", "niobium", "molybdenum", "technetium", "ruthenium",
    "rhodium", "palladium", "silver", "cadmium", "indium", "tin", "antimony", "tellurium", "iodine", "xenon",
];

/// The element's English name, if listed (Z = 1..=54); `None` beyond — show the symbol instead.
pub fn name(z: u8) -> Option<&'static str> {
    (z >= 1).then(|| NAMES.get(z as usize).copied()).flatten()
}

/// The heaviest element with a symbol.
pub const Z_MAX: u8 = 118;

/// The element symbol for Z = 1..=118 ("?" for 0 and beyond).
pub fn symbol(z: u8) -> &'static str {
    SYMBOLS.get(z as usize).copied().unwrap_or("?")
}

/// Z for a symbol, case-insensitively ("o" → 8, "NE" → 10); `None` for an unknown one.
pub fn z_of_symbol(s: &str) -> Option<u8> {
    let s = s.trim();
    (1..=Z_MAX).find(|&z| SYMBOLS[z as usize].eq_ignore_ascii_case(s))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spot checks across the table (a shifted row would move every symbol after it).
    #[test]
    fn symbols_sit_at_their_atomic_numbers() {
        for (z, s) in [(1, "H"), (10, "Ne"), (11, "Na"), (18, "Ar"), (26, "Fe"), (36, "Kr"), (54, "Xe"), (57, "La"), (71, "Lu"), (79, "Au"), (86, "Rn"), (89, "Ac"), (92, "U"), (103, "Lr"), (118, "Og")] {
            assert_eq!(symbol(z), s, "Z = {z}");
        }
        assert_eq!(symbol(0), "?");
        assert_eq!(symbol(119), "?");
        assert_eq!(z_of_symbol("o"), Some(8));
        assert_eq!(z_of_symbol("NE"), Some(10));
        assert_eq!(z_of_symbol("Xx"), None);
        // every symbol is unique
        let mut v: Vec<&str> = SYMBOLS[1..].to_vec();
        v.sort();
        v.dedup();
        assert_eq!(v.len(), 118);
        assert_eq!(name(1), Some("hydrogen"));
        assert_eq!(name(8), Some("oxygen"));
        assert_eq!(name(54), Some("xenon"));
        assert_eq!(name(55), None);
        assert_eq!(name(0), None);
    }
}
