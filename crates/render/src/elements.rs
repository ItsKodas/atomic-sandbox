//! Display-only element table. Colours follow the common CPK convention; radii are legibility choices.
//! Nothing here is physics (CLAUDE.md rule 1): the simulation never reads these values.

/// Display colour (linear RGB) for atomic number `z`; pink for anything beyond Ne.
pub fn colour(z: u8) -> [f32; 3] {
    match z {
        1 => [0.95, 0.95, 0.95], 2 => [0.85, 1.0, 1.0], 3 => [0.80, 0.50, 1.0], 4 => [0.76, 1.0, 0.0],
        5 => [1.0, 0.71, 0.71], 6 => [0.56, 0.56, 0.56], 7 => [0.19, 0.31, 0.97], 8 => [1.0, 0.05, 0.05],
        9 => [0.56, 0.88, 0.31], 10 => [0.70, 0.89, 0.96], _ => [1.0, 0.08, 0.58],
    }
}

/// Display radius in Bohr: `0.35 · Z^(1/3)` (legibility, not physics).
pub fn radius(z: u8) -> f32 { 0.35 * (z.max(1) as f32).cbrt() }

/// Element symbol for `z` (Z = 1..=118; "?" for 0 and beyond): the workspace's one symbol table,
/// `units::elements` (M6 Task 10 — it was duplicated here and in `units::isotopes`).
pub use units::elements::symbol;
