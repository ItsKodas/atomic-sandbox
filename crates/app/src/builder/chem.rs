//! The builder's assist (spec §3, §4): the **only** home of its two per-element tables — the usual
//! valences and the covalent radii — and the arithmetic on them: allowed valences with a formal
//! charge, lone pairs, the warnings, Fill hydrogens and the group stamps.
//!
//! **Rule 1 (signed off by the user 2026-10-20, spec §3).** These tables are a drawing aid for the
//! starting guess: they decide where the builder first puts nuclei and which atoms it marks. They
//! never reach `sim`, `qm`, `qm-gpu`, `providers`, the forces, the energies, the bond detector or
//! the names' graph (the names shown afterwards are read off the density), and the solver corrects
//! the guess (the Relax after the hand-off).
//!
//! References: B. Cordero et al., "Covalent radii revisited", Dalton Trans. 2008, 2832 (single-bond
//! radii; C is the sp³ value); multiple bonds scaled from the single ones by the C–C / C=C / C≡C
//! ratios 1.54 / 1.34 / 1.20 Å (× 0.87 double, × 0.78 triple).
use super::{Atom, Drawing};

/// The usual neutral valence (spec §3's list).
pub fn usual_valence(z: u8) -> u8 {
    match z {
        1 => 1,
        2 => 0,
        3 => 1,
        4 => 2,
        5 => 3,
        6 => 4,
        7 => 3,
        8 => 2,
        9 => 1,
        10 => 0,
        11 => 1,
        12 => 2,
        13 => 3,
        14 => 4,
        15 => 3,
        16 => 2,
        17 => 1,
        _ => 0,
    }
}

/// Cordero et al. 2008 single-bond covalent radius, Å (H–Ar).
fn covalent_radius_angstrom(z: u8) -> f64 {
    match z {
        1 => 0.31,
        2 => 0.28,
        3 => 1.28,
        4 => 0.96,
        5 => 0.84,
        6 => 0.76,
        7 => 0.71,
        8 => 0.66,
        9 => 0.57,
        10 => 0.58,
        11 => 1.66,
        12 => 1.41,
        13 => 1.21,
        14 => 1.11,
        15 => 1.07,
        16 => 1.05,
        17 => 1.02,
        18 => 1.06,
        _ => 1.5,
    }
}

/// The covalent radius in Bohr (the embedding works in atomic units).
pub fn covalent_radius_bohr(z: u8) -> f64 {
    units::angstrom_to_bohr(covalent_radius_angstrom(z))
}

/// The bond length's factor for a bond order (spec §3).
pub fn order_factor(order: u8) -> f64 {
    match order {
        2 => 0.87,
        3 => 0.78,
        _ => 1.0,
    }
}

/// r₀ of a bond: the radius sum × the order factor (Bohr).
pub fn bond_length(za: u8, zb: u8, order: u8) -> f64 {
    (covalent_radius_bohr(za) + covalent_radius_bohr(zb)) * order_factor(order)
}

/// Valence electrons from Z's place in its period (Z ≤ 18; no table: 1–2, then Z − 2, then Z − 10).
pub fn valence_electrons(z: u8) -> u8 {
    match z {
        0..=2 => z,
        3..=10 => z - 2,
        11..=18 => z - 10,
        _ => 0,
    }
}

/// The bond-order sums an atom of `z` with formal charge `q` usually has, smallest first. Neutral:
/// the usual valence (P 3 or 5, S 2, 4 or 6). Charged: the octet (duet for H, He) count of its
/// v − q valence electrons — e electrons give e bonds up to half the shell, else shell − e — which
/// is spec §3's charge rule: N, O, S, P with +1 gain a bond (NH₄⁺, H₃O⁺), C/N/O with −1 lose one
/// (OH⁻), B with −1 gains one (BH₄⁻); at q = 0 it reproduces the neutral list (a test checks).
pub fn allowed(z: u8, q: i8) -> Vec<u8> {
    if q == 0 {
        return match z {
            15 => vec![3, 5],
            16 => vec![2, 4, 6],
            _ => vec![usual_valence(z)],
        };
    }
    vec![octet_bonds(z, q)]
}

fn octet_bonds(z: u8, q: i8) -> u8 {
    let shell: i32 = if z <= 2 { 2 } else { 8 };
    let e = valence_electrons(z) as i32 - q as i32;
    let v = if e <= shell / 2 { e } else { shell - e };
    v.max(0) as u8
}

/// Lone pairs left on an atom with this bond-order sum (main group): (v − q − Σ order) / 2, never
/// negative; an odd electron (a radical) is not a pair.
pub fn lone_pairs(z: u8, q: i8, bond_sum: u32) -> u32 {
    let free = valence_electrons(z) as i64 - q as i64 - bond_sum as i64;
    (free.max(0) / 2) as u32
}

/// An atom the window marks (spec §4): over its allowed valence (red), or under it (amber, shown
/// once Fill hydrogens was used on the drawing).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Warning {
    Over,
    Under,
}

/// Every marked atom, by index.
pub fn warnings(d: &Drawing) -> Vec<(usize, Warning)> {
    d.atoms
        .iter()
        .enumerate()
        .filter_map(|(i, a)| {
            let sum = d.bond_sum(i);
            let ok = allowed(a.z, a.charge);
            let max = *ok.last().unwrap_or(&0) as u32;
            if sum > max {
                Some((i, Warning::Over))
            } else if !ok.contains(&(sum as u8)) {
                Some((i, Warning::Under))
            } else {
                None
            }
        })
        .collect()
}

/// One line naming the warnings ("C 1 has 5 bonds (at most 4); O 3 has 1 (usually 2)"), atoms
/// numbered from 1 as drawn.
pub fn describe(d: &Drawing, w: &[(usize, Warning)]) -> String {
    w.iter()
        .map(|&(i, kind)| {
            let a = &d.atoms[i];
            let sum = d.bond_sum(i);
            let ok = allowed(a.z, a.charge);
            let s = render::elements::symbol(a.z);
            match kind {
                Warning::Over => format!("{s} {} has {sum} bonds (at most {})", i + 1, ok.last().unwrap_or(&0)),
                Warning::Under => format!("{s} {} has {sum} (usually {})", i + 1, ok.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" or ")),
            }
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// The drawn length of a bond to hydrogen (canvas units; any other bond is drawn 1 long).
pub const H_DRAWN: f32 = 0.8;

/// The canvas length a new bond between these elements is drawn at (the Atom tool's snap, Fill
/// hydrogens, the stamps): 1, or [`H_DRAWN`] to a hydrogen. Drawing only — the embedding takes
/// the real lengths from the radii.
pub fn drawn_length(za: u8, zb: u8) -> f32 {
    if za == 1 || zb == 1 {
        H_DRAWN
    } else {
        1.0
    }
}

/// Fill hydrogens (spec §4): every atom other than H gets single-bonded hydrogens up to the
/// smallest allowed valence at or above its bond-order sum, drawn in the free directions around
/// it. Returns how many were added.
pub fn fill_hydrogens(d: &mut Drawing) -> usize {
    let n = d.atoms.len();
    let mut added = 0;
    for i in 0..n {
        let a = d.atoms[i];
        if a.z == 1 {
            continue;
        }
        let sum = d.bond_sum(i);
        let Some(&target) = allowed(a.z, a.charge).iter().find(|&&v| v as u32 >= sum) else { continue };
        let missing = target as u32 - sum;
        if missing == 0 {
            continue;
        }
        let lp = lone_pairs(a.z, a.charge, target as u32);
        for dir in free_directions(d, i, missing as usize, lp as usize) {
            let h = d.add_atom(1, [a.pos[0] + dir[0] * H_DRAWN, a.pos[1] + dir[1] * H_DRAWN]);
            d.bond(i, h);
            added += 1;
        }
    }
    added
}

/// `m` unit directions (canvas) for new bonds from atom `i`: with no bonds yet, spread over the
/// atom's domains (bonds + lone pairs) from the lower left; with one, the next domains round from it
/// (a zigzag for OH, a cross for CH₃); with more, each in the widest gap left.
pub fn free_directions(d: &Drawing, i: usize, m: usize, lone_pairs: usize) -> Vec<[f32; 2]> {
    use std::f32::consts::TAU;
    let here = d.atoms[i].pos;
    let mut angles: Vec<f32> = d
        .neighbours(i)
        .into_iter()
        .map(|j| {
            let p = d.atoms[j].pos;
            (p[1] - here[1]).atan2(p[0] - here[0])
        })
        .collect();
    let domains = (angles.len() + m + lone_pairs).max(1) as f32;
    let step = TAU / domains;
    let out: Vec<f32> = match angles.len() {
        0 => (0..m).map(|k| 150f32.to_radians() + k as f32 * step).collect(),
        1 => (1..=m).map(|k| angles[0] + k as f32 * step).collect(),
        _ => {
            let mut v = Vec::new();
            for _ in 0..m {
                let a = widest_gap(&angles);
                angles.push(a);
                v.push(a);
            }
            v
        }
    };
    out.into_iter().map(|a| [a.cos(), a.sin()]).collect()
}

/// The middle of the widest angular gap between `angles` (radians; one angle: its opposite).
fn widest_gap(angles: &[f32]) -> f32 {
    use std::f32::consts::{PI, TAU};
    if angles.is_empty() {
        return 0.0;
    }
    let mut a: Vec<f32> = angles.iter().map(|x| x.rem_euclid(TAU)).collect();
    a.sort_by(f32::total_cmp);
    if a.len() == 1 {
        return a[0] + PI;
    }
    let mut best = (0.0, 0.0);
    for k in 0..a.len() {
        let (s, e) = (a[k], if k + 1 < a.len() { a[k + 1] } else { a[0] + TAU });
        if e - s > best.0 {
            best = (e - s, s + (e - s) / 2.0);
        }
    }
    best.1
}

/// A group stamp (spec §4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stamp {
    Methyl,
    Hydroxyl,
    Amino,
    Carboxyl,
    Benzene,
}

impl Stamp {
    pub const ALL: [Stamp; 5] = [Stamp::Methyl, Stamp::Hydroxyl, Stamp::Amino, Stamp::Carboxyl, Stamp::Benzene];

    pub fn label(self) -> &'static str {
        match self {
            Stamp::Methyl => "CH₃",
            Stamp::Hydroxyl => "OH",
            Stamp::Amino => "NH₂",
            Stamp::Carboxyl => "COOH",
            Stamp::Benzene => "benzene",
        }
    }

    pub fn tip(self) -> &'static str {
        match self {
            Stamp::Methyl => "a methyl group with its hydrogens, bonded to the selected atom (free if none)",
            Stamp::Hydroxyl => "a hydroxyl group, bonded to the selected atom (free if none)",
            Stamp::Amino => "an amino group with its hydrogens, bonded to the selected atom (free if none)",
            Stamp::Carboxyl => "a carboxyl group C(=O)OH, bonded to the selected atom (free if none)",
            Stamp::Benzene => "a benzene ring (alternating single and double bonds, no hydrogens: Fill hydrogens adds them), bonded to the selected atom (free if none)",
        }
    }
}

/// Places a stamp: bonded to `anchor` along its widest free direction, or free with its first atom
/// at `at`. Returns the new atoms (the first is the one bonded to the anchor).
pub fn stamp(d: &mut Drawing, s: Stamp, anchor: Option<usize>, at: [f32; 2]) -> Vec<usize> {
    let anchor = anchor.filter(|&i| i < d.atoms.len());
    let (origin, dir) = match anchor {
        Some(i) => {
            let dir = free_directions(d, i, 1, 0)[0];
            let p = d.atoms[i].pos;
            ([p[0] + dir[0], p[1] + dir[1]], dir)
        }
        None => (at, [1.0, 0.0]),
    };
    // local frame: x outward along `dir`, y to its left
    let place = |x: f32, y: f32| [origin[0] + dir[0] * x - dir[1] * y, origin[1] + dir[1] * x + dir[0] * y];
    let h = H_DRAWN;
    let (s60, c60) = (60f32.to_radians().sin(), 0.5);
    // (Z, local x, local y) and bonds (local indices, order); atom 0 bonds to the anchor
    let (atoms, bonds): (Vec<(u8, f32, f32)>, Vec<(usize, usize, u8)>) = match s {
        Stamp::Methyl => (vec![(6, 0.0, 0.0), (1, h, 0.0), (1, 0.0, h), (1, 0.0, -h)], vec![(0, 1, 1), (0, 2, 1), (0, 3, 1)]),
        Stamp::Hydroxyl => (vec![(8, 0.0, 0.0), (1, h * c60, -h * s60)], vec![(0, 1, 1)]),
        Stamp::Amino => (vec![(7, 0.0, 0.0), (1, h * c60, h * s60), (1, h * c60, -h * s60)], vec![(0, 1, 1), (0, 2, 1)]),
        Stamp::Carboxyl => (
            vec![(6, 0.0, 0.0), (8, c60, s60), (8, c60, -s60), (1, c60 + h, -s60)],
            vec![(0, 1, 2), (0, 2, 1), (2, 3, 1)],
        ),
        Stamp::Benzene => {
            // a regular hexagon of side 1 whose first vertex points back at the anchor
            let back = std::f32::consts::PI;
            let ring = (0..6).map(|k| {
                let a = back + k as f32 * std::f32::consts::FRAC_PI_3;
                (6u8, 1.0 + a.cos(), a.sin())
            });
            (ring.collect(), (0..6).map(|k| (k, (k + 1) % 6, if k % 2 == 0 { 2 } else { 1 })).collect())
        }
    };
    let first = d.atoms.len();
    for &(z, x, y) in &atoms {
        d.atoms.push(Atom { z, pos: place(x, y), charge: 0 });
    }
    for &(a, b, order) in &bonds {
        d.bonds.push(super::Bond { a: first + a, b: first + b, order });
    }
    if let Some(i) = anchor {
        d.bond(i, first);
    }
    (first..d.atoms.len()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lone(z: u8, q: i8) -> Drawing {
        let mut d = Drawing::default();
        d.add_atom(z, [0.0, 0.0]);
        d.atoms[0].charge = q;
        d
    }

    fn formula(d: &Drawing) -> String {
        d.label(crate::names::NameStyle::FormulaOnly)
    }

    #[test]
    fn neutral_valences_match_the_spec_table() {
        let spec = [(1, 1), (2, 0), (3, 1), (4, 2), (5, 3), (6, 4), (7, 3), (8, 2), (9, 1), (10, 0), (11, 1), (12, 2), (13, 3), (14, 4), (15, 3), (16, 2), (17, 1), (18, 0)];
        for (z, v) in spec {
            assert_eq!(usual_valence(z), v, "Z = {z}");
            // the octet count of the neutral atom agrees with the table
            assert_eq!(octet_bonds(z, 0), v, "Z = {z}");
            assert_eq!(allowed(z, 0)[0], v);
        }
        assert_eq!(allowed(15, 0), vec![3, 5]);
        assert_eq!(allowed(16, 0), vec![2, 4, 6]);
        // the charge rule: N, O, S, P +1 gain one; C, N, O −1 lose one; B −1 gains one
        for (z, q, v) in [(7, 1, 4), (8, 1, 3), (16, 1, 3), (15, 1, 4), (6, -1, 3), (7, -1, 2), (8, -1, 1), (5, -1, 4)] {
            assert_eq!(allowed(z, q), vec![v], "Z = {z}, q = {q}");
        }
        // C–C single, double, triple from Cordero's 0.76 Å
        assert!((units::bohr_to_angstrom(bond_length(6, 6, 1)) - 1.52).abs() < 1e-9);
        assert!((units::bohr_to_angstrom(bond_length(6, 6, 2)) - 1.52 * 0.87).abs() < 1e-9);
        assert!((units::bohr_to_angstrom(bond_length(6, 6, 3)) - 1.52 * 0.78).abs() < 1e-9);
    }

    #[test]
    fn fill_h_on_c_n_o() {
        for (z, want) in [(6, "CH₄"), (7, "NH₃"), (8, "H₂O"), (9, "HF"), (5, "BH₃"), (14, "SiH₄"), (15, "PH₃"), (16, "H₂S")] {
            let mut d = lone(z, 0);
            fill_hydrogens(&mut d);
            assert_eq!(formula(&d), want);
            assert!(warnings(&d).is_empty(), "{want}: {:?}", warnings(&d));
        }
        // noble gases and hydrogen itself get none
        for z in [1, 2, 10, 18] {
            let mut d = lone(z, 0);
            assert_eq!(fill_hydrogens(&mut d), 0);
        }
        // a chain: C–C–O → ethanol
        let mut d = Drawing::default();
        d.add_atom(6, [0.0, 0.0]);
        d.add_atom(6, [1.0, 0.0]);
        d.add_atom(8, [1.5, 0.87]);
        d.bond(0, 1);
        d.bond(1, 2);
        assert_eq!(fill_hydrogens(&mut d), 6);
        assert_eq!(d.label(crate::names::NameStyle::Common), "C₂H₆O — ethanol");
        // a double bond is counted by order: C=O gets formaldehyde's two H on C, none on O
        let mut d = Drawing::default();
        d.add_atom(6, [0.0, 0.0]);
        d.add_atom(8, [1.0, 0.0]);
        d.bond(0, 1);
        d.cycle_order(0);
        fill_hydrogens(&mut d);
        assert_eq!(d.label(crate::names::NameStyle::Common), "CH₂O — formaldehyde");
    }

    #[test]
    fn fill_h_on_charged_n_and_o() {
        for (z, q, want) in [(7, 1, "NH₄⁺"), (8, -1, "HO⁻"), (8, 1, "H₃O⁺"), (5, -1, "BH₄⁻"), (6, 1, "CH₃⁺"), (6, -1, "CH₃⁻")] {
            let mut d = lone(z, q);
            fill_hydrogens(&mut d);
            assert_eq!(formula(&d), want);
            assert!(warnings(&d).is_empty(), "{want}");
        }
    }

    #[test]
    fn warnings_flag_over_and_under() {
        // a carbon with five bonds, a carbon with three (a radical), a fine hydrogen
        let mut d = Drawing::default();
        let c = d.add_atom(6, [0.0, 0.0]);
        for k in 0..5 {
            let h = d.add_atom(1, [k as f32, 1.0]);
            d.bond(c, h);
        }
        let c2 = d.add_atom(6, [5.0, 0.0]);
        for k in 0..3 {
            let h = d.add_atom(1, [5.0 + k as f32, 1.0]);
            d.bond(c2, h);
        }
        let w = warnings(&d);
        assert_eq!(w, vec![(c, Warning::Over), (c2, Warning::Under)]);
        let line = describe(&d, &w);
        assert!(line.contains("C 1 has 5 bonds (at most 4)") && line.contains("C 7 has 3 (usually 4)"), "{line}");
        // S with 4 and 6 bonds is allowed, 3 is under, 7 over
        for (n, want) in [(2, None), (3, Some(Warning::Under)), (4, None), (6, None), (7, Some(Warning::Over))] {
            let mut d = lone(16, 0);
            for k in 0..n {
                let f = d.add_atom(9, [k as f32, 1.0]);
                d.bond(0, f);
            }
            assert_eq!(warnings(&d).first().filter(|w| w.0 == 0).map(|w| w.1), want, "S with {n}");
        }
        // a charge cures the over-bonded nitrogen
        let mut d = lone(7, 0);
        for k in 0..4 {
            let h = d.add_atom(1, [k as f32, 1.0]);
            d.bond(0, h);
        }
        assert_eq!(warnings(&d), vec![(0, Warning::Over)]);
        d.atoms[0].charge = 1;
        assert!(warnings(&d).is_empty());
    }

    #[test]
    fn lone_pairs_count_the_free_electrons() {
        assert_eq!(lone_pairs(8, 0, 2), 2);
        assert_eq!(lone_pairs(7, 0, 3), 1);
        assert_eq!(lone_pairs(6, 0, 4), 0);
        assert_eq!(lone_pairs(7, 1, 4), 0);
        assert_eq!(lone_pairs(8, -1, 1), 3);
        assert_eq!(lone_pairs(6, 0, 3), 0, "a radical's odd electron is no pair");
    }

    #[test]
    fn benzene_stamp_is_a_regular_alternating_ring() {
        let mut d = Drawing::default();
        let ring = stamp(&mut d, Stamp::Benzene, None, [0.0, 0.0]);
        assert_eq!(ring.len(), 6);
        assert_eq!(d.bonds.len(), 6);
        let orders: Vec<u8> = d.bonds.iter().map(|b| b.order).collect();
        assert_eq!(orders, vec![2, 1, 2, 1, 2, 1]);
        for b in &d.bonds {
            let (p, q) = (d.atoms[b.a].pos, d.atoms[b.b].pos);
            let len = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "side {len}");
        }
        fill_hydrogens(&mut d);
        assert_eq!(d.label(crate::names::NameStyle::Common), "C₆H₆ — benzene");
        // bonded to a selected atom: one more bond, from the anchor to the ring's first atom
        let mut d = Drawing::default();
        let c = d.add_atom(6, [0.0, 0.0]);
        let ring = stamp(&mut d, Stamp::Benzene, Some(c), [0.0, 0.0]);
        assert!(d.bond_between(c, ring[0]).is_some());
        fill_hydrogens(&mut d);
        assert_eq!(d.label(crate::names::NameStyle::FormulaOnly), "C₇H₈");
    }

    #[test]
    fn group_stamps_carry_their_hydrogens() {
        let mut d = Drawing::default();
        let c = d.add_atom(6, [0.0, 0.0]);
        stamp(&mut d, Stamp::Carboxyl, Some(c), [0.0, 0.0]);
        fill_hydrogens(&mut d);
        assert_eq!(d.label(crate::names::NameStyle::Common), "C₂H₄O₂ — acetic acid");
        assert!(warnings(&d).is_empty());
        for (s, want) in [(Stamp::Methyl, "C₂H₆ — ethane"), (Stamp::Hydroxyl, "CH₄O — methanol"), (Stamp::Amino, "CH₅N")] {
            let mut d = Drawing::default();
            let c = d.add_atom(6, [0.0, 0.0]);
            stamp(&mut d, s, Some(c), [0.0, 0.0]);
            fill_hydrogens(&mut d);
            assert!(d.label(crate::names::NameStyle::Common).starts_with(want), "{:?}: {}", s, d.label(crate::names::NameStyle::Common));
            assert!(warnings(&d).is_empty());
        }
    }
}
