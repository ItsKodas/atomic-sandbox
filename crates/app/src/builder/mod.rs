//! The 2D molecule builder (spec `docs/superpowers/specs/2026-10-20-molecule-builder-design.md`):
//! File ▸ Builder… opens a window where a molecule is sketched as a structural formula — atoms,
//! bonds of order 1–3, formal charges — and handed to the scene as a 3D starting geometry.
//!
//! **Rule 1 (spec §3, signed off by the user 2026-10-20).** The builder's two per-element tables
//! (usual valences and covalent radii, [`chem`]) are a drawing aid: they decide where nuclei are
//! first put and which atoms the window marks, nothing else. The scene receives nuclei only (Z,
//! position, the default isotope) and the total charge; the forces, energies, the bond detector and
//! the names shown afterwards are the solver's, and the Relax after the hand-off corrects the guess.
//!
//! This file is the drawing itself ([`Drawing`]: atoms, bonds, charges and its operations) and the
//! window's state ([`Builder`], kept in `UiState` while the window is closed: reopening shows the
//! last drawing). The window is [`ui`].
pub mod ui;

use std::collections::BTreeSet;

/// How many drawings undo keeps (spec §9: at least 50).
pub const UNDO_DEPTH: usize = 100;

/// One drawn atom: its element, its place on the canvas (canvas units: a drawn bond is about 1)
/// and its formal charge (−1, 0 or +1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Atom {
    pub z: u8,
    pub pos: [f32; 2],
    pub charge: i8,
}

/// A drawn bond between atoms `a` and `b` (indices into [`Drawing::atoms`]) of order 1, 2 or 3.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bond {
    pub a: usize,
    pub b: usize,
    pub order: u8,
}

impl Bond {
    /// The other end of the bond, if `i` is one of its ends.
    pub fn other(&self, i: usize) -> Option<usize> {
        if self.a == i {
            Some(self.b)
        } else if self.b == i {
            Some(self.a)
        } else {
            None
        }
    }
}

/// The structural formula as drawn. What is drawn is what goes into the scene: hydrogens are
/// atoms like any other (Fill hydrogens adds them), there is no implicit hydrogen.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Drawing {
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
}

impl Drawing {
    pub fn add_atom(&mut self, z: u8, pos: [f32; 2]) -> usize {
        self.atoms.push(Atom { z, pos, charge: 0 });
        self.atoms.len() - 1
    }

    pub fn set_z(&mut self, i: usize, z: u8) {
        if let Some(a) = self.atoms.get_mut(i) {
            a.z = z;
        }
    }

    /// The bond between `a` and `b`, either way round.
    pub fn bond_between(&self, a: usize, b: usize) -> Option<usize> {
        self.bonds.iter().position(|k| (k.a == a && k.b == b) || (k.a == b && k.b == a))
    }

    /// Bonds `a` and `b` (a new single bond); an existing bond between them cycles its order
    /// instead. No bond to itself, nor to an atom that does not exist. Returns the bond.
    pub fn bond(&mut self, a: usize, b: usize) -> Option<usize> {
        if a == b || a >= self.atoms.len() || b >= self.atoms.len() {
            return None;
        }
        if let Some(k) = self.bond_between(a, b) {
            self.cycle_order(k);
            return Some(k);
        }
        self.bonds.push(Bond { a, b, order: 1 });
        Some(self.bonds.len() - 1)
    }

    /// 1 → 2 → 3 → 1.
    pub fn cycle_order(&mut self, k: usize) {
        if let Some(b) = self.bonds.get_mut(k) {
            b.order = b.order % 3 + 1;
        }
    }

    /// 0 → +1 → −1 → 0.
    pub fn cycle_charge(&mut self, i: usize) {
        if let Some(a) = self.atoms.get_mut(i) {
            a.charge = match a.charge {
                0 => 1,
                1 => -1,
                _ => 0,
            };
        }
    }

    /// Removes atom `i` and its bonds; the later atoms move down one index (the bonds follow).
    pub fn remove_atom(&mut self, i: usize) {
        if i >= self.atoms.len() {
            return;
        }
        self.atoms.remove(i);
        self.bonds.retain(|b| b.a != i && b.b != i);
        for b in &mut self.bonds {
            if b.a > i {
                b.a -= 1;
            }
            if b.b > i {
                b.b -= 1;
            }
        }
    }

    pub fn remove_bond(&mut self, k: usize) {
        if k < self.bonds.len() {
            self.bonds.remove(k);
        }
    }

    /// The atoms bonded to `i`.
    pub fn neighbours(&self, i: usize) -> Vec<usize> {
        self.bonds.iter().filter_map(|b| b.other(i)).collect()
    }

    /// The sum of the orders of `i`'s bonds.
    pub fn bond_sum(&self, i: usize) -> u32 {
        self.bonds.iter().filter(|b| b.other(i).is_some()).map(|b| b.order as u32).sum()
    }

    /// The net formal charge.
    pub fn charge(&self) -> i32 {
        self.atoms.iter().map(|a| a.charge as i32).sum()
    }

    /// The connected components (sorted index lists, lone atoms included), in order of their
    /// lowest atom.
    pub fn components(&self) -> Vec<Vec<usize>> {
        let n = self.atoms.len();
        let mut seen = vec![false; n];
        let mut out = Vec::new();
        for s in 0..n {
            if seen[s] {
                continue;
            }
            let mut comp = vec![s];
            seen[s] = true;
            let mut k = 0;
            while k < comp.len() {
                for j in self.neighbours(comp[k]) {
                    if !seen[j] {
                        seen[j] = true;
                        comp.push(j);
                    }
                }
                k += 1;
            }
            comp.sort_unstable();
            out.push(comp);
        }
        out
    }

    /// The live label of what is drawn (spec §2: the `names` module, as the 3D view names the
    /// density's molecules): one label per connected piece, joined by " + ", in `style`.
    pub fn label(&self, style: crate::names::NameStyle) -> String {
        if self.atoms.is_empty() {
            return String::new();
        }
        let z: Vec<u8> = self.atoms.iter().map(|a| a.z).collect();
        let graph = crate::names::MoleculeGraph {
            bonds: self.bonds.iter().map(|b| crate::names::GraphBond { i: b.a, j: b.b, multi_centre: false }).collect(),
            components: self.components(),
        };
        let (names, note) = crate::names::label_components(&z, &graph, self.charge());
        let mut s = names.iter().map(|n| n.label_in(style)).collect::<Vec<_>>().join(" + ");
        if let Some(note) = note {
            s += &format!("  ({note})");
        }
        s
    }
}

/// The canvas tool (spec §4; the default is Atom with carbon).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Tool {
    #[default]
    Atom,
    Bond,
    Charge,
    Erase,
    Select,
}

/// The window's state: the drawing, its undo history, the tool and element, the view. Display
/// state (not recorded): only the hand-off reaches the worker, through `ui::Action`.
pub struct Builder {
    pub open: bool,
    pub drawing: Drawing,
    undo: Vec<Drawing>,
    redo: Vec<Drawing>,
    pub tool: Tool,
    pub element: u8,
    pub selection: BTreeSet<usize>,
    /// The canvas point at the canvas area's centre, and points per canvas unit / [`ui::UNIT`].
    pub pan: [f32; 2],
    pub zoom: f32,
    /// What a press on the canvas started, while the button is held.
    pub(crate) gesture: ui::Gesture,
    /// "Relax after" (spec §6: on by default).
    pub relax_after: bool,
    /// The search field of the "more…" periodic table.
    pub ptable_query: String,
}

impl Default for Builder {
    fn default() -> Builder {
        Builder {
            open: false,
            drawing: Drawing::default(),
            undo: Vec::new(),
            redo: Vec::new(),
            tool: Tool::Atom,
            element: 6,
            selection: BTreeSet::new(),
            pan: [0.0, 0.0],
            zoom: 1.0,
            gesture: ui::Gesture::None,
            relax_after: true,
            ptable_query: String::new(),
        }
    }
}

impl Builder {
    /// Keeps the drawing as it is now for undo: call before every edit (once per drag). A
    /// drawing equal to the last one kept is not kept twice.
    pub fn checkpoint(&mut self) {
        if self.undo.last() != Some(&self.drawing) {
            self.undo.push(self.drawing.clone());
            if self.undo.len() > UNDO_DEPTH {
                self.undo.remove(0);
            }
        }
        self.redo.clear();
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn undo(&mut self) {
        if let Some(d) = self.undo.pop() {
            self.redo.push(std::mem::replace(&mut self.drawing, d));
            self.forget_indices();
        }
    }

    pub fn redo(&mut self) {
        if let Some(d) = self.redo.pop() {
            self.undo.push(std::mem::replace(&mut self.drawing, d));
            self.forget_indices();
        }
    }

    /// Clear: an empty drawing (undoable).
    pub fn clear(&mut self) {
        if !self.drawing.atoms.is_empty() {
            self.checkpoint();
            self.drawing = Drawing::default();
            self.forget_indices();
        }
    }

    /// Removes atom `i` (undoable); the selection and any gesture holding indices are dropped.
    pub fn erase_atom(&mut self, i: usize) {
        if i < self.drawing.atoms.len() {
            self.checkpoint();
            self.drawing.remove_atom(i);
            self.forget_indices();
        }
    }

    pub fn erase_bond(&mut self, k: usize) {
        if k < self.drawing.bonds.len() {
            self.checkpoint();
            self.drawing.remove_bond(k);
        }
    }

    /// Removes the selected atoms (the highest index first), undoable as one step.
    pub fn erase_selection(&mut self) {
        if self.selection.is_empty() {
            return;
        }
        self.checkpoint();
        for &i in self.selection.iter().rev() {
            self.drawing.remove_atom(i);
        }
        self.forget_indices();
    }

    /// Atom indices changed: nothing may hold one any more.
    fn forget_indices(&mut self) {
        self.selection.clear();
        self.gesture = ui::Gesture::None;
    }

    /// The one selected atom, when exactly one is.
    pub fn selected_one(&self) -> Option<usize> {
        (self.selection.len() == 1).then(|| *self.selection.iter().next().unwrap()).filter(|&i| i < self.drawing.atoms.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain(zs: &[u8]) -> Drawing {
        let mut d = Drawing::default();
        for (i, &z) in zs.iter().enumerate() {
            d.add_atom(z, [i as f32, 0.0]);
            if i > 0 {
                d.bond(i - 1, i);
            }
        }
        d
    }

    #[test]
    fn erase_reindexes_bonds_and_undo_restores() {
        let mut b = Builder { drawing: chain(&[6, 6, 8, 1]), ..Default::default() };
        let before = b.drawing.clone();
        b.selection.insert(3);
        b.erase_atom(1);
        assert_eq!(b.drawing.atoms.iter().map(|a| a.z).collect::<Vec<_>>(), vec![6, 8, 1]);
        // C0–C1 and C1–O2 went with atom 1; O–H was 2–3, now 1–2
        assert_eq!(b.drawing.bonds, vec![Bond { a: 1, b: 2, order: 1 }]);
        assert!(b.selection.is_empty(), "indices moved: the selection is dropped");
        b.undo();
        assert_eq!(b.drawing, before);
        b.redo();
        assert_eq!(b.drawing.atoms.len(), 3);
        b.undo();
        assert_eq!(b.drawing, before);
    }

    #[test]
    fn drag_onto_partner_cycles_not_duplicates() {
        let mut d = chain(&[6, 6]);
        assert_eq!(d.bond(1, 0), Some(0));
        assert_eq!(d.bonds.len(), 1);
        assert_eq!(d.bonds[0].order, 2);
        d.bond(0, 1);
        assert_eq!(d.bonds[0].order, 3);
        d.bond(0, 1);
        assert_eq!(d.bonds[0].order, 1);
    }

    #[test]
    fn no_self_bond() {
        let mut d = chain(&[6]);
        assert_eq!(d.bond(0, 0), None);
        assert_eq!(d.bond(0, 5), None);
        assert!(d.bonds.is_empty());
    }

    #[test]
    fn charges_cycle_and_sum() {
        let mut d = chain(&[7, 8]);
        d.cycle_charge(0);
        assert_eq!(d.atoms[0].charge, 1);
        d.cycle_charge(1);
        d.cycle_charge(1);
        assert_eq!(d.atoms[1].charge, -1);
        assert_eq!(d.charge(), 0);
        d.cycle_charge(1);
        assert_eq!(d.charge(), 1);
    }

    #[test]
    fn formula_and_name_of_a_drawn_ethanol() {
        // CH3–CH2–OH drawn with every hydrogen
        let mut d = chain(&[6, 6, 8]);
        for (heavy, n) in [(0, 3), (1, 2), (2, 1)] {
            for _ in 0..n {
                let h = d.add_atom(1, [0.0, 1.0]);
                d.bond(heavy, h);
            }
        }
        assert_eq!(d.label(crate::names::NameStyle::Common), "C₂H₆O — ethanol");
        // its isomer: dimethyl ether
        let mut e = chain(&[6, 8, 6]);
        for heavy in [0, 2] {
            for _ in 0..3 {
                let h = e.add_atom(1, [0.0, 1.0]);
                e.bond(heavy, h);
            }
        }
        assert_eq!(e.label(crate::names::NameStyle::Common), "C₂H₆O — dimethyl ether");
        // two pieces
        let two = chain(&[1, 1]);
        let mut two2 = two.clone();
        two2.add_atom(2, [5.0, 0.0]);
        assert_eq!(two2.components(), vec![vec![0, 1], vec![2]]);
        assert!(two2.label(crate::names::NameStyle::Common).contains(" + "));
        assert_eq!(Drawing::default().label(crate::names::NameStyle::Common), "");
    }

    #[test]
    fn undo_depth_is_capped() {
        let mut b = Builder::default();
        for i in 0..(UNDO_DEPTH + 30) {
            b.checkpoint();
            b.drawing.add_atom(6, [i as f32, 0.0]);
        }
        let mut n = 0;
        while b.can_undo() {
            b.undo();
            n += 1;
        }
        assert_eq!(n, UNDO_DEPTH);
        assert_eq!(b.drawing.atoms.len(), 30);
    }
}
