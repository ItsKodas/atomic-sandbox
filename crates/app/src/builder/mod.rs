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
pub mod chem;
pub mod embed;
pub mod ui;

use crate::scene::{Scene, SceneNucleus};
use crate::sim::Command;
use std::collections::BTreeSet;

/// How many drawings undo keeps (spec §9: at least 50).
pub const UNDO_DEPTH: usize = 100;

/// What the window hands to the scene (`ui::Action::Builder`, spec §6): the nuclei in Bohr (an
/// Insert's already placed beside the scene's atoms), the drawing's net formal charge, Relax after,
/// and the enlarged grid when the user accepted one.
#[derive(Clone, Debug, PartialEq)]
pub struct Handoff {
    pub replace: bool,
    pub nuclei: Vec<(u8, [f64; 3])>,
    pub charge: i32,
    pub relax: bool,
    pub grid: Option<(usize, f64)>,
    /// The scene chip's formula after a Replace ("C₂H₆O": "built C₂H₆O").
    pub label: String,
}

/// The margin "enlarge the box" leaves between the farthest nucleus and the cell's face (Bohr,
/// spec §6).
pub const ENLARGE_MARGIN: f64 = 4.0;

/// The largest |coordinate| a hand-off may put a nucleus at: the cell less the providers' margin
/// (`tools::clamp_into_cell`'s limit) and, with the walls on, inside their soft layer.
pub fn fit_limit(n: usize, h: f64, walls: Option<f64>) -> f64 {
    let cell = sim::Cell { n, h };
    let lim = (cell.max_coord() - 0.5 * cell.h).max(0.0);
    walls.map_or(lim, |w| lim.min(w - sim::dynamics::WALL_LAYER))
}

/// The largest |coordinate| of `x` (0 for none).
pub fn max_abs(x: &[[f64; 3]]) -> f64 {
    x.iter().flat_map(|p| p.iter().map(|c| c.abs())).fold(0.0, f64::max)
}

/// The smallest odd n ≥ `from` at spacing h whose half-box holds `max_abs` with
/// [`ENLARGE_MARGIN`] to spare (spec §6's "enlarge the box").
pub fn enlarge_n(max_abs: f64, h: f64, from: usize) -> usize {
    let mut n = from.max(3) | 1;
    while (n as f64 - 1.0) * h / 2.0 < max_abs + ENLARGE_MARGIN {
        n += 2;
    }
    n
}

/// Replace's scene (spec §6): the built nuclei at rest with their default isotopes, the drawing's
/// charge, multiplicity auto; σ and the ionic policy kept, the grid kept unless enlarged.
pub fn replace_scene(now: &Scene, h: &Handoff) -> Scene {
    let (grid_n, grid_h) = h.grid.unwrap_or((now.grid_n, now.grid_h));
    Scene {
        nuclei: h.nuclei.iter().map(|&(z, pos)| SceneNucleus { z, a: None, pos, vel: [0.0; 3] }).collect(),
        charge: h.charge,
        multiplicity: None,
        grid_n,
        grid_h,
        sigma: now.sigma,
        ionic: now.ionic,
    }
}

/// Insert's commands after any grid change (spec §6): one `AddNucleus` per atom (as Place does),
/// then one `SetCharge` with the new total when the drawing carries a net charge.
pub fn insert_commands(now: &Scene, h: &Handoff) -> Vec<Command> {
    let mut v: Vec<Command> = h.nuclei.iter().map(|&(z, pos)| Command::AddNucleus { z, a: None, pos }).collect();
    if h.charge != 0 {
        v.push(Command::SetCharge(now.charge + h.charge));
    }
    v
}

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
    /// Fill hydrogens was used on this drawing: under-bonded atoms are marked amber from then on
    /// (before it every bare carbon would be; spec §4). Clear resets it.
    pub filled: bool,
    /// The search field of the "more…" periodic table.
    pub ptable_query: String,
    /// The last Replace, for the scene chip's Reset.
    pub last_replace: Option<Handoff>,
    /// The enlarged grid the user accepted for the next hand-off (spec §6: offered when the
    /// molecule does not fit; used while it still holds it).
    pub enlarge: Option<(usize, f64)>,
    /// A hand-off pressed with warnings: (Replace?, the drawing then) — the same button again on the
    /// same drawing goes ahead (spec §4: radicals are legitimate).
    pub armed: Option<(bool, Drawing)>,
    /// The embedding of the drawing as it was (spec §5), recomputed only when the drawing changes.
    pub(crate) embedded: Option<(Drawing, Vec<[f64; 3]>)>,
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
            filled: false,
            ptable_query: String::new(),
            last_replace: None,
            enlarge: None,
            armed: None,
            embedded: None,
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
        self.filled = false;
        self.enlarge = None;
        self.armed = None;
    }

    /// Fill hydrogens (undoable): returns how many were added.
    pub fn fill_hydrogens(&mut self) -> usize {
        let before = self.drawing.clone();
        let n = chem::fill_hydrogens(&mut self.drawing);
        if n > 0 {
            let after = std::mem::replace(&mut self.drawing, before);
            self.checkpoint();
            self.drawing = after;
        }
        self.filled = true;
        n
    }

    /// Places a group stamp bonded to the one selected atom, else free at `at` (undoable); the
    /// stamp's first atom becomes the selection, so stamps chain.
    pub fn stamp(&mut self, s: chem::Stamp, at: [f32; 2]) {
        let anchor = self.selected_one();
        self.checkpoint();
        let new = chem::stamp(&mut self.drawing, s, anchor, at);
        self.selection = new.first().copied().into_iter().collect();
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

    /// The drawing in 3D (Bohr, centred), cached until the drawing changes.
    pub fn embedding(&mut self) -> &[[f64; 3]] {
        if self.embedded.as_ref().is_none_or(|(d, _)| *d != self.drawing) {
            self.embedded = Some((self.drawing.clone(), embed::embed(&self.drawing)));
        }
        &self.embedded.as_ref().unwrap().1
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

    fn scene(n: usize, h: f64, charge: i32) -> Scene {
        Scene { nuclei: vec![SceneNucleus { z: 1, a: None, pos: [0.0; 3], vel: [0.0; 3] }], charge, multiplicity: Some(2), grid_n: n, grid_h: h, sigma: Some(0.2), ionic: sim::IonicPolicy::Accepted }
    }

    fn handoff(replace: bool, charge: i32) -> Handoff {
        Handoff { replace, nuclei: vec![(7, [0.0, 0.0, 0.0]), (1, [1.9, 0.0, 0.0])], charge, relax: true, grid: None, label: "HN".into() }
    }

    #[test]
    fn replace_scene_charge_is_the_formal_sum() {
        // NH₄⁺ drawn: the scene's charge is the drawing's +1, whatever the old scene's was
        let mut d = Drawing::default();
        d.add_atom(7, [0.0, 0.0]);
        d.atoms[0].charge = 1;
        chem::fill_hydrogens(&mut d);
        let now = scene(81, 0.15, -2);
        let s = replace_scene(&now, &handoff(true, d.charge()));
        assert_eq!(s.charge, 1);
        assert_eq!(s.multiplicity, None, "multiplicity auto");
        assert_eq!((s.grid_n, s.grid_h, s.sigma, s.ionic), (81, 0.15, Some(0.2), sim::IonicPolicy::Accepted), "grid, σ and policy kept");
        assert_eq!(s.nuclei.iter().map(|n| (n.z, n.a)).collect::<Vec<_>>(), vec![(7, None), (1, None)], "default isotopes");
        assert!(s.nuclei.iter().all(|n| n.vel == [0.0; 3]));
        // an accepted enlargement is the new scene's grid
        let s = replace_scene(&now, &Handoff { grid: Some((101, 0.15)), ..handoff(true, 0) });
        assert_eq!((s.grid_n, s.grid_h), (101, 0.15));
    }

    #[test]
    fn insert_sends_one_set_charge_with_the_new_total() {
        let now = scene(81, 0.15, 1);
        let v = insert_commands(&now, &handoff(false, -1));
        assert_eq!(v.len(), 3);
        assert!(matches!(v[0], Command::AddNucleus { z: 7, a: None, .. }) && matches!(v[1], Command::AddNucleus { z: 1, .. }));
        assert!(matches!(v[2], Command::SetCharge(0)), "1 + (−1)");
        // a neutral drawing changes no charge
        let v = insert_commands(&now, &handoff(false, 0));
        assert_eq!(v.iter().filter(|c| matches!(c, Command::SetCharge(_))).count(), 0);
    }

    #[test]
    fn fit_check_and_enlarge_n() {
        // 81³ × 0.15: half-box 6 Bohr, the providers' 3h margin and half a step: 5.475
        let lim = fit_limit(81, 0.15, None);
        assert!((lim - (6.0 - 0.45 - 0.075)).abs() < 1e-9, "{lim}");
        // walls at 4.5: their soft layer starts at 3.5
        assert!((fit_limit(81, 0.15, Some(4.5)) - 3.5).abs() < 1e-9);
        assert_eq!(max_abs(&[[1.0, -7.25, 2.0], [0.5, 0.0, 3.0]]), 7.25);
        // a molecule reaching 7.25 Bohr: the half-box must be ≥ 11.25 → n ≥ 151 at h = 0.15
        let n = enlarge_n(7.25, 0.15, 81);
        assert_eq!(n % 2, 1);
        assert!((n as f64 - 1.0) * 0.15 / 2.0 >= 7.25 + ENLARGE_MARGIN);
        assert!((n as f64 - 3.0) * 0.15 / 2.0 < 7.25 + ENLARGE_MARGIN, "the smallest such n: {n}");
        // and the enlarged cell fits it, walls at their default included
        let walls = crate::ui::default_wall_half_width(&scene(n, 0.15, 0));
        assert!(fit_limit(n, 0.15, Some(walls)) >= 7.25);
        // never smaller than the box now
        assert_eq!(enlarge_n(0.1, 0.15, 121), 121);
    }

    #[test]
    fn handoff_is_not_replayable() {
        let a = crate::ui::Action::Builder(handoff(true, 0));
        assert_eq!(a.describe(), None);
        assert_eq!(crate::replay::unreplayable_kind(&a), Some("builder"));
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
