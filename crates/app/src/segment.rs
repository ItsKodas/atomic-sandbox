//! "Train this segment" (M8a Task 10, spec §4.5): the selected atoms cut out of the scene with
//! their surroundings, the cut bonds capped with hydrogen, as a scene of its own whose samples
//! teach the shared model only the forces of its interior atoms (`sim::Scene::train_mask`; the
//! masked fit is `surrogate::fit`'s "Masked samples").
//!
//! # The cut (rulings P9, P15, P24, P26)
//!
//! - **The reach R = 2·r_cut + δ** (ruling P26, after the capping gate failed at 11.96× with
//!   r_cut): an atom's force F_i = −Σ_j ∂E_j/∂r_i runs over every centre j within r_cut of i, and
//!   each E_j sees everything within r_cut of j — so a force row is an exact equation of the source
//!   molecule only when everything within 2·r_cut of i is the source's. r_cut is the model's cutoff (the attached model's, else the default basis
//!   `surrogate::Basis::new(..).r_cut`; never written here), δ the sampling pass's largest planned
//!   move (`providers::Sampler::largest_planned_displacement` of the segment's own pass under the
//!   plan it opens with — [`cut_for_plan`] — so no planned perturbation or scan point brings a cap
//!   within r_cut of an interior atom; ruling P24).
//! - **Kept**: the selection and every atom within R of a selected atom.
//! - **Bonds** that can be cut are covalent ones: a pair at most [`COVALENT_FACTOR`] ×
//!   `builder::chem::bond_length(za, zb, 1)` apart — and, when the snapshot has a density bond graph
//!   for this scene, also bonded in it (the density decides; the length test keeps a hydrogen bond or
//!   another closed-shell contact from being "capped": a contact left behind needs no cap). A bond
//!   is cut when one end is kept and the other is not.
//! - **A hydrogen goes with its partner** (the approved mockup (d)): a cut bond with an H at either
//!   end keeps its other end instead (capping an H with an H would make H₂, and capping a lone H on
//!   a kept carbon would only move it).
//! - **Grown to a fixed point** (P15): an atom whose cap would land within R of a selected atom is
//!   kept instead, and the cut is made again, until nothing changes — so every selected atom is
//!   interior.
//! - **Caps**: each cut bond gets an H on the kept atom, along the old bond, at
//!   `bond_length(z_kept, 1, 1)` — the molecule builder's covalent tables (`builder::chem`, the one
//!   signed-off per-element table, rule 1): placement only, it never reaches `sim`/`qm`/`providers`.
//!   Caps are never trained.
//! - **Interior** (P15 with P26's reach): a kept atom whose every source atom within R is kept and which has no cap
//!   within R. The mask holds at the cut geometry and, by δ, at every planned geometry of the pass;
//!   the hot burst (a trajectory, not planned) is not covered — its few-tenths-of-a-Bohr excursions
//!   enter f_c's tail quadratically (physics review of Task 10a, Q4).
//!
//! The segment's box is its atoms' extent plus [`BOX_MARGIN`] on every side (mockup (d)), at the
//! source's spacing, σ, ionic policy and charge (so its samples land in the same collection); the
//! atoms are re-centred on the box. Refusals, each named, change nothing: nothing selected, a
//! selected atom with no neighbour within r_cut, a segment that is the whole molecule (train the scene itself), a box beyond the
//! largest grid, and one over the GPU memory budget ([`check_budget`]).
use crate::builder::chem;
use crate::quality::VramBudget;
use crate::scene::{Scene, SceneNucleus};

/// The box margin around the segment's atoms on every side (Bohr; mockup (d), approved: "the
/// segment plus 5 Bohr each side").
pub const BOX_MARGIN: f64 = 5.0;

/// A pair is a covalent bond that can be cut when at most this × the builder's single-bond length
/// apart (ruling P9; the builder's own geometric bond test).
pub const COVALENT_FACTOR: f64 = 1.25;

/// The refusal for an empty selection (mockup (d), refusals).
pub const NOTHING_SELECTED: &str = "Nothing to train: select atoms with the Edit tool (click, Shift-click or a box), then press Train this segment.";

/// A capped cut-out of a scene (module doc).
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    /// The segment as a scene: the kept atoms in source order, then the caps; re-centred in its own
    /// box; `train_mask = Some(interior)`.
    pub scene: Scene,
    /// Per segment atom: its whole R neighbourhood is the source's (P15) — the atoms it teaches.
    pub interior: Vec<bool>,
    /// The caps' indices in [`Segment::scene`].
    pub caps: Vec<usize>,
    /// Per segment atom: its index in the source scene (`None` for a cap).
    pub source_indices: Vec<Option<usize>>,
    /// Per cap (in [`Segment::caps`]' order): the cut bond, (kept, left behind) source indices.
    pub cut_bonds: Vec<(usize, usize)>,
    /// The selection (source indices, sorted, deduplicated).
    pub selected: Vec<usize>,
    /// segment position = source position + `shift` (the re-centring).
    pub shift: [f64; 3],
    /// The radii it was cut at: r_cut, δ and the reach R = 2·r_cut + δ ([`reach`]).
    pub r_cut: f64,
    pub delta: f64,
    pub reach: f64,
}

impl Segment {
    /// The atoms it teaches.
    pub fn n_interior(&self) -> usize {
        self.interior.iter().filter(|&&b| b).count()
    }

    /// A cap's position in the source scene's frame (the preview draws it there).
    pub fn cap_in_source(&self, cap: usize) -> [f64; 3] {
        let p = self.scene.nuclei[cap].pos;
        std::array::from_fn(|a| p[a] - self.shift[a])
    }
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// The pairs a cut may break (module doc, "Bonds"), i < j.
fn covalent_bonds(scene: &Scene, bonds: Option<&sim::BondGraph>) -> Vec<(usize, usize)> {
    let nuc = &scene.nuclei;
    let covalent = |i: usize, j: usize| dist(nuc[i].pos, nuc[j].pos) <= COVALENT_FACTOR * chem::bond_length(nuc[i].z, nuc[j].z, 1);
    match bonds.filter(|g| g.n_nuclei == nuc.len()) {
        Some(g) => {
            let mut v: Vec<(usize, usize)> = g.bonds.iter().filter(|b| b.i != b.j && b.i.max(b.j) < nuc.len()).map(|b| (b.i.min(b.j), b.i.max(b.j))).filter(|&(i, j)| covalent(i, j)).collect();
            v.sort_unstable();
            v.dedup();
            v
        }
        None => (0..nuc.len()).flat_map(|i| (i + 1..nuc.len()).map(move |j| (i, j))).filter(|&(i, j)| covalent(i, j)).collect(),
    }
}

/// The cap on the kept end `k` of a cut bond to `o` (module doc, "Caps").
fn cap_position(scene: &Scene, k: usize, o: usize) -> [f64; 3] {
    let (pk, po) = (scene.nuclei[k].pos, scene.nuclei[o].pos);
    let r = dist(pk, po);
    let l = chem::bond_length(scene.nuclei[k].z, 1, 1);
    std::array::from_fn(|a| pk[a] + l * (po[a] - pk[a]) / r)
}

fn who(scene: &Scene, i: usize) -> String {
    format!("{} #{i}", render::elements::symbol(scene.nuclei[i].z))
}

/// The reach of a cut (ruling P26): 2·r_cut + δ.
pub fn reach(r_cut: f64, delta: f64) -> f64 {
    2.0 * r_cut + delta
}

/// Cuts `atoms` (source indices) out of `scene` with r_cut and the margin δ (module doc), at the
/// reach [`reach`]`(r_cut, δ)`; `bonds` is the snapshot's density bond graph when it has one for
/// this scene.
pub fn cut_segment(scene: &Scene, atoms: &[usize], r_cut: f64, delta: f64, bonds: Option<&sim::BondGraph>) -> Result<Segment, String> {
    let seg = cut_with_reach(scene, atoms, r_cut, reach(r_cut, delta), bonds)?;
    Ok(Segment { delta, ..seg })
}

/// The cut at a given reach `big_r` (r_cut only names a lone atom): [`cut_segment`]'s body, and
/// the tests' way to cut a small molecule at a small reach.
fn cut_with_reach(scene: &Scene, atoms: &[usize], r_cut: f64, big_r: f64, bonds: Option<&sim::BondGraph>) -> Result<Segment, String> {
    if atoms.is_empty() {
        return Err(NOTHING_SELECTED.into());
    }
    let n = scene.nuclei.len();
    if let Some(&bad) = atoms.iter().find(|&&i| i >= n) {
        return Err(format!("atom #{bad} is not in the scene ({n} nuclei)"));
    }
    let mut selected = atoms.to_vec();
    selected.sort_unstable();
    selected.dedup();
    let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|m| m.pos).collect();
    if let Some(&lone) = selected.iter().find(|&&s| !(0..n).any(|j| j != s && dist(pos[s], pos[j]) <= r_cut)) {
        return Err(format!(
            "{} has no neighbour within {r_cut:.1} Bohr (the model's cutoff): there are no surroundings to learn. A lone atom is the isolated-atom reference, which every pass solves already.",
            who(scene, lone)
        ));
    }
    let near_selection = |p: [f64; 3]| selected.iter().any(|&s| dist(p, pos[s]) <= big_r);
    let mut kept: Vec<bool> = (0..n).map(|j| near_selection(pos[j])).collect();
    let bonds = covalent_bonds(scene, bonds);
    // the fixed point: an H keeps its partner, a cap that would land near the selection keeps its atom
    loop {
        let mut changed = false;
        for &(i, j) in &bonds {
            let (k, o) = match (kept[i], kept[j]) {
                (true, false) => (i, j),
                (false, true) => (j, i),
                _ => continue,
            };
            if scene.nuclei[k].z == 1 || scene.nuclei[o].z == 1 || near_selection(cap_position(scene, k, o)) {
                kept[o] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    if kept.iter().all(|&k| k) {
        return Err(format!(
            "the segment is the whole molecule — train the scene itself (Model ▸ Start sampling): every atom is within {big_r:.1} Bohr of the selection (twice the model's cutoff, plus the pass's largest planned move) or bonded to one that is."
        ));
    }
    let cut: Vec<(usize, usize)> = bonds
        .iter()
        .filter_map(|&(i, j)| match (kept[i], kept[j]) {
            (true, false) => Some((i, j)),
            (false, true) => Some((j, i)),
            _ => None,
        })
        .collect();
    let caps_src: Vec<[f64; 3]> = cut.iter().map(|&(k, o)| cap_position(scene, k, o)).collect();
    let kept_idx: Vec<usize> = (0..n).filter(|&j| kept[j]).collect();
    let mut interior: Vec<bool> = kept_idx
        .iter()
        .map(|&i| (0..n).all(|j| kept[j] || dist(pos[i], pos[j]) > big_r) && caps_src.iter().all(|&c| dist(pos[i], c) > big_r))
        .collect();
    interior.extend(caps_src.iter().map(|_| false));
    debug_assert!(selected.iter().all(|s| interior[kept_idx.iter().position(|k| k == s).unwrap()]), "P15: every selected atom is interior");
    // the box: the extent plus the margin, the atoms re-centred on it
    let all: Vec<[f64; 3]> = kept_idx.iter().map(|&i| pos[i]).chain(caps_src.iter().copied()).collect();
    let lo: [f64; 3] = std::array::from_fn(|a| all.iter().map(|p| p[a]).fold(f64::INFINITY, f64::min));
    let hi: [f64; 3] = std::array::from_fn(|a| all.iter().map(|p| p[a]).fold(f64::NEG_INFINITY, f64::max));
    let shift: [f64; 3] = std::array::from_fn(|a| -0.5 * (lo[a] + hi[a]));
    let extent = (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f64::max);
    let grid_n = crate::quality::grid_n(extent + 2.0 * BOX_MARGIN, scene.grid_h);
    let moved = |p: [f64; 3]| -> [f64; 3] { std::array::from_fn(|a| p[a] + shift[a]) };
    let mut nuclei: Vec<SceneNucleus> = kept_idx.iter().map(|&i| SceneNucleus { pos: moved(pos[i]), vel: [0.0; 3], ..scene.nuclei[i].clone() }).collect();
    nuclei.extend(caps_src.iter().map(|&c| SceneNucleus { z: 1, a: None, pos: moved(c), vel: [0.0; 3] }));
    let caps: Vec<usize> = (kept_idx.len()..nuclei.len()).collect();
    let source_indices: Vec<Option<usize>> = kept_idx.iter().map(|&i| Some(i)).chain(caps.iter().map(|_| None)).collect();
    let seg = Scene {
        nuclei,
        charge: scene.charge,
        multiplicity: None,
        grid_n,
        grid_h: scene.grid_h,
        sigma: scene.sigma,
        ionic: scene.ionic,
        train_mask: Some(interior.clone()),
    };
    Ok(Segment { scene: seg, interior, caps, source_indices, cut_bonds: cut, selected, shift, r_cut, delta: big_r - 2.0 * r_cut, reach: big_r })
}

/// δ for a segment: the largest planned move of the pass `plan` would make over it (ruling P24;
/// `providers::Sampler::largest_planned_displacement`). A plan the pass itself refuses is refused.
pub fn plan_margin(seg: &Scene, plan: &sim::SamplingPlan, r_cut: f64) -> Result<f64, String> {
    let s = seg.to_sim()?;
    let pass = providers::Sampler::with_cutoff(&s, plan, r_cut).map_err(|e| format!("the segment's sampling pass is refused: {e}"))?;
    Ok(pass.largest_planned_displacement())
}

/// [`cut_segment`] with δ the pass's own (ruling P24): cut at δ, measure the segment's planned
/// moves, and cut again at the larger δ until the segment's own δ is covered. δ only grows and the
/// segment with it, so this ends within the source's atom count.
pub fn cut_for_plan(scene: &Scene, atoms: &[usize], r_cut: f64, plan: &sim::SamplingPlan, bonds: Option<&sim::BondGraph>) -> Result<Segment, String> {
    let mut delta = 0.0f64;
    for _ in 0..=scene.nuclei.len() + 1 {
        let seg = cut_segment(scene, atoms, r_cut, delta, bonds)?;
        let need = plan_margin(&seg.scene, plan, r_cut)?;
        if need <= delta {
            return Ok(seg);
        }
        delta = need;
    }
    Err("the segment's margin did not settle (a bug): nothing changed".into())
}

/// The segment's own refusals before it opens: a box beyond the largest grid, and the GPU memory
/// budget for `solvers` GPU solvers (the estimated solver's teacher and auditor). A coarser grid is
/// not offered: h is part of the collection's key, so it would teach another collection.
pub fn check_budget(seg: &Segment, solvers: u64, budget: VramBudget) -> Result<(), String> {
    let s = &seg.scene;
    let box_len = crate::quality::box_len(s);
    if s.grid_n > crate::scene::MAX_GRID_N {
        return Err(format!(
            "the segment ({} atoms) needs a {box_len:.0} Bohr box, {}³ × {} Bohr — beyond the largest grid ({}³). Select fewer atoms.",
            s.nuclei.len(),
            s.grid_n,
            s.grid_h,
            crate::scene::MAX_GRID_N
        ));
    }
    let need = solvers * s.to_sim().map(|x| providers::gpu_footprint_bytes(&x)).unwrap_or(0);
    if solvers > 0 && need > budget.bytes {
        return Err(format!(
            "the segment ({} atoms, a {box_len:.0} Bohr box, {}³ × {} Bohr) needs ≈ {:.1} GB of GPU memory ({} solver{}), over the {} budget ({}). Select fewer atoms — a coarser grid would fit, but it would teach another collection (h is part of its key).",
            s.nuclei.len(),
            s.grid_n,
            s.grid_h,
            need as f64 / 1e9,
            solvers,
            if solvers == 1 { "" } else { "s: teacher + auditor" },
            budget.gb(),
            budget.why()
        ));
    }
    Ok(())
}

/// How the 3D view marks an atom of a segment (mockup (d)): taught (green), an edge atom (context,
/// not taught), a cap (teal; never taught).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mark {
    Taught,
    Edge,
    Cap,
}

/// The rings to draw over `scene` (the positions drawn now): a preview's kept atoms and its caps
/// (where they would go, in the source's frame), else an opened segment's own atoms — from its
/// training mask while it describes the scene (an added or removed nucleus drops it) and its caps.
pub fn marks(scene: &Scene, preview: Option<&Segment>, segment_caps: Option<&[usize]>) -> Vec<([f64; 3], Mark)> {
    if let Some(seg) = preview {
        let mut v: Vec<([f64; 3], Mark)> = seg
            .source_indices
            .iter()
            .zip(&seg.interior)
            .filter_map(|(i, &taught)| Some((scene.nuclei.get((*i)?)?.pos, if taught { Mark::Taught } else { Mark::Edge })))
            .collect();
        v.extend(seg.caps.iter().map(|&c| (seg.cap_in_source(c), Mark::Cap)));
        return v;
    }
    let (Some(mask), Some(caps)) = (scene.train_mask.as_ref().filter(|m| m.len() == scene.nuclei.len()), segment_caps) else { return Vec::new() };
    scene
        .nuclei
        .iter()
        .enumerate()
        .map(|(i, n)| (n.pos, if caps.contains(&i) { Mark::Cap } else if mask[i] { Mark::Taught } else { Mark::Edge }))
        .collect()
}

/// A straight-chain alkane CₙH₂ₙ₊₂ for the tests: a zig-zag carbon chain along x (C–C 2.88 Bohr,
/// CCC ≈ 112°), two H on each carbon and one more on each end at the builder's C–H length, at
/// h = 0.15 in a box that holds it.
#[cfg(test)]
pub(crate) fn alkane(n: usize) -> Scene {
    let ch = chem::bond_length(6, 1, 1);
    let x0 = -1.2 * (n as f64 - 1.0);
    let mut nuclei: Vec<SceneNucleus> = (0..n).map(|k| SceneNucleus { z: 6, a: None, pos: [x0 + 2.4 * k as f64, if k % 2 == 0 { 0.8 } else { -0.8 }, 0.0], vel: [0.0; 3] }).collect();
    for k in 0..n {
        let c = nuclei[k].pos;
        let up = c[1].signum();
        for s in [-1.0, 1.0] {
            nuclei.push(SceneNucleus { z: 1, a: None, pos: [c[0], c[1] + up * ch * 0.55, s * ch * 0.835], vel: [0.0; 3] });
        }
    }
    for (k, dx) in [(0usize, -1.0), (n - 1, 1.0)] {
        let c = nuclei[k].pos;
        nuclei.push(SceneNucleus { z: 1, a: None, pos: [c[0] + dx * ch * 0.9, c[1] - c[1].signum() * ch * 0.43, 0.0], vel: [0.0; 3] });
    }
    let half = nuclei.iter().flat_map(|m| m.pos.iter().map(|x| x.abs())).fold(0.0f64, f64::max) + 4.0;
    Scene { nuclei, charge: 0, multiplicity: None, grid_n: crate::quality::grid_n(2.0 * half, 0.15), grid_h: 0.15, sigma: None, ionic: sim::IonicPolicy::Accepted, train_mask: None }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// n-butane as `scenes/butane.scene` has it (this solver's relaxed anti geometry, Bohr): C1–C4 are
    /// #0–#3, then H5 on C1, H6/H7 on C1, H8/H9 on C2, H10/H11 on C3, H12 on C4, H13/H14 on C4.
    fn butane() -> Scene {
        let at: [(u8, [f64; 3]); 14] = [
            (6, [-3.5854, -0.7806, 0.0]),
            (6, [-1.1971, 0.7826, 0.0]),
            (6, [1.1971, -0.7826, 0.0]),
            (6, [3.5854, 0.7806, 0.0]),
            (1, [-5.2795, 0.4218, 0.0]),
            (1, [-3.6741, -2.0072, 1.6812]),
            (1, [-3.6741, -2.0072, -1.6812]),
            (1, [-1.1917, 2.0423, 1.6652]),
            (1, [-1.1917, 2.0423, -1.6652]),
            (1, [1.1917, -2.0423, -1.6652]),
            (1, [1.1917, -2.0423, 1.6652]),
            (1, [5.2795, -0.4218, 0.0]),
            (1, [3.6741, 2.0072, 1.6812]),
            (1, [3.6741, 2.0072, -1.6812]),
        ];
        Scene {
            nuclei: at.iter().map(|&(z, pos)| SceneNucleus { z, a: None, pos, vel: [0.0; 3] }).collect(),
            charge: 0,
            multiplicity: None,
            grid_n: 125,
            grid_h: 0.15,
            sigma: None,
            ionic: sim::IonicPolicy::Accepted,
            train_mask: None,
        }
    }

    /// The cut at reach `big_r`: butane is cut at small reaches here (at P26's 2·6 + δ Bohr every
    /// butane atom is within reach); a lone atom is one with no neighbour within 3 Bohr.
    fn cut(src: &Scene, atoms: &[usize], big_r: f64) -> Segment {
        cut_with_reach(src, atoms, 3.0, big_r, None).unwrap()
    }

    /// P15 by brute force from the source and the segment: every source atom within R of i is in
    /// the segment, and no cap is within R of i.
    fn p15(src: &Scene, seg: &Segment) -> Vec<bool> {
        let big_r = seg.reach;
        let kept: Vec<usize> = seg.source_indices.iter().flatten().copied().collect();
        seg.source_indices
            .iter()
            .map(|s| match s {
                None => false,
                Some(i) => {
                    let p = src.nuclei[*i].pos;
                    src.nuclei.iter().enumerate().all(|(j, m)| kept.contains(&j) || dist(p, m.pos) > big_r) && seg.caps.iter().all(|&c| dist(p, seg.cap_in_source(c)) > big_r)
                }
            })
            .collect()
    }

    /// Butane, C2 selected, cut at a reach of 3.5 Bohr: C1, C2, C3 and their hydrogens are kept (an H goes with
    /// its carbon), the C3–C4 bond is the one cut, and its cap sits on C3 along the old bond at the
    /// builder's C–H length; the mask is P15's, the selection interior, and the segment is re-centred
    /// in its own box at the source's spacing.
    #[test]
    fn cut_caps_every_broken_bond() {
        let src = butane();
        let seg = cut(&src, &[1], 3.5);
        let kept: Vec<usize> = seg.source_indices.iter().flatten().copied().collect();
        assert_eq!(kept, vec![0, 1, 2, 4, 5, 6, 7, 8, 9, 10], "C1–C3 and their seven H");
        assert_eq!(seg.cut_bonds, vec![(2, 3)], "the C3–C4 bond");
        assert_eq!(seg.caps, vec![10]);
        assert_eq!(seg.scene.nuclei.len(), 11);
        assert_eq!(seg.scene.nuclei[10].z, 1);
        let cap = seg.cap_in_source(10);
        let (c3, c4) = (src.nuclei[2].pos, src.nuclei[3].pos);
        assert!((dist(cap, c3) - chem::bond_length(6, 1, 1)).abs() < 1e-12, "at the builder's C–H length");
        let along: f64 = (0..3).map(|a| (cap[a] - c3[a]) * (c4[a] - c3[a])).sum::<f64>() / (dist(cap, c3) * dist(c4, c3));
        assert!((along - 1.0).abs() < 1e-12, "along the old bond");
        // the caps of every cut bond, the interior per P15
        assert_eq!(seg.cut_bonds.len(), seg.caps.len());
        assert_eq!(seg.interior, p15(&src, &seg));
        assert_eq!(seg.scene.train_mask.as_ref(), Some(&seg.interior));
        let taught: Vec<usize> = seg.interior.iter().enumerate().filter(|(_, &b)| b).map(|(k, _)| seg.source_indices[k].unwrap()).collect();
        assert_eq!(taught, vec![0, 1, 4, 5, 6, 7, 8], "C1, C2 and their H; C3 (C4 at 2.85 Bohr was cut) and its H (the cap at 3.35) are not");
        assert!(!seg.interior[2] && !seg.interior[10], "C3 sees the missing C4; a cap is never taught");
        // the box: the extent plus 5 Bohr a side, at h = 0.15, re-centred
        let s = &seg.scene;
        assert_eq!((s.grid_h, s.sigma, s.ionic, s.charge, s.multiplicity), (0.15, None, sim::IonicPolicy::Accepted, 0, None));
        let ext = (0..3).map(|a| s.nuclei.iter().map(|m| m.pos[a]).fold(f64::NEG_INFINITY, f64::max) - s.nuclei.iter().map(|m| m.pos[a]).fold(f64::INFINITY, f64::min)).fold(0.0, f64::max);
        assert_eq!(s.grid_n, crate::quality::grid_n(ext + 2.0 * BOX_MARGIN, 0.15));
        for a in 0..3 {
            let (lo, hi) = (s.nuclei.iter().map(|m| m.pos[a]).fold(f64::INFINITY, f64::min), s.nuclei.iter().map(|m| m.pos[a]).fold(f64::NEG_INFINITY, f64::max));
            assert!((lo + hi).abs() < 1e-12, "axis {a} centred");
        }
        assert!(s.validate().is_ok());
        // round trip through the scene file (the temp file "back to the scene" is another one)
        assert_eq!(Scene::parse(&s.to_text()).unwrap().train_mask, s.train_mask);
        // at the model's 2 × 6 Bohr every butane atom is within reach of C2: refused
        let whole = cut_segment(&src, &[1], 6.0, 0.0, None).unwrap_err();
        assert!(whole.starts_with("the segment is the whole molecule — train the scene itself"), "{whole}");
    }

    /// The reach is P26's 2·r_cut + δ: at r_cut 2.1 Bohr (reach 4.2) C2's reach takes C1, C3 and
    /// their H, and the cap on C3–C4 would sit 4.09 Bohr from C2 — so C4 is kept, and the segment is
    /// the whole molecule; the same reach as an explicit cut agrees.
    #[test]
    fn the_reach_is_twice_the_cutoff_plus_the_margin() {
        assert_eq!(reach(6.0, 0.0), 12.0);
        assert_eq!(reach(6.0, 2.25), 14.25);
        let src = butane();
        assert!(cut_segment(&src, &[1], 2.1, 0.0, None).unwrap_err().contains("whole molecule"));
        assert!(cut_with_reach(&src, &[1], 2.1, 4.2, None).unwrap_err().contains("whole molecule"));
        let seg = cut_segment(&src, &[1], 1.5, 0.5, None).map(|s| (s.reach, s.delta));
        assert!(seg.unwrap_err().contains("no neighbour within 1.5 Bohr"), "a lone atom is judged at r_cut, not the reach");
    }

    /// P15's fixed point. Butane, C2 selected, cut at a reach of 2.5 Bohr: only C2's own H are that close, so
    /// the plain cut would cap C2–C1 and C2–C3 with H 2.02 Bohr from C2 — inside its 2.5 Bohr, which
    /// would leave C2 untaught. The cut keeps C1 and C3 instead (and their H), caps C3–C4 where the
    /// cap is 4.1 Bohr from C2, and C2 is interior.
    #[test]
    fn the_cut_grows_until_the_selection_is_interior() {
        let src = butane();
        let seg = cut(&src, &[1], 2.5);
        let kept: Vec<usize> = seg.source_indices.iter().flatten().copied().collect();
        assert_eq!(kept, vec![0, 1, 2, 4, 5, 6, 7, 8, 9, 10]);
        assert_eq!(seg.cut_bonds, vec![(2, 3)]);
        assert!(seg.interior[1], "C2 is interior");
        assert_eq!(seg.interior, p15(&src, &seg));
        assert!(seg.caps.iter().all(|&c| dist(seg.cap_in_source(c), src.nuclei[1].pos) > 2.5));
        // a chain whose first cap lands by the selection, and so does the next: C0 selected, reach
        // 3.4 Bohr; C1 (2.9 Bohr) is kept by reach; C2 (3.65) is not, but C1's cap toward it would sit
        // 3.12 Bohr from C0 — kept; C3 (3.64) is not, but C2's cap toward it would sit 3.35 Bohr from
        // C0 — kept; C3's cap toward C4 sits 4.86 Bohr away: the cut
        let c = |x: f64, y: f64, z: f64| SceneNucleus { z: 6, a: None, pos: [x, y, z], vel: [0.0; 3] };
        let chain = Scene { nuclei: vec![c(0.0, 0.0, 0.0), c(2.9, 0.0, 0.0), c(2.2, 2.5, 1.5), c(-0.6, 3.5, 0.8), c(-1.2, 5.3, -1.6)], ..butane() };
        assert_eq!(covalent_bonds(&chain, None), vec![(0, 1), (1, 2), (2, 3), (3, 4)], "a bonded chain, nothing else within covalent reach");
        let seg = cut(&chain, &[0], 3.4);
        let kept: Vec<usize> = seg.source_indices.iter().flatten().copied().collect();
        assert_eq!(kept, vec![0, 1, 2, 3], "grown twice: C2, then C3");
        assert!(dist(seg.cap_in_source(seg.caps[0]), chain.nuclei[0].pos) > 3.4);
        assert_eq!(seg.cut_bonds, vec![(3, 4)]);
        assert!(seg.interior[0]);
        assert_eq!(seg.interior, p15(&chain, &seg));
    }

    /// The density bond graph decides what is a bond (P9), within covalent reach: a pair it does not
    /// bond is not capped, and a contact it does bond but at more than 1.25 × the covalent length is
    /// not capped either.
    #[test]
    fn the_density_bond_graph_decides_what_is_cut() {
        let src = butane();
        let bond = |i: usize, j: usize| sim::Bond {
            i,
            j,
            rho_bcp: 0.2,
            enhancement: 1.0,
            s: 0.5,
            route: sim::Route::Saddle,
            multi_centre: false,
            laplacian_bcp: -0.5,
            laplacian_noise: 0.01,
        };
        let geometric = covalent_bonds(&src, None);
        assert_eq!(geometric.len(), 13, "3 C–C + 10 C–H");
        let mut g = sim::BondGraph { bonds: geometric.iter().filter(|&&p| p != (2, 3)).map(|&(i, j)| bond(i, j)).collect(), components: Vec::new(), step: 0, sim_time: 0.0, n_nuclei: 14 };
        g.bonds.push(bond(0, 3)); // a "bond" 7.2 Bohr long: beyond covalent reach
        assert_eq!(covalent_bonds(&src, Some(&g)), geometric.iter().copied().filter(|&p| p != (2, 3)).collect::<Vec<_>>());
        let seg = cut_with_reach(&src, &[1], 3.0, 3.5, Some(&g)).unwrap();
        assert!(seg.caps.is_empty() && seg.cut_bonds.is_empty(), "C3–C4 is no bond in this graph: nothing to cap");
        assert!(!seg.interior[2], "C3 still sees the missing C4");
        // a graph of another scene (other nucleus count) is not used
        g.n_nuclei = 13;
        assert_eq!(cut_with_reach(&src, &[1], 3.0, 3.5, Some(&g)).unwrap().cut_bonds, vec![(2, 3)]);
    }

    /// Each refusal names its reason: nothing selected, a lone atom (named), and a segment over the
    /// GPU memory budget (its atoms, box, grid and the budget named); none returns a segment.
    #[test]
    fn segment_refusals_name_the_reason() {
        let src = butane();
        assert_eq!(cut_segment(&src, &[], 6.0, 0.0, None).unwrap_err(), NOTHING_SELECTED);
        let mut lone = src.clone();
        lone.nuclei.push(SceneNucleus { z: 2, a: None, pos: [0.0, 0.0, 8.0], vel: [0.0; 3] });
        let e = cut_segment(&lone, &[1, 14], 3.5, 0.0, None).unwrap_err();
        assert!(e.starts_with("He #14 has no neighbour within 3.5 Bohr"), "{e}");
        assert!(cut_segment(&src, &[99], 3.5, 0.0, None).unwrap_err().contains("#99 is not in the scene"));
        let seg = cut(&src, &[1], 3.5);
        let e = check_budget(&seg, 2, VramBudget::fixed(1_000)).unwrap_err();
        assert!(e.starts_with(&format!("the segment (11 atoms, a {:.0} Bohr box, {}³ × 0.15 Bohr) needs ≈", crate::quality::box_len(&seg.scene), seg.scene.grid_n)), "{e}");
        assert!(e.contains("2 solvers: teacher + auditor") && e.contains("over the 0.0 GB budget") && e.contains("Select fewer atoms"), "{e}");
        assert_eq!(check_budget(&seg, 2, VramBudget::fixed(u64::MAX)), Ok(()));
        assert_eq!(check_budget(&seg, 0, VramBudget::fixed(1_000)), Ok(()), "no GPU solver: no GPU budget");
        let mut huge = seg.clone();
        huge.scene.grid_n = 251;
        assert!(check_budget(&huge, 2, VramBudget::fixed(u64::MAX)).unwrap_err().contains("beyond the largest grid (201³)"));
    }

    /// P24: the cut for the Standard plan covers the pass's own planned moves — δ is the segment's
    /// `largest_planned_displacement`, every interior atom is more than 2·r_cut + δ from every cap and
    /// from every atom left behind. Decane (a longer chain, so 2·6 Bohr + δ leaves something behind).
    #[test]
    fn the_cut_for_a_plan_covers_its_planned_moves() {
        let decane = alkane(10);
        let plan = sim::SamplingPlan::default();
        let r_cut = surrogate::Basis::new(&[]).r_cut;
        let seg = cut_for_plan(&decane, &[0], r_cut, &plan, None).unwrap();
        let need = plan_margin(&seg.scene, &plan, r_cut).unwrap();
        println!("decane end carbon: {} atoms kept + {} caps, {} taught; δ = {:.3} Bohr (needed {need:.3})", seg.source_indices.iter().flatten().count(), seg.caps.len(), seg.n_interior(), seg.delta);
        assert!(seg.delta > 0.0 && need <= seg.delta, "δ {} covers the pass's {need}", seg.delta);
        assert_eq!(seg.reach, 2.0 * r_cut + seg.delta);
        assert!(!seg.caps.is_empty(), "something was left behind and capped");
        assert_eq!(seg.interior, p15(&decane, &seg));
        assert!(seg.interior[0], "the selection is interior");
    }

    /// The rings: a preview marks the source's kept atoms and the caps' places; an opened segment its
    /// own atoms, and nothing once its mask is gone.
    #[test]
    fn the_rings_follow_the_preview_then_the_mask() {
        let src = butane();
        let seg = cut(&src, &[1], 3.5);
        let m = marks(&src, Some(&seg), None);
        assert_eq!(m.len(), 11);
        assert_eq!(m.iter().filter(|x| x.1 == Mark::Taught).count(), 7);
        assert_eq!(m.iter().filter(|x| x.1 == Mark::Cap).map(|x| x.0).collect::<Vec<_>>(), vec![seg.cap_in_source(10)]);
        assert_eq!(m[1], (src.nuclei[1].pos, Mark::Taught));
        let open = marks(&seg.scene, None, Some(&seg.caps));
        assert_eq!(open.iter().map(|x| x.1).collect::<Vec<_>>(), seg.interior.iter().enumerate().map(|(i, &t)| if i == 10 { Mark::Cap } else if t { Mark::Taught } else { Mark::Edge }).collect::<Vec<_>>());
        assert!(marks(&src, None, Some(&seg.caps)).is_empty(), "no mask: no rings");
    }

    /// The mask describes its nuclei: an edit of their count drops it in the app's scene copy (the
    /// worker's mirror, `worker::commands`), a move keeps it.
    #[test]
    fn the_mask_is_dropped_when_a_nucleus_is_added_or_removed() {
        use crate::sim::Command;
        let seg = cut(&butane(), &[1], 3.5);
        let mut s = seg.scene.clone();
        Command::MoveNucleus { index: 0, pos: [0.1, 0.0, 0.0] }.apply(&mut s);
        assert_eq!(s.train_mask.as_ref(), Some(&seg.interior), "a move keeps it");
        Command::RemoveNucleus(10).apply(&mut s);
        assert_eq!(s.train_mask, None, "a removed cap ends it");
        let mut s = seg.scene.clone();
        Command::AddNucleus { z: 1, a: None, pos: [0.0, 0.0, 4.0] }.apply(&mut s);
        assert_eq!(s.train_mask, None, "an added atom ends it");
    }
}
