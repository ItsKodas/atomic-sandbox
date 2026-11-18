//! Stray atoms (spec 2026-11-18, `docs/superpowers/specs/2026-11-18-stray-atom-composition-design.md`): in strict
//! mode a scene made of a **bonded part** plus one or more **stray atoms** (no neighbour within the model's r_cut) is
//! answered by the model once it has been *measured* that this combination's DFT energy splits as the bonded part's
//! plus the free atoms'. This module holds the pure logic — the split, the signature, the static floor, the composition
//! check's verdict and the cleared region with its book — and the provider's wiring of the checks and the sample
//! hold-out (plan Task 3; below, "The checks on the auditor"); answering inside a region is `evaluate`'s (plan Task 4).
//! Atomic units (Hartree, Bohr).
//!
//! # Why the model's answer is already "bonded part + free atoms" (spec §2)
//!
//! The model is E = Σ_i [E_ref(Z_i) + Σ_k c̃_{s_i,k} B_{i,k}] with mean 0 and no intercept. f_c and the radial basis
//! vanish at and beyond r_cut, pairs at r ≥ r_cut are excluded and the body-3/4 terms are built from neighbours inside
//! r_cut only, so a stray atom s has B_s = 0 and enters no other atom's descriptor; there is no whole-scene term:
//!
//! ```text
//!   E_model(scene) = E_model(bonded part alone) + Σ_s E_ref(Z_s)      (to a few ulp: the order of the Neumaier sum)
//!   F_s = 0 exactly;  F_i (bonded) = the bonded part's own forces, bit for bit
//! ```
//!
//! E is C¹ and F C⁰ at r_cut. The question is whether the **teacher** agrees:
//!
//! ```text
//!   E_DFT(scene) =? E_DFT(bonded part alone, same box) + Σ_s E_DFT(atom s alone at R_s, same box)          (C)
//! ```
//!
//! Each part is solved in the same box at its scene position, so the box confinement and the egg-box term cancel
//! exactly and (C) measures composition alone; teacher energies are compared with teacher energies, raw throughout.
//! (C) fails for two physical reasons:
//!
//! 1. **Spin.** The teacher solves the whole cell at one (N↑, N↓). (C) holds when every part's excess spin is aligned
//!    up: M_scene = M_frag + Σ_s (m_s − 1), M_frag the bonded part's automatic multiplicity 1 + (Σ Z_frag mod 2) and
//!    m_s the reference multiplicity of Z_s. Sufficient, not necessary: the antiparallel product composes too by
//!    spin-flip symmetry, but nothing seeds it, so refusing it is the conservative side.
//! 2. **Charge transfer at any separation.** One μ_σ per spin spans the cell (Fermi–Dirac, kT = `qm::ELECTRON_KT` =
//!    5 mHa), and LSDA has no derivative discontinuity: a stray HOMO near the fragment's LUMO in the same spin channel
//!    (or the reverse) moves charge across any gap. A fractional transfer pins both levels at μ and gains only
//!    ~Δε²/2η, often below 1 mHa — the energy test alone misses the onset, so the check measures a frontier margin.
//!    It depends on the fragment's geometry (a stretching bond lowers its LUMO), hence the local region below.
//!
//! # The static floor (spec §4.2; [`floor_refusal`])
//!
//! Refused before any solve, never cleared, never fitted, the reason shown; in this order:
//!
//! - a **charged** scene (the atoms' solves are neutral);
//! - a **forced multiplicity** (v1: a forced-M store's samples are whole-scene samples at that M, and collections
//!   refuse forced M);
//! - the **spin rule** at automatic: 1 + (Σ Z mod 2) ≠ M_frag + Σ_s (m_s − 1); an unknown m_s refuses (`lone`'s
//!   reference-multiplicity source: the scan's record, or the single allowed M);
//! - **two or more stray atoms** of mixed species, or of one species not known to end at a gap
//!   ([`lone::closed_spin_channels`]; `lone` module doc, "The multiplicity rule and the occupations": copies whose
//!   levels the box splits can fill one copy's open shell first);
//! - a **shell-check fence** on (a stray atom's species, any other species of the scene): the cutoff is measured not
//!   to hold for that pair;
//! - a stray atom within [`STRAY_FACE_MARGIN`] = 4 Bohr of a face (the box rule's edge: +1.8e-3 Ha of confinement
//!   for a lone H there, `lone` module doc), before a solve; after one, a box-affected face density voids the check.
//!
//! # The composition check (spec §4.3; [`judge`])
//!
//! From the scene's own fallback solve, the bonded part alone (at M_frag) and each stray atom alone (at m_s), all in the
//! same box at the scene's positions. **Void** (not counted; the next eligible step tries again) when any solve is not
//! clean (a warning, untrainable, a box-affected face density): nothing it reports is a measurement. Otherwise it
//! **fails** on the first measured test that fails, in the order below; when none fails but a level some test needs
//! was not computed, it is void (`Void("frontier level not computed")`, never a pass); else it **passes**:
//!
//! - **Energy:** |E_scene − E_frag − Σ_s E_atom(R_s)| ≤ `energy_std_per_atom` × n_stray — the error belongs to the
//!   stray atoms, not to the scene's size (compensated sum).
//! - **Forces:** every bonded atom's scene force against the fragment's, every stray atom's against its own solve's,
//!   each within [`FallbackPolicy::force_residual`]'s bound.
//! - **Frontier margin:** per spin and per stray atom, (fragment acceptor − stray donor) and (stray acceptor −
//!   fragment donor) ≥ [`frontier_margin_min`] = 27.6 kT = 138 mHa. That is `lone`'s gap criterion: a level with
//!   min(f, 1 − f) < `PARTIAL_OCCUPATION_MIN` = 1e-6 lies at least ln(1/1e-6) = 13.8 kT from μ, so a donor and an
//!   acceptor that are both clear of μ on their own sides are ≥ 2 × 13.8 = 27.6 kT apart. Every level with
//!   f > `PARTIAL_OCCUPATION_MIN` donates and every level with f < 1 − `PARTIAL_OCCUPATION_MIN` accepts, so a part's
//!   **donor** is the higher of its HOMO and its highest partial level, and its **acceptor** the lower of its LUMO and
//!   its lowest partial level. `Frontier` splits a channel at f = ½, so its HOMO alone would miss a partial level at
//!   f ≥ ½ as an acceptor (an open-shell stray O at its M = 1 partial ensemble: its 2p at f ≈ 2/3 is the HOMO in both
//!   spins and sits ~40 mHa above H₂'s σg, while its 3s LUMO is ~0.4 Ha away; review I1). A channel without an
//!   occupied or partial level (the stray H's ↓) has no donor: that direction needs no level. **Two parts both open in
//!   one spin channel always fail**: each one's partial level is at once its donor and its acceptor, so one of the two
//!   directions is ≤ 0. That is the physics, not a strictness: one shared μ_σ redistributes charge between open shells
//!   (`lone`'s open-shell copies, here across species).
//! - **Levels:** per spin, the scene's partial levels (`EvalStats::partial`, renumbered) are the parts' together and its
//!   frontier levels the union's (its HOMO the parts' highest, its LUMO their lowest). Partial levels are paired in ε
//!   order: the counts must agree, each ε within [`STRAY_LEVEL_TOL`] = 1.8 mHa (the largest box splitting measured,
//!   `lone` module doc; on a line the sorted pairing minimises the largest ε difference, so a failure there is not a
//!   mispairing), Σf per spin within [`level_occupation_tol`] (with nothing transferred the open part's electron count
//!   fixes it; a difference is a transferred fraction), and each f within the same tol / (2 kT). The f bound: a channel
//!   with partial levels has **exactly one open part** here — the fragment and a stray atom both open in it fail the
//!   margin above (it tests each stray atom against the fragment), and two open stray atoms never reach a check: the
//!   floor refuses mixed stray species, and copies of one species unless `closed_spin_channels` says each ends at a
//!   gap (the margin never compares two stray atoms) — so the scene's μ_σ is that
//!   part's own μ_σ up to the level shifts; if every level moves by at most tol, so does μ, so ε − μ moves by at most
//!   2 tol and f = 1/(1 + e^{(ε−μ)/kT}) by at most 2 tol × max|∂f/∂ε| = 2 tol / (4 kT). Where a level has a neighbour
//!   within 2 tol (near-degenerate: the ε order need not pair the same orbitals) an f outside the bound is **void**,
//!   not a failure. A transfer that pins a level at μ shows as a partial level the parts do not have. A scene that
//!   reports no frontier level at all while its parts do is void (a wiring slip is not a measurement).
//!
//! `Pass { residual, margin }` carries the absolute energy residual and the smallest margin (Ha).
//!
//! # The cleared region (spec §4.3, "Clearance is local"; [`Region`], [`StrayBook`])
//!
//! A [`Signature`] (sorted bonded Z, sorted stray Z, charge, the surface's reference settings; the multiplicity is
//! automatic, the floor refuses the rest) is cleared only **inside a region**: the per-feature envelope (min, max per
//! species, the existing `surrogate::Envelope`) of the bonded-part descriptors of its passing checks. The region opens
//! after [`STRAY_CHECKS`] = 3 passes, each at a bonded geometry beyond the store's thinning radius from the others
//! (`THIN_FRACTION` × `max_descriptor_distance` training spreads, the store's per-sample metric: the worst atom's
//! nearest same-species counterpart, L∞ in the model's training-spread units, `quarantine::spread_distance`). A pass
//! differing only in stray distance has the same bonded descriptors and does not count. Each check tests an identity,
//! so a fourth adds no statistical power; the region is the geometry coverage. A pass outside an open region grows it.
//! **One failing check refuses the signature for the session.** Session-only state (spec §4.6). The region lives in
//! the descriptor space of the basis it was built with: a model with another basis must clear the book.
//!
//! # Answering inside a region (spec §4.4; plan Task 4)
//!
//! [`stray_cleared`](SurrogateKs::stray_cleared) decides, before the envelope: the scene splits, every stray species is
//! in the model's basis (rule 2), the signature is `Cleared`, the floor passes **at these positions** (the signature
//! does not carry a forced multiplicity, a stray atom's distance to a face or a shell fence since, so the floor is
//! re-checked every step) and the region covers the bonded part's descriptors. Then the strict rules judge the scene
//! with the stray atoms exempt and the bonded part on its own (`evaluate`, mirrored by `would_answer`, `atom_verdicts`,
//! the readiness readout and `geometry_refusal`):
//!
//! - **5a–5e**: the stray atoms are skipped (`envelope_outside_except`); the bonded atoms' descriptors are the bonded
//!   part's own (a stray atom is no atom's neighbour), so they are judged exactly as the bonded part alone. Rule 6
//!   needs no exemption: a stray atom's force rows are zero, so its epistemic force std is exactly 0. Rule 4b's key of
//!   an isolated atom is none.
//! - **Rule 7 per bonded atom** (review I4): the prediction's floor (n s_E)² is removed exactly and the epistemic rest,
//!   the same as the bonded part's alone (B_s = 0 adds nothing to the energy row), is divided by n_bonded — rule 7 on
//!   the bonded part's own prediction, floor (n_bonded s_E)² included.
//! - **5d and the fences on the bonded part alone** (`policy::judged_part`; controller-facing ruling of Task 4): 5d's
//!   energy row is per atom over the atoms it is given (`energy_per_atom` models), so the scene's row would be the
//!   bonded part's × n_bonded/n, its ν × (n_bonded/n)²; and the quarantine and audit fences take the worst query atom's
//!   distance to a fenced geometry, so a stray atom (no counterpart in a fenced H₂) would carry a fenced bonded
//!   geometry outside. A refusal names the scene's atom.
//!
//! The answer is `predict(scene)`, the sum above (to a few ulp, the bonded forces bit for bit, the stray atoms' exactly
//! 0), with the step note "stray atoms: model for the bonded part + free atoms (composition measured, 3 checks; worst
//! residual … mHa, frontier margin … mHa)". A step outside the region is rule 5a's, as before, and one more check.
//! **Re-entry** (review M3): a stray atom back within r_cut is no longer stray and the ordinary rules judge the scene;
//! the switch to the teacher's surface is booked by `transition_jump`, as is the switch back. Rule 5b refuses there
//! (the returning atom's coordination far below any training atom's; `reentry_and_exit_book_every_switch`) even with
//! released stray samples fitted: their stray atoms are no training atoms — not in the envelope (`surrogate::envelope`
//! module doc, "Isolated atoms are not training atoms": admitted, a B = 0 would set every coordination minimum of its
//! species to 0 and switch 5b off for ordinary scenes too) and not in the force rows (their mask,
//! [`mask_stray_atoms`]: zero rows that would bias σ_f down). The audit criterion's n_bonded is plan Task 5's.
//!
//! **Draft mode** answers these scenes as before, but a cleared scene is no longer an extrapolation there: its step
//! carries no extrapolation readout (no rule refuses it) and an audit of it is an ordinary audit, not a draft guess's
//! (its sample is kept in the fit when the region covers it).
//!
//! # The checks on the auditor and the sample hold-out (spec §4.3, review C1; plan Task 3)
//!
//! - **When.** A strict-mode step refused by rule 5a (`TeachRule::IsolatedAtom`, the model consulted) whose scene
//!   splits and passes the floor falls back as before; its solve is the check's scene answer, and
//!   [`queue_stray_check`](SurrogateKs::queue_stray_check) queues the bonded part alone (`JobKind::Fragment`) and each
//!   stray atom alone (`JobKind::AtomAt`) on the background auditor's side queue — behind the job in flight, never
//!   replacing an audit. The parts keep the scene's cell, σ, ionic policy and provenance (review M5: never the
//!   fragment's own surface tag); the fragment is neutral at automatic multiplicity, each atom at its reference
//!   multiplicity. At most one check is in flight; a newer fallback does not replace it unless the scene was edited
//!   since (`reset` / `discontinuity` bump the scene-edit generation, and a check of an older one is dropped silently
//!   when its answer arrives). No check is queued for a refused signature, inside a cleared region, or at a bonded
//!   geometry that would not count (not distinct from the region's points). **Without a background auditor** (the CPU
//!   teacher's synchronous audits) no check runs: a synchronous fragment solve would stall the step (plan ruling).
//! - **Judged.** When every part has answered, [`judge`] gives the verdict and [`StrayBook::record`] books it at the
//!   bonded descriptors under the model's basis; one Activity event says what it did (`TeachOutcome::StrayChecking`,
//!   `StrayCheckingNew`, `StrayRefused`; a void check `Discarded`, a failed part solve `Failed`).
//! - **The hold-out.** A kept sample with stray atoms (`keep_sample_masked`) is held out of the fit
//!   (`quarantine::Q_STRAY`, never written) unless a cleared region of its signature covers its bonded geometry; a
//!   floor-refused one is never kept (`TeachOutcome::StrayNotKept`). A pass releases the check's own sample; a cleared
//!   signature releases every held sample of it its region covers
//!   ([`release_stray_samples`](SurrogateKs::release_stray_samples)); a refused one holds them for good. Loaded
//!   samples are held the same way (`hold_out_stray`), never rewritten. The sampling pass's samples go through the
//!   same rules (`add_pass_sample`: a floor refusal is not kept, any other is held unless covered).
//! - **What a released sample carries (rule 7, physics honesty).** Its fit label E − Σ_i E_ref(Z_i) holds, besides the
//!   bonded part's energy, E_atom(R_s) − E_ref(Z_s) for each stray atom: that atom's own confinement and egg-box term at
//!   R_s against the reference. The check cannot see it — it solves the atom at R_s, so the term cancels in (C) — and
//!   the stray atom has no features, so the fit attributes it to the bonded part's coefficients. It is bounded by the
//!   box rule (a stray atom ≥ 4 Bohr from a face, the check's solves within `box_affected_face_density`): of order
//!   1 mHa (1.8 mHa for a lone H at the 4-Bohr edge, `lone` module doc), the size of any trained sample's box error.
//! - **The book's lifetime.** Session-only (spec §4.6), and built in one basis's descriptor space: a model with
//!   another basis or surface, a Forget and an installed model clear it, drop the check in flight and release
//!   nothing ([`stray_book_current`](SurrogateKs::stray_book_current); controller ruling B).
use super::lone::{closed_spin_channels, compensated_sum, reference_settings_key};
use super::quarantine::spread_distance;
use super::*;
use sim::{Frontier, PartialLevel};
use std::collections::HashMap;

/// Passing checks at distinct bonded geometries before a signature's region opens (spec §4.3; the reviewer's K).
pub(super) const STRAY_CHECKS: usize = 3;

/// A stray atom nearer than this to a face (Bohr) is floor-refused before a solve: the box rule's edge (spec §4.2).
pub(super) const STRAY_FACE_MARGIN: f64 = 4.0;

/// The level-matching tolerance (Ha): the largest box splitting of two copies' levels measured (an H 4 Bohr from a
/// face; `lone` module doc).
pub(super) const STRAY_LEVEL_TOL: f64 = 1.8e-3;

/// The frontier margin (Ha): 27.6 kT = 2 ln(1/`PARTIAL_OCCUPATION_MIN`) kT, 138 mHa at kT = 5 mHa (module doc).
pub(super) fn frontier_margin_min() -> f64 {
    27.6 * qm::ELECTRON_KT
}

/// The occupation-matching tolerance of a partial level: [`STRAY_LEVEL_TOL`] / (2 kT) (module doc, "Levels").
pub(super) fn level_occupation_tol() -> f64 {
    STRAY_LEVEL_TOL / (2.0 * qm::ELECTRON_KT)
}

/// A scene's atoms split into the bonded part and the stray atoms (indices into the scene, ascending).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Split {
    pub bonded: Vec<usize>,
    pub stray: Vec<usize>,
}

/// The split at `r_cut` (Bohr; a neighbour is strictly inside, as `no_neighbours`): an atom is stray when no other atom
/// is its neighbour. `None` unless there are ≥ 2 atoms, at least one stray and at least one bonded (every atom alone is
/// rule 0b's, `lone`).
pub(super) fn split(pos: &[[f64; 3]], r_cut: f64) -> Option<Split> {
    if pos.len() < 2 || no_neighbours(pos, r_cut) {
        return None;
    }
    let (stray, bonded): (Vec<usize>, Vec<usize>) =
        (0..pos.len()).partition(|&i| (0..pos.len()).filter(|&j| j != i).all(|j| no_neighbours(&[pos[i], pos[j]], r_cut)));
    (!stray.is_empty() && !bonded.is_empty()).then_some(Split { bonded, stray })
}

impl Split {
    /// The bonded part alone: its positions and species in [`Split::bonded`]'s order (module doc, "Answering inside a
    /// region": what rule 5d and the fences judge in a cleared scene).
    pub(super) fn bonded_part(&self, pos: &[[f64; 3]], z: &[u8]) -> (Vec<[f64; 3]>, Vec<u8>) {
        (self.bonded.iter().map(|&i| pos[i]).collect(), self.bonded.iter().map(|&i| z[i]).collect())
    }

    /// The scene's index of the bonded part's atom `j` (a refusal names the scene's atom).
    pub(super) fn scene_atom(&self, j: Option<usize>) -> Option<usize> {
        j.and_then(|j| self.bonded.get(j).copied())
    }

    /// The bonded part's per-atom values `v` as the scene's `n` atoms', `fill` at the stray atoms.
    pub(super) fn scene_values<T: Clone>(&self, v: &[T], n: usize, fill: T) -> Vec<T> {
        let mut out = vec![fill; n];
        for (&i, x) in self.bonded.iter().zip(v) {
            out[i] = x.clone();
        }
        out
    }
}

/// What a clearance or a refusal is keyed by (spec §4.1): the sorted Z of the bonded part and of the stray atoms, the
/// scene's charge, and the surface (the reference settings `lone::same_reference_settings` compares: grid, h, σ,
/// tolerance, teacher, ionic). The multiplicity is automatic: the floor refuses a forced one.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(super) struct Signature {
    pub bonded_z: Vec<u8>,
    pub stray_z: Vec<u8>,
    pub charge: i32,
    pub surface: String,
}

impl Signature {
    /// The signature of a scene with species `z` split by `split`, of charge `charge`, solved at `prov`.
    pub(super) fn new(z: &[u8], split: &Split, charge: i32, prov: &Provenance) -> Signature {
        let sorted = |idx: &[usize]| {
            let mut v: Vec<u8> = idx.iter().map(|&i| z[i]).collect();
            v.sort_unstable();
            v
        };
        Signature { bonded_z: sorted(&split.bonded), stray_z: sorted(&split.stray), charge, surface: reference_settings_key(prov) }
    }
}

/// Element symbols, comma-separated.
fn symbols(z: &[u8]) -> String {
    z.iter().map(|&zi| symbol(zi)).collect::<Vec<_>>().join(", ")
}

/// The static floor (module doc, spec §4.2): the reason words, or `None` when the scene may be checked. `m_ref(z)` is the
/// reference multiplicity of Z and its partial-level flag (`AtomReference::fractional`) on this surface — `lone`'s
/// source; `fenced(a, b)` whether a shell check fenced the pair (a, b); `cell_faces(r)` the distance (Bohr) from `r` to
/// the cell's nearest face.
pub(super) fn floor_refusal(
    scene: &Scene,
    split: &Split,
    m_ref: &dyn Fn(u8) -> Option<(u32, Option<bool>)>,
    fenced: &dyn Fn(u8, u8) -> bool,
    cell_faces: &dyn Fn([f64; 3]) -> f64,
) -> Option<String> {
    let z: Vec<u8> = scene.nuclei.iter().map(|n| n.z).collect();
    let stray_z: Vec<u8> = split.stray.iter().map(|&i| z[i]).collect();
    if scene.charge != 0 {
        return Some(format!("charge: the scene's charge is {:+}, and the stray atoms' solves are neutral atoms", scene.charge));
    }
    if let Some(m) = scene.multiplicity {
        return Some(format!(
            "spin: multiplicity {m} is forced — stray atoms are cleared at automatic only (a forced-M store's samples are whole-scene samples at that M)"
        ));
    }
    // the spin rule: the parts' excess spins aligned
    let electrons: u32 = z.iter().map(|&zi| zi as u32).sum();
    let frag_electrons: u32 = split.bonded.iter().map(|&i| z[i] as u32).sum();
    let (have, m_frag) = (1 + electrons % 2, 1 + frag_electrons % 2);
    let mut want = m_frag;
    let mut partial = Vec::with_capacity(stray_z.len());
    for &zs in &stray_z {
        match m_ref(zs) {
            Some((m, p)) if m >= 1 => {
                want += m - 1;
                partial.push(p);
            }
            _ => return Some(format!("spin: the spin state of the {} reference is not known here (a sampling pass records it)", symbol(zs))),
        }
    }
    if have != want {
        return Some(format!(
            "spin: {} stray {} at automatic — multiplicity {have}, but the bonded part's {m_frag} with the stray atoms' reference spins aligned is {want}",
            stray_z.len(),
            symbols(&stray_z)
        ));
    }
    // two or more stray atoms: one species, known to end at a gap
    if stray_z.len() >= 2 {
        let mut species = stray_z.clone();
        species.sort_unstable();
        species.dedup();
        if species.len() > 1 {
            return Some(format!(
                "mixed stray species ({}): with their spins aligned one atom can lose charge to another's partly empty shell under the cell-wide filling (LSDA's separated-atom charge transfer)",
                symbols(&species)
            ));
        }
        if !closed_spin_channels(species[0], partial[0]) {
            let why = match partial[0] {
                Some(_) => "its reference has a partly occupied level",
                None => "whether its reference's shell is open is not recorded here",
            };
            return Some(format!(
                "{} stray {} atoms: {why} — one Fermi level spans the box, so copies whose levels the box splits can fill one copy's shell first",
                stray_z.len(),
                symbol(species[0])
            ));
        }
    }
    // a shell-check fence on (a stray species, any other atom's species)
    for &s in &split.stray {
        if let Some(j) = (0..z.len()).find(|&j| j != s && fenced(z[s], z[j])) {
            return Some(format!(
                "fence: a shell check measured {}–{} still interacting near the cutoff, so zero interaction beyond it is not the teacher's",
                symbol(z[s]),
                symbol(z[j])
            ));
        }
    }
    // a stray atom near a face (the box's confinement, not the free atom)
    for &s in &split.stray {
        let d = cell_faces(scene.nuclei[s].pos);
        if !(d >= STRAY_FACE_MARGIN) {
            return Some(format!("box: the stray {} is {d:.2} Bohr from a face (within {STRAY_FACE_MARGIN} Bohr: box-affected)", symbol(z[s])));
        }
    }
    None
}

/// One solve of a composition check: the scene's own, the bonded part's alone or one stray atom's alone (forces in the
/// part's atom order: the bonded part's in [`Split::bonded`]'s order, an atom's one force). `clean`: no warning,
/// trainable, face density within the box rule.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct PartAnswer {
    pub energy: f64,
    pub forces: Vec<[f64; 3]>,
    pub frontier: [Frontier; 2],
    pub partial: Vec<PartialLevel>,
    pub clean: bool,
}

/// A composition check's verdict (module doc): `Pass` with the absolute energy residual and the smallest frontier margin
/// (Ha); `Fail` with the failed test and its numbers (words); `Void` not counted, and why.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum CheckVerdict {
    Pass { residual: f64, margin: f64 },
    Fail(String),
    Void(String),
}

/// The words of a void for a level a test needs that the solve did not compute (ruling 4: never a pass).
const NOT_COMPUTED: &str = "frontier level not computed";

const SPIN: [&str; 2] = ["↑", "↓"];

/// The composition check (module doc, spec §4.3): `scene` the scene's own solve, `frag` the bonded part alone, `atoms`
/// one solve per stray atom in [`Split::stray`]'s order.
pub(super) fn judge(policy: &FallbackPolicy, scene: &PartAnswer, frag: &PartAnswer, atoms: &[PartAnswer], split: &Split) -> CheckVerdict {
    let n = split.bonded.len() + split.stray.len();
    if atoms.len() != split.stray.len() || frag.forces.len() != split.bonded.len() || scene.forces.len() != n || atoms.iter().any(|a| a.forces.len() != 1)
    {
        debug_assert!(false, "stray::judge: the parts' answers do not match the split");
        return CheckVerdict::Void("the parts' answers do not match the split".into());
    }
    if !scene.clean || !frag.clean || atoms.iter().any(|a| !a.clean) {
        return CheckVerdict::Void("a check solve is not clean (a warning, untrainable, or box-affected)".into());
    }
    // energy, absolute: the error belongs to the stray atoms
    let mut terms = Vec::with_capacity(2 + atoms.len());
    terms.push(scene.energy);
    terms.push(-frag.energy);
    terms.extend(atoms.iter().map(|a| -a.energy));
    let residual = compensated_sum(&terms).abs();
    let tol = policy.energy_std_per_atom * split.stray.len() as f64;
    if !(residual <= tol) {
        return CheckVerdict::Fail(format!(
            "energy: |E_scene − E_frag − Σ E_atom| = {:.3} mHa > {:.3} mHa ({} stray × the energy std per atom)",
            residual * 1e3,
            tol * 1e3,
            split.stray.len()
        ));
    }
    // forces: bonded against the fragment alone, stray against its own solve
    for (k, &i) in split.bonded.iter().enumerate() {
        let (d, bound) = policy.force_residual(&scene.forces[i], &frag.forces[k]);
        if !(d <= bound) {
            return CheckVerdict::Fail(format!("force: bonded atom {i} differs from the fragment alone by {d:.2e} Ha/Bohr (bound {bound:.2e})"));
        }
    }
    for (a, &i) in atoms.iter().zip(&split.stray) {
        let (d, bound) = policy.force_residual(&scene.forces[i], &a.forces[0]);
        if !(d <= bound) {
            return CheckVerdict::Fail(format!("force: stray atom {i} differs from its own solve by {d:.2e} Ha/Bohr (bound {bound:.2e})"));
        }
    }
    // frontier margin, per spin and per stray atom, both directions. Every level with f > PARTIAL_OCCUPATION_MIN donates
    // and every level with f < 1 − PARTIAL_OCCUPATION_MIN accepts (`lone`'s gap criterion), so a part's partial levels
    // are both: a partial level at f ≥ ½ is the HOMO and would otherwise never be tested as an acceptor (review I1)
    let mut missing = false;
    let mut margin = f64::INFINITY;
    let floor = frontier_margin_min();
    for (spin, word) in SPIN.iter().enumerate() {
        for (a, &i) in atoms.iter().zip(&split.stray) {
            let pairs = [
                (donor(a, spin), acceptor(frag, spin), "fragment acceptor − stray donor"),
                (donor(frag, spin), acceptor(a, spin), "stray acceptor − fragment donor"),
            ];
            for (give, take, what) in pairs {
                // no occupied or partial level: an empty channel donates nothing
                let Some(give) = give else { continue };
                let Some(take) = take else {
                    missing = true;
                    continue;
                };
                let m = take - give;
                if !(m >= floor) {
                    return CheckVerdict::Fail(format!(
                        "frontier margin: {what} {word} (stray atom {i}) = {:.1} mHa < {:.1} mHa (27.6 kT): charge can move across the gap",
                        m * 1e3,
                        floor * 1e3
                    ));
                }
                margin = margin.min(m);
            }
        }
    }
    // levels: the scene's frontier and partial levels are the union of the parts'
    let parts: Vec<&PartAnswer> = std::iter::once(frag).chain(atoms).collect();
    let reported = |p: &PartAnswer| p.frontier.iter().any(|f| f.homo.is_some() || f.lumo.is_some());
    if !reported(scene) && parts.iter().any(|p| reported(p)) {
        // the scene's solve reported no level at all: a wiring slip, never a measured refusal
        return CheckVerdict::Void(NOT_COMPUTED.into());
    }
    let mut ambiguous = false;
    for (spin, word) in SPIN.iter().enumerate() {
        // partial levels, renumbered: one per part level, each ε within the box splitting, Σf conserved, each f within
        // the occupation tolerance (module doc, "Levels")
        let mut have: Vec<&PartialLevel> = scene.partial.iter().filter(|l| l.spin as usize == spin).collect();
        let mut want: Vec<&PartialLevel> = parts.iter().flat_map(|p| &p.partial).filter(|l| l.spin as usize == spin).collect();
        have.sort_by(|a, b| a.eps.total_cmp(&b.eps));
        want.sort_by(|a, b| a.eps.total_cmp(&b.eps));
        if have.len() != want.len() {
            return CheckVerdict::Fail(format!("levels: the scene has {} partly occupied {word} levels, the parts {}", have.len(), want.len()));
        }
        for (h, w) in have.iter().zip(&want) {
            if !((h.eps - w.eps).abs() <= STRAY_LEVEL_TOL) {
                return CheckVerdict::Fail(format!(
                    "levels: a partly occupied {word} level is at {:.1} mHa in the scene, {:.1} mHa in its part (beyond {:.1} mHa)",
                    h.eps * 1e3,
                    w.eps * 1e3,
                    STRAY_LEVEL_TOL * 1e3
                ));
            }
        }
        let (f_have, f_want) = (have.iter().map(|l| l.f).sum::<f64>(), want.iter().map(|l| l.f).sum::<f64>());
        if !((f_have - f_want).abs() <= level_occupation_tol()) {
            return CheckVerdict::Fail(format!(
                "levels: the {word} partial levels hold Σf = {f_have:.3} in the scene, {f_want:.3} in the parts (charge moved between them)"
            ));
        }
        for (k, (h, w)) in have.iter().zip(&want).enumerate() {
            if !((h.f - w.f).abs() <= level_occupation_tol()) {
                if crowded(&have, k) || crowded(&want, k) {
                    // near-degenerate levels: the ε order need not pair the same orbitals, so f cannot be compared
                    ambiguous = true;
                    continue;
                }
                return CheckVerdict::Fail(format!(
                    "levels: a partly occupied {word} level at {:.1} mHa has f {:.3} in the scene, {:.3} in its part",
                    h.eps * 1e3,
                    h.f,
                    w.f
                ));
            }
        }
        // frontier levels: the scene's HOMO the parts' highest, its LUMO their lowest
        let homo = parts.iter().filter_map(|p| p.frontier[spin].homo).reduce(f64::max);
        match (scene.frontier[spin].homo, homo) {
            (None, None) => {}
            (Some(s), Some(u)) if (s - u).abs() <= STRAY_LEVEL_TOL => {}
            (s, u) => return CheckVerdict::Fail(format!("levels: the scene's HOMO {word} is {}, the parts' {}", level_words(s), level_words(u))),
        }
        let lumos: Option<Vec<f64>> = parts.iter().map(|p| p.frontier[spin].lumo).collect();
        match (scene.frontier[spin].lumo, lumos.and_then(|l| l.into_iter().reduce(f64::min))) {
            (Some(s), Some(u)) if !((s - u).abs() <= STRAY_LEVEL_TOL) => {
                return CheckVerdict::Fail(format!("levels: the scene's LUMO {word} is {}, the parts' {}", level_words(Some(s)), level_words(Some(u))));
            }
            (Some(_), Some(_)) => {}
            _ => missing = true,
        }
    }
    if missing {
        return CheckVerdict::Void(NOT_COMPUTED.into());
    }
    if ambiguous {
        return CheckVerdict::Void("near-degenerate partial levels cannot be paired between the scene and its parts".into());
    }
    CheckVerdict::Pass { residual, margin }
}

/// A part's highest donating level in `spin` (Ha): its HOMO or its highest partial level, whichever is higher; `None`
/// for a channel with neither (nothing to donate).
fn donor(p: &PartAnswer, spin: usize) -> Option<f64> {
    p.partial.iter().filter(|l| l.spin as usize == spin).map(|l| l.eps).chain(p.frontier[spin].homo).reduce(f64::max)
}

/// A part's lowest accepting level in `spin` (Ha): its LUMO or its lowest partial level, whichever is lower; `None`
/// when neither was computed. An uncomputed LUMO lies above every computed level, so a partial level alone answers.
fn acceptor(p: &PartAnswer, spin: usize) -> Option<f64> {
    p.partial.iter().filter(|l| l.spin as usize == spin).map(|l| l.eps).chain(p.frontier[spin].lumo).reduce(f64::min)
}

/// Level `k` of the ε-sorted `levels` has a neighbour within 2 × [`STRAY_LEVEL_TOL`] (two levels each moved by up to
/// the tolerance can swap order): its ε rank does not identify it.
fn crowded(levels: &[&PartialLevel], k: usize) -> bool {
    let near = |j: usize| levels.get(j).is_some_and(|l| (l.eps - levels[k].eps).abs() <= 2.0 * STRAY_LEVEL_TOL);
    (k > 0 && near(k - 1)) || near(k + 1)
}

fn level_words(e: Option<f64>) -> String {
    e.map_or_else(|| "absent".into(), |e| format!("{:.1} mHa", e * 1e3))
}

/// A signature's cleared region (module doc): the per-feature envelope of the bonded-part descriptors of its passing
/// checks, and the points themselves (per point, per bonded atom: its species index and descriptor) for the
/// distinctness test.
#[derive(Clone, Debug)]
pub(super) struct Region {
    env: Envelope,
    points: Vec<Vec<(usize, Vec<f64>)>>,
}

impl Region {
    pub(super) fn new(basis: &Basis) -> Region {
        Region { env: Envelope::new(basis), points: Vec::new() }
    }

    /// Add one bonded geometry: its atoms' descriptors and species indices (the region's basis).
    pub(super) fn add(&mut self, bonded_desc: &[Vec<f64>], species: &[usize]) {
        for (&s, b) in species.iter().zip(bonded_desc) {
            self.env.add_atom(s, b);
        }
        self.points.push(species.iter().copied().zip(bonded_desc.iter().cloned()).collect());
    }

    /// The geometry lies beyond `thin` from every stored point in the store's thinning metric (module doc): per point,
    /// the worst atom's distance to its nearest same-species counterpart, L∞ in `units`' training spreads (the model's
    /// envelope, as `nearest_stored`). A species `units` never saw cannot be measured: not distinct (the conservative
    /// side: such a check never counts towards [`STRAY_CHECKS`]).
    pub(super) fn distinct(&self, bonded_desc: &[Vec<f64>], species: &[usize], units: &Envelope, thin: f64) -> bool {
        debug_assert_eq!(self.env.n_features, units.n_features, "Region::distinct: the units envelope is of another basis");
        if species.iter().any(|&s| units.count.get(s).is_none_or(|&c| c == 0)) {
            return false;
        }
        self.points.iter().all(|point| {
            let d = species.iter().zip(bonded_desc).fold(0.0f64, |worst, (&qs, qd)| {
                worst.max(point.iter().filter(|(t, _)| *t == qs).map(|(t, b)| spread_distance(units, *t, qd, b)).fold(f64::INFINITY, f64::min))
            });
            d > thin
        })
    }

    /// Every bonded atom lies within its species' per-feature [min, max] of the region; a species absent from the
    /// region (or an empty geometry) is not contained.
    pub(super) fn contains<D: AsRef<[f64]>>(&self, bonded_desc: &[D], species: &[usize]) -> bool {
        let p = self.env.n_features;
        !bonded_desc.is_empty()
            && species.len() == bonded_desc.len()
            && species.iter().zip(bonded_desc).all(|(&s, b)| {
                let b = b.as_ref();
                self.env.count.get(s).is_some_and(|&c| c > 0)
                    && b.len() == p
                    && b.iter().enumerate().all(|(k, &v)| self.env.min[s * p + k] <= v && v <= self.env.max[s * p + k])
            })
    }
}

/// A signature's state this session (module doc). `Checking` and `Cleared` carry the worst energy residual and the
/// smallest frontier margin of the passes so far (Ha), the step note's numbers.
#[derive(Clone, Debug)]
pub(super) enum StrayState {
    Checking { passes: usize, region: Region, residual: f64, margin: f64 },
    /// `passes`: every passing check booked so far (the [`STRAY_CHECKS`] that opened the region and each that grew it).
    Cleared { region: Region, residual: f64, margin: f64, passes: usize },
    Refused(String),
}

/// The session's signatures and their states (module doc; session-only, spec §4.6).
#[derive(Clone, Debug, Default)]
pub(super) struct StrayBook {
    map: HashMap<Signature, StrayState>,
}

impl StrayBook {
    pub(super) fn state(&self, s: &Signature) -> Option<&StrayState> {
        self.map.get(s)
    }

    /// Book a check's verdict at bonded geometry `desc`/`species` (under `basis`; `units` the model's training envelope
    /// and `thin` the store's thinning radius in its spreads, for [`Region::distinct`]). Pass: at a distinct geometry
    /// one more pass and the region grows, opening at [`STRAY_CHECKS`]; an open region grows by any pass. Fail: refused
    /// for the session. Void, or anything after a refusal: no change.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record(&mut self, s: Signature, v: CheckVerdict, desc: &[Vec<f64>], species: &[usize], basis: &Basis, units: &Envelope, thin: f64) {
        let (r, m) = match v {
            CheckVerdict::Void(_) => return,
            CheckVerdict::Fail(why) => {
                self.map.insert(s, StrayState::Refused(why));
                return;
            }
            CheckVerdict::Pass { residual, margin } => (residual, margin),
        };
        let st = self.map.entry(s).or_insert_with(|| StrayState::Checking {
            passes: 0,
            region: Region::new(basis),
            residual: 0.0,
            margin: f64::INFINITY,
        });
        let next = match std::mem::replace(st, StrayState::Refused(String::new())) {
            StrayState::Refused(why) => StrayState::Refused(why),
            StrayState::Checking { mut passes, mut region, residual, margin } => {
                let (residual, margin) = (residual.max(r), margin.min(m));
                if region.distinct(desc, species, units, thin) {
                    passes += 1;
                    region.add(desc, species);
                }
                if passes >= STRAY_CHECKS {
                    StrayState::Cleared { region, residual, margin, passes }
                } else {
                    StrayState::Checking { passes, region, residual, margin }
                }
            }
            StrayState::Cleared { mut region, residual, margin, passes } => {
                region.add(desc, species);
                StrayState::Cleared { region, residual: residual.max(r), margin: margin.min(m), passes: passes + 1 }
            }
        };
        *st = next;
    }

    /// Refuse `s` for the session (a floor refusal, a failed audit), whatever its state.
    pub(super) fn refuse(&mut self, s: &Signature, why: String) {
        self.map.insert(s.clone(), StrayState::Refused(why));
    }

    /// Forget every signature (a new model, a surface change, a reset).
    pub(super) fn clear(&mut self) {
        self.map.clear();
    }
}

mod wiring;
pub(super) use wiring::{mask_stray_atoms, part_answer, signature_label, StrayCheck};
