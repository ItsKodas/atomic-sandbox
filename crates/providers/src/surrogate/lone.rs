//! The lone-atom rule (user request 2026-11-12; rule 0b of the decision list in the module doc):
//! in strict mode, a scene in which **no atom has a neighbour within r_cut** is answered from the
//! isolated-atom references, whether or not a model exists and whatever the trust verdict says.
//! Atomic units (Hartree, Bohr).
//!
//! # Why the answer is the model's own
//!
//! The model is E = Σ_i [E_ref(Z_i) + Σ_k c̃_{s_i,k} B_{i,k}] with every descriptor B_i a sum over
//! neighbours j of terms carrying f_c(r_ij), and the standardisation mean is 0 with no intercept
//! (`surrogate::fit`, "Isolated-atom limit"). With no neighbour inside r_cut every B_i is empty, so
//!
//! ```text
//!   E_model = Σ_i E_ref(Z_i)          (summed in the model's order, Neumaier: bitwise the model's)
//!   F_i     = −∂E/∂R_i = 0            (the references do not depend on the positions)
//! ```
//!
//! whatever the coefficients are — an untrusted fit, or no fit at all, answers the same. A DFT
//! solve there buys nothing for the model: it is never kept (`no_neighbours`, the keep rule), and
//! what it would report differs from Σ E_ref only by the isolated atoms' grid and box artefacts,
//! which no model can or should learn: the egg-box ripple of an off-centre atom (≈ 1e-6 Ha, measured)
//! and the Dirichlet walls' confinement, which grows toward the faces — a lone H in the default
//! 12-Bohr box is +1.8e-3 Ha (force 2.6e-3 Ha/Bohr toward the centre) 4 Bohr from the nearest face,
//! the box rule's edge, and +3.5e-2 Ha 2 Bohr from it (`diag_lone_h_wobble`; `validate/references.md`).
//! There the reference — the free atom — is the more physical answer of the two.
//!
//! # The energy's surface
//!
//! Per scene the references are the model's (`SurrogateKs::refs`, one per species, of the store's
//! surface). In collection mode they are the **canonical** ones (`shared` module doc, "References
//! per box"): a teacher answer in box n is reported as E − Σ_i (ref(Z_i, n) − ref_canonical(Z_i)),
//! so DFT's lone atoms in box n would be reported at Σ ref_canonical + wobble — the reference answer
//! is that without the wobble, continuous with every other report in that box. The answer is booked
//! like a model answer (`on_model`; `Source::Reference` is on `Surface::Model`), with transitions to
//! and from teacher-surface reports booked along the forces (`transition_jump`), and a canonical
//! reference arriving under a reference report books its change (`book_reference_arrival`).
//!
//! # When it stands aside (the step takes today's path)
//!
//! - **Draft mode** and the **relax verification** (the teacher was asked for): checked in `evaluate`.
//! - **A charged scene**: the references are neutral atoms; lone ions are not their sum.
//! - **Another surface**: per scene, a store/model provenance that differs from the scene's in grid,
//!   h, σ, tolerance, teacher or ionic surface (charge and multiplicity are the rules below — the
//!   references are neutral atoms at their own spin state, whatever the scene's); no store and no
//!   model (no surface pinned) stands aside too. In collection mode, a scene the collection refuses.
//! - **A missing reference**: per scene a species without `refs`; in collection mode a species
//!   whose reference of this box is not solved yet (`box_refs_missing`, rule 3's own condition — the
//!   provider has asked the auditor for it) or that has no canonical one.
//! - **The multiplicity** (rule 7; derivation below): the scene's 2S + 1 (its own, or the solver's
//!   automatic lowest one, 1 + (Σ Z mod 2)) must be the **parallel-spin product** of the atoms'
//!   reference states, M* = 1 + Σ_i (m_i − 1). An atom whose reference multiplicity is unknown here
//!   stands aside.
//! - **More than one species** among two or more lone atoms (below).
//!
//! # The multiplicity rule
//!
//! The teacher solves collinear LSDA with fixed (N↑, N↓) = ((N + M − 1)/2, (N − M + 1)/2) and
//! per-spin aufbau over the **whole** cell (degenerate shells shared, `qm::scf` "Occupations"). For
//! well-separated atoms the Kohn-Sham Hamiltonian is block-diagonal to the overlap of their tails,
//! so the SCF energy is Σ_i E_i(n↑_i, n↓_i) with the per-atom spin counts the global aufbau hands
//! out. Each reference E_ref(Z) is the atom at its lowest-energy multiplicity m (ruling S9: n↑ − n↓ =
//! m − 1). Σ_i E_ref is therefore reached only by an (N↑, N↓) that the global aufbau can split into
//! every atom's own (n↑_i, n↓_i):
//!
//! - **Parallel product, M* = 1 + Σ (m_i − 1)** — every atom's excess spin up. For **one species**
//!   the spin-up spectrum is the union of identical atomic spectra (split only by the egg-box
//!   ripple, ≪ the shell gaps and the sharing window τ_c = 0.1 Ha), so aufbau fills each atom's
//!   levels exactly as in its own reference — a partly filled shell shared over k atoms' copies at
//!   the same fraction per level as in one. Two lone H at M = 3: N↑ = 2, both 1s↑, E = 2 E_ref.
//! - **Any other M** puts at least one atom into another spin state, or — the measured case — the
//!   teacher's SCF converges to the **spin-restricted** delocalised state: two H at M = 1 (also
//!   what "automatic" picks for an even electron count) is the restricted H₂ singlet, which lies
//!   4.6e-2 Ha above 2 E_ref at r = 6.4 Bohr (rule 5a's measurement, module doc). An antiparallel
//!   broken-symmetry product would also reach Σ E_ref in exact LSDA, but nothing steers the cold
//!   start there; refused (if in doubt, DFT).
//! - **Mixed species** stand aside even at M*: with the excess spins aligned, an atom whose
//!   highest occupied spin-up level lies above another atom's partly empty spin-up shell (an alkali
//!   2s↑ near −0.12 Ha against a C 2p↑ near −0.2 Ha in LSDA) loses charge to it under the global
//!   aufbau — LSDA's separated-atom charge-transfer error — and DFT reports neither neutral atom.
//!   Whether that happens depends on eigenvalues the provider does not have, so the rule does not
//!   guess.
//! - **One atom**: M* = m. Under "automatic" that holds only when the reference is the lowest
//!   multiplicity of its parity (H, and every atom the scan solves lowest at 1 or 2).
//!
//! The reference multiplicity m of species Z on this surface is the one its scan chose when the
//! provider knows it (`ref_cache`, keyed by the atom's solve settings: a sampling pass's scan), else
//! the only one the scan can solve when `allowed_multiplicities(Z)` has a single entry (Z = 1: 2 —
//! hydrogen's reference needs no record), else unknown → stand aside.
use super::*;

/// What the lone-atom rule decides at a scene (module doc above).
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Lone {
    /// Some atom has a neighbour within r_cut (or the scene is empty): the rule does not apply.
    NotLone,
    /// Answered from the references: E = Σ_i E_ref(Z_i) on the reported surface, every force 0.
    Answer { energy: f64 },
    /// Lone atoms the references cannot answer, and why (words): the step takes today's path.
    StandAside(String),
}

/// Neumaier's compensated sum in the order given — the same additions as the model's energy
/// (`surrogate::energy_and_forces`), so a reference answer equals the model's at a lone geometry
/// bit for bit when the references are the model's.
fn compensated_sum(values: &[f64]) -> f64 {
    let (mut sum, mut comp) = (0.0f64, 0.0f64);
    for &x in values {
        let t = sum + x;
        if sum.abs() >= x.abs() {
            comp += (sum - t) + x;
        } else {
            comp += (x - t) + sum;
        }
        sum = t;
    }
    sum + comp
}

/// The grid, h, σ, tolerance, teacher and ionic surface of `a` and `b` agree — the settings an
/// isolated-atom reference is solved at (`sim::references::ReferenceKey` without Z); charge and
/// multiplicity are not compared (the references are neutral atoms at their own spin state).
fn same_reference_settings(a: &Provenance, b: &Provenance) -> bool {
    a.grid_n == b.grid_n && a.h == b.h && a.sigma == b.sigma && a.tol == b.tol && a.teacher == b.teacher && a.ionic == b.ionic
}

/// The note of a step the rule answered (`Source::Reference`): the solve log's and the chip's words.
pub(super) fn reference_note(r_cut: f64) -> String {
    format!(
        "surrogate: lone atoms: answered from the atom references — no atom has a neighbour within r_cut = {r_cut:.2} Bohr, so E = Σ E_ref and every force is 0, the model's own answer here whatever its trust (DFT in this box would differ only by its grid and box artefacts: the egg-box ripple, ~1e-6 Ha, and the walls' confinement, ~2e-3 Ha 4 Bohr from a face and more nearer; nothing is learned)"
    )
}

impl SurrogateKs {
    /// The lone-atom rule at `scene` solved at `prov` (`scene_provenance`): module doc above. Strict
    /// mode, draft mode and the relax verification are the caller's to check. Read-only.
    pub(super) fn lone_reference(&self, scene: &Scene, prov: &Provenance) -> Lone {
        let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        let r_cut = self.cutoff();
        if pos.is_empty() || !no_neighbours(&pos, r_cut) {
            return Lone::NotLone;
        }
        let z: Vec<u8> = scene.nuclei.iter().map(|n| n.z).collect();
        if scene.charge != 0 {
            return Lone::StandAside(format!("the scene's charge is {:+}: the references are neutral atoms", scene.charge));
        }
        let mut species = z.clone();
        species.sort_unstable();
        species.dedup();
        if z.len() > 1 && species.len() > 1 {
            return Lone::StandAside(
                "lone atoms of more than one species: with their spins aligned one atom can lose charge to another's partly empty shell under the cell-wide aufbau (LSDA's separated-atom charge transfer), so DFT's answer need not be Σ E_ref".into(),
            );
        }
        // the references of this surface, per atom, in the scene's order
        let mut refs = Vec::with_capacity(z.len());
        match self.shared.as_ref() {
            Some(sh) => {
                if let Some(why) = self.collection_refuses(prov) {
                    return Lone::StandAside(why);
                }
                if let Some(&zm) = self.box_refs_missing(&z, prov.grid_n).first() {
                    return Lone::StandAside(format!("the isolated-atom reference of {} in this {}³ box is not solved yet", symbol(zm), prov.grid_n));
                }
                for &zi in &z {
                    match shared::canonical_reference(sh, zi) {
                        Some(e) => refs.push(e),
                        None => return Lone::StandAside(format!("no canonical {} reference yet", symbol(zi))),
                    }
                }
            }
            None => {
                let pinned = self.store.as_ref().map(|s| &s.provenance).or_else(|| self.model.as_ref().map(|m| &m.provenance));
                let Some(p) = pinned else {
                    return Lone::StandAside("no references on this surface yet (nothing learned or loaded)".into());
                };
                if !same_reference_settings(p, prov) {
                    return Lone::StandAside(format!("the model's references are of another surface ({}; the scene: {})", p.describe(), prov.describe()));
                }
                for &zi in &z {
                    match self.refs.get(&zi) {
                        Some(&e) => refs.push(e),
                        None => return Lone::StandAside(format!("no isolated-atom reference of {} yet", symbol(zi))),
                    }
                }
            }
        }
        // the multiplicity: the parallel-spin product of the atoms' reference states
        let mut want = 1u32;
        for &zi in &z {
            match self.reference_multiplicity(zi, prov) {
                Some(m) => want += m - 1,
                None => return Lone::StandAside(format!("the spin state of the {} reference is not known here (a sampling pass records it)", symbol(zi))),
            }
        }
        let electrons: u32 = z.iter().map(|&zi| zi as u32).sum();
        let have = scene.multiplicity.unwrap_or(1 + electrons % 2);
        if have != want {
            let how = if scene.multiplicity.is_some() { "" } else { " (automatic)" };
            return Lone::StandAside(format!(
                "multiplicity {have}{how} is not the atoms' reference spin states aligned ({want}): DFT's separated-atom limit at {have} is not Σ E_ref (two lone H at 1 are the restricted singlet, 4.6e-2 Ha above at 6.4 Bohr)"
            ));
        }
        Lone::Answer { energy: compensated_sum(&refs) }
    }

    /// The multiplicity of species `z`'s isolated-atom reference on `prov`'s surface (module doc,
    /// "The multiplicity rule"): the scan's choice when it is cached, else the only one the scan can
    /// solve, else `None`.
    fn reference_multiplicity(&self, z: u8, prov: &Provenance) -> Option<u32> {
        if let Some(m) = self.ref_cache.get(&atom_key(prov, z)).and_then(|r| r.multiplicity) {
            return Some(m);
        }
        match sim::allowed_multiplicities(z as u32).as_slice() {
            [m] => Some(*m),
            _ => None,
        }
    }

    /// The step answered by the lone-atom rule: `Source::Reference`, E = `energy`, F = 0, no
    /// display, no teacher call, nothing kept, audited or counted as a solve; booked like a model
    /// answer (module doc above, "The energy's surface").
    pub(super) fn reference_answer(&mut self, energy: f64, prov: Provenance, pos: Vec<[f64; 3]>, z: Vec<u8>, t0: Instant) -> ForceResult {
        let forces = vec![[0.0; 3]; pos.len()];
        let mut stats = EvalStats::default();
        stats.source = Source::Reference;
        if let Some((tag, summaries)) = &self.last_pseudised {
            if *tag == prov.ionic && prov.ionic != "none" {
                stats.pseudised = summaries.clone();
            }
        }
        // from a teacher-surface report the transition is booked along the forces; on the model's
        // surface a refit's pending jump (its value at these positions is Σ E_ref too)
        stats.refit_jump = self.transition_jump(energy, true, &forces, &pos);
        self.take_offset_jump(&mut stats);
        stats.energy.total = energy;
        stats.solve_ms = t0.elapsed().as_millis() as u64;
        stats.notes.push(reference_note(self.cutoff()));
        stats.notes.append(&mut self.pending_notes);
        self.ledger.counts.reference += 1;
        self.streak = 0;
        self.streak_box = 0;
        self.streak_box_face = 0.0;
        self.class_streak = 0;
        self.last = Some(LastEval { pos, z, energy, on_model: true, provenance: prov, flip_back: false, surface_forces: forces.clone(), reference: true });
        ForceResult { energy, forces, stats, display: None, orbitals: None }
    }
}
