//! The lone-atom rule (user request 2026-11-12; rule 0b of the decision list in the module doc):
//! in strict mode, a scene in which **no atom has a neighbour within r_cut** is answered from the
//! isolated-atom references, whether or not a model exists and whatever the trust verdict says —
//! and, when the references cannot answer it, by the teacher, never by the model (fix round 1,
//! I1). Atomic units (Hartree, Bohr).
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
//! whatever the coefficients are — an untrusted fit, or no fit at all, answers the same. "Lone" is
//! judged at the **model's** r_cut when a model exists (the cutoff its descriptors use; else the
//! cutoff a fit would use: [`SurrogateKs::lone_cutoff`], the one source for the rule and the audit
//! skip). A DFT solve there buys nothing for the model: it is never kept (`no_neighbours`, the keep
//! rule). For the scenes the rule answers (below) what DFT would report differs from Σ E_ref only by
//! the isolated atoms' grid and box artefacts, which no model can or should learn: the egg-box
//! ripple of an off-centre atom and the grid box's **confinement** — its Dirichlet boundary
//! (ψ = 0 at the faces), a finite-box artefact, not the app's container walls (those are `sim`'s,
//! added on top of every provider's forces and booked in `energy_conserved`; they act on a reference
//! step as on any other). Measured for a lone H in the default 12-Bohr box (`diag_lone_h_wobble`;
//! `validate/references.md`): ≈ 1e-6 Ha of ripple; +1.8e-3 Ha (force 2.6e-3 Ha/Bohr toward the
//! centre) 4 Bohr from the nearest face, the box rule's edge; +3.5e-2 Ha 2 Bohr from it. There the
//! reference — the free atom — is the more physical answer of the two. Open-shell atoms add the
//! occupation caveats below.
//!
//! # The energy's surface
//!
//! - **With a model on this surface that covers the species** the answer is the model's own:
//!   Σ model.ref_energy, bit for bit `Model::predict` there (`Lone::Answer { from_model: true }`).
//!   The model's references lag the provider's (`refs`; in collection mode the canonical ones after
//!   `rebase_collection`) until the next refit, and answering from the provider's would put an
//!   unbooked Σ(refs − model.ref_energy) step into `E_conserved` at every model ↔ reference switch
//!   between (review M1). The refit then moves both answers together, booked by the refit's own jump
//!   (`surface_moved`, which also books it when the last report was a reference answer and the fit
//!   is untrusted: the next lone step is still the model's own).
//! - **Otherwise** (no model, another provenance than the model's, a species it lacks) the
//!   provider's references: per scene `refs` (of the store's surface), in collection mode the
//!   **canonical** ones (`shared` module doc, "References per box": a teacher answer in box n is
//!   reported as E − Σ_i (ref(Z_i, n) − ref_canonical(Z_i)), so DFT's lone atoms in box n would be
//!   reported at Σ ref_canonical + artefacts — the reference answer is that without them). A
//!   canonical reference arriving under such a report is booked (`book_reference_arrival`).
//!
//! The answer is booked like a model answer (`on_model`; `Source::Reference` is on `Surface::Model`),
//! with transitions to and from teacher-surface reports booked along the forces (`transition_jump`).
//!
//! # When it declines
//!
//! In strict mode a declined lone scene goes to the teacher with its reason ([`Lone::StandAside`]):
//! after rules 1–4 pass, `evaluate` refuses it as rule 5a does (`TeachRule::IsolatedAtom`, the model
//! consulted for the booking and the history, the answer reported on the teacher's surface and never
//! kept). Before fix round 1 a one-atom scene went on to the model through 5a's one-atom exemption
//! and was answered E_ref for a spin or charge state the scene is not solved at.
//!
//! - **Draft mode** and the **relax verification** (the teacher was asked for): checked in
//!   `evaluate` — draft mode keeps its behaviour (the model answers lone atoms with the same Σ E_ref).
//! - **A charged scene**: the references are neutral atoms; lone ions are not their sum.
//! - **Another surface**: per scene, a store/model provenance that differs from the scene's in grid,
//!   h, σ, tolerance, teacher or ionic surface (charge and multiplicity are the rules below — the
//!   references are neutral atoms at their own spin state, whatever the scene's); no store and no
//!   model (no surface pinned) declines too. In collection mode, a scene the collection refuses.
//! - **A missing reference**: per scene a species without `refs`; in collection mode a species
//!   whose reference of this box is not solved yet (`box_refs_missing`, rule 3's own condition — the
//!   provider has asked the auditor for it) or that has no canonical one.
//! - **The multiplicity** (rule 7; derivation below): the scene's 2S + 1 (its own, or the solver's
//!   automatic lowest one, 1 + (Σ Z mod 2)) must be the **parallel-spin product** of the atoms'
//!   reference states, M* = 1 + Σ_i (m_i − 1). An atom whose reference multiplicity is unknown here
//!   declines.
//! - **More than one species** among two or more lone atoms (below).
//! - **Two or more atoms that may be open-shell** (fix round 1, I4; below).
//! - **A shell-check fence** (fix round 1, I2): a pair of the scene whose species a sampling pass's
//!   shell check measured still interacting near r_cut (`AuditOrigin::ShellCheck`,
//!   `trust::shell_fence_refusal`). The cutoff is known not to hold for that pair, so zero
//!   interaction just beyond it is not "only artefacts".
//!
//! # The multiplicity rule and the occupations
//!
//! The teacher solves collinear LSDA with fixed (N↑, N↓) = ((N + M − 1)/2, (N − M + 1)/2) and
//! per-spin occupations over the **whole** cell (`qm::scf` "Occupations"). For well-separated atoms
//! the Kohn-Sham Hamiltonian is block-diagonal to the overlap of their tails, so the SCF energy is
//! Σ_i E_i(n↑_i, n↓_i) with the per-atom occupations the cell-wide filling hands out. Each reference
//! E_ref(Z) is the atom at its lowest-energy multiplicity m (ruling S9: n↑ − n↓ = m − 1). Σ_i E_ref
//! is therefore reached only by an (N↑, N↓) whose cell-wide filling reproduces every atom's own
//! occupation:
//!
//! - **Parallel product, M* = 1 + Σ (m_i − 1)** — every atom's excess spin up. This composes
//!   exactly for references whose spin channels **end at a gap** (no partly filled shell: H, and
//!   closed shells per spin): the spin-up spectrum is the union of identical atomic spectra whose
//!   occupied and empty parts are separated by a shell gap far above any box splitting, so the
//!   filling gives each copy its own occupation. Two lone H at M = 3: N↑ = 2, both 1s↑ — σg + σu
//!   filled is a unitary rotation of a + b, the same density — E = 2 E_ref.
//! - **Open shells do not compose for k ≥ 2 copies.** Under `DegenerateShells` (the occupations of
//!   every solve with a pseudised species, `providers::ks`) a shared cluster survives only while its
//!   converged spread is below τ = `DEGENERACY_TAU` = 1e-3 Ha — not τ_c = 0.1 Ha, the candidate
//!   window, as the first version of this derivation said. The box splits two copies' levels by
//!   more than that wherever their confinement differs (H: 1.8 mHa 4 Bohr from a face), the cluster
//!   is released (`released_occupations`: aufbau over the groups, the pattern then frozen) and the
//!   lower copy's shell fills first — two O at M = 5 put both minority-p electrons on one atom, an
//!   O⁻ + O⁺ state hundreds of mHa from 2 E_ref (the review's derivation from the occupation rule;
//!   not measured). Under integer aufbau (`Aufbau`, every all-electron solve) an open shell is
//!   integer-occupied and does not show as fractional at all, and the same cross-copy filling can
//!   happen once the box splitting exceeds the self-polarisation gap (O: 11.6–11.8 mHa). So two or
//!   more lone atoms are answered only when their reference is known to end at a gap
//!   ([`closed_spin_channels`]): Z = 1 (one electron: the only multiplicity the scan can solve), or,
//!   on a pseudised surface (`DegenerateShells`), a reference whose winning solve shared no shell
//!   (`AtomReference::fractional == Some(false)`, recorded by the scan next to its multiplicity);
//!   unknown, or any Z ≥ 2 on the all-electron surface, declines. One open-shell atom alone is still
//!   answered: it is its own reference, but near a face its shell can be released into the integer,
//!   symmetry-broken state, a few mHa up to ≈ 12 mHa (O) below the shared one (`qm::scf`) — another
//!   finite-box occupation artefact, and the free atom is again the physical answer.
//! - **Any other M** puts at least one atom into another spin state, or — the measured case — the
//!   teacher's SCF converges to the **spin-restricted** delocalised state (the cold start splits
//!   one blob n↑/N : n↓/N, so ρ↑ ≡ ρ↓ is kept): two H at M = 1 (also what "automatic" picks for an
//!   even electron count) is the restricted H₂ singlet, 4.6e-2 Ha above 2 E_ref at r = 6.4 Bohr
//!   (rule 5a's measurement; its R → ∞ limit, 2 E_LDA(H), is ≈ +66 mHa above 2 E_LSD(H)). An
//!   antiparallel broken-symmetry product would reach Σ E_ref, but nothing seeds it; declined.
//! - **Mixed species** decline even at M*: the occupations are filled cell-wide, so an atom whose
//!   highest occupied spin-up level lies above another atom's partly empty spin-up shell (an alkali
//!   2s↑ near −0.12 Ha against a C 2p↑ near −0.2 Ha in LSDA) loses charge to it at any separation —
//!   LSDA's separated-atom charge-transfer error. Whether that happens depends on eigenvalues the
//!   provider does not have, so the rule does not guess.
//! - **One atom**: M* = m. Under "automatic" that holds only when the reference is the lowest
//!   multiplicity of its parity (H, and every atom the scan solves lowest at 1 or 2).
//!
//! The reference multiplicity m (and the fractional flag) of species Z on this surface is the
//! scan's record when the provider holds it (`ref_cache`, keyed by the atom's solve settings: a
//! sampling pass's scan), else, for the multiplicity, the only one the scan can solve when
//! `allowed_multiplicities(Z)` has a single entry (Z = 1: 2), else unknown → decline. Model files and
//! collections carry the energies only, so a Z ≥ 2 reference loaded from them is unknown.
use super::*;

/// What the lone-atom rule decides at a scene (module doc above).
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Lone {
    /// Some atom has a neighbour within r_cut (or the scene is empty): the rule does not apply.
    NotLone,
    /// Answered from the references: E = Σ_i E_ref(Z_i) on the reported surface, every force 0;
    /// `from_model`: the references are the model's own (`Model::ref_energy`), so E is
    /// `Model::predict`'s there bit for bit.
    Answer { energy: f64, from_model: bool },
    /// Lone atoms the references cannot answer, and why (words): in strict mode the teacher answers.
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

/// Whether the reference of species `z` is known to **end at a gap** in each spin channel (module
/// doc, "The multiplicity rule and the occupations"), so k ≥ 2 copies compose: one electron (the
/// only multiplicity the scan can solve, Z = 1), or, on a pseudised surface (`all_electron` false:
/// the `DegenerateShells` occupations, where an open shell shows as a shared cluster), a scan that
/// recorded no shared shell. Under the all-electron surface's integer aufbau an open shell does not
/// show, so a Z ≥ 2 reference there is not known to compose; nor is an unrecorded one.
pub(super) fn closed_spin_channels(z: u8, all_electron: bool, fractional: Option<bool>) -> bool {
    if sim::allowed_multiplicities(z as u32).len() == 1 {
        return true;
    }
    !all_electron && fractional == Some(false)
}

/// The note of a step the rule answered (`Source::Reference`): the solve log's and the chip's words.
pub(super) fn reference_note(r_cut: f64) -> String {
    format!(
        "surrogate: lone atoms: answered from the atom references — no atom has a neighbour within r_cut = {r_cut:.2} Bohr, so E = Σ E_ref and every force is 0, the model's own answer here whatever its trust (DFT in this box would differ by its grid and box artefacts — measured for a lone H: ~1e-6 Ha of egg-box ripple, and the grid box's confinement, its Dirichlet boundary — not the container walls, which still act — ~2e-3 Ha 4 Bohr from a face and more nearer; nothing is learned)"
    )
}

impl SurrogateKs {
    /// The cutoff "lone" is judged at: the model's own r_cut when a model exists (the cutoff of the
    /// descriptors whose emptiness makes the answer the model's), else the one a fit would use
    /// ([`cutoff`](Self::cutoff)). The one source for rule 0b and the audit skip (review M2).
    pub(super) fn lone_cutoff(&self) -> f64 {
        self.model.as_ref().map_or_else(|| self.cutoff(), |m| m.basis.r_cut)
    }

    /// The lone-atom rule at `scene` solved at `prov` (`scene_provenance`): module doc above. Strict
    /// mode, draft mode and the relax verification are the caller's to check. Read-only.
    pub(super) fn lone_reference(&self, scene: &Scene, prov: &Provenance) -> Lone {
        let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        let r_cut = self.lone_cutoff();
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
        // a measured shell-check fence on a pair of the scene's species (review I2)
        let key = |a: u8, b: u8| if a <= b { (a, b) } else { (b, a) };
        let fenced = self.audit_fences.iter().filter(|c| c.origin == AuditOrigin::ShellCheck).filter_map(|c| c.pair).map(|(a, b)| key(a, b));
        for (a, b) in fenced {
            let present = |x: u8| z.iter().filter(|&&zi| zi == x).count();
            if (a == b && present(a) >= 2) || (a != b && present(a) >= 1 && present(b) >= 1) {
                return Lone::StandAside(format!(
                    "a shell check measured {}–{} still interacting near the cutoff (fenced): zero interaction just beyond it is not the teacher's",
                    symbol(a),
                    symbol(b)
                ));
            }
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
                    match shared::canonical_ref(sh, zi) {
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
        // two or more copies compose only when the reference ends at a gap (review I4)
        if z.len() >= 2 {
            let all_electron = prov.ionic == "none";
            let fractional = self.ref_cache.get(&atom_key(prov, z[0])).and_then(|r| r.fractional);
            if !closed_spin_channels(z[0], all_electron, fractional) {
                let why = match (all_electron, fractional) {
                    (true, _) => "the all-electron surface's integer occupations do not show whether its shell is open".to_string(),
                    (false, Some(true)) => "its reference shares a partly filled shell".to_string(),
                    (false, _) => "whether its reference's shell is open is not recorded here (a sampling pass records it)".to_string(),
                };
                return Lone::StandAside(format!(
                    "{} lone {} atoms: {why} — copies whose levels the box splits by more than 1e-3 Ha can fill one copy's shell first (charge moves between them), so DFT's answer need not be Σ E_ref",
                    z.len(),
                    symbol(z[0])
                ));
            }
        }
        // the model's own references when it covers this scene on this surface (review M1)
        if let Some(m) = self.model.as_ref().filter(|m| self.same_surface(&m.provenance, prov)) {
            if let Ok(s) = m.basis.species_indices(&z) {
                let own: Vec<f64> = s.iter().map(|&si| m.ref_energy[si]).collect();
                return Lone::Answer { energy: compensated_sum(&own), from_model: true };
            }
        }
        Lone::Answer { energy: compensated_sum(&refs), from_model: false }
    }

    /// The multiplicity of species `z`'s isolated-atom reference on `prov`'s surface (module doc,
    /// "The multiplicity rule and the occupations"): the scan's choice when it is cached, else the
    /// only one the scan can solve, else `None`.
    fn reference_multiplicity(&self, z: u8, prov: &Provenance) -> Option<u32> {
        if let Some(m) = self.ref_cache.get(&atom_key(prov, z)).and_then(|r| r.multiplicity) {
            return Some(m);
        }
        match sim::allowed_multiplicities(z as u32).as_slice() {
            [m] => Some(*m),
            _ => None,
        }
    }

    /// The refusal of a lone scene rule 0b declined, in strict mode once rules 1–4 pass (review
    /// I1): rule 5a's code, so the teacher answers on its own surface and the answer is never kept.
    pub(super) fn lone_refusal(why: &str) -> Refusal {
        Refusal::new(format!("surrogate: lone atoms the atom references cannot answer ({why}) — the teacher answers, not the model"), TeachRule::IsolatedAtom, Some(0))
    }

    /// The step answered by the lone-atom rule: `Source::Reference`, E = `energy`, F = 0, no
    /// display, no teacher call, nothing kept, audited or counted as a solve; booked like a model
    /// answer (module doc above, "The energy's surface").
    pub(super) fn reference_answer(&mut self, energy: f64, from_model: bool, prov: Provenance, pos: Vec<[f64; 3]>, z: Vec<u8>, t0: Instant) -> ForceResult {
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
        stats.notes.push(reference_note(self.lone_cutoff()));
        stats.notes.append(&mut self.pending_notes);
        self.ledger.counts.reference += 1;
        self.streak = 0;
        self.streak_box = 0;
        self.streak_box_face = 0.0;
        self.class_streak = 0;
        let reference = if from_model { LastReference::Model } else { LastReference::Provider };
        self.last = Some(LastEval { pos, z, energy, on_model: true, provenance: prov, flip_back: false, surface_forces: forces.clone(), reference: Some(reference) });
        ForceResult { energy, forces, stats, display: None, orbitals: None }
    }
}
