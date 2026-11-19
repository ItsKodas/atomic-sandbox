//! The provider's wiring of the stray-atom checks and the sample hold-out (plan Task 3; the parent module doc, "The
//! checks on the auditor and the sample hold-out"): the auditor's fragment and atom jobs, their answers, the verdict's
//! booking, the releases and the book's lifetime. The pure logic is the parent's.
use super::super::quarantine::sample_key;
use super::*;
use super::super::lone::reference_settings_key;
use surrogate::history::PointKind;

/// A solve as a check's part: its energy (`energy`, raw — never the collection's canonical one), forces and levels,
/// and whether it is clean (no warning, trainable, its face density within the box rule at `h`).
pub(in crate::surrogate) fn part_answer(r: &ForceResult, energy: f64, h: f64) -> PartAnswer {
    PartAnswer {
        energy,
        forces: r.forces.clone(),
        frontier: r.stats.frontier,
        partial: r.stats.partial.clone(),
        clean: clean_solve(!r.stats.warnings.is_empty(), r.stats.untrainable(), r.stats.face_density, h),
    }
}

/// The stray atoms of `s` at `r_cut` (Bohr) out of its force rows (`Sample::force_mask`; Task 4 fix round 1, I2,
/// controller ruling 8): a stray atom's force rows are identically zero (B_s = 0 and it is in no other atom's
/// descriptor) and its label is the teacher's ≈ 0, so fitted they add 3 rows of no information each to the force
/// count, Σ F² and the residual degrees of freedom — biasing the fitted force noise σ_f down, and rules 6 and 7 looser
/// with it. Masked out they enter no count, no envelope and no active set (the mask's contract); the energy row is
/// kept (`use_energy` unchanged) and an existing mask (a capped segment's) is narrowed, never widened. A sample with
/// no stray atom is left as it is. Applied where samples enter the store (`push_sample_coded`, `hold_out_stray`, a
/// collection's mirror): in memory only — a loaded record is never rewritten, nor written with it
/// ([`unmasked_record`]).
///
/// A whole sample's mask (`use_energy`: no other source of one) is **derived afresh** at `r_cut` (Task 5 fix round 1,
/// M7): an atom no longer stray at this cutoff gets its force rows back, so a mask never outlives the cutoff it was a
/// fact at. A reach change within a session must call this over the store before its refit (none exists yet: every
/// basis has r_cut 6). A capped segment's own mask is only narrowed.
pub(in crate::surrogate) fn mask_stray_atoms(s: &mut Sample, r_cut: f64) {
    let n = s.pos.len();
    if s.use_energy {
        s.force_mask = None;
    }
    let Some(sp) = split(&s.pos, r_cut) else { return };
    let mut mask = s.force_mask.take().filter(|m| m.len() == n).unwrap_or_else(|| vec![true; n]);
    for &i in &sp.stray {
        mask[i] = false;
    }
    s.force_mask = Some(mask);
}

/// The record of `s` as a collection or a model file holds it (plan Task 5, B; module doc, "Answering inside a
/// region"): [`mask_stray_atoms`] is in memory only, so the stray atoms' entries of its force mask go back to true and
/// a whole sample's mask (`use_energy`: no other source of a mask) is dropped once nothing is masked. A capped
/// segment's own mask (`use_energy` false) keeps its other entries: its masked atoms are caps and edge atoms, which have
/// neighbours, so none of them is stray and nothing of it is lost. Every load masks again ([`mask_stray_atoms`] at
/// `push_sample_coded`, `hold_out_stray` and a collection's mirror), at the r_cut of that session's model.
pub(in crate::surrogate) fn unmasked_record(s: &Sample, r_cut: f64) -> Sample {
    let mut out = s.clone();
    let Some(sp) = split(&s.pos, r_cut) else { return out };
    if let Some(m) = out.force_mask.as_mut() {
        for &i in &sp.stray {
            if let Some(x) = m.get_mut(i) {
                *x = true;
            }
        }
    }
    if out.use_energy && out.force_mask.as_ref().is_some_and(|m| m.iter().all(|&b| b)) {
        out.force_mask = None;
    }
    out
}

/// What an attribution check carries (parent module doc, "Audits of a cleared answer"): the failed audit's case as the
/// audit recorded it (the scene's geometry and canonical labels: what counts against the model when the check cannot
/// attribute), the box its labels are canonical in, the audit's gap and words, the surface it was judged on (a check
/// of another surface is discarded, Task 5 fix round 1 I2) and the refit count then (M5).
#[derive(Clone, Debug)]
pub(in crate::surrogate) struct Attribution {
    pub case: FailedAudit,
    pub grid_n: usize,
    pub gap: f64,
    pub why: String,
    pub surface: String,
    pub fits: u64,
}

/// A failed audit's attribution check waiting behind another (Task 5 fix round 1, M4): what `enqueue_stray_check` takes.
#[derive(Clone, Debug)]
pub(in crate::surrogate) struct AttributionRequest {
    pub scene: Scene,
    pub ans: PartAnswer,
    pub split: Split,
    pub sig: Signature,
    pub attribution: Attribution,
}

/// How many failed stray-atom audits may wait for their attribution check behind the one in flight; beyond, one counts
/// against the model at once (the conservative side). Each costs a fragment solve and one solve per stray atom.
pub(in crate::surrogate) const STRAY_ATTRIBUTION_QUEUE: usize = 4;

/// What [`SurrogateKs::stray_audit_failed`] did with a failed audit.
#[derive(Clone, Debug, PartialEq)]
pub(in crate::surrogate) enum StrayAudit {
    /// The scene has no stray atom: an ordinary failed audit.
    NotStray,
    /// The signature refused, its samples held, an attribution check queued: the failure does not count against the
    /// model unless the check says so. The note.
    Attributing(String),
    /// The signature refused and its samples held, but no attribution check can run: the failure counts against the
    /// model now, the caller records it as any failed audit (the conservative side). The note.
    CountsNow(String),
}

/// A check's solve is clean ([`PartAnswer::clean`]): no warning, trainable, its face density (if reported) within the
/// box rule at grid spacing `h` (a NaN density is not within it).
fn clean_solve(warned: bool, unfit: Option<&str>, face: Option<f64>, h: f64) -> bool {
    !warned && unfit.is_none() && face.is_none_or(|rho| rho <= box_affected_face_density(h))
}

/// The composition check in flight on the auditor (module doc, "The checks on the auditor"): the signature and split,
/// the scene's own answer, the parts' answers as they arrive (`atoms` in [`Split::stray`]'s order), the scene-edit
/// generation it was queued in, the store index of the scene's kept sample (`sample`, with its `sample_key`: a purge
/// may move indices), the scene's positions and species (the bonded descriptors are taken when it is judged), the
/// id its jobs carry and the parts' solve time; `attribution` for a failed audit's check (never booked; parent module
/// doc, "Audits of a cleared answer").
#[derive(Clone, Debug)]
pub(in crate::surrogate) struct StrayCheck {
    pub sig: Signature,
    pub split: Split,
    pub scene: PartAnswer,
    pub frag: Option<PartAnswer>,
    pub atoms: Vec<Option<PartAnswer>>,
    pub scene_gen: u64,
    pub sample: Option<usize>,
    pub sample_key: Option<u64>,
    pub pos: Vec<[f64; 3]>,
    pub z: Vec<u8>,
    pub id: u64,
    pub ms: f64,
    pub attribution: Option<Attribution>,
    /// The scene's box (grid points per side): the stray atoms' box references are of it (ε_pos, ruling 9a's log).
    pub grid_n: usize,
}

/// "H2 + H", "H2 + 2 H", "H2O + O": the bonded part's formula, then the stray atoms as separate atoms — a count and
/// the element per species, never a formula (two stray H are not an H₂: Task 6 review I1) — the Activity's, the notes'
/// and the Model tab's label of a signature. The app writes the bonded part's counts as subscripts.
pub(in crate::surrogate) fn signature_label(sig: &Signature) -> String {
    let mut stray: Vec<(u8, usize)> = Vec::new();
    for &z in &sig.stray_z {
        match stray.iter_mut().find(|(zi, _)| *zi == z) {
            Some((_, k)) => *k += 1,
            None => stray.push((z, 1)),
        }
    }
    let atoms: Vec<String> = stray.iter().map(|&(z, k)| if k == 1 { symbol(z).to_string() } else { format!("{k} {}", symbol(z)) }).collect();
    format!("{} + {}", super::shared::formula(&sig.bonded_z), atoms.join(" + "))
}

impl SurrogateKs {
    /// The split of `pos` at the model's cutoff, its signature on `prov` and the floor's refusal (module doc, "The
    /// static floor"); `None` when the geometry has no stray atom.
    pub(in crate::surrogate) fn stray_split(&self, pos: &[[f64; 3]], z: &[u8], prov: &Provenance) -> Option<(Split, Signature, Option<String>)> {
        let split = split(pos, self.cutoff())?;
        let sig = Signature::new(z, &split, prov.charge, prov);
        let floor = self.stray_floor(pos, z, prov, &split);
        Some((split, sig, floor))
    }

    /// [`floor_refusal`] on the provider's sources: `m_ref` is `lone`'s (the scan's record, or the single allowed M,
    /// with the record's partial-level flag), `fenced` the shell-check fences, the faces those of `prov`'s cell.
    fn stray_floor(&self, pos: &[[f64; 3]], z: &[u8], prov: &Provenance, split: &Split) -> Option<String> {
        let scene = Scene {
            nuclei: pos.iter().zip(z).map(|(&p, &zi)| sim::Nucleus::new(zi, p)).collect(),
            cell: sim::Cell { n: prov.grid_n, h: prov.h },
            charge: prov.charge,
            multiplicity: prov.multiplicity,
            sigma: Some(prov.sigma),
            external_field: [0.0; 3],
            ionic: sim::IonicPolicy::None,
            train_mask: None,
        };
        let m_ref = |zs: u8| self.reference_multiplicity(zs, prov).map(|m| (m, self.ref_cache.get(&atom_key(prov, zs)).and_then(|r| r.fractional)));
        let fenced = |a: u8, b: u8| {
            let key = (a.min(b), a.max(b));
            self.audit_fences.iter().any(|c| c.origin == AuditOrigin::ShellCheck && c.pair.map(|(p, q)| (p.min(q), p.max(q))) == Some(key))
        };
        let half = scene.cell.half_extent();
        let faces = |r: [f64; 3]| half - r.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        floor_refusal(&scene, split, &m_ref, &fenced, &faces)
    }

    /// The bonded part's descriptors and species indices under the model's basis (`None` without a model, or for a
    /// species it lacks).
    pub(in crate::surrogate) fn bonded_descriptors(&self, pos: &[[f64; 3]], z: &[u8], split: &Split) -> Option<(Vec<Vec<f64>>, Vec<usize>)> {
        let basis = &self.model.as_ref()?.basis;
        let bp: Vec<[f64; 3]> = split.bonded.iter().map(|&i| pos[i]).collect();
        let bz: Vec<u8> = split.bonded.iter().map(|&i| z[i]).collect();
        Some((describe_values(basis, &bp, &bz).ok()?, basis.species_indices(&bz).ok()?))
    }

    /// The book under the model that stands (controller ruling B): when the model's basis or surface differs from the
    /// one its regions were built under — a refit with another basis, an installed model, a Forget — the book is
    /// cleared and an ordinary check in flight dropped; no held sample is released. True when it cleared.
    ///
    /// A failed audit's attribution check survives a **basis** change (Task 5 fix round 1, I2): its parts are teacher
    /// solves at a fixed geometry, independent of the basis, and it is judged with the model current when they answer,
    /// as an audit is. A **surface** change discards it and its queue without counting (I1: nothing is booked for a
    /// model that is gone); a Forget or an install discards it in `reset_audits`.
    pub(in crate::surrogate) fn stray_book_current(&mut self) -> bool {
        let same = match (self.model.as_ref(), self.stray_key.as_ref()) {
            (Some(m), Some((b, s))) => m.basis == *b && reference_settings_key(&m.provenance) == *s,
            (None, None) => true,
            _ => false,
        };
        if same {
            return false;
        }
        self.stray_book.clear();
        let surface = self.model.as_ref().map(|m| reference_settings_key(&m.provenance));
        let attributing = self.stray_pending.as_ref().and_then(|p| p.attribution.as_ref()).is_some_and(|a| Some(&a.surface) == surface.as_ref());
        if !attributing {
            self.discard_stray_check();
        }
        self.stray_key = self.model.as_ref().map(|m| (m.basis.clone(), reference_settings_key(&m.provenance)));
        true
    }

    /// The session's stray-atom signatures for the Model tab (`sim::TrainableStatus::stray`; spec §5): (label, state
    /// words), sorted by label. Empty when the book is not the standing model's (it is cleared lazily, at the next
    /// step: [`stray_book_current`](Self::stray_book_current)), so a stale region is never shown as cleared.
    pub(in crate::surrogate) fn stray_status(&self) -> Vec<(String, String)> {
        let current = match (self.model.as_ref(), self.stray_key.as_ref()) {
            (Some(m), Some((b, s))) => m.basis == *b && reference_settings_key(&m.provenance) == *s,
            _ => false,
        };
        if !current {
            return Vec::new();
        }
        let mut v: Vec<(String, String)> = self.stray_book.entries().map(|(sig, st)| (signature_label(sig), super::state_words(st))).collect();
        v.sort();
        v
    }

    /// Drops the check in flight and its queued parts (an answer already on its way is told apart by its job id). An
    /// attribution check's failed audit then counts against the model (the conservative side; parent module doc,
    /// "Audits of a cleared answer"): its check did not complete (a part solve failed, an answer matched no stray atom).
    pub(in crate::surrogate) fn drop_stray_check(&mut self) {
        let pending = self.stray_pending.take();
        self.side_queue.retain(|j| !matches!(j.2, JobKind::Fragment | JobKind::AtomAt(_)));
        if let Some(a) = pending.and_then(|p| p.attribution) {
            self.count_stray_audit(a, "its check did not complete");
            self.next_attribution();
        }
    }

    /// Drops the check in flight, its queued parts and every waiting attribution **without counting anything** (Task 5
    /// fix round 1, I1): a Forget, an installed model (`reset_audits`: "nothing is booked") or a surface change — the
    /// model the failure was about is gone.
    pub(in crate::surrogate) fn discard_stray_check(&mut self) {
        self.stray_pending = None;
        self.stray_attributions.clear();
        self.side_queue.retain(|j| !matches!(j.2, JobKind::Fragment | JobKind::AtomAt(_)));
    }

    /// The atom count of `pos`'s per-atom energy quantities — every audit tier, and the fit's own row
    /// (`surrogate::energy_row_atoms`, controller ruling 9a): the atoms with a neighbour within the model's cutoff,
    /// n_bonded at a stray-atom scene, exactly n otherwise (parent module doc, "Audits of a cleared answer"); never 0.
    pub(in crate::surrogate) fn audit_n(&self, pos: &[[f64; 3]]) -> usize {
        surrogate::energy_row_atoms_at(self.cutoff(), pos).max(1)
    }

    /// "Ha/atom", or "Ha per bonded atom" where `pos`'s per-atom tiers divide by n_bonded (Task 5 fix round 1, M8): a
    /// note's unit, byte-identical for every scene without a stray atom.
    pub(in crate::surrogate) fn per_atom_unit(&self, pos: &[[f64; 3]]) -> &'static str {
        if self.audit_n(pos) == pos.len().max(1) {
            "Ha/atom"
        } else {
            "Ha per bonded atom"
        }
    }

    /// A failed strict audit at `case`'s geometry (spec §4.5, review I6; parent module doc, "Audits of a cleared
    /// answer"): when the scene has stray atoms, its signature is refused for the session (the audit's `why` as the
    /// reason), its samples are held out ([`Q_STRAY`]; the refit armed) and an attribution check of the bonded part
    /// alone and each stray atom alone is queued at that geometry with the audit's own answer (`raw_energy`, the
    /// teacher's before the canonical surface, and `forces`) as its scene — behind another in flight (a short queue,
    /// M4). `case` is the failed audit as the caller would record it (canonical labels), `gap` its teacher − model.
    /// The caller records `case` (and its history point) as any failed audit unless this says
    /// [`StrayAudit::Attributing`]: then the history point is the check's (ruling 9b).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::surrogate) fn stray_audit_failed(&mut self, scene: &Scene, raw_energy: f64, forces: &[[f64; 3]], case: &FailedAudit, gap: f64, why: String) -> StrayAudit {
        let prov = self.scene_provenance(scene);
        let Some((split, sig, _)) = self.stray_split(&case.pos, &case.z, &prov) else { return StrayAudit::NotStray };
        self.stray_book_current();
        self.stray_book.refuse(&sig, format!("audit: {why}"));
        let held = self.hold_stray_signature(&sig);
        if held > 0 {
            self.arm_stray_refit();
        }
        let label = signature_label(&sig);
        let base = format!(
            "surrogate: stray atoms ({label}) — the audit of the model's answer failed ({why}): DFT answers this combination for the session and its samples stay out of the fit{}",
            if held > 0 { format!(" ({held} taken back out)") } else { String::new() }
        );
        let ans = PartAnswer { energy: raw_energy, forces: forces.to_vec(), frontier: [Frontier::default(); 2], partial: Vec::new(), clean: true };
        let surface = self.model.as_ref().map_or_else(String::new, |m| reference_settings_key(&m.provenance));
        let a = Attribution { case: case.clone(), grid_n: prov.grid_n, gap, why, surface, fits: self.fits };
        let busy = self.stray_pending.as_ref().is_some_and(|p| p.attribution.is_some());
        if self.auditor.is_none() || (busy && self.stray_attributions.len() >= STRAY_ATTRIBUTION_QUEUE) {
            let reason = if busy { "the attribution queue is full" } else { "no background teacher" };
            return StrayAudit::CountsNow(format!("{base}; no check can attribute the failure ({reason}), so it counts against the model"));
        }
        let words = format!("{base}; a check of the bonded part alone and each stray atom alone attributes the failure — the model's trust is unchanged unless the model is wrong on the bonded part");
        if busy {
            self.stray_attributions.push_back(AttributionRequest { scene: scene.clone(), ans, split, sig, attribution: a });
            return StrayAudit::Attributing(format!("{words} (queued behind {} other)", self.stray_attributions.len()));
        }
        self.enqueue_stray_check(scene, ans, split, sig, None, Some(a));
        StrayAudit::Attributing(words)
    }

    /// The next waiting attribution check, when none is in flight (M4).
    fn next_attribution(&mut self) {
        if self.stray_pending.as_ref().is_some_and(|p| p.attribution.is_some()) {
            return;
        }
        if let Some(r) = self.stray_attributions.pop_front() {
            self.enqueue_stray_check(&r.scene, r.ans, r.split, r.sig, None, Some(r.attribution));
        }
    }

    /// Every sample of `sig` in the fit or on its way there ([`Q_NONE`], `Q_REF_WAIT`, `Q_DRAFT`) back to [`Q_STRAY`]:
    /// a refused signature's samples are held out for good (spec §4.3, C1); how many. The caller arms the refit.
    fn hold_stray_signature(&mut self, sig: &Signature) -> usize {
        let Some(store) = self.store.as_ref() else { return 0 };
        let r_cut = self.cutoff();
        let mut hold = Vec::new();
        for (i, s) in store.samples.iter().enumerate() {
            if !matches!(self.quarantine.get(i).copied().unwrap_or(Q_NONE), Q_NONE | Q_REF_WAIT | Q_DRAFT) {
                continue;
            }
            let Some(sp) = split(&s.pos, r_cut) else { continue };
            let prov = self.sample_provenance(i);
            if Signature::new(&s.z, &sp, prov.charge, &prov) == *sig {
                hold.push(i);
            }
        }
        if let Some(&last) = hold.last() {
            if self.quarantine.len() <= last {
                self.quarantine.resize(last + 1, Q_NONE);
            }
        }
        for &i in &hold {
            self.quarantine[i] = Q_STRAY;
        }
        hold.len()
    }

    /// A failed audit no check attributed (`why`): it counts against the model as any failed audit (rule 8, with the
    /// scene's own labels): untrusted until re-earned, the flip booked at the next evaluation when the model's surface
    /// was the last, and its history point recorded now at the scene's geometry with the model current (deferred at
    /// the audit, ruling 9b; the conservative side, as a `CountsNow` audit records it).
    fn count_stray_audit(&mut self, a: Attribution, why: &str) {
        let case = a.case;
        let pred = self.model.as_ref().and_then(|m| m.predict(&case.pos, &case.z, true).ok());
        match pred {
            Some(p) => {
                let stds = match (p.energy_std, p.force_std.as_ref(), p.force_std_epistemic.as_ref()) {
                    (Some(e), Some(f), Some(fe)) => Some((e, f.clone(), fe.clone())),
                    _ => None,
                };
                self.record_history_point(PointKind::Audit, &case.pos, &case.z, stds.as_ref().map(|(e, f, fe)| (*e, &f[..], &fe[..])), p.energy, &p.forces, case.energy, &case.forces, false, false, false);
            }
            None => self.audit_history.unrecorded.no_variance += 1,
        }
        self.failed_audits.push(case);
        self.sync_audit_failed();
        self.audit_flip = self.last.as_ref().is_some_and(|l| l.on_model);
        self.pending_notes.push(format!(
            "surrogate: a failed audit at a scene with stray atoms (teacher − model {:+.2e} Ha; {}) is not attributed — {why} — so it counts against the model: untrusted until a refit predicts that geometry without the audited sample",
            a.gap, a.why
        ));
    }

    /// The model may answer this scene as "bonded part + free atoms" (parent module doc, "Answering inside a region";
    /// spec §4.4): it splits at the model's cutoff, every stray species is in the model's basis (rule 2, review M2), its
    /// signature is `Cleared`, the floor still passes at these positions (a stray atom drifting within
    /// [`STRAY_FACE_MARGIN`] of a face, a forced multiplicity, a shell fence since — the signature does not carry them)
    /// and the region covers the bonded part's descriptors. The split, the region's worst energy residual and smallest
    /// frontier margin (Ha) and its passing checks so far (the step note's numbers; fix round 1 M1); `None` otherwise.
    /// Read-only: a book built under another basis or surface than the model's
    /// covers nothing (it is cleared at the next `stray_book_current`, controller ruling B).
    pub(in crate::surrogate) fn stray_cleared(&self, pos: &[[f64; 3]], z: &[u8], prov: &Provenance) -> Option<(Split, f64, f64, usize)> {
        self.stray_cleared_with(pos, z, prov, None)
    }

    /// [`stray_cleared`](Self::stray_cleared) with the scene's descriptors under the model's basis when the caller
    /// has them (`evaluate`, which shares them with the envelope; fix round 1 M6): a bonded atom's descriptor in the
    /// scene is the bonded part's own (a stray atom is no atom's neighbour, so no term and no order changes), so the
    /// region is read from them without describing the bonded part again. Cheapest tests first: a scene that does not
    /// split or whose signature is not cleared costs a split and a lookup.
    pub(in crate::surrogate) fn stray_cleared_with(&self, pos: &[[f64; 3]], z: &[u8], prov: &Provenance, desc: Option<&[Vec<f64>]>) -> Option<(Split, f64, f64, usize)> {
        let model = self.model.as_ref()?;
        if pos.len() != z.len() {
            return None;
        }
        // the model's own cutoff (fix round 1 M2: the one its descriptors and rule 5a use)
        let split = split(pos, model.basis.r_cut)?;
        if split.stray.iter().any(|&i| model.basis.species_index(z[i]).is_none()) {
            return None;
        }
        let sig = Signature::new(z, &split, prov.charge, prov);
        let Some(StrayState::Cleared { region, residual, margin, passes }) = self.stray_book.state(&sig) else { return None };
        let (basis, key) = self.stray_key.as_ref()?;
        if model.basis != *basis || reference_settings_key(&model.provenance) != *key {
            return None;
        }
        if self.stray_floor(pos, z, prov, &split).is_some() {
            return None;
        }
        let bz: Vec<u8> = split.bonded.iter().map(|&i| z[i]).collect();
        let species = model.basis.species_indices(&bz).ok()?;
        let inside = match desc.filter(|d| d.len() == pos.len()) {
            Some(d) => region.contains(&split.bonded.iter().map(|&i| &d[i]).collect::<Vec<_>>(), &species),
            None => self.bonded_descriptors(pos, z, &split).is_some_and(|(d, _)| region.contains(&d, &species)),
        };
        inside.then_some((split, *residual, *margin, *passes))
    }

    /// Whether a cleared region of `sig` covers the bonded geometry of `pos` (module doc, "The hold-out").
    pub(in crate::surrogate) fn stray_covered(&mut self, pos: &[[f64; 3]], z: &[u8], split: &Split, sig: &Signature) -> bool {
        self.stray_book_current();
        let Some(StrayState::Cleared { region, .. }) = self.stray_book.state(sig) else { return false };
        self.bonded_descriptors(pos, z, split).is_some_and(|(d, s)| region.contains(&d, &s))
    }

    /// Queues the composition check of `scene` (module doc, "The checks on the auditor"): `ans` is the scene's own
    /// fallback answer, `split` and `sig` its split and signature (the floor passed: the caller's), `kept` the store
    /// index of its kept sample. True when queued; false without a background auditor (controller ruling C), while
    /// another check of this scene is in flight, for a refused signature, inside a cleared region, at a geometry that
    /// would not count, or without a model to measure the geometry with.
    pub(in crate::surrogate) fn queue_stray_check(&mut self, scene: &Scene, ans: PartAnswer, split: Split, sig: Signature, kept: Option<usize>) -> bool {
        if self.auditor.is_none() {
            return false;
        }
        self.stray_book_current();
        // at most one in flight, and a newer fallback does not replace it — unless it is of a scene edited since (its
        // answers are dropped on arrival anyway); a failed audit's attribution check is never replaced
        if self.stray_pending.as_ref().is_some_and(|p| p.scene_gen == self.scene_gen || p.attribution.is_some()) {
            return false;
        }
        let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        let z: Vec<u8> = scene.nuclei.iter().map(|n| n.z).collect();
        let Some((desc, species)) = self.bonded_descriptors(&pos, &z, &split) else { return false };
        let Some(units) = self.envelope.as_ref() else { return false };
        let thin = THIN_FRACTION * self.policy.max_descriptor_distance;
        match self.stray_book.state(&sig) {
            Some(StrayState::Refused(_)) => return false,
            Some(StrayState::Cleared { region, .. }) if region.contains(&desc, &species) => return false,
            Some(StrayState::Checking { region, .. }) if !region.distinct(&desc, &species, units, thin) => return false,
            _ => {}
        }
        self.enqueue_stray_check(scene, ans, split, sig, kept, None);
        true
    }

    /// Queues the parts of a check of `scene` (the caller's gates passed): the bonded part alone and each stray atom
    /// alone on the auditor's side queue, replacing the check in flight (an ordinary one: the callers never replace an
    /// attribution check). `attribution`: a failed audit's check (parent module doc, "Audits of a cleared answer").
    fn enqueue_stray_check(&mut self, scene: &Scene, ans: PartAnswer, split: Split, sig: Signature, kept: Option<usize>, attribution: Option<Attribution>) {
        self.drop_stray_check();
        self.stray_checks += 1;
        let id = self.stray_checks;
        let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        let z: Vec<u8> = scene.nuclei.iter().map(|n| n.z).collect();
        // the scene's provenance and ionic policy (review M5): the parts are solved on the scene's surface
        let prov = self.scene_provenance(scene);
        let mut frag = scene.clone();
        frag.nuclei = split.bonded.iter().map(|&i| scene.nuclei[i].clone()).collect();
        frag.charge = 0;
        frag.multiplicity = None;
        frag.train_mask = None;
        self.side_queue.push_back((frag, prov.clone(), JobKind::Fragment, id));
        for &s in &split.stray {
            let mut atom = scene.clone();
            atom.nuclei = vec![scene.nuclei[s].clone()];
            atom.charge = 0;
            // the floor passed, so the reference multiplicity is known
            atom.multiplicity = self.reference_multiplicity(scene.nuclei[s].z, &prov);
            atom.train_mask = None;
            self.side_queue.push_back((atom, prov.clone(), JobKind::AtomAt(scene.nuclei[s].z), id));
        }
        let sample_key = kept.and_then(|i| self.store.as_ref().and_then(|st| st.samples.get(i)).map(sample_key));
        let atoms = vec![None; split.stray.len()];
        self.stray_pending =
            Some(StrayCheck { sig, split, scene: ans, frag: None, atoms, scene_gen: self.scene_gen, sample: kept.filter(|_| sample_key.is_some()), sample_key, pos, z, id, ms: 0.0, attribution, grid_n: prov.grid_n });
        self.pump_side_queue();
    }

    /// One part's answer from the auditor (`handle_answer`; module doc, "The checks on the auditor"): filed with the
    /// check in flight; when every part has answered, the check is judged. A part of another check (dropped) is ignored,
    /// and a check of a scene edited since is dropped silently; a failed part solve voids the check (a `Failed` event).
    pub(in crate::surrogate) fn take_stray_answer(&mut self, ans: AuditAnswer) {
        let Some(p) = self.stray_pending.as_mut() else { return };
        if ans.job.id != p.id {
            return;
        }
        // (an attribution check is of a geometry, not of the scene's history: an edit does not drop it)
        if p.scene_gen != self.scene_gen && p.attribution.is_none() {
            return self.drop_stray_check();
        }
        let sig = p.sig.clone();
        let attributing = p.attribution.is_some();
        let solved = match ans.result {
            Ok(s) => s,
            Err(e) => {
                // the solves this check made: the parts answered so far and this failed one (one row, Task 6)
                let solves = 1 + u32::from(p.frag.is_some()) + p.atoms.iter().filter(|a| a.is_some()).count() as u32;
                // and their time together, as the row's solves (review minor 2)
                let solve_ms = p.ms + ans.ms;
                self.drop_stray_check();
                let what = if ans.job.kind == JobKind::Fragment { "the bonded part alone" } else { "a stray atom alone" };
                let then = if attributing { "the failed audit it was attributing counts against the model" } else { "not counted, the next fallback checks again" };
                let note = format!(
                    "surrogate: stray atoms ({}) — the composition check's solve of {what} failed on the background teacher ({e}); {then}",
                    signature_label(&sig)
                );
                let mut ev = TeachEvent::new(TeachOrigin::Audit, TeachRule::IsolatedAtom, TeachOutcome::Failed);
                ev.detail = note.clone();
                ev.n_atoms = ans.job.scene.nuclei.len();
                ev.solve_ms = solve_ms;
                ev.solves = solves;
                self.push_event(ev);
                return self.pending_notes.push(note);
            }
        };
        let clean = clean_solve(solved.warned, solved.unfit, solved.face, ans.job.prov.h);
        let part = PartAnswer { energy: solved.energy, forces: solved.forces, frontier: solved.frontier, partial: solved.partial, clean };
        p.ms += ans.ms;
        match ans.job.kind {
            JobKind::Fragment => p.frag = Some(part),
            JobKind::AtomAt(_) => {
                // the stray atom at this answer's position (two stray atoms of one species are told apart by it)
                let at = ans.job.scene.nuclei.first().map(|n| n.pos);
                match (0..p.split.stray.len()).find(|&k| p.atoms[k].is_none() && Some(p.pos[p.split.stray[k]]) == at) {
                    Some(k) => p.atoms[k] = Some(part),
                    None => {
                        // no stray atom of the check waits at this position: the check cannot complete — dropped, so a
                        // later fallback checks again (review M4: never a wedged check)
                        self.drop_stray_check();
                        return self.pending_notes.push(format!(
                            "surrogate: stray atoms ({}) — a stray-atom answer of the composition check matched none of its stray atoms; the check is dropped, the next fallback checks again",
                            signature_label(&sig)
                        ));
                    }
                }
            }
            _ => return,
        }
        if p.frag.is_none() || p.atoms.iter().any(Option::is_none) {
            return;
        }
        let p = self.stray_pending.take().expect("checked");
        self.judge_stray_check(p);
    }

    /// Judges a complete check, books its verdict and releases what it covers (module doc).
    fn judge_stray_check(&mut self, p: StrayCheck) {
        let StrayCheck { sig, split, scene, frag, atoms, sample, sample_key: key, pos, z, ms, attribution, grid_n, .. } = p;
        let frag = frag.expect("complete");
        let atoms: Vec<PartAnswer> = atoms.into_iter().map(|a| a.expect("complete")).collect();
        let label = signature_label(&sig);
        let n_atoms = pos.len();
        let n_stray = atoms.len() as u32;
        if let Some(a) = attribution {
            return self.attribute_stray_audit(a, &label, &split, &scene, &frag, &atoms, &pos, &z, ms);
        }
        // a model with another basis or surface since the dispatch: the book is cleared and this verdict not booked
        if self.stray_book_current() {
            return self.pending_notes.push(format!("surrogate: stray atoms ({label}) — the composition check's answer arrived after the model changed basis or settings; not booked"));
        }
        let verdict = judge(&self.policy, &scene, &frag, &atoms, &split);
        // the stray atoms' box term, logged at every check (controller ruling 9a: it enters a released sample's label
        // and the attribution's identity; the check itself cannot see it, it cancels in (C))
        let eps = self.stray_box_terms(&split, &atoms, &z, grid_n).map_or(String::new(), |e| format!(" [the stray atoms' box term Σ (E_atom − E_box) = {:+.3} mHa]", e * 1e3));
        let event = |outcome: TeachOutcome, detail: String| {
            let mut ev = TeachEvent::new(TeachOrigin::Audit, TeachRule::IsolatedAtom, outcome);
            ev.detail = detail;
            ev.n_atoms = n_atoms;
            ev.solve_ms = ms;
            // the bonded part alone and each stray atom alone: 1 + n_stray solves in this one row (Task 6)
            ev.solves = 1 + n_stray;
            ev
        };
        let (outcome, detail) = match verdict.clone() {
            CheckVerdict::Void(why) => (TeachOutcome::Discarded, format!("surrogate: stray atoms ({label}) — composition check void ({why}); not counted, the next fallback checks again{eps}")),
            CheckVerdict::Fail(why) => {
                self.stray_book.refuse(&sig, why.clone());
                (TeachOutcome::StrayRefused, format!("surrogate: stray atoms ({label}) — composition fails: {why}; DFT answers this combination for the session and its samples stay out of the fit{eps}"))
            }
            CheckVerdict::Pass { residual, margin } => {
                let measured = self.bonded_descriptors(&pos, &z, &split);
                let (Some((desc, species)), Some(model), Some(units)) = (measured, self.model.as_ref(), self.envelope.as_ref()) else {
                    let note = format!("surrogate: stray atoms ({label}) — the composition check passed, but the model cannot describe the bonded part (a species it lacks); not counted");
                    let ev = event(TeachOutcome::Discarded, note.clone());
                    self.push_event(ev);
                    return self.pending_notes.push(note);
                };
                let (basis, units) = (model.basis.clone(), units.clone());
                let was_cleared = matches!(self.stray_book.state(&sig), Some(StrayState::Cleared { .. }));
                let thin = THIN_FRACTION * self.policy.max_descriptor_distance;
                self.stray_book.record(sig.clone(), verdict, &desc, &species, &basis, &units, thin);
                // the check's own sample is measured (unless the signature was refused while it was in flight); a cleared
                // signature releases every held sample its region covers
                let mut released = 0;
                let refused = matches!(self.stray_book.state(&sig), Some(StrayState::Refused(_)));
                if let Some(i) = sample.filter(|&i| !refused && key.is_some() && self.store.as_ref().and_then(|st| st.samples.get(i)).map(sample_key) == key) {
                    if self.quarantine.get(i) == Some(&Q_STRAY) {
                        self.quarantine[i] = self.stray_release_code(i);
                        released += 1;
                    }
                }
                let (outcome, state) = match self.stray_book.state(&sig) {
                    Some(StrayState::Checking { passes, .. }) => (TeachOutcome::StrayChecking { n: *passes, of: STRAY_CHECKS }, format!("{passes} of {STRAY_CHECKS} passes at distinct geometries")),
                    Some(StrayState::Cleared { passes, .. }) if was_cleared => (TeachOutcome::StrayCheckingNew, format!("cleared; the region grows ({passes} checks)")),
                    Some(StrayState::Cleared { passes, .. }) => (TeachOutcome::StrayChecking { n: STRAY_CHECKS, of: STRAY_CHECKS }, format!("cleared ({passes} checks)")),
                    Some(StrayState::Refused(why)) => (TeachOutcome::StrayRefused, format!("refused already ({why})")),
                    None => (TeachOutcome::Discarded, "not booked".to_string()),
                };
                if matches!(self.stray_book.state(&sig), Some(StrayState::Cleared { .. })) {
                    released += self.release_stray_samples(&sig);
                }
                if released > 0 {
                    self.arm_stray_refit();
                }
                (
                    outcome,
                    format!(
                        "surrogate: stray atoms ({label}) — composition check passed (|E_scene − E_frag − Σ E_atom| = {:.3} mHa, frontier margin {:.0} mHa): {state}{}{eps}",
                        residual * 1e3,
                        margin * 1e3,
                        if released > 0 { format!("; {released} held sample{} released into the fit", if released == 1 { "" } else { "s" }) } else { String::new() }
                    ),
                )
            }
        };
        let ev = event(outcome, detail.clone());
        self.push_event(ev);
        self.pending_notes.push(detail);
    }

    /// Σ_s ε_pos,s = Σ_s (E_atom(R_s) − E_box(Z_s)) (Ha): each stray atom's solve at its scene position against its
    /// box reference — on the canonical surface, `to_canonical` of the raw atom less the model's E_ref (the canonical
    /// reference; per scene the box's own). The stray atom's confinement and egg-box term: what a released sample's
    /// label carries besides the bonded part's, and a term of an audit's gap (`attribute_stray_audit`). `None`
    /// without a model or for a species it lacks.
    fn stray_box_terms(&self, split: &Split, atoms: &[PartAnswer], z: &[u8], grid_n: usize) -> Option<f64> {
        let m = self.model.as_ref()?;
        let eps: Option<Vec<f64>> = split.stray.iter().zip(atoms).map(|(&i, x)| m.basis.species_index(z[i]).map(|k| self.to_canonical(&[z[i]], grid_n, x.energy) - m.ref_energy[k])).collect();
        Some(compensated_sum(&eps?))
    }

    /// A failed audit's attribution check answered (step 3 of the parent module doc's "Audits of a cleared answer"):
    /// the model's prediction of the bonded part alone (the model current now, as an audit is judged; a basis change
    /// since is fine, I2) against the fragment's solve, by the audit's own criterion — the energy per bonded atom within
    /// `energy_std_per_atom`, the worst atom's force within its bound and the audit's floor (`worst_force_residual`,
    /// `audit_force_floor`).
    ///
    /// The audit's gap is exactly the sum of three terms (canonical surface; review identity):
    ///
    /// ```text
    ///   gap_audit = gap_frag + comp + Σ_s ε_pos,s
    ///   gap_frag  = E_frag − E_model(frag),   comp = E_scene − E_frag − Σ_s E_atom(R_s)   (raw),
    ///   ε_pos,s   = E_atom(R_s) − E_box(Z_s)  (the stray atom's confinement and egg-box term against its box reference)
    /// ```
    ///
    /// (to rounding, when no refit came between the audit and its check). Wrong on the bonded part: the failure is
    /// the model's, recorded at the scene's geometry with the label the model failed to meet (the fragment's canonical
    /// energy plus Σ_s E_ref(Z_s), its forces on the bonded atoms, 0 on the stray ones; never fitted, so `index` is
    /// none; `AuditOrigin::StrayAttributed`), and the audit history takes the **fragment's** point (ruling 9b). Right
    /// there: the composition's or the box term's, and the model's trust is untouched — no history point (an
    /// `unrecorded.composition` count), and after a refit since the audit the next answered steps are audited
    /// (`AUDIT_BURST`, M5, as a re-earned failed audit). A fragment solve that is not clean or a model that cannot
    /// describe the bonded part cannot attribute: the audit counts as recorded (the conservative side). A model gone
    /// (a Forget) or of another surface: discarded, nothing counted (I1).
    #[allow(clippy::too_many_arguments)]
    fn attribute_stray_audit(&mut self, a: Attribution, label: &str, split: &Split, scene: &PartAnswer, frag: &PartAnswer, atoms: &[PartAnswer], pos: &[[f64; 3]], z: &[u8], ms: f64) {
        self.stray_book_current();
        if self.model.as_ref().map(|m| reference_settings_key(&m.provenance)) != Some(a.surface.clone()) {
            self.pending_notes.push(format!("surrogate: stray atoms ({label}) — a failed audit's attribution check answered after the model's surface changed; discarded"));
            return self.next_attribution();
        }
        if !frag.clean {
            self.count_stray_audit(a, "the bonded part's solve is not clean (a warning, untrainable, or box-affected)");
            return self.next_attribution();
        }
        let (fp, fz) = split.bonded_part(pos, z);
        let model = self.model.as_ref();
        let pred = model.and_then(|m| m.predict(&fp, &fz, true).ok());
        let refs: Option<Vec<f64>> = model.and_then(|m| split.stray.iter().map(|&i| m.basis.species_index(z[i]).map(|k| m.ref_energy[k])).collect());
        let (Some(pred), Some(refs)) = (pred, refs) else {
            self.count_stray_audit(a, "the model cannot describe the bonded part");
            return self.next_attribution();
        };
        let e_frag = self.to_canonical(&fz, a.grid_n, frag.energy);
        let n = split.bonded.len().max(1) as f64;
        let tier = self.policy.energy_std_per_atom;
        let gap = e_frag - pred.energy;
        let force_fail = worst_force_residual(&self.policy, &pred.forces, &frag.forces, audit_force_floor(&a.case.ionic)).filter(|(d, b)| !(d <= b));
        let wrong = !(gap.abs() / n <= tier) || force_fail.is_some();
        // the three terms of the audit's gap: the model on the bonded part, the composition (raw energies, as the
        // check's energy test), and each stray atom's box term against its box reference (canonical: the reference
        // the model's E_ref is)
        let mut terms = vec![scene.energy, -frag.energy];
        terms.extend(atoms.iter().map(|x| -x.energy));
        let composition = compensated_sum(&terms);
        let eps_sum = self.stray_box_terms(split, atoms, z, a.grid_n).unwrap_or(f64::NAN);
        let terms_words = format!(
            "of teacher − model {:+.3} mHa: the model on the bonded part {:+.3} mHa, the composition E_scene − E_frag − Σ E_atom {:+.3} mHa, the stray atoms' box term Σ (E_atom − E_box) {:+.3} mHa",
            a.gap * 1e3,
            gap * 1e3,
            composition * 1e3,
            eps_sum * 1e3
        );
        let on_frag = match force_fail {
            Some((d, b)) => format!("its worst bonded atom's force is off by {d:.1e} > {b:.1e} Ha/Bohr"),
            None => format!("{:.1e} {} {tier:.1e} Ha per bonded atom", gap.abs() / n, if wrong { ">" } else { "≤" }),
        };
        let detail = if wrong {
            let mut case = a.case;
            case.energy = e_frag + compensated_sum(&refs);
            case.forces = split.scene_values(&frag.forces, pos.len(), [0.0; 3]);
            case.index = None;
            case.origin = AuditOrigin::StrayAttributed;
            // the history's point is the fragment's (ruling 9b): its geometry, the model's prediction of it, the
            // fragment's teacher answer — n_atoms n_bonded, the (n_bonded s_E)² floor, err_energy = gap_frag
            let stds = match (pred.energy_std, pred.force_std.as_ref(), pred.force_std_epistemic.as_ref()) {
                (Some(e), Some(f), Some(fe)) => Some((e, f.clone(), fe.clone())),
                _ => None,
            };
            self.record_history_point(PointKind::Audit, &fp, &fz, stds.as_ref().map(|(e, f, fe)| (*e, &f[..], &fe[..])), pred.energy, &pred.forces, e_frag, &frag.forces, false, false, false);
            self.failed_audits.push(case);
            self.sync_audit_failed();
            self.audit_flip = self.last.as_ref().is_some_and(|l| l.on_model);
            format!("surrogate: stray atoms ({label}) — the failed audit is the model's: on the bonded part alone {on_frag} ({terms_words}) — untrusted until a refit predicts that geometry")
        } else {
            // not the model's: no history point (a composition count), and a refit since the audit is audited at once
            self.audit_history.unrecorded.composition += 1;
            if self.fits != a.fits {
                self.audit_burst = AUDIT_BURST;
            }
            let cause = if eps_sum.abs() > composition.abs() { "the stray atoms' place in the box" } else { "the composition" };
            format!(
                "surrogate: stray atoms ({label}) — the failed audit is not the model's (mostly {cause}): on the bonded part alone the model is within its tier, {on_frag} ({terms_words}); the model's trust is unchanged, the combination stays with DFT"
            )
        };
        let mut ev = TeachEvent::new(TeachOrigin::Audit, TeachRule::Audit, TeachOutcome::StrayAuditAttributed { model: wrong });
        ev.detail = detail.clone();
        ev.gap = Some(gap);
        ev.n_atoms = pos.len();
        ev.solve_ms = ms;
        ev.solves = 1 + atoms.len() as u32;
        self.push_event(ev);
        self.pending_notes.push(detail);
        self.next_attribution();
    }

    /// The store samples of `sig` held out ([`Q_STRAY`]) whose bonded geometry its cleared region covers go back to
    /// the fit (`stray_release_code`: [`Q_NONE`], `Q_REF_WAIT` while its box's reference is missing, a draft guess's
    /// audit to `Q_DRAFT`); how many. Nothing
    /// unless `sig` is cleared. The caller arms the refit ([`arm_stray_refit`](Self::arm_stray_refit)).
    pub(in crate::surrogate) fn release_stray_samples(&mut self, sig: &Signature) -> usize {
        self.stray_book_current();
        let Some(StrayState::Cleared { region, .. }) = self.stray_book.state(sig) else { return 0 };
        let region = region.clone();
        let Some(store) = self.store.as_ref() else { return 0 };
        let r_cut = self.cutoff();
        let mut release = Vec::new();
        for (i, s) in store.samples.iter().enumerate() {
            if self.quarantine.get(i) != Some(&Q_STRAY) {
                continue;
            }
            let Some(sp) = split(&s.pos, r_cut) else { continue };
            let prov = self.sample_provenance(i);
            if Signature::new(&s.z, &sp, prov.charge, &prov) != *sig {
                continue;
            }
            if self.bonded_descriptors(&s.pos, &s.z, &sp).is_some_and(|(d, k)| region.contains(&d, &k)) {
                release.push(i);
            }
        }
        for &i in &release {
            self.quarantine[i] = self.stray_release_code(i);
        }
        release.len()
    }

    /// The code a released stray-atom sample returns to: a draft guess's audit (its tag, `SampleTag::DraftAudit`)
    /// to the draft hold-out `Q_DRAFT` — strict mode promotes it there (`judge_held_out`) — any other to
    /// `released_code` (`Q_NONE`, or `Q_REF_WAIT` while its box's reference is missing).
    fn stray_release_code(&self, i: usize) -> u8 {
        match self.store.as_ref().and_then(|st| st.samples.get(i)).map(|s| s.tag) {
            Some(SampleTag::DraftAudit) => Q_DRAFT,
            _ => self.released_code(i),
        }
    }

    /// Whether a sample at `pos`/`z` solved at `prov` is held out ([`Q_STRAY`]): it has stray atoms and no cleared
    /// region of its signature covers its bonded geometry (a floor-refused one is never covered).
    pub(in crate::surrogate) fn stray_held(&mut self, pos: &[[f64; 3]], z: &[u8], prov: &Provenance) -> bool {
        match self.stray_split(pos, z, prov) {
            None => false,
            Some((_, _, Some(_))) => true,
            Some((split, sig, None)) => !self.stray_covered(pos, z, &split, &sig),
        }
    }

    /// Samples came back into the fit: the next refit rebuilds the accumulator (they may lie inside its prefix), and an
    /// attached collection is offered the ones that waited outside it.
    pub(in crate::surrogate) fn arm_stray_refit(&mut self) {
        self.species_dirty = true;
        self.pending_refit = true;
        self.sync_collection_status();
    }

    /// The provenance stored sample `i` was solved at: the store's, in collection mode at the sample's own box.
    pub(in crate::surrogate) fn sample_provenance(&self, i: usize) -> Provenance {
        let mut p = self.store.as_ref().map(|s| s.provenance.clone()).expect("a stored sample");
        if let Some(n) = self.shared.as_ref().and_then(|sh| sh.box_n.get(i)) {
            p.grid_n = *n;
        }
        p
    }
}
