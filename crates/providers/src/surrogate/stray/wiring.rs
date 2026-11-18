//! The provider's wiring of the stray-atom checks and the sample hold-out (plan Task 3; the parent module doc, "The
//! checks on the auditor and the sample hold-out"): the auditor's fragment and atom jobs, their answers, the verdict's
//! booking, the releases and the book's lifetime. The pure logic is the parent's.
use super::super::quarantine::sample_key;
use super::*;
use super::super::lone::reference_settings_key;

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

/// A check's solve is clean ([`PartAnswer::clean`]): no warning, trainable, its face density (if reported) within the
/// box rule at grid spacing `h` (a NaN density is not within it).
fn clean_solve(warned: bool, unfit: Option<&str>, face: Option<f64>, h: f64) -> bool {
    !warned && unfit.is_none() && face.is_none_or(|rho| rho <= box_affected_face_density(h))
}

/// The composition check in flight on the auditor (module doc, "The checks on the auditor"): the signature and split,
/// the scene's own answer, the parts' answers as they arrive (`atoms` in [`Split::stray`]'s order), the scene-edit
/// generation it was queued in, the store index of the scene's kept sample (`sample`, with its `sample_key`: a purge
/// may move indices), the scene's positions and species (the bonded descriptors are taken when it is judged), the
/// id its jobs carry and the parts' solve time.
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
}

/// "H2 + H": the bonded part's formula and the stray atoms' (the Activity's and the notes' label of a signature).
pub(in crate::surrogate) fn signature_label(sig: &Signature) -> String {
    format!("{} + {}", super::shared::formula(&sig.bonded_z), super::shared::formula(&sig.stray_z))
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
    /// cleared and the check in flight dropped; no held sample is released. True when it cleared.
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
        self.drop_stray_check();
        self.stray_key = self.model.as_ref().map(|m| (m.basis.clone(), reference_settings_key(&m.provenance)));
        true
    }

    /// Drops the check in flight and its queued parts (an answer already on its way is told apart by its job id).
    pub(in crate::surrogate) fn drop_stray_check(&mut self) {
        self.stray_pending = None;
        self.side_queue.retain(|j| !matches!(j.2, JobKind::Fragment | JobKind::AtomAt(_)));
    }

    /// The model may answer this scene as "bonded part + free atoms" (parent module doc, "Answering inside a region";
    /// spec §4.4): it splits at the model's cutoff, every stray species is in the model's basis (rule 2, review M2), its
    /// signature is `Cleared`, the floor still passes at these positions (a stray atom drifting within
    /// [`STRAY_FACE_MARGIN`] of a face, a forced multiplicity, a shell fence since — the signature does not carry them)
    /// and the region covers the bonded part's descriptors. The split, the region's worst energy residual and smallest
    /// frontier margin (Ha); `None` otherwise. Read-only: a book built under another basis or surface than the model's
    /// covers nothing (it is cleared at the next `stray_book_current`, controller ruling B).
    pub(in crate::surrogate) fn stray_cleared(&self, pos: &[[f64; 3]], z: &[u8], prov: &Provenance) -> Option<(Split, f64, f64)> {
        let model = self.model.as_ref()?;
        if pos.len() != z.len() {
            return None;
        }
        let split = split(pos, self.cutoff())?;
        if split.stray.iter().any(|&i| model.basis.species_index(z[i]).is_none()) {
            return None;
        }
        let (basis, key) = self.stray_key.as_ref()?;
        if model.basis != *basis || reference_settings_key(&model.provenance) != *key {
            return None;
        }
        let sig = Signature::new(z, &split, prov.charge, prov);
        let Some(StrayState::Cleared { region, residual, margin }) = self.stray_book.state(&sig) else { return None };
        if self.stray_floor(pos, z, prov, &split).is_some() {
            return None;
        }
        let (desc, species) = self.bonded_descriptors(pos, z, &split)?;
        region.contains(&desc, &species).then_some((split, *residual, *margin))
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
        // answers are dropped on arrival anyway)
        if self.stray_pending.as_ref().is_some_and(|p| p.scene_gen == self.scene_gen) {
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
        self.drop_stray_check();
        self.stray_checks += 1;
        let id = self.stray_checks;
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
        self.stray_pending = Some(StrayCheck { sig, split, scene: ans, frag: None, atoms, scene_gen: self.scene_gen, sample: kept.filter(|_| sample_key.is_some()), sample_key, pos, z, id, ms: 0.0 });
        self.pump_side_queue();
        true
    }

    /// One part's answer from the auditor (`handle_answer`; module doc, "The checks on the auditor"): filed with the
    /// check in flight; when every part has answered, the check is judged. A part of another check (dropped) is ignored,
    /// and a check of a scene edited since is dropped silently; a failed part solve voids the check (a `Failed` event).
    pub(in crate::surrogate) fn take_stray_answer(&mut self, ans: AuditAnswer) {
        let Some(p) = self.stray_pending.as_mut() else { return };
        if ans.job.id != p.id {
            return;
        }
        if p.scene_gen != self.scene_gen {
            return self.drop_stray_check();
        }
        let sig = p.sig.clone();
        let solved = match ans.result {
            Ok(s) => s,
            Err(e) => {
                self.drop_stray_check();
                let what = if ans.job.kind == JobKind::Fragment { "the bonded part alone" } else { "a stray atom alone" };
                let note = format!(
                    "surrogate: stray atoms ({}) — the composition check's solve of {what} failed on the background teacher ({e}); not counted, the next fallback checks again",
                    signature_label(&sig)
                );
                let mut ev = TeachEvent::new(TeachOrigin::Audit, TeachRule::IsolatedAtom, TeachOutcome::Failed);
                ev.detail = note.clone();
                ev.n_atoms = ans.job.scene.nuclei.len();
                ev.solve_ms = ans.ms;
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
        let StrayCheck { sig, split, scene, frag, atoms, sample, sample_key: key, pos, z, ms, .. } = p;
        let frag = frag.expect("complete");
        let atoms: Vec<PartAnswer> = atoms.into_iter().map(|a| a.expect("complete")).collect();
        let label = signature_label(&sig);
        let n_atoms = pos.len();
        // a model with another basis or surface since the dispatch: the book is cleared and this verdict not booked
        if self.stray_book_current() {
            return self.pending_notes.push(format!("surrogate: stray atoms ({label}) — the composition check's answer arrived after the model changed basis or settings; not booked"));
        }
        let verdict = judge(&self.policy, &scene, &frag, &atoms, &split);
        let event = |outcome: TeachOutcome, detail: String| {
            let mut ev = TeachEvent::new(TeachOrigin::Audit, TeachRule::IsolatedAtom, outcome);
            ev.detail = detail;
            ev.n_atoms = n_atoms;
            ev.solve_ms = ms;
            ev
        };
        let (outcome, detail) = match verdict.clone() {
            CheckVerdict::Void(why) => (TeachOutcome::Discarded, format!("surrogate: stray atoms ({label}) — composition check void ({why}); not counted, the next fallback checks again")),
            CheckVerdict::Fail(why) => {
                self.stray_book.refuse(&sig, why.clone());
                (TeachOutcome::StrayRefused, format!("surrogate: stray atoms ({label}) — composition fails: {why}; DFT answers this combination for the session and its samples stay out of the fit"))
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
                    Some(StrayState::Cleared { .. }) if was_cleared => (TeachOutcome::StrayCheckingNew, "cleared; the region grows".to_string()),
                    Some(StrayState::Cleared { .. }) => (TeachOutcome::StrayChecking { n: STRAY_CHECKS, of: STRAY_CHECKS }, format!("cleared ({STRAY_CHECKS} checks)")),
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
                        "surrogate: stray atoms ({label}) — composition check passed (|E_scene − E_frag − Σ E_atom| = {:.3} mHa, frontier margin {:.0} mHa): {state}{}",
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
