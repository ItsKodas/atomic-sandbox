//! The training store (the `surrogate` module doc, "Species, references, provenance"): samples in
//! (`add_sample*`, the fallback `keep_sample`), the provenance they carry, the reference energies,
//! the `.model` sidecar out and in, and `forget`. Split out of `surrogate_ks.rs` in M6 Task 1; the
//! narrative stays in the parent module doc.

use super::*;

impl SurrogateKs {
    /// Samples held (fallback steps and `add_sample`).
    pub fn samples(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.len())
    }

    /// Teach from an existing sample (a sampling pass, a loaded sidecar). `scene` supplies the
    /// settings the sample was produced at (its provenance); the first sample pins the store's.
    pub fn add_sample(&mut self, s: Sample, scene: &Scene) -> Result<(), StoreError> {
        let prov = self.scene_provenance(scene);
        self.add_sample_with_provenance(s, &prov)
    }

    /// The primitive behind [`add_sample`](Self::add_sample): a sample with the provenance it
    /// was produced at (what a sidecar stores).
    pub fn add_sample_with_provenance(&mut self, s: Sample, prov: &Provenance) -> Result<(), StoreError> {
        self.push_sample(s, prov, f64::NAN)
    }

    /// [`add_sample`](Self::add_sample) with the teacher's reported face density
    /// (`EvalStats::face_density`; `None` when its solver reports none) — what a sampling pass
    /// uses (final review I1), so the box rule applies to pass samples as it does to the
    /// essential live samples (`surrogate` module doc, "Box-affected answers"): kept, stored with their
    /// density, never quarantine candidates, excluded from the fit once the trust floor is met
    /// without them.
    pub fn add_sample_with_face(&mut self, s: Sample, scene: &Scene, face: Option<f64>) -> Result<(), StoreError> {
        let prov = self.scene_provenance(scene);
        self.push_sample(s, &prov, face.unwrap_or(f64::NAN))
    }

    /// Isolated-atom energy of species `z` (Hartree), the model's reference (Z-only: computed
    /// by the teacher, never a table). A change after a fit rebuilds the accumulator at the next
    /// refit. The caller vouches that `e` follows the reference rule (ruling S9: the atom's
    /// lowest-energy multiplicity — what `crate::sampling` computes); the sidecar records it.
    pub fn set_reference_energy(&mut self, z: u8, e: f64) {
        self.legacy_refs.remove(&z);
        self.failed_refs.remove(&z);
        if self.refs.insert(z, e) != Some(e) {
            self.refs_dirty = true;
        }
    }

    /// The provenance a sample produced at this scene's settings by this provider's teacher
    /// carries (what the store and a loaded model are compared with).
    pub fn provenance_for(&self, scene: &Scene) -> Provenance {
        self.scene_provenance(scene)
    }

    /// The isolated-atom energy set for `z`, if any (Hartree).
    pub fn reference_energy(&self, z: u8) -> Option<f64> {
        self.refs.get(&z).copied()
    }

    /// The training samples held (`None` before the first one).
    pub fn store(&self) -> Option<&SampleStore> {
        self.store.as_ref()
    }

    /// The sufficient statistics of the current fit (`None` before the first refit).
    pub fn accumulator(&self) -> Option<&Accumulator> {
        self.accumulator.as_ref()
    }

    /// Drops the model, its samples, statistics and reference energies: the provider is a
    /// pass-through to its teacher again (rule 1 of the policy). If the model was the reported
    /// surface the epoch moves, so the driver re-evaluates at once; that evaluation is on the
    /// teacher surface without a jump and the driver restarts its conserved-energy history
    /// (the honest outcome: the surface the invariant lived on is gone).
    pub fn forget(&mut self) {
        let was_model = self.last.as_ref().is_some_and(|l| l.on_model);
        if self.trusted() || was_model {
            self.epoch += 1;
        }
        self.store = None;
        self.accumulator = None;
        self.accumulated = 0;
        self.refs.clear();
        self.legacy_refs.clear();
        self.refs_dirty = false;
        self.species_dirty = false;
        self.model = None;
        self.model_gen += 1;
        self.fit_members = (0, Vec::new());
        self.curve_cache = None;
        self.report = None;
        self.envelope = None;
        self.null_space = None;
        self.active_set = None;
        self.audit_force = None;
        self.training_residuals = None;
        self.class_loo = Vec::new();
        self.class_worst = Vec::new();
        self.class_blockers = Vec::new();
        self.teach_rounds.clear();
        self.teach_classes.clear();
        self.teach_start = None;
        self.teach_session = super::teach::TeachSession::default();
        self.strict_verdict = None;
        self.pass_start = None;
        self.pass_coverage = None;
        self.audit_failed = None;
        self.failed_audits.clear();
        self.audit_fences.clear();
        self.pending_refit = false;
        self.pending_jump = None;
        self.consults = 0;
        self.last_variance = None;
        self.answered = 0;
        self.quarantine.clear();
        self.face.clear();
        self.fence_votes.clear();
        self.fence_kept.clear();
        self.vote_pending.clear();
        self.degree_cap = None;
        self.since_kept = THIN_EVERY;
        self.desc_cache.clear();
        self.desc_basis = None;
        self.recovery = None;
        self.failed_refs.clear();
        self.last_fit_error = None;
        self.reset_audits();
        // a new model's lifetime starts here (the queued events stay: they happened)
        self.ledger.counts = SolveCounts::default();
    }

    /// Writes the full `.model` sidecar (`surrogate::sidecar`): model, accumulator, last fit
    /// report and every sample held — including any taken since the last refit, which a load
    /// restores as pending — and the lifetime teaching solves (`teacher_solves`, with
    /// `complete: false` while earlier solves are unrecorded: plan review M4). Before any fit
    /// formed (module doc, "No model yet") the samples alone, with the reference energies known
    /// (`surrogate::sidecar` version 6). `InvalidInput` with no samples at all.
    pub fn save_sidecar(&self, path: &std::path::Path) -> std::io::Result<()> {
        let samples = self.store.as_ref().map_or(&[][..], |s| &s.samples[..]);
        let rule = self.legacy_refs.is_empty().then_some(surrogate::sidecar::REFERENCE_RULE);
        // `Q_DRAFT` is not a file code: the `DraftAudit` tag carries it (install restores it)
        let flags: Vec<u8> = (0..samples.len()).map(|i| self.quarantine.get(i).copied().filter(|&q| q != Q_DRAFT).unwrap_or(Q_NONE)).collect();
        // the face densities only when one matters (a file is otherwise unchanged)
        let face: Vec<f64> = if (0..samples.len()).any(|i| self.is_box(i)) {
            (0..samples.len()).map(|i| self.face.get(i).copied().unwrap_or(f64::NAN)).collect()
        } else {
            Vec::new()
        };
        let c = self.ledger.counts;
        let solves = TeacherSolves { pass: c.pass, fallback: c.fallback, audit: c.audit, verification: c.verification, kept: c.kept, complete: !c.unrecorded_before };
        let (Some(model), Some(acc)) = (&self.model, &self.accumulator) else {
            let Some(store) = self.store.as_ref().filter(|s| !s.is_empty()) else {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "nothing to save: no samples yet (start sampling first)"));
            };
            // no fit formed: the samples and what is known of the references (a failed fit must
            // not lose minutes of DFT — the 2026-10-20 PH₃ session)
            let set = surrogate::SampleSet {
                provenance: store.provenance.clone(),
                samples: samples.to_vec(),
                references: self.refs.iter().filter(|(_, e)| e.is_finite()).map(|(&z, &e)| (z, e)).collect(),
                reference_rule: rule.map(str::to_string),
                quarantine: flags,
                face_density: face,
                teacher_solves: Some(solves),
                teacher_solves_note: None,
                created_by: surrogate::sidecar::CREATED_BY.to_string(),
            };
            return surrogate::sidecar::write_samples(path, &set);
        };
        // the failed audits not yet re-earned, and the audit fences (round 1, SHOULD-2)
        let case = |c: &FailedAudit, fenced: bool| surrogate::sidecar::AuditCase {
            pos: c.pos.clone(),
            z: c.z.clone(),
            energy: c.energy,
            forces: c.forces.clone(),
            index: c.index,
            ionic: c.ionic.clone(),
            rejudges: c.rejudges,
            fenced,
            origin: c.origin.word().to_string(),
            pair: c.pair,
        };
        let audits: Vec<surrogate::sidecar::AuditCase> = self.failed_audits.iter().map(|c| case(c, false)).chain(self.audit_fences.iter().map(|c| case(c, true))).collect();
        surrogate::sidecar::write_full_with_audits(path, model, acc, self.report.as_ref(), samples, rule, &flags, &face, Some(&solves), &audits)
    }

    /// Replaces everything learned with a samples-only file's contents (`surrogate::sidecar`
    /// version 6; module doc, "No model yet"): the samples with their flags and face densities,
    /// the reference energies the file knew and its solve record — then a fit when one can form
    /// (every species' reference known and Q′ usable samples). A fit that cannot form leaves the
    /// samples installed and unfitted: `no_model` says why and the Retry re-solves a missing
    /// reference. The caller has checked the provenance against the scene.
    pub fn install_samples(&mut self, set: surrogate::SampleSet) -> Result<(), String> {
        let surrogate::SampleSet { provenance, samples, references, reference_rule, quarantine, face_density, teacher_solves, teacher_solves_note, .. } = set;
        if !quarantine.is_empty() && quarantine.len() != samples.len() {
            return Err(format!("the file has {} quarantine flags for {} samples", quarantine.len(), samples.len()));
        }
        let mut store = SampleStore::new(provenance);
        for (i, smp) in samples.into_iter().enumerate() {
            store.push(smp).map_err(|e| format!("sample {i}: {e}"))?;
        }
        // a draft guess's audit is held out of the fit (`Q_DRAFT`): restored from its tag
        let mut quarantine = quarantine;
        if store.samples.iter().any(|s| s.tag == SampleTag::DraftAudit) {
            quarantine.resize(store.len(), Q_NONE);
            for (i, s) in store.samples.iter().enumerate() {
                if s.tag == SampleTag::DraftAudit && quarantine[i] == Q_NONE {
                    quarantine[i] = Q_DRAFT;
                }
            }
        }
        // a clean slate (what `forget` leaves), the epoch moved if a model was answering
        self.forget();
        let n = store.len();
        self.face = if face_density.is_empty() { vec![f64::NAN; n] } else { face_density };
        self.store = Some(store);
        self.quarantine = quarantine;
        // references of another rule are not fitted on (a new fit would bake them in — the model has
        // no intercept, ruling S5): they count as missing, and the Retry or a pass re-solves them
        let s9 = reference_rule.as_deref() == Some(surrogate::sidecar::REFERENCE_RULE);
        if s9 {
            for &(z, e) in &references {
                self.refs.insert(z, e);
            }
        } else if !references.is_empty() {
            self.user_notes.push("samples file: its reference energies predate ruling S9 (lowest-energy multiplicity) — not used; Retry or a sampling pass re-solves them".into());
        }
        self.refs_dirty = true;
        self.pending_refit = true;
        self.ledger.counts = match teacher_solves {
            Some(t) => SolveCounts { pass: t.pass, fallback: t.fallback, audit: t.audit, verification: t.verification, kept: t.kept, unrecorded_before: !t.complete },
            None => SolveCounts { unrecorded_before: true, ..SolveCounts::default() },
        };
        if let Some(n) = teacher_solves_note {
            self.user_notes.push(format!("surrogate: {n}"));
        }
        // the fit, when one can form (the refit's own errors are the reason shown otherwise)
        if self.no_model().is_some_and(|nm| nm.missing.is_empty() && nm.retry) {
            if let Err(e) = SurrogateKs::refit(self) {
                self.last_fit_error = Some(e.to_string());
            }
        }
        Ok(())
    }

    /// Species whose reference energy predates ruling S9 (from a sidecar without the rule).
    pub fn legacy_references(&self) -> Vec<u8> {
        self.legacy_refs.iter().copied().collect()
    }

    /// Replaces everything learned with a sidecar's contents (the caller has checked its
    /// provenance against the scene — `Dynamics` refuses a mismatch). The file must hold the
    /// fit report and at least the samples its accumulator was built from (a model-only file
    /// from `Model::write_sidecar` cannot be judged by the trust rule's training residuals and
    /// is refused, and so is a file without the posterior covariance). The training residuals are recomputed from the samples; the surface
    /// bookkeeping is a refit's (epoch, jump at the last evaluated positions). The lifetime
    /// teaching solves come from the file's record (absent or unreadable: unrecorded before, the
    /// count starting at zero, with a note for the unreadable one), set only once the install
    /// succeeded (plan review N11: a refused file leaves them alone).
    pub fn install_sidecar(&mut self, sc: surrogate::Sidecar) -> Result<(), String> {
        let surrogate::Sidecar { model, accumulator, report, samples, reference_rule, quarantine, face_density, teacher_solves, teacher_solves_note, failed_audits, .. } = sc;
        let Some(report) = report else {
            return Err("the file has no fit report (a model-only sidecar): it cannot be judged by the trust rule".into());
        };
        // without the posterior factor no answer can be judged by rules 5d, 6 and 7, and
        // `evaluate` would find no variance verdict at its first consultation (activity Task 4
        // review I4: it panicked there)
        if model.sigma_inv_chol.is_none() {
            return Err("the file has no posterior covariance: rules 5d, 6 and 7 cannot judge the model's answers".into());
        }
        if accumulator.basis != model.basis {
            return Err("the file's accumulator and model disagree on the basis".into());
        }
        if samples.len() < accumulator.n_samples {
            return Err(format!(
                "the file holds {} samples but its fit used {}: without them the training residuals cannot be checked (a model-only sidecar)",
                samples.len(),
                accumulator.n_samples
            ));
        }
        if model.ref_energy.len() != model.basis.n_species() {
            return Err("the file's reference energies do not match its species".into());
        }
        // a draft guess's audit is held out of the fit (`Q_DRAFT`): restored from its tag
        let mut quarantine = quarantine;
        if let Some(last) = samples.iter().rposition(|s| s.tag == SampleTag::DraftAudit) {
            quarantine.resize(quarantine.len().max(last + 1), Q_NONE);
            for (i, s) in samples.iter().enumerate() {
                if s.tag == SampleTag::DraftAudit && quarantine[i] == Q_NONE {
                    quarantine[i] = Q_DRAFT;
                }
            }
        }
        let mut store = SampleStore::new(model.provenance.clone());
        for (i, smp) in samples.into_iter().enumerate() {
            store.push(smp).map_err(|e| format!("sample {i}: {e}"))?;
        }
        let was_trusted = self.trusted();
        let was_model = self.last.as_ref().is_some_and(|l| l.on_model);
        self.refs = model.basis.species.iter().copied().zip(model.ref_energy.iter().copied()).collect();
        self.legacy_refs = match reference_rule.as_deref() {
            Some(surrogate::sidecar::REFERENCE_RULE) => Default::default(),
            _ => model.basis.species.iter().copied().collect(),
        };
        self.refs_dirty = false;
        self.species_dirty = store.samples.iter().flat_map(|s| s.z.iter()).any(|&z| model.basis.species_index(z).is_none());
        // the accumulated store prefix: the samples the fit used plus the quarantined ones among
        // them (quarantine only ever marks samples inside the prefix)
        let q = |i: usize| quarantine.get(i).is_some_and(|&c| c != Q_NONE);
        let mut prefix = 0usize;
        let mut fitted = 0usize;
        while prefix < store.len() && (fitted < accumulator.n_samples || q(prefix)) {
            if !q(prefix) {
                fitted += 1;
            }
            prefix += 1;
        }
        self.accumulated = prefix;
        self.pending_refit = store.len() > self.accumulated;
        self.envelope = Some(accumulator.envelope.clone());
        self.null_space = Some(NullSpace::of(&accumulator, &model.scale));
        self.accumulator = Some(accumulator);
        self.face = if face_density.is_empty() { vec![f64::NAN; store.len()] } else { face_density };
        self.store = Some(store);
        self.quarantine = quarantine;
        self.pass_start = None;
        self.pass_coverage = None;
        self.fence_votes.clear();
        self.fence_kept.clear();
        self.vote_pending.clear();
        self.degree_cap = (model.basis.body3 && model.basis.degree_cap != Basis::new(&[]).degree_cap).then_some(model.basis.degree_cap);
        self.desc_cache.clear();
        self.desc_basis = None;
        self.since_kept = THIN_EVERY;
        self.recovery = None;
        self.reset_audits();
        self.model = Some(model);
        self.report = Some(report);
        self.fits += 1;
        // readouts (activity Task 4 review): the views' cache key and their `Fitted`
        self.model_gen += 1;
        self.fit_members = (self.accumulated, self.quarantine.clone());
        self.audit_failed = None;
        self.training_residuals = self.compute_training_residuals();
        self.refresh_class_loo();
        self.rebuild_active_set();
        // round 1, SHOULD-4: the reconstructed fit membership must be the fit's own — a release
        // between a fit and a save (Q_HELD → Q_NONE) shifts the prefix and would give a non-member
        // a leave-one-out "as if fitted". Checked against the report's row and sample counts; on a
        // mismatch the fit is rebuilt from the stored samples (no DFT)
        // Counts alone cannot see a shift between samples of the same size, so the accumulator's
        // label sums are recomputed from the members too — yy_E = Σ w y_E² with y_E = E − Σ E_ref
        // (÷ N for a per-atom fit) and yy_F = Σ w Σ_j |F_j|², no descriptor needed, Neumaier-
        // compensated — and compared at 1e-9 relative
        let neumaier = |acc: &mut (f64, f64), v: f64| {
            let t = acc.0 + v;
            acc.1 += if acc.0.abs() >= v.abs() { (acc.0 - t) + v } else { (v - t) + acc.0 };
            acc.0 = t;
        };
        let (members, rows, yy_e, yy_f) = {
            let st = self.store.as_ref().expect("installed");
            let model = self.model.as_ref().expect("installed");
            let (mut m, mut r, mut ye, mut yf) = (0usize, 0usize, (0.0f64, 0.0f64), (0.0f64, 0.0f64));
            for i in (0..self.accumulated).filter(|&i| !self.excluded(i)) {
                let smp = &st.samples[i];
                m += 1;
                r += 1 + 3 * smp.pos.len();
                let refs: f64 = smp.z.iter().map(|&z| model.basis.species_index(z).map_or(f64::NAN, |k| model.ref_energy[k])).sum();
                let y = if model.energy_per_atom { (smp.energy - refs) / smp.pos.len().max(1) as f64 } else { smp.energy - refs };
                neumaier(&mut ye, smp.weight * y * y);
                for x in smp.forces.iter().flatten() {
                    neumaier(&mut yf, smp.weight * x * x);
                }
            }
            (m, r, ye.0 + ye.1, yf.0 + yf.1)
        };
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b.abs().max(f64::MIN_POSITIVE);
        let (acc_ye, acc_yf) = self.accumulator.as_ref().map_or((yy_e, yy_f), |a| {
            let s = a.sums();
            (s.yy_e, s.yy_f)
        });
        let (want_n, want_rows) = self.report.as_ref().map_or((members, rows), |r| (r.n_samples, r.n_rows));
        if (members, rows) != (want_n, want_rows) || !close(yy_e, acc_ye) || !close(yy_f, acc_yf) {
            self.species_dirty = true;
            self.fit_store().map_err(|e| format!("the file's fit membership ({members} samples, {rows} rows) does not match its report ({want_n}, {want_rows}) and the rebuild failed: {e}"))?;
            self.user_notes.push(format!(
                "surrogate: the file's fit membership ({members} samples, {rows} rows) did not match its report ({want_n}, {want_rows}) or its label sums — the fit was rebuilt from its stored samples (no DFT)"
            ));
        }
        // round 1, SHOULD-2: the failed audits it still held, re-judged on this fit (not counted
        // as a refit's re-judgement)
        let n_store = self.samples();
        for c in failed_audits {
            let origin = AuditOrigin::from_word(&c.origin);
            // the reader enforces this (sidecar format 5); a shell fence without its pair would
            // refuse nothing, so it is never installed silently
            if (origin == AuditOrigin::ShellCheck) != c.pair.is_some() {
                return Err("the file's failed audits: a shell-check fence must carry its species pair, and only one may".into());
            }
            let fa = FailedAudit { pos: c.pos, z: c.z, energy: c.energy, forces: c.forces, index: c.index.filter(|&i| i < n_store), ionic: c.ionic, rejudges: c.rejudges, origin, pair: c.pair };
            if c.fenced {
                self.audit_fences.push(fa);
            } else {
                self.failed_audits.push(fa);
            }
        }
        self.rejudge_failed_audits(false);
        self.consults = 0;
        self.last_variance = None;
        self.answered = 0;
        // Cannot fail here: `surface_moved` predicts only at a geometry whose species the new
        // basis covers (it checks `species_indices` first), and `describe` has no other error;
        // the state above is therefore never left half-installed (review m7).
        self.surface_moved(was_trusted, was_model).map_err(|e| e.to_string())?;
        self.ledger.counts = match teacher_solves {
            Some(t) => SolveCounts { pass: t.pass, fallback: t.fallback, audit: t.audit, verification: t.verification, kept: t.kept, unrecorded_before: !t.complete },
            None => SolveCounts { unrecorded_before: true, ..SolveCounts::default() },
        };
        if let Some(n) = teacher_solves_note {
            self.user_notes.push(format!("surrogate: {n}"));
        }
        Ok(())
    }

    /// Provenance of a sample produced at this scene's settings by this teacher; `masses` is the
    /// scene's isotope set (sorted unique (Z, A)), recorded but never compared (spec §5.3, P2).
    pub(super) fn scene_provenance(&self, scene: &Scene) -> Provenance {
        let mut masses: Vec<(u8, u16)> = scene.nuclei.iter().map(|n| (n.z, n.a)).collect();
        masses.sort_unstable();
        masses.dedup();
        // the compared tolerance is the nominal one; a pseudo surface's GPU stop rule max(tol, 2·floor(N)) (spec §7
        // risk 6) is part of the ionic tag (the rule, not its N-dependent value — an Add or Remove must not split a
        // model's surface), and each answer's effective tolerance is recorded in `EvalStats::scf_tol_used`
        let ionic = crate::pseudo::provenance_tag(scene);
        let tol = self.teacher_tol;
        Provenance {
            grid_n: scene.cell.n,
            h: scene.cell.h,
            sigma: scene.sigma.unwrap_or(scene.cell.h),
            tol,
            charge: scene.charge,
            multiplicity: scene.multiplicity,
            teacher: self.teacher.name().to_string(),
            masses,
            ionic,
        }
    }

    /// Into the store (created with `prov` if absent); marks the basis dirty for a new species.
    pub(super) fn push_sample(&mut self, s: Sample, prov: &Provenance, face: f64) -> Result<(), StoreError> {
        let z = s.z.clone();
        let n = self.store.as_ref().map_or(0, |st| st.len());
        if !self.quarantine.is_empty() {
            self.quarantine.resize(n, Q_NONE);
        }
        match &mut self.store {
            Some(store) => {
                store.push_checked(s, prov)?;
                // the record grows with the isotopes the samples' scenes carried (never gated on)
                store.provenance.record_masses(&prov.masses);
            }
            None => {
                let mut store = SampleStore::new(prov.clone());
                store.push(s)?;
                self.store = Some(store);
            }
        }
        self.face.resize(n, f64::NAN);
        self.face.push(face);
        if let Some(acc) = &self.accumulator {
            if z.iter().any(|&zi| acc.basis.species_index(zi).is_none()) {
                self.species_dirty = true;
            }
        }
        Ok(())
    }

    /// Into the store as a fallback sample (an admitted teacher answer); marks the refit pending.
    /// False when the store refused it (the note says why; the outcome is `StoreRefused`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn keep_sample(&mut self, pos: &[[f64; 3]], z: &[u8], energy: f64, forces: &[[f64; 3]], face: Option<f64>, prov: &Provenance, tag: SampleTag, notes: &mut Vec<String>) -> bool {
        self.since_kept = 0;
        let sample = Sample { pos: pos.to_vec(), z: z.to_vec(), energy, forces: forces.to_vec(), tag, weight: 1.0 };
        match self.push_sample(sample, prov, face.unwrap_or(f64::NAN)) {
            Ok(()) => {
                if tag == SampleTag::DraftAudit {
                    // a draft guess's audit: held out of the strict fit (`Q_DRAFT`)
                    let n = self.samples();
                    self.quarantine.resize(n, Q_NONE);
                    self.quarantine[n - 1] = Q_DRAFT;
                    notes.push("surrogate: kept, held out of the strict fit — a draft-mode audit at a geometry strict mode refuses (it would otherwise judge the strict model by the guess's own excursions)".into());
                }
                self.pending_refit = true;
                true
            }
            Err(e) => {
                notes.push(format!("surrogate: sample not kept ({e})"));
                false
            }
        }
    }

    /// [`admit`](Self::admit) then [`keep_sample`](Self::keep_sample) when admitted: the
    /// admission's outcome, `StoreRefused` when the store refused an admitted answer. A refusal
    /// after `admit` released or superseded fenced samples keeps those flips (admit made them; as
    /// before Task 2): the outcome then says `StoreRefused` and the release survives in admit's
    /// note, which the event's detail carries (Task 2 review M5).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn admit_and_keep(
        &mut self,
        pos: &[[f64; 3]],
        z: &[u8],
        energy: f64,
        forces: &[[f64; 3]],
        face: Option<f64>,
        model: Option<(f64, &[[f64; 3]])>,
        near: &[usize],
        keep: bool,
        thin: bool,
        dedup: bool,
        essential: bool,
        prov: &Provenance,
        unfit: Option<&str>,
        tag: SampleTag,
        notes: &mut Vec<String>,
    ) -> TeachOutcome {
        // M7 (spec §5.6; Task 6 review A.6): an answer with overlapping cores, or whose shared-shell pattern changed,
        // is used for its step and never kept — before `admit` can quarantine, fence or release anything on it
        if let Some(why) = unfit {
            notes.push(format!("surrogate: sample not kept ({why} — spec §5.6)"));
            return TeachOutcome::UnfitNotKept;
        }
        let a = self.admit(pos, z, energy, forces, face, model, near, keep, thin, dedup, essential, prov.h, notes);
        if a.kept && !self.keep_sample(pos, z, energy, forces, face, prov, tag, notes) {
            return TeachOutcome::StoreRefused;
        }
        a.outcome
    }
}
