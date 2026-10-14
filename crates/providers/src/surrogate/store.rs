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
        self.report = None;
        self.envelope = None;
        self.null_space = None;
        self.audit_force = None;
        self.training_residuals = None;
        self.audit_failed = None;
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
        self.reset_audits();
    }

    /// Writes the full `.model` sidecar (`surrogate::sidecar`): model, accumulator, last fit
    /// report and every sample held — including any taken since the last refit, which a load
    /// restores as pending. `InvalidInput` if nothing has been fitted yet.
    pub fn save_sidecar(&self, path: &std::path::Path) -> std::io::Result<()> {
        let (Some(model), Some(acc)) = (&self.model, &self.accumulator) else {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "no fitted model to save (sample and refit first)"));
        };
        let samples = self.store.as_ref().map_or(&[][..], |s| &s.samples[..]);
        let rule = self.legacy_refs.is_empty().then_some(surrogate::sidecar::REFERENCE_RULE);
        let flags: Vec<u8> = (0..samples.len()).map(|i| self.quarantine.get(i).copied().unwrap_or(Q_NONE)).collect();
        // the face densities only when one matters (a file is otherwise unchanged)
        let face: Vec<f64> = if (0..samples.len()).any(|i| self.is_box(i)) {
            (0..samples.len()).map(|i| self.face.get(i).copied().unwrap_or(f64::NAN)).collect()
        } else {
            Vec::new()
        };
        surrogate::sidecar::write_with_flags(path, model, acc, self.report.as_ref(), samples, rule, &flags, &face)
    }

    /// Species whose reference energy predates ruling S9 (from a sidecar without the rule).
    pub fn legacy_references(&self) -> Vec<u8> {
        self.legacy_refs.iter().copied().collect()
    }

    /// Replaces everything learned with a sidecar's contents (the caller has checked its
    /// provenance against the scene — `Dynamics` refuses a mismatch). The file must hold the
    /// fit report and at least the samples its accumulator was built from (a model-only file
    /// from `Model::write_sidecar` cannot be judged by the trust rule's training residuals and
    /// is refused). The training residuals are recomputed from the samples; the surface
    /// bookkeeping is a refit's (epoch, jump at the last evaluated positions).
    pub fn install_sidecar(&mut self, sc: surrogate::Sidecar) -> Result<(), String> {
        let surrogate::Sidecar { model, accumulator, report, samples, reference_rule, quarantine, face_density, .. } = sc;
        let Some(report) = report else {
            return Err("the file has no fit report (a model-only sidecar): it cannot be judged by the trust rule".into());
        };
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
        self.audit_failed = None;
        self.training_residuals = self.compute_training_residuals();
        self.consults = 0;
        self.last_variance = None;
        self.answered = 0;
        // Cannot fail here: `surface_moved` predicts only at a geometry whose species the new
        // basis covers (it checks `species_indices` first), and `describe` has no other error;
        // the state above is therefore never left half-installed (review m7).
        self.surface_moved(was_trusted, was_model).map_err(|e| e.to_string())
    }

    /// Provenance of a sample produced at this scene's settings by this teacher; `masses` is the
    /// scene's isotope set (sorted unique (Z, A)), recorded but never compared (spec §5.3, P2).
    pub(super) fn scene_provenance(&self, scene: &Scene) -> Provenance {
        let mut masses: Vec<(u8, u16)> = scene.nuclei.iter().map(|n| (n.z, n.a)).collect();
        masses.sort_unstable();
        masses.dedup();
        Provenance {
            grid_n: scene.cell.n,
            h: scene.cell.h,
            sigma: scene.sigma.unwrap_or(scene.cell.h),
            tol: self.teacher_tol,
            charge: scene.charge,
            multiplicity: scene.multiplicity,
            teacher: self.teacher.name().to_string(),
            masses,
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
    #[allow(clippy::too_many_arguments)]
    pub(super) fn keep_sample(&mut self, pos: &[[f64; 3]], z: &[u8], energy: f64, forces: &[[f64; 3]], face: Option<f64>, prov: &Provenance, notes: &mut Vec<String>) {
        self.since_kept = 0;
        let sample = Sample { pos: pos.to_vec(), z: z.to_vec(), energy, forces: forces.to_vec(), tag: SampleTag::Fallback, weight: 1.0 };
        match self.push_sample(sample, prov, face.unwrap_or(f64::NAN)) {
            Ok(()) => self.pending_refit = true,
            Err(e) => notes.push(format!("surrogate: sample not kept ({e})")),
        }
    }
}
