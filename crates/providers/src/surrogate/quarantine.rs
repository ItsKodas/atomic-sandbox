//! Trust recovery, quarantine, fences and thinning (the `surrogate` module doc, "Trust recovery and
//! quarantine" — ruling S15/S16: the degree-cap retry, quarantine rounds, release and supersede,
//! box-affected answers, fence voting, fallback thinning, the no-model cap) and `admit`, which
//! decides whether a teacher answer becomes a sample. Split out of `surrogate_ks.rs` in M6 Task 1;
//! the narrative stays in the parent module doc.

use super::*;

/// Fallback thinning and the quarantine neighbourhood, as a fraction of
/// `FallbackPolicy::max_descriptor_distance` (`surrogate` module doc, "Trust recovery").
pub const THIN_FRACTION: f64 = 0.25;


/// One in this many thinned candidates (fallback answers close to a stored sample) is kept
/// anyway (`surrogate` module doc, "Fallback thinning"; also the no-model cap's keep rate).
pub const THIN_EVERY: usize = 10;

/// Without a model, teacher answers are kept unthinned up to this many times Q′ (the default
/// basis of the held species) samples, then one in [`THIN_EVERY`] (`surrogate` module doc, "No model yet").
pub const NO_MODEL_SAMPLE_FACTOR: usize = 4;

/// A geometry within this fraction of `max_descriptor_distance` (training-spread units) of a
/// quarantined sample falls back (`surrogate` module doc): small — the quarantined geometry itself and its
/// immediate neighbourhood, not the region around it the model still represents. A geometric
/// choice, not a measured one (no measurement sets it; `references.md` policy table says so).
pub const QUARANTINE_FRACTION: f64 = 0.1;

/// The body-3 degree cap the first recovery step retries with (the bent-molecule setting).
pub const RECOVERY_DEGREE_CAP: usize = 6;

/// Fence voting (Task 9 fix round 1, I3): this many mutually consistent fresh teacher answers
/// inside a quarantine fence that all disagree with the model keep one of them for the fit…
pub const FENCE_VOTES: usize = 3;

/// … at most this many times per fenced sample (then the region stays fenced: the model could
/// not learn it from what it was given).
pub const FENCE_VOTE_LIMIT: u32 = 3;

/// Quarantine codes per stored sample (the sidecar's `sample_quarantine` values): aliases of the
/// `surrogate` crate's [`surrogate::quarantine`] codes, their one source of truth since M8a Task 3
/// (ruling P4; the meaning of each is documented there). Values unchanged.
pub(super) const Q_NONE: u8 = surrogate::quarantine::Q_NONE;

pub(super) const Q_HELD: u8 = surrogate::quarantine::Q_HELD;

pub(super) const Q_SUPERSEDED: u8 = surrogate::quarantine::Q_SUPERSEDED;

/// A collection sample whose box has no isolated-atom reference yet (M8a Task 4a fix round 2,
/// `shared` module doc, "References per box"): its fit target E − Σ ref of its own box cannot be
/// formed, so it waits out of the fit like a pending sample until that reference arrives. A
/// provider-only code: never written to a file or a collection (a collection's status is its
/// quarantine, and this is not one).
pub(super) const Q_REF_WAIT: u8 = 5;

/// A box-affected sample ([`surrogate::quarantine::Q_BOX`]).
pub(super) const Q_BOX: u8 = surrogate::quarantine::Q_BOX;

/// A draft-mode audit's answer held out of the strict fit ([`surrogate::quarantine::Q_DRAFT`]).
pub(super) const Q_DRAFT: u8 = surrogate::quarantine::Q_DRAFT;

/// Quarantine rounds per refit (`surrogate` module doc).
pub const QUARANTINE_ROUNDS: usize = 5;

/// Never quarantine more than this fraction of the store.
pub const QUARANTINE_MAX_FRACTION: f64 = 0.1;

/// One fresh teacher answer inside a quarantine fence that disagreed with the model (fence voting).
pub(super) struct FenceVote {
    pub(super) near: Vec<usize>,
    pub(super) n: usize,
    pub(super) energy: f64,
    pub(super) forces: Vec<[f64; 3]>,
}

/// L∞ distance between two descriptors of species `s` in the envelope's training-spread units
/// (a feature the training never varied counts as infinitely far once the two differ).
pub(super) fn spread_distance(env: &Envelope, s: usize, a: &[f64], b: &[f64]) -> f64 {
    let mut worst = 0.0f64;
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        let gap = (x - y).abs();
        if gap == 0.0 {
            continue;
        }
        let u = env.unit(s, k);
        worst = worst.max(if u > 0.0 { gap / u } else { f64::INFINITY });
    }
    worst
}

/// One stored sample's atoms as (species index, descriptor) under `basis`: the descriptor
/// cache's entry (`ensure_desc_cache`). A species the basis lacks gives no atoms: never "near"
/// anything.
pub(super) fn describe_sample(basis: &Basis, s: &Sample) -> Vec<(usize, Vec<f64>)> {
    match (describe_values(basis, &s.pos, &s.z), basis.species_indices(&s.z)) {
        (Ok(d), Ok(sp)) => sp.into_iter().zip(d).collect(),
        _ => Vec::new(),
    }
}

/// What `admit` decided about a teacher answer: kept or not, and which branch decided it
/// (activity spec A3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Admission {
    pub(super) kept: bool,
    pub(super) outcome: TeachOutcome,
}

impl SurrogateKs {
    /// Samples held in quarantine (`surrogate` module doc, "Trust recovery"): excluded from the fit, their
    /// neighbourhood falls back.
    pub fn quarantined(&self) -> usize {
        self.quarantine.iter().filter(|&&q| q == Q_HELD).count()
    }

    /// Quarantined samples a fresh teacher answer contradicted (`surrogate` module doc, "Release"): held,
    /// excluded, no longer fencing.
    pub fn superseded(&self) -> usize {
        self.quarantine.iter().filter(|&&q| q == Q_SUPERSEDED).count()
    }

    /// Whether stored sample `i` is quarantined (and fencing).
    pub fn is_quarantined(&self, i: usize) -> bool {
        self.quarantine.get(i) == Some(&Q_HELD)
    }

    /// Box-affected samples taken while the model was being built and excluded from the fit
    /// since (`surrogate` module doc, "Box-affected answers").
    pub fn box_excluded(&self) -> usize {
        self.quarantine.iter().filter(|&&q| q == Q_BOX).count()
    }

    /// Draft-mode audit answers held out of the strict fit (`Q_DRAFT`; `surrogate` module doc,
    /// "Draft mode").
    pub fn held_out(&self) -> usize {
        self.quarantine.iter().filter(|&&q| q == Q_DRAFT).count()
    }

    /// Stored samples kept since the last refit (not in the fit yet; M6 Task 11's reconciliation,
    /// `sim::TrainableStatus::unfitted`).
    pub fn unfitted(&self) -> usize {
        self.samples().saturating_sub(self.accumulated)
    }

    /// Stored samples whose teacher answer was box-affected (its face density above the box
    /// rule), excluded from the fit yet or not.
    pub fn box_affected(&self) -> usize {
        (0..self.samples()).filter(|&i| self.is_box(i)).count()
    }

    /// Whether stored sample `i`'s teacher answer is box-affected (its face density above the box
    /// rule at the store's grid spacing).
    pub(super) fn is_box(&self, i: usize) -> bool {
        let h = self.store.as_ref().map_or(0.25, |s| s.provenance.h);
        self.face.get(i).is_some_and(|&rho| rho > box_affected_face_density(h))
    }

    /// The trust floor of the stored species (Q′ + `MIN_RESIDUAL_DOF` of their default basis): the
    /// clean sample count the box-affected samples' exclusion waits for. `None` without a store.
    pub fn trust_floor(&self) -> Option<usize> {
        let store = self.store.as_ref()?;
        let mut species: Vec<u8> = store.samples.iter().flat_map(|s| s.z.iter().copied()).collect();
        species.sort_unstable();
        species.dedup();
        Some(Basis::new(&species).column_classes().len() + MIN_RESIDUAL_DOF as usize)
    }

    /// Once the store holds enough other samples for the trust floor (Q′ + 30), the box-affected
    /// ones leave the fit (`Q_BOX`; a rebuild). Whether anything changed.
    pub(super) fn exclude_box_samples(&mut self) -> bool {
        let Some(store) = self.store.as_ref() else { return false };
        let n = store.len();
        let boxed: Vec<usize> = (0..n).filter(|&i| self.is_box(i) && self.quarantine.get(i).is_none_or(|&q| q == Q_NONE)).collect();
        if boxed.is_empty() {
            return false;
        }
        let Some(floor) = self.trust_floor() else { return false };
        let others = (0..n).filter(|&i| !self.is_box(i) && self.quarantine.get(i).is_none_or(|&q| q == Q_NONE)).count();
        if others < floor {
            return false;
        }
        self.quarantine.resize(n, Q_NONE);
        for i in boxed {
            self.quarantine[i] = Q_BOX;
        }
        self.species_dirty = true;
        true
    }

    /// Whether stored sample `i` is left out of the fit (quarantined, superseded, box-excluded or a
    /// held-out draft guess).
    pub(super) fn excluded(&self, i: usize) -> bool {
        self.quarantine.get(i).is_some_and(|&q| q != Q_NONE)
    }

    /// What the last refit's trust recovery did ("degree_cap 6 …", "N samples quarantined …"),
    /// `None` when it was not needed.
    pub fn recovery_note(&self) -> Option<&str> {
        self.recovery.as_deref()
    }

    /// Ruling S15 (`surrogate` module doc, "Trust recovery"): the degree-cap retry, then quarantine rounds.
    /// Returns the report of the fit that stands; a failed attempt restores the previous fit.
    pub(super) fn recover_trust(&mut self, report: FitReport) -> FitReport {
        // 1. the bent-molecule setting
        let cap = self.basis().map(|b| (b.body3, b.degree_cap));
        if let Some((true, d)) = cap.filter(|_| self.degree_cap.is_none()) {
            if d < RECOVERY_DEGREE_CAP {
                let saved = self.save_state();
                self.degree_cap = Some(RECOVERY_DEGREE_CAP);
                self.species_dirty = true;
                // kept only if the trust rule's counts then hold too: a retry that trades a residual
                // failure for a sample shortfall (Q′ grows with the cap: 176 → 344 for H/O) is not a
                // recovery — the 2026-10-18 water session's E1(e): one H 0.9 Bohr from O took the
                // model to n 333 < Q′ 344, every step DFT, then rule 4b by that one sample
                match self.fit_store() {
                    Ok(r) if !self.recoverable_fail() && !self.trust_needs_samples() => {
                        self.recovery = Some(format!("refit with degree_cap {RECOVERY_DEGREE_CAP} (the bent-molecule setting) restored the training residuals"));
                        return r;
                    }
                    _ => self.restore_state(saved),
                }
            }
        }
        // 2. quarantine the worst offenders, round by round
        let mut report = report;
        let n_store = self.samples();
        let before = self.quarantined();
        for _ in 0..QUARANTINE_ROUNDS {
            if !self.recoverable_fail() {
                break;
            }
            let worst = self.worst_offenders();
            let total = self.quarantined() + worst.len();
            if worst.is_empty() || total as f64 > QUARANTINE_MAX_FRACTION * n_store as f64 {
                break;
            }
            let saved = self.save_state();
            let flags = self.quarantine.clone();
            self.quarantine.resize(n_store, Q_NONE);
            for &i in &worst {
                self.quarantine[i] = Q_HELD;
            }
            self.species_dirty = true; // rebuild without them
            match self.fit_store() {
                Ok(r) => report = r,
                Err(_) => {
                    self.restore_state(saved);
                    self.quarantine = flags;
                    break;
                }
            }
        }
        let added = self.quarantined() - before;
        if added > 0 {
            self.recovery = Some(format!(
                "{} samples quarantined — the model cannot represent them ({added} this refit; kept, excluded from the fit; geometries near them fall back)",
                self.quarantined()
            ));
        }
        report
    }

    /// Whether the model reproduces (energy, forces) at a geometry within the policy's thresholds.
    pub(super) fn reproduces(&self, pos: &[[f64; 3]], z: &[u8], energy: f64, forces: &[[f64; 3]]) -> bool {
        let Some(model) = self.model.as_ref() else { return false };
        let Ok(pr) = model.predict(pos, z, false) else { return false };
        Self::agree(&self.policy, pos.len(), pr.energy, &pr.forces, energy, forces)
    }

    /// Two answers agree within the policy's thresholds (per-atom energy; every atom's force
    /// against its bound for the reference force).
    pub(super) fn agree(p: &FallbackPolicy, n: usize, e_a: f64, f_a: &[[f64; 3]], e_b: f64, f_b: &[[f64; 3]]) -> bool {
        (e_a - e_b).abs() / n.max(1) as f64 <= p.energy_std_per_atom
            && f_a.iter().zip(f_b).all(|(a, b)| {
                let (d, bound) = p.force_residual(a, b);
                d <= bound
            })
    }

    /// The refit's verdict on the samples fence votes kept (`surrogate` module doc, "Fence voting"): one the
    /// model (fitted with it) reproduces stays in the fit; one it does not is quarantined — a
    /// geometry the model cannot represent — and its region (the fenced samples it was kept next
    /// to, and itself) takes no further vote. How many were quarantined.
    pub(super) fn judge_votes(&mut self) -> usize {
        let pending = std::mem::take(&mut self.vote_pending);
        let mut rejected = 0;
        for (k, origin) in pending {
            let Some(s) = self.store.as_ref().and_then(|st| st.samples.get(k)).cloned() else { continue };
            if self.excluded(k) || self.reproduces(&s.pos, &s.z, s.energy, &s.forces) {
                continue;
            }
            self.quarantine.resize(self.samples(), Q_NONE);
            self.quarantine[k] = Q_HELD;
            for i in origin.into_iter().chain([k]) {
                self.fence_kept.insert(i, FENCE_VOTE_LIMIT);
            }
            rejected += 1;
        }
        rejected
    }

    /// Releases the quarantined samples the current model reproduces; how many.
    pub(super) fn release_reproduced(&mut self) -> usize {
        let held: Vec<usize> = (0..self.quarantine.len()).filter(|&i| self.quarantine[i] == Q_HELD).collect();
        let Some(store) = self.store.as_ref() else { return 0 };
        let back: Vec<usize> = held
            .into_iter()
            .filter(|&i| store.samples.get(i).is_some_and(|s| self.reproduces(&s.pos, &s.z, s.energy, &s.forces)))
            .collect();
        for &i in &back {
            self.quarantine[i] = self.released_code(i);
        }
        back.len()
    }

    /// The quarantined (fencing) samples within the quarantine radius of the geometry: for each
    /// held sample, the worst query atom's distance to that sample's nearest atom of its species
    /// (training-spread units, L∞ over features). One description of the query for all of them
    /// (Task 9: it used to re-describe the query once per held sample).
    pub(super) fn near_quarantined(&mut self, pos: &[[f64; 3]], z: &[u8]) -> Vec<usize> {
        let limit = QUARANTINE_FRACTION * self.policy.max_descriptor_distance;
        self.quarantine_distances(pos, z).into_iter().filter(|&(_, d)| d <= limit).map(|(i, _)| i).collect()
    }

    /// (index, distance) of every held quarantined sample from the geometry (`near_quarantined`).
    /// Fills the descriptor cache, then reads it ([`quarantine_distances_cached`](Self::quarantine_distances_cached)).
    pub(super) fn quarantine_distances(&mut self, pos: &[[f64; 3]], z: &[u8]) -> Vec<(usize, f64)> {
        if self.quarantined() == 0 {
            return Vec::new();
        }
        self.ensure_desc_cache();
        self.quarantine_distances_cached(pos, z)
    }

    /// [`quarantine_distances`](Self::quarantine_distances), read-only (activity plan review S1:
    /// `would_answer` is `&self`). It reads the descriptor cache the caller filled
    /// (`ensure_desc_cache`; `training_views` does); a held sample the cache does not cover under
    /// the model's basis (not filled, or filled under an older basis) is described on the fly —
    /// the same `describe_values` the cache would have stored, so the distances are the same
    /// numbers either way, only slower. Without a model there is nothing to fence with (the
    /// filling path's cache basis is the model's, `None` without one).
    pub(super) fn quarantine_distances_cached(&self, pos: &[[f64; 3]], z: &[u8]) -> Vec<(usize, f64)> {
        if self.quarantined() == 0 {
            return Vec::new();
        }
        let (Some(model), Some(env), Some(store)) = (self.model.as_ref(), self.envelope.as_ref(), self.store.as_ref()) else { return Vec::new() };
        let basis = &model.basis;
        let (Ok(desc), Ok(sp)) = (describe_values(basis, pos, z), basis.species_indices(z)) else { return Vec::new() };
        let cached = self.desc_basis.as_ref() == Some(basis);
        (0..store.len())
            .filter(|&i| self.quarantine.get(i) == Some(&Q_HELD))
            .map(|i| {
                let owned;
                let atoms: &[(usize, Vec<f64>)] = match self.desc_cache.get(i).filter(|_| cached) {
                    Some(a) => a,
                    None => {
                        owned = describe_sample(basis, &store.samples[i]);
                        &owned
                    }
                };
                let worst = sp.iter().zip(&desc).fold(0.0f64, |worst, (qs, qd)| {
                    let best = atoms
                        .iter()
                        .filter(|(s, _)| s == qs)
                        .map(|(s, b)| spread_distance(env, *s, qd, b))
                        .fold(f64::INFINITY, f64::min);
                    worst.max(best)
                });
                (i, worst)
            })
            .collect()
    }

    /// [`near_quarantined`](Self::near_quarantined), read-only
    /// ([`quarantine_distances_cached`](Self::quarantine_distances_cached)).
    pub(super) fn near_quarantined_cached(&self, pos: &[[f64; 3]], z: &[u8]) -> Vec<usize> {
        let limit = QUARANTINE_FRACTION * self.policy.max_descriptor_distance;
        self.quarantine_distances_cached(pos, z).into_iter().filter(|&(_, d)| d <= limit).map(|(i, _)| i).collect()
    }

    /// The training samples (not quarantined) whose residual ratio (per-atom energy residual
    /// over its threshold, or worst atom force residual over its bound) exceeds 1 and is within a
    /// factor 2 of the worst ratio. The force bound is the policy's per-atom bound while the
    /// training residuals fail; when only the force noise does (σ_f above `force_std_abs`, the
    /// trust rule's noise condition) it is the absolute `force_std_abs`: the samples whose force
    /// residuals put σ_f above the tier are the ones the recovery takes out.
    pub(super) fn worst_offenders(&self) -> Vec<usize> {
        let (Some(model), Some(store)) = (self.model.as_ref(), self.store.as_ref()) else { return Vec::new() };
        let p = &self.policy;
        let absolute = !self.residuals_fail();
        let ratios: Vec<(usize, f64)> = store.samples[..self.accumulated.min(store.len())]
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.excluded(*i) && !self.is_box(*i))
            .filter_map(|(i, s)| {
                let pr = model.predict(&s.pos, &s.z, false).ok()?;
                let n = s.pos.len().max(1) as f64;
                let mut r = (pr.energy - s.energy).abs() / n / p.energy_std_per_atom;
                for (fm, fs) in pr.forces.iter().zip(&s.forces) {
                    let (d, bound) = p.force_residual(fm, fs);
                    r = r.max(d / if absolute { p.force_std_abs } else { bound });
                }
                Some((i, r))
            })
            .collect();
        let worst = ratios.iter().map(|x| x.1).fold(0.0, f64::max);
        if worst <= 1.0 {
            return Vec::new();
        }
        ratios.into_iter().filter(|&(_, r)| r > 1.0 && r >= 0.5 * worst).map(|(i, _)| i).collect()
    }

    /// The extrapolation grade's active sets (rule 5e) over the atoms of the samples the current
    /// fit was built from (the accumulated prefix, none excluded), standardised as the model is.
    pub(super) fn rebuild_active_set(&mut self) {
        self.ensure_desc_cache();
        let (Some(model), Some(basis)) = (self.model.as_ref(), self.desc_basis.as_ref()) else {
            self.active_set = None;
            return;
        };
        let n = self.accumulated.min(self.desc_cache.len());
        let atoms = (0..n).filter(|&i| !self.excluded(i)).flat_map(|i| self.desc_cache[i].iter().map(|(s, b)| (*s, &b[..])));
        let set = ActiveSet::build(basis, &model.scale, atoms);
        if !set.converged() {
            // the γ ≤ 1 + δ guarantee is void: every query falls back on 5e until a refit converges
            self.user_notes.push(format!(
                "surrogate: the extrapolation grade's MaxVol did not converge within {} swaps — every step falls back (rule 5e) until a refit converges",
                surrogate::grade::MAXVOL_MAX_SWAPS
            ));
        }
        self.active_set = Some(set);
    }

    /// Fills the per-sample descriptor cache under the current model's basis.
    pub(super) fn ensure_desc_cache(&mut self) {
        let Some(basis) = self.model.as_ref().map(|m| m.basis.clone()) else { return };
        if self.desc_basis.as_ref() != Some(&basis) {
            self.desc_cache.clear();
            self.desc_basis = Some(basis.clone());
        }
        let Some(store) = self.store.as_ref() else { return };
        for s in &store.samples[self.desc_cache.len()..] {
            self.desc_cache.push(describe_sample(&basis, s));
        }
    }

    /// The distance (training-spread units, L∞ over features; the worst query atom's nearest
    /// stored atom of its species) from the geometry to the nearest stored sample (thinning),
    /// held-out draft guesses (`Q_DRAFT`) left out — they are not the strict store's data, and
    /// counting them would thin (and, through `is_label_copy`, refuse) the fallbacks that teach
    /// strict mode the region (round 1 re-review, R1-1). ∞ without a model, an envelope or a candidate.
    pub(super) fn nearest_stored(&mut self, pos: &[[f64; 3]], z: &[u8]) -> f64 {
        self.ensure_desc_cache();
        let (Some(basis), Some(env)) = (self.desc_basis.as_ref(), self.envelope.as_ref()) else { return f64::INFINITY };
        let (Ok(desc), Ok(sp)) = (describe_values(basis, pos, z), basis.species_indices(z)) else { return f64::INFINITY };
        let mut worst_atom = 0.0f64;
        for (qs, qd) in sp.iter().zip(&desc) {
            let mut best = f64::INFINITY;
            for (i, atoms) in self.desc_cache.iter().enumerate() {
                if self.quarantine.get(i) == Some(&Q_DRAFT) {
                    continue;
                }
                for (s, b) in atoms {
                    if s == qs {
                        best = best.min(spread_distance(env, *s, qd, b));
                    }
                }
            }
            worst_atom = worst_atom.max(best);
        }
        worst_atom
    }

    /// Whether a teacher answer becomes a training sample (`surrogate` module doc: box-affected answers,
    /// quarantined neighbourhoods, thinning); when not, the reason goes to `notes`. `model` is the
    /// model's answer at the geometry when it was consulted; `near` the quarantined samples
    /// fencing it; `keep` the rule's own verdict; `thin` whether thinning applies. The
    /// [`Admission`] carries the verdict (`kept`) and which branch gave it (`outcome`, activity
    /// spec A3) — each return below is one branch; the outcome is read off it, never decides.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn admit(
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
        h: f64,
        notes: &mut Vec<String>,
    ) -> Admission {
        let no = |outcome| Admission { kept: false, outcome };
        let yes = |outcome| Admission { kept: true, outcome };
        if !keep {
            return no(TeachOutcome::NotOffered);
        }
        // no model yet (`surrogate` module doc, "No model yet"): unthinned up to the cap, then one in THIN_EVERY
        if self.model.is_none() {
            let cap = self.no_model_cap(z);
            if self.samples() >= cap {
                if self.since_kept + 1 < THIN_EVERY {
                    self.since_kept += 1;
                    notes.push(format!(
                        "surrogate: waiting for a sampling pass — {} samples held without a model (≥ {cap}); one in {THIN_EVERY} further teacher answers is kept",
                        self.samples()
                    ));
                    return no(TeachOutcome::WaitingForPass);
                }
            }
        }
        let rule = box_affected_face_density(h);
        // box-affected answers are the box, not the molecule (`surrogate` module doc) — except the samples a
        // model is still being built from (no model, a new species, trust short of samples), which
        // a sampling pass would take on this box anyway
        if let Some(rho) = face.filter(|&rho| !essential && rho > rule) {
            notes.push(format!(
                "surrogate: sample not kept — the teacher's density on the outermost grid plane is {rho:.1e} > {rule:.1e} (box-affected: used for this step, not trained on)"
            ));
            return no(TeachOutcome::BoxNotKept);
        }
        // a quarantined neighbourhood (`surrogate` module doc, "Release")
        if !near.is_empty() {
            let agrees = model.is_some_and(|(e, f)| Self::agree(&self.policy, pos.len(), e, f, energy, forces));
            if !agrees {
                // fence voting (`surrogate` module doc, "Fence voting"): fresh answers that agree with each
                // other but not with the model are evidence the model is wrong here
                let n = pos.len();
                self.fence_votes.push(FenceVote { near: near.to_vec(), n, energy, forces: forces.to_vec() });
                if self.fence_votes.len() > FENCE_VOTES {
                    self.fence_votes.remove(0);
                }
                // the last k answers: the same fence, mutually consistent (every pair agrees
                // within the per-atom energy threshold and every atom's force bound)
                let votes = &self.fence_votes;
                let consistent = votes.len() == FENCE_VOTES
                    && votes.iter().all(|v| v.n == n && v.near.iter().any(|i| near.contains(i)))
                    && votes.iter().enumerate().all(|(a, va)| {
                        votes[a + 1..].iter().all(|vb| Self::agree(&self.policy, n, va.energy, &va.forces, vb.energy, &vb.forces))
                    });
                // one vote per region until a refit has judged it; at most FENCE_VOTE_LIMIT per
                // fenced sample
                let pending = self.vote_pending.iter().any(|(_, o)| o.iter().any(|i| near.contains(i)));
                let room = !pending && near.iter().any(|i| self.fence_kept.get(i).copied().unwrap_or(0) < FENCE_VOTE_LIMIT);
                if consistent && room {
                    for &i in near {
                        *self.fence_kept.entry(i).or_insert(0) += 1;
                    }
                    self.fence_votes.clear();
                    self.vote_pending.push((self.samples(), near.to_vec()));
                    notes.push(format!(
                        "surrogate: kept — {FENCE_VOTES} mutually consistent fresh teacher answers next to a quarantined geometry disagree with the model; the next refit decides"
                    ));
                    return yes(TeachOutcome::KeptByVote);
                }
                notes.push("surrogate: sample not kept — next to a quarantined geometry, and the teacher still disagrees with the model there".into());
                return no(TeachOutcome::FenceDisagrees);
            }
            self.fence_votes.clear();
            let (mut released, mut superseded) = (0, 0);
            for &i in near {
                let s = self.store.as_ref().expect("a quarantined sample is stored").samples[i].clone();
                if self.reproduces(&s.pos, &s.z, s.energy, &s.forces) {
                    self.quarantine[i] = self.released_code(i);
                    released += 1;
                } else {
                    self.quarantine[i] = Q_SUPERSEDED;
                    superseded += 1;
                }
            }
            self.species_dirty = true; // the fit's sample set changed: rebuild
            notes.push(format!(
                "surrogate: a fresh teacher answer near quarantined samples agrees with the model — {released} released, {superseded} superseded"
            ));
            return yes(TeachOutcome::KeptReleased { released, superseded });
        }
        // thinning (`surrogate` module doc, "Trust recovery"): audits and the samples trust needs are
        // kept — but never a copy where copies could satisfy the rule (`is_label_copy`; round 1 of
        // the 2026-10-18 review, MUST-2)
        if thin || dedup {
            let d = self.nearest_stored(pos, z);
            let limit = THIN_FRACTION * self.policy.max_descriptor_distance;
            if dedup && d <= limit {
                if let Some(i) = self.is_label_copy(pos, z, energy, forces) {
                    notes.push(format!("surrogate: sample not kept — a copy of stored sample #{i} (its labels within the fit's own noise)"));
                    return no(TeachOutcome::Thinned);
                }
            }
            if thin && self.since_kept + 1 < THIN_EVERY && d <= limit {
                self.since_kept += 1;
                notes.push(format!(
                    "surrogate: sample not kept — {d:.2} training spreads from a stored one (≤ {limit:.2}; one in {THIN_EVERY} such is kept)"
                ));
                return no(TeachOutcome::Thinned);
            }
        }
        // past the box rule above only below it, or as an essential answer (the exemption): the
        // latter is kept box-affected
        yes(if face.is_some_and(|rho| rho > rule) { TeachOutcome::KeptBoxAffected } else { TeachOutcome::Kept })
    }

    /// A **copy** (round 1 of the 2026-10-18 trust review, MUST-2): the fitted (or pending) sample — not
    /// an excluded one: held out, quarantined, superseded, box-excluded (R1-1) — of the same atom
    /// sequence, inside the thinning radius and the nearest by the per-sample descriptor distance (the quarantine fence's: the
    /// worst atom's distance to its counterpart's species in that sample, training-spread units),
    /// whose labels this answer repeats within the fit's own noise — |E − E_s| ≤ N·s_E (s_E per atom
    /// for a per-atom fit) and every atom's force magnitude within s_F of its counterpart's (the
    /// magnitudes: rotation-invariant, so a rotating molecule's repeats are seen). Such a row tells
    /// the fit nothing it does not know to within its noise; fitted, a run of them shrinks the
    /// two-noise evidence's s_F (rows that add no residual) and with it rule 6's bound everywhere,
    /// and meets the count conditions by repetition — a molecule parked under rule 4, 4b or 5. A
    /// vibrating one is not a copy: its labels move by k·Δr per step, far above s_F (the active
    /// learning run: 6e-4 against 4.5e-5 Ha/Bohr). No constant beyond the fit's own noise levels.
    /// `None` without a fit.
    pub(super) fn is_label_copy(&self, pos: &[[f64; 3]], z: &[u8], energy: f64, forces: &[[f64; 3]]) -> Option<usize> {
        let (model, env, store, report) = (self.model.as_ref()?, self.envelope.as_ref()?, self.store.as_ref()?, self.report.as_ref()?);
        let basis = &model.basis;
        let (Ok(desc), Ok(sp)) = (describe_values(basis, pos, z), basis.species_indices(z)) else { return None };
        let cached = self.desc_basis.as_ref() == Some(basis);
        let (mut best, mut best_d) = (None, f64::INFINITY);
        for (i, s) in store.samples.iter().enumerate() {
            // only a sample that is (or will be) fitted: the rationale — fitted copies shrink s_F,
            // copies meet the counts — concerns the fit (round 1 re-review, R1-1: a held-out guess
            // compared against would refuse the very fallbacks that let strict mode reach it)
            if s.z != z || self.excluded(i) {
                continue;
            }
            let owned;
            let atoms: &[(usize, Vec<f64>)] = match self.desc_cache.get(i).filter(|_| cached) {
                Some(a) => a,
                None => {
                    owned = describe_sample(basis, s);
                    &owned
                }
            };
            let d = sp.iter().zip(&desc).fold(0.0f64, |worst, (qs, qd)| {
                worst.max(atoms.iter().filter(|(t, _)| t == qs).map(|(t, b)| spread_distance(env, *t, qd, b)).fold(f64::INFINITY, f64::min))
            });
            if d < best_d {
                (best, best_d) = (Some(i), d);
            }
        }
        let i = best?;
        if !(best_d <= THIN_FRACTION * self.policy.max_descriptor_distance) {
            return None;
        }
        let s = &store.samples[i];
        let n = z.len().max(1) as f64;
        let s_e = if model.energy_per_atom { n * report.sigma_e } else { report.sigma_e };
        let mag = |f: &[f64; 3]| (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
        let same = (energy - s.energy).abs() <= s_e && forces.iter().zip(&s.forces).all(|(a, b)| (mag(a) - mag(b)).abs() <= report.sigma_f);
        same.then_some(i)
    }

    /// The no-model cap (`surrogate` module doc, "No model yet"): [`NO_MODEL_SAMPLE_FACTOR`] × Q′ of the
    /// default basis over the held species and `z`'s.
    pub(super) fn no_model_cap(&self, z: &[u8]) -> usize {
        let mut species: Vec<u8> = z.to_vec();
        if let Some(store) = self.store.as_ref() {
            species.extend(store.samples.iter().flat_map(|s| s.z.iter().copied()));
        }
        species.sort_unstable();
        species.dedup();
        NO_MODEL_SAMPLE_FACTOR * Basis::new(&species).column_classes().len()
    }
}
