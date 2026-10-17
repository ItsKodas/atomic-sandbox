//! The training views (activity spec §3.3.5, §9, A6): what the Activity tab plots, built on
//! demand from the store and the current posterior — never a teacher call, never on the
//! evaluate path — and [`would_answer`](SurrogateKs::would_answer), the fallback policy's verdict
//! at a geometry, read-only by type.
//!
//! # What is computed (atomic units: Hartree, Bohr; F = −∇E)
//!
//! For a stored sample s with N atoms of species Z_i, teacher energy E_s and forces F_s, and the
//! model's prediction E_m, F_m = `Model::predict(pos, z, false)` at its positions:
//!
//! - **Energy parity** (spec §9.1, A4): x = (E_s − Σ_i E_ref(Z_i))/N, y = (E_m − Σ_i E_ref(Z_i))/N,
//!   Ha/atom, with E_ref = `Model::ref_energy` — the references the fit subtracted (plan review
//!   S4), not `self.refs`, which a later pass may have re-solved. y − x is the training residual
//!   |E_m − E_s|/N the trust rule bounds by `energy_std_per_atom`. Every stored sample, with its
//!   [`SampleState`]; omitted and counted in `parity_omitted` (S18): a sample whose species the
//!   basis lacks (no reference), and one whose label or prediction is not finite (review m5: a
//!   NaN point is no point).
//! - **Force parity**: (F_s,ja, F_m,ja) per Cartesian component, Ha/Bohr, of the **fitted**
//!   samples only, thinned to ≤ `FORCE_PARITY_CAP` components by whole atoms ([`thin_atoms`],
//!   S5). f32: these are plot coordinates, never summed.
//! - **Pair curve** of species (a, b) on r_k = r_min + (0.999 r_cut − r_min)·k/(K − 1), k = 0…K−1
//!   (r_min = `CURVE_R_MIN`, K = `CURVE_POINTS`; 0.999 r_cut keeps the last point inside the
//!   cutoff, where an atom still has a neighbour), the isolated dimer at (∓r/2, 0, 0) — every
//!   number below is invariant under the dimer's orientation (the descriptors are, and rule 6's
//!   per-atom trace is):
//!   - energy E_m(r) − E_ref(a) − E_ref(b), → 0 as r → r_cut (every feature with a neighbour
//!     vanishes there; scale-only standardisation, `surrogate::model`);
//!   - band ±σ_epi(r), σ_epi = √(max(σ_E(r)² − f², 0)), σ_E = `Prediction::energy_std` (the
//!     total energy's std, the floor included), f = N·s_E = 2 s_E for an `energy_per_atom` model
//!     (else s_E) — the floor `predict` adds in quadrature; NaN propagates (Task 3b). σ_epi is
//!     the **model's (coefficient) uncertainty given its basis — what rule 7 judges, not an error
//!     bar against DFT**: it contains no basis misfit, so the model's error can exceed it (the
//!     audits and the parity plot measure that error). Rule 7 compares σ_epi/N with
//!     `energy_std_per_atom`: the lane plots σ_epi/2 ÷ tier against 1. `dimer_floor` = f is
//!     **the fit's residual scatter s_E: teacher noise plus what this basis cannot fit** — more
//!     samples of the same kind do not remove it, a richer basis may (Task 4 review I1: not "the
//!     teacher's scatter"; s_E is the evidence search's homoscedastic residual, like σ_f);
//!   - rule 6's ratio max_j σ_F,epi,j / (√3 · b_j), b_j = min(max(abs, rel·|F_m,j|), 1e-2): the
//!     numbers `variance_check` compares with 1 (`VarianceVerdict::atoms`);
//!   - the verdict per r: [`would_answer`](SurrogateKs::would_answer)'s rule, from the same
//!     prediction (one `predict(…, true)` per point serves the band, the lane and the verdict;
//!     plan review S2). It is the dominant encoding of the plot (M5): the band does not say
//!     where the model answers — rules 4, 4b, 5a–5e, 6 and the fence refuse independently of it.
//!     It is asked under the model's own provenance, so rule 3 never fires on the curve.
//!   - dots: the stored two-atom samples of the pair, (r, E_s − E_ref(a) − E_ref(b)), any state.
//!
//!   **The curve is a two-body slice of a many-body model, not a pair potential.** The model is a
//!   sum of per-atom terms whose descriptors couple every neighbour (the dimer itself carries
//!   body-3 self-terms), so E_m(r) − E_ref(a) − E_ref(b) is the model's energy of an *isolated*
//!   a–b pair and nothing else: it is not the model's prediction for an a–b bond inside a
//!   molecule, where the model has no per-pair energy at all. The band, the lane and the
//!   verdicts are likewise for the isolated pair (spec §9.4, plan review S3): an isolated dimer
//!   is class Pair, so inside a molecule the model may answer this pair where the curve is
//!   refused, and refuse it where the curve answers.
//! - **Coverage**: per species pair, the distances r < r_cut between every two atoms of each
//!   **fitted** sample (any atom count), in bins of 0.1 Bohr from 0 (spec §3.3.5, N12).
//!
//! What is not here: the rmse per refit (the worker's fit history, spec §4.2, A7) and the burst
//! temperatures (the app's per-session accumulator of `TeachEvent::burst_t`, S6) — app-side.
//!
//! # Sample states
//!
//! `Fitted` means "in the fit that stands": below the `accumulated` prefix *and* not excluded
//! when that fit was installed (`fit_members`, snapshotted at every installed fit, restored with
//! a restored state) *and* not excluded now. A quarantined sample a fresh teacher answer released
//! since (`admit`, `KeptReleased`) reads `Unfitted` until the next refit takes it in, and the
//! samples an incremental solve error put into the accumulator without a model read `Unfitted`
//! too (Task 4 review, the released-sample ruling). A sample excluded now reads its exclusion
//! (quarantined, superseded, box-excluded).
//!
//! # Cost and the curve cache
//!
//! Per build: one `predict(…, false)` per stored sample (parity), and the curves. Everything
//! expensive in a curve — the energy, σ, the force ratio and the verdict past the gate up to the
//! fence (non-finite, envelope, 5d, 4b, 6/7), K = 96 `predict(…, true)` per pair at O(Q′²) per row,
//! 3N + 1 = 7 rows — is a function of the installed model and the policy only, so it is cached
//! per model generation (`model_gen`, bumped wherever either changes; Task 4 review, the cost
//! ruling). Each build recomputes the two ends of the verdict order, which can move between
//! refits: the gate (rule 4 flips on a failed audit) once per pair, and the fence (a release or a
//! vote) on the cached answered points while samples are quarantined. Gate first and fence last
//! in the order, so the composed verdict is exactly `would_answer`'s. Measured:
//! `training_views_build_time_is_reported` (the task report). Nothing here runs inside
//! `evaluate`.
use super::trust::max_nan;
use super::*;
use sim::activity::{
    PairCurve, PairHistogram, ParityPoint, SampleKind, SampleState, TeachRule, TrainingViews, CURVE_POINTS, CURVE_R_MIN, FORCE_PARITY_CAP,
};
use surrogate::DescribeError;

/// The fallback policy's verdict at one geometry: `Ok` where the model would answer, else the
/// rule that refuses and the atom it names (`TeachEvent::{rule, atom}` of the step `evaluate`
/// would hand to the teacher).
pub type Verdict = Result<(), (TeachRule, Option<usize>)>;

/// Pair-distance histogram bin (Bohr).
const BIN: f64 = 0.1;

/// The pair curves of one model generation: everything but the gate and the fence (module doc,
/// "Cost and the curve cache").
pub(super) struct CurveCache {
    model_gen: u64,
    curves: Vec<CachedCurve>,
    skipped: usize,
}

/// One pair's cached curve: `middle` is the verdict past the gate without the fence.
struct CachedCurve {
    z: (u8, u8),
    r: Vec<f64>,
    energy: Vec<f64>,
    sigma: Vec<f64>,
    force_ratio: Vec<f64>,
    middle: Vec<Option<TeachRule>>,
}

/// The curve's dimer at distance `x` (Bohr).
fn dimer_at(x: f64) -> [[f64; 3]; 2] {
    [[-x / 2.0, 0.0, 0.0], [x / 2.0, 0.0, 0.0]]
}

impl SurrogateKs {
    /// The fallback policy's verdict at (`pos`, `z`) under `prov`: the rules `evaluate` walks, in
    /// its priority order, with a fresh variance check —
    ///
    /// 1. the gate (rules 1–4, as `gate`: no model, other settings, new species, untrusted);
    /// 2. the model's prediction with its variance; one that is not finite refuses ahead of every
    ///    other rule (`NonFiniteAnswer`, activity Task 3b: `evaluate` checks it before the rest);
    /// 3. the envelope (5a, 5b, 5c, 5e);
    /// 4. the novelty (5d);
    /// 5. the environment class (4b);
    /// 6. the variance (rules 6, then 7);
    /// 7. the quarantine fence.
    ///
    /// `evaluate` computes 3–6 and takes the first refusal (`envelope.or(novel).or(class)
    /// .or(variance)`, then the fence); returning at the first refusal in the same order gives the
    /// same rule and atom. Read-only by type (plan review S1): no variance cadence, audit counter,
    /// streak, store, quarantine, RNG or cache moves; the teacher is never called. (The one
    /// interior write on the path is `Model::representatives`, a `OnceLock` `predict` fills
    /// lazily with a pure function of the basis — `evaluate`'s own prediction fills it with the
    /// same.) The fence reads the descriptor cache (`ensure_desc_cache`) and describes any held
    /// sample it does not cover.
    ///
    /// Not part of this verdict: rule 0 (the cell) and the empty scene, draft mode (it answers
    /// past rules 4–7 and the fence; this is the policy's verdict, draft or not), audits, the
    /// relax verification (`force_teacher_next`) and the cadence's reuse of an accepting variance
    /// between fresh checks. Every installed model has a posterior (`install_sidecar` refuses a
    /// file without one, Task 4 review I4), so 5d and 6–7 are always judged; the branch without
    /// stds is unreachable and kept only so a hand-built model cannot panic here.
    pub fn would_answer(&self, pos: &[[f64; 3]], z: &[u8], prov: &Provenance) -> Verdict {
        self.gate(z, prov).map_err(|g| (g.reason.rule, g.reason.atom))?;
        let model = self.model.as_ref().expect("the gate passed: a model exists");
        let pred = model.predict(pos, z, true).map_err(|e| match e {
            // unreachable past the gate (it refuses a species the basis lacks, rule 2), and
            // what rule 2 names; `evaluate` returns an error there with no rule
            DescribeError::UnknownSpecies(zu) => (TeachRule::NewSpecies(zu), None),
        })?;
        self.verdict_past_gate(model, pos, z, &pred)
    }

    /// [`would_answer`](Self::would_answer) past the gate, on the caller's prediction at
    /// (`pos`, `z`) — `model.predict(pos, z, true)` (the curve's band shares it, plan review S2).
    pub(super) fn verdict_past_gate(&self, model: &Model, pos: &[[f64; 3]], z: &[u8], pred: &surrogate::Prediction) -> Verdict {
        self.verdict_middle(model, pos, z, pred)?;
        self.fence_verdict(pos, z)
    }

    /// The verdict between the gate and the fence: a function of the model and the policy only
    /// (what the curve cache keeps).
    fn verdict_middle(&self, model: &Model, pos: &[[f64; 3]], z: &[u8], pred: &surrogate::Prediction) -> Verdict {
        let code = |r: Refusal| (r.rule, r.atom);
        if let Some(r) = non_finite_answer(pred) {
            return Err(code(r));
        }
        if let (_, Some(r)) = self.envelope_outside(model, pos, z) {
            return Err(code(r));
        }
        let posterior = match (pred.energy_std, pred.force_std_epistemic.as_ref()) {
            (Some(e), Some(f)) => Some((e, f)),
            _ => None,
        };
        // 5d beside the variance, as `evaluate` (the same threshold: ν > 1)
        if posterior.is_some() {
            if let Some((_, at)) = self.null_space.as_ref().and_then(|ns| ns.novelty(pos, z).ok()).filter(|(nu, _)| *nu > 1.0) {
                return Err((TeachRule::NewEnvironment, at));
            }
        }
        let rc2 = model.basis.r_cut * model.basis.r_cut;
        if let Some((r, _)) = self.class_untrusted(&neighbour_counts(pos, rc2)) {
            return Err(code(r));
        }
        if let Some((e, f)) = posterior {
            let v = self.variance_check(model, pred, e, f);
            if !v.accept {
                return Err(code(v.reason.expect("a refusing variance verdict names its rule")));
            }
        }
        Ok(())
    }

    /// The last rule of the order: the quarantine fence.
    fn fence_verdict(&self, pos: &[[f64; 3]], z: &[u8]) -> Verdict {
        if !self.near_quarantined_cached(pos, z).is_empty() {
            return Err((TeachRule::NearQuarantine, None));
        }
        Ok(())
    }

    /// The model generation (`model_gen`): changes at every installed, restored or forgotten
    /// model and every policy change, an untrusted → untrusted refit included. A readout.
    pub fn model_generation(&self) -> u64 {
        self.model_gen
    }

    /// The training views (module doc); `None` without a fitted model and a store. Rebuilds the
    /// curve cache when the model generation moved, fills the fence's descriptor cache while
    /// samples are quarantined (review m4: the fence works without it otherwise), then reads only.
    /// Both caches decide nothing.
    pub fn training_views(&mut self) -> Option<TrainingViews> {
        let t0 = Instant::now();
        if self.quarantined() > 0 {
            self.ensure_desc_cache();
        }
        if self.curve_cache.as_ref().is_none_or(|c| c.model_gen != self.model_gen) {
            self.curve_cache = self.build_curve_cache();
        }
        self.views_from_cache(t0)
    }

    /// The expensive part of the curves for the current model generation (module doc).
    fn build_curve_cache(&self) -> Option<CurveCache> {
        let model = self.model.as_ref()?;
        let r_cut = model.basis.r_cut;
        let floor2 = dimer_floor(model);
        let species = &model.basis.species;
        let (mut curves, mut skipped) = (Vec::new(), 0usize);
        for (ia, &za) in species.iter().enumerate() {
            'pair: for &zb in &species[ia..] {
                let zs = [za, zb];
                let mut c = CachedCurve {
                    z: (za.min(zb), za.max(zb)),
                    r: Vec::with_capacity(CURVE_POINTS),
                    energy: Vec::with_capacity(CURVE_POINTS),
                    sigma: Vec::with_capacity(CURVE_POINTS),
                    force_ratio: Vec::with_capacity(CURVE_POINTS),
                    middle: Vec::with_capacity(CURVE_POINTS),
                };
                let Some(e_ab) = fit_ref(model, &zs) else {
                    skipped += 1;
                    continue 'pair;
                };
                for k in 0..CURVE_POINTS {
                    let x = CURVE_R_MIN + (r_cut * 0.999 - CURVE_R_MIN) * k as f64 / (CURVE_POINTS - 1) as f64;
                    let pos = dimer_at(x);
                    // one prediction per point serves the band, the lane and the verdict (plan
                    // review S2); a failed one skips this curve, counted (S18)
                    let Ok(p) = model.predict(&pos, &zs, true) else {
                        skipped += 1;
                        continue 'pair;
                    };
                    c.energy.push(p.energy - e_ab);
                    if let (Some(e), Some(f)) = (p.energy_std, p.force_std_epistemic.as_ref()) {
                        // `max_nan`: a NaN variance stays NaN (Task 3b), never a zero band
                        c.sigma.push(max_nan(e * e - floor2 * floor2, 0.0).sqrt());
                        // rule 6's ratio: max over the atoms of σ_F,epi/(√3 · bound) (plan review M5)
                        c.force_ratio.push(self.variance_check(model, &p, e, f).atoms.iter().copied().fold(0.0, max_nan));
                    }
                    c.middle.push(self.verdict_middle(model, &pos, &zs, &p).err().map(|e| e.0));
                    c.r.push(x);
                }
                curves.push(c);
            }
        }
        Some(CurveCache { model_gen: self.model_gen, curves, skipped })
    }

    fn views_from_cache(&self, t0: Instant) -> Option<TrainingViews> {
        let (model, store, cache) = (self.model.as_ref()?, self.store.as_ref()?, self.curve_cache.as_ref()?);
        let (fit_n, fit_q) = (&self.fit_members.0, &self.fit_members.1);
        let state = |i: usize| -> SampleState {
            match self.quarantine.get(i).copied().unwrap_or(Q_NONE) {
                Q_HELD => SampleState::Quarantined,
                Q_SUPERSEDED => SampleState::Superseded,
                Q_BOX => SampleState::BoxExcluded,
                Q_DRAFT => SampleState::HeldOut,
                Q_NONE if i < *fit_n && fit_q.get(i).copied().unwrap_or(Q_NONE) == Q_NONE => SampleState::Fitted,
                Q_NONE => SampleState::Unfitted,
                // a code the sidecar does not write (it validates 0–3): excluded, as `excluded`
                // reads it, and not fencing — never shown as fitted (review m2)
                _ => SampleState::Superseded,
            }
        };
        // audits are tagged Fallback in the store until sidecar v4 (spec A6)
        let kind = |t: SampleTag| match t {
            SampleTag::Fallback | SampleTag::Audit | SampleTag::DraftAudit => SampleKind::Fallback,
            SampleTag::LiveStep => SampleKind::Live,
            SampleTag::Perturbation | SampleTag::HotBurst | SampleTag::Scan | SampleTag::Reference => SampleKind::Pass,
        };
        let finite = |e: f64, f: &[[f64; 3]]| e.is_finite() && f.iter().flatten().all(|x| x.is_finite());
        let r_cut = model.basis.r_cut;
        let n_bins = ((r_cut / BIN).ceil() as usize).max(1);
        let mut parity_energy = Vec::with_capacity(store.len());
        let mut parity_omitted = 0usize;
        let mut force_atoms: Vec<[[f32; 2]; 3]> = Vec::new();
        let mut dots: BTreeMap<(u8, u8), Vec<(f64, f64, SampleState)>> = BTreeMap::new();
        let mut hist: BTreeMap<(u8, u8), Vec<u32>> = BTreeMap::new();
        for (i, smp) in store.samples.iter().enumerate() {
            // one failed or non-finite prediction (or label) omits that sample, counted, never the
            // views (plan review S18, review m5)
            let (Some(eref), Ok(pred)) = (fit_ref(model, &smp.z), model.predict(&smp.pos, &smp.z, false)) else {
                parity_omitted += 1;
                continue;
            };
            if !finite(smp.energy, &smp.forces) || !finite(pred.energy, &pred.forces) {
                parity_omitted += 1;
                continue;
            }
            let n = smp.pos.len() as f64;
            let st = state(i);
            parity_energy.push(ParityPoint { dft: (smp.energy - eref) / n, model: (pred.energy - eref) / n, kind: kind(smp.tag), state: st, index: i });
            if st == SampleState::Fitted {
                for (ft, fm) in smp.forces.iter().zip(&pred.forces) {
                    force_atoms.push([0, 1, 2].map(|a| [ft[a] as f32, fm[a] as f32]));
                }
            }
            for a in 0..smp.pos.len() {
                for b in a + 1..smp.pos.len() {
                    let d = dist(smp.pos[a], smp.pos[b]);
                    let key = (smp.z[a].min(smp.z[b]), smp.z[a].max(smp.z[b]));
                    if smp.pos.len() == 2 {
                        dots.entry(key).or_default().push((d, smp.energy - eref, st));
                    }
                    if st == SampleState::Fitted && d < r_cut {
                        let h = hist.entry(key).or_insert_with(|| vec![0; n_bins]);
                        h[((d / BIN) as usize).min(n_bins - 1)] += 1;
                    }
                }
            }
        }
        let force_points_total = 3 * force_atoms.len();
        let parity_force = thin_atoms(&force_atoms, FORCE_PARITY_CAP);
        // the curves: the cached middle of the verdict, the gate and the fence fresh (module doc)
        let prov = &model.provenance;
        let fence = self.quarantined() > 0;
        let curves = cache
            .curves
            .iter()
            .map(|c| {
                // rules 1–4 depend on the species and the settings only: one gate per pair
                let gate = self.gate(&[c.z.0, c.z.1], prov).err().map(|g| g.reason.rule);
                let refused = c
                    .r
                    .iter()
                    .zip(&c.middle)
                    .map(|(&x, middle)| match (gate, middle) {
                        (Some(rule), _) => Some(rule),
                        (None, Some(rule)) => Some(*rule),
                        (None, None) if fence => self.fence_verdict(&dimer_at(x), &[c.z.0, c.z.1]).err().map(|e| e.0),
                        (None, None) => None,
                    })
                    .collect();
                PairCurve {
                    z: c.z,
                    r: c.r.clone(),
                    energy: c.energy.clone(),
                    sigma: c.sigma.clone(),
                    force_ratio: c.force_ratio.clone(),
                    refused,
                    // the key is (min, max) whatever order the basis lists its species (review m1)
                    dots: dots.remove(&c.z).unwrap_or_default(),
                    r_cut,
                }
            })
            .collect();
        let pairs = hist.into_iter().map(|(z, counts)| PairHistogram { z, bin: BIN, counts }).collect();
        Some(TrainingViews {
            epoch: self.epoch,
            model_gen: self.model_gen,
            samples: store.len(),
            parity_energy,
            parity_force,
            force_points_total,
            parity_omitted,
            curves,
            curves_skipped: cache.skipped,
            pairs,
            dimer_floor: dimer_floor(model),
            energy_tier: self.policy.energy_std_per_atom,
            build_ms: t0.elapsed().as_secs_f64() * 1e3,
        })
    }
}

/// Σ E_ref of a species list, from the fit's own references (plan review S4); `None` for a
/// species the basis lacks.
fn fit_ref(model: &Model, zs: &[u8]) -> Option<f64> {
    let idx = model.basis.species_indices(zs).ok()?;
    Some(idx.iter().map(|&i| model.ref_energy[i]).sum())
}

/// The floor `predict` adds in quadrature to a dimer's energy std (N = 2): 2 s_E for a per-atom
/// model, else s_E — the fit's residual scatter (module doc).
fn dimer_floor(model: &Model) -> f64 {
    if model.energy_per_atom { 2.0 * model.sigma_e } else { model.sigma_e }
}

/// Deterministic thinning of the force parity to at most `cap` components, by whole atoms (plan
/// review S5): a component stride that is a multiple of 3 would keep one Cartesian component
/// only. Keeps every `stride`-th atom, stride = ⌈atoms / ⌊cap/3⌋⌉, so at most ⌊cap/3⌋ atoms —
/// ≤ cap components — and x, y and z equally often.
pub(super) fn thin_atoms(atoms: &[[[f32; 2]; 3]], cap: usize) -> Vec<[f32; 2]> {
    let stride = atoms.len().div_ceil((cap / 3).max(1)).max(1);
    atoms.iter().step_by(stride).flat_map(|a| a.iter().copied()).collect()
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}
