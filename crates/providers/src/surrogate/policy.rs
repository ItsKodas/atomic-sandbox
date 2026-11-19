//! The fallback policy's rules (the `surrogate` module doc, "The fallback policy, as a decision
//! list"): [`FallbackPolicy`] and its bounds, the box rule [`box_affected_face_density`], and the
//! checks `evaluate` walks — the gate (rules 1–4), the envelope (rule 5), the variance (rules 6–7).
//! Split out of `surrogate_ks.rs` in M6 Task 1; the narrative stays in the parent module doc.
//!
//! Every refusal is a [`Refusal`]: the note text it always carried (byte-identical), the rule's
//! code (`sim::activity::TeachRule`) and the atom it names (Activity Task 2). The code is a label
//! on a decision already made — no predicate, order or constant reads it.

use super::*;
use sim::activity::{EnvClass, TeachRule};
use super::stray::Split;
use super::trust::max_nan;
use surrogate::envelope::COORDINATION_RATIO;

/// A refusal: the text the notes have always carried (byte-identical), the rule's code and the
/// atom it names (activity spec A3).
#[derive(Clone, Debug)]
pub(super) struct Refusal {
    pub(super) text: String,
    pub(super) rule: TeachRule,
    pub(super) atom: Option<usize>,
}

impl Refusal {
    pub(super) fn new(text: impl Into<String>, rule: TeachRule, atom: Option<usize>) -> Refusal {
        Refusal { text: text.into(), rule, atom }
    }
}

/// The note text (what every `format!` of a reason printed before the codes).
impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.text)
    }
}

/// The class rule's code for class index `c` (`ENV_CLASSES` order).
pub(super) fn env_class_code(c: usize) -> EnvClass {
    if c == 0 { EnvClass::Pair } else { EnvClass::ManyBody }
}

/// The relative force criterion's bound never exceeds this (Ha/Bohr).
pub const FORCE_STD_BOUND_MAX: f64 = 1e-2;

/// Rule 5e: an atom whose extrapolation grade γ (`surrogate::grade`) exceeds this falls back.
/// Derivation and choice: the `surrogate` module doc, rule 5e; `references.md` policy table.
pub const GRADE_MAX: f64 = 2.0;

/// The box rule at h = 0.25 Bohr (e/Bohr³): a teacher answer whose reported face density
/// (`EvalStats::face_density`: max ρ on the outermost grid plane, one spacing h inside the
/// Dirichlet wall) exceeds [`box_affected_face_density`]`(h)` = this × (h / 0.25 Bohr)² is not
/// trained on and not judged as an audit (`surrogate` module doc, "Box-affected answers"). A calibration
/// at h = 0.25 for hydrogen, not a universal scale — see below.
///
/// Why this scale (Task 9; derivation by the Task 9 review, ruling S17). A hard wall against an
/// exponential tail ψ ∝ e^{−κx} (κ = √(−2ε_HOMO)) raises the energy by ΔE = κ ρ₀ per unit area,
/// ρ₀ the *unconfined* density at the wall; for a 1s-like tail at distance d the patch has an
/// effective area π d/κ, so ΔE ≈ π d ρ₀. The solver reports the plane one h inside the wall,
/// where the confined orbital is already pinched: ρ_reported ≈ 4κ²h² ρ₀. Hence
/// ΔE ≈ π d ρ_reported / (4κ²h²) ≈ 50–80 Ha per e/Bohr³ for H₂ (κ ≈ 0.85) at h = 0.25 and
/// d = 2.6–4 Bohr — the 40–55 measured (`diag_confinement_vs_face_density`, 41³ and 49³). So the
/// same face density means 4× the confinement at h = 0.125, so a quarter of the density is
/// tolerated there: the threshold scales as h² (the κ dependence — less confinement per unit
/// face density for more tightly bound species — is not modelled; an energy-relative rule with κ from the HOMO eigenvalue is a handoff item).
/// 2e-5 at h = 0.25 is ≈ 1e-3 Ha of position-dependent confinement — half the H₂ audit
/// threshold (N × 1e-3 Ha): acceptable, not tight (the review prefers 1e-5, the solver's own
/// warning level; 2e-5 keeps the 49³ hot-burst solves that warn at 1.0–1.4e-5 trainable). Fix
/// round 4 had 1e-3 (≈ 5e-2 Ha), which let the confinement through (Task 9 item 1).
pub const BOX_AFFECTED_FACE_DENSITY: f64 = 2e-5;

/// The box rule at grid spacing `h` (Bohr): [`BOX_AFFECTED_FACE_DENSITY`] × (h / 0.25)²
/// (ruling S17, direction corrected in Task 9 fix round 2; the derivation is on the constant):
/// the confinement per unit of reported face density grows as 1/h², so the tolerated density
/// shrinks as h².
pub fn box_affected_face_density(h: f64) -> f64 {
    BOX_AFFECTED_FACE_DENSITY * (h / 0.25).powi(2)
}

pub(super) const SQRT_3: f64 = 1.732_050_807_568_877_2;

/// Per atom: the number of other atoms within r_cut (r² < `rc2`), the variance cadence's
/// coordination test (module doc, "Variance cadence").
pub(super) fn neighbour_counts(pos: &[[f64; 3]], rc2: f64) -> Vec<usize> {
    let mut c = vec![0; pos.len()];
    for i in 0..pos.len() {
        for j in i + 1..pos.len() {
            if (0..3).map(|a| (pos[i][a] - pos[j][a]).powi(2)).sum::<f64>() < rc2 {
                c[i] += 1;
                c[j] += 1;
            }
        }
    }
    c
}

/// Per atom (Z, the distinct species of its neighbours within r_cut, ascending) — with the atom's
/// class, its [`surrogate::EnvKey`] (the calibrated class verdict's key, M8a user-test fix 2). Z is
/// a key only (rule 1).
pub(super) fn neighbour_species(pos: &[[f64; 3]], z: &[u8], rc2: f64) -> Vec<(u8, Vec<u8>)> {
    (0..pos.len())
        .map(|i| {
            let mut nb: Vec<u8> = (0..pos.len()).filter(|&j| j != i && (0..3).map(|a| (pos[i][a] - pos[j][a]).powi(2)).sum::<f64>() < rc2).map(|j| z[j]).collect();
            nb.sort_unstable();
            nb.dedup();
            (z[i], nb)
        })
        .collect()
}

/// Per atom its [`surrogate::EnvKey`] (class, Z, neighbour species), `None` for an isolated atom.
pub(super) fn env_keys(pos: &[[f64; 3]], z: &[u8], rc2: f64) -> Vec<Option<surrogate::EnvKey>> {
    neighbour_counts(pos, rc2)
        .into_iter()
        .zip(neighbour_species(pos, z, rc2))
        .map(|(k, (z, neighbours))| atom_class(k).map(|c| surrogate::EnvKey { class: c as u8 + 1, z, neighbours }))
        .collect()
}

/// The force-label noise prior of a pseudopotential scene, Ha/Bohr per component (M7 Task 9; Task 8b review
/// ruling (b), Task 8a review): the teacher's own path scatter on the GPU at the sampling tolerance — measured
/// 4.7–4.8e-5 on lone pseudo-O and on OH, up to ≈ 7e-5 on warm teacher chains — which no model can represent.
/// It floors the fit's force noise s_F (`FitOptions::sigma_f_min`: the posterior is never more confident about
/// forces than the labels can support) and, tripled, the tier a force audit judges against
/// ([`audit_force_floor`]). The default tiers (`FallbackPolicy::force_std_abs` = 2e-3) are 40× above it, so the
/// floors are inert there; they bind only on a tightened policy.
pub const PSEUDO_FORCE_NOISE: f64 = 5e-5;

/// The label-noise prior of the surface `ionic` (`Provenance::ionic`): [`PSEUDO_FORCE_NOISE`] for a pseudised
/// surface, 0 for the all-electron one (whose labels keep the evidence-estimated noise, bitwise as before) — the legacy
/// `"none"` and the Fermi–Dirac `none/fd5e-3/pin1e-5` alike (`crate::pseudo::is_all_electron_tag`, spec 2026-11-13 §6.2).
pub(super) fn force_noise_prior(ionic: &str) -> f64 {
    if crate::pseudo::is_all_electron_tag(ionic) { 0.0 } else { PSEUDO_FORCE_NOISE }
}

/// The least tier an audit's force check may use on the surface `ionic`: 3σ of ONE label's noise, 0 on the
/// all-electron surface. Deliberately the stricter 3σ, not 3√2σ (which would account for the model's labels and the
/// audit's both being noisy): rule 6 forbids loosening a tolerance, and the floor is inert at the default tiers. The
/// price is a false-fail rate for a tightened policy: with the warm-chain scatter of 7e-5 per component, 1.5e-4 is
/// 2.1σ of the per-atom RMS-of-3 statistic, P(χ²₃ > 13.8) ≈ 3e-3 per atom (≈ 3 % per audit of a 10-atom scene).
pub(super) fn audit_force_floor(ionic: &str) -> f64 {
    3.0 * force_noise_prior(ionic)
}

/// The worst atom's force residual ‖F_model − F_teacher‖/√3 against its bound (the variance
/// rule's, at the teacher's force, never below `floor`): (residual, bound) of the atom with the
/// largest ratio.
pub(super) fn worst_force_residual(p: &FallbackPolicy, model: &[[f64; 3]], teacher: &[[f64; 3]], floor: f64) -> Option<(f64, f64)> {
    let residuals = model.iter().zip(teacher).map(|(fm, ft)| {
        let (d, b) = p.force_residual(fm, ft);
        (d, b.max(floor))
    });
    // a NaN residual is the worst (`total_cmp` ranks a negative NaN below everything; activity
    // Task 3b), else the largest ratio as before
    residuals.clone().find(|(d, b)| (d / b).is_nan()).or_else(|| residuals.max_by(|a, b| (a.0 / a.1).total_cmp(&(b.0 / b.1))))
}

/// A prediction that is not finite (its energy, a force, or a std when evaluated): the refusal,
/// else `None`. Never an answer, draft mode included (activity Task 3b, `TeachRule::NonFiniteAnswer`).
pub(super) fn non_finite_answer(pred: &surrogate::Prediction) -> Option<Refusal> {
    let bad_forces = pred.forces.iter().flatten().filter(|x| !x.is_finite()).count();
    let stds = pred.energy_std.iter().chain(pred.force_std.iter().flatten()).chain(pred.force_std_epistemic.iter().flatten());
    let bad_std = stds.clone().any(|x| !x.is_finite());
    (!pred.energy.is_finite() || bad_forces > 0 || bad_std).then(|| {
        Refusal::new(
            format!(
                "surrogate: the model's answer is not finite here (energy {:.3e}, {bad_forces} non-finite force components{}) — never an answer, draft mode included",
                pred.energy,
                if bad_std { ", a non-finite std" } else { "" }
            ),
            TeachRule::NonFiniteAnswer,
            None,
        )
    })
}

/// Whether a refusal is a fact about the geometry rather than about the fit's global state
/// (round 1 of the 2026-10-18 trust review, MUST-1): outside the training envelope (5a–5c), an
/// unspanned direction (5d, and 5e at γ = ∞), beyond the grade (5e), a non-finite answer, a
/// quarantine fence, or an atom near a misfit (`NearMisfit`, spec 2026-11-13 §4.3: a fact about
/// the geometry under the fit that stands, as the quarantine fence is). Rule 4 (untrusted), 4b
/// (class), 5e without a grade (MaxVol unconverged: global) and 6–7 (uncertainty) lift with more
/// data or a refit, so a draft-mode audit refused only by those is an ordinary audit.
pub(super) fn geometry_specific(rule: TeachRule) -> bool {
    matches!(
        rule,
        TeachRule::IsolatedAtom
            | TeachRule::Coordination
            | TeachRule::Spread
            | TeachRule::NewEnvironment
            | TeachRule::UnspannedAtom
            | TeachRule::Grade
            | TeachRule::NonFiniteAnswer
            | TeachRule::NearQuarantine
            | TeachRule::NearMisfit
    )
}

/// Rule 5d's refusal at novelty `nu` (> 1), naming the atom whose force rows reach furthest
/// (`None`: the energy row) — the text `evaluate` has always carried.
pub(super) fn novelty_refusal(nu: f64, at: Option<usize>) -> Refusal {
    Refusal::new(
        format!(
            "surrogate: new environment (feature never seen in training) — {} {nu:.1e}× farther into directions no training row spans than rounding allows",
            at.map_or("the energy row reaches".to_string(), |j| format!("atom {j}'s force rows reach"))
        ),
        TeachRule::NewEnvironment,
        at,
    )
}

/// Rule 5a: the atom with descriptor `b` has no neighbour the model can see — every coordination
/// feature C_t = Σ_{j∈t} f_c(r_ij) is exactly zero (no neighbour inside r_cut; a coincident one
/// counts, f_c(0) = 1). O(S) per atom from the descriptor already computed. The predicate is the
/// `surrogate` crate's [`surrogate::envelope::isolated_atom`], which also keeps such an atom out of
/// the fit's envelope, so rule 5a and the envelope cannot disagree on what is isolated.
pub(super) fn lone_atom(basis: &Basis, b: &[f64]) -> bool {
    surrogate::envelope::isolated_atom(basis, b)
}

/// When the model may answer (`surrogate` module doc, "The fallback policy"). Defaults are the spec's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FallbackPolicy {
    /// Absolute bound on the per-component force std, Ha/Bohr (2e-3).
    pub force_std_abs: f64,
    /// Relative bound, fraction of |F_j| (0.10); the resulting bound is capped at
    /// [`FORCE_STD_BOUND_MAX`].
    pub force_std_rel: f64,
    /// Bound on the epistemic energy std per atom, Ha (1e-3).
    pub energy_std_per_atom: f64,
    /// Out-of-distribution envelope: training spreads outside the per-feature training range
    /// (3.0).
    pub max_descriptor_distance: f64,
    /// The O(Q′²) variance (rules 6–7) is evaluated afresh at least on every k-th consultation,
    /// and always on the first after an edit, a refit, a policy change or a rejection (module
    /// doc); 4 (Task 9: measured, `references.md` speed table). In between, the reported
    /// `EvalStats::uncertainty` is the reused verdict's number (≤ k − 1 consultations old).
    pub variance_cadence: usize,
    /// Audit (ruling S7): every `audit_every`-th step the model would have answered, the
    /// teacher is called instead and the gap recorded; 0 disables (2000, ruling S14 interim;
    /// S7 had 200). With a background auditor that the model outruns (full speed) the effective
    /// cadence is one audit per teacher solve, whatever this says (`surrogate` module doc, "Audit").
    pub audit_every: usize,
    /// Draft mode (`surrogate` module doc): the model answers every step it can be asked at all (rules 1–3
    /// pass), extrapolating beyond rules 4–7 and the quarantine fence; audits only in the
    /// background. Off by default.
    pub never_fallback: bool,
    /// Rule 5e: the extrapolation grade's limit ([`GRADE_MAX`]).
    pub max_grade: f64,
}

impl Default for FallbackPolicy {
    fn default() -> FallbackPolicy {
        FallbackPolicy {
            force_std_abs: 2e-3,
            force_std_rel: 0.10,
            energy_std_per_atom: 1e-3,
            max_descriptor_distance: 3.0,
            variance_cadence: 4,
            audit_every: 2000,
            never_fallback: false,
            max_grade: GRADE_MAX,
        }
    }
}

impl FallbackPolicy {
    /// The per-component force bound for an atom with force magnitude `fmag`:
    /// min(max(abs, rel·|F|), [`FORCE_STD_BOUND_MAX`]).
    pub fn force_bound(&self, fmag: f64) -> f64 {
        self.force_std_abs.max(self.force_std_rel * fmag).min(FORCE_STD_BOUND_MAX)
    }

    /// One atom's force residual against a reference force: (‖f − f_ref‖/√3, the per-component
    /// equivalent; [`force_bound`](Self::force_bound)(|f_ref|)). The one definition behind the
    /// training-residual trust rule, the worst-offender ranking and the agreement test.
    pub fn force_residual(&self, f: &[f64; 3], f_ref: &[f64; 3]) -> (f64, f64) {
        let d = ((f[0] - f_ref[0]).powi(2) + (f[1] - f_ref[1]).powi(2) + (f[2] - f_ref[2]).powi(2)).sqrt() / SQRT_3;
        (d, self.force_bound((f_ref[0] * f_ref[0] + f_ref[1] * f_ref[1] + f_ref[2] * f_ref[2]).sqrt()))
    }
}

/// The strict policy's verdict per atom ([`SurrogateKs::atom_verdicts`]; M8a Task 6): what M8b's
/// per-atom freezing and the transfer test read where `would_answer` gives one verdict for the
/// whole geometry.
#[derive(Clone, Debug, PartialEq)]
pub struct AtomVerdicts {
    /// A rule that refuses the whole geometry: the gate (rules 1–4: no model, an unknown species,
    /// other settings, an untrusted fit), a non-finite answer, 5e without a grade, 5d on the energy
    /// row, rule 7 (the energy std is the whole geometry's), the fences. `None`: none does.
    pub geometry: Option<TeachRule>,
    /// Per atom, the first per-atom rule that refuses it in `evaluate`'s order — 5a, 5b, 5c, 5e
    /// (γ = ∞, then γ > γ_max), 5d on its force rows, 4b on its environment key, 6 on its force
    /// std — or `None`: the atom is trusted. Every `None` when the gate refused.
    pub atoms: Vec<Option<TeachRule>>,
}

impl AtomVerdicts {
    /// The geometry is answered by the model: no geometry rule and every atom trusted (exactly
    /// `would_answer(..).is_ok()`).
    pub fn answers(&self) -> bool {
        self.geometry.is_none() && self.atoms.iter().all(Option::is_none)
    }

    /// The atom is trusted and the geometry not refused as a whole.
    pub fn trusted(&self, i: usize) -> bool {
        self.geometry.is_none() && self.atoms[i].is_none()
    }
}

/// The variance verdict (rules 6–7) of the last consultation that evaluated it.
#[derive(Clone)]
pub(super) struct VarianceVerdict {
    pub(super) accept: bool,
    /// max_j of the epistemic per-component force std (rule 6's number).
    pub(super) uncertainty: f64,
    pub(super) reason: Option<Refusal>,
    /// Per atom: the epistemic force_std/√3 ÷ its bound.
    pub(super) atoms: Vec<f64>,
    /// The rejection is rule 5d (an unspanned direction), not the variance.
    pub(super) novel: bool,
}

/// Why the model is not consulted (rules 1–4), whether the teacher's answer is kept, and
/// whether that sample is subject to thinning (`surrogate` module doc, "Trust recovery").
pub(super) struct Gate {
    pub(super) reason: Refusal,
    pub(super) keep_sample: bool,
    pub(super) thin: bool,
    /// A copy of a stored sample is never kept (`is_label_copy`; round 1, MUST-2).
    pub(super) dedup: bool,
    /// Rule 4 (an untrusted fit): the only gate draft mode answers past.
    pub(super) untrusted: bool,
}

impl SurrogateKs {
    pub fn policy(&self) -> &FallbackPolicy {
        &self.policy
    }

    /// Replaces the policy; the variance cadence restarts and the training residuals are
    /// re-judged against the new thresholds.
    pub fn set_policy(&mut self, p: FallbackPolicy) {
        // a stray-atom region was cleared under the check's tolerances (its energy and force tests) and the thinning
        // radius: when one of them changes, every region is checked again; a refusal stays (final review minor 5,
        // controller ruling 13; `stray` module doc, "The cleared region")
        let judged = |q: &FallbackPolicy| (q.energy_std_per_atom, q.force_std_abs, q.force_std_rel, q.max_descriptor_distance);
        if judged(&p) != judged(&self.policy) {
            self.stray_book.clear_regions();
        }
        self.policy = p;
        self.model_gen += 1; // the views' cache key: the verdicts read the policy
        self.consults = 0;
        self.last_variance = None;
        (self.training_residuals, self.misfit) = self.compute_residuals().unzip();
        self.refresh_class_loo();
        self.commit_class_memory();
        // the re-judged verdict and M: a failed recovery's note says which (spec 2026-11-13 §4.3)
        self.settle_recovery_note();
    }

    /// Rules 1–4: whether the model may be consulted for this scene.
    pub(super) fn gate(&self, z: &[u8], prov: &Provenance) -> Result<(), Gate> {
        let Some(model) = &self.model else {
            return Err(Gate { reason: Refusal::new("surrogate: no model yet", TeachRule::NoModel, None), keep_sample: true, thin: false, dedup: false, untrusted: false });
        };
        if !self.same_surface(&model.provenance, prov) {
            // in collection mode the collection's own reason (a charged scene keeps a per-scene
            // model; `shared` module doc), else the two provenances
            let text = match self.collection_refuses(prov) {
                Some(why) => format!("surrogate: {why} — sample not kept"),
                None => format!("surrogate: scene ({}) differs from the model's ({}) — sample not kept", prov.describe(), model.provenance.describe()),
            };
            return Err(Gate {
                reason: Refusal::new(
                    text,
                    TeachRule::OtherSettings,
                    None,
                ),
                keep_sample: false,
                thin: false,
                dedup: false,
                untrusted: false,
            });
        }
        if let Some(&zu) = z.iter().find(|&&zi| model.basis.species_index(zi).is_none()) {
            let known: Vec<String> = model.basis.species.iter().map(|z| units::isotopes::symbol(*z).to_string()).collect();
            return Err(Gate {
                reason: Refusal::new(
                    format!(
                        "surrogate: species {} is not in the model ({}) — the model is rebuilt with it at the next refit",
                        units::isotopes::symbol(zu),
                        known.join(", ")
                    ),
                    TeachRule::NewSpecies(zu),
                    None,
                ),
                keep_sample: true,
                thin: false,
                dedup: false,
                untrusted: false,
            });
        }
        // collection mode: the scene's box must have the isolated-atom reference of every species —
        // its answers are fitted against them (`shared` module doc, "References per box"); kept,
        // waiting for them (the provider asks the auditor: `request_box_references`)
        if let Some(&zm) = self.box_refs_missing(z, prov.grid_n).first() {
            return Err(Gate {
                reason: Refusal::new(
                    format!(
                        "surrogate: the shared collection has no isolated-atom reference of {} in this {}³ box yet — the teacher answers, and its answers wait for that reference before they enter the fit",
                        units::isotopes::symbol(zm),
                        prov.grid_n
                    ),
                    TeachRule::NewSpecies(zm),
                    None,
                ),
                keep_sample: true,
                thin: false,
                dedup: false,
                untrusted: false,
            });
        }
        if let Some((key, why)) = self.untrusted_check() {
            return Err(Gate {
                reason: Refusal::new(format!("surrogate untrusted: {why}"), TeachRule::Untrusted(key), None),
                keep_sample: true,
                // thinned unless the fit is short of samples; never a copy, short or not:
                // copies would meet the counts and shrink the fitted noise (round 1, MUST-2)
                thin: !self.trust_needs_samples(),
                dedup: true,
                untrusted: true,
            });
        }
        Ok(())
    }

    /// Rule 5 (the module doc's 5a–5c): each atom's distance from the training envelope in units
    /// of the limit (> 1 is outside; draft mode's readout: the larger of the spread distance over
    /// `max_descriptor_distance` and ln(coordination ratio)/ln [`COORDINATION_RATIO`]; ∞ for an
    /// isolated atom) and the reason naming the rule that fired (and the atom), if any atom is
    /// outside.
    pub(super) fn envelope_outside(&self, model: &Model, pos: &[[f64; 3]], z: &[u8]) -> (Vec<f64>, Option<Refusal>) {
        self.envelope_outside_except(model, pos, z, &[], None)
    }

    /// [`envelope_outside`](Self::envelope_outside) with the atoms `stray` exempt from 5a–5e: the stray atoms of a
    /// cleared, in-region scene (`stray` module doc, "Answering inside a region"; spec §4.4) — the model's answer there
    /// is the free atom's E_ref with zero force, measured to be the teacher's. Their distance is 0; the bonded part is
    /// judged as before (its descriptors are the bonded part's own: a stray atom is no atom's neighbour). `desc`: the
    /// scene's descriptors under the model's basis when the caller has them (`evaluate`'s, shared with the cleared
    /// decision), else they are computed here.
    pub(super) fn envelope_outside_except(&self, model: &Model, pos: &[[f64; 3]], z: &[u8], stray: &[usize], desc: Option<&[Vec<f64>]>) -> (Vec<f64>, Option<Refusal>) {
        let Some(env) = self.envelope.as_ref() else { return (Vec::new(), None) };
        let described;
        let desc = match desc.filter(|d| d.len() == pos.len()) {
            Some(d) => d,
            None => match describe_values(&model.basis, pos, z) {
                Ok(d) => {
                    described = d;
                    &described[..]
                }
                Err(_) => return (Vec::new(), None),
            },
        };
        let Ok(s) = model.basis.species_indices(z) else { return (Vec::new(), None) };
        let n = pos.len();
        let limit = self.policy.max_descriptor_distance;
        let r_cut = model.basis.r_cut;
        let mut isolated: Option<usize> = None;
        let mut worst = (0.0f64, 0usize, 0usize);
        let mut worst_ratio = (0.0f64, 0usize, 0usize);
        let mut worst_grade = (0.0f64, 0usize);
        let mut atoms = Vec::with_capacity(desc.len());
        for (i, d) in desc.iter().enumerate() {
            if stray.contains(&i) {
                atoms.push(0.0);
                continue;
            }
            let (dist, k) = env.distance(s[i], d);
            // 5a and 5b only in a scene of more than one atom: a lone atom in a one-atom scene is
            // the reference itself (the model answers E_ref exactly, ruling S5, and the reference
            // is the teacher's own isolated atom, ruling S9)
            let lone = n >= 2 && lone_atom(&model.basis, d);
            let (ratio, kc) = if n >= 2 { env.coordination_ratio(&model.basis, s[i], d) } else { (0.0, 0) };
            let rel = if ratio > 1.0 { ratio.ln() / COORDINATION_RATIO.ln() } else { 0.0 };
            // 5e: the extrapolation grade against the MaxVol active set
            let grade = self.active_set.as_ref().map_or(0.0, |a| a.grade(s[i], d));
            if grade > worst_grade.0 {
                worst_grade = (grade, i);
            }
            atoms.push(if lone { f64::INFINITY } else { (dist / limit).max(rel).max(grade / self.policy.max_grade) });
            if lone && isolated.is_none() {
                isolated = Some(i);
            }
            if dist > worst.0 {
                worst = (dist, i, k);
            }
            if ratio > worst_ratio.0 {
                worst_ratio = (ratio, i, kc);
            }
        }
        let grade_void = self.active_set.as_ref().is_some_and(|a| !a.converged());
        let reason = if let Some(i) = isolated {
            // 5a
            Some(Refusal::new(
                format!(
                    "surrogate: isolated atom — atom {i} has no neighbour within r_cut = {r_cut} Bohr in a scene of {n} atoms (the model's free-atom limit Σ E_ref is not the teacher's: a dissociated pair keeps the scene's spin state)"
                ),
                TeachRule::IsolatedAtom,
                Some(i),
            ))
        } else if worst_ratio.0 > COORDINATION_RATIO {
            // 5b
            Some(Refusal::new(
                format!(
                    "surrogate: atom {} is outside the descriptor envelope — its coordination {} is {:.1}× below the smallest the fit has seen (limit {COORDINATION_RATIO})",
                    worst_ratio.1,
                    model.basis.feature_names()[worst_ratio.2],
                    worst_ratio.0,
                ),
                TeachRule::Coordination,
                Some(worst_ratio.1),
            ))
        } else if worst.0 > limit {
            // 5c
            Some(Refusal::new(
                format!(
                    "surrogate: atom {} is {:.1} training spreads outside the descriptor envelope (feature {}, limit {})",
                    worst.1,
                    worst.0,
                    model.basis.feature_names()[worst.2],
                    limit
                ),
                TeachRule::Spread,
                Some(worst.1),
            ))
        } else if grade_void {
            // 5e without a grade: MaxVol did not converge (`surrogate::grade`)
            Some(Refusal::new(
                "surrogate: extrapolating: no grade — the active set's MaxVol did not converge, so no training row is known to lie within it",
                TeachRule::NoGrade,
                None,
            ))
        } else if worst_grade.0 == f64::INFINITY {
            // 5e at γ = ∞: a component outside the training atoms' span (5d's per-atom case;
            // its own code, plan review S7)
            Some(Refusal::new(
                format!(
                    "surrogate: new environment (feature never seen in training) — atom {} has a component outside the span of every training environment (grade γ = ∞)",
                    worst_grade.1
                ),
                TeachRule::UnspannedAtom,
                Some(worst_grade.1),
            ))
        } else if worst_grade.0 > self.policy.max_grade {
            // 5e
            Some(Refusal::new(
                format!(
                    "surrogate: extrapolating: grade γ = {:.2} > γ_max = {} (atom {}: beyond the active set of its training environments)",
                    worst_grade.0, self.policy.max_grade, worst_grade.1
                ),
                TeachRule::Grade,
                Some(worst_grade.1),
            ))
        } else {
            None
        };
        (atoms, reason)
    }

    /// The strict policy's verdict **per atom** at `scene`'s geometry ([`AtomVerdicts`]; M8a Task 6,
    /// the transfer test's "atoms it calls trusted"). Read-only, the teacher is never called, and
    /// every number is the one [`would_answer`](Self::would_answer) reads: `would_answer` is `Ok`
    /// exactly when [`AtomVerdicts::answers`]. `None` for an empty scene. The fences read the
    /// descriptor cache as `would_answer` does (`ensure_desc_cache` first while samples are held
    /// or M has misfit atoms). Near a misfit refuses per atom, never the geometry: exactly the
    /// atoms near a misfit atom (`near_misfit_atoms`; spec 2026-11-13 §4.3).
    pub fn atom_verdicts(&self, scene: &Scene) -> Option<AtomVerdicts> {
        let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        let z: Vec<u8> = scene.nuclei.iter().map(|n| n.z).collect();
        let n = pos.len();
        if n == 0 {
            return None;
        }
        let whole = |rule: TeachRule| Some(AtomVerdicts { geometry: Some(rule), atoms: vec![None; n] });
        let prov = self.scene_provenance(scene);
        if let Err(g) = self.gate(&z, &prov) {
            return whole(g.reason.rule);
        }
        // a cleared stray-atom scene (spec §4.4, `evaluate`'s): its stray atoms are exempt, the bonded part judged alone
        let cleared = self.stray_cleared(&pos, &z, &prov);
        let part = cleared.as_ref().map(|c| &c.0);
        let model = self.model.as_ref().expect("the gate passed: a model exists");
        // past the gate an unknown species is unreachable (rule 2), named as `would_answer` names it
        let pred = match model.predict(&pos, &z, true) {
            Ok(p) => p,
            Err(surrogate::DescribeError::UnknownSpecies(zu)) => return whole(TeachRule::NewSpecies(zu)),
        };
        if let Some(r) = non_finite_answer(&pred) {
            return whole(r.rule);
        }
        let mut atoms: Vec<Option<TeachRule>> = vec![None; n];
        let mut geometry: Option<TeachRule> = None;
        let refuse = |atoms: &mut Vec<Option<TeachRule>>, i: usize, rule: TeachRule| {
            if atoms[i].is_none() {
                atoms[i] = Some(rule);
            }
        };
        // 5a–5c and 5e per atom (`envelope_outside`'s tests, atom by atom, in its order; a cleared scene's stray atoms
        // exempt, `envelope_outside_except`)
        if let (Some(env), Ok(desc), Ok(sp)) = (self.envelope.as_ref(), describe_values(&model.basis, &pos, &z), model.basis.species_indices(&z)) {
            if self.active_set.as_ref().is_some_and(|a| !a.converged()) {
                geometry = geometry.or(Some(TeachRule::NoGrade));
            }
            for (i, d) in desc.iter().enumerate() {
                if part.is_some_and(|c| c.stray.contains(&i)) {
                    continue;
                }
                let (dist, _) = env.distance(sp[i], d);
                let lone = n >= 2 && lone_atom(&model.basis, d);
                let (ratio, _) = if n >= 2 { env.coordination_ratio(&model.basis, sp[i], d) } else { (0.0, 0) };
                let grade = self.active_set.as_ref().map_or(0.0, |a| a.grade(sp[i], d));
                if lone {
                    refuse(&mut atoms, i, TeachRule::IsolatedAtom);
                } else if ratio > COORDINATION_RATIO {
                    refuse(&mut atoms, i, TeachRule::Coordination);
                } else if dist > self.policy.max_descriptor_distance {
                    refuse(&mut atoms, i, TeachRule::Spread);
                } else if grade == f64::INFINITY {
                    refuse(&mut atoms, i, TeachRule::UnspannedAtom);
                } else if grade > self.policy.max_grade {
                    refuse(&mut atoms, i, TeachRule::Grade);
                }
            }
        }
        let posterior = match (pred.energy_std, pred.force_std_epistemic.as_ref()) {
            (Some(e), Some(f)) => Some((e, f)),
            _ => None,
        };
        // 5d: the energy row refuses the geometry, an atom's force rows the atom
        if posterior.is_some() {
            if let Some((nu_e, nu_atoms)) = self.novelty_per_atom_judged(&pos, &z, part) {
                if nu_e > 1.0 {
                    geometry = geometry.or(Some(TeachRule::NewEnvironment));
                }
                for (i, nu) in nu_atoms.into_iter().enumerate() {
                    if nu > 1.0 {
                        refuse(&mut atoms, i, TeachRule::NewEnvironment);
                    }
                }
            }
        }
        // 4b: each atom's own environment key
        let rc2 = model.basis.r_cut * model.basis.r_cut;
        for (i, key) in env_keys(&pos, &z, rc2).into_iter().enumerate() {
            if let Some((r, _)) = key.and_then(|k| self.class_untrusted(&[Some(k)])) {
                refuse(&mut atoms, i, r.rule);
            }
        }
        // 6 per atom, 7 for the geometry
        if let Some((e, f)) = posterior {
            let v = self.variance_check(model, &pred, e, f, &pos, &z);
            for (i, ratio) in v.atoms.iter().enumerate() {
                if !(*ratio <= 1.0) {
                    refuse(&mut atoms, i, TeachRule::ForceUncertainty);
                }
            }
            if let Some(r) = self.energy_variance_refusal(model, &pos, e, &self.atom_inflation(model, &pos, &z)) {
                geometry = geometry.or(Some(r.rule));
            }
        }
        // near a misfit (spec 2026-11-13 §4.3): per atom, not the geometry — exactly the atoms
        // whose environment is near a misfit atom (`would_answer` refuses when any is)
        for (i, near) in self.near_misfit_atoms_judged(&pos, &z, part).into_iter().enumerate() {
            if near {
                refuse(&mut atoms, i, TeachRule::NearMisfit);
            }
        }
        // the quarantine and audit fences refuse the geometry
        if geometry.is_none() {
            if !self.near_quarantined_judged(&pos, &z, part).is_empty() {
                geometry = Some(TeachRule::NearQuarantine);
            } else if let Some(r) = self.audit_fence_refusal_judged(&pos, &z, part) {
                geometry = Some(r.rule);
            }
        }
        Some(AtomVerdicts { geometry, atoms })
    }

    /// The largest per-atom extrapolation grade γ of a geometry against the current active set
    /// (rule 5e; a readout — `None` without a fit or for an unknown species).
    pub fn extrapolation_grade(&self, pos: &[[f64; 3]], z: &[u8]) -> Option<f64> {
        let (model, act) = (self.model.as_ref()?, self.active_set.as_ref()?);
        let desc = describe_values(&model.basis, pos, z).ok()?;
        let s = model.basis.species_indices(z).ok()?;
        Some(desc.iter().zip(&s).map(|(d, &si)| act.grade(si, d)).fold(0.0f64, f64::max))
    }

    /// The variance inflation per atom of a geometry (M8a Task 2 fix round 3; `calibrate`, "The
    /// variance inflation"): λ of the atom's environment key (class, Z, neighbour species — M8a
    /// user-test fix 2; `SpeciesVerdict::inflation`), 1 for an isolated atom or before the class
    /// has numbers.
    pub(super) fn atom_inflation(&self, model: &Model, pos: &[[f64; 3]], z: &[u8]) -> Vec<f64> {
        let rc2 = model.basis.r_cut * model.basis.r_cut;
        if z.len() != pos.len() {
            return vec![1.0; pos.len()];
        }
        env_keys(pos, z, rc2).into_iter().map(|k| k.map_or(1.0, |k| self.key_inflation(&k))).collect()
    }

    /// Rules 6–7 on a prediction that carries stds; `f_std` is the **epistemic** per-atom force
    /// std (`Prediction::force_std_epistemic`: force_std² = 3σ_f² + epistemic², exact) — rule 6
    /// leaves the label noise out as rule 7 does for the energy (module doc). Since M8a Task 2 fix
    /// round 3 each atom's epistemic force std is multiplied by its class's variance inflation λ
    /// (`atom_inflation`; 1 for a calibrated or unmeasured class) and the epistemic energy std by
    /// the largest λ among the geometry's atoms — one scale of the posterior covariance per class,
    /// measured where the posterior decides (`calibrate`, "The variance inflation").
    ///
    /// At a stray-atom scene both rules are the bonded part's own without an exemption: rule 7 is per energy-row atom
    /// ([`energy_variance_refusal`](Self::energy_variance_refusal)), and a stray atom's force rows are zero, so its
    /// epistemic force std is exactly 0 (rule 6).
    pub(super) fn variance_check(&self, model: &Model, pred: &surrogate::Prediction, e_std: f64, f_std: &[f64], pos: &[[f64; 3]], z: &[u8]) -> VarianceVerdict {
        let n = pred.forces.len();
        let p = &self.policy;
        let infl = self.atom_inflation(model, pos, z);
        // 6. forces, epistemic only, inflated by the class's calibration
        let mut uncertainty = 0.0f64;
        let mut worst: Option<(usize, f64, f64)> = None;
        let mut atoms = Vec::with_capacity(n);
        for (j, (&sd, f)) in f_std.iter().zip(&pred.forces).enumerate() {
            let fs = sd / SQRT_3 * infl.get(j).copied().unwrap_or(1.0);
            uncertainty = max_nan(uncertainty, fs);
            let bound = p.force_bound((f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt());
            atoms.push(fs / bound);
            // "not within the bound": a NaN std refuses (activity Task 3b), and is the worst
            let over = !(fs <= bound);
            if over && worst.is_none_or(|(_, s, b)| !(s / b).is_nan() && (fs.is_nan() || fs / bound > s / b)) {
                worst = Some((j, fs, bound));
            }
        }
        // the refusal says when the class's calibration decided (byte-identical without it)
        let scaled = |k: f64| if k == 1.0 { String::new() } else { format!(" (the model's std × {k:.2}, its measured calibration for this environment class)") };
        if let Some((j, fs, bound)) = worst {
            return VarianceVerdict {
                accept: false,
                uncertainty,
                reason: Some(Refusal::new(
                    format!("surrogate: epistemic force std {fs:.2e} > bound {bound:.2e} Ha/Bohr on atom {j}{}", scaled(infl.get(j).copied().unwrap_or(1.0))),
                    TeachRule::ForceUncertainty,
                    Some(j),
                )),
                atoms,
                novel: false,
            };
        }
        // 7. energy, per atom, epistemic only
        if let Some(r) = self.energy_variance_refusal(model, pos, e_std, &infl) {
            return VarianceVerdict { accept: false, uncertainty, reason: Some(r), atoms, novel: false };
        }
        VarianceVerdict { accept: true, uncertainty, reason: None, atoms, novel: false }
    }

    /// Rule 7 alone ([`variance_check`](Self::variance_check)'s energy half): the epistemic energy
    /// std per atom of an `n`-atom prediction whose total std is `e_std`, inflated by the largest
    /// of the atoms' class factors `infl`, against `energy_std_per_atom`.
    ///
    /// Per atom over the geometry's N_e energy-row atoms (`surrogate::energy_row_atoms`: N without an isolated atom,
    /// controller ruling 9a; spec §4.4, review I4): `e_std` is `Model::predict`'s, whose aleatoric floor (N_e s_E)² is
    /// removed exactly, and the epistemic rest is divided by N_e. At a stray-atom scene both are the bonded part's own —
    /// a stray atom adds nothing to the energy row (B_s = 0), so the prediction's std is its bonded part's bit for bit —
    /// so the number is rule 7 on the bonded part alone, cleared or not (Task 4's per-bonded-atom special case, now
    /// exactly what the general rule computes).
    pub(super) fn energy_variance_refusal(&self, model: &Model, pos: &[[f64; 3]], e_std: f64, infl: &[f64]) -> Option<Refusal> {
        let p = &self.policy;
        let scaled = |k: f64| if k == 1.0 { String::new() } else { format!(" (the model's std × {k:.2}, its measured calibration for this environment class)") };
        let n_e = surrogate::energy_row_atoms(&model.basis, pos).max(1);
        let floor = if model.energy_per_atom { n_e as f64 * model.sigma_e } else { model.sigma_e };
        // `max(0.0)` would turn a NaN variance into 0: keep it (activity Task 3b)
        let e_infl = infl.iter().copied().fold(1.0f64, max_nan);
        let epistemic_per_atom = max_nan(e_std * e_std - floor * floor, 0.0).sqrt() / n_e as f64 * e_infl;
        (!(epistemic_per_atom <= p.energy_std_per_atom)).then(|| {
            Refusal::new(
                format!("surrogate: epistemic energy std {epistemic_per_atom:.2e} > {:.2e} Ha/atom{}", p.energy_std_per_atom, scaled(e_infl)),
                TeachRule::EnergyUncertainty,
                None,
            )
        })
    }
}

/// `f` at the atoms rule 5d and the fences judge (`stray` module doc, "Answering inside a region"; spec §4.4): the
/// bonded part alone of a cleared stray-atom scene (`cleared`, `stray_cleared`'s split), else the scene. The bonded part
/// must pass them **on its own**: 5d's energy row is per atom over the atoms it is given (n_bonded, review I4), and a
/// fence that takes the worst atom's distance to a fenced geometry (the quarantine and audit fences) would let a stray
/// atom, which no fenced H₂ has a counterpart for, carry an otherwise fenced bonded geometry outside. The caller maps
/// the atoms named back to the scene's ([`Split::scene_atom`], [`Split::scene_values`]).
pub(super) fn judged_part<T>(pos: &[[f64; 3]], z: &[u8], cleared: Option<&Split>, f: impl FnOnce(&[[f64; 3]], &[u8]) -> T) -> T {
    match cleared {
        Some(sp) => {
            let (p, q) = sp.bonded_part(pos, z);
            f(&p, &q)
        }
        None => f(pos, z),
    }
}

/// A refusal [`judged_part`] gave, its atom in the scene's numbering (its text names no atom index: the quarantine,
/// shell and audit fences').
pub(super) fn scene_refusal(mut r: Refusal, cleared: Option<&Split>) -> Refusal {
    if let Some(sp) = cleared {
        r.atom = sp.scene_atom(r.atom);
    }
    r
}

impl SurrogateKs {
    /// Rule 5d's (ν, the atom whose force rows reach furthest) at a scene, the bonded part alone when `cleared`
    /// ([`judged_part`]); the atom in the scene's numbering. `None` without a null space or for a species it lacks.
    pub(super) fn novelty_judged(&self, pos: &[[f64; 3]], z: &[u8], cleared: Option<&Split>) -> Option<(f64, Option<usize>)> {
        let ns = self.null_space.as_ref()?;
        let (nu, at) = judged_part(pos, z, cleared, |p, q| ns.novelty(p, q).ok())?;
        Some((nu, cleared.map_or(at, |sp| sp.scene_atom(at))))
    }

    /// Rule 5d per atom (`NullSpace::novelty_per_atom`) as [`novelty_judged`](Self::novelty_judged): a stray atom's ν
    /// is 0 (its force rows are zero).
    pub(super) fn novelty_per_atom_judged(&self, pos: &[[f64; 3]], z: &[u8], cleared: Option<&Split>) -> Option<(f64, Vec<f64>)> {
        let ns = self.null_space.as_ref()?;
        let (nu_e, atoms) = judged_part(pos, z, cleared, |p, q| ns.novelty_per_atom(p, q).ok())?;
        Some((nu_e, cleared.map_or_else(|| atoms.clone(), |sp| sp.scene_values(&atoms, pos.len(), 0.0))))
    }

    /// Which atoms are near a misfit (`near_misfit_atoms`), the bonded part's alone when `cleared` (a stray atom is
    /// exempt: false); read-only.
    pub(super) fn near_misfit_atoms_judged(&self, pos: &[[f64; 3]], z: &[u8], cleared: Option<&Split>) -> Vec<bool> {
        let flags = judged_part(pos, z, cleared, |p, q| self.near_misfit_atoms(p, q));
        cleared.map_or_else(|| flags.clone(), |sp| sp.scene_values(&flags, pos.len(), false))
    }

    /// The shell and audit fences (`audit_fence_refusal`) at the judged atoms, the atom in the scene's numbering.
    pub(super) fn audit_fence_refusal_judged(&self, pos: &[[f64; 3]], z: &[u8], cleared: Option<&Split>) -> Option<Refusal> {
        judged_part(pos, z, cleared, |p, q| self.audit_fence_refusal(p, q)).map(|r| scene_refusal(r, cleared))
    }

    /// The shell fence alone (`shell_fence_refusal`) at the judged atoms, the atom in the scene's numbering.
    pub(super) fn shell_fence_refusal_judged(&self, pos: &[[f64; 3]], z: &[u8], cleared: Option<&Split>) -> Option<Refusal> {
        judged_part(pos, z, cleared, |p, q| self.shell_fence_refusal(p, q)).map(|r| scene_refusal(r, cleared))
    }

    /// The fencing samples near the judged atoms (`near_quarantined_cached`; read-only).
    pub(super) fn near_quarantined_judged(&self, pos: &[[f64; 3]], z: &[u8], cleared: Option<&Split>) -> Vec<usize> {
        judged_part(pos, z, cleared, |p, q| self.near_quarantined_cached(p, q))
    }
}
