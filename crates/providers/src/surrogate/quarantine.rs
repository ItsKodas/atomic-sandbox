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

/// A stored sample of **lone atoms** — no atom has a neighbour within the model's cutoff
/// ([`no_neighbours`]) — loaded from a collection or a file written before the keep rule refused
/// such answers (`TeachOutcome::LoneNotKept`; user test M8a: 202 lone-H samples spread by 1.3 mHa
/// failed rule 4's self-check and untrusted a whole collection). Every descriptor of such a
/// sample is empty, so the model's answer there is Σ ref exactly (no intercept, `surrogate::fit`
/// "Isolated-atom limit") and its label E − Σ ref(own box) is only the isolated atom's grid/box
/// artefact: it teaches nothing. Set at load by [`SurrogateKs::hold_out_lone`] and **held out of
/// the fit for good** — not fitted, not in the envelope, the grade, the leave-one-out or the
/// thinning distance, never fencing, never released or superseded (only `Q_HELD` is), never
/// promoted (only `Q_DRAFT` is), never moved by the box or reference rules (they touch only
/// `Q_NONE` and `Q_REF_WAIT`). A provider-only code like [`Q_REF_WAIT`]: never written — a
/// collection gets no status record for it (its own code stands), a saved file writes the code the
/// sample was loaded with (`lone_codes`) — so nothing on disk changes and the samples stay for a
/// model that may one day learn from them (long-range terms).
pub(super) const Q_LONE: u8 = 6;

/// A box-affected sample ([`surrogate::quarantine::Q_BOX`]).
pub(super) const Q_BOX: u8 = surrogate::quarantine::Q_BOX;

/// A draft-mode audit's answer held out of the strict fit ([`surrogate::quarantine::Q_DRAFT`]).
pub(super) const Q_DRAFT: u8 = surrogate::quarantine::Q_DRAFT;

/// A sample the user forgot ([`surrogate::quarantine::Q_FORGOTTEN`], spec R2): excluded and
/// fencing like [`Q_HELD`], never released or superseded (only [`SurrogateKs::restore_sample`]
/// clears it), counted with the held samples against [`QUARANTINE_MAX_FRACTION`].
pub(super) const Q_FORGOTTEN: u8 = surrogate::quarantine::Q_FORGOTTEN;

/// A forgotten sample whose fence was lifted ([`surrogate::quarantine::Q_FORGOTTEN_UNFENCED`];
/// `surrogate` module doc, "Lifting a forgotten sample's fence"): still forgotten — excluded, never
/// released or superseded, restorable, purgeable — but no longer fencing.
pub(super) const Q_FORGOTTEN_UNFENCED: u8 = surrogate::quarantine::Q_FORGOTTEN_UNFENCED;

/// Quarantine rounds per refit (`surrogate` module doc).
pub const QUARANTINE_ROUNDS: usize = 5;

/// The Model tab's "Worst-fitting samples" lists this many ([`SurrogateKs::sample_suspects`],
/// spec R2): a display length, not a physics parameter.
pub const SUSPECTS_SHOWN: usize = 8;

/// A stored sample's check value for the Model tab's buttons ([`sim::SampleId`], fix round 1 M8):
/// FNV-1a over its Z and the exact bits of its positions, energy and forces (re-review N1: two
/// stores built from one preset hold the same geometry at the same index, so positions alone do
/// not tell their samples apart; the teacher answer almost always does, and a collision only ever
/// refuses on the safe side or names the same answer).
pub(super) fn sample_key(s: &Sample) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |b: u8| h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
    for &z in &s.z {
        eat(z);
    }
    for p in s.pos.iter().chain(&s.forces) {
        for c in p {
            c.to_bits().to_le_bytes().into_iter().for_each(&mut eat);
        }
    }
    s.energy.to_bits().to_le_bytes().into_iter().for_each(&mut eat);
    h
}

/// Where a stored sample came from, in a word or two (the Model tab's row: `#1804 · HO · fallback
/// · 38× its bound`).
pub(super) fn origin_word(tag: SampleTag) -> &'static str {
    match tag {
        SampleTag::Perturbation => "pass perturbation",
        SampleTag::Scan => "pass scan",
        SampleTag::HotBurst => "pass hot burst",
        SampleTag::Fallback => "fallback",
        SampleTag::LiveStep => "live step",
        SampleTag::Reference => "reference",
        SampleTag::Audit => "audit",
        SampleTag::DraftAudit => "draft audit",
        SampleTag::Targeted => "teach-around",
    }
}

/// Never quarantine more than this fraction of the store.
pub const QUARANTINE_MAX_FRACTION: f64 = 0.1;

/// The note of a trust recovery that restored nothing and left the model untrusted (spec
/// 2026-11-13 §4.2; [`SurrogateKs::settle_recovery_note`] rewords it when the refit ends trusted).
const RECOVERY_FAILED_NOTE: &str = "recovery did not restore trust; no samples held";
/// The settled note's head when the refit ends trusted (followed by "{n} samples the model cannot
/// reproduce stay in the fit").
const RECOVERY_SETTLED_NOTE: &str = "recovery did not restore the training residuals; no samples held;";

/// One fresh teacher answer inside a quarantine fence that disagreed with the model (fence voting):
/// the answer and, when the model predicted there, its residual teacher − model (energy, forces).
#[derive(Clone)]
pub(super) struct FenceVote {
    pub(super) near: Vec<usize>,
    pub(super) n: usize,
    pub(super) energy: f64,
    pub(super) forces: Vec<[f64; 3]>,
    pub(super) residual: Option<(f64, Vec<[f64; 3]>)>,
    /// The model generation the residual was taken against: votes combine only within one
    /// (fix round 1, the review's I1 — a refit in between changes what "the model's error" is).
    pub(super) gen: u64,
}

/// Two fence votes agree (`surrogate` module doc, "Fence voting"): their residuals teacher − model
/// do — the per-atom energy residuals within `energy_std_per_atom` and, per atom, the force
/// residuals' magnitudes within the policy's force bound of that atom's teacher force in the second
/// vote (`force_bound(|F_teacher|)`, as everywhere else) — when both carry one; else the
/// raw answers ([`SurrogateKs::agree`]). Residuals, not answers (M8a Task 5, the user's
/// compressed-H₂ log): votes are fresh answers at *different* geometries along a trajectory
/// (a vibration enters the fence once per period), whose raw energies and forces differ by the
/// surface's own slope — on a steep inner wall more than the tiers — while the model's error
/// there varies slowly; magnitudes, because the molecule turns between votes and a force
/// residual turns with it. Magnitudes drop the direction: two residuals of equal size pointing
/// differently relative to the molecule agree here (a direction-aware invariant — the residual
/// projected on the interatomic unit vectors, or Kabsch-aligned frames — would not); the vote only
/// keeps one sample for the next refit, whose `judge_votes` decides on the full vectors.
pub(super) fn votes_agree(p: &FallbackPolicy, va: &FenceVote, vb: &FenceVote) -> bool {
    match (&va.residual, &vb.residual) {
        (Some((ea, fa)), Some((eb, fb))) => {
            let norm = |v: &[f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            (ea - eb).abs() / va.n.max(1) as f64 <= p.energy_std_per_atom
                && fa.iter().zip(fb).zip(&vb.forces).all(|((a, b), t)| (norm(a) - norm(b)).abs() / 3f64.sqrt() <= p.force_bound(norm(t)))
        }
        _ => SurrogateKs::agree(p, va.n, va.energy, &va.forces, vb.energy, &vb.forces),
    }
}

/// One fitted sample's **leave-one-out ratio** (spec `2026-11-12-sample-recovery-design.md` R1;
/// the recovery's offender ranking since then, `worst_offenders`; the Model tab's "worst-fitting
/// samples"). Atomic units throughout (Hartree, Bohr).
///
/// `surrogate::loo` gives the sample's residual under the fit **without it**, at the fit's own
/// hyperparameters: r = y − ŷ₋ (target minus the prediction of the fit that never saw the
/// sample) — the energy part r_E (Hartree per atom when the fit is per atom, else Hartree) when
/// its energy was fitted, and per fitted atom j the force part r_j (Ha/Bohr). The fit without the
/// sample therefore predicts F_j − r_j at atom j, and with N the sample's atom count:
///
/// ```text
///   ρ_E = |r_E| / energy_std_per_atom            per-atom fit
///   ρ_E = |r_E| / N / energy_std_per_atom        whole-energy fit   (0 when the energy was not fitted)
///   d_j = force_residual(F_j − r_j, F_j).0 = ‖r_j‖/√3,   b_j = force_bound(|F_j|) = force_residual(…).1
///   ρ   = max(ρ_E, max_j d_j / b_j)              each atom against its own bound (the class rule's reading)
///   ρ_σ = max(ρ_E, max_j d_j / force_std_abs)    the absolute tier (the recovery when only σ_f fails)
/// ```
///
/// exactly the numbers the class pass reads (`trust::compute_class_loo_full`: `e_ratio` and
/// `force_residual(&[f − r], f)`), over the same fitted rows (a masked sample's interior atoms;
/// `r_j` is `None` for the others). **Ruling R1a:** an undefined residual (`l` is `None`: the
/// sample's block I − H_ss is singular — it alone spans a direction of the fit — as the class
/// rule's `a.3 = ∞`) and a ratio that is not a number are ∞: such a sample ranks first. Returns
/// (ρ, ρ_σ, the [`Channel`] that attains ρ_σ) — the channel so the noise-only recovery note names
/// the tier it measured against (Task 1 re-review N1: a high-leverage energy outlier's ρ_σ is its
/// energy ratio, against 1e-3 Ha/atom, not the force tier). The force channel wins a tie; an
/// undefined residual reports the force channel (no channel was measured).
pub(super) fn loo_ratio(p: &FallbackPolicy, per_atom: bool, s: &Sample, l: Option<&surrogate::loo::SampleLoo>) -> (f64, f64, Channel) {
    let Some(l) = l else { return (f64::INFINITY, f64::INFINITY, Channel::Force) };
    let n = s.pos.len().max(1) as f64;
    let e = if s.use_energy { l.energy.map_or(f64::INFINITY, |e| (if per_atom { e.abs() } else { e.abs() / n }) / p.energy_std_per_atom) } else { 0.0 };
    let (mut own, mut force_tier) = (e, 0.0f64);
    for (r, f) in l.forces.iter().zip(&s.forces) {
        let Some(r) = r else { continue };
        let (d, b) = p.force_residual(&[f[0] - r[0], f[1] - r[1], f[2] - r[2]], f);
        own = super::trust::max_nan(own, d / b);
        force_tier = super::trust::max_nan(force_tier, d / p.force_std_abs);
    }
    let tier = super::trust::max_nan(e, force_tier);
    let channel = if e > force_tier { Channel::Energy } else { Channel::Force };
    let undefined_is_worst = |x: f64| if x.is_nan() { f64::INFINITY } else { x };
    (undefined_is_worst(own), undefined_is_worst(tier), channel)
}

/// Which of a leave-one-out ratio's two channels attains it ([`loo_ratio`]'s third number).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Channel {
    /// The per-atom energy residual over `energy_std_per_atom`.
    Energy,
    /// The worst fitted atom's force residual over its bound (or `force_std_abs`).
    Force,
}

/// The per-sample leave-one-out ratios ([`loo_ratio`]) of one fit (spec ruling R1b): filled by
/// the class pass (`refresh_class_loo`), which computes every fitted sample's leave-one-out
/// anyway, or by [`SurrogateKs::ensure_loo_ratios`] for a fit whose class pass was deferred.
/// **Keyed to the fit**: valid only while the provider's `model_gen` is still `gen` (bumped by
/// every fit, restored state, install, attach, forget and policy change) **and** the `rows`' indices
/// are the standing fit's own members (`fit_members`: the samples that fit was built from). A stale
/// cache is never read: it is recomputed, over those members. A hold, a release or a forget between
/// fits changes the quarantine codes but not the fit: the ratios stay the standing fit's — a sample
/// held since is still listed (the model still holds it), a sample released since is not (the
/// model never saw it: a leave-one-out "as if fitted" would downdate rows the fit does not hold).
/// The next fit replaces them.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct LooRatios {
    /// The `model_gen` of the fit they were computed for.
    pub(super) gen: u64,
    /// Per fitted sample in store order: (store index, ρ — each atom against its own bound).
    pub(super) rows: Vec<(usize, f64)>,
    /// Per row: ρ_σ — the forces against `force_std_abs`.
    pub(super) tier: Vec<f64>,
    /// Per row: the channel that attains ρ_σ ([`loo_ratio`]; the noise-only note names it).
    pub(super) tier_channel: Vec<Channel>,
}

/// Which offender ranking a quarantine round takes its samples by ([`SurrogateKs::quarantine_rounds`];
/// spec R1 and its fix round 1, option A).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Ranking {
    /// The leave-one-out ratio ([`loo_ratio`], [`SurrogateKs::worst_offenders`]): tried first.
    LeaveOneOut,
    /// The in-sample training residual ([`SurrogateKs::training_offenders`], the ranking before
    /// 2026-11-12): the fallback when the leave-one-out rounds do not recover.
    Training,
}

/// What a round's worst ratio was measured against, for the note ([`Held`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Bound {
    /// Each sample's own bound: the per-atom energy threshold and each atom's force bound (the
    /// training residuals failed).
    Own,
    /// Only the force noise failed: the absolute force tier `force_std_abs` (Ha/Bohr) attained it.
    ForceTier(f64),
    /// Only the force noise failed, and the worst leave-one-out ratio is the sample's **energy**
    /// channel, against `energy_std_per_atom` (Ha/atom; re-review N1).
    EnergyTier(f64),
}

/// What a refit's quarantine rounds held, for the note: per bound a held sample was measured
/// against ([`Bound`]), the largest ratio among the samples held under it (0 / `None` when none;
/// ∞ for an undefined leave-one-out, ruling R1a). Each held sample's ratio is so stated against the
/// bound it was held under (fix round 1, M6: one maximum with the bound of the round that attained
/// it understated an earlier round's samples measured against another bound).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Held {
    /// Against each sample's own bound.
    pub(super) own: f64,
    /// Against the force-noise tier: (ratio, the tier in Ha/Bohr).
    pub(super) force_tier: Option<(f64, f64)>,
    /// Against the energy tier: (ratio, the tier in Ha/atom).
    pub(super) energy_tier: Option<(f64, f64)>,
}

impl Held {
    pub(super) const NONE: Held = Held { own: 0.0, force_tier: None, energy_tier: None };

    /// One held sample's `ratio` measured against `bound`.
    pub(super) fn add(&mut self, ratio: f64, bound: Bound) {
        let up = |x: f64| if ratio <= x { x } else { ratio };
        match bound {
            Bound::Own => self.own = up(self.own),
            Bound::ForceTier(t) => self.force_tier = Some((up(self.force_tier.map_or(0.0, |x| x.0)), t)),
            Bound::EnergyTier(t) => self.energy_tier = Some((up(self.energy_tier.map_or(0.0, |x| x.0)), t)),
        }
    }

    /// The largest ratio held, whatever its bound (0 when none).
    pub(super) fn worst(&self) -> f64 {
        [self.own, self.force_tier.map_or(0.0, |x| x.0), self.energy_tier.map_or(0.0, |x| x.0)].into_iter().fold(0.0, |w, r| if r <= w { w } else { r })
    }
}

/// What the fit without the held samples did, for the note's last clause (fix round 1, the
/// coordinator's ruling on concern 1: the measured fact only — a recovered result does not show
/// that the failures were the held samples' neighbours, nor which held sample caused them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Outcome {
    /// Its training residuals or noise still fail: no clause.
    Fails,
    /// Its residuals and noise pass, its trust counts do not: ", and the fit without them passes".
    Passes,
    /// Residuals, noise and counts pass ([`SurrogateKs::recovered`]): ", and the fit without them
    /// recovers".
    Recovers,
}

/// A ratio as the notes and the Model tab round it: a whole number from 10×, one decimal below,
/// "∞" when undefined (ruling R1a).
pub fn ratio_words(r: f64) -> String {
    if !r.is_finite() {
        "∞".to_string()
    } else if r >= 10.0 {
        format!("{r:.0}")
    } else {
        format!("{r:.1}")
    }
}

/// The recovery note's words for quarantine rounds (spec R1, "The recovery note says what was
/// done"; fix round 2, M3): `added` samples quarantined **by this refit** — the ones whose ratio
/// it measured — with `held` in all, and `h` the largest ratio among them by `ranking` (the
/// ranking whose result stands, fix round 1), against each sample's own bound or, in the
/// noise-only mode, the absolute tier whose channel attained it ([`Bound`], re-review N1) — each
/// bound a sample was held under named with its own maximum (fix round 1, M6). Rounded
/// ([`ratio_words`]). The last clause states what the fit without them measurably did
/// ([`Outcome`]; re-review N2 and fix round 1: a leave-one-out ratio above 1 says only that the
/// rest of the data cannot predict the sample).
pub(super) fn quarantine_words(held: usize, added: usize, h: Held, ranking: Ranking, outcome: Outcome) -> String {
    let worst = h.worst();
    let mut parts: Vec<String> = Vec::new();
    if h.own > 0.0 || (h.force_tier.is_none() && h.energy_tier.is_none()) {
        parts.push(format!("{}× its bound", ratio_words(h.own)));
    }
    if let Some((r, t)) = h.force_tier {
        parts.push(format!("{}× the {t:.0e} Ha/Bohr force-noise tier", ratio_words(r)));
    }
    if let Some((r, t)) = h.energy_tier {
        parts.push(format!("{}× its {t:.0e} Ha/atom energy tier", ratio_words(r)));
    }
    let up_to = parts.join(" and ");
    let clause = match outcome {
        Outcome::Fails => "",
        Outcome::Passes => ", and the fit without them passes",
        Outcome::Recovers => ", and the fit without them recovers",
    };
    let why = match ranking {
        Ranking::LeaveOneOut if worst.is_finite() => {
            format!("the fit without each of them predicts it at up to {up_to}{clause}")
        }
        Ranking::LeaveOneOut => format!("the fit without one of them cannot predict it at all (it alone spans a direction of the fit){clause}"),
        Ranking::Training => format!(
            "ranked by their own training residual (the leave-one-out ranking did not recover the fit), the fit misses each by up to {up_to}: the model cannot represent them"
        ),
    };
    format!("{added} samples quarantined this refit ({held} held in all) — {why} (kept, excluded from the fit; geometries near them fall back)")
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

    /// Samples the user forgot ([`Q_FORGOTTEN`], spec R2): excluded from the fit and fencing like
    /// the quarantined ones, never released; the Model tab's "N forgotten".
    pub fn forgotten(&self) -> usize {
        self.quarantine.iter().filter(|&&q| q == Q_FORGOTTEN || q == Q_FORGOTTEN_UNFENCED).count()
    }

    /// Of the [`forgotten`](Self::forgotten) samples, those whose fence was lifted
    /// ([`Q_FORGOTTEN_UNFENCED`]).
    pub fn forgotten_unfenced(&self) -> usize {
        self.quarantine.iter().filter(|&&q| q == Q_FORGOTTEN_UNFENCED).count()
    }

    /// Of the [`forgotten`](Self::forgotten) samples, those the standing fit still holds (forgotten
    /// since it, or while the background refit that installed it ran): the "fit vs held" line
    /// counts them in the fit until the next refit, never twice (fix round 1, M5).
    pub fn forgotten_in_fit(&self) -> usize {
        self.fit_member_indices().filter(|&i| self.is_forgotten(i)).count()
    }

    /// Whether stored sample `i` was forgotten by the user ([`Q_FORGOTTEN`]).
    pub fn is_forgotten(&self, i: usize) -> bool {
        matches!(self.quarantine.get(i), Some(&Q_FORGOTTEN) | Some(&Q_FORGOTTEN_UNFENCED))
    }

    /// Whether stored sample `i` fences its neighbourhood (geometries within the quarantine radius
    /// fall back): a quarantined ([`Q_HELD`]) or a forgotten ([`Q_FORGOTTEN`]) sample.
    pub(super) fn fences(&self, i: usize) -> bool {
        matches!(self.quarantine.get(i), Some(&Q_HELD) | Some(&Q_FORGOTTEN))
    }

    /// Whether stored sample `i` is one of the standing fit's members (`fit_members`: below its
    /// prefix and code 0 when that fit was installed) — a forgotten sample is, until the refit
    /// after its Forget.
    pub(super) fn in_standing_fit(&self, i: usize) -> bool {
        i < self.fit_members.0 && self.fit_members.1.get(i).is_none_or(|&q| q == Q_NONE)
    }

    /// Whether the fence at a geometry is forgotten samples' alone (`near` = the fencing samples
    /// near it, `near_quarantined`): the Model tab's "near samples you forgot".
    pub(super) fn fence_is_forgotten(&self, near: &[usize]) -> bool {
        !near.is_empty() && near.iter().all(|&i| self.quarantine.get(i) == Some(&Q_FORGOTTEN))
    }

    /// The quarantine fence's refusal text at a geometry fenced by `near` (rule "quarantine"): the
    /// forgotten-only fence says how it lifts.
    pub(super) fn fence_refusal_text(&self, near: &[usize]) -> String {
        let d = QUARANTINE_FRACTION * self.policy.max_descriptor_distance;
        if self.fence_is_forgotten(near) {
            format!("surrogate: within {d:.2} training spreads of samples you forgot (Model tab, Worst-fitting samples): the model answers here again once {FENCE_VOTES} fresh teacher answers in a row agree with it, or Restore the samples")
        } else {
            format!("surrogate: within {d:.2} training spreads of a sample held out of the fit (quarantined: the rest of the data could not predict it, or the fit could not reproduce it; or forgotten by the user)")
        }
    }

    /// How many stored samples fence ([`fences`](Self::fences)).
    pub(super) fn fencing(&self) -> usize {
        self.quarantine.iter().filter(|&&q| q == Q_HELD || q == Q_FORGOTTEN).count()
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

    /// Stored samples of lone atoms held out of the fit ([`Q_LONE`]).
    pub fn lone_held_out(&self) -> usize {
        self.quarantine.iter().filter(|&&q| q == Q_LONE).count()
    }

    /// Holds every stored sample in which no atom has a neighbour within `r_cut` (Bohr) out of the
    /// fit ([`Q_LONE`]), remembering the code it had (`lone_codes`: what a saved file writes back);
    /// how many were newly held. Called where stored samples are installed (a collection's attach,
    /// a model or samples file, a session-only file), never on a fresh teacher answer (the keep
    /// rule refuses those).
    pub(super) fn hold_out_lone(&mut self, r_cut: f64) -> usize {
        let Some(store) = self.store.as_ref() else { return 0 };
        let lone: Vec<usize> = (0..store.len()).filter(|&i| self.quarantine.get(i) != Some(&Q_LONE) && no_neighbours(&store.samples[i].pos, r_cut)).collect();
        if lone.is_empty() {
            return 0;
        }
        let n = store.len();
        self.quarantine.resize(n, Q_NONE);
        for &i in &lone {
            self.lone_codes.insert(i, self.quarantine[i]);
            self.quarantine[i] = Q_LONE;
        }
        lone.len()
    }

    /// The code sample `i` is written to a file with: a lone sample's ([`Q_LONE`]) is the code it
    /// was loaded with; the provider-only codes `Q_DRAFT` (its tag carries it) and `Q_REF_WAIT` (it
    /// waits again at the reload) write as `Q_NONE`.
    pub(super) fn persisted_code(&self, i: usize) -> u8 {
        let q = match self.quarantine.get(i).copied().unwrap_or(Q_NONE) {
            Q_LONE => self.lone_codes.get(&i).copied().unwrap_or(Q_NONE),
            q => q,
        };
        if q == Q_DRAFT || q == Q_REF_WAIT || q == Q_LONE {
            Q_NONE
        } else {
            q
        }
    }

    /// The user's note on [`hold_out_lone`](Self::hold_out_lone)'s count `k` of the samples from
    /// `source` ("shared collection <slug>", "model file …").
    pub(super) fn note_lone_held_out(&mut self, source: &str, k: usize, r_cut: f64) {
        if k > 0 {
            let r = format!("{r_cut}");
            self.user_notes.push(format!(
                "{source}: {k} sample{} of lone atoms (no neighbour within {r} Bohr) held out of the fit — they teach nothing (kept on disk, unchanged)",
                if k == 1 { "" } else { "s" }
            ));
        }
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

    /// The trust floor of the stored samples (the columns their usable atoms touch in the default
    /// basis, |∪ S_k| — `fit_floor`, since M8a user-fix 3's ruling on concern 2, before it the
    /// basis's Q′ — plus `MIN_RESIDUAL_DOF`): the clean sample count the box-affected samples'
    /// exclusion waits for. `None` without a store.
    pub fn trust_floor(&self) -> Option<usize> {
        let store = self.store.as_ref()?;
        let mut species: Vec<u8> = store.samples.iter().flat_map(|s| s.z.iter().copied()).collect();
        species.sort_unstable();
        species.dedup();
        Some(self.fit_floor(&self.default_basis(&species)) + MIN_RESIDUAL_DOF as usize)
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

    /// Whether stored sample `i` is left out of the fit (quarantined, superseded, box-excluded, a
    /// held-out draft guess, waiting for its box's reference, lone atoms, or forgotten by the user).
    pub(super) fn excluded(&self, i: usize) -> bool {
        self.quarantine.get(i).is_some_and(|&q| q != Q_NONE)
    }

    /// What the last refit's trust recovery did ("degree_cap 6 …", "N samples quarantined …"),
    /// `None` when it was not needed.
    pub fn recovery_note(&self) -> Option<&str> {
        self.recovery.as_deref()
    }

    /// Ruling S15 (`surrogate` module doc, "Trust recovery"): the degree-cap retry, then quarantine rounds;
    /// and (M8a user-test fix) a cap an earlier recovery set is re-judged against the default basis.
    /// Returns the report of the fit that stands; a failed attempt restores the previous fit.
    ///
    /// No ratchet (spec 2026-11-13 §4.2): the recovery ends with `!recoverable_fail()` or with the
    /// store's fit, quarantine codes, `degree_cap` and `species_dirty` exactly as on entry — a
    /// recovery that cannot restore the training residuals holds nothing (before it, the samples
    /// its rounds named stayed held and the next refit's rounds started from them, up to the 10 %
    /// cap), so the collection gets no status record from it either (`sync_collection_status`
    /// finds no code changed). The entry report is returned, the note says so.
    pub(super) fn recover_trust(&mut self, report: FitReport) -> FitReport {
        let body3 = self.basis().is_some_and(|b| b.body3);
        // the state on entry: what a recovery that does not restore the residuals returns to
        let (entry_state, entry_flags, entry_dirty) = (self.save_state(), self.quarantine.clone(), self.species_dirty);
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
                match self.fit_store_deferred() {
                    Ok(r) if !self.recoverable_fail() && !self.trust_needs_samples() => {
                        self.recovery = Some(format!("refit with degree_cap {RECOVERY_DEGREE_CAP} (the bent-molecule setting) restored the training residuals"));
                        return r;
                    }
                    _ => self.restore_state(saved),
                }
            }
        }
        // 1b. a cap set by an earlier recovery is not kept for good (M8a user-test fix, module doc
        // "Trust recovery"): when the capped fit fails, the default basis is tried as step 1 tried
        // the cap — kept if it alone restores the residuals with the trust rule's counts
        let recapped = body3 && self.degree_cap.is_some();
        if recapped {
            let saved = self.save_state();
            self.degree_cap = None;
            self.species_dirty = true;
            match self.fit_store_deferred() {
                Ok(r) if !self.recoverable_fail() && !self.trust_needs_samples() => {
                    self.recovery = Some(format!(
                        "refit with the default basis restored the training residuals (the degree_cap {RECOVERY_DEGREE_CAP} an earlier recovery set no longer did)"
                    ));
                    return r;
                }
                _ => self.restore_state(saved),
            }
        }
        // 2. quarantine the worst offenders on the standing basis, round by round (the
        // leave-one-out ranking first, the training-residual ranking its fallback: spec
        // 2026-11-12 R1, option A)
        let before = self.quarantined();
        // step 3's start (the state on entry: step 1 does not run under a cap, 1b restored its own)
        let pre_rounds = recapped.then(|| self.save_state());
        let (rounds, worst, ranking) = self.quarantine_rounds(report.clone());
        // 3. the capped basis's quarantine could not recover (a misfit the capped fit spreads over
        // the whole store names no offender: the user's H collection, `references.md` "M8a stuck
        // collection"): the same quarantine from the default basis, from the state on entry; it
        // stands only if it recovers (the trust rule, its counts included), else step 2's result
        // does when it passes the residuals, else the entry state (no ratchet, below). Run
        // whenever step 2's result is **not recovered** (sample-recovery fix round 2, I1, the
        // coordinator's ruling) — not only while its residuals or noise fail: a step-2 result that
        // passes its residuals but is short of samples (Q′ grows with the cap) must not suppress
        // the default-basis retry, or option A's leave-one-out-first order could be worse than
        // the ranking before it
        if let Some(saved) = pre_rounds.filter(|_| !self.recovered()) {
            let capped = (self.save_state(), self.quarantine.clone(), rounds.clone(), self.species_dirty, self.recovery.clone());
            self.restore_state(saved);
            self.quarantine = entry_flags.clone();
            self.species_dirty = entry_dirty;
            let entry_held = self.quarantined();
            self.degree_cap = None;
            self.species_dirty = true;
            if let Ok(r) = self.fit_store_deferred() {
                let (r, worst, ranking) = self.quarantine_rounds(r);
                if self.recovered() {
                    self.recovery = Some(format!(
                        "back to the default basis (the degree_cap {RECOVERY_DEGREE_CAP} an earlier recovery set could not be recovered by quarantine): {}",
                        quarantine_words(self.quarantined(), self.quarantined() - entry_held, worst, ranking, Outcome::Recovers)
                    ));
                    return r;
                }
            }
            let (state, q, rep, d, note) = capped;
            self.restore_state(state);
            self.quarantine = q;
            self.species_dirty = d;
            self.recovery = note;
            if !self.recoverable_fail() {
                return self.note_quarantined(before, rep, worst, ranking);
            }
        } else if !self.recoverable_fail() {
            return self.note_quarantined(before, rounds, worst, ranking);
        }
        // no step restored the residuals (step 2 by both rankings, and step 3 when it ran): back
        // to the state on entry (no ratchet, spec 2026-11-13 §4.2) — the codes (a forgotten
        // sample's code 7 among them, untouched: the recovery never writes it), the degree cap and
        // the fit; whatever still fails is the misfit set M of the entry fit (`trust` module doc)
        self.restore_state(entry_state);
        self.quarantine = entry_flags;
        self.species_dirty = entry_dirty;
        self.recovery = Some(RECOVERY_FAILED_NOTE.into());
        report
    }

    /// The trust recovery's verdict on a fit (steps 1, 1b, 3 and the rankings' fallback): the
    /// training residuals and the noise pass, and the trust rule's counts hold.
    fn recovered(&self) -> bool {
        !self.recoverable_fail() && !self.trust_needs_samples()
    }

    /// The failed recovery's note once the refit's verdict is known (spec 2026-11-13 §4.3): a refit
    /// that ends trusted — its misfit set within its limit — did not restore the training
    /// residuals, but it did not fail trust either, so the note says what stays in the fit; one
    /// that ends untrusted keeps [`RECOVERY_FAILED_NOTE`]. Called by `refit` where the trust
    /// verdict and M of the fit that stands are both final, and by `set_policy` after it re-judged
    /// them (either way: a policy change can untrust a settled fit or trust a failed one).
    pub(super) fn settle_recovery_note(&mut self) {
        let failed = self.recovery.as_deref().is_some_and(|r| r == RECOVERY_FAILED_NOTE || r.starts_with(RECOVERY_SETTLED_NOTE));
        if !failed {
            return;
        }
        self.recovery = Some(if self.trusted() {
            let n = self.misfit.as_ref().map_or(0, |m| m.samples.len());
            format!("{RECOVERY_SETTLED_NOTE} {n} samples the model cannot reproduce stay in the fit")
        } else {
            RECOVERY_FAILED_NOTE.into()
        });
    }

    /// Steps 2 and 3 of [`recover_trust`](Self::recover_trust): the quarantine rounds (spec R1, fix
    /// round 1 — the coordinator's option A). First by the **leave-one-out** ratio
    /// ([`worst_offenders`](Self::worst_offenders)); when those rounds do not recover the fit
    /// ([`recovered`](Self::recovered): residuals, noise and the trust rule's counts), the entry
    /// state of this call is restored — the fit, the quarantine codes, `species_dirty`, the note —
    /// and the rounds run again by the **training residual** ([`training_offenders`](Self::training_offenders),
    /// the ranking before 2026-11-12). Their result stands only if it recovers; else the
    /// leave-one-out result does — unless it fails the residuals and the training result passes
    /// them (the open-shell merge, 2026-11-14): `recover_trust` keeps a result that is not
    /// recovered only when it passes the residuals and noise (no ratchet, spec 2026-11-13 §4.2),
    /// so preferring the failing one would discard a result the ranking before R1 kept. So the
    /// recovery is never worse than the training ranking's and keeps the leave-one-out ranking's
    /// wins, for up to [`QUARANTINE_ROUNDS`] more fits on a refit that is failing anyway (with
    /// step 3 gated on "not recovered", fix round 2). Measured: the stuck-H collection's step 3
    /// needs the fallback (leave-one-out 47 held, 1.60×; training 38 held, trusted), a
    /// high-leverage outlier the first (`tests/recovery_loo.rs`). Returns the report that stands,
    /// what its rounds held ([`Held`]) and the ranking whose result stands.
    pub(super) fn quarantine_rounds(&mut self, report: FitReport) -> (FitReport, Held, Ranking) {
        // tests: the training ranking alone, the recovery before spec R1
        #[cfg(test)]
        if self.training_ranking {
            let (r, w) = self.rounds_by(report, Ranking::Training);
            return (r, w, Ranking::Training);
        }
        let entry = (self.save_state(), self.quarantine.clone(), self.species_dirty, self.recovery.clone(), report.clone());
        let (r, w) = self.rounds_by(report, Ranking::LeaveOneOut);
        if self.recovered() {
            return (r, w, Ranking::LeaveOneOut);
        }
        let loo_fails = self.recoverable_fail();
        let loo = (self.save_state(), self.quarantine.clone(), self.species_dirty, self.recovery.clone(), r, w);
        let (state, flags, dirty, note, report) = entry;
        self.restore_state(state);
        self.quarantine = flags;
        self.species_dirty = dirty;
        self.recovery = note;
        let (r, w) = self.rounds_by(report, Ranking::Training);
        if self.recovered() || (loo_fails && !self.recoverable_fail()) {
            return (r, w, Ranking::Training);
        }
        let (state, flags, dirty, note, r, w) = loo;
        self.restore_state(state);
        self.quarantine = flags;
        self.species_dirty = dirty;
        self.recovery = note;
        (r, w, Ranking::LeaveOneOut)
    }

    /// The quarantine rounds by one `ranking`: holds the round's offenders and refits without them,
    /// up to [`QUARANTINE_ROUNDS`] rounds, never past [`QUARANTINE_MAX_FRACTION`] of the store held
    /// and forgotten together (spec R2) —
    /// the cap is checked before a round holds anything, so a round whose offenders would cross it
    /// (many undefined leave-one-out residuals, all ranked ∞, among them) holds none of them and
    /// ends the rounds; a failed fit restores the previous one and stops. Returns the report that
    /// stands and what it held ([`Held`]).
    fn rounds_by(&mut self, report: FitReport, ranking: Ranking) -> (FitReport, Held) {
        let mut report = report;
        let mut held = Held::NONE;
        let n_store = self.samples();
        for _ in 0..QUARANTINE_ROUNDS {
            if !self.recoverable_fail() {
                break;
            }
            // the bound both rankings rank against this round (their own `absolute` switch) and,
            // in the noise-only mode, the channel that attains the worst ratio (re-review N1; the
            // in-sample energy passes then, so the training ranking's worst is its force channel)
            // — per held sample, its ratio and the bound it is held under (fix round 1, M6)
            let (worst, measured): (Vec<usize>, Vec<(f64, Bound)>) = match ranking {
                Ranking::LeaveOneOut => {
                    let (rows, _) = self.worst_offenders_bound();
                    (rows.iter().map(|x| x.0).collect(), rows.iter().map(|x| (x.1, x.2)).collect())
                }
                Ranking::Training => {
                    // every one of them is within the round's worst, against the one bound
                    let (w, r) = self.training_offenders();
                    let b = if self.residuals_fail() { Bound::Own } else { Bound::ForceTier(self.policy.force_std_abs) };
                    (w, vec![(r, b)])
                }
            };
            // forgotten samples count with the held ones (spec R2: forgetting is no way round the cap)
            let total = self.quarantined() + self.forgotten() + worst.len();
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
            match self.fit_store_deferred() {
                Ok(r) => {
                    report = r;
                    for (ratio, bound) in measured {
                        held.add(ratio, bound);
                    }
                }
                Err(_) => {
                    self.restore_state(saved);
                    self.quarantine = flags;
                    break;
                }
            }
        }
        (report, held)
    }

    /// The recovery note of quarantine rounds that added samples since `before` held; `worst` is
    /// what they held by `ranking`, the one whose result stands.
    fn note_quarantined(&mut self, before: usize, report: FitReport, worst: Held, ranking: Ranking) -> FitReport {
        let added = self.quarantined().saturating_sub(before);
        if added > 0 {
            let outcome = if self.recovered() {
                Outcome::Recovers
            } else if !self.recoverable_fail() {
                Outcome::Passes
            } else {
                Outcome::Fails
            };
            self.recovery = Some(quarantine_words(self.quarantined(), added, worst, ranking, outcome));
        }
        report
    }

    /// Whether the model reproduces (energy, forces) at a geometry within the policy's thresholds.
    pub(super) fn reproduces(&self, pos: &[[f64; 3]], z: &[u8], energy: f64, forces: &[[f64; 3]]) -> bool {
        let Some(model) = self.model.as_ref() else { return false };
        let Ok(pr) = model.predict(pos, z, false) else { return false };
        Self::agree(&self.policy, pos.len(), pr.energy, &pr.forces, energy, forces)
    }

    /// [`reproduces`](Self::reproduces) for a stored sample, on the rows it was fitted with (M8a Task
    /// 10, a capped segment's: no energy when unused, the interior atoms' forces only — its caps
    /// and edge atoms are no answer the model is held to).
    pub(super) fn reproduces_sample(&self, s: &Sample) -> bool {
        if s.is_whole() {
            return self.reproduces(&s.pos, &s.z, s.energy, &s.forces);
        }
        let Some(model) = self.model.as_ref() else { return false };
        let Ok(pr) = model.predict(&s.pos, &s.z, false) else { return false };
        let p = &self.policy;
        (!s.use_energy || (pr.energy - s.energy).abs() / s.pos.len().max(1) as f64 <= p.energy_std_per_atom)
            && pr.forces.iter().zip(&s.forces).enumerate().filter(|(j, _)| s.trains_atom(*j)).all(|(_, (a, b))| {
                let (d, bound) = p.force_residual(a, b);
                d <= bound
            })
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
            if self.excluded(k) || self.reproduces_sample(&s) {
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
            .filter(|&i| store.samples.get(i).is_some_and(|s| self.reproduces_sample(s)))
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

    /// (index, distance) of every fencing sample — held quarantined or forgotten by the user
    /// ([`fences`](Self::fences)) — from the geometry (`near_quarantined`). Fills the descriptor
    /// cache, then reads it ([`quarantine_distances_cached`](Self::quarantine_distances_cached)).
    pub(super) fn quarantine_distances(&mut self, pos: &[[f64; 3]], z: &[u8]) -> Vec<(usize, f64)> {
        if self.fencing() == 0 {
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
        if self.fencing() == 0 {
            return Vec::new();
        }
        let (Some(model), Some(env), Some(store)) = (self.model.as_ref(), self.envelope.as_ref(), self.store.as_ref()) else { return Vec::new() };
        let basis = &model.basis;
        let (Ok(desc), Ok(sp)) = (describe_values(basis, pos, z), basis.species_indices(z)) else { return Vec::new() };
        let cached = self.desc_basis.as_ref() == Some(basis);
        (0..store.len())
            .filter(|&i| self.fences(i))
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

    /// [`near_misfit_atoms`](Self::near_misfit_atoms) after filling the descriptor cache while M
    /// has misfit atoms (`ensure_desc_cache`): the step's path (`evaluate`, the audit's judgement),
    /// which would otherwise describe M's samples afresh at every call.
    pub(super) fn near_misfit(&mut self, pos: &[[f64; 3]], z: &[u8]) -> Vec<bool> {
        if self.misfit_fence_stands() {
            self.ensure_desc_cache();
        }
        self.near_misfit_atoms(pos, z)
    }

    /// Whether the misfit fence stands (`misfit_fence`: M has misfit atoms, within its limit).
    pub(super) fn misfit_fence_stands(&self) -> bool {
        self.misfit_fence().is_some()
    }

    /// Which query atoms are **near a misfit** (spec 2026-11-13 §4.3; `trust` module doc, "M, the
    /// misfit set"): one entry per atom of (`pos`, `z`), true when the atom's descriptor is within
    /// the near-quarantine distance (`QUARANTINE_FRACTION × max_descriptor_distance` training
    /// spreads, [`spread_distance`] — the per-atom distance the quarantine fence is built from) of a
    /// misfit atom of its species ([`Misfit::atoms`]). Per atom, not the worst atom: the DFT
    /// solvers refuse a geometry when **any** atom is near, `atom_verdicts` refuses exactly those.
    /// Read-only: the misfit atoms are read from the descriptor cache when it covers the model's
    /// basis (`ensure_desc_cache`), else described on the fly, as
    /// [`quarantine_distances_cached`](Self::quarantine_distances_cached) does — the same numbers
    /// either way. All false without a misfit fence (`misfit_fence`: no misfit atom, or M beyond
    /// its limit — the model is then untrusted as a whole; described nothing), a model or an
    /// envelope. As in the quarantine fence, a query or misfit sample that cannot be described
    /// and a NaN envelope unit read as "far" (unreachable for known species on a finite fit).
    pub(super) fn near_misfit_atoms(&self, pos: &[[f64; 3]], z: &[u8]) -> Vec<bool> {
        let none = vec![false; pos.len()];
        let Some(m) = self.misfit_fence() else { return none };
        let (Some(model), Some(env), Some(store)) = (self.model.as_ref(), self.envelope.as_ref(), self.store.as_ref()) else { return none };
        let basis = &model.basis;
        let (Ok(desc), Ok(sp)) = (describe_values(basis, pos, z), basis.species_indices(z)) else { return none };
        let limit = QUARANTINE_FRACTION * self.policy.max_descriptor_distance;
        let cached = self.desc_basis.as_ref() == Some(basis);
        // the misfit atoms as (species index, descriptor), each sample described once
        let mut targets: Vec<(usize, &[f64])> = Vec::with_capacity(m.atoms.len());
        let mut owned: Vec<(usize, Vec<(usize, Vec<f64>)>)> = Vec::new();
        for &i in &m.samples {
            if !(cached && self.desc_cache.get(i).is_some()) {
                if let Some(s) = store.samples.get(i) {
                    owned.push((i, describe_sample(basis, s)));
                }
            }
        }
        for &(i, j) in &m.atoms {
            let atoms: Option<&[(usize, Vec<f64>)]> = match self.desc_cache.get(i).filter(|_| cached) {
                Some(a) => Some(a),
                None => owned.iter().find(|(k, _)| *k == i).map(|(_, a)| &a[..]),
            };
            if let Some((s, d)) = atoms.and_then(|a| a.get(j)) {
                targets.push((*s, d));
            }
        }
        sp.iter()
            .zip(&desc)
            .map(|(qs, qd)| targets.iter().any(|(s, d)| s == qs && spread_distance(env, *s, qd, d) <= limit))
            .collect()
    }

    /// The recovery's offenders (spec R1, since 2026-11-12): the fitted samples (not excluded — at
    /// the fit or since — and not box-affected) whose **leave-one-out ratio** ([`loo_ratio`]: how far the fit *without* the
    /// sample misses it — the per-atom energy over its threshold or the worst atom's force over
    /// its bound) exceeds 1 and is within a factor 2 of the worst such ratio; and that worst ratio
    /// (0 without a fit). The force bound is the policy's per-atom bound while the training
    /// residuals fail; when only the force noise does (σ_f above `force_std_abs`, the trust rule's
    /// noise condition) it is the absolute `force_std_abs` (ρ_σ). An undefined leave-one-out is ∞
    /// (ruling R1a): such samples rank first, and only they are taken while one exists.
    ///
    /// Why not the training residual first (the ranking before; [`training_offenders`](Self::training_offenders),
    /// now the fallback of [`quarantine_rounds`](Self::quarantine_rounds)): a high-leverage sample is reproduced at its neighbours'
    /// expense and sits inside its own bound. The user's collection (2026-11-12): a compressed OH
    /// at 0.72 Ha/Bohr, inside its 1e-2 cap, bent the shared O–H pair term and 34 H₂O/H₃O samples
    /// failed; ranked by their own residuals the rounds held those victims, round after round, and
    /// never the sample. Its leave-one-out ratio (38.5× in force, 962× in energy) is rank 1.
    ///
    /// The ratios are the class pass's for the fit that stands when it ran (ruling R1b), else
    /// computed here once for this fit ([`ensure_loo_ratios`](Self::ensure_loo_ratios); the fits
    /// inside a refit defer the class pass).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn worst_offenders(&mut self) -> (Vec<usize>, f64) {
        let (rows, w) = self.worst_offenders_bound();
        (rows.into_iter().map(|x| x.0).collect(), w)
    }

    /// [`worst_offenders`](Self::worst_offenders) with each offender's ratio and what it was
    /// measured against ([`Bound`]: its own bound, or in the noise-only mode the tier of the
    /// channel that attains its ratio — re-review N1, fix round 1 M6), and the worst ratio.
    pub(super) fn worst_offenders_bound(&mut self) -> (Vec<(usize, f64, Bound)>, f64) {
        let absolute = !self.residuals_fail();
        let (e_tier, f_tier) = (self.policy.energy_std_per_atom, self.policy.force_std_abs);
        let Some(c) = self.ensure_loo_ratios() else { return (Vec::new(), 0.0) };
        let ratios: Vec<(usize, f64, Channel)> = c
            .rows
            .iter()
            .zip(&c.tier)
            .zip(&c.tier_channel)
            .map(|((&(i, own), &tier), &ch)| if absolute { (i, tier, ch) } else { (i, own, Channel::Force) })
            .collect();
        // the standing fit's members, less any sample excluded since that fit (a hold, a forget:
        // fix round 2, M1 — never named twice, never counted twice against the cap) and the
        // box-affected ones
        let ratios: Vec<(usize, f64, Channel)> = ratios.into_iter().filter(|&(i, _, _)| !self.excluded(i) && !self.is_box(i)).collect();
        // the worst as `f64::max` folds it (a NaN cannot occur: `loo_ratio` maps it to ∞)
        let worst = ratios.iter().map(|x| x.1).fold(0.0, f64::max);
        let bound = |ch: Channel| match (absolute, ch) {
            (false, _) => Bound::Own,
            (true, Channel::Energy) => Bound::EnergyTier(e_tier),
            (true, Channel::Force) => Bound::ForceTier(f_tier),
        };
        if worst <= 1.0 {
            return (Vec::new(), worst);
        }
        (ratios.into_iter().filter(|&(_, r, _)| r > 1.0 && r >= 0.5 * worst).map(|(i, r, ch)| (i, r, bound(ch))).collect(), worst)
    }

    /// The standing fit's members in store order (`fit_members`: the accumulated prefix and the
    /// quarantine codes when it was installed, the samples with no code). Right after a fit they are
    /// the class pass's and the training residuals' members; between fits a code may move without
    /// the fit (a hold, a release, a forget), and these stay the fit's.
    fn fit_member_indices(&self) -> impl Iterator<Item = usize> + '_ {
        let (n, codes) = (&self.fit_members.0, &self.fit_members.1);
        let n = self.store.as_ref().map_or(0, |st| (*n).min(st.len()));
        (0..n).filter(move |&i| codes.get(i).is_none_or(|&q| q == Q_NONE))
    }

    /// The per-sample leave-one-out ratios of the fit that stands, when the cache holds this fit's
    /// ([`LooRatios`]: the same model generation and the fit's own members); `None` when stale or
    /// without a model. Read-only, for a caller that cannot compute (`&self`): every refit leaves
    /// the cache fresh (`refit_inner`), and it stays fresh until the next fit.
    pub(super) fn fresh_loo_ratios(&self) -> Option<&LooRatios> {
        let c = self.loo_ratios.as_ref()?;
        (self.model.is_some() && c.gen == self.model_gen && c.rows.iter().map(|r| r.0).eq(self.fit_member_indices())).then_some(c)
    }

    /// [`fresh_loo_ratios`](Self::fresh_loo_ratios), computing them for the fit that stands when the
    /// cache is stale (`surrogate::loo::leave_one_out_many` over the members, then [`loo_ratio`] —
    /// the class pass's numbers, bit for bit). `None` without a model or a store.
    pub(super) fn ensure_loo_ratios(&mut self) -> Option<&LooRatios> {
        if self.fresh_loo_ratios().is_none() {
            let (model, store) = (self.model.as_ref()?, self.store.as_ref()?);
            let members: Vec<usize> = self.fit_member_indices().collect();
            let samples: Vec<&Sample> = members.iter().map(|&i| &store.samples[i]).collect();
            let loos = surrogate::loo::leave_one_out_many(model, &samples);
            let (mut rows, mut tier, mut tier_channel) = (Vec::with_capacity(members.len()), Vec::with_capacity(members.len()), Vec::with_capacity(members.len()));
            for ((&i, s), l) in members.iter().zip(&samples).zip(&loos) {
                let (own, t, ch) = loo_ratio(&self.policy, model.energy_per_atom, s, l.as_ref());
                rows.push((i, own));
                tier.push(t);
                tier_channel.push(ch);
            }
            self.loo_ratios = Some(LooRatios { gen: self.model_gen, rows, tier, tier_channel });
        }
        self.loo_ratios.as_ref()
    }

    /// **The per-sample leave-one-out ranking's accessor** (spec R1b; the Model tab's
    /// "Worst-fitting samples", plan Task 2): per fitted sample of the fit that stands, in store
    /// order, (store index, leave-one-out ratio ρ). ρ is [`loo_ratio`]'s first number: the larger
    /// of the per-atom energy residual over `energy_std_per_atom` and the worst atom's force
    /// residual over that atom's own bound `force_bound(|F|)`, both of the fit without the sample.
    /// `f64::INFINITY` stands for an undefined residual (the sample alone spans a direction of the
    /// fit, ruling R1a). The rows are the standing fit's members: a sample held or forgotten since
    /// that fit is still listed until the next fit (filter by its current code), one released since
    /// is not. Computed when the cache is stale (another fit since); `None` without a model or a
    /// store. Not sorted: rank by the ratio. ([`fresh_loo_ratios`](Self::fresh_loo_ratios) is the
    /// read-only form.)
    pub fn loo_ratios(&mut self) -> Option<&[(usize, f64)]> {
        self.ensure_loo_ratios().map(|c| &c.rows[..])
    }

    /// The offender ranking before 2026-11-12 (spec R1): the training samples (not excluded, not
    /// box-affected) whose **in-sample** residual ratio — the per-atom energy residual over its
    /// threshold, or the worst atom's force residual over its bound (`force_std_abs` when only the
    /// force noise fails) — exceeds 1 and is within a factor 2 of the worst, and that worst. The
    /// fallback ranking of [`quarantine_rounds`](Self::quarantine_rounds) (fix round 1, option A):
    /// it names the samples that fail, where the leave-one-out ranking, once a label error is out,
    /// can go on naming high-leverage samples whose own fit is fine (the stuck-H collection).
    pub(super) fn training_offenders(&self) -> (Vec<usize>, f64) {
        let (Some(model), Some(store)) = (self.model.as_ref(), self.store.as_ref()) else { return (Vec::new(), 0.0) };
        let p = &self.policy;
        let absolute = !self.residuals_fail();
        let ratios: Vec<(usize, f64)> = store.samples[..self.accumulated.min(store.len())]
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.excluded(*i) && !self.is_box(*i))
            .filter_map(|(i, s)| {
                let pr = model.predict(&s.pos, &s.z, false).ok()?;
                let n = s.pos.len().max(1) as f64;
                // the fitted rows only (a capped segment's: no energy, its interior atoms)
                let mut r = if s.use_energy { (pr.energy - s.energy).abs() / n / p.energy_std_per_atom } else { 0.0 };
                for (_, (fm, fs)) in pr.forces.iter().zip(&s.forces).enumerate().filter(|(j, _)| s.trains_atom(*j)) {
                    let (d, bound) = p.force_residual(fm, fs);
                    r = r.max(d / if absolute { p.force_std_abs } else { bound });
                }
                Some((i, r))
            })
            .collect();
        let worst = ratios.iter().map(|x| x.1).fold(0.0, f64::max);
        if worst <= 1.0 {
            return (Vec::new(), worst);
        }
        (ratios.into_iter().filter(|&(_, r)| r > 1.0 && r >= 0.5 * worst).map(|(i, _)| i).collect(), worst)
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
        // a capped segment's interior atoms only: the active set is what the model has learned,
        // and a cap or an edge atom is not (M8a Task 10; the cache keeps every atom, for the
        // fences' nearness — the conservative side there)
        let store = self.store.as_ref();
        let trains = |i: usize, j: usize| store.and_then(|st| st.samples.get(i)).is_none_or(|s| s.trains_atom(j));
        let atoms = (0..n).filter(|&i| !self.excluded(i)).flat_map(|i| self.desc_cache[i].iter().enumerate().filter(move |(j, _)| trains(i, *j)).map(|(_, (s, b))| (*s, &b[..])));
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
    /// strict mode the region (round 1 re-review, R1-1) — and so are lone atoms (`Q_LONE`: never
    /// data). ∞ without a model, an envelope or a candidate.
    pub(super) fn nearest_stored(&mut self, pos: &[[f64; 3]], z: &[u8]) -> f64 {
        self.ensure_desc_cache();
        let (Some(basis), Some(env)) = (self.desc_basis.as_ref(), self.envelope.as_ref()) else { return f64::INFINITY };
        let (Ok(desc), Ok(sp)) = (describe_values(basis, pos, z), basis.species_indices(z)) else { return f64::INFINITY };
        let mut worst_atom = 0.0f64;
        for (qs, qd) in sp.iter().zip(&desc) {
            let mut best = f64::INFINITY;
            for (i, atoms) in self.desc_cache.iter().enumerate() {
                if matches!(self.quarantine.get(i), Some(&Q_DRAFT) | Some(&Q_LONE)) {
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
    /// fencing it; `near_misfit` whether an atom is near a misfit (`near_misfit_atoms`); `keep`
    /// the rule's own verdict; `thin` whether thinning applies. The
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
        near_misfit: bool,
        keep: bool,
        thin: bool,
        dedup: bool,
        essential: bool,
        h: f64,
        mask: Option<&[bool]>,
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
            // what fences here, in words (fix round 1, M4: a forgotten sample is not "quarantined")
            let forgotten_near = near.iter().filter(|&&i| self.is_forgotten(i)).count();
            let (geometry, samples) = match (forgotten_near, near.len() - forgotten_near) {
                (0, _) => ("a quarantined geometry", "quarantined samples"),
                (_, 0) => ("a forgotten sample", "forgotten samples"),
                _ => ("quarantined and forgotten samples", "quarantined and forgotten samples"),
            };
            let agrees = model.is_some_and(|(e, f)| Self::agree(&self.policy, pos.len(), e, f, energy, forces));
            if !agrees {
                // a disagreeing answer ends a run of agreeing ones toward lifting a forgotten fence
                self.lift_votes.clear();
                // fence voting (`surrogate` module doc, "Fence voting"): fresh answers that agree with each
                // other but not with the model are evidence the model is wrong here
                let n = pos.len();
                let residual = model.map(|(e, f)| (energy - e, forces.iter().zip(f).map(|(t, m)| [t[0] - m[0], t[1] - m[1], t[2] - m[2]]).collect()));
                self.fence_votes.push(FenceVote { near: near.to_vec(), n, energy, forces: forces.to_vec(), residual, gen: self.model_gen });
                if self.fence_votes.len() > FENCE_VOTES {
                    self.fence_votes.remove(0);
                }
                // the last k answers: the same fence, mutually consistent (every pair agrees
                // within the per-atom energy threshold and every atom's force bound)
                let votes = &self.fence_votes;
                let consistent = votes.len() == FENCE_VOTES
                    && votes.iter().all(|v| v.n == n && v.gen == self.model_gen && v.near.iter().any(|i| near.contains(i)))
                    && votes.iter().enumerate().all(|(a, va)| {
                        votes[a + 1..].iter().all(|vb| votes_agree(&self.policy, va, vb))
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
                        "surrogate: kept — {FENCE_VOTES} mutually consistent fresh teacher answers next to {geometry} disagree with the model; the next refit decides"
                    ));
                    return yes(TeachOutcome::KeptByVote);
                }
                // no model answered in a capped segment (`TeachRule::Segment`) or wherever a rule
                // refused before consulting it: nothing disagreed (Task 10a review, finding 7)
                notes.push(match (model, mask) {
                    (Some(_), _) => format!("surrogate: sample not kept — next to {geometry}, and the teacher still disagrees with the model there"),
                    (None, Some(_)) => format!("surrogate: sample not kept — a capped segment next to {geometry} (the model does not answer in a segment); a fence vote keeps one once {FENCE_VOTES} mutually consistent teacher answers agree here"),
                    (None, None) => format!("surrogate: sample not kept — next to {geometry}, and the model was not consulted here; a fence vote keeps one once {FENCE_VOTES} mutually consistent teacher answers agree here"),
                });
                return no(TeachOutcome::FenceDisagrees);
            }
            self.fence_votes.clear();
            let (mut released, mut superseded) = (0, 0);
            // a forgotten sample fences but is never released or superseded: only the user's
            // Restore clears it (spec R2)
            let movable: Vec<usize> = near.iter().copied().filter(|&i| !self.is_forgotten(i)).collect();
            let kept_forgotten = near.len() - movable.len();
            // only forgotten samples fence here (final review I1): nothing is released or
            // superseded, so the fence stays and the model goes on refusing here for as long as
            // they stay forgotten — the answer is an ordinary agreeing one and goes to the
            // thinning and copy checks below (the caller thins and deduplicates a fence's
            // answers, `evaluate`'s `by_fence`), instead of a new sample at every step
            // a vote counts only when the model that agreed was fitted without every forgotten
            // sample here (review Critical 1: a Forget sets the code at once, the refit runs later
            // — a model still trained on the sample agrees next to it by construction)
            let blocked = if !movable.is_empty() {
                Some("a quarantined sample fences here too")
            } else if self.pending_refit {
                Some("a refit is due — the model that agreed may still hold them")
            } else if self.background.is_some() {
                Some("a refit is running")
            } else if near.iter().any(|&i| self.in_standing_fit(i)) {
                Some("the model that agreed was fitted with them — they leave it at the next refit")
            } else {
                None
            };
            if let Some(why) = blocked {
                // the run of agreeing answers must be consecutive: anything else starts it over
                self.lift_votes.clear();
                if movable.is_empty() {
                    notes.push(format!(
                        "surrogate: a fresh teacher answer near forgotten samples agrees with the model — {kept_forgotten} forgotten kept (only Restore brings a forgotten sample back); no vote toward lifting their fence: {why}; the answer goes to its rule's thinning and copy checks"
                    ));
                }
            } else {
                // lifting the fence (`surrogate` module doc, "Lifting a forgotten sample's fence"):
                // an agreeing vote; FENCE_VOTES consecutive, mutually consistent ones next to the
                // same forgotten samples under one model lift the fence of the samples every one
                // of them was next to, and keep the last answer
                let n = pos.len();
                let residual = model.map(|(e, f)| (energy - e, forces.iter().zip(f).map(|(t, m)| [t[0] - m[0], t[1] - m[1], t[2] - m[2]]).collect()));
                self.lift_votes.push(FenceVote { near: near.to_vec(), n, energy, forces: forces.to_vec(), residual, gen: self.model_gen });
                if self.lift_votes.len() > FENCE_VOTES {
                    self.lift_votes.remove(0);
                }
                let votes = &self.lift_votes;
                // the forgotten samples every vote of the run was next to (review Important 2: the
                // approved rule is "3 answers near it", so never one only a single vote was near)
                let common: Vec<usize> = near.iter().copied().filter(|i| self.quarantine.get(*i) == Some(&Q_FORGOTTEN) && votes.iter().all(|v| v.near.contains(i))).collect();
                let consistent = votes.len() == FENCE_VOTES
                    && !common.is_empty()
                    && votes.iter().all(|v| v.n == n && v.gen == self.model_gen)
                    && votes.iter().enumerate().all(|(a, va)| votes[a + 1..].iter().all(|vb| votes_agree(&self.policy, va, vb)));
                if consistent {
                    let mut lift = common;
                    lift.sort_unstable();
                    lift.dedup();
                    self.lift_votes.clear();
                    let mut failed: Option<String> = None;
                    for &i in &lift {
                        self.quarantine[i] = Q_FORGOTTEN_UNFENCED;
                        self.note_user_code(i, Q_FORGOTTEN_UNFENCED);
                        if let Err(e) = self.write_status_now(i) {
                            failed.get_or_insert(e);
                        }
                    }
                    if let Some(e) = failed {
                        self.note_write_failure(&e);
                    }
                    let k = lift.len();
                    notes.push(format!(
                        "surrogate: kept — {FENCE_VOTES} mutually consistent fresh teacher answers near forgotten samples agree with the model: the fence of {k} forgotten sample{} is lifted (still forgotten and out of the fit — Restore brings {} back); the model answers here again",
                        if k == 1 { "" } else { "s" },
                        if k == 1 { "it" } else { "them" }
                    ));
                    return yes(TeachOutcome::FenceLifted { lifted: k as u32 });
                }
                notes.push(format!(
                    "surrogate: a fresh teacher answer near forgotten samples agrees with the model — {kept_forgotten} forgotten kept (only Restore brings a forgotten sample back); agreeing answer {} of {FENCE_VOTES} toward lifting their fence; the answer goes to its rule's thinning and copy checks",
                    self.lift_votes.len()
                ));
            }
            if !movable.is_empty() {
                for i in movable {
                    let s = self.store.as_ref().expect("a quarantined sample is stored").samples[i].clone();
                    if self.reproduces_sample(&s) {
                        self.quarantine[i] = self.released_code(i);
                        released += 1;
                    } else {
                        self.quarantine[i] = Q_SUPERSEDED;
                        superseded += 1;
                    }
                }
                if released + superseded > 0 {
                    self.species_dirty = true; // the fit's sample set changed: rebuild
                }
                notes.push(format!(
                    "surrogate: a fresh teacher answer near {samples} agrees with the model — {released} released, {superseded} superseded{}",
                    if kept_forgotten > 0 { format!(", {kept_forgotten} forgotten kept (only Restore brings a forgotten sample back)") } else { String::new() }
                ));
                return yes(TeachOutcome::KeptReleased { released, superseded });
            }
        }
        // next to a training sample the model cannot reproduce (spec 2026-11-13 §4.3): the
        // near-quarantine admission without its votes — kept (on the normal path below) only when
        // it agrees with the model, so a region the model cannot fit stays refused without a parked
        // scene growing M; it is learned when a later refit fits M's samples
        if near_misfit {
            let agrees = model.is_some_and(|(e, f)| Self::agree(&self.policy, pos.len(), e, f, energy, forces));
            if !agrees {
                notes.push(match model {
                    Some(_) => "surrogate: sample not kept — next to a training sample the model cannot reproduce, and the teacher disagrees with the model there (a later refit at more samples or a higher body-4 degree may fit it)".into(),
                    None => "surrogate: sample not kept — next to a training sample the model cannot reproduce, and the model was not consulted here (a later refit at more samples or a higher body-4 degree may fit it)".into(),
                });
                return no(TeachOutcome::MisfitDisagrees);
            }
        }
        // thinning (`surrogate` module doc, "Trust recovery"): audits and the samples trust needs are
        // kept — but never a copy where copies could satisfy the rule (`is_label_copy`; round 1 of
        // the 2026-10-18 review, MUST-2)
        if thin || dedup {
            let d = self.nearest_stored(pos, z);
            let limit = THIN_FRACTION * self.policy.max_descriptor_distance;
            if dedup && d <= limit {
                if let Some(i) = self.is_label_copy(pos, z, energy, forces, mask) {
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
    ///
    /// `mask` is the answer's training mask (a capped segment's, `Scene::train_mask`; `None` for a
    /// whole answer). Only stored samples fitted with the same rows are candidates (M8a Task 10a
    /// review, finding 4): the same atoms taught and the energy used or not alike — a whole answer
    /// is never a copy of a segment, nor a segment of one cut with another mask. The labels are
    /// compared on those fitted rows only: the masked-out atoms' forces are not compared, and with
    /// no energy row the energy test is skipped (a segment's E includes its caps; the fit never
    /// reads it).
    pub(super) fn is_label_copy(&self, pos: &[[f64; 3]], z: &[u8], energy: f64, forces: &[[f64; 3]], mask: Option<&[bool]>) -> Option<usize> {
        let (model, env, store, report) = (self.model.as_ref()?, self.envelope.as_ref()?, self.store.as_ref()?, self.report.as_ref()?);
        let basis = &model.basis;
        let (Ok(desc), Ok(sp)) = (describe_values(basis, pos, z), basis.species_indices(z)) else { return None };
        let cached = self.desc_basis.as_ref() == Some(basis);
        // the answer's fitted rows (as `store::keep_sample_masked` will store it)
        let mask = mask.filter(|m| m.len() == z.len());
        let use_energy = mask.is_none();
        let taught = |j: usize| mask.is_none_or(|m| m[j]);
        let (mut best, mut best_d) = (None, f64::INFINITY);
        for (i, s) in store.samples.iter().enumerate() {
            // only a sample that is (or will be) fitted: the rationale — fitted copies shrink s_F,
            // copies meet the counts — concerns the fit (round 1 re-review, R1-1: a held-out guess
            // compared against would refuse the very fallbacks that let strict mode reach it);
            // and only one fitted with the same rows (finding 4 above)
            if s.z != z || self.excluded(i) || s.use_energy != use_energy || (0..z.len()).any(|j| s.trains_atom(j) != taught(j)) {
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
        let same = (!use_energy || (energy - s.energy).abs() <= s_e)
            && forces.iter().zip(&s.forces).enumerate().filter(|(j, _)| taught(*j)).all(|(_, (a, b))| (mag(a) - mag(b)).abs() <= report.sigma_f);
        same.then_some(i)
    }

    /// The identity of stored sample `i` the Model tab's buttons carry ([`sim::SampleId`]; fix
    /// round 1, M8): its index and [`sample_key`] of its Z and the bits of its positions, energy
    /// and forces. `None` for no such sample.
    pub fn sample_id(&self, i: usize) -> Option<sim::SampleId> {
        let s = self.store.as_ref()?.samples.get(i)?;
        Some(sim::SampleId { index: i, key: sample_key(s) })
    }

    /// The stored sample `id` names, when it still does (fix round 1, M8: a model switch, attach or
    /// load queued before the click can put another sample at that index); else the refusal.
    fn check_id(&self, id: sim::SampleId) -> Result<usize, String> {
        let n = self.samples();
        match self.sample_id(id.index) {
            None => Err(format!("no sample #{} (the model holds {n}) — the sample list changed — pick it again", id.index)),
            Some(now) if now.key != id.key => Err(format!("sample #{} is another sample now — the sample list changed — pick it again", id.index)),
            Some(_) => Ok(id.index),
        }
    }

    /// **Forgets** stored sample `id` (spec R2; the Model tab's "Worst-fitting samples" › Forget):
    /// its code becomes [`Q_FORGOTTEN`] — out of the fit, the envelope, the grade and the
    /// leave-one-out, fencing, never released or superseded. Only a sample whose code is
    /// [`Q_NONE`] (in the fit, or kept since it) can be forgotten (fix round 1, M3: a held,
    /// superseded, box-excluded, draft, waiting or lone sample is refused — so Restore, which
    /// returns it to the fit, restores exactly what it was). The status is written **at once**: a
    /// status record in the attached collection when it holds the sample
    /// ([`write_status_now`](Self::write_status_now)), else the store's code, which the model file
    /// and the session autosave write. A refit is due (`pending_refit`, a rebuild); the caller
    /// starts it. While a background refit runs, the code is also recorded for its install, which
    /// re-applies it over the copy's (`background` module doc, "The user's codes"): the refit that
    /// lands afterwards never brings the sample back. A failed audit whose sample this is stays a
    /// failed audit (rule 8, fix round 1 I1). `Err`, nothing changed: the id no longer names the
    /// sample (M8), or its code is not [`Q_NONE`].
    pub fn forget_sample(&mut self, id: sim::SampleId) -> Result<(), String> {
        let i = self.check_id(id)?;
        let n = self.samples();
        self.forgettable(i)?;
        let written = self.set_user_code(i, Q_FORGOTTEN);
        let out = (self.quarantined() + self.forgotten()) as f64;
        let cap = if out > QUARANTINE_MAX_FRACTION * n as f64 {
            format!(
                "; held and forgotten together are now {} of {n} samples, past the {:.0} % the trust recovery may hold out — it holds no more",
                self.quarantined() + self.forgotten(),
                100.0 * QUARANTINE_MAX_FRACTION
            )
        } else {
            String::new()
        };
        self.user_notes.push(format!(
            "surrogate: sample #{i} forgotten — held out of the fit until Restore, and geometries near it fall back to DFT{written}; the refit decides without it (if it was the only sample of its kind, the trust rule's sample counts may refuse the model until more are sampled){cap}"
        ));
        Ok(())
    }

    /// Whether stored sample `i` may be forgotten: only one whose code is [`Q_NONE`] (fix round 1,
    /// M3); else the refusal.
    fn forgettable(&self, i: usize) -> Result<(), String> {
        match self.quarantine.get(i).copied().unwrap_or(Q_NONE) {
            Q_NONE => Ok(()),
            Q_FORGOTTEN | Q_FORGOTTEN_UNFENCED => Err(format!("sample #{i} is forgotten already")),
            q => Err(format!(
                "sample #{i} is not in the fit ({}): only a sample in the fit can be forgotten",
                match q {
                    Q_HELD => "quarantined by the trust recovery",
                    Q_SUPERSEDED => "superseded by a fresh answer",
                    Q_BOX => "box-excluded",
                    Q_DRAFT => "a draft guess held out of the strict fit",
                    Q_REF_WAIT => "waiting for its box's isolated-atom reference",
                    Q_LONE => "lone atoms, held out for good",
                    _ => "held out of the fit",
                }
            )),
        }
    }

    /// The batch's store indices (2026-11-14, the batched forget): every id checked
    /// ([`check_id`](Self::check_id)) and by `ok`, none twice; the first refusal otherwise —
    /// nothing changed.
    fn batch_indices(&self, ids: &[sim::SampleId], ok: impl Fn(&Self, usize) -> Result<(), String>) -> Result<Vec<usize>, String> {
        if ids.is_empty() {
            return Err("no samples selected".into());
        }
        let mut out: Vec<usize> = Vec::with_capacity(ids.len());
        for &id in ids {
            let i = self.check_id(id)?;
            if out.contains(&i) {
                return Err(format!("sample #{i} is selected twice"));
            }
            ok(self, i)?;
            out.push(i);
        }
        Ok(out)
    }

    /// The words on where a batch's statuses were written: each distinct one of
    /// [`set_user_code`](Self::set_user_code)'s, in order.
    fn batch_written(written: Vec<String>) -> String {
        let mut seen: Vec<String> = Vec::new();
        for w in written {
            if !seen.contains(&w) {
                seen.push(w);
            }
        }
        seen.concat()
    }

    /// "#3, #7, #12" (a batch's note).
    fn index_list(idx: &[usize]) -> String {
        idx.iter().map(|i| format!("#{i}")).collect::<Vec<_>>().join(", ")
    }

    /// **Forgets** every stored sample `ids` names **as one batch** (2026-11-14; the Model tab's
    /// "Forget N selected"): all or nothing — every id is checked first ([`check_id`](Self::check_id),
    /// each must be forgettable as for [`forget_sample`](Self::forget_sample), none twice), and a
    /// batch that would take the held and forgotten samples together past
    /// [`QUARANTINE_MAX_FRACTION`] of the store is **refused whole** with the reason (a single
    /// Forget past the cap is still applied, with [`forget_sample`](Self::forget_sample)'s warning:
    /// one deliberate click; a batch past it is more likely a slip). Each status is written at once,
    /// one refit is due, and one note says what was forgotten. One id is
    /// [`forget_sample`](Self::forget_sample). `Err`, nothing changed.
    pub fn forget_samples(&mut self, ids: &[sim::SampleId]) -> Result<(), String> {
        if let [id] = ids {
            return self.forget_sample(*id);
        }
        let idx = self.batch_indices(ids, |s, i| s.forgettable(i))?;
        let n = self.samples();
        let out = self.quarantined() + self.forgotten() + idx.len();
        if out as f64 > QUARANTINE_MAX_FRACTION * n as f64 {
            return Err(format!(
                "forgetting these {} samples would hold out {out} of {n} (quarantined and forgotten together), past the {:.0} % the trust recovery may hold out — nothing forgotten; forget fewer, or one at a time if you mean to go past it",
                idx.len(),
                100.0 * QUARANTINE_MAX_FRACTION
            ));
        }
        let written: Vec<String> = idx.iter().map(|&i| self.set_user_code(i, Q_FORGOTTEN)).collect();
        self.user_notes.push(format!(
            "surrogate: {} samples forgotten ({}) — held out of the fit until Restore, and geometries near them fall back to DFT{}; the refit decides without them",
            idx.len(),
            Self::index_list(&idx),
            Self::batch_written(written)
        ));
        Ok(())
    }

    /// Puts every forgotten sample `ids` names back **as one batch** (2026-11-14; "Restore N
    /// selected"): all or nothing, as [`forget_samples`](Self::forget_samples) — each must be
    /// forgotten; written at once, one refit due, one note. One id is
    /// [`restore_sample`](Self::restore_sample). `Err`, nothing changed.
    pub fn restore_samples(&mut self, ids: &[sim::SampleId]) -> Result<(), String> {
        if let [id] = ids {
            return self.restore_sample(*id);
        }
        let idx = self.batch_indices(ids, |s, i| if s.is_forgotten(i) { Ok(()) } else { Err(format!("sample #{i} is not forgotten")) })?;
        let written: Vec<String> = idx
            .iter()
            .map(|&i| {
                let code = self.released_code(i);
                self.set_user_code(i, code)
            })
            .collect();
        self.user_notes.push(format!(
            "surrogate: {} samples restored ({}) — back in the fit at the refit{}",
            idx.len(),
            Self::index_list(&idx),
            Self::batch_written(written)
        ));
        Ok(())
    }

    /// Puts a forgotten sample ([`forget_sample`](Self::forget_sample)) back: its code returns to
    /// [`Q_NONE`], the code every forgotten sample had (M3) — through `released_code`, which in
    /// collection mode would give `Q_REF_WAIT` if its box had no reference (it had one when it was
    /// forgotten, and references are never removed); written at once and a refit due, as for a
    /// forget.
    /// `Err`, nothing changed: the id no longer names the sample (M8), or it is not forgotten.
    pub fn restore_sample(&mut self, id: sim::SampleId) -> Result<(), String> {
        let i = self.check_id(id)?;
        if !self.is_forgotten(i) {
            return Err(format!("sample #{i} is not forgotten"));
        }
        let code = self.released_code(i);
        let written = self.set_user_code(i, code);
        self.user_notes.push(format!("surrogate: sample #{i} restored — back in the fit at the refit{written}"));
        Ok(())
    }

    /// Sets the user's code `code` of sample `i` (Forget, Restore): the rebuild is due, the status
    /// written at once, and a running background refit's install told. The note's words on where it
    /// was written.
    fn set_user_code(&mut self, i: usize, code: u8) -> String {
        let n = self.samples();
        self.quarantine.resize(n, Q_NONE);
        self.quarantine[i] = code;
        self.species_dirty = true;
        self.pending_refit = true;
        // one more change no fit holds (`user_changes_pending`: the worker's debounce, the banner)
        self.user_changes += 1;
        self.note_user_code(i, code);
        match self.write_status_now(i) {
            Ok(true) => " (written to the shared collection)".to_string(),
            Ok(false) if self.shared.is_some() => " (a session-only sample: kept with this session's samples)".to_string(),
            Ok(false) => " (saved with the model file)".to_string(),
            Err(e) => {
                self.note_write_failure(&e);
                format!(" (not written to the shared collection: {e} — this session only)")
            }
        }
    }

    /// The worst-fitting samples of the fit that stands (spec R2; the Model tab's "Worst-fitting
    /// samples"): its members ranked by their leave-one-out ratio ([`loo_ratios`](Self::loo_ratios),
    /// ∞ first), worst first, at most [`SUSPECTS_SHOWN`], **only those fitted now** — a sample held,
    /// forgotten or otherwise excluded since that fit is left out. Read from **the fit's own cache
    /// only** ([`fresh_loo_ratios`](Self::fresh_loo_ratios)), never recomputed between refits: a
    /// recompute after a reference rebase would pair the new labels with the old fit (the
    /// `rebase_collection` comment). Empty without a model or a fresh cache (every refit, load and
    /// policy change leaves one). One gap, accepted (fix round 1, M7): a policy change between a
    /// forget and its refit (`set_policy` → `refresh_class_loo`, e.g. the audit cadence) refills the
    /// cache from the live codes, which are no longer the fit's members — the list reads "no
    /// ranking yet" until that refit, the next tick.
    pub fn sample_suspects(&self) -> Vec<sim::SampleSuspect> {
        let Some(c) = self.fresh_loo_ratios() else { return Vec::new() };
        let mut rows: Vec<(usize, f64)> = c.rows.iter().copied().filter(|&(i, _)| !self.excluded(i)).collect();
        rows.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        rows.truncate(SUSPECTS_SHOWN);
        rows.into_iter().filter_map(|(i, r)| self.suspect(i, Some(r))).collect()
    }

    /// The samples the user forgot, in store order, each with its leave-one-out ratio when the fit
    /// that stands still holds it (forgotten since that fit; else `None`).
    pub fn forgotten_samples(&self) -> Vec<sim::SampleSuspect> {
        let cache = self.fresh_loo_ratios();
        (0..self.quarantine.len())
            .filter(|&i| self.is_forgotten(i))
            .filter_map(|i| {
                let r = cache.and_then(|c| c.rows.binary_search_by_key(&i, |x| x.0).ok().map(|k| c.rows[k].1));
                self.suspect(i, r)
            })
            .collect()
    }

    fn suspect(&self, i: usize, loo_ratio: Option<f64>) -> Option<sim::SampleSuspect> {
        let s = self.store.as_ref()?.samples.get(i)?;
        Some(sim::SampleSuspect { id: sim::SampleId { index: i, key: sample_key(s) }, formula: shared::formula(&s.z), origin: origin_word(s.tag).to_string(), loo_ratio })
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
        NO_MODEL_SAMPLE_FACTOR * self.default_basis(&species).column_classes().len()
    }
}
