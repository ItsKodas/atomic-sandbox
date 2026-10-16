//! Plain wording (activity spec §5): codes in, sentences out; the provider's technical text stays
//! on hover. Element symbols from `units` (display only, rule 1); mHa and Å at this edge (rule 3).
use providers::surrogate::{FENCE_VOTES, THIN_EVERY};
use sim::activity::{EnvClass, PassJob, TeachOutcome as K, TeachRule as R, TrustKey};
use units::isotopes::symbol;

/// What a log row (or a stuck streak) knew about its answer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RowCtx {
    /// The teacher's answer was box-affected.
    pub box_affected: bool,
    /// The named atom was within `d_box(h)` of a cell face.
    pub near_face: bool,
    /// The answer was kept as a sample (`TeachOutcome::is_kept`).
    pub kept: bool,
}

/// `box_advice`'s distance: where a hydrogen-like tail e^(−2r)/π falls to the box rule's face
/// density (5.35 Bohr at h = 0.15) — the stated estimate, unchanged.
pub fn d_box(h: f64) -> f64 {
    (1.0 / (std::f64::consts::PI * providers::box_affected_face_density(h))).ln() / 2.0
}

/// Distance (Bohr) from `p` to the nearest face of `scene`'s cell.
pub fn face_distance(scene: &sim::Scene, p: [f64; 3]) -> f64 {
    let half = (scene.cell.n as f64 - 1.0) * scene.cell.h / 2.0;
    half - p.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

/// A sampling pass stage as the chip and the job row word it.
pub fn stage_word(stage: &str) -> &str {
    match stage {
        "references" => "reference",
        "perturbations" => "random nudges",
        "scans" => "bond scan",
        "hot burst" => "hot MD burst",
        s => s,
    }
}

/// The short code of a log row's rule, as the user wrote them (spec §3.3.3). 5d and 5e at γ = ∞
/// are different tests and read differently (plan review S7).
pub fn rule_label(rule: R, burst_t: Option<f64>) -> String {
    match rule {
        R::Cell => "0 cell".into(),
        R::NoModel => "1 no model".into(),
        R::NewSpecies(z) => format!("2 new element {}", symbol(z)),
        R::OtherSettings => "3 other settings".into(),
        R::Untrusted(_) => "4 untrusted".into(),
        R::Class(_) => "4b class".into(),
        R::IsolatedAtom => "5a lone atom".into(),
        R::Coordination => "5b coordination".into(),
        R::Spread => "5c envelope".into(),
        R::NewEnvironment => "5d new environment".into(),
        R::UnspannedAtom => "5e new environment".into(),
        R::NoGrade | R::Grade => "5e grade".into(),
        R::ForceUncertainty => "6 force σ".into(),
        R::EnergyUncertainty => "7 energy σ".into(),
        R::NonFiniteAnswer => "non-finite".into(),
        R::NearQuarantine => "fence".into(),
        R::Audit => "8 audit".into(),
        R::RelaxVerification => "relax check".into(),
        R::Pass(PassJob::Reference { z, multiplicity }) => format!("reference {} m{multiplicity}", symbol(z)),
        R::Pass(PassJob::Perturbation) => "nudge".into(),
        R::Pass(PassJob::Scan) => "bond scan".into(),
        R::Pass(PassJob::HotBurst) => match burst_t {
            Some(t) => format!("hot burst {t:.0} K"),
            None => "hot burst".into(),
        },
    }
}

/// A trust condition by its key alone: a log row is history, so it never shows today's numbers
/// (plan review S14; the numbers at the time are in the row's technical text).
pub fn untrusted_title(key: TrustKey) -> &'static str {
    match key {
        TrustKey::Samples => "too few samples in the fit",
        TrustKey::ResidualDofE | TrustKey::ResidualDofF => "too little data beyond the fit's own parameters",
        TrustKey::EnergyResidual => "it did not reproduce its own training energies",
        TrustKey::ForceResidual => "it did not reproduce its own training forces",
        TrustKey::ForceNoise => "its forces did not fit the data to within the force tier",
        TrustKey::Audit | TrustKey::AuditForces => "the last check against DFT had failed",
    }
}

const OUTSIDE: &str = "is outside what the model has learned";

/// Why the teacher answered, in a sentence (spec §5.1). Every clause that could be false is
/// conditioned on the facts in `cx` (plan review S13).
pub fn rule_sentence(rule: R, atom: Option<usize>, cx: RowCtx) -> String {
    let a = |i: Option<usize>| i.map_or("an atom".to_string(), |i| format!("atom {i}"));
    // the box clause; `subject`: the sentence's subject is the atom itself (the user's sentence)
    let with_box = |s: String, subject: bool| match (cx.box_affected, cx.near_face) {
        (true, true) if subject => format!("{s} and is touching the box edge — move it inward"),
        (true, true) => format!("{s}, and {} is touching the box edge — move it inward", a(atom)),
        (true, false) if cx.kept => format!("{s} — this answer also touches the box walls, so it stays in the fit only until enough others exist; a bigger cell would help"),
        (true, false) => format!("{s} — this answer also touches the box walls, so it is not learned here; a bigger cell would help"),
        _ => s,
    };
    match rule {
        R::Cell => format!("{} is too close to the edge of the simulation cell — move it inward", a(atom)),
        R::NoModel => "there is no model yet — this answer helps build one".into(),
        // the refit needs El's reference, and the larger basis is untrusted until El samples
        // accumulate (plan review S13)
        R::NewSpecies(z) => {
            let el = symbol(z);
            format!("the model has never seen {el} — answers like this one teach it; it can answer for {el} after a refit with {el}'s reference energy and enough {el} samples")
        }
        R::OtherSettings => "this scene's settings (grid, smearing, charge or spin) differ from the model's — the answer cannot be used for it".into(),
        R::Untrusted(k) => format!("the model was not trusted then: {}", untrusted_title(k)),
        R::Class(EnvClass::Pair) => "the model hasn't learned two-atom arrangements well enough yet".into(),
        R::Class(EnvClass::ManyBody) => "the model hasn't learned three-or-more-atom arrangements well enough yet".into(),
        R::IsolatedAtom => with_box(format!("{} has no neighbour within reach — the model can't describe a lone atom inside a molecule's spin state", a(atom)), true),
        R::Coordination | R::Spread | R::NoGrade | R::UnspannedAtom | R::NewEnvironment | R::Grade => with_box(format!("{} {OUTSIDE}", a(atom)), true),
        R::ForceUncertainty | R::EnergyUncertainty => {
            let base = match atom {
                Some(i) => format!("the model is unsure here (atom {i})"),
                None => "the model is unsure here".into(),
            };
            // "teaches it" only for a kept answer that no box clause explains (plan review S13)
            if cx.box_affected {
                with_box(base, false)
            } else if cx.kept {
                format!("{base} — this answer teaches it")
            } else {
                base
            }
        }
        R::NonFiniteAnswer => "the model's answer was not a finite number, so it was not used — this answer is the teacher's".into(),
        R::NearQuarantine => "this geometry is next to one the model could not learn".into(),
        R::Audit => "a routine check of the model against DFT".into(),
        R::RelaxVerification => "checking the relaxed geometry with DFT".into(),
        R::Pass(PassJob::Reference { z, multiplicity }) => format!("sampling: the isolated {} atom (spin multiplicity {multiplicity}), for its reference energy", symbol(z)),
        R::Pass(PassJob::Perturbation) => "sampling: a random nudge of the scene".into(),
        R::Pass(PassJob::Scan) => "sampling: a bond-length scan".into(),
        R::Pass(PassJob::HotBurst) => "sampling: a hot MD burst".into(),
    }
}

/// The short outcome of a log row.
pub fn outcome_label(o: K) -> String {
    match o {
        K::Kept => "kept".into(),
        K::KeptBoxAffected => "kept (box)".into(),
        K::KeptByVote => "kept by vote".into(),
        K::KeptReleased { released, .. } => format!("kept, released {released}"),
        K::Thinned => "skipped: similar".into(),
        K::BoxNotKept => "not kept: box".into(),
        K::FenceDisagrees => "not kept: fence".into(),
        K::WaitingForPass => "not kept: waiting for a pass".into(),
        K::NotOffered => "not kept: other settings".into(),
        K::Discarded => "discarded".into(),
        K::StoreRefused => "refused by store".into(),
        K::Reference => "reference".into(),
        K::Failed => "failed".into(),
        K::NotSolved => "not solved".into(),
    }
}

/// What the outcome means (spec §5.2); the numbers are the policy's constants, read in code.
pub fn outcome_sentence(o: K) -> String {
    match o {
        K::Kept => "kept: it will be in the next fit".into(),
        K::KeptBoxAffected => "kept while the model is being built; it touches the box walls and leaves the fit once enough others exist".into(),
        K::KeptByVote => format!("kept: {FENCE_VOTES} fresh answers agree with each other but not with the model — the next refit decides"),
        K::KeptReleased { released, superseded } => format!("kept: it agrees with the model, so {released} quarantined samples were released ({superseded} superseded)"),
        K::Thinned => format!("skipped: too similar to a sample already held (one in {THIN_EVERY} such is kept)"),
        K::BoxNotKept => "not kept: it touches the box walls — the box's squeeze, not the molecule".into(),
        K::FenceDisagrees => "not kept: next to a geometry the model could not learn, and the teacher still disagrees with the model there".into(),
        K::WaitingForPass => "not kept: many answers are held without a model — run a sampling pass".into(),
        K::NotOffered => "used for this step, not kept for this model: its settings differ".into(),
        K::Discarded => "not judged and not kept: the model changed settings or species before the background answer arrived".into(),
        K::StoreRefused => "the store refused it (shape or settings)".into(),
        K::Reference => "a reference energy, not a sample".into(),
        K::Failed => "the solve failed".into(),
        K::NotSolved => "not solved".into(),
    }
}
