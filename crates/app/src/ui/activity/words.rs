//! Plain wording (activity spec §5): codes in, sentences out; the provider's technical text stays
//! on hover. Element symbols from `units` (display only, rule 1); mHa and Å at this edge (rule 3).
use providers::surrogate::{FENCE_VOTES, THIN_EVERY};
use sim::activity::{EnvClass, PassJob, TeachOutcome as K, TeachRule as R, TrustKey};
use units::isotopes::symbol;

/// "2,077" for 2077.
pub fn thousands(n: u64) -> String {
    let d = n.to_string();
    let mut out = String::with_capacity(d.len() + d.len() / 3);
    for (i, c) in d.chars().enumerate() {
        if i > 0 && (d.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// The running refit's stage in plain words (the refit picture, `sim::refit_live`): "gathering
/// samples", "solving the model", "checking each sample", the passes without a counter ("active
/// set", "null space", "training residuals", "counting environments"), "starting" before the first;
/// a stage of the mix W's learn (a channel collection) is "learning the mix W · gathering" /
/// "· solving".
pub fn refit_stage_name(l: &sim::RefitLive) -> String {
    use sim::{LiveStage as S, OtherPass as O};
    let name = match l.stage {
        S::Starting => "starting",
        S::Gather if l.mix => "learning the mix W · gathering",
        S::Solve if l.mix => "learning the mix W · solving",
        S::Gather => "gathering samples",
        S::Solve => "solving the model",
        S::Check => "checking each sample",
        S::Other(O::Counts) => "counting environments",
        S::Other(O::Residuals) => "training residuals",
        S::Other(O::NullSpace) => "null space",
        S::Other(O::ActiveSet) => "active set",
    };
    name.into()
}

/// The running stage's counter: "1,200 of 2,655 samples", "round 3 · 12 decompositions of
/// 4,140 × 4,140 · 0.64 s each" ("first decomposition" before one finished: no time per one yet),
/// "1,240 of 2,655 · 3 beyond their bound"; empty for a pass without a counter.
pub fn refit_counter_words(l: &sim::RefitLive) -> String {
    use sim::LiveStage as S;
    match l.stage {
        S::Gather => format!("{} of {} samples", thousands(l.done as u64), thousands(l.total as u64)),
        S::Solve => {
            let q = thousands(l.q as u64);
            match l.decompositions {
                0 => format!("round {} · first decomposition of {q} × {q}", l.round.max(1)),
                k => {
                    let each = if l.decomposition_s >= 1.0 { format!("{:.1} s", l.decomposition_s) } else { format!("{:.2} s", l.decomposition_s) };
                    format!("round {} · {k} decomposition{} of {q} × {q} · {each} each", l.round.max(1), if k == 1 { "" } else { "s" })
                }
            }
        }
        S::Check => {
            let beyond = match l.beyond {
                0 => String::new(),
                1 => " · 1 beyond its bound".into(),
                b => format!(" · {} beyond their bound", thousands(b as u64)),
            };
            format!("{} of {}{beyond}", thousands(l.done as u64), thousands(l.total as u64))
        }
        S::Starting | S::Other(_) => String::new(),
    }
}

/// The running stage's time left, from its own counter (`sim::RefitLive::stage_left_s`): "about
/// 0:12 left in this stage", rounded up (never 0:00 while it runs); `None` for a stage without a
/// total (the search, the passes) or before its rate is measured.
pub fn refit_left_words(l: &sim::RefitLive) -> Option<String> {
    let s = l.stage_left_s()?.max(0.0).ceil() as u64;
    Some(if s == 0 { "finishing this stage".into() } else { format!("about {}:{:02} left in this stage", s / 60, s % 60) })
}

/// The trust recovery's part: "trust round 2 of up to 5 · 41 held out", "trust recovery · basis
/// retry" before its quarantine rounds; `None` outside one.
pub fn refit_trust_words(l: &sim::RefitLive) -> Option<String> {
    let t = l.trust?;
    Some(match t.round {
        0 => "trust recovery · basis retry".into(),
        r => format!("trust round {r} of up to {} · {} held out", t.max, thousands(t.held as u64)),
    })
}

/// A running refit in one phrase (the headline, the Model banner, the job's detail): its stage and
/// that stage's time left — "checking each sample · about 0:04 left in this stage", "solving the
/// model", "trust round 2 of up to 5 · 41 held out · solving the model".
pub fn refit_now_words(v: &sim::RefitView) -> String {
    let l = &v.live;
    let mut parts: Vec<String> = refit_trust_words(l).into_iter().collect();
    parts.push(refit_stage_name(l));
    parts.extend(refit_left_words(l));
    parts.join(" · ")
}

/// The job card's count line: the trust round, the stage's counter and its time left —
/// "trust round 2 of up to 5 · 41 held out · 1,240 of 2,655 · about 0:04 left in this stage".
pub fn refit_count_line(v: &sim::RefitView) -> String {
    let l = &v.live;
    let mut parts: Vec<String> = refit_trust_words(l).into_iter().collect();
    let c = refit_counter_words(l);
    parts.push(if c.is_empty() { refit_stage_name(l) } else { c });
    parts.extend(refit_left_words(l));
    parts.join(" · ")
}

/// What the refit's words are, for their hover.
pub const REFIT_ETA_TIP: &str = "What the refit is doing now, counted by the refit itself: gathering the samples into the fit's statistics, learning the mix W (a channel collection, when it is due), solving the model (the hyperparameter search: each round decomposes the Q′ × Q′ matrix, and the search stops when its levels settle, so it has no fixed number of rounds), checking each sample (leave-one-out: each sample's error with the sample left out, against its bound), and the trust recovery's rounds. The time left is the running stage's only, from its own rate: a stage that has not started has no number";

/// What a log row (or a stuck streak) knew about its answer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RowCtx {
    /// The teacher's answer was box-affected.
    pub box_affected: bool,
    /// The named atom was within `d_box(h)` of a cell face.
    pub near_face: bool,
    /// The answer was kept as a sample (`TeachOutcome::is_kept`).
    pub kept: bool,
    /// A rule-6/7 refusal that passes at λ_sum alone (history code 24, `TeachEvent::history_code`):
    /// the tail-aware inflations decided — the model was measured too confident here, not unsure
    /// (Task 8 review M4).
    pub tail: bool,
}

/// `box_advice`'s distance: where a hydrogen-like tail e^(−2r)/π falls to the box rule's face
/// density (5.35 Bohr at h = 0.15) — the stated estimate, unchanged
/// (`crate::quality::provisional_cloud_radius`, the estimated box's bound before a profile).
pub fn d_box(h: f64) -> f64 {
    crate::quality::provisional_cloud_radius(h)
}

/// Distance (Bohr) from `p` to the nearest face of `scene`'s cell.
pub fn face_distance(scene: &sim::Scene, p: [f64; 3]) -> f64 {
    let half = (scene.cell.n as f64 - 1.0) * scene.cell.h / 2.0;
    half - p.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

/// A pass's stage in words, a teach-around pass with its blocker ("teaching around #345").
pub fn pass_stage(p: &sim::SamplingProgress) -> String {
    match (p.stage, p.target) {
        ("teaching", Some(t)) => format!("teaching around #{t}"),
        (s, _) => stage_word(s).to_string(),
    }
}

/// The same, taking the stage of the job being solved now from the pass's ghost when there is one
/// (the progress's stage is the last finished job's); the teach-around blocker comes from the
/// progress.
pub fn pass_stage_of(p: &sim::SamplingProgress, ghost: Option<&sim::PassGhost>) -> String {
    let stage = ghost.map_or(p.stage, |g| g.stage);
    match (stage, p.target) {
        ("teaching", Some(t)) => format!("teaching around #{t}"),
        (s, _) => stage_word(s).to_string(),
    }
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
        R::UnrepresentedNeighbour(z) => format!("3b no channel for {}", symbol(z)),
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
        R::NearMisfit => "misfit".into(),
        R::Audit => "8 audit".into(),
        R::RelaxVerification => "relax check".into(),
        R::Segment => "segment".into(),
        R::Pass(PassJob::Reference { z, multiplicity }) => format!("reference {} m{multiplicity}", symbol(z)),
        R::Pass(PassJob::Perturbation) => "nudge".into(),
        R::Pass(PassJob::Scan) => "bond scan".into(),
        R::Pass(PassJob::Teach { blocker, holdout: false }) => format!("teach #{blocker}"),
        R::Pass(PassJob::Teach { blocker, holdout: true }) => format!("check #{blocker}"),
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
        TrustKey::Misfit => "too many training samples it could not reproduce",
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
        R::UnrepresentedNeighbour(z) => {
            let el = symbol(z);
            format!("{} has two or more neighbours and one is {el}, which the model's chemical channels don't cover yet — answers like this one teach it; the channels are relearnt once {el} has enough samples like this", a(atom))
        }
        R::Untrusted(k) => format!("the model was not trusted then: {}", untrusted_title(k)),
        R::Class(EnvClass::Pair) => "the model hasn't learned two-atom arrangements well enough yet".into(),
        R::Class(EnvClass::ManyBody) => "the model hasn't learned three-or-more-atom arrangements well enough yet".into(),
        R::IsolatedAtom => with_box(format!("{} has no neighbour within reach — the model can't describe a lone atom inside a molecule's spin state", a(atom)), true),
        R::Coordination | R::Spread | R::NoGrade | R::UnspannedAtom | R::NewEnvironment | R::Grade => with_box(format!("{} {OUTSIDE}", a(atom)), true),
        R::ForceUncertainty | R::EnergyUncertainty => {
            let here = match atom {
                Some(i) => format!("here (atom {i})"),
                None => "here".into(),
            };
            let base = if cx.tail {
                // code 24: its own std passed; DFT checks measured it too small for this kind of
                // environment (λ_T / λ_E), and the scaled std refuses
                format!("the model was sure {here}, but DFT checks found it too confident in places like this (its uncertainty scaled by λ_T/λ_E is over the bound)")
            } else {
                format!("the model is unsure {here}")
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
        R::Segment => "this scene is a cut-out segment — DFT answers every step and teaches the model the forces of its interior atoms; the model does not answer here".into(),
        R::NearQuarantine => "this geometry is next to one the model could not learn".into(),
        R::NearMisfit => "an atom here is next to a training sample the model cannot reproduce".into(),
        R::Audit => "a routine check of the model against DFT".into(),
        R::RelaxVerification => "checking the relaxed geometry with DFT".into(),
        R::Pass(PassJob::Reference { z, multiplicity }) => format!("sampling: the isolated {} atom (spin multiplicity {multiplicity}), for its reference energy", symbol(z)),
        R::Pass(PassJob::Perturbation) => "sampling: a random nudge of the scene".into(),
        R::Pass(PassJob::Scan) => "sampling: a bond-length scan".into(),
        R::Pass(PassJob::Teach { blocker, holdout: false }) => format!("teaching: a geometry near training sample #{blocker}, which the rest of the data could not predict"),
        R::Pass(PassJob::Teach { blocker, holdout: true }) => format!("teaching: a held-out check near training sample #{blocker} (never fitted — it tests the result)"),
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
        K::FenceLifted { .. } => "kept, fence lifted".into(),
        K::KeptStrayHeld => "held: stray atoms (until measured)".into(),
        K::Thinned => "skipped: similar".into(),
        K::BoxNotKept => "not kept: box".into(),
        K::UnfitNotKept => "not kept: cores / occupations".into(),
        K::ShellNotKept => "not kept: cutoff shell".into(),
        K::LoneNotKept => "not kept: lone atoms".into(),
        K::StrayNotKept => "not kept: stray atoms".into(),
        K::StrayChecking { n, of } => format!("checking composition {n} of {of}"),
        K::StrayCheckingNew => "checking composition (new geometry)".into(),
        K::StrayRefused => "composition fails".into(),
        K::StrayAuditAttributed { model: true } => "audit failed: the model's".into(),
        K::StrayAuditAttributed { model: false } => "audit failed: not the model".into(),
        K::FenceDisagrees => "not kept: fence".into(),
        K::MisfitDisagrees => "not kept: misfit".into(),
        K::WaitingForPass => "not kept: waiting for a pass".into(),
        K::NotOffered => "not kept: other settings".into(),
        K::HeldOut => "not kept: held-out check".into(),
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
        K::KeptStrayHeld => "held until measured: the molecule has stray atoms (no neighbour within the model's cutoff) and no composition check has measured, at this geometry, that its energy is the molecule's plus the free atoms' — it stays out of the fit until a check passes here".into(),
        K::FenceLifted { lifted } => format!(
            "kept: {FENCE_VOTES} fresh answers here agree with the model, so the fence of {lifted} forgotten sample{} is lifted — the model answers here again; they stay forgotten (Restore brings them back)",
            if lifted == 1 { "" } else { "s" }
        ),
        K::Thinned => format!("skipped: too similar to a sample already held (one in {THIN_EVERY} such is kept)"),
        K::BoxNotKept => "not kept: it touches the box walls — the box's squeeze, not the molecule".into(),
        K::UnfitNotKept => "used for this step, not kept: two atoms' cores overlap (the frozen-core approximation fails there) or F changed by more than the forces' work since the last step (the SCF moved to another stationary state)".into(),
        K::ShellNotKept => "used for this step, not kept and not judged: a pair lies in the cutoff shell a shell check measured — the model's cutoff cannot represent it (a shell fence)".into(),
        K::LoneNotKept => "used for this step, not kept: no atom has a neighbour within the model's cutoff — a lone atom has nothing to learn (the isolated atom is the reference energy)".into(),
        K::StrayNotKept => "used for this step, not kept: the molecule has stray atoms (no neighbour within the model's cutoff) in a combination whose DFT energy is not known to split into the molecule's and the free atoms' (the reason is in the detail)".into(),
        K::StrayChecking { n, of } => format!("a composition check passed on the background teacher ({n} of the {of} at distinct geometries the model needs before it answers this molecule with its stray atoms)"),
        K::StrayCheckingNew => "a composition check passed at a new geometry of the molecule: the region the model answers it in grows".into(),
        K::StrayRefused => "a composition check failed: this molecule with its stray atoms stays with DFT for the session, and its samples stay out of the fit".into(),
        K::StrayAuditAttributed { model: true } => "an audit of the model's answer with stray atoms failed, and a check of the molecule alone shows the model is wrong there: it counts as a failed audit; the molecule with its stray atoms stays with DFT for the session".into(),
        K::StrayAuditAttributed { model: false } => "an audit of the model's answer with stray atoms failed, but the model is right on the molecule alone: the difference is the combination's (the composition, or where the stray atoms sit in the box — the detail gives each term), so the molecule with its stray atoms stays with DFT for the session and the model's trust is unchanged".into(),
        K::FenceDisagrees => "not kept: next to a geometry the model could not learn, and the teacher still disagrees with the model there".into(),
        K::MisfitDisagrees => "not kept: next to a training sample the model cannot reproduce, and the teacher disagrees with the model there".into(),
        K::WaitingForPass => "not kept: many answers are held without a model — run a sampling pass".into(),
        K::NotOffered => "used for this step, not kept for this model: its settings differ".into(),
        K::HeldOut => "a DFT solve held out on purpose to test the teaching — never trained on".into(),
        K::Discarded => "not judged and not kept: the model changed settings or species before the background answer arrived".into(),
        K::StoreRefused => "the store refused it (shape or settings)".into(),
        K::Reference => "a reference energy, not a sample".into(),
        K::Failed => "the solve failed".into(),
        K::NotSolved => "not solved".into(),
    }
}

/// Hartree → mHa (or Ha/Bohr → mHa/Bohr) for the plain lines: two significant figures, trailing
/// zeros dropped ("1", "2.2", "0.45", "0.045", "12"; plan review N7).
pub fn mha(x: f64) -> String {
    let v = x * 1e3;
    if v == 0.0 || !v.is_finite() {
        return format!("{v}");
    }
    let decimals = (1 - v.abs().log10().floor() as i32).max(0) as usize;
    let s = format!("{v:.decimals$}");
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

/// What is known of the model's last check against DFT: never audited, audited only before the
/// last refit (the record describes an earlier fit), or audited this fit (activity Task 3 review
/// I1). Three states, so "not checked since the refit" is never worded as "passed".
#[derive(Clone, Copy, Debug)]
pub enum AuditSeen<'a> {
    Never,
    EarlierFit(&'a sim::AuditRecord),
    ThisFit(&'a sim::AuditRecord),
}

impl<'a> AuditSeen<'a> {
    pub fn of(m: &'a crate::sim::ModelStatus) -> AuditSeen<'a> {
        match (&m.last_audit, m.last_audit_this_fit) {
            (None, _) => AuditSeen::Never,
            (Some(a), true) => AuditSeen::ThisFit(a),
            (Some(a), false) => AuditSeen::EarlierFit(a),
        }
    }
}

fn verdict_word(v: sim::AuditVerdict) -> &'static str {
    match v {
        sim::AuditVerdict::Passed => "passed",
        sim::AuditVerdict::Failed => "failed",
        sim::AuditVerdict::BoxAffected => "box-affected, not judged",
    }
}

/// Where a residual check's worst value sits, as the trust line names it (2026-11-13):
/// ` in #1234 · H3O (atom 2, O)` for a force, ` in #1234 · H3O` for an energy — the store index
/// and formula as the Worst-fitting rows show them, the atom index within the sample (from 0, as
/// the other atom words count) and its element. Empty when the provider gave no location (an old
/// status, or the sample changed since), so the line reads as before.
pub fn worst_at(c: &sim::activity::TrustCheck) -> String {
    match &c.worst {
        None => String::new(),
        Some(w) => {
            let atom = w.atom.map_or(String::new(), |(j, z)| format!(" (atom {j}, {})", symbol(z)));
            format!(" in #{} · {}{atom}", w.id.index, w.formula)
        }
    }
}

/// One trust condition as a plain line (spec §5.3). `seen` and `audit_every` word the met audit
/// line (plan review M1, Task 3 review I1): it says what the last audit was, that none has run
/// since the refit, or that none ever did — never "passed" by default.
pub fn trust_line(c: &sim::activity::TrustCheck, seen: AuditSeen<'_>, audit_every: usize) -> String {
    match c.key {
        TrustKey::Samples => format!("Enough samples in the fit — {} of {} needed (one per independent direction the model can bend)", c.value as usize, c.limit as usize),
        TrustKey::ResidualDofE => format!("Enough data beyond the fit's own parameters to measure its noise — energies {:.0} of {:.0}", c.value, c.limit),
        TrustKey::ResidualDofF => format!("Enough data beyond the fit's own parameters to measure its noise — forces {:.0} of {:.0}", c.value, c.limit),
        TrustKey::EnergyResidual => format!("Reproduces its own training energies to the accuracy it claims — worst {} mHa/atom{}, allowed {} mHa/atom", mha(c.value), worst_at(c), mha(c.limit)),
        TrustKey::ForceResidual => format!("Reproduces its own training forces — worst {}{}, allowed {} mHa/Bohr for that atom", mha(c.value), worst_at(c), mha(c.limit)),
        // M: fitted samples over their bound, judged apart from the two lines above (spec 2026-11-13 §4.3)
        TrustKey::Misfit => format!(
            "Few training samples it cannot reproduce — {} of at most {:.1} (10 % of the fitted samples){}",
            c.value as usize,
            c.limit,
            c.worst.as_ref().map_or(String::new(), |_| format!("; worst{}", worst_at(c)))
        ),
        // σ_f is label noise and basis misfit together (plan review S8): never "the labels disagree"
        TrustKey::ForceNoise => format!("Its forces fit the data to within the force tier on average — scatter σ_f {}, allowed {} mHa/Bohr", mha(c.value), mha(c.limit)),
        TrustKey::AuditForces => format!("Last check against DFT — found forces off by {}, allowed {} mHa/Bohr (until a refit predicts it without that sample)", mha(c.value), mha(c.limit)),
        TrustKey::Audit if !c.met => format!("Last check against DFT — found the energy off by {} mHa/atom, allowed {} mHa/atom (until a refit predicts it without that sample)", mha(c.value), mha(c.limit)),
        TrustKey::Audit => {
            let what = match seen {
                AuditSeen::Never if audit_every == 0 => "none, audits are off".to_string(),
                AuditSeen::Never => format!("none yet, one every {audit_every} answered steps"),
                AuditSeen::EarlierFit(a) => format!("none since the last refit — the fit before it was last checked: {}", verdict_word(a.verdict)),
                AuditSeen::ThisFit(a) => verdict_word(a.verdict).to_string(),
            };
            format!("Last check against DFT — no failed check since the last refit (last check: {what})")
        }
    }
}

/// The class line from the class's own verdict (`ClassLoo::refusal`, plan review M3): an
/// energy-only failure is an energy error, and each number is named by what measured it — "DFT
/// checks measured" only for the error the teacher measured, "leave-one-out" only for the fit's own
/// left-out error (M8a Task 2 review I3).
pub fn class_line(l: &sim::ClassLoo) -> String {
    verdict_line(class_name(l.class), l.refusal, Counts { evidence: l.evidence, fits: l.fits, floor_points: l.floor_points })
}

/// The plain name of an environment class.
fn class_name(class: &str) -> &'static str {
    if class == "pair" { "Two-atom arrangements" } else { "Three-or-more-atom arrangements" }
}

/// One environment species' line under its class (M8a user-test fix 2: an atom answers by its own
/// species' verdict): "Three-or-more-atom arrangements around O, with H — learned". A key the
/// class rule trusts but its tail-aware inflations refuse is never worded plain "learned" (trust
/// tightening, Review Focus 3): "— learned, but …" with [`tail_note`].
pub fn species_line(class: &str, s: &sim::SpeciesVerdict) -> String {
    let name = format!("{} {}", class_name(class), s.words());
    match tail_note(s).filter(|_| s.trusted && s.refusal.is_none()) {
        Some(note) => format!("{name} — learned, but {note}"),
        None => verdict_line(&name, s.refusal, Counts { evidence: s.evidence, fits: s.fits, floor_points: s.floor_points }),
    }
}

/// The evidence counts a verdict line words: the evidence (`ClassLoo::evidence`) and the evidence
/// floor's n_fit and n_pts (trust tightening, spec 2026-11-18 §3).
#[derive(Clone, Copy, Debug)]
struct Counts {
    evidence: usize,
    fits: usize,
    floor_points: usize,
}

/// The evidence floor's two counts in words, as the provider's refusal says them
/// (`providers::surrogate::floor_count`): "1 of 12 fits / 4 of 60 checks", "11 of 12 fits / 80
/// checks (60 needed)" — never "80 of 60".
pub fn floor_counts(fits: usize, floor_points: usize) -> String {
    use providers::surrogate::{floor_count, FLOOR_FITS, FLOOR_POINTS};
    format!("{} / {}", floor_count(fits, FLOOR_FITS, "fits"), floor_count(floor_points, FLOOR_POINTS, "checks"))
}

/// An inflation factor as the provider's refusal words print it: one decimal, "∞", "NaN".
pub fn factor(x: f64) -> String {
    if x.is_nan() {
        "NaN".into()
    } else if x.is_infinite() {
        "∞".into()
    } else {
        format!("{x:.1}")
    }
}

/// What the tail-aware inflations do to a key's steps beyond its λ_sum (trust tightening, spec
/// 2026-11-18 §4; the steps they refuse that pass at λ_sum alone are history code 24), `None`
/// while λ_T and λ_E are both at most the λ_sum the rules read for it
/// (`SpeciesVerdict::rule_inflation`: a sparse key's class floor included, Task 8 review I2):
///
/// - a factor that is not a number, or infinite, refuses every step with the key (rule 6 at ∞·0 =
///   ∞, rule 7 alike) until the comparison that set it ages out of the evidence window ("No lockout
///   follows": the refused steps keep being measured) — the factor named, NaN before ∞, λ_T before
///   λ_E on a tie;
/// - a finite one above λ_sum refuses the steps where the model's std passes at λ_sum but not at
///   the factor: DFT checks measured errors larger than the std claimed where it was confident —
///   the larger factor named, λ_T on a tie.
pub fn tail_note(s: &sim::SpeciesVerdict) -> Option<String> {
    let (t, e) = (s.tail_inflation, s.energy_inflation);
    let named = |x: f64, name: &str| format!("{name} {}", factor(x));
    let locked = "every step with it goes to DFT until a check ages out of the window";
    if t.is_nan() || e.is_nan() {
        let which = if t.is_nan() { named(t, "λ_T") } else { named(e, "λ_E") };
        return Some(format!("{locked}: a DFT comparison was not a number ({which})"));
    }
    if t.is_infinite() || e.is_infinite() {
        let which = if t.is_infinite() { named(t, "λ_T") } else { named(e, "λ_E") };
        return Some(format!("{locked}: a DFT check found an error beyond its limit where the model's std was 0 ({which})"));
    }
    let (x, name) = if t >= e { (t, "λ_T") } else { (e, "λ_E") };
    (x > s.rule_inflation).then(|| {
        format!(
            "steps its std alone would pass can still go to DFT: DFT checks measured larger errors where the model was sure ({} against λ_sum {})",
            named(x, name),
            factor(s.rule_inflation)
        )
    })
}

fn verdict_line(name: &str, refusal: Option<(sim::ClassKey, f64, f64)>, counts: Counts) -> String {
    use sim::ClassKey;
    let evidence = counts.evidence;
    let checks = |n: usize| format!("{n} DFT check{}", if n == 1 { "" } else { "s" });
    match refusal {
        None => format!("{name} — learned"),
        // the floor is the key's own column count (M8a user-fix 3)
        Some((ClassKey::Samples, n, q)) => format!("{name} — not yet ({n:.0} of {q:.0} samples needed)"),
        Some((ClassKey::Force, x, _)) => format!("{name} — not yet (leave-one-out force error {x:.1}× the bound)"),
        Some((ClassKey::Energy, x, _)) => format!("{name} — not yet (leave-one-out energy error {x:.1}× the tier)"),
        // open-shell Task 7c: the energy RMS with the samples the model cannot reproduce
        Some((ClassKey::EnergyRms, x, _)) => format!("{name} — not yet (leave-one-out energy error {x:.1}× the tier on average, training samples the model cannot reproduce included)"),
        // M8a Task 2: one atom a DFT check found beyond `WORST_ATOM` = 3× its bound — both ways
        // below the evidence floor (trust tightening §3, Task 4; fewer than three checks is below it
        // too), on entry only at the floor
        Some((ClassKey::WorstAtom, x, lim)) if counts.fits < providers::surrogate::FLOOR_FITS || counts.floor_points < providers::surrogate::FLOOR_POINTS => format!(
            "{name} — not yet (a DFT check found an atom's force {x:.1}× its bound, limit {lim:.0}× while it has too few independent DFT checks: {})",
            floor_counts(counts.fits, counts.floor_points)
        ),
        Some((ClassKey::WorstAtom, x, lim)) => format!("{name} — not yet (a recent DFT check found an atom's force {x:.1}× its bound; it starts answering only without one beyond {lim:.0}×)"),
        // the calibrated verdict: the error the DFT checks measured where the class answers …
        Some((ClassKey::Calibrated, x, lim)) => format!("{name} — not yet (DFT checks measured its force error at {x:.2}× the bound over {}, limit {lim:.2}×)", checks(evidence)),
        // … its floor, the fit's own left-out error …
        Some((ClassKey::LooFloor, x, lim)) => format!("{name} — not yet (leave-one-out force error {x:.2}× the bound on average, limit {lim:.2}×)"),
        // … and the energy the DFT checks measured
        Some((ClassKey::Window, n, w)) => format!("{name} — not yet (within its bound over {n:.0} DFT checks; it starts answering before the next refit only after {w:.0})"),
        Some((ClassKey::MeasuredEnergy, x, lim)) => format!("{name} — not yet (DFT checks measured its energy error at {x:.2}× the tier over {}, limit {lim:.2}×)", checks(evidence)),
        // trust tightening §3, the evidence floor: within the bound by the DFT checks, but too few
        // independent ones (fits, checks) to open it alone, and the fit's own leave-one-out fails
        // (the counts as the provider's refusal says them; a count over its floor reads "80 checks
        // (60 needed)", the Task 4 review's minor 3)
        Some((ClassKey::Floor, x, _)) => format!(
            "{name} — not yet (within the bound by too few independent DFT checks — {} — and its leave-one-out error is {x:.1}× the bound)",
            floor_counts(counts.fits, counts.floor_points)
        ),
    }
}

/// A trusted model that keeps falling back, in plain words (spec §5.3, the stuck variant). With
/// no stored reason no rule is invented (plan review S10).
pub fn stuck_sentence(i: &sim::activity::StuckInfo, scene: &sim::Scene) -> String {
    let Some(rule) = i.rule else {
        return format!("trusted, but not answering here (the last {} steps were DFT)", i.steps);
    };
    let near = i.atom.and_then(|a| scene.nuclei.get(a)).is_some_and(|n| face_distance(scene, n.pos) < d_box(scene.cell.h));
    let s = rule_sentence(rule, i.atom, RowCtx { box_affected: i.box_streak, near_face: near, kept: false, tail: false });
    format!("trusted, but not answering here: {s} (the last {} steps were DFT)", i.steps)
}
