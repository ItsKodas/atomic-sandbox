//! **Teaching around the blockers** (2026-10-18; the physics review's ruling 5, user-approved): a
//! targeted sampling pass around the training samples that keep an environment class untrusted.
//!
//! The class rule (4b) judges a class by the *maximum* leave-one-out error over its samples, so a
//! single sample the rest of the data cannot predict — the user's #345, a lone added H, 12.0× its
//! force bound — refuses the whole class everywhere. Such a sample blocks for one of two reasons:
//! - **leverage**: it alone spans some feature directions (in-sample residual small, LOO large) —
//!   the region around it has no data; adding data there is the only fix that loosens nothing;
//! - **structure**: the basis cannot represent its label (the in-sample residual is large too) —
//!   no data helps; the existing recovery (degree cap, quarantine) is what applies.
//!
//! # The rule (no new tolerance)
//! 1. **Diagnosis** ([`SurrogateKs::teach_diagnosis`]): per class that is untrusted by its LOO
//!    numbers with n ≥ Q′ (a class short of samples needs more samples of any kind, not this), its
//!    fitted samples beyond 1× (`class_blockers`), worst first. The in-sample ratio — the standing
//!    model at the sample, the training-residual criterion (|ΔE|/N over the tier, every atom's
//!    force residual over its bound) — decides: ≤ 1 teachable, > 1 refused ("the model cannot
//!    represent it"). At most [`TEACH_MAX_BLOCKERS`] per offer, and at most [`TEACH_ROUNDS`]
//!    rounds per environment class (then the offer names the blockers and stops) — per class, not
//!    per blocker: a round's own samples can become the next blockers (measured on the user's
//!    water + H₂: #438 at 10.6× → #664 at 19.6× → #688 at 1.47×), and a per-blocker count would
//!    offer rounds without end. The count restarts after a regular sampling pass, after an accepted
//!    round, and per class once the class is trusted (review SHOULD-T6); it is not saved.
//! 2. **Geometries** per blocker (`teach_jobs`), the blocker's own composition and cell:
//!    [`TEACH_PERTURBATIONS`] displacements by the pass's own generator (independent Gaussians of
//!    the plan's `perturbation_sigma` per coordinate; the width grows ×1.5 after every 40 rejected
//!    draws, at most twice) and [`TEACH_PATH_POINTS`] along the path to the nearest fitted sample
//!    of the same atom sequence — atom-matched over the same-species permutations, Kabsch-aligned
//!    onto the blocker, linear in between (SHOULD-T1) — or, without such a sample, the blocker's
//!    most isolated atom brought toward its nearest neighbour to 90 … 60 % of the distance. Every
//!    geometry kept is (a) inside the cell, (b) farther than `THIN_FRACTION`·`max_descriptor_distance`
//!    from the blocker and from each other — no copy can defeat the leave-one-out — and (c) **no
//!    farther from the rest of the data than the blocker is** (descriptor distance, the worst
//!    atom's nearest fitted atom; SHOULD-T2): teaching fills in around and inward, it never pushes
//!    the frontier out. A path point is dropped when its closest pair is closer than the blocker's
//!    and every fitted geometry's of its atom sequence (atoms passing through each other).
//! 3. **Held-out checks**: per blocker one nudge and one path point (two nudges without a path)
//!    are solved but kept out of the refit. After it each is judged by the audit criterion against
//!    the plain prediction (energy tier, every atom's force bound at 1×, the pseudo floor), then
//!    **kept** as a sample for the next refit (SHOULD-T4: validate, then train on all); a failed
//!    one is a failed audit with its sample's index (rule 8 and its leave-one-out re-judge) and the
//!    origin "teaching check", which the reason and the chip name. A check outside strict mode's
//!    domain is held out like a draft guess and judged once strict mode reaches it.
//! 4. **Acceptance** ([`teach_verdict`], review MUST-T1): the unchanged class rule over *all* the
//!    samples — the blocker and the new ones included; it is never removed, fenced or down-weighted —
//!    with **every class that holds samples trusted** (LOO ≤ 1 and n ≥ Q′), nothing taught around
//!    still blocking or out of the fit, every judged check passing, and at least one check judged
//!    near every blocker (round 3 R3-2: a blocker whose checks all lie outside strict mode's domain
//!    waits for them to be judged). A class whose leave-one-out
//!    passes but which is short of samples (the refit's basis grew: Q′ rose) is its own outcome —
//!    not accepted, "a sampling pass on this scene adds them"; a short class whose leave-one-out
//!    fails is "not accepted". A cancelled round is not a round spent. The outcome (SHOULD-T5) names
//!    per class the change of its worst leave-one-out, who blocks now (a sample of this teaching, the
//!    blocker itself, an older sample) and what is next; the basis growth; the checks; and the
//!    model's first unmet trust condition. The offer states its cost (stalls skipped), the risk (a
//!    failed check untrusts until re-earned), the sample floor a richer basis would set (SHOULD-T3)
//!    and a pair blocker's pair against r_cut.
//!
//! # The prompt (the live case, 2026-10-18)
//! After [`TEACH_PROMPT_STREAK`] consecutive class-rule fallbacks, when the refusing class's worst
//! blocker is farther than `max_descriptor_distance` from the geometry being asked, the offer
//! carries a prompt ([`SurrogateKs::teach_prompt`]): the steps here cannot cure a blocker there
//! (their answers teach this region, and within the thinning radius are not even kept), so instead
//! of silent DFT the status chip, a card in the view and the Model tab say why and offer the pass —
//! one click, never run automatically.
//!
//! # Measured (Morse teacher, `tests.rs`)
//! - `gap_blocked(2.0, 3.6)` (60 dimers on 1.1–2.0 Bohr and one at 3.6): leave-one-out 5.2×,
//!   in-sample 2.8e-7 → 9 solves (one nudge fell inside the cell margin's filter), 7 kept, 2 held
//!   out passing at 0.00× → **accepted**, pair class trusted, the blocker in the fit. The same at
//!   gaps to 3.2–4.4 Bohr (leave-one-out 1.5–11.6×): all accepted after one round.
//! - The tail at 5.1 Bohr (60 dimers on 1.1–2.9): leave-one-out 1.05×, in-sample 2.4e-7 — a
//!   leverage blocker by the diagnosis — **not accepted**: with the nudges out to 5.44 Bohr in the
//!   fit, the f_c-damped basis (r_cut = 6) cannot carry the Morse binding left at the cutoff
//!   (V(6) ≈ −3.4e-3 Ha): the fit matches every force and shifts every energy by the tail value
//!   (all residuals +1.88× the energy tier per atom), both held-out checks fail and become failed
//!   audits, and the next offer refuses the class's samples as structural. A leverage outlier near
//!   r_cut can be a truncation the data only reveal once its neighbourhood is sampled; the pass
//!   reports it rather than loosening anything.
//!
//! # Measured (GPU teacher)
//! - `validate --test surrogate`, `teaching_around_a_blocker_restores_its_class`: H₂ 49³, 40 solves
//!   on 1.1–2.0 Bohr + one at 3.6 (leave-one-out 113.9×, in-sample 0.015×) → 10 solves, 8 kept,
//!   2 held out at 0.02× → accepted; pair class max 0.10×.
//! - A one-off measurement of a user session (2026-10-18; the user's saved water + H₂ model, a copy,
//!   95³, O cores-only, 657 samples, loaded over the GPU teacher, up to two rounds; blockers #345 pair 12.0×, #438 10.6×, #439 3.56×, #87 1.52× — all
//!   in-sample ≤ 0.40×). Round 1: 36 solves (4 near #87 stalled in the SCF and were skipped),
//!   29 kept, 7 of 7 held-out checks pass (worst 0.76×); pair max 12.0× → 0.83×, but the refit's
//!   basis grew to Q′ 344 (the loaded fit stood at a lower degree), so the pair class is now short
//!   of samples (197 of 344); many-body 19.6× at a round-1 sample (#664). Round 2 on #664, #667,
//!   #663: 28 solves, 22 kept; many-body 1.47×, 5 of 6 checks pass, one at 1.66× → a failed audit.
//!   Not accepted after two rounds, 64 solves, 896 s of GPU time: the verdict at the scene went
//!   from "pair untrusted, #345 12.0×" to "untrusted: an audit found forces off by 3.3e-3 Ha/Bohr".
//!   (Before the review's fix round.) Which jobs became the next blockers (measured once on the
//!   CPU, 2026-10-18, by regenerating round 1's and round 2's jobs from that file and matching
//!   them to the taught copy's samples): #664 was not a permuted
//!   Kabsch path — #345 (O1H3) has no fitted sample of its atom sequence — but the isolated-atom
//!   path's 60 % point (the far H brought from 5.92 to ≈ 3.55 Bohr from O): 11.27 spreads from the
//!   rest of the data against its blocker's 4.50, an outward step the inward rule (2c) now drops;
//!   #663 (8.30) likewise. #688 was an *inward* nudge of #664 (8.58 against 10.84): the inward rule
//!   would have kept it. With the fix round's generator the user's blockers get #345 6 nudges (no
//!   path point is inward), #438 10, #439 10, #87 9 geometries.
use super::*;
use crate::sampling::{Holdout, TargetJob};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

/// Displacements per blocker (the review: ≈ 6).
pub const TEACH_PERTURBATIONS: usize = 6;
/// Points on the path to the nearest fitted sample of the blocker's atom sequence (≈ 4).
pub const TEACH_PATH_POINTS: usize = 4;
/// Blockers per offer (the review: at most 5).
pub const TEACH_MAX_BLOCKERS: usize = 5;
/// Rounds per environment class (pair, many-body): one, and one more on what still blocks it.
/// Per class, not per blocker (a round's own samples can become the next blockers). The count
/// restarts after a regular sampling pass and once the class is trusted; it lives in memory only,
/// so a save and load restarts it as well — that costs DFT solves, never trust.
pub const TEACH_ROUNDS: u32 = 2;
/// The permutations of same-species atoms tried when matching a path's two endpoints (the product
/// of the species groups' factorials). A cost bound, not physics: 6! covers six identical atoms;
/// beyond it the stored order is kept and the closest-pair and inward filters still apply.
const PERMUTATION_LIMIT: usize = 720;

/// Consecutive class-rule (4b) fallbacks after which a far blocker prompts the teach-around offer
/// ([`SurrogateKs::teach_prompt`]). A display cadence, not physics: five steps show the refusal
/// persists past one step and a refit (the worker refits pending samples at once), and are a few
/// seconds of DFT at the default grid — long before the stuck note's 50.
pub const TEACH_PROMPT_STREAK: usize = 5;

/// The environment classes in plain words (the outcome and the offer).
pub(super) const CLASS_WORDS: [&str; 2] = ["two-atom environments", "three-or-more-atom environments"];

/// One blocker as diagnosed.
#[derive(Clone, Debug, PartialEq)]
pub struct Blocker {
    pub index: usize,
    pub class: usize,
    pub loo: f64,
    pub in_sample: f64,
}

impl Blocker {
    /// A leverage outlier: data around it can make it predictable.
    pub fn teachable(&self) -> bool {
        self.in_sample <= 1.0 && !(self.loo <= 1.0)
    }
}

/// What a teach-around round ended as ([`teach_verdict`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TeachVerdict {
    Accepted,
    /// The leave-one-out numbers pass, but a class holds fewer samples than the basis's Q′: not
    /// accepted, and what is missing is samples of any kind (a sampling pass), not teaching.
    NeedsSamples,
    NotAccepted,
    Cancelled,
}

/// The acceptance rule (ruling 5; review MUST-T1, R3-2): accepted only when nothing it taught
/// around still blocks or left the fit, every held-out check that could be judged passed, **at
/// least one check was judged near every blocker** (`unjudged`: the blockers with none),
/// and **every class that holds samples is trusted** — its leave-one-out numbers within 1× *and*
/// n ≥ Q′. A class short of samples is its own outcome ([`TeachVerdict::NeedsSamples`]) when its
/// leave-one-out numbers pass, and "not accepted" when they do not (they are computed for a short
/// class too).
pub(super) fn teach_verdict(classes: &[sim::ClassLoo], still_blocking: bool, out_of_fit: bool, failed: usize, unjudged: usize, cancelled: bool) -> TeachVerdict {
    if cancelled {
        return TeachVerdict::Cancelled;
    }
    let present: Vec<&sim::ClassLoo> = classes.iter().filter(|c| c.samples > 0).collect();
    let loo_fails = |c: &sim::ClassLoo| !(c.force_max <= 1.0 && c.force_rms <= 1.0 && c.energy_max <= 1.0);
    // a class the calibrated verdict refuses (M8a Task 2: its audits found the model off where it
    // answers) is not accepted either — only a class short of samples is "needs samples"
    let refused_by_verdict = |c: &sim::ClassLoo| !c.trusted && !matches!(c.refusal, Some((sim::ClassKey::Samples, ..)));
    if still_blocking || out_of_fit || failed > 0 || unjudged > 0 || present.iter().any(|c| loo_fails(c) || refused_by_verdict(c)) {
        return TeachVerdict::NotAccepted;
    }
    if present.iter().any(|c| !c.trusted) {
        return TeachVerdict::NeedsSamples;
    }
    TeachVerdict::Accepted
}

/// Optimal rotation (Kabsch) of the points `p` onto `q` (both N × 3, centred by the caller):
/// the R minimising Σ |R p_i − q_i|², a proper rotation.
pub(super) fn kabsch(p: &[[f64; 3]], q: &[[f64; 3]]) -> nalgebra::Matrix3<f64> {
    let mut h = nalgebra::Matrix3::<f64>::zeros();
    for (a, b) in p.iter().zip(q) {
        for i in 0..3 {
            for j in 0..3 {
                h[(i, j)] += a[i] * b[j];
            }
        }
    }
    let svd = h.svd(true, true);
    let (u, vt) = (svd.u.expect("u"), svd.v_t.expect("v_t"));
    let v = vt.transpose();
    let d = (v * u.transpose()).determinant().signum();
    let mut corr = nalgebra::Matrix3::<f64>::identity();
    corr[(2, 2)] = if d == 0.0 { 1.0 } else { d };
    v * corr * u.transpose()
}

pub(super) fn centroid(p: &[[f64; 3]]) -> [f64; 3] {
    let n = p.len().max(1) as f64;
    std::array::from_fn(|a| p.iter().map(|x| x[a]).sum::<f64>() / n)
}

/// The closest pair distance of a geometry (∞ for one atom).
pub(super) fn closest_pair(p: &[[f64; 3]]) -> f64 {
    let mut m = f64::INFINITY;
    for i in 0..p.len() {
        for j in i + 1..p.len() {
            m = m.min((0..3).map(|a| (p[i][a] - p[j][a]).powi(2)).sum::<f64>().sqrt());
        }
    }
    m
}

/// All orderings of `v`.
fn permutations(v: &[usize]) -> Vec<Vec<usize>> {
    if v.len() <= 1 {
        return vec![v.to_vec()];
    }
    let mut out = Vec::new();
    for i in 0..v.len() {
        let mut rest = v.to_vec();
        let x = rest.remove(i);
        for mut p in permutations(&rest) {
            p.insert(0, x);
            out.push(p);
        }
    }
    out
}

/// `q`'s atoms matched to `p`'s and rigidly aligned onto them (review SHOULD-T1): both geometries
/// have the atom sequence `z`; over the permutations within each species (at most
/// [`PERMUTATION_LIMIT`] in all, else the stored order) the one whose Kabsch-aligned RMSD onto `p`
/// is least. Returns the aligned partner in `p`'s frame, atom i matched to `p`'s atom i — the
/// descriptor distance that chose the partner is permutation-invariant, the straight path between
/// the endpoints is not: an unmatched order drags identical atoms through each other's places.
pub(super) fn matched_partner(p: &[[f64; 3]], z: &[u8], q: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let mut species: Vec<u8> = z.to_vec();
    species.sort_unstable();
    species.dedup();
    let groups: Vec<Vec<usize>> = species.iter().map(|&s| (0..z.len()).filter(|&i| z[i] == s).collect()).collect();
    let count = groups.iter().map(|g| (1..=g.len()).product::<usize>()).try_fold(1usize, |a, b| a.checked_mul(b));
    let orders: Vec<Vec<usize>> = if count.is_some_and(|c| c <= PERMUTATION_LIMIT) {
        let mut all: Vec<Vec<usize>> = vec![(0..z.len()).collect()];
        for g in &groups {
            let perms = permutations(g);
            let mut next = Vec::with_capacity(all.len() * perms.len());
            for base in &all {
                for perm in &perms {
                    let mut o = base.clone();
                    for (slot, &src) in g.iter().zip(perm) {
                        o[*slot] = src;
                    }
                    next.push(o);
                }
            }
            all = next;
        }
        all
    } else {
        vec![(0..z.len()).collect()]
    };
    let cp = centroid(p);
    let pc: Vec<[f64; 3]> = p.iter().map(|x| std::array::from_fn(|a| x[a] - cp[a])).collect();
    let mut best: Option<(f64, Vec<[f64; 3]>)> = None;
    for o in orders {
        let qo: Vec<[f64; 3]> = o.iter().map(|&i| q[i]).collect();
        let cq = centroid(&qo);
        let qc: Vec<[f64; 3]> = qo.iter().map(|x| std::array::from_fn(|a| x[a] - cq[a])).collect();
        let r = kabsch(&qc, &pc);
        let aligned: Vec<[f64; 3]> = qc
            .iter()
            .map(|x| {
                let v = r * nalgebra::Vector3::new(x[0], x[1], x[2]);
                [v[0] + cp[0], v[1] + cp[1], v[2] + cp[2]]
            })
            .collect();
        let msd: f64 = aligned.iter().zip(p).map(|(a, b)| (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>()).sum();
        if best.as_ref().is_none_or(|(m, _)| msd < *m) {
            best = Some((msd, aligned));
        }
    }
    best.expect("one order at least").1
}

/// A teach-around round's state at its start, for the outcome's words.
#[derive(Clone, Debug)]
pub(super) struct TeachStart {
    /// Per class: the worst leave-one-out ratio before the round.
    pub(super) loo_before: Vec<f64>,
    /// Store size before the round: a blocker at or past it is a sample of this teaching.
    pub(super) samples_before: usize,
    /// The basis's degree cap and Q′ before the round (`None`: no body-3 basis).
    pub(super) degree_before: Option<usize>,
    pub(super) q_before: usize,
    pub(super) started: Instant,
}

/// The teach-around rounds since the counter last restarted: rounds, DFT solves, seconds.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct TeachSession {
    pub(super) rounds: u32,
    pub(super) solves: usize,
    pub(super) secs: f64,
}

/// "15 min", "40 s".
fn duration_words(s: f64) -> String {
    if s >= 90.0 {
        format!("{:.0} min", s / 60.0)
    } else {
        format!("{s:.0} s")
    }
}

impl SurrogateKs {
    /// The diagnosis (module doc, 1.): every blocker of every class untrusted by its LOO numbers
    /// with n ≥ Q′, worst first (one entry per sample: the class where it is worst).
    pub fn teach_diagnosis(&self) -> Vec<Blocker> {
        self.diagnose(false)
    }

    /// [`teach_diagnosis`](Self::teach_diagnosis); with `include_short` the classes short of
    /// samples too (their leave-one-out numbers are computed all the same): what still blocks after
    /// a round (review MUST-T1), never what is offered.
    pub fn diagnose(&self, include_short: bool) -> Vec<Blocker> {
        let (Some(model), Some(store)) = (self.model.as_ref(), self.store.as_ref()) else { return Vec::new() };
        let mut out: Vec<Blocker> = Vec::new();
        for (c, l) in self.class_loo.iter().enumerate() {
            let short = matches!(l.refusal, Some((sim::ClassKey::Samples, ..)));
            if (l.trusted || short) && !(include_short && short) {
                continue;
            }
            for &(i, loo) in self.class_blockers.get(c).map_or(&[][..], |v| &v[..]) {
                if let Some(b) = out.iter_mut().find(|b| b.index == i) {
                    if loo > b.loo {
                        (b.loo, b.class) = (loo, c);
                    }
                    continue;
                }
                let s = &store.samples[i];
                let in_sample = match model.predict(&s.pos, &s.z, false) {
                    Ok(p) => {
                        let n = s.pos.len().max(1) as f64;
                        let e = (p.energy - s.energy).abs() / n / self.policy.energy_std_per_atom;
                        p.forces.iter().zip(&s.forces).fold(e, |m, (fm, fs)| {
                            let (d, b) = self.policy.force_residual(fm, fs);
                            super::trust::max_nan(m, d / b)
                        })
                    }
                    Err(_) => f64::INFINITY,
                };
                out.push(Blocker { index: i, class: c, loo, in_sample });
            }
        }
        out.sort_by(|a, b| b.loo.total_cmp(&a.loo));
        out
    }

    /// What a teach-around pass would do now (`TrainableStatus::teach_offer`): the teachable
    /// blockers with rounds left in their class (at most `TEACH_MAX_BLOCKERS`), the refused ones
    /// with why, the solves (`TEACH_PERTURBATIONS` + `TEACH_PATH_POINTS` per blocker — the generator
    /// may make fewer, and a stretched geometry may stall and be skipped), and the offer's words
    /// (review SHOULD-T3/T5): the risk, the sample floor a richer basis would set, and for a pair
    /// blocker its pair against r_cut. `None` when nothing blocks.
    pub fn teach_offer(&self) -> Option<sim::TeachOffer> {
        let d = self.teach_diagnosis();
        if d.is_empty() {
            return None;
        }
        let rounds = |c: usize| self.teach_rounds.get(&c).copied().unwrap_or(0);
        let mut blockers = Vec::new();
        let mut refused = Vec::new();
        for b in &d {
            let class = ENV_CLASSES[b.class];
            if !b.teachable() {
                refused.push((b.index, class, b.in_sample, "the model cannot represent it (its own residual is beyond the bound)"));
            } else if rounds(b.class) >= TEACH_ROUNDS {
                refused.push((b.index, class, b.in_sample, "two teach-around rounds on this class did not restore it"));
            } else if blockers.len() < TEACH_MAX_BLOCKERS {
                blockers.push((b.index, class, b.loo, b.in_sample));
            }
        }
        let solves = blockers.len() * (TEACH_PERTURBATIONS + TEACH_PATH_POINTS);
        let refused_total = refused.len();
        refused.truncate(TEACH_MAX_BLOCKERS);
        let mut notes = Vec::new();
        if !blockers.is_empty() {
            notes.push(format!("≈ {solves} DFT solves (a stretched geometry may stall in the SCF and be skipped); 2 per blocker are held out as checks — a check that fails makes the model untrusted until it predicts that geometry again"));
            if let Some(w) = self.richer_basis_floor() {
                notes.push(w);
            }
            for &(i, class, ..) in &blockers {
                if class == ENV_CLASSES[0] {
                    if let Some(w) = self.pair_words(i) {
                        notes.push(w);
                    }
                }
            }
        }
        let prompt = if blockers.is_empty() { None } else { self.teach_prompt() };
        Some(sim::TeachOffer { blockers, refused, refused_total, solves, notes, prompt })
    }

    /// The live case (2026-10-18, the user's preview: a trusted water + 2 H model whose many-body
    /// class one refit refused everywhere because of a pass scan point at r_cut): after
    /// [`TEACH_PROMPT_STREAK`] consecutive class-rule fallbacks, when the refusing class's worst
    /// blocker is **far** from the geometry being asked — farther than `max_descriptor_distance`
    /// (the envelope's own limit) — the fallbacks here can never cure it (they teach the region
    /// here, and those within the thinning radius are not even kept: "skipped: similar"), so the
    /// offer says so: one click, never run automatically. `None` otherwise.
    pub(super) fn teach_prompt(&self) -> Option<String> {
        if self.class_streak < TEACH_PROMPT_STREAK {
            return None;
        }
        let (pos, z, c) = self.class_streak_at.as_ref()?;
        let c = *c;
        let store = self.store.as_ref()?;
        // the class that refused at the streak geometry (round 3 Minor: a query needs every class
        // among its atoms, so it can be the one with the smaller worst ratio)
        let l = self.class_loo.get(c)?;
        if l.trusted || matches!(l.refusal, Some((sim::ClassKey::Samples, ..))) {
            return None;
        }
        let (worst, ratio) = self.class_worst.get(c).copied().flatten()?;
        let b = store.samples.get(worst)?;
        let d = self.geometry_distance((pos, z), (&b.pos, &b.z));
        (d > self.policy.max_descriptor_distance).then(|| {
            format!(
                "the last {} steps fell back because {} are untrusted everywhere — the worst is #{worst} ({ratio:.1}× its bound), {d:.1} training spreads from here, so steps here cannot fix it: teach around it",
                self.class_streak, CLASS_WORDS[c]
            )
        })
    }

    /// SHOULD-T3: when the standing body-3 basis is below the recovery's degree and the recovery
    /// may still take it, the sample floor the richer basis would set and the classes it would
    /// leave short — the same `Basis` count the refit uses, no estimate.
    fn richer_basis_floor(&self) -> Option<String> {
        let model = self.model.as_ref()?;
        let b = &model.basis;
        if !b.body3 || b.degree_cap >= RECOVERY_DEGREE_CAP || self.degree_cap.is_some() {
            return None;
        }
        let mut species: Vec<u8> = self.store.as_ref()?.samples.iter().flat_map(|s| s.z.iter().copied()).collect();
        species.sort_unstable();
        species.dedup();
        let mut rich = Basis::new(&species);
        rich.degree_cap = RECOVERY_DEGREE_CAP;
        let q = rich.column_classes().len();
        let short: Vec<String> = self.class_loo.iter().enumerate().filter(|(_, c)| c.samples > 0 && c.samples < q).map(|(k, c)| format!("{} have {}", CLASS_WORDS[k], c.samples)).collect();
        (!short.is_empty()).then(|| {
            format!(
                "if the new samples need the richer basis (degree {} → {RECOVERY_DEGREE_CAP}), every kind of environment needs ≥ {q} samples: {} — a sampling pass would then be needed as well",
                b.degree_cap,
                short.join(", ")
            )
        })
    }

    /// A pair blocker's defining pair against the cutoff (SHOULD-T5; informational, decides
    /// nothing): its atom with exactly one neighbour within r_cut at the longest such distance.
    fn pair_words(&self, i: usize) -> Option<String> {
        let model = self.model.as_ref()?;
        let s = self.store.as_ref()?.samples.get(i)?;
        let rc = model.basis.r_cut;
        let dist = |a: usize, b: usize| (0..3).map(|k| (s.pos[a][k] - s.pos[b][k]).powi(2)).sum::<f64>().sqrt();
        let n = s.pos.len();
        let mut best: Option<(f64, usize, usize)> = None;
        for a in 0..n {
            let near: Vec<usize> = (0..n).filter(|&b| b != a && dist(a, b) < rc).collect();
            if let [b] = near[..] {
                if best.is_none_or(|(r, ..)| dist(a, b) > r) {
                    best = Some((dist(a, b), a, b));
                }
            }
        }
        let (r, a, b) = best?;
        Some(format!(
            "#{i}: {} {r:.2} Bohr from {} (the model sees out to r_cut = {rc:.2} Bohr; a blocker near r_cut can be a truncation teaching only reveals)",
            units::elements::symbol(s.z[a]),
            units::elements::symbol(s.z[b])
        ))
    }

    /// A stored sample as a scene with `template`'s cell, σ, charge, multiplicity and ionic policy
    /// (isotopes: the template's nucleus of the same Z, else the default).
    fn sample_scene(s: &Sample, template: &Scene) -> Scene {
        let nuclei = s
            .pos
            .iter()
            .zip(&s.z)
            .map(|(p, &z)| {
                let mut n = template.nuclei.iter().find(|n| n.z == z).cloned().unwrap_or_else(|| sim::Nucleus::new(z, *p));
                n.pos = *p;
                n.vel = [0.0; 3];
                n
            })
            .collect();
        Scene { nuclei, external_field: [0.0; 3], ..template.clone() }
    }

    /// The per-sample descriptor distance of two geometries of the same species, symmetrised (the
    /// larger of the two worst-atom-nearest distances), training-spread units; ∞ without a model.
    pub(super) fn geometry_distance(&self, a: (&[[f64; 3]], &[u8]), b: (&[[f64; 3]], &[u8])) -> f64 {
        let (Some(model), Some(env)) = (self.model.as_ref(), self.envelope.as_ref()) else { return f64::INFINITY };
        let basis = &model.basis;
        let one = |p: (&[[f64; 3]], &[u8]), q: (&[[f64; 3]], &[u8])| -> f64 {
            let (Ok(dp), Ok(sp), Ok(dq), Ok(sq)) = (describe_values(basis, p.0, p.1), basis.species_indices(p.1), describe_values(basis, q.0, q.1), basis.species_indices(q.1)) else {
                return f64::INFINITY;
            };
            sp.iter().zip(&dp).fold(0.0f64, |w, (s, x)| {
                w.max(sq.iter().zip(&dq).filter(|(t, _)| *t == s).map(|(t, y)| spread_distance(env, *t, x, y)).fold(f64::INFINITY, f64::min))
            })
        };
        one(a, b).max(one(b, a))
    }

    /// The descriptor distance (training-spread units; the worst atom's nearest fitted atom of its
    /// species) from a geometry to the fitted data other than sample `skip` (review SHOULD-T2:
    /// "no farther from the data than the blocker is"). Reads the descriptor cache
    /// (`ensure_desc_cache` first); ∞ without one.
    pub(super) fn data_distance(&self, pos: &[[f64; 3]], z: &[u8], skip: usize) -> f64 {
        let (Some(basis), Some(env)) = (self.desc_basis.as_ref(), self.envelope.as_ref()) else { return f64::INFINITY };
        let (Ok(desc), Ok(sp)) = (describe_values(basis, pos, z), basis.species_indices(z)) else { return f64::INFINITY };
        let n = self.accumulated.min(self.desc_cache.len());
        let mut worst = 0.0f64;
        for (qs, qd) in sp.iter().zip(&desc) {
            let mut best = f64::INFINITY;
            for i in (0..n).filter(|&i| i != skip && !self.excluded(i)) {
                for (s, b) in &self.desc_cache[i] {
                    if s == qs {
                        best = best.min(spread_distance(env, *s, qd, b));
                    }
                }
            }
            worst = worst.max(best);
        }
        worst
    }

    /// The geometries around one blocker (module doc, 2.–3.). Each kept geometry is inside the
    /// cell, beyond the thinning radius from the blocker and from every geometry kept before it,
    /// and **no farther from the rest of the data than the blocker is** (SHOULD-T2: teaching
    /// fills in around and inward, it does not push the frontier out — the acceptance run's #664,
    /// a path point 11.3 spreads from the rest against its blocker's 4.5, became the next blocker
    /// at 19.6×). Path points join the Kabsch-aligned, atom-matched partner (SHOULD-T1) and are
    /// dropped when their closest pair is closer than the blocker's and every fitted geometry's of the
/// same atom sequence (no constant).
    pub(super) fn teach_jobs(&mut self, blocker: usize, template: &Scene, seed: u64) -> Vec<TargetJob> {
        self.ensure_desc_cache();
        let Some(store) = self.store.as_ref() else { return Vec::new() };
        let s = store.samples[blocker].clone();
        let base = Self::sample_scene(&s, template);
        let radius = THIN_FRACTION * self.policy.max_descriptor_distance;
        let d_blocker = self.data_distance(&s.pos, &s.z, blocker);
        let mut kept: Vec<Scene> = Vec::new();
        let admissible = |me: &Self, sc: &Scene, kept: &[Scene]| -> bool {
            let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
            let z: Vec<u8> = sc.nuclei.iter().map(|n| n.z).collect();
            sc.check_inside_cell().is_ok()
                && me.geometry_distance((&pos, &z), (&s.pos, &s.z)) > radius
                && kept.iter().all(|k| {
                    let kp: Vec<[f64; 3]> = k.nuclei.iter().map(|n| n.pos).collect();
                    me.geometry_distance((&pos, &z), (&kp, &z)) > radius
                })
                && me.data_distance(&pos, &z, blocker) <= d_blocker
        };
        // nudges: the pass's generator and width, widened ×1.5 after every 40 rejections
        let mut rng = StdRng::seed_from_u64(seed ^ (blocker as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let sigma0 = SamplingPlan::default().perturbation_sigma;
        let mut perturbed = Vec::new();
        for attempt in 0..120 {
            if perturbed.len() >= TEACH_PERTURBATIONS {
                break;
            }
            let sigma = sigma0 * 1.5f64.powi((attempt / 40) as i32);
            let mut sc = base.clone();
            for n in &mut sc.nuclei {
                for x in &mut n.pos {
                    *x += sigma * sim::thermostat::standard_normal(&mut rng);
                }
            }
            if admissible(self, &sc, &kept) {
                kept.push(sc.clone());
                perturbed.push(sc);
            }
        }
        // the path: toward the nearest fitted sample of the same atom sequence, matched and aligned
        let store = self.store.as_ref().expect("checked");
        let partner = (0..self.accumulated.min(store.len()))
            .filter(|&i| i != blocker && !self.excluded(i) && store.samples[i].z == s.z)
            .map(|i| (i, self.geometry_distance((&store.samples[i].pos, &store.samples[i].z), (&s.pos, &s.z))))
            .filter(|(_, d)| d.is_finite())
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i);
        let mut path = Vec::new();
        let (points, floor): (Vec<Vec<[f64; 3]>>, f64) = match partner {
            Some(i) => {
                let aligned = matched_partner(&s.pos, &s.z, &store.samples[i].pos);
                // the closest pair any fitted geometry of this atom sequence has (the partner's
                // included) and the blocker's: below both, atoms pass through each other. Not the
                // two endpoints alone — a bond that turns between them shortens on the straight
                // path (its chord): 1.914606 → 1.914562 Bohr on the user's #438, measured
                let data_min = (0..self.accumulated.min(store.len()))
                    .filter(|&j| !self.excluded(j) && store.samples[j].z == s.z)
                    .map(|j| closest_pair(&store.samples[j].pos))
                    .fold(f64::INFINITY, f64::min);
                let floor = closest_pair(&s.pos).min(data_min);
                let pts = (1..=TEACH_PATH_POINTS)
                    .map(|k| {
                        let t = k as f64 / (TEACH_PATH_POINTS + 1) as f64;
                        s.pos.iter().zip(&aligned).map(|(a, b)| std::array::from_fn(|c| a[c] + t * (b[c] - a[c]))).collect()
                    })
                    .collect();
                (pts, floor)
            }
            None if s.pos.len() >= 2 => {
                // the most isolated atom toward its nearest neighbour, to 90 … 60 % of the distance
                let nn = |k: usize| -> (usize, f64) {
                    (0..s.pos.len())
                        .filter(|&m| m != k)
                        .map(|m| (m, (0..3).map(|a| (s.pos[k][a] - s.pos[m][a]).powi(2)).sum::<f64>().sqrt()))
                        .min_by(|a, b| a.1.total_cmp(&b.1))
                        .expect("two atoms")
                };
                let k = (0..s.pos.len()).max_by(|&a, &b| nn(a).1.total_cmp(&nn(b).1)).expect("atoms");
                let (m, _) = nn(k);
                let pts = (1..=TEACH_PATH_POINTS)
                    .map(|j| {
                        let f = 1.0 - 0.1 * j as f64;
                        let mut p = s.pos.clone();
                        p[k] = std::array::from_fn(|a| s.pos[m][a] + f * (s.pos[k][a] - s.pos[m][a]));
                        p
                    })
                    .collect();
                (pts, 0.0)
            }
            None => (Vec::new(), 0.0),
        };
        for p in points {
            if closest_pair(&p) < floor {
                continue;
            }
            let mut sc = base.clone();
            for (n, x) in sc.nuclei.iter_mut().zip(p) {
                n.pos = x;
            }
            if admissible(self, &sc, &kept) {
                kept.push(sc.clone());
                path.push(sc);
            }
        }
        // the held-out checks: one nudge and one path point (two nudges without a path)
        let mut jobs: Vec<TargetJob> = Vec::new();
        let hold_p = if path.is_empty() { 2 } else { 1 };
        for (k, sc) in perturbed.into_iter().enumerate() {
            jobs.push(TargetJob { scene: sc, blocker, holdout: k < hold_p });
        }
        let mid = path.len() / 2;
        for (k, sc) in path.into_iter().enumerate() {
            jobs.push(TargetJob { scene: sc, blocker, holdout: k == mid });
        }
        jobs
    }

    /// Starts a teach-around pass over the offer's blockers (module doc; `Trainable::start_teach`).
    pub fn start_teach(&mut self, scene: &Scene) -> Result<(), String> {
        if self.pass.is_some() {
            return Err("a pass is already running (cancel it first)".into());
        }
        if self.model.is_none() {
            return Err("no model yet — run a sampling pass first".into());
        }
        let Some(offer) = self.teach_offer() else {
            return Err("no environment class is blocked by samples the rest of the data cannot predict".into());
        };
        if offer.blockers.is_empty() {
            let why: Vec<String> = offer.refused.iter().map(|(i, c, r, why)| format!("#{i} ({c}, in-sample {r:.2}×): {why}")).collect();
            return Err(format!("nothing to teach: {}", why.join("; ")));
        }
        let class_index = |name: &str| ENV_CLASSES.iter().position(|c| *c == name).unwrap_or(0);
        let mut classes: Vec<usize> = offer.blockers.iter().map(|b| class_index(b.1)).collect();
        classes.sort_unstable();
        classes.dedup();
        let mut jobs = Vec::new();
        for &(i, c, ..) in &offer.blockers {
            let round = self.teach_rounds.get(&class_index(c)).copied().unwrap_or(0);
            jobs.extend(self.teach_jobs(i, scene, 0x7EAC_u64 + round as u64));
        }
        if jobs.is_empty() {
            return Err("no geometry around the blockers lies inside the cell and no farther from the data than the blockers".into());
        }
        let names: Vec<String> = offer.blockers.iter().map(|(i, c, loo, _)| format!("#{i} ({c}, leave-one-out {loo:.1}×)")).collect();
        let held = jobs.iter().filter(|j| j.holdout).count();
        self.user_notes.push(format!(
            "teaching around {}: {} solves, {held} of them held out to check the result",
            names.join(", "),
            jobs.len()
        ));
        for &c in &classes {
            *self.teach_rounds.entry(c).or_insert(0) += 1;
        }
        self.teach_classes = classes;
        let basis = self.model.as_ref().map(|m| m.basis.clone());
        self.teach_start = Some(TeachStart {
            loo_before: self.class_loo.iter().map(|c| c.force_max.max(c.energy_max)).collect(),
            samples_before: self.samples(),
            degree_before: basis.as_ref().filter(|b| b.body3).map(|b| b.degree_cap),
            q_before: self.report.as_ref().map_or(0, |r| r.q_prime),
            started: Instant::now(),
        });
        self.pass_start = Some(self.samples());
        self.pass = Some(crate::sampling::Sampler::targeted(scene, jobs));
        Ok(())
    }

    /// After a teach-around pass and its refit (module doc, 3.–4.): the held-out checks judged and
    /// then kept (SHOULD-T4), the verdict ([`teach_verdict`]), the outcome in plain words
    /// (SHOULD-T5; `user_notes`). A failed check is a failed audit with its sample's index.
    pub(super) fn finish_teach(&mut self, targets: &[usize], holdouts: Vec<Holdout>, progress: &SamplingProgress) {
        let cancelled = progress.cancelled;
        let classes_counted = std::mem::take(&mut self.teach_classes);
        if cancelled {
            // a cancelled round is not a round spent
            for c in classes_counted {
                if let Some(r) = self.teach_rounds.get_mut(&c) {
                    *r = r.saturating_sub(1);
                }
            }
        }
        let start = self.teach_start.take();
        let secs = start.as_ref().map_or(0.0, |s| s.started.elapsed().as_secs_f64());
        let solves = progress.done.saturating_sub(progress.skipped);
        if !cancelled {
            self.teach_session.rounds += 1;
        }
        self.teach_session.solves += solves;
        self.teach_session.secs += secs;
        let targets_s: Vec<String> = targets.iter().map(|i| format!("#{i}")).collect();
        let Some(model) = self.model.clone() else {
            self.user_notes.push(format!("teaching around {}: not accepted — no model after the pass (not enough samples to fit)", targets_s.join(", ")));
            return;
        };
        let rule = box_affected_face_density(self.store.as_ref().map_or(0.25, |s| s.provenance.h));
        let ionic = self.store.as_ref().map_or("none".to_string(), |s| s.provenance.ionic.clone());
        let has_store = self.store.is_some();
        let (mut judged, mut passed, mut deferred, mut worst) = (0usize, 0usize, 0usize, 0.0f64);
        let mut failed: Vec<FailedAudit> = Vec::new();
        // R3-2: the checks judged near each blocker
        let mut judged_near: BTreeMap<usize, usize> = targets.iter().map(|&t| (t, 0)).collect();
        let mut deferred_near: BTreeMap<usize, usize> = BTreeMap::new();
        for h in holdouts {
            if h.unfit {
                continue; // must not be trained on, not judged either
            }
            let pos: Vec<[f64; 3]> = h.scene.nuclei.iter().map(|n| n.pos).collect();
            let z: Vec<u8> = h.scene.nuclei.iter().map(|n| n.z).collect();
            if !has_store {
                continue;
            }
            // the settings of the scene the check was solved on (its box included: a shared
            // collection records it; per scene this equals the store's provenance)
            let prov = &self.scene_provenance(&h.scene);
            let mut sink = Vec::new();
            // outside strict mode's domain (Minor): held out like a draft guess, judged once strict
            // mode reaches it (`judge_held_out`)
            if self.geometry_refusal(&pos, &z).is_some() {
                if self.keep_sample(&pos, &z, h.energy, &h.forces, h.face, None, prov, SampleTag::DraftAudit, &mut sink) {
                    deferred += 1;
                    *deferred_near.entry(h.blocker).or_insert(0) += 1;
                }
                continue;
            }
            let boxed = h.face.is_some_and(|f| f > rule);
            let verdict = if boxed {
                None
            } else {
                match model.predict(&pos, &z, false) {
                    Ok(p) => {
                        let n = pos.len().max(1) as f64;
                        let e = (h.energy - p.energy).abs() / n / self.policy.energy_std_per_atom;
                        let f = worst_force_residual(&self.policy, &p.forces, &h.forces, audit_force_floor(&ionic)).map_or(0.0, |(d, b)| d / b);
                        Some(e.max(f))
                    }
                    Err(_) => None,
                }
            };
            // judged, then kept (SHOULD-T4): validate, then train on all
            if !self.keep_sample(&pos, &z, h.energy, &h.forces, h.face, None, prov, SampleTag::Targeted, &mut sink) {
                continue;
            }
            let index = self.samples() - 1;
            if let Some(r) = verdict {
                judged += 1;
                *judged_near.entry(h.blocker).or_insert(0) += 1;
                worst = worst.max(r);
                if r <= 1.0 {
                    passed += 1;
                } else {
                    failed.push(FailedAudit { pos, z, energy: h.energy, forces: h.forces, index: Some(index), ionic: ionic.clone(), rejudges: 0, origin: AuditOrigin::TeachCheck, pair: None });
                }
            }
        }
        let n_failed = failed.len();
        self.failed_audits.extend(failed);
        self.sync_audit_failed();
        // the class rule, unchanged — read over the short classes too (MUST-T1)
        let all = self.diagnose(true);
        let still: Vec<&Blocker> = all.iter().filter(|b| targets.contains(&b.index)).collect();
        let out_of_fit: Vec<usize> = targets.iter().copied().filter(|&i| i >= self.accumulated || self.excluded(i)).collect();
        let unjudged: Vec<usize> = judged_near.iter().filter(|(_, &n)| n == 0).map(|(&t, _)| t).collect();
        let verdict = teach_verdict(&self.class_loo, !still.is_empty(), !out_of_fit.is_empty(), n_failed, unjudged.len(), cancelled);
        let s = self.teach_session;
        let headline = format!(
            "teaching around {}: {} — round {} ({solves} DFT solves, {}; {} in {} round{} since the count restarted)",
            targets_s.join(", "),
            match verdict {
                TeachVerdict::Accepted => "accepted",
                TeachVerdict::NeedsSamples => "not accepted yet: the blockers are predicted, but a kind of environment is short of samples",
                TeachVerdict::NotAccepted => "not accepted",
                TeachVerdict::Cancelled => "cancelled",
            },
            s.rounds.max(1),
            duration_words(secs),
            s.solves,
            s.rounds.max(1),
            if s.rounds > 1 { "s" } else { "" }
        );
        let mut lines = vec![headline];
        // the basis growth (SHOULD-T3)
        let degree_now = Some(model.basis.degree_cap).filter(|_| model.basis.body3);
        let q_now = self.report.as_ref().map_or(0, |r| r.q_prime);
        let grew = start.as_ref().and_then(|st| match (st.degree_before, degree_now) {
            (Some(a), Some(b)) if b != a => Some((a, b, st.q_before)),
            _ => None,
        });
        if let Some((a, b, q0)) = grew {
            lines.push(format!("– the taught region needed the richer basis (degree {a} → {b}): the sample floor of every kind of environment rose {q0} → {q_now}"));
        }
        let samples_before = start.as_ref().map_or(usize::MAX, |st| st.samples_before);
        for (c, l) in self.class_loo.iter().enumerate() {
            if l.samples == 0 {
                continue;
            }
            let now = l.force_max.max(l.energy_max);
            let before = start.as_ref().and_then(|st| st.loo_before.get(c).copied());
            let change = before.map_or(format!("{now:.2}×"), |b| format!("{b:.1}× → {now:.2}×"));
            let words = CLASS_WORDS[c];
            let line = match l.refusal {
                None => format!("– {words}: trusted (leave-one-out {change})"),
                Some((sim::ClassKey::Samples, n, q)) if now <= 1.0 => format!(
                    "– {words}: predicted (leave-one-out {change}), but at this basis they need {q:.0} samples and have {n:.0}{} — a sampling pass on this scene adds them",
                    if grew.is_some() { " (the floor rose with the basis, no sample was removed)" } else { "" }
                ),
                Some(_) => {
                    let w = self.class_worst.get(c).copied().flatten().map(|(i, _)| i);
                    let who = match w {
                        Some(i) if i >= samples_before => format!("; the worst is now #{i}, a sample from this teaching"),
                        Some(i) if targets.contains(&i) => format!("; the worst is still #{i}, the blocker itself"),
                        Some(i) => format!("; the worst is now #{i}, an older sample"),
                        None => String::new(),
                    };
                    let rounds = self.teach_rounds.get(&c).copied().unwrap_or(0);
                    let teachable = w.and_then(|i| all.iter().find(|b| b.index == i)).is_none_or(|b| b.teachable());
                    let next = if !teachable {
                        "; the model cannot represent it (its own fit residual is beyond the bound): no more rounds".to_string()
                    } else if rounds >= TEACH_ROUNDS {
                        "; no further teach rounds are offered for this kind (two spent)".to_string()
                    } else {
                        "; one more round is offered".to_string()
                    };
                    let short = match l.refusal {
                        Some((sim::ClassKey::Samples, n, q)) => format!(" and short of samples ({n:.0} of {q:.0})"),
                        _ => String::new(),
                    };
                    format!("– {words}: leave-one-out {change}{short}{who}{next}")
                }
            };
            lines.push(line);
        }
        if judged == 0 && deferred == 0 {
            lines.push("– no held-out check could be judged (box-affected or unfit answers)".into());
        } else {
            let mut l = format!(
                "– {} of {} check geometries judged{}; {passed} of {judged} pass the audit criterion (worst {worst:.2}× its bound)",
                judged,
                judged + deferred,
                if deferred > 0 { format!(", {deferred} held until strict mode reaches them") } else { String::new() }
            );
            if n_failed > 0 {
                l.push_str(&format!(
                    "; {n_failed} off, so the model is untrusted until it predicts {} — meanwhile every step is DFT. This is the check working, not a fault",
                    if n_failed == 1 { "that geometry" } else { "those geometries" }
                ));
            }
            lines.push(l);
        }
        for t in &unjudged {
            lines.push(format!(
                "– no check near #{t} could be judged{} — not accepted yet; {} judged when strict mode reaches {}",
                if deferred_near.get(t).copied().unwrap_or(0) > 0 { " (outside strict mode's domain)" } else { " (box-affected, unfit or failed solves)" },
                if deferred_near.get(t).copied().unwrap_or(0) > 0 { "they will be" } else { "a later round's checks will be" },
                if deferred_near.get(t).copied().unwrap_or(0) > 0 { "them" } else { "its region" }
            ));
        }
        if let Some(why) = self.untrusted_reason() {
            lines.push(format!("– the model as a whole: untrusted — {why}"));
        }
        if !out_of_fit.is_empty() {
            let o: Vec<String> = out_of_fit.iter().map(|i| format!("#{i}")).collect();
            lines.push(format!("– {} left the fit at the refit (quarantine), so the class rule no longer reads it — not an acceptance", o.join(", ")));
        }
        let kept_holdouts = judged + deferred;
        if kept_holdouts > 0 {
            lines.push(format!("– the {kept_holdouts} check geometr{} kept as samples (the judged ones fitted at the next refit, those outside strict mode's domain held out until it reaches them)", if kept_holdouts == 1 { "y is" } else { "ies are" }));
        }
        if verdict == TeachVerdict::Accepted {
            self.reset_teach_rounds();
        }
        self.user_notes.push(lines.join("\n"));
    }

    /// A check's verdict against the model that stands (the audit criterion: |ΔE|/N over the energy
    /// tier, the worst atom's force residual over its bound with the surface's pseudo floor): the
    /// two ratios, `None` when the model cannot predict there.
    pub(super) fn check_ratios(&self, pos: &[[f64; 3]], z: &[u8], energy: f64, forces: &[[f64; 3]]) -> Option<(f64, f64)> {
        let model = self.model.as_ref()?;
        let ionic = self.store.as_ref().map_or("none".to_string(), |s| s.provenance.ionic.clone());
        let p = model.predict(pos, z, false).ok()?;
        let n = pos.len().max(1) as f64;
        let e = (energy - p.energy).abs() / n / self.policy.energy_std_per_atom;
        let f = worst_force_residual(&self.policy, &p.forces, forces, audit_force_floor(&ionic)).map_or(0.0, |(d, b)| d / b);
        Some((e, f))
    }

    /// The pass's shell checks after its refit (`sampling` module doc, 3.; round 3 review R3-1):
    /// each judged by [`check_ratios`](Self::check_ratios); within 1× → "the model's cutoff holds for
    /// <pair>"; beyond → the named limitation and, by the **Ruling (shell → fence)** (2026-10-19), an
    /// audit fence of origin "shell check" at once: a measured structural limitation of the cutoff,
    /// not a model error a refit can cure, so it neither untrusts the model globally nor waits
    /// `AUDIT_REJUDGE_LIMIT` refits — strict mode refuses geometries with that pair in the shell band
    /// (`shell_fence_refusal`; the fenced pair is the check's own `ShellCheck::pair`, carried on
    /// the case — final review I2) and near the case (the quarantine radius), and answers everywhere
    /// else as before; the class verdicts are untouched. Never a sample, and never lifted by a
    /// refit's re-judgement (final review C1): only a later pass's shell check for the pair
    /// measures the shell again. Box-affected or unfit answers are reported as not judged.
    #[cfg(test)]
    pub(super) fn judge_shell_checks(&mut self, shells: Vec<crate::sampling::ShellCheck>) {
        self.judge_shell_checks_with(shells, None)
    }

    /// [`judge_shell_checks`](Self::judge_shell_checks) with a `caveat` on the fit it judges against (an untrusted fit, or
    /// one without this pass's samples — `finish_pass`): the verdict is reached and a failure fenced as before, and each
    /// note adds "(<caveat>: possibly the fit's error, not the cutoff's — the next pass's shell check measures it again)".
    pub(super) fn judge_shell_checks_with(&mut self, shells: Vec<crate::sampling::ShellCheck>, caveat: Option<&str>) {
        let tail = caveat.map_or(String::new(), |c| format!(" ({c}: possibly the fit's error, not the cutoff's — the next pass's shell check measures it again)"));
        let rule = box_affected_face_density(self.store.as_ref().map_or(0.25, |s| s.provenance.h));
        let ionic = self.store.as_ref().map_or("none".to_string(), |s| s.provenance.ionic.clone());
        for c in shells {
            let name = format!("{}–{}", units::elements::symbol(c.pair.0), units::elements::symbol(c.pair.1));
            if c.unfit || c.face.is_some_and(|f| f > rule) {
                self.user_notes.push(format!(
                    "sampling: the shell check of {name} at r = {:.2} Bohr could not be judged ({}) — the cutoff shell of this pair stays unmeasured",
                    c.r,
                    if c.unfit { "an answer that must not be trained on" } else { "box-affected" }
                ));
                continue;
            }
            let pos: Vec<[f64; 3]> = c.scene.nuclei.iter().map(|n| n.pos).collect();
            let z: Vec<u8> = c.scene.nuclei.iter().map(|n| n.z).collect();
            let Some((e, f)) = self.check_ratios(&pos, &z, c.energy, &c.forces) else {
                self.leave_shell_checks_unjudged(std::slice::from_ref(&c), "the model cannot predict this geometry");
                continue;
            };
            if e <= 1.0 && f <= 1.0 {
                self.user_notes.push(format!(
                    "sampling: the model's cutoff holds for {name}: at r = {:.2} Bohr the interaction left is within the tiers (energy {e:.2}× the tier, forces {f:.2}× their bound){tail}",
                    c.r
                ));
            } else {
                let gap = self.model.as_ref().and_then(|m| m.predict(&pos, &z, false).ok()).map_or(f64::NAN, |p| c.energy - p.energy);
                self.user_notes.push(format!(
                    "sampling: {name} still interacts at r = {:.2} Bohr (teacher − model {gap:+.1e} Ha: energy {e:.2}× the tier, forces {f:.2}× their bound); the model's cutoff ({:.2} Bohr) truncates it — a structural limitation: geometries with {name} in the cutoff shell ({:.2}–{:.2} Bohr) fall back to DFT (a shell fence); everywhere else the model answers as before{tail}",
                    c.r,
                    c.r_cut,
                    crate::sampling::SCAN_CUTOFF_FRACTION * c.r_cut,
                    c.r_cut
                ));
                let pair = (c.pair.0.min(c.pair.1), c.pair.0.max(c.pair.1));
                self.audit_fences.push(FailedAudit { pos, z, energy: c.energy, forces: c.forces, index: None, ionic: ionic.clone(), rejudges: AUDIT_REJUDGE_LIMIT, origin: AuditOrigin::ShellCheck, pair: Some(pair) });
            }
        }
    }

    /// The pass's shell checks that are not judged (final review M2): solved, but the pass left no
    /// fit of its own to judge them against (`why`), so each pair's cutoff shell stays unmeasured —
    /// said, never a silent drop, and never a verdict of an earlier fit. No fence, no sample.
    pub(super) fn leave_shell_checks_unjudged(&mut self, shells: &[crate::sampling::ShellCheck], why: &str) {
        for c in shells {
            self.user_notes.push(format!(
                "sampling: the shell check of {}–{} at r = {:.2} Bohr could not be judged ({why}) — the cutoff shell of this pair stays unmeasured",
                units::elements::symbol(c.pair.0),
                units::elements::symbol(c.pair.1),
                c.r
            ));
        }
    }

    /// The per-class round counter restarts (SHOULD-T6): after a regular sampling pass, after an
    /// accepted round, and per class once that class is trusted (`refresh_class_loo`).
    pub(super) fn reset_teach_rounds(&mut self) {
        self.teach_rounds.clear();
        self.teach_session = TeachSession::default();
    }
}

/// The private helpers, for the unit tests.
#[cfg(test)]
pub(super) mod tests_api {
    pub(in crate::surrogate) use super::{centroid, closest_pair, kabsch, matched_partner, teach_verdict, TeachVerdict};
}
