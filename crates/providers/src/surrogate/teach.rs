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
//!    offer rounds without end.
//! 2. **Geometries** per blocker (`teach_jobs`), the blocker's own composition and cell:
//!    [`TEACH_PERTURBATIONS`] displacements by the pass's own generator (independent Gaussians of
//!    the plan's `perturbation_sigma` per coordinate; the width grows ×1.5 after every 40 rejected
//!    draws, at most twice) each farther than `THIN_FRACTION`·`max_descriptor_distance` (the
//!    thinning radius, training-spread units) from the blocker and from each other — no copy can
//!    defeat the leave-one-out; and [`TEACH_PATH_POINTS`] along the path to the nearest fitted
//!    sample of the same atom sequence (Kabsch-aligned onto the blocker, linear in between): the
//!    region *between* is what an outlier speaks for (the fence experiment's 6.33×). Without such a
//!    sample, the blocker's most isolated atom is brought toward its nearest neighbour, to 90, 80,
//!    70 and 60 % of the distance (the review's #345: the H approaching O from 5.9 to ≈ 3.5 Bohr;
//!    chosen geometry, not a tolerance). The cell guard, the box rule and the unfit rules apply as
//!    to any pass sample.
//! 3. **Held-out checks**: per blocker one perturbation and one path point (two perturbations
//!    without a path) are solved but never fitted. After the refit each must pass the audit
//!    criterion against the plain prediction (energy tier, every atom's force bound at 1×, the
//!    pseudo floor); one that fails is a failed audit (rule 8: untrusted until re-earned).
//! 4. **Acceptance is the unchanged class rule**: the class's max LOO ≤ 1 over *all* its fitted
//!    samples — the blocker and the new ones included; it is never removed, fenced or
//!    down-weighted — energy ≤ tier, n ≥ Q′; plus 3. One round, at most one more on what still
//!    blocks; then the outcome names the blocker with its in-sample and LOO numbers and stops. The
//!    blocker must still be in the fit after the refit (a recovery that quarantined it is not an
//!    acceptance); a cancelled round is not a round spent.
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
//! - `surrogate_water::teach_around_the_users_water_h2_model` (a copy of the user's file, 95³, O
//!   cores-only, 657 samples; blockers #345 pair 12.0×, #438 10.6×, #439 3.56×, #87 1.52× — all
//!   in-sample ≤ 0.40×). Round 1: 36 solves (4 near #87 stalled in the SCF and were skipped),
//!   29 kept, 7 of 7 held-out checks pass (worst 0.76×); pair max 12.0× → 0.83×, but the refit's
//!   basis grew to Q′ 344 (the loaded fit stood at a lower degree), so the pair class is now short
//!   of samples (197 of 344); many-body 19.6× at a round-1 sample (#664). Round 2 on #664, #667,
//!   #663: 28 solves, 22 kept; many-body 1.47×, 5 of 6 checks pass, one at 1.66× → a failed audit.
//!   Not accepted after two rounds, 64 solves, 896 s of GPU time: the verdict at the scene went
//!   from "pair untrusted, #345 12.0×" to "untrusted: an audit found forces off by 3.3e-3 Ha/Bohr".
use super::*;
use crate::sampling::{Holdout, TargetJob};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Displacements per blocker (the review: ≈ 6).
pub const TEACH_PERTURBATIONS: usize = 6;
/// Points on the path to the nearest fitted sample of the blocker's atom sequence (≈ 4).
pub const TEACH_PATH_POINTS: usize = 4;
/// Blockers per offer (the review: at most 5).
pub const TEACH_MAX_BLOCKERS: usize = 5;
/// Rounds per blocker: one, and one more on what still blocks.
pub const TEACH_ROUNDS: u32 = 2;

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

impl SurrogateKs {
    /// The diagnosis (module doc, 1.): every blocker of every class untrusted by its LOO numbers
    /// with n ≥ Q′, worst first (one entry per sample: the class where it is worst).
    pub fn teach_diagnosis(&self) -> Vec<Blocker> {
        let (Some(model), Some(store)) = (self.model.as_ref(), self.store.as_ref()) else { return Vec::new() };
        let mut out: Vec<Blocker> = Vec::new();
        for (c, l) in self.class_loo.iter().enumerate() {
            if l.trusted || matches!(l.refusal, Some((sim::ClassKey::Samples, ..))) {
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
    /// blockers with rounds left (at most `TEACH_MAX_BLOCKERS`), the refused ones with why, and the
    /// solves (`TEACH_PERTURBATIONS` + `TEACH_PATH_POINTS` per blocker — the generator may make
    /// fewer). `None` when nothing blocks.
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
        Some(sim::TeachOffer { blockers, refused, refused_total, solves })
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

    /// The geometries around one blocker (module doc, 2.–3.): (job, is a path point).
    pub(super) fn teach_jobs(&self, blocker: usize, template: &Scene, seed: u64) -> Vec<TargetJob> {
        let Some(store) = self.store.as_ref() else { return Vec::new() };
        let s = &store.samples[blocker];
        let base = Self::sample_scene(s, template);
        let radius = THIN_FRACTION * self.policy.max_descriptor_distance;
        let mut kept: Vec<Scene> = Vec::new();
        let far_enough = |me: &Self, sc: &Scene, kept: &[Scene]| -> bool {
            let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
            let z: Vec<u8> = sc.nuclei.iter().map(|n| n.z).collect();
            sc.check_inside_cell().is_ok()
                && me.geometry_distance((&pos, &z), (&s.pos, &s.z)) > radius
                && kept.iter().all(|k| {
                    let kp: Vec<[f64; 3]> = k.nuclei.iter().map(|n| n.pos).collect();
                    me.geometry_distance((&pos, &z), (&kp, &z)) > radius
                })
        };
        // perturbations: the pass's generator and width, widened ×1.5 after every 40 rejections
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
            if far_enough(self, &sc, &kept) {
                kept.push(sc.clone());
                perturbed.push(sc);
            }
        }
        // the path: toward the nearest fitted sample of the same atom sequence, aligned
        let partner = (0..self.accumulated.min(store.len()))
            .filter(|&i| i != blocker && !self.excluded(i) && store.samples[i].z == s.z)
            .map(|i| (i, self.geometry_distance((&store.samples[i].pos, &store.samples[i].z), (&s.pos, &s.z))))
            .filter(|(_, d)| d.is_finite())
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i);
        let mut path = Vec::new();
        let points: Vec<Vec<[f64; 3]>> = match partner {
            Some(i) => {
                let q = &store.samples[i].pos;
                let (cb, cq) = (centroid(&s.pos), centroid(q));
                let pb: Vec<[f64; 3]> = s.pos.iter().map(|x| std::array::from_fn(|a| x[a] - cb[a])).collect();
                let pq: Vec<[f64; 3]> = q.iter().map(|x| std::array::from_fn(|a| x[a] - cq[a])).collect();
                let r = kabsch(&pq, &pb);
                let aligned: Vec<[f64; 3]> = pq
                    .iter()
                    .map(|x| {
                        let v = r * nalgebra::Vector3::new(x[0], x[1], x[2]);
                        [v[0] + cb[0], v[1] + cb[1], v[2] + cb[2]]
                    })
                    .collect();
                (1..=TEACH_PATH_POINTS)
                    .map(|k| {
                        let t = k as f64 / (TEACH_PATH_POINTS + 1) as f64;
                        s.pos.iter().zip(&aligned).map(|(a, b)| std::array::from_fn(|c| a[c] + t * (b[c] - a[c]))).collect()
                    })
                    .collect()
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
                (1..=TEACH_PATH_POINTS)
                    .map(|j| {
                        let f = 1.0 - 0.1 * j as f64;
                        let mut p = s.pos.clone();
                        p[k] = std::array::from_fn(|a| s.pos[m][a] + f * (s.pos[k][a] - s.pos[m][a]));
                        p
                    })
                    .collect()
            }
            None => Vec::new(),
        };
        for p in points {
            let mut sc = base.clone();
            for (n, x) in sc.nuclei.iter_mut().zip(p) {
                n.pos = x;
            }
            if far_enough(self, &sc, &kept) {
                kept.push(sc.clone());
                path.push(sc);
            }
        }
        // the held-out checks: one perturbation and one path point (two perturbations without a path)
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
        let class_of = |i: usize| ENV_CLASSES.iter().position(|c| offer.blockers.iter().any(|b| b.0 == i && b.1 == *c)).unwrap_or(0);
        let mut classes: Vec<usize> = offer.blockers.iter().map(|b| class_of(b.0)).collect();
        classes.sort_unstable();
        classes.dedup();
        let mut jobs = Vec::new();
        for &(i, ..) in &offer.blockers {
            let round = self.teach_rounds.get(&class_of(i)).copied().unwrap_or(0);
            jobs.extend(self.teach_jobs(i, scene, 0x7EAC_u64 + round as u64));
        }
        if jobs.is_empty() {
            return Err("no geometry around the blockers lies inside the cell".into());
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
        self.pass_start = Some(self.samples());
        self.pass = Some(crate::sampling::Sampler::targeted(scene, jobs));
        Ok(())
    }

    /// After a teach-around pass and its refit (module doc, 3.–4.): the held-out checks judged, the
    /// class rule read, the outcome in plain words (`user_notes`). A failed check is a failed audit.
    pub(super) fn finish_teach(&mut self, targets: &[usize], holdouts: Vec<Holdout>, cancelled: bool) {
        if cancelled {
            // a cancelled round is not a round spent
            for c in std::mem::take(&mut self.teach_classes) {
                if let Some(r) = self.teach_rounds.get_mut(&c) {
                    *r = r.saturating_sub(1);
                }
            }
        }
        self.teach_classes.clear();
        let Some(model) = self.model.as_ref() else {
            self.user_notes.push("teaching around the blockers: no model after the pass (not enough samples to fit) — not accepted".into());
            return;
        };
        let rule = box_affected_face_density(self.store.as_ref().map_or(0.25, |s| s.provenance.h));
        let ionic = self.store.as_ref().map_or("none".to_string(), |s| s.provenance.ionic.clone());
        let (mut judged, mut passed, mut worst) = (0usize, 0usize, 0.0f64);
        let mut failed: Vec<FailedAudit> = Vec::new();
        for h in &holdouts {
            if h.unfit || h.face.is_some_and(|f| f > rule) {
                continue;
            }
            let pos: Vec<[f64; 3]> = h.scene.nuclei.iter().map(|n| n.pos).collect();
            let z: Vec<u8> = h.scene.nuclei.iter().map(|n| n.z).collect();
            let Ok(p) = model.predict(&pos, &z, false) else { continue };
            let n = pos.len().max(1) as f64;
            let e = (h.energy - p.energy).abs() / n / self.policy.energy_std_per_atom;
            let f = worst_force_residual(&self.policy, &p.forces, &h.forces, audit_force_floor(&ionic)).map_or(0.0, |(d, b)| d / b);
            judged += 1;
            worst = worst.max(e.max(f));
            if e <= 1.0 && f <= 1.0 {
                passed += 1;
            } else {
                failed.push(FailedAudit { pos, z, energy: h.energy, forces: h.forces.clone(), index: None, ionic: ionic.clone(), rejudges: 0 });
            }
        }
        let n_failed = failed.len();
        self.failed_audits.extend(failed);
        self.sync_audit_failed();
        // the class rule, unchanged: every class a target blocked
        let diag = self.teach_diagnosis();
        let still: Vec<&Blocker> = diag.iter().filter(|b| targets.contains(&b.index)).collect();
        let classes_ok = self.class_loo.iter().all(|c| c.trusted || matches!(c.refusal, Some((sim::ClassKey::Samples, ..))));
        // the blocker must still be in the fit: a refit recovery that held it out (quarantine) does
        // not count — the class rule would no longer be read over it
        let out_of_fit: Vec<usize> = targets.iter().copied().filter(|&i| i >= self.accumulated || self.excluded(i)).collect();
        let accepted = !cancelled && still.is_empty() && out_of_fit.is_empty() && classes_ok && n_failed == 0 && judged > 0;
        let targets_s: Vec<String> = targets.iter().map(|i| format!("#{i}")).collect();
        let checks = if judged == 0 {
            "no held-out check could be judged (box-affected or unfit answers)".to_string()
        } else {
            format!("{passed} of {judged} held-out checks pass the audit criterion (worst {worst:.2}× its bound)")
        };
        let classes: Vec<String> = self
            .class_loo
            .iter()
            .filter(|c| c.samples > 0)
            .map(|c| {
                let v = match c.refusal {
                    None => "trusted".to_string(),
                    Some((sim::ClassKey::Samples, n, q)) => format!("not learned yet ({n:.0} of {q:.0} samples)"),
                    Some(_) => format!("untrusted ({:.1}× leave-one-out)", c.force_max.max(c.energy_max)),
                };
                format!("{} {v}", c.class)
            })
            .collect();
        let msg = if accepted {
            format!("teaching around {}: accepted — {}; {}", targets_s.join(", "), classes.join(", "), checks)
        } else {
            let left: Vec<String> = still
                .iter()
                .map(|b| {
                    let rounds = self.teach_rounds.get(&b.class).copied().unwrap_or(0);
                    format!(
                        "#{} still blocks ({}: in-sample {:.2}×, leave-one-out {:.1}×{})",
                        b.index,
                        ENV_CLASSES[b.class],
                        b.in_sample,
                        b.loo,
                        if !b.teachable() {
                            " — with its neighbourhood sampled the model cannot represent it (its own residual is beyond the bound): no more rounds"
                        } else if rounds >= TEACH_ROUNDS {
                            " — two rounds spent on this class, stopping"
                        } else {
                            " — one more round is offered"
                        }
                    )
                })
                .collect();
            format!(
                "teaching around {}: {} — {}; {}{}{}{}",
                targets_s.join(", "),
                if cancelled { "cancelled" } else { "not accepted" },
                classes.join(", "),
                checks,
                if left.is_empty() { String::new() } else { format!("; {}", left.join("; ")) },
                if out_of_fit.is_empty() {
                    String::new()
                } else {
                    let o: Vec<String> = out_of_fit.iter().map(|i| format!("#{i}")).collect();
                    format!("; {} left the fit at the refit (quarantine), so the class rule no longer reads it", o.join(", "))
                },
                if n_failed > 0 { format!("; {n_failed} failed check(s) recorded as failed audits (untrusted until re-earned)") } else { String::new() }
            )
        };
        self.user_notes.push(msg);
    }
}

/// The private helpers, for the unit tests.
#[cfg(test)]
pub(super) mod tests_api {
    pub(in crate::surrogate) use super::{centroid, kabsch};
}
