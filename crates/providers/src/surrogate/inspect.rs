//! A read-only diagnostic report of the surrogate's state ([`SurrogateKs::inspect`]) and the
//! policy's verdict at a geometry with the refusal's own text ([`SurrogateKs::explain`]): the
//! instrument of the 2026-10-18 pseudo-O water investigation (`.superpowers`, E0–E2), kept because
//! every question it answers ("why does a trusted model fall back here?", "are this scene's own
//! pass samples in the fit?") is one the badge alone cannot. Nothing here decides anything: no
//! field is written, the teacher is never called, and every number is one the policy already
//! computes (the trust checks, the class leave-one-out, the envelope, 5d, 5e, the fence).
//!
//! **Readiness** ([`SurrogateKs::readiness`], `TrainableStatus::readiness`; fix (a) of the trust
//! review): the strict policy's verdict at the scene's current geometry, so that the readout means
//! "strict mode will answer here" rather than only "the global trust rule passes" — the user's
//! model read "trusted" while every step fell back on rule 4b or 5. `evaluate` records the verdict
//! it reached (before draft mode or the relax verification overrode it; an accepting variance
//! verdict reused by the cadence counts as it does for the step) with the geometry, the model
//! generation and the trust flag; `readiness` returns it while the scene is that geometry under
//! that fit and trust, and otherwise evaluates [`first_refusal`](SurrogateKs::first_refusal) afresh
//! — the same rules in `evaluate`'s order (gate, non-finite, envelope 5a–5c/5e, novelty 5d, class
//! 4b, variance 6–7, fence) with a fresh variance. A class refusal names the class's worst
//! leave-one-out sample and where it came from (a pass stage, a fallback, an audit — strict or
//! draft — or a vote), because one sample alone decides a class (module doc, "Class-wise trust").
//!
//! Units: Hartree, Bohr; force ratios are ‖ΔF_j‖/√3 over the atom's bound (`force_bound`), energy
//! ratios |ΔE|/N over `energy_std_per_atom`, as the trust rule's.
use super::*;
use sim::Readiness;
use std::fmt::Write as _;

/// The strict verdict `evaluate` reached at a geometry (module doc, "Readiness"), with what it
/// depends on besides the positions: the model generation and the global trust flag.
pub(super) struct StrictVerdict {
    pos: Vec<[f64; 3]>,
    z: Vec<u8>,
    key: (u64, bool),
    refusal: Option<Refusal>,
}

/// Where a stored sample came from, in words (the readiness readout's worst-sample note).
pub(super) fn origin_words(tag: SampleTag) -> &'static str {
    match tag {
        SampleTag::Perturbation => "a sampling-pass perturbation",
        SampleTag::Scan => "a sampling-pass pair scan point",
        SampleTag::HotBurst => "a sampling-pass hot-burst step",
        SampleTag::Fallback => "a fallback answer",
        SampleTag::LiveStep => "a live step",
        SampleTag::Reference => "a reference",
        SampleTag::Audit => "a strict-mode audit",
        SampleTag::DraftAudit => "a draft-mode audit",
    }
}

/// A stored sample's state for the report: in the standing fit, kept since it (not fitted yet), or
/// one of the exclusions (quarantine codes).
fn state_name(code: Option<u8>, below_prefix: bool) -> &'static str {
    match code {
        Some(Q_HELD) => "quarantined",
        Some(Q_SUPERSEDED) => "superseded",
        Some(Q_BOX) => "box-excluded",
        Some(Q_DRAFT) => "held out (draft guess)",
        _ if !below_prefix => "unfitted",
        _ => "fitted",
    }
}

/// "O1H2"-style composition of a species list (sorted by Z, descending — O before H).
fn composition(z: &[u8]) -> String {
    let mut counts: BTreeMap<u8, usize> = BTreeMap::new();
    for &zi in z {
        *counts.entry(zi).or_default() += 1;
    }
    counts.iter().rev().map(|(z, n)| format!("{}{n}", units::isotopes::symbol(*z))).collect()
}

/// Every pair distance of a geometry (i < j), Bohr.
fn pair_distances(pos: &[[f64; 3]]) -> Vec<(usize, usize, f64)> {
    let mut v = Vec::new();
    for i in 0..pos.len() {
        for j in i + 1..pos.len() {
            v.push((i, j, (0..3).map(|a| (pos[i][a] - pos[j][a]).powi(2)).sum::<f64>().sqrt()));
        }
    }
    v
}

/// The smallest distance of a nucleus to a face of its cell (Bohr): `max_coord` − max |x|.
fn face_gap(scene: &Scene) -> f64 {
    let m = scene.cell.max_coord();
    scene.nuclei.iter().flat_map(|n| n.pos).fold(f64::INFINITY, |g, x| g.min(m - x.abs()))
}

impl SurrogateKs {
    /// Records the strict verdict `evaluate` reached at (`pos`, `z`) (module doc, "Readiness").
    pub(super) fn record_verdict(&mut self, pos: &[[f64; 3]], z: &[u8], refusal: Option<Refusal>) {
        let key = (self.model_gen, self.trusted());
        self.strict_verdict = Some(StrictVerdict { pos: pos.to_vec(), z: z.to_vec(), key, refusal });
    }

    /// The strict policy's first refusal at (`pos`, `z`) under `prov`, in `evaluate`'s order with a
    /// fresh variance (module doc, "Readiness"); `None` where it would answer. Read-only.
    pub(super) fn first_refusal(&self, pos: &[[f64; 3]], z: &[u8], prov: &Provenance) -> Option<Refusal> {
        if let Err(g) = self.gate(z, prov) {
            return Some(g.reason);
        }
        let model = self.model.as_ref()?;
        let pred = match model.predict(pos, z, true) {
            Ok(p) => p,
            Err(e) => return Some(Refusal::new(format!("surrogate: {e}"), TeachRule::NonFiniteAnswer, None)),
        };
        if let Some(r) = non_finite_answer(&pred) {
            return Some(r);
        }
        if let (_, Some(r)) = self.envelope_outside(model, pos, z) {
            return Some(r);
        }
        let posterior = match (pred.energy_std, pred.force_std_epistemic.as_ref()) {
            (Some(e), Some(f)) => Some((e, f)),
            _ => None,
        };
        if posterior.is_some() {
            if let Some((nu, at)) = self.novelty(pos, z).filter(|(nu, _)| *nu > 1.0) {
                return Some(novelty_refusal(nu, at));
            }
        }
        let rc2 = model.basis.r_cut * model.basis.r_cut;
        if let Some((r, _)) = self.class_untrusted(&neighbour_counts(pos, rc2)) {
            return Some(r);
        }
        if let Some((e, f)) = posterior {
            let v = self.variance_check(model, &pred, e, f);
            if !v.accept {
                return v.reason;
            }
        }
        if !self.near_quarantined_cached(pos, z).is_empty() {
            return Some(Refusal::new(
                format!("surrogate: within {:.2} training spreads of a quarantined geometry the model cannot represent", QUARANTINE_FRACTION * self.policy.max_descriptor_distance),
                TeachRule::NearQuarantine,
                None,
            ));
        }
        None
    }

    /// Whether the strict policy answers at `scene`'s geometry (module doc, "Readiness"); `None`
    /// without a model or nuclei. Read-only: the verdict `evaluate` recorded there when it still
    /// applies, else [`first_refusal`](Self::first_refusal) afresh.
    pub fn readiness(&self, scene: &Scene) -> Option<Readiness> {
        self.model.as_ref()?;
        if scene.nuclei.is_empty() {
            return None;
        }
        let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        let z: Vec<u8> = scene.nuclei.iter().map(|n| n.z).collect();
        let key = (self.model_gen, self.trusted());
        let refusal = match &self.strict_verdict {
            Some(v) if v.key == key && v.pos == pos && v.z == z => v.refusal.clone(),
            _ => self.first_refusal(&pos, &z, &self.scene_provenance(scene)),
        };
        // fix (e): the scene's own last pass, none of it in the fit
        let mut species = z.clone();
        species.sort_unstable();
        let unlearned = self.pass_coverage.as_ref().filter(|c| c.species == species && c.fitted == 0).map(|c| {
            format!("this scene is not learned — {}", c.words().unwrap_or_default())
        });
        Some(match refusal {
            None => Readiness { answers: true, rule: None, atom: None, text: "strict mode answers here".into(), unlearned },
            Some(r) => {
                let mut text = match r.text.strip_prefix("surrogate: ") {
                    Some(t) => t.to_string(),
                    None => r.text.replace("surrogate untrusted: ", "untrusted: "),
                };
                if let TeachRule::Class(c) = r.rule {
                    let ci = if c == sim::activity::EnvClass::Pair { 0 } else { 1 };
                    if let Some(note) = self.worst_sample_note(ci) {
                        text = format!("{text} — {note}");
                    }
                }
                Readiness { answers: false, rule: Some(r.rule), atom: r.atom, text, unlearned }
            }
        })
    }

    /// The class's worst leave-one-out sample in words: "worst leave-one-out sample #N (a
    /// draft-mode audit, O1H2, pairs 0-1 1.81 …): 3.2× its bound".
    fn worst_sample_note(&self, class: usize) -> Option<String> {
        let (i, ratio) = (*self.class_worst.get(class)?)?;
        let s = self.store.as_ref()?.samples.get(i)?;
        let pairs = pair_distances(&s.pos);
        let shown: Vec<String> = pairs.iter().take(6).map(|(a, b, r)| format!("{a}-{b} {r:.2}")).collect();
        Some(format!(
            "worst leave-one-out sample #{i} ({}, {}, pairs {}{}): {ratio:.1}× its bound",
            origin_words(s.tag),
            composition(&s.z),
            shown.join(", "),
            if pairs.len() > 6 { ", …" } else { "" }
        ))
    }

    /// The rule-5d novelty ν of a geometry (> 1 refuses) and the atom whose force rows reach
    /// furthest (`None`: the energy row); `None` without a fit or for a species the basis lacks.
    pub fn novelty(&self, pos: &[[f64; 3]], z: &[u8]) -> Option<(f64, Option<usize>)> {
        self.null_space.as_ref()?.novelty(pos, z).ok()
    }

    /// Whether the extrapolation grade's MaxVol converged at the last fit (`None` without one).
    pub fn active_set_converged(&self) -> Option<bool> {
        self.active_set.as_ref().map(|a| a.converged())
    }

    /// The fit's unspanned directions (rule 5d): their number and the threshold τ.
    pub fn null_space_size(&self) -> Option<(usize, f64)> {
        self.null_space.as_ref().map(|n| (n.dimension(), n.tau))
    }

    /// The policy's verdict at `scene` in words: the rule that refuses (or "answers"), the text a
    /// step there would carry, and the numbers behind every rule — gate, prediction, envelope
    /// (per atom), 5d ν, 5e γ, environment class, variance, fence. Draft mode is not applied (this
    /// is the strict policy's verdict, as [`would_answer`](SurrogateKs::would_answer), whose code it
    /// also prints). Read-only.
    pub fn explain(&self, scene: &Scene) -> String {
        let mut o = String::new();
        let prov = self.scene_provenance(scene);
        let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        let z: Vec<u8> = scene.nuclei.iter().map(|n| n.z).collect();
        let d = pair_distances(&pos);
        let dmin = d.iter().map(|x| x.2).fold(f64::INFINITY, f64::min);
        let _ = writeln!(
            o,
            "    geometry: {} ({} atoms), min pair distance {dmin:.3} Bohr, nearest face {:.2} Bohr; pairs {}",
            composition(&z),
            z.len(),
            face_gap(scene),
            d.iter().map(|(i, j, r)| format!("{i}-{j} {r:.3}")).collect::<Vec<_>>().join(", ")
        );
        let _ = writeln!(o, "    would_answer: {:?}", self.would_answer(&pos, &z, &prov));
        if let Err(g) = self.gate(&z, &prov) {
            let _ = writeln!(o, "    REFUSED by the gate ({:?}): {}", g.reason.rule, g.reason.text);
            return o;
        }
        let model = self.model.as_ref().expect("the gate passed");
        let pred = match model.predict(&pos, &z, true) {
            Ok(p) => p,
            Err(e) => {
                let _ = writeln!(o, "    predict failed: {e}");
                return o;
            }
        };
        let n = pos.len().max(1) as f64;
        let e_ref: f64 = z.iter().map(|zi| self.refs.get(zi).copied().unwrap_or(f64::NAN)).sum();
        let fmax = pred.forces.iter().map(|f| (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt()).fold(0.0f64, f64::max);
        let _ = writeln!(
            o,
            "    model: E {:.6} Ha (E − Σ E_ref = {:+.5} Ha, {:+.5}/atom), max |F| {fmax:.4} Ha/Bohr, energy std {:?}",
            pred.energy,
            pred.energy - e_ref,
            (pred.energy - e_ref) / n,
            pred.energy_std
        );
        let mut refusals: Vec<String> = Vec::new();
        if let Some(r) = non_finite_answer(&pred) {
            refusals.push(format!("{:?}: {}", r.rule, r.text));
        }
        let (atoms, env) = self.envelope_outside(model, &pos, &z);
        let _ = writeln!(o, "    envelope per atom (> 1 outside; ∞ isolated): [{}]", atoms.iter().map(|a| format!("{a:.2}")).collect::<Vec<_>>().join(", "));
        if let Some(r) = env {
            refusals.push(format!("{:?}: {}", r.rule, r.text));
        }
        let grade = self.extrapolation_grade(&pos, &z);
        let nu = self.novelty(&pos, &z);
        let _ = writeln!(o, "    5e grade γ_max {:?} (limit {}; MaxVol converged {:?}); 5d novelty ν {:?} (> 1 refuses)", grade, self.policy.max_grade, self.active_set_converged(), nu);
        if let Some((nu, at)) = nu.filter(|(nu, _)| *nu > 1.0) {
            refusals.push(format!("NewEnvironment: ν = {nu:.2e} ({})", at.map_or("energy row".to_string(), |j| format!("atom {j}'s force rows"))));
        }
        let rc2 = model.basis.r_cut * model.basis.r_cut;
        let counts = neighbour_counts(&pos, rc2);
        let class = env_class(&counts);
        let _ = writeln!(o, "    neighbour counts within r_cut {:?} → class {}", counts, class.map_or("isolated", |c| ENV_CLASSES[c]));
        if let Some((r, short)) = self.class_untrusted(&counts) {
            refusals.push(format!("{:?}{}: {}", r.rule, if short { " (short of samples)" } else { "" }, r.text));
        }
        if let (Some(e), Some(f)) = (pred.energy_std, pred.force_std_epistemic.as_ref()) {
            let v = self.variance_check(model, &pred, e, f);
            let _ = writeln!(o, "    variance: per-atom epistemic force std / bound [{}]", v.atoms.iter().map(|a| format!("{a:.2e}")).collect::<Vec<_>>().join(", "));
            if let Some(r) = v.reason.filter(|_| !v.accept) {
                refusals.push(format!("{:?}: {}", r.rule, r.text));
            }
        }
        if !self.near_quarantined_cached(&pos, &z).is_empty() {
            refusals.push("NearQuarantine: within the quarantine radius of a held sample".into());
        }
        if refusals.is_empty() {
            let _ = writeln!(o, "    → the strict policy ANSWERS here");
        } else {
            let _ = writeln!(o, "    → refused; every rule that fires (evaluate shows the first of envelope, 5d, class, variance, fence):");
            for r in refusals {
                let _ = writeln!(o, "      {r}");
            }
        }
        o
    }

    /// The diagnostic report (module doc): the fit, the trust checks, the class leave-one-out with
    /// its worst samples, the store by origin × state and by composition × state with face
    /// densities and training residuals, the references and atomisation energies, the MaxVol and
    /// null-space state, and [`explain`](Self::explain) at each `probes` scene.
    pub fn inspect(&self, probes: &[(&str, &Scene)]) -> String {
        let mut o = String::new();
        let p = &self.policy;
        let _ = writeln!(
            o,
            "surrogate inspection: {} samples, fitted {}, trusted {}, epoch {}, model generation {}, draft {}, audit_every {}",
            self.samples(),
            self.model.is_some(),
            self.trusted(),
            self.epoch,
            self.model_gen,
            p.never_fallback,
            p.audit_every
        );
        let Some(model) = self.model.as_ref() else {
            let _ = writeln!(o, "  no model");
            return o;
        };
        let b = &model.basis;
        let _ = writeln!(
            o,
            "  basis: species {:?}, n_max {}, l_max {}, r_cut {}, body3 {}, degree cap {} (recovery cap {:?}), P {} features, Q′ {}",
            b.species,
            b.n_max,
            b.l_max,
            b.r_cut,
            b.body3,
            b.degree_cap,
            self.degree_cap,
            b.n_features(),
            b.n_independent_columns()
        );
        if let Some(r) = self.report.as_ref() {
            let _ = writeln!(
                o,
                "  fit: n {} (Q′ {}, {} rows), rmse E/atom {:.2e} F {:.2e}, s_E {:.2e} s_F {:.2e}, λ {:.2e}, dof_E {:.1} dof_F {:.1}, κ {:.1e}, converged {}",
                r.n_samples,
                r.q_prime,
                r.n_rows,
                r.rmse_energy_per_atom,
                r.rmse_force,
                r.sigma_e,
                r.sigma_f,
                r.lambda,
                r.residual_dof_e,
                r.residual_dof_f,
                r.condition_estimate,
                r.converged
            );
        }
        let _ = writeln!(o, "  recovery: {:?}", self.recovery);
        let _ = writeln!(o, "  trust checks (the badge: all must be met):");
        for c in self.trust_checks() {
            let _ = writeln!(o, "    {} {:?}: value {:.3e}, limit {:.3e}", if c.met { "met " } else { "FAIL" }, c.key, c.value, c.limit);
        }
        let _ = writeln!(o, "  untrusted reason: {:?}; last audit {:?} (this fit {})", self.untrusted_reason(), self.last_audit, self.last_audit_this_fit());
        if let Some(t) = self.training_residuals {
            let _ = writeln!(o, "  training residuals: E/atom {:.2e} (tier {:.0e}), worst F {:.2e} (bound {:.2e})", t.energy_per_atom, p.energy_std_per_atom, t.force, t.force_bound);
        }
        // the class rule (4b), with its worst samples
        let _ = writeln!(o, "  class leave-one-out (rule 4b):");
        for c in &self.class_loo {
            let _ = writeln!(
                o,
                "    {}: {} samples, LOO force RMS {:.3}× max {:.3}× the bound, energy {:.3}× the tier → {} {:?}",
                c.class,
                c.samples,
                c.force_rms,
                c.force_max,
                c.energy_max,
                if c.trusted { "trusted" } else { "UNTRUSTED" },
                c.refusal
            );
        }
        let store = self.store.as_ref();
        let samples: &[Sample] = store.map_or(&[], |s| &s.samples[..]);
        let rc2 = b.r_cut * b.r_cut;
        let rule = box_affected_face_density(store.map_or(0.25, |s| s.provenance.h));
        // per fitted sample: (index, class, force ratio max, energy ratio)
        let mut loo: Vec<(usize, usize, f64, f64)> = Vec::new();
        for (i, s) in samples[..self.accumulated.min(samples.len())].iter().enumerate() {
            if self.excluded(i) {
                continue;
            }
            // the sample's largest atom class (a readout; the rule classes each atom: `class_loo`)
            let Some(c) = neighbour_counts(&s.pos, rc2).into_iter().filter_map(atom_class).max() else { continue };
            let n = s.pos.len().max(1) as f64;
            match surrogate::loo::leave_one_out(model, s) {
                None => loo.push((i, c, f64::INFINITY, f64::INFINITY)),
                Some(l) => {
                    let e = if model.energy_per_atom { l.energy.abs() } else { l.energy.abs() / n } / p.energy_std_per_atom;
                    let f = l
                        .forces
                        .iter()
                        .zip(&s.forces)
                        .map(|(r, f)| {
                            let (d, bd) = p.force_residual(&[f[0] - r[0], f[1] - r[1], f[2] - r[2]], f);
                            d / bd
                        })
                        .fold(0.0f64, super::trust::max_nan);
                    loo.push((i, c, f, e));
                }
            }
        }
        // the M6 rule for comparison: one class per geometry (its largest atom class), max over the class
        for (c, class) in ENV_CLASSES.iter().enumerate() {
            let rows: Vec<&(usize, usize, f64, f64)> = loo.iter().filter(|x| x.1 == c).collect();
            let (fm, em) = rows.iter().fold((0.0f64, 0.0f64), |(f, e), r| (super::trust::max_nan(f, r.2), super::trust::max_nan(e, r.3)));
            let q = self.report.as_ref().map_or(0, |r| r.q_prime);
            let _ = writeln!(o, "    (the per-geometry rule before 2026-10-18) {class}: {} samples, LOO force max {fm:.3}×, energy {em:.3}× → {}", rows.len(), if rows.len() >= q && fm <= 1.0 && em <= 1.0 { "trusted" } else { "UNTRUSTED" });
        }
        for (c, class) in ENV_CLASSES.iter().enumerate() {
            let mut worst: Vec<&(usize, usize, f64, f64)> = loo.iter().filter(|x| x.1 == c).collect();
            worst.sort_by(|a, b| b.2.max(b.3).total_cmp(&a.2.max(a.3)));
            let over = worst.iter().filter(|x| !(x.2 <= 1.0 && x.3 <= 1.0)).count();
            let _ = writeln!(o, "    {class}: {over} samples beyond 1× (force or energy); the worst:");
            for &&(i, _, f, e) in worst.iter().take(5) {
                let s = &samples[i];
                let d = pair_distances(&s.pos);
                let _ = writeln!(
                    o,
                    "      #{i} {:?} {} face {:.1e}: LOO force {f:.2}×, energy {e:.2}×; pairs {}",
                    s.tag,
                    composition(&s.z),
                    self.face.get(i).copied().unwrap_or(f64::NAN),
                    d.iter().map(|(a, bb, r)| format!("{a}-{bb} {r:.2}")).collect::<Vec<_>>().join(" ")
                );
            }
        }
        // the store: origin × state, composition × state, with face densities and residuals
        let mut by_tag: BTreeMap<String, BTreeMap<&'static str, usize>> = BTreeMap::new();
        #[derive(Default)]
        struct Comp {
            states: BTreeMap<&'static str, usize>,
            boxed: usize,
            faces: Vec<f64>,
            unknown_face: usize,
            e_max: f64,
            f_ratio: f64,
        }
        let mut by_comp: BTreeMap<String, Comp> = BTreeMap::new();
        for (i, s) in samples.iter().enumerate() {
            let st = state_name(self.quarantine.get(i).copied(), i < self.accumulated);
            *by_tag.entry(format!("{:?}", s.tag)).or_default().entry(st).or_default() += 1;
            let c = by_comp.entry(composition(&s.z)).or_default();
            *c.states.entry(st).or_default() += 1;
            let face = self.face.get(i).copied().unwrap_or(f64::NAN);
            if face.is_nan() {
                c.unknown_face += 1;
            } else {
                c.faces.push(face);
            }
            if face > rule {
                c.boxed += 1;
            }
            if let Ok(pr) = model.predict(&s.pos, &s.z, false) {
                let n = s.pos.len().max(1) as f64;
                c.e_max = super::trust::max_nan(c.e_max, (pr.energy - s.energy).abs() / n);
                for (fm, fs) in pr.forces.iter().zip(&s.forces) {
                    let (d, bd) = p.force_residual(fm, fs);
                    c.f_ratio = super::trust::max_nan(c.f_ratio, d / bd);
                }
            }
        }
        let _ = writeln!(o, "  store by origin × state (fitted = in the standing fit; prefix {} of {}):", self.accumulated, samples.len());
        for (tag, m) in &by_tag {
            let _ = writeln!(o, "    {tag}: {}", m.iter().map(|(k, v)| format!("{k} {v}")).collect::<Vec<_>>().join(", "));
        }
        let _ = writeln!(o, "  store by composition × state (box rule {rule:.2e} e/Bohr³; residuals of the standing model over ALL these samples):");
        for (comp, c) in by_comp.iter_mut() {
            c.faces.sort_by(f64::total_cmp);
            let q = |f: f64| c.faces.get(((c.faces.len() as f64 - 1.0) * f).round() as usize).copied().unwrap_or(f64::NAN);
            let _ = writeln!(
                o,
                "    {comp}: {}; box-affected {} (face density median {:.2e}, 90 % {:.2e}, max {:.2e}; unknown {}); max |E_model − E|/N {:.2e} Ha ({:.1}× tier), max force residual {:.2}× bound",
                c.states.iter().map(|(k, v)| format!("{k} {v}")).collect::<Vec<_>>().join(", "),
                c.boxed,
                q(0.5),
                q(0.9),
                q(1.0),
                c.unknown_face,
                c.e_max,
                c.e_max / p.energy_std_per_atom,
                c.f_ratio
            );
        }
        let _ = writeln!(o, "  trust floor (box exclusion waits for it): {:?}; box-excluded {}, box-affected {}, quarantined {}, superseded {}, unfitted {}", self.trust_floor(), self.box_excluded(), self.box_affected(), self.quarantined(), self.superseded(), self.unfitted());
        // references and atomisation energies (an outlier flags a reference mix-up)
        let _ = writeln!(
            o,
            "  references: {} (legacy {:?}); the model's: {:?}",
            self.refs.iter().map(|(z, e)| format!("{} {e:.6}", units::isotopes::symbol(*z))).collect::<Vec<_>>().join(", "),
            self.legacy_refs,
            model.ref_energy
        );
        let mut lowest: BTreeMap<String, (f64, usize)> = BTreeMap::new();
        for (i, s) in samples.iter().enumerate() {
            let e = lowest.entry(composition(&s.z)).or_insert((f64::INFINITY, i));
            if s.energy < e.0 {
                *e = (s.energy, i);
            }
        }
        for (comp, (e, i)) in &lowest {
            let sum: f64 = samples[*i].z.iter().map(|z| self.refs.get(z).copied().unwrap_or(f64::NAN)).sum();
            let _ = writeln!(o, "    {comp}: lowest stored E {e:.6} Ha (#{i}); atomisation Σ E_ref − E = {:.4} Ha ({:.2} eV)", sum - e, (sum - e) * 27.211386);
        }
        // the grade and the null space
        if let Some(a) = self.active_set.as_ref() {
            let per: Vec<String> = a
                .species
                .iter()
                .zip(&b.species)
                .map(|(sa, z)| match sa {
                    Some(sa) => format!("{} active {} converged {}", units::isotopes::symbol(*z), sa.active.len(), sa.converged),
                    None => format!("{} none", units::isotopes::symbol(*z)),
                })
                .collect();
            let _ = writeln!(o, "  MaxVol active sets: {} (overall converged {})", per.join("; "), a.converged());
        }
        if let Some((dim, tau)) = self.null_space_size() {
            let _ = writeln!(o, "  null space (5d): {dim} unspanned directions, τ {tau:.3e}");
        }
        if let Some(env) = self.envelope.as_ref() {
            let _ = writeln!(o, "  envelope atoms per species: {:?}", env.species.iter().zip(&env.count).map(|(z, c)| format!("{} {c}", units::isotopes::symbol(*z))).collect::<Vec<_>>());
        }
        for (what, sc) in probes {
            let _ = writeln!(o, "  probe {what}:");
            o += &self.explain(sc);
        }
        o
    }
}
