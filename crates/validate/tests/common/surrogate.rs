//! The surrogate helpers shared by the surrogate suite (`surrogate.rs`) and the transfer test
//! (`transfer.rs`; M8a Task 6 moved them here from `surrogate.rs` unchanged in behaviour): the
//! student over the locked GPU teacher, one sampling pass + refit, the fit's one-line summaries,
//! the model-vs-teacher comparison at one geometry, and a reader for the shipped `scenes/*.scene`
//! files (the app's parser lives in the `app` crate, which `validate` does not depend on).
//! Atomic units throughout.
use super::{ms, never, solve, EvalLog, Locked, TOL_H2};
use ::surrogate::{FitReport, Prediction};
use gpu::GpuContext;
use providers::{FallbackPolicy, Sampler, SurrogateKs, TrainingResiduals};
use sim::{Cell, ForceProvider, ForceResult, IonicPolicy, Nucleus, SamplingPlan, SamplingProgress, Scene};
use std::fmt::Write as _;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub fn policy(audit_every: usize) -> FallbackPolicy {
    FallbackPolicy { audit_every, ..FallbackPolicy::default() }
}

/// A `SurrogateKs` over the suite's locked, logged GPU teacher at tol 1e-6.
pub fn student(ctx: &GpuContext, audit_every: usize) -> (SurrogateKs, Arc<Mutex<EvalLog>>) {
    let (p, log) = Locked::new(ctx.clone(), TOL_H2);
    (SurrogateKs::new(Box::new(p), policy(audit_every)).with_teacher_tol(TOL_H2), log)
}

pub fn residuals(t: Option<TrainingResiduals>) -> String {
    t.map_or("—".into(), |t| format!("E/atom {:.2e}, F {:.2e} (bound {:.2e})", t.energy_per_atom, t.force, t.force_bound))
}

/// The per-class leave-one-out numbers (round 6) and the calibrated verdict's inputs (M8a Task 2):
/// "pair n 273, LOO F rms 0.05× max 0.40×, E rms 0.04× max 0.10×, κ 1.41 (3 evidence), trusted".
pub fn class_summary(s: &SurrogateKs) -> String {
    s.class_loo()
        .iter()
        .map(|c| {
            format!(
                "{} n {}, LOO F rms {:.3}× max {:.3}×, E rms {:.3}× max {:.3}×, κ {:.2} ({} evidence), {}",
                c.class, c.samples, c.force_rms, c.force_max, c.energy_rms, c.energy_max, c.calibration, c.evidence,
                if c.trusted { "trusted" } else { "untrusted" }
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

pub fn fit_summary(rep: &FitReport, s: &SurrogateKs) -> String {
    format!("n {} (Q′ {}, {} rows), rmse E/atom {:.2e} F {:.2e}, s_E {:.2e}/atom s_F {:.2e}, λ {:.2e}, γ {:.1} (E {:.1}, F {:.1}), dof_E {:.0} dof_F {:.0}, κ {:.1e}, {} hyper rounds{}; training residuals {}; trusted: {}{}; classes: {}",
            rep.n_samples, rep.q_prime, rep.n_rows, rep.rmse_energy_per_atom, rep.rmse_force, rep.sigma_e, rep.sigma_f, rep.lambda, rep.dof_eff, rep.gamma_e, rep.gamma_f,
            rep.residual_dof_e, rep.residual_dof_f, rep.condition_estimate, rep.hyper_iterations, if rep.converged { "" } else { " (not converged)" },
            residuals(s.training_residuals()), s.trusted(), s.untrusted_reason().map_or(String::new(), |r| format!(" ({r})")), class_summary(s))
}

/// One sampling pass with the surrogate's own teacher (`Sampler::step_own_teacher`, what
/// `Dynamics` drives), then one refit. Appends the pass summary and the sampler's notes. Asserts
/// that no job failed or was skipped (the surrogate suite's H₂ and H₃⁺ passes: no scan reaches
/// the cutoff shell); [`sample_and_fit_counted`] returns the counts instead.
pub fn sample_and_fit(s: &mut SurrogateKs, sc: &Scene, plan: &SamplingPlan, out: &mut String) -> (FitReport, Duration) {
    let (rep, pass, p) = sample_and_fit_counted(s, sc, plan, out);
    assert_eq!((p.failed, p.skipped), (0, 0), "sampling pass: {p:?}");
    (rep, pass)
}

/// [`sample_and_fit`] without the assertion: the pass's progress (its skipped and failed jobs)
/// is returned. The pass's cutoff is the provider's (`SurrogateKs::cutoff`: its basis's, else the
/// basis a fit would use — a shared collection's reach).
pub fn sample_and_fit_counted(s: &mut SurrogateKs, sc: &Scene, plan: &SamplingPlan, out: &mut String) -> (FitReport, Duration, SamplingProgress) {
    let t = Instant::now();
    let r_cut = s.cutoff();
    let mut sampler = Sampler::with_cutoff(sc, plan, r_cut).unwrap_or_else(|e| panic!("sampling plan: {e}")).with_body4(s.collection().is_some());
    while sampler.step_own_teacher(s, &mut never()).unwrap_or_else(|e| panic!("sampling: {e}")).is_some() {}
    let p = sampler.progress();
    let notes = sampler.take_notes();
    s.discontinuity();
    let pass = t.elapsed();
    let t = Instant::now();
    let rep = s.refit().unwrap_or_else(|e| panic!("refit: {e}"));
    let fit_ms = ms(t.elapsed());
    let _ = writeln!(out, "  sampling pass ({} perturbations σ {}, {} scan points/pair over {:?} × r₀, {}-step burst at {} K, seed {}): {} samples of {} jobs ({} skipped, {} failed) in {:.1} s; E_ref(H) = {:+.6} Ha",
                     plan.perturbations, plan.perturbation_sigma, plan.scan_points_per_pair, plan.scan_range, plan.hot_burst_steps, plan.hot_burst_t, plan.seed,
                     sampler.samples_added(), p.total, p.skipped, p.failed, pass.as_secs_f64(), s.reference_energy(1).unwrap_or(f64::NAN));
    for n in &notes { let _ = writeln!(out, "    note: {n}"); }
    let b = s.basis().expect("fitted");
    let _ = writeln!(out, "  basis: n_max {}, l_max {}, r_cut {}, degree cap {}, body3 {}, body4 {}, species {:?}, P {} features, Q′ {}; refit {fit_ms:.0} ms: {}",
                     b.n_max, b.l_max, b.r_cut, b.degree_cap, b.body3, b.body4, b.species, b.n_features(), b.n_independent_columns(), fit_summary(&rep, s));
    (rep, pass, p)
}

/// The model's prediction (with its stds) and the teacher's answer at one geometry, the teacher
/// called directly (a held-out comparison: nothing is kept as a sample).
pub fn model_vs_dft(s: &mut SurrogateKs, sc: &Scene, what: &str) -> (Prediction, ForceResult) {
    let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
    let z: Vec<u8> = sc.nuclei.iter().map(|n| n.z).collect();
    let pred = s.model().expect("a fitted model").predict(&pos, &z, true).unwrap_or_else(|e| panic!("{what}: predict: {e}"));
    let dft = solve(s.teacher_mut(), sc, what);
    (pred, dft)
}

/// The repository's `scenes/` folder.
pub fn scenes_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scenes")
}

/// A shipped scene file (`atomic sandbox scene v1`: `grid n h`, `charge q`, `ionic <word>`,
/// `nucleus Z x y z` in Bohr; `#` comments) as a `sim::Scene` — the subset of the app's format
/// (`app::scene::Scene::parse`) the shipped files use; anything else is an error, never skipped.
pub fn read_scene(name: &str) -> Scene {
    let path = scenes_dir().join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    parse_scene(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// [`read_scene`]'s parser.
pub fn parse_scene(text: &str) -> Result<Scene, String> {
    let mut cell: Option<Cell> = None;
    let mut charge = 0i32;
    let mut ionic = IonicPolicy::None;
    let mut nuclei = Vec::new();
    for (k, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let w: Vec<&str> = line.split_whitespace().collect();
        let num = |i: usize| w.get(i).and_then(|x| x.parse::<f64>().ok()).ok_or_else(|| format!("line {}: '{line}'", k + 1));
        match w[0] {
            "grid" => cell = Some(Cell { n: num(1)? as usize, h: num(2)? }),
            "charge" => charge = num(1)? as i32,
            "ionic" => ionic = w.get(1).and_then(|x| IonicPolicy::parse(x)).ok_or_else(|| format!("line {}: '{line}'", k + 1))?,
            "nucleus" => nuclei.push(Nucleus::new(num(1)? as u8, [num(2)?, num(3)?, num(4)?])),
            _ => return Err(format!("line {}: unknown keyword in '{line}'", k + 1)),
        }
    }
    let cell = cell.ok_or("no grid line")?;
    Ok(Scene { nuclei, cell, charge, multiplicity: None, sigma: None, external_field: [0.0; 3], ionic })
}

/// A scene file's text for `sc` (the shipped format; positions to 4 decimals, as the shipped
/// files), after `comment` lines (each prefixed with `# `).
pub fn scene_text(sc: &Scene, comment: &[String]) -> String {
    let mut t = String::from("# atomic sandbox scene v1\n");
    for c in comment {
        let _ = writeln!(t, "# {c}");
    }
    let _ = writeln!(t, "grid {} {}", sc.cell.n, sc.cell.h);
    let _ = writeln!(t, "charge {}", sc.charge);
    let _ = writeln!(t, "ionic {}", sc.ionic.word());
    for n in &sc.nuclei {
        let _ = writeln!(t, "nucleus {} {:.4} {:.4} {:.4}", n.z, n.pos[0], n.pos[1], n.pos[2]);
    }
    t
}
