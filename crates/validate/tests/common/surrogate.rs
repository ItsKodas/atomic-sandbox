//! The surrogate helpers shared by the surrogate suite (`surrogate.rs`) and the transfer test
//! (`transfer.rs`; M8a Task 6 moved them here from `surrogate.rs` unchanged in behaviour): the
//! student over the locked GPU teacher, one sampling pass + refit, the fit's one-line summaries,
//! the model-vs-teacher comparison at one geometry, and a reader for the shipped `scenes/*.scene`
//! files (the app's parser lives in the `app` crate, which `validate` does not depend on), and
//! the stray-atom helpers of `surrogate.rs` and `dynamics.rs` (plan 2026-11-18 Task 7: the grid,
//! the scenes, the student with a background auditor, the H₂ pass, the composition checks).
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
    assert_eq!((p.failed, p.skipped), (0, 0), "sampling pass: {p:?}; the pass's lines:\n{out}");
    (rep, pass)
}

/// [`sample_and_fit`] without the assertion: the pass's progress (its skipped and failed jobs)
/// is returned. The pass's cutoff is the provider's (`SurrogateKs::cutoff`: its basis's, else the
/// basis a fit would use — a shared collection's reach).
pub fn sample_and_fit_counted(s: &mut SurrogateKs, sc: &Scene, plan: &SamplingPlan, out: &mut String) -> (FitReport, Duration, SamplingProgress) {
    sample_and_fit_with_floor(s, sc, plan, None, out)
}

/// [`sample_and_fit_counted`] with the sampler's time-limit floor (s) set
/// (`Sampler::with_time_limit_floor`; `None` keeps the sampler's own). `f64::INFINITY` turns the
/// limit off: a pass whose teacher takes the suite's `GPU` lock per evaluation, while other tests
/// in the process hold that lock for minutes, would otherwise count a lock wait as a failed solve
/// (the limit is a viewer-responsiveness device, not part of a measurement; a real SCF stall still
/// counts as failed).
pub fn sample_and_fit_with_floor(s: &mut SurrogateKs, sc: &Scene, plan: &SamplingPlan, floor_s: Option<f64>, out: &mut String) -> (FitReport, Duration, SamplingProgress) {
    let t = Instant::now();
    let r_cut = s.cutoff();
    let mut sampler = Sampler::with_cutoff(sc, plan, r_cut).unwrap_or_else(|e| panic!("sampling plan: {e}")).with_body4(s.collection().is_some());
    if let Some(f) = floor_s {
        sampler = sampler.with_time_limit_floor(f);
    }
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
    Ok(Scene { nuclei, cell, charge, multiplicity: None, sigma: None, external_field: [0.0; 3], ionic, train_mask: None })
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

// ---- stray atoms (spec 2026-11-18 §6; plan Task 7): shared by `surrogate.rs` and `dynamics.rs` ----

/// The stray-atom tests' grid: h = σ = 0.20 Bohr, `ionic accepted`, 81³ (a 16 Bohr cube). The plan's 61³ (12 Bohr) cannot
/// hold H₂ + H at 8 Bohr: on the body diagonal the stray atom is 3.44 Bohr from a face (the stray floor refuses below 4)
/// and every solve is box-affected (face density 3.0e-5 > the rule's 1.28e-5; `stray_box_probe`, `references.md`,
/// "Stray atoms"). At 81³ the smallest face distance is 5.44 Bohr and the face densities 0.3–0.8e-6.
pub const STRAY_N: usize = 81;
pub const STRAY_H: f64 = 0.20;
/// The stray atom's distance from the bonded part's centre (Bohr; spec §6).
pub const STRAY_D: f64 = 8.0;
/// The H₂ pass's bond length (Bohr): H₂'s R_e on this surface (pseudo-H, `references.md` "M7 default policy": 1.434).
pub const STRAY_R0: f64 = 1.434;
/// The composition checks' bond lengths, tried in order until three passes at distinct geometries clear H₂ + H.
pub const STRAY_CHECK_R: [f64; 9] = [1.434, 1.15, 1.75, 1.30, 1.60, 1.05, 1.90, 1.22, 1.68];

/// A neutral scene at automatic multiplicity on an `n`³ × `h` grid under `ionic accepted` (σ = h).
pub fn accepted(n: usize, h: f64, nuclei: Vec<Nucleus>) -> Scene {
    Scene { nuclei, cell: Cell { n, h }, charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3], ionic: IonicPolicy::Accepted, train_mask: None }
}

/// A diatomic (`za`, `zb`, bond `r`) plus one atom `zs` at `d` from the diatomic's centre, on the body diagonal of the
/// cube: the stray atom at +a·u, the diatomic's centre at −b·u (u = (1,1,1)/√3, a + b = d), its axis along
/// e = (1,−1,0)/√2 ⊥ u, so the nearest pair distance is √(d² + r²/4). a − b = r·√3/(2√2) puts the stray atom and the
/// diatomic's outer atoms the same distance from the faces (max |coordinate| a/√3 for both): the diagonal is the
/// longest line in a cube, so this is the smallest box that keeps every atom a given distance from every face.
pub fn diatomic_and_atom(n: usize, h: f64, za: u8, zb: u8, r: f64, zs: u8, d: f64) -> Scene {
    let s3 = 3f64.sqrt();
    let u = [1.0 / s3; 3];
    let e = [1.0 / 2f64.sqrt(), -1.0 / 2f64.sqrt(), 0.0];
    let a = 0.5 * (d + r * s3 / (2.0 * 2f64.sqrt()));
    let b = d - a;
    let at = |c: f64, k: f64| -> [f64; 3] { std::array::from_fn(|i| c * u[i] + k * e[i]) };
    accepted(n, h, vec![Nucleus::new(za, at(-b, 0.5 * r)), Nucleus::new(zb, at(-b, -0.5 * r)), Nucleus::new(zs, at(a, 0.0))])
}

/// H₂ (bond `r`) + H at `STRAY_D` on the stray tests' grid (`diatomic_and_atom`).
pub fn h2_and_h(r: f64) -> Scene {
    h2_and_h_in(STRAY_N, r)
}

/// [`h2_and_h`] on an `n`³ × `STRAY_H` grid.
pub fn h2_and_h_in(n: usize, r: f64) -> Scene {
    diatomic_and_atom(n, STRAY_H, 1, 1, r, 1, STRAY_D)
}

/// The scene's atoms `idx` alone (neutral; `m` the multiplicity, `None` automatic), same cell, σ and policy.
pub fn part(sc: &Scene, idx: &[usize], m: Option<u32>) -> Scene {
    let mut p = sc.clone();
    p.nuclei = idx.iter().map(|&i| sc.nuclei[i].clone()).collect();
    p.multiplicity = m;
    p
}

/// The smallest face distance of the scene's atoms (Bohr): the stray floor's measure (`stray` module doc).
pub fn face_margin(sc: &Scene) -> f64 {
    let half = sc.cell.half_extent();
    sc.nuclei.iter().map(|n| half - n.pos.iter().fold(0.0f64, |m, x| m.max(x.abs()))).fold(f64::INFINITY, f64::min)
}

/// A strict-mode `SurrogateKs` over the locked GPU teacher (tol 1e-6) **with a background auditor** (a second locked
/// teacher: the composition checks run there, `stray` module doc, "The checks on the auditor"); no audits
/// (`audit_every` 0: the tests compare against the teacher themselves). The teacher's and the auditor's logs.
pub fn stray_student(ctx: &GpuContext) -> (SurrogateKs, Arc<Mutex<EvalLog>>, Arc<Mutex<EvalLog>>) {
    let (p, log) = Locked::new(ctx.clone(), TOL_H2);
    let (a, alog) = Locked::new(ctx.clone(), TOL_H2);
    let s = SurrogateKs::new(Box::new(p), FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() }).with_teacher_tol(TOL_H2).with_auditor(Box::new(a));
    (s, log, alog)
}

/// One default sampling pass (seed `seed`) on H₂ at `r` centred in the stray tests' grid, through the `Trainable`
/// face (so the provider's reference cache holds H's scan, the stray floor's m_s source), and the refit the pass ends
/// with (one more if the pass left it pending). Asserts a trusted model. The pass's line is appended to `out`.
pub fn train_h2(s: &mut SurrogateKs, n: usize, r: f64, seed: u64, out: &mut String) {
    use sim::Trainable;
    let t = Instant::now();
    let sc = accepted(n, STRAY_H, vec![Nucleus::new(1, [0.5 * r, 0.0, 0.0]), Nucleus::new(1, [-0.5 * r, 0.0, 0.0])]);
    let plan = SamplingPlan { seed, ..SamplingPlan::default() };
    let mut refs = sim::AtomReferences::default();
    s.start_pass(&sc, &plan, &mut refs).unwrap_or_else(|e| panic!("start_pass: {e}"));
    loop {
        match s.pass_step(&mut never()) {
            Ok(Some(_)) => {}
            Ok(None) => break,
            Err(sim::ProviderError::Cancelled) => continue,
            Err(e) => panic!("pass: {e}"),
        }
    }
    let notes = s.take_notes();
    let _ = s.take_teach_events();
    if s.pending_refit() || s.model().is_none() {
        s.refit().unwrap_or_else(|e| panic!("refit: {e}"));
    }
    let b = s.basis().expect("fitted");
    let _ = writeln!(
        out,
        "  H₂ pass ({n}³ × {STRAY_H}, accepted, seed {seed}): {} samples in {:.0} s; basis r_cut {}, species {:?}, Q′ {}; E_ref(H) {:+.6} Ha; trusted {} ({:?}); classes: {}",
        s.samples(),
        t.elapsed().as_secs_f64(),
        b.r_cut,
        b.species,
        b.n_independent_columns(),
        s.reference_energy(1).unwrap_or(f64::NAN),
        s.trusted(),
        s.untrusted_reason(),
        class_summary(s)
    );
    for n in notes.iter().filter(|n| n.contains("finished") || n.contains("refit") || n.contains("fail")) {
        let _ = writeln!(out, "    note: {n}");
    }
    assert!(s.trusted(), "the H₂ model must be trusted for strict-mode answers: {:?}\n{out}", s.untrusted_reason());
}

/// Waits (polling the auditor; at most 30 min) for the composition check in flight to be judged and returns its
/// Activity event (`TeachOrigin::Audit`, `TeachRule::IsolatedAtom`); every other event is appended to `seen`.
pub fn await_check(s: &mut SurrogateKs, seen: &mut Vec<sim::TeachEvent>) -> sim::TeachEvent {
    use sim::Trainable;
    let t = Instant::now();
    loop {
        // `take_atom_densities` polls the auditor's answers (a check's part is filed and, when complete, judged)
        let _ = s.take_atom_densities();
        let mut found = None;
        for e in s.take_teach_events() {
            if found.is_none() && e.origin == sim::activity::TeachOrigin::Audit && e.rule == sim::activity::TeachRule::IsolatedAtom {
                found = Some(e);
            } else {
                seen.push(e);
            }
        }
        if let Some(e) = found {
            return e;
        }
        assert!(t.elapsed() < Duration::from_secs(1800), "no composition check answered in 30 min: {seen:?}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// One composition check through the provider: a fresh step (`discontinuity`), the strict-mode fallback (rule 5a)
/// that queues the check, and the check's verdict. `None` when no check was queued (a geometry that would not count:
/// not distinct from the region's points); else the event (its outcome and detail).
pub fn stray_check(s: &mut SurrogateKs, sc: &Scene, out: &mut String) -> Option<sim::TeachEvent> {
    s.discontinuity();
    let r = s.evaluate(sc, &mut never()).unwrap_or_else(|e| panic!("stray-atom step: {e}"));
    let queued = r.stats.notes.iter().any(|n| n.contains("composition check") && n.contains("queued"));
    let _ = writeln!(out, "    step: source {:?}, E {:+.8}; check queued: {queued}", r.stats.source, r.energy);
    for n in r.stats.notes.iter().filter(|n| n.contains("stray") || n.contains("isolated")) {
        let _ = writeln!(out, "      note: {n}");
    }
    if !queued {
        return None;
    }
    let mut seen = Vec::new();
    let ev = await_check(s, &mut seen);
    let _ = writeln!(out, "    check: {:?} ({} solves, {:.0} ms): {}", ev.outcome, ev.solves, ev.solve_ms, ev.detail);
    Some(ev)
}

/// Clears H₂ + H's region: checks at the bond lengths `rs` in order until the signature is cleared (K = 3 passes at
/// distinct geometries). Returns the bond lengths whose checks passed. Panics on a refusal or when `rs` runs out.
pub fn clear_h2_h(s: &mut SurrogateKs, n: usize, rs: &[f64], out: &mut String) -> Vec<f64> {
    let mut passed = Vec::new();
    for &r in rs {
        let _ = writeln!(out, "  H₂ ({r}) + H at {STRAY_D}:");
        let Some(ev) = stray_check(s, &h2_and_h_in(n, r), out) else { continue };
        match ev.outcome {
            sim::activity::TeachOutcome::StrayChecking { n, of } => {
                passed.push(r);
                if n == of && ev.detail.contains("cleared") {
                    return passed;
                }
            }
            sim::activity::TeachOutcome::StrayRefused => panic!("H₂ + H refused at r = {r}: {}\n{out}", ev.detail),
            _ => {}
        }
    }
    panic!("H₂ + H not cleared over {rs:?} (passes at {passed:?})\n{out}")
}
