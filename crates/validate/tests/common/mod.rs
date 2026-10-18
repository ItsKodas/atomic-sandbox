//! Helpers shared by the GPU test binaries (`gpu_vs_cpu.rs`, `dynamics.rs`, `surrogate.rs`,
//! `tools.rs`; M6 added the scene helpers of `tools.rs` and `Locked`'s opt-in side solvers).
//! Each integration test is its own crate, so this is pulled in with `mod common;` and every
//! item is allowed to be unused by one of them.
//!
//! The M4 dynamics helpers live here since M5 so the surrogate suite reuses them instead of
//! duplicating them (controller ruling): the fixed discrete H₂ problem (σ = h = 0.25, 49³,
//! tol 1e-6), the force-bisection R_e and 3-point curvature (`locate_surface`), the
//! upward-zero-crossing period, the least-squares drift line, the quenched relaxation, the H₃⁺
//! builder, and the `Locked` GPU teacher that serialises the suite's GPU work and logs it.
#![allow(dead_code)]
use gpu::GpuContext;
use providers::GpuKs;
use sim::{Cell, ForceProvider, ForceResult, Integrator, Nucleus, ProviderError, Scene};
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Serialises the GPU sections of concurrently running tests within one binary (poisoning is
/// harmless here). Distinct test binaries never run at the same time under `cargo test`.
pub static GPU: Mutex<()> = Mutex::new(());

/// A headless device, or `None` (with a note naming the suite) when there is no adapter. With a device, the providers'
/// dimer gate (M7 Task 11: pseudo-H and pseudo-He come from `providers::pseudo`'s cache through a 3-D dimer check) runs
/// on it — ≈ 1 min, against ≈ 40 min per element on the CPU fallback — so a suite's first `table_for(1)` never runs on
/// the CPU by accident.
pub fn gpu(suite: &str) -> Option<GpuContext> {
    let ctx = GpuContext::new_headless();
    match &ctx {
        Some(c) => providers::pseudo::set_dimer_backend(providers::pseudo::DimerBackend::Gpu(c.clone())),
        None => eprintln!("{suite}: no GPU adapter — skipped"),
    }
    ctx
}

pub fn ms(d: Duration) -> f64 { d.as_secs_f64() * 1e3 }

/// Today's date from the system clock without a date crate: days since the Unix epoch →
/// civil date (Howard Hinnant's algorithm), UTC.
pub fn chrono_free_date() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let z = (secs / 86_400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

// ---- the discrete problem of the dynamics and surrogate suites ----
pub const H: f64 = 0.25;
pub const N_H2: usize = 49;
pub const TOL_H2: f64 = 1e-6;
pub const DT_FS: f64 = 0.25;

// ---- H₂ vibration (M4) ----
/// Bracket for R_e(σ = 0.25) (the fixed-σ minimum is near 1.79 Bohr; `gpu_vs_cpu` curve shape).
pub const RE_BRACKET: (f64, f64) = (1.6, 2.0);
/// Bisection width at which R_e is accepted (12 halvings of 0.4 Bohr).
pub const RE_TOL: f64 = 1e-4;
/// Initial stretch of the MD and the step of the 3-point curvature (same range, same surface).
pub const DISPLACEMENT: f64 = 0.05;

/// E_h/(hc) in cm⁻¹ (CODATA 2018): an angular frequency in atomic units (E_h/ħ) is the
/// wavenumber ω·E_h/(hc). Display only (rule 3: the suite's physics stays in a.u.).
pub const HARTREE_IN_CM: f64 = 219_474.631_363_2;

// ---- H₃⁺ (M4 thermostat, M5 tier) ----
/// H₃⁺ before relaxation: an equilateral triangle of this side (Bohr; exact R_e = 1.65, the
/// σ = 0.25 minimum is wider, as H₂'s 1.79 vs 1.45).
pub const H3_SIDE0: f64 = 1.9;
/// Relaxation on the test surface: quenched velocity Verlet (velocities zeroed whenever
/// Σ F·v < 0) at `RELAX_DT_FS` until max_i |F_i| < `RELAX_FTOL` Ha/Bohr — the largest per-atom
/// force norm since M6 (`sim::relax::max_norm`, FIRE's F_TOL definition; the largest Cartesian
/// component before, so the criterion only got tighter) (the egg-box floor for H is
/// ≲ 1e-4; the thermostat only needs the start to be a minimum to within thermal forces,
/// k_BT/0.1 Bohr ≈ 1e-2), at most `RELAX_MAX_STEPS` steps.
pub const RELAX_DT_FS: f64 = 0.5;
pub const RELAX_FTOL: f64 = 1e-3;
pub const RELAX_MAX_STEPS: usize = 300;

pub fn never() -> impl FnMut() -> bool { || false }
pub fn no_augment() -> impl FnMut(&Scene, &mut Vec<[f64; 3]>) { |_, _| {} }

pub fn scene(n: usize, nuclei: Vec<Nucleus>) -> Scene {
    Scene { nuclei, cell: Cell { n, h: H }, charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3], ionic: sim::IonicPolicy::None }
}

/// H₂ along x at bond length `r`, nucleus 0 at +r/2, at rest.
pub fn h2(r: f64) -> Scene {
    scene(N_H2, vec![Nucleus::new(1, [r / 2.0, 0.0, 0.0]), Nucleus::new(1, [-r / 2.0, 0.0, 0.0])])
}

pub fn bond(s: &Scene) -> f64 {
    let (a, b) = (&s.nuclei[0].pos, &s.nuclei[1].pos);
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// H₃⁺: an equilateral triangle of side `side` in the xy-plane, centred on the origin, charge +1
/// (2 electrons, closed shell: no occupied near-degeneracy anywhere near its minimum).
pub fn h3_plus(side: f64) -> Scene {
    let rad = side / 3f64.sqrt();
    let nuclei = [90.0f64, 210.0, 330.0].iter().map(|deg| {
        let a = deg.to_radians();
        Nucleus::new(1, [rad * a.cos(), rad * a.sin(), 0.0])
    }).collect();
    let mut s = scene(N_H2, nuclei);
    s.charge = 1;
    s
}

/// H₂ along x at bond length `r` with both nuclei of mass number `a` (M6: ¹H = 1, ²H = 2).
pub fn h2_isotope(r: f64, a: u16) -> Scene {
    let n = |x: f64| Nucleus::isotope(1, a, [x, 0.0, 0.0]).unwrap_or_else(|e| panic!("{e}"));
    scene(N_H2, vec![n(r / 2.0), n(-r / 2.0)])
}

/// Two He along x at separation `r`, centred on the origin, on a `n`³ grid.
pub fn he2(n: usize, r: f64) -> Scene {
    scene(n, vec![Nucleus::new(2, [r / 2.0, 0.0, 0.0]), Nucleus::new(2, [-r / 2.0, 0.0, 0.0])])
}

/// HeH⁺ along x at separation `r` (He at −r/2, H at +r/2), charge +1, on a `n`³ grid.
pub fn heh_plus(n: usize, r: f64) -> Scene {
    let mut s = scene(n, vec![Nucleus::new(2, [-r / 2.0, 0.0, 0.0]), Nucleus::new(1, [r / 2.0, 0.0, 0.0])]);
    s.charge = 1;
    s
}

/// LiH along x at separation `r`, centred on the origin (Li at −r/2, H at +r/2), on a `n`³ grid.
pub fn lih(n: usize, r: f64) -> Scene {
    scene(n, vec![Nucleus::new(3, [-r / 2.0, 0.0, 0.0]), Nucleus::new(1, [r / 2.0, 0.0, 0.0])])
}

/// Pair distances of a 3-nucleus scene, "d01/d02/d12".
pub fn triangle(s: &Scene) -> String {
    let d = |a: usize, b: usize| {
        let (p, q) = (&s.nuclei[a].pos, &s.nuclei[b].pos);
        ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
    };
    format!("{:.4}/{:.4}/{:.4}", d(0, 1), d(0, 2), d(1, 2))
}

/// What the wrapped provider did (shared with the test through an `Arc` because the driver
/// takes the provider by `Box`).
#[derive(Default, Clone)]
pub struct EvalLog {
    pub calls: usize,
    pub scf_steps: usize,
    pub readbacks: usize,
    pub gpu: Duration,
    pub max_solve: Duration,
    pub warnings: BTreeSet<String>,
    /// Successful evaluations that carried at least one warning (phases are told apart by
    /// snapshotting this counter).
    pub warned: usize,
    pub extrapolated: usize,
    /// Successful evaluations whose warm solve stalled and was retried cold (R15): call index
    /// (1-based; call 1 is the driver's initial evaluation, call k + 1 is step k) and the note.
    pub cold_retries: Vec<(usize, String)>,
}

impl EvalLog {
    pub fn summary(&self) -> String {
        let n = self.calls.max(1) as f64;
        format!("{} solves, GPU {:.1} s ({:.0} ms/solve, max {:.0} ms), {:.1} SCF steps/solve, {:.0} readbacks/solve, {} extrapolated, {} cold retries (R15) at solves {:?}, {} solves warned: {:?}",
                self.calls, self.gpu.as_secs_f64(), ms(self.gpu) / n, ms(self.max_solve), self.scf_steps as f64 / n, self.readbacks as f64 / n, self.extrapolated,
                self.cold_retries.len(), self.cold_retries, self.warned, self.warnings)
    }
}

/// `GpuKs` behind the suite's GPU lock, logging every evaluation. `name()` passes the inner
/// name through, so a `SurrogateKs` built over it files its samples under the real teacher.
pub struct Locked {
    pub inner: GpuKs,
    pub log: Arc<Mutex<EvalLog>>,
    ctx: GpuContext,
    tol: f64,
    /// `Some` once [`with_side_solver`](Self::with_side_solver) was called: the log of every
    /// side solver handed out (`ForceProvider::side_solver`, the driver's isolated-atom
    /// references, M6 spec §3.2). `None` (the default): no side solver, as before M6 — the
    /// dynamics and surrogate suites' drivers detect no bonds and solve no references.
    side_log: Option<Arc<Mutex<EvalLog>>>,
}

impl Locked {
    pub fn new(ctx: GpuContext, tol: f64) -> (Locked, Arc<Mutex<EvalLog>>) {
        // the dimer gate of pseudo-H/He on this device (M7 Task 11; `gpu` does the same for suites that never build one)
        providers::pseudo::set_dimer_backend(providers::pseudo::DimerBackend::Gpu(ctx.clone()));
        let mut inner = GpuKs::new(ctx.clone(), tol);
        inner.set_display(false);
        let log = Arc::new(Mutex::new(EvalLog::default()));
        (Locked { inner, log: log.clone(), ctx, tol, side_log: None }, log)
    }

    /// Hands out side solvers (fresh locked `GpuKs` instances at the same tolerance, logging
    /// into the returned log, not this one's), so a `Dynamics` over this provider solves its bond
    /// references off the stepping path while the stepping log counts only the stepping solves.
    pub fn with_side_solver(mut self) -> (Locked, Arc<Mutex<EvalLog>>) {
        let side = Arc::new(Mutex::new(EvalLog::default()));
        self.side_log = Some(side.clone());
        (self, side)
    }
}

impl ForceProvider for Locked {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let t = Instant::now();
        let r = self.inner.evaluate(scene, should_abort);
        let dt = t.elapsed();
        let mut log = self.log.lock().unwrap();
        log.gpu += dt;
        log.max_solve = log.max_solve.max(dt);
        if let Ok(r) = &r {
            log.calls += 1;
            log.scf_steps += r.stats.scf_steps;
            log.readbacks += r.stats.readbacks.unwrap_or(0);
            log.warnings.extend(r.stats.warnings.iter().cloned());
            log.warned += (!r.stats.warnings.is_empty()) as usize;
            log.extrapolated += r.stats.notes.iter().any(|n| n.contains("extrapolated")) as usize;
            if let Some(n) = r.stats.notes.iter().find(|n| n.contains("retried cold")) {
                let call = log.calls;
                log.cold_retries.push((call, n.clone()));
            }
        }
        r
    }
    fn name(&self) -> &'static str { self.inner.name() }
    fn reset(&mut self) { self.inner.reset() }
    fn discontinuity(&mut self) { self.inner.discontinuity() }
    fn sample_density(&mut self, points: &[[f64; 3]]) -> Option<Vec<f64>> {
        let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
        self.inner.sample_density(points)
    }
    fn scf_tol(&self) -> Option<f64> { self.inner.scf_tol() }
    fn side_solver(&self) -> Option<Box<dyn ForceProvider>> {
        let log = self.side_log.as_ref()?.clone();
        let mut inner = GpuKs::new(self.ctx.clone(), self.tol);
        inner.set_display(false);
        Some(Box::new(Locked { inner, log, ctx: self.ctx.clone(), tol: self.tol, side_log: None }))
    }
}

pub fn solve(p: &mut dyn ForceProvider, s: &Scene, what: &str) -> ForceResult {
    p.evaluate(s, &mut never()).unwrap_or_else(|e| panic!("{what}: {e}"))
}

/// The static side of the H₂ surface at this σ: R_e by force bisection, k by the 3-point
/// second difference of E (cold solves), the mode energy of a start at R_e + Δ from rest, and
/// the forces of that start (the MD's `forces_old`).
pub struct Surface {
    pub r_e: f64,
    pub k: f64,
    /// −[F(R_e+Δ) − F(R_e−Δ)]/(2Δ): the same curvature from the Hellmann-Feynman forces (reported).
    pub k_from_forces: f64,
    pub e: [f64; 3],
    pub e_mode: f64,
    pub omega: f64,
    pub mu: f64,
    pub start_forces: Vec<[f64; 3]>,
    pub bisections: usize,
    pub log: String,
}

/// `locate_surface` on any provider of the H₂ surface (the DFT teacher, or a surrogate whose
/// own minimum and curvature are wanted). `cold` resets the provider before each of the three
/// curvature solves (bit-reproducible on the GPU; a no-op for a surrogate).
pub fn locate_surface(p: &mut dyn ForceProvider) -> Surface {
    let mut log = String::new();
    let (mut lo, mut hi) = RE_BRACKET;
    let f_at = |p: &mut dyn ForceProvider, r: f64| solve(p, &h2(r), &format!("H₂ R = {r}")).forces[0][0];
    let (mut f_lo, mut f_hi) = (f_at(p, lo), f_at(p, hi));
    assert!(f_lo > 0.0 && f_hi < 0.0, "R_e not bracketed by {RE_BRACKET:?}: F_x(nucleus 0) = {f_lo:+.3e}, {f_hi:+.3e}");
    let mut bisections = 0;
    while hi - lo > RE_TOL {
        let mid = 0.5 * (lo + hi);
        let f = f_at(p, mid);
        bisections += 1;
        let _ = writeln!(log, "  bisection {bisections}: R = {mid:.5}, F_x(0) = {f:+.3e}");
        if f > 0.0 { lo = mid; f_lo = f; } else { hi = mid; f_hi = f; }
    }
    // the root by linear interpolation inside the last bracket
    let r_e = lo + (hi - lo) * f_lo / (f_lo - f_hi);
    // three cold solves (bit-reproducible, path scatter 8e-8 between cold solves at different R)
    let rs = [r_e - DISPLACEMENT, r_e, r_e + DISPLACEMENT];
    let mut e = [0.0; 3];
    let mut f = [0.0; 3];
    let mut start_forces = Vec::new();
    for (i, &r) in rs.iter().enumerate() {
        p.reset();
        let res = solve(p, &h2(r), &format!("H₂ cold R = {r}"));
        e[i] = res.energy;
        f[i] = res.forces[0][0];
        let _ = writeln!(log, "  cold R = {r:.5}: E = {:+.9} Ha, F_x(0) = {:+.3e} Ha/Bohr, {} SCF steps", res.energy, f[i], res.stats.scf_steps);
        if i == 2 { start_forces = res.forces; }
    }
    let k = (e[2] - 2.0 * e[1] + e[0]) / (DISPLACEMENT * DISPLACEMENT);
    let k_from_forces = -(f[2] - f[0]) / (2.0 * DISPLACEMENT);
    let s = h2(r_e);
    let (m0, m1) = (s.nuclei[0].mass, s.nuclei[1].mass);
    let mu = m0 * m1 / (m0 + m1);
    let omega = (k / mu).sqrt();
    let e_mode = e[2] - e[1];
    let _ = writeln!(log, "  R_e(σ = {H}) = {r_e:.5} Bohr ({bisections} bisections, bracket width {:.1e}); k = {k:.5} Ha/Bohr² (from forces {k_from_forces:.5}, ratio {:.4}); μ = {mu:.2} mₑ; ω_static = {omega:.6e} a.u. = {:.1} cm⁻¹; E_mode = {e_mode:.4e} Ha",
                     hi - lo, k_from_forces / k, omega * HARTREE_IN_CM);
    assert!(k > 0.0 && e_mode > 0.0, "not a minimum: k = {k}, E_mode = {e_mode}");
    Surface { r_e, k, k_from_forces, e, e_mode, omega, mu, start_forces, bisections, log }
}

/// Period from the upward zero crossings of `x` sampled every `dt` (linear interpolation);
/// upward-only so an offset of the oscillation centre cancels exactly. Returns (period, count).
pub fn period_from_upward_crossings(x: &[f64], dt: f64) -> (f64, usize) {
    let mut ts = Vec::new();
    for i in 1..x.len() {
        if x[i - 1] < 0.0 && x[i] >= 0.0 {
            ts.push((i - 1) as f64 * dt + dt * (-x[i - 1]) / (x[i] - x[i - 1]));
        }
    }
    assert!(ts.len() >= 3, "only {} upward crossings", ts.len());
    ((ts[ts.len() - 1] - ts[0]) / (ts.len() - 1) as f64, ts.len())
}

/// Least-squares slope and the residual's peak to peak (centred sums), and the mean.
pub fn fit_line(t: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let n = t.len() as f64;
    let (tm, ym) = (t.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let stt: f64 = t.iter().map(|x| (x - tm) * (x - tm)).sum();
    let sty: f64 = t.iter().zip(y).map(|(x, v)| (x - tm) * (v - ym)).sum();
    let slope = sty / stt;
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for (x, v) in t.iter().zip(y) {
        let r = v - ym - slope * (x - tm);
        lo = lo.min(r);
        hi = hi.max(r);
    }
    (slope, hi - lo, ym)
}

pub fn au_per_ps() -> f64 { units::fs_to_au_time(1000.0) }

/// Quenched velocity Verlet to the nearest minimum of the provider's surface (constants above).
/// Returns the relaxed scene at rest, its forces, and the steps taken.
pub fn relax(p: &mut dyn ForceProvider, s: Scene, log: &mut String) -> (Scene, Vec<[f64; 3]>, usize) {
    relax_with(p, s, RELAX_DT_FS, log)
}

/// [`relax`] at another time step (fs) — the tools suite compares FIRE with the quench at FIRE's
/// own starting Δt as well.
pub fn relax_with(p: &mut dyn ForceProvider, mut s: Scene, dt_fs: f64, log: &mut String) -> (Scene, Vec<[f64; 3]>, usize) {
    let integ = Integrator::new_fs(dt_fs);
    // the per-atom vector norm, the same F_TOL definition as `sim::Fire` (M6 re-review SHOULD:
    // the energy comparison of the tools suite rests on |F|₂² ≤ N·F_TOL²); before M6 this was the
    // largest Cartesian component, so every relaxation here only got tighter
    let maxf = |f: &[[f64; 3]]| sim::relax::max_norm(f);
    let r = solve(p, &s, "relax start");
    let mut f = r.forces;
    let _ = writeln!(log, "  relax start: E = {:+.6} Ha, max|F| = {:.2e}", r.energy, maxf(&f));
    let mut steps = 0;
    let mut quenches = 0;
    while maxf(&f) >= RELAX_FTOL && steps < RELAX_MAX_STEPS {
        let (f_new, st) = integ.step(&mut s, &f, p, &mut no_augment(), &mut never()).unwrap_or_else(|e| panic!("relax step {steps}: {e}"));
        f = f_new;
        steps += 1;
        let power: f64 = s.nuclei.iter().zip(&f).map(|(n, f)| n.vel[0] * f[0] + n.vel[1] * f[1] + n.vel[2] * f[2]).sum();
        if power < 0.0 {
            for n in &mut s.nuclei { n.vel = [0.0; 3]; }
            quenches += 1;
        }
        if steps % 25 == 0 {
            let _ = writeln!(log, "  relax {steps}: E_KS = {:+.6}, max|F| = {:.2e}, {quenches} quenches", st.energy_ks, maxf(&f));
        }
    }
    for n in &mut s.nuclei { n.vel = [0.0; 3]; }
    let r = solve(p, &s, "relaxed");
    let _ = writeln!(log, "  relaxed after {steps} steps ({quenches} quenches): E = {:+.6} Ha, max|F| = {:.2e} (target < {RELAX_FTOL:.0e})", r.energy, maxf(&r.forces));
    assert!(maxf(&r.forces) < RELAX_FTOL, "relaxation did not converge in {RELAX_MAX_STEPS} steps: max|F| = {:.2e}", maxf(&r.forces));
    (s, r.forces, steps)
}

/// One row of a suite's `last-run-*.md` timing table.
pub fn timing_row(test: &str, system: &str, n: usize, tol: f64, log: &EvalLog, wall: Duration, result: &str) -> String {
    let c = log.calls.max(1) as f64;
    format!("| {test} | {system} | {n}³ | {tol:.0e} | {} | {:.1} | {:.0} | {:.1} | {:.0} | {:.1} | {result} |",
            log.calls, log.gpu.as_secs_f64(), ms(log.gpu) / c, log.scf_steps as f64 / c, log.readbacks as f64 / c, wall.as_secs_f64())
}
