//! Born-Oppenheimer dynamics on the GPU solver (M4 Task 7; numbers of record in `references.md`,
//! section "Dynamics (M4)"). Run with `cargo test -p validate --release --test dynamics`
//! (release-only; skips cleanly without a GPU adapter; `--nocapture` prints every number; each
//! test appends its timing row to the gitignored `last-run-dynamics.md`). Wall times are
//! reported, never asserted (ruling R3: the viewer may share the GPU).
//!
//! Everything here runs on one fixed discrete problem — σ = h = 0.25 Bohr, 49³, SCF tol 1e-6 for
//! H₂ (1e-5, the viewer's, for the thermostat) — and compares the *dynamics* against *statics of the same
//! surface*, so nothing depends on how far that surface is from the σ → 0 LSDA limit or from
//! experiment; those gaps are reported (CLAUDE.md rule 7), not asserted. H₂ runs at 49³ (12 Bohr
//! box), not the brief's 33³: at 33³ both solvers warn "density on box faces … box too small"
//! for H₂ (Task 2 report, face density ~4e-5 > 1e-5), and a vibrating molecule's energy would
//! then carry a box-truncation component that changes with R; at 49³ there is no warning (the
//! tests assert that) and the warm step costs about the same (latency-bound, `references.md`).
//!
//! # Tests and their bounds
//! - `h2_vibration_frequency_matches_curvature_of_the_same_surface`: R_e(σ) by force bisection
//!   (F_x on nucleus 0 = −dE/dR for the symmetric molecule; ≈ 1.79 Bohr at σ = 0.25, not the
//!   σ → 0 1.45), then k = [E(R_e+Δ) − 2E(R_e) + E(R_e−Δ)]/Δ² from three cold solves (Δ = 0.05),
//!   ω_static = √(k/μ), against ω_MD from the upward zero crossings of R(t) − R_e over 400
//!   velocity-Verlet steps at dt = 0.25 fs started at R_e + 0.05 from rest. Bound
//!   `OMEGA_BOUND`. The measured decomposition of ω_MD/ω_static − 1 (run of record −5.6e-4):
//!   Verlet's own frequency shift +h²/24 = +9.7e-4 (h = 0.1525, exact); the 3-point truncation
//!   E⁗Δ²/12 of k, measured as k_F/k_E = 1.0016 (the force central difference carries E⁗Δ²/6),
//!   so ω_static is high by +0.8e-3 and the ratio carries −0.8e-3; the anharmonic softening at
//!   this amplitude is the remainder, −0.7e-3 (ω_MD/ω̃ − 1 = −1.5e-3 is truncation + anharmonic
//!   together; a Morse well of this k and D ≈ 0.09 would give −1.3e-3); the cold-solve path
//!   scatter in k, 4·8e-8/Δ² = 1.3e-4 on k = 0.2 → 3e-4 on ω (6 % of the bound; Δ = 0.05 is
//!   near the optimum for that scatter). Aligned sum of the magnitudes: 2.5e-3. The ratio is also
//!   reported against Verlet's ω̃ = acos(1 − h²/2)/dt, which removes the first term.
//! - `nve_energy_drift_is_bounded`: the same start through the `Dynamics` driver (so its
//!   bookkeeping — `energy_conserved`, the history, `drift_per_ps` — is exercised end to end)
//!   for 2000 steps = 0.5 ps. |drift| < `DRIFT_BOUND` = 2e-4 Ha/ps (brief; the derivation of the
//!   budget is in `references.md`). The energy fluctuation bound is **derived, not pinned**
//!   (controller ruling after R6): Verlet's true energy swings by exactly h²/4 of the mode
//!   energy peak to peak (`sim::integrator` module doc: E ∈ [Ẽ, Ẽ/(1 − h²/4)]), with
//!   E_mode = E(R_e+Δ) − E(R_e) from the same cold solves and h = ω_static·dt, plus the f32 path
//!   scatter of the GPU energy (`H2_PATH_SCATTER` = 6e-7 Ha per solve, Task 7 of M3; white on the
//!   readout, ±1 scatter peak to peak) — bound = `FLUCT_MARGIN` × (h²/4·E_mode + 2·scatter).
//!   The brief's fixed 1e-5 Ha is reported against. The fluctuation is measured as the peak-to-
//!   peak residual about the least-squares line (the drift has its own bound); the raw peak to
//!   peak is reported too.
//! - `thermostat_reaches_target_temperature`: a 6-DOF molecule (N_f = 3N − 3 = 6) on the
//!   realistic path (R7): the driver's `SetThermostat` on a molecule at rest draws
//!   Maxwell–Boltzmann velocities at the target, then CSVR holds it; asserted: the mean T over
//!   the steps after `THERMO_SETTLE_STEPS` within `T_BOUND` = 20 % of 300 K. **The system is
//!   H₃⁺, not the brief's H₂O**, relaxed on the test surface first (quenched Verlet, `relax`):
//!   the `h2o` preset geometry is 0.16 Ha above the minimum of the σ = h = 0.25 surface (57³),
//!   which is near-linear, so the run measures how fast the released 0.16 Ha drains (⟨T⟩ 544 K
//!   over the brief's window), and warm-started GPU solves stall along the resulting motion
//!   (R15's cold retry now carries it through, at 10 min for 600 steps); it is kept as the
//!   `#[ignore]`d `thermostat_h2o_preset_known_limitation` with the numbers in its doc. H₃⁺ (2 electrons, closed shell, one occupied level) has no
//!   near-degeneracy anywhere near its minimum, the same N_f = 6 (3 vibrations + 3 rotations)
//!   and costs a k = 1 solve per step (~80 ms at 49³), which buys the window the statistics
//!   need. **Expected fluctuation** (the reason only the mean is testable, and the reason for the
//!   τ and window chosen): instantaneous σ_T/T = √(2/N_f) = 58 % (measured 61 %). The window
//!   mean is governed by the slow part of K — the vibrational mode energies E_i (each
//!   exponential, Var = (k_BT)²) and the rotational kinetic energies (χ²₁, Var = (k_BT)²/2),
//!   S = ΣE_i/2 + ΣK_rot with S̄ = 3k_BT and Var S = 2.25(k_BT)², i.e. σ_S/S̄ = 50 % — which CSVR
//!   relaxes with time constant τ (dK = (K̄ − K)dt/τ + noise), so over a window T_w ≫ τ
//!   σ_mean/T ≈ 0.50·√(2τ/T_w) (exact exponential-autocorrelation form in the code). A toy
//!   model of exactly this ensemble (3 harmonic modes + 3 free kinetic DOF, Verlet dt, CSVR
//!   each step, 300 seeds; Task 7 report, fix round 1) gives 6.7 % at τ = 5 fs over 675 fs
//!   (9.4 % at τ = 10 fs over the same window, 3.0 % at 1 fs) with ⟨T⟩ = 1.006 ± 0.004 of the
//!   target, so the formula is right for uncoupled modes. At the brief's τ = 10 fs over the
//!   100 fs window its 600 steps leave, σ_mean/T is **21 %: the 20 % bound would be a 1σ
//!   statement** (a third of fresh trajectories would fail it); τ = 10 fs needs a 1.1 ps
//!   window for 3σ. With `THERMO_TAU_FS` = 5 fs over `THERMO_STEPS` − `THERMO_SETTLE_STEPS`
//!   = 2700 steps = 675 fs the analytic σ_mean/T is 6.1 %, i.e. the bound is 3.3σ *for
//!   uncoupled modes* (asserted as a design guard — the analytic σ must be ≤ a third of the
//!   bound — so the window cannot be shortened unnoticed; the guard is necessary, not
//!   sufficient, see next). CSVR integrates its OU step exactly for any dt/τ and stays
//!   canonical (Bussi 2007, §II), so the question the test asks — does the thermostat hold the
//!   target on DFT forces with the right K̄ and DOF — does not depend on τ. The settle window
//!   is 25 fs = 5τ, longer than any H₃⁺ period, for the in-phase Maxwell–Boltzmann start's
//!   K → V transient (Task 4 note). T is collected per step from the snapshots (the driver's
//!   history keeps only the last 2000 points); the in-run estimates of σ_mean — block averages
//!   and σ_T·√(2τ_int/T_w) from the integrated autocorrelation time of T — are printed.
//!   **What two independent seeds say (fix round 1).** Seed 7 (`SEED`, the run of record):
//!   ⟨T⟩ = 335.1 K (+11.7 %); seed 11 (`DYNAMICS_SEED=11`): 240.4 K (−19.9 %, passing by
//!   0.1 %). Each run's own estimates agree with the analytic σ (blocks 17.0 / 14.1 K, τ_int
//!   3.5 / 3.9 fs → 18.5 / 16.2 K), yet the two means differ by 95 K = 3.7σ of their
//!   difference: on the real molecule the window mean is wider than the uncoupled-mode theory
//!   says. The autocorrelation of T shows why the in-run estimates cannot see it — a plateau
//!   ρ ≈ 0.05–0.1 from 5 to 25 fs that neither 25 fs blocks nor the first-zero τ_int (12 fs)
//!   resolve: a slow component with ~10 % of the variance and a correlation time of order
//!   100 fs (rotation–vibration energy exchange on the egg-box, large-amplitude motion of the
//!   soft smeared molecule; the toy has neither) would give σ_mean ≈ 35 K, with both seeds
//!   inside 1.7σ. No bias is indicated — the two deviations have opposite signs, and no
//!   positive systematic exists in the mechanism: CSVR pins the velocity marginal at the true
//!   β, so ⟨K⟩ = K̄ exactly whatever the integrator does to the positions (Verlet's shadow term
//!   goes into ⟨V⟩ = K̄/(1 − h²/4), as the NVE run shows: ⟨T⟩ 24.9 vs the harmonic 25.1 K,
//!   −0.6 %); the warm-path heating, the E_conserved drift +1.8e-5 Ha/ps, shifts the OU mean
//!   by Pτ/K̄ ≈ 3e-5; the 5τ settle leaves −0.4 K of the start transient. **Consequently the
//!   20 % gate is empirically a 1.5–3σ statement on a fresh trajectory, not 3.3σ**: it is a
//!   gross-error detector (a DOF or K̄ error of 2× fails it outright) that is deterministic on
//!   one adapter (bit-reproducible GPU, seeded RNG) and may trip on another by sampling alone.
//!   What would settle the slow component is a 3 ps run (~12 000 steps, ≈ 15 min solo), which
//!   the suite's budget does not hold; it is the handoff's recommendation, with the gate kept
//!   at the brief's 20 % (never loosened) until then.
//! - `rewind_retraces_on_the_gpu`: 50 steps, `flip_velocities`, 50 steps, flip → the start.
//!   Verlet is exactly reversible in f64 (`sim::integrator` tests: 1e-10 over 200 steps); here
//!   the forces on the way back are *re-solved* from different warm states and carry the GPU's
//!   warm-path force scatter δF (≤ 3.7e-6 Ha/Bohr, `gpu_vs_cpu` `F_BOUND_H2` history), which
//!   accumulates as δr ≈ (δF dt/m)·dt·√(N³/3) ≈ 1e-4 Bohr at δF = 3e-6 and N = 100, 3e-5 at
//!   1e-6. Measured 5.8e-6 Bohr on this adapter, i.e. an effective δF ≈ 1.7e-7 between the two
//!   passes. Bound `REWIND_BOUND` = 5e-5 (brief 1e-4; the margin is adapter-pinned, see the
//!   constant).
//!
//! The GPU work of concurrently running tests is serialised through `GPU`, one evaluation at a
//! time, so the per-solve times in the timing rows are uncontended while the tests interleave.
use gpu::GpuContext;
use sim::{DynCommand, Dynamics, Integrator, Nucleus, Scene};
use std::fmt::Write as _;
use std::sync::Mutex;
use std::time::Instant;

mod common;
use common::{
    au_per_ps, bond, chrono_free_date, fit_line, h2, h3_plus, locate_surface, ms, never, no_augment, period_from_upward_crossings, relax, scene, solve,
    timing_row, triangle, EvalLog, Locked, DISPLACEMENT, DT_FS, H, H3_SIDE0, HARTREE_IN_CM, N_H2, TOL_H2,
};

/// Timing rows for `last-run-dynamics.md`, rewritten whenever a test finishes.
static TIMING: Mutex<Vec<String>> = Mutex::new(Vec::new());

// ---- the discrete problem (`common`: H, N_H2, TOL_H2, DT_FS; H₂ vibration: RE_BRACKET,
// RE_TOL, DISPLACEMENT; H₃⁺: H3_SIDE0, RELAX_*; display: HARTREE_IN_CM) ----
/// H₂O's grid: the viewer's `h2o` preset (`crates/app/presets/h2o.scene`: 57³ = 14.25 Bohr), used
/// only by the ignored limitation test; the brief's 33³ (8 Bohr) warns "box too small".
const N_H2O_PRESET: usize = 57;

// ---- H₂ vibration ----
const VIB_STEPS: usize = 400;
/// |ω_MD/ω_static − 1| bound. Brief 3 %. Achieved −5.6e-4 (2026-10-13, `references.md`), but
/// that is a partial cancellation of three known systematic terms — Verlet +9.7e-4 (h = 0.1525,
/// exact), the 3-point truncation of k (k_F/k_E = 1.0016 ⇒ −0.8e-3 on the ratio) and the
/// anharmonic softening at this amplitude (the remainder, −0.7e-3) — whose aligned sum is
/// 2.5e-3 (module doc). A bound must not rely on the cancellation, so it is set at 2× that sum
/// (9× the achieved), 6× tighter than the brief's. Never loosened from here (CLAUDE.md rule 6).
const OMEGA_BOUND: f64 = 5e-3;

// ---- NVE ----
const NVE_STEPS: usize = 2000;
/// |slope of E_total| bound, Ha/ps (brief; budget in `references.md`).
const DRIFT_BOUND: f64 = 2e-4;
/// f32 path scatter of the GPU H₂ energy per solve (M3 Task 7: cold vs warm-from-neighbour).
const H2_PATH_SCATTER: f64 = 6e-7;
/// Margin on the derived fluctuation bound (controller: ≥ 3×).
const FLUCT_MARGIN: f64 = 3.0;
/// The brief's fixed fluctuation figure, asserted on the raw (un-detrended) peak to peak as
/// well (achieved 3.57e-6 Ha, 2.8× margin); the derived bound above is the primary one.
const BRIEF_FLUCT: f64 = 1e-5;

// ---- thermostat ----
const T_TARGET: f64 = 300.0;
/// The interactive SCF tolerance (the viewer's): the thermostat test is the one that runs the
/// MD path at it.
const TOL_INTERACTIVE: f64 = 1e-5;
/// CSVR coupling time (fs). Module doc: with σ_mean/T ≈ 0.5·√(2τ/T_w) the brief's τ = 10 fs over
/// its 100 fs window makes the 20 % bound a 1σ statement; 5 fs over the 675 fs window below
/// makes it 3.3σ for uncoupled modes (a user-plausible equilibration coupling; 10 fs would
/// need a 1.1 ps window) — empirically 1.5–3σ on a fresh trajectory (two seeds, module doc:
/// a ~100 fs slow component the uncoupled-mode σ does not see).
const THERMO_TAU_FS: f64 = 5.0;
const THERMO_STEPS: usize = 2800;
/// Steps excluded from the average: 25 fs = 5τ, and longer than the slowest H₃⁺ period, for the
/// in-phase Maxwell–Boltzmann start's K → V transient (Task 4 note).
const THERMO_SETTLE_STEPS: usize = 100;
/// |⟨T⟩/T_target − 1| bound (brief).
const T_BOUND: f64 = 0.2;
/// Default seed of the driver (its thermostat's RNG); `DYNAMICS_SEED` overrides it for an offline
/// second-seed run (reported, never part of the gate).
const SEED: u64 = 7;

fn seed() -> u64 {
    std::env::var("DYNAMICS_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(SEED)
}
/// The brief's H₂O configuration (`thermostat_h2o_preset_known_limitation`, `#[ignore]`d).
const H2O_TAU_FS: f64 = 10.0;
const H2O_STEPS: usize = 600;
const H2O_SETTLE_STEPS: usize = 200;

// ---- rewind ----
const REWIND_STEPS: usize = 50;
/// Start of the rewind run (≈ R_e(σ) + 0.05; the exact minimum is irrelevant to reversibility).
const REWIND_R0: f64 = 1.84;
/// max |Δr| after n → flip → n → flip, Bohr. Brief 1e-4; achieved 5.8e-6 on the RX 7900 XTX
/// (2026-10-13) → **5e-5** (2× tighter than the brief, 9× the achieved). Not 3× the achieved:
/// the GPU is bit-reproducible on one adapter, so the single measured value carries no scatter
/// sample, and what it measures is unmodelled — the warm-path force irreproducibility between
/// the two passes (the backward leg warm-starts from the forward leg's parked state and its
/// first density extrapolation points the wrong way in time; the SCF then converges to tol
/// from a different guess). Inverting the random-walk estimate of the module doc, 5.8e-6 Bohr
/// over 100 steps is an rms δF ≈ 1.7e-7 Ha/Bohr per step, 20× below the GPU-vs-CPU force
/// figure (3.7e-6, which contains the CPU's own Δρ term); another adapter's f32 path may sit a
/// few × away. The bound corresponds to δF ≈ 1.5e-6. f64 Verlet itself retraces to 1e-10
/// (`sim::integrator::tests::verlet_is_time_reversible`).
const REWIND_BOUND: f64 = 5e-5;
/// H₂ harmonic frequency, experiment (NIST CCCBDB, ω_e) and LSDA (CCCBDB, SVWN/aug-cc-pVQZ).
const OMEGA_E_EXPT_CM: f64 = 4401.21;
const OMEGA_E_LSDA_CM: f64 = 4201.0;

fn gpu() -> Option<GpuContext> { common::gpu("dynamics") }

/// H₂O at the `h2o` preset geometry (O–H 1.81 Bohr, 104.5°), at rest.
fn h2o() -> Scene {
    scene(N_H2O_PRESET, vec![Nucleus::new(8, [0.0; 3]), Nucleus::new(1, [1.43, 1.11, 0.0]), Nucleus::new(1, [-1.43, 1.11, 0.0])])
}



/// Appends a row and rewrites `last-run-dynamics.md`.
fn record(ctx: &GpuContext, row: String) {
    let mut rows = TIMING.lock().unwrap_or_else(|e| e.into_inner());
    rows.push(row);
    let adapter = ctx.adapter.get_info();
    let mut md = String::new();
    let _ = writeln!(md, "# Dynamics suite timing (`validate::dynamics`, {} UTC)\n", chrono_free_date());
    let _ = writeln!(md, "Adapter: {} ({:?}, {:?}); release build; σ = h = 0.25; dt = {DT_FS} fs. Tests run concurrently with their GPU evaluations serialised, so \"GPU\" is the uncontended sum of this test's solves and \"wall\" includes waiting for the others. Wall times are reported, never asserted (R3).\n", adapter.name, adapter.backend, adapter.device_type);
    let _ = writeln!(md, "| test | system | grid | tol | solves | GPU s | ms/solve | SCF steps/solve | readbacks/solve | test wall s | result |\n|---|---|---|---|---|---|---|---|---|---|---|");
    for r in rows.iter() { md += r; md.push('\n'); }
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/last-run-dynamics.md");
    std::fs::write(path, &md).unwrap_or_else(|e| panic!("writing {path}: {e}"));
}


#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test dynamics")]
fn h2_vibration_frequency_matches_curvature_of_the_same_surface() {
    let Some(ctx) = gpu() else { return };
    let t_test = Instant::now();
    let (mut p, log) = Locked::new(ctx.clone(), TOL_H2);
    let surf = locate_surface(&mut p);
    let integ = Integrator::new_fs(DT_FS);
    let mut s = h2(surf.r_e + DISPLACEMENT);
    let mut f = surf.start_forces.clone();
    let mut x = vec![bond(&s) - surf.r_e];
    let mut e_tot = Vec::with_capacity(VIB_STEPS);
    let t_md = Instant::now();
    for _ in 0..VIB_STEPS {
        let (f_new, st) = integ.step(&mut s, &f, &mut p, &mut no_augment(), &mut never()).unwrap_or_else(|e| panic!("MD step: {e}"));
        f = f_new;
        x.push(bond(&s) - surf.r_e);
        e_tot.push(st.total);
    }
    let md_wall = t_md.elapsed();
    let (period, crossings) = period_from_upward_crossings(&x, integ.dt);
    let omega_md = 2.0 * std::f64::consts::PI / period;
    let h = surf.omega * integ.dt;
    let omega_verlet = (1.0 - h * h / 2.0).acos() / integ.dt;
    let ratio = omega_md / surf.omega;
    let (x_min, x_max) = x.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let (e_lo, e_hi) = e_tot.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let cm = |w: f64| w * HARTREE_IN_CM;
    let log = log.lock().unwrap().clone();
    let mut out = String::new();
    let _ = writeln!(out, "H₂ vibration, {N_H2}³, σ = h = {H}, tol {TOL_H2:.0e}, dt = {DT_FS} fs, {VIB_STEPS} steps from R_e + {DISPLACEMENT} at rest");
    out += &surf.log;
    let _ = writeln!(out, "  MD: {crossings} upward crossings, period {period:.3} a.u. = {:.4} fs; R − R_e ∈ [{x_min:+.5}, {x_max:+.5}] Bohr; E_total ∈ [{e_lo:+.9}, {e_hi:+.9}] (span {:.2e} Ha)", units::au_time_to_fs(period), e_hi - e_lo);
    let _ = writeln!(out, "  ω_MD = {omega_md:.6e} a.u. = {:.1} cm⁻¹; ω_static = {:.6e} = {:.1} cm⁻¹; ω_MD/ω_static − 1 = {:+.3e} (bound {OMEGA_BOUND:.0e}); h = ω dt = {h:.4}, Verlet ω̃/ω − 1 = {:+.3e}, ω_MD/ω̃ − 1 = {:+.3e}",
                     cm(omega_md), surf.omega, cm(surf.omega), ratio - 1.0, omega_verlet / surf.omega - 1.0, omega_md / omega_verlet - 1.0);
    let _ = writeln!(out, "  approximation gap (reported, not asserted): ω_MD {:.0} cm⁻¹ at σ = 0.25 vs experiment ω_e = {OMEGA_E_EXPT_CM} cm⁻¹ ({:+.1} %) and LSDA (CCCBDB SVWN/aug-cc-pVQZ) {OMEGA_E_LSDA_CM:.0} cm⁻¹ ({:+.1} %); the σ → 0 curvature of this solver, E″ ≈ 0.37 Ha/Bohr² (M1 Task 10, h = 0.15), gives {:.0} ± 60 cm⁻¹ (a two-digit E″ from a bias argument: it cannot discriminate experiment from LSDA) — the gap is the Gaussian smearing (repulsive 2πσ²Σρ(R_I) flattens the well: R_e {:.3} vs 1.451 Bohr), not the integrator",
                     cm(omega_md), 100.0 * (cm(omega_md) / OMEGA_E_EXPT_CM - 1.0), 100.0 * (cm(omega_md) / OMEGA_E_LSDA_CM - 1.0), cm((0.37 / surf.mu).sqrt()), surf.r_e);
    let _ = writeln!(out, "  {}; MD wall {:.1} s ({:.0} ms/step); test wall {:.1} s", log.summary(), md_wall.as_secs_f64(), ms(md_wall) / VIB_STEPS as f64, t_test.elapsed().as_secs_f64());
    println!("{out}");
    let result = format!("R_e {:.4}, k {:.4} ({} bisections, E(R_e) {:+.6}), ω_MD/ω_static − 1 = {:+.2e}", surf.r_e, surf.k, surf.bisections, surf.e[1], ratio - 1.0);
    record(&ctx, timing_row("h2_vibration", "H₂", N_H2, TOL_H2, &log, t_test.elapsed(), &result));
    assert!(log.warnings.iter().all(|w| !w.contains("box too small")), "the {N_H2}³ box is too small for H₂: {:?}", log.warnings);
    assert!(x_max > 0.9 * DISPLACEMENT && x_min < -0.8 * DISPLACEMENT, "not oscillating about R_e: R − R_e ∈ [{x_min}, {x_max}]");
    assert!((ratio - 1.0).abs() <= OMEGA_BOUND, "ω_MD/ω_static − 1 = {:+.3e} (bound {OMEGA_BOUND:.0e})", ratio - 1.0);
    // the two curvature estimates differ by their truncation terms (E⁗Δ²/12 vs E⁗Δ²/6): a
    // disagreement beyond a few 1e-3 would mean the forces are not the gradient of the energy
    assert!((surf.k_from_forces / surf.k - 1.0).abs() < 1e-2, "k from forces {} vs from energies {}", surf.k_from_forces, surf.k);
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test dynamics")]
fn nve_energy_drift_is_bounded() {
    let Some(ctx) = gpu() else { return };
    let t_test = Instant::now();
    let (mut p, log) = Locked::new(ctx.clone(), TOL_H2);
    let surf = locate_surface(&mut p);
    let h = surf.omega * units::fs_to_au_time(DT_FS);
    let verlet_swing = h * h / 4.0 * surf.e_mode;
    let fluct_bound = FLUCT_MARGIN * (verlet_swing + 2.0 * H2_PATH_SCATTER);

    let mut d = Dynamics::new(h2(surf.r_e + DISPLACEMENT), Box::new(p), DT_FS, seed());
    let s0 = d.snapshot();
    assert!(s0.last_error.is_none() && !s0.forces_stale, "{:?}", s0.last_error);
    assert_eq!(s0.kinetic, 0.0);
    let (mut t, mut e_tot, mut e_cons, mut x, mut temp) = (Vec::new(), Vec::new(), Vec::new(), vec![DISPLACEMENT], Vec::new());
    let t_md = Instant::now();
    for i in 1..=NVE_STEPS {
        d.apply(DynCommand::Step(1));
        let s = d.tick(f64::INFINITY, &mut never());
        assert!(s.last_error.is_none(), "step {i}: {:?}", s.last_error);
        assert_eq!((s.step, s.pending_steps, s.paused), (i as u64, 0, true));
        // NVE with no tools: the driver's invariant is the plain total, bitwise
        assert_eq!(s.energy_conserved.to_bits(), s.total.to_bits(), "step {i}: energy_conserved {} ≠ total {}", s.energy_conserved, s.total);
        t.push(s.sim_time);
        e_tot.push(s.total);
        e_cons.push(s.energy_conserved);
        x.push(bond(&s.scene) - surf.r_e);
        temp.push(s.temperature);
    }
    let md_wall = t_md.elapsed();
    let last = d.snapshot();
    let drift_driver = last.drift_per_ps.expect("≥ 100 points");
    let (slope_au, fluct_detrended, e_mean) = fit_line(&t, &e_tot);
    let drift_here = slope_au * au_per_ps();
    let (e_lo, e_hi) = e_tot.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let (period, crossings) = period_from_upward_crossings(&x, units::fs_to_au_time(DT_FS));
    let omega_md = 2.0 * std::f64::consts::PI / period;
    let t_mean = temp.iter().sum::<f64>() / temp.len() as f64;
    let (x_min, x_max) = x.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    // the history the UI plots is the same series
    assert_eq!(last.history.len(), NVE_STEPS);
    assert_eq!(last.history[NVE_STEPS - 1].1.to_bits(), e_tot[NVE_STEPS - 1].to_bits());
    let log = log.lock().unwrap().clone();
    let mut out = String::new();
    let _ = writeln!(out, "H₂ NVE, {N_H2}³, σ = h = {H}, tol {TOL_H2:.0e}, dt = {DT_FS} fs, {NVE_STEPS} steps = {:.0} fs through `Dynamics` from R_e + {DISPLACEMENT} at rest", units::au_time_to_fs(t[NVE_STEPS - 1]));
    out += &surf.log;
    let _ = writeln!(out, "  drift: driver {drift_driver:+.3e} Ha/ps, refit here {drift_here:+.3e} Ha/ps (bound {DRIFT_BOUND:.0e}); E_total mean {e_mean:+.9}, start {:+.9}, end {:+.9}", e_tot[0], e_tot[NVE_STEPS - 1]);
    let _ = writeln!(out, "  fluctuation: detrended peak-to-peak {fluct_detrended:.3e} Ha, raw {:.3e} Ha; derived bound {fluct_bound:.2e} = {FLUCT_MARGIN} × (h²/4·E_mode {verlet_swing:.2e} + 2 × {H2_PATH_SCATTER:.0e}) with h = {h:.4}, E_mode = {:.3e}; brief's fixed figure {BRIEF_FLUCT:.0e}: {}",
                     e_hi - e_lo, surf.e_mode, if fluct_detrended < BRIEF_FLUCT { "met" } else { "not met" });
    let _ = writeln!(out, "  trajectory: {crossings} upward crossings over {NVE_STEPS} steps, ω_MD = {omega_md:.6e} (ω_MD/ω_static − 1 = {:+.3e}); R − R_e ∈ [{x_min:+.5}, {x_max:+.5}]; ⟨T⟩ = {t_mean:.1} K (= E_mode/(3 k_B) ⇒ {:.1} K for a harmonic mode)",
                     omega_md / surf.omega - 1.0, units::hartree_to_kelvin(surf.e_mode / 3.0));
    let _ = writeln!(out, "  {}; MD wall {:.1} s ({:.0} ms/step, driver ms_per_step {:.0}); test wall {:.1} s", log.summary(), md_wall.as_secs_f64(), ms(md_wall) / NVE_STEPS as f64, last.ms_per_step, t_test.elapsed().as_secs_f64());
    println!("{out}");
    let result = format!("drift {drift_driver:+.2e} Ha/ps, fluct {fluct_detrended:.2e} Ha (bound {fluct_bound:.1e})");
    record(&ctx, timing_row("nve_drift", "H₂", N_H2, TOL_H2, &log, t_test.elapsed(), &result));
    assert!(log.warnings.iter().all(|w| !w.contains("box too small")), "{:?}", log.warnings);
    assert!((drift_driver - drift_here).abs() <= 1e-9 * drift_here.abs().max(1e-6), "driver drift {drift_driver} vs refit {drift_here}");
    assert!(drift_driver.abs() <= DRIFT_BOUND, "drift {drift_driver:+.3e} Ha/ps (bound {DRIFT_BOUND:.0e})");
    assert!(fluct_detrended <= fluct_bound, "E_total fluctuation {fluct_detrended:.3e} Ha (derived bound {fluct_bound:.2e})");
    // the brief's statement, on the raw (un-detrended) series: achieved 3.57e-6 (2.8× margin)
    assert!(e_hi - e_lo <= BRIEF_FLUCT, "raw E_total peak-to-peak {:.3e} Ha (brief's {BRIEF_FLUCT:.0e})", e_hi - e_lo);
}


struct ThermoRun {
    mean: f64,
    sigma_mean: f64,
    failed: Option<(usize, String)>,
    log: EvalLog,
}

/// One thermostatted run through the driver: optional relaxation on the surface, R7 start,
/// `steps` steps, statistics over the steps after `settle` (module doc). A provider failure ends
/// the run (recorded in `failed`) so the ignored H₂O test can report instead of panicking.
fn thermostat_run(ctx: &GpuContext, label: &str, start: Scene, tol: f64, tau_fs: f64, steps: usize, settle: usize, do_relax: bool) -> ThermoRun {
    let t_test = Instant::now();
    let (mut p, log) = Locked::new(ctx.clone(), tol);
    let mut out = String::new();
    let n = start.cell.n;
    let _ = writeln!(out, "{label} CSVR, {n}³, σ = h = {H}, tol {tol:.0e}, dt = {DT_FS} fs, T = {T_TARGET} K, τ = {tau_fs} fs (dt/τ = {:.3}), {steps} steps = {:.0} fs, Maxwell–Boltzmann start (R7, seed {}); start geometry {}",
                     DT_FS / tau_fs, steps as f64 * DT_FS, seed(), triangle(&start));
    let start = if do_relax {
        let (s, _, _) = relax(&mut p, start, &mut out);
        let _ = writeln!(out, "  relaxed geometry {} (positions {:?})", triangle(&s), s.nuclei.iter().map(|n| n.pos).collect::<Vec<_>>());
        s
    } else { start };
    let mut d = Dynamics::new(start, Box::new(p), DT_FS, seed());
    let s0 = d.snapshot();
    assert!(s0.last_error.is_none() && !s0.forces_stale, "{:?}", s0.last_error);
    let e_ks0 = s0.energy_ks;
    let max_f0 = s0.forces.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
    d.apply(DynCommand::SetThermostat(Some((T_TARGET, tau_fs))));
    let s1 = d.snapshot();
    assert!(s1.last_rejection.is_none(), "{:?}", s1.last_rejection);
    assert!(s1.notes.last().is_some_and(|n| n.contains("Maxwell")), "R7 start expected: {:?}", s1.notes);
    assert_eq!(s1.scene.dof(), 6);
    // the draw rescales to exactly K̄: the readout starts at the target, with the COM at rest
    assert!((s1.temperature / T_TARGET - 1.0).abs() < 1e-9, "T after the draw = {}", s1.temperature);
    let com = s1.scene.com_velocity();
    assert!(com.iter().all(|v| v.abs() < 1e-12), "COM velocity after the draw {com:?}");
    let t_md = Instant::now();
    let mut failed = None;
    let mut done = 0;
    let mut temps: Vec<f64> = Vec::with_capacity(steps);
    for i in 1..=steps {
        d.apply(DynCommand::Step(1));
        let s = d.tick(f64::INFINITY, &mut never());
        if let Some(e) = s.last_error {
            failed = Some((i, e));
            break;
        }
        assert_eq!(s.step, i as u64);
        temps.push(s.temperature);
        done = i;
    }
    let md_wall = t_md.elapsed();
    let last = d.snapshot();
    // the history (the UI's series) is the tail of the same numbers
    assert_eq!(last.history.len(), done.min(sim::dynamics::HISTORY_LEN));
    if let Some(p) = last.history.last() { assert_eq!(p.2.to_bits(), temps[done - 1].to_bits()); }
    let log = log.lock().unwrap().clone();
    let _ = writeln!(out, "  E_KS at the start {e_ks0:+.6} Ha (max|F| {max_f0:.1e}); first steps: T = {:.0}, {:.0}, {:.0}, {:.0} K; E_KS at the end {:+.6} (ΔE_KS {:+.2e})",
                     temps.first().copied().unwrap_or(f64::NAN), temps.get(1).copied().unwrap_or(f64::NAN), temps.get(2).copied().unwrap_or(f64::NAN), temps.get(3).copied().unwrap_or(f64::NAN), last.energy_ks, last.energy_ks - e_ks0);
    let at = |k: usize| temps.get(k - 1).map_or("—".to_string(), |t| format!("{t:.0}"));
    let _ = writeln!(out, "  T at steps 5 / 10 / 20 / 50 / 100 / 200 / 400 / {steps}: {} / {} / {} / {} / {} / {} / {} / {} K; final geometry {}",
                     at(5), at(10), at(20), at(50), at(100), at(200), at(400), at(steps), triangle(&last.scene));
    if let Some((i, e)) = &failed {
        let _ = writeln!(out, "  PROVIDER FAILURE at step {i} ({:.0} fs): {e}; last good geometry {}, T = {:.0} K; {} steps completed", *i as f64 * DT_FS, triangle(&last.scene), last.temperature, done);
    }
    let (mut mean, mut sigma_mean) = (f64::NAN, f64::NAN);
    if temps.len() > settle + 50 {
        let window = &temps[settle..];
        mean = window.iter().sum::<f64>() / window.len() as f64;
        let var = window.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / window.len() as f64;
        let (t_lo, t_hi) = window.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        // block averages (100 steps = 25 fs ≥ 5τ) → a standard error of the mean from the data
        let block = 100;
        let blocks: Vec<f64> = window.chunks_exact(block).map(|c| c.iter().sum::<f64>() / block as f64).collect();
        let bm = blocks.iter().sum::<f64>() / blocks.len() as f64;
        let se_blocks = (blocks.iter().map(|b| (b - bm).powi(2)).sum::<f64>() / (blocks.len() as f64 - 1.0).max(1.0) / blocks.len() as f64).sqrt();
        // the derived expectation (module doc): σ_mean/T = 0.5 √(2τ(1 − (τ/T_w)(1 − e^{−T_w/τ}))/T_w)
        let t_w = window.len() as f64 * DT_FS;
        sigma_mean = 0.5 * (2.0 * tau_fs / t_w * (1.0 - tau_fs / t_w * (1.0 - (-t_w / tau_fs).exp()))).sqrt() * T_TARGET;
        let settle_mean = temps[..settle.min(temps.len())].iter().sum::<f64>() / settle.min(temps.len()) as f64;
        let _ = writeln!(out, "  mean over the first {settle} steps {settle_mean:.1} K; window of {} steps ({t_w:.0} fs): ⟨T⟩ = {mean:.1} K ({:+.1} % of target, bound ±{:.0} %), sample σ_T = {:.0} K ({:.0} %; canonical √(2/N_f) = {:.0} %), T ∈ [{t_lo:.0}, {t_hi:.0}] K",
                         window.len(), 100.0 * (mean / T_TARGET - 1.0), 100.0 * T_BOUND, var.sqrt(), 100.0 * var.sqrt() / T_TARGET, 100.0 * (2.0f64 / 6.0).sqrt());
        let _ = writeln!(out, "  block means ({block} steps): {}", blocks.iter().map(|b| format!("{b:.0}")).collect::<Vec<_>>().join(" "));
        // data-driven standard error: integrated autocorrelation time of T over the window,
        // τ_int = (½ + Σ_{k≥1} ρ_k)·dt summed to the first non-positive ρ_k (lag ≤ n/4)
        let n = window.len();
        let rho = |lag: usize| (0..n - lag).map(|i| (window[i] - mean) * (window[i + lag] - mean)).sum::<f64>() / (n - lag) as f64 / var;
        let (mut tau_int_steps, mut lag) = (0.5, 1usize);
        while lag <= n / 4 {
            let r = rho(lag);
            if r <= 0.0 { break; }
            tau_int_steps += r;
            lag += 1;
        }
        let tau_int_fs = tau_int_steps * DT_FS;
        let se_acf = var.sqrt() * (2.0 * tau_int_fs / t_w).sqrt();
        let _ = writeln!(out, "  autocorrelation of T: ρ(lag) = {:.2} / {:.2} / {:.2} / {:.2} / {:.2} at 4 / 20 / 40 / 100 / 200 steps; first ρ ≤ 0 at lag {lag} ({:.0} fs); τ_int = {tau_int_fs:.1} fs = {:.1} τ → SE(⟨T⟩) = σ_T√(2τ_int/T_w) = {se_acf:.1} K ({:.1} %; |⟨T⟩ − T| = {:.2}σ_acf)",
                         rho(4), rho(20), rho(40), rho(100.min(n / 4)), rho(200.min(n / 4)), lag as f64 * DT_FS, tau_int_fs / tau_fs, 100.0 * se_acf / T_TARGET, (mean - T_TARGET).abs() / se_acf);
        let _ = writeln!(out, "  standard error of ⟨T⟩: derived {sigma_mean:.1} K ({:.1} %; the bound is {:.1}σ), from {} blocks {se_blocks:.1} K ({:.1} %); |⟨T⟩ − T| = {:.2}σ_derived",
                         100.0 * sigma_mean / T_TARGET, T_BOUND * T_TARGET / sigma_mean, blocks.len(), 100.0 * se_blocks / T_TARGET, (mean - T_TARGET).abs() / sigma_mean);
    }
    let _ = writeln!(out, "  bookkeeping: E_conserved drift {:+.3e} Ha/ps (Bussi's effective energy, reported), W = total − conserved = {:+.3e} Ha, notes {:?}",
                     last.drift_per_ps.unwrap_or(f64::NAN), last.total - last.energy_conserved, last.notes);
    let _ = writeln!(out, "  {}; MD wall {:.1} s ({:.0} ms/step); test wall {:.1} s", log.summary(), md_wall.as_secs_f64(), ms(md_wall) / done.max(1) as f64, t_test.elapsed().as_secs_f64());
    println!("{out}");
    let result = match &failed {
        Some((i, _)) => format!("provider failure at step {i}; ⟨T⟩ = {mean:.0} K"),
        None => format!("⟨T⟩ = {mean:.0} K ({:+.1} %, {:.2}σ)", 100.0 * (mean / T_TARGET - 1.0), (mean - T_TARGET).abs() / sigma_mean),
    };
    record(ctx, timing_row(&format!("thermostat ({label})"), label, n, tol, &log, t_test.elapsed(), &result));
    ThermoRun { mean, sigma_mean, failed, log }
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test dynamics")]
fn thermostat_reaches_target_temperature() {
    let Some(ctx) = gpu() else { return };
    let r = thermostat_run(&ctx, "H₃⁺", h3_plus(H3_SIDE0), TOL_INTERACTIVE, THERMO_TAU_FS, THERMO_STEPS, THERMO_SETTLE_STEPS, true);
    assert!(r.failed.is_none(), "provider failure: {:?}", r.failed);
    assert!(r.log.warnings.iter().all(|w| !w.contains("box too small")), "{:?}", r.log.warnings);
    // the analytic design guard (necessary, not sufficient — module doc): the window may not be
    // shortened to where the uncoupled-mode σ alone makes the bound < 3σ
    assert!(r.sigma_mean <= T_BOUND * T_TARGET / 3.0, "test design: the analytic σ of the window mean ({:.1} K) must be ≤ a third of the bound", r.sigma_mean);
    assert!((r.mean / T_TARGET - 1.0).abs() <= T_BOUND, "⟨T⟩ = {:.1} K vs {T_TARGET} (bound {:.0} %)", r.mean, 100.0 * T_BOUND);
}

/// The brief's configuration — H₂O at the `h2o` preset geometry, tol 1e-5, τ = 10 fs, 600 steps —
/// on the preset's own 57³ grid (the brief said 33³, whose 8 Bohr box the solver flags "box too
/// small"), `#[ignore]`d as a documented limitation: it runs to the end but cannot be a gate (all
/// numbers 57³ unless marked). The preset geometry (O–H 1.81 Bohr, 104.5°, the physical one) is
/// **0.160 Ha above the minimum of the σ = h = 0.25 surface**, which is near-linear: quenched
/// relaxation from the preset converges in 74 steps to O–H 2.338 Bohr, 177°, E −37.6898 vs −37.5298
/// Ha at the preset (no box warning on either). Released into motion that energy — 2ΔE/(6 k_B) ≈
/// 16 800 K over the molecule's six degrees of freedom — heats the molecule while it opens toward
/// linear.
///
/// **Measured after R15 (2026-10-13, RX 7900 XTX, seed 7; the run of record).** All 600 steps
/// complete. The warm, density-extrapolated SCF still stalls twice — at step 2 (the stall that
/// used to end the run: Δρ 8.5e-5 after 200 iterations, the same deterministic warm attempt) and
/// at step 281 — and both times the provider's cold retry converges (R15: the costlier of the two
/// steps took 24 s instead of ≈ 1 s, with a note; never a wrong answer). T after the Maxwell–Boltzmann start: 583,
/// 1899, 3876, 6097 K over the first four steps, 11 539 K at step 10, 3381 K at step 100, 862 K at
/// 200, 294 K at 400, 44 K at 600; E_KS falls by 0.156 Ha (−37.5298 → −37.6853 Ha, the bent →
/// near-linear release; final O–H 2.41/2.42 Bohr, H–H 4.74 Bohr ≈ 158°) and the thermostat books
/// −0.154 Ha of work. ⟨T⟩ over the brief's window (steps 200–600, 100 fs) is **544 K (+81 %)**,
/// block means 784 / 595 / 480 / 316 K: with τ = 10 fs and the slow bend–stretch exchange of a
/// molecule this hot, 50 fs of settling does not drain 0.16 Ha, so the bound fails on the
/// physics of the start, not on the thermostat. E_conserved drifts −9.4e-4 Ha/ps (tol 1e-5 SCF lag
/// and Verlet error on 0.3 Ha/Bohr forces at thousands of kelvin, not separated; the H₃⁺ gate
/// at 300 K: +1.8e-5). Cost:
/// 1.03 s per step (9.1 SCF steps, 369 readbacks), test wall 618 s — over any gate budget. Before
/// R15 (Task 7 fix round 1) the run paused at step 2; the stall also occurs in plain NVE from the
/// preset, with or without thermal velocities and at tol 1e-6, at dt 0.5 fs the warm path converges
/// every step but needs 52–179 iterations on some, and at 33³ it stalled at step 110 near 172°
/// where a cold GPU solve (13 steps) and the CPU reference (14) both converge — a DIIS stagnation
/// at Δρ ≈ 1e-4 on the warm path, trigger unresolved. Not a gate: the window mean measures how
/// fast 0.16 Ha drains, and the run takes 10 min. A gate would start from the relaxed near-linear
/// minimum (as `thermostat_reaches_target_temperature` does for H₃⁺).
/// Run it with `cargo test -p validate --release --test dynamics -- --ignored --nocapture thermostat_h2o`.
#[test]
#[ignore = "documented limitation: the h2o preset is 0.16 Ha above the σ = 0.25 minimum (near-linear); the release heats it to ~11 500 K and τ = 10 fs has not drained it in the brief's window (⟨T⟩ 544 K); 10 min wall (see doc)"]
fn thermostat_h2o_preset_known_limitation() {
    let Some(ctx) = gpu() else { return };
    let r = thermostat_run(&ctx, "H₂O", h2o(), TOL_INTERACTIVE, H2O_TAU_FS, H2O_STEPS, H2O_SETTLE_STEPS, false);
    assert!(r.failed.is_none(), "provider failure: {:?}", r.failed);
    assert!((r.mean / T_TARGET - 1.0).abs() <= T_BOUND, "⟨T⟩ = {:.1} K vs {T_TARGET} (bound {:.0} %)", r.mean, 100.0 * T_BOUND);
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test dynamics")]
fn rewind_retraces_on_the_gpu() {
    let Some(ctx) = gpu() else { return };
    let t_test = Instant::now();
    let (mut p, log) = Locked::new(ctx.clone(), TOL_H2);
    let integ = Integrator::new_fs(DT_FS);
    let start = h2(REWIND_R0);
    let mut s = start.clone();
    let r0 = solve(&mut p, &s, "start");
    let mut f = r0.forces.clone();
    let e_start = r0.energy;
    let mut e_turn = 0.0;
    for _ in 0..REWIND_STEPS {
        let (f_new, st) = integ.step(&mut s, &f, &mut p, &mut no_augment(), &mut never()).unwrap_or_else(|e| panic!("forward step: {e}"));
        f = f_new;
        e_turn = st.total;
    }
    let far = (bond(&s) - REWIND_R0).abs();
    let v_turn = s.nuclei[0].vel[0];
    Integrator::flip_velocities(&mut s);
    let mut e_back = 0.0;
    for _ in 0..REWIND_STEPS {
        let (f_new, st) = integ.step(&mut s, &f, &mut p, &mut no_augment(), &mut never()).unwrap_or_else(|e| panic!("backward step: {e}"));
        f = f_new;
        e_back = st.total;
    }
    Integrator::flip_velocities(&mut s);
    let (mut dr, mut dv) = (0.0f64, 0.0f64);
    for (a, b) in s.nuclei.iter().zip(&start.nuclei) {
        for k in 0..3 {
            dr = dr.max((a.pos[k] - b.pos[k]).abs());
            dv = dv.max((a.vel[k] - b.vel[k]).abs());
        }
    }
    let log = log.lock().unwrap().clone();
    let mut out = String::new();
    let _ = writeln!(out, "H₂ rewind, {N_H2}³, σ = h = {H}, tol {TOL_H2:.0e}, dt = {DT_FS} fs: R₀ = {REWIND_R0} at rest, {REWIND_STEPS} steps → flip → {REWIND_STEPS} steps → flip");
    let _ = writeln!(out, "  at the turn: |R − R₀| = {far:.4} Bohr, v_x(0) = {v_turn:+.3e}; E_total start {e_start:+.9}, turn {e_turn:+.9}, back {e_back:+.9} (back − start {:+.2e} Ha)", e_back - e_start);
    let _ = writeln!(out, "  max |Δr| = {dr:.3e} Bohr (bound {REWIND_BOUND:.0e}), max |Δv| = {dv:.3e} Bohr/a.u. (= {:.2e} Bohr per step)", dv * integ.dt);
    let _ = writeln!(out, "  {}; test wall {:.1} s", log.summary(), t_test.elapsed().as_secs_f64());
    println!("{out}");
    record(&ctx, timing_row("rewind", "H₂", N_H2, TOL_H2, &log, t_test.elapsed(), &format!("max Δr = {dr:.1e} Bohr")));
    assert!(far > 0.02, "test assumption: the molecule moved ({far})");
    assert!(dr <= REWIND_BOUND, "rewind missed the start by {dr:.3e} Bohr (bound {REWIND_BOUND:.0e})");
}
