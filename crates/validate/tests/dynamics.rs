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
//! - V5 (Fermi–Dirac, spec 2026-11-13 §9; plan Task 11; `references.md` "V5"):
//!   `work_integral_matches_free_energy_change` (CPU f64: the forces' work over 0.2 Bohr is −ΔF, not −ΔE) and
//!   `fd_nve_conserves_f_plus_k_through_the_ts_swing` (GPU: CO₂⁺'s bend through a 6.4 mHa TS swing conserves F + K).
//!   Both fail their pre-registered bounds as of 2026-11-17, each diagnosed to a term the bound lacks; a spec decision
//!   is pending (their docs). They add ≈ 15 min (CPU) and ≈ 17 min (GPU) to the suite.
//!
//! The GPU work of concurrently running tests is serialised through `GPU`, one evaluation at a
//! time, so the per-solve times in the timing rows are uncontended while the tests interleave.
use gpu::GpuContext;
use sim::{DynCommand, Dynamics, ForceProvider, Integrator, Nucleus, Scene};
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

// ---- V5: the work integral (spec 2026-11-13 §9 "V5 new tests" 1; plan Task 11; ruling R10) ----
//
// Derivation (atomic units: Ha, Bohr, Ha/Bohr). `qm::forces` returns F_x = −dF/dx of the Mermin free energy
// F = E − TS (stationary in ψ and f; `qm::forces` module doc, V3 confirms it on this geometry). Along a path x ∈ [0, L]
// of one coordinate, the work the forces do is W = ∫₀ᴸ F_x dx = −[F(L) − F(0)], so
//   ΔF + W = 0   and   ΔE + W = ΔE − ΔF = Δ(TS),
// nonzero wherever a partly occupied level's f moves (ruling R10: W is the work the forces do, +∫F·dx; the spec's
// "W = −∫F_HF·dx" written with "|W + ΔF|" is inconsistent, and `branch_jump` checks the same physics as
// |ΔF − (−∫F·dR)|).
// Quadrature: x_i = iΔ, Δ = L/40, i = 0 … 40. W₄₀ = composite Simpson on the 41 points, W₂₀ = Simpson on the 21 even
// points (spacing 2Δ). Simpson's error is −L(Δ⁴/180)F_x⁗ + O(Δ⁶), so W₄₀ − W_exact ≈ (W₂₀ − W₄₀)/15 in magnitude: the
// estimate |W₄₀ − W₂₀|/15.
// Noise: each solve's F_x carries a path scatter σ_F and F a scatter σ_E (warm vs cold at the start point, measured
// here). W₄₀ = Σ w_i F_x(x_i) with w_i > 0 and Σ w_i = L, so its noise is at most L·σ_F (fully correlated, the worst
// case); ΔF = F(L) − F(0) carries √2·σ_E. Bound (spec §9 V5 1, re-review R-SHOULD-C, the brief's verbatim form):
//   bound = |W₄₀ − W₂₀|/15 + 3·(L·σ_F + √2·σ_E),   |ΔF + W₄₀| ≤ bound,   |ΔE + W₄₀| ≥ 5·bound.
// The second line tells F's derivative from E's: Δ(TS) = −5.6 mHa over this path (measured: TS 10.42 → 4.83 mHa as
// the pinned pair's split widens; V3: d(TS)/dx ≈ −2.2e-2 Ha/Bohr at x = 0).

/// The work integral's path: atom 0 of `common::v3_h4()` moved along x (toward atom 1) from its V3 position by 0 … L,
/// L = 0.2 Bohr, in 40 steps (41 points). x ∈ [0, 0.2] widens the pinned pair's split from 12.9 mHa (u = 0.65, V3's
/// certified point) away from the square, where the broken-symmetry branch competes (spec §4.7(e)).
const WORK_PATH: f64 = 0.2;
const WORK_POINTS: usize = 41;
/// SCF and eigensolver tolerances of the work integral's CPU solves (plan Task 11: tol = eig_tol = 1e-9; `qm` asks its
/// Poisson solves for 1e-12 at this tol).
const WORK_TOL: f64 = 1e-9;
const WORK_EIG_TOL: f64 = 1e-9;

/// Neumaier-compensated sum (CLAUDE.md rule 4: energy reductions are compensated).
fn neumaier(terms: impl Iterator<Item = f64>) -> f64 {
    let (mut sum, mut c) = (0.0f64, 0.0f64);
    for x in terms {
        let t = sum + x;
        c += if sum.abs() >= x.abs() { (sum - t) + x } else { (x - t) + sum };
        sum = t;
    }
    sum + c
}

/// Composite Simpson on equally spaced samples `y` (odd count ≥ 3) at spacing `h`: (h/3)[y₀ + 4y₁ + 2y₂ + … + 4y_{n−1} + y_n].
fn simpson(y: &[f64], h: f64) -> f64 {
    assert!(y.len() >= 3 && y.len() % 2 == 1, "Simpson needs an odd number ≥ 3 of samples, got {}", y.len());
    let n = y.len() - 1;
    h / 3.0 * neumaier(y.iter().enumerate().map(|(i, &v)| v * if i == 0 || i == n { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 }))
}

/// `f` over `items` on at most 4 threads (`qm` is single-threaded; the machine is shared), results in input order.
fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out: Vec<Mutex<Option<R>>> = items.iter().map(|_| Mutex::new(None)).collect();
    std::thread::scope(|sc| {
        for _ in 0..4.min(items.len()) {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if i >= items.len() { break; }
                *out[i].lock().unwrap() = Some(f(&items[i]));
            });
        }
    });
    out.into_iter().map(|m| m.into_inner().unwrap().expect("every item ran")).collect()
}

/// One work-integral solve: x, F (`total`), E (`internal`), TS, F_x on the moved atom, SCF steps, final Δρ, face density.
struct WorkPoint { x: f64, f: f64, e: f64, ts: f64, force: f64, iterations: usize, delta_rho: f64, face: f64 }

/// V5 work integral [MUST-6] (spec 2026-11-13 §9 "V5 new tests" 1; plan Task 11; the derivation above): on V3's pinned
/// H₄ (`common::v3_h4()`, CPU f64, 41³ × 0.30, σ = h, tol = eig_tol = 1e-9, `FermiDirac { kt: ELECTRON_KT }`), the work
/// the Hellmann–Feynman forces do over 0.2 Bohr is −ΔF, and not −ΔE. 41 cold solves along the path (4 threads), the
/// start re-solved warm from x₁ (σ_F, σ_E: its warm − cold) and the end re-solved warm from x₃₉ (reported). Every solve
/// must converge with no warning but the box warning (V3's: the same Dirichlet box at every x, part of the discrete
/// problem). Asserted, verbatim from the brief: |ΔF + W₄₀| ≤ bound and |ΔE + W₄₀| ≥ 5·bound,
/// bound = |W₄₀ − W₂₀|/15 + 3·(0.2·σ_F + √2·σ_E). Numbers: `references.md` "V5".
/// **Run of record 2026-11-17: FAILS as pre-registered** — ΔF + W₄₀ = +7.97e-8 Ha against a bound of 2.03e-10, while
/// ΔE + W₄₀ = Δ(TS) = −5.59e-3 passes. The residual is the discrete force's box offset b = HF − (−dF/dx) (V3's +2.09e-7
/// Ha/Bohr at x = 0; +2.2 … +6.1e-7 along the path; 16× smaller in a 15 Bohr box, present on gapped H₂), for which the
/// bound has no term. Not re-bounded: a spec decision (`references.md` "V5", Step 3).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test dynamics")]
fn work_integral_matches_free_energy_change() {
    use qm::{forces, scf, KsState, Occupations, ScfSettings, System, ELECTRON_KT};
    let t_test = Instant::now();
    let (g, sys, atom, axis) = common::v3_h4();
    let mut set = ScfSettings::for_grid(&g);
    (set.tol, set.eig_tol, set.occupations) = (WORK_TOL, WORK_EIG_TOL, Occupations::FermiDirac { kt: ELECTRON_KT });
    let dx = WORK_PATH / (WORK_POINTS - 1) as f64;
    let xs: Vec<f64> = (0..WORK_POINTS).map(|i| i as f64 * dx).collect();
    let moved = |x: f64| {
        let mut nuclei = sys.nuclei.clone();
        nuclei[atom].pos[axis] += x;
        System { nuclei, n_up: sys.n_up, n_dn: sys.n_dn, ionic: None }
    };
    let point = |x: f64, warm: Option<&KsState>| -> (KsState, WorkPoint) {
        let m = moved(x);
        let (st, stats) = scf(&g, &m, &set, warm).unwrap_or_else(|e| panic!("work integral x = {x} ({}): {e}", if warm.is_some() { "warm" } else { "cold" }));
        let is_box = |w: &String| w.starts_with("density on box faces is ") && w.ends_with(": box too small");
        assert!(stats.warnings.iter().all(is_box) && stats.warnings.len() <= 1, "work integral x = {x}: {:?}", stats.warnings);
        let force = forces(&g, &st, &m.nuclei, set.sigma).unwrap()[atom][axis];
        let p = WorkPoint { x, f: stats.energy.total, e: stats.energy.internal, ts: stats.energy.ts, force, iterations: stats.iterations, delta_rho: stats.delta_rho, face: stats.boundary_density };
        (st, p)
    };
    let solved = par_map(&xs, |&x| point(x, None));
    let (start_warm, end_warm) = std::thread::scope(|sc| {
        let a = sc.spawn(|| point(xs[0], Some(&solved[1].0)).1);
        let b = sc.spawn(|| point(xs[WORK_POINTS - 1], Some(&solved[WORK_POINTS - 2].0)).1);
        (a.join().unwrap(), b.join().unwrap())
    });
    let pts: Vec<&WorkPoint> = solved.iter().map(|(_, p)| p).collect();
    let fx: Vec<f64> = pts.iter().map(|p| p.force).collect();
    let fx_even: Vec<f64> = fx.iter().step_by(2).copied().collect();
    let w40 = simpson(&fx, dx);
    let w20 = simpson(&fx_even, 2.0 * dx);
    let (first, last) = (pts[0], pts[WORK_POINTS - 1]);
    let (d_f, d_e, d_ts) = (last.f - first.f, last.e - first.e, last.ts - first.ts);
    let sigma_f = (start_warm.force - first.force).abs();
    let sigma_e = (start_warm.f - first.f).abs();
    let simpson_err = (w40 - w20).abs() / 15.0;
    let noise = 3.0 * (WORK_PATH * sigma_f + 2f64.sqrt() * sigma_e);
    let bound = simpson_err + noise;
    let wall = t_test.elapsed().as_secs_f64();
    let mut out = String::new();
    let _ = writeln!(out, "V5 work integral: H₄ 1.8 × 1.95 (`v3_h4`), atom {atom} along axis {axis}, x ∈ [0, {WORK_PATH}] in {} steps; 41³ × 0.30, σ = h, CPU f64, tol {WORK_TOL:.0e}, eig_tol {WORK_EIG_TOL:.0e}, FD kT {ELECTRON_KT}; wall {wall:.0} s", WORK_POINTS - 1);
    for p in &pts {
        let _ = writeln!(out, "  x {:.3}  F {:+.12}  E {:+.12}  TS {:.9e}  F_x {:+.10}  {} steps  Δρ {:.1e}  face {:.2e}", p.x, p.f, p.e, p.ts, p.force, p.iterations, p.delta_rho, p.face);
    }
    let _ = writeln!(out, "  start warm (from x₁) − cold: F {:+.3e}, E {:+.3e}, F_x {:+.3e} ({} steps); end warm (from x₃₉) − cold: F {:+.3e}, E {:+.3e}, F_x {:+.3e} ({} steps)",
                     start_warm.f - first.f, start_warm.e - first.e, start_warm.force - first.force, start_warm.iterations,
                     end_warm.f - last.f, end_warm.e - last.e, end_warm.force - last.force, end_warm.iterations);
    let _ = writeln!(out, "  W₄₀ {w40:+.12e}  W₂₀ {w20:+.12e}  |W₄₀ − W₂₀|/15 {simpson_err:.3e}");
    let _ = writeln!(out, "  ΔF {d_f:+.12e}  ΔE {d_e:+.12e}  Δ(TS) {d_ts:+.6e}");
    let _ = writeln!(out, "  ΔF + W₄₀ {:+.3e} (≤ bound {bound:.3e} = {simpson_err:.2e} + 3·({WORK_PATH}·σ_F {sigma_f:.2e} + √2·σ_E {sigma_e:.2e}) = {simpson_err:.2e} + {noise:.2e})", d_f + w40);
    let _ = writeln!(out, "  ΔE + W₄₀ {:+.6e} (≥ 5·bound {:.3e}); ΔF + W₂₀ {:+.3e}", d_e + w40, 5.0 * bound, d_f + w20);
    println!("{out}");
    assert!((d_f + w40).abs() <= bound, "|ΔF + W₄₀| = {:.3e} > bound {bound:.3e}: the forces' work is not −ΔF", (d_f + w40).abs());
    assert!((d_e + w40).abs() >= 5.0 * bound, "|ΔE + W₄₀| = {:.3e} < 5 × bound {bound:.3e}: the test does not tell F from E here", (d_e + w40).abs());
}

// ---- V5: Fermi–Dirac NVE through the TS swing (spec 2026-11-13 §9 "V5 new tests" 2; plan Task 11) ----
/// CO₂⁺'s grid (A1 note §2.3): 71³ × 0.20 = a 14 Bohr box, h_req of pseudo-O; the origin is a grid node.
const CO2_N: usize = 71;
const CO2_H: f64 = 0.20;
/// The start: both C–O bonds at R = 2.226 Bohr (R_e = 2.225), bent by β = 70° from linear (bond angle 110°), at rest.
const CO2_R: f64 = 2.226;
const CO2_BETA_DEG: f64 = 70.0;
/// The SCF tolerance asked of the GPU (the suite's); every solve along this trajectory has a partly occupied ↓ π level,
/// so the GPU stops at `qm::GPU_PARTIAL_TOL` = 1e-5 (spec §5.1) whatever is asked — `scf_tol_used` records it.
const CO2_TOL: f64 = TOL_H2;
/// Verlet steps of the NVE run (pre-registered, `references.md` "V5" Step 1): 600 × 0.25 fs = 150 fs ≥ 2 bend periods
/// (harmonic 260 steps; measured 292 with the bend–stretch exchange, so 2.05 periods — the design guard holds, thinly),
/// at 1.35–1.7 s per warm step inside one 15-min GPU batch.
const CO2_NVE_STEPS: usize = 600;
/// The bend's harmonic frequency (a.u.) and the energy the start puts in it, F(70°) − F(0°) (Ha): A1 note §2.2.
const CO2_OMEGA_B: f64 = 2.335e-3;
const CO2_E_MODE: f64 = 98.75e-3;
/// σ_path of F at the start under the 1e-5 partial-level stop (Ha): the largest |F_warm − F_cold| of five repeats
/// (2.154e-6), rounded up to two digits (`v5_co2_plus_cost_and_path_scatter`, `references.md` "V5" Step 1).
const CO2_PATH_SCATTER: f64 = 2.2e-6;
/// max TS − min TS along the run must reach this (Ha; spec §9 V5 2): the trajectory crosses the TS swing.
const TS_SWING_MIN: f64 = 1e-3;

/// CO₂⁺ (charge +1, auto multiplicity = doublet, `Accepted`: pseudised C and O) bent by `beta_deg` from linear with both
/// bonds `r`: C at the origin (a grid node), the O at (∓r sin(θ/2), −r cos(θ/2), 0) with θ = 180° − β the bond angle.
fn co2_plus(beta_deg: f64, r: f64) -> Scene {
    let half = (180.0 - beta_deg).to_radians() / 2.0;
    let (sx, cy) = (r * half.sin(), r * half.cos());
    let nuclei = vec![Nucleus::new(6, [0.0; 3]), Nucleus::new(8, [-sx, -cy, 0.0]), Nucleus::new(8, [sx, -cy, 0.0])];
    Scene { nuclei, cell: sim::Cell { n: CO2_N, h: CO2_H }, charge: 1, multiplicity: None, sigma: None, external_field: [0.0; 3], ionic: sim::IonicPolicy::Accepted, train_mask: None }
}

/// The bend β = 180° − ∠O–C–O (degrees) and the two C–O bond lengths of a CO₂ scene (C first).
fn co2_shape(s: &Scene) -> (f64, f64, f64) {
    let c = s.nuclei[0].pos;
    let d = |i: usize| -> [f64; 3] { std::array::from_fn(|k| s.nuclei[i].pos[k] - c[k]) };
    let (a, b) = (d(1), d(2));
    let (la, lb) = ((a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt(), (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt());
    let cos = ((a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (la * lb)).clamp(-1.0, 1.0);
    (180.0 - cos.acos().to_degrees(), la, lb)
}

/// V5 Step 1 (plan Task 11), a measurement, not a gate: CO₂⁺ at the NVE start on the GPU — one cold solve, then five
/// warm-vs-cold repeats (reset, a cold solve at a neighbour bent by δβ ∈ {+0.5°, −0.5°, +1°, −1°, +2°}, then the start
/// warm from it: F_warm − F_cold and the largest force-component change), then 20 Verlet steps from rest timed one by
/// one. Its numbers pre-register the NVE bounds in `references.md` ("Fermi–Dirac migration", V5).
/// Run: `cargo test -p validate --release --test dynamics -- --ignored --nocapture v5_co2_plus_cost_and_path_scatter`.
#[test]
#[ignore = "measurement (plan 2026-11-13 Task 11 Step 1): σ_path and the per-step cost of the V5 NVE case; numbers in references.md"]
fn v5_co2_plus_cost_and_path_scatter() {
    let Some(ctx) = gpu() else { return };
    let (mut p, log) = Locked::new(ctx.clone(), CO2_TOL);
    let start = co2_plus(CO2_BETA_DEG, CO2_R);
    let t = Instant::now();
    let cold = solve(&mut p, &start, "CO₂⁺ start, cold");
    let cold_s = t.elapsed().as_secs_f64();
    let e = &cold.stats.energy;
    println!("CO₂⁺ {CO2_N}³ × {CO2_H}, β = {CO2_BETA_DEG}°, R = {CO2_R}: cold F {:+.9} E {:+.9} TS {:.6e}, {} SCF steps, {cold_s:.1} s, tol used {:?}, face {:?} (rule {:.2e}), partial {:?}, forces {:?}",
             e.total, e.internal, e.ts, cold.stats.scf_steps, cold.stats.scf_tol_used, cold.stats.face_density, providers::box_affected_face_density(CO2_H),
             cold.stats.partial, cold.forces);
    let (mut d_f, mut d_e, mut d_force) = (Vec::new(), Vec::new(), Vec::new());
    for db in [0.5, -0.5, 1.0, -1.0, 2.0] {
        p.reset();
        let t = Instant::now();
        let nb = solve(&mut p, &co2_plus(CO2_BETA_DEG + db, CO2_R), "neighbour");
        let nb_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let w = solve(&mut p, &start, "CO₂⁺ start, warm");
        let w_s = t.elapsed().as_secs_f64();
        let df = w.stats.energy.total - e.total;
        let de = w.stats.energy.internal - e.internal;
        let dfo = w.forces.iter().flatten().zip(cold.forces.iter().flatten()).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        println!("  δβ {db:+.1}°: neighbour {} steps {nb_s:.1} s; warm {} steps {w_s:.1} s: F_warm − F_cold {df:+.3e}, E {de:+.3e}, max |ΔF_I| {dfo:.3e} Ha/Bohr, tol used {:?}",
                 nb.stats.scf_steps, w.stats.scf_steps, w.stats.scf_tol_used);
        d_f.push(df);
        d_e.push(de);
        d_force.push(dfo);
    }
    let rms = |v: &[f64]| (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt();
    let maxabs = |v: &[f64]| v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    println!("  path scatter of F: max |ΔF| {:.3e}, RMS {:.3e} (RMS/√2 {:.3e}); of E: max {:.3e}; forces: max {:.3e} Ha/Bohr",
             maxabs(&d_f), rms(&d_f), rms(&d_f) / 2f64.sqrt(), maxabs(&d_e), maxabs(&d_force));
    // 20 Verlet steps from rest, timed
    p.reset();
    let integ = Integrator::new_fs(DT_FS);
    let mut s = start.clone();
    let r0 = solve(&mut p, &s, "MD start");
    let mut f = r0.forces.clone();
    let mut walls = Vec::new();
    for i in 1..=20 {
        let t = Instant::now();
        let (f_new, st) = integ.step(&mut s, &f, &mut p, &mut no_augment(), &mut never()).unwrap_or_else(|e| panic!("MD step {i}: {e}"));
        let w = t.elapsed().as_secs_f64();
        f = f_new;
        let (beta, ra, rb) = co2_shape(&s);
        println!("  step {i}: {w:.2} s, {} SCF steps, F + K {:+.9}, TS {:.4e}, β {beta:.3}°, R {ra:.4}/{rb:.4}, tol used {:?}, branch_jump {:?}",
                 st.eval.scf_steps, st.total, st.eval.energy.ts, st.eval.scf_tol_used, st.eval.branch_jump);
        walls.push(w);
    }
    let mean = walls.iter().sum::<f64>() / walls.len() as f64;
    println!("  warm MD step: mean {mean:.2} s over 20 (max {:.2}); ⇒ 600 steps ≈ {:.1} min", walls.iter().fold(0.0f64, |m, &x| m.max(x)), 600.0 * mean / 60.0);
    println!("  {}", log.lock().unwrap().summary());
}

/// V5 NVE [MUST-6] (spec 2026-11-13 §9 "V5 new tests" 2; plan Task 11; bounds pre-registered in `references.md` "V5"
/// before this test existed). Under Fermi–Dirac occupations the forces are −dF/dR (F = E − TS, `qm::forces`), so
/// velocity Verlet conserves F + K, not E + K (E + K = (F + K) + TS moves with TS). CO₂⁺ (linear ²Π_g: at linearity the
/// ↓ π pair holds one hole and TS = 2kT ln 2 = 6.93 mHa; bending splits the pair and TS falls toward 0), started at
/// rest bent by 70° (TS 0.51 mHa), swings through linear and back, so TS sweeps its whole range twice per bend period.
/// `CO2_NVE_STEPS` velocity-Verlet steps at dt = 0.25 fs through `Integrator::step` on the GPU teacher. Asserted:
/// - the drift of F + K (least-squares slope) ≤ `DRIFT_BOUND` = 2e-4 Ha/ps (the suite's; this trajectory's budget,
///   ≤ 1.0e-4 Ha/ps, is in `references.md`);
/// - the peak-to-peak residual of F + K about that line ≤ FLUCT_MARGIN × (h_b²/4·E_mode + 2σ_path), h_b = ω_b·dt,
///   E_mode = 98.75 mHa and ω_b = 2.335e-3 from A1, σ_path = `CO2_PATH_SCATTER` (`nve_energy_drift_is_bounded`'s rule;
///   never scaled per atom);
/// - max TS − min TS ≥ `TS_SWING_MIN` = 1 mHa;
/// - the face density ≤ `providers::box_affected_face_density(0.20)` at every evaluation;
/// - the run covers ≥ 2 bend periods (a design guard on the step count: the period from the zero crossings of the
///   signed bend coordinate ȳ_O − y_C).
/// Recorded, not asserted (Task 8 amendment I1): the `branch_jump` flag count (expected 0) and the largest m/b, from a
/// replica of the provider's guard fed the same evaluations (its flags must equal the provider's). Printed, not
/// asserted: the shadow Hamiltonian H̃ (below), which separates the Verlet swing from force/energy inconsistency.
/// **Run of record 2026-11-17: FAILS the fluctuation bound as pre-registered** — 7.92e-5 Ha against 5.63e-5 (drift
/// +4.5e-6 Ha/ps, TS swing 6.42e-3, face 1.75e-6, 2.05 periods, 0 `branch_jump` flags, largest m/b 0.34: all pass).
/// H̃ keeps a 4.9e-5 slow wave: the GPU pseudo forces are −dF/dR only to ≈ 3e-5 Ha over the 70° bend (gapped neutral
/// CO₂ the same), and the bend–stretch exchange puts ≈ 40 mHa into the stiffer stretch; the pre-registered formula has
/// neither. Not re-bounded: a spec decision (`references.md` "V5", Step 3).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test dynamics")]
fn fd_nve_conserves_f_plus_k_through_the_ts_swing() {
    use providers::branch_jump::{noise_tier, BranchJumpGuard, Evaluation};
    let Some(ctx) = gpu() else { return };
    let t_test = Instant::now();
    let (mut p, log) = Locked::new(ctx.clone(), CO2_TOL);
    let integ = Integrator::new_fs(DT_FS);
    let h_b = CO2_OMEGA_B * integ.dt;
    let verlet_swing = h_b * h_b / 4.0 * CO2_E_MODE;
    let fluct_bound = FLUCT_MARGIN * (verlet_swing + 2.0 * CO2_PATH_SCATTER);
    let face_rule = providers::box_affected_face_density(CO2_H);
    let origin = qm::Grid::cube(CO2_N, CO2_H).origin;
    let z: Vec<u8> = co2_plus(CO2_BETA_DEG, CO2_R).nuclei.iter().map(|n| n.z).collect();
    // the replica of the provider's `branch_jump` guard (`providers::ks::judge_branch`: the same inputs; N↑ ≠ N↓, so no spin measure)
    let mut guard = BranchJumpGuard::new();
    let mut judge = |s: &Scene, forces: &[[f64; 3]], stats: &sim::EvalStats| {
        let positions: Vec<[f64; 3]> = s.nuclei.iter().map(|n| n.pos).collect();
        let tol_used = stats.scf_tol_used.expect("a Kohn-Sham evaluation records its stop");
        guard.judge(&Evaluation { z: &z, positions: &positions, forces, energy: stats.energy.total, cell: s.cell.clone(), origin, spin_abs: None,
                                  tier: noise_tier(false, tol_used, stats.energy.total) })
    };
    // the signed bend coordinate: mean O height minus C height (negative at the start, 0 at linear)
    let bend_y = |s: &Scene| 0.5 * (s.nuclei[1].pos[1] + s.nuclei[2].pos[1]) - s.nuclei[0].pos[1];

    let mut s = co2_plus(CO2_BETA_DEG, CO2_R);
    let r0 = solve(&mut p, &s, "CO₂⁺ NVE start");
    let j0 = judge(&s, &r0.forces, &r0.stats);
    assert!(!j0.judged && r0.stats.branch_jump.is_none());
    let mut f = r0.forces.clone();
    // per evaluation: velocities and forces, for the shadow-Hamiltonian diagnostic below (reported, never asserted)
    let (mut vels, mut frcs) = (vec![s.nuclei.iter().map(|n| n.vel).collect::<Vec<_>>()], vec![f.clone()]);
    let masses: Vec<f64> = s.nuclei.iter().map(|n| n.mass).collect();
    let mut shape = vec![(CO2_BETA_DEG, CO2_R)];
    let (mut t, mut total, mut ts, mut e_int, mut y) = (vec![0.0], vec![r0.energy], vec![r0.stats.energy.ts], vec![r0.stats.energy.internal], vec![bend_y(&s)]);
    let (mut face_max, mut tol_max, mut iters) = (r0.stats.face_density.unwrap_or(f64::NAN), r0.stats.scf_tol_used.unwrap_or(f64::NAN), vec![r0.stats.scf_steps]);
    let (mut flags, mut judged, mut worst) = (0usize, 0usize, (0.0f64, 0usize, 0.0f64, 0.0f64));
    let (mut beta_lo, mut beta_hi, mut r_lo, mut r_hi) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    let t_md = Instant::now();
    for i in 1..=CO2_NVE_STEPS {
        let (f_new, st) = integ.step(&mut s, &f, &mut p, &mut no_augment(), &mut never()).unwrap_or_else(|e| panic!("CO₂⁺ NVE step {i}: {e}"));
        f = f_new;
        let j = judge(&s, &f, &st.eval);
        assert_eq!(j.flagged, st.eval.branch_jump.is_some(), "step {i}: the replica guard disagrees with the provider's ({j:?} vs {:?})", st.eval.branch_jump);
        if j.judged {
            judged += 1;
            if j.mismatch / j.bound > worst.0 { worst = (j.mismatch / j.bound, i, j.mismatch, j.bound); }
        }
        flags += st.eval.branch_jump.is_some() as usize;
        t.push(i as f64 * integ.dt);
        total.push(st.total);
        ts.push(st.eval.energy.ts);
        e_int.push(st.eval.energy.internal);
        y.push(bend_y(&s));
        iters.push(st.eval.scf_steps);
        vels.push(s.nuclei.iter().map(|n| n.vel).collect());
        frcs.push(f.clone());
        face_max = face_max.max(st.eval.face_density.unwrap_or(f64::NAN));
        tol_max = tol_max.max(st.eval.scf_tol_used.unwrap_or(f64::NAN));
        let (beta, ra, rb) = co2_shape(&s);
        shape.push((beta, 0.5 * (ra + rb)));
        (beta_lo, beta_hi) = (beta_lo.min(beta), beta_hi.max(beta));
        (r_lo, r_hi) = (r_lo.min(ra.min(rb)), r_hi.max(ra.max(rb)));
        if i % 50 == 0 {
            println!("  step {i}: F + K {:+.9}, TS {:.4e}, β {beta:.2}°, R {ra:.4}/{rb:.4}, {} SCF steps, face {:.2e}", st.total, st.eval.energy.ts, st.eval.scf_steps,
                     st.eval.face_density.unwrap_or(f64::NAN));
        }
    }
    let md_wall = t_md.elapsed();
    let (slope_au, fluct, mean) = fit_line(&t, &total);
    let drift = slope_au * au_per_ps();
    let (e_lo, e_hi) = total.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let (ts_lo, ts_hi) = ts.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    // E + K = (F + K) + TS: the quantity that is *not* conserved, for contrast
    let ek: Vec<f64> = total.iter().zip(&ts).map(|(a, b)| a + b).collect();
    let (_, ek_fluct, _) = fit_line(&t, &ek);
    // the bend period from every zero crossing of the signed bend coordinate (half a period apart)
    let mut crossings = Vec::new();
    for k in 1..y.len() {
        if (y[k - 1] < 0.0) != (y[k] < 0.0) { crossings.push(t[k - 1] + (t[k] - t[k - 1]) * y[k - 1] / (y[k - 1] - y[k])); }
    }
    assert!(crossings.len() >= 2, "the bend crossed linear only {} times", crossings.len());
    let period = 2.0 * (crossings[crossings.len() - 1] - crossings[0]) / (crossings.len() - 1) as f64;
    // Diagnostic (reported, never asserted): velocity Verlet conserves the shadow Hamiltonian
    //   H̃ = H + dt²[(1/12)·vᵀV″v − (1/24)·Σ_I |F_I|²/m_I] + O(dt⁴)
    // (Hairer, Lubich & Wanner, Geometric Numerical Integration, 2nd ed., §IX.8; up to a multiple of H, which is
    // conserved to the same order), not H = F + K, which swings about it by h²/4 of each mode's energy. With
    // dF/dt = −V″v, vᵀV″v ≈ −v_n·(F_{n+1} − F_{n−1})/(2dt). What H̃ still does — its drift and its detrended peak to
    // peak — is the force and energy error of the solves, the Verlet swing removed to O(dt⁴).
    let dt = integ.dt;
    let shadow: Vec<f64> = (1..total.len() - 1).map(|n| {
        let vf: f64 = (0..masses.len()).flat_map(|i| (0..3).map(move |k| (i, k))).map(|(i, k)| vels[n][i][k] * (frcs[n + 1][i][k] - frcs[n - 1][i][k])).sum();
        let ff: f64 = (0..masses.len()).map(|i| frcs[n][i].iter().map(|c| c * c).sum::<f64>() / masses[i]).sum();
        total[n] + dt * dt * (-vf / (2.0 * dt) / 12.0 - ff / 24.0)
    }).collect();
    let (sh_slope, sh_fluct, _) = fit_line(&t[1..t.len() - 1], &shadow);
    let mut series = String::from("  step: F + K − start, H̃ − H̃₁ (µHa); TS (mHa); β (°); R (Bohr)
");
    for n in (1..total.len() - 1).step_by(20) {
        let _ = writeln!(series, "    {n:4}: {:+8.2} {:+8.2}  {:.3}  {:6.2}  {:.4}", (total[n] - total[0]) * 1e6, (shadow[n - 1] - shadow[0]) * 1e6, ts[n] * 1e3, shape[n].0, shape[n].1);
    }
    let periods = t[t.len() - 1] / period;
    let log = log.lock().unwrap().clone();
    let mut out = String::new();
    let _ = writeln!(out, "CO₂⁺ FD NVE, {CO2_N}³ × {CO2_H}, `Accepted`, charge +1, tol asked {CO2_TOL:.0e} (stopped at ≤ {tol_max:.0e}), dt = {DT_FS} fs, {CO2_NVE_STEPS} steps from β = {CO2_BETA_DEG}°, R = {CO2_R} at rest");
    let _ = writeln!(out, "  F + K: start {:+.9}, mean {mean:+.9}, end {:+.9}; drift {drift:+.3e} Ha/ps (bound {DRIFT_BOUND:.0e})", total[0], total[total.len() - 1]);
    let _ = writeln!(out, "  fluctuation: detrended peak-to-peak {fluct:.3e} Ha (pre-registered bound {fluct_bound:.3e} = {FLUCT_MARGIN} × (h_b²/4·E_mode {verlet_swing:.3e} + 2 × {CO2_PATH_SCATTER:.1e}), h_b = {h_b:.5}); raw {:.3e}; E + K for contrast: detrended {ek_fluct:.3e}",
                     e_hi - e_lo);
    out += &series;
    let _ = writeln!(out, "  shadow Hamiltonian H̃ (diagnostic, Verlet swing removed to O(dt⁴)): drift {:+.3e} Ha/ps, detrended peak-to-peak {sh_fluct:.3e} Ha", sh_slope * au_per_ps());
    let _ = writeln!(out, "  TS ∈ [{ts_lo:.6e}, {ts_hi:.6e}] (swing {:.3e}, needs ≥ {TS_SWING_MIN:.0e}); E ∈ [{:+.9}, {:+.9}]", ts_hi - ts_lo,
                     e_int.iter().fold(f64::INFINITY, |m, &v| m.min(v)), e_int.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v)));
    let _ = writeln!(out, "  bend: β ∈ [{beta_lo:.2}°, {beta_hi:.2}°], R ∈ [{r_lo:.4}, {r_hi:.4}]; {} linear crossings, period {:.0} a.u. = {:.1} fs = {:.0} steps ⇒ {periods:.2} periods",
                     crossings.len(), period, units::au_time_to_fs(period), period / integ.dt);
    let _ = writeln!(out, "  face density max {face_max:.2e} (rule {face_rule:.2e}); SCF steps per solve {}–{} (mean {:.1})", iters.iter().min().unwrap(), iters.iter().max().unwrap(),
                     iters.iter().sum::<usize>() as f64 / iters.len() as f64);
    let _ = writeln!(out, "  branch_jump: {flags} flagged of {judged} judged; largest m/b {:.3} at step {} (m {:.3e}, b {:.3e})", worst.0, worst.1, worst.2, worst.3);
    let _ = writeln!(out, "  {}; MD wall {:.1} s ({:.0} ms/step); test wall {:.1} s", log.summary(), md_wall.as_secs_f64(), ms(md_wall) / CO2_NVE_STEPS as f64, t_test.elapsed().as_secs_f64());
    println!("{out}");
    let result = format!("drift {drift:+.2e} Ha/ps, fluct {fluct:.2e} Ha (bound {fluct_bound:.1e}), TS swing {:.2e}, {flags} branch_jump", ts_hi - ts_lo);
    record(&ctx, timing_row("fd_nve_co2_plus", "CO₂⁺", CO2_N, CO2_TOL, &log, t_test.elapsed(), &result));
    assert!(face_max <= face_rule, "face density {face_max:.3e} above the box rule {face_rule:.3e}");
    assert!(periods >= 2.0, "test design: {CO2_NVE_STEPS} steps cover {periods:.2} bend periods (< 2)");
    assert!(ts_hi - ts_lo >= TS_SWING_MIN, "TS swing {:.3e} Ha < {TS_SWING_MIN:.0e}: the trajectory did not cross the TS swing", ts_hi - ts_lo);
    assert!(drift.abs() <= DRIFT_BOUND, "F + K drift {drift:+.3e} Ha/ps (bound {DRIFT_BOUND:.0e})");
    assert!(fluct <= fluct_bound, "F + K fluctuation {fluct:.3e} Ha (pre-registered bound {fluct_bound:.3e})");
}
