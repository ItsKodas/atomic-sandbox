//! The surrogate force provider against its own teacher (M5 Task 7; numbers of record in
//! `references.md`, section "Surrogate (M5)"). Run with
//! `cargo test -p validate --release --test surrogate` (release-only; skips cleanly without a GPU
//! adapter; `--nocapture` prints every number; each test appends its timing row to the gitignored
//! `last-run-surrogate.md`). Wall times are reported, never asserted (ruling R3: the viewer may
//! share the GPU). `SURROGATE_SEED` overrides the sampling/held-out seed (reported runs only).
//!
//! Teacher: `GpuKs` at σ = h = 0.25, 49³, tol 1e-6 — the M4 dynamics suite's discrete problem,
//! so the M4 pins (R_e = 1.789 Bohr, k = 0.200 Ha/Bohr², ω_MD = 3237 cm⁻¹) are the same surface
//! this suite's models learn, and H₃⁺ at the same 49³ (the budget allows it; at 33³ the box
//! confinement adds a geometry-dependent energy that the model would have to learn as physics).
//! Every test builds its own `SurrogateKs` over its own locked teacher (the passes are seeded and
//! the GPU is bit-reproducible, so the three H₂ models are the same model) and nothing in it is
//! per element beyond Z (the species label) and the isotope mass.
//!
//! What a bound here can mean. The surrogate is a linear model E = Σ_i E_atom(Z_i) + Σ_k c_k B_k
//! on ACE-style invariants (Drautz, PRB 99, 014104 (2019)) fitted by Bayesian ridge regression
//! with evidence-chosen hyperparameters (Bishop, PRML §3.3–3.5; MacKay, Neural Comput. 4, 415
//! (1992)); its reported uncertainty is the posterior predictive std — a *model* uncertainty
//! relative to the teacher, not a physics one (spec §7, honesty gap). The teacher itself has a
//! floor: the egg-box ripple of the real-space grid and the SCF path scatter (6e-7 Ha per H₂
//! solve, M3; measured for H₃⁺ below), so "model vs fresh DFT" differences are
//! √(model error² + teacher floor²) and the tiers are judged with that in mind.
//!
//! # Tests and their bounds
//! - `surrogate_reproduces_the_h2_frequency`: one sampling pass on H₂ at R = 1.6 (the brief's
//!   scene: 12-point scan 0.7–1.8 × 1.6 = 1.12–2.88 Bohr, 60 Gaussian perturbations of 0.15 Bohr,
//!   200-step burst at 1000 K) → refit → (i) E and F at 20 *fresh* on-axis scan points 1.325 +
//!   0.05k Bohr (never a training scan point: 50j − 160k = −205 has no integer solution) against
//!   fresh DFT: rms |ΔE| ≤ `E_TIER_H2` = 1e-5 Ha, rms per-component |ΔF| ≤ `F_TIER_H2` = 2e-4
//!   Ha/Bohr (spec §7 tier on the rms, as in the H₃⁺ fixture of Task 6); the spec's per-point
//!   "within" is asserted on the energy as well, max |ΔE| ≤ `E_TIER_H2` (achieved 5.2e-6,
//!   deterministic per adapter), and the force max is reported (1.27e-4 at the sparse end of
//!   the scan, 63 % of the tier — a pin on this adapter, not asserted); (ii) surrogate-MD from the M4 start (DFT R_e + 0.05 Bohr at rest, dt 0.25 fs,
//!   4000 steps = 1 ps, every step answered by the model — asserted) → ω from the upward zero
//!   crossings against the M4 pinned DFT-MD ω_MD = 1.474855e-2 a.u. = 3236.9 cm⁻¹
//!   (`references.md`, "Dynamics (M4)", 400 steps / 9 crossings; the 2000-step value differs by
//!   4e-6): |ω_sur/ω_MD − 1| ≤ `OMEGA_BOUND`. The spec's 0.5 % is a statement about the
//!   model's *curvature*: ω ∝ √k, so it is a 1 % bound on k(R_e) of the model against the DFT
//!   surface (Verlet's +h²/24 and the anharmonic shift are the same in both runs — same start,
//!   same dt — and cancel exactly); a force error that is a constant offset does not move ω, one
//!   with a slope dΔF/dR = δk does — at the force tier (2e-4) over the 0.1 Bohr the oscillation
//!   spans, a worst-case slope would be 2e-3 Ha/Bohr² = 1 % of k: the force tier implies the
//!   spec's 0.5 % (the tightened 1e-3 below is a fit-noise statement instead, so a model exactly
//!   at the force tier with a slope could pass F and fail ω). The fit averages the teacher's
//!   force scatter (s_F = 4.5e-5) over ~270 samples spread over ~0.15 Bohr rms, so the expected
//!   δk/k is s_F/(k σ_R √n) ≈ 1e-4 and δω/ω ≈ 5e-5. **Achieved −7.0e-6 (k_model/k_DFT − 1 =
//!   −2.2e-5), so the bound is tightened from 5e-3 to 1e-3: 20× the modelled scatter, 5× tighter
//!   than the spec**, never loosened from here (rule 6). The decomposition is reported: ω_sur
//!   against the model's own 3-point ω_static (−5.5e-4 = M4's −5.6e-4: the same Verlet +
//!   anharmonic + truncation terms on the same curve) and k_model/k_DFT − 1.
//!   The sampling pass visits stretched and hot geometries whose density reaches the 49³ box's
//!   faces at the 1e-5 threshold (the teacher's "box too small" note at 1.0–1.4e-5 on a few of
//!   its 273 solves — reported, not asserted: those samples are the teacher's answer for that
//!   box, model and DFT are compared on the same surface); the statics and the 20 fresh points
//!   near equilibrium must carry none (asserted, as in M4).
//! - `surrogate_nve_is_tight`: 10 ps (40 000 steps) of velocity Verlet on the surrogate from its
//!   own R_e + 0.05 at rest. The model surface is smooth and deterministic (no SCF, no f32
//!   path), so Verlet's energy error is the pure shadow-Hamiltonian oscillation: for a harmonic
//!   mode the computed E = K + V satisfies E ∈ [Ẽ, Ẽ/(1 − h²/4)] with Ẽ the Verlet invariant
//!   (`sim::integrator` module doc; the invariant is ½mv² + ½kx²(1 − h²/4), so E − Ẽ =
//!   (h²/4)·½kx²), and for a start from rest at x = Δ, Ẽ = E_mode(1 − h²/4), hence the peak-to-
//!   peak swing is **exactly h²/4·E_mode** with h = ω_static·dt and E_mode = E(R_e+Δ) − E(R_e)
//!   on the same (model) surface. Bound: the detrended peak-to-peak within `FLUCT_TOL` = 10 % of
//!   h²/4·E_mode (controller) — what remains is anharmonicity at Δ = 0.05 (a Morse well of this
//!   k and D ≈ 0.09 changes the O(h²) swing by a few %) and the 3-point truncation of k
//!   (E⁗Δ²/12 ≈ 0.1 %); achieved ratio 1.032 — the size and sign anharmonicity predicts: with
//!   k = 0.20 and D ≈ 0.09 Ha the Morse α = √(k/2D) ≈ 1.05 Bohr⁻¹, so |F|² at the compressed
//!   turning point exceeds the stretched one by ~2·1.5αΔ ≈ 16 % and the |F|²-weighted
//!   turning-point term of the Verlet error grows by a few %. **What bounds a force–energy
//!   mismatch of the model in this test is the swing ratio, not the drift**: 10 % of
//!   h²/4·E_mode = 1.4e-7 Ha over Δ = 0.05 Bohr is |δF| ≲ 3e-6 Ha/Bohr (δk/k ≲ 5e-4). Drift:
//!   |slope of E_total| ≤ `DRIFT_BOUND`. This run is on-axis from rest — 1-D motion under a
//!   central force — so ANY mismatch δF(R) between the model's forces and −dE_model/dR is the
//!   gradient of some Φ(R): E_total = [K + Φ]_shadow + [E_model − Φ](R), a conserved part plus
//!   a periodic function of R, never a secular term (a non-conservative mismatch needs a 3-D
//!   trajectory to pump energy; the rotating active-learning start is one). The drift here
//!   therefore measures only f64 rounding plus the least-squares slope that the 1.4e-6 Ha
//!   Verlet oscillation leaks into a line over a non-integer number of periods,
//!   ≤ A/(T·N_periods) ≈ 1.4e-10 Ha/ps; achieved −2.6e-11; **bound 1e-8 Ha/ps** (spec 1e-6),
//!   70× the leakage floor — a rounding/bookkeeping statement about the integrator on an
//!   analytic surface, and nothing more is claimed for it (rule 7).
//!   Rewind: 1000 steps, flip, 1000, flip → max|Δr| ≤ `REWIND_BOUND`. The model is bitwise
//!   reproducible, so only rounding accumulates: √2000 × ε × R ≈ 5e-15 Bohr (achieved 1.5e-14;
//!   f64 Verlet in `sim::integrator` retraces to 1e-10 over 200 steps on its harmonic fixture);
//!   **bound 1e-12**, 100× the rounding walk and far below the GPU figure 5.8e-6 of M4, where
//!   the backward forces are re-solved. Steps per second reported (spec: ≥ 10⁴ expected for H₂;
//!   achieved 1.1e5).
//! - `surrogate_knows_what_it_does_not_know`: with the H₂ model, (a) R = 4.0 Bohr (the
//!   training scan ends at 2.88) → `Fallback` with the model consulted (`uncertainty` present:
//!   rules 5–7 of the policy, never the gates 1–4) and the std and the rule that fired printed;
//!   the rule the policy applies is **fallback by uncertainty**: an unseen *species* or a
//!   different charge/grid is refused unconditionally (rules 2–3), while an unseen *size* or
//!   *geometry* with known species is allowed only when the model's own posterior and its
//!   training envelope say so (rules 5–7). (b) charge +1 at the same positions → `Fallback`
//!   unconditionally (rule 3: another Kohn-Sham surface; the sample is not kept — asserted).
//!   (c) a third H (neutral H₃, side 1.8 Bohr) → `Fallback` with `uncertainty` present (species
//!   seen, so the model was consulted and its variance/envelope refused: the body-3 features of
//!   an atom with two neighbours lie outside everything the dimer taught it). (d) After the far
//!   fallback plus a walk of fallback steps along the dissociation path (R = 3.6, 3.8, 4.0, 4.2,
//!   4.4 — what a user dragging the molecule apart produces, each kept as a sample) and one
//!   refit, R = 4.0 → `Surrogate` with |E_model − E_DFT| ≤ `OOD_E_BOUND` = 1e-4 Ha (a cold DFT
//!   reference) and H₂ at 1.8 still `Surrogate`. The H₃ probe (c) comes *after* that check, and
//!   its own refit is reported, not asserted: the first run had it before the walk, and the one
//!   neutral-H₃ sample (a 3-electron radical the dimer model is 2.4 Ha wrong about) made the
//!   evidence call the forces noisy (s_F 4.5e-5 → 2.4e-3) and the S8 trust rule refuse the whole
//!   model ("max training force residual 3.2e-2 > 4.2e-3") — the designed answer to one sample
//!   of a new bonding situation (a sampling pass on H₃ is what earns it back), and the reason
//!   the order matters.
//! - `h3plus_tier`: H₃⁺ (charge +1, 2 electrons, closed shell) relaxed on the teacher surface
//!   (quenched Verlet, M4's helper: side ≈ 2.12 Bohr), the teacher's egg-box floor measured in
//!   place (8 rigid sub-grid shifts of ±h/4 per axis, cold solves: rms of E and of the force
//!   components about their means), one sampling pass (default plan: 60 perturbations, 3 × 12
//!   scan points, 200-step 2000 K burst) → body-3 model (Q′ = 29) → (i) 20 held-out geometries
//!   (Gaussian 0.15 Bohr per coordinate, a seed the pass never used) against fresh DFT: rms
//!   per-component |ΔF| ≤ `F_TIER_H3` = 2e-3 Ha/Bohr; energy tier `E_TIER_H3` = 1e-4 Ha judged
//!   as √(rms² − floor²) (controller ruling: the floor is the teacher's own scatter, which the
//!   held-out DFT values carry and no model can remove; the raw rms is reported against the tier
//!   as well). **Measured at 49³ the floor is 3.1e-7 Ha / 2.2e-6 Ha/Bohr** — 500× below Task 6's
//!   33³ figure (1.67e-4 / 2.53e-4), which was therefore the 8 Bohr box's confinement changing
//!   with the molecule's distance to the faces, not the grid; at 49³ the correction is inert and
//!   the raw rms is the number. Achieved rms 3.6e-6 Ha (max 1.3e-5) and 8.9e-6 Ha/Bohr (max
//!   3.9e-5) at a fit noise of s_E 5e-6/atom, s_F 2e-5, so **the tiers are tightened to 3e-5 Ha
//!   (8× the rms) and 2e-4 Ha/Bohr (22× the rms, 5× the max)**: the held-out draws are seeded,
//!   a different seed moves a 20-point rms by ~20 %, and the GPU teacher is bit-reproducible per
//!   adapter. (ii) the symmetric stretch (A₁′: all radii × 1.03, at rest; the trajectory stays
//!   equilateral by symmetry) and the bend (one E′ component: atom 0 radially out by 2δ, atoms
//!   1–2 in by δ, δ = 1.5 % of the radius, recentred; the C₂ᵥ subspace is preserved, so Q =
//!   (2d₀ − d₁ − d₂)/√6 oscillates at ω_bend with the breathing admixture anharmonic coupling
//!   adds — identical in both runs) from the same starts through the teacher (DFT-MD) and
//!   through the surrogate, 320 steps = 80 fs each (≥ 4 upward crossings of the slower mode:
//!   measured 2509 and 1960 cm⁻¹ on this σ = 0.25 surface — experiment: ν₁ 3178, ν₂ 2521 cm⁻¹,
//!   the same ≈ 0.78 softening as H₂'s 3237/4401, reported under rule 7), |ω_sur/ω_DFT − 1| ≤
//!   `FREQ_BOUND`: spec 2 % → **5e-3** (achieved −3.0e-4 / −4.5e-4; a period from 4–6 crossings
//!   located by linear interpolation at 44–68 samples per period is good to ~1e-3, the bound 5×
//!   that).
//! - `active_learning_run_converges`: no model, no pass. H₂ at the M4 R_e + 0.05 along a
//!   generic (non-grid) axis, rotating with k_B·300 K of kinetic energy about its centre (the
//!   descriptors are rotation invariant, so the rotation adds no new features — it makes the
//!   egg-box ripple a *noise* the fit must classify as such, the realistic case). A references-
//!   only pass first (every count 0: the one job is the isolated H atom by ruling S9 — no fit
//!   exists without E_atom(H), and this is how the worker's pass computes it), then 300 steps of
//!   velocity Verlet through `SurrogateKs` with the GPU teacher, `audit_every` = 50 (S7), and
//!   `refit()` whenever `pending_refit()`: the first steps are rule-1 fallbacks (no model), the
//!   model earns trust once n ≥ Q′ = 29 samples with ≥ 30 residual degrees of freedom and
//!   training residuals inside the policy's thresholds (~35–45 steps), then answers; audits
//!   every 50 answered steps are the only teacher calls left. Asserted: fallback fraction over
//!   the last `AL_WINDOW` = 100 steps < `AL_FALLBACK_BOUND` (controller 10 % → 5 %: achieved
//!   2 %, the audit cadence itself), |teacher_gap| on the last fallback < `AL_GAP_BOUND`
//!   (controller 1e-4 → 3e-5 Ha: 5× the largest of the five audit gaps, an empirical margin —
//!   the gaps sit 3–7× above the N·s_E floor of 1.6e-6; achieved 5.1e-6), every audit gap finite and <
//!   `AUDIT_GAP_BOUND` = 1e-3 Ha (reported beyond that — measured 1.1–5.8e-6), the model
//!   trusted at the end. Measured: 31 rule-1/rule-4 fallbacks, first trusted fit before step 32
//!   (n = 32 ≥ Q′ = 29 with dof_E 31 ≥ 30), then 5 audits; 38 DFT calls in all for 300 steps.
//!   Total DFT calls reported.
//!
//! The GPU work of concurrently running tests is serialised through `common::GPU` (the `Locked`
//! teacher), one evaluation at a time, so the per-solve times in the timing rows are uncontended
//! while the tests interleave; the surrogate's own hot path never takes the lock.
use gpu::GpuContext;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sim::surrogate_ks::TrainingResiduals;
use sim::{FallbackPolicy, ForceProvider, ForceResult, Integrator, Nucleus, Sampler, SamplingPlan, Scene, Source, SurrogateKs};
use std::fmt::Write as _;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use surrogate::{Basis, FitReport, Prediction};

mod common;
use common::{
    au_per_ps, bond, chrono_free_date, fit_line, h2, h3_plus, locate_surface, ms, never, no_augment, period_from_upward_crossings, relax, scene, solve,
    timing_row, triangle, EvalLog, Locked, DISPLACEMENT, DT_FS, H, H3_SIDE0, HARTREE_IN_CM, N_H2, TOL_H2,
};

/// Timing rows for `last-run-surrogate.md`, rewritten whenever a test finishes.
static TIMING: Mutex<Vec<String>> = Mutex::new(Vec::new());

// ---- the M4 pins this suite is measured against (`references.md`, "Dynamics (M4)") ----
/// DFT-MD ω of H₂ at σ = h = 0.25, 49³, tol 1e-6 from R_e + 0.05 at rest, dt 0.25 fs (a.u.;
/// 3236.9 cm⁻¹; 400 steps, 9 upward crossings; over 2000 steps the ratio to ω_static moved by
/// 4e-6). The surrogate's MD from the same start is compared with it.
const OMEGA_MD_M4: f64 = 1.474855e-2;
/// R_e(σ = 0.25, 49³) of the same run, Bohr (sanity: the teacher here is that surface).
const RE_M4: f64 = 1.78894;

// ---- training ----
/// The brief's H₂ scene: the pass scans 0.7–1.8 × this bond length.
const SCAN_R0: f64 = 1.6;
/// The brief's H₂ burst temperature (the plan's default 2000 K is used for H₃⁺).
const H2_BURST_T: f64 = 1000.0;
const SEED: u64 = 7;

fn seed() -> u64 {
    std::env::var("SURROGATE_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(SEED)
}

// ---- H₂ surface reproduction ----
const FRESH_POINTS: usize = 20;
const FRESH_R0: f64 = 1.325;
const FRESH_DR: f64 = 0.05;
/// rms |E_model − E_DFT| over the fresh points, Ha (spec §7 tier).
const E_TIER_H2: f64 = 1e-5;
/// rms per-component |F_model − F_DFT| over the fresh points, Ha/Bohr (spec §7 tier).
const F_TIER_H2: f64 = 2e-4;
const VIB_STEPS: usize = 4000;
/// |ω_sur/ω_MD(M4) − 1|. Spec §7: 0.5 % → **1e-3** (module doc: achieved −7.0e-6; the modelled
/// curvature scatter of the fit is ~5e-5 on ω, the bound 20× that).
const OMEGA_BOUND: f64 = 1e-3;

// ---- NVE ----
const NVE_STEPS: usize = 40_000;
/// |slope of E_total|, Ha/ps. Spec §7: 1e-6 → **1e-8** (module doc: achieved −2.6e-11, 70× the
/// least-squares leakage floor 1.4e-10; a 1-D on-axis run cannot see a force–energy mismatch —
/// the swing ratio bounds that at |δF| ≲ 3e-6 Ha/Bohr).
const DRIFT_BOUND: f64 = 1e-8;
/// |fluctuation/(h²/4·E_mode) − 1| (controller).
const FLUCT_TOL: f64 = 0.10;
const REWIND_STEPS: usize = 1000;
/// max |Δr| after n → flip → n → flip, Bohr. Controller 1e-9 → **1e-12** (module doc: achieved
/// 1.5e-14, the rounding walk √2000·ε·R ≈ 5e-15; the bound is 100× that).
const REWIND_BOUND: f64 = 1e-12;

// ---- out of distribution ----
const R_FAR: f64 = 4.0;
const R_IN: f64 = 1.8;
/// The dissociation walk whose fallback samples teach the far region (Bohr).
const FAR_WALK: [f64; 5] = [3.6, 3.8, 4.0, 4.2, 4.4];
/// |E_model − E_DFT| at R_FAR after the refit, Ha (controller).
const OOD_E_BOUND: f64 = 1e-4;
/// Neutral H₃ (3 electrons, doublet): a third atom of a seen species.
const H3_SIDE_OOD: f64 = 1.8;

// ---- H₃⁺ tier ----
const HELD_OUT: usize = 20;
const HELD_OUT_SIGMA: f64 = 0.15;
/// XORed into the seed for the held-out draws, so no plan seed reproduces them.
const HELD_OUT_SALT: u64 = 0x5eed_0f43;
/// rms |ΔE| after the floor is removed in quadrature, Ha. Spec §7 tier 1e-4 → **3e-5** (module
/// doc: achieved 3.6e-6 rms, max 1.3e-5, at a fit noise of 5e-6/atom; 8× the rms).
const E_TIER_H3: f64 = 3e-5;
/// rms per-component |ΔF|, Ha/Bohr. Spec §7 tier 2e-3 → **2e-4** (achieved 8.9e-6 rms, max
/// 3.9e-5; 22× the rms, 5× the max).
const F_TIER_H3: f64 = 2e-4;
/// Rigid sub-grid shifts of the relaxed molecule for the egg-box floor: the 8 corners of
/// (±h/4)³ — each nucleus sees a different grid phase, the geometry does not change.
const EGGBOX_SHIFT: f64 = H / 4.0;
/// Mode starts: the symmetric stretch scales every radius by 1 + this; the bend moves atom 0
/// out by 2× and atoms 1, 2 in by 1× this fraction of the radius.
const STRETCH_AMPL: f64 = 0.03;
const BEND_AMPL: f64 = 0.015;
const MODE_STEPS: usize = 320;
/// |ω_sur/ω_DFT − 1| for both modes. Spec §7: 2 % → **5e-3** (achieved −3.0e-4 / −4.5e-4; the
/// crossing-count precision of an 80 fs run is ~1e-3 on the slower mode, the bound 5× that).
const FREQ_BOUND: f64 = 5e-3;
/// Surrogate steps for the N = 3 speed figure (cheap).
const SPEED_STEPS_H3: usize = 10_000;

// ---- active learning ----
const AL_STEPS: usize = 300;
const AL_WINDOW: usize = 100;
/// Fallback fraction over the last `AL_WINDOW` steps. Controller < 10 % → **5 %** (achieved
/// 2 %, which is the audit cadence's own floor: 2 audits per 100 answered steps; the bound
/// leaves room for two variance fallbacks on top).
const AL_FALLBACK_BOUND: f64 = 0.05;
/// |teacher_gap| on the last fallback, Ha. Controller 1e-4 → **3e-5**: an empirical margin, not a
/// budget — 5× the largest of the five audit gaps (1.1–5.8e-6; achieved on the last fallback
/// 5.1e-6), which sit 3–7× above the N·s_E floor of 1.6e-6 (the mild misspecification of a
/// 29-column fit over a 0.1 Bohr window with 32–37 samples). If it ever trips: more samples,
/// never a looser bound.
const AL_GAP_BOUND: f64 = 3e-5;
const AUDIT_EVERY: usize = 50;
const AUDIT_GAP_BOUND: f64 = 1e-3;
/// Rotational kinetic energy of the active-learning start, as a temperature (k_B T).
const AL_T_ROT: f64 = 300.0;

fn gpu() -> Option<GpuContext> { common::gpu("surrogate") }

fn policy(audit_every: usize) -> FallbackPolicy {
    FallbackPolicy { audit_every, ..FallbackPolicy::default() }
}

/// A `SurrogateKs` over the suite's locked, logged GPU teacher at tol 1e-6.
fn student(ctx: &GpuContext, audit_every: usize) -> (SurrogateKs, Arc<Mutex<EvalLog>>) {
    let (p, log) = Locked::new(ctx.clone(), TOL_H2);
    (SurrogateKs::new(Box::new(p), policy(audit_every)).with_teacher_tol(TOL_H2), log)
}

fn h2_plan() -> SamplingPlan {
    SamplingPlan { hot_burst_t: H2_BURST_T, seed: seed(), ..SamplingPlan::default() }
}

fn residuals(t: Option<TrainingResiduals>) -> String {
    t.map_or("—".into(), |t| format!("E/atom {:.2e}, F {:.2e} (bound {:.2e})", t.energy_per_atom, t.force, t.force_bound))
}

fn fit_summary(rep: &FitReport, s: &SurrogateKs) -> String {
    format!("n {} (Q′ {}, {} rows), rmse E/atom {:.2e} F {:.2e}, s_E {:.2e}/atom s_F {:.2e}, λ {:.2e}, γ {:.1} (E {:.1}, F {:.1}), dof_E {:.0} dof_F {:.0}, κ {:.1e}, {} hyper rounds{}; training residuals {}; trusted: {}{}",
            rep.n_samples, rep.q_prime, rep.n_rows, rep.rmse_energy_per_atom, rep.rmse_force, rep.sigma_e, rep.sigma_f, rep.lambda, rep.dof_eff, rep.gamma_e, rep.gamma_f,
            rep.residual_dof_e, rep.residual_dof_f, rep.condition_estimate, rep.hyper_iterations, if rep.converged { "" } else { " (not converged)" },
            residuals(s.training_residuals()), s.trusted(), s.untrusted_reason().map_or(String::new(), |r| format!(" ({r})")))
}

/// One sampling pass with the surrogate's own teacher (`Sampler::step_own_teacher`, what
/// `Dynamics` drives), then one refit. Appends the pass summary and the sampler's notes.
fn sample_and_fit(s: &mut SurrogateKs, sc: &Scene, plan: &SamplingPlan, out: &mut String) -> (FitReport, Duration) {
    let t = Instant::now();
    let r_cut = s.basis().map_or(Basis::new(&[]).r_cut, |b| b.r_cut);
    let mut sampler = Sampler::with_cutoff(sc, plan, r_cut).unwrap_or_else(|e| panic!("sampling plan: {e}"));
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
    let _ = writeln!(out, "  basis: n_max {}, l_max {}, r_cut {}, degree cap {}, body3 {}, species {:?}, P {} features, Q′ {}; refit {fit_ms:.0} ms: {}",
                     b.n_max, b.l_max, b.r_cut, b.degree_cap, b.body3, b.species, b.n_features(), b.n_independent_columns(), fit_summary(&rep, s));
    assert_eq!((p.failed, p.skipped), (0, 0), "sampling pass: {p:?}");
    (rep, pass)
}

/// rms and max of |ΔE| and of the per-component |ΔF| over a set of geometries.
#[derive(Default)]
struct ErrStats {
    n: usize,
    se2: f64,
    de_max: f64,
    nf: usize,
    sf2: f64,
    df_max: f64,
}

impl ErrStats {
    fn add(&mut self, de: f64, df: impl Iterator<Item = f64>) {
        self.n += 1;
        self.se2 += de * de;
        self.de_max = self.de_max.max(de.abs());
        for d in df {
            self.nf += 1;
            self.sf2 += d * d;
            self.df_max = self.df_max.max(d.abs());
        }
    }
    fn rmse_e(&self) -> f64 { (self.se2 / self.n.max(1) as f64).sqrt() }
    fn rmse_f(&self) -> f64 { (self.sf2 / self.nf.max(1) as f64).sqrt() }
}

/// The model's prediction (with its stds) and the teacher's answer at one geometry, the teacher
/// called directly (a held-out comparison: nothing is kept as a sample).
fn model_vs_dft(s: &mut SurrogateKs, sc: &Scene, what: &str) -> (Prediction, ForceResult) {
    let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
    let z: Vec<u8> = sc.nuclei.iter().map(|n| n.z).collect();
    let pred = s.model().expect("a fitted model").predict(&pos, &z, true).unwrap_or_else(|e| panic!("{what}: predict: {e}"));
    let dft = solve(s.teacher_mut(), sc, what);
    (pred, dft)
}

fn force_diffs<'a>(a: &'a [[f64; 3]], b: &'a [[f64; 3]]) -> impl Iterator<Item = f64> + 'a {
    a.iter().zip(b).flat_map(|(p, q)| (0..3).map(move |k| p[k] - q[k]))
}

fn max_abs(f: &[[f64; 3]]) -> f64 { f.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs())) }

/// Per-component equivalent of the model's expected ‖δF_j‖ on its worst atom (what the policy
/// compares with its bound), and the epistemic energy std per atom (the aleatoric floor
/// (N s_E)² removed, as rule 7 does).
fn stds(pred: &Prediction, s: &SurrogateKs, n: usize) -> (f64, f64) {
    let f = pred.force_std.as_ref().map_or(f64::NAN, |v| v.iter().fold(0.0f64, |m, x| m.max(*x)) / 3f64.sqrt());
    let m = s.model().expect("model");
    let floor = if m.energy_per_atom { (n as f64 * m.sigma_e).powi(2) } else { m.sigma_e * m.sigma_e };
    let e = pred.energy_std.map_or(f64::NAN, |e| (e * e - floor).max(0.0).sqrt() / n as f64);
    (f, e)
}

/// One MD run through `p`: `coord(scene)` after every step (the start value first), E_total,
/// the time, who answered, and the teacher gaps of the fallback steps.
struct MdRun {
    x: Vec<f64>,
    t: Vec<f64>,
    e_tot: Vec<f64>,
    surrogate: usize,
    fallback: usize,
    dft: usize,
    gaps: Vec<(usize, f64)>,
    wall: Duration,
}

impl MdRun {
    fn steps_per_s(&self) -> f64 { (self.surrogate + self.fallback + self.dft) as f64 / self.wall.as_secs_f64() }
}

fn run_md(p: &mut dyn ForceProvider, sc: &mut Scene, forces: &mut Vec<[f64; 3]>, steps: usize, coord: &dyn Fn(&Scene) -> f64) -> MdRun {
    let integ = Integrator::new_fs(DT_FS);
    let mut run = MdRun { x: vec![coord(sc)], t: Vec::with_capacity(steps), e_tot: Vec::with_capacity(steps), surrogate: 0, fallback: 0, dft: 0, gaps: Vec::new(), wall: Duration::ZERO };
    let t0 = Instant::now();
    for i in 1..=steps {
        let (f_new, st) = integ.step(sc, forces, p, &mut no_augment(), &mut never()).unwrap_or_else(|e| panic!("MD step {i}: {e}"));
        *forces = f_new;
        run.x.push(coord(sc));
        run.t.push(i as f64 * integ.dt);
        run.e_tot.push(st.total);
        match st.eval.source {
            Source::Surrogate => run.surrogate += 1,
            Source::Fallback => { run.fallback += 1; if let Some(g) = st.eval.teacher_gap { run.gaps.push((i, g)); } }
            Source::Dft => run.dft += 1,
        }
    }
    run.wall = t0.elapsed();
    run
}

fn omega_from(run: &MdRun) -> (f64, f64, usize) {
    let (period, crossings) = period_from_upward_crossings(&run.x, units::fs_to_au_time(DT_FS));
    (2.0 * std::f64::consts::PI / period, period, crossings)
}

fn cm(w: f64) -> f64 { w * HARTREE_IN_CM }

/// Appends a row and rewrites `last-run-surrogate.md`.
fn record(ctx: &GpuContext, row: String) {
    let mut rows = TIMING.lock().unwrap_or_else(|e| e.into_inner());
    rows.push(row);
    let adapter = ctx.adapter.get_info();
    let mut md = String::new();
    let _ = writeln!(md, "# Surrogate suite timing (`validate::surrogate`, {} UTC)\n", chrono_free_date());
    let _ = writeln!(md, "Adapter: {} ({:?}, {:?}); release build; teacher GpuKs σ = h = 0.25, 49³, tol 1e-6; dt = {DT_FS} fs. Tests run concurrently with their GPU evaluations serialised, so \"GPU\" is the uncontended sum of this test's teacher solves (sampling, references, held-out points, DFT-MD and fallbacks) and \"wall\" includes waiting for the others and the surrogate's own CPU work. Wall times are reported, never asserted (R3).\n", adapter.name, adapter.backend, adapter.device_type);
    let _ = writeln!(md, "| test | system | grid | tol | teacher solves | GPU s | ms/solve | SCF steps/solve | readbacks/solve | test wall s | result |\n|---|---|---|---|---|---|---|---|---|---|---|");
    for r in rows.iter() { md += r; md.push('\n'); }
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/last-run-surrogate.md");
    // the timing table of `tests/surrogate_perf.rs` (Task 9) lives in the same file: keep it
    if let Some(perf) = std::fs::read_to_string(path).ok().and_then(|old| old.find("\n## Performance").map(|i| old[i..].to_string())) {
        md += &perf;
    }
    std::fs::write(path, &md).unwrap_or_else(|e| panic!("writing {path}: {e}"));
}

fn snapshot(log: &Arc<Mutex<EvalLog>>) -> EvalLog { log.lock().unwrap().clone() }

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test surrogate")]
fn surrogate_reproduces_the_h2_frequency() {
    let Some(ctx) = gpu() else { return };
    let t_test = Instant::now();
    let (mut s, log) = student(&ctx, 0);
    let mut out = String::new();
    let _ = writeln!(out, "H₂ surface reproduction and frequency, {N_H2}³, σ = h = {H}, teacher tol {TOL_H2:.0e}");
    // the DFT statics of this surface (the M4 helper): R_e, k, E_mode and the MD start's forces
    let surf = locate_surface(s.teacher_mut());
    out += &surf.log;
    let statics = snapshot(&log);
    let dft_calls_statics = statics.calls;
    let _ = writeln!(out, "  M4 pins: R_e {RE_M4} (here {:.5}, Δ {:+.1e}), ω_MD {:.6e} a.u. = {:.1} cm⁻¹", surf.r_e, surf.r_e - RE_M4, OMEGA_MD_M4, cm(OMEGA_MD_M4));
    // train
    let (rep, pass_wall) = sample_and_fit(&mut s, &h2(SCAN_R0), &h2_plan(), &mut out);
    let after_pass = snapshot(&log);
    let dft_calls_pass = after_pass.calls - dft_calls_statics;
    assert!(s.trusted(), "the pass did not earn trust: {:?}", s.untrusted_reason());
    // (i) fresh scan points against fresh DFT
    let mut st = ErrStats::default();
    let mut lines = String::new();
    s.discontinuity();
    for k in 0..FRESH_POINTS {
        let r = FRESH_R0 + k as f64 * FRESH_DR;
        let sc = h2(r);
        let (pred, dft) = model_vs_dft(&mut s, &sc, &format!("fresh R = {r}"));
        let de = pred.energy - dft.energy;
        let dfx = pred.forces[0][0] - dft.forces[0][0];
        let (f_std, e_std) = stds(&pred, &s, 2);
        st.add(de, force_diffs(&pred.forces, &dft.forces));
        let _ = writeln!(lines, "    R = {r:.3}: E_DFT {:+.9}, E_model − E_DFT {de:+.2e} Ha; F_x(0) DFT {:+.6e}, model − DFT {dfx:+.2e} Ha/Bohr; model std: F {f_std:.1e}/component, E {e_std:.1e}/atom (epistemic)",
                         dft.energy, dft.forces[0][0]);
    }
    let _ = writeln!(out, "  {FRESH_POINTS} fresh on-axis points R = {FRESH_R0} + {FRESH_DR}k vs fresh DFT: rms |ΔE| {:.2e} Ha (max {:.2e}; tier {E_TIER_H2:.0e}), rms |ΔF| {:.2e} Ha/Bohr per component (max {:.2e}; tier {F_TIER_H2:.0e}); the fit's own s_E {:.2e}/atom, s_F {:.2e}",
                     st.rmse_e(), st.de_max, st.rmse_f(), st.df_max, rep.sigma_e, rep.sigma_f);
    out += &lines;
    // (ii) surrogate-MD from the M4 start, every step answered by the model
    let mut sc = h2(surf.r_e + DISPLACEMENT);
    let r0 = s.evaluate(&sc, &mut never()).unwrap_or_else(|e| panic!("start: {e}"));
    assert_eq!(r0.stats.source, Source::Surrogate, "the start must be answered by the model: {:?}", r0.stats.notes);
    let mut f = r0.forces.clone();
    let dft_before_md = snapshot(&log).calls;
    let run = run_md(&mut s, &mut sc, &mut f, VIB_STEPS, &|sc| bond(sc) - surf.r_e);
    let dft_during_md = snapshot(&log).calls - dft_before_md;
    let (omega_sur, period, crossings) = omega_from(&run);
    let (x_min, x_max) = run.x.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let (slope, fluct, _) = fit_line(&run.t, &run.e_tot);
    // the model's own statics (the bisection and the three solves are model evaluations)
    let dft_before_statics = snapshot(&log).calls;
    let msurf = locate_surface(&mut s);
    assert_eq!(snapshot(&log).calls, dft_before_statics, "the model's statics must not call the teacher");
    let h = msurf.omega * units::fs_to_au_time(DT_FS);
    let omega_verlet = (1.0 - h * h / 2.0).acos() / units::fs_to_au_time(DT_FS);
    let ratio_m4 = omega_sur / OMEGA_MD_M4 - 1.0;
    let _ = writeln!(out, "  surrogate-MD: {VIB_STEPS} steps = {:.0} fs from DFT R_e + {DISPLACEMENT} at rest ({} Surrogate / {} Fallback / {} Dft steps, {dft_during_md} teacher calls); {crossings} upward crossings, period {:.4} fs; R − R_e ∈ [{x_min:+.5}, {x_max:+.5}]; E_total drift {:+.2e} Ha/ps, detrended p-p {:.2e} Ha; {:.0} steps/s ({:.1} µs/step)",
                     VIB_STEPS as f64 * DT_FS, run.surrogate, run.fallback, run.dft, units::au_time_to_fs(period), slope * au_per_ps(), fluct, run.steps_per_s(), 1e6 / run.steps_per_s());
    let _ = writeln!(out, "  ω_sur = {omega_sur:.6e} a.u. = {:.1} cm⁻¹; ω_MD(M4, DFT) = {:.1} cm⁻¹: ω_sur/ω_MD − 1 = {ratio_m4:+.3e} (bound {OMEGA_BOUND:.0e}); vs this run's DFT ω_static {:.1} cm⁻¹: {:+.3e}",
                     cm(omega_sur), cm(OMEGA_MD_M4), cm(surf.omega), omega_sur / surf.omega - 1.0);
    out += &msurf.log;
    let _ = writeln!(out, "  model statics: R_e^model − R_e^DFT = {:+.2e} Bohr, k_model/k_DFT − 1 = {:+.3e} (E_mode model {:.4e} vs DFT {:.4e}); ω_sur/ω_static^model − 1 = {:+.3e} (Verlet's ω̃/ω − 1 = {:+.3e} at h = {h:.4}; ω_sur/ω̃ − 1 = {:+.3e}, the anharmonic + truncation remainder, M4: −1.53e-3)",
                     msurf.r_e - surf.r_e, msurf.k / surf.k - 1.0, msurf.e_mode, surf.e_mode, omega_sur / msurf.omega - 1.0, omega_verlet / msurf.omega - 1.0, omega_sur / omega_verlet - 1.0);
    let l = snapshot(&log);
    let warned_fresh = l.warned - after_pass.warned;
    let _ = writeln!(out, "  teacher: {} ({dft_calls_statics} statics, {dft_calls_pass} pass, {FRESH_POINTS} fresh; solves with a warning: statics {}, pass {}, fresh {warned_fresh}); pass wall {:.1} s; test wall {:.1} s",
                     l.summary(), statics.warned, after_pass.warned - statics.warned, pass_wall.as_secs_f64(), t_test.elapsed().as_secs_f64());
    println!("{out}");
    let result = format!("rms ΔE {:.1e}, rms ΔF {:.1e}; ω_sur/ω_MD − 1 = {ratio_m4:+.1e}; {:.0} steps/s", st.rmse_e(), st.rmse_f(), run.steps_per_s());
    record(&ctx, timing_row("h2_frequency", "H₂", N_H2, TOL_H2, &l, t_test.elapsed(), &result));
    // the box is right for the equilibrium region (M4); the pass's stretched/hot samples may
    // touch the 1e-5 face-density threshold and are reported above
    assert_eq!((statics.warned, warned_fresh), (0, 0), "warnings near equilibrium: {:?}", l.warnings);
    assert!((surf.r_e - RE_M4).abs() < 2e-3, "not the M4 surface: R_e = {}", surf.r_e);
    assert_eq!((run.fallback, run.dft, dft_during_md), (0, 0, 0), "surrogate-MD must be answered by the model throughout");
    assert!(x_max > 0.9 * DISPLACEMENT && x_min < -0.8 * DISPLACEMENT, "not oscillating about R_e: [{x_min}, {x_max}]");
    assert!(st.rmse_e() <= E_TIER_H2, "energy tier: rms |ΔE| = {:.3e} Ha (tier {E_TIER_H2:.0e}; max {:.2e})", st.rmse_e(), st.de_max);
    // spec §7's per-point "within" on the energy (achieved 5.2e-6; the force max is reported)
    assert!(st.de_max <= E_TIER_H2, "energy tier per point: max |ΔE| = {:.3e} Ha (tier {E_TIER_H2:.0e})", st.de_max);
    assert!(st.rmse_f() <= F_TIER_H2, "force tier: rms |ΔF| = {:.3e} Ha/Bohr (tier {F_TIER_H2:.0e}; max {:.2e})", st.rmse_f(), st.df_max);
    assert!(ratio_m4.abs() <= OMEGA_BOUND, "ω_sur/ω_MD(M4) − 1 = {ratio_m4:+.3e} (bound {OMEGA_BOUND:.0e})");
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test surrogate")]
fn surrogate_nve_is_tight() {
    let Some(ctx) = gpu() else { return };
    let t_test = Instant::now();
    let (mut s, log) = student(&ctx, 0);
    let mut out = String::new();
    let _ = writeln!(out, "H₂ NVE on the surrogate, {N_H2}³ teacher, dt = {DT_FS} fs, {NVE_STEPS} steps = {:.0} ps", NVE_STEPS as f64 * DT_FS / 1000.0);
    let (_rep, pass_wall) = sample_and_fit(&mut s, &h2(SCAN_R0), &h2_plan(), &mut out);
    assert!(s.trusted(), "the pass did not earn trust: {:?}", s.untrusted_reason());
    let dft_after_pass = snapshot(&log).calls;
    // the model's own statics: R_e, k, E_mode (no teacher call — asserted)
    let msurf = locate_surface(&mut s);
    out += &msurf.log;
    let dt = units::fs_to_au_time(DT_FS);
    let h = msurf.omega * dt;
    let expected_swing = h * h / 4.0 * msurf.e_mode;
    // NVE
    let mut sc = h2(msurf.r_e + DISPLACEMENT);
    let mut f = msurf.start_forces.clone();
    let run = run_md(&mut s, &mut sc, &mut f, NVE_STEPS, &|sc| bond(sc) - msurf.r_e);
    let (slope, fluct, e_mean) = fit_line(&run.t, &run.e_tot);
    let drift = slope * au_per_ps();
    let (e_lo, e_hi) = run.e_tot.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let (omega_md, _, crossings) = omega_from(&run);
    let fluct_ratio = fluct / expected_swing;
    // rewind: the same start, 1000 → flip → 1000 → flip
    let integ = Integrator::new_fs(DT_FS);
    let start = h2(msurf.r_e + DISPLACEMENT);
    let mut sc = start.clone();
    let mut f = msurf.start_forces.clone();
    let t_rw = Instant::now();
    for _ in 0..REWIND_STEPS {
        let (f_new, _) = integ.step(&mut sc, &f, &mut s, &mut no_augment(), &mut never()).unwrap_or_else(|e| panic!("forward: {e}"));
        f = f_new;
    }
    let far = (bond(&sc) - bond(&start)).abs();
    Integrator::flip_velocities(&mut sc);
    for _ in 0..REWIND_STEPS {
        let (f_new, _) = integ.step(&mut sc, &f, &mut s, &mut no_augment(), &mut never()).unwrap_or_else(|e| panic!("backward: {e}"));
        f = f_new;
    }
    Integrator::flip_velocities(&mut sc);
    let rewind_wall = t_rw.elapsed();
    let (mut dr, mut dv) = (0.0f64, 0.0f64);
    for (a, b) in sc.nuclei.iter().zip(&start.nuclei) {
        for k in 0..3 {
            dr = dr.max((a.pos[k] - b.pos[k]).abs());
            dv = dv.max((a.vel[k] - b.vel[k]).abs());
        }
    }
    let dft_after = snapshot(&log).calls;
    let _ = writeln!(out, "  NVE: {} Surrogate / {} Fallback / {} Dft steps; {crossings} upward crossings (ω_MD/ω_static^model − 1 = {:+.3e}); E_total mean {e_mean:+.9}, start {:+.9}, end {:+.9}, raw p-p {:.3e}",
                     run.surrogate, run.fallback, run.dft, omega_md / msurf.omega - 1.0, run.e_tot[0], run.e_tot[NVE_STEPS - 1], e_hi - e_lo);
    let _ = writeln!(out, "  drift {drift:+.3e} Ha/ps (bound {DRIFT_BOUND:.0e}); fluctuation: detrended p-p {fluct:.4e} Ha vs Verlet's h²/4·E_mode = {expected_swing:.4e} (h = {h:.4}, E_mode {:.4e}): ratio {fluct_ratio:.4} (bound 1 ± {FLUCT_TOL})", msurf.e_mode);
    let _ = writeln!(out, "  rewind: {REWIND_STEPS} → flip → {REWIND_STEPS} → flip: |R − R₀| at the turn {far:.4} Bohr; max |Δr| {dr:.2e} Bohr (bound {REWIND_BOUND:.0e}), max |Δv| {dv:.2e} ({:.2e} Bohr/step); {:.0} steps/s", dv * dt, 2.0 * REWIND_STEPS as f64 / rewind_wall.as_secs_f64());
    let _ = writeln!(out, "  speed: {:.0} surrogate steps/s ({:.1} µs/step, N = 2) over the NVE run; teacher calls after the pass: {} (statics, NVE, rewind)", run.steps_per_s(), 1e6 / run.steps_per_s(), dft_after - dft_after_pass);
    let l = snapshot(&log);
    let _ = writeln!(out, "  teacher: {}; pass wall {:.1} s; test wall {:.1} s", l.summary(), pass_wall.as_secs_f64(), t_test.elapsed().as_secs_f64());
    println!("{out}");
    let result = format!("drift {drift:+.1e} Ha/ps, swing ratio {fluct_ratio:.3}, rewind {dr:.1e} Bohr, {:.0} steps/s", run.steps_per_s());
    record(&ctx, timing_row("nve", "H₂", N_H2, TOL_H2, &l, t_test.elapsed(), &result));
    assert_eq!(dft_after, dft_after_pass, "the model must answer every evaluation after the pass");
    assert_eq!((run.fallback, run.dft), (0, 0));
    assert!(far > 0.02, "test assumption: the molecule moved ({far})");
    assert!(drift.abs() <= DRIFT_BOUND, "drift {drift:+.3e} Ha/ps (bound {DRIFT_BOUND:.0e})");
    assert!((fluct_ratio - 1.0).abs() <= FLUCT_TOL, "fluctuation {fluct:.3e} vs h²/4·E_mode {expected_swing:.3e}: ratio {fluct_ratio:.4}");
    assert!(dr <= REWIND_BOUND, "rewind missed the start by {dr:.2e} Bohr (bound {REWIND_BOUND:.0e})");
}

/// Neutral H₃: an equilateral triangle in the xy-plane, centred (3 electrons, doublet).
fn h3_neutral(side: f64) -> Scene {
    let mut sc = h3_plus(side);
    sc.charge = 0;
    sc
}

fn which_rule(notes: &[String]) -> &'static str {
    let n = notes.join(" | ");
    if n.contains("envelope") { "rule 5 (envelope)" }
    else if n.contains("force std") { "rule 6 (force std)" }
    else if n.contains("energy std") { "rule 7 (energy std)" }
    else if n.contains("differs from the model") { "rule 3 (other settings)" }
    else if n.contains("species") { "rule 2 (species)" }
    else if n.contains("audit") { "rule 8 (audit)" }
    else if n.contains("untrusted") { "rule 4 (untrusted)" }
    else if n.contains("no model") { "rule 1 (no model)" }
    else { "?" }
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test surrogate")]
fn surrogate_knows_what_it_does_not_know() {
    let Some(ctx) = gpu() else { return };
    let t_test = Instant::now();
    let (mut s, log) = student(&ctx, 0);
    let mut out = String::new();
    let _ = writeln!(out, "H₂ model out of distribution, {N_H2}³ teacher");
    let (_rep, pass_wall) = sample_and_fit(&mut s, &h2(SCAN_R0), &h2_plan(), &mut out);
    assert!(s.trusted(), "the pass did not earn trust: {:?}", s.untrusted_reason());
    let n0 = s.samples();
    let probe = |s: &mut SurrogateKs, sc: &Scene, what: &str, out: &mut String| -> ForceResult {
        let r = s.evaluate(sc, &mut never()).unwrap_or_else(|e| panic!("{what}: {e}"));
        let (f_std, e_std) = s.model().map_or((f64::NAN, f64::NAN), |m| {
            let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
            let z: Vec<u8> = sc.nuclei.iter().map(|n| n.z).collect();
            m.predict(&pos, &z, true).map_or((f64::NAN, f64::NAN), |p| stds(&p, s, sc.nuclei.len()))
        });
        let fmax = max_abs(&r.forces);
        let _ = writeln!(out, "  {what}: {:?} by {}; uncertainty {:?} Ha/Bohr per component; model std at this geometry: F {f_std:.2e} (policy bound {:.1e} for |F| ≤ {fmax:.1e}), E {e_std:.2e}/atom epistemic (bound {:.0e}); teacher − model {:?}; samples {}; notes {:?}",
                         r.stats.source, which_rule(&r.stats.notes), r.stats.uncertainty, s.policy().force_bound(fmax), s.policy().energy_std_per_atom, r.stats.teacher_gap, s.samples(), r.stats.notes);
        r
    };
    // in range: answered
    let r_in = probe(&mut s, &h2(R_IN), &format!("H₂ R = {R_IN}"), &mut out);
    assert_eq!(r_in.stats.source, Source::Surrogate);
    // (a) far: the model consulted and refused
    s.discontinuity();
    let r_far = probe(&mut s, &h2(R_FAR), &format!("(a) H₂ R = {R_FAR}"), &mut out);
    assert_eq!(r_far.stats.source, Source::Fallback, "{:?}", r_far.stats.notes);
    assert!(r_far.stats.uncertainty.is_some(), "the model must have been consulted (rules 5–7), not gated: {:?}", r_far.stats.notes);
    assert_eq!(s.samples(), n0 + 1, "the fallback sample is kept");
    // the brief's "because the variance says so": the envelope fires first by design, and the
    // model's own force std at R_FAR exceeds the policy bound as well (1.26e-2 vs 4.6e-3)
    {
        let far = h2(R_FAR);
        let pos: Vec<[f64; 3]> = far.nuclei.iter().map(|n| n.pos).collect();
        let z: Vec<u8> = far.nuclei.iter().map(|n| n.z).collect();
        let pred = s.model().expect("model").predict(&pos, &z, true).expect("predict");
        let (f_std, _) = stds(&pred, &s, 2);
        let fmag = pred.forces.iter().map(|f| (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt()).fold(0.0f64, f64::max);
        assert!(f_std > s.policy().force_bound(fmag), "the variance alone would not have refused R = {R_FAR}: force std {f_std:.2e} ≤ bound {:.2e}", s.policy().force_bound(fmag));
    }
    // (b) charge +1: refused unconditionally, not kept
    s.reset();
    let mut ion = h2(R_IN);
    ion.charge = 1;
    let r_ion = probe(&mut s, &ion, "(b) H₂⁺ (charge +1) R = 1.8", &mut out);
    assert_eq!(r_ion.stats.source, Source::Fallback);
    assert!(r_ion.stats.notes.iter().any(|n| n.contains("charge 1")), "{:?}", r_ion.stats.notes);
    assert!(r_ion.stats.uncertainty.is_none() && r_ion.stats.teacher_gap.is_none(), "rule 3 never consults the model: {:?}", r_ion.stats);
    assert_eq!(s.samples(), n0 + 1, "a sample from another surface is not kept");
    // (d) the dissociation walk, a refit, and R_FAR again
    let mut walk_fallbacks = 0;
    for &r in &FAR_WALK {
        let res = probe(&mut s, &h2(r), &format!("(d) walk R = {r}"), &mut out);
        walk_fallbacks += (res.stats.source == Source::Fallback) as usize;
    }
    assert!(s.pending_refit());
    let n_before = s.samples();
    let t = Instant::now();
    let rep = s.refit().unwrap_or_else(|e| panic!("refit: {e}"));
    let _ = writeln!(out, "  refit over {n_before} samples ({} new: far + {walk_fallbacks} walk fallbacks) in {:.0} ms: {}", n_before - n0, ms(t.elapsed()), fit_summary(&rep, &s));
    let trusted_after_walk = s.trusted();
    s.reset();
    let dft_far = solve(s.teacher_mut(), &h2(R_FAR), "cold DFT at R_FAR");
    let r_far2 = probe(&mut s, &h2(R_FAR), &format!("(d) H₂ R = {R_FAR} after the refit"), &mut out);
    let de = r_far2.energy - dft_far.energy;
    let df = force_diffs(&r_far2.forces, &dft_far.forces).fold(0.0f64, |m, x| m.max(x.abs()));
    let _ = writeln!(out, "  at R = {R_FAR}: E_model {:+.9} vs cold DFT {:+.9}: ΔE {de:+.2e} Ha (bound {OOD_E_BOUND:.0e}), max |ΔF| {df:.2e} Ha/Bohr; before the walk the teacher − model gap there was {:+.2e}",
                     r_far2.energy, dft_far.energy, r_far.stats.teacher_gap.unwrap_or(f64::NAN));
    let r_in2 = probe(&mut s, &h2(R_IN), &format!("H₂ R = {R_IN} after the refit"), &mut out);
    // (c) a third H of a seen species: consulted, refused by its own uncertainty; then what one
    // such sample does to the fit (reported: the first run showed the S8 rule refusing the model)
    s.reset();
    let n_c = s.samples();
    let r_h3 = probe(&mut s, &h3_neutral(H3_SIDE_OOD), &format!("(c) neutral H₃ side {H3_SIDE_OOD}"), &mut out);
    assert_eq!(r_h3.stats.source, Source::Fallback, "{:?}", r_h3.stats.notes);
    assert!(r_h3.stats.uncertainty.is_some(), "species seen: the model is consulted and its variance/envelope decides: {:?}", r_h3.stats.notes);
    assert_eq!(s.samples(), n_c + 1);
    s.reset();
    let t = Instant::now();
    let rep_c = s.refit().unwrap_or_else(|e| panic!("refit with H₃: {e}"));
    let _ = writeln!(out, "  refit with the one H₃ sample ({} samples) in {:.0} ms: {}", s.samples(), ms(t.elapsed()), fit_summary(&rep_c, &s));
    let r_in3 = probe(&mut s, &h2(R_IN), &format!("H₂ R = {R_IN} after the H₃ refit (reported)"), &mut out);
    let l = snapshot(&log);
    let _ = writeln!(out, "  teacher: {}; pass wall {:.1} s; test wall {:.1} s", l.summary(), pass_wall.as_secs_f64(), t_test.elapsed().as_secs_f64());
    println!("{out}");
    let result = format!("far {:?} ({}), ion Fallback, walk+refit ΔE(4.0) {de:+.1e}; H₃ {:?} ({}), then {} ({:?} at 1.8)", r_far.stats.source, which_rule(&r_far.stats.notes), r_h3.stats.source, which_rule(&r_h3.stats.notes),
                         if s.trusted() { "trusted".to_string() } else { format!("untrusted: {}", s.untrusted_reason().unwrap_or_default()) }, r_in3.stats.source);
    record(&ctx, timing_row("ood", "H₂/H₃", N_H2, TOL_H2, &l, t_test.elapsed(), &result));
    assert!(trusted_after_walk, "the refit after the walk lost trust: {:?}", rep);
    assert_eq!(r_far2.stats.source, Source::Surrogate, "R = {R_FAR} still refused after the walk and the refit: {:?}", r_far2.stats.notes);
    assert!(de.abs() <= OOD_E_BOUND, "|E_model − E_DFT| at R = {R_FAR} = {:.3e} (bound {OOD_E_BOUND:.0e})", de.abs());
    assert_eq!(r_in2.stats.source, Source::Surrogate, "{:?}", r_in2.stats.notes);
}

/// Standard normal by Box–Muller from `rng`.
fn normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-300);
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn centroid(sc: &Scene) -> [f64; 3] {
    let n = sc.nuclei.len() as f64;
    let mut c = [0.0; 3];
    for a in &sc.nuclei { for k in 0..3 { c[k] += a.pos[k] / n; } }
    c
}

/// Distances of the three nuclei from their centroid.
fn radii(sc: &Scene) -> [f64; 3] {
    let c = centroid(sc);
    std::array::from_fn(|i| {
        let p = sc.nuclei[i].pos;
        ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2)).sqrt()
    })
}

/// Scale the radial coordinate of each nucleus about the centroid by `f[i]`, then recentre.
fn displaced(sc: &Scene, f: [f64; 3]) -> Scene {
    let mut d = sc.clone();
    let c = centroid(sc);
    for (i, a) in d.nuclei.iter_mut().enumerate() {
        for k in 0..3 { a.pos[k] = c[k] + f[i] * (a.pos[k] - c[k]); }
        a.vel = [0.0; 3];
    }
    let c2 = centroid(&d);
    for a in &mut d.nuclei { for k in 0..3 { a.pos[k] -= c2[k]; } }
    d
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test surrogate")]
fn h3plus_tier() {
    let Some(ctx) = gpu() else { return };
    let t_test = Instant::now();
    let (mut s, log) = student(&ctx, 0);
    let mut out = String::new();
    let _ = writeln!(out, "H₃⁺ tier, {N_H2}³, σ = h = {H}, teacher tol {TOL_H2:.0e}");
    // relax on the teacher surface
    let (s0, _f0, relax_steps) = relax(s.teacher_mut(), h3_plus(H3_SIDE0), &mut out);
    let rad_e = radii(&s0);
    let _ = writeln!(out, "  relaxed in {relax_steps} steps: sides {} (M4 at tol 1e-5: 2.1174), radii {:.5}/{:.5}/{:.5} Bohr", triangle(&s0), rad_e[0], rad_e[1], rad_e[2]);
    // the teacher's egg-box floor at this geometry: 8 rigid shifts, cold
    let mut es = Vec::new();
    let mut fs: Vec<Vec<[f64; 3]>> = Vec::new();
    for corner in 0..8 {
        let mut sh = s0.clone();
        for a in &mut sh.nuclei {
            for k in 0..3 { a.pos[k] += if corner >> k & 1 == 1 { EGGBOX_SHIFT } else { -EGGBOX_SHIFT }; }
        }
        s.reset();
        let r = solve(s.teacher_mut(), &sh, "egg-box shift");
        es.push(r.energy);
        fs.push(r.forces);
    }
    let e_mean = es.iter().sum::<f64>() / 8.0;
    let floor_e = (es.iter().map(|e| (e - e_mean).powi(2)).sum::<f64>() / 8.0).sqrt();
    let mut f_mean = vec![[0.0; 3]; 3];
    for f in &fs { for (m, g) in f_mean.iter_mut().zip(f) { for k in 0..3 { m[k] += g[k] / 8.0; } } }
    let floor_f = (fs.iter().flat_map(|f| f.iter().zip(&f_mean).flat_map(|(g, m)| (0..3).map(move |k| (g[k] - m[k]).powi(2)))).sum::<f64>() / 72.0).sqrt();
    let _ = writeln!(out, "  egg-box floor (8 rigid shifts of ±{EGGBOX_SHIFT} per axis, cold): E ∈ [{:+.6}, {:+.6}], rms {floor_e:.2e} Ha; force components rms {floor_f:.2e} Ha/Bohr about their means (Task 6 at 33³: 1.67e-4 / 2.53e-4)",
                     es.iter().cloned().fold(f64::INFINITY, f64::min), es.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    let before_pass = snapshot(&log);
    // train (the default plan: 2000 K burst)
    let plan = SamplingPlan { seed: seed(), ..SamplingPlan::default() };
    let (rep, pass_wall) = sample_and_fit(&mut s, &s0, &plan, &mut out);
    let after_pass = snapshot(&log);
    let trusted = s.trusted();
    // (i) held-out geometries
    let mut rng = StdRng::seed_from_u64(seed() ^ HELD_OUT_SALT);
    let mut st = ErrStats::default();
    let mut lines = String::new();
    for i in 0..HELD_OUT {
        let mut sc = s0.clone();
        for a in &mut sc.nuclei { for k in 0..3 { a.pos[k] += HELD_OUT_SIGMA * normal(&mut rng); } }
        s.discontinuity();
        let (pred, dft) = model_vs_dft(&mut s, &sc, &format!("held-out {i}"));
        let de = pred.energy - dft.energy;
        let dfm = force_diffs(&pred.forces, &dft.forces).fold(0.0f64, |m, x| m.max(x.abs()));
        let (f_std, e_std) = stds(&pred, &s, 3);
        st.add(de, force_diffs(&pred.forces, &dft.forces));
        let _ = writeln!(lines, "    {i:2}: sides {} E_DFT {:+.6} ΔE {de:+.2e} Ha, max |ΔF| {dfm:.2e} (|F| ≤ {:.2e}) Ha/Bohr; model std F {f_std:.1e} E {e_std:.1e}/atom",
                         triangle(&sc), dft.energy, max_abs(&dft.forces));
    }
    let after_held_out = snapshot(&log);
    let e_model = (st.rmse_e().powi(2) - floor_e * floor_e).max(0.0).sqrt();
    let _ = writeln!(out, "  {HELD_OUT} held-out (Gaussian {HELD_OUT_SIGMA} Bohr per coordinate): rms |ΔE| {:.2e} Ha raw (max {:.2e}; tier {E_TIER_H3:.0e} {}), floor-corrected √(rms² − {floor_e:.2e}²) = {e_model:.2e}; rms |ΔF| {:.2e} Ha/Bohr per component (max {:.2e}; tier {F_TIER_H3:.0e}); fit s_E {:.2e}/atom s_F {:.2e}",
                     st.rmse_e(), st.de_max, if st.rmse_e() <= E_TIER_H3 { "met raw" } else { "not met raw" }, st.rmse_f(), st.df_max, rep.sigma_e, rep.sigma_f);
    out += &lines;
    // (ii) the two modes: DFT-MD and surrogate-MD from the same starts
    let stretch = displaced(&s0, [1.0 + STRETCH_AMPL; 3]);
    let bend = displaced(&s0, [1.0 + 2.0 * BEND_AMPL, 1.0 - BEND_AMPL, 1.0 - BEND_AMPL]);
    let d_e = (rad_e[0] + rad_e[1] + rad_e[2]) / 3.0;
    let breathing = move |sc: &Scene| { let r = radii(sc); (r[0] + r[1] + r[2]) / 3.0 - d_e };
    let e_prime = |sc: &Scene| { let r = radii(sc); (2.0 * r[0] - r[1] - r[2]) / 6f64.sqrt() };
    let mut freqs = Vec::new();
    for (name, start, coord) in [("symmetric stretch", &stretch, &breathing as &dyn Fn(&Scene) -> f64), ("bend (E′)", &bend, &e_prime)] {
        // DFT-MD
        s.discontinuity();
        let mut sc = start.clone();
        let mut f = solve(s.teacher_mut(), &sc, name).forces;
        let dft_run = run_md(s.teacher_mut(), &mut sc, &mut f, MODE_STEPS, coord);
        let (w_dft, p_dft, c_dft) = omega_from(&dft_run);
        // surrogate-MD
        let mut sc = start.clone();
        let r0 = s.evaluate(&sc, &mut never()).unwrap_or_else(|e| panic!("{name} start: {e}"));
        let mut f = r0.forces.clone();
        let sur_run = run_md(&mut s, &mut sc, &mut f, MODE_STEPS, coord);
        let (w_sur, p_sur, c_sur) = omega_from(&sur_run);
        let (x_lo, x_hi) = dft_run.x.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        let _ = writeln!(out, "  {name}: start {} ({:+.4} Bohr), {MODE_STEPS} steps = {:.0} fs; DFT-MD: {c_dft} crossings, period {:.3} fs, ω {:.1} cm⁻¹ (coordinate ∈ [{x_lo:+.4}, {x_hi:+.4}]); surrogate-MD ({:?} start, {} Surrogate / {} Fallback steps): {c_sur} crossings, period {:.3} fs, ω {:.1} cm⁻¹; ω_sur/ω_DFT − 1 = {:+.3e} (bound {FREQ_BOUND})",
                         triangle(start), coord(start), MODE_STEPS as f64 * DT_FS, units::au_time_to_fs(p_dft), cm(w_dft), r0.stats.source, sur_run.surrogate, sur_run.fallback, units::au_time_to_fs(p_sur), cm(w_sur), w_sur / w_dft - 1.0);
        freqs.push((name, w_sur / w_dft - 1.0, r0.stats.source, sur_run.fallback));
    }
    // speed, N = 3
    let mut sc = stretch.clone();
    let mut f = s.evaluate(&sc, &mut never()).unwrap().forces;
    let speed_run = run_md(&mut s, &mut sc, &mut f, SPEED_STEPS_H3, &|_| 0.0);
    let l = snapshot(&log);
    let warned = (before_pass.warned, after_pass.warned - before_pass.warned, after_held_out.warned - after_pass.warned, l.warned - after_held_out.warned);
    let _ = writeln!(out, "  speed: {:.0} surrogate steps/s ({:.1} µs/step, N = 3, {} fallbacks over {SPEED_STEPS_H3} steps) vs the teacher's {:.0} ms/solve", speed_run.steps_per_s(), 1e6 / speed_run.steps_per_s(), speed_run.fallback, ms(l.gpu) / l.calls.max(1) as f64);
    let _ = writeln!(out, "  teacher: {} (solves with a warning: relax + floor {}, pass {}, held-out {}, DFT-MD {}); pass wall {:.1} s; test wall {:.1} s", l.summary(), warned.0, warned.1, warned.2, warned.3, pass_wall.as_secs_f64(), t_test.elapsed().as_secs_f64());
    println!("{out}");
    let result = format!("rms ΔE {:.1e} (floor {floor_e:.1e} → {e_model:.1e}), rms ΔF {:.1e}; ω ratios {:+.1e} / {:+.1e}", st.rmse_e(), st.rmse_f(), freqs[0].1, freqs[1].1);
    record(&ctx, timing_row("h3plus_tier", "H₃⁺", N_H2, TOL_H2, &l, t_test.elapsed(), &result));
    // near equilibrium the box must be right; the pass's hot/stretched samples and the ±0.15
    // Bohr held-out draws may touch the 1e-5 face-density threshold (reported above)
    assert_eq!((warned.0, warned.3), (0, 0), "warnings near equilibrium: {:?}", l.warnings);
    assert!(trusted, "the pass did not earn trust: {:?}", s.untrusted_reason());
    assert!(st.rmse_f() <= F_TIER_H3, "force tier: rms |ΔF| = {:.3e} Ha/Bohr (tier {F_TIER_H3:.0e}; max {:.2e})", st.rmse_f(), st.df_max);
    assert!(e_model <= E_TIER_H3, "energy tier: √(rms² − floor²) = {e_model:.3e} Ha (raw rms {:.3e}, floor {floor_e:.2e}; tier {E_TIER_H3:.0e})", st.rmse_e());
    for (name, ratio, source, fallbacks) in &freqs {
        assert_eq!((*source, *fallbacks), (Source::Surrogate, 0), "{name}: surrogate-MD must be answered by the model throughout");
        assert!(ratio.abs() <= FREQ_BOUND, "{name}: ω_sur/ω_DFT − 1 = {ratio:+.3e} (bound {FREQ_BOUND})");
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test surrogate")]
fn active_learning_run_converges() {
    let Some(ctx) = gpu() else { return };
    let t_test = Instant::now();
    let (mut s, log) = student(&ctx, AUDIT_EVERY);
    let mut out = String::new();
    // the start: R_e + Δ along a generic axis, rotating with k_B·T_rot about its centre
    let u = [0.6, 0.64, 0.48];
    let w = { let (a, b) = (0.64, -0.6); let n = (a * a + b * b) as f64; [a / n.sqrt(), b / n.sqrt(), 0.0] };
    let r0 = RE_M4 + DISPLACEMENT;
    let mut sc = scene(N_H2, vec![Nucleus::new(1, [r0 / 2.0 * u[0], r0 / 2.0 * u[1], r0 / 2.0 * u[2]]), Nucleus::new(1, [-r0 / 2.0 * u[0], -r0 / 2.0 * u[1], -r0 / 2.0 * u[2]])]);
    let v = (units::kelvin_to_hartree(AL_T_ROT) / sc.nuclei[0].mass).sqrt();
    sc.nuclei[0].vel = [v * w[0], v * w[1], v * w[2]];
    sc.nuclei[1].vel = [-v * w[0], -v * w[1], -v * w[2]];
    let _ = writeln!(out, "H₂ active learning, {N_H2}³ teacher, no model: R₀ = {r0:.5} Bohr along {u:?}, rotating with K = k_B·{AL_T_ROT} K (v = {v:.2e} a.u., period {:.0} fs), dt = {DT_FS} fs, {AL_STEPS} steps, audit every {AUDIT_EVERY} answered steps, refit whenever a sample is pending",
                     units::au_time_to_fs(2.0 * std::f64::consts::PI * (0.5 * r0) / v));
    // the reference energy first (ruling S9, the sampling pass's rule): a references-only pass —
    // no model can be fitted without E_atom(H), and the worker's pass computes it the same way
    let mut refs = Sampler::new(&sc, &SamplingPlan { perturbations: 0, hot_burst_steps: 0, scan_points_per_pair: 0, ..SamplingPlan::default() }).unwrap();
    while refs.step_own_teacher(&mut s, &mut never()).unwrap_or_else(|e| panic!("reference: {e}")).is_some() {}
    s.reset();
    let _ = writeln!(out, "  references-only pass: {} teacher calls, E_ref(H) = {:+.6} Ha, {} samples", snapshot(&log).calls, s.reference_energy(1).unwrap_or(f64::NAN), s.samples());
    let integ = Integrator::new_fs(DT_FS);
    let r_first = s.evaluate(&sc, &mut never()).unwrap_or_else(|e| panic!("start: {e}"));
    assert_eq!((r_first.stats.source, s.samples()), (Source::Fallback, 1), "{:?}", r_first.stats.notes);
    let _ = writeln!(out, "  step 0: {:?} ({})", r_first.stats.source, which_rule(&r_first.stats.notes));
    let mut f = r_first.forces.clone();
    let mut sources = Vec::with_capacity(AL_STEPS);
    let mut last_gap: Option<(usize, f64)> = None;
    let mut last_fallback = 0;
    let mut audits: Vec<(usize, f64, bool)> = Vec::new();
    let mut refits = 0;
    let mut first_trusted: Option<usize> = None;
    let mut refit_log = String::new();
    let mut e_prev: Option<f64> = None;
    let mut jumps = Vec::new();
    let mut untrusted_after: Vec<(usize, String)> = Vec::new();
    let t_md = Instant::now();
    for i in 1..=AL_STEPS {
        if s.pending_refit() {
            let t = Instant::now();
            let rep = s.refit().unwrap_or_else(|e| panic!("refit before step {i}: {e}"));
            refits += 1;
            let tr = s.trusted();
            if tr && first_trusted.is_none() {
                first_trusted = Some(i);
                let _ = writeln!(refit_log, "    first trusted fit before step {i} ({:.0} ms): {}", ms(t.elapsed()), fit_summary(&rep, &s));
            } else if !tr && first_trusted.is_some() {
                untrusted_after.push((i, s.untrusted_reason().unwrap_or_default()));
            }
            if refits <= 3 || refits % 10 == 0 {
                let _ = writeln!(refit_log, "    refit {refits} before step {i}: n {} dof_E {:.0} dof_F {:.0} s_E {:.1e} s_F {:.1e}, residuals {}, trusted {}", rep.n_samples, rep.residual_dof_e, rep.residual_dof_f, rep.sigma_e, rep.sigma_f, residuals(s.training_residuals()), tr);
            }
        }
        let (f_new, st) = integ.step(&mut sc, &f, &mut s, &mut no_augment(), &mut never()).unwrap_or_else(|e| panic!("step {i}: {e}"));
        f = f_new;
        sources.push(st.eval.source);
        if st.eval.source == Source::Fallback {
            last_fallback = i;
            if let Some(g) = st.eval.teacher_gap { last_gap = Some((i, g)); }
            if st.eval.notes.iter().any(|n| n.contains("audit")) {
                let failed = st.eval.notes.iter().any(|n| n.contains("audit failed"));
                audits.push((i, st.eval.teacher_gap.or(st.eval.refit_jump).unwrap_or(f64::NAN), failed));
            }
        }
        if let Some(j) = st.eval.refit_jump { jumps.push((i, j)); }
        if let Some(p) = e_prev { if (st.total - p).abs() > 1e-4 && st.eval.refit_jump.is_none() { let _ = writeln!(refit_log, "    step {i}: E_total moved by {:+.2e} ({:?}, {})", st.total - p, st.eval.source, which_rule(&st.eval.notes)); } }
        e_prev = Some(st.total);
    }
    let md_wall = t_md.elapsed();
    if s.pending_refit() { s.refit().unwrap_or_else(|e| panic!("final refit: {e}")); refits += 1; }
    let l = snapshot(&log);
    let count = |range: std::ops::Range<usize>, src: Source| sources[range].iter().filter(|&&x| x == src).count();
    let n_fb = count(0..AL_STEPS, Source::Fallback);
    let n_fb_last = count(AL_STEPS - AL_WINDOW..AL_STEPS, Source::Fallback);
    let fraction = n_fb_last as f64 / AL_WINDOW as f64;
    let mut timeline = String::new();
    for (k, chunk) in sources.chunks(25).enumerate() {
        let _ = write!(timeline, "{}{}", if k > 0 { " " } else { "" }, chunk.iter().map(|x| match x { Source::Surrogate => 'S', Source::Fallback => 'F', Source::Dft => 'D' }).collect::<String>());
    }
    let _ = writeln!(out, "  steps: {timeline}");
    out += &refit_log;
    let _ = writeln!(out, "  {n_fb} fallbacks over {AL_STEPS} steps ({refits} refits; first trusted before step {:?}; lost trust after {:?}); last {AL_WINDOW} steps: {n_fb_last} fallbacks = {:.1} % (bound {:.0} %); last fallback at step {last_fallback}; last teacher gap {:?} Ha (bound {AL_GAP_BOUND:.0e})",
                     first_trusted, untrusted_after, 100.0 * fraction, 100.0 * AL_FALLBACK_BOUND, last_gap);
    let _ = writeln!(out, "  audits (step, teacher − model, failed): {:?} (reported; bound on |gap| {AUDIT_GAP_BOUND:.0e})", audits.iter().map(|(i, g, f)| format!("({i}, {g:+.2e}, {f})")).collect::<Vec<_>>());
    let _ = writeln!(out, "  refit jumps booked on the first evaluate after each refit (step, E_new − E_reported): {:?}", jumps.iter().map(|(i, j)| format!("({i}, {j:+.2e})")).collect::<Vec<_>>());
    let _ = writeln!(out, "  final model: {}; final bond {:.5} Bohr, T {:.0} K", s.report().map_or("—".into(), |r| fit_summary(r, &s)), bond(&sc), sc.temperature());
    let _ = writeln!(out, "  total DFT calls {} (1 start + {n_fb} fallback steps incl. {} audits); MD wall {:.1} s ({:.0} ms/step incl. refits); teacher: {}; test wall {:.1} s", l.calls, audits.len(), md_wall.as_secs_f64(), ms(md_wall) / AL_STEPS as f64, l.summary(), t_test.elapsed().as_secs_f64());
    println!("{out}");
    let result = format!("{n_fb} fallbacks/{AL_STEPS} ({:.0} % in the last {AL_WINDOW}), last gap {:+.1e}, {} DFT calls", 100.0 * fraction, last_gap.map_or(f64::NAN, |g| g.1), l.calls);
    record(&ctx, timing_row("active_learning", "H₂", N_H2, TOL_H2, &l, t_test.elapsed(), &result));
    assert!(l.warnings.iter().all(|w| !w.contains("box too small")), "{:?}", l.warnings);
    assert!(s.trusted(), "not trusted at the end: {:?}", s.untrusted_reason());
    assert!(fraction < AL_FALLBACK_BOUND, "fallback fraction over the last {AL_WINDOW} steps {fraction:.3} (bound {AL_FALLBACK_BOUND})");
    let (gap_step, gap) = last_gap.expect("a fallback with a trusted model happened");
    assert_eq!(gap_step, last_fallback, "the last fallback (step {last_fallback}) carried no teacher gap: the model was untrusted there");
    assert!(gap.abs() < AL_GAP_BOUND, "|teacher_gap| on the last fallback = {:.3e} (bound {AL_GAP_BOUND:.0e})", gap.abs());
    for (i, g, failed) in &audits {
        assert!(g.is_finite() && g.abs() < AUDIT_GAP_BOUND, "audit at step {i}: gap {g:+.3e} (failed {failed})");
    }
}
