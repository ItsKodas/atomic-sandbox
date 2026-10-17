//! Spin-polarized Kohn-Sham SCF on the GPU — the port of `qm::scf` (read its module doc first:
//! the total-energy functional and why it is evaluated from the *output* density with a fresh
//! Poisson solve, the Pulay/DIIS derivation with the two spin channels stacked and the
//! charge-conservation argument, the electron-count/multiplicity rules, the warm-start rules
//! and the box-size warning). The algorithm is step for step the CPU's:
//!   V_eff,σ = V_nuc + V_H[ρ_in] + v_xc,σ[ρ_in↑, ρ_in↓]  →  lowest n_σ eigenpairs of H_σ per spin
//!   →  ρ_out,σ = Σ_i ψ_{iσ}²,  Δρ = ∫(|ρ_out↑ − ρ_in↑| + |ρ_out↓ − ρ_in↓|)  →  DIIS mix,
//! and the energy E = T_s + E_ext + E_H + E_xc + E_nn from ρ_out and the orbitals. This doc
//! covers only what the GPU changes.
//!
//! # What stays f64 (CLAUDE.md rule 4)
//! Every reduction finish (`Reducer`: f32 workgroup partials → Neumaier f64), the DIIS Gram
//! matrix and its bordered LU solve, the Rayleigh–Ritz inside LOBPCG, the CG scalars inside the
//! Poisson solver, the eigenvalues, E_nn (`qm::nuclear::nuclear_repulsion`) and the final sum of
//! the energy components. Everything on the grid — orbitals, densities, potentials, residuals,
//! the DIIS history — is f32 in padded `Field`s; the cold-start guess (`deposit_charge` at
//! σ = 1, `qm::scf::initial_orbitals`) is built in f64 on the CPU and uploaded once.
//!
//! # The three f32 floors and how the loop lives with them
//! - **Poisson** (`GpuPoisson`): the true residual floors at ~6e-6 (33³) … 2e-5 (97³) relative.
//!   Every solve is warm-started from the previous V_H and asked for `POISSON_TOL` = 1e-8, i.e.
//!   *below* the floor: the CG recurrence runs until it reaches 1e-8 or the floor signature is
//!   detected, and a stop at the floor (`Stop::Floor`/`Stagnated`) is "use V" — only `MaxIter`
//!   warns. Stopping the recurrence at 1e-6 instead (the first choice) leaves a *smooth* error
//!   of that relative size in V_H that differs from one warm restart to the next, which showed
//!   up as a Δρ stall at 2–5e-6 for H₂ (19 steps vs the CPU's 10); at 1e-8 the leftover is the
//!   floor's high-frequency rounding drift, which the density barely sees (10 steps, +15% CG
//!   iterations, equal wall time; measured in `diag_trajectory_and_knobs`). E_H of a unit
//!   charge at the floor is good to ~1e-6 Ha (Task 4).
//! - **LOBPCG** (`eigen`): the residual of an f32-stored vector floors at
//!   `residual_floor_estimate(g)` ≈ 1.8e-6 at h = 0.25. The eigensolver's tightest tol (the
//!   "M3 tol"; since M4 the per-step tol follows Δρ above it, next section) is
//!   `tol = max(settings.eig_tol, 3 × floor)`, so the default `eig_tol` = 1e-6 does not cost a
//!   12-iteration stall window per call (factor 1 was measured: H₂ 11 steps, Li 26 vs 17, and
//!   1.3–1.7× the wall time). A consequence worth knowing (Task 10, traced on H₂ 33³): from
//!   SCF step ~8 the warm orbitals' residual (~4.9e-6) already sits below 3× the floor and
//!   every later eigensolve exits after **0 iterations** — the orbitals freeze while Δρ is
//!   still ~1e-5 and the mixing alone finishes the convergence. That is not a defect: an
//!   orbital *update* injects fresh f32 rounding (~1e-7 in Δρ for H₂), so forcing ≥ 1
//!   Rayleigh–Ritz step per call (measured) plants a Δρ floor — tol 1e-7 then never converges
//!   (117 steps) — whereas frozen orbitals make ρ_out bit-identical between steps and Δρ can
//!   fall to the mixing's own f32 floor. The step count does depend on *when* the freeze
//!   happens: a late small update (a warm residual a hair above tol, three Rayleigh–Ritz
//!   steps, ρ_out moved by ~1e-5) makes the DIIS history inconsistent with the new frozen
//!   map and costs a history length of slow steps (measured 10 vs 15 steps for H₂ 33³ between
//!   two correct eigensolvers whose exit residuals differed by 4 %). So the freeze has a
//!   hysteresis band: a warm guess within `EIG_GUESS_FACTOR` = 1.5× tol is accepted as it is
//!   (`lobpcg_with_guess_tol`), otherwise the solve runs to tol. The final-step residual can
//!   then be up to 1.5× tol — within the `LAST_STEP_RESIDUAL_FACTOR` warning and second
//!   order in the energy. `max_eigen_residual` reports the true residual, and a `Floor`
//!   exit above tol is *informational* (ruled): counted in `SolveProfile::eigen_floor_accepts`
//!   with a `notes` line, never a warning and never a failure. So is a `MaxIter` exit on a
//!   non-final SCF step (ruled, CPU parity with M1: an early cold eigensolve on a near-degenerate
//!   pair — Li's 2s/2p — can hit the 200-iteration cap and the SCF still converges; the next step
//!   finishes the job): counted and summarised in `notes`. `ScfStats::warnings` carries only
//!   actionable conditions: box too small, Poisson at its iteration cap, and LOBPCG *not
//!   converged on the final SCF step* (a `MaxIter` exit there, or a final-step residual above
//!   `LAST_STEP_RESIDUAL_FACTOR` × the effective tol — the state the energy and forces are
//!   evaluated from is then not what was asked for). Eigenvalues and the density are second
//!   order in the residual.
//! - **The mixing step** is where f32 would hurt most: a plain `lincomb` Σ cᵢ(ρᵢ + α rᵢ) with
//!   coefficients of both signs rounds at |c|·ρ size and leaves ~3e-7·|c|·N of noise in Δρ, in
//!   competition with tol = 1e-6 for N ≥ 2. The `diis_mix` kernel evaluates the same
//!   combination about the latest input density, ρ_m + Σ_{i<m} cᵢ(ρᵢ − ρ_m) + αΣ cᵢ rᵢ
//!   (identical since Σ cᵢ = 1): the differences are exact (Sterbenz) and the small terms are
//!   summed before one ρ-sized rounding, which puts the floor at the storage rounding of ρ_in
//!   itself (~3e-8·N). The negative clamp is folded into the same dispatch.
//! - **Charge renormalisation.** The clamp changes ∫ρ_in by a tiny amount (as on the CPU, where
//!   it is left alone — the M1 deferred minor); here each channel is rescaled afterwards (one
//!   readback of two sums) to ∫ρ_out,σ of the step just taken, *not* to n_σ: the f32-normalised
//!   orbitals integrate to n_σ(1 + ε) with |ε| ≲ 1e-7 (measured −5e-8 per channel for H₂ at
//!   33³: ∫ρ = 2 − 1.0e-7), and a fixed point whose input charge is n_σ while its output charge
//!   is n_σ(1 + ε) has Δρ ≥ N|ε| — up to ~1e-6 for H₂O against tol 1e-6. Renormalising to
//!   ∫ρ_out keeps the fixed point charge-consistent and only repairs the clamp's loss. Measured
//!   (Task 8, the ruled one-off run with the old n_σ target): H₂O 49³ needs 27 DIIS steps
//!   instead of 17 at tol 1e-6 (43 vs 33 at 1e-7) — asserted as a step-count guard in
//!   `validate::gpu_vs_cpu::energies_match_at_fixed_sigma`; H₂'s N|ε| ≈ 1e-7 sits at the
//!   tolerance and cannot discriminate.
//!
//! # The eigen tolerance follows Δρ (M4)
//! Each SCF step asks LOBPCG for `eig_tol_for` = clamp(`EIG_TOL_FRACTION`·Δρ_{k−1}, lo,
//! `EIG_TOL_MAX`) with lo the M3 tol above; the first step (no Δρ yet) uses lo. Why the orbital
//! residual bound must track Δρ: an orbital set with residual r carries a density error of
//! ~κ·r, κ ≲ 2N_σ/gap (∫|δρ| ≤ 2Σ‖δψ_i‖, ‖δψ‖ ≤ r/gap); measured κ ≈ 10 for H₂ 49³ from the
//! size of the Δρ jump when a frozen set is re-solved (8e-6 → 8.6e-5). The SCF is then an
//! inexact fixed-point iteration whose map error κ·τ_k must stay below what one DIIS step
//! removes (Δρ contracts ~0.1–0.3 per step): κ·F ≲ 0.03, F ≈ 0.001–0.003, measured 0.0015.
//! Two failure modes bracket it, both measured with `diag_eig_fraction`: a loose F (0.1, the
//! first guess) keeps orbitals frozen through most of a solve inside the 1.5× guess band
//! (warm H₂ witness 51 steps, H₂O 129); the M3 fixed tol converges every step's orbitals to
//! the floor tol from step 1, so in a warm solve they freeze early (step 6 of the app's
//! witness, while Δρ is still 6× tol) and the potential's drift re-solves them later — the
//! orbital lag of `docs/superpowers/notes/2026-10-13-m3-task10-orbital-lag.md`. With the rule
//! the early steps' eigensolves stop at a tol matched to the SCF error (a few iterations
//! instead of 30–90) and the orbitals reach the floor tol only once Δρ ≲ lo/F ≈ 4e-3.
//! What the 3·floor clamp protects: a tol below ~2× the f32 residual floor ends every call on
//! the `STALL_WINDOW` (12 extra iterations) and, for tol 1e-7, re-solves inject fresh rounding
//! into ρ_out and plant a Δρ floor (Task 10) — so the rule never asks for less than M3 did.
//! Tightening *below* the M3 tol as Δρ → 0 (Task 10's remedy 1 read literally) was measured
//! too: more steps everywhere (H₂ cold 9 → 12, H₂ 33³ tol 1e-7 11 → 16).
//!
//! **DIIS restart on a re-solved frozen set.** When the eigensolves of a step take ≥ 1
//! iteration after a step whose took 0 (frozen orbitals), the map ρ_in ↦ ρ_out has jumped and
//! the history's residuals belong to the old map; DIIS then extrapolates from inconsistent
//! entries — measured as Δρ stuck at the same value (7.071e-5) for four steps with density
//! extrapolation — so the history restarts from the new entry (`SolveProfile::diis_resets`).
//! It never fires in a solve without a freeze/re-solve sequence (cold solves above are
//! unchanged by it). Task 10's remedy 3.
//!
//! Measured (RX 7900 XTX, `diag_eig_fraction`, readbacks = the deterministic cost; M3 → M4):
//! the solver-level witness at tol 1e-5 (H₂ 49³ 0.70 → 0.72, `diag`) warm 15 steps / 553 → 7 / 365,
//! cold of the displaced 9 / 593 → 9 / 495 (the app's R13 teleport path, `GpuKs::discontinuity`
//! with extrapolation dropped, gives 381 / 477 = 0.80, `moving_a_nucleus_warm_starts`); at tol 1e-6 warm 16 / 569 → 9 / 396; H₂O 49³ cold 16 / 1072 →
//! 15 / 760, warm 12 / 638 → 12 / 511; Li 20 / 1899 → 21 / 1478; LiH 16 / 1155 → 15 / 853; He
//! 13 / 680 → 11 / 546; H₂ 33³ tol 1e-7 11 / 532 → 13 / 484 (CPU 11; that tol sits on the
//! mixing's f32 Δρ plateau ≈ 1e-7, where the step count is a coin flip between 11 and 19
//! across rounding-level variants). Energies move by up to 4.3e-6 Ha (the validate H₂O case,
//! +2.35e-7 → −4.10e-6 vs the CPU), inside the documented 5.4e-6 H₂O GPU path scatter.
//! `SolveProfile::{eig_tol_used, eig_tol_steps, delta_rho_steps}` record the tols and Δρs.
//! A Δρ already below `settings.tol` always buys a tight step (review I1), so the
//! "converge only at the tight tol" rule terminates for any SCF tol.
//!
//! **Scope of the rule.** On an extrapolated MD path (next section) the first step's Δρ is
//! already small (1.6e-4 on the H₂ path), so F·Δρ < lo from step 2 on and every step runs at
//! the tight tol: the rule is inert there and the MD gain comes from the extrapolation plus the
//! DIIS restart. Its measured value is on cold solves and larger warm jumps (viewer drags).
//! F encodes H₂'s κ ≈ 10; κ ≲ 2N/gap grows with the electron count (H₂O's bound ≈ 80, though
//! measured H₂O behaves far better — orbital errors do not add coherently). A derived
//! τ_k = c·Δρ_{k−1}/N (gap taken as O(0.3) Ha, c ≈ 0.005) would scale without fitting and with
//! no per-element data — future work, before systems much larger than H₂O.
//!
//! # Density extrapolation (M4, opt-in)
//! `set_density_extrapolation(true)`: a warm solve after two converged solves starts from
//! ρ_in = max(2ρ_t − ρ_{t−1}, 0) per spin (`diis_mix` about ρ_t: the difference is exact, one
//! rounding, the clamp folded in), ρ_{t−1} parked in two pool fields. On an MD-like path (H₂
//! 33³, R = 1.40 … 1.49 in 0.01 steps) the first-step Δρ drops 100× (1.5e-2 → 1.6e-4) and the
//! warm solves take 7.0 steps instead of 9.0 (2152 vs 3287 readbacks), energies equal to
//! 8e-8 — but only with the DIIS restart above: without it the extrapolated start froze the
//! orbitals at step 2, re-solved them at step 3 and stalled (11.3 steps, measured at F = 0.002).
//!
//! # Readbacks per SCF step (the sync points)
//! Poisson: 3 + 2·(CG iterations) + checks + 1 (warm from the previous V_H: a few iterations
//! after the first step); LOBPCG per non-empty channel: 1 for the guess and 1 per iteration
//! (Task 10; was k + 2 and 2k + 1); then exactly two for the SCF itself — Δρ together with the new DIIS Gram row
//! (one `Reducer::many_in`), and the two channel charges for the renormalisation. The energy
//! after convergence is one Poisson solve plus a single readback (T_s per orbital, E_ext, E_H,
//! E_xc, the face maximum). Everything between sync points is recorded into one `Batch`
//! (`Poisson` and `lobpcg` run their own batches; the SCF batch is flushed before each).
//!
//! # Determinism
//! Two identical cold solves give bit-identical energies: every reduction has a fixed
//! workgroup count, per-thread order and tree shape (`reduce`), every pointwise kernel writes
//! its own element, LOBPCG's re-seeding and the DIIS LU are deterministic, and no per-op state
//! depends on timing. `solve_is_deterministic` pins it (same solver twice, and a fresh solver).
//!
//! # Fields and VRAM
//! The solver owns a pool of padded fields that grows to the peak concurrent need and is
//! reused across solves (nothing is allocated per step). Per solve: K = n↑ + n↓ orbitals, 3
//! densities, V_H, V_nuc in the returned `GpuState`; ~8 working fields; 4 per DIIS history
//! entry (24 at the default history 6); K kinetic scratch fields for the energy. Plus one
//! `GpuPoisson` (54 fields, 227 MB at 97³) and one `LobpcgWork` (8k + 2 fields) per solver.
//! At 97³ with K = 8 that is ~700 MB. A warm solve reuses the previous state's fields in place.
//!
//! # Ionic species and occupations (M7 Task 8a)
//! `qm::scf`'s module doc ("Ionic species (M7)", "Occupations") is the physics; the GPU runs the same algorithm:
//! - `System.ionic == None` is everything above, the same calls in the same order (`gpu_none_policy_is_bitwise_today`
//!   pins the energy and force bits of H₂ and LiH measured before the change). Under a setup, V_nuc is the local ionic
//!   potential V_loc (`ionic::GpuIonic::local_potential_in`: per-centre erf + the uniform V_SR tables), XC runs on
//!   (ρ↑ + ½ρ_pc, ρ↓ + ½ρ_pc) through two scratch fields when any ion has a partial core, LOBPCG gets
//!   `GpuHamiltonian { v, nl }` (V_NL inside every H·x; the Jacobi preconditioner stays local), the external energy adds
//!   E_NL = Σ f E_KB c² (one more readback, f64 finish), E_nn is `ionic_repulsion`, the core overlaps and the LSDA-domain
//!   count (max(−n_σ) in the energy readback; the points counted on the CPU only when there are any) are warnings as on
//!   the CPU. The `GpuIonic` is kept by the solver and updated per geometry (no allocation per step or per MD step).
//! - `Occupations::DegenerateShells`: the CPU's two-phase rule step for step, with its helpers
//!   (`qm::scf::{degenerate_shell_occupations, decide, ascending, shared_runs}`, τ_c, the growth cap and the seeds of the
//!   grown witnesses): one witness per occupied spin on a cold start, growth until the gap witness clears τ_c (the LOBPCG
//!   work space grows with it), the weighted density (`mul_add_sq_w`), the weighted kinetic energy and E_NL (f = 0
//!   skipped), and at the first convergence the release/freeze and a DIIS restart. `GpuState` records the setup, the
//!   occupations and the scheme; the warm guard requires the same grid, counts, scheme and ionic physics.
//! - Measured (references.md "GPU vs CPU: pseudopotentials (M7 Task 8)"): |ΔE| ≤ 6.1e-7 Ha and |Δε| ≤ 3.1e-6 Ha against
//!   the CPU on pseudo-O (CoresOnly triplet, shared), CH₄ (CoresOnly) and H₂O (All). A degenerate shared p cluster
//!   floors ~2× above LOBPCG's model floor (`eigen` "The f32 residual floor": the cluster is rotated every call), so the
//!   orbitals freeze late and the pseudo-O 61³ solve takes 24 steps against the CPU's 13 (3.2 s).
//!
//! # Display access (orbital viewer, plan Task 4)
//! The display eigensolve for unoccupied orbitals (`crate::orbitals::OrbitalView::unoccupied`)
//! runs on the converged state between solves through three crate-private accessors, none of which
//! changes what a solve does: `display_work` (the converged state and this solver's `LobpcgWork`,
//! **grown** to the display block n_σ + m + `UNOCC_GUARD` — +8 fields per vector over the SCF's
//! own n_σ, ≈ 0.22 GB at 97³ with m = 4), `release_display_work` (drops the work space; the next
//! solve re-creates it at its own size) and `effective_eig_tol` (the M3 tol the SCF itself uses,
//! one formula). Sharing the work space is safe because LOBPCG carries no state across calls:
//! it uses only the first `x.len()` slots, starts with no P columns (`np = 0`) and recomputes HX
//! from the guess at entry, so a work space that a display solve grew, wrote or released leaves
//! the next SCF solve bit-for-bit what it was (pinned by `providers`'
//! `unoccupied_solve_leaves_scf_state_untouched` over 20 MD steps).
//! Under an ionic setup `display_work` also lends the solver's `GpuIonic` (the converged state's
//! geometry): the display H^out is then the SCF's own operator — V_loc in `v_nuc`, v_xc of
//! (ρ↑ + ½ρ_pc, ρ↓ + ½ρ_pc) through `xc_in`, and V_NL inside every H·x (`GpuHamiltonian { v, nl }`).
//! The display solve writes only the nonlocal operator's projection scratch (`proj_dot` partials,
//! `proj_finish` coefficients), which every apply recomputes from its input, so the next solve is
//! unaffected.
use crate::batch::Batch;
use crate::eigen::{lobpcg_with_guess_tol, residual_floor_estimate, EigenStats, GpuHamiltonian, LobpcgWork, Stop as EigenStop};
use crate::grid::{clear_readback_fault, take_readback_fault, Field, GpuGrid};
use crate::ionic::GpuIonic;
use crate::kernels::{encoder, Kernels};
use crate::physics::{nuclear_potential_in, xc_lsda_in};
use crate::pointwise::{axpy_in, diis_mix_in, fill_in, lincomb_in, mul_add_sq_in, mul_add_sq_w_in, scale_in, unpad_rec, unpad_row_stride, DiisTerm};
use crate::poisson::{GpuPoisson, PoissonStats, Stop as PoissonStop};
use crate::reduce::{neumaier, Reducer, Reduction};
use crate::stencil::kinetic_in;
use gpu::{GpuContext, Volume3D};
use nalgebra::{DMatrix, DVector};
use qm::ionic::{core_overlaps, IonicSetup, CORE_OVERLAP_WARNING};
use qm::nuclear::{centres, deposit_centres, deposit_charge, ionic_repulsion, nuclear_repulsion, Centre};
use qm::scf::{ascending, decide, degenerate_shell_occupations, initial_orbitals, shared_runs, FractionalShell, CANDIDATE_TAU, MAX_WITNESS_GROWTH};
use qm::{Energy, Occupations, QmError, ScfSettings, ScfStats, System};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The latched readback failure of this thread's last readbacks (`grid::take_readback_fault`) as
/// the solve's error: `QmError::Device`. A failed readback hands the solver zeros, which can make
/// a sub-solver report a numerical failure or even a spurious convergence; this is checked after
/// each readback that feeds a decision, and wins over whatever the zeros caused.
pub(crate) fn take_device_error() -> Option<QmError> {
    take_readback_fault().map(|f| QmError::Device(f.to_string()))
}

/// Relative residual requested from every Poisson solve — below the f32 floor on purpose (module
/// doc), so each solve runs to the floor and the floor is accepted — and its iteration cap.
pub const POISSON_TOL: f64 = 1e-8;
const POISSON_MAX_ITER: usize = 5000;
/// LOBPCG Rayleigh–Ritz steps per call (as on the CPU); also the display eigensolve's cap.
pub(crate) const EIGEN_MAX_ITER: usize = 200;
/// LOBPCG tolerance = max(settings.eig_tol, this × `residual_floor_estimate`).
const EIG_TOL_FLOOR_FACTOR: f64 = 3.0;
/// A warm orbital set whose residual is below this × the effective tol is kept as it is (no
/// Rayleigh–Ritz step) — the hysteresis band of the orbital freeze (module doc).
const EIG_GUESS_FACTOR: f64 = 1.5;
/// The Δρ-tied eigen tol (`eig_tol_for`, module doc "The eigen tolerance follows Δρ"): the
/// LOBPCG tol of SCF step k is this × Δρ of step k − 1, clamped to [M3 tol, `EIG_TOL_MAX`].
/// Derived there: κ·F ≲ 0.03 with κ ≈ 10 (H₂) ⇒ F ≈ 0.001–0.003; measured 0.0015
/// (`diag_eig_fraction`; any F in 0.001–0.003 is physically equivalent, 0.0015 is the measured
/// window centre). The brief's ≈ 0.1 is 70× too loose: κ·F ≈ 1 lets the orbitals freeze
/// for most of a solve (H₂O 129 steps, the H₂ warm witness 51). F encodes H₂'s κ; the scaling
/// form c·Δρ/N is future work (module doc).
pub const EIG_TOL_FRACTION: f64 = 0.0015;
/// The loosest Δρ-tied eigen tol (`eig_tol_for`). Measured: 1e-2 lost Li (its near-degenerate
/// 2s/2p pair; no convergence in 200 steps), 1e-4 saves about half as much as 1e-3.
pub const EIG_TOL_MAX: f64 = 1e-3;

/// The LOBPCG tol of one SCF step (module doc): `clamp(fraction·Δρ_prev, lo, hi)` with `lo` the
/// M3 tol max(settings.eig_tol, `EIG_TOL_FLOOR_FACTOR` × floor) and `hi` raised to at least lo. The
/// first step (Δρ_prev = ∞) and `fraction` = 0 (the M3 behaviour, diagnostics) use `lo`.
fn eig_tol_for(delta_prev: f64, fraction: f64, lo: f64, hi: f64) -> f64 {
    if fraction <= 0.0 || !delta_prev.is_finite() { return lo; }
    (fraction * delta_prev).clamp(lo, hi.max(lo))
}

/// The M3 LOBPCG tol (module doc): max(`eig_tol`, `floor_factor` × `residual_floor_estimate`).
/// The SCF's tightest per-step tol and the display eigensolve's tol (one formula).
fn effective_eig_tol(eig_tol: f64, floor_factor: f64, g: &GpuGrid) -> f64 {
    eig_tol.max(floor_factor * residual_floor_estimate(g))
}

/// A final-step eigen residual above this × the effective tol is a warning (module doc).
const LAST_STEP_RESIDUAL_FACTOR: f64 = 3.0;
/// Density on a box face above this (e/Bohr³) triggers the box-size warning (`qm::scf`).
const BOUNDARY_DENSITY_WARN: f64 = 1e-5;
/// Relative diagonal regularisation of the (scaled) residual Gram matrix (`qm::scf`).
const DIIS_REG: f64 = 1e-12;

/// Converged Kohn-Sham state on the GPU: orbitals, eigenvalues, output densities (`rho` =
/// `rho_up` + `rho_dn`) and the potentials (`v_h` is V_H of `rho`), all padded f32 fields.
/// `grid` and `sigma` record what it was converged on. Owned by the `GpuSolver` that made it
/// (it is the warm start of the next solve); borrow it through `solve`.
///
/// M7 Task 8a (`qm::KsState`'s fields of the same names): `ionic` the setup the state was converged under (`None`:
/// all-electron; `v_nuc` is then today's V_nuc, else the local ionic potential V_loc), `occ_up`/`occ_dn` one occupation
/// per stored orbital (all 1 under `Aufbau`; under `DegenerateShells` the orbitals include the gap witnesses, f = 0) and
/// `occupations` the scheme. A warm start requires the same scheme and the same ionic physics (`qm::scf`'s guard).
pub struct GpuState {
    pub grid: qm::Grid, pub sigma: f64,
    pub orbitals_up: Vec<Field>, pub orbitals_dn: Vec<Field>,
    pub evals_up: Vec<f64>, pub evals_dn: Vec<f64>,
    pub rho: Field, pub rho_up: Field, pub rho_dn: Field,
    pub v_h: Field, pub v_nuc: Field,
    pub ionic: Option<Arc<IonicSetup>>,
    pub occ_up: Vec<f64>, pub occ_dn: Vec<f64>, pub occupations: Occupations,
}

/// Cost accounting and informational diagnostics of the last `solve` — kept for a *failed*
/// solve too (`ScfNotConverged`), whose `ScfStats` never exists: SCF steps, GPU→CPU readbacks
/// by origin, inner iteration totals, wall time in total and per phase (reported, not asserted
/// — hardware-dependent), the LOBPCG bookkeeping that is *not* a warning: how many eigensolves
/// ended on the f32 floor above tol (`eigen_floor_accepts`, worst residual `eigen_floor_worst`)
/// or at the iteration cap on a non-final step, the max residual of the final SCF step's
/// eigensolves (`eigen_last_step_residual` — `ScfStats::max_eigen_residual` is the worst over
/// the tight-tol steps and cannot say whether the last one converged), the worst residual of
/// any call including the deliberately loosened early steps (`eigen_worst_residual_any_step`),
/// `notes` (informational lines) and
/// `warnings` (the same list as `ScfStats::warnings`, so a failed solve's are not lost).
///
/// Phase walls: `start_wall` = building and uploading the cold-start guess (CPU f64; zero on a
/// warm start), `poisson_wall` = inside `GpuPoisson::solve` (every SCF step and the final
/// energy solve), `eigen_wall` = inside `lobpcg` (both channels). Each sub-solver begins and
/// ends on a readback and the SCF batch is flushed before it, so the wall between those sync
/// points is that phase's; `wall − start − poisson − eigen` is the SCF's own pointwise work,
/// DIIS, the two SCF readbacks per step and the energy evaluation.
#[derive(Debug, Clone, Default)]
pub struct SolveProfile {
    pub steps: usize,
    pub readbacks: usize,
    pub poisson_readbacks: usize,
    pub eigen_readbacks: usize,
    pub poisson_iterations: usize,
    pub eigen_iterations: usize,
    /// LOBPCG compute dispatches, and iterations that needed its MGS fallback (`qm_gpu::eigen`).
    pub eigen_dispatches: usize,
    pub eigen_fallbacks: usize,
    pub wall: Duration,
    pub start_wall: Duration,
    pub poisson_wall: Duration,
    pub eigen_wall: Duration,
    pub eigen_floor_accepts: usize,
    pub eigen_floor_worst: f64,
    /// Eigensolves that exited at the iteration cap (any step) and the worst such residual;
    /// a warning only if one of them was on the final step (`Inner::finish`).
    pub eigen_cap_hits: usize,
    pub eigen_cap_worst: f64,
    pub eigen_last_step_residual: f64,
    /// Worst LOBPCG exit residual over every call, the loosened (Δρ-tied) steps included — not
    /// a statement about the converged state (`ScfStats::max_eigen_residual` is, M4 review I2).
    pub eigen_worst_residual_any_step: f64,
    /// The LOBPCG tol of the final SCF step (`eig_tol_for`: tied to the previous step's Δρ) and
    /// of every step in order.
    pub eig_tol_used: f64,
    pub eig_tol_steps: Vec<f64>,
    /// DIIS history restarts: a frozen orbital set (0 LOBPCG iterations) re-solved (module doc).
    pub diis_resets: usize,
    /// Δρ of every step in order (the convergence history; `eig_tol_steps` is its partner).
    pub delta_rho_steps: Vec<f64>,
    /// The warm start's input density was extrapolated, ρ_in = 2ρ_t − ρ_{t−1}
    /// (`GpuSolver::set_density_extrapolation`).
    pub extrapolated: bool,
    pub notes: Vec<String>,
    pub warnings: Vec<String>,
}

/// What a display eigensolve borrows from the solver (`GpuSolver::display_work`, module doc
/// "Display access"): its kernels, its converged state (read only), its LOBPCG work space and —
/// when the state was solved under an ionic setup — the ionic terms of that state's geometry
/// (V_NL and ρ_pc for H^out; `None` for an all-electron state).
pub(crate) struct DisplayWork<'a> {
    pub k: &'a Kernels,
    pub state: &'a GpuState,
    pub work: &'a mut LobpcgWork,
    pub ionic: Option<&'a GpuIonic>,
}

/// A pool of padded fields reused across solves (never shrinks). Every consumer overwrites the
/// interior of what it takes; ghosts are zero by the crate invariant.
struct Pool { free: Vec<Field> }

impl Pool {
    fn take(&mut self, ctx: &GpuContext, g: &GpuGrid) -> Field {
        self.free.pop().unwrap_or_else(|| Field::new(ctx, g, "scf field"))
    }
    fn give(&mut self, f: Field) { self.free.push(f); }
    fn give_state(&mut self, s: GpuState) {
        let GpuState { orbitals_up, orbitals_dn, rho, rho_up, rho_dn, v_h, v_nuc, .. } = s;
        self.free.extend(orbitals_up.into_iter().chain(orbitals_dn).chain([rho, rho_up, rho_dn, v_h, v_nuc]));
    }
}

/// One Pulay history entry (input densities and residuals per spin) as fields.
struct DiisEntry { rho_up: Field, rho_dn: Field, r_up: Field, r_dn: Field }

/// The DIIS history with its Gram matrix B_ij = ⟨r_i↑, r_j↑⟩ + ⟨r_i↓, r_j↓⟩ kept on the CPU and
/// grown by one row per step (the dots come with the Δρ readback), shrunk when an entry drops.
struct Diis { entries: Vec<DiisEntry>, gram: Vec<Vec<f64>>, cap: usize }

impl Diis {
    fn new(cap: usize) -> Diis { Diis { entries: Vec::new(), gram: Vec::new(), cap: cap.max(1) } }

    /// Appends `e` whose Gram row `row` = [⟨r_e, r_0⟩, …, ⟨r_e, r_{m−1}⟩, ⟨r_e, r_e⟩]; drops the
    /// oldest entry when over capacity.
    fn push(&mut self, e: DiisEntry, row: Vec<f64>, pool: &mut Pool) {
        let m = self.entries.len();
        assert_eq!(row.len(), m + 1);
        for (i, r) in self.gram.iter_mut().enumerate() { r.push(row[i]); }
        self.gram.push(row);
        self.entries.push(e);
        if self.entries.len() > self.cap { self.drop_oldest(pool); }
    }

    fn drop_oldest(&mut self, pool: &mut Pool) {
        let e = self.entries.remove(0);
        pool.give(e.rho_up); pool.give(e.rho_dn); pool.give(e.r_up); pool.give(e.r_dn);
        self.gram.remove(0);
        for r in &mut self.gram { r.remove(0); }
    }

    fn clear(&mut self, pool: &mut Pool) { while !self.entries.is_empty() { self.drop_oldest(pool); } }

    /// Pulay coefficients (`qm::scf::diis_coefficients`): bordered system on the Gram matrix
    /// scaled by its largest diagonal, +1e-12 relative on the diagonal, LU. `None` on failure.
    fn coefficients(&self) -> Option<Vec<f64>> {
        let m = self.entries.len();
        let mut b = DMatrix::<f64>::zeros(m + 1, m + 1);
        for i in 0..m { for j in 0..m { b[(i, j)] = self.gram[i][j]; } }
        let scale = (0..m).map(|i| b[(i, i)]).fold(0.0f64, f64::max);
        if scale.is_nan() || scale <= 0.0 { return None; }
        for i in 0..m { for j in 0..m { b[(i, j)] /= scale; } b[(i, i)] += DIIS_REG; }
        for i in 0..m { b[(i, m)] = 1.0; b[(m, i)] = 1.0; }
        let mut rhs = DVector::<f64>::zeros(m + 1);
        rhs[m] = 1.0;
        let x = b.lu().solve(&rhs)?;
        let c: Vec<f64> = x.iter().take(m).copied().collect();
        if c.iter().all(|v| v.is_finite()) { Some(c) } else { None }
    }

    /// ρ_in,new,σ = Σ cᵢ (ρ_in,i,σ + α r_i,σ) clamped at 0, evaluated about the latest entry
    /// (module doc). Drops the oldest entries while the coefficient solve fails; one entry is
    /// linear mixing.
    #[allow(clippy::too_many_arguments)]
    fn mix(&mut self, b: &mut Batch, k: &Kernels, g: &GpuGrid, alpha: f64, out_up: &Field, out_dn: &Field, pool: &mut Pool) {
        let c = loop {
            if self.entries.len() == 1 { break vec![1.0]; }
            match self.coefficients() {
                Some(c) => break c,
                None => self.drop_oldest(pool),
            }
        };
        let m = self.entries.len();
        let base = &self.entries[m - 1];
        let terms = |rho: &dyn Fn(&DiisEntry) -> &Field, r: &dyn Fn(&DiisEntry) -> &Field| -> Vec<DiisTerm> {
            (0..m - 1).map(|i| (c[i] as f32, rho(&self.entries[i]), true))
                .chain((0..m).map(|i| ((alpha * c[i]) as f32, r(&self.entries[i]), false)))
                .collect()
        };
        diis_mix_in(b, k, g, out_up, &base.rho_up, &terms(&|e| &e.rho_up, &|e| &e.r_up), true);
        diis_mix_in(b, k, g, out_dn, &base.rho_dn, &terms(&|e| &e.rho_dn, &|e| &e.r_dn), true);
    }
}

/// NaN-propagating max (a non-finite residual must not hide behind `f64::max`).
fn nan_max(a: f64, b: f64) -> f64 { if a.is_nan() || b.is_nan() { f64::NAN } else { a.max(b) } }

/// Worst-case inner-solver diagnostics over a solve (`qm::scf::InnerStats` with the GPU's stop
/// semantics). `warnings` hold actionable conditions only: Poisson `MaxIter`, and LOBPCG not
/// converged on the *final* SCF step (a `MaxIter` exit there, or `eigen_last_step_residual`
/// above `LAST_STEP_RESIDUAL_FACTOR` × tol). A LOBPCG `Floor` exit above tol and a `MaxIter`
/// exit on a non-final step are informational (ruled) and go to `SolveProfile::{eigen_floor_accepts,
/// eigen_floor_worst, notes}`, never to `warnings`.
struct Inner {
    eigen_residual: f64, poisson_residual: f64, poisson_iterations: usize, warnings: Vec<String>,
    eigen_calls: usize, step_residual: f64, step_max_iter: bool,
    /// The tightest per-call tol (`max(settings.eig_tol, 3·floor)`, for the notes) and the tol of
    /// the current SCF step's eigensolves (the final step's decides the warning).
    eig_tol: f64, step_tol: f64,
    profile: SolveProfile,
}

impl Inner {
    fn new(eig_tol: f64) -> Inner {
        Inner { eigen_residual: 0.0, poisson_residual: 0.0, poisson_iterations: 0, warnings: Vec::new(),
                eigen_calls: 0, step_residual: 0.0, step_max_iter: false, eig_tol, step_tol: eig_tol, profile: SolveProfile::default() }
    }

    /// Call at the start of each SCF step with the step's eigen tol: resets the per-step eigen
    /// residual and cap flag and records the tol (`SolveProfile::{eig_tol_used, eig_tol_steps}`).
    fn begin_step(&mut self, tol: f64) {
        self.step_residual = 0.0;
        self.step_max_iter = false;
        self.step_tol = tol;
        self.profile.eig_tol_used = tol;
        self.profile.eig_tol_steps.push(tol);
    }

    fn eigen(&mut self, s: EigenStats, wall: Duration) {
        self.eigen_calls += 1;
        // `ScfStats::max_eigen_residual` keeps its M1/M3 meaning — the worst residual among the
        // solves at the tight tol, the ones the final state rests on (review I2); the loosened
        // early steps only go into the profile's any-step worst
        if self.step_tol <= self.eig_tol { self.eigen_residual = nan_max(self.eigen_residual, s.max_residual); }
        self.profile.eigen_worst_residual_any_step = nan_max(self.profile.eigen_worst_residual_any_step, s.max_residual);
        self.step_residual = nan_max(self.step_residual, s.max_residual);
        self.profile.eigen_last_step_residual = self.step_residual;
        self.profile.eigen_readbacks += s.readbacks;
        self.profile.eigen_iterations += s.iterations;
        self.profile.eigen_dispatches += s.dispatches;
        self.profile.eigen_fallbacks += s.fallbacks;
        self.profile.eigen_wall += wall;
        match s.stop {
            EigenStop::Floor if s.max_residual > self.step_tol => {
                self.profile.eigen_floor_accepts += 1;
                self.profile.eigen_floor_worst = self.profile.eigen_floor_worst.max(s.max_residual);
            }
            EigenStop::MaxIter => {
                self.profile.eigen_cap_hits += 1;
                self.profile.eigen_cap_worst = self.profile.eigen_cap_worst.max(s.max_residual);
                self.step_max_iter = true;
            }
            _ => {}
        }
    }

    fn poisson(&mut self, s: PoissonStats, what: &str, wall: Duration) {
        self.poisson_residual = nan_max(self.poisson_residual, s.residual);
        self.poisson_iterations = self.poisson_iterations.max(s.iterations);
        self.profile.poisson_readbacks += s.readbacks;
        self.profile.poisson_iterations += s.iterations;
        self.profile.poisson_wall += wall;
        if s.stop == PoissonStop::MaxIter {
            self.warnings.push(format!("{what}: Poisson stopped after {} iterations at relative residual {:.2e} > {POISSON_TOL:.0e}", s.iterations, s.residual));
        }
    }

    /// The warnings and the completed profile of a solve (converged or not): `steps`, the
    /// total readback count, the wall, the LOBPCG notes, and the final-step LOBPCG warning if
    /// its last eigensolves did not converge (`step_max_iter`/`step_residual` are the final
    /// step's — the last `begin_step` was its start). `extra` (the box warning) goes first.
    fn finish(mut self, steps: usize, wall: Duration, extra: Vec<String>) -> (Vec<String>, SolveProfile) {
        if self.profile.eigen_floor_accepts > 0 {
            self.profile.notes.push(format!("LOBPCG accepted at the f32 floor in {} of {} calls (worst residual {:.2e}, above its call's tol; tightest tol {:.2e})",
                                            self.profile.eigen_floor_accepts, self.eigen_calls, self.profile.eigen_floor_worst, self.eig_tol));
        }
        if self.profile.eigen_cap_hits > 0 {
            self.profile.notes.push(format!("LOBPCG hit its {EIGEN_MAX_ITER}-iteration cap in {} of {} calls (worst residual {:.2e} > tol {:.2e})",
                                            self.profile.eigen_cap_hits, self.eigen_calls, self.profile.eigen_cap_worst, self.eig_tol));
        }
        // a NaN residual warns too (it is not ≤ anything)
        let last = self.profile.eigen_last_step_residual;
        let last_ok = last <= LAST_STEP_RESIDUAL_FACTOR * self.eig_tol; // the tight tol: what the final state was asked for
        if self.step_max_iter || !last_ok {
            self.warnings.push(format!("LOBPCG not converged on the final SCF step {steps}: residual {last:.2e} vs tol {:.2e}{}",
                                       self.eig_tol, if self.step_max_iter { format!(" (stopped at its {EIGEN_MAX_ITER}-iteration cap)") } else { String::new() }));
        }
        let mut warnings = extra;
        warnings.append(&mut self.warnings);
        self.profile.steps = steps;
        self.profile.readbacks += self.profile.poisson_readbacks + self.profile.eigen_readbacks;
        self.profile.wall = wall;
        self.profile.warnings = warnings.clone();
        (warnings, self.profile)
    }
}

/// Submits everything recorded so far (before handing the queue to a sub-solver with its own batch).
fn flush(b: &mut Batch) {
    let enc = b.take_encoder();
    b.ctx().queue.submit([enc.finish()]);
}

/// LSDA on (ρ↑, ρ↓), or on (ρ↑ + ½ρ_pc, ρ↓ + ½ρ_pc) with a partial core `core` = (ρ_pc, two scratch fields) —
/// `qm::scf`'s `IonicTerms::xc` (recorded). Without a core it is exactly today's `xc_lsda_in` call. Also the
/// display eigensolve's v_xc of ρ_out (`crate::orbitals`, one formula).
#[allow(clippy::too_many_arguments)]
pub(crate) fn xc_in(b: &mut Batch, k: &Kernels, g: &GpuGrid, up: &Field, dn: &Field, core: Option<(&Field, &Field, &Field)>, exc: &Field, v_up: &Field, v_dn: &Field) {
    match core {
        None => xc_lsda_in(b, k, g, up, dn, exc, v_up, v_dn),
        Some((pc, nu, nd)) => {
            lincomb_in(b, k, g, nu, &[(1.0, up), (0.5, pc)]);
            lincomb_in(b, k, g, nd, &[(1.0, dn), (0.5, pc)]);
            xc_lsda_in(b, k, g, nu, nd, exc, v_up, v_dn);
        }
    }
}

/// ρ = Σ_i ψ_i² (`occ` None: today's sequence of `mul_add_sq`), or Σ_i f_i ψ_i² with f = 0 skipped and f = 1 by the
/// unweighted kernel (`qm::scf::weighted_density`; recorded).
fn accumulate_density(b: &mut Batch, k: &Kernels, g: &GpuGrid, rho: &Field, orbitals: &[Field], occ: Option<&[f64]>) {
    fill_in(b, k, g, rho, 0.0);
    match occ {
        None => for psi in orbitals { mul_add_sq_in(b, k, g, rho, psi); },
        Some(f) => for (psi, &fi) in orbitals.iter().zip(f) {
            if fi == 1.0 { mul_add_sq_in(b, k, g, rho, psi); } else if fi != 0.0 { mul_add_sq_w_in(b, k, g, rho, psi, fi as f32); }
        },
    }
}

/// `qm::scf::counts_match` on a GPU state: exactly N_σ orbitals per spin under `Aufbau` (today's test); under
/// `DegenerateShells` the same electron count Σf with at least one witness per occupied spin.
fn counts_match(s: &GpuState, n_up: usize, n_dn: usize, occ: Occupations) -> bool {
    match occ {
        Occupations::Aufbau => s.orbitals_up.len() == n_up && s.orbitals_dn.len() == n_dn,
        Occupations::DegenerateShells { .. } => {
            let fits = |len: usize, f: &[f64], n: usize| len == f.len() && (f.iter().sum::<f64>() - n as f64).abs() < 1e-9
                && (if n == 0 { len == 0 } else { len > n });
            fits(s.orbitals_up.len(), &s.occ_up, n_up) && fits(s.orbitals_dn.len(), &s.occ_dn, n_dn)
        }
    }
}

/// The warm-start guard (`qm::scf::same_ionic`): both all-electron without a setup, or the same physics.
fn same_ionic(a: Option<&IonicSetup>, b: Option<&IonicSetup>) -> bool {
    match (a, b) { (None, None) => true, (Some(x), Some(y)) => x.same_physics(y), _ => false }
}

/// One spin channel's eigenproblem under `DegenerateShells` (`qm::scf::solve_shared`).
struct Channel<'a> {
    ctx: &'a GpuContext, k: &'a Kernels, g: &'a GpuGrid, r: &'a Reducer, grid: &'a qm::Grid, nuclei: &'a [qm::Nucleus],
    h: GpuHamiltonian<'a>, n: usize, spin: u8, step_tol: f64,
}

impl Channel<'_> {
    /// One LOBPCG call on all of `orbs` (the work space grown to their count first). A latched readback failure
    /// (`grid::take_readback_fault`) is `QmError::Device`, winning over whatever the zeros it returned caused —
    /// before the occupations rule reads the eigenvalues.
    fn solve(&self, orbs: &mut [Field], evals: &mut [f64], work: &mut Option<LobpcgWork>, work_k: &mut usize, inner: &mut Inner) -> Result<(), QmError> {
        if work.is_none() || *work_k < orbs.len() {
            *work = Some(LobpcgWork::new(self.ctx, self.g, orbs.len()));
            *work_k = orbs.len();
        }
        let t = Instant::now();
        let es = lobpcg_with_guess_tol(self.ctx, self.k, self.g, self.r, &self.h, orbs, evals, work.as_mut().expect("work sized above"),
                                       self.step_tol, EIG_GUESS_FACTOR * self.step_tol, EIGEN_MAX_ITER, None)
            .map_err(|e| take_device_error().unwrap_or(e))?;
        if let Some(e) = take_device_error() { return Err(e); }
        inner.eigen(es, t.elapsed());
        Ok(())
    }

    /// Phase 1 (`by_rank` None): solve, grow the block by one vector until the gap witness clears τ_c, share a
    /// straddling cluster. Phase 2: the frozen occupations by rank. Returns the occupations and the shared clusters.
    /// The CPU's rule step for step (`qm::scf::solve_shared`; the same seeds for the grown vectors).
    #[allow(clippy::too_many_arguments)]
    fn solve_shared(&self, orbs: &mut Vec<Field>, evals: &mut Vec<f64>, by_rank: Option<&[f64]>, work: &mut Option<LobpcgWork>, work_k: &mut usize,
                    pool: &mut Pool, inner: &mut Inner) -> Result<(Vec<f64>, Vec<FractionalShell>), QmError> {
        if self.n == 0 { return Ok((vec![0.0; orbs.len()], Vec::new())); }
        if let Some(by_rank) = by_rank {
            self.solve(orbs, evals, work, work_k, inner)?;
            let order = ascending(evals);
            let mut f = vec![0.0; evals.len()];
            for (r, &i) in order.iter().enumerate() { f[i] = by_rank[r]; }
            return Ok((f, shared_runs(evals, &order, by_rank, self.spin)));
        }
        loop {
            self.solve(orbs, evals, work, work_k, inner)?;
            if let Some((f, shell)) = degenerate_shell_occupations(evals, self.n, CANDIDATE_TAU) {
                let spin = self.spin;
                return Ok((f, shell.map(|(first, count, f, spread)| FractionalShell { spin, first, count, f, spread }).into_iter().collect()));
            }
            if orbs.len() >= self.n + MAX_WITNESS_GROWTH {
                return Err(QmError::Numerical(format!("no gap witness within {MAX_WITNESS_GROWTH} extra levels of spin {} (τ_c = {CANDIDATE_TAU} Ha)", self.spin)));
            }
            let seed = 1000 + 2 * orbs.len() as u64 + self.spin as u64;
            let f = pool.take(self.ctx, self.g);
            f.upload_interior(self.ctx, self.g, &initial_orbitals(self.grid, self.nuclei, 1, seed)[0]);
            orbs.push(f);
            evals.push(*evals.last().expect("n > 0 vectors"));
        }
    }
}

/// The Kohn-Sham solver for one grid: kernels, reducer, Poisson solver, LOBPCG work space, a
/// field pool and the last converged state (the warm start). `Send` (one per worker thread).
pub struct GpuSolver {
    pub(crate) ctx: GpuContext, pub(crate) k: Kernels, r: Reducer, pub(crate) g: GpuGrid, grid: qm::Grid,
    poisson: GpuPoisson,
    work: Option<LobpcgWork>, work_k: usize,
    pool: Pool,
    /// The next warm start: the last converged state, or the in-progress state an abandoned
    /// `solve_with` parked (`converged` says which; only a converged one is exposed by `state`).
    warm: Option<GpuState>,
    converged: bool,
    /// Row-padded staging buffer for `copy_density_to_volume` and its row stride in floats.
    rows: wgpu::Buffer, row_stride: u32,
    profile: SolveProfile,
    /// `POISSON_TOL` and `EIG_TOL_FLOOR_FACTOR` as instance knobs (tests probe them).
    poisson_tol: f64, eig_floor_factor: f64,
    /// `EIG_TOL_FRACTION` (0 = the M3 fixed tol) and `EIG_TOL_MAX` as instance knobs.
    eig_fraction: f64, eig_tol_max: f64,
    /// Density extrapolation across warm solves (`set_density_extrapolation`) and ρ_{t−1},σ: the
    /// converged output densities of the solve before the last converged one (pool fields).
    extrapolate: bool, prev_rho: Option<(Field, Field)>,
    /// The forces' cached partials/staging pair (`forces::ForceScratch`; `forces` takes `&self`).
    pub(crate) forces: std::cell::RefCell<crate::forces::ForceScratch>,
    /// The density sampler's buffers and counters (`sample::SampleScratch`; `sample_density` takes `&self`).
    pub(crate) sampler: std::cell::RefCell<crate::sample::SampleScratch>,
    /// The ionic terms of the last pseudopotential solve (`GpuIonic`, buffers reused across geometries; M7 Task 8a).
    pub(crate) ionic: Option<GpuIonic>,
    /// The scratch field of `copy_density_with_core_to_volume` (allocated on first use, one field).
    core_display: std::cell::RefCell<Option<Field>>,
}

impl GpuSolver {
    /// Compiles the kernels and builds the Poisson solver (49 R_lm fields + ghost table) for `grid`.
    pub fn new(ctx: GpuContext, grid: &qm::Grid) -> GpuSolver {
        let g = GpuGrid::new(grid);
        let k = Kernels::new(&ctx);
        let r = Reducer::new(&ctx, &k, &g);
        let poisson = GpuPoisson::new(&ctx, &k, &g);
        let row_stride = unpad_row_stride(g.n);
        let rows = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("density rows"), size: row_stride as u64 * (g.n * g.n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
        });
        GpuSolver { ctx, k, r, g, grid: grid.clone(), poisson, work: None, work_k: 0, pool: Pool { free: Vec::new() },
                    warm: None, converged: false, rows, row_stride, profile: SolveProfile::default(),
                    poisson_tol: POISSON_TOL, eig_floor_factor: EIG_TOL_FLOOR_FACTOR,
                    eig_fraction: EIG_TOL_FRACTION, eig_tol_max: EIG_TOL_MAX, extrapolate: false, prev_rho: None,
                    forces: Default::default(), sampler: Default::default(), ionic: None, core_display: Default::default() }
    }

    /// Diagnostic knobs: the Poisson tolerance and the LOBPCG tolerance floor factor.
    #[cfg(test)]
    fn set_knobs(&mut self, poisson_tol: f64, eig_floor_factor: f64) { self.poisson_tol = poisson_tol; self.eig_floor_factor = eig_floor_factor; }

    /// Diagnostic knobs of the Δρ-tied eigen tol (`eig_tol_for`): the fraction (0 = the M3
    /// fixed tol) and the loose cap.
    #[cfg(test)]
    pub(crate) fn set_eig_knobs(&mut self, fraction: f64, max: f64) { self.eig_fraction = fraction; self.eig_tol_max = max; }

    /// Test hook: writes a NaN into one interior value (the box centre) of the parked warm
    /// state's ρ↑ — the input density of the next warm solve.
    #[cfg(test)]
    fn poison_parked_density(&mut self) {
        let s = self.warm.as_ref().expect("a parked state to poison");
        let c = self.g.n / 2;
        let offset = (self.g.pidx(c, c, c) * 4) as u64;
        self.ctx.queue.write_buffer(&s.rho_up.buffer, offset, bytemuck::bytes_of(&f32::NAN));
    }

    pub fn grid(&self) -> &qm::Grid { &self.grid }

    /// Density extrapolation for warm starts along a trajectory (M4 dynamics; off by default):
    /// a warm solve that follows two consecutive *converged* solves starts from
    /// ρ_in,σ = max(2ρ_t,σ − ρ_{t−1},σ, 0) — the linear extrapolation of the converged output
    /// densities in time, exact to O(Δt²) for equal time steps (the caller's contract; a
    /// geometry step of a different size makes it a worse, never an invalid, guess) — instead of
    /// ρ_t,σ. ρ_{t−1} is kept in two parked pool fields; it is dropped by a cold solve, a solve
    /// that did not converge (error or abandoned), a change of orbital counts, and by turning
    /// extrapolation off. `SolveProfile::extrapolated` says whether a solve used it. Orbitals
    /// and V_H are warm-started from step t as without extrapolation. A discontinuous move (a
    /// viewer drag during MD, a changed time step) must reset the chain — call
    /// `set_density_extrapolation(false)` then `(true)` — or the next start doubles the density
    /// at the new site (still valid, several steps slower). The extrapolated ρ_in is not
    /// renormalised: the clamp loses ~1e-7 of charge, ≪ the first step's Δρ (1.6e-4 on the H₂
    /// path), and the M3 renormalisation to ∫ρ_out applies after every mix from step 1 on.
    pub fn set_density_extrapolation(&mut self, on: bool) {
        self.extrapolate = on;
        if !on { self.drop_prev_rho(); }
    }

    /// Returns the parked ρ_{t−1} fields to the pool.
    fn drop_prev_rho(&mut self) {
        if let Some((u, d)) = self.prev_rho.take() { self.pool.give(u); self.pool.give(d); }
    }

    /// The last converged state, if the last solve converged (`None` after an error or an
    /// abandoned `solve_with`, even though the latter parks a warm start).
    pub fn state(&self) -> Option<&GpuState> { self.warm.as_ref().filter(|_| self.converged) }

    /// Cost accounting of the last `solve`.
    pub fn profile(&self) -> &SolveProfile { &self.profile }

    /// Same contract as `qm::scf::scf` on this solver's grid: `Ok((state, stats))` or
    /// `Err(QmError)` (coincident nuclei; no convergence within `settings.max_iter`, including a
    /// non-finite Δρ; and, GPU only, `QmError::Numerical` when a sub-solver's numerics break —
    /// a non-finite Gram entry, residual or Ritz value, rank loss or a failed Cholesky in LOBPCG,
    /// p·Ap ≤ 0, r·z ≤ 0 or a non-finite residual in the Poisson CG — instead of a panic, so a
    /// dynamics loop can react without rebuilding the solver; and `QmError::Device` when a GPU →
    /// CPU readback fails, e.g. under contention for the device, checked after every readback
    /// that feeds a decision — see `grid::take_readback_fault` — and winning over whatever the
    /// zeros it returned caused; the solver then holds no state, as after `Numerical`).
    /// `warm` reuses the previous converged state's orbitals, densities and V_H
    /// (or the state an abandoned `solve_with` parked) when its orbital counts match `system` (a
    /// different σ or geometry is fine); otherwise the cold path runs. After an SCF error the
    /// solver holds no state (the next solve is cold; a parked warm start is discarded too) and
    /// `profile()` is the failed solve's, with a note naming a numerical failure; after the coincident-nuclei error, which
    /// is raised before anything is touched, `state()` is `None` but the previous state is kept
    /// as the warm start.
    /// The returned `&GpuState` borrows the solver mutably (it lives inside it); to use it with
    /// `density_f32`/`copy_density_to_volume`/`density_max`, re-borrow it through `state()`.
    pub fn solve(&mut self, system: &System, settings: &ScfSettings, warm: bool) -> Result<(&GpuState, ScfStats), QmError> {
        Ok(self.solve_with(system, settings, warm, &mut || false)?.expect("never asked to abort"))
    }

    /// `solve` with a cancellation point: `should_abort` is called once before every SCF step
    /// (never inside a kernel, a Poisson solve or an eigensolve — the latency of an abort is at
    /// most one SCF step). When it returns true the solve is abandoned: `Ok(None)`, `state()` is
    /// `None`, and the in-progress state (current orbitals and eigenvalues, the mixed input
    /// densities ρ_in,σ as `rho_up`/`rho_dn` with `rho` their sum, the last V_H) is parked as the
    /// next warm start — an interactive caller that abandons solve after solve while the user
    /// drags still makes progress, and the solve after the drag starts close to the answer. A
    /// later `warm = false`, a different grid or different orbital counts discard it as they
    /// would a converged state. The profile counts the completed steps and carries a note.
    pub fn solve_with(&mut self, system: &System, settings: &ScfSettings, warm: bool, should_abort: &mut dyn FnMut() -> bool)
                      -> Result<Option<(&GpuState, ScfStats)>, QmError> {
        match self.run(system, settings, warm, should_abort)? {
            Some(stats) => Ok(Some((self.warm.as_ref().expect("run stores the converged state"), stats))),
            None => Ok(None),
        }
    }

    fn run(&mut self, system: &System, settings: &ScfSettings, warm: bool, should_abort: &mut dyn FnMut() -> bool) -> Result<Option<ScfStats>, QmError> {
        let t0 = Instant::now();
        clear_readback_fault(); // a fault latched by an earlier, unrelated readback is not this solve's
        // spec §5.7 (module doc "Ionic species and occupations (M7)"): τ of DegenerateShells, None under Aufbau
        let shared_tau = match settings.occupations { Occupations::Aufbau => None, Occupations::DegenerateShells { tau } => Some(tau) };
        let prev_converged = self.converged; // the warm start is a converged state (not a parked one)
        self.converged = false;
        // E_nn (valence charges under a setup) and the core overlaps, before anything is touched (coincident nuclei)
        let (e_nn, overlaps) = match system.ionic.as_deref() {
            None => (nuclear_repulsion(&system.nuclei)?, Vec::new()),
            Some(s) => {
                assert!(s.sigma_ae == settings.sigma, "IonicSetup σ_AE {} ≠ settings σ {}", s.sigma_ae, settings.sigma);
                assert!(s.h == self.grid.h, "IonicSetup h {} ≠ grid h {}", s.h, self.grid.h);
                (ionic_repulsion(&system.nuclei, s)?, core_overlaps(s, &system.nuclei))
            }
        };
        let (n_up, n_dn) = (system.n_up, system.n_dn);
        let GpuSolver { ctx, k, r, g, grid, poisson, work, work_k, pool, warm: slot, converged, profile, poisson_tol, eig_floor_factor,
                        eig_fraction, eig_tol_max, extrapolate, prev_rho, ionic: ion_cache, .. } = self;
        let extrapolate = *extrapolate;
        let (poisson_tol, eig_floor_factor) = (*poisson_tol, *eig_floor_factor);
        let (eig_fraction, eig_tol_max) = (*eig_fraction, *eig_tol_max);
        let (ctx, k, r, g): (&GpuContext, &Kernels, &Reducer, &GpuGrid) = (ctx, k, r, g);
        // the ionic terms of this geometry (`GpuIonic`, its buffers reused across solves)
        let gion: Option<&GpuIonic> = match system.ionic.as_deref() {
            None => None,
            Some(s) => {
                match ion_cache.as_mut() { Some(ion) => ion.update(ctx, g, &system.nuclei, s), None => *ion_cache = Some(GpuIonic::new(ctx, g, &system.nuclei, s)) }
                ion_cache.as_ref()
            }
        };
        let nl = gion.and_then(|i| i.nonlocal());
        let eig_tol = effective_eig_tol(settings.eig_tol, eig_floor_factor, g);
        let mut inner = Inner::new(eig_tol);
        let trace = std::env::var_os("SCF_TRACE").is_some(); // per-step diagnostics on stderr
        let mut batch = Batch::new(ctx, "scf");

        // --- starting point ---
        let t_start = Instant::now();
        let prev = slot.take();
        // the warm guard (`qm::scf`): same grid, matching orbital counts, the same occupation scheme and ionic physics
        let reuse = warm && prev.as_ref().is_some_and(|s| s.grid == *grid && counts_match(s, n_up, n_dn, settings.occupations)
                                                         && s.occupations == settings.occupations && same_ionic(s.ionic.as_deref(), system.ionic.as_deref()));
        let (mut orb_up, mut orb_dn, mut ev_up, mut ev_dn, mut rho_in_up, mut rho_in_dn, v_h, v_nuc) = match prev {
            Some(s) if reuse => {
                let GpuState { orbitals_up, orbitals_dn, evals_up, evals_dn, rho, rho_up, rho_dn, v_h, v_nuc, .. } = s;
                pool.give(rho);
                // density extrapolation (`set_density_extrapolation`): ρ_t = this converged
                // output, ρ_{t−1} parked by the solve before; ρ_t becomes the next ρ_{t−1}
                let (rho_up, rho_dn) = match (extrapolate && prev_converged, prev_rho.take()) {
                    (true, Some((pu, pd))) => {
                        let (eu, ed) = (pool.take(ctx, g), pool.take(ctx, g));
                        // 2ρ_t − ρ_{t−1} = ρ_t − (ρ_{t−1} − ρ_t): the difference is exact, one rounding, clamp at 0
                        diis_mix_in(&mut batch, k, g, &eu, &rho_up, &[(-1.0, &pu, true)], true);
                        diis_mix_in(&mut batch, k, g, &ed, &rho_dn, &[(-1.0, &pd, true)], true);
                        pool.give(pu); pool.give(pd);
                        *prev_rho = Some((rho_up, rho_dn));
                        inner.profile.extrapolated = true;
                        (eu, ed)
                    }
                    (true, None) => {
                        let (cu, cd) = (pool.take(ctx, g), pool.take(ctx, g));
                        lincomb_in(&mut batch, k, g, &cu, &[(1.0, &rho_up)]);
                        lincomb_in(&mut batch, k, g, &cd, &[(1.0, &rho_dn)]);
                        *prev_rho = Some((cu, cd));
                        (rho_up, rho_dn)
                    }
                    (false, parked) => {
                        if let Some((pu, pd)) = parked { pool.give(pu); pool.give(pd); }
                        (rho_up, rho_dn)
                    }
                };
                (orbitals_up, orbitals_dn, evals_up, evals_dn, rho_up, rho_dn, v_h, v_nuc)
            }
            other => {
                if let Some(s) = other { pool.give_state(s); }
                if let Some((pu, pd)) = prev_rho.take() { pool.give(pu); pool.give(pd); }
                let total = (n_up + n_dn) as f64;
                // broad atomic-like blobs (σ = 1 Bohr) carrying each centre's charge (Z, or Z_v when pseudised)
                let (z_tot, guess) = match system.ionic.as_deref() {
                    None => (system.nuclei.iter().map(|q| q.z as f64).sum::<f64>(), deposit_charge(grid, &system.nuclei, 1.0)),
                    Some(s) => {
                        let c: Vec<Centre> = centres(&system.nuclei, s).into_iter().map(|c| Centre { sigma: 1.0, ..c }).collect();
                        (c.iter().map(|c| c.charge).sum::<f64>(), deposit_centres(grid, &c))
                    }
                };
                let scale = if z_tot > 0.0 { total / z_tot } else { 0.0 };
                let fu = if total > 0.0 { n_up as f64 / total } else { 0.0 };
                let fd = if total > 0.0 { n_dn as f64 / total } else { 0.0 };
                let upload = |pool: &mut Pool, data: &[f64]| { let f = pool.take(ctx, g); f.upload_interior(ctx, g, data); f };
                let orbitals = |pool: &mut Pool, count: usize, seed: u64| -> Vec<Field> {
                    initial_orbitals(grid, &system.nuclei, count, seed).iter().map(|d| upload(pool, d)).collect()
                };
                // one gap witness per occupied spin channel under DegenerateShells (`qm::scf` "Occupations")
                let extra = |n: usize| usize::from(n > 0 && shared_tau.is_some());
                let (ku, kd) = (n_up + extra(n_up), n_dn + extra(n_dn));
                let orb_up = orbitals(pool, ku, 1);
                let orb_dn = orbitals(pool, kd, 2);
                let rho_up = upload(pool, &guess.iter().map(|q| q * scale * fu).collect::<Vec<_>>());
                let rho_dn = upload(pool, &guess.iter().map(|q| q * scale * fd).collect::<Vec<_>>());
                let v_h = pool.take(ctx, g);
                fill_in(&mut batch, k, g, &v_h, 0.0); // cold CG start
                (orb_up, orb_dn, vec![-1.0; ku], vec![-1.0; kd], rho_up, rho_dn, v_h, pool.take(ctx, g))
            }
        };
        if !reuse { inner.profile.start_wall = t_start.elapsed(); }
        let kmax = orb_up.len().max(orb_dn.len());
        if kmax > 0 && (work.is_none() || *work_k < kmax) {
            *work = Some(LobpcgWork::new(ctx, g, kmax));
            *work_k = kmax;
        }
        match gion {
            None => nuclear_potential_in(&mut batch, k, g, &v_nuc, &system.nuclei, settings.sigma),
            Some(ion) => ion.local_potential_in(&mut batch, k, g, &v_nuc),
        }
        // the partial core (ρ_pc, fixed per geometry) and the two XC input fields ρ_σ + ½ρ_pc; None without a core
        let rho_pc: Option<Field> = gion.filter(|i| i.has_partial_core()).map(|ion| { let f = pool.take(ctx, g); ion.partial_core_in(&mut batch, k, g, &f); f });
        let core_n: Option<(Field, Field)> = rho_pc.as_ref().map(|_| (pool.take(ctx, g), pool.take(ctx, g)));
        let core = rho_pc.as_ref().zip(core_n.as_ref()).map(|(pc, (a, b))| (pc, a, b));
        let (rho_out_up, rho_out_dn, rho_tot) = (pool.take(ctx, g), pool.take(ctx, g), pool.take(ctx, g));
        let (v_up, v_dn, exc, xc_up, xc_dn) = (pool.take(ctx, g), pool.take(ctx, g), pool.take(ctx, g), pool.take(ctx, g), pool.take(ctx, g));
        let mut hist = Diis::new(settings.diis_history);
        let mut frozen = false; // the last step's eigensolves took 0 iterations (module doc)
        // spec §5.7: the occupations (all 1 under Aufbau), the shared clusters of the last step, phase 2's frozen
        // pattern by rank per spin (None = phase 1) and the released clusters
        let (mut occ_up, mut occ_dn) = (vec![1.0; orb_up.len()], vec![1.0; orb_dn.len()]);
        let mut fractional: Vec<FractionalShell> = Vec::new();
        let mut pattern: Option<(Vec<f64>, Vec<f64>)> = None;
        let mut released: Vec<FractionalShell> = Vec::new();

        // --- the loop ---
        let mut iterations = 0;
        let mut delta_rho = f64::INFINITY;
        let mut abandoned = false;
        // broken numerics in a sub-solver (`QmError::Numerical`): ends the loop, then the error path
        let mut failure: Option<QmError> = None;
        // a readback that failed (`grid::map_read_f32` latches it and returns zeros) ends the loop
        // like a sub-solver failure, as `QmError::Device`
        macro_rules! device_fault {
            () => { if let Some(e) = take_device_error() { failure = Some(e); break; } };
        }
        while iterations < settings.max_iter {
            if should_abort() { abandoned = true; break; }
            iterations += 1;
            // the step's eigen tol from the previous step's Δρ (∞ before the first step)
            // a Δρ already below tol buys a tight step, whatever F·Δρ (review I1: for tol > lo/F
            // the rule alone would loosen it again and the convergence guard below never fires)
            let step_tol = if delta_rho < settings.tol { eig_tol } else { eig_tol_for(delta_rho, eig_fraction, eig_tol, eig_tol_max) };
            inner.begin_step(step_tol);
            let eig_it_before = inner.profile.eigen_iterations;
            lincomb_in(&mut batch, k, g, &rho_tot, &[(1.0, &rho_in_up), (1.0, &rho_in_dn)]);
            flush(&mut batch);
            let t = Instant::now();
            match poisson.solve(ctx, k, g, r, &rho_tot, &v_h, poisson_tol, POISSON_MAX_ITER) {
                Ok(ps) => inner.poisson(ps, &format!("SCF step {iterations}"), t.elapsed()),
                Err(e) => { failure = Some(take_device_error().unwrap_or(e)); break; }
            }
            device_fault!();
            xc_in(&mut batch, k, g, &rho_in_up, &rho_in_dn, core, &exc, &xc_up, &xc_dn);
            lincomb_in(&mut batch, k, g, &v_up, &[(1.0, &v_nuc), (1.0, &v_h), (1.0, &xc_up)]);
            lincomb_in(&mut batch, k, g, &v_dn, &[(1.0, &v_nuc), (1.0, &v_h), (1.0, &xc_dn)]);
            flush(&mut batch);
            match shared_tau {
                None => {
                    if n_up > 0 {
                        let h = GpuHamiltonian { v: &v_up, nl };
                        let t = Instant::now();
                        match lobpcg_with_guess_tol(ctx, k, g, r, &h, &mut orb_up, &mut ev_up, work.as_mut().expect("work sized above"), step_tol, EIG_GUESS_FACTOR * step_tol, EIGEN_MAX_ITER, None) {
                            Ok(es) => inner.eigen(es, t.elapsed()),
                            Err(e) => { failure = Some(take_device_error().unwrap_or(e)); break; }
                        }
                        device_fault!();
                    }
                    if n_dn > 0 {
                        let h = GpuHamiltonian { v: &v_dn, nl };
                        let t = Instant::now();
                        match lobpcg_with_guess_tol(ctx, k, g, r, &h, &mut orb_dn, &mut ev_dn, work.as_mut().expect("work sized above"), step_tol, EIG_GUESS_FACTOR * step_tol, EIGEN_MAX_ITER, None) {
                            Ok(es) => inner.eigen(es, t.elapsed()),
                            Err(e) => { failure = Some(take_device_error().unwrap_or(e)); break; }
                        }
                        device_fault!();
                    }
                }
                Some(_) => {
                    let mut shells = Vec::new();
                    let mut err = None;
                    for (spin, n, v, orbs, evals) in [(0u8, n_up, &v_up, &mut orb_up, &mut ev_up), (1, n_dn, &v_dn, &mut orb_dn, &mut ev_dn)] {
                        let by_rank = pattern.as_ref().map(|(u, d)| if spin == 0 { u.as_slice() } else { d.as_slice() });
                        let ch = Channel { ctx, k, g, r, grid, nuclei: &system.nuclei, h: GpuHamiltonian { v, nl }, n, spin, step_tol };
                        match ch.solve_shared(orbs, evals, by_rank, work, work_k, pool, &mut inner) {
                            Ok((f, s)) => { if spin == 0 { occ_up = f } else { occ_dn = f } shells.extend(s); }
                            Err(e) => { err = Some(e); break; }
                        }
                    }
                    if let Some(e) = err { failure = Some(take_device_error().unwrap_or(e)); break; }
                    device_fault!();
                    fractional = shells;
                }
            }
            accumulate_density(&mut batch, k, g, &rho_out_up, &orb_up, shared_tau.map(|_| occ_up.as_slice()));
            accumulate_density(&mut batch, k, g, &rho_out_dn, &orb_dn, shared_tau.map(|_| occ_dn.as_slice()));
            // residuals, Δρ, ∫ρ_out,σ and the new Gram row in one readback
            let (r_up, r_dn) = (pool.take(ctx, g), pool.take(ctx, g));
            lincomb_in(&mut batch, k, g, &r_up, &[(1.0, &rho_out_up), (-1.0, &rho_in_up)]);
            lincomb_in(&mut batch, k, g, &r_dn, &[(1.0, &rho_out_dn), (-1.0, &rho_in_dn)]);
            let (row, q_out) = {
                let mut ops = vec![Reduction::AbsDiffSum(&rho_out_up, &rho_in_up), Reduction::AbsDiffSum(&rho_out_dn, &rho_in_dn),
                                   Reduction::Sum(&rho_out_up), Reduction::Sum(&rho_out_dn)];
                for e in &hist.entries { ops.push(Reduction::Dot(&r_up, &e.r_up)); ops.push(Reduction::Dot(&r_dn, &e.r_dn)); }
                ops.push(Reduction::Dot(&r_up, &r_up));
                ops.push(Reduction::Dot(&r_dn, &r_dn));
                let vals = r.many_in(&mut batch, k, g, &ops);
                inner.profile.readbacks += 1;
                delta_rho = vals[0] + vals[1];
                inner.profile.delta_rho_steps.push(delta_rho);
                if trace { eprintln!("scf step {iterations}: Δρ {delta_rho:.3e} (↑ {:.2e} ↓ {:.2e}), DIIS history {}, eig tol {step_tol:.2e}, eigen last-step residual {:.2e}, eigen it this step {}{}", vals[0], vals[1], hist.entries.len(), inner.step_residual, inner.profile.eigen_iterations - eig_it_before,
                                     if shared_tau.is_some() { format!(", shared {:?}", fractional.iter().map(|c| (c.spin, c.count, c.f, c.spread)).collect::<Vec<_>>()) } else { String::new() }); }
                (vals[4..].chunks(2).map(|c| c[0] + c[1]).collect::<Vec<f64>>(), [vals[2], vals[3]])
            };
            device_fault!(); // before Δρ = 0 from a failed readback reads as convergence
            // converged, or a non-finite Δρ (the error path below)
            // converged only on a step whose eigensolves ran at the tight (M3) tol: the energy and
            // forces are evaluated from these orbitals, so a Δρ that dropped below tol on a
            // loosened step takes one more step (forced tight at the top of the loop)
            let mut released_now = false;
            if delta_rho.is_nan() || (delta_rho < settings.tol && step_tol <= eig_tol) {
                // spec §5.7 phase 2 (`qm::scf`): at the first convergence, release every shared cluster whose spread is
                // ≥ τ, freeze the pattern and converge again; without a release the phase-1 state is the answer
                match (shared_tau, &pattern) {
                    (Some(tau), None) if !delta_rho.is_nan() && fractional.iter().any(|c| c.spread >= tau) => {
                        let (su, sd): (Vec<FractionalShell>, Vec<FractionalShell>) = fractional.iter().cloned().partition(|c| c.spin == 0);
                        let up = decide(&ev_up, &occ_up, &su, tau, &mut released);
                        let dn = decide(&ev_dn, &occ_dn, &sd, tau, &mut released);
                        pattern = Some((up, dn));
                        hist.clear(pool);
                        delta_rho = f64::INFINITY;
                        released_now = true;
                    }
                    _ => { pool.give(r_up); pool.give(r_dn); break; }
                }
            }
            // a frozen orbital set re-solved changes the map ρ_in ↦ ρ_out: the history's residuals
            // were measured on the old map and trap the extrapolation (module doc) — restart it
            let step_iterations = inner.profile.eigen_iterations - eig_it_before;
            let row = if released_now {
                vec![*row.last().expect("the row ends with ⟨r, r⟩")]
            } else if frozen && step_iterations > 0 && !hist.entries.is_empty() {
                hist.clear(pool);
                inner.profile.diis_resets += 1;
                vec![*row.last().expect("the row ends with ⟨r, r⟩")]
            } else { row };
            frozen = step_iterations == 0;
            hist.push(DiisEntry { rho_up: rho_in_up, rho_dn: rho_in_dn, r_up, r_dn }, row, pool);
            rho_in_up = pool.take(ctx, g);
            rho_in_dn = pool.take(ctx, g);
            hist.mix(&mut batch, k, g, settings.mixing, &rho_in_up, &rho_in_dn, pool);
            // charge renormalisation per channel after the clamp: to ∫ρ_out,σ, not to n_σ — the
            // f32-normalised orbitals give ∫ρ_out = n_σ(1 + ε), ε ~ 1e-7, and a fixed point whose
            // input and output charges differ by n_σ ε would floor Δρ at N|ε| (module doc).
            let q_in = r.many_in(&mut batch, k, g, &[Reduction::Sum(&rho_in_up), Reduction::Sum(&rho_in_dn)]);
            inner.profile.readbacks += 1;
            device_fault!();
            for (count, target, sum, f) in [(n_up, q_out[0], q_in[0], &rho_in_up), (n_dn, q_out[1], q_in[1], &rho_in_dn)] {
                if count > 0 && sum > 0.0 && target > 0.0 { scale_in(&mut batch, k, g, f, (target / sum) as f32); }
            }
        }
        hist.clear(pool);
        let ionic_fields = |pool: &mut Pool, rho_pc: Option<Field>, core_n: Option<(Field, Field)>| {
            if let Some(f) = rho_pc { pool.give(f); }
            if let Some((a, b)) = core_n { pool.give(a); pool.give(b); }
        };
        if abandoned {
            // park the in-progress state as the next warm start (`solve_with` doc); `rho` is the
            // sum of the input densities so the parked state has the converged one's shape
            lincomb_in(&mut batch, k, g, &rho_tot, &[(1.0, &rho_in_up), (1.0, &rho_in_dn)]);
            batch.submit(); // with the nuclear potential recorded before the loop
            let mut prof = inner.profile;
            prof.steps = iterations;
            prof.readbacks += prof.poisson_readbacks + prof.eigen_readbacks;
            prof.wall = t0.elapsed();
            prof.notes.push(format!("abandoned before SCF step {} (should_abort)", iterations + 1));
            *profile = prof;
            for f in [rho_out_up, rho_out_dn, v_up, v_dn, exc, xc_up, xc_dn] { pool.give(f); }
            ionic_fields(pool, rho_pc, core_n);
            *slot = Some(GpuState { grid: grid.clone(), sigma: settings.sigma, orbitals_up: orb_up, orbitals_dn: orb_dn,
                                    evals_up: ev_up, evals_dn: ev_dn, rho: rho_tot, rho_up: rho_in_up, rho_dn: rho_in_dn, v_h, v_nuc,
                                    ionic: system.ionic.clone(), occ_up, occ_dn, occupations: settings.occupations });
            return Ok(None);
        }
        // The error paths: broken numerics in a sub-solver (`QmError::Numerical`, M4), or no
        // convergence (a NaN Δρ is an error too, not a convergence). Keep the failed solve's
        // profile (its notes and warnings have no `ScfStats` to live in), return every field to
        // the pool and hold no state — the warm start taken at the top is discarded, so the next
        // solve is cold. The readbacks and wall of a sub-solver call that failed are not counted.
        macro_rules! fail {
            ($err:expr, $steps:expr) => {{
                let err: QmError = take_device_error().unwrap_or($err);
                if let QmError::Numerical(m) = &err { inner.profile.notes.push(format!("numerical failure in SCF step {}: {m}", $steps)); }
                let (_, prof) = inner.finish($steps, t0.elapsed(), Vec::new());
                *profile = prof;
                drop(batch); // a failed sub-solver leaves nothing the SCF needs pending
                for f in [rho_in_up, rho_in_dn, rho_out_up, rho_out_dn, rho_tot, v_up, v_dn, exc, xc_up, xc_dn, v_h, v_nuc].into_iter().chain(orb_up).chain(orb_dn) { pool.give(f); }
                ionic_fields(pool, rho_pc, core_n);
                if let Some((pu, pd)) = prev_rho.take() { pool.give(pu); pool.give(pd); }
                return Err(err);
            }};
        }
        if let Some(e) = failure { fail!(e, iterations); }
        if delta_rho.is_nan() || delta_rho >= settings.tol { fail!(QmError::ScfNotConverged(iterations, delta_rho), iterations); }

        // --- energy from the output density and orbitals (qm::scf module doc) ---
        lincomb_in(&mut batch, k, g, &rho_tot, &[(1.0, &rho_out_up), (1.0, &rho_out_dn)]);
        flush(&mut batch);
        let t = Instant::now();
        match poisson.solve(ctx, k, g, r, &rho_tot, &v_h, poisson_tol, POISSON_MAX_ITER) {
            Ok(ps) => inner.poisson(ps, "final energy", t.elapsed()),
            Err(e) => fail!(e, iterations),
        }
        if let Some(e) = take_device_error() { fail!(e, iterations); }
        xc_in(&mut batch, k, g, &rho_out_up, &rho_out_dn, core, &exc, &xc_up, &xc_dn);
        // T_s = Σ f_i ⟨ψ_i|−½∇²|ψ_i⟩ (f = 1 under Aufbau; f = 0 orbitals skipped, as on the CPU)
        let weighted: Vec<(&Field, f64)> = match shared_tau {
            None => orb_up.iter().chain(&orb_dn).map(|psi| (psi, 1.0)).collect(),
            Some(_) => orb_up.iter().zip(occ_up.iter().copied()).chain(orb_dn.iter().zip(occ_dn.iter().copied())).filter(|t| t.1 != 0.0).collect(),
        };
        let kin: Vec<Field> = (0..weighted.len()).map(|_| pool.take(ctx, g)).collect();
        for ((psi, _), t) in weighted.iter().zip(&kin) { kinetic_in(&mut batch, k, g, psi, t); }
        // the LSDA domain (`qm::scf`): max(−n_σ) > 0 ⇔ some n_σ = ρ_σ + ½ρ_pc < 0 (v_up/v_dn are free scratch here)
        if let Some((_, nu, nd)) = core {
            lincomb_in(&mut batch, k, g, &v_up, &[(-1.0, nu)]);
            lincomb_in(&mut batch, k, g, &v_dn, &[(-1.0, nd)]);
        }
        let (energy, boundary_density, negative) = {
            let mut ops = vec![Reduction::Dot(&rho_tot, &v_nuc), Reduction::Dot(&rho_tot, &v_h), Reduction::Sum(&exc), Reduction::FaceMax(&rho_tot)];
            ops.extend(weighted.iter().zip(&kin).map(|((psi, _), t)| Reduction::Dot(psi, t)));
            if core.is_some() { ops.push(Reduction::Max(&v_up)); ops.push(Reduction::Max(&v_dn)); }
            let vals = r.many_in(&mut batch, k, g, &ops);
            inner.profile.readbacks += 1;
            if let Some(e) = take_device_error() { fail!(e, iterations); }
            let nk = weighted.len();
            let kinetic = match shared_tau {
                None => neumaier(vals[4..4 + nk].iter().copied()),
                Some(_) => neumaier(vals[4..4 + nk].iter().zip(&weighted).map(|(v, (_, f))| f * v)),
            };
            // E_NL = Σ f_iσ Σ E_KB c² (one more readback; `GpuNonlocal::energy_terms`, f64 finish)
            let e_nl = nl.map_or(0.0, |nl| { inner.profile.readbacks += 1; nl.energy_terms(ctx, k, g, &weighted) });
            if let Some(e) = take_device_error() { fail!(e, iterations); }
            let (external, hartree, xc, boundary) = (vals[0] + e_nl, 0.5 * vals[1], vals[2], vals[3]);
            let negative = core.is_some() && (vals[4 + nk] > 0.0 || vals[5 + nk] > 0.0);
            (Energy { kinetic, external, hartree, xc, nuclear: e_nn, total: kinetic + external + hartree + xc + e_nn }, boundary, negative)
        };
        // the clamped points, counted exactly only when some exist (`qm::scf`'s `domain_violations`)
        let clamped = match core {
            Some((_, nu, nd)) if negative => {
                let (u, d) = (nu.read_interior_f32(ctx, g), nd.read_interior_f32(ctx, g));
                u.iter().zip(&d).filter(|(a, b)| **a < 0.0 || **b < 0.0).count()
            }
            _ => 0,
        };
        if let Some(e) = take_device_error() { fail!(e, iterations); } // a failed count would read as no clamped point
        drop(weighted);
        drop(batch); // empty after the last readback
        for f in [rho_in_up, rho_in_dn, v_up, v_dn, exc, xc_up, xc_dn].into_iter().chain(kin) { pool.give(f); }
        ionic_fields(pool, rho_pc, core_n);

        let mut box_warning = Vec::new();
        if boundary_density > BOUNDARY_DENSITY_WARN { box_warning.push(format!("density on box faces is {boundary_density:.2e}: box too small")); }
        let (max_eigen_residual, poisson_residual, poisson_iterations) = (inner.eigen_residual, inner.poisson_residual, inner.poisson_iterations);
        let (mut warnings, mut prof) = inner.finish(iterations, t0.elapsed(), box_warning);
        // the ionic warnings after the solver's, as on the CPU
        for &(i, j, d) in &overlaps { warnings.push(format!("{CORE_OVERLAP_WARNING}: {i}–{j} at R = {d:.3} Bohr")); }
        if clamped > 0 { warnings.push(format!("LSDA domain: ρ_σ + ½ρ_pc < 0 at {clamped} grid points (the XC clamp engages; spec §4.5, T8(c))")); }
        prof.warnings = warnings.clone();
        *profile = prof;
        *slot = Some(GpuState { grid: grid.clone(), sigma: settings.sigma, orbitals_up: orb_up, orbitals_dn: orb_dn,
                                evals_up: ev_up, evals_dn: ev_dn, rho: rho_tot, rho_up: rho_out_up, rho_dn: rho_out_dn, v_h, v_nuc,
                                ionic: system.ionic.clone(), occ_up, occ_dn, occupations: settings.occupations });
        *converged = true;
        Ok(Some(ScfStats { iterations, delta_rho, energy, boundary_density, max_eigen_residual, poisson_residual, poisson_iterations, warnings,
                           core_overlaps: overlaps, fractional, released }))
    }

    /// The LOBPCG tol the SCF converges its final step to (module doc, "Display access").
    pub(crate) fn effective_eig_tol(&self, settings: &ScfSettings) -> f64 {
        effective_eig_tol(settings.eig_tol, self.eig_floor_factor, &self.g)
    }

    /// The kernels, the converged state and this solver's `LobpcgWork` grown to at least `k`
    /// vectors (module doc, "Display access"); None without a converged state.
    pub(crate) fn display_work(&mut self, k: usize) -> Option<DisplayWork<'_>> {
        let GpuSolver { ctx, k: kernels, g, work, work_k, warm, converged, ionic, .. } = self;
        if !*converged || k == 0 {
            return None;
        }
        let state = warm.as_ref()?;
        if work.is_none() || *work_k < k {
            *work = Some(LobpcgWork::new(ctx, g, k));
            *work_k = k;
        }
        // the cached `GpuIonic` is the converged state's geometry and setup: `run` updates it before
        // anything else and a converged state exists only after that run converged; a later
        // all-electron solve leaves a stale cache, hence the gate on the state's own setup
        let ionic = if state.ionic.is_some() { Some(ionic.as_ref().expect("a pseudo state was solved with the ionic terms")) } else { None };
        Some(DisplayWork { k: kernels, state, work: work.as_mut().expect("sized above"), ionic })
    }

    /// The vectors the LOBPCG work space holds now (tests: the display solve grows it).
    #[cfg(test)]
    pub(crate) fn work_capacity(&self) -> usize { if self.work.is_some() { self.work_k } else { 0 } }

    /// Drops the work space a display solve grew (module doc, "Display access"); the next solve
    /// re-creates it at its own size.
    pub(crate) fn release_display_work(&mut self) {
        self.work = None;
        self.work_k = 0;
    }

    pub(crate) fn check_state(&self, state: &GpuState) {
        assert!(state.grid == self.grid, "GpuSolver: state from another grid ({:?} vs {:?})", state.grid.n, self.grid.n);
    }

    /// The state as a CPU `qm::KsState` (f32 → f64, one readback per field; M7 Task 8a): for f64 post-processing of a
    /// GPU solve on the CPU — the frozen-state force probes of `validate`'s pseudo reports, and Task 8b's comparisons.
    pub fn ks_state(&self, state: &GpuState) -> qm::KsState {
        self.check_state(state);
        let read = |f: &Field| f.read_interior(&self.ctx, &self.g);
        qm::KsState {
            orbitals_up: state.orbitals_up.iter().map(read).collect(), orbitals_dn: state.orbitals_dn.iter().map(read).collect(),
            evals_up: state.evals_up.clone(), evals_dn: state.evals_dn.clone(),
            rho_up: read(&state.rho_up), rho_dn: read(&state.rho_dn), v_nuc: read(&state.v_nuc), v_h: read(&state.v_h),
            grid: state.grid.clone(), sigma: state.sigma, ionic: state.ionic.clone(),
            occ_up: state.occ_up.clone(), occ_dn: state.occ_dn.clone(), occupations: state.occupations,
        }
    }

    /// Interior total density as f32 in `qm::Grid` layout (one readback).
    pub fn density_f32(&self, state: &GpuState) -> Vec<f32> {
        self.check_state(state);
        state.rho.read_interior_f32(&self.ctx, &self.g)
    }

    /// Copies the interior density into `volume` (an n³ `Volume3D`) on the GPU: `unpad` into
    /// the row-padded staging buffer, then `copy_buffer_to_texture` with `bytes_per_row` =
    /// 4·stride (stride = n rounded up to a multiple of 64 floats, wgpu's 256-byte row
    /// alignment) and `rows_per_image` = n. No CPU round trip; does not wait.
    pub fn copy_density_to_volume(&self, state: &GpuState, volume: &Volume3D) {
        self.check_state(state);
        let n = self.g.n as u32;
        assert_eq!(volume.dims, [n; 3], "copy_density_to_volume: volume dims {:?} != grid n = {n}", volume.dims);
        let mut enc = encoder(&self.ctx, "density to volume");
        unpad_rec(&mut enc, &self.ctx, &self.k, &self.g, &state.rho, &self.rows, self.row_stride);
        enc.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo { buffer: &self.rows, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * self.row_stride), rows_per_image: Some(n) } },
            wgpu::TexelCopyTextureInfo { texture: &volume.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::Extent3d { width: n, height: n, depth_or_array_layers: n },
        );
        self.ctx.queue.submit([enc.finish()]);
    }

    /// [`copy_density_to_volume`](Self::copy_density_to_volume) of ρ + ρ_pc (M7 Task 9): the density plus the
    /// partial core of the pseudised species, summed on the device into one scratch field first. ρ_pc is the
    /// generator's model of the frozen core's density that the exchange–correlation functional sees (spec §4.5)
    /// — a display of that model at this geometry, not a solved quantity; nothing reads it back into the solve. A
    /// state solved without an ionic setup, or with no partial core, is copied as it is. Returns the maximum of
    /// the field shown (what `density_max` is for the plain copy: the display's normalisation).
    pub fn copy_density_with_core_to_volume(&self, state: &GpuState, volume: &Volume3D) -> f32 {
        self.check_state(state);
        let ion = match (&state.ionic, &self.ionic) {
            (Some(_), Some(ion)) if ion.has_partial_core() => ion,
            _ => {
                self.copy_density_to_volume(state, volume);
                return self.density_max(state);
            }
        };
        let n = self.g.n as u32;
        assert_eq!(volume.dims, [n; 3], "copy_density_with_core_to_volume: volume dims {:?} != grid n = {n}", volume.dims);
        let mut slot = self.core_display.borrow_mut();
        let scratch = slot.get_or_insert_with(|| Field::new(&self.ctx, &self.g, "density + partial core"));
        let mut b = Batch::new(&self.ctx, "density with core");
        ion.partial_core_in(&mut b, &self.k, &self.g, scratch);
        axpy_in(&mut b, &self.k, &self.g, scratch, 1.0, &state.rho);
        b.submit();
        let mut enc = encoder(&self.ctx, "density with core to volume");
        unpad_rec(&mut enc, &self.ctx, &self.k, &self.g, scratch, &self.rows, self.row_stride);
        enc.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo { buffer: &self.rows, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * self.row_stride), rows_per_image: Some(n) } },
            wgpu::TexelCopyTextureInfo { texture: &volume.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::Extent3d { width: n, height: n, depth_or_array_layers: n },
        );
        self.ctx.queue.submit([enc.finish()]);
        self.r.many(&self.ctx, &self.k, &self.g, &[Reduction::Max(scratch)])[0] as f32
    }

    /// max ρ over the interior (two-level max reduction; one readback).
    pub fn density_max(&self, state: &GpuState) -> f32 {
        self.check_state(state);
        self.r.many(&self.ctx, &self.k, &self.g, &[Reduction::Max(&state.rho)])[0] as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qm::{Grid, Nucleus};
    use std::time::Instant;

    fn hydrogen(pos: [f64; 3]) -> System { System::new(vec![Nucleus { pos, z: 1 }], 0, None) }
    fn h2(shift: f64) -> System {
        System::new(vec![Nucleus { pos: [0.7 + shift, 0.0, 0.0], z: 1 }, Nucleus { pos: [-0.7 + shift, 0.0, 0.0], z: 1 }], 0, None)
    }
    fn lithium() -> System { System::new(vec![Nucleus { pos: [0.0; 3], z: 3 }], 0, None) }

    /// A solver on an n³, h = 0.25 cube (σ = h = 0.25, tol 1e-6 from `ScfSettings::for_grid`).
    fn solver(n: usize) -> Option<(GpuSolver, Grid, ScfSettings)> {
        let ctx = GpuContext::new_headless()?;
        let grid = Grid::cube(n, 0.25);
        let settings = ScfSettings::for_grid(&grid);
        Some((GpuSolver::new(ctx, &grid), grid, settings))
    }

    fn cpu(grid: &Grid, system: &System, settings: &ScfSettings) -> (qm::KsState, ScfStats, f64) {
        let t = Instant::now();
        let (state, stats) = qm::scf(grid, system, settings, None).expect("cpu reference converges");
        (state, stats, t.elapsed().as_secs_f64())
    }

    /// Per-component comparison of a GPU energy with the CPU's; prints the table and asserts
    /// the plan bounds: |ΔE_total| < `total_bound`, each component < `comp_bound`.
    fn compare(name: &str, gpu: &ScfStats, cpu: &ScfStats, total_bound: f64, comp_bound: f64) -> f64 {
        let (a, b) = (gpu.energy, cpu.energy);
        let rows = [("kinetic", a.kinetic, b.kinetic), ("external", a.external, b.external), ("hartree", a.hartree, b.hartree),
                    ("xc", a.xc, b.xc), ("nuclear", a.nuclear, b.nuclear), ("total", a.total, b.total)];
        println!("{name}: GPU {} steps (Δρ {:.2e}, eig res {:.2e}, Poisson res {:.2e} / {} it, warnings {:?}) vs CPU {} steps (Δρ {:.2e})",
                 gpu.iterations, gpu.delta_rho, gpu.max_eigen_residual, gpu.poisson_residual, gpu.poisson_iterations, gpu.warnings, cpu.iterations, cpu.delta_rho);
        for (what, g, c) in rows { println!("  {what:>8}: gpu {g:+.8} cpu {c:+.8} Δ {:+.2e}", g - c); }
        for (what, g, c) in &rows[..5] { assert!((g - c).abs() < comp_bound, "{name} {what}: gpu {g} cpu {c}"); }
        let d = (a.total - b.total).abs();
        assert!(d < total_bound, "{name} total: gpu {} cpu {} (|Δ| = {d:.3e})", a.total, b.total);
        d
    }

    /// Per-step eigen tols, compact (diagnostics).
    fn tols(t: &[f64]) -> String { t.iter().map(|v| format!("{v:.1e}")).collect::<Vec<_>>().join(" ") }

    fn charge(s: &GpuSolver, state: &GpuState) -> f64 {
        s.density_f32(state).iter().map(|v| *v as f64).sum::<f64>() * s.grid().dv()
    }

    #[test]
    fn hydrogen_matches_cpu_at_fixed_sigma() {
        let Some((mut s, grid, settings)) = solver(33) else { return };
        let sys = hydrogen([0.0; 3]);
        let (_, cpu_stats, cpu_s) = cpu(&grid, &sys, &settings);
        let t = Instant::now();
        let stats = s.solve(&sys, &settings, false).unwrap().1;
        let gpu_s = t.elapsed().as_secs_f64();
        let state = s.state().unwrap();
        let q = charge(&s, state);
        println!("H 33³ σ = 0.25: GPU {:.2} s ({:?}), CPU {cpu_s:.2} s; ∫ρ = {q:.9}", gpu_s, s.profile());
        let d = compare("H 33³", &stats, &cpu_stats, 1e-5, 2e-5); // brief 1e-4 / 2e-4; achieved 9.6e-8 / ≤ 3.1e-7
        println!("H 33³: |ΔE| = {d:.2e}");
        assert!((q - 1.0).abs() < 1e-5, "∫ρ = {q}");
        assert_eq!(state.orbitals_dn.len(), 0);
        assert!(stats.delta_rho < settings.tol);
    }

    /// H₂ 33³ against the CPU at tol 1e-6 (energy components, charge, DIIS steps within ±3 —
    /// a real CPU-agreement check at that tol) and at tol 1e-7: converged and the same energy.
    /// At tol 1e-7 the *step count* is not asserted (change of measure, M4 review/controller
    /// ruling): that tol sits on the mixing's f32 Δρ plateau (≈ 1e-7, frozen orbitals), where
    /// the step count is a coin flip between ~11 and ~19 across rounding-level solver variants
    /// (M3 Task 10 note; M4 measured 13–19 over eigen-tol fractions 0.001–0.005). The cost
    /// witness there is the readback count: a Δρ floor or a crawl would run to many times the
    /// tol-1e-6 count, a plateau coin flip costs a few steps (~35 readbacks each). No energy or
    /// convergence bound changed.
    #[test]
    fn h2_matches_cpu_at_fixed_sigma() {
        let Some((mut s, grid, settings)) = solver(33) else { return };
        let sys = h2(0.0);
        let (_, cpu_stats, cpu_s) = cpu(&grid, &sys, &settings);
        let t = Instant::now();
        let stats = s.solve(&sys, &settings, false).unwrap().1;
        let gpu_s = t.elapsed().as_secs_f64();
        let state = s.state().unwrap();
        let q = charge(&s, state);
        println!("H₂ 33³ R = 1.4: GPU {:.2} s ({:?}), CPU {cpu_s:.2} s; ∫ρ = {q:.9}", gpu_s, s.profile());
        let d = compare("H₂ 33³", &stats, &cpu_stats, 1e-5, 2e-5); // brief 1e-4 / 2e-4; achieved 9.1e-8 / ≤ 2.4e-6
        println!("H₂ 33³: |ΔE| = {d:.2e}");
        assert!((q - 2.0).abs() < 1e-5, "∫ρ = {q}");
        assert!((stats.iterations as i64 - cpu_stats.iterations as i64).abs() <= 3, "DIIS steps: gpu {} cpu {}", stats.iterations, cpu_stats.iterations);
        let s_readbacks_1e6 = s.profile().readbacks;
        println!("H₂ 33³: Δρ_final = {:.2e} (tol {:.0e}), final-step eigen residual {:.2e}", stats.delta_rho, settings.tol, s.profile().eigen_last_step_residual);
        // No Δρ floor planted by the charge bookkeeping: ρ_in is renormalised to ∫ρ_out, not to N,
        // so a fixed point with ∫ρ_out = N(1 + ε), ε ≈ −5e-8 per channel here, does not force
        // Δρ ≥ N|ε| ≈ 1e-7. That would-be floor sits AT tol 1e-7 for H₂, so this is a no-floor
        // sanity check, not a discriminator (Task 8 measured the old n_σ target passing it too:
        // 11 steps, Δρ_final 9.4e-8 vs 8.8e-8); the discriminating check is H₂O's step count in
        // `validate::gpu_vs_cpu::energies_match_at_fixed_sigma` (27 vs 17 steps). A cold solve at
        // tol 1e-7 must converge, in about the CPU's step count. (A warm continuation is not a
        // test: with the same geometry LOBPCG takes 0
        // iterations, ρ_out ≡ ρ_in and Δρ = 0 exactly.)
        let mut tight = ScfSettings::for_grid(&grid);
        tight.tol = 1e-7;
        let (_, cpu_tight, cpu_tight_s) = cpu(&grid, &sys, &tight);
        let t = Instant::now();
        let tight_stats = s.solve(&sys, &tight, false).unwrap_or_else(|e| panic!("H₂ at tol 1e-7 did not converge (a Δρ floor): {e}")).1;
        println!("H₂ 33³ cold at tol 1e-7: GPU {} steps, {:.0} ms, Δρ_final {:.2e}, E {:+.9}; CPU {} steps, {cpu_tight_s:.2} s, Δρ_final {:.2e}, E {:+.9} (|ΔE| {:.2e})",
                 tight_stats.iterations, t.elapsed().as_secs_f64() * 1e3, tight_stats.delta_rho, tight_stats.energy.total,
                 cpu_tight.iterations, cpu_tight.delta_rho, cpu_tight.energy.total, (tight_stats.energy.total - cpu_tight.energy.total).abs());
        let (rb_6, rb_7) = (s_readbacks_1e6, s.profile().readbacks);
        println!("H₂ 33³ readbacks: tol 1e-6 {rb_6}, tol 1e-7 {rb_7} (steps GPU {} vs CPU {} — reported, not asserted)", tight_stats.iterations, cpu_tight.iterations);
        assert!(tight_stats.delta_rho < 1e-7);
        // measured 484 vs 443 at tol 1e-6 (M4); bound 2×: a plateau coin flip (~+6 steps) stays inside
        assert!(rb_7 <= 2 * rb_6, "tol 1e-7 took {rb_7} readbacks vs {rb_6} at tol 1e-6: a Δρ floor or a crawl");
        assert!((tight_stats.energy.total - cpu_tight.energy.total).abs() < 1e-5);
    }

    /// Two cold solves on the same solver, and one on a fresh solver, give bit-identical
    /// energies and iteration counts (Review Focus 1: fixed reduction order, no timing-dependent
    /// state).
    #[test]
    fn solve_is_deterministic() {
        let Some((mut s, grid, settings)) = solver(33) else { return };
        let sys = h2(0.0);
        let e1 = { let (_, st) = s.solve(&sys, &settings, false).unwrap(); st };
        let e2 = { let (_, st) = s.solve(&sys, &settings, false).unwrap(); st };
        let mut fresh = GpuSolver::new(s.ctx.clone(), &grid);
        let e3 = { let (_, st) = fresh.solve(&sys, &settings, false).unwrap(); st };
        println!("deterministic: {:+.12} / {:+.12} / {:+.12} ({} / {} / {} steps)", e1.energy.total, e2.energy.total, e3.energy.total, e1.iterations, e2.iterations, e3.iterations);
        for (name, a, b) in [("same solver", &e1, &e2), ("fresh solver", &e1, &e3)] {
            assert_eq!(a.energy.total.to_bits(), b.energy.total.to_bits(), "{name}: {} vs {}", a.energy.total, b.energy.total);
            assert_eq!(a.energy.kinetic.to_bits(), b.energy.kinetic.to_bits(), "{name} kinetic");
            assert_eq!(a.energy.hartree.to_bits(), b.energy.hartree.to_bits(), "{name} hartree");
            assert_eq!(a.delta_rho.to_bits(), b.delta_rho.to_bits(), "{name} Δρ");
            assert_eq!(a.iterations, b.iterations, "{name} steps");
        }
    }

    /// As the CPU test: a warm start on the displaced geometry is cheaper than a cold solve of
    /// the same geometry and lands on the same energy. "Cheaper" is measured in readbacks (the
    /// deterministic cost), not SCF steps (change of measure, M4 review/controller ruling): near
    /// the f32 Δρ plateau the step count of an H solve is a ±1 coin flip between rounding-level
    /// variants (M3 Task 10 note; M4 measured warm 7 vs cold 7 steps at 180 vs 245 readbacks for
    /// one eigen-tol fraction, 6 vs 7 at 168 vs 244 for the chosen one). Steps are printed.
    #[test]
    fn warm_start_is_cheaper_and_agrees() {
        let Some((mut s, _, settings)) = solver(33) else { return };
        let (_, cold) = s.solve(&hydrogen([0.0; 3]), &settings, false).unwrap();
        let cold_prof = s.profile().clone();
        let (_, warm) = s.solve(&hydrogen([0.02, 0.0, 0.0]), &settings, true).unwrap();
        let warm_prof = s.profile().clone();
        let (_, cold_disp) = s.solve(&hydrogen([0.02, 0.0, 0.0]), &settings, false).unwrap();
        let disp_prof = s.profile().clone();
        println!("warm start 33³: cold {} steps {:.0} ms, warm {} steps {:.0} ms (E {:+.8}), cold displaced {} steps {:.0} ms (E {:+.8}); readbacks {} / {} / {}",
                 cold.iterations, cold_prof.wall.as_secs_f64() * 1e3, warm.iterations, warm_prof.wall.as_secs_f64() * 1e3, warm.energy.total,
                 cold_disp.iterations, disp_prof.wall.as_secs_f64() * 1e3, cold_disp.energy.total, cold_prof.readbacks, warm_prof.readbacks, disp_prof.readbacks);
        assert!(warm_prof.readbacks < disp_prof.readbacks, "warm {} vs cold {} readbacks", warm_prof.readbacks, disp_prof.readbacks);
        assert!((warm.energy.total - cold_disp.energy.total).abs() < 1e-5, "warm {} vs cold {}", warm.energy.total, cold_disp.energy.total);
        // a warm request with mismatched counts falls back to the cold path
        let (_, h2_after_h) = s.solve(&h2(0.0), &settings, true).unwrap();
        let (_, h2_cold) = s.solve(&h2(0.0), &settings, false).unwrap();
        assert_eq!(h2_after_h.energy.total.to_bits(), h2_cold.energy.total.to_bits(), "mismatched warm state was not ignored");
    }

    /// H (n↓ = 0: the down channel is skipped) and Li (n↑ = 2, n↓ = 1) converge; Li's energy
    /// matches the CPU on the 49³ grid of `qm::scf`'s Li test.
    #[test]
    fn empty_channel_and_lithium_run() {
        let Some((mut s, grid, settings)) = solver(49) else { return };
        let h = s.solve(&hydrogen([0.0; 3]), &settings, false).unwrap().1;
        let state = s.state().unwrap();
        assert!(h.delta_rho < settings.tol);
        assert!(state.orbitals_dn.is_empty() && state.evals_dn.is_empty());
        let dn: Vec<f32> = state.rho_dn.read_interior_f32(&s.ctx, &s.g);
        assert!(dn.iter().all(|v| *v == 0.0), "empty channel density must be exactly 0");
        println!("H 49³: {} steps, E {:+.8}, ∫ρ {:.9}, {:?}", h.iterations, h.energy.total, charge(&s, state), s.profile());
        let li = lithium();
        let (_, cpu_stats, cpu_s) = cpu(&grid, &li, &settings);
        let t = Instant::now();
        let stats = s.solve(&li, &settings, false).unwrap().1;
        let gpu_s = t.elapsed().as_secs_f64();
        let state = s.state().unwrap();
        let q = charge(&s, state);
        println!("Li 49³: GPU {:.2} s ({:?}), CPU {cpu_s:.2} s; ∫ρ = {q:.9}; evals up {:?} dn {:?}", gpu_s, s.profile(), state.evals_up, state.evals_dn);
        assert_eq!((state.orbitals_up.len(), state.orbitals_dn.len()), (2, 1));
        assert!(stats.delta_rho < settings.tol);
        assert!((q - 3.0).abs() < 1e-5, "∫ρ = {q}");
        let d = compare("Li 49³", &stats, &cpu_stats, 5e-5, 1e-4); // brief 5e-4; achieved 3.6e-7 / ≤ 3.6e-6
        let p = s.profile().clone();
        println!("Li 49³: |ΔE| = {d:.2e}; Δρ_final {:.2e}; floor accepts {} (worst {:.2e}), last-step eigen residual {:.2e}, notes {:?}",
                 stats.delta_rho, p.eigen_floor_accepts, p.eigen_floor_worst, p.eigen_last_step_residual, p.notes);
        // floor accepts are informational: in the profile (with a note), never in `warnings`
        // fragile by nature (M4 review M12): the Δρ-tied tol loosens the early solves, so fewer
        // of them reach the floor — measured 5 of 44 (M3), 1–2 of 42 (M4)
        assert!(p.eigen_floor_accepts >= 1, "Li's 2s/2p near-degeneracy ends some eigensolves on the floor (measured 5 of 44 in M3, 1–2 in M4)");
        assert!(p.eigen_floor_worst > 0.0 && p.notes.iter().any(|n| n.contains("floor")), "{p:?}");
        assert!(stats.warnings.iter().all(|w| !w.to_lowercase().contains("floor")), "floor accept leaked into warnings: {:?}", stats.warnings);
        assert!(p.eigen_last_step_residual > 0.0 && p.eigen_last_step_residual <= stats.max_eigen_residual);
        // early cold eigensolves that hit the 200-iteration cap on the 2s/2p pair (measured 2 of
        // 44, residual 1.5e-3 on this driver — a solver weakness, not pinned) are a note, not a
        // warning (ruled: only the final step warns); the ruling itself is pinned on `Inner` in
        // `lobpcg_cap_warns_only_on_the_final_step`
        println!("Li 49³: LOBPCG cap hits {} (worst {:.2e})", p.eigen_cap_hits, p.eigen_cap_worst);
        if p.eigen_cap_hits > 0 {
            assert!(p.notes.iter().any(|n| n.contains("cap")), "cap hits on non-final steps must be noted: {p:?}");
            // cap hits can sit on loosened early steps, which `max_eigen_residual` excludes (review I2)
            assert!(p.eigen_cap_worst > 0.0 && p.eigen_cap_worst <= p.eigen_worst_residual_any_step, "{p:?}");
        }
        assert!(stats.warnings.iter().all(|w| !w.contains("LOBPCG")), "a non-final-step cap hit (or none) leaked into warnings: {:?}", stats.warnings);
        assert_eq!(p.warnings, stats.warnings);
    }

    /// The final-step rule, on `Inner` directly (no GPU): a `MaxIter` exit on a non-final step
    /// is a note; a `MaxIter` exit on the final step, or a final-step residual above
    /// `LAST_STEP_RESIDUAL_FACTOR` × tol, is a warning; a converged final step after an early
    /// cap hit is clean. `finish` also fills `steps`, `readbacks`, `wall` and `warnings`.
    #[test]
    fn lobpcg_cap_warns_only_on_the_final_step() {
        let tol = 1e-6;
        let es = |stop: EigenStop, res: f64| EigenStats { iterations: 7, max_residual: res, stop, readbacks: 15, dispatches: 80, fallbacks: 0, alpha_worst: 1.0 };
        let ms = Duration::from_millis(1);
        // early cap hit, final step converged: note, no warning
        let mut a = Inner::new(tol);
        a.begin_step(tol); a.eigen(es(EigenStop::MaxIter, 1.5e-3), ms); a.eigen(es(EigenStop::Converged, 8e-7), ms);
        a.begin_step(tol); a.eigen(es(EigenStop::Converged, 9e-7), ms); a.eigen(es(EigenStop::Floor, 2.5e-6), ms);
        let (w, p) = a.finish(2, Duration::from_millis(40), vec!["box".into()]);
        assert_eq!(w, vec!["box".to_string()], "{w:?}");
        assert_eq!(p.warnings, w);
        assert!(p.notes.iter().any(|n| n.contains("cap in 1 of 4 calls") && n.contains("1.50e-3")), "{:?}", p.notes);
        assert_eq!((p.eigen_cap_hits, p.eigen_cap_worst), (1, 1.5e-3));
        assert!(p.notes.iter().any(|n| n.contains("floor in 1 of 4 calls")), "{:?}", p.notes);
        assert_eq!((p.steps, p.eigen_readbacks, p.readbacks, p.eigen_iterations), (2, 60, 60, 28));
        assert_eq!((p.wall, p.eigen_wall), (Duration::from_millis(40), Duration::from_millis(4)));
        assert_eq!(p.eigen_last_step_residual, 2.5e-6);
        // cap hit on the final step: warning (and still the note)
        let mut b = Inner::new(tol);
        b.begin_step(tol); b.eigen(es(EigenStop::Converged, 8e-7), ms);
        b.begin_step(tol); b.eigen(es(EigenStop::MaxIter, 4e-5), ms);
        let (w, p) = b.finish(2, ms, Vec::new());
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].contains("final SCF step 2") && w[0].contains("4.00e-5") && w[0].contains("cap"), "{w:?}");
        assert!(p.notes.iter().any(|n| n.contains("cap in 1 of 2")));
        // final-step residual above 3× tol without a cap hit (a wide floor accept): warning
        let mut c = Inner::new(tol);
        c.begin_step(tol); c.eigen(es(EigenStop::Floor, 3.5e-6), ms);
        let (w, _) = c.finish(1, ms, Vec::new());
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].contains("final SCF step 1") && !w[0].contains("cap"), "{w:?}");
        // at exactly 3× tol: no warning; NaN: warning
        let mut d = Inner::new(tol);
        d.begin_step(tol); d.eigen(es(EigenStop::Floor, LAST_STEP_RESIDUAL_FACTOR * tol), ms);
        assert!(d.finish(1, ms, Vec::new()).0.is_empty());
        let mut e = Inner::new(tol);
        e.begin_step(tol); e.eigen(es(EigenStop::Floor, f64::NAN), ms);
        assert_eq!(e.finish(1, ms, Vec::new()).0.len(), 1);
    }

    /// A solve that fails (`ScfNotConverged` from a 2-step cap) keeps its profile — steps,
    /// readbacks, walls — and leaves the solver stateless; the next solve runs cold on the
    /// returned pool fields and is bit-identical to a fresh cold solve.
    #[test]
    fn failed_solve_keeps_profile_and_pool() {
        let Some((mut s, grid, settings)) = solver(33) else { return };
        let sys = hydrogen([0.0; 3]);
        let mut capped = ScfSettings::for_grid(&grid);
        capped.max_iter = 2;
        match s.solve(&sys, &capped, false) {
            Err(QmError::ScfNotConverged(2, d)) => assert!(d.is_finite() && d > capped.tol, "Δρ after 2 steps {d}"),
            other => panic!("expected ScfNotConverged(2, _), got {:?}", other.map(|(_, st)| st)),
        }
        let p = s.profile().clone();
        println!("failed solve (2-step cap) profile: {p:?}");
        assert_eq!(p.steps, 2);
        assert!(p.readbacks > 0 && p.poisson_readbacks > 0 && p.eigen_readbacks > 0);
        assert!(p.wall > Duration::ZERO && p.poisson_wall > Duration::ZERO && p.eigen_wall > Duration::ZERO && p.start_wall > Duration::ZERO);
        assert!(p.wall > p.poisson_wall + p.eigen_wall + p.start_wall, "phase walls must not exceed the total: {p:?}");
        assert!(s.state().is_none(), "a failed solve must leave no state");
        let (_, after) = s.solve(&sys, &settings, false).unwrap();
        let mut fresh = GpuSolver::new(s.ctx.clone(), &grid);
        let (_, reference) = fresh.solve(&sys, &settings, false).unwrap();
        assert_eq!(after.energy.total.to_bits(), reference.energy.total.to_bits(), "{} vs {}", after.energy.total, reference.energy.total);
        let p = s.profile();
        assert_eq!(p.steps, after.iterations);
        assert_eq!(p.warnings, after.warnings);
        assert!(p.start_wall > Duration::ZERO, "cold start must be timed");
        let (_, warm) = s.solve(&sys, &settings, true).unwrap();
        assert_eq!(s.profile().start_wall, Duration::ZERO, "a warm start has no guess to build");
        assert!(warm.delta_rho < settings.tol);
    }

    /// `eig_tol_for` on its own (no GPU): the first step and `fraction` = 0 use the M3 tol `lo`;
    /// later steps follow fraction × Δρ_prev between `lo` and the cap; a cap below `lo` is `lo`.
    #[test]
    fn eig_tol_for_clamps_the_delta_rho_rule() {
        let lo = 5.4e-6;
        assert_eq!(eig_tol_for(f64::INFINITY, EIG_TOL_FRACTION, lo, EIG_TOL_MAX), lo);
        assert_eq!(eig_tol_for(0.3, 0.0, lo, EIG_TOL_MAX), lo);
        assert_eq!(eig_tol_for(0.7, 0.002, lo, 1e-3), 1e-3);
        assert!((eig_tol_for(0.05, 0.002, lo, 1e-3) - 1e-4).abs() < 1e-18);
        assert_eq!(eig_tol_for(1e-3, 0.002, lo, 1e-3), lo);
        assert_eq!(eig_tol_for(0.05, 0.002, lo, 1e-6), lo);
        assert_eq!(eig_tol_for(f64::NAN, 0.002, lo, 1e-3), lo);
    }

    /// Review I1: a Δρ below `settings.tol` must always buy a tight-tol step, whatever the
    /// SCF tol. Before the fix, `settings.tol` > lo/F (≈ 3.6e-3 at h = 0.25) let F·Δρ stay above
    /// lo after Δρ < tol, so every following step was loosened again and the convergence guard
    /// never fired (→ `ScfNotConverged` at `max_iter`). Case (a): the defaults at a deliberately
    /// loose tol 5e-3 on H 25³ (the window is not hit there by chance); case (b): F = 1e3 keeps every
    /// step at the loose cap for any Δρ > 1e-6, so a Δρ < tol 1e-2 can only converge through the fix.
    #[test]
    fn loose_scf_tol_still_converges_on_a_tight_step() {
        let Some((mut s, grid, _)) = solver(25) else { return };
        let lo = ScfSettings::for_grid(&grid).eig_tol.max(EIG_TOL_FLOOR_FACTOR * residual_floor_estimate(&s.g));
        for (label, fraction, tol) in [("defaults, tol 5e-3", EIG_TOL_FRACTION, 5e-3), ("F = 1e3, tol 1e-2", 1e3, 1e-2)] {
            s.set_eig_knobs(fraction, EIG_TOL_MAX);
            let mut st = ScfSettings::for_grid(&grid);
            st.tol = tol;
            st.max_iter = 30;
            let (_, stats) = s.solve(&hydrogen([0.0; 3]), &st, false).unwrap_or_else(|e| panic!("{label}: {e}"));
            let p = s.profile().clone();
            println!("H 25³ {label}: {} steps, Δρ per step {}, eig tol per step {}", stats.iterations, tols(&p.delta_rho_steps), tols(&p.eig_tol_steps));
            assert_eq!(p.eig_tol_used, lo, "{label}: the converged step must run at the tight tol");
            // the first step with Δρ < tol is followed by exactly one more (tight) step, or is the last
            let first = p.delta_rho_steps.iter().position(|d| *d < tol).expect("converged");
            assert!(p.steps <= first + 2, "{label}: Δρ < tol at step {} but the solve took {} steps", first + 1, p.steps);
            if first + 1 < p.steps { assert_eq!(p.eig_tol_steps[first + 1], lo, "{label}: the step after Δρ < tol must be tight"); }
        }
    }

    /// M4 orbital lag (`docs/superpowers/notes/2026-10-13-m3-task10-orbital-lag.md`) on the app's
    /// witness (`app::sim::moving_a_nucleus_warm_starts`: H₂ 49³, nucleus 0 moved 0.70 → 0.72
    /// Bohr, tol 1e-5): with the M3 fixed eigen tol the warm solve froze its orbitals at step 6,
    /// re-solved them at step 7 (Δρ jumped 1.7e-5 → 8.6e-5) and crawled — 15 steps / 553
    /// readbacks vs 9 / 593 for a cold solve of the displaced geometry. With the Δρ-tied tol
    /// (`eig_tol_for`) the warm solve takes no more steps than the cold one and clearly fewer
    /// readbacks, lands on the same energy, and its per-step tols follow the rule (first and
    /// last step at the M3 tol, loosened in between, never above `EIG_TOL_MAX`).
    #[test]
    fn eigen_tol_follows_delta_rho_and_removes_orbital_lag() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        let grid = Grid::cube(49, 0.25);
        let mut settings = ScfSettings::for_grid(&grid);
        settings.tol = 1e-5; // the viewer's interactive tol
        let at = |x0: f64| System::new(vec![Nucleus { pos: [x0, 0.0, 0.0], z: 1 }, Nucleus { pos: [-0.7, 0.0, 0.0], z: 1 }], 0, None);
        let mut s = GpuSolver::new(ctx, &grid);
        let lo = settings.eig_tol.max(EIG_TOL_FLOOR_FACTOR * residual_floor_estimate(&s.g));
        s.solve(&at(0.70), &settings, false).unwrap();
        let (_, warm) = s.solve(&at(0.72), &settings, true).unwrap();
        let pw = s.profile().clone();
        let (_, cold) = s.solve(&at(0.72), &settings, false).unwrap();
        let pc = s.profile().clone();
        println!("orbital lag witness (H₂ 49³ 0.70 → 0.72, tol 1e-5): warm {} steps / {} readbacks (LOBPCG {} it), cold of the displaced {} steps / {} readbacks (LOBPCG {} it); |ΔE| {:.2e}",
                 warm.iterations, pw.readbacks, pw.eigen_iterations, cold.iterations, pc.readbacks, pc.eigen_iterations, (warm.energy.total - cold.energy.total).abs());
        println!("  eig tol per step: warm {}; cold {}", tols(&pw.eig_tol_steps), tols(&pc.eig_tol_steps));
        assert!(warm.iterations <= cold.iterations, "warm {} steps vs cold {}: the orbital lag is back", warm.iterations, cold.iterations);
        // M3: 553 / 593 = 0.93; Δρ-tied tol (measured on the RX 7900 XTX): 365 / 495 = 0.74 here
        // (the app's R13 path, `moving_a_nucleus_warm_starts`, gives 381 / 477 = 0.80). One
        // extra warm step is ≈ +55 readbacks (→ 0.85, borderline), so on another adapter the
        // ratio is robust to one step only; `steps ≤` above is the primary lag witness (M7)
        assert!((pw.readbacks as f64) < 0.85 * pc.readbacks as f64, "warm {} vs cold {} readbacks", pw.readbacks, pc.readbacks);
        assert!((warm.energy.total - cold.energy.total).abs() < 1e-5, "warm {} vs cold {}", warm.energy.total, cold.energy.total);
        // review I2: `max_eigen_residual` describes the tight-tol solves the state rests on (M3
        // meaning), the loosened early steps only the profile's any-step worst
        for (st, p) in [(&warm, &pw), (&cold, &pc)] {
            println!("  max_eigen_residual {:.2e}, eigen_worst_residual_any_step {:.2e}, last step {:.2e}", st.max_eigen_residual, p.eigen_worst_residual_any_step, p.eigen_last_step_residual);
            assert!(st.max_eigen_residual <= LAST_STEP_RESIDUAL_FACTOR * lo, "max_eigen_residual {:.2e} leaks a loosened step (lo {lo:.2e})", st.max_eigen_residual);
            assert!(st.max_eigen_residual >= p.eigen_last_step_residual && p.eigen_worst_residual_any_step >= st.max_eigen_residual);
        }
        assert!(pc.eigen_worst_residual_any_step > 10.0 * lo, "the cold solve's loosened steps are visible in the profile");
        for p in [&pw, &pc] {
            assert_eq!(p.eig_tol_steps.len(), p.steps);
            assert_eq!(p.eig_tol_steps[0], lo, "the first step uses the M3 tol");
            assert_eq!(p.eig_tol_used, *p.eig_tol_steps.last().unwrap());
            assert_eq!(p.eig_tol_used, lo, "the converged step runs at the M3 tol (the convergence guard)");
            assert!(p.eig_tol_steps.iter().all(|t| *t >= lo && *t <= EIG_TOL_MAX), "{:?}", p.eig_tol_steps);
            assert!(p.eig_tol_steps.iter().any(|t| *t > lo), "the early steps must be loosened: {:?}", p.eig_tol_steps);
        }
    }

    /// Density extrapolation (`set_density_extrapolation`) along an MD-like path: H₂ 33³ at
    /// R = 1.40, 1.41, … 1.49 (nucleus 0 moving in equal steps), each solve warm from the last.
    /// With ρ_in = 2ρ_t − ρ_{t−1} the warm solves take at least one SCF step fewer on average and
    /// land on the same energies (to 1e-6, the f32 path scatter is ~1e-7 here). The first warm
    /// solve has no ρ_{t−1} yet and is identical either way.
    #[test]
    fn density_extrapolation_cuts_warm_steps_on_a_smooth_path() {
        let Some((mut plain, grid, settings)) = solver(33) else { return };
        let mut extr = GpuSolver::new(plain.ctx.clone(), &grid);
        extr.set_density_extrapolation(true);
        let path: Vec<System> = (0..10).map(|i| {
            let r = 1.40 + 0.01 * i as f64;
            System::new(vec![Nucleus { pos: [r - 0.7, 0.0, 0.0], z: 1 }, Nucleus { pos: [-0.7, 0.0, 0.0], z: 1 }], 0, None)
        }).collect();
        let (mut steps_plain, mut steps_extr) = (Vec::new(), Vec::new());
        let (mut rb_plain, mut rb_extr) = (0usize, 0usize);
        let mut worst = 0.0f64;
        for (i, sys) in path.iter().enumerate() {
            let (_, a) = plain.solve(sys, &settings, i > 0).unwrap();
            let pa = plain.profile().clone();
            let (_, b) = extr.solve(sys, &settings, i > 0).unwrap();
            let pb = extr.profile().clone();
            assert!(!pa.extrapolated);
            assert_eq!(pb.extrapolated, i >= 2, "step {i}: extrapolation needs two converged solves before it");
            if i > 0 {
                steps_plain.push(a.iterations); steps_extr.push(b.iterations);
                rb_plain += pa.readbacks; rb_extr += pb.readbacks;
            }
            worst = worst.max((a.energy.total - b.energy.total).abs());
        }
        let mean = |v: &[usize]| v.iter().sum::<usize>() as f64 / v.len() as f64;
        let (mp, me) = (mean(&steps_plain), mean(&steps_extr));
        println!("H₂ 33³ R = 1.40 … 1.49, warm solves: steps without extrapolation {steps_plain:?} (mean {mp:.2}, {rb_plain} readbacks), with {steps_extr:?} (mean {me:.2}, {rb_extr} readbacks); max |ΔE| {worst:.2e}");
        assert!(me <= mp - 1.0, "extrapolation must save ≥ 1 SCF step per warm solve: {me:.2} vs {mp:.2}");
        assert!(worst < 1e-6, "energies with and without extrapolation differ by {worst:.2e}");
        // turning it off drops ρ_{t−1}: the next warm solve is a plain one
        extr.set_density_extrapolation(false);
        extr.solve(&path[9], &settings, true).unwrap();
        assert!(!extr.profile().extrapolated);
    }

    /// A non-finite state is `Err(QmError::Numerical)`, not a panic (M4: the dynamics loop must
    /// be able to react without rebuilding the solver). A NaN injected into the parked warm ρ_in
    /// reaches the first Poisson solve (its ‖b‖ readback is NaN). The failed solve discards the
    /// parked state — a following warm request runs cold — keeps its profile with a note, and the
    /// next solve is bit-identical to a fresh solver's cold solve (the reference energy).
    #[test]
    fn nan_density_returns_numerical_error_not_panic() {
        let Some((mut s, grid, settings)) = solver(33) else { return };
        let sys = hydrogen([0.0; 3]);
        s.solve(&sys, &settings, false).unwrap();
        s.poison_parked_density();
        match s.solve_with(&sys, &settings, true, &mut || false) {
            Err(QmError::Numerical(msg)) => println!("NaN in the warm density: Err(Numerical(\"{msg}\"))"),
            other => panic!("expected Err(Numerical), got {:?}", other.map(|o| o.map(|(_, st)| st))),
        }
        assert!(s.state().is_none(), "a failed solve must expose no state");
        let p = s.profile().clone();
        println!("failed solve profile: {p:?}");
        assert_eq!(p.steps, 1, "the NaN is caught inside the first SCF step");
        assert!(p.notes.iter().any(|n| n.contains("numerical failure")), "the failure must be noted in the profile: {:?}", p.notes);
        // the parked state was discarded: a warm request runs cold
        let (_, after) = s.solve(&sys, &settings, true).unwrap();
        assert!(s.profile().start_wall > Duration::ZERO, "the warm request after a failure must run cold");
        let mut fresh = GpuSolver::new(s.ctx.clone(), &grid);
        let (_, reference) = fresh.solve(&sys, &settings, false).unwrap();
        assert_eq!(after.energy.total.to_bits(), reference.energy.total.to_bits(), "{} vs {}", after.energy.total, reference.energy.total);
    }

    /// A readback the device does not deliver (the dropped map callback behind the user's
    /// `RecvError` panic of 2026-10-17) is `Err(QmError::Device)` wherever in the solve it
    /// happens — the first Poisson solve, an eigensolve, the Δρ reduction, the energy — never a
    /// panic and never a success built on the zeros the failed readback returned (a zero Δρ would
    /// read as convergence). The failed solve leaves no state, and the next one is bit-identical
    /// to a fresh solver's. `forces` reports it too.
    #[test]
    fn a_failed_readback_is_a_device_error_wherever_it_strikes() {
        let Some((mut s, _, settings)) = solver(33) else { return };
        let sys = hydrogen([0.0; 3]);
        let before = crate::grid::readbacks_on_this_thread();
        let reference = s.solve(&sys, &settings, false).unwrap().1;
        let total = crate::grid::readbacks_on_this_thread() - before;
        assert!(total > 10, "test premise: a solve reads back many times ({total})");
        for skip in [0, 1, 2, 3, total / 4, total / 2, total - 3, total - 1] {
            gpu::readback::fail_readbacks_after(skip, 1);
            let r = s.solve_with(&sys, &settings, false, &mut || false);
            gpu::readback::fail_readbacks_after(0, 0);
            match r {
                Err(QmError::Device(m)) => assert!(m.contains("callback"), "skip {skip}: {m}"),
                other => panic!("skip {skip} of {total}: expected Err(Device), got {:?}", other.map(|o| o.map(|(_, st)| st))),
            }
            assert!(s.state().is_none(), "skip {skip}: a failed solve must expose no state");
            assert!(!crate::grid::has_readback_fault(), "skip {skip}: the fault was consumed by the error");
        }
        let (_, after) = s.solve(&sys, &settings, true).unwrap();
        assert_eq!(after.energy.total.to_bits(), reference.energy.total.to_bits(), "the solve after failures is a cold one");
        // forces: the electronic part is one readback
        let state = s.state().expect("converged");
        gpu::readback::fail_readbacks_after(0, 1);
        let f = s.forces(state, &sys.nuclei, settings.sigma);
        gpu::readback::fail_readbacks_after(0, 0);
        assert!(matches!(f, Err(QmError::Device(_))), "{f:?}");
        assert!(s.forces(s.state().expect("still converged"), &sys.nuclei, settings.sigma).is_ok());
    }

    /// `solve_with` asks `should_abort` once before every SCF step. An abandoned solve returns
    /// `Ok(None)`, exposes no state (`state()` is for converged states only) but parks its
    /// orbitals and densities as the next warm start: resuming from 2 completed steps converges
    /// in fewer steps than a cold solve and to the same energy; `warm = false` discards the
    /// parked state (bit-identical to a fresh cold solve); an abort before the first step does
    /// no SCF work at all.
    #[test]
    fn aborted_solve_parks_its_state_for_a_warm_resume() {
        let Some((mut s, _, settings)) = solver(33) else { return };
        let sys = h2(0.0);
        let (_, cold) = s.solve(&sys, &settings, false).unwrap();
        // abort when asked the 6th time: after steps 1–5 (a 5-step head start: the step count of
        // a cold H₂ solve scatters by ±2 between rounding-level solver variants, module doc, so a
        // 2-step head start would make "fewer steps than cold" a coin flip)
        let mut asked = 0;
        let out = s.solve_with(&sys, &settings, false, &mut || { asked += 1; asked == 6 }).unwrap();
        assert!(out.is_none(), "an aborted solve must not return a state");
        assert_eq!(asked, 6);
        assert!(s.state().is_none(), "a parked (unconverged) state must not be exposed");
        let p = s.profile().clone();
        assert_eq!(p.steps, 5, "{p:?}");
        assert!(p.notes.iter().any(|n| n.contains("abandoned")), "{:?}", p.notes);
        let (_, resumed) = s.solve(&sys, &settings, true).unwrap();
        println!("abort after 5 steps, warm resume: {} steps E {:+.9} vs cold {} steps E {:+.9}",
                 resumed.iterations, resumed.energy.total, cold.iterations, cold.energy.total);
        assert!(resumed.iterations < cold.iterations, "resume {} vs cold {}", resumed.iterations, cold.iterations);
        assert!((resumed.energy.total - cold.energy.total).abs() < 1e-5);
        assert!(s.state().is_some());
        // a parked state is discarded by a cold request
        let mut never = 0;
        assert!(s.solve_with(&sys, &settings, false, &mut || { never += 1; never == 2 }).unwrap().is_none());
        let (_, recold) = s.solve(&sys, &settings, false).unwrap();
        assert_eq!(recold.energy.total.to_bits(), cold.energy.total.to_bits());
        // abort before the first step: no SCF step, no state
        assert!(s.solve_with(&sys, &settings, true, &mut || true).unwrap().is_none());
        assert_eq!(s.profile().steps, 0);
        assert!(s.state().is_none());
    }

    /// `copy_density_to_volume` into an n³ `Volume3D` equals `density_f32` voxel for voxel
    /// (no padding leaked, x fastest) and the density of a displaced H peaks at the right
    /// (i, j, k) (Review Focus 2).
    #[test]
    fn density_copy_to_volume_is_unpadded_and_centred() {
        let Some((mut s, grid, settings)) = solver(33) else { return };
        // nucleus at x = +1 Bohr = 4 cells off centre: the peak must land at (20, 16, 16), so a
        // transposed or shifted copy (or a ghost cell read as data) fails, not only a non-centred one
        s.solve(&hydrogen([1.0, 0.0, 0.0]), &settings, false).unwrap();
        let state = s.state().unwrap();
        let vol = Volume3D::new(&s.ctx, [33; 3]);
        s.copy_density_to_volume(state, &vol);
        let from_texture = vol.read_back(&s.ctx).expect("readback");
        let direct = s.density_f32(state);
        assert_eq!(from_texture.len(), direct.len());
        assert_eq!(from_texture, direct, "texture copy differs from the direct readback");
        let (argmax, &peak) = direct.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
        assert_eq!(argmax, grid.idx(20, 16, 16), "density peak at {:?}, expected [1, 0, 0]", grid.point(argmax));
        assert_eq!(grid.point(argmax), [1.0, 0.0, 0.0]);
        let dmax = s.density_max(state);
        println!("volume copy 33³: peak {peak} at {:?}; density_max {dmax}", grid.point(argmax));
        assert_eq!(dmax, peak);
        assert!(peak > 0.1 && peak < 0.5, "H peak density {peak} (σ-smeared 1/π ≈ 0.32)");
        // a corner voxel is tiny: the interior, not a ghost cell or a shifted copy
        assert!(direct[0] < 1e-4 && direct[0] >= 0.0, "corner voxel {}", direct[0]);
    }

    /// Diagnostic (ignored; `cargo test -p qm-gpu --release -- --ignored --nocapture diag_`):
    /// Δρ per SCF step for H₂ at 33³, GPU vs CPU, obtained by capping `max_iter` at 1, 2, … and
    /// reading Δρ from the `ScfNotConverged` error; then the step count, energy and cost of H₂,
    /// H and Li as a function of the Poisson tolerance and the LOBPCG tolerance floor factor
    /// (the measurements behind `POISSON_TOL` and `EIG_TOL_FLOOR_FACTOR`).
    #[test]
    #[ignore]
    fn diag_trajectory_and_knobs() {
        let Some((mut s, grid, _)) = solver(33) else { return };
        let sys = h2(0.0);
        let traj = |run: &mut dyn FnMut(usize) -> Result<f64, f64>| -> Vec<f64> {
            let mut out = Vec::new();
            for n in 1..=30 { match run(n) { Ok(d) => { out.push(d); break; } Err(d) => out.push(d) } }
            out
        };
        let cpu_traj = traj(&mut |n| {
            let mut st = ScfSettings::for_grid(&grid); st.max_iter = n;
            match qm::scf(&grid, &sys, &st, None) { Ok((_, x)) => Ok(x.delta_rho), Err(QmError::ScfNotConverged(_, d)) => Err(d), Err(e) => panic!("{e}") }
        });
        let gpu_traj = traj(&mut |n| {
            let mut st = ScfSettings::for_grid(&grid); st.max_iter = n;
            match s.solve(&sys, &st, false) { Ok((_, x)) => Ok(x.delta_rho), Err(QmError::ScfNotConverged(_, d)) => Err(d), Err(e) => panic!("{e}") }
        });
        println!("step   CPU Δρ      GPU Δρ");
        for i in 0..cpu_traj.len().max(gpu_traj.len()) {
            println!("{:>4}   {:>9}   {:>9}", i + 1, cpu_traj.get(i).map_or("-".into(), |d| format!("{d:.2e}")), gpu_traj.get(i).map_or("-".into(), |d| format!("{d:.2e}")));
        }
        for (name, n, sys) in [("H₂", 33, h2(0.0)), ("H", 33, hydrogen([0.0; 3])), ("Li", 49, lithium())] {
            let grid = Grid::cube(n, 0.25);
            let settings = ScfSettings::for_grid(&grid);
            let mut s = GpuSolver::new(s.ctx.clone(), &grid);
            for (ptol, factor) in [(1e-6, 3.0), (1e-8, 3.0), (1e-8, 1.0)] {
                s.set_knobs(ptol, factor);
                let t = Instant::now();
                let stats = s.solve(&sys, &settings, false).unwrap().1;
                let p = s.profile().clone();
                println!("{name} {n}³ Poisson tol {ptol:.0e}, eig floor factor {factor}: {} steps in {:.0} ms, E {:+.9}, Δρ {:.2e}, eig res {:.2e}, Poisson res {:.2e}, Poisson it {} ({:.1}/solve), LOBPCG it {}, readbacks {}",
                         stats.iterations, t.elapsed().as_secs_f64() * 1e3, stats.energy.total, stats.delta_rho, stats.max_eigen_residual, stats.poisson_residual,
                         p.poisson_iterations, p.poisson_iterations as f64 / (p.steps + 1) as f64, p.eigen_iterations, p.readbacks);
            }
        }
    }

    /// Diagnostic (ignored; `cargo test -p qm-gpu --release -- --ignored --nocapture diag_eig`):
    /// the LOBPCG tolerance floor factor against the SCF step count, LOBPCG iterations, floor
    /// accepts and wall for H₂ (49³), Li (49³) and H₂O (49³, the `h2o` preset), cold and warm
    /// (shifted 0.02 Bohr) — the measurement behind `EIG_TOL_FLOOR_FACTOR`.
    #[test]
    #[ignore]
    fn diag_eig_tol_factor() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        let h2o = |shift: f64| System::new(vec![Nucleus { pos: [shift, 0.0, 0.0], z: 8 }, Nucleus { pos: [1.43 + shift, 1.11, 0.0], z: 1 }, Nucleus { pos: [-1.43 + shift, 1.11, 0.0], z: 1 }], 0, None);
        let grid = Grid::cube(49, 0.25);
        let settings = ScfSettings::for_grid(&grid);
        type Make = Box<dyn Fn(f64) -> System>;
        let cases: Vec<(&str, Make)> = vec![
            ("H₂", Box::new(h2)), ("Li", Box::new(|s| System::new(vec![Nucleus { pos: [s, 0.0, 0.0], z: 3 }], 0, None))), ("H₂O", Box::new(h2o)),
        ];
        for (name, make) in &cases {
            let mut s = GpuSolver::new(ctx.clone(), &grid);
            let _ = s.solve(&make(0.0), &settings, false).unwrap();
            for factor in [1.0, 1.5, 2.0, 3.0] {
                s.set_knobs(POISSON_TOL, factor);
                let report = |label: &str, st: &ScfStats, p: &SolveProfile| {
                    println!("{name} 49³ factor {factor}: {label} {} steps, {:.0} ms (LOBPCG {:.0} ms, {} it, floor accepts {}, cap hits {}), E {:+.9}, Δρ {:.2e}, eig res {:.2e}",
                             st.iterations, p.wall.as_secs_f64() * 1e3, p.eigen_wall.as_secs_f64() * 1e3, p.eigen_iterations, p.eigen_floor_accepts, p.eigen_cap_hits, st.energy.total, st.delta_rho, st.max_eigen_residual);
                };
                let (_, cold) = s.solve(&make(0.0), &settings, false).unwrap();
                report("cold", &cold, &s.profile().clone());
                let (_, warm) = s.solve(&make(0.02), &settings, true).unwrap();
                report("warm", &warm, &s.profile().clone());
            }
        }
    }

    /// Diagnostic (ignored; `SCF_TRACE=1 cargo test -p qm-gpu --release -- --ignored --nocapture diag_lag`):
    /// the app's warm-start witness (`app::sim::moving_a_nucleus_warm_starts`: H₂ 49³, nucleus 0
    /// moved 0.70 → 0.72 Bohr, tol 1e-5, eig_tol 1e-6) step by step — cold at 0.70, warm at
    /// 0.72, cold at 0.72 — the measurement behind the Δρ-tied eigen tolerance.
    #[test]
    #[ignore]
    fn diag_lag_witness() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        let grid = Grid::cube(49, 0.25);
        let mut settings = ScfSettings::for_grid(&grid);
        settings.tol = std::env::var("DIAG_TOL").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-5);
        let at = |x0: f64| System::new(vec![Nucleus { pos: [x0, 0.0, 0.0], z: 1 }, Nucleus { pos: [-0.7, 0.0, 0.0], z: 1 }], 0, None);
        let mut s = GpuSolver::new(ctx, &grid);
        let report = |label: &str, st: &ScfStats, p: &SolveProfile| {
            println!("{label}: {} steps, {} readbacks (LOBPCG {} it / {} readbacks), {:.0} ms, E {:+.9}, Δρ {:.2e}, eig res {:.2e}, eig tol per step {}",
                     st.iterations, p.readbacks, p.eigen_iterations, p.eigen_readbacks, p.wall.as_secs_f64() * 1e3, st.energy.total, st.delta_rho, st.max_eigen_residual, tols(&p.eig_tol_steps));
        };
        let (_, cold) = s.solve(&at(0.70), &settings, false).unwrap();
        report("cold 0.70", &cold, &s.profile().clone());
        eprintln!("--- warm 0.72 ---");
        let (_, warm) = s.solve(&at(0.72), &settings, true).unwrap();
        report("warm 0.72", &warm, &s.profile().clone());
        eprintln!("--- cold 0.72 ---");
        let (_, cold2) = s.solve(&at(0.72), &settings, false).unwrap();
        report("cold 0.72", &cold2, &s.profile().clone());
        println!("|ΔE| warm vs cold-of-displaced: {:.2e}", (warm.energy.total - cold2.energy.total).abs());
    }

    /// Diagnostic (ignored; `cargo test -p qm-gpu --release -- --ignored --nocapture diag_eig_fraction`):
    /// the Δρ-tied eigen tol (`eig_tol_for`) swept over `EIG_TOL_FRACTION` and `EIG_TOL_MAX` —
    /// steps, readbacks, LOBPCG iterations and energies of the app witness (H₂ 49³ 0.70 → 0.72,
    /// tol 1e-5 and 1e-6: cold, warm, cold of the displaced), H₂O 49³ cold + warm (0.02), Li 49³
    /// cold, the CPU-parity cases (H₂ 33³ tol 1e-6 and 1e-7; LiH, He, H₂ R = 3.2, H at 49³) — the
    /// measurement behind the constants. Variants from `DIAG_VARIANTS` = "f,max;…", a subset of
    /// the groups from `DIAG_ONLY` (witness, h2o, li, parity, tight).
    #[test]
    #[ignore]
    fn diag_eig_fraction() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        let variants: Vec<(f64, f64)> = match std::env::var("DIAG_VARIANTS") {
            Ok(v) => v.split(';').map(|t| { let f: Vec<&str> = t.split(',').collect(); (f[0].parse().unwrap(), f[1].parse().unwrap()) }).collect(),
            Err(_) => vec![(0.0, EIG_TOL_MAX), (0.001, EIG_TOL_MAX), (EIG_TOL_FRACTION, EIG_TOL_MAX), (0.003, EIG_TOL_MAX), (0.01, EIG_TOL_MAX), (EIG_TOL_FRACTION, 1e-4)],
        };
        let only: Option<String> = std::env::var("DIAG_ONLY").ok();
        let run = |name: &str| only.as_ref().is_none_or(|o| name.contains(o.as_str()));
        let g49 = Grid::cube(49, 0.25);
        let g33 = Grid::cube(33, 0.25);
        let h2at = |x0: f64| System::new(vec![Nucleus { pos: [x0, 0.0, 0.0], z: 1 }, Nucleus { pos: [-0.7, 0.0, 0.0], z: 1 }], 0, None);
        let h2o = |shift: f64| System::new(vec![Nucleus { pos: [shift, 0.0, 0.0], z: 8 }, Nucleus { pos: [1.43 + shift, 1.11, 0.0], z: 1 }, Nucleus { pos: [-1.43 + shift, 1.11, 0.0], z: 1 }], 0, None);
        let line = |label: &str, st: &ScfStats, p: &SolveProfile| format!("{label}: {} steps, {} rb (LOBPCG {} it), {:.0} ms, E {:+.9}, Δρ {:.2e}, eig res {:.2e}, floor acc {}, warn {}",
            st.iterations, p.readbacks, p.eigen_iterations, p.wall.as_secs_f64() * 1e3, st.energy.total, st.delta_rho, st.max_eigen_residual, p.eigen_floor_accepts, st.warnings.len());
        for (f, max) in variants {
            println!("=== fraction {f}, max {max:.0e} ===");
            let mut s49 = GpuSolver::new(ctx.clone(), &g49);
            s49.set_eig_knobs(f, max);
            for tol in [1e-5, 1e-6] {
                if !run("witness") { break; }
                let mut st = ScfSettings::for_grid(&g49);
                st.tol = tol;
                let (_, c) = s49.solve(&h2at(0.70), &st, false).unwrap(); let pc = s49.profile().clone();
                let (_, w) = s49.solve(&h2at(0.72), &st, true).unwrap(); let pw = s49.profile().clone();
                let (_, c2) = s49.solve(&h2at(0.72), &st, false).unwrap(); let pc2 = s49.profile().clone();
                println!("  witness tol {tol:.0e}: {} | {} | {} | |ΔE| warm−cold {:.2e}", line("cold", &c, &pc), line("warm", &w, &pw), line("cold disp", &c2, &pc2), (w.energy.total - c2.energy.total).abs());
                println!("    eig tol per step: warm {}", tols(&pw.eig_tol_steps));
            }
            if run("h2o") {
                let st = ScfSettings::for_grid(&g49);
                let (_, c) = s49.solve(&h2o(0.0), &st, false).unwrap(); let pc = s49.profile().clone();
                let (_, w) = s49.solve(&h2o(0.02), &st, true).unwrap(); let pw = s49.profile().clone();
                println!("  H₂O 49³ tol 1e-6: {} | {}", line("cold", &c, &pc), line("warm 0.02", &w, &pw));
            }
            if run("li") {
                let st = ScfSettings::for_grid(&g49);
                match s49.solve(&lithium(), &st, false) {
                    Ok((_, c)) => { let pc = s49.profile().clone(); println!("  Li 49³ tol 1e-6: {}", line("cold", &c, &pc)); }
                    Err(e) => println!("  Li 49³ tol 1e-6: FAILED {e}"),
                }
            }
            if run("parity") {
                let mut s33 = GpuSolver::new(ctx.clone(), &g33);
                s33.set_eig_knobs(f, max);
                let st = ScfSettings::for_grid(&g33);
                let (_, c) = s33.solve(&h2(0.0), &st, false).unwrap(); let pc = s33.profile().clone();
                println!("  H₂ 33³ tol 1e-6 (CPU 10): {}", line("cold", &c, &pc));
                let st = ScfSettings::for_grid(&g49);
                let lih = System::new(vec![Nucleus { pos: [-0.8, 0.0, 0.0], z: 3 }, Nucleus { pos: [2.2, 0.0, 0.0], z: 1 }], 0, None);
                let he = System::new(vec![Nucleus { pos: [0.0; 3], z: 2 }], 0, None);
                let h2_32 = System::new(vec![Nucleus { pos: [1.6, 0.0, 0.0], z: 1 }, Nucleus { pos: [-1.6, 0.0, 0.0], z: 1 }], 0, None);
                for (name, sys) in [("LiH", lih), ("He", he), ("H₂ R=3.2", h2_32), ("H", hydrogen([0.0; 3]))] {
                    let (_, c) = s49.solve(&sys, &st, false).unwrap(); let pc = s49.profile().clone();
                    println!("  {name} 49³ tol 1e-6: {}", line("cold", &c, &pc));
                }
            }
            if run("tight") {
                let mut s33 = GpuSolver::new(ctx.clone(), &g33);
                s33.set_eig_knobs(f, max);
                let mut st = ScfSettings::for_grid(&g33);
                st.tol = 1e-7;
                match s33.solve(&h2(0.0), &st, false) {
                    Ok((_, c)) => { let pc = s33.profile().clone(); println!("  H₂ 33³ tol 1e-7: {}", line("cold", &c, &pc)); }
                    Err(e) => println!("  H₂ 33³ tol 1e-7: FAILED {e}"),
                }
            }
        }
    }

    /// Diagnostic (ignored): one cold H₂O solve on the grid given by `DIAG_N` (default 97) with
    /// the per-step trace; `SCF_TRACE=1 cargo test -p qm-gpu --release -- --ignored --nocapture diag_h2o`.
    #[test]
    #[ignore]
    fn diag_h2o_cold() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        let n: usize = std::env::var("DIAG_N").ok().and_then(|v| v.parse().ok()).unwrap_or(97);
        let h2o = System::new(vec![Nucleus { pos: [0.0, 0.0, 0.0], z: 8 }, Nucleus { pos: [1.43, 1.11, 0.0], z: 1 }, Nucleus { pos: [-1.43, 1.11, 0.0], z: 1 }], 0, None);
        let grid = Grid::cube(n, 0.25);
        let settings = ScfSettings::for_grid(&grid);
        let mut s = GpuSolver::new(ctx, &grid);
        let (_, st) = s.solve(&h2o, &settings, false).unwrap();
        let p = s.profile();
        println!("H₂O {n}³ cold: {} steps, {:.0} ms (Poisson {:.0}, LOBPCG {:.0}), E {:+.9}, Δρ {:.2e}, eigen it {}, floor accepts {}, notes {:?}",
                 st.iterations, p.wall.as_secs_f64() * 1e3, p.poisson_wall.as_secs_f64() * 1e3, p.eigen_wall.as_secs_f64() * 1e3, st.energy.total, st.delta_rho, p.eigen_iterations, p.eigen_floor_accepts, p.notes);
    }

    /// Timing for the report: `cargo test -p qm-gpu --release -- --ignored --nocapture bench_scf`.
    /// H₂ cold and warm (displaced by 0.02 Bohr) at 33³ and 49³: ms per step and per solve,
    /// readbacks per step by origin.
    #[test]
    #[ignore]
    fn bench_scf_33_49() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        for n in [33usize, 49] {
            let grid = Grid::cube(n, 0.25);
            let settings = ScfSettings::for_grid(&grid);
            let mut s = GpuSolver::new(ctx.clone(), &grid);
            let _ = s.solve(&h2(0.0), &settings, false).unwrap(); // warm the pipelines/pool
            let report = |label: &str, stats: &ScfStats, p: SolveProfile| {
                let ms = p.wall.as_secs_f64() * 1e3;
                println!("{n}³ H₂ {label}: {} steps, {ms:.1} ms ({:.1} ms/step), E {:+.8}, Δρ {:.2e}; readbacks {} ({:.1}/step: Poisson {:.1}, LOBPCG {:.1}, SCF {:.1}); Poisson {} it, LOBPCG {} it; eig res {:.2e}, Poisson res {:.2e}",
                         stats.iterations, ms / stats.iterations as f64, stats.energy.total, stats.delta_rho, p.readbacks, p.readbacks as f64 / p.steps as f64,
                         p.poisson_readbacks as f64 / p.steps as f64, p.eigen_readbacks as f64 / p.steps as f64,
                         (p.readbacks - p.poisson_readbacks - p.eigen_readbacks) as f64 / p.steps as f64,
                         p.poisson_iterations, p.eigen_iterations, stats.max_eigen_residual, stats.poisson_residual);
            };
            let (_, cold) = s.solve(&h2(0.0), &settings, false).unwrap();
            report("cold", &cold, s.profile().clone());
            let (_, warm) = s.solve(&h2(0.02), &settings, true).unwrap();
            report("warm (shifted 0.02)", &warm, s.profile().clone());
            let (_, cold2) = s.solve(&h2(0.02), &settings, false).unwrap();
            report("cold (shifted 0.02)", &cold2, s.profile().clone());
        }
    }

    /// The two `IonicPolicy::None` scenes of the bitwise pin (M7 Task 8a Step 1; the CPU's Task 6 twin uses the same
    /// geometries): H₂ at R = 1.4 and LiH (Li at (−0.8, 0.1, 0), H at (2.2, 0, 0)), 33³, h = σ = 0.25, cold.
    fn none_policy_scenes() -> [(&'static str, System); 2] {
        [("H2", System::new(vec![Nucleus { pos: [0.7, 0.0, 0.0], z: 1 }, Nucleus { pos: [-0.7, 0.0, 0.0], z: 1 }], 0, None)),
         ("LIH", System::new(vec![Nucleus { pos: [-0.8, 0.1, 0.0], z: 3 }, Nucleus { pos: [2.2, 0.0, 0.0], z: 1 }], 0, None))]
    }

    // Measured by `gpu_none_policy_baseline` on the code before M7 Task 8a (8866c91, RX 7900 XTX, Vulkan).
    const GPU_BASE_H2_E: u64 = 0xbfee7a76ab97234d; // -0.952449164522
    const GPU_BASE_H2_FX: u64 = 0x3fc07b00b495fc7c; // +0.128753746201
    const GPU_BASE_LIH_E: u64 = 0xc014e1309967ef7d; // -5.219911953896
    const GPU_BASE_LIH_FX: u64 = 0xbf973859f2492440; // -0.022675900853

    /// M7 Task 8a: `IonicPolicy::None` (no setup, `Aufbau`) is today's GPU solver bit for bit — the energy and F_x of
    /// nucleus 0 of H₂ and LiH equal the bits measured on the code before the pseudopotential path (driver-specific:
    /// re-pin with `gpu_none_policy_baseline` on the pre-M7 commit after a driver change, never edit the numbers).
    #[test]
    fn gpu_none_policy_is_bitwise_today() {
        let Some((mut s, _, settings)) = solver(33) else { return };
        for ((name, sys), (e_bits, f_bits)) in none_policy_scenes().into_iter().zip([(GPU_BASE_H2_E, GPU_BASE_H2_FX), (GPU_BASE_LIH_E, GPU_BASE_LIH_FX)]) {
            let (_, stats) = s.solve(&sys, &settings, false).unwrap();
            let f = s.forces(s.state().unwrap(), &sys.nuclei, settings.sigma).unwrap();
            println!("{name}: E {:+.12} F_x0 {:+.12}", stats.energy.total, f[0][0]);
            assert_eq!(stats.energy.total.to_bits(), e_bits, "{name} energy");
            assert_eq!(f[0][0].to_bits(), f_bits, "{name} F_x0");
            let st = s.state().unwrap();
            assert!(st.ionic.is_none() && st.occupations == Occupations::Aufbau && stats.core_overlaps.is_empty() && stats.fractional.is_empty());
            assert!(st.occ_up.iter().chain(&st.occ_dn).all(|&f| f == 1.0));
        }
    }

    /// The GPU pseudo solve vs `qm::scf` on one discrete problem (M7 Task 8a): energy components and every eigenvalue
    /// (witnesses included). Returns (|ΔE_total|, max component |Δ|, max |Δε|, then the forces of Task 8b: the kernel
    /// part max |F_gpu(S_g) − F_cpu(S_g)| — GPU forces against the CPU f64 forces on the same downloaded GPU state —, and
    /// max |ΔF_term| per term [ion–ion, LR, SR, NL, ρ_pc] on that state). The across-solver difference
    /// max |F_gpu(S_g) − F_cpu(S_c)| and its state part max |F_cpu(S_g) − F_cpu(S_c)| are printed, not returned: they are
    /// first order in each state's error and gated once, in `validate`'s gpu_vs_cpu (`F_BOUND_PSEUDO`, held). A pin from
    /// one cold-vs-cold value here would sit at a lucky value (Task 8b review SHOULD-1).
    fn pseudo_vs_cpu(name: &str, grid: &Grid, sys: &System, settings: &ScfSettings) -> (f64, f64, f64, f64, [f64; 5]) {
        let ctx = GpuContext::new_headless().expect("adapter checked by the caller");
        let (cpu_state, cpu_stats, cpu_s) = cpu(grid, sys, settings);
        let mut s = GpuSolver::new(ctx, grid);
        let t = Instant::now();
        let stats = s.solve(sys, settings, false).unwrap_or_else(|e| panic!("{name}: GPU {e}")).1;
        let gpu_s = t.elapsed().as_secs_f64();
        let st = s.state().unwrap();
        let (a, b) = (stats.energy, cpu_stats.energy);
        let comps = [a.kinetic - b.kinetic, a.external - b.external, a.hartree - b.hartree, a.xc - b.xc, a.nuclear - b.nuclear];
        let dc = comps.iter().fold(0.0f64, |m, d| m.max(d.abs()));
        let sorted = |e: &[f64]| -> Vec<f64> { let mut v = e.to_vec(); v.sort_by(f64::total_cmp); v };
        let (gu, gd, cu, cd) = (sorted(&st.evals_up), sorted(&st.evals_dn), sorted(&cpu_state.evals_up), sorted(&cpu_state.evals_dn));
        assert_eq!((gu.len(), gd.len()), (cu.len(), cd.len()), "{name}: stored orbital counts (witnesses included)");
        let deps = gu.iter().zip(&cu).chain(gd.iter().zip(&cd)).map(|(g, c)| (g - c).abs()).fold(0.0, f64::max);
        println!("{name}: GPU {} steps {gpu_s:.2} s ({:?}), CPU {} steps {cpu_s:.1} s; E gpu {:+.9} cpu {:+.9} Δ {:+.2e}; components Δ [T {:+.1e}, ext {:+.1e}, H {:+.1e}, xc {:+.1e}, nn {:+.1e}]; max |Δε| {deps:.2e}",
                 stats.iterations, s.profile().notes, cpu_stats.iterations, a.total, b.total, a.total - b.total, comps[0], comps[1], comps[2], comps[3], comps[4]);
        println!("  ε↑ gpu {gu:.6?} cpu {cu:.6?}; ε↓ gpu {gd:.6?} cpu {cd:.6?}");
        println!("  occ ↑ gpu {:?} cpu {:?}; occ ↓ gpu {:?} cpu {:?}; shared gpu {:?} cpu {:?}; warnings gpu {:?} cpu {:?}",
                 st.occ_up, cpu_state.occ_up, st.occ_dn, cpu_state.occ_dn, stats.fractional, cpu_stats.fractional, stats.warnings, cpu_stats.warnings);
        assert!((st.occ_up.iter().sum::<f64>() - sys.n_up as f64).abs() < 1e-12 && (st.occ_dn.iter().sum::<f64>() - sys.n_dn as f64).abs() < 1e-12);
        assert_eq!(stats.fractional.len(), cpu_stats.fractional.len(), "{name}: the same shared clusters");
        assert_eq!(stats.released.len(), cpu_stats.released.len(), "{name}: the same releases");
        assert_eq!(stats.core_overlaps.len(), cpu_stats.core_overlaps.len());
        // M7 Task 8b: forces, each solver on its own state, and per term on the same state
        let f_gpu = s.forces(st, &sys.nuclei, settings.sigma).unwrap();
        let f_cpu = qm::forces(grid, &cpu_state, &sys.nuclei, settings.sigma).unwrap();
        let df = f_gpu.iter().flatten().zip(f_cpu.iter().flatten()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        let t_gpu = s.force_terms(st, &sys.nuclei).unwrap();
        let t_cpu = qm::forces::pseudo_force_terms(grid, &s.ks_state(st), &sys.nuclei).unwrap();
        let dt: [f64; 5] = std::array::from_fn(|t| t_gpu[t].iter().flatten().zip(t_cpu[t].iter().flatten()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max));
        // the split ΔF = [F_gpu(S_g) − F_cpu(S_g)] (kernels) + [F_cpu(S_g) − F_cpu(S_c)] (state)
        let f_cpu_sg: Vec<[f64; 3]> = (0..sys.nuclei.len()).map(|i| std::array::from_fn(|a| crate::reduce::neumaier(t_cpu.iter().map(|p| p[i][a])))).collect();
        let maxd = |x: &[[f64; 3]], y: &[[f64; 3]]| x.iter().flatten().zip(y.iter().flatten()).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max);
        let (kernel, state) = (maxd(&f_gpu, &f_cpu_sg), maxd(&f_cpu_sg, &f_cpu));
        println!("  forces: gpu {f_gpu:+.8?}\n          cpu {f_cpu:+.8?}\n          max |ΔF| {df:.2e} = kernels {kernel:.2e} + state {state:.2e}; same state per term max |Δ| [ii {:.2e}, LR {:.2e}, SR {:.2e}, NL {:.2e}, ρ_pc {:.2e}]",
                 dt[0], dt[1], dt[2], dt[3], dt[4]);
        ((a.total - b.total).abs(), dc, deps, kernel, dt)
    }

    /// Task 8b's force assertions on one GPU state (`pseudo_vs_cpu`): the kernel part of the force and each term, against
    /// bounds pinned by the 3× rule from the measured values (each caller's comment).
    fn assert_pseudo_forces(name: &str, kernel: f64, dt: [f64; 5], kernel_bound: f64, term_bounds: [f64; 5]) {
        assert!(kernel <= kernel_bound, "{name}: GPU vs CPU forces on the same GPU state differ by {kernel:.2e} > {kernel_bound:.0e}");
        for (t, n) in ["ion–ion", "LR", "SR", "NL", "ρ_pc"].iter().enumerate() {
            assert!(dt[t] <= term_bounds[t], "{name}: the {n} term on one state differs by {:.2e} > {:.0e}", dt[t], term_bounds[t]);
        }
    }

    /// Pseudo-O (`CoresOnly`, triplet, `DegenerateShells`: the spec §5.7 shared minority p) on the GPU vs the CPU at 33³,
    /// h = σ = 0.25 (an 8 Bohr box; uncertified `from_tables`: a solver comparison on one discrete problem). Bounds by the
    /// suite's 3× rule: measured |ΔE| 7.8e-7 (GPU path scatter 5.2e-7, `validate`) → 3e-6; components 3.3e-5 → 1e-4;
    /// eigenvalues 3.0e-6 → 1e-5 Ha.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm-gpu --release")]
    fn pseudo_o_triplet_matches_cpu() {
        if GpuContext::new_headless().is_none() { return; }
        use crate::ionic::tests::filtered_at;
        use qm::ionic::{Ionic, IonicPolicy, IonicSetup};
        let h = 0.25;
        let grid = Grid::cube(33, h);
        let setup = std::sync::Arc::new(IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, std::collections::BTreeMap::from([(8, Ionic::Pseudo(filtered_at(8, h)))])));
        let sys = System::with_ionic(vec![Nucleus { pos: [0.07, -0.03, 0.02], z: 8 }], 0, Some(3), setup);
        let mut settings = ScfSettings::for_grid(&grid);
        settings.occupations = Occupations::DegenerateShells { tau: qm::DEGENERACY_TAU };
        let (de, dc, deps, kernel, dt) = pseudo_vs_cpu("pseudo-O M=3 33³", &grid, &sys, &settings);
        assert!(de < 3e-6 && dc < 1e-4 && deps < 1e-5, "|ΔE| {de:.2e}, component {dc:.2e}, ε {deps:.2e}");
        // Task 8b (forces, review round 1): the f32 evaluation on one GPU state, pinned by the 3× rule from the measured
        // values (2026-10-17): kernel part 1.82e-7 → 6e-7; ion–ion 0 (the same f64 function) → 0; LR 1.34e-7 → held at
        // round 0's 4e-7 (2.99×; never raised); SR 4.31e-8 → 2e-7; NL 2.87e-8 → 9e-8; ρ_pc 1.77e-8 → 6e-8. The
        // across-solver difference (1.19e-6 = kernels 1.8e-7 + state 1.18e-6) is printed and gated in gpu_vs_cpu.
        assert_pseudo_forces("pseudo-O M=3 33³", kernel, dt, 6e-7, [0.0, 4e-7, 2e-7, 9e-8, 6e-8]);
    }

    /// `IonicPolicy::All` (pseudo-H everywhere: no partial core, Z_v = Z) on H₂ under `Aufbau` (closed shell), 33³,
    /// h = 0.25. Bounds: the H₂ class of `validate::gpu_vs_cpu` (2e-6; measured 3.2e-8), components 2e-5 (the 33³ H₂
    /// test's; measured 3.9e-6), eigenvalues 3e-6 (3× the measured 7.9e-7).
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm-gpu --release")]
    fn pseudo_h2_all_matches_cpu() {
        if GpuContext::new_headless().is_none() { return; }
        use crate::ionic::tests::filtered_at;
        use qm::ionic::{Ionic, IonicPolicy, IonicSetup};
        let h = 0.25;
        let grid = Grid::cube(33, h);
        let setup = std::sync::Arc::new(IonicSetup::from_tables(IonicPolicy::All, h, h, std::collections::BTreeMap::from([(1, Ionic::Pseudo(filtered_at(1, h)))])));
        let sys = System::with_ionic(vec![Nucleus { pos: [0.72, 0.03, -0.02], z: 1 }, Nucleus { pos: [-0.69, 0.0, 0.01], z: 1 }], 0, None, setup);
        let settings = ScfSettings::for_grid(&grid);
        let (de, dc, deps, kernel, dt) = pseudo_vs_cpu("pseudo-H₂ (All) 33³", &grid, &sys, &settings);
        assert!(de < 2e-6 && dc < 2e-5 && deps < 3e-6, "|ΔE| {de:.2e}, component {dc:.2e}, ε {deps:.2e}");
        // Task 8b (forces, review round 1): the no-core branch without a projector for H (NL and ρ_pc exactly 0 on both
        // sides); pinned by the 3× rule from the measured values (2026-10-17): kernel part 9.3e-9 → 3e-8; LR 1.37e-8 →
        // 5e-8; SR 4.36e-9 → 2e-8; ion–ion, NL, ρ_pc 0 → 0. The across-solver difference (2.58e-6 = kernels 9.3e-9 + state
        // 2.57e-6) is printed and gated in gpu_vs_cpu.
        assert_pseudo_forces("pseudo-H₂ (All) 33³", kernel, dt, 3e-8, [0.0, 5e-8, 2e-8, 0.0, 0.0]);
    }

    /// `a_failed_readback_is_a_device_error_wherever_it_strikes` on the M7 path (the main merge of 2026-10-17): pseudo-O
    /// (`CoresOnly`: V_NL and the partial core) as a triplet under `DegenerateShells` (the shared-occupation channel solves,
    /// the witness growth), 33³, h = 0.25. A readback that fails anywhere — a phase-1 eigensolve whose zero eigenvalues
    /// would feed the occupation rule, Δρ, the final reduction, the E_NL readback — is `Err(QmError::Device)`, leaves no
    /// state and is consumed; the next cold solve is bit-identical to the first. `forces` / `force_terms` (one readback
    /// for the four electronic terms) report it too.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm-gpu --release")]
    fn a_failed_readback_on_the_pseudo_path_is_a_device_error() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        use crate::ionic::tests::filtered_at;
        use qm::ionic::{Ionic, IonicPolicy, IonicSetup};
        let h = 0.25;
        let grid = Grid::cube(33, h);
        let setup = std::sync::Arc::new(IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, std::collections::BTreeMap::from([(8, Ionic::Pseudo(filtered_at(8, h)))])));
        let sys = System::with_ionic(vec![Nucleus { pos: [0.07, -0.03, 0.02], z: 8 }], 0, Some(3), setup);
        let mut settings = ScfSettings::for_grid(&grid);
        settings.occupations = Occupations::DegenerateShells { tau: qm::DEGENERACY_TAU };
        let mut s = GpuSolver::new(ctx, &grid);
        let before = crate::grid::readbacks_on_this_thread();
        let reference = s.solve(&sys, &settings, false).unwrap().1;
        let total = crate::grid::readbacks_on_this_thread() - before;
        assert!(total > 20, "test premise: a solve reads back many times ({total})");
        for skip in [0, 1, 2, 3, total / 8, total / 4, total / 2, 3 * total / 4, total - 3, total - 2, total - 1] {
            gpu::readback::fail_readbacks_after(skip, 1);
            let r = s.solve_with(&sys, &settings, false, &mut || false);
            gpu::readback::fail_readbacks_after(0, 0);
            match r {
                Err(QmError::Device(m)) => assert!(m.contains("callback"), "skip {skip}: {m}"),
                other => panic!("skip {skip} of {total}: expected Err(Device), got {:?}", other.map(|o| o.map(|(_, st)| st))),
            }
            assert!(s.state().is_none(), "skip {skip}: a failed solve must expose no state");
            assert!(!crate::grid::has_readback_fault(), "skip {skip}: the fault was consumed by the error");
        }
        let (_, after) = s.solve(&sys, &settings, false).unwrap();
        assert_eq!(after.energy.total.to_bits(), reference.energy.total.to_bits(), "the solve after failures is a cold one");
        let state = s.state().expect("converged");
        gpu::readback::fail_readbacks_after(0, 1);
        let f = s.forces(state, &sys.nuclei, settings.sigma);
        gpu::readback::fail_readbacks_after(0, 0);
        assert!(matches!(f, Err(QmError::Device(_))), "{f:?}");
        gpu::readback::fail_readbacks_after(0, 1);
        let t = s.force_terms(state, &sys.nuclei);
        gpu::readback::fail_readbacks_after(0, 0);
        assert!(matches!(t, Err(QmError::Device(_))), "{:?}", t.map(|_| ()));
        assert!(!crate::grid::has_readback_fault());
        assert!(s.forces(s.state().expect("still converged"), &sys.nuclei, settings.sigma).is_ok());
    }

    /// M7 Task 9: the partial-core display (`copy_density_with_core_to_volume`) shows rho + rho_pc — its returned
    /// maximum is the maximum of the CPU sum of the state's density and `qm::nuclear::partial_core_field` (f32 on
    /// the device) and exceeds the plain density's near the core — and is a plain copy (the plain maximum) for an
    /// all-electron state. The solve itself is untouched by it.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm-gpu --release")]
    fn the_core_display_adds_the_partial_core_to_the_density() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        use crate::ionic::tests::filtered_at;
        use qm::ionic::{Ionic, IonicPolicy, IonicSetup};
        let h = 0.25;
        let grid = Grid::cube(33, h);
        let setup = std::sync::Arc::new(IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, std::collections::BTreeMap::from([(8, Ionic::Pseudo(filtered_at(8, h)))])));
        let sys = System::with_ionic(vec![Nucleus { pos: [0.07, -0.03, 0.02], z: 8 }], 0, Some(3), setup.clone());
        let mut settings = ScfSettings::for_grid(&grid);
        settings.occupations = Occupations::DegenerateShells { tau: qm::DEGENERACY_TAU };
        let mut s = GpuSolver::new(ctx.clone(), &grid);
        let (_, stats) = s.solve(&sys, &settings, false).unwrap();
        let state = s.state().expect("converged");
        let volume = Volume3D::new(&ctx, [33; 3]);
        let plain = s.density_max(state);
        let with_core = s.copy_density_with_core_to_volume(state, &volume);
        let rho = s.density_f32(state);
        let pc = qm::nuclear::partial_core_field(&grid, &sys.nuclei, &setup);
        let want = rho.iter().zip(&pc).map(|(&r, &c)| r as f64 + c).fold(0.0, f64::max);
        println!("max rho {plain:.5}, max rho + rho_pc {with_core:.5} (CPU sum {want:.5}); E {:.6}", stats.energy.total);
        assert!(with_core > plain, "the core adds density near the nucleus");
        assert!((with_core as f64 - want).abs() < 2e-4 * want, "{with_core} vs {want}");
        // the plain copy is unchanged, and the solve still gives the same energy afterwards
        assert_eq!(s.density_max(state), plain);
        // an all-electron state: a plain copy
        let h2 = System::new(vec![Nucleus { pos: [0.7, 0.0, 0.0], z: 1 }, Nucleus { pos: [-0.7, 0.0, 0.0], z: 1 }], 0, None);
        let mut s2 = GpuSolver::new(ctx.clone(), &grid);
        s2.solve(&h2, &ScfSettings::for_grid(&grid), false).unwrap();
        let st2 = s2.state().expect("converged");
        assert_eq!(s2.copy_density_with_core_to_volume(st2, &volume), s2.density_max(st2));
    }

    /// Diagnostic (M7 Task 8a): a GPU pseudo-O triplet solve (`CoresOnly`, `DegenerateShells`) at h = 0.20 in a 12 Bohr
    /// box (61³), cold, with the per-step trace: `SCF_TRACE=1 cargo test -p qm-gpu --release -- --ignored --nocapture
    /// diag_pseudo_o`. `DIAG_N` overrides n.
    #[test]
    #[ignore]
    fn diag_pseudo_o() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        use crate::ionic::tests::filtered_at;
        use qm::ionic::{Ionic, IonicPolicy, IonicSetup};
        let h = 0.20;
        let n: usize = std::env::var("DIAG_N").ok().and_then(|v| v.parse().ok()).unwrap_or(61);
        let grid = Grid::cube(n, h);
        let setup = std::sync::Arc::new(IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, std::collections::BTreeMap::from([(8, Ionic::Pseudo(filtered_at(8, h)))])));
        let sys = System::with_ionic(vec![Nucleus { pos: [0.07, -0.03, 0.02], z: 8 }], 0, Some(3), setup);
        let mut settings = ScfSettings::for_grid(&grid);
        settings.occupations = Occupations::DegenerateShells { tau: qm::DEGENERACY_TAU };
        let mut s = GpuSolver::new(ctx, &grid);
        let (_, st) = s.solve(&sys, &settings, false).unwrap();
        let p = s.profile();
        println!("pseudo-O {n}³ cold: {} steps, {:.0} ms (Poisson {:.0}, LOBPCG {:.0}), E {:+.9}, Δρ {:.2e}, eigen it {}, DIIS resets {}, shared {:?}, released {:?}, notes {:?}",
                 st.iterations, p.wall.as_secs_f64() * 1e3, p.poisson_wall.as_secs_f64() * 1e3, p.eigen_wall.as_secs_f64() * 1e3, st.energy.total, st.delta_rho,
                 p.eigen_iterations, p.diis_resets, st.fractional, st.released, p.notes);
        // the all-electron O on the same grid (σ = h, the same occupation rule) for the cost comparison
        let ae = System::new(vec![Nucleus { pos: [0.07, -0.03, 0.02], z: 8 }], 0, Some(3));
        let (_, st) = s.solve(&ae, &settings, false).unwrap();
        let p = s.profile();
        println!("all-electron O {n}³ (σ = h) cold: {} steps, {:.0} ms (Poisson {:.0}, LOBPCG {:.0}), E {:+.9}, eigen it {}, shared {:?}",
                 st.iterations, p.wall.as_secs_f64() * 1e3, p.poisson_wall.as_secs_f64() * 1e3, p.eigen_wall.as_secs_f64() * 1e3, st.energy.total, p.eigen_iterations, st.fractional);
    }

    /// Step 1 printer (M7 Task 8a): run on the code before the pseudopotential path to print the `GPU_BASE_…` pins of
    /// `gpu_none_policy_is_bitwise_today`. `cargo test -p qm-gpu --release gpu_none_policy_baseline -- --ignored --nocapture`.
    #[test]
    #[ignore = "Step 1 printer: run on the pre-M7 code"]
    fn gpu_none_policy_baseline() {
        let Some((mut s, _, settings)) = solver(33) else { return };
        for (name, sys) in none_policy_scenes() {
            let (_, stats) = s.solve(&sys, &settings, false).unwrap();
            let e = stats.energy.total;
            let f = s.forces(s.state().unwrap(), &sys.nuclei, settings.sigma).unwrap();
            println!("const GPU_BASE_{name}_E: u64 = 0x{:016x}; // {e:+.12}", e.to_bits());
            println!("const GPU_BASE_{name}_FX: u64 = 0x{:016x}; // {:+.12}", f[0][0].to_bits(), f[0][0]);
        }
    }
}
