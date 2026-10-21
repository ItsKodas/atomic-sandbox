//! The fit: sample store, the design matrix of energies and forces, scale-only standardisation,
//! ridge least squares as Bayesian linear regression with separate energy and force noise
//! levels, evidence-chosen hyperparameters and the predictive variance. Pure f64, CPU. The
//! sufficient statistics live in [`Accumulator`] on the Q′ independent columns, so a refit after
//! new samples is one hyperparameter search (a few hundred Choleskys of Q′, no pass over the
//! samples) and the sidecar (Task 5) can store the statistics instead of the samples.
//!
//! # Design matrix
//!
//! A sample is a cluster of N atoms with species indices s_i (into `Basis::species`), the
//! teacher's total energy E and forces F. The model (crate doc) is
//! E = Σ_i ref[s_i] + Σ_i Σ_k c_{s_i,k} B_{i,k}, linear in the stacked coefficient vector
//! c ∈ ℝ^Q, Q = S·P, laid out in species blocks: column q = s·P + k holds c_{s,k}. Every sample
//! contributes
//!
//! * one **energy row**: features φ_E with block s = Σ_{i: s_i = s} B_i, target
//!   y_E = E − Σ_i ref[s_i] (the binding energy; the references are per species and are not
//!   fitted — see "Isolated-atom limit"), both divided by N when `energy_per_atom` (default),
//!   so that one energy row carries a per-atom quantity (spec review C2c). Row weight
//!   w = `Sample::weight`.
//! * **3N force rows**, one per atom j and component a: features φ_{F,ja} with block s =
//!   −Σ_{i: s_i = s} ∂B_i/∂r_{j,a} (the gradient lists of [`describe`]; the sum over centre
//!   atoms i is what makes the force depend on every atom's coefficient row), target F_{j,a},
//!   weight w = `Sample::weight`. Forces do not depend on the references.
//!
//! # Tied columns (hetero pairs): the Q′ independent columns
//!
//! For every hetero species pair {s, t} and radial index n the two body-2 columns
//! (centre s, neighbour t, n) and (centre t, neighbour s, n) are the same function of the
//! positions on every row (energy: Σ_{i∈s, j∈t} R_n f_c(r_ij) counted from either end; forces:
//! its gradient), so only the SUM of their coefficients is identifiable and the full Gram has an
//! exact null direction per such pair — which no ridge of the size the evidence chooses can
//! resolve in f64 on low-noise data. Everything therefore runs on the Q′ = n_max·S(S+1)/2
//! independent columns of [`Basis::column_classes`]: the accumulator stores the statistics of
//! the class representatives only (the dropped duplicates are identical columns, so this is
//! lossless and Q′² instead of Q² memory), the prior N(0, 1/λ) and the evidence are on the Q′
//! class SUMS (not the N(0, 2/λ) a per-column prior would induce — absorbed by the
//! evidence-chosen λ), and the solution is expanded once, at the end, by an equal split in raw
//! coefficient space, c̃_{s,(t,n)} = c̃_{t,(s,n)} = θ̃/2. The split is a gauge: it never changes
//! a prediction (the hot path only ever uses the pair sum) or its variance (`Model::predict`
//! restricts the query vector to the same representatives). Conventions: the hetero class is
//! standardised with scale[(t, n)] of its s < t representative; `Model::lambda` and
//! `Model::sigma_inv_chol` (Q′×Q′) are in this reduced space; `FitReport::q_prime` is Q′.
//!
//! # Standardisation at solve time (ruling S5: scale only, no mean, no intercept)
//!
//! A single ridge penalty on raw columns penalises in arbitrary units (C1), so the solve works in
//! standardised features φ̃ = D φ with D = diag(1/scale_k) (the same scale for every species
//! block of feature k, which is what `Model::scale` holds), scale_k² = (1/n_E) Σ_{energy rows}
//! Σ_s φ_{E,(s,k)}² — the RMS of feature k over the energy rows, summed over all species blocks
//! including the duplicate of a tied column. A feature identically zero on the energy rows gets
//! scale 1. No mean is subtracted: the energy of an atom with no neighbour inside r_cut must be
//! exactly ref[s] and the binding energy must go to 0 beyond r_cut, which a centred feature or
//! an intercept would break (`Model` docs); the ridge shrinks towards the reference atoms, which
//! is the physically right prior. Force rows use the same D.
//!
//! The scale is a function of the data, so it cannot be fixed before the first sample is seen.
//! The statistics are therefore accumulated RAW — G_κ = ΦᵀWΦ, b_κ = ΦᵀWy, yᵀWy, n_κ per kind
//! κ ∈ {E, F}, together with the per-feature sum of squares over the energy rows — and the
//! scaling is applied as exact algebra at solve time: Φ̃ᵀWΦ̃ = D G D, Φ̃ᵀWy = D b.
//! `Accumulator::add` never needs the final scale, and adding a sample later changes the scale
//! without any re-accumulation.
//!
//! # Two-noise Bayesian linear regression (ruling S6)
//!
//! Energy rows and force rows have different noise: y_r ~ N(φ̃_rᵀc, s_κ²/w_r) with s_E for energy
//! rows and s_F for force rows (per-atom energy residuals of DFT data are ~1e-5 Ha against
//! ~3e-4 Ha/Bohr for forces; one pooled level would enter the energies with 15–40× too little
//! weight — 2–8× worse energies and up to 16× inflated energy variance, review of round 0).
//! With the prior c ~ N(0, λ⁻¹ I) the posterior is Gaussian,
//!
//!   c | data ~ N(m, Σ),   Σ⁻¹ = A = G̃_E/s_E² + G̃_F/s_F² + λ I,   m = A⁻¹ (b̃_E/s_E² + b̃_F/s_F²),
//!
//! i.e. a ridge solve on the whitened Gram. λ is the **prior precision** of a standardised
//! coefficient (1/√λ its prior standard deviation, Hartree per unit standardised feature); it is
//! the λ of `Model::lambda`, `FitOptions::lambda` and the grid below. The former `force_weight`
//! is now the learned ratio s_E²/s_F². The solve is one Cholesky of A (f64; Q′ = 560 at four
//! species is 6e7 flop).
//!
//! # Predictive variance
//!
//! For a query with standardised features φ̃ the predictive distribution of an observation of
//! kind κ is N(φ̃ᵀm, σ²(φ̃)) with
//!
//!   σ²(φ̃) = s_κ² + φ̃ᵀ Σ φ̃ = s_κ² + ‖L_A⁻¹ φ̃‖²,   L_A L_Aᵀ = A,
//!
//! and `Model::sigma_inv_chol` stores L_A. The quadratic form is the leverage of the query in
//! standardised feature space: small in directions the data span, → ‖φ̃_⊥‖²/λ in directions they
//! do not (C3). `Model::predict` evaluates it per atom for the forces (the 3 gradient-feature
//! vectors of each atom, see the model docs for how the per-atom scalar is formed) and once for
//! the total energy (φ̃ = Σ_i φ̃_i); O(Q′²) per vector, Task 9 adds the cadence.
//!
//! # Hyperparameters from the data: evidence and residuals (C2)
//!
//! The noise levels are NOT the solver's path scatter (5e-6 Ha/Bohr): that is the
//! reproducibility of the teacher, not the residual a translation-invariant model can reach
//! against it. The egg-box force (ΣF ≈ 3e-4 Ha/Bohr, not representable) and SCF lag are the
//! floor, ~60× higher; a constant 5e-6 would make every variance 3 600× overconfident and the
//! fallback would never fire. So all three hyperparameters come from the data by type-II maximum
//! likelihood (Bishop PRML §3.5, MacKay 1992). Integrating the coefficients out of
//! Π_r N(y_r | φ̃_rᵀc, s_κ²/w_r)·N(c | 0, λ⁻¹I) gives the log marginal likelihood
//!
//!   ln p(y | λ, s_E, s_F) = −(n_E/2) ln(2πs_E²) − (n_F/2) ln(2πs_F²) + (Q′/2) ln λ − ½ ln det A
//!                           − RSS_E/(2s_E²) − RSS_F/(2s_F²) − λ‖m‖²/2 + ½ Σ_rows ln w,
//!
//! with RSS_κ = Σ_{r∈κ} w_r (y_r − φ̃_rᵀm)² = yᵀWy_κ − 2 mᵀb̃_κ + mᵀG̃_κm (from the statistics;
//! clamped at 0 against cancellation). Stationarity in each noise level at fixed λ (Bishop
//! 3.95 per kind, exact because ∇_m vanishes at m) and in λ at fixed noise (MacKay) read
//!
//!   s_E² = RSS_E/(n_E − γ_E),  γ_E = tr(Σ G̃_E)/s_E²;   s_F² = RSS_F/(n_F − γ_F),  γ_F = tr(Σ G̃_F)/s_F²;
//!   λ = γ/‖m‖²,  γ = γ_E + γ_F = Q′ − λ tr Σ,
//!
//! γ being the effective number of parameters (`dof_eff`), partitioned by kind, and n_κ − γ_κ
//! the residual degrees of freedom of each kind (`residual_dof_e/f`: the relative error of s_κ²
//! is ≈ √(2/dof)). They are solved by alternation: given (s_E, s_F), λ maximises the evidence
//! over the half-decade log grid 1e-8..1e2, extended in the same steps while the maximum sits
//! on an edge (to 1e-14 / 1e8), then refined by golden-section search in ln λ between the grid
//! neighbours of the best point; given λ, the two noise fixed points are iterated together to
//! 1e-6 relative (≤ 100 steps); the outer alternation stops when λ, s_E² and s_F² all change by
//! less than 1e-3 relative (≤ 30 rounds; `FitReport::{hyper_iterations, converged}` say what
//! happened — convergence took 3–6 rounds on every fixture, more when one noise level walks
//! down to its floor). `FitOptions::{lambda, sigma_e, sigma_f}` pin any subset; the rest are
//! still estimated. The noise levels are floored at s_κ² ≥ 1e-16·yᵀWy_κ/n_κ, the f64 resolution
//! of a residual computed from the statistics, so noise-free data give s ≈ 1e-8·rms(y) rather
//! than 0 or a failed Cholesky. When a kind has no residual degrees of freedom (n_κ − γ_κ < 1,
//! i.e. an interpolating fit) the divisor is clamped at 1 — the estimate is then meaningless,
//! which is what `residual_dof_κ` reports.
//!
//! **Ridge floor.** The whitened Gram is only known to ~ε‖A‖, so a ridge below that cannot be
//! resolved by the factorisation (κ > 1/ε: the noise-free Morse fixture reached 2.7e16 before
//! the floor). The factored precision is therefore A = G̃_E/s_E² + G̃_F/s_F² + λ_eff I with
//! λ_eff = max(λ, 4Q′ε·max_k (G̃_E/s_E² + G̃_F/s_F²)_kk): inert for any noisy data (λ is orders of
//! magnitude larger than the floor there), and for noise-free data or `lambda: Some(0.0)` it is
//! the minimum-norm regulariser (dead columns get coefficient 0; their posterior variance is
//! then 1/λ_eff, i.e. "no prior"). The evidence keeps the nominal λ in its (Q′/2) ln λ and
//! λ‖m‖² terms while ln det A and m are at λ_eff; in the floor regime the evidence maximum is
//! then MacKay's α = Q′/‖m‖² at the minimum-norm solution, which is what a noise-free fit
//! reports as λ. The floor scales with Q′ because the Cholesky backward error is O(Q′ε‖A‖)
//! (Task 3 review minor 6): a body-3 basis on a small molecule has Q′ ≈ 300 near-collinear
//! columns whose rounding eigenvalues sit at −Q′ε‖A‖ and broke the 4ε floor on noise-free
//! data; at Q′ = 6 064 the floor is still only 5e-12 relative. Exact ties must be removed by
//! column classes, never left to the floor.
//! [`FitError::Singular`] is left for a Gram that is entirely zero (every feature zero on every
//! row, e.g. all atoms beyond r_cut) with λ = 0, or a numerical failure at the chosen
//! hyperparameters. A failed factorisation at one point of the λ grid only scores that point
//! −∞ (a near-collinear basis — body-3 on a small molecule — has rounding eigenvalues of the
//! whitened Gram near −Q′ε·max A_kk at the 1e-8 end of the grid, where no evidence maximum
//! ever lies); the search is `Singular` only when no grid point factors. With λ = 0 the
//! (Q′/2) ln λ term makes `log_evidence` −∞ (an improper prior has no evidence).
//!
//! A column that is identically zero (a (centre, neighbour) species pair never seen, or a
//! feature that is zero on every row) has A_qq = λ_eff, m_q = 0, posterior variance exactly
//! 1/λ_eff (the prior) and contributes exactly 0 to the evidence ((½) ln λ from the (Q′/2) ln λ
//! term cancels the (½) ln λ of the determinant) and to γ. No special casing is needed.
//!
//! # Isolated-atom limit and the references
//!
//! `ref_energy[s]` is the teacher's isolated-atom energy per species (Z-only: computed by the
//! same solver at the same settings, never a table). Because mean = 0 and there is no
//! intercept, an atom with no neighbour inside r_cut has every feature 0 and energy exactly
//! ref[s]; two atoms beyond r_cut have E = ref[s_1] + ref[s_2] bitwise (the test pins it).
//!
//! # Trust rule (for the provider, Task 4)
//!
//! The fit never decides whether it is trustworthy; it exposes the numbers. The rule Task 4
//! applies: use the surrogate only when `n_samples` ≳ Q′ (distinct geometries — rows are the
//! wrong unit, the 3N force rows of one sample probe one geometry) AND both `residual_dof_e`
//! and `residual_dof_f` ≥ 30 (so the noise levels are known to ≈ 25 %). Units of the stds
//! (`Model::predict`): `energy_std` is the TOTAL energy, floor N·s_E (the row was E/N);
//! `force_std[j]` is the expected ‖δF_j‖, floor √3·s_F — a per-component threshold such as C5's
//! 2e-3 Ha/Bohr compares against force_std/√3.
//!
//! # The large-basis path (M8a Task 5: a shared collection's body-4 basis)
//!
//! A body-4 basis (`Basis::body4`, a shared collection's) has Q′ up to 2 688 at four species, where
//! the search above — a few hundred factorisations and inverses of Q′, ≈ 2.4 Q′³ each — took 639 s
//! already at Q′ 1 232. [`Accumulator::refit`] (what the provider calls) therefore solves a body-4
//! basis by two exact reformulations; every other basis goes through [`Accumulator::solve`]
//! unchanged, bit for bit.
//!
//! **Spectral search.** Write ρ = s_F²/s_E² and t = s_F². Then
//!
//!   A = G̃_E/s_E² + G̃_F/s_F² + λ_eff I = (H_ρ + μ I)/t,   H_ρ = G̃_F + ρ G̃_E,   μ = λ_eff·t,
//!
//! so one eigen-decomposition H_ρ = U diag(h) Uᵀ gives, for every (λ, t) at that ratio, with
//! ĉ = Uᵀ(b̃_F + ρ b̃_E) and m̂_k = ĉ_k/(h_k + μ) (m = U m̂ does not depend on t at fixed μ):
//!
//!   ln det A = Σ_k ln((h_k + μ)/t),   ‖m‖² = Σ_k m̂_k²,   γ = Σ_k h_k/(h_k + μ),
//!   RSS_E/s_E² + RSS_F/s_F² = (ρ yᵀWy_E + yᵀWy_F − 2 m̂·ĉ + Σ_k h_k m̂_k²)/t,
//!
//! i.e. the whole evidence (the formula above, with the nominal λ in (Q′/2) ln λ and λ‖m‖²) in O(Q′)
//! per evaluation. No eigenvectors are formed: H_ρ is reduced to tridiagonal form by Householder
//! reflections (`linalg::Tridiagonal`, 2Q′³) and diagonalised by implicit QL, the reflections and
//! rotations applied to b̃_F + ρ b̃_E alone. Per outer round: the λ search (the same grid and
//! golden section, now O(Q′) per point) and — with neither noise level pinned — the common scale t
//! by MacKay's β update at fixed ρ, t = ρRSS_E + RSS_F over n_E + n_F − γ (stationarity in t:
//! d ln p/dt = [−(n_E + n_F) + γ]/(2t) + (ρRSS_E + RSS_F)/(2t²)), alternated to 1e-6; then one
//! exact factorisation and inverse at (λ, s_E², s_F²) (the vectorised kernels, ≈ Q′³) gives
//! RSS_κ and γ_κ per kind and the per-kind fixed points s_κ² = RSS_κ/(n_κ − γ_κ) the next ρ, as a
//! step in ln ρ whose multiplier halves each time the step reverses direction (the bare fixed point
//! cycled — ρ 460 → 2.1e4 → 1.4e3 → 460 — on a near-noise-free collection fixture; an evidence
//! comparison cannot steer it: ln p carries ~5e-6 relative of rounding between two spectra, the
//! near-null eigenvalues' ε‖H‖ against the ridge floor). A multiplier below 2⁻¹² without the
//! stopping test met is reported as a stall (`RefitKind::Spectral::stalled`, `converged` false).
//! The stopping test, the floors, the pins and `sigma_f_min` are the reference search's; the
//! returned posterior is factored at the final levels. A numerical failure of the spectral search
//! falls back to the reference search. Measured on a Q′ 250 body-4 fixture
//! (`large_basis_search_finds_the_reference_search_optimum`): λ equal to 5 digits, σ_E within
//! 0.05 %, ln p equal to 7 digits, 49 ms against 4 007 ms; at Q′ 2 688 and 10 000 samples a search
//! takes 27.8 s (6 rounds; `references.md` "M8a body-4").
//!
//! **Warm refit.** Between searches the hyperparameters and the standardisation D stand (the
//! search's, frozen), and a new sample only adds rows: A ← A + U Uᵀ with U the sample's k = 3N + 1
//! whitened rows √w Dφ/s_κ. The upper factor R = Lᵀ is updated by k rank-one Givens updates
//! (`linalg::cholesky_update_upper`, O(Q′²) each), the mean solved with it, and γ carried exactly
//! by Woodbury: with V = ΣU and C = I + UᵀV (k×k),
//!
//!   Σ' = Σ − V C⁻¹ Vᵀ,   tr Σ' = tr Σ − tr(C⁻¹ VᵀV),
//!   tr(Σ' G̃_E') = tr(Σ G̃_E) − tr(C⁻¹ Vᵀ G̃_E V) + s_E² (M₀₀ − M₀ᵀ C⁻¹ M₀),   M = UᵀV,
//!
//! (G̃_E the energy Gram before the sample, M₀ the energy row's column; the last term is the new
//! energy row's own w ẽᵀΣ'ẽ), γ_E = tr(Σ G̃_E)/s_E², γ = Q′ − λ_eff tr Σ, γ_F = γ − γ_E. The result
//! is the exact posterior at those hyperparameters — not an approximation of it:
//! `warm_refit_is_the_exact_posterior_at_the_standing_hyperparameters` finds the factor's backward
//! error at 0.5–1.3e-15 and predictions within the spread of two fresh factorisations (both are
//! determined only to κ(A)·ε, κ ~ 6e12 there). What it does not do is re-choose λ and the noise
//! levels; the search runs again once the store has grown by [`WARM_GROWTH`] (×1.25) since the
//! last one, after more than [`WARM_MAX_PENDING`] (64) samples at once, when the options change, or
//! when the accumulator is rebuilt. Every search starts from the data's own levels (never from an
//! earlier search's), so it is a function of the accumulated statistics alone: the warm refit is
//! where the evidence is "warm", and a reopened collection's first fit equals the session's last
//! search bit for bit. The report of a warm refit carries the search's
//! `condition_estimate` and `converged` and 0 `hyper_iterations`. At Q′ 2 688 and 10 000 samples a
//! sample's add + warm refit takes 0.17 s. The warm refit also declines (searches) once the ridge
//! floor 4Q′ε·max diag A of the grown data passes the standing λ_eff: the frozen ridge would no
//! longer be a fresh solve's (`a_long_warm_chain_stays_the_exact_posterior`: after 800 + 10 × 19
//! samples, std within 4.5e-6 and γ within 2e-6 of a fresh solve where λ is above the floor; where
//! the floor binds the chain searches again after 8 warm refits) (`surrogate_perf`, `refit_is_interactive_at_ten_thousand_samples`).
//!
//! **The accumulation** of a body-4 basis sums a sample's 3N force rows into one block first and
//! adds that compensated (one Neumaier step per entry per sample, not per row; threaded over the
//! Gram's rows): the block's own plain sum of ≤ 3N terms is a few ε relative, the totals stay
//! compensated (CLAUDE.md rule 4).
//!
//! **No truncated Σ.** The predictive variance keeps the exact posterior factor. The M5 handoff's
//! truncation Σ ≈ Σ_{k<K} s_k u_k u_kᵀ + s_min I is conservative with s_min = s_K (T − Σ ⪰ 0) but
//! measured useless with it — on a collection-sized body-4 fit (Q′ 284) the floored variance is
//! 39–3·10⁴× the exact one at K = 200 and 10⁹–10¹¹× at K = 50, because the top of Σ's spectrum is
//! the prior's 1/λ on the directions no sample spans (54 of 284 there) — and biased low without
//! it (median ratio 0.004 at K = 50, 0.63 at K = 200):
//! `spectral_truncation_of_the_covariance_is_measured_and_floored`.
//!
//! # Accuracy notes
//!
//! * RSS from the statistics cancels to ~ε·yᵀWy, so `Accumulator::solve` cannot report an RMSE
//!   below ~1e-8·rms(y); [`fit`] has the samples and reports the exact residual RMSE from the
//!   hot path instead (the hyperparameters are the same either way: both go through
//!   `Accumulator::solve`).
//! * The Gram accumulation is a plain f64 sum of rank-1 updates (≤ 1e5 rows → relative error
//!   ≤ 1e-11, and all terms of a diagonal entry are non-negative); the energy reductions of
//!   CLAUDE.md rule 4 are the physics sums in `model.rs`, which stay compensated.
//! * `condition_estimate` is κ₂(A) from 40 power iterations (largest eigenvalue) and 40 inverse
//!   iterations through the Cholesky factor (smallest) — a lower bound that is usually tight to
//!   a few percent.

use crate::descriptor::{describe, Basis, DescribeError};
use crate::envelope::Envelope;
use crate::linalg::{
    cholesky_in_place, cholesky_in_place_fast, cholesky_inverse, cholesky_logdet, cholesky_solve, cholesky_update_upper, dot,
    inverse_from_lower_inverse, lower_inverse_transposed, mat_vec, mat_vec_fast, tridiagonal_ql, upper_solve, Tridiagonal,
};
use crate::model::{energy_and_forces, Model, Provenance};
use std::fmt;
use std::sync::OnceLock;

/// Where a training sample came from (bookkeeping for the panel and the sidecar; the fit does
/// not weight by tag).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleTag {
    /// A step of the live DFT run.
    LiveStep,
    /// A random displacement of the scene (sampling pass).
    Perturbation,
    /// A step of a hot DFT-MD burst (sampling pass).
    HotBurst,
    /// A point of a pair-distance scan (sampling pass).
    Scan,
    /// A step the surrogate handed back to the teacher.
    Fallback,
    /// An isolated-atom reference solve.
    Reference,
    /// A strict-mode audit's teacher answer (the model answered the step; the teacher checked it).
    /// Before 2026-10-18 audits were stored as `Fallback`.
    Audit,
    /// A draft-mode audit's teacher answer: a geometry the guessing model chose, outside what the
    /// strict policy would answer when `providers::surrogate` says so (its module doc, "Draft
    /// mode": such samples are held out of the strict fit).
    DraftAudit,
    /// A teach-around pass's solve near a blocking sample (`providers::surrogate`, "Teaching
    /// around the blockers").
    Targeted,
}

/// One teacher evaluation: positions (Bohr), nuclear charges, total Kohn-Sham energy (Hartree)
/// and forces (Hartree/Bohr), with a relative row weight (1.0 = normal).
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub pos: Vec<[f64; 3]>,
    pub z: Vec<u8>,
    /// Total E_KS, Hartree (the fit subtracts Σ ref_energy itself).
    pub energy: f64,
    pub forces: Vec<[f64; 3]>,
    pub tag: SampleTag,
    pub weight: f64,
}

impl Sample {
    fn shape_ok(&self) -> Result<(), StoreError> {
        if self.pos.len() != self.z.len() || self.pos.len() != self.forces.len() {
            return Err(StoreError::ShapeMismatch {
                n_pos: self.pos.len(),
                n_z: self.z.len(),
                n_forces: self.forces.len(),
            });
        }
        if !(self.weight > 0.0) || !self.weight.is_finite() {
            return Err(StoreError::NonPositiveWeight(self.weight));
        }
        Ok(())
    }
}

/// Samples of one teacher at one set of solve settings.
#[derive(Clone, Debug, PartialEq)]
pub struct SampleStore {
    pub provenance: Provenance,
    pub samples: Vec<Sample>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum StoreError {
    /// The sample was produced at other solve settings than the store's.
    ProvenanceMismatch,
    ShapeMismatch {
        n_pos: usize,
        n_z: usize,
        n_forces: usize,
    },
    NonPositiveWeight(f64),
    /// The same teacher answer (atoms, positions, energy and forces, bit for bit) is already held:
    /// a second copy would double the geometry's weight and give it a leave-one-out twin
    /// (M8a Task 4a fix round 1; the shared collection's duplicate rule, `crate::collection`).
    Duplicate,
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoreError::ProvenanceMismatch => write!(f, "sample provenance differs from the store's"),
            StoreError::ShapeMismatch { n_pos, n_z, n_forces } => write!(
                f,
                "sample has {n_pos} positions, {n_z} charges and {n_forces} forces"
            ),
            StoreError::NonPositiveWeight(w) => write!(f, "sample weight {w} is not positive"),
            StoreError::Duplicate => write!(f, "an identical teacher answer is already held"),
        }
    }
}

impl std::error::Error for StoreError {}

impl SampleStore {
    pub fn new(provenance: Provenance) -> SampleStore {
        SampleStore {
            provenance,
            samples: Vec::new(),
        }
    }

    /// Add a sample known to come from this store's settings (shape and weight are checked).
    pub fn push(&mut self, s: Sample) -> Result<(), StoreError> {
        s.shape_ok()?;
        self.samples.push(s);
        Ok(())
    }

    /// Add a sample produced at `provenance`, which must equal the store's.
    pub fn push_checked(&mut self, s: Sample, provenance: &Provenance) -> Result<(), StoreError> {
        if *provenance != self.provenance {
            return Err(StoreError::ProvenanceMismatch);
        }
        self.push(s)
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Fit settings. The hyperparameters are `None` by default: estimated from the data (module
/// doc); `Some` pins them.
#[derive(Clone, Debug, PartialEq)]
pub struct FitOptions {
    /// Prior precision λ of the standardised coefficients.
    pub lambda: Option<f64>,
    /// Noise level of a unit-weight energy row (Hartree; per atom when `energy_per_atom`).
    pub sigma_e: Option<f64>,
    /// Noise level of a unit-weight force-component row (Hartree/Bohr).
    pub sigma_f: Option<f64>,
    /// A floor on the force-component noise level s_F (Hartree/Bohr; default 0 = none): a prior that the labels
    /// are noisy at least this much, for a teacher whose own path scatter is measured and is not representable
    /// (M7: the GPU pseudopotential path, ≈ 5e-5 Ha/Bohr per component at the sampling tolerance, up to ~7e-5 on
    /// warm chains — `providers`' `PSEUDO_FORCE_NOISE`). When the estimated s_F falls below it the fit is redone
    /// with s_F pinned at the floor (λ and s_E are still estimated), so the posterior is never more confident
    /// about forces than the labels can support. Ignored when `sigma_f` is pinned.
    pub sigma_f_min: f64,
    /// Energy rows carry E/N and features/N (default true).
    pub energy_per_atom: bool,
}

impl Default for FitOptions {
    fn default() -> Self {
        FitOptions {
            lambda: None,
            sigma_e: None,
            sigma_f: None,
            sigma_f_min: 0.0,
            energy_per_atom: true,
        }
    }
}

/// What the fit found. RMSEs are over the training rows: from the exact residuals in [`fit`],
/// from the statistics (cancellation-limited at ~1e-8·rms(y)) in [`Accumulator::solve`].
#[derive(Clone, Debug, PartialEq)]
pub struct FitReport {
    /// Hartree per atom (per sample if `energy_per_atom` is off).
    pub rmse_energy_per_atom: f64,
    /// Hartree/Bohr per component.
    pub rmse_force: f64,
    /// Energy-row noise level s_E (per atom when `energy_per_atom`).
    pub sigma_e: f64,
    /// Force-component noise level s_F.
    pub sigma_f: f64,
    /// Prior precision λ (module doc).
    pub lambda: f64,
    pub log_evidence: f64,
    /// Effective number of parameters γ = γ_E + γ_F.
    pub dof_eff: f64,
    /// Effective parameters determined by the energy rows / the force rows.
    pub gamma_e: f64,
    pub gamma_f: f64,
    /// Residual degrees of freedom n_E − γ_E and n_F − γ_F (trust rule: both ≥ 30).
    pub residual_dof_e: f64,
    pub residual_dof_f: f64,
    pub n_rows: usize,
    pub n_samples: usize,
    /// Independent columns Q′ (trust rule: n_samples ≳ Q′).
    pub q_prime: usize,
    /// κ₂ estimate of the posterior precision A. **After a warm refit** (`Accumulator::refit` on a
    /// body-4 basis, module doc "The large-basis path"): the last search's, not re-estimated.
    pub condition_estimate: f64,
    /// Outer alternation rounds of the hyperparameter search, and whether it met 1e-3. **After a
    /// warm refit**: 0 rounds, and `converged` is the last search's (no search ran).
    pub hyper_iterations: usize,
    pub converged: bool,
}

impl FitReport {
    /// Total residual degrees of freedom n_rows − γ.
    pub fn residual_dof(&self) -> f64 {
        self.residual_dof_e + self.residual_dof_f
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum FitError {
    NoSamples,
    /// No finite reference energy for this Z.
    MissingReference(u8),
    /// `ref_energy` has a different length than the basis has species.
    ReferenceLength { expected: usize, got: usize },
    /// The posterior precision is not positive definite at any λ the search tried: λ = 0 with an
    /// all-zero Gram (every feature zero on every row), or rounding in the Gram whose negative
    /// null eigenvalues, whitened by a noise floor, outweigh the ridge (module doc, "Ridge floor";
    /// a naive sum did this at 30 000 noise-free dimers — the Gram is compensated since M6 Task 11
    /// round 3).
    Singular,
    Describe(DescribeError),
    /// The solve produced a non-finite number (named): never installed, so the caller keeps its
    /// previous model (activity Task 3b; a NaN label or an overflow in the solve).
    NonFinite(&'static str),
}

impl fmt::Display for FitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FitError::NoSamples => write!(f, "no samples to fit"),
            FitError::MissingReference(z) => write!(f, "no reference energy for Z = {z}"),
            FitError::ReferenceLength { expected, got } => {
                write!(f, "{got} reference energies for {expected} species")
            }
            FitError::Singular => write!(f, "posterior precision is not positive definite at any λ tried (an all-zero Gram, or the Gram's rounding outweighing the ridge)"),
            FitError::Describe(e) => write!(f, "{e}"),
            FitError::NonFinite(what) => write!(f, "the fit's {what} is not finite (a NaN or infinite label, or an overflow in the solve): not installed"),
        }
    }
}

impl std::error::Error for FitError {}

impl From<DescribeError> for FitError {
    fn from(e: DescribeError) -> Self {
        FitError::Describe(e)
    }
}

/// Sufficient statistics of the weighted least-squares problem on the Q′ independent columns,
/// raw (unstandardised, unwhitened; module doc), kept per row kind so the two noise levels can
/// be estimated. The statistics are public for the sidecar (`new` then fill them); `add` is
/// the only sane way to compute them.
///
/// **Compensated** (M6 Task 11 round 3, CLAUDE.md rule 4): G, b and yᵀy are sums over every row
/// (Σ(1 + 3N) per sample), accumulated with Neumaier's algorithm — the public fields hold the
/// running sums, a private compensation the lost low-order parts, and [`Accumulator::sums`] their
/// total. A naive sum's rounding grows with the row count (the null eigenvalues of the
/// standardised Gram, which `novelty` compares with Q′·ε·tr: 3.6e-16·tr at 300 dimers,
/// 9.4e-16·tr at 3000 — margin 18× → 7×, gone near 10⁵ samples); compensated it stays at the
/// eigensolver's own level (`references.md`, "Null-space margin"). A sidecar stores the totals
/// and reads them back with a zero compensation.
#[derive(Clone, Debug)]
pub struct Accumulator {
    pub basis: Basis,
    /// Independent columns Q′ (`Basis::column_classes().len()`).
    pub q_prime: usize,
    /// Energy rows: ΦᵀWΦ (Q′×Q′ row-major), ΦᵀWy, yᵀWy, row count, Σ ln w.
    pub gram_e: Vec<f64>,
    pub rhs_e: Vec<f64>,
    pub yy_e: f64,
    pub n_e: usize,
    pub sum_log_w_e: f64,
    /// Force rows, likewise.
    pub gram_f: Vec<f64>,
    pub rhs_f: Vec<f64>,
    pub yy_f: f64,
    pub n_f: usize,
    pub sum_log_w_f: f64,
    /// Per feature k: Σ over energy rows and species blocks (duplicates included) of
    /// φ_{E,(s,k)}² (for the scale).
    pub sumsq: Vec<f64>,
    /// Range and spread of the per-atom descriptors seen, per (species, feature): the
    /// out-of-distribution envelope of the provider (`envelope` module doc).
    pub envelope: Envelope,
    pub n_samples: usize,
    /// `energy_per_atom` of the first `add`; later adds must agree.
    pub energy_per_atom: Option<bool>,
    /// Full column q = s·P + k → its class index, `None` for the duplicate of a tied column.
    col_index: Vec<Option<usize>>,
    classes: Vec<Vec<usize>>,
    /// Neumaier compensations of gram_e, rhs_e, yy_e, gram_f, rhs_f, yy_f (struct doc).
    comp: Compensation,
    /// The large-basis path (module doc): the warm state of the last search, the rows of the
    /// samples added since the last refit (kept only while a warm state stands), how the last
    /// refit went.
    warm: Option<Box<WarmState>>,
    pending: Vec<PendingRows>,
    last_refit: RefitKind,
    // Scratch buffers reused across `add` calls.
    phi: Vec<f64>,
    rows: Vec<f64>,
    support: Vec<usize>,
}

impl Accumulator {
    pub fn new(basis: &Basis) -> Accumulator {
        let p = basis.n_features();
        let q = basis.n_species() * p;
        let classes = basis.column_classes();
        let qr = classes.len();
        let mut col_index = vec![None; q];
        for (c, class) in classes.iter().enumerate() {
            col_index[class[0]] = Some(c);
        }
        Accumulator {
            basis: basis.clone(),
            q_prime: qr,
            gram_e: vec![0.0; qr * qr],
            rhs_e: vec![0.0; qr],
            yy_e: 0.0,
            n_e: 0,
            sum_log_w_e: 0.0,
            gram_f: vec![0.0; qr * qr],
            rhs_f: vec![0.0; qr],
            yy_f: 0.0,
            n_f: 0,
            sum_log_w_f: 0.0,
            sumsq: vec![0.0; p],
            envelope: Envelope::new(basis),
            n_samples: 0,
            energy_per_atom: None,
            col_index,
            classes,
            comp: Compensation {
                gram_e: vec![0.0; qr * qr],
                rhs_e: vec![0.0; qr],
                yy_e: 0.0,
                gram_f: vec![0.0; qr * qr],
                rhs_f: vec![0.0; qr],
                yy_f: 0.0,
            },
            warm: None,
            pending: Vec::new(),
            last_refit: RefitKind::None,
            phi: vec![0.0; qr],
            rows: Vec::new(),
            support: Vec::with_capacity(qr),
        }
    }

    /// The compensated totals of G, b and yᵀy (struct doc): what the fit, the null-space test and
    /// the sidecar use.
    pub fn sums(&self) -> Sums {
        let add = |s: &[f64], c: &[f64]| -> Vec<f64> { s.iter().zip(c).map(|(a, b)| a + b).collect() };
        Sums {
            gram_e: add(&self.gram_e, &self.comp.gram_e),
            rhs_e: add(&self.rhs_e, &self.comp.rhs_e),
            yy_e: self.yy_e + self.comp.yy_e,
            gram_f: add(&self.gram_f, &self.comp.gram_f),
            rhs_f: add(&self.rhs_f, &self.comp.rhs_f),
            yy_f: self.yy_f + self.comp.yy_f,
        }
    }

    /// Rows accumulated so far (energy + force).
    pub fn n_rows(&self) -> usize {
        self.n_e + self.n_f
    }

    /// Add one sample's energy row and 3N force rows.
    ///
    /// # Panics
    /// If `basis` differs from the accumulator's, the sample's shapes disagree, `ref_energy`
    /// is shorter than the species list, or `opts.energy_per_atom` differs from the first
    /// call's.
    pub fn add(
        &mut self,
        sample: &Sample,
        basis: &Basis,
        ref_energy: &[f64],
        opts: &FitOptions,
    ) -> Result<(), DescribeError> {
        assert_eq!(*basis, self.basis, "Accumulator::add: basis differs from the accumulator's");
        assert!(sample.shape_ok().is_ok(), "Accumulator::add: sample shapes disagree");
        assert!(
            ref_energy.len() >= basis.n_species(),
            "Accumulator::add: ref_energy shorter than the species list"
        );
        let per_atom = *self.energy_per_atom.get_or_insert(opts.energy_per_atom);
        assert_eq!(
            per_atom, opts.energy_per_atom,
            "Accumulator::add: energy_per_atom differs from the first sample's"
        );
        let s = basis.species_indices(&sample.z)?;
        let desc = describe(basis, &sample.pos, &sample.z)?;
        let p = basis.n_features();
        let qr = self.q_prime;
        let n = sample.pos.len();
        let w = sample.weight;

        // Energy row on the representatives (a duplicate column's value is its representative's).
        let mut phi = std::mem::take(&mut self.phi);
        phi.iter_mut().for_each(|v| *v = 0.0);
        let mut y = sample.energy;
        for (i, d) in desc.iter().enumerate() {
            self.envelope.add_atom(s[i], &d.b);
            y -= ref_energy[s[i]];
            let off = s[i] * p;
            for k in 0..p {
                if let Some(c) = self.col_index[off + k] {
                    phi[c] += d.b[k];
                }
            }
        }
        if per_atom {
            let inv = 1.0 / n as f64;
            phi.iter_mut().for_each(|v| *v *= inv);
            y *= inv;
        }
        rank1_update(&mut self.gram_e, &mut self.comp.gram_e, &mut self.rhs_e, &mut self.comp.rhs_e, qr, &phi, y, w, &mut self.support);
        neumaier_add(&mut self.yy_e, &mut self.comp.yy_e, w * y * y);
        self.n_e += 1;
        self.sum_log_w_e += w.ln();
        for (c, class) in self.classes.iter().enumerate() {
            let v2 = phi[c] * phi[c];
            for &q in class {
                self.sumsq[q % p] += v2;
            }
        }
        self.phi = phi;

        // Force rows: φ_{F,(j,a)} block s = −Σ_{i: s_i = s} ∂B_i/∂r_{j,a}, 3N rows of Q′.
        let mut rows = std::mem::take(&mut self.rows);
        rows.clear();
        rows.resize(3 * n * qr, 0.0);
        for (i, d) in desc.iter().enumerate() {
            let off = s[i] * p;
            for (j, g) in &d.grad {
                let base = 3 * j * qr;
                for k in 0..p {
                    if let Some(c) = self.col_index[off + k] {
                        rows[base + c] -= g[k][0];
                        rows[base + qr + c] -= g[k][1];
                        rows[base + 2 * qr + c] -= g[k][2];
                    }
                }
            }
        }
        if self.basis.body4 {
            // the large-basis path (module doc): the sample's 3N rows summed into one block first,
            // which is then added compensated — one Neumaier step per entry per sample instead of
            // per row, threaded over the Gram's rows
            let ys: Vec<f64> = sample.forces.iter().flat_map(|f| f.iter().copied()).collect();
            batch_update(&mut self.gram_f, &mut self.comp.gram_f, &mut self.rhs_f, &mut self.comp.rhs_f, qr, &rows[..3 * n * qr], &ys, w);
            for &f in &ys {
                neumaier_add(&mut self.yy_f, &mut self.comp.yy_f, w * f * f);
                self.n_f += 1;
                self.sum_log_w_f += w.ln();
            }
        } else {
            for j in 0..n {
                for a in 0..3 {
                    let f = sample.forces[j][a];
                    let row = &rows[(3 * j + a) * qr..(3 * j + a + 1) * qr];
                    rank1_update(&mut self.gram_f, &mut self.comp.gram_f, &mut self.rhs_f, &mut self.comp.rhs_f, qr, row, f, w, &mut self.support);
                    neumaier_add(&mut self.yy_f, &mut self.comp.yy_f, w * f * f);
                    self.n_f += 1;
                    self.sum_log_w_f += w.ln();
                }
            }
        }
        if self.warm.is_some() {
            self.pending.push(PendingRows { w, e: self.phi.clone(), f: rows[..3 * n * qr].to_vec() });
        }
        self.rows = rows;
        self.n_samples += 1;
        Ok(())
    }

    /// Standardise, choose the hyperparameters, solve, and build the model (module doc).
    pub fn solve(
        &self,
        opts: &FitOptions,
        provenance: &Provenance,
        ref_energy: &[f64],
    ) -> Result<(Model, FitReport), FitError> {
        let (scale, d, problem) = self.prepare(ref_energy, None)?;
        let (hyper, sol) = problem.estimate(opts)?;
        let log_evidence = problem.evidence(hyper.lambda, hyper.s_e2, hyper.s_f2, &sol);
        let condition = condition_estimate(&sol.a, &sol.chol, self.q_prime);
        self.assemble(opts, provenance, ref_energy, &scale, &d, &hyper, &sol, log_evidence, condition)
    }

    /// The standardisation (from the energy rows, or `frozen` = a warm state's (scale, d)) and the
    /// standardised two-noise problem (module doc). Refuses an empty or non-finite accumulator.
    fn prepare(&self, ref_energy: &[f64], frozen: Option<(&[f64], &[f64])>) -> Result<(Vec<f64>, Vec<f64>, TwoNoise), FitError> {
        if self.n_samples == 0 || self.n_e == 0 {
            return Err(FitError::NoSamples);
        }
        check_references(&self.basis, ref_energy)?;
        let p = self.basis.n_features();
        let q = self.q_prime;
        let n_e = self.n_e as f64;
        let n_f = self.n_f as f64;
        let (scale, d) = match frozen {
            Some((s, d)) => (s.to_vec(), d.to_vec()),
            None => {
                // Scale from the energy rows; D per representative column through k = q_full mod P.
                let scale: Vec<f64> = self
                    .sumsq
                    .iter()
                    .map(|&ss| if ss > 0.0 { (ss / n_e).sqrt() } else { 1.0 })
                    .collect();
                let d: Vec<f64> = self.classes.iter().map(|c| 1.0 / scale[c[0] % p]).collect();
                (scale, d)
            }
        };
        let standardise = |g: &[f64]| -> Vec<f64> {
            let mut out = vec![0.0; q * q];
            for i in 0..q {
                for j in 0..q {
                    out[i * q + j] = d[i] * g[i * q + j] * d[j];
                }
            }
            out
        };
        let sums = self.sums();
        // a non-finite label (or feature) makes every statistic it touches non-finite: refuse it
        // by name rather than let the λ search report it as `Singular` (activity Task 3b)
        let finite = |v: &[f64]| v.iter().all(|x| x.is_finite());
        if !(finite(&sums.gram_e) && finite(&sums.gram_f) && finite(&sums.rhs_e) && finite(&sums.rhs_f) && sums.yy_e.is_finite() && sums.yy_f.is_finite()) {
            return Err(FitError::NonFinite("training statistics (a label or a feature)"));
        }
        let problem = TwoNoise {
            g_e: standardise(&sums.gram_e),
            g_f: standardise(&sums.gram_f),
            b_e: (0..q).map(|i| d[i] * sums.rhs_e[i]).collect(),
            b_f: (0..q).map(|i| d[i] * sums.rhs_f[i]).collect(),
            yy_e: sums.yy_e,
            yy_f: sums.yy_f,
            n_e,
            n_f,
            q,
            sum_log_w: self.sum_log_w_e + self.sum_log_w_f,
            floor_e2: (1e-16 * sums.yy_e / n_e).max(f64::MIN_POSITIVE),
            floor_f2: if n_f > 0.0 {
                (1e-16 * sums.yy_f / n_f).max(f64::MIN_POSITIVE)
            } else {
                f64::MIN_POSITIVE
            },
        };
        Ok((scale, d, problem))
    }

    /// The model and the report from a solve (module doc): the reduced solution expanded over the
    /// column classes, every non-finite outcome refused.
    #[allow(clippy::too_many_arguments)]
    fn assemble(
        &self,
        opts: &FitOptions,
        provenance: &Provenance,
        ref_energy: &[f64],
        scale: &[f64],
        d: &[f64],
        hyper: &Hyper,
        sol: &Solved,
        log_evidence: f64,
        condition: f64,
    ) -> Result<(Model, FitReport), FitError> {
        let basis = &self.basis;
        let p = basis.n_features();
        let n_s = basis.n_species();
        let q = self.q_prime;
        let n_e = self.n_e as f64;
        let n_f = self.n_f as f64;
        let n_rows = self.n_e + self.n_f;
        let finite = |v: &[f64]| v.iter().all(|x| x.is_finite());
        let sigma_e = hyper.s_e2.sqrt();
        let sigma_f = hyper.s_f2.sqrt();

        // Expand the reduced solution: raw θ̃ = θ/scale_rep, split equally over the class, and
        // re-standardise each full column by its own feature scale (`Model::coeffs` contract).
        let mut full = vec![0.0; n_s * p];
        for (i, class) in self.classes.iter().enumerate() {
            let theta_raw = sol.m[i] * d[i] / class.len() as f64;
            for &c in class {
                full[c] = theta_raw * scale[c % p];
            }
        }
        let coeffs: Vec<Vec<f64>> = (0..n_s).map(|s| full[s * p..(s + 1) * p].to_vec()).collect();
        // nothing non-finite is installed (activity Task 3b): the caller keeps its previous model
        if !finite(&full) {
            return Err(FitError::NonFinite("coefficients"));
        }
        if !(sigma_e.is_finite() && sigma_f.is_finite() && hyper.lambda.is_finite()) {
            return Err(FitError::NonFinite("noise levels or λ"));
        }
        if !finite(&sol.chol) {
            return Err(FitError::NonFinite("posterior factor"));
        }
        if !(n_e - sol.gamma_e).is_finite() || !(n_f - sol.gamma_f).is_finite() || !(sol.rss_e.is_finite() && sol.rss_f.is_finite()) {
            return Err(FitError::NonFinite("residual degrees of freedom or residuals"));
        }
        let model = Model {
            basis: basis.clone(),
            coeffs,
            mean: vec![0.0; p],
            scale: scale.to_vec(),
            ref_energy: ref_energy.to_vec(),
            sigma_e,
            sigma_f,
            energy_per_atom: self.energy_per_atom.unwrap_or(opts.energy_per_atom),
            lambda: hyper.lambda,
            sigma_inv_chol: Some(sol.chol.clone()),
            n_samples: self.n_samples,
            provenance: provenance.clone(),
            representatives: OnceLock::new(),
        };
        let report = FitReport {
            rmse_energy_per_atom: if n_e > 0.0 { (sol.rss_e / n_e).sqrt() } else { 0.0 },
            rmse_force: if n_f > 0.0 { (sol.rss_f / n_f).sqrt() } else { 0.0 },
            sigma_e,
            sigma_f,
            lambda: hyper.lambda,
            log_evidence,
            dof_eff: sol.gamma_e + sol.gamma_f,
            gamma_e: sol.gamma_e,
            gamma_f: sol.gamma_f,
            residual_dof_e: n_e - sol.gamma_e,
            residual_dof_f: n_f - sol.gamma_f,
            n_rows,
            n_samples: self.n_samples,
            q_prime: q,
            condition_estimate: condition,
            hyper_iterations: hyper.iterations,
            converged: hyper.converged,
        };
        Ok((model, report))
    }
}

/// How the last [`Accumulator::refit`] solved (module doc, "The large-basis path").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefitKind {
    /// Not refitted yet.
    None,
    /// A basis without body-4 (every per-scene model): [`Accumulator::solve`], bit for bit.
    Full,
    /// A body-4 basis: the hyperparameters searched spectrally, `rounds` outer rounds (one
    /// factorisation and ≥ 1 tridiagonalisations each); `stalled` when the ρ step found no ascent
    /// before the 1e-3 test was met (the search stopped at the profile's maximum to the evidence's
    /// resolution; the report says `converged: false` then).
    Spectral { rounds: usize, stalled: bool },
    /// A body-4 basis: the samples added since the last refit folded into the posterior at the
    /// standing hyperparameters and standardisation by rank-one updates of its factor.
    Warm { samples: usize },
}

/// A warm refit searches the hyperparameters again once the stored samples have grown by this
/// factor since the last search (module doc, "The large-basis path").
pub const WARM_GROWTH: f64 = 1.25;
/// … or when more than this many samples arrived since the last refit (each costs ≈ (3N+1)·8Q′²
/// flops warm; a search costs ≈ 3Q′³ per round).
pub const WARM_MAX_PENDING: usize = 64;

/// What a warm refit continues from (module doc, "The large-basis path"): the last search's
/// standardisation and hyperparameters, the upper factor R = Lᵀ of the posterior precision at
/// them, tr Σ and tr(Σ G̃_E) (γ by Sherman–Morrison–Woodbury), and the bookkeeping of the search.
#[derive(Clone, Debug)]
struct WarmState {
    scale: Vec<f64>,
    d: Vec<f64>,
    lambda: f64,
    lambda_eff: f64,
    s_e2: f64,
    s_f2: f64,
    r: Vec<f64>,
    tr_sigma: f64,
    tr_sigma_ge: f64,
    opts: FitOptions,
    n_search: usize,
    converged: bool,
    condition: f64,
}

/// One sample's raw rows on the Q′ columns, kept for the next warm refit: the energy row (÷ N
/// when per atom) and its 3N force rows (row-major), with the sample weight.
#[derive(Clone, Debug)]
struct PendingRows {
    w: f64,
    e: Vec<f64>,
    f: Vec<f64>,
}

impl Accumulator {
    /// The refit the provider calls (module doc, "The large-basis path"): [`Accumulator::solve`]
    /// for a basis without body-4 (bit for bit); for a body-4 basis a warm refit when one is
    /// valid (the same options, the store grown by less than [`WARM_GROWTH`] since the last search,
    /// at most [`WARM_MAX_PENDING`] samples since the last refit), else the spectral search.
    pub fn refit(&mut self, opts: &FitOptions, provenance: &Provenance, ref_energy: &[f64]) -> Result<(Model, FitReport), FitError> {
        if !self.basis.body4 {
            self.warm = None;
            self.pending.clear();
            self.last_refit = RefitKind::Full;
            return self.solve(opts, provenance, ref_energy);
        }
        let pending = self.pending.len();
        let warm_ok = self.warm.as_ref().is_some_and(|w| {
            w.opts == *opts && (self.n_samples as f64) < WARM_GROWTH * w.n_search as f64 && pending <= WARM_MAX_PENDING
        });
        if warm_ok {
            if let Ok(out) = self.solve_warm(opts, provenance, ref_energy) {
                self.last_refit = RefitKind::Warm { samples: pending };
                return Ok(out);
            }
        }
        match self.solve_spectral(opts, provenance, ref_energy) {
            Ok(out) => Ok(out),
            // the reference search as the fallback (slow at a large Q′, never wrong): a spectral
            // failure is numerical, not the data's verdict
            Err(FitError::Singular) => {
                self.last_refit = RefitKind::Full;
                self.solve(opts, provenance, ref_energy)
            }
            Err(e) => Err(e),
        }
    }

    /// How the last [`refit`](Self::refit) solved.
    pub fn last_refit(&self) -> RefitKind {
        self.last_refit
    }

    /// The spectral search (module doc), from the data's own starting levels — never from an
    /// earlier search's, so a search is a function of the accumulated statistics alone (a
    /// reopened collection's first fit equals the session's last search, bit for bit); leaves a
    /// warm state for the next refit.
    fn solve_spectral(&mut self, opts: &FitOptions, provenance: &Provenance, ref_energy: &[f64]) -> Result<(Model, FitReport), FitError> {
        let start = None;
        self.warm = None;
        self.pending.clear();
        let (scale, d, problem) = self.prepare(ref_energy, None)?;
        let (hyper, sol, extra) = problem.estimate_spectral(opts, start)?;
        let log_evidence = problem.evidence(hyper.lambda, hyper.s_e2, hyper.s_f2, &sol);
        let q = self.q_prime;
        let condition = condition_estimate(&sol.a, &sol.chol, q);
        let out = self.assemble(opts, provenance, ref_energy, &scale, &d, &hyper, &sol, log_evidence, condition)?;
        let mut r = vec![0.0; q * q];
        for i in 0..q {
            for j in 0..=i {
                r[j * q + i] = sol.chol[i * q + j];
            }
        }
        self.warm = Some(Box::new(WarmState {
            scale,
            d,
            lambda: hyper.lambda,
            lambda_eff: extra.lambda_eff,
            s_e2: hyper.s_e2,
            s_f2: hyper.s_f2,
            r,
            tr_sigma: extra.tr_sigma,
            tr_sigma_ge: extra.tr_sigma_ge,
            opts: opts.clone(),
            n_search: self.n_samples,
            converged: hyper.converged,
            condition,
        }));
        self.last_refit = RefitKind::Spectral { rounds: hyper.iterations, stalled: extra.stalled };
        Ok(out)
    }

    /// The warm refit (module doc): every pending sample's rows folded into R by rank-one updates,
    /// tr Σ and tr(Σ G̃_E) carried by Woodbury, the mean solved with the updated factor. On any
    /// failure the warm state is dropped (the caller searches).
    fn solve_warm(&mut self, opts: &FitOptions, provenance: &Provenance, ref_energy: &[f64]) -> Result<(Model, FitReport), FitError> {
        let mut w = self.warm.take().expect("refit checked the warm state");
        let pending = std::mem::take(&mut self.pending);
        let (_, _, problem) = self.prepare(ref_energy, Some((&w.scale, &w.d)))?;
        let q = self.q_prime;
        let (ie, i_f) = (1.0 / w.s_e2, 1.0 / w.s_f2);
        // the ridge floor 4Q′ε·max diag A grows with the data; once it passes the standing λ_eff
        // the warm posterior (whose ridge is frozen) is no longer the one a fresh solve gives —
        // search instead (fix round 1, the review's M2: a long chain measured the gap)
        let max_diag = (0..q).map(|i| problem.g_e[i * q + i] * ie + problem.g_f[i * q + i] * i_f).fold(0.0f64, f64::max);
        if 4.0 * q.max(1) as f64 * f64::EPSILON * max_diag > w.lambda_eff {
            return Err(FitError::Singular);
        }
        // the standardised energy rows of the pending samples: G̃_E before sample j is the
        // accumulator's minus the rows of samples j.. (they are already in its sums)
        let e_rows: Vec<Vec<f64>> = pending.iter().map(|ps| (0..q).map(|i| w.d[i] * ps.e[i]).collect()).collect();
        let mut tr_s = w.tr_sigma;
        let mut t_e = w.tr_sigma_ge;
        let mut gv = vec![0.0; q];
        for (j, ps) in pending.iter().enumerate() {
            let n3 = ps.f.len() / q;
            let k = 1 + n3;
            // U: the whitened rows (A += Σ u uᵀ)
            let (se, sf) = ((ps.w * ie).sqrt(), (ps.w * i_f).sqrt());
            let mut u: Vec<Vec<f64>> = Vec::with_capacity(k);
            u.push(e_rows[j].iter().map(|x| x * se).collect());
            for rr in 0..n3 {
                u.push((0..q).map(|i| w.d[i] * ps.f[rr * q + i] * sf).collect());
            }
            // V = Σ U, C = I + UᵀV (k×k, symmetric positive definite)
            let v: Vec<Vec<f64>> = u
                .iter()
                .map(|x| {
                    let mut y = x.clone();
                    upper_solve(&w.r, q, &mut y);
                    y
                })
                .collect();
            let mut c = vec![0.0; k * k];
            for a in 0..k {
                for b in a..k {
                    let m = 0.5 * (dot(&u[a], &v[b]) + dot(&u[b], &v[a]));
                    c[a * k + b] = m + if a == b { 1.0 } else { 0.0 };
                    c[b * k + a] = c[a * k + b];
                }
            }
            let mut cl = c.clone();
            cholesky_in_place(&mut cl, k).map_err(|_| FitError::Singular)?;
            // tr(C⁻¹ X) for a symmetric k×k X
            let tr_cinv = |x: &[f64]| -> f64 {
                let mut t = 0.0;
                let mut col = vec![0.0; k];
                for b in 0..k {
                    for a in 0..k {
                        col[a] = x[a * k + b];
                    }
                    cholesky_solve(&cl, k, &mut col);
                    t += col[b];
                }
                t
            };
            // tr Σ' = tr Σ − tr(C⁻¹ VᵀV)
            let mut vtv = vec![0.0; k * k];
            for a in 0..k {
                for b in a..k {
                    vtv[a * k + b] = dot(&v[a], &v[b]);
                    vtv[b * k + a] = vtv[a * k + b];
                }
            }
            tr_s -= tr_cinv(&vtv);
            // tr(Σ' G̃_pre) = tr(Σ G̃_pre) − tr(C⁻¹ Vᵀ G̃_pre V), G̃_pre = G̃_E − Σ_{i ≥ j} w_i ẽ_i ẽ_iᵀ
            let ve: Vec<Vec<f64>> = (j..pending.len()).map(|i| v.iter().map(|vb| dot(vb, &e_rows[i])).collect()).collect();
            let mut vgv = vec![0.0; k * k];
            for b in 0..k {
                mat_vec_fast(&problem.g_e, q, &v[b], &mut gv);
                for a in 0..=b {
                    let mut x = dot(&v[a], &gv);
                    for (ii, row) in ve.iter().enumerate() {
                        x -= pending[j + ii].w * row[a] * row[b];
                    }
                    vgv[a * k + b] = x;
                    vgv[b * k + a] = x;
                }
            }
            t_e -= tr_cinv(&vgv);
            // + w ẽᵀ Σ' ẽ = s_E² (M₀₀ − M₀ᵀ C⁻¹ M₀), M = UᵀV = C − I, M₀ its first column
            let m0: Vec<f64> = (0..k).map(|a| c[a * k] - if a == 0 { 1.0 } else { 0.0 }).collect();
            let mut x = m0.clone();
            cholesky_solve(&cl, k, &mut x);
            t_e += w.s_e2 * (m0[0] - dot(&m0, &x));
            // fold the rows into the factor
            for mut col in u {
                cholesky_update_upper(&mut w.r, q, &mut col);
            }
        }
        let mut m: Vec<f64> = (0..q).map(|i| problem.b_e[i] * ie + problem.b_f[i] * i_f).collect();
        upper_solve(&w.r, q, &mut m);
        let rss_e = residual_ss(&problem.g_e, &problem.b_e, problem.yy_e, q, &m);
        let rss_f = residual_ss(&problem.g_f, &problem.b_f, problem.yy_f, q, &m);
        let gamma_e = t_e * ie;
        let gamma = q as f64 - w.lambda_eff * tr_s;
        let mut chol = vec![0.0; q * q];
        for i in 0..q {
            for j in 0..=i {
                chol[i * q + j] = w.r[j * q + i];
            }
        }
        let logdet = cholesky_logdet(&chol, q);
        let m_norm2 = m.iter().map(|v| v * v).sum();
        let sol = Solved { a: Vec::new(), chol, m, rss_e, rss_f, gamma_e, gamma_f: gamma - gamma_e, logdet, m_norm2 };
        let hyper = Hyper { lambda: w.lambda, s_e2: w.s_e2, s_f2: w.s_f2, iterations: 0, converged: w.converged };
        let log_evidence = problem.evidence(hyper.lambda, hyper.s_e2, hyper.s_f2, &sol);
        let out = self.assemble(opts, provenance, ref_energy, &w.scale, &w.d, &hyper, &sol, log_evidence, w.condition)?;
        w.tr_sigma = tr_s;
        w.tr_sigma_ge = t_e;
        self.warm = Some(w);
        Ok(out)
    }
}

/// `ref_energy` must have exactly one finite entry per species of the basis.
fn check_references(basis: &Basis, ref_energy: &[f64]) -> Result<(), FitError> {
    for (s, &z) in basis.species.iter().enumerate() {
        match ref_energy.get(s) {
            Some(r) if r.is_finite() => {}
            _ => return Err(FitError::MissingReference(z)),
        }
    }
    if ref_energy.len() != basis.n_species() {
        return Err(FitError::ReferenceLength {
            expected: basis.n_species(),
            got: ref_energy.len(),
        });
    }
    Ok(())
}

/// The compensated totals of an accumulator's sums ([`Accumulator::sums`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Sums {
    pub gram_e: Vec<f64>,
    pub rhs_e: Vec<f64>,
    pub yy_e: f64,
    pub gram_f: Vec<f64>,
    pub rhs_f: Vec<f64>,
    pub yy_f: f64,
}

/// The Neumaier compensations of an accumulator (private; [`Accumulator::sums`]).
#[derive(Clone, Debug)]
struct Compensation {
    gram_e: Vec<f64>,
    rhs_e: Vec<f64>,
    yy_e: f64,
    gram_f: Vec<f64>,
    rhs_f: Vec<f64>,
    yy_f: f64,
}

/// s + x with Neumaier's compensation c (the total is s + c): the low-order part of each
/// addition that rounding drops is kept in c, so the error stays O(ε)·Σ|x| instead of growing
/// with the number of terms.
#[inline]
fn neumaier_add(s: &mut f64, c: &mut f64, x: f64) {
    let t = *s + x;
    if s.abs() >= x.abs() {
        *c += (*s - t) + x;
    } else {
        *c += (x - t) + *s;
    }
    *s = t;
}

/// G += w v vᵀ, b += w y v over the support of v (the species blocks a cluster touches), each
/// entry compensated (`neumaier_add`; `gc`, `bc` the compensations).
#[allow(clippy::too_many_arguments)]
fn rank1_update(
    gram: &mut [f64],
    gc: &mut [f64],
    rhs: &mut [f64],
    bc: &mut [f64],
    q: usize,
    v: &[f64],
    y: f64,
    w: f64,
    support: &mut Vec<usize>,
) {
    support.clear();
    support.extend((0..q).filter(|&i| v[i] != 0.0));
    for &i in support.iter() {
        let wi = w * v[i];
        neumaier_add(&mut rhs[i], &mut bc[i], wi * y);
        for &j in support.iter() {
            let k = i * q + j;
            neumaier_add(&mut gram[k], &mut gc[k], wi * v[j]);
        }
    }
}

/// G += w Σ_r v_r v_rᵀ, b += w Σ_r y_r v_r for the k rows v_r of `rows` (k×q row-major): the block
/// summed plainly over its k rows (k ≤ 3N: a few ε relative per entry), then added to each entry
/// with one compensated step (`neumaier_add`) — the large-basis path's form of [`rank1_update`],
/// threaded over the rows of G (each entry written by one thread; G stays exactly symmetric,
/// the block's (i, j) and (j, i) being the same commutative sums).
#[allow(clippy::too_many_arguments)]
fn batch_update(gram: &mut [f64], gc: &mut [f64], rhs: &mut [f64], bc: &mut [f64], q: usize, rows: &[f64], ys: &[f64], w: f64) {
    let k = ys.len();
    debug_assert_eq!(rows.len(), k * q);
    let support: Vec<usize> = (0..q).filter(|&i| (0..k).any(|r| rows[r * q + i] != 0.0)).collect();
    if support.is_empty() {
        return;
    }
    // column-major copy of the support: t[s] = (rows[r][support[s]])_r, contiguous in r
    let t: Vec<Vec<f64>> = support.iter().map(|&i| (0..k).map(|r| rows[r * q + i]).collect()).collect();
    for (s, &i) in support.iter().enumerate() {
        neumaier_add(&mut rhs[i], &mut bc[i], w * crate::linalg::dot(&t[s], ys));
    }
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(16).max(1);
    let per = support.len().div_ceil(threads);
    // each thread owns whole rows i of G: split the rows of the support among them
    let mut row_ids: Vec<Vec<usize>> = support.chunks(per).map(|c| c.to_vec()).collect();
    let mut work: Vec<Vec<(usize, &mut [f64], &mut [f64])>> = (0..row_ids.len()).map(|_| Vec::new()).collect();
    {
        let mut owner = vec![usize::MAX; q];
        for (c, ids) in row_ids.iter().enumerate() {
            for &i in ids {
                owner[i] = c;
            }
        }
        for (i, (g_row, c_row)) in gram.chunks_mut(q).zip(gc.chunks_mut(q)).enumerate() {
            if owner[i] != usize::MAX {
                work[owner[i]].push((i, g_row, c_row));
            }
        }
    }
    row_ids.clear();
    let index: Vec<usize> = {
        let mut ix = vec![usize::MAX; q];
        for (s, &i) in support.iter().enumerate() {
            ix[i] = s;
        }
        ix
    };
    let (t, support, index) = (&t, &support, &index);
    std::thread::scope(|scope| {
        for chunk in work {
            scope.spawn(move || {
                for (i, g_row, c_row) in chunk {
                    let ti = &t[index[i]];
                    for (sj, &j) in support.iter().enumerate() {
                        neumaier_add(&mut g_row[j], &mut c_row[j], w * crate::linalg::dot(ti, &t[sj]));
                    }
                }
            });
        }
    });
}

/// yᵀWy − 2 mᵀb + mᵀGm, clamped at 0 (cancellation).
fn residual_ss(gram: &[f64], rhs: &[f64], yy: f64, q: usize, m: &[f64]) -> f64 {
    let mut gm = vec![0.0; q];
    mat_vec(gram, q, m, &mut gm);
    let mgm: f64 = m.iter().zip(&gm).map(|(a, b)| a * b).sum();
    let mb: f64 = m.iter().zip(rhs).map(|(a, b)| a * b).sum();
    (yy - 2.0 * mb + mgm).max(0.0)
}

/// tr(M G) for row-major symmetric M, G.
fn trace_product(m: &[f64], g: &[f64], q: usize) -> f64 {
    let mut t = 0.0;
    for i in 0..q {
        for j in 0..q {
            t += m[i * q + j] * g[j * q + i];
        }
    }
    t
}

/// One posterior solve at (λ, s_E², s_F²): the precision A (with λ_eff on the diagonal), its
/// factor, the mean, per-kind RSS and γ, ln det A.
struct Solved {
    a: Vec<f64>,
    chol: Vec<f64>,
    m: Vec<f64>,
    rss_e: f64,
    rss_f: f64,
    gamma_e: f64,
    gamma_f: f64,
    logdet: f64,
    m_norm2: f64,
}

/// The chosen hyperparameters and how the search went.
struct Hyper {
    lambda: f64,
    s_e2: f64,
    s_f2: f64,
    iterations: usize,
    converged: bool,
}

/// The standardised two-noise least-squares problem on the Q′ columns (module doc).
struct TwoNoise {
    g_e: Vec<f64>,
    g_f: Vec<f64>,
    b_e: Vec<f64>,
    b_f: Vec<f64>,
    yy_e: f64,
    yy_f: f64,
    n_e: f64,
    n_f: f64,
    q: usize,
    sum_log_w: f64,
    floor_e2: f64,
    floor_f2: f64,
}

const LAMBDA_GRID_LO: f64 = 1e-8;
const LAMBDA_GRID_HI: f64 = 1e2;
const LAMBDA_EXTEND_LO: f64 = 1e-14;
const LAMBDA_EXTEND_HI: f64 = 1e8;
const OUTER_ROUNDS: usize = 30;
const OUTER_TOL: f64 = 1e-3;
const NOISE_STEPS: usize = 100;
const NOISE_TOL: f64 = 1e-6;

impl TwoNoise {
    fn solve(&self, lambda: f64, s_e2: f64, s_f2: f64) -> Result<Solved, FitError> {
        let q = self.q;
        let (ie, i_f) = (1.0 / s_e2, 1.0 / s_f2);
        let mut a = vec![0.0; q * q];
        let mut max_diag = 0.0f64;
        for i in 0..q {
            for j in 0..q {
                a[i * q + j] = self.g_e[i * q + j] * ie + self.g_f[i * q + j] * i_f;
            }
            max_diag = max_diag.max(a[i * q + i]);
        }
        let lambda_eff = lambda.max(4.0 * q.max(1) as f64 * f64::EPSILON * max_diag);
        for i in 0..q {
            a[i * q + i] += lambda_eff;
        }
        let mut chol = a.clone();
        cholesky_in_place(&mut chol, q).map_err(|_| FitError::Singular)?;
        let mut m: Vec<f64> = (0..q).map(|i| self.b_e[i] * ie + self.b_f[i] * i_f).collect();
        cholesky_solve(&chol, q, &mut m);
        let rss_e = residual_ss(&self.g_e, &self.b_e, self.yy_e, q, &m);
        let rss_f = residual_ss(&self.g_f, &self.b_f, self.yy_f, q, &m);
        let inv = cholesky_inverse(&chol, q);
        let gamma_e = trace_product(&inv, &self.g_e, q) * ie;
        let gamma_f = trace_product(&inv, &self.g_f, q) * i_f;
        let logdet = cholesky_logdet(&chol, q);
        let m_norm2 = m.iter().map(|v| v * v).sum();
        Ok(Solved {
            a,
            chol,
            m,
            rss_e,
            rss_f,
            gamma_e,
            gamma_f,
            logdet,
            m_norm2,
        })
    }

    /// Log marginal likelihood at (λ, s_E², s_F²) given the solve there (module doc).
    fn evidence(&self, lambda: f64, s_e2: f64, s_f2: f64, sol: &Solved) -> f64 {
        let two_pi = 2.0 * std::f64::consts::PI;
        -0.5 * self.n_e * (two_pi * s_e2).ln() - 0.5 * self.n_f * (two_pi * s_f2).ln()
            + 0.5 * self.q as f64 * lambda.ln()
            - 0.5 * sol.logdet
            - 0.5 * sol.rss_e / s_e2
            - 0.5 * sol.rss_f / s_f2
            - 0.5 * lambda * sol.m_norm2
            + 0.5 * self.sum_log_w
    }

    /// The noise fixed points s_κ² = RSS_κ/(n_κ − γ_κ) at fixed λ, iterated together; pinned
    /// levels stay put. Returns the converged levels and the solve at them.
    fn noise_fixed_point(
        &self,
        lambda: f64,
        mut s_e2: f64,
        mut s_f2: f64,
        pin_e: bool,
        pin_f: bool,
    ) -> Result<(f64, f64, Solved), FitError> {
        let mut sol = self.solve(lambda, s_e2, s_f2)?;
        if pin_e && pin_f {
            return Ok((s_e2, s_f2, sol));
        }
        // A level within two decades of its floor is pinned AT the floor: the RSS formed from
        // the statistics carries ~ε·yᵀWy of absolute error, so such a value is rounding noise
        // and iterating on it would never settle (noise-free fixtures).
        let floored = |rss: f64, dof: f64, floor: f64| {
            let v = rss / dof.max(1.0);
            if v < 100.0 * floor {
                floor
            } else {
                v
            }
        };
        for _ in 0..NOISE_STEPS {
            let next_e = if pin_e {
                s_e2
            } else {
                floored(sol.rss_e, self.n_e - sol.gamma_e, self.floor_e2)
            };
            let next_f = if pin_f {
                s_f2
            } else {
                floored(sol.rss_f, self.n_f - sol.gamma_f, self.floor_f2)
            };
            let converged = rel_change(next_e, s_e2) <= self.resolution_tol(NOISE_TOL, self.yy_e, sol.rss_e)
                && rel_change(next_f, s_f2) <= self.resolution_tol(NOISE_TOL, self.yy_f, sol.rss_f);
            s_e2 = next_e;
            s_f2 = next_f;
            sol = self.solve(lambda, s_e2, s_f2)?;
            if converged {
                break;
            }
        }
        Ok((s_e2, s_f2, sol))
    }

    /// A relative tolerance no tighter than the statistics resolve: RSS_κ carries ~ε·yᵀWy_κ of
    /// absolute error, so changes below 10ε·yᵀWy_κ/RSS_κ are rounding (matters only for
    /// noise-free data, where RSS is the model's representability error).
    fn resolution_tol(&self, tol: f64, yy: f64, rss: f64) -> f64 {
        tol.max(10.0 * f64::EPSILON * yy / rss.max(f64::MIN_POSITIVE))
    }

    /// λ maximising the evidence at fixed noise levels: half-decade grid 1e-8..1e2, extended
    /// while the maximum is on an edge (to 1e-14 / 1e8), then golden-section in ln λ between
    /// the grid neighbours of the best point.
    fn lambda_search(&self, s_e2: f64, s_f2: f64) -> Result<f64, FitError> {
        // A grid point whose factorisation fails scores −∞ instead of failing the search: at
        // the small-λ end the whitened Gram of a near-collinear basis (body-3 on a 3-atom
        // molecule: Q′ = 109 columns for 3 internal degrees of freedom) can have negative
        // rounding eigenvalues larger than the 4ε floor, which says nothing about the λ the
        // evidence will pick. Only a search with NO factorable point is `Singular`.
        let at = |lambda: f64| -> f64 {
            match self.solve(lambda, s_e2, s_f2) {
                Ok(sol) => self.evidence(lambda, s_e2, s_f2, &sol),
                Err(_) => f64::NEG_INFINITY,
            }
        };
        lambda_search_with(at)
    }

    /// [`estimate_free`](Self::estimate_free), then — when `FitOptions::sigma_f_min` is set and the estimated
    /// s_F fell below it — once more with s_F pinned at the floor.
    fn estimate(&self, opts: &FitOptions) -> Result<(Hyper, Solved), FitError> {
        assert!(opts.sigma_f_min >= 0.0 && opts.sigma_f_min.is_finite(), "FitOptions::sigma_f_min must be ≥ 0");
        let free = self.estimate_free(opts)?;
        if opts.sigma_f.is_none() && self.n_f > 0.0 && free.0.s_f2 < opts.sigma_f_min * opts.sigma_f_min {
            return self.estimate_free(&FitOptions { sigma_f: Some(opts.sigma_f_min), ..opts.clone() });
        }
        Ok(free)
    }

    /// Alternate λ search and noise fixed points until all three change by < 1e-3 relative
    /// (module doc). Pinned values are used as given.
    fn estimate_free(&self, opts: &FitOptions) -> Result<(Hyper, Solved), FitError> {
        let pin_e = opts.sigma_e.is_some();
        let pin_f = opts.sigma_f.is_some();
        for (name, v) in [("sigma_e", opts.sigma_e), ("sigma_f", opts.sigma_f)] {
            if let Some(v) = v {
                assert!(v > 0.0 && v.is_finite(), "FitOptions::{name} must be positive");
            }
        }
        if let Some(l) = opts.lambda {
            assert!(l >= 0.0 && l.is_finite(), "FitOptions::lambda must be ≥ 0");
        }
        let mut s_e2 = opts
            .sigma_e
            .map(|s| s * s)
            .unwrap_or(self.yy_e / self.n_e)
            .max(self.floor_e2);
        let mut s_f2 = opts
            .sigma_f
            .map(|s| s * s)
            .unwrap_or(if self.n_f > 0.0 { self.yy_f / self.n_f } else { 1.0 })
            .max(self.floor_f2);
        if let (Some(lambda), true, true) = (opts.lambda, pin_e, pin_f) {
            let sol = self.solve(lambda, s_e2, s_f2)?;
            return Ok((
                Hyper {
                    lambda,
                    s_e2,
                    s_f2,
                    iterations: 0,
                    converged: true,
                },
                sol,
            ));
        }
        let mut lambda_prev: Option<f64> = None;
        let mut lambda = 0.0;
        let mut last: Option<Solved> = None;
        let mut iterations = 0;
        let mut converged = false;
        for round in 0..OUTER_ROUNDS {
            iterations = round + 1;
            lambda = match opts.lambda {
                Some(l) => l,
                None => self.lambda_search(s_e2, s_f2)?,
            };
            let (next_e, next_f, sol) = self.noise_fixed_point(lambda, s_e2, s_f2, pin_e, pin_f)?;
            let noise_ok = rel_change(next_e, s_e2) <= self.resolution_tol(OUTER_TOL, self.yy_e, sol.rss_e)
                && rel_change(next_f, s_f2) <= self.resolution_tol(OUTER_TOL, self.yy_f, sol.rss_f);
            let lambda_change = match (opts.lambda, lambda_prev) {
                (Some(_), _) => 0.0,
                (None, Some(prev)) => rel_change(lambda, prev),
                (None, None) => f64::INFINITY,
            };
            s_e2 = next_e;
            s_f2 = next_f;
            last = Some(sol);
            lambda_prev = Some(lambda);
            if noise_ok && lambda_change <= OUTER_TOL {
                converged = true;
                break;
            }
        }
        Ok((
            Hyper {
                lambda,
                s_e2,
                s_f2,
                iterations,
                converged,
            },
            last.expect("at least one round"),
        ))
    }
}

/// What the large-basis solve adds to [`Solved`] for a warm state: λ_eff, tr Σ and tr(Σ G̃_E).
struct LargeExtra {
    lambda_eff: f64,
    tr_sigma: f64,
    tr_sigma_ge: f64,
    /// The spectral search stopped on a ρ stall (`RefitKind::Spectral::stalled`).
    stalled: bool,
}

/// The spectrum of H_ρ = G̃_F + ρ G̃_E for one noise ratio ρ = s_F²/s_E² (module doc, "The
/// large-basis path"): eigenvalues h_k and ĉ = Uᵀ(b̃_F + ρ b̃_E), with the floor μ of λ_eff·s_F².
struct Spectrum {
    rho: f64,
    h: Vec<f64>,
    c: Vec<f64>,
    mu_floor: f64,
}

impl Spectrum {
    /// (evidence, ρ·RSS_E + RSS_F, γ) at (λ, s_F² = t, s_E² = t/ρ); `None` when a shifted
    /// eigenvalue is not positive (the spectral twin of a failed factorisation).
    fn at(&self, tn: &TwoNoise, lambda: f64, t: f64) -> Option<(f64, f64, f64)> {
        let mu = (lambda * t).max(self.mu_floor);
        let (mut mc, mut hm2, mut mn2, mut logdet, mut gamma) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (&h, &c) in self.h.iter().zip(&self.c) {
            let den = h + mu;
            if !(den > 0.0) {
                return None;
            }
            let mk = c / den;
            mc += mk * c;
            hm2 += h * mk * mk;
            mn2 += mk * mk;
            logdet += (den / t).ln();
            gamma += h / den;
        }
        let comb = (self.rho * tn.yy_e + tn.yy_f - 2.0 * mc + hm2).max(0.0);
        let (s_e2, s_f2) = (t / self.rho, t);
        let two_pi = 2.0 * std::f64::consts::PI;
        let ev = -0.5 * tn.n_e * (two_pi * s_e2).ln() - 0.5 * tn.n_f * (two_pi * s_f2).ln() + 0.5 * tn.q as f64 * lambda.ln() - 0.5 * logdet - 0.5 * comb / t - 0.5 * lambda * mn2 + 0.5 * tn.sum_log_w;
        Some((ev, comb, gamma))
    }
}

impl TwoNoise {
    /// The spectrum at ratio ρ: one Householder tridiagonalisation of H_ρ and an implicit QL
    /// iteration that carries Qᵀc along (no eigenvectors are formed).
    fn spectrum(&self, rho: f64) -> Result<Spectrum, FitError> {
        let q = self.q;
        let mut h = vec![0.0; q * q];
        let mut max_diag = 0.0f64;
        for i in 0..q {
            for j in 0..q {
                h[i * q + j] = self.g_f[i * q + j] + rho * self.g_e[i * q + j];
            }
            max_diag = max_diag.max(h[i * q + i]);
        }
        let tri = Tridiagonal::of(&h, q);
        let mut c: Vec<f64> = (0..q).map(|i| self.b_f[i] + rho * self.b_e[i]).collect();
        tri.apply_qt(&mut c);
        let (mut d, mut e) = (tri.d.clone(), tri.e.clone());
        tridiagonal_ql(&mut d, &mut e, |i, cc, ss| {
            let (x, y) = (c[i], c[i + 1]);
            c[i] = cc * x - ss * y;
            c[i + 1] = ss * x + cc * y;
        })
        .map_err(|_| FitError::Singular)?;
        Ok(Spectrum { rho, h: d, c, mu_floor: 4.0 * q.max(1) as f64 * f64::EPSILON * max_diag })
    }

    /// [`TwoNoise::solve`] with the large-basis kernels (the vectorised Cholesky and the
    /// triangular inverse: ≈ Q′³ flops instead of ≈ 2.3 Q′³), plus tr Σ and tr(Σ G̃_E).
    fn solve_large(&self, lambda: f64, s_e2: f64, s_f2: f64) -> Result<(Solved, LargeExtra), FitError> {
        let q = self.q;
        let (ie, i_f) = (1.0 / s_e2, 1.0 / s_f2);
        let mut a = vec![0.0; q * q];
        let mut max_diag = 0.0f64;
        for i in 0..q {
            for j in 0..q {
                a[i * q + j] = self.g_e[i * q + j] * ie + self.g_f[i * q + j] * i_f;
            }
            max_diag = max_diag.max(a[i * q + i]);
        }
        let lambda_eff = lambda.max(4.0 * q.max(1) as f64 * f64::EPSILON * max_diag);
        for i in 0..q {
            a[i * q + i] += lambda_eff;
        }
        let mut chol = a.clone();
        cholesky_in_place_fast(&mut chol, q).map_err(|_| FitError::Singular)?;
        let mut m: Vec<f64> = (0..q).map(|i| self.b_e[i] * ie + self.b_f[i] * i_f).collect();
        cholesky_solve(&chol, q, &mut m);
        let rss_e = residual_ss(&self.g_e, &self.b_e, self.yy_e, q, &m);
        let rss_f = residual_ss(&self.g_f, &self.b_f, self.yy_f, q, &m);
        let inv = inverse_from_lower_inverse(&lower_inverse_transposed(&chol, q), q);
        let tr_sigma_ge = trace_product(&inv, &self.g_e, q);
        let gamma_e = tr_sigma_ge * ie;
        let gamma_f = trace_product(&inv, &self.g_f, q) * i_f;
        let tr_sigma = (0..q).map(|i| inv[i * q + i]).sum();
        let logdet = cholesky_logdet(&chol, q);
        let m_norm2 = m.iter().map(|v| v * v).sum();
        Ok((Solved { a, chol, m, rss_e, rss_f, gamma_e, gamma_f, logdet, m_norm2 }, LargeExtra { lambda_eff, tr_sigma, tr_sigma_ge, stalled: false }))
    }

    /// [`TwoNoise::estimate`] on the large-basis path: the same stationarity conditions and
    /// tolerances, the same `sigma_f_min` rule.
    fn estimate_spectral(&self, opts: &FitOptions, start: Option<(f64, f64, f64)>) -> Result<(Hyper, Solved, LargeExtra), FitError> {
        assert!(opts.sigma_f_min >= 0.0 && opts.sigma_f_min.is_finite(), "FitOptions::sigma_f_min must be ≥ 0");
        let free = self.estimate_spectral_free(opts, start)?;
        if opts.sigma_f.is_none() && self.n_f > 0.0 && free.0.s_f2 < opts.sigma_f_min * opts.sigma_f_min {
            return self.estimate_spectral_free(&FitOptions { sigma_f: Some(opts.sigma_f_min), ..opts.clone() }, start);
        }
        Ok(free)
    }

    /// The search over the noise ratio ρ (module doc, "The large-basis path"). The profile
    /// E(ρ) = max over λ and — with neither level pinned — the common scale t = s_F² of ln p, each
    /// from one spectrum of H_ρ (λ by the grid-and-golden search, t by MacKay's β update
    /// t = (ρ RSS_E + RSS_F)/(n − γ), alternated to 1e-6; with a pinned level t follows the pin).
    /// At the profile's point one exact solve gives RSS_κ and γ_κ per kind, and the per-kind fixed
    /// points s_κ² = RSS_κ/(n_κ − γ_κ) the next ρ — taken as a step in ln ρ whose multiplier halves
    /// each time the step reverses direction (the bare fixed point can cycle: ρ 460 → 2.1e4 → 1.4e3
    /// → 460 on a near-noise-free collection fixture). Converged when λ, s_E² and s_F² all move by
    /// less than 1e-3 (the original alternation's test); a multiplier below 2⁻¹² without that is a
    /// stall (`LargeExtra::stalled`, `converged` false). The returned solve is at the final levels.
    fn estimate_spectral_free(&self, opts: &FitOptions, start: Option<(f64, f64, f64)>) -> Result<(Hyper, Solved, LargeExtra), FitError> {
        let pin_e = opts.sigma_e.is_some();
        let pin_f = opts.sigma_f.is_some();
        for (name, v) in [("sigma_e", opts.sigma_e), ("sigma_f", opts.sigma_f)] {
            if let Some(v) = v {
                assert!(v > 0.0 && v.is_finite(), "FitOptions::{name} must be positive");
            }
        }
        if let Some(l) = opts.lambda {
            assert!(l >= 0.0 && l.is_finite(), "FitOptions::lambda must be ≥ 0");
        }
        let start = start.filter(|(l, e, f)| l.is_finite() && *l > 0.0 && e.is_finite() && *e > 0.0 && f.is_finite() && *f > 0.0);
        let s_e2_0 = opts
            .sigma_e
            .map(|s| s * s)
            .or(start.map(|s| s.1 * s.1))
            .unwrap_or(self.yy_e / self.n_e)
            .max(self.floor_e2);
        let s_f2_0 = opts
            .sigma_f
            .map(|s| s * s)
            .or(start.map(|s| s.2 * s.2))
            .unwrap_or(if self.n_f > 0.0 { self.yy_f / self.n_f } else { 1.0 })
            .max(self.floor_f2);
        if let (Some(lambda), true, true) = (opts.lambda, pin_e, pin_f) {
            let (sol, extra) = self.solve_large(lambda, s_e2_0, s_f2_0)?;
            return Ok((Hyper { lambda, s_e2: s_e2_0, s_f2: s_f2_0, iterations: 0, converged: true }, sol, extra));
        }
        let floored = |rss: f64, dof: f64, floor: f64| {
            let v = rss / dof.max(1.0);
            if v < 100.0 * floor {
                floor
            } else {
                v
            }
        };
        let n_all = self.n_e + self.n_f;
        // the profile at ρ: (λ, t, ln p); `t0` starts the β update
        let profile = |rho: f64, t0: f64| -> Result<(f64, f64, f64), FitError> {
            let spec = self.spectrum(rho)?;
            let t_floor = self.floor_f2.max(rho * self.floor_e2);
            let mut t = match (pin_e, pin_f) {
                (_, true) => s_f2_0,
                (true, false) => (rho * s_e2_0).max(self.floor_f2),
                (false, false) => t0.max(t_floor),
            };
            let mut lambda = opts.lambda.unwrap_or(1.0);
            for _ in 0..NOISE_STEPS {
                lambda = match opts.lambda {
                    Some(l) => l,
                    None => lambda_search_with(|l| spec.at(self, l, t).map_or(f64::NEG_INFINITY, |x| x.0))?,
                };
                if pin_e || pin_f {
                    break;
                }
                let (_, comb, gamma) = spec.at(self, lambda, t).ok_or(FitError::Singular)?;
                let t_new = floored(comb, n_all - gamma, t_floor);
                let done = rel_change(t_new, t) <= self.resolution_tol(NOISE_TOL, rho * self.yy_e + self.yy_f, comb);
                t = t_new;
                if done {
                    break;
                }
            }
            let ev = spec.at(self, lambda, t).ok_or(FitError::Singular)?.0;
            Ok((lambda, t, ev))
        };
        let mut rho = s_f2_0 / s_e2_0;
        let (mut lambda, mut t, _) = profile(rho, s_f2_0)?;
        let levels = |rho: f64, t: f64| -> (f64, f64) {
            match (pin_e, pin_f) {
                (true, _) => (s_e2_0, t),
                (false, _) => ((t / rho).max(self.floor_e2), t),
            }
        };
        let mut lambda_prev: Option<f64> = None;
        let mut iterations = 0;
        let mut converged = false;
        let mut stalled = false;
        let (mut alpha, mut prev_step) = (1.0f64, None);
        for round in 0..OUTER_ROUNDS {
            iterations = round + 1;
            let (s_e2, s_f2) = levels(rho, t);
            let (sol, _) = self.solve_large(lambda, s_e2, s_f2)?;
            let next_e = if pin_e { s_e2 } else { floored(sol.rss_e, self.n_e - sol.gamma_e, self.floor_e2) };
            let next_f = if pin_f { s_f2 } else { floored(sol.rss_f, self.n_f - sol.gamma_f, self.floor_f2) };
            let noise_ok = rel_change(next_e, s_e2) <= self.resolution_tol(OUTER_TOL, self.yy_e, sol.rss_e)
                && rel_change(next_f, s_f2) <= self.resolution_tol(OUTER_TOL, self.yy_f, sol.rss_f);
            let lambda_change = match (opts.lambda, lambda_prev) {
                (Some(_), _) => 0.0,
                (None, Some(prev)) => rel_change(lambda, prev),
                (None, None) => f64::INFINITY,
            };
            lambda_prev = Some(lambda);
            if noise_ok && lambda_change <= OUTER_TOL {
                converged = true;
                break;
            }
            if noise_ok && (pin_e && pin_f) {
                converged = true;
                break;
            }
            // the step in ln ρ the per-kind fixed points ask for, damped: the multiplier halves
            // whenever the step reverses the previous one's direction (the bare fixed point can
            // cycle). Not an evidence comparison: the profile's ln p carries ~5e-6 relative of
            // rounding between two spectra (the near-null eigenvalues' ε‖H‖ against the ridge
            // floor), more than a converging step changes it.
            let step = (next_f / next_e).ln() - rho.ln();
            if prev_step.is_some_and(|p: f64| p * step < 0.0) {
                alpha *= 0.5;
            }
            if alpha < 1.0 / 4096.0 {
                // damped to nothing without meeting the test: stopped, reported as a stall and
                // not as converged (review M5)
                stalled = true;
                break;
            }
            prev_step = Some(step);
            rho *= (alpha * step).exp();
            let t_try = if pin_e || pin_f { t } else { next_f };
            (lambda, t, _) = profile(rho, t_try)?;
        }
        let (s_e2, s_f2) = levels(rho, t);
        let (sol, mut extra) = self.solve_large(lambda, s_e2, s_f2)?;
        extra.stalled = stalled;
        Ok((Hyper { lambda, s_e2, s_f2, iterations, converged }, sol, extra))
    }
}

/// The λ search of [`TwoNoise::lambda_search`] over an evidence `at(λ)` (−∞ where a solve fails).
fn lambda_search_with(at: impl Fn(f64) -> f64) -> Result<f64, FitError> {
    let half_decade = 10f64.sqrt();
    let mut grid: Vec<(f64, f64)> = Vec::new(); // (ln λ, evidence)
    let mut lambda = LAMBDA_GRID_LO;
    while lambda <= LAMBDA_GRID_HI * 1.0001 {
        grid.push((lambda.ln(), at(lambda)));
        lambda *= half_decade;
    }
    if grid.iter().all(|g| g.1 == f64::NEG_INFINITY) {
        return Err(FitError::Singular);
    }
    let best_index = |g: &[(f64, f64)]| {
        g.iter()
            .enumerate()
            .max_by(|a, b| a.1 .1.partial_cmp(&b.1 .1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap()
    };
    loop {
        let i = best_index(&grid);
        if i == grid.len() - 1 && grid[i].0.exp() < LAMBDA_EXTEND_HI {
            let l = grid[i].0.exp() * half_decade;
            grid.push((l.ln(), at(l)));
        } else if i == 0 && grid[0].0.exp() > LAMBDA_EXTEND_LO {
            let l = grid[0].0.exp() / half_decade;
            grid.insert(0, (l.ln(), at(l)));
        } else {
            break;
        }
    }
    let i = best_index(&grid);
    let (mut lo, mut hi) = (
        grid[i.saturating_sub(1)].0,
        grid[(i + 1).min(grid.len() - 1)].0,
    );
    let mut best = grid[i];
    if hi > lo {
        let phi = 0.5 * (3.0 - 5f64.sqrt());
        let mut x1 = lo + phi * (hi - lo);
        let mut x2 = hi - phi * (hi - lo);
        let mut f1 = at(x1.exp());
        let mut f2 = at(x2.exp());
        // 20 steps resolve ln λ to 0.618²⁰·2.3 ≈ 1.5e-4, below the outer tolerance (the
        // alternation cannot converge if the λ search is coarser than its own test).
        for _ in 0..20 {
            if f1 > f2 {
                hi = x2;
                x2 = x1;
                f2 = f1;
                x1 = lo + phi * (hi - lo);
                f1 = at(x1.exp());
            } else {
                lo = x1;
                x1 = x2;
                f1 = f2;
                x2 = hi - phi * (hi - lo);
                f2 = at(x2.exp());
            }
        }
        for (x, f) in [(x1, f1), (x2, f2)] {
            if f > best.1 {
                best = (x, f);
            }
        }
    }
    Ok(best.0.exp())
}

fn rel_change(a: f64, b: f64) -> f64 {
    (a - b).abs() / a.abs().max(b.abs()).max(f64::MIN_POSITIVE)
}

/// κ₂(A) by power iteration (largest eigenvalue) and inverse iteration through the Cholesky
/// factor (smallest), 40 steps each from a fixed start vector.
fn condition_estimate(a: &[f64], chol: &[f64], q: usize) -> f64 {
    if q == 0 {
        return 1.0;
    }
    let normalise = |v: &mut [f64]| -> f64 {
        let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n > 0.0 {
            v.iter_mut().for_each(|x| *x /= n);
        }
        n
    };
    let mut v: Vec<f64> = (0..q).map(|i| 1.0 + 0.01 * i as f64).collect();
    normalise(&mut v);
    let mut w = vec![0.0; q];
    let mut largest = 0.0;
    for _ in 0..40 {
        mat_vec(a, q, &v, &mut w);
        largest = normalise(&mut w);
        std::mem::swap(&mut v, &mut w);
    }
    let mut u: Vec<f64> = (0..q).map(|i| 1.0 - 0.01 * i as f64).collect();
    normalise(&mut u);
    let mut inv_largest = 0.0;
    for _ in 0..40 {
        cholesky_solve(chol, q, &mut u);
        inv_largest = normalise(&mut u);
    }
    if inv_largest > 0.0 {
        largest * inv_largest
    } else {
        f64::INFINITY
    }
}

/// Fit E_KS − Σ_i ref_energy[s_i] and the forces jointly. `ref_energy[s]` (Hartree, indexed like
/// `basis.species`) must be finite for every species. Equivalent to filling an [`Accumulator`]
/// and calling `solve`, with the report's RMSEs taken from the exact residuals.
pub fn fit(
    store: &SampleStore,
    basis: &Basis,
    ref_energy: &[f64],
    opts: &FitOptions,
) -> Result<(Model, FitReport), FitError> {
    if store.is_empty() {
        return Err(FitError::NoSamples);
    }
    check_references(basis, ref_energy)?;
    let mut acc = Accumulator::new(basis);
    for s in &store.samples {
        acc.add(s, basis, ref_energy, opts)?;
    }
    let (model, mut report) = acc.solve(opts, &store.provenance, ref_energy)?;

    // Exact residuals through the hot path (the statistics cancel below ~1e-8·rms(y)).
    let mut ss_e = 0.0;
    let mut ss_f = 0.0;
    let mut n_f = 0usize;
    for s in &store.samples {
        let (e, f) = energy_and_forces(&model, &s.pos, &s.z)?;
        let scale = if opts.energy_per_atom { 1.0 / s.pos.len() as f64 } else { 1.0 };
        ss_e += s.weight * ((e - s.energy) * scale).powi(2);
        for (fj, tj) in f.iter().zip(&s.forces) {
            for a in 0..3 {
                ss_f += s.weight * (fj[a] - tj[a]).powi(2);
                n_f += 1;
            }
        }
    }
    report.rmse_energy_per_atom = (ss_e / store.len() as f64).sqrt();
    report.rmse_force = if n_f > 0 { (ss_f / n_f as f64).sqrt() } else { 0.0 };
    Ok((model, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    const H: u8 = 1;
    const O: u8 = 8;

    fn provenance() -> Provenance {
        Provenance {
            grid_n: 49,
            h: 0.25,
            sigma: 0.25,
            tol: 1e-6,
            charge: 0,
            multiplicity: None,
            teacher: "synthetic".to_string(),
            masses: vec![(1, 1), (8, 16)],
            ionic: "none".into(),
        }
    }

    /// Standard normal by Box–Muller (rand 0.8 has no Normal without rand_distr).
    fn normal(rng: &mut StdRng) -> f64 {
        let u1: f64 = rng.gen_range(f64::EPSILON..1.0);
        let u2: f64 = rng.gen_range(0.0..1.0);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn unit_vector(rng: &mut StdRng) -> [f64; 3] {
        loop {
            let v = [normal(rng), normal(rng), normal(rng)];
            let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if n > 1e-3 {
                return [v[0] / n, v[1] / n, v[2] / n];
            }
        }
    }

    const E_ATOM_H: f64 = -0.478;
    const MORSE_D: f64 = 0.17;
    const MORSE_A: f64 = 1.0;
    const MORSE_RE: f64 = 1.4;

    /// Morse dimer: total energy 2·E_atom + D[(1 − e^{−a(r−r_e)})² − 1] and dE/dr.
    fn morse(r: f64) -> (f64, f64) {
        let x = (-MORSE_A * (r - MORSE_RE)).exp();
        let e = 2.0 * E_ATOM_H + MORSE_D * ((1.0 - x) * (1.0 - x) - 1.0);
        let de = 2.0 * MORSE_D * MORSE_A * (1.0 - x) * x;
        (e, de)
    }

    /// An H₂ sample at separation r along `dir`, with optional Gaussian noise (energy, force
    /// component).
    fn dimer_sample(
        r: f64,
        dir: [f64; 3],
        origin: [f64; 3],
        noise: Option<(f64, f64, &mut StdRng)>,
    ) -> Sample {
        let (mut e, de) = morse(r);
        let p1 = [origin[0] + r * dir[0], origin[1] + r * dir[1], origin[2] + r * dir[2]];
        // F_2 = −dE/dr·u, F_1 = +dE/dr·u with u from atom 1 to atom 2.
        let mut forces = vec![
            [de * dir[0], de * dir[1], de * dir[2]],
            [-de * dir[0], -de * dir[1], -de * dir[2]],
        ];
        if let Some((se, sf, rng)) = noise {
            e += se * normal(rng);
            for f in &mut forces {
                for a in 0..3 {
                    f[a] += sf * normal(rng);
                }
            }
        }
        Sample {
            pos: vec![origin, p1],
            z: vec![H, H],
            energy: e,
            forces,
            tag: SampleTag::Scan,
            weight: 1.0,
        }
    }

    fn morse_store(
        rng: &mut StdRng,
        r_lo: f64,
        r_hi: f64,
        n: usize,
        noise: Option<(f64, f64)>,
    ) -> SampleStore {
        let mut store = SampleStore::new(provenance());
        for i in 0..n {
            let r = r_lo + (r_hi - r_lo) * i as f64 / (n - 1) as f64;
            let dir = unit_vector(rng);
            let origin = [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)];
            let s = match noise {
                Some((se, sf)) => dimer_sample(r, dir, origin, Some((se, sf, rng))),
                None => dimer_sample(r, dir, origin, None),
            };
            store.push(s).unwrap();
        }
        store
    }

    fn h2_basis(n_max: usize) -> Basis {
        let mut b = Basis::new(&[H]);
        b.body3 = false; // body-2 regression fixture (the default is body-3)
        b.n_max = n_max;
        b
    }

    /// Random H/O cluster of `n` atoms in a cube of side `side`, no pair closer than 0.8 Bohr.
    fn random_cluster(rng: &mut StdRng, n: usize, side: f64) -> (Vec<[f64; 3]>, Vec<u8>) {
        let z: Vec<u8> = (0..n).map(|i| if i % 2 == 0 { H } else { O }).collect();
        loop {
            let pos: Vec<[f64; 3]> = (0..n)
                .map(|_| [rng.gen_range(0.0..side), rng.gen_range(0.0..side), rng.gen_range(0.0..side)])
                .collect();
            let ok = (0..n).all(|i| {
                (i + 1..n).all(|j| {
                    let d: f64 = (0..3).map(|a| (pos[i][a] - pos[j][a]).powi(2)).sum::<f64>().sqrt();
                    d > 0.8
                })
            });
            if ok {
                return (pos, z);
            }
        }
    }

    /// A random linear teacher on the given basis (raw coefficients in [−1, 1], mean 0, scale
    /// 1, so c̃ = coeffs), used to generate exactly representable data.
    fn random_teacher(rng: &mut StdRng, basis: &Basis) -> Model {
        let p = basis.n_features();
        let s = basis.n_species();
        Model {
            basis: basis.clone(),
            coeffs: (0..s).map(|_| (0..p).map(|_| rng.gen_range(-1.0..1.0)).collect()).collect(),
            mean: vec![0.0; p],
            scale: vec![1.0; p],
            ref_energy: vec![-0.478, -74.8],
            sigma_e: 0.0,
            sigma_f: 0.0,
            energy_per_atom: true,
            lambda: 0.0,
            sigma_inv_chol: None,
            n_samples: 0,
            provenance: provenance(),
            representatives: OnceLock::new(),
        }
    }

    fn teacher_sample(
        teacher: &Model,
        pos: Vec<[f64; 3]>,
        z: Vec<u8>,
        noise: Option<(f64, f64, &mut StdRng)>,
    ) -> Sample {
        let (mut e, mut f) = energy_and_forces(teacher, &pos, &z).unwrap();
        if let Some((se, sf, rng)) = noise {
            e += se * normal(rng);
            for fj in &mut f {
                for a in 0..3 {
                    fj[a] += sf * normal(rng);
                }
            }
        }
        Sample {
            pos,
            z,
            energy: e,
            forces: f,
            tag: SampleTag::Perturbation,
            weight: 1.0,
        }
    }

    fn total_energy_rmse(model: &Model, store: &SampleStore) -> f64 {
        let mut ss = 0.0;
        for s in &store.samples {
            let (e, _) = energy_and_forces(model, &s.pos, &s.z).unwrap();
            ss += (e - s.energy).powi(2);
        }
        (ss / store.len() as f64).sqrt()
    }

    /// Held-out calibration statistics of a model on `samples`: (2σ coverage of the total
    /// energy, mean z²_E, 2σ coverage of ‖δF_j‖ against force_std, mean z²_F, held-out rmse E).
    struct Calibration {
        cov_e: f64,
        z2_e: f64,
        cov_f: f64,
        z2_f: f64,
        rmse_e: f64,
    }

    fn calibration(model: &Model, samples: &[Sample]) -> Calibration {
        let (mut hits_e, mut z2_e, mut ss_e) = (0usize, 0.0, 0.0);
        let (mut hits_f, mut z2_f, mut n_f) = (0usize, 0.0, 0usize);
        for s in samples {
            let pred = model.predict(&s.pos, &s.z, true).unwrap();
            let std_e = pred.energy_std.unwrap();
            let resid = pred.energy - s.energy;
            ss_e += resid * resid;
            z2_e += (resid / std_e).powi(2);
            if resid.abs() <= 2.0 * std_e {
                hits_e += 1;
            }
            let std_f = pred.force_std.unwrap();
            for j in 0..s.pos.len() {
                let d2: f64 = (0..3).map(|a| (pred.forces[j][a] - s.forces[j][a]).powi(2)).sum();
                z2_f += d2 / (std_f[j] * std_f[j]);
                n_f += 1;
                if d2.sqrt() <= 2.0 * std_f[j] {
                    hits_f += 1;
                }
            }
        }
        let n = samples.len() as f64;
        Calibration {
            cov_e: hits_e as f64 / n,
            z2_e: z2_e / n,
            cov_f: hits_f as f64 / n_f as f64,
            z2_f: z2_f / n_f as f64,
            rmse_e: (ss_e / n).sqrt(),
        }
    }

    #[test]
    fn fit_recovers_a_synthetic_linear_model_exactly() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut basis = Basis::new(&[H, O]);
        basis.body3 = false; // body-2 regression fixture (the default is body-3)
        basis.n_max = 4;
        let teacher = random_teacher(&mut rng, &basis);
        let mut store = SampleStore::new(provenance());
        for i in 0..120 {
            let n = 3 + i % 3;
            let (pos, z) = random_cluster(&mut rng, n, 5.0);
            store.push(teacher_sample(&teacher, pos, z, None)).unwrap();
        }
        let (model, report) = fit(&store, &basis, &teacher.ref_energy, &FitOptions::default()).unwrap();
        let (ct, off) = model.effective_coefficients();
        // Identifiable quantities: the homo coefficients, and for the hetero pair the SUM
        // c̃_{H,(O,n)} + c̃_{O,(H,n)} (the two columns are the same function of the positions;
        // the fit splits the sum equally — pinned below). Raw c̃ = coeffs/scale.
        let n_max = basis.n_max;
        let (t_h, t_o) = (0usize, 1usize);
        let mut worst = 0.0f64;
        assert_eq!(off, vec![0.0, 0.0], "no offset: mean is zero");
        for nn in 0..n_max {
            let (hh, oo) = (basis.body2_index(t_h, nn), basis.body2_index(t_o, nn));
            worst = worst.max((ct[t_h][hh] - teacher.coeffs[t_h][hh]).abs());
            worst = worst.max((ct[t_o][oo] - teacher.coeffs[t_o][oo]).abs());
            let sum_fit = ct[t_h][oo] + ct[t_o][hh];
            let sum_true = teacher.coeffs[t_h][oo] + teacher.coeffs[t_o][hh];
            worst = worst.max((sum_fit - sum_true).abs());
            assert!(
                (ct[t_h][oo] - ct[t_o][hh]).abs() < 1e-12,
                "equal split of the hetero sum at n = {nn}"
            );
        }
        assert!(worst < 1e-10, "max identifiable-coefficient error {worst:e}");
        assert!(model.mean.iter().all(|m| *m == 0.0));
        assert!(
            report.rmse_energy_per_atom < 1e-12 && report.rmse_force < 1e-12,
            "rmse E/atom {:e}, F {:e}",
            report.rmse_energy_per_atom,
            report.rmse_force
        );
        assert!(report.sigma_e < 1e-7 && report.sigma_f < 1e-7, "noise-free data: s_E {:e}, s_F {:e}", report.sigma_e, report.sigma_f);
        let s0 = &store.samples[0];
        let pred = model.predict(&s0.pos, &s0.z, true).unwrap();
        let e_std = pred.energy_std.expect("variance requested");
        let f_std = pred.force_std.expect("variance requested");
        assert!(e_std < 1e-6, "energy std {e_std:e} should be ≈ 0");
        assert!(f_std.iter().all(|s| *s < 1e-6), "force std {f_std:?} should be ≈ 0");
        assert!((pred.energy - s0.energy).abs() < 1e-11);
        assert_eq!(report.n_rows, store.samples.iter().map(|s| 1 + 3 * s.pos.len()).sum::<usize>());
        assert_eq!(report.q_prime, 12, "Q′ = n_max·S(S+1)/2 independent columns");
        assert_eq!(report.n_samples, 120);
        assert!(report.condition_estimate.is_finite() && report.condition_estimate >= 1.0);
        assert!(report.converged, "hyperparameter search converged in {} rounds", report.hyper_iterations);
    }

    /// M7 Task 9 (task-8b review ruling (b)): `sigma_f_min` is a floor on the force noise, not a pin. Noise-free
    /// data would estimate s_F ≈ 1e-8; with the floor the report says exactly the floor, the model's force
    /// standard deviation is at least √3 times it, and a floor below the estimate changes nothing (bitwise).
    #[test]
    fn sigma_f_min_floors_the_force_noise_and_is_inert_below_the_estimate() {
        let mut rng = StdRng::seed_from_u64(2);
        let store = morse_store(&mut rng, 1.0, 3.0, 60, None);
        let basis = h2_basis(12);
        let (free_model, free) = fit(&store, &basis, &[E_ATOM_H], &FitOptions::default()).unwrap();
        assert!(free.sigma_f < 1e-6, "noise-free: s_F {:e}", free.sigma_f);
        let floor = 5e-5;
        let (model, report) = fit(&store, &basis, &[E_ATOM_H], &FitOptions { sigma_f_min: floor, ..FitOptions::default() }).unwrap();
        assert_eq!(report.sigma_f, floor);
        assert_eq!(model.sigma_f, floor);
        let s0 = &store.samples[0];
        let f_std = model.predict(&s0.pos, &s0.z, true).unwrap().force_std.expect("variance requested");
        assert!(f_std.iter().all(|s| *s >= 3f64.sqrt() * floor * 0.999), "force std {f_std:?} below √3·floor");
        let (_, inert) = fit(&store, &basis, &[E_ATOM_H], &FitOptions { sigma_f_min: free.sigma_f / 10.0, ..FitOptions::default() }).unwrap();
        assert_eq!(inert, free, "a floor below the estimate changes nothing");
        let _ = free_model;
    }

    #[test]
    fn fit_recovers_a_morse_dimer_to_1e6_with_body2() {
        let mut rng = StdRng::seed_from_u64(2);
        let store = morse_store(&mut rng, 1.0, 3.0, 60, None);
        let basis = h2_basis(12);
        let (model, report) = fit(&store, &basis, &[E_ATOM_H], &FitOptions::default()).unwrap();
        let rmse_e = total_energy_rmse(&model, &store);
        eprintln!(
            "morse: rmse E_total {rmse_e:.3e} Ha, E/atom {:.3e}, F {:.3e} Ha/Bohr, s_E {:.3e}, s_F {:.3e}, λ {:.3e}, γ_E {:.2} γ_F {:.2}, κ {:.3e}, rounds {} conv {}",
            report.rmse_energy_per_atom,
            report.rmse_force,
            report.sigma_e,
            report.sigma_f,
            report.lambda,
            report.gamma_e,
            report.gamma_f,
            report.condition_estimate,
            report.hyper_iterations,
            report.converged
        );
        assert!(rmse_e < 1e-6, "rmse E = {rmse_e:e}");
        assert!(report.rmse_force < 1e-5, "rmse F = {:e}", report.rmse_force);
        // Beyond r_cut the energy is exactly the two reference atoms and the forces vanish.
        for r in [6.0, 6.5, 10.0] {
            let pos = [[0.1, -0.2, 0.3], [0.1 + r, -0.2, 0.3]];
            let pred = model.predict(&pos, &[H, H], true).unwrap();
            assert_eq!(pred.energy, 2.0 * E_ATOM_H, "r = {r}");
            assert_eq!(pred.forces, vec![[0.0; 3]; 2]);
            assert!(pred.energy_std.is_some() && pred.force_std.is_some());
        }
        // Inside the window the model reproduces the Morse forces by the hot path too.
        let pos = [[0.0; 3], [1.7, 0.0, 0.0]];
        let pred = model.predict(&pos, &[H, H], false).unwrap();
        let (_, de) = morse(1.7);
        assert!((pred.forces[1][0] + de).abs() < 1e-5);
        assert!(pred.energy_std.is_none(), "not requested");
    }

    #[test]
    fn noise_level_comes_from_residuals_not_from_a_constant() {
        let mut rng = StdRng::seed_from_u64(3);
        let sigma_f = 3e-4;
        let store = morse_store(&mut rng, 1.0, 3.0, 60, Some((0.0, sigma_f)));
        let basis = h2_basis(12);
        let (model, report) = fit(&store, &basis, &[E_ATOM_H], &FitOptions::default()).unwrap();
        eprintln!(
            "noise: s_E {:.3e} s_F {:.3e} λ {:.3e} γ_E {:.2} γ_F {:.2} dof_E {:.1} dof_F {:.1} rmse_E/atom {:.3e} rmse_F {:.3e} rounds {} conv {}",
            report.sigma_e,
            report.sigma_f,
            report.lambda,
            report.gamma_e,
            report.gamma_f,
            report.residual_dof_e,
            report.residual_dof_f,
            report.rmse_energy_per_atom,
            report.rmse_force,
            report.hyper_iterations,
            report.converged
        );
        assert!((report.sigma_f - sigma_f).abs() < 0.2 * sigma_f, "s_F = {:e} vs injected {sigma_f:e}", report.sigma_f);
        // Energies were noise-free: with their own noise level they are fitted to the Morse
        // representability (~1e-9), not dragged by the force noise.
        assert!(report.sigma_e < 1e-2 * sigma_f, "s_E = {:e}", report.sigma_e);
        assert!(report.rmse_energy_per_atom < 1e-7, "rmse E/atom {:e}", report.rmse_energy_per_atom);
        let s0 = &store.samples[10];
        let pred = model.predict(&s0.pos, &s0.z, true).unwrap();
        let f_std = pred.force_std.unwrap();
        for (j, s) in f_std.iter().enumerate() {
            assert!(*s >= 0.8 * sigma_f, "force std of atom {j} = {s:e} < 0.8·σ_F");
            assert!(*s < 3.0 * sigma_f, "force std of atom {j} = {s:e} is far above the noise");
        }
        // M6 Task 11: the split into label noise and coefficient uncertainty, checked against an
        // independent recomputation of the posterior predictive (physics review, round 3, M2):
        // Σ = (L Lᵀ)⁻¹ formed explicitly (`cholesky_inverse`) and the force-row features
        // g̃_{j,a} = −∂/∂r_{j,a} Σ_i B̃_i by central differences of `describe_values` (h = 1e-5
        // Bohr; truncation O(h²)), epistemic_j² = Σ_a g̃ᵀΣg̃. The explicit inverse is good to
        // Q′·ε·κ relative, so the check runs on a fit pinned to a well-conditioned posterior
        // (σ_e = σ_f = 1e-2, λ = 1e3: κ 1.8e5, so Q′εκ ≈ 3e-10; measured 2e-10) — what is checked is the formula (the rows,
        // the standardisation, the 3σ_f² term), not the conditioning of a particular fit: the
        // model's `force_std_epistemic` (triangular solves on analytic gradients) must agree to
        // 1e-7 relative, and its `force_std`² must be 3σ_f² + that independent epistemic².
        let f_epi = pred.force_std_epistemic.expect("variance requested");
        for (j, e) in f_epi.iter().enumerate() {
            assert!(*e < 0.5 * model.sigma_f, "atom {j}: epistemic {e:e} vs σ_f {:e}: the noise dominates in-sample", model.sigma_f);
        }
        {
            let opts_c = FitOptions { sigma_e: Some(1e-2), sigma_f: Some(1e-2), lambda: Some(1e3), ..Default::default() };
            let (mc, rc) = fit(&store, &basis, &[E_ATOM_H], &opts_c).unwrap();
            let pc = mc.predict(&s0.pos, &s0.z, true).unwrap();
            let q = mc.basis.n_independent_columns();
            let sigma = crate::linalg::cholesky_inverse(mc.sigma_inv_chol.as_ref().unwrap(), q);
            let inv_scale: Vec<f64> = mc.scale.iter().map(|&s| if s == 0.0 { 1.0 } else { 1.0 / s }).collect();
            let feat = |pos: &[[f64; 3]]| -> Vec<f64> {
                let d = crate::descriptor::describe_values(&mc.basis, pos, &s0.z).unwrap();
                (0..q).map(|k| d.iter().map(|b| b[k]).sum::<f64>() * inv_scale[k]).collect()
            };
            let h = 1e-5;
            eprintln!("independent predictive check: pinned fit κ {:.2e}", rc.condition_estimate);
            assert!(rc.condition_estimate < 1e6, "the check needs a well-conditioned posterior: κ {:.1e}", rc.condition_estimate);
            for (j, (s, e)) in pc.force_std.as_ref().unwrap().iter().zip(pc.force_std_epistemic.as_ref().unwrap()).enumerate() {
                let mut epi2 = 0.0;
                for a in 0..3 {
                    let (mut p, mut m) = (s0.pos.clone(), s0.pos.clone());
                    p[j][a] += h;
                    m[j][a] -= h;
                    let (fp, fm) = (feat(&p), feat(&m));
                    let g: Vec<f64> = (0..q).map(|k| -(fp[k] - fm[k]) / (2.0 * h)).collect();
                    epi2 += (0..q).map(|r| (0..q).map(|c| g[r] * sigma[r * q + c] * g[c]).sum::<f64>()).sum::<f64>();
                }
                let independent = epi2.sqrt();
                eprintln!("atom {j}: epistemic {e:.9e} vs independent {independent:.9e} (relative {:.1e})", (e - independent).abs() / independent);
                assert!((e - independent).abs() <= 1e-7 * independent, "atom {j}: epistemic {e:e} vs independent {independent:e}");
                let split = 3.0 * mc.sigma_f * mc.sigma_f + epi2;
                assert!((s * s - split).abs() <= 1e-7 * s * s, "atom {j}: force_std² {:e} vs 3σ_f² + independent epistemic² {split:e}", s * s);
            }
        }
        // Pinning overrides the estimates; pinning everything makes a plain whitened ridge solve.
        let pinned = FitOptions { sigma_f: Some(1e-2), ..Default::default() };
        let (m2, r2) = fit(&store, &basis, &[E_ATOM_H], &pinned).unwrap();
        assert_eq!(r2.sigma_f, 1e-2);
        assert_eq!(m2.sigma_f, 1e-2);
        assert!(r2.sigma_e < 1e-5, "s_E still estimated: {:e}", r2.sigma_e);
        let all = FitOptions { sigma_e: Some(1e-5), sigma_f: Some(1e-2), lambda: Some(1e-3), ..Default::default() };
        let (_, r3) = fit(&store, &basis, &[E_ATOM_H], &all).unwrap();
        assert_eq!((r3.sigma_e, r3.sigma_f, r3.lambda), (1e-5, 1e-2, 1e-3));
        assert_eq!(r3.hyper_iterations, 0);
    }

    #[test]
    fn predictive_variance_is_calibrated_on_held_out_samples() {
        let mut rng = StdRng::seed_from_u64(4);
        let mut basis = Basis::new(&[H, O]);
        basis.body3 = false; // body-2 regression fixture (the default is body-3)
        basis.n_max = 4;
        let teacher = random_teacher(&mut rng, &basis);
        let (se, sf) = (1e-3, 3e-4);
        let samples: Vec<Sample> = (0..200)
            .map(|_| {
                let (pos, z) = random_cluster(&mut rng, 3, 4.0);
                teacher_sample(&teacher, pos, z, Some((se, sf, &mut rng)))
            })
            .collect();
        let mut train = SampleStore::new(provenance());
        for s in &samples[..150] {
            train.push(s.clone()).unwrap();
        }
        let (model, report) = fit(&train, &basis, &teacher.ref_energy, &FitOptions::default()).unwrap();
        let c = calibration(&model, &samples[150..]);
        eprintln!(
            "calibration (ratio ~1): cov_E {:.2} z²_E {:.2} cov_F {:.2} z²_F {:.2}, held-out rmse_E {:.3e} (injected {se:e}), 3·s_E {:.3e}, s_F {:.3e}, λ {:.3e}, γ_E {:.1} γ_F {:.1}, rounds {}",
            c.cov_e, c.z2_e, c.cov_f, c.z2_f, c.rmse_e,
            3.0 * report.sigma_e,
            report.sigma_f,
            report.lambda,
            report.gamma_e,
            report.gamma_f,
            report.hyper_iterations
        );
        assert!(c.cov_e >= 0.90, "energy 2σ coverage {}", c.cov_e);
        // Calibrated: E[(resid/std)²] = 1; with 50 draws its std is √(2/50) = 0.2.
        assert!((0.5..1.5).contains(&c.z2_e), "mean normalised squared residual {}", c.z2_e);
        // The recovered noise floors are the injected ones (within 20 %): s_E is per atom.
        assert!((3.0 * report.sigma_e - se).abs() < 0.2 * se, "3·s_E = {:e}", 3.0 * report.sigma_e);
        assert!((report.sigma_f - sf).abs() < 0.2 * sf, "s_F = {:e}", report.sigma_f);
    }

    /// The realistic case: per-atom energy noise 30× below the force noise (1e-5 Ha/atom vs
    /// 3e-4 Ha/Bohr). A pooled noise level under-weights the energies by ~900 and inflates the
    /// energy variance (z²_E 0.39 in round 0); the two-noise posterior must be calibrated for
    /// both kinds.
    #[test]
    fn predictive_variance_is_calibrated_at_the_realistic_noise_ratio() {
        let mut rng = StdRng::seed_from_u64(8);
        let mut basis = Basis::new(&[H, O]);
        basis.body3 = false; // body-2 regression fixture (the default is body-3)
        basis.n_max = 4;
        let teacher = random_teacher(&mut rng, &basis);
        let n_atoms = 3;
        let (se_atom, sf) = (1e-5, 3e-4);
        let se_total = n_atoms as f64 * se_atom;
        let samples: Vec<Sample> = (0..250)
            .map(|_| {
                let (pos, z) = random_cluster(&mut rng, n_atoms, 4.0);
                teacher_sample(&teacher, pos, z, Some((se_total, sf, &mut rng)))
            })
            .collect();
        let mut train = SampleStore::new(provenance());
        for s in &samples[..150] {
            train.push(s.clone()).unwrap();
        }
        let (model, report) = fit(&train, &basis, &teacher.ref_energy, &FitOptions::default()).unwrap();
        let c = calibration(&model, &samples[150..]);
        eprintln!(
            "calibration (ratio 30): cov_E {:.2} z²_E {:.2} cov_F {:.2} z²_F {:.2}, held-out rmse_E {:.3e} (injected {se_total:e}), s_E {:.3e} (injected {se_atom:e}), s_F {:.3e}, λ {:.3e}, γ_E {:.1} γ_F {:.1}, dof_E {:.0} dof_F {:.0}, rounds {} conv {}",
            c.cov_e, c.z2_e, c.cov_f, c.z2_f, c.rmse_e,
            report.sigma_e,
            report.sigma_f,
            report.lambda,
            report.gamma_e,
            report.gamma_f,
            report.residual_dof_e,
            report.residual_dof_f,
            report.hyper_iterations,
            report.converged
        );
        assert!(c.cov_e >= 0.90, "energy 2σ coverage {}", c.cov_e);
        assert!(c.cov_f >= 0.90, "force 2σ coverage {}", c.cov_f);
        assert!((0.6..1.5).contains(&c.z2_e), "energy z² {}", c.z2_e);
        assert!((0.6..1.5).contains(&c.z2_f), "force z² {}", c.z2_f);
        assert!((report.sigma_e - se_atom).abs() < 0.3 * se_atom, "s_E = {:e}", report.sigma_e);
        assert!((report.sigma_f - sf).abs() < 0.2 * sf, "s_F = {:e}", report.sigma_f);
        assert!(report.converged);
        assert!(report.residual_dof_e >= 30.0 && report.residual_dof_f >= 30.0);
    }

    #[test]
    fn variance_grows_outside_the_trained_range() {
        let mut rng = StdRng::seed_from_u64(5);
        let store = morse_store(&mut rng, 1.2, 2.2, 30, Some((1e-4, 1e-4)));
        let basis = h2_basis(8);
        let (model, report) = fit(&store, &basis, &[E_ATOM_H], &FitOptions::default()).unwrap();
        let std_at = |r: f64| {
            let pos = [[0.0; 3], [r, 0.0, 0.0]];
            let pred = model.predict(&pos, &[H, H], true).unwrap();
            (pred.force_std.unwrap()[1], pred.energy_std.unwrap())
        };
        let (f_in, e_in) = std_at(1.7);
        let (f_out, e_out) = std_at(3.0);
        let (f_far, e_far) = std_at(5.0);
        eprintln!(
            "ood: force std 1.7 → {f_in:.3e}, 3.0 → {f_out:.3e}, 5.0 → {f_far:.3e}; energy std {e_in:.3e} / {e_out:.3e} / {e_far:.3e}; λ {:.3e} s_E {:.3e} s_F {:.3e}",
            report.lambda, report.sigma_e, report.sigma_f
        );
        assert!(f_out >= 5.0 * f_in, "force std out/in = {}", f_out / f_in);
        assert!(e_out >= 5.0 * e_in, "energy std out/in = {}", e_out / e_in);
        // Inside the window the std is at the noise floor (within a factor 2).
        assert!(f_in < 2.0 * 3f64.sqrt() * report.sigma_f, "in-window force std {f_in:e}");
    }

    /// A smooth non-linear teacher for the large-basis tests: pairwise Morse terms whose depth
    /// depends on the pair's species, plus a three-body bend term at the H–O–H vertex; exact forces.
    fn morse_bend(pos: &[[f64; 3]], z: &[u8]) -> (f64, Vec<[f64; 3]>) {
        let n = pos.len();
        let mut e = z.iter().map(|&zz| if zz == H { -0.478 } else { -74.8 }).sum::<f64>();
        let mut f = vec![[0.0; 3]; n];
        for i in 0..n {
            for j in i + 1..n {
                let d: [f64; 3] = std::array::from_fn(|a| pos[j][a] - pos[i][a]);
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                let depth = if z[i] == z[j] { 0.05 } else { 0.17 };
                let x = (-(r - 1.8)).exp();
                e += depth * ((1.0 - x) * (1.0 - x) - 1.0);
                let de = 2.0 * depth * (1.0 - x) * x;
                for a in 0..3 {
                    f[i][a] += de * d[a] / r;
                    f[j][a] -= de * d[a] / r;
                }
            }
        }
        // bend: k (cos θ − cos θ₀)² at every O vertex over its H pairs within 3 Bohr, by central
        // differences of this term alone (smooth; the test needs a non-pair surface, not its gradient analytically)
        let bend = |p: &[[f64; 3]]| -> f64 {
            let mut eb = 0.0;
            for c in 0..n {
                if z[c] != O {
                    continue;
                }
                for a in 0..n {
                    for b in a + 1..n {
                        if z[a] != H || z[b] != H {
                            continue;
                        }
                        let u: [f64; 3] = std::array::from_fn(|k| p[a][k] - p[c][k]);
                        let v: [f64; 3] = std::array::from_fn(|k| p[b][k] - p[c][k]);
                        let (ru, rv) = ((u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt(), (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt());
                        let w = (-(ru - 1.8).powi(2)).exp() * (-(rv - 1.8).powi(2)).exp();
                        let cos = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (ru * rv);
                        eb += 0.02 * w * (cos + 0.25).powi(2);
                    }
                }
            }
            eb
        };
        e += bend(pos);
        let h = 1e-6;
        for j in 0..n {
            for a in 0..3 {
                let mut p = pos.to_vec();
                p[j][a] += h;
                let up = bend(&p);
                p[j][a] -= 2.0 * h;
                let dn = bend(&p);
                f[j][a] -= (up - dn) / (2.0 * h);
            }
        }
        (e, f)
    }

    /// `n` noisy morse-bend samples of 5-atom H/O clusters (1e-5 Ha, 1e-4 Ha/Bohr scatter).
    fn morse_bend_samples(rng: &mut StdRng, n: usize) -> Vec<Sample> {
        morse_bend_noisy(rng, n, 1e-5, 1e-4)
    }

    fn morse_bend_noisy(rng: &mut StdRng, n: usize, se: f64, sf: f64) -> Vec<Sample> {
        (0..n)
            .map(|_| {
                let (pos, z) = random_cluster(rng, 5, 3.2);
                let (mut e, mut f) = morse_bend(&pos, &z);
                e += se * normal(rng);
                for fj in &mut f {
                    for a in 0..3 {
                        fj[a] += sf * normal(rng);
                    }
                }
                Sample { pos, z, energy: e, forces: f, tag: SampleTag::Perturbation, weight: 1.0 }
            })
            .collect()
    }

    /// A small body-4 basis for the large-basis path tests (Q′ in the low hundreds).
    fn small_body4_basis() -> Basis {
        let mut b = Basis::new(&[H, O]);
        b.body4 = true;
        b.n_max = 4;
        b.l_max = 2;
        b
    }

    /// The large-basis path (module doc) on a body-4 basis: the spectral search lands on the
    /// reference search's evidence maximum (both stop at 1e-3 relative), and its posterior is the
    /// reference's at that maximum.
    #[test]
    fn large_basis_search_finds_the_reference_search_optimum() {
        let mut rng = StdRng::seed_from_u64(77);
        let basis = small_body4_basis();
        let refs = [-0.478, -74.8];
        let opts = FitOptions::default();
        let mut acc = Accumulator::new(&basis);
        for s in morse_bend_samples(&mut rng, 150) {
            acc.add(&s, &basis, &refs, &opts).unwrap();
        }
        let t = std::time::Instant::now();
        let (m_ref, r_ref) = acc.solve(&opts, &provenance(), &refs).unwrap();
        let t_ref = t.elapsed();
        let t = std::time::Instant::now();
        let (m_sp, r_sp) = acc.refit(&opts, &provenance(), &refs).unwrap();
        let t_sp = t.elapsed();
        let rounds = match acc.last_refit() {
            RefitKind::Spectral { rounds, .. } => rounds,
            k => panic!("expected a spectral search, got {k:?}"),
        };
        eprintln!(
            "Q′ {}: reference λ {:.4e} σ_E {:.4e} σ_F {:.4e} ln p {:.6e} γ {:.2}/{:.2} ({} rounds, {:.0} ms); spectral λ {:.4e} σ_E {:.4e} σ_F {:.4e} ln p {:.6e} γ {:.2}/{:.2} ({rounds} rounds, {:.0} ms)",
            r_ref.q_prime, r_ref.lambda, r_ref.sigma_e, r_ref.sigma_f, r_ref.log_evidence, r_ref.gamma_e, r_ref.gamma_f, r_ref.hyper_iterations, t_ref.as_secs_f64() * 1e3,
            r_sp.lambda, r_sp.sigma_e, r_sp.sigma_f, r_sp.log_evidence, r_sp.gamma_e, r_sp.gamma_f, t_sp.as_secs_f64() * 1e3
        );
        assert!(r_ref.converged && r_sp.converged);
        // both searches stop when λ and the noise levels move by < 1e-3 per round: the optima agree
        // to a few times that
        for (what, a, b) in [("λ", r_ref.lambda, r_sp.lambda), ("σ_E²", r_ref.sigma_e.powi(2), r_sp.sigma_e.powi(2)), ("σ_F²", r_ref.sigma_f.powi(2), r_sp.sigma_f.powi(2))] {
            assert!(rel_change(a, b) < 1e-2, "{what}: reference {a:e} vs spectral {b:e}");
        }
        // the evidence at the two optima: equal to the curvature of a flat maximum
        assert!((r_ref.log_evidence - r_sp.log_evidence).abs() < 1e-6 * r_ref.log_evidence.abs().max(1.0), "{} vs {}", r_ref.log_evidence, r_sp.log_evidence);
        // predictions and stds on held-out clusters agree far below the noise levels
        for s in morse_bend_samples(&mut rng, 20) {
            let (a, b) = (m_ref.predict(&s.pos, &s.z, true).unwrap(), m_sp.predict(&s.pos, &s.z, true).unwrap());
            assert!((a.energy - b.energy).abs() < 0.05 * r_ref.sigma_e * 5.0, "energy {} vs {}", a.energy, b.energy);
            for (fa, fb) in a.forces.iter().zip(&b.forces) {
                for k in 0..3 {
                    assert!((fa[k] - fb[k]).abs() < 0.05 * r_ref.sigma_f, "force {} vs {}", fa[k], fb[k]);
                }
            }
            for (x, y) in a.force_std.unwrap().iter().zip(b.force_std.unwrap()) {
                assert!(rel_change(*x, y) < 1e-2, "force std {x} vs {y}");
            }
        }
    }

    /// The warm refit (module doc) is the exact posterior at the standing hyperparameters and
    /// standardisation: after samples arrive, its mean, factor, γ_E and γ_F equal a fresh
    /// factorisation's to rounding; growth beyond `WARM_GROWTH` searches again.
    #[test]
    fn a_long_warm_chain_stays_the_exact_posterior() {
        // fix round 1 (the review's M2): γ is carried by Woodbury subtraction across up to a
        // quarter of the store; ten warm refits of 19 samples each after a search at 800 (190
        // updates, ≈ 3 000 rank-one steps) against a fresh solve at the standing settings
        let mut rng = StdRng::seed_from_u64(79);
        let basis = small_body4_basis();
        let refs = [-0.478, -74.8];
        let opts = FitOptions::default();
        let mut acc = Accumulator::new(&basis);
        // labels at a DFT-like scatter (1e-4 Ha, 2e-3 Ha/Bohr): λ above the ridge floor; at the
        // 1e-4 Ha/Bohr scatter the floor binds and grows with the data, and the warm refit
        // correctly declines (asserted below)
        for s in morse_bend_noisy(&mut rng, 800, 1e-4, 2e-3) {
            acc.add(&s, &basis, &refs, &opts).unwrap();
        }
        acc.refit(&opts, &provenance(), &refs).unwrap();
        let mut last = None;
        for _ in 0..10 {
            for s in morse_bend_noisy(&mut rng, 19, 1e-4, 2e-3) {
                acc.add(&s, &basis, &refs, &opts).unwrap();
            }
            last = Some(acc.refit(&opts, &provenance(), &refs).unwrap());
            assert_eq!(acc.last_refit(), RefitKind::Warm { samples: 19 });
        }
        let (m_w, r_w) = last.unwrap();
        let w = acc.warm.as_ref().unwrap();
        let (_, _, problem) = acc.prepare(&refs, Some((&w.scale, &w.d))).unwrap();
        let (sol, extra) = problem.solve_large(w.lambda, w.s_e2, w.s_f2).unwrap();
        eprintln!("λ {:e}, λ_eff warm {:e}, fresh {:e}", w.lambda, w.lambda_eff, extra.lambda_eff);
        let q = acc.q_prime;
        let chol = m_w.sigma_inv_chol.as_ref().unwrap();
        let amax = sol.a.iter().fold(0.0f64, |x, v| x.max(v.abs()));
        let mut worst_a: f64 = 0.0;
        for i in 0..q {
            for j in 0..=i {
                worst_a = worst_a.max((dot(&chol[i * q..i * q + j + 1], &chol[j * q..j * q + j + 1]) - sol.a[i * q + j]).abs() / amax);
            }
        }
        let hyper = Hyper { lambda: w.lambda, s_e2: w.s_e2, s_f2: w.s_f2, iterations: 0, converged: true };
        let (m_f, _) = acc.assemble(&opts, &provenance(), &refs, &w.scale, &w.d, &hyper, &sol, 0.0, 1.0).unwrap();
        let (mut worst_f, mut worst_std): (f64, f64) = (0.0, 0.0);
        for s in morse_bend_samples(&mut StdRng::seed_from_u64(600), 10) {
            let (a, b) = (m_w.predict(&s.pos, &s.z, true).unwrap(), m_f.predict(&s.pos, &s.z, true).unwrap());
            for (fa, fb) in a.forces.iter().zip(&b.forces) {
                for k in 0..3 {
                    worst_f = worst_f.max((fa[k] - fb[k]).abs() / r_w.sigma_f);
                }
            }
            for (x, y) in a.force_std_epistemic.unwrap().iter().zip(b.force_std_epistemic.unwrap()) {
                worst_std = worst_std.max(rel_change(*x, y));
            }
        }
        let bound = 10.0 * r_w.condition_estimate * f64::EPSILON;
        eprintln!(
            "long warm chain (800 + 10×19, Q′ {q}): backward error {worst_a:.1e}, |ΔF|/σ_F {worst_f:.1e} (bound {bound:.1e}), std rel {worst_std:.1e}, γ_E {:.6} vs {:.6}, γ_F {:.6} vs {:.6}",
            r_w.gamma_e, sol.gamma_e, r_w.gamma_f, sol.gamma_f
        );
        assert_eq!(extra.lambda_eff, w.lambda_eff, "λ above the floor");
        assert!(worst_a < 1e-13 && worst_f < bound && worst_std < 1e-5);
        assert!(rel_change(r_w.gamma_e, sol.gamma_e) < 1e-5 && rel_change(r_w.gamma_f, sol.gamma_f) < 1e-5);
        // the floor-bound case: the warm refit declines once the floor passes the standing λ_eff
        let mut acc = Accumulator::new(&basis);
        for s in morse_bend_samples(&mut rng, 800) {
            acc.add(&s, &basis, &refs, &opts).unwrap();
        }
        acc.refit(&opts, &provenance(), &refs).unwrap();
        let mut kinds = Vec::new();
        for _ in 0..10 {
            for s in morse_bend_samples(&mut rng, 19) {
                acc.add(&s, &basis, &refs, &opts).unwrap();
            }
            acc.refit(&opts, &provenance(), &refs).unwrap();
            kinds.push(acc.last_refit());
            let w = acc.warm.as_ref().unwrap();
            let (_, _, problem) = acc.prepare(&refs, Some((&w.scale, &w.d))).unwrap();
            let (_, extra) = problem.solve_large(w.lambda, w.s_e2, w.s_f2).unwrap();
            assert_eq!(extra.lambda_eff, w.lambda_eff, "whatever the kind, the standing posterior's ridge is the fresh one's ({kinds:?})");
        }
        eprintln!("floor-bound chain: {kinds:?}");
    }

    #[test]
    fn warm_refit_is_the_exact_posterior_at_the_standing_hyperparameters() {
        let mut rng = StdRng::seed_from_u64(78);
        let basis = small_body4_basis();
        let refs = [-0.478, -74.8];
        let opts = FitOptions::default();
        let mut acc = Accumulator::new(&basis);
        for s in morse_bend_samples(&mut rng, 120) {
            acc.add(&s, &basis, &refs, &opts).unwrap();
        }
        acc.refit(&opts, &provenance(), &refs).unwrap();
        assert!(matches!(acc.last_refit(), RefitKind::Spectral { .. }));
        for round in 0..3 {
            for s in morse_bend_samples(&mut rng, 3 + round) {
                acc.add(&s, &basis, &refs, &opts).unwrap();
            }
            let (m_w, r_w) = acc.refit(&opts, &provenance(), &refs).unwrap();
            assert_eq!(acc.last_refit(), RefitKind::Warm { samples: 3 + round });
            // the reference: a fresh dense solve at the warm state's frozen standardisation and
            // hyperparameters
            let w = acc.warm.as_ref().unwrap();
            let (_, _, problem) = acc.prepare(&refs, Some((&w.scale, &w.d))).unwrap();
            let (sol, extra) = problem.solve_large(w.lambda, w.s_e2, w.s_f2).unwrap();
            assert_eq!(extra.lambda_eff, w.lambda_eff, "noisy data: λ is far above the floor");
            let q = acc.q_prime;
            // the factor: backward error of L Lᵀ against the fresh precision A (the factor itself is
            // only determined to κ(A)·ε in its weak directions — κ ~ 1e9 here — like the mean)
            let chol = m_w.sigma_inv_chol.as_ref().unwrap();
            let amax = sol.a.iter().fold(0.0f64, |x, v| x.max(v.abs()));
            let mut worst_a: f64 = 0.0;
            for i in 0..q {
                for j in 0..=i {
                    let llt = dot(&chol[i * q..i * q + j + 1], &chol[j * q..j * q + j + 1]);
                    worst_a = worst_a.max((llt - sol.a[i * q + j]).abs() / amax);
                }
            }
            // the fresh model at the same hyperparameters and standardisation
            let hyper = Hyper { lambda: w.lambda, s_e2: w.s_e2, s_f2: w.s_f2, iterations: 0, converged: true };
            let (m_f, _) = acc.assemble(&opts, &provenance(), &refs, &w.scale, &w.d, &hyper, &sol, 0.0, 1.0).unwrap();
            let (mut worst_e, mut worst_f, mut worst_std): (f64, f64, f64) = (0.0, 0.0, 0.0);
            for s in morse_bend_samples(&mut StdRng::seed_from_u64(500 + round as u64), 10) {
                let (a, b) = (m_w.predict(&s.pos, &s.z, true).unwrap(), m_f.predict(&s.pos, &s.z, true).unwrap());
                worst_e = worst_e.max((a.energy - b.energy).abs() / (5.0 * r_w.sigma_e));
                for (fa, fb) in a.forces.iter().zip(&b.forces) {
                    for k in 0..3 {
                        worst_f = worst_f.max((fa[k] - fb[k]).abs() / r_w.sigma_f);
                    }
                }
                for (x, y) in a.force_std_epistemic.unwrap().iter().zip(b.force_std_epistemic.unwrap()) {
                    worst_std = worst_std.max(rel_change(*x, y));
                }
                worst_std = worst_std.max(rel_change(a.energy_std.unwrap(), b.energy_std.unwrap()));
            }
            eprintln!(
                "warm round {round}: Q′ {q}, κ(A) {:.1e}, backward error of LLᵀ {worst_a:.1e}, held-out |ΔE|/(Nσ_E) {worst_e:.1e} |ΔF|/σ_F {worst_f:.1e}, std rel {worst_std:.1e}, γ_E {:.6} vs {:.6}, γ_F {:.6} vs {:.6}",
                r_w.condition_estimate, r_w.gamma_e, sol.gamma_e, r_w.gamma_f, sol.gamma_f
            );
            // two backward-stable factorisations of the same A (κ ~ 6e12 on this fixture) give
            // solutions that differ by up to ~κ(A)·ε relative — the reference path's own Cholesky
            // against the vectorised one differs by as much (printed) — so the prediction bound
            // is 10 κ ε in noise units, the std's 1e-5 (a quadratic form: far less sensitive)
            let (m_old, _) = {
                let sol_old = problem.solve(w.lambda, w.s_e2, w.s_f2).unwrap();
                acc.assemble(&opts, &provenance(), &refs, &w.scale, &w.d, &hyper, &sol_old, 0.0, 1.0).unwrap()
            };
            let mut spread: f64 = 0.0;
            for s in morse_bend_samples(&mut StdRng::seed_from_u64(500 + round as u64), 10) {
                let (a, b) = (m_old.predict(&s.pos, &s.z, false).unwrap(), m_f.predict(&s.pos, &s.z, false).unwrap());
                for (fa, fb) in a.forces.iter().zip(&b.forces) {
                    for k in 0..3 {
                        spread = spread.max((fa[k] - fb[k]).abs() / r_w.sigma_f);
                    }
                }
            }
            let bound = 10.0 * r_w.condition_estimate * f64::EPSILON;
            eprintln!("  the two fresh Cholesky kernels against each other: |ΔF|/σ_F {spread:.1e}; bound 10κε = {bound:.1e}");
            assert!(worst_a < 1e-13, "backward error {worst_a:e}");
            assert!(worst_e < bound && worst_f < bound && worst_std < 1e-5, "predictions {worst_e:e} {worst_f:e} {worst_std:e} (bound {bound:e})");
            assert!(rel_change(r_w.gamma_e, sol.gamma_e) < 1e-5 && rel_change(r_w.gamma_f, sol.gamma_f) < 1e-5);
            assert!(rel_change(r_w.rmse_force, (sol.rss_f / problem.n_f).sqrt()) < 1e-6);
        }
        // past the growth bound the hyperparameters are searched again
        let n_now = acc.n_samples;
        for s in morse_bend_samples(&mut rng, (n_now as f64 * (WARM_GROWTH - 1.0)) as usize + 1) {
            acc.add(&s, &basis, &refs, &opts).unwrap();
        }
        acc.refit(&opts, &provenance(), &refs).unwrap();
        assert!(matches!(acc.last_refit(), RefitKind::Spectral { .. }), "{:?}", acc.last_refit());
        // a basis without body-4 never takes the large-basis path
        let mut plain = Accumulator::new(&h2_basis(4));
        let mut rng2 = StdRng::seed_from_u64(3);
        for s in morse_store(&mut rng2, 1.0, 3.0, 40, Some((1e-5, 1e-4))).samples {
            plain.add(&s, &h2_basis(4), &[E_ATOM_H], &opts).unwrap();
        }
        let (a, _) = plain.refit(&opts, &provenance(), &[E_ATOM_H]).unwrap();
        let (b, _) = plain.solve(&opts, &provenance(), &[E_ATOM_H]).unwrap();
        assert_eq!(plain.last_refit(), RefitKind::Full);
        assert_eq!(a.coeffs, b.coeffs);
    }

    /// The spectral truncation of the posterior covariance Σ ≈ Σ_{k<K} s_k u_k u_kᵀ + s_min I
    /// (the M5 handoff's remedy for the O(N·Q′²) variance; M8a Task 5 ruling: measure what it does
    /// to the calibration, floor it so it never claims more confidence than Σ). With s_min = s_K —
    /// the largest dropped eigenvalue — T − Σ = Σ_{k<K} s_K u_k u_kᵀ + Σ_{k≥K} (s_K − s_k) u_k u_kᵀ ⪰ 0,
    /// so the truncated variance is never below the exact one (asserted on held-out rows); without
    /// the floor it is biased low. Printed: the ratios on the held-out design rows of a collection-
    /// sized body-4 fit (S = 2, Q′ 284) — the numbers of record in `references.md` ("M8a body-4").
    #[test]
    fn spectral_truncation_of_the_covariance_is_measured_and_floored() {
        let mut rng = StdRng::seed_from_u64(91);
        let mut basis = Basis::new(&[H, O]);
        basis.body4 = true;
        let refs = [-0.478, -74.8];
        let opts = FitOptions::default();
        let mut acc = Accumulator::new(&basis);
        for s in morse_bend_samples(&mut rng, 400) {
            acc.add(&s, &basis, &refs, &opts).unwrap();
        }
        let (model, report) = acc.refit(&opts, &provenance(), &refs).unwrap();
        let q = report.q_prime;
        let l = model.sigma_inv_chol.as_ref().unwrap();
        let sigma = inverse_from_lower_inverse(&lower_inverse_transposed(l, q), q);
        let (s, u) = crate::linalg::symmetric_eigen_householder(&sigma, q);
        let mut order: Vec<usize> = (0..q).collect();
        order.sort_by(|&a, &b| s[b].total_cmp(&s[a]));
        let prior = 1.0 / report.lambda;
        let at_prior = s.iter().filter(|&&v| v > 0.5 * prior).count();
        // held-out design rows (energy and force rows, standardised as the fit's)
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for smp in morse_bend_samples(&mut rng, 40) {
            let (e, f) = crate::loo::design_rows(&model, &smp.pos, &smp.z).unwrap();
            rows.push(e);
            rows.extend(f);
        }
        let proj: Vec<Vec<f64>> = rows.iter().map(|r| order.iter().map(|&k| (0..q).map(|i| u[i * q + k] * r[i]).sum::<f64>()).collect()).collect();
        let sorted: Vec<f64> = order.iter().map(|&k| s[k]).collect();
        eprintln!("Q′ {q}, λ {:.3e} (prior variance {prior:.3e}); eigenvalues of Σ above half the prior's: {at_prior}; s_1 {:.3e}, s_50 {:.3e}, s_100 {:.3e}, s_Q′ {:.3e}", report.lambda, sorted[0], sorted[49], sorted[99], sorted[q - 1]);
        let pct = |v: &mut Vec<f64>, p: f64| {
            v.sort_by(f64::total_cmp);
            v[((v.len() - 1) as f64 * p).round() as usize]
        };
        for k_keep in [25usize, 50, 100, 200] {
            let s_min = sorted[k_keep];
            let (mut floored, mut bare) = (Vec::new(), Vec::new());
            for p in &proj {
                let exact: f64 = p.iter().zip(&sorted).map(|(x, s)| s * x * x).sum();
                let kept: f64 = p[..k_keep].iter().zip(&sorted).map(|(x, s)| s * x * x).sum();
                let norm2: f64 = p.iter().map(|x| x * x).sum();
                if exact > 0.0 {
                    floored.push((kept + s_min * norm2) / exact);
                    bare.push(kept / exact);
                }
            }
            let min_f = floored.iter().copied().fold(f64::INFINITY, f64::min);
            assert!(min_f >= 1.0 - 1e-9, "K {k_keep}: the floored truncation claimed more confidence than Σ ({min_f})");
            eprintln!(
                "K {k_keep}: floored (s_min = s_K {s_min:.2e}) variance ratio truncated/exact min {min_f:.3} median {:.3} p95 {:.3e} max {:.3e}; unfloored min {:.3e} median {:.3}",
                pct(&mut floored.clone(), 0.5),
                pct(&mut floored.clone(), 0.95),
                pct(&mut floored, 1.0),
                pct(&mut bare.clone(), 0.0),
                pct(&mut bare, 0.5)
            );
        }
    }

    #[test]
    fn accumulator_refit_equals_a_batch_fit() {
        let mut rng = StdRng::seed_from_u64(6);
        let store = morse_store(&mut rng, 1.0, 3.0, 40, Some((1e-4, 3e-4)));
        let basis = h2_basis(8);
        let opts = FitOptions::default();
        let (batch, rb) = fit(&store, &basis, &[E_ATOM_H], &opts).unwrap();
        let mut acc = Accumulator::new(&basis);
        for s in &store.samples {
            acc.add(s, &basis, &[E_ATOM_H], &opts).unwrap();
        }
        assert_eq!(acc.n_rows(), rb.n_rows);
        let (inc, ri) = acc.solve(&opts, &store.provenance, &[E_ATOM_H]).unwrap();
        for s in 0..basis.n_species() {
            for k in 0..basis.n_features() {
                assert!(
                    (inc.coeffs[s][k] - batch.coeffs[s][k]).abs() <= 1e-12 * batch.coeffs[s][k].abs().max(1.0),
                    "coefficient ({s},{k}): {} vs {}",
                    inc.coeffs[s][k],
                    batch.coeffs[s][k]
                );
            }
        }
        assert_eq!(inc.scale, batch.scale);
        assert_eq!((inc.sigma_e, inc.sigma_f, inc.lambda), (batch.sigma_e, batch.sigma_f, batch.lambda));
        assert_eq!(inc.sigma_inv_chol, batch.sigma_inv_chol);
        assert_eq!((ri.sigma_e, ri.sigma_f, ri.lambda, ri.log_evidence), (rb.sigma_e, rb.sigma_f, rb.lambda, rb.log_evidence));
        // The statistics-based RMSE agrees with the exact one when the residual is not tiny.
        assert!((ri.rmse_force - rb.rmse_force).abs() < 1e-6 * rb.rmse_force);

        // Adding the same samples in reverse order changes only the rounding of the sums. At the
        // batch's own hyperparameters (pinned) that is all that differs: the coefficients agree to
        // the rounding of compensated sums. With the hyperparameters re-estimated the evidence
        // search may stop at another λ within its resolution (`OUTER_TOL` = 1e-3 relative; measured
        // 6e-5 since the Gram became compensated, M6 Task 11 round 3), which moves the coefficients
        // by that resolution, not by the sums — reported, not compared at 1e-8.
        let mut rev = Accumulator::new(&basis);
        for s in store.samples.iter().rev() {
            rev.add(s, &basis, &[E_ATOM_H], &opts).unwrap();
        }
        let pinned = FitOptions { sigma_e: Some(rb.sigma_e), sigma_f: Some(rb.sigma_f), lambda: Some(rb.lambda), ..opts };
        let (fwd, _) = acc.solve(&pinned, &store.provenance, &[E_ATOM_H]).unwrap();
        let (rm, _) = rev.solve(&pinned, &store.provenance, &[E_ATOM_H]).unwrap();
        for k in 0..basis.n_features() {
            assert!((rm.coeffs[0][k] - fwd.coeffs[0][k]).abs() < 1e-10 * fwd.coeffs[0][k].abs().max(1.0), "k {k}: {} vs {}", rm.coeffs[0][k], fwd.coeffs[0][k]);
        }
        let (free, _) = rev.solve(&opts, &store.provenance, &[E_ATOM_H]).unwrap();
        eprintln!("reverse order, hyperparameters re-estimated: λ {} vs {} (relative {:.1e}; search resolution {OUTER_TOL:.0e})", free.lambda, batch.lambda, (free.lambda / batch.lambda - 1.0).abs());
        assert!((free.lambda / batch.lambda - 1.0).abs() < OUTER_TOL);
    }

    /// The Q′ accumulation must give the same Gram as the full-layout one restricted to the
    /// representatives: check on a 2-species store that the hetero class gets the pair sum once.
    #[test]
    fn accumulator_on_representatives_matches_the_descriptor_sums() {
        let mut rng = StdRng::seed_from_u64(9);
        let mut basis = Basis::new(&[H, O]);
        basis.body3 = false; // body-2 regression fixture (the default is body-3)
        basis.n_max = 3;
        let teacher = random_teacher(&mut rng, &basis);
        let (pos, z) = random_cluster(&mut rng, 4, 4.0);
        let sample = teacher_sample(&teacher, pos.clone(), z.clone(), None);
        let opts = FitOptions { energy_per_atom: false, ..Default::default() };
        let mut acc = Accumulator::new(&basis);
        acc.add(&sample, &basis, &teacher.ref_energy, &opts).unwrap();
        // Energy-row feature of class c = Σ_{i∈s} B_i[k] for its representative (s, k).
        let desc = describe(&basis, &pos, &z).unwrap();
        let p = basis.n_features();
        let classes = basis.column_classes();
        let s: Vec<usize> = basis.species_indices(&z).unwrap();
        let mut phi = vec![0.0; classes.len()];
        for (c, class) in classes.iter().enumerate() {
            let (sp, k) = (class[0] / p, class[0] % p);
            for (i, d) in desc.iter().enumerate() {
                if s[i] == sp {
                    phi[c] += d.b[k];
                }
            }
            // The duplicate column carries the identical value.
            if class.len() == 2 {
                let (sp2, k2) = (class[1] / p, class[1] % p);
                let dup: f64 = desc.iter().enumerate().filter(|(i, _)| s[*i] == sp2).map(|(_, d)| d.b[k2]).sum();
                assert!((dup - phi[c]).abs() < 1e-12, "tied columns agree");
            }
        }
        let y = sample.energy - z.iter().map(|&zi| teacher.ref_energy[basis.species_index(zi).unwrap()]).sum::<f64>();
        for i in 0..classes.len() {
            assert!((acc.rhs_e[i] - phi[i] * y).abs() < 1e-12);
            for j in 0..classes.len() {
                assert!((acc.gram_e[i * classes.len() + j] - phi[i] * phi[j]).abs() < 1e-12);
            }
        }
        assert_eq!(acc.n_e, 1);
        assert_eq!(acc.n_f, 12);
        assert_eq!(acc.q_prime, 9);
    }

    #[test]
    fn errors_name_the_missing_reference_and_empty_store() {
        let mut basis = Basis::new(&[H, O]);
        basis.body3 = false; // body-2 regression fixture (the default is body-3)
        let store = SampleStore::new(provenance());
        assert_eq!(
            fit(&store, &basis, &[-0.5, -75.0], &FitOptions::default()).unwrap_err(),
            FitError::NoSamples
        );
        let mut rng = StdRng::seed_from_u64(7);
        let store = morse_store(&mut rng, 1.0, 2.0, 3, None);
        assert_eq!(
            fit(&store, &basis, &[-0.5], &FitOptions::default()).unwrap_err(),
            FitError::MissingReference(O)
        );
        assert_eq!(
            fit(&store, &basis, &[-0.5, f64::NAN], &FitOptions::default()).unwrap_err(),
            FitError::MissingReference(O)
        );
        assert_eq!(
            fit(&store, &basis, &[-0.5, -75.0, 1.0], &FitOptions::default()).unwrap_err(),
            FitError::ReferenceLength { expected: 2, got: 3 }
        );
        let mut li = Basis::new(&[3]);
        li.body3 = false; // body-2 regression fixture (the default is body-3)
        assert_eq!(
            fit(&store, &li, &[-7.0], &FitOptions::default()).unwrap_err(),
            FitError::Describe(DescribeError::UnknownSpecies(H))
        );
        // λ = 0 with species blocks that have no data (no O atom in an H₂ store): the ridge
        // floor 4ε·max A_kk gives the minimum-norm solution, zero in the dead blocks, and the
        // H–H block still reproduces the data.
        let no_ridge = FitOptions { lambda: Some(0.0), ..Default::default() };
        let (m0, r0) = fit(&store, &basis, &[-0.5, -75.0], &no_ridge).unwrap();
        assert_eq!(r0.lambda, 0.0);
        let n_max = basis.n_max;
        for nn in 0..n_max {
            assert_eq!(m0.coeffs[0][basis.body2_index(1, nn)], 0.0, "dead (H, O) block");
            assert_eq!(m0.coeffs[1][basis.body2_index(0, nn)], 0.0, "dead (O, H) block");
            assert_eq!(m0.coeffs[1][basis.body2_index(1, nn)], 0.0, "dead (O, O) block");
        }
        assert!(r0.rmse_force < 1e-8, "3 noise-free points, 8 radial functions: interpolated");
        assert!(r0.log_evidence == f64::NEG_INFINITY, "improper prior has no evidence");
        // Genuinely singular: every feature is zero (atoms beyond r_cut), the Gram is all zero.
        let mut far = SampleStore::new(provenance());
        far.push(dimer_sample(7.0, [1.0, 0.0, 0.0], [0.0; 3], None)).unwrap();
        far.push(dimer_sample(8.0, [0.0, 1.0, 0.0], [0.0; 3], None)).unwrap();
        assert_eq!(fit(&far, &basis, &[-0.5, -75.0], &no_ridge).unwrap_err(), FitError::Singular);
        // With a prior it is not singular: every coefficient is 0 and the energy is Σ ref.
        let (m_far, _) = fit(&far, &basis, &[-0.5, -75.0], &FitOptions::default()).unwrap();
        assert!(m_far.coeffs.iter().flatten().all(|c| *c == 0.0));
        assert_eq!(m_far.predict(&far.samples[0].pos, &[H, H], false).unwrap().energy, -1.0);
    }

    #[test]
    fn store_checks_shapes_weights_and_provenance() {
        let mut store = SampleStore::new(provenance());
        let good = dimer_sample(1.5, [1.0, 0.0, 0.0], [0.0; 3], None);
        store.push(good.clone()).unwrap();
        let mut bad = good.clone();
        bad.forces.pop();
        assert_eq!(
            store.push(bad),
            Err(StoreError::ShapeMismatch { n_pos: 2, n_z: 2, n_forces: 1 })
        );
        let mut w = good.clone();
        w.weight = 0.0;
        assert_eq!(store.push(w), Err(StoreError::NonPositiveWeight(0.0)));
        let mut other = provenance();
        other.tol = 1e-5;
        assert_eq!(store.push_checked(good.clone(), &other), Err(StoreError::ProvenanceMismatch));
        store.push_checked(good, &provenance()).unwrap();
        assert_eq!(store.len(), 2);
    }
}
