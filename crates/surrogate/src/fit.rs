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
//! λ_eff = max(λ, 4ε·max_k (G̃_E/s_E² + G̃_F/s_F²)_kk): inert for any noisy data (λ is orders of
//! magnitude larger than the floor there), and for noise-free data or `lambda: Some(0.0)` it is
//! the minimum-norm regulariser (dead columns get coefficient 0; their posterior variance is
//! then 1/λ_eff, i.e. "no prior"). The evidence keeps the nominal λ in its (Q′/2) ln λ and
//! λ‖m‖² terms while ln det A and m are at λ_eff; in the floor regime the evidence maximum is
//! then MacKay's α = Q′/‖m‖² at the minimum-norm solution, which is what a noise-free fit
//! reports as λ. The constant 4ε does not grow with Q′ (the factorisation's backward error does,
//! O(Q′ε)), so exact ties must be removed by column classes, never left to the floor.
//! [`FitError::Singular`] is left for a Gram that is entirely zero (every feature zero on every
//! row, e.g. all atoms beyond r_cut) with λ = 0, or a numerical failure at the chosen
//! hyperparameters. A failed factorisation at one point of the λ grid only scores that point
//! −∞ (a near-collinear basis — body-3 on a small molecule — has rounding eigenvalues of the
//! whitened Gram below −4ε·max A_kk at the 1e-8 end of the grid, where no evidence maximum
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
    cholesky_in_place, cholesky_inverse, cholesky_logdet, cholesky_solve, mat_vec,
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
    /// Energy rows carry E/N and features/N (default true).
    pub energy_per_atom: bool,
}

impl Default for FitOptions {
    fn default() -> Self {
        FitOptions {
            lambda: None,
            sigma_e: None,
            sigma_f: None,
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
    /// κ₂ estimate of the posterior precision A.
    pub condition_estimate: f64,
    /// Outer alternation rounds of the hyperparameter search, and whether it met 1e-3.
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
    /// The posterior precision is not positive definite: λ = 0 with an all-zero Gram (every
    /// feature zero on every row), or a numerical failure of the factorisation (module doc,
    /// "Ridge floor").
    Singular,
    Describe(DescribeError),
}

impl fmt::Display for FitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FitError::NoSamples => write!(f, "no samples to fit"),
            FitError::MissingReference(z) => write!(f, "no reference energy for Z = {z}"),
            FitError::ReferenceLength { expected, got } => {
                write!(f, "{got} reference energies for {expected} species")
            }
            FitError::Singular => write!(f, "posterior precision is singular (λ = 0 with an all-zero Gram)"),
            FitError::Describe(e) => write!(f, "{e}"),
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
            phi: vec![0.0; qr],
            rows: Vec::new(),
            support: Vec::with_capacity(qr),
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
        rank1_update(&mut self.gram_e, &mut self.rhs_e, qr, &phi, y, w, &mut self.support);
        self.yy_e += w * y * y;
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
        for j in 0..n {
            for a in 0..3 {
                let f = sample.forces[j][a];
                let row = &rows[(3 * j + a) * qr..(3 * j + a + 1) * qr];
                rank1_update(&mut self.gram_f, &mut self.rhs_f, qr, row, f, w, &mut self.support);
                self.yy_f += w * f * f;
                self.n_f += 1;
                self.sum_log_w_f += w.ln();
            }
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
        if self.n_samples == 0 || self.n_e == 0 {
            return Err(FitError::NoSamples);
        }
        check_references(&self.basis, ref_energy)?;
        let basis = &self.basis;
        let p = basis.n_features();
        let n_s = basis.n_species();
        let q = self.q_prime;
        let n_e = self.n_e as f64;
        let n_f = self.n_f as f64;
        let n_rows = self.n_e + self.n_f;

        // Scale from the energy rows; D per representative column through k = q_full mod P.
        let scale: Vec<f64> = self
            .sumsq
            .iter()
            .map(|&ss| if ss > 0.0 { (ss / n_e).sqrt() } else { 1.0 })
            .collect();
        let d: Vec<f64> = self.classes.iter().map(|c| 1.0 / scale[c[0] % p]).collect();
        let standardise = |g: &[f64]| -> Vec<f64> {
            let mut out = vec![0.0; q * q];
            for i in 0..q {
                for j in 0..q {
                    out[i * q + j] = d[i] * g[i * q + j] * d[j];
                }
            }
            out
        };
        let problem = TwoNoise {
            g_e: standardise(&self.gram_e),
            g_f: standardise(&self.gram_f),
            b_e: (0..q).map(|i| d[i] * self.rhs_e[i]).collect(),
            b_f: (0..q).map(|i| d[i] * self.rhs_f[i]).collect(),
            yy_e: self.yy_e,
            yy_f: self.yy_f,
            n_e,
            n_f,
            q,
            sum_log_w: self.sum_log_w_e + self.sum_log_w_f,
            floor_e2: (1e-16 * self.yy_e / n_e).max(f64::MIN_POSITIVE),
            floor_f2: if n_f > 0.0 {
                (1e-16 * self.yy_f / n_f).max(f64::MIN_POSITIVE)
            } else {
                f64::MIN_POSITIVE
            },
        };
        let (hyper, sol) = problem.estimate(opts)?;
        let sigma_e = hyper.s_e2.sqrt();
        let sigma_f = hyper.s_f2.sqrt();
        let log_evidence = problem.evidence(hyper.lambda, hyper.s_e2, hyper.s_f2, &sol);

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
        let model = Model {
            basis: basis.clone(),
            coeffs,
            mean: vec![0.0; p],
            scale,
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
            condition_estimate: condition_estimate(&sol.a, &sol.chol, q),
            hyper_iterations: hyper.iterations,
            converged: hyper.converged,
        };
        Ok((model, report))
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

/// G += w v vᵀ, b += w y v over the support of v (the species blocks a cluster touches).
fn rank1_update(
    gram: &mut [f64],
    rhs: &mut [f64],
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
        rhs[i] += wi * y;
        let row = &mut gram[i * q..(i + 1) * q];
        for &j in support.iter() {
            row[j] += wi * v[j];
        }
    }
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
        let lambda_eff = lambda.max(4.0 * f64::EPSILON * max_diag);
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

    /// Alternate λ search and noise fixed points until all three change by < 1e-3 relative
    /// (module doc). Pinned values are used as given.
    fn estimate(&self, opts: &FitOptions) -> Result<(Hyper, Solved), FitError> {
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

        // Adding the same samples in reverse order changes only the rounding of the sums.
        let mut rev = Accumulator::new(&basis);
        for s in store.samples.iter().rev() {
            rev.add(s, &basis, &[E_ATOM_H], &opts).unwrap();
        }
        let (rm, _) = rev.solve(&opts, &store.provenance, &[E_ATOM_H]).unwrap();
        for k in 0..basis.n_features() {
            assert!((rm.coeffs[0][k] - batch.coeffs[0][k]).abs() < 1e-8 * batch.coeffs[0][k].abs().max(1.0));
        }
    }

    /// The Q′ accumulation must give the same Gram as the full-layout one restricted to the
    /// representatives: check on a 2-species store that the hetero class gets the pair sum once.
    #[test]
    fn accumulator_on_representatives_matches_the_descriptor_sums() {
        let mut rng = StdRng::seed_from_u64(9);
        let mut basis = Basis::new(&[H, O]);
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
        let basis = Basis::new(&[H, O]);
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
        let li = Basis::new(&[3]);
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
