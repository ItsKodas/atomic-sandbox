//! The fit: sample store, the design matrix of energies and forces, scale-only standardisation,
//! ridge least squares as Bayesian linear regression, evidence-chosen hyperparameters and the
//! predictive variance. Pure f64, CPU. The sufficient statistics live in [`Accumulator`], so a
//! refit after new samples is one Cholesky and the sidecar (Task 5) can store the statistics
//! instead of the samples.
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
//!   so that one energy row carries a per-atom quantity and is not drowned by 3N force rows
//!   (spec review C2c). Row weight w = `Sample::weight`.
//! * **3N force rows**, one per atom j and component a: features φ_{F,ja} with block s =
//!   −Σ_{i: s_i = s} ∂B_i/∂r_{j,a} (the gradient lists of [`describe`]; the sum over centre
//!   atoms i is what makes the force depend on every atom's coefficient row), target F_{j,a}.
//!   Row weight w = `Sample::weight`·`force_weight`. Forces do not depend on the references.
//!
//! # Standardisation at solve time (ruling S5: scale only, no mean, no intercept)
//!
//! A single ridge penalty on raw columns penalises in arbitrary units (C1), so the solve works in
//! standardised features φ̃ = D φ with D = diag(1/scale_k) (the same scale for every species
//! block of feature k, which is what `Model::scale` holds), scale_k² = (1/n_E) Σ_{energy rows}
//! Σ_s φ_{E,(s,k)}² — the RMS of feature k over the energy rows. A feature identically zero on
//! the energy rows gets scale 1 (its column is then all-zero, the ridge drives its coefficient to
//! 0 and the prior alone decides its variance, see "Dead columns"). No mean is subtracted: the
//! energy of an atom with no neighbour inside r_cut must be exactly ref[s] and the binding
//! energy must go to 0 beyond r_cut, which a centred feature or an intercept would break
//! (`Model` docs); the ridge shrinks towards the reference atoms, which is the physically right
//! prior. Force rows use the same D.
//!
//! The scale is a function of the data, so it cannot be fixed before the first sample is seen.
//! The statistics are therefore accumulated RAW — G = ΦᵀWΦ, b = ΦᵀWy, yᵀWy, per kind (energy /
//! force), together with the per-feature sum of squares over the energy rows — and the scaling
//! is applied as exact algebra at solve time: Φ̃ᵀWΦ̃ = D G D, Φ̃ᵀWy = D b. `Accumulator::add`
//! never needs the final scale, and adding a sample later changes the scale without any
//! re-accumulation.
//!
//! # Ridge = Bayesian linear regression
//!
//! With the weighted rows ỹ = W^{1/2} y, Φ̂ = W^{1/2} Φ̃, the likelihood ỹ ~ N(Φ̂ c, σ_n² I) (so a
//! row of weight w has noise σ_n²/w) and the prior c ~ N(0, λ⁻¹ I) give the posterior
//!
//!   c | data ~ N(m, Σ),   Σ = (Φ̂ᵀΦ̂/σ_n² + λ I)⁻¹ = (G̃/σ_n² + λ I)⁻¹,   m = Σ Φ̂ᵀỹ/σ_n² = (G̃ + λσ_n² I)⁻¹ b̃,
//!
//! i.e. m is the ridge solution with ridge strength λ_r = λ σ_n². λ is the **prior precision**
//! in standardised coefficient units (1/√λ is the prior standard deviation of a coefficient,
//! Hartree per unit standardised feature); this is the λ of `Model::lambda`, `FitOptions::lambda`
//! and the grid below. Parametrising by λ rather than by λ_r makes Σ exactly the brief's formula
//! and lets noise-free data fit without ridge bias (λ_r → 0 with σ_n → 0) while noisy data get
//! the ridge the evidence asks for. The solve is one Cholesky of G̃ + λ_r I (f64, P = 560 at four
//! species is 6e7 flop).
//!
//! # Predictive variance
//!
//! For a query with standardised features φ̃ the predictive distribution of the (unit-weight)
//! observation is N(φ̃ᵀm, σ²(φ̃)) with
//!
//!   σ²(φ̃) = σ_noise² + φ̃ᵀ Σ φ̃ = σ_noise² + ‖L_A⁻¹ φ̃‖²,   L_A L_Aᵀ = Σ⁻¹ = G̃/σ_n² + λ I,
//!
//! and `Model::sigma_inv_chol` stores L_A (= L/σ_n for the ridge factor L of G̃ + λ_r I). The
//! quadratic form is the leverage of the query in standardised feature space: it is small in
//! directions the data span and → ‖φ̃_⊥‖²/λ in directions they do not (C3). `Model::predict`
//! evaluates it per atom for the forces (the 3 gradient-feature vectors of each atom, see the
//! model docs for how the per-atom scalar is formed) and once for the total energy
//! (φ̃ = Σ_i φ̃_i); O(Q²) per vector, Task 9 adds the cadence.
//!
//! # Hyperparameters from the data: evidence and residuals (C2)
//!
//! σ_n is NOT the solver's path scatter (5e-6 Ha/Bohr): that is the reproducibility of the
//! teacher, not the residual a translation-invariant model can reach against it. The egg-box
//! force (ΣF ≈ 3e-4 Ha/Bohr, not representable) and SCF lag are the floor, ~60× higher; a
//! constant σ_n = 5e-6 would make every variance 3 600× overconfident and the fallback would
//! never fire. So both hyperparameters come from the data by type-II maximum likelihood
//! (Bishop PRML §3.5, MacKay 1992). The log marginal likelihood of the weighted rows is
//!
//!   ln p(ỹ | λ, σ_n²) = −(n/2) ln(2πσ_n²) + (Q/2) ln(λσ_n²) − ½ ln det(G̃ + λσ_n² I)
//!                       − RSS(m)/(2σ_n²) − λ‖m‖²/2 + ½ Σ_rows ln w,
//!
//! with n the number of rows (energy + force), RSS = ‖ỹ − Φ̂m‖² = yᵀWy − 2 mᵀb̃ + mᵀG̃m (from the
//! statistics; clamped at 0 against cancellation) and Q the number of columns. (Derivation:
//! Bishop 3.86 with α = λ, β = 1/σ_n², A = αI + βΦ̂ᵀΦ̂ = (G̃ + λσ_n² I)/σ_n², so ln det A =
//! ln det(G̃ + λσ_n² I) − Q ln σ_n² and the (Q/2) ln σ_n² terms cancel.) Stationarity in σ_n² at
//! fixed λ (Bishop 3.95, exact because ∇_m vanishes at m) is
//!
//!   σ_n² = RSS/(n − γ),   γ = tr(G̃ (G̃ + λσ_n² I)⁻¹) = Q − λσ_n² tr((G̃ + λσ_n² I)⁻¹),
//!
//! the residual sum of squares over the residual degrees of freedom, γ being the effective
//! number of parameters (the `dof_eff` of the brief). Because γ depends on λσ_n², the solve
//! iterates this fixed point to convergence for each λ (a handful of Choleskys; λσ_n² is usually
//! ≪ the eigenvalues of G̃ so it converges in 2–5 steps), then evaluates the evidence. λ is
//! chosen by maximising the profile evidence over the log grid 1e-8..1e2 in half decades,
//! extended in the same steps while the maximum sits on an edge (up to 1e-14 / 1e8), and
//! refined by golden-section search in ln λ between the grid neighbours of the best point.
//! `FitOptions::lambda` / `sigma_n` pin either hyperparameter and the other is still estimated.
//! σ_n² is floored at 1e-16·yᵀWy/n, the f64 resolution of a residual computed from the
//! statistics, so noise-free data give σ_n ≈ 1e-8·rms(y) rather than 0 or a failed Cholesky.
//!
//! **Per-kind noise floors.** The pooled σ_n is the noise of a unit-weight row, dominated by
//! the 3N force rows. The energy and force residuals of real DFT data differ by more than 10×
//! (per-atom energy ~1e-5 Ha at tol 1e-6 vs force ~3e-4 Ha/Bohr), so the aleatoric floor of the
//! predictive variance is kept per kind: with M = (G̃ + λ_r I)⁻¹ and the per-kind statistics
//! (sample weights only, no `force_weight`),
//!
//!   s_E² = RSS_E/(n_E − γ_E),  γ_E = tr(G̃_E M);   s_F² = RSS_F/(n_F − γ_F),  γ_F = w_F tr(G̃_F M),
//!
//! (γ_E + γ_F = γ), stored as `Model::sigma_e` (per energy row, i.e. per atom when
//! `energy_per_atom`) and `Model::sigma_f` (per force component). The epistemic part φ̃ᵀΣφ̃ uses
//! the pooled posterior Σ; when s_E ≪ σ_n this over-states the energy's epistemic variance
//! (conservative). A full two-noise model (β_E, β_F in the posterior precision) is the next
//! refinement if Task 4 finds the energy criterion too conservative; the statistics are already
//! kept per kind for it. If a kind has no residual degrees of freedom (n_κ − γ_κ < 1) its floor
//! falls back to the pooled value.
//!
//! # Tied columns (hetero pairs) and dead columns
//!
//! The stacked layout has Q = S·P columns, but for every hetero species pair {s, t} and radial
//! index n the two body-2 columns (centre s, neighbour t, n) and (centre t, neighbour s, n) are
//! the same function of the positions on every row (energy: Σ_{i∈s, j∈t} R_n f_c(r_ij) counted
//! from either end; forces: its gradient), so only the SUM of their coefficients is
//! identifiable and the Gram has an exact null direction per such pair — which no ridge of
//! λσ_n² ≈ 1e-24 (noise-free data) can resolve in f64. The solve therefore works on the Q′
//! independent columns of [`Basis::column_classes`] (one representative per class: G̃ and b̃
//! restricted to the representatives, the dropped duplicates being identical columns), the
//! prior and the evidence are on the Q′ sums, and the solution is expanded by an equal split in
//! raw coefficient space, c̃_{s,(t,n)} = c̃_{t,(s,n)} = θ̃/2 — a convention that never changes a
//! prediction (every query's feature vector is symmetric in the same way) or its variance (the
//! query vector is restricted to the same representatives, `Model::predict`). `Model::lambda`
//! and `Model::sigma_inv_chol` (Q′×Q′) are in this reduced space; `FitReport::n_features` is Q′.
//!
//! A column that is identically zero (a (centre, neighbour) species pair never seen, or a
//! feature that is zero on every row) has G̃_qq = 0, b̃_q = 0, hence m_q = 0, posterior variance
//! exactly 1/λ (the prior), and contributes exactly 0 to the evidence ((½) ln λ from the (Q′/2)
//! ln λ term cancels the (½) ln λ of the determinant) and to γ. No special casing is needed
//! while λ > 0.
//!
//! **Ridge floor.** The Gram is only known to ~ε·‖G̃‖, so a ridge strength below that cannot be
//! resolved by the factorisation (κ > 1/ε: the noise-free Morse fixture reached 2.7e16). The
//! factored ridge is therefore max(λσ_n², 4ε·max_k G̃_kk): inert for any noisy data (λσ_n² is
//! orders of magnitude larger), and for noise-free data or `lambda: Some(0.0)` it is the
//! minimum-norm least-squares regulariser (dead columns get coefficient 0; their posterior
//! variance is then σ_n²/floor, i.e. "no prior"). [`FitError::Singular`] is left for a Gram that
//! is entirely zero (every feature zero on every row, e.g. all atoms beyond r_cut) with λ = 0, or
//! a numerical failure of the factorisation. With λ = 0 the (Q′/2) ln λ term makes
//! `log_evidence` −∞ (an improper prior has no evidence).
//!
//! # Isolated-atom limit and the references
//!
//! `ref_energy[s]` is the teacher's isolated-atom energy per species (Z-only: computed by the
//! same solver at the same settings, never a table). Because mean = 0 and there is no
//! intercept, an atom with no neighbour inside r_cut has every feature 0 and energy exactly
//! ref[s]; two atoms beyond r_cut have E = ref[s_1] + ref[s_2] bitwise (the test pins it).
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
//! * `condition_estimate` is κ₂(G̃ + λ_r I) from 40 power iterations (largest eigenvalue) and 40
//!   inverse iterations through the Cholesky factor (smallest) — a lower bound that is usually
//!   tight to a few percent.

use crate::descriptor::{describe, Basis, DescribeError};
use crate::linalg::{
    cholesky_in_place, cholesky_inverse, cholesky_logdet, cholesky_solve, cholesky_trace_inverse,
    mat_vec,
};
use crate::model::{energy_and_forces, Model, Provenance};
use std::fmt;

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

/// Fit settings. `lambda` and `sigma_n` are `None` by default: estimated from the data (module
/// doc); `Some` pins them.
#[derive(Clone, Debug, PartialEq)]
pub struct FitOptions {
    /// Prior precision λ of the standardised coefficients (ridge strength λ·σ_n²).
    pub lambda: Option<f64>,
    /// Pooled noise level of a unit-weight row (Hartree for energy rows).
    pub sigma_n: Option<f64>,
    /// Relative weight of a force-component row against an energy row (default 1.0).
    pub force_weight: f64,
    /// Energy rows carry E/N and features/N (default true).
    pub energy_per_atom: bool,
}

impl Default for FitOptions {
    fn default() -> Self {
        FitOptions {
            lambda: None,
            sigma_n: None,
            force_weight: 1.0,
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
    /// Pooled unit-weight noise σ_n (what Σ is built with).
    pub sigma_n: f64,
    /// Energy-row noise floor s_E (per atom when `energy_per_atom`).
    pub sigma_e: f64,
    /// Force-component noise floor s_F.
    pub sigma_f: f64,
    /// Prior precision λ (module doc).
    pub lambda: f64,
    pub log_evidence: f64,
    /// Effective number of parameters γ.
    pub dof_eff: f64,
    pub n_rows: usize,
    pub n_samples: usize,
    /// Columns Q = S·P.
    pub n_features: usize,
    /// κ₂ estimate of the standardised ridge matrix G̃ + λσ_n² I.
    pub condition_estimate: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum FitError {
    NoSamples,
    /// No finite reference energy for this Z.
    MissingReference(u8),
    /// The ridge matrix is not positive definite: λ = 0 with an all-zero Gram (every feature zero
    /// on every row), or a numerical failure of the factorisation (module doc, "Ridge floor").
    Singular,
    Describe(DescribeError),
}

impl fmt::Display for FitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FitError::NoSamples => write!(f, "no samples to fit"),
            FitError::MissingReference(z) => write!(f, "no reference energy for Z = {z}"),
            FitError::Singular => write!(f, "ridge matrix is singular (λ = 0 with rank-deficient data)"),
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

/// Sufficient statistics of the weighted least-squares problem, raw (unstandardised, module
/// doc), kept per row kind so the noise floors can be separated. All fields are public for the
/// sidecar; `add` is the only sane way to fill them.
#[derive(Clone, Debug, PartialEq)]
pub struct Accumulator {
    pub basis: Basis,
    /// Columns Q = S·P.
    pub q: usize,
    /// Energy rows: ΦᵀWΦ (Q×Q row-major), ΦᵀWy, yᵀWy, row count, Σ ln w.
    pub gram_e: Vec<f64>,
    pub rhs_e: Vec<f64>,
    pub yy_e: f64,
    pub n_e: usize,
    pub sum_log_w_e: f64,
    /// Force rows, with sample weights only (`force_weight` is applied at solve time).
    pub gram_f: Vec<f64>,
    pub rhs_f: Vec<f64>,
    pub yy_f: f64,
    pub n_f: usize,
    pub sum_log_w_f: f64,
    /// Per feature k: Σ over energy rows and species blocks of φ_{E,(s,k)}² (for the scale).
    pub sumsq: Vec<f64>,
    pub n_samples: usize,
    /// `energy_per_atom` of the first `add`; later adds must agree.
    pub energy_per_atom: Option<bool>,
}

impl Accumulator {
    pub fn new(basis: &Basis) -> Accumulator {
        let q = basis.n_species() * basis.n_features();
        Accumulator {
            basis: basis.clone(),
            q,
            gram_e: vec![0.0; q * q],
            rhs_e: vec![0.0; q],
            yy_e: 0.0,
            n_e: 0,
            sum_log_w_e: 0.0,
            gram_f: vec![0.0; q * q],
            rhs_f: vec![0.0; q],
            yy_f: 0.0,
            n_f: 0,
            sum_log_w_f: 0.0,
            sumsq: vec![0.0; basis.n_features()],
            n_samples: 0,
            energy_per_atom: None,
        }
    }

    /// Rows accumulated so far (energy + force).
    pub fn n_rows(&self) -> usize {
        self.n_e + self.n_f
    }

    /// Add one sample's energy row and 3N force rows.
    ///
    /// # Panics
    /// If `basis` differs from the accumulator's, the sample's shapes disagree, or
    /// `opts.energy_per_atom` differs from the first call's.
    pub fn add(
        &mut self,
        sample: &Sample,
        basis: &Basis,
        ref_energy: &[f64],
        opts: &FitOptions,
    ) -> Result<(), DescribeError> {
        assert_eq!(*basis, self.basis, "Accumulator::add: basis differs from the accumulator's");
        assert!(sample.shape_ok().is_ok(), "Accumulator::add: sample shapes disagree");
        let per_atom = *self.energy_per_atom.get_or_insert(opts.energy_per_atom);
        assert_eq!(
            per_atom, opts.energy_per_atom,
            "Accumulator::add: energy_per_atom differs from the first sample's"
        );
        let s = basis.species_indices(&sample.z)?;
        let desc = describe(basis, &sample.pos, &sample.z)?;
        let p = basis.n_features();
        let q = self.q;
        let n = sample.pos.len();
        let w = sample.weight;

        // Energy row.
        let mut phi = vec![0.0; q];
        let mut y = sample.energy;
        for (i, d) in desc.iter().enumerate() {
            y -= ref_energy[s[i]];
            let block = &mut phi[s[i] * p..(s[i] + 1) * p];
            for k in 0..p {
                block[k] += d.b[k];
            }
        }
        if per_atom {
            let inv = 1.0 / n as f64;
            phi.iter_mut().for_each(|v| *v *= inv);
            y *= inv;
        }
        rank1_update(&mut self.gram_e, &mut self.rhs_e, q, &phi, y, w);
        self.yy_e += w * y * y;
        self.n_e += 1;
        self.sum_log_w_e += w.ln();
        for t in 0..basis.n_species() {
            for k in 0..p {
                let v = phi[t * p + k];
                self.sumsq[k] += v * v;
            }
        }

        // Force rows: φ_{F,(j,a)} block s = −Σ_{i: s_i = s} ∂B_i/∂r_{j,a}.
        let mut rows = vec![vec![0.0; q]; 3 * n];
        for (i, d) in desc.iter().enumerate() {
            let off = s[i] * p;
            for (j, g) in &d.grad {
                for k in 0..p {
                    rows[3 * j][off + k] -= g[k][0];
                    rows[3 * j + 1][off + k] -= g[k][1];
                    rows[3 * j + 2][off + k] -= g[k][2];
                }
            }
        }
        for j in 0..n {
            for a in 0..3 {
                let f = sample.forces[j][a];
                rank1_update(&mut self.gram_f, &mut self.rhs_f, q, &rows[3 * j + a], f, w);
                self.yy_f += w * f * f;
                self.n_f += 1;
                self.sum_log_w_f += w.ln();
            }
        }
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
        let q = self.q;
        let n_e = self.n_e as f64;
        let n_f = self.n_f as f64;
        let n_rows = self.n_e + self.n_f;
        let w_f = opts.force_weight;
        assert!(w_f > 0.0 && w_f.is_finite(), "FitOptions::force_weight must be positive");

        // Scale from the energy rows; D is per column through k = q mod P. The solve runs on
        // the Q′ independent columns (module doc, "Tied columns"): restrict the raw statistics
        // to the class representatives, then standardise.
        let scale: Vec<f64> = self
            .sumsq
            .iter()
            .map(|&ss| if ss > 0.0 { (ss / n_e).sqrt() } else { 1.0 })
            .collect();
        let classes = basis.column_classes();
        let rep: Vec<usize> = classes.iter().map(|c| c[0]).collect();
        let qr = rep.len();
        let d: Vec<f64> = rep.iter().map(|&c| 1.0 / scale[c % p]).collect();
        let full_q = q;
        let reduce = |g: &[f64]| -> Vec<f64> {
            let mut out = vec![0.0; qr * qr];
            for i in 0..qr {
                for j in 0..qr {
                    out[i * qr + j] = d[i] * g[rep[i] * full_q + rep[j]] * d[j];
                }
            }
            out
        };
        let g_e = reduce(&self.gram_e);
        let g_f = reduce(&self.gram_f);
        let mut gram = vec![0.0; qr * qr];
        for i in 0..qr * qr {
            gram[i] = g_e[i] + w_f * g_f[i];
        }
        let b_e: Vec<f64> = (0..qr).map(|i| d[i] * self.rhs_e[rep[i]]).collect();
        let b_f: Vec<f64> = (0..qr).map(|i| d[i] * self.rhs_f[rep[i]]).collect();
        let rhs: Vec<f64> = (0..qr).map(|i| b_e[i] + w_f * b_f[i]).collect();
        let yy = self.yy_e + w_f * self.yy_f;
        let sum_log_w = self.sum_log_w_e + self.sum_log_w_f + n_f * w_f.ln();
        let q = qr;

        let max_diag = (0..q).map(|i| gram[i * q + i]).fold(0.0, f64::max);
        let problem = Ridge {
            gram: &gram,
            rhs: &rhs,
            yy,
            q,
            n: n_rows as f64,
            sum_log_w,
            sigma2_floor: (1e-16 * yy / n_rows as f64).max(f64::MIN_POSITIVE),
            ridge_floor: 4.0 * f64::EPSILON * max_diag,
        };
        let (lambda, sigma2, sol) = problem.choose_hyperparameters(opts.lambda, opts.sigma_n.map(|s| s * s))?;
        let lambda_r = sol.lambda_r;
        let sigma_n = sigma2.sqrt();
        let log_evidence = problem.evidence(lambda, sigma2, &sol);

        // Per-kind residuals and degrees of freedom at the chosen point.
        let m_inv = cholesky_inverse(&sol.chol, q);
        let trace_prod = |g: &[f64]| -> f64 {
            let mut t = 0.0;
            for i in 0..q {
                for j in 0..q {
                    t += g[i * q + j] * m_inv[j * q + i];
                }
            }
            t
        };
        let gamma_e = trace_prod(&g_e);
        let gamma_f = w_f * trace_prod(&g_f);
        let rss_e = residual_ss(&g_e, &b_e, self.yy_e, q, &sol.m);
        let rss_f = residual_ss(&g_f, &b_f, self.yy_f, q, &sol.m);
        let kind_floor = |rss: f64, n_k: f64, gamma_k: f64, pooled: f64| -> f64 {
            let dof = n_k - gamma_k;
            if n_k > 0.0 && dof >= 1.0 {
                (rss / dof).max(problem.sigma2_floor).sqrt()
            } else {
                pooled.sqrt()
            }
        };
        let sigma_e = kind_floor(rss_e, n_e, gamma_e, sigma2);
        let sigma_f = kind_floor(rss_f, n_f, gamma_f, sigma2 / w_f);
        let rmse_energy = if n_e > 0.0 { (rss_e / n_e).sqrt() } else { 0.0 };
        let rmse_force = if n_f > 0.0 { (rss_f / n_f).sqrt() } else { 0.0 };

        // Posterior factor: Σ⁻¹ = (G̃ + λ_r I)/σ_n² = (L/σ_n)(L/σ_n)ᵀ.
        let inv_sigma = 1.0 / sigma_n;
        let sigma_inv_chol: Vec<f64> = sol.chol.iter().map(|v| v * inv_sigma).collect();

        // Expand the reduced solution: raw θ̃ = θ/scale_rep, split equally over the class, and
        // re-standardise each full column by its own feature scale (`Model::coeffs` contract).
        let mut full = vec![0.0; full_q];
        for (i, class) in classes.iter().enumerate() {
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
            sigma_n,
            sigma_e,
            sigma_f,
            energy_per_atom: self.energy_per_atom.unwrap_or(opts.energy_per_atom),
            lambda,
            sigma_inv_chol: Some(sigma_inv_chol),
            n_samples: self.n_samples,
            provenance: provenance.clone(),
        };
        let report = FitReport {
            rmse_energy_per_atom: rmse_energy,
            rmse_force,
            sigma_n,
            sigma_e,
            sigma_f,
            lambda,
            log_evidence,
            dof_eff: sol.gamma,
            n_rows,
            n_samples: self.n_samples,
            n_features: q,
            condition_estimate: condition_estimate(&gram, &sol.chol, q, lambda_r),
        };
        Ok((model, report))
    }
}

/// `ref_energy` must have one finite entry per species of the basis.
fn check_references(basis: &Basis, ref_energy: &[f64]) -> Result<(), FitError> {
    for (s, &z) in basis.species.iter().enumerate() {
        match ref_energy.get(s) {
            Some(r) if r.is_finite() => {}
            _ => return Err(FitError::MissingReference(z)),
        }
    }
    Ok(())
}

/// G += w v vᵀ, b += w y v over the support of v (the species blocks a cluster touches).
fn rank1_update(gram: &mut [f64], rhs: &mut [f64], q: usize, v: &[f64], y: f64, w: f64) {
    let support: Vec<usize> = (0..q).filter(|&i| v[i] != 0.0).collect();
    for &i in &support {
        let wi = w * v[i];
        rhs[i] += wi * y;
        let row = &mut gram[i * q..(i + 1) * q];
        for &j in &support {
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

/// One ridge solve at ridge strength λ_r: the factor of G̃ + λ_r I, the mean, RSS, γ, ln det.
struct Solved {
    chol: Vec<f64>,
    m: Vec<f64>,
    rss: f64,
    gamma: f64,
    logdet: f64,
    m_norm2: f64,
    /// The ridge strength actually factored (after the floor).
    lambda_r: f64,
}

/// The standardised, pooled least-squares problem (module doc).
struct Ridge<'a> {
    gram: &'a [f64],
    rhs: &'a [f64],
    yy: f64,
    q: usize,
    n: f64,
    sum_log_w: f64,
    sigma2_floor: f64,
    /// Smallest ridge strength the factorisation can resolve (4ε·max G̃_kk).
    ridge_floor: f64,
}

impl Ridge<'_> {
    fn solve(&self, lambda_r: f64) -> Result<Solved, FitError> {
        let q = self.q;
        // The Gram is known to ~ε·‖G̃‖ only; a ridge below 4ε·max G̃_kk cannot be resolved by
        // the factorisation (κ > 1/ε), so floor it there. Inert for noisy data (λσ_n² is orders
        // of magnitude larger); for noise-free data it is the minimum-norm regulariser.
        let lambda_r = lambda_r.max(self.ridge_floor);
        let mut chol = self.gram.to_vec();
        for i in 0..q {
            chol[i * q + i] += lambda_r;
        }
        cholesky_in_place(&mut chol, q).map_err(|_| FitError::Singular)?;
        let mut m = self.rhs.to_vec();
        cholesky_solve(&chol, q, &mut m);
        let rss = residual_ss(self.gram, self.rhs, self.yy, q, &m);
        let gamma = q as f64 - lambda_r * cholesky_trace_inverse(&chol, q);
        let logdet = cholesky_logdet(&chol, q);
        let m_norm2 = m.iter().map(|v| v * v).sum();
        Ok(Solved {
            chol,
            m,
            rss,
            gamma,
            logdet,
            m_norm2,
            lambda_r,
        })
    }

    /// Log marginal likelihood at (λ, σ_n²) given the solve at λ_r = λσ_n² (module doc).
    fn evidence(&self, lambda: f64, sigma2: f64, sol: &Solved) -> f64 {
        let q = self.q as f64;
        -0.5 * self.n * (2.0 * std::f64::consts::PI * sigma2).ln() + 0.5 * q * (lambda * sigma2).ln()
            - 0.5 * sol.logdet
            - 0.5 * sol.rss / sigma2
            - 0.5 * lambda * sol.m_norm2
            + 0.5 * self.sum_log_w
    }

    /// σ_n² at fixed λ by the fixed point σ² = RSS/(n − γ) (Bishop 3.95), starting from `start`.
    fn profile_sigma2(&self, lambda: f64, start: f64) -> Result<(f64, Solved), FitError> {
        let mut s2 = start.max(self.sigma2_floor);
        let mut sol = self.solve(lambda * s2)?;
        for _ in 0..100 {
            let dof = (self.n - sol.gamma).max(1.0);
            let next = (sol.rss / dof).max(self.sigma2_floor);
            let converged = (next - s2).abs() <= 1e-10 * next;
            s2 = next;
            sol = self.solve(lambda * s2)?;
            if converged {
                break;
            }
        }
        Ok((s2, sol))
    }

    /// Evidence at λ with σ_n² either pinned or profiled; returns (σ², solve, ln evidence).
    fn at(&self, lambda: f64, sigma2_pinned: Option<f64>, start: f64) -> Result<(f64, Solved, f64), FitError> {
        let (s2, sol) = match sigma2_pinned {
            Some(s2) => (s2, self.solve(lambda * s2)?),
            None => self.profile_sigma2(lambda, start)?,
        };
        let ev = self.evidence(lambda, s2, &sol);
        Ok((s2, sol, ev))
    }

    /// (λ, σ_n², solve): pinned values are used as given, the others maximise the evidence over
    /// the half-decade grid 1e-8..1e2 (extended while the maximum is on an edge, to 1e-14 /
    /// 1e8) followed by a golden-section refinement in ln λ between the grid neighbours.
    fn choose_hyperparameters(
        &self,
        lambda_pinned: Option<f64>,
        sigma2_pinned: Option<f64>,
    ) -> Result<(f64, f64, Solved), FitError> {
        let start = sigma2_pinned.unwrap_or(self.yy / self.n);
        if let Some(lambda) = lambda_pinned {
            assert!(lambda >= 0.0 && lambda.is_finite(), "FitOptions::lambda must be ≥ 0");
            let (s2, sol, _) = self.at(lambda, sigma2_pinned, start)?;
            return Ok((lambda, s2, sol));
        }
        let half_decade = 10f64.sqrt();
        let mut grid: Vec<(f64, f64, f64)> = Vec::new(); // (ln λ, σ², evidence)
        let mut s2_prev = start;
        let mut lambda = 1e-8;
        while lambda <= 1e2 * 1.0001 {
            let (s2, _, ev) = self.at(lambda, sigma2_pinned, s2_prev)?;
            grid.push((lambda.ln(), s2, ev));
            s2_prev = s2;
            lambda *= half_decade;
        }
        let best_index = |g: &[(f64, f64, f64)]| {
            g.iter()
                .enumerate()
                .max_by(|a, b| a.1 .2.partial_cmp(&b.1 .2).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap()
        };
        // Extend upward / downward while the maximum sits on an edge.
        loop {
            let i = best_index(&grid);
            if i == grid.len() - 1 && grid[i].0.exp() < 1e8 {
                let l = grid[i].0.exp() * half_decade;
                let (s2, _, ev) = self.at(l, sigma2_pinned, grid[i].1)?;
                grid.push((l.ln(), s2, ev));
            } else if i == 0 && grid[0].0.exp() > 1e-14 {
                let l = grid[0].0.exp() / half_decade;
                let (s2, _, ev) = self.at(l, sigma2_pinned, grid[0].1)?;
                grid.insert(0, (l.ln(), s2, ev));
            } else {
                break;
            }
        }
        let i = best_index(&grid);
        let (mut lo, mut hi) = (
            grid[i.saturating_sub(1)].0,
            grid[(i + 1).min(grid.len() - 1)].0,
        );
        let mut best = (grid[i].0, grid[i].1, grid[i].2);
        if hi > lo {
            // Golden-section refinement of the profile evidence in ln λ.
            let phi = 0.5 * (3.0 - 5f64.sqrt());
            let mut x1 = lo + phi * (hi - lo);
            let mut x2 = hi - phi * (hi - lo);
            let mut f1 = self.at(x1.exp(), sigma2_pinned, best.1)?;
            let mut f2 = self.at(x2.exp(), sigma2_pinned, best.1)?;
            for _ in 0..12 {
                if f1.2 > f2.2 {
                    hi = x2;
                    x2 = x1;
                    f2 = f1;
                    x1 = lo + phi * (hi - lo);
                    f1 = self.at(x1.exp(), sigma2_pinned, best.1)?;
                } else {
                    lo = x1;
                    x1 = x2;
                    f1 = f2;
                    x2 = hi - phi * (hi - lo);
                    f2 = self.at(x2.exp(), sigma2_pinned, best.1)?;
                }
            }
            for (x, f) in [(x1, &f1), (x2, &f2)] {
                if f.2 > best.2 {
                    best = (x, f.0, f.2);
                }
            }
        }
        let lambda = best.0.exp();
        let (s2, sol, _) = self.at(lambda, sigma2_pinned, best.1)?;
        Ok((lambda, s2, sol))
    }
}

/// κ₂(G̃ + λ_r I) by power iteration (largest eigenvalue) and inverse iteration through the
/// Cholesky factor (smallest), 40 steps each from a fixed start vector.
fn condition_estimate(gram: &[f64], chol: &[f64], q: usize, lambda_r: f64) -> f64 {
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
        mat_vec(gram, q, &v, &mut w);
        for i in 0..q {
            w[i] += lambda_r * v[i];
        }
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
            sigma_n: 0.0,
            sigma_e: 0.0,
            sigma_f: 0.0,
            energy_per_atom: true,
            lambda: 0.0,
            sigma_inv_chol: None,
            n_samples: 0,
            provenance: provenance(),
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
        assert!(report.sigma_n < 1e-7, "noise-free data: σ_n = {:e}", report.sigma_n);
        let s0 = &store.samples[0];
        let pred = model.predict(&s0.pos, &s0.z, true).unwrap();
        let e_std = pred.energy_std.expect("variance requested");
        let f_std = pred.force_std.expect("variance requested");
        assert!(e_std < 1e-6, "energy std {e_std:e} should be ≈ 0");
        assert!(f_std.iter().all(|s| *s < 1e-6), "force std {f_std:?} should be ≈ 0");
        assert!((pred.energy - s0.energy).abs() < 1e-11);
        assert_eq!(report.n_rows, store.samples.iter().map(|s| 1 + 3 * s.pos.len()).sum::<usize>());
        assert_eq!(report.n_features, 12, "Q′ = n_max·S(S+1)/2 independent columns");
        assert!(report.condition_estimate.is_finite() && report.condition_estimate >= 1.0);
    }

    #[test]
    fn fit_recovers_a_morse_dimer_to_1e6_with_body2() {
        let mut rng = StdRng::seed_from_u64(2);
        let store = morse_store(&mut rng, 1.0, 3.0, 60, None);
        let basis = h2_basis(12);
        let (model, report) = fit(&store, &basis, &[E_ATOM_H], &FitOptions::default()).unwrap();
        let rmse_e = total_energy_rmse(&model, &store);
        eprintln!(
            "morse: rmse E_total {rmse_e:.3e} Ha, E/atom {:.3e}, F {:.3e} Ha/Bohr, σ_n {:.3e}, λ {:.3e}, γ {:.2}, κ {:.3e}, ln ev {:.2}",
            report.rmse_energy_per_atom,
            report.rmse_force,
            report.sigma_n,
            report.lambda,
            report.dof_eff,
            report.condition_estimate,
            report.log_evidence
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
            "noise: σ_n {:.3e} s_E {:.3e} s_F {:.3e} λ {:.3e} γ {:.2} rmse_F {:.3e}",
            report.sigma_n, report.sigma_e, report.sigma_f, report.lambda, report.dof_eff, report.rmse_force
        );
        // Pooled: 6 noisy force rows per energy row → σ_n ≈ √(6/7)·σ_F.
        assert!(
            (report.sigma_n - sigma_f).abs() < 0.2 * sigma_f,
            "σ_n = {:e} vs injected {sigma_f:e}",
            report.sigma_n
        );
        assert!((report.sigma_f - sigma_f).abs() < 0.2 * sigma_f, "s_F = {:e}", report.sigma_f);
        assert!(report.sigma_e < report.sigma_f, "energies were noise-free: s_E {:e}", report.sigma_e);
        let s0 = &store.samples[10];
        let pred = model.predict(&s0.pos, &s0.z, true).unwrap();
        let f_std = pred.force_std.unwrap();
        for (j, s) in f_std.iter().enumerate() {
            assert!(*s >= 0.8 * sigma_f, "force std of atom {j} = {s:e} < 0.8·σ_F");
            assert!(*s < 3.0 * sigma_f, "force std of atom {j} = {s:e} is far above the noise");
        }
        // Pinning σ_n overrides the estimate; pinning λ too makes a plain ridge solve.
        let pinned = FitOptions { sigma_n: Some(1e-2), ..Default::default() };
        let (m2, r2) = fit(&store, &basis, &[E_ATOM_H], &pinned).unwrap();
        assert_eq!(r2.sigma_n, 1e-2);
        assert_eq!(m2.sigma_n, 1e-2);
        let both = FitOptions { sigma_n: Some(1e-2), lambda: Some(1e-3), ..Default::default() };
        let (_, r3) = fit(&store, &basis, &[E_ATOM_H], &both).unwrap();
        assert_eq!((r3.sigma_n, r3.lambda), (1e-2, 1e-3));
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
        let mut hits_e = 0;
        let mut hits_f = 0;
        let mut n_f = 0;
        let mut ss_e = 0.0;
        let mut z2 = 0.0;
        for s in &samples[150..] {
            let pred = model.predict(&s.pos, &s.z, true).unwrap();
            let std_e = pred.energy_std.unwrap();
            let resid = pred.energy - s.energy;
            ss_e += resid * resid;
            z2 += (resid / std_e).powi(2);
            if resid.abs() <= 2.0 * std_e {
                hits_e += 1;
            }
            let std_f = pred.force_std.unwrap();
            for j in 0..s.pos.len() {
                let d: f64 = (0..3).map(|a| (pred.forces[j][a] - s.forces[j][a]).powi(2)).sum::<f64>().sqrt();
                n_f += 1;
                if d <= 2.0 * std_f[j] {
                    hits_f += 1;
                }
            }
        }
        let cov_e = hits_e as f64 / 50.0;
        let cov_f = hits_f as f64 / n_f as f64;
        let mean_z2 = z2 / 50.0;
        eprintln!(
            "calibration: energy coverage {cov_e:.2}, force-norm coverage {cov_f:.2}, mean z² {mean_z2:.2}, held-out rmse_E {:.3e} (injected {se:e}), s_E {:.3e}, s_F {:.3e}, σ_n {:.3e}, λ {:.3e}, γ {:.1}",
            (ss_e / 50.0).sqrt(),
            report.sigma_e,
            report.sigma_f,
            report.sigma_n,
            report.lambda,
            report.dof_eff
        );
        assert!(cov_e >= 0.90, "energy 2σ coverage {cov_e}");
        // Calibrated: E[(resid/std)²] = 1; with 50 draws its std is √(2/50) = 0.2.
        assert!((0.5..1.5).contains(&mean_z2), "mean normalised squared residual {mean_z2}");
        // The recovered noise floors are the injected ones (within 20 %): s_E is per atom.
        assert!((3.0 * report.sigma_e - se).abs() < 0.2 * se, "3·s_E = {:e}", 3.0 * report.sigma_e);
        assert!((report.sigma_f - sf).abs() < 0.2 * sf, "s_F = {:e}", report.sigma_f);
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
            "ood: force std 1.7 → {f_in:.3e}, 3.0 → {f_out:.3e}, 5.0 → {f_far:.3e}; energy std {e_in:.3e} / {e_out:.3e} / {e_far:.3e}; λ {:.3e} σ_n {:.3e}",
            report.lambda, report.sigma_n
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
        assert_eq!((inc.sigma_n, inc.lambda), (batch.sigma_n, batch.lambda));
        assert_eq!(inc.sigma_inv_chol, batch.sigma_inv_chol);
        assert_eq!((ri.sigma_n, ri.lambda, ri.log_evidence), (rb.sigma_n, rb.lambda, rb.log_evidence));
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
        let li = Basis::new(&[3]);
        assert_eq!(
            fit(&store, &li, &[-7.0], &FitOptions::default()).unwrap_err(),
            FitError::Describe(DescribeError::UnknownSpecies(H))
        );
        // λ = 0 with species blocks that have no data (no O atom in an H₂ store): the ridge
        // floor 4ε·max G̃_kk gives the minimum-norm solution, zero in the dead blocks, and the
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
