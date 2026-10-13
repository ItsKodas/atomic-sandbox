//! The fitted linear model and its evaluation: total energy and analytic forces (this task), the
//! predictive variance (Task 3 fills `sigma_inv_chol` and the `*_std` outputs).
//!
//! # Energy
//!
//! The coefficients live in standardised feature space (Task 3 fits there so the ridge penalty is
//! scale-free): with per-feature `mean_k` and `scale_k`, the energy of atom i of species s_i is
//!
//!   E_i = ref_energy[s_i] + Σ_k c_{s_i,k} (B_{i,k} − mean_k)/scale_k,
//!
//! and a feature with `scale_k == 0` (constant over the training set) is divided by 1 instead.
//! Expanding gives the form the hot path evaluates,
//!
//!   E_i = ref_energy[s_i] + offset_{s_i} + Σ_k c̃_{s_i,k} B_{i,k},
//!   c̃_{s,k} = c_{s,k}/scale'_k,   offset_s = −Σ_k c̃_{s,k} mean_k,
//!
//! so standardisation costs one S×P pass per call and nothing per pair. Note the isolated-atom
//! limit (no neighbour inside r_cut, B = 0) is ref_energy[s] + offset_s: it equals the reference
//! energy only when the fit leaves offset_s = 0 (mean = 0, or a fit that pins it), which is Task
//! 3's responsibility; the brief's "two atoms beyond r_cut → Σ ref" holds for mean = 0.
//!
//! # Forces, pair-symmetric
//!
//! F_j = −∂E/∂r_j = −Σ_i Σ_k c̃_{s_i,k} ∂B_{i,k}/∂r_j. For body-2 features every B_{i,k} is a sum
//! over pairs, so E = Σ_i (ref + offset) + Σ_{pairs i<j} e_ij(r_ij) with
//!
//!   e_ij(r) = Σ_n [c̃_{s_i,(s_j,n)} + c̃_{s_j,(s_i,n)}] R_n(r) f_c(r)
//!
//! (the pair contributes to i's descriptor on j's species channel and to j's on i's). Its
//! derivative a_ij = de_ij/dr uses the same bracket with g_n = d[R_n f_c]/dr, and with
//! u_ij = (r_j − r_i)/r the chain rule gives F_j −= a_ij u_ij, F_i += a_ij u_ij. Each pair is
//! visited once and the same vector is added to one atom and subtracted from the other, so
//! ΣF = 0 holds to rounding by construction (translation invariance of the descriptors), and
//! Σ r×F = Σ_pairs a_ij (r_i − r_j) × u_ij = −Σ a_ij r_ij u_ij × u_ij = 0 (rotation invariance:
//! every pair force is central). No per-atom gradient list is built; the per-pair work is one
//! `radial` call plus 2·n_max fused multiply-adds, and the (s_i, s_j) coefficient blocks are
//! contiguous slices because features are ordered (neighbour species, n).
//!
//! The energy is accumulated with Neumaier (compensated) summation (CLAUDE.md rule 4: energy
//! reductions are compensated); forces are plain sums of ≤ N terms per atom.

use crate::descriptor::{pair_geometry, Basis, DescribeError};
use crate::basis::radial;

/// Settings the training data were produced with; a model is only valid for a scene solved with
/// the same ones (the Kohn-Sham surface depends on grid, σ and tolerance).
#[derive(Clone, Debug, PartialEq)]
pub struct Provenance {
    pub grid_n: usize,
    pub h: f64,
    pub sigma: f64,
    pub tol: f64,
    pub charge: i32,
    pub multiplicity: Option<u32>,
    pub teacher: String,
}

/// A fitted linear model. All fields are public: `fit` (Task 3) builds it, the sidecar (Task 5)
/// serialises it.
#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub basis: Basis,
    /// `coeffs[centre species][feature]`, in STANDARDISED feature space.
    pub coeffs: Vec<Vec<f64>>,
    /// Per-feature mean of the standardisation (shared by all centre species).
    pub mean: Vec<f64>,
    /// Per-feature scale of the standardisation; 0 marks a constant feature (divided by 1).
    pub scale: Vec<f64>,
    /// Isolated-atom energy per species index, Hartree.
    pub ref_energy: Vec<f64>,
    /// Noise level of the fit, Hartree (per energy row).
    pub sigma_n: f64,
    /// Ridge strength.
    pub lambda: f64,
    /// Cholesky factor (lower, row-major P×P) of ΦᵀΦ/σ_n² + λI, for the predictive variance.
    /// `None` until Task 3 fits it; `predict` then reports no uncertainty.
    pub sigma_inv_chol: Option<Vec<f64>>,
    pub n_samples: usize,
    pub provenance: Provenance,
}

/// Energy (Hartree, total including Σ ref_energy) and forces (Hartree/Bohr), with the
/// predictive standard deviations when the model carries a covariance factor.
#[derive(Clone, Debug, PartialEq)]
pub struct Prediction {
    pub energy: f64,
    pub forces: Vec<[f64; 3]>,
    pub energy_std: Option<f64>,
    /// Per atom, Hartree/Bohr.
    pub force_std: Option<Vec<f64>>,
}

impl Model {
    /// Total energy and forces; `with_variance` additionally asks for the predictive standard
    /// deviations, which are `None` while `sigma_inv_chol` is `None` (this task).
    pub fn predict(
        &self,
        pos: &[[f64; 3]],
        z: &[u8],
        with_variance: bool,
    ) -> Result<Prediction, DescribeError> {
        let (energy, forces) = energy_and_forces(self, pos, z)?;
        // Task 3: when `with_variance && self.sigma_inv_chol.is_some()`, evaluate
        // σ²(x) = σ_n² + φᵀ Σ φ per atom (energy) and through the gradient features (forces).
        let _ = with_variance;
        Ok(Prediction {
            energy,
            forces,
            energy_std: None,
            force_std: None,
        })
    }

    /// Effective coefficients c̃_{s,k} = c_{s,k}/scale'_k and per-species offsets
    /// −Σ_k c̃_{s,k} mean_k (see the module doc), so E_i = ref + offset + c̃·B.
    pub fn effective_coefficients(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        self.check_shapes();
        let p = self.basis.n_features();
        let mut c_tilde = Vec::with_capacity(self.coeffs.len());
        let mut offset = Vec::with_capacity(self.coeffs.len());
        for row in &self.coeffs {
            let mut ct = Vec::with_capacity(p);
            let mut off = Neumaier::default();
            for (k, &c_k) in row.iter().enumerate() {
                let scale = if self.scale[k] == 0.0 { 1.0 } else { self.scale[k] };
                let c = c_k / scale;
                ct.push(c);
                off.add(-c * self.mean[k]);
            }
            c_tilde.push(ct);
            offset.push(off.sum());
        }
        (c_tilde, offset)
    }

    /// Shape consistency of the public fields against the basis.
    ///
    /// # Panics
    /// With a message naming the field, if any length disagrees with the basis.
    pub fn check_shapes(&self) {
        let s = self.basis.n_species();
        let p = self.basis.n_features();
        assert_eq!(self.coeffs.len(), s, "Model.coeffs: one row per species");
        for (i, row) in self.coeffs.iter().enumerate() {
            assert_eq!(row.len(), p, "Model.coeffs[{i}]: one entry per feature");
        }
        assert_eq!(self.mean.len(), p, "Model.mean: one entry per feature");
        assert_eq!(self.scale.len(), p, "Model.scale: one entry per feature");
        assert_eq!(self.ref_energy.len(), s, "Model.ref_energy: one entry per species");
    }
}

/// The per-step hot path: total energy and forces without materialising per-atom descriptor
/// gradients (module doc). O(N²) pair sweep, `radial` once per pair inside r_cut.
///
/// # Panics
/// If `pos.len() != z.len()`, or the model's field lengths disagree with its basis.
pub fn energy_and_forces(
    model: &Model,
    pos: &[[f64; 3]],
    z: &[u8],
) -> Result<(f64, Vec<[f64; 3]>), DescribeError> {
    assert_eq!(pos.len(), z.len(), "energy_and_forces: pos and z lengths differ");
    let basis = &model.basis;
    let s = basis.species_indices(z)?;
    let (c_tilde, offset) = model.effective_coefficients();
    let n_max = basis.n_max;
    let n = pos.len();

    let mut energy = Neumaier::default();
    for &si in &s {
        energy.add(model.ref_energy[si] + offset[si]);
    }
    let mut forces = vec![[0.0; 3]; n];
    let mut rn = vec![0.0; n_max];
    let mut drn = vec![0.0; n_max];
    for i in 0..n {
        for j in i + 1..n {
            let Some((r, u)) = pair_geometry(pos[i], pos[j], basis.r_cut) else {
                continue;
            };
            radial(n_max, r, basis.r_cut, &mut rn, &mut drn);
            let ci = &c_tilde[s[i]][basis.body2_index(s[j], 0)..basis.body2_index(s[j], n_max)];
            let cj = &c_tilde[s[j]][basis.body2_index(s[i], 0)..basis.body2_index(s[i], n_max)];
            let mut e_pair = 0.0;
            let mut a = 0.0;
            for nn in 0..n_max {
                let c = ci[nn] + cj[nn];
                e_pair += c * rn[nn];
                a += c * drn[nn];
            }
            energy.add(e_pair);
            let f = [a * u[0], a * u[1], a * u[2]];
            for k in 0..3 {
                forces[i][k] += f[k];
                forces[j][k] -= f[k];
            }
        }
    }
    Ok((energy.sum(), forces))
}

/// Neumaier compensated sum (Kahan with the improved correction for |term| > |sum|).
#[derive(Default)]
struct Neumaier {
    sum: f64,
    comp: f64,
}

impl Neumaier {
    #[inline]
    fn add(&mut self, x: f64) {
        let t = self.sum + x;
        if self.sum.abs() >= x.abs() {
            self.comp += (self.sum - t) + x;
        } else {
            self.comp += (x - t) + self.sum;
        }
        self.sum = t;
    }

    #[inline]
    fn sum(&self) -> f64 {
        self.sum + self.comp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neumaier_recovers_a_cancelling_sum_that_plain_addition_loses() {
        let mut n = Neumaier::default();
        let terms = [1.0, 1e100, 1.0, -1e100];
        let plain: f64 = terms.iter().sum();
        for t in terms {
            n.add(t);
        }
        assert_eq!(plain, 0.0);
        assert_eq!(n.sum(), 2.0);
    }

    fn tiny_model() -> Model {
        let basis = Basis::new(&[1, 8]);
        let p = basis.n_features();
        Model {
            basis,
            coeffs: vec![vec![0.5; p], vec![-0.25; p]],
            mean: vec![2.0; p],
            scale: vec![4.0; p],
            ref_energy: vec![-0.5, -75.0],
            sigma_n: 0.0,
            lambda: 0.0,
            sigma_inv_chol: None,
            n_samples: 0,
            provenance: Provenance {
                grid_n: 1,
                h: 1.0,
                sigma: 1.0,
                tol: 1.0,
                charge: 0,
                multiplicity: None,
                teacher: String::new(),
            },
        }
    }

    #[test]
    fn effective_coefficients_fold_the_standardisation() {
        let mut m = tiny_model();
        m.scale[3] = 0.0; // constant feature: divided by 1
        let (ct, off) = m.effective_coefficients();
        assert_eq!(ct[0][0], 0.125);
        assert_eq!(ct[0][3], 0.5);
        assert_eq!(ct[1][0], -0.0625);
        // offset_s = −Σ_k c̃_{s,k}·mean_k = −(15·0.125 + 0.5)·2 for s = 0.
        assert!((off[0] + 2.0 * (15.0 * 0.125 + 0.5)).abs() < 1e-15);
        assert!((off[1] - 2.0 * (15.0 * 0.0625 + 0.25)).abs() < 1e-15);
    }

    #[test]
    #[should_panic(expected = "Model.mean: one entry per feature")]
    fn check_shapes_names_the_offending_field() {
        let mut m = tiny_model();
        m.mean.pop();
        m.check_shapes();
    }
}
