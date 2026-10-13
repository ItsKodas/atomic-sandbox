//! The fitted linear model and its evaluation: total energy and analytic forces, and the
//! Bayesian predictive variance when the model carries the posterior factor from the fit.
//!
//! # Energy
//!
//! The coefficients live in standardised feature space (`fit` solves there so the ridge penalty
//! is scale-free): with per-feature `mean_k` and `scale_k`, the energy of atom i of species s_i is
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
//! energy only when offset_s = 0, which `fit` guarantees by never centring (mean = 0, scale-only
//! standardisation, no intercept — ruling S5); "two atoms beyond r_cut → Σ ref" then holds
//! exactly.
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
//!
//! # Body-3 forces by the adjoint of the atomic basis
//!
//! With the body-3 block on, atom i's energy also holds E_i^{(3)} = Σ_q c̃_{s_i,q} Σ_m
//! A_{c₁(q),lm}(i) A_{c₂(q),lm}(i) (`descriptor` docs). Instead of forming the P₃ feature
//! gradients, the hot path contracts the coefficients first,
//!
//!   W_{c,lm}(i) = ∂E_i/∂A_{c,lm} = Σ_{q: c₁ = c} c̃_q A_{c₂,lm} + Σ_{q: c₂ = c} c̃_q A_{c₁,lm}
//!
//! (both sums contribute when c₁ = c₂, giving 2c̃A), and then, since A_{(t,n),lm}(i) =
//! Σ_{j∈t} φ_{n,lm}(r_j − r_i),
//!
//!   ∂E_i/∂r_j = Σ_n Σ_{lm} W_{(s_j,n),lm}(i) ∇φ_{n,lm}(r_j − r_i)   (j ≠ i),   ∂E_i/∂r_i = −Σ_j ∂E_i/∂r_j,
//!
//! so F_j −= ∂E_i/∂r_j and F_i += the same vector: ΣF = 0 by construction again. Per centre
//! the cost is one `radial` + one `solid_harmonics` per neighbour (their ∇φ kept in a scratch
//! slot), O(P₃·(2l+1)) for the energy and W, and O(neighbours · n_max · (l_max+1)²) for the
//! forces — O(N · neighbours · features) overall with no per-pair allocation (the scratch is
//! sized once per call). The per-neighbour body-3 force is not central, so Σ r×F vanishes by
//! the rotation invariance of E_i (cancellation between neighbours, to rounding) rather than
//! pair by pair. The body-2 block keeps its pair sweep above.
//!
//! # Coincident atoms (policy)
//!
//! Two atoms at the same position have r_ij = 0 and no direction. The descriptor keeps the value
//! R_n(0) f_c(0) = 1 (so the pair counts as a neighbour and the energy stays finite:
//! E = Σ ref + Σ_n [c̃_i + c̃_j]_n) and the hot path adds exactly zero force to both atoms
//! (`pair_geometry` returns u = 0). The energy has a |r| cusp there; 0 is a valid subgradient,
//! it keeps the pair antisymmetry (ΣF = 0 exactly) and, unlike a NaN or an arbitrary direction,
//! it cannot inject momentum. Dynamics never reaches r = 0 on a repulsive surface; the policy is
//! for edits that drop one atom onto another, where the next step's forces are finite.
//!
//! # Predictive variance (`predict` with `with_variance`)
//!
//! With the posterior covariance Σ of the standardised coefficients on the Q′ independent
//! columns, held as the Cholesky factor L_A of Σ⁻¹ = G̃_E/σ_e² + G̃_F/σ_f² + λI (`fit` docs), a
//! linear functional φ̃ᵀc of the coefficients has epistemic variance φ̃ᵀΣφ̃ = ‖L_A⁻¹φ̃‖². The
//! outputs:
//!
//! * `energy_std`: the TOTAL energy, φ̃ = Σ_i φ̃_i with φ̃_i the standardised descriptor of atom i
//!   in its species block, plus the aleatoric floor of the energy rows: (N·σ_e)² when the fit
//!   used energies per atom (the row was E/N with noise σ_e), σ_e² otherwise.
//! * `force_std[j]`: √(Σ_a var_{j,a}) with var_{j,a} = σ_f² + g̃_{j,a}ᵀΣg̃_{j,a}, where g̃_{j,a}
//!   is the force-row feature vector −Σ_i ∂B̃_i/∂r_{j,a} — the expected squared norm of the
//!   force error on atom j, the quantity that compares to |F_j| in the fallback threshold; a
//!   per-component std lies between force_std/√3 and force_std.
//!
//! Cost: one `describe` plus 3N + 1 triangular solves of size Q′ (the independent columns,
//! `Basis::n_independent_columns`), O(Q′²) each.

use crate::descriptor::{describe, pair_geometry, AtomicBasisScratch, Basis, DescribeError};
use crate::basis::radial;
use crate::linalg::cholesky_quad_form;
use std::sync::OnceLock;

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
    /// Noise level of an energy row: per atom when `energy_per_atom`, else per sample (Hartree).
    pub sigma_e: f64,
    /// Noise level of a force component, Hartree/Bohr.
    pub sigma_f: f64,
    /// Whether the energy rows of the fit were E/N (decides the energy floor in `predict`).
    pub energy_per_atom: bool,
    /// Prior precision λ of the standardised coefficients (`fit` docs).
    pub lambda: f64,
    /// Cholesky factor (lower, row-major Q′×Q′) of the posterior precision
    /// Σ⁻¹ = G̃_E/σ_e² + G̃_F/σ_f² + λI over the Q′ independent columns of
    /// [`Basis::column_classes`] (representative order), for the predictive variance. `None`
    /// for a model without a fit; `predict` then reports no uncertainty.
    pub sigma_inv_chol: Option<Vec<f64>>,
    pub n_samples: usize,
    pub provenance: Provenance,
    /// Lazily cached representative columns of `basis.column_classes()` (one full column index
    /// per class); derived data, `OnceLock::new()` in a constructor. Replace the cell if `basis`
    /// is changed in place.
    pub representatives: OnceLock<Vec<usize>>,
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
    /// deviations (module doc), which are `None` while `sigma_inv_chol` is `None`.
    ///
    /// # Panics
    /// If `sigma_inv_chol` is `Some` with a length other than Q′² (`Basis::n_independent_columns`).
    pub fn predict(
        &self,
        pos: &[[f64; 3]],
        z: &[u8],
        with_variance: bool,
    ) -> Result<Prediction, DescribeError> {
        let (energy, forces) = energy_and_forces(self, pos, z)?;
        let (energy_std, force_std) = match (&self.sigma_inv_chol, with_variance) {
            (Some(l), true) => {
                let (e, f) = self.predictive_std(l, pos, z)?;
                (Some(e), Some(f))
            }
            _ => (None, None),
        };
        Ok(Prediction {
            energy,
            forces,
            energy_std,
            force_std,
        })
    }

    /// Standardised feature index map: 1/scale'_k per feature.
    fn inverse_scale(&self) -> Vec<f64> {
        self.scale
            .iter()
            .map(|&s| if s == 0.0 { 1.0 } else { 1.0 / s })
            .collect()
    }

    /// Representative full-column index of every class of `basis.column_classes()`, cached.
    pub fn representatives(&self) -> &[usize] {
        self.representatives
            .get_or_init(|| self.basis.column_classes().into_iter().map(|c| c[0]).collect())
    }

    /// Total-energy std and per-atom force std (module doc).
    fn predictive_std(
        &self,
        l: &[f64],
        pos: &[[f64; 3]],
        z: &[u8],
    ) -> Result<(f64, Vec<f64>), DescribeError> {
        let basis = &self.basis;
        let p = basis.n_features();
        let q = basis.n_species() * p;
        // The posterior lives on the independent columns (`fit` docs, "Tied columns"): restrict
        // every query vector to the class representatives before the quadratic form.
        let rep = self.representatives();
        let qr = rep.len();
        assert_eq!(l.len(), qr * qr, "Model.sigma_inv_chol: Q′×Q′ over Basis::column_classes()");
        let s = basis.species_indices(z)?;
        let desc = describe(basis, pos, z)?;
        let inv_scale = self.inverse_scale();
        let n = pos.len();
        let mut reduced = vec![0.0; qr];
        fn restrict(dst: &mut [f64], rep: &[usize], v: &[f64]) {
            for (r, &c) in dst.iter_mut().zip(rep) {
                *r = v[c];
            }
        }

        // Energy: φ̃ = Σ_i φ̃_i.
        let mut phi = vec![0.0; q];
        for (i, d) in desc.iter().enumerate() {
            let block = &mut phi[s[i] * p..(s[i] + 1) * p];
            for k in 0..p {
                block[k] += d.b[k] * inv_scale[k];
            }
        }
        restrict(&mut reduced, rep, &phi);
        let epistemic_e = cholesky_quad_form(l, qr, &mut reduced);
        let floor_e = if self.energy_per_atom {
            (n as f64 * self.sigma_e).powi(2)
        } else {
            self.sigma_e.powi(2)
        };
        let energy_std = (floor_e + epistemic_e).sqrt();

        // Forces: g̃_{j,a} = −Σ_i ∂B̃_i/∂r_{j,a}, gathered from every centre's gradient list.
        let mut rows = vec![vec![0.0; q]; 3 * n];
        for (i, d) in desc.iter().enumerate() {
            let off = s[i] * p;
            for (j, g) in &d.grad {
                for k in 0..p {
                    let gk = g[k];
                    let sc = inv_scale[k];
                    rows[3 * j][off + k] -= gk[0] * sc;
                    rows[3 * j + 1][off + k] -= gk[1] * sc;
                    rows[3 * j + 2][off + k] -= gk[2] * sc;
                }
            }
        }
        let mut force_std = Vec::with_capacity(n);
        for j in 0..n {
            let mut var = 3.0 * self.sigma_f * self.sigma_f;
            for a in 0..3 {
                restrict(&mut reduced, rep, &rows[3 * j + a]);
                var += cholesky_quad_form(l, qr, &mut reduced);
            }
            force_std.push(var.sqrt());
        }
        Ok((energy_std, force_std))
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
    if basis.body3 {
        body3_energy_and_forces(basis, &c_tilde, &s, pos, &mut energy, &mut forces);
    }
    Ok((energy.sum(), forces))
}

/// The body-3 block of the hot path (module doc, "Body-3 forces by the adjoint"): adds every
/// atom's E_i^{(3)} to `energy` and the pair-antisymmetric force contributions to `forces`.
fn body3_energy_and_forces(
    basis: &Basis,
    c_tilde: &[Vec<f64>],
    s: &[usize],
    pos: &[[f64; 3]],
    energy: &mut Neumaier,
    forces: &mut [[f64; 3]],
) {
    let n = pos.len();
    let n_max = basis.n_max;
    let p2 = basis.n_body2();
    let terms = basis.body3_terms();
    let mut sc = AtomicBasisScratch::new(basis, n);
    let nlm = sc.nlm;
    let mut w = vec![0.0; basis.n_body2() * nlm];
    for i in 0..n {
        sc.clear();
        for j in 0..n {
            if j == i {
                continue;
            }
            let Some((r, u)) = pair_geometry(pos[i], pos[j], basis.r_cut) else {
                continue;
            };
            let d = [pos[j][0] - pos[i][0], pos[j][1] - pos[i][1], pos[j][2] - pos[i][2]];
            sc.add_neighbour(basis, j, s[j], d, r, u);
        }
        if sc.neighbours.is_empty() {
            continue;
        }
        // Energy and the adjoint W = ∂E_i/∂A.
        let ci = &c_tilde[s[i]][p2..];
        w.iter_mut().for_each(|v| *v = 0.0);
        let mut e3 = Neumaier::default();
        for (q, t) in terms.iter().enumerate() {
            let c = ci[q];
            if c == 0.0 {
                continue;
            }
            let (c1, c2) = (basis.body2_index(t.t1, t.n1), basis.body2_index(t.t2, t.n2));
            let range = AtomicBasisScratch::l_range(t.l);
            let mut dot = 0.0;
            for lm in range.clone() {
                let (a1, a2) = (sc.a[c1 * nlm + lm], sc.a[c2 * nlm + lm]);
                dot += a1 * a2;
                w[c1 * nlm + lm] += c * a2;
                w[c2 * nlm + lm] += c * a1;
            }
            e3.add(c * dot);
        }
        energy.add(e3.sum());
        // Forces: ∂E_i/∂r_j = Σ_{n,lm} W_{(s_j,n),lm} ∇φ_{n,lm}(d_ij).
        for (slot, &j) in sc.neighbours.iter().enumerate() {
            let mut g = [0.0; 3];
            for nn in 0..n_max {
                let wrow = &w[(s[j] * n_max + nn) * nlm..(s[j] * n_max + nn + 1) * nlm];
                let dp = sc.dphi(slot, nn);
                for lm in 0..nlm {
                    let wv = wrow[lm];
                    g[0] += wv * dp[lm][0];
                    g[1] += wv * dp[lm][1];
                    g[2] += wv * dp[lm][2];
                }
            }
            for a in 0..3 {
                forces[j][a] -= g[a];
                forces[i][a] += g[a];
            }
        }
    }
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
        let mut basis = Basis::new(&[1, 8]);
        basis.body3 = false; // body-2 regression fixture (the default is body-3)
        let p = basis.n_features();
        Model {
            basis,
            coeffs: vec![vec![0.5; p], vec![-0.25; p]],
            mean: vec![2.0; p],
            scale: vec![4.0; p],
            ref_energy: vec![-0.5, -75.0],
            sigma_e: 0.0,
            sigma_f: 0.0,
            energy_per_atom: true,
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
            representatives: OnceLock::new(),
        }
    }

    #[test]
    fn coincident_atoms_in_the_hot_path_give_a_finite_energy_and_zero_force() {
        let mut m = tiny_model();
        m.mean = vec![0.0; m.basis.n_features()];
        let n_max = m.basis.n_max;
        let pos = [[0.3, -0.1, 2.0], [0.3, -0.1, 2.0], [0.3, -0.1, 2.0 + 1.5]];
        let z = [1, 8, 1];
        let (e, f) = energy_and_forces(&m, &pos, &z).unwrap();
        assert!(e.is_finite());
        // The coincident pair (H at 0, O at 1) contributes Σ_n [c̃_H,(O,n) + c̃_O,(H,n)]·1 (R_n(0) f_c(0) = 1).
        let (ct, _) = m.effective_coefficients();
        let pair_01: f64 = (0..n_max)
            .map(|nn| ct[0][m.basis.body2_index(1, nn)] + ct[1][m.basis.body2_index(0, nn)])
            .sum();
        // Remove atom 2's pairs by evaluating the two-atom system separately.
        let (e01, f01) = energy_and_forces(&m, &pos[..2], &z[..2]).unwrap();
        assert!((e01 - (m.ref_energy[0] + m.ref_energy[1] + pair_01)).abs() < 1e-12);
        assert_eq!(f01, vec![[0.0; 3]; 2], "no direction at r = 0 → exactly zero force");
        // In the three-atom system the coincident pair still adds no force; atoms 0 and 1 feel
        // only atom 2, along z, and momentum is conserved.
        assert_eq!((f[0][0], f[0][1], f[1][0], f[1][1]), (0.0, 0.0, 0.0, 0.0));
        let sum: f64 = f.iter().map(|v| v[2]).sum();
        assert!(sum.abs() < 1e-15);
        assert!(f[2][2].abs() > 0.0);
        // Variance path with a coincident pair stays finite too (identity posterior factor on
        // the Q′ independent columns).
        let q = m.basis.n_independent_columns();
        let mut l = vec![0.0; q * q];
        for i in 0..q {
            l[i * q + i] = 1.0;
        }
        m.sigma_inv_chol = Some(l);
        let pred = m.predict(&pos, &z, true).unwrap();
        assert!(pred.energy_std.unwrap().is_finite());
        assert!(pred.force_std.unwrap().iter().all(|s| s.is_finite()));
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
