//! The unspanned directions of a fit: which standardised feature directions no training row
//! ever had a component along, and how far a query's rows reach into them (M6 Task 11; the
//! provider's rule 5d, "new environment").
//!
//! # Why the posterior is not enough
//!
//! The fit's posterior precision is Σ⁻¹ = G̃_E/σ_e² + G̃_F/σ_f² + λI over the Q′ independent
//! standardised columns (`fit` docs). Along a direction u with uᵀ(G̃_E + G̃_F)u = Σ_rows (u·row)²
//! = 0 no training row — no energy row, no force row — has a component: the data never spanned
//! u. There the posterior is the prior: the coefficient along u is pinned to 0 by the ridge and
//! its variance is 1/λ, so a query whose rows reach a distance ‖φ_⊥‖ into such directions gets
//! an epistemic variance ‖φ_⊥‖²/λ — with λ chosen by the evidence (6.75 for the H₂ pass), small
//! whenever the reach is modest, while the answer along u is a guess at zero. A feature column
//! that was zero in every training row is the axis-aligned case; the general one is a linear
//! dependency the training geometries obey and a query breaks. The H₂-only model is the user's
//! case: every body-3 feature of an atom with one neighbour is T_{n₁}T_{n₂} f_c² (r/r_c)^{2l} of
//! that one distance — no column vanishes, but the 21 body-3 columns span only a handful of
//! functions of r (polynomials of degree ≤ 7 times f_c², overlapping between the l), so the
//! dimer data leave 17 of the 29 directions at rounding level (measured, below), and an atom
//! with two neighbours — cross terms A_{c₁,lm}(j)A_{c₂,lm}(k) between them — has components
//! there. Its posterior std looked fine while its forces were off by 4–22× their bound (the
//! Task 11 repro).
//!
//! The extrapolation grade's infinite case (`grade`, rule 5e) is the per-atom analogue of this
//! test, not the same test: this one checks a query's design rows (energy and force rows), that
//! one each atom's descriptor against the training atoms' — a query's force rows can leave the
//! design span while every atom stays inside the per-atom span. Both are kept.
//!
//! # Zero to rounding, derived
//!
//! G̃ = D (G_E + G_F) D, D the fit's standardisation (1/scale per representative column), the
//! accumulator's own sums. Its entries carry the rounding of their accumulation and the
//! eigensolver adds O(ε)·‖G̃‖ (`linalg::symmetric_eigen`); both are bounded by Q′·ε·tr(G̃) in
//! the Frobenius norm (each entry's error is O(ε) of the products it sums, √(G_ii G_jj) ≤ tr by
//! Cauchy–Schwarz, Q′ such entries per row), so an eigenvalue
//!
//!   λ_k ≤ τ = Q′ · ε · tr(G̃),   ε = `f64::EPSILON`,
//!
//! is zero to rounding and its eigenvector is an unspanned direction. The descriptors themselves
//! are exact to a few ε relative, so a training row's own null component is O(ε)‖row‖ and its
//! null energy O(ε²)‖row‖² ≪ τ: the floor is the Gram's, not the descriptors'.
//!
//! **The query test.** All training rows together put Σ_null λ_k ≤ N_null·τ of energy into the
//! null space, so no single training row carries more than that. A query row v (built exactly as
//! the accumulator builds training rows: the energy row Σ_i φ_i (÷ N when the fit used energies
//! per atom) and the 3N force rows −Σ_i ∂φ_i/∂r_{j,a}, standardised by D) with
//!
//!   ν(v) = Σ_null (u_k·v)² / (N_null·τ) > 1
//!
//! reaches farther into the unspanned directions than any training row could to rounding: a new
//! environment. No tuned constant: ε, Q′ and the Gram's trace.
//!
//! **The margin as the store grows** (physics review, round 3, I2). τ assumes the Gram's
//! rounding stays at the eigensolver's level, which needs the accumulation compensated
//! (`fit::Accumulator`, Neumaier; CLAUDE.md rule 4). Measured on Morse dimer stores
//! (`tests/novelty.rs`), max |λ_null|/tr against τ/tr = 6.44e-15:
//!
//! | dimers | naive sum | margin | compensated | margin |
//! |---|---|---|---|---|
//! | 300 | 4.2e-16 | 15× | 1.7e-16 | 37× |
//! | 3 000 | 1.0e-15 | 6× | 1.8e-16 | 35× |
//! | 30 000 | 2.1e-15 | 3× (and the fit `Singular`) | 2.0e-16 | 32× |
//!
//! Compensated, the floor does not grow with the row count, so τ = Q′·ε·tr needs no √N_rows
//! factor; the smallest spanned eigenvalue stays at 4.9–5.2e-13·tr (76–80× above τ).
//!
//! Measured on a 300-sample Morse
//! dimer fit (`novelty_flags_a_third_atom_and_not_a_dimer`): eigenvalues |λ|/tr ≤ 3.1e-16 for 17
//! directions, the next 5.2e-13 — a gap of two orders above τ/tr = 6.4e-15; dimer queries
//! ν ≤ 3.3e-6, H₂ + H with the third atom at 2, 3, 5 Bohr ν ≥ 5.6e8.

use crate::descriptor::{describe, Basis, DescribeError};
use crate::fit::Accumulator;
use crate::linalg::symmetric_eigen;

/// The unspanned directions of a fit and the rounding threshold (module doc).
#[derive(Clone, Debug, PartialEq)]
pub struct NullSpace {
    pub basis: Basis,
    /// Full column q = s·P + k → reduced column, `None` for a tied duplicate (as the accumulator).
    pub col_index: Vec<Option<usize>>,
    /// Per reduced column: the standardisation 1/scale of its representative.
    pub d: Vec<f64>,
    /// Unit eigenvectors (length Q′) of the eigenvalues ≤ `tau`.
    pub vectors: Vec<Vec<f64>>,
    /// τ = Q′·ε·tr(G̃).
    pub tau: f64,
    /// Whether the energy rows were per atom.
    pub energy_per_atom: bool,
}

impl NullSpace {
    /// The unspanned directions of `acc`'s training rows under the standardisation `scale` (the
    /// fitted model's per-feature scale; 0 means 1, as in the fit).
    pub fn of(acc: &Accumulator, scale: &[f64]) -> NullSpace {
        let basis = acc.basis.clone();
        let p = basis.n_features();
        let classes = basis.column_classes();
        let q = classes.len();
        let mut col_index = vec![None; basis.n_species() * p];
        for (c, class) in classes.iter().enumerate() {
            col_index[class[0]] = Some(c);
        }
        let d: Vec<f64> = classes.iter().map(|c| { let s = scale[c[0] % p]; if s > 0.0 { 1.0 / s } else { 1.0 } }).collect();
        let sums = acc.sums();
        let mut g = vec![0.0; q * q];
        for i in 0..q {
            for j in 0..q {
                g[i * q + j] = d[i] * (sums.gram_e[i * q + j] + sums.gram_f[i * q + j]) * d[j];
            }
        }
        let trace: f64 = (0..q).map(|i| g[i * q + i]).sum();
        let tau = q as f64 * f64::EPSILON * trace;
        let (ev, v) = symmetric_eigen(&g, q);
        let vectors = (0..q).filter(|&i| ev[i] <= tau).map(|i| (0..q).map(|k| v[k * q + i]).collect()).collect();
        NullSpace { basis, col_index, d, vectors, tau, energy_per_atom: acc.energy_per_atom.unwrap_or(true) }
    }

    /// Number of unspanned directions.
    pub fn dimension(&self) -> usize {
        self.vectors.len()
    }

    /// ν of one standardised row (module doc).
    fn nu(&self, row: &[f64]) -> f64 {
        let e: f64 = self.vectors.iter().map(|u| u.iter().zip(row).map(|(a, b)| a * b).sum::<f64>().powi(2)).sum();
        e / (self.dimension() as f64 * self.tau)
    }

    /// The largest ν over the query's energy row and force rows, and where it was: `None` for the
    /// energy row, `Some(j)` for atom j's force rows. (0, None) when nothing is unspanned.
    pub fn novelty(&self, pos: &[[f64; 3]], z: &[u8]) -> Result<(f64, Option<usize>), DescribeError> {
        if self.vectors.is_empty() || !(self.tau > 0.0) {
            return Ok((0.0, None));
        }
        let basis = &self.basis;
        let s = basis.species_indices(z)?;
        let desc = describe(basis, pos, z)?;
        let p = basis.n_features();
        let q = self.d.len();
        let n = pos.len();
        let mut e_row = vec![0.0; q];
        let mut f_rows = vec![0.0; 3 * n * q];
        for (i, di) in desc.iter().enumerate() {
            let off = s[i] * p;
            for k in 0..p {
                if let Some(c) = self.col_index[off + k] {
                    e_row[c] += di.b[k];
                }
            }
            for (j, g) in &di.grad {
                for k in 0..p {
                    if let Some(c) = self.col_index[off + k] {
                        for a in 0..3 {
                            f_rows[(3 * j + a) * q + c] -= g[k][a];
                        }
                    }
                }
            }
        }
        let per = if self.energy_per_atom { 1.0 / n as f64 } else { 1.0 };
        for (c, v) in e_row.iter_mut().enumerate() {
            *v *= per * self.d[c];
        }
        let mut worst = (self.nu(&e_row), None);
        for j in 0..n {
            for a in 0..3 {
                let row = &mut f_rows[(3 * j + a) * q..(3 * j + a + 1) * q];
                for (c, v) in row.iter_mut().enumerate() {
                    *v *= self.d[c];
                }
                let nu = self.nu(row);
                if nu > worst.0 {
                    worst = (nu, Some(j));
                }
            }
        }
        Ok(worst)
    }
}
