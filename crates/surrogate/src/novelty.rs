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

use crate::descriptor::{Basis, DescribeError};
use crate::fit::Accumulator;
use crate::linalg::{symmetric_eigen, symmetric_eigen_householder};

/// The unspanned directions of a fit and the rounding threshold (module doc).
#[derive(Clone, Debug, PartialEq)]
pub struct NullSpace {
    pub basis: Basis,
    /// Full column q = s·P + k → reduced column, `None` for a tied duplicate (as the accumulator).
    pub col_index: Vec<Option<usize>>,
    /// Per reduced column: the standardisation 1/scale of its representative.
    pub d: Vec<f64>,
    /// The unit eigenvectors u_v (length Q′) of the eigenvalues ≤ `tau`, stored by column
    /// component: `columns[k * dim + v]` is component k of u_v (the layout `nu_rows` reads one
    /// query entry at a time with; 2026-11-17).
    pub columns: Vec<f64>,
    /// Number of unspanned directions.
    pub dim: usize,
    /// τ = Q′·ε·tr(G̃).
    pub tau: f64,
    /// Whether the energy rows were per atom.
    pub energy_per_atom: bool,
}

impl NullSpace {
    /// The unspanned directions of `acc`'s training rows under the standardisation `scale` (the
    /// fitted model's per-feature scale; 0 means 1, as in the fit).
    pub fn of(acc: &Accumulator, scale: &[f64]) -> NullSpace {
        let _prof = crate::prof::span("fit: NullSpace::of");
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
        // a body-4 basis (a shared collection's, Q′ up to 2 688) takes Householder–QL: the same
        // O(ε)‖G̃‖ backward error the threshold rests on (`linalg` tests: null eigenvalues at
        // 8e-17·tr against τ/tr = Q′ε), minutes faster than Jacobi there; every other basis keeps
        // Jacobi bit for bit. Since 2026-11-22 (the user: refits of a four-element collection took
        // ~171 s, this eigendecomposition on one thread most of it) the body-4 one is faer's blocked,
        // threaded tridiagonalisation with the same implicit QL (`dense::symmetric_eigen`: the same
        // algorithm and backward error, eigenvalues sorted); the unthreaded `linalg` one only if its
        // QL does not converge on a block (never seen)
        let (ev, v) = if basis.body4 {
            crate::dense::symmetric_eigen(&g, q).unwrap_or_else(|_| symmetric_eigen_householder(&g, q))
        } else {
            symmetric_eigen(&g, q)
        };
        let kept: Vec<usize> = (0..q).filter(|&i| ev[i] <= tau).collect();
        let (v, kept_ref) = (&v, &kept);
        let columns = (0..q).flat_map(|k| kept_ref.iter().map(move |&i| v[k * q + i])).collect();
        NullSpace { basis, col_index, d, columns, dim: kept.len(), tau, energy_per_atom: acc.energy_per_atom.unwrap_or(true) }
    }

    /// Number of unspanned directions.
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Unspanned direction `v` (length Q′).
    pub fn vector(&self, v: usize) -> Vec<f64> {
        debug_assert!(v < self.dim, "direction {v} of {}", self.dim);
        if self.dim == 0 {
            return Vec::new();
        }
        self.columns.iter().skip(v).step_by(self.dim).copied().collect()
    }

    /// ν of each standardised row (module doc): Σ_v (u_v·row)² / (N_null·τ), every dot product
    /// summed over k = 0 … Q′ − 1 in order. Bit for bit the row-by-row form Σ_k u_v[k]·row[k]
    /// (2026-11-17): the dots are accumulated one query entry k at a time over all directions at
    /// once (contiguous, vectorisable, the directions read once for all rows), and an entry that is
    /// exactly zero — a species block or neighbour channel the query does not touch — is skipped:
    /// its products are ±0, which leave a nonzero partial sum unchanged and at most flip the sign
    /// of a zero one, gone in the square.
    fn nu_rows(&self, rows: &[&[f64]]) -> Vec<f64> {
        let dim = self.dim;
        // the callers return ν = 0 before asking when nothing is unspanned
        debug_assert!(dim > 0, "nu_rows without unspanned directions");
        if dim == 0 {
            return Vec::new();
        }
        let q = self.d.len();
        let mut dots = vec![0.0f64; rows.len() * dim];
        for k in 0..q {
            let col = &self.columns[k * dim..(k + 1) * dim];
            for (r, row) in rows.iter().enumerate() {
                let x = row[k];
                if x == 0.0 {
                    continue;
                }
                for (a, u) in dots[r * dim..(r + 1) * dim].iter_mut().zip(col) {
                    *a += u * x;
                }
            }
        }
        dots.chunks(dim.max(1)).take(rows.len()).map(|d| d.iter().map(|x| x.powi(2)).sum::<f64>() / (dim as f64 * self.tau)).collect()
    }

    /// The largest ν over the query's energy row and force rows, and where it was: `None` for the
    /// energy row, `Some(j)` for atom j's force rows. (0, None) when nothing is unspanned.
    pub fn novelty(&self, pos: &[[f64; 3]], z: &[u8]) -> Result<(f64, Option<usize>), DescribeError> {
        if self.dim == 0 || !(self.tau > 0.0) {
            return Ok((0.0, None));
        }
        // the same row builder as the LOO's (`loo::standardised_rows`; final review M7)
        let (e_row, f_rows) = crate::loo::standardised_rows(&self.basis, &self.col_index, &self.d, self.energy_per_atom, pos, z)?;
        let nus = self.nu_rows(&std::iter::once(&e_row[..]).chain(f_rows.iter().map(|r| &r[..])).collect::<Vec<_>>());
        let mut worst = (nus[0], None);
        for (r, &nu) in nus[1..].iter().enumerate() {
            if nu > worst.0 {
                worst = (nu, Some(r / 3));
            }
        }
        Ok(worst)
    }

    /// ν of the query's energy row and, per atom, the largest ν of its three force rows (the
    /// split of [`novelty`](Self::novelty) a per-atom verdict reads; the same rows, the same ν):
    /// `novelty` is (max of all of them, where). Zeros when nothing is unspanned.
    pub fn novelty_per_atom(&self, pos: &[[f64; 3]], z: &[u8]) -> Result<(f64, Vec<f64>), DescribeError> {
        if self.dim == 0 || !(self.tau > 0.0) {
            return Ok((0.0, vec![0.0; pos.len()]));
        }
        let (e_row, f_rows) = crate::loo::standardised_rows(&self.basis, &self.col_index, &self.d, self.energy_per_atom, pos, z)?;
        let nus = self.nu_rows(&std::iter::once(&e_row[..]).chain(f_rows.iter().map(|r| &r[..])).collect::<Vec<_>>());
        let mut atoms = vec![0.0f64; pos.len()];
        for (r, &nu) in nus[1..].iter().enumerate() {
            atoms[r / 3] = atoms[r / 3].max(nu);
        }
        Ok((nus[0], atoms))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// The batched ν is the row-by-row form's bit for bit (2026-11-17: `vectors[v]·row` summed
    /// left to right, squared, summed over v), with zero entries in the rows and every row count.
    #[test]
    fn batched_nu_is_the_row_by_row_form_bit_for_bit() {
        let mut rng = StdRng::seed_from_u64(23);
        for (q, dim) in [(5, 1), (29, 17), (64, 40), (130, 3)] {
            let vectors: Vec<Vec<f64>> = (0..dim).map(|_| (0..q).map(|_| rng.gen_range(-1.0..1.0)).collect()).collect();
            let columns = (0..q).flat_map(|k| vectors.iter().map(move |u| u[k])).collect();
            let ns = NullSpace { basis: Basis::new(&[1]), col_index: Vec::new(), d: vec![1.0; q], columns, dim, tau: 1e-13, energy_per_atom: true };
            for v in 0..dim {
                assert_eq!(ns.vector(v), vectors[v]);
            }
            for m in [1, 3, 7, 10] {
                let rows: Vec<Vec<f64>> = (0..m).map(|_| (0..q).map(|_| if rng.gen_bool(0.5) { 0.0 } else { rng.gen_range(-2.0..2.0) }).collect()).collect();
                let got = ns.nu_rows(&rows.iter().map(|r| &r[..]).collect::<Vec<_>>());
                for (row, nu) in rows.iter().zip(&got) {
                    let e: f64 = vectors.iter().map(|u| u.iter().zip(row).map(|(a, b)| a * b).sum::<f64>().powi(2)).sum();
                    let want = e / (dim as f64 * ns.tau);
                    assert_eq!(nu.to_bits(), want.to_bits(), "q {q} dim {dim} m {m}: {nu:e} vs {want:e}");
                }
            }
        }
    }

    /// The body-4 null space's eigensolver (2026-11-22: faer's threaded tridiagonalisation,
    /// `dense::symmetric_eigen`) finds the unthreaded `linalg::symmetric_eigen_householder`'s null
    /// space: on rank-deficient Grams G = AᵀA (graded columns, as standardised descriptors are),
    /// the same count of eigenvalues at or below τ = Q′ε·tr(G), none of them near τ, and the same
    /// subspace (the projectors agree to 1e-9).
    #[test]
    fn the_threaded_eigensolver_finds_the_same_null_space() {
        let mut rng = StdRng::seed_from_u64(7);
        for (q, rank) in [(40, 25), (180, 120), (400, 330)] {
            let scale: Vec<f64> = (0..q).map(|j| 10f64.powf(-3.0 * j as f64 / q as f64)).collect();
            let a: Vec<f64> = (0..rank * q).map(|i| rng.gen_range(-1.0..1.0) * scale[i % q]).collect();
            let mut g = vec![0.0; q * q];
            for i in 0..q {
                for j in 0..q {
                    g[i * q + j] = (0..rank).map(|r| a[r * q + i] * a[r * q + j]).sum();
                }
            }
            let tau = q as f64 * f64::EPSILON * (0..q).map(|i| g[i * q + i]).sum::<f64>();
            let null = |ev: &[f64], v: &[f64]| {
                let kept: Vec<usize> = (0..q).filter(|&i| ev[i] <= tau).collect();
                // no eigenvalue within a factor 100 of τ: the count does not hinge on rounding
                assert!(ev.iter().all(|&e| e <= tau / 100.0 || e >= tau * 100.0), "q {q}: an eigenvalue near τ");
                let mut p = vec![0.0; q * q];
                for i in 0..q {
                    for j in 0..q {
                        p[i * q + j] = kept.iter().map(|&k| v[i * q + k] * v[j * q + k]).sum();
                    }
                }
                (kept.len(), p)
            };
            let (ev_new, v_new) = crate::dense::symmetric_eigen(&g, q).expect("QL converges");
            let (ev_old, v_old) = symmetric_eigen_householder(&g, q);
            let ((n_new, p_new), (n_old, p_old)) = (null(&ev_new, &v_new), null(&ev_old, &v_old));
            assert_eq!((n_new, n_old), (q - rank, q - rank), "q {q}: the null dimension");
            let worst = p_new.iter().zip(&p_old).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
            assert!(worst < 1e-9, "q {q}: projectors differ by {worst:e}");
        }
    }
}
