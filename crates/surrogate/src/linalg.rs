//! Dense f64 linear algebra for the fit: Cholesky factorisation, the triangular solves the ridge
//! system and the predictive variance need, log-determinant, inverse. Hand-written (CLAUDE.md
//! rule 2: crates for plumbing only). All matrices are square, row-major (`a[i * n + j]`), and a
//! Cholesky factor L is lower triangular with the strict upper triangle zeroed, so A = L Lᵀ.
//!
//! The factorisation is the unpivoted Cholesky–Banachiewicz loop: for j = 0..n,
//! L_jj = √(A_jj − Σ_{k<j} L_jk²) and L_ij = (A_ij − Σ_{k<j} L_ik L_jk)/L_jj for i > j. It is
//! backward stable for symmetric positive definite input; a non-positive pivot means the matrix
//! is not (numerically) positive definite and is reported with its column. No pivoting: the
//! callers add a positive ridge to a Gram matrix, which is SPD by construction unless λ = 0 and
//! the data are rank deficient, and that case is an error the caller reports (`FitError::Singular`).

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinalgError {
    /// The pivot at this column was ≤ 0 (or NaN): the matrix is not positive definite.
    NotPositiveDefinite(usize),
}

impl fmt::Display for LinalgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinalgError::NotPositiveDefinite(k) => {
                write!(f, "matrix is not positive definite (pivot {k} <= 0)")
            }
        }
    }
}

impl std::error::Error for LinalgError {}

/// Replace the symmetric positive definite `a` (n×n, row-major; only the lower triangle is read)
/// by its lower Cholesky factor L with A = L Lᵀ. The strict upper triangle is set to zero. On
/// failure `a` is left partially overwritten.
///
/// # Panics
/// If `a.len() != n * n`.
pub fn cholesky_in_place(a: &mut [f64], n: usize) -> Result<(), LinalgError> {
    assert_eq!(a.len(), n * n, "cholesky_in_place: slice is not n×n");
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if !(d > 0.0) {
            return Err(LinalgError::NotPositiveDefinite(j));
        }
        let ljj = d.sqrt();
        a[j * n + j] = ljj;
        let inv = 1.0 / ljj;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s * inv;
        }
    }
    for i in 0..n {
        for j in i + 1..n {
            a[i * n + j] = 0.0;
        }
    }
    Ok(())
}

/// Solve L x = b in place (forward substitution), L lower triangular from [`cholesky_in_place`].
pub fn forward_substitute(l: &[f64], n: usize, b: &mut [f64]) {
    debug_assert_eq!(l.len(), n * n);
    debug_assert_eq!(b.len(), n);
    for i in 0..n {
        let row = &l[i * n..i * n + i];
        let mut s = b[i];
        for (k, lik) in row.iter().enumerate() {
            s -= lik * b[k];
        }
        b[i] = s / l[i * n + i];
    }
}

/// Solve Lᵀ x = b in place (back substitution with the transposed lower factor).
pub fn backward_substitute_transposed(l: &[f64], n: usize, b: &mut [f64]) {
    debug_assert_eq!(l.len(), n * n);
    debug_assert_eq!(b.len(), n);
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in i + 1..n {
            s -= l[k * n + i] * b[k];
        }
        b[i] = s / l[i * n + i];
    }
}

/// Solve (L Lᵀ) x = b in place.
pub fn cholesky_solve(l: &[f64], n: usize, b: &mut [f64]) {
    forward_substitute(l, n, b);
    backward_substitute_transposed(l, n, b);
}

/// ln det(L Lᵀ) = 2 Σ_i ln L_ii.
pub fn cholesky_logdet(l: &[f64], n: usize) -> f64 {
    debug_assert_eq!(l.len(), n * n);
    2.0 * (0..n).map(|i| l[i * n + i].ln()).sum::<f64>()
}

/// The full symmetric inverse (L Lᵀ)⁻¹, row-major n×n, by solving against the identity.
pub fn cholesky_inverse(l: &[f64], n: usize) -> Vec<f64> {
    let mut inv = vec![0.0; n * n];
    let mut col = vec![0.0; n];
    for j in 0..n {
        col.iter_mut().for_each(|v| *v = 0.0);
        col[j] = 1.0;
        cholesky_solve(l, n, &mut col);
        for i in 0..n {
            inv[i * n + j] = col[i];
        }
    }
    // Symmetrise to remove the rounding asymmetry of column-wise solves.
    for i in 0..n {
        for j in i + 1..n {
            let m = 0.5 * (inv[i * n + j] + inv[j * n + i]);
            inv[i * n + j] = m;
            inv[j * n + i] = m;
        }
    }
    inv
}

/// tr((L Lᵀ)⁻¹) = ‖L⁻¹‖_F², without forming the inverse: forward-substitute every unit vector.
pub fn cholesky_trace_inverse(l: &[f64], n: usize) -> f64 {
    let mut tr = 0.0;
    let mut col = vec![0.0; n];
    for j in 0..n {
        col.iter_mut().for_each(|v| *v = 0.0);
        col[j] = 1.0;
        forward_substitute(l, n, &mut col);
        tr += col[j..].iter().map(|v| v * v).sum::<f64>();
    }
    tr
}

/// The quadratic form vᵀ (L Lᵀ)⁻¹ v = ‖L⁻¹ v‖²; `v` is overwritten with L⁻¹ v.
pub fn cholesky_quad_form(l: &[f64], n: usize, v: &mut [f64]) -> f64 {
    forward_substitute(l, n, v);
    v.iter().map(|x| x * x).sum()
}

/// y = A x for a row-major n×n matrix.
pub fn mat_vec(a: &[f64], n: usize, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(a.len(), n * n);
    for i in 0..n {
        let row = &a[i * n..(i + 1) * n];
        y[i] = row.iter().zip(x).map(|(a, b)| a * b).sum();
    }
}

/// Eigen-decomposition of the symmetric `a` (n×n, row-major) by the cyclic Jacobi method:
/// returns the eigenvalues and the eigenvectors as the COLUMNS of a row-major n×n matrix
/// (`v[k * n + i]` is component k of eigenvector i). Each rotation annihilates one off-diagonal
/// pair (Golub & Van Loan, *Matrix Computations*, §8.5, the stable tan θ formula); sweeps stop
/// when the off-diagonal Frobenius norm is below `f64::EPSILON²`·‖A‖_F (rounding level), at most
/// 60 sweeps. Jacobi is backward stable with eigenvalue errors O(ε)·‖A‖ — which is what the
/// rounding threshold of [`crate::novelty`] relies on — at O(n³) per sweep: ≈ 1 ms at n = 29,
/// tens of ms at n = 176 (S = 2), and seconds at n = 1232 (S = 4; a handoff item).
///
/// # Panics
/// If `a.len() != n * n`.
pub fn symmetric_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(a.len(), n * n, "symmetric_eigen: slice is not n×n");
    let mut a = a.to_vec();
    // symmetrise (the caller's matrix is symmetric to rounding)
    for i in 0..n {
        for j in i + 1..n {
            let m = 0.5 * (a[i * n + j] + a[j * n + i]);
            a[i * n + j] = m;
            a[j * n + i] = m;
        }
    }
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let norm2: f64 = a.iter().map(|x| x * x).sum();
    let stop = (f64::EPSILON * f64::EPSILON) * (f64::EPSILON * f64::EPSILON) * norm2;
    for _ in 0..60 {
        let off: f64 = (0..n).flat_map(|p| (p + 1..n).map(move |q| (p, q))).map(|(p, q)| 2.0 * a[p * n + q] * a[p * n + q]).sum();
        if off <= stop {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = if theta >= 0.0 { 1.0 / (theta + (theta * theta + 1.0).sqrt()) } else { -1.0 / (-theta + (theta * theta + 1.0).sqrt()) };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i * n + i]).collect(), v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Jacobi on a random symmetric matrix and on a rank-deficient Gram: A v = λ v to rounding,
    /// V orthonormal, the null eigenvalues of a rank-r Gram at the ε·‖A‖ level.
    #[test]
    fn symmetric_eigen_reconstructs_and_finds_a_null_space() {
        let mut rng = StdRng::seed_from_u64(5);
        let n = 12;
        // rank 7 Gram from 40 random rows in a 7-dimensional subspace
        let basis: Vec<Vec<f64>> = (0..7).map(|_| (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect()).collect();
        let mut g = vec![0.0; n * n];
        for _ in 0..40 {
            let c: Vec<f64> = (0..7).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let row: Vec<f64> = (0..n).map(|k| (0..7).map(|i| c[i] * basis[i][k]).sum()).collect();
            for i in 0..n {
                for j in 0..n {
                    g[i * n + j] += row[i] * row[j];
                }
            }
        }
        let (ev, v) = symmetric_eigen(&g, n);
        let norm = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        for i in 0..n {
            for k in 0..n {
                let av: f64 = (0..n).map(|j| g[k * n + j] * v[j * n + i]).sum();
                assert!((av - ev[i] * v[k * n + i]).abs() < 1e-12 * norm, "A v ≠ λ v");
            }
            for j in 0..n {
                let d: f64 = (0..n).map(|k| v[k * n + i] * v[k * n + j]).sum();
                assert!((d - if i == j { 1.0 } else { 0.0 }).abs() < 1e-13);
            }
        }
        let mut sorted = ev.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(sorted[..5].iter().all(|x| x.abs() < 1e-14 * norm), "{sorted:?}");
        assert!(sorted[5] > 1e-6 * norm, "{sorted:?}");
    }

    /// Determinant by unpivoted Gaussian elimination (fine for SPD), as an independent check.
    fn det_by_elimination(a: &[f64], n: usize) -> f64 {
        let mut m = a.to_vec();
        let mut det = 1.0;
        for k in 0..n {
            let p = m[k * n + k];
            det *= p;
            for i in k + 1..n {
                let f = m[i * n + k] / p;
                for j in k..n {
                    m[i * n + j] -= f * m[k * n + j];
                }
            }
        }
        det
    }

    #[test]
    fn cholesky_solves_a_random_spd_system() {
        let n = 50;
        let mut rng = StdRng::seed_from_u64(11);
        let a: Vec<f64> = (0..n * n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        // M = AᵀA + I is SPD with eigenvalues ≥ 1.
        let mut m = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = if i == j { 1.0 } else { 0.0 };
                for k in 0..n {
                    s += a[k * n + i] * a[k * n + j];
                }
                m[i * n + j] = s;
            }
        }
        let b: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let mut l = m.clone();
        cholesky_in_place(&mut l, n).unwrap();
        for i in 0..n {
            for j in i + 1..n {
                assert_eq!(l[i * n + j], 0.0, "upper triangle zeroed");
            }
        }
        // L Lᵀ reproduces M.
        for i in 0..n {
            for j in 0..n {
                let s: f64 = (0..n).map(|k| l[i * n + k] * l[j * n + k]).sum();
                assert!((s - m[i * n + j]).abs() < 1e-12, "LLᵀ ≠ M at ({i},{j})");
            }
        }
        let mut x = b.clone();
        cholesky_solve(&l, n, &mut x);
        let mut mx = vec![0.0; n];
        mat_vec(&m, n, &x, &mut mx);
        let resid = mx.iter().zip(&b).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max);
        assert!(resid < 1e-10, "|Mx − b|∞ = {resid:e}");

        // logdet against the product of the elimination pivots.
        let det = det_by_elimination(&m, n);
        let logdet = cholesky_logdet(&l, n);
        assert!(
            (logdet - det.ln()).abs() < 1e-8 * logdet.abs(),
            "logdet {logdet} vs ln det {}",
            det.ln()
        );
        // Also against the explicit product of the Cholesky pivots squared.
        let prod: f64 = (0..n).map(|i| l[i * n + i] * l[i * n + i]).product();
        assert!((prod.ln() - logdet).abs() < 1e-10 * logdet.abs());

        // Inverse and trace of the inverse agree with each other and with M·M⁻¹ = I.
        let inv = cholesky_inverse(&l, n);
        let tr: f64 = (0..n).map(|i| inv[i * n + i]).sum();
        assert!((tr - cholesky_trace_inverse(&l, n)).abs() < 1e-12 * tr);
        for i in 0..n {
            for j in 0..n {
                let s: f64 = (0..n).map(|k| m[i * n + k] * inv[k * n + j]).sum();
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((s - expect).abs() < 1e-10, "M·M⁻¹ ≠ I at ({i},{j}): {s}");
            }
        }
        // Quadratic form vᵀM⁻¹v = vᵀ(M⁻¹v).
        let mut v = b.clone();
        let qf = cholesky_quad_form(&l, n, &mut v);
        let direct: f64 = b
            .iter()
            .enumerate()
            .map(|(i, bi)| bi * (0..n).map(|k| inv[i * n + k] * b[k]).sum::<f64>())
            .sum();
        assert!((qf - direct).abs() < 1e-10 * direct.abs());
    }

    #[test]
    fn cholesky_small_known_determinant_and_failure_column() {
        // det = 4(15 − 1) − 2(6 − 0) = 44.
        let m = [4.0, 2.0, 0.0, 2.0, 5.0, 1.0, 0.0, 1.0, 3.0];
        let mut l = m;
        cholesky_in_place(&mut l, 3).unwrap();
        assert!((cholesky_logdet(&l, 3) - 44f64.ln()).abs() < 1e-14);
        // Indefinite: the 2×2 leading minor is 4·1 − 3·3 < 0, so column 1 fails.
        let mut bad = [4.0, 3.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(
            cholesky_in_place(&mut bad, 3),
            Err(LinalgError::NotPositiveDefinite(1))
        );
        let mut zero = [0.0; 1];
        assert_eq!(cholesky_in_place(&mut zero, 1), Err(LinalgError::NotPositiveDefinite(0)));
    }
}
