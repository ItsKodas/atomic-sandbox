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

// ---- the large-Q′ path (M8a Task 5): the body-4 basis of a shared collection ----
//
// The routines below serve the fit of a body-4 basis (`fit` module doc, "The large-basis path"),
// where Q′ reaches 2 688 at four species and the O(n³) kernels above — scalar dot products with
// one accumulator, Jacobi's sweeps — cost minutes. They are new functions, not replacements:
// every per-scene model keeps the routines above and its results bit for bit.

/// Σ a_i b_i with eight independent partial sums (vectorisable; the summation order differs
/// from a left-to-right loop, so results differ from [`forward_substitute`]-style loops in the
/// last bits).
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f64; 8];
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut tail = 0.0;
    for (x, y) in ra.iter().zip(rb) {
        tail += x * y;
    }
    ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7])) + tail
}

/// y += α x.
#[inline]
pub fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += alpha * xi;
    }
}

/// [`cholesky_in_place`] with the vectorised [`dot`] in the inner loop: the same algorithm and
/// backward error, a different rounding.
pub fn cholesky_in_place_fast(a: &mut [f64], n: usize) -> Result<(), LinalgError> {
    assert_eq!(a.len(), n * n, "cholesky_in_place_fast: slice is not n×n");
    for j in 0..n {
        let (head, tail) = a.split_at_mut((j + 1) * n);
        let row_j = &mut head[j * n..];
        let d = row_j[j] - dot(&row_j[..j], &row_j[..j]);
        if !(d > 0.0) {
            return Err(LinalgError::NotPositiveDefinite(j));
        }
        let ljj = d.sqrt();
        row_j[j] = ljj;
        let inv = 1.0 / ljj;
        let lj = &row_j[..j];
        for i in j + 1..n {
            let row_i = &mut tail[(i - j - 1) * n..(i - j) * n];
            row_i[j] = (row_i[j] - dot(&row_i[..j], lj)) * inv;
        }
    }
    for i in 0..n {
        a[i * n + i + 1..(i + 1) * n].iter_mut().for_each(|v| *v = 0.0);
    }
    Ok(())
}

/// The transpose of L⁻¹ for a lower factor L: `xt[j * n + i]` = (L⁻¹)_ij, nonzero for i ≥ j
/// (row j of the result is column j of L⁻¹). n³/3 flops.
pub fn lower_inverse_transposed(l: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(l.len(), n * n);
    let mut xt = vec![0.0; n * n];
    for j in 0..n {
        let row = &mut xt[j * n..(j + 1) * n];
        row[j] = 1.0 / l[j * n + j];
        for i in j + 1..n {
            let s = dot(&l[i * n + j..i * n + i], &row[j..i]);
            row[i] = -s / l[i * n + i];
        }
    }
    xt
}

/// (L Lᵀ)⁻¹ = L⁻ᵀ L⁻¹ from [`lower_inverse_transposed`]'s output, full symmetric, row-major
/// (n³/3 flops).
pub fn inverse_from_lower_inverse(xt: &[f64], n: usize) -> Vec<f64> {
    let mut inv = vec![0.0; n * n];
    for a in 0..n {
        for b in a..n {
            let v = dot(&xt[a * n + b..(a + 1) * n], &xt[b * n + b..(b + 1) * n]);
            inv[a * n + b] = v;
            inv[b * n + a] = v;
        }
    }
    inv
}

/// A symmetric matrix reduced to tridiagonal form T = Qᵀ A Q by Householder reflections
/// (Golub & Van Loan §8.3.1): `d` the diagonal, `e[k]` = T_{k,k+1} (e[n−1] = 0), and
/// Q = P₀ P₁ ⋯ P_{n−3}, P_k = I − β_k v_k v_kᵀ acting on coordinates k+1.. (`v_k` stored from
/// coordinate k+1). Backward stable: T is exactly similar to A + E with ‖E‖ = O(nε)‖A‖.
pub struct Tridiagonal {
    pub d: Vec<f64>,
    pub e: Vec<f64>,
    reflectors: Vec<(Vec<f64>, f64)>,
}

impl Tridiagonal {
    /// Reduce the symmetric `a` (n×n row-major; read in full, symmetrised first). 2n³ flops.
    pub fn of(a: &[f64], n: usize) -> Tridiagonal {
        assert_eq!(a.len(), n * n, "Tridiagonal::of: slice is not n×n");
        let mut a = a.to_vec();
        for i in 0..n {
            for j in i + 1..n {
                let m = 0.5 * (a[i * n + j] + a[j * n + i]);
                a[i * n + j] = m;
                a[j * n + i] = m;
            }
        }
        let mut d = vec![0.0; n];
        let mut e = vec![0.0; n];
        let mut reflectors = Vec::with_capacity(n.saturating_sub(2));
        let mut p = vec![0.0; n];
        for k in 0..n.saturating_sub(2) {
            let m = n - k - 1; // size of the trailing block
            // the reflector from x scaled to max |x| = 1: a rank-deficient matrix leaves trailing
            // blocks at 1e-140 and below, whose ‖x‖² underflows (β = ∞, then NaN) unscaled; P =
            // I − β v vᵀ does not depend on v's scale
            let xs = &a[k * n + k + 1..(k + 1) * n];
            let big = xs.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            d[k] = a[k * n + k];
            if big == 0.0 || !big.is_finite() {
                e[k] = if big.is_finite() { 0.0 } else { f64::NAN };
                reflectors.push((vec![0.0; m], 0.0));
                continue;
            }
            let mut v: Vec<f64> = xs.iter().map(|x| x / big).collect();
            let norm_s = dot(&v, &v).sqrt();
            let alpha_s = if v[0] >= 0.0 { -norm_s } else { norm_s };
            v[0] -= alpha_s;
            let alpha = alpha_s * big;
            let beta = 2.0 / dot(&v, &v);
            // p = β B v over the trailing block B = a[k+1.., k+1..]
            for (r, pr) in p[..m].iter_mut().enumerate() {
                let row = &a[(k + 1 + r) * n + k + 1..(k + 2 + r) * n];
                *pr = beta * dot(row, &v);
            }
            let kk = 0.5 * beta * dot(&p[..m], &v);
            for r in 0..m {
                p[r] -= kk * v[r];
            }
            // B ← B − v wᵀ − w vᵀ, w = p
            for r in 0..m {
                let (vr, wr) = (v[r], p[r]);
                let row = &mut a[(k + 1 + r) * n + k + 1..(k + 2 + r) * n];
                for c in 0..m {
                    row[c] -= vr * p[c] + wr * v[c];
                }
            }
            e[k] = alpha;
            reflectors.push((v, beta));
        }
        if n >= 2 {
            d[n - 2] = a[(n - 2) * n + n - 2];
            e[n - 2] = a[(n - 2) * n + n - 1];
        }
        if n >= 1 {
            d[n - 1] = a[n * n - 1];
        }
        Tridiagonal { d, e, reflectors }
    }

    /// x ← Qᵀ x (= P_{n−3} ⋯ P₀ x: P₀ first).
    pub fn apply_qt(&self, x: &mut [f64]) {
        for (k, (v, beta)) in self.reflectors.iter().enumerate() {
            let seg = &mut x[k + 1..];
            let s = beta * dot(v, seg);
            axpy(-s, v, seg);
        }
    }

    /// x ← Q x (= P₀ ⋯ P_{n−3} x: P_{n−3} first).
    pub fn apply_q(&self, x: &mut [f64]) {
        for (k, (v, beta)) in self.reflectors.iter().enumerate().rev() {
            let seg = &mut x[k + 1..];
            let s = beta * dot(v, seg);
            axpy(-s, v, seg);
        }
    }
}

/// The eigenvalues of the symmetric tridiagonal (d, e) (`e[k]` = T_{k,k+1}) by the implicit QL
/// method with Wilkinson-type shifts (Numerical Recipes `tqli`; Golub & Van Loan §8.3.5): on return
/// `d` holds them (unsorted) and `e` is destroyed. Every Givens rotation of the iteration is handed
/// to `rot(i, c, s)` — it acts on the eigen-index pair (i, i+1) as (x_i, x_{i+1}) ←
/// (c x_i − s x_{i+1}, s x_i + c x_{i+1}) — so a caller can carry vectors along: applied to the
/// rows of an identity it builds Wᵀ (the eigenvectors of T as rows), applied to a vector y it
/// builds Wᵀ y. `Err` after 60 iterations on one eigenvalue (never seen on symmetric input).
pub fn tridiagonal_ql(d: &mut [f64], e: &mut [f64], mut rot: impl FnMut(usize, f64, f64)) -> Result<(), LinalgError> {
    let n = d.len();
    if n == 0 {
        return Ok(());
    }
    e[n - 1] = 0.0;
    // an off-diagonal entry at rounding level of the whole matrix is a split as well (a zero
    // diagonal block — a column no row ever touched — would otherwise never deflate against its
    // zero neighbours): the perturbation is ≤ ε‖T‖, the reduction's own backward error
    let norm = d.iter().chain(e.iter()).fold(0.0f64, |m, v| m.max(v.abs()));
    let tiny = f64::EPSILON * norm;
    for l in 0..n {
        let mut iter = 0;
        loop {
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= f64::EPSILON * dd || e[m].abs() <= tiny {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iter += 1;
            if iter > 60 {
                return Err(LinalgError::NotPositiveDefinite(l));
            }
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = g.hypot(1.0);
            g = d[m] - d[l] + e[l] / (g + if g >= 0.0 { r.abs() } else { -r.abs() });
            let (mut s, mut c, mut p) = (1.0, 1.0, 0.0);
            let mut early = false;
            let mut i = m;
            while i > l {
                i -= 1;
                let f = s * e[i];
                let b = c * e[i];
                r = f.hypot(g);
                e[i + 1] = r;
                if r == 0.0 {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    early = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                // NR: z_{k,i+1} ← s z_{k,i} + c z_{k,i+1}; z_{k,i} ← c z_{k,i} − s z_{k,i+1}
                rot(i, c, s);
            }
            if early {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
    Ok(())
}

/// Eigen-decomposition of the symmetric `a` by Householder tridiagonalisation and implicit QL
/// ([`Tridiagonal`], [`tridiagonal_ql`]): the same output layout as [`symmetric_eigen`] (values
/// unsorted; `v[k * n + i]` is component k of eigenvector i), the same O(ε)‖A‖ backward error,
/// ≈ 7n³ flops instead of Jacobi's ≈ 10n³ per sweep over 6–10 sweeps.
///
/// # Panics
/// If `a.len() != n * n`, or (never seen) the QL iteration does not converge.
pub fn symmetric_eigen_householder(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let t = Tridiagonal::of(a, n);
    let mut d = t.d.clone();
    let mut e = t.e.clone();
    // Wᵀ as rows, rotated pairwise
    let mut wt = vec![0.0; n * n];
    for i in 0..n {
        wt[i * n + i] = 1.0;
    }
    tridiagonal_ql(&mut d, &mut e, |i, c, s| {
        let (lo, hi) = wt.split_at_mut((i + 1) * n);
        let (ri, rj) = (&mut lo[i * n..], &mut hi[..n]);
        for k in 0..n {
            let (x, y) = (ri[k], rj[k]);
            ri[k] = c * x - s * y;
            rj[k] = s * x + c * y;
        }
    })
    .expect("symmetric_eigen_householder: QL did not converge");
    // eigenvector i of A = Q · (row i of Wᵀ); written as columns
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        let row = &mut wt[i * n..(i + 1) * n];
        t.apply_q(row);
        for k in 0..n {
            v[k * n + i] = row[k];
        }
    }
    (d, v)
}

/// R ← the upper factor of RᵀR + u uᵀ (R = Lᵀ upper triangular, row-major; `u` is destroyed):
/// the rank-one Cholesky update by Givens rotations (Golub & Van Loan §6.5.4), 4n²/2 flops on
/// contiguous rows.
pub fn cholesky_update_upper(r: &mut [f64], n: usize, u: &mut [f64]) {
    for k in 0..n {
        let rkk = r[k * n + k];
        let uk = u[k];
        if uk == 0.0 {
            continue;
        }
        let rr = rkk.hypot(uk);
        let (c, s) = (rr / rkk, uk / rkk);
        r[k * n + k] = rr;
        let row = &mut r[k * n + k + 1..(k + 1) * n];
        let us = &mut u[k + 1..];
        for (rj, uj) in row.iter_mut().zip(us.iter_mut()) {
            let v = (*rj + s * *uj) / c;
            *uj = c * *uj - s * v;
            *rj = v;
        }
    }
}

/// x ← (RᵀR)⁻¹ x for an upper factor R (row-major): Rᵀ z = x then R x = z, both on rows.
pub fn upper_solve(r: &[f64], n: usize, x: &mut [f64]) {
    // Rᵀ z = x, column-oriented: z_k = x_k / R_kk, then x_j −= R_kj z_k for j > k
    for k in 0..n {
        let zk = x[k] / r[k * n + k];
        x[k] = zk;
        let (_, rest) = x.split_at_mut(k + 1);
        axpy(-zk, &r[k * n + k + 1..(k + 1) * n], rest);
    }
    // R x = z
    for k in (0..n).rev() {
        let s = dot(&r[k * n + k + 1..(k + 1) * n], &x[k + 1..]);
        x[k] = (x[k] - s) / r[k * n + k];
    }
}

/// y = A x for a row-major n×n matrix with [`dot`] per row.
pub fn mat_vec_fast(a: &[f64], n: usize, x: &[f64], y: &mut [f64]) {
    for (i, yi) in y.iter_mut().enumerate().take(n) {
        *yi = dot(&a[i * n..(i + 1) * n], x);
    }
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

    fn random_spd(rng: &mut StdRng, n: usize, ridge: f64) -> Vec<f64> {
        let m: Vec<f64> = (0..n * n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = (0..n).map(|k| m[i * n + k] * m[j * n + k]).sum::<f64>();
            }
            a[i * n + i] += ridge;
        }
        a
    }

    /// The large-Q′ kernels against the reference ones: Householder–QL eigenpairs (A v = λ v,
    /// orthonormal V, the same spectrum as Jacobi), the fast Cholesky, the triangular inverse and
    /// the rank-one update, at sizes that cross the 8-wide dot product's remainder.
    #[test]
    fn large_basis_kernels_match_the_reference_kernels() {
        let mut rng = StdRng::seed_from_u64(9);
        for n in [1usize, 2, 3, 9, 17, 40] {
            let a = random_spd(&mut rng, n, 0.1);
            let norm = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            let (ev, v) = symmetric_eigen_householder(&a, n);
            for i in 0..n {
                for k in 0..n {
                    let av: f64 = (0..n).map(|j| a[k * n + j] * v[j * n + i]).sum();
                    assert!((av - ev[i] * v[k * n + i]).abs() < 1e-12 * norm, "n {n} pair {i}");
                }
                for j in 0..n {
                    let vv: f64 = (0..n).map(|k| v[k * n + i] * v[k * n + j]).sum();
                    assert!((vv - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12, "n {n} orthonormality {i} {j}");
                }
            }
            let (mut ej, _) = symmetric_eigen(&a, n);
            let mut eh = ev.clone();
            ej.sort_by(f64::total_cmp);
            eh.sort_by(f64::total_cmp);
            for (x, y) in ej.iter().zip(&eh) {
                assert!((x - y).abs() < 1e-12 * norm);
            }
            // Cholesky, inverse
            let mut l_ref = a.clone();
            cholesky_in_place(&mut l_ref, n).unwrap();
            let mut l = a.clone();
            cholesky_in_place_fast(&mut l, n).unwrap();
            for (x, y) in l.iter().zip(&l_ref) {
                assert!((x - y).abs() < 1e-12 * norm.sqrt());
            }
            let inv_ref = cholesky_inverse(&l_ref, n);
            let inv = inverse_from_lower_inverse(&lower_inverse_transposed(&l, n), n);
            let inv_norm = inv_ref.iter().map(|x| x * x).sum::<f64>().sqrt();
            for (x, y) in inv.iter().zip(&inv_ref) {
                assert!((x - y).abs() < 1e-10 * inv_norm, "n {n}: inverse {x} vs {y}");
            }
            // rank-one update of the upper factor, and the solve with it
            let u: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let mut r = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    r[i * n + j] = l[j * n + i];
                }
            }
            let mut uu = u.clone();
            cholesky_update_upper(&mut r, n, &mut uu);
            let mut a2 = a.clone();
            for i in 0..n {
                for j in 0..n {
                    a2[i * n + j] += u[i] * u[j];
                }
            }
            for i in 0..n {
                for j in 0..n {
                    let rr: f64 = (0..n).map(|k| r[k * n + i] * r[k * n + j]).sum();
                    assert!((rr - a2[i * n + j]).abs() < 1e-12 * norm, "n {n}: update ({i},{j})");
                }
            }
            let b: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let mut x = b.clone();
            upper_solve(&r, n, &mut x);
            for i in 0..n {
                let ax: f64 = (0..n).map(|j| a2[i * n + j] * x[j]).sum();
                assert!((ax - b[i]).abs() < 1e-9 * norm * x.iter().map(|v| v.abs()).fold(1.0, f64::max), "n {n}: solve row {i}");
            }
        }
    }

    /// A rank-deficient Gram: the Householder–QL null eigenvalues sit at the ε·‖A‖ level like
    /// Jacobi's (the novelty threshold's premise).
    #[test]
    fn householder_eigen_resolves_a_null_space_to_rounding() {
        let mut rng = StdRng::seed_from_u64(21);
        let (n, r) = (60, 23);
        let basis: Vec<Vec<f64>> = (0..r).map(|_| (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect()).collect();
        let mut g = vec![0.0; n * n];
        for _ in 0..200 {
            let c: Vec<f64> = (0..r).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let row: Vec<f64> = (0..n).map(|k| (0..r).map(|i| c[i] * basis[i][k]).sum()).collect();
            for i in 0..n {
                for j in 0..n {
                    g[i * n + j] += row[i] * row[j];
                }
            }
        }
        let tr: f64 = (0..n).map(|i| g[i * n + i]).sum();
        let (ev, _) = symmetric_eigen_householder(&g, n);
        let mut sorted = ev.clone();
        sorted.sort_by(|a, b| a.abs().total_cmp(&b.abs()));
        let null_max = sorted[n - r - 1].abs() / tr;
        let spanned_min = sorted[n - r].abs() / tr;
        eprintln!("householder null space: max |λ_null|/tr {null_max:.2e}, smallest spanned {spanned_min:.2e}, τ/tr {:.2e}", n as f64 * f64::EPSILON);
        assert!(null_max < n as f64 * f64::EPSILON);
        assert!(spanned_min > 1e3 * n as f64 * f64::EPSILON);
    }

    /// A trailing block at 1e-160 (a rank-deficient Gram's rounding remainder: the collection
    /// test that found it had 1e-142 at k = 244 of 284): ‖x‖² underflows unscaled (β = ∞, then
    /// NaN); the scaled reflector keeps every eigenvalue finite and right to ε‖A‖.
    #[test]
    fn householder_survives_an_underflowing_trailing_block() {
        let mut rng = StdRng::seed_from_u64(31);
        let (big, small) = (random_spd(&mut rng, 6, 0.5), random_spd(&mut rng, 5, 0.5));
        let n = 11;
        let mut a = vec![0.0; n * n];
        for i in 0..6 {
            for j in 0..6 {
                a[i * n + j] = big[i * 6 + j];
            }
        }
        for i in 0..5 {
            for j in 0..5 {
                a[(6 + i) * n + 6 + j] = 1e-160 * small[i * 5 + j];
            }
        }
        let (ev, _) = symmetric_eigen_householder(&a, n);
        assert!(ev.iter().all(|v| v.is_finite()), "{ev:?}");
        let (mut eb, _) = symmetric_eigen(&big, 6);
        let (mut es, _) = symmetric_eigen(&small, 5);
        let mut got = ev.clone();
        got.sort_by(f64::total_cmp);
        eb.sort_by(f64::total_cmp);
        es.sort_by(f64::total_cmp);
        // normwise backward stable (the QL splits at ε‖T‖): the tiny block's eigenvalues are right
        // to ε‖A‖ absolutely, not relatively — all the null-space test and the evidence need
        let norm = eb.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (g, w) in got[..5].iter().zip(&es) {
            assert!((g - 1e-160 * w).abs() < 1e-14 * norm, "{g:e} vs {:e}", 1e-160 * w);
        }
        for (g, w) in got[5..].iter().zip(&eb) {
            assert!((g - w).abs() < 1e-12 * w.abs());
        }
    }
}
