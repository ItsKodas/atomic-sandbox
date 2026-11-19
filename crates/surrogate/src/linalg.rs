//! Dense f64 linear algebra for the fit: Cholesky factorisation, the triangular solves the ridge
//! system and the predictive variance need, log-determinant, inverse. Hand-written; the
//! large-basis fit's O(n³) factorisations, inverse and spectra run on faer instead (`dense`,
//! CLAUDE.md rule 2 as of 2026-11-16: a library where it beats ours). All matrices are square, row-major (`a[i * n + j]`), and a
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

/// [`cholesky_quad_form`] of every vector in `vs` (each of length n), bit for bit the one-at-a-time
/// values: each vector's forward substitution subtracts the same products in the same order as
/// [`forward_substitute`] (Rust never fuses a multiply and an add), and its squares are summed as
/// there. What changes is only how the arithmetic is scheduled — up to [`QUAD_LANES`] vectors side
/// by side and [`QUAD_ROWS`] rows of L at a time, so L is read once per lane group and the
/// independent subtraction chains overlap their latencies (the predictive variance's 1 + 3N rows
/// per step, 2026-11-17: 7 serial substitutions of Q′ 1 902 cost 6–7 ms).
pub fn cholesky_quad_forms(l: &[f64], n: usize, vs: &[&[f64]]) -> Vec<f64> {
    debug_assert_eq!(l.len(), n * n);
    let mut out = Vec::with_capacity(vs.len());
    let mut x = vec![[0.0f64; QUAD_LANES]; n];
    for group in vs.chunks(QUAD_LANES) {
        // the vectors interleaved, lane r = vector r of the group; unused lanes stay zero (their
        // substitution gives zeros, read by nobody)
        for (k, xk) in x.iter_mut().enumerate() {
            *xk = [0.0; QUAD_LANES];
            for (r, v) in group.iter().enumerate() {
                debug_assert_eq!(v.len(), n);
                xk[r] = v[k];
            }
        }
        forward_substitute_lanes(l, n, &mut x);
        for r in 0..group.len() {
            out.push(x.iter().map(|xk| xk[r] * xk[r]).sum());
        }
    }
    out
}

/// Vectors substituted side by side in [`cholesky_quad_forms`].
pub const QUAD_LANES: usize = 8;
/// Rows of L substituted together in [`cholesky_quad_forms`].
pub const QUAD_ROWS: usize = 4;

/// [`forward_substitute`] of `QUAD_LANES` interleaved right-hand sides (`x[k][r]`), each lane's
/// arithmetic exactly the single solve's: row i's sum starts at b_i and subtracts l_ik·x_k for
/// k = 0, 1, …, i − 1 in that order, then divides by l_ii. `QUAD_ROWS` rows go together: their
/// subtractions over the solved prefix k < i₀ are independent of each other, and the block's own
/// corner (k = i₀ … i − 1) follows in order.
fn forward_substitute_lanes(l: &[f64], n: usize, x: &mut [[f64; QUAD_LANES]]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU has AVX2 (checked just now). The kernel is the same Rust with wider
            // registers: separate multiplies and subtractions (no FMA is enabled, and Rust never
            // contracts), so every lane's numbers are the portable kernel's bit for bit.
            unsafe { forward_substitute_lanes_avx2(l, n, x) };
            return;
        }
    }
    forward_substitute_lanes_kernel(l, n, x)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn forward_substitute_lanes_avx2(l: &[f64], n: usize, x: &mut [[f64; QUAD_LANES]]) {
    forward_substitute_lanes_kernel(l, n, x)
}

#[inline(always)]
fn forward_substitute_lanes_kernel(l: &[f64], n: usize, x: &mut [[f64; QUAD_LANES]]) {
    let mut i0 = 0;
    while i0 < n {
        let h = (n - i0).min(QUAD_ROWS);
        let mut s = [[0.0f64; QUAD_LANES]; QUAD_ROWS];
        s[..h].copy_from_slice(&x[i0..i0 + h]);
        if h == QUAD_ROWS {
            let row = |j: usize| &l[(i0 + j) * n..(i0 + j) * n + i0];
            let [mut s0, mut s1, mut s2, mut s3] = s;
            for ((((a, b), c), d), xk) in row(0).iter().zip(row(1)).zip(row(2)).zip(row(3)).zip(&x[..i0]) {
                for r in 0..QUAD_LANES {
                    s0[r] -= a * xk[r];
                    s1[r] -= b * xk[r];
                    s2[r] -= c * xk[r];
                    s3[r] -= d * xk[r];
                }
            }
            s = [s0, s1, s2, s3];
        } else {
            for (j, sj) in s[..h].iter_mut().enumerate() {
                let row = &l[(i0 + j) * n..(i0 + j) * n + i0];
                for (lk, xk) in row.iter().zip(&x[..i0]) {
                    for r in 0..QUAD_LANES {
                        sj[r] -= lk * xk[r];
                    }
                }
            }
        }
        for j in 0..h {
            let i = i0 + j;
            let mut sj = s[j];
            for k in i0..i {
                let lk = l[i * n + k];
                let xk = x[k];
                for r in 0..QUAD_LANES {
                    sj[r] -= lk * xk[r];
                }
            }
            let d = l[i * n + i];
            for r in 0..QUAD_LANES {
                x[i][r] = sj[r] / d;
            }
        }
        i0 += h;
    }
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

/// ---- the predictive variance on a support (spec 2026-11-20-variance-support-design.md) ----
//
// For a lower factor L of A = L Lᵀ (n × n) and an ascending set S of m indices (N the rest),
// every v supported on S has
//
//   vᵀA⁻¹v = v_Sᵀ G v_S,   G = (A⁻¹)_SS = XᵀX,   X = L⁻¹E_S (n × m; column j is L⁻¹ e_{S_j}),
//
// and with G = RᵀR (R upper triangular) the value is the sum of squares ‖R v_S‖², at m²/2 per
// vector instead of the full substitution's n²/2.
//
// **What F is (review M1).** With W = R⁻¹ and J the reversal of S, F = J W J is lower triangular
// with a positive diagonal and F Fᵀ = J G⁻¹ J = J C J, C = A_SS − A_SN A_NN⁻¹ A_NS the Schur
// complement. So F is the Cholesky factor of J C J: the trailing |S| × |S| block of the Cholesky
// factor of A with its indices ordered (N, reversed S). The support forms ‖F⁻¹ J v_S‖² are then
// the full path's own forward substitution through a reordered exact factor of the same A, whose
// N block meets a zero right-hand side (y_N = 0 exactly) and drops out. F is that block computed
// in double-double and rounded once to f64; the forms are in the full path's accuracy class.
// That is a statement of class, not a derived bound: the agreement is **measured** (the stds
// within 4.4e-13 relative of the full path's in the `model` tests, the forms within 2× the full
// substitution's own error on a posterior-shaped factor in the tests below), and every new factor
// is self-checked against the full path at build (`model::SupportCache`, bound 1e-10).
//
// **Why double-double.** Built in f64, F loses digits to cancellation: the unit columns L⁻¹e_s
// carry the prior's variance (of order 1/λ) while a query in a direction the data determine has
// a variance many orders smaller, and C is a cancelling difference of A_SS. Measured on fitted
// factors against a double-double reference (2026-11-20): the full substitution ≤ 2.2e-13
// relative, ‖R v_S‖² with R from the f64 QR of X (the spec's first form) up to 3.5e-11, C's factor
// by f64 Householder or Givens elimination of L up to 1.8e-10 and 5.9e-12. X, G, R and W are
// therefore computed in double-double arithmetic (≈ 2⁻¹⁰⁶; Dekker's products and Knuth's sums,
// no FMA, so the result is the same on every CPU and for every thread count).
/// Double-double arithmetic for [`support_factor`]: a value hi + lo with |lo| ≤ ulp(hi)/2.
/// Error-free transformations by Knuth (sum) and Dekker/Veltkamp (product); Rust never contracts
/// a multiply and an add, so every operation is the same on every CPU.
mod dd {
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Dd {
        pub hi: f64,
        pub lo: f64,
    }

    /// s + e = a + b exactly.
    #[inline(always)]
    pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let bb = s - a;
        (s, (a - (s - bb)) + (b - bb))
    }

    /// s + e = a + b exactly, for |a| ≥ |b| (or a = 0).
    #[inline(always)]
    fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        (s, b - (s - a))
    }

    /// Veltkamp's split: hi + lo = a with 26-bit halves.
    #[inline(always)]
    pub fn split(a: f64) -> (f64, f64) {
        let t = 134_217_729.0 * a; // 2²⁷ + 1
        let hi = t - (t - a);
        (hi, a - hi)
    }

    /// p + e = a·b exactly (Dekker), with b's split given.
    #[inline(always)]
    pub fn two_prod_split(a: f64, b: f64, (bh, bl): (f64, f64)) -> (f64, f64) {
        let p = a * b;
        let (ah, al) = split(a);
        (p, ((ah * bh - p) + ah * bl + al * bh) + al * bl)
    }

    #[inline(always)]
    pub fn two_prod(a: f64, b: f64) -> (f64, f64) {
        two_prod_split(a, b, split(b))
    }

    impl Dd {
        pub const fn from(a: f64) -> Dd {
            Dd { hi: a, lo: 0.0 }
        }

        /// A normalised pair from s + e (|e| small against s, or s = 0).
        #[inline(always)]
        pub fn renorm(s: f64, e: f64) -> Dd {
            let (hi, lo) = two_sum(s, e);
            Dd { hi, lo }
        }

        #[inline(always)]
        pub fn add(self, b: Dd) -> Dd {
            let (s, e) = two_sum(self.hi, b.hi);
            let (t, f) = two_sum(self.lo, b.lo);
            let (s, e) = fast_two_sum(s, e + t);
            let (hi, lo) = fast_two_sum(s, e + f);
            Dd { hi, lo }
        }

        #[inline(always)]
        pub fn neg(self) -> Dd {
            Dd { hi: -self.hi, lo: -self.lo }
        }

        #[inline(always)]
        pub fn sub(self, b: Dd) -> Dd {
            self.add(b.neg())
        }

        #[cfg(test)]
        #[inline(always)]
        pub fn mul(self, b: Dd) -> Dd {
            let (p, e) = two_prod(self.hi, b.hi);
            let (hi, lo) = fast_two_sum(p, e + (self.hi * b.lo + self.lo * b.hi));
            Dd { hi, lo }
        }

        #[inline(always)]
        pub fn mul_f(self, b: f64) -> Dd {
            let (p, e) = two_prod(self.hi, b);
            let (hi, lo) = fast_two_sum(p, e + self.lo * b);
            Dd { hi, lo }
        }

        /// self / b, two quotient digits and a correction.
        pub fn div(self, b: Dd) -> Dd {
            let q1 = self.hi / b.hi;
            let r = self.sub(b.mul_f(q1));
            let q2 = r.hi / b.hi;
            let r = r.sub(b.mul_f(q2));
            let q3 = r.hi / b.hi;
            let (q1, q2) = fast_two_sum(q1, q2);
            Dd { hi: q1, lo: q2 }.add(Dd::from(q3))
        }

        /// √self for self > 0: one Newton step on the f64 root.
        pub fn sqrt(self) -> Dd {
            let q = self.hi.sqrt();
            let (p, e) = two_prod(q, q);
            let r = self.sub(Dd { hi: p, lo: e });
            let (hi, lo) = fast_two_sum(q, r.hi / (2.0 * q));
            Dd { hi, lo }
        }
    }

    /// Σ a_k (x_k.hi + x_k.lo) as if in twice the precision (Ogita, Rump & Oishi's Dot2, four
    /// independent partial sums): the products are exact, the sums' rounding is carried. (The
    /// tests' double-double substitution; `unit_solves_lanes` inlines the same steps per lane.)
    #[cfg(test)]
    pub fn dot2_f_dd(a: &[f64], xh: &[f64], xl: &[f64]) -> Dd {
        debug_assert!(a.len() == xh.len() && a.len() == xl.len());
        let mut s = [0.0f64; 4];
        let mut c = [0.0f64; 4];
        let chunks = a.len() / 4;
        for q in 0..chunks {
            for r in 0..4 {
                let k = 4 * q + r;
                let (p, e) = two_prod(a[k], xh[k]);
                let (t, sigma) = two_sum(s[r], p);
                s[r] = t;
                c[r] += e + sigma + a[k] * xl[k];
            }
        }
        for k in 4 * chunks..a.len() {
            let (p, e) = two_prod(a[k], xh[k]);
            let (t, sigma) = two_sum(s[0], p);
            s[0] = t;
            c[0] += e + sigma + a[k] * xl[k];
        }
        let mut acc = Dd::default();
        for r in 0..4 {
            acc = acc.add(Dd::renorm(s[r], c[r]));
        }
        acc
    }

    /// Σ x_k y_k for double-double x, y (Dot2 on the exact hi·hi products, the cross terms
    /// carried in the correction).
    pub fn dot2_dd_dd(xh: &[f64], xl: &[f64], yh: &[f64], yl: &[f64]) -> Dd {
        let mut s = [0.0f64; 4];
        let mut c = [0.0f64; 4];
        let n = xh.len();
        let chunks = n / 4;
        for q in 0..chunks {
            for r in 0..4 {
                let k = 4 * q + r;
                let (p, e) = two_prod(xh[k], yh[k]);
                let (t, sigma) = two_sum(s[r], p);
                s[r] = t;
                c[r] += e + sigma + (xh[k] * yl[k] + xl[k] * yh[k]);
            }
        }
        for k in 4 * chunks..n {
            let (p, e) = two_prod(xh[k], yh[k]);
            let (t, sigma) = two_sum(s[0], p);
            s[0] = t;
            c[0] += e + sigma + (xh[k] * yl[k] + xl[k] * yh[k]);
        }
        let mut acc = Dd::default();
        for r in 0..4 {
            acc = acc.add(Dd::renorm(s[r], c[r]));
        }
        acc
    }
}

/// Unit-vector substitutions solved side by side in [`support_factor`].
const UNIT_LANES: usize = 4;

/// Double-double L x_r = e_{cols[r]} for up to [`UNIT_LANES`] ascending columns at once, written
/// column-major into `xh`/`xl` (column r at `r * n`). Every lane runs Ogita–Rump–Oishi's Dot2 over
/// k = cols[0] … i − 1 (a lane is exactly zero above its own column: its products are 0 and leave
/// its sums alone), sharing one split of l_ik; each x_k's high part is split once when solved.
fn unit_solves_lanes(l: &[f64], n: usize, cols: &[usize], xh: &mut [f64], xl: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU has AVX2 (checked just now). The same Rust with wider registers: no
            // FMA is enabled and Rust never contracts, so the numbers are the portable kernel's.
            unsafe { unit_solves_lanes_avx2(l, n, cols, xh, xl) };
            return;
        }
    }
    unit_solves_lanes_kernel(l, n, cols, xh, xl)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn unit_solves_lanes_avx2(l: &[f64], n: usize, cols: &[usize], xh: &mut [f64], xl: &mut [f64]) {
    unit_solves_lanes_kernel(l, n, cols, xh, xl)
}

#[inline(always)]
fn unit_solves_lanes_kernel(l: &[f64], n: usize, cols: &[usize], xh: &mut [f64], xl: &mut [f64]) {
    use dd::{split, Dd};
    const W: usize = UNIT_LANES;
    debug_assert!(!cols.is_empty() && cols.len() <= W);
    let mut col = [usize::MAX; W];
    col[..cols.len()].copy_from_slice(cols);
    let c0 = cols[0];
    // x_k per lane: hi, lo, and the split halves of hi
    let mut hi = vec![[0.0f64; W]; n];
    let mut lo = vec![[0.0f64; W]; n];
    let mut hh = vec![[0.0f64; W]; n];
    let mut hl = vec![[0.0f64; W]; n];
    for i in c0..n {
        let row = &l[i * n..(i + 1) * n];
        let mut s = [0.0f64; W];
        let mut c = [0.0f64; W];
        for k in c0..i {
            let a = row[k];
            let (ah, al) = split(a);
            let (xh_k, xl_k, bh, bl) = (&hi[k], &lo[k], &hh[k], &hl[k]);
            for r in 0..W {
                let p = a * xh_k[r];
                let e = ((ah * bh[r] - p) + ah * bl[r] + al * bh[r]) + al * bl[r];
                let t = s[r] + p;
                let bb = t - s[r];
                let sigma = (s[r] - (t - bb)) + (p - bb);
                s[r] = t;
                c[r] += e + sigma + a * xl_k[r];
            }
        }
        let d = Dd::from(row[i]);
        for r in 0..W {
            let b = if col[r] == i { 1.0 } else { 0.0 };
            let x = Dd::from(b).sub(Dd::renorm(s[r], c[r])).div(d);
            let (h1, h2) = split(x.hi);
            hi[i][r] = x.hi;
            lo[i][r] = x.lo;
            hh[i][r] = h1;
            hl[i][r] = h2;
        }
    }
    for r in 0..cols.len() {
        for i in c0..n {
            xh[r * n + i] = hi[i][r];
            xl[r * n + i] = lo[i][r];
        }
    }
}

/// The m × m lower factor F (row-major, in the REVERSED order of `support`) with
/// vᵀ(L Lᵀ)⁻¹v = ‖F⁻¹ (J v_S)‖² for every v supported on the ascending index set S = `support`,
/// J v_S = (v_{S_{m−1}}, …, v_{S_0}) — so [`cholesky_quad_forms`]`(F, m, …)` of the reversed
/// gathered vectors gives the quadratic forms (section comment above: F is the trailing
/// Cholesky block of A ordered (N, reversed S)). Built in double-double:
/// X = L⁻¹E_S by unit-column substitutions started at row S_j (the columns in parallel),
/// G = XᵀX, G = L_G L_Gᵀ, and F_ab = (L_G⁻¹)_{m−1−b, m−1−a} (F = J L_G⁻ᵀ J), rounded once to f64.
/// Σ_j (n − S_j)²/2 + n m²/2 double-double multiply-adds and O(m³).
///
/// `None` when L is not a usable factor (a pivot of G that is not positive, or a value that is
/// not finite: a zero or non-finite entry on L's diagonal); the caller then takes the full path,
/// whose numbers are what L gives.
///
/// # Panics
/// If `support` is not ascending within 0..n.
pub fn support_factor(l: &[f64], n: usize, support: &[usize]) -> Option<Vec<f64>> {
    use dd::Dd;
    use rayon::prelude::*;
    debug_assert_eq!(l.len(), n * n);
    assert!(support.windows(2).all(|w| w[0] < w[1]), "support_factor: support is not ascending");
    assert!(support.last().is_none_or(|&s| s < n), "support_factor: support outside 0..{n}");
    let m = support.len();
    if m == 0 {
        return Some(Vec::new());
    }
    // X = L⁻¹E_S, column-major hi and lo parts; column j is zero above row S_j. Four columns
    // at a time (`unit_solves_lanes`), the groups in parallel.
    let mut xh = vec![0.0; n * m];
    let mut xl = vec![0.0; n * m];
    xh.par_chunks_mut(UNIT_LANES * n).zip(xl.par_chunks_mut(UNIT_LANES * n)).zip(support.par_chunks(UNIT_LANES)).for_each(|((gh, gl), cols)| {
        unit_solves_lanes(l, n, cols, gh, gl);
    });
    // G = XᵀX (lower triangle, hi and lo parts), rows in parallel
    let mut gh = vec![0.0; m * m];
    let mut gl = vec![0.0; m * m];
    gh.par_chunks_mut(m).zip(gl.par_chunks_mut(m)).enumerate().for_each(|(a, (rh, rl))| {
        for b in 0..=a {
            let from = support[a].max(support[b]);
            let (ra, rb) = (a * n + from..(a + 1) * n, b * n + from..(b + 1) * n);
            let v = dd::dot2_dd_dd(&xh[ra.clone()], &xl[ra], &xh[rb.clone()], &xl[rb]);
            rh[b] = v.hi;
            rl[b] = v.lo;
        }
    });
    drop((xh, xl));
    // G = L_G L_Gᵀ in place (lower; Cholesky–Crout: column j's pivot from row j's prefix, then
    // the rows below independently, in parallel)
    for j in 0..m {
        let d = Dd { hi: gh[j * m + j], lo: gl[j * m + j] }.sub(dd::dot2_dd_dd(&gh[j * m..j * m + j], &gl[j * m..j * m + j], &gh[j * m..j * m + j], &gl[j * m..j * m + j]));
        if !(d.hi > 0.0 && d.hi.is_finite()) {
            return None;
        }
        let ljj = d.sqrt();
        gh[j * m + j] = ljj.hi;
        gl[j * m + j] = ljj.lo;
        let (head_h, tail_h) = gh.split_at_mut((j + 1) * m);
        let (head_l, tail_l) = gl.split_at_mut((j + 1) * m);
        let (pj_h, pj_l) = (&head_h[j * m..j * m + j], &head_l[j * m..j * m + j]);
        let column = |(rh, rl): (&mut [f64], &mut [f64])| {
            let s = Dd { hi: rh[j], lo: rl[j] }.sub(dd::dot2_dd_dd(&rh[..j], &rl[..j], pj_h, pj_l)).div(ljj);
            rh[j] = s.hi;
            rl[j] = s.lo;
        };
        if (m - j) * j > 4096 {
            tail_h.par_chunks_mut(m).zip(tail_l.par_chunks_mut(m)).for_each(column);
        } else {
            tail_h.chunks_mut(m).zip(tail_l.chunks_mut(m)).for_each(column);
        }
    }
    // Y = L_G⁻¹ (lower), column-major: unit-vector substitutions, the columns in parallel
    let mut yh = vec![0.0; m * m];
    let mut yl = vec![0.0; m * m];
    yh.par_chunks_mut(m).zip(yl.par_chunks_mut(m)).enumerate().for_each(|(c, (ch, cl))| {
        let y = Dd::from(1.0).div(Dd { hi: gh[c * m + c], lo: gl[c * m + c] });
        ch[c] = y.hi;
        cl[c] = y.lo;
        for i in c + 1..m {
            let s = dd::dot2_dd_dd(&gh[i * m + c..i * m + i], &gl[i * m + c..i * m + i], &ch[c..i], &cl[c..i]);
            let y = s.neg().div(Dd { hi: gh[i * m + i], lo: gl[i * m + i] });
            ch[i] = y.hi;
            cl[i] = y.lo;
        }
    });
    // F = J Yᵀ J: F_ab = Y_{m−1−b, m−1−a} (Y column-major: column m−1−a, row m−1−b), lower
    let mut f = vec![0.0; m * m];
    for a in 0..m {
        for b in 0..=a {
            f[a * m + b] = yh[(m - 1 - a) * m + (m - 1 - b)];
        }
    }
    f.iter().all(|v| v.is_finite()).then_some(f)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// The batched quadratic forms' speed at the collection's size (Q′ 1 902, the 7 rows of an H₂
    /// variance step) against one substitution per row; reported, bit-equality asserted.
    /// `cargo test -p surrogate --release --lib quad_forms_speed -- --ignored --nocapture`
    #[test]
    #[ignore = "a measurement"]
    fn quad_forms_speed() {
        let mut rng = StdRng::seed_from_u64(3);
        let n = 1902;
        // a well-conditioned lower factor with a positive diagonal
        let mut l = vec![0.0; n * n];
        for i in 0..n {
            for k in 0..i {
                l[i * n + k] = rng.gen_range(-1.0..1.0) / n as f64;
            }
            l[i * n + i] = 1.0 + rng.gen_range(0.0..1.0);
        }
        let vs: Vec<Vec<f64>> = (0..7).map(|_| (0..n).map(|k| if k < 634 { rng.gen_range(-1.0..1.0) } else { 0.0 }).collect()).collect();
        let refs: Vec<&[f64]> = vs.iter().map(|v| &v[..]).collect();
        let reps = 20;
        let t = std::time::Instant::now();
        let mut one = Vec::new();
        for _ in 0..reps {
            one = vs.iter().map(|v| cholesky_quad_form(&l, n, &mut v.clone())).collect();
        }
        let t_one = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
        let t = std::time::Instant::now();
        let mut many = Vec::new();
        for _ in 0..reps {
            many = cholesky_quad_forms(&l, n, &refs);
        }
        let t_many = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
        println!("7 quadratic forms at n = {n}: one at a time {t_one:.3} ms, batched {t_many:.3} ms (avx2 {})", std::arch::is_x86_feature_detected!("avx2"));
        assert!(one.iter().zip(&many).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    /// The batched quadratic forms are the one-at-a-time ones bit for bit, for every batch size
    /// around the lane and row widths and with zero entries in the vectors (the query rows'
    /// untouched species blocks).
    #[test]
    fn batched_quad_forms_are_the_single_ones_bit_for_bit() {
        let mut rng = StdRng::seed_from_u64(17);
        for n in [1, 2, 3, 4, 5, 6, 9, 37, 64, 66] {
            let a: Vec<f64> = (0..n * n).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let mut g = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    g[i * n + j] = (0..n).map(|k| a[i * n + k] * a[j * n + k]).sum::<f64>() + if i == j { 0.1 } else { 0.0 };
                }
            }
            cholesky_in_place(&mut g, n).unwrap();
            for m in [1, 2, 7, 8, 9, 17] {
                let vs: Vec<Vec<f64>> = (0..m).map(|_| (0..n).map(|_| if rng.gen_bool(0.3) { 0.0 } else { rng.gen_range(-3.0..3.0) }).collect()).collect();
                let refs: Vec<&[f64]> = vs.iter().map(|v| &v[..]).collect();
                let got = cholesky_quad_forms(&g, n, &refs);
                for (v, q) in vs.iter().zip(&got) {
                    let want = cholesky_quad_form(&g, n, &mut v.clone());
                    assert_eq!(q.to_bits(), want.to_bits(), "n {n} m {m}: {q:e} vs {want:e}");
                }
                // the portable kernel itself, whatever the CPU (the dispatch takes AVX2 when present)
                if m <= QUAD_LANES {
                    let mut x = vec![[0.0f64; QUAD_LANES]; n];
                    for (k, xk) in x.iter_mut().enumerate() {
                        for (r, v) in vs.iter().enumerate() {
                            xk[r] = v[k];
                        }
                    }
                    forward_substitute_lanes_kernel(&g, n, &mut x);
                    for (r, q) in got.iter().enumerate() {
                        let portable: f64 = x.iter().map(|xk| xk[r] * xk[r]).sum();
                        assert_eq!(portable.to_bits(), q.to_bits(), "portable kernel, n {n} m {m}");
                    }
                }
            }
        }
    }

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
    /// ‖L⁻¹v‖² in double-double (L and v exact as given): the reference of the support tests,
    /// here and in `model`.
    pub(crate) fn quad_form_dd(l: &[f64], n: usize, v: &[f64]) -> f64 {
        use super::dd::{dot2_f_dd, Dd};
        let mut yh = vec![0.0; n];
        let mut yl = vec![0.0; n];
        for i in 0..n {
            let s = dot2_f_dd(&l[i * n..i * n + i], &yh[..i], &yl[..i]);
            let y = Dd::from(v[i]).sub(s).div(Dd::from(l[i * n + i]));
            yh[i] = y.hi;
            yl[i] = y.lo;
        }
        let mut acc = Dd::default();
        for i in 0..n {
            let y = Dd { hi: yh[i], lo: yl[i] };
            acc = acc.add(y.mul(y));
        }
        acc.hi
    }

    /// A random SPD factor L = chol(BBᵀ/n + 0.1 I).
    fn random_factor(rng: &mut StdRng, n: usize) -> Vec<f64> {
        let a: Vec<f64> = (0..n * n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let mut g = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                g[i * n + j] = (0..n).map(|k| a[i * n + k] * a[j * n + k]).sum::<f64>() / n as f64 + if i == j { 0.1 } else { 0.0 };
            }
        }
        cholesky_in_place(&mut g, n).unwrap();
        g
    }

    /// The support path's quadratic forms for vectors on `support`: gathered in reverse order
    /// through `support_factor`.
    fn support_forms(l: &[f64], n: usize, support: &[usize], vs: &[Vec<f64>]) -> Vec<f64> {
        let f = support_factor(l, n, support).unwrap();
        let rev: Vec<Vec<f64>> = vs.iter().map(|v| support.iter().rev().map(|&k| v[k]).collect()).collect();
        cholesky_quad_forms(&f, support.len(), &rev.iter().map(|v| &v[..]).collect::<Vec<_>>())
    }

    /// The double-double operations carry about 106 bits: 1/3·3, √2², a cancelling product.
    #[test]
    fn double_double_arithmetic_carries_twice_the_bits() {
        use super::dd::{two_prod, Dd};
        let third = Dd::from(1.0).div(Dd::from(3.0));
        let r = third.mul_f(3.0).sub(Dd::from(1.0));
        assert!((r.hi + r.lo).abs() < 1e-31, "1/3·3 − 1 = {:e}", r.hi);
        let s = Dd::from(2.0).sqrt();
        let r = s.mul(s).sub(Dd::from(2.0));
        assert!((r.hi + r.lo).abs() < 1e-31, "√2² − 2 = {:e}", r.hi);
        // (1 + 2⁻³⁰)² = 1 + 2⁻²⁹ + 2⁻⁶⁰: the product is exact in hi + lo
        let a = 1.0 + 2f64.powi(-30);
        let (p, e) = two_prod(a, a);
        assert_eq!(p, 1.0 + 2f64.powi(-29));
        assert_eq!(e, 2f64.powi(-60));
    }

    /// For v supported on S, ‖F⁻¹ J v_S‖² equals the full ‖L⁻¹v‖² to rounding, for supports from
    /// one index to all of them, the double-double reference within 1e-13 of both.
    #[test]
    fn support_factor_gives_the_full_quadratic_forms() {
        let mut rng = StdRng::seed_from_u64(37);
        let mut worst = (0.0f64, 0.0f64);
        for n in [1, 3, 37, 120] {
            let l = random_factor(&mut rng, n);
            for p in [0.0, 0.05, 0.3, 1.0] {
                let mut support: Vec<usize> = (0..n).filter(|_| rng.gen_bool(p)).collect();
                if support.is_empty() {
                    support.push(n / 2);
                }
                let vs: Vec<Vec<f64>> = (0..9)
                    .map(|_| (0..n).map(|k| if support.contains(&k) && rng.gen_bool(0.8) { rng.gen_range(-3.0..3.0) } else { 0.0 }).collect())
                    .collect();
                let got = support_forms(&l, n, &support, &vs);
                for (v, g) in vs.iter().zip(&got) {
                    let full = cholesky_quad_form(&l, n, &mut v.clone());
                    let exact = quad_form_dd(&l, n, v);
                    if exact == 0.0 {
                        assert_eq!((*g, full), (0.0, 0.0));
                        continue;
                    }
                    let (es, ef) = ((g - exact).abs() / exact, (full - exact).abs() / exact);
                    worst = (worst.0.max(es), worst.1.max(ef));
                    assert!(es < 1e-13 && ef < 1e-13, "n {n} |S| {}: support {g:e}, full {full:e}, exact {exact:e}", support.len());
                }
            }
        }
        println!("support factor, random factors: worst relative error support {:.2e}, full {:.2e}", worst.0, worst.1);
    }

    /// A posterior-shaped factor (prior λ = 1e-2, strong data rows on part of the columns,
    /// σ² = 1e-8) with queries in the data's span. A representation that multiplies v_S by a
    /// stored factor of (A⁻¹)_SS rounds at ε·Σ_k |v_k|·√(A⁻¹)_kk against the result √(vᵀA⁻¹v):
    /// that amplification is asserted to exceed 1e3 here (the regime where the f64 QR of L⁻¹E_S
    /// lost four digits on fitted factors). The support path's worst error against the
    /// double-double value stays within twice the full substitution's (both are triangular
    /// solves of componentwise-accurate factors).
    #[test]
    fn support_factor_keeps_its_accuracy_where_the_products_cancel() {
        let mut rng = StdRng::seed_from_u64(59);
        let n = 90;
        let mut worst = (0.0f64, 0.0f64);
        let mut amplification = 0.0f64;
        for trial in 0..4 {
            let support: Vec<usize> = (0..n).filter(|k| k % 3 == trial % 3 || *k < 6).collect();
            // data rows: smooth profiles over the support (correlated columns), a few elsewhere
            let rows: Vec<Vec<f64>> = (0..40)
                .map(|r| {
                    let w = rng.gen_range(0.5..3.0);
                    (0..n).map(|k| if support.contains(&k) || r % 5 == 0 { (w * k as f64 / n as f64 + r as f64).sin() } else { 0.0 }).collect()
                })
                .collect();
            let mut a = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    a[i * n + j] = rows.iter().map(|g| g[i] * g[j]).sum::<f64>() / 1e-8 + if i == j { 1e-2 } else { 0.0 };
                }
            }
            cholesky_in_place(&mut a, n).unwrap();
            // queries: combinations of the data rows restricted to S (well determined) and random
            // vectors on S (prior dominated)
            let mut vs: Vec<Vec<f64>> = (0..6)
                .map(|_| {
                    let c: Vec<f64> = (0..rows.len()).map(|_| rng.gen_range(-1.0..1.0)).collect();
                    (0..n).map(|k| if support.contains(&k) { rows.iter().zip(&c).map(|(g, ci)| g[k] * ci).sum() } else { 0.0 }).collect()
                })
                .collect();
            vs.push((0..n).map(|k| if support.contains(&k) { rng.gen_range(-1.0..1.0) } else { 0.0 }).collect());
            let prior_std: Vec<f64> = (0..n)
                .map(|k| {
                    let mut e = vec![0.0; n];
                    e[k] = 1.0;
                    cholesky_quad_form(&a, n, &mut e).sqrt()
                })
                .collect();
            let got = support_forms(&a, n, &support, &vs);
            for (v, g) in vs.iter().zip(&got) {
                let full = cholesky_quad_form(&a, n, &mut v.clone());
                let exact = quad_form_dd(&a, n, v);
                let (es, ef) = ((g - exact).abs() / exact, (full - exact).abs() / exact);
                worst = (worst.0.max(es), worst.1.max(ef));
                let amp = v.iter().zip(&prior_std).map(|(vk, sk)| vk.abs() * sk).sum::<f64>() / exact.sqrt();
                amplification = amplification.max(amp);
            }
        }
        println!(
            "support factor, posterior-shaped: worst relative error support {:.2e}, full {:.2e}; largest multiply amplification {amplification:.2e}",
            worst.0, worst.1
        );
        assert!(amplification > 1e3, "the queries cancel: amplification {amplification:.2e}");
        assert!(worst.0 <= 2.0 * worst.1, "support {:.2e} vs full {:.2e}", worst.0, worst.1);
    }

    /// The support factor's build cost at the collection's size (n = 1 902) for supports of an H₂,
    /// an H₂O and a near-Q′/2 scene. `cargo test -p surrogate --release --lib support_factor_build_cost -- --ignored --nocapture`
    #[test]
    #[ignore = "a measurement"]
    fn support_factor_build_cost() {
        let mut rng = StdRng::seed_from_u64(3);
        let n = 1902;
        let mut l = vec![0.0; n * n];
        for i in 0..n {
            for k in 0..i {
                l[i * n + k] = rng.gen_range(-1.0..1.0) / n as f64;
            }
            l[i * n + i] = 1.0 + rng.gen_range(0.0..1.0);
        }
        for m in [52, 282, 950] {
            let support: Vec<usize> = (0..m).map(|k| k * 2).collect();
            let t = std::time::Instant::now();
            let f = support_factor(&l, n, &support).unwrap();
            println!("support_factor: |S| {m}: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
            assert_eq!(f.len(), m * m);
        }
    }
}
