//! The large-basis fit's O(n³) dense kernels on [faer](https://codeberg.org/sarah-quinones/faer)
//! (MIT, pure Rust, f64; pinned `=0.24.4`; CLAUDE.md rule 2, the refit speed-up of 2026-11-17):
//! the Cholesky factorisation, the SPD inverse from the factor, the Householder
//! tridiagonalisation with Qᵀ applied to one vector, and the symmetric eigendecomposition — all
//! blocked (BLAS-3 where the algorithm allows) and threaded, where `linalg`'s are unblocked loops
//! on one thread. Every routine takes and returns `linalg`'s layout (square, row-major,
//! `a[i * n + j]`; a lower factor with its strict upper triangle zeroed), so a caller swaps one
//! call for another.
//!
//! **Threads.** faer runs on a rayon pool of [`crate::threads::worker_threads`] threads (the
//! user's CPU share), each lowered to below-normal priority as it starts
//! ([`crate::threads::lower_priority`]), one pool per thread count, built on first use.
//! **The same numbers on any thread count**: faer splits its work by the `Par::Rayon(k)` it is
//! given, not by the pool it runs on (the tridiagonalisation sums k partial columns, the matrix
//! products tile the output by k), so the split is a constant [`SPLIT`] and the pool only decides
//! how many of those pieces run at once — the fit's thread-count invariance
//! (`fit::tests::the_fit_is_the_same_on_one_thread_and_on_the_capped_count`) holds through faer.
//! faer picks its SIMD kernels at run time, so its last bits can differ between CPUs of different
//! instruction sets (AVX2 against AVX-512); on one machine every call is reproducible.
//!
//! **Accuracy.** The same algorithms as `linalg`'s (unpivoted Cholesky, Householder
//! tridiagonalisation; the symmetric eigensolver is faer's divide and conquer on the tridiagonal
//! instead of QL), with the same normwise backward errors — the results differ from `linalg`'s in
//! the last bits (pinned in the tests below against the `linalg` kernels).

use crate::linalg::LinalgError;
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::cholesky::llt;
use faer::linalg::evd;
use faer::linalg::householder;
use faer::{Conj, Mat, Par};
use std::sync::{Arc, Mutex, OnceLock};

/// The fixed work split handed to faer (module doc): the result does not depend on the pool's
/// thread count; 16 = `threads::MAX_THREADS`, so no pool has more threads than pieces.
pub const SPLIT: usize = crate::threads::MAX_THREADS;

fn par() -> Par {
    Par::rayon(SPLIT)
}

/// Runs `f` on the fit's rayon pool of [`crate::threads::worker_threads`] threads (module doc).
pub fn install<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    static POOLS: OnceLock<Mutex<Vec<(usize, Arc<rayon::ThreadPool>)>>> = OnceLock::new();
    let n = crate::threads::worker_threads();
    let pool = {
        let mut pools = POOLS.get_or_init(|| Mutex::new(Vec::new())).lock().unwrap_or_else(|e| e.into_inner());
        match pools.iter().find(|(k, _)| *k == n) {
            Some((_, p)) => p.clone(),
            None => {
                let p = Arc::new(
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(n)
                        .thread_name(|i| format!("surrogate-fit-{i}"))
                        .start_handler(|_| crate::threads::lower_priority())
                        .build()
                        .expect("the fit's thread pool"),
                );
                pools.push((n, p.clone()));
                p
            }
        }
    };
    pool.install(f)
}

/// A faer matrix from the lower triangle of the row-major symmetric `a` (the upper is not read).
fn lower_to_mat(a: &[f64], n: usize) -> Mat<f64> {
    Mat::from_fn(n, n, |i, j| if i >= j { a[i * n + j] } else { 0.0 })
}

/// The lower Cholesky factor L (A = L Lᵀ) of the symmetric positive definite `a` (n×n row-major;
/// only the lower triangle is read): faer's blocked left-looking factorisation, threaded. The
/// strict upper triangle of the result is zero. `Err` with the column of the first pivot that is
/// not positive (or NaN), as [`crate::linalg::cholesky_in_place`].
pub fn cholesky_lower(a: &[f64], n: usize) -> Result<Vec<f64>, LinalgError> {
    assert_eq!(a.len(), n * n, "cholesky_lower: slice is not n×n");
    install(|| {
        let mut m = lower_to_mat(a, n);
        let mut buf = MemBuffer::new(llt::factor::cholesky_in_place_scratch::<f64>(n, par(), Default::default()));
        llt::factor::cholesky_in_place(m.as_mut(), Default::default(), par(), MemStack::new(&mut buf), Default::default())
            .map_err(|llt::factor::LltError::NonPositivePivot { index }| LinalgError::NotPositiveDefinite(index))?;
        let mut l = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                l[i * n + j] = m[(i, j)];
            }
        }
        Ok(l)
    })
}

/// (L Lᵀ)⁻¹, full symmetric row-major, from a lower factor L ([`cholesky_lower`]'s layout): faer's
/// triangular inverse and the triangular product L⁻ᵀL⁻¹, threaded (≈ 2n³/3 flops).
pub fn spd_inverse(l: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(l.len(), n * n, "spd_inverse: slice is not n×n");
    install(|| {
        let lm = lower_to_mat(l, n);
        let mut out = Mat::<f64>::zeros(n, n);
        let mut buf = MemBuffer::new(llt::inverse::inverse_scratch::<f64>(n, par()));
        llt::inverse::inverse(out.as_mut(), lm.as_ref(), par(), MemStack::new(&mut buf));
        // faer writes the lower triangle; the result is symmetric by construction
        let mut inv = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let v = out[(i, j)];
                inv[i * n + j] = v;
                inv[j * n + i] = v;
            }
        }
        inv
    })
}

/// The Householder tridiagonalisation T = Qᵀ A Q of the symmetric `a` (n×n row-major; the lower
/// triangle is read), with `c ← Qᵀ c`: faer's blocked reduction, threaded. Returns T's diagonal
/// `d` and off-diagonal `e` (`e[k]` = T_{k,k+1}, `e[n−1]` = 0) — [`crate::linalg::Tridiagonal`]'s
/// `d`, `e` and `apply_qt` in one call, for [`crate::linalg::tridiagonal_ql`].
pub fn tridiagonal_qtc(a: &[f64], n: usize, c: &mut [f64]) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(a.len(), n * n, "tridiagonal_qtc: slice is not n×n");
    assert_eq!(c.len(), n, "tridiagonal_qtc: c is not of length n");
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    install(|| {
        let mut m = lower_to_mat(a, n);
        let bs = faer::linalg::qr::no_pivoting::factor::recommended_block_size::<f64>(n, n);
        let mut h = Mat::<f64>::zeros(bs, n - 1);
        let mut buf = MemBuffer::new(StackReq::any_of(&[
            evd::tridiag::tridiag_in_place_scratch::<f64>(n, par(), Default::default()),
            householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<f64>(n - 1, bs, 1),
        ]));
        evd::tridiag::tridiag_in_place(m.as_mut(), h.as_mut(), par(), MemStack::new(&mut buf), Default::default());
        if n > 1 {
            let mut cm = Mat::from_fn(n - 1, 1, |i, _| c[i + 1]);
            householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
                m.as_ref().submatrix(1, 0, n - 1, n - 1),
                h.as_ref(),
                Conj::No,
                cm.as_mut(),
                par(),
                MemStack::new(&mut buf),
            );
            for i in 0..n - 1 {
                c[i + 1] = cm[(i, 0)];
            }
        }
        let d: Vec<f64> = (0..n).map(|i| m[(i, i)]).collect();
        let e: Vec<f64> = (0..n).map(|i| if i + 1 < n { m[(i + 1, i)] } else { 0.0 }).collect();
        (d, e)
    })
}

/// The eigendecomposition of the symmetric `a` (n×n row-major; the lower triangle is read):
/// faer's blocked tridiagonalisation, divide and conquer, and the blocked back-transformation,
/// threaded. [`crate::linalg::symmetric_eigen`]'s output layout — the eigenvalues, and the
/// eigenvectors as the COLUMNS of a row-major n×n matrix (`v[k * n + i]` is component k of
/// eigenvector i) — with the eigenvalues in nondecreasing order.
///
/// # Panics
/// If `a.len() != n * n`, or (never seen) faer's iteration does not converge.
pub fn symmetric_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(a.len(), n * n, "dense::symmetric_eigen: slice is not n×n");
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    install(|| {
        let m = lower_to_mat(a, n);
        let mut s = faer::diag::Diag::<f64>::zeros(n);
        let mut u = Mat::<f64>::zeros(n, n);
        let mut buf = MemBuffer::new(evd::self_adjoint_evd_scratch::<f64>(n, evd::ComputeEigenvectors::Yes, par(), Default::default()));
        evd::self_adjoint_evd(m.as_ref(), s.as_mut(), Some(u.as_mut()), par(), MemStack::new(&mut buf), Default::default()).expect("dense::symmetric_eigen: faer's eigensolver did not converge");
        let ev: Vec<f64> = (0..n).map(|i| s[i]).collect();
        let mut v = vec![0.0; n * n];
        for k in 0..n {
            for i in 0..n {
                v[k * n + i] = u[(k, i)];
            }
        }
        (ev, v)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

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

    /// faer's kernels against `linalg`'s at sizes on both sides of faer's blocking and parallel
    /// thresholds: the factor, the inverse, T's spectrum with Qᵀc (through the QL), and the
    /// eigenpairs, each to rounding of the matrix's norm.
    #[test]
    fn faer_kernels_match_the_linalg_kernels() {
        let mut rng = StdRng::seed_from_u64(17);
        for n in [1usize, 2, 3, 9, 40, 130, 420] {
            let a = random_spd(&mut rng, n, 0.1);
            let norm = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            // Cholesky
            let mut l_ref = a.clone();
            linalg::cholesky_in_place(&mut l_ref, n).unwrap();
            let l = cholesky_lower(&a, n).unwrap();
            for (x, y) in l.iter().zip(&l_ref) {
                assert!((x - y).abs() < 1e-11 * norm.sqrt(), "n {n}: factor {x} vs {y}");
            }
            assert!((0..n).all(|i| (i + 1..n).all(|j| l[i * n + j] == 0.0)), "the strict upper triangle is zero");
            // inverse
            let inv_ref = linalg::cholesky_inverse(&l_ref, n);
            let inv = spd_inverse(&l, n);
            let inv_norm = inv_ref.iter().map(|x| x * x).sum::<f64>().sqrt();
            for (x, y) in inv.iter().zip(&inv_ref) {
                assert!((x - y).abs() < 1e-9 * inv_norm, "n {n}: inverse {x} vs {y}");
            }
            // tridiagonal + Qᵀc: the spectrum and |ĉ|² per eigenvalue (sign-free) match linalg's
            let c0: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let spectrum = |d: Vec<f64>, e: Vec<f64>, c: Vec<f64>| {
                let (mut d, mut e, mut c) = (d, e, c);
                linalg::tridiagonal_ql(&mut d, &mut e, |i, cc, ss| {
                    let (x, y) = (c[i], c[i + 1]);
                    c[i] = cc * x - ss * y;
                    c[i + 1] = ss * x + cc * y;
                })
                .unwrap();
                let mut p: Vec<(f64, f64)> = d.into_iter().zip(c.into_iter().map(|x| x * x)).collect();
                p.sort_by(|x, y| x.0.total_cmp(&y.0));
                p
            };
            let t = linalg::Tridiagonal::of(&a, n);
            let mut c_ref = c0.clone();
            t.apply_qt(&mut c_ref);
            let want = spectrum(t.d.clone(), t.e.clone(), c_ref);
            let mut c = c0.clone();
            let (d, e) = tridiagonal_qtc(&a, n, &mut c);
            let got = spectrum(d, e, c);
            for (w, g) in want.iter().zip(&got) {
                assert!((w.0 - g.0).abs() < 1e-12 * norm, "n {n}: eigenvalue {} vs {}", w.0, g.0);
            }
            // ĉ² summed over each cluster of (nearly) equal eigenvalues is basis-free; random
            // spectra have no clusters, so per eigenvalue
            for (w, g) in want.iter().zip(&got) {
                assert!((w.1 - g.1).abs() < 1e-9 * (1.0 + w.1), "n {n}: ĉ² {} vs {}", w.1, g.1);
            }
            // eigenpairs
            let (ev, v) = symmetric_eigen(&a, n);
            assert!(ev.windows(2).all(|w| w[0] <= w[1]), "nondecreasing");
            for i in 0..n {
                for k in 0..n {
                    let av: f64 = (0..n).map(|j| a[k * n + j] * v[j * n + i]).sum();
                    assert!((av - ev[i] * v[k * n + i]).abs() < 1e-11 * norm, "n {n} pair {i}");
                }
                for j in 0..n {
                    let vv: f64 = (0..n).map(|k| v[k * n + i] * v[k * n + j]).sum();
                    assert!((vv - if i == j { 1.0 } else { 0.0 }).abs() < 1e-11, "n {n} orthonormality {i} {j}");
                }
            }
            for (x, y) in ev.iter().zip(&got) {
                assert!((x - y.0).abs() < 1e-12 * norm);
            }
        }
    }

    /// A rank-deficient Gram: faer's null eigenvalues sit at the ε·‖A‖ level like `linalg`'s (the
    /// τ = n·ε·tr threshold's premise, `novelty` and `grade`), with the spanned ones far above.
    #[test]
    fn faer_eigen_resolves_a_null_space_to_rounding() {
        let mut rng = StdRng::seed_from_u64(21);
        let (n, r) = (300, 117);
        let basis: Vec<Vec<f64>> = (0..r).map(|_| (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect()).collect();
        let mut g = vec![0.0; n * n];
        for _ in 0..600 {
            let c: Vec<f64> = (0..r).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let row: Vec<f64> = (0..n).map(|k| (0..r).map(|i| c[i] * basis[i][k]).sum()).collect();
            for i in 0..n {
                for j in 0..n {
                    g[i * n + j] += row[i] * row[j];
                }
            }
        }
        let tr: f64 = (0..n).map(|i| g[i * n + i]).sum();
        let (ev, _) = symmetric_eigen(&g, n);
        let mut sorted = ev.clone();
        sorted.sort_by(|a, b| a.abs().total_cmp(&b.abs()));
        let null_max = sorted[n - r - 1].abs() / tr;
        let spanned_min = sorted[n - r].abs() / tr;
        eprintln!("faer null space: max |λ_null|/tr {null_max:.2e}, smallest spanned {spanned_min:.2e}, τ/tr {:.2e}", n as f64 * f64::EPSILON);
        assert!(null_max < n as f64 * f64::EPSILON);
        assert!(spanned_min > 1e3 * n as f64 * f64::EPSILON);
    }

    /// The thread count changes the wall only (module doc): bit for bit on 1, 5 and the capped
    /// count, above the parallel thresholds.
    #[test]
    fn faer_kernels_are_the_same_on_any_thread_count() {
        let mut rng = StdRng::seed_from_u64(23);
        let n = 600;
        let a = random_spd(&mut rng, n, 0.1);
        let c0: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let run = |threads: usize| {
            crate::threads::with_threads(threads, || {
                let l = cholesky_lower(&a, n).unwrap();
                let inv = spd_inverse(&l, n);
                let mut c = c0.clone();
                let (d, e) = tridiagonal_qtc(&a, n, &mut c);
                let (ev, v) = symmetric_eigen(&a, n);
                let bits = |x: &[f64]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                (bits(&l), bits(&inv), bits(&d), bits(&e), bits(&c), bits(&ev), bits(&v))
            })
        };
        let one = run(1);
        assert!(one == run(5), "5 threads");
        assert!(one == run(crate::threads::worker_threads()), "the capped count");
    }
}
