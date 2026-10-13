//! Task 1 acceptance tests for `surrogate::basis` (the brief's five, plus index layout).
//! Finite differences use a 5-point central stencil: exact for polynomials of degree ≤ 4
//! (the solid harmonics), O(h⁴) otherwise, so the measured error is essentially rounding.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use surrogate::basis::{cutoff, lm_index, radial, solid_harmonics, L_MAX_SUPPORTED};

const R_CUT: f64 = 6.0;

fn fd5(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (-f(x + 2.0 * h) + 8.0 * f(x + h) - 8.0 * f(x - h) + f(x - 2.0 * h)) / (12.0 * h)
}

fn legendre(l: usize, t: f64) -> f64 {
    let (mut p0, mut p1) = (1.0, t);
    if l == 0 {
        return p0;
    }
    for n in 1..l {
        let p2 = ((2 * n + 1) as f64 * t * p1 - n as f64 * p0) / (n + 1) as f64;
        p0 = p1;
        p1 = p2;
    }
    p1
}

fn random_vec(rng: &mut StdRng, scale: f64) -> [f64; 3] {
    [
        rng.gen_range(-scale..scale),
        rng.gen_range(-scale..scale),
        rng.gen_range(-scale..scale),
    ]
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Rotation matrix from a random unit quaternion (uniform on SO(3)).
fn random_rotation(rng: &mut StdRng) -> [[f64; 3]; 3] {
    let q: [f64; 4] = {
        let mut q = [0.0; 4];
        loop {
            for c in &mut q {
                *c = rng.gen_range(-1.0..1.0);
            }
            let n2: f64 = q.iter().map(|c| c * c).sum();
            if n2 > 0.1 && n2 <= 1.0 {
                let n = n2.sqrt();
                for c in &mut q {
                    *c /= n;
                }
                break q;
            }
        }
    };
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

fn apply(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}

fn sh(l_max: usize, v: [f64; 3]) -> (Vec<f64>, Vec<[f64; 3]>) {
    let n = (l_max + 1) * (l_max + 1);
    let mut out = vec![0.0; n];
    let mut grad = vec![[0.0; 3]; n];
    solid_harmonics(l_max, v, &mut out, &mut grad);
    (out, grad)
}

#[test]
fn lm_index_is_contiguous_and_ordered() {
    let mut expect = 0;
    for l in 0..=L_MAX_SUPPORTED {
        for m in -(l as isize)..=(l as isize) {
            assert_eq!(lm_index(l, m), expect, "l = {l}, m = {m}");
            expect += 1;
        }
    }
    assert_eq!(expect, (L_MAX_SUPPORTED + 1) * (L_MAX_SUPPORTED + 1));
}

#[test]
fn cutoff_is_smooth_and_vanishes_at_r_cut() {
    assert_eq!(cutoff(0.0, R_CUT).0, 1.0);
    assert_eq!(cutoff(0.0, R_CUT).1, 0.0);
    assert_eq!(cutoff(R_CUT, R_CUT), (0.0, 0.0), "exact zeros at r_cut");
    assert_eq!(cutoff(R_CUT + 1e-12, R_CUT), (0.0, 0.0));
    assert_eq!(cutoff(2.0 * R_CUT, R_CUT), (0.0, 0.0));
    // Continuity of f_c and f_c' from the inside at r_cut.
    let (f, df) = cutoff(R_CUT * (1.0 - 1e-9), R_CUT);
    assert!(f.abs() < 1e-15, "f_c just inside r_cut = {f:e}");
    assert!(df.abs() < 1e-9, "f_c' just inside r_cut = {df:e}");
    // Pinned interior values: f_c(r_cut/2) = 1/2 and f_c(4) = 1/4 for r_cut = 6 (review B1).
    assert!((cutoff(3.0, R_CUT).0 - 0.5).abs() < 1e-15);
    assert!((cutoff(4.0, R_CUT).0 - 0.25).abs() < 1e-15);

    let mut rng = StdRng::seed_from_u64(11);
    let mut worst = 0.0f64;
    for _ in 0..20 {
        let r = rng.gen_range(0.05..R_CUT - 0.05);
        let (_, df) = cutoff(r, R_CUT);
        let fd = fd5(|s| cutoff(s, R_CUT).0, r, 1e-4);
        worst = worst.max((fd - df).abs());
        assert!((fd - df).abs() < 1e-9, "r = {r}: analytic {df} vs FD {fd}");
    }
    eprintln!("cutoff: max |FD − analytic| over 20 points = {worst:.2e}");
}

#[test]
#[allow(clippy::needless_range_loop)] // out[n] and d_out[n] are indexed together
fn radial_matches_chebyshev_recurrence_and_its_derivative() {
    let n_max = 10;
    let mut rng = StdRng::seed_from_u64(23);
    let mut out = vec![0.0; n_max];
    let mut d_out = vec![0.0; n_max];
    let (mut worst_val, mut worst_der) = (0.0f64, 0.0f64);
    for _ in 0..40 {
        let r = rng.gen_range(0.05..R_CUT - 0.05);
        radial(n_max, r, R_CUT, &mut out, &mut d_out);
        let x = 2.0 * (1.0 - r / R_CUT) - 1.0;
        let (fc, _) = cutoff(r, R_CUT);
        for n in 0..n_max {
            let direct = (n as f64 * x.acos()).cos() * fc;
            worst_val = worst_val.max((out[n] - direct).abs());
            assert!(
                (out[n] - direct).abs() < 1e-12,
                "n = {n}, r = {r}: {} vs {direct}",
                out[n]
            );
            let fd = fd5(
                |s| {
                    let mut o = vec![0.0; n_max];
                    let mut d = vec![0.0; n_max];
                    radial(n_max, s, R_CUT, &mut o, &mut d);
                    o[n]
                },
                r,
                1e-4,
            );
            let tol = 1e-8 * d_out[n].abs().max(1.0);
            worst_der = worst_der.max((fd - d_out[n]).abs() / d_out[n].abs().max(1.0));
            assert!(
                (fd - d_out[n]).abs() < tol,
                "n = {n}, r = {r}: d {} vs FD {fd}",
                d_out[n]
            );
        }
    }
    eprintln!(
        "radial: max |value − direct| = {worst_val:.2e}, max relative |FD − d| = {worst_der:.2e}"
    );
    // End points: x(0) = 1 gives T_n = 1 for every n; at and beyond r_cut everything is zero.
    radial(n_max, 0.0, R_CUT, &mut out, &mut d_out);
    for n in 0..n_max {
        assert_eq!(out[n], 1.0, "T_n(1) = 1, n = {n}");
    }
    for r in [R_CUT, R_CUT + 1e-9, 2.0 * R_CUT] {
        out.fill(f64::NAN);
        d_out.fill(f64::NAN);
        radial(n_max, r, R_CUT, &mut out, &mut d_out);
        assert!(
            out.iter().chain(d_out.iter()).all(|&v| v == 0.0),
            "r = {r}: {out:?} {d_out:?}"
        );
    }
    // n_max = 0 is legal and writes nothing.
    radial(0, 1.0, R_CUT, &mut [], &mut []);
}

#[test]
fn solid_harmonics_satisfy_the_addition_theorem() {
    let mut rng = StdRng::seed_from_u64(37);
    let mut worst = 0.0f64;
    for _ in 0..50 {
        let a = random_vec(&mut rng, 3.0);
        let b = random_vec(&mut rng, 3.0);
        let (sa, _) = sh(L_MAX_SUPPORTED, a);
        let (sb, _) = sh(L_MAX_SUPPORTED, b);
        let (na, nb) = (norm(a), norm(b));
        let t = dot(a, b) / (na * nb);
        for l in 0..=L_MAX_SUPPORTED {
            let lhs: f64 = (-(l as isize)..=(l as isize))
                .map(|m| sa[lm_index(l, m)] * sb[lm_index(l, m)])
                .sum();
            let rhs = na.powi(l as i32) * nb.powi(l as i32) * legendre(l, t);
            // Scale-free tolerance: |a|^l|b|^l reaches ~ (3√3)^8 ≈ 5e5 for l = 4.
            let scale = (na * nb).powi(l as i32).max(1.0);
            worst = worst.max((lhs - rhs).abs() / scale);
            assert!(
                (lhs - rhs).abs() < 1e-12 * scale,
                "l = {l}: Σ_m = {lhs} vs {rhs}"
            );
        }
    }
    eprintln!("addition theorem: max scaled residual = {worst:.2e}");
    // Pins the normalisation: on the unit sphere Σ_m S_lm² = 1 and S_l0(ẑ) = 1.
    let (sz, _) = sh(L_MAX_SUPPORTED, [0.0, 0.0, 1.0]);
    let (sx, _) = sh(L_MAX_SUPPORTED, [1.0, 0.0, 0.0]);
    for l in 0..=L_MAX_SUPPORTED {
        assert!(
            (sz[lm_index(l, 0)] - 1.0).abs() < 1e-15,
            "S_{l}0(ẑ) = {}",
            sz[lm_index(l, 0)]
        );
        assert!(
            (sx[lm_index(l, 0)] - legendre(l, 0.0)).abs() < 1e-15,
            "S_{l}0(x̂) = {} vs P_l(0) = {}",
            sx[lm_index(l, 0)],
            legendre(l, 0.0)
        );
    }
    // Independent pin of the m-labelling (the theorem fixes each l-block only up to an orthogonal
    // mixing): the sectoral pair (|m| = l) is (Re, Im)(x + iy)^l · √(C(2l, l)/2) / 2^(l−1).
    let v = [0.7, -1.3, 0.4];
    let (s, _) = sh(L_MAX_SUPPORTED, v);
    let central_binomial = [2.0, 6.0, 20.0, 70.0]; // C(2l, l), l = 1..=4
    let (mut re, mut im) = (1.0, 0.0);
    for l in 1..=L_MAX_SUPPORTED {
        (re, im) = (re * v[0] - im * v[1], re * v[1] + im * v[0]);
        let c = (central_binomial[l - 1] / 2.0f64).sqrt() / 2f64.powi(l as i32 - 1);
        let (cos_part, sin_part) = (s[lm_index(l, l as isize)], s[lm_index(l, -(l as isize))]);
        assert!(
            (cos_part - c * re).abs() < 1e-14 * c * re.abs().max(1.0),
            "S_{l},{l} = {cos_part} vs {}",
            c * re
        );
        assert!(
            (sin_part - c * im).abs() < 1e-14 * c * im.abs().max(1.0),
            "S_{l},−{l} = {sin_part} vs {}",
            c * im
        );
    }
}

#[test]
#[allow(clippy::needless_range_loop)] // grad[i] and the FD closure share the index i
fn solid_harmonic_gradients_match_finite_differences() {
    let mut rng = StdRng::seed_from_u64(41);
    let n = (L_MAX_SUPPORTED + 1) * (L_MAX_SUPPORTED + 1);
    let mut worst = 0.0f64;
    for _ in 0..30 {
        let v = random_vec(&mut rng, 3.0);
        let (_, grad) = sh(L_MAX_SUPPORTED, v);
        for i in 0..n {
            for c in 0..3 {
                let fd = fd5(
                    |s| {
                        let mut w = v;
                        w[c] = s;
                        sh(L_MAX_SUPPORTED, w).0[i]
                    },
                    v[c],
                    1e-3,
                );
                let g = grad[i][c];
                let err = (fd - g).abs() / g.abs().max(1.0);
                worst = worst.max(err);
                assert!(
                    err < 1e-8,
                    "S index {i} (l² + l + m), component {c}, v = {v:?}: ∇ = {g} vs FD {fd}"
                );
            }
        }
    }
    eprintln!("solid harmonic gradients: max |FD − ∇S| / max(1, |∇S|) over 30 points × 25 × 3 = {worst:.2e}");
    // Lower l_max writes exactly the leading block.
    let v = [0.3, -1.1, 0.7];
    let (full, gfull) = sh(L_MAX_SUPPORTED, v);
    for l_max in 0..L_MAX_SUPPORTED {
        let (part, gpart) = sh(l_max, v);
        assert_eq!(part.len(), (l_max + 1) * (l_max + 1));
        assert_eq!(&part[..], &full[..part.len()]);
        assert_eq!(&gpart[..], &gfull[..part.len()]);
    }
}

#[test]
fn solid_harmonics_rotate_as_a_representation() {
    let mut rng = StdRng::seed_from_u64(53);
    let mut worst = 0.0f64;
    for _ in 0..50 {
        let rot = random_rotation(&mut rng);
        let a = random_vec(&mut rng, 3.0);
        let b = random_vec(&mut rng, 3.0);
        let (sa, _) = sh(L_MAX_SUPPORTED, a);
        let (sra, _) = sh(L_MAX_SUPPORTED, apply(&rot, a));
        let (sb, _) = sh(L_MAX_SUPPORTED, b);
        let (srb, _) = sh(L_MAX_SUPPORTED, apply(&rot, b));
        for l in 0..=L_MAX_SUPPORTED {
            let ms = -(l as isize)..=(l as isize);
            let n2: f64 = ms.clone().map(|m| sa[lm_index(l, m)].powi(2)).sum();
            let rn2: f64 = ms.clone().map(|m| sra[lm_index(l, m)].powi(2)).sum();
            let scale = n2.max(1.0);
            worst = worst.max((n2 - rn2).abs() / scale);
            assert!(
                (n2 - rn2).abs() < 1e-12 * scale,
                "l = {l}: |S(v)|² = {n2} vs |S(Rv)|² = {rn2}"
            );
            // The bilinear invariant too (what Task 6 contracts): Σ_m S_lm(Ra) S_lm(Rb) = Σ_m S_lm(a) S_lm(b).
            let ab: f64 = ms
                .clone()
                .map(|m| sa[lm_index(l, m)] * sb[lm_index(l, m)])
                .sum();
            let rab: f64 = ms.map(|m| sra[lm_index(l, m)] * srb[lm_index(l, m)]).sum();
            let scale = (norm(a) * norm(b)).powi(l as i32).max(1.0);
            assert!(
                (ab - rab).abs() < 1e-12 * scale,
                "l = {l}: Σ_m S(a)S(b) = {ab} vs rotated {rab}"
            );
        }
    }
    eprintln!("rotation: max relative norm change per l = {worst:.2e}");
}
