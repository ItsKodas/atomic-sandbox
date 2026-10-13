//! One-particle basis for the surrogate: Chebyshev radial functions under a cosine cutoff and
//! real solid harmonics to l = 4, all with analytic derivatives. Pure f64, allocation-free.
//!
//! # Scaled distance and radial functions
//!
//! x(r) = 2·(1 − r/r_cut) − 1 = 1 − 2r/r_cut, so x(0) = 1 and x(r_cut) = −1; dx/dr = −2/r_cut.
//! (This is the brief's convention. Since T_n(−x) = (−1)ⁿ T_n(x) it differs from the review
//! note's x = 2r/r_cut − 1 only by the sign of the odd-n columns, which a linear fit absorbs.)
//! The Chebyshev polynomials of the first kind are T_0 = 1, T_1 = x, T_{n+1} = 2x T_n − T_{n−1};
//! differentiating the recursion gives T'_0 = 0, T'_1 = 1, T'_{n+1} = 2 T_n + 2x T'_n − T'_{n−1},
//! which is exact (polynomial arithmetic, no U_n or 1/√(1−x²) needed at the end points).
//! `radial` returns R_n(r) f_c(r) = T_n(x(r)) f_c(r) for n = 0..n_max (half-open, n_max values:
//! the count convention Task 2's n_features = n_max·S assumes) and
//! d/dr [T_n f_c] = T'_n(x)·(−2/r_cut)·f_c + T_n·f_c'.
//!
//! # Cutoff
//!
//! f_c(r) = ½(1 + cos(π r/r_cut)) for r < r_cut, 0 otherwise; f_c'(r) = −(π/(2 r_cut)) sin(π r/r_cut).
//! f_c(r_cut) = 0 and f_c'(r_cut) = 0, so every term in the energy and every force component goes
//! to zero continuously (C¹) when a neighbour leaves the sphere: no energy jump when a pair
//! crosses r_cut, hence an NVE-conserving surface for Verlet. The r ≥ r_cut branch returns exact
//! zeros (sin(π) is 1.2e−16 in f64, not 0).
//!
//! # Real solid harmonics
//!
//! Why solid and not spherical harmonics (spec review B2): S_lm(v) = |v|^l Y_lm(v̂) is a
//! homogeneous polynomial of degree l in (x, y, z), so its Cartesian gradient is a polynomial too
//! — no derivative on the sphere, no 1/r singularity, no angle chain rule. The 1/|v|^l needed to
//! recover Y_lm is folded into the radial part by the caller (Task 2/6), where it is harmless
//! (r ≥ 1 Bohr in any sample).
//!
//! Normalisation: Racah ("regular" solid harmonics), S_lm = |v|^l √(4π/(2l+1)) Y_lm^real(v̂),
//! i.e. the real spherical harmonics normalised to ∫|Y|² dΩ = 1 with the (2l+1)/4π of the
//! addition theorem folded in. The addition theorem then reads exactly
//!
//!   Σ_{m=−l}^{l} S_lm(a) S_lm(b) = |a|^l |b|^l P_l(â·b̂),
//!
//! and in particular Σ_m S_lm(v̂)² = 1 on the unit sphere. This is what Task 6's body-3
//! contraction Σ_m A_{n₁lm} A_{n₂lm} relies on: for real harmonics the invariant is the plain
//! sum over m of products of equal m (the A_{lm}A_{l,−m} form is the complex-basis one and is NOT
//! rotation invariant here). No Condon–Shortley phase; the m > 0 member is the cos(mφ) partner,
//! m < 0 the sin(mφ) partner. Signs do not affect any Σ_m contraction.
//!
//! Index: `lm_index(l, m) = l² + l + m`, m = −l..=l, so the (l_max+1)² values are contiguous
//! and ordered (0,0), (1,−1), (1,0), (1,1), (2,−2), …
//!
//! Closed forms (r² = x² + y² + z², ρ² = x² + y²), derived from the orthonormal real Y_lm tables
//! multiplied by √(4π/(2l+1)), and checked by hand at v̂ = ẑ (S_l0 = 1) and v̂ = x̂ (Σ_m S_lm² = 1,
//! S_l0 = P_l(0)):
//!
//! l = 0:  S_00 = 1
//! l = 1:  S_1,−1 = y,  S_10 = z,  S_11 = x
//! l = 2:  S_2,−2 = √3 xy,  S_2,−1 = √3 yz,  S_20 = z² − ½ρ²,  S_21 = √3 xz,  S_22 = (√3/2)(x² − y²)
//! l = 3:  S_3,−3 = (√10/4)(3x²y − y³),  S_3,−2 = √15 xyz,  S_3,−1 = (√6/4) y(4z² − ρ²),
//!         S_30 = z³ − (3/2) z ρ²,  S_31 = (√6/4) x(4z² − ρ²),  S_32 = (√15/2) z(x² − y²),
//!         S_33 = (√10/4)(x³ − 3xy²)
//! l = 4:  S_4,−4 = (√35/2) xy(x² − y²),  S_4,−3 = (√70/4) yz(3x² − y²),
//!         S_4,−2 = (√5/2) xy(6z² − ρ²),  S_4,−1 = (√10/4) yz(4z² − 3ρ²),
//!         S_40 = z⁴ − 3z²ρ² + (3/8)ρ⁴  [= (35z⁴ − 30z²r² + 3r⁴)/8],
//!         S_41 = (√10/4) xz(4z² − 3ρ²),  S_42 = (√5/4)(x² − y²)(6z² − ρ²),
//!         S_43 = (√70/4) xz(x² − 3y²),  S_44 = (√35/8)(x⁴ − 6x²y² + y⁴)
//!
//! The |m| = l pairs are Re/Im of (x + iy)^l up to one constant, so S_{l,l}² + S_{l,−l}² ∝ ρ^{2l};
//! the sin/cos partners of every |m| share one constant. Gradients are the term-by-term partial
//! derivatives of these polynomials (written out in the code next to each value);
//! `solid_harmonic_gradients_match_finite_differences` checks all 25 against a 5-point stencil,
//! which is exact for quartics, so the only FD error is rounding.
//!
//! Derivative pitfalls this layout avoids (spec review B2/B4): differentiating Y_lm(r̂) on the
//! sphere (needs ∂r̂/∂r = (I − r̂r̂ᵀ)/r and is singular at r = 0); mixing the complex-basis
//! contraction into the real one; a Condon–Shortley sign on only one of a sin/cos pair.

/// Highest l the closed forms below cover.
pub const L_MAX_SUPPORTED: usize = 4;

// Normalisation constants of the real solid harmonics (Racah), as literals so the hot path does
// no square roots; `constants_are_the_square_roots_they_claim` pins each to `f64::sqrt`.
const SQRT_3: f64 = 1.7320508075688772;
const SQRT_3_OVER_2: f64 = 0.8660254037844386;
const SQRT_6_OVER_4: f64 = 0.6123724356957945;
const SQRT_10_OVER_4: f64 = 0.7905694150420949;
const SQRT_15: f64 = 3.872983346207417;
const SQRT_15_OVER_2: f64 = 1.9364916731037085;
const SQRT_5: f64 = 2.23606797749979;
const SQRT_5_OVER_2: f64 = 1.118033988749895;
const SQRT_5_OVER_4: f64 = 0.5590169943749475;
const SQRT_35_OVER_2: f64 = 2.958039891549808;
const SQRT_35_OVER_8: f64 = 0.739509972887452;
const SQRT_70_OVER_4: f64 = 2.091650066335189;

/// Cosine cutoff f_c(r) = ½(1 + cos(π r/r_cut)) for r < r_cut, else 0; returns (f_c, df_c/dr).
/// Exact zeros at and beyond r_cut, so E and F are C¹ there.
#[inline]
pub fn cutoff(r: f64, r_cut: f64) -> (f64, f64) {
    debug_assert!(r_cut > 0.0, "r_cut must be positive, got {r_cut}");
    if r >= r_cut {
        return (0.0, 0.0);
    }
    let k = std::f64::consts::PI / r_cut;
    let (s, c) = (k * r).sin_cos();
    (0.5 * (1.0 + c), -0.5 * k * s)
}

/// R_n(r)·f_c(r) for n = 0..n_max (half-open: `out.len() == d_out.len() == n_max`) with
/// R_n = T_n(x(r)), x(r) = 1 − 2r/r_cut; `d_out` is d/dr of each. r ≥ r_cut → all zeros.
pub fn radial(n_max: usize, r: f64, r_cut: f64, out: &mut [f64], d_out: &mut [f64]) {
    assert_eq!(out.len(), n_max, "radial: out.len() must equal n_max");
    assert_eq!(d_out.len(), n_max, "radial: d_out.len() must equal n_max");
    let (fc, dfc) = cutoff(r, r_cut);
    if r >= r_cut || n_max == 0 {
        out.fill(0.0);
        d_out.fill(0.0);
        return;
    }
    let x = 2.0 * (1.0 - r / r_cut) - 1.0;
    let dx_dr = -2.0 / r_cut;
    // T_n and T'_n by the three-term recursion and its derivative; combine with f_c on the fly.
    let (mut t_prev, mut t) = (1.0, x); // T_0, T_1
    let (mut dt_prev, mut dt) = (0.0, 1.0); // T'_0, T'_1
    out[0] = fc;
    d_out[0] = dfc;
    if n_max > 1 {
        out[1] = x * fc;
        d_out[1] = dx_dr * fc + x * dfc;
    }
    for n in 2..n_max {
        let t_next = 2.0 * x * t - t_prev;
        let dt_next = 2.0 * t + 2.0 * x * dt - dt_prev;
        out[n] = t_next * fc;
        d_out[n] = dt_next * dx_dr * fc + t_next * dfc;
        t_prev = t;
        t = t_next;
        dt_prev = dt;
        dt = dt_next;
    }
}

/// Position of S_lm in the `solid_harmonics` output: l² + l + m, with m = −l..=l.
#[inline]
pub const fn lm_index(l: usize, m: isize) -> usize {
    debug_assert!(m.unsigned_abs() <= l, "|m| must not exceed l");
    ((l * l + l) as isize + m) as usize
}

/// Real solid harmonics S_lm(v) = |v|^l Y_lm(v̂), Racah normalised
/// (Σ_m S_lm(a)S_lm(b) = |a|^l|b|^l P_l(â·b̂)), for l ≤ l_max ≤ [`L_MAX_SUPPORTED`], at index
/// [`lm_index`]; `grad[i]` is the Cartesian gradient of `out[i]`. Both slices have length (l_max+1)².
pub fn solid_harmonics(l_max: usize, v: [f64; 3], out: &mut [f64], grad: &mut [[f64; 3]]) {
    assert!(
        l_max <= L_MAX_SUPPORTED,
        "solid_harmonics: l_max {l_max} > {L_MAX_SUPPORTED}"
    );
    let n = (l_max + 1) * (l_max + 1);
    assert_eq!(
        out.len(),
        n,
        "solid_harmonics: out.len() must be (l_max+1)²"
    );
    assert_eq!(
        grad.len(),
        n,
        "solid_harmonics: grad.len() must be (l_max+1)²"
    );
    let [x, y, z] = v;

    // l = 0: S_00 = 1.
    out[0] = 1.0;
    grad[0] = [0.0; 3];
    if l_max == 0 {
        return;
    }

    // l = 1: (y, z, x).
    out[1] = y;
    grad[1] = [0.0, 1.0, 0.0];
    out[2] = z;
    grad[2] = [0.0, 0.0, 1.0];
    out[3] = x;
    grad[3] = [1.0, 0.0, 0.0];
    if l_max == 1 {
        return;
    }

    let (xx, yy, zz) = (x * x, y * y, z * z);
    let (xy, yz, xz) = (x * y, y * z, x * z);
    let rho2 = xx + yy;

    // l = 2.
    out[4] = SQRT_3 * xy; // S_2,−2 = √3 xy
    grad[4] = [SQRT_3 * y, SQRT_3 * x, 0.0];
    out[5] = SQRT_3 * yz; // S_2,−1 = √3 yz
    grad[5] = [0.0, SQRT_3 * z, SQRT_3 * y];
    out[6] = zz - 0.5 * rho2; // S_20 = z² − ½ρ²
    grad[6] = [-x, -y, 2.0 * z];
    out[7] = SQRT_3 * xz; // S_21 = √3 xz
    grad[7] = [SQRT_3 * z, 0.0, SQRT_3 * x];
    out[8] = SQRT_3_OVER_2 * (xx - yy); // S_22 = (√3/2)(x² − y²)
    grad[8] = [SQRT_3 * x, -SQRT_3 * y, 0.0];
    if l_max == 2 {
        return;
    }

    // l = 3.
    out[9] = SQRT_10_OVER_4 * (3.0 * xx * y - yy * y); // S_3,−3 = (√10/4)(3x²y − y³)
    grad[9] = [
        SQRT_10_OVER_4 * 6.0 * xy,
        SQRT_10_OVER_4 * 3.0 * (xx - yy),
        0.0,
    ];
    out[10] = SQRT_15 * xy * z; // S_3,−2 = √15 xyz
    grad[10] = [SQRT_15 * yz, SQRT_15 * xz, SQRT_15 * xy];
    let q = 4.0 * zz - rho2; // 4z² − ρ²
    out[11] = SQRT_6_OVER_4 * y * q; // S_3,−1 = (√6/4) y(4z² − x² − y²)
    grad[11] = [
        SQRT_6_OVER_4 * (-2.0 * xy),
        SQRT_6_OVER_4 * (4.0 * zz - xx - 3.0 * yy),
        SQRT_6_OVER_4 * 8.0 * yz,
    ];
    out[12] = zz * z - 1.5 * z * rho2; // S_30 = z³ − (3/2) z ρ²
    grad[12] = [-3.0 * xz, -3.0 * yz, 3.0 * zz - 1.5 * rho2];
    out[13] = SQRT_6_OVER_4 * x * q; // S_31 = (√6/4) x(4z² − x² − y²)
    grad[13] = [
        SQRT_6_OVER_4 * (4.0 * zz - 3.0 * xx - yy),
        SQRT_6_OVER_4 * (-2.0 * xy),
        SQRT_6_OVER_4 * 8.0 * xz,
    ];
    out[14] = SQRT_15_OVER_2 * z * (xx - yy); // S_32 = (√15/2) z(x² − y²)
    grad[14] = [SQRT_15 * xz, -SQRT_15 * yz, SQRT_15_OVER_2 * (xx - yy)];
    out[15] = SQRT_10_OVER_4 * (xx * x - 3.0 * x * yy); // S_33 = (√10/4)(x³ − 3xy²)
    grad[15] = [
        SQRT_10_OVER_4 * 3.0 * (xx - yy),
        SQRT_10_OVER_4 * (-6.0 * xy),
        0.0,
    ];
    if l_max == 3 {
        return;
    }

    // l = 4.
    out[16] = SQRT_35_OVER_2 * xy * (xx - yy); // S_4,−4 = (√35/2)(x³y − xy³)
    grad[16] = [
        SQRT_35_OVER_2 * (3.0 * xx * y - yy * y),
        SQRT_35_OVER_2 * (xx * x - 3.0 * x * yy),
        0.0,
    ];
    out[17] = SQRT_70_OVER_4 * yz * (3.0 * xx - yy); // S_4,−3 = (√70/4)(3x²yz − y³z)
    grad[17] = [
        SQRT_70_OVER_4 * 6.0 * xy * z,
        SQRT_70_OVER_4 * 3.0 * z * (xx - yy),
        SQRT_70_OVER_4 * y * (3.0 * xx - yy),
    ];
    let p = 6.0 * zz - rho2; // 6z² − ρ²
    out[18] = SQRT_5_OVER_2 * xy * p; // S_4,−2 = (√5/2)(6xyz² − x³y − xy³)
    grad[18] = [
        SQRT_5_OVER_2 * y * (6.0 * zz - 3.0 * xx - yy),
        SQRT_5_OVER_2 * x * (6.0 * zz - xx - 3.0 * yy),
        SQRT_5_OVER_2 * 12.0 * xy * z,
    ];
    let w = 4.0 * zz - 3.0 * rho2; // 4z² − 3ρ²
    out[19] = SQRT_10_OVER_4 * yz * w; // S_4,−1 = (√10/4)(4yz³ − 3x²yz − 3y³z)
    grad[19] = [
        SQRT_10_OVER_4 * (-6.0 * xy * z),
        SQRT_10_OVER_4 * z * (4.0 * zz - 3.0 * xx - 9.0 * yy),
        SQRT_10_OVER_4 * y * (12.0 * zz - 3.0 * rho2),
    ];
    out[20] = zz * zz - 3.0 * zz * rho2 + 0.375 * rho2 * rho2; // S_40 = z⁴ − 3z²ρ² + (3/8)ρ⁴
    let s40_xy = 1.5 * rho2 - 6.0 * zz;
    grad[20] = [x * s40_xy, y * s40_xy, 4.0 * zz * z - 6.0 * z * rho2];
    out[21] = SQRT_10_OVER_4 * xz * w; // S_41 = (√10/4)(4xz³ − 3x³z − 3xy²z)
    grad[21] = [
        SQRT_10_OVER_4 * z * (4.0 * zz - 9.0 * xx - 3.0 * yy),
        SQRT_10_OVER_4 * (-6.0 * xy * z),
        SQRT_10_OVER_4 * x * (12.0 * zz - 3.0 * rho2),
    ];
    out[22] = SQRT_5_OVER_4 * (xx - yy) * p; // S_42 = (√5/4)(6x²z² − 6y²z² − x⁴ + y⁴)
    grad[22] = [
        SQRT_5 * x * (3.0 * zz - xx),
        SQRT_5 * y * (yy - 3.0 * zz),
        SQRT_5 * 3.0 * z * (xx - yy),
    ];
    out[23] = SQRT_70_OVER_4 * xz * (xx - 3.0 * yy); // S_43 = (√70/4)(x³z − 3xy²z)
    grad[23] = [
        SQRT_70_OVER_4 * 3.0 * z * (xx - yy),
        SQRT_70_OVER_4 * (-6.0 * xy * z),
        SQRT_70_OVER_4 * x * (xx - 3.0 * yy),
    ];
    out[24] = SQRT_35_OVER_8 * (xx * xx - 6.0 * xx * yy + yy * yy); // S_44 = (√35/8)(x⁴ − 6x²y² + y⁴)
    grad[24] = [
        SQRT_35_OVER_2 * (xx * x - 3.0 * x * yy),
        SQRT_35_OVER_2 * (yy * y - 3.0 * xx * y),
        0.0,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_are_the_square_roots_they_claim() {
        // Literal constants must be the correctly rounded f64 of the exact values (divisions by
        // powers of two are exact, and `sqrt` is correctly rounded), so equality is exact.
        assert_eq!(SQRT_3, 3f64.sqrt());
        assert_eq!(SQRT_3_OVER_2, 3f64.sqrt() / 2.0);
        assert_eq!(SQRT_6_OVER_4, 6f64.sqrt() / 4.0);
        assert_eq!(SQRT_10_OVER_4, 10f64.sqrt() / 4.0);
        assert_eq!(SQRT_15, 15f64.sqrt());
        assert_eq!(SQRT_15_OVER_2, 15f64.sqrt() / 2.0);
        assert_eq!(SQRT_5, 5f64.sqrt());
        assert_eq!(SQRT_5_OVER_2, 5f64.sqrt() / 2.0);
        assert_eq!(SQRT_5_OVER_4, 5f64.sqrt() / 4.0);
        assert_eq!(SQRT_35_OVER_2, 35f64.sqrt() / 2.0);
        assert_eq!(SQRT_35_OVER_8, 35f64.sqrt() / 8.0);
        assert_eq!(SQRT_70_OVER_4, 70f64.sqrt() / 4.0);
    }

    #[test]
    #[should_panic(expected = "out.len() must equal n_max")]
    fn radial_rejects_a_wrong_output_length() {
        radial(4, 1.0, 6.0, &mut [0.0; 3], &mut [0.0; 4]);
    }

    #[test]
    #[should_panic(expected = "l_max 5 > 4")]
    fn solid_harmonics_reject_l_above_the_supported_maximum() {
        solid_harmonics(5, [1.0, 0.0, 0.0], &mut [0.0; 36], &mut [[0.0; 3]; 36]);
    }
}
