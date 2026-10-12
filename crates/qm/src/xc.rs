//! LSDA exchange-correlation: Dirac exchange (spin-scaled) + Perdew–Wang 1992 correlation.
//!
//! Everything is in Hartree atomic units (ħ = mₑ = e = 1). Densities are per-spin
//! electron densities n↑, n↓ (electrons/Bohr³); n = n↑ + n↓. Energies are negative
//! (binding); potentials v_xc,σ = δE_xc/δn_σ are negative.
//!
//! # Exchange (Dirac, spin-scaled)
//!
//! For a spin-unpolarised density, E_x[n] = −C ∫ n^{4/3} d³r with C = (3/4)(3/π)^{1/3}.
//! The exact spin-scaling relation (Oliver–Perdew) gives the polarised functional:
//!
//!   E_x[n↑, n↓] = ½ ( E_x[2n↑] + E_x[2n↓] )
//!              = −C·2^{1/3} ∫ ( n↑^{4/3} + n↓^{4/3} ) d³r,
//!
//! since ½·(2n_σ)^{4/3} = 2^{1/3} n_σ^{4/3}. The energy density is e_x = −C·2^{1/3}(n↑^{4/3}+n↓^{4/3})
//! and, differentiating one spin channel,
//!
//!   v_x,σ = ∂e_x/∂n_σ = −(4/3)·C·2^{1/3}·n_σ^{1/3} = −(6 n_σ/π)^{1/3},
//!
//! using (4/3)(3/4)(3/π)^{1/3} 2^{1/3} = (6/π)^{1/3}. Unpolarised check: n_σ = n/2 gives
//! e_x = −C n^{4/3} and v_x = −(3n/π)^{1/3}, the familiar Dirac/Slater results.
//!
//! # Correlation (Perdew & Wang, Phys. Rev. B 45, 13244 (1992))
//!
//! With r_s = (3/(4πn))^{1/3} and ζ = (n↑ − n↓)/n, each of the three fitted quantities
//! ε_c(r_s,0), ε_c(r_s,1), −α_c(r_s) has the same analytic form (PW92 eq. 10, p = 1):
//!
//!   G(r_s; A, α₁, β₁..β₄) = Q₀ ln(1 + 1/Q₁),
//!   Q₀ = −2A(1 + α₁ r_s),
//!   Q₁ = 2A(β₁ r_s^{1/2} + β₂ r_s + β₃ r_s^{3/2} + β₄ r_s²).
//!
//! Note the third fit parameterises **−α_c** (the spin stiffness enters with a minus sign
//! in the paper's Table I), so α_c = −G(MALPHA params) and dα_c/dr_s = −dG/dr_s.
//!
//! Interpolation in ζ (PW92 eq. 8):
//!
//!   ε_c(r_s, ζ) = ε_c(r_s,0) + α_c(r_s) f(ζ)/f″(0) (1 − ζ⁴) + [ε_c(r_s,1) − ε_c(r_s,0)] f(ζ) ζ⁴,
//!   f(ζ) = [(1+ζ)^{4/3} + (1−ζ)^{4/3} − 2] / (2^{4/3} − 2),   f″(0) = 1.709921,
//!   f′(ζ) = (4/3)[(1+ζ)^{1/3} − (1−ζ)^{1/3}] / (2^{4/3} − 2).
//!
//! At ζ = 0: f = 0 so ε_c = ε_c(r_s,0). At ζ = ±1: f = 1, ζ⁴ = 1 so ε_c = ε_c(r_s,1).
//!
//! ## Derivative of G
//!
//!   dG/dr_s = Q₀′ ln(1 + 1/Q₁) + Q₀ · d/dr_s ln(1 + 1/Q₁)
//!           = −2Aα₁ ln(1 + 1/Q₁) − Q₀ Q₁′ / (Q₁² + Q₁),
//!
//! since d/dr_s ln(1 + 1/Q₁) = (1/(1 + 1/Q₁))·(−Q₁′/Q₁²) = −Q₁′/(Q₁² + Q₁), and
//!
//!   Q₁′ = 2A( β₁/(2 r_s^{1/2}) + β₂ + (3/2) β₃ r_s^{1/2} + 2 β₄ r_s )
//!       = A( β₁ r_s^{−1/2} + 2β₂ + 3β₃ r_s^{1/2} + 4β₄ r_s ).
//!
//! ## Potential
//!
//! E_c = ∫ n ε_c(r_s, ζ) d³r, so v_c,σ = ∂(n ε_c)/∂n_σ = ε_c + n (∂ε_c/∂r_s ∂r_s/∂n_σ + ∂ε_c/∂ζ ∂ζ/∂n_σ).
//! Because n = n↑ + n↓, ∂r_s/∂n_σ = ∂r_s/∂n = −r_s/(3n). For ζ:
//! ∂ζ/∂n↑ = 1/n − (n↑−n↓)/n² = (1 − ζ)/n and ∂ζ/∂n↓ = −1/n − (n↑−n↓)/n² = −(1 + ζ)/n,
//! i.e. ∂ζ/∂n_σ = (s_σ − ζ)/n with s↑ = +1, s↓ = −1. Hence
//!
//!   v_c,σ = ε_c − (r_s/3) ∂ε_c/∂r_s − (ζ − s_σ) ∂ε_c/∂ζ.
//!
//! The ζ-derivative of the interpolation formula is
//!
//!   ∂ε_c/∂ζ = (α_c/f″(0)) [ f′(1 − ζ⁴) − 4ζ³ f ] + [ε_c(r_s,1) − ε_c(r_s,0)] [ f′ ζ⁴ + 4ζ³ f ],
//!
//! and ∂ε_c/∂r_s is the same interpolation applied to the three dG/dr_s values.
//!
//! # Numerics
//!
//! Points with n < N_MIN return exactly zero (energy density and both potentials) so the
//! vacuum region never produces NaN from r_s → ∞ or 0/0 in ζ. Negative inputs are clamped
//! to zero. These are per-point functions; the caller integrates `energy_density` with the
//! grid's compensated summation.

use std::f64::consts::PI;

/// Parameters of one PW92 fit function G(r_s) (PW92 Table I, p = 1).
pub struct Pw92Params {
    a: f64,
    a1: f64,
    b1: f64,
    b2: f64,
    b3: f64,
    b4: f64,
}

/// ε_c(r_s, ζ = 0): paramagnetic correlation energy per electron.
pub const PARA: Pw92Params = Pw92Params {
    a: 0.031091,
    a1: 0.21370,
    b1: 7.5957,
    b2: 3.5876,
    b3: 1.6382,
    b4: 0.49294,
};
/// ε_c(r_s, ζ = 1): ferromagnetic correlation energy per electron.
pub const FERRO: Pw92Params = Pw92Params {
    a: 0.015545,
    a1: 0.20548,
    b1: 14.1189,
    b2: 6.1977,
    b3: 3.3662,
    b4: 0.62517,
};
/// −α_c(r_s): minus the spin stiffness (PW92 fits the negative of α_c).
pub const MALPHA: Pw92Params = Pw92Params {
    a: 0.016887,
    a1: 0.11125,
    b1: 10.357,
    b2: 3.6231,
    b3: 0.88026,
    b4: 0.49671,
};
/// f″(0) for the spin-interpolation function f(ζ).
const FPP0: f64 = 1.709921;
/// Densities below this are treated as vacuum.
const N_MIN: f64 = 1e-20;

/// Returns (G, dG/dr_s) for the PW92 fit function with parameters `p`.
pub fn pw92_g(rs: f64, p: &Pw92Params) -> (f64, f64) {
    let srs = rs.sqrt();
    let q0 = -2.0 * p.a * (1.0 + p.a1 * rs);
    let q1 = 2.0 * p.a * (p.b1 * srs + p.b2 * rs + p.b3 * rs * srs + p.b4 * rs * rs);
    let q1p = p.a * (p.b1 / srs + 2.0 * p.b2 + 3.0 * p.b3 * srs + 4.0 * p.b4 * rs);
    let ln = (1.0 + 1.0 / q1).ln();
    (q0 * ln, -2.0 * p.a * p.a1 * ln - q0 * q1p / (q1 * q1 + q1))
}

/// Returns (f(ζ), f′(ζ)).
fn f_zeta(z: f64) -> (f64, f64) {
    let denom = 2f64.powf(4.0 / 3.0) - 2.0;
    let f = ((1.0 + z).powf(4.0 / 3.0) + (1.0 - z).powf(4.0 / 3.0) - 2.0) / denom;
    let fp = (4.0 / 3.0) * ((1.0 + z).powf(1.0 / 3.0) - (1.0 - z).powf(1.0 / 3.0)) / denom;
    (f, fp)
}

/// (n·ε_c, v_c averaged over spin — used by tests; see exc_point for the full result)
pub fn correlation_only(n_up: f64, n_dn: f64) -> (f64, f64) {
    let (e, vu, vd) = correlation(n_up, n_dn);
    (e, 0.5 * (vu + vd))
}

/// Returns (n·ε_c, v_c,↑, v_c,↓).
fn correlation(n_up: f64, n_dn: f64) -> (f64, f64, f64) {
    let n = n_up + n_dn;
    if n < N_MIN {
        return (0.0, 0.0, 0.0);
    }
    let rs = (3.0 / (4.0 * PI * n)).powf(1.0 / 3.0);
    let z = ((n_up - n_dn) / n).clamp(-1.0, 1.0);
    let (ec0, dec0) = pw92_g(rs, &PARA);
    let (ec1, dec1) = pw92_g(rs, &FERRO);
    let (mac, dmac) = pw92_g(rs, &MALPHA);
    let (ac, dac) = (-mac, -dmac);
    let (f, fp) = f_zeta(z);
    let z3 = z * z * z;
    let z4 = z3 * z;
    let ec = ec0 + ac * f / FPP0 * (1.0 - z4) + (ec1 - ec0) * f * z4;
    let dec_drs = dec0 + dac * f / FPP0 * (1.0 - z4) + (dec1 - dec0) * f * z4;
    let dec_dz =
        ac / FPP0 * (fp * (1.0 - z4) - 4.0 * z3 * f) + (ec1 - ec0) * (fp * z4 + 4.0 * z3 * f);
    let common = ec - rs / 3.0 * dec_drs;
    (n * ec, common - (z - 1.0) * dec_dz, common - (z + 1.0) * dec_dz)
}

/// Returns (e_x, v_x,↑, v_x,↓) for spin-scaled Dirac exchange.
fn exchange(n_up: f64, n_dn: f64) -> (f64, f64, f64) {
    let c = (3.0 / 4.0) * (3.0 / PI).powf(1.0 / 3.0) * 2f64.powf(1.0 / 3.0);
    let e = -c * (n_up.powf(4.0 / 3.0) + n_dn.powf(4.0 / 3.0));
    (
        e,
        -(6.0 * n_up / PI).powf(1.0 / 3.0),
        -(6.0 * n_dn / PI).powf(1.0 / 3.0),
    )
}

/// Per grid point: XC energy density n·ε_xc, and potentials v_xc,↑ and v_xc,↓.
pub fn exc_point(n_up: f64, n_dn: f64) -> (f64, f64, f64) {
    let (n_up, n_dn) = (n_up.max(0.0), n_dn.max(0.0));
    if n_up + n_dn < N_MIN {
        return (0.0, 0.0, 0.0);
    }
    let (ex, vxu, vxd) = exchange(n_up, n_dn);
    let (ec, vcu, vcd) = correlation(n_up, n_dn);
    (ex + ec, vxu + vcu, vxd + vcd)
}

/// XC energy density and spin potentials on every grid point.
pub struct XcResult {
    pub energy_density: Vec<f64>,
    pub v_up: Vec<f64>,
    pub v_dn: Vec<f64>,
}

/// LSDA (Dirac + PW92) on a whole grid. `n_up` and `n_dn` must have equal length.
pub fn lsda_pw92(n_up: &[f64], n_dn: &[f64]) -> XcResult {
    assert_eq!(n_up.len(), n_dn.len(), "spin densities must have equal length");
    let n = n_up.len();
    let mut r = XcResult {
        energy_density: vec![0.0; n],
        v_up: vec![0.0; n],
        v_dn: vec![0.0; n],
    };
    for i in 0..n {
        let (e, vu, vd) = exc_point(n_up[i], n_dn[i]);
        r.energy_density[i] = e;
        r.v_up[i] = vu;
        r.v_dn[i] = vd;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    fn energy(nu: f64, nd: f64) -> f64 {
        exc_point(nu, nd).0
    }

    #[test]
    fn exchange_unpolarized_is_dirac() {
        // correlation switched off by comparing against the analytic exchange with PW92 correlation subtracted
        let n = 0.1;
        let (e, vu, vd) = exc_point(n / 2.0, n / 2.0);
        let ex = -(3.0 / 4.0) * (3.0 / PI).powf(1.0 / 3.0) * n.powf(4.0 / 3.0);
        let (ec_density, vc) = correlation_only(n / 2.0, n / 2.0);
        assert_relative_eq!(e - ec_density, ex, epsilon = 1e-12);
        assert_relative_eq!(vu - vc, -(3.0 * n / PI).powf(1.0 / 3.0), epsilon = 1e-12);
        assert_relative_eq!(vu, vd, epsilon = 1e-15);
    }

    #[test]
    fn correlation_matches_pw92_reference() {
        // PW92 Table I / libxc LDA_C_PW, unpolarised: ε_c(r_s=2) = −0.04476 Ha, ε_c(r_s=10) = −0.01855 Ha
        for (rs, eps_ref) in [(2.0, -0.04476), (10.0, -0.01855)] {
            let n = 3.0 / (4.0 * PI * rs * rs * rs);
            let (ec_density, _) = correlation_only(n / 2.0, n / 2.0);
            assert_relative_eq!(ec_density / n, eps_ref, epsilon = 5e-4);
        }
    }

    #[test]
    fn fully_polarized_uses_ferromagnetic_branch() {
        let n = 0.02;
        let (e_pol, _, _) = exc_point(n, 0.0);
        assert!(e_pol.is_finite());
        // ζ = 1 ⇒ f = 1, ζ⁴ = 1 ⇒ ε_c = ε_c(r_s, 1); check against the FERRO fit directly
        let rs = (3.0 / (4.0 * PI * n)).powf(1.0 / 3.0);
        let (ec1, _) = pw92_g(rs, &FERRO);
        let (ec_density, _) = correlation_only(n, 0.0);
        assert_relative_eq!(ec_density / n, ec1, epsilon = 1e-12);
    }

    #[test]
    fn potentials_match_finite_difference() {
        for (nu, nd) in [(0.3, 0.3), (0.05, 0.2), (0.001, 0.0004), (0.1, 0.0)] {
            let (_, vu, vd) = exc_point(nu, nd);
            let d = 1e-6 * (nu + nd);
            let fd_u = (energy(nu + d, nd) - energy(nu - d, nd)) / (2.0 * d);
            assert_relative_eq!(vu, fd_u, epsilon = 1e-7, max_relative = 1e-6);
            if nd > 0.0 {
                let fd_d = (energy(nu, nd + d) - energy(nu, nd - d)) / (2.0 * d);
                assert_relative_eq!(vd, fd_d, epsilon = 1e-7, max_relative = 1e-6);
            }
        }
    }

    #[test]
    fn zero_density_gives_zero_not_nan() {
        assert_eq!(exc_point(0.0, 0.0), (0.0, 0.0, 0.0));
        let r = lsda_pw92(&[0.0, 1e-30, 0.1], &[0.0, 0.0, 0.1]);
        assert!(r
            .energy_density
            .iter()
            .chain(&r.v_up)
            .chain(&r.v_dn)
            .all(|x| x.is_finite()));
    }
}
