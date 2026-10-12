// LSDA exchange-correlation per grid point: Dirac exchange (spin-scaled) + Perdew–Wang 1992
// correlation, transcribed line by line from qm::xc::{exchange, correlation, pw92_g, f_zeta}
// (see that module for the derivation) into f32. Writes n·ε_xc, v_xc,↑, v_xc,↓ on the interior;
// ghosts untouched (stay 0). Points with n↑ + n↓ < N_MIN (1e-18 here vs 1e-20 on the CPU) are
// vacuum: exactly 0 in all three outputs. Negative inputs are clamped to 0, as on the CPU.
//
// f32 deviations from a literal transcription, all for accuracy:
// - ln(1 + 1/Q₁) is evaluated as log1p(u), u = 1/Q₁, via 2·atanh(u/(2+u)) (odd series, u ≤ 1).
//   For large r_s (low density) u ≪ 1 and 1 + u rounds in f32 (u = 2e-3 at r_s = 100 loses
//   ~3e-5 relative; u < 6e-8 gives ln = 0), and the GPU `log` is only specified to 2⁻²¹
//   *absolute* near 1. The series has neither problem. Measured with plain log(1 + 1/Q₁): v off
//   by 8.7e-6 relative on the smooth test density, 2.1e-4 in an empty spin channel, 7.9e-3 at
//   n ~ 4e-10 (vs ≤ 1.5e-6 everywhere with the series).
// - f(ζ), f′(ζ) take 1 ± ζ computed directly as 2n↑/n and 2n↓/n, not 1 ± (n↑ − n↓)/n. f′ has an
//   infinite slope at |ζ| = 1 ((1 ∓ ζ)^{1/3}); GPU division is only 2.5 ulp, so (n↑ − 0)/n came out
//   as 1 − 6e-8 and the empty channel's v_c was 2.4e-3 relative off. 2n↓/n is exactly 0 when
//   n↓ = 0 and keeps relative precision when n↓ ≪ n↑.
// - cbrt(x) = pow(x, 1/3) plus one Newton step for normal x > 0, exactly 0 for x ≤ 0 and
//   denormal x (pow(0, y) is undefined in WGSL; a flushed pow would make the Newton step 0/0). GPU pow = exp2(y·log2 x) loses ~|log2 x|·ulp: without the Newton step the
//   potentials were 6e-7 relative off on smooth densities and 2e-6 at n ~ 1e-9; with it 1.7e-7.
//
// Reassociation (the AMD LLPC fast-math issue of stencil.wgsl): the `keep` barrier was measured
// around every order-sensitive sub-expression here (log1p and Newton Horner steps, the f(ζ)
// numerator, ε_c(r_s,1) − ε_c(r_s,0)) and changed nothing beyond noise (worst v 1.49e-6 → 1.37e-6
// relative), so it is not used; `zero` stays in the uniform for parity with the stencil kernel.
//
// Achieved vs qm::xc::lsda_pw92 (RX 7900 XTX): ≤ 2.2e-7 relative in n·ε_xc everywhere; ≤ 1.7e-7 in
// v on smooth densities; ≤ 1.5e-6 in v at the two f32-conditioned spots: the empty channel at
// |ζ| = 1 (ε_c(r_s,1) − ε_c(r_s,0) cancels ×2.5, then ∂ε_c/∂ζ cancels ×2.2) and n ~ 1e-8 there.

struct XcParams { p: u32, n: u32, zero: u32 /* always 0, unused (see above) */, pad: u32 }

@group(0) @binding(0) var<uniform> X: XcParams;
@group(0) @binding(1) var<storage, read> rho_up: array<f32>;
@group(0) @binding(2) var<storage, read> rho_dn: array<f32>;
@group(0) @binding(3) var<storage, read_write> exc: array<f32>;
@group(0) @binding(4) var<storage, read_write> v_up: array<f32>;
@group(0) @binding(5) var<storage, read_write> v_dn: array<f32>;

const N_MIN: f32 = 1e-18;
const C_X: f32 = 0.9305257363491;        // (3/4)(3/π)^{1/3}·2^{1/3}
const CBRT_6_OVER_PI: f32 = 1.2407009817988; // (6/π)^{1/3}
const RS_FAC: f32 = 0.6203504908994;     // (3/(4π))^{1/3}: r_s = RS_FAC / n^{1/3}
const FZ_DENOM: f32 = 0.5198420997897464; // 2^{4/3} − 2
const FPP0: f32 = 1.709921;              // f″(0)

fn cbrt(x: f32) -> f32 {
    // Below the smallest normal f32 (and for x ≤ 0) return 0: pow(denormal, 1/3) may flush to 0
    // (log2 of a denormal), and the Newton step would then divide by 0 ⇒ inf/NaN. Denormal
    // densities do occur (tails of a nearly empty spin channel's Σψ²); v_x there is ~1e-13.
    if (x < 1.1754944e-38) { return 0.0; }
    let y = pow(x, 1.0 / 3.0);
    return y - (y - x / (y * y)) / 3.0;
}

// ln(1 + u) for u ≥ 0.
fn log1p_pos(u: f32) -> f32 {
    if (u > 1.0) { return log(1.0 + u); }
    // ln(1+u) = 2 atanh(s) = 2(s + s³/3 + s⁵/5 + …), s = u/(2+u) ≤ 1/3: truncation after s¹⁷/17
    // is ≤ (1/9)⁹/19 relative ≈ 1.4e-10.
    let s = u / (2.0 + u);
    let s2 = s * s;
    var q = 1.0 / 17.0;
    q = 1.0 / 15.0 + s2 * q;
    q = 1.0 / 13.0 + s2 * q;
    q = 1.0 / 11.0 + s2 * q;
    q = 1.0 / 9.0 + s2 * q;
    q = 1.0 / 7.0 + s2 * q;
    q = 1.0 / 5.0 + s2 * q;
    q = 1.0 / 3.0 + s2 * q;
    q = 1.0 + s2 * q;
    return 2.0 * s * q;
}

// (G, dG/dr_s) of the PW92 fit function with parameters (A, α₁, β₁..β₄).
fn pw92_g(rs: f32, a: f32, a1: f32, b1: f32, b2: f32, b3: f32, b4: f32) -> vec2<f32> {
    let srs = sqrt(rs);
    let q0 = -2.0 * a * (1.0 + a1 * rs);
    let q1 = 2.0 * a * (b1 * srs + b2 * rs + b3 * rs * srs + b4 * rs * rs);
    let q1p = a * (b1 / srs + 2.0 * b2 + 3.0 * b3 * srs + 4.0 * b4 * rs);
    let ln = log1p_pos(1.0 / q1);
    return vec2<f32>(q0 * ln, -2.0 * a * a1 * ln - q0 * q1p / (q1 * q1 + q1));
}

// (f(ζ), f′(ζ)) from opz = 1 + ζ, omz = 1 − ζ.
fn f_zeta(opz: f32, omz: f32) -> vec2<f32> {
    let cp = cbrt(opz);
    let cm = cbrt(omz);
    let f = (opz * cp + omz * cm - 2.0) / FZ_DENOM;
    let fp = (4.0 / 3.0) * (cp - cm) / FZ_DENOM;
    return vec2<f32>(f, fp);
}

// (n·ε_c, v_c,↑, v_c,↓) for n = n↑ + n↓ ≥ N_MIN.
fn correlation(n_up: f32, n_dn: f32) -> vec3<f32> {
    let n = n_up + n_dn;
    let rs = RS_FAC / cbrt(n);
    let z = clamp((n_up - n_dn) / n, -1.0, 1.0);
    let g0 = pw92_g(rs, 0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294);   // PARA
    let g1 = pw92_g(rs, 0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517);  // FERRO
    let gm = pw92_g(rs, 0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671);  // MALPHA (−α_c)
    let ec0 = g0.x;
    let dec0 = g0.y;
    let ec1 = g1.x;
    let dec1 = g1.y;
    let ac = -gm.x;
    let dac = -gm.y;
    let fz = f_zeta(min(2.0 * n_up / n, 2.0), min(2.0 * n_dn / n, 2.0));
    let f = fz.x;
    let fp = fz.y;
    let z3 = z * z * z;
    let z4 = z3 * z;
    let ec = ec0 + ac * f / FPP0 * (1.0 - z4) + (ec1 - ec0) * f * z4;
    let dec_drs = dec0 + dac * f / FPP0 * (1.0 - z4) + (dec1 - dec0) * f * z4;
    let dec_dz = ac / FPP0 * (fp * (1.0 - z4) - 4.0 * z3 * f) + (ec1 - ec0) * (fp * z4 + 4.0 * z3 * f);
    let vc_common = ec - rs / 3.0 * dec_drs;
    return vec3<f32>(n * ec, vc_common - (z - 1.0) * dec_dz, vc_common - (z + 1.0) * dec_dz);
}

// (e_x, v_x,↑, v_x,↓), spin-scaled Dirac exchange.
fn exchange(n_up: f32, n_dn: f32) -> vec3<f32> {
    let cu = cbrt(n_up);
    let cd = cbrt(n_dn);
    let e = -C_X * (n_up * cu + n_dn * cd);
    return vec3<f32>(e, -CBRT_6_OVER_PI * cu, -CBRT_6_OVER_PI * cd);
}

// Must match GpuGrid::pidx.
fn pidx_of(idx: u32, n: u32, p: u32) -> u32 {
    let i = idx % n;
    let t = idx / n;
    let j = t % n;
    let k = t / n;
    return (i + 4u) + p * ((j + 4u) + p * (k + 4u));
}

@compute @workgroup_size(256)
fn xc_lsda(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let total = X.n * X.n * X.n;
    for (var idx = gid.x; idx < total; idx = idx + nw.x * 256u) {
        let q = pidx_of(idx, X.n, X.p);
        let nu = max(rho_up[q], 0.0);
        let nd = max(rho_dn[q], 0.0);
        var r = vec3<f32>(0.0);
        if (nu + nd >= N_MIN) {
            r = exchange(nu, nd) + correlation(nu, nd);
        }
        exc[q] = r.x;
        v_up[q] = r.y;
        v_dn[q] = r.z;
    }
}
