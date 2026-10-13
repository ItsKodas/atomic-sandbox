// Analytic potential energy of an electron in the field of Gaussian-smeared nuclei
// (qm::nuclear::nuclear_potential):
//   V(r) = −Σ_I Z_I erf(d_I/(σ√2))/d_I,   d_I = |r − R_I|,   V(R_I) = −Z_I √(2/π)/σ.
// Written as V = −Σ Z·g(x)·(1/(σ√2)) with x = d/(σ√2) and g(x) = erf(x)/x, which is smooth and
// finite at x = 0 (g(0) = 2/√π), so the on-site point needs no branch.
//
// g(x), absolute error vs the exact erf(x)/x (f64 analysis, before f32 rounding):
//   x < 1: Maclaurin series of erf(x)/x to x²⁰ (truncation ≤ 1.1e-9·2/√π at x = 1);
//   x ≥ 1: erf = 1 − erfc with erfc(x) = e^{−x²}·f(x)/(√π x), f a degree-12 Chebyshev fit on
//          [1, 4) and its asymptotic series beyond (`erfc_scaled` below): |Δerf| ≤ 6.1e-9 with
//          the f32 coefficients, i.e. ≤ 7e-9 relative in g. Task 10 replaced Abramowitz–Stegun
//          7.1.26 (|Δerf| ≤ 1.4e-7 on [1, 4], a smooth, sign-changing systematic error — the
//          leading candidate for the H₂O GPU-vs-CPU energy shift; see `validate/references.md`).
// The closed form is not used near the nucleus: an absolute erf error ε becomes ε/x *relative*
// in erf(x)/x and at x = 0 it is 0/0. With the series, the approximation error of g is below
// the f32 rounding of its evaluation (~1 ulp) at every x (measured end-to-end figures: see `physics.rs`).
//
// Precision of d: the nucleus is passed as an integer cell (i32) plus an f32 fraction of a cell,
// so the displacement (i − cell) − frac is an exact integer minus a fraction: d has *relative*
// f32 precision (~1e-7·d) at every distance instead of an absolute 1e-6 Bohr from subtracting
// two ~10 Bohr coordinates. Grid-stride over the interior, ghosts untouched (stay 0).
//
// Reassociation: the `keep` barrier of stencil.wgsl around every Horner step (series and, in
// Task 7, A&S) and the 1 − erfc product was measured to change nothing (identical results), so
// it is not used; `zero` stays in the uniform for parity. Achieved vs qm::nuclear (RX 7900 XTX,
// Task 10 fit): see `physics.rs` (the test comments carry the measured figures).

struct NucParams { p: u32, n: u32, count: u32, zero: u32 /* always 0, unused */, h: f32, inv_s2: f32 /* 1/(σ√2) */, pad0: f32, pad1: f32 }
struct Nucleus { cell: vec4<i32>, frac: vec4<f32> /* .xyz fraction of a cell, .w = Z */ }

@group(0) @binding(0) var<uniform> U: NucParams;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> nuclei: array<Nucleus>;

const TWO_OVER_SQRT_PI: f32 = 1.1283791670955126;
// f(x) = √π·x·e^{x²}·erfc(x) for x ≥ 1 (0.758 at x = 1, → 1 as x → ∞): on [1, 4) a degree-12
// Chebyshev series in t = (2x − 5)/3 (Clenshaw), from 4 on the asymptotic series in 1/x². Fit
// against f64 erfc (Task 10): |Δerf| ≤ 9.7e-10 with f64 coefficients, 6.1e-9 with these f32
// ones (the rounding of CH0 = 1.80 dominates; degree 14/16 change nothing), vs A&S 7.1.26's
// 1.4e-7; the asymptotic branch is 7e-6 relative at x = 4, i.e. 1e-13 absolute in erf.
const SQRT_PI: f32 = 1.7724538509055159;
const CH0: f32 = 1.8048557623e+00;
const CH1: f32 = 9.5036126435e-02;
const CH2: f32 = -3.4561538972e-02;
const CH3: f32 = 1.0820666035e-02;
const CH4: f32 = -3.0682293564e-03;
const CH5: f32 = 8.0784264219e-04;
const CH6: f32 = -2.0040632772e-04;
const CH7: f32 = 4.7291524862e-05;
const CH8: f32 = -1.0687072521e-05;
const CH9: f32 = 2.3243735801e-06;
const CH10: f32 = -4.8843173186e-07;
const CH11: f32 = 9.9470529692e-08;
const CH12: f32 = -1.9682356004e-08;

fn erfc_scaled(x: f32) -> f32 {
    if (x < 4.0) {
        let t = (2.0 * x - 5.0) / 3.0;
        let t2 = 2.0 * t;
        var b1 = 0.0;
        var b2 = 0.0;
        var b0 = 0.0;
        b0 = t2 * b1 - b2 + CH12; b2 = b1; b1 = b0;
        b0 = t2 * b1 - b2 + CH11; b2 = b1; b1 = b0;
        b0 = t2 * b1 - b2 + CH10; b2 = b1; b1 = b0;
        b0 = t2 * b1 - b2 + CH9; b2 = b1; b1 = b0;
        b0 = t2 * b1 - b2 + CH8; b2 = b1; b1 = b0;
        b0 = t2 * b1 - b2 + CH7; b2 = b1; b1 = b0;
        b0 = t2 * b1 - b2 + CH6; b2 = b1; b1 = b0;
        b0 = t2 * b1 - b2 + CH5; b2 = b1; b1 = b0;
        b0 = t2 * b1 - b2 + CH4; b2 = b1; b1 = b0;
        b0 = t2 * b1 - b2 + CH3; b2 = b1; b1 = b0;
        b0 = t2 * b1 - b2 + CH2; b2 = b1; b1 = b0;
        b0 = t2 * b1 - b2 + CH1; b2 = b1; b1 = b0;
        return t * b1 - b2 + 0.5 * CH0;
    }
    let u = 1.0 / (x * x);
    return 1.0 + u * (-0.5 + u * (0.75 + u * (-1.875 + u * (6.5625 + u * (-29.53125)))));
}

// erfc(x) = e^{−x²}·f(x)/(√π x) for x ≥ 1.
fn erfc_ge1(x: f32) -> f32 { return exp(-x * x) * erfc_scaled(x) / (SQRT_PI * x); }
// erf(x)/x = (2/√π) Σ_k c_k x^{2k},  c_k = (−1)^k / (k!(2k+1))
const S1: f32 = -0.3333333333333333;
const S2: f32 = 0.1;
const S3: f32 = -0.023809523809523808;
const S4: f32 = 0.004629629629629629;
const S5: f32 = -0.0007575757575757576;
const S6: f32 = 0.00010683760683760684;
const S7: f32 = -1.3227513227513228e-05;
const S8: f32 = 1.4589169000933706e-06;
const S9: f32 = -1.4503852223150468e-07;
const S10: f32 = 1.3122532963802806e-08;

// erf(x)/x for x ≥ 0.
fn erf_over_x(x: f32) -> f32 {
    if (x < 1.0) {
        let y = x * x;
        var s = S10;
        s = S9 + y * s;
        s = S8 + y * s;
        s = S7 + y * s;
        s = S6 + y * s;
        s = S5 + y * s;
        s = S4 + y * s;
        s = S3 + y * s;
        s = S2 + y * s;
        s = S1 + y * s;
        s = 1.0 + y * s;
        return TWO_OVER_SQRT_PI * s;
    }
    return (1.0 - erfc_ge1(x)) / x;
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
fn nuclear_potential(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let n = U.n;
    let total = n * n * n;
    for (var idx = gid.x; idx < total; idx = idx + nw.x * 256u) {
        let i = vec3<i32>(vec3<u32>(idx % n, (idx / n) % n, idx / (n * n)));
        var v = 0.0;
        for (var a = 0u; a < U.count; a = a + 1u) {
            let nuc = nuclei[a];
            let dc = vec3<f32>(i - nuc.cell.xyz) - nuc.frac.xyz; // exact integer − fraction
            let d = length(dc) * U.h;
            v = v - nuc.frac.w * erf_over_x(d * U.inv_s2);
        }
        dst[pidx_of(idx, n, U.p)] = v * U.inv_s2;
    }
}
