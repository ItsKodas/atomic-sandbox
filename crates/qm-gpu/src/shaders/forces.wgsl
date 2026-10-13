// Hellmann-Feynman electronic force partials for ONE nucleus (qm::forces in the GPU form of
// forces.rs — read its module doc first):
//   F_I^el = ∫ ρ ∇_r V_I d³r,  V_I = −Z erf(d/a)/d,  a = σ√2,  d = |r − R_I|,  x = d/a,
//   ∇_r V_I = −(Z/a³) · G(x) · (r − R_I),   G(x) = E′(x)/x,  E(x) = erf(x)/x.
// On the grid r_p − R_I = h·dc_p with dc_p = (i − cell) − frac (exact integer minus f32
// fraction, nuclear.wgsl's layout: relative f32 precision in d at every distance), so the term
// of grid point p is  ρ_p · c · G(x_p) · dc_p,  c = −Z h/a³,  x_p = |dc_p|·(h/a).
// Nothing divides by d: on the nucleus dc = 0 and the term is exactly 0.
//
// G(x), even and finite at 0 (G(0) = −(2/√π)·2/3):
//   x < 1: Maclaurin series (2/√π) Σ_{k=1..12} s_k x^{2k−2}, s_k = 2k c_k = (−1)^k·2/((k−1)!(2k+1)),
//          truncation ≤ 1.7e-10 at x = 1 (4e-10 relative);
//   x ≥ 1: G(x) = [(2/√π) x e^{−x²} − 1 + erfc(x)]/x³ with erfc from the same Chebyshev/
//          asymptotic fit as nuclear.wgsl (`erfc_ge1`, |Δerf| ≤ 6.1e-9; Task 10 replaced A&S
//          7.1.26's 1.4e-7). The bracket is −0.4276 at x = 1 and → −1, so no cancellation; the
//          fit error is ≤ 1.5e-8 relative at the switch and below the f32 rounding of the
//          closed form (~2e-7) everywhere.
// The step at x = 1 is now the f32 rounding of the two branches. Measured on this shader
// (forces.rs `shader_g_over_x_probe`, ρ = 1 at one grid point): see the test's comments for
// the Task 10 figures (Task 7 with A&S: step −4.05e-7 = 5.8 ulps, max 5.0e-7 at x = 1.024);
// x = 0 gives exactly 0. The Rust mirror in forces.rs uses these same decimal literals.
//
// Reduction: as reduce.wgsl — each workgroup of 256 grid-strides over the n³ interior,
// accumulates a vec3 per thread, tree-reduces in workgroup memory and writes one vec4 partial
// (xyz, 0) per workgroup; the CPU sums the partials in f64 (Neumaier) × dv. Fixed workgroup
// count, per-thread order and tree shape ⇒ deterministic. No `keep` barrier: the only sums are
// the loop-carried accumulator and the barrier-separated tree (neither can be reassociated by
// the driver), and the closed-form bracket has no cancellation to protect.

struct ForceParams {
    cell: vec4<i32>,          // integer cell of the nucleus (xyz)
    frac: vec4<f32>,          // fraction of a cell (xyz); w unused
    p: u32, n: u32, pad0: u32, pad1: u32,
    h_over_a: f32,            // h/(σ√2): x = |dc|·h_over_a
    c: f32,                   // −Z h/a³
    pad2: f32, pad3: f32,
}

@group(0) @binding(0) var<uniform> U: ForceParams;
@group(0) @binding(1) var<storage, read> rho: array<f32>;
@group(0) @binding(2) var<storage, read_write> partials: array<vec4<f32>>;

var<workgroup> sh: array<vec3<f32>, 256>;

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
// s_k = (−1)^k · 2/((k−1)!(2k+1)), k = 1..12
const S1: f32 = -2.0 / 3.0;
const S2: f32 = 2.0 / 5.0;
const S3: f32 = -1.0 / 7.0;
const S4: f32 = 1.0 / 27.0;
const S5: f32 = -1.0 / 132.0;
const S6: f32 = 1.0 / 780.0;
const S7: f32 = -1.0 / 5400.0;
const S8: f32 = 1.0 / 42840.0;
const S9: f32 = -1.0 / 383040.0;
const S10: f32 = 1.0 / 3810240.0;
const S11: f32 = -1.0 / 41731200.0;
const S12: f32 = 1.0 / 498960000.0;

// G(x) = E′(x)/x for x ≥ 0 (forces.rs module doc).
fn g_over_x(x: f32) -> f32 {
    if (x < 1.0) {
        let y = x * x;
        var s = S12;
        s = S11 + y * s;
        s = S10 + y * s;
        s = S9 + y * s;
        s = S8 + y * s;
        s = S7 + y * s;
        s = S6 + y * s;
        s = S5 + y * s;
        s = S4 + y * s;
        s = S3 + y * s;
        s = S2 + y * s;
        s = S1 + y * s;
        return TWO_OVER_SQRT_PI * s;
    }
    return ((TWO_OVER_SQRT_PI * x * exp(-x * x) - 1.0) + erfc_ge1(x)) / (x * x * x);
}

@compute @workgroup_size(256)
fn force_partials(@builtin(local_invocation_index) li: u32,
                  @builtin(workgroup_id) wg: vec3<u32>,
                  @builtin(num_workgroups) nw: vec3<u32>) {
    let n = U.n;
    let total = n * n * n;
    let stride = nw.x * 256u;
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    for (var idx = wg.x * 256u + li; idx < total; idx = idx + stride) {
        let iu = vec3<u32>(idx % n, (idx / n) % n, idx / (n * n));
        let q = (iu.x + 4u) + U.p * ((iu.y + 4u) + U.p * (iu.z + 4u)); // GpuGrid::pidx
        let dc = vec3<f32>(vec3<i32>(iu) - U.cell.xyz) - U.frac.xyz;   // exact integer − fraction
        let x = length(dc) * U.h_over_a;
        let w = rho[q] * U.c * g_over_x(x);
        acc = acc + w * dc;
    }
    sh[li] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (li < s) { sh[li] = sh[li] + sh[li + s]; }
        workgroupBarrier();
    }
    if (li == 0u) { partials[wg.x] = vec4<f32>(sh[0], 0.0); }
}
