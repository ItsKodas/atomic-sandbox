// Analytic potential energy of an electron in the field of Gaussian-smeared nuclei
// (qm::nuclear::nuclear_potential):
//   V(r) = −Σ_I Z_I erf(d_I/(σ√2))/d_I,   d_I = |r − R_I|,   V(R_I) = −Z_I √(2/π)/σ.
// Written as V = −Σ Z·g(x)·(1/(σ√2)) with x = d/(σ√2) and g(x) = erf(x)/x, which is smooth and
// finite at x = 0 (g(0) = 2/√π), so the on-site point needs no branch.
//
// g(x), absolute error vs the exact erf(x)/x (f64 analysis, before f32 rounding):
//   x < 1: Maclaurin series of erf(x)/x to x²⁰ (truncation ≤ 1.1e-9·2/√π at x = 1);
//   x ≥ 1: Abramowitz–Stegun 7.1.26, erf(x) = 1 − t(a1 + t(a2 + t(a3 + t(a4 + t·a5))))·e^{−x²},
//          t = 1/(1 + p·x), |error in erf| ≤ 1.5e-7, i.e. ≤ 1.5e-7/erf(1) = 1.8e-7 relative.
// A&S alone is not used near the nucleus: its 1.5e-7 *absolute* erf error becomes 1.5e-7/x
// *relative* in erf(x)/x (5e-5 at 1e-3 Bohr from a nucleus with σ = 0.25), and at x = 0 it is
// 0/0. With the series the approximation error of g is ≤ ~2e-7 relative at every x, about two
// f32 ulps (measured end-to-end figures: see `physics.rs`).
//
// Precision of d: the nucleus is passed as an integer cell (i32) plus an f32 fraction of a cell,
// so the displacement (i − cell) − frac is an exact integer minus a fraction: d has *relative*
// f32 precision (~1e-7·d) at every distance instead of an absolute 1e-6 Bohr from subtracting
// two ~10 Bohr coordinates. Grid-stride over the interior, ghosts untouched (stay 0).
//
// Reassociation: the `keep` barrier of stencil.wgsl around every Horner step (series and A&S) and
// the 1 − poly·e^{−x²} product was measured to change nothing (identical results), so it is not
// used; `zero` stays in the uniform for parity. Achieved vs qm::nuclear (RX 7900 XTX): ≤ 2.3e-7
// relative pointwise, 1.3e-7 of max|V| at 33³, 2e-8 at a nucleus on a grid point.

struct NucParams { p: u32, n: u32, count: u32, zero: u32 /* always 0, unused */, h: f32, inv_s2: f32 /* 1/(σ√2) */, pad0: f32, pad1: f32 }
struct Nucleus { cell: vec4<i32>, frac: vec4<f32> /* .xyz fraction of a cell, .w = Z */ }

@group(0) @binding(0) var<uniform> U: NucParams;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> nuclei: array<Nucleus>;

const TWO_OVER_SQRT_PI: f32 = 1.1283791670955126;
// A&S 7.1.26
const AS_P: f32 = 0.3275911;
const AS_A1: f32 = 0.254829592;
const AS_A2: f32 = -0.284496736;
const AS_A3: f32 = 1.421413741;
const AS_A4: f32 = -1.453152027;
const AS_A5: f32 = 1.061405429;
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
    let t = 1.0 / (1.0 + AS_P * x);
    let poly = t * (AS_A1 + t * (AS_A2 + t * (AS_A3 + t * (AS_A4 + t * AS_A5))));
    return (1.0 - poly * exp(-x * x)) / x;
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
