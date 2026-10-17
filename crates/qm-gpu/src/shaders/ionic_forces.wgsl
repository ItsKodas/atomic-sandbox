// Pseudised Hellmann–Feynman force terms (M7 Task 8b). CPU references: qm::forces::{local_lr_forces,
// local_sr_forces, nonlocal_forces, partial_core_forces} (read the `qm::forces` module doc, "Pseudised ions (M7)",
// for what is differentiated, and `qm_gpu::ionic`'s module doc, "Forces", for the layout). Atomic units.
//
// G(x) below is a VERBATIM copy of forces.wgsl's TWO_OVER_SQRT_PI, SQRT_PI, CH0..CH12, erfc_scaled, erfc_ge1,
// S1..S12 and g_over_x (their accuracy analysis is in forces.wgsl's header and `qm_gpu::forces`' module doc).
// forces.wgsl itself is not touched, so the IonicPolicy::None force path stays bitwise; keep the two copies identical.
//
// Why a second force module: the pseudised terms are odd integrands that cancel to a small net — a lone O's
// partial-core term is ~1e-6 Ha/Bohr out of a gross Σ_p|t_p| ~ 1 (Σ|t|/|F| ≈ 2e6, `qm::forces`
// `partial_core_force_is_the_radial_lattice_sum`) — so a plain f32 sum (relative error ~u·Σ|t| per level of the tree,
// u = 6e-8) can be a sizeable fraction of the term. Every sum here is therefore a compensated (double-f32) sum: Knuth's
// TwoSum on each addition, the error words accumulated beside the sum and carried through the workgroup tree; the CPU
// finishes hi + lo in f64 (Neumaier). What is left is the rounding of each product (random, ~u·sqrt(Σ t²)) and of the
// inputs. Reassociation: the AMD Vulkan compiler applies fast-math reassociation to naga's SPIR-V (stencil.wgsl), which
// would fold TwoSum's (s − a) − b back to 0; `keep4` (an OR with the uniform `zero`, 0 at run time) materialises every
// intermediate.
//
// Output: one vec4 pair (hi, lo) per workgroup in `out4` at `base + 2·slot` (lanes xyz the force components, w the
// projection c of the nonlocal rows; 0 where unused). Fixed workgroup counts, per-thread order and tree shape ⇒
// deterministic.

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

// ---- the compensated accumulator ----

// A double-f32 sum on four lanes: hi + lo, |lo| ≲ u·|hi|.
struct Acc { hi: vec4<f32>, lo: vec4<f32> }

// Identity the shader compiler cannot see through (`zero` == 0 at run time; stencil.wgsl's `keep` on four lanes).
fn keep4(v: vec4<f32>, zero: u32) -> vec4<f32> { return bitcast<vec4<f32>>(bitcast<vec4<u32>>(v) | vec4<u32>(zero)); }

// x + b with Knuth's TwoSum (s + e = x.hi + b exactly; every intermediate materialised), the error into lo.
fn acc_add(x: Acc, b: vec4<f32>, zero: u32) -> Acc {
    let s = keep4(x.hi + b, zero);
    let bb = keep4(s - x.hi, zero);
    let e = keep4(x.hi - keep4(s - bb, zero), zero) + keep4(b - bb, zero);
    return Acc(s, x.lo + e);
}

// x + y: TwoSum on the high words, the low words added.
fn acc_merge(x: Acc, y: Acc, zero: u32) -> Acc {
    let r = acc_add(x, y.hi, zero);
    return Acc(r.hi, keep4(r.lo + y.lo, zero));
}

var<workgroup> sh_hi: array<vec4<f32>, 256>;
var<workgroup> sh_lo: array<vec4<f32>, 256>;

// The workgroup tree of compensated partials; the workgroup's sum is left in (sh_hi[0], sh_lo[0]).
fn tree(li: u32, a: Acc, zero: u32) {
    sh_hi[li] = a.hi;
    sh_lo[li] = a.lo;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (li < s) {
            let m = acc_merge(Acc(sh_hi[li], sh_lo[li]), Acc(sh_hi[li + s], sh_lo[li + s]), zero);
            sh_hi[li] = m.hi;
            sh_lo[li] = m.lo;
        }
        workgroupBarrier();
    }
}

// ---- local, long range: F_I^LR = h³ Σ_p ρ_p V_LR′(d_p)(r_p − R_I)/d_p for one centre (q, σ) ----
// forces.wgsl's integrand with the centre's own charge and width (qm::forces::local_lr_forces: (Z_v, σ_LR) for a
// pseudised ion, (Z, σ_AE) for an all-electron species in the setup): ρ_p · c · G(x_p) · dc_p, c = −q h/a³,
// x_p = |dc_p|·(h/a), a = σ√2, dc_p = (i − cell) − frac (exact integer minus f32 fraction). Grid-stride over the n³
// interior, as force_partials; one (hi, lo) pair per workgroup at base + 2·wg.
struct CentreForceParams {
    cell: vec4<i32>,
    frac: vec4<f32>,
    p: u32, n: u32, base: u32, zero: u32,
    h_over_a: f32, c: f32, pad0: f32, pad1: f32,
}

@group(0) @binding(0) var<uniform> C: CentreForceParams;
@group(0) @binding(1) var<storage, read> rho: array<f32>;
@group(0) @binding(2) var<storage, read_write> out4: array<vec4<f32>>;

@compute @workgroup_size(256)
fn centre_force(@builtin(local_invocation_index) li: u32,
                @builtin(workgroup_id) wg: vec3<u32>,
                @builtin(num_workgroups) nw: vec3<u32>) {
    let n = C.n;
    let total = n * n * n;
    let stride = nw.x * 256u;
    var a = Acc(vec4<f32>(0.0), vec4<f32>(0.0));
    for (var idx = wg.x * 256u + li; idx < total; idx = idx + stride) {
        let iu = vec3<u32>(idx % n, (idx / n) % n, idx / (n * n));
        let q = (iu.x + 4u) + C.p * ((iu.y + 4u) + C.p * (iu.z + 4u)); // GpuGrid::pidx
        let dc = vec3<f32>(vec3<i32>(iu) - C.cell.xyz) - C.frac.xyz;   // exact integer − fraction
        let x = length(dc) * C.h_over_a;
        let w = rho[q] * C.c * g_over_x(x);
        a = acc_add(a, keep4(vec4<f32>(w * dc, 0.0), C.zero), C.zero);
    }
    tree(li, a, C.zero);
    if (li == 0u) {
        out4[C.base + 2u * wg.x] = sh_hi[0];
        out4[C.base + 2u * wg.x + 1u] = sh_lo[0];
    }
}

// ---- gathers over point lists: Σ_{p in the row's list} w_p · x_p, four lanes ----
// One row = (point offset, point count, weight offset, nucleus) in `rows`; its weights w4[w_off + i] belong to the
// padded grid index points[pt_off + i]. The weights are formed on the CPU in f64 per geometry (`ionic::ForceBufs`):
// - local SR (x = ρ): (V_SR′(d)/d)(r − R), 0 (qm::forces::local_sr_forces, over the V_SR table's sphere);
// - partial core (x = v̄ = ½(v_xc↑ + v_xc↓) on ρ_σ + ½ρ_pc): (ρ_pc′(d)/d)(r − R), 0 (partial_core_forces);
// - nonlocal (x = ψ_i): (∇β_lm, β_lm), so .xyz = ⟨∇β_lm|ψ_i⟩/h³ and .w = c_lm/h³ (nonlocal_forces).
// Workgroup (chunk, row): chunk = workgroup_id.x strides the row's list by chunks·256; row = row0 + workgroup_id.y; the
// pair lands at base + 2·(workgroup_id.y·chunks + chunk).
struct GatherParams { chunks: u32, row0: u32, base: u32, zero: u32 }

@group(0) @binding(10) var<uniform> G: GatherParams;
@group(0) @binding(11) var<storage, read> points: array<u32>;
@group(0) @binding(12) var<storage, read> w4: array<vec4<f32>>;
@group(0) @binding(13) var<storage, read> x: array<f32>;
@group(0) @binding(15) var<storage, read> rows: array<vec4<u32>>;

@compute @workgroup_size(256)
fn gather_dot4(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let row = rows[G.row0 + wid.y];
    var a = Acc(vec4<f32>(0.0), vec4<f32>(0.0));
    for (var i = wid.x * 256u + li; i < row.y; i = i + G.chunks * 256u) {
        a = acc_add(a, keep4(w4[row.z + i] * x[points[row.x + i]], G.zero), G.zero);
    }
    tree(li, a, G.zero);
    if (li == 0u) {
        let o = G.base + 2u * (wid.y * G.chunks + wid.x);
        out4[o] = sh_hi[0];
        out4[o + 1u] = sh_lo[0];
    }
}
