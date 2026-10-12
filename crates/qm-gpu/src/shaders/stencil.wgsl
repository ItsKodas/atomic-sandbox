// 8th-order central Laplacian on padded fields:
//   dst = scale · Σ_{m=1..4} c_m Σ_axes ((f[q+m·s] − f[q]) + (f[q−m·s] − f[q])),
// scale = 1/h² (Laplacian) or −1/(2h²) (kinetic). One thread per interior point, workgroup (8,8,4).
//
// Coefficients are qm::stencil::COEFF (c = [−205/72, 8/5, −1/5, 8/315, −1/560]). The centre term is
// folded into second differences using c0 = −2(c1 + c2 + c3 + c4), which holds exactly in real
// arithmetic, so this is the same operator as the CPU's `3·c0·f + Σ c_m (f₊ + f₋)`; in f32 it is
// far more accurate: a constant field gives exactly 0 (an f32-rounded c0 would leave ~1e-7·f/h²),
// and the large centre/neighbour cancellation happens in f₊ − f, which is exact (Sterbenz) whenever
// f₊ is within a factor 2 of f, i.e. for any smooth field.
//
// Reassociation barrier: the AMD Vulkan driver (LLPC, 26.8.1) compiles naga's SPIR-V with fast-math
// reassociation — WGSL has no `precise` and naga emits no NoContraction — and was measured to
// redistribute these differences back into the large-cancellation form (up to >100 ulp, 1.2e-5
// absolute at 33³, h = 0.25, vs ≤ 1e-7 for the IEEE evaluation of this code). `keep` is an identity
// the compiler cannot see through (OR with a uniform that is 0 at run time), so each exact
// difference is materialised before it is summed. Summation of the (small) differences may still
// be reordered; that only perturbs the result at the ulp level of the differences.
//
// The buffer is padded by 4 zero ghost cells per side, so every q ± m·s (m ≤ 4) is in bounds and
// reads 0 outside the box: Dirichlet for free, no branches in the stencil. Only interior points are
// written, so the output's ghosts stay 0. `dst` must not alias `f` (enforced on the Rust side).

struct StencilParams { scale: f32, p: u32, n: u32, zero: u32 /* always 0: see `keep` */ }

@group(0) @binding(0) var<uniform> S: StencilParams;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> f: array<f32>;

const C1: f32 = 1.6;              //  8/5
const C2: f32 = -0.2;             // −1/5
const C3: f32 = 0.025396825;      //  8/315
const C4: f32 = -0.0017857143;    // −1/560

// Identity that the shader compiler cannot reassociate across (S.zero == 0 at run time).
fn keep(x: f32) -> f32 { return bitcast<f32>(bitcast<u32>(x) | S.zero); }

// f[q+s] − f[q] + f[q−s] − f[q] with both (exact) differences materialised.
fn sd(q: u32, s: u32, fq: f32) -> f32 { return keep(f[q + s] - fq) + keep(f[q - s] - fq); }

// Σ over the three axes of the m-th second difference.
fn d2(q: u32, m: u32, sy: u32, sz: u32, fq: f32) -> f32 {
    return (sd(q, m, fq) + sd(q, m * sy, fq)) + sd(q, m * sz, fq);
}

@compute @workgroup_size(8, 8, 4)
fn laplacian(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = S.n;
    if (gid.x >= n || gid.y >= n || gid.z >= n) { return; }
    let p = S.p;
    let sz = p * p;
    let q = (gid.x + 4u) + p * ((gid.y + 4u) + p * (gid.z + 4u)); // must match GpuGrid::pidx
    let fq = f[q];
    // smallest coefficients first
    var acc = C4 * d2(q, 4u, p, sz, fq);
    acc = acc + C3 * d2(q, 3u, p, sz, fq);
    acc = acc + C2 * d2(q, 2u, p, sz, fq);
    acc = acc + C1 * d2(q, 1u, p, sz, fq);
    dst[q] = acc * S.scale;
}
