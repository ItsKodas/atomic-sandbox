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
//
// Fused entries (the LOBPCG hot path, eigen.rs): `hamiltonian` = −½∇²f + V·f in one pass, and
// `jacobi_step`, one damped Jacobi sweep on (H − λ) w = r with H applied in place:
//   dst = w + ω/max(a + V − λ, 0.1) · (r − (−½∇²w + V·w − λ·w)),   a = diag(−½∇²) = −1.5 c₀/h².
// Both read the stencil of `f` (= w) through the same `d2`, so their kinetic part is bitwise the
// `kinetic` entry's. They use their own uniform (binding 4) so the plain `laplacian` layout is
// unchanged; `keep` takes the uniform's zero as an argument for that reason.

struct StencilParams { scale: f32, p: u32, n: u32, zero: u32 /* always 0: see `keep` */ }
struct HamParams { scale: f32, p: u32, n: u32, zero: u32, omega: f32, lambda: f32, diag_t: f32, pad: f32 }

@group(0) @binding(0) var<uniform> S: StencilParams;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> f: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<uniform> H: HamParams;
@group(0) @binding(5) var<storage, read> r: array<f32>;

const C1: f32 = 1.6;              //  8/5
const C2: f32 = -0.2;             // −1/5
const C3: f32 = 0.025396825;      //  8/315
const C4: f32 = -0.0017857143;    // −1/560

// Identity that the shader compiler cannot reassociate across (`zero` == 0 at run time).
fn keep(x: f32, zero: u32) -> f32 { return bitcast<f32>(bitcast<u32>(x) | zero); }

// f[q+s] − f[q] + f[q−s] − f[q] with both (exact) differences materialised.
fn sd(q: u32, s: u32, fq: f32, zero: u32) -> f32 { return keep(f[q + s] - fq, zero) + keep(f[q - s] - fq, zero); }

// Σ over the three axes of the m-th second difference.
fn d2(q: u32, m: u32, sy: u32, sz: u32, fq: f32, zero: u32) -> f32 {
    return (sd(q, m, fq, zero) + sd(q, m * sy, fq, zero)) + sd(q, m * sz, fq, zero);
}

// Σ_m c_m · (m-th second difference), smallest coefficients first; × 1/h² is the Laplacian.
fn lap_unscaled(q: u32, p: u32, fq: f32, zero: u32) -> f32 {
    let sz = p * p;
    var acc = C4 * d2(q, 4u, p, sz, fq, zero);
    acc = acc + C3 * d2(q, 3u, p, sz, fq, zero);
    acc = acc + C2 * d2(q, 2u, p, sz, fq, zero);
    acc = acc + C1 * d2(q, 1u, p, sz, fq, zero);
    return acc;
}

@compute @workgroup_size(8, 8, 4)
fn laplacian(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = S.n;
    if (gid.x >= n || gid.y >= n || gid.z >= n) { return; }
    let p = S.p;
    let q = (gid.x + 4u) + p * ((gid.y + 4u) + p * (gid.z + 4u)); // must match GpuGrid::pidx
    dst[q] = lap_unscaled(q, p, f[q], S.zero) * S.scale;
}

// dst = scale·Σ c_m d2 + v·f, scale = −1/(2h²): H f = −½∇²f + V f in one dispatch.
@compute @workgroup_size(8, 8, 4)
fn hamiltonian(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = H.n;
    if (gid.x >= n || gid.y >= n || gid.z >= n) { return; }
    let p = H.p;
    let q = (gid.x + 4u) + p * ((gid.y + 4u) + p * (gid.z + 4u));
    let fq = f[q];
    // fma: the V·f product (|V| ≈ 20 Ha on a shifted problem) is not rounded on its own — the
    // kinetic product rounds at its own small scale and the sum once at |H f|. Without it the
    // extra ½-ulp of |V||f| in every stored H x showed up as a λ-dependent residual floor
    // (`floor_independent_of_eigenvalue_shift`: +22 % instead of +7 % under a −20 Ha shift).
    dst[q] = fma(v[q], fq, lap_unscaled(q, p, fq, H.zero) * H.scale);
}

// dst = f + ω · (1/max(diag_t + v − λ, 0.1)) · (r − (H f − λ f)), H f = scale·Σ c_m d2 + v·f.
// One damped Jacobi sweep of eigen.rs's preconditioner (f = the current w; dst must not alias
// f, so sweeps ping-pong between two fields).
@compute @workgroup_size(8, 8, 4)
fn jacobi_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = H.n;
    if (gid.x >= n || gid.y >= n || gid.z >= n) { return; }
    let p = H.p;
    let q = (gid.x + 4u) + p * ((gid.y + 4u) + p * (gid.z + 4u));
    let w = f[q];
    let vq = v[q];
    // (V − λ)·w in one fma: V and λ nearly cancel on a shifted problem
    let hw = fma(vq, w, lap_unscaled(q, p, w, H.zero) * H.scale);
    let inv_d = 1.0 / max(H.diag_t + vq - H.lambda, 0.1);
    dst[q] = w + H.omega * inv_d * (r[q] - (hw - H.lambda * w));
}
