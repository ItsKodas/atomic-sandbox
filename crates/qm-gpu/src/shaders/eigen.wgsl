// LOBPCG preconditioner kernels (see eigen.rs): the first damped-Jacobi sweep w₀ = D⁻¹ r with
// D = max(diag(−½∇²) + V − λ, 0.1). The later sweeps are `jacobi_step` in stencil.wgsl (they
// need the stencil of w). Grid-strides over the n³ interior and never touches ghost cells.
//
// Not order-sensitive: the preconditioner only has to be a symmetric positive approximation of
// (H − λ)⁻¹ on the high end of the spectrum, and driver reassociation perturbs it at the ulp
// level, which LOBPCG's Rayleigh–Ritz step absorbs. No `keep` barrier.

struct Params { a: f32, b: f32, p: u32, n: u32 }

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> x2: array<f32>;

// Must match GpuGrid::pidx.
fn pidx_of(idx: u32, n: u32, p: u32) -> u32 {
    let i = idx % n;
    let t = idx / n;
    let j = t % n;
    let k = t / n;
    return (i + 4u) + p * ((j + 4u) + p * (k + 4u));
}

// y = x / max(a + x2 − b, 0.1): x = r, x2 = V, a = diag(−½∇²) = −1.5·c₀/h², b = λ, with the 0.1
// floor of qm::eigen (keeps D positive where V − λ < −a).
@compute @workgroup_size(256)
fn precond_init(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let total = P.n * P.n * P.n;
    for (var idx = gid.x; idx < total; idx = idx + nw.x * 256u) {
        let q = pidx_of(idx, P.n, P.p);
        y[q] = x[q] / max(P.a + x2[q] - P.b, 0.1);
    }
}
