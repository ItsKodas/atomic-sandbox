// LOBPCG preconditioner kernels (see eigen.rs): the Jacobi diagonal and one damped sweep.
// Both grid-stride over the n³ interior and never touch ghost cells (they stay 0).
//
// Neither is order-sensitive: the preconditioner only has to be a symmetric positive
// approximation of (H − λ)⁻¹ on the high end of the spectrum, and driver reassociation
// perturbs it at the ulp level, which LOBPCG's Rayleigh–Ritz step absorbs. No `keep` barrier.

struct Params { a: f32, b: f32, p: u32, n: u32 }

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> x2: array<f32>;
@group(0) @binding(4) var<storage, read> x3: array<f32>;

// Must match GpuGrid::pidx.
fn pidx_of(idx: u32, n: u32, p: u32) -> u32 {
    let i = idx % n;
    let t = idx / n;
    let j = t % n;
    let k = t / n;
    return (i + 4u) + p * ((j + 4u) + p * (k + 4u));
}

// y = 1 / max(a + x − b, 0.1): inverse Jacobi diagonal of H − λ with a = diag(−½∇²) = −1.5·c₀/h²,
// x = V, b = λ, and the 0.1 floor of qm::eigen (keeps D positive where V − λ < −a).
@compute @workgroup_size(256)
fn inv_diag(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let total = P.n * P.n * P.n;
    for (var idx = gid.x; idx < total; idx = idx + nw.x * 256u) {
        let q = pidx_of(idx, P.n, P.p);
        y[q] = 1.0 / max(P.a + x[q] - P.b, 0.1);
    }
}

// One damped Jacobi sweep on (H − λ) w = r:  y = w, x = D⁻¹, x2 = r, x3 = H w, a = ω, b = λ:
//   w += ω · D⁻¹ · (r − (H w − λ w))
@compute @workgroup_size(256)
fn jacobi_sweep(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let total = P.n * P.n * P.n;
    for (var idx = gid.x; idx < total; idx = idx + nw.x * 256u) {
        let q = pidx_of(idx, P.n, P.p);
        let w = y[q];
        y[q] = w + P.a * x[q] * (x2[q] - (x3[q] - P.b * w));
    }
}
