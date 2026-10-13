// Geometric multigrid kernels for the Poisson preconditioner (qm_gpu::multigrid — read its module
// doc first). Every level is a padded field (layout (i+4) + p*((j+4) + p*(k+4)), ghosts 0); the
// operator on every level is the 2nd-order 7-point Laplacian at that level's spacing H,
//   A x = −(Σ_axes (x₊ − 2x + x₋))/H²,
// with the Dirichlet boundary a distance d·H beyond the first/last point of each axis (d_low on
// the low side, d_high on the high side, the same for the three axes). d = 1 is the zero ghost;
// otherwise the ghost is the linear extrapolation to zero at the boundary, x_ghost = g·x₀ with
// g = 1 − 1/d, so a boundary point's stencil is −(x_in + g x₀ − 2x₀)/H² and its diagonal
// (2 − g)/H² per touched face. One thread per output point in (8, 8, 4) workgroups; only
// interior points are written.
//
// Not order-sensitive: a preconditioner has to be a symmetric positive approximation of A⁻¹,
// and ulp-level reassociation by the driver perturbs it as such; CG sees the exact A through
// the 8th-order stencil kernel. No `keep` barrier.

struct MgParams {
    n: u32, p: u32,      // this level (for restrict: the coarse output grid)
    nf: u32, pf: u32,    // the fine grid of restrict / prolong
    h2: f32,             // H² of the level the operator acts on
    omega: f32,          // Jacobi damping
    g_low: f32,          // ghost factors 1 − 1/d on the low and high faces
    g_high: f32,
}

@group(0) @binding(0) var<uniform> M: MgParams;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;

fn pidx(i: u32, j: u32, k: u32, p: u32) -> u32 { return (i + 4u) + p * ((j + 4u) + p * (k + 4u)); }

// Σ_axes (x₊ + x₋) over the in-grid neighbours (ghost reads are 0) and the diagonal factor
// 6 − Σ_touched faces g: with the extrapolated ghosts g·x₀ the stencil is
//   H²·A x = (6 − Σg)·x − Σ_in x_nb = diag·x − nb.
struct Stencil { nb: f32, diag: f32 }

fn stencil(gid: vec3<u32>, q: u32, p: u32, n: u32) -> Stencil {
    let sz = p * p;
    let nb = ((x[q + 1u] + x[q - 1u]) + (x[q + p] + x[q - p])) + (x[q + sz] + x[q - sz]);
    var g = 0.0;
    if (gid.x == 0u) { g = g + M.g_low; }
    if (gid.y == 0u) { g = g + M.g_low; }
    if (gid.z == 0u) { g = g + M.g_low; }
    if (gid.x == n - 1u) { g = g + M.g_high; }
    if (gid.y == n - 1u) { g = g + M.g_high; }
    if (gid.z == n - 1u) { g = g + M.g_high; }
    return Stencil(nb, 6.0 - g);
}

// One damped Jacobi sweep on A x = b, D = diag/H²:
//   dst = x + ω (b − A x)/D = x + (ω/diag) (H² b + nb − diag·x).
@compute @workgroup_size(8, 8, 4)
fn mg_jacobi(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = M.n;
    if (gid.x >= n || gid.y >= n || gid.z >= n) { return; }
    let q = pidx(gid.x, gid.y, gid.z, M.p);
    let s = stencil(gid, q, M.p, n);
    let xq = x[q];
    dst[q] = xq + (M.omega / s.diag) * (M.h2 * b[q] + s.nb - s.diag * xq);
}

// dst = b − A x = b + (nb − diag·x)/H².
@compute @workgroup_size(8, 8, 4)
fn mg_residual(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = M.n;
    if (gid.x >= n || gid.y >= n || gid.z >= n) { return; }
    let q = pidx(gid.x, gid.y, gid.z, M.p);
    let s = stencil(gid, q, M.p, n);
    dst[q] = b[q] + (s.nb - s.diag * x[q]) / M.h2;
}

// Full weighting of the fine field x (grid nf, pf) onto the coarse point (i, j, k), which sits
// at fine (2i+1, 2j+1, 2k+1): Σ over the 27 fine neighbours with weight Π_axes (½ at offset 0,
// ¼ at ±1). A fine read at index nf hits the zero ghost (the Dirichlet residual is 0 there).
// dst is the coarse grid (M.n, M.p).
@compute @workgroup_size(8, 8, 4)
fn mg_restrict(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = M.n;
    if (gid.x >= n || gid.y >= n || gid.z >= n) { return; }
    let pf = M.pf;
    let qf = pidx(2u * gid.x + 1u, 2u * gid.y + 1u, 2u * gid.z + 1u, pf);
    let sz = pf * pf;
    var s = 0.0;
    for (var dz = 0u; dz < 3u; dz = dz + 1u) {
        let wz = select(0.25, 0.5, dz == 1u);
        for (var dy = 0u; dy < 3u; dy = dy + 1u) {
            let wy = select(0.25, 0.5, dy == 1u);
            for (var dx = 0u; dx < 3u; dx = dx + 1u) {
                let wx = select(0.25, 0.5, dx == 1u);
                let q = qf + (dx - 1u) + pf * (dy - 1u) + sz * (dz - 1u);
                s = s + wx * wy * wz * x[q];
            }
        }
    }
    dst[pidx(gid.x, gid.y, gid.z, M.p)] = s;
}

// Trilinear prolongation of the coarse field x (grid M.n = nc, M.p = pc) added onto the fine
// point (i, j, k) of dst (grid nf, pf): an odd fine index 2c+1 reads coarse c with weight 1, an
// even index 2c the two coarse neighbours c − 1 and c with weight ½ each; coarse index −1 or nc
// is the coarse zero ghost (one coarse spacing beyond — the fine boundary for the first point,
// and the right place to within the boundary offset for the last).
@compute @workgroup_size(8, 8, 4)
fn mg_prolong_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nf = M.nf;
    if (gid.x >= nf || gid.y >= nf || gid.z >= nf) { return; }
    let pc = M.p;
    let sz = pc * pc;
    // odd index 2c+1 → corner c only; even index 2c → corners c−1 and c (c−1 = −1 is the ghost
    // at padded index 3); c = index/2 in both cases
    let oi = (gid.x & 1u) == 1u;
    let oj = (gid.y & 1u) == 1u;
    let ok = (gid.z & 1u) == 1u;
    // padded index of corner c; the other corner (even case) is at −1 along that axis
    let q0 = pidx(gid.x / 2u, gid.y / 2u, gid.z / 2u, pc);
    var s = 0.0;
    for (var c = 0u; c < 8u; c = c + 1u) {
        let mi = (c & 1u) == 1u; let mj = (c & 2u) == 2u; let mk = (c & 4u) == 4u; // use the c−1 corner along the axis
        if ((mi && oi) || (mj && oj) || (mk && ok)) { continue; }
        let w = select(1.0, 0.5, !oi) * select(1.0, 0.5, !oj) * select(1.0, 0.5, !ok);
        let q = q0 - select(0u, 1u, mi) - select(0u, pc, mj) - select(0u, sz, mk);
        s = s + w * x[q];
    }
    let qf = pidx(gid.x, gid.y, gid.z, M.pf);
    dst[qf] = dst[qf] + s;
}
