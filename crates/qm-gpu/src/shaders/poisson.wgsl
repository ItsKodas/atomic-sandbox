// Multipole ghost fill for the Poisson solver (qm_gpu::poisson).
//
// For each tabulated ghost cell c (the 6·4·n² face-slab cells the stencil can read; table built
// on the CPU once per grid):
//   dst[cell[c]] = Σ_{lm < 49} q[lm] · table[c·49 + lm],   table[c][lm] = R_lm(r_c) / |r_c|^{2l+1},
// i.e. V_bc(r_c) = Σ_{l ≤ 6, m} Q_lm R_lm(r_c)/|r_c|^{2l+1} with the moments q = Q_lm in f32.
// The sum runs from lm = 48 (l = 6) down to 0 (l = 0): the terms fall off geometrically in l, so
// the small ones are accumulated first. Only the listed ghost cells are written; the interior of
// `dst` (kept at 0 by the Rust side) and the unlisted edge/corner ghosts are untouched.

const N_MOMENTS: u32 = 49u;

struct GhostParams { count: u32, pad0: u32, pad1: u32, pad2: u32 }

@group(0) @binding(0) var<uniform> G: GhostParams;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> cell: array<u32>;
@group(0) @binding(3) var<storage, read> table: array<f32>;
@group(0) @binding(4) var<storage, read> q: array<f32>;

@compute @workgroup_size(256)
fn ghost_fill(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    for (var c = gid.x; c < G.count; c = c + nw.x * 256u) {
        let base = c * N_MOMENTS;
        var s = 0.0;
        for (var lm = N_MOMENTS; lm > 0u; lm = lm - 1u) {
            s = s + q[lm - 1u] * table[base + lm - 1u];
        }
        dst[cell[c]] = s;
    }
}
