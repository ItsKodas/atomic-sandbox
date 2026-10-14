// Trilinear sampling of a padded f32 field at a list of points (qm_gpu::sample).
//
// The CPU resolves each point with qm::Grid::trilinear_cell in f64 — the padded index of its
// cell's lower corner (`base`, 0xFFFFFFFF for a point outside the grid) and the fractions
// t per axis as f32 — in [0, 1) in f64, [0, 1] after rounding, exactly 0 on a node (on an upper
// face t = 0 and the +1 neighbours are the zero ghosts)
// — so the cell choice is the CPU reference's exactly and the shader
// holds no origin or spacing. One thread per point (grid-stride beyond 65535 workgroups): 8
// reads, blended a·(1 − t) + b·t along x, then y, then z (the order of Grid::trilinear).

struct SampleParams { p: u32, count: u32, pad0: u32, pad1: u32 }
struct Point { t: vec3<f32>, base: u32 }

@group(0) @binding(0) var<uniform> P: SampleParams;
@group(0) @binding(1) var<storage, read> field: array<f32>;
@group(0) @binding(2) var<storage, read> points: array<Point>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const OUTSIDE: u32 = 0xFFFFFFFFu;

fn lerp(a: f32, b: f32, t: f32) -> f32 { return a * (1.0 - t) + b * t; }

// 64 = qm_gpu::sample::SAMPLE_WORKGROUP (also the stride below): change both together.
@compute @workgroup_size(64)
fn sample_density(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let stride = nw.x * 64u;
    let sy = P.p;
    let sz = P.p * P.p;
    for (var i = gid.x; i < P.count; i = i + stride) {
        let q = points[i];
        if (q.base == OUTSIDE) {
            out[i] = 0.0;
            continue;
        }
        let b = q.base;
        let x00 = lerp(field[b], field[b + 1u], q.t.x);
        let x10 = lerp(field[b + sy], field[b + sy + 1u], q.t.x);
        let x01 = lerp(field[b + sz], field[b + sz + 1u], q.t.x);
        let x11 = lerp(field[b + sz + sy], field[b + sz + sy + 1u], q.t.x);
        out[i] = lerp(lerp(x00, x10, q.t.y), lerp(x01, x11, q.t.y), q.t.z);
    }
}
