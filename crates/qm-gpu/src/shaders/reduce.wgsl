// Two-level reductions over the interior of padded fields (layout (i+4) + p*((j+4) + p*(k+4))).
// Level 1 (here): each workgroup of 256 grid-strides over the n³ interior points, accumulates in a
// register, tree-reduces in workgroup memory and writes one f32 partial per workgroup.
// Level 2 (CPU): the partials are summed in f64 (Neumaier) and multiplied by dv.

struct Params { a: f32, b: f32, p: u32, n: u32 }

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> fa: array<f32>;
@group(0) @binding(2) var<storage, read> fb: array<f32>;
@group(0) @binding(3) var<storage, read_write> partials: array<f32>;

var<workgroup> sh: array<f32, 256>;

// Linear interior index (qm layout i + n*(j + n*k)) -> padded index. Must match GpuGrid::pidx.
fn pidx(idx: u32) -> u32 {
    let i = idx % P.n;
    let t = idx / P.n;
    let j = t % P.n;
    let k = t / P.n;
    return (i + 4u) + P.p * ((j + 4u) + P.p * (k + 4u));
}

fn tree_reduce(li: u32, wg: u32, acc: f32) {
    sh[li] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (li < s) { sh[li] = sh[li] + sh[li + s]; }
        workgroupBarrier();
    }
    if (li == 0u) { partials[wg] = sh[0]; }
}

@compute @workgroup_size(256)
fn dot_partials(@builtin(local_invocation_index) li: u32,
                @builtin(workgroup_id) wg: vec3<u32>,
                @builtin(num_workgroups) nw: vec3<u32>) {
    let total = P.n * P.n * P.n;
    let stride = nw.x * 256u;
    var acc = 0.0;
    for (var idx = wg.x * 256u + li; idx < total; idx = idx + stride) {
        let q = pidx(idx);
        acc = acc + fa[q] * fb[q];
    }
    tree_reduce(li, wg.x, acc);
}

@compute @workgroup_size(256)
fn sum_partials(@builtin(local_invocation_index) li: u32,
                @builtin(workgroup_id) wg: vec3<u32>,
                @builtin(num_workgroups) nw: vec3<u32>) {
    let total = P.n * P.n * P.n;
    let stride = nw.x * 256u;
    var acc = 0.0;
    for (var idx = wg.x * 256u + li; idx < total; idx = idx + stride) {
        acc = acc + fa[pidx(idx)];
    }
    tree_reduce(li, wg.x, acc);
}
