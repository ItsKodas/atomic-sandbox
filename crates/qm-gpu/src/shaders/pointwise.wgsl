// Pointwise vector ops on padded fields. Every kernel grid-strides over the n³ interior points
// (linear index -> padded index via pidx) and never touches ghost cells, so ghosts stay 0.
// Output buffers must not alias input buffers (enforced on the Rust side).

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

// y += a·x
@compute @workgroup_size(256)
fn axpy(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let total = P.n * P.n * P.n;
    for (var idx = gid.x; idx < total; idx = idx + nw.x * 256u) {
        let q = pidx_of(idx, P.n, P.p);
        y[q] = y[q] + P.a * x[q];
    }
}

// y *= a
@compute @workgroup_size(256)
fn scale(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let total = P.n * P.n * P.n;
    for (var idx = gid.x; idx < total; idx = idx + nw.x * 256u) {
        let q = pidx_of(idx, P.n, P.p);
        y[q] = P.a * y[q];
    }
}

// y = a (interior only; the in-pass replacement for a buffer clear)
@compute @workgroup_size(256)
fn fill(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let total = P.n * P.n * P.n;
    for (var idx = gid.x; idx < total; idx = idx + nw.x * 256u) {
        y[pidx_of(idx, P.n, P.p)] = P.a;
    }
}

// y = x·x2
@compute @workgroup_size(256)
fn mul(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let total = P.n * P.n * P.n;
    for (var idx = gid.x; idx < total; idx = idx + nw.x * 256u) {
        let q = pidx_of(idx, P.n, P.p);
        y[q] = x[q] * x2[q];
    }
}

// y += x² (density accumulate)
@compute @workgroup_size(256)
fn mul_add_sq(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let total = P.n * P.n * P.n;
    for (var idx = gid.x; idx < total; idx = idx + nw.x * 256u) {
        let q = pidx_of(idx, P.n, P.p);
        let v = x[q];
        y[q] = y[q] + v * v;
    }
}

// lincomb: y = (accumulate ? y : 0) + Σ_{t < count} c[t]·x_t, count ≤ 7. Own uniform at binding 4.
struct LincombParams { n: u32, p: u32, count: u32, accumulate: u32, c: array<vec4<f32>, 2> }

@group(0) @binding(4) var<uniform> L: LincombParams;
@group(0) @binding(5) var<storage, read> t0: array<f32>;
@group(0) @binding(6) var<storage, read> t1: array<f32>;
@group(0) @binding(7) var<storage, read> t2: array<f32>;
@group(0) @binding(8) var<storage, read> t3: array<f32>;
@group(0) @binding(9) var<storage, read> t4: array<f32>;
@group(0) @binding(10) var<storage, read> t5: array<f32>;
@group(0) @binding(11) var<storage, read> t6: array<f32>;

@compute @workgroup_size(256)
fn lincomb(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let total = L.n * L.n * L.n;
    for (var idx = gid.x; idx < total; idx = idx + nw.x * 256u) {
        let q = pidx_of(idx, L.n, L.p);
        var s = 0.0;
        if (L.accumulate != 0u) { s = y[q]; }
        s = s + L.c[0].x * t0[q];
        if (L.count > 1u) { s = s + L.c[0].y * t1[q]; }
        if (L.count > 2u) { s = s + L.c[0].z * t2[q]; }
        if (L.count > 3u) { s = s + L.c[0].w * t3[q]; }
        if (L.count > 4u) { s = s + L.c[1].x * t4[q]; }
        if (L.count > 5u) { s = s + L.c[1].y * t5[q]; }
        if (L.count > 6u) { s = s + L.c[1].z * t6[q]; }
        y[q] = s;
    }
}
