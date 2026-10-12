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

// y += x·x2 (e.g. H ψ = −½∇²ψ + V·ψ: `kinetic` into y, then y += V·ψ)
@compute @workgroup_size(256)
fn mul_add(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let total = P.n * P.n * P.n;
    for (var idx = gid.x; idx < total; idx = idx + nw.x * 256u) {
        let q = pidx_of(idx, P.n, P.p);
        y[q] = y[q] + x[q] * x2[q];
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

// clamp_min: y = max(y, a)
@compute @workgroup_size(256)
fn clamp_min(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let total = P.n * P.n * P.n;
    for (var idx = gid.x; idx < total; idx = idx + nw.x * 256u) {
        let q = pidx_of(idx, P.n, P.p);
        y[q] = max(y[q], P.a);
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

// diis_mix: the Pulay mixing step of scf.rs evaluated about the latest input density ("base"):
//   y = (accumulate ? y : base) + Σ_{t < count} c[t]·(sub_mask bit t ? keep(x_t − base) : x_t),
//   then max(y, 0) if clamp != 0.
// Why not a plain lincomb Σ cᵢ ρᵢ: the cᵢ have both signs with Σ cᵢ = 1, so the partial sums
// are |c|·ρ-sized and every f32 rounding leaves ~6e-8·|c|·ρ of noise in the next input density —
// a Δρ floor of ~3e-7·|c|·N that competes with tol = 1e-6. Here the small terms (differences
// ρᵢ − base of size Δρ, and the residuals) are accumulated first in `acc` and added to the
// base once, so the result rounds once at ρ-size (~2e-8·ρ). `keep` (an OR with the uniform
// `zero`, which is 0 at run time; see stencil.wgsl) stops the driver from distributing
// c·(x − base) into c·x − c·base, which would reintroduce the cancellation.
// base at binding 2 (x), terms at bindings 5..10 (t0..t5), own uniform at binding 12.
struct DiisParams { n: u32, p: u32, count: u32, accumulate: u32, sub_mask: u32, clamp: u32, zero: u32, pad: u32, c: array<vec4<f32>, 2> }

@group(0) @binding(12) var<uniform> D: DiisParams;

fn keep_d(v: f32) -> f32 { return bitcast<f32>(bitcast<u32>(v) | D.zero); }

fn dterm(v: f32, base: f32, t: u32) -> f32 {
    if ((D.sub_mask & (1u << t)) != 0u) { return keep_d(v - base); }
    return v;
}

@compute @workgroup_size(256)
fn diis_mix(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let total = D.n * D.n * D.n;
    for (var idx = gid.x; idx < total; idx = idx + nw.x * 256u) {
        let q = pidx_of(idx, D.n, D.p);
        let b = x[q];
        var acc = 0.0;
        if (D.count > 0u) { acc = acc + D.c[0].x * dterm(t0[q], b, 0u); }
        if (D.count > 1u) { acc = acc + D.c[0].y * dterm(t1[q], b, 1u); }
        if (D.count > 2u) { acc = acc + D.c[0].z * dterm(t2[q], b, 2u); }
        if (D.count > 3u) { acc = acc + D.c[0].w * dterm(t3[q], b, 3u); }
        if (D.count > 4u) { acc = acc + D.c[1].x * dterm(t4[q], b, 4u); }
        if (D.count > 5u) { acc = acc + D.c[1].y * dterm(t5[q], b, 5u); }
        var s = b;
        if (D.accumulate != 0u) { s = y[q]; }
        s = s + acc;
        if (D.clamp != 0u) { s = max(s, 0.0); }
        y[q] = s;
    }
}

// unpad: interior of the padded field x → the row-padded buffer y that a buffer→texture copy
// reads: y[i + stride·(j + n·k)] = x[pidx(i, j, k)], `stride` floats per row (a multiple of 64
// so each row is a multiple of wgpu's 256-byte COPY_BYTES_PER_ROW_ALIGNMENT). Own uniform at 13.
struct UnpadParams { n: u32, p: u32, stride: u32, pad: u32 }

@group(0) @binding(13) var<uniform> U: UnpadParams;

@compute @workgroup_size(256)
fn unpad(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let n = U.n;
    let total = n * n * n;
    for (var idx = gid.x; idx < total; idx = idx + nw.x * 256u) {
        let i = idx % n;
        let t = idx / n;
        let j = t % n;
        let k = t / n;
        y[i + U.stride * (j + n * k)] = x[(i + 4u) + U.p * ((j + 4u) + U.p * (k + 4u))];
    }
}
