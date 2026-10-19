// Ionic species (M7 Task 8a; prefixed by erf.wgsl, kernels.rs SHADERS). The CPU references are qm::nuclear
// (smeared_potential, ionic_local_potential, partial_core_field) and qm::nonlocal (NonlocalOp::{project, apply_add}).
// The local kernels grid-stride over the interior and leave the ghosts untouched; every grid point owns its value
// (no races). The bindings are distinct across the two layouts (0–4 local, 10–17 projectors).
//
// Positions: an ion/centre is an integer cell plus an f32 fraction of a cell (as nuclear.wgsl), so the displacement
// (i − cell) − frac is an exact integer minus a fraction and d keeps *relative* f32 precision at every distance.
//
// Uniform tables (spec §5.1 "Sampling"; Task 5 ruling item 2): f(k·Δ), Δ = h/256, filled exactly on the CPU in f64
// (`FilteredTable::{v_sr_u, rho_pc_u}`) and uploaded as f32. The lookup is `qm::radial::filter::uniform_lagrange`:
// s = d/Δ = |dc|·(h/Δ) (h/Δ = 256 exactly), i₀ = ⌊s⌋ − 1, t = s − i₀, the 4-point Lagrange weights in t, index −1
// mirrored onto index 1 (the tabulated functions are even in d), and 0 once i₀ + 3 reaches the table's end. Never the
// log grid. A value beyond the table radius (s > s_max) is 0, as `FilteredTable::read`.

struct IonParams { p: u32, n: u32, count: u32, pad0: u32, h: f32, scale: f32 /* h/Δ */, pad1: f32, pad2: f32 }
// One smeared centre of V_LR (qm::nuclear::Centre): .w of frac the charge q, .x of inv 1/(σ√2).
struct Centre { cell: vec4<i32>, frac: vec4<f32>, inv: vec4<f32> }
// One pseudised ion: off = (V_SR table offset, its length, ρ_pc table offset, its length — 0 without a partial core),
// lim = (V_SR radius/Δ, ρ_pc radius/Δ, unused, unused).
struct Ion { cell: vec4<i32>, frac: vec4<f32>, off: vec4<u32>, lim: vec4<f32> }

@group(0) @binding(0) var<uniform> U: IonParams;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> centres: array<Centre>;
@group(0) @binding(3) var<storage, read> ions: array<Ion>;
@group(0) @binding(4) var<storage, read> tables: array<f32>;

// Must match GpuGrid::pidx.
fn pidx_of(idx: u32, n: u32, p: u32) -> u32 {
    let i = idx % n;
    let t = idx / n;
    let j = t % n;
    let k = t / n;
    return (i + 4u) + p * ((j + 4u) + p * (k + 4u));
}

// uniform_lagrange (header) on the table at `off` of length `len`, at s = d/Δ ≥ 0.
fn interp(off: u32, len: u32, s: f32) -> f32 {
    let fl = floor(s);
    let i0 = i32(fl) - 1;
    if (i0 + 3 >= i32(len)) { return 0.0; }
    let t = (s - fl) + 1.0; // s − i₀, with the fraction taken exactly first
    let a = tables[off + u32(abs(i0))];
    let b = tables[off + u32(i0 + 1)];
    let c = tables[off + u32(i0 + 2)];
    let e = tables[off + u32(i0 + 3)];
    return a * (-(t - 1.0) * (t - 2.0) * (t - 3.0) / 6.0) + b * (t * (t - 2.0) * (t - 3.0) / 2.0)
         + c * (-t * (t - 1.0) * (t - 3.0) / 2.0) + e * (t * (t - 1.0) * (t - 2.0) / 6.0);
}

fn interior_cell(idx: u32, n: u32) -> vec3<i32> {
    return vec3<i32>(vec3<u32>(idx % n, (idx / n) % n, idx / (n * n)));
}

// V_LR = −Σ_c q_c erf(d/(σ_c√2))/d, the d → 0 limit −q√(2/π)/σ (qm::nuclear::smeared_potential), WRITTEN into dst:
// nuclear.wgsl's form with a per-centre width (count = number of centres).
@compute @workgroup_size(256)
fn smeared_potential(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let n = U.n;
    let total = n * n * n;
    for (var idx = gid.x; idx < total; idx = idx + nw.x * 256u) {
        let i = interior_cell(idx, n);
        var v = 0.0;
        for (var a = 0u; a < U.count; a = a + 1u) {
            let c = centres[a];
            let d = length(vec3<f32>(i - c.cell.xyz) - c.frac.xyz) * U.h;
            v = v - c.frac.w * erf_over_x(d * c.inv.x) * c.inv.x;
        }
        dst[pidx_of(idx, n, U.p)] = v;
    }
}

// dst += Σ_ions V_SR(d) from each ion's one V_SR table within its radius (qm::nuclear::ionic_local_potential's second
// loop; count = number of pseudised ions). Recorded after smeared_potential.
@compute @workgroup_size(256)
fn local_sr(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let n = U.n;
    let total = n * n * n;
    for (var idx = gid.x; idx < total; idx = idx + nw.x * 256u) {
        let i = interior_cell(idx, n);
        var v = 0.0;
        for (var a = 0u; a < U.count; a = a + 1u) {
            let ion = ions[a];
            let s = length(vec3<f32>(i - ion.cell.xyz) - ion.frac.xyz) * U.scale;
            if (s <= ion.lim.x) { v = v + interp(ion.off.x, ion.off.y, s); }
        }
        let q = pidx_of(idx, n, U.p);
        dst[q] = dst[q] + v;
    }
}

// ρ_pc = Σ_ions ρ_pc(d) within each ion's partial-core radius (qm::nuclear::partial_core_field), WRITTEN into dst; an
// ion without a partial core (the no-core branch: off.w = 0) adds nothing.
@compute @workgroup_size(256)
fn nlcc_deposit(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let n = U.n;
    let total = n * n * n;
    for (var idx = gid.x; idx < total; idx = idx + nw.x * 256u) {
        let i = interior_cell(idx, n);
        var r = 0.0;
        for (var a = 0u; a < U.count; a = a + 1u) {
            let ion = ions[a];
            if (ion.off.w == 0u) { continue; }
            let s = length(vec3<f32>(i - ion.cell.xyz) - ion.frac.xyz) * U.scale;
            if (s <= ion.lim.y) { r = r + interp(ion.off.z, ion.off.w, s); }
        }
        dst[pidx_of(idx, n, U.p)] = r;
    }
}

// ---- The Kleinman–Bylander operator (qm::nonlocal; spec §4.6, §5.3; plan Spec question 7) ----
// The point list of an ion is the ball of its largest projector radius (the others' balls are subsets: same centre);
// one *row* per (projector, m) holds β_lm on that list (0 beyond its own radius, exactly as `beta_radial`), computed
// in f64 on the CPU by `NonlocalOp::build` and uploaded as f32. Row = (point offset, point count, β offset, ion),
// s.x = E_KB·h³.
struct ProjParams { chunks: u32, n_rows: u32, base: u32, row0: u32, n_pts: u32, pt_off: u32, nr: u32, pad: u32 }
struct Row { a: vec4<u32>, s: vec4<f32> }

@group(0) @binding(10) var<uniform> P: ProjParams;
@group(0) @binding(11) var<storage, read> points: array<u32>;          // padded indices
@group(0) @binding(12) var<storage, read> beta: array<f32>;
@group(0) @binding(13) var<storage, read> x: array<f32>;               // a padded field
@group(0) @binding(14) var<storage, read_write> partials: array<f32>;  // [base + row·chunks + chunk]
@group(0) @binding(15) var<storage, read> rows: array<Row>;
@group(0) @binding(16) var<storage, read_write> coeffs: array<f32>;    // [row]
@group(0) @binding(17) var<storage, read_write> out: array<f32>;       // a padded field

var<workgroup> acc: array<f32, 256>;

// partials[base + row·chunks + chunk] = Σ_{points of this chunk} β_row x (workgroup tree sum; chunk = workgroup_id.x,
// row = workgroup_id.y; the chunk's points are a grid stride of chunks·256 over the row's list).
@compute @workgroup_size(256)
fn proj_dot(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let row = rows[wid.y];
    var s = 0.0;
    for (var i = wid.x * 256u + lid.x; i < row.a.y; i = i + P.chunks * 256u) {
        s = s + beta[row.a.z + i] * x[points[row.a.x + i]];
    }
    acc[lid.x] = s;
    workgroupBarrier();
    for (var w = 128u; w > 0u; w = w >> 1u) {
        if (lid.x < w) { acc[lid.x] = acc[lid.x] + acc[lid.x + w]; }
        workgroupBarrier();
    }
    if (lid.x == 0u) { partials[P.base + wid.y * P.chunks + wid.x] = acc[0]; }
}

// coeffs[row] = E_KB·h³·Σ_chunks partials[row][·] (≤ 64 chunks; pairwise): E_KB c_lm, finished on the GPU inside an H
// application (plan Spec question 7). One workgroup per row.
@compute @workgroup_size(64)
fn proj_finish(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let r = wid.x;
    var v = 0.0;
    if (lid.x < P.chunks) { v = partials[r * P.chunks + lid.x]; } // guarded: no read past the row (review NOTE-3)
    acc[lid.x] = v;
    workgroupBarrier();
    for (var w = 32u; w > 0u; w = w >> 1u) {
        if (lid.x < w) { acc[lid.x] = acc[lid.x] + acc[lid.x + w]; }
        workgroupBarrier();
    }
    if (lid.x == 0u) { coeffs[r] = rows[r].s.x * acc[0]; }
}

// out[points[i]] += Σ_{rows of one ion} coeffs[row]·β_row[i] over that ion's list (one dispatch per ion, recorded in
// order: overlapping spheres of different ions never race; within one ion every point is distinct).
@compute @workgroup_size(256)
fn proj_axpy(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    for (var i = gid.x; i < P.n_pts; i = i + nw.x * 256u) {
        var s = 0.0;
        for (var r = P.row0; r < P.row0 + P.nr; r = r + 1u) { s = s + coeffs[r] * beta[rows[r].a.z + i]; }
        let q = points[P.pt_off + i];
        out[q] = out[q] + s;
    }
}
