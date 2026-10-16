// Orbital-viewer scalars of ONE orbital (qm_gpu::orbitals, spec §4.2/§5; the CPU references
// are qm::orbitals::{moments, first_sign_functionals, outer_weight, radial_profiles}). One
// dispatch per orbital, plus one "geometry" dispatch (U.geometry = 1) that accumulates the
// ψ-independent normalisers instead. Workgroups [0, mom_wg) and [mom_wg, mom_wg + n_bins) do
// two different reductions:
//
// - Moment workgroups grid-stride over the n³ interior (stride mom_wg·256, as reduce.wgsl) and
//   accumulate 44 lanes per thread: b_lm = ψ·S_lm(dc) (16; dc in grid units — the CPU scales
//   by h^l), c_lm = ψ·Y_lm(dc) (16), the nine windowed first-sign functionals ψ·w_i (spec §5.4
//   item 3: monopole Σ_I e^{−|r−R_I|²}, dipoles (x_a − X̄_a)·g, quadrupoles S_2m(r − X̄)·g with
//   g = e^{−|r−X̄|²}, all in Bohr), and the outer-L/6 weight w_out·ψ² with the per-point
//   cell-fraction weight w_out = 1 − Π_a(1 − f_a) of qm::orbitals::outer_weight; lanes 42–43
//   are unused. The geometry dispatch puts w_i² in the first-sign lanes (‖w_i‖², the
//   functionals' normalisers) and zero elsewhere. Each lane is tree-reduced in workgroup
//   memory, four at a time; one f32 partial per lane per workgroup lands at
//   out[lane·mom_wg + wg]; the CPU finishes in f64 (Neumaier) × dv.
//
// - Bin workgroups (lone atom only; n_bins = 0 otherwise): bin_split workgroups per radial bin
//   k = ⌊|dc|·bin_scale⌋ (bin width RADIAL_BIN_H·h = h/2 about the centre, inside the inscribed
//   sphere |dc| < r_max); workgroup mom_wg + k·bin_split + s is chunk s of bin k. WGSL has no
//   float atomics, so no histogram: a bin's workgroups share out the columns (i, j) of the
//   bin's bounding square (column c goes to chunk ⌊c/256⌋ mod bin_split, thread c mod 256) and,
//   in each column, visit the z points of the shell — two z intervals from the analytic shell
//   bounds widened by BIN_MARGIN — keeping a point only if the CANONICAL bin formula — the same
//   expression in every workgroup — gives k. So the bins partition the points exactly, whatever
//   f32 does at a bin edge; the counts n_k are this formula's (the geometry dispatch's lane 16),
//   and f32 r/dr may put a boundary point in another bin than the f64 CPU reference does.
//   Lanes: 16 sums ψ·Y_lm(dc), then the count (geometry dispatch only), 3 unused; tree-reduced
//   per workgroup, one partial per (bin, chunk) at out[44·mom_wg + 20·(k·bin_split + s) + lane];
//   the CPU sums a bin's bin_split partials in f64 (Neumaier).
//
// Positions are (integer − cell) − frac (the nuclear.wgsl layout): relative f32 precision in dc
// at every distance. Fixed workgroup counts, per-thread orders and tree shapes ⇒ deterministic.

struct MomParams {
    cell: vec4<i32>,   // xyz: integer cell of the centre (the nuclear centroid); w: nuclei count
    frac: vec4<f32>,   // xyz: fraction of a cell; w: h (Bohr)
    n: u32, p: u32, mom_wg: u32, n_bins: u32,
    r_max: f32,        // inscribed-sphere radius about the centre, grid units
    bin_scale: f32,    // 1 / RADIAL_BIN_H
    slab: f32,         // L/6 in grid units, L = (n + 1)h
    geometry: u32,     // 1: the normalisers (‖w_i‖², bin counts) instead of the ψ sums
    bin_split: u32,    // workgroups per radial bin
    pad0: u32, pad1: u32, pad2: u32,
}

struct NucleusRecord { cell: vec4<i32>, frac: vec4<f32> }

@group(0) @binding(0) var<uniform> U: MomParams;
@group(0) @binding(1) var<storage, read> psi: array<f32>;
@group(0) @binding(2) var<storage, read> nuclei: array<NucleusRecord>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> sh: array<vec4<f32>, 256>;

const MOMENT_LANES: u32 = 44u;
const BIN_LANES: u32 = 20u;
// Widening of a bin's shell for the candidate z ranges, grid units. A point of bin k has
// canonical |dc| within ε ≲ 1e-5 of [k, k + 1)/bin_scale (f32 rounding of an O(100) distance);
// the candidate test needs dz² ≤ (r_o + δ)² − ρ², i.e. a slack 2·r·δ ≫ the f32 rounding of
// r_o² − ρ² (≈ 2e-4 at r = 50), which δ = 0.05 gives by 2·50·0.05 = 5 (and by 2·0.5·0.05 =
// 0.05 ≫ 1e-7 at the innermost bins). The integer floor/ceil of the z ends add more.
const BIN_MARGIN: f32 = 0.05;

// real regular solid harmonics, Racah normalisation (qm::poisson::solid_harmonics), l ≤ 3,
// index l² + l + m: a = (0..3), b = (4..7), c = (8..11), d = (12..15)
struct Harm { a: vec4<f32>, b: vec4<f32>, c: vec4<f32>, d: vec4<f32> }

const SQ3: f32 = 1.7320508075688772;
const SQ3_2: f32 = 0.8660254037844386;
const SQ10_4: f32 = 0.7905694150420949;
const SQ15: f32 = 3.872983346207417;
const SQ6_4: f32 = 0.6123724356957945;
const SQ15_2: f32 = 1.9364916731037085;
// √((2l + 1)/4π)
const K0: f32 = 0.28209479177387814;
const K1: f32 = 0.4886025119029199;
const K2: f32 = 0.6307831305050401;
const K3: f32 = 0.7463526651802308;

fn solid(v: vec3<f32>) -> Harm {
    let x = v.x; let y = v.y; let z = v.z;
    let x2 = x * x; let y2 = y * y; let z2 = z * z;
    let r2 = x2 + y2 + z2;
    return Harm(
        vec4<f32>(1.0, y, z, x),
        vec4<f32>(SQ3 * x * y, SQ3 * y * z, 0.5 * (3.0 * z2 - r2), SQ3 * x * z),
        vec4<f32>(SQ3_2 * (x2 - y2), SQ10_4 * (3.0 * x2 - y2) * y, SQ15 * x * y * z, SQ6_4 * y * (5.0 * z2 - r2)),
        vec4<f32>(0.5 * z * (5.0 * z2 - 3.0 * r2), SQ6_4 * x * (5.0 * z2 - r2), SQ15_2 * z * (x2 - y2), SQ10_4 * x * (x2 - 3.0 * y2)),
    );
}

// unit real harmonics Y_lm(v̂) = √((2l+1)/4π)·S_lm(v)/|v|^l; at v = 0 only Y_00 (the CPU's rule)
fn unit(v: vec3<f32>, s: Harm) -> Harm {
    let r = sqrt(dot(v, v));
    if (r == 0.0) { return Harm(vec4<f32>(K0, 0.0, 0.0, 0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0)); }
    let i1 = 1.0 / r;
    let i2 = i1 * i1;
    let i3 = i2 * i1;
    return Harm(s.a * vec4<f32>(K0, K1 * i1, K1 * i1, K1 * i1), s.b * (K2 * i2),
                s.c * vec4<f32>(K2 * i2, K3 * i3, K3 * i3, K3 * i3), s.d * (K3 * i3));
}

// fraction of point q's cell inside the outer L/6 slab of an axis of n points (outer_fraction)
fn outer_frac(q: u32, n: u32) -> f32 {
    let near = f32(min(q + 1u, n - q));
    return clamp(U.slab - (near - 0.5), 0.0, 1.0);
}

// position of grid point (i, j, k) relative to (cell, frac), grid units
fn rel(i: i32, j: i32, k: i32, cell: vec4<i32>, frac: vec4<f32>) -> vec3<f32> {
    return vec3<f32>(f32(i - cell.x) - frac.x, f32(j - cell.y) - frac.y, f32(k - cell.z) - frac.z);
}

fn tree(li: u32, v: vec4<f32>) -> vec4<f32> {
    sh[li] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (li < s) { sh[li] = sh[li] + sh[li + s]; }
        workgroupBarrier();
    }
    let r = sh[0];
    workgroupBarrier();
    return r;
}

fn put_moment(group: u32, v: vec4<f32>, w: u32) {
    let m = U.mom_wg;
    out[(4u * group) * m + w] = v.x;
    out[(4u * group + 1u) * m + w] = v.y;
    out[(4u * group + 2u) * m + w] = v.z;
    out[(4u * group + 3u) * m + w] = v.w;
}

fn put_bin(group: u32, v: vec4<f32>, slot: u32) {
    let o = MOMENT_LANES * U.mom_wg + BIN_LANES * slot + 4u * group;
    out[o] = v.x;
    out[o + 1u] = v.y;
    out[o + 2u] = v.z;
    out[o + 3u] = v.w;
}

@compute @workgroup_size(256)
fn orbital_moments(@builtin(local_invocation_index) li: u32,
                   @builtin(workgroup_id) wg: vec3<u32>) {
    let n = U.n;
    let h = U.frac.w;
    if (wg.x < U.mom_wg) {
        var b0 = vec4<f32>(0.0); var b1 = vec4<f32>(0.0); var b2 = vec4<f32>(0.0); var b3 = vec4<f32>(0.0);
        var c0 = vec4<f32>(0.0); var c1 = vec4<f32>(0.0); var c2 = vec4<f32>(0.0); var c3 = vec4<f32>(0.0);
        var f0 = vec4<f32>(0.0); var f1 = vec4<f32>(0.0); var f2 = vec4<f32>(0.0);
        let total = n * n * n;
        for (var idx = wg.x * 256u + li; idx < total; idx = idx + U.mom_wg * 256u) {
            let i = idx % n;
            let t = idx / n;
            let j = t % n;
            let k = t / n;
            let dc = rel(i32(i), i32(j), i32(k), U.cell, U.frac);
            let d = dc * h;
            var w0 = 0.0;
            for (var a = 0; a < U.cell.w; a = a + 1) {
                let e = rel(i32(i), i32(j), i32(k), nuclei[a].cell, nuclei[a].frac) * h;
                w0 = w0 + exp(-dot(e, e));
            }
            let gw = exp(-dot(d, d));
            let s = solid(dc);
            let h2g = h * h * gw;
            let wa = vec4<f32>(w0, d.x * gw, d.y * gw, d.z * gw);
            let wb = s.b * h2g;   // S_2m(r − X̄)·g, m = −2..1 (S_2 scales as h²)
            let w8 = s.c.x * h2g; // S_22(r − X̄)·g
            if (U.geometry != 0u) {
                f0 = f0 + wa * wa;
                f1 = f1 + wb * wb;
                f2.x = f2.x + w8 * w8;
            } else {
                let v = psi[(i + 4u) + U.p * ((j + 4u) + U.p * (k + 4u))];
                let y = unit(dc, s);
                b0 = b0 + v * s.a; b1 = b1 + v * s.b; b2 = b2 + v * s.c; b3 = b3 + v * s.d;
                c0 = c0 + v * y.a; c1 = c1 + v * y.b; c2 = c2 + v * y.c; c3 = c3 + v * y.d;
                f0 = f0 + v * wa;
                f1 = f1 + v * wb;
                f2.x = f2.x + v * w8;
                let ow = 1.0 - (1.0 - outer_frac(i, n)) * (1.0 - outer_frac(j, n)) * (1.0 - outer_frac(k, n));
                f2.y = f2.y + ow * v * v;
            }
        }
        let r0 = tree(li, b0); if (li == 0u) { put_moment(0u, r0, wg.x); }
        let r1 = tree(li, b1); if (li == 0u) { put_moment(1u, r1, wg.x); }
        let r2 = tree(li, b2); if (li == 0u) { put_moment(2u, r2, wg.x); }
        let r3 = tree(li, b3); if (li == 0u) { put_moment(3u, r3, wg.x); }
        let r4 = tree(li, c0); if (li == 0u) { put_moment(4u, r4, wg.x); }
        let r5 = tree(li, c1); if (li == 0u) { put_moment(5u, r5, wg.x); }
        let r6 = tree(li, c2); if (li == 0u) { put_moment(6u, r6, wg.x); }
        let r7 = tree(li, c3); if (li == 0u) { put_moment(7u, r7, wg.x); }
        let r8 = tree(li, f0); if (li == 0u) { put_moment(8u, r8, wg.x); }
        let r9 = tree(li, f1); if (li == 0u) { put_moment(9u, r9, wg.x); }
        let r10 = tree(li, f2); if (li == 0u) { put_moment(10u, r10, wg.x); }
    } else {
        let kb = (wg.x - U.mom_wg) / U.bin_split;
        let chunk = (wg.x - U.mom_wg) % U.bin_split;
        var a0 = vec4<f32>(0.0); var a1 = vec4<f32>(0.0); var a2 = vec4<f32>(0.0); var a3 = vec4<f32>(0.0);
        var cnt = 0.0;
        let ni = i32(n);
        let cx = f32(U.cell.x) + U.frac.x;
        let cy = f32(U.cell.y) + U.frac.y;
        let cz = f32(U.cell.z) + U.frac.z;
        // the shell's bounds widened by BIN_MARGIN (≫ the f32 error of the canonical r)
        let ro = f32(kb + 1u) / U.bin_scale + BIN_MARGIN;
        let ri = f32(kb) / U.bin_scale - BIN_MARGIN;
        let ilo = max(i32(floor(cx - ro)), 0);
        let ihi = min(i32(ceil(cx + ro)), ni - 1);
        let jlo = max(i32(floor(cy - ro)), 0);
        let jhi = min(i32(ceil(cy + ro)), ni - 1);
        let wi = ihi - ilo + 1;
        let wj = jhi - jlo + 1;
        if (wi > 0 && wj > 0) {
            let cols = u32(wi * wj);
            for (var col = chunk * 256u + li; col < cols; col = col + U.bin_split * 256u) {
                let i = ilo + i32(col % u32(wi));
                let j = jlo + i32(col / u32(wi));
                let dx = f32(i - U.cell.x) - U.frac.x;
                let dy = f32(j - U.cell.y) - U.frac.y;
                let rho2 = dx * dx + dy * dy;
                if (rho2 > ro * ro) { continue; }
                let zh = sqrt(ro * ro - rho2);
                let za = max(i32(floor(cz - zh)), 0);
                let zb = min(i32(ceil(cz + zh)), ni - 1);
                // |z − cz| < zl ⇒ r < ri ⇒ an inner bin: the gap (cz − zl, cz + zl) is skipped
                var e1 = zb;
                var s2 = zb + 1;
                if (ri > 0.0 && ri * ri > rho2) {
                    let zl = sqrt(ri * ri - rho2);
                    e1 = min(i32(ceil(cz - zl)), zb);
                    s2 = max(i32(floor(cz + zl)), e1 + 1);
                }
                for (var part = 0; part < 2; part = part + 1) {
                    var z0 = za;
                    var z1 = e1;
                    if (part == 1) { z0 = s2; z1 = zb; }
                    for (var z = z0; z <= z1; z = z + 1) {
                        let dz = f32(z - U.cell.z) - U.frac.z;
                        // the canonical bin formula (module comment): every workgroup evaluates
                        // exactly this for a point, so each point is in exactly one bin
                        let r = sqrt(dx * dx + dy * dy + dz * dz);
                        if (r >= U.r_max) { continue; }
                        let bin = min(u32(r * U.bin_scale), U.n_bins - 1u);
                        if (bin != kb) { continue; }
                        if (U.geometry != 0u) {
                            cnt = cnt + 1.0;
                        } else {
                            let v = psi[u32(i + 4) + U.p * (u32(j + 4) + U.p * u32(z + 4))];
                            let dcv = vec3<f32>(dx, dy, dz);
                            let y = unit(dcv, solid(dcv));
                            a0 = a0 + v * y.a; a1 = a1 + v * y.b; a2 = a2 + v * y.c; a3 = a3 + v * y.d;
                        }
                    }
                }
            }
        }
        let slot = wg.x - U.mom_wg;
        let q0 = tree(li, a0); if (li == 0u) { put_bin(0u, q0, slot); }
        let q1 = tree(li, a1); if (li == 0u) { put_bin(1u, q1, slot); }
        let q2 = tree(li, a2); if (li == 0u) { put_bin(2u, q2, slot); }
        let q3 = tree(li, a3); if (li == 0u) { put_bin(3u, q3, slot); }
        let q4 = tree(li, vec4<f32>(cnt, 0.0, 0.0, 0.0)); if (li == 0u) { put_bin(4u, q4, slot); }
    }
}
