// Hirshfeld shares of up to ROWS displayed orbitals on up to ATOMS nuclei per workgroup row
// (qm_gpu::orbitals::OrbitalView::shares, module doc "Orbital shares"; the CPU f64 reference is
// sim::charges::{Weights::at, orbital_shares_many}):
//
//   s_iA = h³ Σ_nodes w_A(r_n) ψ̃_i(r_n)²,   w_A = ρ_A⁰(|r − R_A|) / Σ_B ρ_B⁰(|r − R_B|).
//
// One dispatch per block of ROWS orbitals (bindings 4..7; slots ≥ U.rows re-bind the first and
// are ignored); workgroup row wg.y is the block of nuclei [ATOMS·wg.y, ATOMS·wg.y + ATOMS).
//
// The free-atom lookup is sim::charges' Lookup: ρ⁰ on r = k·dr (dr = h/16), linear in ρ between
// entries i = ⌊r/dr⌋ and i + 1, the last entry where i + 1 ≥ the length. f32 cannot hold the
// tails (ρ⁰ < 1e-38 inside the cell — and the CPU's ρ⁰ reaches 1e-300), so the table holds
// L_k = ln ρ_k (f32; ZERO for an exact ρ_k = 0) and the same linear-in-ρ interpolation is done
// in logs: ln((1 − f)ρ_i + fρ_{i+1}) = m + ln((1 − f)e^{L_i − m} + f e^{L_{i+1} − m}), m =
// max(L_i, L_{i+1}). The weights are then exp(L_A − L_max) / Σ_B exp(L_B − L_max) (an online
// max-and-sum, one pass over the nuclei) — the CPU's ratio, without f32 underflow. Where every
// ρ_A⁰ is exactly 0 (the CPU's Σ = 0: a tail below f64's range, a table cut to 0) the nearest
// nucleus takes the node whole (strict <: the lower index on a tie, as Weights::raw).
//
// Nodes where every bound ψ̃ is 0 are skipped (adding 0 changes no sum, as the CPU). Each thread
// grid-strides the interior (stride wgs·256) into 2 vec4 accumulators per row (ATOMS = 8 lanes),
// each tree-reduced in workgroup memory; one f32 partial per (atom block, row, lane, workgroup)
// at out[((wg.y·ROWS + row)·ATOMS + lane)·wgs + wg.x]. The CPU finishes in f64 (Neumaier) × h³.
// Positions are (integer − cell) − frac in grid units (the nuclear.wgsl layout). Fixed workgroup
// counts, per-thread orders and tree shapes ⇒ deterministic.

struct ShareParams {
    n: u32, p: u32, wgs: u32, rows: u32,
    atoms: u32,
    scale: f32,      // h/dr: lookup steps per grid unit (16)
    pad0: u32, pad1: u32,
}

// a nucleus: its cell and fraction (grid units), its table's (offset, length) in `tables`
struct AtomRecord { cell: vec4<i32>, frac: vec4<f32>, table: vec4<u32> }

@group(0) @binding(0) var<uniform> U: ShareParams;
@group(0) @binding(1) var<storage, read> tables: array<f32>;
@group(0) @binding(2) var<storage, read> atoms: array<AtomRecord>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> psi0: array<f32>;
@group(0) @binding(5) var<storage, read> psi1: array<f32>;
@group(0) @binding(6) var<storage, read> psi2: array<f32>;
@group(0) @binding(7) var<storage, read> psi3: array<f32>;

const ROWS: u32 = 4u;
const ATOMS: u32 = 8u;
// ln 0: an exact zero of the f64 table (any L below ZERO_BELOW is one)
const ZERO: f32 = -1.0e30;
const ZERO_BELOW: f32 = -1.0e29;

var<workgroup> sh: array<vec4<f32>, 256>;

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

fn put(row: u32, half: u32, v: vec4<f32>, wy: u32, wx: u32) {
    let w = U.wgs;
    let base = (wy * ROWS + row) * ATOMS + 4u * half;
    out[base * w + wx] = v.x;
    out[(base + 1u) * w + wx] = v.y;
    out[(base + 2u) * w + wx] = v.z;
    out[(base + 3u) * w + wx] = v.w;
}

// ln ρ_A⁰ at x = r/dr (Lookup::at in logs; ZERO for ρ = 0)
fn log_rho(t: vec4<u32>, x: f32) -> f32 {
    let len = t.y;
    let i = u32(x);
    if (i + 1u >= len) { return tables[t.x + len - 1u]; }
    let f = x - f32(i);
    let a = tables[t.x + i];
    let b = tables[t.x + i + 1u];
    let m = max(a, b);
    if (m < ZERO_BELOW) { return ZERO; }
    let s = exp(a - m) * (1.0 - f) + exp(b - m) * f;
    if (s <= 0.0) { return ZERO; }
    return m + log(s);
}

@compute @workgroup_size(256)
fn orbital_shares(@builtin(local_invocation_index) li: u32,
                  @builtin(workgroup_id) wg: vec3<u32>) {
    let n = U.n;
    let first = wg.y * ATOMS;
    var acc: array<vec4<f32>, 8>;   // [row·2 + half]
    for (var q = 0u; q < 8u; q = q + 1u) { acc[q] = vec4<f32>(0.0); }
    let total = n * n * n;
    for (var idx = wg.x * 256u + li; idx < total; idx = idx + U.wgs * 256u) {
        let i = idx % n;
        let t = idx / n;
        let j = t % n;
        let k = t / n;
        let pi = (i + 4u) + U.p * ((j + 4u) + U.p * (k + 4u));
        var v = vec4<f32>(psi0[pi], psi1[pi], psi2[pi], psi3[pi]);
        v = select(vec4<f32>(0.0), v, vec4<u32>(0u, 1u, 2u, 3u) < vec4<u32>(U.rows));
        if (all(v == vec4<f32>(0.0))) { continue; }
        let p = v * v;
        // one pass over the nuclei: L_max, Σ exp(L − L_max), the nearest nucleus, this block's L
        var lmax = ZERO;
        var sum = 0.0;
        var nearest = 0u;
        var best = 3.4e38;
        var lb0 = vec4<f32>(ZERO);
        var lb1 = vec4<f32>(ZERO);
        for (var a = 0u; a < U.atoms; a = a + 1u) {
            let rec = atoms[a];
            let d = vec3<f32>(f32(i32(i) - rec.cell.x) - rec.frac.x, f32(i32(j) - rec.cell.y) - rec.frac.y, f32(i32(k) - rec.cell.z) - rec.frac.z);
            let r2 = dot(d, d);
            if (r2 < best) { best = r2; nearest = a; }
            let l = log_rho(rec.table, sqrt(r2) * U.scale);
            if (a >= first && a < first + ATOMS) {
                let s = a - first;
                if (s < 4u) { lb0[s] = l; } else { lb1[s - 4u] = l; }
            }
            if (l < ZERO_BELOW) { continue; }
            if (l > lmax) { sum = sum * exp(lmax - l) + 1.0; lmax = l; } else { sum = sum + exp(l - lmax); }
        }
        var w0 = vec4<f32>(0.0);
        var w1 = vec4<f32>(0.0);
        if (sum > 0.0) {
            let inv = 1.0 / sum;
            w0 = select(vec4<f32>(0.0), exp(lb0 - lmax) * inv, lb0 >= vec4<f32>(ZERO_BELOW));
            w1 = select(vec4<f32>(0.0), exp(lb1 - lmax) * inv, lb1 >= vec4<f32>(ZERO_BELOW));
        } else if (nearest >= first && nearest < first + ATOMS) {
            // every ρ⁰ is 0 here: the nearest nucleus takes the node
            let s = nearest - first;
            if (s < 4u) { w0[s] = 1.0; } else { w1[s - 4u] = 1.0; }
        }
        acc[0] = acc[0] + w0 * p.x; acc[1] = acc[1] + w1 * p.x;
        acc[2] = acc[2] + w0 * p.y; acc[3] = acc[3] + w1 * p.y;
        acc[4] = acc[4] + w0 * p.z; acc[5] = acc[5] + w1 * p.z;
        acc[6] = acc[6] + w0 * p.w; acc[7] = acc[7] + w1 * p.w;
    }
    for (var q = 0u; q < 8u; q = q + 1u) {
        let r = tree(li, acc[q]);
        if (li == 0u) { put(q / 2u, q % 2u, r, wg.y, wg.x); }
    }
}
