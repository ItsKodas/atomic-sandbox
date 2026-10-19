// The boundary-surface orbitals (user request 2026-10-19; display only): per displayed orbital the
// level of ψ̃²/peak above which the grid holds `share` of the orbital's probability (its 90 %
// enclosure, the textbook boundary surface).
//
// enclosure_hist: one dispatch per orbital — a BINS-bin histogram of log10(ψ̃²/peak) over
// [−DECADES, 0] (point counts, u32 atomics) into hist[term·BINS ..]; values below 10^−DECADES of
// the peak carry < 1e-8 of it per point and are left out.
// enclosure_level: one thread per orbital — the probability per bin is its count × the bin's centre
// value (bins 1/32 decade wide: ≤ 3.7 % off per point), summed from the top bin down; the lower
// edge of the bin where the sum reaches `share` × the total is the level (relative to the peak),
// written to levels[term]. 0 when the orbital is empty (the mask then keeps everything).

const BINS: u32 = 256u;
const DECADES: f32 = 8.0;
const LOG2_10: f32 = 3.321928;

struct HistParams { n: u32, p: u32, term: u32, inv_peak: f32 }

@group(0) @binding(0) var<uniform> H: HistParams;
@group(0) @binding(1) var<storage, read> psi: array<f32>;
@group(0) @binding(2) var<storage, read_write> hist: array<atomic<u32>>;

@compute @workgroup_size(256)
fn enclosure_hist(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let n = H.n;
    let total = n * n * n;
    for (var idx = gid.x; idx < total; idx = idx + nw.x * 256u) {
        let i = idx % n;
        let t = idx / n;
        let j = t % n;
        let k = t / n;
        let q = (i + 4u) + H.p * ((j + 4u) + H.p * (k + 4u));
        let s = psi[q];
        let v = s * s * H.inv_peak;
        if (v > 1e-8) {
            let x = (log2(v) / LOG2_10 + DECADES) / DECADES * f32(BINS);
            let b = u32(clamp(x, 0.0, f32(BINS - 1u)));
            atomicAdd(&hist[H.term * BINS + b], 1u);
        }
    }
}

struct LevelParams { count: u32, share: f32, pad0: u32, pad1: u32 }

@group(0) @binding(4) var<uniform> L: LevelParams;
@group(0) @binding(5) var<storage, read> counts: array<u32>;
@group(0) @binding(6) var<storage, read_write> levels: array<f32>;

// ψ̃²/peak at the bin's fraction f of its width (0 = lower edge, 0.5 = centre)
fn bin_value(b: u32, f: f32) -> f32 {
    return exp2(((f32(b) + f) / f32(BINS) * DECADES - DECADES) * LOG2_10);
}

@compute @workgroup_size(64)
fn enclosure_level(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= L.count) { return; }
    var total = 0.0;
    for (var b = 0u; b < BINS; b = b + 1u) {
        total = total + f32(counts[t * BINS + b]) * bin_value(b, 0.5);
    }
    var level = 0.0;
    if (total > 0.0) {
        var cum = 0.0;
        for (var b = i32(BINS) - 1; b >= 0; b = b - 1) {
            cum = cum + f32(counts[t * BINS + u32(b)]) * bin_value(u32(b), 0.5);
            if (cum >= L.share * total) {
                level = bin_value(u32(b), 0.0);
                break;
            }
        }
    }
    levels[t] = level;
}
