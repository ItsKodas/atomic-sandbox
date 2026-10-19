// The orbital viewer's colour volume (spec §5.6; CPU reference qm::orbitals::colour_volume):
//   rgb = Σ_i f_i c_i ψ̃_i² / P_i,   a = Σ_i f_i ψ̃_i² / P_i
// over the included displayed orbitals, one thread per interior point (grid stride), written to
// an rgba16float storage texture with textureStore. The per-orbital weights
// (f_i c_i / P_i, f_i / P_i) are in the uniform. A dispatch binds at most 6 orbitals
// (6 + the accumulator = 7 storage buffers ≤ WebGPU's default 8); more orbitals are chained:
// the first chunk starts from 0, intermediate chunks keep the running sum in `acc` (one vec4 f32
// per interior point, the qm layout), the last chunk adds its terms and stores the texel. With
// one chunk `acc` is never touched. f32 sums, rounded once to f16 at the store.
//
// Boundary surfaces (flags bit 2; user request 2026-10-19): each orbital adds its weight
// w = (f c, f) times a mask instead of f ψ̃²/P — 1 inside its enclosure level (`levels`, written by
// enclosure.wgsl as ψ̃²/peak) and 0 outside, softened over the last 30 % of the level; with bit 3
// (shell) only a band just inside the boundary, ψ̃²/peak in [lo, SHELL·lo], is kept (the hollow
// textbook look). m[t] = (1/peak, the orbital's index in `levels`, 0, 0).

struct ColourParams {
    n: u32, p: u32,
    count: u32,       // orbitals bound in this chunk (≤ 6)
    flags: u32,       // bit 0: first chunk (start from 0, not acc); bit 1: last chunk (store the texel); bit 2: boundary; bit 3: shell
    w: array<vec4<f32>, 6>,   // (f c_r / P, f c_g / P, f c_b / P, f / P) per bound orbital; boundary: (f c, f)
    m: array<vec4<f32>, 6>,   // boundary: (1 / peak, index into `levels`, 0, 0)
}

const SHELL: f32 = 3.0;

fn mask(s: f32, t: u32) -> f32 {
    let lo = levels[u32(C.m[t].y)];
    if (lo <= 0.0) { return 1.0; }
    let r = s * s * C.m[t].x;
    var m = smoothstep(0.7 * lo, lo, r);
    if ((C.flags & 8u) != 0u) {
        let hi = SHELL * lo;
        m = m * (1.0 - smoothstep(hi, 1.4 * hi, r));
    }
    return m;
}

fn term(s: f32, t: u32) -> vec4<f32> {
    if ((C.flags & 4u) != 0u) { return C.w[t] * mask(s, t); }
    return C.w[t] * (s * s);
}

@group(0) @binding(0) var<uniform> C: ColourParams;
@group(0) @binding(1) var<storage, read> o0: array<f32>;
@group(0) @binding(2) var<storage, read> o1: array<f32>;
@group(0) @binding(3) var<storage, read> o2: array<f32>;
@group(0) @binding(4) var<storage, read> o3: array<f32>;
@group(0) @binding(5) var<storage, read> o4: array<f32>;
@group(0) @binding(6) var<storage, read> o5: array<f32>;
@group(0) @binding(7) var<storage, read_write> acc: array<vec4<f32>>;
@group(0) @binding(8) var out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(9) var<storage, read> levels: array<f32>;

@compute @workgroup_size(256)
fn colour_accumulate(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let n = C.n;
    let total = n * n * n;
    for (var idx = gid.x; idx < total; idx = idx + nw.x * 256u) {
        let i = idx % n;
        let t = idx / n;
        let j = t % n;
        let k = t / n;
        let q = (i + 4u) + C.p * ((j + 4u) + C.p * (k + 4u));
        var v = vec4<f32>(0.0);
        if ((C.flags & 1u) == 0u) { v = acc[idx]; }
        if (C.count > 0u) { v = v + term(o0[q], 0u); }
        if (C.count > 1u) { v = v + term(o1[q], 1u); }
        if (C.count > 2u) { v = v + term(o2[q], 2u); }
        if (C.count > 3u) { v = v + term(o3[q], 3u); }
        if (C.count > 4u) { v = v + term(o4[q], 4u); }
        if (C.count > 5u) { v = v + term(o5[q], 5u); }
        if ((C.flags & 2u) != 0u) {
            textureStore(out, vec3<i32>(i32(i), i32(j), i32(k)), v);
        } else {
            acc[idx] = v;
        }
    }
}
