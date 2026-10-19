// "Probability dots": samples of the displayed field
// (ρ, an orbital's |ψ|², or the colour volume's a), drawn by dots_draw.wgsl.
//
// The sampled density is the field's TRILINEAR interpolation (what the volume raymarch draws),
// so it is continuous across cells: no piecewise-constant voxels, whose steps showed as cube
// edges around a sharp peak and as lattice planes when viewed along an axis. A cell is the cube
// between grid points (x..x+1, y..y+1, z..z+1); the trilinear field integrates over it to the mean
// of its 8 corners, which is the cell's weight. Three passes:
//   row_sums  : one thread per (y, z) row of cells, Σ_x weight, and the row's largest corner value;
//   row_scan  : one workgroup turns the row sums into an inclusive prefix sum (the row CDF) in
//               place and reduces the global maximum (for the colour ramp);
//   gen_points: one thread per dot — a uniform u in [0, total) picks the row (binary search over
//               the CDF) and the cell in it (a walk along x); then an exact draw from the cell's
//               trilinear density: trilinear = Σ_corners v_c Π_axes (1 − t or t), a mixture of
//               product densities each integrating to 1/8, so pick corner c with probability
//               v_c / Σ v and each axis independently from the linear density 2(1 − t) (t = 1 − √ξ)
//               or 2t (t = √ξ). No rejection, no bias.
// Every dot draws its cell independently (a multinomial over cells, so per-cell counts are
// stochastic, never rounded), its random numbers from a PCG hash chain of (dot index, frame seed).
// Nothing of the field leaves the GPU. `Params` must match `DotsUniform` in dots.rs.
struct Params {
    box_min: vec3<f32>, use_colour: u32,
    box_max: vec3<f32>, count: u32,
    dims: vec3<u32>, seed: u32,
    tint: vec4<f32>,
    tint_neg: vec4<f32>,
    size_px: f32, intensity: f32, colour_by: u32, signed_field: u32,
    // the floating dots: Metropolis step (grid cells; 0 = a fresh sample every time), the share
    // re-drawn fresh each frame, glow (additive soft sprites) on/off
    walk: f32, refresh: f32, glow: u32, _pad: u32,
};
@group(0) @binding(0) var<uniform> prm: Params;
@group(0) @binding(1) var scal: texture_3d<f32>;
@group(0) @binding(2) var col: texture_3d<f32>;
// [0, n_rows): row sums → row CDF; [n_rows, 2 n_rows): row maxima; [2 n_rows]: global maximum
@group(0) @binding(3) var<storage, read_write> rows: array<f32>;
@group(0) @binding(4) var<storage, read_write> pts: array<vec4<f32>>;

fn field(c: vec3<i32>) -> f32 {
    if (prm.use_colour != 0u) { return max(textureLoad(col, c, 0).a, 0.0); }
    let v = textureLoad(scal, c, 0).r;
    if (prm.signed_field != 0u) { return abs(v); }
    return max(v, 0.0);
}

fn cells() -> vec3<u32> { return max(prm.dims, vec3(2u)) - vec3(1u); }

// the 8 corner values of cell c, index bit 0 = x, bit 1 = y, bit 2 = z
fn corners(c: vec3<i32>) -> array<f32, 8> {
    var v: array<f32, 8>;
    for (var k = 0; k < 8; k = k + 1) {
        v[k] = field(c + vec3(k & 1, (k >> 1) & 1, (k >> 2) & 1));
    }
    return v;
}

fn cell_weight(c: vec3<i32>) -> f32 {
    let v = corners(c);
    return (v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7]) * 0.125;
}

@compute @workgroup_size(64)
fn row_sums(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nc = cells();
    let n_rows = nc.y * nc.z;
    let r = gid.x;
    if (r >= n_rows) { return; }
    let y = i32(r % nc.y);
    let z = i32(r / nc.y);
    var s = 0.0;
    var m = 0.0;
    for (var x = 0; x < i32(nc.x); x = x + 1) {
        let v = corners(vec3(x, y, z));
        s = s + (v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7]) * 0.125;
        m = max(m, max(max(max(v[0], v[1]), max(v[2], v[3])), max(max(v[4], v[5]), max(v[6], v[7]))));
    }
    rows[r] = s;
    rows[n_rows + r] = m;
}

var<workgroup> part: array<f32, 256>;
var<workgroup> pmax: array<f32, 256>;

@compute @workgroup_size(256)
fn row_scan(@builtin(local_invocation_id) lid: vec3<u32>) {
    let nc = cells();
    let n_rows = nc.y * nc.z;
    let chunk = (n_rows + 255u) / 256u;
    let t = lid.x;
    let lo = min(t * chunk, n_rows);
    let hi = min(lo + chunk, n_rows);
    var s = 0.0;
    var m = 0.0;
    for (var i = lo; i < hi; i = i + 1u) { s = s + rows[i]; m = max(m, rows[n_rows + i]); }
    part[t] = s;
    pmax[t] = m;
    workgroupBarrier();
    // Hillis-Steele inclusive scan of the 256 chunk sums (and a max scan alongside)
    for (var off = 1u; off < 256u; off = off * 2u) {
        var v = part[t];
        var w = pmax[t];
        if (t >= off) { v = v + part[t - off]; w = max(w, pmax[t - off]); }
        workgroupBarrier();
        part[t] = v;
        pmax[t] = w;
        workgroupBarrier();
    }
    if (t == 255u) { rows[2u * n_rows] = pmax[255]; }
    var run = 0.0;
    if (t > 0u) { run = part[t - 1u]; }
    for (var i = lo; i < hi; i = i + 1u) { run = run + rows[i]; rows[i] = run; }
}

// PCG hash (Jarzynski & Olano 2020)
fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
fn unit(h: u32) -> f32 { return f32(h >> 8u) * (1.0 / 16777216.0); }

// deep blue -> magenta -> orange -> white: the volume raymarch's `ramp` (volume.wgsl)
fn ramp(s: f32) -> vec3<f32> {
    let c0 = vec3(0.05, 0.02, 0.25); let c1 = vec3(0.6, 0.1, 0.6); let c2 = vec3(1.0, 0.55, 0.15); let c3 = vec3(1.0, 1.0, 0.9);
    if (s < 0.33) { return mix(c0, c1, s / 0.33); }
    if (s < 0.66) { return mix(c1, c2, (s - 0.33) / 0.33); }
    return mix(c2, c3, (s - 0.66) / 0.34);
}

// a draw from the density 2(1 − t) (bit 0) or 2t (bit 1) on [0, 1)
fn linear_draw(bit: i32, xi: f32) -> f32 {
    if (bit == 1) { return sqrt(xi); }
    return 1.0 - sqrt(1.0 - xi);
}

// the trilinear field at a world position (0 outside the box)
fn field_at(p: vec3<f32>) -> f32 {
    let dims = vec3<f32>(prm.dims);
    let g = (p - prm.box_min) / (prm.box_max - prm.box_min) * (dims - vec3(1.0));
    if (any(g < vec3(0.0)) || any(g > dims - vec3(1.0))) { return 0.0; }
    let cell = min(vec3<i32>(floor(g)), vec3<i32>(cells()) - vec3(1));
    let f = g - vec3<f32>(cell);
    let v = corners(cell);
    let x0 = mix(v[0], v[1], f.x); let x1 = mix(v[2], v[3], f.x);
    let x2 = mix(v[4], v[5], f.x); let x3 = mix(v[6], v[7], f.x);
    return mix(mix(x0, x1, f.y), mix(x2, x3, f.y), f.z);
}

// a dot's colour at a world position (as the fresh draw colours it)
fn colour_at(p: vec3<f32>) -> vec3<f32> {
    let dims = vec3<f32>(prm.dims);
    let g = clamp((p - prm.box_min) / (prm.box_max - prm.box_min) * (dims - vec3(1.0)), vec3(0.0), dims - vec3(1.0));
    let near = vec3<i32>(round(g));
    if (prm.colour_by == 1u && prm.use_colour != 0u) {
        let q = textureLoad(col, near, 0);
        return clamp(q.rgb / max(q.a, 1e-6), vec3(0.0), vec3(1.0));
    } else if (prm.colour_by == 2u) {
        return prm.tint.rgb;
    } else if (prm.colour_by == 3u && prm.use_colour == 0u) {
        let q = textureLoad(scal, near, 0).r;
        return select(prm.tint_neg.rgb, prm.tint.rgb, q >= 0.0);
    }
    let nc = cells();
    let gmax = rows[2u * nc.y * nc.z];
    let s = sqrt(clamp(field_at(p) / max(gmax, 1e-30), 0.0, 1.0));
    return ramp(0.1 + 0.9 * s);
}

// one dot per thread: with `walk` > 0 a live dot takes one Metropolis step on the field (a
// Gaussian proposal of `walk` cells, accepted with min(1, ρ_new / ρ_old)) — the cloud stays a
// sample of the field while each dot drifts smoothly ("floating"); a `refresh` share, and every
// dead or new slot, is drawn fresh (`gen_points` below), so no chain stays stuck
@compute @workgroup_size(64)
fn gen_points(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= prm.count) { return; }
    let nc = cells();
    if (prm.walk > 0.0) {
        let old = pts[i];
        let alive = unpack4x8unorm(bitcast<u32>(old.w)).a > 0.0;
        var hw = pcg(i ^ pcg(prm.seed ^ 0x9e3779b9u));
        // a dot where the field is zero (a new scene, a moved box) is drawn fresh instead
        let rp = select(0.0, field_at(old.xyz), alive);
        if (alive && rp > 0.0 && unit(hw) >= prm.refresh) {
            let cell_size = (prm.box_max - prm.box_min) / (vec3<f32>(prm.dims) - vec3(1.0));
            hw = pcg(hw); let u1 = max(unit(hw), 1e-7);
            hw = pcg(hw); let u2 = unit(hw);
            hw = pcg(hw); let u3 = max(unit(hw), 1e-7);
            hw = pcg(hw); let u4 = unit(hw);
            let r1 = sqrt(-2.0 * log(u1)); let r2 = sqrt(-2.0 * log(u3));
            let gauss = vec3(r1 * cos(6.2831853 * u2), r1 * sin(6.2831853 * u2), r2 * cos(6.2831853 * u4));
            let q = old.xyz + gauss * prm.walk * cell_size;
            let rq = field_at(q);
            hw = pcg(hw);
            var p = old.xyz;
            if (unit(hw) * rp < rq) { p = q; }
            pts[i] = vec4(p, bitcast<f32>(pack4x8unorm(vec4(colour_at(p), 1.0))));
            return;
        }
    }
    let n_rows = nc.y * nc.z;
    let total = rows[n_rows - 1u];
    if (!(total > 0.0)) { pts[i] = vec4(0.0, 0.0, 0.0, 0.0); return; }
    var h = pcg(i ^ pcg(prm.seed));
    let u = unit(h) * total;
    // first row whose inclusive prefix exceeds u
    var lo = 0u; var hi = n_rows - 1u;
    loop {
        if (lo >= hi) { break; }
        let mid = (lo + hi) / 2u;
        if (rows[mid] > u) { hi = mid; } else { lo = mid + 1u; }
    }
    let r = lo;
    var tgt = u;
    if (r > 0u) { tgt = u - rows[r - 1u]; }
    let y = i32(r % nc.y);
    let z = i32(r / nc.y);
    // the cell along the row (skipping empty cells, so a dot never lands in one)
    var x = 0;
    var acc = 0.0;
    var last_nonzero = 0;
    loop {
        let w = cell_weight(vec3(x, y, z));
        if (w > 0.0) { last_nonzero = x; }
        acc = acc + w;
        if (acc > tgt) { break; }
        if (x >= i32(nc.x) - 1) { x = last_nonzero; break; }
        x = x + 1;
    }
    let cell = vec3(x, y, z);
    let v = corners(cell);
    // the corner, with probability v_c / Σ v
    h = pcg(h);
    var pick = unit(h) * (v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7]);
    var k = 7;
    for (var c = 0; c < 8; c = c + 1) {
        pick = pick - v[c];
        if (pick < 0.0) { k = c; break; }
    }
    h = pcg(h); let tx = linear_draw(k & 1, unit(h));
    h = pcg(h); let ty = linear_draw((k >> 1) & 1, unit(h));
    h = pcg(h); let tz = linear_draw((k >> 2) & 1, unit(h));
    let f = vec3(tx, ty, tz);
    let dims = vec3<f32>(prm.dims);
    let g = vec3<f32>(cell) + f;
    let p = prm.box_min + g / (dims - vec3(1.0)) * (prm.box_max - prm.box_min);
    // the field at the dot (trilinear), for the colour
    let x0 = mix(v[0], v[1], f.x); let x1 = mix(v[2], v[3], f.x);
    let x2 = mix(v[4], v[5], f.x); let x3 = mix(v[6], v[7], f.x);
    let val = mix(mix(x0, x1, f.y), mix(x2, x3, f.y), f.z);
    var c: vec3<f32>;
    if (prm.colour_by == 1u && prm.use_colour != 0u) {
        // the orbitals' mix colour at the dot's nearest grid point
        let q = textureLoad(col, cell + vec3<i32>(round(f)), 0);
        c = clamp(q.rgb / max(q.a, 1e-6), vec3(0.0), vec3(1.0));
    } else if (prm.colour_by == 2u) {
        c = prm.tint.rgb;
    } else if (prm.colour_by == 3u && prm.use_colour == 0u) {
        // the sign of the field at the dot's nearest grid point
        let q = textureLoad(scal, cell + vec3<i32>(round(f)), 0).r;
        c = select(prm.tint_neg.rgb, prm.tint.rgb, q >= 0.0);
    } else {
        // density ramp: √ of the field over its maximum (the volume mode's default gamma 0.5),
        // faintest = deep purple, then magenta/pink, orange/yellow, white at the densest core
        let gmax = rows[2u * n_rows];
        let s = sqrt(clamp(val / max(gmax, 1e-30), 0.0, 1.0));
        c = ramp(0.1 + 0.9 * s);
    }
    pts[i] = vec4(p, bitcast<f32>(pack4x8unorm(vec4(c, 1.0))));
}
