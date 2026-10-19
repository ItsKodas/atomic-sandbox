// The Camera view (user request 2026-10-19): each atom between a soft electron glow (solidity 0:
// no surface, the column density of a smooth cloud) and a solid shaded ball (solidity 1, like a
// space-filling model), drawn at every instant of the shutter with weight 1/N and summed (a
// time-average: motion blur). With any solidity, each instant first lays the balls' depth
// (`fs_depth`), so a ball hides what is behind it at that instant. Layout of `Camera` must match
// `CameraUniform`.
// Smooth union (look.w = k > 0, user request 2026-10-19): the solid is not N balls but one surface,
// the polynomial smooth minimum of the balls' distances (k = the neck's width, Bohr), sphere-traced
// over this instant's atoms (`atoms`, the instance buffer read as storage); bonded atoms flow into
// each other through a soft neck instead of a crease. Each atom's quad traces the same surface
// near itself; the point is coloured once, by the atom whose own ball is nearest (the owner), with
// the colours blended across the neck.
struct Camera { view_proj: mat4x4<f32>, inv_view_proj: mat4x4<f32>, eye: vec3<f32>, _p0: f32, viewport: vec2<f32>, origin: vec2<f32> };
@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> look: vec4<f32>;   // x = exposure gain, y = solidity 0–1, z = atoms per instant, w = smoothness k (Bohr)

struct Atom { centre: vec3<f32>, radius: f32, colour: vec3<f32>, weight: f32 };
@group(0) @binding(2) var<storage, read> atoms: array<Atom>;

// `weight` = 1/N of the shutter's instants (the instance's `selected` slot)
struct Inst { @location(0) centre: vec3<f32>, @location(1) radius: f32, @location(2) colour: vec3<f32>, @location(3) weight: f32 };
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) centre: vec3<f32>, @location(1) radius: f32, @location(2) colour: vec3<f32>, @location(3) weight: f32, @location(4) @interpolate(flat) inst: u32 };
struct FOut { @location(0) colour: vec4<f32>, @builtin(frag_depth) depth: f32 };

// the glow reaches this many radii (the quad's half-size)
const REACH: f32 = 2.2;

@vertex
fn vs_glow(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32, inst: Inst) -> VOut {
    var corners = array<vec2<f32>, 6>(vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(1.0,1.0), vec2(-1.0,-1.0), vec2(1.0,1.0), vec2(-1.0,1.0));
    let c = corners[vi % 6u];
    let to_c = inst.centre - cam.eye;
    let fwd = normalize(to_c);
    let cam_up = normalize((cam.inv_view_proj * vec4(0.0, 1.0, 0.0, 0.0)).xyz);
    var right = cross(fwd, cam_up);
    if (dot(right, right) < 1e-8) { right = normalize((cam.inv_view_proj * vec4(1.0, 0.0, 0.0, 0.0)).xyz); }
    right = normalize(right);
    let up = cross(right, fwd);
    let world = inst.centre + (right * c.x + up * c.y) * inst.radius * REACH;
    var o: VOut;
    o.pos = cam.view_proj * vec4(world, 1.0);
    o.centre = inst.centre; o.radius = inst.radius; o.colour = inst.colour; o.weight = inst.weight; o.inst = ii;
    return o;
}

struct Ray { o: vec3<f32>, d: vec3<f32> };

fn ray_of(pos: vec4<f32>) -> Ray {
    let ndc = vec2((pos.x - cam.origin.x) / cam.viewport.x * 2.0 - 1.0, 1.0 - (pos.y - cam.origin.y) / cam.viewport.y * 2.0);
    let p_near = cam.inv_view_proj * vec4(ndc, 0.0, 1.0);
    let p_far = cam.inv_view_proj * vec4(ndc, 1.0, 1.0);
    let o = p_near.xyz / p_near.w;
    return Ray(o, normalize(p_far.xyz / p_far.w - o));
}

fn depth_at(p: vec3<f32>) -> f32 {
    let clip = cam.view_proj * vec4(p, 1.0);
    return clip.z / clip.w;
}

// where the ray meets the ball (w = 0: it misses) — one formula for both passes, so the colour
// pass meets exactly the depth the depth pass laid
fn ball_hit(r: Ray, centre: vec3<f32>, radius: f32) -> vec4<f32> {
    let oc = r.o - centre;
    let b = dot(oc, r.d);
    let disc = b * b - (dot(oc, oc) - radius * radius);
    if (disc < 0.0) { return vec4(0.0); }
    return vec4(r.o + r.d * (-b - sqrt(disc)), 1.0);
}

// ---- the smooth union ----

fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}

// this instant's atoms: [first, first + count)
fn instant_of(inst: u32) -> vec2<u32> {
    let per = max(u32(look.z), 1u);
    return vec2((inst / per) * per, per);
}

fn sdf(p: vec3<f32>, span: vec2<u32>) -> f32 {
    var d = 1e9;
    for (var i = span.x; i < span.x + span.y; i = i + 1u) {
        let a = atoms[i];
        d = smin(d, length(p - a.centre) - a.radius, look.w);
    }
    return d;
}

// the first point of the union surface along the ray near this atom (w = 0: none here) — traced
// from where the ray enters this atom's reach; already inside the union there means the surface
// is nearer, in another atom's quad
fn union_hit(r: Ray, centre: vec3<f32>, radius: f32, span: vec2<u32>) -> vec4<f32> {
    let along = dot(centre - r.o, r.d);
    let reach = radius * REACH + look.w;
    var t = max(along - reach, 0.0);
    let t_end = along + reach;
    if (sdf(r.o + r.d * t, span) < 0.0) { return vec4(0.0); }
    for (var s = 0; s < 64; s = s + 1) {
        let p = r.o + r.d * t;
        let d = sdf(p, span);
        if (d < 1e-3) { return vec4(p, 1.0); }
        t = t + max(d, 1e-3);
        if (t > t_end) { break; }
    }
    return vec4(0.0);
}

fn sdf_normal(p: vec3<f32>, span: vec2<u32>) -> vec3<f32> {
    let e = 2e-3;
    let k1 = vec3(1.0, -1.0, -1.0); let k2 = vec3(-1.0, -1.0, 1.0); let k3 = vec3(-1.0, 1.0, -1.0); let k4 = vec3(1.0, 1.0, 1.0);
    return normalize(k1 * sdf(p + k1 * e, span) + k2 * sdf(p + k2 * e, span) + k3 * sdf(p + k3 * e, span) + k4 * sdf(p + k4 * e, span));
}

// the surface colour at p: the atoms' colours weighted by how close their own balls are (a soft
// argmin, sharp away from a neck), and whether `inst` is the owner (its ball the nearest)
fn union_colour(p: vec3<f32>, span: vec2<u32>, inst: u32) -> vec4<f32> {
    var best = 1e9;
    var owner = span.x;
    var acc = vec3(0.0);
    var wsum = 0.0;
    let k = max(look.w, 1e-3);
    for (var i = span.x; i < span.x + span.y; i = i + 1u) {
        let a = atoms[i];
        let d = length(p - a.centre) - a.radius;
        if (d < best) { best = d; owner = i; }
        let w = exp(-max(d, 0.0) * 4.0 / k);
        acc = acc + a.colour * w;
        wsum = wsum + w;
    }
    return vec4(acc / max(wsum, 1e-6), select(0.0, 1.0, owner == inst));
}

// the solid's surface for this fragment: (point, hit?) — the ball, or the smooth union
fn solid_hit(r: Ray, in: VOut) -> vec4<f32> {
    if (look.w > 0.0) { return union_hit(r, in.centre, in.radius, instant_of(in.inst)); }
    let oc = in.centre - r.o;
    let perp = length(oc - r.d * dot(oc, r.d)) / in.radius;
    if (perp > 1.0) { return vec4(0.0); }
    return ball_hit(r, in.centre, in.radius);
}

// the solid's depth only (an instant's first pass, with any solidity)
@fragment
fn fs_depth(in: VOut) -> @builtin(frag_depth) f32 {
    let r = ray_of(in.pos);
    let hit = solid_hit(r, in);
    if (hit.w == 0.0) { discard; }
    return depth_at(hit.xyz);
}

@fragment
fn fs_glow(in: VOut) -> FOut {
    let r = ray_of(in.pos);
    let oc = in.centre - r.o;
    let along = dot(oc, r.d);
    let perp = length(oc - r.d * along) / in.radius;
    if (perp > REACH) { discard; }
    let h = look.y;
    // the glow: the column density of a smooth cloud through that line, bright core, long soft fade
    let core = exp(-perp * perp * 3.0);
    let halo = exp(-perp * 1.6) * 0.35;
    let gf = (core + halo) * (1.0 - smoothstep(REACH * 0.8, REACH, perp));
    let glow_col = mix(in.colour, vec3(1.0), core * 0.35) * gf;
    let glow_a = clamp(gf, 0.0, 1.0);
    // the ball: lit from over the viewer's shoulder, a soft specular, an anti-aliased rim
    var ball_col = vec3(0.0);
    var ball_a = 0.0;
    var depth = depth_at(in.centre);
    let bh = solid_hit(r, in);
    var colour = in.colour;
    var owns = 1.0;
    if (bh.w > 0.0 && look.w > 0.0) {
        let c = union_colour(bh.xyz, instant_of(in.inst), in.inst);
        colour = c.rgb;
        owns = c.w;
    }
    if (bh.w > 0.0 && owns > 0.0) {
        let hit = bh.xyz;
        var n = normalize(hit - in.centre);
        if (look.w > 0.0) { n = sdf_normal(hit, instant_of(in.inst)); }
        let v = -r.d;
        let cam_up = normalize((cam.inv_view_proj * vec4(0.0, 1.0, 0.0, 0.0)).xyz);
        let cam_right = normalize(cross(r.d, cam_up));
        let l = normalize(v + cam_up * 0.6 - cam_right * 0.45);
        let diffuse = max(dot(n, l), 0.0);
        let spec = pow(max(dot(n, normalize(l + v)), 0.0), 40.0);
        let rim = pow(1.0 - max(dot(n, v), 0.0), 2.0);
        ball_col = colour * (0.22 + 0.78 * diffuse) + vec3(0.35) * spec + colour * rim * 0.15;
        // a crisp edge: within an instant the atoms are added, so a soft rim double-counts where
        // balls overlap (the smooth-union surface, one surface per pixel, is the real fix)
        ball_a = 1.0;
        ball_col = ball_col * ball_a;
        // a hair in front of the depth pass's value: equal surfaces always pass
        if (h > 0.0) { depth = max(depth_at(hit) - 1e-6, 0.0); }
    }
    let col = mix(glow_col, ball_col, h);
    let a = mix(glow_a, ball_a, h);
    var out: FOut;
    out.colour = vec4(col * in.weight, a * in.weight);
    out.depth = depth;
    return out;
}
