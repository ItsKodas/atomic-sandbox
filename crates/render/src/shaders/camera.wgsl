// The Camera view (user request 2026-10-19): each atom between a soft electron glow (solidity 0:
// no surface, the column density of a smooth cloud) and a solid shaded ball (solidity 1, like a
// space-filling model), drawn at every instant of the shutter with weight 1/N and summed (a
// time-average: motion blur). With any solidity, each instant first lays the balls' depth
// (`fs_depth`), so a ball hides what is behind it at that instant. Layout of `Camera` must match
// `CameraUniform`.
struct Camera { view_proj: mat4x4<f32>, inv_view_proj: mat4x4<f32>, eye: vec3<f32>, _p0: f32, viewport: vec2<f32>, origin: vec2<f32> };
@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> look: vec4<f32>;   // x = exposure gain, y = solidity 0–1

// `weight` = 1/N of the shutter's instants (the instance's `selected` slot)
struct Inst { @location(0) centre: vec3<f32>, @location(1) radius: f32, @location(2) colour: vec3<f32>, @location(3) weight: f32 };
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) centre: vec3<f32>, @location(1) radius: f32, @location(2) colour: vec3<f32>, @location(3) weight: f32 };
struct FOut { @location(0) colour: vec4<f32>, @builtin(frag_depth) depth: f32 };

// the glow reaches this many radii (the quad's half-size)
const REACH: f32 = 2.2;

@vertex
fn vs_glow(@builtin(vertex_index) vi: u32, inst: Inst) -> VOut {
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
    o.centre = inst.centre; o.radius = inst.radius; o.colour = inst.colour; o.weight = inst.weight;
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

// the ball's depth only (an instant's first pass, with any solidity)
@fragment
fn fs_depth(in: VOut) -> @builtin(frag_depth) f32 {
    let r = ray_of(in.pos);
    let oc = in.centre - r.o;
    let perp = length(oc - r.d * dot(oc, r.d)) / in.radius;
    if (perp > 1.0) { discard; }
    let hit = ball_hit(r, in.centre, in.radius);
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
    let bh = ball_hit(r, in.centre, in.radius);
    if (perp < 1.0 && bh.w > 0.0) {
        let hit = bh.xyz;
        let n = normalize(hit - in.centre);
        let v = -r.d;
        let cam_up = normalize((cam.inv_view_proj * vec4(0.0, 1.0, 0.0, 0.0)).xyz);
        let cam_right = normalize(cross(r.d, cam_up));
        let l = normalize(v + cam_up * 0.6 - cam_right * 0.45);
        let diffuse = max(dot(n, l), 0.0);
        let spec = pow(max(dot(n, normalize(l + v)), 0.0), 40.0);
        let rim = pow(1.0 - max(dot(n, v), 0.0), 2.0);
        ball_col = in.colour * (0.22 + 0.78 * diffuse) + vec3(0.35) * spec + in.colour * rim * 0.15;
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
