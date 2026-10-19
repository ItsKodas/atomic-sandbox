// The Camera view (user request 2026-10-19): each atom as a soft electron glow — no surface, the
// projected density of a smooth cloud fading from its centre — drawn at every instant of the
// shutter with weight 1/N and added up (a time-average: motion blur), then composited over the
// scene with an exposure and a filmic curve. Layout of `Camera` must match `CameraUniform`.
struct Camera { view_proj: mat4x4<f32>, inv_view_proj: mat4x4<f32>, eye: vec3<f32>, _p0: f32, viewport: vec2<f32>, origin: vec2<f32> };
@group(0) @binding(0) var<uniform> cam: Camera;

// `weight` = 1/N of the shutter's instants (the instance's `selected` slot)
struct Inst { @location(0) centre: vec3<f32>, @location(1) radius: f32, @location(2) colour: vec3<f32>, @location(3) weight: f32 };
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) centre: vec3<f32>, @location(1) radius: f32, @location(2) colour: vec3<f32>, @location(3) weight: f32 };

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

@fragment
fn fs_glow(in: VOut) -> @location(0) vec4<f32> {
    let ndc = vec2((in.pos.x - cam.origin.x) / cam.viewport.x * 2.0 - 1.0, 1.0 - (in.pos.y - cam.origin.y) / cam.viewport.y * 2.0);
    let p_near = cam.inv_view_proj * vec4(ndc, 0.0, 1.0);
    let p_far = cam.inv_view_proj * vec4(ndc, 1.0, 1.0);
    let o = p_near.xyz / p_near.w;
    let d = normalize(p_far.xyz / p_far.w - o);
    // the ray's closest approach to the centre, in radii
    let oc = in.centre - o;
    let along = dot(oc, d);
    let perp = length(oc - d * along) / in.radius;
    if (perp > REACH) { discard; }
    // the column density of a smooth cloud through that line: bright core, long soft fade
    let core = exp(-perp * perp * 3.0);
    let halo = exp(-perp * 1.6) * 0.35;
    let f = (core + halo) * (1.0 - smoothstep(REACH * 0.8, REACH, perp));
    let a = clamp(f, 0.0, 1.0);
    // a hint of white in the brightest middle: the glow is light, not paint
    let col = mix(in.colour, vec3(1.0), core * 0.35) * f;
    return vec4(col * in.weight, a * in.weight);
}

