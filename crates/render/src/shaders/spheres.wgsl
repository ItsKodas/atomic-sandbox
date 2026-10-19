// Instanced impostor spheres. Layout of `Camera` must match `CameraUniform` in spheres.rs (and the volume shader).
struct Camera { view_proj: mat4x4<f32>, inv_view_proj: mat4x4<f32>, eye: vec3<f32>, _p0: f32, viewport: vec2<f32>, origin: vec2<f32> };
@group(0) @binding(0) var<uniform> cam: Camera;

struct Inst { @location(0) centre: vec3<f32>, @location(1) radius: f32, @location(2) colour: vec3<f32>, @location(3) selected: f32 };
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) centre: vec3<f32>, @location(1) radius: f32, @location(2) colour: vec3<f32>, @location(3) selected: f32 };

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, inst: Inst) -> VOut {
    var corners = array<vec2<f32>, 6>(vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(1.0,1.0), vec2(-1.0,-1.0), vec2(1.0,1.0), vec2(-1.0,1.0));
    let c = corners[vi % 6u];
    // Camera right/up in world space: inv_view_proj maps clip-space directions (w = 0) to world directions.
    let cam_right = normalize((cam.inv_view_proj * vec4(1.0, 0.0, 0.0, 0.0)).xyz);
    let cam_up = normalize((cam.inv_view_proj * vec4(0.0, 1.0, 0.0, 0.0)).xyz);
    // Quad perpendicular to the eye->centre line, through the centre. The silhouette cone (half-angle a,
    // sin a = r/L) cuts that plane in a circle of radius r*L/sqrt(L^2 - r^2), so a square of that half-side
    // contains the exact perspective silhouette wherever the sphere is on screen (a screen-parallel quad of
    // fixed margin clips off-axis spheres, whose silhouettes stretch radially by ~1/cos(off-axis angle)).
    let to_c = inst.centre - cam.eye;
    let dist = length(to_c);
    let fwd = to_c / max(dist, 1e-20);   // centre at the eye: fwd = 0, half = 0 below, vertices stay finite
    var right = cross(fwd, cam_up);
    if (dot(right, right) < 1e-8) { right = cam_right; }   // centre straight above/below the eye: off screen anyway
    right = normalize(right);
    let up = cross(right, fwd);
    let s2 = dist * dist - inst.radius * inst.radius;
    // eye inside the sphere: collapse the quad (the fragment ray test would reject every pixel anyway)
    var half = 0.0;
    if (s2 > 0.0) { half = inst.radius * dist * inverseSqrt(s2) * 1.01; }   // 1% guard for f32 rounding
    let world = inst.centre + (right * c.x + up * c.y) * half;
    var o: VOut;
    o.pos = cam.view_proj * vec4(world, 1.0);
    o.centre = inst.centre; o.radius = inst.radius; o.colour = inst.colour; o.selected = inst.selected;
    return o;
}

struct FOut { @location(0) colour: vec4<f32>, @builtin(frag_depth) depth: f32 };

@fragment
fn fs_main(in: VOut) -> FOut {
    let ndc = vec2((in.pos.x - cam.origin.x) / cam.viewport.x * 2.0 - 1.0, 1.0 - (in.pos.y - cam.origin.y) / cam.viewport.y * 2.0);
    let p_near = cam.inv_view_proj * vec4(ndc, 0.0, 1.0);
    let p_far = cam.inv_view_proj * vec4(ndc, 1.0, 1.0);
    let o = p_near.xyz / p_near.w;
    let d = normalize(p_far.xyz / p_far.w - o);
    let oc = o - in.centre;
    let b = dot(oc, d);
    let cc = dot(oc, oc) - in.radius * in.radius;
    let disc = b * b - cc;
    if (disc < 0.0) { discard; }
    let t = -b - sqrt(disc);
    if (t < 0.0) { discard; }
    let hit = o + d * t;
    let n = normalize(hit - in.centre);
    let l = -d;
    let diffuse = max(dot(n, l), 0.0);
    let rim = pow(1.0 - max(dot(n, l), 0.0), 3.0);
    var col = in.colour * (0.15 + 0.85 * diffuse) + vec3(0.25) * rim;
    if (in.selected > 0.5) { col = mix(col, vec3(1.0, 0.9, 0.3), rim * 0.9 + 0.1); }
    let clip = cam.view_proj * vec4(hit, 1.0);
    var out: FOut;
    // a negative `selected` is a motion-blur ghost's alpha (the blended pass): no rim, and faded
    // toward the outline (disc / r² is 1 through the centre, 0 at the edge) so stacked copies smear
    var a = 1.0;
    if (in.selected < 0.0) {
        let e = clamp(disc / (in.radius * in.radius), 0.0, 1.0);
        a = -in.selected * e * e;
        col = in.colour * (0.35 + 0.65 * diffuse);
    }
    out.colour = vec4(col, a);
    out.depth = clip.z / clip.w;
    return out;
}
