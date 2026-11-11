// "Probability dots": each dot from dots_compute.wgsl as a small round sprite
// (premultiplied over), hidden behind the spheres by the scene depth. `Camera` matches spheres.rs, `Params`
// matches dots_compute.wgsl.
struct Camera { view_proj: mat4x4<f32>, inv_view_proj: mat4x4<f32>, eye: vec3<f32>, _p0: f32, viewport: vec2<f32>, origin: vec2<f32> };
struct Params {
    box_min: vec3<f32>, use_colour: u32,
    box_max: vec3<f32>, count: u32,
    dims: vec3<u32>, seed: u32,
    tint: vec4<f32>,
    tint_neg: vec4<f32>,
    size_px: f32, intensity: f32, colour_by: u32, signed_field: u32,
    // the floating dots: Metropolis step (grid cells; 0 = a fresh sample every time), the share
    // re-drawn fresh each frame, glow (additive soft sprites) on/off
    walk: f32, refresh: f32, glow: u32, cut: f32,
    // the blur: 0 crisp … 1 soft, 0.5 = the look before the slider
    blur: f32, _b0: f32, _b1: f32, _b2: f32,
};
@group(0) @binding(0) var<uniform> cam: Camera;
@group(1) @binding(0) var<uniform> prm: Params;
@group(1) @binding(1) var<storage, read> pts: array<vec4<f32>>;
@group(1) @binding(2) var scene_depth: texture_depth_2d;

struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) colour: vec4<f32> };

@vertex fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
    var corners = array<vec2<f32>, 6>(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
    let q = corners[vi];
    let raw = pts[ii];
    // the display warp (warp.wgsl): a dot of the solved field, drawn where its atoms are now
    let pt = vec4(warp_forward(raw.xyz), raw.w);
    let c = unpack4x8unorm(bitcast<u32>(pt.w));
    var o: VOut;
    var clip = cam.view_proj * vec4(pt.xyz, 1.0);
    // an empty slot (alpha 0) or a dot behind the eye: off screen
    if (c.a <= 0.0 || clip.w <= 0.0) { o.pos = vec4(2.0, 2.0, 2.0, 1.0); o.uv = q; o.colour = vec4(0.0); return o; }
    var fade = 1.0;
    if (prm.glow != 0u) {
        // glowing dots have a size in the world (size_px × 0.012 Bohr): tiny far away, large up
        // close, like the reference — a billboard facing the camera
        let right = normalize((cam.inv_view_proj * vec4(1.0, 0.0, 0.0, 0.0)).xyz);
        let up = normalize((cam.inv_view_proj * vec4(0.0, 1.0, 0.0, 0.0)).xyz);
        let r = prm.size_px * 0.012;
        let wb = cam.view_proj * vec4(pt.xyz + (right * q.x + up * q.y) * r, 1.0);
        // its radius on screen (px); under 0.75 px a quad covers no pixel and the dot would vanish:
        // kept at 0.75 px and dimmed by the area it lacks, so tiny dots thin out instead
        let px = length((wb.xy / wb.w - clip.xy / clip.w) * cam.viewport * 0.5) / length(q);
        if (px < 0.75) {
            fade = (px / 0.75) * (px / 0.75);
            clip = vec4(clip.xy + q * 0.75 / cam.viewport * 2.0 * clip.w, clip.z, clip.w);
        } else {
            clip = wb;
        }
    } else {
        clip = vec4(clip.xy + q * prm.size_px / cam.viewport * 2.0 * clip.w, clip.z, clip.w);
    }
    o.pos = clip;
    o.uv = q;
    // the feather's opacity rides in the packed alpha
    o.colour = vec4(c.rgb * fade, c.a);
    return o;
}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let d2 = dot(in.uv, in.uv);
    if (d2 > 1.0) { discard; }
    let z = textureLoad(scene_depth, vec2<i32>(in.pos.xy), 0);
    if (in.pos.z > z) { discard; }
    // the blur scales the falloff by 4^(1 − 2·blur): 1 at 0.5 (the look before the slider), ×4
    // sharper at 0, ×4 softer at 1
    let s = pow(4.0, 1.0 - 2.0 * clamp(prm.blur, 0.0, 1.0));
    var w = (1.0 - pow(d2, 2.0 * s)) * prm.intensity;
    // glow: a soft Gaussian sprite, added (dense regions burn toward white)
    if (prm.glow != 0u) { w = exp(-4.0 * s * d2) * prm.intensity; }
    w = w * in.colour.a;
    return vec4(in.colour.rgb * w, w);
}
