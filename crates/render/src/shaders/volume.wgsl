// Fullscreen density raymarch (volume / isosurface, and the signed orbital / spin modes), composited over the
// sphere pass via its depth buffer.
// `Camera` must match `CameraUniform` in spheres.rs; `Params` must match `VolumeUniform` in volume.rs.
// This module is completed by one of volume_filtered.wgsl / volume_manual.wgsl (appended by
// `VolumePass`), which declares the density textures `vol` (current, group 1, binding 1) and
// `vol_prev` (previous, binding 4) and `sample_one(texture, p)`; `sample_rho` crossfades them.
// volume_colour.wgsl is appended after that: the colour-volume entry point `fs_colour`, which shares
// everything above (camera, params, vertex stage, ray span, depth test).
struct Camera { view_proj: mat4x4<f32>, inv_view_proj: mat4x4<f32>, eye: vec3<f32>, _p0: f32, viewport: vec2<f32>, origin: vec2<f32> };
struct Params {
    box_min: vec3<f32>, mode: u32,
    box_max: vec3<f32>, scale: f32,
    gamma: f32, iso: f32, step: f32, opacity: f32,
    fade: f32, mix_w: f32, knee: f32, additive: u32,
    colour_pos: vec3<f32>, pad0: f32,
    colour_neg: vec3<f32>, pad1: f32,
};
@group(0) @binding(0) var<uniform> cam: Camera;
@group(1) @binding(0) var<uniform> prm: Params;
@group(1) @binding(2) var scene_depth: texture_depth_2d;

struct VOut { @builtin(position) pos: vec4<f32> };
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var o: VOut; o.pos = vec4(p[vi], 0.0, 1.0); return o;
}

// deep blue -> magenta -> orange -> white
fn ramp(s: f32) -> vec3<f32> {
    let c0 = vec3(0.05, 0.02, 0.25); let c1 = vec3(0.6, 0.1, 0.6); let c2 = vec3(1.0, 0.55, 0.15); let c3 = vec3(1.0, 1.0, 0.9);
    if (s < 0.33) { return mix(c0, c1, s / 0.33); }
    if (s < 0.66) { return mix(c1, c2, (s - 0.33) / 0.33); }
    return mix(c2, c3, (s - 0.66) / 0.34);
}

// Grid point (i, j, k) sits at box_min + (i, j, k) * (box_max - box_min) / (dims - 1): box_min/box_max are the
// first/last sample positions. Both `sample_rho` variants interpolate in continuous grid-index space (texel
// centres) from this position, clamped to the box: g ∈ [0, dims − 1].
fn grid_coord(p: vec3<f32>, dims: vec3<f32>) -> vec3<f32> {
    return clamp((p - prm.box_min) / (prm.box_max - prm.box_min), vec3(0.0), vec3(1.0)) * (dims - vec3(1.0));
}

// The density at p: the previous and current textures crossfaded by mix_w (0 = previous, 1 = current;
// `VolumeUniform::mix`). Sampling is linear in the data, so this is the sample of the blended grid —
// which is also right for a signed field (an orbital or the spin density), because the sign rule keeps
// consecutive fields in phase.
fn sample_rho(p_now: vec3<f32>) -> f32 {
    // the display warp (warp.wgsl): the field as solved, at where this point was then
    let p = warp_back(p_now);
    if (prm.mix_w >= 1.0) { return sample_one(vol, p); }
    if (prm.mix_w <= 0.0) { return sample_one(vol_prev, p); }
    return mix(sample_one(vol_prev, p), sample_one(vol, p), prm.mix_w);
}

// A soft knee on the opacity — 0 below knee/8 of the maximum, smoothstep up to 1 at `knee` (0 = no
// knee) — so a single orbital's far tail fades out without a hard shell, while the colour keeps the
// full gamma-shaped gradient.
fn knee_w(r: f32) -> f32 {
    if (prm.knee <= 0.0) { return 1.0; }
    return smoothstep(prm.knee * 0.125, prm.knee, r);
}

// Scene depth at this pixel as a distance along the ray (o, d). The sphere pass stores clip.z / clip.w
// (0..1, DirectX/WebGPU convention), so unprojecting (ndc.xy, z) with the same pixel-centre NDC recovers the
// sphere hit point on this very ray.
fn depth_to_t(frag: vec2<f32>, o: vec3<f32>, d: vec3<f32>) -> f32 {
    let z = textureLoad(scene_depth, vec2<i32>(frag), 0);
    if (z >= 1.0) { return 1e30; }
    let ndc = vec2((frag.x - cam.origin.x) / cam.viewport.x * 2.0 - 1.0, 1.0 - (frag.y - cam.origin.y) / cam.viewport.y * 2.0);
    let pw = cam.inv_view_proj * vec4(ndc, z, 1.0);
    return dot(pw.xyz / pw.w - o, d);
}

// The pixel's ray (origin o, direction d) and the span [t0, t1] of it inside the box and in front of the
// spheres (empty when t1 <= t0).
struct Span { o: vec3<f32>, d: vec3<f32>, t0: f32, t1: f32 };
fn ray_span(frag: vec2<f32>) -> Span {
    // same ray reconstruction as spheres.wgsl, so depth_to_t lines up with the sphere hits
    let ndc = vec2((frag.x - cam.origin.x) / cam.viewport.x * 2.0 - 1.0, 1.0 - (frag.y - cam.origin.y) / cam.viewport.y * 2.0);
    let pn = cam.inv_view_proj * vec4(ndc, 0.0, 1.0);
    let pf = cam.inv_view_proj * vec4(ndc, 1.0, 1.0);
    let o = pn.xyz / pn.w;
    let d = normalize(pf.xyz / pf.w - o);
    // slab test
    let inv = 1.0 / d;
    let t0 = (prm.box_min - o) * inv; let t1 = (prm.box_max - o) * inv;
    let tmin = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let tmax = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    return Span(o, d, max(tmin, 0.0), min(tmax, depth_to_t(frag, o, d)));
}

// Mode 2: the signed isosurface. The first sample with |s| >= iso is the hit, refined on |s| as in
// mode 1; |s| = |f|/scale, so ∇|f| = sign(f) ∇f and the outward normal is -sign(f) ∇f (the surface of
// a positive lobe faces away from the lobe's peak, and so does a negative one's). Lit and coloured by
// the lobe's sign.
fn march_signed_iso(o: vec3<f32>, d: vec3<f32>, t_enter: f32, t_exit: f32) -> vec4<f32> {
    var t = t_enter;
    var prev = -1.0;   // no previous sample yet
    loop {
        if (t > t_exit) { break; }
        let s = abs(sample_rho(o + d * t)) / prm.scale;
        if (s >= prm.iso) {
            var tb = t;
            if (prev >= 0.0) { tb = t - prm.step * (s - prm.iso) / max(s - prev, 1e-6); }
            let p = o + d * tb;
            let e = prm.step;
            let sign_f = select(-1.0, 1.0, sample_rho(p) >= 0.0);
            let g = sign_f * vec3(sample_rho(p + vec3(e, 0.0, 0.0)) - sample_rho(p - vec3(e, 0.0, 0.0)),
                                  sample_rho(p + vec3(0.0, e, 0.0)) - sample_rho(p - vec3(0.0, e, 0.0)),
                                  sample_rho(p + vec3(0.0, 0.0, e)) - sample_rho(p - vec3(0.0, 0.0, e)));
            let n = -g * inverseSqrt(max(dot(g, g), 1e-30));
            let diffuse = max(dot(n, -d), 0.0);
            let base = select(prm.colour_neg, prm.colour_pos, sign_f > 0.0);
            return vec4(base * (0.2 + 0.8 * diffuse), 1.0) * prm.fade;
        }
        prev = s; t = t + prm.step;
    }
    return vec4(0.0);
}

// Mode 3: the signed volume — emission-absorption on |s| (today's transfer function, gamma and soft knee
// included) with the sign's colour. A field of zeros adds nothing (a = 0 at every sample).
fn march_signed_volume(o: vec3<f32>, d: vec3<f32>, t_enter: f32, t_exit: f32) -> vec4<f32> {
    var t = t_enter;
    var colour = vec3(0.0); var alpha = 0.0;
    loop {
        if (t > t_exit || alpha > 0.98) { break; }
        let v = clamp(sample_rho(o + d * t) / prm.scale, -1.0, 1.0);
        let r = abs(v);
        let s = select(pow(r, prm.gamma), 0.0, r <= 0.0);   // pow(0, y) is not guaranteed in WGSL
        let a = 1.0 - exp(-prm.opacity * s * knee_w(r) * prm.step);
        let base = select(prm.colour_neg, prm.colour_pos, v >= 0.0);
        colour = colour + (1.0 - alpha) * a * clamp(base * (0.45 + 0.9 * s), vec3(0.0), vec3(1.0));
        alpha = alpha + (1.0 - alpha) * a;
        t = t + prm.step;
    }
    return vec4(colour, alpha) * prm.fade;
}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let sp = ray_span(in.pos.xy);
    let o = sp.o; let d = sp.d; let t_enter = sp.t0; let t_exit = sp.t1;
    if (t_exit <= t_enter) { return vec4(0.0); }
    if (prm.mode == 2u) { return march_signed_iso(o, d, t_enter, t_exit); }
    if (prm.mode == 3u) { return march_signed_volume(o, d, t_enter, t_exit); }
    var colour = vec3(0.0); var alpha = 0.0;
    var t = t_enter;
    if (prm.mode == 1u) {
        var prev = -1.0;   // no previous sample yet
        loop {
            if (t > t_exit) { break; }
            let s = sample_rho(o + d * t) / prm.scale;
            if (s >= prm.iso) {
                // refine between the previous (below-iso) sample and this one by linear interpolation;
                // the first sample already inside the surface stays where it is
                var tb = t;
                if (prev >= 0.0) { tb = t - prm.step * (s - prm.iso) / max(s - prev, 1e-6); }
                let p = o + d * tb;
                let e = prm.step;
                let g = vec3(sample_rho(p + vec3(e, 0.0, 0.0)) - sample_rho(p - vec3(e, 0.0, 0.0)),
                             sample_rho(p + vec3(0.0, e, 0.0)) - sample_rho(p - vec3(0.0, e, 0.0)),
                             sample_rho(p + vec3(0.0, 0.0, e)) - sample_rho(p - vec3(0.0, 0.0, e)));
                // outward normal = -grad rho (density falls off outward); zero gradient -> unlit ambient
                let n = -g * inverseSqrt(max(dot(g, g), 1e-30));
                let diffuse = max(dot(n, -d), 0.0);
                return vec4(ramp(prm.iso) * (0.2 + 0.8 * diffuse), 1.0) * prm.fade;
            }
            prev = s; t = t + prm.step;
        }
        return vec4(0.0);
    }
    loop {
        if (t > t_exit || alpha > 0.98) { break; }
        let r = clamp(sample_rho(o + d * t) / prm.scale, 0.0, 1.0);
        let s = select(pow(r, prm.gamma), 0.0, r <= 0.0);   // pow(0, y) is not guaranteed in WGSL
        let a = 1.0 - exp(-prm.opacity * s * knee_w(r) * prm.step);
        colour = colour + (1.0 - alpha) * a * ramp(s);
        alpha = alpha + (1.0 - alpha) * a;
        t = t + prm.step;
    }
    return vec4(colour, alpha) * prm.fade; // premultiplied: fading scales colour and alpha alike
}
