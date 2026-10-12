// Fullscreen density raymarch (volume / isosurface), composited over the sphere pass via its depth buffer.
// `Camera` must match `CameraUniform` in spheres.rs; `Params` must match `VolumeUniform` in volume.rs.
struct Camera { view_proj: mat4x4<f32>, inv_view_proj: mat4x4<f32>, eye: vec3<f32>, _p0: f32, viewport: vec2<f32>, _p1: vec2<f32> };
struct Params { box_min: vec3<f32>, mode: u32, box_max: vec3<f32>, scale: f32, gamma: f32, iso: f32, step: f32, opacity: f32 };
@group(0) @binding(0) var<uniform> cam: Camera;
@group(1) @binding(0) var<uniform> prm: Params;
// r32float is unfilterable without FLOAT32_FILTERABLE, so the volume is read with textureLoad and
// interpolated trilinearly by hand (no sampler binding).
@group(1) @binding(1) var vol: texture_3d<f32>;
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
// first/last sample positions, so interpolate in continuous grid-index space (texel centres), clamped to the box.
fn sample_rho(p: vec3<f32>) -> f32 {
    let dims = vec3<i32>(textureDimensions(vol));
    let top = vec3<f32>(dims - vec3(1));
    let g = clamp((p - prm.box_min) / (prm.box_max - prm.box_min), vec3(0.0), vec3(1.0)) * top;
    let gf = floor(g);
    let f = g - gf;
    let i0 = min(vec3<i32>(gf), dims - vec3(1));
    let i1 = min(i0 + vec3(1), dims - vec3(1));
    let c000 = textureLoad(vol, vec3(i0.x, i0.y, i0.z), 0).r;
    let c100 = textureLoad(vol, vec3(i1.x, i0.y, i0.z), 0).r;
    let c010 = textureLoad(vol, vec3(i0.x, i1.y, i0.z), 0).r;
    let c110 = textureLoad(vol, vec3(i1.x, i1.y, i0.z), 0).r;
    let c001 = textureLoad(vol, vec3(i0.x, i0.y, i1.z), 0).r;
    let c101 = textureLoad(vol, vec3(i1.x, i0.y, i1.z), 0).r;
    let c011 = textureLoad(vol, vec3(i0.x, i1.y, i1.z), 0).r;
    let c111 = textureLoad(vol, vec3(i1.x, i1.y, i1.z), 0).r;
    let x00 = mix(c000, c100, f.x); let x10 = mix(c010, c110, f.x);
    let x01 = mix(c001, c101, f.x); let x11 = mix(c011, c111, f.x);
    return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}

// Scene depth at this pixel as a distance along the ray (o, d). The sphere pass stores clip.z / clip.w
// (0..1, DirectX/WebGPU convention), so unprojecting (ndc.xy, z) with the same pixel-centre NDC recovers the
// sphere hit point on this very ray.
fn depth_to_t(frag: vec2<f32>, o: vec3<f32>, d: vec3<f32>) -> f32 {
    let z = textureLoad(scene_depth, vec2<i32>(frag), 0);
    if (z >= 1.0) { return 1e30; }
    let ndc = vec2(frag.x / cam.viewport.x * 2.0 - 1.0, 1.0 - frag.y / cam.viewport.y * 2.0);
    let pw = cam.inv_view_proj * vec4(ndc, z, 1.0);
    return dot(pw.xyz / pw.w - o, d);
}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    // same ray reconstruction as spheres.wgsl, so depth_to_t lines up with the sphere hits
    let ndc = vec2(in.pos.x / cam.viewport.x * 2.0 - 1.0, 1.0 - in.pos.y / cam.viewport.y * 2.0);
    let pn = cam.inv_view_proj * vec4(ndc, 0.0, 1.0);
    let pf = cam.inv_view_proj * vec4(ndc, 1.0, 1.0);
    let o = pn.xyz / pn.w;
    let d = normalize(pf.xyz / pf.w - o);
    // slab test
    let inv = 1.0 / d;
    let t0 = (prm.box_min - o) * inv; let t1 = (prm.box_max - o) * inv;
    let tmin = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let tmax = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    let t_enter = max(tmin, 0.0);
    let t_exit = min(tmax, depth_to_t(in.pos.xy, o, d));
    if (t_exit <= t_enter) { return vec4(0.0); }
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
                return vec4(ramp(prm.iso) * (0.2 + 0.8 * diffuse), 1.0);
            }
            prev = s; t = t + prm.step;
        }
        return vec4(0.0);
    }
    loop {
        if (t > t_exit || alpha > 0.98) { break; }
        let r = clamp(sample_rho(o + d * t) / prm.scale, 0.0, 1.0);
        let s = select(pow(r, prm.gamma), 0.0, r <= 0.0);   // pow(0, y) is not guaranteed in WGSL
        let a = 1.0 - exp(-prm.opacity * s * prm.step);
        colour = colour + (1.0 - alpha) * a * ramp(s);
        alpha = alpha + (1.0 - alpha) * a;
        t = t + prm.step;
    }
    return vec4(colour, alpha); // premultiplied
}
