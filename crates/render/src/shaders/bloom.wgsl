// Bloom (render::bloom; display only): the bright parts of the finished 3D frame, blurred over a
// chain of half-size images and added back — the glow of dense dots, a hot volume core, the
// Camera view's lit atoms. Fullscreen triangles; one sampler (linear, clamp).
//   fs_prefilter: the frame → level 0 (half size): a 13-tap downsample of the soft-thresholded
//                 colour (brightness b = max channel; kept above `threshold` with a soft knee).
//   fs_down:      level k → k + 1, the same 13-tap filter (no threshold).
//   fs_up:        level k + 1 → k, a 3×3 tent, added onto level k (blend One, One).
//   fs_composite: level 0 → the frame, a tent × `intensity`, added (blend One, One).
struct BloomU { texel: vec2<f32>, threshold: f32, knee: f32, intensity: f32, pad0: f32, pad1: f32, pad2: f32 };
@group(0) @binding(0) var<uniform> u: BloomU;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;

struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var o: VOut;
    o.pos = vec4(p[vi], 0.0, 1.0);
    o.uv = vec2(p[vi].x * 0.5 + 0.5, 0.5 - p[vi].y * 0.5);
    return o;
}

fn tap(uv: vec2<f32>, dx: f32, dy: f32) -> vec3<f32> {
    return textureSampleLevel(src, smp, uv + vec2(dx, dy) * u.texel, 0.0).rgb;
}

// the 13-tap downsample (Jimenez 2014): a box of boxes, no shimmer
fn down13(uv: vec2<f32>) -> vec3<f32> {
    let a = tap(uv, -2.0, -2.0); let b = tap(uv, 0.0, -2.0); let c = tap(uv, 2.0, -2.0);
    let d = tap(uv, -1.0, -1.0); let e = tap(uv, 1.0, -1.0);
    let f = tap(uv, -2.0, 0.0); let g = tap(uv, 0.0, 0.0); let h = tap(uv, 2.0, 0.0);
    let i = tap(uv, -1.0, 1.0); let j = tap(uv, 1.0, 1.0);
    let k = tap(uv, -2.0, 2.0); let l = tap(uv, 0.0, 2.0); let m = tap(uv, 2.0, 2.0);
    return (d + e + i + j) * 0.125 + (a + b + f + g) * 0.03125 + (b + c + g + h) * 0.03125
         + (f + g + k + l) * 0.03125 + (g + h + l + m) * 0.03125;
}

fn tent(uv: vec2<f32>) -> vec3<f32> {
    return (tap(uv, -1.0, -1.0) + 2.0 * tap(uv, 0.0, -1.0) + tap(uv, 1.0, -1.0)
          + 2.0 * tap(uv, -1.0, 0.0) + 4.0 * tap(uv, 0.0, 0.0) + 2.0 * tap(uv, 1.0, 0.0)
          + tap(uv, -1.0, 1.0) + 2.0 * tap(uv, 0.0, 1.0) + tap(uv, 1.0, 1.0)) / 16.0;
}

@fragment fn fs_prefilter(in: VOut) -> @location(0) vec4<f32> {
    let c = down13(in.uv);
    let b = max(c.r, max(c.g, c.b));
    let soft = clamp(b - u.threshold + u.knee, 0.0, 2.0 * u.knee);
    let w = max(soft * soft / (4.0 * u.knee + 1e-5), b - u.threshold) / max(b, 1e-4);
    return vec4(c * w, 1.0);
}

@fragment fn fs_down(in: VOut) -> @location(0) vec4<f32> {
    return vec4(down13(in.uv), 1.0);
}

@fragment fn fs_up(in: VOut) -> @location(0) vec4<f32> {
    return vec4(tent(in.uv), 1.0);
}

@fragment fn fs_composite(in: VOut) -> @location(0) vec4<f32> {
    return vec4(tent(in.uv) * u.intensity, 0.0);
}
