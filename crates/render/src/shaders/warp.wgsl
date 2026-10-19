// The display warp (render::warp): group 2. `warp_back(p)` maps a point now to where the field was
// solved (the raymarches sample there); `warp_forward(p)` maps a point of the solved field to now
// (the dots are drawn there). Both are the identity while the warp is off.
struct WarpU { box_min: vec3<f32>, on: u32, box_max: vec3<f32>, pad: f32 };
@group(2) @binding(0) var<uniform> wp: WarpU;
@group(2) @binding(1) var warp_inv: texture_3d<f32>;
@group(2) @binding(2) var warp_fwd: texture_3d<f32>;
@group(2) @binding(3) var warp_smp: sampler;

// texel centres: the first and last texels sit on box_min and box_max
fn warp_uvw(p: vec3<f32>) -> vec3<f32> {
    let n = 32.0;
    let t = clamp((p - wp.box_min) / (wp.box_max - wp.box_min), vec3(0.0), vec3(1.0));
    return (t * (n - 1.0) + vec3(0.5)) / n;
}

fn warp_back(p: vec3<f32>) -> vec3<f32> {
    if (wp.on == 0u) { return p; }
    return p - textureSampleLevel(warp_inv, warp_smp, warp_uvw(p), 0.0).xyz;
}

fn warp_forward(p: vec3<f32>) -> vec3<f32> {
    if (wp.on == 0u) { return p; }
    return p + textureSampleLevel(warp_fwd, warp_smp, warp_uvw(p), 0.0).xyz;
}
