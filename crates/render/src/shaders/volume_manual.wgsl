// sample_one for devices without FLOAT32_FILTERABLE: r32float is then unfilterable, so the volume is
// read with eight textureLoads and interpolated trilinearly by hand (no sampler binding).
@group(1) @binding(1) var vol: texture_3d<f32>;
@group(1) @binding(4) var vol_prev: texture_3d<f32>;

fn sample_one(t: texture_3d<f32>, p: vec3<f32>) -> f32 {
    let dims = vec3<i32>(textureDimensions(t));
    let g = grid_coord(p, vec3<f32>(dims));
    let gf = floor(g);
    let f = g - gf;
    let i0 = min(vec3<i32>(gf), dims - vec3(1));
    let i1 = min(i0 + vec3(1), dims - vec3(1));
    let c000 = textureLoad(t, vec3(i0.x, i0.y, i0.z), 0).r;
    let c100 = textureLoad(t, vec3(i1.x, i0.y, i0.z), 0).r;
    let c010 = textureLoad(t, vec3(i0.x, i1.y, i0.z), 0).r;
    let c110 = textureLoad(t, vec3(i1.x, i1.y, i0.z), 0).r;
    let c001 = textureLoad(t, vec3(i0.x, i0.y, i1.z), 0).r;
    let c101 = textureLoad(t, vec3(i1.x, i0.y, i1.z), 0).r;
    let c011 = textureLoad(t, vec3(i0.x, i1.y, i1.z), 0).r;
    let c111 = textureLoad(t, vec3(i1.x, i1.y, i1.z), 0).r;
    let x00 = mix(c000, c100, f.x); let x10 = mix(c010, c110, f.x);
    let x01 = mix(c001, c101, f.x); let x11 = mix(c011, c111, f.x);
    return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}
