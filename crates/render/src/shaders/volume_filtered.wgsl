// sample_rho with FLOAT32_FILTERABLE: one hardware-trilinear textureSampleLevel through the volume's linear,
// clamp-to-edge sampler. Texel i's centre is at normalised coordinate (i + 0.5) / dims, so grid coordinate g
// maps to (g + 0.5) / dims. textureSampleLevel (explicit LOD) is valid in non-uniform control flow.
@group(1) @binding(1) var vol: texture_3d<f32>;
@group(1) @binding(3) var vol_sampler: sampler;

fn sample_rho(p: vec3<f32>) -> f32 {
    let dims = vec3<f32>(textureDimensions(vol));
    let uvw = (grid_coord(p, dims) + vec3(0.5)) / dims;
    return textureSampleLevel(vol, vol_sampler, uvw, 0.0).r;
}
