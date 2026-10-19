// The Camera view's composite (camera.wgsl): the accumulated glow over the scene.
// ---- the composite: the accumulated glow over the scene, exposed and tone-mapped
@group(0) @binding(0) var accum: texture_2d<f32>;
@group(0) @binding(1) var<uniform> look: vec4<f32>;   // x = exposure (linear gain)

@vertex
fn vs_full(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let p = vec2(f32((vi << 1u) & 2u), f32(vi & 2u));
    return vec4(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_composite(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let s = textureLoad(accum, vec2<i32>(pos.xy), 0);
    // filmic: overlapping glows saturate toward white instead of clipping
    let rgb = vec3(1.0) - exp(-s.rgb * look.x * 1.6);
    let a = 1.0 - exp(-s.a * 1.8);
    return vec4(rgb, a);   // premultiplied over the scene (One, OneMinusSrcAlpha)
}
