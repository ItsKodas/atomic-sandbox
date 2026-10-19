// The Camera view's composite (camera.wgsl): the accumulated atoms over the scene, exposed and
// tone-mapped — a filmic curve for the glow (overlaps saturate toward white), plain for the balls.
@group(0) @binding(0) var accum: texture_2d<f32>;
@group(0) @binding(1) var<uniform> look: vec4<f32>;   // x = exposure gain, y = solidity 0–1

@vertex
fn vs_full(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let p = vec2(f32((vi << 1u) & 2u), f32(vi & 2u));
    return vec4(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_composite(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    // the atoms were drawn at 2× (glow::SS): a 4 × 4 tent around this pixel's 2 × 2 block — smooth,
    // slightly soft outlines and intersection lines
    let base = vec2<i32>(pos.xy) * 2 - vec2(1, 1);
    var wt = array<f32, 4>(1.0, 3.0, 3.0, 1.0);
    var s = vec4(0.0);
    for (var j = 0; j < 4; j++) {
        for (var i = 0; i < 4; i++) {
            let at = clamp(base + vec2(i, j), vec2(0, 0), vec2<i32>(textureDimensions(accum)) - vec2(1, 1));
            s += textureLoad(accum, at, 0) * (wt[i] * wt[j] / 64.0);
        }
    }
    let h = look.y;
    let filmic = vec3(1.0) - exp(-s.rgb * look.x * 1.6);
    let plain = clamp(s.rgb * look.x, vec3(0.0), vec3(1.0));
    let rgb = mix(filmic, plain, h);
    let a = mix(1.0 - exp(-s.a * 1.8), clamp(s.a, 0.0, 1.0), h);
    return vec4(rgb, a);   // premultiplied over the scene (One, OneMinusSrcAlpha)
}
