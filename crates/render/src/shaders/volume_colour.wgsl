// The colour volume's raymarch (`ColourVolumePass`), appended to volume.wgsl and sharing its camera, params,
// vertex stage and ray span. The volume is rgba16float (always filterable: one sampling path): rgb =
// Σ_i f_i c_i s_i, a = Σ_i f_i s_i over the included orbitals (spec §5.6; s_i = ψ_i²/P_i), so rgb/a is the hue
// of what dominates locally and a is how much there is. Two of them are bound — previous and current — and
// every sample is their lerp by `mix_w`: linear in the data, like the scalar crossfade.
//
// Two blends (`prm.additive`):
//   additive (1): emission only — the colours sum along the ray, an inner shell shines through the outer
//     ones. The sum saturates smoothly (1 − e^−m on its brightest channel m, the hue kept) so overlapping
//     shells do not clip to white.
//   weighted (0): front-to-back emission-absorption — the front shell hides what is behind it.
// Both use today's transfer function (gamma on the strength a) and the premultiplied output (`fade`
// scales colour and alpha alike).
@group(1) @binding(5) var colour_cur: texture_3d<f32>;
@group(1) @binding(6) var colour_sampler: sampler;
@group(1) @binding(7) var colour_prev: texture_3d<f32>;

fn sample_colour_one(t: texture_3d<f32>, p: vec3<f32>) -> vec4<f32> {
    let dims = vec3<f32>(textureDimensions(t));
    let uvw = (grid_coord(p, dims) + vec3(0.5)) / dims;
    return textureSampleLevel(t, colour_sampler, uvw, 0.0);
}

fn sample_colour(p_now: vec3<f32>) -> vec4<f32> {
    let p = warp_back(p_now);
    if (prm.mix_w >= 1.0) { return sample_colour_one(colour_cur, p); }
    if (prm.mix_w <= 0.0) { return sample_colour_one(colour_prev, p); }
    return mix(sample_colour_one(colour_prev, p), sample_colour_one(colour_cur, p), prm.mix_w);
}

@fragment fn fs_colour(in: VOut) -> @location(0) vec4<f32> {
    let sp = ray_span(in.pos.xy);
    if (sp.t1 <= sp.t0) { return vec4(0.0); }
    var colour = vec3(0.0); var alpha = 0.0;
    var t = sp.t0;
    let additive = prm.additive != 0u;
    loop {
        if (t > sp.t1 || (!additive && alpha > 0.98)) { break; }
        let c = sample_colour(sp.o + sp.d * t);
        let r = clamp(c.a, 0.0, 1.0);
        let s = select(pow(r, prm.gamma), 0.0, r <= 0.0);   // pow(0, y) is not guaranteed in WGSL
        let a = 1.0 - exp(-prm.opacity * s * knee_w(r) * prm.step);
        let hue = clamp(c.rgb / max(c.a, 1e-6), vec3(0.0), vec3(1.0)) * (0.4 + 0.8 * s);
        if (additive) {
            colour = colour + a * hue;
        } else {
            colour = colour + (1.0 - alpha) * a * hue;
            alpha = alpha + (1.0 - alpha) * a;
        }
        t = t + prm.step;
    }
    if (additive) {
        // saturate the brightest channel, keep the hue: the ratio of the channels is the ratio of what was summed
        let m = max(max(colour.r, colour.g), max(colour.b, 1e-6));
        return vec4(colour / m * (1.0 - exp(-m)), 0.0) * prm.fade;
    }
    return vec4(colour, alpha) * prm.fade;
}
