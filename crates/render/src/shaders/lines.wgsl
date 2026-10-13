// Line overlays. `Camera` must match `CameraUniform` in spheres.rs.
struct Camera { view_proj: mat4x4<f32>, inv_view_proj: mat4x4<f32>, eye: vec3<f32>, _p0: f32, viewport: vec2<f32>, _p1: vec2<f32> };
@group(0) @binding(0) var<uniform> cam: Camera;

struct VIn { @location(0) pos: vec3<f32>, @location(1) colour: vec4<f32> };
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) colour: vec4<f32> };

@vertex
fn vs_main(in: VIn) -> VOut {
    var o: VOut;
    o.pos = cam.view_proj * vec4(in.pos, 1.0);
    o.colour = in.colour;
    return o;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    return in.colour;
}
