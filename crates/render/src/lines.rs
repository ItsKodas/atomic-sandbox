//! Depth-tested line overlays: the simulation cell, a floor grid, the placement plane, axes.
//! One `LineList` pipeline over the shared camera uniform. Lines test against the depth the
//! spheres wrote but do not write depth themselves, so the volume pass (which composites
//! against the depth buffer) still draws the cloud over them instead of being cut at every line.
//! The builders at the bottom are pure and unit-tested; the pass just uploads what they produce.
use crate::spheres::CameraUniform;
use gpu::{GpuContext, UniformBuffer, DEPTH_FORMAT};

/// One line endpoint (28 bytes, no padding): world position (Bohr) and straight-alpha colour.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LineVertex { pub pos: [f32; 3], pub colour: [f32; 4] }

const _: () = assert!(std::mem::size_of::<LineVertex>() == 28);

pub struct LinePass { pipeline: wgpu::RenderPipeline, bind_group: wgpu::BindGroup, vertices: wgpu::Buffer, capacity: usize, count: u32 }

impl LinePass {
    pub fn new(ctx: &GpuContext, colour_format: wgpu::TextureFormat, camera: &UniformBuffer<CameraUniform>) -> LinePass {
        let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("lines"), source: wgpu::ShaderSource::Wgsl(include_str!("shaders/lines.wgsl").into()) });
        let bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("lines camera bgl"), entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0, visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }] });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("lines camera bg"), layout: &bgl, entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera.buffer.as_entire_binding() }] });
        let layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("lines"), layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<LineVertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4] })] },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_main"), compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: colour_format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::LineList, cull_mode: None, ..Default::default() },
            // test against the spheres' depth, never write: the volume pass keys on the depth buffer
            depth_stencil: Some(wgpu::DepthStencilState { format: DEPTH_FORMAT, depth_write_enabled: Some(false), depth_compare: Some(wgpu::CompareFunction::LessEqual), stencil: Default::default(), bias: Default::default() }),
            multisample: Default::default(), multiview_mask: None, cache: None,
        });
        let vertices = Self::make_buffer(ctx, 256);
        LinePass { pipeline, bind_group, vertices, capacity: 256, count: 0 }
    }

    fn make_buffer(ctx: &GpuContext, capacity: usize) -> wgpu::Buffer {
        ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("line vertices"), size: (capacity * std::mem::size_of::<LineVertex>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
    }

    /// Replaces the line list (pairs of vertices); the buffer grows (power of two) and never shrinks.
    pub fn set_lines(&mut self, ctx: &GpuContext, vertices: &[LineVertex]) {
        debug_assert!(vertices.len() % 2 == 0, "line list needs vertex pairs");
        if vertices.len() > self.capacity { self.capacity = vertices.len().next_power_of_two(); self.vertices = Self::make_buffer(ctx, self.capacity); }
        if !vertices.is_empty() { ctx.queue.write_buffer(&self.vertices, 0, bytemuck::cast_slice(vertices)); }
        self.count = vertices.len() as u32;
    }

    /// Draws into an already-begun pass with the painter's colour format and DEPTH_FORMAT attached.
    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.count == 0 { return; }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.draw(0..self.count, 0..1);
    }
}

// ---- builders (pure) ----

pub fn segment(out: &mut Vec<LineVertex>, a: [f32; 3], b: [f32; 3], colour: [f32; 4]) {
    out.push(LineVertex { pos: a, colour });
    out.push(LineVertex { pos: b, colour });
}

/// The 12 edges of the axis-aligned box `[min, max]`.
pub fn wire_box(out: &mut Vec<LineVertex>, min: [f32; 3], max: [f32; 3], colour: [f32; 4]) {
    let c = |mask: u8| [if mask & 1 != 0 { max[0] } else { min[0] }, if mask & 2 != 0 { max[1] } else { min[1] }, if mask & 4 != 0 { max[2] } else { min[2] }];
    for i in 0..8u8 {
        for bit in [1u8, 2, 4] {
            if i & bit == 0 { segment(out, c(i), c(i | bit), colour); }
        }
    }
}

/// A square grid in the plane through `centre` spanned by the unit vectors `u`, `v`: lines at
/// k·`spacing` for |k·spacing| ≤ `half_extent` along both directions; every `major_every`-th
/// line (k ≡ 0) gets the `major` colour. `half_extent` is rounded down to a whole number of spacings.
pub fn grid(out: &mut Vec<LineVertex>, centre: [f32; 3], u: [f32; 3], v: [f32; 3], half_extent: f32, spacing: f32, major_every: i32, minor: [f32; 4], major: [f32; 4]) {
    assert!(spacing > 0.0 && major_every > 0);
    let n = (half_extent / spacing).floor() as i32;
    let ext = n as f32 * spacing;
    let at = |a: f32, b: f32| [centre[0] + u[0] * a + v[0] * b, centre[1] + u[1] * a + v[1] * b, centre[2] + u[2] * a + v[2] * b];
    for k in -n..=n {
        let s = k as f32 * spacing;
        let colour = if k % major_every == 0 { major } else { minor };
        segment(out, at(s, -ext), at(s, ext), colour); // along v at u = s
        segment(out, at(-ext, s), at(ext, s), colour); // along u at v = s
    }
}

/// x (red), y (green), z (blue) from `origin`, each `len` long.
pub fn axes(out: &mut Vec<LineVertex>, origin: [f32; 3], len: f32) {
    let o = origin;
    segment(out, o, [o[0] + len, o[1], o[2]], [0.95, 0.30, 0.30, 0.9]);
    segment(out, o, [o[0], o[1] + len, o[2]], [0.40, 0.90, 0.40, 0.9]);
    segment(out, o, [o[0], o[1], o[2] + len], [0.35, 0.55, 1.00, 0.9]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::OrbitCamera;
    use crate::headless::{Headless, HEADLESS_FORMAT};
    use crate::spheres::{SphereInstance, SpherePass};
    use glam::Vec3;

    #[test]
    fn wire_box_has_twelve_edges_inside_the_box() {
        let mut v = Vec::new();
        wire_box(&mut v, [-1.0, -2.0, -3.0], [1.0, 2.0, 3.0], [1.0; 4]);
        assert_eq!(v.len(), 24);
        assert!(v.iter().all(|p| p.pos[0].abs() == 1.0 && p.pos[1].abs() == 2.0 && p.pos[2].abs() == 3.0));
        // every edge changes exactly one coordinate
        for e in v.chunks(2) { assert_eq!((0..3).filter(|&k| e[0].pos[k] != e[1].pos[k]).count(), 1); }
    }

    #[test]
    fn grid_line_count_and_major_lines() {
        let mut v = Vec::new();
        // half-extent 5.4 at spacing 1 → k = -5..=5 → 11 lines per direction, 22 segments
        grid(&mut v, [0.0; 3], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 5.4, 1.0, 5, [0.0; 4], [1.0; 4]);
        assert_eq!(v.len(), 44);
        let major = v.chunks(2).filter(|s| s[0].colour == [1.0; 4]).count();
        assert_eq!(major, 6); // k ∈ {-5, 0, 5} in each direction
        assert!(v.iter().all(|p| p.pos[1] == 0.0 && p.pos[0].abs() <= 5.0 && p.pos[2].abs() <= 5.0));
    }

    #[test]
    fn axes_are_rgb_from_the_origin() {
        let mut v = Vec::new();
        axes(&mut v, [1.0, 1.0, 1.0], 2.0);
        assert_eq!(v.len(), 6);
        assert_eq!(v[1].pos, [3.0, 1.0, 1.0]);
        assert_eq!(v[3].pos, [1.0, 3.0, 1.0]);
        assert_eq!(v[5].pos, [1.0, 1.0, 3.0]);
        assert!(v[0].colour[0] > v[0].colour[1] && v[2].colour[1] > v[2].colour[0] && v[4].colour[2] > v[4].colour[0]);
    }

    /// A line through the camera target lights the centre pixel; the same line is hidden where a
    /// sphere drawn first is in front of it (depth test on, no depth write needed for that).
    #[test]
    fn lines_light_pixels_and_are_occluded_by_spheres() {
        let (w, h) = (96u32, 96u32);
        let Some(hl) = Headless::new(w, h) else { return };
        let cam = OrbitCamera::new(Vec3::ZERO, 10.0);
        let ub = UniformBuffer::<CameraUniform>::new(&hl.ctx, "camera");
        ub.write(&hl.ctx, &CameraUniform::from_camera(&cam, w as f32, h as f32));
        let mut lines = LinePass::new(&hl.ctx, HEADLESS_FORMAT, &ub);
        let mut spheres = SpherePass::new(&hl.ctx, HEADLESS_FORMAT, &ub);
        let mut v = Vec::new();
        // a long horizontal green line through the origin, in the camera's right direction
        let right = (cam.view_proj(1.0).inverse() * glam::Vec4::new(1.0, 0.0, 0.0, 0.0)).truncate().normalize();
        segment(&mut v, (-right * 8.0).to_array(), (right * 8.0).to_array(), [0.0, 1.0, 0.0, 1.0]);
        lines.set_lines(&hl.ctx, &v);
        // a 1-px line through the exact centre rasterises into row h/2 or h/2 − 1: look at both
        let centre = |px: &[[u8; 4]], ch: usize| (h / 2 - 1..=h / 2).map(|y| px[(y * w + w / 2) as usize][ch]).max().unwrap();
        // no sphere: a centre pixel is green
        let mut enc = hl.ctx.device.create_command_encoder(&Default::default());
        { let mut rp = hl.begin(&mut enc, wgpu::Color::BLACK); lines.draw(&mut rp); }
        hl.ctx.queue.submit([enc.finish()]);
        let px = hl.read_rgba();
        let lit = px.iter().filter(|p| p[1] > 128).count();
        assert!(centre(&px, 1) > 128 && lit >= (w / 2) as usize, "line not drawn: centre green {}, lit {lit}", centre(&px, 1));
        // a red sphere at the origin drawn first hides the line's middle but not its ends
        spheres.set_instances(&hl.ctx, &[SphereInstance { centre: [0.0; 3], radius: 1.0, colour: [1.0, 0.0, 0.0], selected: 0.0 }]);
        let mut enc = hl.ctx.device.create_command_encoder(&Default::default());
        { let mut rp = hl.begin(&mut enc, wgpu::Color::BLACK); spheres.draw(&mut rp); lines.draw(&mut rp); }
        hl.ctx.queue.submit([enc.finish()]);
        let px = hl.read_rgba();
        assert!(centre(&px, 0) > 100 && centre(&px, 1) < 100, "sphere should occlude the line at the centre: red {} green {}", centre(&px, 0), centre(&px, 1));
        assert!(px.iter().any(|p| p[1] > 128 && p[0] < 64), "line ends should still be visible");
    }
}
