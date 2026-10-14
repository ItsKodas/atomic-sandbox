//! Instanced impostor spheres: one camera-facing quad per nucleus, analytic ray-sphere in the fragment shader,
//! true sphere depth written so the volume pass can composite against it.
use crate::camera::{OrbitCamera, Viewport};
use gpu::{GpuContext, UniformBuffer, DEPTH_FORMAT};

/// One sphere, as a per-instance vertex attribute (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SphereInstance { pub centre: [f32; 3], pub radius: f32, pub colour: [f32; 3], pub selected: f32 }

/// Camera uniform shared by the sphere and volume shaders (160 bytes). Field order is the WGSL `Camera`
/// struct's — change both together.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// `viewport` is the size and `origin` the top-left corner (framebuffer pixels) of the rect the scene is
/// drawn into: the shaders map a fragment's framebuffer position to NDC as (pos − origin) / viewport.
pub struct CameraUniform { pub view_proj: [[f32; 4]; 4], pub inv_view_proj: [[f32; 4]; 4], pub eye: [f32; 3], pub _pad0: f32, pub viewport: [f32; 2], pub origin: [f32; 2] }

const _: () = assert!(std::mem::size_of::<CameraUniform>() == 160 && std::mem::size_of::<SphereInstance>() == 32);

impl CameraUniform {
    /// `width`/`height` are the render target size in pixels (used for the aspect and for pixel → NDC).
    pub fn from_camera(c: &OrbitCamera, width: f32, height: f32) -> CameraUniform {
        Self::in_viewport(c, Viewport::full(width, height))
    }

    /// For a scene drawn into `vp` (the render pass's viewport): aspect and pixel → NDC from its rect.
    pub fn in_viewport(c: &OrbitCamera, vp: Viewport) -> CameraUniform {
        let m = c.view_proj(vp.aspect());
        CameraUniform { view_proj: m.to_cols_array_2d(), inv_view_proj: m.inverse().to_cols_array_2d(), eye: c.eye().to_array(), _pad0: 0.0, viewport: vp.size, origin: vp.origin }
    }
}

pub struct SpherePass { pipeline: wgpu::RenderPipeline, bind_group: wgpu::BindGroup, instances: wgpu::Buffer, capacity: usize, count: u32 }

impl SpherePass {
    pub fn new(ctx: &GpuContext, colour_format: wgpu::TextureFormat, camera: &UniformBuffer<CameraUniform>) -> SpherePass {
        let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("spheres"), source: wgpu::ShaderSource::Wgsl(include_str!("shaders/spheres.wgsl").into()) });
        let bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("camera bgl"), entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }] });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("camera bg"), layout: &bgl, entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera.buffer.as_entire_binding() }] });
        let layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("spheres"), layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<SphereInstance>() as u64, step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32, 2 => Float32x3, 3 => Float32] })] },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_main"), compilation_options: Default::default(), targets: &[Some(colour_format.into())] }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: DEPTH_FORMAT, depth_write_enabled: Some(true), depth_compare: Some(wgpu::CompareFunction::Less), stencil: Default::default(), bias: Default::default() }),
            multisample: Default::default(), multiview_mask: None, cache: None,
        });
        let instances = Self::make_buffer(ctx, 16);
        SpherePass { pipeline, bind_group, instances, capacity: 16, count: 0 }
    }

    fn make_buffer(ctx: &GpuContext, capacity: usize) -> wgpu::Buffer {
        ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("sphere instances"), size: (capacity * std::mem::size_of::<SphereInstance>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
    }

    /// Replaces the instance list; the buffer grows (power of two) and never shrinks.
    pub fn set_instances(&mut self, ctx: &GpuContext, instances: &[SphereInstance]) {
        if instances.len() > self.capacity { self.capacity = instances.len().next_power_of_two(); self.instances = Self::make_buffer(ctx, self.capacity); }
        if !instances.is_empty() { ctx.queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(instances)); }
        self.count = instances.len() as u32;
    }

    /// Draws into an already-begun render pass whose colour target has `colour_format` and depth DEPTH_FORMAT.
    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.count == 0 { return; }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.instances.slice(..));
        pass.draw(0..6, 0..self.count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::OrbitCamera;
    use crate::headless::{Headless, HEADLESS_FORMAT};
    use gpu::UniformBuffer;
    use glam::Vec3;

    /// Renders one sphere of radius r at the origin, camera looking at the origin from distance d, and
    /// returns the set of lit pixels.
    fn render_one(r: f32, d: f32, w: u32, h: u32) -> Option<Vec<(u32, u32)>> {
        let cam = OrbitCamera::new(Vec3::ZERO, d);
        render_lit(&cam, SphereInstance { centre: [0.0; 3], radius: r, colour: [1.0, 1.0, 1.0], selected: 0.0 }, w, h)
    }

    /// Renders one sphere with the given camera and returns the lit pixels (row-major order).
    fn render_lit(cam: &OrbitCamera, inst: SphereInstance, w: u32, h: u32) -> Option<Vec<(u32, u32)>> {
        let hl = Headless::new(w, h)?;
        let ub = UniformBuffer::<CameraUniform>::new(&hl.ctx, "camera");
        ub.write(&hl.ctx, &CameraUniform::from_camera(cam, w as f32, h as f32));
        let mut pass = SpherePass::new(&hl.ctx, HEADLESS_FORMAT, &ub);
        pass.set_instances(&hl.ctx, &[inst]);
        let mut enc = hl.ctx.device.create_command_encoder(&Default::default());
        { let mut rp = hl.begin(&mut enc, wgpu::Color::BLACK); pass.draw(&mut rp); }
        hl.ctx.queue.submit([enc.finish()]);
        let px = hl.read_rgba();
        Some((0..h).flat_map(|y| (0..w).map(move |x| (x, y))).filter(|&(x, y)| px[(y * w + x) as usize][0] > 8).collect())
    }

    /// Off-axis spheres project to ellipses stretched radially by ~1/cos θ; near the corners of a wide
    /// viewport that exceeds any fixed screen-parallel quad margin. The GPU silhouette must match the CPU
    /// ray-sphere reference pixel for pixel (up to boundary rounding).
    #[test]
    fn off_axis_sphere_matches_cpu_ray_cast() {
        let (w, h) = (320u32, 180u32);
        let cam = OrbitCamera::new(Vec3::ZERO, 10.0);
        let r = 1.0f32;
        for &(px, py) in &[(40.0f32, 40.0f32), (290.0, 150.0), (160.0, 90.0)] {
            // centre on the focus plane under pixel (px, py)
            let (o, dir) = cam.pixel_ray(px, py, w as f32, h as f32);
            let (p, n) = cam.focus_plane();
            let centre = crate::camera::ray_plane(o, dir, p, n).unwrap();
            let inst = SphereInstance { centre: centre.to_array(), radius: r, colour: [1.0; 3], selected: 0.0 };
            let Some(lit) = render_lit(&cam, inst, w, h) else { return };
            let gpu: std::collections::HashSet<(u32, u32)> = lit.into_iter().collect();
            let cpu: std::collections::HashSet<(u32, u32)> = (0..h).flat_map(|y| (0..w).map(move |x| (x, y)))
                .filter(|&(x, y)| {
                    let (o, d) = cam.pixel_ray(x as f32 + 0.5, y as f32 + 0.5, w as f32, h as f32);
                    crate::camera::ray_sphere(o, d, centre, r).is_some()
                })
                .collect();
            let missing = cpu.difference(&gpu).count();
            let extra = gpu.difference(&cpu).count();
            eprintln!("sphere under px ({px},{py}): cpu {} px, gpu {} px, missing {missing}, extra {extra}", cpu.len(), gpu.len());
            assert!(missing <= 4 && extra <= 4, "silhouette mismatch at ({px},{py}): cpu {} gpu {} missing {missing} extra {extra}", cpu.len(), gpu.len());
        }
    }

    #[test]
    fn sphere_covers_a_disc_of_the_projected_radius() {
        let (w, h, r, d) = (256u32, 256u32, 1.0f32, 10.0f32);
        let Some(lit) = render_one(r, d, w, h) else { return };
        let fov = 50f32.to_radians();
        let expect_px = r / (d * (fov / 2.0).tan()) * (h as f32 / 2.0); // projected radius in pixels (small-angle)
        let area = lit.len() as f32;
        let r_meas = (area / std::f32::consts::PI).sqrt();
        eprintln!("disc: measured radius {r_meas} px vs expected {expect_px} px; centre ({}, {})", lit.iter().map(|p| p.0 as f32).sum::<f32>() / area, lit.iter().map(|p| p.1 as f32).sum::<f32>() / area);
        assert!((r_meas - expect_px).abs() < 2.0, "measured {r_meas} px vs expected {expect_px} px");
        let (cx, cy) = lit.iter().fold((0.0f32, 0.0f32), |(a, b), &(x, y)| (a + x as f32, b + y as f32));
        assert!((cx / area - 127.5).abs() < 1.5 && (cy / area - 127.5).abs() < 1.5, "disc off-centre");
    }

    /// A small sphere lying between a big sphere's centre and its front surface is hidden only if the big
    /// sphere writes its true surface depth (a flat impostor quad at the centre's depth would let it show).
    #[test]
    fn writes_true_sphere_depth() {
        let (w, h) = (64u32, 64u32);
        let Some(hl) = Headless::new(w, h) else { return };
        let cam = OrbitCamera::new(Vec3::ZERO, 10.0);
        let ub = UniformBuffer::<CameraUniform>::new(&hl.ctx, "camera");
        ub.write(&hl.ctx, &CameraUniform::from_camera(&cam, w as f32, h as f32));
        let mut pass = SpherePass::new(&hl.ctx, HEADLESS_FORMAT, &ub);
        let toward_eye = cam.eye().normalize();
        let big = SphereInstance { centre: [0.0; 3], radius: 1.0, colour: [1.0, 0.0, 0.0], selected: 0.0 };
        let small = SphereInstance { centre: (toward_eye * 0.5).to_array(), radius: 0.2, colour: [0.0, 1.0, 0.0], selected: 0.0 };
        for order in [[big, small], [small, big]] {
            pass.set_instances(&hl.ctx, &order);
            let mut enc = hl.ctx.device.create_command_encoder(&Default::default());
            { let mut rp = hl.begin(&mut enc, wgpu::Color::BLACK); pass.draw(&mut rp); }
            hl.ctx.queue.submit([enc.finish()]);
            let px = hl.read_rgba();
            let centre = px[(h / 2 * w + w / 2) as usize];
            // the rim term adds a little white, so test for green-dominant pixels rather than any green
            assert!(centre[0] > 100 && centre[1] < centre[0], "centre pixel {centre:?}: big sphere should occlude the small one");
            assert!(px.iter().all(|p| p[1] <= p[0]), "small sphere visible through the big one");
        }
    }

    #[test]
    fn no_instances_draws_nothing() {
        let Some(hl) = Headless::new(32, 32) else { return };
        let cam = OrbitCamera::new(Vec3::ZERO, 5.0);
        let ub = UniformBuffer::<CameraUniform>::new(&hl.ctx, "camera");
        ub.write(&hl.ctx, &CameraUniform::from_camera(&cam, 32.0, 32.0));
        let pass = SpherePass::new(&hl.ctx, HEADLESS_FORMAT, &ub);
        let mut enc = hl.ctx.device.create_command_encoder(&Default::default());
        { let mut rp = hl.begin(&mut enc, wgpu::Color::BLACK); pass.draw(&mut rp); }
        hl.ctx.queue.submit([enc.finish()]);
        assert!(hl.read_rgba().iter().all(|p| p[0] == 0));
    }

    #[test]
    fn element_table_is_display_only_and_total() {
        for z in 0..=20u8 { let _ = (crate::elements::colour(z), crate::elements::radius(z), crate::elements::symbol(z)); }
        assert_eq!(crate::elements::symbol(1), "H");
        assert_eq!(crate::elements::symbol(8), "O");
        assert!(crate::elements::radius(8) > crate::elements::radius(1));
    }
}
