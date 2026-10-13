//! Fullscreen raymarch of the density volume (volume and isosurface modes), composited over the sphere
//! pass by reading its depth buffer as a texture.
use crate::spheres::CameraUniform;
use gpu::{GpuContext, UniformBuffer, Volume3D};

/// Raymarch parameters (48 bytes). Field order is the WGSL `Params` struct's — change both together.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VolumeUniform {
    /// World position (Bohr) of the first grid sample; `box_max` is the last. 0 = volume, 1 = isosurface.
    pub box_min: [f32; 3], pub mode: u32,
    /// Density divisor: rho/scale ∈ [0, 1] at the peak.
    pub box_max: [f32; 3], pub scale: f32,
    /// Power-law exponent; iso threshold (fraction of scale); march step in Bohr.
    pub gamma: f32, pub iso: f32, pub step: f32,
    /// Volume mode: alpha per unit length at rho/scale = 1.
    pub opacity: f32,
}

const _: () = assert!(std::mem::size_of::<VolumeUniform>() == 48);

impl Default for VolumeUniform {
    /// Empty box (draws nothing until a grid is bound and the box set), otherwise the viewer's defaults.
    fn default() -> Self { VolumeUniform { box_min: [0.0; 3], mode: 0, box_max: [0.0; 3], scale: 1.0, gamma: 0.5, iso: 0.3, step: 0.1, opacity: 2.0 } }
}

/// How the raymarch reads the r32float density between grid points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sampling {
    /// One hardware-trilinear `textureSampleLevel` through `Volume3D::sampler`. Needs
    /// `Features::FLOAT32_FILTERABLE` (`GpuContext::float32_filterable`).
    Filtered,
    /// Eight `textureLoad`s and a trilinear blend in the shader: works on every device.
    Manual,
}

impl Sampling {
    /// `Filtered` when the device has FLOAT32_FILTERABLE, else `Manual`.
    pub fn best(ctx: &GpuContext) -> Sampling { if ctx.float32_filterable { Sampling::Filtered } else { Sampling::Manual } }
}

pub struct VolumePass {
    pipeline: wgpu::RenderPipeline,
    camera_bg: wgpu::BindGroup,
    volume_bgl: wgpu::BindGroupLayout,
    volume_bg: Option<wgpu::BindGroup>,
    params: UniformBuffer<VolumeUniform>,
    sampling: Sampling,
}

impl VolumePass {
    /// With the best sampling path the device supports ([`Sampling::best`]).
    pub fn new(ctx: &GpuContext, colour_format: wgpu::TextureFormat, camera: &UniformBuffer<CameraUniform>) -> VolumePass {
        Self::with_sampling(ctx, colour_format, camera, Sampling::best(ctx))
    }

    /// With an explicit sampling path (tests compare the two). Panics on `Filtered` without
    /// FLOAT32_FILTERABLE (the layout would not validate).
    pub fn with_sampling(ctx: &GpuContext, colour_format: wgpu::TextureFormat, camera: &UniformBuffer<CameraUniform>, sampling: Sampling) -> VolumePass {
        assert!(sampling == Sampling::Manual || ctx.float32_filterable, "Sampling::Filtered needs Features::FLOAT32_FILTERABLE");
        let dev = &ctx.device;
        let filtered = sampling == Sampling::Filtered;
        let variant = if filtered { include_str!("shaders/volume_filtered.wgsl") } else { include_str!("shaders/volume_manual.wgsl") };
        let source = format!("{}\n{variant}", include_str!("shaders/volume.wgsl"));
        let shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("volume"), source: wgpu::ShaderSource::Wgsl(source.into()) });
        let frag = wgpu::ShaderStages::FRAGMENT;
        // group 0: same layout as SpherePass, so one camera uniform buffer serves both passes
        let camera_bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("volume camera bgl"), entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }] });
        let camera_bg = dev.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("volume camera bg"), layout: &camera_bgl, entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera.buffer.as_entire_binding() }] });
        // group 1: params, density (r32float: filterable + a linear sampler at binding 3, or
        // unfilterable and read with textureLoad), scene depth (textureLoad)
        let mut entries = vec![
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: frag, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: frag, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: filtered }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: frag, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
        ];
        if filtered {
            entries.push(wgpu::BindGroupLayoutEntry { binding: 3, visibility: frag, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None });
        }
        let volume_bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("volume bgl"), entries: &entries });
        let layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&camera_bgl), Some(&volume_bgl)], immediate_size: 0 });
        let pipeline = dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("volume"), layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_main"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState {
                format: colour_format,
                // premultiplied over: out = src + (1 - src.a) * dst, for colour and alpha
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })] }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            // no depth attachment: the scene depth is sampled as a texture instead
            depth_stencil: None,
            multisample: Default::default(), multiview_mask: None, cache: None,
        });
        let params = UniformBuffer::<VolumeUniform>::new(ctx, "volume params");
        params.write(ctx, &VolumeUniform::default());
        VolumePass { pipeline, camera_bg, volume_bgl, volume_bg: None, params, sampling }
    }

    pub fn sampling(&self) -> Sampling { self.sampling }

    /// Rebind when the Volume3D or the depth view changes (grid change / resize / a new texture
    /// from the GPU solver). The volume must be unpadded n³ (`Volume3D` contract).
    pub fn bind(&mut self, ctx: &GpuContext, volume: &Volume3D, scene_depth: &wgpu::TextureView) {
        let mut entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: self.params.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&volume.view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(scene_depth) },
        ];
        if self.sampling == Sampling::Filtered {
            entries.push(wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&volume.sampler) });
        }
        self.volume_bg = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("volume bg"), layout: &self.volume_bgl, entries: &entries }));
    }

    pub fn set_params(&self, ctx: &GpuContext, params: &VolumeUniform) { self.params.write(ctx, params); }

    /// Fullscreen triangle; depth test off, blends over the colour target; reads the sphere depth to stop rays.
    /// Must run in a pass without a depth attachment (see [`overlay_pass`]). No-op until [`VolumePass::bind`].
    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        let Some(bg) = &self.volume_bg else { return };
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.camera_bg, &[]);
        pass.set_bind_group(1, bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// A colour-only pass that keeps (loads) the existing colour, for the volume to blend over. It has no depth
/// attachment because the volume samples the depth texture.
pub fn overlay_pass<'a>(enc: &'a mut wgpu::CommandEncoder, colour: &'a wgpu::TextureView) -> wgpu::RenderPass<'a> {
    enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("volume overlay"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: colour, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }, depth_slice: None })],
        depth_stencil_attachment: None,
        timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::OrbitCamera;
    use crate::headless::{Headless, HEADLESS_FORMAT};
    use crate::spheres::{CameraUniform, SphereInstance};
    use crate::Renderer;
    use gpu::{UniformBuffer, Volume3D};
    use glam::Vec3;

    /// Gaussian density of width 1 Bohr centred at `centre` on a 33³ grid spanning [−4, 4]³.
    fn gaussian_volume(ctx: &gpu::GpuContext, centre: [f32; 3]) -> (Volume3D, [f32; 3], [f32; 3]) {
        let n = 33u32; let h = 0.25f32; let lo = -4.0f32;
        let mut data = Vec::with_capacity((n * n * n) as usize);
        for k in 0..n { for j in 0..n { for i in 0..n {
            let (x, y, z) = (lo + i as f32 * h, lo + j as f32 * h, lo + k as f32 * h);
            let r2 = (x - centre[0]).powi(2) + (y - centre[1]).powi(2) + (z - centre[2]).powi(2);
            data.push((-r2 / 2.0).exp());
        }}}
        let vol = Volume3D::new(ctx, [n; 3]);
        vol.upload(ctx, &data);
        (vol, [lo; 3], [lo + (n - 1) as f32 * h; 3])
    }

    fn render(centre: [f32; 3], mode: u32, w: u32, h: u32) -> Option<Vec<[u8; 4]>> {
        let hl = Headless::new(w, h)?;
        let sampling = Sampling::best(&hl.ctx);
        Some(render_with(&hl, sampling, centre, mode))
    }

    /// One frame of the Gaussian through a `VolumePass` with the given sampling path.
    fn render_with(hl: &Headless, sampling: Sampling, centre: [f32; 3], mode: u32) -> Vec<[u8; 4]> {
        let (w, h) = (hl.width, hl.height);
        let cam = OrbitCamera::new(Vec3::ZERO, 12.0);
        let ub = UniformBuffer::<CameraUniform>::new(&hl.ctx, "camera");
        ub.write(&hl.ctx, &CameraUniform::from_camera(&cam, w as f32, h as f32));
        let (vol, bmin, bmax) = gaussian_volume(&hl.ctx, centre);
        let mut pass = VolumePass::with_sampling(&hl.ctx, HEADLESS_FORMAT, &ub, sampling);
        assert_eq!(pass.sampling(), sampling);
        pass.bind(&hl.ctx, &vol, &hl.depth);
        pass.set_params(&hl.ctx, &VolumeUniform { box_min: bmin, mode, box_max: bmax, scale: 1.0, gamma: 0.5, iso: 0.3, step: 0.1, opacity: 2.0 });
        let mut enc = hl.ctx.device.create_command_encoder(&Default::default());
        // The depth texture is sampled by the volume pass, so it cannot also be the attachment of the pass
        // that draws the volume: clear colour + depth first, then march in a colour-only overlay pass.
        drop(hl.begin(&mut enc, wgpu::Color::BLACK));
        { let mut rp = overlay_pass(&mut enc, &hl.colour_view); pass.draw(&mut rp); }
        hl.ctx.queue.submit([enc.finish()]);
        hl.read_rgba()
    }

    /// The hardware-filtered path (one `textureSampleLevel`) and the manual trilinear path (eight
    /// `textureLoad`s) must draw the same picture: same texel-centre convention, same clamping.
    /// Hardware filtering quantises the interpolation weights (8 fractional bits on common GPUs),
    /// hence a tolerance rather than equality. Off-centre so the box is not symmetric about the
    /// texel grid, and both modes (the isosurface's normals come from differences of samples).
    #[test]
    fn filterable_and_manual_paths_agree() {
        let (w, h) = (128u32, 128u32);
        let Some(hl) = Headless::new(w, h) else { return };
        if !hl.ctx.float32_filterable {
            eprintln!("adapter lacks FLOAT32_FILTERABLE: only the manual path exists; skipping");
            return;
        }
        assert_eq!(Sampling::best(&hl.ctx), Sampling::Filtered);
        for (mode, centre) in [(0u32, [0.3f32, -0.2, 0.1]), (1, [0.3, -0.2, 0.1])] {
            let a = render_with(&hl, Sampling::Filtered, centre, mode);
            let b = render_with(&hl, Sampling::Manual, centre, mode);
            let (mut worst, mut at, mut lit) = (0i32, 0usize, 0usize);
            for (i, (p, q)) in a.iter().zip(&b).enumerate() {
                let d = (0..4).map(|c| (p[c] as i32 - q[c] as i32).abs()).max().unwrap();
                if d > worst { worst = d; at = i; }
                if p[0] as u32 + p[1] as u32 + p[2] as u32 > 30 { lit += 1; }
            }
            eprintln!("mode {mode}: max channel diff {worst}/255 at ({}, {}); {lit} lit pixels", at as u32 % w, at as u32 / w);
            assert!(lit > 500, "mode {mode}: test is vacuous ({lit} lit pixels)");
            assert!(worst < 8, "mode {mode}: filtered vs manual differ by {worst}/255 at pixel {at}");
        }
    }

    fn brightness(px: &[[u8; 4]], w: u32, x: u32, y: u32) -> u32 { let p = px[(y * w + x) as usize]; p[0] as u32 + p[1] as u32 + p[2] as u32 }

    #[test]
    fn centred_gaussian_is_bright_in_the_middle_and_symmetric() {
        let (w, h) = (128u32, 128u32);
        let Some(px) = render([0.0; 3], 0, w, h) else { return };
        let centre = brightness(&px, w, 64, 64);
        let edge = brightness(&px, w, 4, 64);
        let left: u32 = (0..h).flat_map(|y| (0..64).map(move |x| (x, y))).map(|(x, y)| brightness(&px, w, x, y)).sum();
        let right: u32 = (0..h).flat_map(|y| (64..128).map(move |x| (x, y))).map(|(x, y)| brightness(&px, w, x, y)).sum();
        let asym = (left as f32 - right as f32).abs() / (left + right) as f32;
        eprintln!("volume: centre {centre} edge {edge} left {left} right {right} asymmetry {asym}");
        assert!(centre > 200 && edge < 30, "centre {centre} edge {edge}");
        assert!(asym < 0.02, "left/right asymmetry {asym}");
    }

    #[test]
    fn offset_gaussian_moves_the_bright_spot() {
        let (w, h) = (128u32, 128u32);
        let Some(px) = render([2.0, 0.0, 0.0], 0, w, h) else { return };
        let (mut best, mut bx) = (0u32, 0u32);
        for x in 0..w { let b = brightness(&px, w, x, 64); if b > best { best = b; bx = x; } }
        eprintln!("offset gaussian: brightest x on row 64 = {bx} ({best})");
        assert!(bx != 64, "bright spot did not move: x = {bx}");
    }

    #[test]
    fn isosurface_mode_renders_an_opaque_blob() {
        let (w, h) = (128u32, 128u32);
        let Some(px) = render([0.0; 3], 1, w, h) else { return };
        eprintln!("isosurface: centre {:?} brightness {} corner {}", px[(64 * w + 64) as usize], brightness(&px, w, 64, 64), brightness(&px, w, 2, 2));
        assert_eq!(px[(64 * w + 64) as usize][3], 255, "iso surface must be opaque at the centre");
        assert!(brightness(&px, w, 64, 64) > 150);
        assert!(brightness(&px, w, 2, 2) < 10);
    }

    /// Renders a single sphere (offset `along_view` Bohr from the volume centre along the eye direction;
    /// positive = toward the eye) through the full `Renderer`, three ways: sphere only, sphere + volume,
    /// volume only. Returns the centre pixel of each.
    fn composite(along_view: f32) -> Option<([u8; 4], [u8; 4], [u8; 4])> {
        let (w, h) = (128u32, 128u32);
        let hl = Headless::new(w, h)?;
        let cam = OrbitCamera::new(Vec3::ZERO, 12.0);
        let mut r = Renderer::new(&hl.ctx, HEADLESS_FORMAT);
        r.update_camera(&hl.ctx, &cam, w as f32, h as f32);
        let (vol, bmin, bmax) = gaussian_volume(&hl.ctx, [0.0; 3]);
        r.volume_params = VolumeUniform { box_min: bmin, mode: 0, box_max: bmax, scale: 1.0, gamma: 0.5, iso: 0.3, step: 0.1, opacity: 2.0 };
        r.sync_volume_params(&hl.ctx);
        let sphere = SphereInstance { centre: (cam.eye().normalize() * along_view).to_array(), radius: 1.0, colour: [0.2, 0.9, 0.2], selected: 0.0 };
        let frame = |r: &Renderer| {
            let mut enc = hl.ctx.device.create_command_encoder(&Default::default());
            r.render(&mut enc, &hl.colour_view, &hl.depth, wgpu::Color::BLACK);
            hl.ctx.queue.submit([enc.finish()]);
            hl.read_rgba()[(h / 2 * w + w / 2) as usize]
        };
        r.spheres.set_instances(&hl.ctx, &[sphere]);
        let sphere_only = frame(&r); // volume not bound yet: the volume pass draws nothing
        r.volume.bind(&hl.ctx, &vol, &hl.depth);
        let both = frame(&r);
        r.spheres.set_instances(&hl.ctx, &[]);
        let volume_only = frame(&r);
        Some((sphere_only, both, volume_only))
    }

    fn diff(a: [u8; 4], b: [u8; 4]) -> [i32; 3] { [0, 1, 2].map(|i| a[i] as i32 - b[i] as i32) }

    /// Sphere surface at 5 Bohr toward the eye (inside the box, beyond essentially all the density): the
    /// march must stop at the sphere, so its pixel is the bare sphere colour, not the bright volume.
    #[test]
    fn sphere_in_front_of_the_volume_is_not_tinted() {
        let Some((sphere, both, vol)) = composite(4.0) else { return };
        eprintln!("in front: sphere only {sphere:?}, sphere + volume {both:?}, volume only {vol:?}");
        assert!(diff(both, sphere).iter().all(|d| d.abs() <= 2), "sphere tinted by volume behind it: {both:?} vs {sphere:?}");
        assert!(diff(vol, sphere).iter().map(|d| d.abs()).sum::<i32>() > 100, "test is vacuous: volume looks like the sphere");
    }

    /// Sphere behind the density peak: the optically thick volume in front tints (here almost hides) it.
    #[test]
    fn sphere_behind_the_volume_is_tinted() {
        let Some((sphere, both, vol)) = composite(-4.0) else { return };
        eprintln!("behind: sphere only {sphere:?}, sphere + volume {both:?}, volume only {vol:?}");
        assert!(diff(both, sphere).iter().map(|d| d.abs()).sum::<i32>() > 100, "sphere not tinted by volume in front: {both:?} vs {sphere:?}");
        assert!(diff(both, vol).iter().all(|d| d.abs() <= 12), "thick volume in front should dominate: {both:?} vs volume {vol:?}");
    }
}
