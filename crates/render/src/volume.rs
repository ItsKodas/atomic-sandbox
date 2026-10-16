//! Fullscreen raymarch of the density volume (volume and isosurface modes), composited over the sphere
//! pass by reading its depth buffer as a texture.
//!
//! Two densities are bound — the previous and the current — and every sample is
//! lerp(previous, current, `mix`) (the viewer's density crossfade: the same α as the nuclei's
//! smooth motion, so cloud and spheres move together). Sampling is linear in the data, so the blend
//! of two samples is the sample of the blended grid: a crossfade at α draws exactly the density
//! (1 − α) ρ_prev + α ρ_cur. `mix = 1` (the default) or [`VolumePass::bind`] (one texture in both
//! slots) is the plain single-density raymarch.
//!
//! # Signed fields and the colour volume (orbital viewer, spec §6)
//! The same pass draws a **signed** scalar field — one Kohn–Sham orbital ψ̃, or the spin density
//! m = ρ↑ − ρ↓ — in [`MODE_SIGNED_ISO`] (the first sample with |s| ≥ iso is a hit, lit with the normal
//! −sign(s)∇s and coloured by the lobe's sign) and [`MODE_SIGNED_VOLUME`] (emission–absorption on |s|
//! with the sign's colour); `s = value / scale`, the colours are `colour_pos` / `colour_neg`. Signed
//! fields crossfade correctly because the sign rule (`qm::orbitals` §5.4) keeps consecutive fields in
//! phase. [`ColourVolumePass`] is the second pipeline, for the rgba16float colour volume of "all
//! orbitals coloured": the same group 0 and the same params, with the previous and the current
//! colour volume bound as a pair, blended **additively** (emission only: inner shells shine through
//! the outer ones) or **weighted** (emission–absorption: the front shell hides what is behind it).
//! `fade` scales every mode's premultiplied output, the surrogate's faded display included.
use crate::spheres::CameraUniform;
use gpu::{ColourVolume3D, GpuContext, UniformBuffer, Volume3D};

/// `VolumeUniform::mode`: the density as an emission volume.
pub const MODE_VOLUME: u32 = 0;
/// `VolumeUniform::mode`: the density as an isosurface.
pub const MODE_ISO: u32 = 1;
/// `VolumeUniform::mode`: a signed field as a ± isosurface (surfaces at ±iso·scale, coloured by sign).
pub const MODE_SIGNED_ISO: u32 = 2;
/// `VolumeUniform::mode`: a signed field as a signed volume (opacity from |s|, colour from its sign).
pub const MODE_SIGNED_VOLUME: u32 = 3;
/// `VolumeUniform::mode` for the probability dots (`crate::dots`), drawn instead of the raymarch
/// (never reaches the shader).
pub const MODE_DOTS: u32 = 4;

/// Phase colours (spec §3): + is red, − is blue, the textbook convention (linear RGB).
pub const PHASE_POS: [f32; 3] = [0.90, 0.30, 0.20];
pub const PHASE_NEG: [f32; 3] = [0.20, 0.45, 0.90];
/// Spin colours (spec §3): ↑ amber, ↓ teal — a different hue pair, so phase and spin cannot be
/// confused.
pub const SPIN_UP: [f32; 3] = [0.95, 0.65, 0.15];
pub const SPIN_DOWN: [f32; 3] = [0.15, 0.70, 0.65];

/// Raymarch parameters (96 bytes). Field order is the WGSL `Params` struct's — change both together.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VolumeUniform {
    /// World position (Bohr) of the first grid sample; `box_max` is the last. `mode`: one of the
    /// `MODE_*` constants (the raymarch handles 0–3).
    pub box_min: [f32; 3], pub mode: u32,
    /// Field divisor: value/scale ∈ [0, 1] at the peak (|value|/scale for the signed modes).
    pub box_max: [f32; 3], pub scale: f32,
    /// Power-law exponent; iso threshold (fraction of scale); march step in Bohr.
    pub gamma: f32, pub iso: f32, pub step: f32,
    /// Volume modes: alpha per unit length at value/scale = 1.
    pub opacity: f32,
    /// Multiplies the pass's (premultiplied) output in every mode: 1 = as is; the viewer draws a
    /// density that is not the current step's (a surrogate step has none) at 0.3.
    pub fade: f32,
    /// Crossfade weight of the current field against the previous one bound with
    /// [`VolumePass::bind_pair`] / [`ColourVolumePass::bind_pair`]: 0 = previous, 1 = current
    /// (module doc).
    pub mix: f32,
    /// Volume modes: a soft knee on the opacity, 0 below knee/8 of `scale` rising (smoothstep) to
    /// full at `knee` (0 = off), so a single orbital's low tail fades out instead of swamping its
    /// lobes, without a hard shell.
    pub knee: f32,
    /// Colour volume only: 1 = additive blend, 0 = weighted (emission–absorption).
    pub additive: u32,
    /// Signed modes: the colour of positive and of negative values.
    pub colour_pos: [f32; 3], pub _pad0: f32,
    pub colour_neg: [f32; 3], pub _pad1: f32,
}

const _: () = assert!(std::mem::size_of::<VolumeUniform>() == 96);

impl Default for VolumeUniform {
    /// Empty box (draws nothing until a grid is bound and the box set), otherwise the viewer's defaults.
    fn default() -> Self {
        VolumeUniform {
            box_min: [0.0; 3], mode: MODE_VOLUME, box_max: [0.0; 3], scale: 1.0, gamma: 0.5, iso: 0.3, step: 0.1, opacity: 2.0, fade: 1.0, mix: 1.0, knee: 0.0, additive: 0,
            colour_pos: PHASE_POS, _pad0: 0.0, colour_neg: PHASE_NEG, _pad1: 0.0,
        }
    }
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

/// The shader module both pipelines share: the scalar raymarch (`fs_main`, with the sampling
/// variant's `sample_one`) and the colour volume's (`fs_colour`).
fn shader_source(sampling: Sampling) -> String {
    let variant = if sampling == Sampling::Filtered { include_str!("shaders/volume_filtered.wgsl") } else { include_str!("shaders/volume_manual.wgsl") };
    format!("{}\n{variant}\n{}", include_str!("shaders/volume.wgsl"), include_str!("shaders/volume_colour.wgsl"))
}

/// Group 0 of both pipelines: the camera uniform, the layout of `SpherePass`, so one camera buffer
/// serves every pass.
fn camera_group(ctx: &GpuContext, camera: &UniformBuffer<CameraUniform>) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
    let dev = &ctx.device;
    let bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("volume camera bgl"), entries: &[wgpu::BindGroupLayoutEntry {
        binding: 0, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }] });
    let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("volume camera bg"), layout: &bgl, entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera.buffer.as_entire_binding() }] });
    (bgl, bg)
}

/// The fullscreen pipeline of entry point `fs` over `layout`: premultiplied-over blending onto the
/// colour target, no depth attachment (the scene depth is sampled as a texture instead).
fn fullscreen_pipeline(dev: &wgpu::Device, shader: &wgpu::ShaderModule, layout: &wgpu::PipelineLayout, colour_format: wgpu::TextureFormat, label: &str, fs: &str) -> wgpu::RenderPipeline {
    dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label), layout: Some(layout),
        vertex: wgpu::VertexState { module: shader, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[] },
        fragment: Some(wgpu::FragmentState { module: shader, entry_point: Some(fs), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState {
            format: colour_format,
            // premultiplied over: out = src + (1 - src.a) * dst, for colour and alpha
            blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        })] }),
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
        depth_stencil: None,
        multisample: Default::default(), multiview_mask: None, cache: None,
    })
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }
}

fn depth_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None }
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
        let shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("volume"), source: wgpu::ShaderSource::Wgsl(shader_source(sampling).into()) });
        let frag = wgpu::ShaderStages::FRAGMENT;
        // group 0: same layout as SpherePass, so one camera uniform buffer serves both passes
        let (camera_bgl, camera_bg) = camera_group(ctx, camera);
        // group 1: params, current density (r32float: filterable + a linear sampler at binding 3,
        // or unfilterable and read with textureLoad), scene depth (textureLoad), previous density
        // at binding 4 (same kind as binding 1; module doc)
        let density = wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: filtered }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false };
        let mut entries = vec![
            uniform_entry(0),
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: frag, ty: density, count: None },
            depth_entry(2),
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: frag, ty: density, count: None },
        ];
        if filtered {
            entries.push(wgpu::BindGroupLayoutEntry { binding: 3, visibility: frag, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None });
        }
        let volume_bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("volume bgl"), entries: &entries });
        let layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&camera_bgl), Some(&volume_bgl)], immediate_size: 0 });
        let pipeline = fullscreen_pipeline(dev, &shader, &layout, colour_format, "volume", "fs_main");
        let params = UniformBuffer::<VolumeUniform>::new(ctx, "volume params");
        params.write(ctx, &VolumeUniform::default());
        VolumePass { pipeline, camera_bg, volume_bgl, volume_bg: None, params, sampling }
    }

    pub fn sampling(&self) -> Sampling { self.sampling }

    /// Rebind when the Volume3D or the depth view changes (grid change / resize / a new texture
    /// from the GPU solver). The volume must be unpadded n³ (`Volume3D` contract). Binds it in
    /// both slots: no crossfade whatever `mix` is.
    pub fn bind(&mut self, ctx: &GpuContext, volume: &Volume3D, scene_depth: &wgpu::TextureView) {
        self.bind_pair(ctx, volume, volume, scene_depth);
    }

    /// Binds the previous and the current density for the crossfade (module doc; `mix` picks the
    /// weight). Both are sampled with the same box (`box_min`/`box_max`), so they must be the same
    /// grid: with different dims the current one is bound in both slots (the snap).
    pub fn bind_pair(&mut self, ctx: &GpuContext, previous: &Volume3D, current: &Volume3D, scene_depth: &wgpu::TextureView) {
        let previous = if previous.dims == current.dims { previous } else { current };
        let mut entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: self.params.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&current.view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(scene_depth) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&previous.view) },
        ];
        if self.sampling == Sampling::Filtered {
            entries.push(wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&current.sampler) });
        }
        self.volume_bg = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("volume bg"), layout: &self.volume_bgl, entries: &entries }));
    }

    /// Drops the bound densities: [`draw`](Self::draw) is a no-op again until the next
    /// [`bind`](Self::bind) (the app's empty scene has no density to show).
    pub fn unbind(&mut self) { self.volume_bg = None; }

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

/// The colour volume's pipeline (module doc): "all orbitals coloured". Same camera group and the
/// same [`VolumeUniform`] as [`VolumePass`] (its own buffer: [`set_params`](Self::set_params)),
/// group 1 = params, scene depth, the previous and the current rgba16float [`ColourVolume3D`] and
/// the linear sampler. rgba16float is filterable on every device, so there is one sampling path.
pub struct ColourVolumePass {
    pipeline: wgpu::RenderPipeline,
    camera_bg: wgpu::BindGroup,
    bgl: wgpu::BindGroupLayout,
    bg: Option<wgpu::BindGroup>,
    params: UniformBuffer<VolumeUniform>,
}

impl ColourVolumePass {
    pub fn new(ctx: &GpuContext, colour_format: wgpu::TextureFormat, camera: &UniformBuffer<CameraUniform>) -> ColourVolumePass {
        let dev = &ctx.device;
        // the module's scalar half is unused here, so any sampling variant serves; the manual one
        // needs no extra bindings
        let shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("volume colour"), source: wgpu::ShaderSource::Wgsl(shader_source(Sampling::Manual).into()) });
        let frag = wgpu::ShaderStages::FRAGMENT;
        let (camera_bgl, camera_bg) = camera_group(ctx, camera);
        let tex = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: frag, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false }, count: None };
        let bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("volume colour bgl"), entries: &[
            uniform_entry(0),
            depth_entry(2),
            tex(5),
            wgpu::BindGroupLayoutEntry { binding: 6, visibility: frag, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            tex(7),
        ] });
        let layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&camera_bgl), Some(&bgl)], immediate_size: 0 });
        let pipeline = fullscreen_pipeline(dev, &shader, &layout, colour_format, "volume colour", "fs_colour");
        let params = UniformBuffer::<VolumeUniform>::new(ctx, "volume colour params");
        params.write(ctx, &VolumeUniform::default());
        ColourVolumePass { pipeline, camera_bg, bgl, bg: None, params }
    }

    /// Binds the previous and the current colour volume (and the scene depth). Both are sampled
    /// with the same box, so they must be the same grid: with different dims the current one is
    /// bound in both slots (the snap).
    pub fn bind_pair(&mut self, ctx: &GpuContext, previous: &ColourVolume3D, current: &ColourVolume3D, scene_depth: &wgpu::TextureView) {
        let previous = if previous.dims == current.dims { previous } else { current };
        self.bg = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("volume colour bg"), layout: &self.bgl, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: self.params.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(scene_depth) },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&current.view) },
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&current.sampler) },
            wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&previous.view) },
        ] }));
    }

    /// Drops the bound volumes: [`draw`](Self::draw) is a no-op until the next
    /// [`bind_pair`](Self::bind_pair).
    pub fn unbind(&mut self) { self.bg = None; }

    pub fn set_params(&self, ctx: &GpuContext, params: &VolumeUniform) { self.params.write(ctx, params); }

    /// As [`VolumePass::draw`]. No-op until [`ColourVolumePass::bind_pair`].
    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        let Some(bg) = &self.bg else { return };
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
    use crate::camera::{OrbitCamera, Viewport};
    use crate::headless::{Headless, HEADLESS_FORMAT};
    use crate::spheres::{CameraUniform, SphereInstance};
    use crate::Renderer;
    use gpu::{UniformBuffer, Volume3D};
    use glam::Vec3;

    /// Gaussian density of width 1 Bohr centred at `centre` on a 33³ grid spanning [−4, 4]³.
    fn gaussian_data(centre: [f32; 3]) -> Vec<f32> {
        let n = 33u32; let h = 0.25f32; let lo = -4.0f32;
        let mut data = Vec::with_capacity((n * n * n) as usize);
        for k in 0..n { for j in 0..n { for i in 0..n {
            let (x, y, z) = (lo + i as f32 * h, lo + j as f32 * h, lo + k as f32 * h);
            let r2 = (x - centre[0]).powi(2) + (y - centre[1]).powi(2) + (z - centre[2]).powi(2);
            data.push((-r2 / 2.0).exp());
        }}}
        data
    }

    /// A 33³ volume holding `data`, and its box.
    fn volume_of(ctx: &gpu::GpuContext, data: &[f32]) -> (Volume3D, [f32; 3], [f32; 3]) {
        let vol = Volume3D::new(ctx, [33; 3]);
        vol.upload(ctx, data);
        (vol, [-4.0; 3], [4.0; 3])
    }

    fn gaussian_volume(ctx: &gpu::GpuContext, centre: [f32; 3]) -> (Volume3D, [f32; 3], [f32; 3]) {
        volume_of(ctx, &gaussian_data(centre))
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
        pass.set_params(&hl.ctx, &VolumeUniform { box_min: bmin, mode, box_max: bmax, scale: 1.0, gamma: 0.5, iso: 0.3, step: 0.1, opacity: 2.0, ..Default::default() });
        let mut enc = hl.ctx.device.create_command_encoder(&Default::default());
        // The depth texture is sampled by the volume pass, so it cannot also be the attachment of the pass
        // that draws the volume: clear colour + depth first, then march in a colour-only overlay pass.
        drop(hl.begin(&mut enc, wgpu::Color::BLACK));
        { let mut rp = overlay_pass(&mut enc, &hl.colour_view); pass.draw(&mut rp); }
        hl.ctx.queue.submit([enc.finish()]);
        hl.read_rgba().expect("readback")
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

    /// `fade` scales the pass's premultiplied output in both modes (the viewer's faded density on
    /// surrogate steps): over black the centre pixel is the faded fraction of the full one, within
    /// 8-bit rounding.
    #[test]
    fn fade_scales_the_output_in_both_modes() {
        let (w, h) = (64u32, 64u32);
        let Some(hl) = Headless::new(w, h) else { return };
        let cam = OrbitCamera::new(Vec3::ZERO, 12.0);
        let ub = UniformBuffer::<CameraUniform>::new(&hl.ctx, "camera");
        ub.write(&hl.ctx, &CameraUniform::from_camera(&cam, w as f32, h as f32));
        let (vol, bmin, bmax) = gaussian_volume(&hl.ctx, [0.0; 3]);
        let mut pass = VolumePass::new(&hl.ctx, HEADLESS_FORMAT, &ub);
        pass.bind(&hl.ctx, &vol, &hl.depth);
        let centre = |mode: u32, fade: f32| {
            pass.set_params(&hl.ctx, &VolumeUniform { box_min: bmin, mode, box_max: bmax, fade, ..Default::default() });
            let mut enc = hl.ctx.device.create_command_encoder(&Default::default());
            drop(hl.begin(&mut enc, wgpu::Color::BLACK));
            { let mut rp = overlay_pass(&mut enc, &hl.colour_view); pass.draw(&mut rp); }
            hl.ctx.queue.submit([enc.finish()]);
            hl.read_rgba().expect("readback")[(h / 2 * w + w / 2) as usize]
        };
        for mode in [0u32, 1] {
            let (full, faded) = (centre(mode, 1.0), centre(mode, 0.3));
            eprintln!("mode {mode}: centre {full:?} → faded {faded:?}");
            assert!(full[0] as u32 + full[1] as u32 + full[2] as u32 > 100, "mode {mode}: vacuous ({full:?})");
            for c in 0..3 {
                let want = 0.3 * full[c] as f32;
                assert!((faded[c] as f32 - want).abs() <= 1.5, "mode {mode} channel {c}: {} vs 0.3 × {}", faded[c], full[c]);
            }
        }
    }

    /// One frame of a `VolumePass` over black with `previous`/`current` bound as a pair at `mix`.
    fn frame_pair(hl: &Headless, sampling: Sampling, previous: &Volume3D, current: &Volume3D, mode: u32, mix: f32) -> Vec<[u8; 4]> {
        let (w, h) = (hl.width, hl.height);
        let cam = OrbitCamera::new(Vec3::ZERO, 12.0);
        let ub = UniformBuffer::<CameraUniform>::new(&hl.ctx, "camera");
        ub.write(&hl.ctx, &CameraUniform::from_camera(&cam, w as f32, h as f32));
        let mut pass = VolumePass::with_sampling(&hl.ctx, HEADLESS_FORMAT, &ub, sampling);
        pass.bind_pair(&hl.ctx, previous, current, &hl.depth);
        pass.set_params(&hl.ctx, &VolumeUniform { box_min: [-4.0; 3], mode, box_max: [4.0; 3], mix, ..Default::default() });
        let mut enc = hl.ctx.device.create_command_encoder(&Default::default());
        drop(hl.begin(&mut enc, wgpu::Color::BLACK));
        { let mut rp = overlay_pass(&mut enc, &hl.colour_view); pass.draw(&mut rp); }
        hl.ctx.queue.submit([enc.finish()]);
        hl.read_rgba().expect("readback")
    }

    fn max_channel_diff(a: &[[u8; 4]], b: &[[u8; 4]]) -> i32 {
        a.iter().zip(b).map(|(p, q)| (0..4).map(|c| (p[c] as i32 - q[c] as i32).abs()).max().unwrap()).max().unwrap()
    }

    /// The density crossfade (module doc): with two Gaussians a and b bound as previous/current,
    /// mix 0 draws a alone and mix 1 b alone (bit for bit), and mix 0.5 draws the grid ½(a + b)
    /// bound on its own — the blend of two samples is the sample of the blended density — in
    /// both modes and on both sampling paths (the hardware-filtered one when the device has it).
    #[test]
    fn crossfade_draws_the_blended_density() {
        let (w, h) = (96u32, 96u32);
        let Some(hl) = Headless::new(w, h) else { return };
        let (da, db) = (gaussian_data([-1.2, 0.3, 0.0]), gaussian_data([1.0, -0.4, 0.2]));
        let avg: Vec<f32> = da.iter().zip(&db).map(|(x, y)| 0.5 * (x + y)).collect();
        let (a, _, _) = volume_of(&hl.ctx, &da);
        let (b, _, _) = volume_of(&hl.ctx, &db);
        let (m, _, _) = volume_of(&hl.ctx, &avg);
        let mut paths = vec![Sampling::Manual];
        if hl.ctx.float32_filterable {
            paths.push(Sampling::Filtered);
        }
        for sampling in paths {
            for mode in [0u32, 1] {
                let only_a = frame_pair(&hl, sampling, &a, &a, mode, 1.0);
                let only_b = frame_pair(&hl, sampling, &b, &b, mode, 1.0);
                let only_m = frame_pair(&hl, sampling, &m, &m, mode, 1.0);
                assert!(max_channel_diff(&only_a, &only_b) > 100, "{sampling:?} mode {mode}: vacuous (a and b look alike)");
                assert_eq!(frame_pair(&hl, sampling, &a, &b, mode, 0.0), only_a, "{sampling:?} mode {mode}: mix 0 is the previous density");
                assert_eq!(frame_pair(&hl, sampling, &a, &b, mode, 1.0), only_b, "{sampling:?} mode {mode}: mix 1 is the current density");
                let half = frame_pair(&hl, sampling, &a, &b, mode, 0.5);
                let d = max_channel_diff(&half, &only_m);
                let centre = half[(h / 2 * w + w / 2) as usize];
                eprintln!("{sampling:?} mode {mode}: mix 0.5 vs ½(a + b) max channel diff {d}/255; centre {centre:?} (a {:?}, b {:?})",
                          only_a[(h / 2 * w + w / 2) as usize], only_b[(h / 2 * w + w / 2) as usize]);
                // equal up to f32 rounding of the two sums (measured 0/255 on the RX 7900 XTX)
                assert!(d <= 1, "{sampling:?} mode {mode}: mix 0.5 differs from ½(a + b) by {d}/255");
            }
        }
    }

    /// With different dims the pair snaps to the current density (it is bound in both slots).
    #[test]
    fn a_pair_of_different_grids_snaps_to_the_current() {
        let (w, h) = (64u32, 64u32);
        let Some(hl) = Headless::new(w, h) else { return };
        let (b, _, _) = gaussian_volume(&hl.ctx, [0.5, 0.0, 0.0]);
        let small = Volume3D::new(&hl.ctx, [17; 3]);
        small.upload(&hl.ctx, &vec![1.0; 17 * 17 * 17]);
        let only_b = frame_pair(&hl, Sampling::Manual, &b, &b, 0, 1.0);
        assert_eq!(frame_pair(&hl, Sampling::Manual, &small, &b, 0, 0.0), only_b);
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
        r.volume_params = VolumeUniform { box_min: bmin, mode: 0, box_max: bmax, scale: 1.0, gamma: 0.5, iso: 0.3, step: 0.1, opacity: 2.0, ..Default::default() };
        r.sync_volume_params(&hl.ctx);
        let sphere = SphereInstance { centre: (cam.eye().normalize() * along_view).to_array(), radius: 1.0, colour: [0.2, 0.9, 0.2], selected: 0.0 };
        let frame = |r: &Renderer| {
            let mut enc = hl.ctx.device.create_command_encoder(&Default::default());
            r.render(&mut enc, &hl.colour_view, &hl.depth, wgpu::Color::BLACK);
            hl.ctx.queue.submit([enc.finish()]);
            hl.read_rgba().expect("readback")[(h / 2 * w + w / 2) as usize]
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

    /// A scene confined to an offset viewport (the viewer's central rect): the sphere at the camera
    /// target is drawn at the viewport's centre, the sphere + volume composite there matches the
    /// full-target render's centre pixel (the shaders' pixel → NDC follows the origin, and the volume
    /// reads the sphere depth at the right pixel), and nothing is drawn outside the rect.
    #[test]
    fn a_scene_in_an_offset_viewport_lands_at_its_centre_and_nowhere_else() {
        let (w, h) = (160u32, 120u32);
        let Some(hl) = Headless::new(w, h) else { return };
        let cam = OrbitCamera::new(Vec3::ZERO, 12.0);
        let mut r = Renderer::new(&hl.ctx, HEADLESS_FORMAT);
        let (vol, bmin, bmax) = gaussian_volume(&hl.ctx, [0.0; 3]);
        r.volume_params = VolumeUniform { box_min: bmin, mode: 0, box_max: bmax, scale: 1.0, gamma: 0.5, iso: 0.3, step: 0.1, opacity: 2.0, ..Default::default() };
        r.sync_volume_params(&hl.ctx);
        r.volume.bind(&hl.ctx, &vol, &hl.depth);
        let sphere = SphereInstance { centre: (cam.eye().normalize() * 4.0).to_array(), radius: 1.0, colour: [0.2, 0.9, 0.2], selected: 0.0 };
        r.spheres.set_instances(&hl.ctx, &[sphere]);
        let shot = |r: &Renderer, vp: Option<Viewport>| {
            match vp {
                Some(v) => r.update_camera_in(&hl.ctx, &cam, v),
                None => r.update_camera(&hl.ctx, &cam, w as f32, h as f32),
            }
            let mut enc = hl.ctx.device.create_command_encoder(&Default::default());
            r.render_in(&mut enc, &hl.colour_view, &hl.depth, wgpu::Color::BLACK, vp.map(|v| (v, [w, h])));
            hl.ctx.queue.submit([enc.finish()]);
            hl.read_rgba().expect("readback")
        };
        // a square viewport the same size as the full target's height keeps the sphere's pixel size
        let full_sq = Viewport { origin: [20.0, 0.0], size: [120.0, 120.0] };
        let vp = Viewport { origin: [32.0, 8.0], size: [120.0, 112.0] };
        let a = shot(&r, Some(full_sq));
        let b = shot(&r, Some(vp));
        let at = |img: &[[u8; 4]], x: u32, y: u32| img[(y * w + x) as usize];
        let (cx, cy) = vp.centre();
        let centre = at(&b, cx as u32, cy as u32);
        let reference = at(&a, 80, 60);
        eprintln!("offset viewport centre {centre:?}, reference {reference:?}");
        assert!(centre[1] > 100, "no sphere at the viewport's centre: {centre:?}");
        assert!(diff(centre, reference).iter().all(|d| d.abs() <= 3), "{centre:?} vs {reference:?}");
        for (x, y) in [(10, 60), (5, 5), (31, 100), (80, 4)] {
            assert_eq!(at(&b, x, y)[..3], [0, 0, 0], "drawn outside the viewport at ({x}, {y})");
        }
    }
}

/// The orbital viewer's modes (Task 6): the signed isosurface and volume, the colour-volume
/// pipeline with its two blends, the colour crossfade, `fade` in every mode, and the frame time of
/// each mode at 97³ (reported, never asserted).
#[cfg(test)]
mod orbital_tests {
    use super::*;
    use crate::camera::{OrbitCamera, Viewport};
    use crate::headless::{Headless, HEADLESS_FORMAT};
    use crate::spheres::CameraUniform;
    use crate::Renderer;
    use glam::Vec3;
    use gpu::{ColourVolume3D, UniformBuffer, Volume3D};

    const N: u32 = 33;
    const LO: f32 = -4.0;

    /// `f` at every point of the N³ grid over [−4, 4]³ (x fastest, the volumes' layout).
    fn grid<T>(n: u32, f: impl Fn(Vec3) -> T) -> Vec<T> {
        let h = -2.0 * LO / (n - 1) as f32;
        let mut out = Vec::with_capacity((n * n * n) as usize);
        for k in 0..n { for j in 0..n { for i in 0..n {
            out.push(f(Vec3::new(LO + i as f32 * h, LO + j as f32 * h, LO + k as f32 * h)));
        }}}
        out
    }

    /// A p_x-like orbital: x e^{−r²/2}, positive on +x, negative on −x, a node on x = 0; its
    /// extrema ±e^{−1/2} sit at x = ±1.
    fn px(n: u32) -> Vec<f32> { grid(n, |p| p.x * (-p.length_squared() / 2.0).exp()) }
    const PX_MAX: f32 = 0.606_530_66;

    fn scalar_volume(hl: &Headless, n: u32, data: &[f32]) -> Volume3D {
        let v = Volume3D::new(&hl.ctx, [n; 3]);
        v.upload(&hl.ctx, data);
        v
    }

    fn colour_volume(hl: &Headless, n: u32, data: &[[f32; 4]]) -> ColourVolume3D {
        let v = ColourVolume3D::new(&hl.ctx, [n; 3]);
        v.upload_f32(&hl.ctx, data);
        v
    }

    /// Looking down −z: +x is screen-right, so a lobe's pixel is its projected position.
    fn camera() -> OrbitCamera {
        let mut c = OrbitCamera::new(Vec3::ZERO, 12.0);
        c.yaw = 0.0;
        c.pitch = 0.0;
        c
    }

    fn camera_buffer(hl: &Headless) -> UniformBuffer<CameraUniform> {
        let ub = UniformBuffer::<CameraUniform>::new(&hl.ctx, "camera");
        ub.write(&hl.ctx, &CameraUniform::from_camera(&camera(), hl.width as f32, hl.height as f32));
        ub
    }

    fn params(mode: u32) -> VolumeUniform {
        VolumeUniform { box_min: [LO; 3], box_max: [-LO; 3], mode, ..Default::default() }
    }

    /// The pixel a world point projects to in a frame of `hl`.
    fn pixel_of(hl: &Headless, p: Vec3) -> (u32, u32) {
        let (x, y) = Viewport::full(hl.width as f32, hl.height as f32).project(&camera(), p).expect("in front of the camera");
        (x as u32, y as u32)
    }

    fn at(hl: &Headless, frame: &[[u8; 4]], p: Vec3) -> [u8; 4] {
        let (x, y) = pixel_of(hl, p);
        frame[(y * hl.width + x) as usize]
    }

    /// One frame over black: the scalar pass over (previous, current) with `p`.
    fn scalar_frame(hl: &Headless, sampling: Sampling, previous: &Volume3D, current: &Volume3D, p: VolumeUniform) -> Vec<[u8; 4]> {
        let ub = camera_buffer(hl);
        let mut pass = VolumePass::with_sampling(&hl.ctx, HEADLESS_FORMAT, &ub, sampling);
        pass.bind_pair(&hl.ctx, previous, current, &hl.depth);
        pass.set_params(&hl.ctx, &p);
        let mut enc = hl.ctx.device.create_command_encoder(&Default::default());
        drop(hl.begin(&mut enc, wgpu::Color::BLACK));
        { let mut rp = overlay_pass(&mut enc, &hl.colour_view); pass.draw(&mut rp); }
        hl.ctx.queue.submit([enc.finish()]);
        hl.read_rgba().expect("readback")
    }

    /// One frame over black: the colour pass over (previous, current) with `p`.
    fn colour_frame(hl: &Headless, previous: &ColourVolume3D, current: &ColourVolume3D, p: VolumeUniform) -> Vec<[u8; 4]> {
        let ub = camera_buffer(hl);
        let mut pass = ColourVolumePass::new(&hl.ctx, HEADLESS_FORMAT, &ub);
        pass.bind_pair(&hl.ctx, previous, current, &hl.depth);
        pass.set_params(&hl.ctx, &p);
        let mut enc = hl.ctx.device.create_command_encoder(&Default::default());
        drop(hl.begin(&mut enc, wgpu::Color::BLACK));
        { let mut rp = overlay_pass(&mut enc, &hl.colour_view); pass.draw(&mut rp); }
        hl.ctx.queue.submit([enc.finish()]);
        hl.read_rgba().expect("readback")
    }

    fn max_diff(a: &[[u8; 4]], b: &[[u8; 4]]) -> i32 {
        a.iter().zip(b).map(|(p, q)| (0..4).map(|c| (p[c] as i32 - q[c] as i32).abs()).max().unwrap()).max().unwrap()
    }

    /// The phase isosurface of a p_x: the +x lobe is red, the −x lobe blue (the textbook
    /// convention), and the hardware-filtered and manual sampling paths draw the same picture in
    /// both signed modes.
    #[test]
    fn phase_isosurface_colours_lobes_by_sign() {
        let Some(hl) = Headless::new(128, 128) else { return };
        let v = scalar_volume(&hl, N, &px(N));
        let p = VolumeUniform { scale: PX_MAX, iso: 0.15, ..params(MODE_SIGNED_ISO) };
        let frame = scalar_frame(&hl, Sampling::best(&hl.ctx), &v, &v, p);
        let (pos, neg) = (at(&hl, &frame, Vec3::new(1.0, 0.0, 0.0)), at(&hl, &frame, Vec3::new(-1.0, 0.0, 0.0)));
        eprintln!("signed iso: +x lobe {pos:?}, −x lobe {neg:?}");
        assert!(pos[0] as i32 - pos[2] as i32 > 60, "the +x lobe must be red: {pos:?}");
        assert!(neg[2] as i32 - neg[0] as i32 > 60, "the −x lobe must be blue: {neg:?}");
        assert_eq!((pos[3], neg[3]), (255, 255), "the surfaces are opaque");
        if hl.ctx.float32_filterable {
            for mode in [MODE_SIGNED_ISO, MODE_SIGNED_VOLUME] {
                let p = VolumeUniform { scale: PX_MAX, iso: 0.15, ..params(mode) };
                let a = scalar_frame(&hl, Sampling::Filtered, &v, &v, p);
                let b = scalar_frame(&hl, Sampling::Manual, &v, &v, p);
                let d = max_diff(&a, &b);
                let bad: Vec<(usize, i32)> = a.iter().zip(&b).enumerate().map(|(i, (p, q))| (i, (0..4).map(|c| (p[c] as i32 - q[c] as i32).abs()).max().unwrap())).filter(|(_, d)| *d > 8).collect();
                eprintln!("mode {mode}: filtered vs manual max channel diff {d}/255; {} pixels over 8: first {:?}", bad.len(), bad.iter().take(8).map(|(i, d)| ((*i as u32 % hl.width, *i as u32 / hl.width), *d)).collect::<Vec<_>>());
                // Hardware filtering quantises the interpolation weights (8 bits), so a ray grazing the
                // iso threshold can hit one march step earlier or later: a handful of pixels on the
                // lobe's rim (measured: 4 of 16384, in a symmetric ring) flip between two lit values.
                // Everywhere else the paths agree to 8/255, as for the density modes.
                assert!(bad.len() <= 8, "mode {mode}: {} pixels differ by more than 8/255", bad.len());
                assert!(bad.iter().all(|&(_, d)| d < 64), "mode {mode}: a rim pixel differs by {d}/255");
            }
        }
        // the signed volume colours the same lobes the same way
        let p = VolumeUniform { scale: PX_MAX, ..params(MODE_SIGNED_VOLUME) };
        let frame = scalar_frame(&hl, Sampling::best(&hl.ctx), &v, &v, p);
        let (pos, neg) = (at(&hl, &frame, Vec3::new(1.0, 0.0, 0.0)), at(&hl, &frame, Vec3::new(-1.0, 0.0, 0.0)));
        eprintln!("signed volume: +x lobe {pos:?}, −x lobe {neg:?}");
        assert!(pos[0] as i32 - pos[2] as i32 > 60 && neg[2] as i32 - neg[0] as i32 > 60, "{pos:?} {neg:?}");
    }

    /// A field of zeros draws nothing in the signed modes and in the colour pass: every pixel is
    /// exactly the clear colour (no NaN from pow(0), no ambient wash).
    #[test]
    fn signed_volume_of_zero_draws_nothing() {
        let Some(hl) = Headless::new(96, 96) else { return };
        let zero = scalar_volume(&hl, N, &vec![0.0; (N * N * N) as usize]);
        let clear = vec![[0u8, 0, 0, 255]; (hl.width * hl.height) as usize];
        for mode in [MODE_SIGNED_ISO, MODE_SIGNED_VOLUME] {
            let frame = scalar_frame(&hl, Sampling::best(&hl.ctx), &zero, &zero, VolumeUniform { scale: 1.0, ..params(mode) });
            assert_eq!(frame, clear, "mode {mode}");
        }
        let czero = colour_volume(&hl, N, &vec![[0.0; 4]; (N * N * N) as usize]);
        for additive in [0, 1] {
            let frame = colour_frame(&hl, &czero, &czero, VolumeUniform { additive, ..params(MODE_VOLUME) });
            assert_eq!(frame, clear, "colour, additive {additive}");
        }
    }

    /// An inner Gaussian (σ 0.5, colour A = red) inside an outer shell (r = 3 ± 0.5, colour B = blue):
    /// the ray through the centre crosses the shell, the inner blob and the shell again. Additive
    /// (emission only) shows A's hue; weighted (front-to-back absorption) hides it behind the
    /// front shell.
    #[test]
    fn additive_colour_shows_the_inner_shell() {
        let Some(hl) = Headless::new(96, 96) else { return };
        let (ca, cb) = ([1.0f32, 0.1, 0.1], [0.1f32, 0.1, 1.0]);
        let data = grid(N, |p| {
            let r = p.length();
            let sa = (-r * r / (2.0 * 0.25)).exp();
            let sb = (-(r - 3.0) * (r - 3.0) / (2.0 * 0.25)).exp();
            [ca[0] * sa + cb[0] * sb, ca[1] * sa + cb[1] * sb, ca[2] * sa + cb[2] * sb, sa + sb]
        });
        let v = colour_volume(&hl, N, &data);
        let centre = |additive| {
            let frame = colour_frame(&hl, &v, &v, VolumeUniform { additive, ..params(MODE_VOLUME) });
            at(&hl, &frame, Vec3::ZERO)
        };
        let (add, weighted) = (centre(1), centre(0));
        eprintln!("colour centre pixel: additive {add:?}, weighted {weighted:?}");
        assert!(add[0] as f32 > 0.3 * add[2] as f32, "additive must show the inner shell's red: {add:?}");
        assert!(weighted[2] as f32 > 3.0 * weighted[0] as f32, "weighted: the front shell (blue) dominates: {weighted:?}");
        assert!(weighted[0] < add[0], "the front shell hides the inner one: {weighted:?} vs {add:?}");
    }

    /// The colour crossfade is linear: mix ½ of two colour volumes draws the same frame as their
    /// average bound on its own (equal up to f16 rounding), in both blends.
    #[test]
    fn colour_crossfade_is_linear() {
        let Some(hl) = Headless::new(96, 96) else { return };
        let blob = |c: Vec3, col: [f32; 3]| {
            grid(N, move |p| {
                let s = (-(p - c).length_squared() / 2.0).exp();
                [col[0] * s, col[1] * s, col[2] * s, s]
            })
        };
        let (da, db) = (blob(Vec3::new(-1.2, 0.3, 0.0), [0.9, 0.2, 0.1]), blob(Vec3::new(1.0, -0.4, 0.2), [0.1, 0.3, 0.9]));
        let avg: Vec<[f32; 4]> = da.iter().zip(&db).map(|(x, y)| std::array::from_fn(|c| 0.5 * (x[c] + y[c]))).collect();
        let (a, b, m) = (colour_volume(&hl, N, &da), colour_volume(&hl, N, &db), colour_volume(&hl, N, &avg));
        for additive in [0, 1] {
            let base = VolumeUniform { additive, ..params(MODE_VOLUME) };
            let only_a = colour_frame(&hl, &a, &a, base);
            let only_b = colour_frame(&hl, &b, &b, base);
            assert!(max_diff(&only_a, &only_b) > 100, "additive {additive}: vacuous (a and b look alike)");
            assert_eq!(colour_frame(&hl, &a, &b, VolumeUniform { mix: 0.0, ..base }), only_a, "mix 0 is the previous volume");
            assert_eq!(colour_frame(&hl, &a, &b, VolumeUniform { mix: 1.0, ..base }), only_b, "mix 1 is the current volume");
            let half = colour_frame(&hl, &a, &b, VolumeUniform { mix: 0.5, ..base });
            let want = colour_frame(&hl, &m, &m, base);
            let d = max_diff(&half, &want);
            eprintln!("additive {additive}: mix 0.5 vs the average volume, max channel diff {d}/255");
            assert!(d <= 3, "additive {additive}: mix 0.5 differs from the average by {d}/255");
        }
    }

    /// `fade` scales the premultiplied output of every mode: at 0.3 a lobe's pixel is 0.3 of the
    /// full one (8-bit rounding).
    #[test]
    fn fade_scales_every_mode() {
        let Some(hl) = Headless::new(96, 96) else { return };
        let v = scalar_volume(&hl, N, &px(N));
        let blob = grid(N, |p| { let s = (-(p - Vec3::new(1.0, 0.0, 0.0)).length_squared() / 2.0).exp(); [0.9 * s, 0.4 * s, 0.1 * s, s] });
        let cv = colour_volume(&hl, N, &blob);
        let lobe = Vec3::new(1.0, 0.0, 0.0);
        let check = |what: &str, full: [u8; 4], faded: [u8; 4]| {
            eprintln!("{what}: {full:?} → faded {faded:?}");
            assert!(full[0] as u32 + full[1] as u32 + full[2] as u32 > 100, "{what}: vacuous ({full:?})");
            // (alpha over an opaque clear colour stays 255, so only the colour is scaled)
            for c in 0..3 {
                let want = 0.3 * full[c] as f32;
                assert!((faded[c] as f32 - want).abs() <= 2.0, "{what} channel {c}: {} vs 0.3 × {}", faded[c], full[c]);
            }
        };
        for mode in [MODE_SIGNED_ISO, MODE_SIGNED_VOLUME] {
            let p = VolumeUniform { scale: PX_MAX, ..params(mode) };
            let full = at(&hl, &scalar_frame(&hl, Sampling::best(&hl.ctx), &v, &v, p), lobe);
            let faded = at(&hl, &scalar_frame(&hl, Sampling::best(&hl.ctx), &v, &v, VolumeUniform { fade: 0.3, ..p }), lobe);
            check(&format!("mode {mode}"), full, faded);
        }
        for additive in [0, 1] {
            let p = VolumeUniform { additive, ..params(MODE_VOLUME) };
            let full = at(&hl, &colour_frame(&hl, &cv, &cv, p), lobe);
            let faded = at(&hl, &colour_frame(&hl, &cv, &cv, VolumeUniform { fade: 0.3, ..p }), lobe);
            check(&format!("colour, additive {additive}"), full, faded);
        }
    }

    /// Frame time of each mode at 97³ in a 1400 × 900 frame (headless; REPORTED, not asserted):
    /// the mean of 20 submitted-and-waited frames after 3 warm-ups.
    #[test]
    fn frame_times_at_97_cubed() {
        let Some(hl) = Headless::new(1400, 900) else { return };
        const M: u32 = 97;
        let sdata = px(M);
        let sv = scalar_volume(&hl, M, &sdata);
        let cdata = grid(M, |p| {
            let r = p.length();
            let (sa, sb) = ((-r * r / 0.5).exp(), (-(r - 3.0) * (r - 3.0) / 0.5).exp());
            [sa + 0.1 * sb, 0.1 * sa + 0.1 * sb, 0.1 * sa + sb, sa + sb]
        });
        let cv = colour_volume(&hl, M, &cdata);
        let cam = camera();
        let mut r = Renderer::new(&hl.ctx, HEADLESS_FORMAT);
        r.update_camera(&hl.ctx, &cam, hl.width as f32, hl.height as f32);
        r.volume.bind(&hl.ctx, &sv, &hl.depth);
        r.colour.bind_pair(&hl.ctx, &cv, &cv, &hl.depth);
        r.dots.bind(&hl.ctx, &sv, None, &hl.depth);
        let h = 8.0 / (M - 1) as f32;
        let base = VolumeUniform { scale: PX_MAX, step: (h / 2.0).min(0.1), ..params(MODE_VOLUME) };
        let rows: [(&str, VolumeUniform, bool); 8] = [
            ("volume (density)", VolumeUniform { scale: 0.4, ..base }, false),
            ("isosurface (density)", VolumeUniform { scale: 0.4, mode: MODE_ISO, ..base }, false),
            ("signed isosurface", VolumeUniform { mode: MODE_SIGNED_ISO, iso: 0.15, ..base }, false),
            ("signed volume", VolumeUniform { mode: MODE_SIGNED_VOLUME, knee: 0.08, ..base }, false),
            ("colour, weighted", VolumeUniform { scale: 1.0, additive: 0, ..base }, true),
            ("colour, additive", VolumeUniform { scale: 1.0, additive: 1, ..base }, true),
            ("dots (200k)", VolumeUniform { mode: MODE_DOTS, ..base }, false),
            ("dots, signed (200k)", VolumeUniform { mode: MODE_DOTS, ..base }, false),
        ];
        eprintln!("frame times, 97³ in 1400×900 (headless, mean of 20):");
        for (name, p, colour) in rows {
            r.volume_params = p;
            r.draw_colour = colour;
            r.sync_volume_params(&hl.ctx);
            if p.mode == MODE_DOTS {
                let signed = name.contains("signed");
                r.dots.set_params(&hl.ctx, &crate::dots::DotsUniform {
                    box_min: [LO; 3], use_colour: 0, box_max: [-LO; 3], count: 200_000, dims: [0; 3], seed: 7,
                    tint: [0.9, 0.3, 0.2, 1.0], tint_neg: [0.2, 0.45, 0.9, 1.0], size_px: 1.25, intensity: 0.35,
                    colour_by: if signed { crate::dots::COLOUR_BY_SIGN } else { crate::dots::COLOUR_BY_DENSITY }, signed: u32::from(signed),
                });
            }
            let frame = || {
                let mut enc = hl.ctx.device.create_command_encoder(&Default::default());
                r.render(&mut enc, &hl.colour_view, &hl.depth, wgpu::Color::BLACK);
                let t = std::time::Instant::now();
                hl.ctx.queue.submit([enc.finish()]);
                hl.ctx.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
                t.elapsed().as_secs_f64() * 1e3
            };
            for _ in 0..3 { frame(); }
            let ms = (0..20).map(|_| frame()).sum::<f64>() / 20.0;
            eprintln!("  {name:<22} {ms:6.2} ms");
        }
    }
}
