//! "Probability dots": the displayed field —
//! ρ, an orbital's |ψ|², or the colour volume's a — drawn as random samples, each a small round
//! sprite composited "over" with a per-dot opacity (`intensity`). Generated on the GPU every frame
//! from the bound display texture (no field data is read back). The sampled density is the
//! field's trilinear interpolation, drawn exactly (cell by a row CDF and a walk along the row,
//! then corner by weight and a linear draw per axis: `shaders/dots_compute.wgsl`), so it is
//! continuous across cells — no voxel steps or lattice planes. A dot is coloured by the volume
//! mode's ramp of √(f/max) at its position ([`COLOUR_BY_DENSITY`]), by the orbitals' mix colour
//! ([`COLOUR_BY_ORBITAL`], colour volume only) or by `tint`. Resampled every frame with a new
//! seed (shimmer) or kept with a fixed one.
use crate::spheres::CameraUniform;
use gpu::{ColourVolume3D, GpuContext, UniformBuffer, Volume3D};

/// Most dots per frame (the points buffer's size).
pub const MAX_DOTS: u32 = 1_000_000;

/// Field order is the WGSL `Params` (both shaders) — change all together.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DotsUniform {
    pub box_min: [f32; 3], pub use_colour: u32,
    pub box_max: [f32; 3], pub count: u32,
    pub dims: [u32; 3], pub seed: u32,
    /// [`COLOUR_BY_TINT`]'s colour, and for [`COLOUR_BY_SIGN`] the colour of positive values
    /// (`tint_neg`: of negative ones).
    pub tint: [f32; 4],
    pub tint_neg: [f32; 4],
    /// Sprite half-size in pixels; a dot's opacity at its centre.
    pub size_px: f32, pub intensity: f32,
    /// [`COLOUR_BY_DENSITY`], [`COLOUR_BY_ORBITAL`] (the colour volume's mix), [`COLOUR_BY_TINT`]
    /// or [`COLOUR_BY_SIGN`].
    pub colour_by: u32,
    /// 1 = the field bound is signed (the spin density): dots are sampled in proportion to |field|
    /// (an orbital is drawn as its square instead, which is smooth through its nodes). 0 = a
    /// non-negative field (negative noise clamped to 0).
    pub signed: u32,
    /// The floating dots (user request 2026-10-19): a Metropolis step of `walk` grid cells per
    /// frame for each live dot (0 = a fresh sample every time), `refresh` = the share re-drawn
    /// fresh each frame, `glow` = soft sprites added (the glow pipeline).
    pub walk: f32,
    pub refresh: f32,
    pub glow: u32,
    /// Feather (user request 2026-10-19): dots where √(field / max) is below `cut` fade out (over
    /// 0.4 cut … cut); 0 = every dot shown.
    pub cut: f32,
}

/// Dot colour: the volume mode's ramp of the field at the dot (log over 3 decades of its max).
pub const COLOUR_BY_DENSITY: u32 = 0;
/// Dot colour: the orbitals' mix colour (colour volume only; else the density ramp).
pub const COLOUR_BY_ORBITAL: u32 = 1;
/// Dot colour: `tint`.
pub const COLOUR_BY_TINT: u32 = 2;
/// Dot colour: `tint` where the (scalar) field is >= 0 at the dot's nearest grid point,
/// `tint_neg` where it is negative.
pub const COLOUR_BY_SIGN: u32 = 3;

const _: () = assert!(std::mem::size_of::<DotsUniform>() == 112);

struct Bound { compute_bg: wgpu::BindGroup, draw_bg: wgpu::BindGroup, n_rows: u32, dims: [u32; 3], colour: bool }

pub struct DotsPass {
    row_sums: wgpu::ComputePipeline,
    row_scan: wgpu::ComputePipeline,
    gen_points: wgpu::ComputePipeline,
    draw: wgpu::RenderPipeline,
    /// The same sprites added (One, One): the glowing dots.
    draw_glow: wgpu::RenderPipeline,
    glow: bool,
    compute_bgl: wgpu::BindGroupLayout,
    draw_bgl: wgpu::BindGroupLayout,
    camera_bg: wgpu::BindGroup,
    params: UniformBuffer<DotsUniform>,
    points: wgpu::Buffer,
    /// The floating dots' velocities (one vec4 per dot).
    velocities: wgpu::Buffer,
    rows: Option<(wgpu::Buffer, u64)>,
    dummy_scalar: Volume3D,
    dummy_colour: ColourVolume3D,
    bound: Option<Bound>,
    count: u32,
}

impl DotsPass {
    pub fn new(ctx: &GpuContext, colour_format: wgpu::TextureFormat, camera: &UniformBuffer<CameraUniform>) -> DotsPass {
        let dev = &ctx.device;
        let cs = dev.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("dots compute"), source: wgpu::ShaderSource::Wgsl(include_str!("shaders/dots_compute.wgsl").into()) });
        let ds = dev.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("dots draw"), source: wgpu::ShaderSource::Wgsl(include_str!("shaders/dots_draw.wgsl").into()) });
        let uniform = |binding, visibility| wgpu::BindGroupLayoutEntry { binding, visibility, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let storage = |binding, visibility, read_only| wgpu::BindGroupLayoutEntry { binding, visibility, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None }, count: None };
        // r32float (unfilterable) and rgba16float both bind as unfilterable float for textureLoad
        let tex3 = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false }, count: None };
        let c = wgpu::ShaderStages::COMPUTE;
        let compute_bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("dots compute bgl"), entries: &[uniform(0, c), tex3(1), tex3(2), storage(3, c, false), storage(4, c, false), storage(5, c, false)] });
        let cl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&compute_bgl)], immediate_size: 0 });
        let pipe = |entry: &str| dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(entry), layout: Some(&cl), module: &cs, entry_point: Some(entry), compilation_options: Default::default(), cache: None });
        let (row_sums, row_scan, gen_points) = (pipe("row_sums"), pipe("row_scan"), pipe("gen_points"));
        let camera_bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("dots camera bgl"), entries: &[uniform(0, wgpu::ShaderStages::VERTEX_FRAGMENT)] });
        let camera_bg = dev.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("dots camera bg"), layout: &camera_bgl, entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera.buffer.as_entire_binding() }] });
        let vf = wgpu::ShaderStages::VERTEX_FRAGMENT;
        let draw_bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("dots draw bgl"), entries: &[
            uniform(0, vf),
            storage(1, wgpu::ShaderStages::VERTEX, true),
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
        ] });
        let dl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&camera_bgl), Some(&draw_bgl)], immediate_size: 0 });
        // premultiplied "over" for the colour (a dense region saturates to the dots' own colour,
        // so the orbitals' hues survive where additive blending would go white); the target's
        // alpha is kept
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
            alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
        };
        let draw = dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dots"), layout: Some(&dl),
            vertex: wgpu::VertexState { module: &ds, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &ds, entry_point: Some("fs_main"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format: colour_format, blend: Some(additive), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: None, multisample: Default::default(), multiview_mask: None, cache: None,
        });
        let add = wgpu::BlendState {
            color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
            alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
        };
        let draw_glow = dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dots glow"), layout: Some(&dl),
            vertex: wgpu::VertexState { module: &ds, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &ds, entry_point: Some("fs_main"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format: colour_format, blend: Some(add), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: None, multisample: Default::default(), multiview_mask: None, cache: None,
        });
        let velocities = dev.create_buffer(&wgpu::BufferDescriptor { label: Some("dots velocities"), size: MAX_DOTS as u64 * 16, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        let points = dev.create_buffer(&wgpu::BufferDescriptor { label: Some("dots points"), size: MAX_DOTS as u64 * 16, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        DotsPass {
            row_sums, row_scan, gen_points, draw, draw_glow, glow: false, compute_bgl, draw_bgl, camera_bg,
            params: UniformBuffer::new(ctx, "dots params"), points, velocities, rows: None,
            dummy_scalar: Volume3D::new(ctx, [1; 3]), dummy_colour: ColourVolume3D::new(ctx, [1; 3]),
            bound: None, count: 0,
        }
    }

    /// Binds the field to sample (`colour`, when given, is sampled by its a and coloured by it
    /// instead of `scalar`) and the scene depth.
    pub fn bind(&mut self, ctx: &GpuContext, scalar: &Volume3D, colour: Option<&ColourVolume3D>, scene_depth: &wgpu::TextureView) {
        let dims = colour.map_or(scalar.dims, |c| c.dims);
        let use_colour = colour.is_some();
        // rows of cells (the cubes between grid points): sums, maxima, the global maximum
        let n_rows = (dims[1].max(2) - 1) * (dims[2].max(2) - 1);
        let bytes = (2 * n_rows as u64 + 1) * 4;
        if self.rows.as_ref().is_none_or(|(_, cap)| *cap < bytes) {
            self.rows = Some((ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("dots rows"), size: bytes, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false }), bytes));
        }
        let (rows, _) = self.rows.as_ref().expect("created above");
        // the unused texture slot gets a 1³ dummy
        let (scalar, colour) = match colour {
            Some(c) => (&self.dummy_scalar, c),
            None => (scalar, &self.dummy_colour),
        };
        let compute_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("dots compute bg"), layout: &self.compute_bgl, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: self.params.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&scalar.view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&colour.view) },
            wgpu::BindGroupEntry { binding: 3, resource: rows.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: self.points.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: self.velocities.as_entire_binding() },
        ] });
        let draw_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("dots draw bg"), layout: &self.draw_bgl, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: self.params.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: self.points.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(scene_depth) },
        ] });
        self.bound = Some(Bound { compute_bg, draw_bg, n_rows, dims, colour: use_colour });
    }

    /// Nothing to sample (the empty scene): draws nothing until the next [`DotsPass::bind`].
    pub fn unbind(&mut self) { self.bound = None; }

    /// This frame's parameters (`count` is clamped to [`MAX_DOTS`]; `dims` and `use_colour` are
    /// the bound field's).
    pub fn set_params(&mut self, ctx: &GpuContext, p: &DotsUniform) {
        self.glow = p.glow != 0;
        let Some(b) = &self.bound else { return };
        let p = DotsUniform { count: p.count.min(MAX_DOTS), dims: b.dims, use_colour: u32::from(b.colour), ..*p };
        self.count = p.count;
        self.params.write(ctx, &p);
    }

    /// The three compute passes (outside any render pass). No-op until bound.
    pub fn generate(&self, enc: &mut wgpu::CommandEncoder) {
        let Some(b) = &self.bound else { return };
        if self.count == 0 { return; }
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("dots"), timestamp_writes: None });
        pass.set_bind_group(0, &b.compute_bg, &[]);
        pass.set_pipeline(&self.row_sums);
        pass.dispatch_workgroups(b.n_rows.div_ceil(64), 1, 1);
        pass.set_pipeline(&self.row_scan);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&self.gen_points);
        pass.dispatch_workgroups(self.count.div_ceil(64), 1, 1);
    }

    /// The current dots (world position, packed colour) read back — for tests and the headless
    /// checks only (the frame path never reads them).
    pub fn read_points(&self, ctx: &GpuContext) -> Result<Vec<([f32; 3], [u8; 4])>, gpu::ReadbackError> {
        let n = self.count as u64;
        if n == 0 { return Ok(Vec::new()); }
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("dots readback"), size: n * 16, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.points, 0, &staging, 0, n * 16);
        ctx.queue.submit([enc.finish()]);
        gpu::read_buffer(ctx, &staging, 0..n * 16, |bytes| {
            let v: &[[f32; 4]] = bytemuck::cast_slice(bytes);
            v.iter().map(|p| ([p[0], p[1], p[2]], p[3].to_bits().to_le_bytes())).collect()
        })
    }

    /// The sprites, in a pass without a depth attachment (the scene depth is read as a texture).
    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        let Some(b) = &self.bound else { return };
        if self.count == 0 { return; }
        pass.set_pipeline(if self.glow { &self.draw_glow } else { &self.draw });
        pass.set_bind_group(0, &self.camera_bg, &[]);
        pass.set_bind_group(1, &b.draw_bg, &[]);
        pass.draw(0..6, 0..self.count);
    }
}

#[cfg(test)]
mod tests {
    //! The sampler's statistics on synthetic fields (GPU; skipped without an adapter): dots read
    //! back from the points buffer, histograms compared with what the trilinear field implies,
    //! by a χ² against Poisson errors.
    use super::*;
    use crate::headless::{Headless, HEADLESS_FORMAT};

    const N: u32 = 17;
    const LO: f32 = -2.0;
    const HI: f32 = 2.0;
    const DOTS: u32 = 400_000;

    /// Dots of the field `f(x, y, z)` (grid point coordinates in Bohr) on N³ over [LO, HI]³.
    fn dots_of(f: impl Fn(f32, f32, f32) -> f32, seed: u32, signed: u32) -> Option<Vec<[f32; 3]>> {
        let hl = Headless::new(16, 16)?;
        let ctx = &hl.ctx;
        let h = (HI - LO) / (N - 1) as f32;
        let mut data = Vec::with_capacity((N * N * N) as usize);
        for k in 0..N {
            for j in 0..N {
                for i in 0..N {
                    data.push(f(LO + i as f32 * h, LO + j as f32 * h, LO + k as f32 * h));
                }
            }
        }
        let vol = Volume3D::new(ctx, [N; 3]);
        vol.upload(ctx, &data);
        let camera = UniformBuffer::<CameraUniform>::new(ctx, "camera");
        let mut pass = DotsPass::new(ctx, HEADLESS_FORMAT, &camera);
        pass.bind(ctx, &vol, None, &hl.depth);
        pass.set_params(ctx, &DotsUniform {
            box_min: [LO; 3], use_colour: 0, box_max: [HI; 3], count: DOTS, dims: [0; 3], seed,
            tint: [1.0; 4], tint_neg: [1.0; 4], size_px: 1.0, intensity: 1.0, colour_by: COLOUR_BY_DENSITY, signed,
            walk: 0.0, refresh: 0.0, glow: 0, cut: 0.0,
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        pass.generate(&mut enc);
        ctx.queue.submit([enc.finish()]);
        Some(pass.read_points(ctx).expect("readback").into_iter().map(|(p, _)| p).collect())
    }

    /// χ² of observed counts against expected ones (Poisson errors).
    fn chi2(obs: &[f64], exp: &[f64]) -> f64 {
        obs.iter().zip(exp).map(|(o, e)| (o - e) * (o - e) / e).sum()
    }

    /// A constant field: dots fill the box uniformly, so every axis's fractional cell coordinate
    /// is uniform (no lattice: no clustering at grid points, planes or cell centres).
    #[test]
    fn constant_field_fractional_cell_coordinates_are_uniform() {
        let Some(p) = dots_of(|_, _, _| 1.0, 7, 0) else { return };
        assert_eq!(p.len(), DOTS as usize);
        let h = (HI - LO) / (N - 1) as f32;
        const BINS: usize = 20;
        for a in 0..3 {
            let mut hist = [0.0f64; BINS];
            for q in &p {
                assert!(q[a] >= LO - 1e-4 && q[a] <= HI + 1e-4, "dot outside the box: {q:?}");
                let g = (q[a] - LO) / h;
                let fr = (g - g.floor()).clamp(0.0, 0.999_999);
                hist[(fr * BINS as f32) as usize] += 1.0;
            }
            let exp = vec![p.len() as f64 / BINS as f64; BINS];
            let c = chi2(&hist, &exp);
            eprintln!("axis {a}: fractional-coordinate χ² = {c:.1} (19 dof)");
            // 19 degrees of freedom: P(χ² > 50) ≈ 1e-4
            assert!(c < 50.0, "axis {a}: fractional coordinate histogram χ² = {c:.1} (19 dof): {hist:?}");
        }
    }

    /// A linear ramp along x, f = 2.5 + x: the trilinear field is exactly that ramp, so the dot
    /// density along x is linear — no staircase of piecewise-constant voxels (whose steps at the
    /// cell faces showed as cube edges around a sharp peak and as lattice planes along the axes).
    /// Fine bins (8 per cell) compared with the ramp's integral over each bin; the test's own
    /// sensitivity is checked too: the staircase a nearest-voxel sampler would give must fail it.
    #[test]
    fn dot_density_follows_the_trilinear_field_without_steps() {
        let Some(p) = dots_of(|x, _, _| 2.5 + x, 11, 0) else { return };
        let bins = 8 * (N as usize - 1);
        let w = (HI - LO) as f64 / bins as f64;
        let mut hist = vec![0.0f64; bins];
        for q in &p {
            hist[(((q[0] - LO) as f64 / w) as usize).min(bins - 1)] += 1.0;
        }
        let f = |x: f64| 2.5 * x + x * x / 2.0; // the ramp's antiderivative
        let total = f(HI as f64) - f(LO as f64);
        let n = p.len() as f64;
        let exp: Vec<f64> = (0..bins)
            .map(|b| {
                let (a, c) = (LO as f64 + b as f64 * w, LO as f64 + (b + 1) as f64 * w);
                n * (f(c) - f(a)) / total
            })
            .collect();
        let limit = |dof: f64| dof + 5.0 * (2.0 * dof).sqrt();
        let dof = (bins - 1) as f64;
        // the staircase: each grid point's value over its voxel [x_i − h/2, x_i + h/2] (clipped)
        let h = (HI - LO) as f64 / (N - 1) as f64;
        let stair_w: f64 = (0..N).map(|i| { let x = LO as f64 + i as f64 * h; let e = if i == 0 || i == N - 1 { 0.5 } else { 1.0 }; (2.5 + x) * h * e }).sum();
        let stair: Vec<f64> = (0..bins)
            .map(|b| {
                let mid = LO as f64 + (b as f64 + 0.5) * w;
                let i = ((mid - LO as f64) / h).round();
                n * (2.5 + LO as f64 + i * h) * w / stair_w
            })
            .collect();
        let stair_chi = chi2(&stair, &exp);
        assert!(stair_chi > 2.0 * limit(dof), "the test would not see a staircase: χ² {stair_chi:.1}");
        let c = chi2(&hist, &exp);
        eprintln!("ramp: χ² = {c:.1} for {dof} dof (a staircase would give {stair_chi:.0})");
        assert!(c < limit(dof), "χ² = {c:.1} for {dof} dof: the density along x is not the linear ramp");
    }

    /// A sharp peak (Gaussian, σ = h, centred on a grid point): the dots' radial profile out to
    /// 3h matches the trilinear field's (computed on the CPU by Monte Carlo) in both direction
    /// classes — along the axes and along the diagonals — so no cube-shaped excess around the
    /// peak beyond what the interpolant itself has.
    #[test]
    fn sharp_peak_matches_its_trilinear_field_along_axes_and_diagonals() {
        let h = (HI - LO) / (N - 1) as f32; // σ = h
        let g = move |x: f32, y: f32, z: f32| (-(x * x + y * y + z * z) / (2.0 * h * h)).exp();
        let Some(p) = dots_of(g, 3, 0) else { return };
        // the trilinear interpolant of the grid samples
        let tri = |x: f32, y: f32, z: f32| -> f32 {
            let c = [x, y, z].map(|v| (v - LO) / h);
            let i = c.map(|v| (v.floor() as i32).clamp(0, N as i32 - 2));
            let t: [f32; 3] = std::array::from_fn(|a| c[a] - i[a] as f32);
            let mut acc = 0.0;
            for k in 0..8 {
                let b = [k & 1, (k >> 1) & 1, (k >> 2) & 1];
                let w: f32 = (0..3).map(|a| if b[a] == 1 { t[a] } else { 1.0 - t[a] }).product();
                let q: [f32; 3] = std::array::from_fn(|a| LO + (i[a] + b[a]) as f32 * h);
                acc += w * g(q[0], q[1], q[2]);
            }
            acc
        };
        // direction classes: axis-like (one component dominates) or diagonal-like
        let class = |q: [f32; 3]| -> Option<usize> {
            let r = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
            if !(0.05..3.0 * h).contains(&r) {
                return None;
            }
            let m = q.iter().map(|v| v.abs() / r).fold(0.0f32, f32::max);
            if m > 0.95 { Some(0) } else if m < 0.65 { Some(1) } else { None }
        };
        const RB: usize = 6;
        let rbin = |q: [f32; 3]| (((q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt() / (3.0 * h) * RB as f32) as usize).min(RB - 1);
        let mut obs = [[0.0f64; RB]; 2];
        for q in &p {
            if let Some(c) = class(*q) {
                obs[c][rbin(*q)] += 1.0;
            }
        }
        // expected: uniform Monte Carlo points in [−3h, 3h]³ weighted by the trilinear field (the
        // peak's mass outside that cube is below 1e-3 of the total and is added from the grid)
        let mut rng = 0x1234_5678u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 11) as f64 / (1u64 << 53) as f64
        };
        let m = 4_000_000usize;
        let side = 3.0 * h;
        let mut exp = [[0.0f64; RB]; 2];
        let mut exp_sq = [[0.0f64; RB]; 2];
        for _ in 0..m {
            let q = [0; 3].map(|_| -side + 2.0 * side * next() as f32);
            if let Some(c) = class(q) {
                let w = tri(q[0], q[1], q[2]) as f64;
                exp[c][rbin(q)] += w;
                exp_sq[c][rbin(q)] += w * w;
            }
        }
        // total trilinear mass = Σ grid values × h³ (each sample's tent integrates to h³); the
        // inner cube's mass = mean weight × its volume
        let total: f64 = (0..N * N * N)
            .map(|n| {
                let (i, j, k) = (n % N, (n / N) % N, n / (N * N));
                g(LO + i as f32 * h, LO + j as f32 * h, LO + k as f32 * h) as f64
            })
            .sum::<f64>()
            * (h as f64).powi(3);
        let vol = (2.0 * side as f64).powi(3);
        let scale = vol / m as f64 / total * p.len() as f64;
        for c in 0..2 {
            for b in 0..RB {
                exp[c][b] *= scale;
                exp_sq[c][b] *= scale * scale; // the Monte Carlo variance of exp[c][b]
            }
        }
        for c in 0..2 {
            // Poisson error of the dots plus the Monte Carlo error of the expectation
            let chi: f64 = (0..RB)
                .filter(|&b| exp[c][b] > 20.0)
                .map(|b| {
                    let d = obs[c][b] - exp[c][b];
                    d * d / (exp[c][b] + exp_sq[c][b])
                })
                .sum();
            eprintln!("{}: χ² {chi:.1}; observed {:?} expected {:?}", ["axes", "diagonals"][c], obs[c], exp[c].map(|e| e.round()));
            assert!(chi < 25.0, "class {}: χ² {chi:.1}; observed {:?} expected {:?}", ["axes", "diagonals"][c], obs[c], exp[c]);
        }
    }

    /// A signed field (the spin density): with `signed` set the dots follow |f|, on both sides of
    /// the node. f = x on a grid with a point at x = 0 makes |f| exactly piecewise linear, so the
    /// per-cell counts along x are the integral of |x| over each bin — and without the flag the
    /// negative half would get none (clamped to 0).
    #[test]
    fn signed_fields_are_sampled_by_their_absolute_value() {
        let Some(p) = dots_of(|x, _, _| x, 5, 1) else { return };
        const BINS: usize = 16;
        let w = (HI - LO) as f64 / BINS as f64;
        let mut hist = [0.0f64; BINS];
        for q in &p {
            hist[(((q[0] - LO) as f64 / w) as usize).min(BINS - 1)] += 1.0;
        }
        let f = |x: f64| x * x.abs() / 2.0; // the antiderivative of |x|
        let n = p.len() as f64;
        let exp: Vec<f64> = (0..BINS).map(|b| { let (a, c) = (LO as f64 + b as f64 * w, LO as f64 + (b + 1) as f64 * w); n * (f(c) - f(a)) / 4.0 }).collect();
        let c = chi2(&hist, &exp);
        eprintln!("signed ramp: χ² = {c:.1} for {} dof; negative half {:.0} of {n}", BINS - 1, hist[..BINS / 2].iter().sum::<f64>());
        assert!(c < (BINS - 1) as f64 + 5.0 * (2.0 * (BINS - 1) as f64).sqrt(), "χ² = {c:.1}: {hist:?}");
        let Some(clamped) = dots_of(|x, _, _| x, 5, 0) else { return };
        assert!(clamped.iter().all(|q| q[0] >= -1e-4), "an unsigned field must not put dots where it is negative");
    }
}
