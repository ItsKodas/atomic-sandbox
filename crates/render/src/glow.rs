//! The Camera view's atoms (user request 2026-10-19): each atom between a soft electron glow and a
//! solid shaded ball (the solidity), drawn at every instant of the shutter with weight 1/N and
//! summed into a float target (a time-average: real motion blur), then composited over the scene
//! with an exposure (`shaders/camera.wgsl`, `shaders/camera_composite.wgsl`). With any solidity,
//! each instant first lays its balls' depth so a ball hides what is behind it at that instant.
//! Display only.
use crate::camera::Viewport;
use crate::spheres::{CameraUniform, SphereInstance};
use gpu::{GpuContext, UniformBuffer};

const ACCUM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Supersampling: the atoms are drawn at this many times the view's resolution and filtered down
/// (smooth outlines and intersection lines; a solid ball's edge cannot be blended in place).
pub const SS: u32 = 2;

pub struct GlowPass {
    glow: wgpu::RenderPipeline,
    /// The balls' depth for one instant (solidity > 0).
    prepass: wgpu::RenderPipeline,
    glow_bg: wgpu::BindGroup,
    composite: wgpu::RenderPipeline,
    composite_bgl: wgpu::BindGroupLayout,
    look: UniformBuffer<[f32; 4]>,
    /// The camera at the supersampled resolution (the viewport and origin × [`SS`]).
    cam: UniformBuffer<CameraUniform>,
    /// The float target, its composite bind group, its size and the depth target.
    accum: Option<(wgpu::TextureView, wgpu::BindGroup, [u32; 2], wgpu::TextureView)>,
    instances: wgpu::Buffer,
    capacity: usize,
    count: u32,
    /// Instances per instant (laid out instant after instant), and the solidity.
    per_instant: u32,
    solidity: f32,
}

fn instance_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRS: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32, 2 => Float32x3, 3 => Float32];
    wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<SphereInstance>() as u64, step_mode: wgpu::VertexStepMode::Instance, attributes: &ATTRS }
}

impl GlowPass {
    pub fn new(ctx: &GpuContext, colour_format: wgpu::TextureFormat, _camera: &UniformBuffer<CameraUniform>) -> GlowPass {
        let dev = &ctx.device;
        let cam = UniformBuffer::<CameraUniform>::new(ctx, "glow camera");
        let glow_src = dev.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("glow"), source: wgpu::ShaderSource::Wgsl(include_str!("shaders/camera.wgsl").into()) });
        let comp_src = dev.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("glow composite"), source: wgpu::ShaderSource::Wgsl(include_str!("shaders/camera_composite.wgsl").into()) });
        let uniform = |binding: u32, visibility: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let look = UniformBuffer::<[f32; 4]>::new(ctx, "glow look");
        look.write(ctx, &[1.0, 0.0, 0.0, 0.0]);
        let cam_bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("glow camera"),
            entries: &[uniform(0, wgpu::ShaderStages::VERTEX_FRAGMENT), uniform(1, wgpu::ShaderStages::VERTEX_FRAGMENT)],
        });
        let glow_bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("glow camera"),
            layout: &cam_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: cam.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: look.buffer.as_entire_binding() },
            ],
        });
        let glow_layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&cam_bgl)], immediate_size: 0 });
        let add = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add };
        let tri = wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() };
        let depth = |write: bool, cmp: wgpu::CompareFunction| {
            Some(wgpu::DepthStencilState { format: DEPTH, depth_write_enabled: Some(write), depth_compare: Some(cmp), stencil: Default::default(), bias: Default::default() })
        };
        let glow = dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("glow"),
            layout: Some(&glow_layout),
            vertex: wgpu::VertexState { module: &glow_src, entry_point: Some("vs_glow"), compilation_options: Default::default(), buffers: &[Some(instance_layout())] },
            fragment: Some(wgpu::FragmentState {
                module: &glow_src,
                entry_point: Some("fs_glow"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: ACCUM_FORMAT, blend: Some(wgpu::BlendState { color: add, alpha: add }), write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: tri,
            depth_stencil: depth(false, wgpu::CompareFunction::LessEqual),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let prepass = dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("glow depth"),
            layout: Some(&glow_layout),
            vertex: wgpu::VertexState { module: &glow_src, entry_point: Some("vs_glow"), compilation_options: Default::default(), buffers: &[Some(instance_layout())] },
            fragment: Some(wgpu::FragmentState { module: &glow_src, entry_point: Some("fs_depth"), compilation_options: Default::default(), targets: &[] }),
            primitive: tri,
            depth_stencil: depth(true, wgpu::CompareFunction::Less),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let composite_bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("glow composite"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                uniform(1, wgpu::ShaderStages::FRAGMENT),
            ],
        });
        let comp_layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&composite_bgl)], immediate_size: 0 });
        let over = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add };
        let composite = dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("glow composite"),
            layout: Some(&comp_layout),
            vertex: wgpu::VertexState { module: &comp_src, entry_point: Some("vs_full"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &comp_src,
                entry_point: Some("fs_composite"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: colour_format, blend: Some(wgpu::BlendState { color: over, alpha: over }), write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: tri,
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let instances = Self::make_buffer(ctx, 64);
        GlowPass { glow, prepass, glow_bg, composite, composite_bgl, look, cam, accum: None, instances, capacity: 64, count: 0, per_instant: 0, solidity: 0.0 }
    }

    fn make_buffer(ctx: &GpuContext, capacity: usize) -> wgpu::Buffer {
        ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("glow instances"),
            size: (capacity * std::mem::size_of::<SphereInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Every instant's atoms, `per_instant` each, instant after instant (`selected` = the
    /// instant's weight, 1/N), the exposure (EV) and the solidity (0 glow … 1 ball).
    pub fn set(&mut self, ctx: &GpuContext, instances: &[SphereInstance], per_instant: usize, exposure_ev: f32, solidity: f32) {
        if instances.len() > self.capacity {
            self.capacity = instances.len().next_power_of_two();
            self.instances = Self::make_buffer(ctx, self.capacity);
        }
        if !instances.is_empty() {
            ctx.queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(instances));
        }
        self.count = instances.len() as u32;
        self.per_instant = per_instant as u32;
        self.solidity = solidity;
        self.look.write(ctx, &[2f32.powf(exposure_ev), solidity, 0.0, 0.0]);
    }

    /// The camera of this frame (`Renderer::update_camera_in`), scaled to the supersampled targets.
    pub fn set_camera(&self, ctx: &GpuContext, mut u: CameraUniform) {
        let s = SS as f32;
        u.viewport = [u.viewport[0] * s, u.viewport[1] * s];
        u.origin = [u.origin[0] * s, u.origin[1] * s];
        self.cam.write(ctx, &u);
    }

    /// The float and depth targets at [`SS`] × the colour target's size (made or remade on a resize).
    pub fn ensure(&mut self, ctx: &GpuContext, size: [u32; 2]) {
        if self.accum.as_ref().is_some_and(|a| a.2 == size) || size[0] == 0 || size[1] == 0 {
            return;
        }
        let extent = wgpu::Extent3d { width: size[0] * SS, height: size[1] * SS, depth_or_array_layers: 1 };
        let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("glow accum"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: ACCUM_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        let depth = ctx
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("glow depth"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("glow composite"),
            layout: &self.composite_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: self.look.buffer.as_entire_binding() },
            ],
        });
        self.accum = Some((view, bg, size, depth));
    }

    /// The atoms into the float target, then over `colour`; `confine` sets each pass's viewport and
    /// scissor (false: nothing to draw).
    pub fn render(&self, enc: &mut wgpu::CommandEncoder, colour: &wgpu::TextureView, vp: Option<(Viewport, [u32; 2])>, confine: &dyn Fn(&mut wgpu::RenderPass<'_>) -> bool) {
        let Some((view, bg, size, depth)) = &self.accum else { return };
        // the supersampled passes' viewport and scissor: the view's, × SS
        let s = SS as f32;
        let hi = |pass: &mut wgpu::RenderPass<'_>| -> bool {
            let (v, target) = match vp {
                Some((v, t)) => (Viewport { origin: [v.origin[0] * s, v.origin[1] * s], size: [v.size[0] * s, v.size[1] * s] }, [t[0] * SS, t[1] * SS]),
                None => (Viewport::full((size[0] * SS) as f32, (size[1] * SS) as f32), [size[0] * SS, size[1] * SS]),
            };
            let Some([x, y, w, h]) = v.scissor(target) else { return false };
            pass.set_viewport(v.origin[0], v.origin[1], v.size[0], v.size[1], 0.0, 1.0);
            pass.set_scissor_rect(x, y, w, h);
            true
        };
        let colour_pass = |enc: &mut wgpu::CommandEncoder, clear: bool| -> wgpu::RenderPass<'static> {
            let load = if clear { wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT) } else { wgpu::LoadOp::Load };
            let dload = if clear { wgpu::LoadOp::Clear(1.0) } else { wgpu::LoadOp::Load };
            enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("glow"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view, resolve_target: None, ops: wgpu::Operations { load, store: wgpu::StoreOp::Store }, depth_slice: None })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: depth, depth_ops: Some(wgpu::Operations { load: dload, store: wgpu::StoreOp::Store }), stencil_ops: None }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            })
            .forget_lifetime()
        };
        let draw = |pass: &mut wgpu::RenderPass<'_>, pipeline: &wgpu::RenderPipeline, range: std::ops::Range<u32>| {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.glow_bg, &[]);
            pass.set_vertex_buffer(0, self.instances.slice(..));
            pass.draw(0..6, range);
        };
        if self.solidity <= 0.0 || self.count == 0 {
            // glows only: no instant hides another — one pass for all of them
            let mut pass = colour_pass(enc, true);
            if self.count > 0 && hi(&mut pass) {
                draw(&mut pass, &self.glow, 0..self.count);
            }
        } else {
            // per instant: the balls' depth, then the atoms against it
            drop(colour_pass(enc, true));
            let per = self.per_instant.max(1);
            let mut start = 0;
            while start < self.count {
                let range = start..(start + per).min(self.count);
                {
                    let mut pass = enc
                        .begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("glow depth"),
                            color_attachments: &[],
                            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: depth, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        })
                        .forget_lifetime();
                    if hi(&mut pass) {
                        draw(&mut pass, &self.prepass, range.clone());
                    }
                }
                let mut pass = colour_pass(enc, false);
                if hi(&mut pass) {
                    draw(&mut pass, &self.glow, range);
                }
                start += per;
            }
        }
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("glow composite"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: colour, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }, depth_slice: None })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        if !confine(&mut pass) {
            return;
        }
        pass.set_pipeline(&self.composite);
        pass.set_bind_group(0, bg, &[]);
        pass.draw(0..3, 0..1);
    }
}
