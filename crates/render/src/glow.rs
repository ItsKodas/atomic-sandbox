//! The Camera view's atoms (user request 2026-10-19): each atom a soft electron glow, drawn at
//! every instant of the shutter with weight 1/N and summed into a float target (a time-average:
//! real motion blur), then composited over the scene with an exposure and a filmic curve
//! (`shaders/camera.wgsl`, `shaders/camera_composite.wgsl`). Display only.
use crate::spheres::{CameraUniform, SphereInstance};
use gpu::{GpuContext, UniformBuffer};

const ACCUM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub struct GlowPass {
    glow: wgpu::RenderPipeline,
    glow_bg: wgpu::BindGroup,
    composite: wgpu::RenderPipeline,
    composite_bgl: wgpu::BindGroupLayout,
    look: UniformBuffer<[f32; 4]>,
    accum: Option<(wgpu::TextureView, wgpu::BindGroup, [u32; 2])>,
    instances: wgpu::Buffer,
    capacity: usize,
    count: u32,
}

impl GlowPass {
    pub fn new(ctx: &GpuContext, colour_format: wgpu::TextureFormat, camera: &UniformBuffer<CameraUniform>) -> GlowPass {
        let dev = &ctx.device;
        let glow_src = dev.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("glow"), source: wgpu::ShaderSource::Wgsl(include_str!("shaders/camera.wgsl").into()) });
        let comp_src = dev.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("glow composite"), source: wgpu::ShaderSource::Wgsl(include_str!("shaders/camera_composite.wgsl").into()) });
        let cam_bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("glow camera"), entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }] });
        let glow_bg = dev.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("glow camera"), layout: &cam_bgl, entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera.buffer.as_entire_binding() }] });
        let glow_layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&cam_bgl)], immediate_size: 0 });
        let add = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add };
        let glow = dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("glow"), layout: Some(&glow_layout),
            vertex: wgpu::VertexState { module: &glow_src, entry_point: Some("vs_glow"), compilation_options: Default::default(), buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<SphereInstance>() as u64, step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32, 2 => Float32x3, 3 => Float32] })] },
            fragment: Some(wgpu::FragmentState { module: &glow_src, entry_point: Some("fs_glow"), compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: ACCUM_FORMAT, blend: Some(wgpu::BlendState { color: add, alpha: add }), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: None, multisample: Default::default(), multiview_mask: None, cache: None,
        });
        let composite_bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("glow composite"), entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ] });
        let comp_layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&composite_bgl)], immediate_size: 0 });
        let over = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add };
        let composite = dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("glow composite"), layout: Some(&comp_layout),
            vertex: wgpu::VertexState { module: &comp_src, entry_point: Some("vs_full"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &comp_src, entry_point: Some("fs_composite"), compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: colour_format, blend: Some(wgpu::BlendState { color: over, alpha: over }), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: None, multisample: Default::default(), multiview_mask: None, cache: None,
        });
        let look = UniformBuffer::<[f32; 4]>::new(ctx, "glow look");
        look.write(ctx, &[1.0, 0.0, 0.0, 0.0]);
        let instances = Self::make_buffer(ctx, 64);
        GlowPass { glow, glow_bg, composite, composite_bgl, look, accum: None, instances, capacity: 64, count: 0 }
    }

    fn make_buffer(ctx: &GpuContext, capacity: usize) -> wgpu::Buffer {
        ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("glow instances"), size: (capacity * std::mem::size_of::<SphereInstance>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
    }

    /// Every instant's atoms (`selected` = that instant's weight, 1/N), and the exposure (EV).
    pub fn set(&mut self, ctx: &GpuContext, instances: &[SphereInstance], exposure_ev: f32) {
        if instances.len() > self.capacity { self.capacity = instances.len().next_power_of_two(); self.instances = Self::make_buffer(ctx, self.capacity); }
        if !instances.is_empty() { ctx.queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(instances)); }
        self.count = instances.len() as u32;
        self.look.write(ctx, &[2f32.powf(exposure_ev), 0.0, 0.0, 0.0]);
    }

    /// The float target at the colour target's size (made or remade on a resize).
    pub fn ensure(&mut self, ctx: &GpuContext, size: [u32; 2]) {
        if self.accum.as_ref().is_some_and(|a| a.2 == size) || size[0] == 0 || size[1] == 0 {
            return;
        }
        let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("glow accum"), size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: ACCUM_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING, view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("glow composite"), layout: &self.composite_bgl, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
            wgpu::BindGroupEntry { binding: 1, resource: self.look.buffer.as_entire_binding() },
        ] });
        self.accum = Some((view, bg, size));
    }

    /// The glow into the float target, then over `colour`; `confine` sets each pass's viewport and
    /// scissor (false: nothing to draw).
    pub fn render(&self, enc: &mut wgpu::CommandEncoder, colour: &wgpu::TextureView, confine: &dyn Fn(&mut wgpu::RenderPass<'_>) -> bool) {
        let Some((view, bg, _)) = &self.accum else { return };
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("glow"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store }, depth_slice: None })],
                depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
            });
            if self.count == 0 || !confine(&mut pass) {
                return;
            }
            pass.set_pipeline(&self.glow);
            pass.set_bind_group(0, &self.glow_bg, &[]);
            pass.set_vertex_buffer(0, self.instances.slice(..));
            pass.draw(0..6, 0..self.count);
        }
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("glow composite"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: colour, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }, depth_slice: None })],
            depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
        });
        if !confine(&mut pass) {
            return;
        }
        pass.set_pipeline(&self.composite);
        pass.set_bind_group(0, bg, &[]);
        pass.draw(0..3, 0..1);
    }
}
