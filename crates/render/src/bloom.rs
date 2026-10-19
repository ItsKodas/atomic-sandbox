//! Bloom (user request 2026-10-19/20; display only): the bright parts of the finished 3D frame
//! spill a soft glow — dense glowing dots, a hot volume core, the Camera view's lit atoms. Drawn
//! after the scene and before the UI, reading the frame as a texture (the target needs
//! `TEXTURE_BINDING`): a soft threshold into a half-size image, a chain of [`LEVELS`] 13-tap
//! downsamples, tent upsamples added back level by level, and the result added onto the frame
//! × the intensity (`shaders/bloom.wgsl`). The chain is rgba16float and follows the target size.
use gpu::{GpuContext, UniformBuffer};

/// Half-size images in the chain (1/2 … 1/32 of the frame).
pub const LEVELS: usize = 5;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct BloomUniform {
    texel: [f32; 2],
    threshold: f32,
    knee: f32,
    intensity: f32,
    pad: [f32; 3],
}

pub struct Bloom {
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    prefilter: wgpu::RenderPipeline,
    down: wgpu::RenderPipeline,
    up: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    /// The chain (texture views, sizes) for `size`.
    chain: Vec<(wgpu::TextureView, [u32; 2])>,
    size: [u32; 2],
    /// One uniform per draw of a frame (prefilter, downs, ups, composite).
    uniforms: Vec<UniformBuffer<BloomUniform>>,
}

impl Bloom {
    /// For frames of `target_format` (the composite's blend target).
    pub fn new(ctx: &GpuContext, target_format: wgpu::TextureFormat) -> Bloom {
        let dev = &ctx.device;
        let shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("bloom"), source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bloom.wgsl").into()) });
        let f = wgpu::ShaderStages::FRAGMENT;
        let bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: f, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: f, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: f, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });
        let layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let add = wgpu::BlendState {
            color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
            alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
        };
        let pipe = |entry: &str, format: wgpu::TextureFormat, blend: Option<wgpu::BlendState>| {
            dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[] },
                fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some(entry), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend, write_mask: wgpu::ColorWrites::ALL })] }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let sampler = dev.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bloom sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniforms = (0..2 * LEVELS + 2).map(|_| UniformBuffer::<BloomUniform>::new(ctx, "bloom")).collect();
        Bloom {
            prefilter: pipe("fs_prefilter", FORMAT, None),
            down: pipe("fs_down", FORMAT, None),
            up: pipe("fs_up", FORMAT, Some(add)),
            composite: pipe("fs_composite", target_format, Some(add)),
            bgl,
            sampler,
            chain: Vec::new(),
            size: [0, 0],
            uniforms,
        }
    }

    fn ensure(&mut self, ctx: &GpuContext, size: [u32; 2]) {
        if self.size == size && !self.chain.is_empty() {
            return;
        }
        self.size = size;
        self.chain = (0..LEVELS)
            .map(|k| {
                let s = [(size[0] >> (k + 1)).max(1), (size[1] >> (k + 1)).max(1)];
                let t = ctx.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("bloom level"),
                    size: wgpu::Extent3d { width: s[0], height: s[1], depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                (t.create_view(&Default::default()), s)
            })
            .collect();
    }

    /// Adds the bloom of `frame` (a sampleable view of the target, `size` pixels) onto `target`
    /// (the same texture's render view). `intensity` 0 does nothing.
    #[allow(clippy::too_many_arguments)]
    pub fn apply(&mut self, ctx: &GpuContext, enc: &mut wgpu::CommandEncoder, frame: &wgpu::TextureView, target: &wgpu::TextureView, size: [u32; 2], intensity: f32, threshold: f32) {
        if intensity <= 0.0 || size[0] < 4 || size[1] < 4 {
            return;
        }
        self.ensure(ctx, size);
        let mut u = 0;
        let mut draw = |enc: &mut wgpu::CommandEncoder, pipeline: &wgpu::RenderPipeline, src: &wgpu::TextureView, src_size: [u32; 2], dst: &wgpu::TextureView, clear: bool, intensity: f32| {
            let uni = &self.uniforms[u];
            u += 1;
            uni.write(ctx, &BloomUniform { texel: [1.0 / src_size[0] as f32, 1.0 / src_size[1] as f32], threshold, knee: 0.5 * threshold.max(0.05), intensity, pad: [0.0; 3] });
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bloom bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: uni.buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(src) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            });
            let load = if clear { wgpu::LoadOp::Clear(wgpu::Color::BLACK) } else { wgpu::LoadOp::Load };
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: dst, resolve_target: None, ops: wgpu::Operations { load, store: wgpu::StoreOp::Store }, depth_slice: None })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        };
        let chain = &self.chain;
        draw(enc, &self.prefilter, frame, size, &chain[0].0, true, 0.0);
        for k in 0..LEVELS - 1 {
            draw(enc, &self.down, &chain[k].0, chain[k].1, &chain[k + 1].0, true, 0.0);
        }
        for k in (0..LEVELS - 1).rev() {
            draw(enc, &self.up, &chain[k + 1].0, chain[k + 1].1, &chain[k].0, false, 0.0);
        }
        draw(enc, &self.composite, &chain[0].0, chain[0].1, target, false, intensity);
    }
}
