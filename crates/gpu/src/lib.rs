//! wgpu plumbing shared by the renderer and (later) the GPU solver. Nothing here knows about physics.
pub mod volume;
pub use volume::Volume3D;
use std::marker::PhantomData;

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub struct GpuContext { pub instance: wgpu::Instance, pub adapter: wgpu::Adapter, pub device: wgpu::Device, pub queue: wgpu::Queue }

impl GpuContext {
    fn request(instance: wgpu::Instance, surface: Option<&wgpu::Surface<'_>>) -> Option<GpuContext> {
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: surface,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })).map_err(|e| eprintln!("gpu: request_adapter failed: {e}")).ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("atomic-sandbox"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })).map_err(|e| eprintln!("gpu: request_device failed: {e}")).ok()?;
        Some(GpuContext { instance, adapter, device, queue })
    }
    /// Headless (no surface). Returns None when no adapter is available.
    pub fn new_headless() -> Option<GpuContext> {
        Self::request(wgpu::Instance::default(), None)
    }
    /// For a window: creates the surface first so the adapter is compatible with it.
    pub fn new_for_surface(target: wgpu::SurfaceTarget<'static>) -> Option<(GpuContext, wgpu::Surface<'static>)> {
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(target).ok()?;
        let ctx = Self::request(instance, Some(&surface))?;
        Some((ctx, surface))
    }
}

pub struct SurfaceState { pub surface: wgpu::Surface<'static>, pub config: wgpu::SurfaceConfiguration, pub depth: wgpu::TextureView }

impl SurfaceState {
    pub fn new(ctx: &GpuContext, surface: wgpu::Surface<'static>, width: u32, height: u32) -> SurfaceState {
        let caps = surface.get_capabilities(&ctx.adapter);
        let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format,
            width: width.max(1), height: height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync, alpha_mode: caps.alpha_modes[0],
            view_formats: vec![], desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&ctx.device, &config);
        let depth = Self::make_depth(ctx, config.width, config.height);
        SurfaceState { surface, config, depth }
    }
    fn make_depth(ctx: &GpuContext, w: u32, h: u32) -> wgpu::TextureView {
        ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"), size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING, view_formats: &[],
        }).create_view(&Default::default())
    }
    /// No-op for width or height == 0 (minimised window).
    pub fn resize(&mut self, ctx: &GpuContext, width: u32, height: u32) {
        if width == 0 || height == 0 { return; }
        self.config.width = width; self.config.height = height;
        self.surface.configure(&ctx.device, &self.config);
        self.depth = Self::make_depth(ctx, width, height);
    }
}

pub struct UniformBuffer<T: bytemuck::Pod> { pub buffer: wgpu::Buffer, _t: PhantomData<T> }
impl<T: bytemuck::Pod> UniformBuffer<T> {
    pub fn new(ctx: &GpuContext, label: &str) -> Self {
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label), size: std::mem::size_of::<T>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        UniformBuffer { buffer, _t: PhantomData }
    }
    pub fn write(&self, ctx: &GpuContext, value: &T) { ctx.queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(value)); }
}
