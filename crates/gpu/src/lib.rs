//! wgpu plumbing shared by the renderer and (later) the GPU solver. Nothing here knows about physics.
pub mod volume;
pub use volume::Volume3D;
use std::marker::PhantomData;

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// One device + queue. `Clone` shares the same device (all four handles are reference-counted):
/// the viewer's render thread and the sim worker's GPU solver use clones of one context, so the
/// solver's density texture can be bound by the renderer.
#[derive(Clone)]
pub struct GpuContext {
    pub instance: wgpu::Instance, pub adapter: wgpu::Adapter, pub device: wgpu::Device, pub queue: wgpu::Queue,
    /// `Features::FLOAT32_FILTERABLE` was enabled on `device` (requested whenever the adapter
    /// offers it): `R32Float` textures such as `Volume3D` can then be sampled through a linear
    /// sampler. Without it they are read with `textureLoad` only.
    pub float32_filterable: bool,
}

impl GpuContext {
    fn request(instance: wgpu::Instance, surface: Option<&wgpu::Surface<'_>>) -> Option<GpuContext> {
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: surface,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })).map_err(|e| eprintln!("gpu: request_adapter failed: {e}")).ok()?;
        let optional = adapter.features() & wgpu::Features::FLOAT32_FILTERABLE;
        // The solver's fused Gram kernel (`qm_gpu::reduce`) binds up to 15 fields plus its
        // partials buffer in one dispatch; WebGPU's default is 8 storage buffers per stage. Ask
        // for what the adapter offers up to 16, never below the default (the kernel adapts).
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffers_per_shader_stage = adapter.limits().max_storage_buffers_per_shader_stage.clamp(8, 16);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("atomic-sandbox"),
            required_features: optional,
            required_limits: limits,
            ..Default::default()
        })).map_err(|e| eprintln!("gpu: request_device failed: {e}")).ok()?;
        let float32_filterable = device.features().contains(wgpu::Features::FLOAT32_FILTERABLE);
        Some(GpuContext { instance, adapter, device, queue, float32_filterable })
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
    /// Reconfigures the surface (and the depth texture) at `width × height`. No-op for 0 (minimised).
    ///
    /// `Err` means wgpu refused the reconfigure; the surface, `config` and `depth` are then left
    /// exactly as they were and the caller retries on a later frame. The refusal that matters is
    /// the shared-device race: `Surface::configure` first waits for the device to go idle, and
    /// wgpu 30 reports `ConfigureSurfaceError::GpuWaitTimeout` ("Failed to wait for GPU to come
    /// idle before reconfiguring the Surface") when the queue is non-empty again after that wait,
    /// i.e. another thread submitted meanwhile (wgpu-core `device/resource.rs`, the
    /// `PollStatus::WaitSucceeded` arm). The sim worker does exactly that whenever it samples or
    /// solves, so a resize can land on it; wgpu's default handler would turn the resulting
    /// validation error into a panic. It is captured here in a `Validation` error scope instead.
    /// wgpu's error scopes are per thread (`ErrorSinkRaw::scopes` is keyed by thread id), so the
    /// worker's own submissions are never caught by this scope. Verified by
    /// `crates/app/tests/surface_reconfigure.rs`.
    pub fn resize(&mut self, ctx: &GpuContext, width: u32, height: u32) -> Result<(), wgpu::Error> {
        if width == 0 || height == 0 { return Ok(()); }
        let previous = (self.config.width, self.config.height);
        self.config.width = width; self.config.height = height;
        let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        self.surface.configure(&ctx.device, &self.config);
        if let Some(err) = pollster::block_on(scope.pop()) {
            (self.config.width, self.config.height) = previous;
            return Err(err);
        }
        self.depth = Self::make_depth(ctx, width, height);
        Ok(())
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
