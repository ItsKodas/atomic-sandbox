//! Render-to-texture harness for tests: Rgba8Unorm colour + Depth32Float, with CPU readback.
//! Public (not `cfg(test)`) so the app crate's tests can reuse it.
use gpu::{GpuContext, DEPTH_FORMAT};

/// Linear (non-sRGB) so readback bytes are the shader's output values ×255.
pub const HEADLESS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

pub struct Headless { pub ctx: GpuContext, pub colour: wgpu::Texture, pub colour_view: wgpu::TextureView, pub depth: wgpu::TextureView, pub width: u32, pub height: u32 }

impl Headless {
    /// `None` when no GPU adapter is available (tests then skip).
    pub fn new(width: u32, height: u32) -> Option<Headless> {
        let ctx = GpuContext::new_headless()?;
        let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
        let colour = ctx.device.create_texture(&wgpu::TextureDescriptor { label: Some("headless colour"), size, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: HEADLESS_FORMAT, usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC, view_formats: &[] });
        let colour_view = colour.create_view(&Default::default());
        let depth = ctx.device.create_texture(&wgpu::TextureDescriptor { label: Some("headless depth"), size, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: DEPTH_FORMAT, usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING, view_formats: &[] }).create_view(&Default::default());
        Some(Headless { ctx, colour, colour_view, depth, width, height })
    }

    /// Begins a pass that clears colour to `clear` and depth to 1.0 (far).
    pub fn begin<'a>(&'a self, enc: &'a mut wgpu::CommandEncoder, clear: wgpu::Color) -> wgpu::RenderPass<'a> {
        enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("headless"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &self.colour_view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Store }, depth_slice: None })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &self.depth, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
            timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
        })
    }

    /// Colour target as RGBA8, row-major, top-left first. Blocks until the GPU is done.
    pub fn read_rgba(&self) -> Vec<[u8; 4]> {
        let row = 4 * self.width;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = row.div_ceil(align) * align;
        let buf = self.ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("readback"), size: padded as u64 * self.height as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &self.colour, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(self.height) } },
            wgpu::Extent3d { width: self.width, height: self.height, depth_or_array_layers: 1 });
        self.ctx.queue.submit([enc.finish()]);
        let slice = buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.ctx.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        rx.recv().expect("map callback").expect("map ok");
        let bytes = slice.get_mapped_range().expect("Headless::read_rgba: mapped range");
        let mut out = Vec::with_capacity((self.width * self.height) as usize);
        for y in 0..self.height { for x in 0..self.width {
            let i = (y * padded + 4 * x) as usize;
            out.push([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
        }}
        drop(bytes); buf.unmap();
        out
    }
}
