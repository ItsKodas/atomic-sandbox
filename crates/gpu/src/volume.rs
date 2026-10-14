//! 3D r32float texture holding a scalar field in grid layout i + nx*(j + ny*k) (x fastest).
use crate::GpuContext;

/// `Send + Sync` (wgpu handles): the sim worker creates these on its thread, copies the solver's
/// density into one and hands it to the render thread inside an `Arc`.
pub struct Volume3D { pub texture: wgpu::Texture, pub view: wgpu::TextureView, pub sampler: wgpu::Sampler, pub dims: [u32; 3] }

impl Volume3D {
    /// r32float 3D texture, linear sampler, clamp-to-edge. The sampler is usable only on a
    /// device with `GpuContext::float32_filterable`; otherwise read the texture with `textureLoad`.
    pub fn new(ctx: &GpuContext, dims: [u32; 3]) -> Volume3D {
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("density volume"),
            size: wgpu::Extent3d { width: dims[0], height: dims[1], depth_or_array_layers: dims[2] },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volume sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge, address_mode_v: wgpu::AddressMode::ClampToEdge, address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default()
        });
        Volume3D { texture, view, sampler, dims }
    }

    /// Number of voxels (computed in usize so large grids cannot overflow u32).
    pub fn len(&self) -> usize { self.dims.iter().map(|&d| d as usize).product() }

    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Panics with a clear message if data.len() != dims product.
    pub fn upload(&self, ctx: &GpuContext, data: &[f32]) {
        assert_eq!(data.len(), self.len(), "Volume3D::upload: data length {} != dims {:?}", data.len(), self.dims);
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &self.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(data),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * self.dims[0]), rows_per_image: Some(self.dims[1]) },
            wgpu::Extent3d { width: self.dims[0], height: self.dims[1], depth_or_array_layers: self.dims[2] },
        );
    }

    /// Headless readback for tests. Rows are padded to COPY_BYTES_PER_ROW_ALIGNMENT in the staging buffer.
    pub fn read_back(&self, ctx: &GpuContext) -> Vec<f32> {
        let row_bytes = 4 * self.dims[0];
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = row_bytes.div_ceil(align) * align;
        let size = padded as u64 * self.dims[1] as u64 * self.dims[2] as u64;
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volume readback"), size, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &self.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &staging, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(self.dims[1]) } },
            wgpu::Extent3d { width: self.dims[0], height: self.dims[1], depth_or_array_layers: self.dims[2] },
        );
        ctx.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        ctx.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        rx.recv().expect("map callback").expect("map ok");
        let bytes = slice.get_mapped_range().expect("Volume3D::read_back: mapped range");
        let mut out = Vec::with_capacity(self.len());
        for z in 0..self.dims[2] { for y in 0..self.dims[1] {
            let start = (z as usize * self.dims[1] as usize + y as usize) * padded as usize;
            out.extend_from_slice(bytemuck::cast_slice(&bytes[start..start + row_bytes as usize]));
        }}
        drop(bytes); staging.unmap();
        out
    }
}

/// 3D rgba16float texture (the orbital viewer's colour volume: rgb = Σ fᵢcᵢsᵢ, a = Σ fᵢsᵢ),
/// grid layout as [`Volume3D`]. rgba16float is filterable on every device, so the sampler
/// (linear, clamp-to-edge) always works. Usage STORAGE_BINDING (the solver's
/// `colour_accumulate` kernel writes it with `textureStore`), TEXTURE_BINDING (the renderer
/// samples it), COPY_DST (`upload_f32`, the CPU backend's path) and COPY_SRC (`read_back`).
/// `Send + Sync` like `Volume3D`.
pub struct ColourVolume3D { pub texture: wgpu::Texture, pub view: wgpu::TextureView, pub sampler: wgpu::Sampler, pub dims: [u32; 3] }

/// Bytes per rgba16float texel.
const RGBA16F_BYTES: u32 = 8;

impl ColourVolume3D {
    pub fn new(ctx: &GpuContext, dims: [u32; 3]) -> Self {
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("colour volume"),
            size: wgpu::Extent3d { width: dims[0], height: dims[1], depth_or_array_layers: dims[2] },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("colour volume sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge, address_mode_v: wgpu::AddressMode::ClampToEdge, address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default()
        });
        ColourVolume3D { texture, view, sampler, dims }
    }

    /// Number of voxels.
    pub fn len(&self) -> usize { self.dims.iter().map(|&d| d as usize).product() }

    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Uploads f32 RGBA (grid layout), rounded to f16 (round to nearest even, `half`).
    pub fn upload_f32(&self, ctx: &GpuContext, rgba: &[[f32; 4]]) {
        assert_eq!(rgba.len(), self.len(), "ColourVolume3D::upload_f32: data length {} != dims {:?}", rgba.len(), self.dims);
        let bits: Vec<u16> = rgba.iter().flat_map(|v| v.map(|c| half::f16::from_f32(c).to_bits())).collect();
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &self.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&bits),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(RGBA16F_BYTES * self.dims[0]), rows_per_image: Some(self.dims[1]) },
            wgpu::Extent3d { width: self.dims[0], height: self.dims[1], depth_or_array_layers: self.dims[2] },
        );
    }

    /// Headless readback for tests: f16 texels widened to f32 (exact).
    pub fn read_back(&self, ctx: &GpuContext) -> Vec<[f32; 4]> {
        let row_bytes = RGBA16F_BYTES * self.dims[0];
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = row_bytes.div_ceil(align) * align;
        let size = padded as u64 * self.dims[1] as u64 * self.dims[2] as u64;
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("colour volume readback"), size, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &self.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &staging, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(self.dims[1]) } },
            wgpu::Extent3d { width: self.dims[0], height: self.dims[1], depth_or_array_layers: self.dims[2] },
        );
        ctx.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        ctx.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        rx.recv().expect("map callback").expect("map ok");
        let bytes = slice.get_mapped_range().expect("ColourVolume3D::read_back: mapped range");
        let mut out = Vec::with_capacity(self.len());
        for z in 0..self.dims[2] { for y in 0..self.dims[1] {
            let start = (z as usize * self.dims[1] as usize + y as usize) * padded as usize;
            let row: &[u16] = bytemuck::cast_slice(&bytes[start..start + row_bytes as usize]);
            out.extend(row.as_chunks::<4>().0.iter().map(|t| t.map(|b| half::f16::from_bits(b).to_f32())));
        }}
        drop(bytes); staging.unmap();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GpuContext;

    fn ctx() -> Option<GpuContext> {
        let c = GpuContext::new_headless();
        if c.is_none() { eprintln!("no GPU adapter: skipping"); }
        c
    }

    #[test]
    fn upload_then_read_back_is_identity() {
        let Some(ctx) = ctx() else { return };
        let dims = [5u32, 7, 3];
        let n = (dims[0] * dims[1] * dims[2]) as usize;
        let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 3.0).collect();
        let vol = Volume3D::new(&ctx, dims);
        vol.upload(&ctx, &data);
        assert_eq!(vol.read_back(&ctx), data);
    }

    #[test]
    #[should_panic(expected = "Volume3D::upload")]
    fn upload_with_wrong_length_panics() {
        let Some(ctx) = ctx() else { panic!("Volume3D::upload (no adapter; forced)") };
        let vol = Volume3D::new(&ctx, [4, 4, 4]);
        vol.upload(&ctx, &[0.0; 10]);
    }

    /// rgba16float round trip: f16-exact values come back exactly, others within half an f16
    /// ulp (rel 2⁻¹¹); x is fastest; a non-multiple-of-32 width exercises the row padding.
    #[test]
    fn colour_volume_upload_then_read_back() {
        let Some(ctx) = ctx() else { return };
        let dims = [7u32, 5, 3];
        let n = (dims[0] * dims[1] * dims[2]) as usize;
        let data: Vec<[f32; 4]> = (0..n).map(|i| [i as f32 * 0.25, -(i as f32), 1.0 / (1.0 + i as f32), 0.1 * i as f32]).collect();
        let vol = ColourVolume3D::new(&ctx, dims);
        vol.upload_f32(&ctx, &data);
        let back = vol.read_back(&ctx);
        assert_eq!(back.len(), n);
        for (b, d) in back.iter().zip(&data) {
            assert_eq!(b[0], d[0]); // multiples of 1/4 below 2048 are exact in f16
            assert_eq!(b[1], d[1]);
            for c in [2, 3] { assert!((b[c] - d[c]).abs() <= d[c].abs() * 4.9e-4 + 1e-7, "{b:?} vs {d:?}"); }
        }
    }

    #[test]
    fn layout_is_x_fastest() {
        let Some(ctx) = ctx() else { return };
        let dims = [3u32, 2, 2];
        let mut data = vec![0.0f32; 12];
        data[1] = 1.0;           // i=1, j=0, k=0
        data[3] = 2.0;           // i=0, j=1, k=0
        data[6] = 3.0;           // i=0, j=0, k=1
        let vol = Volume3D::new(&ctx, dims);
        vol.upload(&ctx, &data);
        let back = vol.read_back(&ctx);
        assert_eq!(back[1], 1.0); assert_eq!(back[3], 2.0); assert_eq!(back[6], 3.0);
    }
}
