//! Frames from the GPU without waiting for it: each capture copies its texture into a mappable
//! buffer in the frame's own submission, the map is requested after the submit, and later frames
//! collect whatever has arrived ([`Pool::collect`], a non-blocking device poll). A few buffers are
//! in flight at most; a capture beyond that is refused (the caller counts it as dropped).
use super::writer::Image;
use gpu::GpuContext;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

/// Copies in flight at most (a 4K frame's buffer is 33 MB).
pub const IN_FLIGHT: usize = 6;

const WAITING: u8 = 0;
const MAPPED: u8 = 1;
const FAILED: u8 = 2;

/// Where a frame read back goes.
#[derive(Clone, Debug, PartialEq)]
pub enum Tag {
    Screenshot(std::path::PathBuf),
    /// Frame `index` of recording `session`.
    Frame { session: u64, index: u64 },
}

struct InFlight {
    buf: wgpu::Buffer,
    size: u64,
    w: u32,
    h: u32,
    padded: u32,
    bgra: bool,
    state: Arc<AtomicU8>,
    requested: bool,
    tag: Tag,
}

#[derive(Default)]
pub struct Pool {
    flying: Vec<InFlight>,
    spare: Vec<(wgpu::Buffer, u64)>,
}

/// A colour format the capture can read: 8-bit RGBA or BGRA (sRGB or not — the bytes are what the
/// window shows either way).
pub fn readable(format: wgpu::TextureFormat) -> Option<bool> {
    use wgpu::TextureFormat as F;
    match format {
        F::Rgba8Unorm | F::Rgba8UnormSrgb => Some(false),
        F::Bgra8Unorm | F::Bgra8UnormSrgb => Some(true),
        _ => None,
    }
}

impl Pool {
    /// Room for one more copy.
    pub fn has_room(&self) -> bool {
        self.flying.len() < IN_FLIGHT
    }

    /// Records the copy of `texture` (`w × h`, `format`) into `enc`; false when the pool is full
    /// or the format is not readable. Call [`Pool::submitted`] after submitting `enc`.
    pub fn copy(&mut self, ctx: &GpuContext, enc: &mut wgpu::CommandEncoder, texture: &wgpu::Texture, w: u32, h: u32, format: wgpu::TextureFormat, tag: Tag) -> bool {
        let Some(bgra) = readable(format) else { return false };
        if !self.has_room() || w == 0 || h == 0 {
            return false;
        }
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = (4 * w).div_ceil(align) * align;
        let size = padded as u64 * h as u64;
        let buf = match self.spare.iter().position(|(_, s)| *s == size) {
            Some(k) => self.spare.swap_remove(k).0,
            None => ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("capture readback"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
        };
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(h) } },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.flying.push(InFlight { buf, size, w, h, padded, bgra, state: Arc::new(AtomicU8::new(WAITING)), requested: false, tag });
        true
    }

    /// The copies recorded since the last call were submitted: request their maps.
    pub fn submitted(&mut self) {
        for f in self.flying.iter_mut().filter(|f| !f.requested) {
            f.requested = true;
            let state = f.state.clone();
            f.buf.slice(..).map_async(wgpu::MapMode::Read, move |r| state.store(if r.is_ok() { MAPPED } else { FAILED }, Ordering::Release));
        }
    }

    /// The frames that have arrived (in capture order), without blocking.
    pub fn collect(&mut self, ctx: &GpuContext) -> Vec<(Tag, Result<Image, String>)> {
        if self.flying.is_empty() {
            return Vec::new();
        }
        ctx.device.poll(wgpu::PollType::Poll).ok();
        let mut out = Vec::new();
        while self.flying.first().is_some_and(|f| f.requested && f.state.load(Ordering::Acquire) != WAITING) {
            let f = self.flying.remove(0);
            let img = if f.state.load(Ordering::Acquire) == MAPPED {
                let img = match f.buf.slice(..).get_mapped_range() {
                    Ok(view) => Ok(unpad(&view, f.w, f.h, f.padded, f.bgra)),
                    Err(e) => Err(e.to_string()),
                };
                f.buf.unmap();
                img
            } else {
                Err("the GPU did not deliver the frame".into())
            };
            self.spare.push((f.buf, f.size));
            self.spare.truncate(IN_FLIGHT);
            out.push((f.tag, img));
        }
        out
    }

    /// Frames still on their way.
    pub fn pending(&self) -> usize {
        self.flying.len()
    }

    /// Frames of recording `session` still on their way.
    pub fn pending_for(&self, session: u64) -> bool {
        self.flying.iter().any(|f| matches!(f.tag, Tag::Frame { session: s, .. } if s == session))
    }
}

/// Tightly packed RGBA from padded rows (BGRA swapped), alpha forced opaque.
fn unpad(bytes: &[u8], w: u32, h: u32, padded: u32, bgra: bool) -> Image {
    let mut rgba = Vec::with_capacity((4 * w * h) as usize);
    for y in 0..h as usize {
        let row = &bytes[y * padded as usize..][..4 * w as usize];
        for p in row.chunks_exact(4) {
            let (r, b) = if bgra { (p[2], p[0]) } else { (p[0], p[2]) };
            rgba.extend_from_slice(&[r, p[1], b, 255]);
        }
    }
    Image { w, h, rgba }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_unpadded_and_bgra_swapped() {
        // 2 × 2, rows padded to 12 bytes
        let bytes = [1, 2, 3, 9, 4, 5, 6, 9, 0, 0, 0, 0, 7, 8, 9, 9, 10, 11, 12, 9, 0, 0, 0, 0];
        let img = unpad(&bytes, 2, 2, 12, true);
        assert_eq!(img.rgba, vec![3, 2, 1, 255, 6, 5, 4, 255, 9, 8, 7, 255, 12, 11, 10, 255]);
        assert_eq!(unpad(&bytes, 2, 2, 12, false).rgba[..4], [1, 2, 3, 255]);
    }
}
