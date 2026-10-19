//! The display warp (user request 2026-10-20): a field solved at one geometry (the last DFT
//! density, an orbital payload) drawn as if it followed the atoms to where they are now. Display
//! only — a warp moves what was solved, it cannot make new chemistry (a bond that broke since the
//! solve is drawn stretched until the next solve), and nothing here reaches the solver.
//!
//! The displacement of a point is the inverse-distance (Shepard) average of the atoms' moves,
//! w_i = 1 / (|x − a_i|² + A²)², so a point near an atom moves with it, a point mid-bond with
//! both ends, and the far field with the molecule as a whole. Two `WARP_N`³ rgba16float textures
//! over the field's box hold it, computed on the CPU (N³ × atoms, well under a millisecond for a
//! molecule): `inv` anchored at the atoms now, read by the raymarches (a point x now samples the
//! field at x − d(x)); `fwd` anchored at the solve's positions, read by the dots, which are drawn
//! from the solved field and moved to x + d(x). Group 2 of the volume, colour-volume and dots
//! pipelines ([`layout`]); the passes hold a clone of [`Warp::bind_group`], and [`Warp::set`]
//! rewrites the textures in place.
use gpu::{GpuContext, UniformBuffer};

/// Points per axis of the displacement textures.
pub const WARP_N: u32 = 32;
/// The weights' softening (Bohr): A in w = 1 / (r² + A²)².
const SOFTEN: f32 = 0.3;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WarpUniform {
    pub box_min: [f32; 3],
    pub on: u32,
    pub box_max: [f32; 3],
    pub pad: f32,
}

/// Group 2 of the pipelines that read the warp: the uniform, `inv`, `fwd`, a linear sampler.
pub fn layout(dev: &wgpu::Device) -> wgpu::BindGroupLayout {
    let vf = wgpu::ShaderStages::VERTEX_FRAGMENT;
    let tex = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vf,
        ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false },
        count: None,
    };
    dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("warp bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: vf, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            tex(1),
            tex(2),
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: vf, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
        ],
    })
}

/// The WGSL of group 2 (appended to the shaders that read it): `warp_back(p)` (the raymarches)
/// and `warp_forward(p)` (the dots); both the identity while the warp is off.
pub const WGSL: &str = include_str!("shaders/warp.wgsl");

pub struct Warp {
    inv: wgpu::Texture,
    fwd: wgpu::Texture,
    uniform: UniformBuffer<WarpUniform>,
    bind_group: wgpu::BindGroup,
    /// What the textures hold now (to skip identical rewrites).
    last: Option<(Vec<[f32; 3]>, Vec<[f32; 3]>, [f32; 3], [f32; 3])>,
}

impl Warp {
    /// Off (the identity) until [`set`](Self::set).
    pub fn new(ctx: &GpuContext) -> Warp {
        let dev = &ctx.device;
        let make = |label| dev.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: WARP_N, height: WARP_N, depth_or_array_layers: WARP_N },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let (inv, fwd) = (make("warp inv"), make("warp fwd"));
        let sampler = dev.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("warp sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniform = UniformBuffer::<WarpUniform>::new(ctx, "warp");
        uniform.write(ctx, &WarpUniform::default());
        let (vi, vf) = (inv.create_view(&Default::default()), fwd.create_view(&Default::default()));
        let bind_group = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("warp bg"),
            layout: &layout(dev),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&vi) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&vf) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });
        Warp { inv, fwd, uniform, bind_group, last: None }
    }

    /// The bind group the passes set as group 2.
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// The field over the box `box_min..box_max` (its first and last grid points) was solved with
    /// the atoms at `solved`; they are at `now` (same atoms, same order). Different counts, or no
    /// atoms, turn the warp off.
    pub fn set(&mut self, ctx: &GpuContext, box_min: [f32; 3], box_max: [f32; 3], solved: &[[f32; 3]], now: &[[f32; 3]]) {
        if solved.len() != now.len() || now.is_empty() || box_max.iter().zip(&box_min).any(|(a, b)| a <= b) {
            self.off(ctx);
            return;
        }
        let key = (solved.to_vec(), now.to_vec(), box_min, box_max);
        if self.last.as_ref() == Some(&key) {
            return;
        }
        let moved: Vec<[f32; 3]> = now.iter().zip(solved).map(|(n, s)| [n[0] - s[0], n[1] - s[1], n[2] - s[2]]).collect();
        let write = |tex: &wgpu::Texture, anchors: &[[f32; 3]]| {
            let data = displacement(box_min, box_max, anchors, &moved);
            ctx.queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                bytemuck::cast_slice(&data),
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(WARP_N * 8), rows_per_image: Some(WARP_N) },
                wgpu::Extent3d { width: WARP_N, height: WARP_N, depth_or_array_layers: WARP_N },
            );
        };
        write(&self.inv, now);
        write(&self.fwd, solved);
        self.uniform.write(ctx, &WarpUniform { box_min, on: 1, box_max, pad: 0.0 });
        self.last = Some(key);
    }

    /// The identity: fields are drawn where they were solved.
    pub fn off(&mut self, ctx: &GpuContext) {
        if self.last.take().is_some() {
            self.uniform.write(ctx, &WarpUniform::default());
        }
    }
}

/// The displacement at every texel (half floats, rgba with a = 0): the Shepard average of `moved`
/// weighted by the distance to `anchors`.
fn displacement(box_min: [f32; 3], box_max: [f32; 3], anchors: &[[f32; 3]], moved: &[[f32; 3]]) -> Vec<u16> {
    let n = WARP_N as usize;
    let step: [f32; 3] = std::array::from_fn(|k| (box_max[k] - box_min[k]) / (n - 1) as f32);
    let mut out = Vec::with_capacity(n * n * n * 4);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let x = [box_min[0] + i as f32 * step[0], box_min[1] + j as f32 * step[1], box_min[2] + k as f32 * step[2]];
                let (mut num, mut den) = ([0.0f32; 3], 0.0f32);
                for (a, d) in anchors.iter().zip(moved) {
                    let r2 = (x[0] - a[0]).powi(2) + (x[1] - a[1]).powi(2) + (x[2] - a[2]).powi(2);
                    let w = 1.0 / (r2 + SOFTEN * SOFTEN).powi(2);
                    den += w;
                    for c in 0..3 {
                        num[c] += w * d[c];
                    }
                }
                for c in 0..3 {
                    out.push(f16_bits(num[c] / den));
                }
                out.push(0);
            }
        }
    }
    out
}

/// f32 → IEEE half bits (round toward zero; subnormals flush to zero — displacements are Bohr-sized).
fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    if !x.is_finite() || e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        return sign;
    }
    sign | ((e as u16) << 10) | ((b >> 13) & 0x3ff) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn half(h: u16) -> f32 {
        let e = ((h >> 10) & 0x1f) as i32;
        let m = (h & 0x3ff) as f32;
        let v = if e == 0 { 0.0 } else { (1.0 + m / 1024.0) * 2f32.powi(e - 15) };
        if h & 0x8000 != 0 { -v } else { v }
    }

    /// A rigid move moves every point by the same amount; next to one atom of two that moved apart,
    /// a point follows that atom.
    #[test]
    fn the_displacement_follows_the_atoms() {
        let (lo, hi) = ([-4.0f32; 3], [4.0f32; 3]);
        let rigid = displacement(lo, hi, &[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], &[[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]);
        assert!(rigid.chunks(4).all(|t| (half(t[0]) - 0.5).abs() < 1e-3 && half(t[1]) == 0.0));
        let apart = displacement(lo, hi, &[[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], &[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        // texel nearest (−2, 0, 0): i = (−2 + 4) / (8/31) ≈ 7.75 → 8, j = k = 15.5 → 15
        let at = |i: usize, j: usize, k: usize| half(apart[((k * 32 + j) * 32 + i) * 4]);
        assert!(at(8, 15, 15) < -0.9, "{}", at(8, 15, 15));
        assert!(at(23, 15, 15) > 0.9, "{}", at(23, 15, 15));
        // mid-way the two moves cancel: texels 15 and 16 straddle x = 0 symmetrically
        assert!((at(15, 15, 15) + at(16, 15, 15)).abs() < 1e-2 && at(16, 15, 15).abs() < 0.3, "mid-way: {} {}", at(15, 15, 15), at(16, 15, 15));
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(-2.5), 0xc100);
    }
}
