//! The display warp (user request 2026-10-20): a field solved at one geometry (the last DFT
//! density, an orbital payload) drawn as if it followed the atoms to where they are now. Display
//! only — a warp moves what was solved, it cannot make new chemistry (a bond that broke since the
//! solve is drawn stretched until the next solve), and nothing here reaches the solver.
//!
//! Skinning (user 2026-10-20: a translation-only average sheared the cloud when bonds twisted): every
//! atom carries a rigid motion — its own move and the rotation of its neighbourhood, from a frame
//! built on its nearest neighbours at the solve and now ([`atom_frames`]) — and a point moves by the
//! inverse-distance (Shepard, w_i = 1 / (|x − a_i|² + A²)²) blend of those motions with each atom
//! as its own pivot, as a skeleton bends a mesh: near an atom it turns with that atom's bonds,
//! mid-bond it blends both ends, and a molecule that turns rigidly turns its field exactly (every
//! atom then has the same rotation). Two `WARP_N`³ rgba16float textures
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
        let rot = atom_frames(solved, now);
        let write = |tex: &wgpu::Texture, forward: bool| {
            let data = displacement(box_min, box_max, solved, now, &rot, forward);
            ctx.queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                bytemuck::cast_slice(&data),
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(WARP_N * 8), rows_per_image: Some(WARP_N) },
                wgpu::Extent3d { width: WARP_N, height: WARP_N, depth_or_array_layers: WARP_N },
            );
        };
        write(&self.inv, false);
        write(&self.fwd, true);
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

/// Neighbours (within this, at the solve) that set an atom's frame (Bohr).
const NEIGHBOUR_RADIUS: f32 = 3.6;

/// Each atom's rotation from the solve to now: the frame of its nearest neighbour (e1) and the
/// next one off that line (e2, e3 = e1 × e2) at both geometries, R = F_now · F_solvedᵀ; with one
/// usable neighbour the shortest arc turning e1; with none, no rotation.
fn atom_frames(solved: &[[f32; 3]], now: &[[f32; 3]]) -> Vec<glam::Mat3> {
    use glam::{Mat3, Quat, Vec3};
    let v = |p: &[f32; 3]| Vec3::from_array(*p);
    (0..solved.len())
        .map(|i| {
            let (si, ni) = (v(&solved[i]), v(&now[i]));
            let mut near: Vec<(f32, usize)> = (0..solved.len())
                .filter(|&j| j != i)
                .map(|j| (v(&solved[j]).distance(si), j))
                .filter(|(d, _)| *d < NEIGHBOUR_RADIUS && *d > 1e-3)
                .collect();
            near.sort_by(|a, b| a.0.total_cmp(&b.0));
            let Some(&(_, a)) = near.first() else { return Mat3::IDENTITY };
            let (s1, n1) = ((v(&solved[a]) - si).normalize(), (v(&now[a]) - ni).normalize_or_zero());
            if n1 == Vec3::ZERO {
                return Mat3::IDENTITY;
            }
            // the next neighbour that is clearly off the first bond's line (> ~15°)
            let second = near[1..].iter().find_map(|&(_, b)| {
                let (sb, nb) = (v(&solved[b]) - si, v(&now[b]) - ni);
                let (s2, n2) = (sb - s1 * sb.dot(s1), nb - n1 * nb.dot(n1));
                (s2.length() > 0.26 * sb.length() && n2.length() > 0.26 * nb.length()).then(|| (s2.normalize(), n2.normalize()))
            });
            match second {
                Some((s2, n2)) => {
                    let fs = Mat3::from_cols(s1, s2, s1.cross(s2));
                    let fn_ = Mat3::from_cols(n1, n2, n1.cross(n2));
                    fn_ * fs.transpose()
                }
                None => Mat3::from_quat(Quat::from_rotation_arc(s1, n1)),
            }
        })
        .collect()
}

/// The displacement at every texel (half floats, rgba with a = 0). `forward` (the dots' texture):
/// a point of the solved field moves to Σ w_i [R_i (x − s_i) + n_i] (weights by the distance to the
/// solve's positions s); otherwise (the raymarches') a point now came from
/// Σ w_i [R_iᵀ (x − n_i) + s_i] (weights by the distance to the positions now n), stored as
/// x − that.
fn displacement(box_min: [f32; 3], box_max: [f32; 3], solved: &[[f32; 3]], now: &[[f32; 3]], rot: &[glam::Mat3], forward: bool) -> Vec<u16> {
    use glam::Vec3;
    let n = WARP_N as usize;
    let step: [f32; 3] = std::array::from_fn(|k| (box_max[k] - box_min[k]) / (n - 1) as f32);
    let (from, to): (Vec<Vec3>, Vec<Vec3>) = if forward {
        (solved.iter().map(|p| Vec3::from_array(*p)).collect(), now.iter().map(|p| Vec3::from_array(*p)).collect())
    } else {
        (now.iter().map(|p| Vec3::from_array(*p)).collect(), solved.iter().map(|p| Vec3::from_array(*p)).collect())
    };
    let rot: Vec<glam::Mat3> = if forward { rot.to_vec() } else { rot.iter().map(|r| r.transpose()).collect() };
    let mut out = Vec::with_capacity(n * n * n * 4);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let x = Vec3::new(box_min[0] + i as f32 * step[0], box_min[1] + j as f32 * step[1], box_min[2] + k as f32 * step[2]);
                let (mut num, mut den) = (Vec3::ZERO, 0.0f32);
                for ((a, b), r) in from.iter().zip(&to).zip(&rot) {
                    let w = 1.0 / (x.distance_squared(*a) + SOFTEN * SOFTEN).powi(2);
                    den += w;
                    num += w * (*r * (x - *a) + *b);
                }
                let mapped = num / den;
                let d = if forward { mapped - x } else { x - mapped };
                out.extend([f16_bits(d.x), f16_bits(d.y), f16_bits(d.z), 0]);
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

    /// A rigid move moves every point by the same amount; a molecule turned rigidly turns its field
    /// exactly (every atom has the same rotation, each about itself); both directions agree.
    #[test]
    fn the_displacement_follows_the_atoms() {
        let (lo, hi) = ([-4.0f32; 3], [4.0f32; 3]);
        let mol = [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [-0.45, 1.75, 0.0]];
        let shifted: Vec<[f32; 3]> = mol.iter().map(|p| [p[0] + 0.5, p[1], p[2]]).collect();
        let rot = atom_frames(&mol, &shifted);
        let rigid = displacement(lo, hi, &mol, &shifted, &rot, true);
        assert!(rigid.chunks(4).all(|t| (half(t[0]) - 0.5).abs() < 2e-3 && half(t[1]).abs() < 2e-3), "a translation is uniform");
        // 90° about z through the origin
        let turn = |p: &[f32; 3]| [-p[1], p[0], p[2]];
        let turned: Vec<[f32; 3]> = mol.iter().map(turn).collect();
        let rot = atom_frames(&mol, &turned);
        let fwd = displacement(lo, hi, &mol, &turned, &rot, true);
        let inv = displacement(lo, hi, &mol, &turned, &rot, false);
        let step = 8.0 / 31.0;
        for (i, j, k) in [(16, 16, 16), (20, 12, 15), (10, 22, 18)] {
            let x = [-4.0 + i as f32 * step, -4.0 + j as f32 * step, -4.0 + k as f32 * step];
            let t = ((k * 32 + j) * 32 + i) * 4;
            let want = turn(&x);
            for c in 0..3 {
                assert!((x[c] + half(fwd[t + c]) - want[c]).abs() < 0.02, "forward {c} at {x:?}");
                // the inverse: the point now at x came from the point that turns onto x
                let back = [x[1], -x[0], x[2]];
                assert!((x[c] - half(inv[t + c]) - back[c]).abs() < 0.02, "back {c} at {x:?}");
            }
        }
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(-2.5), 0xc100);
    }
}
