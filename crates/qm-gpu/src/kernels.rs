//! Pipeline cache: every WGSL entry point is compiled once per device, each with an explicit
//! bind-group layout (group 0) described by a `Spec`. Callers build bind groups by passing
//! resources in the order of the spec's binding list, then `record`/`run` a dispatch.
use gpu::GpuContext;
use std::collections::HashMap;

/// Workgroup size of every kernel in this crate (1D, `@workgroup_size(256)`).
pub const WORKGROUP: u32 = 256;
/// Upper bound on workgroups for grid-stride kernels (WebGPU default per-dimension limit).
pub const MAX_WORKGROUPS: u32 = 65535;

/// One binding of group 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot { Uniform, Read, ReadWrite }

struct Spec {
    entry: &'static str,
    shader: &'static str,
    bindings: &'static [(u32, Slot)],
}

use Slot::{Read as R, ReadWrite as RW, Uniform as U};

const SHADERS: &[(&str, &str)] = &[
    ("reduce", include_str!("shaders/reduce.wgsl")),
    ("pointwise", include_str!("shaders/pointwise.wgsl")),
];

/// lincomb: uniform at 4, out at 1, inputs at 5..=11 (7 reads + 1 read_write = 8 storage
/// buffers, the default `max_storage_buffers_per_shader_stage`).
pub const LINCOMB_MAX_TERMS: usize = 7;

const SPECS: &[Spec] = &[
    Spec { entry: "dot_partials", shader: "reduce", bindings: &[(0, U), (1, R), (2, R), (3, RW)] },
    Spec { entry: "sum_partials", shader: "reduce", bindings: &[(0, U), (1, R), (3, RW)] },
    Spec { entry: "axpy", shader: "pointwise", bindings: &[(0, U), (1, RW), (2, R)] },
    Spec { entry: "scale", shader: "pointwise", bindings: &[(0, U), (1, RW)] },
    Spec { entry: "mul", shader: "pointwise", bindings: &[(0, U), (1, RW), (2, R), (3, R)] },
    Spec { entry: "mul_add_sq", shader: "pointwise", bindings: &[(0, U), (1, RW), (2, R)] },
    Spec {
        entry: "lincomb",
        shader: "pointwise",
        bindings: &[(4, U), (1, RW), (5, R), (6, R), (7, R), (8, R), (9, R), (10, R), (11, R)],
    },
];

struct Entry {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    bindings: &'static [(u32, Slot)],
}

pub struct Kernels {
    entries: HashMap<&'static str, Entry>,
}

impl Kernels {
    pub fn new(ctx: &GpuContext) -> Kernels {
        let modules: HashMap<&str, wgpu::ShaderModule> = SHADERS.iter().map(|(name, src)| {
            (*name, ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name), source: wgpu::ShaderSource::Wgsl((*src).into()),
            }))
        }).collect();
        let entries = SPECS.iter().map(|s| {
            let layout_entries: Vec<wgpu::BindGroupLayoutEntry> = s.bindings.iter().map(|&(binding, slot)| {
                let ty = match slot {
                    Slot::Uniform => wgpu::BufferBindingType::Uniform,
                    Slot::Read => wgpu::BufferBindingType::Storage { read_only: true },
                    Slot::ReadWrite => wgpu::BufferBindingType::Storage { read_only: false },
                };
                wgpu::BindGroupLayoutEntry {
                    binding, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None }, count: None,
                }
            }).collect();
            let layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(s.entry), entries: &layout_entries });
            let pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(s.entry), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
            let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(s.entry), layout: Some(&pl), module: &modules[s.shader], entry_point: Some(s.entry),
                compilation_options: Default::default(), cache: None,
            });
            (s.entry, Entry { pipeline, layout, bindings: s.bindings })
        }).collect();
        Kernels { entries }
    }

    fn entry(&self, name: &str) -> &Entry {
        self.entries.get(name).unwrap_or_else(|| panic!("Kernels: unknown kernel {name:?}"))
    }

    pub fn pipeline(&self, name: &str) -> &wgpu::ComputePipeline { &self.entry(name).pipeline }

    pub fn bind_group_layout(&self, name: &str) -> &wgpu::BindGroupLayout { &self.entry(name).layout }

    /// Bind group for kernel `name`; `resources` follow the order of the kernel's binding list.
    pub fn bind_group(&self, ctx: &GpuContext, name: &str, resources: &[wgpu::BindingResource<'_>]) -> wgpu::BindGroup {
        let e = self.entry(name);
        assert_eq!(resources.len(), e.bindings.len(), "Kernels::bind_group({name}): expected {} resources", e.bindings.len());
        let entries: Vec<wgpu::BindGroupEntry> = e.bindings.iter().zip(resources)
            .map(|(&(binding, _), r)| wgpu::BindGroupEntry { binding, resource: r.clone() }).collect();
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(name), layout: &e.layout, entries: &entries })
    }

    /// Records one dispatch of `name` (1D, `workgroups` × 256 threads) into `enc`.
    pub fn record(&self, enc: &mut wgpu::CommandEncoder, name: &str, bind_group: &wgpu::BindGroup, workgroups: u32) {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(name), timestamp_writes: None });
        pass.set_pipeline(self.pipeline(name));
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    /// Fresh encoder, one dispatch, submit (does not wait).
    pub fn run(&self, ctx: &GpuContext, name: &str, resources: &[wgpu::BindingResource<'_>], workgroups: u32) {
        let bg = self.bind_group(ctx, name, resources);
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(name) });
        self.record(&mut enc, name, &bg, workgroups);
        ctx.queue.submit([enc.finish()]);
    }
}

/// Workgroups for a grid-stride kernel over `len` items: ceil(len / 256), clamped to [1, MAX_WORKGROUPS].
pub fn workgroups_for(len: usize) -> u32 {
    len.div_ceil(WORKGROUP as usize).clamp(1, MAX_WORKGROUPS as usize) as u32
}

/// A small uniform buffer initialised with `value` (one per dispatch, so batched encoders stay correct).
pub fn uniform<T: bytemuck::Pod>(ctx: &GpuContext, label: &str, value: &T) -> wgpu::Buffer {
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label), size: std::mem::size_of::<T>() as u64,
        usage: wgpu::BufferUsages::UNIFORM, mapped_at_creation: true,
    });
    buffer.slice(..).get_mapped_range_mut().expect("uniform: mapped range").copy_from_slice(bytemuck::bytes_of(value));
    buffer.unmap();
    buffer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kernel_compiles() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        let k = Kernels::new(&ctx);
        for s in SPECS { let _ = k.pipeline(s.entry); }
    }

    #[test]
    fn workgroup_counts() {
        assert_eq!(workgroups_for(0), 1);
        assert_eq!(workgroups_for(256), 1);
        assert_eq!(workgroups_for(257), 2);
        assert_eq!(workgroups_for(usize::MAX / 2), MAX_WORKGROUPS);
    }
}
