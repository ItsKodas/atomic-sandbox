//! Pipeline cache: every WGSL entry point is compiled once per device, each with an explicit
//! bind-group layout (group 0) described by a `Spec`. Callers build bind groups by passing
//! resources in the order of the spec's binding list, then `record`/`run` a dispatch.
//!
//! Batching: every op in this crate has a `*_rec(r, …)` form that records into a `batch::Recorder`:
//! a caller-owned encoder (`encoder`; each dispatch its own compute pass) or a `batch::Batch`
//! (all dispatches in one pass — the fast path, see `batch`). wgpu orders the dispatches and
//! inserts the barriers in recording order either way.
//!
//! Generated kernels: the fused Gram kernel `gram{F}` (`gram_source`) is emitted at device
//! creation for a few field counts F chosen from the device's storage-buffer limit (the `gpu`
//! crate asks for up to 16 per stage; WebGPU's default is 8), so the WGSL is straight-line code
//! with one accumulator per pair and no dynamically indexed private arrays.
use gpu::GpuContext;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Workgroup size of the 1D kernels in this crate (`@workgroup_size(256)`).
pub const WORKGROUP: u32 = 256;
/// Upper bound on workgroups for grid-stride kernels (WebGPU default per-dimension limit).
pub const MAX_WORKGROUPS: u32 = 65535;

/// One binding of group 0: a buffer (uniform, read-only or read-write storage), or a
/// write-only rgba16float 3D storage texture (the orbital viewer's colour volume).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot { Uniform, Read, ReadWrite, WriteTexRgba16f3d }

struct Spec {
    entry: &'static str,
    shader: &'static str,
    bindings: &'static [(u32, Slot)],
}

use Slot::{Read as R, ReadWrite as RW, Uniform as U, WriteTexRgba16f3d as TW};

const SHADERS: &[(&str, &str)] = &[
    ("reduce", include_str!("shaders/reduce.wgsl")),
    ("pointwise", include_str!("shaders/pointwise.wgsl")),
    ("stencil", include_str!("shaders/stencil.wgsl")),
    ("nuclear", include_str!("shaders/nuclear.wgsl")),
    ("xc", include_str!("shaders/xc.wgsl")),
    ("poisson", include_str!("shaders/poisson.wgsl")),
    ("eigen", include_str!("shaders/eigen.wgsl")),
    ("forces", include_str!("shaders/forces.wgsl")),
    ("multigrid", include_str!("shaders/multigrid.wgsl")),
    ("orbital_moments", include_str!("shaders/orbital_moments.wgsl")),
    ("colour_accumulate", include_str!("shaders/colour_accumulate.wgsl")),
    ("enclosure", include_str!("shaders/enclosure.wgsl")),
    ("sample", include_str!("shaders/sample.wgsl")),
    // M7 Task 8a: erf.wgsl is a verbatim copy of nuclear.wgsl's erf (nuclear.wgsl untouched: the None path is bitwise)
    ("ionic", concat!(include_str!("shaders/erf.wgsl"), include_str!("shaders/ionic.wgsl"))),
    // M7 Task 8b: its G(x) is a verbatim copy of forces.wgsl's (forces.wgsl untouched: the None force path is bitwise)
    ("ionic_forces", include_str!("shaders/ionic_forces.wgsl")),
];

/// lincomb: uniform at 4, out at 1, inputs at 5..=11 (7 reads + 1 read_write = 8 storage
/// buffers, the default `max_storage_buffers_per_shader_stage`).
pub const LINCOMB_MAX_TERMS: usize = 7;
/// diis_mix: uniform at 12, out at 1, base at 2, terms at 5..=10 (6 terms + base + out = 8).
pub const DIIS_MAX_TERMS: usize = 6;

const SPECS: &[Spec] = &[
    Spec { entry: "dot_partials", shader: "reduce", bindings: &[(0, U), (1, R), (2, R), (3, RW)] },
    Spec { entry: "sum_partials", shader: "reduce", bindings: &[(0, U), (1, R), (3, RW)] },
    Spec { entry: "abs_diff_sum_partials", shader: "reduce", bindings: &[(0, U), (1, R), (2, R), (3, RW)] },
    Spec { entry: "max_partials", shader: "reduce", bindings: &[(0, U), (1, R), (3, RW)] },
    Spec { entry: "face_max_partials", shader: "reduce", bindings: &[(0, U), (1, R), (3, RW)] },
    Spec { entry: "abs_max_partials", shader: "reduce", bindings: &[(0, U), (1, R), (3, RW)] },
    Spec { entry: "axpy", shader: "pointwise", bindings: &[(0, U), (1, RW), (2, R)] },
    Spec { entry: "scale", shader: "pointwise", bindings: &[(0, U), (1, RW)] },
    Spec { entry: "fill", shader: "pointwise", bindings: &[(0, U), (1, RW)] },
    Spec { entry: "clamp_min", shader: "pointwise", bindings: &[(0, U), (1, RW)] },
    Spec { entry: "mul", shader: "pointwise", bindings: &[(0, U), (1, RW), (2, R), (3, R)] },
    Spec { entry: "mul_add", shader: "pointwise", bindings: &[(0, U), (1, RW), (2, R), (3, R)] },
    Spec { entry: "mul_add_sq", shader: "pointwise", bindings: &[(0, U), (1, RW), (2, R)] },
    Spec {
        entry: "lincomb",
        shader: "pointwise",
        bindings: &[(4, U), (1, RW), (5, R), (6, R), (7, R), (8, R), (9, R), (10, R), (11, R)],
    },
    Spec {
        entry: "diis_mix",
        shader: "pointwise",
        bindings: &[(12, U), (1, RW), (2, R), (5, R), (6, R), (7, R), (8, R), (9, R), (10, R)],
    },
    Spec { entry: "unpad", shader: "pointwise", bindings: &[(13, U), (1, RW), (2, R)] },
    Spec { entry: "laplacian", shader: "stencil", bindings: &[(0, U), (1, RW), (2, R)] },
    Spec { entry: "hamiltonian", shader: "stencil", bindings: &[(4, U), (1, RW), (2, R), (3, R)] },
    Spec { entry: "jacobi_step", shader: "stencil", bindings: &[(4, U), (1, RW), (2, R), (3, R), (5, R)] },
    Spec { entry: "nuclear_potential", shader: "nuclear", bindings: &[(0, U), (1, RW), (2, R)] },
    Spec { entry: "xc_lsda", shader: "xc", bindings: &[(0, U), (1, R), (2, R), (3, RW), (4, RW), (5, RW)] },
    Spec { entry: "ghost_fill", shader: "poisson", bindings: &[(0, U), (1, RW), (2, R), (3, R), (4, R)] },
    Spec { entry: "precond_init", shader: "eigen", bindings: &[(0, U), (1, RW), (2, R), (3, R)] },
    Spec { entry: "force_partials", shader: "forces", bindings: &[(0, U), (1, R), (2, RW)] },
    Spec { entry: "mg_jacobi", shader: "multigrid", bindings: &[(0, U), (1, RW), (2, R), (3, R)] },
    Spec { entry: "mg_residual", shader: "multigrid", bindings: &[(0, U), (1, RW), (2, R), (3, R)] },
    Spec { entry: "mg_restrict", shader: "multigrid", bindings: &[(0, U), (1, RW), (2, R)] },
    Spec { entry: "mg_prolong_add", shader: "multigrid", bindings: &[(0, U), (1, RW), (2, R)] },
    Spec { entry: "orbital_moments", shader: "orbital_moments", bindings: &[(0, U), (1, R), (2, R), (3, RW)] },
    Spec {
        entry: "colour_accumulate",
        shader: "colour_accumulate",
        bindings: &[(0, U), (1, R), (2, R), (3, R), (4, R), (5, R), (6, R), (7, RW), (8, TW), (9, R)],
    },
    Spec { entry: "enclosure_hist", shader: "enclosure", bindings: &[(0, U), (1, R), (2, RW), (3, RW)] },
    Spec { entry: "enclosure_level", shader: "enclosure", bindings: &[(4, U), (5, R), (6, RW)] },
    Spec { entry: "sample_density", shader: "sample", bindings: &[(0, U), (1, R), (2, R), (3, RW)] },
    // M7 Task 8a (shaders/ionic.wgsl, `ionic`; the occupation-weighted density in pointwise.wgsl)
    Spec { entry: "mul_add_sq_w", shader: "pointwise", bindings: &[(0, U), (1, RW), (2, R)] },
    Spec { entry: "smeared_potential", shader: "ionic", bindings: &[(0, U), (1, RW), (2, R)] },
    Spec { entry: "local_sr", shader: "ionic", bindings: &[(0, U), (1, RW), (3, R), (4, R)] },
    Spec { entry: "nlcc_deposit", shader: "ionic", bindings: &[(0, U), (1, RW), (3, R), (4, R)] },
    Spec { entry: "proj_dot", shader: "ionic", bindings: &[(10, U), (11, R), (12, R), (13, R), (14, RW), (15, R)] },
    Spec { entry: "proj_finish", shader: "ionic", bindings: &[(10, U), (14, RW), (15, R), (16, RW)] },
    Spec { entry: "proj_axpy", shader: "ionic", bindings: &[(10, U), (11, R), (12, R), (15, R), (16, RW), (17, RW)] },
    // M7 Task 8b (shaders/ionic_forces.wgsl): the compensated pseudised force partials
    Spec { entry: "centre_force", shader: "ionic_forces", bindings: &[(0, U), (1, R), (2, RW)] },
    Spec { entry: "gather_dot4", shader: "ionic_forces", bindings: &[(10, U), (11, R), (12, R), (13, R), (2, RW), (15, R)] },
];

/// Orbitals bound per `colour_accumulate` dispatch (6 + its accumulator + the boundary levels =
/// 8 storage buffers, WebGPU's default); more are chained through the accumulator.
pub const COLOUR_MAX_TERMS: usize = 6;

/// Workgroups (= partials per pair) of the fused Gram kernel: fewer than `reduce::MAX_PARTIALS`
/// because each thread does F loads and F(F+1)/2 FMAs per point (the work per thread is what
/// hides latency, not the thread count) and because every partial is summed on the CPU — at
/// 256 the f64 finish of a 245-pair LOBPCG readback is ~60k adds instead of ~250k.
pub const GRAM_WORKGROUPS: u32 = 256;

/// Name of the generated fused Gram kernel variant for `f` fields.
pub fn gram_name(f: usize) -> String { format!("gram{f}") }

/// WGSL of the fused Gram kernel over `f` fields (bindings: 0 uniform `Params`, 1..=f the
/// fields read-only, f+1 the partials read-write). Every thread grid-strides over the interior,
/// loads the f values at a point and accumulates all f(f+1)/2 products v_a·v_b, a ≥ b, in
/// scalar registers; the accumulators are then tree-reduced four at a time (one `vec4` per
/// thread in workgroup memory) and the workgroup's partial for pair p = a(a+1)/2 + b lands at
/// `partials[p * GRAM_WORKGROUPS + wg]`. Same per-thread order and tree shape as
/// `dot_partials` for a given workgroup count, hence deterministic.
pub fn gram_source(f: usize) -> String {
    use std::fmt::Write as _;
    let np = f * (f + 1) / 2;
    let mut s = String::new();
    let _ = writeln!(s, "// Generated by qm_gpu::kernels::gram_source({f}): all {np} pairs among {f} fields in one dispatch.");
    let _ = writeln!(s, "struct Params {{ a: f32, b: f32, p: u32, n: u32 }}");
    let _ = writeln!(s, "@group(0) @binding(0) var<uniform> P: Params;");
    for i in 0..f { let _ = writeln!(s, "@group(0) @binding({}) var<storage, read> f{i}: array<f32>;", i + 1); }
    let _ = writeln!(s, "@group(0) @binding({}) var<storage, read_write> partials: array<f32>;", f + 1);
    let _ = writeln!(s, "var<workgroup> sh: array<vec4<f32>, 256>;");
    let _ = writeln!(s, "fn pidx(idx: u32) -> u32 {{ let i = idx % P.n; let t = idx / P.n; let j = t % P.n; let k = t / P.n; return (i + 4u) + P.p * ((j + 4u) + P.p * (k + 4u)); }}");
    let _ = writeln!(s, "@compute @workgroup_size(256)\nfn gram(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {{");
    let _ = writeln!(s, "    let total = P.n * P.n * P.n;\n    let stride = nw.x * 256u;");
    for a in 0..f { for b in 0..=a { let _ = writeln!(s, "    var a{a}_{b} = 0.0;"); } }
    let _ = writeln!(s, "    for (var idx = wg.x * 256u + li; idx < total; idx = idx + stride) {{\n        let q = pidx(idx);");
    for i in 0..f { let _ = writeln!(s, "        let v{i} = f{i}[q];"); }
    for a in 0..f { for b in 0..=a { let _ = writeln!(s, "        a{a}_{b} = a{a}_{b} + v{a} * v{b};"); } }
    let _ = writeln!(s, "    }}");
    let pairs: Vec<(usize, usize)> = (0..f).flat_map(|a| (0..=a).map(move |b| (a, b))).collect();
    for (gi, group) in pairs.chunks(4).enumerate() {
        let comps: Vec<String> = (0..4).map(|c| group.get(c).map_or("0.0".to_string(), |(a, b)| format!("a{a}_{b}"))).collect();
        let _ = writeln!(s, "    sh[li] = vec4<f32>({});\n    workgroupBarrier();", comps.join(", "));
        let _ = writeln!(s, "    for (var s = 128u; s > 0u; s = s >> 1u) {{ if (li < s) {{ sh[li] = sh[li] + sh[li + s]; }} workgroupBarrier(); }}");
        let _ = writeln!(s, "    if (li == 0u) {{");
        for (c, _) in group.iter().enumerate() {
            let p = gi * 4 + c;
            let _ = writeln!(s, "        partials[{p}u * {GRAM_WORKGROUPS}u + wg.x] = sh[0].{};", ["x", "y", "z", "w"][c]);
        }
        let _ = writeln!(s, "    }}\n    workgroupBarrier();");
    }
    let _ = writeln!(s, "}}");
    s
}

struct Entry {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    bindings: Vec<(u32, Slot)>,
}

/// Bytes per uniform slot of a ring (`batch::Batch`): WebGPU's largest allowed
/// `min_uniform_buffer_offset_alignment`, so a slot offset is valid on every device, and more
/// than any uniform struct in this crate (≤ 64 B).
pub const UNIFORM_SLOT: u64 = 256;
/// Slots per ring buffer (64 KiB): more than one LOBPCG iteration's dispatches at k = 5 (≈ 75)
/// between two sync points; a longer segment chains a further ring.
pub const RING_SLOTS: u64 = 256;

/// Ring buffers (`UNIFORM_SLOT` × `RING_SLOTS` bytes, UNIFORM | COPY_DST) not in use by a
/// `Batch`, and how many were ever created.
#[derive(Default)]
pub(crate) struct RingPool { free: Vec<wgpu::Buffer>, created: usize }

impl RingPool {
    /// A free ring buffer, or a new one.
    pub(crate) fn take(pool: &Mutex<RingPool>, ctx: &GpuContext) -> wgpu::Buffer {
        let mut p = pool.lock().expect("ring pool poisoned");
        p.free.pop().unwrap_or_else(|| {
            p.created += 1;
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("uniform ring"), size: UNIFORM_SLOT * RING_SLOTS,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
            })
        })
    }

    /// Returns ring buffers to the pool (their commands must have been submitted, or dropped).
    pub(crate) fn give(pool: &Mutex<RingPool>, rings: impl IntoIterator<Item = wgpu::Buffer>) {
        pool.lock().expect("ring pool poisoned").free.extend(rings);
    }
}

pub struct Kernels {
    entries: HashMap<String, Entry>,
    /// Field counts of the compiled `gram{F}` variants, ascending.
    gram_variants: Vec<usize>,
    /// The uniform rings `Batch`es take their per-dispatch uniforms from (module doc of `batch`).
    pub(crate) rings: Arc<Mutex<RingPool>>,
}

impl Kernels {
    pub fn new(ctx: &GpuContext) -> Kernels {
        let modules: HashMap<&str, wgpu::ShaderModule> = SHADERS.iter().map(|(name, src)| {
            (*name, ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name), source: wgpu::ShaderSource::Wgsl((*src).into()),
            }))
        }).collect();
        let make = |label: &str, module: &wgpu::ShaderModule, entry_point: &str, bindings: Vec<(u32, Slot)>| -> Entry {
            let layout_entries: Vec<wgpu::BindGroupLayoutEntry> = bindings.iter().map(|&(binding, slot)| {
                let buffer = |ty| wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None };
                let ty = match slot {
                    Slot::Uniform => buffer(wgpu::BufferBindingType::Uniform),
                    Slot::Read => buffer(wgpu::BufferBindingType::Storage { read_only: true }),
                    Slot::ReadWrite => buffer(wgpu::BufferBindingType::Storage { read_only: false }),
                    Slot::WriteTexRgba16f3d => wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                };
                wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None }
            }).collect();
            let layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries: &layout_entries });
            let pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
            let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label), layout: Some(&pl), module, entry_point: Some(entry_point),
                compilation_options: Default::default(), cache: None,
            });
            Entry { pipeline, layout, bindings }
        };
        let mut entries: HashMap<String, Entry> = SPECS.iter().map(|s| {
            (s.entry.to_string(), make(s.entry, &modules[s.shader], s.entry, s.bindings.to_vec()))
        }).collect();
        // fused Gram variants: F fields + the partials buffer ≤ the device's storage-buffer limit
        let limit = ctx.device.limits().max_storage_buffers_per_shader_stage as usize;
        let gram_variants: Vec<usize> = [4usize, 8, 15].into_iter().filter(|f| *f < limit)
            .chain(std::iter::once((limit - 1).min(15))).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        for &f in &gram_variants {
            let name = gram_name(f);
            let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&name), source: wgpu::ShaderSource::Wgsl(gram_source(f).into()),
            });
            let mut bindings = vec![(0, U)];
            bindings.extend((1..=f as u32).map(|b| (b, R)));
            bindings.push((f as u32 + 1, RW));
            entries.insert(name.clone(), make(&name, &module, "gram", bindings));
        }
        Kernels { entries, gram_variants, rings: Arc::new(Mutex::new(RingPool::default())) }
    }

    /// Uniform ring buffers created so far on this device's kernel set (they are pooled and
    /// reused by later `Batch`es; the count only grows when more are in use at once).
    pub fn uniform_rings_created(&self) -> usize { self.rings.lock().expect("ring pool poisoned").created }

    /// Field counts of the compiled fused Gram variants (ascending; the largest is the most a
    /// `reduce::block_dots` dispatch binds).
    pub fn gram_variants(&self) -> &[usize] { &self.gram_variants }

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
        self.record_3d(enc, name, bind_group, [workgroups, 1, 1]);
    }

    /// Records one dispatch of `name` with a 3D workgroup grid into `enc` (its own compute pass).
    pub fn record_3d(&self, enc: &mut wgpu::CommandEncoder, name: &str, bind_group: &wgpu::BindGroup, [x, y, z]: [u32; 3]) {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(name), timestamp_writes: None });
        pass.set_pipeline(self.pipeline(name));
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(x, y, z);
    }

    /// Fresh encoder, one dispatch, submit (does not wait).
    pub fn run(&self, ctx: &GpuContext, name: &str, resources: &[wgpu::BindingResource<'_>], workgroups: u32) {
        let bg = self.bind_group(ctx, name, resources);
        let mut enc = encoder(ctx, name);
        self.record(&mut enc, name, &bg, workgroups);
        ctx.queue.submit([enc.finish()]);
    }
}

/// A fresh command encoder for recording a batch of `*_rec` ops; submit it with
/// `ctx.queue.submit([enc.finish()])` (or hand it to a `Reducer::*_after`, which submits it).
pub fn encoder(ctx: &GpuContext, label: &str) -> wgpu::CommandEncoder {
    ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
}

/// Fresh encoder → `record` → submit (does not wait). The immediate form of every `*_rec` op.
pub(crate) fn immediate(ctx: &GpuContext, label: &str, record: impl FnOnce(&mut wgpu::CommandEncoder)) {
    let mut enc = encoder(ctx, label);
    record(&mut enc);
    ctx.queue.submit([enc.finish()]);
}

/// Workgroups for a grid-stride kernel over `len` items: ceil(len / 256), clamped to [1, MAX_WORKGROUPS].
pub fn workgroups_for(len: usize) -> u32 {
    len.div_ceil(WORKGROUP as usize).clamp(1, MAX_WORKGROUPS as usize) as u32
}

/// A small uniform buffer initialised with `value`: the bare-encoder path of
/// `Recorder::uniform` (one buffer per dispatch); a `Batch` uses its ring instead.
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
        // the generated Gram variants: at least the 4- and 7-field ones exist on any WebGPU device
        println!("storage buffers per stage: {}; gram variants {:?}", ctx.device.limits().max_storage_buffers_per_shader_stage, k.gram_variants());
        assert!(k.gram_variants().contains(&4));
        assert!(k.gram_variants().iter().any(|f| *f >= 7));
        for f in k.gram_variants() { let _ = k.pipeline(&gram_name(*f)); }
    }

    /// The generated source has one accumulator per pair and writes every pair exactly once.
    #[test]
    fn gram_source_covers_every_pair() {
        for f in [1usize, 4, 7, 15] {
            let src = gram_source(f);
            let np = f * (f + 1) / 2;
            for a in 0..f { for b in 0..=a {
                assert_eq!(src.matches(&format!("a{a}_{b} = a{a}_{b} + v{a} * v{b};")).count(), 1, "f = {f}: pair ({a},{b})");
            }}
            for p in 0..np { assert_eq!(src.matches(&format!("partials[{p}u * ")).count(), 1, "f = {f}: pair {p} written once"); }
            assert_eq!(src.matches(&format!("partials[{np}u * ")).count(), 0);
        }
    }

    #[test]
    fn workgroup_counts() {
        assert_eq!(workgroups_for(0), 1);
        assert_eq!(workgroups_for(256), 1);
        assert_eq!(workgroups_for(257), 2);
        assert_eq!(workgroups_for(usize::MAX / 2), MAX_WORKGROUPS);
    }
}
