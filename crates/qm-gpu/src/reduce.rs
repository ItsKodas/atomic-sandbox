//! Two-level reductions: `dot_partials`/`sum_partials` write one f32 partial per workgroup
//! (≤ `MAX_PARTIALS` workgroups grid-striding over the interior); the CPU sums the partials in
//! f64 with Neumaier compensation and multiplies by dv. No atomics, no serial GPU sums.
//!
//! Batching: a reduction submits and blocks on its readback, so it is the natural sync point of
//! a batch. `dot_after`/`sum_after`/`gram_after` take the caller's pending encoder (`*_rec`
//! ops recorded but not submitted), append the reduction to it and submit everything at once.
//! The plain `dot`/`sum`/`gram` use a fresh encoder: callers holding an unsubmitted encoder
//! must use the `_after` form (or submit first), or the reduction will not see those writes.
//! `dot_in`/`sum_in`/`gram_in` do the same for a `Batch` and leave it empty but usable.
use crate::batch::Batch;
use crate::grid::{map_read_f32, Field, GpuGrid};
use crate::kernels::{encoder, uniform, Kernels, WORKGROUP};
use crate::pointwise::Params;
use gpu::GpuContext;

/// Bound on the number of workgroups (and so partials) per reduction.
pub const MAX_PARTIALS: u32 = 1024;
/// Byte stride between per-pair partial blocks in `gram` (a multiple of the 256-byte storage offset alignment).
const PAIR_STRIDE: u64 = MAX_PARTIALS as u64 * 4;

pub struct Reducer {
    partials: wgpu::Buffer,
    staging: wgpu::Buffer,
    /// Workgroups dispatched = partials produced: min(ceil(n³/256), 1024).
    workgroups: u32,
    n: usize,
}

impl Reducer {
    pub fn new(ctx: &GpuContext, _k: &Kernels, g: &GpuGrid) -> Reducer {
        let (partials, staging) = Self::buffers(ctx, 1);
        let workgroups = (g.interior_len().div_ceil(WORKGROUP as usize) as u32).clamp(1, MAX_PARTIALS);
        Reducer { partials, staging, workgroups, n: g.n }
    }

    fn buffers(ctx: &GpuContext, pairs: usize) -> (wgpu::Buffer, wgpu::Buffer) {
        let size = pairs as u64 * PAIR_STRIDE;
        let partials = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reduce partials"), size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
        });
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reduce staging"), size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
        });
        (partials, staging)
    }

    fn check(&self, g: &GpuGrid, fields: &[&Field]) {
        assert_eq!(g.n, self.n, "Reducer: built for n = {}, called with n = {}", self.n, g.n);
        for f in fields { assert_eq!(f.len, g.padded_len(), "Reducer: field/grid size mismatch"); }
    }

    /// Appends `pairs` reductions (entry `name`, one per (a, b)) to `enc`, writing consecutive
    /// partial blocks of `partials`, copies them to `staging`, submits `enc`, waits, and returns
    /// the f64 sums × dv.
    #[allow(clippy::too_many_arguments)]
    fn reduce(&self, mut enc: wgpu::CommandEncoder, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, name: &str,
              pairs: &[(&Field, &Field)], (partials, staging): (&wgpu::Buffer, &wgpu::Buffer)) -> Vec<f64> {
        let u = uniform(ctx, name, &Params::new(g, 0.0, 0.0));
        let wg = self.workgroups;
        for (pi, (a, b)) in pairs.iter().enumerate() {
            let block = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: partials, offset: pi as u64 * PAIR_STRIDE, size: wgpu::BufferSize::new(wg as u64 * 4),
            });
            let res = if name == "sum_partials" {
                vec![u.as_entire_binding(), a.buffer.as_entire_binding(), block]
            } else {
                vec![u.as_entire_binding(), a.buffer.as_entire_binding(), b.buffer.as_entire_binding(), block]
            };
            let bg = k.bind_group(ctx, name, &res);
            k.record(&mut enc, name, &bg, wg);
        }
        let bytes = pairs.len() as u64 * PAIR_STRIDE;
        enc.copy_buffer_to_buffer(partials, 0, staging, 0, Some(bytes));
        ctx.queue.submit([enc.finish()]);
        let raw = map_read_f32(ctx, staging, pairs.len() * MAX_PARTIALS as usize);
        let dv = g.dv();
        raw.chunks(MAX_PARTIALS as usize)
            .map(|block| neumaier(block[..wg as usize].iter().map(|v| *v as f64)) * dv)
            .collect()
    }

    /// Σ_interior a·b · dv, f32 workgroup partials summed in f64 (Neumaier).
    pub fn dot(&self, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, a: &Field, b: &Field) -> f64 {
        self.dot_after(encoder(ctx, "dot"), ctx, k, g, a, b)
    }

    /// `dot` appended to the caller's pending encoder `enc` (submitted together, then read back).
    pub fn dot_after(&self, enc: wgpu::CommandEncoder, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, a: &Field, b: &Field) -> f64 {
        self.check(g, &[a, b]);
        self.reduce(enc, ctx, k, g, "dot_partials", &[(a, b)], (&self.partials, &self.staging))[0]
    }

    /// `dot` as the sync point of a `Batch`: ends its pass, appends the reduction to everything
    /// recorded so far, submits, waits for the readback, and leaves the batch empty but usable
    /// (later ops record into a fresh encoder and pass).
    pub fn dot_in(&self, b: &mut Batch, k: &Kernels, g: &GpuGrid, x: &Field, y: &Field) -> f64 {
        self.dot_after(b.take_encoder(), b.ctx(), k, g, x, y)
    }

    /// Σ_interior a · dv.
    pub fn sum(&self, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, a: &Field) -> f64 {
        self.sum_after(encoder(ctx, "sum"), ctx, k, g, a)
    }

    /// `sum` appended to the caller's pending encoder `enc`.
    pub fn sum_after(&self, enc: wgpu::CommandEncoder, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, a: &Field) -> f64 {
        self.check(g, &[a]);
        self.reduce(enc, ctx, k, g, "sum_partials", &[(a, a)], (&self.partials, &self.staging))[0]
    }

    /// `sum` as the sync point of a `Batch` (see `dot_in`).
    pub fn sum_in(&self, b: &mut Batch, k: &Kernels, g: &GpuGrid, a: &Field) -> f64 {
        self.sum_after(b.take_encoder(), b.ctx(), k, g, a)
    }

    /// Batched: all pairwise dots of `xs` against `ys` in one submission (Gram matrices).
    /// Returns row-major `[i * ys.len() + j] = ⟨xs[i], ys[j]⟩`. One dispatch per pair, one
    /// encoder, one readback. Bit-identical to `dot` per pair (same kernel, same partial order).
    /// Allocates its partials/staging per call (sized xs.len()·ys.len()·4 KiB).
    pub fn gram(&self, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, xs: &[&Field], ys: &[&Field]) -> Vec<f64> {
        if xs.is_empty() || ys.is_empty() { return Vec::new(); }
        self.gram_after(encoder(ctx, "gram"), ctx, k, g, xs, ys)
    }

    /// `gram` appended to the caller's pending encoder `enc`. With empty `xs`/`ys` the pending
    /// work is still submitted (and an empty matrix returned, without waiting).
    pub fn gram_after(&self, enc: wgpu::CommandEncoder, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, xs: &[&Field], ys: &[&Field]) -> Vec<f64> {
        if xs.is_empty() || ys.is_empty() {
            ctx.queue.submit([enc.finish()]);
            return Vec::new();
        }
        self.check(g, xs);
        self.check(g, ys);
        let pairs: Vec<(&Field, &Field)> = xs.iter().flat_map(|x| ys.iter().map(move |y| (*x, *y))).collect();
        let (partials, staging) = Self::buffers(ctx, pairs.len());
        self.reduce(enc, ctx, k, g, "dot_partials", &pairs, (&partials, &staging))
    }

    /// `gram` as the sync point of a `Batch` (see `dot_in`; pending work is submitted even when
    /// `xs` or `ys` is empty).
    pub fn gram_in(&self, b: &mut Batch, k: &Kernels, g: &GpuGrid, xs: &[&Field], ys: &[&Field]) -> Vec<f64> {
        self.gram_after(b.take_encoder(), b.ctx(), k, g, xs, ys)
    }
}

/// Neumaier-compensated f64 summation (same algorithm as `qm::grid`'s private helper).
pub fn neumaier(iter: impl Iterator<Item = f64>) -> f64 {
    let (mut sum, mut c) = (0.0f64, 0.0f64);
    for x in iter {
        let t = sum + x;
        if sum.abs() >= x.abs() { c += (sum - t) + x; } else { c += (x - t) + sum; }
        sum = t;
    }
    sum + c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{Field, GpuGrid};
    use crate::kernels::Kernels;

    #[test]
    fn dot_matches_cpu_grid_dot_to_f32_precision() {
        let Some(ctx) = gpu::GpuContext::new_headless() else { return };
        let cg = qm::Grid::cube(33, 0.25);
        let g = GpuGrid::new(&cg);
        let a = cg.sample(|[x, y, z]| (-(x * x + y * y + z * z)).exp());
        let b = cg.sample(|[x, _, _]| 1.0 + 0.1 * x);
        let k = Kernels::new(&ctx);
        let r = Reducer::new(&ctx, &k, &g);
        let (fa, fb) = (Field::new(&ctx, &g, "a"), Field::new(&ctx, &g, "b"));
        fa.upload_interior(&ctx, &g, &a);
        fb.upload_interior(&ctx, &g, &b);
        let gpu = r.dot(&ctx, &k, &g, &fa, &fb);
        let cpu = cg.dot(&a, &b);
        println!("dot 33³: gpu {gpu} cpu {cpu} rel {:.2e}", ((gpu - cpu) / cpu).abs());
        assert!((gpu - cpu).abs() < 1e-7 * cpu.abs().max(1.0), "gpu {gpu} cpu {cpu}"); // achieved 1.2e-8; plan bound 1e-6 tightened
    }

    #[test]
    fn sum_of_ones_is_exact_volume() {
        let Some(ctx) = gpu::GpuContext::new_headless() else { return };
        let cg = qm::Grid::cube(49, 0.25);
        let g = GpuGrid::new(&cg);
        let k = Kernels::new(&ctx);
        let r = Reducer::new(&ctx, &k, &g);
        let f = Field::new(&ctx, &g, "ones");
        f.upload_interior(&ctx, &g, &vec![1.0; cg.len()]);
        let s = r.sum(&ctx, &k, &g, &f);
        assert!((s - cg.len() as f64 * cg.dv()).abs() < 1e-9, "{s}"); // partials are exact integers in f32 up to 2^24
    }

    /// Exercises the grid-stride loop (more interior points than 1024 workgroups × 256 threads)
    /// and the ghost exclusion: a field that is 1 everywhere in the interior sums to n³ exactly.
    #[test]
    fn sum_with_grid_stride_is_exact() {
        let Some(ctx) = gpu::GpuContext::new_headless() else { return };
        let cg = qm::Grid::cube(65, 0.1); // 274625 > 262144 threads
        let g = GpuGrid::new(&cg);
        let k = Kernels::new(&ctx);
        let r = Reducer::new(&ctx, &k, &g);
        let f = Field::new(&ctx, &g, "ones");
        f.upload_interior(&ctx, &g, &vec![1.0; cg.len()]);
        let s = r.sum(&ctx, &k, &g, &f) / g.dv();
        assert!((s - cg.len() as f64).abs() < 1e-6, "{s} vs {}", cg.len());
    }

    #[test]
    fn gram_matches_pairwise_dots() {
        let Some(ctx) = gpu::GpuContext::new_headless() else { return };
        let cg = qm::Grid::cube(21, 0.3);
        let g = GpuGrid::new(&cg);
        let k = Kernels::new(&ctx);
        let r = Reducer::new(&ctx, &k, &g);
        let xs_cpu = [
            cg.sample(|[x, y, z]| (-(x * x + y * y + z * z)).exp()),
            cg.sample(|[x, y, _]| x * (-(x * x + y * y)).exp()),
            cg.sample(|[_, _, z]| (0.7 * z).cos()),
        ];
        let ys_cpu = [cg.sample(|[x, _, _]| 1.0 + 0.1 * x), cg.sample(|[x, y, z]| (x + 2.0 * y - z).sin())];
        let up = |d: &Vec<f64>, l: &str| { let f = Field::new(&ctx, &g, l); f.upload_interior(&ctx, &g, d); f };
        let xs: Vec<Field> = xs_cpu.iter().map(|d| up(d, "x")).collect();
        let ys: Vec<Field> = ys_cpu.iter().map(|d| up(d, "y")).collect();
        let xr: Vec<&Field> = xs.iter().collect();
        let yr: Vec<&Field> = ys.iter().collect();
        let m = r.gram(&ctx, &k, &g, &xr, &yr);
        assert_eq!(m.len(), 6);
        for (i, x) in xs.iter().enumerate() {
            for (j, y) in ys.iter().enumerate() {
                let d = r.dot(&ctx, &k, &g, x, y);
                let got = m[i * ys.len() + j];
                assert!((got - d).abs() <= 1e-9 * d.abs().max(1e-30), "gram[{i},{j}] = {got} vs dot {d}");
                let cpu = cg.dot(&xs_cpu[i], &ys_cpu[j]);
                let scale = cg.dot(&xs_cpu[i], &xs_cpu[i]).sqrt() * cg.dot(&ys_cpu[j], &ys_cpu[j]).sqrt();
                assert!((got - cpu).abs() < 1e-6 * scale, "gram[{i},{j}] = {got} vs cpu {cpu}");
            }
        }
        // empty inputs give an empty matrix without touching the GPU
        assert!(r.gram(&ctx, &k, &g, &[], &yr).is_empty());
    }

    /// `dot_after`/`sum_after`/`gram_after` submit the caller's pending encoder first, so the
    /// reduction sees the recorded (not yet submitted) writes.
    #[test]
    fn reductions_after_pending_encoder_see_its_writes() {
        let Some(ctx) = gpu::GpuContext::new_headless() else { return };
        let cg = qm::Grid::cube(15, 0.3);
        let g = GpuGrid::new(&cg);
        let k = Kernels::new(&ctx);
        let r = Reducer::new(&ctx, &k, &g);
        let ones = vec![1.0; cg.len()];
        let (a, b) = (Field::new(&ctx, &g, "a"), Field::new(&ctx, &g, "b"));
        a.upload_interior(&ctx, &g, &ones);
        b.upload_interior(&ctx, &g, &ones);
        let vol = cg.len() as f64 * cg.dv();
        let pending = |scale: f32| {
            let mut enc = crate::kernels::encoder(&ctx, "pending");
            crate::pointwise::scale_rec(&mut enc, &ctx, &k, &g, &a, scale); // a *= scale, not yet submitted
            enc
        };
        let d = r.dot_after(pending(3.0), &ctx, &k, &g, &a, &b);
        assert!((d - 3.0 * vol).abs() < 1e-9 * vol, "dot_after {d} vs {}", 3.0 * vol);
        let s = r.sum_after(pending(2.0), &ctx, &k, &g, &a);
        assert!((s - 6.0 * vol).abs() < 1e-9 * vol, "sum_after {s} vs {}", 6.0 * vol);
        let m = r.gram_after(pending(0.5), &ctx, &k, &g, &[&a], &[&b, &a]);
        assert!((m[0] - 3.0 * vol).abs() < 1e-9 * vol && (m[1] - 9.0 * vol).abs() < 1e-9 * vol, "gram_after {m:?}");
        // empty gram still submits the pending work
        assert!(r.gram_after(pending(2.0), &ctx, &k, &g, &[], &[&b]).is_empty());
        let s = r.sum(&ctx, &k, &g, &a);
        assert!((s - 6.0 * vol).abs() < 1e-9 * vol, "pending work of the empty gram was dropped: {s}");
    }

    #[test]
    fn neumaier_recovers_cancellation() {
        let v = [1.0, 1e100, 1.0, -1e100];
        assert_eq!(neumaier(v.iter().copied()), 2.0);
    }

    /// Timing probe for the report: `cargo test -p qm-gpu --release -- --ignored --nocapture bench_97`.
    #[test]
    #[ignore]
    fn bench_97_dot_and_axpy() {
        let Some(ctx) = gpu::GpuContext::new_headless() else { return };
        let cg = qm::Grid::cube(97, 0.2);
        let g = GpuGrid::new(&cg);
        let k = Kernels::new(&ctx);
        let r = Reducer::new(&ctx, &k, &g);
        let a = cg.sample(|[x, y, z]| (-(x * x + y * y + z * z)).exp());
        let (fa, fb) = (Field::new(&ctx, &g, "a"), Field::new(&ctx, &g, "b"));
        fa.upload_interior(&ctx, &g, &a);
        fb.upload_interior(&ctx, &g, &a);
        let reps = 200;
        for _ in 0..10 { r.dot(&ctx, &k, &g, &fa, &fb); }
        let t = std::time::Instant::now();
        for _ in 0..reps { std::hint::black_box(r.dot(&ctx, &k, &g, &fa, &fb)); }
        let dot_ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
        let sync = || { ctx.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll"); };
        for _ in 0..10 { crate::pointwise::axpy(&ctx, &k, &g, &fb, 1e-3, &fa); }
        sync();
        let t = std::time::Instant::now();
        for _ in 0..reps { crate::pointwise::axpy(&ctx, &k, &g, &fb, 1e-3, &fa); }
        sync();
        let axpy_ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
        let t = std::time::Instant::now();
        for _ in 0..reps { crate::pointwise::axpy(&ctx, &k, &g, &fb, 1e-3, &fa); sync(); }
        let axpy_sync_ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
        println!("97^3: dot {dot_ms:.3} ms (incl. readback), axpy {axpy_ms:.3} ms (pipelined), axpy {axpy_sync_ms:.3} ms (submit+wait each)");
    }
}
