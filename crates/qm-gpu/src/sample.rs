//! Trilinear sampling of the converged density at a list of points (M6 Task 2, spec §3.3):
//! `GpuSolver::sample_density`, the GPU side of `sim::ForceProvider::sample_density`.
//!
//! The CPU resolves every point with `qm::Grid::trilinear_cell` in f64 — the lower corner of
//! its cell and the fractions t ∈ [0, 1) per axis (f64; after rounding to f32 t ∈ [0, 1], and
//! exactly 0 on a node — what the exact node read rests on) — and uploads (t as f32, the corner's padded
//! index) per point, or a sentinel for a point outside [origin, origin + (n−1)h]³. The kernel
//! (`shaders/sample.wgsl`, `@workgroup_size(64)`, one thread per point) reads the 8 corners of
//! the padded `rho` field (on an upper face the +1 neighbours are the zero ghosts, weight 0)
//! and blends them in the order of `qm::Grid::trilinear`, so the two differ only by f32
//! rounding: the cell choice is identical by construction (a point within an f32 ulp of a cell
//! face cannot land in different cells on the two sides), and t carries f32 rounding of the
//! fraction (≤ 6e-8 relative), not of a grid coordinate (which at 97³ would be ~4e-6 of a
//! cell). A node is read with t = 0 on every axis and returns the stored f32 exactly (the atom
//! references read nodes, spec §3.2) — the reason a node gets t = 0, never t = 1: this driver evaluates the
//! blend in a form that is not exact at t = 1 (measured: a 1-ulp miss at a box corner before
//! the convention changed). One dispatch, one copy into a staging buffer, **one readback** of
//! `points.len()` f32 per call.
//!
//! Buffers (points, output, staging, uniform) live in a `SampleScratch` on the solver, grown
//! to the largest request seen and never shrunk: the detector calls this every evaluation, so
//! no buffer is allocated per call once warm (a call still builds its bind group and the
//! packed-point and result `Vec`s, as the forces do; `sampling_buffer_allocations` is the witness, as
//! `force_buffer_allocations` is for the forces).
use crate::grid::map_read_f32;
use crate::kernels::{encoder, MAX_WORKGROUPS};
use crate::scf::{GpuSolver, GpuState};
use gpu::GpuContext;

/// Threads per workgroup of `sample_density`: `@workgroup_size(64)` and the stride `nw.x * 64u` in
/// `sample.wgsl` must match it.
const SAMPLE_WORKGROUP: u32 = 64;
/// The kernel's "outside the grid" marker in `Point::base`.
const OUTSIDE: u32 = u32::MAX;

/// One point as the kernel reads it (`struct Point` in `sample.wgsl`: vec3<f32> + u32 = 16 B).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Point { t: [f32; 3], base: u32 }

/// `SampleParams` in `sample.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SampleParams { p: u32, count: u32, pad0: u32, pad1: u32 }

/// The sampler's buffers (module doc) and its counters.
#[derive(Default)]
pub(crate) struct SampleScratch {
    bufs: Option<SampleBuffers>,
    /// Buffer sets created (grow-only reuse witness).
    allocations: usize,
    /// GPU→CPU readbacks made by `sample_density` (one per non-empty call).
    readbacks: usize,
}

struct SampleBuffers { points: wgpu::Buffer, out: wgpu::Buffer, staging: wgpu::Buffer, uniform: wgpu::Buffer, capacity: usize }

impl SampleScratch {
    /// Buffers for `count` points (allocated, or grown to `count`, only when too small).
    fn ensure(&mut self, ctx: &GpuContext, count: usize) -> &SampleBuffers {
        if self.bufs.as_ref().is_none_or(|b| b.capacity < count) {
            use wgpu::BufferUsages as U;
            let make = |label, size: u64, usage| ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
            let n = count as u64;
            self.bufs = Some(SampleBuffers {
                points: make("sample points", n * std::mem::size_of::<Point>() as u64, U::STORAGE | U::COPY_DST),
                out: make("sample out", n * 4, U::STORAGE | U::COPY_SRC),
                staging: make("sample staging", n * 4, U::COPY_DST | U::MAP_READ),
                uniform: make("sample params", std::mem::size_of::<SampleParams>() as u64, U::UNIFORM | U::COPY_DST),
                capacity: count,
            });
            self.allocations += 1;
        }
        self.bufs.as_ref().expect("allocated above")
    }
}

impl GpuSolver {
    /// ρ↑ + ρ↓ of `state` at `points` (Bohr): one dispatch over the padded `rho` field with the
    /// same cell and weights as `qm::Grid::trilinear` (0 outside [origin, origin + (n−1)h]³),
    /// one readback of `points.len()` f32 (module doc). No points: no GPU work. Panics if
    /// `state` is from another grid, or if the request exceeds the device's storage-binding
    /// limit (16 B per point: > 8 M points at the default 128 MiB).
    pub fn sample_density(&self, state: &GpuState, points: &[[f64; 3]]) -> Vec<f32> {
        self.check_state(state);
        if points.is_empty() { return Vec::new(); }
        let count = points.len();
        let limit = self.ctx.device.limits().max_storage_buffer_binding_size;
        assert!((count * std::mem::size_of::<Point>()) as u64 <= limit,
                "sample_density: {count} points exceed the storage-binding limit ({limit} B)");
        let g = self.g;
        let packed: Vec<Point> = points.iter().map(|&p| match self.grid().trilinear_cell(p) {
            Some(([i, j, k], t)) => Point { t: t.map(|v| v as f32), base: g.pidx(i, j, k) as u32 },
            None => Point { t: [0.0; 3], base: OUTSIDE },
        }).collect();
        let mut scratch = self.sampler.borrow_mut();
        let b = scratch.ensure(&self.ctx, count);
        let params = SampleParams { p: g.p as u32, count: count as u32, pad0: 0, pad1: 0 };
        self.ctx.queue.write_buffer(&b.uniform, 0, bytemuck::bytes_of(&params));
        self.ctx.queue.write_buffer(&b.points, 0, bytemuck::cast_slice(&packed));
        let bg = self.k.bind_group(&self.ctx, "sample_density", &[
            b.uniform.as_entire_binding(), state.rho.buffer.as_entire_binding(), b.points.as_entire_binding(), b.out.as_entire_binding(),
        ]);
        let wg = count.div_ceil(SAMPLE_WORKGROUP as usize).min(MAX_WORKGROUPS as usize) as u32;
        let mut enc = encoder(&self.ctx, "sample density");
        self.k.record(&mut enc, "sample_density", &bg, wg);
        enc.copy_buffer_to_buffer(&b.out, 0, &b.staging, 0, Some(count as u64 * 4));
        self.ctx.queue.submit([enc.finish()]);
        let out = map_read_f32(&self.ctx, &b.staging, count);
        scratch.readbacks += 1;
        out
    }

    /// GPU→CPU readbacks made by `sample_density` so far (one per non-empty call).
    pub fn sampling_readbacks(&self) -> usize { self.sampler.borrow().readbacks }

    /// Buffer sets the sampler has allocated so far (grow-only; the reuse witness).
    pub fn sampling_buffer_allocations(&self) -> usize { self.sampler.borrow().allocations }
}

#[cfg(test)]
mod tests {
    use crate::grid::{Field, GpuGrid};
    use crate::scf::{GpuSolver, GpuState};
    use gpu::GpuContext;
    use qm::{Grid, Nucleus, ScfSettings, System};
    use std::time::Instant;

    /// A deterministic xorshift64* stream of uniforms in [0, 1).
    fn uniforms(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
            (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// |gpu − cpu| ≤ 1e-6·max(1, |cpu|) (the f32 bound of the brief); returns the worst relative-to-bound ratio.
    fn check(points: &[[f64; 3]], gpu: &[f32], cpu: impl Fn([f64; 3]) -> f64, what: &str) -> f64 {
        assert_eq!(gpu.len(), points.len(), "{what}: one sample per point");
        let mut worst = 0.0f64;
        for (p, &v) in points.iter().zip(gpu) {
            let c = cpu(*p);
            let bound = 1e-6 * c.abs().max(1.0);
            let d = (v as f64 - c).abs();
            worst = worst.max(d / bound);
            assert!(d <= bound, "{what}: p = {p:?}: gpu {v} cpu {c} (|Δ| {d:.2e} > {bound:.1e})");
        }
        worst
    }

    /// A state holding `rho` (n³, `qm::Grid` layout) as its total density — no solve; only
    /// `grid` and `rho` are read by `sample_density`.
    fn fake_state(ctx: &GpuContext, grid: &Grid, rho: &[f64]) -> GpuState {
        let g = GpuGrid::new(grid);
        let field = |label| Field::new(ctx, &g, label);
        let r = field("fake rho");
        r.upload_interior(ctx, &g, rho);
        GpuState { grid: grid.clone(), sigma: grid.h, orbitals_up: Vec::new(), orbitals_dn: Vec::new(), evals_up: Vec::new(), evals_dn: Vec::new(),
                   rho: r, rho_up: field("fake up"), rho_dn: field("fake dn"), v_h: field("fake vh"), v_nuc: field("fake vnuc") }
    }

    /// The GPU line sampler against `qm::Grid::trilinear` on the same f32 values: an H atom at
    /// 25³, h = 0.25 (σ = h) solved on the GPU at tol 1e-5; 300 points — 270 random inside the
    /// box (the faces' last cells included), 20 outside (some by 1e-9 Bohr), 10 exactly on grid
    /// points (corners and faces among them). Bound: |gpu − cpu| ≤ 1e-6·max(1, ρ); outside == 0
    /// exactly; on a grid point == the stored f32 exactly.
    #[test]
    fn gpu_sampling_matches_the_cpu_trilinear_to_f32() {
        let Some(ctx) = GpuContext::new_headless() else { eprintln!("qm-gpu: no GPU adapter — skipped"); return };
        let grid = Grid::cube(25, 0.25); // [-3, 3]³
        let mut settings = ScfSettings::for_grid(&grid);
        settings.tol = 1e-5;
        let mut s = GpuSolver::new(ctx, &grid);
        let sys = System::new(vec![Nucleus { pos: [0.13, -0.21, 0.05], z: 1 }], 0, None);
        s.solve(&sys, &settings, false).expect("H converges");
        let state = s.state().expect("converged");
        let rho32 = s.density_f32(state);
        let rho: Vec<f64> = rho32.iter().map(|&v| v as f64).collect();
        let mut u = uniforms(0x5a3_1e5);
        let half = 3.0;
        let mut points: Vec<[f64; 3]> = (0..270).map(|_| [half * (2.0 * u() - 1.0), half * (2.0 * u() - 1.0), half * (2.0 * u() - 1.0)]).collect();
        // concentrate a third of them near the nucleus, where ρ and its curvature are largest
        for p in points.iter_mut().take(90) { for (c, x) in p.iter_mut().enumerate() { *x = [0.13, -0.21, 0.05][c] + 0.6 * (2.0 * u() - 1.0); } }
        let outside: Vec<[f64; 3]> = (0..20).map(|i| {
            let mut p = [half * (2.0 * u() - 1.0), half * (2.0 * u() - 1.0), half * (2.0 * u() - 1.0)];
            let axis = i % 3;
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            p[axis] = sign * (half + if i < 6 { 1e-9 } else { 0.5 + 4.0 * u() });
            p
        }).collect();
        let nodes: Vec<usize> = vec![grid.idx(12, 12, 12), grid.idx(0, 0, 0), grid.idx(24, 24, 24), grid.idx(24, 0, 12), grid.idx(13, 11, 12),
                                     grid.idx(0, 24, 3), grid.idx(5, 6, 7), grid.idx(12, 12, 24), grid.idx(24, 12, 12), grid.idx(1, 23, 17)];
        let n_inside = points.len();
        points.extend(&outside);
        points.extend(nodes.iter().map(|&i| grid.point(i)));
        assert_eq!(points.len(), 300);

        let t = Instant::now();
        let gpu = s.sample_density(state, &points);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let worst = check(&points, &gpu, |p| grid.trilinear(&rho, p), "H 25³");
        let peak = gpu.iter().copied().fold(0.0f32, f32::max);
        println!("H 25³: 300 samples in {ms:.2} ms (incl. first-call buffers); worst |Δ| = {worst:.3} of the 1e-6·max(1, ρ) bound; sampled peak {peak:.5}");
        for (k, v) in gpu[n_inside..n_inside + 20].iter().enumerate() { assert_eq!(*v, 0.0, "outside point {:?}", outside[k]); }
        for (k, &i) in nodes.iter().enumerate() {
            assert_eq!(gpu[n_inside + 20 + k], rho32[i], "grid point {:?}: the stored value exactly", grid.point(i));
        }
        assert!(peak > 0.1, "an H atom's density was sampled (peak {peak})");
    }

    /// One readback per call however many points (3 000 here), and the buffers are reused: a
    /// smaller call allocates nothing, a larger one grows them once. An analytic field (no solve)
    /// on a non-power-of-two 17³ grid, checked against the CPU trilinear at the same bound.
    #[test]
    fn sampling_reads_back_once() {
        let Some(ctx) = GpuContext::new_headless() else { eprintln!("qm-gpu: no GPU adapter — skipped"); return };
        let grid = Grid::cube(17, 0.3);
        let s = GpuSolver::new(ctx.clone(), &grid);
        let f = |[x, y, z]: [f64; 3]| (-(x * x + 0.5 * y * y + 0.8 * z * z)).exp() * (1.0 + 0.3 * x - 0.2 * y * z);
        let rho64 = grid.sample(f);
        let state = fake_state(&ctx, &grid, &rho64);
        let rho: Vec<f64> = rho64.iter().map(|&v| v as f32 as f64).collect();
        let mut u = uniforms(0xbeef);
        let lim = 2.6; // 2.4 is the face: some points fall outside
        let points: Vec<[f64; 3]> = (0..3000).map(|_| [lim * (2.0 * u() - 1.0), lim * (2.0 * u() - 1.0), lim * (2.0 * u() - 1.0)]).collect();

        assert!(s.sample_density(&state, &[]).is_empty());
        assert_eq!(s.sampling_readbacks(), 0, "no points, no GPU work");
        let before = s.sampling_readbacks();
        let t = Instant::now();
        let gpu = s.sample_density(&state, &points);
        let first = t.elapsed().as_secs_f64() * 1e3;
        assert_eq!(s.sampling_readbacks() - before, 1, "one readback for 3000 points");
        let worst = check(&points, &gpu, |p| grid.trilinear(&rho, p), "17³ analytic");
        let outside = gpu.iter().filter(|v| **v == 0.0).count();
        let allocs = s.sampling_buffer_allocations();
        assert_eq!(allocs, 1);
        let t = Instant::now();
        let again = s.sample_density(&state, &points[..1000]);
        let second = t.elapsed().as_secs_f64() * 1e3;
        assert_eq!(again, gpu[..1000], "same points, same values");
        assert_eq!(s.sampling_readbacks() - before, 2);
        assert_eq!(s.sampling_buffer_allocations(), allocs, "a smaller call reuses the buffers");
        let more: Vec<[f64; 3]> = points.iter().chain(&points).copied().collect();
        let both = s.sample_density(&state, &more);
        assert_eq!(both[3000..], gpu[..]);
        assert_eq!(s.sampling_buffer_allocations(), allocs + 1, "a larger call grows them once");
        println!("17³ analytic: 3000 points ({outside} outside → 0) in {first:.2} ms (first call), 1000 in {second:.2} ms; worst |Δ| = {worst:.3} of the bound");
    }
    /// The detector's round-1 cost at production grid sizes: 45 chords × 33 points (a
    /// 10-nucleus scene) on 57³ and 97³ fields (an analytic field — the cost does not depend on
    /// the values), median of 50 calls, and one n³ node read (the atom references, spec §3.2).
    /// Reported, not asserted (hardware-dependent).
    #[test]
    fn sampling_timing_at_production_grids() {
        let Some(ctx) = GpuContext::new_headless() else { eprintln!("qm-gpu: no GPU adapter — skipped"); return };
        for n in [57usize, 97] {
            let grid = Grid::cube(n, 0.25);
            let s = GpuSolver::new(ctx.clone(), &grid);
            let rho = grid.sample(|[x, y, z]| (-(x * x + y * y + z * z).sqrt()).exp());
            let state = fake_state(&ctx, &grid, &rho);
            let mut u = uniforms(n as u64);
            let half = grid.origin[0].abs() - 1.0;
            let nuclei: Vec<[f64; 3]> = (0..10).map(|_| [half * (2.0 * u() - 1.0), half * (2.0 * u() - 1.0), half * (2.0 * u() - 1.0)]).collect();
            let points: Vec<[f64; 3]> = (0..10).flat_map(|i| ((i + 1)..10).map(move |j| (i, j))).flat_map(|(i, j)| {
                let (a, b) = (nuclei[i], nuclei[j]);
                (0..33).map(move |k| { let t = k as f64 / 32.0; std::array::from_fn(|c| a[c] + t * (b[c] - a[c])) })
            }).collect();
            assert_eq!(points.len(), 1485);
            let mut walls: Vec<f64> = (0..50).map(|_| {
                let t = Instant::now();
                let v = s.sample_density(&state, &points);
                assert_eq!(v.len(), points.len());
                t.elapsed().as_secs_f64() * 1e3
            }).collect();
            walls.sort_by(f64::total_cmp);
            let nodes: Vec<[f64; 3]> = (0..grid.len()).map(|i| grid.point(i)).collect();
            let t = Instant::now();
            let all = s.sample_density(&state, &nodes);
            let node_ms = t.elapsed().as_secs_f64() * 1e3;
            let rho32: Vec<f32> = rho.iter().map(|&v| v as f32).collect();
            assert!(all == rho32, "{n}³: sampling every node returns the stored field exactly");
            println!("{n}³: 1485 points (45 chords × 33): median {:.3} ms, min {:.3}, p90 {:.3}; all {} nodes in {node_ms:.1} ms (exact)",
                     walls[25], walls[0], walls[45], nodes.len());
        }
    }
}
