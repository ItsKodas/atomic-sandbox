//! `Batch`: one command encoder with ONE open compute pass that many ops record into.
//!
//! Why: recording into a bare encoder (the `*_rec(&mut CommandEncoder, …)` form) opens one
//! compute pass per dispatch, which costs ≈ 9 µs/op of CPU recording vs ≈ 4.8 µs/op in a shared
//! pass, plus GPU-side pass overhead (measurements below). A `Batch` keeps a single pass open across ops; each op only creates its uniform + bind group
//! and records `set_pipeline`/`set_bind_group`/`dispatch`. WebGPU makes every dispatch its own
//! synchronisation scope, so wgpu inserts the storage-buffer barriers between consecutive
//! dispatches inside the pass: an op always sees the writes of the ops recorded before it.
//!
//! Measured (RX 7900 XTX, release, 200 mixed ops, `bench_batch_vs_rec`; wall = record + submit +
//! GPU): 17³ 19.6 → 9.2 µs/op, 49³ 18 → 9.7 µs/op, 97³ 24 → 20.7 µs/op (GPU-bound there). CPU
//! recording drops from 9 to 4.8 µs/op; what remains is the per-op uniform buffer + bind group
//! (a shared uniform records in ~2 µs/op).
//!
//! Recording: every op takes `r: &mut impl Recorder` in its `*_rec` form, so it records into a
//! plain `wgpu::CommandEncoder` (one pass per dispatch, the Task 1/2 behaviour) or into a
//! `Batch` (shared pass). The `*_in(batch, …)` forms are the same ops taking their
//! `GpuContext` from the batch.
//!
//! Things that cannot live inside a compute pass (buffer clears and copies): there is no
//! `zero_in`; use `pointwise::fill_in(batch, …, 0.0)`, which writes the interior (ghosts are
//! invariantly 0). For a copy, `Batch::encoder` ends the open pass and hands out the encoder;
//! the next dispatch re-opens a pass (each break costs one pass, i.e. tens of µs).
//!
//! Reductions (`Reducer::{dot_in, sum_in, gram_in}`) are sync points: they end the pass, append
//! the reduction to the pending work, submit everything recorded so far, wait for the readback
//! and leave the batch empty but usable — later ops record into a fresh encoder/pass.
//!
//! Dropping a `Batch` without `finish`/`submit` discards everything recorded in it.
//!
//! # The uniform ring (M4)
//! Every op has a small uniform (its parameters). Through a bare encoder each dispatch creates
//! its own buffer (`kernels::uniform`, mapped at creation); in a `Batch` that `create_buffer`
//! was most of the remaining ≈ 4.8 µs/op of CPU recording (a shared uniform records in ≈ 2), at
//! ~75 dispatches per LOBPCG iteration. A `Batch` now takes the uniform from the next 256-byte
//! slot (`kernels::UNIFORM_SLOT`) of a ring buffer (`RING_SLOTS` slots, 64 KiB) borrowed from
//! the `Kernels`' pool on the first dispatch: the bytes are staged on the CPU and written with
//! one `queue.write_buffer` per ring at the next sync point (`take_encoder`, `submit`,
//! `finish`), which wgpu executes before the command buffer submitted next — so the dispatches
//! see their own values. The slot cursor then resets: the next segment's write is queued after
//! this submit and cannot overwrite what it reads (WebGPU queue ordering). A segment longer
//! than one ring writes the full ring and chains the next one. The bind group binds the ring at
//! the slot's offset (a static offset: the bind group is per dispatch anyway, since it names
//! the op's fields, so dynamic offsets would buy nothing). The rings go back to the pool when
//! the batch is dropped after `submit`, or are released to wgpu after `finish` (whose command
//! buffer the caller submits later — another batch must not rewrite them before that). The
//! uniform *values* are unchanged, so results are bit-identical to the bare-encoder path
//! (`uniform_ring_is_bit_identical_and_faster`). Measured (RX 7900 XTX, with the viewer
//! running on the same GPU, so walls are contended): CPU recording 4.7–4.9 → 2.8–3.0 µs/op
//! (`bench_batch_vs_rec`); LOBPCG interleaved A/B against the pre-ring build, 49³ k = 2
//! 0.74–0.76 → 0.67–0.70 ms/it, 97³ k = 5 4.28–4.38 → 4.25–4.36 ms/it (GPU-bound there).
//! Contract: work taken out with `take_encoder`
//! is submitted before anything else is recorded into this batch (every caller in the crate
//! submits it immediately).
use crate::kernels::{Kernels, RingPool, RING_SLOTS, UNIFORM_SLOT};
use gpu::GpuContext;
use std::sync::{Arc, Mutex};

/// Something a compute dispatch can be recorded into: a bare `CommandEncoder` (one compute
/// pass per dispatch) or a `Batch` (all dispatches in one pass).
pub trait Recorder {
    /// Records one dispatch of kernel `name` with `bind_group` at group 0 and a 3D workgroup grid.
    fn dispatch(&mut self, k: &Kernels, name: &str, bind_group: &wgpu::BindGroup, workgroups: [u32; 3]);

    /// A uniform holding `bytes` (≤ `UNIFORM_SLOT`) for the next dispatch recorded here: a fresh
    /// buffer for a bare encoder, a ring slot for a `Batch` (module doc).
    fn uniform_bytes(&mut self, ctx: &GpuContext, k: &Kernels, label: &str, bytes: &[u8]) -> UniformSlot;

    /// `uniform_bytes` of a `Pod` value.
    fn uniform<T: bytemuck::Pod>(&mut self, ctx: &GpuContext, k: &Kernels, label: &str, value: &T) -> UniformSlot where Self: Sized {
        self.uniform_bytes(ctx, k, label, bytemuck::bytes_of(value))
    }
}

/// A uniform for one dispatch: a buffer and the byte range holding the value.
pub struct UniformSlot { buffer: wgpu::Buffer, offset: u64, size: u64 }

impl UniformSlot {
    /// The binding for the kernel's uniform slot.
    pub fn binding(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &self.buffer, offset: self.offset, size: wgpu::BufferSize::new(self.size) })
    }
}

impl Recorder for wgpu::CommandEncoder {
    fn dispatch(&mut self, k: &Kernels, name: &str, bind_group: &wgpu::BindGroup, workgroups: [u32; 3]) {
        k.record_3d(self, name, bind_group, workgroups);
    }

    fn uniform_bytes(&mut self, ctx: &GpuContext, _k: &Kernels, label: &str, bytes: &[u8]) -> UniformSlot {
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label), size: bytes.len() as u64, usage: wgpu::BufferUsages::UNIFORM, mapped_at_creation: true,
        });
        buffer.slice(..).get_mapped_range_mut().expect("uniform: mapped range").copy_from_slice(bytes);
        buffer.unmap();
        UniformSlot { buffer, offset: 0, size: bytes.len() as u64 }
    }
}

/// The ring state of one `Batch` (module doc): the ring buffers it holds, the index of the one
/// being filled, its next free slot and the staged bytes of its slots `0..cursor`.
#[derive(Default)]
struct Ring {
    pool: Option<Arc<Mutex<RingPool>>>,
    buffers: Vec<wgpu::Buffer>,
    current: usize,
    cursor: u64,
    staged: Vec<u8>,
}

impl Ring {
    fn slot(&mut self, ctx: &GpuContext, k: &Kernels, bytes: &[u8]) -> UniformSlot {
        assert!(bytes.len() as u64 <= UNIFORM_SLOT, "uniform of {} bytes > the {UNIFORM_SLOT}-byte ring slot", bytes.len());
        if self.cursor == RING_SLOTS {
            self.write(ctx); // full: write it and chain the next ring
            self.current += 1;
        }
        if self.current == self.buffers.len() {
            let pool = self.pool.get_or_insert_with(|| k.rings.clone());
            self.buffers.push(RingPool::take(pool, ctx));
        }
        let offset = self.cursor * UNIFORM_SLOT;
        self.staged.resize(offset as usize, 0);
        self.staged.extend_from_slice(bytes);
        self.cursor += 1;
        UniformSlot { buffer: self.buffers[self.current].clone(), offset, size: bytes.len() as u64 }
    }

    /// Queues the staged bytes of the current ring (executed before the next submit's commands).
    fn write(&mut self, ctx: &GpuContext) {
        if self.cursor > 0 { ctx.queue.write_buffer(&self.buffers[self.current], 0, &self.staged); }
        self.staged.clear();
        self.cursor = 0;
    }

    /// At a sync point: write what is staged and start the next segment at ring 0, slot 0.
    fn sync(&mut self, ctx: &GpuContext) {
        self.write(ctx);
        self.current = 0;
    }
}

impl Drop for Ring {
    fn drop(&mut self) {
        if let Some(pool) = &self.pool { RingPool::give(pool, self.buffers.drain(..)); }
    }
}

pub struct Batch<'a> {
    /// The open pass, if any. Declared before `encoder` so it is dropped (ended) first. Its
    /// lifetime is erased with `forget_lifetime`; wgpu then locks `encoder` until the pass is
    /// dropped, and every method below drops it before touching the encoder.
    pass: Option<wgpu::ComputePass<'static>>,
    encoder: wgpu::CommandEncoder,
    ctx: &'a GpuContext,
    label: String,
    /// Dispatches recorded through this batch over its lifetime (sync points included).
    dispatches: usize,
    /// The per-dispatch uniforms (module doc, "The uniform ring").
    ring: Ring,
}

impl<'a> Batch<'a> {
    /// An empty batch. The compute pass is opened lazily by the first dispatch.
    pub fn new(ctx: &'a GpuContext, label: &str) -> Batch<'a> {
        Batch { pass: None, encoder: new_encoder(ctx, label), ctx, label: label.to_owned(), dispatches: 0, ring: Ring::default() }
    }

    /// The context this batch records for (the `*_in` ops take theirs from here).
    pub fn ctx(&self) -> &'a GpuContext { self.ctx }

    /// Dispatches recorded through this batch so far (a running total across sync points;
    /// reductions appended by a `Reducer` are not counted here).
    pub fn dispatches(&self) -> usize { self.dispatches }

    /// Ends the open pass (if any) and returns the encoder, e.g. to record a buffer copy. The
    /// next dispatch opens a new pass after whatever is recorded here.
    pub fn encoder(&mut self) -> &mut wgpu::CommandEncoder {
        self.pass = None;
        &mut self.encoder
    }

    /// Ends the open pass and takes the pending work out as an encoder (to hand to a
    /// `Reducer::*_after`, or to submit), leaving this batch empty with a fresh encoder. The
    /// staged uniforms are queued for the next submit: submit the encoder before recording
    /// anything else into this batch (module doc).
    pub fn take_encoder(&mut self) -> wgpu::CommandEncoder {
        self.pass = None;
        self.ring.sync(self.ctx);
        std::mem::replace(&mut self.encoder, new_encoder(self.ctx, &self.label))
    }

    /// Ends the pass and finishes the encoder; submit the buffer with `ctx.queue.submit`. The
    /// batch's rings are released to wgpu, not pooled (the caller submits later; module doc).
    pub fn finish(mut self) -> wgpu::CommandBuffer {
        self.ring.sync(self.ctx);
        self.ring.pool = None; // keep them out of the pool: nothing may rewrite them before the submit
        let Batch { pass, encoder, .. } = self;
        drop(pass);
        encoder.finish()
    }

    /// Ends the pass, submits (does not wait) and returns the rings to the pool.
    pub fn submit(mut self) {
        self.pass = None;
        self.ring.sync(self.ctx);
        let Batch { pass, encoder, ctx, ring, .. } = self;
        drop(pass);
        ctx.queue.submit([encoder.finish()]);
        drop(ring); // back to the pool, after the submit that reads it
    }
}

impl Recorder for Batch<'_> {
    fn uniform_bytes(&mut self, ctx: &GpuContext, k: &Kernels, _label: &str, bytes: &[u8]) -> UniformSlot {
        self.ring.slot(ctx, k, bytes)
    }

    fn dispatch(&mut self, k: &Kernels, name: &str, bind_group: &wgpu::BindGroup, [x, y, z]: [u32; 3]) {
        self.dispatches += 1;
        let (encoder, label) = (&mut self.encoder, &self.label);
        let pass = self.pass.get_or_insert_with(|| {
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: None }).forget_lifetime()
        });
        pass.set_pipeline(k.pipeline(name));
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(x, y, z);
    }
}

fn new_encoder(ctx: &GpuContext, label: &str) -> wgpu::CommandEncoder {
    ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{assert_ghosts_zero, Field, GpuGrid};
    use crate::physics::{nuclear_potential, nuclear_potential_in, xc_lsda, xc_lsda_in};
    use crate::pointwise::{axpy_rec, fill_rec, lincomb_rec, mul_add_sq_rec, mul_rec, scale_rec};
    use crate::reduce::Reducer;
    use crate::stencil::{kinetic_rec, laplacian_rec};
    use std::time::Instant;

    /// The test's state: five fields on the GPU (x, y, z, w, l) and their f64 CPU mirror.
    struct State { f: [Field; 5] }
    const X: usize = 0;
    const Y: usize = 1;
    const Z: usize = 2;
    const W: usize = 3;
    const L: usize = 4;

    #[derive(Clone, Copy, Debug)]
    enum Op { Lap, AxpyX(f32), Scale(f32), Fill(f32), Lincomb([f32; 3]), Mul, MulAddSq, Kin, AxpyY(f32) }

    /// 9-op cycle in which every result is consumed later: l = ∇²x; x += τl (heat step); y *= s;
    /// w = v; y = a·y + b·x + c·w (self-aliasing lincomb: scale + lincomb); w = x·y; z += w²;
    /// l = −½∇²y; y += −τ'l (heat step on y).
    fn ops(count: usize, h: f64) -> Vec<Op> {
        (0..count).map(|i| {
            let s = (i / 9) as f32;
            match i % 9 {
                0 => Op::Lap,
                1 => Op::AxpyX(((0.02 + 0.002 * (i % 5) as f64) * h * h) as f32),
                2 => Op::Scale(1.0 - 0.003 * (i % 7) as f32),
                3 => Op::Fill(0.05 * (i % 4) as f32),
                4 => Op::Lincomb([0.6 + 0.01 * (s % 3.0), 0.3, -0.2]),
                5 => Op::Mul,
                6 => Op::MulAddSq,
                7 => Op::Kin,
                _ => Op::AxpyY((-0.03 * h * h) as f32),
            }
        }).collect()
    }

    fn record(r: &mut impl Recorder, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, s: &State, op: Op) {
        let f = &s.f;
        match op {
            Op::Lap => laplacian_rec(r, ctx, k, g, &f[X], &f[L]),
            Op::AxpyX(t) => axpy_rec(r, ctx, k, g, &f[X], t, &f[L]),
            Op::Scale(a) => scale_rec(r, ctx, k, g, &f[Y], a),
            Op::Fill(v) => fill_rec(r, ctx, k, g, &f[W], v),
            Op::Lincomb([a, b, c]) => lincomb_rec(r, ctx, k, g, &f[Y], &[(a, &f[Y]), (b, &f[X]), (c, &f[W])]),
            Op::Mul => mul_rec(r, ctx, k, g, &f[W], &f[X], &f[Y]),
            Op::MulAddSq => mul_add_sq_rec(r, ctx, k, g, &f[Z], &f[W]),
            Op::Kin => kinetic_rec(r, ctx, k, g, &f[Y], &f[L]),
            Op::AxpyY(t) => axpy_rec(r, ctx, k, g, &f[Y], t, &f[L]),
        }
    }

    #[allow(clippy::needless_range_loop)] // several fields of `c` indexed in step
    fn cpu_apply(cg: &qm::Grid, c: &mut [Vec<f64>; 5], op: Op) {
        let lap = |v: &[f64]| { let mut o = vec![0.0; v.len()]; qm::stencil::laplacian(cg, v, &mut o); o };
        let n = cg.len();
        match op {
            Op::Lap => c[L] = lap(&c[X]),
            Op::AxpyX(t) => for i in 0..n { c[X][i] += t as f64 * c[L][i] },
            Op::Scale(a) => for v in c[Y].iter_mut() { *v *= a as f64 },
            Op::Fill(v) => c[W] = vec![v as f64; n],
            Op::Lincomb([a, b, cc]) => for i in 0..n { c[Y][i] = a as f64 * c[Y][i] + b as f64 * c[X][i] + cc as f64 * c[W][i] },
            Op::Mul => for i in 0..n { c[W][i] = c[X][i] * c[Y][i] },
            Op::MulAddSq => for i in 0..n { c[Z][i] += c[W][i] * c[W][i] },
            Op::Kin => c[L] = lap(&c[Y]).iter().map(|v| -0.5 * v).collect(),
            Op::AxpyY(t) => for i in 0..n { c[Y][i] += t as f64 * c[L][i] },
        }
    }

    fn initial(cg: &qm::Grid) -> [Vec<f64>; 5] {
        [
            cg.sample(|[x, y, z]| (-(x * x + 0.7 * y * y + 1.3 * z * z)).exp()),
            cg.sample(|[x, y, z]| 1.0 + 0.2 * (x - 0.5 * y).sin() * (-(0.3 * z * z)).exp()),
            cg.sample(|[x, y, z]| 0.5 * (-(x * x + y * y + z * z) / 2.0).exp()),
            vec![0.0; cg.len()],
            vec![0.0; cg.len()],
        ]
    }

    fn upload(ctx: &GpuContext, g: &GpuGrid, init: &[Vec<f64>; 5]) -> State {
        State { f: std::array::from_fn(|i| { let f = Field::new(ctx, g, "s"); f.upload_interior(ctx, g, &init[i]); f }) }
    }

    /// 200 mixed ops (Laplacian, kinetic, axpy, scale, fill, self-aliasing lincomb, mul,
    /// mul_add_sq) in ONE `Batch`, with a `dot_in` sync point after op 100 and the nuclear and
    /// XC kernels recorded at the end. Each op reads what earlier ops wrote, so a missing barrier
    /// between dispatches in the shared pass shows up as a mismatch. Checked against the f64 CPU
    /// mirror and bit-for-bit against the same ops recorded with `*_rec` into a bare encoder
    /// (one pass per dispatch).
    #[test]
    fn batch_of_200_mixed_ops_matches_cpu_and_rec_path() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        let cg = qm::Grid::cube(25, 0.25);
        let g = GpuGrid::new(&cg);
        let k = Kernels::new(&ctx);
        let r = Reducer::new(&ctx, &k, &g);
        let init = initial(&cg);
        let ops = ops(200, cg.h);
        let split = 100;
        let nuclei = [qm::Nucleus { pos: [0.3, -0.2, 0.1], z: 2 }, qm::Nucleus { pos: [-1.0, 0.5, 0.0], z: 1 }];

        // CPU mirror
        let mut cpu = init.clone();
        for op in &ops[..split] { cpu_apply(&cg, &mut cpu, *op); }
        let cpu_dot = cg.dot(&cpu[X], &cpu[Y]);
        for op in &ops[split..] { cpu_apply(&cg, &mut cpu, *op); }

        // Batch: one pass for ops[..split], dot_in (submit + readback), one pass for the rest
        let sb = upload(&ctx, &g, &init);
        let (vn_b, exc_b, vu_b, vd_b) = (Field::new(&ctx, &g, "vn"), Field::new(&ctx, &g, "exc"), Field::new(&ctx, &g, "vu"), Field::new(&ctx, &g, "vd"));
        let mut b = Batch::new(&ctx, "200 ops");
        for op in &ops[..split] { record(&mut b, &ctx, &k, &g, &sb, *op); }
        let dot_b = r.dot_in(&mut b, &k, &g, &sb.f[X], &sb.f[Y]);
        for op in &ops[split..] { record(&mut b, &ctx, &k, &g, &sb, *op); }
        nuclear_potential_in(&mut b, &k, &g, &vn_b, &nuclei, 0.3);
        xc_lsda_in(&mut b, &k, &g, &sb.f[Z], &sb.f[Z], &exc_b, &vu_b, &vd_b);
        b.submit();

        // `_rec` path: same ops into a bare encoder, dot_after as the sync point
        let se = upload(&ctx, &g, &init);
        let mut enc = crate::kernels::encoder(&ctx, "200 ops rec");
        for op in &ops[..split] { record(&mut enc, &ctx, &k, &g, &se, *op); }
        let dot_e = r.dot_after(enc, &ctx, &k, &g, &se.f[X], &se.f[Y]);
        let mut enc = crate::kernels::encoder(&ctx, "200 ops rec");
        for op in &ops[split..] { record(&mut enc, &ctx, &k, &g, &se, *op); }
        ctx.queue.submit([enc.finish()]);

        println!("dot after {split} ops: batch {dot_b} rec {dot_e} cpu {cpu_dot} (rel {:.2e})", ((dot_b - cpu_dot) / cpu_dot).abs());
        assert_eq!(dot_b.to_bits(), dot_e.to_bits(), "dot_in vs dot_after");
        assert!((dot_b - cpu_dot).abs() < 1e-5 * cpu_dot.abs(), "dot_in {dot_b} vs cpu {cpu_dot}");
        for (i, name) in ["x", "y", "z", "w", "l"].iter().enumerate() {
            let got = sb.f[i].read_interior(&ctx, &g);
            assert_eq!(got, se.f[i].read_interior(&ctx, &g), "{name}: Batch and _rec paths must be bit-identical");
            assert_ghosts_zero(&g, &sb.f[i].read_padded_raw(&ctx));
            let scale = cpu[i].iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let moved = cpu[i].iter().zip(&init[i]).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            let err = got.iter().zip(&cpu[i]).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            println!("{name}: max |gpu − cpu| = {err:.2e} (max|ref| {scale:.3}, moved by {moved:.3})");
            assert!(moved > 1e-2, "{name} barely changed: the test would be vacuous");
            assert!(err < 1e-5 * scale, "{name}: {err:.3e} vs max|ref| {scale}");
        }
        // physics kernels recorded in the batch == their immediate forms on the same inputs
        let (vn, exc, vu, vd) = (Field::new(&ctx, &g, "vn"), Field::new(&ctx, &g, "exc"), Field::new(&ctx, &g, "vu"), Field::new(&ctx, &g, "vd"));
        nuclear_potential(&ctx, &k, &g, &vn, &nuclei, 0.3);
        xc_lsda(&ctx, &k, &g, &sb.f[Z], &sb.f[Z], &exc, &vu, &vd);
        for (a, bb) in [(&vn_b, &vn), (&exc_b, &exc), (&vu_b, &vu), (&vd_b, &vd)] {
            let got = a.read_padded_raw(&ctx);
            assert!(got.iter().any(|v| *v != 0.0));
            assert_eq!(got, bb.read_padded_raw(&ctx));
        }
    }

    /// `encoder()` ends the pass for a copy; dispatches after it see the copy.
    #[test]
    fn batch_encoder_break_for_copy() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        let cg = qm::Grid::cube(9, 0.5);
        let g = GpuGrid::new(&cg);
        let k = Kernels::new(&ctx);
        let (a, c) = (Field::new(&ctx, &g, "a"), Field::new(&ctx, &g, "c"));
        let mut b = Batch::new(&ctx, "copy");
        crate::pointwise::fill_in(&mut b, &k, &g, &a, 2.0);
        b.encoder().copy_buffer_to_buffer(&a.buffer, 0, &c.buffer, 0, None);
        crate::pointwise::scale_in(&mut b, &k, &g, &c, 3.0);
        let cmd = b.finish();
        ctx.queue.submit([cmd]);
        assert!(c.read_interior(&ctx, &g).iter().all(|v| *v == 6.0));
        assert!(a.read_interior(&ctx, &g).iter().all(|v| *v == 2.0));
        assert_ghosts_zero(&g, &c.read_padded_raw(&ctx));
    }

    /// `sum_in`/`gram_in` see the batch's pending writes, and the batch keeps working after each.
    #[test]
    fn reductions_in_batch_see_pending_work() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        let cg = qm::Grid::cube(15, 0.3);
        let g = GpuGrid::new(&cg);
        let k = Kernels::new(&ctx);
        let r = Reducer::new(&ctx, &k, &g);
        let vol = cg.len() as f64 * cg.dv();
        let (a, c) = (Field::new(&ctx, &g, "a"), Field::new(&ctx, &g, "c"));
        let mut b = Batch::new(&ctx, "reductions");
        crate::pointwise::fill_in(&mut b, &k, &g, &a, 2.0);
        let s = r.sum_in(&mut b, &k, &g, &a);
        assert!((s - 2.0 * vol).abs() < 1e-9 * vol, "sum_in {s}");
        crate::pointwise::fill_in(&mut b, &k, &g, &c, 3.0);
        crate::pointwise::scale_in(&mut b, &k, &g, &a, 0.5);
        let m = r.gram_in(&mut b, &k, &g, &[&a, &c], &[&c]);
        assert!((m[0] - 3.0 * vol).abs() < 1e-9 * vol && (m[1] - 9.0 * vol).abs() < 1e-9 * vol, "gram_in {m:?}");
        crate::pointwise::axpy_in(&mut b, &k, &g, &a, 1.0, &c);
        assert!(r.gram_in(&mut b, &k, &g, &[], &[&c]).is_empty()); // submits the pending axpy
        b.submit();
        assert!(a.read_interior(&ctx, &g).iter().all(|v| *v == 4.0));
    }

    /// M4 uniform ring (module doc): a `Batch` takes its per-dispatch uniforms from 256-byte
    /// slots of pooled ring buffers instead of one `create_buffer` per dispatch. The same ops
    /// recorded into a bare encoder still get a fresh buffer each (the pre-ring path), so the
    /// two must agree bit for bit — over a segment long enough to overflow one ring (700 ops
    /// between sync points, ~780 dispatches > `RING_SLOTS`) and across sync points that reset
    /// it. A second round on the same `Kernels` creates no new ring buffer (the pool is reused).
    /// The per-op cost of both paths is printed (a report, never asserted).
    #[test]
    fn uniform_ring_is_bit_identical_and_faster() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        let cg = qm::Grid::cube(17, 0.3);
        let g = GpuGrid::new(&cg);
        let k = Kernels::new(&ctx);
        let r = Reducer::new(&ctx, &k, &g);
        let init = initial(&cg);
        let ops = ops(1000, cg.h);
        let (split, long) = (100, 800);
        let round = |label: &str| -> (State, f64, f64, std::time::Duration) {
            let s = upload(&ctx, &g, &init);
            let t = Instant::now();
            let mut b = Batch::new(&ctx, label);
            for op in &ops[..split] { record(&mut b, &ctx, &k, &g, &s, *op); }
            let d1 = r.dot_in(&mut b, &k, &g, &s.f[X], &s.f[Y]);
            for op in &ops[split..long] { record(&mut b, &ctx, &k, &g, &s, *op); } // one 700-op segment
            let d2 = r.dot_in(&mut b, &k, &g, &s.f[Y], &s.f[Z]);
            for op in &ops[long..] { record(&mut b, &ctx, &k, &g, &s, *op); }
            b.submit();
            gpu::wait_idle(&ctx).expect("wait idle");
            (s, d1, d2, t.elapsed())
        };
        let (sb, d1, d2, t_ring) = round("ring");
        let rings = k.uniform_rings_created();
        let (sb2, e1, e2, t_ring2) = round("ring again");
        assert!(rings >= 2, "the 700-op segment must have overflowed one ring: {rings} ring buffers");
        assert_eq!(k.uniform_rings_created(), rings, "a second round must reuse the pooled rings");
        // the pre-ring path: same ops into bare encoders, one uniform buffer per dispatch
        let se = upload(&ctx, &g, &init);
        let t = Instant::now();
        let mut enc = crate::kernels::encoder(&ctx, "rec");
        for op in &ops[..split] { record(&mut enc, &ctx, &k, &g, &se, *op); }
        let f1 = r.dot_after(enc, &ctx, &k, &g, &se.f[X], &se.f[Y]);
        let mut enc = crate::kernels::encoder(&ctx, "rec");
        for op in &ops[split..long] { record(&mut enc, &ctx, &k, &g, &se, *op); }
        let f2 = r.dot_after(enc, &ctx, &k, &g, &se.f[Y], &se.f[Z]);
        let mut enc = crate::kernels::encoder(&ctx, "rec");
        for op in &ops[long..] { record(&mut enc, &ctx, &k, &g, &se, *op); }
        ctx.queue.submit([enc.finish()]);
        gpu::wait_idle(&ctx).expect("wait idle");
        let t_rec = t.elapsed();
        for (a, b) in [(d1, f1), (d2, f2), (e1, f1), (e2, f2)] { assert_eq!(a.to_bits(), b.to_bits(), "sync-point dots: ring {a} vs per-dispatch {b}"); }
        for i in 0..5 {
            let want = se.f[i].read_padded_raw(&ctx);
            assert!(want.iter().any(|v| *v != 0.0));
            assert_eq!(sb.f[i].read_padded_raw(&ctx), want, "field {i}: ring and per-dispatch uniforms must be bit-identical");
            assert_eq!(sb2.f[i].read_padded_raw(&ctx), want, "field {i}, second round");
        }
        let per = |d: std::time::Duration| d.as_secs_f64() * 1e6 / ops.len() as f64;
        println!("17³, {} ops (~{} dispatches) with 2 sync points: Batch + ring {:.1} / {:.1} µs/op, bare encoder + per-dispatch uniforms {:.1} µs/op; {rings} ring buffers",
                 ops.len(), ops.len() * 10 / 9, per(t_ring), per(t_ring2), per(t_rec));
    }

    /// Timing for the report: `cargo test -p qm-gpu --release -- --ignored --nocapture --test-threads=1 bench_batch`.
    /// 200 mixed ops (the test's cycle; ~222 dispatches) at 17³ (overhead-bound), 49³ and 97³:
    /// `_rec` into a bare encoder (one pass per dispatch) vs one `Batch`. Wall = record + submit +
    /// GPU, sequential (no overlap of recording with execution).
    #[test]
    #[ignore]
    fn bench_batch_vs_rec() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        let k = Kernels::new(&ctx);
        for n in [17, 49, 97] {
            let cg = qm::Grid::cube(n, 0.2);
            let g = GpuGrid::new(&cg);
            let s = upload(&ctx, &g, &initial(&cg));
            let ops = ops(200, cg.h);
            let sync = || { gpu::wait_idle(&ctx).expect("wait idle"); };
            let rounds = 20;
            let rec_round = || -> (f64, f64) {
                let t = Instant::now();
                let mut enc = crate::kernels::encoder(&ctx, "rec");
                for op in &ops { record(&mut enc, &ctx, &k, &g, &s, *op); }
                let cb = enc.finish();
                let rec = t.elapsed().as_secs_f64();
                ctx.queue.submit([cb]);
                sync();
                (rec, t.elapsed().as_secs_f64())
            };
            let batch_round = || -> (f64, f64) {
                let t = Instant::now();
                let mut b = Batch::new(&ctx, "batch");
                for op in &ops { record(&mut b, &ctx, &k, &g, &s, *op); }
                let cb = b.finish();
                let rec = t.elapsed().as_secs_f64();
                ctx.queue.submit([cb]);
                sync();
                (rec, t.elapsed().as_secs_f64())
            };
            for _ in 0..3 { rec_round(); batch_round(); }
            let report = |name: &str, f: &dyn Fn() -> (f64, f64)| {
                let (mut rec, mut wall) = (0.0, 0.0);
                for _ in 0..rounds { let (r, w) = f(); rec += r; wall += w; }
                let per = |t: f64| t * 1e6 / (rounds * ops.len()) as f64;
                println!("{n}³ {name}: {:.1} µs/op wall, {:.1} µs/op CPU record+finish", per(wall), per(rec));
                per(wall)
            };
            let rec = report("_rec (pass per dispatch)", &rec_round);
            let bat = report("Batch (one pass)      ", &batch_round);
            println!("{n}³ 200 mixed ops: Batch {:.2}× faster per op", rec / bat);
        }
    }
}