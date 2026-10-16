//! Padded grid geometry and f32 fields on the GPU.
//!
//! Layout: interior point (i, j, k) of an n³ cube lives at `(i+4) + p*((j+4) + p*(k+4))`,
//! p = n + 8 (4 ghost cells per side, enough for the 9-point-per-axis stencil). Ghost cells hold
//! 0; every kernel in this crate writes the interior only, so they stay 0 — except
//! `poisson::ghost_fill`, which writes the face-slab ghosts of `GpuPoisson`'s private boundary
//! field.
use gpu::GpuContext;

/// Ghost cells per side.
pub const GHOST: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuGrid { pub n: usize, pub h: f64, pub origin: [f64; 3], pub p: usize }

impl GpuGrid {
    /// Cubic grids only.
    pub fn new(grid: &qm::Grid) -> GpuGrid {
        let [n0, n1, n2] = grid.n;
        assert!(n0 == n1 && n1 == n2, "GpuGrid::new: grid must be cubic, got {:?}", grid.n);
        GpuGrid { n: n0, h: grid.h, origin: grid.origin, p: n0 + 2 * GHOST }
    }
    pub fn padded_len(&self) -> usize { self.p * self.p * self.p }
    pub fn interior_len(&self) -> usize { self.n * self.n * self.n }
    pub fn pidx(&self, i: usize, j: usize, k: usize) -> usize {
        (i + GHOST) + self.p * ((j + GHOST) + self.p * (k + GHOST))
    }
    pub fn dv(&self) -> f64 { self.h * self.h * self.h }
}

/// A padded f32 field on the GPU. Ghost cells are always zero after `upload`/`zero`.
pub struct Field { pub buffer: wgpu::Buffer, pub len: usize /* padded */ }

impl Field {
    /// Zero-initialised (wgpu zero-fills new buffers).
    pub fn new(ctx: &GpuContext, g: &GpuGrid, label: &str) -> Field {
        let len = g.padded_len();
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label), size: (len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Field { buffer, len }
    }

    /// f64 interior in `qm::Grid` layout (`i + n*(j + n*k)`) → f32 padded, ghosts written as 0.
    pub fn upload_interior(&self, ctx: &GpuContext, g: &GpuGrid, data: &[f64]) {
        assert_eq!(self.len, g.padded_len(), "Field::upload_interior: field/grid size mismatch");
        assert_eq!(data.len(), g.interior_len(), "Field::upload_interior: data length {} != n³ = {}", data.len(), g.interior_len());
        let n = g.n;
        let mut padded = vec![0.0f32; self.len];
        for k in 0..n { for j in 0..n {
            let src = n * (j + n * k);
            let dst = g.pidx(0, j, k);
            for (d, s) in padded[dst..dst + n].iter_mut().zip(&data[src..src + n]) { *d = *s as f32; }
        }}
        ctx.queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&padded));
    }

    /// Padded f32 → f64 interior in `qm::Grid` layout.
    pub fn read_interior(&self, ctx: &GpuContext, g: &GpuGrid) -> Vec<f64> {
        assert_eq!(self.len, g.padded_len(), "Field::read_interior: field/grid size mismatch");
        let raw = self.read_padded_raw(ctx);
        let n = g.n;
        let mut out = Vec::with_capacity(g.interior_len());
        for k in 0..n { for j in 0..n {
            let s = g.pidx(0, j, k);
            out.extend(raw[s..s + n].iter().map(|v| *v as f64));
        }}
        out
    }

    /// Padded f32 → f32 interior in `qm::Grid` layout (the stored values, unconverted).
    pub fn read_interior_f32(&self, ctx: &GpuContext, g: &GpuGrid) -> Vec<f32> {
        assert_eq!(self.len, g.padded_len(), "Field::read_interior_f32: field/grid size mismatch");
        let raw = self.read_padded_raw(ctx);
        let n = g.n;
        let mut out = Vec::with_capacity(g.interior_len());
        for k in 0..n { for j in 0..n {
            let s = g.pidx(0, j, k);
            out.extend_from_slice(&raw[s..s + n]);
        }}
        out
    }

    /// The whole padded buffer (ghosts included). Test/diagnostic helper.
    pub fn read_padded_raw(&self, ctx: &GpuContext) -> Vec<f32> {
        read_f32(ctx, &self.buffer, 0, self.len)
    }

    /// Whole-buffer GPU copy (ghosts included).
    pub fn copy_from(&self, ctx: &GpuContext, src: &Field) {
        assert_eq!(self.len, src.len, "Field::copy_from: length mismatch");
        assert!(self.buffer != src.buffer, "Field::copy_from: source and destination alias");
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("field copy") });
        enc.copy_buffer_to_buffer(&src.buffer, 0, &self.buffer, 0, Some((self.len * 4) as u64));
        ctx.queue.submit([enc.finish()]);
    }
}

/// Blocking readback of `count` f32s starting at element `offset` (fresh staging buffer).
pub(crate) fn read_f32(ctx: &GpuContext, buffer: &wgpu::Buffer, offset: usize, count: usize) -> Vec<f32> {
    let size = (count * 4) as u64;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("f32 readback"), size, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
    });
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("f32 readback") });
    enc.copy_buffer_to_buffer(buffer, (offset * 4) as u64, &staging, 0, Some(size));
    ctx.queue.submit([enc.finish()]);
    map_read_f32(ctx, &staging, count)
}

thread_local! {
    /// Buffer maps (GPU → CPU readbacks) made by `map_read_f32` on this thread.
    static MAPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Readbacks (`map_read_f32` calls, i.e. buffer maps) made on the calling thread so far. Every
/// readback in this crate goes through `map_read_f32`, and it is synchronous on the calling
/// thread, so the difference of two calls around a function counts exactly the maps that
/// function made, whatever other threads do (the orbital view's "one readback" witness).
pub fn readbacks_on_this_thread() -> usize { MAPS.with(|m| m.get()) }

/// Maps the first `count` f32s of a MAP_READ buffer (after its copy was submitted), copies them out, unmaps.
pub(crate) fn map_read_f32(ctx: &GpuContext, staging: &wgpu::Buffer, count: usize) -> Vec<f32> {
    MAPS.with(|m| m.set(m.get() + 1));
    let slice = staging.slice(..(count * 4) as u64);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    ctx.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    rx.recv().expect("map callback").expect("map ok");
    let out = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range().expect("mapped range")).to_vec();
    staging.unmap();
    out
}

/// Asserts every cell outside the interior [4, 4+n)³ of a raw padded buffer is exactly 0.
#[cfg(test)]
pub(crate) fn assert_ghosts_zero(g: &GpuGrid, raw: &[f32]) {
    assert_eq!(raw.len(), g.padded_len());
    let inner = GHOST..GHOST + g.n;
    for k in 0..g.p { for j in 0..g.p { for i in 0..g.p {
        let inside = inner.contains(&i) && inner.contains(&j) && inner.contains(&k);
        if !inside { assert_eq!(raw[i + g.p * (j + g.p * k)], 0.0, "ghost ({i},{j},{k}) non-zero"); }
    }}}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_unpad_round_trip_and_ghosts_zero() {
        let Some(ctx) = gpu::GpuContext::new_headless() else { return };
        let cg = qm::Grid::cube(9, 0.5);
        let g = GpuGrid::new(&cg);
        let data: Vec<f64> = (0..cg.len()).map(|i| i as f64 * 0.25 - 10.0).collect();
        let f = Field::new(&ctx, &g, "t");
        f.upload_interior(&ctx, &g, &data);
        let back = f.read_interior(&ctx, &g);
        assert_eq!(back.len(), data.len());
        for (a, b) in data.iter().zip(&back) { assert!((a - b).abs() < 1e-5, "{a} vs {b}"); }
        let raw = f.read_padded_raw(&ctx);
        assert_eq!(raw.len(), g.padded_len());
        assert_ghosts_zero(&g, &raw);
        // the interior of the raw buffer sits exactly where pidx says
        for k in 0..g.n { for j in 0..g.n { for i in 0..g.n {
            assert_eq!(raw[g.pidx(i, j, k)], data[cg.idx(i, j, k)] as f32);
        }}}
    }

    #[test]
    fn layout_matches_qm_grid_idx() {
        let cg = qm::Grid::cube(5, 1.0);
        let g = GpuGrid::new(&cg);
        assert_eq!(g.p, 13);
        assert_eq!(g.padded_len(), 13 * 13 * 13);
        assert_eq!(g.interior_len(), 125);
        assert_eq!(g.pidx(0, 0, 0), 4 + 13 * (4 + 13 * 4));
        assert_eq!(g.pidx(1, 2, 3) - g.pidx(0, 2, 3), 1);
        assert_eq!(g.pidx(0, 1, 0) - g.pidx(0, 0, 0), 13);
        assert_eq!(g.pidx(0, 0, 1) - g.pidx(0, 0, 0), 169);
        assert_eq!(g.dv(), cg.dv());
    }

    #[test]
    fn copy_from_copies_whole_padded_buffer() {
        let Some(ctx) = gpu::GpuContext::new_headless() else { return };
        let cg = qm::Grid::cube(7, 0.5);
        let g = GpuGrid::new(&cg);
        let data: Vec<f64> = (0..cg.len()).map(|i| (i as f64).sin()).collect();
        let (a, b) = (Field::new(&ctx, &g, "a"), Field::new(&ctx, &g, "b"));
        a.upload_interior(&ctx, &g, &data);
        b.copy_from(&ctx, &a);
        assert_eq!(a.read_padded_raw(&ctx), b.read_padded_raw(&ctx));
    }
}
