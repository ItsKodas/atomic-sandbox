//! The ionic species on the GPU (M7 Task 8a): the local potential V_loc = Σ V_LR + Σ V_SR, the partial core ρ_pc and
//! the Kleinman–Bylander operator V_NL of a scene under an [`IonicSetup`] — the GPU counterparts of
//! `qm::nuclear::{smeared_potential, ionic_local_potential, partial_core_field}` and `qm::nonlocal::NonlocalOp`
//! (read their module docs for the physics: the exact split V_loc = V_LR + V_SR, the uniform Δ = h/256 tables, the
//! projector convention β_lm = t_l·√((2l+1)/4π)·R_lm, the sphere radii r_β = max(r_c, r_nlc)). Kernels:
//! `shaders/ionic.wgsl` (prefixed by `shaders/erf.wgsl`, a verbatim copy of `nuclear.wgsl`'s erf). The force kernels
//! are Task 8b's.
//!
//! # Design choices (spec §5.3, §5.5 `qm-gpu` bullet; plan Task 8, Spec questions 6 and 7)
//! - **V_LR** is the analytic erf per centre (`smeared_potential`: `nuclear_potential` with a per-centre width — σ_AE
//!   for an all-electron species, σ_LR = 2h for a pseudised one), never tabulated (Spec question 6).
//! - **V_SR and ρ_pc** sample the filtered **uniform tables** (spec §5.1 "Sampling": Δ = h/256, filled exactly on the
//!   CPU in f64 by `radial::filter`, uploaded as f32) with the same 4-point Lagrange as `FilteredTable::{v_sr,
//!   rho_pc_at}`, the index −1 mirrored onto 1. The distance is formed as an exact cell offset minus an f32 fraction,
//!   s = d/Δ = |dc|·256 (exact scaling), so the lookup keeps relative f32 precision in d; there is no logarithm, so the
//!   plan review's f32 ln r₀ caveat is moot. Each grid point owns its value: no races. The tables of every pseudised
//!   species of the setup live in one storage buffer with per-ion offsets.
//! - **V_NL**: the projector values are computed on the CPU in f64 by the reference itself
//!   (`NonlocalOp::build`, the point lists of `points_within` at the derived r_β radius) and uploaded as f32 once per
//!   geometry (spec §5.3: "index lists uploaded per step"). Per ion the list is the ball of its largest projector
//!   radius (the smaller ones are subsets: same centre), and every (projector, m) is a *row* over that list, zero
//!   beyond its own radius — exactly `beta_radial`'s zero, so nothing changes but the summation order. One H·x is
//!   `proj_dot` (every row × ≤ 64 chunks of 256 points, workgroup tree sums), `proj_finish` (per row: E_KB·h³·Σ chunks,
//!   the coefficient finished on the GPU, Spec question 7) and one `proj_axpy` per ion (out += Σ_rows coeff·β, ions
//!   serialised in the batch so overlapping spheres never race). E_NL = Σ f E_KB c² is finished in f64 on the CPU
//!   from the chunk partials (Neumaier, CLAUDE.md rule 4): one readback for every (orbital, row).
//! - **Buffers** are grow-only and reused across geometries (`GpuSolver` keeps one `GpuIonic` and `update`s it per
//!   solve); nothing is allocated per SCF step.
//!
//! # Precision (measured in the tests below)
//! V_loc and ρ_pc are pointwise f32 evaluations of f64-exact tables: the error is the f32 rounding of d, of the table
//! values and of the Lagrange weights, ~1e-7 relative. c_lm is an f32 tree sum over ≤ 2·10⁴ points (~1e-7 relative);
//! E_NL's f64 finish adds nothing. Bounds: `local_sr_matches_cpu`, `nlcc_deposit_matches_cpu`,
//! `proj_apply_and_energy_match_cpu`.
use crate::batch::{Batch, Recorder};
use crate::grid::{map_read_f32, Field, GpuGrid};
use crate::kernels::{workgroups_for, Kernels};
use crate::pointwise::{check, no_alias};
use crate::reduce::neumaier;
use gpu::GpuContext;
use qm::ionic::{Ionic, IonicSetup};
use qm::nonlocal::NonlocalOp;
use qm::nuclear::{centres, Nucleus};
use std::cell::RefCell;
use std::collections::BTreeMap;

/// Most chunks per projector row: `proj_finish` sums them in one 64-thread workgroup.
const MAX_CHUNKS: usize = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct IonParams { p: u32, n: u32, count: u32, pad0: u32, h: f32, scale: f32, pad1: f32, pad2: f32 }

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct CentreRecord { cell: [i32; 4], frac: [f32; 4], inv: [f32; 4] }

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct IonRecord { cell: [i32; 4], frac: [f32; 4], off: [u32; 4], lim: [f32; 4] }

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RowRecord { a: [u32; 4], s: [f32; 4] }

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ProjParams { chunks: u32, n_rows: u32, base: u32, row0: u32, n_pts: u32, pt_off: u32, nr: u32, pad: u32 }

/// A storage buffer that only grows (power-of-two capacity), rewritten in place per geometry.
struct GrowBuf { buf: wgpu::Buffer, cap: u64, usage: wgpu::BufferUsages, label: &'static str, allocations: usize }

impl GrowBuf {
    fn new(ctx: &GpuContext, label: &'static str, usage: wgpu::BufferUsages) -> GrowBuf {
        GrowBuf { buf: make_buffer(ctx, label, usage, 256), cap: 256, usage, label, allocations: 1 }
    }
    fn reserve(&mut self, ctx: &GpuContext, bytes: u64) {
        if bytes > self.cap {
            // 25 % headroom before the power of two: a moved geometry's sphere lists differ by a few points
            let cap = (bytes + bytes / 4).next_power_of_two().max(256);
            self.buf = make_buffer(ctx, self.label, self.usage, cap);
            self.cap = cap;
            self.allocations += 1;
        }
    }
    fn write(&mut self, ctx: &GpuContext, data: &[u8]) {
        self.reserve(ctx, data.len() as u64);
        if !data.is_empty() { ctx.queue.write_buffer(&self.buf, 0, data); }
    }
}

fn make_buffer(ctx: &GpuContext, label: &str, usage: wgpu::BufferUsages, size: u64) -> wgpu::Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
}

const STORAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_DST).union(wgpu::BufferUsages::COPY_SRC);

/// A position as the kernels see it (as `physics::NucleusRecord`): integer cell + f32 fraction of a cell.
fn cell_frac(g: &GpuGrid, pos: [f64; 3]) -> ([i32; 4], [f32; 4]) {
    let (mut cell, mut frac) = ([0i32; 4], [0f32; 4]);
    for a in 0..3 {
        let c = (pos[a] - g.origin[a]) / g.h;
        let fl = c.floor();
        cell[a] = fl as i32;
        frac[a] = (c - fl) as f32;
    }
    (cell, frac)
}

/// The `qm::Grid` of a `GpuGrid` (cubic).
fn qm_grid(g: &GpuGrid) -> qm::Grid { qm::Grid { n: [g.n; 3], h: g.h, origin: g.origin } }

/// The ionic terms of one geometry on the GPU (module doc).
pub struct GpuIonic {
    centres: GrowBuf, n_centres: u32,
    ions: GrowBuf, n_ions: u32,
    tables: GrowBuf,
    /// h/Δ of the uniform tables (256 at this recipe version).
    scale: f32,
    has_pc: bool,
    nl: GpuNonlocal,
    nl_active: bool,
}

impl GpuIonic {
    /// The terms of `nuclei` under `setup` on the grid `g` (per geometry).
    pub fn new(ctx: &GpuContext, g: &GpuGrid, nuclei: &[Nucleus], setup: &IonicSetup) -> GpuIonic {
        let mut s = GpuIonic {
            centres: GrowBuf::new(ctx, "ionic centres", STORAGE), n_centres: 0,
            ions: GrowBuf::new(ctx, "ionic ions", STORAGE), n_ions: 0,
            tables: GrowBuf::new(ctx, "ionic tables", STORAGE),
            scale: 0.0, has_pc: false, nl: GpuNonlocal::new(ctx), nl_active: false,
        };
        s.update(ctx, g, nuclei, setup);
        s
    }

    /// Rewrites every buffer for a new geometry (or setup) in place, growing a buffer only when it is too small.
    pub fn update(&mut self, ctx: &GpuContext, g: &GpuGrid, nuclei: &[Nucleus], setup: &IonicSetup) {
        assert!(setup.h == g.h, "IonicSetup h {} ≠ grid h {}", setup.h, g.h);
        let cs: Vec<CentreRecord> = centres(nuclei, setup).iter().map(|c| {
            let (cell, mut frac) = cell_frac(g, c.pos);
            frac[3] = c.charge as f32;
            CentreRecord { cell, frac, inv: [(1.0 / (c.sigma * std::f64::consts::SQRT_2)) as f32, 0.0, 0.0, 0.0] }
        }).collect();
        self.centres.write(ctx, bytemuck::cast_slice(&cs));
        self.n_centres = cs.len() as u32;
        // the uniform tables of every pseudised species: (V_SR offset, length, ρ_pc offset, length)
        let mut data: Vec<f32> = Vec::new();
        let mut offsets: BTreeMap<u8, [u32; 4]> = BTreeMap::new();
        let mut du: Option<f64> = None;
        for (&z, ionic) in &setup.per_z {
            let Ionic::Pseudo(t) = ionic else { continue };
            assert!(t.h == g.h, "a table filtered at h = {} used at h = {}", t.h, g.h);
            match du { None => du = Some(t.du), Some(d) => assert!(d == t.du, "the setup's uniform tables differ in Δ") }
            let sr = [data.len() as u32, t.v_sr_u.len() as u32];
            data.extend(t.v_sr_u.iter().map(|&v| v as f32));
            let pc = if t.pc_radius > 0.0 { let o = data.len() as u32; data.extend(t.rho_pc_u.iter().map(|&v| v as f32)); [o, t.rho_pc_u.len() as u32] } else { [0, 0] };
            offsets.insert(z, [sr[0], sr[1], pc[0], pc[1]]);
        }
        self.tables.write(ctx, bytemuck::cast_slice(&data));
        let du = du.unwrap_or(g.h / 256.0);
        self.scale = (g.h / du) as f32;
        let ions: Vec<IonRecord> = nuclei.iter().filter_map(|n| {
            let Ionic::Pseudo(t) = setup.ionic(n.z) else { return None };
            let (cell, frac) = cell_frac(g, n.pos);
            Some(IonRecord { cell, frac, off: offsets[&n.z], lim: [(t.r_sr / du) as f32, (t.pc_radius.max(0.0) / du) as f32, 0.0, 0.0] })
        }).collect();
        self.has_pc = ions.iter().any(|i| i.off[3] > 0);
        self.ions.write(ctx, bytemuck::cast_slice(&ions));
        self.n_ions = ions.len() as u32;
        let op = NonlocalOp::build(&qm_grid(g), nuclei, setup);
        self.nl_active = !op.is_empty();
        if self.nl_active { self.nl.update(ctx, g, &op); }
    }

    /// out = V_loc = Σ_centres V_LR + Σ_pseudised V_SR (recorded; `qm::nuclear::ionic_local_potential`).
    pub fn local_potential_in(&self, b: &mut Batch, k: &Kernels, g: &GpuGrid, out: &Field) {
        check(g, &[out]);
        let ctx = b.ctx();
        let params = |count: u32| IonParams { p: g.p as u32, n: g.n as u32, count, pad0: 0, h: g.h as f32, scale: self.scale, pad1: 0.0, pad2: 0.0 };
        let u = b.uniform(ctx, k, "smeared_potential", &params(self.n_centres));
        let bg = k.bind_group(ctx, "smeared_potential", &[u.binding(), out.buffer.as_entire_binding(), self.centres.buf.as_entire_binding()]);
        b.dispatch(k, "smeared_potential", &bg, [workgroups_for(g.interior_len()), 1, 1]);
        if self.n_ions > 0 {
            let u = b.uniform(ctx, k, "local_sr", &params(self.n_ions));
            let bg = k.bind_group(ctx, "local_sr", &[u.binding(), out.buffer.as_entire_binding(), self.ions.buf.as_entire_binding(), self.tables.buf.as_entire_binding()]);
            b.dispatch(k, "local_sr", &bg, [workgroups_for(g.interior_len()), 1, 1]);
        }
    }

    /// out = ρ_pc = Σ_ions ρ_pc,I (recorded; `qm::nuclear::partial_core_field`); 0 without a partial core.
    pub fn partial_core_in(&self, b: &mut Batch, k: &Kernels, g: &GpuGrid, out: &Field) {
        check(g, &[out]);
        let ctx = b.ctx();
        let params = IonParams { p: g.p as u32, n: g.n as u32, count: self.n_ions, pad0: 0, h: g.h as f32, scale: self.scale, pad1: 0.0, pad2: 0.0 };
        let u = b.uniform(ctx, k, "nlcc_deposit", &params);
        let bg = k.bind_group(ctx, "nlcc_deposit", &[u.binding(), out.buffer.as_entire_binding(), self.ions.buf.as_entire_binding(), self.tables.buf.as_entire_binding()]);
        b.dispatch(k, "nlcc_deposit", &bg, [workgroups_for(g.interior_len()), 1, 1]);
    }

    /// Whether any ion carries a partial core (`qm::nuclear::has_partial_core`).
    pub fn has_partial_core(&self) -> bool { self.has_pc }

    /// V_NL, if any pseudised ion has a projector.
    pub fn nonlocal(&self) -> Option<&GpuNonlocal> { self.nl_active.then_some(&self.nl) }

    /// Buffer (re)allocations so far (a reuse witness: constant across geometries of one scene).
    pub fn buffer_allocations(&self) -> usize {
        self.centres.allocations + self.ions.allocations + self.tables.allocations + self.nl.allocations()
    }
}

/// One ion's point list and rows (module doc).
#[derive(Clone, Copy, Debug)]
struct Span { pt_off: u32, n_pts: u32, row0: u32, nr: u32 }

/// The chunk partials and staging buffer of `GpuNonlocal::energy_terms`.
struct EnergyScratch { partials: GrowBuf, staging: GrowBuf }

/// V_NL on the GPU for one geometry (module doc): point lists, β rows, the chunk partials and the coefficients.
pub struct GpuNonlocal {
    points: GrowBuf, beta: GrowBuf, rows: GrowBuf, partials: GrowBuf, coeffs: GrowBuf,
    chunks: u32, n_rows: u32,
    spans: Vec<Span>,
    /// E_KB per row (Ha) and h³.
    e_kb: Vec<f64>, dv: f64,
    energy: RefCell<EnergyScratch>,
}

impl GpuNonlocal {
    fn new(ctx: &GpuContext) -> GpuNonlocal {
        GpuNonlocal {
            points: GrowBuf::new(ctx, "nl points", STORAGE), beta: GrowBuf::new(ctx, "nl beta", STORAGE), rows: GrowBuf::new(ctx, "nl rows", STORAGE),
            partials: GrowBuf::new(ctx, "nl partials", STORAGE), coeffs: GrowBuf::new(ctx, "nl coeffs", STORAGE),
            chunks: 1, n_rows: 0, spans: Vec::new(), e_kb: Vec::new(), dv: 0.0,
            energy: RefCell::new(EnergyScratch {
                partials: GrowBuf::new(ctx, "nl energy partials", STORAGE),
                staging: GrowBuf::new(ctx, "nl energy staging", wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
            }),
        }
    }

    fn allocations(&self) -> usize {
        let e = self.energy.borrow();
        self.points.allocations + self.beta.allocations + self.rows.allocations + self.partials.allocations + self.coeffs.allocations
            + e.partials.allocations + e.staging.allocations
    }

    /// Uploads `op`'s spheres, grouped per ion onto the ion's largest ball (module doc).
    fn update(&mut self, ctx: &GpuContext, g: &GpuGrid, op: &NonlocalOp) {
        let mut by_ion: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, s) in op.spheres.iter().enumerate() { by_ion.entry(s.ion).or_default().push(i); }
        let n = g.n;
        let (mut points, mut beta, mut rows, mut e_kb, mut spans) = (Vec::<u32>::new(), Vec::<f32>::new(), Vec::<RowRecord>::new(), Vec::new(), Vec::new());
        for (&ion, idx) in &by_ion {
            let union = &op.spheres[*idx.iter().max_by_key(|&&i| op.spheres[i].points.len()).expect("an ion with spheres")].points;
            let (pt_off, n_pts, row0) = (points.len() as u32, union.len() as u32, rows.len() as u32);
            points.extend(union.iter().map(|&p| g.pidx(p % n, (p / n) % n, p / (n * n)) as u32));
            for &si in idx {
                let s = &op.spheres[si];
                // s.points ⊆ union, both ascending: the column of each of s's points in the union list
                let mut col = Vec::with_capacity(s.points.len());
                let mut j = 0;
                for &p in &s.points {
                    while j < union.len() && union[j] != p { j += 1; }
                    assert!(j < union.len(), "a projector ball of ion {ion} is not inside its largest ball");
                    col.push(j);
                }
                for bm in &s.beta {
                    let off = beta.len();
                    beta.resize(off + union.len(), 0.0);
                    for (&c, &v) in col.iter().zip(bm) { beta[off + c] = v as f32; }
                    rows.push(RowRecord { a: [pt_off, n_pts, off as u32, ion as u32], s: [(s.e_kb * op.dv) as f32, 0.0, 0.0, 0.0] });
                    e_kb.push(s.e_kb);
                }
            }
            spans.push(Span { pt_off, n_pts, row0, nr: rows.len() as u32 - row0 });
        }
        let max_pts = spans.iter().map(|s| s.n_pts as usize).max().unwrap_or(0);
        self.chunks = max_pts.div_ceil(256).clamp(1, MAX_CHUNKS) as u32;
        self.n_rows = rows.len() as u32;
        self.points.write(ctx, bytemuck::cast_slice(&points));
        self.beta.write(ctx, bytemuck::cast_slice(&beta));
        self.rows.write(ctx, bytemuck::cast_slice(&rows));
        self.partials.reserve(ctx, 4 * (self.n_rows * self.chunks) as u64);
        self.coeffs.reserve(ctx, 4 * self.n_rows as u64);
        self.spans = spans;
        self.e_kb = e_kb;
        self.dv = op.dv;
    }

    /// Rows (projector × m) over all ions, and the chunks per row.
    pub fn rows(&self) -> (usize, usize) { (self.n_rows as usize, self.chunks as usize) }

    fn params(&self) -> ProjParams { ProjParams { chunks: self.chunks, n_rows: self.n_rows, ..ProjParams::default() } }

    /// Records `proj_dot` of `x` into `partials` at `base`.
    fn dot_rec(&self, b: &mut Batch, k: &Kernels, x: &Field, partials: &wgpu::Buffer, base: u32) {
        let ctx = b.ctx();
        let u = b.uniform(ctx, k, "proj_dot", &ProjParams { base, ..self.params() });
        let bg = k.bind_group(ctx, "proj_dot", &[u.binding(), self.points.buf.as_entire_binding(), self.beta.buf.as_entire_binding(),
                                                 x.buffer.as_entire_binding(), partials.as_entire_binding(), self.rows.buf.as_entire_binding()]);
        b.dispatch(k, "proj_dot", &bg, [self.chunks, self.n_rows, 1]);
    }

    /// out += V_NL x = Σ E_KB c_lm[x] β_lm (recorded: `proj_dot`, `proj_finish`, one `proj_axpy` per ion).
    pub fn apply_in(&self, b: &mut Batch, k: &Kernels, g: &GpuGrid, x: &Field, out: &Field) {
        check(g, &[x, out]);
        no_alias("proj_axpy", out, x);
        let ctx = b.ctx();
        self.dot_rec(b, k, x, &self.partials.buf, 0);
        let u = b.uniform(ctx, k, "proj_finish", &self.params());
        let bg = k.bind_group(ctx, "proj_finish", &[u.binding(), self.partials.buf.as_entire_binding(), self.rows.buf.as_entire_binding(), self.coeffs.buf.as_entire_binding()]);
        b.dispatch(k, "proj_finish", &bg, [self.n_rows, 1, 1]);
        for sp in &self.spans {
            let u = b.uniform(ctx, k, "proj_axpy", &ProjParams { row0: sp.row0, n_pts: sp.n_pts, pt_off: sp.pt_off, nr: sp.nr, ..self.params() });
            let bg = k.bind_group(ctx, "proj_axpy", &[u.binding(), self.points.buf.as_entire_binding(), self.beta.buf.as_entire_binding(),
                                                      self.rows.buf.as_entire_binding(), self.coeffs.buf.as_entire_binding(), out.buffer.as_entire_binding()]);
            b.dispatch(k, "proj_axpy", &bg, [workgroups_for(sp.n_pts as usize), 1, 1]);
        }
    }

    /// The chunk partials of c_lm/h³ for every vector of `xs`: [vector][row][chunk], one readback (submits and waits).
    fn partials_of(&self, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, xs: &[&Field]) -> Vec<f32> {
        for x in xs { check(g, &[x]); }
        let per = (self.n_rows * self.chunks) as usize;
        let total = per * xs.len();
        if total == 0 { return Vec::new(); }
        let mut e = self.energy.borrow_mut();
        e.partials.reserve(ctx, 4 * total as u64);
        e.staging.reserve(ctx, 4 * total as u64);
        let mut b = Batch::new(ctx, "nonlocal partials");
        for (v, x) in xs.iter().enumerate() { self.dot_rec(&mut b, k, x, &e.partials.buf, (v * per) as u32); }
        b.encoder().copy_buffer_to_buffer(&e.partials.buf, 0, &e.staging.buf, 0, Some(4 * total as u64));
        b.submit();
        map_read_f32(ctx, &e.staging.buf, total)
    }

    /// c[vector][row] = h³ Σ_chunks partials, finished in f64 (Neumaier) — `NonlocalOp::project` with the rows of all
    /// spheres flattened in sphere order, m ascending.
    pub fn project(&self, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, xs: &[&Field]) -> Vec<Vec<f64>> {
        let p = self.partials_of(ctx, k, g, xs);
        let ch = self.chunks as usize;
        let per = self.n_rows as usize * ch;
        (0..xs.len()).map(|v| (0..self.n_rows as usize).map(|r| {
            let s = &p[v * per + r * ch..v * per + (r + 1) * ch];
            neumaier(s.iter().map(|&x| x as f64)) * self.dv
        }).collect()).collect()
    }

    /// E_NL = Σ_i f_i Σ_rows E_KB c²[ψ_i] over `terms` = (ψ_i, f_i), f = 0 skipped, finished in f64 (Neumaier;
    /// `NonlocalOp::energy_weighted`). One readback.
    pub fn energy_terms(&self, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, terms: &[(&Field, f64)]) -> f64 {
        let live: Vec<(&Field, f64)> = terms.iter().copied().filter(|t| t.1 != 0.0).collect();
        let xs: Vec<&Field> = live.iter().map(|t| t.0).collect();
        let c = self.project(ctx, k, g, &xs);
        neumaier(live.iter().zip(&c).flat_map(|((_, f), cv)| cv.iter().zip(&self.e_kb).map(move |(c, e)| f * e * c * c)))
    }

    /// E_NL of singly occupied orbitals (`NonlocalOp::energy`).
    pub fn energy(&self, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, orbitals: &[Field]) -> f64 {
        let terms: Vec<(&Field, f64)> = orbitals.iter().map(|o| (o, 1.0)).collect();
        self.energy_terms(ctx, k, g, &terms)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use qm::ionic::{Ionic, IonicPolicy, IonicSetup};
    use qm::radial::filter::{filtered, FilteredTable};
    use qm::radial::{generate, PseudoTable};
    use std::collections::{BTreeMap, HashMap};
    use std::sync::{Arc, Mutex, OnceLock};

    /// The recipe's table of Z, generated once per test binary.
    pub(crate) fn table(z: u8) -> Arc<PseudoTable> {
        static T: OnceLock<Mutex<HashMap<u8, Arc<PseudoTable>>>> = OnceLock::new();
        if let Some(t) = T.get_or_init(Default::default).lock().unwrap().get(&z) { return t.clone(); }
        let t = Arc::new(generate(z).unwrap_or_else(|e| panic!("Z = {z}: {e}")));
        T.get_or_init(Default::default).lock().unwrap().entry(z).or_insert(t).clone()
    }

    /// Z's table filtered at h, once per binary.
    pub(crate) fn filtered_at(z: u8, h: f64) -> Arc<FilteredTable> {
        static T: OnceLock<Mutex<HashMap<(u8, u64), Arc<FilteredTable>>>> = OnceLock::new();
        if let Some(t) = T.get_or_init(Default::default).lock().unwrap().get(&(z, h.to_bits())) { return t.clone(); }
        let t = Arc::new(filtered(&table(z), h));
        T.get_or_init(Default::default).lock().unwrap().entry((z, h.to_bits())).or_insert(t).clone()
    }

    /// Pseudo-O beside an all-electron H (a mixed `CoresOnly` scene), uncertified at h (`from_tables`): a kernel test.
    fn o_setup(h: f64) -> Arc<IonicSetup> {
        Arc::new(IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, BTreeMap::from([(8, Ionic::Pseudo(filtered_at(8, h))), (1, Ionic::AllElectron { sigma: h })])))
    }
    fn nuclei() -> Vec<Nucleus> { vec![Nucleus { pos: [0.11, -0.07, 0.05], z: 8 }, Nucleus { pos: [1.92, 0.31, -0.2], z: 1 }] }

    /// The 33³ grid at h = 0.25 plus a second, non-power-of-two-friendly 30³ grid at h = 0.2 whose −x face the O
    /// spheres cross (O 0.4 Bohr inside it): the boundary-clipped case.
    fn cases() -> Vec<(qm::Grid, Vec<Nucleus>)> {
        let small = qm::Grid::cube(30, 0.2);
        let edge = small.origin[0] + 0.4;
        vec![(qm::Grid::cube(33, 0.25), nuclei()),
             (small, vec![Nucleus { pos: [edge, 0.13, -0.05], z: 8 }, Nucleus { pos: [edge + 1.8, 0.2, 0.1], z: 1 }])]
    }

    /// The local potential (V_LR per centre + V_SR from the uniform table) vs `qm::nuclear::ionic_local_potential`.
    /// Bound: 2e-6 of max(|V|, 1) pointwise (the brief's); measured in the report.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm-gpu --release")]
    fn local_sr_matches_cpu() {
        let Some(ctx) = gpu::GpuContext::new_headless() else { return };
        let k = Kernels::new(&ctx);
        for (grid, nuc) in cases() {
            let g = GpuGrid::new(&grid);
            let s = o_setup(grid.h);
            let cpu = qm::nuclear::ionic_local_potential(&grid, &nuc, &s);
            let ion = GpuIonic::new(&ctx, &g, &nuc, &s);
            let out = Field::new(&ctx, &g, "v_loc");
            let mut b = Batch::new(&ctx, "t");
            ion.local_potential_in(&mut b, &k, &g, &out);
            b.submit();
            crate::grid::assert_ghosts_zero(&g, &out.read_padded_raw(&ctx));
            let got = out.read_interior(&ctx, &g);
            let worst = cpu.iter().zip(&got).map(|(a, b)| (a - b).abs() / a.abs().max(1.0)).fold(0.0, f64::max);
            let vmax = cpu.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            println!("V_loc {}³ h = {}: max |V| {vmax:.4}, max relative |ΔV| {worst:.2e}", grid.n[0], grid.h);
            assert!(worst < 2e-6, "max relative |ΔV| = {worst:.2e}");
        }
    }

    /// ρ_pc deposit vs `qm::nuclear::partial_core_field`: 1e-6 of max ρ_pc (the brief's bound).
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm-gpu --release")]
    fn nlcc_deposit_matches_cpu() {
        let Some(ctx) = gpu::GpuContext::new_headless() else { return };
        let k = Kernels::new(&ctx);
        for (grid, nuc) in cases() {
            let g = GpuGrid::new(&grid);
            let s = o_setup(grid.h);
            let cpu = qm::nuclear::partial_core_field(&grid, &nuc, &s);
            let ion = GpuIonic::new(&ctx, &g, &nuc, &s);
            assert!(ion.has_partial_core());
            let out = Field::new(&ctx, &g, "rho_pc");
            let mut b = Batch::new(&ctx, "t");
            ion.partial_core_in(&mut b, &k, &g, &out);
            b.submit();
            crate::grid::assert_ghosts_zero(&g, &out.read_padded_raw(&ctx));
            let max = cpu.iter().cloned().fold(0.0, f64::max);
            let worst = cpu.iter().zip(out.read_interior(&ctx, &g)).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            println!("ρ_pc {}³ h = {}: max ρ_pc {max:.4e}, max |Δρ_pc| {worst:.2e} ({:.2e} of max)", grid.n[0], grid.h, worst / max);
            assert!(worst < 1e-6 * max, "max |Δρ_pc| = {worst:.2e} (max ρ_pc {max:.2e})");
        }
    }

    /// V_NL x on random x vs `qm::nonlocal::NonlocalOp::apply_add` (1e-5 of max |V_NL x|, the brief's), and E_NL of a
    /// set of vectors (1e-6 relative), on both grids (the second clips the O spheres at the −x face).
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm-gpu --release")]
    fn proj_apply_and_energy_match_cpu() {
        let Some(ctx) = gpu::GpuContext::new_headless() else { return };
        let k = Kernels::new(&ctx);
        for (grid, nuc) in cases() {
            let g = GpuGrid::new(&grid);
            let s = o_setup(grid.h);
            let cpu_nl = qm::nonlocal::NonlocalOp::build(&grid, &nuc, &s);
            let mut seed = 7u64;
            let mut rnd = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5 };
            let xs: Vec<Vec<f64>> = (0..3).map(|_| (0..grid.len()).map(|_| rnd()).collect()).collect();
            let mut cpu = vec![0.0; grid.len()];
            cpu_nl.apply_add(&xs[0], &mut cpu);
            let ion = GpuIonic::new(&ctx, &g, &nuc, &s);
            let nl = ion.nonlocal().expect("O has a projector");
            let fields: Vec<Field> = xs.iter().map(|x| { let f = Field::new(&ctx, &g, "x"); f.upload_interior(&ctx, &g, x); f }).collect();
            let out = Field::new(&ctx, &g, "out");
            let mut b = Batch::new(&ctx, "t");
            nl.apply_in(&mut b, &k, &g, &fields[0], &out);
            b.submit();
            crate::grid::assert_ghosts_zero(&g, &out.read_padded_raw(&ctx));
            let got = out.read_interior(&ctx, &g);
            let scale = cpu.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let worst = cpu.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            // the f32 sums' own scale: V_NL applied to |x| (no cancellation in c), for the clipped case below
            let abs_x: Vec<f64> = xs[0].iter().map(|v| v.abs()).collect();
            let mut cpu_abs = vec![0.0; grid.len()];
            cpu_nl.apply_add(&abs_x, &mut cpu_abs);
            let scale_abs = cpu_abs.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            // the f32 input itself: the CPU on the f32-rounded vectors
            let x32: Vec<Vec<f64>> = xs.iter().map(|x| x.iter().map(|v| *v as f32 as f64).collect()).collect();
            let e_gpu = nl.energy(&ctx, &k, &g, &fields);
            let e_cpu = cpu_nl.energy(&x32);
            let (rows, chunks) = nl.rows();
            println!("V_NL {}³ h = {}: {} sphere(s), {rows} rows × {chunks} chunks; max |ΔV_NL x| {worst:.2e} of {scale:.2e} ({:.2e}; of max |V_NL |x|| {scale_abs:.2e}: {:.2e}); E_NL gpu {e_gpu:.12} cpu {e_cpu:.12} (relative {:.2e})",
                     grid.n[0], grid.h, cpu_nl.spheres.len(), worst / scale, worst / scale_abs, (e_gpu - e_cpu) / e_cpu);
            // the brief's bound on its own case (33³, unclipped); every case against the sums' scale. The clipped case's c
            // is a near-cancellation (measured: max |V_NL x| 7e-5 of max |V_NL |x||), so its max |V_NL x| measures the random x, not the kernel
            if grid.n[0] == 33 { assert!(worst < 1e-5 * scale, "max |ΔV_NL x| = {worst:.2e} of {scale:.2e}"); }
            assert!(worst < 1e-5 * scale_abs, "max |ΔV_NL x| = {worst:.2e} of max |V_NL |x|| {scale_abs:.2e}");
            assert!((e_gpu - e_cpu).abs() < 1e-6 * e_cpu.abs(), "E_NL {e_gpu} vs {e_cpu}");
            // the coefficients themselves, against NonlocalOp::project (rows in sphere order, m ascending)
            let c_gpu = nl.project(&ctx, &k, &g, &[&fields[1]]);
            let c_cpu: Vec<f64> = cpu_nl.project(&x32[1]).into_iter().flatten().collect();
            // the scale of the f32 sum: h³ Σ|β||x| per row
            let x1 = &x32[1];
            let cabs: f64 = cpu_nl.spheres.iter().flat_map(|sp| sp.beta.iter().map(move |b| sp.points.iter().zip(b).map(|(&p, bv)| (bv * x1[p]).abs()).sum::<f64>()))
                .fold(0.0, f64::max) * grid.dv();
            let cmax = c_cpu.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let cworst = c_cpu.iter().zip(&c_gpu[0]).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            println!("  c_lm: max |Δc| {cworst:.2e} of |c| {cmax:.2e}, of h³Σ|βx| {cabs:.2e} ({:.2e})", cworst / cabs);
            assert!(cworst < 1e-6 * cabs, "max |Δc| {cworst:.2e} of h³Σ|βx| {cabs:.2e}");
        }
    }

    /// Timing for the report (spec §5.3, "the GPU share is Task 8's measurement"): one H·x, local (`hamiltonian`) and
    /// with V_NL, for pseudo-O at h = 0.20 on 61³ (one s projector, the cost report's case) and for O₂ on 71³; 400
    /// applications per batch, best of 3. `cargo test -p qm-gpu --release -- --ignored --nocapture bench_nonlocal_share`.
    #[test]
    #[ignore]
    fn bench_nonlocal_share() {
        let Some(ctx) = gpu::GpuContext::new_headless() else { return };
        let k = Kernels::new(&ctx);
        let h = 0.20;
        let s = Arc::new(IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, BTreeMap::from([(8, Ionic::Pseudo(filtered_at(8, h)))])));
        for (name, n, nuc) in [("O", 61usize, vec![Nucleus { pos: [0.07, -0.03, 0.02], z: 8 }]),
                               ("O₂", 71, vec![Nucleus { pos: [-1.14, 0.07, 0.03], z: 8 }, Nucleus { pos: [1.14, 0.07, 0.03], z: 8 }])] {
            let grid = qm::Grid::cube(n, h);
            let g = GpuGrid::new(&grid);
            let ion = GpuIonic::new(&ctx, &g, &nuc, &s);
            let (v, x, out) = (Field::new(&ctx, &g, "v"), Field::new(&ctx, &g, "x"), Field::new(&ctx, &g, "out"));
            x.upload_interior(&ctx, &g, &grid.sample(|r| (-(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) / 4.0).exp()));
            let time = |nl: Option<&GpuNonlocal>| -> f64 {
                let h = crate::eigen::GpuHamiltonian { v: &v, nl };
                (0..3).map(|_| {
                    let t = std::time::Instant::now();
                    let mut b = Batch::new(&ctx, "bench");
                    for _ in 0..400 { h.apply_in(&mut b, &k, &g, &x, &out); }
                    b.submit();
                    ctx.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
                    t.elapsed().as_secs_f64() * 1e3 / 400.0
                }).fold(f64::INFINITY, f64::min)
            };
            let (local, pseudo) = (time(None), time(ion.nonlocal()));
            let (rows, chunks) = ion.nonlocal().unwrap().rows();
            println!("{name} {n}³ h = {h}: H·x local {local:.3} ms, with V_NL {pseudo:.3} ms → nonlocal share {:.1} % ({rows} rows × {chunks} chunks)", 100.0 * (pseudo - local) / pseudo);
        }
    }

    /// Moving the geometry reuses every buffer (no allocation per geometry once the scene's sizes are reached), and the
    /// updated terms equal a freshly built `GpuIonic`'s bit for bit.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm-gpu --release")]
    fn update_reuses_buffers() {
        let Some(ctx) = gpu::GpuContext::new_headless() else { return };
        let k = Kernels::new(&ctx);
        let grid = qm::Grid::cube(33, 0.25);
        let g = GpuGrid::new(&grid);
        let s = o_setup(0.25);
        let mut ion = GpuIonic::new(&ctx, &g, &nuclei(), &s);
        let x = Field::new(&ctx, &g, "x");
        x.upload_interior(&ctx, &g, &grid.sample(|r| (-(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) / 3.0).exp()));
        let _ = ion.nonlocal().unwrap().energy(&ctx, &k, &g, std::slice::from_ref(&x)); // sizes the energy scratch
        let before = ion.buffer_allocations();
        let moved: Vec<Nucleus> = nuclei().iter().map(|n| Nucleus { pos: [n.pos[0] + 0.03, n.pos[1] - 0.02, n.pos[2]], ..*n }).collect();
        ion.update(&ctx, &g, &moved, &s);
        ion.update(&ctx, &g, &nuclei(), &s);
        ion.update(&ctx, &g, &moved, &s);
        let fresh = GpuIonic::new(&ctx, &g, &moved, &s);
        let (a, b) = (Field::new(&ctx, &g, "a"), Field::new(&ctx, &g, "b"));
        let mut bt = Batch::new(&ctx, "t");
        ion.local_potential_in(&mut bt, &k, &g, &a);
        fresh.local_potential_in(&mut bt, &k, &g, &b);
        bt.submit();
        assert_eq!(a.read_padded_raw(&ctx), b.read_padded_raw(&ctx));
        assert_eq!(ion.nonlocal().unwrap().energy(&ctx, &k, &g, std::slice::from_ref(&x)).to_bits(),
                   fresh.nonlocal().unwrap().energy(&ctx, &k, &g, std::slice::from_ref(&x)).to_bits());
        assert_eq!(ion.buffer_allocations(), before, "a moved geometry of the same scene allocated");
    }
}
