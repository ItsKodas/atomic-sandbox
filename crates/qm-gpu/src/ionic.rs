//! The ionic species on the GPU (M7 Task 8a): the local potential V_loc = Σ V_LR + Σ V_SR, the partial core ρ_pc and
//! the Kleinman–Bylander operator V_NL of a scene under an [`IonicSetup`] — the GPU counterparts of
//! `qm::nuclear::{smeared_potential, ionic_local_potential, partial_core_field}` and `qm::nonlocal::NonlocalOp`
//! (read their module docs for the physics: the exact split V_loc = V_LR + V_SR, the uniform Δ = h/256 tables, the
//! projector convention β_lm = t_l·√((2l+1)/4π)·R_lm, the sphere radii r_β = max(r_c, r_nlc)). Kernels:
//! `shaders/ionic.wgsl` (prefixed by `shaders/erf.wgsl`, a verbatim copy of `nuclear.wgsl`'s erf). The force terms
//! (Task 8b, section "Forces" below; `shaders/ionic_forces.wgsl`) are the GPU form of `qm::forces`' pseudised terms.
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
//! # Forces (M7 Task 8b; `pseudo_electronic_forces`, `GpuSolver::force_terms`)
//! The physics is `qm::forces`' module doc, "Pseudised ions (M7)": F_I = −∂E_h/∂R_I at frozen ψ and f, five terms, each
//! the R-derivative of one frozen energy. On the GPU:
//! - **Local LR** (every centre, pseudised (Z_v, σ_LR) or all-electron (Z, σ_AE)): `centre_force`, the integrand of
//!   `forces.wgsl`'s `force_partials` with the centre's own q and σ over the whole interior, one dispatch per centre.
//!   Its G(x) is a verbatim copy, so the all-electron `force_partials` (the `None` path) is untouched.
//! - **Local SR, partial core, nonlocal**: `gather_dot4` over point lists with per-point vec4 weights, formed in f64 on
//!   the CPU per geometry by the reference's own functions and uploaded as f32 (spec §5.3, "index lists uploaded per
//!   step"; the brief's design): (V_SR′(d)/d)(r − R) over `points_within(sr_radius)` (x = ρ), (ρ_pc′(d)/d)(r − R) over
//!   `points_within(pc_radius)` (x = v̄), and (∇β_lm, β_lm) per projector sphere and m from
//!   `NonlocalOp::build_with_gradients` (x = ψ_iσ, one dispatch per orbital with f ≠ 0: .xyz = ⟨∇β|ψ⟩/h³, .w = c/h³).
//!   Points with d < 1e-8 get weight 0 (the CPU's on-site skip). The CPU finishes F^NL = Σ 2f E_KB c⟨∇β|ψ⟩ in f64.
//! - **v̄** = ½(v_xc↑ + v_xc↓) on (ρ↑ + ½ρ_pc, ρ↓ + ½ρ_pc) is recomputed from the state's output spin densities with
//!   the SCF's own `scf::xc_in` (ρ_pc deposited by `nlcc_deposit`), i.e. bit for bit the v̄ of the solver's final XC
//!   evaluation, instead of keeping one more field in every `GpuState` (the brief's `v_avg` input). Six scratch fields
//!   are allocated on the first cored force call and reused.
//! - **Ion–ion** is f64 on the CPU (`qm::forces::ion_ion_forces`, coincident nuclei an error), in `GpuSolver::force_terms`.
//! - **Compensated sums.** The terms are odd integrands that cancel to a small net (a lone O's ρ_pc term: Σ_p|t_p|/|F|
//!   ≈ 2e6, `qm::forces` `partial_core_force_is_the_radial_lattice_sum`). A plain f32 tree would leave ~u·Σ|t|,
//!   comparable to such a term, so both kernels accumulate in double-f32 (TwoSum with `keep` barriers against the
//!   driver's reassociation, `shaders/ionic_forces.wgsl`) and hand (hi, lo) per workgroup to the f64 Neumaier finish.
//!   What remains is per-point rounding (products, f32 weights, v̄ and G(x) in f32, the f32 fraction of R), measured
//!   in `pseudo_force_terms_match_cpu_on_frozen_states`.
//! - **Cost.** One submission and one readback per force call; the weights are rebuilt on the CPU once per geometry
//!   (`GpuIonic::update` invalidates them), into grow-only buffers.
//!   β is built twice per geometry, by `update`'s `NonlocalOp::build` and by the force weights'
//!   `build_with_gradients` — part of the 7–14 ms per MD step; building the gradients once in `update` when forces
//!   follow is deferred until pseudo MD rates matter (Task 10; Task 8b review Minor-4). Timings in references.md "GPU vs CPU:
//!   pseudopotentials — forces (M7 Task 8b)".
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
use qm::nuclear::{centres, points_within, Nucleus};
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
    /// The geometry and the setup these terms were built for (`update`): the forces must be asked for the same nuclei
    /// under a setup with the same physics (`IonicSetup::same_physics`).
    nuclei: Vec<Nucleus>,
    setup: Option<IonicSetup>,
    /// The force weights of this geometry and their buffers (M7 Task 8b; built on the first force call after an
    /// `update`, grow-only).
    forces: RefCell<ForceBufs>,
}

impl GpuIonic {
    /// The terms of `nuclei` under `setup` on the grid `g` (per geometry).
    pub fn new(ctx: &GpuContext, g: &GpuGrid, nuclei: &[Nucleus], setup: &IonicSetup) -> GpuIonic {
        let mut s = GpuIonic {
            centres: GrowBuf::new(ctx, "ionic centres", STORAGE), n_centres: 0,
            ions: GrowBuf::new(ctx, "ionic ions", STORAGE), n_ions: 0,
            tables: GrowBuf::new(ctx, "ionic tables", STORAGE),
            scale: 0.0, has_pc: false, nl: GpuNonlocal::new(ctx), nl_active: false,
            nuclei: Vec::new(), setup: None, forces: RefCell::new(ForceBufs::new(ctx)),
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
        self.nuclei.clear();
        self.nuclei.extend_from_slice(nuclei);
        if !self.setup.as_ref().is_some_and(|s| s.same_physics(setup)) { self.setup = Some(setup.clone()); }
        self.forces.get_mut().valid = false; // the force weights are rebuilt for this geometry on demand
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
        self.centres.allocations + self.ions.allocations + self.tables.allocations + self.nl.allocations() + self.forces.borrow().allocations()
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

// ---- Forces (M7 Task 8b; module doc "Forces") ----

/// Below this distance (Bohr) a grid point is on the nucleus and its radial weight is 0: `qm::forces::ON_SITE` itself.
const FORCE_ON_SITE: f64 = qm::forces::ON_SITE;

/// Uniform of `centre_force` (shaders/ionic_forces.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct CentreForceParams { cell: [i32; 4], frac: [f32; 4], p: u32, n: u32, base: u32, zero: u32, h_over_a: f32, c: f32, pad: [f32; 2] }

/// Uniform of `gather_dot4` (shaders/ionic_forces.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GatherParams { chunks: u32, row0: u32, base: u32, zero: u32 }

/// Workgroups (= compensated partial pairs) per `centre_force` dispatch: the `Reducer`'s count, as `forces.rs`.
fn centre_workgroups(g: &GpuGrid) -> u32 {
    (g.interior_len().div_ceil(crate::kernels::WORKGROUP as usize) as u32).clamp(1, crate::reduce::MAX_PARTIALS)
}

/// The force weights of one geometry (module doc "Forces"), their buffers and the v̄ scratch fields. Every buffer is
/// grow-only and rewritten in place when the geometry changes; `valid` is cleared by `GpuIonic::update`.
struct ForceBufs {
    points: GrowBuf, w4: GrowBuf, rows: GrowBuf, partials: GrowBuf, staging: GrowBuf,
    /// Row ranges [start, end) in `rows`: the local SR rows (x = ρ), the partial-core rows (x = v̄), the nonlocal rows
    /// (x = ψ_i; one per sphere and m).
    sr: (u32, u32), pc: (u32, u32), nl: (u32, u32),
    /// The nucleus of each row, and E_KB (Ha) of each nonlocal row.
    row_ion: Vec<usize>, nl_e_kb: Vec<f64>,
    /// Chunks (workgroups) per row: min(⌈longest list/256⌉, 64).
    chunks: u32,
    valid: bool,
    /// v̄ scratch: ρ_pc, n↑, n↓ (then v̄), e_xc, v↑, v↓ — allocated on the first force call with a partial core.
    fields: Option<[Field; 6]>,
    field_allocations: usize,
}

impl ForceBufs {
    fn new(ctx: &GpuContext) -> ForceBufs {
        ForceBufs {
            points: GrowBuf::new(ctx, "force points", STORAGE), w4: GrowBuf::new(ctx, "force weights", STORAGE),
            rows: GrowBuf::new(ctx, "force rows", STORAGE), partials: GrowBuf::new(ctx, "force partials (pseudo)", STORAGE),
            staging: GrowBuf::new(ctx, "force staging (pseudo)", wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
            sr: (0, 0), pc: (0, 0), nl: (0, 0), row_ion: Vec::new(), nl_e_kb: Vec::new(), chunks: 1, valid: false,
            fields: None, field_allocations: 0,
        }
    }

    fn allocations(&self) -> usize {
        self.points.allocations + self.w4.allocations + self.rows.allocations + self.partials.allocations + self.staging.allocations + self.field_allocations
    }

    /// The weights of `nuclei` under `setup` in f64 on the CPU (module doc "Forces"), uploaded as f32.
    fn build(&mut self, ctx: &GpuContext, g: &GpuGrid, nuclei: &[Nucleus], setup: &IonicSetup) {
        let grid = qm_grid(g);
        let n = g.n;
        let pidx = |p: usize| g.pidx(p % n, (p / n) % n, p / (n * n)) as u32;
        let (mut points, mut w4, mut rows, mut row_ion) = (Vec::<u32>::new(), Vec::<[f32; 4]>::new(), Vec::<[u32; 4]>::new(), Vec::<usize>::new());
        let mut max_pts = 0usize;
        // one radial row: (f′(d)/d)(r − R) over the list (qm::forces' `radial_force` weight; 0 on the nucleus)
        let mut radial = |pts: &[usize], ion: usize, pos: [f64; 3], df: &dyn Fn(f64) -> f64| {
            let (pt_off, w_off) = (points.len() as u32, w4.len() as u32);
            for &p in pts {
                let r = grid.point(p);
                let d = [r[0] - pos[0], r[1] - pos[1], r[2] - pos[2]];
                let dd = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                let s = if dd < FORCE_ON_SITE { 0.0 } else { df(dd) / dd };
                points.push(pidx(p));
                w4.push([(s * d[0]) as f32, (s * d[1]) as f32, (s * d[2]) as f32, 0.0]);
            }
            rows.push([pt_off, pts.len() as u32, w_off, ion as u32]);
            row_ion.push(ion);
            max_pts = max_pts.max(pts.len());
        };
        // local SR: the one V_SR table over its sphere (qm::forces::local_sr_forces)
        let mut sr_end = 0u32; // rows so far (the closure holds `rows`)
        for (i, nuc) in nuclei.iter().enumerate() {
            if let Ionic::Pseudo(t) = setup.ionic(nuc.z) { radial(&points_within(&grid, nuc.pos, t.sr_radius()), i, nuc.pos, &|d| t.dv_sr(d)); sr_end += 1; }
        }
        let mut pc_end = sr_end;
        // partial core over its sphere (qm::forces::partial_core_forces; the no-core branch has no row)
        for (i, nuc) in nuclei.iter().enumerate() {
            if let Ionic::Pseudo(t) = setup.ionic(nuc.z) {
                if t.pc_radius > 0.0 { radial(&points_within(&grid, nuc.pos, t.pc_radius), i, nuc.pos, &|d| t.drho_pc_at(d)); pc_end += 1; }
            }
        }
        // nonlocal: (∇β_lm, β_lm) per sphere and m, from the reference's own f64 build (qm::forces::nonlocal_forces)
        let op = NonlocalOp::build_with_gradients(&grid, nuclei, setup);
        let mut nl_e_kb = Vec::new();
        for s in &op.spheres {
            let pt_off = points.len() as u32;
            points.extend(s.points.iter().map(|&p| pidx(p)));
            for (b, gb) in s.beta.iter().zip(&s.grad) {
                let w_off = w4.len() as u32;
                w4.extend(b.iter().zip(gb).map(|(&bv, gv)| [gv[0] as f32, gv[1] as f32, gv[2] as f32, bv as f32]));
                rows.push([pt_off, s.points.len() as u32, w_off, s.ion as u32]);
                row_ion.push(s.ion);
                nl_e_kb.push(s.e_kb);
            }
            max_pts = max_pts.max(s.points.len());
        }
        let nl_end = rows.len() as u32;
        self.points.write(ctx, bytemuck::cast_slice(&points));
        self.w4.write(ctx, bytemuck::cast_slice(&w4));
        self.rows.write(ctx, bytemuck::cast_slice(&rows));
        (self.sr, self.pc, self.nl) = ((0, sr_end), (sr_end, pc_end), (pc_end, nl_end));
        self.row_ion = row_ion;
        self.nl_e_kb = nl_e_kb;
        self.chunks = max_pts.div_ceil(256).clamp(1, MAX_CHUNKS) as u32;
        self.valid = true;
    }
}

/// The pseudised electronic force terms on the GPU, `[F_LR, F_SR, F_NL, F_pc]` per nucleus (Ha/Bohr) — the GPU form of
/// `qm::forces::{local_lr_forces, local_sr_forces, nonlocal_forces, partial_core_forces}` (module doc "Forces"):
/// - `rho` is the output density ρ↑ + ρ↓ (the local terms), `rho_up`/`rho_dn` the output spin densities (v̄ of the
///   partial-core term is formed here from them with the SCF's own `xc_in`, so it is bit for bit the v̄ of the solver's
///   final XC evaluation);
/// - `orbitals` are (ψ_iσ, f_iσ) over both spins, f = 0 skipped (the nonlocal term);
/// - `nuclei` must be the geometry `ionic` was last `update`d for, and `setup` the same physics (panics otherwise).
///
/// One submission and one readback; every sum is compensated on the GPU and finished in f64 on the CPU. The ion–ion
/// term is the caller's (`qm::forces::ion_ion_forces`; `GpuSolver::force_terms`).
#[allow(clippy::too_many_arguments)]
pub fn pseudo_electronic_forces(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, ionic: &GpuIonic, rho: &Field, rho_up: &Field, rho_dn: &Field,
                                orbitals: &[(&Field, f64)], nuclei: &[Nucleus], setup: &IonicSetup) -> [Vec<[f64; 3]>; 4] {
    check(g, &[rho, rho_up, rho_dn]);
    for (psi, _) in orbitals { check(g, &[psi]); }
    assert!(ionic.nuclei == nuclei, "pseudo_electronic_forces: the nuclei differ from the geometry the ionic terms were built for (call with the state's own nuclei)");
    assert!(ionic.setup.as_ref().is_some_and(|s| s.same_physics(setup)), "pseudo_electronic_forces: the setup differs from the one the ionic terms were built for");
    let n_nuc = nuclei.len();
    if n_nuc == 0 { return std::array::from_fn(|_| Vec::new()); }
    let mut fb = ionic.forces.borrow_mut();
    if !fb.valid { fb.build(ctx, g, nuclei, setup); }
    let live: Vec<(&Field, f64)> = orbitals.iter().copied().filter(|o| o.1 != 0.0).collect();
    let (wg, ch) = (centre_workgroups(g), fb.chunks);
    let (n_sr, n_pc, n_nl) = (fb.sr.1 - fb.sr.0, fb.pc.1 - fb.pc.0, fb.nl.1 - fb.nl.0);
    // the partials' layout (vec4 pairs): every centre, then the SR rows, the partial-core rows, the nonlocal rows per orbital
    let lr_base = |i: usize| 2 * wg * i as u32;
    let sr_base = 2 * wg * n_nuc as u32;
    let pc_base = sr_base + 2 * ch * n_sr;
    let nl_base = |o: usize| pc_base + 2 * ch * n_pc + 2 * ch * n_nl * o as u32;
    let total = nl_base(live.len()) as usize; // vec4 slots
    fb.partials.reserve(ctx, 16 * total as u64);
    fb.staging.reserve(ctx, 16 * total as u64);
    if n_pc > 0 && fb.fields.is_none() {
        fb.fields = Some(std::array::from_fn(|i| Field::new(ctx, g, ["force ρ_pc", "force n↑ / v̄", "force n↓", "force e_xc", "force v↑", "force v↓"][i])));
        fb.field_allocations += 6;
    }
    let fb = &*fb;
    let mut b = Batch::new(ctx, "pseudo forces");
    // v̄ = ½(v_xc↑ + v_xc↓) on (ρ↑ + ½ρ_pc, ρ↓ + ½ρ_pc): the SCF's final XC call (`scf::xc_in`), then into fields[1]
    if let Some(f) = fb.fields.as_ref().filter(|_| n_pc > 0) {
        ionic.partial_core_in(&mut b, k, g, &f[0]);
        crate::scf::xc_in(&mut b, k, g, rho_up, rho_dn, Some((&f[0], &f[1], &f[2])), &f[3], &f[4], &f[5]);
        crate::pointwise::lincomb_in(&mut b, k, g, &f[1], &[(0.5, &f[4]), (0.5, &f[5])]);
    }
    // local LR: one dispatch per centre, (q, σ) of `qm::nuclear::centres`
    for (i, c) in centres(nuclei, setup).iter().enumerate() {
        let (cell, frac) = cell_frac(g, c.pos);
        let a = c.sigma * std::f64::consts::SQRT_2;
        let params = CentreForceParams { cell, frac, p: g.p as u32, n: g.n as u32, base: lr_base(i), zero: 0,
                                         h_over_a: (g.h / a) as f32, c: (-c.charge * g.h / (a * a * a)) as f32, pad: [0.0; 2] };
        let u = b.uniform(ctx, k, "centre_force", &params);
        let bg = k.bind_group(ctx, "centre_force", &[u.binding(), rho.buffer.as_entire_binding(), fb.partials.buf.as_entire_binding()]);
        b.dispatch(k, "centre_force", &bg, [wg, 1, 1]);
    }
    // the gathers: (row range, base, x)
    let mut jobs: Vec<((u32, u32), u32, &Field)> = vec![(fb.sr, sr_base, rho)];
    if let Some(f) = fb.fields.as_ref().filter(|_| n_pc > 0) { jobs.push((fb.pc, pc_base, &f[1])); }
    for (o, (psi, _)) in live.iter().enumerate() { jobs.push((fb.nl, nl_base(o), psi)); }
    for ((r0, r1), base, x) in jobs {
        if r1 == r0 { continue; }
        let u = b.uniform(ctx, k, "gather_dot4", &GatherParams { chunks: ch, row0: r0, base, zero: 0 });
        let bg = k.bind_group(ctx, "gather_dot4", &[u.binding(), fb.points.buf.as_entire_binding(), fb.w4.buf.as_entire_binding(),
                                                    x.buffer.as_entire_binding(), fb.partials.buf.as_entire_binding(), fb.rows.buf.as_entire_binding()]);
        b.dispatch(k, "gather_dot4", &bg, [ch, r1 - r0, 1]);
    }
    b.encoder().copy_buffer_to_buffer(&fb.partials.buf, 0, &fb.staging.buf, 0, Some(16 * total as u64));
    b.submit();
    let raw = map_read_f32(ctx, &fb.staging.buf, 4 * total);
    // f64 finish: lane `lane` of the `pairs` (hi, lo) pairs from vec4 slot `base`, Neumaier, × h³
    let dv = g.dv();
    let sum = |base: u32, pairs: u32, lane: usize| -> f64 {
        let s = &raw[4 * base as usize..4 * (base + 2 * pairs) as usize];
        neumaier(s.chunks(4).map(|v| v[lane] as f64)) * dv
    };
    let lr: Vec<[f64; 3]> = (0..n_nuc).map(|i| std::array::from_fn(|a| sum(lr_base(i), wg, a))).collect();
    let radial = |(r0, r1): (u32, u32), base: u32| -> Vec<[f64; 3]> {
        let mut terms: Vec<[Vec<f64>; 3]> = vec![[Vec::new(), Vec::new(), Vec::new()]; n_nuc];
        for r in r0..r1 {
            let at = base + 2 * ch * (r - r0);
            for (a, t) in terms[fb.row_ion[r as usize]].iter_mut().enumerate() { t.push(sum(at, ch, a)); }
        }
        terms.into_iter().map(|t| std::array::from_fn(|a| neumaier(t[a].iter().copied()))).collect()
    };
    let sr = radial(fb.sr, sr_base);
    let pc = radial(fb.pc, pc_base);
    // nonlocal: F_I = Σ_iσ Σ_lm 2 f E_KB c ⟨∇β|ψ⟩ (qm::forces::nonlocal_forces)
    let mut terms: Vec<[Vec<f64>; 3]> = vec![[Vec::new(), Vec::new(), Vec::new()]; n_nuc];
    for (o, (_, f)) in live.iter().enumerate() {
        for r in fb.nl.0..fb.nl.1 {
            let j = r - fb.nl.0;
            let at = nl_base(o) + 2 * ch * j;
            let c = sum(at, ch, 3);
            let e_kb = fb.nl_e_kb[j as usize];
            for (a, t) in terms[fb.row_ion[r as usize]].iter_mut().enumerate() { t.push(2.0 * f * e_kb * c * sum(at, ch, a)); }
        }
    }
    let nl: Vec<[f64; 3]> = terms.into_iter().map(|t| std::array::from_fn(|a| neumaier(t[a].iter().copied()))).collect();
    [lr, sr, nl, pc]
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
            // Every case against the f32 sums' own scale (Higham: u·log₂n·Σ|β||x|), V_NL applied to |x|. The brief's
            // relative bound on x₀ holds on the 33³ case. On the clipped grid x₀'s c is a seed coincidence, not a
            // property of clipping (Task 8a review item 3, CPU-checked): c[x₀] = 4.49e-5 against an rms of 0.077, a
            // 5.8e-4σ draw, so 1e-5·max |V_NL x₀| = 1.9e-9 sits below any f32 floor (~3e-7 there). The brief's bound
            // is therefore asserted on x₁ on both grids, the clipped geometry included.
            if grid.n[0] == 33 { assert!(worst < 1e-5 * scale, "max |ΔV_NL x₀| = {worst:.2e} of {scale:.2e}"); }
            assert!(worst < 1e-5 * scale_abs, "max |ΔV_NL x₀| = {worst:.2e} of max |V_NL |x₀|| {scale_abs:.2e}");
            let mut cpu1 = vec![0.0; grid.len()];
            cpu_nl.apply_add(&xs[1], &mut cpu1);
            let out1 = Field::new(&ctx, &g, "out1");
            let mut b = Batch::new(&ctx, "t1");
            nl.apply_in(&mut b, &k, &g, &fields[1], &out1);
            b.submit();
            let got1 = out1.read_interior(&ctx, &g);
            let scale1 = cpu1.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let worst1 = cpu1.iter().zip(&got1).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            println!("  x₁: max |ΔV_NL x₁| {worst1:.2e} of {scale1:.2e} ({:.2e}; the brief's bound)", worst1 / scale1);
            assert!(worst1 < 1e-5 * scale1, "max |ΔV_NL x₁| = {worst1:.2e} of {scale1:.2e}");
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

    // ---- M7 Task 8b: the force terms ----

    fn dist(a: [f64; 3], b: [f64; 3]) -> f64 { ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt() }

    /// A field holding `data` (f32), and the f32-rounded values the GPU sees, as f64: the CPU references below run on
    /// the rounded inputs, so the comparison measures the kernels and not the input rounding.
    fn upload(ctx: &GpuContext, g: &GpuGrid, data: &[f64], label: &str) -> (Field, Vec<f64>) {
        let f = Field::new(ctx, g, label);
        f.upload_interior(ctx, g, data);
        let back = f.read_interior(ctx, g);
        (f, back)
    }

    /// Σ_p |t_p| per component of a radial term h³Σ_p x_p f′(d_p)(r_p − R)_a/d_p (`qm::forces`' `radial_force`): the
    /// gross that the odd integrand cancels down to its net — the scale of an f32 evaluation's error.
    fn radial_gross(grid: &qm::Grid, pts: Option<&[usize]>, c: [f64; 3], x: &[f64], df: &dyn Fn(f64) -> f64) -> [f64; 3] {
        let all: Vec<usize> = (0..grid.len()).collect();
        let mut out = [0.0f64; 3];
        for &p in pts.unwrap_or(&all) {
            let r = grid.point(p);
            let dd = dist(r, c);
            if dd < 1e-8 { continue; }
            let s = (x[p] * df(dd) / dd).abs();
            for a in 0..3 { out[a] += s * (r[a] - c[a]).abs(); }
        }
        out.map(|v| v * grid.dv())
    }

    /// The gross of each pseudised electronic term [LR, SR, NL, ρ_pc] per nucleus and component (`radial_gross`; for NL
    /// Σ 2f|E_KB c|·h³Σ_p|∂_aβ ψ| per orbital, sphere and m), on the inputs the CPU reference sees.
    #[allow(clippy::too_many_arguments)]
    fn gross_terms(grid: &qm::Grid, rho: &[f64], up: &[f64], dn: &[f64], orbitals: &[&[f64]], occ: &[f64], nuclei: &[Nucleus], s: &IonicSetup) -> [Vec<[f64; 3]>; 4] {
        let lr = nuclei.iter().map(|n| match s.ionic(n.z) {
            Ionic::AllElectron { sigma } => {
                let (z, a) = (n.z as f64, sigma * std::f64::consts::SQRT_2);
                radial_gross(grid, None, n.pos, rho, &|d| { let x = d / a; z * ((2.0 / std::f64::consts::PI.sqrt()) * x * (-x * x).exp() - libm::erf(x)) / (d * d) })
            }
            Ionic::Pseudo(t) => radial_gross(grid, None, n.pos, rho, &|d| t.dv_lr(d)),
        }).collect();
        let sr = nuclei.iter().map(|n| match s.ionic(n.z) {
            Ionic::Pseudo(t) => radial_gross(grid, Some(&points_within(grid, n.pos, t.sr_radius())), n.pos, rho, &|d| t.dv_sr(d)),
            _ => [0.0; 3],
        }).collect();
        let op = NonlocalOp::build_with_gradients(grid, nuclei, s);
        let mut nl = vec![[0.0f64; 3]; nuclei.len()];
        for sp in &op.spheres {
            for (psi, &f) in orbitals.iter().zip(occ) {
                for (b, gb) in sp.beta.iter().zip(&sp.grad) {
                    let c: f64 = sp.points.iter().zip(b).map(|(&p, bv)| bv * psi[p]).sum::<f64>() * op.dv;
                    for a in 0..3 { nl[sp.ion][a] += 2.0 * f * (sp.e_kb * c).abs() * sp.points.iter().zip(gb).map(|(&p, gv)| (gv[a] * psi[p]).abs()).sum::<f64>() * op.dv; }
                }
            }
        }
        let pc_field = qm::nuclear::partial_core_field(grid, nuclei, s);
        let u: Vec<f64> = up.iter().zip(&pc_field).map(|(a, c)| a + 0.5 * c).collect();
        let d: Vec<f64> = dn.iter().zip(&pc_field).map(|(a, c)| a + 0.5 * c).collect();
        let xc = qm::xc::lsda_pw92(&u, &d);
        let v_avg: Vec<f64> = xc.v_up.iter().zip(&xc.v_dn).map(|(a, b)| 0.5 * (a + b)).collect();
        let pc = nuclei.iter().map(|n| match s.ionic(n.z) {
            Ionic::Pseudo(t) if t.pc_radius > 0.0 => radial_gross(grid, Some(&points_within(grid, n.pos, t.pc_radius)), n.pos, &v_avg, &|d| t.drho_pc_at(d)),
            _ => [0.0; 3],
        }).collect();
        [lr, sr, nl, pc]
    }

    const TERM_NAMES: [&str; 4] = ["LR", "SR", "NL", "ρ_pc"];

    /// Every pseudised force term on the GPU (`pseudo_electronic_forces`) against `qm::forces`' term functions, on frozen
    /// synthetic states (as `qm::forces`' `force_terms_are_exact_derivatives_of_frozen_energies`: a density blob per
    /// O, spin 0.6/0.4, an s-like and a p-like orbital with f = 1 and ⅔, plus an f = 0 witness that must be skipped),
    /// the CPU fed the f32-rounded inputs. Geometries: `cases()` (pseudo-O beside an all-electron H on 33³ × 0.25, and
    /// the non-power-of-two 30³ × 0.20 whose −x face clips the O spheres), and Task 7's fixture (two pseudo-O with
    /// overlapping spheres and an all-electron H, 33³ × 0.20, spheres clipped by the 6.4 Bohr box).
    ///
    /// Tolerance (f32; module doc "Forces"): the sums are compensated, so what is left is per-point rounding — the f32
    /// weights and products (u = 6e-8 each), v̄, G(x) and ρ_pc in f32 (≲ 3e-7 relative), the f32 fraction of R — i.e.
    /// at worst ~3e-7 of the term's gross Σ_p|t_p| per component, at random far less. Measured (2026-10-17, RX 7900 XTX):
    /// worst |Δ|/gross LR 9.1e-8, SR 5.2e-9, NL 5.9e-9, ρ_pc 1.4e-8 (≤ 1.4e-7 Ha/Bohr absolute). The bounds started at
    /// 1e-6 and are pinned per term by the 3× rule (Task 8b review SHOULD-2: one LR-driven bound for all four would pass
    /// an uncompensated gather, ~u·Σ|t| ≈ 6e-8·gross): |ΔF| ≤ b·gross + 1e-12 with b = 3e-7 (LR), 2e-8 (SR), 2e-8 (NL),
    /// 5e-8 (ρ_pc), for every nucleus and component. The all-electron H's SR, NL and ρ_pc terms are exactly 0.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm-gpu --release")]
    fn pseudo_force_terms_match_cpu_on_frozen_states() {
        let Some(ctx) = gpu::GpuContext::new_headless() else { return };
        let k = Kernels::new(&ctx);
        let fixture = (qm::Grid::cube(33, 0.2), vec![Nucleus { pos: [-1.13, 0.07, 0.03], z: 8 }, Nucleus { pos: [1.16, -0.04, 0.05], z: 8 }, Nucleus { pos: [0.11, 1.62, -0.09], z: 1 }]);
        let mut worst_rel = [0.0f64; 4];
        let bound = [3e-7, 2e-8, 2e-8, 5e-8]; // × gross, per term (doc)
        for (grid, nuc) in cases().into_iter().chain(std::iter::once(fixture)) {
            let g = GpuGrid::new(&grid);
            let s = o_setup(grid.h);
            let os: Vec<[f64; 3]> = nuc.iter().filter(|n| n.z == 8).map(|n| n.pos).collect();
            let blob = |c: [f64; 3], w: f64| grid.sample(|r| (-dist(r, c).powi(2) / w).exp());
            let rho: Vec<f64> = os.iter().map(|&c| blob(c, 1.0)).fold(vec![0.0; grid.len()], |acc, b| acc.iter().zip(&b).map(|(x, y)| x + y).collect());
            let s_orb = blob(os[0], 1.5);
            let p_orb: Vec<f64> = grid.sample(|r| r[0] - os[0][0]).iter().zip(&s_orb).map(|(x, s)| x * s).collect();
            let w_orb = blob(os[os.len() - 1], 0.7);
            let (up_f, up) = upload(&ctx, &g, &rho.iter().map(|x| 0.6 * x).collect::<Vec<_>>(), "up");
            let (dn_f, dn) = upload(&ctx, &g, &rho.iter().map(|x| 0.4 * x).collect::<Vec<_>>(), "dn");
            let (rho_f, rho32) = upload(&ctx, &g, &rho, "rho");
            let (s_f, s32) = upload(&ctx, &g, &s_orb, "s");
            let (p_f, p32) = upload(&ctx, &g, &p_orb, "p");
            let (w_f, _) = upload(&ctx, &g, &w_orb, "witness");
            let ion = GpuIonic::new(&ctx, &g, &nuc, &s);
            let t0 = std::time::Instant::now();
            let gpu = pseudo_electronic_forces(&ctx, &k, &g, &ion, &rho_f, &up_f, &dn_f, &[(&s_f, 1.0), (&p_f, 2.0 / 3.0), (&w_f, 0.0)], &nuc, &s);
            let first_ms = t0.elapsed().as_secs_f64() * 1e3;
            let orbs: [&[f64]; 2] = [&s32, &p32];
            let occ = [1.0, 2.0 / 3.0];
            let cpu = [qm::forces::local_lr_forces(&grid, &rho32, &nuc, &s), qm::forces::local_sr_forces(&grid, &rho32, &nuc, &s),
                       qm::forces::nonlocal_forces(&grid, &orbs, &occ, &nuc, &s), qm::forces::partial_core_forces(&grid, &up, &dn, &nuc, &s)];
            let gross = gross_terms(&grid, &rho32, &up, &dn, &orbs, &occ, &nuc, &s);
            for (t, name) in TERM_NAMES.iter().enumerate() {
                let (mut worst, mut at) = (0.0f64, (0, 0));
                for i in 0..nuc.len() { for a in 0..3 {
                    let (d, gr) = ((gpu[t][i][a] - cpu[t][i][a]).abs(), gross[t][i][a]);
                    assert!(d <= bound[t] * gr + 1e-12, "{name}, {}³ × {}, nucleus {i}, axis {a}: gpu {:+.10e} cpu {:+.10e} (|Δ| {d:.2e}, gross {gr:.3e})", grid.n[0], grid.h, gpu[t][i][a], cpu[t][i][a]);
                    if gr > 0.0 && d / gr > worst { worst = d / gr; at = (i, a); }
                }}
                worst_rel[t] = worst_rel[t].max(worst);
                let (i, a) = at;
                println!("{}³ × {}: {name:<4} worst |Δ|/gross {worst:.2e} (nucleus {i}, axis {a}: gpu {:+.6e} cpu {:+.6e}, |Δ| {:.2e}, gross {:.3e})",
                         grid.n[0], grid.h, gpu[t][i][a], cpu[t][i][a], (gpu[t][i][a] - cpu[t][i][a]).abs(), gross[t][i][a]);
            }
            for (i, n) in nuc.iter().enumerate() {
                if n.z == 1 { for t in 1..4 { assert_eq!(gpu[t][i], [0.0; 3], "{} on the all-electron H", TERM_NAMES[t]); } }
            }
            let t1 = std::time::Instant::now();
            let _ = pseudo_electronic_forces(&ctx, &k, &g, &ion, &rho_f, &up_f, &dn_f, &[(&s_f, 1.0), (&p_f, 2.0 / 3.0), (&w_f, 0.0)], &nuc, &s);
            println!("{}³ × {}: first call (weights built) {first_ms:.1} ms, second {:.1} ms", grid.n[0], grid.h, t1.elapsed().as_secs_f64() * 1e3);
        }
        println!("worst |Δ|/gross over the geometries: LR {:.2e}, SR {:.2e}, NL {:.2e}, ρ_pc {:.2e} (bounds 3e-7, 2e-8, 2e-8, 5e-8)", worst_rel[0], worst_rel[1], worst_rel[2], worst_rel[3]);
    }

    /// NLCC ruling, Ruling 6 (Task 8b): the lone O's partial-core force term on the GPU. The frozen spherical filtered O
    /// Hund atom (the radial KB atom's valence spin densities sampled on the grid, as `qm::forces`'
    /// `partial_core_force_is_the_radial_lattice_sum`) at R = (h/4)(1, 1, 1) and a low-symmetry offset, h = 0.20 (O's
    /// h_req) and 0.15 (its h_eb): GPU F_pc vs `qm::forces::partial_core_forces` on the f32-rounded densities, and vs
    /// T8(e)'s lattice sum `pc_lattice_force` (the CPU's own identity). This is the most cancelling term of the scene:
    /// Σ_p|t_p|/|F| ≈ 2e6 (Task 7), so a plain f32 sum (~u·Σ|t| per tree level) would be of the order of F itself — the
    /// reason the kernels sum in double-f32. Measured (2026-10-17): |ΔF|∞ ≤ 1.2e-8 Ha/Bohr, 8.4e-9 of Σ|t| and 0.4–1.9 %
    /// of |F_cpu| (the lattice sum is printed beside them). Asserted: |ΔF|∞ ≤ 3e-8·Σ|t| (3× the measured; it
    /// started at 1e-7) and ≤ 5 % of |F_cpu| (2.7× the measured).
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm-gpu --release")]
    fn lone_o_partial_core_force_matches_cpu() {
        let Some(ctx) = gpu::GpuContext::new_headless() else { return };
        let k = Kernels::new(&ctx);
        let hund_ae = qm::radial::ground_state(8, qm::radial::SpinMode::Polarised).expect("the Hund atom");
        let hund = qm::radial::transfer_configs(&table(8), &hund_ae).into_iter().find(|c| c.name == "Hund").expect("the Hund configuration");
        for h in [0.20, 0.15] {
            let ft = filtered_at(8, h);
            assert!(ft.pc_radius > 0.0, "O has a partial core");
            let atom = qm::radial::kb::solve_kb_atom(&ft.grid, &ft.input(), &hund.pp, hund.spin).expect("the filtered Hund KB atom");
            let rho_v = |d: f64| [ft.grid.interp(&atom.rho[0], d), ft.grid.interp(&atom.rho[1], d)];
            let rho_pc = |d: f64| (ft.rho_pc_at(d), ft.drho_pc_at(d));
            let s = Arc::new(IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, BTreeMap::from([(8, Ionic::Pseudo(ft.clone()))])));
            let n = 2 * ((ft.pc_radius / h).ceil() as usize + 3) + 1;
            let grid = qm::Grid::cube(n, h);
            let g = GpuGrid::new(&grid);
            for r in [[0.25 * h, 0.25 * h, 0.25 * h], [0.37 * h, -0.11 * h, 0.23 * h]] {
                let nuc = [Nucleus { pos: r, z: 8 }];
                let (up_f, up) = upload(&ctx, &g, &grid.sample(|p| rho_v(dist(p, r))[0]), "up");
                let (dn_f, dn) = upload(&ctx, &g, &grid.sample(|p| rho_v(dist(p, r))[1]), "dn");
                let rho_f = Field::new(&ctx, &g, "rho"); // the local terms are not under test here
                let ion = GpuIonic::new(&ctx, &g, &nuc, &s);
                let gpu = pseudo_electronic_forces(&ctx, &k, &g, &ion, &rho_f, &up_f, &dn_f, &[], &nuc, &s)[3][0];
                let cpu = qm::forces::partial_core_forces(&grid, &up, &dn, &nuc, &s)[0];
                let lattice = qm::radial::filter::pc_lattice_force(h, ft.pc_radius, r, &rho_v, &rho_pc);
                let gross = gross_terms(&grid, &vec![0.0; grid.len()], &up, &dn, &[], &[], &nuc, &s)[3][0].iter().cloned().fold(0.0, f64::max);
                let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                let d = (0..3).map(|a| (gpu[a] - cpu[a]).abs()).fold(0.0, f64::max);
                println!("lone O Hund, h = {h}, {n}³, R = {r:.4?}: F_pc gpu ({:+.6e}, {:+.6e}, {:+.6e}) cpu ({:+.6e}, {:+.6e}, {:+.6e}) lattice ({:+.6e}, {:+.6e}, {:+.6e}); |Δ|∞ {d:.2e} = {:.2e} of |F_cpu| {:.3e}, {:.2e} of the gross {gross:.3e}",
                         gpu[0], gpu[1], gpu[2], cpu[0], cpu[1], cpu[2], lattice[0], lattice[1], lattice[2], d / norm(cpu), norm(cpu), d / gross);
                assert!(d <= 3e-8 * gross, "h = {h}, R = {r:?}: |ΔF_pc|∞ {d:.2e} > 3e-8·Σ|t| = {:.2e}", 3e-8 * gross);
                assert!(d <= 0.05 * norm(cpu), "h = {h}, R = {r:?}: |ΔF_pc|∞ {d:.2e} > 5 % of |F_cpu| {:.2e}", norm(cpu));
            }
        }
    }

    /// The force weights follow the geometry (`update` invalidates them; the next force call rebuilds them in place): a
    /// moved and a moved-back geometry allocate nothing once the scene's sizes are reached, the moved-back forces equal
    /// the first ones bit for bit, and a fresh `GpuIonic` at the moved geometry gives the moved forces bit for bit.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm-gpu --release")]
    fn force_weights_follow_the_geometry_without_allocating() {
        let Some(ctx) = gpu::GpuContext::new_headless() else { return };
        let k = Kernels::new(&ctx);
        let grid = qm::Grid::cube(33, 0.25);
        let g = GpuGrid::new(&grid);
        let s = o_setup(0.25);
        let blob = grid.sample(|r| (-(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) / 2.0).exp());
        let (rho, _) = upload(&ctx, &g, &blob, "rho");
        let (up, _) = upload(&ctx, &g, &blob.iter().map(|x| 0.5 * x).collect::<Vec<_>>(), "up");
        let (psi, _) = upload(&ctx, &g, &grid.sample(|r| (r[0] + 0.3) * (-(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) / 3.0).exp()), "psi");
        let moved: Vec<Nucleus> = nuclei().iter().map(|n| Nucleus { pos: [n.pos[0] + 0.03, n.pos[1] - 0.02, n.pos[2]], ..*n }).collect();
        let mut ion = GpuIonic::new(&ctx, &g, &nuclei(), &s);
        let forces = |ion: &GpuIonic, nuc: &[Nucleus]| pseudo_electronic_forces(&ctx, &k, &g, ion, &rho, &up, &up, &[(&psi, 1.0)], nuc, &s);
        let f0 = forces(&ion, &nuclei());
        ion.update(&ctx, &g, &moved, &s);
        let f1 = forces(&ion, &moved);
        ion.update(&ctx, &g, &nuclei(), &s);
        let before = ion.buffer_allocations();
        let f2 = forces(&ion, &nuclei());
        ion.update(&ctx, &g, &moved, &s);
        let f3 = forces(&ion, &moved);
        assert_eq!(ion.buffer_allocations(), before, "a moved geometry of the same scene allocated");
        let bits = |f: &[Vec<[f64; 3]>; 4]| f.iter().flatten().flatten().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&f0), bits(&f2), "the same geometry gives the same forces");
        assert_eq!(bits(&f1), bits(&f3));
        assert_ne!(bits(&f0), bits(&f1), "the weights were rebuilt for the moved geometry");
        let fresh = GpuIonic::new(&ctx, &g, &moved, &s);
        assert_eq!(bits(&forces(&fresh, &moved)), bits(&f1));
    }

    /// The forces refuse a geometry the ionic terms were not built for (the weights and ρ_pc would be another scene's).
    #[test]
    #[should_panic(expected = "the nuclei differ")]
    fn forces_refuse_another_geometry() {
        let Some(ctx) = gpu::GpuContext::new_headless() else { panic!("the nuclei differ (no adapter: skipped)") };
        let k = Kernels::new(&ctx);
        let grid = qm::Grid::cube(17, 0.25);
        let g = GpuGrid::new(&grid);
        let s = o_setup(0.25);
        let ion = GpuIonic::new(&ctx, &g, &nuclei(), &s);
        let f = Field::new(&ctx, &g, "zero");
        let mut other = nuclei();
        other[0].pos[0] += 0.1;
        let _ = pseudo_electronic_forces(&ctx, &k, &g, &ion, &f, &f, &f, &[], &other, &s);
    }
}
