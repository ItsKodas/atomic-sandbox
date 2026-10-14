//! Orbital display products on the device (orbital-viewer spec §4.2, §5;
//! `docs/superpowers/specs/2026-10-15-orbital-viewer-design.md`; plan Task 2). The CPU f64
//! numerics — shells, alignment, labels, tracks, signs, colour slots — are `qm::orbitals`; this
//! module gives them their inputs from the GPU state with **one scalar readback** and turns their
//! output into display fields and textures **GPU-to-GPU**. Display only: nothing here writes a
//! solver field or touches the solver's warm state, scratch or knobs (rule 6; pinned by
//! `update_does_not_write_solver_fields` and
//! `display_updates_between_solves_leave_the_scf_bitwise_unchanged`). Atomic units.
//!
//! # One update ([`OrbitalView::update`])
//! 1. **One readback** ([`OrbitalView::read_scalars`]; `Reducer::combined_in`: one submission,
//!    one staging buffer, one map — counted by `grid::readbacks_on_this_thread`):
//!    - the signed overlaps ⟨ψ_i|ψ̃_j^prev⟩ of the solver's orbitals with this view's previously
//!      displayed ones, per channel, and ⟨ψ_i↑|ψ_j↓⟩ (the fused Gram kernel, `block_dots`' path);
//!    - max|ψ_i| (`Reduction::AbsMax`, one pass, exact);
//!    - the spin readout ∫m, ∫|m|, max|m| (m = ρ↑ − ρ↓ formed by a lincomb into display scratch)
//!      and max ρ;
//!    - per orbital, the `orbital_moments` kernel (`shaders/orbital_moments.wgsl`): the 16 b_lm
//!      and 16 c_lm moments about the nuclear centroid, the nine windowed first-sign functionals,
//!      the outer-L/6 weight with the cell-fraction weight, and — lone atom only — the radial-
//!      shell profile (16 Y_lm sums per bin of h/2); plus one geometry dispatch for the
//!      functionals' normalisers ‖w_i‖² and the bin point counts n_k.
//!
//!    Every sum is f32 per thread → a workgroup tree → f64 Neumaier on the CPU (rule 4; no float
//!    atomics). Maxima are exact.
//! 2. **CPU f64**: `qm::orbitals::Tracker::update` on those scalars.
//! 3. **GPU writes, no wait**: lincombs ψ̃ = Σ_j ψ_j U_jk into this view's own display fields
//!    (`lincomb_in`; they become the next update's "previous" set), then the product: the
//!    selected orbital (or ψ̃², via `mul` into scratch) or the spin density `unpad`ed and copied
//!    into an r32float `Volume3D` exactly as `GpuSolver::copy_density_to_volume` does, or the
//!    `colour_accumulate` kernel (`shaders/colour_accumulate.wgsl`) writing an rgba16float
//!    `ColourVolume3D`. One submission; nothing waits for it.
//!
//! No orbital or field value crosses to the CPU: the readback carries O(K² + 42K) finished
//! scalars (+ 16·N_bins per orbital for a lone atom), from partials of a few hundred KB.
//!
//! **Dispatches per update** (K = n↑ + n↓ orbitals, P = previous displayed ones, G = the fused
//! Gram dispatches for the K·P + n↑n↓ overlap pairs, ≈ 1–4 for K ≤ 16): readback batch
//! 1 lincomb (m) + K + 1 `orbital_moments` + G + K `abs_max` + 4 spin/ρ reductions; display
//! batch ≈ K lincombs (one per displayed orbital; a set of d > 7 needs ⌈d/7⌉) + the product:
//! 1 unpad + 1 copy (orbital phase, spin), 1 mul + 1 unpad + 1 copy (|ψ|²), ⌈included/6⌉
//! `colour_accumulate` (colour), 0 (none).
//!
//! **Cost** (RX 7900 XTX shared with other suites, so contended; minimum of 20,
//! `diag_orbital_update_phases`): 97³, K = 10: the scalar readback 0.9–1.1 ms for a molecule,
//! 2.0–2.4 ms for a lone atom (the radial profiles), the whole update 1.3–3.6 ms; the CPU step
//! (`Tracker::update`) ≈ 6 µs. Two measured choices: `MOMENT_WORKGROUPS` 256 instead of 64
//! (readback 2.4–2.6 → 1.0–1.5 ms, interleaved A/B; more threads hide the per-point latency) and
//! the bin workgroups' column split with a 0.05-grid-unit shell margin instead of one workgroup
//! per bin and a 1-unit margin (radial extra 2.8–6.1 → 1.0–1.3 ms).
//!
//! # Display fields and VRAM (spec §4.4)
//! The view owns its fields (its own pool, never the solver's): the previous displayed set
//! (K + m padded fields; m = 0 until the unoccupied orbitals of plan Task 4, which add m more
//! guess vectors: K + 2m), one scratch field (m, or ψ̃²), the row-padded copy buffer
//! (stride·n²·4 B), the colour accumulator (n³ × 16 B, only while more than
//! `COLOUR_MAX_TERMS` orbitals are included) and small buffers (nuclei, moment partials, its
//! `Reducer`'s scratch). At 97³ a padded field is 4.63 MB: K = 10 is 46 MB + 4.6 MB scratch +
//! 4.0 MB rows (+ 14.6 MB accumulator). Display fields are dropped on [`OrbitalView::reset`]
//! and on a grid change (the next update rebuilds them).
//!
//! # Precision
//! The moments are the CPU's (`qm::orbitals::moments`, …) to the f32 partial-sum level:
//! |Δ| ≲ (points per thread + 8 tree levels + ~8 for the harmonics)·2⁻²⁴·Σ|ψ·w|·dv
//! (`moments_match_the_cpu_reference`). The radial profiles bin a point by its f32 |dc|·2, so a
//! point within ~1e-6 grid units of a bin edge may sit in a neighbouring bin relative to the f64
//! CPU; the counts n_k come from the same kernel formula, so each side's R̄_k is consistent
//! (`radial_profiles_match_the_cpu_reference`). The displayed fields are f32 lincombs with f32
//! coefficients: |ψ̃_gpu − ψ̃_cpu| ≲ d·2⁻²⁴·max|ψ| (`display_copy_matches_cpu_reference`).
use crate::batch::{Batch, Recorder};
use crate::grid::{readbacks_on_this_thread, Field, GpuGrid};
use crate::kernels::{workgroups_for, Kernels, COLOUR_MAX_TERMS, WORKGROUP};
use crate::physics::{NucleusRecord, MAX_NUCLEI};
use crate::pointwise::{lincomb_in, mul_in, unpad_rec, unpad_row_stride};
use crate::reduce::{neumaier, Reducer, Reduction};
use crate::scf::{GpuSolver, GpuState};
use gpu::{ColourVolume3D, GpuContext, Volume3D};
use nalgebra::DMatrix;
use qm::orbitals::{Tracker, UpdateInput, N_LM, RADIAL_BIN_H};
pub use qm::orbitals::{Moments, RadialProfile, Spin, SpinSummary, UpdateOutput};

/// The colours of the 12 colour slots of `qm::orbitals` (`PALETTE_SLOTS`; spec §3: a fixed
/// qualitative palette — display data, not physics). Linear RGB in [0, 1].
pub const ORBITAL_PALETTE: [[f32; 3]; 12] = [
    [0.122, 0.467, 0.706], [1.000, 0.498, 0.055], [0.173, 0.627, 0.173], [0.839, 0.153, 0.157],
    [0.580, 0.404, 0.741], [0.549, 0.337, 0.294], [0.890, 0.467, 0.761], [0.498, 0.498, 0.498],
    [0.737, 0.741, 0.133], [0.090, 0.745, 0.812], [0.682, 0.780, 0.910], [1.000, 0.733, 0.471],
];

/// Upper bound on the `orbital_moments` moment workgroups (= f32 partials per lane).
pub const MOMENT_WORKGROUPS: u32 = 256;
/// Lanes per moment workgroup: b (16), c (16), first-sign (9), outer weight, 2 unused.
const MOMENT_LANES: usize = 44;
/// Lanes per radial bin: 16 Y_lm sums, the count, 3 unused.
const BIN_LANES: usize = 20;
/// Columns (i, j) per thread of a radial bin's workgroups: a bin's bounding square is shared out
/// over ⌈columns / (256·this)⌉ workgroups (sized by the outermost bin), so each thread scans a
/// few columns instead of one workgroup scanning the whole shell (`orbital_moments.wgsl`).
const BIN_COLUMNS_PER_THREAD: usize = 4;
/// Lane offsets in a moment block.
const LANE_C: usize = 16;
const LANE_SIGN: usize = 32;
const LANE_OUTER: usize = 41;
const LANE_COUNT: usize = 16;

/// What one update writes (spec §3's modes). `track` selects by track (spec §5.4), not index.
#[derive(Clone, Copy)]
pub enum Product<'a> {
    /// The displayed orbital of `spin` with `track`: ψ̃ (signed) or ψ̃² (`squared`) into `out`.
    Orbital { spin: Spin, track: u32, squared: bool, out: &'a Volume3D },
    /// The colour volume over the included tracks: rgb = Σ f c ψ̃²/P, a = Σ f ψ̃²/P with
    /// P = (max|ψ| over the orbital's set)² or, `physical`, max ρ (spec §5.6).
    Colour { include: &'a [u32], physical: bool, out: &'a ColourVolume3D },
    /// m = ρ↑ − ρ↓ (signed) into `out`.
    Spin { out: &'a Volume3D },
    /// The list only (scalars, rotations, display fields), no texture.
    None,
}

impl std::fmt::Debug for Product<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Product::Orbital { spin, track, squared, .. } => write!(f, "Orbital {{ {spin:?}, track {track}, squared {squared} }}"),
            Product::Colour { include, physical, .. } => write!(f, "Colour {{ {} tracks, physical {physical} }}", include.len()),
            Product::Spin { .. } => write!(f, "Spin"),
            Product::None => write!(f, "None"),
        }
    }
}

/// The result of one update. `scale` is what 1.0 means in the written texture (spec §3):
/// max|ψ| over the orbital's set (phase), its square (|ψ|²), `SpinSummary::scale` (spin), 1
/// (colour: the kernel already divides by P). `scale == 0.0` ⇔ nothing was written
/// (`Product::None`, or an `Orbital` whose track is not shown by this update). `readbacks` is
/// the number of GPU→CPU maps the update made (always 1).
#[derive(Debug)]
pub struct ViewUpdate { pub out: UpdateOutput, pub spin: SpinSummary, pub scale: f32, pub readbacks: usize }

/// The finished scalars of one readback (step 1 of the module doc), per channel `[↑, ↓]`, rows
/// the solver's orbitals in order. `moments` are about the nuclear centroid (used only for a
/// lone atom); `radial` is `Some` iff exactly one nucleus; `overlaps_prev[ch]` is `None` when
/// the view displays nothing yet in that channel; `cross` is `None` unless both channels have
/// orbitals.
#[derive(Debug, Clone)]
pub struct Scalars {
    pub overlaps_prev: [Option<DMatrix<f64>>; 2],
    pub cross: Option<DMatrix<f64>>,
    pub moments: [Vec<Moments>; 2],
    pub radial: Option<[Vec<RadialProfile>; 2]>,
    pub first_sign: [Vec<[f64; 9]>; 2],
    pub max_abs: [Vec<f64>; 2],
    pub outer_weight: [Vec<f64>; 2],
    pub spin: SpinSummary,
}

/// Uniform of `orbital_moments` (layout: `shaders/orbital_moments.wgsl`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct MomParams { cell: [i32; 4], frac: [f32; 4], n: u32, p: u32, mom_wg: u32, n_bins: u32, r_max: f32, bin_scale: f32, slab: f32, geometry: u32, bin_split: u32, pad: [u32; 3] }

/// Uniform of `colour_accumulate`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ColourParams { n: u32, p: u32, count: u32, flags: u32, w: [[f32; 4]; COLOUR_MAX_TERMS] }

/// The display products of one producer (spec §4.3: one per provider; its `Tracker` holds the
/// tracks, slots and signs). Built for the solver's grid and device; `Send`.
pub struct OrbitalView {
    ctx: GpuContext,
    grid: qm::Grid,
    g: GpuGrid,
    reducer: Reducer,
    tracker: Tracker,
    /// The previous update's displayed orbitals per channel, in shown-row (index) order.
    prev: [Vec<Field>; 2],
    /// Display fields not in use (reused when the orbital count changes; dropped on reset).
    pool: Vec<Field>,
    /// Display scratch: m = ρ↑ − ρ↓ (every update), then ψ̃² for a squared product.
    scratch: Option<Field>,
    /// Row-padded buffer for the texture copies and its stride in floats (`unpad_row_stride`).
    rows: wgpu::Buffer,
    row_stride: u32,
    /// The kernels' nucleus records (`MAX_NUCLEI`, rewritten each update).
    nuclei: wgpu::Buffer,
    /// `orbital_moments` output, all dispatches' blocks (grow-only), and its size in bytes.
    moments_out: Option<(wgpu::Buffer, u64)>,
    /// Colour running sum when more than `COLOUR_MAX_TERMS` orbitals are included.
    colour_acc: Option<wgpu::Buffer>,
    /// Bound in place of the accumulator when a single chunk does not use it.
    dummy: wgpu::Buffer,
}

impl OrbitalView {
    /// A view for `solver`'s grid and device (its own reducer, buffers and tracker). Display
    /// fields are created by the first update.
    pub fn new(solver: &GpuSolver) -> OrbitalView {
        let ctx = solver.ctx.clone();
        let g = solver.g;
        let (rows, row_stride) = Self::rows(&ctx, &g);
        let nuclei = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orbital view nuclei"), size: (MAX_NUCLEI * std::mem::size_of::<NucleusRecord>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let dummy = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orbital view dummy"), size: 16, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
        });
        OrbitalView {
            reducer: Reducer::new(&ctx, &solver.k, &g), grid: solver.grid().clone(), g, tracker: Tracker::new(),
            prev: [Vec::new(), Vec::new()], pool: Vec::new(), scratch: None, rows, row_stride, nuclei,
            moments_out: None, colour_acc: None, dummy, ctx,
        }
    }

    fn rows(ctx: &GpuContext, g: &GpuGrid) -> (wgpu::Buffer, u32) {
        let stride = unpad_row_stride(g.n);
        let rows = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orbital view rows"), size: stride as u64 * (g.n * g.n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
        });
        (rows, stride)
    }

    /// Forgets the previous display (spec §5.4 item 5: grid change, provider reset, view off)
    /// and drops the display fields. Track ids keep counting (`Tracker::reset`).
    pub fn reset(&mut self) {
        self.tracker.reset();
        self.prev = [Vec::new(), Vec::new()];
        self.pool.clear();
        self.scratch = None;
        self.colour_acc = None;
    }

    /// Rebuilds the grid-sized resources and resets the tracker if `solver`'s grid changed.
    fn follow_grid(&mut self, solver: &GpuSolver, state: &GpuState) {
        assert!(state.grid == *solver.grid(), "OrbitalView: the state is from another grid than the solver's");
        if self.grid == *solver.grid() { return; }
        self.reset();
        self.grid = solver.grid().clone();
        self.g = solver.g;
        self.reducer = Reducer::new(&self.ctx, &solver.k, &self.g);
        (self.rows, self.row_stride) = Self::rows(&self.ctx, &self.g);
        self.moments_out = None;
    }

    fn field(pool: &mut Vec<Field>, ctx: &GpuContext, g: &GpuGrid) -> Field {
        pool.pop().unwrap_or_else(|| Field::new(ctx, g, "orbital display"))
    }

    /// Step 1 of the module doc: every scalar the CPU step needs, in one readback. Records the
    /// spin scratch m = ρ↑ − ρ↓ as a side effect (the Spin product reads it). Reads the solver's
    /// fields only.
    pub fn read_scalars(&mut self, solver: &GpuSolver, state: &GpuState, nuclei: &[qm::Nucleus]) -> Scalars {
        self.follow_grid(solver, state);
        assert!(nuclei.len() <= MAX_NUCLEI, "OrbitalView: {} nuclei > MAX_NUCLEI", nuclei.len());
        let (k, g, grid) = (&solver.k, self.g, &self.grid);
        let raw: [&[Field]; 2] = [&state.orbitals_up, &state.orbitals_dn];
        let n_ch = [raw[0].len(), raw[1].len()];
        let n_orb = n_ch[0] + n_ch[1];
        let lone = nuclei.len() == 1;

        // the centre of the moments, windows and bins: the nuclear centroid (the nucleus of a
        // lone atom), split into cell + fraction as the nuclear kernels do
        let centre: [f64; 3] = if nuclei.is_empty() {
            std::array::from_fn(|a| grid.origin[a] + (grid.n[a] - 1) as f64 * grid.h / 2.0)
        } else {
            std::array::from_fn(|a| nuclei.iter().map(|n| n.pos[a]).sum::<f64>() / nuclei.len() as f64)
        };
        let crec = NucleusRecord::new(&g, &qm::Nucleus { pos: centre, z: 0 });
        // the radial bins exactly as `qm::orbitals::radial_profiles` sizes them (f64)
        let (r_max, n_bins) = if lone {
            let r = (0..3).map(|a| {
                let lo = centre[a] - grid.origin[a];
                let hi = grid.origin[a] + (grid.n[a] - 1) as f64 * grid.h - centre[a];
                lo.min(hi)
            }).fold(f64::INFINITY, f64::min).max(0.0);
            (r, (r / (RADIAL_BIN_H * grid.h)).ceil() as usize)
        } else { (0.0, 0) };
        let mom_wg = (g.interior_len().div_ceil(WORKGROUP as usize) as u32).clamp(1, MOMENT_WORKGROUPS) as usize;
        // workgroups per bin: the outermost bin's bounding square (side ≤ n) over 256·BIN_COLUMNS_PER_THREAD
        let side = (2.0 * (n_bins as f64 * RADIAL_BIN_H + 0.05) + 3.0).ceil().min(g.n as f64) as usize;
        let bin_split = (side * side).div_ceil(WORKGROUP as usize * BIN_COLUMNS_PER_THREAD).max(1);
        let block = (MOMENT_LANES * mom_wg + BIN_LANES * n_bins * bin_split).div_ceil(64) * 64; // floats; 256-byte aligned blocks
        let bytes = ((n_orb + 1) * block * 4) as u64;
        if self.moments_out.as_ref().is_none_or(|(_, cap)| *cap < bytes) {
            self.moments_out = Some((self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("orbital moments"), size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
            }), bytes));
        }
        let records: Vec<NucleusRecord> = if nuclei.is_empty() { vec![NucleusRecord::default()] }
                                           else { nuclei.iter().map(|n| NucleusRecord::new(&g, n)).collect() };
        self.ctx.queue.write_buffer(&self.nuclei, 0, bytemuck::cast_slice(&records));
        if self.scratch.is_none() { self.scratch = Some(Self::field(&mut self.pool, &self.ctx, &g)); }
        let scratch = self.scratch.as_ref().expect("created above");
        let (mom_buf, _) = self.moments_out.as_ref().expect("created above");

        let mut b = Batch::new(&self.ctx, "orbital view scalars");
        let ctx = b.ctx();
        lincomb_in(&mut b, k, &g, scratch, &[(1.0, &state.rho_up), (-1.0, &state.rho_dn)]);
        let base = MomParams {
            cell: [crec.cell[0], crec.cell[1], crec.cell[2], nuclei.len() as i32],
            frac: [crec.frac[0], crec.frac[1], crec.frac[2], grid.h as f32],
            n: g.n as u32, p: g.p as u32, mom_wg: mom_wg as u32, n_bins: 0,
            r_max: (r_max / grid.h) as f32, bin_scale: (1.0 / RADIAL_BIN_H) as f32, slab: (g.n + 1) as f32 / 6.0, geometry: 0,
            bin_split: bin_split as u32, pad: [0; 3],
        };
        let sources = raw[0].iter().chain(raw[1]).map(|f| (f, 0u32)).chain(std::iter::once((&state.rho, 1u32)));
        for (slot, (psi, geometry)) in sources.enumerate() {
            let params = MomParams { n_bins: n_bins as u32, geometry, ..base };
            let u = b.uniform(ctx, k, "orbital_moments", &params);
            let out = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: mom_buf, offset: (slot * block * 4) as u64, size: wgpu::BufferSize::new((block * 4) as u64),
            });
            let bg = k.bind_group(ctx, "orbital_moments", &[u.binding(), psi.buffer.as_entire_binding(), self.nuclei.as_entire_binding(), out]);
            b.dispatch(k, "orbital_moments", &bg, [(mom_wg + n_bins * bin_split) as u32, 1, 1]);
        }
        // overlaps with the previous display (rows: current, columns: previous), then ↑·↓
        let mut pairs: Vec<(&Field, &Field)> = Vec::new();
        for (psis, shown) in raw.iter().zip(&self.prev) {
            for psi in *psis { for prev in shown { pairs.push((psi, prev)); } }
        }
        let cross = n_ch[0] > 0 && n_ch[1] > 0;
        if cross { for u in raw[0] { for d in raw[1] { pairs.push((u, d)); } } }
        let mut ops: Vec<Reduction> = raw[0].iter().chain(raw[1]).map(Reduction::AbsMax).collect();
        ops.extend([Reduction::Sum(scratch), Reduction::AbsDiffSum(&state.rho_up, &state.rho_dn), Reduction::AbsMax(scratch), Reduction::Max(&state.rho)]);
        let c = self.reducer.combined_in(&mut b, k, &g, &pairs, &ops, Some((mom_buf, bytes)));
        drop(b);

        // overlaps
        let mut at = 0;
        let overlaps_prev: [Option<DMatrix<f64>>; 2] = std::array::from_fn(|ch| {
            let np = self.prev[ch].len();
            if np == 0 { return None; }
            let m = DMatrix::from_fn(n_ch[ch], np, |i, j| c.dots[at + i * np + j]);
            at += n_ch[ch] * np;
            Some(m)
        });
        let cross = cross.then(|| DMatrix::from_fn(n_ch[0], n_ch[1], |i, j| c.dots[at + i * n_ch[1] + j]));
        // reductions
        let max_abs = [c.ops[..n_ch[0]].to_vec(), c.ops[n_ch[0]..n_orb].to_vec()];
        let s = &c.ops[n_orb..];
        let spin = SpinSummary { n_up: n_ch[0], n_dn: n_ch[1], integral: s[0], abs_integral: s[1], max_abs: s[2], rho_max: s[3] };
        // moments, functionals, outer weights, radial profiles
        let dv = grid.dv();
        let lane = |blk: &[f32], l: usize| neumaier(blk[l * mom_wg..(l + 1) * mom_wg].iter().map(|v| *v as f64)) * dv;
        let geo = &c.extra[n_orb * block..(n_orb + 1) * block];
        let norm: [f64; 9] = std::array::from_fn(|i| lane(geo, LANE_SIGN + i).sqrt());
        let mut moments = [Vec::new(), Vec::new()];
        let mut first_sign = [Vec::new(), Vec::new()];
        let mut outer_weight = [Vec::new(), Vec::new()];
        let mut radial = [Vec::new(), Vec::new()];
        // a bin's value: the f64 sum of its bin_split partials (Neumaier for the ψ sums; the counts
        // are exact small integers in f32)
        let bin_lane = |blk: &[f32], kk: usize, l: usize| -> f64 {
            neumaier((0..bin_split).map(|s| blk[MOMENT_LANES * mom_wg + BIN_LANES * (kk * bin_split + s) + l] as f64))
        };
        let counts: Vec<u32> = (0..n_bins).map(|kk| bin_lane(geo, kk, LANE_COUNT) as u32).collect();
        for slot in 0..n_orb {
            let ch = usize::from(slot >= n_ch[0]);
            let blk = &c.extra[slot * block..(slot + 1) * block];
            let hl = |lm: usize| grid.h.powi((lm as f64).sqrt() as i32);
            moments[ch].push(Moments {
                b: std::array::from_fn(|lm| lane(blk, lm) * hl(lm)),
                c: std::array::from_fn(|lm| lane(blk, LANE_C + lm)),
            });
            first_sign[ch].push(std::array::from_fn(|i| {
                if nuclei.is_empty() || norm[i] <= 0.0 { 0.0 } else { lane(blk, LANE_SIGN + i) / norm[i] }
            }));
            outer_weight[ch].push(lane(blk, LANE_OUTER));
            if lone {
                let a = (0..n_bins).map(|kk| std::array::from_fn::<f64, N_LM, _>(|lm| bin_lane(blk, kk, lm) * dv)).collect();
                radial[ch].push(RadialProfile { dr: RADIAL_BIN_H * grid.h, a, count: counts.clone() });
            }
        }
        Scalars { overlaps_prev, cross, moments, radial: lone.then_some(radial), first_sign, max_abs, outer_weight, spin }
    }

    /// One display update (spec §4.2; module doc): one scalar readback, the CPU f64 step
    /// (`Tracker::update`), lincombs into the display fields, the product. Submits the GPU writes
    /// and returns without waiting for them. Orbitals are the state's occupied ones (n_computed =
    /// n_occ: unoccupied orbitals arrive with plan Task 4). `nuclei` are the geometry `state` was
    /// solved at; exactly one nucleus is a lone atom.
    pub fn update(&mut self, solver: &GpuSolver, state: &GpuState, nuclei: &[qm::Nucleus], tol_scf: f64, product: Product) -> ViewUpdate {
        let maps = readbacks_on_this_thread();
        let sc = self.read_scalars(solver, state, nuclei);
        let raw: [&[Field]; 2] = [&state.orbitals_up, &state.orbitals_dn];
        let n_occ = [raw[0].len(), raw[1].len()];
        assert_eq!((state.evals_up.len(), state.evals_dn.len()), (n_occ[0], n_occ[1]), "OrbitalView: one eigenvalue per orbital");
        let occ = [vec![1.0; n_occ[0]], vec![1.0; n_occ[1]]];
        let lone = nuclei.len() == 1;
        let input = UpdateInput {
            evals: [&state.evals_up, &state.evals_dn],
            occupations: [&occ[0], &occ[1]],
            overlaps_prev: [sc.overlaps_prev[0].as_ref(), sc.overlaps_prev[1].as_ref()],
            cross: sc.cross.as_ref(),
            moments: lone.then(|| [&sc.moments[0][..], &sc.moments[1][..]]),
            radial: sc.radial.as_ref().map(|r| [&r[0][..], &r[1][..]]),
            first_sign: [&sc.first_sign[0], &sc.first_sign[1]],
            max_abs: [&sc.max_abs[0], &sc.max_abs[1]],
            outer_weight: [&sc.outer_weight[0], &sc.outer_weight[1]],
            n_occ,
            n_computed: n_occ,
            n_core: [0; 4],
            z_sigma: lone.then(|| nuclei[0].z as f64 * state.sigma),
            tol_scf,
        };
        let out = self.tracker.update(&input);

        // the display fields: ψ̃ = Σ_j ψ_j U_jk per rotation block, in shown-row order
        let g = self.g;
        for ch in 0..2 {
            let shown: usize = out.rotations[ch].iter().map(|(r, _)| r.len()).sum();
            while self.prev[ch].len() > shown { let f = self.prev[ch].pop().expect("non-empty"); self.pool.push(f); }
            while self.prev[ch].len() < shown { let f = Self::field(&mut self.pool, &self.ctx, &g); self.prev[ch].push(f); }
        }
        let k = &solver.k;
        let mut b = Batch::new(&self.ctx, "orbital view display");
        let ctx = b.ctx();
        for ((rotations, psis), shown) in out.rotations.iter().zip(raw).zip(&self.prev) {
            for (r, u) in rotations {
                for kk in 0..r.len() {
                    let terms: Vec<(f32, &Field)> = (0..r.len()).filter(|&j| u[(j, kk)] != 0.0).map(|j| (u[(j, kk)] as f32, &psis[r.start + j])).collect();
                    lincomb_in(&mut b, k, &g, &shown[r.start + kk], &terms);
                }
            }
        }
        let scratch = self.scratch.as_ref().expect("read_scalars made it");
        let scale = match product {
            Product::None => 0.0,
            Product::Orbital { spin, track, squared, out: vol } => {
                let ch = spin.index();
                match out.rows.iter().find(|r| r.spin == spin && r.track == track) {
                    None => 0.0,
                    Some(row) => {
                        let f = &self.prev[ch][row.index];
                        let s = out.scale[ch][row.index] as f32;
                        if squared {
                            mul_in(&mut b, k, &g, scratch, f, f);
                            self.copy_to_volume(&mut b, k, scratch, vol);
                            s * s
                        } else {
                            self.copy_to_volume(&mut b, k, f, vol);
                            s
                        }
                    }
                }
            }
            Product::Spin { out: vol } => {
                self.copy_to_volume(&mut b, k, scratch, vol);
                sc.spin.scale() as f32
            }
            Product::Colour { include, physical, out: cv } => {
                let terms: Vec<(&Field, [f32; 4])> = out.rows.iter().filter(|r| include.contains(&r.track)).filter_map(|r| {
                    let ch = r.spin.index();
                    let pk = if physical { sc.spin.rho_max } else { out.scale[ch][r.index].powi(2) };
                    if pk <= 0.0 || r.occupation == 0.0 { return None; }
                    let c = ORBITAL_PALETTE[r.slot as usize % ORBITAL_PALETTE.len()];
                    let w = r.occupation / pk;
                    Some((&self.prev[ch][r.index], [(c[0] as f64 * w) as f32, (c[1] as f64 * w) as f32, (c[2] as f64 * w) as f32, w as f32]))
                }).collect();
                let chunks = terms.len().div_ceil(COLOUR_MAX_TERMS).max(1);
                if chunks > 1 && self.colour_acc.is_none() {
                    self.colour_acc = Some(self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("colour accumulator"), size: (g.interior_len() * 16) as u64,
                        usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
                    }));
                }
                assert_eq!(cv.dims, [g.n as u32; 3], "OrbitalView: colour volume dims {:?} != grid n = {}", cv.dims, g.n);
                let acc = if chunks > 1 { self.colour_acc.as_ref().expect("created above") } else { &self.dummy };
                for ci in 0..chunks {
                    let chunk = &terms[(ci * COLOUR_MAX_TERMS).min(terms.len())..((ci + 1) * COLOUR_MAX_TERMS).min(terms.len())];
                    let mut w = [[0.0f32; 4]; COLOUR_MAX_TERMS];
                    for (slot, (_, t)) in w.iter_mut().zip(chunk) { *slot = *t; }
                    let flags = u32::from(ci == 0) | (u32::from(ci + 1 == chunks) << 1);
                    let params = ColourParams { n: g.n as u32, p: g.p as u32, count: chunk.len() as u32, flags, w };
                    let u = b.uniform(ctx, k, "colour_accumulate", &params);
                    let first = chunk.first().map_or(scratch, |t| t.0);
                    let mut res = vec![u.binding()];
                    // unused slots re-bind the chunk's first orbital (read-only; `count` guards them)
                    res.extend((0..COLOUR_MAX_TERMS).map(|t| chunk.get(t).map_or(first, |c| c.0).buffer.as_entire_binding()));
                    res.push(acc.as_entire_binding());
                    res.push(wgpu::BindingResource::TextureView(&cv.view));
                    let bg = k.bind_group(ctx, "colour_accumulate", &res);
                    b.dispatch(k, "colour_accumulate", &bg, [workgroups_for(g.interior_len()), 1, 1]);
                }
                1.0
            }
        };
        b.submit();
        ViewUpdate { out, spin: sc.spin, scale, readbacks: readbacks_on_this_thread() - maps }
    }

    /// `unpad` + `copy_buffer_to_texture` of `f` into `vol`, recorded into `b`
    /// (`GpuSolver::copy_density_to_volume`'s path, on this view's own row buffer).
    fn copy_to_volume(&self, b: &mut Batch, k: &Kernels, f: &Field, vol: &Volume3D) {
        let n = self.g.n as u32;
        assert_eq!(vol.dims, [n; 3], "OrbitalView: volume dims {:?} != grid n = {n}", vol.dims);
        let ctx = b.ctx();
        unpad_rec(b, ctx, k, &self.g, f, &self.rows, self.row_stride);
        b.encoder().copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo { buffer: &self.rows, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * self.row_stride), rows_per_image: Some(n) } },
            wgpu::TexelCopyTextureInfo { texture: &vol.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::Extent3d { width: n, height: n, depth_or_array_layers: n },
        );
    }
}

#[cfg(test)]
mod tests;
