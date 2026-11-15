//! The Harris display solve (live orbitals Phase A; spec `2026-11-14-live-orbitals-design.md` §4.1–4.2):
//! one **non-self-consistent** Kohn–Sham diagonalisation in the potential of a given input density, for the
//! orbital display only. Nothing here is a DFT answer: no energy, no forces, no SCF loop.
//!
//! # Physics (atomic units: Hartree, Bohr; ħ = mₑ = e = 1)
//! Input: ρ_in,σ on the interior (σ = ↑, ↓; the caller's split — spec §4.1 gives it as ρ_in·n_σ/N),
//! ρ_in = ρ_in↑ + ρ_in↓, and under an ionic setup with a partial core ρ_pc (the generator's frozen-core
//! model, a fixed function of the geometry). Then
//!
//! ```text
//!   ∇²V_H = −4π ρ_in                         (V_H > 0 where ρ_in > 0; multipole Dirichlet ghosts)
//!   v_xc,σ = ∂(n ε_xc^LSDA)/∂n_σ  at  n_σ = ρ_in,σ + ½ρ_pc     (NLCC: ρ_pc in xc only, never in V_H)
//!   v_nuc  = −Σ_A Z_A erf(|r − R_A|/(√2 σ))/|r − R_A|   all-electron (σ = settings.sigma; < 0, attractive)
//!          = V_loc = Σ_A V_LR,A + Σ_ions V_SR,I          under a setup (and V_NL = Σ |β⟩E_KB⟨β| inside H)
//!   V_σ    = v_nuc + V_H[ρ_in] + v_xc,σ                  (one `lincomb`, the SCF's term order)
//!   H_σ    = −½∇² + V_σ (+ V_NL),   H_σ ψ_iσ = ε_iσ ψ_iσ,   ⟨ψ_iσ|ψ_jσ⟩ = h³ Σ ψ_iσ ψ_jσ = δ_ij
//!   f_iσ   = qm::one_shot_occupations(ε_σ, n_σ, rule)   (Σ_i f_iσ = n_σ; Aufbau, or DegenerateShells once)
//!   ρ_out,σ = Σ_i f_iσ ψ_iσ²                              (h³ Σ ρ_out,σ = n_σ)
//! ```
//!
//! — exactly the potential of one SCF step (`scf` module doc: `xc_in`, then `lincomb_in` of
//! [v_nuc, V_H, v_xc,σ] with unit weights, ρ_pc added to each spin's xc input as ½ρ_pc and never to the
//! Poisson source), evaluated once. A converged SCF density is a fixed point of that map, so the Harris
//! solve on ρ_in = ρ_SCF reproduces the SCF's eigenpairs to its tolerance
//! (`harris_on_the_converged_density_reproduces_the_scf`).
//!
//! **The block: the lowest n_occ,σ + m + G states** (spec §4.1; m the unoccupied count asked for, G =
//! `UNOCC_GUARD`, n_occ,σ the levels with f > 0). n_occ,σ is known only after the occupations, and under
//! `DegenerateShells` a shared shell makes it exceed n_σ (pseudo-O triplet ↓: n = 2, f = 1, ⅓, ⅓, ⅓ ⇒
//! n_occ = 4). So each channel starts from k_σ = n_σ + m + G plus the previous call's n_occ − n_σ (when it
//! had the same n_σ), solves (one LOBPCG at the SCF's effective tol, `effective_eig_tol`, and cap,
//! `EIGEN_MAX_ITER`), applies `qm::one_shot_occupations`, and grows the block (`block_need`) when
//! n_occ + m + G > k_σ **or** the occupations found no gap witness (`window_short`: the top computed level
//! within τ_c of ε_F — which can come with a released cluster and n_occ = n_σ, so the count alone would
//! not see it; then one more level, as the SCF's growth does): the new columns are fresh
//! `qm::scf::initial_orbitals` ones (seed 5 ↑ / 6 ↓, the columns past those already held) behind the
//! current Ritz vectors, and it solves again — the converged columns need no further work, but the solve
//! runs until the new columns converge or reach the cap; this is the SCF's witness growth (`qm::scf`
//! "Occupations"), bounded by `MAX_WITNESS_GROWTH` — then re-applies the occupations. A block larger than needed (the shell split since the last call) is trimmed from the
//! top. So every completed solve keeps exactly n_occ + m + G levels: G above the occupied ones and above
//! the m asked for, which hold the `DegenerateShells` gap witness whenever the shell has one within that
//! reach, and the unoccupied record always has m + G vectors, the count `block_edge` reads m from. A
//! short window is left only at the growth bound (the note says so, and is emitted only then). The guess is this state's
//! previous vectors (trimmed, or topped up with fresh columns), else `initial_orbitals`. Occupied and
//! unoccupied come out of the **same** Rayleigh–Ritz, so there is no separate unoccupied step and no
//! `occupied_disagreement` check (spec §4.2: the builder's choice). The f > 0 levels (the lowest by rank
//! under both rules) become the state's orbitals with their f; the rest go through the view's own tail
//! (`orbitals::record_unoccupied`: re-measured ε and residual, sorted, τ-set face density, block edge)
//! into this state's `Unoccupied`. With m = 0 the G guard levels are still solved (the witness) but
//! `HarrisState::unoccupied` offers none (`qm::orbitals`: n_computed = n_occ when no row is asked for).
//!
//! **Three roles, three tolerances** (task 8b perf: at 109³ every refresh ended on the 200-iteration cap
//! polishing the guard levels — box states — for 3.6 s warm; measured at 81³, the column holding the cap
//! was always the **top** one, residual 1e-2–3e-2, every other ≤ 1e-3: the last column of a block
//! converges at a rate set by the gap to the first level outside it, and a large box's states are a near
//! continuum). The stop test (`eigen::lobpcg_split`) holds the lowest k − G columns — the displayed
//! n_occ + m — to the SCF's effective tol, the guard columns below the top to `GUARD_TOL` = τ_c/100 =
//! 1e-3 Ha, and does not test the top column: it is a **buffer** vector (standard block-eigensolver
//! practice), iterated to speed up the others, whose Ritz value is used for nothing that needs it. To keep
//! that true the block always holds one level above the levels that decide the occupations
//! (`decisive_levels`: the smallest b whose lowest b levels give no short window): when the witness would
//! be the top column, the block grows by one. Why `GUARD_TOL` is enough: the witness test (`qm::scf::degenerate_shell_occupations`) compares a guard Ritz value with
//! ε_F + τ_c, τ_c = 0.1 Ha; a normalised vector with residual r has an eigenvalue within r of its Ritz
//! value (|θ − λ| ≤ ‖Hx − θx‖), and a Ritz value is an **upper** bound on its eigenvalue (θ_j ≥ λ_j,
//! Courant–Fischer on the Rayleigh–Ritz subspace), so the error is one-sided: with the levels below it
//! resolved (as for any LOBPCG block: no level is missed below a Ritz value), θ − λ ≤ 1e-3 Ha. A level is
//! then taken as a witness only if its true energy is ≥ ε_F + τ_c − 1e-3, i.e. only a level within 1 % of
//! the threshold can be classified differently than at full convergence — the same knife edge any f32
//! rounding or 0.01 Bohr move already flips, and a display decision. The direction of the error is the
//! safe-for-display one: a guard can make a true in-window level look like a witness (that shell is then
//! drawn with integer occupations a hair early), never create a spurious in-window level or drop an
//! occupied one. A capped tested guard (ending above 1e-3) widens that margin to its residual, which the
//! at-cap note prints. The buffer's Ritz value, likewise an upper bound, enters only `block_edge`'s
//! "does a set reach the last vector" test, where being too high can split a degenerate set at the top
//! (a ≥ 3-fold set straddling the window into both guards would then show its lower members, each held
//! to the full tol by the shown-set rule below) and never joins one. When m > 0 the first level above the occupied ones — the witness that decides — is a
//! displayed level and held to the full tol anyway; the guard tolerance decides only at m = 0 and for a
//! shell reaching past the m rows. The displayed levels keep the
//! full tol. **A displayed τ-set reaching into the guards** (the m-th row degenerate with the next level
//! — H₂ 33³'s box pair at 0.20857 Ha straddles m = 2): `block_edge` shows the whole set, so after each
//! solve the shown count is checked against the tight head and, when it reaches past it, the head is
//! raised to cover the shown rows and the solve continues (warm: only those columns move). Every shown
//! row is then held to the full tol; the set's membership itself is judged on Ritz values whose error is
//! ~r²/δ ≤ (1e-3)²/δ — at τ = 1e-4 a guard can be misjudged as in or out of the set only when its
//! distance δ to the rest of the spectrum is ≲ 1e-2 Ha, and a level then shown is re-solved to tol by
//! this same rule. The raised head is remembered for the next call (`last`).
//!
//! **Block memory.** A released short window needs levels above n_occ + m + G for its witness
//! (`minimal_block`); those are kept as spare guess vectors (not in the record) and the next call starts
//! at that block, so a warm refresh does not re-grow it.
//!
//! **Fermi–Dirac (the approved smearing work) must revisit the split.** Every FD level has f > 0, so
//! "f > 0" would make every computed level occupied (n_occ = k, no unoccupied rows) and the growth rule
//! above would grow to its bound. The smearing work must split occupied from unoccupied by rank (n_occ by
//! a threshold on f or by the electron count), not by f > 0; `one_shot_occupations`' exhaustive match makes
//! it touch this path anyway.
//!
//! **Spin density.** The state's `rho_up`/`rho_dn` are the **output** ρ_out,σ (what `OrbitalView`'s Spin
//! product reads, ρ_out↑ − ρ_out↓), never the input's proportional split; `rho` is their sum. `v_h` is
//! V_H[ρ_in] (the potential the orbitals solved in), `v_nuc` this geometry's v_nuc.
//!
//! # What is and is not touched
//! `GpuSolver::harris` borrows only the solver's stateless machinery through `harris_work` (`scf` module
//! doc, "Display access"): kernels, reducer, `GpuPoisson` and the shared `LobpcgWork` (grown to the
//! largest block). It never reads or writes the SCF's warm state, `converged`, the parked ρ_{t−1}, the field
//! pool or the solver's cached `GpuIonic`. Everything else — ρ_in, V_H[ρ_in], v_nuc, the potentials, the
//! vectors, ρ_out, the ionic terms of the Harris geometry (its own `GpuIonic`) — lives in [`HarrisState`].
//! Why sharing is safe: LOBPCG uses only the first `x.len()` work slots, starts with no P columns and
//! recomputes HX from its guess at entry; `rayleigh_residuals` overwrites the HX/R slots it uses; the
//! Poisson solve rebuilds b, the ghost table and r = b − A·v from its inputs and each V-cycle starts from
//! x ← 0; the reducer's partial buffers are written before every read. None of them carries a value from
//! one call to the next, so a Harris call between two SCF solves leaves the second bitwise what it would
//! have been (`harris_leaves_the_scf_state_bitwise_unchanged`).
//!
//! # Fields and VRAM
//! Per call: ρ_in↑↓, ρ, V_H, v_nuc, v↑↓, exc, v_xc↑↓, ρ_out↑↓ and one scratch (13 fields), + ρ_pc and the
//! two xc inputs ρ_σ + ½ρ_pc under a partial core (3), + Σ_σ k_σ vectors; all owned by the state (its own
//! free list, reused across calls; `clear` drops them), plus the `LobpcgWork` growth 8·(k_max − k_SCF).
//!
//! # Precision
//! Grid fields f32; ε, f, residuals and the occupation rule f64 on the CPU; every reduction is the
//! `Reducer`'s f32 partials finished with Neumaier in f64.
use crate::batch::Batch;
use crate::eigen::{lobpcg_split, GpuHamiltonian, LobpcgWork, Stop as EigenStop};
use crate::grid::{clear_readback_fault, Field, GpuGrid};
use crate::ionic::GpuIonic;
use crate::orbitals::record_unoccupied;
use crate::physics::nuclear_potential_in;
use crate::pointwise::{fill_in, lincomb_in};
use crate::scf::{accumulate_density, take_device_error, xc_in, GpuSolver, GpuState, HarrisWork, EIGEN_MAX_ITER, POISSON_MAX_ITER};
use gpu::GpuContext;
use qm::orbitals::{block_edge, tau, Unoccupied, UNOCC_GUARD};
use qm::scf::{ascending, initial_orbitals, CANDIDATE_TAU, MAX_WITNESS_GROWTH};
use qm::Occupations;
use qm::{one_shot_occupations, QmError, ScfSettings, System};
use std::time::{Duration, Instant};

/// What one Harris solve is asked for: the scene (`system`: nuclei, n↑/n↓, the ionic setup), its settings
/// (σ, the occupation rule, the tolerances the eigensolve and τ-sets follow) and the input density ρ_in↑,
/// ρ_in↓ on the interior in `qm::Grid` layout; `unoccupied` = m, the unoccupied rows wanted per channel.
#[derive(Clone, Copy)]
pub struct HarrisInput<'a> {
    pub system: &'a System,
    pub settings: &'a ScfSettings,
    pub rho_in: [&'a [f64]; 2],
    pub unoccupied: usize,
}

/// The outcome of one Harris solve. `evals`/`occ`: every kept level per channel in two parts, each
/// ascending — the occupied ones (f > 0: the LOBPCG Ritz values), then the unoccupied record's re-measured
/// ε with f = 0 (Rayleigh quotients of the same vectors; the two agree to the f32 noise, so the parts can
/// interleave only at a degenerate HOMO/LUMO pair under `Aufbau`); `n_occ` the
/// levels with f > 0 (the state's orbitals); `notes` the informational lines (a released shell, a short
/// window, an eigensolve at its cap); `iterations`/`at_cap` per channel (summed over the block's growth
/// solves; at the cap in any of them); `solves` the LOBPCG calls per channel (1, plus one per growth of the
/// block; 0 for a skipped channel); `cold` = some non-empty channel had no previous vector to start
/// from (all from `initial_orbitals`); `wall` the call's wall time.
#[derive(Clone, Debug)]
pub struct HarrisSolve {
    pub evals: [Vec<f64>; 2],
    pub occ: [Vec<f64>; 2],
    pub n_occ: [usize; 2],
    pub notes: Vec<String>,
    pub iterations: [usize; 2],
    pub at_cap: [bool; 2],
    pub solves: [usize; 2],
    pub cold: bool,
    pub wall: Duration,
}

/// The Harris display solve's own state (module doc): the last solve's `GpuState` and unoccupied vectors
/// with their record, or — after an abort — the vectors and V_H parked as the next call's guess; its own
/// ionic terms and its own free fields. Built for one solver (grid and device): a call on another grid
/// clears it first; a caller that swaps the device drops it.
pub struct HarrisState {
    grid: Option<qm::Grid>,
    state: Option<GpuState>,
    unocc: [Vec<Field>; 2],
    unocc_info: Option<Unoccupied>,
    /// The next call's guess per channel while no state is held (after an abort), ascending.
    guess: [Vec<Field>; 2],
    /// V_H of the last ρ_in while no state is held: the next Poisson guess.
    v_h: Option<Field>,
    ionic: Option<GpuIonic>,
    free: Vec<Field>,
    /// Levels the last solve kept above its record only as the next guess (`minimal_block`: the gap
    /// witness of a released short window), per channel, ascending.
    spare: [Vec<Field>; 2],
    /// (n_σ, b − (n_σ + m + G), head − (k − G)) of the last completed solve per channel — b its minimal
    /// block (`minimal_block`), so the next call starts at the block it needed, and how far its tight head
    /// was raised over k − G for a shown τ-set — and its m.
    last: [Option<(usize, usize, usize)>; 2],
    m: usize,
}

impl Default for HarrisState {
    fn default() -> Self { HarrisState::new() }
}

impl HarrisState {
    /// Empty: the first call is cold.
    pub fn new() -> HarrisState {
        HarrisState { grid: None, state: None, unocc: [Vec::new(), Vec::new()], unocc_info: None, guess: [Vec::new(), Vec::new()],
                      v_h: None, ionic: None, free: Vec::new(), spare: [Vec::new(), Vec::new()], last: [None; 2], m: 0 }
    }

    /// The last completed solve (`None` before one, after an abort or an error, and after `clear`). Its
    /// orbitals are the levels with f > 0, `rho_up`/`rho_dn` the output density (module doc).
    pub fn state(&self) -> Option<&GpuState> { self.state.as_ref() }

    /// The last completed solve's unoccupied vectors per channel (ascending ε; m + `UNOCC_GUARD` each) and
    /// their record, for `OrbitalView::adopt_unoccupied`; empty and `None` without a state, and when that
    /// solve asked for m = 0 (its guard levels were solved for the witness, not for display).
    pub fn unoccupied(&self) -> ([&[Field]; 2], Option<&Unoccupied>) {
        if self.state.is_none() || self.m == 0 { return ([&[], &[]], None); }
        ([&self.unocc[0], &self.unocc[1]], self.unocc_info.as_ref())
    }

    /// Drops everything (fields, ionic terms): the next call is cold.
    pub fn clear(&mut self) { *self = HarrisState::new(); }

    /// The last solve becomes the next call's starting point: its vectors (occupied, unoccupied, spare;
    /// ascending) the guess, its V_H the Poisson guess, its other fields free. A no-op without a state.
    fn park(&mut self) {
        let Some(st) = self.state.take() else { return };
        let GpuState { orbitals_up, orbitals_dn, rho, rho_up, rho_dn, v_h, v_nuc, .. } = st;
        let [uu, ud] = std::mem::take(&mut self.unocc);
        let [su, sd] = std::mem::take(&mut self.spare);
        for (guess, (o, u, sp)) in self.guess.iter_mut().zip([(orbitals_up, uu, su), (orbitals_dn, ud, sd)]) {
            self.free.append(guess);
            *guess = o.into_iter().chain(u).chain(sp).collect();
        }
        if let Some(old) = self.v_h.replace(v_h) { self.free.push(old); }
        self.free.extend([rho, rho_up, rho_dn, v_nuc]);
        self.unocc_info = None;
    }
}

/// One channel's finished solve: the f > 0 levels (ascending), the unoccupied vectors (the record's order)
/// and the spare levels kept only as the next guess.
struct Channel { orbitals: Vec<Field>, evals: Vec<f64>, occ: Vec<f64>, unocc: Vec<Field>, spare: Vec<Field> }

impl Channel {
    fn empty() -> Channel { Channel { orbitals: Vec::new(), evals: Vec::new(), occ: Vec::new(), unocc: Vec::new(), spare: Vec::new() } }

    /// Every vector, ascending: the next call's guess.
    fn vectors(self) -> Vec<Field> { self.orbitals.into_iter().chain(self.unocc).chain(self.spare).collect() }
}

/// The per-call fields (module doc, "Fields and VRAM"), from the state's free list.
struct Scratch { ru: Field, rd: Field, rho: Field, v_nuc: Field, v_h: Field, exc: Field, xu: Field, xd: Field, vu: Field, vd: Field,
                 pc: Option<Field>, core: Option<(Field, Field)> }

impl Scratch {
    /// (ρ_pc, n↑, n↓) for `xc_in` when an ion carries a partial core.
    fn core(&self) -> Option<(&Field, &Field, &Field)> {
        self.pc.as_ref().zip(self.core.as_ref()).map(|(pc, (a, b))| (pc, a, b))
    }

    /// Gives the potentials' fields back to `free` and returns (ρ, v_nuc, V_H).
    fn finish(self, free: &mut Vec<Field>) -> (Field, Field, Field) {
        let Scratch { ru, rd, rho, v_nuc, v_h, exc, xu, xd, vu, vd, pc, core } = self;
        free.extend([ru, rd, exc, xu, xd, vu, vd]);
        free.extend(pc);
        if let Some((a, b)) = core { free.extend([a, b]); }
        (rho, v_nuc, v_h)
    }
}

fn take(free: &mut Vec<Field>, ctx: &GpuContext, g: &GpuGrid) -> Field {
    free.pop().unwrap_or_else(|| Field::new(ctx, g, "harris field"))
}

/// Tops `x` up to `to` columns with `initial_orbitals(.., to, seed)` past the ones it holds (a no-op when
/// it holds enough).
#[allow(clippy::too_many_arguments)]
fn top_up(x: &mut Vec<Field>, to: usize, free: &mut Vec<Field>, ctx: &GpuContext, g: &GpuGrid, grid: &qm::Grid, nuclei: &[qm::Nucleus], seed: u64) {
    if x.len() >= to { return; }
    for d in initial_orbitals(grid, nuclei, to, seed).into_iter().skip(x.len()) {
        let f = take(free, ctx, g);
        f.upload_interior(ctx, g, &d);
        x.push(f);
    }
}

/// The solver's `LobpcgWork` grown to at least `k` vectors (re-created, as the SCF does; it carries no
/// state between calls).
fn grown<'w>(work: &'w mut Option<LobpcgWork>, work_k: &mut usize, ctx: &GpuContext, g: &GpuGrid, k: usize) -> &'w mut LobpcgWork {
    if work.is_none() || *work_k < k {
        *work = Some(LobpcgWork::new(ctx, g, k));
        *work_k = k;
    }
    work.as_mut().expect("sized above")
}

/// The guard/witness levels' LOBPCG tolerance (module doc, "Two tolerances"): τ_c/100 = 1e-3 Ha, so a
/// guard Ritz value θ is within ‖r‖ ≤ 1e-3 Ha of an eigenvalue — 1 % of the gap the witness test reads.
const GUARD_TOL: f64 = CANDIDATE_TAU / 100.0;

/// The fewest lowest levels of `sorted` (ascending) that decide the occupations of `n` electrons under
/// `rule`: the smallest b > n whose `one_shot_occupations` has no short window (`sorted.len()` when none
/// does). Aufbau: n + 1 (it never reads a level above the n-th; the + 1 is the gap level).
fn decisive_levels(sorted: &[f64], n: usize, spin: u8, rule: Occupations) -> usize {
    let mut b = (n + 1).min(sorted.len());
    while b < sorted.len() && one_shot_occupations(&sorted[..b], n, spin, rule).window_short { b += 1; }
    b
}

/// The block a completed solve keeps (≤ `sorted.len()`): the record's `keep` = n_occ + m + G levels, and
/// at least one buffer level above the decisive ones (`decisive_levels`) — more than `keep` only when a
/// released short window needed levels above it for its witness. The next call starts there.
fn minimal_block(sorted: &[f64], n: usize, spin: u8, rule: Occupations, keep: usize) -> usize {
    keep.max(decisive_levels(sorted, n, spin, rule) + 1).min(sorted.len())
}

/// The levels the block must hold after an occupation pass (module doc): n_occ + m + G, and at least one
/// more than it holds when there was no gap witness (`window_short`), capped at `k_max`. Larger than `len`
/// ⇔ grow and solve again.
fn block_need(n_occ: usize, m: usize, window_short: bool, len: usize, k_max: usize) -> usize {
    let need = n_occ + m + UNOCC_GUARD;
    let need = if window_short { need.max(len + 1) } else { need };
    need.min(k_max)
}

const SPIN: [&str; 2] = ["↑", "↓"];

impl GpuSolver {
    /// One Harris display solve (module doc) of `input` into `hs`, never into this solver's SCF state.
    /// `Ok(Some(_))`: `hs.state()` and `hs.unoccupied()` hold the result. `Ok(None)`: `should_abort` returned
    /// true — polled once before anything is computed and once per LOBPCG iteration — and `hs.state()` is
    /// `None`; its vectors (the current iterate of an interrupted channel) and V_H stay as the next call's
    /// guess. `Err`: a sub-solver's numerics broke (`QmError::Numerical`) or a readback failed
    /// (`QmError::Device`); `hs` then holds no state and no guess (the next call is cold).
    /// Panics if ρ_in does not have one value per grid point, or the ionic setup is not for this grid's h
    /// and `settings.sigma` (as `solve`).
    pub fn harris(&mut self, hs: &mut HarrisState, input: &HarrisInput, should_abort: &mut dyn FnMut() -> bool)
                  -> Result<Option<HarrisSolve>, QmError> {
        let t0 = Instant::now();
        clear_readback_fault(); // a fault latched by an earlier, unrelated readback is not this call's
        let (system, settings, m) = (input.system, input.settings, input.unoccupied);
        let len = self.grid().len();
        assert!(input.rho_in.iter().all(|r| r.len() == len), "HarrisInput: ρ_in has {} / {} values for a grid of {len}",
                input.rho_in[0].len(), input.rho_in[1].len());
        if let Some(s) = system.ionic.as_deref() {
            assert!(s.sigma_ae == settings.sigma, "IonicSetup σ_AE {} ≠ settings σ {}", s.sigma_ae, settings.sigma);
            assert!(s.h == self.grid().h, "IonicSetup h {} ≠ grid h {}", s.h, self.grid().h);
        }
        if hs.grid.as_ref() != Some(self.grid()) {
            hs.clear();
            hs.grid = Some(self.grid().clone());
        }
        hs.park();
        if should_abort() { return Ok(None); }
        let n = [system.n_up, system.n_dn];
        let tol = self.effective_eig_tol(settings);
        let tau_scf = tau(settings.tol);
        let HarrisWork { ctx, k, r, g, grid, poisson, poisson_tol, work, work_k } = self.harris_work();
        let HarrisState { state: slot, unocc: slot_unocc, unocc_info: slot_info, guess, v_h: v_h_prev, ionic: ion_cache, free,
                          spare: slot_spare, last, m: slot_m, .. } = hs;

        // --- ρ_in,σ (uploaded before anything is recorded) and this geometry's own ionic terms ---
        let (ru, rd) = (take(free, ctx, g), take(free, ctx, g));
        ru.upload_interior(ctx, g, input.rho_in[0]);
        rd.upload_interior(ctx, g, input.rho_in[1]);
        let gion: Option<&GpuIonic> = match system.ionic.as_deref() {
            None => None,
            Some(s) => {
                match ion_cache.as_mut() { Some(ion) => ion.update(ctx, g, &system.nuclei, s), None => *ion_cache = Some(GpuIonic::new(ctx, g, &system.nuclei, s)) }
                ion_cache.as_ref()
            }
        };
        let mut b = Batch::new(ctx, "harris potential");
        // v_nuc, ρ_pc (xc only) and ρ = ρ↑ + ρ↓ (the Poisson source: never ρ_pc), as `run` records them
        let v_nuc = take(free, ctx, g);
        match gion {
            None => nuclear_potential_in(&mut b, k, g, &v_nuc, &system.nuclei, settings.sigma),
            Some(ion) => ion.local_potential_in(&mut b, k, g, &v_nuc),
        }
        let pc = gion.filter(|i| i.has_partial_core()).map(|ion| { let f = take(free, ctx, g); ion.partial_core_in(&mut b, k, g, &f); f });
        let core = pc.as_ref().map(|_| (take(free, ctx, g), take(free, ctx, g)));
        let rho = take(free, ctx, g);
        lincomb_in(&mut b, k, g, &rho, &[(1.0, &ru), (1.0, &rd)]);
        let v_h = match v_h_prev.take() {
            Some(f) => f,
            None => { let f = take(free, ctx, g); fill_in(&mut b, k, g, &f, 0.0); f } // cold CG start
        };
        let [exc, xu, xd, vu, vd] = std::array::from_fn(|_| take(free, ctx, g));
        let p = Scratch { ru, rd, rho, v_nuc, v_h, exc, xu, xd, vu, vd, pc, core };
        b.submit();

        // an error: every field back to the free list, no state and no guess (the next call is cold)
        macro_rules! fail {
            ($err:expr, $p:expr) => {{
                let err: QmError = take_device_error().unwrap_or($err);
                let (rho, v_nuc, v_h) = $p.finish(free);
                free.extend([rho, v_nuc, v_h]);
                for gs in guess.iter_mut() { free.append(gs); }
                return Err(err);
            }};
        }

        // --- V_H[ρ_in], warm from the previous Harris V_H; then V_σ = v_nuc + V_H + v_xc,σ (the SCF's term order) ---
        if let Err(e) = poisson.solve(ctx, k, g, r, &p.rho, &p.v_h, poisson_tol, POISSON_MAX_ITER) { fail!(e, p); }
        if let Some(e) = take_device_error() { fail!(e, p); }
        let mut b = Batch::new(ctx, "harris xc");
        xc_in(&mut b, k, g, &p.ru, &p.rd, p.core(), &p.exc, &p.xu, &p.xd);
        lincomb_in(&mut b, k, g, &p.vu, &[(1.0, &p.v_nuc), (1.0, &p.v_h), (1.0, &p.xu)]);
        lincomb_in(&mut b, k, g, &p.vd, &[(1.0, &p.v_nuc), (1.0, &p.v_h), (1.0, &p.xd)]);
        b.submit();

        // --- one eigensolve per channel, occupied and unoccupied together ---
        let nl = gion.and_then(|i| i.nonlocal());
        let mut info = Unoccupied::empty(tol);
        let mut done: [Option<Channel>; 2] = [None, None];
        let (mut cold, mut notes) = (false, Vec::new());
        let mut solves = [0usize; 2];
        let mut blocks = [0usize; 2]; // the minimal block per channel (`minimal_block`)
        let mut heads = [0usize; 2]; // the tight head's raise over block − G per channel
        // Ok(true): aborted
        let mut outcome: Result<bool, QmError> = Ok(false);
        let sum = take(free, ctx, g);
        'channels: for ch in 0..2 {
            let nc = n[ch];
            if nc + m == 0 {
                free.append(&mut guess[ch]);
                done[ch] = Some(Channel::empty());
                continue;
            }
            let seed = 5 + ch as u64;
            // the block (module doc): n_σ + m + G, plus what the last solve's minimal block needed beyond it
            let (extra, head_extra) = match last[ch] { Some((ln, e, he)) if ln == nc => (e.min(MAX_WITNESS_GROWTH), he), _ => (0, 0) };
            let k_max = nc + m + UNOCC_GUARD + MAX_WITNESS_GROWTH;
            let mut x = std::mem::take(&mut guess[ch]);
            if x.is_empty() { cold = true; }
            let k0 = nc + extra + m + UNOCC_GUARD;
            if x.len() > k0 { free.extend(x.drain(k0..)); } // ascending: the highest go
            top_up(&mut x, k0, free, ctx, g, grid, &system.nuclei, seed);
            // the columns held to the full tol: the displayed n_occ + m (all but the top G), raised over a
            // shown τ-set reaching into the guards (module doc, "Two tolerances")
            let mut head = (k0 - UNOCC_GUARD + head_extra).min(k0);
            let h = GpuHamiltonian { v: if ch == 0 { &p.vu } else { &p.vd }, nl };
            let (ev, one) = loop {
                let mut ev = vec![0.0; x.len()];
                let w = grown(work, work_k, ctx, g, x.len());
                solves[ch] += 1;
                // the displayed levels (all but the top G) at the SCF's tol, the guard/witness levels at
                // `GUARD_TOL` (module doc, "Two tolerances")
                // the top column is the buffer (module doc), the guards below it at GUARD_TOL
                let checked = (x.len() - 1).max(head);
                let es = match lobpcg_split(ctx, k, g, r, &h, &mut x, &mut ev, w, tol, head, GUARD_TOL.max(tol), checked, EIGEN_MAX_ITER, Some(&mut *should_abort)) {
                    Ok(es) => es,
                    Err(e) => { free.append(&mut x); outcome = Err(take_device_error().unwrap_or(e)); break 'channels; }
                };
                if let Some(e) = take_device_error() { free.append(&mut x); outcome = Err(e); break 'channels; }
                if es.stop == EigenStop::Aborted {
                    guess[ch] = x; // the current iterate: the next call's guess
                    outcome = Ok(true);
                    break 'channels;
                }
                info.iterations[ch] += es.iterations;
                if es.stop == EigenStop::MaxIter && !info.at_cap[ch] {
                    info.at_cap[ch] = true;
                    notes.push(format!("spin {}: the eigensolve ended at its {EIGEN_MAX_ITER}-iteration cap (max residual {:.1e} > tol {tol:.1e})", SPIN[ch], es.max_residual));
                }
                // the occupations of this one diagonalisation; the f > 0 levels are the lowest by rank
                let one = one_shot_occupations(&ev, nc, ch as u8, settings.occupations);
                let n_occ = one.f.iter().filter(|f| **f > 0.0).count();
                let mut sorted = ev.clone();
                sorted.sort_by(f64::total_cmp);
                // one buffer level above the levels that decide the occupations (module doc)
                let decisive = decisive_levels(&sorted, nc, ch as u8, settings.occupations);
                let need = block_need(n_occ, m, one.window_short, x.len(), k_max).max((decisive + 1).min(k_max));
                if need > x.len() {
                    // a shared shell reached into the block, or no gap witness: grow it behind the Ritz vectors
                    top_up(&mut x, need, free, ctx, g, grid, &system.nuclei, seed);
                    head = head.max(x.len() - UNOCC_GUARD);
                    continue;
                }
                // a shown τ-set reaching into the guard columns: hold it to the full tol as well
                if m > 0 {
                    let keep = (n_occ + m + UNOCC_GUARD).min(sorted.len());
                    let (shown, _) = block_edge(&sorted[..keep], n_occ, keep, tau_scf);
                    if shown > head {
                        head = shown;
                        continue;
                    }
                }
                break (ev, one);
            };
            if one.released { notes.push(format!("spin {}: a shared shell of spread ≥ τ was split (one-shot release)", SPIN[ch])); }
            if one.window_short {
                debug_assert!(x.len() >= k_max, "a short window below the growth bound grows the block");
                notes.push(format!("spin {}: no gap witness among the {} computed levels (growth bound reached): every level within τ_c of ε_F shares", SPIN[ch], x.len()));
            }
            let order = ascending(&ev);
            let n_occ = order.iter().take_while(|&&i| one.f[i] > 0.0).count();
            assert!(order[n_occ..].iter().all(|&i| one.f[i] == 0.0), "the occupied levels are the lowest ({:?})", one.f);
            let keep = (n_occ + m + UNOCC_GUARD).min(order.len());
            // the levels the rule needed (`minimal_block`): the record's, and above them a released short
            // window's witness, kept as spare guess vectors; a block larger than that loses its top levels
            let sorted: Vec<f64> = order.iter().map(|&i| ev[i]).collect();
            let block = minimal_block(&sorted, nc, ch as u8, settings.occupations, keep);
            blocks[ch] = block;
            heads[ch] = head.saturating_sub(block.saturating_sub(UNOCC_GUARD));
            let mut slots: Vec<Option<Field>> = x.into_iter().map(Some).collect();
            let mut pick = |idx: &[usize]| -> Vec<Field> { idx.iter().map(|&i| slots[i].take().expect("each index once")).collect() };
            let orbitals = pick(&order[..n_occ]);
            let phis = pick(&order[n_occ..keep]);
            let spare = pick(&order[keep..block]);
            free.extend(pick(&order[block..]));
            let evals: Vec<f64> = order[..n_occ].iter().map(|&i| ev[i]).collect();
            let occ: Vec<f64> = order[..n_occ].iter().map(|&i| one.f[i]).collect();
            let w = grown(work, work_k, ctx, g, phis.len().max(1));
            match record_unoccupied(ctx, k, g, r, &h, w, phis, &evals, tol, tau_scf, &sum, &mut info, ch) {
                Ok(unocc) => done[ch] = Some(Channel { orbitals, evals, occ, unocc, spare }),
                Err((e, phis)) => {
                    free.extend(orbitals.into_iter().chain(phis).chain(spare));
                    outcome = Err(take_device_error().unwrap_or(e));
                    break;
                }
            }
            if let Some(e) = take_device_error() { outcome = Err(e); break; }
        }
        free.push(sum);
        match outcome {
            Err(e) => {
                for c in done.into_iter().flatten() { free.extend(c.vectors()); }
                fail!(e, p);
            }
            Ok(true) => {
                // the finished channels' vectors and this V_H[ρ_in] are the next call's guess
                for (gs, c) in guess.iter_mut().zip(done) {
                    if let Some(c) = c { *gs = c.vectors(); }
                }
                let (rho, v_nuc, v_h) = p.finish(free);
                free.extend([rho, v_nuc]);
                *v_h_prev = Some(v_h);
                return Ok(None);
            }
            Ok(false) => {}
        }

        // --- ρ_out,σ = Σ f ψ² (the Spin product's input), ρ their sum; the state ---
        let [up, dn] = done.map(|c| c.expect("both channels finished"));
        let (ou, od) = (take(free, ctx, g), take(free, ctx, g));
        let (rho, v_nuc, v_h) = p.finish(free);
        let mut b = Batch::new(ctx, "harris density");
        accumulate_density(&mut b, k, g, &ou, &up.orbitals, Some(&up.occ));
        accumulate_density(&mut b, k, g, &od, &dn.orbitals, Some(&dn.occ));
        lincomb_in(&mut b, k, g, &rho, &[(1.0, &ou), (1.0, &od)]);
        b.submit();
        let level = |c: &Channel, ch: usize| -> (Vec<f64>, Vec<f64>) {
            (c.evals.iter().chain(&info.evals[ch]).copied().collect(), c.occ.iter().copied().chain(info.evals[ch].iter().map(|_| 0.0)).collect())
        };
        let ((eu, fu), (ed, fd)) = (level(&up, 0), level(&dn, 1));
        let solve = HarrisSolve { evals: [eu, ed], occ: [fu, fd], n_occ: [up.orbitals.len(), dn.orbitals.len()], notes,
                                  iterations: info.iterations, at_cap: info.at_cap, solves, cold, wall: t0.elapsed() };
        let Channel { orbitals: orbitals_up, evals: evals_up, occ: occ_up, unocc: unocc_up, spare: spare_up } = up;
        let Channel { orbitals: orbitals_dn, evals: evals_dn, occ: occ_dn, unocc: unocc_dn, spare: spare_dn } = dn;
        *slot = Some(GpuState { grid: grid.clone(), sigma: settings.sigma, orbitals_up, orbitals_dn, evals_up, evals_dn,
                                rho, rho_up: ou, rho_dn: od, v_h, v_nuc, ionic: system.ionic.clone(), occ_up, occ_dn,
                                occupations: settings.occupations });
        *slot_unocc = [unocc_up, unocc_dn];
        *slot_info = Some(info);
        *last = [(n[0] + m > 0).then_some((n[0], blocks[0].saturating_sub(n[0] + m + UNOCC_GUARD), heads[0])),
                 (n[1] + m > 0).then_some((n[1], blocks[1].saturating_sub(n[1] + m + UNOCC_GUARD), heads[1]))];
        *slot_spare = [spare_up, spare_dn];
        *slot_m = m;
        Ok(Some(solve))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orbitals::{OrbitalView, Product, Spin};
    use crate::reduce::neumaier;
    use qm::{Grid, KsState, Nucleus, Occupations};

    /// The superposed free-atom 1s densities e^{−2r}/π of H atoms at `pos`, renormalised on the grid to
    /// one electron each, split equally between the spins (a cheap stand-in for the app's ρ_in).
    fn superposed_h(grid: &Grid, pos: &[[f64; 3]]) -> [Vec<f64>; 2] {
        let one = |c: [f64; 3]| {
            let v = grid.sample(|p| { let r = ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2)).sqrt(); (-2.0 * r).exp() / std::f64::consts::PI });
            let q = neumaier(v.iter().copied()) * grid.dv();
            v.into_iter().map(|x| x / q).collect::<Vec<f64>>()
        };
        let mut total = vec![0.0; grid.len()];
        for c in pos { for (t, v) in total.iter_mut().zip(one(*c)) { *t += v; } }
        let half: Vec<f64> = total.iter().map(|t| 0.5 * t).collect();
        [half.clone(), half]
    }

    /// Diagnostic (task 8b perf): the wall of a warm Harris refresh after a 0.1 Bohr move, H₂ (R = 1.4) on
    /// n³ × h (env `HARRIS_DIAG_N`, default 81; `HARRIS_DIAG_H`, default 0.15 — the app's Accurate grid),
    /// m = 4 (the app's default), ρ_in the superposed free-atom densities. Prints the cold call and three
    /// warm moves (wall, iterations, solves, at cap).
    #[test]
    #[ignore = "diagnostic; run: HARRIS_DIAG_N=81 cargo test -p qm-gpu --release diag_harris_warm_refresh -- --ignored --nocapture"]
    fn diag_harris_warm_refresh_wall() {
        let Some(ctx) = ctx() else { return };
        let n: usize = std::env::var("HARRIS_DIAG_N").ok().and_then(|v| v.parse().ok()).unwrap_or(81);
        let h: f64 = std::env::var("HARRIS_DIAG_H").ok().and_then(|v| v.parse().ok()).unwrap_or(0.15);
        let grid = Grid::cube(n, h);
        let settings = ScfSettings::for_grid(&grid);
        let mut s = GpuSolver::new(ctx, &grid);
        let mut hs = HarrisState::new();
        for step in 0..4 {
            let dx = 0.1 * step as f64;
            let pos = [[0.7 + dx, 0.0, 0.0], [-0.7, 0.0, 0.0]];
            let sys = System::new(pos.iter().map(|p| Nucleus { pos: *p, z: 1 }).collect(), 0, None);
            let rho = superposed_h(&grid, &pos);
            let input = HarrisInput { system: &sys, settings: &settings, rho_in: [&rho[0], &rho[1]], unoccupied: 4 };
            let out = s.harris(&mut hs, &input, &mut || false).expect("no failure").expect("not aborted");
            println!("H₂ {n}³ × {h} m = 4, move {dx:.1}: {} wall {:.3} s, iterations {:?}, solves {:?}, at cap {:?}, ε↑ {:.5?}",
                     if out.cold { "cold" } else { "warm" }, out.wall.as_secs_f64(), out.iterations, out.solves, out.at_cap, out.evals[0]);
            if let (_, Some(u)) = hs.unoccupied() {
                for ch in 0..2 { println!("    {}: unoccupied ε {:.5?} residuals {:?} (tol {:.1e}); notes {:?}", SPIN[ch], u.evals[ch], u.residuals[ch].iter().map(|r| format!("{r:.1e}")).collect::<Vec<_>>(), u.tol, out.notes); }
            }
        }
    }

    /// The growth trigger (review round 2): the qm unit case of a released short window
    /// (`one_shot_marks_a_short_window`: ε = −1.0, −0.40, −0.39, −0.35, n = 2 ⇒ released, f = 1, 1, 0, 0,
    /// n_occ = n = 2, `window_short`) needs one more level although n_occ + m + G = 4 = the block at m = 0; a
    /// witnessed block of the right size needs none; a shared shell needs n_occ + m + G; the bound caps both.
    #[test]
    fn block_need_grows_a_short_window() {
        let o = one_shot_occupations(&[-1.0, -0.40, -0.39, -0.35], 2, 0, Occupations::DegenerateShells { tau: qm::DEGENERACY_TAU });
        let n_occ = o.f.iter().filter(|f| **f > 0.0).count();
        assert!(o.window_short && o.released && n_occ == 2, "{o:?}");
        let k_max = 2 + UNOCC_GUARD + MAX_WITNESS_GROWTH;
        assert_eq!(block_need(n_occ, 0, o.window_short, 4, k_max), 5, "a short window grows by one level");
        assert_eq!(block_need(n_occ, 0, false, 4, k_max), 4, "a witnessed block of n_occ + G levels stays");
        assert_eq!(block_need(4, 2, false, 6, k_max), 8, "a shared shell: n_occ + m + G");
        assert_eq!(block_need(n_occ, 0, true, k_max, k_max), k_max, "at the bound: no further growth");
    }

    /// Block memory (task 8b, the deferred round-2 minor): the released short window of the qm unit case
    /// (ε = −1.0, −0.40, −0.39, −0.35, then 0.2; n = 2, m = 0 ⇒ n_occ + m + G = 4) needs the 5th level for its
    /// witness, so the minimal block is 5 — the next call starts there instead of re-growing; a witnessed
    /// block, Aufbau and a block without the 5th level stay at what they have.
    #[test]
    fn minimal_block_keeps_a_released_short_windows_witness() {
        let ds = Occupations::DegenerateShells { tau: qm::DEGENERACY_TAU };
        let levels = [-1.0, -0.40, -0.39, -0.35, 0.2];
        assert_eq!(minimal_block(&levels, 2, 0, ds, 4), 5, "the witness 0.2 is needed");
        assert_eq!(minimal_block(&levels[..4], 2, 0, ds, 4), 4, "nothing above: the block is what was computed");
        assert_eq!(minimal_block(&[-1.0, -0.40, 0.1, 0.2, 0.3], 2, 0, ds, 4), 4, "witnessed within n_occ + G");
        assert_eq!(minimal_block(&levels, 2, 0, Occupations::Aufbau, 4), 4, "Aufbau needs no witness");
        // the decisive levels: the witness 0.2 is the 5th, so a block of 5 would make it the (untested) top
        // buffer column — the block needs 6
        assert_eq!(decisive_levels(&levels, 2, 0, ds), 5);
        assert_eq!(decisive_levels(&[-1.0, -0.40, 0.1, 0.2, 0.3], 2, 0, ds), 3, "the witness 0.1 is the 3rd level");
        assert_eq!(decisive_levels(&levels, 2, 0, Occupations::Aufbau), 3, "Aufbau: n + 1");
    }

    fn ctx() -> Option<GpuContext> {
        let c = GpuContext::new_headless();
        if c.is_none() { eprintln!("no GPU adapter: skipping"); }
        c
    }

    fn h2() -> System {
        System::new(vec![Nucleus { pos: [0.7, 0.0, 0.0], z: 1 }, Nucleus { pos: [-0.7, 0.0, 0.0], z: 1 }], 0, None)
    }

    /// A converged GPU SCF of H₂ (R = 1.4) on 33³ × 0.25 (σ = h, tol 1e-6) and its state on the CPU.
    fn solved_h2(ctx: &GpuContext) -> (GpuSolver, ScfSettings, KsState) {
        let grid = Grid::cube(33, 0.25);
        let settings = ScfSettings::for_grid(&grid);
        let mut s = GpuSolver::new(ctx.clone(), &grid);
        s.solve(&h2(), &settings, false).expect("the SCF converges");
        let ks = s.ks_state(s.state().unwrap());
        (s, settings, ks)
    }

    /// Every number of a CPU copy of a state, as bits (the bitwise comparisons).
    fn bits(ks: &KsState) -> Vec<u64> {
        let fields = ks.orbitals_up.iter().chain(&ks.orbitals_dn).chain([&ks.rho_up, &ks.rho_dn, &ks.v_nuc, &ks.v_h]);
        fields.flatten().chain(&ks.evals_up).chain(&ks.evals_dn).chain(&ks.occ_up).chain(&ks.occ_dn).map(|v| v.to_bits()).collect()
    }

    fn charge(grid: &Grid, rho: &[f64]) -> f64 { neumaier(rho.iter().copied()) * grid.dv() }

    /// The SCF's fixed point (module doc): Harris on ρ_in = ρ_SCF solves in the SCF's own last potential
    /// (to its Δρ < 1e-6), so every occupied ε agrees to 1e-5 Ha and the occupied subspaces coincide,
    /// (1/n_σ)‖⟨ψ_H|ψ_SCF⟩‖_F² ≥ 1 − 1e-6. The state's ρ_up/ρ_dn are the output Σ f ψ² — checked against the
    /// state's own orbitals on the CPU and, with an input split unlike the output's (ρ_in↑ = 0.6ρ,
    /// ρ_in↓ = 0.4ρ: 1.2 and 0.8 electrons), by the charges h³Σρ_out,σ = n_σ = 1, never the input's.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm-gpu --release")]
    fn harris_on_the_converged_density_reproduces_the_scf() {
        let Some(ctx) = ctx() else { return };
        let (mut s, settings, scf) = solved_h2(&ctx);
        let grid = s.grid().clone();
        let sys = h2();
        let mut hs = HarrisState::new();
        let input = HarrisInput { system: &sys, settings: &settings, rho_in: [&scf.rho_up, &scf.rho_dn], unoccupied: 0 };
        let out = s.harris(&mut hs, &input, &mut || false).expect("no failure").expect("not aborted");
        println!("Harris on ρ_SCF: {out:?}");
        assert!(out.cold, "the first call starts from initial_orbitals");
        let hk = s.ks_state(hs.state().expect("a state"));
        for (ch, (h_ev, s_ev, h_psi, s_psi, n)) in [(&hk.evals_up, &scf.evals_up, &hk.orbitals_up, &scf.orbitals_up, sys.n_up),
                                                     (&hk.evals_dn, &scf.evals_dn, &hk.orbitals_dn, &scf.orbitals_dn, sys.n_dn)].into_iter().enumerate() {
            assert_eq!((h_ev.len(), out.n_occ[ch]), (n, n), "spin {ch}: the f > 0 levels are the n_σ occupied ones");
            for i in 0..n {
                let d = (h_ev[i] - s_ev[i]).abs();
                println!("  spin {ch} #{i}: ε_H {:+.8} ε_SCF {:+.8} |Δ| {d:.2e}", h_ev[i], s_ev[i]);
                assert!(d <= 1e-5, "spin {ch} #{i}: |ε_H − ε_SCF| = {d:.3e} > 1e-5 Ha");
            }
            let gr = &grid;
            let f2 = neumaier(h_psi.iter().flat_map(|a| s_psi.iter().map(move |b| gr.dot(a, b).powi(2)))) / n as f64;
            println!("  spin {ch}: (1/n)‖⟨ψ_H|ψ_SCF⟩‖_F² = 1 − {:.2e}", 1.0 - f2);
            assert!(f2 >= 1.0 - 1e-6, "spin {ch}: the occupied subspaces differ, (1/n)‖S‖_F² = {f2}");
        }
        // ρ_out,σ = Σ f ψ² of the state's own orbitals (f32 rounding of the accumulation only)
        for (rho, psis, f, n) in [(&hk.rho_up, &hk.orbitals_up, &hk.occ_up, sys.n_up), (&hk.rho_dn, &hk.orbitals_dn, &hk.occ_dn, sys.n_dn)] {
            let peak = rho.iter().fold(0.0f64, |m, v| m.max(*v));
            let worst = (0..grid.len()).map(|p| (rho[p] - psis.iter().zip(f).map(|(psi, fi)| fi * psi[p] * psi[p]).sum::<f64>()).abs()).fold(0.0, f64::max);
            let q = charge(&grid, rho);
            println!("  ρ_out: h³Σρ = {q:.9} (n = {n}), max |ρ − Σfψ²| = {worst:.2e} (peak {peak:.3e})");
            assert!(worst <= 1e-6 * peak, "the state's density is not Σ f ψ² of its orbitals ({worst:.2e})");
            assert!((q - n as f64).abs() <= 1e-5, "h³Σρ_out = {q}, n = {n}");
        }
        // an input split unlike the output's: the output still carries n_σ, never the input's charges
        let total: Vec<f64> = scf.rho_up.iter().zip(&scf.rho_dn).map(|(a, b)| a + b).collect();
        let (su, sd): (Vec<f64>, Vec<f64>) = total.iter().map(|r| (0.6 * r, 0.4 * r)).unzip();
        let skewed = HarrisInput { rho_in: [&su, &sd], ..input };
        s.harris(&mut hs, &skewed, &mut || false).expect("no failure").expect("not aborted");
        let hk = s.ks_state(hs.state().unwrap());
        let (qu, qd) = (charge(&grid, &hk.rho_up), charge(&grid, &hk.rho_dn));
        println!("  skewed input (1.2 / 0.8 e): ρ_out charges {qu:.9} / {qd:.9}");
        assert!((qu - 1.0).abs() <= 1e-5 && (qd - 1.0).abs() <= 1e-5, "ρ_out,σ must carry n_σ = 1, got {qu} / {qd}");
    }

    /// Two Harris calls between SCF solves (m = 2: the shared `LobpcgWork` grows to 5 vectors) leave the
    /// SCF's state bit for bit as it was, and the next warm solve takes the same steps to the same energy
    /// bits as on a twin solver that never ran one.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm-gpu --release")]
    fn harris_leaves_the_scf_state_bitwise_unchanged() {
        let Some(ctx) = ctx() else { return };
        let (mut a, settings, scf) = solved_h2(&ctx);
        let (mut b, _, _) = solved_h2(&ctx);
        let before = bits(&a.ks_state(a.state().unwrap()));
        let moved = System::new(vec![Nucleus { pos: [0.75, 0.02, 0.0], z: 1 }, Nucleus { pos: [-0.7, 0.0, 0.0], z: 1 }], 0, None);
        let mut hs = HarrisState::new();
        let input = HarrisInput { system: &moved, settings: &settings, rho_in: [&scf.rho_up, &scf.rho_dn], unoccupied: 2 };
        for call in 0..2 {
            let out = a.harris(&mut hs, &input, &mut || false).expect("no failure").expect("not aborted");
            println!("Harris call {call}: cold {}, iterations {:?}, at cap {:?}, ε {:?}", out.cold, out.iterations, out.at_cap, out.evals);
        }
        assert!(a.work_capacity() >= 1 + 2 + UNOCC_GUARD, "the shared work space grew to the Harris block");
        assert!(a.state().is_some(), "the SCF's converged state is still exposed");
        assert!(bits(&a.ks_state(a.state().unwrap())) == before, "a Harris call wrote the SCF's state");
        let next = System::new(vec![Nucleus { pos: [0.72, 0.0, 0.0], z: 1 }, Nucleus { pos: [-0.7, 0.0, 0.0], z: 1 }], 0, None);
        let ea = a.solve(&next, &settings, true).unwrap().1;
        let eb = b.solve(&next, &settings, true).unwrap().1;
        println!("warm solve: touched {} steps E {:+.12}, twin {} steps E {:+.12}", ea.iterations, ea.energy.total, eb.iterations, eb.energy.total);
        assert_eq!(ea.iterations, eb.iterations, "the same SCF steps");
        assert_eq!(ea.energy.total.to_bits(), eb.energy.total.to_bits(), "warm solve after Harris: {} vs {}", ea.energy.total, eb.energy.total);
    }

    /// Pseudo-O (`IonicPolicy::All`, triplet, `DegenerateShells`) at the origin, 49³: a Harris solve with the
    /// atom moved 0.3 Bohr builds its own `GpuIonic` for the moved geometry; the solver's cache (what
    /// `display_work` lends for the SCF state's own display) still holds the SCF's nuclei.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm-gpu --release")]
    fn harris_ionic_cache_is_its_own() {
        let Some(ctx) = ctx() else { return };
        use crate::ionic::tests::filtered_at;
        use qm::ionic::{Ionic, IonicPolicy, IonicSetup};
        let h = 0.25;
        let grid = Grid::cube(49, h);
        let setup = std::sync::Arc::new(IonicSetup::from_tables(IonicPolicy::All, h, h, std::collections::BTreeMap::from([(8, Ionic::Pseudo(filtered_at(8, h)))])));
        let at = System::with_ionic(vec![Nucleus { pos: [0.0; 3], z: 8 }], 0, Some(3), setup.clone());
        let moved = System::with_ionic(vec![Nucleus { pos: [0.3, 0.0, 0.0], z: 8 }], 0, Some(3), setup);
        let mut settings = ScfSettings::for_grid(&grid);
        settings.occupations = Occupations::DegenerateShells { tau: qm::DEGENERACY_TAU };
        let mut s = GpuSolver::new(ctx, &grid);
        s.solve(&at, &settings, false).expect("the SCF converges");
        let scf = s.ks_state(s.state().unwrap());
        let mut hs = HarrisState::new();
        let input = HarrisInput { system: &moved, settings: &settings, rho_in: [&scf.rho_up, &scf.rho_dn], unoccupied: 0 };
        let out = s.harris(&mut hs, &input, &mut || false).expect("no failure").expect("not aborted");
        println!("pseudo-O moved 0.3: ε {:?}, f {:?}, notes {:?}; SCF ε↑ {:?} f↑ {:?}", out.evals, out.occ, out.notes, scf.evals_up, scf.occ_up);
        for ch in 0..2 { assert!((neumaier(out.occ[ch].iter().copied()) - [at.n_up, at.n_dn][ch] as f64).abs() < 1e-12, "Σf = n_σ"); }
        assert_eq!(hs.ionic.as_ref().expect("its own ionic terms").nuclei(), &moved.nuclei[..], "Harris's cache is the moved geometry");
        assert!(hs.state().unwrap().ionic.is_some(), "the Harris state records the setup");
        let dw = s.display_work(1).expect("the SCF state is still converged");
        let ion = dw.ionic.expect("the SCF state's ionic terms");
        assert_eq!(ion.nuclei(), &at.nuclei[..], "the solver's cache was rebuilt by a Harris call");
    }

    /// An abort on the first poll: `Ok(None)`, no state, nothing to adopt; the next call starts warm from the
    /// vectors parked before the abort.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm-gpu --release")]
    fn harris_abort_leaves_no_state() {
        let Some(ctx) = ctx() else { return };
        let (mut s, settings, scf) = solved_h2(&ctx);
        let sys = h2();
        let mut hs = HarrisState::new();
        let input = HarrisInput { system: &sys, settings: &settings, rho_in: [&scf.rho_up, &scf.rho_dn], unoccupied: 1 };
        let first = s.harris(&mut hs, &input, &mut || false).expect("no failure").expect("not aborted");
        assert!(first.cold && hs.state().is_some());
        let mut polls = 0;
        let aborted = s.harris(&mut hs, &input, &mut || { polls += 1; true }).expect("no failure");
        assert!(aborted.is_none() && polls == 1, "aborted at the first poll ({polls} polls)");
        assert!(hs.state().is_none(), "an aborted call leaves no state");
        let (v, info) = hs.unoccupied();
        assert!(v[0].is_empty() && v[1].is_empty() && info.is_none(), "nothing to adopt after an abort");
        let next = s.harris(&mut hs, &input, &mut || false).expect("no failure").expect("not aborted");
        println!("after the abort: cold {}, iterations {:?} (first call {:?})", next.cold, next.iterations, first.iterations);
        assert!(!next.cold && hs.state().is_some(), "the next call succeeds warm");
    }

    /// m = 2 on H₂ 33³: `adopt_unoccupied` copies the Harris unoccupied vectors into the view, and the next
    /// `update` on the Harris state shows per channel the occupied row (σ_g, index 0, f = 1) followed by the
    /// unoccupied rows the block-edge rule keeps, ε ascending and above the HOMO. The count is **3**, not m:
    /// the box states of this cube are 0.12774, 0.20857 ×2, 0.36268 Ha (`references.md`, the CPU
    /// `unoccupied_cpu` row for H₂ 33³ m = 2), so the window of 2 cuts the degenerate pair and the rule
    /// (`qm::orbitals` §6, `block_edge`) shows that τ-set whole — the plan's "2 + 2 rows" did not account
    /// for it. Cross-check: the same rows from the converged path (`OrbitalView::unoccupied` on the SCF
    /// state, H^out of ρ_SCF — the same operator up to the SCF's Δρ) agree to 1e-6 Ha (measured ≤ 2.5e-8).
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm-gpu --release")]
    fn adopted_unoccupied_rows_follow_the_occupied() {
        let Some(ctx) = ctx() else { return };
        let (mut s, settings, scf) = solved_h2(&ctx);
        let sys = h2();
        let mut hs = HarrisState::new();
        let input = HarrisInput { system: &sys, settings: &settings, rho_in: [&scf.rho_up, &scf.rho_dn], unoccupied: 2 };
        let out = s.harris(&mut hs, &input, &mut || false).expect("no failure").expect("not aborted");
        let (vectors, info) = hs.unoccupied();
        let info = info.expect("a record").clone();
        assert_eq!(info.count(), [1 + 2 + UNOCC_GUARD - 1; 2], "k_σ − n_occ,σ unoccupied vectors per channel");
        let mut view = OrbitalView::new(&s);
        view.adopt_unoccupied(&s, vectors, info.clone());
        let st = hs.state().unwrap();
        let upd = view.update(&s, st, &sys.nuclei, settings.tol, Product::None);
        let tau_scf = tau(settings.tol);
        for (ch, spin) in [Spin::Up, Spin::Down].into_iter().enumerate() {
            let rows: Vec<_> = upd.out.rows.iter().filter(|r| r.spin == spin).collect();
            let homo = out.evals[ch][0];
            let unocc: Vec<f64> = rows.iter().filter(|r| r.index >= 1).map(|r| r.energy).collect();
            let all: Vec<f64> = out.evals[ch].clone();
            let (shown, hidden) = qm::orbitals::block_edge(&all, 1, all.len(), tau_scf);
            println!("spin {ch}: rows {:?}, ε {:?}, block edge shown {shown} hidden {hidden}",
                     rows.iter().map(|r| (r.index, r.energy, r.occupation)).collect::<Vec<_>>(), out.evals[ch]);
            assert_eq!(rows.iter().filter(|r| r.index == 0 && r.occupation == 1.0).count(), 1, "spin {ch}: the occupied σ_g row");
            assert_eq!(unocc.len(), shown - 1, "spin {ch}: the unoccupied rows the block edge keeps");
            assert_eq!(unocc.len(), 3, "spin {ch}: m = 2 plus the degenerate pair's partner");
            assert!(unocc.windows(2).all(|w| w[0] <= w[1]) && unocc.iter().all(|e| *e > homo), "spin {ch}: unoccupied ε ascending above the HOMO {homo}");
            assert!(rows.iter().filter(|r| r.index >= 1).all(|r| r.occupation == 0.0));
        }
        // the converged path's display solve on the SCF state: the same unoccupied levels
        let mut conv = OrbitalView::new(&s);
        let u = conv.unoccupied(&mut s, &sys.nuclei, 2, &settings, &mut || false).expect("not aborted");
        assert!(u.rejected.is_none(), "{:?}", u.rejected);
        for ch in 0..2 {
            for i in 0..u.evals[ch].len() {
                let d = (u.evals[ch][i] - info.evals[ch][i]).abs();
                println!("  spin {ch} unoccupied #{i}: Harris {:+.8} ({:.1e}{}) converged path {:+.8} ({:.1e}{}) |Δ| {d:.1e}",
                         info.evals[ch][i], info.residuals[ch][i], if info.converged[ch][i] { "" } else { " NOT" },
                         u.evals[ch][i], u.residuals[ch][i], if u.converged[ch][i] { "" } else { " NOT" });
                if info.converged[ch][i] && u.converged[ch][i] { assert!(d <= 1e-6, "spin {ch} #{i}: |Δε| = {d:.2e}"); }
            }
            assert!(info.converged[ch][..3].iter().all(|c| *c), "spin {ch}: the shown Harris rows are converged");
        }
    }

    /// Pseudo-O (`IonicPolicy::All`: V_loc, KB projectors and a partial core), triplet (n↑ = 4, n↓ = 2),
    /// `DegenerateShells` (the ↓ p shell shared ⅓ each), 33³ × 0.25, converged by the SCF. Returns the solver,
    /// the system, its settings and the SCF state on the CPU.
    fn solved_pseudo_o(ctx: &GpuContext) -> (GpuSolver, System, ScfSettings, KsState) {
        use crate::ionic::tests::filtered_at;
        use qm::ionic::{Ionic, IonicPolicy, IonicSetup};
        let h = 0.25;
        let grid = Grid::cube(33, h);
        let setup = std::sync::Arc::new(IonicSetup::from_tables(IonicPolicy::All, h, h, std::collections::BTreeMap::from([(8, Ionic::Pseudo(filtered_at(8, h)))])));
        let sys = System::with_ionic(vec![Nucleus { pos: [0.07, -0.03, 0.02], z: 8 }], 0, Some(3), setup);
        let mut settings = ScfSettings::for_grid(&grid);
        settings.occupations = Occupations::DegenerateShells { tau: qm::DEGENERACY_TAU };
        let mut s = GpuSolver::new(ctx.clone(), &grid);
        s.solve(&sys, &settings, false).expect("the SCF converges");
        let ks = s.ks_state(s.state().unwrap());
        (s, sys, settings, ks)
    }

    /// The fixed-point identity on a pseudised, open-shell, partial-core species (review I2): Harris on
    /// ρ_in = the SCF's ρ↑/ρ↓ of pseudo-O triplet at the same geometry reproduces every f > 0 level of both
    /// spins to 1e-5 Ha with the SCF's occupations. Every ionic term and the spin assignment enter: V_loc in
    /// v_nuc, V_NL inside H, ρ_pc in v_xc only (in the Poisson source it would add the core's ~2 e to V_H),
    /// v_xc↑ on ↑ (↑ and ↓ differ by the exchange splitting here). At m = 0 the guard levels still hold the
    /// gap witness above the shared ↓ p shell — n_occ↓ = 4, so the block grows from n + G = 4 to 6 — no
    /// short-window note, and nothing is offered for display.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm-gpu --release")]
    fn harris_reproduces_a_pseudised_open_shell_scf() {
        let Some(ctx) = ctx() else { return };
        let (mut s, sys, settings, scf) = solved_pseudo_o(&ctx);
        let mut hs = HarrisState::new();
        let input = HarrisInput { system: &sys, settings: &settings, rho_in: [&scf.rho_up, &scf.rho_dn], unoccupied: 0 };
        let out = s.harris(&mut hs, &input, &mut || false).expect("no failure").expect("not aborted");
        println!("pseudo-O triplet: {out:?}");
        let ion = hs.ionic.as_ref().expect("its own ionic terms");
        assert!(ion.has_partial_core() && ion.nonlocal().is_some(), "the O table carries a partial core and KB projectors");
        let st = hs.state().unwrap();
        for (ch, (h_ev, h_f, s_ev, s_f)) in [(&st.evals_up, &st.occ_up, &scf.evals_up, &scf.occ_up), (&st.evals_dn, &st.occ_dn, &scf.evals_dn, &scf.occ_dn)].into_iter().enumerate() {
            // the SCF's f > 0 levels by ε (its stored orbitals include the f = 0 witnesses)
            let mut occupied: Vec<(f64, f64)> = s_ev.iter().zip(s_f).filter(|(_, f)| **f > 0.0).map(|(e, f)| (*e, *f)).collect();
            occupied.sort_by(|a, b| a.0.total_cmp(&b.0));
            assert_eq!(h_ev.len(), occupied.len(), "spin {ch}: the same f > 0 levels");
            for (i, ((e, f), (es, fs))) in h_ev.iter().zip(h_f).zip(&occupied).enumerate() {
                let d = (e - es).abs();
                println!("  spin {ch} #{i}: ε_H {e:+.8} (f {f:.4}) ε_SCF {es:+.8} (f {fs:.4}) |Δ| {d:.2e}");
                assert!(d <= 1e-5, "spin {ch} #{i}: |ε_H − ε_SCF| = {d:.3e} > 1e-5 Ha");
                assert!((f - fs).abs() <= 1e-12, "spin {ch} #{i}: f_H {f} vs f_SCF {fs}");
            }
        }
        assert_eq!(out.n_occ, [4, 4], "↑ s + 3p; ↓ s + the shared p shell");
        assert_eq!([out.evals[0].len(), out.evals[1].len()], [4 + UNOCC_GUARD, 4 + UNOCC_GUARD], "n_occ + G levels kept per channel");
        assert!(!out.notes.iter().any(|n| n.contains("no gap witness")), "the witness is present at m = 0: {:?}", out.notes);
        let (v, info) = hs.unoccupied();
        assert!(v[0].is_empty() && v[1].is_empty() && info.is_none(), "m = 0 offers no unoccupied rows");
    }

    /// The block follows n_occ (review I1, spec §4.1 "the lowest n_occ,σ + m states"): pseudo-O triplet at
    /// m = 2 keeps m + G unoccupied vectors in **both** spins although ↓'s shared p shell makes n_occ↓ = 4 > n↓,
    /// and the view shows ↓ unoccupied rows (the block edge's count from m = 2: at least 2, more only when a
    /// τ-set straddles the window). A second call starts from the grown block (warm, sized from the last
    /// n_occ: no growth solve).
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm-gpu --release")]
    fn harris_block_follows_a_shared_shell() {
        let Some(ctx) = ctx() else { return };
        let (mut s, sys, settings, scf) = solved_pseudo_o(&ctx);
        let mut hs = HarrisState::new();
        let input = HarrisInput { system: &sys, settings: &settings, rho_in: [&scf.rho_up, &scf.rho_dn], unoccupied: 2 };
        let out = s.harris(&mut hs, &input, &mut || false).expect("no failure").expect("not aborted");
        println!("pseudo-O m = 2: {out:?}");
        assert_eq!(out.n_occ, [4, 4]);
        let (vectors, info) = hs.unoccupied();
        let info = info.expect("a record").clone();
        assert_eq!(info.count(), [2 + UNOCC_GUARD; 2], "m + G unoccupied vectors in both spins");
        let mut view = OrbitalView::new(&s);
        view.adopt_unoccupied(&s, vectors, info.clone());
        let upd = view.update(&s, hs.state().unwrap(), &sys.nuclei, settings.tol, Product::None);
        let tau_scf = tau(settings.tol);
        for (ch, spin) in [Spin::Up, Spin::Down].into_iter().enumerate() {
            let unocc: Vec<f64> = upd.out.rows.iter().filter(|r| r.spin == spin && r.occupation == 0.0).map(|r| r.energy).collect();
            let (shown, _) = qm::orbitals::block_edge(&out.evals[ch], 4, out.evals[ch].len(), tau_scf);
            println!("  spin {ch}: unoccupied rows {unocc:?}, block edge shown {shown}");
            assert_eq!(unocc.len(), shown - 4, "spin {ch}: the block edge's rows");
            assert!(unocc.len() >= 2, "spin {ch}: at least m = 2 unoccupied rows");
        }
        let again = s.harris(&mut hs, &input, &mut || false).expect("no failure").expect("not aborted");
        println!("second call: cold {}, iterations {:?}, solves {:?} (first call {:?} in {:?} solves, at cap {:?})",
                 again.cold, again.iterations, again.solves, out.iterations, out.solves, out.at_cap);
        assert!(out.solves[1] >= 2, "the first call grew ↓ (n_occ↓ = 4 > n↓): {:?} solves", out.solves);
        assert!(!again.cold && again.evals[1].len() == 4 + 2 + UNOCC_GUARD);
        assert_eq!(again.solves, [1, 1], "sized from the last block and head: no growth or head solve");
        // a channel that converged on the first call is converged on entry (before the guard tolerance of
        // task 8b, ↓ ended on the cap here — guard levels near 0.407 Ha at residual 5.7e-4 — and the warm
        // restart continued it for 54 iterations); a capped one would be continued
        for ch in 0..2 {
            if !out.at_cap[ch] { assert_eq!(again.iterations[ch], 0, "spin {ch}: converged on entry"); }
        }
        assert_eq!(out.at_cap, [false, false], "the guard levels meet GUARD_TOL within the cap");
    }

    /// Task 8b perf: a warm refresh after a 0.1 Bohr move ends below the 200-iteration cap — the guard
    /// levels are held to `GUARD_TOL`, the displayed ones to the SCF's tol — on H₂ (49³ × 0.25, superposed
    /// free-atom ρ_in) and pseudo-O triplet (33³, its SCF density at each geometry), m = 4 (the app's
    /// default). Before the change both ended on the cap (H₂ 81³: warm 1.6–2.1 s, ↓ 200 iterations).
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm-gpu --release")]
    fn warm_refresh_after_a_move_stays_below_the_cap() {
        let Some(ctx) = ctx() else { return };
        // H₂, 49³
        let grid = Grid::cube(49, 0.25);
        let settings = ScfSettings::for_grid(&grid);
        let mut s = GpuSolver::new(ctx.clone(), &grid);
        let mut hs = HarrisState::new();
        for (step, dx) in [0.0, 0.1].into_iter().enumerate() {
            let pos = [[0.7 + dx, 0.0, 0.0], [-0.7, 0.0, 0.0]];
            let sys = System::new(pos.iter().map(|p| Nucleus { pos: *p, z: 1 }).collect(), 0, None);
            let rho = superposed_h(&grid, &pos);
            let out = s.harris(&mut hs, &HarrisInput { system: &sys, settings: &settings, rho_in: [&rho[0], &rho[1]], unoccupied: 4 }, &mut || false)
                .expect("no failure").expect("not aborted");
            println!("H₂ 49³ move {dx}: {} wall {:.3} s, iterations {:?}, solves {:?}, at cap {:?}, notes {:?}",
                     if out.cold { "cold" } else { "warm" }, out.wall.as_secs_f64(), out.iterations, out.solves, out.at_cap, out.notes);
            if step == 1 { assert!(!out.cold && out.at_cap == [false, false], "H₂: the warm refresh ended on the cap"); }
        }
        // pseudo-O triplet, 33³
        let (mut s, sys, settings, _) = solved_pseudo_o(&ctx);
        let mut hs = HarrisState::new();
        for (step, dx) in [0.0, 0.1].into_iter().enumerate() {
            let moved = System::with_ionic(vec![Nucleus { pos: [0.07 + dx, -0.03, 0.02], z: 8 }], 0, Some(3), sys.ionic.clone().expect("a setup"));
            s.solve(&moved, &settings, true).expect("the SCF converges");
            let ks = s.ks_state(s.state().unwrap());
            let out = s.harris(&mut hs, &HarrisInput { system: &moved, settings: &settings, rho_in: [&ks.rho_up, &ks.rho_dn], unoccupied: 4 }, &mut || false)
                .expect("no failure").expect("not aborted");
            println!("pseudo-O 33³ move {dx}: {} wall {:.3} s, iterations {:?}, solves {:?}, at cap {:?}, notes {:?}",
                     if out.cold { "cold" } else { "warm" }, out.wall.as_secs_f64(), out.iterations, out.solves, out.at_cap, out.notes);
            if step == 1 { assert!(!out.cold && out.at_cap == [false, false], "pseudo-O: the warm refresh ended on the cap"); }
        }
    }
}
