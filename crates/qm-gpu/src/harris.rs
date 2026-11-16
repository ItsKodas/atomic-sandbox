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
//!   f_iσ   = qm::one_shot_occupations(ε_σ, n_σ, rule)   (Σ_i f_iσ = n_σ; Aufbau, or FermiDirac: qm::fermi_dirac at
//!                                                         the rule's kT)
//!   ρ_out,σ = Σ_{i occupied} f_iσ ψ_iσ²                   (h³ Σ ρ_out,σ = n_σ; under FermiDirac to n_σ − Σ of the
//!                                                         f < PARTIAL_OCCUPATION_MIN tail, ≤ k·1e-6)
//! ```
//!
//! — exactly the potential of one SCF step (`scf` module doc: `xc_in`, then `lincomb_in` of
//! [v_nuc, V_H, v_xc,σ] with unit weights, ρ_pc added to each spin's xc input as ½ρ_pc and never to the
//! Poisson source), evaluated once. A converged SCF density is a fixed point of that map, so the Harris
//! solve on ρ_in = ρ_SCF reproduces the SCF's eigenpairs to its tolerance
//! (`harris_on_the_converged_density_reproduces_the_scf`).
//!
//! **The block: the lowest n_occ,σ + m + G states** (spec §4.1; m the unoccupied count
//! asked for, G = `UNOCC_GUARD`, n_occ,σ the levels with f > 0). n_occ,σ is known only after the
//! occupations, and under `FermiDirac` a partly filled shell makes it exceed n_σ (pseudo-O triplet ↓: n = 2,
//! f = 1, ⅓, ⅓, ⅓ ⇒ n_occ = 4). So each channel starts from a split (`start_split`: the last call's need when
//! it had the same n_σ, else n_σ + m + G vectors), solves (one LOBPCG at the SCF's effective tol,
//! `effective_eig_tol`, and cap, `EIGEN_MAX_ITER`), applies `qm::one_shot_occupations`, and asks
//! `split_need` what this pass needs: n_occ + m + G tested levels, **or** every level that decides the
//! occupations when that is more (`decisive_levels` under `Aufbau`: n_σ + 1; `decisive_levels_fd` under `FermiDirac`,
//! the smallest prefix holding a witness on the lower bound θ − r — with none, `window_short`, every computed level
//! decides and the extra block doubles, as the SCF's growth does; the count n_occ alone would not see it), and one
//! buffer (below). When
//! the block is short the new columns are fresh `qm::scf::initial_orbitals` ones (seed 5 ↑ / 6 ↓, the
//! columns past those already held) behind the current Ritz vectors, and it solves again — the converged
//! columns need no further work, but the solve runs until the new columns converge or reach the cap; this
//! is the SCF's witness growth (`qm::scf` "Occupations"), bounded by `MAX_WITNESS_GROWTH` tested levels
//! beyond n_σ + m + G (`split_bound`) — then re-applies the occupations. A block larger than needed (the
//! shell split since the last call) is trimmed from the top. So every completed solve keeps n_occ + m + G
//! levels in its record: G above the occupied ones and above the m asked for, which hold the witness whenever
//! the shell has one within that reach, and the unoccupied record
//! always has m + G vectors, the count `block_edge` reads m from. A short window is left only at the growth
//! bound (the note says so, and is emitted only then). The guess is this state's previous vectors (trimmed,
//! or topped up with fresh columns), else `initial_orbitals`. Occupied and unoccupied come out of the
//! **same** Rayleigh–Ritz, so there is no separate unoccupied step and no `occupied_disagreement` check
//! (spec §4.2: the builder's choice). The occupied levels (`occupied`; the lowest by rank under both rules) become
//! the state's orbitals with their f; the rest go through the view's own tail (`orbitals::record_unoccupied`:
//! re-measured ε and residual, sorted, τ-set face density, block edge) into this state's `Unoccupied`.
//! With m = 0 the G guard levels are still solved (the witness) but `HarrisState::unoccupied` offers none
//! (`qm::orbitals`: n_computed = n_occ when no row is asked for).
//!
//! **Three roles, three tolerances** (task 8b perf: at 109³ every refresh ended on the 200-iteration cap
//! polishing the guard levels — box states — for 3.6 s warm; measured at 81³, the column holding the cap
//! was always the **top** one, residual 1e-2–3e-2, every other ≤ 1e-3: the last column of a block
//! converges at a rate set by the gap to the first level outside it, and a large box's states are a near
//! continuum). The stop test (`eigen::lobpcg_split`; the tiers are a `Split`) holds the lowest columns —
//! the displayed n_occ + m — to the SCF's effective tol, the tested columns above them — the record's
//! guards, and any witness levels above the record — to `GUARD_TOL` = `WITNESS_GAP`/100 = 1e-3 Ha, and does not test
//! the top column: it is a **buffer** vector (standard block-eigensolver practice), iterated to speed up
//! the others, whose Ritz value must decide nothing. The witness is never the buffer (`decisive_levels`:
//! when it would be, the block grows by one). Only at the growth bound, where no buffer fits, is every
//! column tested — and when the window is still short there, every computed level may carry a non-negligible
//! f (in ρ_out), so every column is held to the SCF's tol (review M1).
//!
//! **The record's edge** (task 8b fix 1, review I2). Normally the buffer is the record's last vector, and
//! `block_edge` reads every record value: a τ-set reaching the last vector is hidden whole, so a buffer
//! Ritz value too high by r²/δ ≫ τ (residual 1e-2–3e-2) could split a set truly reaching the edge and show
//! its lower members as complete. It can do so only when the window's last row (position n_occ + m − 1)
//! and the first guard (n_occ + m) are one τ-set: `block_edge` shows the single-linkage τ-sets that start
//! inside the window (start < m) and hides one that ends on the last vector; with a gap > τ between
//! positions m − 1 and m (relative to n_occ) every set starting inside the window ends by m − 1, two short
//! of the last vector (G = 2), the set after it starts at m and is neither shown nor counted, and the last
//! value takes part in neither — the shown rows and the hidden count are the same whatever it is (the
//! view's update recomputes `block_edge` from the same values with the same τ: `providers::ks` hands the
//! view the solve's own tolerance, `settings.tol` after the GPU floor raise; and a τ-set's face density
//! concerns only shown sets). So when, after a solve, those two positions are within 2τ (`record_edge_matters`; the
//! margin covers the record's f32 re-measure of the same vectors) the last record vector is tested at
//! `GUARD_TOL` and one buffer is added above the record; otherwise the record's last vector stays the
//! buffer. The reviewer's fix — a buffer above the record always — was measured first: one vector more
//! costs 1.4–1.55× per LOBPCG iteration (the Gram readback is ∝ (3k)²), and with the record's last vector
//! tested in the near continuum, 7 of 16 warm 0.1 Bohr refreshes of H₂ (81³, 109³, m = 4) took 140–186
//! iterations instead of ≈ 90 — each one a converged σ_g column replaced, in one Rayleigh–Ritz step, by a
//! combination of the W and P directions (no X component, residual 5 Ha; a known fragility of LOBPCG's raw
//! [X W P] basis, Hetmaniuk & Lehoucq, J. Comput. Phys. 218 (2006) 324) and re-converged from there; the
//! 8b layout had none in the same 16 (`harris_record_edge_is_a_tested_guard` exercises the edge case).
//!
//! Why `GUARD_TOL` is enough: the witness test (`qm::witness_bound_reached`, arm 1) compares a guard Ritz value
//! with ε_F + `WITNESS_GAP` (0.1 Ha; arm 2 with μ + 36 kT = μ + 0.18 Ha); a normalised vector with residual r has an eigenvalue within r
//! of its Rayleigh quotient (|θ − λ| ≤ ‖Hx − θx‖), so with the levels below it resolved (as for any LOBPCG
//! block: no level is missed below a Ritz value) |θ − λ| ≤ 1e-3 Ha. A level is then classified as a
//! witness or not as at full convergence unless its true energy is within 1e-3 of the threshold, i.e. within
//! 1 % of the gap it reads — the same knife edge any f32 rounding or 0.01 Bohr move already flips, and a
//! display decision. After a Rayleigh–Ritz step the error is moreover one-sided, θ_j ≥ λ_j (Courant–Fischer
//! on the Rayleigh–Ritz subspace), the safe direction for display: a guard can make a true in-window level
//! look like a witness (that shell is then drawn with integer occupations a hair early), never create a
//! spurious in-window level or drop an occupied one. A warm guess accepted at iteration 0 is not rotated
//! (`eigen` module doc: the guess is neither rotated nor rescaled), so its Rayleigh quotients carry only
//! the two-sided bound: there the 1 % band applies on either side of the threshold (the same margin, no
//! guaranteed direction). A capped tested guard (ending above 1e-3) widens that band to its residual, which
//! the at-cap note prints (the displayed and the guard tiers' residuals, `EigenStats::split_residual`; the
//! buffer's enters neither — task 8b printed the max over all columns, the buffer's included, review M2). When m > 0 the first level above the occupied ones — the witness that decides
//! — is a displayed level and held to the full tol anyway; the guard tolerance decides only at m = 0 and
//! for a shell reaching past the m rows. The displayed levels keep the full tol.
//!
//! **A displayed τ-set reaching into the guards** (the m-th row degenerate with the next level — H₂ 33³'s
//! box pair at 0.20857 Ha straddles m = 2): `block_edge` shows the whole set, so after each solve the shown
//! count is checked against the head and, when it reaches past it, the head is raised to cover the shown
//! rows and the solve continues (warm: only those columns move). Every shown row is then held to the full
//! tol; the set's membership itself is judged on Ritz values whose error is ~r²/δ ≤ (1e-3)²/δ — at
//! τ = 1e-4 a guard (the record's last vector included, when it matters) can be misjudged as in or out of
//! the set only when its distance δ to the rest of the spectrum is ≲ 1e-2 Ha, and a level then shown is
//! re-solved to tol by this same rule.
//!
//! **Block memory.** A completed solve leaves the split its final occupation pass needed (`Memory`: the
//! block — a short window's witness levels and a buffer above the record are kept as spare
//! guess vectors — the raised head and the tested count), and the next call with the same n_σ starts
//! there, so a warm refresh does not re-grow its block or re-raise its head. It is the **need** that is
//! remembered, never the split the call ran with (raised from its start, never lowered within a call):
//! task 8b remembered the head it ran with, so every raise and every shrink of the block accumulated — one
//! way, until n_σ changed — up to every column at the full tol, the buffer included (review I1). A call
//! right after n_occ dropped still runs the previous split once (its columns already meet it: the warm
//! guess); the next is back at the minimal one (`split_memory_is_no_ratchet`,
//! `harris_split_follows_n_occ_between_calls`).
//!
//! **Fermi–Dirac** (the scf-stall ← main merge ruling, 2026-11-16; smearing spec 2026-11-13 §4, plan Task 9's display
//! rule). Under `FermiDirac { kt }` — every scene's occupations on the smearing surface — the f are
//! `qm::fermi_dirac` of this solve's own levels at the same kT, per spin at the scene's (n↑, n↓): the rule every SCF
//! step applies, applied once. Its window is the SCF's witness on the levels' lower bounds (plan 2026-11-13 Task 9b,
//! `decisive_levels_fd`: the smallest prefix whose top θ − r ≥ ε_(n_σ) + `WITNESS_GAP` and ≥ μ_σ +
//! `WITNESS_TAIL_KT`·kT, r the column's true residual from `lobpcg_split`); short of it the extra levels double, as the
//! SCF's growth does, up to the solver's witness cap (`GpuSolver::witness_cap`) of tested levels beyond n_σ + m + G
//! (`split_bound`), and a window still short at that
//! bound leaves the f approximate — a note in the payload ("the occupations are approximate"), never an error. Every FD
//! level has f > 0, so "occupied" is not f > 0: the occupied levels (the state's orbitals, ρ_out's terms, the displayed
//! occupied rows) are those with f ≥ `PARTIAL_OCCUPATION_MIN` (1e-6, `qm::orbitals::shown_columns`' rule) — the lowest
//! by rank, since f falls with ε — and never the block's top column (the witness; it holds f < 2.4e-16 whenever the
//! witness is reached, so this binds only at a short window at the bound). The levels below the threshold go to the
//! unoccupied record (f shown 0), so ρ_out,σ misses their ≤ k·1e-6 electrons. A gapped channel (H₂: f = 1, ~0) gives the
//! `Aufbau` result; a partly filled shell (pseudo-O at M = 3: ↓ 2s + one electron over 2p) gives ⅓ per p level, the
//! SCF's FD f within its tolerance (`harris_fermi_dirac_*` tests).
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
use qm::scf::{ascending, initial_orbitals};
use qm::{fermi_dirac, grown_extras, witness_bound_reached, Occupations, PARTIAL_OCCUPATION_MIN, WITNESS_GAP};
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
/// ascending — the occupied ones (f > 0; under `FermiDirac` f ≥ `PARTIAL_OCCUPATION_MIN`, module doc
/// "Fermi–Dirac": the LOBPCG Ritz values with their raw f), then the unoccupied record's re-measured
/// ε with f = 0 (Rayleigh quotients of the same vectors; the two agree to the f32 noise, so the parts can
/// interleave only at a degenerate HOMO/LUMO pair under `Aufbau`); `n_occ` the
/// occupied levels (the state's orbitals); `notes` the informational lines (a short
/// window, an eigensolve at its cap); `iterations`/`at_cap` per channel (summed over the block's growth
/// solves; at the cap in any of them); `solves` the LOBPCG calls per channel (1, plus one per growth of the
/// block; 0 for a skipped channel); `split` per channel the last LOBPCG solve's (block length, head,
/// checked): the lowest `head` columns held to the SCF's tol, `head..checked` to `GUARD_TOL`, the rest
/// buffers (module doc, "Three roles, three tolerances"; (0, 0, 0) for a skipped channel); `cold` = some
/// non-empty channel had no previous vector to start from (all from `initial_orbitals`); `wall` the
/// call's wall time.
#[derive(Clone, Debug)]
pub struct HarrisSolve {
    pub evals: [Vec<f64>; 2],
    pub occ: [Vec<f64>; 2],
    pub n_occ: [usize; 2],
    pub notes: Vec<String>,
    pub iterations: [usize; 2],
    pub at_cap: [bool; 2],
    /// Per channel: its last LOBPCG solve ended on the f32 floor (`eigen::Stop::Floor`, "as good as f32 allows",
    /// possibly above tol), not on tol (plan 2026-11-13 Task 9b fix round 1).
    pub at_floor: [bool; 2],
    pub solves: [usize; 2],
    pub split: [(usize, usize, usize); 2],
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
    /// Levels the last solve kept above its record only as the next guess (`split_need`: a short
    /// window's witness levels and the buffer), per channel, ascending.
    spare: [Vec<Field>; 2],
    /// The last completed solve's final split need per channel (`Memory`), so the next call starts at the
    /// block and tiers it needed.
    last: [Option<Memory>; 2],
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

/// The guard/witness levels' LOBPCG tolerance (module doc, "Three roles, three tolerances"): `WITNESS_GAP`/100 =
/// 1e-3 Ha, so a guard Ritz value θ is within ‖r‖ ≤ 1e-3 Ha of an eigenvalue — 1 % of the gap the witness test's first
/// arm reads (and ≤ 0.6 % of its 36 kT = 0.18 Ha tail arm).
const GUARD_TOL: f64 = WITNESS_GAP / 100.0;

/// The fewest lowest levels of `sorted` (ascending) that decide the `Aufbau` occupations of `n` electrons: n + 1 (it
/// never reads a level above the n-th; the + 1 is the gap level), at most `sorted.len()`. `FermiDirac`'s is
/// `decisive_levels_fd`.
fn decisive_levels(sorted: &[f64], n: usize) -> usize { (n + 1).min(sorted.len()) }

/// Whether a level of occupation `f` is an occupied one (module doc, "Fermi–Dirac"): f > 0 under `Aufbau`;
/// f ≥ `PARTIAL_OCCUPATION_MIN` under `FermiDirac` (every FD level has f > 0).
fn occupied(f: f64, fermi_dirac: bool) -> bool { if fermi_dirac { f >= PARTIAL_OCCUPATION_MIN } else { f > 0.0 } }

/// The occupied count of a channel whose f are `f`, levels ascending by `order` (module doc, "Fermi–Dirac"): the
/// lowest levels by rank that are `occupied`; under `FermiDirac` never the top column (the witness).
fn occupied_count(f: &[f64], order: &[usize], fermi_dirac: bool) -> usize {
    let n = order.iter().take_while(|&&i| occupied(f[i], fermi_dirac)).count();
    if fermi_dirac { n.min(order.len().saturating_sub(1)) } else { n }
}

/// One channel's eigensolve split (module doc, "Three roles, three tolerances"): `len` vectors, the lowest
/// `head` held to the SCF's tol, `head..checked` to `GUARD_TOL`, `checked..len` untested buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Split { len: usize, head: usize, checked: usize }

/// The largest block of a channel with n_σ electrons and m unoccupied rows: n_σ + m + G tested levels, up to `w` more
/// for the witness (the solver's cap, `GpuSolver::witness_cap`: `MAX_WITNESS_GROWTH` or what the VRAM budget fits —
/// the SCF's growth bound), and the buffer.
fn split_bound(n: usize, m: usize, w: usize) -> usize { n + m + UNOCC_GUARD + w + 1 }

/// `decisive_levels` under `FermiDirac` (plan 2026-11-13 Task 9b, the SCF's rule: `qm::occupations` "Witness columns at
/// a guard tolerance"): the smallest b > n whose lowest b levels (ascending, `res` their residuals) hold a witness on
/// the lower bound θ − r (`qm::witness_bound_reached`, μ of those b levels); when no prefix does, every computed level
/// decides and the extra levels above n double (`qm::grown_extras`, the SCF's growth): 2·len − n.
fn decisive_levels_fd(sorted: &[f64], res: &[f64], n: usize, kt: f64) -> usize {
    if n == 0 { return sorted.len().min(1); } // an empty channel needs no witness (`decisive_levels`' n + 1)
    for b in (n + 1)..=sorted.len() {
        if let Ok(Some(mu)) = fermi_dirac(&sorted[..b], n, kt).map(|o| o.mu) {
            if witness_bound_reached(&sorted[..b], &res[..b], n, mu, kt) { return b; }
        }
    }
    n + grown_extras(sorted.len().saturating_sub(n), usize::MAX)
}

/// Whether the record's last vector can change what `block_edge` shows (module doc, "The record's edge"):
/// only when the window's last row (position n_occ + m − 1 of `sorted`, ascending) and the first guard
/// (n_occ + m) are one τ-set — tested with 2τ, a margin over the record's re-measure of the same vectors
/// (f32 rounding, ~1e-8 Ha). False at m = 0 (no window) and while the block does not reach the first guard.
fn record_edge_matters(sorted: &[f64], n_occ: usize, m: usize, tau: f64) -> bool {
    m > 0 && sorted.len() > n_occ + m && sorted[n_occ + m] - sorted[n_occ + m - 1] <= 2.0 * tau
}

/// The split one occupation pass asks for (module doc): the displayed n_occ + m levels — or up to a shown
/// τ-set's end `shown` — at the SCF's tol; tested, every level that decides the occupations (`decisive`,
/// `decisive_levels`) and every record vector (n_occ + m + G) but the last, and the last too when
/// `edge` (`record_edge_matters`: its Ritz value can change the shown rows, review I2); one untested buffer
/// above them — the record's last vector when it is not tested, never the witness. With no room for a
/// buffer under `k_max` every column is tested; and a short window there (`window_short`: no witness among the
/// computed levels, so each may carry a non-negligible f in ρ_out) holds every column to the SCF's tol (review M1).
fn split_need(n_occ: usize, m: usize, shown: usize, decisive: usize, edge: bool, window_short: bool, k_max: usize) -> Split {
    let head = (n_occ + m).max(shown);
    let record = n_occ + m + UNOCC_GUARD;
    let tested = (if edge { record } else { record - 1 }).max(decisive).max(head);
    if tested < k_max { return Split { len: tested + 1, head, checked: tested }; }
    Split { len: k_max, head: if window_short { k_max } else { head.min(k_max) }, checked: k_max }
}

/// What a completed solve leaves for the next call of a channel (`HarrisState::last`): its n_σ and its
/// final occupation pass's `split_need`, as offsets over n_σ + m — never the head or block the call ran
/// with, which a too-large start would otherwise carry on for ever (review I1: a one-way ratchet).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Memory { n: usize, len: usize, head: usize, checked: usize }

impl Memory {
    fn of(need: Split, n: usize, m: usize) -> Memory {
        let base = n + m;
        Memory { n, len: need.len.saturating_sub(base), head: need.head.saturating_sub(base), checked: need.checked.saturating_sub(base) }
    }
}

/// The split a call starts from: the last call's need (`Memory`) when it had this n_σ, clamped to the
/// minimal split — the displayed n_σ + m at tol, the G guards, the last of them the buffer — and to
/// `split_bound` at the witness cap `w`; the minimal split otherwise. The final pass then raises it where this call needs more.
fn start_split(mem: Option<Memory>, n: usize, m: usize, w: usize) -> Split {
    let base = n + m;
    let min = Split { len: base + UNOCC_GUARD, head: base, checked: base + UNOCC_GUARD - 1 };
    match mem {
        Some(mm) if mm.n == n => {
            let len = (base + mm.len).clamp(min.len, split_bound(n, m, w));
            let checked = (base + mm.checked).clamp(min.checked, len);
            Split { len, head: (base + mm.head).clamp(min.head, checked), checked }
        }
        _ => min,
    }
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
        let w_cap = self.witness_cap();
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
        let mut at_floor = [false; 2];
        let mut splits = [(0usize, 0usize, 0usize); 2];
        let mut needs: [Option<Memory>; 2] = [None, None]; // the last occupation pass's split, for `last`
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
            let guard_tol = GUARD_TOL.max(tol);
            let k_max = split_bound(nc, m, w_cap);
            // the split to start from (module doc, "Block memory"): what the last call's final occupation
            // pass needed, when it had this n_σ; else the displayed n_σ + m and G guards, the last the buffer
            let mut cur = start_split(last[ch], nc, m, w_cap);
            let mut x = std::mem::take(&mut guess[ch]);
            if x.is_empty() { cold = true; }
            if x.len() > cur.len { free.extend(x.drain(cur.len..)); } // ascending: the highest go
            top_up(&mut x, cur.len, free, ctx, g, grid, &system.nuclei, seed);
            let h = GpuHamiltonian { v: if ch == 0 { &p.vu } else { &p.vd }, nl };
            let (ev, one, need, fd) = loop {
                let mut ev = vec![0.0; x.len()];
                let mut res = vec![0.0; x.len()];
                let w = grown(work, work_k, ctx, g, x.len());
                solves[ch] += 1;
                splits[ch] = (cur.len, cur.head, cur.checked);
                // the lowest `head` at the SCF's tol, the guards up to `checked` at GUARD_TOL, the rest
                // buffers (module doc, "Three roles, three tolerances")
                let es = match lobpcg_split(ctx, k, g, r, &h, &mut x, &mut ev, w, tol, cur.head, guard_tol, cur.checked, EIGEN_MAX_ITER, Some(&mut *should_abort), Some(&mut res)) {
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
                at_floor[ch] = es.stop == EigenStop::Floor;
                if es.restores_exhausted {
                    notes.push(format!("spin {}: the eigensolve's Ritz-vector formation failed after {} restores (a level may be missing)", SPIN[ch], es.restores));
                }
                if es.stop == EigenStop::MaxIter && !info.at_cap[ch] {
                    info.at_cap[ch] = true;
                    // the tested tiers' residuals (review M2): the buffers' is in `max_residual` only
                    let (hr, gr) = es.split_residual.expect("a split solve");
                    notes.push(format!("spin {}: the eigensolve ended at its {EIGEN_MAX_ITER}-iteration cap (displayed levels' residual {hr:.1e}, tol {tol:.1e}; guard levels' {gr:.1e}, tol {guard_tol:.0e})", SPIN[ch]));
                }
                // what counts as occupied: f > 0 (the lowest by rank) under `Aufbau`; f ≥ PARTIAL_OCCUPATION_MIN under
                // `FermiDirac`, never the witness. The match is exhaustive on purpose (final review M7; module doc
                // "Fermi–Dirac"): a new variant must decide it
                let fd = match settings.occupations {
                    Occupations::Aufbau => false,
                    Occupations::FermiDirac { .. } => true,
                };
                // the occupations of this one diagonalisation; the occupied levels are the lowest by rank
                let mut one = match one_shot_occupations(&ev, nc, ch as u8, settings.occupations) {
                    Ok(o) => o,
                    Err(e) => { free.append(&mut x); outcome = Err(e); break 'channels; }
                };
                let order = ascending(&ev);
                let n_occ = occupied_count(&one.f, &order, fd);
                let sorted: Vec<f64> = order.iter().map(|&i| ev[i]).collect();
                // under FermiDirac the window reads the lower bound θ − r and a short one doubles the extra levels (plan
                // 2026-11-13 Task 9b: the SCF's rule; `decisive_levels_fd`); the untested buffer's residual is its own
                let decisive = match settings.occupations {
                    Occupations::FermiDirac { kt } => {
                        let sorted_res: Vec<f64> = order.iter().map(|&i| res[i]).collect();
                        let d = decisive_levels_fd(&sorted, &sorted_res, nc, kt);
                        one.window_short = d > sorted.len();
                        d
                    }
                    Occupations::Aufbau => decisive_levels(&sorted, nc),
                };
                // the record's shown rows (a τ-set reaching into the guards raises the head over n_occ + m)
                let keep = n_occ + m + UNOCC_GUARD;
                let shown = if keep <= sorted.len() { block_edge(&sorted[..keep], n_occ, keep, tau_scf).0 } else { 0 };
                let edge = record_edge_matters(&sorted, n_occ, m, tau_scf);
                let need = split_need(n_occ, m, shown, decisive, edge, one.window_short, k_max);
                // never lower within a call (the next solve is warm: only the raised columns move)
                let next = Split { len: cur.len.max(need.len), head: cur.head.max(need.head), checked: cur.checked.max(need.checked) };
                if next == cur { break (ev, one, need, fd); }
                // a partly filled shell or a witness reached past the tested columns, or a shown set into the guards:
                // grow behind the Ritz vectors and/or raise the tiers, and solve again
                top_up(&mut x, next.len, free, ctx, g, grid, &system.nuclei, seed);
                cur = next;
            };
            if one.window_short {
                // only under FermiDirac (`Aufbau`'s window is never short)
                debug_assert!(fd && x.len() >= k_max, "a short window below the growth bound grows the block");
                notes.push(format!("spin {}: no Fermi–Dirac witness among the {} computed levels (growth bound reached): the occupations are approximate", SPIN[ch], x.len()));
            }
            let order = ascending(&ev);
            let n_occ = occupied_count(&one.f, &order, fd);
            // the occupied levels are the lowest by rank (FD's f fall with ε; its witness cap drops the top column only)
            let below = order.len().saturating_sub(usize::from(fd)).max(n_occ);
            assert!(order[n_occ..below].iter().all(|&i| !occupied(one.f[i], fd)), "the occupied levels are the lowest ({:?})", one.f);
            let keep = (n_occ + m + UNOCC_GUARD).min(order.len());
            // the block the final pass needed (`split_need`): the record, above it the witness levels a
            // short window needed and the buffer when it is not the record's last vector, kept as
            // spare guess vectors; a block larger
            // than that (the shell split since the last call) loses its top levels
            let block = need.len.min(order.len());
            needs[ch] = Some(Memory::of(need, nc, m));
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
                                  iterations: info.iterations, at_cap: info.at_cap, at_floor, solves, split: splits, cold, wall: t0.elapsed() };
        let Channel { orbitals: orbitals_up, evals: evals_up, occ: occ_up, unocc: unocc_up, spare: spare_up } = up;
        let Channel { orbitals: orbitals_dn, evals: evals_dn, occ: occ_dn, unocc: unocc_dn, spare: spare_dn } = dn;
        *slot = Some(GpuState { grid: grid.clone(), sigma: settings.sigma, orbitals_up, orbitals_dn, evals_up, evals_dn,
                                rho, rho_up: ou, rho_dn: od, v_h, v_nuc, ionic: system.ionic.clone(), occ_up, occ_dn,
                                occupations: settings.occupations });
        *slot_unocc = [unocc_up, unocc_dn];
        *slot_info = Some(info);
        *last = needs;
        *slot_spare = [spare_up, spare_dn];
        *slot_m = m;
        Ok(Some(solve))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qm::scf::MAX_WITNESS_GROWTH;
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
    /// m = 4 (the app's default), ρ_in the superposed free-atom densities. Prints the cold call and
    /// `HARRIS_DIAG_MOVES` (default 3) warm +0.1 Bohr moves (wall, iterations, solves, at cap).
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
        let moves: usize = std::env::var("HARRIS_DIAG_MOVES").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
        for step in 0..=moves {
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

    fn split(len: usize, head: usize, checked: usize) -> Split { Split { len, head, checked } }

    /// The split one occupation pass asks for. The growth trigger (review round 2; plan 2026-11-13 Task 10 restated it
    /// on `FermiDirac`'s window, the shared-shell premise being deleted): a short window (ε = −1.0, −0.40, −0.39, −0.35,
    /// n = 2: the top level is no witness, `window_short`) has every computed level deciding, so the block grows
    /// although n_occ + m + G = 4 = the block at m = 0; with a witness computed it is tested, with the buffer above it
    /// (block memory, task 8b: the next call starts there instead of re-growing). A witnessed block tests its record but the last vector, which is the buffer
    /// — unless that vector can change the shown rows (`edge`, review I2: then it is tested and one buffer
    /// sits above the record); a shared shell and a shown τ-set raise the tiers; at the bound no buffer
    /// fits and every column is tested, and a short window there holds every column to the SCF's tol
    /// (review M1: those levels are all f > 0).
    #[test]
    fn split_need_tests_the_record_and_the_witness_below_a_buffer() {
        let kt = qm::ELECTRON_KT;
        let fd = Occupations::FermiDirac { kt };
        let short = [-1.0, -0.40, -0.39, -0.35];
        let o = one_shot_occupations(&short, 2, 0, fd).expect("one shot");
        assert!(o.window_short, "{o:?}");
        let k_max = split_bound(2, 0, MAX_WITNESS_GROWTH);
        assert_eq!(k_max, 2 + UNOCC_GUARD + MAX_WITNESS_GROWTH + 1);
        assert_eq!(decisive_levels_fd(&short, &[0.0; 4], 2, kt), 2 + 2 * 2, "no witness: every computed level decides, the extra levels double (2 → 4)");
        assert_eq!(split_need(2, 0, 2, 4, false, true, k_max), split(5, 2, 4), "a short window grows the block");
        let levels = [-1.0, -0.40, -0.39, -0.35, 0.2];
        assert_eq!(decisive_levels_fd(&levels, &[0.0; 5], 2, kt), 5, "the witness 0.2 is the 5th level");
        assert_eq!(split_need(2, 0, 2, 5, false, false, k_max), split(6, 2, 5), "the witness tested, the buffer above it");
        assert_eq!(decisive_levels(&levels, 2), 3, "Aufbau: n + 1");
        assert_eq!(decisive_levels(&levels[..2], 2), 2, "Aufbau: at most the computed levels");
        // FermiDirac on the lower bound (plan 2026-11-13 Task 9b): μ at the mid-gap of −0.40 / 0.1 for n = 2 (−0.15) → the
        // witness needs θ − r ≥ μ + 36 kT = 0.03 and ≥ ε_(2) + 0.1 = −0.3
        let fdl = [-1.0, -0.40, 0.1, 0.2, 0.3];
        assert_eq!(decisive_levels_fd(&fdl, &[0.0; 5], 2, kt), 3, "0.1 clears both arms by θ alone");
        let r = [0.0, 0.0, 0.1, 0.0, 0.0];
        assert_eq!(decisive_levels_fd(&fdl, &r, 2, kt), 4, "a residual of 0.1 Ha drops its lower bound to 0.0 < 0.03: the next level");
        assert_eq!(decisive_levels_fd(&fdl[..3], &r[..3], 2, kt), 2 + 2, "no witness: the extra levels double (1 → 2)");
        assert_eq!(decisive_levels_fd(&[0.1, 0.2], &[0.0; 2], 0, kt), 1, "an empty channel needs none");
        assert_eq!(split_need(2, 0, 2, 3, false, false, k_max), split(4, 2, 3), "the record n_occ + G, its last vector the buffer");
        let k_max = split_bound(2, 2, MAX_WITNESS_GROWTH);
        assert_eq!(split_need(4, 2, 6, 5, false, false, k_max), split(8, 6, 7), "a partly filled shell: n_occ + m at tol, + G − 1 tested");
        assert_eq!(split_need(4, 2, 7, 5, true, false, k_max), split(9, 7, 8), "a shown τ-set into the guards: at tol, the record's edge tested");
        assert_eq!(split_need(4, 2, 6, 9, false, false, k_max), split(10, 6, 9), "a witness above the record: tested, a buffer above");
        assert_eq!(split_need(4, 2, 6, k_max, false, false, k_max), split(k_max, 6, k_max), "at the bound: no buffer");
        assert_eq!(split_need(k_max, 2, k_max, k_max, false, true, k_max), split(k_max, k_max, k_max), "a short window at the bound: all at tol");
        // the record's edge: the window's last row and the first guard one τ-set (τ = 1e-4; 2τ margin)
        let tau = 1e-4;
        assert!(record_edge_matters(&[-0.5, 0.1, 0.2, 0.2 + 5e-5, 0.3], 1, 2, tau), "linked across the window edge");
        assert!(!record_edge_matters(&[-0.5, 0.1, 0.2, 0.25, 0.3], 1, 2, tau), "a gap at the window edge");
        assert!(!record_edge_matters(&[-0.5, 0.1, 0.1 + 5e-5, 0.25, 0.3], 1, 2, tau), "a set inside the window");
        assert!(!record_edge_matters(&[-0.5, 0.1, 0.1, 0.1], 1, 0, tau), "m = 0: no window");
        assert!(!record_edge_matters(&[-0.5, 0.1, 0.2], 1, 2, tau), "the block does not reach the first guard");
    }

    /// Review I1 (task 8b fix 1; the reviewer's `ratchet.py`): a call starts from the last call's final
    /// need, never from the split it ran with, so a block or head raised for one call is not carried on.
    /// ↓ of pseudo-O triplet (n = 2, m = 4), each call simulated as the in-call rule (raise to the need, never
    /// lower, until stable) on (n_occ, shown): the shared p shell (4, 8), Aufbau (2, 6), a shown set reaching
    /// one guard (2, 7: the record's edge then matters, so its last vector is tested under a buffer). A call right after a drop still runs the previous split (its columns already met
    /// it: the warm guess), once; the next is back at the minimal split. Before the fix the head ratcheted
    /// to the block's length (every column at the SCF's tol, the buffer included) and stayed there.
    #[test]
    fn split_memory_is_no_ratchet() {
        let (n, m) = (2, 4);
        let k_max = split_bound(n, m, MAX_WITNESS_GROWTH);
        assert_eq!(start_split(None, n, m, MAX_WITNESS_GROWTH), split(8, 6, 7), "minimal: n + m at tol, G guards, the last the buffer");
        let mut last = None;
        let mut ran = Vec::new();
        for (n_occ, shown) in [(4, 8), (2, 6), (2, 6), (2, 7), (2, 6), (2, 6), (4, 8), (2, 6), (2, 6)] {
            let mut cur = start_split(last, n, m, MAX_WITNESS_GROWTH);
            let need = loop {
                let need = split_need(n_occ, m, shown, n + 1, shown > n_occ + m, false, k_max);
                let next = split(cur.len.max(need.len), cur.head.max(need.head), cur.checked.max(need.checked));
                if next == cur { break need; }
                cur = next;
            };
            last = Some(Memory::of(need, n, m));
            ran.push(cur);
        }
        let (shared, aufbau, raised) = (split(10, 8, 9), split(8, 6, 7), split(9, 7, 8));
        assert_eq!(ran, [shared, shared, aufbau, raised, raised, aufbau, shared, shared, aufbau]);
        // another n_σ: the memory does not apply
        assert_eq!(start_split(Some(Memory::of(shared, n, m)), n + 1, m, MAX_WITNESS_GROWTH), split(9, 7, 8));
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

    /// Pseudo-O (`IonicPolicy::All`, triplet, `FermiDirac` at `ELECTRON_KT`: `for_grid`'s; plan 2026-11-13 Task 10 moved
    /// it off the deleted shared-shell rule) at the origin, 49³: a Harris solve with the
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
        let settings = ScfSettings::for_grid(&grid);
        let mut s = GpuSolver::new(ctx, &grid);
        s.solve(&at, &settings, false).expect("the SCF converges");
        let scf = s.ks_state(s.state().unwrap());
        let mut hs = HarrisState::new();
        let input = HarrisInput { system: &moved, settings: &settings, rho_in: [&scf.rho_up, &scf.rho_dn], unoccupied: 0 };
        let out = s.harris(&mut hs, &input, &mut || false).expect("no failure").expect("not aborted");
        println!("pseudo-O moved 0.3: ε {:?}, f {:?}, notes {:?}; SCF ε↑ {:?} f↑ {:?}", out.evals, out.occ, out.notes, scf.evals_up, scf.occ_up);
        // Σf over the occupied levels: n_σ less the FD tail below PARTIAL_OCCUPATION_MIN (module doc, "Fermi–Dirac": ≤ k·1e-6)
        for ch in 0..2 { assert!((neumaier(out.occ[ch].iter().copied()) - [at.n_up, at.n_dn][ch] as f64).abs() < 1e-5, "Σf = n_σ"); }
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
            assert!(unocc.iter().all(|e| *e > homo), "spin {ch}: unoccupied ε above the HOMO {homo}");
            // Controller ruling (2), 2026-11-16 (scf-stall ← main merge): the order of eigenvalues inside a degenerate
            // cluster is not physical — on the FD SCF density ↓'s box pair at 0.20857 Ha came out of the view 1.7e-10 Ha
            // inverted (the tracker orders a set's rows by character). Clusters are the display's own test (the ×n
            // badges: `qm::orbitals::clusters` at τ = `tau(tol)`); the rows ascend strictly between clusters, and within
            // a cluster they are, as a set, the Harris record's levels to within the converged residual bound.
            let mut by_e = unocc.clone();
            by_e.sort_by(f64::total_cmp);
            let sets = qm::orbitals::clusters(&by_e, tau_scf);
            let set_of = |e: f64| sets.iter().position(|r| by_e[r.clone()].contains(&e)).expect("every row is in a cluster");
            let ids: Vec<usize> = unocc.iter().map(|&e| set_of(e)).collect();
            assert!(ids.windows(2).all(|w| w[0] <= w[1]), "spin {ch}: rows out of order between clusters: {unocc:?} (clusters {sets:?})");
            assert!(sets.windows(2).all(|w| by_e[w[0].end - 1] < by_e[w[1].start]), "spin {ch}: clusters not strictly ascending");
            let record = &info.evals[ch][..by_e.len()];
            for r in &sets {
                let bound = r.clone().map(|j| info.residuals[ch][j]).fold(0.0, f64::max);
                for j in r.clone() {
                    let d = (by_e[j] - record[j]).abs();
                    assert!(d <= bound, "spin {ch}: cluster {r:?} row ε {} vs the record's {} (|Δ| {d:.2e} > residual bound {bound:.2e})", by_e[j], record[j]);
                }
            }
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
    /// `FermiDirac` at `ELECTRON_KT` (`for_grid`'s: the ↓ p shell ⅓ each; plan 2026-11-13 Task 10 moved it off the
    /// deleted shared-shell rule, which gave the same ⅓), 33³ × 0.25, converged by the SCF. Returns the solver, the
    /// system, its settings and the SCF state on the CPU.
    fn solved_pseudo_o(ctx: &GpuContext) -> (GpuSolver, System, ScfSettings, KsState) {
        use crate::ionic::tests::filtered_at;
        use qm::ionic::{Ionic, IonicPolicy, IonicSetup};
        let h = 0.25;
        let grid = Grid::cube(33, h);
        let setup = std::sync::Arc::new(IonicSetup::from_tables(IonicPolicy::All, h, h, std::collections::BTreeMap::from([(8, Ionic::Pseudo(filtered_at(8, h)))])));
        let sys = System::with_ionic(vec![Nucleus { pos: [0.07, -0.03, 0.02], z: 8 }], 0, Some(3), setup);
        let settings = ScfSettings::for_grid(&grid);
        let mut s = GpuSolver::new(ctx.clone(), &grid);
        s.solve(&sys, &settings, false).expect("the SCF converges");
        let ks = s.ks_state(s.state().unwrap());
        (s, sys, settings, ks)
    }

    /// The block follows n_occ (review I1, spec §4.1 "the lowest n_occ,σ + m states"): pseudo-O triplet at
    /// m = 2 keeps m + G unoccupied vectors in **both** spins although ↓'s partly filled p shell makes n_occ↓ = 4 > n↓,
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
        // review M4, the measured counts pinned: ↑ (n_occ = n = 4) starts at (8, 6, 7) and solves once more
        // for the shown pair 0.37513 / 0.37515 Ha (2.2e-5 apart < τ = 1e-4) straddling m = 2 — the head raised
        // to 7 and, the record's edge mattering, its last vector tested under a buffer; ↓ starts at (6, 4, 5),
        // grows to 8 for the shared shell (n_occ↓ = 4 > n↓ = 2), then makes the same raise
        assert_eq!(out.solves, [2, 3], "↑ one head raise; ↓ one growth and one head raise");
        assert_eq!(out.split, [(9, 7, 8); 2], "the record n_occ + m + G = 8 tested, a buffer above, the shown 7 at tol");
        assert!(!again.cold && again.evals[1].len() == 4 + 2 + UNOCC_GUARD);
        assert_eq!(again.solves, [1, 1], "sized from the last block and head: no growth or head solve");
        assert_eq!(again.split, out.split, "started at the last call's need");
        // a channel that converged on the first call is converged on entry (before the guard tolerance of
        // task 8b, ↓ ended on the cap here — guard levels near 0.407 Ha at residual 5.7e-4 — and the warm
        // restart continued it for 54 iterations); a capped one would be continued. A channel whose last solve ended
        // on the f32 floor did not converge to tol (plan 2026-11-13 Task 9b fix round 1: ↓'s p triplet floors at
        // 9.2e-6 against tol 5.4e-6, 2× the model as the eigen module doc measures for rotated degenerate sets), so
        // its re-measured guess is above tol and is continued — never at the cap
        for ch in 0..2 {
            if !out.at_cap[ch] && !out.at_floor[ch] { assert_eq!(again.iterations[ch], 0, "spin {ch}: converged on entry"); }
            assert!(!again.at_cap[ch], "spin {ch}: the warm call ended on the cap");
        }
        println!("first call at floor {:?}", out.at_floor);
        assert_eq!(out.at_cap, [false, false], "the guard levels meet GUARD_TOL within the cap");
    }

    /// Review I1 (task 8b fix 1): what a warm call starts from follows the last call's needs, never a
    /// ratchet of earlier calls. Pseudo-O triplet 33³ (its SCF density), m = 4, the occupation rule switched
    /// between calls: ↓ (n↓ = 2) has n_occ = 4 under `FermiDirac` (the partly filled p shell; the shared-shell rule's
    /// before plan 2026-11-13 Task 10) and 2 under
    /// `Aufbau`, at the same n↓, so the block memory carries over; ↑ has n_occ = 4 under both. Every call's
    /// last solve has exactly one untested buffer (len = checked + 1), tests the record (n_occ + m + G) but
    /// its last vector — that one too when the record's edge matters (`record_edge_matters` on the returned
    /// levels: the buffer is then above the record) — and holds at least the displayed levels to tol
    /// (head ≥ max(n_occ + m, shown)); a call with the previous call's n_occ holds exactly those — a call
    /// right after n_occ dropped may still hold the previous block, once. No warm call ends on the cap.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm-gpu --release")]
    fn harris_split_follows_n_occ_between_calls() {
        let Some(ctx) = ctx() else { return };
        let (mut s, sys, fd, scf) = solved_pseudo_o(&ctx);
        let ab = ScfSettings { occupations: Occupations::Aufbau, ..fd };
        let tau_scf = tau(fd.tol);
        let m = 4;
        let mut hs = HarrisState::new();
        let mut prev: Option<[usize; 2]> = None;
        let mut failures = Vec::new();
        for (call, (rule, st)) in [("FD", &fd), ("AB", &ab), ("AB", &ab), ("AB", &ab), ("FD", &fd), ("FD", &fd), ("AB", &ab)].into_iter().enumerate() {
            let input = HarrisInput { system: &sys, settings: st, rho_in: [&scf.rho_up, &scf.rho_dn], unoccupied: m };
            let out = s.harris(&mut hs, &input, &mut || false).expect("no failure").expect("not aborted");
            for ch in 0..2 {
                let n_occ = out.n_occ[ch];
                let keep = n_occ + m + UNOCC_GUARD;
                let (shown, _) = qm::orbitals::block_edge(&out.evals[ch], n_occ, out.evals[ch].len(), tau_scf);
                let want_head = (n_occ + m).max(shown);
                let edge = record_edge_matters(&out.evals[ch], n_occ, m, tau_scf);
                let want_checked = if edge { keep } else { keep - 1 };
                let (len, head, checked) = out.split[ch];
                let steady = prev.is_some_and(|p| p[ch] == n_occ);
                println!("call {call} {rule} spin {}: n_occ {n_occ}, shown {shown}, edge {edge}, split (len {len}, head {head}, checked {checked}), want ({want_head}, {want_checked}), steady {steady}, iterations {}, solves {}, at cap {}, cold {}",
                         SPIN[ch], out.iterations[ch], out.solves[ch], out.at_cap[ch], out.cold);
                let mut fail = |what: String| failures.push(format!("call {call} {rule} spin {}: {what}", SPIN[ch]));
                if out.evals[ch].len() != keep { fail(format!("{} kept levels, not n_occ + m + G = {keep}", out.evals[ch].len())); }
                if len != checked + 1 { fail(format!("len {len} ≠ checked {checked} + 1: not exactly one buffer")); }
                if checked < want_checked || head < want_head { fail(format!("head {head} / checked {checked} below {want_head} / {want_checked}")); }
                if steady && (head, checked) != (want_head, want_checked) { fail(format!("steady: (head, checked) = ({head}, {checked}), want ({want_head}, {want_checked})")); }
                if call > 0 && (out.cold || out.at_cap[ch]) { fail(format!("warm call: cold {} at cap {}", out.cold, out.at_cap[ch])); }
            }
            prev = Some(out.n_occ);
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }

    /// Review I2 (task 8b fix 1): the record's last vector is a tested guard, never the untested buffer, so a
    /// degenerate set reaching the record edge is judged on Ritz values whose residual is ≤ `GUARD_TOL` (error
    /// ~r²/δ) and hidden whole. An H atom at the centre of 49³ × 0.25 (a grid point: the cubic symmetry is
    /// exact on the grid, so the p- and d-like box levels are 3-fold degenerate to the f32 rounding),
    /// superposed free-atom ρ_in split equally (so V↑ = V↓), m = 6. Measured spectrum (Ha): 1s −0.16392,
    /// p 0.10735 ×3, s 0.15515, d 0.23802 ×3, 0.28852. ↓ (n↓ = 0) keeps m + G = 8 levels: 1s, p ×3, s and the
    /// d triplet, which starts inside the window (position 5 < m) and ends on the record's last vector, so
    /// `block_edge` hides all three (shown 5, hidden 3); ↑ (n_occ 1) keeps 9, the d triplet inside and 0.28852
    /// last (shown 8). In both channels the window's last row and the first guard are d members, so the
    /// record's edge matters (`record_edge_matters`) and the last record vector is tested under a buffer.
    /// Before the fix the ↓ record's last vector was the buffer at residual 5.3e-3 (its Ritz value 1.4e-5
    /// high — inside τ here, but bounded only by r²/δ ≈ 6e-4 > τ = 1e-4 with δ ≈ 0.05 Ha).
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm-gpu --release")]
    fn harris_record_edge_is_a_tested_guard() {
        let Some(ctx) = ctx() else { return };
        let grid = Grid::cube(49, 0.25);
        let settings = ScfSettings::for_grid(&grid);
        let tau_scf = tau(settings.tol);
        let mut s = GpuSolver::new(ctx, &grid);
        let sys = System::new(vec![Nucleus { pos: [0.0; 3], z: 1 }], 0, None);
        assert_eq!((sys.n_up, sys.n_dn), (1, 0));
        let rho = superposed_h(&grid, &[[0.0; 3]]);
        let mut hs = HarrisState::new();
        let m = 6;
        let out = s.harris(&mut hs, &HarrisInput { system: &sys, settings: &settings, rho_in: [&rho[0], &rho[1]], unoccupied: m }, &mut || false)
            .expect("no failure").expect("not aborted");
        let info = hs.unoccupied().1.expect("a record").clone();
        println!("H atom 49³ m = {m}: iterations {:?}, solves {:?}, at cap {:?}, split {:?}, notes {:?}", out.iterations, out.solves, out.at_cap, out.split, out.notes);
        for (ch, spin) in SPIN.iter().enumerate() {
            println!("  spin {spin}: ε {:.6?}; record residuals {:?}, hidden {}", out.evals[ch],
                     info.residuals[ch].iter().map(|r| format!("{r:.1e}")).collect::<Vec<_>>(), info.hidden[ch]);
        }
        assert_eq!(out.at_cap, [false, false]);
        for (spin, residuals) in SPIN.iter().zip(&info.residuals) {
            let last = *residuals.last().expect("a record");
            assert!(last <= GUARD_TOL, "spin {spin}: the record's last vector has residual {last:.2e} > GUARD_TOL: it is not a tested guard");
        }
        let (up, dn) = (&out.evals[0], &out.evals[1]);
        assert_eq!((up.len(), dn.len()), (1 + m + UNOCC_GUARD, m + UNOCC_GUARD));
        // ↓: the d triplet on the record's last three vectors, one set (spread ≤ τ, apart from the s below),
        // at the ↑ channel's triplet (the same operator; there it lies inside the record)
        let spread = dn[7] - dn[5];
        assert!(spread <= tau_scf, "↓ d triplet spread {spread:.2e} > τ {tau_scf:.0e}: the set is split ({dn:?})");
        assert!(dn[5] - dn[4] > tau_scf, "the triplet is a set of its own ({dn:?})");
        for (i, e) in dn.iter().enumerate().skip(5) { assert!((e - up[6]).abs() <= tau_scf, "↓ #{i} {e} vs ↑ triplet {}", up[6]); }
        assert_eq!(qm::orbitals::block_edge(dn, 0, dn.len(), tau_scf), (5, 3), "the set reaching the record edge is hidden whole");
        assert_eq!(info.hidden[1], 3, "the record hides the triplet");
        assert_eq!(qm::orbitals::block_edge(up, 1, up.len(), tau_scf), (8, 0), "↑: the triplet lies inside the record");
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

    /// Harris under `FermiDirac` on a gapped channel (the scf-stall ← main merge ruling; module doc "Fermi–Dirac"):
    /// H₂ (33³ × 0.25, its FD SCF density, m = 2) solved once under the SCF's `FermiDirac { ELECTRON_KT }` and once
    /// under `Aufbau` on the same ρ_in, each on a fresh `HarrisState`. The gap (σ_g −0.37 Ha, the first box state
    /// +0.13 Ha) puts every FD f within e^{−48} of 0 or 1, so the payload is the `Aufbau` one: the same occupied count
    /// (1 per spin; the FD witness columns are never occupied), f = {1, 0} within 1e-12, the same levels (the same
    /// block — the witness is already among n + G levels, so no growth — and the same LOBPCG: ε within 1e-8), and no
    /// approximate-occupation note.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm-gpu --release")]
    fn harris_fermi_dirac_on_gapped_h2_is_aufbau() {
        let Some(ctx) = ctx() else { return };
        let (mut s, fd, scf) = solved_h2(&ctx);
        assert_eq!(fd.occupations, Occupations::FermiDirac { kt: qm::ELECTRON_KT }, "for_grid is the FD surface");
        let ab = ScfSettings { occupations: Occupations::Aufbau, ..fd };
        let sys = h2();
        let mut run = |settings: &ScfSettings| {
            let mut hs = HarrisState::new();
            let input = HarrisInput { system: &sys, settings, rho_in: [&scf.rho_up, &scf.rho_dn], unoccupied: 2 };
            s.harris(&mut hs, &input, &mut || false).expect("no failure").expect("not aborted")
        };
        let (f, a) = (run(&fd), run(&ab));
        for (rule, o) in [("FD", &f), ("Aufbau", &a)] {
            println!("H₂ Harris {rule}: n_occ {:?}, f {:?}, ε {:?}, split {:?}, solves {:?}, notes {:?}", o.n_occ, o.occ, o.evals, o.split, o.solves, o.notes);
        }
        assert_eq!(f.n_occ, [1, 1], "one occupied level per spin under FD");
        assert_eq!(f.n_occ, a.n_occ);
        for ch in 0..2 {
            assert_eq!(f.occ[ch].len(), a.occ[ch].len(), "spin {ch}: the same kept levels");
            for (i, (x, y)) in f.occ[ch].iter().zip(&a.occ[ch]).enumerate() {
                assert!((x - y).abs() <= 1e-12, "spin {ch} #{i}: f_FD {x} vs f_Aufbau {y}");
            }
            assert_eq!(a.occ[ch][0], 1.0);
            for (i, (x, y)) in f.evals[ch].iter().zip(&a.evals[ch]).enumerate() {
                assert!((x - y).abs() <= 1e-8, "spin {ch} #{i}: ε_FD {x} vs ε_Aufbau {y}");
            }
        }
        assert!(!f.notes.iter().any(|n| n.contains("approximate")), "{:?}", f.notes);
    }

    /// Harris under `FermiDirac` on a partly filled shell (the scf-stall ← main merge ruling): pseudo-O (`All`, 33³ ×
    /// 0.25, off-site at (0.07, −0.03, 0.02)) at M = 3, n↑ = 4, n↓ = 2, converged by the FD SCF (its partial-level stop,
    /// `GPU_PARTIAL_TOL`), then Harris on that density at m = 0. ↓ holds 2s and one electron over the three 2p: f ≈ ⅓
    /// each (the lattice splits the p by ≪ kT). The occupied levels (f ≥ `PARTIAL_OCCUPATION_MIN`) are the SCF's, 4 per
    /// spin, the witness columns never among them; ε within the V-A1 gate 1e-5 Ha; f within the display tolerance that
    /// gate implies through FD's slope, |Δf| ≤ max f(1 − f)·2·1e-5/kT (|Δε_i| + |Δμ|, each ≤ 1e-5; 8.9e-4 at f = ⅓):
    /// the SCF stops at Δρ < 1e-5, so the Harris potential differs from its last one by that much, and a partial f
    /// follows ε at a slope f(1 − f)/kT = 44 per Ha. Σf = n_σ to 1e-6 per channel (the dropped f < 1e-6 tail).
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm-gpu --release")]
    fn harris_fermi_dirac_on_pseudo_o_m3_matches_the_scf() {
        let Some(ctx) = ctx() else { return };
        use crate::ionic::tests::filtered_at;
        use qm::ionic::{Ionic, IonicPolicy, IonicSetup};
        let h = 0.25;
        let grid = Grid::cube(33, h);
        let setup = std::sync::Arc::new(IonicSetup::from_tables(IonicPolicy::All, h, h, std::collections::BTreeMap::from([(8, Ionic::Pseudo(filtered_at(8, h)))])));
        let sys = System::with_ionic(vec![Nucleus { pos: [0.07, -0.03, 0.02], z: 8 }], 0, Some(3), setup);
        assert_eq!((sys.n_up, sys.n_dn), (4, 2));
        let settings = ScfSettings::for_grid(&grid);
        let kt = match settings.occupations { Occupations::FermiDirac { kt } => kt, o => panic!("for_grid is the FD surface: {o:?}") };
        let mut s = GpuSolver::new(ctx, &grid);
        let st = s.solve(&sys, &settings, false).expect("the FD SCF converges").1;
        let scf = s.ks_state(s.state().unwrap());
        println!("pseudo-O M = 3 FD SCF: {} steps, tol_used {:e}, TS {:.6e}", st.iterations, st.tol_used, st.energy.ts);
        let mut hs = HarrisState::new();
        let input = HarrisInput { system: &sys, settings: &settings, rho_in: [&scf.rho_up, &scf.rho_dn], unoccupied: 0 };
        let out = s.harris(&mut hs, &input, &mut || false).expect("no failure").expect("not aborted");
        println!("pseudo-O M = 3 Harris FD: n_occ {:?}, f {:?}, ε {:?}, split {:?}, solves {:?}, notes {:?}", out.n_occ, out.occ, out.evals, out.split, out.solves, out.notes);
        assert!(!out.notes.iter().any(|n| n.contains("approximate")), "the witness is reached: {:?}", out.notes);
        assert_eq!(out.n_occ, [4, 4], "↑ s + 3p; ↓ s + the partly filled p shell");
        let hst = hs.state().unwrap();
        let tol_f = |f: f64| f * (1.0 - f) * 2.0 * 1e-5 / kt;
        for (ch, (h_ev, h_f, s_ev, s_f, n)) in [(&hst.evals_up, &hst.occ_up, &scf.evals_up, &scf.occ_up, sys.n_up), (&hst.evals_dn, &hst.occ_dn, &scf.evals_dn, &scf.occ_dn, sys.n_dn)].into_iter().enumerate() {
            // the SCF's occupied levels by ε: f ≥ PARTIAL_OCCUPATION_MIN (its stored columns include the witnesses)
            let mut occupied: Vec<(f64, f64)> = s_ev.iter().zip(s_f).filter(|(_, f)| **f >= PARTIAL_OCCUPATION_MIN).map(|(e, f)| (*e, *f)).collect();
            occupied.sort_by(|a, b| a.0.total_cmp(&b.0));
            assert_eq!(h_ev.len(), occupied.len(), "spin {ch}: the same occupied levels");
            assert!(s_ev.len() > occupied.len(), "spin {ch}: the SCF's witness columns are not occupied");
            for (i, ((e, f), (es, fs))) in h_ev.iter().zip(h_f).zip(&occupied).enumerate() {
                let (de, df) = ((e - es).abs(), (f - fs).abs());
                println!("  spin {ch} #{i}: ε_H {e:+.8} (f {f:.9}) ε_SCF {es:+.8} (f {fs:.9}) |Δε| {de:.2e} |Δf| {df:.2e} (bound {:.2e})", tol_f(*fs).max(1e-12));
                assert!(de <= 1e-5, "spin {ch} #{i}: |ε_H − ε_SCF| = {de:.3e} > 1e-5 Ha");
                assert!(df <= tol_f(*fs).max(1e-12), "spin {ch} #{i}: |f_H − f_SCF| = {df:.3e} > {:.3e}", tol_f(*fs).max(1e-12));
            }
            let sum = neumaier(h_f.iter().copied());
            assert!((sum - n as f64).abs() <= 1e-6, "spin {ch}: Σf = {sum} for n = {n}");
        }
        for (i, f) in hst.occ_dn[1..4].iter().enumerate() {
            assert!((f - 1.0 / 3.0).abs() <= 1e-2, "↓ p #{i}: f = {f}, expected ≈ ⅓");
        }
    }
}
