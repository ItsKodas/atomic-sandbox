//! Spin-polarized Kohn-Sham SCF on the real-space grid: LSDA, Gaussian-smeared nuclei,
//! Dirichlet box. Atomic units throughout (ħ = mₑ = e = 1; Hartree, Bohr).
//!
//! # Total energy with the fixed grid basis
//! For N = n↑ + n↓ electrons in orthonormal spin orbitals ψ_{iσ} (⟨ψ, ψ⟩ = `Grid::dot`, the
//! h³-weighted L² product, with ψ = 0 outside the box) and spin densities ρ_σ = Σ_i ψ_{iσ}²,
//! ρ = ρ↑ + ρ↓:
//!   E[{ψ}] = T_s + E_ext + E_H + E_xc + E_nn,
//!   T_s   = Σ_{iσ} ⟨ψ_{iσ}, −½∇² ψ_{iσ}⟩        (`stencil::kinetic`, 8th order, Dirichlet),
//!   E_ext = ∫ ρ V_nuc,  V_nuc = −Σ_I Z_I erf(|r−R_I|/σ√2)/|r−R_I|  (`nuclear_potential`, < 0),
//!   E_H   = ½ ∫ ρ V_H,  ∇²V_H = −4πρ with multipole boundary          (`poisson`, > 0),
//!   E_xc  = ∫ n ε_xc[ρ↑, ρ↓]                                           (`xc::lsda_pw92`, < 0),
//!   E_nn  = Σ_{I<J} Z_I Z_J/|R_I − R_J|   (point nuclei: only the electrons see them smeared).
//! Signs: ρ is the electron *number* density (positive); V_nuc is already the potential energy
//! of an electron (the −1 charge is folded in); E_H = ½∫∫ρρ′/|r−r′| is positive whatever the
//! sign of the carriers. Stationarity of E under orthonormality constraints gives the KS
//! equations H_σ ψ_{iσ} = ε_{iσ} ψ_{iσ}, H_σ = −½∇² + V_eff,σ, V_eff,σ = V_nuc + V_H + v_xc,σ,
//! and the usual identity Σ_{iσ} ε_{iσ} = T_s + ∫ρ V_eff (used only as a cross-check).
//!
//! # Why the energy is evaluated from the *output* density and orbitals
//! Each SCF step builds H_σ from an input density ρ_in and returns orbitals ψ (hence ρ_out).
//! E[{ψ}] above is the exact discrete functional evaluated at a legitimate trial point
//! (the ψ are orthonormal), so it is a variational upper bound on the discrete ground-state
//! energy, and because the functional is stationary at the self-consistent orbitals its error
//! is *second* order in (ψ − ψ_scf). A mixed evaluation (T_s from ψ but E_H, E_xc from ρ_in,
//! or Σε with the double-counting terms of ρ_in) is first order in ρ_out − ρ_in. The price is
//! one extra Poisson solve for V_H[ρ_out] after the loop; the returned `KsState::v_h` is that
//! potential, consistent with `rho_up/rho_dn` so a warm start is self-consistent.
//!
//! # The loop
//!   V_eff,σ = V_nuc + V_H[ρ_in] + v_xc,σ[ρ_in↑, ρ_in↓]
//!   lowest n_σ eigenpairs of H_σ per spin (LOBPCG, warm from the previous orbitals, Jacobi
//!     preconditioner shifted by the previous lowest eigenvalue)
//!   ρ_out,σ = Σ_i ψ_{iσ}²,   r_σ = ρ_out,σ − ρ_in,σ,   Δρ = ∫ (|r↑| + |r↓|)
//!   converged if Δρ < tol, else ρ_in ← Pulay/DIIS mix of the history (below).
//!
//! # Mixing: why not linear, and Pulay/DIIS
//! Linear mixing ρ_in ← ρ_in + α r, linearised about the fixed point with J = ∂ρ_out/∂ρ_in,
//! contracts the error by (1 − α) + αJ per step per mode: stable iff every eigenvalue of J
//! exceeds 1 − 2/α, and the step count ≈ ln(Δρ₀/tol)/|ln(1 − α + αJ)|. The *rate* is fixed by
//! α and J — a warm start only lowers Δρ₀ (measured for H at α = 0.3: ratio 0.70–0.72 per step
//! from any start, 38 cold / 28 warm steps), and a soft mode with J ≈ 1 (Li 2s/2p near
//! degeneracy at σ = 0.25) never converges. Pulay's DIIS (Chem. Phys. Lett. 73, 393 (1980))
//! removes the rate limit: keep the last m ≤ `diis_history` pairs (ρ_in,i, r_i), with the two
//! spin channels stacked into one vector so a single coefficient set applies to both, and look
//! for the combination ρ̄ = Σ c_i ρ_in,i, Σ c_i = 1, whose *predicted* residual is smallest.
//! Near the fixed point r is affine in ρ_in, so r(ρ̄) ≈ Σ c_i r_i, and minimising
//! ‖Σ c_i r_i‖² subject to Σ c_i = 1 (Lagrange multiplier λ) gives the bordered system
//!   [ B   1 ] [ c ]   [ 0 ]          B_ij = ⟨r_i, r_j⟩ = ⟨r_i↑, r_j↑⟩ + ⟨r_i↓, r_j↓⟩ (`Grid::dot`),
//!   [ 1ᵀ  0 ] [ −λ ] = [ 1 ]
//! solved by LU (`nalgebra::DMatrix::lu`). The next input is the optimal combination plus a
//! linear-mixing step along its (minimised) residual,
//!   ρ_in,new = Σ c_i (ρ_in,i + α r_i) = ρ̄ + α r̄,   α = `settings.mixing`,
//! which with m = 1 is exactly linear mixing (c = [1]): the first step needs no special case.
//! Charge: ∫r_i = 0 for every entry (ρ_in and ρ_out both integrate to n_σ) and Σ c_i = 1, so
//! ∫ρ_in,new = n_σ before clamping. The extrapolation can overshoot below zero where ρ is tiny
//! (c_i of both signs); those points are clamped to 0, which changes the charge of ρ_in by a
//! negligible amount (ρ_out, from orbitals, stays exact) and keeps `lsda_pw92` on physical
//! input. Conditioning: as the residuals shrink they become nearly parallel and B (entries
//! ~Δρ²) nearly singular; B is scaled by its largest diagonal and given a relative diagonal
//! regularisation of 1e-12, and if the solve still fails (or returns non-finite c) the oldest
//! entry is dropped and the solve retried — down to m = 1, i.e. linear mixing.
//! ∫ρ_in,σ = n_σ holds for the cold guess up to the 1e-8 aliasing floor of `deposit_charge`
//! at σ = 1, and exactly for ρ_out (orbitals orthonormal in `Grid::dot`, and `Grid::integrate`
//! is the same sum).
//!
//! # Starting point
//! Cold: ρ_in = `deposit_charge(σ = 1)` (a Z-weighted Gaussian blob of width 1 Bohr on each
//! nucleus, i.e. the right total charge in roughly the right places) scaled to N and split
//! n↑/N : n↓/N; orbitals are a deterministic pseudo-random field under a Gaussian envelope on
//! the nuclei (LOBPCG orthonormalises them); V_H = 0 (cold CG). Warm: a previous `KsState`
//! converged on the same grid (`KsState::grid == grid`) whose orbital counts match the system —
//! the orbitals, eigenvalues, densities and V_H are reused; if the grid or the counts do not
//! match the warm state is ignored and the cold path runs.
//! An empty spin channel (hydrogen n↓ = 0, or H⁺ with no electrons at all) is a zero density
//! and an empty orbital list: `lsda_pw92` takes the ζ = ±1 branch with v_xc,↓ = 0 at every
//! point, `density_from` of no orbitals is exactly zero, and the eigensolver is skipped.
//!
//! # Electron count and multiplicity
//! N = Σ Z_I − charge. Multiplicity M = 2S + 1 means M − 1 unpaired electrons: n↑ = (N + M − 1)/2,
//! n↓ = (N − M + 1)/2, which requires 1 ≤ M ≤ N + 1 and N − (M − 1) even. `None` picks the
//! lowest multiplicity (M = 1 for even N, 2 for odd N). That is a *guess*, not a rule: Hund's
//! first rule makes e.g. C, N, O high-spin, and nothing here knows about shells, so callers
//! that care must pass M (the plan exposes it in the UI). Nothing per element enters: Z only.
//!
//! # Box-size warning
//! The Dirichlet wall (ψ = 0 on the ghost points, V_H from the multipole expansion) confines
//! the system. If max |ρ| on the six faces exceeds 1e-5 e/Bohr³ the energy is being raised by
//! the wall and a warning is recorded (not an error — a user can shrink the box deliberately).
//!
//! # σ → 0 extrapolation (`extrapolate_sigma`)
//! `nuclear.rs` derives the smearing error of the energy: ΔE = 2π Z ρ_e(R) σ² + O(σ³) + O(σ⁴);
//! no odd power below σ³ (the linear term of ρ_e about R vanishes by symmetry) and σ³ from the
//! density cusp. For hydrogen the first-order shift has the closed form (a = σ√2, p = 2):
//!   ΔE₁(σ) = 4∫₀^∞ d e^{−2d} erfc(d/a) dd = 1 − 2a/√π + (2a² − 1) e^{a²} erfc(a)
//!         = 2σ² − (16√2/3√π) σ³ + 6σ⁴ − (128√2/15√π) σ⁵ + …
//! (from ∫₀^∞ e^{−pd} erfc(d/a) dd = (1 − e^{p²a²/4} erfc(pa/2))/p, differentiated in p, and
//! e^{x²}erfc(x) = Σ (−x)ⁿ/Γ(n/2 + 1)), giving 0.0766, 0.1010, 0.1262, 0.1516 Ha at
//! σ = 0.25, 0.30, 0.35, 0.40. The series alternates with ratio ≈ 2σ, so at σ ≳ 0.25 the
//! σ⁵ and higher terms are several mHa and the truncated model E = a + bσ² + cσ³ + dσ⁴ fitted
//! through four points has a bias in a: +1.73 mHa on the exact ΔE₁ over this σ range (a test
//! pins it). The fit is the 4×4 Vandermonde-like solve [1 σ² σ³ σ⁴] x = E by LU; a is a
//! linear combination of the four energies with weights (17.7, −36.8, 27.0, −6.9), |w|₁ ≈ 88,
//! so σ-dependent noise in E(σ) of δ gives up to 88δ in a: the SCF, eigen and Poisson
//! tolerances must hold E(σ) to ≪ 1e-4 Ha for a 10 mHa extrapolation. Choosing σ ≥ h keeps
//! the smeared density resolved; the grid error itself varies smoothly with σ and is partly
//! fitted away, partly extrapolated — which is why the point-nucleus test is a coarse-grid
//! sanity check here and the real oracle lives in `validate` on a finer grid.
//!
//! # Precision
//! All grid sums go through `Grid::integrate`/`Grid::dot` (Neumaier). Densities, potentials
//! and mixing are pointwise.
//!
//! # Ionic species (M7; spec §5.2, §5.5, §5.6)
//! `System.ionic == None` is everything above, the same code path bit for bit
//! (`none_policy_is_bitwise_today` pins the energy and force bits measured on the code before M7).
//! With `Some(setup)` (an [`IonicSetup`], `crate::ionic`) the electrons are the *valence* electrons
//! of the pseudised species plus all electrons of the all-electron ones, N = Σ_I q_I − charge
//! (q = Z_v pseudised, Z all-electron; `valence_electrons`), and with ρ = ρ↑ + ρ↓ that density,
//!   H_σ = −½∇² + V_loc + V_NL + V_H[ρ] + v_xc,σ[ρ↑ + ½ρ_pc, ρ↓ + ½ρ_pc],
//!   E   = T_s + ∫ρ V_loc + E_NL + E_H[ρ] + E_xc[ρ↑ + ½ρ_pc, ρ↓ + ½ρ_pc] + E_ii,
//! where V_loc = Σ_I V_LR,I + Σ_{pseudised} V_SR,I (`nuclear::ionic_local_potential`; an all-electron
//! species' term is exactly today's smeared nucleus at σ_AE = `settings.sigma`), V_NL = Σ E_KB|β⟩⟨β|
//! and E_NL = Σ_{iσ} Σ_{I,lm} E_KB c_lm[ψ_iσ]² (`nonlocal`), ρ_pc = Σ_I ρ_pc,I the model partial core
//! (unpolarised: half per spin, spec §4.5; it enters E_xc only, never V_H or N), and
//! E_ii = Σ_{I<J} q_I q_J/R_IJ (`nuclear::ionic_repulsion`). Stationarity: ∂E_NL/∂ψ = 2Σ E_KB c β =
//! 2V_NLψ, and since n_σ = ρ_σ + ½ρ_pc with ρ_pc fixed, ∂E_xc/∂ρ_σ = v_xc,σ(n↑, n↓); so H_σ above is
//! the gradient of E, and everything said above about evaluating E from the output orbitals holds
//! unchanged. The `Energy` struct is unchanged: `external` = ∫ρ V_loc + E_NL (the whole ion–electron
//! energy), `xc` includes the partial core, `nuclear` = E_ii with valence charges; `total` is the
//! *valence* total energy of a pseudised system (the frozen cores' own energy is not in it).
//! The cold guess deposits each centre's charge q_I (σ = 1 Bohr blobs, `nuclear::deposit_centres`).
//! Without a partial core anywhere in the scene (the no-core branch: H/He under `All`, or every
//! species all-electron) XC is `lsda_pw92` on ρ↑, ρ↓ directly.
//! Guards: `scf` asserts setup.h = grid.h and setup.σ_AE = settings.sigma (a table filtered for another
//! h, or a mixed scene's all-electron species at another σ, is a programming error); a warm `KsState`
//! is reused only if it was converged under the same physics (`KsState.ionic`, `None` vs `None` or
//! `IonicSetup::same_physics`) — the "converged on a different σ" guard of spec §5.5 becomes a
//! setup-equality guard; otherwise the cold path runs. Core overlaps (spec §5.6) are reported in
//! `ScfStats.core_overlaps` and as a warning per pair; the solve proceeds. LSDA domain (spec §4.5,
//! T8(c)): the points of the output density with n_σ = ρ_σ + ½ρ_pc < 0, where `xc::exc_point`'s
//! clamp engages, are counted and warned about (T8(c) certifies the domain radially; this makes the
//! "never engages" claim checkable in 3-D, Task 6 review S3).
//!
//! # Occupations (spec 2026-11-13 §4–5; `DegenerateShells`: pseudopotential spec §5.7, M7 Task 6 review, ruling A)
//! `ScfSettings.occupations` picks how each spin channel σ (N_σ electrons) fills its levels. Three schemes:
//! - **`Aufbau`** (the default of `for_grid` until plan Task 7 flipped it to `FermiDirac`; the providers' choice under
//!   `IonicPolicy::None` until plan Task 9 switches it; afterwards only the kT → 0 reference) is everything above: each
//!   spin fills its N_σ lowest levels, k_σ = N_σ vectors are solved, not one operation differs from the solver before
//!   M7.
//! - **`FermiDirac { kt }`** (spec 2026-11-13; the default of `for_grid` since plan Task 7, at the production kT =
//!   `ELECTRON_KT` = 5 mHa, T_el = kT/k_B ≈ 1 579 K): the
//!   Mermin–Kohn–Sham ensemble at fixed N_σ per spin (the fixed-spin-moment ensemble, so a forced or "auto"
//!   multiplicity keeps its meaning), with ground-state LSDA xc. All of it is in atomic units (ε, μ, kT in Ha, k_B = 1).
//!   - *The rule* (`occupations::fermi_dirac`, its module doc has the derivation): on every SCF step, from that step's
//!     levels ε_iσ of H[ρ_in],
//!       f_iσ = 1/(1 + e^{x}),   x = (ε_iσ − μ_σ)/kT,   Σ_i f_iσ = N_σ,
//!     μ_σ by two tie-directed bisections reported at their midpoint (in a gapped channel Σf == N_σ exactly over a
//!     plateau of μ, and μ is the plateau midpoint). The f are recomputed from each step's levels and never mixed (the
//!     mixer sees densities only); ρ_σ = Σ_i f_iσ ψ_iσ². An empty channel is not solved: no levels, μ_σ = `None`.
//!   - *The witness* (spec §4.2): FD needs every level with a non-negligible f, so each occupied channel grows its
//!     block by one vector at a time, re-solving every level, until its highest level ε_kσ satisfies
//!       ε_kσ ≥ ε_(N_σ) + `WITNESS_GAP`   and   ε_kσ − μ_σ ≥ `WITNESS_TAIL_KT`·kT        (`witness_reached`)
//!     (ε_(N_σ) the N_σ-th lowest; at 36 kT the first unsolved level would hold f < 2.4e-16, below f64 resolution next
//!     to 1). At most `MAX_WITNESS_GROWTH` extra vectors, else `QmError::Numerical`. A cold start solves N_σ + 1; a warm
//!     start keeps its grown block; growth uses `DegenerateShells`' seeds. Under `Aufbau` an all-electron scene solved
//!     exactly N_σ vectors, so FD costs it at least one extra vector per occupied channel.
//!   - *The energies* (spec §4.3): E (`Energy::internal`) = T_s + E_ext (incl. E_NL) + E_H + E_xc + E_nn as above with
//!     T_s = Σ f⟨ψ|−½∇²|ψ⟩ and E_NL = Σ f Σ E_KB c², all from the output orbitals and ρ_out of the final step;
//!       S = Σ_σ Σ_i s(f_iσ),   s(f) = −[f ln f + (1 − f) ln(1 − f)] ≥ 0,
//!     evaluated from x (`fd_entropy`: ln_1p(e^{−|x|}) + |x|·min(f, 1 − f), the 0·ln 0 = 0 limit built in, no
//!     cancellation at f → 0 or 1), summed with Neumaier per channel and over the two channels; `Energy::ts` = kT·S of
//!     the final step's f, and `Energy::total` = F = E − kT·S, the Mermin free energy, bitwise `internal − ts`. The
//!     per-iteration Harris–Foulkes energy is F_HF = Σfε + double counting − kT·S of that step's f ("Progress").
//!   - *Stationarity, and the forces* (spec §4.4): F[{ψ}, {f}] at fixed N_σ is stationary in ψ (the KS equations with
//!     the f-weighted density) and in f: ∂F/∂f_iσ = ε_iσ − μ_σ + kT ln(f_iσ/(1 − f_iσ)) = 0 is exactly the FD
//!     condition, μ_σ the multiplier of Σ_i f_iσ = N_σ. So dF/dR = ∂F/∂R|_{ψ,f}: S has no explicit R, and the force is
//!     the Hellmann–Feynman force of the f-weighted density and projector sums `forces` already computes — −dF/dR,
//!     not −dE/dR (they differ by d(TS)/dR: 0 when gapped, up to a few mHa/Bohr where a level crosses μ).
//!   - *Why it replaces `DegenerateShells`*: integer aufbau has no well-posed fixed point for a partly filled
//!     degenerate shell (below), and the two-phase equal-share rule that answered it decides by thresholds on a
//!     converged spread (τ, τ_c), so it needs a second convergence and a release that restarts the mixer and the
//!     stall watch. FD's f are a continuous function of the levels: there is no decision to depend on the path, the
//!     solve converges to a stationary point of F, and an exactly degenerate partly filled shell gets the same
//!     equal-weight ensemble (so the same E), now with −TS (4 kT ln 2 for the H₄ square). Plan Task 10 deletes the
//!     two phases.
//!   - *What the state is* (rule 7, spec §4.7). Gapped (≥ 74 kT around μ): every f is 1.0 or ≤ e^{−36}, S ≤ 1e-14,
//!     F = E. A smaller frontier gap Δ with μ at mid-gap carries TS ≈ 2kT(1 + y)e^{−y}, y = Δ/(2kT), per channel
//!     (4e-10 Ha at 0.2 Ha, 5e-6 Ha at 0.1 Ha). Near-degenerate frontier ("pinned") levels put F below E by
//!     O(kT) per pinned level. S is the entropy of the KS occupations, not the molecule's physical electronic entropy.
//!     Smearing converges to the nearest stationary state, not the lowest (a spin-symmetric state can lie above a
//!     broken-symmetry one: spec §4.7(e)); one μ_σ spans the box, so separated fragments with aligned frontier levels
//!     share charge or spin (LSDA delocalisation, §4.7(h)).
//!   - *Diagnostics*: `ScfStats.entropy` (S), `mu` (μ_σ), `partial` (min(f, 1 − f) ≥ `PARTIAL_OCCUPATION_MIN`, i.e.
//!     |ε − μ| ≲ 13.8 kT).
//!   - *Warm start* (spec §5): `KsState.occupations` records the scheme with its kT and the guard compares it with
//!     `==`, so a state converged at another kT or under another scheme never seeds the solve (the cold path runs).
//! - **`DegenerateShells { tau }`** (the providers' choice under `CoresOnly` and `All` until plan Task 9; removed in
//!   plan Task 10), τ = `DEGENERACY_TAU`:
//!   - Why: with integer aufbau a partly filled degenerate shell has no well-posed fixed point. For
//!     pseudo-O at h = 0.20, M = 1 (two of three p per spin) has no aufbau-consistent state (the empty
//!     p falls below an occupied one in their own potential: Δρ 1.2–2.7 for 60 steps), and M = 3's one
//!     minority p is aufbau-consistent but its orientation is a zero mode of the fixed-point set in the
//!     continuum and nearly so on the grid (|p_n̂|² has angular content l = 0, 2 only, the lattice's
//!     anisotropy starts at l = 4; measured contraction 0.9991 per step, Δρ stalled at 7e-5). No mixer
//!     or eigensolver setting removes a zero mode.
//!   - Rule (state-defined; Task 6 re-review, ruling 1): a level set at ε_F,σ is shared iff, in the
//!     self-consistent state in which it is shared, its eigenvalue spread is < τ. Two phases; every
//!     solve, cold or warm, starts in the first, so the outcome is a property of the converged state,
//!     not of the SCF path:
//!     1. Candidates. For each spin with N_σ > 0 LOBPCG solves k_σ = N_σ + g_σ states, g_σ grown one
//!        at a time (`MAX_WITNESS_GROWTH`) until the gap witness satisfies ε_{k_σ} ≥ ε_F,σ + τ_c,
//!        ε_F,σ the N_σ-th lowest level; all k_σ are converged to eig_tol (a witness stopped at the
//!        LOBPCG cap shows in `max_eigen_residual`). The levels within τ_c of ε_F,σ form the cluster
//!        C_σ; if it straddles ε_F,σ (reaches rank N_σ) it is shared equally, f = (N_σ − n_below)/|C_σ|
//!        (`degenerate_shell_occupations`), else the f are aufbau's. The SCF runs to `tol`.
//!     2. Decision. A shared cluster whose spread is ≥ τ is released: its levels are split wherever a
//!        neighbour gap is ≥ τ and filled by aufbau over the groups (`released_occupations`). The
//!        pattern (f by rank) is then frozen for the rest of the solve — no re-sharing, so no
//!        oscillation — and the SCF runs to `tol` again. Releases are reported (`ScfStats.released`).
//!
//!     A closed shell's cluster ends at N_σ and is never shared; a second convergence happens only when
//!     ε_F lies within τ_c of a genuinely split set. Then ρ_σ = Σ f_i ψ_i², T_s = Σ f_i⟨ψ_i|−½∇²|ψ_i⟩,
//!     E_NL = Σ f_iσ Σ E_KB c_lm². The f are constants of the state (fixed by the rule, not varied), and E is
//!     stationary in ψ at those fixed f, so the energy above and the forces hold with the weights f_iσ
//!     (`KsState.occ_up/occ_dn`).
//!   - Why two phases: a one-shot test on each step's levels is path-dependent. At a low-symmetry site
//!     the cold-guess orbitals (compact, ε ≈ −1.05 Ha for O 2p) are split by the lattice beyond 1e-4 at
//!     step 1 (1.5e-4 at (0, h/2, h/2), h = 0.20), and an integer occupation then self-polarises
//!     (11.6–11.8 mHa): the round-1 one-shot rule left a lone O 4.1 mHa lower at that site than at the
//!     cubic ones, while the same site's shared state has a spread of only 5.3e-5.
//!   - What it is (rule 7): the equal-weight ensemble of the degenerate shell, for an atom the
//!     spherical atom — the object the radial generator builds (spec §4.1) and every T3 gate compares.
//!     LSDA lowers the non-spherical integer state by a few mHa (the symmetry dilemma of approximate
//!     functionals); that gap is measured and reported (spec §6.3), never hidden.
//!   - τ = `DEGENERACY_TAU` = 1e-3 Ha: 10× the largest converged lattice spread of a symmetry-degenerate
//!     shared level at any site and certified h (measured 5.3e-5 Ha for O 2p at the tetragonal site,
//!     h = 0.20; validated per species at h_req by `validate`'s `degenerate_spread_at_low_symmetry_sites`),
//!     and below k_B·300 K ≈ 0.95 mHa (levels this close are thermally co-occupied at the sandbox's
//!     temperatures; sharing a genuinely split pair with δ < τ moves E by ≤ δ/2 at first order).
//!     τ_c = `WITNESS_GAP` = 0.1 Ha (the value of the former `CANDIDATE_TAU`) exceeds the splitting that integer
//!     occupation induces in a partly filled set (O: 11.6–11.8 mHa), so neither a transient nor a warm state can
//!     decide the pattern. Solver constants; they change no table.
//!   - Molecules: closed shells and integer-per-spin open shells (H₂O, CH₄, the O₂ triplet) get aufbau's
//!     f; symmetry-enforced degenerate partly filled sets get the ensemble. A set whose converged spread
//!     crosses τ during dynamics switches between shared and integer occupation (`ScfStats.fractional`
//!     reports every shared cluster of the final state, `ScfStats.released` every release).
//!   - The warm-start guard also requires the same scheme (`KsState.occupations`), so an integer,
//!     broken-symmetry state never seeds a shared solve.
//!
//! # The stall stop (`StallWatch`, `ScfSettings::stall_window`)
//! A solve that cannot converge used to run to `max_iter` (200) before failing, and the providers'
//! R15 cold retry doubles that: 400 iterations for one sampling-pass geometry whose Δρ oscillates
//! (a half-broken bond whose near-degenerate occupations flip between iterations). The rule, the
//! same in both solvers (`qm_gpu::scf` uses this `StallWatch`):
//!   anchor ← Δρ_i and a ← i  whenever Δρ_i ≤ ½·anchor (the first finite Δρ sets it);
//!   stop with `QmError::ScfStalled` when i − a ≥ W  (W = `stall_window`, 0 = off)
//!                                   and best Δρ ≥ L = `SCF_STALL_LEVEL_PER_ELECTRON` · N,
//! N the electrons the solve carries (n↑ + n↓: the valence electrons under a pseudopotential
//! setup) — the Δρ = ∫|ρ_out − ρ_in| L1 norm scales with N. The level is what separates the two
//! kinds of long plateau measured (`SCF_STALL_LEVEL_PER_ELECTRON`): a solve oscillating at a Δρ
//! of order one never converges, while a slow oscillation far below it (stretched spin-symmetric
//! H₂) can still converge after 150 iterations. A plateau below L is never stopped: it converges
//! or runs to the cap (the old price).
//! checked after the convergence test of iteration i (and never on an iteration whose Δρ is already
//! below tol), so it only ever ends a solve that would otherwise have continued: a converged result
//! is bitwise what it was without the rule, and the iteration cap stays as the backstop. The
//! converse holds too and is what the oracles rely on: a solve the rule ends might have converged
//! later without it — the rule turns slow convergence into an error, it never turns one number into
//! another. A non-finite Δρ is not a stall: it ends the solve at once as `ScfNotConverged`. In words: the best Δρ (the anchor is always the best Δρ at
//! the time of the last 2× gain) has not improved by a factor ≥ 2 within the last W iterations.
//! Slow but steady progress never stops: any rate faster than 2× per W iterations re-anchors.
//! At the `DegenerateShells` release (phase 2 restarts the convergence with Δρ = ∞) the watch
//! restarts too. W = `SCF_STALL_WINDOW`; its measurement is cited there and in `validate`'s
//! `references.md` ("SCF stall stop").
//!
//! # Progress (`scf_observed`)
//! An observer gets one `ScfProgress` per iteration, after Δρ and before the convergence test: the
//! iteration, Δρ, the stall watch's state and the Harris–Foulkes energy of the iteration's input
//! density (Harris, PRB 31, 1770 (1985); Foulkes & Haydock, PRB 39, 12520 (1989)),
//!   E_HF[ρ_in] = Σ_{iσ} f_iσ ε_iσ − ½∫ρ_in V_H[ρ_in] − Σ_σ ∫ρ_in,σ v_xc,σ[ρ_in] + E_xc[ρ_in] + E_nn,
//! with ε the eigenvalues of H[ρ_in] just solved (the nonlocal energy is inside Σfε, the partial
//! core inside v_xc and E_xc as in H). Σfε = T_s + ∫ρ_out(V_loc + V_H[ρ_in] + v_xc[ρ_in]) + E_NL, so
//! at ρ_out = ρ_in it is the Kohn–Sham energy above, and its error is second order in ρ_out − ρ_in:
//! a display of the iteration's energy (its change per iteration), never the reported energy.
//! Under `FermiDirac` the display is F_HF = E_HF − kT·S, S the entropy of that iteration's f ("Occupations"), so at
//! ρ_out = ρ_in it is the reported F; under the other schemes S = 0 and nothing changes.
//! Computed only when an observer is attached; `scf` attaches none.

use crate::eigen::{jacobi_precond, lobpcg, EigenStats, LocalHamiltonian};
use crate::error::QmError;
use crate::grid::{neumaier, Grid};
use crate::ionic::{core_overlaps, IonicSetup, CORE_OVERLAP_WARNING};
use crate::nonlocal::{NonlocalOp, PseudoHamiltonian};
use crate::nuclear::{centres, deposit_centres, has_partial_core, ionic_local_potential, ionic_repulsion, partial_core_field, Centre};
use crate::nuclear::{deposit_charge, nuclear_potential, nuclear_repulsion, Nucleus};
use crate::poisson::{hartree_energy, PoissonSolver, PoissonStats};
use crate::stencil::kinetic;
use crate::xc::{lsda_pw92, XcResult};
use nalgebra::{DMatrix, DVector, Matrix4, Vector4};
use std::sync::Arc;

// The Fermi–Dirac rule lives in `occupations`; re-exported so `qm::scf::fermi_dirac` resolves (spec §5).
pub use crate::occupations::*;

/// How each spin's electrons fill its Kohn–Sham levels (module doc, "Occupations"). `PartialEq` compares kT, so the
/// warm-start guard (`KsState.occupations == settings.occupations`) refuses a state converged at another kT.
/// `DegenerateShells` is removed in plan Task 10.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Occupations { Aufbau, DegenerateShells { tau: f64 }, FermiDirac { kt: f64 } }

/// τ of `Occupations::DegenerateShells` (Ha; module doc, "Occupations"): the decision threshold on a shared
/// cluster's converged spread. A solver constant.
pub const DEGENERACY_TAU: f64 = 1e-3;

/// The most extra vectors a spin channel may grow to find the gap witness (a method literal and a cost cap only: the
/// largest symmetry degeneracy of a Z ≤ 18 open shell is 3, plus box continuum states above it; spec 2026-11-13 §4.2
/// measured at most 15 under `FermiDirac` at kT = 5 mHa in a 14 Bohr box).
///
/// Measured (plan 2026-11-13 Task 7 fix rounds 1–2): a cold all-electron H₂O at 97³ × 0.25 (a 24.5 Bohr box) reaches
/// 16 on its 3rd and 4th SCF steps. Their spectrum is far from self-consistent: Δρ_prev 4.8 and 2.3, ε_N = −0.152 /
/// −0.132 Ha against ≈ −0.26 converged, μ ≈ −0.067 / −0.053. So the window top μ + 36 kT ≈ +0.11–0.13 Ha holds
/// 15–16 box states. The block stays orthonormal (max |⟨ψ_i|ψ_j⟩| ≤ 2.9e-8, Ritz gaps ≥ 2.4e-4 Ha): no collapsed
/// columns. A cap of 32 passed step 4 (19 extras) but did not converge within a 15-minute GPU run, because every growth
/// re-solves the ~20-vector block, so the cap stays 16. The cap and its memory budget (`providers::ks`, which counts it)
/// are decided in plan Task 9b. `pub` for `qm-gpu::scf`, which implements the same rule.
pub const MAX_WITNESS_GROWTH: usize = 16;

/// A shared (fractionally occupied) cluster of one spin channel: spin (0 ↑, 1 ↓), the index of its first
/// level in ascending order, its size, the occupation f of each level and its eigenvalue spread (Ha).
#[derive(Clone, Debug, PartialEq)]
pub struct FractionalShell { pub spin: u8, pub first: usize, pub count: usize, pub f: f64, pub spread: f64 }

/// Spec §5.7's phase-1 (candidate) occupations of `evals` (any order; f is returned in the same order) for `n`
/// electrons with the window `tau` (τ_c in the solver): `None` if the gap witness (the highest level) lies within
/// `tau` of ε_F (the n-th lowest) — solve another vector. The levels within `tau` of ε_F form the cluster; it is
/// shared equally iff it straddles ε_F (reaches rank n). The second value is the shared cluster (first, count, f,
/// spread), `None` when the f are aufbau's.
pub fn degenerate_shell_occupations(evals: &[f64], n: usize, tau: f64) -> Option<(Vec<f64>, Option<(usize, usize, f64, f64)>)> {
    if n == 0 { return Some((vec![0.0; evals.len()], None)); }
    let k = evals.len();
    assert!(k > n, "the rule needs at least one vector above the {n} electrons ({k} given)");
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| evals[a].total_cmp(&evals[b]));
    let e_f = evals[order[n - 1]];
    if evals[order[k - 1]] - e_f < tau { return None; }
    let first = order.iter().position(|&i| (evals[i] - e_f).abs() < tau).expect("ε_F is in its own cluster");
    let count = order[first..].iter().take_while(|&&i| (evals[i] - e_f).abs() < tau).count();
    let fc = (n - first) as f64 / count as f64;
    let mut f = vec![0.0; k];
    for (rank, &i) in order.iter().enumerate() {
        f[i] = if rank < first { 1.0 } else if rank < first + count { fc } else { 0.0 };
    }
    let shell = (first + count != n).then(|| {
        let c = &order[first..first + count];
        let spread = c.iter().map(|&i| evals[i]).fold(f64::MIN, f64::max) - c.iter().map(|&i| evals[i]).fold(f64::MAX, f64::min);
        (first, count, fc, spread)
    });
    Some((f, shell))
}

/// Spec §5.7's phase-2 release of a shared cluster: its levels `sorted` (ascending) are split wherever a neighbour
/// gap is ≥ `tau`, and `electrons` are filled by aufbau over the groups, shared equally within a partly filled group.
/// Returns f per level (same order).
pub fn released_occupations(sorted: &[f64], electrons: f64, tau: f64) -> Vec<f64> {
    debug_assert!(sorted.windows(2).all(|w| w[0] <= w[1]), "levels must be ascending");
    let mut groups: Vec<(usize, usize)> = Vec::new();
    let mut start = 0;
    for i in 1..=sorted.len() {
        if i == sorted.len() || sorted[i] - sorted[i - 1] >= tau { groups.push((start, i)); start = i; }
    }
    let (mut f, mut left) = (vec![0.0; sorted.len()], electrons);
    for (a, b) in groups {
        let size = (b - a) as f64;
        let take = left.min(size);
        for x in &mut f[a..b] { *x = take / size; }
        left -= take;
    }
    f
}

/// Nuclei, the number of electrons in each spin channel, and the ionic setup (`None`: today's
/// all-electron solver; module doc "Ionic species (M7)").
pub struct System { pub nuclei: Vec<Nucleus>, pub n_up: usize, pub n_dn: usize, pub ionic: Option<Arc<IonicSetup>> }

/// (n↑, n↓) for `total` electrons: `multiplicity` = 2S+1; `None` ⇒ lowest guess (paired; one unpaired
/// if N odd). Panics if N < 0 or the multiplicity is impossible for N (see module doc).
fn spin_split(total: i64, multiplicity: Option<u32>) -> (usize, usize) {
    assert!(total >= 0, "negative electron count");
    let n = total as usize;
    let unpaired = match multiplicity {
        Some(m) => {
            assert!(m >= 1 && (m as usize - 1) <= n && (n - (m as usize - 1)) % 2 == 0, "bad multiplicity");
            m as usize - 1
        }
        None => n % 2,
    };
    let n_dn = (n - unpaired) / 2;
    (n_dn + unpaired, n_dn)
}

/// Σ q_I − charge: q = Z for all-electron species (and without a setup), Z_v for pseudised ones.
pub fn valence_electrons(nuclei: &[Nucleus], ionic: Option<&IonicSetup>, charge: i32) -> i64 {
    nuclei.iter().map(|n| ionic.map_or(n.z as f64, |s| s.valence_charge(n.z)).round() as i64).sum::<i64>() - charge as i64
}

impl System {
    /// N = ΣZ − charge. `multiplicity` = 2S+1; `None` ⇒ lowest guess (paired; one unpaired if N odd).
    /// Panics if N < 0 or the multiplicity is impossible for N (see module doc).
    pub fn new(nuclei: Vec<Nucleus>, charge: i32, multiplicity: Option<u32>) -> System {
        let (n_up, n_dn) = spin_split(valence_electrons(&nuclei, None, charge), multiplicity);
        System { nuclei, n_up, n_dn, ionic: None }
    }

    /// N = Σ q_I − charge (spec §5.2: Z_v for pseudised species). Panics like `new` on an impossible
    /// count (providers pre-validate) and for a Z not in the setup.
    pub fn with_ionic(nuclei: Vec<Nucleus>, charge: i32, multiplicity: Option<u32>, ionic: Arc<IonicSetup>) -> System {
        let (n_up, n_dn) = spin_split(valence_electrons(&nuclei, Some(&ionic), charge), multiplicity);
        System { nuclei, n_up, n_dn, ionic: Some(ionic) }
    }
}

/// SCF controls. `sigma`: nuclear smearing width (Bohr); `tol`: on Δρ = ∫|ρ_out − ρ_in|;
/// `mixing`: the DIIS step α along the minimised residual (plain linear-mixing α on the first
/// step); `eig_tol`: LOBPCG residual tolerance; `diis_history`: pairs kept by the Pulay mixer
/// (1 = linear mixing). `Copy` so the orbital display can keep the settings of the state it shows.
/// `occupations`: how each spin fills its levels (module doc, "Occupations").
/// `stall_window`: the stall stop's W (module doc, "The stall stop"; 0 = off, the cap alone).
#[derive(Clone, Copy, Debug)]
pub struct ScfSettings { pub sigma: f64, pub tol: f64, pub max_iter: usize, pub mixing: f64, pub eig_tol: f64, pub diis_history: usize, pub occupations: Occupations, pub stall_window: usize }

impl ScfSettings {
    /// σ = h (smallest width the grid resolves), tol 1e-6, 200 iterations, α = 0.3, eig_tol 1e-6,
    /// DIIS history 6, Fermi–Dirac occupations at kT = `ELECTRON_KT` (module doc, "Occupations"; spec 2026-11-13 §5:
    /// the oracles test the path the app runs; plan Task 7), the stall stop at `SCF_STALL_WINDOW`. A caller that needs
    /// the kT → 0 reference sets `Occupations::Aufbau` explicitly.
    pub fn for_grid(grid: &Grid) -> ScfSettings {
        ScfSettings { sigma: grid.h, tol: 1e-6, max_iter: 200, mixing: 0.3, eig_tol: 1e-6, diis_history: 6,
                      occupations: Occupations::FermiDirac { kt: ELECTRON_KT }, stall_window: SCF_STALL_WINDOW }
    }
}

/// W of the stall stop (module doc, "The stall stop"): a solve whose best Δρ has not improved 2×
/// within this many iterations — while it is still at or above the stall level
/// (`SCF_STALL_LEVEL_PER_ELECTRON`) — ends with `QmError::ScfStalled`. Measured (`providers`'
/// `scf_stall_measurement`, RX 7900 XTX, `validate/references.md` "SCF stall stop"): over 36 solves
/// that converged — GPU at tol 1e-6: all-electron H₂ 49³, H₂O 57³, LiH, He and H₃⁺ 49³, the app's
/// pseudo H₂O at 95³ × 0.15 (preset, five pass perturbations, O–H scan points cold and a whole scan
/// warm as the pass runs it), pseudo-O M = 1 and 3, pseudo O–H at 2.5 and 3.5 Bohr (the latter
/// with a shell release); CPU H₂ 33³ at the oracles' tol 1e-9 / eig_tol 1e-8 — the longest run without a
/// 2× gain of the best Δρ was 11 iterations (O–H at 3.5 Bohr, phase 1; it converged in 62); every
/// other was ≤ 4. W = 50 is the user-facing target and 4.5× that (the rule asks ≥ 2×). The failing
/// O–H point at 1.8 r₀ oscillates between Δρ ≈ 0.3 and 1.5 for 200 iterations (best 0.28); W = 50
/// ends it after 55 (warm) and 62 (cold retry) iterations instead of 200 + 200. The σ → 0 oracles
/// are gated by `validate` (any SCF error fails them).
/// Review M1 (2026-10-18): stretched spin-symmetric H₂ (R = 5.75–6 Bohr, GPU, tol 1e-6) converges after no-gain runs
/// of 118–152 iterations (`providers`' `scf_stall_stretched_h2`), so W alone cannot separate it from a failure under
/// the cap; ruling (a): the stop also needs the best Δρ at or above the stall level, which those plateaus (best
/// ≤ 9.6e-5) never reach.
pub const SCF_STALL_WINDOW: usize = 50;

/// The stall level per electron (e): the stall stop ends a solve only while its best Δρ is at or above
/// L = this × N (module doc, "The stall stop"; ruling (a) of the review, 2026-10-18). Measured
/// (`validate/references.md` "SCF stall stop"): every long plateau of a solve that went on to converge sat at a
/// best Δρ ≤ 9.6e-5 (stretched H₂, N = 2: L = 2e-3, 20× above it), every measured non-converging solve at a best
/// Δρ of 0.12–0.43 (water's O–H at 1.8 r₀ and the OH radical, N = 7–8 valence: L = 7–8e-3, ≥ 15× below it).
/// A slow oscillation below L still runs to the cap: it converges, or costs what it cost before the stop.
/// **Tested range** (re-review r1-m2): N = 1–8 electrons (H, H₂, He, LiH, H₃⁺, all-electron H₂O at 57³, pseudo-O,
/// pseudo H₂O and OH at 95³); the worst no-gain run *above* L of any solve that converged was 11 iterations (the OH
/// radical at 3.5 Bohr, phase 1). Unmeasured: large N. L grows with N while a failure localised on one bond keeps a
/// Δρ of order one — in a large molecule such a failure can sit below L and then runs to the cap, as before the stop.
pub const SCF_STALL_LEVEL_PER_ELECTRON: f64 = 1e-3;

/// What the CPU loop does after Δρ of an iteration (review M2, re-review r1-m3): a non-finite Δρ ends the solve at
/// once (`Abort` → `ScfNotConverged(i, Δρ)` after the loop), whatever the stall watch says; a Δρ below tol is the
/// convergence test (`Converged`: the shell release or the end); otherwise the stall stop (`Stall`) or the next
/// iteration (`Go`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AfterDeltaRho { Abort, Converged, Stall, Go }

fn after_delta_rho(delta_rho: f64, tol: f64, stalled: bool) -> AfterDeltaRho {
    if !delta_rho.is_finite() {
        AfterDeltaRho::Abort
    } else if delta_rho < tol {
        AfterDeltaRho::Converged
    } else if stalled {
        AfterDeltaRho::Stall
    } else {
        AfterDeltaRho::Go
    }
}

/// The stall stop's bookkeeping (module doc, "The stall stop"), shared by `scf` and `qm_gpu::scf`.
#[derive(Clone, Debug)]
pub struct StallWatch { window: usize, level: f64, anchor: f64, anchor_at: usize, best: f64 }

impl StallWatch {
    /// `window` = W (0: never stalls), `level` = L (a best Δρ below it never stalls).
    pub fn new(window: usize, level: f64) -> StallWatch { StallWatch { window, level, anchor: f64::INFINITY, anchor_at: 0, best: f64::INFINITY } }

    /// The watch of a solve: W = `settings.stall_window`, L = `SCF_STALL_LEVEL_PER_ELECTRON` × the electrons the
    /// system carries (n↑ + n↓; valence under a pseudopotential setup).
    pub fn for_solve(settings: &ScfSettings, system: &System) -> StallWatch {
        StallWatch::new(settings.stall_window, SCF_STALL_LEVEL_PER_ELECTRON * (system.n_up + system.n_dn) as f64)
    }

    /// Records Δρ of iteration `i` (1-based) and returns the iterations since the last 2× gain
    /// (0 on a gain). A non-finite Δρ is never a gain (a guard: the solvers end on a non-finite Δρ before it
    /// can matter, so only the unit tests feed one).
    pub fn observe(&mut self, i: usize, delta_rho: f64) -> usize {
        if delta_rho < self.best { self.best = delta_rho; }
        if delta_rho.is_finite() && (!self.anchor.is_finite() || delta_rho <= 0.5 * self.anchor) {
            self.anchor = delta_rho;
            self.anchor_at = i;
        }
        i.saturating_sub(self.anchor_at)
    }

    /// Whether `since_gain` (from `observe`) ends the solve: W iterations without a 2× gain while the best Δρ is
    /// still at or above the level.
    pub fn stalled(&self, since_gain: usize) -> bool { self.window > 0 && since_gain >= self.window && self.best >= self.level }

    /// A new convergence phase starts after iteration `i` (the `DegenerateShells` release).
    pub fn restart(&mut self, i: usize) { *self = StallWatch { anchor_at: i, ..StallWatch::new(self.window, self.level) }; }

    /// L, the stall level (e).
    pub fn level(&self) -> f64 { self.level }

    /// The lowest Δρ observed (since the last restart).
    pub fn best(&self) -> f64 { self.best }

    pub fn window(&self) -> usize { self.window }

    /// The stall error after `iterations`.
    pub fn error(&self, iterations: usize, tol: f64) -> QmError {
        QmError::ScfStalled { iterations, best_delta_rho: self.best, tol, window: self.window, level: self.level }
    }
}

/// One SCF iteration as an observer sees it (module doc, "Progress"): Δρ and its target, the
/// Harris–Foulkes energy of the iteration's input density (Ha; F_HF, with −kT·S of the iteration's f, under
/// `FermiDirac`; NaN when not computed), and the
/// stall watch: the best Δρ so far, the iterations since its last 2× gain, W (0 = off) and the
/// stall level L.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScfProgress {
    pub iteration: usize,
    pub max_iter: usize,
    pub delta_rho: f64,
    pub tol: f64,
    pub energy: f64,
    pub best_delta_rho: f64,
    pub since_gain: usize,
    pub window: usize,
    pub stall_level: f64,
}

/// Converged Kohn-Sham state: orbitals, eigenvalues, output densities, and the potentials
/// (`v_h` is V_H of `rho_up + rho_dn`). Reusable as a warm start on the same `grid`.
/// `grid`, `sigma` and `ionic` record what the state was converged on: `scf` ignores a warm state
/// from a different grid or another ionic setup (module doc, "Ionic species (M7)"), and `forces`
/// asserts the grid and σ match its arguments. `v_nuc` is the local ionic potential V_loc (today's
/// V_nuc when `ionic` is `None`); `rho_up`/`rho_dn` are the (valence) electron densities, without ρ_pc.
/// `occ_up`/`occ_dn` hold one occupation per stored orbital (all 1 under `Aufbau`, where exactly N_σ
/// orbitals are stored; under `DegenerateShells` the stored orbitals include the gap witnesses, f = 0; under
/// `FermiDirac` they are the final step's Fermi–Dirac f over the grown block, the witnesses included with their small
/// but nonzero f — never clipped to 0 — so Σf = N_σ to ≈ 1e-13), and `occupations` the scheme the state was converged
/// under, kT included (the warm-start guard compares it with `==`). `rho_up`/`rho_dn` are Σ f ψ² of those f.
pub struct KsState {
    pub orbitals_up: Vec<Vec<f64>>, pub orbitals_dn: Vec<Vec<f64>>,
    pub evals_up: Vec<f64>, pub evals_dn: Vec<f64>,
    pub rho_up: Vec<f64>, pub rho_dn: Vec<f64>,
    pub v_nuc: Vec<f64>, pub v_h: Vec<f64>,
    pub grid: Grid, pub sigma: f64,
    pub ionic: Option<Arc<IonicSetup>>,
    pub occ_up: Vec<f64>, pub occ_dn: Vec<f64>, pub occupations: Occupations,
}

impl KsState {
    /// ρ↑ + ρ↓
    pub fn rho(&self) -> Vec<f64> { self.rho_up.iter().zip(&self.rho_dn).map(|(a, b)| a + b).collect() }
}

/// Energy components (Hartree). `internal` = E = kinetic + external + hartree + xc + nuclear (that left-to-right
/// sum), the Kohn–Sham energy with the f-weights; `ts` = kT·S ≥ 0, the electronic entropy term of the final step's
/// occupations (0.0 under `Aufbau` and `DegenerateShells`); `total` = F = `internal − ts`, bitwise — the Mermin free
/// energy under `FermiDirac` (module doc, "Occupations"), E itself otherwise. Every consumer of `total` gets F. Under an
/// ionic setup (module doc, "Ionic species (M7)"): `external` = ∫ρ V_loc + E_NL, the whole
/// ion–electron energy; `xc` = E_xc[ρ↑ + ½ρ_pc, ρ↓ + ½ρ_pc]; `nuclear` = E_ii with valence charges Z_v
/// for pseudised species; `internal` and `total` are then valence energies.
#[derive(Debug, Clone, Copy)]
pub struct Energy { pub kinetic: f64, pub external: f64, pub hartree: f64, pub xc: f64, pub nuclear: f64, pub internal: f64, pub ts: f64, pub total: f64 }

/// SCF diagnostics. `boundary_density` is max |ρ| on the outermost grid plane of each of the six
/// sides — one spacing h inside the Dirichlet wall (ψ = 0 on the ghost plane beyond it), where
/// a confined orbital is already pinched (ρ ∝ h² there). The inner solvers are
/// reported as their worst case over the whole run: `max_eigen_residual` is the largest LOBPCG
/// exit residual over both spin channels and every SCF step; `poisson_residual` and
/// `poisson_iterations` are the largest relative residual and iteration count over every
/// Poisson solve, including the final one for the energy. `max_eigen_residual ≥ eig_tol` means
/// LOBPCG stopped at its iteration cap at least once (see `InnerStats::eigen` for why that is
/// not a warning); a Poisson solve that stops at its cap above its tolerance (`poisson_tol`) adds a warning.
/// `core_overlaps`: the pairs (I, J, R_IJ) whose core radii overlap (spec §5.6; empty without an
/// ionic setup), each also a warning starting with `ionic::CORE_OVERLAP_WARNING`. `fractional`: the
/// shared clusters of the final step under `DegenerateShells` (pseudopotential spec §5.7; empty under `Aufbau`,
/// `FermiDirac` and for integer patterns); `released`: the phase-1 clusters whose converged spread was ≥ τ, as they
/// were when released (both go with `DegenerateShells` in plan Task 10). Under `FermiDirac` (module doc,
/// "Occupations"), of the final step's occupations: `entropy` = S = Σ_σ Σ_i s(f_iσ) (dimensionless, k_B = 1;
/// `energy.ts` = kT·S); `mu` = μ_σ per spin [↑, ↓] (`None` for an empty channel; a physical number only where the
/// channel has a partial level, spec §4.1); `partial` = every level with min(f, 1 − f) ≥ `PARTIAL_OCCUPATION_MIN`, in
/// ascending ε per spin, ↑ first. Under the other schemes S = 0, μ = [None, None] and `partial` is empty.
/// `tol_used`: the Δρ tolerance the converged step was tested against — always `settings.tol` on the CPU (f64 has no
/// occupation-noise floor); on the GPU max(`settings.tol`, `GPU_PARTIAL_TOL`) when that step's occupations have a
/// partial level (spec §5.1, `qm_gpu::scf` "Fermi–Dirac occupations"), else `settings.tol`.
#[derive(Debug, Clone)]
pub struct ScfStats {
    pub iterations: usize, pub delta_rho: f64, pub energy: Energy, pub boundary_density: f64,
    pub max_eigen_residual: f64, pub poisson_residual: f64, pub poisson_iterations: usize,
    pub warnings: Vec<String>,
    pub core_overlaps: Vec<(usize, usize, f64)>,
    pub fractional: Vec<FractionalShell>,
    pub released: Vec<FractionalShell>,
    pub entropy: f64,
    pub mu: [Option<f64>; 2],
    pub partial: Vec<PartialLevel>,
    pub tol_used: f64,
}

/// Density on a box face above this (e/Bohr³) triggers the box-size warning.
const BOUNDARY_DENSITY_WARN: f64 = 1e-5;

/// Relative residual tolerance of the SCF's Poisson solves at any requested tol ≥ 1e-7, and the iteration cap of every
/// Poisson solve in the SCF. Below tol 1e-7 the tolerance tightens with tol (`poisson_tol`).
const POISSON_TOL: f64 = 1e-10;
const POISSON_MAX_ITER: usize = 5000;

/// The Poisson tolerance of an SCF asked for Δρ < `tol`: `POISSON_TOL` capped at 1e-3·tol, i.e. exactly 1e-10 for every
/// tol ≥ 1e-7 (1e-3·1e-7 == 1e-10 in f64; those solves are bit-identical to the fixed tolerance) and 1e-12 at tol 1e-9.
/// Why it follows tol: under `FermiDirac` a partly occupied pair responds to a level shift with ∂f/∂ε = f(1 − f)/kT
/// (≈ 34/Ha at u = Δ/4kT = 0.65, kT = 5 mHa), plus the intra-pair rotation (f₁ − f₂)/Δ (≈ 45/Ha), together ≈ 80–100×,
/// which amplifies an inner-solver error into a density floor. Measured on V3's pinned H₄ (`validate` oracles, plan
/// 2026-11-13 Task 6; 41³ × 0.30, tol 1e-9): at a fixed 1e-10 Δρ wanders between 2e-9 and 2e-7 for 200 steps, whatever
/// eig_tol (1e-9 … 1e-11); at 1e-11, 1e-12 and 1e-13 the cold solve converges in 18–20 steps. The factor is 1e-3, not
/// 1e-2 (Task 6 review): at 1e-2 the floor (≈ 1e-9) sat at tol and warm solves took 184 and 92 of 200 steps — a dip
/// below a noise floor rather than a contraction. Gapped systems (≈ 3/Ha) and exactly degenerate shells (f = ½,
/// insensitive) barely amplify it.
fn poisson_tol(tol: f64) -> f64 { POISSON_TOL.min(1e-3 * tol) }

/// Worst-case inner-solver diagnostics accumulated over an SCF run (see `ScfStats`).
struct InnerStats { eigen_residual: f64, poisson_residual: f64, poisson_iterations: usize, poisson_tol: f64, warnings: Vec<String> }

impl InnerStats {
    /// `poisson_tol`: the tolerance the run's Poisson solves were asked for (`poisson_tol(settings.tol)`).
    fn new(poisson_tol: f64) -> InnerStats { InnerStats { eigen_residual: 0.0, poisson_residual: 0.0, poisson_iterations: 0, poisson_tol, warnings: Vec::new() } }

    /// Records one LOBPCG exit. A residual not below `eig_tol` means the 200-iteration cap was
    /// hit; that is reported through `max_eigen_residual` only, not as a warning: on the oracle
    /// grids (81³ at h = 0.15, 101³ at h = 0.10) LOBPCG routinely stops at the cap with a
    /// residual of 1.0–1.4 × eig_tol, including on the final SCF step, and the oracles treat
    /// any warning as a failure. Whether such a stall should warn is pending a ruling.
    fn eigen(&mut self, stats: Option<EigenStats>) {
        let Some(s) = stats else { return };
        self.eigen_residual = self.eigen_residual.max(s.max_residual);
    }

    /// Records one Poisson solve; a residual above the run's Poisson tolerance means the iteration cap was hit.
    fn poisson(&mut self, stats: PoissonStats, what: &str) {
        self.poisson_residual = self.poisson_residual.max(stats.residual);
        self.poisson_iterations = self.poisson_iterations.max(stats.iterations);
        if !(stats.residual <= self.poisson_tol) {
            self.warnings.push(format!("{what}: Poisson stopped after {} iterations at relative residual {:.2e} > {:.0e}", stats.iterations, stats.residual, self.poisson_tol));
        }
    }
}

/// Deterministic Gaussian-enveloped pseudo-random guess, orthogonalised later by LOBPCG.
/// `pub` so that `qm-gpu::scf` starts its cold solves from the same guess (uploaded as f32);
/// it is not part of the solver's contract.
pub fn initial_orbitals(grid: &Grid, nuclei: &[Nucleus], k: usize, seed0: u64) -> Vec<Vec<f64>> {
    let mut seed = seed0;
    (0..k).map(|_| grid.sample(|r| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (seed >> 11) as f64 / (1u64 << 53) as f64;
        let env: f64 = nuclei.iter().map(|n| (-((r[0]-n.pos[0]).powi(2)+(r[1]-n.pos[1]).powi(2)+(r[2]-n.pos[2]).powi(2))/4.0).exp()).sum();
        (u + 0.5) * env
    })).collect()
}

/// ρ = Σ_i ψ_i² (exactly zero for an empty orbital list).
fn density_from(grid: &Grid, orbitals: &[Vec<f64>]) -> Vec<f64> {
    let mut rho = vec![0.0; grid.len()];
    for psi in orbitals { for i in 0..rho.len() { rho[i] += psi[i] * psi[i]; } }
    rho
}

/// ρ = Σ_i f_i ψ_i² (spec §5.7).
fn weighted_density(grid: &Grid, orbitals: &[Vec<f64>], f: &[f64]) -> Vec<f64> {
    let mut rho = vec![0.0; grid.len()];
    for (psi, &fi) in orbitals.iter().zip(f) {
        if fi == 0.0 { continue; }
        for i in 0..rho.len() { rho[i] += fi * psi[i] * psi[i]; }
    }
    rho
}

/// max |f| over the six faces of the box.
fn max_on_faces(grid: &Grid, f: &[f64]) -> f64 {
    let [n0, n1, n2] = grid.n;
    let mut m: f64 = 0.0;
    for k in 0..n2 { for j in 0..n1 { for i in 0..n0 {
        if i == 0 || j == 0 || k == 0 || i == n0 - 1 || j == n1 - 1 || k == n2 - 1 { m = m.max(f[grid.idx(i, j, k)].abs()); }
    }}}
    m
}

/// One Pulay history entry: the input densities and their residuals r = ρ_out − ρ_in, per spin.
struct DiisEntry { rho_up: Vec<f64>, rho_dn: Vec<f64>, r_up: Vec<f64>, r_dn: Vec<f64> }

/// Relative diagonal regularisation of the (scaled) residual Gram matrix.
const DIIS_REG: f64 = 1e-12;

/// Pulay coefficients for `hist`: solve the bordered system of the module doc on the Gram
/// matrix scaled by its largest diagonal. `None` if the solve fails or gives non-finite c.
fn diis_coefficients(grid: &Grid, hist: &[DiisEntry]) -> Option<Vec<f64>> {
    let m = hist.len();
    let mut b = DMatrix::<f64>::zeros(m + 1, m + 1);
    for i in 0..m { for j in 0..=i {
        let v = grid.dot(&hist[i].r_up, &hist[j].r_up) + grid.dot(&hist[i].r_dn, &hist[j].r_dn);
        b[(i, j)] = v; b[(j, i)] = v;
    }}
    let scale = (0..m).map(|i| b[(i, i)]).fold(0.0f64, f64::max);
    if !(scale > 0.0) { return None; }
    for i in 0..m { for j in 0..m { b[(i, j)] /= scale; } b[(i, i)] += DIIS_REG; }
    for i in 0..m { b[(i, m)] = 1.0; b[(m, i)] = 1.0; }
    let mut rhs = DVector::<f64>::zeros(m + 1);
    rhs[m] = 1.0;
    let x = b.lu().solve(&rhs)?;
    let c: Vec<f64> = x.iter().take(m).copied().collect();
    if c.iter().all(|v| v.is_finite()) { Some(c) } else { None }
}

/// Pulay/DIIS mix (module doc): ρ_in,new = Σ c_i (ρ_in,i + α r_i), negatives clamped to 0.
/// Drops the oldest entries while the coefficient solve fails; with one entry it is linear mixing.
fn diis_mix(grid: &Grid, hist: &mut Vec<DiisEntry>, alpha: f64, rho_up: &mut [f64], rho_dn: &mut [f64]) {
    let c = loop {
        if hist.len() == 1 { break vec![1.0]; }
        match diis_coefficients(grid, hist) {
            Some(c) => break c,
            None => { hist.remove(0); }
        }
    };
    for i in 0..rho_up.len() {
        let (mut u, mut d) = (0.0, 0.0);
        for (e, &ci) in hist.iter().zip(&c) {
            u += ci * (e.rho_up[i] + alpha * e.r_up[i]);
            d += ci * (e.rho_dn[i] + alpha * e.r_dn[i]);
        }
        rho_up[i] = u.max(0.0);
        rho_dn[i] = d.max(0.0);
    }
}

/// The ionic terms of one geometry (module doc, "Ionic species (M7)"): the local potential, the
/// partial core (`None` without one anywhere: the no-core branch), the nonlocal operator (`None`
/// without projectors), E_ii and the core overlaps. Without a setup these are today's arrays,
/// computed by today's functions.
struct IonicTerms { v_loc: Vec<f64>, rho_pc: Option<Vec<f64>>, nl: Option<NonlocalOp>, e_nn: f64, overlaps: Vec<(usize, usize, f64)> }

impl IonicTerms {
    fn new(grid: &Grid, system: &System, settings: &ScfSettings) -> Result<IonicTerms, QmError> {
        match system.ionic.as_deref() {
            None => {
                let e_nn = nuclear_repulsion(&system.nuclei)?;
                Ok(IonicTerms { v_loc: nuclear_potential(grid, &system.nuclei, settings.sigma), rho_pc: None, nl: None, e_nn, overlaps: Vec::new() })
            }
            Some(s) => {
                assert!(s.sigma_ae == settings.sigma, "IonicSetup σ_AE {} ≠ settings σ {}", s.sigma_ae, settings.sigma);
                assert!(s.h == grid.h, "IonicSetup h {} ≠ grid h {}", s.h, grid.h);
                let e_nn = ionic_repulsion(&system.nuclei, s)?;
                let nl = NonlocalOp::build(grid, &system.nuclei, s);
                Ok(IonicTerms {
                    v_loc: ionic_local_potential(grid, &system.nuclei, s),
                    rho_pc: has_partial_core(&system.nuclei, s).then(|| partial_core_field(grid, &system.nuclei, s)),
                    nl: (!nl.is_empty()).then_some(nl), e_nn, overlaps: core_overlaps(s, &system.nuclei),
                })
            }
        }
    }

    /// The points with n_σ = ρ_σ + ½ρ_pc < 0 for either spin (where `exc_point` clamps; module doc).
    fn domain_violations(&self, up: &[f64], dn: &[f64]) -> usize {
        self.rho_pc.as_ref().map_or(0, |pc| (0..pc.len()).filter(|&i| up[i] + 0.5 * pc[i] < 0.0 || dn[i] + 0.5 * pc[i] < 0.0).count())
    }

    /// LSDA on ρ_σ + ½ρ_pc per spin (the core is unpolarised; spec §4.5, §5.2); plain `lsda_pw92`
    /// without a partial core.
    fn xc(&self, up: &[f64], dn: &[f64]) -> XcResult { xc_with_core(up, dn, self.rho_pc.as_deref()) }
}

/// LSDA on (ρ↑ + ½ρ_pc, ρ↓ + ½ρ_pc), or plain `lsda_pw92(ρ↑, ρ↓)` without a partial core: the SCF's
/// v_xc (`IonicTerms::xc`) and the orbital display's H^out (`orbitals::output_potentials`), one formula.
pub(crate) fn xc_with_core(up: &[f64], dn: &[f64], rho_pc: Option<&[f64]>) -> XcResult {
    match rho_pc {
        None => lsda_pw92(up, dn),
        Some(pc) => {
            let u: Vec<f64> = up.iter().zip(pc).map(|(a, c)| a + 0.5 * c).collect();
            let d: Vec<f64> = dn.iter().zip(pc).map(|(a, c)| a + 0.5 * c).collect();
            lsda_pw92(&u, &d)
        }
    }
}

/// The warm-start guard (module doc): both all-electron without a setup, or the same physics.
fn same_ionic(a: Option<&IonicSetup>, b: Option<&IonicSetup>) -> bool {
    match (a, b) { (None, None) => true, (Some(x), Some(y)) => x.same_physics(y), _ => false }
}

/// Lowest `orbitals.len()` eigenpairs of −½∇² + v_eff (+ V_NL when `nl` is given), warm from
/// `orbitals`/`evals`. The preconditioner is shifted by the previous lowest eigenvalue (−1 Ha
/// placeholder on a cold start, harmless: it only shifts the Jacobi diagonal); it is the local
/// one in both cases (spec §5.5: "the preconditioner is unchanged").
/// Returns `None` for an empty channel.
fn solve_channel(grid: &Grid, v_eff: &[f64], nl: Option<&NonlocalOp>, orbitals: &mut Vec<Vec<f64>>, evals: &mut Vec<f64>, eig_tol: f64) -> Option<EigenStats> {
    if orbitals.is_empty() { return None; }
    let lambda = evals.first().copied().unwrap_or(-1.0);
    let pc = jacobi_precond(grid, v_eff, lambda);
    Some(match nl {
        None => lobpcg(grid, &LocalHamiltonian { grid, v: v_eff }, orbitals, evals, Some(&pc), eig_tol, 200, None),
        Some(nl) => lobpcg(grid, &PseudoHamiltonian { grid, v: v_eff, nl }, orbitals, evals, Some(&pc), eig_tol, 200, None),
    })
}

/// One spin channel under `DegenerateShells` (module doc, "Occupations"). Phase 1 (`frozen` = None): solve, grow
/// the block by one vector until the gap witness clears τ_c, share a straddling cluster. Phase 2: the frozen
/// occupations by rank (no growth). Returns the occupations and the shared clusters (f not 0 or 1).
#[allow(clippy::too_many_arguments)]
fn solve_shared(grid: &Grid, v_eff: &[f64], nl: Option<&NonlocalOp>, orbitals: &mut Vec<Vec<f64>>, evals: &mut Vec<f64>, n: usize,
                frozen: Option<&[f64]>, eig_tol: f64, nuclei: &[Nucleus], spin: u8, inner: &mut InnerStats) -> Result<(Vec<f64>, Vec<FractionalShell>), QmError> {
    if n == 0 { return Ok((vec![0.0; orbitals.len()], Vec::new())); }
    if let Some(by_rank) = frozen {
        inner.eigen(solve_channel(grid, v_eff, nl, orbitals, evals, eig_tol));
        let order = ascending(evals);
        let mut f = vec![0.0; evals.len()];
        for (r, &i) in order.iter().enumerate() { f[i] = by_rank[r]; }
        return Ok((f, shared_runs(evals, &order, by_rank, spin)));
    }
    loop {
        inner.eigen(solve_channel(grid, v_eff, nl, orbitals, evals, eig_tol));
        if let Some((f, shell)) = degenerate_shell_occupations(evals, n, WITNESS_GAP) {
            return Ok((f, shell.map(|(first, count, f, spread)| FractionalShell { spin, first, count, f, spread }).into_iter().collect()));
        }
        if orbitals.len() >= n + MAX_WITNESS_GROWTH {
            return Err(QmError::Numerical(format!("no gap witness within {MAX_WITNESS_GROWTH} extra levels of spin {spin} (τ_c = {WITNESS_GAP} Ha)")));
        }
        orbitals.extend(initial_orbitals(grid, nuclei, 1, 1000 + 2 * orbitals.len() as u64 + spin as u64));
        evals.push(*evals.last().expect("n > 0 vectors"));
    }
}

/// One spin channel under `FermiDirac { kt }` (module doc, "Occupations"): solve the block, take `fermi_dirac`'s
/// occupations of the step's levels, and while the highest level is no witness (`witness_reached`: ≥ ε_(n) +
/// `WITNESS_GAP` and ≥ μ + `WITNESS_TAIL_KT`·kT) grow the block by one vector and re-solve every level. The cap and the
/// grown vectors' seeds are `solve_shared`'s. Errors: more than `MAX_WITNESS_GROWTH` extra vectors, and
/// `fermi_dirac`'s refusals (a non-finite level). An empty channel (n = 0) is not solved: f = [], μ = None, S = 0.
#[allow(clippy::too_many_arguments)]
fn solve_fd(grid: &Grid, v_eff: &[f64], nl: Option<&NonlocalOp>, orbitals: &mut Vec<Vec<f64>>, evals: &mut Vec<f64>, n: usize, kt: f64,
            eig_tol: f64, nuclei: &[Nucleus], spin: u8, inner: &mut InnerStats) -> Result<FdOccupations, QmError> {
    if n == 0 { return fermi_dirac(evals, 0, kt); }
    loop {
        inner.eigen(solve_channel(grid, v_eff, nl, orbitals, evals, eig_tol));
        let occ = fermi_dirac(evals, n, kt)?;
        if witness_reached(evals, n, occ.mu.expect("an occupied channel has μ"), kt) { return Ok(occ); }
        if orbitals.len() >= n + MAX_WITNESS_GROWTH {
            return Err(QmError::Numerical(format!("no gap witness within {MAX_WITNESS_GROWTH} extra levels of spin {spin} \
                                                   (WITNESS_GAP = {WITNESS_GAP} Ha above ε_N and {WITNESS_TAIL_KT} kT above μ)")));
        }
        orbitals.extend(initial_orbitals(grid, nuclei, 1, 1000 + 2 * orbitals.len() as u64 + spin as u64));
        evals.push(*evals.last().expect("n > 0 vectors"));
    }
}

/// Indices of `evals` in ascending order (`pub` for `qm-gpu::scf`).
pub fn ascending(evals: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..evals.len()).collect();
    order.sort_by(|&a, &b| evals[a].total_cmp(&evals[b]));
    order
}

/// The maximal runs of equal fractional f (0 < f < 1) in a by-rank pattern, as shells with their spread (`pub` for
/// `qm-gpu::scf`).
pub fn shared_runs(evals: &[f64], order: &[usize], by_rank: &[f64], spin: u8) -> Vec<FractionalShell> {
    let mut out = Vec::new();
    let mut r = 0;
    while r < by_rank.len() {
        let f = by_rank[r];
        let mut e = r + 1;
        while e < by_rank.len() && by_rank[e] == f { e += 1; }
        if f > 0.0 && f < 1.0 {
            let lv: Vec<f64> = order[r..e].iter().map(|&i| evals[i]).collect();
            out.push(FractionalShell { spin, first: r, count: e - r, f, spread: lv.iter().cloned().fold(f64::MIN, f64::max) - lv.iter().cloned().fold(f64::MAX, f64::min) });
        }
        r = e;
    }
    out
}

/// Phase 2's decision (module doc): the occupations by rank of one spin, with every shared cluster of spread ≥ τ
/// released (`released_occupations`); the released clusters are appended to `released`. `pub` for `qm-gpu::scf`.
pub fn decide(evals: &[f64], f: &[f64], shells: &[FractionalShell], tau: f64, released: &mut Vec<FractionalShell>) -> Vec<f64> {
    let order = ascending(evals);
    let mut by_rank: Vec<f64> = order.iter().map(|&i| f[i]).collect();
    for c in shells.iter().filter(|c| c.spread >= tau) {
        let sorted: Vec<f64> = order[c.first..c.first + c.count].iter().map(|&i| evals[i]).collect();
        let fr = released_occupations(&sorted, (c.f * c.count as f64).round(), tau);
        by_rank[c.first..c.first + c.count].copy_from_slice(&fr);
        released.push(c.clone());
    }
    by_rank
}

/// Runs the SCF loop (module doc). `warm` reuses a previous state's orbitals, densities and
/// V_H as the starting point when it was converged on the same grid, under the same ionic
/// physics (module doc, "Ionic species (M7)"), and its orbital counts match `system`; otherwise
/// it is ignored. Without a setup a different σ is fine (the σ ladder warm-starts along σ); a
/// setup's σ_AE is part of its physics. Errors: coincident
/// nuclei; no convergence within `settings.max_iter` (including a non-finite Δρ). Panics if a
/// setup's h or σ_AE differ from the grid's and the settings'.
pub fn scf(grid: &Grid, system: &System, settings: &ScfSettings, warm: Option<&KsState>) -> Result<(KsState, ScfStats), QmError> {
    scf_observed(grid, system, settings, warm, None)
}

/// `scf` with a per-iteration observer (module doc, "Progress"): `observe` gets one
/// `ScfProgress` per iteration, its Harris–Foulkes energy included. With `None` it is `scf`; an
/// observer changes no number of the solve (it only reads).
pub fn scf_observed(grid: &Grid, system: &System, settings: &ScfSettings, warm: Option<&KsState>, mut observe: Option<&mut dyn FnMut(&ScfProgress)>)
                    -> Result<(KsState, ScfStats), QmError> {
    let n = grid.len();
    let terms = IonicTerms::new(grid, system, settings)?;
    let v_nuc = &terms.v_loc;
    let mut poisson = PoissonSolver::new(grid);
    let mut inner = InnerStats::new(poisson_tol(settings.tol));

    // starting point
    let (mut orb_up, mut orb_dn, mut ev_up, mut ev_dn, mut rho_in_up, mut rho_in_dn, mut v_h) = match warm {
        Some(w) if w.grid == *grid && counts_match(w, system, settings.occupations) && w.occupations == settings.occupations
                   && same_ionic(w.ionic.as_deref(), system.ionic.as_deref()) => (
            w.orbitals_up.clone(), w.orbitals_dn.clone(), w.evals_up.clone(), w.evals_dn.clone(),
            w.rho_up.clone(), w.rho_dn.clone(), w.v_h.clone()),
        _ => {
            let total = (system.n_up + system.n_dn) as f64;
            // broad atomic-like blobs (σ = 1 Bohr) carrying each centre's charge (Z, or Z_v when pseudised)
            let (z_tot, guess) = match system.ionic.as_deref() {
                None => (system.nuclei.iter().map(|q| q.z as f64).sum::<f64>(), deposit_charge(grid, &system.nuclei, 1.0)),
                Some(s) => {
                    let c: Vec<Centre> = centres(&system.nuclei, s).into_iter().map(|c| Centre { sigma: 1.0, ..c }).collect();
                    (c.iter().map(|c| c.charge).sum::<f64>(), deposit_centres(grid, &c))
                }
            };
            let scale = if z_tot > 0.0 { total / z_tot } else { 0.0 };
            let fu = if total > 0.0 { system.n_up as f64 / total } else { 0.0 };
            let fd = if total > 0.0 { system.n_dn as f64 / total } else { 0.0 };
            // one gap witness per occupied spin channel under DegenerateShells (module doc, "Occupations")
            let extra = |n: usize| usize::from(n > 0 && settings.occupations != Occupations::Aufbau);
            let (ku, kd) = (system.n_up + extra(system.n_up), system.n_dn + extra(system.n_dn));
            (initial_orbitals(grid, &system.nuclei, ku, 1), initial_orbitals(grid, &system.nuclei, kd, 2),
             vec![-1.0; ku], vec![-1.0; kd],
             guess.iter().map(|g| g * scale * fu).collect(), guess.iter().map(|g| g * scale * fd).collect(),
             vec![0.0; n])
        }
    };

    let mut iterations = 0;
    let mut delta_rho = f64::INFINITY;
    let (mut rho_out_up, mut rho_out_dn) = (rho_in_up.clone(), rho_in_dn.clone());
    let mut hist: Vec<DiisEntry> = Vec::new();
    let (mut occ_up, mut occ_dn) = (vec![1.0; orb_up.len()], vec![1.0; orb_dn.len()]);
    let mut fractional: Vec<FractionalShell> = Vec::new();
    // `FermiDirac` (module doc, "Occupations"): S and μ_σ of the latest step's f, its partial levels (empty otherwise)
    let (mut entropy, mut mu, mut partial): (f64, [Option<f64>; 2], Vec<PartialLevel>) = (0.0, [None, None], Vec::new());
    // spec §5.7: phase 1 (None) until the first convergence, then the frozen pattern by rank per spin
    let mut frozen: Option<(Vec<f64>, Vec<f64>)> = None;
    let mut released: Vec<FractionalShell> = Vec::new();
    let mut watch = StallWatch::for_solve(settings, system);
    let mut stalled = false;
    while iterations < settings.max_iter {
        iterations += 1;
        let rho_in: Vec<f64> = rho_in_up.iter().zip(&rho_in_dn).map(|(a, b)| a + b).collect();
        let ps = poisson.solve(grid, &rho_in, &mut v_h, inner.poisson_tol, POISSON_MAX_ITER);
        inner.poisson(ps, &format!("SCF step {iterations}"));
        let xc = terms.xc(&rho_in_up, &rho_in_dn);
        // the Harris–Foulkes double-counting terms of ρ_in (module doc, "Progress"), only for an observer
        let hf_dc = observe.as_ref().map(|_| {
            -hartree_energy(grid, &rho_in, &v_h) - grid.dot(&rho_in_up, &xc.v_up) - grid.dot(&rho_in_dn, &xc.v_dn)
                + grid.integrate(&xc.energy_density) + terms.e_nn
        });
        let v_up: Vec<f64> = (0..n).map(|i| v_nuc[i] + v_h[i] + xc.v_up[i]).collect();
        let v_dn: Vec<f64> = (0..n).map(|i| v_nuc[i] + v_h[i] + xc.v_dn[i]).collect();
        match settings.occupations {
            Occupations::Aufbau => {
                inner.eigen(solve_channel(grid, &v_up, terms.nl.as_ref(), &mut orb_up, &mut ev_up, settings.eig_tol));
                inner.eigen(solve_channel(grid, &v_dn, terms.nl.as_ref(), &mut orb_dn, &mut ev_dn, settings.eig_tol));
                rho_out_up = density_from(grid, &orb_up);
                rho_out_dn = density_from(grid, &orb_dn);
            }
            Occupations::DegenerateShells { .. } => {
                let (fzu, fzd) = match &frozen { Some((u, d)) => (Some(u.as_slice()), Some(d.as_slice())), None => (None, None) };
                let (fu, su) = solve_shared(grid, &v_up, terms.nl.as_ref(), &mut orb_up, &mut ev_up, system.n_up, fzu, settings.eig_tol, &system.nuclei, 0, &mut inner)?;
                let (fd, sd) = solve_shared(grid, &v_dn, terms.nl.as_ref(), &mut orb_dn, &mut ev_dn, system.n_dn, fzd, settings.eig_tol, &system.nuclei, 1, &mut inner)?;
                rho_out_up = weighted_density(grid, &orb_up, &fu);
                rho_out_dn = weighted_density(grid, &orb_dn, &fd);
                (occ_up, occ_dn) = (fu, fd);
                fractional = su.into_iter().chain(sd).collect();
            }
            Occupations::FermiDirac { kt } => {
                // the step's own levels decide the f (never mixed; spec 2026-11-13 §5 "Unchanged")
                let ou = solve_fd(grid, &v_up, terms.nl.as_ref(), &mut orb_up, &mut ev_up, system.n_up, kt, settings.eig_tol, &system.nuclei, 0, &mut inner)?;
                let od = solve_fd(grid, &v_dn, terms.nl.as_ref(), &mut orb_dn, &mut ev_dn, system.n_dn, kt, settings.eig_tol, &system.nuclei, 1, &mut inner)?;
                rho_out_up = weighted_density(grid, &orb_up, &ou.f);
                rho_out_dn = weighted_density(grid, &orb_dn, &od.f);
                entropy = neumaier([ou.entropy, od.entropy].into_iter());
                mu = [ou.mu, od.mu];
                partial = partial_levels(&ev_up, &ou.f, 0).into_iter().chain(partial_levels(&ev_dn, &od.f, 1)).collect();
                (occ_up, occ_dn) = (ou.f, od.f);
            }
        }
        let diff: Vec<f64> = (0..n).map(|i| (rho_out_up[i] - rho_in_up[i]).abs() + (rho_out_dn[i] - rho_in_dn[i]).abs()).collect();
        delta_rho = grid.integrate(&diff);
        let since_gain = watch.observe(iterations, delta_rho);
        if let (Some(f), Some(dc)) = (observe.as_deref_mut(), hf_dc) {
            // Σ f ε over both spins (all f = 1 under Aufbau), Neumaier like every energy sum; under FermiDirac F_HF, with
            // −kT·S of this step's f (S = 0 otherwise)
            let band = neumaier(ev_up.iter().zip(&occ_up).chain(ev_dn.iter().zip(&occ_dn)).map(|(e, f)| e * f));
            f(&ScfProgress { iteration: iterations, max_iter: settings.max_iter, delta_rho, tol: settings.tol, energy: band + dc - kt_of(settings.occupations) * entropy,
                             best_delta_rho: watch.best(), since_gain, window: watch.window(), stall_level: watch.level() });
        }
        // a non-finite Δρ ends the solve at once as `ScfNotConverged(i, Δρ)` below — a numerical blow-up, not a
        // stall (review M2): it must not run W more iterations and be reported as a lack of 2× progress
        let next = after_delta_rho(delta_rho, settings.tol, watch.stalled(since_gain));
        if next == AfterDeltaRho::Abort { break; }
        if next == AfterDeltaRho::Converged {
            // spec §5.7 phase 2: at the first convergence, release every shared cluster whose spread is ≥ τ, freeze
            // the pattern and converge again; without a release the phase-1 state is the answer
            match (settings.occupations, &frozen) {
                (Occupations::DegenerateShells { tau }, None) if fractional.iter().any(|c| c.spread >= tau) => {
                    let (su, sd): (Vec<FractionalShell>, Vec<FractionalShell>) = fractional.iter().cloned().partition(|c| c.spin == 0);
                    let up = decide(&ev_up, &occ_up, &su, tau, &mut released);
                    let dn = decide(&ev_dn, &occ_dn, &sd, tau, &mut released);
                    frozen = Some((up, dn));
                    hist.clear();
                    delta_rho = f64::INFINITY;
                    watch.restart(iterations); // a new convergence phase (module doc, "The stall stop")
                }
                _ => break,
            }
        } else if next == AfterDeltaRho::Stall {
            // never on a Δρ below tol (`after_delta_rho`); the GPU mirrors it, where a Δρ below tol on a loosened eigen
            // step takes one more step and must never be called stalled (review S1)
            stalled = true;
            break;
        }
        hist.push(DiisEntry {
            r_up: (0..n).map(|i| rho_out_up[i] - rho_in_up[i]).collect(),
            r_dn: (0..n).map(|i| rho_out_dn[i] - rho_in_dn[i]).collect(),
            rho_up: std::mem::take(&mut rho_in_up),
            rho_dn: std::mem::take(&mut rho_in_dn),
        });
        if hist.len() > settings.diis_history.max(1) { hist.remove(0); }
        rho_in_up = vec![0.0; n];
        rho_in_dn = vec![0.0; n];
        diis_mix(grid, &mut hist, settings.mixing, &mut rho_in_up, &mut rho_in_dn);
    }
    if stalled { return Err(watch.error(iterations, settings.tol)); }
    // written so that a NaN Δρ (both comparisons false) is an error, not a convergence
    if !(delta_rho < settings.tol) { return Err(QmError::ScfNotConverged(iterations, delta_rho)); }

    // energy from the output density and orbitals (module doc)
    let rho: Vec<f64> = rho_out_up.iter().zip(&rho_out_dn).map(|(a, b)| a + b).collect();
    let ps = poisson.solve(grid, &rho, &mut v_h, inner.poisson_tol, POISSON_MAX_ITER);
    inner.poisson(ps, "final energy");
    let mut t = vec![0.0; n];
    let mut kinetic_e = 0.0;
    match settings.occupations {
        Occupations::Aufbau => for psi in orb_up.iter().chain(orb_dn.iter()) { kinetic(grid, psi, &mut t); kinetic_e += grid.dot(psi, &t); },
        Occupations::DegenerateShells { .. } | Occupations::FermiDirac { .. } => for (psi, &f) in orb_up.iter().zip(&occ_up).chain(orb_dn.iter().zip(&occ_dn)) {
            if f == 0.0 { continue; }
            kinetic(grid, psi, &mut t);
            kinetic_e += f * grid.dot(psi, &t);
        },
    }
    let xc = terms.xc(&rho_out_up, &rho_out_dn);
    let clamped = terms.domain_violations(&rho_out_up, &rho_out_dn);
    let e_nn = terms.e_nn;
    let energy = {
        // ∫ρ V_loc, plus E_NL = Σ_{iσ} Σ E_KB c² under a setup with projectors (module doc)
        let external = match &terms.nl {
            None => grid.dot(&rho, v_nuc),
            Some(nl) => grid.dot(&rho, v_nuc) + match settings.occupations {
                Occupations::Aufbau => nl.energy(&orb_up) + nl.energy(&orb_dn),
                Occupations::DegenerateShells { .. } | Occupations::FermiDirac { .. } => nl.energy_weighted(&orb_up, &occ_up) + nl.energy_weighted(&orb_dn, &occ_dn),
            },
        };
        let hartree = hartree_energy(grid, &rho, &v_h);
        let xc_e = grid.integrate(&xc.energy_density);
        // E as before (the same left-to-right sum: `Aufbau` and `DegenerateShells` keep their bits, ts = 0.0 there), and
        // F = E − kT·S of the final step's f (module doc, "Occupations")
        let internal = kinetic_e + external + hartree + xc_e + e_nn;
        let ts = kt_of(settings.occupations) * entropy;
        Energy { kinetic: kinetic_e, external, hartree, xc: xc_e, nuclear: e_nn, internal, ts, total: internal - ts }
    };
    let boundary_density = max_on_faces(grid, &rho);
    let mut warnings = Vec::new();
    if boundary_density > BOUNDARY_DENSITY_WARN { warnings.push(format!("density on box faces is {boundary_density:.2e}: box too small")); }
    warnings.append(&mut inner.warnings);
    let IonicTerms { v_loc, overlaps, .. } = terms;
    for &(i, j, d) in &overlaps { warnings.push(format!("{CORE_OVERLAP_WARNING}: {i}–{j} at R = {d:.3} Bohr")); }
    if clamped > 0 { warnings.push(format!("LSDA domain: ρ_σ + ½ρ_pc < 0 at {clamped} grid points (the XC clamp engages; spec §4.5, T8(c))")); }

    Ok((KsState { orbitals_up: orb_up, orbitals_dn: orb_dn, evals_up: ev_up, evals_dn: ev_dn,
                  rho_up: rho_out_up, rho_dn: rho_out_dn, v_nuc: v_loc, v_h, grid: grid.clone(), sigma: settings.sigma,
                  ionic: system.ionic.clone(), occ_up, occ_dn, occupations: settings.occupations },
        ScfStats { iterations, delta_rho, energy, boundary_density,
                   max_eigen_residual: inner.eigen_residual, poisson_residual: inner.poisson_residual,
                   poisson_iterations: inner.poisson_iterations, warnings, core_overlaps: overlaps, fractional, released,
                   entropy, mu, partial, tol_used: settings.tol }))
}

/// kT of a scheme (Ha): the `FermiDirac` temperature, 0 for `Aufbau` and `DegenerateShells` (whose S is 0).
fn kt_of(occ: Occupations) -> f64 {
    match occ { Occupations::FermiDirac { kt } => kt, Occupations::Aufbau | Occupations::DegenerateShells { .. } => 0.0 }
}

/// Whether a warm state's orbitals fit `system` under `occ`: exactly N_σ per spin under `Aufbau` (today's
/// test), the same electron count Σf with at least one witness per occupied spin under `DegenerateShells` and
/// `FermiDirac`. The scheme itself, kT included, is the caller's separate `==` guard.
fn counts_match(w: &KsState, system: &System, occ: Occupations) -> bool {
    match occ {
        Occupations::Aufbau => w.orbitals_up.len() == system.n_up && w.orbitals_dn.len() == system.n_dn,
        Occupations::DegenerateShells { .. } | Occupations::FermiDirac { .. } => {
            let fits = |orbs: &[Vec<f64>], f: &[f64], n: usize| orbs.len() == f.len() && (f.iter().sum::<f64>() - n as f64).abs() < 1e-9
                && (if n == 0 { orbs.is_empty() } else { orbs.len() > n });
            fits(&w.orbitals_up, &w.occ_up, system.n_up) && fits(&w.orbitals_dn, &w.occ_dn, system.n_dn)
        }
    }
}

/// Point-nucleus limit of a smeared-nucleus energy series: solves E_i = a + bσ_i² + cσ_i³ + dσ_i⁴
/// (4×4 linear system via nalgebra LU) and returns a. Panics if the σ are not distinct (singular).
/// Model and conditioning: module doc.
pub fn extrapolate_sigma(sigmas: &[f64; 4], energies: &[f64; 4]) -> f64 {
    let a = Matrix4::from_fn(|i, j| match j { 0 => 1.0, _ => sigmas[i].powi(j as i32 + 1) });
    let b = Vector4::from_column_slice(energies);
    let x = a.lu().solve(&b).expect("extrapolate_sigma: the four σ must be distinct");
    x[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use crate::nuclear::Nucleus;
    use approx::assert_relative_eq;

    fn hydrogen(pos: [f64; 3]) -> System { System::new(vec![Nucleus { pos, z: 1 }], 0, None) }

    fn h_energy(grid: &Grid, sigma: f64, pos: [f64; 3]) -> (f64, ScfStats) {
        let mut s = ScfSettings::for_grid(grid);
        s.sigma = sigma; s.tol = 1e-7;
        let (_, stats) = scf(grid, &hydrogen(pos), &s, None).unwrap();
        (stats.energy.total, stats)
    }

    /// The SCF's Poisson tolerance follows the requested tol only below 1e-7 (plan 2026-11-13 Task 6 ruling and its fix
    /// round 1): every tol ≥ 1e-7 keeps `POISSON_TOL` = 1e-10 bit for bit; tol 1e-9 asks 1e-12.
    #[test]
    fn poisson_tol_follows_tight_scf_tol_only() {
        assert_eq!(poisson_tol(1e-6).to_bits(), 1e-10f64.to_bits());
        assert_eq!(poisson_tol(1e-7).to_bits(), 1e-10f64.to_bits());
        assert!((poisson_tol(1e-9) - 1e-12).abs() <= 1e-27, "{}", poisson_tol(1e-9));
    }

    #[test]
    fn spin_counts() {
        let h = hydrogen([0.0; 3]);
        assert_eq!((h.n_up, h.n_dn), (1, 0));
        let li = System::new(vec![Nucleus { pos: [0.0; 3], z: 3 }], 0, None);
        assert_eq!((li.n_up, li.n_dn), (2, 1));
        let he_triplet = System::new(vec![Nucleus { pos: [0.0; 3], z: 2 }], 0, Some(3));
        assert_eq!((he_triplet.n_up, he_triplet.n_dn), (2, 0));
        let h_plus = System::new(vec![Nucleus { pos: [0.0; 3], z: 1 }], 1, None);
        assert_eq!((h_plus.n_up, h_plus.n_dn), (0, 0));
    }

    #[test]
    fn hydrogen_converges_and_conserves_charge() {
        let g = Grid::cube(49, 0.25);
        let (state, stats) = scf(&g, &hydrogen([0.0; 3]), &ScfSettings::for_grid(&g), None).unwrap();
        assert!(stats.delta_rho < 1e-6, "{stats:?}");
        assert_relative_eq!(g.integrate(&state.rho_up), 1.0, epsilon = 1e-8);
        assert_eq!(g.integrate(&state.rho_dn), 0.0);
        assert!(stats.energy.total < -0.40 && stats.energy.total > -0.50, "E = {}", stats.energy.total);
        assert!(stats.warnings.is_empty(), "{:?}", stats.warnings);
    }

    #[test]
    fn hydrogen_energy_extrapolated_to_point_nucleus() {
        // Smearing error: ΔE ≈ 2π·Z·ρ(R)·σ² + O(σ³) (density cusp) + O(σ⁴).
        // Fit E = a + bσ² + cσ³ + dσ⁴ through four σ and read off a. At h = 0.25 this is a
        // coarse-grid sanity check (15 mHa); the 5 mHa oracle lives in `validate` on a finer grid.
        // Reference: NIST LSD total energy of H = −0.478671 Ha (Kotochigova et al.).
        let g = Grid::cube(49, 0.25);
        let sig = [0.25, 0.30, 0.35, 0.40];
        let e: Vec<f64> = sig.iter().map(|&s| h_energy(&g, s, [0.0; 3]).0).collect();
        for w in e.windows(2) { assert!(w[1] > w[0], "E(σ) must increase with σ: {e:?}"); }
        let a = extrapolate_sigma(&sig, &[e[0], e[1], e[2], e[3]]);
        assert!((a - (-0.478671)).abs() < 1.5e-2, "extrapolated E = {a}, raw = {e:?}");
    }

    #[test]
    fn egg_box_ripple_is_small() {
        let g = Grid::cube(49, 0.25);
        let (e0, _) = h_energy(&g, 0.25, [0.0; 3]);
        let (e1, _) = h_energy(&g, 0.25, [0.125, 0.0, 0.0]);
        assert!((e1 - e0).abs() < 2e-3, "egg-box ΔE = {}", e1 - e0);
    }

    #[test]
    fn small_box_raises_energy_and_warns() {
        let big = Grid::cube(49, 0.25);  // 12 Bohr
        let small = Grid::cube(25, 0.25); // 6 Bohr
        let (e_big, _) = h_energy(&big, 0.25, [0.0; 3]);
        let (e_small, stats) = h_energy(&small, 0.25, [0.0; 3]);
        assert!(e_small > e_big, "confinement must raise the energy: {e_small} vs {e_big}");
        assert!(!stats.warnings.is_empty(), "expected a box-size warning");
    }

    #[test]
    fn lithium_runs_with_unequal_spin_channels() {
        let g = Grid::cube(49, 0.25);
        let li = System::new(vec![Nucleus { pos: [0.0; 3], z: 3 }], 0, None);
        let (state, stats) = scf(&g, &li, &ScfSettings::for_grid(&g), None).unwrap();
        // re-pinned (plan Task 7): all-electron now solves witness vectors (FD, spec 2026-11-13) — (2, 1) electrons in
        // more stored levels per spin (were exactly 2 and 1 under Aufbau)
        assert!(state.orbitals_up.len() > 2 && state.orbitals_dn.len() > 1, "{} / {}", state.orbitals_up.len(), state.orbitals_dn.len());
        assert!((state.occ_up.iter().sum::<f64>() - 2.0).abs() < 1e-12 && (state.occ_dn.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        eprintln!("Li 49³ (FD): stored {} / {}, TS {:.3e}, partial {:?}", state.orbitals_up.len(), state.orbitals_dn.len(), stats.energy.ts, stats.partial);
        assert!(stats.delta_rho < 1e-6);
        assert_relative_eq!(g.integrate(&state.rho()), 3.0, epsilon = 1e-8);
    }

    #[test]
    fn warm_start_converges_in_few_iterations() {
        let g = Grid::cube(33, 0.25);
        let s = ScfSettings::for_grid(&g);
        // The brief's original criterion was warm·3 < cold, a linear-mixing-era prediction: with
        // DIIS the cold solve is already ~7 steps and both starts spend steps 1–2 seeding the
        // extrapolation, so an iteration *ratio* no longer measures what a warm start buys.
        // Measured (33³): cold 7, warm 6. The two properties that matter physically are that the
        // warm start is cheaper than a cold solve of the same displaced geometry and lands on the
        // same energy (the warm state is a legitimate starting point, not a bias).
        let (state, cold) = scf(&g, &hydrogen([0.0; 3]), &s, None).unwrap();
        let (_, warm) = scf(&g, &hydrogen([0.02, 0.0, 0.0]), &s, Some(&state)).unwrap();
        let (_, cold_disp) = scf(&g, &hydrogen([0.02, 0.0, 0.0]), &s, None).unwrap();
        assert!(warm.iterations < cold_disp.iterations, "cold {} warm {}", cold_disp.iterations, warm.iterations);
        assert!((warm.energy.total - cold_disp.energy.total).abs() < 1e-5, "warm {} vs cold {}", warm.energy.total, cold_disp.energy.total);
        let _ = cold;
    }

    #[test]
    fn warm_start_from_a_different_grid_runs_cold() {
        // Same orbital counts, different grid: the warm state must be ignored. Same n with a
        // different h is the dangerous case (vector lengths agree, so nothing would panic and
        // orbitals would be silently reinterpreted); a different n would panic inside LOBPCG.
        // Taking the cold path makes the result bit-identical to an explicit cold solve.
        let g = Grid::cube(25, 0.25);
        let s = ScfSettings::for_grid(&g);
        let (cold_state, cold) = scf(&g, &hydrogen([0.0; 3]), &s, None).unwrap();
        assert_eq!((&cold_state.grid, cold_state.sigma), (&g, s.sigma));
        // the inner-solver diagnostics are populated
        assert!(cold.max_eigen_residual > 0.0 && cold.max_eigen_residual.is_finite(), "{cold:?}");
        assert!(cold.poisson_iterations > 0 && cold.poisson_residual <= 1e-10, "{cold:?}");
        for other in [Grid::cube(25, 0.3), Grid::cube(21, 0.25)] {
            let (foreign, _) = scf(&other, &hydrogen([0.0; 3]), &ScfSettings::for_grid(&other), None).unwrap();
            assert_eq!(foreign.grid, other);
            let (state, warm) = scf(&g, &hydrogen([0.0; 3]), &s, Some(&foreign)).unwrap();
            assert_eq!(state.grid, g);
            assert_eq!(warm.iterations, cold.iterations, "warm state from {:?} was not ignored", other.n);
            assert_eq!(warm.energy.total, cold.energy.total);
        }
    }

    // ---- M7 Task 6: the ionic setup in the SCF ----

    // Step 1's output on the code before M7 (86974e3), verbatim. Held: asserted under an explicit `Occupations::Aufbau`
    // (the kT → 0 reference path, unchanged by plan Task 7).
    const BASE_H2_E: u64 = 0xbfee7a76b2327147;
    const BASE_H2_FX: u64 = 0x3fc07b1358ca116a; // 10 SCF steps
    const BASE_LIH_E: u64 = 0xc014e13070f24df5;
    const BASE_LIH_FX: u64 = 0xbf97397795d2fce0; // 13 SCF steps
    // Re-pinned (plan Task 7, 2026-11-16; printed by this test): the same scenes under `ScfSettings::for_grid`'s default
    // since Task 7, FermiDirac at kT = 5 mHa — all-electron now solves witness vectors (FD, spec 2026-11-13). The move
    // from the Aufbau bits above and the −TS of each are recorded beside the values.
    const FD_H2_E: u64 = 0xbfee7a76b232ffed; // -0.952449176832 (Aufbau −0.952449176828: −4.5e-12; TS 4.3e-20; 10 steps, 2 / 2 stored)
    const FD_H2_FX: u64 = 0x3fc07b139b8e8122; // +0.128755999523 (Aufbau +0.128755968432)
    const FD_LIH_E: u64 = 0xc014e130719f64f8; // -5.219911361078 (Aufbau −5.219911351003: −1.0e-8 = −TS 1.59e-7 + E − E_A +1.49e-7; 15 steps, 6 / 6 stored)
    const FD_LIH_FX: u64 = 0xbf973980c2ecf2d0; // -0.022680293939 (Aufbau −0.022680157203)

    /// Global constraint: IonicPolicy::None is today's solver, bit for bit (energies and forces) — under `Aufbau`, set
    /// explicitly since plan Task 7 made FermiDirac `for_grid`'s default; the for_grid (FD) solves are pinned beside it.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
    fn none_policy_is_bitwise_today() {
        let g = Grid::cube(33, 0.25);
        let fd = ScfSettings::for_grid(&g);
        let aufbau = ScfSettings { occupations: Occupations::Aufbau, ..fd };
        let scenes = || [("H2", System::new(vec![Nucleus { pos: [0.7, 0.0, 0.0], z: 1 }, Nucleus { pos: [-0.7, 0.0, 0.0], z: 1 }], 0, None)),
                         ("LIH", System::new(vec![Nucleus { pos: [-0.8, 0.1, 0.0], z: 3 }, Nucleus { pos: [2.2, 0.0, 0.0], z: 1 }], 0, None))];
        for ((_, sys), (e_bits, f_bits)) in scenes().into_iter().zip([(BASE_H2_E, BASE_H2_FX), (BASE_LIH_E, BASE_LIH_FX)]) {
            assert!(sys.ionic.is_none());
            let (st, stats) = scf(&g, &sys, &aufbau, None).unwrap();
            let f = crate::forces(&g, &st, &sys.nuclei, aufbau.sigma).unwrap();
            assert_eq!(stats.energy.total.to_bits(), e_bits);
            assert_eq!(f[0][0].to_bits(), f_bits);
            assert!(stats.core_overlaps.is_empty() && st.ionic.is_none());
        }
        for ((name, sys), (e_bits, f_bits)) in scenes().into_iter().zip([(FD_H2_E, FD_H2_FX), (FD_LIH_E, FD_LIH_FX)]) {
            let (st, stats) = scf(&g, &sys, &fd, None).unwrap();
            let f = crate::forces(&g, &st, &sys.nuclei, fd.sigma).unwrap();
            eprintln!("{name} (FD, for_grid): F {:+.12} (TS {:.3e}, E {:+.12}, {} steps, stored {} / {}) F_x0 {:+.12}",
                      stats.energy.total, stats.energy.ts, stats.energy.internal, stats.iterations, st.orbitals_up.len(), st.orbitals_dn.len(), f[0][0]);
            eprintln!("    const FD_{name}_E: u64 = 0x{:016x}; // {:+.12}", stats.energy.total.to_bits(), stats.energy.total);
            eprintln!("    const FD_{name}_FX: u64 = 0x{:016x}; // {:+.12}", f[0][0].to_bits(), f[0][0]);
            assert_eq!(stats.energy.total.to_bits(), e_bits, "{name} FD energy");
            assert_eq!(f[0][0].to_bits(), f_bits, "{name} FD F_x0");
            assert!(stats.core_overlaps.is_empty() && st.ionic.is_none() && st.occupations == fd.occupations);
        }
    }

    /// Review Focus 2: a warm state converged under another ionic setup (H under None, then H under All —
    /// same grid, same orbital counts) is ignored: the solve equals a cold one bit for bit.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
    fn warm_state_from_another_ionic_setup_is_not_reused() {
        use crate::ionic::{Ionic, IonicPolicy, IonicSetup};
        use crate::radial::test_cache::filtered_at;
        let h = 0.25;
        let g = Grid::cube(33, h);
        let s = ScfSettings::for_grid(&g);
        let nuc = vec![Nucleus { pos: [0.1, 0.0, 0.0], z: 1 }];
        let (ae_state, _) = scf(&g, &System::new(nuc.clone(), 0, None), &s, None).unwrap();
        let setup = std::sync::Arc::new(IonicSetup::from_tables(IonicPolicy::All, h, h, std::collections::BTreeMap::from([(1, Ionic::Pseudo(filtered_at(1, h)))])));
        let sys = System::with_ionic(nuc, 0, None, setup);
        let (pseudo_state, warm) = scf(&g, &sys, &s, Some(&ae_state)).unwrap();
        let (_, cold) = scf(&g, &sys, &s, None).unwrap();
        assert_eq!(warm.energy.total.to_bits(), cold.energy.total.to_bits());
        assert_eq!(warm.iterations, cold.iterations);
        // and the other way round: a pseudo state is no warm start for the all-electron solve
        let (_, ae_cold) = scf(&g, &System::new(vec![Nucleus { pos: [0.1, 0.0, 0.0], z: 1 }], 0, None), &s, None).unwrap();
        let (_, ae_warm) = scf(&g, &System::new(vec![Nucleus { pos: [0.1, 0.0, 0.0], z: 1 }], 0, None), &s, Some(&pseudo_state)).unwrap();
        assert_eq!(ae_warm.energy.total.to_bits(), ae_cold.energy.total.to_bits());
        // a state under the same physics *is* reused (the guard is not a blanket refusal)
        let (_, rewarm) = scf(&g, &sys, &s, Some(&pseudo_state)).unwrap();
        assert!(rewarm.iterations < cold.iterations, "same setup: warm {} vs cold {}", rewarm.iterations, cold.iterations);
        eprintln!("pseudo-H (All) at h = {h}: cold {} steps, E {:.8}; from the AE state {} steps; from its own state {} steps",
                  cold.iterations, cold.energy.total, warm.iterations, rewarm.iterations);
    }

    /// Review Focus 3: an ion exactly on a grid point gives finite fields and a converged SCF. Solved at
    /// tol = eig_tol = 1e-9 (plan review S6), because Task 7 extends this test with the force, whose
    /// zero by inversion symmetry holds only to the eigensolver residual.
    /// The open-shell O triplet is solved with `DegenerateShells` (spec §5.7, the providers' choice under the
    /// pseudo policies): under integer aufbau it does not converge (Task 6 report: Δρ 5e-4–1e-2 after 46
    /// steps here). The minority 2p cluster must come out shared (1/3 each) and tight (spread < τ/10).
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
    fn ion_on_a_grid_point_is_finite() {
        use crate::ionic::{Ionic, IonicPolicy, IonicSetup};
        use crate::radial::test_cache::{filtered_at, table};
        let t = table(8);
        let h = t.record.h_req.unwrap();
        let n = (2.0 * (6.0 / h).round()) as usize + 1;         // a ≈ 12 Bohr box with a grid point at the origin
        let g = Grid::cube(n, h);
        let setup = std::sync::Arc::new(IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, std::collections::BTreeMap::from([(8, Ionic::Pseudo(filtered_at(8, h)))])));
        let nuc = vec![Nucleus { pos: [0.0; 3], z: 8 }];
        assert!(crate::nuclear::ionic_local_potential(&g, &nuc, &setup).iter().all(|v| v.is_finite()));
        assert!(crate::nuclear::partial_core_field(&g, &nuc, &setup).iter().all(|v| v.is_finite()));
        let sys = System::with_ionic(nuc, 0, Some(3), setup);
        let mut s = ScfSettings::for_grid(&g);
        (s.tol, s.eig_tol) = (1e-9, 1e-9);
        s.occupations = Occupations::DegenerateShells { tau: DEGENERACY_TAU };
        let (st, stats) = scf(&g, &sys, &s, None).expect("converges");
        eprintln!("pseudo-O triplet on a grid point, h = {h}, {n}³: E {:.8} in {} SCF steps, max eigen residual {:.2e}, shared {:?}, occ ↑ {:?} ↓ {:?}, warnings {:?}",
                  stats.energy.total, stats.iterations, stats.max_eigen_residual, stats.fractional, st.occ_up, st.occ_dn, stats.warnings);
        assert!(stats.energy.total.is_finite() && sys.n_up == 4 && sys.n_dn == 2);
        assert_eq!(stats.fractional.len(), 1, "one shared cluster: the minority 2p");
        let c = &stats.fractional[0];
        assert!(c.spin == 1 && c.first == 1 && c.count == 3 && (c.f - 1.0 / 3.0).abs() < 1e-15 && c.spread < DEGENERACY_TAU / 10.0, "{c:?}");
        // M7 Task 7 (plan review S6): every force integrand is odd about R, so the force vanishes for a parity-pure
        // state; the iterative eigenvectors carry a parity admixture of first order in the eigensolver residual, so
        // |F| ~ residual·|∇V|, bounded by 1e-6 at tol = eig_tol = 1e-9
        let f = crate::forces(&g, &st, &sys.nuclei, s.sigma).unwrap();
        eprintln!("pseudo-O on a grid point: F = ({:+.2e}, {:+.2e}, {:+.2e}) Ha/Bohr", f[0][0], f[0][1], f[0][2]);
        assert!(f[0].iter().all(|x| x.is_finite() && x.abs() < 1e-6), "{:?}", f[0]);
    }

    /// Spec §5.6, the CPU SCF's half (Task 7 review M2; `validate`'s `core_overlap_warns_and_still_steps` solves on the GPU
    /// when there is one): two pseudo-O cores pushed to 0.8·(r_core + r_core) are reported in `ScfStats.core_overlaps` and
    /// as one warning starting with `CORE_OVERLAP_WARNING`, and the step is taken. One SCF step (tol = ∞) on 33³ at
    /// h = 0.20: the warning is emitted after the loop whatever the convergence.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
    fn cpu_scf_warns_on_core_overlap() {
        use crate::ionic::{Ionic, IonicPolicy, IonicSetup, CORE_OVERLAP_WARNING};
        use crate::radial::test_cache::filtered_at;
        let h = 0.2;
        let g = Grid::cube(33, h);
        let t = filtered_at(8, h);
        let r = 0.8 * 2.0 * t.r_core;
        let setup = std::sync::Arc::new(IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, std::collections::BTreeMap::from([(8, Ionic::Pseudo(t))])));
        let sys = System::with_ionic(vec![Nucleus { pos: [-0.5 * r, 0.03, 0.0], z: 8 }, Nucleus { pos: [0.5 * r, 0.03, 0.0], z: 8 }], 0, None, setup);
        let mut s = ScfSettings::for_grid(&g);
        (s.tol, s.max_iter, s.eig_tol) = (f64::INFINITY, 1, 1e-6);
        let (st, stats) = scf(&g, &sys, &s, None).expect("one step is taken");
        assert_eq!(stats.core_overlaps.len(), 1, "{:?}", stats.core_overlaps);
        let (i, j, d) = stats.core_overlaps[0];
        assert!((i, j) == (0, 1) && (d - r).abs() < 1e-12, "{:?}", stats.core_overlaps);
        let warned: Vec<&String> = stats.warnings.iter().filter(|w| w.starts_with(CORE_OVERLAP_WARNING)).collect();
        assert_eq!(warned.len(), 1, "{:?}", stats.warnings);
        assert_eq!(warned[0], &format!("{CORE_OVERLAP_WARNING}: 0–1 at R = {r:.3} Bohr"));
        assert!(stats.energy.total.is_finite() && st.rho().iter().all(|x| x.is_finite()));
        assert!(crate::forces(&g, &st, &sys.nuclei, s.sigma).unwrap().iter().flatten().all(|x| x.is_finite()));
    }

    /// Review Focus 3, the fields half of `ion_on_a_grid_point_is_finite`: an ion exactly on a grid point (d = 0
    /// at one point: V_LR's on-site limit, V_SR, ρ_pc and t_l read at the uniform tables' index 0) gives finite fields.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
    fn ion_on_a_grid_point_fields_are_finite() {
        use crate::ionic::{Ionic, IonicPolicy, IonicSetup};
        use crate::radial::test_cache::{filtered_at, table};
        let h = table(8).record.h_req.unwrap();
        let n = (2.0 * (6.0 / h).round()) as usize + 1;
        let g = Grid::cube(n, h);
        let setup = IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, std::collections::BTreeMap::from([(8, Ionic::Pseudo(filtered_at(8, h)))]));
        let nuc = [Nucleus { pos: [0.0; 3], z: 8 }];
        let c = g.idx(n / 2, n / 2, n / 2);
        assert!(g.point(c).iter().all(|x| x.abs() < 1e-12), "a grid point at the ion: {:?}", g.point(c));
        let v = crate::nuclear::ionic_local_potential(&g, &nuc, &setup);
        let pc = crate::nuclear::partial_core_field(&g, &nuc, &setup);
        let nl = crate::nonlocal::NonlocalOp::build(&g, &nuc, &setup);
        assert!(v.iter().chain(&pc).all(|x| x.is_finite()));
        assert!(nl.spheres.iter().all(|s| s.beta.iter().flatten().all(|x| x.is_finite())));
        let k = nl.spheres[0].points.iter().position(|&p| p == c).expect("the ion's own point is in its sphere");
        eprintln!("O on a grid point, h = {h}: V_loc(0) = {:.6} Ha, ρ_pc(0) = {:.6} e/Bohr³, β_s(0) = {:.6}", v[c], pc[c], nl.spheres[0].beta[0][k]);
    }

    /// Spec §5.2: an all-electron species inside a setup is today's smeared nucleus. H₂ and LiH under a setup whose
    /// every species is all-electron at σ_AE = σ reproduce the `None` path. The two paths differ only in the
    /// rounding of the potential (`smeared_potential` vs `nuclear_potential`, 1e-16 relative) and the cold
    /// guess, so they stop at different points within Δρ < 1e-6; the energy is second order in that residual
    /// (module doc), so 1e-9 Ha.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
    fn all_electron_species_in_a_setup_are_todays_physics() {
        use crate::ionic::{Ionic, IonicPolicy, IonicSetup};
        let g = Grid::cube(33, 0.25);
        let s = ScfSettings::for_grid(&g);
        for nuc in [vec![Nucleus { pos: [0.7, 0.0, 0.0], z: 1 }, Nucleus { pos: [-0.7, 0.0, 0.0], z: 1 }],
                    vec![Nucleus { pos: [-0.8, 0.1, 0.0], z: 3 }, Nucleus { pos: [2.2, 0.0, 0.0], z: 1 }]] {
            let per_z = nuc.iter().map(|n| (n.z, Ionic::AllElectron { sigma: s.sigma })).collect();
            let setup = std::sync::Arc::new(IonicSetup::from_tables(IonicPolicy::None, g.h, s.sigma, per_z));
            let (_, plain) = scf(&g, &System::new(nuc.clone(), 0, None), &s, None).unwrap();
            let (st, with) = scf(&g, &System::with_ionic(nuc, 0, None, setup), &s, None).unwrap();
            eprintln!("all-electron setup vs None: E {:.12} vs {:.12} (Δ {:+.1e}), {} vs {} SCF steps",
                      with.energy.total, plain.energy.total, with.energy.total - plain.energy.total, with.iterations, plain.iterations);
            assert!((with.energy.total - plain.energy.total).abs() < 1e-9);
            assert!((with.energy.nuclear - plain.energy.nuclear).abs() < 1e-15 && with.core_overlaps.is_empty() && st.ionic.is_some());
        }
    }

    /// Spec §5.7's rule on synthetic levels (any order in, the same order out): a three-fold cluster with one
    /// electron is shared 1/3 each; a full cluster is integer; a degenerate pair across ε_F shares the
    /// remainder; a cluster that reaches the last vector asks for another vector; n = 0 is empty.
    #[test]
    fn degenerate_shell_occupations_on_levels() {
        let tau = DEGENERACY_TAU;
        // O-triplet minority: s, then p ×3 within 1e-7, then a witness 0.3 Ha up; listed out of order
        let (f, c) = degenerate_shell_occupations(&[-0.27, -0.79, 0.07, -0.27 + 1e-7, -0.27 - 5e-8], 2, tau).unwrap();
        assert_eq!(f[1], 1.0);
        for i in [0, 3, 4] { assert!((f[i] - 1.0 / 3.0).abs() < 1e-15, "{f:?}"); }
        assert_eq!(f[2], 0.0);
        let (first, count, fc, spread) = c.unwrap();
        assert_eq!((first, count), (1, 3));
        assert!((fc - 1.0 / 3.0).abs() < 1e-15 && (spread - 1.5e-7).abs() < 1e-15);
        // a full cluster (Ne p⁶ per spin: 1 + 3 electrons) is aufbau's
        let (f, c) = degenerate_shell_occupations(&[-1.3, -0.5, -0.5, -0.5, 0.05], 4, tau).unwrap();
        assert_eq!((f, c), (vec![1.0, 1.0, 1.0, 1.0, 0.0], None));
        // two of three shared: 2/3 each
        let (f, _) = degenerate_shell_occupations(&[-0.9, -0.34, -0.34, -0.34, 0.04], 3, tau).unwrap();
        assert!(f[1..4].iter().all(|x| (x - 2.0 / 3.0).abs() < 1e-15) && f[0] == 1.0 && f[4] == 0.0);
        // levels farther apart than τ are aufbau's even when close (2τ)
        let (f, c) = degenerate_shell_occupations(&[-0.5, -0.5 + 2.0 * tau, 0.1], 1, tau).unwrap();
        assert_eq!((f, c), (vec![1.0, 0.0, 0.0], None));
        // the witness inside the cluster: ask for another vector
        assert!(degenerate_shell_occupations(&[-0.79, -0.27, -0.27], 2, tau).is_none());
        assert!(degenerate_shell_occupations(&[-0.79, -0.27, -0.27 + 0.5 * tau], 2, tau).is_none());
        assert_eq!(degenerate_shell_occupations(&[], 0, tau), Some((vec![], None)));
        // phase 1 uses the candidate window τ_c = 0.1: an O-like minority p set split by 5 mHa is still one candidate
        let (f, c) = degenerate_shell_occupations(&[-0.80, -0.275, -0.27, -0.27, 0.07], 2, WITNESS_GAP).unwrap();
        assert!(f[1..4].iter().all(|x| (x - 1.0 / 3.0).abs() < 1e-15) && (c.unwrap().3 - 0.005).abs() < 1e-12);
        // phase 2 releases (spec §5.7): split at neighbour gaps ≥ τ, aufbau over the groups
        assert_eq!(released_occupations(&[-0.30, -0.30 + 0.2 * tau, -0.30 + 3.0 * tau], 1.0, tau), vec![0.5, 0.5, 0.0]);
        assert_eq!(released_occupations(&[-0.30, -0.30 + 3.0 * tau, -0.30 + 6.0 * tau], 2.0, tau), vec![1.0, 1.0, 0.0]);
        assert_eq!(released_occupations(&[-0.30, -0.30 + 3.0 * tau, -0.30 + 6.0 * tau], 1.0, tau), vec![1.0, 0.0, 0.0]);
        let f = released_occupations(&[-0.30, -0.30 + 0.5 * tau, -0.30 + 0.9 * tau], 1.0, tau);
        assert!(f.iter().all(|x| (x - 1.0 / 3.0).abs() < 1e-15), "gaps < τ: one group, shared: {f:?}");
        // decide: a shared cluster of spread ≥ τ is released, one below τ is kept
        let evals = [-0.80, -0.27, -0.27 + 5.0 * tau, -0.27 + 5.0 * tau, 0.07];
        let shell = FractionalShell { spin: 1, first: 1, count: 3, f: 1.0 / 3.0, spread: 5.0 * tau };
        let mut rel = Vec::new();
        let by_rank = decide(&evals, &[1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0], &[shell.clone()], tau, &mut rel);
        assert_eq!((by_rank, rel), (vec![1.0, 1.0, 0.0, 0.0, 0.0], vec![shell]));
        let kept = FractionalShell { spin: 1, first: 1, count: 3, f: 1.0 / 3.0, spread: 0.05 * tau };
        let mut rel = Vec::new();
        let by_rank = decide(&[-0.80, -0.27, -0.27, -0.27, 0.07], &[1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0], &[kept], tau, &mut rel);
        assert!(rel.is_empty() && (by_rank[1] - 1.0 / 3.0).abs() < 1e-15);
    }

    /// Spec §5.7: the rule is inert on closed shells. H₂ and LiH (as in `none_policy_is_bitwise_today`) under
    /// `DegenerateShells` get integer occupations (zero-weight witnesses above τ_c) and the `Aufbau` energy
    /// to 1e-9 Ha: the two runs converge the same discrete problem to Δρ < 1e-6 along different LOBPCG
    /// paths, and E is second order in that residual. A warm state converged under the other scheme is
    /// ignored (the same bits as a cold solve).
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
    fn closed_shells_are_aufbau_under_the_rule() {
        let g = Grid::cube(33, 0.25);
        // `Aufbau` explicitly: `for_grid` defaults to FermiDirac since plan Task 7 (this test is DS against aufbau)
        let aufbau = ScfSettings { occupations: Occupations::Aufbau, ..ScfSettings::for_grid(&g) };
        let mut shared = ScfSettings::for_grid(&g);
        shared.occupations = Occupations::DegenerateShells { tau: DEGENERACY_TAU };
        for nuc in [vec![Nucleus { pos: [0.7, 0.0, 0.0], z: 1 }, Nucleus { pos: [-0.7, 0.0, 0.0], z: 1 }],
                    vec![Nucleus { pos: [-0.8, 0.1, 0.0], z: 3 }, Nucleus { pos: [2.2, 0.0, 0.0], z: 1 }]] {
            let sys = System::new(nuc, 0, None);
            let (sa, a) = scf(&g, &sys, &aufbau, None).unwrap();
            let (st, b) = scf(&g, &sys, &shared, None).unwrap();
            eprintln!("closed shell under the rule: E {:.12} vs aufbau {:.12} (Δ {:+.1e}), {} vs {} steps, occ ↑ {:?} ↓ {:?}",
                      b.energy.total, a.energy.total, b.energy.total - a.energy.total, b.iterations, a.iterations, st.occ_up, st.occ_dn);
            assert!(b.fractional.is_empty());
            for (occ, n) in [(&st.occ_up, sys.n_up), (&st.occ_dn, sys.n_dn)] {
                assert!(occ.len() > n, "at least one witness: {occ:?}");
                assert!(occ[..n].iter().all(|&f| f == 1.0) && occ[n..].iter().all(|&f| f == 0.0), "{occ:?}");
            }
            assert!(b.released.is_empty(), "{:?}", b.released);
            assert!((b.energy.total - a.energy.total).abs() < 1e-9);
            assert_eq!((sa.occ_up.len(), sa.occupations), (sys.n_up, Occupations::Aufbau));
            // the scheme guard, both ways
            let cold_b = b.energy.total;
            let (_, from_a) = scf(&g, &sys, &shared, Some(&sa)).unwrap();
            assert_eq!(from_a.energy.total.to_bits(), cold_b.to_bits());
            let (_, from_b) = scf(&g, &sys, &aufbau, Some(&st)).unwrap();
            assert_eq!(from_b.energy.total.to_bits(), a.energy.total.to_bits());
        }
    }

    /// N = Σ q_I − charge (spec §5.2): Z_v for pseudised species, Z for all-electron ones; `System::new` is
    /// `valence_electrons` without a setup (today's count).
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
    fn electron_counts_use_valence_charges() {
        use crate::ionic::{Ionic, IonicPolicy, IonicSetup};
        use crate::radial::test_cache::filtered_at;
        let h = 0.2;
        let water = vec![Nucleus { pos: [0.0; 3], z: 8 }, Nucleus { pos: [1.8, 0.0, 0.0], z: 1 }, Nucleus { pos: [-0.45, 1.75, 0.0], z: 1 }];
        assert_eq!(valence_electrons(&water, None, 0), 10);
        let setup = std::sync::Arc::new(IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, std::collections::BTreeMap::from([
            (1, Ionic::AllElectron { sigma: h }), (8, Ionic::Pseudo(filtered_at(8, h)))])));
        assert_eq!(valence_electrons(&water, Some(&setup), 0), 8);
        assert_eq!(valence_electrons(&water, Some(&setup), -1), 9);
        let s = System::with_ionic(water.clone(), 0, None, setup.clone());
        assert_eq!((s.n_up, s.n_dn), (4, 4));
        let s = System::with_ionic(vec![water[0]], 0, Some(3), setup);
        assert_eq!((s.n_up, s.n_dn), (4, 2));
        let s = System::new(water, 1, None);
        assert_eq!((s.n_up, s.n_dn, s.ionic.is_none()), (5, 4, true));
    }

    /// Spec §5.5: `scf` refuses a setup filtered for another h or with another σ_AE than the run's (a
    /// programming error, not a physics outcome).
    #[test]
    #[should_panic(expected = "IonicSetup h")]
    fn a_setup_for_another_h_is_refused() {
        use crate::ionic::{Ionic, IonicPolicy, IonicSetup};
        let g = Grid::cube(9, 0.25);
        let setup = std::sync::Arc::new(IonicSetup::from_tables(IonicPolicy::None, 0.3, 0.25, std::collections::BTreeMap::from([(1, Ionic::AllElectron { sigma: 0.25 })])));
        let _ = scf(&g, &System::with_ionic(vec![Nucleus { pos: [0.0; 3], z: 1 }], 0, None, setup), &ScfSettings::for_grid(&g), None);
    }

    /// Prints the bits Task 6's `none_policy_is_bitwise_today` pins (run once, before the M7 change).
    #[test]
    #[ignore = "M7 Task 6 Step 1 only"]
    fn none_policy_baseline() {
        let g = Grid::cube(33, 0.25);
        let s = ScfSettings::for_grid(&g);
        for (name, sys) in [("H2", System::new(vec![Nucleus { pos: [0.7, 0.0, 0.0], z: 1 }, Nucleus { pos: [-0.7, 0.0, 0.0], z: 1 }], 0, None)),
                            ("LIH", System::new(vec![Nucleus { pos: [-0.8, 0.1, 0.0], z: 3 }, Nucleus { pos: [2.2, 0.0, 0.0], z: 1 }], 0, None))] {
            let (st, stats) = scf(&g, &sys, &s, None).unwrap();
            let f = crate::forces(&g, &st, &sys.nuclei, s.sigma).unwrap();
            println!("const BASE_{name}_E: u64 = 0x{:016x};", stats.energy.total.to_bits());
            println!("const BASE_{name}_FX: u64 = 0x{:016x}; // {} SCF steps", f[0][0].to_bits(), stats.iterations);
        }
    }

    // ---- σ → 0 extrapolation helper ----

    #[test]
    fn extrapolate_sigma_recovers_exact_quartic() {
        // E(σ) = a + bσ² + cσ³ + dσ⁴ sampled at four σ must return a to rounding.
        let (a, b, c, d) = (-0.478671, 2.0, -4.2567, 6.0);
        let sig = [0.25, 0.30, 0.35, 0.40];
        let e = sig.map(|s| a + b * s * s + c * s * s * s + d * s * s * s * s);
        assert_relative_eq!(extrapolate_sigma(&sig, &e), a, epsilon = 1e-12);
    }

    /// Exact first-order smearing shift of the hydrogen 1s density (module doc):
    /// ΔE₁(σ) = 1 − 2a/√π + (2a² − 1) e^{a²} erfc(a), a = σ√2.
    fn h_first_order_shift(sigma: f64) -> f64 {
        let a = sigma * 2f64.sqrt();
        let e = (a * a).exp() * libm::erfc(a);
        1.0 - 2.0 * a / std::f64::consts::PI.sqrt() + (2.0 * a * a - 1.0) * e
    }

    #[test]
    fn extrapolation_model_error_on_exact_first_order_shift() {
        // The fit model is the small-σ expansion 2σ² − 4.257σ³ + 6σ⁴ − 6.81σ⁵ + …, an alternating
        // series with ratio ≈ 2σ, so at σ ∈ [0.25, 0.40] the dropped σ⁵+ terms are a few mHa and
        // leak into a. Measured on the exact (noise-free) ΔE₁: a = +1.73 mHa where the true limit
        // is 0. This is the model's intrinsic bias at this σ range and is part of the 15 mHa
        // budget of `hydrogen_energy_extrapolated_to_point_nucleus`; the test pins it so a change
        // in the model shows up here, not in the solver test.
        let sig = [0.25, 0.30, 0.35, 0.40];
        assert_relative_eq!(h_first_order_shift(0.25), 0.0766, epsilon = 5e-5);
        assert_relative_eq!(h_first_order_shift(0.30), 0.1010, epsilon = 5e-5);
        assert_relative_eq!(h_first_order_shift(0.35), 0.1262, epsilon = 5e-5);
        let a = extrapolate_sigma(&sig, &sig.map(h_first_order_shift));
        assert!(a > 0.0 && a < 2.0e-3, "model bias on the exact first-order shift = {a:.3e} Ha");
    }

    /// Diagnostic, not a test (ignored by default): prints the lowest 6 spin-up eigenvalues of
    /// H_up built from a loosely converged Li density, their radial extents and face amplitudes,
    /// to inspect the 2s–2p gap behind Task 9's Li convergence finding. Run with
    /// `cargo test -p qm scf --release -- --ignored --nocapture li_spectrum_diagnostic`.
    #[test]
    #[ignore]
    fn li_spectrum_diagnostic() {
        use crate::eigen::{jacobi_precond, lobpcg, LocalHamiltonian};
        let g = Grid::cube(49, 0.25);
        let li = System::new(vec![Nucleus { pos: [0.0; 3], z: 3 }], 0, None);
        let mut s = ScfSettings::for_grid(&g);
        s.tol = 2e-2;
        let (state, st) = scf(&g, &li, &s, None).unwrap();
        eprintln!("loose Li: iters {} dRho {:.2e} E {:+.6} face {:.2e} ev_up {:?} ev_dn {:?}",
            st.iterations, st.delta_rho, st.energy.total, st.boundary_density, state.evals_up, state.evals_dn);
        let xc = lsda_pw92(&state.rho_up, &state.rho_dn);
        let v: Vec<f64> = (0..g.len()).map(|i| state.v_nuc[i] + state.v_h[i] + xc.v_up[i]).collect();
        let op = LocalHamiltonian { grid: &g, v: &v };
        let mut x = initial_orbitals(&g, &li.nuclei, 6, 7);
        let mut e = vec![0.0; 6];
        let pc = jacobi_precond(&g, &v, state.evals_up[0]);
        let es = lobpcg(&g, &op, &mut x, &mut e, Some(&pc), 1e-6, 500, None);
        eprintln!("H_up lowest 6: {e:?}  ({es:?})");
        // radial extent of each state: <r^2>^(1/2)
        for (i, psi) in x.iter().enumerate() {
            let r2: Vec<f64> = (0..g.len()).map(|j| { let p = g.point(j); psi[j] * psi[j] * (p[0]*p[0]+p[1]*p[1]+p[2]*p[2]) }).collect();
            eprintln!("  state {i}: e {:+.5}  sqrt<r^2> {:.3}  max|psi| on faces {:.2e}", e[i], g.integrate(&r2).sqrt(), max_on_faces(&g, psi));
        }
    }

    /// Diagnostic, not a test (ignored by default): prints the H E(σ) table with energy
    /// components, the σ → 0 fit against the exact first-order shift, the egg-box ΔE, the
    /// small-box energy, the Li outcome and cold/warm iteration counts with wall times. Run with
    /// `cargo test -p qm scf --release -- --ignored --nocapture report_numbers`.
    #[test]
    #[ignore]
    fn report_numbers() {
        use std::time::Instant;
        let g = Grid::cube(49, 0.25);
        let sig = [0.25, 0.30, 0.35, 0.40];
        let mut e = [0.0; 4];
        eprintln!("H atom, 49^3, h = 0.25, tol 1e-7:");
        for (k, &s) in sig.iter().enumerate() {
            let t0 = Instant::now();
            let (et, st) = h_energy(&g, s, [0.0; 3]);
            e[k] = et;
            let d1 = h_first_order_shift(s);
            eprintln!("  sigma {s:.2}  E {et:+.6}  {:?}  iters {}  dRho {:.2e}  face {:.2e}  {:.2}s  dE1(exact 1st order) {d1:.4}",
                st.energy, st.iterations, st.delta_rho, st.boundary_density, t0.elapsed().as_secs_f64());
        }
        let a = extrapolate_sigma(&sig, &e);
        eprintln!("  extrapolated a = {a:+.6}  (NIST LSD −0.478671; miss {:+.2} mHa)", (a + 0.478671) * 1e3);
        for (k, &s) in sig.iter().enumerate() {
            eprintln!("  sigma {s:.2}: E − a = {:+.5}  vs exact first-order {:+.5}  diff {:+.2} mHa",
                e[k] - a, h_first_order_shift(s), (e[k] - a - h_first_order_shift(s)) * 1e3);
        }
        let t0 = Instant::now();
        let (e1, _) = h_energy(&g, 0.25, [0.125, 0.0, 0.0]);
        eprintln!("egg-box: E(0.125) − E(0) = {:+.3e} Ha  ({:.2}s)", e1 - e[0], t0.elapsed().as_secs_f64());
        let small = Grid::cube(25, 0.25);
        let (es, st) = h_energy(&small, 0.25, [0.0; 3]);
        eprintln!("small box 25^3: E {es:+.6} (ΔE {:+.3e}), face {:.2e}, warnings {:?}", es - e[0], st.boundary_density, st.warnings);
        let t0 = Instant::now();
        let li = System::new(vec![Nucleus { pos: [0.0; 3], z: 3 }], 0, None);
        match scf(&g, &li, &ScfSettings::for_grid(&g), None) {
            Ok((state, st)) => eprintln!("Li 49^3: E {:+.6} {:?} iters {} dRho {:.2e} ev_up {:?} ev_dn {:?} N {:.9} face {:.2e} {:.2}s",
                st.energy.total, st.energy, st.iterations, st.delta_rho, state.evals_up, state.evals_dn,
                g.integrate(&state.rho()), st.boundary_density, t0.elapsed().as_secs_f64()),
            Err(e) => eprintln!("Li 49^3: {e} ({:.2}s)", t0.elapsed().as_secs_f64()),
        }
        let g33 = Grid::cube(33, 0.25);
        let s = ScfSettings::for_grid(&g33);
        let t0 = Instant::now();
        let (state, cold) = scf(&g33, &hydrogen([0.0; 3]), &s, None).unwrap();
        let tc = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let (_, warm) = scf(&g33, &hydrogen([0.02, 0.0, 0.0]), &s, Some(&state)).unwrap();
        eprintln!("warm start 33^3: cold {} iters ({tc:.2}s, E {:+.6}), warm {} iters ({:.2}s, E {:+.6})",
            cold.iterations, cold.energy.total, warm.iterations, t0.elapsed().as_secs_f64(), warm.energy.total);
    }

    /// The stall rule on synthetic Δρ sequences (module doc, "The stall stop"): a plateau after a
    /// converging start stops exactly W iterations after the last 2× gain; slow but steady progress
    /// (2× every 7 iterations, a factor 0.906 per iteration) never stops; an oscillation whose best
    /// does not halve stops; a restart (the shell release) gives a fresh window.
    #[test]
    fn stall_watch_stops_a_plateau_and_never_a_steady_crawl() {
        let w = 50;
        // converging to 1e-3 over 10 iterations (a gain every iteration), then flat at 1e-3 ± 30 % (level 1e-4: above it)
        let mut s = StallWatch::new(w, 1e-4);
        let mut stop = None;
        for i in 1..=400 {
            let d = if i <= 10 { 0.5f64.powi(i as i32) } else { 0.5f64.powi(10) * (1.0 + 0.3 * ((i as f64) * 1.7).sin()) };
            let k = s.observe(i, d);
            if s.stalled(k) { stop = Some((i, k)); break; }
        }
        // the last gain: iteration 10 (or a dip in the plateau ≤ ½ of the anchor — none: −30 % at most)
        assert_eq!(stop, Some((10 + w, w)), "plateau");
        assert!(s.best() <= 0.5f64.powi(10) && s.best() > 0.69 * 0.5f64.powi(10));
        // slow but steady: never stops in 400 iterations (the cap is the backstop)
        let mut s = StallWatch::new(w, 0.0);
        for i in 1..=400 {
            let k = s.observe(i, 0.906f64.powi(i as i32));
            assert!(k < 8 && !s.stalled(k), "steady crawl stopped at {i} (since gain {k})");
        }
        // an oscillation between 2e-4 and 3e-4 after a gain to 2e-4 at iteration 3 (level 1e-4)
        let mut s = StallWatch::new(w, 1e-4);
        let seq = |i: usize| if i < 3 { 1e-2 / i as f64 } else if i % 2 == 0 { 3e-4 } else { 2e-4 };
        let first = (1..=400).find(|&i| { let k = s.observe(i, seq(i)); s.stalled(k) });
        assert_eq!(first, Some(3 + w));
        // ruling (a): the same oscillation below the level never stalls — it converges or runs to the cap
        let mut s = StallWatch::new(w, 1e-3);
        assert!((1..=400).all(|i| { let k = s.observe(i, seq(i)); !s.stalled(k) }), "a plateau below the level must never stall");
        // a level exactly at the best still stalls (≥)
        let mut s = StallWatch::new(w, 2e-4);
        assert_eq!((1..=400).find(|&i| { let k = s.observe(i, seq(i)); s.stalled(k) }), Some(3 + w));
        // a restart opens a fresh window; W = 0 never stalls
        let mut s = StallWatch::new(5, 0.0);
        for i in 1..=4 { s.observe(i, 1e-3); }
        s.restart(4);
        assert_eq!(s.observe(5, f64::INFINITY), 1, "∞ (the release's reset Δρ) is no gain");
        assert_eq!(s.observe(6, 0.4), 0, "the first finite Δρ after a restart anchors");
        let off = StallWatch::new(0, 0.0);
        assert!(!off.stalled(10_000));
        // the error carries what the message needs
        let e = s.error(77, 1e-6);
        assert!(matches!(e, QmError::ScfStalled { iterations: 77, window: 5, .. }), "{e:?}");
        assert!(e.to_string().starts_with("SCF stalled: Δρ best 4.0e-1 not improved 2× in 5 iterations, still above the stall level 0.0e0 (target 1.00e-6"), "{e}");
    }

    /// Review M2 / re-review r1-m3: the CPU loop's decision after Δρ. A real NaN solve cannot be built safely (a NaN
    /// geometry or potential reaches `nalgebra::SymmetricEigen`, which has no iteration limit; a NaN mixing factor is
    /// clamped to 0 by `diis_mix`), so the decision is a function, tested here: a non-finite Δρ aborts whatever the
    /// stall watch says, a Δρ below tol is the convergence test even when the watch says stalled, and only then the
    /// stall stop.
    #[test]
    fn a_non_finite_delta_rho_aborts_before_any_stall() {
        use AfterDeltaRho::*;
        for d in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(after_delta_rho(d, 1e-6, true), Abort, "{d}");
            assert_eq!(after_delta_rho(d, 1e-6, false), Abort, "{d}");
        }
        assert_eq!(after_delta_rho(5e-7, 1e-6, true), Converged, "below tol is never a stall");
        assert_eq!(after_delta_rho(5e-7, 1e-6, false), Converged);
        assert_eq!(after_delta_rho(1e-6, 1e-6, true), Stall, "tol itself is not converged (strict <)");
        assert_eq!(after_delta_rho(0.3, 1e-6, true), Stall);
        assert_eq!(after_delta_rho(0.3, 1e-6, false), Go);
        // and a watch fed only non-finite Δρ never anchors (a guard: the loop aborts before it matters)
        let mut w = StallWatch::new(3, 0.0);
        assert_eq!((1..=5).map(|i| w.observe(i, f64::NAN)).last(), Some(5));
        assert!(w.best().is_infinite());
    }

    /// A real solve that cannot converge stops at the stall window, not the cap: with mixing α = 0
    /// the input density never moves (linear mixing and DIIS both return ρ_in), so Δρ stays at the
    /// first iteration's value and the solve ends W iterations after it with `ScfStalled`.
    #[test]
    fn a_frozen_density_stalls_before_the_cap() {
        let g = Grid::cube(21, 0.4);
        let mut s = ScfSettings::for_grid(&g);
        s.mixing = 0.0;
        s.stall_window = 6;
        let mut seen = Vec::new();
        let r = scf_observed(&g, &hydrogen([0.0; 3]), &s, None, Some(&mut |p: &ScfProgress| seen.push(*p)));
        println!("α = 0: {:?}; Δρ {:?}", r.as_ref().err(), seen.iter().map(|p| p.delta_rho).collect::<Vec<_>>());
        match r {
            Err(QmError::ScfStalled { iterations, best_delta_rho, tol, window: 6, level }) => {
                assert_eq!(iterations, 7, "anchored at iteration 1, stopped at 1 + W");
                assert!(best_delta_rho > 0.1 && tol == s.tol);
                assert_eq!(level, SCF_STALL_LEVEL_PER_ELECTRON, "one electron");
            }
            other => panic!("expected ScfStalled, got {:?}", other.map(|(_, st)| st)),
        }
        assert_eq!(seen.len(), 7);
        assert_eq!(seen.last().map(|p| (p.since_gain, p.window, p.max_iter)), Some((6, 6, 200)));
    }

    /// The stop and the observer change no converged number: H₂ with the stall stop off, at its
    /// default W and observed gives the same bits, and the last iteration's Harris–Foulkes energy
    /// lands on the reported energy (second order in Δρ < tol).
    #[test]
    fn the_stall_stop_and_the_observer_leave_a_converged_solve_bitwise() {
        let g = Grid::cube(25, 0.3);
        let sys = System::new(vec![Nucleus { pos: [0.7, 0.0, 0.0], z: 1 }, Nucleus { pos: [-0.7, 0.0, 0.0], z: 1 }], 0, None);
        let on = ScfSettings::for_grid(&g);
        let off = ScfSettings { stall_window: 0, ..on };
        let (_, a) = scf(&g, &sys, &on, None).unwrap();
        let (_, b) = scf(&g, &sys, &off, None).unwrap();
        let mut seen = Vec::new();
        let (_, c) = scf_observed(&g, &sys, &on, None, Some(&mut |p: &ScfProgress| seen.push(*p))).unwrap();
        for (name, x) in [("off", &b), ("observed", &c)] {
            assert_eq!(a.energy.total.to_bits(), x.energy.total.to_bits(), "{name}");
            assert_eq!(a.delta_rho.to_bits(), x.delta_rho.to_bits(), "{name}");
            assert_eq!(a.iterations, x.iterations, "{name}");
        }
        assert_eq!(seen.len(), a.iterations);
        let last = seen.last().unwrap();
        println!("H₂ 25³: {} iterations, E {:+.10}, E_HF(last) {:+.10} (Δ {:.2e}); Δρ {:?}", a.iterations, a.energy.total, last.energy,
                 last.energy - a.energy.total, seen.iter().map(|p| format!("{:.1e}", p.delta_rho)).collect::<Vec<_>>());
        assert_eq!(last.delta_rho.to_bits(), a.delta_rho.to_bits());
        assert!((last.energy - a.energy.total).abs() < 1e-5, "Harris–Foulkes {} vs E {}", last.energy, a.energy.total);
    }

    /// Review S3. The bitwise and Harris–Foulkes claims on the pseudo path — pseudo-O M = 3 (`CoresOnly`,
    /// `DegenerateShells`: NLCC, E_NL, the shared minority p with f = 1/3 in the band sum), 33³, h = σ = 0.25: the
    /// stall stop off, at its default W, and observed give the same bits, and the last Harris–Foulkes energy lands on
    /// the reported one.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
    fn the_stall_stop_and_the_observer_leave_a_pseudo_solve_bitwise() {
        use crate::ionic::{Ionic, IonicPolicy, IonicSetup};
        use crate::radial::test_cache::filtered_at;
        let h = 0.25;
        let g = Grid::cube(33, h);
        let setup = std::sync::Arc::new(IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, std::collections::BTreeMap::from([(8, Ionic::Pseudo(filtered_at(8, h)))])));
        let sys = System::with_ionic(vec![Nucleus { pos: [0.07, -0.03, 0.02], z: 8 }], 0, Some(3), setup);
        let mut on = ScfSettings::for_grid(&g);
        on.occupations = Occupations::DegenerateShells { tau: DEGENERACY_TAU };
        let off = ScfSettings { stall_window: 0, ..on };
        let (_, a) = scf(&g, &sys, &on, None).unwrap();
        let (_, b) = scf(&g, &sys, &off, None).unwrap();
        let mut seen = Vec::new();
        let (_, c) = scf_observed(&g, &sys, &on, None, Some(&mut |p: &ScfProgress| seen.push(*p))).unwrap();
        for (name, x) in [("off", &b), ("observed", &c)] {
            assert_eq!(a.energy.total.to_bits(), x.energy.total.to_bits(), "{name}");
            assert_eq!(a.delta_rho.to_bits(), x.delta_rho.to_bits(), "{name}");
            assert_eq!(a.iterations, x.iterations, "{name}");
        }
        assert_eq!(seen.len(), a.iterations);
        let last = seen.last().unwrap();
        println!("pseudo-O M=3 33³ (CPU): {} iterations, E {:+.10}, E_HF(last) {:+.10} (Δ {:.2e}); shared {:?}; released {:?}; longest no-gain run {}",
                 a.iterations, a.energy.total, last.energy, last.energy - a.energy.total, a.fractional, a.released, seen.iter().map(|p| p.since_gain).max().unwrap_or(0));
        assert!(!a.fractional.is_empty(), "the minority p is shared: the fractional band sum is exercised");
        assert!((last.energy - a.energy.total).abs() < 1e-5, "Harris–Foulkes {} vs E {}", last.energy, a.energy.total);
    }

    // ---- Fermi–Dirac occupations (spec 2026-11-13 §4–5; plan Task 5). All-electron, σ = h (ruling R3). ----

    fn fd(kt: f64) -> Occupations { Occupations::FermiDirac { kt } }

    /// H₂ along x at bond length `r`, centred.
    fn h2_sys(r: f64) -> System {
        System::new(vec![Nucleus { pos: [0.5 * r, 0.0, 0.0], z: 1 }, Nucleus { pos: [-0.5 * r, 0.0, 0.0], z: 1 }], 0, None)
    }

    /// H₂O at `validate`'s `gpu_vs_cpu` geometry (the `h2o` preset): O at the origin, H at (±1.43, 1.11, 0).
    fn h2o_sys() -> System {
        System::new(vec![Nucleus { pos: [0.0; 3], z: 8 }, Nucleus { pos: [1.43, 1.11, 0.0], z: 1 }, Nucleus { pos: [-1.43, 1.11, 0.0], z: 1 }], 0, None)
    }

    /// The frontier gap of a state that stores levels above N_σ (a Fermi–Dirac state): the lowest level of rank > N_σ
    /// minus the highest of rank ≤ N_σ, over both channels.
    fn frontier_gap(st: &KsState, sys: &System) -> f64 {
        let (mut homo, mut lumo) = (f64::NEG_INFINITY, f64::INFINITY);
        for (e, n) in [(&st.evals_up, sys.n_up), (&st.evals_dn, sys.n_dn)] {
            let o = ascending(e);
            if n > 0 { homo = homo.max(e[o[n - 1]]); }
            if let Some(&i) = o.get(n) { lumo = lumo.min(e[i]); }
        }
        lumo - homo
    }

    /// Spec §5 invariant: `total == internal − ts` bitwise, and the five components sum to `internal` (one FD solve,
    /// H₂ R = 1.4, 33³ × 0.30).
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
    fn fd_energy_invariant() {
        let g = Grid::cube(33, 0.30);
        let s = ScfSettings { occupations: fd(ELECTRON_KT), ..ScfSettings::for_grid(&g) };
        let (_, st) = scf(&g, &h2_sys(1.4), &s, None).unwrap();
        let e = st.energy;
        eprintln!("H₂ FD 33³: {e:?}, S = {:e}, μ = {:?}, {} steps", st.entropy, st.mu, st.iterations);
        assert_eq!(e.total.to_bits(), (e.internal - e.ts).to_bits(), "{e:?}");
        let parts = neumaier([e.kinetic, e.external, e.hartree, e.xc, e.nuclear].into_iter());
        assert!((parts - e.internal).abs() <= 1e-12 * e.internal.abs(), "Σ parts {parts} vs internal {}", e.internal);
        assert_eq!(e.ts.to_bits(), (ELECTRON_KT * st.entropy).to_bits(), "ts = kT·S");
        assert!(e.ts >= 0.0 && st.entropy >= 0.0);
        assert_eq!(st.tol_used, s.tol);
    }

    /// [SHOULD-6] The H₄ square (a = 1.8, axes-aligned and centred: the x ↔ y swap is exact on the grid, so the e pair is
    /// degenerate to the eigensolver): each spin holds 1 electron in the pair, f = ½ each, so TS = 4 kT ln 2 exactly
    /// (spec §4.3 table), and E is the equal-share ensemble's, i.e. `DegenerateShells`' (spec §4.7(b)). The last
    /// iteration's Harris–Foulkes energy is F_HF and lands on F (it would miss by TS = 1.4e-2 without −kT·S).
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
    fn fd_h4_square_closed_form() {
        let g = Grid::cube(41, 0.30);
        let c = 0.5 * 1.8;
        let sys = System::new([[c, c], [c, -c], [-c, c], [-c, -c]].iter().map(|&[x, y]| Nucleus { pos: [x, y, 0.0], z: 1 }).collect(), 0, None);
        let mut s = ScfSettings::for_grid(&g);
        (s.tol, s.eig_tol, s.occupations) = (1e-9, 1e-9, fd(ELECTRON_KT));
        let mut seen = Vec::new();
        let (st, f) = scf_observed(&g, &sys, &s, None, Some(&mut |p: &ScfProgress| seen.push(*p))).unwrap();
        let ds = ScfSettings { occupations: Occupations::DegenerateShells { tau: DEGENERACY_TAU }, ..s };
        let (_, d) = scf(&g, &sys, &ds, None).unwrap();
        let e_ds = d.energy.total;
        let closed = 4.0 * ELECTRON_KT * std::f64::consts::LN_2;
        let last = seen.last().unwrap();
        eprintln!("H₄ square 41³ × 0.30: E_DS = {e_ds:.12} (Task 10's literal; {} steps, shared {:?})", d.iterations, d.fractional);
        eprintln!("  FD: F {:.12}, E {:.12} (E − E_DS {:+.2e}), TS {:.12e} (− 4kT ln 2 {:+.2e}), μ {:?}, partial {:?}, {} steps, \
                   F_HF(last) − F {:+.2e}, stored ↑ {} ↓ {}",
                  f.energy.total, f.energy.internal, f.energy.internal - e_ds, f.energy.ts, f.energy.ts - closed, f.mu, f.partial,
                  f.iterations, last.energy - f.energy.total, st.orbitals_up.len(), st.orbitals_dn.len());
        assert_eq!(f.energy.total.to_bits(), (f.energy.internal - f.energy.ts).to_bits());
        assert!((f.energy.ts - closed).abs() <= 1e-8, "TS {} vs 4 kT ln 2 = {closed}", f.energy.ts);
        assert!((f.energy.internal - e_ds).abs() <= 1e-8, "E {} vs E_DS {e_ds}", f.energy.internal);
        assert_eq!(f.partial.len(), 4, "the e pair in each spin: {:?}", f.partial);
        assert!(f.partial.iter().all(|p| (p.f - 0.5).abs() < 1e-6), "{:?}", f.partial);
        assert!(f.mu[0].is_some() && f.mu[1].is_some());
        assert!(f.fractional.is_empty() && f.released.is_empty(), "no DegenerateShells bookkeeping under FD");
        assert!((last.energy - f.energy.total).abs() < 1e-6, "F_HF {} vs F {}", last.energy, f.energy.total);
    }

    /// Spec §5 (V1): on gapped systems Fermi–Dirac is aufbau's kT → 0 limit. At an explicit kT = 1e-3 (spec ruling 46
    /// [kT-5]; the 1e-8 bound holds for any gap ≥ 0.05 Ha there) |F − E_Aufbau| ≤ 1e-8 and TS ≤ 1e-12, H₂ (R = 1.4) and
    /// H₂O, 41³ × 0.30, tol = eig_tol = 1e-9. The production kT's numbers are `fd_at_production_kt_on_gapped_measurement`.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
    fn fd_matches_aufbau_on_gapped() {
        let g = Grid::cube(41, 0.30);
        let mut a = ScfSettings::for_grid(&g);
        (a.tol, a.eig_tol, a.occupations) = (1e-9, 1e-9, Occupations::Aufbau);
        for (name, sys) in [("H₂ R=1.4", h2_sys(1.4)), ("H₂O", h2o_sys())] {
            let (_, ea) = scf(&g, &sys, &a, None).unwrap();
            let (_, e1) = scf(&g, &sys, &ScfSettings { occupations: fd(1e-3), ..a }, None).unwrap();
            eprintln!("{name} 41³ × 0.30: E_Aufbau {:.12} ({} steps); kT 1e-3: F − E_Aufbau {:+.2e}, TS {:.2e} ({} steps)",
                      ea.energy.total, ea.iterations, e1.energy.total - ea.energy.total, e1.energy.ts, e1.iterations);
            assert!((e1.energy.total - ea.energy.total).abs() <= 1e-8, "{name}: F {} vs E_Aufbau {}", e1.energy.total, ea.energy.total);
            assert!(e1.energy.ts <= 1e-12, "{name}: TS {:e}", e1.energy.ts);
            assert!(ea.energy.ts == 0.0 && ea.entropy == 0.0 && ea.mu == [None, None] && ea.partial.is_empty());
        }
    }

    /// The production-kT companion of `fd_matches_aufbau_on_gapped` (print only, no gate; Task 5 review ruling: moved out
    /// of the qm suite's gate for time): the same systems and settings at kT = `ELECTRON_KT`, printing the frontier gap,
    /// TS, F − E_Aufbau, the partial-level count and the stored block. Task 5 measured H₂ gap 0.361915 Ha, TS 1.45e-16,
    /// F − E_A −5.0e-10; H₂O gap 0.290994 Ha, TS 1.39e-13, F − E_A +5.77e-9.
    #[test]
    #[ignore = "measurement: numbers recorded in references.md"]
    fn fd_at_production_kt_on_gapped_measurement() {
        let g = Grid::cube(41, 0.30);
        let mut a = ScfSettings::for_grid(&g);
        (a.tol, a.eig_tol, a.occupations) = (1e-9, 1e-9, Occupations::Aufbau);
        for (name, sys) in [("H₂ R=1.4", h2_sys(1.4)), ("H₂O", h2o_sys())] {
            let (_, ea) = scf(&g, &sys, &a, None).unwrap();
            let (st5, e5) = scf(&g, &sys, &ScfSettings { occupations: fd(ELECTRON_KT), ..a }, None).unwrap();
            eprintln!("{name} 41³ × 0.30: E_Aufbau {:.12} ({} steps); kT {ELECTRON_KT}: frontier gap {:.6} Ha, TS {:.2e}, \
                       F − E_Aufbau {:+.2e}, partial {}, stored ↑ {} ↓ {} ({} steps)",
                      ea.energy.total, ea.iterations, frontier_gap(&st5, &sys), e5.energy.ts, e5.energy.total - ea.energy.total,
                      e5.partial.len(), st5.orbitals_up.len(), st5.orbitals_dn.len(), e5.iterations);
        }
    }

    /// Spec §4.2: an all-electron FD solve grows witnesses — at least N_σ + 1 stored orbitals per occupied channel, and the
    /// highest one satisfies both witness arms at the reported μ.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
    fn fd_all_electron_grows_witnesses() {
        let g = Grid::cube(33, 0.30);
        let sys = h2_sys(1.4);
        let (st, stats) = scf(&g, &sys, &ScfSettings { occupations: fd(ELECTRON_KT), ..ScfSettings::for_grid(&g) }, None).unwrap();
        eprintln!("H₂ FD 33³: stored ↑ {} ↓ {}, ε↑ {:?}, ε↓ {:?}, μ {:?}", st.orbitals_up.len(), st.orbitals_dn.len(), st.evals_up, st.evals_dn, stats.mu);
        for (spin, orbs, evals, occ, n) in [(0, &st.orbitals_up, &st.evals_up, &st.occ_up, sys.n_up), (1, &st.orbitals_dn, &st.evals_dn, &st.occ_dn, sys.n_dn)] {
            assert!(n > 0 && orbs.len() >= n + 1 && evals.len() == orbs.len() && occ.len() == orbs.len(), "spin {spin}");
            let mu = stats.mu[spin].expect("an occupied channel has μ");
            assert!(witness_reached(evals, n, mu, ELECTRON_KT), "spin {spin}: ε {evals:?}, μ {mu}");
            assert!((neumaier(occ.iter().copied()) - n as f64).abs() < 1e-12, "spin {spin}: Σf {occ:?}");
        }
        assert_eq!(st.occupations, fd(ELECTRON_KT));
    }

    /// Spec §4.1: an empty channel is not solved and has no μ. The H atom: μ↑ is reported, μ↓ is None, no level is
    /// partial (1s against the box states), TS ≤ 1e-12.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
    fn fd_empty_channel_has_no_mu() {
        let g = Grid::cube(33, 0.30);
        let sys = hydrogen([0.0; 3]);
        let (st, stats) = scf(&g, &sys, &ScfSettings { occupations: fd(ELECTRON_KT), ..ScfSettings::for_grid(&g) }, None).unwrap();
        eprintln!("H FD 33³: F {:.10}, TS {:.2e}, μ {:?}, ε↑ {:?}, gap {:.4}", stats.energy.total, stats.energy.ts, stats.mu, st.evals_up, frontier_gap(&st, &sys));
        assert!(stats.mu[0].is_some() && stats.mu[1].is_none(), "{:?}", stats.mu);
        assert!(stats.partial.is_empty(), "{:?}", stats.partial);
        assert!(stats.energy.ts <= 1e-12, "TS {:e}", stats.energy.ts);
        assert!(st.orbitals_dn.is_empty() && st.occ_dn.is_empty() && g.integrate(&st.rho_dn) == 0.0);
    }

    /// Review Focus 1 / spec §5 "Warm start": `KsState.occupations` carries kT and the guard compares it with `==`, so a
    /// state converged at another kT, under `DegenerateShells` or under `Aufbau`, never seeds an FD solve — the result is
    /// the cold solve's bits. The other-kT and `DegenerateShells` states fit FD's counts (N_σ in Σf, a witness above), so
    /// they reach the `==` guard itself (Task 5 review); the `Aufbau` one (exactly N_σ orbitals) is refused by
    /// `counts_match` first.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
    fn fd_warm_state_from_other_scheme_is_cold() {
        let g = Grid::cube(33, 0.30);
        let sys = h2_sys(1.4);
        let base = ScfSettings::for_grid(&g);
        let target = ScfSettings { occupations: fd(ELECTRON_KT), ..base };
        let (_, cold) = scf(&g, &sys, &target, None).unwrap();
        let (other_kt, _) = scf(&g, &sys, &ScfSettings { occupations: fd(2e-3), ..base }, None).unwrap();
        let (shells, _) = scf(&g, &sys, &ScfSettings { occupations: Occupations::DegenerateShells { tau: DEGENERACY_TAU }, ..base }, None).unwrap();
        let (aufbau, _) = scf(&g, &sys, &ScfSettings { occupations: Occupations::Aufbau, ..base }, None).unwrap();
        assert!(counts_match(&other_kt, &sys, target.occupations) && counts_match(&shells, &sys, target.occupations),
                "the other-kT and DegenerateShells legs must reach the == guard");
        assert!(!counts_match(&aufbau, &sys, target.occupations), "the Aufbau leg is refused by counts_match");
        for (name, w) in [("FD kT 2e-3", &other_kt), ("DegenerateShells", &shells), ("Aufbau", &aufbau)] {
            let (_, warm) = scf(&g, &sys, &target, Some(w)).unwrap();
            assert_eq!(warm.energy.total.to_bits(), cold.energy.total.to_bits(), "{name}");
            assert_eq!(warm.energy.ts.to_bits(), cold.energy.ts.to_bits(), "{name}");
            assert_eq!(warm.delta_rho.to_bits(), cold.delta_rho.to_bits(), "{name}");
            assert_eq!(warm.iterations, cold.iterations, "{name}");
        }
        // the guard is not a blanket refusal: the state's own scheme is reused
        let (own, _) = scf(&g, &sys, &target, None).unwrap();
        let (_, rewarm) = scf(&g, &sys, &target, Some(&own)).unwrap();
        assert!(rewarm.iterations < cold.iterations, "same scheme: warm {} vs cold {}", rewarm.iterations, cold.iterations);
    }
}
