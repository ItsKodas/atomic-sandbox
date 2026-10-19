//! Orbital display numerics in f64 (orbital-viewer spec §5,
//! `docs/superpowers/specs/2026-10-15-orbital-viewer-design.md`; plan Task 1). Display only:
//! nothing here feeds back into the SCF. Atomic units throughout (Hartree, Bohr); orbitals are
//! real and L²-normalised on the grid, Σ ψ² h³ = 1 (`Grid::dot`). Every grid reduction is a
//! Neumaier-compensated f64 sum (rule 4). Per-element input: none — Z enters only through the
//! dimensionless Zσ of the smeared-1s tag (rule 1).
//!
//! Inputs are the converged Kohn–Sham orbitals of each spin channel in ascending energy order,
//! occupied first (aufbau), followed by any unoccupied ones from the display eigensolve. The
//! GPU path (`qm_gpu::orbitals`) reads back only the scalars named in [`UpdateInput`]; the
//! functions [`moments`], [`radial_profile`], [`first_sign_functionals`], [`outer_weight`],
//! [`spin_summary`] and [`colour_volume`] are the CPU references of its kernels.
//!
//! # 1. Degenerate clusters by energy (spec §5.1; molecules, unoccupied rows, tags)
//!
//! In each channel the sorted eigenvalues are joined into a cluster while consecutive gaps are
//! ≤ τ (single linkage, [`clusters`]), with
//!
//!   τ = max(`TAU_PER_TOL` · tol_scf · 1 Ha/e, `TAU_MIN`) = 1 mHa at tol 1e-5, 1e-4 Ha at 1e-6.
//!
//! *Why τ is tied to tol.* At SCF tolerance tol (∫|ρ_out − ρ_in| in electrons) each member of a
//! degenerate set sees a potential inconsistent by δV ≈ δV_H + f_xc δρ; first-order perturbation
//! theory shifts its eigenvalue by ⟨ψ|δV|ψ⟩, O(tol) Ha per electron of Δρ when the density error
//! sits on the atom. The factor 100 covers the anisotropic part of δV (which splits the set).
//!
//! Clusters never cross a change of occupation: rotating an occupied orbital into an empty one
//! would change the displayed density (the colour view's Σ fᵢψ̃ᵢ² must stay ρ). Occupations are
//! compared exactly; they are integers today (aufbau), and equal within a degenerate set under
//! any future smearing rule that depends on ε only.
//!
//! *Spread and residual.* Single linkage lets a cluster of d members spread s = max ε − min ε up
//! to (d − 1)τ, not τ, so the tag shows the measured s. For a displayed combination
//! ψ = Σⱼ cⱼφⱼ (Σcⱼ² = 1) of eigenvectors Hφⱼ = εⱼφⱼ, the Rayleigh quotient is ε̄ = Σcⱼ²εⱼ and
//!
//!   ‖(H − ε̄)ψ‖² = Σⱼ cⱼ²(εⱼ − ε̄)²,
//!
//! the variance of a distribution supported in an interval of width s; by Popoviciu's
//! inequality it is ≤ s²/4, so **the residual is ≤ s/2**: each displayed member of a rotated set
//! is an eigenfunction to within s/2. The row's energy is that Rayleigh quotient ε̄ (exact for
//! unrotated rows, where it is the eigenvalue).
//!
//! # 2. Why a lone atom's shells are *not* found by τ (review MUST 1)
//!
//! The finite-difference Laplacian's symbol Σ_a s(k_a h) is invariant under the cube group O_h,
//! and the l = 1 harmonics span an irreducible 3-dimensional representation of O_h, so the stencil
//! alone cannot split a p shell (it does split d into e_g + t_2g, and f). What splits p is the
//! **position-dependent sampling of the σ = h smeared nucleus** (the egg-box): off a
//! cube-symmetric site the sampled potential is not O_h-invariant about the nucleus. Task 0
//! measured the self-consistent LSDA 2p spread
//! (`docs/superpowers/notes/2026-10-15-orbital-viewer-task0-spike.md`): ≤ 2.2e-7 Ha on a grid
//! point or cube centre (f32/SCF noise, as the O_h argument says), but 4.97e-4 Ha off-grid and
//! **7.05e-4 Ha at (h/2, 0, 0)** for Ne at h = 0.25 (1.17e-4 at h = 0.15), i.e. up to
//! 7 × `TAU_MIN`. A moving lone atom would cross τ and its labels would flicker, so a lone atom's
//! shells are grouped by **angular character and occupation** (§4 below), and the measured spread
//! is reported ("split by the grid by s mHa"). No per-Z τ exists (rule 1).
//!
//! # 3. Moments and angular character (spec §5.2)
//!
//! For each orbital and l ≤ 3 (16 pairs, index l² + l + m), both **linear in ψ**:
//! - b_lm = ⟨ψ|S_lm(r − R)⟩, with S_lm the real regular solid harmonics of `qm::poisson`
//!   (Racah normalisation, reused unchanged). They are homogeneous polynomials, exactly covariant
//!   under rotations about R: a rotation within a set acts on b_l· by the real Wigner matrix.
//!   Used for the Procrustes alignment.
//! - c_lm = ⟨ψ|Y_lm(r̂)⟩, Y_lm = √((2l+1)/4π)·S_lm(r)/r^l the unit real spherical harmonics
//!   (Y_00 = 1/√(4π) at r = 0, Y_lm = 0 there for l > 0). Used for angular character, without
//!   the r^l weight that would let the diffuse tail dominate.
//!
//! W_{l,i} = Σ_m c²_{lm,i} / Σ_{l'≤3,m} c²_{l'm,i} ([`character`]); for a set C,
//! W_l(C) = Σ_{i∈C}Σ_m c²_{lm,i} / Σ_{i∈C}Σ_{l',m} c²_{l'm,i} ([`set_character`]).
//! *Invariance:* a mixing ψ'_k = Σᵢ ψᵢQ_ik with Q orthogonal maps the row (lm) of the matrix
//! c_{lm,i} to c'_{lm,·} = c_{lm,·}Q, whose squared norm Σ_k c'²_{lm,k} = c Q Qᵀ cᵀ = Σᵢ c²_{lm,i}
//! is unchanged; so numerator and denominator of W_l(C) are each invariant, exactly (to
//! rounding), and W_l(C) = 1 for an ideal shell. (The draft's |ψ|-weighted moment was nonlinear
//! and not invariant: review SHOULD 5.)
//!
//! # 4. Lone atom: shells, alignment, labels (spec §5.3; controller rulings 1, 2, 4)
//!
//! A lone atom is a scene with exactly one nucleus (`UpdateInput::moments` is `Some`). Only its
//! **occupied** orbitals are classified; unoccupied rows are always `#k` (review MUST 3).
//!
//! 1. *Character.* Orbital i has dominant l when W_{l,i} ≥ `L_CHAR_MIN` (unique, as 0.9 > ½).
//!    Task 0: every real 2p had W_1 = 1.0000, 2s W_0 = 1.0000, the worst 1s W_0 = 0.9973.
//! 2. *Shells by character, not by gap.* Walk the occupied orbitals in energy order; consecutive
//!    orbitals with the same dominant l form a run; a run of length k(2l+1) is cut, in order, into
//!    k shells of 2l+1 ([`lone_atom_shells`]). Each shell must pass W_l(shell) ≥ `L_CHAR_MIN` and
//!    σ_min/σ_max ≥ `ALIGN_RANK_MIN` for M_aj = b_{l,m_a}(ψ_j) (the shell spans every m: for an
//!    ideal shell M = c × orthogonal and the ratio is 1). Its spread is recorded, however large.
//!    Runs need not be ordered by l: with the smeared nucleus the runs of Ne are s(1), p(3), s(1).
//!    A level of another l falling *inside* a grid-split shell would break the run: the atom is
//!    then `#k` for that geometry (it needs an accidental degeneracy within the ≲ 1 mHa split).
//! 3. *Open shells, from occupation, per atom* (review SHOULD 7). The atom is labelled only if in
//!    **both** channels every occupied orbital has a dominant l, every run is a multiple of 2l+1
//!    and every shell passes. An incomplete occupied l-set in either channel makes the density,
//!    hence the KS potential, non-central for *both* channels (C's ↓ "2s" sits in the anisotropic
//!    field of the ↑ p²): then the whole atom is `#k`, unaligned, on continuity (§5).
//! 4. *Alignment* (orthogonal Procrustes, [`alignment`]). ψ'_k = Σⱼψⱼ U_jk with k ↔ the k-th
//!    harmonic in display order (p: x, y, z; d, f: m = −l…l). Maximise Σ_k⟨ψ'_k|S_{m_k}⟩ = tr(MU)
//!    over orthogonal U. With M = WΣVᵀ, tr(MU) = tr(Σ·VᵀUW) = Σ_k σ_k Z_kk for the orthogonal
//!    Z = VᵀUW, |Z_kk| ≤ 1, maximal at Z = I: **U = VWᵀ**. Then MU = WΣWᵀ is symmetric positive
//!    semidefinite, its diagonal is ≥ 0, so **the sign rule is built in** (p_x's positive lobe on
//!    +x). For a nonsingular M the maximiser is unique (the polar factor). Check (review MUST 2):
//!    φⱼ = Σ_a R_ja p_a gives M = cRᵀ, W = Rᵀ, V = I, U = R, and ψ'_k = Σ_a (RᵀR)_ak p_a = p_k;
//!    hence RᵀU = I. The rotation is recomputed every update: stateless, deterministic, no drift.
//! 5. *Labels: angular character and radial node count, never energy order* (ruling 1).
//!    l comes from item 1. The radial quantum number is the **node count** ν of the orbital's
//!    radial function, counted on radial shells about the nucleus ([`radial_profile`],
//!    [`profile_nodes`]); n = l + 1 + N_core,l + ν (N_core,l = 0 until M7). The Sturm–Liouville
//!    oscillation theorem (the k-th eigenstate of fixed l in a local central potential has k − 1
//!    radial nodes) is used as a **consistency check**, not as the source: within each channel
//!    and l, the shells sorted by energy must read ν = 0, 1, 2, …, and every member of a shell the
//!    same ν; otherwise the atom is `#k` with reason [`Unlabelled::NodesUnresolved`]. m is named
//!    from the harmonic the member was aligned to.
//!
//! *Node count on radial shells.* Bins of width Δ = `RADIAL_BIN_H`·h about R, out to the radius of
//! the largest sphere about R inside the grid. Per bin k and (l, m):
//! a_k,lm = Σ_{p∈k} ψ(r_p)Y_lm(r̂_p) h³ and n_k = the number of points (so Σ_k a_k = c up to the
//! points outside the sphere). For an orbital of dominant l with direction ĉ = c_l·/|c_l·|, the
//! bin amplitude is R̄_k = Σ_m ĉ_m a_k,lm / n_k. For l = 0 this is R times the bin mean of
//! Y_00 = const, so **its sign is exactly the sign of R** wherever R keeps its sign across the
//! bin, on any lattice and for any centre. For l ≥ 1 it is exact only on O_h-symmetric sites:
//! for ψ = R(r)Y_ĉ′(r̂) the whole-grid moment is c = Gĉ′ with G = Σ_p R(r_p)Y(r̂_p)Y(r̂_p)ᵀh³,
//! which is ∝ I only on such sites, so ĉ ≠ ĉ′ off them and a sparse bin's Σ_p R Y_ĉ Y_ĉ′ can
//! vanish or flip. Harmless today (the occupied p of H–Ne are nodeless, and the Sturm–Liouville
//! check rejects a single miscount); it matters for 3p/3d under M7. (The magnitude is ≈ R/4π
//! on a well-sampled shell, within a factor 2l + 1 on a sparse one — used only against the
//! floor.) ν = the number of sign changes in the sequence of bins with
//! |R̄_k| ≥ `NODE_FLOOR`·max_k|R̄_k|. A bin straddling a node lies between two bins of opposite
//! sign, so whatever its own sign it adds no spurious change.
//!
//! *Why the smeared nucleus inverts 2s/2p.* The Gaussian nucleus removes the cusp that
//! penetrating s states rely on. One electron: the smeared well differs from −Z/r by
//! ΔV = (Z/r)·erfc(r/a) ≥ 0 (a = σ√2), with ∫ΔV d³r = 2πZσ². To first order only states with
//! ψ(0) ≠ 0 feel it: |ψ_2s(0)|² = Z³/8π gives E_2s − E_2p ≈ Z²(Zσ)²/4 + O(σ⁴), so **for one
//! electron 2p lies below 2s for every σ > 0** (radial solve, Z = 1: 2.42e-5 Ha at Zσ = 0.01 vs
//! 2.50e-5 first order; 1.61e-2 at 0.5; 2.63e-2 at 2.5). In a screened atom the point-nucleus
//! order 2s < 2p comes from core penetration (0.82 Ha for LSDA Ne: 2s ≈ −1.32, 2p ≈ −0.50); the
//! order inverts where the smearing shift of 2s exceeds that splitting. There is no closed-form
//! threshold: Task 0 measured inverted Ne at Zσ = 2.5 and 1.5 and inverted O at Zσ = 2 and 1.2
//! (Ne at h = σ = 0.25: 2p −0.845 Ha, 2s −0.298 Ha; the O singlet fills 1s and 2p, not 2s). Only
//! in the large-Zσ limit is the well harmonic, V ≈ V(0) + ½kd² with k = 4Z/(3√π a³), ordering
//! levels as ω(2ν + l + 3/2); Ne's Zσ = 2.5 is far from it (harmonic: V(0) −31.9, ω 13.05 Ha,
//! 1s −12.3, 2p +0.70, 2s +13.7 Ha; the radial continuum solve gives −16.702, −9.737, −7.106 Ha,
//! and the 41³ grid of `smeared_one_electron_atom_is_labelled_by_nodes` −16.712, −9.743,
//! −7.120). Labels by node count are immune to the inversion; the inversion is reported, as
//! `LoneAtom::Labelled { shell_order_inverted }`: true when, in either channel, a shell of lower l
//! lies above a shell of higher l with the same n (the ruling's s-above-p case; within one n the
//! order of any neutral point-nucleus atom is s < p < d, so the flag means "not physical").
//!
//! *Why the count is feasible.* The smearing also moves the inner s node out. Radial solve of the
//! bare well (2s node in units of σ, by Z-scaling a function of Zσ only): 5.12 (Zσ = 0.5), 3.23
//! (1), **1.98 (2.5, Ne at h = 0.25: 0.494 Bohr)**, 1.46 (5), 1.13 (10), 0.99 (15), 0.87 (22), 0.80
//! (30). With σ = h every centre has a grid point within (√3/2)h; at the worst site (a cube
//! centre) Ne's inner lobe there still has R(0.866h)/R(0) = 0.67, 67 × `NODE_FLOOR`. (The physical
//! point-nucleus Ne 2s node at ≈ 0.2 Bohr < h would not be resolved: smearing is what makes it
//! countable.)
//!
//! *Failure modes of the node count* (each makes the Sturm–Liouville check fail, so the atom
//! falls back to `#k`, never to a wrong name):
//! - a node closer to the nucleus than its nearest grid point, so the inner lobe holds no point:
//!   worst case (√3/2)h = 0.87σ at a cube centre, reached at Zσ ≈ 22 (Z ≈ 88 at h = σ = 0.25),
//!   or σ ≪ h (not today's setting);
//! - a lobe weaker than `NODE_FLOOR` × the largest (the outer lobe of a high-n s state in a
//!   Coulomb-like well: hydrogenic 3s 0.07, 4s 0.04 of R(0); smearing lowers R(0) and helps);
//! - a lobe beyond the inscribed sphere (a state so diffuse that it touches the box: occupied
//!   bound states are not, box states are never labelled);
//! - two nodes within one bin (Δ = h/2; a lobe narrower than the grid cannot be represented);
//! - for l ≥ 1 off O_h sites, a sparse inner bin whose sign is not R's (above);
//! - not a failure mode: grid noise (f32 ≲ 1e-6 relative, four orders under the floor) and the
//!   egg-box's other-l admixture (W_other ≤ 3e-3 measured), which projects out of R̄ on a
//!   well-sampled shell and sits below the floor near nodes, where shells are sparse.
//!
//! # 5. Continuity: tracks, signs, colours (spec §5.4; molecules and whatever §4 did not align)
//!
//! *Why not a purely stateless sign rule.* Any linear functional ⟨ψ|w⟩ vanishes by symmetry on
//! some orbital of opposite parity to w; a nonlinear rule ("largest value positive") flips
//! between symmetry-equivalent lobes (both ends of H₂'s σ_u) from step to step. So the sign is
//! stateless only at first appearance and continuous afterwards (P3).
//!
//! With O_ij = ⟨ψᵢ|ψ̃ⱼ^prev⟩ (current raw vs previously displayed), per channel:
//! 1. *Clusters* of 1 < d ≤ `MAX_ROTATED_CLUSTER` not aligned by §4 are rotated toward the
//!    previous display: P = the d previous rows with the largest Σ_{i∈C}O²ᵢⱼ (sorted by index;
//!    zero columns pad P if fewer exist), A = O_{C,P}. Maximise tr(UᵀA) = Σ_k⟨ψ'_k|ψ̃_{P_k}⟩:
//!    A = WΣVᵀ gives tr(UᵀWΣVᵀ) = tr(Σ·VᵀUᵀW), maximal at VᵀUᵀW = I: **U = WVᵀ** (a reflection is
//!    allowed; it is a sign), and UᵀA = VΣVᵀ has a non-negative diagonal. The rotated overlaps are
//!    O' = UᵀO_{C,·}; nothing is re-read. Larger clusters are runs of box states: continuity only.
//! 2. *Matching*: Hungarian maximum-weight assignment ([`hungarian_max`], O(K³), hand-written) on
//!    |O'| with entries below `MATCH_MIN` set to 0; a pair is matched when |O'| ≥ `MATCH_MIN`.
//!    A matched row inherits the track and its colour slot, and its sign makes O' positive
//!    (aligned lone-atom members keep their built-in sign). An unmatched row starts a new track;
//!    an unmatched previous track retires and frees its slot.
//! 3. *First appearance* (review SHOULD 10, [`first_sign_functionals`]). With X̄ the centroid of
//!    the nuclei and g = e^{−|r−X̄|²}, the nine windowed functionals are the monopole
//!    w₀ = Σ_I e^{−|r−R_I|²}, the dipoles (x_a − X̄_a)g and the quadrupoles S_2m(r − X̄)g; their
//!    normalised values ⟨ψ|w⟩/‖w‖ (‖ψ‖ = 1) are linear in ψ, so a rotated row's values are the
//!    same rotation of the raw ones. The sign makes the first value with magnitude
//!    ≥ `SIGN_MOMENT_MIN` positive. The windows make the values independent of the box once it
//!    contains them (whole-box norms grow as L^{5/2}, L^{7/2}). **If none qualifies the raw
//!    solver sign is kept, and that sign is not deterministic**: it needs an orbital with no
//!    weight within ~1 Bohr of the centroid in any of nine shapes; continuity holds it from then.
//! 4. *Colour slots* (`PALETTE_SLOTS` = 12): a new track takes the lowest free slot, in energy
//!    order, ↑ before ↓; a new ↓ track whose displayed cross overlap with an ↑ row is
//!    ≥ `SHARE_COLOUR_MIN` takes that row's slot (one colour per spatial orbital of a closed
//!    shell). Beyond 12 live tracks the least-used slot repeats.
//! 5. *Resets*: [`Tracker::reset`] (grid change, provider reset, view off) forgets the previous
//!    display; track ids keep counting up, so a stale selection never names a new orbital.
//!
//! # 6. Unoccupied rows and the block edge (spec §4.5, review MUST 3)
//!
//! The display eigensolve computes n_occ + m + `UNOCC_GUARD` vectors, so m = n_computed − n_occ −
//! `UNOCC_GUARD`. The unoccupied spectrum of a cubic box is full of degenerate sets (1, 3, 6, …),
//! and only the cluster containing the last computed vector can have been cut by the block edge.
//! Shown: the unoccupied τ-clusters that start among the first m and do not contain the last
//! computed vector (so a shown set is always whole, and may extend past m). The cluster that
//! starts among the first m but contains the last vector is hidden and its size reported in
//! `hidden_unoccupied`. The occupied labels never depend on m.
//!
//! *Box states* (spec §5.7, review SHOULD 9, [`is_box_state`]): an unoccupied row whose cluster
//! has a member with ε ≥ 0 or a weight > `BOX_STATE_OUTER_WEIGHT` within L/6 of a face
//! ([`outer_weight`]; L = (n + 1)h, the Dirichlet box: ψ = 0 one step outside the grid; each point
//! weighted by the fraction of its cell in that region, so the measure is O(h²)). The lowest
//! box state Πsin²(πx_a/L) puts 1 − (1 − 2(a − sin(2πa)/2π))³ = 0.163 there at a = 1/6, for any L;
//! the threshold is a third of that. A third criterion, the orbital's density on the outermost grid
//! plane ([`box_state`], §8), comes with the display eigensolve.
//!
//! # 7. Scales, colour volume, spin, smeared 1s (spec §5.5, §5.6, §2 item 5)
//!
//! `UpdateOutput::scale` is max|ψ| over the row's set (aligned shell or cluster; the row itself
//! otherwise): a rotated member's own maximum is not known before the rotation; exact for
//! non-degenerate rows, within √d otherwise. [`colour_volume`] writes rgb = Σfᵢcᵢψ̃ᵢ²/Pᵢ,
//! a = Σfᵢψ̃ᵢ²/Pᵢ (Pᵢ = scale² per orbital, or max ρ for physical scaling). [`SpinSummary::scale`]
//! = max(max|m|, `SPIN_FLOOR`·max ρ); a spin density within the solver's noise
//! ([`SpinSummary::is_noise`], ∫|m| ≤ [`SpinSummary::noise_l1`]) is not drawn at all. The `smeared_1s` tag marks the n = 1, l = 0 row when
//! Zσ ≥ `SMEARED_ZSIGMA`: by r → r/Z, E(Z, σ) = Z²E(1, Zσ), and E(1, 0.5) = −0.3751 Ha against −0.5
//! for a point nucleus (25 % shallower; 1-D Numerov, pseudopotential spec §1.1).
//!
//! # 8. Unoccupied orbitals: the display eigensolve (spec §4.5, P1; plan Task 4)
//!
//! The SCF solves only the n_σ occupied states per channel, and enlarging its block would change
//! the solver (rule 6). The unoccupied rows come from a separate, display-only eigensolve on the
//! converged state ([`unoccupied_cpu`]; on the GPU `qm_gpu::orbitals::OrbitalView::unoccupied`)
//! of the Hamiltonian of the converged **output** density, per channel σ (atomic units):
//!
//!   H_σ^out = −½∇² + V_nuc + V_H[ρ_out] + v_xc,σ[ρ_out↑, ρ_out↓].
//!
//! Under an ionic setup (M7, `crate::scf` "Ionic species") it is the SCF's full operator:
//! V_nuc is the state's V_loc, v_xc is of (ρ_out,σ + ½ρ_pc) when an ion has a partial core, and V_NL
//! is added ([`output_potentials`], [`output_nonlocal`]; `PseudoHamiltonian`). Dropping either term
//! moves pseudo-O H₂O's occupied ε by 0.82 Ha (V_NL) or 0.019 Ha (ρ_pc), far beyond τ, so the
//! occupied-agreement guard would reject such a solve (`unoccupied_cpu_uses_the_pseudo_hamiltonian`).
//!
//! `KsState::v_h` is already V_H[ρ_out] (the energy's final Poisson solve), so no Poisson solve is
//! needed; v_xc is re-evaluated from `rho_up`/`rho_dn` (LSDA PW92, `crate::xc`). LOBPCG finds the
//! lowest n_σ + m + `UNOCC_GUARD` eigenpairs from the guess [copies of the SCF's occupied
//! orbitals, then the previous call's unoccupied vectors, then `initial_orbitals` (seed 3 ↑, 4 ↓)
//! for any still missing], with the SCF's preconditioner and its 200-iteration cap. Tolerance:
//! `ScfSettings::eig_tol` on the CPU (f64 has no residual floor); on the GPU the SCF's
//! **effective** tol max(`eig_tol`, 3 × the f32 residual floor) — 5.42e-6 at h = 0.25 (Task 0;
//! the raw 1e-6 sits below the floor and every row would end "not converged", review MUST 4).
//! The state is only borrowed: nothing the next SCF solve starts from is written.
//!
//! *Why the occupied eigenvalues agree with the SCF's (the bound used).* The SCF's last eigensolve
//! used H_σ^in = H_σ[ρ_in], ρ_in the final step's input density, and it stopped with
//! Δρ = ∫(|δρ↑| + |δρ↓|) < tol_scf, δρ_σ = ρ_out,σ − ρ_in,σ. So
//! H_σ^out − H_σ^in = δV_σ = V_H[δρ] + Σ_σ′ f_σσ′ δρ_σ′ + O(δρ²), with f_σσ′ = ∂v_xc,σ/∂ρ_σ′ the LSDA
//! kernel. Both eigenvalues are Rayleigh quotients of stored vectors, accurate to second order
//! in their residuals (Kato–Temple: r²/gap ≲ (1.6e-5)²/0.3 ≈ 1e-9 Ha at 3 × the GPU tol), so to
//! first order in δV
//!
//!   |ε_i^out − ε_i^in| ≈ |⟨ψ_i|δV_σ|ψ_i⟩| ≤ |∫ δρ Φ_i| + Σ_σ′ |∫ ψ_i² f_σσ′ δρ_σ′|,   Φ_i = V_H[ψ_i²],
//!
//! the Hartree term rewritten by the symmetry of the Coulomb kernel. Each term is a supremum
//! times ∫|δρ|:
//! - Hartree: |∫δρ Φ_i| ≤ max Φ_i · Δρ; max Φ_i is the potential of one electron in ψ_i at its
//!   peak, ≈ ⟨1/r⟩_i (a 1s of exponent ζ gives ζ; the smeared nucleus caps it): a few Ha/e here.
//! - Exchange (same spin only): ψ_i² ≤ ρ_σ (one term of ρ_σ = Σ_j ψ_j²) and
//!   v_x,σ = −(6ρ_σ/π)^{1/3} gives ρ_σ |∂v_x,σ/∂ρ_σ| = |v_x,σ|/3, so the term is
//!   ≤ max|v_x,σ|/3 · Δρ.
//! - Correlation (both spins): the PW92 kernel is bounded the same way, ρ|f_c| ≲ |v_c|, an order
//!   of magnitude below exchange in this density range.
//!
//! Hence |ε_i^out − ε_i^in| ≤ C·Δρ < C·tol_scf with C = max Φ_i + max|v_x|/3 + O(|v_c|) ≈ 1–5 Ha/e
//! for the first-row systems on these grids, far below `TAU_PER_TOL` = 100: **the bound used is
//! |Δε_occ| ≤ τ(tol_scf)** (§1; 1e-4 Ha at tol 1e-6). It is a test and a **runtime guard**: a
//! display solve whose lowest n_σ eigenvalues (sorted) differ from the SCF's by more than τ is
//! discarded ([`Unoccupied::rejected`]). Task 0 saw why it is needed: a long GPU display solve
//! returned seven box states having lost σ_g (|Δε| = 0.34 Ha) with every residual ≤ tol, which a
//! residual test alone accepts; in the other 59 of 60 channel solves |Δε_occ| ≤ 5.0e-7 Ha.
//!
//! *Discarded, kept, projected.* The occupied pairs are discarded after that check: the rows show
//! the SCF's own orbitals and eigenvalues, which agree with the readouts. The unoccupied vectors
//! are eigenvectors of H^out, the SCF's occupied ones of H^in; for occupied ψ_j and unoccupied φ,
//! (ε_φ − ε_j)⟨ψ_j|φ⟩ = ⟨(H^out − ε_j)ψ_j|φ⟩ − ⟨ψ_j|(H^out − ε_φ)φ⟩, so |⟨ψ_j|φ⟩| ≤ (‖δV ψ_j‖ + r_j
//! + r_φ)/gap — small (≈ 1e-6) but not zero. They are therefore projected onto the complement of
//! the SCF's occupied orbitals, φ′ = φ − Σ_j ⟨ψ_j|φ⟩ψ_j, so the shown set (SCF occupied ∪
//! unoccupied) is orthonormal; that moves φ by O(⟨ψ|φ⟩) and its Rayleigh quotient by the square.
//! ε and the residual ‖(H^out − ε)φ′‖ are re-measured on the shown vectors. On the CPU φ′ is
//! re-orthonormalised by MGS in energy order; on the GPU its norm changes by Σ_j⟨ψ_j|φ⟩² ~ 1e-12,
//! far below f32's ~1e-7 normalisation, and is left.
//!
//! *Per row.* The block-edge rule is §6 ([`block_edge`], also run here for
//! [`Unoccupied::hidden`]). A row's residual is the largest member residual of its set — the
//! **eigensolve residual of the set**, not of the shown ψ̃ (a rotated row mixes the members; its own
//! residual is then ≤ √d·max r + s/2 ≈ max r + s/2 with s ≤ (d − 1)τ, §1; review M4). The LOBPCG
//! cap is accepted with honest tags (controller ruling, Task 4 review I3): a shown row whose set
//! has a member above the tolerance — the slowest vectors are the last shown ones and the guards,
//! Task 0 — is shown and tagged not converged ([`Row::not_converged`]); every shown row is either
//! at or below its tolerance or so tagged, and the guard vectors are never shown (§6).
//!
//! *Box states, third criterion* (controller ruling, Task 4; review M1): an unoccupied set is also
//! a box state when its density on the outermost grid plane exceeds the caller's `face_limit` — the
//! surrogate's box-affected face density (`providers::surrogate::box_affected_face_density(h)` =
//! 2e-5 e/Bohr³ × (h/0.25)², ≈ 1e-3 Ha of confinement for an H₂-like tail): one electron in that
//! state would make an answer box-affected by the rule the surrogate refuses to train on. The set's
//! value is f_set = max over the outermost-plane points r of Σ_{k∈set} φ_k(r)² ([`face_density_set`];
//! the maximum of the sum, not the sum of the members' maxima), over the same τ-clusters the rows
//! use. By Cauchy–Schwarz it is the supremum, over every normalised state of the level, of that
//! state's face density (|Σ c_k φ_k(r)|² ≤ Σ φ_k(r)² for Σ c_k² = 1, attained at c ∝ φ(r)); so it is
//! invariant under any rotation within the set — the display rotation included, so taking it on the
//! kept vectors is taking it on the shown ψ̃ — and ≥ every member's own value (it never clears a flag
//! a per-member rule would set). (Task 4 first used per-member maxima, which depend on the basis the
//! eigensolver returned: Li 2p↑ members read 1.47–2.04e-5 against the 2e-5 limit.) ([`box_state`]:
//! ε ≥ 0, or the outer L/6 weight above 0.05, or that. The outer weight is still per member; its
//! invariant form, λ_max of W_kl = ∫_outer φ_kφ_l, is left for a later task.)
//!
//! *Physics (rule 7).* These are eigenstates of the ground-state potential, not excited states;
//! their ε are not excitation energies (ε_a − ε_i is only the zeroth-order, uncoupled estimate of a
//! neutral excitation). LSDA, a continuous functional, has no derivative discontinuity; its KS gap
//! typically underestimates IP − EA by a lot. LSDA's potential decays faster than −1/r (the exact
//! KS potential has a −1/r tail and a Rydberg series of bound virtuals; LSDA has few or none): H₂
//! has no unoccupied state below zero in boxes up to 24 Bohr, and H₂O's LSDA virtual at ≈ −0.002 Ha
//! (24 Bohr box, decay length ≈ 16 Bohr) is box-lifted above zero in the 12 Bohr box, not absent.
//! Most rows are therefore states of the box, flagged as such, whose energies are the box's.
//!
//! *Cancellation* (Task 4 review I2): the eigensolve is display work and yields to the user. It
//! polls `should_abort` before each channel and once per LOBPCG iteration (`eigen::lobpcg`'s
//! callback; the SCF passes `None`); an abort returns `None` and changes nothing — the caller keeps
//! its previous vectors (stamped not current).
//!
//! *Cost.* 4–50 × a warm SCF on the GPU (Task 0), so the provider's budget runs it only when paused
//! in practice (`providers::orbitals`). The GPU's `LobpcgWork` grows to n_σ + m + `UNOCC_GUARD`
//! vectors (+8 fields per vector over the SCF's own n_σ).

use crate::eigen::{jacobi_precond, lobpcg, LocalHamiltonian, Operator};
use crate::grid::Grid;
use crate::nonlocal::{NonlocalOp, PseudoHamiltonian};
use crate::nuclear::{has_partial_core, partial_core_field, Nucleus};
use crate::scf::{initial_orbitals, xc_with_core, KsState, ScfSettings};
use crate::poisson::{solid_harmonics, N_MOMENTS};
use nalgebra::DMatrix;
use std::f64::consts::PI;
use std::ops::Range;

/// τ per unit SCF tolerance (Ha per electron of ∫|Δρ|), §1.
pub const TAU_PER_TOL: f64 = 100.0;
/// Floor of τ (Ha), §1.
pub const TAU_MIN: f64 = 1e-4;
/// Largest cluster rotated by continuity Procrustes (an f shell); beyond: box states.
pub const MAX_ROTATED_CLUSTER: usize = 7;
/// Dominant angular character: W_l ≥ this (per orbital and per shell).
pub const L_CHAR_MIN: f64 = 0.9;
/// A shell's b-moment matrix must have σ_min/σ_max ≥ this (it spans every m).
pub const ALIGN_RANK_MIN: f64 = 0.5;
/// Below this |overlap| an orbital is a different one.
pub const MATCH_MIN: f64 = 0.5;
/// ↑/↓ rows with |displayed cross overlap| ≥ this share a colour slot.
pub const SHARE_COLOUR_MIN: f64 = 0.9;
/// A windowed first-sign functional decides when its normalised magnitude is ≥ this.
pub const SIGN_MOMENT_MIN: f64 = 1e-3;
/// Spin-density display floor, as a fraction of max ρ (P4).
pub const SPIN_FLOOR: f64 = 1e-3;
/// Box state: weight within L/6 of any face above this (a third of the lowest box state's 0.163).
pub const BOX_STATE_OUTER_WEIGHT: f64 = 0.05;
/// The smeared-1s tag: Zσ ≥ this (E(1, 0.5) = −0.3751 Ha vs −0.5).
pub const SMEARED_ZSIGMA: f64 = 0.5;
/// Guard vectors of the display eigensolve (block-edge rule, §6).
pub const UNOCC_GUARD: usize = 2;
/// Colour slots of the qualitative palette.
pub const PALETTE_SLOTS: u8 = 12;
/// Radial-shell width for the node count, in units of h (§4; controller ruling 1).
pub const RADIAL_BIN_H: f64 = 0.5;
/// A radial bin counts toward the node count when |R̄_k| ≥ this × the orbital's largest |R̄| (§4).
pub const NODE_FLOOR: f64 = 1e-2;
/// Number of (l, m) pairs with l ≤ 3.
pub const N_LM: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Spin { Up, Down }

impl Spin {
    /// 0 for ↑, 1 for ↓ (the channel index of every `[_; 2]` here).
    pub fn index(self) -> usize { match self { Spin::Up => 0, Spin::Down => 1 } }
}

// ───────────────────────────── §1 clusters ─────────────────────────────

/// τ = max(`TAU_PER_TOL`·tol, `TAU_MIN`) (Ha). The floor branch is decided in tol space
/// (tol ≤ `TAU_MIN`/`TAU_PER_TOL`, exact in f64 at 1e-6) so that tol = 1e-6 yields `TAU_MIN`
/// itself and not the product 100 × 1e-6, which is one ulp above it.
pub fn tau(tol_scf: f64) -> f64 {
    if tol_scf <= TAU_MIN / TAU_PER_TOL { TAU_MIN } else { (TAU_PER_TOL * tol_scf).max(TAU_MIN) }
}

/// Single-linkage clusters of sorted `evals`: consecutive gaps ≤ `tau` join.
pub fn clusters(evals: &[f64], tau: f64) -> Vec<Range<usize>> {
    debug_assert!(evals.windows(2).all(|w| w[0] <= w[1]), "evals must be sorted ascending");
    let mut out = Vec::new();
    let mut start = 0;
    for i in 1..=evals.len() {
        if i == evals.len() || evals[i] - evals[i - 1] > tau {
            out.push(start..i);
            start = i;
        }
    }
    out
}

/// max − min of a set of eigenvalues (0 for fewer than two).
pub fn spread(evals: &[f64]) -> f64 {
    if evals.is_empty() { return 0.0; }
    let (lo, hi) = evals.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &e| (a.min(e), b.max(e)));
    hi - lo
}

// ───────────────────────────── precision ─────────────────────────────

/// Neumaier-compensated accumulator (the same algorithm as `Grid::integrate`).
#[derive(Clone, Copy, Default)]
struct Neumaier { sum: f64, comp: f64 }

impl Neumaier {
    #[inline]
    fn add(&mut self, x: f64) {
        let t = self.sum + x;
        if self.sum.abs() >= x.abs() { self.comp += (self.sum - t) + x; } else { self.comp += (x - t) + self.sum; }
        self.sum = t;
    }
    fn value(self) -> f64 { self.sum + self.comp }
}

// ───────────────────────────── §3 moments ─────────────────────────────

/// b_lm = ⟨ψ|S_lm(r − R)⟩ (alignment) and c_lm = ⟨ψ|Y_lm(r̂)⟩ (character), l ≤ 3, index
/// l² + l + m — both linear in ψ.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Moments { pub b: [f64; N_LM], pub c: [f64; N_LM] }

/// Solid harmonics S_lm(d) (the first 16 of `s` are used) and unit harmonics Y_lm(d̂), l ≤ 3.
#[inline]
fn harmonics(d: [f64; 3], s: &mut [f64; N_MOMENTS], y: &mut [f64; N_LM]) {
    solid_harmonics(d, s);
    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if r == 0.0 {
        *y = [0.0; N_LM];
        y[0] = 1.0 / (4.0 * PI).sqrt();
        return;
    }
    let mut rl = 1.0;
    for l in 0..4usize {
        let f = ((2 * l + 1) as f64 / (4.0 * PI)).sqrt() / rl;
        for i in l * l..(l + 1) * (l + 1) { y[i] = f * s[i]; }
        rl *= r;
    }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }

/// [`Moments`] of one orbital about `centre` (CPU reference of the GPU `orbital_moments` kernel).
pub fn moments(grid: &Grid, psi: &[f64], centre: [f64; 3]) -> Moments {
    moments_many(grid, &[psi], centre).remove(0)
}

/// [`moments`] of several orbitals in one pass over the grid (harmonics evaluated once per point).
pub fn moments_many(grid: &Grid, psis: &[&[f64]], centre: [f64; 3]) -> Vec<Moments> {
    let k = psis.len();
    for p in psis { assert_eq!(p.len(), grid.len()); }
    let mut ab = vec![[Neumaier::default(); N_LM]; k];
    let mut ac = vec![[Neumaier::default(); N_LM]; k];
    let (mut s, mut y) = ([0.0; N_MOMENTS], [0.0; N_LM]);
    for p in 0..grid.len() {
        harmonics(sub(grid.point(p), centre), &mut s, &mut y);
        for o in 0..k {
            let v = psis[o][p];
            if v == 0.0 { continue; }
            for i in 0..N_LM { ab[o][i].add(v * s[i]); ac[o][i].add(v * y[i]); }
        }
    }
    let dv = grid.dv();
    (0..k).map(|o| Moments { b: ab[o].map(|a| a.value() * dv), c: ac[o].map(|a| a.value() * dv) }).collect()
}

/// W_l = Σ_m c²_lm / Σ_{l' ≤ 3, m} c²_l'm for one orbital (zeros if c = 0).
pub fn character(m: &Moments) -> [f64; 4] { set_character(std::slice::from_ref(m)) }

/// W_l(C) of a set: the Frobenius form, invariant under any orthogonal mixing of C (§3).
pub fn set_character(ms: &[Moments]) -> [f64; 4] {
    let mut w = [0.0; 4];
    for m in ms {
        for (l, wl) in w.iter_mut().enumerate() {
            for i in l * l..(l + 1) * (l + 1) { *wl += m.c[i] * m.c[i]; }
        }
    }
    let total: f64 = w.iter().sum();
    if total > 0.0 { w.map(|x| x / total) } else { [0.0; 4] }
}

/// The l with W_l ≥ `L_CHAR_MIN`, if any.
pub fn dominant_l(m: &Moments) -> Option<u8> {
    character(m).iter().position(|&w| w >= L_CHAR_MIN).map(|l| l as u8)
}

/// Index (l² + l + m) of the k-th harmonic of l in display order: p as x, y, z; d and f by
/// m = −l … l.
pub fn display_lm(l: u8, k: usize) -> usize {
    assert!(l <= 3 && k <= 2 * l as usize);
    if l == 1 { [3, 1, 2][k] } else { (l as usize) * (l as usize) + k }
}

/// Name of the k-th real harmonic of l in display order ("" for s).
pub fn harmonic_name(l: u8, k: usize) -> &'static str {
    const NAMES: [&[&str]; 4] = [
        &[""],
        &["x", "y", "z"],
        &["xy", "yz", "z²", "xz", "x²−y²"],
        &["y(3x²−y²)", "xyz", "yz²", "z³", "xz²", "z(x²−y²)", "x(x²−3y²)"],
    ];
    NAMES[l as usize][k]
}

// ───────────────────────────── §4 shells ─────────────────────────────

/// One lone-atom shell: angular momentum l, the orbital indices in its channel (energy order),
/// and its eigenvalue spread (Ha), the grid's splitting of a physically degenerate set.
#[derive(Clone, Debug, PartialEq)]
pub struct Shell { pub l: u8, pub range: Range<usize>, pub spread: f64 }

/// Why a lone atom is shown with `#k` labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unlabelled {
    /// An occupied orbital has no dominant l (W_l < `L_CHAR_MIN` for every l ≤ 3).
    NoCharacter,
    /// An occupied l-set is incomplete in some channel: the potential is non-central (§4 item 3).
    OpenShell,
    /// A shell fails W_l(shell) ≥ `L_CHAR_MIN` or the rank test.
    SetTest,
    /// Radial node counts are missing, disagree within a shell, or violate Sturm–Liouville order.
    NodesUnresolved,
}

/// The lone-atom state of an update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoneAtom {
    /// More (or fewer) than one nucleus: molecular orbitals, `#k` labels.
    NotLoneAtom,
    /// Atomic labels; `shell_order_inverted` when a lower-l shell lies above a higher-l shell of
    /// the same n (the smeared nucleus's harmonic ordering, §4: not physical).
    Labelled { shell_order_inverted: bool },
    Unlabelled(Unlabelled),
}

/// spec §5.3: occupied orbitals of both channels → shells (l, range) per channel, or None (open
/// shell / unclassifiable). [`lone_atom_shells_why`] gives the reason.
pub fn lone_atom_shells(occ_moments: [&[Moments]; 2], evals: [&[f64]; 2]) -> Option<[Vec<Shell>; 2]> {
    lone_atom_shells_why(occ_moments, evals).ok()
}

/// [`lone_atom_shells`] with the reason for a refusal (§4 items 1–3).
pub fn lone_atom_shells_why(occ_moments: [&[Moments]; 2], evals: [&[f64]; 2]) -> Result<[Vec<Shell>; 2], Unlabelled> {
    let mut out = [Vec::new(), Vec::new()];
    for ch in 0..2 {
        let ms = occ_moments[ch];
        let e = evals[ch];
        assert!(e.len() >= ms.len(), "one eigenvalue per occupied orbital");
        let ls: Vec<u8> = ms.iter().map(dominant_l).collect::<Option<_>>().ok_or(Unlabelled::NoCharacter)?;
        let mut i = 0;
        while i < ls.len() {
            let l = ls[i];
            let mut j = i;
            while j < ls.len() && ls[j] == l { j += 1; }
            let d = 2 * l as usize + 1;
            if (j - i) % d != 0 { return Err(Unlabelled::OpenShell); }
            for s in (i..j).step_by(d) {
                let set = &ms[s..s + d];
                if set_character(set)[l as usize] < L_CHAR_MIN { return Err(Unlabelled::SetTest); }
                if alignment(set, l).1 < ALIGN_RANK_MIN { return Err(Unlabelled::SetTest); }
                out[ch].push(Shell { l, range: s..s + d, spread: spread(&e[s..s + d]) });
            }
            i = j;
        }
    }
    Ok(out)
}

/// Lone-atom alignment (§4 item 4): U = VWᵀ for M_aj = b_{l, m_a}(ψ_j) = WΣVᵀ (rows a in display
/// order), so ψ'_k = Σ_j ψ_j U_jk is aligned to the k-th harmonic with ⟨ψ'_k|S_{m_k}⟩ ≥ 0.
/// Also returns σ_min/σ_max of M (0 if M = 0).
pub fn alignment(ms: &[Moments], l: u8) -> (DMatrix<f64>, f64) {
    let d = 2 * l as usize + 1;
    assert_eq!(ms.len(), d, "a shell of l = {l} has {d} members");
    let m = DMatrix::from_fn(d, d, |a, j| ms[j].b[display_lm(l, a)]);
    let svd = m.svd(true, true);
    let (w, vt) = (svd.u.expect("U requested"), svd.v_t.expect("Vᵀ requested"));
    let (lo, hi) = svd.singular_values.iter().fold((f64::INFINITY, 0.0f64), |(a, b), &x| (a.min(x), b.max(x)));
    (vt.transpose() * w.transpose(), if hi > 0.0 { lo / hi } else { 0.0 })
}

/// Continuity Procrustes (§5 item 1): U = WVᵀ for the square A = WΣVᵀ, maximising tr(UᵀA).
pub fn procrustes_continuity(a: &DMatrix<f64>) -> DMatrix<f64> {
    assert!(a.is_square());
    let svd = a.clone().svd(true, true);
    svd.u.expect("U requested") * svd.v_t.expect("Vᵀ requested")
}

// ───────────────────────────── §4 radial nodes ─────────────────────────────

/// Radial-shell projections of one orbital about the nucleus (§4 item 5): bin k covers
/// r ∈ [k·dr, (k+1)·dr) inside the largest sphere about the centre that fits in the grid;
/// `a[k][lm]` = Σ_{p∈k} ψ Y_lm(r̂) h³ and `count[k]` the number of points.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct RadialProfile { pub dr: f64, pub a: Vec<[f64; N_LM]>, pub count: Vec<u32> }

/// Radius of the largest sphere about `centre` inside the grid's point set.
fn inscribed_radius(grid: &Grid, centre: [f64; 3]) -> f64 {
    (0..3).map(|a| {
        let lo = centre[a] - grid.origin[a];
        let hi = grid.origin[a] + (grid.n[a] - 1) as f64 * grid.h - centre[a];
        lo.min(hi)
    }).fold(f64::INFINITY, f64::min).max(0.0)
}

/// [`RadialProfile`] of one orbital (CPU reference of the GPU profile reduction).
pub fn radial_profile(grid: &Grid, psi: &[f64], centre: [f64; 3]) -> RadialProfile {
    radial_profiles(grid, &[psi], centre).remove(0)
}

/// [`radial_profile`] of several orbitals in one pass.
pub fn radial_profiles(grid: &Grid, psis: &[&[f64]], centre: [f64; 3]) -> Vec<RadialProfile> {
    let dr = RADIAL_BIN_H * grid.h;
    let r_max = inscribed_radius(grid, centre);
    let nb = (r_max / dr).ceil() as usize;
    let k = psis.len();
    for p in psis { assert_eq!(p.len(), grid.len()); }
    let mut acc = vec![vec![[Neumaier::default(); N_LM]; nb]; k];
    let mut count = vec![0u32; nb];
    let (mut s, mut y) = ([0.0; N_MOMENTS], [0.0; N_LM]);
    for p in 0..grid.len() {
        let d = sub(grid.point(p), centre);
        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if r >= r_max { continue; }
        let bin = ((r / dr) as usize).min(nb - 1);
        harmonics(d, &mut s, &mut y);
        count[bin] += 1;
        for o in 0..k {
            let v = psis[o][p];
            for i in 0..N_LM { acc[o][bin][i].add(v * y[i]); }
        }
    }
    let dv = grid.dv();
    acc.into_iter().map(|bins| RadialProfile {
        dr,
        a: bins.into_iter().map(|b| b.map(|x| x.value() * dv)).collect(),
        count: count.clone(),
    }).collect()
}

/// Radial node count ν of an orbital of angular momentum `l` from its shell profile (§4 item 5):
/// sign changes of R̄_k = Σ_m ĉ_m a_k,lm / n_k over the bins with |R̄_k| ≥ `NODE_FLOOR`·max|R̄|.
/// None if the orbital has no l component or no populated bin.
pub fn profile_nodes(profile: &RadialProfile, m: &Moments, l: u8) -> Option<usize> {
    let lo = (l as usize) * (l as usize);
    let d = 2 * l as usize + 1;
    let norm = m.c[lo..lo + d].iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm.is_nan() || norm <= 0.0 { return None; }
    let vals: Vec<f64> = profile.a.iter().zip(&profile.count).filter(|(_, &n)| n > 0)
        .map(|(a, &n)| (0..d).map(|i| m.c[lo + i] / norm * a[lo + i]).sum::<f64>() / n as f64)
        .collect();
    count_sign_changes(&vals, NODE_FLOOR)
}

/// Sign changes among the entries with |v| ≥ floor·max|v| (None if all are zero).
fn count_sign_changes(vals: &[f64], floor: f64) -> Option<usize> {
    let max = vals.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    if max.is_nan() || max <= 0.0 { return None; }
    let mut last = 0.0f64;
    let mut n = 0;
    for &v in vals.iter().filter(|v| v.abs() >= floor * max) {
        if last != 0.0 && v.signum() != last.signum() { n += 1; }
        last = v;
    }
    Some(n)
}

/// Nodes of ψ along the ray centre + t·dir (trilinear samples every h/4 until the ray leaves the
/// grid), counted among samples with |ψ| ≥ floor × the ray's max. A test cross-check of
/// [`profile_nodes`]; the runtime never calls it.
pub fn radial_nodes(grid: &Grid, psi: &[f64], centre: [f64; 3], dir: [f64; 3], floor: f64) -> usize {
    let nd = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    let u = [dir[0] / nd, dir[1] / nd, dir[2] / nd];
    let step = grid.h / 4.0;
    let mut vals = Vec::new();
    let mut t = 0.0;
    loop {
        let f: [f64; 3] = [0, 1, 2].map(|a| (centre[a] + t * u[a] - grid.origin[a]) / grid.h);
        if (0..3).any(|a| f[a] < 0.0 || f[a] > (grid.n[a] - 1) as f64) { break; }
        let i0: [usize; 3] = [0, 1, 2].map(|a| (f[a].floor() as usize).min(grid.n[a] - 2));
        let w: [f64; 3] = [0, 1, 2].map(|a| f[a] - i0[a] as f64);
        let mut v = 0.0;
        for c in 0..8usize {
            let bit = [c & 1, (c >> 1) & 1, (c >> 2) & 1];
            let wt: f64 = (0..3).map(|a| if bit[a] == 1 { w[a] } else { 1.0 - w[a] }).product();
            v += wt * psi[grid.idx(i0[0] + bit[0], i0[1] + bit[1], i0[2] + bit[2])];
        }
        vals.push(v);
        t += step;
    }
    count_sign_changes(&vals, floor).unwrap_or(0)
}

/// Labels of the lone atom's shells (§4 item 5): per channel and shell, n = l + 1 + N_core,l + ν.
#[derive(Clone, Debug, PartialEq)]
pub struct LabelSet { pub n: [Vec<u8>; 2], pub shell_order_inverted: bool }

/// n per shell from the per-orbital node counts (`nodes[ch][i]` for occupied orbital i), with the
/// Sturm–Liouville consistency check, and the inversion flag (§4 item 5).
pub fn label_shells(shells: &[Vec<Shell>; 2], nodes: [&[Option<usize>]; 2], evals: [&[f64]; 2], n_core: [u8; 4])
    -> Result<LabelSet, Unlabelled> {
    let mut n_out = [Vec::new(), Vec::new()];
    let mut inverted = false;
    for ch in 0..2 {
        let sh = &shells[ch];
        let mean = |s: &Shell| evals[ch][s.range.clone()].iter().sum::<f64>() / s.range.len() as f64;
        let mut nu = Vec::with_capacity(sh.len());
        for s in sh {
            let first = nodes[ch][s.range.start].ok_or(Unlabelled::NodesUnresolved)?;
            if s.range.clone().any(|i| nodes[ch][i] != Some(first)) { return Err(Unlabelled::NodesUnresolved); }
            nu.push(first);
        }
        for l in 0..4u8 {
            let mut idx: Vec<usize> = (0..sh.len()).filter(|&i| sh[i].l == l).collect();
            idx.sort_by(|&a, &b| mean(&sh[a]).total_cmp(&mean(&sh[b])));
            for (rank, &i) in idx.iter().enumerate() {
                if nu[i] != rank { return Err(Unlabelled::NodesUnresolved); }
            }
        }
        let n: Vec<u8> = sh.iter().zip(&nu).map(|(s, &v)| s.l + 1 + n_core[s.l as usize] + v as u8).collect();
        for a in 0..sh.len() {
            for b in 0..sh.len() {
                if sh[a].l < sh[b].l && n[a] == n[b] && mean(&sh[a]) > mean(&sh[b]) { inverted = true; }
            }
        }
        n_out[ch] = n;
    }
    Ok(LabelSet { n: n_out, shell_order_inverted: inverted })
}

// ───────────────────────────── §5 signs and matching ─────────────────────────────

/// The nine windowed first-appearance functionals ⟨ψ|w⟩/‖w‖ about the nuclear centroid (§5
/// item 3): monopole, three dipoles, five quadrupoles (S_2m order). Zeros without nuclei.
pub fn first_sign_functionals(grid: &Grid, psi: &[f64], nuclei: &[[f64; 3]]) -> [f64; 9] {
    if nuclei.is_empty() { return [0.0; 9]; }
    let nn = nuclei.len() as f64;
    let xbar = [0, 1, 2].map(|a| nuclei.iter().map(|r| r[a]).sum::<f64>() / nn);
    let mut pw = [Neumaier::default(); 9];
    let mut ww = [Neumaier::default(); 9];
    let mut s = [0.0; N_MOMENTS];
    for p in 0..grid.len() {
        let r = grid.point(p);
        let w0: f64 = nuclei.iter().map(|n| { let d = sub(r, *n); (-(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])).exp() }).sum();
        let d = sub(r, xbar);
        let g = (-(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])).exp();
        solid_harmonics(d, &mut s);
        let w = [w0, d[0] * g, d[1] * g, d[2] * g, s[4] * g, s[5] * g, s[6] * g, s[7] * g, s[8] * g];
        for i in 0..9 { pw[i].add(psi[p] * w[i]); ww[i].add(w[i] * w[i]); }
    }
    let dv = grid.dv();
    let mut out = [0.0; 9];
    for i in 0..9 {
        let n2 = ww[i].value() * dv;
        if n2 > 0.0 { out[i] = pw[i].value() * dv / n2.sqrt(); }
    }
    out
}

/// The sign that makes the first functional with |value| ≥ `SIGN_MOMENT_MIN` positive, and its
/// index; (1, None) when none qualifies — the raw solver sign is kept (not deterministic, §5).
pub fn first_appearance_sign(f: &[f64; 9]) -> (f64, Option<usize>) {
    match f.iter().position(|v| v.abs() >= SIGN_MOMENT_MIN) {
        Some(i) => (if f[i] > 0.0 { 1.0 } else { -1.0 }, Some(i)),
        None => (1.0, None),
    }
}

/// Maximum-weight assignment of rows to columns (Hungarian / Kuhn–Munkres with potentials,
/// O(K³) on the square padding of `w`). Every row gets a column when there are enough columns,
/// zero-weight pairs included; the caller thresholds.
pub fn hungarian_max(w: &DMatrix<f64>) -> Vec<Option<usize>> {
    let (n, m) = w.shape();
    if n == 0 { return Vec::new(); }
    if m == 0 { return vec![None; n]; }
    let sz = n.max(m);
    let cost = |i: usize, j: usize| if i < n && j < m { -w[(i, j)] } else { 0.0 };
    let (mut u, mut v) = (vec![0.0; sz + 1], vec![0.0; sz + 1]);
    let (mut p, mut way) = (vec![0usize; sz + 1], vec![0usize; sz + 1]);
    for i in 1..=sz {
        p[0] = i;
        let mut j0 = 0;
        let mut minv = vec![f64::INFINITY; sz + 1];
        let mut used = vec![false; sz + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let (mut delta, mut j1) = (f64::INFINITY, 0);
            for j in 1..=sz {
                if used[j] { continue; }
                let cur = cost(i0 - 1, j - 1) - u[i0] - v[j];
                if cur < minv[j] { minv[j] = cur; way[j] = j0; }
                if minv[j] < delta { delta = minv[j]; j1 = j; }
            }
            for j in 0..=sz {
                if used[j] { u[p[j]] += delta; v[j] -= delta; } else { minv[j] -= delta; }
            }
            j0 = j1;
            if p[j0] == 0 { break; }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 { break; }
        }
    }
    let mut ans = vec![None; n];
    for j in 1..=sz {
        if p[j] != 0 && p[j] - 1 < n && j - 1 < m { ans[p[j] - 1] = Some(j - 1); }
    }
    ans
}

// ───────────────────────────── §6, §7 ─────────────────────────────

/// Fraction of point q's cell [(q + ½)h, (q + 3/2)h] (distance from the low face, which lies one
/// step outside the grid) that falls within L/6 of either face of an axis of n points.
fn outer_fraction(q: usize, n: usize) -> f64 {
    let slab = (n + 1) as f64 / 6.0; // L/6 in units of h
    let near = (q + 1).min(n - q) as f64; // distance to the nearer face, in h
    (slab - (near - 0.5)).clamp(0.0, 1.0)
}

/// ∫ψ² over the region within L/6 of any face, L = (n + 1)h per axis (the Dirichlet box) (§6).
/// Each point carries the fraction of its cell inside that region, 1 − Π_a(1 − f_a) (the inner
/// box and the cells are product sets), so the measure converges as O(h²) and does not depend on
/// where L/6 falls between grid points; counting whole points with distance < L/6 instead is an
/// O(h/L) error (0.149 instead of 0.163 for the lowest box state at 33³).
pub fn outer_weight(grid: &Grid, psi: &[f64]) -> f64 {
    let f: [Vec<f64>; 3] = [0, 1, 2].map(|a| (0..grid.n[a]).map(|q| outer_fraction(q, grid.n[a])).collect());
    let mut acc = Neumaier::default();
    for k in 0..grid.n[2] {
        for j in 0..grid.n[1] {
            let inner_jk = (1.0 - f[1][j]) * (1.0 - f[2][k]);
            for i in 0..grid.n[0] {
                let w = 1.0 - (1.0 - f[0][i]) * inner_jk;
                if w > 0.0 { let v = psi[grid.idx(i, j, k)]; acc.add(w * v * v); }
            }
        }
    }
    acc.value() * grid.dv()
}

/// ε ≥ 0 or outer-L/6 weight > `BOX_STATE_OUTER_WEIGHT` (§6).
pub fn is_box_state(eps: f64, outer_weight: f64) -> bool { eps >= 0.0 || outer_weight > BOX_STATE_OUTER_WEIGHT }

/// [`is_box_state`], or the orbital's density on the outermost grid plane above `face_limit`
/// (e/Bohr³ for one electron in it; §8 — the caller passes the surrogate's box-affected face
/// density, ∞ to disable).
pub fn box_state(eps: f64, outer_weight: f64, face_density: f64, face_limit: f64) -> bool {
    is_box_state(eps, outer_weight) || face_density > face_limit
}

/// max ψ² over the outermost grid plane of each of the six sides (`qm::scf`'s `max_on_faces`
/// applied to ψ²): the density one electron in ψ puts one spacing h inside the Dirichlet wall (§8).
pub fn face_density(grid: &Grid, psi: &[f64]) -> f64 { face_density_set(grid, &[psi]) }

/// The face density of a degenerate set (§8, review M1): max over the outermost-plane points of
/// Σ_k ψ_k² — rotation-invariant within the set, the supremum of every member state's face density.
pub fn face_density_set(grid: &Grid, set: &[&[f64]]) -> f64 {
    let [n0, n1, n2] = grid.n;
    let mut m = 0.0f64;
    for k in 0..n2 {
        for j in 0..n1 {
            let edge_jk = j == 0 || k == 0 || j == n1 - 1 || k == n2 - 1;
            for i in 0..n0 {
                if edge_jk || i == 0 || i == n0 - 1 {
                    let p = grid.idx(i, j, k);
                    m = m.max(set.iter().map(|psi| psi[p] * psi[p]).sum::<f64>());
                }
            }
        }
    }
    m
}

/// Spin-density readout (spec §5.5): electron counts, ∫m, ∫|m|, max|m|, max ρ.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpinSummary { pub n_up: usize, pub n_dn: usize, pub integral: f64, pub abs_integral: f64, pub max_abs: f64, pub rho_max: f64 }

/// Unit roundoff of f32 arithmetic, 2⁻²⁴: the relative error of one rounded operation.
pub const F32_EPS: f64 = 5.960464477539063e-8;

impl SpinSummary {
    /// Display scale max(max|m|, `SPIN_FLOOR`·max ρ) (P4).
    pub fn scale(&self) -> f64 { self.max_abs.max(SPIN_FLOOR * self.rho_max) }

    /// The L1 norm ∫|m| (electrons) up to which m = ρ↑ − ρ↓ is taken to be the solver's noise, at
    /// SCF tolerance `scf_tol` (on Δρ) and eigensolver tolerance `eig_tol` (the residual
    /// ‖Hψ − εψ‖ in Hartree that the last SCF step ran at).
    ///
    /// **An empirical envelope, not a proven bound** (rule 7). For a closed shell m = 0 exactly,
    /// so all of m is error; m is the difference of two separately solved channels. Each term
    /// below says what is exact and what is an assumption.
    /// - *SCF convergence, 2·`scf_tol` — a rule of thumb.* The loop stops when
    ///   Δρ = Σσ ‖ρ_out,σ − ρ_in,σ‖₁ < `scf_tol`, so ‖m_out − m_in‖₁ < `scf_tol` (exact, triangle
    ///   inequality). That the mixed input m_in is within another `scf_tol` of the fixed point is
    ///   an **assumption**: the distance to a fixed point is step/(1 − κ) for a contraction
    ///   factor κ, which is large near stagnation, and DIIS guarantees no κ.
    /// - *Eigenvector error, 2·(n↑ + n↓)·`eig_tol` — a bound with an assumed gap.* An orbital with
    ///   residual r has ‖δψ‖ ≤ r/g, g the gap to the orbitals excluded from the set (exact), the
    ///   density moves by 2ψδψ, ∫|2ψδψ| ≤ 2‖δψ‖ (Cauchy–Schwarz, exact), and the n↑ + n↓
    ///   orbitals' errors add (exact). The display does not know g, so **g = 1 Ha is assumed**.
    ///   A real HOMO–LUMO gap is typically 0.1–0.4 Ha (LiH's is the small one) and shrinks in
    ///   larger systems, so the formal bound for those is 2.5–10× larger than this term; the
    ///   actual residuals sit far below `eig_tol` (LOBPCG stops at it), which is what makes the
    ///   assumption hold in the measurements. This is the grid-aware term: the GPU solver's
    ///   effective `eig_tol` rises as 1/h² (`GpuSolver::effective_eig_tol`: 3.8e-6 at
    ///   h = 0.30 → 3.4e-5 at h = 0.10).
    /// - *f32 density precision, (Kσ + 2)·`F32_EPS`·Nσ per channel — an honest worst case.*
    ///   ρ_σ(r) = Σᵢ ψᵢ² is accumulated from f32 orbitals, a rounding per square and per addition,
    ///   a relative error of at most (Kσ + 2)·`F32_EPS` per point (Kσ = Nσ orbitals at
    ///   occupation 1).
    ///
    /// **What backs the two assumptions is measurement, with a thin margin.** Closed-shell H₂ and
    /// LiH at h = 0.30, 0.25, 0.15, 0.10, tol 1e-5 (`validate` `diag_spin_noise_tail`: 60 warm
    /// MD-like steps with density extrapolation): worst ∫|m| 6.0e-5 (LiH Accurate, the default)
    /// against a budget of 1.4e-4, 2.3× (H₂ Precise 3.1e-5 against 1.6e-4); the gated test
    /// `spin_view_hides_closed_shell_noise_and_keeps_real_spin` (9 steps, also He and H₂O) has its
    /// tightest case, H₂ Fast, at 3.0×. The data are heavy-tailed and not monotone in h (H₂ 81³
    /// 4.7e-7, 121³ 3.1e-5), so 2–3× over 60 steps is not a bound over a long run. The budget was
    /// not widened to fit.
    ///
    /// **What exceeding it costs is cosmetic:** the verdict has no hysteresis, so a closed shell
    /// whose noise tail crosses the budget is drawn, at the `SPIN_FLOOR` scale, until it falls
    /// back (the old behaviour, which cannot hide a real spin either way); a spin-broken
    /// singlet near its onset (stretched H₂ past the Coulson–Fischer point) is hidden until its
    /// ∫|m| passes the budget (~1e-4), and the readout always carries the numbers.
    ///
    /// L1 rather than a pointwise ceiling because the budget is in the norm the solver converges
    /// in: a ceiling on max|m| needs the volume the noise is spread over (one cell at the
    /// least, η/h³ — at Precise that exceeds the spin density of a real unpaired electron).
    /// Real spin never hides in an L1 budget: it integrates to |∫m| = |n↑ − n↓| ≥ 1 electron
    /// (≥ 4400× the budget, measured: lone Li ∫|m| = 1.00, the H₂ triplet 2.00, at every grid).
    pub fn noise_l1(&self, scf_tol: f64, eig_tol: f64) -> f64 {
        let f32_term = |n: usize| F32_EPS * (n as f64 + 2.0) * n as f64;
        2.0 * scf_tol + 2.0 * (self.n_up + self.n_dn) as f64 * eig_tol + f32_term(self.n_up) + f32_term(self.n_dn)
    }

    /// True when m is within the solver's noise ([`Self::noise_l1`]): a closed shell, or m too
    /// small to resolve. An open shell (|∫m| ≥ 1) never is.
    pub fn is_noise(&self, scf_tol: f64, eig_tol: f64) -> bool { self.abs_integral <= self.noise_l1(scf_tol, eig_tol) }
}

/// [`SpinSummary`] of m = ρ↑ − ρ↓ (CPU reference; compensated sums).
pub fn spin_summary(grid: &Grid, rho_up: &[f64], rho_dn: &[f64], n_up: usize, n_dn: usize) -> SpinSummary {
    let (mut int, mut abs) = (Neumaier::default(), Neumaier::default());
    let (mut max_abs, mut rho_max) = (0.0f64, 0.0f64);
    for (u, d) in rho_up.iter().zip(rho_dn) {
        let m = u - d;
        int.add(m);
        abs.add(m.abs());
        max_abs = max_abs.max(m.abs());
        rho_max = rho_max.max(u + d);
    }
    SpinSummary { n_up, n_dn, integral: int.value() * grid.dv(), abs_integral: abs.value() * grid.dv(), max_abs, rho_max }
}

/// CPU reference of the colour volume (spec §5.6): per point rgb = Σ fᵢcᵢψ̃ᵢ²/Pᵢ, a = Σ fᵢψ̃ᵢ²/Pᵢ
/// for `shown` = (ψ̃ᵢ, fᵢ, cᵢ, Pᵢ).
pub fn colour_volume(grid: &Grid, shown: &[(&[f64], f64, [f64; 3], f64)]) -> Vec<[f64; 4]> {
    (0..grid.len()).map(|p| {
        let mut v = [0.0; 4];
        for &(psi, f, c, pk) in shown {
            let s = f * psi[p] * psi[p] / pk;
            v[0] += c[0] * s;
            v[1] += c[1] * s;
            v[2] += c[2] * s;
            v[3] += s;
        }
        v
    }).collect()
}

// ───────────────────────────── the update ─────────────────────────────

/// Everything one display update needs, per channel `[↑, ↓]`. Rows are orbitals in ascending
/// energy, occupied first; `n_computed` = n_occ + m + `UNOCC_GUARD` when m unoccupied rows are
/// requested, else n_occ. Per-orbital slices have `n_computed` entries (`moments`/`radial`: at
/// least `n_occ`).
pub struct UpdateInput<'a> {
    pub evals: [&'a [f64]; 2],
    pub occupations: [&'a [f64]; 2],
    /// ⟨ψᵢ|ψ̃ⱼ^prev⟩: rows the current raw orbitals, columns the rows shown by the previous update
    /// (index order). None: no previous display.
    pub overlaps_prev: [Option<&'a DMatrix<f64>>; 2],
    /// ⟨ψᵢ↑|ψⱼ↓⟩ of the raw orbitals.
    pub cross: Option<&'a DMatrix<f64>>,
    /// Some iff lone atom (exactly one nucleus), together with `radial`.
    pub moments: Option<[&'a [Moments]; 2]>,
    /// Radial profiles about the nucleus (controller ruling 1: labels by node count).
    pub radial: Option<[&'a [RadialProfile]; 2]>,
    /// [`first_sign_functionals`] of the raw orbitals.
    pub first_sign: [&'a [[f64; 9]]; 2],
    pub max_abs: [&'a [f64]; 2],
    pub outer_weight: [&'a [f64]; 2],
    pub n_occ: [usize; 2],
    pub n_computed: [usize; 2],
    /// N_core,l for l ≤ 3 (zeros until M7).
    pub n_core: [u8; 4],
    /// Lone atom: Z·σ for the smeared-1s tag.
    pub z_sigma: Option<f64>,
    pub tol_scf: f64,
    /// The display eigensolve that produced rows n_occ.. (§8): their residuals, convergence and
    /// face densities (`n_computed[ch]` = n_occ + `evals[ch].len()` of it). None without one.
    pub unoccupied: Option<&'a Unoccupied>,
    /// The box rule's face-density limit (§8; e/Bohr³ for one electron; ∞ disables it).
    pub face_limit: f64,
}

/// A row label: `#k` (1-based index in the channel) or an atomic n l m name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Label { Index(usize), Atomic { n: u8, l: u8, m: &'static str } }

impl std::fmt::Display for Label {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Label::Index(k) => write!(f, "#{k}"),
            Label::Atomic { n, l, m } => {
                write!(f, "{n}{}", ['s', 'p', 'd', 'f'][*l as usize])?;
                if m.is_empty() { Ok(()) } else { write!(f, "_{m}") }
            }
        }
    }
}

/// One shown orbital. `energy` is ⟨ψ̃|H|ψ̃⟩ = Σⱼ U²ⱼₖεⱼ (the eigenvalue unless rotated);
/// `eps_range` = (lowest, highest) **eigenvalue** of the row's set (both = `energy` for a row of
/// its own) — what HOMO, LUMO and the gap are read off, because for a rotated member `energy` is a
/// Rayleigh quotient between them, not an eigenvalue;
/// `cluster` = (size, spread) of its set when > 1; `grid_split` = the spread of an aligned
/// lone-atom shell (degenerate in physics, split by the grid). Unoccupied rows of a display
/// eigensolve (§8): `residual` = the largest member residual of the row's set, `not_converged`
/// = a member ended above the eigensolve's tolerance; None/false for occupied rows.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub spin: Spin, pub index: usize, pub track: u32, pub slot: u8, pub energy: f64, pub eps_range: (f64, f64), pub occupation: f64,
    pub label: Label, pub cluster: Option<(usize, f64)>, pub rotated: bool, pub grid_split: Option<f64>,
    pub box_state: bool, pub smeared_1s: bool, pub residual: Option<f64>, pub not_converged: bool,
}

/// The result of an update. `rotations[ch]` partitions the shown rows of the channel into ranges
/// with their d × d matrices U (signs folded in): ψ̃_{start+k} = Σⱼ ψ_{start+j} U_jk.
/// `scale[ch][i]` = max|ψ| over row i's set.
#[derive(Clone, Debug)]
pub struct UpdateOutput {
    pub rotations: [Vec<(Range<usize>, DMatrix<f64>)>; 2],
    pub rows: Vec<Row>,
    pub scale: [Vec<f64>; 2],
    pub hidden_unoccupied: [usize; 2],
    pub lone_atom: LoneAtom,
}

#[derive(Clone, Copy, PartialEq)]
enum Kind { Shell { l: u8, n: u8 }, Cluster { rotated: bool } }

struct Block { range: Range<usize>, u: DMatrix<f64>, kind: Kind, spread: f64 }

struct Plan { n_shown: usize, hidden: usize, blocks: Vec<Block>, matched: Vec<Option<usize>>, energies: Vec<f64> }

impl Plan {
    /// The n_shown × n_shown block-diagonal rotation (signs included).
    fn matrix(&self) -> DMatrix<f64> {
        let mut d = DMatrix::zeros(self.n_shown, self.n_shown);
        for b in &self.blocks {
            let n = b.range.len();
            d.view_mut((b.range.start, b.range.start), (n, n)).copy_from(&b.u);
        }
        d
    }
}

/// Rows shown and vectors hidden by the block-edge rule (§6) for one channel's sorted `evals`
/// (occupied first): (n_shown, hidden).
pub fn block_edge(evals: &[f64], n_occ: usize, n_computed: usize, tau: f64) -> (usize, usize) {
    if n_computed <= n_occ { return (n_computed, 0); }
    let m = n_computed.saturating_sub(n_occ + UNOCC_GUARD);
    let n_unocc = n_computed - n_occ;
    let mut shown = n_occ;
    for c in clusters(&evals[n_occ..n_computed], tau) {
        if c.start >= m { break; }
        if c.end == n_unocc { return (shown, c.len()); }
        shown = n_occ + c.end;
    }
    (shown, 0)
}

type AtomLabels = ([Vec<Shell>; 2], LabelSet);

/// Tracks, colour slots and the previous display (§5). One per producer (spec §4.3).
pub struct Tracker {
    /// (track, slot) of each row shown by the previous update, per channel, in index order.
    prev: [Vec<(u32, u8)>; 2],
    started: bool,
    next_id: u32,
}

impl Default for Tracker { fn default() -> Self { Tracker::new() } }

impl Tracker {
    pub fn new() -> Tracker { Tracker { prev: [Vec::new(), Vec::new()], started: false, next_id: 0 } }

    /// Forget the previous display (grid change, provider reset, view off). Ids keep counting.
    pub fn reset(&mut self) {
        self.prev = [Vec::new(), Vec::new()];
        self.started = false;
    }

    /// One display update: lone-atom shells and labels (§4), clusters and the block edge (§1, §6),
    /// alignment or continuity rotations, matching, signs, tracks and colour slots (§5).
    pub fn update(&mut self, input: &UpdateInput) -> UpdateOutput {
        let tau = tau(input.tol_scf);
        for ch in 0..2 {
            let n = input.n_computed[ch];
            assert!(input.n_occ[ch] <= n, "n_occ ≤ n_computed");
            for (len, what) in [(input.evals[ch].len(), "evals"), (input.occupations[ch].len(), "occupations"),
                                (input.first_sign[ch].len(), "first_sign"), (input.max_abs[ch].len(), "max_abs"),
                                (input.outer_weight[ch].len(), "outer_weight")] {
                assert_eq!(len, n, "{what}[{ch}] needs one entry per computed orbital");
            }
            if let Some(u) = input.unoccupied {
                let k = u.evals[ch].len();
                assert_eq!(n, input.n_occ[ch] + k, "n_computed[{ch}] = n_occ + the display eigensolve's vectors");
                assert!(u.residuals[ch].len() == k && u.converged[ch].len() == k && u.face_density[ch].len() == k, "Unoccupied[{ch}] lengths");
            }
        }
        let (lone_atom, atom) = analyse_lone_atom(input);
        let plans: Vec<Plan> = (0..2).map(|ch| self.plan_channel(input, ch, tau, atom.as_ref())).collect();

        // tracks and slots: survivors first, then new tracks (↑ before ↓, energy order)
        let mut usage = [0u32; PALETTE_SLOTS as usize];
        let mut assigned: [Vec<Option<(u32, u8)>>; 2] = [vec![None; plans[0].n_shown], vec![None; plans[1].n_shown]];
        for ch in 0..2 {
            for (i, m) in plans[ch].matched.iter().enumerate() {
                if let Some(j) = *m {
                    let t = self.prev[ch][j];
                    assigned[ch][i] = Some(t);
                    usage[t.1 as usize] += 1;
                }
            }
        }
        let cross = input.cross.map(|x| {
            let (n0, n1) = (plans[0].n_shown, plans[1].n_shown);
            assert!(x.nrows() >= n0 && x.ncols() >= n1, "cross overlaps cover the shown rows");
            plans[0].matrix().transpose() * x.view((0, 0), (n0, n1)) * plans[1].matrix()
        });
        for ch in 0..2 {
            for i in 0..plans[ch].n_shown {
                if assigned[ch][i].is_some() { continue; }
                let shared = if ch == 1 {
                    cross.as_ref().and_then(|x| {
                        (0..x.nrows()).max_by(|&a, &b| x[(a, i)].abs().total_cmp(&x[(b, i)].abs()))
                            .filter(|&u| x[(u, i)].abs() >= SHARE_COLOUR_MIN)
                            .and_then(|u| assigned[0][u].map(|t| t.1))
                    })
                } else { None };
                let slot = shared.unwrap_or_else(|| {
                    (0..PALETTE_SLOTS).min_by_key(|&s| (usage[s as usize], s)).expect("at least one slot")
                });
                usage[slot as usize] += 1;
                assigned[ch][i] = Some((self.next_id, slot));
                self.next_id += 1;
            }
        }

        let mut rows = Vec::new();
        let mut rotations = [Vec::new(), Vec::new()];
        let mut scale = [Vec::new(), Vec::new()];
        for ch in 0..2 {
            let plan = &plans[ch];
            let spin = if ch == 0 { Spin::Up } else { Spin::Down };
            let (e, occ) = (input.evals[ch], input.occupations[ch]);
            scale[ch] = vec![0.0; plan.n_shown];
            let n_occ = input.n_occ[ch];
            // the display eigensolve's data of computed orbital i (§8), unoccupied rows only
            let unocc = |i: usize| input.unoccupied.filter(|_| i >= n_occ).map(|u| (u.residuals[ch][i - n_occ], u.converged[ch][i - n_occ], u.face_density[ch][i - n_occ]));
            for b in &plan.blocks {
                let d = b.range.len();
                let smax = b.range.clone().map(|i| input.max_abs[ch][i]).fold(0.0, f64::max);
                let eps_range = b.range.clone().map(|i| e[i]).fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| (lo.min(x), hi.max(x)));
                let boxy = b.range.clone().any(|i| box_state(e[i], input.outer_weight[ch][i], unocc(i).map_or(0.0, |u| u.2), input.face_limit));
                // clusters never cross an occupation change, so a block is all unoccupied or none
                let residual = b.range.clone().filter_map(|i| unocc(i).map(|u| u.0)).reduce(f64::max);
                let not_converged = b.range.clone().any(|i| unocc(i).is_some_and(|u| !u.1));
                for k in 0..d {
                    let i = b.range.start + k;
                    let (track, slot) = assigned[ch][i].expect("every shown row has a track");
                    let (label, rotated, grid_split, smeared_1s) = match b.kind {
                        Kind::Shell { l, n } => (
                            Label::Atomic { n, l, m: harmonic_name(l, k) },
                            l >= 1,
                            (l >= 1).then_some(b.spread),
                            l == 0 && n == 1 && input.z_sigma.is_some_and(|z| z >= SMEARED_ZSIGMA),
                        ),
                        Kind::Cluster { rotated } => (Label::Index(i + 1), rotated, None, false),
                    };
                    rows.push(Row {
                        spin, index: i, track, slot, energy: plan.energies[i], eps_range, occupation: occ[i], label,
                        cluster: (d > 1).then_some((d, b.spread)), rotated, grid_split,
                        box_state: i >= n_occ && boxy, smeared_1s, residual, not_converged,
                    });
                    scale[ch][i] = smax;
                }
                rotations[ch].push((b.range.clone(), b.u.clone()));
            }
            self.prev[ch] = assigned[ch].iter().map(|a| a.expect("assigned")).collect();
        }
        self.started = true;
        UpdateOutput { rotations, rows, scale, hidden_unoccupied: [plans[0].hidden, plans[1].hidden], lone_atom }
    }

    fn plan_channel(&self, input: &UpdateInput, ch: usize, tau: f64, atom: Option<&AtomLabels>) -> Plan {
        let (e, occ) = (input.evals[ch], input.occupations[ch]);
        let n_occ = input.n_occ[ch];
        let (n_shown, hidden) = block_edge(e, n_occ, input.n_computed[ch], tau);

        // partition: aligned shells (labelled lone atom), then τ clusters within occupation runs
        let mut blocks = Vec::new();
        let mut free_from = 0;
        if let (Some((shells, labels)), Some(ms)) = (atom, input.moments) {
            for (si, sh) in shells[ch].iter().enumerate() {
                let u = if sh.l == 0 { DMatrix::identity(1, 1) } else { alignment(&ms[ch][sh.range.clone()], sh.l).0 };
                blocks.push(Block { range: sh.range.clone(), u, kind: Kind::Shell { l: sh.l, n: labels.n[ch][si] }, spread: sh.spread });
            }
            free_from = n_occ;
        }
        let mut i = free_from;
        while i < n_shown {
            let mut j = i + 1;
            while j < n_shown && occ[j] == occ[i] { j += 1; }
            for c in clusters(&e[i..j], tau) {
                let r = i + c.start..i + c.end;
                blocks.push(Block { u: DMatrix::identity(r.len(), r.len()), spread: spread(&e[r.clone()]), range: r, kind: Kind::Cluster { rotated: false } });
            }
            i = j;
        }

        // the previous display
        let n_prev = self.prev[ch].len();
        let o: Option<DMatrix<f64>> = match input.overlaps_prev[ch] {
            Some(o) if self.started && n_prev > 0 => {
                assert_eq!(o.ncols(), n_prev, "overlaps_prev[{ch}] columns = rows shown by the previous update");
                assert!(o.nrows() >= n_shown, "overlaps_prev[{ch}] rows cover the shown orbitals");
                Some(o.rows(0, n_shown).into_owned())
            }
            _ => None,
        };

        // continuity Procrustes on the unaligned clusters (§5 item 1)
        if let Some(o) = &o {
            for b in blocks.iter_mut() {
                let d = b.range.len();
                if !matches!(b.kind, Kind::Cluster { .. }) || !(2..=MAX_ROTATED_CLUSTER).contains(&d) { continue; }
                let weight = |j: usize| b.range.clone().map(|i| o[(i, j)] * o[(i, j)]).sum::<f64>();
                let mut cols: Vec<usize> = (0..n_prev).collect();
                cols.sort_by(|&a, &c| weight(c).total_cmp(&weight(a)).then(a.cmp(&c)));
                cols.truncate(d);
                cols.sort_unstable();
                let a = DMatrix::from_fn(d, d, |i, k| if k < cols.len() { o[(b.range.start + i, cols[k])] } else { 0.0 });
                b.u = procrustes_continuity(&a);
                b.kind = Kind::Cluster { rotated: true };
            }
        }

        // rotated overlaps O' = Bᵀ O, and matching on |O'| (§5 item 2)
        let o_rot = o.map(|o| {
            let mut r = o.clone();
            for b in &blocks {
                let d = b.range.len();
                let rotated = b.u.transpose() * o.rows(b.range.start, d);
                r.rows_mut(b.range.start, d).copy_from(&rotated);
            }
            r
        });
        let matched: Vec<Option<usize>> = match &o_rot {
            Some(or) => {
                let w = or.map(|v| if v.abs() >= MATCH_MIN { v.abs() } else { 0.0 });
                hungarian_max(&w).into_iter().enumerate().map(|(i, j)| j.filter(|&j| w[(i, j)] > 0.0)).collect()
            }
            None => vec![None; n_shown],
        };

        // signs (§4 item 4 built in; §5 items 2–3)
        for b in blocks.iter_mut() {
            let d = b.range.len();
            for k in 0..d {
                let i = b.range.start + k;
                let s = match (b.kind, matched[i]) {
                    (Kind::Shell { l, .. }, _) if l >= 1 => 1.0,
                    (_, Some(j)) => if o_rot.as_ref().expect("matched ⇒ overlaps")[(i, j)] > 0.0 { 1.0 } else { -1.0 },
                    (_, None) => {
                        let mut f = [0.0; 9];
                        for jj in 0..d {
                            for (q, fq) in f.iter_mut().enumerate() { *fq += b.u[(jj, k)] * input.first_sign[ch][b.range.start + jj][q]; }
                        }
                        first_appearance_sign(&f).0
                    }
                };
                if s < 0.0 { for jj in 0..d { b.u[(jj, k)] = -b.u[(jj, k)]; } }
            }
        }

        let mut energies = vec![0.0; n_shown];
        for b in &blocks {
            for k in 0..b.range.len() {
                energies[b.range.start + k] = (0..b.range.len()).map(|j| b.u[(j, k)] * b.u[(j, k)] * e[b.range.start + j]).sum();
            }
        }
        Plan { n_shown, hidden, blocks, matched, energies }
    }
}

/// §4: shells, node counts and labels of a lone atom, or the reason it is `#k`.
fn analyse_lone_atom(input: &UpdateInput) -> (LoneAtom, Option<AtomLabels>) {
    let Some(ms) = input.moments else { return (LoneAtom::NotLoneAtom, None) };
    let n_occ = input.n_occ;
    for ch in 0..2 { assert!(ms[ch].len() >= n_occ[ch], "moments[{ch}] cover the occupied orbitals"); }
    let occ_ms = [&ms[0][..n_occ[0]], &ms[1][..n_occ[1]]];
    let occ_e = [&input.evals[0][..n_occ[0]], &input.evals[1][..n_occ[1]]];
    let shells = match lone_atom_shells_why(occ_ms, occ_e) {
        Ok(s) => s,
        Err(why) => return (LoneAtom::Unlabelled(why), None),
    };
    let Some(radial) = input.radial else { return (LoneAtom::Unlabelled(Unlabelled::NodesUnresolved), None) };
    let mut nodes = [vec![None; n_occ[0]], vec![None; n_occ[1]]];
    for ch in 0..2 {
        for sh in &shells[ch] {
            for i in sh.range.clone() {
                nodes[ch][i] = radial[ch].get(i).and_then(|p| profile_nodes(p, &ms[ch][i], sh.l));
            }
        }
    }
    match label_shells(&shells, [&nodes[0], &nodes[1]], occ_e, input.n_core) {
        Ok(ls) => (LoneAtom::Labelled { shell_order_inverted: ls.shell_order_inverted }, Some((shells, ls))),
        Err(why) => (LoneAtom::Unlabelled(why), None),
    }
}

/// The displayed orbitals of a channel: ψ̃_{start+k} = Σⱼ ψ_{start+j} U_jk over `out.rotations`
/// (the shown rows, index order). `orbitals` are the channel's raw computed orbitals.
pub fn apply_rotation(orbitals: &[Vec<f64>], out: &UpdateOutput, spin: Spin) -> Vec<Vec<f64>> {
    let mut res = Vec::new();
    for (r, u) in &out.rotations[spin.index()] {
        for k in 0..r.len() {
            let mut f = vec![0.0; orbitals[r.start].len()];
            for j in 0..r.len() {
                let w = u[(j, k)];
                if w == 0.0 { continue; }
                for (o, v) in f.iter_mut().zip(&orbitals[r.start + j]) { *o += w * v; }
            }
            res.push(f);
        }
    }
    res
}

// ───────────────────────────── §8 the display eigensolve ─────────────────────────────

/// LOBPCG cap of the display eigensolve: the SCF's own (`qm::scf`'s channel solve and
/// `qm_gpu::scf`, 200 Rayleigh–Ritz steps).
pub const DISPLAY_EIGEN_MAX_ITER: usize = 200;

/// A guess vector within this (relative norm after projection) of the span of the earlier ones is
/// replaced by fresh noise before the CPU solve ([`unoccupied_cpu`]), so `lobpcg`'s independence
/// assertion cannot fire on a display path (the GPU's LOBPCG re-seeds such vectors itself).
const GUESS_DROP: f64 = 1e-6;

/// One display eigensolve's unoccupied orbitals, per channel `[↑, ↓]` (§8). Per vector, ascending
/// in ε: the eigenvalue (the Rayleigh quotient of the shown, projected vector; Ha), its residual
/// ‖(H^out − ε)φ‖ (Ha, ‖φ‖ = 1), `converged` = residual ≤ `tol`, and the face density of the vector's
/// τ-set ([`face_density_set`], the same value for every member; e/Bohr³ for one electron). `hidden`: §6's hidden
/// count with the SCF's occupied eigenvalues in front. `iterations`, `at_cap`: LOBPCG's (at the cap
/// with its max residual still ≥ tol). `occupied_error`: max |ε_occ(display) − ε_occ(SCF)| over
/// the channel's occupied pairs — the runtime guard. `rejected` says why nothing is returned (that
/// guard; on the GPU also broken numerics); every per-vector list is then empty.
#[derive(Clone, Debug, PartialEq)]
pub struct Unoccupied {
    pub evals: [Vec<f64>; 2],
    pub residuals: [Vec<f64>; 2],
    pub converged: [Vec<bool>; 2],
    pub face_density: [Vec<f64>; 2],
    pub hidden: [usize; 2],
    pub tol: f64,
    pub iterations: [usize; 2],
    pub at_cap: [bool; 2],
    pub occupied_error: [f64; 2],
    pub rejected: Option<String>,
}

impl Unoccupied {
    /// No vectors, at tolerance `tol`.
    pub fn empty(tol: f64) -> Unoccupied {
        Unoccupied {
            evals: [Vec::new(), Vec::new()], residuals: [Vec::new(), Vec::new()], converged: [Vec::new(), Vec::new()],
            face_density: [Vec::new(), Vec::new()], hidden: [0; 2], tol, iterations: [0; 2], at_cap: [false; 2],
            occupied_error: [0.0; 2], rejected: None,
        }
    }

    /// Drops every vector's data and records why (§8: nothing of this solve is shown).
    pub fn reject(&mut self, why: String) {
        for ch in 0..2 {
            self.evals[ch].clear();
            self.residuals[ch].clear();
            self.converged[ch].clear();
            self.face_density[ch].clear();
            self.hidden[ch] = 0;
        }
        self.rejected = Some(why);
    }

    /// Vectors per channel.
    pub fn count(&self) -> [usize; 2] { [self.evals[0].len(), self.evals[1].len()] }
}

/// The occupied-agreement guard's number (§8): max_i |display_i − scf_i| over the SCF's occupied
/// eigenvalues (both ascending); ∞ when the display solve returned fewer.
pub fn occupied_error(display: &[f64], scf: &[f64]) -> f64 {
    if display.len() < scf.len() { return f64::INFINITY; }
    scf.iter().zip(display).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)
}

/// The guard's message when a channel's [`occupied_error`] exceeds τ, else None.
pub fn occupied_disagreement(error: [f64; 2], tau: f64) -> Option<String> {
    (0..2).find(|&ch| !(error[ch] <= tau)).map(|ch| {
        format!("the display eigensolve's occupied eigenvalues disagree with the SCF's by {:.2e} Ha ({}) > τ = {tau:.0e}: unoccupied orbitals not shown",
                error[ch], if ch == 0 { "↑" } else { "↓" })
    })
}

/// v_σ = V_nuc + V_H[ρ_out] + v_xc,σ[ρ_out↑, ρ_out↓] of a converged state solved at `nuclei`, both
/// channels (§8): `v_h` is the state's own V_H[ρ_out]; v_xc is LSDA PW92 of the output densities.
/// Under an ionic setup (M7, `scf` "Ionic species") `v_nuc` is already the state's V_loc and v_xc is
/// of (ρ_σ + ½ρ_pc) when an ion has a partial core (`scf::xc_with_core`, the SCF's formula).
pub fn output_potentials(state: &KsState, nuclei: &[Nucleus]) -> [Vec<f64>; 2] {
    let rho_pc = state.ionic.as_deref().filter(|s| has_partial_core(nuclei, s)).map(|s| partial_core_field(&state.grid, nuclei, s));
    let xc = xc_with_core(&state.rho_up, &state.rho_dn, rho_pc.as_deref());
    let v = |x: &[f64]| (0..state.grid.len()).map(|i| state.v_nuc[i] + state.v_h[i] + x[i]).collect::<Vec<f64>>();
    [v(&xc.v_up), v(&xc.v_dn)]
}

/// V_NL of a state solved at `nuclei` under an ionic setup with projectors (the SCF's
/// `NonlocalOp`); `None` for an all-electron state or a setup without projectors.
pub fn output_nonlocal(state: &KsState, nuclei: &[Nucleus]) -> Option<NonlocalOp> {
    let op = NonlocalOp::build(&state.grid, nuclei, state.ionic.as_deref()?);
    (!op.is_empty()).then_some(op)
}

/// x ← x − c·y.
fn sub_scaled(x: &mut [f64], c: f64, y: &[f64]) {
    for (a, b) in x.iter_mut().zip(y) { *a -= c * b; }
}

/// MGS on the guess in place (unit norms); a vector within `GUESS_DROP` of the span of the earlier
/// ones is replaced by deterministic noise under the nuclear envelope (`initial_orbitals`, seeds
/// from `seed` on) and re-projected.
fn independent_guess(grid: &Grid, x: &mut [Vec<f64>], nuclei: &[Nucleus], mut seed: u64) {
    for i in 0..x.len() {
        loop {
            let n0 = grid.dot(&x[i], &x[i]).sqrt();
            let (done, rest) = x.split_at_mut(i);
            let v = &mut rest[0];
            for q in done.iter() { let c = grid.dot(q, v); sub_scaled(v, c, q); }
            let n1 = grid.dot(v, v).sqrt();
            if n1 > GUESS_DROP * n0 && n1 > 0.0 {
                for a in v.iter_mut() { *a /= n1; }
                break;
            }
            *v = initial_orbitals(grid, nuclei, 1, seed).remove(0);
            seed += 1;
        }
    }
}

/// §8 on the CPU in f64 (the reference of `qm_gpu::orbitals::OrbitalView::unoccupied`): the lowest
/// n_σ + m + `UNOCC_GUARD` eigenpairs of H_σ^out (under an ionic setup the SCF's full operator:
/// [`output_potentials`] with the partial core, plus V_NL from [`output_nonlocal`]) per channel by LOBPCG at `settings.eig_tol`
/// (cap [`DISPLAY_EIGEN_MAX_ITER`]), from [copies of the occupied orbitals, `previous`' vectors,
/// `initial_orbitals(…, seed 3 ↑ / 4 ↓)` for the rest]; the occupied pairs checked against the
/// SCF's (τ(`settings.tol`)) and discarded; the unoccupied ones projected onto the complement of
/// the SCF's occupied orbitals, re-orthonormalised (MGS, twice), sorted, and re-measured.
/// `nuclei` are the geometry `state` was solved at (the guess's envelope). Returns the result and
/// the m + `UNOCC_GUARD` vectors per channel (the next call's `previous`; empty when rejected),
/// or `None` when `should_abort` (polled before each channel and once per LOBPCG iteration) said
/// so (§8, "Cancellation"). Reads `state` only.
pub fn unoccupied_cpu(state: &KsState, nuclei: &[Nucleus], m: usize, settings: &ScfSettings, previous: Option<&[Vec<Vec<f64>>; 2]>,
                      should_abort: &mut dyn FnMut() -> bool) -> Option<(Unoccupied, [Vec<Vec<f64>>; 2])> {
    let grid = &state.grid;
    let tol = settings.eig_tol;
    let tau_scf = tau(settings.tol);
    let k = m + UNOCC_GUARD;
    // H_σ^out is the SCF's own operator: V_loc, the partial core and V_NL under an ionic setup
    let pots = output_potentials(state, nuclei);
    let nl = output_nonlocal(state, nuclei);
    let mut out = Unoccupied::empty(tol);
    let mut vecs: [Vec<Vec<f64>>; 2] = [Vec::new(), Vec::new()];
    for ch in 0..2 {
        if should_abort() {
            return None;
        }
        let (occ, ev_scf) = if ch == 0 { (&state.orbitals_up, &state.evals_up) } else { (&state.orbitals_dn, &state.evals_dn) };
        let n = occ.len();
        let prev: &[Vec<f64>] = previous.map_or(&[], |p| &p[ch][..]);
        let kept = prev.len().min(k);
        let fresh = initial_orbitals(grid, nuclei, k, 3 + ch as u64);
        let mut x: Vec<Vec<f64>> = occ.iter().cloned().chain(prev[..kept].iter().cloned()).chain(fresh.into_iter().skip(kept)).collect();
        independent_guess(grid, &mut x, nuclei, 1000 + 100 * ch as u64);
        let op: Box<dyn Operator + '_> = match &nl {
            None => Box::new(LocalHamiltonian { grid, v: &pots[ch] }),
            Some(nl) => Box::new(PseudoHamiltonian { grid, v: &pots[ch], nl }),
        };
        let op = op.as_ref();
        let pc = jacobi_precond(grid, &pots[ch], ev_scf.first().copied().unwrap_or(-1.0));
        let mut ev = vec![0.0; n + k];
        let stats = lobpcg(grid, op, &mut x, &mut ev, Some(&pc), tol, DISPLAY_EIGEN_MAX_ITER, Some(&mut *should_abort));
        if stats.aborted {
            return None;
        }
        out.iterations[ch] = stats.iterations;
        out.at_cap[ch] = !(stats.max_residual < tol); // `lobpcg` stops on tol or on the cap
        let mut order: Vec<usize> = (0..n + k).collect();
        order.sort_by(|&a, &b| ev[a].total_cmp(&ev[b]));
        let sorted: Vec<f64> = order.iter().map(|&i| ev[i]).collect();
        out.occupied_error[ch] = occupied_error(&sorted[..n], ev_scf);
        // the occupied pairs are discarded; the unoccupied ones projected onto the complement of
        // the SCF's occupied orbitals and re-orthonormalised in energy order (twice: f64 MGS)
        let mut phis: Vec<Vec<f64>> = order[n..].iter().map(|&i| std::mem::take(&mut x[i])).collect();
        for _ in 0..2 {
            for i in 0..phis.len() {
                let (done, rest) = phis.split_at_mut(i);
                let v = &mut rest[0];
                for psi in occ.iter().chain(done.iter()) { let c = grid.dot(psi, v); sub_scaled(v, c, psi); }
                let nv = grid.dot(v, v).sqrt();
                for a in v.iter_mut() { *a /= nv; }
            }
        }
        // re-measure ε and the residual on the shown vectors, then sort (ε moved by ~|⟨ψ|φ⟩|²)
        let mut hphi = vec![0.0; grid.len()];
        let mut rows: Vec<(f64, f64, Vec<f64>)> = phis.into_iter().map(|phi| {
            op.apply(&phi, &mut hphi);
            let e = grid.dot(&phi, &hphi);
            let r: Vec<f64> = hphi.iter().zip(&phi).map(|(h, p)| h - e * p).collect();
            (e, grid.dot(&r, &r).sqrt(), phi)
        }).collect();
        rows.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (e, res, phi) in rows {
            out.evals[ch].push(e);
            out.residuals[ch].push(res);
            out.converged[ch].push(res <= tol);
            vecs[ch].push(phi);
        }
        // the face density of each τ-set (§8, review M1), the same value for its members
        out.face_density[ch] = vec![0.0; vecs[ch].len()];
        for set in clusters(&out.evals[ch], tau_scf) {
            let members: Vec<&[f64]> = vecs[ch][set.clone()].iter().map(|v| v.as_slice()).collect();
            let f = face_density_set(grid, &members);
            for i in set { out.face_density[ch][i] = f; }
        }
        let all: Vec<f64> = ev_scf.iter().chain(&out.evals[ch]).copied().collect();
        out.hidden[ch] = block_edge(&all, n, n + k, tau_scf).1;
    }
    if let Some(why) = occupied_disagreement(out.occupied_error, tau_scf) {
        out.reject(why);
        vecs = [Vec::new(), Vec::new()];
    }
    Some((out, vecs))
}

#[cfg(test)]
mod tests;
