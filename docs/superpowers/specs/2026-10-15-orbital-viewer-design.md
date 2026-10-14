# Orbital viewer: design

Status: physics-reviewed 2026-10-15 (verdict "needs rework, only in one place": 4 MUST, 10 SHOULD, NOTEs; all folded in, §14 lists each change; the review text is `docs/superpowers/notes/2026-10-15-orbital-viewer-spec-review.md`). Written 2026-10-14, dated for its build window. Child of `2026-10-12-atomic-sandbox-design.md` §8 ("Toggles: isosurface at chosen density, per-orbital view, spin density (↑ − ↓)"), the one display item of v1 that was never built. **Built after M6 (tools) merges**: it adds a dock tab, a View-menu section and Ctrl-K entries to the chrome that M6 re-homes, and it uses M6's `providers` crate and `app/src/worker/` split. It is **independent of the user-data feature** (settings profiles, model library): nothing here is persisted per user except the panel state the viewer already keeps. Its §8 (core reconstruction) is built with or after M7 (pseudopotentials); everything else works on today's all-electron solver.

Plan: `docs/superpowers/plans/2026-10-15-orbital-viewer.md` (Task 0 is a measurement spike whose numbers Tasks 1–5 consume).

## 1. What the user asked, and the answer this design is bound by

After a textbook picture of s spheres and p lobes around a nucleus: *"Are we able to have a toggle to visually see those fields? right now there's just a big glow around the atom, I want to see those inner spheres n stuff too, I imagine this relates to their spin as well? you would see them rotate?"*

The answer given (binding on this design):

- **Spin is intrinsic** (↑ or ↓ along an axis), not a rotation of anything. The solver carries it as two channels; the view can show the difference ρ↑ − ρ↓.
- **Orbitals do not rotate.** Born–Oppenheimer Kohn–Sham orbitals are stationary states; they move and deform only because the nuclei move.
- **A lone atom's degenerate set** (the three p, later the five d) has no preferred orientation: any rotation of the set is an equally valid eigenbasis, and the eigensolver may hand back a different one each step. A display rule rotates each degenerate subspace of a lone atom to line up with the cell axes. In molecules the bonds fix the orientation and no axis rule is applied (§5.4 says what keeps them steady instead).
- **After M7 the core shells are not simulated**; the view reconstructs them for display from the pseudopotential generator's own all-electron atom and labels them that way (§8).

## 2. What is on screen, honestly (rule 7)

Every mode's tooltip and the dock tab's header carry these facts. The exact wording is pinned in the plan (Task 7).

1. **A Kohn–Sham orbital is an auxiliary object of DFT,** the eigenfunction of a fictitious non-interacting system with the same density. It is not a measured electron wavefunction. **The total density is the physical observable.** Orbitals are how the solver builds the density.
2. **Eigenvalues.** ε_i are KS eigenvalues. Only the highest occupied one has a physical meaning (−ε_HOMO ≈ the ionisation energy in exact DFT). In LSDA it is far too shallow because of self-interaction: the H atom's ε_1s↑ sits well above its −0.5 Ha total energy. The NIST LSD eigenvalue goes into `references.md` at task time, read off the source; this spec quotes no digit it has not checked. The other eigenvalues are not excitation energies.
3. **Unoccupied orbitals** are eigenstates of the converged ground-state potential, computed for display (§4.5). LSDA's potential decays too fast, so many of them are not bound. In a finite Dirichlet box an unbound state becomes a **state of the box**, flagged as one (§5.7). **The KS gap ε_LUMO − ε_HOMO is not the fundamental gap** (IP − EA): it misses the derivative discontinuity, and in LSDA it is typically much smaller. Unoccupied rows say so. They never get atomic names (§5.3).
4. **Integer occupations break symmetry.** Integer aufbau occupation of a partly filled degenerate shell (O's 2p⁴, its ↓ channel holding one of three p) gives a non-spherical density. The degenerate set then splits, and the occupied member's direction is chosen by the SCF's history, not by physics. The potential is then non-central for *both* channels, so the whole atom is labelled by energy, not as "2p" (§5.3 item 5), and the tooltip says why. Fractional, spherical occupations arrive with Fermi smearing (roadmap).
5. **Smeared nuclei.** Today every nucleus is a Gaussian of width σ = h. A 1s orbital is wider and shallower than the physical one when Zσ ≥ `SMEARED_ZSIGMA` = 0.5. The cut-off is derived, not per element: E(Z, σ) = Z² E(1, Zσ), and the one-electron 1s energy at Zσ = 0.5 is E(1, 0.5) = −0.3751 Ha against −0.5 for a point nucleus, 25 % shallower (pseudopotential spec §10 M6, the 1-D Numerov values; 11 % at Zσ = 0.25). At σ = h = 0.25 that tags every Z ≥ 2. The row says "1s (smeared nucleus: wider than the physical 1s)". The same smearing makes a lone heavy atom's eigenvalues depend on where it sits relative to the grid (the egg-box). Task 0 measured the 1s moving ≈ 1 mHa (O) and 2 mHa (Ne) between sites at h = 0.25 (the reviewer's model said tenths of a Ha), and the p shell splitting by up to 0.7 mHa (§5.1). The tags show the measured spread rather than hide it. The smearing also **inverts 2s and 2p** in the heavier atoms (Task 0: Ne 2p −0.845, 2s −0.298 Ha at h = 0.25; the O singlet is 1s² 2p⁶): it removes the cusp that the penetrating 2s relies on (`qm::orbitals` module doc §4). The labels still read 1s, 2p, 2s (they come from character and radial nodes, §5.3), and the list says "shell order inverted by the smeared nucleus — not physical (needs a finer grid or pseudopotentials)" (`LoneAtom::Labelled { shell_order_inverted }`).
6. **Phase colours** show the sign of ψ. The overall sign of an orbital is arbitrary: a display rule fixes it so the colours do not flicker (§5.4). Only the relative sign of lobes means anything.
7. **Rotated sets** are labelled "rotated within a degenerate set for display", with the measured spread s of the set's eigenvalues. A lone atom's shell is split by the grid, not by physics (§5.3): its tag reads "split by the grid by s mHa". Each member shown is an eigenfunction to within s/2 (the variance bound of §5.1).

## 3. The modes (View ▸ Density ▸ show)

A new `show` selector in View ▸ Density, `Density | Orbitals | All orbitals coloured | Spin density`, with Ctrl-K search entries of the same names. `Density` is today's view, unchanged and the default. The other three need Kohn–Sham orbitals, so they need a DFT step (§7). The existing `render as volume | isosurface` control applies per mode as listed.

| Mode | Field written to the display texture | Render | Scale (what 1.0 means) |
|---|---|---|---|
| Density | ρ = ρ↑ + ρ↓ (today) | volume / isosurface (today) | max ρ (today) |
| Orbitals, *phase* | ψ̃ (signed; the displayed orbital after the rotation and sign of §5) | **± isosurface**: two surfaces at ±iso·scale, coloured by sign; or **signed volume** | max\|ψ\| over the orbital's degenerate cluster (§5.6) |
| Orbitals, *\|ψ\|²* | ψ̃² | today's volume / isosurface with today's ramp | (max\|ψ\| over the cluster)² |
| All orbitals coloured | RGBA: rgb = Σ_i f_i c_i s_i, a = Σ_i f_i s_i (§5.6) | **additive** (emission only: inner shells shine through outer ones, the default) or **weighted** (emission–absorption: the front shell hides what is behind it) | per-orbital peak (default) or the physical ρ_max (§5.6) |
| Spin density | m = ρ↑ − ρ↓ (signed) | ± isosurface or signed volume, coloured by sign | max(max\|m\|, `SPIN_FLOOR`·max ρ) (§5.5) |

Colours: phase + is red (0.90, 0.30, 0.20) and − is blue (0.20, 0.45, 0.90), the textbook convention. Spin ↑ is amber (0.95, 0.65, 0.15) and ↓ is teal (0.15, 0.70, 0.65), a different hue pair so phase and spin cannot be confused. A legend chip in the view's corner always names the pair in use. All orbitals coloured uses a fixed qualitative palette of 12 colours (`ORBITAL_PALETTE`); slots are assigned by §5.4. Iso defaults are 0.15 of scale for phase and 0.1 for spin. Both have sliders.

**Orbitals** mode opens an **Orbitals** dock tab. It holds the list (§3.1) and the per-mode controls: phase / |ψ|², "show unoccupied: m" (0–8, default 4) and the iso slider. **All orbitals coloured** uses the same tab: every row shows its colour swatch, and a checkbox per row includes or excludes it (default: all occupied orbitals). **Spin density** shows a readout block in that tab with n↑, n↓, ∫m = n↑ − n↓, ∫|m| and max|m|/max ρ.

### 3.1 The list

One row per orbital, both spin channels, grouped by channel and in energy order:

`label   spin   ε (Ha)   ε (eV)   occ   tags`

- **label**: the n l m name for a lone atom when it can be derived (§5.3), e.g. `2p_x`. Otherwise `#k`, the 1-based index in its channel by energy. Molecular orbitals never get atomic-style names. A molecular orbital is "#3 ↑ ε = −0.512 Ha", not "2p_x".
- **ε**: Ha, with eV beside it converted only at the UI boundary through `units` (rule 3).
- **occ**: 1 or 0 today (integer aufbau). With smearing later, f_i.
- **tags**: `HOMO` and `LUMO` (over both channels), `degenerate ×d (spread …)`, `rotated for display`, `split by the grid by … mHa` (lone atom), `box state`, `not converged (residual …)` (unoccupied only), `smeared 1s`, and after M7 `core: reconstructed for display` and `valence: pseudo-orbital`.

Picking a row selects that orbital **by track** (§5.4), not by index. The selection follows the orbital as levels reorder while the atoms move.

## 4. What the solver exposes, and the data path

### 4.1 Where the orbitals are today

- **GPU** (`qm_gpu::scf::GpuState`): `orbitals_up`/`orbitals_dn: Vec<Field>`, one padded f32 storage buffer per occupied orbital. Normalisation is L² on the grid, Σ ψ² h³ = 1 (`qm::Grid::dot`; f32 orthonormal to ~1e-7). `evals_up`/`evals_dn` are f64. `rho_up`, `rho_dn` and `rho` are the **output** densities. They are owned by the `GpuSolver`, which overwrites them in place on the next warm solve. The block holds exactly n_σ orbitals per channel: **no unoccupied orbitals are computed** (§4.5).
- **CPU** (`qm::KsState`): the same quantities in f64 `Vec`s.
- The density display copies `rho` into an r32float `Volume3D` on the GPU (`GpuSolver::copy_density_to_volume`: `unpad` into a row-padded buffer, then `copy_buffer_to_texture`). The texture comes from a `TexturePool` that never overwrites a texture someone still holds. The CPU path ships f32 values and the viewer uploads them.

### 4.2 One display update

Run by the provider **after** the step's energy and forces are final, on the converged state, and only when an orbital mode is on:

1. **One readback** of small f64 numbers, through the existing `Reducer` (f32 workgroup partials, Neumaier f64 finish; rule 4):
   - the signed overlaps O = ⟨ψ_i | ψ̃_j^prev⟩ between the current orbitals and the previous update's *displayed* orbitals, per channel, plus the cross-channel overlaps ⟨ψ_i↑ | ψ_j↓⟩;
   - max|ψ_i| per orbital. The `Reducer` has `Max` but no max|·|, so this is a new one-pass `Reduction::AbsMax`, or max ψ and max −ψ as two reductions in the same readback;
   - for a lone atom, the moments of §5.2 per orbital (16 b and 16 c, both linear in ψ) and the radial-shell profile of §5.3 item 4 per occupied orbital (16 × N_bins, bins of h/2 about the nucleus; the bin point counts n_k are geometry);
   - the windowed first-appearance functionals of §5.4 item 3 (9 per orbital) and, for unoccupied orbitals, the outer-shell weight of §5.7;
   - in spin mode, ∫m, ∫|m| and max|m|.
2. **CPU f64 small problems** (`qm::orbitals`): lone-atom shells, clusters, alignment or continuity rotations U (d × d per set), signs, track matching, labels, colour slots. Display-only. Nothing goes back into the SCF.
3. **GPU writes, no wait.** Lincombs apply U and the signs into display-owned fields. They become the next update's "previous" set and the source of the products. Then the mode's product goes into a display texture taken from the pool: the selected orbital (`unpad` + `copy_buffer_to_texture` into an r32float `Volume3D`, as for the density), the spin density (a lincomb ρ↑ − ρ↓ into scratch, then the same copy), or the colour volume (a compute kernel accumulating the weighted set straight into an rgba16float storage texture).

On the GPU path no orbital or field data ever crosses to the CPU; the one readback carries scalars only, O(K² + 42K) of them, where K = n↑ + n↓ plus the unoccupied orbitals shown. The **CPU backend** runs the same steps in f64 with the `qm::orbitals` reference functions, ships the product as f32 values (or RGBA f32, converted to f16 at upload), and the viewer uploads it as it uploads the CPU density today. The CPU products are also the reference the GPU kernels are tested against.

### 4.3 Types and the force boundary (rule 5)

The data types live in `sim::provider` next to `Display`. `sim` carries them opaquely, as it does the density:

- `OrbitalRequest`: `Off | List | Orbital { spin, track, style: Phase|Density } | AllColoured { include, scale: PerOrbital|Physical, blend: Additive|Weighted } | SpinDensity`, plus `unoccupied: u8`.
- `OrbitalDisplay`: the geometry it was solved at, `cell`/`origin`, the rows (`OrbitalInfo { spin, index, track, colour, energy, occupation, label, cluster, tags, residual }`), the product (`OrbitalField::{Gpu { volume, signed, scale } | GpuColour { volume, scale } | Cpu { values, signed, scale } | CpuColour { rgba, scale }}`), the spin summary and notes.
- `ForceProvider` gains three default methods, all no-ops for analytic providers: `set_orbital_view(&mut self, OrbitalRequest)`, `orbital_display_now(&mut self) -> Option<OrbitalDisplay>` (re-run §4.2 on the provider's current converged state with no solve; used when the view is switched on while paused), and the result field `ForceResult::orbitals: Option<OrbitalDisplay>`. `SideDisplay` gains `orbitals: Option<OrbitalDisplay>`. `DynSnapshot` carries the newest one.
- `providers::{CpuKs, GpuKs}` implement them.
- **Estimated mode has exactly one producer** (review SHOULD 12). If the teacher and the auditor each ran an `OrbitalView`, their tracks, slots and signs would disagree, and the viewer would alternate between them. So `SurrogateKs` forwards the request only to its **background side solver**, and only audits and display refreshes carry orbitals. Fallback steps on the stepping teacher carry none. A model step carries none. "Refresh cloud every N s" is the way to get regular orbital updates in estimated mode, and its tooltip says so.
- `qm` gains `qm::orbitals` (pure f64 numerics; no rendering or UI knowledge, rule 5). `qm-gpu` gains `qm_gpu::orbitals` (`OrbitalView`: the display-owned previous fields, the moments and colour kernels, one readback per update). `gpu` gains `ColourVolume3D` (rgba16float, filterable on every device).

### 4.4 Never touching the solve, and the update rate

- The display reads the solver's orbitals and densities. It writes only display-owned fields and textures. It runs after forces. It never changes a solver knob or its warm state. **Pinned:** 20 MD steps of H₂, with an update **forced at every step** (20 payloads asserted, so the rate limit cannot make the test vacuous), give bitwise-identical energies and forces (`==` on f64) with the view off and with it on, switching modes every step (plan Task 3). `solve_is_deterministic` makes the comparison meaningful.
- **Rate.** During play, at most one update per `ORBITAL_MIN_INTERVAL` = 100 ms of wall time. A step inside the interval carries no payload, and the viewer keeps (and crossfades) the last one. Paused, single-stepping, and the first step after the view is turned on always update. The unoccupied-orbital eigensolve (§4.5) has its own interval and budget.
- **Cost.** Per update without unoccupied orbitals: one readback, about 2K + 3 lincombs, one copy or colour dispatch. The plan measures it (Task 0 for the eigensolve, Task 7 for the update) at 49³, 57³ and 97³ and records it in the README. The expectation is a few ms at 97³ for K ≤ 16, against a 1–2 s warm SCF at that size. Numbers are reported, never asserted (M4 R3).
- **VRAM.** K + 2m display fields (padded, 4.6 MB each at 97³): the K + m previous displayed orbitals, plus the m (+ `UNOCC_GUARD`) unoccupied vectors kept between eigensolves as the next guess. Add one product texture pair (r32float 3.6 MB, or rgba16float 7.3 MB, × 2 for the crossfade) and, with unoccupied orbitals, the solver's `LobpcgWork` grown to n_σ + m + `UNOCC_GUARD` (+8 fields per vector). That is about 0.25 GB at 97³ with m = 4, on a 24 GB card. All of it is freed when the view is switched off or the grid changes.

### 4.5 Unoccupied orbitals (provisional ruling P1)

The SCF solves only the n_σ occupied states per channel, and enlarging its block would change the solver (rule 6). Unoccupied orbitals therefore come from a **display eigensolve**:

- **What it solves.** The lowest n_σ + m + `UNOCC_GUARD` eigenpairs (`UNOCC_GUARD` = 2) of H_σ = −½∇² + V_nuc + V_H[ρ_out] + v_xc,σ[ρ_out↑, ρ_out↓], the potential of the converged *output* density. No Poisson solve is needed: `GpuState` already holds v_h = V_H[ρ_out] and ρ_out,σ. This potential differs from the SCF's last V[ρ_in] only within the SCF tolerance.
- **Tolerance.** The same LOBPCG (`qm::eigen::lobpcg` / `qm_gpu::eigen::lobpcg_with_guess_tol`) runs at the SCF's **effective** tolerance, max(`ScfSettings::eig_tol`, `EIG_TOL_FLOOR_FACTOR` × `residual_floor_estimate`). That is ≈ 1.8e-6 at h = 0.25 (`qm-gpu/src/scf.rs`, the `eig_tol` of `run`). The raw `eig_tol` = 1e-6 sits below the f32 floor, so every display solve would end on the stall or the cap and every row would read "not converged" (review MUST 4). The CPU f64 solve uses `eig_tol` itself.
- **Guess.** A **copy** of the occupied orbitals, plus the previous unoccupied vectors, or `initial_orbitals` (seeds 3/4) for new ones.
- **Where it runs.** In display fields and in the solver's `LobpcgWork` scratch, which carries no state between calls (`np = 0` and HX recomputed at entry — verified in review), on the worker thread between steps.
- **Nothing is written to the solver's state.** Pinned by the bitwise test with m = 4 over 20 steps.
- **The block edge (review MUST 3).** The unoccupied spectrum of a cubic box is full of degenerate sets (sizes 1, 3, 6, …), so a block cut at m + G can slice one. The rows show only the clusters (τ, §5.1) that start among the first m unoccupied and do not contain the last computed vector, because only the last cluster can be cut. So the tab may show fewer than m, and it says "m requested, k shown: a set cut by the block edge is hidden". Unoccupied rows never get atomic names and never affect the occupied labels (§5.3).
- **Budget (review SHOULD 11).** During play the eigensolve runs at most once per `UNOCCUPIED_MIN_INTERVAL` = 1 s, and only while its own last wall time is ≤ `UNOCCUPIED_BUDGET` = 0.1 × the summed SCF wall of the steps since its previous run (≤ 10 % of the worker). Otherwise it runs only when paused, and the rows say "unoccupied from t = … (updated when paused at this grid size)". Task 0 measures its cost at 81³ and 97³; Task 4 measures it inside the worker.

The occupied eigenpairs it returns are discarded: the rows show the SCF's own orbitals and eigenvalues, so they agree with the readouts. The test checks that they agree to τ (§5.1). Each unoccupied row shows its residual. One not converged at the LOBPCG cap is tagged `not converged (residual …)` and still shown. The CPU backend runs the same solve in f64.

## 5. Display numerics (all f64 on the CPU from the readback; rule 4; display only)

### 5.1 Degenerate clusters by energy (molecules, unoccupied rows, tags)

In each channel, sorted eigenvalues are joined into a cluster while consecutive gaps are ≤ τ (single linkage). τ is

  **τ = max(`TAU_PER_TOL` · tol_scf · 1 Ha/e, `TAU_MIN`)**, with `TAU_PER_TOL` = 100 and `TAU_MIN` = 1e-4 Ha (provisional ruling P2).

That is 1 mHa at the display backends' `INTERACTIVE_TOL` 1e-5 and 1e-4 Ha at `SAMPLE_TOL` 1e-6. Why it is tied to tol: at SCF tolerance tol (∫|Δρ| in electrons), the potential each member of a degenerate set sees is inconsistent by δV ≈ δV_H + f_xc δρ. That shifts the eigenvalues by |⟨ψ|δV|ψ⟩|, which is O(tol) Ha per electron of Δρ when the density error sits on the atom. The factor 100 covers the anisotropic part.

**τ is not used to find a lone atom's shells** (review MUST 1). The cubic stencil cannot split p: Σ_a s(k_a h) is O_h-invariant and p spans an irreducible three-dimensional representation of O_h. (It does split d into e_g + t_2g, and f.) What splits p is the **position-dependent sampling of the σ = h smeared nucleus** (the egg-box). In the reviewer's one-electron model at σ = h = 0.25, the spread is ≤ 5e-8 Ha on a cube-symmetric site, but 2.5e-3 Ha (Z = 8) and 1.3e-2 Ha (Z = 10) at an off-grid point, i.e. 10–100 × τ. A moving lone atom of Z ≳ 6 would cross τ every few steps and its labels would flicker. Lone atoms are therefore grouped by angular character (§5.3). Task 0 measures the real LSDA spread. No per-Z τ exists (rule 1).

Single linkage allows a cluster's spread s (max ε − min ε) to reach (d − 1)τ, not τ, and the tag shows the measured s. For a combination ψ = Σ_j c_j φ_j of eigenvectors with eigenvalues ε_j, the residual at the Rayleigh quotient is ‖(H − ε̄)ψ‖ = (Σ_j c_j² (ε_j − ε̄)²)^{1/2}, a standard deviation of values in an interval of width s. So it is **≤ s/2** (Popoviciu's variance bound). That bound is what §2 item 7 states. A cluster larger than `MAX_ROTATED_CLUSTER` = 7 is not rotated; it gets continuity only (§5.4), because it is a run of box states.

### 5.2 Moments (lone atom only; R the nucleus)

For each orbital ψ and l ≤ 3 (16 (l, m) pairs), both **linear in ψ**:

- b_lm = ⟨ψ | S_lm(r − R)⟩. S_lm are the real solid harmonics already public in `qm::poisson` (Racah normalisation, `L_MAX` = 6, used by `qm-gpu` too), reused and not re-derived. They are polynomials and exactly covariant under rotations about R, so a rotation within a set acts on b_l· by the real Wigner matrix. **Used for the Procrustes alignment.**
- c_lm = ⟨ψ | Y_lm(r̂)⟩, with Y_lm the unit real spherical harmonics, 0 at r = 0 for l > 0. **Used for angular character**, without the r^l weighting that would let the diffuse tail dominate. Per orbital the fraction is W_l,i = Σ_m c_lm,i² / Σ_{l'≤3} Σ_m c_l'm,i². For a set C it is W_l(C) = Σ_{i∈C} Σ_m c_lm,i² / Σ_{i∈C} Σ_{l'≤3} Σ_m c_l'm,i². Because c is linear, the set sum is the squared Frobenius norm of a block that an orthogonal mixing of C multiplies by an orthogonal matrix. So W_l(C) is **exactly invariant** under any rotation within C, and exactly 1 for an ideal shell. (The first draft's |ψ|-weighted moment was not invariant: the reviewer measured W_1 = 0.939 for an ideal p set, a 0.04 margin over the threshold. Review SHOULD 5.)

The GPU computes all 32 per orbital in one dispatch (kernel `orbital_moments`, S_lm and Y_lm evaluated in-kernel; no harmonic fields are stored).

### 5.3 Lone atom: shells, alignment and labels (the binding rule)

A **lone atom** is a scene with exactly one nucleus. Only its **occupied** orbitals are classified. Unoccupied rows are always `#k` with continuity (§5.4), whatever m is (review MUST 3).

1. **Character.** Each occupied orbital i has a dominant l_i when W_l,i ≥ `L_CHAR_MIN` = 0.9 for one l ≤ 3. Otherwise it has none.
2. **Shells by character, not by gap** (review MUST 1). In each channel, walk the occupied orbitals in energy order. Consecutive orbitals with the same dominant l form a run. A run of length k(2l + 1) is cut, in energy order, into k shells of 2l + 1. Each shell must pass the set tests: W_l(shell) ≥ `L_CHAR_MIN`, and σ_min/σ_max ≥ `ALIGN_RANK_MIN` = 0.5 for the (2l+1) × (2l+1) matrix M = (b_lm,j). The rank test says the shell spans every m of its l: for an ideal shell M = c × orthogonal and the ratio is 1. The shell's eigenvalue spread s is recorded, however large: a lone atom's shell is degenerate in physics and split only by the grid.
3. **Rotation** (orthogonal Procrustes). The rotated set is ψ'_k = Σ_j ψ_j U_jk, k ↔ m_k. Maximise Σ_k ⟨ψ'_k | S_l m_k⟩ = tr(M U). With M = W Σ Vᵀ, U = V Wᵀ. The diagonal of MU = W Σ Wᵀ is then non-negative, so the **sign rule is built in**: every aligned member has ⟨ψ'_k | S_l m_k⟩ ≥ 0 (p_x's positive lobe on +x). The rotation is recomputed every update. It is stateless, so it is deterministic and cannot drift. For a split shell the members shown are eigenfunctions to within s/2 (§5.1), and the tag says "rotated for display; split by the grid by s mHa".
4. **Labels: angular character and radial node count, never energy order** (controller ruling 1, Task 1). **n = l + 1 + N_core,l + ν**, with ν the radial node count, counted at run time on radial shells of width `RADIAL_BIN_H`·h = h/2 about the nucleus: the bin amplitude R̄_k = Σ_m ĉ_m a_k,lm / n_k (a_k,lm = Σ_{p∈k} ψ Y_lm h³, ĉ the orbital's own c_l direction), ν = the sign changes among bins with |R̄_k| ≥ `NODE_FLOOR`·max|R̄|. N_core,l = 0 today (§8 for M7). The Sturm–Liouville oscillation theorem (the k-th eigenstate of fixed l in a local central potential has k − 1 radial nodes) is the consistency check: per channel and l, the shells in energy order must read ν = 0, 1, 2, … and every member of a shell the same ν, else the atom is `#k` (`NodesUnresolved`). The rule is immune to the smeared nucleus's 2s/2p inversion (§2 item 5). It is feasible because the smearing moves the inner s node out: the bare-well 2s node is 1.98σ at Zσ = 2.5 (Ne at h = 0.25: 0.494 Bohr; the physical point-nucleus node at ≈ 0.2 Bohr < h would not be resolvable), and the nearest grid point is within (√3/2)h, so the inner lobe is sampled up to Zσ ≈ 22. The failure modes (all fall back to `#k`) are in the `qm::orbitals` module doc. Node counting along a ray (`radial_nodes`) remains a test cross-check. m is named from the real harmonic it was aligned to: p_x, p_y, p_z; d_xy, d_yz, d_z², d_xz, d_x²−y²; the seven real f names.
5. **Open shells, from occupation, per atom** (review SHOULD 7). The atom is labelled only if, in **both** channels, every occupied orbital has a dominant l, every run length is a multiple of 2l + 1, and every shell passes step 2. An incomplete occupied l-set in either channel means the density, and hence the potential, is non-central for *both* channels. For example, C's ↓ "2s" sits in the anisotropic field of the ↑ p². In that case the whole atom gets `#k` labels and no axis alignment, and its orbitals fall back to continuity (§5.4).

This rule needs Z only (rule 1): l comes from angular character, shells from occupation and character, n from ordering. There is no shell table and no per-Z threshold.

### 5.4 Continuity: tracks, signs and colours (molecules, and everything §5.3 does not align)

**Why not a purely stateless sign rule.** Any linear functional rule (sign of ⟨ψ|w⟩) vanishes, by symmetry, on some orbital with the opposite parity to w. A non-linear rule ("the largest value is positive") flips between symmetry-equivalent lobes, such as the two ends of H₂'s σ_u, from one step to the next. So the sign is stateless only at first appearance and continuous after that (provisional ruling P3).

Per channel, with O the signed overlap of the current raw orbitals with the previous displayed ones:

1. **Clusters** (§5.1) of dimension d > 1 that §5.3 did not align are rotated by Procrustes toward the previous display. Take P = the d previous tracks with the largest Σ_{i∈C} O_ij² (weight inside span C) and A = O_{C,P}. Maximise tr(Uᵀ A): with A = W Σ Vᵀ, U = W Vᵀ (a reflection is allowed; it is a sign). The rotated overlaps are Uᵀ A. Nothing is re-read.
2. **Matching.** Hungarian maximum-weight assignment on |O| (after step 1). A pair is matched when |O| ≥ `MATCH_MIN` = 0.5. A matched orbital inherits the track and its colour slot, and its sign makes the overlap positive. An unmatched current orbital starts a new track. An unmatched previous track retires and frees its slot.
3. **First appearance** (review SHOULD 10). The sign is set by the first of these **windowed** functionals, taken about the nuclear centroid X̄ with g(r) = e^{−|r − X̄|²}, whose normalised value |⟨ψ|w⟩| / ‖w‖ (‖ψ‖ = 1) is ≥ `SIGN_MOMENT_MIN` = 1e-3:
   - the monopole w₀ = Σ_I e^{−|r − R_I|²};
   - the dipoles (x_a − X̄_a)·g, for a = x, y, z;
   - the five quadrupoles S_2m(r − X̄)·g.
   
   The sign makes that value positive. The windows make the ratio independent of the box once the box contains them. Whole-box norms grow as L^{5/2} (dipoles) and L^{7/2} (quadrupoles), so unwindowed functionals would rarely qualify. **If none qualifies, the raw solver sign is kept**, and that sign is *not* deterministic. This is rare: it needs an orbital with no weight within ~1 Bohr of the centroid in any of nine shapes. Continuity holds it steady from then on, and the module doc says so.
4. **Colour slots.** A new track takes the lowest free slot of the 12, in energy order. A ↓ orbital whose cross-channel overlap with an ↑ track is ≥ `SHARE_COLOUR_MIN` = 0.9 shares that track's slot, so a closed shell shows one colour per spatial orbital. Beyond 12 tracks, slots repeat and the tab says so.
5. **Resets.** The tracker resets (fresh tracks in energy order) on a grid change, on a provider `reset()` (a cold re-solve: nuclei added or removed, charge, multiplicity, replaced scene), and when the view is switched off. A `discontinuity()` (teleport while paused) does not reset it: matching decides. A far move simply makes new tracks.

The **lone-atom** case uses §5.3 for aligned shells and this section for the rest: its s orbitals take first appearance, then continuity; its unoccupied rows take continuity only. The track of a lone atom's 2p_x is the one matched to the previous update's 2p_x. Alignment fixes that set's identity every update, so matching is trivially stable.

### 5.5 Spin density

m(r) = ρ↑_out − ρ↓_out from the state's output densities. ∫m h³ = n↑ − n↓ by construction (each orbital normalised; f32 ~1e-7 per channel). It is shown together with ∫|m|, the absolute magnetisation, which reveals a broken-symmetry singlet (stretched H₂ beyond the Coulson–Fischer point: ∫m = 0, ∫|m| > 0). That is a real LSDA feature, and the view shows it rather than hides it.

**Normalisation.** Scale = max(max|m|, `SPIN_FLOOR` · max ρ) with `SPIN_FLOOR` = 1e-3 (provisional ruling P4). A closed shell's m is f32 and eigen-residual noise. Normalising by max|m| alone would paint that noise at full colour. The floor keeps closed-shell noise under the default 0.1 iso **only if** max|m| < 0.1 · `SPIN_FLOOR` · max ρ = 1e-4 · max ρ (review SHOULD 8). Task 0 measures that ratio on H₂, and Task 5 asserts < 1e-4. If it fails, the floor is a ruling to revisit, not a bound to loosen. The readout says "no net spin (max|m| = … · max ρ)".

### 5.6 The colour volume and the orbital scales

For the included orbitals i, with occupation f_i, slot colour c_i, and

  s_i = ψ̃_i² / P_i, with P_i = (max|ψ| over i's set)²  (*per-orbital peak*, default), or  s_i = ψ̃_i² / max ρ  (*physical*; after M7, max ρ includes the reconstructed core, §8),

the kernel writes rgb = Σ_i f_i c_i s_i and a = Σ_i f_i s_i. In *physical* scaling a summed over every occupied orbital is the total density, and the hue is which orbital dominates locally: an honest decomposition of the observable. In *per-orbital* scaling every shell is equally visible. That is the "inner spheres" the user asked for, and the legend says "each orbital scaled to its own peak". The set maximum is used because a rotated member's own maximum is not known before the rotation. It is exact for non-degenerate orbitals and within √d otherwise. It is display-only and clamped in the shader.

The shader shades a sample with colour rgb/a and strength from a (with today's γ):

- **additive**: emission only, so the colours sum along the ray and a compact 1s glows through the 2s/2p halo;
- **weighted**: today's front-to-back emission–absorption.

### 5.7 Box states

An unoccupied orbital is tagged `box state` when ε ≥ 0, or when its weight in the **outer sixth of the box**, ∫ψ² over the region within L/6 of any face (L = (n + 1)h, the Dirichlet box edge; each grid point weighted by the fraction of its cell inside the region, 1 − Π_a(1 − f_a), so the measure is O(h²) and does not jump as L/6 crosses a grid point — counting whole points gave 0.149 at 33³ against 0.163), exceeds `BOX_STATE_OUTER_WEIGHT` = 0.05 (review SHOULD 9). The first draft's "within 2h of a face" weight depends on the box size. For the lowest Dirichlet state it is ≈ 6·(2/L)(π/L)²(2h)³/3: 2.9e-3 at 49³ but 3.6e-4 at 97³, below that draft's 1e-3. The L/6 shell is scale-free. The lowest box state, a product of sin² with 1-D weight a − sin(2πa)/(2π) per face at a = 1/6, puts 1 − (1 − 2 × 0.0288)³ ≈ 0.163 there, and higher box states put more. The threshold is a third of that. A bound state with that much weight so far out is box-affected anyway. The tooltip says "a state of the simulation box, not of the molecule (unbound in LSDA)". The weight is part of the same readback.

## 6. Rendering (`render::volume`)

- The scalar pipeline gains two modes, sharing the ray setup, depth test and crossfade:
  - mode 2, **signed isosurface**: march; the first sample with |s| ≥ iso is a hit. Refine as today on |s|, take the normal from −sign(s)·∇f, and colour by sign.
  - mode 3, **signed volume**: emission–absorption on |s| with the sign's colour.
  
  Modes 0/1 (today) serve |ψ|² and the density. Signed fields crossfade correctly because the sign rule keeps consecutive fields in phase.
- A **second pipeline** for the colour volume: the same group 0, and group 1 with an rgba16float `ColourVolume3D` pair (current and previous, for the crossfade). rgba16float is filterable on every device, so there is a single sampling path. Blend modes are additive or weighted (§5.6). The output stays premultiplied, as today.
- `fade` (the surrogate's faded display) applies unchanged to every mode.
- The legend chip and the per-mode label (§7) are drawn by the app's overlay, not the shader.

## 7. Situations

| Situation | Behaviour |
|---|---|
| GPU backend | §4.2 on the device; products are textures bound by the renderer without a CPU copy. |
| CPU backend | §4.2 in f64 (`qm::orbitals`); products shipped as f32 and uploaded by the viewer. Updates come per step, which is seconds anyway; the unoccupied solve runs at the same cadence. |
| Estimated (surrogate) | A model step has no electrons: the view keeps the **last solved** orbital display, **faded** (`SURROGATE_DENSITY_FADE`) and labelled "orbitals from t = … fs (last DFT solve). The estimated solver has no electrons". It stays at that geometry, like the density today. In estimated mode the one producer is the background side solver (§4.3: audits and display refreshes; fallback steps carry no orbitals), so tracks, slots and signs come from one tracker. M5b's atomic superposition is density-only; orbital modes never use it. "Refresh cloud every N s" also refreshes orbitals. |
| View switched on while paused | `orbital_display_now()` runs §4.2 on the converged state with no solve. A provider without one (fresh scene, a surrogate whose side solver has not solved yet) shows "needs a DFT step". |
| Empty scene / zero electrons (H⁺) | No rows. Each mode draws nothing, with the note "no electrons — nothing to show". The spin readout shows n↑ = n↓ = 0. |
| An empty channel (H's ↓) | The channel's group reads "↓: none occupied". Its unoccupied rows, if requested, are eigenstates of the ↓ potential. |
| Orbital count changes (add/remove, charge, multiplicity) | A provider `reset()` resets the tracker (§5.4 item 5). The selection falls back to the same channel and index if that exists, else the HOMO, with the note "selected orbital no longer exists — showing …". |
| Grid change | Display fields and textures are rebuilt and the tracker resets. |
| Failed evaluation | No payload. The last one stays, labelled with its time. |
| Rewind (velocity flip) | Continuous; nothing special. |
| Smooth motion on | Payloads of the same mode, selection and grid crossfade with the nuclei's α, as the density does. A change of mode, selection or grid snaps. |

## 8. The pseudopotential era (M7) — core reconstruction (provisional ruling P5)

Under an M7 `IonicPolicy` that pseudises species I (pseudopotential spec §4.7), the 3-D solver holds **valence pseudo-orbitals** only.

- **Valence rows** are tagged `valence: pseudo-orbital`. The tooltip reads "equals the all-electron orbital beyond r_c = … Bohr; smooth and without core nodes inside". r_c comes from the generation record.
- **Valence n labels** use N_core,l from the partition (§4.2 there), so O's valence s is "2s", not "1s". The Sturm–Liouville ordering of §5.3 is exact only for a local central potential. The KB form is nonlocal, and its ordering of same-l states holds only when the generator's **ghost-state gate** (pseudopotential spec §4.6) has passed for that species, i.e. no spurious state below the reference valence state of any l. Labels are given only for species whose generation record shows that gate passed. Otherwise the atom is `#k` (review SHOULD 14).
- **Core rows**: for each pseudised nucleus, one row per core (n, l, m) of the generator's **spin-restricted reference atom** (review SHOULD 14). That is the atom §4.3 there constructs the potential from, and the NLCC core density comes from it too. Its u_nl(r) and ε_nl give **one row per spatial core orbital with occupation 2** (shown under both channels' headings as "core ↑↓"). The rows are tagged **`core: reconstructed for display`**, and the tooltip reads "frozen core of the isolated all-electron atom, not simulated; its energy is the isolated atom's". Core p and d sets are built aligned (real Y_lm), so no rotation is needed.
- **Evaluation**: analytic, in the shader. The orbital is φ(r) = u_nl(|r − R_I|)/|r − R_I| · Y_lm, taken from a 1-D radial table texture (one row per (Z, n, l), log-grid resampled to a uniform r grid) and evaluated at each ray sample inside r_core,I (pseudopotential spec §5.6), with the march step refined to ≤ r_core/8 there. The interpolation error is bounded **relative to the shell's max|u/r|**: 1e-5 · max|u/r|. A pointwise relative bound fails next to the radial nodes of 2s/2p cores (Al–Ar). The orbital is **not** sampled on the solver grid: an O 1s is about 0.13 Bohr, below h = 0.25.
- **Modes**:
  - Orbitals: a selected core row renders analytically.
  - All orbitals coloured: core contributions are added in the colour shader with their own slots. In *physical* scaling max ρ includes the reconstructed core.
  - Spin density: the core is **exactly** spin-neutral, because the reference atom is spin-restricted; no approximation is needed.
  - Density mode: unchanged; M7's own "with partial core" toggle stays as M7 specifies.
- **Not done**: no PAW-style reconstruction of the all-electron *valence* inside r_c (non-goal); no orthogonalisation of reconstructed core to pseudo-valence (display only, stated).

## 9. Non-goals

Time-dependent electron dynamics (TDDFT, real-time propagation, animated "orbiting" electrons). Complex orbitals and current densities (periodic k-points, magnetic fields). Natural orbitals or Dyson orbitals. Excited states. Localised orbitals (Boys, Wannier). Orbital-resolved forces or energies. An enclosed-probability iso (iso by "90 % inside"); it is a later option. Persisting orbital views per user (user-data feature).

## 10. Constants (the complete ledger; all display-only, none per element)

| Constant | Value | Why |
|---|---|---|
| `TAU_PER_TOL`, `TAU_MIN` | 100, 1e-4 Ha | §5.1 (P2): clusters by energy for molecules, unoccupied rows and tags; never for a lone atom's shells |
| `MAX_ROTATED_CLUSTER` | 7 | §5.1: beyond an f shell it is a run of box states |
| `L_CHAR_MIN` | 0.9 | §5.3: dominant angular character (linear c_lm, invariant W_l) |
| `ALIGN_RANK_MIN` | 0.5 | §5.3: the set spans every m |
| `MATCH_MIN` | 0.5 | §5.4: below it an orbital is a different one |
| `SHARE_COLOUR_MIN` | 0.9 | §5.4: ↑/↓ the same spatial orbital |
| `SIGN_MOMENT_MIN` | 1e-3 | §5.4: a windowed moment safely away from zero |
| `SPIN_FLOOR` | 1e-3 | §5.5 (P4); needs closed-shell noise < 1e-4·max ρ |
| `BOX_STATE_OUTER_WEIGHT` | 0.05 in the outer L/6 | §5.7: a third of the lowest box state's 0.163, box-size independent |
| `SMEARED_ZSIGMA` | 0.5 | §2 item 5: E(1, 0.5) = −0.3751 Ha, 25 % shallower than −0.5 |
| `UNOCC_GUARD` | 2 | §4.5: block-edge detection |
| `RADIAL_BIN_H` | 0.5 (bins of h/2) | §5.3 item 4: the node count's radial resolution (Task 1, ruling 1) |
| `NODE_FLOOR` | 1e-2 of the orbital's largest bin amplitude | §5.3 item 4: a lobe counts above it; hydrogenic outer lobes 0.07 (3s), 0.04 (4s); f32 noise ≲ 1e-6 |
| `UNOCCUPIED_BUDGET` | 0.1 of the SCF wall | §4.5: ≤ 10 % of the worker in play |
| `ORBITAL_MIN_INTERVAL`, `UNOCCUPIED_MIN_INTERVAL` | 100 ms, 1 s | §4.4, §4.5 |
| unoccupied m | 0–8, default 4 | §3 |
| `ORBITAL_PALETTE` | 12 qualitative colours | §5.4 |
| iso defaults | 0.15 (phase), 0.1 (spin) | §3 |

They go into `crates/validate/references.md` under "Orbital viewer" with these reasons (plan Task 5), next to the Task 0 measurements that justify the lone-atom rule.

## 11. Validation (rules 6 and 7; the plan pins each)

- **Task 0 spike first** (measure only). It measures:
  - the real LSDA 2p spread of a lone O (singlet: the triplet does not converge) and Ne on the GPU at h = 0.25 and 0.15, at four positions: a grid point, a cube centre, the plan's off-grid point (0.37h, −0.21h, 0.12h), and (h/2, 0, 0);
  - Ne's convergence at 57³;
  - the H₂ singlet's max|m|/max ρ;
  - the display eigensolve's wall time at 49³, 81³ and 97³.
  
  Tasks 1–5 consume these numbers. They replace the reviewer's one-electron model estimates.
- **Normalisation**: H atom 1s on the display texture, Σ ψ̃² h³ = 1 ± 1e-4 (GPU) and ± 1e-10 (CPU f64).
- **Alignment**:
  - analytic hydrogenic 2p and 3d sets under a seeded random rotation R: dominance > 0.9999 and 0.999, and **RᵀU = I to 1e-8** (U = R for φ_j = Σ_a R_ja p_a; review MUST 2);
  - Ne atom (57³, at a grid point, at the cube centre and at the off-grid point), its three 2p per channel after an injected rotation: |⟨p_k | x_k⟩| / ‖⟨p_k | x⟩‖ > 0.99 (and < 0.9 before, so the test is not vacuous), labels 1s 2s 2p_x 2p_y 2p_z at every position **whatever the measured spread**, and the spread shown in the tag (review MUST 1).
- **Signs and tracks**: 100 MD steps of a vibrating H₂ (GPU, 49³, from the 1.6 Bohr preset) **with m = 2 unoccupied** (so σ_u is tested on the real solver; review SHOULD 13): every consecutive displayed overlap ≥ 0.99 and tracks and colours constant. A CPU variant injects random sign flips and degenerate rotations every frame.
- **Spin**:
  - H: ∫m = 1 ± 1e-4.
  - Li (2↑1↓): 1 ± 1e-4.
  - O triplet: 2 ± 1e-4, `#[ignore]`d. The open-shell all-electron O does not converge at the default grids (`references.md` "The O reference is slow and keeps the singlet"). The test is **un-ignored only under an M7 policy that pseudises O**, at M7's Hund gate.
  - H₂ singlet: max|m| < 0.1 · `SPIN_FLOOR` · max ρ = 1e-4 · max ρ (review SHOULD 8).
- **No impact on the solve**: bitwise energies and forces with the view on or off, with an update forced at every step (20 payloads asserted) and unoccupied orbitals included (m = 4, 20 steps) (§4.4).
- **Parity**: GPU products against the `qm::orbitals` f64 reference on the same f32 data. Unoccupied eigenvalues GPU vs CPU for H₂ (fixed σ = h, 49³) to 1e-4 Ha.
- **Rendering**: headless pixel tests (a phase-coloured p_x is red on +x and blue on −x; additive shows the inner shell's hue through an outer one, weighted does not).

These live in `cargo test -p qm` (unit), `-p qm-gpu`, `-p render`, `-p app` and `crates/validate/tests/orbitals.rs`. The last is a display suite: it gates the orbital-view code, not solver changes, and joins `cargo test -p validate --release`.

## 12. Provisional rulings (for the controller)

- **P1** Unoccupied orbitals come from a display-only eigensolve on V[ρ_out]: block n_σ + m + 2 guard vectors, at the SCF's effective (f32-floor) tolerance, ≤ 1 Hz and ≤ 10 % of the SCF wall in play, otherwise only when paused. Never from enlarging the SCF block.
- **P2** τ = max(100 · tol_scf Ha/e, 1e-4 Ha), 1 mHa at the interactive tolerance. It is used for molecules, unoccupied rows and tags. **A lone atom's shells are found by angular character and occupation, not by τ.**
- **P3** Sign: windowed stateless moment rule at first appearance, overlap continuity after. Degenerate sets in molecules: Procrustes toward the previous display. Lone atom = exactly one nucleus. Estimated mode has a single producer: the side solver.
- **P4** Spin display floor: 1e-3 · max ρ.
- **P5** M7 core shells come from the spin-restricted reference atom and are evaluated analytically in the shader from radial tables, not sampled on the solver grid.
- **P6** n, l labels come from angular character, occupation and the radial node count on shells about the nucleus, with Sturm–Liouville order as the consistency check (after M7 also the ghost gate); never from energy order across l (controller ruling 1, Task 1). They cover occupied orbitals only, all or nothing per atom. Molecules and unoccupied rows get `#k` labels.
- **P7** All orbitals coloured: per-orbital peak scaling and additive blending by default. ↓ shares ↑'s colour at overlap ≥ 0.9.

## 13. Self-review (done before commit; repeated after the review)

- **Placeholders.** None as TBD. The one literature number mentioned (the LSD H eigenvalue) is assigned to the task that writes it into `references.md`, not quoted. The grid splitting of 2p is measured in Task 0. The lone-atom rule no longer depends on its size; the spread is only shown.
- **Rule 1.** Labels come from angular character, occupation, ordering and (M7) the generator's own partition and gates. `SMEARED_ZSIGMA` is a threshold on the dimensionless Zσ, derived from E(Z, σ) = Z² E(1, Zσ), not an element list. Colours are display. No per-element input anywhere.
- **Rule 3.** eV appears only in the list, through `units`.
- **Rule 4.** Every reduction is finished in f64. All rotations, matching and labels are f64 on the CPU. Grid products are f32 (rgba16float for colour, display only).
- **Rule 5.** `sim` carries `OrbitalRequest` and `OrbitalDisplay` opaquely. `qm::orbitals` knows no rendering. Providers produce payloads. The renderer never learns what computed them.
- **Rule 6.** No solver change, and the unoccupied solve is outside the SCF. The bitwise test (forced updates, m = 4) is the guard. The validate suite gates display code only.
- **Rule 7.** §2 lists every gap: auxiliary orbitals, eigenvalue meaning, the KS gap vs the fundamental gap, box states, integer-occupation symmetry breaking, smeared 1s, grid-split shells, arbitrary sign and rotation, reconstructed core.
- **Consistency.** The brief's "copy one orbital (or a weighted set) GPU-to-GPU" is §4.2 step 3. "CPU path uploads" is §4.2. The binding answer's axis rule is §5.3, "lone atoms only" is §5.3's definition, and molecules are §5.4. The O triplet test is ignored with its cause cited, not loosened.
- **Scope check.** Nine plan tasks (Task 0 spike plus 1–8). §8 is M7-gated and does not block the rest.

## 14. Review (physics review 2026-10-15, `docs/superpowers/notes/2026-10-15-orbital-viewer-spec-review.md`; verdict "needs rework, only in one place")

Each MUST and how it was folded in:

- **MUST 1 (§5.1, §5.3, P2; plan Tasks 0/1/5)** — τ = 1e-4…1e-3 Ha cannot absorb the grid's 2p splitting. The splitting is the egg-box of the σ = h smeared nucleus (p is an O_h irrep, so the stencil itself cannot split it). The reviewer's model gives 2.5e-3 to 2e-2 Ha off-grid, so a moving lone atom's labels would flicker.
  - **Change:** a lone atom's shells are now found by angular character (each orbital's W_l ≥ 0.9; runs of k(2l+1) cut into shells; set tests), never by an energy gap. The measured spread is shown in the tag ("split by the grid by s mHa"). There is no per-Z τ.
  - Open shells are detected from occupation, per atom.
  - The O/Ne splitting measurement moved into a new **Task 0 spike** before Task 1.
  - Task 5's Ne test now asserts alignment and labels at all positions, with no stop on the spread.
- **MUST 2 (plan Task 1, Review Focus 2)** — for φ_j = Σ_a R_ja p_a, M = c·Rᵀ, and the correct Procrustes gives U = R. The pinned "U·R = I" would fail a correct implementation.
  - **Change:** the test asserts **RᵀU = I** to 1e-8. Review Focus 2 now states U = R.
- **MUST 3 (§4.5, §5.3)** — "show unoccupied: m" could slice a degenerate box-state set, and a truncated or l-less set would wipe the occupied labels as the slider moved.
  - **Change:** labels cover occupied orbitals only, and unoccupied rows are always `#k`. The eigensolve carries `UNOCC_GUARD` = 2 guard vectors, and a cluster containing the last computed vector is hidden ("m requested, k shown"). Box states never get atomic names.
- **MUST 4 (§4.5; plan Task 4)** — the display eigensolve must run at the SCF's effective tolerance max(eig_tol, 3 × the f32 residual floor) ≈ 1.8e-6 at h = 0.25, not the raw 1e-6.
  - **Change:** stated in §4.5, pinned in Task 4.

SHOULDs, folded in at the sections named:

- S5 (§5.2): the angular character uses the linear c_lm = ⟨ψ|Y_lm⟩, whose set sum is an invariant Frobenius norm.
- S6 (§5.1): the spread can reach (d − 1)τ; the residual is ≤ spread/2 by the variance bound.
- S7 (§5.3 item 5): open shells from occupation, per atom, both channels.
- S8 (§5.5, Task 5): the noise criterion is 1e-4·max ρ; Task 0 measures it and Task 5 asserts it.
- S9 (§5.7): the box-state test is weight in the outer L/6 ≥ 0.05, box-size independent.
- S10 (§5.4 item 3): windowed sign functionals; the "kept raw sign" case is stated as non-deterministic.
- S11 (§4.5): the unoccupied solve is budgeted to 10 % of the SCF wall in play, otherwise paused-only; costs are measured in Tasks 0 and 4.
- S12 (§4.3, §7): a single producer in estimated mode (the side solver).
- S13 (Tasks 3/4/5): the bitwise test forces 20 payloads; Task 4 uses m = 4 over 20 steps; Task 5's H₂ test runs with m = 2.
- S14 (§8): core rows come from the spin-restricted reference atom; the table error is relative to max|u/r|; valence labels are conditional on the ghost gate.

NOTEs adopted:

- the `pub(crate)` change to `qm::poisson` is dropped (`solid_harmonics` is already public);
- max|ψ| is a new `AbsMax` reduction, or two reductions;
- the O-triplet test is un-ignored only under a policy that pseudises O;
- the smeared-1s tag is derived from Zσ ≥ 0.5;
- M7 physical scaling includes the core in max ρ;
- VRAM is K + 2m fields;
- the unoccupied rows say that the KS gap is not the fundamental gap.

Not checked by the reviewer: a real Ne/O LSDA GPU SCF, which Task 0 measures, and the WGSL rendering maths beyond −sign(s)∇s, which Task 6's pixel tests cover.

## 15. Task 1 rulings and review (2026-10-14/15)

- **Controller rulings on Task 1** (after the Task 0 spike): labels from angular character and radial node count, never energy order, with `shell_order_inverted` reported (§5.3 item 4, §2 item 5); grouping by angular character with the measured spread shown; RᵀU = I; open shells from occupation per atom; no per-Z constant, every threshold in `references.md` "Orbital viewer".
- **Interfaces added by Task 1**: `UpdateInput::radial` (the radial-shell profiles, lone atom only, with `moments`) and `UpdateOutput::lone_atom` (`NotLoneAtom | Labelled { shell_order_inverted } | Unlabelled(reason)`); the plan's Task 1 block lists them.
- **Physics review of Task 1** (fix round 1): the inversion derivation was rewritten — for one electron the smearing puts 2p below 2s for every σ > 0 (E_2s − E_2p ≈ Z²(Zσ)²/4), and in a screened atom the order inverts where that shift exceeds the core-penetration splitting; the oscillator is only the large-Zσ limit. The node positions come from a radial solve (1.98σ at Zσ = 2.5; nearest-point failure at Zσ ≈ 22). §5.7's weight is the cell-fraction quadrature.
