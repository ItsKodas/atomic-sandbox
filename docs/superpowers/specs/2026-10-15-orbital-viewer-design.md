# Orbital viewer: design

Status: draft for review (written 2026-10-14, dated for its build window). Child of `2026-10-12-atomic-sandbox-design.md` §8 ("Toggles: isosurface at chosen density, per-orbital view, spin density (↑ − ↓)"), the one display item of v1 that was never built. **Built after M6 (tools) merges**: it adds a dock tab, a View-menu section and Ctrl-K entries to the chrome that M6 re-homes, and it uses M6's `providers` crate and `app/src/worker/` split. It is **independent of the user-data feature** (settings profiles, model library): nothing here is persisted per user except the panel state the viewer already keeps. Its §8 (core reconstruction) is built with or after M7 (pseudopotentials); everything else works on today's all-electron solver.

Plan: `docs/superpowers/plans/2026-10-15-orbital-viewer.md`.

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
3. **Unoccupied orbitals** are eigenstates of the converged ground-state potential, computed for display (§4.5). LSDA's potential decays too fast, so many of them are not bound. In a finite Dirichlet box an unbound state becomes a **state of the box**, flagged as one (§5.7).
4. **Integer occupations break symmetry.** Integer aufbau occupation of a partly filled degenerate shell (O's 2p⁴, its ↓ channel holding one of three p) gives a non-spherical density. The degenerate set then splits, and the occupied member's direction is chosen by the SCF's history, not by physics. Such channels are labelled by energy, not as "2p" (§5.3), and their tooltip says why. Fractional, spherical occupations arrive with Fermi smearing (roadmap).
5. **Smeared nuclei.** Today every nucleus is a Gaussian of width σ = h. A 1s orbital of Z ≥ 3 is therefore wider and shallower than the physical one (pseudopotential spec §1.2), and it spans only a few voxels at h = 0.25. Its row says "1s (smeared nucleus: wider than the physical 1s)".
6. **Phase colours** show the sign of ψ. The overall sign of an orbital is arbitrary: a display rule fixes it so the colours do not flicker (§5.4). Only the relative sign of lobes means anything.
7. **Rotated degenerate sets** are labelled "rotated within a degenerate set (spread ≤ τ) for display". Each member shown is an eigenfunction to within τ (§5.1).

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
- **tags**: `HOMO` and `LUMO` (over both channels), `degenerate ×d (spread …)`, `rotated for display`, `box state`, `not converged (residual …)` (unoccupied only), `smeared 1s`, and after M7 `core: reconstructed for display` and `valence: pseudo-orbital`.

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
   - max|ψ_i| per orbital;
   - for a lone atom, the moments of §5.2 per orbital (16 linear and 16 weighted);
   - in spin mode, ∫m, ∫|m| and max|m|.
2. **CPU f64 small problems** (`qm::orbitals`): clusters, alignment or continuity rotations U (d × d per cluster), signs, track matching, labels, colour slots. Display-only. Nothing goes back into the SCF.
3. **GPU writes, no wait.** Lincombs apply U and the signs into display-owned fields. They become the next update's "previous" set and the source of the products. Then the mode's product goes into a display texture taken from the pool: the selected orbital (`unpad` + `copy_buffer_to_texture` into an r32float `Volume3D`, as for the density), the spin density (a lincomb ρ↑ − ρ↓ into scratch, then the same copy), or the colour volume (a compute kernel accumulating the weighted set straight into an rgba16float storage texture).

On the GPU path no orbital or field data ever crosses to the CPU. The readback is O(K² + 32K) scalars, where K = n↑ + n↓ plus the unoccupied orbitals shown. The **CPU backend** runs the same steps in f64 with the `qm::orbitals` reference functions, ships the product as f32 values (or RGBA f32, converted to f16 at upload), and the viewer uploads it as it uploads the CPU density today. The CPU products are also the reference the GPU kernels are tested against.

### 4.3 Types and the force boundary (rule 5)

The data types live in `sim::provider` next to `Display`. `sim` carries them opaquely, as it does the density:

- `OrbitalRequest`: `Off | List | Orbital { spin, track, style: Phase|Density } | AllColoured { include, scale: PerOrbital|Physical, blend: Additive|Weighted } | SpinDensity`, plus `unoccupied: u8`.
- `OrbitalDisplay`: the geometry it was solved at, `cell`/`origin`, the rows (`OrbitalInfo { spin, index, track, colour, energy, occupation, label, cluster, tags, residual }`), the product (`OrbitalField::{Gpu { volume, signed, scale } | GpuColour { volume, scale } | Cpu { values, signed, scale } | CpuColour { rgba, scale }}`), the spin summary and notes.
- `ForceProvider` gains three default methods, all no-ops for analytic providers: `set_orbital_view(&mut self, OrbitalRequest)`, `orbital_display_now(&mut self) -> Option<OrbitalDisplay>` (re-run §4.2 on the provider's current converged state with no solve; used when the view is switched on while paused), and the result field `ForceResult::orbitals: Option<OrbitalDisplay>`. `SideDisplay` gains `orbitals: Option<OrbitalDisplay>`. `DynSnapshot` carries the newest one.
- `providers::{CpuKs, GpuKs}` implement them. `SurrogateKs` forwards the request to its teacher and to its background auditor, so fallback steps, audits and display refreshes carry orbitals. A model step carries none.
- `qm` gains `qm::orbitals` (pure f64 numerics; no rendering or UI knowledge, rule 5). `qm-gpu` gains `qm_gpu::orbitals` (`OrbitalView`: the display-owned previous fields, the moments and colour kernels, one readback per update). `gpu` gains `ColourVolume3D` (rgba16float, filterable on every device).

### 4.4 Never touching the solve, and the update rate

- The display reads the solver's orbitals and densities. It writes only display-owned fields and textures. It runs after forces. It never changes a solver knob or its warm state. **Pinned:** 20 MD steps of H₂ give bitwise-identical energies and forces (`==` on f64) with the view off and with it on, switching modes every step (plan Task 3). `solve_is_deterministic` makes the comparison meaningful.
- **Rate.** During play, at most one update per `ORBITAL_MIN_INTERVAL` = 100 ms of wall time. A step inside the interval carries no payload, and the viewer keeps (and crossfades) the last one. Paused, single-stepping, and the first step after the view is turned on always update. The unoccupied-orbital eigensolve (§4.5) is rate-limited separately, to one per `UNOCCUPIED_MIN_INTERVAL` = 1 s during play.
- **Cost.** Per update without unoccupied orbitals: one readback, about 2K + 3 lincombs, one copy or colour dispatch. The plan's Task 7 measures it at 49³, 57³ and 97³ and records it in the README. The expectation is a few ms at 97³ for K ≤ 16, against a 1–2 s warm SCF at that size. Numbers are reported, never asserted (M4 R3).
- **VRAM.** K + m display fields (padded, 4.6 MB each at 97³), one product texture pair (r32float 3.6 MB, or rgba16float 7.3 MB, × 2 for the crossfade) and, with unoccupied orbitals, the solver's `LobpcgWork` grown to n_σ + m (+8m fields). That is about 0.2 GB at 97³ with m = 4, on a 24 GB card. All of it is freed when the view is switched off or the grid changes.

### 4.5 Unoccupied orbitals (provisional ruling P1)

The SCF solves only the n_σ occupied states per channel, and enlarging its block would change the solver (rule 6). Unoccupied orbitals therefore come from a **display eigensolve**:

- It finds the lowest n_σ + m eigenpairs of H_σ = −½∇² + V_nuc + V_H[ρ_out] + v_xc,σ[ρ_out↑, ρ_out↓], the potential of the converged *output* density. It differs from the SCF's last V[ρ_in] only within the SCF tolerance.
- It uses the same LOBPCG (`qm::eigen::lobpcg` / `qm_gpu::eigen::lobpcg_with_guess_tol`) at `ScfSettings::eig_tol`.
- The guess is a **copy** of the occupied orbitals, plus the previous update's unoccupied fields, or `initial_orbitals` for new ones.
- It runs in the display fields and the solver's `LobpcgWork` scratch (only used during a solve), on the worker thread between steps.
- **Nothing is written to the solver's state.** Pinned by the bitwise test with m = 4.

The occupied eigenpairs it returns are discarded: the rows show the SCF's own orbitals and eigenvalues, so they agree with the readouts. The test checks that they agree to τ (§5.1). Each unoccupied row shows its residual. One not converged at the LOBPCG cap is tagged `not converged (residual …)` and still shown. The CPU backend runs the same solve in f64.

## 5. Display numerics (all f64 on the CPU from the readback; rule 4; display only)

### 5.1 Degenerate clusters

In each channel, sorted eigenvalues are joined into a cluster while consecutive gaps are ≤ τ (single linkage). τ is

  **τ = max(`TAU_PER_TOL` · tol_scf · 1 Ha/e, `TAU_MIN`)**, with `TAU_PER_TOL` = 100 and `TAU_MIN` = 1e-4 Ha (provisional ruling P2).

That is 1 mHa at the display backends' `INTERACTIVE_TOL` 1e-5 and 1e-4 Ha at `SAMPLE_TOL` 1e-6. Why it is tied to tol: at SCF tolerance tol (∫|Δρ| in electrons), the potential each member of a degenerate set sees is inconsistent by δV ≈ δV_H + f_xc δρ. That shifts the eigenvalues by |⟨ψ|δV|ψ⟩|, which is O(tol) Ha per electron of Δρ when the density error sits on the atom. The factor 100 covers the anisotropic part, which is what splits a degenerate set. `TAU_MIN` covers the grid's own splitting of a degenerate set when the atom is off a cubic-symmetry site. The plan measures that splitting on Ne's 2p (Task 5) and stops if it exceeds τ/10. The rule is never loosened to make that pass.

A cluster larger than 7 (beyond an f shell) is not rotated; it gets continuity only (§5.4), because it is a run of box states. A combination of eigenvectors whose eigenvalues lie within τ has residual ≤ τ, so "an eigenfunction to within τ" (§2 item 7) is exact.

### 5.2 Moments (lone atom only; R the nucleus)

For each orbital ψ and l ≤ 3 (16 (l, m) pairs):

- linear: b_lm = ⟨ψ | S_lm(r − R)⟩, where S_lm are the real solid harmonics. They are polynomials with the Racah normalisation already in `qm::poisson`'s multipole code, reused and not re-derived. They are exactly covariant under rotations about R, so a rotation within a degenerate set acts on b_l· by the real Wigner matrix.
- weighted: c_lm = ⟨ψ | |ψ| · Y_lm(r̂)⟩, with Y_lm the unit real spherical harmonics, 0 at r = 0 for l > 0. This is a dimensionless angular character. For a cluster C, W_l = Σ_{i∈C} Σ_m c_lm,i² / Σ_{l'≤3} Σ_{i∈C} Σ_m c_l'm,i² is rotation-invariant per l.

The GPU computes all 32 per orbital in one dispatch (kernel `orbital_moments`, S_lm and Y_lm evaluated in-kernel; no harmonic fields are stored).

### 5.3 Lone atom: alignment and labels (the binding rule)

A **lone atom** is a scene with exactly one nucleus. For each channel and each cluster C of dimension d:

1. **Angular momentum.** l is accepted when d = 2l + 1, W_l ≥ `L_CHAR_MIN` = 0.9, and the d × d matrix M = (b_lm,j) has σ_min/σ_max ≥ `ALIGN_RANK_MIN` = 0.5. The last condition says the set spans every m of that l, as a proper shell does: for an ideal shell M is c × orthogonal and the ratio is 1.
2. **Rotation** (orthogonal Procrustes). The rotated set is ψ'_k = Σ_j ψ_j U_jk, k ↔ m_k. Maximise Σ_k ⟨ψ'_k | S_l m_k⟩ = tr(M U). With M = W Σ Vᵀ, U = V Wᵀ. The diagonal of MU = W Σ Wᵀ is then non-negative, so the **sign rule is built in**: every aligned member has ⟨ψ'_k | S_l m_k⟩ ≥ 0 (p_x's positive lobe on +x). The rotation is recomputed every update. It is stateless, so it is deterministic and cannot drift.
3. **Labels.** In a central potential, the k-th eigenstate of fixed l has k − 1 radial nodes (Sturm–Liouville oscillation theorem). So **n = l + 1 + N_core,l + (rank of the cluster among same-l clusters by energy)**, with N_core,l = 0 today (§8 for M7). m is named from the real harmonic it was aligned to: p_x, p_y, p_z; d_xy, d_yz, d_z², d_xz, d_x²−y²; the seven real f names. Node counting along a ray is a test cross-check (Task 1), not a runtime input: the grid cannot resolve, for example, Ne's 2s node at r ≈ 0.2 Bohr < h.
4. **All or nothing per channel.** If any cluster in the channel fails step 1 (an open shell split by integer occupation, §2 item 4), that channel gets `#k` labels and no axis alignment. Its clusters fall back to continuity (§5.4).

This rule needs Z only (rule 1): l comes from degeneracy and moments, n from ordering. There is no shell table.

### 5.4 Continuity: tracks, signs and colours (molecules, and everything §5.3 does not align)

**Why not a purely stateless sign rule.** Any linear functional rule (sign of ⟨ψ|w⟩) vanishes, by symmetry, on some orbital with the opposite parity to w. A non-linear rule ("the largest value is positive") flips between symmetry-equivalent lobes, such as the two ends of H₂'s σ_u, from one step to the next. So the sign is stateless only at first appearance and continuous after that (provisional ruling P3).

Per channel, with O the signed overlap of the current raw orbitals with the previous displayed ones:

1. **Clusters** (§5.1) of dimension d > 1 that §5.3 did not align are rotated by Procrustes toward the previous display. Take P = the d previous tracks with the largest Σ_{i∈C} O_ij² (weight inside span C) and A = O_{C,P}. Maximise tr(Uᵀ A): with A = W Σ Vᵀ, U = W Vᵀ (a reflection is allowed; it is a sign). The rotated overlaps are Uᵀ A. Nothing is re-read.
2. **Matching.** Hungarian maximum-weight assignment on |O| (after step 1). A pair is matched when |O| ≥ `MATCH_MIN` = 0.5. A matched orbital inherits the track and its colour slot, and its sign makes the overlap positive. An unmatched current orbital starts a new track. An unmatched previous track retires and frees its slot.
3. **First appearance.** The sign is set by the first of these, about the nuclear centroid X̄, with magnitude above `SIGN_MOMENT_MIN` = 1e-3 (normalised by ‖ψ‖ and the functional's norm): ⟨ψ | Σ_I e^{−|r−R_I|²}⟩, then the dipoles along x, y, z, then the five quadrupoles. It is made positive. If none qualifies, the sign is +. This is deterministic for a given geometry.
4. **Colour slots.** A new track takes the lowest free slot of the 12, in energy order. A ↓ orbital whose cross-channel overlap with an ↑ track is ≥ `SHARE_COLOUR_MIN` = 0.9 shares that track's slot, so a closed shell shows one colour per spatial orbital. Beyond 12 tracks, slots repeat and the tab says so.
5. **Resets.** The tracker resets (fresh tracks in energy order) on a grid change, on a provider `reset()` (a cold re-solve: nuclei added or removed, charge, multiplicity, replaced scene), and when the view is switched off. A `discontinuity()` (teleport while paused) does not reset it: matching decides. A far move simply makes new tracks.

The **lone-atom** case uses §5.3 for aligned clusters and this section for the rest (its s orbitals: first appearance, then continuity). The track of a lone atom's 2p_x is the one matched to the previous update's 2p_x. Alignment fixes that set's identity every update, so matching is trivially stable.

### 5.5 Spin density

m(r) = ρ↑_out − ρ↓_out from the state's output densities. ∫m h³ = n↑ − n↓ by construction (each orbital normalised; f32 ~1e-7 per channel). It is shown together with ∫|m|, the absolute magnetisation, which reveals a broken-symmetry singlet (stretched H₂ beyond the Coulson–Fischer point: ∫m = 0, ∫|m| > 0). That is a real LSDA feature, and the view shows it rather than hides it.

**Normalisation.** Scale = max(max|m|, `SPIN_FLOOR` · max ρ) with `SPIN_FLOOR` = 1e-3 (provisional ruling P4). A closed shell's m is f32 and eigen-residual noise, around 1e-6 of max ρ. Normalising by max|m| alone would paint that noise at full colour. With the floor it stays below the 0.1 iso, and the readout says "no net spin (max|m| = … · max ρ)".

### 5.6 The colour volume and the orbital scales

For the included orbitals i, with occupation f_i, slot colour c_i, and

  s_i = ψ̃_i² / P_i, with P_i = (max|ψ| over i's cluster)²  (*per-orbital peak*, default), or  s_i = ψ̃_i² / max ρ  (*physical*),

the kernel writes rgb = Σ_i f_i c_i s_i and a = Σ_i f_i s_i. In *physical* scaling a summed over every occupied orbital is the total density, and the hue is which orbital dominates locally: an honest decomposition of the observable. In *per-orbital* scaling every shell is equally visible. That is the "inner spheres" the user asked for, and the legend says "each orbital scaled to its own peak". The cluster maximum is used because a rotated member's own maximum is not known before the rotation. It is exact for non-degenerate orbitals and within √d otherwise. It is display-only and clamped in the shader.

The shader shades a sample with colour rgb/a and strength from a (with today's γ):

- **additive**: emission only, so the colours sum along the ray and a compact 1s glows through the 2s/2p halo;
- **weighted**: today's front-to-back emission–absorption.

### 5.7 Box states

An unoccupied orbital is tagged `box state` when ε ≥ 0, or when its weight on the outermost grid planes, Σ_faces ψ² h³ over points within 2h of a face, exceeds `BOX_STATE_FACE_WEIGHT` = 1e-3. Its tooltip says "a state of the simulation box, not of the molecule (unbound in LSDA)". The face weight is part of the same readback.

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
| Estimated (surrogate) | A model step has no electrons: the view keeps the **last solved** orbital display (fallback step, audit or refresh), **faded** (`SURROGATE_DENSITY_FADE`) and labelled "orbitals from t = … fs (last DFT solve). The estimated solver has no electrons". It stays at that geometry, like the density today. M5b's atomic superposition is density-only; orbital modes never use it. "Refresh cloud every N s" also refreshes orbitals. |
| View switched on while paused | `orbital_display_now()` runs §4.2 on the converged state with no solve. A provider without one (fresh scene, surrogate with no teacher state) shows "needs a DFT step". |
| Empty scene / zero electrons (H⁺) | No rows. Each mode draws nothing, with the note "no electrons — nothing to show". The spin readout shows n↑ = n↓ = 0. |
| An empty channel (H's ↓) | The channel's group reads "↓: none occupied". Its unoccupied rows, if requested, are eigenstates of the ↓ potential. |
| Orbital count changes (add/remove, charge, multiplicity) | A provider `reset()` resets the tracker (§5.4 item 5). The selection falls back to the same channel and index if that exists, else the HOMO, with the note "selected orbital no longer exists — showing …". |
| Grid change | Display fields and textures are rebuilt and the tracker resets. |
| Failed evaluation | No payload. The last one stays, labelled with its time. |
| Rewind (velocity flip) | Continuous; nothing special. |
| Smooth motion on | Payloads of the same mode, selection and grid crossfade with the nuclei's α, as the density does. A change of mode, selection or grid snaps. |

## 8. The pseudopotential era (M7) — core reconstruction (provisional ruling P5)

Under an M7 `IonicPolicy` that pseudises species I (pseudopotential spec §4.7), the 3-D solver holds **valence pseudo-orbitals** only.

- **Valence rows** are tagged `valence: pseudo-orbital`. The tooltip reads "equals the all-electron orbital beyond r_c = … Bohr; smooth and without core nodes inside". r_c comes from the generation record. Their n labels use N_core,l from the partition (§4.2 there), so O's valence s is "2s", not "1s".
- **Core rows**: for each pseudised nucleus, one row per core (n, l, m, σ) of the generator's all-electron atom (§4.1 there: u_nlσ(r) and ε_nlσ, spin-polarised), tagged **`core: reconstructed for display`**. The tooltip reads "frozen core of the isolated all-electron atom, not simulated; its energy is the isolated atom's". Core p and d sets are built aligned (real Y_lm), so no rotation is needed.
- **Evaluation**: analytic, in the shader. The orbital is φ(r) = u_nl(|r − R_I|)/|r − R_I| · Y_lm, taken from a 1-D radial table texture (one row per (Z, n, l, σ), log-grid resampled to uniform r with 1e-5 relative interpolation error) and evaluated at each ray sample inside r_core,I (pseudopotential spec §5.6), with the march step refined to ≤ r_core/8 there. **Not** sampled on the solver grid: an O 1s is about 0.13 Bohr, below h = 0.25.
- **Modes**:
  - Orbitals: a selected core row renders analytically.
  - All orbitals coloured: core contributions are added in the colour shader with their own slots.
  - Spin density: the core is drawn spin-neutral. The AE core's tiny polarisation is ignored and the readout says so.
  - Density mode: unchanged; M7's own "with partial core" toggle stays as M7 specifies.
- **Not done**: no PAW-style reconstruction of the all-electron *valence* inside r_c (non-goal); no orthogonalisation of reconstructed core to pseudo-valence (display only, stated).

## 9. Non-goals

Time-dependent electron dynamics (TDDFT, real-time propagation, animated "orbiting" electrons). Complex orbitals and current densities (periodic k-points, magnetic fields). Natural orbitals or Dyson orbitals. Excited states. Localised orbitals (Boys, Wannier). Orbital-resolved forces or energies. An enclosed-probability iso (iso by "90 % inside"); it is a later option. Persisting orbital views per user (user-data feature).

## 10. Constants (the complete ledger; all display-only, none per element)

| Constant | Value | Why |
|---|---|---|
| `TAU_PER_TOL`, `TAU_MIN` | 100, 1e-4 Ha | §5.1 (P2) |
| `L_CHAR_MIN` | 0.9 | §5.3: dominant angular character |
| `ALIGN_RANK_MIN` | 0.5 | §5.3: the set spans every m |
| `MATCH_MIN` | 0.5 | §5.4: below it an orbital is a different one |
| `SHARE_COLOUR_MIN` | 0.9 | §5.4: ↑/↓ the same spatial orbital |
| `SIGN_MOMENT_MIN` | 1e-3 | §5.4: a moment safely away from zero |
| `SPIN_FLOOR` | 1e-3 | §5.5 (P4) |
| `BOX_STATE_FACE_WEIGHT` | 1e-3 | §5.7 |
| `ORBITAL_MIN_INTERVAL`, `UNOCCUPIED_MIN_INTERVAL` | 100 ms, 1 s | §4.4 |
| unoccupied m | 0–8, default 4 | §3 |
| `ORBITAL_PALETTE` | 12 qualitative colours | §5.4 |
| iso defaults | 0.15 (phase), 0.1 (spin) | §3 |

They go into `crates/validate/references.md` under "Orbital viewer" with these reasons (plan Task 5).

## 11. Validation (rules 6 and 7; the plan pins each)

- **Normalisation**: H atom 1s on the display texture, Σ ψ̃² h³ = 1 ± 1e-4 (GPU) and ± 1e-10 (CPU f64).
- **Alignment**:
  - analytic hydrogenic 2p and 3d sets under a seeded random rotation: dominance > 0.9999 and 0.999;
  - Ne atom (57³, at the cell centre and off-grid), its three 2p per channel after an injected rotation: |⟨p_k | x_k⟩| / ‖⟨p_k | x⟩‖ > 0.99 (and < 0.9 before, so the test is not vacuous), labels 1s 2s 2p_x 2p_y 2p_z, measured 2p spread ≤ τ/10.
- **Signs and tracks**: 100 MD steps of a vibrating H₂ (GPU, 49³, from the 1.6 Bohr preset): every consecutive displayed overlap ≥ 0.99 and tracks and colours constant. A CPU variant injects random sign flips and degenerate rotations every frame.
- **Spin**:
  - H: ∫m = 1 ± 1e-4.
  - Li (2↑1↓): 1 ± 1e-4.
  - O triplet: 2 ± 1e-4, `#[ignore]`d. The open-shell O does not converge at the default grids (`references.md` "The O reference is slow and keeps the singlet"). Un-ignore at M7's Hund gate.
  - H₂ singlet: max|m| < `SPIN_FLOOR`·max ρ.
- **No impact on the solve**: bitwise energies and forces with the view on or off, unoccupied orbitals included (§4.4).
- **Parity**: GPU products against the `qm::orbitals` f64 reference on the same f32 data. Unoccupied eigenvalues GPU vs CPU for H₂ (fixed σ = h, 49³) to 1e-4 Ha.
- **Rendering**: headless pixel tests (a phase-coloured p_x is red on +x and blue on −x; additive shows the inner shell's hue through an outer one, weighted does not).

These live in `cargo test -p qm` (unit), `-p qm-gpu`, `-p render`, `-p app` and `crates/validate/tests/orbitals.rs`. The last is a display suite: it gates the orbital-view code, not solver changes, and joins `cargo test -p validate --release`.

## 12. Provisional rulings (for the controller)

- **P1** Unoccupied orbitals come from a display-only eigensolve on V[ρ_out] (block n_σ + m, rate ≤ 1 Hz in play), never from enlarging the SCF block.
- **P2** τ = max(100 · tol_scf Ha/e, 1e-4 Ha): 1 mHa at the interactive tolerance.
- **P3** Sign: stateless moment rule at first appearance, overlap continuity after. Degenerate sets in molecules: Procrustes toward the previous display. Lone atom = exactly one nucleus.
- **P4** Spin display floor: 1e-3 · max ρ.
- **P5** M7 core shells are evaluated analytically in the shader from radial tables, not sampled on the solver grid.
- **P6** n, l labels come from degeneracy, angular character and Sturm–Liouville order, not node counting, and are all or nothing per channel. Molecules get `#k` labels.
- **P7** All orbitals coloured: per-orbital peak scaling and additive blending by default. ↓ shares ↑'s colour at overlap ≥ 0.9.

## 13. Self-review (done before commit)

- **Placeholders.** None as TBD. The one literature number mentioned (the LSD H eigenvalue) is assigned to the task that writes it into `references.md`, not quoted. The grid splitting of Ne's 2p is a measurement with a stop condition, not an assumption.
- **Rule 1.** Labels come from degeneracy, moments, ordering and (M7) the generator's own partition. Colours are display. No per-element input anywhere.
- **Rule 3.** eV appears only in the list, through `units`.
- **Rule 4.** Every reduction is finished in f64. All rotations, matching and labels are f64 on the CPU. Grid products are f32 (rgba16float for colour, display only).
- **Rule 5.** `sim` carries `OrbitalRequest` and `OrbitalDisplay` opaquely. `qm::orbitals` knows no rendering. Providers produce payloads. The renderer never learns what computed them.
- **Rule 6.** No solver change, and the unoccupied solve is outside the SCF. The bitwise test is the guard. The validate suite gates display code only.
- **Rule 7.** §2 lists every gap: auxiliary orbitals, eigenvalue meaning, box states, integer-occupation symmetry breaking, smeared 1s, arbitrary sign and rotation, reconstructed core.
- **Consistency.** The brief's "copy one orbital (or a weighted set) GPU-to-GPU" is §4.2 step 3. "CPU path uploads" is §4.2. The binding answer's axis rule is §5.3, "lone atoms only" is §5.3's definition, and molecules are §5.4. The O triplet test is ignored with its cause cited, not loosened.
- **Scope check.** Eight plan tasks. §8 is M7-gated and does not block the rest.
