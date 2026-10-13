# M6 — Tools: Design Spec

**Date:** 2026-10-14
**Status:** Written from the user's captured requests (no open design questions were put to the user; provisional rulings are marked **[provisional]** and listed at the end)
**Author:** Claude (Fable 5.1) with Dakoda
**Parent:** `docs/superpowers/specs/2026-10-12-atomic-sandbox-design.md` (§5 architecture, §7 dynamics, §8 tooling, §9 validation, §11 roadmap)
**Inputs:** `docs/superpowers/notes/2026-10-13-m5-user-requests.md` (items 1–4 "For M5" — now M6 — the nuclear-events ruling and its display rule, the VSEPR ruling, the relax tool), `docs/superpowers/notes/2026-10-14-m5-handoff.md` ("Notes for M6", deferred items), `docs/superpowers/notes/2026-10-13-m4-handoff.md` (deferred list: `sim` depends on `qm`/`qm-gpu`/`gpu` against §5; the §8 hover-force readout is absent), `docs/superpowers/notes/2026-10-13-m2-viewer-checklist.md` (the promised input-replay harness).

## 1. Purpose

M6 gives the sandbox the *reading* tools a chemist expects on top of the physics that already runs: bonds and molecules read off the electron density (never off a distance table), formulas and names for what the density says is there, isotopes as a per-nucleus mass number, a one-click relaxation to find shapes, a composition badge that shows the nucleus as data, and an input-replay harness so the viewer's manual checklist stops being manual. It also pays the structural debts M4/M5 deferred: the force providers leave `sim` for their own crate behind a `Trainable` trait, the two 3 k-line files are split, the automatic sampling pass is sized from a median, and the far-drag regression is re-pinned at 49³.

Nothing in M6 adds physics to the simulation. Every tool either reads the state (bonds, names, badge), re-labels an input the physics already takes (isotope mass — one of the two allowed per-element inputs), or drives the existing provider (relax). Rule 1 holds throughout: the only per-element inputs remain Z and measured isotope masses; names, symbols and colours are display data.

### What success looks like

Build water from an O and two H; after the solve converges two thin lines appear (O–H, O–H, never H–H), the label reads **H₂O — water**; press **Relax** and the molecule settles to the minimum of the σ = 0.25 surface with the energy it shed shown as work done by the tool; swap one H for ²H in the picker and the badge reads "²H · 1 p · 1 n"; in the validation suite D₂ vibrates slower than H₂ by √(m_D/m_H) = 1.41367 on the same surface; a recorded session of the viewer checklist replays headlessly and asserts what the checklist used to ask a person to look at.

## 2. Non-goals

- **Periodic cells, pressure, pseudopotentials, Z > 10** — the next milestones (user-requests note, "Beyond M5"). The VSEPR angles (H₂O 104.5°, NH₃ 106.7°, CH₄ 109.5°) are the *oracle of the pseudopotential milestone*; M6's relax tool is what will measure them, and at h = 0.25 all-electron it will find water near-linear (documented resolution limit, M4 handoff). The relax validation here uses H₃⁺ and H₂.
- **Nuclear events** (decay, transmutation, the nuclear clock) — the later "nuclear events" milestone; M6's isotopes and badge are its inputs.
- **Bader basin charges and bond orders.** Molecular charges are not partitioned (§4.1 rule for the charge superscript); bond lines carry no order. Bond *strength* is reported as the density at the bond point, nothing more.
- **Hover/select pixel-driving of the egui panel in replays** — panel actions are replayed by name (§8), not by pixel.
- **The `ui.rs` split** — that file is being restructured on the UI-redesign branch (user-requests note, "UI redesign"); M6 does not touch its layout (§10).
- **The energy-relative box rule (κ from the HOMO eigenvalue) and the spectral Σ / warm evidence for S = 4** — surrogate performance items from the M5 handoff; they ride to the pseudopotential milestone, where S = 4 (C, N, O, H) becomes usable.
- **Smooth density crossfade, draft mode, container walls** — queued on the viewer-extras branch (`m5b-viewer-extras`), not here.

## 3. Bond detection from the density

### 3.1 The physics

Bader's theory of atoms in molecules (Bader, *Atoms in Molecules: A Quantum Theory*, Oxford 1990) defines a bond by the topology of ρ: two nuclei are linked by a bond path when a (3, −1) critical point of ρ — a minimum of ρ along the path between them and a maximum in the two directions across it — lies between them and its gradient paths terminate at both nuclei. The density at that bond critical point, ρ_b, measures the bond: covalent bonds have ρ_b ≳ 0.2 e/Bohr³, closed-shell contacts (ionic, hydrogen-bonded, van der Waals) ≲ 0.1 (Bader & Essén, J. Chem. Phys. 80, 1943 (1984)). A bond path also exists between two *repelling* closed-shell atoms (He₂ at any separation has a (3, −1) point between the nuclei — the "bond path ≠ bond" discussion), so topology alone does not say "bonded"; what does is whether the density between the atoms is *accumulated* relative to the atoms merely sitting next to each other.

M6's detector approximates the bond path by the straight internuclear segment and tests three element-agnostic conditions on a candidate pair (A, B), every constant universal and documented:

1. **A saddle on the segment.** Sample ρ at M = 33 points s_k = k/32 along R_A → R_B. The minimum over the interior points (k = 1 … 31) must be a genuine interior minimum: ρ(s*) < ρ at both endpoints and ρ(s*) ≤ ρ(s* ± Δs). At the point p* = R_A + s*(R_B − R_A), ρ must not increase across the segment: for two unit vectors e₁, e₂ ⊥ (R_B − R_A), ρ(p* ± h e_k) ≤ (1 + ε_⊥) ρ(p*) with ε_⊥ = 0.02 (the straight segment misses a bent bond path by a little; the 2 % tolerance absorbs that and the trilinear interpolation error, and is far below the ~7 % per-h drop a real transverse maximum shows — for H₂ the transverse curvature at the BCP is ≈ −0.6 a.u., i.e. ρ(h) ≈ ρ_b − 0.019 on ρ_b ≈ 0.27). This kills the false "bond" between the two H of water: the segment's minimum sits on the HOH bisector 1.1 Bohr from O, where ρ rises steeply toward O.
2. **A floor.** ρ(p*) ≥ ρ_floor = 1e-2 e/Bohr³. Below Bader's closed-shell range nothing is a bond; at 1e-2 two hydrogens 5 Bohr apart (ρ_mid ≈ 2 e⁻⁵/π = 4e-3) are not bonded, two at 2.5 Bohr (ρ_mid ≈ 0.05 before accumulation) are candidates. The floor is a universal classification constant like the surrogate's 2e-5 box rule, not a per-element number.
3. **Accumulation.** ρ(p*) / ρ_pro(p*) ≥ 1 + δ with δ = 0.10, where ρ_pro(p) = Σ_C ρ̄_{Z_C}(|p − R_C|) is the **promolecular** density: the sum over *all* nuclei of the spherically averaged density of the isolated neutral atom of that species, solved by this solver at the scene's own grid, σ and tolerance (§3.2). Covalent bonds accumulate density between the nuclei (H₂ at its σ = 0.25 R_e: ρ_mid ≈ 0.2 against a promolecule of ≈ 0.12 — ratio ≈ 1.7; the exact-H₂ figure at 1.4 Bohr is 0.27 / 0.157); closed-shell contacts *deplete* it (He₂, Pauli repulsion: ratio < 1 even where ρ(p*) > ρ_floor — the discriminator the floor alone cannot be). δ = 0.10 sits an order of magnitude above the noise of the ratio (f32 density ~1e-6 relative, radial-table interpolation ~1e-3) and an order below a covalent enhancement.

**Why the full promolecule and not ρ_A + ρ_B** (the brief's wording): with only the pair's own atoms, the H–H segment in water compares ρ at its midpoint (≈ 0.3, oxygen's valence density) with 2ρ_H(1.43) ≈ 0.036 — a ratio of 8 and a false H–H bond. The sum over all atoms is the promolecule of the deformation-density literature and makes the test mean what it says: *is there more density here than the atoms would put here on their own*. In a charged scene the neutral promolecule over-counts electrons by |charge| (H₃⁺: three neutral H have one electron too many), which lowers the ratio — H₃⁺'s edge midpoints are still far above 1 + δ (oracle, §11). Pairs farther apart than R_max = 8 Bohr are not sampled (no atom H…Ne has ρ(4 Bohr) above 1e-3 e/Bohr³, so the floor would reject them; the cutoff caps cost only).

**What this is not.** Not QTAIM: no gradient-path integration, no basins, no bond order, no ellipticity. The straight segment is exact for symmetric bonds and close for the nearly straight bond paths of the small molecules in range; ring bonds (H₃⁺'s three edges) and bent paths are sampled on their chords. Ionic bonds are the open validation question (§3.6).

### 3.2 Isolated-atom references

The promolecule needs ρ̄_Z(r), the spherical average of the isolated neutral atom's density. The surrogate's sampling pass already solves exactly these atoms (one nucleus at the cell centre, every allowed multiplicity solved cold, the lowest energy kept — ruling S9; `sim::sampling` module doc, "References") and keeps only their energies. M6 keeps their **radial density** too:

- `RadialDensity { dr: f64, values: Vec<f64> }` with dr = h/2 and r_k = k·dr up to r_max = half-extent − 3h; built by sampling ρ (§3.3) at r_k·d_j over 64 Fibonacci-sphere directions d_j and averaging per k (≈ 2 800 points at 49³, one readback). Evaluation is linear interpolation; r > r_max reads 0.
- `AtomReference { energy: f64, multiplicity: u32, radial: RadialDensity }` keyed by `ReferenceKey { n, h, sigma, tol, provider: &'static str, z }` in `sim::references::AtomReferences` — the M5 `ReferenceCache` generalised (same key semantics: the scene's grid/σ/tolerance/teacher, neutral, lowest-energy multiplicity). `sim::references::solve_atom(provider, cell, sigma, z, mass, should_abort)` is the one routine that produces an entry; the sampling pass and the driver both call it.
- **Who solves them and when.** The driver (`Dynamics`) owns the cache. Whenever the scene's species set gains a species without an entry (scene load, `AddNucleus`), the driver queues one reference job per missing species and runs them in its ticks *before* any step — one teacher solve per tick, like a sampling pass, but **without pausing** the dynamics: while a species is missing the bond graph is simply absent and the snapshot says `bonds_pending = [Z, …]` ("atom references: solving O (multiplicity 3 of 5)"). Under the surrogate the solves go to the teacher (`Trainable::teacher_mut`), never through the model; after a reference solve the provider is `reset()` (its warm state is the atom's). Cost: H 0.2 s; O at 57³ ≈ 5 cold solves ≈ 9 s on load of the `h2o` preset, shown as progress. Entries persist for the session; the surrogate sidecar's reference energies continue to seed the cache (the radial table is re-solved once when needed — the sidecar format does not change in M6).

### 3.3 Sampling ρ: a provider method and a GPU kernel

`ForceProvider` gains one optional method:

```rust
/// The latest converged density ρ↑ + ρ↓ (e/Bohr³) of this provider at arbitrary points (Bohr,
/// trilinear interpolation on its grid; 0 outside), or `None` when the provider holds no density
/// (a surrogate-answered step, an analytic provider, nothing solved yet).
fn sample_density(&mut self, points: &[[f64; 3]]) -> Option<Vec<f64>> { None }
```

- `CpuKs`: trilinear in f64 on `KsState::rho()` (`qm::Grid::trilinear`, the CPU reference).
- `GpuKs`: a WGSL kernel `sample_density` over the solver's padded `rho` field (`GpuState::rho`, the converged ρ_out — not the display texture, so it works with `set_display(false)` and reads the solver's own layout): one thread per point, 8 `textureLoad`-free buffer reads, trilinear weights in f32, written to an output buffer and read back **once** (one small readback: 33 floats per candidate pair in round 1, 4 per surviving candidate in round 2, ≈ 3 k floats for the radial table). No `float32-filterable` dependence.
- `SurrogateKs`: delegates to its *stepping* teacher only when the latest evaluation was a `Fallback` (the teacher's state is then this geometry's); otherwise `None`. The background auditor's state is never sampled (it is another geometry).
- `sim` stays ignorant of what produced the numbers: the detector consumes `Vec<f64>` and never sees a grid.

### 3.4 When the detector runs; staleness under the surrogate

`Dynamics` runs the detector after every evaluation whose `sample_density` returns `Some` (every DFT step, every surrogate fallback) and every reference-cache change, on the worker thread, before the snapshot is published (two readbacks ≈ 0.5 ms; nothing on the render thread). The snapshot carries

```rust
pub struct Bond { pub i: usize, pub j: usize, pub rho_bcp: f64, pub enhancement: f64, pub s: f64 }
pub struct BondGraph { pub bonds: Vec<Bond>, pub components: Vec<Vec<usize>>, pub step: u64, pub sim_time: f64, pub n_nuclei: usize }
// DynSnapshot:
pub bonds: Option<Arc<BondGraph>>,   // the latest graph (None before the first density or while references are pending)
pub bonds_stale: bool,               // true when the latest evaluation carried no density (a surrogate step) or the nucleus count changed since the graph
pub bonds_pending: Vec<u8>,          // species whose reference is still being solved
pub bond_rejections: Vec<(usize, usize, Rejection)>,   // why each sampled pair is not a bond (TooFar | NoInteriorMinimum | NotATransverseMaximum | BelowFloor | NotAccumulated), for the hover readout and the oracles
```

Under the surrogate the graph is stale for up to an audit interval, exactly as the density is (M5 handoff): the viewer draws the *topology* of the last DFT graph between the *current* nuclear positions, faded and labelled "bonds from the last DFT step (t = … fs) — the surrogate has no electrons", the same treatment as the faded cloud. A nucleus count change drops the graph until the next DFT evaluation (a stale graph with wrong indices is never drawn).

### 3.5 Molecules

A molecule is a connected component of the bond graph (union–find over the bonds; isolated nuclei are one-atom components). Components are listed with their formula and name (§4) next to the component's centroid in the 3D view and in the selection panel. The honest copy the user asked for goes with them: "bonds and names are read off the density after a converged DFT step" and "H₃ is not a stable molecule; H₃⁺ is".

### 3.6 Honest limits (stated in the UI copy and README)

- **Ionic bonds are the open question.** The accumulation test is a covalent criterion. In LiH the density at the bond point is small (ρ_b ≈ 0.04 in the literature) and the deformation density there combines H's gain with Li's loss of its diffuse 2s; whether the straight-segment minimum shows enhancement ≥ 1.1 at σ = h = 0.25 is *measured* by the LiH oracle (§11), asserted as bonded. A miss is reported as a finding about the criterion (rule 7) and investigated — not fixed by lowering δ (rule 6). The candidate refinement, recorded here so the investigation has a starting point: a second channel for closed-shell bonds using the sign of ∇²ρ at p* and the charge transfer visible in ρ along the segment.
- **Heavy atoms at h = 0.25** are smeared (M4 handoff): bond *topology* is right (the oracle asserts it for water), ρ_b values are not physical numbers and are shown with that caveat.
- **Charged scenes** lower every ratio by the promolecule's extra electrons; a cation's bonds can look weaker than they are.
- The straight segment is a chord of the true bond path.

## 4. Formulas and names

### 4.1 Hill formula with charge

`hill_formula(counts, charge) -> String`: C first, then H, then the rest alphabetically by symbol; without carbon all symbols alphabetically, H included (Hill, J. Am. Chem. Soc. 22, 478 (1900)). Counts as Unicode subscripts (1 omitted), charge as a superscript: ⁺, ⁻, ²⁺, ²⁻ … The scene's net charge belongs to the whole scene, and the density is not partitioned (§2), so **[provisional]**: the superscript is shown when the scene has exactly one component; with several components every formula is neutral-looking and the scene label says "net charge +1 — not assignable to one molecule without a density partition". Examples: `H₂`, `H₂O`, `LiH`, `H₃⁺`, `HeH⁺`, `C₂H₆`.

### 4.2 Structural classification from the bond graph

`classify(graph, z, charge) -> Structure` with, in this order of precedence:

1. **Trivial-name table hit** (§4.3) on (formula, charge, a graph signature where the table needs one — peroxide vs. dioxide).
2. **Homonuclear diatomic** X₂ → "di" + element name ("dihydrogen", "dioxygen"); X₃⁺ with a triangle → "tri…" is a table entry (trihydrogen cation).
3. **Peroxide**: an O–O bond where both O have at least one further bond → "{other part} peroxide" (hydrogen peroxide is the table hit; the structural rule covers e.g. a methyl peroxide seen later).
4. **Simple hydrocarbon**: every heavy atom is C and the carbons are mutually connected through C–C bonds; n_C ≤ 2 named by H count on the chain — n_H = 2n+2 "-ane", 2n "-ene", 2n−2 "-yne" with the meth/eth stem (methane, ethane, ethene, ethyne); n_C ≥ 3, rings or branches → "C_n hydrocarbon" (no isomer naming in M6).
5. **Binary compound** A_m B_n: A is the element earlier in the IUPAC electronegativity-order sequence for Z ≤ 10 — Li, Be, B, C, N, H, O, F, then He, Ne (IUPAC Red Book 2005, IR-4.4.2.1, Table VI restricted to H–Ne; a display table, rule 1) — named "{prefix_m} A-name {prefix_n} B-stem-ide" with mono omitted on A and all prefixes omitted when A is hydrogen: "hydrogen oxide" (table → water), "lithium hydride", "boron trihydride" (table → borane), "carbon dioxide", "dinitrogen oxide". Stems: hydr-, hel-, lith-, beryll-, bor-, carb-, nitr-, ox-, fluor-, neon- ("neonide" never occurs).
6. Anything else: formula only, `Structure::Unnamed`.

The result is `Name { text: String, kind: Trivial | Structural | None }`; the UI marks structural names "(systematic)".

### 4.3 The trivial-name table

A small `&[(&str /* Hill formula with charge */, &str /* name */)]` in `app::names`: H₂ dihydrogen, O₂ dioxygen, N₂ dinitrogen, F₂ difluorine, Li₂ dilithium, H₂O water, H₂O₂ hydrogen peroxide, NH₃ ammonia, NH₄⁺ ammonium, CH₄ methane, C₂H₆ ethane, C₂H₄ ethene, C₂H₂ ethyne, HF hydrogen fluoride, LiH lithium hydride, LiF lithium fluoride, BeH₂ beryllium hydride, BH₃ borane, H₃⁺ trihydrogen cation, HeH⁺ helium hydride cation, OH⁻ hydroxide, OH hydroxyl, H₃O⁺ hydronium, CO carbon monoxide, CO₂ carbon dioxide, NO nitric oxide, HCN hydrogen cyanide, CH₃OH methanol (graph signature: C–O bond). **Why a table is allowed:** it maps a *result* to a label and feeds nothing back — the same status as element colours and symbols (rule 1 forbids per-element *behaviour*, not display data). The table lives in `app`, never in `sim` or `qm`.

### 4.4 UI copy

Labels: "H₂O — water", "H₃⁺ — trihydrogen cation", "LiH — lithium hydride", "C₂H₆ — ethane (systematic)". Tooltips carry the honesty lines of §3.5 and the ρ_b of each bond on hover ("O–H: ρ_b 0.18 e/Bohr³, 1.6× the atoms' own density here — at h = 0.25 the oxygen is smeared; the topology is right, the number is not physical").

## 5. Isotopes

### 5.1 The mass number on the nucleus

`sim::Nucleus` gains `pub a: u16` (mass number). `Nucleus::new(z, pos)` keeps the most abundant isotope (unchanged behaviour: `units::isotope_mass_amu(z)`, A = `units::most_abundant_a(z)`); `Nucleus::isotope(z, a, pos) -> Result<Nucleus, String>` takes the mass from the table and refuses an unlisted A with the listed ones named. Mass stays the field the integrator uses; A is bookkeeping for the badge, the scene file and the events milestone.

**Scene file:** `nucleus Z x y z [vx vy vz]` as today; a new optional form `nucleus Z A x y z [vx vy vz]` — the parser distinguishes them by field count (5 or 8 → no A; 6 or 9 → A). Writers emit A only when it is not the most abundant one, so every existing file and preset round-trips byte-identically. `app::scene::SceneNucleus` gains `a: Option<u16>` (None = most abundant).

### 5.2 The mass table

`units` gains `isotope_mass_amu_a(z: u8, a: u16) -> Option<f64>`, `isotopes_of(z) -> &'static [u16]` and `most_abundant_a(z) -> Option<u16>`, from AME2020 (Wang, Huang, Kondev, Audi & Naimi, Chinese Phys. C 45, 030003 (2021), Table I; atomic masses, electrons included — the convention `Nucleus::new` already documents): ¹H ²H ³H; ³He ⁴He; ⁶Li ⁷Li; ⁹Be; ¹⁰B ¹¹B; ¹²C ¹³C ¹⁴C; ¹⁴N ¹⁵N; ¹⁶O ¹⁷O ¹⁸O; ¹⁹F; ²⁰Ne ²¹Ne ²²Ne — the stable isotopes plus ³H and ¹⁴C (the user asked for tritium; both are the long-lived ones the events milestone will want). The existing `isotope_mass_amu(z)` becomes the most-abundant lookup into the same table. **Correction while here:** the present table is labelled AME2020 but carries AME2016 digits (¹H 1.00782503223 is AME2016; AME2020 gives 1.007825031898). The retabulation changes masses at the 1e-10 u level — no physics moves — but the citation must be true; the `units` test pins are updated with that reason in the commit. The implementer transcribes from the published table and does not trust the planner's digits.

### 5.3 Changing an isotope

`DynCommand::SetIsotope { index, a }`: accepted at any time (forces do not depend on mass; the provider is not reset); the mass changes at fixed velocity, so K changes by ΔK = ½(m′ − m)|v|², booked as W += ΔK like a thermostat rescale (E_conserved continuous). The thermostat's K̄ is unaffected (it is a function of T and dof). The surrogate's sidecar records the scene's mass set in its provenance **for the record** (the hot burst's trajectory depends on it) but the provenance *gate* does not compare masses **[provisional]**: the Born-Oppenheimer surface the model learns is mass-independent, so refusing an H₂ model for D₂ would be physically wrong. `Provenance::describe` prints the masses.

### 5.4 Composition badge (§7) and the picker

The element picker gains an isotope choice (A from `isotopes_of`); the selection panel shows the selected nucleus's isotope with the same choice; the atom card's title and neutron count come from `a`, not from the rounded most-abundant mass.

### 5.5 Validation: H₂ vs D₂

Vibration frequencies of H₂ and D₂ on the **same** σ = h = 0.25, 49³ surface, the M4 protocol (R_e + 0.05 Bohr from rest, dt 0.25 fs, 400 steps, ω from upward zero crossings): ω_H₂/ω_D₂ against √(m_D/m_H) computed from the table the code uses — √(2.014101777844 / 1.007825031898) = **1.41367** — bound 2e-3 on the ratio. Why the bound and why this number, not √2 or the experiment (rule 7): √2 ignores the neutron–proton mass difference and binding (m_D/m_H = 1.99846, not 2 — the 4e-4 the brief attributed to anharmonicity is mostly this); experiment's ω_e(H₂)/ω_e(D₂) = 4401.21/3115.50 = 1.41268 (NIST CCCBDB, Huber & Herzberg) differs from the BO-with-atomic-masses 1.41367 by −7e-4, the adiabatic/non-adiabatic corrections no BO code reproduces; and on *our* surface the two MD runs share the same amplitude so the anharmonic term cancels, while Verlet's +h²/24 differs between them (h = ω dt: 9.7e-4 for H₂, 4.9e-4 for D₂ → +4.8e-4 on the ratio) — the expected ratio is 1.41367 × (1 + 4.8e-4 ± 3e-4 of cold-solve scatter), inside 2e-3 with margin, and the decomposition is reported. The smearing gap itself (3237 vs 4401 cm⁻¹) cancels in the ratio, which is the point of testing at the smeared σ with its own numbers.

## 6. Relax tool

### 6.1 FIRE

FIRE (Fast Inertial Relaxation Engine; Bitzek, Koskinen, Gähler, Moseler & Gumbsch, PRL 97, 170201 (2006)) is velocity-Verlet MD with three feedback rules on the power P = F·v (3N-vector dot product):

- after each step, mix the velocity toward the force direction: v ← (1 − α) v + α |v| F̂;
- if P > 0 and more than N_min steps have passed since the last P ≤ 0: Δt ← min(f_inc Δt, Δt_max), α ← f_α α;
- if P ≤ 0: v ← 0, Δt ← f_dec Δt, α ← α_start — and (Guénolé et al., Comput. Mater. Sci. 175, 109069 (2020), the "uphill correction") move the positions back by half the last step, r ← r − ½ Δt v_old, before zeroing.

Parameters (the paper's, which it shows are system-independent): N_min = 5, f_inc = 1.1, f_dec = 0.5, α_start = 0.1, f_α = 0.99. Δt starts at the run's dt (0.25 fs) and Δt_max = min(10 dt, 1.0 fs): Bitzek's 10 × Δt_MD would be 2.5 fs, but Verlet is stable only for ω Δt < 2, and a compressed H–X bond on this surface can reach ω ≈ 2 × ω_H₂(σ = 0.25) = 2.9e-2 a.u., so Δt_max = 1.0 fs = 41 a.u. keeps ω_max Δt ≈ 1.2 — FIRE's P ≤ 0 rule would catch an unstable step by halving Δt, but a kick that sends a nucleus into the box wall is not worth catching. Pinned nuclei have F = 0 and v = 0 in the FIRE vectors (the integrator's pins apply as in MD); the COM correction and the drag spring stay in `augment` (relaxing while holding a nucleus on the spring is a legitimate use).

Chosen over damped Verlet (the validation suite's `relax` helper, which quenches v when P < 0) because FIRE's adaptive Δt and α reach |F|max < 1e-3 in a few hundred steps where the quench needs thousands; the helper stays as the independent reference the relax oracle compares against (§11).

### 6.2 In the driver

`DynCommand::Relax` starts, `DynCommand::StopRelax` (and `Pause`) ends. A relax is a **tool mode** like a sampling pass: it runs in the driver's budgeted ticks (as many FIRE steps as fit, at least one), Play and Step are refused while it runs, edits are accepted (stale forces re-evaluated; FIRE restarts its Δt and α after an edit). **Bookkeeping (why it is a tool):** every FIRE velocity operation — the mixing, the quench, the backtrack's velocity zeroing — changes K by a non-Hamiltonian ΔK, booked as W += ΔK (the thermostat's booking), so E_conserved stays continuous and "work done by tools" shows exactly the energy the relaxation drained; the Verlet steps between those operations conserve the shadow energy at FIRE's current Δt, so E_conserved wobbles by up to (ω Δt_max)²/4 of the mode energy during the relax (documented; a few mHa at most). `sim_time` does **not** advance (FIRE steps are not time), the step counter does not move, and the drift history restarts when the relax ends (Δt varied, so a drift fit across it would be meaningless — the `SetDt` rule). Stop criterion: max_i |F_i| < F_TOL = 1e-3 Ha/Bohr (the brief), well above the egg-box ripple (ΣF ≈ 3e-4 at h = 0.25) so convergence is reachable; cap 2000 FIRE steps; stall detector: no 1 % improvement of max|F| over 200 steps ends the relax with a note. On completion velocities are zero and the snapshot's `relax` reads `converged: true/false`, steps, final max|F|, E.

**On the surrogate:** the relax runs on the model (fast); at the end, one teacher solve at the relaxed geometry (`Trainable::teacher_mut`) is kept as a sample and reported as `verified_max_force`; if it exceeds F_TOL the readout says "relaxed on the model; DFT max|F| = …" and the Relax button offers to continue (a second Relax resumes from there, now with fresh samples refitted).

```rust
pub struct RelaxProgress { pub steps: usize, pub max_force: f64, pub energy: f64, pub dt_fs: f64, pub alpha: f64, pub converged: Option<bool>, pub verified_max_force: Option<f64>, pub note: Option<String> }
// DynSnapshot: pub relax: Option<RelaxProgress>   (the running or last relax; None before any)
```

## 7. Composition badge and the hover readout

Hovering a nucleus (the Select tool's pick under the cursor, or the selected one) shows a small badge at the cursor: "¹⁶O · 8 p · 8 n · |F| 0.012 Ha/Bohr" — isotope symbol, proton and neutron counts (N = A − Z, from the nucleus's own `a`), and the magnitude of the raw provider force on it (spec §8's hover-force readout, deferred since M4). The atom card (selected nucleus) uses the same isotope. The nucleus stays a single sphere (ruling 2026-10-13: nucleons are not simulated; the sphere is a labelled legibility choice); the badge is data, and the atom card remains explicitly "schematic — not to scale".

## 8. Input-replay harness

The M2 checklist (`docs/superpowers/notes/2026-10-13-m2-viewer-checklist.md`) has grown three sections of "code reading only — re-check by hand"; posted OS mouse input never reached egui. M6 replaces the hand checks with recorded sessions replayed headlessly.

### 8.1 Event model

`app::replay::Event` is device-independent: `Cursor { x, y }`, `Button { which: Left | Right | Middle, down: bool }`, `Wheel { lines }`, `Key { key: Space | Delete, down }`, `Resize { w, h }`, `Action(ui::Action)`. The window's `App` converts winit events to these and feeds **the same** `Viewer::input(&Event)` the replayer feeds — the tool logic moves out of `main.rs` into `viewer::input` so there is one code path. Panel widgets call `Viewer::apply(Action)` for the things a replay must drive (`Action::Play`, `Pause`, `Step`, `Tool(mode)`, `Element(z)`, `Isotope(a)`, `Backend(b)`, `Preset(name)`, `Relax`, `Pin(i, bool)`, `Speed(n)`, `Thermostat(Option<(T, τ)>)`, `Delete(i)`, …): the egui layout — which the redesign branch is changing — is not part of a replay.

### 8.2 File format

Text, one line per entry, `#` comments, like the scene file:

```
# atomic sandbox replay v1
view 1280 800
t 120 cursor 640 400
t 130 button left down
t 400 cursor 700 420
t 410 button left up
action tool select
action element 8
action play
wait idle
wait ms 500
expect nuclei 3
expect paused false
expect selected 1
expect dist 0 1 1.5 2.5
expect bonds 2
expect molecule 0 H2O
expect no-error
```

`t` is the recorded wall time (ms) — informational; the replayer applies events in order and waits only at `wait` lines (`wait idle` = until the worker's `Busy(false)`; `wait ms`). The recorder (`--record <path>` on the windowed app) writes every event with its time and emits `wait idle` whenever the worker went idle during a gap, so a recorded session replays the way it was seen. `expect` lines assert on the `Viewer` after the preceding waits; a failure names the line.

### 8.3 Running

`--replay <path> [--preset p] [--backend b] [--screenshot out.png]` runs headlessly through the screenshot harness's `Headless` + `Viewer`, exits non-zero on a failed expectation. `cargo test -p app --release` gains `replays_pass`, which runs every `crates/app/replays/*.replay` (GPU; skipped without an adapter). The checklist items of the M2/M4/M5 notes that need a mouse become scripts: place three atoms and build water; select + drag paused (teleport) and playing (spring, no jump); orbit/pan/zoom leave the scene untouched; Delete; Space; pin; a far drag on the surrogate (fallback then recovery); Relax; isotope swap; and the new bond/label expectations. The checklist note's "re-check by hand" rows are replaced by the script names.

## 9. Deferred refactors paid in M6

- **`providers` crate and the `Trainable` trait.** `sim::provider::{CpuKs, GpuKs}`, `sim::surrogate_ks`, `sim::sampling::Sampler` move to `crates/providers` (depends on `sim`, `qm`, `qm-gpu`, `gpu`, `surrogate`, `units`). `sim` keeps the trait, `ForceResult`, `EvalStats` (with its own `EnergyBreakdown` instead of `qm::Energy`), `Display`, `Source`, the data types `SamplingPlan`/`SamplingProgress`/`TrainableStatus`, and the new `references` and `bonds` modules; `sim` then depends on `units` and `gpu` only (`Display::Gpu` holds a texture handle; `gpu` is plumbing — the §5 "units only" line is amended to say so, the alternative being a type-erased payload). The driver reaches a trainable provider through `ForceProvider::as_trainable(&mut self) -> Option<&mut dyn Trainable>` and never downcasts:

  ```rust
  pub trait Trainable {
      fn start_pass(&mut self, scene: &Scene, plan: &SamplingPlan, refs: &mut AtomReferences) -> Result<(), String>;
      /// One teacher solve of the running pass; `Ok(None)` when it has finished.
      fn pass_step(&mut self, should_abort: &mut dyn FnMut() -> bool) -> Result<Option<SamplingProgress>, ProviderError>;
      fn cancel_pass(&mut self);
      fn pass_active(&self) -> bool;
      fn progress(&self) -> Option<SamplingProgress>;
      fn refit(&mut self) -> Result<String, String>;
      fn forget(&mut self);
      fn save(&self, path: &Path) -> std::io::Result<()>;
      fn load(&mut self, path: &Path, scene: &Scene) -> Result<(), String>;
      fn teacher_mut(&mut self) -> &mut dyn ForceProvider;
      fn status(&self) -> TrainableStatus;
      fn set_audit_every(&mut self, n: usize);
      /// The next `evaluate` goes to the teacher regardless of the model (the relax verification, §6.2).
      fn force_teacher_next(&mut self);
      /// Shrink the running pass to a re-scaled plan (the median sizing below); references and the trust minimum are kept.
      fn rescale_pass(&mut self, plan: &SamplingPlan);
  }
  ```

  The app's `ModelStatus` is built from `TrainableStatus` plus the worker's own fields. A second trainable provider (the pretrained-potential door of §3 of the parent spec) then needs no driver change.
- **Splits:** `surrogate_ks.rs` (3.7 k) → `providers/src/surrogate/{mod, policy, trust, quarantine, audit, store}.rs`; `app/src/sim.rs` (2.8 k) → `app/src/worker/{mod, commands, model, autosave, tests}.rs`. Behaviour-preserving; every existing test keeps its name.
- **Automatic pass sizing from a median:** the first pass's solve count is sized from the median wall time of its first three teacher solves (the references and the first perturbations, which run first anyway) instead of 1.6 × one cold solve; the plan is re-scaled once, after the third solve, to the 2-minute target (never below the trust minimum); the offer rule is unchanged.
- **Far-drag regression re-pinned at 49³:** `after_a_far_drag_the_surrogate_answers_again` runs at 49³ (12 Bohr box), where the stretched scan samples are not box-affected, and its bound is re-pinned at the measured count × 2 (expected ≈ 1–3 fallback steps, bound ≤ 6; the 41³ run's 15 against 20 was the box, M5 handoff).

## 10. UI: seams now, chrome later

The viewer's panel is being redesigned on a sibling branch (user-requests note, "UI redesign": bottom toolbar, top menu bar, right dock with tabs, in-view overlays; acceptance = the control-mapping table). M6's UI work is therefore split: every task before the last produces **data and seams only** — snapshot fields, pure formatting functions (`hill_formula`, `classify`, `badge_text`), overlay line builders (bond lines through `render::lines`), `ui::Action` handlers — testable without a window; the final task wires them into the redesigned chrome **after that branch has merged to main** (explicit dependency: rebase `m6-tools` on main at that point; if the redesign is late, the final task wires into the current panel under a marked "M6" heading and the move is a one-task follow-up). Items for the chrome: bond lines (thin, faded + labelled when stale), molecule labels (formula — name) at component centroids and in the Selection tab, the composition/force badge on hover, the isotope choice in the element picker and the Selection tab, **Relax** on the bottom toolbar with its progress and verification line, "atom references: solving …" in the status chip, the `--record`/`--replay` flags in Help, and the honesty copy of §3.5.

## 11. Validation

New suite `crates/validate/tests/tools.rs` (release, GPU, skips without an adapter; σ = h = 0.25, 49³ unless stated; every number of record in `references.md`, section "Tools (M6)"):

| Test | What it pins | Bound | Reference |
|---|---|---|---|
| `h2_is_bonded_at_equilibrium_and_not_at_six_bohr` | H₂ at R_e(σ) = 1.789: one bond, ρ_b and enhancement reported; R = 6.0 (65³, 16 Bohr box): no bond; the enhancement decreases monotonically over R ∈ {1.8, 2.5, 3.5, 5.0} and the R where it crosses 1 + δ is reported | bonded / not bonded; enhancement at R_e ≥ 1.3 | Bader 1990 (BCP); Bader & Essén 1984 (ρ_b ranges) |
| `he2_is_a_closed_shell_contact_not_a_bond` | two He at 3.0 Bohr: a saddle exists (reported), ρ(p*) > ρ_floor, enhancement < 1 → no bond | enhancement < 1.0 | the "bond path ≠ bond" literature; the deformation density of He₂ is negative between the nuclei |
| `heh_plus_and_h3_plus_are_bonded` | HeH⁺ at 1.46 Bohr: bonded; H₃⁺ equilateral (relaxed on this surface): three bonds, one component (the formula `H₃⁺` is the app's `names` test) | bonded; 3 bonds | — |
| `water_has_two_oh_bonds_and_no_hh_bond` | the `h2o` preset at 57³ (physical geometry): bonds {O–H, O–H} exactly, formula `H₂O`, name "water"; the H–H segment's saddle test fails (reported which condition) | exactly those bonds | §3.1 |
| `lih_is_bonded` | LiH at 3.0 Bohr: bonded; enhancement reported (the ionic case, §3.6) | bonded | Bader 1990 ch. 7 (LiH as the closed-shell/ionic exemplar) |
| `d2_vibrates_slower_than_h2_by_the_mass_ratio` | §5.5 | \|ratio / 1.41367 − 1\| ≤ 2e-3 | AME2020; NIST CCCBDB (Huber & Herzberg) for the reported experimental 1.41268 |
| `fire_relaxes_h3_plus_to_the_quenched_minimum` | FIRE from a 0.2 Bohr-perturbed H₃⁺ vs the suite's quenched-Verlet `relax` helper: same geometry (side lengths), fewer evaluations; max\|F\| < 1e-3; E_conserved continuous across the relax (W = E_start − E_end to the Verlet shadow bound) | \|Δside\| ≤ 2e-3 Bohr; evaluations ≤ 0.5 × quench's | Bitzek 2006; Guénolé 2020 |
| `fire_on_the_surrogate_is_verified_by_the_teacher` | relax H₂ on a trained surrogate: DFT max\|F\| at the result reported; ≤ 3e-3 | reported, asserted ≤ 3e-3 | M5 force tier |

Unit tests pin the detector on analytic densities (Gaussians: a bonded pair, a depleted pair, a third-atom false bond), the formula/name rules, the scene round trip with A, the mass table and ratios, FIRE on the Morse dimer, and the replay parser. `references.md` records every universal constant (M = 33, ε_⊥ = 0.02, ρ_floor = 1e-2, δ = 0.10, R_max = 8, dr = h/2, 64 directions, FIRE's parameters, F_TOL, the Δt_max rule) with its derivation line.

## 12. Risks

- The LiH oracle may fail on physics (§3.6): planned as a finding, not a tolerance change.
- The reference solves for O at 57³ (≈ 9 s on load) are visible; the status chip must say what is happening.
- FIRE on a surface with the egg-box ripple can stall near F_TOL for a soft mode: the stall detector reports instead of spinning.
- The `providers` refactor touches every provider test; it goes first so everything after it builds on the new seams, and its acceptance is "every existing suite green with unchanged numbers".
- The chrome task depends on another branch; the seams are designed so the wiring is small.

## 13. Provisional rulings (made here, flagged for the user)

- **P1** Charge superscript only for a single-component scene (§4.1); multi-component scenes state the unassigned net charge.
- **P2** The surrogate provenance *records* the mass set but does not *gate* on it (§5.3): the BO surface is mass-independent.
- **P3** Δt_max = min(10 dt, 1.0 fs) for FIRE (§6.1), tighter than the paper's 10 Δt_MD, for Verlet stability on compressed H–X modes.
- **P4** `sim` keeps a dependency on `gpu` for the opaque texture handle (§9), amending the parent spec's "units only".
- **P5** The promolecule sums all atoms, not the pair (§3.1), and the detector adds the transverse-maximum test; both depart from the brief's one-line criterion for the reason given there.
- **P6** Bond detection reference solves run without pausing the dynamics (§3.2); the graph is absent until they finish.

## 14. References

- R. F. W. Bader, *Atoms in Molecules: A Quantum Theory*, Oxford University Press (1990).
- R. F. W. Bader & H. Essén, J. Chem. Phys. 80, 1943 (1984) — ρ_b and ∇²ρ_b ranges for shared vs closed-shell interactions.
- E. Bitzek, P. Koskinen, F. Gähler, M. Moseler & P. Gumbsch, Phys. Rev. Lett. 97, 170201 (2006) — FIRE.
- J. Guénolé, W. G. Nöhring, A. Vaid, F. Houllé, Z. Xie, A. Prakash & E. Bitzek, Comput. Mater. Sci. 175, 109069 (2020) — FIRE 2.0 (the uphill correction).
- M. Wang, W. J. Huang, F. G. Kondev, G. Audi & S. Naimi, Chinese Phys. C 45, 030003 (2021) — AME2020 atomic masses.
- NIST Computational Chemistry Comparison and Benchmark Database (CCCBDB), experimental vibrational constants: H₂ ω_e = 4401.21 cm⁻¹, D₂ ω_e = 3115.50 cm⁻¹ (K. P. Huber & G. Herzberg, *Constants of Diatomic Molecules*, 1979).
- E. A. Hill, J. Am. Chem. Soc. 22, 478 (1900) — Hill formula order.
- IUPAC, *Nomenclature of Inorganic Chemistry* (Red Book, 2005), IR-4.4.2.1 and Table VI — element ordering for binary names (display data).
- Spec §7 / M4 handoff — the σ = 0.25 surface numbers (R_e 1.789 Bohr, k 0.200 Ha/Bohr², ω 3237 cm⁻¹, egg-box ΣF ≈ 3e-4 Ha/Bohr).
