# M7 — Self-generated pseudopotentials: design

Status: draft for physics review (Fable), 2026-10-14. Child of `2026-10-12-atomic-sandbox-design.md` (§6 "Nuclear potential … behind a `NuclearPotential` trait so self-generated pseudopotentials can replace it", §11 item 1). Scheduled after M6 (tools) per the roadmap in `docs/superpowers/notes/2026-10-13-m5-user-requests.md` ("pseudopotentials → transfer → electrostatics → periodic cells"). The biggest single physics job left; every task in §8 is tagged.

Rule 1 (CLAUDE.md) is the crux of this document: **the only per-element inputs stay Z and the isotope mass.** A pseudopotential here is a *derived object* — this project's own all-electron atom, transformed by one element-independent recipe whose every constant is universal and listed in §4.9. Nothing is loaded from a pseudopotential library; nothing is tuned per element. Where the standard recipes of the literature use hand-chosen per-element radii, this design replaces them with a universal rule plus element-independent acceptance gates (§4.4, §4.8), and says so.

## 1. The problem, quantitatively

### 1.1 What the grid must resolve

The all-electron Kohn-Sham solver puts every electron of every atom on one uniform grid of spacing h (0.25 Bohr default). The 1s orbital of a nucleus of charge Z is hydrogenic, R_1s ∝ e^{−Z_eff r}, with the maximum of r²R² at r = 1/Z_eff and ⟨r⟩ = 3/(2Z_eff). For O (Z_eff ≈ 7.7) that is 0.13 Bohr; C 0.17, N 0.15, Ne 0.10. The density at the oxygen nucleus is ρ(0) ≈ 2Z_eff³/π ≈ 3 × 10² e/Bohr³ (against 1/π for H).

The grid cannot represent the cusp at all (`crates/qm/src/nuclear.rs`, "Why smear"): the 8th-order stencil of `stencil.rs` turns a cusp into O(1) noise, so each nucleus is deposited as a Gaussian of width σ for the electrons, and the grid must then resolve *that* Gaussian — σ ≥ h for the analytic erf potential, σ ≳ 2h for the Poisson route the GPU uses (`nuclear.rs`, "Resolution of σ by the grid": the stencil error on a feature of width σ falls as (h/σ)⁸ and is 4e-3 at σ = 1.2h, 1.7e-5 at 2.4h). The smearing costs ΔE ≈ 2πZρ(0)σ² + O(σ³) + O(σ⁴), and the validation suite removes it by extrapolating a σ ladder to σ → 0 (`scf::extrapolate_sigma`). That series is only asymptotic for Zσ ≲ 0.2: measured in `crates/validate/references.md` ("Known limitation: Li"), He at Zσ_max = 0.48 misses by 46 mHa at h = 0.15, Li at Zσ_max = 0.72 by 331 mHa, and the stated requirement for the 40 mHa class is **Z·h ≲ 0.2**.

| target | condition | h needed for O (Z = 8) | points vs 57³ at h = 0.25 (14 Bohr box) | solve cost (∝ h⁻⁴, §1.3) |
|---|---|---|---|---|
| 40 mHa class (σ-ladder extrapolation asymptotic) | Zσ_max ≲ 0.2, σ_max = 1.6h | h ≲ 0.016 | 3.8 × 10³× | ~6 × 10⁴× |
| 1 Ha class | Zσ ≲ 1, σ = h | h ≲ 0.12 | 9× | ~20× |
| brief's "resolve the 1s with the stencil" (3.2e-4 (Zh)⁸ relative kinetic error below 1e-4 of T_1s ≈ 32 Ha) | Zh ≲ 0.9 | h ≲ 0.11 | 11× | ~25× |
| the 1s radius resolved by ≥ 3 points, with σ = h and the Poisson floor σ ≥ 2h | r_1s/3 with the floor | h ≲ 0.05 | 125× | ~600× |
| **today** | σ = h = 0.25, **Zσ = 2** | — | 1× | 1× |

### 1.2 What Zσ = 2 does to oxygen (why O is "half" its energy and a singlet)

At σ = 0.25 the smeared potential of O is a well of depth V(0) = −Z√(2/π)/σ = −25.5 Ha that is *harmonic* out to d ≈ a = σ√2 = 0.35 Bohr: expanding erf(d/a)/d gives V(d) ≈ V(0) + ½k d² with k = 4Z/(3√π a³) = 136 Ha/Bohr² (the same expansion as `forces.rs`, "g′(d) and its d → 0 limit"). A 3-D oscillator of that stiffness has ω = √k = 11.7 Ha and a ground state at V(0) + 3ω/2 ≈ −8 Ha with width (1/ω)^{1/2} = 0.29 Bohr — against the all-electron LSDA 1s eigenvalue of ≈ −18.8 Ha and radius 0.13 Bohr. The two 1s electrons therefore sit ~11 Ha too high *each* before any screening, the "core" is twice its true size, the 2s and 2p that must stay orthogonal to it are pushed outward, and the sp energetics that set bond angles are wrong. Measured consequences, all in `references.md` and `crates/sim/src/sampling.rs`:

- O atom: E ≈ −36.6 Ha (33³/57³, σ = h = 0.25) against the resolved LSDA ≈ −74.5 Ha; the **singlet lies 0.227 Ha below the triplet** (−36.565 vs −36.338 at 33³; at 57³ the triplet and quintet do not even converge cold) — Hund's rule does not emerge (`gpu_oxygen_reference_multiplicity`, ignored with the numbers).
- H₂O: the σ = 0.25 surface's minimum is **O–H 2.34 Bohr, 177.2°**, 0.160 Ha below the physical geometry the preset loads (`thermostat_h2o_preset_known_limitation`), so Play heats water past 10 000 K.
- H₂: R_e 1.79 Bohr and ω = 3237 cm⁻¹ at σ = 0.25 against 1.45 Bohr / ≈ 4200 (LSDA) / 4401 (experiment) — H is only at Zσ = 0.25, and still loses 25 % of its frequency to the smearing.
- Li: the σ → 0 oracle misses by 331 mHa and is `#[ignore]`d.
- The egg-box: Σ_I F_I,y = −3.4e-4 Ha/Bohr for H₂O (the Z = 8 Gaussian moving across the grid), vs ≤ 4e-5 for H.

### 1.3 Why finer uniform grids do not scale

Points ∝ h⁻³. The LOBPCG iteration count grows with the condition number of H, whose top is the kinetic maximum ∝ 1/h² (the Jacobi preconditioner removes little of it), so iterations ∝ 1/h and the solve ∝ h⁻⁴ at best; the Poisson multigrid is ∝ h⁻³. From the measured H₂O 57³ solve of 1.8 s (0.25 Bohr), the 1 Ha class costs ~40 s per MD step and the 40 mHa class hours — and that is for the *atom*; the molecule needs the box. The surrogate (M5) removed the dynamics bottleneck but not the teaching one: every teacher solve is one of these. No interactive path runs through h ≲ 0.05.

The smearing is a numerical device for a feature the valence electrons never need: the 1s of O is 0.13 Bohr wide, chemically inert, and 18.8 Ha below the valence. Removing it from the grid problem — rather than resolving it — is the whole content of this milestone.

## 2. Options

### (a) Norm-conserving pseudopotentials generated in-house — **recommended**

Replace, for each species with a core, the −Z/r nucleus plus its core electrons by an ionic potential that (i) is smooth inside a core radius r_c, (ii) equals −Z_v/r outside, (iii) reproduces the all-electron valence eigenvalues and the valence wavefunctions beyond r_c exactly at the reference configuration, and (iv) is norm-conserving (Hamann, Schlüter & Chiang, *Phys. Rev. Lett.* 43, 1494 (1979)), which makes the logarithmic derivative's energy dependence correct to first order and so makes the potential transferable to molecules. Two constructions:

- **Troullier–Martins** (TM; Troullier & Martins, *Phys. Rev. B* 43, 1993 (1991)): one pseudo-wavefunction per l, r^{l+1} e^{p(r)} with an even polynomial p of degree 12 inside r_c, fixed by norm conservation, C⁴ matching at r_c and zero curvature of the screened potential at the origin; the potential follows by inverting the radial equation. Simple, closed-form, designed for smoothness, and the classical choice for first-row elements on real-space grids (Chelikowsky, Troullier & Saad, *Phys. Rev. Lett.* 72, 1240 (1994); Chelikowsky, Troullier, Wu & Saad, *Phys. Rev. B* 50, 11355 (1994) — the PARSEC lineage this solver already follows for its stencil).
- **ONCV** (Hamann, *Phys. Rev. B* 88, 085117 (2013)): two projectors per l, optimised in a spherical-Bessel basis for a residual-kinetic-energy criterion (after Rappe, Rabe, Kaxiras & Joannopoulos, *Phys. Rev. B* 41, 1227 (1990)), with norm conservation generalised to two reference energies. More accurate at a given smoothness and the only way to carry a semicore and a valence state of the same l; much heavier to implement (constrained optimisation, generalised eigenproblems).

Both are "Z in, potential out" once the recipe's choices are fixed universally (§4). The literature fixes r_c, the local channel and the partial-core radius per element by hand; this design derives them (§4.4) — the one place where rule 1 is not automatically satisfied by the method and must be enforced by design.

Trade-offs: a new 1-D solver and generator (CPU, seconds per element); a nonlocal operator in the Hamiltonian (cheap: §5.3); the frozen-core approximation (core–core overlap at short distances must be flagged, §5.6); the egg-box moves from the nuclear Gaussian to the projectors and short-range local part (reduced, not gone — §5.1, §7); ghost states (§4.6). Absolute energies become valence energies (§6.5 — labelled).

### (b) Projector-augmented waves (Blöchl, *Phys. Rev. B* 50, 17953 (1994))

Keeps the all-electron wavefunction reconstructible through partial waves and compensation charges; softer than NC potentials and the standard of GPAW/VASP. Heavier on every axis: partial-wave bases and overlap operator (a generalised eigenproblem in LOBPCG), compensation densities in the Poisson solve, one-centre corrections in every energy and force term, and a much larger generation apparatus. The same rule-1 questions (radii, partial-wave sets) recur. The accuracy gain over NC-TM is not needed for first-row chemistry at the honesty level this project reports. **Rejected for this milestone;** the `Ionic` abstraction of §5 does not preclude it.

### (c) Frozen-core all-electron on a finer local grid near nuclei (multi-resolution)

Keep the cusp (or a narrow σ) on nested finer grids around each nucleus (Ono & Hirose's double grid, *Phys. Rev. Lett.* 82, 5016 (1999), extended to the whole core; adaptive coordinates per Gygi & Galli, *Phys. Rev. B* 52, R2229 (1995) or Modine, Zumbach & Kaxiras, *Phys. Rev. B* 55, 10289 (1997)). Honest and rule-1-trivial (no recipe at all), but every kernel — stencil, Poisson multigrid, density accumulation, LOBPCG's inner products, the renderer's volume — would have to handle non-uniform resolution and the interfaces between levels, the moving nuclei would drag the fine patches across the coarse grid (a worse egg-box), and the f32 GPU path would have to carry 10² e/Bohr³ next to 10⁻³. **Kept as a note:** if a later milestone wants all-electron heavy atoms for a specific reason (core-level spectroscopy, nuclear events needing the core density at the nucleus), this is the route, and the double-grid part of it is already this design's plan B for the projector integrals (§5.1).

### (d) Finer uniform grids with the smeared nucleus

Honest by construction (the existing solver, smaller h) and the right thing for the *oracles*. §1 prices it at 10²–10⁴× per solve for oxygen and more for Z > 10; not an interactive route and not a teaching route. Stays what it is: the validation reference for the all-electron limit (§6.1).

**Recommendation: (a), TM first, KB-separable, with NLCC, under a universal recipe and element-independent gates; ONCV as the stage-3 upgrade for semicore cases (§4.3).** Reasons: it is the smallest change to the real-space solver (one local field, a handful of localised projectors, three force terms), it is what every real-space DFT code of this family does, its generation is seconds on the CPU from the all-electron atom we must write anyway, and its approximations are all measurable against our own all-electron atom (rule 7).

## 3. Non-goals

- No pseudopotential files loaded from anywhere (rule 1). Tables are generated at run time (and cached, keyed by the recipe hash) — never shipped as data.
- No ultrasoft (Vanderbilt, *Phys. Rev. B* 41, 7892 (1990)) or PAW this milestone.
- No relativistic corrections (scalar-relativistic radial equations are a documented later extension for Z ≳ 30; irrelevant for the first two rows).
- No GGA: the generator uses the same LSDA-PW92 as the grid solver, and the XC identity is part of the recipe hash so a later PBE regenerates everything.
- No change to the σ → 0 all-electron oracles: they test the solver and remain as they are.

## 4. The generation pipeline (rule-1-honest)

Everything in this section lives in a new module tree `crates/qm/src/radial/` (CPU, f64, no GPU), with the universal constants in one table (§4.9).

### 4.1 Radial all-electron LSDA atom (`qm::radial::atom`)

Spherical, spin-polarised Kohn-Sham atom on a logarithmic grid r_i = r_0 e^{i·δ} with universal r_0 = 1e-5 Bohr, r_max = 60 Bohr, δ = 0.008 (the grid rule is Z-independent; its usefulness for the 1s requires r_0 ≪ 1/Z, which 1e-5 satisfies for every Z ≤ 118). For each (n, l, σ) the radial equation −½u″ + [l(l+1)/2r² + V_σ(r)]u = ε u is integrated outward from r_0 and inward from r_max and matched at the outermost classical turning point by the shooting method (node counting fixes n; bisection on ε then Newton on the derivative mismatch), in the transformed variable of the log grid with Numerov's rule — the standard atomic-structure method (Hartree, *The Calculation of Atomic Structures*, Wiley 1957; Herman & Skillman, *Atomic Structure Calculations*, Prentice-Hall 1963; Fuchs & Scheffler, *Comput. Phys. Commun.* 119, 67 (1999) for the modern pseudopotential-generator form of it).

- Potential: V_σ = −Z/r + V_H[ρ] + v_xc,σ[ρ↑, ρ↓]. The radial Hartree potential by the two cumulative integrals (Gauss's law); the XC exactly `qm::xc::lsda_pw92` on the radial densities (code reuse, not a re-derivation).
- Occupations: **derived, never tabulated.** The integer aufbau of the 3-D solver (`qm::scf`, "Electron count and multiplicity") applied radially: each SCF step occupies the lowest eigenvalues, with the spherical average over m (fractional occupation per m of an open shell), and the multiplicity found by the S9 scan (solve the candidates, keep the lowest energy — `crates/sim/src/sampling.rs`). Near-degenerate shells (4s/3d and beyond) can make the aufbau order oscillate between SCF steps; the universal remedy is the same scan: every occupation pattern that differs from the aufbau one by moving electrons between shells within Δ_scan = 0.1 Ha of the highest occupied level is solved, and the lowest total energy wins. No ground-state configuration table exists anywhere in the code (rule 1: a configuration table would be per-element data). For Z ≤ 18 the aufbau order is unambiguous and the scan is a no-op.
- Mixing: DIIS on the radial densities (the 3-D code's `diis` logic on 1-D vectors), tol 1e-10 on ∫|Δρ| r² dr.
- Spin-restricted mode: the same code with ρ↑ = ρ↓ (needed by §4.3).
- Outputs: ε_nlσ, u_nlσ(r), ρ_σ(r), the total energy with its components, and the per-shell outermost maximum r_max,nl of |u| and outermost node r_node,nl.

First tests (the per-element *references* here are test oracles, not inputs — same status as the NIST numbers already in `references.md`): H −0.478671, He −2.834836, Li −7.335195 Ha (NIST LSD column, Kotochigova et al. 1997, already cited there) to 1e-5; C, N, O, Ne LSD totals from the same table (read off at task time; the spec does not quote digits it has not checked against the source) to 1e-4; the Hund triplet of O lowest among M = 1, 3, 5 in the spherical LSDA atom; the hydrogenic check of the integrator itself (V_H and v_xc off: ε_nl = −Z²/2n² to 1e-10 for several n, l, Z); the virial ratio −V/T = 2 to 1e-5 for every converged atom.

### 4.2 Core / valence by a universal rule (`qm::radial::partition`)

1. Seed: every occupied shell with the highest occupied principal quantum number n_max is valence.
2. Promote whole lower shells: the shell n = n_max − 1 (all its subshells) is promoted to valence — "semicore" — if any of its occupied subshells has ε_nl > ε_v,min − Δ_core, with ε_v,min the lowest seed eigenvalue and **Δ_core = 2.0 Ha** (universal). Repeat for n_max − 2 (never triggers below the f block). Everything else is core.
3. Channels: a pseudopotential channel is generated for every l ≤ l_max, l_max = max(l occupied in the valence) + 1, provided a bound state of that l exists in the reference configuration (an unbound channel is dropped; its scattering is left to the local part).
4. Consequences (derived, not chosen): H and He have no core → no pseudopotential under the default policy (§4.7 — they stay all-electron smeared, identical to today). Li: 1s (−1.9 Ha) is within 2 Ha of 2s (−0.1) → the whole n = 1 shell is valence → the s channel carries two valence states, which a one-projector TM/KB potential cannot represent (§4.3) → the generator **refuses** Li (reported, not patched) and Li remains the all-electron known limitation until ONCV. Be–Ne: 1s core (Be 1s ≈ −3.9 Ha vs 2s −0.2: gap 3.7). Na, Mg: the n = 2 shell promoted (Na 2p ≈ −1.0 Ha vs 3s −0.1) → 9/10-electron semicore potentials, which again need two s-projectors → ONCV stage. Al–Ar: n = 2 core (Al 2p ≈ −2.6 Ha vs 3p −0.1). These eigenvalues are quoted here only to show what the rule does; the code reads them from its own atom.

### 4.3 Construction per l (`qm::radial::tm`)

Reference configuration: the **spin-restricted ground configuration** (ρ↑ = ρ↓) found in §4.1. The ionic potential must be spin-independent — it stands in for the nucleus plus a closed core — and the standard that makes a spin-restricted-generated potential valid in spin-polarised calculations is the nonlinear core correction (§4.5; Louie, Froyen & Cohen, *Phys. Rev. B* 26, 1738 (1982) introduced the NLCC for exactly this purpose). The spin-polarised atom of §4.1 is the *gate* (§4.8), not the generation input. (The brief's "spin-polarised, lowest-energy multiplicity" is therefore where the radial solver is validated and where the transferability is tested; it is not what the potential is built from. This is a deliberate departure, with the reason stated.)

TM (PRB 43, 1993, eqs. 4–13), for each channel l with reference eigenvalue ε_l and AE radial function u_l^{AE}:
- u_l^{PP}(r) = u_l^{AE}(r) for r ≥ r_c,l; u_l^{PP}(r) = r^{l+1} exp(c₀ + c₂r² + c₄r⁴ + … + c₁₂r¹²) for r < r_c,l, the seven coefficients fixed by: norm conservation ∫₀^{r_c}|u^{PP}|² = ∫₀^{r_c}|u^{AE}|²; continuity of u and its first four derivatives at r_c; and V_l^{scr}″(0) = 0 (the condition 2c₂² + c₄(2l + 5) = 0 in TM's notation). Solved by Newton on c₂ with the linear 5×5 system for the rest at each step, exactly as TM describe.
- V_l^{scr}(r) = ε_l − l(l+1)/(2r²) + u″/(2u) by inversion (nodeless by construction, so the division is safe; the r → 0 limit is taken analytically from the polynomial).
- Unscreening: V_l^{ion} = V_l^{scr} − V_H[ρ_v^{PP}] − v_xc[ρ_v^{PP} + ρ_pc] with ρ_v^{PP} the pseudo-valence density of the reference configuration and ρ_pc the partial core of §4.5 (the unscreening must use the same partial core the molecule will use, or the atom is not reproduced).
- Each channel's potential is extended beyond r_c as −Z_v/r (exact there up to the exponential tail of the AE core density, which is below 1e-8 beyond ~3 r_c for every Z ≤ 18).

Limitations stated: one reference state per l (the lowest valence state of that l). A channel with two valence states of the same l (Li, Na, Mg — §4.2) is **refused**, with the message naming the ONCV stage; the recipe never silently pseudises the lower one and hopes.

### 4.4 Cutoff radii and the local channel by a universal rule (`qm::radial::recipe`)

Per channel, r_c,l = f · r_max,l where r_max,l is the outermost maximum of |u_l^{AE}| (always beyond the outermost node, so the pseudo-wavefunction is nodeless — the TM requirement). f is **not per element**: it is taken from the universal ladder **F = {1.6, 1.4, 1.2, 1.0}**, softest first, and the first f for which the potential passes every gate of §4.6 and §4.8 is kept. If none passes, the element is refused at this recipe version (reported in the generation record; never worked around by hand). The classical hand-chosen TM radii for C/N/O (≈ 1.3–1.5 Bohr, PRB 43, 1993, Table I) correspond to f ≈ 1.4–1.5 on these elements' 2p maxima (≈ 0.9–1.2 Bohr) — the rule lands where the literature did, which is a check, not an input.

Local channel: the universal preference order is
1. V_loc = V_{l_max}^{ion} (the highest generated channel: for the first row, with no bound d state, that is the p channel — the classic p-local O/N/C potentials);
2. a smooth polynomial local (ONCV §II.C-style): V_loc(r) = −Z_v/r for r ≥ r_loc, continued inside by the even polynomial matched C² at r_loc, with r_loc = min_l r_c,l.
The first candidate that passes the ghost test (§4.6) at the current f is used; the remaining channels become projectors. Both candidates are smooth and finite at the origin.

Semicore channels, l_max, Δ_core, F, the order of local candidates, the NLCC crossover (§4.5) and the gate thresholds (§4.8) are the complete list of recipe choices; §4.9 tabulates them with their justification.

### 4.5 Nonlinear core correction (`qm::radial::nlcc`)

LSDA is nonlinear in the density, so E_xc[ρ_v] ≠ E_xc[ρ_v + ρ_c] − E_xc[ρ_c], and the error is largest exactly where it hurts a spin-polarised calculation (the open-shell O triplet). Louie, Froyen & Cohen (PRB 26, 1738 (1982)) keep a *partial* core density in the XC functional: ρ_pc(r) = ρ_c(r) for r ≥ r_nlc, A sin(Br)/r for r < r_nlc with A, B fixed by value and slope continuity at r_nlc. Universal rule for r_nlc: the radius where **ρ_c(r) = ρ_v(r)** (LFC's own criterion, factor 1; the crossover is a property of the atom, not a number entered per element). In the molecule, E_xc is evaluated on ρ_v,σ + ½ρ_pc per spin (the core is spin-unpolarised); ρ_pc never enters the Hartree term, the electron count, or the displayed valence density (it is a labelled option in the renderer, §6.5). It must be smooth enough for the grid, which the Fourier filter of §5.1 guarantees.

### 4.6 Kleinman–Bylander form and the ghost-state gate (`qm::radial::kb`)

The semilocal potential V_loc + Σ_l |δV_l P_l| (δV_l = V_l^{ion} − V_loc, P_l the angular projector) is made separable (Kleinman & Bylander, *Phys. Rev. Lett.* 48, 1425 (1982)):

  V_NL = Σ_{lm} |δV_l φ_lm⟩⟨δV_l φ_lm| / ⟨φ_l|δV_l|φ_l⟩,  φ_lm = u_l^{PP}(r)/r · Y_lm(r̂) (real harmonics),

i.e. a projector β_lm = δV_l φ_lm with the KB energy E_l^{KB} = 1/⟨φ_l|δV_l|φ_l⟩. By construction V_NL reproduces the semilocal operator on the reference state of each l; on other states it is an approximation that can produce **ghost states** — spurious eigenvalues below or near the valence ones. Gonze, Stumpf & Scheffler (*Phys. Rev. B* 44, 8503 (1991)) give the criterion: with ε₀^{loc} < ε₁^{loc} the two lowest eigenvalues of the local-only radial Hamiltonian of channel l (V_loc + screening), and ε_l the reference eigenvalue: no ghost iff (E_l^{KB} < 0 and ε_l < ε₀^{loc}) or (E_l^{KB} > 0 and ε₀^{loc} < ε_l < ε₁^{loc}). Two gates, both element-independent:

- G1 (GSS criterion): must hold for every projector channel.
- G2 (direct): the radial KB atom (the same radial solver with the separable operator, which needs only the radial integral ⟨β_l|u⟩ per channel) reproduces the semilocal TM atom's valence eigenvalues to **< 1 mHa** and has **no extra bound state below ε_v,max + 0.5 Ha** in any channel (found by node counting over the energy window).

G1 is the fast check applied while scanning f and the local candidates; G2 is the proof. A failure moves to the next local candidate, then to the next f (§4.4).

### 4.7 Policy: which species get pseudopotentials

`IonicPolicy` is a scene-wide (never per-species) setting:
- `CoresOnly` (default at first): a species is pseudised iff §4.2 gives it a core; H, He (and any refused element) stay all-electron Gaussian-smeared at σ = h as today.
- `All`: every species is pseudised by the same recipe, including nodeless 1s valence (H: r_c = f · r_max,1s with r_max = 1 Bohr → r_c = 1.6 Bohr at f = 1.6 — the literature's usual pseudo-H range of 1.0–1.25 Bohr sits at f ≈ 1.0–1.25, inside the ladder). This removes σ from the whole system: H₂'s R_e and ω become grid-converged quantities instead of σ-extrapolated ones (§6.4). It is the intended default once its gates pass (§8, Task 11); until then it is a labelled switch.
- `None`: today's all-electron solver (the oracles, the gpu_vs_cpu suite, and the fallback for any species the recipe refuses).

The policy is global, so no per-element decision is ever entered; which species actually carry a potential is an *output* of §4.2.

### 4.8 Transferability gates (element-independent acceptance; rule 6)

Run on every generated potential, in the radial code, before it may be used on the grid. Thresholds are universal; a miss is a refused potential (then the next f), never a loosened threshold.

| gate | what | threshold |
|---|---|---|
| T1 eigenvalues | KB pseudo-atom vs AE, reference configuration, every valence (n, l) | < 1 mHa |
| T2 norm | ∫₀^{r_c} u² equal AE vs PP (TM's construction, as a regression) | < 1e-10 |
| T3 excitation / transfer energies | AE vs KB-PP total-energy *differences* between the reference configuration and each of: the +1 ion; the s → p promotion (one electron from the highest valence s to the lowest valence p, when both exist); the −1 anion if bound in both; the spin-polarised Hund ground multiplet (the §4.1 scan) | each ΔE within **5 mHa** |
| T4 eigenvalues in the T3 configurations | every valence eigenvalue | < 5 mHa |
| T5 log derivatives | D_l(E) = d ln u/dr at r_d = 1.2 · max_l r_c,l, AE vs KB-PP, E ∈ [ε_v,min − 1 Ha, ε_v,max + 1 Ha] on a 10 mHa mesh, compared as phase: max |atan D^{PP} − atan D^{AE}| (mod π) excluding ±25 mHa around each pole | < 0.05 rad, and every pole and zero of D^{AE} in the window has a partner in D^{PP} within 20 mHa |
| T6 hardness (Teter, *Phys. Rev. B* 48, 5031 (1993)) | ∂ε_l/∂f_l′ at ±0.25 e occupation changes, AE vs PP | reported; gated at < 10 mHa/e |
| T7 ghost | G1, G2 of §4.6 | as stated |
| T8 resolution requirement (derived output) | the spherical-Bessel transforms β̃_l(q), Ṽ_loc^{SR}(q), ρ̃_pc(q) (§5.1) fall below 1e-4 of their maximum beyond q₉₉; the potential's required spacing is h_req = π/(q₉₉ · c_q) with **c_q = 1.5** | reported in the generation record; the app warns when the scene's h > h_req |

T1–T4 compare our pseudo-atom with our all-electron atom — the honest reference (rule 7). The thresholds for T1, T3 are the brief's; T5/T6 are set here and are the ones most likely to be mis-set on first contact: if the classical TM oxygen (f ≈ 1.4) fails T5 at the first run, the task's instruction is to report, have the physics reviewer re-derive what 0.05 rad means for a molecular eigenvalue, and change the threshold only in a spec revision with that derivation — before any element-specific result depends on it.

### 4.9 Universal constants of the recipe (the complete rule-1 ledger)

| constant | value | role | why universal |
|---|---|---|---|
| radial grid r_0, r_max, δ | 1e-5 Bohr, 60 Bohr, 0.008 | §4.1 | resolves the 1s of any Z ≤ 118 (r_0 ≪ 1/Z) |
| Δ_core | 2.0 Ha | semicore promotion, §4.2 | a fixed energy window, applied to every element's own eigenvalues |
| Δ_scan | 0.1 Ha | occupation scan near degeneracies, §4.1 | same |
| l_max | max occupied valence l + 1 | channels, §4.2 | derived |
| F | {1.6, 1.4, 1.2, 1.0} | r_c ladder, §4.4 | a fixed ladder, selection by gates |
| local-candidate order | [V_{l_max}, polynomial] | §4.4 | fixed order, selection by the ghost gate |
| r_loc | min_l r_c,l | polynomial local, §4.4 | derived |
| NLCC crossover | ρ_c = ρ_v | §4.5 | LFC's criterion, derived from the atom |
| TM polynomial degree | 12 (7 coefficients) | §4.3 | the published construction |
| gate thresholds | §4.8 | acceptance | fixed numbers |
| filter q_c, γ_r | 0.7 · π/h, 1.5 | §5.1 | tied to the grid, not the element |
| c_q | 1.5 | h_req, §4.8 T8 | a fixed safety factor |
| core-overlap warning | R_IJ < r_c,I + r_c,J | §5.6 | derived radii |
| XC | LSDA-PW92 | generation = solver | the one fitted universal ingredient (design §6) |

Nothing else may be introduced. Any implementer who finds the recipe needs "a slightly different f for nitrogen" has found a design violation (CLAUDE.md rule 1): raise it, and the answer is either a universal change (a new ladder entry, with every element regenerated and re-gated) or a refusal.

`recipe_hash` = a stable hash of this table's values, the recipe version string, and the XC identity. Every generated table carries it; §5.5 threads it into the surrogate's provenance.

## 5. Integration with the real-space solver

### 5.1 The local part on the grid, and the egg-box

For a pseudised species I at R_I the local potential is split *exactly* into a long-range and a short-range part:

  V_loc^I(r) = −Z_v erf(|r − R_I|/(σ_LR√2))/|r − R_I|  +  V_SR^I(|r − R_I|),  V_SR^I(d) = V_loc^I(d) + Z_v erf(d/(σ_LR√2))/d.

The first term is precisely today's smeared nucleus with Z → Z_v (`nuclear.rs`; the GPU already obtains it from the Poisson solve of a deposited Gaussian), with **σ_LR a numerical width tied to the grid, not to the element**: σ_LR = max(2h, r_loc/3) so the Poisson route resolves it (the (h/σ)⁸ floor of `nuclear.rs`) — the choice is exactly cancelled by V_SR, so unlike today's σ it carries **no physical approximation**. V_SR is smooth (V_loc is finite with zero curvature at the origin, the erf term is analytic), equals −Z_v erfc(d/(σ_LR√2))/d beyond r_c (below 1e-6 beyond d = r_c + 5σ_LR), and is tabulated radially once per species and sampled on the grid points within that support — once per geometry, not per orbital, as `nuclear_potential` samples the erf today. The egg-box now comes from three sources: V_SR, the projectors β_lm, and ρ_pc, all localised functions of radius ~r_c sampled on a grid of spacing h.

Fourier filtering (King-Smith, Payne & Lin, *Phys. Rev. B* 44, 13063 (1991)): at generation time each radial function g(r) ∈ {V_SR, β_l, ρ_pc} is spherical-Bessel-transformed on the log grid (a 1-D quadrature — no FFT needed), truncated at **q_c = 0.7 · π/h** (0.7 of the grid Nyquist, universal) with King-Smith's smooth cutoff, transformed back, and the resulting real-space tail multiplied by a smooth mask vanishing at **γ_r · r_c, γ_r = 1.5**. The filtered functions are what the grid sees; they are slightly wider (the projector sphere radius is γ_r r_c) and band-limited so their sampling is translation-invariant to the level set by the two truncations. The filter is a function of h, so the tables are generated per (Z, recipe, h) — cached, seconds.

Plan B (Ono & Hirose, PRL 82, 5016 (1999)): evaluate ⟨β|ψ⟩ on a finer local grid (h/3) inside the projector sphere with ψ interpolated from the coarse grid — the integrand is then band-limited by construction. Kept as a documented fallback for the gate of §6.6 if filtering alone does not reach it. Expected gain of filtering alone, from the mechanism: the O egg-box today is set by a Gaussian of σ = h sampled at h (its k = 1 alias is e^{−2π²} per axis on the *charge*, but its potential and the 1s density it binds are not band-limited); with every ion-centred function cut at 0.7 of Nyquist the residual is the stencil's own translation error on the valence orbitals, the same mechanism that gives H its 4e-5 today. The §6.6 gate (10×, to 3e-5) is set below that expectation on purpose.

### 5.2 The Hamiltonian

  H_σ = −½∇² + V_loc + V_NL + V_H[ρ_v] + v_xc,σ[ρ_v,↑ + ½ρ_pc, ρ_v,↓ + ½ρ_pc],  with ρ_v = Σ_{iσ} f_iσ ψ_iσ², N_v = Σ_I Z_v,I − charge,

V_loc = Σ_I V_loc^I (LR via the Poisson solve of all Z_v Gaussians — and, for all-electron species under `CoresOnly`, their Z Gaussians at σ = h, both in one deposit — plus the sampled V_SR fields), V_NL = Σ_I Σ_lm E_l^{KB} |β_lm^I⟩⟨β_lm^I|. The all-electron species in a mixed scene are exactly today's physics; the pseudised ones add V_SR, V_NL and ρ_pc. Energy:

  E = T_s + ∫ρ_v V_loc + Σ_{iσ} f Σ_{I,lm} E_l^{KB} ⟨ψ_iσ|β_lm^I⟩² + E_H[ρ_v] + E_xc[ρ_v,↑ + ½ρ_pc, ρ_v,↓ + ½ρ_pc] + E_ii,  E_ii = Σ_{I<J} Z_v,I Z_v,J / R_IJ

(Z_v = Z for all-electron species). The same "evaluate from the output density and orbitals" discipline as `scf.rs`. `extrapolate_sigma` and the σ ladder apply to all-electron species only; with `IonicPolicy::All` there is no σ anywhere and energies are plain h-convergent quantities.

### 5.3 Applying the nonlocal operator (cost)

Per ion I: the list of grid points inside its sphere of radius γ_r r_c (rebuilt whenever R_I moves — a bounding-box scan, microseconds), with β_lm^I(r_p) evaluated from the radial table (cubic interpolation) times the real solid harmonic (the Racah-normalised real harmonics already in `qm::poisson`'s multipole code — reused, not re-derived; the surrogate crate's are not used, `qm` stays dependency-free). For each orbital: c_lm = h³ Σ_p β_lm(r_p) ψ(r_p) (n_pts ≈ (4/3)π(γ_r r_c)³/h³ ≈ 2.5 × 10³ points for r_c = 1.4, h = 0.25), then Hψ += Σ_lm E_l^{KB} c_lm β_lm. Cost per orbital per ion: 2 · n_proj · n_pts ≈ 2 · 4 · 2.5e3 ≈ 2 × 10⁴ flops (O: one s projector plus three p for a polynomial local, or just one s for p-local) against the stencil's 25 × N_grid ≈ 5 × 10⁶ at 57³ — under 1 % per ion; a 30-atom scene adds ~10 %. The c_lm are O(N_ions · n_proj · N_orb) scalars: f32 partial sums per sphere finished in f64 on the CPU exactly as `Reducer` does for every other reduction (rule 4).

### 5.4 Forces

Three terms replace `forces.rs`'s single integral for a pseudised ion (the sign chain is that module's: V depends on r − R_I, so ∂/∂R_I = −∇_r):

- local: F_I^{loc} = +∫ ρ_v ∇_r V_loc^I(r − R_I) d³r — the LR part is today's integrand with Z_v and σ_LR, the SR part uses dV_SR/dd from the table;
- nonlocal: with real orbitals, dE_NL/dR_I = 2 Σ_{iσ} f Σ_lm E_l^{KB} c_lm ⟨∂β_lm/∂R_I|ψ⟩ and ∂β/∂R_I = −∇_r β, so **F_I^{NL} = +2 Σ_{iσ} f Σ_lm E_l^{KB} c_lm ⟨∇_r β_lm^I|ψ_iσ⟩** — three more sphere reductions per (orbital, projector), with ∇β from the radial derivative and the analytic gradient of the real solid harmonic;
- NLCC: E_xc depends on R_I through ρ_pc(r − R_I), so **F_I^{pc} = +∫ ½(v_xc,↑ + v_xc,↓) ∇_r ρ_pc^I(r − R_I) d³r**;
- ion–ion: Z_v,I Z_v,J (R_I − R_J)/R_IJ³ as today.

No Pulay term, for the same reason as today: the grid does not move with the ions, and the projectors are explicit functions of R_I whose derivative is taken above. The HF-vs-finite-difference test of `forces.rs` is repeated on a pseudised molecule (§6.6) — the first failing test of Task 7.

### 5.5 What changes where

- `qm`: new `radial/` (atom, partition, tm, nlcc, kb, gates, filter, recipe) and a new `ionic.rs` with `enum Ionic { AllElectron { sigma }, Pseudo(Arc<PseudoTable>) }` per species plus `IonicSetup { policy, h, recipe_hash, per_z: … }` — the `NuclearPotential` boundary the design spec promised; `nuclear.rs` keeps the Gaussian deposit (now with Z_v) and gains the V_SR sampling; `scf.rs` builds H from an `IonicSetup` (the `System` gets `ionic`, N counts Z_v); `forces.rs` gains the three terms; `KsState` records the `IonicSetup` instead of only σ (the "converged on a different σ" guard becomes a setup-equality guard). CPU f64 reference first, as always.
- `qm-gpu`: kernels (Opus): `local_sr` (sample the tabulated V_SR per ion into the potential field; the tables in a storage buffer with per-Z offsets), `nlcc_deposit` (ρ_pc field), `proj_dot` (sphere gather + workgroup partial sums for c_lm and the three gradient components; index lists uploaded per step), `proj_axpy` (scatter E_KB c_lm β_lm into Hψ), the local-force kernel generalised to a tabulated dV/dd, and the NLCC force reduction. LOBPCG's operator gains the projector step; the preconditioner is unchanged. The Poisson route to V_LR needs no change beyond Z_v and σ_LR.
- `sim`: `Nucleus { pos, vel, z, mass }` is unchanged — still Z-only. `Scene` gains `ionic: IonicPolicy` (replacing nothing: `sigma` stays for all-electron species). The providers resolve Z → `Arc<PseudoTable>` through a `PseudoCache` keyed by (Z, recipe_hash, h), generating on first use (seconds; done on the worker thread with a status line), and refuse a scene (`ProviderError::Invalid`, like the cell margin) when a species is refused by the recipe under `All`, or when the scene's h > h_req of any table (T8). Core overlap (§5.6) is a warning in `EvalStats`, not a refusal.
- `surrogate` / `sim::SurrogateKs`: `Provenance` gains `ionic: (IonicPolicy, recipe_hash)` — a model taught on all-electron O is refused on pseudo-O and vice versa (the surfaces differ by 38 Ha and a bond angle); samples with a core-overlap warning are "used, never trained on", like box-affected ones.
- `app`: a panel radio for the policy, a badge on each species ("pseudo, Z_v = 6, r_c = 1.35 Bohr, needs h ≤ 0.33" from the generation record), the valence-energy label (§6.5), the core-overlap tint, and the README/preset updates in the same milestone (CLAUDE.md "Keep README.md current").
- `units`: the isotope table extends past Z = 10 only when Z > 10 is enabled (Task 12) — the mass is the only datum added.

### 5.6 Core overlap — the frozen-core honesty surface

Two pseudised ions closer than r_c,I + r_c,J have overlapping cores; the frozen-core and the norm-conserving construction do not describe that region, so energies and forces there are not the LSDA's. The sandbox lets users do exactly this (push atoms together, hot MD). Rule: a per-step warning "cores overlap: I–J at R = …" in `EvalStats::warnings`, a tint on the pair in the viewer, the step still taken (the potential is still finite and smooth — nothing diverges), the surrogate never trained on it. All-electron smeared species do not have this limit (and do have their own: Zσ).

## 6. Validation (rules 6, 7)

### 6.1 Unchanged

`cargo test -p validate --release --test oracles` (H, He, H₂ at σ → 0; Li ignored) and `--test gpu_vs_cpu` at fixed σ run exactly as today under `IonicPolicy::None`; they test the solver. The dynamics suite's H₂ tests likewise.

### 6.2 Generator gates (per element, element-independent)

§4.8 T1–T8 for every element the recipe accepts, as `qm::radial` tests pinned on C, N, O, F, Ne (and Be, B), with the achieved numbers recorded in `references.md`. Plus the radial AE atom against the NIST LSD totals (§4.1).

### 6.3 Grid vs radial (does the 3-D solver integrate the potential correctly?)

- Pseudo-Ne atom (closed shell, spherical, so the 3-D integer-occupation atom *is* the radial one): E_3D at h = 0.15 vs the radial KB-PP atom **< 2 mHa**, and the h-convergence E(0.35), E(0.30), E(0.25), E(0.20), E(0.15) monotone with |E(0.25) − E(0.15)| reported (the honest statement of what h = 0.25 costs with a pseudopotential — expected mHa, not Ha).
- Pseudo-O atom: the **triplet below the singlet** (Hund emerges, the M5 oracle: `gpu_oxygen_reference_multiplicity` un-ignored), splitting reported against the radial spherical-average value (not gated to it: the 3-D atom with integer occupations is non-spherical and lies a few mHa lower — documented, as the LSD H gap is).
- GPU vs CPU: `gpu_vs_cpu` extended with pseudo-O, pseudo-C systems at the same bounds (energies 2e-5-class, forces 6e-5-class, re-pinned by the same 3× rule).

### 6.4 Molecules (geometry oracles; LSDA literature, same functional)

Relaxed with quenched Verlet / the M6 relax tool at h = 0.25 (and h = 0.20 reported), `CoresOnly` (H all-electron smeared — so these mix a pseudo-O with a σ = 0.25 H) **and** `All`:

| oracle | target | bound (brief) | source |
|---|---|---|---|
| H₂O | ∠HOH ≈ 104.5–105°, r(O–H) ≈ 0.97 Å (LSDA/SVWN; the exact digits pinned at task time from the source, not quoted here unverified) | ±2°, ±0.02 Å | Johnson, Gill & Pople, *J. Chem. Phys.* 98, 5612 (1993), SVWN column; NIST CCCBDB SVWN entries |
| NH₃ | ∠HNH ≈ 106.7° (experiment; LSDA within ~1°) | ±2° | same |
| CH₄ | 109.47° (tetrahedral — a symmetry, functional-independent) | ±0.5° | symmetry |
| O atom | triplet < singlet | > 20 mHa | Hund; §6.3 |
| lone pairs | two lobes on the O isosurface of water at the published density threshold | visual, recorded in the checklist | design §8 |

With H still smeared under `CoresOnly`, the O–H length carries the H σ² error (today's H₂ R_e is 0.34 Bohr long at σ = 0.25), so the O–H bound is expected to be met only under `All` — the table records both, and the `CoresOnly` miss is the measured cost of the remaining smearing, not a pseudopotential defect. The angle is the pseudopotential's own test (it is the O's sp energetics that set it; today's 177° is the smeared O).

H₂ under `All` (pseudo-H): R_e and ω against **this solver's own σ → 0 all-electron curve** (1.451 Bohr, 4405 ± 60 cm⁻¹ in `references.md`, with that number's stated bias) — **R_e within 0.02 Bohr, ω within 5 %** — and reported against LSDA (≈ 4200) and experiment (4401). The H₂ frequency "restored to the 4300 class" is therefore a *consequence* measured against our own all-electron limit, not a target fitted to it. What changes for H under `CoresOnly`: nothing — H keeps σ = h and the 3237 cm⁻¹ surface; the dynamics suite's H₂ pins stay as they are.

### 6.5 Honesty surfaces

- Energies with pseudopotentials are **valence energies**: the readout says "E_KS (valence, pseudopotential)" and the README explains that they are not comparable to all-electron totals (the O atom reads ≈ −16 Ha, not −74.5, and both are right about different things). Energy *differences* on one policy are the physical quantities, as they are for any pseudopotential code.
- The density shown is the valence density; a labelled toggle adds ρ_pc ("with partial core — a display of the generator's core model, not solved").
- The generation record (per species: r_c,l, local channel, E_l^{KB}, f chosen, gate numbers, h_req, recipe hash) is viewable in the panel and written to `references.md` by the validation tasks.

### 6.6 Egg-box gate (the brief's 10×)

Rigid-shift probe as in `validate::surrogate` (8 sub-grid shifts of ±h/4): for H₂O at the relaxed geometry, 57³, h = 0.25, **|Σ_I F_I| < 3e-5 Ha/Bohr** (today −3.4e-4) and rms energy ripple reported; the HF-vs-FD force test at δ = 0.01 on OH and H₂O within the `forces.rs` 2e-3 bound (expected far below). The M4 COM-correction option in `sim` stays, now with a 10× smaller correction to make.

### 6.7 Dynamics

`thermostat_h2o_preset_known_limitation` becomes a gate: the `h2o` preset (physical geometry) under `All` is within 0.02 Ha of the relaxed minimum, 600 steps NVE from rest stay below 1000 K, and the warm SCF needs no cold retry (the 0.34 Ha/Bohr preset forces were the smeared surface's). NVE drift and CSVR tests re-run on pseudo-systems at the existing bounds.

## 7. Cost and risks

**Cost.** Generation: a radial atom is ~10³ log-grid points × ~10 shells × ~30 SCF steps — milliseconds; the gates (KB re-solve, ~300 log-derivative energies × channels, five configurations) and the f/local scans bring an element to seconds on one CPU thread; filtering per h adds a 1-D transform. The grid solve gets *cheaper* than today's all-electron one at equal h (O: 3 orbitals per spin instead of 4; the 1s no longer pins the LOBPCG spectrum) and h = 0.30–0.35 becomes usable where T8 allows; the nonlocal part is < 1 % per ion (§5.3). Memory: tables of ~10³ f32 per species per channel; index lists of ~3 × 10³ points per ion.

**Risks and mitigations.**
1. **Egg-box on the FD grid** (the brief's main risk): projectors and V_SR are harder than the valence orbitals. Mitigation: filter at generation (§5.1) with the gate of §6.6; plan B the Ono–Hirose double grid for the projector integrals; the probe is cheap and runs in every validation round. If both miss 10×, the honest fallback is to report the achieved factor and keep the COM correction — never a per-element softening.
2. **Ghost states**: the KB form of a p-local first-row potential is the textbook ghost case. Mitigation: G1 + G2 (§4.6) on every potential, the two-candidate local order, the f ladder; a refused element is reported, not patched.
3. **The universal r_c rule lands too hard or too soft for some element**: too hard → T8 reports an h_req the sandbox cannot afford (the honest answer: "this element needs h ≤ 0.2"); too soft → T3/T5 fail at f = 1.6 and the ladder descends. Both are *outcomes*. The residual risk is that no f passes for some Z — that element is refused at this recipe version.
4. **Spin-polarised transferability (LSDA, open shells)**: the O triplet depends on the NLCC doing its job. Mitigation: T3's Hund configuration is a gate at 5 mHa; the 3-D triplet-below-singlet oracle.
5. **Semicore / two states per l** (Li, Na, Mg): refused until ONCV (stage 3, after this milestone). Li therefore stays the ignored oracle — stated, not hidden.
6. **f32 on the GPU**: c_lm sums over ~3 × 10³ hard-function terms; β near r = 0 is O(1) while ψ is small. Mitigation: f32 partials → f64 finish (rule 4), the gpu_vs_cpu suite at the existing bounds, the Kahan path if the bound is missed (measured, not assumed).
7. **Mixed scenes** (`CoresOnly`: smeared H next to pseudo-O): correct physics for both species but two different approximations in one molecule — documented in §6.4; `All` is the clean state and the intended default.
8. **Threshold mis-set on first contact** (T5/T6): handled by the rule-6 procedure in §4.8 — a spec revision with a derivation, never a per-element exception.

## 8. Model policy and plan outline

Fable: everything whose wrong answer still looks plausible — the radial solver, the TM inversion, unscreening, KB/ghost analysis, NLCC, the filter, the three force terms, every gate and oracle. Opus: WGSL kernels against the CPU reference, buffers and dispatch, `sim`/`app`/`surrogate` plumbing, README. Each task starts from its first failing test (TDD, house pattern: CPU f64 reference before GPU).

| # | model | task | first failing test |
|---|---|---|---|
| 1 | Fable | `qm::radial::atom` — log-grid, Numerov/shooting, spin-polarised, spherical-average occupations, derived aufbau + S9 multiplicity scan, DIIS | `h_atom_radial_lsd`: −0.478671 Ha to 1e-5 (then He, Li, O triplet lowest, virial 2 to 1e-5) |
| 2 | Fable | `partition` (Δ_core rule, channels, refusals) + `tm` construction and unscreening | `tm_oxygen_2p_matches_ae_beyond_rc`: u^{PP} = u^{AE} to 1e-10 for r ≥ r_c, norm to 1e-10, inverted V finite with V″(0) = 0; `lithium_is_refused_two_s_states` |
| 3 | Fable | `kb` separable form, radial KB atom, G1/G2 ghost gates, local-candidate order, f ladder | `kb_atom_reproduces_semilocal_eigenvalues` (< 1 mHa); `deliberate_bad_local_is_caught_as_ghost` (a constructed V_loc with ε₀^{loc} above ε_s) |
| 4 | Fable | `nlcc` + transferability gates T1–T7 + the generation record and `recipe_hash` | `oxygen_transfer_energies_within_5mha` (ion, s→p, Hund); `gate_thresholds_are_constants` (the ledger of §4.9 is one `const` table, no Z-indexed value anywhere — a grep-style test) |
| 5 | Fable | `filter` (spherical-Bessel transform, q_c cut, γ_r mask) + T8 h_req | `filtered_projector_is_band_limited` (weight beyond q_c < 1e-6, tail beyond γ_r r_c < 1e-6); `o_atom_eggbox_on_grid_after_filter` (rigid-shift E rms at h = 0.25 ≤ 10× H's today) |
| 6 | Fable | `qm::ionic` + SCF integration (V_LR/V_SR split, V_NL in the operator, ρ_pc in XC, energy terms, `System` counting Z_v, `KsState` setup guard) | `pseudo_ne_3d_matches_radial` (< 2 mHa at h = 0.15); `pseudo_o_triplet_below_singlet` |
| 7 | Fable | forces: local SR/LR, nonlocal, NLCC; HF vs FD | `oh_hf_force_matches_fd` (δ = 0.01, within 2e-3, reported Richardson-separated); `h2o_sum_of_forces_below_3e-5` |
| 8 | Opus | GPU kernels `local_sr`, `nlcc_deposit`, `proj_dot`, `proj_axpy`, force kernels; `gpu_vs_cpu` extension | `gpu_pseudo_o_energy_matches_cpu` at the suite's 3×-rule bounds; kernel unit tests vs CPU on tiny grids |
| 9 | Opus | `sim`/`app`/`surrogate` plumbing: `IonicPolicy`, `PseudoCache`, provenance, panel, badges, valence-energy label, core-overlap warning, README | `provenance_refuses_model_across_policies`; `worker_generates_on_first_use_and_caches`; `scene_with_coarser_h_than_h_req_is_invalid` |
| 10 | Fable | validation oracles: §6.3–6.7 (Ne/O atoms, H₂O/NH₃/CH₄ geometries vs literature with the digits pinned from the cited sources, egg-box gate, dynamics re-pins; `references.md` and the generation records) | `h2o_angle_within_2deg_of_lsda` (fails today at 177°) |
| 11 | Fable | `IonicPolicy::All` (pseudo-H/He): gates on H, He; H₂ R_e/ω vs the own σ → 0 curve; the `h2o` preset dynamics gate; default policy decision recorded | `pseudo_h2_omega_within_5pct_of_sigma0_curve` |
| 12 | Opus | Z = 11–18 enablement: `units` isotope masses (AME2020, the only datum), presets, periodic-table picker range, README; the generator runs §6.2 on Al–Ar (Na/Mg refused → reported) | `isotope_mass_table_covers_z_up_to_18`; `al_through_ar_pass_gates_or_are_refused_with_reason` |

Order: 1 → 2 → 3 → 4 → 5 → 6 → 7 (the CPU reference, Fable throughout) → 8 ∥ 9 → 10 → 11 → 12. The physics reviewer gates 1–7 and 10–11 before they land (CLAUDE.md agents). Stage 3 (ONCV, semicore; not planned here) unblocks Li/Na/Mg.

## 9. Self-review (done before commit)

- **Placeholders:** none left as "TBD"; where a literature digit is not verified against its source here (the LSDA H₂O geometry, the NIST LSD totals for C–Ne) the spec says so and assigns the pinning to the task that writes the test, with the brief's bounds fixed now.
- **Contradictions checked:** σ in `Scene` keeps its meaning for all-electron species and is unused for pseudised ones; σ_LR is a different, numerical quantity and is named differently; H under `CoresOnly` is identical to today (§4.7, §6.4) while `All` is where the H₂ frequency changes; the dynamics H₂ pins are untouched because they run under `None`/`CoresOnly`.
- **Rule 1, paranoid pass:** every number that varies by element in this design (r_c,l, the local channel, E_KB, r_nlc, the core/valence split, the configuration, h_req, which species is pseudised) is an *output* of §4 from Z alone; the inputs are the §4.9 ledger of universal constants and Z. The two places the literature would have per-element inputs — the ground configuration and the radii — are replaced by the aufbau/energy scan and by the f ladder with gates. The eigenvalues quoted in §4.2 are illustrations of the rule, not data the code reads. Task 4's `gate_thresholds_are_constants` test makes this mechanical.
- **Rule 1, residual tension (reported, not resolved):** selecting f from a ladder by element-independent gates is an optimisation whose *result* depends on the element, as it must; it is not tuning only as long as the ladder and the gates never change for one element's sake — the §4.8 rule-6 procedure is the guard, and it depends on discipline, not on code.
- **Rule 7:** the valence-energy label, the `CoresOnly` O–H miss, the non-spherical O splitting, the refused Li/Na/Mg, and the core-overlap warning are all stated as what they are.
