//! Bond detection from the density (M6 spec §3.1, §3.3, §3.5): which pairs of nuclei are bonded,
//! read off the converged electron density at a handful of points, with the promolecule of the
//! isolated-atom references (`crate::references`) as the yardstick. Pure f64, no grid: the
//! detector asks for samples at points in rounds and the caller (the driver) fetches them
//! through `ForceProvider::sample_density`. Nothing per element (CLAUDE.md rule 1): the
//! constants below are universal, the references are solved.
//!
//! # The three conditions (spec §3.1), in atomic units (Bohr, e/Bohr³)
//!
//! For a pair (A, B) with R = |R_B − R_A| < [`R_MAX`], the chord p(s) = R_A + s (R_B − R_A) is
//! sampled at s_k = k/32, k = 0 … 32 ([`SEGMENT_SAMPLES`] = 33). Every sample is **bias-corrected**
//! (§3.3, below): ρ̂(p) = ρ(p)·ρ_pro(p)/ρ_pro^tri(p), and the deformation ratio is
//! q(p) = ρ̂(p)/ρ_pro(p) = ρ(p)/ρ_pro^tri(p), where ρ_pro = Σ_C ρ̄_{Z_C}(|p − R_C|) over **all**
//! nuclei of the scene (ruling P5: with the pair's own atoms only, the H–H chord of water
//! compares oxygen's valence density with 2ρ_H and finds a bond).
//!
//! 1. **A bond point.**
//!    - **Route S (saddle):** the interior minimum of ρ̂ (k = 1 … 31) lies below both endpoints
//!      → p* is there. Across the chord, for the unit vectors e₁, e₂ ⊥ u = (R_B − R_A)/R,
//!      ρ̂(p* ± h e_k) ≤ (1 + ε_⊥) ρ̂(p*) ([`EPS_TRANSVERSE`]). A rise above that is followed
//!      before the pair is rejected (the **multi-centre march**, ruling P7 / R-1): a gradient
//!      ascent re-steered at every step. First direction g_⊥ = Σ_k [ρ̂(p* + h e_k) − ρ̂(p* − h e_k)]
//!      e_k from the four probes; each step of h/2 then re-estimates ∇ρ̂ by central differences
//!      from ±h/2 probes on the three lab axes (plus the point itself) and turns along it. It
//!      stops at the first of, in this order: (i) the next point lies within h of any nucleus →
//!      the rise was a neighbour's hill → `NotATransverseMaximum`; (ii) ρ̂ decreased → a
//!      non-nuclear maximum; (iii) |∇ρ̂| < f_g·g₀ ([`MARCH_STATIONARY_FRACTION`] of the gradient
//!      at the first march step) → a non-nuclear stationary point; (iv) 2R_max/h steps →
//!      `NotATransverseMaximum`. Ends (ii) and (iii) keep the pair, flagged `multi_centre`.
//!    - **Route D (deformation, MUST-4):** no interior minimum below both endpoints (at σ = h a
//!      heavy nucleus's smeared tail runs monotonically down to a hydrogen, the saddle degenerates
//!      onto the H nucleus). Then p* is the interior maximum of q along the chord — it must exceed
//!      both endpoint values, else `NoInteriorMinimum` — and the transverse test is applied to q.
//!      No march on this route (the spec defines none).
//!    Route S whenever an interior minimum of ρ̂ exists, Route D only otherwise.
//! 2. **A floor.** ρ̂(p*) ≥ ρ_floor ([`RHO_FLOOR`] = 1e-2): below Bader's closed-shell range
//!    nothing is a bond.
//! 3. **Accumulation.** q(p*) ≥ 1 + δ ([`DELTA`] = 0.11): the atoms put more density between
//!    them than they would merely sitting next to each other (He₂ depletes it).
//! 4. **Attribution** (M6 Task 7, from the real water surface): at p* the pair's own atoms are the
//!    two largest terms of the promolecule — no third nucleus C has ρ̄_C(|p* − R_C|) above
//!    min(ρ̄_A(|p* − R_A|), ρ̄_B(|p* − R_B|)) — else `NeighbourRegion { atom: C }`. Checked last,
//!    only for a pair that passed 1–3.
//!
//! **Why condition 4, and why this form.** On the real smeared water surface (57³, σ = h = 0.25)
//! the H–H chord has no interior minimum (oxygen's valence fills its middle), so it takes Route D,
//! and its q has an interior maximum 0.18 Bohr from one H (q 1.43, transverse ratio on q 1.02,
//! ρ̂ 0.25): every one of conditions 1–3 passes and the first detector reported an H–H bond.
//! That maximum is the O–H bond's own deformation lobe: the O–H chord finds its q maximum
//! 0.23 Bohr from the same H (q 1.47), the two chords leave the H 38° apart, and the H–H chord
//! passes 0.139 Bohr — less than one grid spacing — from that point (the two points are 0.14 Bohr
//! apart in space; the H–H point is 0.11 Bohr off the O–H axis), so a transverse test at
//! ±h (2 % rise against ε_⊥ = 7 %) cannot see that it is off-axis. What tells the two apart is
//! whose density is there: at the H–H point the promolecular terms are H₁ 0.111, O 0.064, H₂
//! 0.002 — the far H contributes a thirtieth of oxygen's — while at the O–H point they are H₁
//! 0.107, O 0.080, H₂ 0.002. A bond point of A–B lies on the interatomic surface where A's and B's
//! basins meet (Bader 1990); the promolecule's stockholder shares w_C = ρ̄_C/ρ_pro (Hirshfeld,
//! Theor. Chim. Acta 44, 129 (1977)) are the parameter-free estimate of whose space a point is,
//! and a point where a third atom outweighs the weaker partner is in that atom's space — the
//! feature there belongs to a bond of that atom's, seen from the side. The same rule rejects the
//! chord of a linear A–B–C's end atoms, whose interior minimum is an A–B bond point (the
//! second-nearest contributor there is B, not C). No constant: two numbers the detector already
//! has are compared, rotation-invariant, and the real-surface margins are a factor 30–45 both
//! ways. **Not an exclusion radius about each nucleus** (the first idea): the legitimate O–H
//! point lies 0.23 Bohr from H and the spurious H–H point 0.18 — their distances from H differ by
//! 0.05 Bohr, both inside one smearing width σ — so any radius separating them would be fitted to
//! water (rule 1/6).
//! **Not the singlet O reference:** scaling the O table by 0.9 or 1.1 leaves the H–H verdict
//! Bonded (q 1.49 / 1.37). Checked last so the verdicts of 1–3 are reported as before: the
//! water-like fixture's H–H pair is still rejected by its march into O.
//!
//! **The transverse basis** e₁, e₂ is not fixed by the spec; it is chosen here by the geometry so
//! that the test is rotation-covariant: e₁ is the component ⊥ u of the direction from the chord
//! to the centroid of the *other* nuclei — where a neighbour's hill or a ring point lies — and
//! e₂ = u × e₁; with no other nucleus, or one on the chord's line, the lab construction
//! (u × the axis least aligned with u). With lab-fixed probes the measured rise of a directional
//! feature varies by up to cos 45° between orientations, which would make `multi_centre` depend
//! on how the molecule is turned.
//!
//! # The same operator on both sides (§3.3, MUST-1)
//!
//! Trilinear interpolation of a tail e^{−κr} on a grid of spacing h errs by +κ²h²/8 at a cell
//! midpoint (3 % for H, 9 % for He at h = 0.25) and depends on where the point sits in its
//! cell. A ratio of two such samples would carry that. So [`Promolecule::trilinear`] applies the
//! provider's own operator — the eight grid nodes around p blended with the weights of
//! `qm::Grid::trilinear`, mirrored here in [`trilinear_cell`] — to the promolecule, and
//! ρ̂ = ρ·exact/trilinear (ρ unchanged where trilinear = 0: outside the grid). The leading bias
//! cancels; what remains is Σ_a (h²/2)·t_a(1 − t_a)·D_aa with D_aa = ∂²_aρ/ρ − ∂²_aρ_pro/ρ_pro
//! over the lab axes (trilinear interpolation errs by the lab-diagonal second derivatives only;
//! its mixed terms are exact), second order in the deformation — between a node and a cell
//! centre at most (h²/8)·Σ_a|D_aa|, which is (h²/8)·|∇²ρ/ρ − ∇²ρ_pro/ρ_pro| only when every
//! D_aa has one sign (review M3). Measured on the gridded bonded-pair fixture (the test
//! `the_same_operator_cancels_the_interpolation_bias`; `validate/references.md`, "Tools (M6)").
//! For samples that were never interpolated (an analytic provider, the exact fixtures) the
//! promolecule is built without a grid and the operator is the identity.
//!
//! # Constants (every one universal; derivations in `references.md`)
//! M = 33 chord samples (h/2-ish spacing on a 4 Bohr chord); ε_⊥ = 0.07 and δ = 0.11, each ≥ 3×
//! the measured fixture noise (transverse ratio spread 0.018, corrected q spread 3.4 % of its
//! mean — the gridded bonded pair over 24 placements; the spec's provisional δ = 0.10 was raised
//! by that measurement under its own ≥ 3× rule); ρ_floor = 1e-2 (two H 5 Bohr apart:
//! ρ_mid ≈ 4e-3, not bonded; 2.5 Bohr apart: 0.05, a candidate); R_max = 8 Bohr (beyond it no
//! two atoms in range share density above the floor); the march step h/2 with the "within h of
//! a nucleus" rule and f_g = 0.25 (a ring point whose in-plane curvature is nearly flat: the
//! gradient falls to a quarter of its first value within ~0.1 Bohr of the point for a quadratic
//! top starting 0.5 Bohr away).
//!
//! # What this is not
//! Not QTAIM: no gradient-path integration, no basins, no bond order. The chord is a chord of the
//! true bond path; the march is a one-dimensional probe of the ring/cage point, not a critical-
//! point search. Ionic and charged closed-shell bonds are the open question (spec §3.6).
//!
//! # Bond strength: the sign of ∇²ρ at the bond point (user request, M6)
//! The verdict above is binary; a stretched contact that passes it looks like a covalent bond. Each
//! bonded (and each `NotAccumulated`) pair therefore also gets the Laplacian of the density at its
//! bond point, ∇²ρ_b = Σ_a ∂²ρ/∂x_a² (e/Bohr⁵, lab axes, the sum is rotation-invariant), and the
//! QTAIM classification by its sign (Bader & Essén, J. Chem. Phys. 80, 1943 (1984); Bader, *Atoms
//! in Molecules: A Quantum Theory*, Oxford 1990, ch. 7):
//! - **∇²ρ_b < 0: shared shell** ([`BondKind::Shared`], covalent): charge is concentrated between
//!   the nuclei. By the local virial theorem ¼∇²ρ = 2G + V (atomic units; G > 0 the kinetic, V < 0
//!   the potential energy density; Bader 1990 eq. 6.31) the potential energy dominates there.
//! - **∇²ρ_b > 0: closed shell** ([`BondKind::ClosedShell`]): charge is depleted at the point,
//!   kinetic energy dominates — ionic, hydrogen-bonded, van der Waals, noble-gas and stretched
//!   contacts.
//! - **|∇²ρ_b| below the noise band: [`BondKind::Indeterminate`]** (below) — the sign is not
//!   resolved on this grid, so it is not guessed.
//! No threshold enters the classification itself: the only number is the measured noise band.
//! At a true bond critical point ∇²ρ = λ₁ + λ₂ + λ₃ with λ₁, λ₂ < 0 across the bond and λ₃ > 0
//! along it. An estimate of where the sign turns: a doubly occupied bonding orbital φ_A + φ_B of
//! 1s-like tails e^{−κr}, R apart, has ρ ∝ (φ_A + φ_B)² ∝ cosh²(κx) along the chord and
//! ∝ 1 − 2κy²/R across it at the midpoint, so ∇²ρ/ρ = 2κ² − 8κ/R: negative (shared) for short R,
//! positive past R ≈ 4/κ (≈ 4 Bohr for hydrogen, κ = 1). Measured (`references.md`, "Bond
//! strength"): on this solver's spin-symmetric H₂ the orbital is more diffuse and ∇²ρ_b reaches
//! zero only at R ≈ 5.0, at about the same R as ρ_b meets the floor (the scan resolves the
//! crossing to ≈ ±0.25 Bohr); in linear H₃ the long contact's ∇²ρ_b crosses zero at R₂ ≈ 2.68,
//! after the detector has dropped it — no bonded closed-shell window in either. What marks those
//! contacts as weaker is ρ_b (the display's width and opacity) — a modest cue: ρ_b is LSDA's own
//! density, inflated by the same delocalisation error.
//!
//! **Where it is evaluated.** At the detector's own bond point p* (the chord sample of Route S's
//! minimum or Route D's q maximum). On Route D p* is not a critical point of ρ but the deformation
//! maximum, so there ∇²ρ_b is the Laplacian at the point the detector calls the bond point, not
//! at a QTAIM bond critical point; the sign is still the local concentration/depletion of charge
//! between the nuclei, but the number is not comparable with a tabulated ∇²ρ_b.
//!
//! **The operator, and why this one.** The solver's own discrete Laplacian
//! ([`LAPLACIAN_COEFF`], the 8th-order Fornberg stencil of `qm::stencil`, Dirichlet ρ = 0 outside
//! the box — what `sample_density` returns there) at grid nodes:
//!   L(x_n) = h⁻² Σ_a Σ_{m=−4}^{4} c_|m| ρ(x_n + m h ê_a),
//! read through `sample_density` at the nodes themselves (where trilinear interpolation returns
//! the stored value), and **trilinearly blended over the eight nodes of p*'s cell** with the
//! weights of [`trilinear_blend`] — the operator the detector already applies to ρ (25 samples per
//! node; one extra round after the verdicts, ≤ 1200 samples per pair with the noise estimate
//! below). Not the nearest node, because (i) the blend is
//! continuous in p*, so the kind and the drawn line do not flicker as a vibrating molecule's bond
//! point crosses a cell face, while the nearest node jumps by ≈ h|∇L| there; (ii) its error is
//! second order, ≤ (h²/8)·Σ_a|∂²_a L| (trilinear interpolation of the nodal values), against the
//! nearest node's first-order (√3h/2)|∇L| — which matters for heteronuclear bonds, where ∇L ≠ 0 at
//! p* (for a homonuclear midpoint ∇L = 0 by symmetry and both are second order). The stencil's own
//! truncation, ≈ 3.2e-4·h⁸·ρ⁽¹⁰⁾ per axis, is negligible beside it. With exact samples (no grid:
//! the analytic providers) the same stencil is centred on p* with spacing h. No bias correction
//! as for ρ̂: the promolecule's exact Laplacian would need the second derivative of a tabulated
//! radial profile; the residual is estimated instead (the noise, below). f32 rounding of the GPU
//! density (2⁻²⁴·ρ per value) gives ≤ 19.5·6e-8·ρ/h² ≈ 1.6e-5 e/Bohr⁵ for ρ = 0.3 at h = 0.15
//! (Σ|c| = 19.505 over the 25 points).
//!
//! **The noise.** What δ's "noise" measures is the placement spread: the same molecule at
//! different sub-cell offsets gives different numbers because the interpolation residual depends
//! on where the point sits in its cell. For ∇²ρ_b that residual has a closed form, so it is
//! estimated **at every bond point from the samples themselves** ([`LaplacianEstimate`]) instead
//! of carried as one fixture-measured constant:
//! - **interpolation:** the blend errs by −Σ_a (h²/2)·t_a(1 − t_a)·∂²_a L + O(h⁴); its largest
//!   value over the cell (t = ½) is (1/8)·Σ_a |Δ²_a L|, with Δ²_a L the second difference of the
//!   nodal Laplacians along a — read from a 32-node block (the 8 corners and each corner's outward
//!   neighbour on each axis; 800 samples; in a cell on a low face of the box the −1 nodes lie
//!   outside and read 0, which inflates the term there — the safe direction);
//! - **location:** p* is resolved to one chord sample (R/32), so the largest change of the blended
//!   L to the neighbouring samples p_{k±1} (2 × 200 samples);
//! - **rounding:** 2⁻²⁴ × 19.505 × max|ρ| / h² (f32 density, [`stencil_abs_sum`]) — a floor, not
//!   a bound: the f32 errors of an orbital-summed ρ are not independent per node.
//!
//! The band is [`LAPLACIAN_SAFETY`] = 3 × their sum; |∇²ρ_b| at or below it → `Indeterminate`.
//! **This is not δ's rule exactly, and is more conservative:** δ is 3× a *measured* spread; here
//! the 3× multiplies an upper *estimate* that is already ≈ 3.5× the actual error on the fixture
//! (|∇²ρ_b − exact| ≤ 0.29 × the estimated noise at all 24 placements, h = 0.25 and 0.125;
//! `the_laplacian_at_the_bond_point_converges_and_its_noise_is_measured`). In numbers: on the
//! fixture at h = 0.25 the mean band is 1.12 against δ-style 3 × spread = 0.60; on the real H₂ at
//! 1.45 Bohr (81³ × 0.15, 12 placements) 0.29 against 0.146 — ≈ 2× wider than δ's rule would give,
//! erring toward "not resolved". The estimate does **not** include the density's own errors —
//! SCF convergence and the egg-box (grid-placement) error of the solved ρ; those are covered only
//! empirically, by the 12-placement H₂ checks of the tools suite (R = 1.45 at h = 0.15, 1.79 at
//! 0.25: spread ≤ 0.17 of twice the largest estimated noise).
//!
//! **Why per point and not one constant ν·ρ_b·h² measured on the fixture, as δ.** The fixture's
//! placement spread of ∇²ρ_b/(ρ_b h²) is 11.7 (h = 0.25) and 13.1 Bohr⁻⁴ (h = 0.125), a 3× band
//! of 39.3·ρ_b·h². Against the per-point band on the measured cases:
//!
//! | case | grid | ρ_b | \|∇²ρ_b\| | constant band 39.3·ρ_b·h² | per-point band | kind (constant / per-point) |
//! |---|---|---|---|---|---|---|
//! | H₃ long contact R₂ = 2.0 | 107³ × 0.15 | 0.104 | 0.179 | 0.092 | 0.090 | shared / shared |
//! | H₃ R₂ = 2.25 | 107³ × 0.15 | 0.077 | 0.073 | 0.068 | 0.054 | shared / shared |
//! | H₃ R₂ = 2.5 | 107³ × 0.15 | 0.057 | 0.020 | 0.050 | 0.034 | indet. / indet. |
//! | H₃ R₂ = 2.55 | 107³ × 0.15 | 0.054 | 0.013 | 0.048 | 0.031 | indet. / indet. |
//! | H₂ R = 4.0 | 107³ × 0.15 | 0.022 | 0.0076 | 0.020 | 0.0011 | indet. / shared |
//! | H₂ R = 4.5 | 107³ × 0.15 | 0.015 | 0.0022 | 0.013 | 0.0005 | indet. / shared |
//! | H₃ R₂ = 3.0 (`NotAccumulated`) | 107³ × 0.15 | 0.033 | 0.017 | 0.029 | 0.014 | indet. / closed |
//! | H₃ R₂ = 3.25 (`NotAccumulated`) | 107³ × 0.15 | 0.025 | 0.0205 | 0.022 | 0.0096 | indet. / closed |
//! | H₂O O–H | 95³ × 0.15 | 0.342 | 2.807 | 0.30 | 2.731 | shared / shared (margin 1.03) |
//! | HeH⁺ | 49³ × 0.25 | 0.271 | 1.616 | 0.67 | 1.756 | shared / indet. |
//!
//! The two agree on every drawn H–H bond of the user's case (same kinds); they differ near the
//! floor, where the constant is wider (H₂ at R = 4.0–4.5, and H₃'s `NotAccumulated` contact at
//! 3.0–3.25), and at asymmetric Route-D points, where the per-point band is 2–9× **wider** (O–H,
//! HeH⁺). That last row is the reason: the fixture is symmetric — its bond point is a midpoint
//! where L is stationary along the chord — so it cannot see the location error at an asymmetric
//! (Route D) bond point, where L is steep; and its ρ_b·h² scaling carries the fixture's length
//! scale everywhere. The per-point estimate adapts to the local curvature and slope of L at each
//! bond point. The 8th-order stencil's own truncation is not in the
//! estimate: with exact samples at h = 0.25 it is 4.4e-4 on the analytic fixture of
//! `the_bond_kind_follows_the_sign_of_the_laplacian` (|∇²ρ_b| = 0.22; the derived bound 1.9e-2),
//! and real densities are smoother than its w = 0.5 Bohr Gaussians.
//!
//! **Smeared nuclei.** The nuclei are Gaussians of width σ = h (`qm::nuclear`), which removes the
//! cusp and flattens ρ within ≈ 2σ of a nucleus. A homonuclear H–H midpoint is R/2 from each
//! nucleus (0.725 Bohr = 4.8σ for H₂ at 1.45 on the Accurate grid, σ = 0.15): outside the
//! smearing. Water's O–H bond points take Route D and sit near the hydrogen, far from the
//! oxygen (measured in the tools suite, `references.md`, "Bond strength"): at the default grid
//! (95³ × 0.15, the `h2o` preset geometry) 0.339 Bohr = 2.3σ from H and 1.471 Bohr = 9.8σ from O;
//! at 57³ × 0.25, 0.23–0.34 Bohr = 0.9–1.4σ from H and 5.9–6.3σ from O. So the oxygen's smearing
//! does not reach the point, the hydrogen's does (≈ 2σ): the point lies where even an isolated H
//! atom's own density is a concentration (∇²ρ < 0 within 1 Bohr of a free 1s nucleus, ρ ∝ e^{−2r}:
//! ∇²ρ = ρ(4 − 4/r)) and where ∇²ρ changes steeply along the chord. **It matters for the number,
//! not the sign:** ∇²ρ_b is negative at every O–H point measured (−1.50 … −2.02 at h = 0.25,
//! −2.81 at 0.15), the standard QTAIM verdict for O–H, but it mixes the bond with the hydrogen's
//! own concentration and is not comparable with literature values; and its noise is as large as
//! a third of it (location ≈ 0.1–0.46 and interpolation ≈ 0.4–0.5 e/Bohr⁵), so |∇²ρ_b|/band is
//! 0.93–1.03 — shared at 6 of 7 measured points, indeterminate at one. The display may therefore
//! switch an O–H line between solid and dotted as water vibrates. **This is not a smearing
//! artefact:** at h = 0.15 the point is 2.3σ from H, which encloses erf(1.63) ≈ 0.98 of the
//! hydrogen's Gaussian charge (its potential is within ≈ 2 % of a point charge's there). The
//! location term is intrinsic to hydrogen's density slope — dL/dr ≈ 8.1 e/Bohr⁶ at r = 0.339
//! times the chord step 1.81/32 = 0.057 Bohr gives 0.46, the measured 0.465 — and a sharper,
//! unsmeared hydrogen would make it larger: neither a smaller σ nor pseudopotentials (M7) remove
//! it. **Known follow-up, not done in M6:** a refined bond point — the parabolic vertex of q
//! (Route D) or ρ̂ (Route S) through the samples k − 1, k, k + 1, with L evaluated there — would
//! cut the location term. No hysteresis on the kind (it would need an underived margin and make
//! the display depend on history).
//!
//! **LSDA.** The sign reads the density this solver produces. LSDA's delocalisation
//! (self-interaction) error spreads density across stretched contacts (H…H₂ is far too attractive
//! in LSDA), so a weak or closed-shell contact here may be weaker or absent in reality; the
//! display says so.
//!
//! # Molecules (§3.5)
//! A molecule is a connected component of the bond graph ([`components`], union–find); isolated
//! nuclei are one-atom components. **Closed-shell and indeterminate contacts still join
//! components** (the grouping is by the binary verdict only; naming is unchanged by the bond
//! strength).
use crate::references::{AtomReferences, RadialDensity, ReferenceKey};
use crate::scene::Cell;

/// Chord samples per pair (s = k/32).
pub const SEGMENT_SAMPLES: usize = 33;
/// The transverse rise a bond point may show before it is followed (module doc).
pub const EPS_TRANSVERSE: f64 = 0.07;
/// The density floor at the bond point (e/Bohr³).
pub const RHO_FLOOR: f64 = 1e-2;
/// The accumulation margin: q(p*) ≥ 1 + δ. The spec's provisional 0.10 (≥ 3 × the re-review's
/// 2.8 % noise), raised to 0.11 by this task's measurement on the gridded fixture (q spread
/// 0.0629 on a mean of 1.852: 3 × 3.4 % = 0.102 — the test
/// `the_same_operator_cancels_the_interpolation_bias`; `references.md`, "Tools (M6)").
pub const DELTA: f64 = 0.11;
/// Pairs farther apart than this (Bohr) are not sampled.
pub const R_MAX: f64 = 8.0;
/// The march stops when |∇ρ̂| falls below this fraction of its value at the first march step.
pub const MARCH_STATIONARY_FRACTION: f64 = 0.25;
/// The solver's 8th-order central second-derivative coefficients c₀ … c₄ (c₋ₘ = cₘ; Fornberg
/// 1988) — a mirror of `qm::stencil::COEFF` (no `qm` in this library; asserted equal in the
/// tests), for ∇²ρ at the bond point (module doc, "Bond strength").
pub const LAPLACIAN_COEFF: [f64; 5] = [-205.0 / 72.0, 8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0];

/// Which rule placed the bond point (module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    Saddle,
    Deformation,
}

/// What the detector concluded for a pair (module doc; `Bonded` is the only accepting verdict).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Bonded { multi_centre: bool },
    TooFar,
    NoInteriorMinimum,
    NotATransverseMaximum,
    BelowFloor,
    NotAccumulated,
    /// Condition 4 (module doc, "Attribution"): at the bond point the promolecular density of
    /// nucleus `atom` — neither of the pair — exceeds the weaker of the pair's own, so the
    /// feature lies in that atom's share of space: another pair's bond seen from the side.
    NeighbourRegion { atom: usize },
}

/// Why a march stopped (module doc, the four stops).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarchStop {
    Nucleus,
    Maximum,
    Stationary,
    StepCap,
}

/// Every sampled pair with the numbers behind its verdict (MUST-5): `s` the bond point's chord
/// fraction, `rho_min` = ρ̂(p*) (the chord minimum on Route S, the density at the q maximum on
/// Route D), `transverse_ratio` the largest probe/bond-point ratio (of ρ̂ on Route S, of q on
/// Route D), `rho_pro` = ρ_pro(p*), `enhancement` = q(p*); `march_end`/`march_stop` where and why
/// a march stopped. Each is `None` when the pair never reached that stage.
#[derive(Clone, Debug, PartialEq)]
pub struct PairReport {
    pub i: usize,
    pub j: usize,
    pub route: Option<Route>,
    pub s: Option<f64>,
    pub rho_min: Option<f64>,
    pub transverse_ratio: Option<f64>,
    pub rho_pro: Option<f64>,
    pub enhancement: Option<f64>,
    pub march_end: Option<[f64; 3]>,
    pub march_stop: Option<MarchStop>,
    pub verdict: Verdict,
    /// ∇²ρ at the bond point (e/Bohr⁵; module doc, "Bond strength") — computed for `Bonded` and
    /// `NotAccumulated` pairs only.
    pub laplacian: Option<f64>,
    /// Its band, [`LaplacianEstimate::band`] (e/Bohr⁵): |∇²ρ_b| at or below it is
    /// `Indeterminate`. Alongside `laplacian`.
    pub laplacian_noise: Option<f64>,
}

impl PairReport {
    /// The QTAIM kind at the bond point, when the Laplacian was computed.
    pub fn kind(&self) -> Option<BondKind> {
        Some(BondKind::classify(self.laplacian?, self.laplacian_noise?))
    }
}

/// The QTAIM classification of a bond point by the sign of ∇²ρ_b (module doc, "Bond strength";
/// Bader & Essén 1984).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BondKind {
    /// ∇²ρ_b < 0: charge concentrated between the nuclei (covalent).
    Shared,
    /// ∇²ρ_b > 0: charge depleted at the bond point (weak, stretched, ionic-like contact).
    ClosedShell,
    /// |∇²ρ_b| within its noise band (or not finite): the sign is not resolved on this grid.
    Indeterminate,
}

impl BondKind {
    /// Shared below −`noise`, closed-shell above +`noise`, else (and for a non-finite Laplacian)
    /// indeterminate.
    pub fn classify(laplacian: f64, noise: f64) -> BondKind {
        if !laplacian.is_finite() || !noise.is_finite() || laplacian.abs() <= noise {
            BondKind::Indeterminate
        } else if laplacian < 0.0 {
            BondKind::Shared
        } else {
            BondKind::ClosedShell
        }
    }
}

/// The accepted subset of the reports.
#[derive(Clone, Debug, PartialEq)]
pub struct Bond {
    pub i: usize,
    pub j: usize,
    /// ρ̂ at the bond point (e/Bohr³).
    pub rho_bcp: f64,
    pub enhancement: f64,
    pub s: f64,
    pub route: Route,
    pub multi_centre: bool,
    /// ∇²ρ at the bond point (e/Bohr⁵; module doc, "Bond strength"); NaN if it could not be
    /// evaluated (then [`Bond::kind`] is `Indeterminate`).
    pub laplacian_bcp: f64,
    /// The band of `laplacian_bcp` (e/Bohr⁵): [`LAPLACIAN_SAFETY`] × its estimated noise
    /// ([`LaplacianEstimate::band`]).
    pub laplacian_noise: f64,
}

impl Bond {
    /// Shared (covalent), closed-shell (weak / stretched contact) or indeterminate — the sign of
    /// ∇²ρ_b against its noise band.
    pub fn kind(&self) -> BondKind {
        BondKind::classify(self.laplacian_bcp, self.laplacian_noise)
    }
}

/// The bonds and the molecules of one evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct BondGraph {
    pub bonds: Vec<Bond>,
    pub components: Vec<Vec<usize>>,
    pub step: u64,
    pub sim_time: f64,
    pub n_nuclei: usize,
}

// ---- small vector helpers ----

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm(sub(a, b))
}

// ---- the grid operator (mirror of qm::Grid::trilinear_cell / trilinear) ----

/// The trilinear cell of `p` on `cell`'s grid (`qm::Grid::cube(n, h)`: origin −(n−1)h/2 on each
/// axis): the lower corner ⌊x⌋ and the fractions t = x − ⌊x⌋ ∈ [0, 1) per axis, x = (p −
/// origin)/h; on an upper face the corner is n − 1 with t = 0. `None` outside the grid (NaN
/// included). The same definition as `qm::Grid::trilinear_cell` (asserted equal in the tests).
pub fn trilinear_cell(cell: &Cell, p: [f64; 3]) -> Option<([usize; 3], [f64; 3])> {
    let origin = -cell.half_extent();
    let mut c = [0usize; 3];
    let mut t = [0.0f64; 3];
    for a in 0..3 {
        let x = (p[a] - origin) / cell.h;
        if !(x >= 0.0 && x <= (cell.n - 1) as f64) {
            return None;
        }
        let i0 = x.floor();
        c[a] = i0 as usize;
        t[a] = x - i0;
    }
    Some((c, t))
}

/// The position of grid node (i, j, k) of `cell`.
pub fn node_position(cell: &Cell, ijk: [usize; 3]) -> [f64; 3] {
    let origin = -cell.half_extent();
    ijk.map(|i| origin + i as f64 * cell.h)
}

/// The 25 points of the 8th-order Laplacian stencil centred on `c` with spacing `h`: c, then per
/// axis a the pairs c ± m h ê_a for m = 1 … 4 (+ first).
pub fn laplacian_stencil(c: [f64; 3], h: f64) -> [[f64; 3]; 25] {
    let mut out = [c; 25];
    for a in 0..3 {
        for m in 1..=4usize {
            let k = 1 + 8 * a + 2 * (m - 1);
            out[k][a] += m as f64 * h;
            out[k + 1][a] -= m as f64 * h;
        }
    }
    out
}

/// ∇² from the 25 samples of [`laplacian_stencil`] at spacing `h`: h⁻² [3c₀ f₀ + Σ_a Σ_m cₘ (f₊ +
/// f₋)] — `qm::stencil::laplacian`'s sum at one node.
pub fn laplacian_from_stencil(f: &[f64], h: f64) -> f64 {
    assert_eq!(f.len(), 25, "laplacian_from_stencil: {} samples", f.len());
    let mut acc = 3.0 * LAPLACIAN_COEFF[0] * f[0];
    for a in 0..3 {
        for m in 1..=4usize {
            let k = 1 + 8 * a + 2 * (m - 1);
            acc += LAPLACIAN_COEFF[m] * (f[k] + f[k + 1]);
        }
    }
    acc / (h * h)
}

/// The sample points for ∇²ρ at `p` (module doc, "The operator"): on a grid, the stencils of the
/// eight nodes of p's cell (25 each, corner order (di, dj, dk) with di fastest; a corner beyond an
/// upper face — weight 0 — still gets its points, they read as 0); without a grid, the one stencil
/// centred on `p` at spacing `h`. `None` when `p` is outside the grid.
pub fn laplacian_points(grid: Option<&Cell>, p: [f64; 3], h: f64) -> Option<Vec<[f64; 3]>> {
    match grid {
        None => Some(laplacian_stencil(p, h).to_vec()),
        Some(cell) => {
            let (c, _) = trilinear_cell(cell, p)?;
            let origin = -cell.half_extent();
            let mut out = Vec::with_capacity(8 * 25);
            for dk in 0..2 {
                for dj in 0..2 {
                    for di in 0..2 {
                        let node = [c[0] + di, c[1] + dj, c[2] + dk].map(|i| origin + i as f64 * cell.h);
                        out.extend_from_slice(&laplacian_stencil(node, cell.h));
                    }
                }
            }
            Some(out)
        }
    }
}

/// ∇²ρ at `p` from the samples at [`laplacian_points`]: the nodal Laplacians blended with the
/// trilinear weights of p's cell (grid), or the one stencil (no grid).
pub fn laplacian_at(grid: Option<&Cell>, p: [f64; 3], h: f64, samples: &[f64]) -> f64 {
    match grid {
        None => laplacian_from_stencil(samples, h),
        Some(cell) => {
            assert_eq!(samples.len(), 200, "laplacian_at: {} samples for the 8 nodal stencils", samples.len());
            let Some((c, _)) = trilinear_cell(cell, p) else { return f64::NAN };
            let nodal: Vec<f64> = samples.chunks(25).map(|f| laplacian_from_stencil(f, cell.h)).collect();
            trilinear_blend(cell, p, |ijk| {
                let d = [ijk[0] - c[0], ijk[1] - c[1], ijk[2] - c[2]];
                nodal[d[0] + 2 * d[1] + 4 * d[2]]
            })
        }
    }
}

/// Σ|w| over the 25 weights of one stencil, in units of h⁻²: 3|c₀| + 6 Σ_{m≥1} |c_m| = 19.505 —
/// the gain of the stencil on independent rounding errors of its samples.
pub fn stencil_abs_sum() -> f64 {
    3.0 * LAPLACIAN_COEFF[0].abs() + 6.0 * LAPLACIAN_COEFF[1..].iter().map(|c| c.abs()).sum::<f64>()
}

/// The 32 node offsets (from the lower corner of p's cell) of the strength block: the eight
/// corners d ∈ {0, 1}³ (x fastest), then for each axis a the outward neighbour of each corner
/// along a (d_a → −1 or 2), corners in the same order — so every corner has three nodes on each
/// axis line through it, enough for the second difference of the nodal Laplacian.
fn block_offsets() -> [[i64; 3]; 32] {
    let mut out = [[0i64; 3]; 32];
    for ci in 0..8 {
        let d = [(ci & 1) as i64, ((ci >> 1) & 1) as i64, ((ci >> 2) & 1) as i64];
        out[ci] = d;
        for a in 0..3 {
            let mut o = d;
            o[a] = if d[a] == 0 { -1 } else { 2 };
            out[8 + 8 * a + ci] = o;
        }
    }
    out
}

/// ∇²ρ at a bond point with its noise (module doc, "The noise"). `value` is the blended nodal
/// Laplacian at p*; the three noise terms are each an estimate of |value − ∇²ρ(true bond point)|
/// from one source; [`LaplacianEstimate::band`] is [`LAPLACIAN_SAFETY`] × their sum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LaplacianEstimate {
    pub value: f64,
    /// Interpolation: (1/8) Σ_a max over the cell's axis lines of |Δ²_a L| (the trilinear blend's
    /// leading error (h²/2)·t(1 − t)·∂²_a L at its worst, t = ½).
    pub interpolation: f64,
    /// Location: the largest change of the blended L to the neighbouring chord samples (the
    /// detector resolves the bond point to one chord sample).
    pub location: f64,
    /// Rounding: 2⁻²⁴ (f32 unit roundoff) × [`stencil_abs_sum`] × max|ρ| / h².
    pub rounding: f64,
}

/// The ≥ 3× rule of δ and ε_⊥ (a value counts as resolved when it is at least three times its
/// noise): the band of ∇²ρ_b is this multiple of [`LaplacianEstimate::noise`].
pub const LAPLACIAN_SAFETY: f64 = 3.0;

impl LaplacianEstimate {
    pub fn noise(&self) -> f64 {
        self.interpolation + self.location + self.rounding
    }

    /// |∇²ρ_b| at or below this is `Indeterminate`.
    pub fn band(&self) -> f64 {
        LAPLACIAN_SAFETY * self.noise()
    }
}

/// How many samples [`strength_points`] asks for: 32 nodal stencils at p* plus the 8-node blends
/// at the two chord neighbours on a grid (800 + 2·200), three stencils without one.
pub fn strength_len(grid: Option<&Cell>) -> usize {
    if grid.is_some() { 32 * 25 + 2 * 200 } else { 3 * 25 }
}

/// The Laplacian round's points for the bond point `p` and its chord neighbours `prev`, `next`
/// (module doc, "The operator", "The noise"); `None` when any of them is outside the grid.
pub fn strength_points(grid: Option<&Cell>, p: [f64; 3], prev: [f64; 3], next: [f64; 3], h: f64) -> Option<Vec<[f64; 3]>> {
    let mut out = Vec::with_capacity(strength_len(grid));
    match grid {
        None => {
            for q in [p, prev, next] {
                out.extend_from_slice(&laplacian_stencil(q, h));
            }
        }
        Some(cell) => {
            let (c, _) = trilinear_cell(cell, p)?;
            let origin = -cell.half_extent();
            for o in block_offsets() {
                let node: [f64; 3] = std::array::from_fn(|a| origin + (c[a] as i64 + o[a]) as f64 * cell.h);
                out.extend_from_slice(&laplacian_stencil(node, cell.h));
            }
            // when p_{k±1} share p*'s cell their 8 nodal stencils repeat samples already in the
            // block (≤ 400 duplicates): left as is — one dispatch, and the layout stays fixed
            out.extend(laplacian_points(grid, prev, h)?);
            out.extend(laplacian_points(grid, next, h)?);
        }
    }
    Some(out)
}

/// ∇²ρ at `p` and its noise from the samples at [`strength_points`].
pub fn strength_from(grid: Option<&Cell>, p: [f64; 3], prev: [f64; 3], next: [f64; 3], h: f64, samples: &[f64]) -> LaplacianEstimate {
    assert_eq!(samples.len(), strength_len(grid), "strength_from: {} samples", samples.len());
    let (value, interpolation, own, rest) = match grid {
        None => (laplacian_from_stencil(&samples[..25], h), 0.0, &samples[..25], &samples[25..]),
        Some(cell) => {
            let nodal: Vec<f64> = samples[..800].chunks(25).map(|f| laplacian_from_stencil(f, cell.h)).collect();
            let value = match trilinear_cell(cell, p) {
                Some((c, _)) => trilinear_blend(cell, p, |ijk| {
                    let d = [ijk[0] - c[0], ijk[1] - c[1], ijk[2] - c[2]];
                    nodal[d[0] + 2 * d[1] + 4 * d[2]]
                }),
                None => f64::NAN,
            };
            let mut interp = 0.0;
            for a in 0..3 {
                let worst = (0..8).map(|ci| (nodal[8 + 8 * a + ci] - 2.0 * nodal[ci] + nodal[ci ^ (1 << a)]).abs()).fold(0.0f64, f64::max);
                interp += worst / 8.0;
            }
            (value, interp, &samples[..800], &samples[800..])
        }
    };
    let half = rest.len() / 2;
    let at_prev = laplacian_at(grid, prev, h, &rest[..half]);
    let at_next = laplacian_at(grid, next, h, &rest[half..]);
    let location = (at_prev - value).abs().max((at_next - value).abs());
    let rho_max = own.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
    let rounding = 2f64.powi(-24) * stencil_abs_sum() * rho_max / (h * h);
    LaplacianEstimate { value, interpolation, location, rounding }
}

/// `f` at the eight nodes around `p`, blended a·(1 − t) + b·t along x, then y, then z — the
/// operator `qm::Grid::trilinear` applies to a stored field (a neighbour beyond an upper face has
/// weight 0 and is not read); 0 outside the grid.
pub fn trilinear_blend(cell: &Cell, p: [f64; 3], f: impl Fn([usize; 3]) -> f64) -> f64 {
    let Some(([i, j, k], [tx, ty, tz])) = trilinear_cell(cell, p) else { return 0.0 };
    let n = cell.n;
    let at = |di: usize, dj: usize, dk: usize| {
        let (a, b, c) = (i + di, j + dj, k + dk);
        if a < n && b < n && c < n { f([a, b, c]) } else { 0.0 }
    };
    let lerp = |a: f64, b: f64, t: f64| a * (1.0 - t) + b * t;
    let x = |dj, dk| if tx == 0.0 { at(0, dj, dk) } else { lerp(at(0, dj, dk), at(1, dj, dk), tx) };
    let y = |dk| if ty == 0.0 { x(0, dk) } else { lerp(x(0, dk), x(1, dk), ty) };
    if tz == 0.0 { y(0) } else { lerp(y(0), y(1), tz) }
}

// ---- the promolecule ----

/// The promolecule of the current scene (module doc, "The same operator"): `exact(p)` = Σ_C
/// ρ̄_{Z_C}(|p − R_C|) from the radial tables; `trilinear(p)` the same quantity at the eight grid
/// nodes around p blended with the provider's weights — or `exact(p)` itself when the samples
/// were not interpolated (`grid` = `None`).
pub struct Promolecule<'a> {
    tables: Vec<&'a RadialDensity>,
    pos: Vec<[f64; 3]>,
    grid: Option<Cell>,
}

impl<'a> Promolecule<'a> {
    /// One table per nucleus, at `pos`; `grid` is the provider's grid (the scene's cell) or
    /// `None` for exact samples.
    pub fn new(tables: Vec<&'a RadialDensity>, pos: &[[f64; 3]], grid: Option<&Cell>) -> Promolecule<'a> {
        assert_eq!(tables.len(), pos.len(), "Promolecule: {} tables for {} nuclei", tables.len(), pos.len());
        Promolecule { tables, pos: pos.to_vec(), grid: grid.cloned() }
    }

    /// From the reference cache: the table of species `z[c]` under `key_for(z[c])` for every
    /// nucleus, on `cell`'s grid. `Err` lists the species without a (non-empty) table, each once,
    /// in scene order.
    pub fn from_refs(refs: &'a AtomReferences, key_for: impl Fn(u8) -> ReferenceKey, cell: &Cell, pos: &[[f64; 3]], z: &[u8]) -> Result<Promolecule<'a>, Vec<u8>> {
        assert_eq!(pos.len(), z.len());
        let mut tables = Vec::with_capacity(z.len());
        let mut missing: Vec<u8> = Vec::new();
        for &zc in z {
            match refs.get(&key_for(zc)).filter(|r| !r.radial.is_empty()) {
                Some(r) => tables.push(&r.radial),
                None => {
                    if !missing.contains(&zc) {
                        missing.push(zc);
                    }
                }
            }
        }
        if !missing.is_empty() {
            return Err(missing);
        }
        Ok(Promolecule { tables, pos: pos.to_vec(), grid: Some(cell.clone()) })
    }

    /// The free-atom table of every nucleus, in scene order (the Hirshfeld partition's ρ_A⁰,
    /// `crate::charges`).
    pub fn tables(&self) -> &[&'a RadialDensity] {
        &self.tables
    }

    /// ρ_pro(p) = Σ_C ρ̄_{Z_C}(|p − R_C|).
    pub fn exact(&self, p: [f64; 3]) -> f64 {
        self.tables.iter().zip(&self.pos).map(|(t, &r)| t.at(dist(p, r))).sum()
    }

    /// Nucleus `c`'s own term of the promolecule at p, ρ̄_{Z_c}(|p − R_c|) — its stockholder
    /// share of `exact(p)` (module doc, "Attribution").
    pub fn atom(&self, c: usize, p: [f64; 3]) -> f64 {
        self.tables[c].at(dist(p, self.pos[c]))
    }

    /// The promolecule put through the provider's interpolation operator (module doc).
    pub fn trilinear(&self, p: [f64; 3]) -> f64 {
        match &self.grid {
            None => self.exact(p),
            Some(cell) => trilinear_blend(cell, p, |ijk| self.exact(node_position(cell, ijk))),
        }
    }

    /// ρ̂ = ρ·exact/trilinear; ρ unchanged where trilinear(p) = 0.
    pub fn corrected(&self, p: [f64; 3], rho: f64) -> f64 {
        let tri = self.trilinear(p);
        if tri > 0.0 { rho * self.exact(p) / tri } else { rho }
    }

    /// q = ρ/trilinear(p) = ρ̂/exact (0 where the promolecule has no reach: beyond every table).
    pub fn ratio(&self, p: [f64; 3], rho: f64) -> f64 {
        let tri = self.trilinear(p);
        if tri > 0.0 { rho / tri } else { 0.0 }
    }
}

// ---- the detector ----

/// One candidate pair's working state (its indices are in its report).
struct Pair {
    e1: [f64; 3],
    e2: [f64; 3],
    /// The chord points p_k.
    p: Vec<[f64; 3]>,
    /// Filled by round 1.
    p_star: [f64; 3],
    /// The chord index of p*.
    k_star: usize,
    rho_star: f64,
    q_star: f64,
    route: Route,
    /// Index into `Detector::reports`.
    report: usize,
}

/// A march in progress (module doc).
struct March {
    pair: usize,
    /// The last accepted point and ρ̂ there.
    pos: [f64; 3],
    rho: f64,
    /// The point being sampled (pos + h/2·dir).
    next: [f64; 3],
    /// |∇ρ̂| at the first march step.
    g0: Option<f64>,
    steps: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Chord,
    Transverse,
    March,
    /// The Laplacian stencils at the bond points (module doc, "Bond strength").
    Laplacian,
    Done,
}

/// The detector runs in rounds (module doc): round 1 the chord samples of every pair within
/// [`R_MAX`]; round 2 the four transverse probes of the survivors; then one small round per
/// march step (7 samples per marching pair: the point and ±h/2 on the three axes) until every
/// march has stopped. `points()` is what to sample next; `feed` takes the samples and returns
/// the next set, `None` when done — the reports are complete then.
pub struct Detector {
    pos: Vec<[f64; 3]>,
    pairs: Vec<Pair>,
    reports: Vec<PairReport>,
    points: Vec<[f64; 3]>,
    stage: Stage,
    /// Pairs the current round's points belong to (round 2).
    active: Vec<usize>,
    marches: Vec<March>,
    /// Pairs whose ∇²ρ_b the Laplacian round samples, in point order.
    lap: Vec<usize>,
    /// The Laplacian round was set up (it runs once, after every verdict is in).
    lap_done: bool,
}

/// e₁ ⊥ u toward the centroid of `others` from `mid` (module doc, "The transverse basis"), the
/// lab construction when that is degenerate; e₂ = u × e₁.
fn transverse_basis(u: [f64; 3], mid: [f64; 3], others: &[[f64; 3]]) -> ([f64; 3], [f64; 3]) {
    let mut e1 = None;
    if !others.is_empty() {
        let mut c = [0.0; 3];
        for o in others {
            c = add(c, *o);
        }
        let v = sub(scale(c, 1.0 / others.len() as f64), mid);
        let v_perp = sub(v, scale(u, dot(v, u)));
        let n = norm(v_perp);
        if n > 1e-9 * norm(v).max(1.0) {
            e1 = Some(scale(v_perp, 1.0 / n));
        }
    }
    let e1 = e1.unwrap_or_else(|| {
        let axis = (0..3).min_by(|&a, &b| u[a].abs().total_cmp(&u[b].abs())).expect("three axes");
        let mut a = [0.0; 3];
        a[axis] = 1.0;
        let c = cross(u, a);
        scale(c, 1.0 / norm(c))
    });
    (e1, cross(u, e1))
}

/// Condition 4 (module doc, "Attribution"): the nucleus other than `i` and `j` whose promolecular
/// term at `p` is largest, if that term exceeds the smaller of the pair's own two — the feature at
/// `p` is then in that atom's share of space. `None` when the pair's atoms are the two largest
/// contributors (ties go to the pair).
fn neighbour_region(pro: &Promolecule<'_>, n: usize, i: usize, j: usize, p: [f64; 3]) -> Option<usize> {
    let weaker = pro.atom(i, p).min(pro.atom(j, p));
    (0..n).filter(|&c| c != i && c != j).map(|c| (c, pro.atom(c, p))).filter(|&(_, w)| w > weaker).max_by(|a, b| a.1.total_cmp(&b.1)).map(|(c, _)| c)
}

impl Detector {
    /// Round 1: every pair within [`R_MAX`] gets [`SEGMENT_SAMPLES`] chord points (the endpoints
    /// are the nuclei themselves); farther pairs are reported `TooFar` and never sampled.
    pub fn new(pos: &[[f64; 3]]) -> Detector {
        let n = pos.len();
        let mut pairs = Vec::new();
        let mut reports = Vec::new();
        let mut points = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                let d = sub(pos[j], pos[i]);
                let r = norm(d);
                let report = PairReport { i, j, route: None, s: None, rho_min: None, transverse_ratio: None, rho_pro: None, enhancement: None, march_end: None, march_stop: None, verdict: Verdict::TooFar, laplacian: None, laplacian_noise: None };
                if !(r < R_MAX) || r == 0.0 {
                    reports.push(report);
                    continue;
                }
                let u = scale(d, 1.0 / r);
                let mid = add(pos[i], scale(d, 0.5));
                let others: Vec<[f64; 3]> = (0..n).filter(|&c| c != i && c != j).map(|c| pos[c]).collect();
                let (e1, e2) = transverse_basis(u, mid, &others);
                let p: Vec<[f64; 3]> = (0..SEGMENT_SAMPLES).map(|k| add(pos[i], scale(d, k as f64 / (SEGMENT_SAMPLES - 1) as f64))).collect();
                points.extend_from_slice(&p);
                pairs.push(Pair { e1, e2, p, p_star: [0.0; 3], k_star: 0, rho_star: 0.0, q_star: 0.0, route: Route::Saddle, report: reports.len() });
                reports.push(report);
            }
        }
        let stage = if pairs.is_empty() { Stage::Done } else { Stage::Chord };
        Detector { pos: pos.to_vec(), pairs, reports, points, stage, active: Vec::new(), marches: Vec::new(), lap: Vec::new(), lap_done: false }
    }

    /// The points to sample for the current round (empty when done).
    pub fn points(&self) -> &[[f64; 3]] {
        &self.points
    }

    /// The reports of every pair, (i < j) order — complete once `feed` has returned `None`.
    pub fn reports(&self) -> &[PairReport] {
        &self.reports
    }

    /// The four transverse probes of a pair: p* + h e₁, p* − h e₁, p* + h e₂, p* − h e₂.
    fn probes(pair: &Pair, h: f64) -> [[f64; 3]; 4] {
        [add(pair.p_star, scale(pair.e1, h)), sub(pair.p_star, scale(pair.e1, h)), add(pair.p_star, scale(pair.e2, h)), sub(pair.p_star, scale(pair.e2, h))]
    }

    /// A march round's seven points at `c`: c, c ± h/2 x̂, c ± h/2 ŷ, c ± h/2 ẑ.
    fn march_points(c: [f64; 3], h: f64) -> [[f64; 3]; 7] {
        let d = 0.5 * h;
        [c, [c[0] + d, c[1], c[2]], [c[0] - d, c[1], c[2]], [c[0], c[1] + d, c[2]], [c[0], c[1] - d, c[2]], [c[0], c[1], c[2] + d], [c[0], c[1], c[2] - d]]
    }

    fn near_nucleus(&self, p: [f64; 3], h: f64) -> bool {
        self.pos.iter().any(|&r| dist(p, r) < h)
    }

    /// Conditions 2, 3 and 4 at the bond point.
    fn finalize(&mut self, idx: usize, multi_centre: bool, pro: &Promolecule<'_>) {
        let pair = &self.pairs[idx];
        let (i, j) = (self.reports[pair.report].i, self.reports[pair.report].j);
        let verdict = if pair.rho_star < RHO_FLOOR {
            Verdict::BelowFloor
        } else if pair.q_star < 1.0 + DELTA {
            Verdict::NotAccumulated
        } else {
            match neighbour_region(pro, self.pos.len(), i, j, pair.p_star) {
                Some(atom) => Verdict::NeighbourRegion { atom },
                None => Verdict::Bonded { multi_centre },
            }
        };
        self.reports[pair.report].verdict = verdict;
    }

    /// Feeds the samples of the last `points()` (the provider's ρ at them, in order) and returns
    /// the next point set, or `None` when every pair has its verdict. `pro` is the scene's
    /// promolecule, `h` the grid spacing (probe offset h, march step h/2, the nucleus rule).
    pub fn feed(&mut self, rho: &[f64], pro: &Promolecule<'_>, h: f64) -> Option<&[[f64; 3]]> {
        assert_eq!(rho.len(), self.points.len(), "Detector::feed: {} samples for {} points", rho.len(), self.points.len());
        assert!(h > 0.0 && h.is_finite(), "Detector::feed: h = {h}");
        let cap = (2.0 * R_MAX / h).round() as usize;
        let mut next: Vec<[f64; 3]> = Vec::new();
        match self.stage {
            Stage::Done => return None,
            Stage::Laplacian => {
                let lap = std::mem::take(&mut self.lap);
                let per = strength_len(pro.grid.as_ref());
                for (&idx, off) in lap.iter().zip((0..).step_by(per)) {
                    let pair = &self.pairs[idx];
                    let k = pair.k_star;
                    let est = strength_from(pro.grid.as_ref(), pair.p_star, pair.p[k - 1], pair.p[k + 1], h, &rho[off..off + per]);
                    let rep = &mut self.reports[pair.report];
                    rep.laplacian = Some(est.value);
                    rep.laplacian_noise = Some(est.band());
                }
                self.stage = Stage::Done;
            }
            Stage::Chord => {
                let mut active = Vec::new();
                for (idx, off) in (0..self.pairs.len()).zip((0..).step_by(SEGMENT_SAMPLES)) {
                    let pair = &mut self.pairs[idx];
                    let raw = &rho[off..off + SEGMENT_SAMPLES];
                    let rho_hat: Vec<f64> = pair.p.iter().zip(raw).map(|(&p, &v)| pro.corrected(p, v)).collect();
                    let q: Vec<f64> = pair.p.iter().zip(raw).map(|(&p, &v)| pro.ratio(p, v)).collect();
                    let last = SEGMENT_SAMPLES - 1;
                    let k_min = (1..last).min_by(|&a, &b| rho_hat[a].total_cmp(&rho_hat[b])).expect("interior points");
                    let (route, k_star, interior) = if rho_hat[k_min] < rho_hat[0] && rho_hat[k_min] < rho_hat[last] {
                        (Route::Saddle, k_min, true)
                    } else {
                        let k_max = (1..last).max_by(|&a, &b| q[a].total_cmp(&q[b])).expect("interior points");
                        (Route::Deformation, k_max, q[k_max] > q[0] && q[k_max] > q[last])
                    };
                    pair.route = route;
                    pair.p_star = pair.p[k_star];
                    pair.k_star = k_star;
                    pair.rho_star = rho_hat[k_star];
                    pair.q_star = q[k_star];
                    let rep = &mut self.reports[pair.report];
                    rep.route = Some(route);
                    rep.s = Some(k_star as f64 / last as f64);
                    rep.rho_min = Some(rho_hat[k_star]);
                    rep.rho_pro = Some(pro.exact(pair.p_star));
                    rep.enhancement = Some(q[k_star]);
                    if !interior {
                        rep.verdict = Verdict::NoInteriorMinimum;
                        continue;
                    }
                    next.extend_from_slice(&Self::probes(pair, h));
                    active.push(idx);
                }
                self.active = active;
                self.stage = Stage::Transverse;
            }
            Stage::Transverse => {
                let mut marches = Vec::new();
                let active = std::mem::take(&mut self.active);
                for (&idx, off) in active.iter().zip((0..).step_by(4)) {
                    let pair = &self.pairs[idx];
                    let probes = Self::probes(pair, h);
                    let raw = &rho[off..off + 4];
                    match pair.route {
                        Route::Deformation => {
                            let qv: Vec<f64> = probes.iter().zip(raw).map(|(&p, &v)| pro.ratio(p, v)).collect();
                            let ratio = qv.iter().cloned().fold(f64::NEG_INFINITY, f64::max) / pair.q_star;
                            self.reports[pair.report].transverse_ratio = Some(ratio);
                            if ratio <= 1.0 + EPS_TRANSVERSE {
                                self.finalize(idx, false, pro);
                            } else {
                                self.reports[pair.report].verdict = Verdict::NotATransverseMaximum;
                            }
                        }
                        Route::Saddle => {
                            let vals: Vec<f64> = probes.iter().zip(raw).map(|(&p, &v)| pro.corrected(p, v)).collect();
                            let ratio = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max) / pair.rho_star;
                            self.reports[pair.report].transverse_ratio = Some(ratio);
                            if ratio <= 1.0 + EPS_TRANSVERSE {
                                self.finalize(idx, false, pro);
                                continue;
                            }
                            // the march: first direction from the probes' transverse gradient
                            let g = add(scale(pair.e1, vals[0] - vals[1]), scale(pair.e2, vals[2] - vals[3]));
                            let gn = norm(g);
                            let dir = if gn > 0.0 {
                                scale(g, 1.0 / gn)
                            } else {
                                // a symmetric rise on both sides: toward the highest probe
                                let best = (0..4).max_by(|&a, &b| vals[a].total_cmp(&vals[b])).expect("four probes");
                                scale(sub(probes[best], pair.p_star), 1.0 / h)
                            };
                            let next_pos = add(pair.p_star, scale(dir, 0.5 * h));
                            let rep_idx = pair.report;
                            if self.near_nucleus(next_pos, h) {
                                let rep = &mut self.reports[rep_idx];
                                rep.march_end = Some(next_pos);
                                rep.march_stop = Some(MarchStop::Nucleus);
                                rep.verdict = Verdict::NotATransverseMaximum;
                                continue;
                            }
                            next.extend_from_slice(&Self::march_points(next_pos, h));
                            marches.push(March { pair: idx, pos: pair.p_star, rho: pair.rho_star, next: next_pos, g0: None, steps: 0 });
                        }
                    }
                }
                self.marches = marches;
                self.stage = if self.marches.is_empty() { Stage::Done } else { Stage::March };
            }
            Stage::March => {
                let marches = std::mem::take(&mut self.marches);
                let mut keep = Vec::new();
                for (mut m, off) in marches.into_iter().zip((0..).step_by(7)) {
                    let pts = Self::march_points(m.next, h);
                    let v: Vec<f64> = pts.iter().zip(&rho[off..off + 7]).map(|(&p, &x)| pro.corrected(p, x)).collect();
                    let rep_idx = self.pairs[m.pair].report;
                    // (ii) ρ̂ decreased: a non-nuclear maximum lies between pos and next
                    if v[0] < m.rho {
                        let rep = &mut self.reports[rep_idx];
                        rep.march_end = Some(m.pos);
                        rep.march_stop = Some(MarchStop::Maximum);
                        self.finalize(m.pair, true, pro);
                        continue;
                    }
                    let g = [(v[1] - v[2]) / h, (v[3] - v[4]) / h, (v[5] - v[6]) / h];
                    let gm = norm(g);
                    m.steps += 1;
                    // (iii) the gradient fell below f_g of its first value: a stationary point
                    let stationary = match m.g0 {
                        None => {
                            m.g0 = Some(gm);
                            gm == 0.0
                        }
                        Some(g0) => gm < MARCH_STATIONARY_FRACTION * g0,
                    };
                    if stationary {
                        let rep = &mut self.reports[rep_idx];
                        rep.march_end = Some(m.next);
                        rep.march_stop = Some(MarchStop::Stationary);
                        self.finalize(m.pair, true, pro);
                        continue;
                    }
                    m.pos = m.next;
                    m.rho = v[0];
                    let next_pos = add(m.pos, scale(g, 0.5 * h / gm));
                    // (i) the next point is a nucleus's: the rise was its hill
                    if self.near_nucleus(next_pos, h) {
                        let rep = &mut self.reports[rep_idx];
                        rep.march_end = Some(next_pos);
                        rep.march_stop = Some(MarchStop::Nucleus);
                        rep.verdict = Verdict::NotATransverseMaximum;
                        continue;
                    }
                    // (iv) the cap
                    if m.steps >= cap {
                        let rep = &mut self.reports[rep_idx];
                        rep.march_end = Some(m.pos);
                        rep.march_stop = Some(MarchStop::StepCap);
                        rep.verdict = Verdict::NotATransverseMaximum;
                        continue;
                    }
                    m.next = next_pos;
                    next.extend_from_slice(&Self::march_points(next_pos, h));
                    keep.push(m);
                }
                self.marches = keep;
                if self.marches.is_empty() {
                    self.stage = Stage::Done;
                }
            }
        }
        // every verdict is in: one more round for ∇²ρ at the bond points that have one worth
        // classifying (the bonds, and the accumulation misses — He₂'s closed-shell contact)
        if next.is_empty() && !self.lap_done {
            self.lap_done = true;
            for (idx, pair) in self.pairs.iter().enumerate() {
                if !matches!(self.reports[pair.report].verdict, Verdict::Bonded { .. } | Verdict::NotAccumulated) {
                    continue;
                }
                let k = pair.k_star;
                if let Some(pts) = strength_points(pro.grid.as_ref(), pair.p_star, pair.p[k - 1], pair.p[k + 1], h) {
                    next.extend_from_slice(&pts);
                    self.lap.push(idx);
                }
            }
            if !self.lap.is_empty() {
                self.stage = Stage::Laplacian;
            }
        }
        self.points = next;
        if self.points.is_empty() {
            self.stage = Stage::Done;
            return None;
        }
        Some(&self.points)
    }

    /// The bonds (the `Bonded` reports) and the molecules of `n` nuclei at `step`/`sim_time`.
    pub fn graph(&self, n: usize, step: u64, sim_time: f64) -> BondGraph {
        let bonds: Vec<Bond> = self
            .reports
            .iter()
            .filter_map(|r| match r.verdict {
                Verdict::Bonded { multi_centre } => Some(Bond {
                    i: r.i,
                    j: r.j,
                    rho_bcp: r.rho_min.expect("a bonded pair has a bond point"),
                    enhancement: r.enhancement.expect("a bonded pair has its ratio"),
                    s: r.s.expect("a bonded pair has its s"),
                    route: r.route.expect("a bonded pair has its route"),
                    multi_centre,
                    laplacian_bcp: r.laplacian.unwrap_or(f64::NAN),
                    laplacian_noise: r.laplacian_noise.unwrap_or(f64::NAN),
                }),
                _ => None,
            })
            .collect();
        let components = components(n, &bonds);
        BondGraph { bonds, components, step, sim_time, n_nuclei: n }
    }
}

/// The connected components of `n` nuclei under `bonds` (union–find): each sorted, listed by
/// their smallest member; isolated nuclei are one-atom components.
pub fn components(n: usize, bonds: &[Bond]) -> Vec<Vec<usize>> {
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for b in bonds {
        if b.i >= n || b.j >= n {
            continue;
        }
        let (a, c) = (find(&mut parent, b.i), find(&mut parent, b.j));
        if a != c {
            parent[a.max(c)] = a.min(c);
        }
    }
    // (root, members): each group keeps its root alongside it (review M8)
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for x in 0..n {
        let root = find(&mut parent, x);
        match groups.iter_mut().find(|g| g.0 == root) {
            Some(g) => g.1.push(x),
            None => groups.push((root, vec![x])),
        }
    }
    groups.sort_by_key(|g| g.1[0]);
    groups.into_iter().map(|g| g.1).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::references::{shell_nodes, shell_radius};
    use crate::test_providers::{Gaussians, GriddedGaussians, Term};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    const H: f64 = 0.25;

    /// A term written as amp·exp(−r²/a) (the fixtures' notation; `Term` stores w with 2w² = a).
    fn g(c: [f64; 3], amp: f64, a: f64) -> Term {
        Term { centre: c, amp, w: (0.5 * a).sqrt() }
    }

    /// An analytic fixture: the density is Σ terms; the first `nuclei.len()` terms are the atoms
    /// (one per nucleus, centred on it) and make the promolecule.
    #[derive(Clone)]
    struct Fixture {
        terms: Vec<Term>,
        nuclei: Vec<[f64; 3]>,
    }

    impl Fixture {
        fn rho(&self, p: [f64; 3]) -> f64 {
            self.terms.iter().map(|t| t.at(p)).sum()
        }

        fn pro(&self, p: [f64; 3]) -> f64 {
            self.terms[..self.nuclei.len()].iter().map(|t| t.at(p)).sum()
        }

        /// Exact radial tables of the atom terms (fine spacing, a function of r only).
        fn tables(&self) -> Vec<RadialDensity> {
            self.terms[..self.nuclei.len()].iter().map(|t| RadialDensity::from_fn(|r| t.amp * (-r * r / (2.0 * t.w * t.w)).exp(), R_MAX + 2.0, 0.005)).collect()
        }

        fn rotated(&self, about: [f64; 3], m: [[f64; 3]; 3]) -> Fixture {
            let rot = |p: [f64; 3]| {
                let d = sub(p, about);
                add(about, [dot(m[0], d), dot(m[1], d), dot(m[2], d)])
            };
            Fixture { terms: self.terms.iter().map(|t| Term { centre: rot(t.centre), ..*t }).collect(), nuclei: self.nuclei.iter().map(|&p| rot(p)).collect() }
        }

        /// The detector on the exact density (no grid, no correction).
        fn detect(&self) -> Detector {
            let tables = self.tables();
            let pro = Promolecule::new(tables.iter().collect(), &self.nuclei, None);
            let mut det = Detector::new(&self.nuclei);
            let mut pts = det.points().to_vec();
            let mut rounds = 0;
            while !pts.is_empty() {
                let rho: Vec<f64> = pts.iter().map(|&p| self.rho(p)).collect();
                rounds += 1;
                assert!(rounds < 200, "the rounds do not end");
                match det.feed(&rho, &pro, H) {
                    Some(next) => pts = next.to_vec(),
                    None => break,
                }
            }
            det
        }
    }

    /// Rodrigues' rotation about the unit `axis` by `angle`.
    fn rotation(axis: [f64; 3], angle: f64) -> [[f64; 3]; 3] {
        let a = scale(axis, 1.0 / norm(axis));
        let (s, c) = angle.sin_cos();
        let t = 1.0 - c;
        [
            [c + a[0] * a[0] * t, a[0] * a[1] * t - a[2] * s, a[0] * a[2] * t + a[1] * s],
            [a[1] * a[0] * t + a[2] * s, c + a[1] * a[1] * t, a[1] * a[2] * t - a[0] * s],
            [a[2] * a[0] * t - a[1] * s, a[2] * a[1] * t + a[0] * s, c + a[2] * a[2] * t],
        ]
    }

    /// The independent fine scan of a pair (MUST-4): 501 chord points, the transverse rise as the
    /// maximum over 72 directions around the chord at radius h, a 0.02 Bohr gradient ascent with
    /// analytic central differences and the four stop rules.
    #[derive(Clone, Debug)]
    struct Scan {
        route: Route,
        verdict: Verdict,
        s: f64,
        rho_min: f64,
        enhancement: f64,
        transverse: f64,
        march_end: Option<[f64; 3]>,
        march_stop: Option<MarchStop>,
    }

    fn grad(f: &dyn Fn([f64; 3]) -> f64, p: [f64; 3]) -> [f64; 3] {
        let d = 1e-5;
        std::array::from_fn(|a| {
            let mut hi = p;
            let mut lo = p;
            hi[a] += d;
            lo[a] -= d;
            (f(hi) - f(lo)) / (2.0 * d)
        })
    }

    /// The scan of pair (i, j) of `fx` with the fixture's own promolecule (its atom terms).
    fn scan_pair(fx: &Fixture, i: usize, j: usize, h: f64) -> Option<Scan> {
        scan_view(&ScanView { rho: &|p| fx.rho(p), pro: &|p| fx.pro(p), atom: &|c, p| fx.terms[c].at(p), nuclei: &fx.nuclei }, i, j, h)
    }

    fn bonded_pair(bump: f64) -> Fixture {
        let (a, b) = ([-0.8, 0.0, 0.0], [0.8, 0.0, 0.0]);
        Fixture { terms: vec![g(a, 0.3, 0.5), g(b, 0.3, 0.5), g([0.0; 3], bump * 0.3, 0.5)], nuclei: vec![a, b] }
    }

    fn report_of<'a>(det: &'a Detector, i: usize, j: usize) -> &'a PairReport {
        det.reports().iter().find(|r| r.i == i && r.j == j).expect("a report per pair")
    }

    #[test]
    fn a_bonded_pair_passes_all_three_conditions() {
        let fx = bonded_pair(0.5);
        let scan = scan_pair(&fx, 0, 1, H).unwrap();
        println!("scan: {scan:?}");
        assert_eq!((scan.route, scan.verdict), (Route::Saddle, Verdict::Bonded { multi_centre: false }));
        let det = fx.detect();
        let r = report_of(&det, 0, 1);
        println!("report: {r:?}");
        assert_eq!(r.verdict, Verdict::Bonded { multi_centre: false });
        assert_eq!(r.route, Some(Route::Saddle));
        assert!((r.s.unwrap() - 0.5).abs() <= 0.03, "s = {:?}", r.s);
        let e = r.enhancement.unwrap();
        assert!((e / scan.enhancement - 1.0).abs() <= 0.02, "enhancement {e} vs the scan's {}", scan.enhancement);
        assert!((scan.enhancement - 1.899).abs() < 0.01, "the fixture's exact ratio (0.3168/0.1668)");
        assert!(r.transverse_ratio.unwrap() < 1.0, "the bump falls off across the chord");
        let graph = det.graph(2, 7, 1.5);
        assert_eq!(graph.bonds.len(), 1);
        assert_eq!((graph.bonds[0].i, graph.bonds[0].j, graph.bonds[0].route, graph.bonds[0].multi_centre), (0, 1, Route::Saddle, false));
        assert_eq!((graph.step, graph.sim_time, graph.n_nuclei), (7, 1.5, 2));
        assert_eq!(graph.components, vec![vec![0, 1]]);
    }

    /// ρ = g_A + g_B − 0.42·g(mid): ρ(mid) ≈ 0.041 > ρ_floor with q ≈ 0.25 — the saddle and the
    /// floor pass, the accumulation does not; every number is in the report.
    #[test]
    fn a_depleted_overlap_is_not_a_bond() {
        let fx = bonded_pair(-0.42);
        let scan = scan_pair(&fx, 0, 1, H).unwrap();
        println!("scan: {scan:?}");
        assert_eq!(scan.verdict, Verdict::NotAccumulated);
        let det = fx.detect();
        let r = report_of(&det, 0, 1);
        println!("report: {r:?}");
        assert_eq!(r.verdict, Verdict::NotAccumulated);
        assert_eq!(r.route, Some(Route::Saddle));
        let (rho_min, e) = (r.rho_min.unwrap(), r.enhancement.unwrap());
        assert!(rho_min > RHO_FLOOR && (rho_min - 0.0408).abs() < 0.002, "ρ̂(p*) = {rho_min}");
        assert!((e - 0.245).abs() < 0.01, "enhancement {e}");
        assert!(r.transverse_ratio.unwrap() < 1.0);
        assert!(det.graph(2, 0, 0.0).bonds.is_empty());
        assert_eq!(det.graph(2, 0, 0.0).components, vec![vec![0], vec![1]]);
    }

    fn water_like() -> Fixture {
        let o = [0.0; 3];
        let h1 = [1.43, 1.11, 0.0];
        let h2 = [-1.43, 1.11, 0.0];
        let bump = |h: [f64; 3]| g(scale(h, 0.55), 0.12, 0.405);
        Fixture {
            terms: vec![g(o, 6.0, 0.18), g(h1, 0.3, 0.5), g(h2, 0.3, 0.5), g(o, 0.5, 1.62), bump(h1), bump(h2)],
            nuclei: vec![o, h1, h2],
        }
    }

    /// The promolecule of the water-like fixture is O's core + valence and the two H's (the O
    /// valence is the fixture's fourth term, so `Fixture::pro` cannot be used): one O table with
    /// both terms, and the H table.
    fn water_tables() -> Vec<RadialDensity> {
        let o_table = RadialDensity::from_fn(|r| 6.0 * (-r * r / 0.18).exp() + 0.5 * (-r * r / 1.62).exp(), R_MAX + 2.0, 0.005);
        let h_table = RadialDensity::from_fn(|r| 0.3 * (-r * r / 0.5).exp(), R_MAX + 2.0, 0.005);
        vec![o_table, h_table.clone(), h_table]
    }

    fn water_pro(fx: &Fixture, p: [f64; 3]) -> f64 {
        fx.terms[..4].iter().map(|t| t.at(p)).sum()
    }

    /// The water-like fixture through the detector with its own promolecule (O core + valence).
    fn detect_water(fx: &Fixture) -> Detector {
        detect_with(fx, &water_tables())
    }

    /// A water-like fixture through the detector with the given promolecule tables.
    fn detect_with(fx: &Fixture, tables: &[RadialDensity]) -> Detector {
        let pro = Promolecule::new(tables.iter().collect(), &fx.nuclei, None);
        let mut det = Detector::new(&fx.nuclei);
        let mut pts = det.points().to_vec();
        while !pts.is_empty() {
            let rho: Vec<f64> = pts.iter().map(|&p| fx.rho(p)).collect();
            match det.feed(&rho, &pro, H) {
                Some(next) => pts = next.to_vec(),
                None => break,
            }
        }
        det
    }

    /// The scan of a water-like fixture (terms: O core, H₁, H₂, O valence, then the bumps) with
    /// the full O promolecule (core + valence).
    fn scan_water(fx: &Fixture, i: usize, j: usize) -> Scan {
        let atom = |c: usize, p: [f64; 3]| if c == 0 { fx.terms[0].at(p) + fx.terms[3].at(p) } else { fx.terms[c].at(p) };
        scan_view(&ScanView { rho: &|p| fx.rho(p), pro: &|p| water_pro(fx, p), atom: &atom, nuclei: &fx.nuclei }, i, j, H).unwrap()
    }

    /// A density, its promolecule, each nucleus's own term of it (`atom(c, p)`, for condition 4)
    /// and the nuclei, for the scan.
    struct ScanView<'a> {
        rho: &'a dyn Fn([f64; 3]) -> f64,
        pro: &'a dyn Fn([f64; 3]) -> f64,
        atom: &'a dyn Fn(usize, [f64; 3]) -> f64,
        nuclei: &'a [[f64; 3]],
    }

    /// The independent fine scan (struct `Scan`'s doc) of pair (i, j).
    fn scan_view(v: &ScanView<'_>, i: usize, j: usize, h: f64) -> Option<Scan> {
        let (a, b) = (v.nuclei[i], v.nuclei[j]);
        let d = sub(b, a);
        let r = norm(d);
        if !(r < R_MAX) {
            return None;
        }
        let u = scale(d, 1.0 / r);
        let n = 500;
        let pts: Vec<[f64; 3]> = (0..=n).map(|k| add(a, scale(d, k as f64 / n as f64))).collect();
        let rho: Vec<f64> = pts.iter().map(|&p| (v.rho)(p)).collect();
        let q: Vec<f64> = pts.iter().map(|&p| (v.rho)(p) / (v.pro)(p)).collect();
        let (e1, e2) = transverse_basis(u, add(a, scale(d, 0.5)), &[]);
        let circle = |f: &dyn Fn([f64; 3]) -> f64, c: [f64; 3]| -> f64 {
            (0..72).map(|m| {
                let th = 2.0 * std::f64::consts::PI * m as f64 / 72.0;
                f(add(c, scale(add(scale(e1, th.cos()), scale(e2, th.sin())), h)))
            }).fold(f64::NEG_INFINITY, f64::max)
        };
        let k_min = (1..n).min_by(|&x, &y| rho[x].total_cmp(&rho[y])).unwrap();
        let (route, k) = if rho[k_min] < rho[0] && rho[k_min] < rho[n] {
            (Route::Saddle, k_min)
        } else {
            let k_max = (1..n).max_by(|&x, &y| q[x].total_cmp(&q[y])).unwrap();
            if !(q[k_max] > q[0] && q[k_max] > q[n]) {
                return Some(Scan { route: Route::Deformation, verdict: Verdict::NoInteriorMinimum, s: k_max as f64 / n as f64, rho_min: rho[k_max], enhancement: q[k_max], transverse: f64::NAN, march_end: None, march_stop: None });
            }
            (Route::Deformation, k_max)
        };
        let p = pts[k];
        let mut scan = Scan { route, verdict: Verdict::TooFar, s: k as f64 / n as f64, rho_min: rho[k], enhancement: q[k], transverse: f64::NAN, march_end: None, march_stop: None };
        let mut multi = false;
        match route {
            Route::Deformation => {
                let qf = |x: [f64; 3]| (v.rho)(x) / (v.pro)(x);
                scan.transverse = circle(&qf, p) / q[k];
                if scan.transverse > 1.0 + EPS_TRANSVERSE {
                    scan.verdict = Verdict::NotATransverseMaximum;
                    return Some(scan);
                }
            }
            Route::Saddle => {
                let rf = |x: [f64; 3]| (v.rho)(x);
                scan.transverse = circle(&rf, p) / rho[k];
                if scan.transverse > 1.0 + EPS_TRANSVERSE {
                    let step = 0.02;
                    let g = grad(&rf, p);
                    let mut dir = sub(g, scale(u, dot(g, u)));
                    dir = scale(dir, 1.0 / norm(dir));
                    let mut pos = p;
                    let mut val = rho[k];
                    let mut g0: Option<f64> = None;
                    let cap = (2.0 * R_MAX / step) as usize;
                    let mut steps = 0;
                    loop {
                        let next = add(pos, scale(dir, step));
                        if v.nuclei.iter().any(|&nn| dist(next, nn) < h) {
                            scan.march_end = Some(next);
                            scan.march_stop = Some(MarchStop::Nucleus);
                            scan.verdict = Verdict::NotATransverseMaximum;
                            return Some(scan);
                        }
                        let val_next = rf(next);
                        if val_next < val {
                            scan.march_end = Some(pos);
                            scan.march_stop = Some(MarchStop::Maximum);
                            multi = true;
                            break;
                        }
                        let gg = grad(&rf, next);
                        let gm = norm(gg);
                        steps += 1;
                        let stationary = match g0 {
                            None => {
                                g0 = Some(gm);
                                gm == 0.0
                            }
                            Some(g0) => gm < MARCH_STATIONARY_FRACTION * g0,
                        };
                        if stationary {
                            scan.march_end = Some(next);
                            scan.march_stop = Some(MarchStop::Stationary);
                            multi = true;
                            break;
                        }
                        if steps >= cap {
                            scan.march_end = Some(next);
                            scan.march_stop = Some(MarchStop::StepCap);
                            scan.verdict = Verdict::NotATransverseMaximum;
                            return Some(scan);
                        }
                        pos = next;
                        val = val_next;
                        dir = scale(gg, 1.0 / gm);
                    }
                }
            }
        }
        scan.verdict = if rho[k] < RHO_FLOOR {
            Verdict::BelowFloor
        } else if q[k] < 1.0 + DELTA {
            Verdict::NotAccumulated
        } else {
            // condition 4 from the analytic atom terms: the largest third-atom term, if it
            // outweighs the weaker of the pair's own
            let weaker = (v.atom)(i, p).min((v.atom)(j, p));
            let third = (0..v.nuclei.len()).filter(|&c| c != i && c != j).map(|c| (c, (v.atom)(c, p))).fold(None, |best: Option<(usize, f64)>, (c, w)| match best {
                Some((_, bw)) if bw >= w => best,
                _ => Some((c, w)),
            });
            match third {
                Some((c, w)) if w > weaker => Verdict::NeighbourRegion { atom: c },
                _ => Verdict::Bonded { multi_centre: multi },
            }
        };
        Some(scan)
    }

    /// Water-like: the O–H chords are monotone (Route D, the bond point at q's interior maximum,
    /// bonded); the H–H chord has an interior minimum whose transverse rise (toward O) is followed
    /// by the march into O's hill → `NotATransverseMaximum`. The expected numbers are the scan's.
    #[test]
    fn a_third_atom_does_not_bond_two_neighbours_and_the_x_h_bonds_take_the_deformation_route() {
        let fx = water_like();
        let oh = scan_water(&fx, 0, 1);
        let hh = scan_water(&fx, 1, 2);
        println!("scan O–H: {oh:?}\nscan H–H: {hh:?}");
        // the brief's pre-computation, re-derived here
        assert_eq!(oh.route, Route::Deformation, "the O–H chord is monotone: ends {:.3}/{:.3}", fx.rho(fx.nuclei[0]), fx.rho(fx.nuclei[1]));
        assert!((fx.rho(fx.nuclei[0]) - 6.52).abs() < 0.02 && (fx.rho(fx.nuclei[1]) - 0.39).abs() < 0.01);
        assert!((oh.enhancement - 1.33).abs() < 0.02 && (oh.s - 0.59).abs() < 0.02, "q max {} at s {}", oh.enhancement, oh.s);
        assert!(oh.transverse < 1.0 + EPS_TRANSVERSE, "{}", oh.transverse);
        assert_eq!(oh.verdict, Verdict::Bonded { multi_centre: false });
        assert_eq!(hh.route, Route::Saddle);
        assert!((hh.rho_min - 0.278).abs() < 0.003 && (hh.s - 0.5).abs() < 1e-9, "H–H minimum {} at {}", hh.rho_min, hh.s);
        assert!((hh.transverse - 1.68).abs() < 0.03, "transverse {}", hh.transverse);
        assert_eq!((hh.verdict, hh.march_stop), (Verdict::NotATransverseMaximum, Some(MarchStop::Nucleus)));
        let end = hh.march_end.unwrap();
        assert!(dist(end, fx.nuclei[0]) < H, "the scan's march ended {} from O", dist(end, fx.nuclei[0]));

        let det = detect_water(&fx);
        for r in det.reports() {
            println!("report: {r:?}");
        }
        let r_oh1 = report_of(&det, 0, 1);
        let r_oh2 = report_of(&det, 0, 2);
        let r_hh = report_of(&det, 1, 2);
        for r in [r_oh1, r_oh2] {
            assert_eq!((r.route, r.verdict), (Some(Route::Deformation), Verdict::Bonded { multi_centre: false }));
            assert!((r.enhancement.unwrap() / oh.enhancement - 1.0).abs() <= 0.02, "{:?}", r.enhancement);
            assert!((r.s.unwrap() - oh.s).abs() <= 0.04);
        }
        assert_eq!(r_hh.verdict, hh.verdict);
        assert_eq!((r_hh.route, r_hh.march_stop), (Some(Route::Saddle), Some(MarchStop::Nucleus)));
        let end = r_hh.march_end.unwrap();
        let d_o = dist(end, fx.nuclei[0]);
        println!("detector H–H march: end {end:?}, {d_o:.3} Bohr from O");
        assert!(d_o < H && (d_o - 0.24).abs() < 0.03, "{d_o}");
        let graph = det.graph(3, 0, 0.0);
        let bonds: Vec<(usize, usize)> = graph.bonds.iter().map(|b| (b.i, b.j)).collect();
        assert_eq!(bonds, vec![(0, 1), (0, 2)]);
        assert!(graph.bonds.iter().all(|b| b.route == Route::Deformation));
        assert_eq!(graph.components, vec![vec![0, 1, 2]]);
    }

    fn ring(bump: f64) -> Fixture {
        let side = 2.117;
        let r = side / 3f64.sqrt();
        let v: Vec<[f64; 3]> = (0..3).map(|k| {
            let th = std::f64::consts::FRAC_PI_2 + 2.0 * std::f64::consts::PI * k as f64 / 3.0;
            [r * th.cos(), r * th.sin(), 0.0]
        }).collect();
        let mut terms: Vec<Term> = v.iter().map(|&p| g(p, 0.3, 0.72)).collect();
        terms.push(g([0.0; 3], bump, 1.28));
        Fixture { terms, nuclei: v }
    }

    /// H₃⁺-like: three atoms on a triangle of side 2.117 plus a bump at the centroid. At bump 0.25
    /// the +11 % rise toward the centroid exceeds ε_⊥ and the march runs: it stalls on the ring
    /// point (the stationary stop) and every edge is `Bonded { multi_centre: true }`. At a bump
    /// whose rise is inside ε_⊥ the plain saddle route applies (`multi_centre: false`), the
    /// components unchanged. The brief named 0.15 for that run; its rise is +8.0 % (measured
    /// below, printed), above ε_⊥ = 0.07, so the march runs there too — 0.10 (+5.6 %) is the
    /// inside-ε run, and 0.15 is checked against the scan like the others.
    #[test]
    fn a_three_centre_ring_is_kept_by_the_march() {
        let fx = ring(0.25);
        let mid = add(fx.nuclei[0], scale(sub(fx.nuclei[1], fx.nuclei[0]), 0.5));
        let centroid = [0.0; 3];
        println!("ρ(edge mid) {:.4}, ρ(centroid) {:.4}, ratio {:.3}, q(edge mid) {:.3}", fx.rho(mid), fx.rho(centroid), fx.rho(centroid) / fx.rho(mid), fx.rho(mid) / fx.pro(mid));
        assert!((fx.rho(mid) - 0.316).abs() < 0.002 && (fx.rho(centroid) / fx.rho(mid) - 1.148).abs() < 0.005);
        let scan = scan_pair(&fx, 0, 1, H).unwrap();
        println!("scan edge (0,1): {scan:?}");
        assert!((scan.transverse - 1.111).abs() < 0.005, "{}", scan.transverse);
        assert_eq!(scan.verdict, Verdict::Bonded { multi_centre: true });
        assert!((scan.enhancement - 2.44).abs() < 0.02);
        let det = fx.detect();
        let graph = det.graph(3, 0, 0.0);
        for r in det.reports() {
            println!("report: {r:?}");
        }
        assert_eq!(graph.bonds.len(), 3);
        for r in det.reports() {
            assert_eq!(r.verdict, Verdict::Bonded { multi_centre: true }, "{r:?}");
            assert!((r.enhancement.unwrap() / scan.enhancement - 1.0).abs() <= 0.03);
            let end = r.march_end.unwrap();
            let d_c = dist(end, centroid);
            let d_n = fx.nuclei.iter().map(|&p| dist(end, p)).fold(f64::INFINITY, f64::min);
            println!("  march end {end:?}: {d_c:.3} from the centroid, {d_n:.3} from the nearest nucleus ({:?})", r.march_stop);
            assert!(d_c <= 0.2 && d_n >= 1.0, "{d_c} {d_n}");
        }
        assert_eq!(graph.components, vec![vec![0, 1, 2]]);
        // the rise inside ε_⊥: the saddle route alone
        for (bump, expect_march) in [(0.10, false), (0.15, true)] {
            let fx = ring(bump);
            let scan = scan_pair(&fx, 0, 1, H).unwrap();
            let det = fx.detect();
            let graph = det.graph(3, 0, 0.0);
            println!("bump {bump}: rise {:.3} (scan {:.3}), scan {:?}", report_of(&det, 0, 1).transverse_ratio.unwrap(), scan.transverse, scan.verdict);
            assert_eq!(graph.bonds.len(), 3);
            assert_eq!(scan.transverse > 1.0 + EPS_TRANSVERSE, expect_march, "bump {bump}: rise {}", scan.transverse);
            for r in det.reports() {
                assert_eq!(r.verdict, scan.verdict, "bump {bump}: {r:?}");
                assert_eq!(r.verdict, Verdict::Bonded { multi_centre: expect_march });
            }
            assert_eq!(graph.components, vec![vec![0, 1, 2]]);
        }
    }

    /// R-1: the water-like fixture rotated about its H–H axis (45°, 90°) and the ring tilted
    /// 45° about x, plus both under one random rotation: every verdict identical to the
    /// axis-aligned run, `rho_min` and `enhancement` equal to 1e-9 (the chord samples are
    /// rotation-covariant), the march end points equal under the rotation to 0.05 Bohr.
    #[test]
    fn the_march_is_rotation_invariant() {
        let mut rng = StdRng::seed_from_u64(7);
        let random_axis: [f64; 3] = std::array::from_fn(|_| rng.gen_range(-1.0..1.0));
        let random_angle: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
        let random = rotation(random_axis, random_angle);
        let compare = |name: &str, base: &Detector, rot: &Detector, m: [[f64; 3]; 3], about: [f64; 3]| {
            let map = |p: [f64; 3]| {
                let d = sub(p, about);
                add(about, [dot(m[0], d), dot(m[1], d), dot(m[2], d)])
            };
            for (a, b) in base.reports().iter().zip(rot.reports()) {
                assert_eq!((a.i, a.j, a.verdict, a.route, a.march_stop), (b.i, b.j, b.verdict, b.route, b.march_stop), "{name}: {a:?} vs {b:?}");
                for (x, y) in [(a.rho_min, b.rho_min), (a.enhancement, b.enhancement), (a.s, b.s)] {
                    match (x, y) {
                        (Some(x), Some(y)) => assert!((x - y).abs() <= 1e-9 * x.abs().max(1.0), "{name}: {x} vs {y}"),
                        (None, None) => {}
                        other => panic!("{name}: {other:?}"),
                    }
                }
                if let (Some(ea), Some(eb)) = (a.march_end, b.march_end) {
                    let d = dist(map(ea), eb);
                    println!("{name} pair ({}, {}): march end {ea:?} → {eb:?}, |Δ| = {d:.4}", a.i, a.j);
                    assert!(d <= 0.05, "{name}: the march ends differ by {d}");
                }
            }
        };
        let w = water_like();
        let base = detect_water(&w);
        let hh_axis = sub(w.nuclei[2], w.nuclei[1]);
        for (name, m) in [("water 45° about H–H", rotation(hh_axis, std::f64::consts::FRAC_PI_4)), ("water 90° about H–H", rotation(hh_axis, std::f64::consts::FRAC_PI_2)), ("water random", random)] {
            let r = w.rotated(w.nuclei[1], m);
            let det = detect_water(&r);
            compare(name, &base, &det, m, w.nuclei[1]);
        }
        let ring_fx = ring(0.25);
        let base = ring_fx.detect();
        for (name, m) in [("ring tilted 45° about x", rotation([1.0, 0.0, 0.0], std::f64::consts::FRAC_PI_4)), ("ring random", random)] {
            let r = ring_fx.rotated([0.0; 3], m);
            let det = r.detect();
            compare(name, &base, &det, m, [0.0; 3]);
            assert!(det.reports().iter().all(|r| r.verdict == Verdict::Bonded { multi_centre: true }));
        }
    }

    /// The real water surface's shape (M6 Task 7, `validate` `water_has_two_oh_bonds_and_no_hh_bond`):
    /// oxygen's valence wide enough to fill the H–H chord's middle (so that chord has no interior
    /// minimum and takes Route D) and each O–H deformation lobe sitting 0.22 Bohr from its H, as
    /// the smeared O–H q maximum does on the GPU surface. Terms: O core, H₁, H₂, O valence, the
    /// two lobes (0.88 of the way from O to each H).
    fn grazed_water() -> Fixture {
        let o = [0.0; 3];
        let h1 = [1.43, 1.11, 0.0];
        let h2 = [-1.43, 1.11, 0.0];
        let lobe = |h: [f64; 3]| g(scale(h, 0.88), 0.09, 0.15);
        Fixture {
            terms: vec![g(o, 6.0, 0.18), g(h1, 0.12, 0.7), g(h2, 0.12, 0.7), g(o, 0.94, 1.08), lobe(h1), lobe(h2)],
            nuclei: vec![o, h1, h2],
        }
    }

    fn grazed_tables() -> Vec<RadialDensity> {
        let o_table = RadialDensity::from_fn(|r| 6.0 * (-r * r / 0.18).exp() + 0.94 * (-r * r / 1.08).exp(), R_MAX + 2.0, 0.005);
        let h_table = RadialDensity::from_fn(|r| 0.12 * (-r * r / 0.7).exp(), R_MAX + 2.0, 0.005);
        vec![o_table, h_table.clone(), h_table]
    }

    /// Condition 4: the H–H chord of `grazed_water` passes conditions 1–3 (Route D: q's interior
    /// maximum next to H₁ is the O–H lobe, seen from the side; transverse ratio inside ε_⊥; q above
    /// 1 + δ) and is rejected only because oxygen's promolecular term outweighs the far H's there —
    /// the scan's independent verdict, from the analytic atom terms. The O–H pairs stay bonded on
    /// Route D. Rotated about the H–H axis by 45° and by the seeded random rotation of
    /// `the_march_is_rotation_invariant`: identical verdicts and numbers.
    #[test]
    fn a_neighbours_bond_seen_from_the_side_is_not_the_pairs() {
        let fx = grazed_water();
        let oh = scan_water(&fx, 0, 1);
        let hh = scan_water(&fx, 1, 2);
        println!("scan O–H: {oh:?}\nscan H–H: {hh:?}");
        assert_eq!((oh.route, oh.verdict), (Route::Deformation, Verdict::Bonded { multi_centre: false }));
        // conditions 1–3 pass on the H–H chord: without condition 4 this would be a bond
        assert_eq!(hh.route, Route::Deformation);
        assert!(hh.enhancement >= 1.0 + DELTA && hh.transverse <= 1.0 + EPS_TRANSVERSE && hh.rho_min >= RHO_FLOOR, "{hh:?}");
        let p_hh = add(fx.nuclei[1], scale(sub(fx.nuclei[2], fx.nuclei[1]), hh.s));
        let p_oh = add(fx.nuclei[0], scale(sub(fx.nuclei[1], fx.nuclei[0]), oh.s));
        let w = |p: [f64; 3]| [fx.terms[0].at(p) + fx.terms[3].at(p), fx.terms[1].at(p), fx.terms[2].at(p)];
        // the H–H chord is symmetric: its q maximum sits next to either H (the scan takes the last)
        let near_h = |p: [f64; 3]| dist(p, fx.nuclei[1]).min(dist(p, fx.nuclei[2]));
        println!("H–H point {:.3} Bohr from the nearer H, weights (O, H₁, H₂) {:?}; O–H point {:.3} from H₁, weights {:?}", near_h(p_hh), w(p_hh), near_h(p_oh), w(p_oh));
        assert!(near_h(p_hh) < 0.2 && near_h(p_oh) < 0.2, "both points inside the H's smear, as on the real surface");
        assert_eq!(hh.verdict, Verdict::NeighbourRegion { atom: 0 });
        let tables = grazed_tables();
        let base = detect_with(&fx, &tables);
        for r in base.reports() {
            println!("report: {r:?}");
        }
        let (r_oh1, r_oh2, r_hh) = (report_of(&base, 0, 1), report_of(&base, 0, 2), report_of(&base, 1, 2));
        for r in [r_oh1, r_oh2] {
            assert_eq!((r.route, r.verdict), (Some(Route::Deformation), oh.verdict));
            assert!((r.enhancement.unwrap() / oh.enhancement - 1.0).abs() <= 0.02, "{:?}", r.enhancement);
        }
        assert_eq!((r_hh.route, r_hh.verdict), (Some(Route::Deformation), hh.verdict));
        assert!((r_hh.enhancement.unwrap() / hh.enhancement - 1.0).abs() <= 0.02 && r_hh.transverse_ratio.unwrap() <= 1.0 + EPS_TRANSVERSE, "{r_hh:?}");
        let graph = base.graph(3, 0, 0.0);
        assert_eq!(graph.bonds.iter().map(|b| (b.i, b.j)).collect::<Vec<_>>(), vec![(0, 1), (0, 2)]);
        let mut rng = StdRng::seed_from_u64(7);
        let random_axis: [f64; 3] = std::array::from_fn(|_| rng.gen_range(-1.0..1.0));
        let random_angle: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
        let hh_axis = sub(fx.nuclei[2], fx.nuclei[1]);
        for (name, m) in [("45° about H–H", rotation(hh_axis, std::f64::consts::FRAC_PI_4)), ("random", rotation(random_axis, random_angle))] {
            let det = detect_with(&fx.rotated(fx.nuclei[1], m), &tables);
            for (a, b) in base.reports().iter().zip(det.reports()) {
                assert_eq!((a.i, a.j, a.verdict, a.route), (b.i, b.j, b.verdict, b.route), "{name}: {a:?} vs {b:?}");
                for (x, y) in [(a.rho_min, b.rho_min), (a.enhancement, b.enhancement)] {
                    assert!((x.unwrap() - y.unwrap()).abs() <= 1e-9 * x.unwrap().abs().max(1.0), "{name}: {x:?} vs {y:?}");
                }
                // the H–H chord is mirror-symmetric: its two equal q maxima (next to either H) tie,
                // and rounding under the rotation may pick the other one — s or 1 − s
                let (x, y) = (a.s.unwrap(), b.s.unwrap());
                assert!((x - y).abs() <= 1e-9 || (a.i, a.j) == (1, 2) && (x - (1.0 - y)).abs() <= 1e-9, "{name}: s {x} vs {y}");
            }
        }
    }

    #[test]
    fn below_the_floor_is_not_a_bond_and_far_pairs_are_not_sampled() {
        let near = Fixture { terms: vec![g([-2.5, 0.0, 0.0], 0.3, 0.5), g([2.5, 0.0, 0.0], 0.3, 0.5)], nuclei: vec![[-2.5, 0.0, 0.0], [2.5, 0.0, 0.0]] };
        let det = near.detect();
        let r = report_of(&det, 0, 1);
        println!("5 Bohr: {r:?}");
        assert_eq!(r.verdict, Verdict::BelowFloor);
        assert_eq!(r.verdict, scan_pair(&near, 0, 1, H).unwrap().verdict);
        assert!(r.rho_min.is_some_and(|x| x < RHO_FLOOR && x > 0.0));
        let far = Fixture { terms: vec![g([-4.5, 0.0, 0.0], 0.3, 0.5), g([4.5, 0.0, 0.0], 0.3, 0.5)], nuclei: vec![[-4.5, 0.0, 0.0], [4.5, 0.0, 0.0]] };
        let det = Detector::new(&far.nuclei);
        assert!(det.points().is_empty(), "no chord points for a pair beyond R_MAX");
        let det = far.detect();
        let r = report_of(&det, 0, 1);
        assert_eq!(*r, PairReport { i: 0, j: 1, route: None, s: None, rho_min: None, transverse_ratio: None, rho_pro: None, enhancement: None, march_end: None, march_stop: None, verdict: Verdict::TooFar, laplacian: None, laplacian_noise: None });
        assert!(scan_pair(&far, 0, 1, H).is_none());
        assert_eq!(det.graph(2, 0, 0.0).components, vec![vec![0], vec![1]]);
    }

    /// MUST-1 / R-2: the bonded-pair fixture on a 49³ h = 0.25 grid, sampled trilinearly, with the
    /// promolecule from shell-binned tables of the same gridded atoms, over 24 placements (the
    /// pair shifted by 0, h/4, h/2, 3h/4 along x, in 6 orientations). The spreads are measured
    /// and recorded (`references.md`, "Tools (M6)").
    ///
    /// Two statistics, two ceilings (review I1):
    /// - **q at fixed s = 0.5** (the midpoint through the operator) over the placements: its
    ///   spread is the same-operator residual itself, bounded by the derivation. Trilinear
    ///   interpolation errs by Σ_a (h²/2)·t_a(1 − t_a)·∂²_a f (exact for the mixed terms), so
    ///   q_tri/q − 1 = Σ_a (h²/2)·t_a(1 − t_a)·D_aa with D_aa = ∂²_aρ/ρ − ∂²_aρ_pro/ρ_pro on the
    ///   lab axes (module doc); from a node (t = 0) to a cell centre (t = ½) that ranges over
    ///   (h²/8)·Σ_a|D_aa|·q. For this fixture at the midpoint D_u = 0.4410/0.3168 −
    ///   1.0410/0.1668 = −4.848 along the chord and D_⊥ = 0 exactly across it (every term has
    ///   the same width), so the ceiling is (h²/8)·4.848·1.8992 = 0.0719 (computed below from
    ///   the Gaussians' second derivatives); the mean below the exact value. Measured 0.070 over
    ///   the 24 placements (u ∥ x alone: 1.9074 / 1.8624 / 1.8411 / 1.8624 across the four
    ///   x-shifts, 0.066; the fourth-order terms at h/w = 0.5 trim the leading order).
    ///   **The table's own term (Task 3 re-review item (a), split in Task 7).** The promolecule
    ///   comes from the shell-binned table, not the exact atom, so q also carries the table's
    ///   error through the same operator: with ε = ρ_pro^tri,table(c)/ρ_pro^tri,exact(c) − 1 (the
    ///   table's promolecule and the exact atoms' blended at the same eight nodes), q = Q/(1 + ε)
    ///   with Q = ρ_tri/ρ_pro^tri,exact the pure same-operator quantity. The 0.070 is 0.066 (u ∥ x:
    ///   the residual node → cell centre) + 0.0037 (1.9111 − 1.9074: chords along y and z, where
    ///   the shift along x is *across* the chord and the residual vanishes identically — every term
    ///   has the same x-profile, so trilinear interpolation in x acts on ρ and ρ_pro alike — so
    ///   what moves q there is ε alone). Asserted separately (Task 7 review M2): the u ∥ x spread
    ///   of q against (h²/8)·|D_u|·q = 0.0719 as the Task 3 re-review proposed, and of Q (the
    ///   residual alone, derived) against the same; for y/z, Q equal to the exact q to 1e-12 at
    ///   every shift (the residual is identically zero there, so the whole spread is ε's); all 24
    ///   against the derived compound spread(Q)/(1 + ε_min) + Q_max·(1/(1 + ε_min) − 1/(1 + ε_max)).
    /// - **the detector's q(p*)**, p* the sampled chord minimum: measure-and-record under the ≥ 3×
    ///   rule (δ follows from it; δ was raised from 0.10 to 0.11 by it: 3 × 0.0629/1.852 =
    ///   0.102). Its derived ceiling is compound (the reviewer's derivation): the fixed-point
    ///   residual plus the wander of the sampled minimum. Along the chord the exact q is
    ///   q − 1 = 0.8992·(1 − 5.12x²) (x the distance from the midpoint, Bohr), flat at the
    ///   bottom, so the minimum search — which picks the chord sample whose residual is most
    ///   negative — wanders ±1–2 samples of 0.05 Bohr (observed: s = 0.4375 … 0.5312), i.e. up
    ///   to x = 0.1 Bohr, where q is lower by ½|q″|x² = 0.8992·5.12·0.01 = 0.044 (computed below
    ///   as q_exact(0) − q_exact(0.1)). Together 0.072 + 0.044 = 0.116. That is also why no
    ///   placement reaches the fixed-point maximum 1.907 and the detector's spread (0.063) sits
    ///   below the fixed-point range. |mean − exact| ≤ 0.06 with the mean BELOW the exact value
    ///   (the safe direction: residual at the chosen points 0.034 + wander 0.014); the raw
    ///   ratio's spread ≥ 0.10 (what the correction removes); the transverse ratio's spread ≤
    ///   0.03; 3× the transverse spread ≤ ε_⊥.
    /// A real H₂ chord is as flat (Task 2: 0.1853 at the midpoint vs 0.1861 one sample away), so
    /// the wander is expected on the real surface too (Task 7 measures it).
    #[test]
    fn the_same_operator_cancels_the_interpolation_bias() {
        let cell = Cell { n: 49, h: H };
        let (amp, a): (f64, f64) = (0.3, 0.5);
        let g_at = |r2: f64, amp: f64| amp * (-r2 / a).exp();
        // ∂² of A·exp(−r²/a) along the axis through its centre at distance d, and across it
        let d2_along = |d2: f64, amp: f64| g_at(d2, amp) * (4.0 * d2 / (a * a) - 2.0 / a);
        let d2_across = |d2: f64, amp: f64| g_at(d2, amp) * (-2.0 / a);
        let rho_pro_mid = 2.0 * g_at(0.64, amp);
        let rho_mid = rho_pro_mid + 0.5 * amp;
        let exact_q = rho_mid / rho_pro_mid;
        let d_u = (2.0 * d2_along(0.64, amp) + d2_along(0.0, 0.5 * amp)) / rho_mid - 2.0 * d2_along(0.64, amp) / rho_pro_mid;
        let d_perp = (2.0 * d2_across(0.64, amp) + d2_across(0.0, 0.5 * amp)) / rho_mid - 2.0 * d2_across(0.64, amp) / rho_pro_mid;
        assert!(d_perp.abs() < 1e-12, "equal widths: D_⊥ = 0 exactly, {d_perp}");
        let bound_q = H * H / 8.0 * (d_u.abs() + 2.0 * d_perp.abs()) * exact_q;
        // the wander term: q along the chord, exact
        let q_along = |x: f64| {
            let atoms = g_at((x + 0.8) * (x + 0.8), amp) + g_at((x - 0.8) * (x - 0.8), amp);
            (atoms + g_at(x * x, 0.5 * amp)) / atoms
        };
        let wander = q_along(0.0) - q_along(0.1);
        let compound = bound_q + wander;
        // the atom's table from the gridded isolated atom at the cell centre
        let nodes = shell_nodes(&cell, shell_radius(&cell));
        let atom = g([0.0; 3], amp, a);
        let values: Vec<f64> = nodes.iter().map(|&p| atom.at(p)).collect();
        let table = RadialDensity::from_shells(&nodes, &values, [0.0; 3], 0.5 * H);
        let orientations: Vec<[f64; 3]> = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]];
        let (mut qc, mut qr, mut tr, mut qf) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        // per orientation class (0: u ∥ x, the shift along the chord; 1: u ∥ y or z, the shift
        // across it; 2: the diagonals): the fixed-s q, the table's operator-weighted error ε at c
        // and Q = q·(1 + ε), the same-operator ratio against the exact atoms
        let (mut q_class, mut big_q_class): ([Vec<f64>; 3], [Vec<f64>; 3]) = Default::default();
        let mut eps_all: Vec<f64> = Vec::new();
        for shift in [0.0, 0.25 * H, 0.5 * H, 0.75 * H] {
            for (oi, o) in orientations.iter().enumerate() {
                let class = match oi { 0 => 0, 1 | 2 => 1, _ => 2 };
                let u = scale(*o, 1.0 / norm(*o));
                let c = [shift, 0.0, 0.0];
                let (ra, rb) = (sub(c, scale(u, 0.8)), add(c, scale(u, 0.8)));
                let terms = vec![g(ra, amp, a), g(rb, amp, a), g(c, 0.5 * amp, a)];
                let gridded = GriddedGaussians::new(terms, cell.clone());
                let pro = Promolecule::new(vec![&table, &table], &[ra, rb], Some(&cell));
                let pro_exact = Promolecule::new(vec![&table, &table], &[ra, rb], None);
                let mut det = Detector::new(&[ra, rb]);
                let mut pts = det.points().to_vec();
                while !pts.is_empty() {
                    let rho = gridded.sample(&pts);
                    match det.feed(&rho, &pro, H) {
                        Some(next) => pts = next.to_vec(),
                        None => break,
                    }
                }
                let r = &det.reports()[0];
                assert_eq!(r.verdict, Verdict::Bonded { multi_centre: false }, "{r:?}");
                let p_star = add(ra, scale(sub(rb, ra), r.s.unwrap()));
                let raw = gridded.sample(&[p_star])[0] / pro_exact.exact(p_star);
                // q at fixed s = 0.5 (the midpoint c) through the same operator
                let q_fixed = pro.ratio(c, gridded.sample(&[c])[0]);
                // the table's error through the operator: the exact atoms blended at the same nodes
                let pro_exact_tri = trilinear_blend(&cell, c, |ijk| {
                    let nd = node_position(&cell, ijk);
                    let (da, db) = (dist(nd, ra), dist(nd, rb));
                    g_at(da * da, amp) + g_at(db * db, amp)
                });
                let eps = pro.trilinear(c) / pro_exact_tri - 1.0;
                q_class[class].push(q_fixed);
                big_q_class[class].push(q_fixed * (1.0 + eps));
                eps_all.push(eps);
                println!("  shift {shift:.4} along {:?}: s {:.4}, q(p*) {:.4}, q(s = 0.5) {q_fixed:.4}, raw {:.4}, transverse {:.4}", o, r.s.unwrap(), r.enhancement.unwrap(), raw, r.transverse_ratio.unwrap());
                qf.push(q_fixed);
                qc.push(r.enhancement.unwrap());
                qr.push(raw);
                tr.push(r.transverse_ratio.unwrap());
            }
        }
        let spread = |v: &[f64]| v.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - v.iter().cloned().fold(f64::INFINITY, f64::min);
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let (sq, sr, st, sf) = (spread(&qc), spread(&qr), spread(&tr), spread(&qf));
        let (mq, mr, mt, mf) = (mean(&qc), mean(&qr), mean(&tr), mean(&qf));
        println!("24 placements, 49³ h 0.25, bonded pair (exact q = {exact_q:.4}):");
        println!("  q at fixed s = 0.5: spread {sf:.4}, mean {mf:.4}; the derivation's residual (h²/8)·Σ|D_aa|·q = {bound_q:.4} (D_u = {d_u:.3}, D_⊥ = 0)");
        println!("  the detector's q(p*): spread {sq:.4}, mean {mq:.4} (exact − mean = {:.4}, {:.2} %); compound ceiling residual + wander {wander:.4} = {compound:.4}", exact_q - mq, 100.0 * (exact_q - mq) / exact_q);
        println!("  raw ratio ρ_tri/ρ_pro^exact: spread {sr:.4}, mean {mr:.4}");
        println!("  transverse ratio: spread {st:.4}, mean {mt:.4}");
        println!("  3 × q(p*) spread / mean = {:.4} (δ = {DELTA}); 3 × transverse spread = {:.4} (ε_⊥ = {EPS_TRANSVERSE})", 3.0 * sq / mq, 3.0 * st);
        let (s_x, s_qx, s_perp) = (spread(&q_class[0]), spread(&big_q_class[0]), spread(&q_class[1]));
        let big_q: Vec<f64> = big_q_class.iter().flatten().cloned().collect();
        let (e_min, e_max) = (eps_all.iter().cloned().fold(f64::INFINITY, f64::min), eps_all.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        let q_big_max = big_q.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let compound_fixed = spread(&big_q) / (1.0 + e_min) + q_big_max * (1.0 / (1.0 + e_min) - 1.0 / (1.0 + e_max));
        let perp_dev = big_q_class[1].iter().map(|&x| (x / exact_q - 1.0).abs()).fold(0.0f64, f64::max);
        println!("  split (re-review item (a), review M2): u ∥ x: q spread {s_x:.4}, Q spread {s_qx:.4} vs the residual {bound_q:.4}; u ∥ y, z: q spread {s_perp:.4}, max |Q/q_exact − 1| = {perp_dev:.1e} (the residual is 0 there); ε ∈ [{e_min:.2e}, {e_max:.2e}]; all 24: q spread {sf:.4}, Q spread {:.4} vs the compound {compound_fixed:.4}", spread(&big_q));
        assert!(s_x <= bound_q, "u ∥ x fixed-s q spread {s_x} beyond the residual {bound_q}");
        assert!(s_qx <= bound_q, "u ∥ x same-operator Q spread {s_qx} beyond the residual {bound_q}");
        assert!(perp_dev <= 1e-12, "u ⊥ shift: Q deviates from the exact q by {perp_dev:e} — the residual should vanish identically");
        assert!(spread(&big_q) <= bound_q, "Q spread {} beyond the residual {bound_q}", spread(&big_q));
        assert!(sf <= compound_fixed + 1e-12, "fixed-s q spread {sf} beyond spread(Q)/(1 + ε_min) + Q_max·Δ(1/(1 + ε)) = {compound_fixed}");
        assert!(mf < exact_q, "the residual lifts the fixed-s mean above the exact value: {mf} vs {exact_q}");
        assert!(sq <= compound, "the detector's q(p*) spread {sq} beyond residual + wander {compound}");
        assert!((mq - exact_q).abs() <= 0.06 && mq < exact_q, "mean {mq} vs exact {exact_q}");
        assert!(sr >= 0.10, "raw spread {sr}: the correction has nothing to remove?");
        assert!(st <= 0.03, "transverse spread {st}");
        assert!(3.0 * sq / mq <= DELTA, "δ must be raised to {:.3}", 3.0 * sq / mq);
        assert!(3.0 * st <= EPS_TRANSVERSE, "ε_⊥ must be raised to {:.3}", 3.0 * st);
    }

    #[test]
    fn corrected_leaves_rho_alone_beyond_the_tables() {
        let cell = Cell { n: 49, h: H };
        let table = RadialDensity::from_fn(|r| 0.3 * (-r * r / 0.5).exp(), 3.0, 0.01);
        let pro = Promolecule::new(vec![&table], &[[0.0; 3]], Some(&cell));
        // outside the grid (10 Bohr from the atom): the operator reads nothing, ρ is left alone
        for p in [[10.0, 0.0, 0.0], [0.0, -7.0, 0.0]] {
            assert_eq!(pro.trilinear(p), 0.0, "{p:?}");
            assert_eq!(pro.corrected(p, 0.123), 0.123, "{p:?}");
            assert!(pro.corrected(p, 0.123).is_finite());
            assert_eq!(pro.ratio(p, 0.123), 0.0);
        }
        // inside the grid beyond the table's last entry (3.0): the tail continues, no step
        // (the Gaussian's log-slope at r = 3 is −12 per Bohr: continuity is checked at ±1e-6)
        let (inside, outside) = (pro.exact([3.0 - 1e-6, 0.0, 0.0]), pro.exact([3.0 + 1e-6, 0.0, 0.0]));
        assert!(outside > 0.0 && (outside / inside - 1.0).abs() < 1e-4, "{inside} vs {outside}");
        assert!(pro.trilinear([5.0, 0.0, 0.0]) > 0.0 && pro.corrected([5.0, 0.0, 0.0], 0.123).is_finite());
        // inside the table: the correction is the ratio of the two operators, ≈ 1 near a node
        let p = [0.5, 0.25, 0.0];
        assert!((pro.corrected(p, 1.0) - 1.0).abs() < 1e-12, "a node is read exactly");
        let q = [0.625, 0.125, 0.0];
        let c = pro.corrected(q, 1.0);
        assert!(c > 1.0 && c < 1.1, "a cell midpoint of a concave profile: trilinear underestimates, so the correction lifts: {c}");
        // ungridded: the identity
        let id = Promolecule::new(vec![&table], &[[0.0; 3]], None);
        assert_eq!(id.corrected(q, 0.7), 0.7);
        assert!((id.ratio(q, 0.7) - 0.7 / id.exact(q)).abs() < 1e-15);
    }

    #[test]
    fn components_are_connected_components() {
        let b = |i, j| Bond { i, j, rho_bcp: 0.2, enhancement: 1.5, s: 0.5, route: Route::Saddle, multi_centre: false, laplacian_bcp: -1.0, laplacian_noise: 0.01 };
        assert_eq!(components(6, &[b(0, 1), b(1, 2), b(3, 4)]), vec![vec![0, 1, 2], vec![3, 4], vec![5]]);
        assert_eq!(components(3, &[]), vec![vec![0], vec![1], vec![2]]);
        assert_eq!(components(4, &[b(2, 3), b(0, 3), b(1, 2)]), vec![vec![0, 1, 2, 3]]);
        assert_eq!(components(0, &[]), Vec::<Vec<usize>>::new());
    }

    #[test]
    fn round_one_has_33_points_per_pair_with_the_endpoints() {
        let fx = Fixture { terms: vec![g([0.0; 3], 0.3, 0.5), g([1.5, 0.0, 0.0], 0.3, 0.5), g([0.0, 1.5, 0.0], 0.3, 0.5)], nuclei: vec![[0.0; 3], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]] };
        let tables = fx.tables();
        let pro = Promolecule::new(tables.iter().collect(), &fx.nuclei, None);
        let mut det = Detector::new(&fx.nuclei);
        assert_eq!(det.points().len(), 3 * 33);
        for (k, (i, j)) in [(0, 1), (0, 2), (1, 2)].into_iter().enumerate() {
            let p = &det.points()[33 * k..33 * (k + 1)];
            for a in 0..3 {
                assert!((p[0][a] - fx.nuclei[i][a]).abs() <= 1e-15 && (p[32][a] - fx.nuclei[j][a]).abs() <= 1e-15);
            }
        }
        let rho: Vec<f64> = det.points().iter().map(|&p| fx.rho(p)).collect();
        let next = det.feed(&rho, &pro, H).expect("round 2").to_vec();
        let survivors = det.reports().iter().filter(|r| r.verdict == Verdict::TooFar && r.route.is_some()).count();
        assert_eq!(next.len(), 4 * survivors);
        assert!(survivors >= 1, "{:?}", det.reports());
    }

    /// The grid operator mirrored here is `qm::Grid::trilinear_cell` / `trilinear`: equal cells
    /// and fractions, equal blends of a stored field, on random points, nodes and faces.
    #[test]
    fn the_sim_trilinear_mirror_matches_qm() {
        let cell = Cell { n: 9, h: 0.5 };
        let grid = qm::Grid::cube(9, 0.5);
        let f = grid.sample(|[x, y, z]| (x * 1.3 - y * y + z * 0.7 * x).exp());
        let mut rng = StdRng::seed_from_u64(11);
        for _ in 0..200 {
            let p: [f64; 3] = std::array::from_fn(|_| rng.gen_range(-2.1..2.1));
            assert_eq!(trilinear_cell(&cell, p), grid.trilinear_cell(p), "{p:?}");
            let mine = trilinear_blend(&cell, p, |ijk| f[grid.idx(ijk[0], ijk[1], ijk[2])]);
            let theirs = grid.trilinear(&f, p);
            assert!((mine - theirs).abs() <= 1e-15 * theirs.abs().max(1.0), "{p:?}: {mine} vs {theirs}");
        }
        for idx in [0, grid.idx(8, 8, 8), grid.idx(8, 0, 4), grid.idx(4, 4, 4), grid.idx(0, 8, 8)] {
            let p = grid.point(idx);
            assert_eq!(node_position(&cell, [idx % 9, (idx / 9) % 9, idx / 81]), p);
            assert_eq!(trilinear_blend(&cell, p, |ijk| f[grid.idx(ijk[0], ijk[1], ijk[2])]).to_bits(), f[idx].to_bits());
        }
        assert_eq!(trilinear_blend(&cell, [2.0000001, 0.0, 0.0], |_| 1.0), 0.0);
    }

    /// `Gaussians::sample_density` is exact and `GriddedGaussians` is the trilinear of the node
    /// values — the test providers the fixtures above rely on.
    #[test]
    fn the_test_providers_sample_as_documented() {
        let terms = vec![g([0.3, -0.2, 0.1], 0.4, 0.6)];
        let mut exact = Gaussians::fixed(terms.clone());
        let cell = Cell { n: 17, h: 0.25 };
        let mut gridded = GriddedGaussians::new(terms.clone(), cell.clone());
        gridded.reset();
        let p = [0.37, 0.11, -0.52];
        assert_eq!(exact.sample(&[p])[0], terms[0].at(p));
        let node = node_position(&cell, [9, 7, 8]);
        assert_eq!(gridded.sample(&[node])[0].to_bits(), terms[0].at(node).to_bits());
        let s = gridded.sample(&[p])[0];
        assert!(s != terms[0].at(p) && (s / terms[0].at(p) - 1.0).abs() < 0.05, "trilinear, close: {s} vs {}", terms[0].at(p));
        assert_eq!(gridded.sample(&[[9.0, 0.0, 0.0]])[0], 0.0);
        // through the trait: the exact provider answers after an evaluation
        use crate::provider::ForceProvider;
        let scene = crate::scene::Scene { nuclei: vec![], cell, charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3], ionic: crate::scene::IonicPolicy::None };
        exact.evaluate(&scene, &mut || false).unwrap();
        assert_eq!(exact.sample_density(&[p]).unwrap()[0], terms[0].at(p));
    }

    // ---- bond strength: ∇²ρ at the bond point (module doc, "Bond strength") ----

    /// ∇² of amp·exp(−r²/2w²) = term·(r²/w⁴ − 3/w²).
    fn lap_term(t: &Term, p: [f64; 3]) -> f64 {
        let r2 = dot(sub(p, t.centre), sub(p, t.centre));
        let w2 = t.w * t.w;
        t.at(p) * (r2 / (w2 * w2) - 3.0 / w2)
    }

    /// The 8th-order stencil's truncation bound for a sum of Gaussian terms (module doc, "The
    /// operator"): per axis |2 Σ m¹⁰ c_m / 10!|·h⁸·|f⁽¹⁰⁾|, with |∂¹⁰_x of amp·e^{−x²/2w²}| ≤
    /// amp·√(10!)·1.0865/w¹⁰ (Cramér's bound on the Hermite functions, the other axes' factors
    /// ≤ 1), three axes.
    fn stencil_truncation_bound(terms: &[Term], h: f64) -> f64 {
        let fact10: f64 = (1..=10).map(|k| k as f64).product();
        let moment: f64 = 2.0 * (1..=4).map(|m| (m as f64).powi(10) * LAPLACIAN_COEFF[m]).sum::<f64>() / fact10;
        terms.iter().map(|t| 3.0 * moment.abs() * h.powi(8) * t.amp.abs() * fact10.sqrt() * 1.0865 / t.w.powi(10)).sum()
    }

    /// The mirror of `qm::stencil::COEFF` is equal, and the one-node sum through samples at the
    /// stencil points (trilinear at nodes = the stored value; outside the box 0) is the solver's
    /// own ∇² at that node — interior, and within 4 nodes of a face (the Dirichlet terms).
    #[test]
    fn the_laplacian_stencil_mirrors_the_solvers() {
        assert_eq!(LAPLACIAN_COEFF, qm::stencil::COEFF);
        assert!((stencil_abs_sum() - (3.0 * 205.0 / 72.0 + 6.0 * (8.0 / 5.0 + 1.0 / 5.0 + 8.0 / 315.0 + 1.0 / 560.0))).abs() < 1e-14, "{}", stencil_abs_sum());
        let grid = qm::grid::Grid::cube(17, 0.3);
        let cell = Cell { n: 17, h: 0.3 };
        let f = grid.sample(|[x, y, z]| 0.4 * (-((x - 0.2) * (x - 0.2) + (y + 0.1) * (y + 0.1) + z * z) / 1.3).exp() + 0.05 * x);
        let mut out = vec![0.0; grid.len()];
        qm::stencil::laplacian(&grid, &f, &mut out);
        for ijk in [[8, 8, 8], [5, 9, 11], [1, 8, 8], [8, 15, 2], [0, 0, 16], [16, 3, 8]] {
            let idx = grid.idx(ijk[0], ijk[1], ijk[2]);
            let pts = laplacian_stencil(node_position(&cell, ijk), cell.h);
            let samples: Vec<f64> = pts.iter().map(|&p| grid.trilinear(&f, p)).collect();
            let mine = laplacian_from_stencil(&samples, cell.h);
            assert!((mine - out[idx]).abs() <= 1e-12 * out[idx].abs().max(1.0), "{ijk:?}: {mine} vs qm {}", out[idx]);
        }
    }

    /// Analytic: ∇² of a Gaussian (w = 0.5) at 64 points. With exact samples the stencil centred
    /// on p errs within its O(h⁸) truncation bound; on a grid, at every point and spacing, the
    /// estimate's interpolation term covers the blend's error (module doc, "The noise"); and the
    /// blend errs at second order — observed over h = 0.2, 0.1, 0.05 at the cell centre nearest
    /// each point (t = ½ on every axis, so the leading error (h²/8)·Σ_a ∂²_a L is not mixed with a
    /// sub-cell position that changes with h).
    #[test]
    fn the_laplacian_at_a_point_converges_at_second_order() {
        let t = Term { centre: [0.03, -0.02, 0.01], amp: 0.3, w: 0.5 };
        let mut rng = StdRng::seed_from_u64(5);
        let pts: Vec<[f64; 3]> = (0..64).map(|_| std::array::from_fn(|_| rng.gen_range(-0.9..0.9))).collect();
        let bound = stencil_truncation_bound(&[t], 0.1);
        let worst_exact = pts
            .iter()
            .map(|&p| {
                let samples: Vec<f64> = laplacian_points(None, p, 0.1).unwrap().iter().map(|&x| t.at(x)).collect();
                (laplacian_at(None, p, 0.1, &samples) - lap_term(&t, p)).abs()
            })
            .fold(0.0f64, f64::max);
        println!("exact samples, h = 0.1: worst |ΔL| = {worst_exact:.2e}, truncation bound {bound:.2e} (scale |L| ≤ {:.2})", 3.0 * t.amp / (t.w * t.w));
        assert!(worst_exact <= bound, "{worst_exact:e} > {bound:e}");
        let mut errs = Vec::new();
        for h in [0.2, 0.1, 0.05] {
            let n = (8.0 / h) as usize + 1;
            let cell = Cell { n, h };
            let (mut worst, mut worst_ratio) = (0.0f64, 0.0f64);
            for &p in &pts {
                let off = scale([1.0, 0.0, 0.0], h);
                let (prev, next) = (sub(p, off), add(p, off));
                let samples: Vec<f64> = strength_points(Some(&cell), p, prev, next, h).unwrap().iter().map(|&x| t.at(x)).collect();
                let est = strength_from(Some(&cell), p, prev, next, h, &samples);
                let blended: Vec<f64> = laplacian_points(Some(&cell), p, h).unwrap().iter().map(|&x| t.at(x)).collect();
                assert_eq!(est.value, laplacian_at(Some(&cell), p, h, &blended), "the estimate's value is the 8-node blend");
                let e = (est.value - lap_term(&t, p)).abs();
                worst = worst.max(e);
                worst_ratio = worst_ratio.max(e / (est.interpolation + est.rounding + stencil_truncation_bound(&[t], h)));
            }
            println!("  h = {h}: worst |ΔL| = {worst:.3e}; worst error / (interpolation + rounding + truncation) = {worst_ratio:.3}");
            assert!(worst_ratio <= 1.0, "h = {h}: the interpolation estimate does not cover the error ({worst_ratio})");
            // the order: at the centre of the cell holding each point
            let centre_err = pts
                .iter()
                .map(|&p0| {
                    let (c, _) = trilinear_cell(&cell, p0).unwrap();
                    let p = add(node_position(&cell, c), [0.5 * h; 3]);
                    let blended: Vec<f64> = laplacian_points(Some(&cell), p, h).unwrap().iter().map(|&x| t.at(x)).collect();
                    (laplacian_at(Some(&cell), p, h, &blended) - lap_term(&t, p)).abs()
                })
                .fold(0.0f64, f64::max);
            errs.push(centre_err);
        }
        let orders: Vec<f64> = errs.windows(2).map(|e| (e[0] / e[1]).log2()).collect();
        println!("gridded, h = 0.2 / 0.1 / 0.05: worst |ΔL| at the cell centres = {:?}; observed orders {orders:.3?}", errs.iter().map(|e| format!("{e:.3e}")).collect::<Vec<_>>());
        for o in &orders {
            assert!((1.8..=2.2).contains(o), "order {o}, expected 2");
        }
    }

    #[test]
    fn bond_kind_classifies_by_sign_against_the_band() {
        assert_eq!(BondKind::classify(-1.0, 0.1), BondKind::Shared);
        assert_eq!(BondKind::classify(0.5, 0.1), BondKind::ClosedShell);
        assert_eq!(BondKind::classify(0.05, 0.1), BondKind::Indeterminate);
        assert_eq!(BondKind::classify(-0.1, 0.1), BondKind::Indeterminate);
        assert_eq!(BondKind::classify(f64::NAN, 0.1), BondKind::Indeterminate);
        assert_eq!(BondKind::classify(-1.0, f64::NAN), BondKind::Indeterminate);
        let e = LaplacianEstimate { value: -1.0, interpolation: 0.01, location: 0.02, rounding: 1e-6 };
        assert!((e.band() - 3.0 * 0.030001).abs() < 1e-15);
    }

    /// Two atoms (amp 0.3, a = 0.5) 2.4 Bohr apart and a wide bump (a = 2) of amplitude b at the
    /// midpoint: the atoms' tails give ∇²ρ = +0.3718 there, the bump −3b, so the sign turns at
    /// b* = 0.12395. Exact samples: every case is `Bonded` on Route S at s = 0.5, ∇²ρ_b is the
    /// analytic value within the stencil's truncation bound, and the kind follows its sign —
    /// shared at b = 0.2, closed-shell at b = 0.05, indeterminate at b*. A He₂-like
    /// `NotAccumulated` pair gets its Laplacian too (positive: closed-shell), a `BelowFloor` one
    /// none.
    #[test]
    fn the_bond_kind_follows_the_sign_of_the_laplacian() {
        let (a, b_) = ([-1.2, 0.0, 0.0], [1.2, 0.0, 0.0]);
        let atoms_lap = 2.0 * lap_term(&g(a, 0.3, 0.5), [0.0; 3]);
        let b_star = atoms_lap / 3.0;
        println!("atoms' ∇²ρ at the midpoint {atoms_lap:.5}, sign change at b* = {b_star:.5}");
        for (b, want) in [(0.2, BondKind::Shared), (0.05, BondKind::ClosedShell), (b_star, BondKind::Indeterminate)] {
            let fx = Fixture { terms: vec![g(a, 0.3, 0.5), g(b_, 0.3, 0.5), g([0.0; 3], b, 2.0)], nuclei: vec![a, b_] };
            let det = fx.detect();
            let r = report_of(&det, 0, 1);
            let exact: f64 = fx.terms.iter().map(|t| lap_term(t, [0.0; 3])).sum();
            let bound = stencil_truncation_bound(&fx.terms, H);
            println!("b = {b:.5}: ∇²ρ_b {:.6} (exact {exact:.6}, truncation bound {bound:.2e}), band {:.4e}, kind {:?}", r.laplacian.unwrap(), r.laplacian_noise.unwrap(), r.kind());
            assert_eq!((r.verdict, r.route, r.s), (Verdict::Bonded { multi_centre: false }, Some(Route::Saddle), Some(0.5)), "{r:?}");
            assert!((r.laplacian.unwrap() - exact).abs() <= bound, "{} vs {exact}", r.laplacian.unwrap());
            assert_eq!(r.kind(), Some(want), "b = {b}: {r:?}");
            let gr = det.graph(2, 0, 0.0);
            assert_eq!(gr.bonds[0].kind(), want);
            assert_eq!((gr.bonds[0].laplacian_bcp, gr.bonds[0].laplacian_noise), (r.laplacian.unwrap(), r.laplacian_noise.unwrap()));
        }
        // a depleted overlap (He₂-like: a dip at the midpoint) is NotAccumulated and still classified
        let (c, d) = ([-0.9, 0.0, 0.0], [0.9, 0.0, 0.0]);
        let he = Fixture { terms: vec![g(c, 0.3, 0.5), g(d, 0.3, 0.5), g([0.0; 3], -0.01, 0.3)], nuclei: vec![c, d] };
        let det = he.detect();
        let r = report_of(&det, 0, 1);
        let exact: f64 = he.terms.iter().map(|t| lap_term(t, [0.0; 3])).sum();
        println!("depleted pair: {:?}, ∇²ρ_b {:.5} (exact {exact:.5}), band {:.4e}", r.verdict, r.laplacian.unwrap(), r.laplacian_noise.unwrap());
        assert_eq!(r.verdict, Verdict::NotAccumulated);
        assert!((r.laplacian.unwrap() - exact).abs() <= stencil_truncation_bound(&he.terms, H) && exact > 0.0);
        assert_eq!(r.kind(), Some(BondKind::ClosedShell));
        let far = Fixture { terms: vec![g([-2.5, 0.0, 0.0], 0.3, 0.5), g([2.5, 0.0, 0.0], 0.3, 0.5)], nuclei: vec![[-2.5, 0.0, 0.0], [2.5, 0.0, 0.0]] };
        let r = report_of(&far.detect(), 0, 1).clone();
        assert_eq!((r.verdict, r.laplacian, r.kind()), (Verdict::BelowFloor, None, None));
    }

    /// The noise measured the way δ was (module doc, "The noise"): the bonded-pair fixture of
    /// `the_same_operator_cancels_the_interpolation_bias` gridded at its 24 placements (4 shifts ×
    /// 6 orientations), at h = 0.25 (49³) and h = 0.125 (97³), ∇²ρ_b read from the detector. At
    /// every placement the estimate's noise covers the distance from ∇²ρ_b to the exact Laplacian
    /// at the true bond point (the midpoint: the fixture is symmetric) — interpolation at p* plus
    /// p*'s chord-sample location — and the spread over the placements is within twice the largest
    /// noise. The mean over placements converges to the exact value.
    #[test]
    fn the_laplacian_at_the_bond_point_converges_and_its_noise_is_measured() {
        let (amp, a): (f64, f64) = (0.3, 0.5);
        let orientations: Vec<[f64; 3]> = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]];
        let exact_mid = 2.0 * lap_term(&g([0.8, 0.0, 0.0], amp, a), [0.0; 3]) + lap_term(&g([0.0; 3], 0.5 * amp, a), [0.0; 3]);
        for (n, h) in [(49usize, 0.25f64), (97, 0.125)] {
            let cell = Cell { n, h };
            let nodes = shell_nodes(&cell, shell_radius(&cell));
            let atom = g([0.0; 3], amp, a);
            let values: Vec<f64> = nodes.iter().map(|&p| atom.at(p)).collect();
            let table = RadialDensity::from_shells(&nodes, &values, [0.0; 3], 0.5 * h);
            let (mut lap, mut noise, mut cover) = (Vec::new(), Vec::new(), 0.0f64);
            let (mut interp, mut loc) = (Vec::new(), Vec::new());
            for shift in [0.0, 0.25 * h, 0.5 * h, 0.75 * h] {
                for o in &orientations {
                    let u = scale(*o, 1.0 / norm(*o));
                    let c = [shift, 0.0, 0.0];
                    let (ra, rb) = (sub(c, scale(u, 0.8)), add(c, scale(u, 0.8)));
                    let terms = vec![g(ra, amp, a), g(rb, amp, a), g(c, 0.5 * amp, a)];
                    // the gridded density, nodes evaluated on demand (the operator of GriddedGaussians)
                    let sample = |pts: &[[f64; 3]]| -> Vec<f64> { pts.iter().map(|&p| trilinear_blend(&cell, p, |ijk| terms.iter().map(|t| t.at(node_position(&cell, ijk))).sum())).collect() };
                    let pro = Promolecule::new(vec![&table, &table], &[ra, rb], Some(&cell));
                    let mut det = Detector::new(&[ra, rb]);
                    let mut pts = det.points().to_vec();
                    while !pts.is_empty() {
                        let rho = sample(&pts);
                        match det.feed(&rho, &pro, h) {
                            Some(next) => pts = next.to_vec(),
                            None => break,
                        }
                    }
                    let r = det.reports()[0].clone();
                    assert_eq!(r.verdict, Verdict::Bonded { multi_centre: false }, "{r:?}");
                    // the estimate's parts, recomputed from the same samples
                    let d = sub(rb, ra);
                    let at = |f: f64| add(ra, scale(d, f));
                    let s = r.s.unwrap();
                    let (p, prev, next) = (at(s), at(s - 1.0 / 32.0), at(s + 1.0 / 32.0));
                    let est = strength_from(Some(&cell), p, prev, next, h, &sample(&strength_points(Some(&cell), p, prev, next, h).unwrap()));
                    assert_eq!((est.value, est.band()), (r.laplacian.unwrap(), r.laplacian_noise.unwrap()));
                    cover = cover.max((est.value - exact_mid).abs() / (est.noise() + stencil_truncation_bound(&terms, h)));
                    lap.push(est.value);
                    noise.push(est.noise());
                    interp.push(est.interpolation);
                    loc.push(est.location);
                }
            }
            let spread = |v: &[f64]| v.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - v.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = |v: &[f64]| v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            println!(
                "h = {h}: ∇²ρ_b mean {:.4} (exact at the midpoint {exact_mid:.4}), spread {:.4e}; noise mean {:.4e} max {:.4e} (interpolation mean {:.4e}, location mean {:.4e}); worst |∇²ρ_b − exact|/noise {cover:.3}; spread / (2 max noise) {:.3}",
                mean(&lap), spread(&lap), mean(&noise), max(&noise), mean(&interp), mean(&loc), spread(&lap) / (2.0 * max(&noise))
            );
            assert!(cover <= 1.0, "h = {h}: the noise estimate misses the error by {cover:.3}×");
            assert!(spread(&lap) <= 2.0 * max(&noise), "h = {h}: spread {} beyond twice the largest noise {}", spread(&lap), max(&noise));
        }
    }
}
