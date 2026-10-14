# Physics review — M7 pseudopotential design spec

Reviewed: `docs/superpowers/specs/2026-10-14-pseudopotential-design.md` (branch `pseudopotential-design`), 2026-10-14, Fable (physics-reviewer, read-only).
Against: CLAUDE.md rules 1, 2, 4, 6, 7; design spec §6; `crates/qm/src/{nuclear.rs,forces.rs,poisson.rs,scf.rs}`; `crates/validate/references.md`.
Method: every equation re-derived; the claims where a wrong number still looks plausible were checked numerically
(1-electron radial eigenvalues of the smeared Coulomb well by a Sturm-sequence count on a 3-point FD Hamiltonian with
Richardson in h, verified on exact H to 2e-6 and on the small-σ first-order shift; the TM curvature condition by explicit
polynomial expansion for l = 0, 1, 2). Scratch scripts not committed; numbers quoted inline.

**Verdict: ready with fixes.** The recommendation (TM + KB + NLCC, generated in-house under a universal recipe) stands and
none of the MUST items changes it; each is a local correction of an equation, a gate, or a rule that would otherwise
produce a wrong-but-plausible result or block the design's own workflow.

---

## MUST

### M1 — §4.3: the TM zero-curvature condition is wrong by a factor of 2
Spec: "V_l^{scr}″(0) = 0 (the condition 2c₂² + c₄(2l + 5) = 0)".
Derivation with u = r^{l+1} e^{p(r)}, p = c₀ + c₂r² + c₄r⁴ + …:
  V_scr = ε + (l+1) p′/r + ½(p′² + p″),
  p′/r = 2c₂ + 4c₄r² + …,  p′² = 4c₂²r² + …,  p″ = 2c₂ + 12c₄r² + …,
  ⇒ V_scr(r) = ε + (2l+3)c₂ + [4(l+1)c₄ + 2c₂² + 6c₄] r² + O(r⁴) = ε + (2l+3)c₂ + 2[c₂² + (2l+5)c₄] r² + …
So V″(0) = 0 ⇔ **c₂² + (2l+5) c₄ = 0** (Troullier–Martins PRB 43, 1993, eq. 12). Verified by expanding the polynomial for
l = 0, 1, 2 with random c₂: the spec's c₄ = −2c₂²/(2l+5) leaves an r² coefficient of −2.2…−4.5 (V″(0) ≠ 0); the correct
c₄ gives 0 to rounding. If implemented as written the screened potential keeps a curved origin, the "designed for
smoothness" property is lost and h_req (T8) is silently worse. The same expansion gives V_scr(0) = ε + (2l+3)c₂ — the
"r → 0 limit taken analytically" the spec promises; state it.

### M2 — §4.1: the virial test "−V/T = 2 to 1e-5" is false for LSDA-PW92 and will fail by ~1 %
Uniform scaling ψ_λ(r) = λ^{3/2}ψ(λr): T_s → λ²T_s, E_en and E_H → λ(…), E_xc^{LDA}[ρ_λ] = ∫ρ(r) ε_xc(λ³ρ(r)) d³r.
Stationarity at λ = 1:
  **2T_s + E_en + E_H + 3[ Σ_σ ∫ρ_σ v_xc,σ d³r − E_xc ] = 0.**
For Dirac exchange (ε_x ∝ ρ^{1/3}) the bracket's exchange part equals E_x, recovering 2T + V = 0 with V = E_en + E_H + E_x.
For PW92 correlation it does not: in the high-density limit ε_c ≈ A ln r_s + …, so ρ dε_c/dρ = −A/3 ≈ −0.0104 Ha and
3∫ρ(v_c − ε_c) ≈ −0.031·N ≈ −0.25 Ha for O, against E_c ≈ −0.40 Ha; the ratio −V/T (T ≈ 74.5) is off at the 2–3e-3
level, 300× the 1e-5 gate, and ~1 % even for H. As written the test can only be "loosened" (rule 6) or misdiagnosed as a
solver bug. Fix: test the generalised relation above to 1e-5 (a stronger check of V_H, v_xc and the eigensolver than the
textbook form) and keep the hydrogenic ε_nl = −Z²/2n² check for the integrator alone.

### M3 — §5.6: the core-overlap rule R_IJ < r_c,I + r_c,J flags every bond under `All` and starves the surrogate
r_c is a pseudisation radius of a soft TM potential, not the extent of the frozen core. With the ladder: pseudo-H r_c = 1.6
(f = 1.6 × r_max,1s = 1.0), pseudo-O r_c,p ≈ 1.6 × 0.9 = 1.44, r_c,s ≈ 1.6 × 1.18 = 1.9. H₂ at R_e = 1.45 < 3.2; O–H at
1.81 < 1.44 + 1.6 = 3.0; every H₂O, NH₃, CH₄ and H₂ geometry of §6.4/§6.7 would carry the warning, the viewer tint, and —
per §5.5 — "the surrogate never trained on it", so `All` (the intended default) could never teach a model of any molecule.
Overlapping pseudisation spheres are the normal regime of norm-conserving potentials (TM's own O at r_c = 1.45 in water);
norm conservation is what makes it valid. The physical quantity is overlap of the *core densities*. Replace by a derived
core radius: r_core,I = radius enclosing all but a stated fraction (1e-3) of N_c in the AE atom (≈ 0.4–0.5 Bohr for O;
zero for H, He, which have no core) and warn when R_IJ < r_core,I + r_core,J. Universal, Z-derived, silent for ordinary bonds.

### M4 — §5.1: masking V_SR at γ_r r_c contradicts its own tail and cuts the erfc term
Paragraph 1: V_SR = −Z_v erfc(d/(σ_LR√2))/d beyond r_c, "below 1e-6 beyond d = r_c + 5σ_LR" (correct: at h = 0.25,
σ_LR = 0.5, that is 3.9 Bohr, where erfc(5.5) ≈ 1e-14). Paragraph 2: "each radial function g ∈ {V_SR, β_l, ρ_pc} …
multiplied by a smooth mask vanishing at γ_r r_c = 1.5 r_c ≈ 2.1 Bohr". At 2.1 Bohr the erfc tail is 6·erfc(2.97)/2.1 =
7.7e-5 Ha (h = 0.25) and 6·erfc(2.12)/2.1 = 7.7e-3 Ha at h = 0.35 (σ_LR = 0.7): masking there discards a long-range piece
of the *exact* split and breaks the "no physical approximation" claim. Fix: split V_SR into (i) the compact part
V_loc(d) + Z_v/d, identically zero beyond max(r_c, r_nlc), which is what gets filtered and masked at γ_r r_c, and (ii) the
analytic −Z_v erfc(d/(σ_LR√2))/d, already band-limited by σ_LR ≥ 2h (aliasing 2e^{−8π²}), sampled directly out to
r_c + 5σ_LR. Same split for dV_SR/dd in the force.

### M5 — §6.6: the 3e-5 egg-box gate under `CoresOnly` is below the floor the spec quotes for smeared H
`forces.rs` bounds the smeared-H egg-box force at ≤ 4e-5 Ha/Bohr per nucleus at σ = h = 0.25, and `CoresOnly` H₂O keeps
two such H's. After pseudising O, |Σ_I F_I| is bounded by the two H contributions (up to 8e-5, typically a few e-5)
whatever the pseudopotential does, so a 3e-5 gate on `CoresOnly` H₂O can fail for a reason the spec would attribute to the
pseudopotential. State the gate under `All` (pseudo-H is far softer: r_c = 1.6) or on an all-pseudo system (O₂, CO), and
record the `CoresOnly` number against the H floor as "expected ≈ 4e-5, the smeared-H residual". Also say which norm:
today's −3.4e-4 is the y-component of the vector sum at one geometry; the 8-shift probe should report max over shifts of
|Σ_I F_I| and the rms energy ripple.

### M6 — §1.1 table: the "1 Ha class at Zσ ≲ 1" row is wrong by ~20× (conclusion unchanged)
By scaling, a hydrogenic 1s in the smeared well has E(Z, σ) = Z² E(1, Zσ). Computed E(1, Zσ) = −0.4429, −0.3751,
−0.2841, −0.1930 at Zσ = 0.25, 0.5, 1, 2 (bare nucleus, one electron; exact H limit −0.49999 reproduced; at σ = 0.1 the
shift 0.0144 is 0.72× the first-order 2πρ(0)σ² = 0.020, matching the 0.71–0.77 relaxation ratio in `references.md`).
For Z = 8 the shift per 1s electron is 3.65, 7.99, 13.8, 19.7 Ha. Two 1s electrons at Zσ = 2 give 39 Ha ≈ the measured
74.5 − 36.6 = 38 Ha — the "halving" is exactly the two 1s electrons, confirmed. But at Zσ = 1 the shift is still ≈ 28 Ha
bare (≈ 20 relaxed), not "1 Ha": the 1 Ha class needs 2·64·0.72·2(Zσ)² ≲ 1 ⇒ Zσ ≈ 0.07 ⇒ σ ≈ 0.009 Bohr — the same h-class
as the 40 mHa row, not 9× the points. Replace the row (or relabel it "~20 Ha class") so no later reader prices h = 0.12 as
a 1 Ha route. Same table, row 1 vs the text: the text cites `references.md`'s "Z·h ≲ 0.2" (⇒ h ≲ 0.025, 10³× points,
10⁴× cost) but the row applies Zσ_max ≲ 0.2 with σ_max = 1.6h (⇒ h ≲ 0.016); pick one.
§1.2 likewise: the harmonic estimate V(0) + 3ω/2 = −8.0 Ha is an upper bound (the true well lies below the parabola); the
actual bare-smeared 1s eigenvalue at Zσ = 2 is 64·(−0.193) = −12.3 Ha against the bare −32 Ha, i.e. ≈ 20 Ha too high per
electron, and comparing −8 with the *screened* AE eigenvalue −18.8 mixes two references. The mechanism paragraph is right;
fix the numbers.

---

## SHOULD

### S1 — §4.6/§5.3: pin the projector normalisation when reusing the Racah solid harmonics
V_NL = |δVφ⟩⟨δVφ|/⟨φ|δV|φ⟩ is invariant to the scale of φ_lm only if the same function is used in numerator and
denominator. §4.6 defines E_l^{KB} = 1/⟨φ_l|δV_l|φ_l⟩ as a radial integral (orthonormal Y_lm implied), while §5.3 builds
β_lm on the grid from `qm::poisson`'s R_lm = r^l S_lm with S_lm Racah-normalised (∫|S_lm|²dΩ = 4π/(2l+1)). Used as
written, V_NL is scaled by 4π/(2l+1) (×4π for s, ×4π/3 for p). Write the grid function explicitly:
  β_lm(r) = δV_l(r) [u_l^{PP}(r)/r^{l+1}] √((2l+1)/4π) R_lm(r − R_I)
(the r^{−(l+1)} table is finite at the origin since u ∝ r^{l+1}), and make Task 6's first failing test the operator
identity on the grid: sample the reference pseudo-orbital φ_l0, apply V_NL, compare to δV_l φ_l0 pointwise (< 1e-6) and
⟨φ|V_NL|φ⟩ to ⟨φ|δV|φ⟩. The §6.3 Ne test would catch the factor, but a day later and inside an SCF.

### S2 — §4.8 T8 / §5.1: the gates certify the unfiltered potential; the grid sees the filtered one
T1–T7 run on the radial TM/KB functions; the grid gets functions cut at q_c = 0.7π/h and masked at 1.5 r_c, and the
molecule's XC uses the filtered ρ_pc while the unscreening (§4.3) used the unfiltered one. Re-run at least T1 (eigenvalues)
and T3 (the four ΔE) in the radial KB atom with the *filtered* tables at the scene's h — seconds — and record the filter's
own ΔE in the generation record. This also replaces T8's two unmotivated constants (1e-4 of max, c_q = 1.5) by one with
units: **h_req = the coarsest h on {0.35, 0.30, 0.25, 0.20, 0.15} at which the filtered radial KB atom reproduces the
unfiltered total energy and valence eigenvalues within 1 mHa** (the T1 number). "1e-4 of the transform's maximum" is a
function-level criterion with no derived relation to an energy; the first-order truncation error
ΔE_l ≈ 2 f_l E_l^{KB} c_l ∫_{q_c}^∞ β̃_l(q) ũ_l(q) q² dq is weighted by the orbital's own spectrum and can be far smaller
than 1e-4 relative suggests — or larger, for the partial core. Until measured, "h = 0.30–0.35 becomes usable" (§7) is
unsupported.

### S3 — §4.5/§5.1: the filtered partial core must stay non-negative
PW92 is evaluated on ρ_v,σ + ½ρ_pc. ρ_pc = A sin(Br)/r inside r_nlc with a slope-matched crossover has a spectrum reaching
~2π/r_nlc (≈ 10–12 Bohr⁻¹ for r_nlc ≈ 0.5–0.6 at O), at or above q_c = 8.8 at h = 0.25; a King-Smith cut can ring
negative in the tail. Add to Task 5's tests: min_r ρ_pc^{filtered}(r) ≥ 0 (or ≥ −1e-8 with a documented clamp in the XC
kernel).

### S4 — §4.8 T5: what 0.05 rad means, and gate in energy units (the derivation the spec asks for)
From the Wronskian of two solutions at neighbouring energies, for −½u″ + (V_l − E)u = 0:
  ∂D/∂E |_{r_d} = −(2/u(r_d)²) ∫₀^{r_d} u² dr,   D = u′/u
— the norm-conservation identity (equal norms inside r_d ⇒ equal ∂D/∂E at ε_l). A log-derivative mismatch δD at energy E
is therefore equivalent to an eigenvalue error
  **δE_equiv(E) = δD · u(r_d)²/(2∫₀^{r_d}u²) = δθ (1 + D²) u(r_d)²/(2∫₀^{r_d}u²),   θ = atan D.**
For O 2p at r_d ≈ 1.7 Bohr (hydrogenic Z_eff = 4.45: u(r_d)² ≈ 0.29, ∫₀^{r_d}u² ≈ 0.7, D ≈ −1.1) this gives
δE_equiv ≈ 0.05 × 2.1 × 0.21 ≈ 22 mHa at the window edges (±1 Ha from ε_l). Because the first-order term vanishes by norm
conservation, the mismatch grows ~quadratically in (E − ε_l), so at the ±0.3–0.5 Ha where bonding orbitals sit it is
≈ 2–5 mHa — coherent with T3/T4's 5 mHa. So 0.05 rad is sound for O, but the *same* radian threshold means a different
mHa for a diffuse channel (small u(r_d)²): the gate is element-independent in form, not in meaning. Gate δE_equiv(E)
instead (≤ 20 mHa over the window), which means the same thing for every channel and element, and state which l are
tested: the projector channels and the local channel — not the ungenerated l (d for the first row), where the local p
potential is not the AE d scattering and the test would fail for a reason the recipe accepts. The pole bookkeeping
(exclude ±25 mHa, partner within 20 mHa) is only just consistent — a PP pole 20 mHa off the AE one leaves |Δθ| ≈ π/2 at
the edge of the excluded band; δE_equiv has no pole and removes the issue.

### S5 — §4.2 step 3: a barely-bound l_max channel gives an absurd r_c
r_c,l = f·r_max,l applied to a weakly bound diffuse state (Al–Ar 3d at ε ≈ −0.02 Ha, r_max ≈ 3–4 Bohr) yields r_c ≈ 5–6
Bohr and a projector sphere of radius 9 Bohr — Task 12 will hit this. Universal fix (matches practice, no per-element
input): drop a channel whose reference eigenvalue lies above ε_v,max (it is not a valence state), or set
r_c,l_max = max_{l<l_max} r_c,l. The first row is unaffected (3d is unbound in the neutral LSDA atom, so s and p only,
one s projector with p local — as the spec says).

### S6 — §4.4/§9: make the ladder's outputs visible, and give H a molecular gate
(a) `gate_thresholds_are_constants` guards the inputs; add the outputs: a pinned table in `references.md` of
(Z → f, local channel, r_c,l, Z_v, E^{KB}, h_req) for every accepted element, regenerated and diffed by the validation
task, so a change to any universal constant that moves one element's f is visible in review — the mechanical form of the
§9 "discipline" guard. (b) Pseudo-H at f = 1.6 (r_c = 1.6 > R_e = 1.45) will pass every *atomic* gate — T5 at
r_d = 1.9 Bohr lies outside the bond — and the ladder has no mechanism to descend if §6.4's H₂ test (R_e within 0.02 Bohr,
ω within 5 % of the own σ → 0 curve) fails. Define that test as a universal gate on f for any element whose all-electron
reference is affordable at the 40 mHa class (Z ≤ 2 today): the homonuclear dimer against the AE σ → 0 curve. It is
Z-derived (the dimer of the element), not per-element data.

### S7 — §8 Task 5: its egg-box test needs Task 6's SCF
`o_atom_eggbox_on_grid_after_filter` (rigid-shift E rms) requires V_NL, V_SR and ρ_pc in the 3-D Hamiltonian, which is
Task 6. Make Task 5's first failing test the property the filter actually claims and which needs no SCF: for each filtered
table g, the lattice sums h³ Σ_p g(r_p − R) and h³ Σ_p g(r_p − R) R_lm(r_p − R) are translation-invariant in R to
< 1e-8 relative over the 8 sub-grid shifts (Poisson summation: exact up to the spectral weight beyond 2π/h). Move the SCF
egg-box to Task 6/7.

### S8 — §6.4: reference choice and the `CoresOnly` angle
Johnson–Gill–Pople 1993 is SVWN/6-31G*, whose water angle runs ~1–1.5° below the basis-set limit; a grid solver is at the
basis-set limit, so pin a large-basis LSDA value (CCCBDB SVWN with an aug-cc-pVTZ/QZ-class basis) or report both. And the
angle under `CoresOnly` is not "the pseudopotential's own test" alone: the two smeared H's set the O–H length (0.34 Bohr
long today) and the H–H repulsion, both of which move the angle. State that the clean angle test is under `All`.

### S9 — §4.2 beyond Z = 18 (scope note, not for this milestone)
"Promote the whole n_max − 1 shell" promotes 3s, 3p and 3d together for Sc–Zn and Ga–Kr (3d within 2 Ha of 4s), giving a
3s/4s two-s-state refusal for the entire d block. The subshell-wise window (promote only the subshells inside Δ_core) is
equally universal and is what practice does (3d semicore only). Record it now so Task 12's successor does not inherit the
rule.

---

## NOTE

- **Confirmed correct (re-derived):** the GSS ghost criterion as stated (rank-1 interlacing: for E^{KB} > 0,
  ε₀^{loc} ≤ ε₀^{KB} ≤ ε₁^{loc}, so ε_l ∈ (ε₀^{loc}, ε₁^{loc}) forces ε_l = ε₀^{KB}; for E^{KB} < 0, ε₀^{KB} ≤ ε₀^{loc}, so
  ε_l < ε₀^{loc} is required); the sign of the spec's E^{KB} = 1/⟨φ|δV|φ⟩ equals the sign of GSS's ⟨φ|δV²|φ⟩/⟨φ|δV|φ⟩, so
  the test is unaffected — but report GSS's energy in the generation record, since that is what literature values mean.
  The three force terms: F^{loc} = +∫ρ_v∇_rV_loc, F^{NL} = +2Σf E^{KB} c_lm⟨∇_rβ|ψ⟩, F^{pc} = +½∫(v↑+v↓)∇_rρ_pc, all from
  F = −∂E/∂R with ∂/∂R_I = −∇_r, consistent with `forces.rs`; no Pulay term on a fixed grid; no double counting in §5.2's
  E (V_loc once, E_H once, E_NL as Σ f E^{KB} c²). The LR/SR split is exact and the Poisson boundary handling is today's
  (the LR Gaussians enter the multipole moments; V_SR is sampled, not solved). §4.2's illustrative eigenvalues agree with
  NIST LDA (Li 1s −1.88, Be 1s −3.86, Na 2p −1.04, Al 2p −2.56); Al's margin against Δ_core = 2.0 is only 0.27 Ha —
  universal, but close. §5.3's cost arithmetic (2.5e3 points, < 1 % per ion) checks. The spin-restricted generation with
  NLCC as the bridge to spin-polarised use is the standard, correctly reasoned choice.
- **T1 is an identity, not a transferability test:** the reference state is an exact eigenstate of the KB Hamiltonian with
  the same screening, so the KB atom reproduces the TM atom to numerical precision by construction. Label it a regression;
  T3–T6 are the transferability gates.
- **G2's radial KB atom** is an integro-differential eigenproblem: solve the homogeneous and the β-sourced particular
  equation, impose c = E^{KB}⟨β|u⟩ self-consistently — the eigenvalue condition is a 2×2 determinant. Say so in Task 3.
- **§5.1 σ_LR = max(2h, r_loc/3):** the r_loc/3 term is not derived (2h alone satisfies the Poisson floor); drop it or
  justify it — fewer constants in §4.9. The Poisson route's own numerical error on V_LR at σ_LR = 2h is ≈ 1.7e-5·(Z_v/2)
  ≈ 5e-5 Ha at the ion for Z_v = 6 (`nuclear.rs` floor), ~1e-5 Ha in energy; fine, but it is not cancelled by V_SR — state it.
- **§4.3 "extended beyond r_c as −Z_v/r":** exact only beyond max(r_c, r_nlc) and the core tail; compute V_ion out to
  where |V_ion + Z_v/r| < 1e-8 and verify, rather than "extend".
- **§4.4:** the literature TM radii correspond to f ≈ 1.2 (C: 1.5/1.23), 1.45 (N), 1.6 (O: 1.45/0.9) — a spread, not
  "1.4–1.5"; the ladder covers it, which is the point.
- **§1.1:** "3.2e-4 (Zh)⁸" is quoted from the brief without a source; label it as such.
- **§4.7 `All` for H, He:** no core ⇒ ρ_c ≡ 0, r_nlc undefined, ρ_pc = 0, N_c = 0; the partition/NLCC code path must take
  that branch explicitly (and the core-overlap radius of M3 is zero).
- **Policy switch at run time:** E_KS jumps by ≈ 38 Ha per O between `None`/`CoresOnly` and `All`; the viewer's
  E_conserved/drift readouts must reset on a policy change (app task), not only the surrogate provenance.
- **§6.3 O triplet–singlet "> 20 mHa":** the LSDA spin-polarisation energy of O is ≈ 50 mHa in the spherical atom; the
  integer-occupation 3-D "singlet" is a broken-symmetry state a few mHa lower than the spherical one, so > 20 mHa is
  achievable and honest. §6.7's 1000 K bound is generous (a 1 mHa preset offset releases ≈ 100 K).
- **Task 2's first failing test** "u^{PP} = u^{AE} beyond r_c to 1e-10" is true by definition; the discriminating first
  test is the seven conditions themselves (norm to 1e-10, five derivative continuities to 1e-8 relative, V″(0) = 0 with the
  M1 expression) plus unscreened V_ion → −Z_v/r beyond max(r_c, r_nlc).

---

**Verdict: ready with fixes.** Six MUSTs, all local: one equation (M1), one test reference (M2), two rules that would
block the design's own workflow or contradict its own numbers (M3, M4), one gate placed below a floor the spec itself
quotes (M5), and one mis-priced row in the problem statement (M6). Apply them in a spec revision before Task 1 starts;
the SHOULDs can land with the tasks they name (S1 → Task 6, S2/S3/S7 → Task 5, S4 → Task 4, S5 → Task 2/12,
S6 → Task 4/11, S8 → Task 10).
