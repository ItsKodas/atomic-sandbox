# Physics review — M7 implementation plan (self-generated NC pseudopotentials)

Reviewed: `docs/superpowers/plans/2026-10-15-m7-pseudopotentials.md` at ccd0fbd (branch `m7-pseudopotentials`),
2026-10-15, physics-reviewer (read-only). Against: the spec `2026-10-14-pseudopotential-design.md`, the earlier
review `2026-10-14-pseudopotential-spec-review.md`, CLAUDE.md rules 1, 3, 4, 6, 7, and `crates/qm/src/{xc,nuclear,forces,scf}.rs`.

Method: every formula the plan writes was re-derived (Numerov transform and source term, Coulomb regular start, Gauss-law
V_H, T_s = Σfε − ∫ρV_in, the scaling identity, TM p′…p⁗ targets, the polynomial local, LFC x·cot x = κ, the KB matching
determinant, GSS, the δE_equiv forms, the force signs, the solid-harmonic gradients, the S1 normalisation). Numbers marked
"measured" come from Python scratch replicas of the plan's own algorithms (not committed):
(a) the Task 1 log-grid Numerov + node-count bisection, line for line;
(b) a radial LSD SCF with PW92 (the repo's constants, `qm/src/xc.rs`) and with the plan's VWN5;
(c) 3-D radial Fourier transforms of the plan's switched V_tail and of a C²-kinked model potential under the plan's
filter and under a King-Smith band completion;
(d) a random-matrix check of the rank-1 inertia count.

**Verdict: ready after amendments.** The architecture, the ledger, the energy expression, the four force terms and the
rule-1 discipline are right. Six MUSTs, all local. Two of them sit on the critical path of Task 5's first test (Q2, Q3).
That test is itself blind to the dominant alias as written (M4). Task 1's first test cannot pass as written (M1).

---

## Spec questions — rulings

### Q1 — NIST LSD column is VWN: ACCEPT, with amendments
Measured (b): radial LSD at the plan's grid.

| atom | PW92 | VWN5 | NIST LSD |
|---|---|---|---|
| H | −0.4787098 | −0.4786698 | −0.478671 |
| He | −2.8344491 | −2.8348296 | −2.834836 |

VWN5 reproduces NIST to ~1e-6. That is the scratch code's own quadrature floor, so the column is VWN5 (the Ceperley–Alder
fit), not VWN-RPA. PW92 misses the 1e-5 gate by 4× for H and 38× for He. The like-for-like ruling is therefore necessary,
not cosmetic. The plan's VWN5 constants and interpolation (plan l. 455–475) were checked term by term against VWN 1980:
- A, x₀, b, c for the para, ferro and α_c fits;
- ε_c = ε_P + α_c f/f″(0)(1−ζ⁴) + (ε_F−ε_P) f ζ⁴;
- f″(0) = 8/(9(2^{4/3}−2)).

Amendments:
- (i) H (ζ = 1 everywhere) and He (ζ = 0) test the ferro and para fits alone. Only Li and C–Ne test the spin interpolation.
  Step 1 must also record NIST's interpolation (VWN's α_c form, as coded), else Li's 1e-5 can miss for a reason that is not
  the solver.
- (ii) The finite-difference v_c step (l. 483–486) is relative to n↑ or n↓ separately. For a minority spin with
  n↓/n↑ ≲ 1e-8 the difference loses ~1 % of v_c↓ to rounding. It is harmless for the variational total, but use
  d = 1e-6·(n↑+n↓) or the analytic derivative.
- (iii) VWN5 stays `#[cfg(test)]`, outside LEDGER and `recipe_hash`. That is correct as planned.

### Q2 — V_cmp = V_loc + Z_v/d is singular: claim CONFIRMED; plan's switch REJECTED; amended split below
The claim holds. V_loc is finite at 0, so V_cmp = V_loc + Z_v/d carries +Z_v/d. The spec's V_tail = −Z_v erfc(d/a)/d
carries −Z_v/d. Each is singular; only their sum V_SR is smooth. The earlier review's M4 introduced exactly this split and
has the same error: I wrote it. A second error sits under it. The spec (§5.1) calls the erfc tail "already band-limited by
σ_LR (aliasing 2e^{−8π²})". It is the opposite: FT[−Z_v erfc(d/a)/d] = −4πZ_v(1 − e^{−q²a²/4})/q², the complement of the
band-limited Gaussian part, decaying only as q⁻².

The plan's remedy (l. 2723, 5828) switches the tail on over [r_c/γ, r_c] with the ψ-step and samples it unfiltered.
That removes the singularity but leaves the switch's high-q content on the grid. Measured (c), Z_v = 6:

| h | r_c | V_tail at r_c/γ | FT(2π/h) / FT(0) | |FT(2π/h)| |
|---|---|---|---|---|
| 0.25 | 1.9 | 0.054 Ha | 1.4e-4 | 4.6e-6 Ha·Bohr³ |
| 0.25 | 1.4 | 0.40 Ha | 7.3e-4 | 2.6e-4 Ha·Bohr³ |
| 0.35 | 1.9 | 0.33 Ha | 5.7e-5 | 4.7e-5 Ha·Bohr³ |
| 0.35 | 1.4 | 1.17 Ha | 2.1e-4 | 6.4e-4 Ha·Bohr³ |

These figures are 10⁴–10⁵ above the 1e-8 standard Task 5 imposes on the filtered tables. No test covers them: the
lattice-sum test (l. 2768–2776) checks V_cmp, ρ_pc and β, not V_tail. With ρ ≈ 0.05 e/Bohr³ in the switch region, the
first-shell force ripple is ≈ 6·ρ·|FT(G)|·G. That is ~3e-5 at h = 0.25, r_c = 1.9, and ~2e-3 at r_c = 1.4: at or far above
the §6.6 gate of 3e-5.

**Ruling (amends spec §5.1, plan Task 5, the Task 8 `local_sr` kernel and the Task 7 force):**
- V_LR(d) = −Z_v erf(d/a)/d, a = σ_LR√2, analytic per centre on the grid (Q6).
- **V_SR(d) = V_loc(d) + Z_v erf(d/a)/d as ONE table**, with no V_cmp/V_tail sub-split.
  - It is smooth and finite at 0: V_loc is finite and even, and erf(d/a)/d → 2/(√π a).
  - It equals −Z_v erfc(d/a)/d (to `v_ion_tail`) beyond r_cmp.
- Its support radius is R_SR = max(r_cmp, r_floor), where r_floor solves Z_v erfc(r/a)/r = `v_tail_floor`. That is
  ≈ 2.5 Bohr at h = 0.25 and ≈ 3.5 at h = 0.35. It is derived from two existing ledger fields; no new constant.
- The table is filtered (Q3) and truncated at γ_r·R_SR. The truncation now acts where |V_SR| ≤ 1e-6 Ha, so M4's
  objection (masking discards the exact split) disappears. V_SR never has to be split at all.
- Interaction with the filter: the grid sees V_loc,<q_c plus V_LR,>q_c. The latter carries a relative weight
  e^{−(q_c a/2)²} = e^{−(0.7π√2)²} ≈ 6e-5 at q_c. It is included in the T8/S2 re-run automatically, because
  `FilteredTable::input()` already sums v_lr + v_sr.
- Interaction with the egg-box gate: every sampled ion-centred function (V_SR, β, ρ_pc) is band-limited up to the filter's
  leakage. V_LR's alias at 2π/h is ∝ e^{−(πa/h)²} = e^{−8π²}. The lattice-sum test must include V_SR.
- Cost: V_SR is sampled once per geometry. A 3.8 Bohr sphere (h = 0.25) is ~1.5e4 points per ion: negligible.
- Force: F^SR = ∫ρ (dV_SR/dd) d̂ from the same table's `interp_deriv`. `dv_tail`/`tail_in`/`tail_out` disappear, and so do
  the tail branch in `local_sr` (l. 4343–4347) and the `radii` fields.

Also update the §4.9 justification of σ_LR = 2h. With Q6 the Poisson-floor argument is moot. The reason is now spectral:
V_LR's content above q_c is e^{−(q_c a/2)²} ≈ 6e-5 at a = 2√2h. That is what makes σ_LR scale with h, and it also sets
r_floor.

### Q3 — "King-Smith's smooth cutoff" = hard cut + ψ-mask: AMEND (implement King-Smith's construction)
Measured (c) on a model with TM's smoothness class, f = −3(1−(r/r_c)²)³ (C² at r_c, f‴ jumps). It was cut at
q_c = 0.7π/h, inverse-transformed, and masked by the plan's ψ-step over [r_c, 1.5r_c]:

| h | r_c | |F(2π/h)|/|F(0)| | |F(√2·2π/h)|/|F(0)| | Gibbs ringing at r_c |
|---|---|---|---|---|
| 0.25 | 1.4 | 1.6e-6 | 5.6e-7 | 6.9e-3 |
| 0.25 | 1.9 | 9.7e-8 | 2.5e-9 | 3.1e-3 |
| 0.35 | 1.4 | 2.2e-5 | 3.6e-6 | 1.3e-2 |
| 0.35 | 1.9 | 1.3e-6 | 6.6e-7 | 6.6e-3 |

That is 10–2000× above the 1e-8 judge. The mask cuts through the hard cut's Gibbs ringing and re-creates high-q content,
so Task 5 would stop at its first test.

The construction King-Smith, Payne & Lin actually use is a variational band completion:
- keep f̂(q) for q ≤ q_c;
- choose f̂ on (q_c, q_max], with q_max = π/h (the grid Nyquist, derived), to minimise ∫_{γ_r R}^∞ |f|² r² dr;
- truncate at γ_r R.

A crude prototype (24 Chebyshev terms, least squares) reached:
- 1.5e-8 and 1.1e-8 at h = 0.25 (r_c = 1.4, 1.9), i.e. 10–100× better than the plan's construction;
- 2e-6 and 2e-7 at h = 0.35.

So the 1e-8 judge is reachable at h = 0.25, the only h the test uses, and only this way. It adds no new constant: q_c and
γ_r come from the ledger, q_max from the grid.

Ruling:
- Implement KSPL as above for V_SR, β and ρ_pc. Keep the ψ-step only as the outer truncation, where the optimised tail is
  already small.
- Keep the 1e-8 judge with M4's shifts.
- If KSPL still misses, the stated remedies stand (Ono–Hirose, or report the factor). They are never a looser bound.
- At h = 0.35 expect T8 or the egg-box to bind. That is an outcome h_req reports, not a defect.

### Q4 — `deliberate_bad_local_is_caught_as_ghost` impossible as worded: claim CONFIRMED; ruling ACCEPTED with test amendments
Re-derived:
- With H_sl = H_loc + δV and H_sl φ = ε_s φ (‖φ‖ = 1): ⟨φ|H_loc|φ⟩ = ε_s − ⟨φ|δV|φ⟩, and the variational principle
  gives ε₀^loc ≤ that.
- So ε₀^loc > ε_s ⇒ ⟨φ|δV|φ⟩ < 0 ⇒ E_KB < 0, and with ε_s < ε₀^loc GSS (PRB 44, 8503 (1991), case E^KB < 0) is ghost-free.
- Conversely E_KB > 0 forces ε₀^loc < ε_s, so GSS's E^KB > 0 condition reduces to ε_s < ε₁^loc.

The two real ghost cases are therefore (A) E_KB > 0 with ε₁^loc < ε_s, and (B) E_KB < 0 with ε₀^loc < ε_s, as the plan
rules. The spec's §8 row-3 wording is a spec erratum; fold the ruling into the spec.

Test amendments (plan l. 1684–1707):
- (a) Case B's existence is not guaranteed at w = 0.15 Bohr. Binding a deep level needs D·w² ≳ O(1–10), while the narrow
  well's contribution to ⟨φ|δV|φ⟩ is D⟨narrow⟩ ∝ D·w³·u′(0)² ∝ w. With w = 0.15 the window D_bind < D < D_flip (where E_KB
  turns positive) can be empty. Use w ≤ 0.05 Bohr, or scan D geometrically to 1e4, and assert the window was found.
- (b) G2's direct check is skipped when the KB SCF fails (`if let Ok(atom)`, l. 1699). A ghost can make that SCF fail, so
  the "independent proof" is vacuous exactly when it matters. Run `kb_eigenvalues` on the fixed reference screening
  (v_loc + `t.screen`), unconditionally, for both cases A and B.

### Q5 — G2 via zeros of the matching determinant + ledger `g2_mesh`: AMEND (use the exact inertia count)
The determinant is right. Re-derived: value/slope matching plus c = E_KB⟨β|u⟩ give M = [[W(u_h,u_i), W(u_p,u_i)],
[E_KB⟨β|u_h⟩, E_KB⟨β|u_p⟩ − 1]]. Moreover det M = E_KB·W(u_h,u_i)·s(E), with s(E) = −1/E_KB − ⟨β|G_loc(E)|β⟩.

By Haynsworth inertia additivity, the number of KB eigenvalues below E is exactly
  **n_KB(E) = n_loc(E) + [s(E) < 0] − [E_KB > 0]**.
Checked: 4000/4000 random rank-1 cases, n = 12. n_loc is node counting of the local channel, which the code already has,
and s = det M/(E_KB·W(u_h,u_i)) comes from the same `kb_solve`.

Ruling:
- G2's "no extra state below ε_v,max + 0.5" is one evaluation of this count at the window top. It is exact and cannot miss
  two roots inside one mesh cell.
- Keep the mesh, and `g2_mesh` in `numerics` (hashed; a numerical knob is not a rule-1 quantity), only for locating roots
  for the 1 mHa eigenvalue comparison.
- Assert that the mesh root count equals the inertia count.

### Q6 — V_LR is the analytic erf: ACCEPT
Verified: `nuclear_potential` (`qm/src/nuclear.rs:115–125`) and `nuclear.wgsl` evaluate −Z erf(d/σ√2)/d analytically.
Spec §5.1/§5.2's "Poisson solve of a deposited Gaussian" is stale. Consequences to write into the spec:
- the "Poisson route's own ~1e-5 Ha floor" is gone (the erf is exact pointwise);
- σ_LR = 2h is re-justified spectrally (see Q2).

### Q7 — c_lm finished on the GPU inside H; E_NL and forces in f64 on the CPU: ACCEPT
c_lm inside H is an operator coefficient, f32 like the stencil. The `proj_dot` tree plus the ≤ 64-term `proj_finish` is
pairwise (rule 4). E_NL and the forces are reductions of energies and forces, finished in f64 from one readback (rule 4
satisfied). `gpu_vs_cpu` is the right judge, with Kahan as the measured remedy.

### Q8 — `All` usable from Task 6; Task 11 re-runs Task 10: ACCEPT, conditional on M6
The no-core branch is generic, so this is sound. But the ungated and gated H tables share one `recipe_hash` (M6). Without
M6's fix:
- Task 10's numbers, any cached H table and any surrogate model trained under `All` between Tasks 9 and 11 stay valid by
  hash after H's potential changes;
- `h_for` (l. 4935–4937) keeps reading the ungated H's h_req even after Task 11.

### Q9 — dimer gate as a provider callback, `dimer_bracket`, He not applicable: ACCEPT with notes
The callback is the only clean way to put a 3-D computation inside the radial ladder. Extrapolating the AE force at each R
over σ = h·{1, 1.2, 1.4, 1.6} is consistent: d/dR of the energy ansatz has the same σ dependence. Zσ_max = 0.24 for H at
h = 0.15 is inside the asymptotic regime.

Notes:
- (i) Justify [1, 4]·r_peak physically in the ledger doc: covalent R_e ≈ 1.5–3 r_peak (H₂: 1.45), while LSDA's spurious
  rare-gas minima lie beyond 4 r_peak. Then it is a universal rule, not a number chosen to make He "not applicable".
- (ii) For "not applicable", record the PP and AE bond forces at both bracket ends in the generation record (report only).
  He–He collisions in the sandbox live on that repulsive wall.
- (iii) k from F(R_e ± h) with h = 0.15 carries an anharmonic bias. Morse: k_FD = k(1 + 7a²h²/6), so ω is ~+1.4 % for H₂.
  It cancels inside the gate (same procedure on both sides) but not in `pseudo_h2_omega_within_5pct_of_sigma0_curve`
  (l. 5184–5198) against the curve-fitted 4405 cm⁻¹. Either compute ω there by the reference's procedure, or state the
  bias in the test doc.

### Q10 — bond-strength kind: AMEND (it exists now)
`sim::BondKind { Shared, ClosedShell, Indeterminate }` with `PairReport::kind() -> Option<BondKind>` is on `m6-tools`
(`crates/sim/src/bonds.rs:349`). `angle_style` (l. 5688) should take the kinds of both bonds:
- any `ClosedShell` → the contact's faint/dashed style;
- `Indeterminate` → its own neutral style, never drawn as Shared (rule 7: the sign is not resolved);
- `None` (Laplacian not computed) → today's style;
- multi-centre stays dashed.

Physics caveat to state in the tooltip and Selection tab: under pseudopotentials ∇²ρ_b is evaluated on the valence (±ρ_pc)
density. For O–H the bond point lies ≈ 1.4 Bohr from O, inside pseudo-O's r_c (1.4–1.9). There the pseudo-density is
smooth by construction, so the QTAIM sign is not the all-electron one. Label kinds on bonds whose BCP is inside an r_c
sphere as "valence-density kind".

### Q11 — h_req = coarsest h such that it and every finer ladder h pass: ACCEPT
This is conservative and well defined. It already combines correctly with `ionic::check` (l. 3427–3438), which also
requires the filtered re-run to pass at the scene's own h. Off-ladder h (Precise 0.10, Custom) are therefore certified
directly, and non-monotone ladders cannot admit an uncertified h.

### Q12 — pole-free δE_equiv: defect CONFIRMED; plan's form REJECTED; use the Wronskian form
The defect holds: (D^PP − D^AE)·u_AE²/(2N_AE) has a pole wherever u_PP(r_d) = 0 and u_AE(r_d) ≠ 0.

The plan's replacement Δ[uu′/(2N)] = ½(D_PP w_PP − D_AE w_AE), w = u(r_d)²/N, is **not** first-order equal to the spec's
quantity:
- The Wronskian identity gives D′w = −2 for each potential at every E. Hence d/dE[Dw/2] = −1 + Dw′/2, and at ε_l the
  plan's form has slope ½D(w′_PP − w′_AE), where w′ = 2D″/D′². Norm conservation matches D and D′ at ε_l, not D″, so the
  plan's form has a term linear in (E − ε_l).
- The spec's form is second order there, since it vanishes with ∂D/∂E. The extra term is −2D/(D′(E−ε)) times the spec's
  quantity: ≈ 1.5× at ±0.3 Ha and ≈ 0.5× at the ±1 Ha edges for O 2p (D ≈ −1.1, D′ ≈ −4.8).
- So the gate would measure a different quantity, the mismatch of ∂D/∂E times D, not the energy-equivalent of a
  log-derivative error.

**Ruling:** δE_equiv(E) = ½·W(û_AE, û_PP)(r_d) = ½[û_AE û′_PP − û′_AE û_PP](r_d), each regular solution normalised to
∫₀^{r_d} û² dr = 1.
- It is pole-free (finite u, N > 0).
- It equals ½(D_PP − D_AE)·√(w_AE w_PP), the spec's quantity times 1 + O(δw), so it agrees to third order at ε_l.
- It has the perturbative meaning ⟨û_AE|V_PP − V_AE|û_PP⟩ on [0, r_d]. The KB terms cancel beyond β's support,
  re-derived.
- The 20 mHa threshold is unchanged.
- In `kb_regular` (l. 1947), also drop the division: use u = (1 − E_KB⟨β|u_p⟩)u_h + E_KB⟨β|u_h⟩u_p.

---

## MUST

**M1 — Task 1 `hydrogenic_eigenvalues_are_exact` (l. 402–420; spec §4.1 l. 92): 1e-10 absolute is unattainable at
δ = 0.008.** Measured (a), the plan's Numerov and bisection replicated line for line:
- clean δ⁴ convergence (the error falls 16.1× from δ = 0.016 to 0.008);
- relative error ≈ (0.008, 0.047, 0.17)·δ⁴ for 1s, 2s, 3s;
- the log grid is scale-free, so the absolute error grows as Z².

Of the 19 listed (Z, n, l), 7 fail:

| Z | n | l | Δε |
|---|---|---|---|
| 2 | 3 | 0 | 1.6e-10 |
| 3 | 1 | 0 | 1.5e-10 |
| 3 | 3 | 1 | 2.0e-10 |
| 8 | 1 | 0 | 1.1e-9 |
| 8 | 2 | 0 | 1.5e-9 |
| 8 | 2 | 1 | 5.1e-10 |
| 8 | 3 | 2 | 3.4e-10 |

(Z = 8, 4f: 2.6e-10 also fails.) Halving δ only reaches 9.7e-11 for Z = 8 2s (margin 1.03).

Spec revision (reviewer, with this derivation): **|ε − (−Z²/2n²)| < 1e-10·Z²**. The worst margin is then 2.5× (Z = 2 3s).
Add an order check: the Z = 1 1s error ratio between δ = 0.016 and 0.008 must lie in [14, 18]. That catches a wrong
Numerov weight that an absolute bound at one δ would let through. δ itself stays.

**M2 — Q2: the spec's V_cmp/V_tail split is singular, and the plan's switched tail aliases unfiltered.** See Q2: one
V_SR table, KSPL-filtered, truncated beyond r_floor. Applies to Task 5 (l. 2721–2726, 2942–2949, 2986–2994), Task 7's
F^SR (l. 3808), and Task 8's `local_sr` (l. 4331–4352).

**M3 — Q3: hard cut + ψ-step cannot pass the 1e-8 lattice judge.** See Q3. Implement KSPL's band completion
(`filter_radial`, l. 2901–2913).

**M4 — Task 5's first test (l. 2744–2779) cannot see the first reciprocal shell.** For any inversion-even g, Poisson
summation gives S(R) = h³Σ_p g(r_p − R) = Σ_G ĝ(G)cos(G·R). The first shell contributes
2ĝ(2π/h)Σ_a cos(2πR_a/h), which is exactly 0 at every R_a = ±h/4. The 8 shifts therefore measure only the √2 and √3
shells, which are 3–10³× smaller in Q3's table, while the dominant alias passes unseen.

Use shifts R ∈ {0, h/2}³, where cos = ±1 (or both sets). The same blindness affects every *energy* rms reported over ±h/4
shifts of an inversion-symmetric system: `o_atom_eggbox_on_grid_after_filter` (l. 3365–3381) and the E-rms lines of
Tasks 7, 10 and 11. Report the energy ripple from {0, h/2} shifts. The *force* probes at ±h/4 are correct (sin = ±1 there:
maximal), so the §6.6 gate itself stands.

**M5 — Q12: replace the plan's δE_equiv by the Wronskian form** (module doc l. 2435; `delta_e_equiv` l. 2504–2517).
See Q12.

**M6 — Task 11: two different H potentials under one `recipe_hash`.**
- `generate_with(z, gate)` (l. 5303–5332) changes H's f, so the table, h_req and E_KB change. `generate(z)` (no gate)
  still returns the ungated H.
- The ledger already contains the dimer fields (l. 324), so the hash does not move.
- Used in different places: `pseudo_gen` pins the ungated H ("dimer: not run", l. 5270), Task 10's `h_for` reads it, while
  `PseudoCache` and surrogate provenance use the gated one. `IonicSetup::same_physics` compares only the hash, so a warm
  start can cross them.

Fix:
- (a) For Z ≤ `dimer_z_max`, `generate` without a gate returns a table marked provisional that `IonicSetup::build`
  refuses, and `pseudo_gen` runs the gate (GPU), or its H/He rows read "provisional".
- (b) Bump `LEDGER.version` in Task 11.
- (c) Global rule (Global Constraints, near l. 51): any change that moves a pinned generation record bumps
  `LEDGER.version`. `generation_record_table_matches_references_md` enforces it: records changed and hash unchanged ⇒ fail
  with "bump LEDGER.version".

## SHOULD

**S1 — Task 2's first test is partly circular (l. 1249–1254).** Conditions (2)–(6) compare `poly_derivative` with
`tm_targets`, the very right-hand side the 5×5 solve consumed. A sign error in the p‴ or p⁗ formula (l. 1476–1477) would
pass. The formulas are correct (re-derived from u″/u = l(l+1)/r² + 2(V−ε)). Test them independently:
- V_scr from the polynomial at r_c⁻ equals V_AE(r_c) to 1e-8;
- its first and second derivatives (analytic, from c) equal those of V_AE (grid derivatives) to 1e-6 relative;
- u^PP′(r_c⁻) equals u^AE′(r_c).
These are the physical content of TM's C⁴ matching.

**S2 — Task 7's first test cannot detect a missing force term below 2e-3 (l. 3867).** `forces.rs:46–64` derives the
HF−FD floor at ≈ 2e-5 (FD truncation), with the 2e-3 bound "~100× above" it. On a fixed grid HF is the exact derivative of
the discrete energy, so the identity is tight. Keep the spec's 2e-3, and add per-term frozen-state FD checks, exact to
~1e-8 and independent of the SCF:
- d/dR of h³Σρ V_SR(r_p − R) at fixed ρ vs F^SR;
- the same for V_LR vs F^LR;
- E_NL(R) = Σ E_KB c² at fixed orbitals vs F^NL;
- E_xc[ρ + ρ_pc(R)] at fixed ρ vs F^pc;
- Σ q_I q_J/R_IJ vs ion–ion.
Also assert |HF − Richardson| < 1e-4 (5× the derived floor) as a reported-then-gated spec revision.

**S3 — Q5's inertia count** (exact; see Q5) for G2 (`ghost_gates`, l. 2066–2076).

**S4 — Q4 test robustness** (narrow width, unconditional G2 on the reference screening; see Q4).

**S5 — Task 10 `ne_h_convergence_is_monotone` (l. 5040) assumes a direction the spec does not give.** The spec says
"monotone". The 8th-order FD symbol underestimates q², pulling coarse-h energies *down*, while the per-h filter smooths
V_SR, pushing them *up*. The sign is not derivable a priori. Assert that the successive differences have one sign, and
report the sign.

**S6 — Task 6/7 `ion_on_a_grid_point_is_finite` force < 1e-8 (l. 3916).** Inversion symmetry makes the force vanish only
for an exactly parity-pure state. The residual parity mixing of iterative eigenvectors is first order in the eigensolver
residual (default `for_grid` tolerances), so |F| ~ residual·|∇V| can be ≫ 1e-8. Run at tol = eig_tol = 1e-9 and bound by
1e-6, or derive the bound from the tolerances.

**S7 — Q10 BondKind styling and the pseudo-density caveat** (see Q10).

**S8 — Task 6 S1 identity tolerance (l. 3216).** "< 1e-6 absolute" pointwise mixes the grid quadrature floor of
⟨β|φ⟩_grid·E_KB into an operator-normalisation test.
- Estimate: the unfiltered δV has C² kinks at r_c, so its alias at |G| = 2π/0.1 is ~1e-7 relative, while max|δVφ| is
  several Ha for O s.
- Make the bound relative: |V_NLφ − δVφ| ≤ 1e-6·max|δVφ|. Assert separately that |E_KB⟨β|φ⟩_grid − 1| < 1e-4.
- The factor the test exists to catch is 4π/(2l+1) ≥ 4.19, so discrimination is unchanged.

## NOTE

- **Task 1, re-derived and correct:**
  - y-transform g = (l+½)² + 2r²(V−ε) and the sourced Numerov S = 2r^{3/2}s (l. 642–647);
  - regular start r^{l+1}(1 − Zr/(l+1)): its O((Zr₀)²) start error leaves an irregular admixture ~(Zr₀)³ ≈ 5e-13 at
    r ~ 1/Z;
  - V_H by Gauss's law, T_s = Σfε − ∫ρV_in, the scaling-identity residual (l. 855–859; it holds for fractional fixed
    occupations);
  - DIIS on ∫ab r² dr.
  - The hydrogenic bisection to 1e-13|ε| and the WKB end (∫√g > 40) put the wall error at ≲ e⁻⁸⁰, below the δ⁴ term.
- **Task 2:**
  - partition: Li's 1s (−1.88) sits 0.22 Ha inside Δ_core of 2s − 2 Ha, so it is promoted and refused as specified; Al's
    2p margin is 0.27 Ha.
  - V_ion → −Z_v/r beyond max r_c to 1e-8 does *not* hold for O at f = 1.6. This solver's O 2s r_peak is 0.872, so
    r_c,s = 1.398 (not ≈ 1.9), and the 1s Hartree tail there is −4.5e-8 (Task 2, measured). The tail is physics; the
    test reads "beyond r_cmp" (Task 2 review, ruling 1).
- **Task 4:**
  - LFC A = ρ_c R/sin(BR) and x·cot x = 1 + Rρ_c′/ρ_c are correct, with one root in (0, π).
  - r_nlc uses the AE valence density, which is f-independent. That is a sensible reading of "ρ_c = ρ_v"; state it in the
    spec.
  - The 5 % crossover tolerance (l. 2235) is marginal: |d ln(ρ_c/ρ_v)/d ln r| ≈ 5 near 0.45 Bohr gives ≈ 4 % per log-grid
    step.
- **Energy (Task 6, l. 3752–3760; `scf.rs:386–393`):** T_s is ⟨ψ|−½∇²|ψ⟩, not Σfε − ∫ρV, so adding E_NL = ΣE_KB c²
  separately is correct. ρ_pc enters only E_xc (and v_xc), never V_H, `external` or N. E_ii uses Z_v. No double counting.
  The KB-atom energy (l. 2018–2028) is consistent with the unscreening.
- **Forces (Task 7, l. 3806–3811, 3963–4038):** all signs re-derived from ∂/∂R = −∇_r.
  - F^loc = +∫ρ V′ d̂;
  - F^NL = +2ΣE_KB c⟨∇β|ψ⟩ with ∇β = t′d̂NR_lm + tN∇R_lm (the solid-harmonic gradients l. 3932–3940 check);
  - F^pc = +∫½(v↑+v↓)ρ_pc′ d̂;
  - ion–ion as stated.
  - Complete with M2's single V_SR. No Pulay term. The interpolant derivative is the exact derivative of the sampled
    energy; its C⁰ kinks at log nodes are ~1e-7 relative.
- **KB normalisation (l. 1630, 3615–3622):**
  β_lm = (δV u/r^{l+1})·√((2l+1)/4π)·R_lm gives ⟨β_lm|φ_lm⟩₃D = ∫u δV u dr = 1/E_KB. Correct.
- **Rule 1:** no Z dispatch outside the ledger-gated `z <= LEDGER.dimer_z_max`. Per-element literals appear only as test
  oracles (NIST, CCCBDB, the H₂ σ → 0 pins) and display symbols.
  - `gate_thresholds_are_constants` catches `const` declarations, not inline literals. The inline numerics I found are
    universal solver knobs, not rule-1 data: 1e-3·umax for r_peak, the ±0.1 Ha search window, WKB 40, 1e-12 support.
  - The r_peak threshold feeds r_c. Note in the code that it cannot change the outermost valence maximum.
- **Spec numbers carried verbatim:** the ledger (l. 301–327) and every gate match spec §4.9/§4.8/§6. Plan-introduced
  numbers are knobs (DIIS 6, mixing 0.3, 400 iterations, `g2_mesh`, `dimer_bracket`). All are hashed, and Q5 and Q9 give
  their status.
- **Task 8 (Opus):** `interp` forms s = (ln d − ln r₀)/δ in f32 with ln r₀ ≈ −11.5 stored in f32. The ≈ 6e-5 index offset
  shifts d by ≈ 5e-7 relative, about the same size as the 2e-6 relative bound of `local_sr_matches_cpu` (l. 4139). Use a
  split ln r₀ or an f32-exact index base if it trips.
- **Task 10:** Gaussian's "SVWN" is VWN-RPA (VWN3), and CCCBDB lists SVWN and SVWN5. Record which one was pinned. The
  geometry difference (~1e-3 Å, ~0.1°) is far inside ±0.02 Å/±2°.
- **Task 12:** Na/Mg fall back to all-electron under `CoresOnly` at Zσ = 1.65 (h = 0.15), the regime spec §1 shows is
  qualitatively wrong (like O today). The tooltip/note should say so (rule 7).
- **Task 13 (l. 5756):** the tooltip asserts "≈ 105°" before Task 10 measures it. Quote the measured value from
  references.md.
- **Q11/§5.5:** under Balanced (0.25) and Fast (0.30) a pseudised species with h_req < h refuses the scene. That is
  correct behaviour, and the README should say so.

---

**Overall verdict: ready after amendments.** Apply M1–M6 and the Q-rulings above before the tasks they touch:
- M1 → Task 1;
- M2, M3 and M4 → before Task 5 starts (they settle Spec questions 2–3, as the plan's gate table requires);
- M5 → Task 4;
- M6 → Task 9/11;
- the SHOULDs with their tasks.

Fold the Q2, Q4, Q6, Q12 rulings and M1's bound into a spec revision, the reviewer's derivations being the ones above.
