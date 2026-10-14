# Orbital viewer — physics review of spec + plan (6e94dee)

Reviewer: physics-reviewer (via the coordinator), 2026-10-15. Controller's rulings on it: apply all MUSTs and SHOULDs as stated; adopt the NOTEs (drop the pub(crate) change; max|ψ| needs two reductions or a new one; the O-triplet test un-ignored only under a policy that pseudises O; derive the smeared-1s cut-off (e.g. Zσ ≳ 0.5) instead of Z ≥ 3; M7 physical scaling includes the core in max ρ; VRAM K + 2m; the "virtual" tooltip says the KS gap is not the fundamental gap). Add a Task 0 spike (Opus, measure only): the real LSDA 2p (and Ne) splitting on the GPU at h = 0.25 and 0.15 vs atom position (grid point, cube centre, the plan's off-grid point, (h/2,0,0)) and the display-eigensolve cost at 81³/97³; Tasks 1–5 consume its numbers. How each item was applied: spec §14.

---

Verdict: needs rework, only in one place — §5.1/§5.3 lone-atom degenerate-set identification (P2) rests on a τ the grid exceeds by ~100×; the plan's Task 5 STOP would fire after Tasks 1–4.

MUST 1 (§5.1 P2 / §5.3, Task 5 `ne_2p_aligns_to_the_cell_axes`): TAU_MIN = 1e-4 Ha cannot absorb the grid's 2p splitting. Not FD anisotropy: Σ_a s(k_a h) is O_h-invariant and p spans an irreducible 3-dim rep, so any cube-symmetric operator leaves p exactly degenerate (it splits d → e_g + t_2g, and f). The splitting comes from the position-dependent sampling of the σ = h smeared nucleus (egg-box). Numeric check (one-electron model, the code's COEFF stencil, V_nuc = −Z_eff·erf(r/(√2σ))/r with screening, σ = h = 0.25, 41³, LOBPCG): on a cube-symmetric site 2p spread ≤ 5e-8; at the plan's (0.37h, −0.21h, 0.12h) spread 2.5e-3 Ha (Z = 8), 1.3e-2 (Z = 10); at (h/2,0,0) 3.7e-3 / 2.0e-2; 1s moves 0.26 / 0.50 Ha between sites. A moving lone Z ≳ 6 atom has p split above τ → the channel falls to `#k` and labels flicker. Fix: for a lone atom group by angular character (2l+1 consecutive orbitals with individual W_l ≥ L_CHAR_MIN), not by energy gap; show the measured spread in the tag ("split by the grid by X mHa"); no per-Z τ (rule 1); detect open shells from occupation (SHOULD 7); move the Ne/O splitting measurement into a Task 0 spike before Task 1.

MUST 2 (Task 1 `procrustes_recovers_…`, Review Focus 2): with φ_j = Σ_a R_ja p_a, b_{m,j} = c·R_jm, M = c·Rᵀ, SVD W = Rᵀ, V = I → U = V·Wᵀ = R; assert RᵀU = I to 1e-8, not UR = I / U = R⁻¹ (seed 0x5EED, max|R − Rᵀ| = 0.50: |U − R| = 2e-16, |RᵀU − I| = 4e-16, |UR − I| = 0.50 — a correct implementation fails the pinned test).

MUST 3 (§5.3 item 4 with §4.5): "Show unoccupied: m" cuts through degenerate box-state sets (cube box groups 1, 3, 6, e.g. (3,2,1) six-fold); a truncated set or one without l (d = 6) wipes the occupied 1s/2s/2p labels as a slider moves. Rule: labelling covers occupied (bound) clusters only; a cluster cut by the block edge is excluded; box states never get atomic names; optionally a few LOBPCG guard vectors and discard the incomplete top cluster.

MUST 4 (§4.5 / Task 4): solve to the effective tolerance max(eig_tol, 3 × the f32 residual floor ≈ 1.8e-6 at h = 0.25, qm-gpu/src/scf.rs:625), not the raw ScfSettings::eig_tol = 1e-6 (else every display solve ends on stall/cap, every row "not converged").

SHOULD 5 (§5.2): W_l with c_lm = ⟨ψ| |ψ| Y_lm⟩ is nonlinear in ψ, not invariant under mixing within a degenerate set (200 random O(2l+1) mixings: p invariant W_1 = 0.939; d 0.984–0.998; f 0.974–0.992; p's 0.939 leaves 0.04 margin over L_CHAR_MIN 0.9). Use the linear c_lm = ⟨ψ|Y_lm(r̂)⟩ — the cluster sum is a Frobenius norm, invariant, exactly 1 for an ideal shell.

SHOULD 6 (§5.1): single linkage allows spread up to (d−1)τ; the residual of a combination is ≤ spread/2 (variance bound) — restate "spread ≤ τ" and "exact".

SHOULD 7 (§5.3 item 4): open-shell detection by occupation, per atom: if any occupied l-set in either channel is incomplete the potential is non-central for both channels → both `#k` (e.g. C's ↓ "2s" in an anisotropic ↑p² field).

SHOULD 8 (§5.5 / Task 5 `h2_singlet_…`): the claim "closed-shell noise stays under the 0.1 iso" needs max|m| < 0.1·SPIN_FLOOR·max ρ = 1e-4·max ρ; the test asserts < 1e-3 — assert 1e-4 or measure first.

SHOULD 9 (§5.7): the face-weight rule is box-size dependent (Dirichlet box state within 2h of faces ≈ 6·(2/L)(π/L)²(2h)³/3: 2.9e-3 at 49³, 1.8e-3 at 57³, 3.6e-4 at 97³ < 1e-3); use a box-scaled criterion, e.g. weight in the outer L/6.

SHOULD 10 (§5.4 item 3): sign functionals normalised by whole-box norms (‖x‖ ∝ L^{5/2}, quadrupoles L^{7/2}) make the 1e-3 ratio box-dependent and quadrupoles rarely qualify; window them, e.g. (x − X̄)·e^{−|r−X̄|²}; "deterministic" is false when none qualifies (the raw solver sign is kept) — say so.

SHOULD 11 (§4.4/§4.5 cost): a display LOBPCG of n + m vectors partly cold every 1 s could take 30–100 % of the worker at 97³; budget it as a fraction of the measured SCF wall time or run only when paused; measure in Task 4.

SHOULD 12 (§7 surrogate): teacher and auditor each running an OrbitalView would disagree on tracks/slots/signs and alternate payloads; one CPU-side tracker, or only one producer.

SHOULD 13 (Task 3 bitwise test): force an update every step and assert 20 payloads (the 100 ms rate limit can make it vacuous); Task 4 extend to m = 4 over 20 steps; Task 5's H₂ test add m = 2 so σ_u (Review Focus 3) is tested on the real solver.

SHOULD 14 (§8 / P5): core rows from the spin-restricted reference atom (pseudopotential spec §4.3 generates from it; NLCC from it; the spin view needs no approximation); Task 8's rel ≤ 1e-5 breaks near radial nodes of 2s cores (Al–Ar) — measure relative to max|u/r| of the shell; valence labels depend on the ghost gate (§4.6) — Sturm–Liouville ordering does not strictly hold for a nonlocal potential.

NOTE: verified safe — sharing LobpcgWork (no carried state: np = 0, HX recomputed at entry); the display potential needs no Poisson solve (GpuState holds v_h = V_H[ρ_out] and ρ_out,σ, scf.rs:831–857); display fields from their own pool. Plan line 58: `solid_harmonics` is already pub (L_MAX = 6, Racah) and used by qm-gpu — drop the pub(crate) change. The Reducer has Max but no max|·| — max|ψ| needs two reductions or a new one. Both Procrustes variants derived correctly; neither uses a vanishing dipole element. O triplet: consistent with M6 references.md:1167; un-ignored only under a policy that pseudises O. "Smeared 1s (Z ≥ 3)": derive (Zσ ≳ 0.5). M7 physical scaling includes the core in max ρ. VRAM K + 2m. Honesty text adequate; add "the KS gap is not the fundamental gap" to the virtual row. "One readback" reads only scalars. Not checked: a real Ne/O LSDA GPU SCF (the splittings are a screened one-electron model, order of magnitude only — Task 0 measures them); WGSL rendering maths beyond −sign(s)∇s.
