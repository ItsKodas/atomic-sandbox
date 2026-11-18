# Stray atoms: the model answers a molecule plus far-away atoms (measured composition)

Status: design, 2026-11-18. User request and choice of approach (B, "measured") in the session of 2026-11-18.
Branch `stray-atoms`, based on the smearing branch (`scf-stall`, 92000979). Atomic units (Hartree, Bohr).

## 1. Intent

The user's scene: H₂ with a third H that has drifted more than r_cut (6 Bohr) from it. Every step goes to DFT
(rule 5a, "isolated atom"), so the run is slow for as long as the H stays away, even when the model knows H₂.
The user's words: "ideally that way it won't fall back to DFT and run smoothly until they interact".

**Goal:** in strict mode, a scene made of a bonded part plus one or more *stray* atoms (no neighbour within r_cut)
is answered by the model, once it has been **measured** that this combination's DFT energy splits as the bonded
part's energy plus the free atoms' references. When a stray atom comes back within r_cut, the ordinary rules take
over. Nothing is hardcoded per element (rule 1): the clearance is a measurement keyed by Z.

**Non-goals.** Making an untrusted bonded part answerable: the bonded part still has to pass every rule on its
own. Spin-aware models. Fragments that are each bonded but far from each other (two H₂ 8 Bohr apart): no atom is
stray, so rule 5a never fires there today and this design does not touch it (§9, an existing gap). Draft mode, which
already answers these scenes with the model.

**Anti-goals.** Answering a scene whose DFT energy does not split (a spin state or a charge transfer the model
cannot see) is worse than falling back: it is a confident wrong answer. Every path below that is not measured to
split stays on DFT, with its reason shown.

## 2. Why the model's answer is already "bonded part + free atoms"

The model is E = Σ_i [E_ref(Z_i) + Σ_k c̃_{s_i,k} B_{i,k}] with every descriptor B_i a sum over neighbours j of terms
carrying f_c(r_ij), no intercept (`surrogate::fit`, "Isolated-atom limit"; `providers::surrogate::lone` module doc).
A stray atom s has B_s = 0, and it appears in no other atom's descriptor (f_c(r) = 0 for r ≥ r_cut). Hence, exactly:

```text
E_model(scene) = E_model(bonded part alone) + Σ_s E_ref(Z_s)
F_s = 0,  F_i (bonded) = the bonded part's own forces
```

So the question is never what the model says. It is whether the **teacher** agrees:

```text
E_DFT(scene) =? E_DFT(bonded part alone, same box) + Σ_s E_ref(Z_s, this box)          (C)
```

(C) can fail for three reasons:
1. **Spin.** The teacher solves the whole cell at one fixed (N↑, N↓). (C) needs the scene's multiplicity to be the
   bonded part's own plus each stray atom's excess spin up: M_scene = M_frag + Σ_s (m_s − 1), where m_s is the
   reference multiplicity of Z_s (the lone-atom rule's parallel product, extended to a fragment). Two stray H next to
   a molecule at "automatic" (even electron count, so M = 1) breaks it: the restricted ensemble lies tens of mHa above
   the sum (4.6e-2 Ha measured for two H at 6.4 Bohr; `lone` module doc).
2. **Charge transfer at any separation.** One Fermi level μ_σ per spin spans the cell (Fermi–Dirac at
   kT = `qm::ELECTRON_KT` = 5 mHa). If a stray atom's highest occupied level lies above the fragment's lowest empty
   one in the same spin channel, or the reverse, LSDA moves charge across any gap (its separated-fragment error;
   e.g. a stray Li 2s↑ against an F₂ σ* LUMO). Whether that happens depends on eigenvalues the provider does not
   have before a solve, and it can change with the fragment's geometry (a stretching bond lowers its LUMO).
3. **Box.** A stray atom near a face differs from its reference by the grid box's confinement: +1.8e-3 Ha 4 Bohr
   from a face, +3.5e-2 Ha at 2 Bohr (`validate/references.md`). The free atom is the more physical answer, but a
   check must not judge composition on a box-affected solve.

## 3. Approaches (the user chose B)

- **A. Static rule** (the lone-atom rule's style): answer when the spin rule holds and every stray species is known to
  end at a gap. It answers from the first step, but it cannot see charge transfer (reason 2), so it would have to
  refuse mixed species outright, as rule 0b does, which is nearly every interesting case.
- **B. Measured composition (chosen).** Test (C) with the teacher at real geometries of the run, and let the model
  answer a combination only after it has passed. Charge transfer and spin effects are caught by measurement, not
  guessed. The cost is a few extra fragment-alone solves per combination, on the background auditor.
- **C. Model-vs-teacher only** (judge the fallbacks DFT already makes against model + Σ E_ref). It costs nothing extra,
  but it mixes the model's own error on the fragment with the composition error. The user's screenshot shows exactly
  that ambiguity: |ΔE| 37 mHa on an untrusted H₂ model. Rejected as the clearance test; kept as the ongoing audit (§6).

B keeps A's static exclusions as a floor (§4.2): they are cases the lone-atom rule already derived cannot split.

## 4. The design

### 4.1 Terms

- **Stray atom:** an atom with no neighbour within the model's r_cut, in a scene of two or more atoms, where at least
  one other atom has a neighbour (so the scene is not "lone"; rule 0b handles all-lone scenes, unchanged).
- **Bonded part:** every non-stray atom, as positioned in the scene (possibly several molecules).
- **Signature:** (sorted Z multiset of the bonded part; sorted Z multiset of the stray atoms; scene charge; the
  scene's multiplicity setting: "automatic" or a forced M; the solver surface, i.e. the store's provenance). A
  clearance and a refusal are per signature.

### 4.2 Static floor: never cleared, always DFT (with the reason shown)

- A charged scene (the references are neutral atoms).
- The spin rule fails: M_scene ≠ M_frag + Σ_s (m_s − 1). M_frag is the bonded part's "automatic" multiplicity
  1 + (Σ Z_frag mod 2) when the scene's setting is "automatic". With a forced scene M, it is the multiplicity the
  bonded part's fallback samples were solved at (it must be one value, else refuse). A stray species whose reference
  multiplicity is unknown on this surface refuses (the `lone` module's record: `ref_cache`, or a single allowed M).
- Two or more stray atoms not known to end at a gap, and mixed species among two or more stray atoms (`lone`'s
  `closed_spin_channels` and mixed-species rules, reused unchanged: the same cell-wide filling argument).
- A stray pair inside a shell-check fence, i.e. a measured interaction near r_cut (`trust::shell_fence_refusal`).
- A stray atom whose position is box-affected: its density at the faces is above `box_affected_face_density(h)`, or,
  before any solve, it lies within 4 Bohr of a face (the box rule's edge, §2 reason 3).

### 4.3 The composition check

When a strict-mode step meets an uncleared signature that passes the floor, the step falls back to DFT as today
(rule 5a). The fallback's own solve gives E_DFT(scene) and its forces. The provider then asks the **background
auditor** for the bonded part alone, in the same box (same grid, h, σ, tolerance, teacher, ionic surface), at
M_frag, with the bonded atoms at the scene's exact positions. With the canonical box references E_ref(Z_s, box), the
check passes when, at the audit criterion (`FallbackPolicy::agree`, n = the scene's atom count):

- energy: |E_DFT(scene) − E_DFT(frag) − Σ_s E_ref(Z_s)| / n ≤ `energy_std_per_atom`;
- forces: every bonded atom's force agrees between the two solves, and every stray atom's teacher force agrees with 0,
  each within `force_residual`'s bound;
- neither solve has a warning, an untrainable flag, or a box-affected face density (else the check is *void*: not
  counted, the next eligible step tries again);
- the scene solve's partial-occupation record (`ScfStats.partial`) holds no level shared between the stray atoms and
  the bonded part. My call, for the reviewer: a partial level at μ is reason 2's signature in the solve itself.

A signature is **cleared** after **K = 3 passing checks** at distinct geometries (each bonded-part geometry beyond
the store's thinning radius from the others, or a stray distance differing by more than 1 Bohr). One failing check
**refuses** the signature for the session: every later step with it is DFT, and the Activity row names the failed
check and its numbers. K = 3 is my call, for the reviewer. It is a physics identity checked at sampled geometries, not
a statistical verdict, but the user has just approved "more than 3 checks" for class trust, and the reviewer may
prefer K = 4 for consistency.

Check solves are never kept as samples. The fallback's own scene answer follows the ordinary keep rules unchanged:
the user's own H₂ + H steps are good data for H₂ when (C) holds, so they are kept as today.

### 4.4 Answering a cleared signature

For a cleared signature, strict mode evaluates the scene with every rule **except** 5a's isolated-atom refusal, the
stray atoms exempt from the per-atom rules 5b–5e and 6–7 as the isolated class already is (`mod.rs`, "The isolated
class is exempt"). The bonded part must pass every rule on its own (rules 1–4, 4b, 5b–5e, 6–7, fences). The answer is
the model's ordinary `predict` on the whole scene, which is §2's sum bit for bit (Neumaier summation in the model's
order). Its `Source` is `Surrogate`, with the step note "stray atoms: model for the bonded part + free-atom references
(composition measured: K checks)".

**Re-entry.** When a stray atom comes within r_cut it is no longer stray. Its pair now has f_c > 0, the scene is
judged by the ordinary rules, and the energy is continuous across r_cut (f_c → 0 smoothly), so no booking is needed
while the model keeps answering. A switch to DFT at re-entry is the existing model ↔ teacher transition, booked along
the forces (`transition_jump`).

### 4.5 Audits

Model answers of cleared signatures are audited like any model answer (the same cadence and criterion, composite
scene against teacher). A failed audit on such an answer refuses its signature for the session, and also counts
against the model as any failed audit does. This is approach C's comparison, now used where it belongs: it catches a
signature whose composition holds at the checked geometries but not at the current one (reason 2's
geometry dependence).

### 4.6 Persistence

Session-only in this version: clearances and refusals live in the provider and are lost on restart, a new model
install, or a surface change. Re-earning costs K fragment solves on the auditor, while the fallbacks run anyway.
Persisting them in the collection is left for later (it needs a format record and a migration story).

## 5. What the user sees

- Activity, while checking: rule "5a stray atom", outcome "checking composition 1 of 3" (then 2, 3). After clearance:
  ordinary model rows, with the note. A refused signature: "5a stray atom · composition fails" with the measured
  |ΔE|/n and the worst force, or the floor's reason in words ("spin: two stray H at automatic would need M = 3").
- Model tab trust card: "Stray atoms: H₂ + H cleared (3 checks)" or "refused (why)", per signature seen this session.
- The orbital and density displays are unchanged (they come from the display solves, not from this rule).
- README: one paragraph under the model's fallback rules.

## 6. Validation

- **Unit (providers, Morse teacher):** a fake composite offset makes (C) fail and keeps the signature on DFT. With no
  offset, three checks clear it, and the model's answer equals E_model(frag) + Σ E_ref bit for bit, with zero stray
  force. The floor: forced charge, the spin rule (two stray H at automatic), mixed stray species, a box-affected stray.
  Re-entry restores the ordinary rules. A failed audit refuses the signature.
- **GPU (validate surrogate, real teacher):** H₂ + H at 8 Bohr, automatic (M = 2), clears and answers, with
  |E_model − E_DFT| within the audit tier at a fourth, unchecked geometry. **F₂ + Li at 8 Bohr** (the charge-transfer
  case) fails the check, or the partial-level test, and stays on DFT. Report the measured transfer, without
  pre-judging its size.
- **Dynamics:** an NVE run in which a stray H returns within r_cut and leaves again: E_conserved drift within the
  dynamics suite's bound across both crossings.
- Gates: `cargo test -p providers --release`, `validate --test surrogate`, `validate --test dynamics`.

## 7. Physics honesty

The answer for a cleared signature is "bonded part + free atoms", the physically right separated limit, which DFT in
a finite box only approximates (box artefacts of up to a few mHa near faces). Where LSDA itself transfers charge
between separated fragments (reason 2), DFT is the less physical of the two, but the model must not answer there:
it would disagree with its own teacher. The Activity row says which.

## 8. Left to the builder

Where the signature map lives (the provider, beside the misfit and fence state), the exact Activity enum names, the
trust card layout, and how the auditor queues a fragment job behind audits and references (the box-reference jobs are
the pattern).

## 9. Known gaps, not in scope

- Two bonded fragments beyond r_cut of each other (no atom stray) are answered by the model today with no spin
  check: two O₂ far apart at automatic (M = 1) would be wrong. That is an existing gap; worth its own item.
- Spin-forced shared collections (the "charged collections" item) will change how M_frag is known; this design reads
  it from the fallback samples until then.
