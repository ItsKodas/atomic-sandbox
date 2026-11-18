# Stray atoms: the model answers a molecule plus far-away atoms (measured composition)

Status: design, 2026-11-18, revised after the physics review (APPROVE WITH CHANGES: C1, I1–I6, M1–M6, all taken).
User request and choice of approach (B, "measured") in the session of 2026-11-18; the user accepted the coordinator's
recommended calls. Branch `stray-atoms`, based on the smearing branch (`scf-stall`, 92000979). Atomic units.

## 1. Intent

The user's scene: H₂ with a third H that has drifted more than r_cut (6 Bohr) from it. Every step goes to DFT
(rule 5a, "isolated atom"), so the run is slow for as long as the H stays away, even when the model knows H₂.
The user's words: "ideally that way it won't fall back to DFT and run smoothly until they interact".

**Goal:** in strict mode, a scene made of a bonded part plus one or more *stray* atoms (no neighbour within r_cut)
is answered by the model, once it has been **measured** that this combination's DFT energy splits as the bonded
part's energy plus the free atoms', for fragment geometries like the current one. When a stray atom comes back within
r_cut, the ordinary rules take over. Nothing is hardcoded per element (rule 1): clearance is a measurement keyed by Z.

**Non-goals.** Making an untrusted bonded part answerable: it still has to pass every rule on its own. Spin-aware
models. Forced multiplicities (v1 refuses them, §4.2). Fragments that are each bonded but far from each other (no
stray atom; §9). Draft mode, which already answers these scenes with the model.

**Anti-goals.** Answering a scene whose DFT energy does not split (a spin state or a charge transfer the model cannot
see) is a confident wrong answer, worse than falling back. **Teaching the model from such a scene is worse still**: a
stray atom has no features, so a composition error lands in the bonded part's coefficients and spoils bonded-only
scenes too (review C1). Every path below that is not measured to split stays on DFT and out of the fit, with the
reason shown.

## 2. Why the model's answer is already "bonded part + free atoms"

The model is E = Σ_i [E_ref(Z_i) + Σ_k c̃_{s_i,k} B_{i,k}], mean 0, no intercept (`model.rs:14-21`). f_c and the radial
basis are exactly 0 at and beyond r_cut (`basis.rs:96-119`), pairs with r ≥ r_cut are excluded (`descriptor.rs:1024-1027`),
and the body-3/body-4 terms are built only from neighbours inside r_cut (`model.rs:629-639`). A stray atom s therefore
has B_s = 0 and appears in no other atom's descriptor, and there is no whole-scene term:

```text
E_model(scene) = E_model(bonded part alone) + Σ_s E_ref(Z_s)     (equal to a few ulp: Neumaier adds the references
F_s = 0 exactly; F_i (bonded) = the bonded part's own forces, bit for bit      in scene order first, model.rs:574-577)
```

E is C¹ and F is C⁰ at r_cut (`basis.rs:18-24`). The question is whether the **teacher** agrees:

```text
E_DFT(scene) =? E_DFT(bonded part alone, same box) + Σ_s E_DFT(stray atom s alone, at R_s, same box)        (C)
```

(C) uses each stray atom solved alone **at its own position in the same box** (review I3), not the canonical
reference: that cancels the box confinement and the egg-box term exactly, as the fragment-alone solve does for the
bonded part. It also gives the eigenvalues §4.3's margin needs. The model's answer uses `model.ref_energy` (the model
surface, booked as today, review M2), and (C) compares teacher energies with teacher energies in one convention:
raw energies throughout, never mixed with canonical ones (`lone.rs:43-48`).

(C) can fail for two physical reasons:
1. **Spin.** The teacher solves the whole cell at one fixed (N↑, N↓). (C) needs M_scene = M_frag + Σ_s (m_s − 1),
   each part's excess spin aligned up: 2S = (M_frag − 1) + Σ (m_s − 1). That is sufficient, not necessary. The
   antiparallel product also composes by spin-flip symmetry, but nothing seeds it, so refusing it is conservative.
2. **Charge transfer at any separation.** One μ_σ per spin spans the cell (Fermi–Dirac, kT = `qm::ELECTRON_KT` =
   5 mHa). LSDA has no derivative discontinuity: if a stray atom's highest occupied level lies near the fragment's
   lowest empty level in the same spin channel, or the reverse, charge moves across any gap. A transfer that stops at
   a fractional amount pins both levels at μ. Its energy gain (~Δε²/2η) can be well below 1 mHa, so the energy test
   alone misses the onset; a whole-electron transfer is caught by the energy (IP − EA scale). It depends on the
   fragment's geometry: a stretching bond lowers its LUMO (stretched H₂'s σu falls toward H 1s↑). Example from the
   review, an estimate and not a measurement: O₂ π* near −0.25 Ha against H 1s↑ near −0.27 Ha.

## 3. Approaches (the user chose B)

- **A. Static rule** (the lone-atom rule's style): answer when the spin rule holds and every stray species ends at a
  gap. It answers from the first step, but it cannot see charge transfer, so it would refuse mixed species outright,
  which is nearly every interesting case.
- **B. Measured composition (chosen).** Test (C) with the teacher at real geometries of the run, plus a frontier
  margin from the same solves. A combination is answered only inside the region of fragment geometries it was
  measured in. The cost is a fragment-alone solve and a cheap stray-atom solve per check, on the background auditor.
- **C. Model-vs-teacher only:** judge the fallbacks DFT already makes against model + Σ E_ref. It costs nothing extra,
  but it mixes the model's error on the fragment with the composition error (the user's |ΔE| 37 mHa on an untrusted
  H₂). Rejected as the clearance test. Audits still compare model and teacher, but a failure is attributed by a
  check (§4.5, review I6).

B keeps A's static exclusions as a floor (§4.2).

## 4. The design

### 4.1 Terms

- **Stray atom:** an atom with no neighbour within the model's r_cut, in a scene of two or more atoms, where at least
  one other atom has a neighbour. A scene in which every atom is alone is rule 0b's, unchanged.
- **Bonded part:** every non-stray atom, as positioned in the scene (possibly several molecules).
- **Signature:** (sorted Z multiset of the bonded part; sorted Z multiset of the stray atoms; scene charge;
  "automatic" multiplicity; the solver surface). A refusal is per signature. A clearance is per signature **and
  region** (§4.3).
- **Fragment surface:** the bonded part's ionic surface is defined by the per-species tables, not by the scene's tag.
  When the only pseudised species are stray atoms, the fragment's per-scene tag differs from the scene's
  (`pseudo.rs:284-296`; review M5). The stray species must be in the model's basis (rule 2, `model.rs:569`).

### 4.2 Static floor: never cleared, always DFT, never fitted (the reason shown)

- A charged scene (the atoms' solves are neutral).
- **A forced multiplicity** (v1; review I5). A forced-M store's samples are whole-scene samples at that M
  (`mod.rs:3187`), and collections refuse forced M anyway (`shared.rs:378`).
- The spin rule fails at "automatic": 1 + (Σ Z mod 2) ≠ M_frag + Σ_s (m_s − 1), with M_frag the bonded part's own
  automatic multiplicity 1 + (Σ Z_frag mod 2). This is the convention the bonded-only samples and ordinary audits
  share. m_s is the reference multiplicity of Z_s on this surface (`lone`'s record: `ref_cache`, or a single allowed M).
  Unknown refuses. Where automatic is not the fragment's ground state (O₂), the scene's single-μ filling can land in
  another state (e.g. O₂(9,7) + H↓). §4.3's measurement catches that, but warm starts make it history-dependent, so
  the audits (§4.5) must keep covering it.
- Two or more stray atoms not known to end at a gap, or mixed species among two or more stray atoms (`lone`'s
  `closed_spin_channels` and mixed-species rules, reused).
- A shell-check fence on any pair of (a stray atom's species, any other species in the scene) (review M4;
  `trust::shell_fence_refusal`).
- A stray atom whose position is box-affected: before a solve, within 4 Bohr of a face (the box rule's edge); after
  one, face density above `box_affected_face_density(h)`.

### 4.3 The composition check and the cleared region

When a strict-mode step meets a signature that passes the floor and whose clearance does not cover the current
fragment geometry, the step falls back to DFT as today (rule 5a), and its solve gives E_DFT(scene), forces and
levels. The provider asks the **background auditor** for the bonded part alone and for each stray atom alone. Both use
the same box (grid, h, σ, tolerance, teacher, per-species ionic tables), at the scene's exact positions, the fragment
at M_frag and each atom at m_s. The check **passes** when all of these hold:

- **Energy, absolute:** |E_scene − E_frag − Σ_s E_atom(R_s)| ≤ `energy_std_per_atom` × n_stray (review I3: the error
  depends on the stray atoms, not on the scene's size).
- **Forces:** every bonded atom's force agrees between the scene and fragment solves, and every stray atom's agrees with
  its own solve's force, each within `FallbackPolicy::force_residual`'s bound.
- **Frontier margin (review I1):** per spin, (fragment LUMO − stray HOMO) ≥ 27.6 kT = 138 mHa and (stray LUMO −
  fragment HOMO) ≥ 138 mHa, from the parts' own solves (the `lone` gap criterion). For two or more stray atoms, each
  one is taken against the fragment.
- **Levels:** the scene's partial levels (`ScfStats.partial`) and frontier levels per spin match the union of the
  parts' levels, within the box splitting (1.8 mHa).
- **Clean solves:** no warning, no untrainable flag, no box-affected face density in any of the solves. Otherwise the
  check is *void*: not counted, and the next eligible step tries again.

`PartialLevel` holds spin, index, ε and f only (`qm/src/occupations.rs:385-394`). If the per-spin HOMO/LUMO
eigenvalues are not already reported by a solve, the builder adds them to `ScfStats` (both teachers).

**Clearance is local (review I2).** A signature's **cleared region** is the per-feature envelope (the existing
`surrogate::Envelope`) of the bonded-part descriptors of its passing checks. It opens after **K = 3 passing checks**,
each with a bonded-part geometry beyond the store's thinning radius from the others. A check that differs only in
stray distance does not count toward K. A strict-mode step is answered (§4.4) only when the current bonded-part
descriptors lie inside the region. A step outside it falls back and is one more check: when it passes, the region
grows. **One failing check refuses the signature for the session.** The Activity row names the failed test and its
numbers. K = 3 is the reviewer's recommendation: each check tests an identity, so a fourth adds no statistical power;
the geometry coverage is what the region provides.

**Samples (review C1).** A scene sample containing a stray atom is **held out of the fit** (an in-memory status like
`Q_LONE`, never written) unless its own check passed or its bonded geometry lies in a cleared region of its signature.
It is released when that becomes true, and stays held out for good if the signature is refused. Floor-refused
signatures' samples are never kept. Check solves (fragment alone, atoms alone) are never kept. Samples with a stray
atom that are already in a loaded collection are held out the same way until a clearance covers them (my call). They
are not deleted or rewritten, so the user's collection is untouched.

### 4.4 Answering inside a cleared region

Strict mode evaluates the scene with every rule except 5a's isolated-atom refusal. The stray atoms are exempt from the
per-atom rules 5b–5e and 6–7, as the isolated class is (`mod.rs`, "The isolated class is exempt"). The bonded part must
pass every rule on its own: rules 1–4, 4b, 5b–5e, 6–7 and the fences. **Every per-atom normalisation uses n_bonded,
not the scene's n** (review I4): rule 7's division and its (Nσ_e)² floor (`mod.rs:189`, `model.rs:482-486`), and the
audit criterion. The answer is `predict(scene)`, which is §2's sum. Its `Source` is `Surrogate`, with the step note
"stray atoms: model for the bonded part + free atoms (composition measured, K checks; worst residual x mHa, frontier
margin y mHa)" (review M6, rule 7).

**Re-entry (review M3).** When a stray atom comes within r_cut it is no longer stray, and the ordinary rules judge the
scene. In practice rule 5b fires first (coordination far below anything trained, `mod.rs:110-116`), so the step goes
to the teacher and the switch is booked by `transition_jump`. The builder must verify that path. The model's own energy
is continuous across r_cut (C¹).

### 4.5 Audits (review I6)

Model answers inside a cleared region are audited like any model answer, at the same cadence, comparing the composite
scene against the teacher, with n = n_bonded. On a failed audit:
1. The signature is refused at once (DFT from the next step, its samples held out).
2. A fragment-alone and atoms-alone check is queued at that geometry, to attribute the failure.
3. The failure counts against the model only if the model is wrong on the fragment alone, by the audit criterion.

### 4.6 Persistence

Session-only in v1: clearances, regions and refusals live in the provider and are lost on restart, a new model
install, or a surface change. Re-earning a region costs K fragment solves plus K cheap atom solves on the auditor,
while the fallbacks run anyway. Persisting them is left for later (a collection record and a migration story).

## 5. What the user sees

- Activity, while checking: rule "5a stray atom", outcome "checking composition 1 of 3", then 2 and 3. Outside a
  region: "checking composition (new geometry)". Inside: ordinary model rows with the note. Refused: "5a stray atom ·
  composition fails", naming the failed test with its numbers (|ΔE|, the worst force, or the frontier margin), or the
  floor's reason in words ("spin: two stray H at automatic").
- Model tab trust card: "Stray atoms: H₂ + H cleared (3 checks)" or "refused (why)", per signature this session. A line
  counting the stray-atom samples held out of the fit.
- README: one paragraph under the model's fallback rules.

## 6. Validation

- **Unit (providers, Morse teacher):**
  - A fake composite offset fails (C) and keeps the signature on DFT with its samples held out.
  - With no offset, three checks at distinct fragment geometries clear the region. The model's answer then equals
    E_model(frag) + Σ E_ref within a few ulp, bonded forces are bit for bit and stray forces exactly 0 (review M1).
  - Three checks differing only in stray distance do not clear.
  - A fragment geometry outside the region falls back and checks.
  - The floor: charge, forced M, the spin rule (two stray H at automatic), mixed stray species, a box-affected stray.
  - Rule 7 and the audit use n_bonded.
  - A failed audit refuses the signature and attributes the failure by a check.
  - A loaded collection's stray samples are held out, not rewritten.
- **GPU (validate surrogate, real teacher):**
  - H₂ + H at 8 Bohr, automatic (M = 2): clears, and at an unchecked fourth geometry inside the region
    |E_model − E_DFT| ≤ the audit tier.
  - **F₂ + Li at 8 Bohr** (the charge-transfer case) fails the margin or the levels test, or the energy test, and stays
    on DFT. Report the measured transfer and margin without pre-judging their size.
  - Stretched H₂ (r ≥ 2.5 Bohr) + H: report the measured σu − 1s margin.
- **Dynamics:** an NVE run in which a stray H returns within r_cut and leaves again. E_conserved drift within the
  dynamics suite's bound, with the booked jumps reported separately, each within the check's tolerance (review M3).
- Gates: `cargo test -p providers --release`, `validate --test surrogate`, `validate --test dynamics`.

## 7. Physics honesty

The answer inside a cleared region is "bonded part + free atoms", the physically right separated limit. The check
removes the box terms, so it measures composition alone. Where LSDA itself transfers charge between separated
fragments, DFT is the less physical answer, but the model must not answer there, because it would disagree with its
own teacher. The Activity row says which test refused, with its numbers.

## 8. Left to the builder

- Where the signature and region state lives (the provider, beside the misfit and fence state).
- The Activity enum names and the trust card layout.
- How the auditor queues and warm-starts the fragment and atom jobs (the box-reference jobs are the pattern).
- Adding per-spin HOMO/LUMO to `ScfStats` if absent.

## 9. Known gaps, not in scope

- Two bonded fragments beyond r_cut of each other (no atom stray) are answered by the model today with no spin check.
  For example, two O₂ far apart at automatic (M = 1) would be wrong. That is an existing gap and worth its own item;
  this design's check would extend to it naturally.
- Spin-forced and charged shared collections (the "charged collections" item) would let v2 lift the forced-M floor.
