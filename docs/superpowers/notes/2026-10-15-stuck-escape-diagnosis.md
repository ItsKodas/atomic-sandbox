# Estimated mode: the fallback storm after an atom edit, the escape, the post-edit cycle — diagnosis

Worktree `D:\Projects\Physics-dbg` (main b466fcd). Diagnosis only: no fix applied. The throwaway
diagnostics are `diag_stuck_escape` / `diag_offcentre` at the end of the tests module in
`crates/app/src/sim.rs` (the worker, GPU teacher, background auditor, worker refits; env-driven) and
`crates/surrogate/tests/diag_envelope.rs` (CPU, reads a saved `.model`). Logs:
`C:\Users\koda\AppData\Local\Temp\claude\D--Projects-Physics\d8535878-a71d-4fab-8d07-5ec70933a386\scratchpad\diag\`
(`probe2.log`, `run1.log`, `run2.log`, `run3.log`, `run5.log`, `off1.log`; models `heated.model` =
`escape.model` (281 samples incl. one r = 5.125 tail sample), `after.model` (post-storm, 274)).
The GPU was shared with the user's running app, so solves were 2–3× slower than the CLAUDE.md numbers.

## Reproduction

```
cargo test -p app --release --no-run   (target-dir D:\Projects\Physics-dbg\target)
DIAG_FULL=1 DIAG_HEAT_T=4000 DIAG_HEAT_FS=400 DIAG_MODE=run DIAG_PLAY_S=60 DIAG_ADD="1:0,3,0" \
  target/release/deps/app-<hash>.exe diag_stuck_escape --nocapture
```
= H₂ preset (49³, h 0.25), estimated, the full default pass (273 solves, trusted on 273), CSVR to
4000 K for 400 fs then NVE at speed 1000, then **add one H** 3 Bohr from the H₂ centre while
playing. The escape alone: `DIAG_HEAT_T=8000` (or 15000) — the hot H₂ dissociates and both atoms
leave the cell under the surrogate (run1: |x| = 287 Bohr before any edit, 3·10⁵ Bohr a minute
later), no edit needed.

## 1. What the edit added, and why the model answered again

**The edit added one H** (a known species). Evidence: the user's first post-edit step is a
`Fallback` *with* a teacher gap, which only rules 5–8 produce (the model was consulted); a new
species is rule 2 → teacher surface, no gap. Probed (probe2.log): He and Li added → "species He/Li
is not in the model (H) — rebuilt at the next refit — teacher answered", energy = teacher's, no
gap, sample kept as essential. **New species are refused correctly.** The numbers fit one H: the
user's teacher energy = E_model + gap = −16.709 + 15.5 ≈ −1.21 Ha ≈ stretched H₂ (≈ −0.80) + one
free H (E_ref = −0.4242 on this grid). Adding H atoms in the probes gave model energies from −3.9
to +12.1 Ha (teacher gaps up to −14.3 Ha) — the model's extrapolation, 20–150 training spreads
outside its descriptor envelope (feature b3[n=0;n=2;l=0] etc.). Every one is a rule-5 fallback.

**Why it came back as "Surrogate, trusted"** — rule 5 (the envelope), not trust, not variance:
the hot H₃ falls apart and an atom whose neighbourhood is (nearly) empty is **inside** the
envelope. The envelope is a per-feature box in *training-spread* units
(`crates/surrogate/src/envelope.rs:116-136`; unit `max(std, 0.1·rms)`). Its module doc
(envelope.rs:36-40) relies on the vanishing direction firing ("g ≈ rms, so g/u ≥ 10"), which
holds only while no training atom ever had a sparse neighbourhood. One kept sample of a stretched
pair is enough to break it — measured (run2.log, `diag_envelope` on `escape.model`): a hot H₂
fallback at r = 5.125 Bohr (face density 1.87e-5, just under the box rule) was kept; it lowered
min(b2[n=0] = Σf_c) from 0.531 to 0.0515 with unit 0.10, so an isolated H is **0.51 spreads**
out (limit 3.0; it was 6.8 before that sample). The variance is no help there either (the
envelope's own argument: features → 0 ⇒ leverage → 0): f_std/√3 = 9e-5. From then on every
isolated atom is answered with E = Σ E_ref and **F = 0**, exactly the user's state at
t ≥ 1691 fs (constant E_total, ~13 000 fs/s). In the user's run the hot-H₂ samples (d ≈ 3.5–4+
at 11–19 kK, the 246→250 audits) had already lowered that minimum (f_c drops below 0.3 at
r ≈ 3.8 Bohr), so the same thing happened when H₃ split.

The answer is wrong, not just unsafe: the teacher's dissociated pair (restricted lowest
multiplicity) lies above Σ E_ref — measured gap +4.6e-2 Ha at r = 6.4 (run5.log) — and its force
is not zero.

**Why the ~15 Ha offset is not caught.** It is not a model error at the end: E_cons was
restarted by the edit (an edit restarts the history) at the model's extrapolated energy (−16.7),
then the fallback steps book ΔF·Δr into W (by design), so E_cons − E_total is the integrated
gap, i.e. the junk energy at the edit geometry, carried for ever. Nothing compares E_cons with
anything external. The audits cannot see the escape: (a) a dissociated pair in the cell has its
density on the faces → the audit is `BoxAffected`, recorded but not judged; (b) once a nucleus is
outside the cell the background teacher **refuses** the geometry (`check_inside_cell`,
`crates/sim/src/provider.rs:315`, via `problem()` :341) and the answer is the
`(JobKind::Audit, Err)` branch (`surrogate_ks.rs:2159`): "background teacher failed — not
judged". Measured (run1.log): `last_audit` stayed `None` over 1.5 M answered steps at audit_every
2000. Saturation is irrelevant here; the judge is blind, not late.

## 2. Why the ~140 fallbacks were never added as samples

**The box rule**, nothing else (no quarantine, no fence, no thinning hits). `admit()`
(`surrogate_ks.rs:1985`): a teacher answer whose face density exceeds
`box_affected_face_density(0.25)` = 2e-5 e/Bohr³ is used for the step and never kept. Measured
(run3.log): all 122 storm fallbacks were rejected for face densities 2.1e-5 … 2.9e-4; samples
stayed 274, epoch 2, quarantine 0 — the user's "samples 250, epoch 5 throughout". In the 12-Bohr
49³ box nearly any H₃/H₄ is box-affected: an H₃ triangle with one atom 2 Bohr from the centre
already reads 4.3e-5 (probe2.log). The user's `D:\Projects\Physics\my.model` (17:00, an H₄ model
on 49³) shows the same thing from the other side: 329/329 samples have face density > 2e-5.
So the model can never learn the edited scene in this cell: the rule is by design (the doc says
the fix is a bigger box), but the user gets "trusted, fallbacks 100 %, stuck" with no statement
that **nothing is being learned and why**.

**230 vs 250** is not a removal: the panel's "last refit N samples" is `FitReport::n_samples`
(`app/src/sim.rs` `refit_if_pending`: `fit.map(|r| r.n_samples)`), which excludes quarantined
**and box-excluded** samples (sidecar code 3, `exclude_box_samples`, surrogate_ks.rs:1033: pass
samples with face density > rule, dropped once the store holds Q′ + 30 without them). The store
count (`samples()`) includes them. `ModelStatus` carries `quarantined` but no box-excluded count,
so the panel cannot explain the 20. (My 49³ pass from the 1.6 Bohr preset had 0 box-affected
samples; the user's 246-solve pass from a hotter/stretched start evidently had ~20.)

## 3. Why the surrogate moves atoms out of the cell

Under DFT every evaluation goes through `problem()` → `check_inside_cell` (R12: 3h from every
face), so a nucleus reaching the margin returns `ProviderError::Invalid` and the driver pauses
with the last good state. `SurrogateKs::evaluate` (`surrogate_ks.rs:2317`) never checks: a model
answer (`:2420`) is returned for any positions — the cell guard exists only inside the teacher.
Combined with §1 (free atoms answered with F = 0), atoms fly out in straight lines. The walls
(`DynCommand::SetWalls`, off by default) act in the driver, independent of the provider, so with
walls on they would contain the atoms under the surrogate too (half-width ≤ max_coord − h/2);
they are not the guard and are off by default.

## 4. E_total = +578 Ha on fallback steps

The one-surface rule (`teach()`, `surrogate_ks.rs:2285`: `(m.energy, true)`) reports E_model on
every fallback with a consulted, trusted model — including rule 5, where the geometry is by
definition outside what the model can answer and E_model is a polynomial extrapolation (probes:
−3.9 … +12 Ha for 3–5 H). It feeds W through the mismatch booking (`dynamics.rs:1441-1451`), so
E_cons stays continuous: run3 — E_total swung −0.38 … −1.66 Ha while E_cons stayed
−0.4316 … −0.4325 and W_mismatch reached −1.2 Ha. So it is mostly bookkeeping, but not only:
(a) the trapezoid booking of ½(ΔF_old + ΔF_new)·Δr with extrapolated model forces of order Ha/Bohr
is O(dt²) with a huge prefactor — the user's E_cons wobbled by 0.8 Ha (−16.1 … −16.9); (b) the
restart after the edit anchors E_cons on the junk value for the rest of the run (§1).

## 5. The cyclic fallbacks near equilibrium after going back to H₂ (coordinator's addendum)

Tested hypotheses:
- **(a) storm answers poisoned the store — refuted.** Both the user's log (samples 250 and epoch 5
  through the whole storm) and the reproduction (122 storm fallbacks, 0 kept, quarantine 0;
  `after.model` reproduces the pass model at r = 1.4–2.0 to 5e-6 Ha) show that nothing from the
  storm entered the store: the box rule (§2) rejected every answer. The 13 samples the user's model
  gained afterwards (250 → 263, epoch 5 → 18) came later (escape/recovery edits), not from the
  storm.
- **(b) a quarantine fence at equilibrium — not produced.** No refit in any run quarantined
  anything (q = 0 throughout); the storm cannot create a fence because nothing from it is kept.
- **The gap +1.3e-2 → +1.6e-3 Ha is the box's confinement, measured.** `diag_offcentre` (off1.log):
  a cold (300 K) equilibrium H₂ moved 2.6 / 2.1 / 1.6 Bohr from a face is answered by the model on
  every step (0 fallbacks), while its audits come back `BoxAffected`, **not judged and not kept**,
  with teacher − model = +5.3e-3 / +1.5e-2 / +2.5e-2 Ha (face density 8.6e-5 / 3.3e-4 / 4.9e-4).
  After the escape the H₂ is off-centre; the user's gaps are exactly this range, positive (the
  wall raises the teacher), decreasing as the molecule drifts inward. Every such teacher answer is
  box-affected → never kept → samples and epoch frozen: **the same mechanism as #2** for the
  "not learning" half.
- **What makes those steps fall back at all — not reproduced** (the reproduced off-centre H₂ with a
  clean model never falls back). The rules that can fire for a trusted model at an in-envelope
  equilibrium geometry and leave the sample unkept are rule 6/7 + box rule, or a fence; the
  cyclic 50–60 % pattern points at **rule 6 on aleatoric noise**: `force_std` includes σ_f
  (`crates/surrogate/src/model.rs:84`: var = σ_f² + g̃ᵀΣg̃), and rule 6 compares it, unlike rule 7
  (which subtracts the floor, `surrogate_ks.rs:1928`), against min(max(2e-3, 0.1|F|), 1e-2)
  (`:1911`). Near the force zero the bound is the absolute 2e-3, near the turning points 0.1|F|:
  with σ_f ≳ 2e-3 the model falls back on every pass through equilibrium and answers near the
  turning points — a vibration-phase cycle. σ_f is a single homoscedastic noise; samples whose
  residuals pass the relative force gate (10 % of |F|, cap 1e-2: hot, dragged, near-wall or
  box-affected-but-exempt pass samples) inflate it. Demonstrated on the user's own `my.model`
  (H₄, 49³, every sample box-affected): σ_f = 9.6e-3 Ha/Bohr, so f_std/√3 = 9.6e-3 at an H₂ at
  r = 1.8 vs a bound of 2e-3. Whether the user's 263-sample model had σ_f that large is unknown
  (its autosave is written only when the worker retires; none exists for the running pid 44512).
  **Discriminator:** the panel's stuck line ("not answering here: … <rule>") or the fallback
  step's note names the rule; "force std … > bound 2.00e-03" confirms this, "near a quarantined
  geometry" would mean a fence.

## Root causes, by design vs bug, and proposed fixes

| # | Cause | Where | Verdict | Proposed fix |
|---|---|---|---|---|
| R1 | No R12 cell guard on a model answer | `surrogate_ks.rs:2317/2420`; M6 `crates/providers/src/surrogate/mod.rs:727/843` | **bug** (DFT pauses, the surrogate does not) | Call `check_inside_cell` (M6: `scene.check_inside_cell()`) at the top of `SurrogateKs::evaluate`, before the gate: same `ProviderError::Invalid`, same pause and message as the teacher. The model is a function of positions only, but the store, the envelope and every audit are of in-cell geometries; the cell is part of the provenance's meaning. |
| R2 | Envelope admits empty/sparse neighbourhoods after one tail sample | `envelope.rs:116-136` (doc :36-40) | **bug in the design premise** | Judge the vanishing direction relatively, not in spread units: for the non-negative coordination features (b2 n = 0 = Σf_c, and the b3 self-products) an atom is outside when b_k < min_k / κ (fixed factor, e.g. 3 — or keep 3.0 and use log-space spreads for these features), so one sample at f_c = 0.05 does not make f_c = 0 "inside"; and an atom with **no neighbour within r_cut in a scene of N > 1** always falls back (the model's Σ E_ref limit is not the teacher's restricted-multiplicity limit: +4.6e-2 Ha measured at r = 6.4). No constant changes. Add a test: train on the pass + one r = 5.1 sample, query r = 7 → Fallback. |
| R3 | Box rule rejects every answer of a box-affected scene → fallback storm that never learns, under "trusted" | `admit()` :1985 | **by design** (an invariant model must not learn confinement); the **readout is the bug** | Keep the rule and the 2e-5 constant. Make the stuck note say *why nothing is learned*: count consecutive fallbacks whose sample was refused as box-affected; when the streak is all box-affected, `stuck()` says "every teacher answer here is box-affected (face ρ … > 2e-5): the model cannot learn this scene in this cell — enlarge the grid", and the panel shows it in amber. Optionally offer the grid change. Show `box_excluded()` in `ModelStatus`/panel so "230 of 250 in the fit" is explained. |
| R4 | One-surface accounting reports E_model outside the envelope | `teach()` :2285 | **by design for rules 6–8, a bug for rule 5** | On a rule-5 fallback (and the first evaluation after an edit) report the **teacher** surface with a booked flip — `refit_jump = E_teacher − E_model(r)`, the failed-audit path that already exists — and flip back with a booked jump on the next model answer. E_cons stays continuous, E_total is physical, the trapezoid never books extrapolated Ha/Bohr forces, and an edit no longer anchors E_cons on junk. |
| R5 | Audits blind exactly where the model is wrong (box-affected not judged; out-of-cell refused) | `judge_audit`, `handle_answer` :2159 | by design (box), consequence of R1 (cell) | R1 removes the out-of-cell case. For box-affected audits keep "not judged", but count them in the stuck/readout as above (R3). |
| R6 | Rule 6 includes the aleatoric σ_f (rule 7 excludes it) | `variance_check` :1911 vs :1928; `model.rs:84` | **bug** (inconsistent with rule 7's own reasoning: a fallback on noise teaches nothing) | Compare the per-atom **epistemic** force std √(f_std² − 3σ_f²)/√3 with the bound, as rule 7 does for energies. A model whose σ_f itself exceeds the tier is a trust question, not a per-step one: add it to the trust rule (σ_f ≤ force_std_abs ⇒ else untrusted with the reason "force noise σ_f … > 2e-3"), which then triggers the existing recovery (degree cap, quarantine of worst offenders, box exclusion) instead of a silent half-DFT cycle. |

The coordinator's suggested fixes, against the evidence: "an edit must not let the previous
geometry's answers poison the store" — nothing was poisoned (§5a), no change needed for that;
"quarantine samples whose gap is orders above the thresholds individually" — such samples never
enter (box rule) in this scenario; "a fence must expire/be re-judged" — no fence formed;
"a trusted model whose recent teacher gaps exceed its claimed accuracy must lose trust and refit"
— **yes, in a guarded form**: gaps from box-affected answers must not count (they are the box),
but a run of non-box-affected fallbacks at in-envelope geometries whose gaps exceed the tier is
the failed-audit condition and should be treated as one (it already untrusts; R6 is what lets the
model fall back without that evidence ever being judged).

## M6 (`D:\Projects\Physics-m6`, read-only)

Same code paths, same bugs: `crates/providers/src/surrogate/mod.rs:727` `evaluate` has no
`check_inside_cell` (the guard is only in `providers/src/ks.rs:89`); the envelope is unchanged
(`crates/surrogate/src/envelope.rs`, identical); `BOX_AFFECTED_FACE_DENSITY = 2e-5`
(`surrogate/policy.rs:31`) with the same `!essential` admission (`surrogate/quarantine.rs:405`);
one-surface `(m.energy, true)` on every consulted fallback (`surrogate/mod.rs:694`). R1–R6 apply
there unchanged.
