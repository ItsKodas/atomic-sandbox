# One model for radicals and closed shells (2026-11-13)

Status: design, awaiting the user's approval. Measurements behind it: `crates/validate/references.md`
"Why the collection fails its residuals at every refit" and "Raising the body-4 degree" (2026-11-12).

## 1. Intent

The user's goal for the estimated model, in their words: "after doing heaps of training, I could place a
heap of atoms down in the estimated model and watch them all form bonds and interact with each other" —
scenes far larger than one DFT box (their training boxes hold up to 6 atoms today). That is the M8b
Sandbox (`2026-10-20-m8-shared-model-and-sandbox-design.md` §5): every step from the trained model, no
DFT, atoms the model does not know frozen.

Today the user's shared H/N/O collection is untrusted after every refit, and each refit quarantines more
of its good H₃O labels until the 10 % cap, after which it stays untrusted for good. The diagnosis:
H₃O (the neutral radical) and H₂O conflict in one fit because the model cannot tell an O with three H
neighbours from one with two well enough — a capacity limit of the four-body (body-4) block, whose degree
is capped at 2. Spin is not the key: on the same samples, raising that degree to 3 makes the collection
trusted and the recovery releases 68 held samples instead of holding more.

A heap of loose atoms is all open-shell fragments at first, and every bond forms through radicals
(H + H, O + H → OH, OH + H → H₂O). So radicals are the main case for Sandbox, not an edge case, and they
must be fitted by the same local model as closed shells.

**Not done:** splitting collections by spin multiplicity. Multiplicity is a property of the whole box;
in a large box it says nothing about any one atom, two distant radicals make the box flip between singlet
and triplet at nearly the same energy, and Sandbox has no teacher to read it from. (Considered and
rejected 2026-11-12/13 with the user.)

## 2. Goals

1. **G1 — capacity.** The shared collection's model carries body-4 terms up to total degree D₄ = 3
   (today 2), stepping up from 2 once the store has the samples to support it (§4.1).
2. **G2 — no ratchet.** A trust recovery that does not end trusted leaves the quarantine as it found
   it. A recovery adds holds only when it restores trust, and never past the held cap (10 %); release (the
   model now reproduces a held sample) works as today. Holds reaching the cap no longer untrust the model
   for good: what is left over becomes local misfit (G3).
3. **G3 — a misfit is local.** When, after recovery, some fitted samples are still beyond the training
   residual bounds, the model is not untrusted as a whole: it answers everywhere else and refuses the
   atoms near the samples it misfits (a new refusal, like today's near-quarantine refusal,
   `TeachRule::NearQuarantine`, ruling S15). The whole model is untrusted by the residual conditions
   only when the misfit is not a small part of the data (§4.3).
4. **G4 — honest about cost.** The refit cost at D₄ = 3 is measured at S = 3 and S = 4 species and
   reported in the README and `references.md`.

## 3. Non-goals and anti-goals

- **Non-goals:** Sandbox itself (M8b; this makes its model usable, nothing more); spin-split
  collections; charged scenes (M9); the per-box H₂ energy offset noted in the diagnosis (≈ +1.6e-4 Ha/atom
  at 81³ vs −3e-4 at 103³ after the per-atom rebase) — recorded, a separate follow-up; the compressed H₃
  geometries that stay held at every D₄ (energy up to 21× the bound) — they stay held.
- **Anti-goals (a failure even if it works):**
  - the model answers where it is wrong: **no rise in missed large errors** (an audit error beyond the
    fit's own in-domain 99th percentile while the verdict said trusted) on the surrogate suite, and the
    stuck-then-escape repro still answered honestly;
  - a tolerance or bound loosened to get trust (CLAUDE.md rule 6) — the 1e-3 Ha/atom and
    max(2e-3, 0.1|F|) ≤ 1e-2 Ha/Bohr bounds stay;
  - anything per element or per molecule (rule 1): nothing may name H₃O, O, a multiplicity or a spin;
  - live play freezing on refits (background refits while playing stay; a paused refit may block).

## 4. Design

### 4.1 Body-4 degree 3 (G1), grown with the store

- The body-4 degree D₄ becomes part of the recorded basis instead of the build constant
  `BODY4_DEGREE_CAP` (which becomes the *ceiling*, 3). Q′ for S = 3 species: 1 032 at D₄ = 2, 1 902 at
  3 (measured); S = 4: 2 688 → ≈ 5 344 (module-doc estimate). The builder recomputes and pins the table.
- **Grown with the store (my call; builder check Q5).** D₄ = 3 roughly doubles the `Samples` floor, the
  per-class floor and the no-model keep cap, so a new collection (a carbon one most) would need about
  twice the DFT before it answers. So each refit picks the highest D₄ ≤ 3 whose `Samples` floor (the
  columns the fitted atoms touch, at that basis) the fitted samples meet; it does not step down again
  unless samples are forgotten or dropped below that floor. A fresh collection starts at 2 and moves to 3
  when it can support it. The chosen D₄ is provider state: it is part of the background refit's
  "something other than new data changed" check, shown in the Model tab's fit line, and written with a
  fitted file.
- The degree-cap-6 recovery basis (step 1 of `recover_trust`) uses D₄ = min(cap, the chosen D₄). It was
  *not* better than the default basis at D₄ = 3 on the user's collection (energy 1.03e-3 on the same
  codes); the recovery order stays (my call; revisit only with a measurement).
- **Old files (my call):** a fitted model file written at D₄ = 2 loads and answers at 2 as today (no
  silent refit, no refusal); the sidecar's existing `body4_cap` field carries the value and the layout
  check compares with the file's own value. Its later refits follow the growth rule. A collection has no
  stored fit; its next refit applies the rule.
- D₄ = 4 is out: measured 46× slower per fit than D₄ = 2 and not better.

### 4.2 No ratchet (G2)

`recover_trust` (quarantine.rs) keeps its steps and order and keeps judging on **today's all-sample
conditions** (`recoverable_fail`: energy and force residuals over every fitted sample, the force noise):
that is how genuinely bad labels are found (my call; builder check Q1). Changes:
- If no step ends trusted, the store's quarantine codes, the degree cap and the collection's persisted
  codes are restored to the entry state before the refit returns (`note_quarantined` /
  `sync_collection_status` write nothing new). The note says so ("recovery did not restore trust; no
  samples held").
- The quarantine rounds stop at the held cap (`QUARANTINE_MAX_FRACTION`, as today) and the recovery
  ends there; whatever still fails becomes M (§4.3) instead of untrusting the model.
- Fence votes (`judge_votes`, bounded by `FENCE_VOTE_LIMIT`) are a separate mechanism and are unchanged
  (my call; builder check Q3): G2 is about the recovery.

### 4.3 Local misfit (G3)

After the recovery, on the fit that stands, let **M** be the fitted samples whose residual ratio —
|ΔE|/N over `energy_std_per_atom`, or the worst trained atom's force residual over its bound, exactly the
quantities of `compute_training_residuals` — exceeds 1. M is recomputed at every refit and never persisted.

- **Trust conditions.** The final verdict evaluates `EnergyResidual` and `ForceResidual` over the fitted
  samples *not in M*, and a new condition `Misfit` fails when |M| exceeds `QUARANTINE_MAX_FRACTION` of
  the fitted samples. The held cap and the misfit limit are **separate counts** (my call; builder check
  Q1: added together, holds at their cap would untrust the model for good again). Its checklist row
  shows |M| / the limit; its refusal names the count and the scenes ("12 training samples the model
  cannot reproduce (H₃O ×12) — refused near them").
- **M stays in the fit** (no peeling: holding them is what made the old recovery chase one trajectory
  band after another). Every sample outside M is within its bounds with M fitted.
- **M is left out of the class rule's statistics** (rule 4b's leave-one-out errors and its aleatoric
  level), exactly as held samples are (my call; builder check Q2). Kept in, M's leave-one-out errors are
  above 1 by construction and would refuse every atom sharing its environment key — H₂O's O and H₃O's O
  share one — so G3 could not work. This is the safety-relevant call of this spec and the physics review
  (§5.7) signs it off specifically: M's region is refused by `NearMisfit`, the class rule judges the
  rest, audits keep calibrating it. Amended 2026-11-13 (Task 7 physics review C1, the user's decision):
  M stays in the class rule's RMS terms and is left out of its max terms, n_k and ā_c — see
  references.md 'Local misfit'.
- **Near a misfit, per atom (my calls; builder check Q4).** A sample's *misfit atoms*: for a force misfit
  the trained atoms whose force ratio exceeds 1; for an energy misfit every trained atom of the sample (an
  energy error has no single atom to blame). A query atom is **near a misfit** when its environment is
  within the near-quarantine distance (`QUARANTINE_FRACTION × max_descriptor_distance` training spreads,
  the same per-atom descriptor distance today's test is built from) of a misfit atom of its species. In
  the DFT solvers a geometry is refused (`TeachRule::NearMisfit`, the teacher answers) when **any** atom
  is near — broader than today's all-atom near-quarantine test, deliberately: it is what Sandbox needs.
  `atom_verdicts` reports `NearMisfit` per atom; Sandbox will freeze exactly those atoms (M8b). Today's
  near-quarantine test and the audit fences are unchanged.
- **Answers near a misfit** take the near-quarantine admission path (my call; builder check Q3): kept
  only when they agree with the model, no fence votes. A region the model cannot fit stays refused
  without a parked scene growing |M|; it is learned when a later refit fits its M samples (a larger
  store, a higher D₄).

**Left to the physics review (with the builder):** whether M's samples should be down-weighted rather
than kept at full weight, and whether the misfit near distance should equal the near-quarantine one; each
justified in `references.md`. Acceptance (§5) is unchanged by them.

## 5. Acceptance

1. **The user's collection** (a copy, `residual_diag`, `#[ignore]`d, `COLLECTION_DIR`): after the
   attach's refit the model is trusted; it holds no more samples than it started with; at least the 68
   releases measured 2026-11-12 recur on the same 181-hold copy; M is listed by scene; a second refit on
   the result gives the same holds and the same M (idempotent).
2. **Same-sample comparison** (recorded in `references.md`): for every scene, the force rms and the
   per-scene leave-one-out force RMS at D₄ = 3 are ≤ D₄ = 2's, or both stay below 1e-4 Ha/Bohr (0.05 of
   the 2e-3 force floor, where a different hyperparameter optimum can move a near-exact scene like N₂ at
   1.4e-5) (my call; builder check Q7). Any other rise is investigated, never accepted.
3. **No ratchet** (unit test, providers): a store whose recovery cannot restore trust ends the refit with
   the codes it started with, in the store and in the collection's file.
4. **Local misfit** (unit tests, providers): a store with a small set of unrepresentable samples is
   trusted, refuses atoms near them with `NearMisfit` and answers far geometries; with misfits beyond the
   limit it is untrusted with `Misfit`; `atom_verdicts` marks exactly the near atoms; the class rule's
   statistics exclude M. The D₄ growth rule: a store below the D₄ = 3 floor fits at 2, one above it at 3.
5. **Safety (my call; builder check Q6):** `cargo test -p validate --release --test surrogate` green
   (per-scene bases: it exercises G2 and G3, not body-4); the `#[ignore]`d `trust_baseline_report` at
   both cadences shows no missed large errors added against today's numbers on the same machine;
   `providers` and the CLAUDE.md "any change" suites green; pinned Q′ values updated to the new layout (an
   update of a pinned count, not a loosened tolerance). Collection-mode (body-4) safety: acceptance 1 plus
   the `providers` collection tests; the hours-long `--test transfer` is not required (it refused every
   atom last time and would not measure this).
6. **Cost** (`refit_cost`, recorded): one fit and one attach refit at D₄ = 3 for the user's S = 3
   collection and a synthetic S = 4 store of comparable size, against D₄ = 2. **Stop condition:** if an
   S = 4 refit at 10 000 samples exceeds 5 minutes on the 9800X3D, stop and bring the numbers to the user
   before merging (options then: D₄ = 3 only for many-body centres, or a smaller n_max in the body-4
   block).
   *Fired 2026-11-13:* the synthetic S = 4 store (10 830 samples) took 1 087 s for its cold attach refit
   at D₄ = 3 (2 fits, ≈ 540 s each, Q′ 5 344); D₄ = 2 took 437 s there (8 fits, the failed recovery
   rerunning). **The user's decision: accept the cost** — D₄ = 3 for every species set, no column
   budget; while playing the refit runs in the background (fix 4), paused it is a wait. Numbers in
   `references.md` "Refit cost at body-4 degree 3 (2026-11-13)".
7. **Physics review** (physics-reviewer agent) of §4.2–4.3 before they land, the class-statistics
   exclusion of M first: that `NearMisfit` plus the class and audit rules leave no path to answering a
   misfit region as trusted.

## 6. Constraints

- CLAUDE.md rules 1–8; the trust rules are safety rules — every change here is visible in the Model
  tab's checklist and the Activity tab, none is silent.
- One heavy job at a time; never build into a target dir whose `app.exe` is running.
- README updated in the merge (Q′ numbers, refit timings, the new refusal and trust row).

## 7. Order

1. Ratchet stop (§4.2) — small, independent, safe on its own.
2. Body-4 degree recorded per basis and grown with the store (§4.1), cost measurement (§5.6, the stop condition).
3. Local misfit (§4.3) with the physics review.
4. Acceptance on the user's collection copy, the suites, README / DEVLOG / dev-reference, merge.

## 8. Left to the builder

Task split and model tags (Opus throughout, per the user's standing instruction); the exact Q′ table;
the refusal and checklist wording beyond the example; whether the per-atom near test needs a new
cache; the synthetic S = 4 store's construction for the cost measurement.
