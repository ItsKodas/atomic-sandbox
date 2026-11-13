# Leave-one-out recovery and per-sample forget (2026-11-12)

User-approved 2026-11-12 ("do both 1 and 2"), after the collection debug (`.superpowers/sdd/collection-debug-report.md` in the docking worktree, summarised here).

## Why

The user's shared collection went untrusted. The cause was one sample: #1804, a lone OH radical squeezed to O–H 1.31 Bohr, with a force of 0.72 Ha/Bohr.

- **The sample stays inside its own bound but bends the fit.** Its own force bound is the 1e-2 cap, and the fit reproduces it within that. But it bends the shared O–H pair term, so 34 H₂O/H₃O samples fail: the worst force ratio goes from 1.00 to 2.36.
- **Holding it alone fixes the fit.** With #1804 held out, the fit is trusted again (ratio 1.00).
- **The recovery can't see it.** Its offender ranking, `worst_offenders()`, ranks each sample by its own training residual against its own bound. It never names #1804.
  - The recovery quarantines the victims instead, round after round. That makes their neighbours worse.
  - It runs out of rounds with #1382 at 3.69e-3 > 2.53e-3 Ha/Bohr.
- **The leave-one-out residual sees it at once.** This is the residual of the fit without the sample, and it is already computed for every sample at every fit by the class rule. It puts #1804 at rank 1 of 1,700, at 38.5× its bound; the next sample is at 5.2×.
- **The user has no way to drop one bad sample.** "Forget" in the Model tab drops the whole collection.

## R1. Recovery ranks offenders by their leave-one-out residual

**What changes.** Step 2 of `recover_trust` (`quarantine_rounds`) chooses each round's samples by their **leave-one-out ratio**, in place of the training-residual ratio.

**The ratio.** For one sample it is the larger of two things:
- its leave-one-out energy residual per atom ÷ `energy_std_per_atom`, when its energy was fitted;
- the worst, over its fitted atoms, of `force_residual(F − r, F)`'s `d / bound`. Here `r` is the atom's leave-one-out force residual.

This is the same reading the class rule uses in `trust.rs`.

**The bounds.** The force bound is the one `worst_offenders` uses today:
- the per-atom bound while the training residuals fail;
- `force_std_abs` when only the force noise fails.

**Which samples a round takes.** The same rule as now, applied to the new ratio: every sample (not excluded, not box) with ratio > 1 and ratio ≥ ½ × the worst ratio.

**Unchanged:**
- `QUARANTINE_ROUNDS` (5) and `QUARANTINE_MAX_FRACTION` (10 %);
- the degree-cap steps 1, 1b and 3;
- release and supersede;
- the trust rule that judges every fit.

No tolerance moves.

**Ruling R1a: an undefined leave-one-out residual ranks first (ratio = ∞).** It is undefined (`leave_one_out` returns `None`) when the sample's block is singular, that is, the sample alone spans a direction of the fit.
- Why: this matches the class rule, which already reads `None` as ∞ (`trust.rs`, `a.3 = f64::INFINITY`). A sample the fit cannot do without is the highest-leverage sample there is.
- What it costs if wrong: such samples are quarantined first. If several are, the 10 % cap and the trust rule still bound the damage, and the trust rule's sample counts still judge the result.

**Ruling R1b: reuse the class pass's leave-one-out.** The class pass computes `leave_one_out_many` at every fit. The provider keeps the per-sample ratios from that pass, keyed to the fit that produced them, so a recovery round costs no extra leave-one-out.
- If no fresh cache exists (a fit path that skips the class pass), the round computes the ratios itself with `leave_one_out_many`.

**The recovery note says what was done.** It says the samples were held because of their leave-one-out residual and names the worst ratio. For example: "3 samples quarantined — the fit without each predicts it at up to 38× its bound, and keeping them breaks the fit of their neighbours (…; kept, excluded from the fit; geometries near them fall back)".
- It no longer says "the model cannot represent them".

**Pinned behaviour:**
- **A unit test with a high-leverage outlier.** The training-residual ranking would have held victims; the new ranking holds the outlier alone and restores trust.
- **An `#[ignore]`d measurement on a copy of the user's pre-1805 snapshot.** Path from an environment variable; never the user's folder. Expected:
  - #1804 is held in round 1;
  - the fit is trusted;
  - fewer than 110 samples are held.

  Record the numbers in `references.md`.
- **Existing recovery tests.** A test whose pinned held set or count changes is investigated, not re-pinned blindly:
  - re-pin only if the new result is still trusted and the new held set is explained;
  - otherwise it is a finding.

## R2. Forget and restore one sample from the Model tab

**A new persisted code, `Q_FORGOTTEN = 7`, in `surrogate::quarantine`.** The user forgot this sample.
- It is written as a collection status record, and to the sidecar's `sample_quarantine` array for a model file.
- 5 and 6 stay the provider's internal codes. Code 7 is new; older builds that read it must fail with a clear "newer build" error, or treat it as excluded — the implementer checks which happens and documents it.

**How the provider treats a forgotten sample:**
- excluded from the fit, the envelope, the grade and the leave-one-out, like a held one;
- **never released** by `release_reproduced` and never superseded. Only the user's Restore clears it.
- **it fences** like a held sample, so geometries near it fall back to DFT. This is the safe side: the user said the model should not learn from it.
- **it counts against the 10 % recovery cap**, together with the held samples. Forgetting is therefore no way round the cap.
- it is counted on its own: `forgotten()`. The Model tab shows "N forgotten" beside the held count.

**What the user sees:** a collapsed **"Worst-fitting samples"** section in the Model tab.
- **The list.** Up to 8 samples, ranked by the leave-one-out ratio from R1b.
  - Each row: `#1804 · HO · fallback · 38× its bound`, with a **Forget** button. The formula comes from Z, the origin from the sample tag.
  - Below the list: "Ranked by how badly the fit without each sample predicts it."
- **Forgotten samples** get their own list, each with a **Restore** button.
- **What a click does.** Forget or Restore writes the status record at once (to the shared collection when one is attached, otherwise into the model's store) and starts a refit.

**Plumbing:**
- The buttons send `Action::ForgetSample(usize)` and `Action::RestoreSample(usize)` through `ui::Action`, so `--record` sees them.
- The worker reaches the provider through `ForceProvider::as_trainable`, with new `Trainable` methods. No downcasts.
- With no model, or no ranking yet, the section says why.

**Pinned behaviour:**
- **Forget:**
  - a forgotten sample is not fitted;
  - it is not released after a refit that reproduces it;
  - it survives a reopen of the collection and a save/load of a model file.
- **Restore:** a restored sample is fitted again.
- **The cap:** the cap counts forgotten samples.
- **The panel:** a UI test of the row text, and that clicking the buttons emits the Actions.

## Not in scope

- Cheaper intermediate recovery fits; re-measure first now that the refit speed-ups are on main.
- Persisting the recovery's `degree_cap`.
- The margin display.
- The re-solve duplicate check.

## Amendments (2026-11-12/13)

Rulings made during implementation and review (the ledger `.superpowers/sdd/2026-11-12-sample-recovery/progress.md` holds each with its reasoning). Where they differ from the text above, they win.

- **R1, option A.** The leave-one-out rounds run first. When they do not recover the fit (residuals, noise and the trust rule's counts), the state the rounds started from is restored and the old training-residual rounds run from it. Their result stands only if it recovers; otherwise the leave-one-out result stands. The recovery is never worse than before R1, at up to 5 extra fits on a refit that is failing anyway. Options C and D were rejected (C changes release, outside R1; D changes the path state).
- **Step 3 runs whenever the result is not recovered.** The default-basis retry is gated on "not recovered" (residuals, noise **or** the counts), not only on failing residuals or noise.
- **The pre-1805 expectation was wrong.** "Fewer than 110 held, trusted" (Pinned behaviour above) came from a debug number that had no release step. The measured result: the first refit ends untrusted at 1.0185× with 128 held (the fallback 165 held at 1.46×), the next refit trusted at 0.9950× with 122 held. The numbers are pinned honestly in `crates/validate/references.md` "Leave-one-out recovery ranking".
- **`SampleId` replaces the bare `usize`.** `Action::ForgetSample` / `RestoreSample` and the `Trainable` methods carry `sim::SampleId`: the store index plus a check value (`sample_key`, FNV-1a over Z and the bits of the positions, energy and forces). An id whose index now holds another sample is refused ("the sample list changed — pick it again").
- **Forget only a code-0 sample.** A held, superseded, box-excluded, draft, waiting or lone sample is refused by name, so Restore (back to code 0) always restores exactly what it was.
- **The final note wording.** The recovery note counts this refit's holds ("K samples quarantined this refit (N held in all) — …"), names the worst leave-one-out ratio held against each one's own bound (or "the 2e-3 Ha/Bohr force-noise tier" in the noise-only mode), and ends ", and the fit without them recovers" (residuals, noise and the counts pass) or ", and the fit without them passes" (residuals and noise pass, short of the counts). When the fallback stands it says "ranked by their own training residual (the leave-one-out ranking did not recover the fit)". The old clause "and keeping them breaks the fit of their neighbours" overclaimed and is gone. Notes next to a forgotten sample say "forgotten", not "quarantined".
- **Rule 8 holds.** The failed-audit drop list excludes code 7: forgetting the sample of a failed audit does not lift the global untrust; the audit is re-judged at every refit as before.
- **Holds made by a failed recovery persist** (and are written to a collection). *Superseded at the merge with the open-shell model (2026-11-14; its spec §4.2, G2):* a recovery whose residuals still fail after every step now ends with the codes, degree cap and fit it entered with (code 7 untouched); holds stay only when the result recovers or passes its residuals and noise. Inside `quarantine_rounds`, when neither ranking recovers, a leave-one-out result that fails the residuals yields to a training-ranking result that passes them. A held sample leaves the hold only when a later fit reproduces it (released) or a fresh agreeing answer next to it supersedes it, which ends its fence but keeps it out of the fit. Restore does not apply to a held sample.
- **Answers next to a forgotten sample are thinned** (final review I1). A forgotten sample's fence is permanent, so an agreeing teacher answer fenced by forgotten samples alone goes through the normal thinning and copy checks instead of being kept at every step.
- **Not in this branch:** `box_blocks_trust` subtracting a forgotten box-affected sample twice (a diagnostic line only, below the trust floor) is a follow-up.
