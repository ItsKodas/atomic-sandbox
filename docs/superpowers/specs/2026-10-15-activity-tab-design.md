# Activity tab: processing feedback, live solve log, training views — Design Spec

**Date:** 2026-10-15
**Status:** Specified from the user's requests and the controller's rulings (below). Decisions this spec makes on its own are marked **[provisional, A*n*]** and listed in §11 with a proposed ruling each. Amended by the plan's physics review (2026-10-15, "ready after amendments"): §12 lists each change and its reason.
**Author:** Claude (Opus 5.5) with Dakoda
**Branch:** `activity-tab`, off `main` at `dc03aaa` (M6 merged). Lands **before the user-data milestone** (controller ruling in the request note).
**Parent:** `docs/superpowers/specs/2026-10-12-atomic-sandbox-design.md` (§5 architecture, §8 tooling); `docs/superpowers/specs/2026-10-13-m5-surrogate-provider-design.md` (the fallback policy, trust rule, audits, quarantine); `docs/superpowers/specs/2026-10-14-m6-tools-design.md` (the `providers` crate, `Trainable`, the worker split).
**Inputs (binding):** `docs/superpowers/notes/2026-10-13-m5-user-requests.md`:
- "Activity view: obvious processing feedback with something to watch (user request 2026-10-15)", with its two follow-ups: "more live details on the training", and the request for a live log of solves with counters that tick per solve (samples in the model, teacher solves ever spent on it, solves this session);
- "Three hydrogens will not reach trust": the refusal must name the unmet condition in plain words. M6's final fix wave added the box-trust message (`app::worker::model::box_blocks_trust`, `ui::box_advice`); this spec builds on it.
**UI conventions:** `docs/superpowers/notes/2026-10-14-ui-mockup.html` and `docs/superpowers/notes/2026-10-14-ui-redesign-checklist.md`: one `Action` layer, dock tabs with a strip icon and hover peek, search anchors `d-*`, hand-painted plots (`ui::dock::graph::sparkline`: `Painter` polylines, no plot crate), amber for "asks for something", tooltips that say where a number comes from.
**Later milestones this feeds:** `docs/superpowers/specs/2026-10-15-user-data-design.md` (the Model tab's information §12, sidecar v4 §7) and `docs/superpowers/specs/2026-10-15-precision-design.md` (the job queue §6, the Models & jobs page §8, ruling P7).

## 1. Purpose

The user cannot tell what the program is doing while it works, and cannot see the model learn. Today a solve shows "solving…" with a spinner. A pass shows "sampling the teacher: scans 41/273". A refit, a background audit or a bond-reference solve is invisible or one dim line. Why the model refuses ("max training energy residual 2.2e-3 > 1.0e-3 Ha/atom", "atom 0 is 4.1 training spreads outside the descriptor envelope (feature b3[n=0;n=2;l=0], limit 3)") is engineering text.

This milestone adds:

1. **Processing feedback** (§3.1–§3.2): the status chip names what runs and how far it is ("Solving · SCF step 7 (last solve 11)", "Sampling 41/273 · bond scan · 1:40 left", "Refitting · 250 samples", "Audit running · SCF step 4"); a thin progress bar on the toolbar while anything runs; all display updates at ≤ 10 Hz, with their cost measured and reported (§6).
2. **An Activity dock tab** (§3.3): the job list; the three live counters; the live solve log (one row per teacher solve, newest first); a live trust checklist in plain words; rmse per refit epoch; the parity plot; per species-pair model curves with a ±σ band and the DFT samples as dots; coverage histograms; an event timeline. Everything is read from records already in memory, plus the minimal additions of §4.3. **No view causes a teacher solve.**
3. **Plain wording** (§5) for every reason the model does not answer, with the technical text kept on hover.

What success looks like: the user picks Estimated on three hydrogens and opens Activity. Rows appear as each pass solve lands ("0:01:12 · pass · bond scan · kept · 140 ms"), and the counters tick. The progress bar fills. When the pass ends the rmse plot gains a point, the parity plot fills in, and the trust checklist shows one red line: "Samples clear of the box walls — 130 in the fit touch them (145 held). Enlarge the cell (Scene ▸ Grid: 123³ at 0.15 Bohr) and sample again." Hovering it shows the condition the fit actually failed ("max training energy residual 2.2e-3 > 1.0e-3 Ha/atom").

## 2. Non-goals

- **The model manager** (precision §8, user-data §11) and **the job queue** (precision §6): background jobs, reordering, persistence across restarts. The job list here shows the work of the live worker only.
- **Precision renders** and anything else from the precision milestone.
- **Persisting the solve log.** The log is per session (a ring, §3.3.3). Only the lifetime solve counter persists (§4.3, A2).
- **New physics or policy.** No rule of `providers::surrogate` changes. The additions in §4.3 record what the rules already decide. The surrogate suites must pass unchanged (§8).
- **Ghost atoms of the geometry being solved and a live SCF-convergence chart** (from the original 2026-10-15 ruling). Both are deferred **[provisional, A3]**. The chart needs the per-step Δρ out of `qm_gpu::GpuSolver::solve_with`, which is a solver change under rule 6. The ghost atoms need the pass geometry and a GPU-to-GPU density copy per step. This milestone shows the SCF step count (§3.1), which needs no solver change.
- **Model tab redesign.** The Model tab keeps its content. Its "samples" row is already live: `ModelStatus::samples` is read per snapshot. Two lines move to plain wording (§5).

**One component, not three.** The job list (§7) is the seed of the precision milestone's job queue and of the user-data model manager. Its row type and view are written so that both later milestones add rows to this list; they do not build their own.

## 3. What the user sees

### 3.1 The status chip (top-left of the 3D view)

The chip keeps its layout (state · who answered · t · step · T · thermostat; error lines; notes). Its **state text** becomes the activity headline whenever work runs. Idle states keep their old text verbatim (row 2 of the checklist: "all states kept"). A spinner is shown while work runs, as today.

| What runs (first match wins) | Headline | Source (§4) |
|---|---|---|
| a sampling pass, references stage | `Sampling 3/273 · reference H · SCF step 9` | `DynSnapshot::sampling`, probe main slot |
| a sampling pass | `Sampling 41/273 · bond scan · 1:40 left` (`· SCF step 7` while the probe reports a step) | `sampling`, `ModelStatus::pass_eta_s`, probe |
| a refit | `Refitting · 250 samples` (`· 0.4 s` after 0.3 s) | probe refit phase |
| a DFT step of the plain DFT backends | `Solving · SCF step 7 (last solve 11)` | probe main slot, `DynSnapshot::eval.scf_steps` |
| a teacher step in estimated mode | `DFT fallback · SCF step 7 (last solve 11)` | probe main slot, backend |
| a CPU solve (no step count) | `Solving (CPU reference) · 12 s` | probe main slot (`scf_step` stays 0) |
| a relax | `Relaxing (FIRE) · step 31` | `DynSnapshot::relax` |
| playing (model answers, no solve in flight) | `playing` (unchanged) | |
| background audit in flight | appended to any of the above as a second pill: `Audit running · SCF step 4` | `ModelStatus::audit_in_flight`, probe side slot |
| bond references | unchanged dim note line (`bond references: O 2/3… (14 s)`) | `DynSnapshot::bond_reference` |
| none of the above | the old idle texts (`paused — converged`, …) | `ui::overlays::state` |

"SCF step k" is the step being solved. The total is unknown before convergence, so the chip shows the last solve's count beside it, never a fake "/N". The texts are rebuilt at most every 100 ms (§6). Without a background auditor (the CPU teacher) an audit runs in the step and reads "DFT fallback" while it runs. The probe cannot tell the two apart mid-solve; the log row names it once it lands (plan review N10, accepted).

### 3.2 The progress bar

A 3 px bar along the **top edge of the bottom toolbar** (the seam between the 3D view and the toolbar) **[provisional, A1]**, drawn while anything in §3.1 runs:

- **determinate** (accent colour, filled to done/total) for a sampling pass and for bond references;
- **indeterminate** (a 20 %-wide segment sweeping left to right once per 1.2 s) for single solves, refits and relaxes;
- an amber tail segment when a background audit runs alongside;
- hidden when idle. Hovering it shows the headline and "Activity tab for details"; clicking it opens the Activity tab.

### 3.3 The Activity dock tab

A sixth dock tab, **Activity** (strip icon: a pulse line; hover peek: the headline, the three counters, the newest log row). The strip icon carries a small animated dot while work runs. The tab is available on every backend. On the DFT backends it shows the job list and the "teaching solves this session" counter; plain DFT steps are not teaching solves, so they add no log rows and do not tick it (the log keeps the session's earlier rows, and its empty text says why). The trust and training sections are shown only for the estimated solver.

Layout, top to bottom. Search anchors in brackets.

```
Activity                                                         [⏸ views]
JOBS                                                           [d-jobs]
 ● Sampling pass  — bond scan · 41/273 ███████░░░░░ 15 % · 1:40 left  [Cancel]
 ● DFT fallback   — SCF step 7 (last solve 11) · 0.4 s   ░▒▓▒░        [Pause]
 ● Audit          — background teacher · SCF step 4                  (can't be cancelled)
COUNTERS                                                       [d-counters]
 samples in the model 412   teacher solves for this model 431 (pass 343 · fallback 70 · audit 18)
 teaching solves this session 96
SOLVE LOG  [pass] [fallback] [audit]  [kept] [not kept]          [d-solve-log]
 time     step   origin    rule              outcome          |ΔE|     solve
 0:04:12  1607   fallback  5e grade          kept             3.1 mHa  92 ms
 0:04:09  1640   audit     8 audit (passed)  kept             0.02 mHa 140 ms
 0:01:12  pass   pass      bond scan         kept (box)        —       131 ms
 …
TRUST                                                          [d-trust]
 ✗ Samples clear of the box walls         130 in the fit touch them (145 held) — enlarge the cell (…)
 ✓ Enough samples in the fit              412 of 40 needed
 ✓ Enough data beyond the fit's own parameters    energies 396 of 30 / forces 2461 of 30
 ✗ Reproduces its own training energies   worst 2.2 mHa/atom, allowed 1 mHa/atom
 ✓ Reproduces its own training forces     …
 ✓ Its forces fit the data to within the force tier on average   σ_f 0.045, allowed 2 mHa/Bohr
 ✓ Last check against DFT — no failed check since the last refit (last check: passed)
 per environment: pair ✓ learned · three or more atoms ✗ not yet (12 < 40 samples)
TRAINING                                                       [d-training]
 rmse per refit       [plot: E (mHa/atom) and F (mHa/Bohr), log y, x = epoch]
 model vs DFT         [plot: parity, E | F toggle, colour = origin, hollow = excluded]
 H–H curve            [plot: E(r) − 2E_ref, ±σ band, dots; dashed and labelled where it would fall back]
                      [lane: σ_epi/N ÷ tier and rule 6's ratio, log y, refuses above 1]
 coverage             [histograms: pair distances per species pair; burst temperatures]
TIMELINE                                                       [d-timeline]
 [one row per kind: fallbacks · audits · refits · quarantines · passes, x = session time]
```

#### 3.3.1 Jobs

One row per running or waiting piece of work of the live worker (§7). Each row has a title, a detail, progress (bar or sweep), elapsed time, ETA when known, and the one control that exists today: **Cancel** (sampling pass → `Action::CancelSampling`), **Stop** (relax → `Action::StopRelax`), **Pause** (a playing run's solve → `Action::Pause`, which aborts the solve in flight per ruling R11), **Start anyway** (the first-pass offer → `Action::StartOffered`). Rows that cannot be cancelled say so on hover, with the reason (a background audit finishes its solve; a CPU solve has no cancellation point).

#### 3.3.2 Counters

Three numbers, ticking per solve:

- **samples in the model**: `ModelStatus::samples` (the store's length, the samples held; the Model tab shows the same live number). The hover says "held; F in the current fit", since the trust check counts the fit's samples (plan review S9);
- **teacher solves for this model**: the provider's lifetime `SolveCounts::total()` (pass + fallback + audit + verification, kept or not), with the split on the line and on hover. It counts completed or failed teaching solves, including the background auditor's failed and discarded ones; a cancelled solve is not counted, and the hover says so (plan review S11). It survives save/load and the session autosave through the optional sidecar key of §4.3 (A2). A model loaded from a file without that key shows `≥ 412 (earlier solves unrecorded)`: the held samples are a lower bound. That caveat is itself persisted (`"complete": false`), so it survives the next autosave (plan review M4);
- **teaching solves this session**: teaching solves that reached the UI since the viewer started, summed over workers and backends (a backend switch does not reset it). Plain DFT steps are not teaching solves and are not counted (plan review S16).

#### 3.3.3 The solve log

Newest first. One row per **teaching solve**: a teacher evaluation whose answer was offered to the model (sampling pass solves including reference multiplicities, fallback steps, audits in the step or in the background, relax verifications). Display refreshes, isolated-atom densities and bond-reference solves are jobs, not log rows **[provisional, A8]**. Two kinds of non-solve rows also appear, because the user named them as rules: a pass geometry the cell guard skipped and the surrogate's rule-0 cell refusal (outcome "not solved", solve time "—", not counted).

Columns:

| column | content | hover |
|---|---|---|
| time | session clock `h:mm:ss` (as the Log tab) | wall time of the event |
| step | the driver step whose evaluation produced the event, stamped by `sim::Dynamics` right after that `evaluate`, so a tick of n steps gives n stamps; `pass` for pass solves. A background audit shows the step it was judged at (plan review M6) | an audit: "judged k answered steps after its dispatch" (`AuditRecord::unaudited_steps` counts answered steps only, so no dispatch step is computed from it) |
| origin | `pass` / `fallback` / `audit` / `check` (relax verification), colour-coded. A relax verification refused by rules 1–4 is logged as `fallback` with that rule (plan review N2) | |
| rule | rule code and short word, as the user wrote them: `0 cell`, `1 no model`, `2 new element H`, `3 other settings`, `4 untrusted`, `4b class`, `5a lone atom`, `5b coordination`, `5c envelope`, `5d new environment` (the null-space novelty), `5e new environment` (an atom along an unspanned direction, γ = ∞; plan review S7), `5e grade`, `6 force σ`, `7 energy σ`, `fence`, `8 audit (passed/failed/box)`, `relax check`; pass rows `reference H m2`, `nudge`, `bond scan`, `hot burst 1830 K` | the plain sentence of §5, then the technical text: every note the provider wrote for that solve (plan review S12) |
| outcome | `kept`, `kept (box)`, `kept by vote`, `kept, released 2`, `skipped: similar`, `not kept: box`, `not kept: fence`, `not kept: waiting for a pass`, `not kept: other settings`, `discarded`, `refused by store`, `reference`, `failed`, `not solved` | what the outcome means (§5.2); "the answer is box-affected" when it is and the outcome does not already say so (a vote can keep one; plan review N1) |
| \|ΔE\| | \|E_teacher − E_model\| in mHa where the model was consulted (rules 4b–8, fence, verification), else `—` | per atom, and the signed value in Ha |
| solve | the teacher's wall time (ms, or s above 10 s) | SCF steps |

Filters: origin chips and kept / not-kept chips. The UI keeps the newest **5 000** rows. Older rows are dropped from view; the counters and the burst-temperature histogram keep counting. A drag held against the cell margin adds one rule-0 row per refused evaluation; they are bounded and not counted (plan review N3). The list is drawn with `ScrollArea::show_rows` (only visible rows are laid out).

#### 3.3.4 The trust checklist

The trust rule's conditions in the order `untrusted_reason` tests them, each with ✓ / ✗, its number against its limit, and the plain line of §5.3. The list and the refusal come from one predicate per condition: the checklist is `met = !(predicate)`, and the refusal is its first unmet entry, so the two agree by construction, at equality and on NaN (plan review M2). The checklist is followed by the class line (per environment class, worded from `ClassLoo::refusal`, the class rule's own verdict; plan review M3) and the box line (`box_blocks_trust`, with `box_advice`'s cell suggestion). The **first ✗ is the reason the model is refused**. It is marked "this is why", and its plain line also replaces today's "trusted: no — n_samples 12 < q_prime 16" in the Model tab and the toolbar tooltip (§5). When the box line is ✗ it is listed first and marked "this is why", because that is what blocks trust (M6's box ruling: whatever condition the fit failed, the box is in the fit it judged). The box line is shown only when the first unmet check is a condition of that fit (training energies, training forces, force noise). It is not shown for a count, which more samples cure, or for a failed audit, which is judged against a fresh teacher answer that is not box-affected (plan review S15).

#### 3.3.5 Training views (estimated solver)

Collapsible; collapsed by default. Their data is built by the worker **only while the Training section is open** (§6). A small "built 0.6 ms ago at epoch 14" line under each plot shows its age.

- **rmse per refit epoch.** Two series, `rmse_energy_per_atom` (mHa/atom) and `rmse_force` (mHa/Bohr), log y, x = refit index of this session (with epoch on hover), points trusted at that refit filled, untrusted hollow (a later failed audit does not change a point; the caption says "trusted at the refit", plan review N9). Source: the fit tuple of each new epoch (§4.2). A loaded model starts the series with the file's fit.
- **Parity (model vs DFT).** Energy mode: x = DFT, y = model, both as (E − Σ E_ref)/N in mHa/atom (the atomization energy per atom, so the axes are not dominated by the reference energies) **[provisional, A4; approved by the review]**. The references are the fit's own (`Model::ref_energy`), which is what the fit subtracted (plan review S4). Force mode: every force component of the fitted samples, Ha/Bohr shown in mHa/Bohr, deterministically thinned to ≤ 20 000 points **by whole atoms**, so x, y and z stay equally represented (plan review S5; the cap is stated). The y = x line, and the energy tier ± `energy_std_per_atom` drawn as a band around it. Colour = origin from the store tag (pass / fallback / live). Audits are tagged `Fallback` in the store until user-data's `SampleTag::Audit` **[provisional, A6]**. Filled = in the fit, hollow = quarantined, superseded, box-excluded or newer than the fit; hover names the sample index, its state and both values. A sample whose prediction fails is omitted and counted in the caption (plan review S18).
- **Pair curves.** One plot per species pair the basis knows (H–H; H–He, He–He; …). Curve: the model's energy for an isolated dimer of that pair at distance r, E_model(r) − E_ref(a) − E_ref(b), on 96 points from 0.5 Bohr to r_cut. Band: ±σ_epi(r), the **epistemic** energy std √(energy_std² − (2 s_E)²), the quantity rule 7 compares (§9.2) **[provisional, A5; approved by the review]**. **Where the model would not answer such a dimer**, read from `would_answer` (rules 4–7 and the fence, read-only, §4.3 A6) and never from σ, the curve is dashed, its band dimmed, and each stretch labelled with the rule that fires. That is the dominant encoding (plan review M5). Under the plot, a log-y **uncertainty lane** shows σ_epi/N ÷ `energy_std_per_atom` (rule 7) and rule 6's ratio max_j σ_F,epi/(√3·bound_j), both against 1, with the fit's residual scatter s_E (teacher noise plus what this basis cannot fit; more samples of the same kind do not remove it) ÷ tier. Rule 6 usually fires before rule 7, and a band 10⁻⁶ Ha wide is invisible against a 0.2 Ha curve, so the comparison needs its own scale. The y range comes from the answered points and the dots, so a refused extrapolation toward 0.5 Bohr does not set it. Dots: the stored two-atom samples of that pair, (r, E − E_ref(a) − E_ref(b)), filled/hollow as in the parity plot. Rug: tick marks at every pair distance < r_cut of that pair in the fitted samples (any atom count). The caption says what it is (plan review S3): "the model's energy of an isolated A–B pair; not a pair potential: in a molecule the model has no per-pair energy. The band, the lane and the dashed stretches are for the isolated pair". It also says what the verdict leaves out (rule 0, audits, the reuse of an accepting uncertainty check), that inside a molecule the model may answer this pair where the curve is dashed and refuse it where it is solid, that in a scene at other settings than the model's it answers nothing (rule 3), and that the band is not an error bar against DFT (Task 4 review I1). In draft mode it adds: "dashed: where it would fall back outside draft mode".
- **Coverage.** Per species pair: a histogram of pair distances < r_cut over the fitted samples (0.1 Bohr bins; the caption says "in the current fit", plan review N12). Burst temperatures: a histogram of the instantaneous kinetic temperature of each hot-burst geometry solved **this session** (100 K bins), with the plan's target T marked. It is kept in its own per-session accumulator, not read from the 5 000-row log ring (plan review S6). The temperature is the Verlet velocities' at the solved geometry, before the thermostat rescales them. The caption says that consecutive points of a burst are correlated (τ = 5 fs at dt = 0.25 fs), and that a loaded model's earlier bursts are unrecorded.

#### 3.3.6 The event timeline

A strip, x = session time, one lane per kind: fallbacks (amber ticks, density-binned when > 1 per pixel), audits (green passed / red failed / grey box), refits (blue diamonds; untrusted hollow), quarantines (purple, with the count), passes (a bar from start to end, cancelled hatched). Hover shows the event's row text. Source: the solve-log events, the fit history and the `quarantined` count deltas (§4.2).

## 4. Data flow

### 4.1 The records that exist today (M6, `dc03aaa`)

| record | where | what it holds | reaches the UI? |
|---|---|---|---|
| `DynSnapshot::sampling: Option<SamplingProgress>` | `sim::sampling` | pass stage, done/total, skipped, failed, last energy | yes, per snapshot |
| `ModelStatus::pass_eta_s`, `solve_s` | `app::worker::model` | pass ETA from its measured rate | yes |
| `DynSnapshot::eval: EvalStats` | `sim::provider` | the last step's `source`, `scf_steps`, `solve_ms`, `face_density`, `teacher_gap`, `notes` | yes, last step only |
| `DynSnapshot::bond_reference`, `relax` | `sim::dynamics` | progress of those jobs | yes |
| `TrainableStatus` → `ModelStatus` | `sim::provider`, `app::worker::model` | trusted, `untrusted_reason` (a string), samples, epoch, `fit` (n, Q′, rmse E, rmse F), `residuals` (E/atom, F, bound), quarantined / superseded / box_excluded / box_affected / unfitted, trust floor, `last_audit: AuditRecord`, `audit_in_flight`, `stuck` (a string), `class_loo` | yes, per snapshot |
| `SampleStore::samples` (pos, z, energy, forces, tag, weight) | `surrogate::fit` | every kept teacher answer; tags LiveStep / Perturbation / HotBurst / Scan / Fallback / Reference | no (provider memory) |
| per-sample quarantine codes (`Q_NONE`/`Q_HELD`/`Q_SUPERSEDED`/`Q_BOX`), face densities | `SurrogateKs::{quarantine, face}` | why a stored sample is out of the fit | counts only |
| `FitReport` | `surrogate::fit` | rmse, s_E, σ_f, λ, `dof_eff`, `residual_dof_e/f`, Q′ | rmse and Q′ only |
| `FallbackPolicy` | `providers::surrogate::policy` | tiers: `energy_std_per_atom` 1e-3, `force_std_abs` 2e-3, … | `force_bound` only |
| `Model::predict(pos, z, with_variance)` | `surrogate::model` | energy, forces, `energy_std` (with the (N s_E)² floor), `force_std_epistemic` | — |
| the reasons a step went to the teacher | `Gate::reason`, `envelope_outside`'s reason, `VarianceVerdict::reason`, `class_untrusted`'s reason | **strings only** | as `EvalStats::notes` of the last step |
| whether an answer was kept | `SurrogateKs::admit` → `bool`, plus a note | **a bool and a string** | as notes |
| burst trajectory | `providers::sampling::Burst::scene` (with velocities) | the burst geometry's velocities | no |

### 4.2 Which records feed each view

| view | fed by | exists? |
|---|---|---|
| chip headline, progress bar, job rows | `sampling`, `pass_eta_s`, `relax`, `bond_reference`, `audit_in_flight`, `eval.scf_steps` (last solve's count); **live SCF step, "a solve is running", refit phase** | partly; **live state is A1 (probe)** |
| counter: samples in the model | `ModelStatus::samples` | yes |
| counter: teacher solves for this model | **`SolveCounts`** (A2) | new |
| counter: teaching solves this session | sum of A3 solve events received (app) | derived |
| solve log rows | **`TeachEvent`** per teaching solve (A3), stamped by `sim::Dynamics` with the step whose evaluation produced it; audit lag (answered steps) from `AuditRecord::unaudited_steps` | new; its contents exist inside the provider at the moment of the solve |
| trust checklist | `FitReport::{n_samples, q_prime, residual_dof_e, residual_dof_f, sigma_f}`, `TrainingResiduals`, policy tiers, `audit_failed`, `audit_force` and the audited atom count → **`TrustCheck` list** (A4); `class_loo` with its structured verdict; `last_audit`; `box_blocks_trust` + `box_advice` | numbers exist; **exposure is A4** |
| plain wording | **rule code + atom index** in each event (A3) and in **`StuckInfo`** (A5); the scene's positions and cell (snapshot); `d_box(h)` (the `box_advice` formula, factored out) | codes new; rest exists |
| rmse per epoch | `ModelStatus::{epoch, fit, trusted}`: the worker appends a point whenever (epoch, fit) changes | yes (app-side history, session) |
| parity | the store's samples, tags, quarantine codes + `Model::predict(…, false)` once per sample, on demand (A6) | data exists; **builder is A6** |
| pair curves | `Model::predict(dimer, true)` on 96 r per pair (one prediction per point serves the band, the lane and the verdict); `Model::ref_energy`; `Model::sigma_e`; the store's two-atom samples; **read-only "would it answer"** (A6) | predictor exists; **builder + read-only rule check are A6** |
| coverage: pair distances | the store (fitted samples), on demand (A6) | data exists |
| coverage: burst temperatures | **`TeachEvent::burst_t`** (A3, from the burst geometry's Verlet velocities before CSVR, via `sim::Scene::temperature`), accumulated per session in the app, and the plan's `hot_burst_t` | new field |
| timeline | A3 events, the rmse history, `quarantined` deltas between snapshots, `sampling` start/finish transitions | derived |

### 4.3 The additions (minimal; nothing enters the fit, the trust rule or the fallback policy)

- **A1. `sim::activity::ActivityProbe`**: live solve state as relaxed atomics, shared by `Arc` between the worker thread, the background auditor's thread and the render thread. Two `SolveSlot`s (`main`: the stepping teacher; `side`: the background auditor), each `{running, scf_step, started_us, solves}`, plus a refit phase `{started_us, samples}`. `providers::GpuKs::with_probe(probe, Slot)` wraps the `should_abort` it hands `solve_with` so each poll (one per SCF step) bumps `scf_step`, and marks the slot running from the start of `evaluate` to its end. `CpuKs::with_probe` marks running only (no step count: `qm::scf` has no cancellation point). The worker constructs the providers, so it attaches the probe; `ForceProvider` does not change. Cost: one relaxed `fetch_add` per SCF step and two stores per solve.
- **A2. `sim::activity::SolveCounts`** in `SurrogateKs`: lifetime counts of teaching solves by origin, plus how many were kept (not kept = total − kept). Reset by `forget`; restored by `install_sidecar` on its success path only (plan review N11); written by `save_sidecar` as an **optional v3 header key** `"teacher_solves": {"pass": …, "fallback": …, "audit": …, "verification": …, "kept": …, "complete": …}` (unknown keys are already ignored by the v3 reader, so older builds read these files unchanged; a file without the key reads as `None` → `unrecorded_before: true`) **[provisional, A2]**. `"complete": false` persists `unrecorded_before`, so a model once loaded from a file without the record keeps its caveat through every later autosave (plan review M4). A present but malformed record reads as `None` with a note; it never refuses the model (plan review S17). User-data's sidecar v4 (§7.1 there) takes it over as `library.solves`. Exposed as `TrainableStatus::solves`.
- **A3. `sim::activity::TeachEvent`**, pushed by `SurrogateKs` at every teaching solve into a bounded queue (2 048 events; overflow drops the oldest and counts `dropped`), drained by the new `Trainable::take_teach_events()` (default: empty). It carries the moment, origin, **`TeachRule` code**, the atom the rule names, the provider's technical text (every note of that solve, plan review S12), the **`TeachOutcome`**, face density and box flag, the teacher − model gap when the model was consulted, the atom count, the solve time, the audit verdict and lag, `burst_t` (post-Verlet, pre-CSVR) and `step`. `sim::Dynamics` drains the provider right after each `evaluate` into its own bounded buffer (2 048) and stamps `step` with the step whose evaluation produced the event; pass events stay unstamped. The worker drains the driver's buffer once per publish and adds nothing (plan review M6; `Trainable::take_teach_events` lives in `sim`, so rule 5 holds). The background auditor's failed solves and its discarded answers (the model changed settings or species) push events too, with origin Audit; a cancelled solve pushes nothing (plan review S11). To fill `rule` and `outcome` without re-deciding anything, the policy's existing decision points return a code next to the string they already build: `Gate`, `envelope_outside`, `VarianceVerdict`, `class_untrusted` and the fence check gain a `TeachRule` beside their reason. `admit` returns `Admission { kept: bool, outcome: TeachOutcome }` instead of `bool`. The pass's `Sampler::{record, sample, burst_step, Job::Reference, Job::Skip}` push their own events through `SurrogateKs::push_event`. Cost: one small struct per teacher solve (≥ 10⁵× cheaper than the solve).
- **A4. `TrainableStatus::trust_checks: Vec<TrustCheck>`**: the trust rule's conditions with value, limit and met, in the rule's order. There is one predicate per condition (plan review M2). The list is built as `met = !(predicate)`; `untrusted_reason` is its first unmet entry, worded with today's strings (byte-identical); `residuals_fail` and `noise_fails` call the same predicates. The audit entry is per atom (plan review M1): |gap|/N against `energy_std_per_atom` (key `Audit`), or, when the forces failed the audit, the worst atom's residual against its bound (key `AuditForces`). With no failed audit since the refit it is met at value 0, and the UI words that line from `last_audit` ("last check: passed / box-affected, not judged / failed before the refit / none yet"), never "passed" by default. A readout field `audit_atoms` (the audited geometry's N) is set beside `audit_failed`; it decides nothing. The class rule's verdict is exported in the same way: `ClassLoo::refusal: Option<(ClassKey {Samples, Force, Energy}, value, limit)>`, computed by the function that sets `ClassLoo::trusted` and read by `class_untrusted`'s wording (plan review M3). Pinned: the first unmet check is the condition `untrusted_reason` names, for every branch, at equality and on NaN.
- **A5. `TrainableStatus::stuck_info: Option<StuckInfo>`**: the rule code (`None` when no reason was stored, never a guessed rule; plan review S10), atom, streak length, whether the streak is all box-affected, the gap and the largest face density, i.e. the structured form of what `stuck()` formats today. `stuck` (the string) stays for the Log.
- **A6. `Trainable::training_views() -> Option<TrainingViews>`** (default `None`): the parity points, the force parity, the pair curves with band, rule 6's ratio, dots and the would-answer verdict per r, and the pair-distance histograms, built on demand from the store and the current model. It counts the samples and curves it had to omit. It needs one new read-only method, `SurrogateKs::would_answer(&self, pos, z, provenance) -> Result<(), (TeachRule, Option<usize>)>`: rules 1–7 and the fence, in `evaluate`'s priority order with early returns and a fresh variance check. It is read-only by type (plan review S1): the fence reads a descriptor cache that `training_views` fills once. It ignores draft mode, rule 0, audits and the cadence's reuse of an accepting verdict. Pinned against `evaluate` (rule and atom, the fence, rule 4, draft), and by a run showing that building views between steps changes no decision (§8).
- **A7 (app only).** The worker's fit history (one point per new (epoch, fit)), `Snapshot::activity: ActivityDelta` (the events since the last publish, already stamped by the driver; the fit points; the views when rebuilt; the driver buffer's drop count), `Command::SetActivityViews(bool)`, and `SimWorker::probe()`.

## 5. Plain wording

`app::ui::activity::words` turns codes into sentences. Every plain sentence has the technical text (the provider's own note, unchanged) on hover. Element symbols come from `units::isotopes::symbol` (display only, rule 1). Distances are converted at this edge (Bohr, Å on hover; rule 3).

### 5.1 Why the teacher answered (rule codes)

| code | plain sentence (atom i, element El) |
|---|---|
| `Cell` (0) | atom i (or "an atom" when the check names none) is too close to the edge of the simulation cell — move it inward |
| `NoModel` (1) | there is no model yet — this answer helps build one |
| `NewSpecies(z)` (2) | the model has never seen El — answers like this one teach it; it can answer for El after a refit with El's reference energy and enough El samples (plan review S13: the refit needs the reference, and the larger basis stays untrusted until El samples accumulate) |
| `OtherSettings` (3) | this scene's settings (grid, smearing, charge or spin) differ from the model's — the answer cannot be used for it |
| `Untrusted(key)` (4) | the model was not trusted then: the key's short title ("too few samples in the fit", "the last check against DFT had failed", …). A log row is history, so it is worded by its key and never shows today's numbers; the numbers at the time are in its technical text (plan review S14). The Model tab and the toolbar, which show the present, use the first unmet line of §5.3 |
| `Class(Pair / ManyBody)` (4b) | the model hasn't learned {two-atom / three-or-more-atom} arrangements well enough yet |
| `IsolatedAtom` (5a) | atom i has no neighbour within reach — the model can't describe a lone atom inside a molecule's spin state (r_cut on hover) |
| `Coordination`, `Spread`, `NoGrade`, `NewEnvironment`, `UnspannedAtom`, `Grade` (5b–5e) | atom i is outside what the model has learned |
| `ForceUncertainty` (6), `EnergyUncertainty` (7) | the model is unsure here (atom i), plus "— this answer teaches it" only when the answer was kept and no box clause applies (plan review S13) |
| `NearQuarantine` | this geometry is next to one the model could not learn |
| `Audit` (8) | a routine check of the model against DFT (the cadence on hover) |
| `RelaxVerification` | checking the relaxed geometry with DFT |
| `Pass(Reference{z, m})` | sampling: the isolated El atom (spin multiplicity m), for its reference energy |
| `Pass(Perturbation / Scan / HotBurst)` | sampling: a random nudge of the scene / a bond-length scan / a hot MD burst at T |

**The box clause.** For atom rules (5a–7), the sentence gains a box clause when the box is involved:

- the answer was box-affected **and** atom i lies within `d_box(h)` of a cell face: "… **and is touching the box edge — move it inward**". This is the user's example sentence, verbatim. Where the sentence's subject is the model (rules 6–7) it reads "…, and atom i is touching the box edge — move it inward".
- the answer was box-affected, but atom i is not the one near a face: "… — this answer also touches the box walls, so it is not learned here; a bigger cell would help". If such an answer was kept (an essential answer while the model is being built), the clause says instead "so it stays in the fit only until enough others exist".
- atom i is within `d_box(h)` of a face, but the answer was not box-affected: no clause.

`d_box(h) = ln(1/(π ρ_rule(h)))/2` is `box_advice`'s distance (5.35 Bohr at h = 0.15), factored out as `ui::d_box`. It is the same stated hydrogen-tail estimate and carries the same caveat.

### 5.2 Outcomes

`Kept` "kept: it will be in the next fit"; `KeptBoxAffected` "kept while the model is being built; it touches the box walls and leaves the fit once enough others exist"; `KeptByVote` "kept: `FENCE_VOTES` fresh answers agree with each other but not with the model — the next refit decides"; `KeptReleased{r, s}` "kept: it agrees with the model, so r quarantined samples were released (s superseded)"; `Thinned` "skipped: too similar to a sample already held (one in `THIN_EVERY` such is kept)"; `BoxNotKept` "not kept: it touches the box walls — the box's squeeze, not the molecule"; `FenceDisagrees` "not kept: next to a geometry the model could not learn, and the teacher still disagrees with the model there"; `WaitingForPass` "not kept: many answers are held without a model — run a sampling pass"; `NotOffered` (label "not kept: other settings") "used for this step, not kept for this model: its settings differ"; `Discarded` "not judged and not kept: the model changed settings or species before the background answer arrived"; `StoreRefused` "the store refused it (shape or settings)"; `Reference` "a reference energy, not a sample"; `Failed` "the solve failed"; `NotSolved` "not solved". The numbers in the vote and thinning sentences are the policy constants `FENCE_VOTES` (3) and `THIN_EVERY` (10), read in code, never written as literals (plan review S13).

### 5.3 Trust conditions

| key | ✓ / ✗ line |
|---|---|
| `Samples` | Enough samples in the fit — n of Q′ needed (one per independent direction the model can bend). n is the fit's count, not the held count beside it on the counters (plan review S9) |
| `ResidualDofE`, `ResidualDofF` | Enough data beyond the fit's own parameters to measure its noise — energies d of 30 / forces d of 30 (one line each) |
| `EnergyResidual` | Reproduces its own training energies to the accuracy it claims — worst x mHa/atom, allowed 1 mHa/atom |
| `ForceResidual` | Reproduces its own training forces — worst x, allowed b mHa/Bohr for that atom |
| `ForceNoise` | Its forces fit the data to within the force tier on average — scatter σ_f x, allowed 2 mHa/Bohr. σ_f is label noise and basis misfit together, so the line does not blame the teacher's labels (plan review S8) |
| `Audit` (met) | Last check against DFT — no failed check since the last refit (last check: passed / box-affected, not judged / failed before the refit / none yet, one every N answered steps / none, audits are off), from `last_audit` (plan review M1) |
| `Audit` (unmet) | Last check against DFT — found the energy off by x mHa/atom, allowed 1 mHa/atom (refit to recover): \|gap\|/N against the per-atom tier |
| `AuditForces` | Last check against DFT — found forces off by d, allowed b mHa/Bohr (refit to recover) |
| box (app, `box_blocks_trust`) | Samples clear of the box walls — b in the fit touch them (n held); then `box_advice`'s sentence (b counts box-affected samples in the fit, n the samples held; plan review N6) |
| class (per `ClassLoo::refusal`) | {Two-atom / Three-or-more-atom} arrangements — learned / not yet (k < Q′ samples) / not yet (leave-one-out force error x× the bound) / not yet (leave-one-out energy error x× the tier). Hover: "two-atom" means every atom has at most one neighbour within r_cut, so two separate H₂ count (plan review N8) |

Values in mHa keep two significant figures (σ_f 4.5e-5 Ha/Bohr reads "0.045"; plan review N7). The Model tab's "trusted" line and the toolbar's trust tooltip use the first unmet line, with the old string on hover. The "stuck" variant uses §5.1 with `StuckInfo`: "trusted, but not answering here: atom 0 is outside what the model has learned and is touching the box edge — move it inward (the last 50 steps were DFT)".

## 6. Rate limits and cost

- **The render thread.** The headline, bar and job rows are rebuilt from the probe and the latest snapshot at most every **100 ms** (`ActivityClock`, injected time in tests). The chip requests a repaint after 100 ms while work runs, so the sweep and the step count move during a long solve that publishes no snapshot. Log rows are appended when snapshots arrive; drawing lays out only the visible rows. The plots paint from cached vectors.
- **The worker.** Draining events (the driver stamps them) happens every round and in every publish; appending fit points, once per publish. `training_views` is built only while the UI has the Training section open (`Command::SetActivityViews(true)`), only when what it shows may have changed (`TrainableStatus::model_gen`, the store's counts, trust: not (epoch, samples) alone, Task 4 review I2), on the worker, and at most once per **max(2 s, 50 × the last build's `build_ms`)** (Task 4 review cost ruling: at most 2 % of the worker; the provider caches the curves per model generation, so a rebuild at the same generation is cheap). A build the interval defers wakes an idle worker at its due time (Task 5 review I1). `views: None` in a snapshot means no new build; the UI drops its views when the model is not fitted. `ActivityDelta::run_started` marks a new run (worker start, scene replacement, recovery); the log divides runs by event time. The result travels as an `Arc` in the snapshot that follows.
- **Measured, never asserted.** The worker keeps `activity_ms` (event drain and stamping, fit history, view builds) and the teacher time (the events' `solve_ms`). The Activity tab's footer shows the ratio ("activity bookkeeping 0.04 % of teacher time; views 3.1 ms per build"). A release worker test (GPU; skipped without an adapter) runs the H₂ 49³ pass with the views on and prints: the ratio; the pass's median solve time with the probe attached versus a run with `with_probe` not called; and the view build time. A release UI test prints the Activity tab's frame build time with 5 000 rows and full views. The ship criterion is a ratio **< 1 % of solve time**, read from the printout by the reviewer and recorded in the task report and the README; it is never asserted (rule 7 practice).

## 7. The job list is one component

`app::ui::activity::jobs` defines

```rust
pub enum JobKey { LiveSolve, Pass, Refit, Audit, Background /* a density refresh or an isolated-atom density on the auditor */, BondReferences, Relax, PassOffer }
pub enum JobProgress { Fraction { done: usize, total: usize }, Steps { step: u32, last: Option<u32> }, Indeterminate, Waiting }
pub struct JobRow { pub key: JobKey, pub title: String, pub detail: String, pub progress: JobProgress,
                    pub elapsed_s: Option<f64>, pub eta_s: Option<f64>, pub control: Option<(String, Action)>, pub no_control_why: Option<&'static str> }
pub fn rows(input: &PanelInput<'_>, probe: Option<&ProbeView>) -> Vec<JobRow>;
pub fn show(ui: &mut egui::Ui, rows: &[JobRow], acts: &mut Vec<Action>);
```

The precision milestone adds `JobKey::Queued(jobs::JobId)` rows from its runner and reuses `show` in its Jobs page (its ruling P7 already puts the live sampling pass in the queue view). The user-data model manager's long operations (import, migration) add keys the same way. The toolbar's future jobs chip (precision §8.4) counts these rows. Nobody builds a second list.

## 8. Testing

- **Providers (`cargo test -p providers --release`).** For each rule path the existing tests already drive (no model; new species; other settings; untrusted on count and on residuals; class; 5a; 5c; 5e; 6–7 via `always_unsure`; fence; audit in the step and in the background; relax verification; a pass), the drained event has the expected `TeachRule`, atom, outcome and origin, and the gap equals the note's. The background auditor's failed and discarded solves leave an event and are counted. Driven by `sim::Dynamics`, a tick of several fallback steps gives distinct, increasing step stamps. `SolveCounts` adds up across a pass, fallbacks and audits, survives a sidecar round trip, reads `unrecorded_before` from a file without the key, keeps it through load → save → load, and a malformed record reads as unrecorded with a note. `trust_checks`' first unmet key matches `untrusted_reason` on every branch, at equality and on NaN; the refusal strings are byte-identical; the audit check is per atom or names forces; the class verdict names its failing condition. `would_answer` agrees with `evaluate` (variance cadence 1, audits off) on rule and atom over a 200-geometry sweep of dimers and trimers, and on the fence, rule 4 and draft mode. Two identical providers, one building views between every step, stay bit-identical over 200 steps. `training_views`: parity points equal `predict` at the stored samples with the fit's references; the band is the epistemic std and the force ratio is rule 6's; the force parity is thinned by whole atoms; it makes no teacher call (`MorseDimer::calls` unchanged). **Gate (rule 6): `cargo test -p providers --release`, `cargo test -p validate --release --test surrogate`, unchanged numbers.**
- **sim.** `ActivityProbe` counts and views; `SolveCounts::total`; the driver's `TeachBuffer` stamps non-pass events with the given step, keeps an existing stamp, and drops and counts the oldest past its cap.
- **Worker (`cargo test -p app --release`).** On a `SurrogateKs` over `MorseDimer` (no GPU): events arrive in snapshots stamped with the driver's step, in order, none lost across ticks; the fit history gains a point per refit; views are built only while enabled and at most once per 2 s; the probe shows `running` during a slow analytic solve. On the GPU (skips without an adapter): the overhead printout of §6.
- **UI (egui, headless, fixtures).** Headline per row of the §3.1 table; the bar's mode; job rows and controls emit the existing actions; the counters; log rows' text, the step cell, filters, the 5 000 cap, the session burst histogram past it; the checklist's ✓/✗ and "this is why"; the box blamed only for a fit condition; the audit line for each `last_audit` state; the class line for an energy-only failure; every §5 sentence including the box clause and the kept-conditional phrases; the pair curve's refused stretches, its y range and the lane ratios; plots draw with empty, one-point and full data; the throttle (a second rebuild inside 100 ms returns the cached strings); the smoke run of every layout includes the Activity tab; every new search anchor is marked.
- **Screenshots.** `--screenshot <png> --ui "tab=activity"` (new value) with `--estimated --sample`, and `tab=activity;wait=sampling` mid-pass. Both are added to the docs screenshot list.
- **Replay.** `surrogate_far_drag_recovers.replay` gains `expect solve-log 1` (at least one solve-log row since the mark, with a rule and an outcome) and `expect counters` (samples counter = `ModelStatus::samples`; session counter ≥ log rows since the mark).

## 9. Physics honesty in the views (Fable decisions, physics-review gate)

1. **Parity axes.** (E − Σ E_ref)/N per sample: the quantity the fit models (it subtracts Σ ref itself; `surrogate::fit`) and the one the energy tier is per atom of. Plotting raw E would put every point on a line set by the references.
2. **The band is σ_epi**, √(energy_std² − (N s_E)²) at N = 2: the model's reducible uncertainty, the exact quantity rule 7 judges (÷ N against `energy_std_per_atom`). The review approved this band (A5). Rule 7 compares a **width** with the tier, so the comparison is drawn where widths live: a log-y lane under the curve with σ_epi/N ÷ tier against 1, beside rule 6's ratio max_j σ_F,epi/(√3·bound_j) against 1. Horizontal ±N·tier lines in a plot of E − Σ E_ref would compare an energy with a width, and would cross the curve only where E − 2E_ref ≈ 0 (the repulsive wall, near r_cut). A trained H₂ band is about 10⁻⁶ to 10⁻⁵ Ha, a fraction of a pixel on a 0.2 Ha curve, so the band alone would read as "no uncertainty" (plan review M5). The floor N·s_E is the fit's residual scatter — teacher noise plus what this basis cannot fit; more samples of the same kind do not remove it, and it is not "the teacher's scatter" (Task 4 review I1). It is shown as a thin grey band on the curve and as a grey line in the lane, labelled "the fit's residual scatter s_E (teacher noise plus what this basis cannot fit; more samples of the same kind do not remove it)". The band is ±1σ model (coefficient) uncertainty, what rule 7 judges. It is **not an error bar against DFT**: it holds no basis misfit, so the model's error can exceed it; the audits and the parity plot measure that error. If the variance is unavailable (no posterior) the band and the lane are omitted and the caption says so.
3. **"Thin band = learned" is not the whole rule.** Rules 4, 4b, 5a–5e, 6 and the fence refuse independently of the energy band, and rule 6 is usually the binding one. So where the policy would refuse is read from `would_answer` (A6), never from σ, and is the dominant encoding: the curve is dashed there, its band dimmed, each stretch labelled with its rule. A narrow band on a dashed stretch reads correctly: "confident, but the model won't answer here: 5e grade". The y range comes from the answered points and the dots, so a refused extrapolation toward 0.5 Bohr does not set the scale.
4. **The dimer is a slice.** The model is a sum of per-atom terms whose descriptors couple every neighbour (even the dimer carries body-3 self-terms), so the curve is the model's energy of an isolated pair. It is not a pair potential: in a molecule the model has no per-pair energy. The caption says so, and says that the band, the lane and the dashed stretches are for the isolated pair. In draft mode the model answers wherever rules 1–3 pass, and the caption says the dashed stretches show where it would fall back outside draft mode. The dots are only two-atom samples; the rug shows where the pair occurs in any sample.
5. **Burst temperature** is the instantaneous kinetic temperature (3N − 3 dof, `sim::Scene::temperature`) of each solved burst geometry, from its Verlet velocities v(t + dt) before the CSVR thermostat rescales them. A 2-atom burst at a 2000 K target spreads widely: with f = 3 the instantaneous T has a relative std of √(2/3) ≈ 0.82, so the histogram is wide by physics. Consecutive points of one burst are correlated (τ = 5 fs at dt = 0.25 fs). The caption says "instantaneous" and "correlated".
6. **Box share** is `box_blocks_trust`'s rule unchanged (no tuned fraction), with `box_advice`'s cell.
7. **Units.** Everything crosses the boundary in Hartree and Bohr. mHa, mHa/Bohr, Å and K are display conversions (rule 3).

## 10. Model assignment

- **Fable 5.1** (wrong answers look plausible here): the rule and outcome codes threaded through the policy (A3's provider side), `trust_checks` and `stuck_info` (A4, A5), `would_answer` and `training_views` (A6, §9). Each gets a mandatory `physics-reviewer` pass before it lands.
- **Opus 5.5**: the probe and its KS wiring, the review's amendments to those records and the driver's event stamping (plan Task 1b), `SolveCounts` and the sidecar key, the worker plumbing, every UI piece, screenshots, replay, docs.

## 11. Provisional decisions and open questions (proposed rulings)

- **A1. Where the bar goes.** The toolbar is at the bottom of the window, so "under the toolbar" is the window's bottom edge, the least-watched place on screen. Proposed: the toolbar's top edge (the seam with the 3D view), always next to the controls.
- **A2. Persisting the lifetime solve count.** Proposed: an optional v3 header key now (older builds ignore it), moved into user-data's v4 `library.solves`. The alternative (in memory only until user-data) would show "≥ samples (earlier unrecorded)" after every backend switch, because the autosave round-trips through the file.
- **A3. Ghost atoms and the live SCF-convergence chart are deferred** to a follow-up that may change `qm-gpu` (rule 6, validation gate). This milestone shows the step number only.
- **A4. Parity in (E − Σ E_ref)/N** (§9.1).
- **A5. Band = epistemic σ, with the fit's residual scatter drawn separately** (§9.2; Task 4 review I1 renamed the floor).
- **A6. Audits stay tagged `Fallback` in the store** (no sidecar tag change before user-data's v4). The solve log knows audits; the parity colours say "fallback (incl. audits)".
- **A7. Rule-3 solves** (other settings) are logged but not counted toward the model's lifetime solves: they were not spent on this model.
- **A8. Side jobs** (density refreshes, isolated-atom densities, bond references) are job rows, not solve-log rows: they teach nothing.

## 12. Plan review 2026-10-15

The physics review of the plan (`docs/superpowers/notes/2026-10-15-activity-plan-review.md`, read-only, at `6dd3904`) found the plan "ready after amendments": 6 MUST, 18 SHOULD, 12 NOTE. It confirmed that Tasks 2–4 cannot change a policy decision, that every refusal path except one has a code, and that `would_answer` follows `evaluate`'s order. It approved the parity axes (A4) and the epistemic band (A5). The plan folds in every MUST and SHOULD and the useful NOTEs. Changes to the records Task 1 had already built (`fcdd9bd`) go into a new plan Task 1b; the rest are in the tasks they name.

| finding | change | why |
|---|---|---|
| M1 | The audit check is \|gap\|/N against the per-atom tier, or the worst atom's force residual against its bound (`TrustKey::AuditForces`); a readout `audit_atoms`; met at value 0 and worded from `last_audit` ("none yet", "passed", "failed before the refit", …) | A total gap was set against a per-atom limit, a force-only failure printed an energy, and "passed" showed when no audit had run |
| M2 | One predicate per trust condition; `trust_checks` is `met = !predicate`, `untrusted_check` its first unmet entry, `residuals_fail`/`noise_fails` call the same | Hand-negated copies disagree on NaN and drift silently |
| M3 | `ClassLoo::refusal: Option<(ClassKey, value, limit)>` from one `class_refusal` that also sets `trusted`; the class line words it | The class line re-derived the branch and called an energy-only failure a force error |
| M4 | `"teacher_solves"` gains `"complete"`; `TeacherSolves::complete` round-trips `unrecorded_before` | The "earlier solves unrecorded" caveat was lost at the first autosave, after which the counter could read below the sample count |
| M5 | The pair curve is dashed, band dimmed and labelled where `would_answer` refuses; a log-y lane shows σ_epi/N ÷ tier and rule 6's ratio against 1 (`PairCurve::force_ratio`) with the floor; y range from answered points and dots; the ±2·tier lines and the 4 px strip are gone | Horizontal tier lines compared an energy with a width; the band is sub-pixel; rules 4–6 and the fence refuse independently of it |
| M6 | `TeachEvent::step`, stamped by `sim::Dynamics` right after each `evaluate` (`TeachBuffer`); the worker no longer stamps; an audit shows "judged k answered steps after its dispatch"; `StampedEvent` and the computed dispatch step are gone | The publish-time stamp gave n rows one step at speed n, and "verdict step − lag" mixed answered and fallback steps |
| S1 | `would_answer(&self)`; the fence reads a descriptor cache that `training_views` fills (`quarantine_distances_cached`); a 200-step pin that view builds change nothing; the sweep compares atoms and covers the fence, rule 4 and draft | "No state moves" enforced by type, and pinned by behaviour |
| S2 | One prediction per curve point serves the band, the lane and the verdict; refusals return early in priority order; the build printout gives the worker's time share | The O(Q′²) variance ran twice per point, and at S = 4 the build runs between steps |
| S3 | Caption: "not a pair potential … for the isolated pair"; the draft caption; what the verdict leaves out | The old caption implied a per-pair energy in the molecule |
| S4 | The views use `Model::ref_energy` | `self.refs` can differ after a re-solved reference, and could panic |
| S5 | Force parity thinned by whole atoms (`thin_atoms`) | A component stride that is a multiple of 3 kept one Cartesian component |
| S6 | A per-session burst histogram apart from the log ring; `burst_t` taken before CSVR; the caption notes correlation | The ring dropped early bursts; the thermostat had already rescaled the velocities |
| S7 | `TeachRule::UnspannedAtom` ("5e new environment") apart from 5d | Two different tests shared one label |
| S8 | `ForceNoise`: "Its forces fit the data to within the force tier on average" | σ_f is noise and misfit together, not the labels disagreeing |
| S9 | "samples in the fit"; the counter hover says "held; F in the current fit" | Two different "samples" numbers sat side by side |
| S10 | `StuckInfo::rule: Option<TeachRule>` | The code invented the fence when no reason was stored |
| S11 | The auditor's failed and discarded solves push events (`TeachOutcome::Discarded`); the hover says cancelled solves are not counted | They were solves spent on the model with no row and no count |
| S12 | `detail` carries every note of the solve | The admission's numbers were lost from the hover |
| S13 | "teaches it" only for a kept answer; the box clause rewritten as a sentence; the new-species sentence no longer promises the next refit; `NotOffered` reads "not kept for this model"; `THIN_EVERY` and `FENCE_VOTES` in code | Each sentence could be false |
| S14 | Log rows word rule 4 by its key | An hour-old row showed today's numbers |
| S15 | `box_blocks_trust` keys on the first unmet `TrustKey` and blames the box only for a fit condition; `COUNT_REFUSALS` goes | String prefixes, and a failed audit is judged against a fresh, non-box answer |
| S16 | "teaching solves this session" | Plain DFT steps never tick it |
| S17 | A malformed record reads as `None` with a note (`teacher_solves_note`); `Json::u64` | An optional key must never refuse a model |
| S18 | Omitted samples and skipped curves are counted (`parity_omitted`, `curves_skipped`) | One failed prediction dropped every view |
| N1, N2, N3, N10 | Documented: box-affected votes on hover; a gated relax verification logs as fallback; rule-0 rows are bounded; an in-step audit reads "DFT fallback" while it runs | Honest edges, no code change beyond the hover |
| N6, N7, N8, N9, N12 | Box line "b in the fit (n held)"; two significant figures in mHa; the class hover; "trusted at the refit"; "in the current fit" | Wording |
| N11 | `install_sidecar` sets the counts only on success | A refused file must leave them alone |

Not changed: N4 (kelvin and milliseconds in display records follow precedent), N5 (`d_box` is the stated hydrogen-tail estimate). `ModelStatus::last_audit_step` (M6 code) had the same units problem as M6(a); fixed in Task 5: it is the driver's stamp on the last judged audit event (the step it was judged at), and the Model tab's lag reads "answered steps".

Task 4 review (2026-10-15, I1): the floor `N·s_E` is the fit's residual scatter, not "the teacher's scatter"; the band is ±1σ model (coefficient) uncertainty and not an error bar against DFT. §3.3.5 and §9.2 above carry the wording; the Activity tab's captions use it (`ui::activity::plots`).
