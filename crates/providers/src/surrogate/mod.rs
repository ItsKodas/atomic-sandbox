//! [`SurrogateKs`]: the surrogate force provider (spec §5). It answers from the fitted linear
//! model (`surrogate::Model`) when the model is trustworthy *and* confident at the current
//! geometry, and otherwise hands the step to its DFT teacher — any [`ForceProvider`] — keeping
//! the teacher's answer as a new training sample. Atomic units throughout; nothing here knows
//! about rendering or the UI (CLAUDE.md rule 5), and nothing is per element beyond Z as a
//! species label (rule 1: the reference energies are inputs computed by the teacher).
//!
//! # The fallback policy, as a decision list
//!
//! `evaluate(scene)` walks this list; the first rule that applies decides. Every rule that hands
//! the step to the teacher writes its reason into `EvalStats::notes` ("surrogate: …").
//!
//! 0. **The cell** (M6 Task 11, R12 under the surrogate): before any rule, the teacher's own
//!    inside-cell check (`Scene::check_inside_cell`: every nucleus ≥ 3h inside every face) —
//!    the same `ProviderError::Invalid` and message as a Kohn-Sham provider, so the driver pauses
//!    with the last good step, in draft mode too. A model answer is a function of the positions
//!    alone and would otherwise carry atoms out of the cell (the user's escape: |x| = 287 Bohr
//!    with no edit, a dissociated H₂ at 8000 K); the store, the envelope and every audit are of
//!    in-cell geometries, so the cell is part of what the model's answer means.
//!
//!    (A capped segment's scene — "Masked scenes" below — is the teacher's before any other rule.)
//! 0b. **Lone atoms** (user request 2026-11-12, fix round 1; [`lone`] holds the derivation): in
//!    strict mode, a scene in which no atom has a neighbour within the model's r_cut
//!    (`no_neighbours`; `lone_cutoff`) is **answered from the isolated-atom references** —
//!    E = Σ_i E_ref(Z_i), every force 0, `Source::Reference` — whether or not a model exists, and
//!    without consulting the trust verdict. That is exactly the model's own answer there (every
//!    descriptor is empty, the fit has no intercept: `surrogate::fit`, "Isolated-atom limit"); with
//!    a model on this surface that covers the species the references summed are its own
//!    `ref_energy`, so the answer is `Model::predict`'s bit for bit and moves only with a refit
//!    (booked). A DFT step there buys nothing: its answer is never kept (the keep rule; the user's
//!    M8a test had 202 lone-H samples spread by 1.3 mHa break trust), and for the scenes the rule
//!    answers it differs from Σ E_ref only by finite-box artefacts no model can learn — measured for
//!    a lone H in the default 12-Bohr box: ≈ 1e-6 Ha of egg-box ripple and the grid box's
//!    confinement (its Dirichlet boundary, not the container walls, which `sim` applies on top),
//!    +1.8e-3 Ha 4 Bohr from the nearest face and +3.5e-2 Ha at 2 Bohr (`validate/references.md`).
//!    Why: an untrusted shared model (the user's single H) fell back to DFT on every step. Without
//!    such a model the references are the provider's: per scene the store's (its grid, h, σ,
//!    tolerance, teacher and ionic tag must be the scene's), in collection mode the canonical ones,
//!    the box's own references solved. The answer is on the model's surface (`Surface::Model`); a
//!    switch to or from a teacher-surface report is booked along the forces (`transition_jump`), a
//!    canonical reference arriving under a provider-reference report is booked as Σ Δref. No teacher
//!    call, no sample, no audit, no teaching event; `SolveCounts::reference` counts it apart (not a
//!    solve, not a fallback, not a model answer). The step's note: "lone atoms: answered from the
//!    atom references". **It declines** for a charged scene (the references are neutral atoms), a
//!    missing or unsolved reference, **mixed species** among two or more lone atoms (cell-wide
//!    aufbau moves charge between species), a pair under a **shell-check fence** (the cutoff was
//!    measured not to hold for it), **two or more atoms not known to end at a gap** (an open shell's
//!    copies, split by the box beyond τ = 1e-3 Ha, release and fill one copy first), and a
//!    **multiplicity other than the parallel-spin product** M* = 1 + Σ (m_i − 1) — two lone H at
//!    "automatic" (1) are the restricted singlet, 4.6e-2 Ha above 2 E_ref at 6.4 Bohr (rule 5a's
//!    measurement), so they go to DFT; at 3 they are answered. A declined lone scene is the
//!    **teacher's** in strict mode: rules 1–4, else rule 5a's refusal (`IsolatedAtom`, the reason
//!    in the step's notes), for one atom too — 5a's one-atom exemption no longer lets the model
//!    answer E_ref for a spin or charge state the scene is not solved at. Draft mode is unchanged
//!    (the model answers lone atoms, the same number), and so is the relax verification (the teacher
//!    was asked for). **Audits are not scheduled at a lone geometry the rule answers** (in draft
//!    mode, the only way to reach them): the gap there is a box artefact, not a model error; a
//!    declined one stays auditable.
//! 1. **No model** (nothing fitted yet) → teacher; the sample is kept; `Source::Fallback`.
//! 2. **Unknown species** (a Z the model's `Basis` has no channel for) → teacher,
//!    unconditionally; the sample is kept and the model is marked for a rebuild with the
//!    enlarged species set at the next [`refit`](SurrogateKs::refit). Never a guess.
//! 3. **Other settings** — the scene's (grid n, h, σ, charge, multiplicity, and since M7 its
//!    *ionic surface*) differ from the model's `Provenance` → teacher, unconditionally; the sample
//!    is NOT kept (it belongs to a different Kohn-Sham surface than the store's). The ionic tag
//!    (`Provenance::ionic`, `crate::pseudo::provenance_tag`) follows the physics, not the menu:
//!    the all-electron tag `none/fd5e-3/pin1e-5` whenever nothing is pseudised (H₂ under `CoresOnly` *is* the
//!    all-electron surface; `crate::pseudo::is_all_electron_tag` also answers for the legacy `"none"`), else
//!    `"<policy>/<surface hash>/fd5e-3/stop2x<floor>/pin1e-5"` (Fermi–Dirac at kT = 5 mHa and the GPU stop rules, plan
//!    2026-11-13 Task 9; the shared-shell-era `…/shared1e-3/1e-1/stop2x5e-8` is the legacy tag); the surfaces differ by tens of Ha per pseudised
//!    atom and in bond angles, so a model taught on one is never consulted on another. **With a
//!    shared collection attached** (M8a Task 4a, `shared` module doc) the comparison is the
//!    collection's surface-key check instead: any box, neutral, automatic multiplicity, the same
//!    h, σ, tolerance, teacher and ionic tag; a charged or spin-forced scene is refused here, its
//!    answers not kept (it keeps a per-scene model).
//! 4. **Untrusted fit** (trust rule below) → teacher; the sample is kept so the fit can earn
//!    trust. The note says which condition failed, e.g. "surrogate untrusted: n_samples 12 <
//!    q_prime 16".
//!    - **4b, untrusted class** (M6 Task 11 round 6; per atom since 2026-10-18; calibrated and
//!      hysteretic since M8a Task 2; "Class-wise trust" below, [`calibrate`]): a class among the
//!      query's atoms (pair or many-body, by each atom's neighbour count within r_cut) is not
//!      trusted — with three or more DFT comparisons of the class in its evidence window (audits
//!      and rule-4b fallbacks of the newest model generations, ≥ 300 points), max(its fit's
//!      leave-one-out force RMS, the force error the teacher measured there, rms in bound units)
//!      or the measured energy error is beyond the verdict's limit (1/√2 to enter, 1 to stay), or,
//!      entering, one atom of the window was measured beyond 3× its bound; with fewer comparisons,
//!      the rule of before (a LOO ratio beyond 1) or one measured atom beyond 3× → teacher; sample
//!      kept (never thinned while the key is short of samples — those earn its trust; thinned
//!      like rule 5 otherwise; never a copy, short or not: `is_label_copy`); reported on the
//!      teacher's surface (the model's energy there is unvalidated); the comparison is a **rule-4b
//!      point of the audit history**, the evidence that calibrates the class. "surrogate: this
//!      kind of environment isn't learned well enough yet (pair: DFT-measured force error 1.40×
//!      the bound over 20 DFT comparisons, limit 1.00 (leave-one-out RMS 0.41×); the model's own
//!      uncertainty was 3.5× too small there; 273 samples)" (`calibrate::refusal_words`, the one
//!      wording of the verdict and this note). Evaluated with the model consulted; when rule 5
//!      fires as well its more specific note is shown. Draft mode answers anyway.
//! 5. **Out of the training envelope** → teacher; sample kept (thinned); reported on the
//!    teacher's surface (M6 Task 11, "One-surface energy accounting"). This guards the directions
//!    the variance cannot: a linear model's predictive variance is a leverage in feature space
//!    and is small wherever the features themselves are small — all body-2 features vanish as a
//!    pair separates, so a dissociating dimer would otherwise be answered "confidently" with
//!    Σ ref by a model that never saw an empty neighbourhood. Three tests, the first that fires
//!    names the reason:
//!    - **5a, isolated atom** (M6 Task 11): in a scene of two or more atoms, an atom with no
//!      neighbour inside r_cut. The model's answer there is its free-atom limit Σ E_ref with zero
//!      force, which is not the teacher's: the scene is solved at one spin state, and a pulled-
//!      apart pair keeps it (the restricted H₂ singlet lies 4.6e-2 Ha above 2 E_ref at r = 6.4,
//!      measured; the diagnosis §1). A one-atom scene is exempt: its answer is E_ref, the
//!      teacher's own isolated atom (rulings S5, S9). So is a **stray atom** of a bonded part
//!      plus stray atoms whose composition checks cleared it for bonded geometries like this one
//!      (`stray` module doc, "Answering inside a region"; spec 2026-11-18 §4.4): exempt from 5a–5e,
//!      the bonded part judged on its own (rule 7 per bonded atom, 5d and the fences on the bonded
//!      part alone).
//!    - **5b, coordination ratio** (M6 Task 11): an atom whose coordination C_t = Σ_{j∈t} f_c(r_ij)
//!      (the n = 0 body-2 feature of neighbour species t) is more than
//!      [`surrogate::envelope::COORDINATION_RATIO`] = 3 below the smallest the fit has seen
//!      among its species' non-isolated training atoms — the vanishing direction judged
//!      relatively, because one kept stretched-pair sample (the user's r = 5.125, f_c 0.05) put an
//!      empty neighbourhood half a training spread inside the old spread box (derivation and the
//!      old/new decision on the repro: the envelope module doc). An isolated atom of a fitted
//!      sample (a released stray atom, or the stray atom of a kept 5a fallback, B = 0) is no
//!      training atom and never enters the envelope (`surrogate::envelope` module doc, "Isolated
//!      atoms are not training atoms"): admitted, it would set the minimum to 0 and switch this
//!      rule off for its species everywhere. A model file whose fit admitted them (written before
//!      this rule) holds those samples out at install (`Q_LONE`, `Q_STRAY`) and its fit is rebuilt
//!      from its stored samples without them (`install_sidecar`; a shared collection keeps no
//!      envelope — its fit is rebuilt from the samples at open).
//!    - **5c, spread**: any feature farther than `max_descriptor_distance` (3.0) training spreads
//!      outside its per-feature training range (`surrogate::Envelope`, unit in its module doc).
//!    - **5d, new environment** (M6 Task 11 round 2, controller ruling): a query row (the energy
//!      row or a force row, built and standardised as the fit's) reaches into a direction no
//!      training row spans — an eigenvector of the standardised training Gram D(G_E + G_F)D with
//!      eigenvalue ≤ τ = Q′·ε·tr, zero to rounding — farther than any training row could:
//!      Σ_null (u·v)² > N_null·τ (`surrogate::NullSpace`; derivation, no tuned constant, in
//!      `surrogate::novelty`). Reason "new environment (feature never seen in training)". The
//!      posterior cannot be trusted there: along an unspanned u the coefficient is the prior's 0
//!      and the variance ‖φ_⊥‖²/λ, small for the evidence's λ, while the answer is a guess. A
//!      feature column zero in every training row is the axis-aligned case; the user's H₂-only
//!      store had no zero column (each body-3 feature of one neighbour is a function of its
//!      distance) but left 17 of 29 directions unspanned, and an H₂ + H geometry reached them
//!      (ν ≈ 1e9–1e10; dimers 3e-6) — the round-1 repro's 4–22× force errors under a confident
//!      posterior. It is evaluated with the variance (same cadence; a rejection is never reused)
//!      and, being an outside-the-data rule, reported on the teacher's surface and thinned like
//!      5a–5c. **Conservative by construction** (physics review, round 3, M5): for a dimer-only
//!      store it fires on *any* neighbour inside r_cut — ν = 1.2e5 (the review's measurement) for H₂ + H with the third atom
//!      at 5.95 Bohr, where the cross terms are tiny but the directions were never seen — and on a
//!      dimer just past a narrow training window (window 1.2–2.2 Bohr: ν = 3.9 at 2.4, 2e2 at
//!      2.6), whose directions that window spans only to rounding. Both are teacher steps that
//!      teach the model; neither is a wrong answer. Cost: one more `describe` per variance check and N_null·(3N + 1) dot products of
//!      length Q′; the eigen-decomposition (Jacobi, O(Q′³)) once per refit — ms at S ≤ 2,
//!      seconds at S = 4 (Q′ 1232; a handoff item).
//!    - **5e, extrapolation grade** (M6 Task 11 round 4, controller ruling; Podryabinkin &
//!      Shapeev, Comput. Mater. Sci. 140, 171 (2017), the D-optimality / MaxVol criterion): per
//!      species, a MaxVol active set of the training atoms' standardised descriptors (in the span
//!      of those rows; rebuilt at every refit and installed model, `surrogate::grade`); a query
//!      atom's grade γ = max_j |(xᵀA⁻¹)_j| writes it as a combination of the active rows. γ ≤ 1
//!      is interpolation inside their parallelotope (every training row has γ ≤ 1 + 1e-3);
//!      γ > [`GRADE_MAX`] = 2 falls back: "extrapolating: grade γ = … > γ_max". A component
//!      outside the training atoms' per-atom span is γ = ∞, reported as "new environment" — the
//!      per-atom analogue of 5d, **not the same test**: 5d checks a query's design rows (its
//!      energy row, the sum of its atoms' x, and its force rows) against the fit's rows, 5e's ∞
//!      each atom's x against the training atoms'; a query's force rows can leave the design span
//!      while every x_i stays inside the per-atom span (and an x_i can leave the per-atom span
//!      while the summed row does not), so both are kept. Per atom (Gubaev, Podryabinkin &
//!      Shapeev, J. Chem. Phys. 148, 241727 (2018); the MLIP package, Novikov et al., Mach. Learn.:
//!      Sci. Technol. 2, 025002 (2021) — the 2017 paper grades whole configurations) because the
//!      model is a sum of per-atom terms: the grade is size-independent and one extrapolating atom
//!      inside an interpolating sum is caught. MaxVol that does not converge within its swap cap
//!      voids the γ ≤ 1 + δ guarantee: every query then falls back ("no grade") and the user is
//!      told, until a refit converges. Why it is needed beside 5d: once a
//!      few three-body samples are kept their rows span the new directions, 5d hands the
//!      geometry to the posterior, and the posterior — misspecified between sparse points —
//!      answered the round-3 hot repro 20.6× off in force with a confident std; the grade keeps
//!      the model to the neighbourhood of what it has actually learned until enough samples
//!      exist (`h2_plus_h_falls_back_on_the_grade_until_enough_is_learned`: no answers for ≤ 10
//!      learned H₃ samples, whose forces elsewhere were 33–65× off; 26/40 at 20, 40/40 at 80, all
//!      ≤ 0.03× the bound). γ_max = 2 is **chosen**, consistent with the MLIP practice of
//!      selecting configurations for training at γ above 1–2 (I could not verify a page or table
//!      for a specific number — no web access in this task): every validated model answers at
//!      γ ≤ 1 (suite unchanged), and the extrapolation it allows is a factor 2 along an active
//!      direction. As a bound on *force* errors the grade is a heuristic (`surrogate::grade`,
//!      "What it does and does not bound"), supported by that transition test and the GPU repro. Fallbacks under 5e are kept like the
//!      other envelope fallbacks (thinned), so the region is learned. Cost: O(P·r + r²) per atom
//!      per consultation (H₂ 6.3 → 7.7 µs/step), MaxVol O(n r²) per swap at a refit.
//! 6. **Force uncertainty, epistemic only** (σ_f excluded since M6 Task 11, as rule 7 excludes
//!    the energy floor): with `force_std_epistemic[j]` the coefficient part of the model's
//!    expected ‖δF_j‖ — the posterior's force variance splits exactly, force_std² = 3σ_f² +
//!    force_std_epistemic² (`surrogate::model`, "Predictive variance") — the per-component
//!    equivalent must satisfy `force_std_epistemic[j]/√3 ≤ min(max(force_std_abs,
//!    force_std_rel·|F_j|), 1e-2)` for every atom (2e-3 Ha/Bohr absolute, 10 % relative, the
//!    relative bound capped at [`FORCE_STD_BOUND_MAX`] = 1e-2 Ha/Bohr so a large force never buys
//!    an unbounded error). Otherwise → teacher; sample kept. σ_f is the label noise: a fallback
//!    on it teaches nothing (no number of samples removes it), and comparing it here made a
//!    noisy fit (the user's σ_f 9.6e-3) fall back on every pass through equilibrium, where the
//!    bound is the absolute 2e-3, and answer near the turning points — a vibration-phase cycle.
//!    Whether such a fit may answer at all is a trust question (the trust rule's noise
//!    condition, below).
//! 7. **Energy uncertainty, per atom and epistemic only**: `energy_std` is the TOTAL energy's
//!    std with the aleatoric floor (N·s_E)² (Task 3), so the criterion is
//!    `√(energy_std² − (N s_E)²) / N ≤ energy_std_per_atom` (1e-3 Ha), N the energy-row atoms
//!    (`surrogate::energy_row_atoms`: the atom count without an isolated atom, the bonded part's at a
//!    stray-atom scene; stray-atoms ruling 9a). The floor is excluded
//!    because it is the teacher's own scatter (egg-box, SCF lag) — the part of the residual no
//!    number of samples removes, so a fallback triggered by it would teach the model nothing;
//!    the epistemic part is what new samples shrink, so it is the part a fallback can act on.
//!    (Per atom the floor contributes s_E ≈ 1e-5 in quadrature, far below 1e-3: including it
//!    would be inert in practice, not wrong; it is excluded for the reason, not the size.)
//!    Otherwise → teacher; sample kept.
//!
//!    **Rules 6 and 7 are calibrated per environment class** (M8a Task 2 fix round 3, [`calibrate`],
//!    "The variance inflation"): each atom's epistemic force std is multiplied by its class's λ
//!    (`ClassLoo::inflation`; the energy's by the largest λ among the atoms) — the teacher's
//!    measured error over the posterior's epistemic std, a ratio of sums in bound units over the
//!    geometries where the std decides (audits, rule-4b and rule-6/7 fallbacks), ≥ 1, used from
//!    three comparisons on. A refusal it decides says "(the model's std × λ, its measured
//!    calibration for this environment class)".
//!
//!    **And by the tail-aware inflations** (trust tightening, spec 2026-11-18 §4; [`calibrate`],
//!    "The tail-aware inflations"): per environment key, λ_F(j) = max(λ_sum, λ_T) on atom j's
//!    force std (rule 6) and max(max_j λ_F(j), max_k λ_E(k)) on σ_E,epi/N (rule 7), N the
//!    normalisation rule 7 and λ_E share (`policy::energy_atoms`). λ_T and λ_E ask rule 6/7's own
//!    question of the measured answer sample P_A: the smallest factor at which every measured
//!    atom (point) the rule would admit is within `WORST_ATOM` (3× the bound; λ_E: 1× the tier)
//!    and, from the evidence floor on, its rms within `ENTER`. A key with fewer than 3 P_A points
//!    of its own takes its class's λ_T and λ_E as floors. The bounds and the tier never move; every
//!    factor is ≥ 1; a NaN or ∞ factor refuses (∞·0 counts as ∞). The factors move at refits,
//!    installs and policy changes only (`refit_cal`), never on a step they decided. A step refused
//!    only by λ_T or λ_E — it passes at λ_sum — is history code 24
//!    ([`HISTORY_RULE_TAIL_INFLATION`]), never in P_A; its words name the factor and its measured
//!    cause ("(the model's std × λ_T 7.7: a measured atom at 4.5× the bound with the model's std
//!    0.13× of it)"; at ∞ "… it lifts as the window ages"), and are byte-identical to the λ_sum
//!    words when λ_sum is the larger.
//! 8. **Audit** (ruling S7): every `audit_every`-th step the model answers (2000 by default;
//!    0 disables) the teacher checks it at that geometry, the sample is kept and the gap is
//!    recorded ([`SurrogateKs::last_audit`]). If |gap|/N > `energy_std_per_atom` — or, since
//!    M6 Task 11 round 2, any atom's force residual ‖F_model − F_teacher‖/√3 exceeds the variance
//!    rule's bound min(max(2e-3, 10 %|F_teacher|), 1e-2) — the audit **fails**: the model is
//!    untrusted (the reason names the forces when they failed it; an energy audit alone passed the
//!    round-1 repro's 4× force errors) until a refit passes it — below. The kept answer is tagged
//!    `SampleTag::Audit` (fallbacks are `Fallback`; before 2026-10-18 audits were too).
//!    - **A failed audit stays failed** (fix (c), 2026-10-18 trust review). Before, a failed
//!      audit untrusted the model "until the next refit" — and the worker refits as soon as a
//!      sample is kept, which the audit's own answer is, so the failure was erased by the refit
//!      its own sample triggered whenever the training residuals then passed (the new fit
//!      reproduces the one sample it was handed; that says nothing about the neighbourhood the
//!      audit caught it wrong in). Now every refit **re-judges** the failed audit at its own
//!      geometry by what the new fit predicts there *without* the audited answer: the exact
//!      leave-one-out prediction (`surrogate::loo`) when the answer is in the fit, the plain
//!      prediction when it was not kept — against the audit's own criteria (energy tier, every
//!      atom's force bound, the pseudo floor). Passed → trust restored ("the failed audit is
//!      passed by the refit's model without the audited sample"); failed → still untrusted, with
//!      the new numbers and "samples near it are needed"; an undefined leave-one-out (the sample
//!      alone spans a direction) fails. If the trust recovery quarantines or supersedes the
//!      audited sample (a glitch the model cannot represent), the failure is handed to the fence:
//!      trust returns and the geometry's neighbourhood falls back, as for any quarantined sample.
//!      Round 1 of the review: a cleared failure audits the next `AUDIT_BURST` (3) answered steps
//!      (a near-twin of the audited geometry can make its leave-one-out pass without testing the
//!      neighbourhood: SHOULD-1); the failed audits not yet re-earned are written to the `.model`
//!      file (header `failed_audits`) and re-judged on load, so a model saved untrusted does not load
//!      trusted (SHOULD-2); a failure still standing after `AUDIT_REJUDGE_LIMIT` (3) refits, or one
//!      whose composition the scene no longer holds (an atom deleted), becomes an **audit fence** —
//!      geometries within the quarantine radius of it fall back, the rest of the policy judges
//!      everything else, and the fence lifts when a re-judge passes (SHOULD-3: otherwise a scene that
//!      never visits it again is DFT everywhere for good). Cost: one more fallback region while
//!      samples arrive — the same price rule 4b already pays, and never a wrong answer (`a_failed_audit_stays_failed_until_a_refit_predicts_it_without_its_sample`,
//!      `a_failed_audit_on_a_glitch_is_handed_to_the_fence`). The force audit is **strict, at 1×
//!    the bound**, like the energy audit (physics review, round 3, M1): a false alarm costs one
//!    untrust and a refit, never a wrong answer. Its false-alarm rate for a calibrated posterior
//!    with per-component std s is P(χ²₃ > 3b²/s²): 0.39 at s = b (the edge of rule 6), 7e-3 at
//!    s = b/2, ≈ 0 in the trained region where s ≪ b (the H₂ models: s ≈ 1e-6–1e-5 against b = 2e-3). A **box-affected** teacher answer
//!    (face density above [`box_affected_face_density`]`(h)`) is recorded but **not judged** and
//!    not kept: its gap is the box's confinement, which the translation-invariant model cannot
//!    and must not learn (Task 9; module doc "Box-affected answers").
//!    - **Asynchronous** (ruling S14, Task 9) when a background auditor is attached
//!      ([`with_auditor`](SurrogateKs::with_auditor): a second teacher instance on its own
//!      thread — the app's GPU teacher). The audit step is the model's own answer; the
//!      geometry goes to the auditor and the model keeps stepping. The verdict is taken at the
//!      first evaluate after the answer arrives and judged by the model current *then* (a refit
//!      in between is what the audit should check). A failed one untrusts the model there, and
//!      the next evaluation — a rule-4 teacher step — reports the teacher's energy with
//!      `refit_jump = E_teacher(r) − E_model(r)` at its own positions (plus a refit's pending
//!      jump), the same booking as the synchronous flip; the note says how many answered steps
//!      since the dispatch went unaudited. At most one audit is in flight: one falling due
//!      meanwhile waits for the auditor. A Forget or an installed model abandons the one in
//!      flight (its answer is discarded). The auditor's warm start is its own (the previous
//!      audit geometry), so its answer at a geometry may differ from the main teacher's by the
//!      SCF tolerance (~1e-6 Ha) — three orders under the verdict's 1e-3 Ha/atom.
//!    - **The effective cadence is the auditor's throughput when the model outruns it.** At
//!      full speed (1000 steps a tick) the model answers ≈ 1.6e5 steps/s, so 2000 answered steps
//!      take ≈ 12 ms while one audit solve takes 130–270 ms: the auditor is saturated, audits
//!      run back to back (the next one is deferred to the first answered step after the
//!      previous answer arrives), and each is judged ≈ 2–4·10⁴ answered steps after its
//!      dispatch — a failed audit arrives that late. `AuditRecord::unaudited_steps` (the panel's
//!      "audit lag") shows it. `audit_every` governs only at paced speeds.
//!    - **In the step** without an auditor (the CPU teacher: a second `CpuKs` would compete for
//!      the same cores, and a CPU solve cannot be abandoned): the step is the teacher's. A
//!      passing audit is an ordinary model-surface Fallback (`teacher_gap`); a failed one is
//!      already reported on the teacher surface — `energy` = teacher's, `teacher_gap` None,
//!      `refit_jump = Some(gap)` (the change of the reported energy at these positions, exactly
//!      as after a refit) — so the driver books the Model → Teacher flip into W continuously
//!      instead of restarting the drift history.
//! 9. Else the model answers: `Source::Surrogate`, no density (`display: None`: the surrogate
//!    has no electrons).
//!
//! **Density samples** (`ForceProvider::sample_density`, M6 spec §3.3, review S4): delegated to
//! the stepping teacher only when the latest `evaluate` was answered by it (`Source::Fallback`,
//! an in-step audit included) — its state is then this geometry's. `None` after a
//! `Source::Surrogate` answer (the teacher still holds an older geometry's density), after a
//! failed evaluation, and after anything that puts another geometry in the teacher or moves the
//! nuclei: `reset`, `discontinuity`, `teacher_mut` (the sampling pass's solves; the driver's
//! references never use it) and a sampling pass step. The background auditor's state is never
//! sampled.
//!
//! `uncertainty` = `max_j force_std_epistemic[j]/√3`, the **per-component** equivalent of the
//! number rule 6 compares (S1's "max per-atom force std" is √3 times it; the label noise σ_f is
//! not in it since M6 Task 11); it is reported whenever the model was consulted (rules 5–9), so
//! the readouts can show how far off the model was on a fallback. Between variance checks
//! (the cadence below) it is the reused verdict's number, at most k − 1 consultations old.
//!
//! **Variance cadence** (Task 9). `variance_cadence` = k: the O(Q′²) variance (rules 6–7) is
//! evaluated afresh on the first consultation after an edit (`reset` / `discontinuity`), a
//! refit, an installed model or a policy change, after every rejection, and otherwise at least
//! on every k-th consultation, and — since M6 Task 11 round 2 — whenever any atom's neighbour
//! count within r_cut differs from the last check's (a neighbour arriving or leaving is a new
//! environment; the round-1 repro answered a three-body geometry on a verdict reused from before
//! the third atom came in); in between the last *accepting* verdict is reused (a rejecting
//! one never is — that would only buy extra DFT steps). The envelope (rule 5) and the
//! quarantine fence are evaluated on every consultation: they are cheap, and a stored envelope
//! verdict must never be replayed at a geometry that is outside. Default k = 4, from the timing
//! table in `references.md` (speed): at S = 4 species (Q′ 1232) the variance is 99 % of a step
//! (44 → 162 steps/s at N = 20 for k = 1 → 4), at S = 2 N = 20 80 % (889 → 1819), at S = 1
//! (H₂) ≈ 2 µs of an 8 µs step; four 0.25 fs steps move a hydrogen at 2000 K (v_rms 3.2e-3
//! Bohr/a.u.) ≈ 0.13 Bohr, inside the 3-spread envelope that is checked every step.
//!
//! **The history's flags and epistemic energy std** (trust tightening Task 3, spec 2026-11-18 §4,
//! "The sample P_A"). Every consulted-model point (`record_history_point`) carries a flags word
//! and the total-energy epistemic std. Neither reads the step's σ or any λ to decide *which* step
//! is flagged — the sample they select must be chosen blind to what it measures:
//!
//! * `CADENCE_FRESH`: `cadence_steps` counts every step that reaches the gate (rules 1–4; the ones
//!   it refuses included — the Task 3 review's minor, so the phase does not follow rule 4's
//!   untrust, which follows audit failures on an answered-step cadence σ-refusals shift) from 0,
//!   and **nothing resets it** (not a refit, an install, a policy change, an edit or a forget); a
//!   step is cadence-fresh when its count c satisfies c mod k = 0, k = `variance_cadence` (≥ 1).
//!   `consults` is not used: it restarts on every fresh variance, and a variance refused by rule
//!   6/7 is fresh on the next step, so its phase follows σ. The bit selects steps; it does not
//!   force the variance: on a cadence-fresh step that `consults` does not make fresh, the last
//!   accepting verdict is reused as before, so nothing any rule decides moves. A draft-mode step
//!   is never cadence-fresh for the history (Task 5: draft-mode points are not in P_A).
//! * `NOVEL` (Task 5; the Task 3 review's MUST): rule 5d's novelty ν > 1 at the geometry. It is
//!   evaluated with every fresh variance (where 5d decides) and, **read-only**, on every
//!   cadence-fresh step that reuses a verdict — where 5d decides nothing and the step may carry a
//!   class code that a fresh verdict would have written as 5d's 11. So on every cadence-fresh
//!   step the bit is a function of the geometry and the fit alone, and `calibrate`'s answer sample
//!   leaves novel points out whichever code they carry. Off the cadence it is set only where the
//!   variance was fresh (and P_A does not read those points). The cost: one null-space projection
//!   per k-th consult at most, beside the variance it would otherwise share.
//! * `FENCED`: the geometry is near a quarantined or forgotten sample (`near` non-empty), an atom
//!   is near a misfit (`near_misfit`), or the shell or audit fence would refuse it. All three are
//!   evaluated on every consulted step — the audit fence read-only (`audit_fence_refusal`) when an
//!   earlier rule (the class rule, 4b) already refused — so a code 5/6/21 point inside a fence is
//!   marked (spec review I4: those codes do not imply that the fences passed). A background audit's
//!   bit is its dispatch step's, OR the fences at its judgement.
//! * `TICK`: the point a calibration tick left (below, "Calibration ticks"): in P_A whichever
//!   code it carries and whether or not it is cadence-fresh, unless fenced or novel.
//! * The epistemic energy std, Ha: σ_E,epi = √max(σ_E² − f², 0), with σ_E the model's total
//!   predictive energy std and f its aleatoric floor — N·s_E for a model fitted on energies per
//!   atom, s_E otherwise (`surrogate::model`, "Predictive variance": σ_E² = f² + φ̃ᵀΣφ̃) — the
//!   same number rule 7 divides by N and inflates (`epistemic_energy_std`). The subtraction
//!   recovers φ̃ᵀΣφ̃ to an absolute error of about ulp(σ_E²), i.e. σ_E,epi to ≈ 1.5e-8·f when the
//!   floor dominates (≈ 1e-13 Ha per atom on a 1e-5 Ha/atom floor, 10⁻¹⁰ of the tier). With the
//!   floor σ_E was built from, the radicand is never negative (σ_E = √fl(f² + φ̃ᵀΣφ̃) ≥ f, and
//!   correctly rounded squaring is monotone); one below 0 is clamped to 0 all the same. A
//!   non-finite or negative value is recorded as absent, never pushed (the history refuses it).
//!
//! **Calibration ticks** (trust tightening Task 7; spec 2026-11-18 §4, "Ticks on their own
//! cadence"). A trusted key's answers are measured on steps nothing selected by σ: every
//! [`TICK_EVERY`] = 25th **in-envelope** step, per provider, independent of `audit_every` (whose
//! default 2000 would let a trusted key's 300-point window never age, and whose 0 would give no
//! ticks at all).
//! * **In-envelope**: a strict-mode step past the gate (rules 1–4), the class rule (4b), the
//!   envelope (5a–5c, 5e and 0b's lone-atom refusal) and every fence (the quarantine and misfit
//!   fences, the shell and audit fences — the `FENCED` bit's set). Rule 5d is not in the decision:
//!   it is evaluated with the variance, which the tick forces; a novel tick's point carries `NOVEL`
//!   and stays out of P_A. `tick_steps` counts these steps and **nothing resets it**; the step whose
//!   count is a multiple of `tick_every` is a tick. A draft-mode step is never in-envelope (its
//!   points are not in P_A) and does not advance the count. The gate is part of the predicate, by
//!   spec (the Task 7 review's M6): after a failed tick or audit untrusts the model (rule 4), the
//!   steps until the refit are gate refusals and leave no point, so the old generation's P_A misses
//!   the interval right after a large error — contained, because the failing comparison itself is
//!   in the history and the refit replaces the model.
//! * **A dead time after every tick** (the Task 7 review's I1 and re-review's R1, the controller's
//!   re-ruling; spec §4's amendment): after **every** due tick — answered or refused, taken or
//!   skipped — the count pauses for D in-envelope steps (`tick_dead`), then 25 counted steps make
//!   the next tick due. D is σ-blind: ⌈EMA(auditor job wall ms) / EMA(in-envelope step wall ms)⌉
//!   (`tick_dead_time`), both EMAs over every job of every kind and every in-envelope step (one
//!   sample capped at 1 s, so a pause is not a step), never selected by an outcome; 0 without an
//!   auditor or before its first job; fixed in tests (`tick_dead_fixed`). So the tick-eligible
//!   steps are the same whatever the ticks' σ outcomes, and with D ≥ the job's duration an answered
//!   tick's job is done before the next one falls due. (The round-1 rule, counting from the previous
//!   tick's completion, paused only after answered ticks — T = t_solve/t_step uncounted steps after
//!   an answered tick, none after a refused one — so refused stretches were over-weighted by
//!   (25 + T)/25; `a_tick_samples_answered_steps_at_their_population_share` fails under it and
//!   under the original count-every-step rule.) A teacher solve **cancelled** on an in-envelope
//!   step undoes what the step booked — its count or dead-time step, and the counts (the step did
//!   not happen; its retry is the tick if this one was — the review's M2: fallback ticks are the
//!   σ-refused ones, so losing them to cancels would be σ-coupled).
//! * **Decided before the variance, on counters and these outcomes alone** — never on σ, a λ, or
//!   whether rules 6/7 will answer — and the decision **forces the variance read-only**: the
//!   novelty (5d) is evaluated for the `NOVEL` bit and the stds for the point even where the
//!   verdict is reused, while the reused verdict, and with it every rule's decision and rule 8's
//!   `answered` count, stays as it was.
//! * **Served.** With a background auditor that is **busy** — an answered tick's job outliving the
//!   dead time, or other work (a rule-8 audit, a refresh, a reference) — the tick is **skipped and
//!   counted**
//!   (`TickCounts::skipped_busy`), whatever answers the step — a fallback tick is skipped too, so
//!   the skip does not read σ — and never deferred. Otherwise: at a **fallback** (rules 6/7, 5d, the relax verification) it needs
//!   no solve — the teacher's answer and the model's stds are both at hand — and the fallback's
//!   point carries `TICK`; at an **answered** step it is an audit, judged by rule 8 as well (a
//!   failed tick untrusts as a failed audit does): asynchronously on the free auditor (an audit job
//!   carrying `TICK`; when rule 8 falls due on the same step the one job serves both), or — with no
//!   auditor (the CPU teacher, the harness) — **in the step** ([`ticks_in_step`](SurrogateKs::ticks_in_step)).
//!   A tick alone is not rule 8's audit: rule 8's cadence (`answered`, `audit_due`, the burst)
//!   and the checklist's box-affected count are untouched by it, and a box-affected tick answer
//!   does not make rule 8 audit again. Its **dispatch** can wait on a tick, as on any job (the
//!   review's M3): a rule-8 audit falling due while a tick's job is on the auditor is deferred to the
//!   next answered step once the auditor is free (`audit_due`), exactly as behind a refresh.
//! * **Counted** ([`tick_counts`](SurrogateKs::tick_counts), booked in one place,
//!   `tick_outcome`): every tick that fell due is exactly one of **recorded** (taken, its point in
//!   the history: [`ticks_taken`](SurrogateKs::ticks_taken)), **unrecorded** (taken, no point: a
//!   box-affected, unfit or variance-less answer — the review's M1: counted apart, so the readout
//!   cannot show calibration while P_A does not grow), **skipped_busy** (the auditor busy with other
//!   work) or **lost** (its auditor job abandoned by a Forget or an installed model, `reset_audits`,
//!   or its thread gone; an answer returned failed or unjudged — discarded, inside a shell band; a
//!   failed teacher solve; no prediction to compare), or is still in flight.
//!   [`ticks_skipped`](SurrogateKs::ticks_skipped) = skipped_busy + lost. `due` counts every due
//!   tick; `answered_due` and `answered_skipped_busy` those whose step the model answered; and the
//!   σ-blind population counters `population` (every in-envelope step, in the dead time and with a
//!   job in flight included) and `population_answered` (of those, the model's answers). The
//!   answered share among due ticks against the population's measures any σ-coupling of which
//!   steps become ticks; the take rates at answered against refused steps, any through skips
//!   (plan Task 9 Step 3b). Readouts; they decide nothing.
//! * **Known limit: the history ring can lift a lock before the window does** (the review's M4).
//!   λ_T = ∞ "lifts as the window ages": the window needs 300 newer P_A points, about 300 × (25 + D)
//!   locked steps (7500 at D = 0). But every locked step records a code-24 point, so a parked scene
//!   evicts the offending point from the 4096-point ring ([`surrogate::HISTORY_CAP`]) after about
//!   4096 steps — earlier, and σ-blind, so benign. More generally, when fallbacks flood the ring, a
//!   key's P_A window is capped at about 4096 × its P_A share.
//! * **Cost** (spec §4, revised by the re-review's R2): the cadence is 25 + D in-envelope steps after
//!   every tick — with an auditor D ≈ t_solve/t_step, so 36–48 at 12 ms/step and 51–79 at 5 ms/step
//!   for 130–270 ms solves; without one D = 0, one teacher solve in the step per 25 answered steps.
//!
//! # A failed GPU (the gpu-fault fix, 2026-11-21)
//! The user's report (v0.2.0, an 8 GB card, estimated mode near the VRAM limit): "a super fast loop where it takes
//! hundreds of samples that seemingly contain no data". Two rules, whatever the device did:
//! - **The session's GPU failure is fatal for this provider.** The first `SolverError::GpuFailed` from any teacher —
//!   the stepping teacher's fallback, a pass's solve or reference, a migration's, the background auditor's job (its
//!   answer carries it, `AuditAnswer::gpu_failed`) — or the stepping teacher's own `device_failure` is latched
//!   ([`SurrogateKs::gpu_failure`]). From then on every `evaluate` is that error — the model answers no step either
//!   (the run is to be saved and the app restarted) — a running pass ends at its next step with it (one job, not the
//!   rest of the queue failing in microseconds), and nothing more is sent to the auditor. `device_failure` reports it,
//!   so the driver stops playing (`sim::dynamics`, "A failed GPU"). The auditor's thread catches a panic inside a job
//!   (the teacher is reset) and answers the job with an error, so a panic no longer ends the thread silently.
//! - **No physically empty answer is kept.** Before a teacher answer becomes a sample or a reference — a fallback or
//!   an in-step audit (`teach`), a background audit or a composition check's part (the auditor), a pass's sample,
//!   scan, burst step, teach-around solve or reference (`crate::sampling`, `sim::references`), a migrated sample —
//!   `sim::answer_refusal` checks it: finite, and for a Kohn-Sham answer at least one SCF iteration, converged, and
//!   ∫ρ within `sim::ELECTRON_COUNT_REL_TOL` (1e-5 relative; derived from the GPU's f32 sums, measured 8.7e-8) of the
//!   scene's electrons. The fallback, the in-step audit and the auditor apply its Kohn-Sham part (`sim::scf_refusal`):
//!   a non-finite answer keeps its pinned handling there (the store never keeps it; a NaN gap fails the audit). A refused answer is counted ([`SurrogateKs::refused_answers`]), leaves a `Failed` activity
//!   event with the reason, and is never kept; a refused fallback is not used for the step either (the step fails, the
//!   driver pauses with the reason).
//!
//! # Draft mode (user request 2026-10-14: `FallbackPolicy::never_fallback`)
//!
//! "Guess instead of falling back (unsafe)": with a model present whose provenance, species and
//! charge match the scene (rules 1–3 pass — otherwise there is nothing to guess with, and those
//! rules still hand the step to the teacher), **every step is answered by the model**, whatever
//! rules 4–7 and the quarantine fence say: an untrusted fit, an atom outside the envelope, an
//! uncertainty above its bound. Such an answer carries `EvalStats::extrapolation` (how many atoms
//! are outside the envelope, the trust rule's reason, the rule that would have fallen back) and,
//! like every consultation, the per-atom uncertainty ratio (`atom_uncertainty`) — the honesty
//! surface the app shows (a red banner, atoms tinted toward red). What the guess *is*: beyond the
//! descriptor cutoff a bond contributes nothing (a pulled-apart pair is two free atoms at Σ E_ref),
//! and a new geometry is answered by the linear fit's extrapolation, confidently wrong where it
//! never saw anything alike. The audits continue, on the background auditor only (an audit in
//! the step would be a teacher step: without an auditor there are none in draft mode); their
//! answers are kept as samples and the worker's refits continue.
//!
//! **Draft-mode audits at geometries strict mode refuses** (fix (b), 2026-10-18 trust review:
//! the user's water session). Such an audit checks the *guess*, at a geometry the guess chose —
//! in the user's run a hole 2–22 Ha below the teacher's surface. Kept as an ordinary sample it
//! entered the strict model's class leave-one-out (rule 4b: one sample alone decides its class)
//! and its novelty/grade/envelope, so a draft session could switch the strict model off
//! afterwards (or, the other way, open its envelope into regions only the guess visited). So the
//! audit job records whether the strict policy refused its geometry **for a reason of the
//! geometry** (round 1, MUST-1: `geometry_specific` — the envelope 5a–5c, 5d, 5e, a non-finite
//! answer, a fence; a refusal by rule 4, 4b, 6 or 7 lifts with trust, so such an audit is an
//! ordinary `Audit` sample, fitted, and a failure is a failed audit) (`AuditJob::draft_guess`) and,
//! when it did: the answer is kept as a `SampleTag::DraftAudit` sample **held out of the strict fit** (`Q_DRAFT`: not fitted, not in
//! the envelope, the grade or the leave-one-out, never fencing; restored from the tag on load;
//! `held_out()`, the Model tab's "fit vs held"), and a failed verdict is recorded and noted but
//! does **not** untrust the strict model — strict mode does not answer there, so the failure
//! says nothing about its answers, and with the held-out sample never fitted it could never be
//! re-judged away (fix (c)). Options weighed: held out of the class leave-one-out only (still
//! fitted) would let strict mode answer where only unjudged guesses taught it — a loosening;
//! quarantined, they would fence the guess's excursions and count against the 10 % cap; held
//! out of the fit loosens nothing, at the cost that draft mode no longer learns from its own
//! excursions (strict-mode fallbacks and passes still teach). **The held-out samples are a test
//! set** (round 1, MUST-1): at every refit each one whose geometry strict mode no longer refuses
//! for a geometry reason (the envelope grew, a fence lifted) is judged by the plain prediction of
//! the fit that never saw it, by the audit's own criteria, promoted into the fit (tagged `Audit`)
//! and refitted; a failure there is a failed audit (rule 8) — so strict mode never answers a
//! region where teacher labels it never fitted contradict it
//! (`a_held_out_guess_is_judged_when_strict_mode_reaches_it`). Files written before 2026-10-18
//! stored every audit, draft-mode ones included, as `Fallback`: those cannot be told apart and stay
//! in the fit (rule 7). A draft-mode audit at a geometry
//! strict mode answers is an ordinary audit (`Audit`, fitted, a failure untrusts as above). A
//! failed ordinary audit no longer stops draft mode answering (it untrusts the model, which draft
//! mode ignores;
//! the flip to the teacher surface is booked when draft mode is switched off, at that step's
//! positions — after a failed audit only: switched off while untrusted for another reason, or on
//! from an untrusted state, the surface changes without a booked jump and the driver restarts the
//! conserved-energy history with a note). A refit in draft mode books its jump whether or not
//! the model is trusted: the next answer is the model's either way (review I1). The energy
//! bookkeeping is otherwise unchanged: every answer is on the model surface. A region whose
//! samples were quarantined is answered by the model too, and re-learned only from the audits
//! there (review M4).
//!
//! # Readiness (fix (a), 2026-10-18 trust review)
//!
//! [`SurrogateKs::trusted`] is the global trust rule (counts, residuals, noise, a failed audit);
//! it never included the class rule (4b) or the envelope (5a–5e), so the user's water model read
//! "trusted" while every step fell back. `TrainableStatus::readiness` is the strict policy's
//! verdict at the scene's current geometry — answers, or the refusing rule and its text (for 4b
//! with the class's worst leave-one-out sample and its origin) — the verdict `evaluate` recorded
//! there (before draft mode overrode it) or, after an edit, a refit or a trust change, the same
//! rules afresh (`inspect::first_refusal`; module doc of `inspect`). The app shows it as
//! "trusted, not here — …" (the status chip and the Model tab), as a line of the Activity tab's
//! trust checklist, and in the play/snapshot stdout lines with the draft flag.
//!
//! # Side densities (M5b, "Density in estimated mode")
//!
//! The surrogate has no electrons, so a model step shows no density. The background auditor's
//! solves are real Kohn-Sham densities, though: an audit's display (its teacher must have display
//! on) is handed out as a [`SideDisplay`] (`SideKind::Audit`) through
//! [`ForceProvider::take_side_display`], with the audited positions and its side-job number
//! (the driver stamps the number with the simulated time of the dispatch). The app can also ask
//! the idle auditor for a **display-only** solve at the current geometry
//! ([`request_display`](SurrogateKs::request_display), `SideKind::Refresh`: never judged, never a
//! sample) and for the **isolated-atom density** of a species on the scene's cell
//! ([`request_atom_density`](SurrogateKs::request_atom_density), taken with
//! [`take_atom_densities`](SurrogateKs::take_atom_densities): the app's atomic superposition is
//! built from them — Z in, the teacher's own atom out, nothing per element). All of them share
//! the auditor's one job slot: an audit falling due while it is busy waits (`audit_due`), and a
//! request made while it is busy is queued and sent as soon as it is free — ahead of a waiting
//! audit, so at full speed (audits back to back) a request is not starved; it delays one audit
//! by one solve: with the auditor saturated, "refresh every N s" costs ≈ t_solve / N of the
//! audits (1–20 % for 0.1–0.2 s solves at N = 1–10 s; review M6). An audit answer taken outside
//! `evaluate` (a side request polled while paused) is held unjudged until the next evaluate, so
//! the queue never waits for Play (review M7). A Forget or an installed model abandons whatever
//! is in flight. Without an auditor (the CPU teacher) none of this exists: the audits run in the
//! step and their density is the step's own.
//!
//! **Harris display jobs** (live-orbitals spec §4.2): `harris_display` queues a display-only
//! Harris solve (`JobKind::HarrisDisplay`: one diagonalisation in the potential of the superposed
//! isolated atoms, orbitals only) and answers `HarrisAnswer::Queued` with its side-job number;
//! `None` without an auditor, for an empty scene or while a sampling pass runs. **At most one is
//! queued** (`harris_queued`): a newer request replaces it, so the latest positions win. It goes
//! **last** — only when the side queue is empty (refreshes, and the atoms and references it needs)
//! and, while playing, no audit is due (a safety check). Paused (the driver's `force_every_step`)
//! it does not wait for a due audit — only an answered step dispatches one, and paused there may be
//! none — so a paused change is shown and the audit goes at the next evaluate. Never during a pass:
//! starting one drops the queued job (the caller asks again). On the auditor's thread it is
//! not an SCF job: no `reset`, no `discontinuity`, the "last job was an atom" flag left as it was;
//! the teacher solves it into its own Harris state (`GpuKs`, `ks` module doc "Harris display"), so
//! the next audit or refresh starts from the same warm state as without it
//! (`harris_never_resets_the_auditor_teacher`). The job's view is set as for a refresh (the rate
//! flag included); it is abandoned with its generation (a Forget, an installed model) and yields
//! to the user's next command (the display interrupt), as display work does. Its payload is kept
//! in `side_harris`, beside `side_display`, as a `SideKind::Harris` [`SideDisplay`] with no density;
//! `take_side_display` hands out the density job's payload first, then the Harris one. A failed
//! Harris solve leaves a note only; a cancelled one (superseded, or yielding to the user) nothing.
//! When the auditor's thread is gone the queued Harris job and the side queue go with it. It is **never** a fallback, an audit, a teacher
//! solve, a sample or a teach event, and no counter moves (G4,
//! `harris_job_is_never_a_fallback_or_audit`). `side_job_in_flight` holds while one is queued or
//! running, and while its payload waits in `side_harris` for the driver (final review M2).
//!
//! **The isolated atoms are valence-only.** The auditor solves every `Atom` and `Reference` job
//! with the display's partial core off, whatever `set_display_partial_core` says: each
//! `AtomProfile` built from them is the valence density (spec §4.1; the Harris input must not
//! contain ρ_pc, `atom_jobs_solve_without_the_partial_core`). The superposition cloud is built
//! from the same profiles, so with "show partial core" on it is valence-only while a solved
//! density shows ρ_pc (`app::cloud` says so).
//!
//! **Orbitals** (orbital-viewer spec §4.3, §7; plan Task 3, review SHOULD 12): estimated mode has
//! **exactly one orbital producer**, the background auditor's teacher. `set_orbital_view` is kept
//! here and travels with every job sent to the auditor; the auditor's thread hands it to its
//! teacher before an audit or a refresh, so their [`SideDisplay`]s carry the orbitals. **The rate
//! flag** (Task 4 review I1): a refresh carries the driver's `force_every_step` (set while paused:
//! a deliberate paused display, which runs the unoccupied eigensolve at once), an audit never — so
//! the auditor teacher's own `UnoccupiedCadence` budgets the eigensolve against *its* SCF wall
//! (≤ 10 %, paused-only in practice: Task 0's 4–50 SCFs per eigensolve), and background audits and
//! "refresh every N s" in play keep their job rate; their payloads then show the earlier unoccupied
//! rows stamped not current. An unforced job's payload also follows the 100 ms orbital rate bound
//! (a teacher solve takes longer, so it rarely skips one). The driver's display interrupt
//! (`set_display_interrupt`) travels with every job, so a forced refresh's eigensolve on the
//! auditor's thread yields to the user's next command and frees the one job slot (Task 4
//! re-review N1: Play pressed during it would otherwise wait out the eigensolve before the next
//! audit). All the payloads
//! (`SideDisplay::orbitals`) come from that teacher's one tracker — tracks, colours and signs
//! continue from one side solve to the next (the thread calls `discontinuity`, never `reset`,
//! between them; a Harris display job calls neither and shares the same tracker, "Side
//! densities"). An isolated-atom job runs with the view off (its teacher is reset around it
//! anyway: that is a new scene for the tracker). So **the tracks restart after an atom or
//! reference job** (Task 3 review M3): the view's producer is dropped with the view off, and the
//! next audit or refresh hands out fresh tracks (never a reused id, `crate::orbitals`); a
//! selected orbital is then found again under its new track by its channel and rank, with no note
//! (`crate::orbitals`, "The selection fallback"). The unoccupied tracks also restart after every
//! audit in play that follows a Harris payload (its converged payload carries no unoccupied rows:
//! `crate::ks`, "Harris display"); a picked unoccupied orbital is left unresolved by that payload,
//! which the driver does not show, and found again in the next Harris one (user report
//! 2026-11-16: it used to switch to the HOMO about every 30 s). The stepping teacher never gets the request:
//! a fallback step carries no orbitals, and a model step has no electrons. So
//! `orbital_display_now` is `None` here; the caller asks for a display refresh instead
//! (`request_display`, which is also how "refresh every N s" updates the orbitals). Without an
//! auditor (the CPU teacher) there is no producer and the estimated solver shows no orbitals.
//! The orbital shares' free-atom tables (`set_share_tables`, orbital-atom-shares spec) travel the
//! same way: kept here, sent with every job and handed to the auditor's teacher before it solves
//! (and to the stepping teacher, which shows no orbitals). The teacher weights them at the
//! positions it evaluated — the job's scene, not the driver's newer one (Review Focus 4) — and a
//! job whose species are not the tables' gets no shares (`crate::orbitals`, "Orbital shares").
//!
//! The same job slot solves the bond detector's **isolated-atom references** (M6 Task 3 review
//! I2; [`request_atom_reference`](SurrogateKs::request_atom_reference), taken with
//! [`take_atom_references`](SurrogateKs::take_atom_references)): `sim::references::solve_atom`'s
//! multiplicity scan on the auditor's teacher, never on the stepping teacher (whose warm state is
//! the molecule's and is never reset for an atom). Each accepted request is answered exactly
//! once, `Cancelled` when a Forget or an installed model abandoned it. Without an auditor the
//! driver solves them on [`ForceProvider::side_solver`] instead — the teacher's, a fresh instance.
//!
//! # Trust rule (from the Task 3 fit, binding)
//!
//! The model may answer only when the last `FitReport` has `n_samples ≥` the independent columns
//! its fitted atoms touch, |∪_k S_k| (`sample_floor`; since M8a user-fix 3's review — before it
//! was the whole basis's `q_prime`, which a species held in a few samples raised by columns no
//! other atom touches; Q′ still without a class judgement; `calibrate` "The sample floor per key")
//! (distinct geometries, not rows: the 3N force rows of one sample probe one geometry) AND
//! `residual_dof_e ≥ 30` AND `residual_dof_f ≥ 30` (the two noise levels are then known to
//! ≈ 25 %, so the variance means something). Below that the fit is a curve through too few
//! points with a noise estimate collapsing towards its floor — overconfident inside its span.
//!
//! **Force noise** (M6 Task 11): AND the fit's force-label noise `sigma_f ≤ force_std_abs`
//! (2e-3 Ha/Bohr; reason "force noise σ_f … > 2.0e-3 Ha/Bohr"). σ_f is the fit's homoscedastic
//! force residual — label noise and basis misfit together, one number for every geometry. A
//! constant carries no geometry-dependent information, so it cannot inform a per-step decision
//! (rule 6 leaves it out) and is gated once, globally, here (physics review, round 3, M2). With
//! both gates an answer's expected per-component force error is √(s² + σ_f²) ≤ √(b² + σ_f²) ≤
//! √2·b at the 2e-3 floor (s the epistemic std, ≤ b by rule 6). The user's model
//! (H₄ on 49³, every sample box-affected) had 9.6e-3 and was trusted (the relative training-
//! residual bound passes noise of up to 10 % of |F|, capped at 1e-2); the validated fits have
//! 4.5e-5 (H₂ pass) and 2e-5 (H₃⁺). A failing noise condition triggers the same recovery as
//! failing training residuals (below): the worst offenders are ranked against the absolute
//! `force_std_abs` then (`worst_offenders`), so the samples that put σ_f above the tier are the
//! ones quarantined, within the usual 10 % cap — beyond it the model stays untrusted (honest: a
//! store that is mostly noisy, e.g. box-affected, is not a model).
//!
//! **Training residuals (ruling S8).** The posterior cannot see basis misspecification: with
//! n_max = 8 and a teacher whose pair energy is not ≈ 0 at r_cut, one training point outside
//! the window made the two-noise evidence absorb a 1e-2 Ha energy offset as "noise" (s_E
//! 8e-4/atom, a 6σ outlier) while the forces and the epistemic stds stayed tiny — no
//! Gaussian-noise criterion fires on that. So after every refit one pass over the store
//! evaluates the model at its own training data, and trust additionally requires
//! max_i |E_model − E_i|/N_i ≤ `energy_std_per_atom` and, over every atom,
//! ‖F_model − F‖/√3 ≤ the policy's force bound for that atom (the same numbers rules 6–7
//! use). "The fit reproduces its own data to the accuracy it claims" is what a model must show
//! before it may answer; [`SurrogateKs::training_residuals`] exposes both maxima and the note
//! says which failed ("surrogate untrusted: max training energy residual 5.0e-3 > 1.0e-3
//! Ha/atom"). A failed audit (rule 8) is the fourth condition, until a refit's model predicts the
//! audited geometry without the audited sample (rule 8, "A failed audit stays failed").
//! [`SurrogateKs::trusted`] exposes the verdict; every evaluate of an untrusted model is a
//! Fallback (rule 4) and the samples accumulate until a refit earns trust.
//!
//! # Class-wise trust (M6 Task 11 round 6, controller ruling; per atom since 2026-10-18)
//!
//! The training residuals ask whether the fit reproduces data it was fitted *to*; the round-5
//! strict hot run showed that is not enough: a trusted H₂ model with a few dozen kept H₃
//! fallbacks answered three-body geometries at interpolation grades (γ 0.86–1.16) 3.1–6.2× off
//! in force — the many-body samples were reproduced, but a fit without any one of them could not
//! predict it. So each refit (and installed model, and policy change) also asks the
//! leave-one-out question, per **environment class**:
//! - **Class rule** (universal, nothing per element, no constant beyond the model's r_cut): per
//!   **atom** its neighbour count within r_cut — 0 **isolated**, 1 **pair**, ≥ 2 **many-body**
//!   ([`atom_class`]). Until 2026-10-18 a whole geometry took its largest atom's class
//!   ([`env_class`], kept as a readout); the user's water + far H₂ then counted entirely as
//!   many-body, the pair class held 0 samples, and the far H₂'s pair environments were never
//!   judged by the samples that contain them. Now a sample is evidence for every class among its
//!   atoms, and a query must have every class among its atoms trusted.
//! - **LOO residuals**: exact, closed-form, at the fit's own hyperparameters and
//!   standardisation, for the whole sample (its energy row and 3N force rows) at once:
//!   r_s = (I − H_ss)⁻¹ e_s with H_ss = Φ_s A⁻¹ Φ_sᵀ W_s (derivation and cost: `surrogate::loo`;
//!   verified against an explicit downdate of the fit to 2.4e-12). Over the fitted samples (box-excluded,
//!   quarantined, superseded and held-out draft guesses skipped, as the training residuals).
//!   **Once per refit, for the fit that stands** (2026-11-12, `references.md` "Refit cost at
//!   scale"): the fits inside one refit — the fence votes', the release's, the trust recovery's
//!   cap retries and quarantine rounds — decide on the training residuals, the force noise and the
//!   sample floor only, so they leave this pass, the active set (rule 5e) and the null space (rule
//!   5d) to the fit that stands (`fit_store_deferred`, `ensure_trust_passes`; before the held-out
//!   guesses are judged, which read rules 5d and 5e). The verdicts are bit for bit those of a pass after every fit
//!   (`a_deferred_refit_leaves_the_eager_refits_state`); only a teach-around round count now
//!   restarts by the standing fit's class verdict, not by an intermediate fit's that the recovery
//!   replaced. The samples' leave-one-out runs on `surrogate::threads::worker_threads` threads
//!   (80 % of the logical CPUs — the user's choice —, at most 16, each below normal priority — 2026-11-15, the
//!   user's PC was laggy at 99 % CPU; `surrogate::loo::leave_one_out_many`, the same numbers in
//!   the same order); a background refit's own thread runs below normal too.
//! - **Trusted class — the calibrated, hysteretic verdict** (M8a Task 2 and its fix rounds 1–2, spec
//!   §4.1; the derivation and the constants: [`calibrate`] module doc; the numbers:
//!   `validate/references.md`, "M8a calibrated verdict"): enough samples — per environment key
//!   (M8a user-fix 3 and its review, `calibrate` "The sample floor per key") the fitted samples
//!   holding an atom of the key, at least as many as the basis columns its atoms touch (not the
//!   class's count against the whole basis's Q′: a species added to a collection must not raise the floor of
//!   keys whose columns it does not touch; the user's water read "967 samples < Q′ 1590" once N₂
//!   was trained in) — and the error the teacher **measured** where this verdict decides
//!   within the verdict's limit. The evidence is the audit history's points in the verdict's
//!   jurisdiction — audits, and the fallbacks rule 4b itself refused (never rule 5/5e's, which
//!   refuse whatever the class verdict says, nor a class refusal rule 6/7 would have refused as
//!   well, coded apart as `HISTORY_RULE_CLASS_AND_VARIANCE`) — one draw per point (ΣF = 0: a dimer
//!   is one draw), a class's rule-4b points weighing together at most as much as its audits (never
//!   fewer than three; the count formed in integers, fix round 2). **The evidence ages by model
//!   generation** (fix round 2): per class the window is the newest whole generations holding ≥
//!   `WINDOW_POINTS` = 300 of its points — a point measures the model that judged it, and once the
//!   newer models' own comparisons fill a window the older model's are aged out (a class refused
//!   on its audits re-enters after a teaching refit and 300 newer comparisons instead of after 4096
//!   later points; more fallbacks of the refusing model never outvote its audits). 300 so that a
//!   window free of 3× atoms makes a 1 % tail of them unlikely (0.99³⁰⁰ < 5 %) — a 12-point
//!   window aged a refused many-body class's bad first points out behind benign ones and let it
//!   answer the gated repro's hot run 27 times beyond 3× the bound (`calibrate`, "Why 300"). Per class: W_c = rms over those points of the per-atom ‖F_teacher −
//!   F_model‖/√3 over the atom's bound `force_bound(|F_teacher|)` (the variance rule's: max(2e-3,
//!   10 %|F|) capped at 1e-2; |F_teacher| is stored per atom in the history for it), W_E,c = rms
//!   of |E_teacher − E_model|/N over `energy_std_per_atom`, each × (1 + 1/√(2n)) — one sigma of
//!   the estimate's own scatter over n points; the verdict's force value is max(R_c, W_c), R_c the
//!   RMS over the class's atoms of the LOO force error ‖r_F,j‖/√3 against its bound — the model is
//!   never trusted beyond its own leave-one-out — and its energy value W_E,c. κ_c = the rms of
//!   z = ‖ΔF‖ / the predicted std (floored at 1, inflated the same way) is **reported** as the
//!   posterior's calibration and decides nothing: a product κ·R overstates the error for a sparse
//!   class by the leverage ratio and z has no floor on its denominator (the fix round's physics
//!   review, I1/I2). **Hysteresis**: an untrusted class enters at both values ≤ `ENTER` = 1/√2
//!   (half the bound's variance) and a trusted one stays while both ≤ `LEAVE` = 1 (the bound
//!   itself) — a √2 band, equal to the estimate's one-sigma scatter at n = 3 and wider from n = 4,
//!   so refit-to-refit jitter cannot flip it; the memory is the verdict that stood when the refit
//!   began (`class_memory`, `ClassLoo::trusted_since`), reset by a forget and **saved with the
//!   model** (the sidecar's `class_memory`, fix round 2: a reload, a worker re-spawn and a resumed
//!   training keep a band verdict). **The worst-atom guard** (`WORST_ATOM` = 3; fix round 2 at
//!   every n): an untrusted class does not enter while an atom of its window was measured beyond
//!   3× its bound (P(χ²₃/3 > 9) = 5.9e-6 for a calibrated posterior at the bound: wrong, not
//!   unlucky) — an rms over a heavy-tailed error's benign early points does not bound its tail.
//!   **Below three points of evidence** (`MIN_EVIDENCE`) the posterior's claims are unverified and
//!   the rule of before stands unchanged — RMS ≤ 1 **and** max ≤ 1 on the force, LOO energy max ≤
//!   1 (`trust::class_refusal`) — and the guard refuses both ways. **The evidence floor** (trust
//!   tightening, spec 2026-11-18 §3; `calibrate`, "The evidence floor"): the calibrated verdict
//!   opens a key alone only once its window's attributed verdict points carrying a fit id span
//!   `FLOOR_FITS` = 12 distinct fits **and** `FLOOR_POINTS` = 60 points; below that, with
//!   evidence ≥ 3, the rule of before must pass as well (`ClassKey::Floor`, named when the
//!   calibrated verdict itself passes) and the worst-atom guard applies both ways. Saved models
//!   and collections are re-judged at load, no grandfathering: legacy points carry no fit id, so
//!   a key saved open with LOO max > 1 loads closed. Nothing is loosened without evidence, and a
//!   class the max refuses for one outlier earns the calibrated verdict from its own rule-4b
//!   fallbacks once they meet the floor (the Morse fixture: 201 samples with one at 3.6 Bohr, LOO
//!   max 1.08×, enters after 120 fallback steps — the 12th fit; 10 before the floor — and holds:
//!   `a_class_refused_for_one_outlier_earns_the_calibrated_verdict_and_holds`). The max over the
//!   class is gone where the floor is met: the verdict is about the expected error over the
//!   class's environments (what rule 6's "std ≤ bound" claims per step), measured by the teacher
//!   at the geometries the class would answer. A sample whose LOO is undefined (I − H_ss singular)
//!   counts as infinitely off (RMS ∞); a comparison in the evidence that is not a number makes W
//!   and κ infinite. The refusal names the term that decided (`ClassKey::Calibrated` the measured
//!   force, `LooFloor` the leave-one-out floor, `MeasuredEnergy`, `WorstAtom`, `Floor` the evidence
//!   floor; the uncalibrated `Force` / `Energy`). The same verdict holds for the user's #344 shape (a compressed H–H in
//!   H₃⁺): a many-body class the max rule refuses for one compressed trimer (LOO max 1.57×)
//!   enters on its own rule-4b comparisons and holds (`a_compressed_trimer_does_not_flip_the_many_body_verdict`).
//! - **Per environment species** (M8a user-test fix 2, the controller's rulings; `calibrate` "Per
//!   environment species"): the evidence window, W, λ, the worst-atom guard, the hysteresis
//!   memory and the leave-one-out force/energy numbers are kept per key (class, Z, the sorted
//!   neighbour species within r_cut — Z a key only), and (since M8a user-fix 3) the sample floor:
//!   the key's own columns, not the whole Q′; an atom answers
//!   by its own key's verdict and a refusal names it ("many-body around H, with H"). The class's
//!   `ClassLoo` is all its keys trusted. A history point recorded before species were kept counts
//!   for every key of its class. New verdict evidence (an audit or a rule-4b point) re-judges the
//!   verdicts at once (`rejudge_classes`), not at the next refit. The user's case: one hot-H₃ point
//!   at 5.66× its bound kept water's many-body environments out of a 1724-sample H/O collection.
//! - **The isolated class is exempt**: in a scene of two or more atoms rule 5a refuses it
//!   anyway, and a one-atom scene's answer is E_ref, the teacher's own isolated atom (S9).
//! - **4b fallbacks**: kept unthinned while the key is short of samples, thinned like rule 5
//!   otherwise, and never a copy of a stored sample (`is_label_copy`; round 1 of the
//!   2026-10-18 review, MUST-2). An interim version never thinned them (the user's water + H₂ showed
//!   every 4b fallback "skipped: similar"); the review showed that copies of a parked geometry add
//!   rows without residual — they shrink the fitted force noise, and with it rule 6's bound,
//!   everywhere — and do not reach a class's blockers (the user's #345 is O1H3, a composition the
//!   O1H4 scene never visits). Learning at the blockers is the teach-around pass's job ("Teaching
//!   around the blockers" below).
//!
//! The per-class numbers (samples, LOO force RMS and max ratios, energy RMS and max ratios, the
//! calibration factor, the measured errors, the worst measured atom and the evidence, the verdict
//! and since when) are in
//! `TrainableStatus::class_loo` for the Model tab; the readiness readout names the class's worst
//! sample and its origin. **Costs of the max** (measured, rule 7; the uncalibrated rule, which
//! stands until a class has three audited comparisons): one sample alone in its region decides
//! its class — the GPU repro's kept tail sample at r = 5.125 (LOO force 14.9× its bound)
//! untrusts the H₂ pair class until the region fills in (before M8a Task 2 all 1600 heating steps
//! fell back; the M8a baseline measured those refused steps at z rms 0.86 — the model was right
//! there — which is what the calibrated verdict now reads), a first H–He sample (LOO energy
//! 1.1e3× the tier: no other row has an H–He feature) untrusts H₂ with it until H–He is learned
//! (`new_species_falls_back_and_is_learned_at_the_next_refit`: trusted again with 40 H–He
//! dimers), and the user's water + H₂ file (657 samples, re-judged from its store on load) is
//! refused at its own scene: pair 179 samples, LOO force 12.0× from #345 (a lone added H),
//! many-body 587, 10.6× from #438 (a stretched H–H scan point). All are teacher steps, never
//! wrong answers; under the calibrated verdict such a class earns trust from its own fallbacks
//! once three of them show the model within its claims there.
//!
//! # Teaching around the blockers (2026-10-18, the physics review's ruling 5)
//! A targeted pass (`teach` module doc; `Trainable::start_teach`, the app's "Teach around N
//! blockers") solves ≈ 10 geometries per blocking sample — 6 nudges and 4 points toward its nearest
//! fitted neighbour of the same atom sequence, each beyond the thinning radius from the blocker and
//! from each other, 2 of them held out — and accepts only on the **unchanged** class rule with the
//! blocker still in the fit, plus the held-out checks at the audit criterion; at most two rounds per
//! environment class. A blocker whose own in-sample residual is beyond its bound is refused ("the model cannot
//! represent it"). Measured on the Morse teacher: a lone sample 1.6 Bohr beyond the data (leave-one-
//! out 5.2×) is accepted after 9 solves; the tail sample at 5.1 Bohr (1.05×) is **not**: once its
//! neighbourhood is sampled the fit cannot carry the Morse binding left at r_cut (V(6) ≈ −3.4e-3 Ha
//! against f_c-damped features that vanish there), every residual moves to 1.9× the energy tier, and
//! the outcome says so — a leverage outlier near the cutoff can turn out structural. GPU: H₂ at 49³
//! with one sample at 3.6 Bohr beyond 1.1–2.0 (leave-one-out 113.9×) → accepted after 10 solves (max
//! 0.10×). The user's water + H₂ file (95³, cores-only O): two rounds, 64 solves, 896 s, **not
//! accepted** — #345 itself became predictable (pair max 12.0× → 0.78×), but the refit that the new
//! samples allowed raised the basis to Q′ 344 (pair class then short of samples, 201 of 344), and
//! the many-body class moved 10.6× → 19.6× (a round-1 sample, #664) → 1.47×, with one held-out
//! check failing at 1.66× (a failed audit: untrusted until re-earned). The review's fix round
//! (MUST-T1, SHOULD-T1–T6): acceptance needs every class with samples trusted (a short class is its
//! own outcome, "needs samples"); geometries never farther from the data than the blocker; path
//! partners atom-matched; held-out checks kept after judging (a failed one with its index and the
//! origin "teaching check"); the offer states the sample floor of a richer basis and the risk; the
//! round count restarts after a regular pass or once the class is trusted.
//!
//! **Rejected (2026-10-18, measured): fencing the outliers.** Setting aside up to 10 % of a class's
//! samples the rest cannot predict and refusing only the queries whose nearest training sample is
//! one of them made the user's file trusted at its scene — and the stuck-escape repro's hot run
//! (`the_stuck_escape_repro_is_answered_honestly`) then answered H₃ geometries up to **6.33× the
//! force bound and 6.2e-3 Ha/atom** off (182 judged; the number of record is 2.7× and 5.2e-4): the
//! outliers mark where the fit is unreliable *between* samples too, not only next to themselves.
//! The max rule stood until M8a Task 2's calibrated verdict (above), which differs from that
//! experiment in both respects that made it fail: the outliers stay in the class's RMS (nothing is
//! set aside), and a class is trusted beyond the max rule only on **evidence** — the teacher's
//! comparisons at visited geometries of the class, which is where that experiment's answers were
//! off (the M8a baseline's many-body fallback points: z rms 12–14) and where a measured error
//! beyond the limit, or one atom beyond 3× its bound, keeps the class refused.
//!
//! **Ruling (shell → fence)** (2026-10-19, the coordinator on the round-4 LiH measurement): a
//! failed shell check (`sampling` module doc, 3.) is a measured **structural** limitation of the
//! cutoff, not a model error a refit can cure. It becomes an audit fence of origin "shell check" at
//! once — no global untrust, no wait for `AUDIT_REJUDGE_LIMIT` refits: strict mode refuses
//! geometries in which a pair of those species lies in the shell band 5/6·r_cut ≤ r < r_cut (and
//! the case's quarantine-radius neighbourhood, as every audit fence), naming the limitation; the
//! pair and class trust verdicts are unchanged and everything outside the fenced shell answers as
//! before. Final review (2026-10-19) C1/I1/I2: the fenced pair is the check's own species pair,
//! carried on the case and in the file (sidecar format 5) — the band is refused for it wherever
//! the check point lay (a short user plan can put it at or beyond r_cut) and a bystander pair of
//! the check's geometry is never fenced; a shell fence is **never lifted** by a refit's or an
//! install's one-point re-judgement (only a later pass's shell check measures the shell again);
//! and a fallback inside a fenced band is used for its step and **held out of the fit**
//! (`keep_sample: false`, the note "shell fence — … not fitted"), whichever rule named the
//! refusal: a fitted band row is the structural outlier R3-1 stopped fitting, and a run through
//! the band would otherwise flood the store with them. Measured (LiH, Li–H 3.0 Bohr, default
//! pass seed 7; `lih_shell_check_measurement`): at
//! 81³ × 0.25 (a 20 Bohr box) the shell point Li–H 5.10 Bohr fails — forces 1.73× their bound
//! (4.2e-3 > 2.4e-3 Ha/Bohr), energy −6.2e-4 Ha (0.31× the tier); at the preset's 57³ × 0.25 it
//! is box-affected (its density reaches the faces) and not judged.
//!
//! **The knife edge (the M6 handoff item; closed by M8a Task 2's calibrated verdict, above — the
//! record of what it was).** Under the max rule one
//! sample decides its class, and the verdict has no hysteresis: in the user's live water + 2 H
//! session (2026-10-18, the preview build) a model trusted for ~16 min kept one audit sample, the
//! refit moved the many-body class's worst leave-one-out to 1.0× — #251, a pass scan point with
//! H–O at 5.97 and 6.03 Bohr, right at r_cut — and 4b refused the class everywhere: every step DFT,
//! every fallback "skipped: similar" (the geometry was already held), so nothing local could change
//! it. The class rule is not changed here; what is: the pass no longer fits the outer sixth of the
//! cutoff (`sampling` module doc, 3., `SCAN_CUTOFF_FRACTION` — where such chronic outliers came
//! from) and measures it instead with one shell check per crossing pair, and consecutive class-rule fallbacks far from the blocker prompt the teach-around offer
//! (`teach::TEACH_PROMPT_STREAK`; one click, never automatic). The calibrated class verdict that
//! closes it (hysteresis, the measured error, evidence by generation) is M8a Task 2's, above;
//! its false-trust rate on the stuck-escape repro is in `validate/references.md`, "M8a calibrated
//! verdict".
//!
//! # Trust recovery and quarantine (ruling S15, Task 8 fix round 3)
//!
//! The training-residual condition is a maximum over the store, so ONE sample the basis cannot
//! represent — a dimer pulled apart beyond the cutoff, where every feature vanishes and the
//! model can only say Σ E_ref while the teacher's restricted-spin dissociation lies far above it;
//! a stray hot-burst geometry — untrusts the whole model at the next refit, and before this
//! round nothing ever gave the trust back: every later step was a rule-4 fallback whose sample
//! (kept on every step) could not repair the maximum. So when a refit leaves the training
//! residuals above the policy's thresholds, [`refit`](SurrogateKs::refit):
//! 1. retries the fit once with `degree_cap = 6` on a body-3 basis (the documented setting for
//!    bent molecules with wide bond excursions) and keeps it if the residuals then pass **and the
//!    trust rule's counts still hold** (n ≥ Q′, both residual dof ≥ 30; since 2026-10-18: Q′ grows
//!    with the cap — 176 → 344 for H/O, 29 → 51 for H — so a retry that fits one off-manifold
//!    sample could trade a residual failure for a sample shortfall: the water investigation's
//!    E1(e), one H 0.9 Bohr from O → n 333 < Q′ 344, every step DFT until 11 more samples, then
//!    the many-body class untrusted by that one sample; the quarantine below handles it instead);
//! 2. otherwise **quarantines** the worst offenders and refits without them, up to
//!    [`QUARANTINE_ROUNDS`] rounds, never more than [`QUARANTINE_MAX_FRACTION`] of the store (the
//!    cap is checked before a round holds anything). **The offenders are ranked by their
//!    leave-one-out ratio** (since 2026-11-12, spec `2026-11-12-sample-recovery-design.md` R1;
//!    `quarantine::loo_ratio`): how far the fit *without* the sample misses it — the larger of its
//!    leave-one-out energy residual per atom over `energy_std_per_atom` (when its energy was
//!    fitted) and, over its fitted atoms, the leave-one-out force residual
//!    `force_residual(F − r, F)`'s d over the atom's bound — the class rule's own reading of the
//!    same `surrogate::loo` residuals. A round takes every fitted, not box-affected sample whose
//!    ratio exceeds 1 and is within a factor 2 of the worst. The bound is each atom's own while the
//!    training residuals fail, `force_std_abs` when only the force noise does. An undefined
//!    leave-one-out (the sample alone spans a direction of the fit) is ∞ and ranks first (ruling
//!    R1a: as the class rule reads it). The ratios are the class pass's (`refresh_class_loo`),
//!    cached per fit (`LooRatios`, keyed by the model generation and the fit's own members; ruling
//!    R1b) — the fits inside a refit defer that pass, so a round computes them once for its own fit
//!    (`ensure_loo_ratios`; ≈ 1 s at Q′ 1 032 and 1 700 samples on 16 threads). Until then the
//!    ranking was the in-sample training residual, which a **high-leverage** sample hides from: the
//!    fit reproduces it inside its own (wide) bound at its neighbours' expense. The user's
//!    collection (2026-11-12, `references.md` "Leave-one-out recovery ranking"): one compressed OH
//!    at 0.72 Ha/Bohr (bound: the 1e-2 cap) bent the shared O–H pair term and 34 H₂O/H₃O samples
//!    failed; the rounds held those victims, round after round, and ended untrusted at 165 held,
//!    while the sample's leave-one-out ratio was 962× (rank 1 of 1 700; the next 5.2×). Pinned on
//!    an analytic fixture (`tests/recovery_loo.rs`): one force-only compressed H₂ among H₃ and H₂ —
//!    the training ranking's rounds hold 7 victims and do not recover (the refit gives them back,
//!    no ratchet), the leave-one-out ranking holds the
//!    outlier alone and trusts. The recovery note names the worst ratio held ("3 samples
//!    quarantined — the fit without each predicts it at up to 38× its bound, and the fit without
//!    them recovers (…)"; "…, and the fit without them passes" when its residuals and noise pass
//!    short of the trust rule's counts). **The fallback** (fix round 1, the coordinator's
//!    option A; `quarantine_rounds`): when the leave-one-out rounds do not recover the fit (the
//!    residuals, the noise and the trust rule's counts), the state the rounds started from is
//!    restored (the fit, the codes, `species_dirty`, the note) and the rounds run again ranked by the
//!    in-sample training residual (`training_offenders`, the ranking before); that result stands only
//!    if it recovers, else the leave-one-out one does, and the note names the ranking that stood
//!    ("ranked by their own training residual (the leave-one-out ranking did not recover the fit)").
//!    Step 3 (below) then runs whenever the result standing after step 2 is not recovered —
//!    residuals, noise **or the counts** (fix round 2, I1: before, only while the residuals or the
//!    noise failed, so a leave-one-out result that passed its residuals short of samples suppressed
//!    the default-basis retry; `step_3_runs_when_step_2_passes_its_residuals_but_is_short_of_samples`).
//!    With both, the recovery is never worse than before, for up to [`QUARANTINE_ROUNDS`] more fits
//!    per ranking on a refit that is failing anyway. The note counts the samples **this refit**
//!    held and measured ("K samples quarantined this refit (N held in all) — …"), against each one's
//!    own bound or, in the noise-only mode, "the 2e-3 Ha/Bohr force-noise tier". Why both: once a label error is held, the leave-one-out ratio
//!    can go on naming high-leverage samples whose own fit is fine — the stuck-H collection's step 3
//!    (below) ends untrusted at 47 held by it, trusted at 38 held by the fallback. On the user's
//!    pre-1805 snapshot #1804 is held in round 1, but the refit's release step (run first, against
//!    the fit #1804 bends) puts #1455 back: the first refit ends untrusted at 1.0185× (128 held,
//!    the fallback 165 held at 1.46×), the next trusted at 0.9950× (122 held) — measured before
//!    the no-ratchet rule below. **No ratchet** (spec 2026-11-13 §4.2, wrapping the whole
//!    recovery since the open-shell merge): a result stands when it recovers, or short of the
//!    counts when it passes the residuals and the noise; when no step (the leave-one-out rounds,
//!    their fallback, step 3) passes the residuals, the refit ends with the codes, the degree cap
//!    and the fit it entered with — a forgotten sample's code untouched — so nothing is held or
//!    written to a collection, the note says "recovery did not restore trust; no samples held"
//!    (`settle_recovery_note` rewords it when the refit ends trusted), and the samples still over
//!    their bounds are the misfit set M (`trust` module doc). Inside `quarantine_rounds`, when
//!    neither ranking recovers, a leave-one-out result that fails the residuals yields to a
//!    training-ranking one that passes them (so the wrapper never discards what the ranking before
//!    R1 kept). The holds of a recovery that passed its residuals stay (and are written to a
//!    collection): a sample held by
//!    the leave-one-out ranking is one the fit without it misses, so a later refit's release by
//!    reproduction rarely takes it back. A held sample leaves the hold only when a later fit
//!    reproduces it (released into the fit) or a fresh agreeing teacher answer next to it
//!    supersedes it — which ends its fence but keeps it out of the fit. The user's Restore does not
//!    apply to it: Restore clears only a forgotten sample, and only a code-0 sample can be
//!    forgotten. Not changed: the
//!    rounds, the cap, the degree-cap steps, release and supersede, and the trust rule that judges
//!    every fit;
//! 3. (M8a user-test fix) a `degree_cap` an earlier recovery set is **not kept for good**: when the
//!    capped fit fails, the default basis is tried first exactly as step 1 tries the cap (kept if
//!    it alone passes, counts included), and when the quarantine on the capped basis cannot
//!    recover either, the same quarantine rounds run again from the default basis (from the state
//!    the refit entered with) — that result stands only if it is trusted, else step 2's does when
//!    it passes the residuals, else the entry state (no ratchet, above). The
//!    case (`references.md` "M8a stuck collection"): one stretched-H₂ label under cap 6 drove the
//!    energy noise to 2e-2 Ha/atom and spread its misfit over the H₂ samples (379 named as
//!    offenders, re-measured 2026-11-12; 380 when first recorded), so step 2 named
//!    no offender within its cap and the cap could never be undone — untrusted at every refit. A quarantined sample stays in the store (and in
//!    the sidecar, `sample_quarantine`); it is excluded from the fit, the envelope and the
//!    residuals; a geometry within [`QUARANTINE_FRACTION`] · `max_descriptor_distance` training
//!    spreads of one falls back ("near a quarantined geometry") — whichever rule fired first — and that
//!    answer is not kept while it disagrees with the model (it would only be quarantined
//!    again, until the cap untrusted the model for good: Task 8 fix round 4) — except by a
//!    bounded fence vote (below); `ForgetModel`
//!    clears it. The trust rule is unchanged — what is excluded is said: `quarantined()` and the
//!    recovery note.
//!
//! **Release.** A fresh teacher answer near a quarantined sample that *agrees* with the model
//! (energy within `energy_std_per_atom` per atom, every atom's force within its bound) settles
//! it: the answer is kept, and the quarantined sample is released back into the fit if the
//! model now reproduces it too, or else marked **superseded** — contradicted by a fresh answer
//! at the same geometry (a teacher glitch): still held, still excluded, no longer fencing its
//! neighbourhood. A refit releases any quarantined sample the new model reproduces.
//!
//! **Near a misfit** (spec 2026-11-13 §4.3; the misfit set M is the `trust` module doc's): a
//! sample's *misfit atoms* are, for a force misfit, its trained atoms whose force ratio exceeds 1,
//! for an energy misfit every trained atom (`Misfit::atoms`). A query atom is near a misfit when
//! its descriptor is within the near-quarantine distance ([`QUARANTINE_FRACTION`] ·
//! `max_descriptor_distance` training spreads, the per-atom L∞ distance the quarantine fence is
//! built from) of a misfit atom of its species (`near_misfit_atoms`). A geometry is refused when
//! **any** atom is near (`TeachRule::NearMisfit`, after the quarantine fence, naming the first such
//! atom) — broader than the quarantine fence's all-atom test, deliberately; `atom_verdicts` refuses
//! exactly those atoms. The teacher's answer there takes the near-quarantine admission without its
//! votes: kept (on the normal path) only when it agrees with the model, else not kept
//! (`TeachOutcome::MisfitDisagrees`), so a region the model cannot fit stays refused without a
//! parked scene growing M. The fence lifts when a refit fits M's samples (a larger store, a higher
//! body-4 degree): M is recomputed at every fit. Sampling-pass answers keep their own path. The
//! fence stands only while M is within its limit (`misfit_fence`): beyond it the model is
//! untrusted as a whole (the `Misfit` row; rule 4 refuses every step) and its answers are kept as
//! any untrusted model's are — fencing an M that may be the whole store would refuse, and never
//! keep, the data a later refit could fit it with (a ratchet; spec §4.2).
//!
//! **Forgotten samples** (spec `2026-11-12-sample-recovery-design.md` R2; `Q_FORGOTTEN` = 7, its
//! doc in `surrogate::quarantine`): the user's Forget in the Model tab's "Worst-fitting samples"
//! (`forget_sample`; the list is `sample_suspects`, the fit's own leave-one-out ranking). Excluded
//! like a held sample (fit, envelope, grade, leave-one-out) and fencing like one, but never
//! released by a refit nor superseded by a fresh answer — only `restore_sample` clears it — and
//! counted with the held samples against [`QUARANTINE_MAX_FRACTION`]. The status is written at
//! once (a collection status record, else the store's code for the model file); a background
//! refit that lands after it re-applies it (`background` module doc, "The user's codes"). Only a
//! sample in the fit (code 0) can be forgotten, so Restore returns it to exactly that; the buttons
//! carry a checked identity (`sim::SampleId`: a changed sample list is refused). A forget never
//! touches the trust rule: a failed audit whose sample is forgotten is re-judged at every refit
//! as before (rule 8, Task 2 fix round 1 I1). A batch (`forget_samples` / `restore_samples`,
//! 2026-11-14) is all or nothing on its ids, and a batch Forget past the cap is applied with a single
//! Forget's warning; every change
//! counts in `user_changes_pending` until a fit holds it (the worker debounces the refit on it). An agreeing teacher answer next to it — where nothing
//! is released or superseded — is an ordinary one: thinned and copy-checked like an envelope (rule
//! 5) fallback's (`admit`; final review I1, `agreeing_answers_next_to_a_forgotten_sample_are_thinned`),
//! not a new sample at every step — and a vote toward lifting its fence (below).
//!
//! **Lifting a forgotten sample's fence** (2026-11-14, user-approved trust change; the user's 45
//! forgotten H₂ perturbation samples fenced ordinary H₂ geometries, every step fell back with
//! |ΔE| ≈ 0.9 mHa — inside the model's bound — and was thinned, forever). The mirror of fence voting:
//! a fresh teacher answer near a geometry whose fencing samples are **all forgotten** that **agrees**
//! with the model (`agree`, the same test a release uses) is an agreeing vote (`lift_votes`, the
//! answer with its residual teacher − model and the model generation) — but only from a model
//! fitted **without** them (review Critical 1, 2026-11-14: a Forget sets the code at once while the
//! refit runs later, and a model still trained on the sample agrees next to it by construction):
//! no vote is cast, and the run starts over, while a refit is due (`pending_refit`), a background
//! refit runs, or any of the fencing samples is still a member of the standing fit
//! (`in_standing_fit`). When [`FENCE_VOTES`] consecutive agreeing votes have the same atom count, the
//! same model generation, a forgotten sample **every one of them** was next to, and agree with each
//! other (`votes_agree`, on their residuals), the fence of the forgotten samples all of them were
//! next to (the intersection — review Important 2: never one a single vote was near) is **lifted**: code
//! `Q_FORGOTTEN_UNFENCED` (8, persisted like 7 — a collection status record written at once, the
//! sidecar's `sample_quarantine`; files from before read as not lifted), and the last answer is
//! kept (`TeachOutcome::FenceLifted`, not thinned) so the region has fresh data; the model answers
//! there again. A lifted sample stays forgotten — out of the fit, never released or superseded,
//! listed with Restore (which returns it to code 0, clearing the lift) and purged by a purge — it
//! only stops fencing. A disagreeing answer next to the fence clears the run (and votes as before:
//! [`FENCE_VOTES`] consistent disagreeing answers keep one, `KeptByVote`); a refit, an install or a
//! load clears it too (the residuals were the old model's). A fence that also holds a quarantined
//! sample behaves as before — the agreeing answer releases or supersedes the quarantined part, the
//! forgotten part keeps fencing, no lift vote is cast and the run starts over. Rule 8 is untouched: a failed audit is
//! re-judged at every refit, whatever its sample's code.
//! **Purge** (2026-11-14, `purge_forgotten`, the `purge` module doc): deletes the forgotten samples
//! for good — from the store and the shared collection's file — once the refit after the last
//! Forget has run; no refit, the fit bitwise unchanged, every index the provider holds remapped.
//!
//! **Re-solved samples** (code 9, `Q_RESOLVED`; Fermi–Dirac spec §7.5–7.6 and §8, plan 2026-11-13 Task 15): a sample
//! the collection migration re-solved for the new solver whose state changed (class S). Out of the fit, the envelope,
//! the grade, the leave-one-out and the class statistics like a held sample (`excluded` is "code ≠ 0"); **not**
//! counted by the recovery's 10 % held cap (`quarantined() + forgotten()`) nor by `fencing()` — it is no misfit hold —
//! and **not the user's**: Forget and Restore refuse it. Before its review it does not fence: the fit simply has no
//! data there. **The review** (`review_resolved`) runs inside a refit, after its first fit and before the trust
//! judgement (votes, release, recovery), only once the migration pass is complete (`migration.txt` at "complete" or
//! "reviewed": every sibling present), while a code-9 sample remains and the store changed since the last review: a
//! candidate is fitted on the fit's members plus every code-9 sample, each one's leave-one-out ratio in it
//! (`loo_ratio`, against its own bounds) is read, ≤ 1 releases it (9 → 0; the refit is rebuilt with it and the trust
//! machinery judges it as any new data), > 1 keeps it at 9. A **kept** one is a stored label the model cannot
//! reproduce: it fences like a held sample (`fences()`, F4; the fence's on-switch is `any_fences()`, controller ruling
//! A, since `fencing()` does not count it) — but never joins M (it is not fitted), is never code 1 (no ratchet), and
//! purge's lift votes never lift it (they take code 7 only). Like a held sample, one fresh teacher answer next to it
//! that agrees with the model settles it in `admit` — released when the model reproduces it, else superseded
//! (controller ruling B) — and a later review can still release it. Its verdict is in `migration.txt`'s review log
//! (collection indices, the latest per sample), read back at attach (ruling R13), renumbered by a purge (F3), and
//! written at the refit's end after the status records it names. Judged by the review: code-9 samples whose box has
//! its references and that have no stray atoms. One with stray atoms waits at 9, unreviewed and not fencing (a known
//! limit, spec §7.6 "Not judged"): the log lists it as `waiting` and stays "complete" while it remains, and the note
//! counts it. The review repeats when the store, the policy's bounds or the basis (degree cap, body-4 degree) changed;
//! its note only when the verdicts changed.
//!
//! **Box-affected answers** (fix round 4; threshold Task 9, h-scaled by ruling S17): a teacher
//! answer whose reported face density (`EvalStats::face_density`: max ρ on the outermost grid
//! plane, one spacing h inside the Dirichlet wall) exceeds [`box_affected_face_density`]`(h)` =
//! 2e-5 e/Bohr³ × (h / 0.25 Bohr)² is used for the step but never kept for training, never
//! judged as an audit and never used to settle a quarantine — it is the box, not the molecule.
//! The scale (derivation on [`BOX_AFFECTED_FACE_DENSITY`]): ΔE_conf ≈ π d ρ_face / (4κ²h²) for a
//! nucleus d from the wall and a tail decaying as e^{−κx}, ≈ 50 Ha per e/Bohr³ for H₂ at
//! h = 0.25 (measured 40–55, 41³ and 49³), so the same reported density means 4× the
//! confinement at h = 0.125, and a quarter of the density is tolerated there. The box's confinement makes the teacher's energy depend on
//! *where* the molecule sits — and, for a centred molecule, on its *orientation* relative to the
//! box axes (a bond along a cube axis has its ends L/2 − R/2 from the walls, along the body
//! diagonal √3 times farther) — while the descriptor is invariant under both, so beyond the
//! threshold the teacher's surface is **not a function of the internal coordinates at all** and
//! no invariant model can represent it: the rule is honest, not over-conservative, and the fix
//! for a stretched molecule it blocks is a bigger box. Task 9 item 1: under the old 1e-3
//! threshold (≈ 5e-2 Ha of confinement) a molecule parked 2.6 Bohr from a face failed its audit
//! on +6.4e-3 Ha of confinement, the refit quarantined that sample, and its fence — in
//! descriptor space — covered the equilibrium bond everywhere in the cell: a trusted model
//! falling back on every step with a stable gap.
//!
//! Exempt from "not kept": the samples a model is still being built from (no model yet, a new
//! species, trust short of samples — the thinning exemptions), and the sampling pass's own
//! samples (`crate::sampling` hands each one over with its teacher's face density through
//! [`SurrogateKs::add_sample_with_face`] — final review I1: before, pass samples carried no
//! density and a scan or hot burst on a small box stored confined labels as ordinary,
//! fence-capable samples; the 33³ GPU round trip needs them kept). They are stored with their face
//! density (sidecar `sample_face_density`) and are **never quarantine candidates** — a
//! box-affected sample must not fence its descriptor neighbourhood (Task 9 fix round 1, S1) —
//! and once the store holds the trust floor (the columns the usable atoms touch + 30; Q′ + 30
//! before M8a user-fix 3's ruling on concern 2) without them, the next refit excludes
//! them from the fit for good (`box_excluded()`, sidecar code 3). Until then the
//! training-residual gate bounds any bias such a sample can leave in an early fit: either the
//! model reproduces it within the tier (so the bias is under the tier) or the model stays
//! untrusted until enough clean samples arrive to exclude it.
//!
//! **Fence voting** (Task 9 fix round 1, I3): a quarantined sample that was a *glitch* (a bad
//! label, not box-affected) in a region where the model is genuinely wrong by more than the tier
//! used to leave no learning path: every fresh answer there disagrees with the model, none was
//! kept, and the fence never released — DFT speed under a trusted badge. So when
//! [`FENCE_VOTES`] (3) consecutive fresh teacher answers inside one fence disagree with the model
//! but are **mutually consistent** — their residuals teacher − model agree: every pair's per-atom
//! energy residuals within `energy_std_per_atom` and every atom's force-residual magnitudes within
//! its force bound (`quarantine::votes_agree`; M8a Task 5: before, the raw answers were compared,
//! and the votes of a vibration entering a fence once per period — different geometries on a steep
//! wall, the molecule turned in between — never agreed although the model's error did) — the
//! third is kept (tag Fallback) and the next refit decides: if the new model reproduces the kept
//! answer it stays in the fit — the fenced glitch is then released if
//! the model reproduces it too, or superseded by the next fresh answer, which now agrees; if the
//! model cannot reproduce the kept answer, it is quarantined itself and its region takes no
//! further vote (an unrepresentable region — a dimer beyond the cutoff — costs one sample, not a
//! chain up to the 10 % cap). One vote per region is pending at a time, at most
//! [`FENCE_VOTE_LIMIT`] per fenced sample.
//!
//! **Stuck** (Task 9): a trusted model that falls back on [`STUCK_STEPS`] consecutive
//! consultations says so ([`SurrogateKs::stuck`]: the rule that fires and the gap) instead of
//! claiming trust while every step is DFT; the next answer clears it. It stays trusted —
//! elsewhere it is right, and declaring it untrusted would make every step DFT everywhere.
//! When the last [`STUCK_STEPS`] of those fallbacks were all box-affected (M6 Task 11) the note
//! says why nothing is being learned: "every answer here touches the box wall — enlarge the grid
//! (or turn container walls on)" — those answers are never kept (the box rule), so samples and
//! epoch stand still under a trusted badge, which is what the user saw for 140 fallbacks.
//!
//! **Fallback thinning.** A fallback answer at a geometry like one already held teaches nothing
//! new, and one per step for thousands of steps (a molecule parked where the model cannot
//! answer) bloats the store and every refit. A fallback sample is therefore kept only if its
//! nearest stored descriptor (every stored sample, quarantined included; L∞ over features in the
//! envelope's training-spread units, the worst atom) is farther than
//! [`THIN_FRACTION`] · `max_descriptor_distance`; of the *thinned candidates* (answers that close
//! to a stored one) one in [`THIN_EVERY`] is kept anyway — the counter advances only on such a
//! candidate and restarts at every kept sample, so at a parked molecule it is one teacher step
//! in ten. Exempt (always
//! kept): audits; the variance rules' fallbacks (6–7: answers the model asked for — active
//! learning); fallbacks with a new species, or while the trust rule still needs more samples
//! (n_samples < Q′ or too few residual degrees of freedom) — those samples are what earns
//! trust; and fallbacks with no model yet, up to [`NO_MODEL_SAMPLE_FACTOR`] · Q′ samples held.
//! Thinned: the envelope rule (5), an untrusted model (rule 4) whose trust does not wait for
//! samples, and an untrusted class that already holds Q′ samples (4b; short of them its fallbacks
//! are exempt); and a refusal by the fence alone next to forgotten samples (final review I1). **Copies** (round 1 of the 2026-10-18 trust review, MUST-2): a fallback under rules
//! 4, 4b or 5 or the forgotten-sample fence, exempt or not, whose labels repeat a stored sample's within the fit's own noise (the
//! nearest one of the same atom sequence inside the thinning radius: |ΔE| ≤ N·s_E, every atom's
//! |F| within s_F — `is_label_copy`) is never kept: copies satisfy the count conditions by
//! repetition and, fitted, shrink the two-noise evidence's s_F and with it rule 6's bound
//! everywhere (measured: 100 copies, s_F ×0.62 and rule 6's std ×0.64); a parked molecule
//! //! then adds nothing (`a_parked_molecule_neither_floods_the_store_nor_shrinks_the_noise`).
//!
//! **No model yet** (final review B2): before the first fit there is no envelope to thin by,
//! and the worker refits only once a model exists (the sampling pass makes the first fit and
//! solves the references it needs) — so while the user plays with a pass offer waiting, every
//! DFT step used to add a sample with no bound. Beyond [`NO_MODEL_SAMPLE_FACTOR`] · Q′ samples
//! (Q′ of the held species' default basis: past the trust floor Q′ + 30 for every species set)
//! one teacher answer in [`THIN_EVERY`] is kept, with a note saying how many are held and
//! that a sampling pass is what builds the model.
//!
//! **A fit that does not form** (2026-10-20, the PH₃ session: its isolated P reference did not
//! solve, so the model stayed unfitted, Save was unavailable and the session autosave deleted
//! the minutes of DFT it had taught). Samples without a model are state worth keeping:
//! - **Saved.** `save_sidecar` writes them as a samples-only file (`surrogate::sidecar` version
//!   6) with the reference energies known and the solve record; the worker's autosave keeps
//!   such a state instead of removing the file. Loading one ([`install_samples`](SurrogateKs::install_samples))
//!   restores the samples and fits when a fit can form.
//! - **Said plainly.** [`no_model`](SurrogateKs::no_model) gives the reason, in this order: a
//!   species whose reference is missing ("the P atom reference did not solve (4 cold attempts:
//!   SCF stalled …)" — the failed scan is remembered per species until an energy arrives), too
//!   few usable samples for the columns their atoms touch (`fit_floor`; the same floor gates the
//!   end-of-pass refit, M8a user-fix 3), the last fit error, a fit not made yet; `TrainableStatus::no_model`
//!   carries the words to the Model and Activity tabs.
//! - **Retried where that can help** (physics review of the fix). A cold solve is deterministic, so
//!   the reference scan retries only a multiplicity lost to the device, once, cold, after its first
//!   round (`sim::references`, "The retry round"; each attempt an Activity event). A reference whose
//!   SCF itself failed (stalled, capped) is said to be reproducible and gets no Retry: this surface
//!   cannot have a model until the solver converges that atom there. A reference not solved yet
//!   (samples loaded from a file, a pass cancelled) or lost to the device gets the Model tab's
//!   Retry ([`retry_fit`](SurrogateKs::retry_fit)): a references-only pass for the missing species
//!   (`Sampler::references_only`, on the scene — refused when the scene's surface is not the
//!   samples', and under an ionic policy when the scene lacks the species, whose lone atom could
//!   otherwise be solved on another surface) whose end refits like any pass; with every reference
//!   known it refits at once; with too few samples it is not offered (sample more). A samples-only
//!   file's references of another rule than S9 are not fitted on (they count as missing). The
//!   reference rule is untouched: the lowest-energy converged multiplicity is the reference.
//!
//! # One-surface energy accounting (spec §5)
//!
//! The driver's `E_conserved` must be computed on ONE potential-energy surface, or every hand-
//! off between model and teacher would look like a drift. So on a Fallback step with a trusted
//! model (rules 6–8 and the fence, the gap inside the tier: the exceptions follow) the result is `ForceResult { energy: E_model(r), forces: F_teacher(r) }` and
//! `teacher_gap = E_teacher − E_model` is reported separately: the atoms move on the teacher's
//! forces (accuracy), the invariant stays on the model surface (continuity), and the gap is a
//! readout of the model's error rather than a corruption of the drift. Relative to the model
//! Hamiltonian, ΔF = F_teacher − F_model is then an **external force** — the same category as
//! the drag spring and the thermostat — and the driver books its work: the result also carries
//! `EvalStats::model_forces` on such a step and `Dynamics` adds ½(ΔF_old + ΔF_new)·Δr per step
//! to W (trapezoid, O(dt²)-exact like the drag booking; `DynSnapshot::work_mismatch` publishes
//! the running sum). Unbooked, `E_conserved` would move by −Δ(teacher_gap) on every fallback
//! step and the drift readout would mix integrator error with the model's force error — which
//! is exactly what the two readouts are meant to separate. When no trusted model
//! exists (rules 1–4) the step is plain DFT: `energy` is the teacher's and `teacher_gap` is
//! `None` — [`EvalStats::energy_surface`] then says `Surface::Teacher`. A change of surface
//! between consecutive evaluations is a discontinuity of the invariant; the driver books it when
//! the provider announces it (`refit_jump`, below) and otherwise restarts the `E_conserved`
//! history, the same path it uses for a discontinuous edit.
//!
//! A refit moves the model surface under the atoms. [`refit`](SurrogateKs::refit) is called by
//! the worker between ticks, never inside `evaluate`; it evaluates the new model at the last
//! evaluated positions and reports, once, on the first evaluate after it,
//! `refit_jump = E_new(r_last) − E_reported(r_last)`, where `E_reported` is whatever energy the
//! last evaluate returned (the old model's on the model surface, the teacher's before the first
//! trusted fit — so the first transition onto the model surface is booked too). The driver adds
//! it to its work term W exactly like thermostat work, and because it notices the refit through
//! [`ForceProvider::surface_epoch`] it re-evaluates at the unchanged positions first: the jump is
//! then exact (no half-kick on mixed forces) and `E_conserved` is continuous to rounding
//! (test: 1e-12). If the new fit is not trusted the surface does not change and nothing is
//! booked; if a rebuild makes a trusted model untrusted (a new species raised Q′) the next
//! evaluate is on the teacher surface with no jump, and the driver's restart path applies.
//!
//! **Which surface a fallback is reported on (M6 Task 11).** The teacher's, with the step's
//! transition booked (below), when (a) rule 5 or 4b fired — the model's energy there is a polynomial
//! extrapolation (the user's +578 Ha), and booking ΔF·Δr with its extrapolated forces made
//! E_conserved wobble by 0.8 Ha; (b) it is the first evaluation after an edit (`reset` /
//! `discontinuity`: the driver restarts the history there, which must not be anchored on a model
//! guess — the user's E_conserved carried a 15 Ha offset for the rest of the run); (c) the
//! teacher − model gap exceeds the energy tier, |gap|/N > `energy_std_per_atom` (not an audit:
//! a failed audit also untrusts) — the model's energy there is not what the model claims, and
//! E_total must stay within the tier of the teacher's on every step the teacher answered (the
//! repro measured rule-7 fallbacks 3.9e-2 Ha/atom off). Otherwise — rules 6–8 and the fence
//! with the gap inside the tier — the model's, as below.
//!
//! **Transitions between steps.** When the reported surface changes from one step to the next
//! (r₀ → r₁), the step is booked in `refit_jump` as J = E₁ − E₀ + ½(m̃₀ + m̃₁)·(r₁ − r₀), m̃
//! the reported surface's forces (`transition_jump`); with the driver's own mismatch term the
//! work booked is E₁ − E₀ + ½(F₀ + F₁)·Δr, F the forces the two half-kicks used, so E_conserved
//! changes by exactly Verlet's ΔK − ½(F₀ + F₁)·Δr = dt²Σ(|F₁|² − |F₀|²)/8m on that step (the
//! GPU repro measures it to 1.6e-16 Ha) — a term that telescopes when the run switches back —
//! and by nothing that depends on either surface between the two points: neither the model at a
//! teacher-surface point (an extrapolation) nor the teacher at a model point (never solved)
//! enters. What no energy booking can carry is the shadow correction of the surface in use:
//! Verlet conserves H̃ = H + dt²(pᵀV″p/12m² − |F|²/24m) + O(dt⁴), and a switch from A to B changes
//! it by δ ≈ (dt²/12)[pᵀ(V″_B − V″_A)p/m² + (|F_B|² − |F_A|²)/m]. So E_conserved keeps, per switch,
//! an **absolute residual O(dt²) whose coefficient is the force / curvature mismatch at the
//! switch**, accumulating over switches — the same order as Verlet's own swing, so the ratio of
//! the spreads plateaus at 1 + O(mismatch) rather than tending to 1 (physics review, round 3:
//! 1D ε sin(kr), ε = 1e-3, k = 10: 1.075 … 1.091 from dt 0.25 to 0.0156 fs). Tests: the matched
//! Morse excursion (per-step change = Verlet's) and the ripple excursion, |ΔQ| within the derived
//! O(dt²) bound at five dt (`a_surface_mismatch_at_the_switches_leaves_an_order_dt2_residual`). A return to the model's surface books J at the first model-surface
//! report after a teacher-surface one with the model consulted (`LastEval::flip_back`); after a
//! rule-1–4 teacher step the refit's pending jump does it, as before.
//!
//! `reset()` / `discontinuity()` pass to the teacher; the model persists (it is a function of
//! positions, not of solver state). The teacher's errors propagate unchanged; a sample is never
//! kept from a failed teacher call.
//!
//! # Pseudopotential scenes (M7 Task 9)
//! - **Never trained on** (before `admit` can quarantine, fence or release anything on it): a teacher
//!   answer with `EvalStats::core_overlaps` (two ions' frozen cores overlap — the approximation fails,
//!   spec §5.6) and one the `branch_jump` guard flagged (`EvalStats::branch_jump`: F changed by more than the
//!   forces' work since the teacher's previous answer — another stationary state, spec 2026-11-13 §6.3; it
//!   replaced the shared-shell occupation watch in plan 2026-11-13 Task 9). The answer is still the step's;
//!   the note reads "surrogate: sample not kept (cores overlap — spec §5.6)" or "(… the forces' work —
//!   Fermi–Dirac spec §6.3)" (`sim::untrainable_citation`), the activity row is `TeachOutcome::UnfitNotKept`.
//!   The same holds for a background audit's answer and a sampling pass's geometry (`EvalStats::untrainable`).
//! - **Force-label noise prior** ([`PSEUDO_FORCE_NOISE`] = 5e-5 Ha/Bohr per component, task-8b review
//!   ruling (b) and task-8a): the GPU pseudo path's own scatter at the sampling tolerance is measured
//!   4.7–4.8e-5 on lone pseudo-O and OH, up to ≈ 7e-5 on warm teacher chains, and no model can represent
//!   it. On a pseudo surface (`!pseudo::is_all_electron_tag(&Provenance::ionic)`) the fit's force noise s_F is floored at it
//!   (`FitOptions::sigma_f_min`: the fit is redone with s_F pinned there when the evidence estimate falls
//!   below it), so the posterior is never more confident about forces than the labels can support; and a
//!   force **audit** is never judged against a tier below 3 × it (`policy::audit_force_floor`). The
//!   default tiers (`force_std_abs` = 2e-3) are 40× above it, so both floors are inert unless the policy
//!   is tightened; the all-electron surface keeps its evidence-estimated noise, bitwise.
//! - **The stop rule is compared, the value is recorded.** Under a pseudo scene the GPU teacher stops its SCF at
//!   max(tol, 2·floor(N)) (spec §7 risk 6; floor = 5e-8·N). That rule is a deterministic property of (teacher,
//!   nominal tolerance), so the ionic tag carries it (`stop2x5e-8`) and `Provenance::tol` stays the nominal
//!   tolerance: putting the N-dependent value into the compared provenance would split a model's surface at every
//!   Add or Remove of an atom above 10 valence electrons (rule 3 would then refuse the model for a tolerance the user
//!   never set). Each answer's effective value is recorded, never compared: `EvalStats::scf_tol_used`.
//!   The first-order label error of 1.2–1.7e-6 against 1e-6 is inside the 5e-5 label-noise prior above.
//! - **Occupation changes** are judged against the provider's *previous evaluation*; under the surrogate the
//!   stepping teacher is called only at fallbacks, so "changed" means "differs from the last teacher call" —
//!   conservative (never trains on more than the Task 6 A.6 rule), coarse (drops the first sample after any
//!   difference, cannot see a switch and switch-back between two calls); per-sample pattern storage would refine
//!   it and is not built.
//!
//! # Masked scenes (M8a Task 10, "Train this segment"; spec §4.5, rulings P10/P15)
//!
//! A capped segment's scene carries `Scene::train_mask` (true = an interior atom: every source atom
//! within r_cut of it is in the segment and no cap is). The sample's own semantics are
//! `surrogate::fit`'s ("Masked samples"): its interior atoms' force rows, no energy row. Each
//! consumer of samples here decides explicitly:
//!
//! - **Keep path.** A teacher answer of a masked scene — a live step, a sampling pass's geometry
//!   (`add_sample`, `add_sample_answer` via `store::masked_by`) — is kept with `force_mask` = the
//!   scene's mask and `use_energy` false. A mask whose length is not the nucleus count is never
//!   used (`Scene::train_mask`); an answer with no interior atom is refused by the store (it teaches
//!   nothing). A `.model` file or a collection reads a sample as it was stored. A sample with
//!   **stray atoms** (`stray` module doc) carries a mask too, its energy fitted: its stray atoms'
//!   force rows are zero and are left out (`stray::mask_stray_atoms`, at every entry to the store;
//!   in memory only for a loaded sample).
//! - **The model never answers in a masked scene** (`TeachRule::Segment`, checked in `evaluate`
//!   before rule 1): every step is the teacher's and is kept; nothing is audited, judged as a fallback
//!   point, thinned or held out there — the caps and edge atoms are not the source's, so a model
//!   answer there would be judged on atoms it is not meant to know. The pass plans no **shell check**
//!   in a masked scene (judged on every atom, it could fence a source pair for an answer the caps
//!   distort). The quarantine fences apply as for any answer (the fence's nearness counts every
//!   atom: the conservative side; with no model answer the fence vote decides, and the note says
//!   no model answered). Dedup (`is_label_copy`) compares an answer only with stored samples of
//!   the same mask and energy use (a whole answer is never a copy of a segment), on the fitted
//!   rows: no energy test without an energy row, no masked-out force (Task 10a review, finding 4).
//! - **Leave-one-out, class and key verdicts** (`trust::compute_class_loo_full`): only the fitted
//!   rows — a masked sample's energy LOO does not exist (`SampleLoo::energy` `None`), its masked-out
//!   atoms have no force LOO and no class or key (`trust::trained_keys`). A class or key counts a
//!   masked sample when it holds an interior atom of it; its energy statistics count the samples with
//!   a fitted energy only, and a key holding segments with fewer than `calibrate::MIN_EVIDENCE` (3)
//!   fitted energies — none, or one or two whole samples among the segments — reads as energy
//!   LOO ∞: refused for want of energy evidence (Task 10a review, finding 1; ruling P25: the
//!   evidence comes from whole-scene fallbacks, never from a force-only energy calibration —
//!   forces determine the energy model, but no energy datum vouches for it). A key of whole
//!   samples only is judged as before. `loo_rms` and `loo_force_errors` likewise.
//! - **Sample floors** (`trust::key_column_counts`, the whole-model union floor, `fit_floor`'s
//!   cache key): only interior atoms touch a key's columns and count as its samples.
//! - **Training residuals, misfit set, recovery and release** (`compute_residuals`, `worst_offenders`,
//!   `reproduces_sample`): the fitted rows only, so a cap's force never quarantines a segment.
//! - **Teach-around**: a masked sample is never a blocker (its geometry is a cut-out the segment
//!   scene's own pass teaches); the failed-audit re-judgement (`judge_case`) of a masked sample —
//!   unreachable, nothing is audited in a masked scene — takes the plain prediction for the rows its
//!   fit left out.
//! - **Envelope, active set (rule 5e), null space (rule 5d)**: interior atoms only (the envelope and
//!   the null space from `Accumulator::add`'s rows; the active set filters the descriptor cache).
//! - **Readouts** (the parity views, `inspect`, the knowledge map, the triple map): interior atoms,
//!   no energy point for a masked sample; a pair is knowledge when one of its atoms is interior, a
//!   three-atom arrangement when its centre is.
//!
//! What the provider cannot check: the mask was computed at the cut geometry; a pass's perturbations,
//! scans and hot bursts move atoms, and a cap can come within r_cut of an interior atom in a moved
//! geometry (the mask is plain data, the provider does not know which atoms are caps). The cut
//! (`app::segment`, Task 10b) should leave a margin for the pass's displacements.
//!
//! # Species, references, provenance
//!
//! The species list of the `Basis` is the sorted, deduplicated set of Z over the samples held
//! when the first refit runs (the first sample's set, in the brief's words, plus anything added
//! before that refit); a later sample with a new species marks the basis dirty and the next
//! refit rebuilds the accumulator from the store with the enlarged set — which needs a
//! reference energy for the new species ([`set_reference_energy`](SurrogateKs::set_reference_energy);
//! Task 5 computes them with the teacher on an isolated atom) and otherwise fails with
//! `FitError::MissingReference`, keeping the old model. Provenance (grid, h, σ, charge,
//! multiplicity, teacher name, teacher tolerance) is derived from the scene a sample came from
//! and pinned by the first sample; the teacher's SCF tolerance is not visible through the
//! boundary, so it is set with [`with_teacher_tol`](SurrogateKs::with_teacher_tol) (0 = not
//! recorded). It is part of the provenance like the grid: a model taught at another tolerance
//! is refused by `Dynamics::LoadModel` with both tolerances in the message, and would fall back
//! under rule 3 if installed directly — labels at 1e-5 and 1e-6 are never mixed in one store.
//! **Which tolerance** (ruling S18): the viewer's estimated solver gives its own teacher — the
//! sampling pass, the fallbacks and the background auditor — the sample tolerance 1e-6 (spec §4:
//! the labels are permanent, and at the interactive 1e-5 the evidence's noise floor would be set
//! by SCF lag), while the plain DFT backends keep the interactive 1e-5.
//!
//! **A shared collection** (M8a Task 4a, `shared` module doc) changes three things here: the
//! store's provenance is the collection's (any box, neutral, automatic multiplicity) and a sample's
//! check is the collection's surface-key check; a reference is offered with its box
//! ([`set_reference_solved`](SurrogateKs::set_reference_solved): the box rule, odd n) and kept per
//! (species, box) — each sample is fitted against its own box's references, a sample whose box has
//! none yet waits (`shared`, "References per box") —; and species growth rebuilds the
//! accumulator from the whole collection-backed store, exactly as above.

use crate::sampling::{atom_key, Sampler};
use sim::activity::{SolveCounts, TeachEvent, TeachOrigin, TeachOutcome, TeachRule};
use sim::references::{AtomReference, AtomReferences};
use sim::sampling::{SamplingPlan, SamplingProgress};
use sim::{
    ClassLoo, Display, EvalStats, Extrapolation, ForceProvider, ForceResult, OrbitalDisplay, OrbitalViewRequest, ProviderError, Scene,
    ShareTables, SideDisplay, SideKind, Source, Trainable, TrainableStatus,
};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Instant;
use units::isotopes::symbol;
use surrogate::{
    describe_values, Accumulator, ActiveSet, Basis, Envelope, FitError, FitOptions, FitReport, Model, NullSpace, Provenance,
    Sample, SampleStore, SampleTag, StoreError,
};
use surrogate::sidecar::TeacherSolves;

mod audit;
mod background;
pub mod calibrate;
mod inspect;
mod knowledge;
mod ledger;
mod lone;
mod policy;
mod purge;
mod quarantine;
mod shared;
pub use shared::{legacy_model_refusal, predecessor_of, LAST_PRE_FD_RELEASE, COLLECTION_CHECKPOINT_PREFIX, PASS_STARTED_WITH_PREVIOUS_SOLVER};
pub use purge::PURGE_WAIT_MIGRATION;
mod store;
pub(crate) use store::PassKeep;
// the stray-atom logic (spec 2026-11-18): the composition checks and the hold-out (plan Task 3); answering inside a
// cleared region is Task 4's, so a few of its items are not used yet
#[allow(dead_code)]
mod stray;
mod teach;
mod triples;
mod trust;
mod views;

use audit::*;
use inspect::StrictVerdict;
use trust::ClassJudge;
use ledger::*;
use lone::Lone;
use policy::*;
use quarantine::*;
pub use audit::{history_rule_code, history_rule_name, AuditRecord, AuditVerdict, HISTORY_RULE_CLASS_AND_VARIANCE, HISTORY_RULE_TAIL_INFLATION};
pub use calibrate::{calibrate, floor_count, refusal_words, Calibration, ClassCalibration, ClassStats, ClassVerdict, WordsInput, ENTER, FLOOR_FITS, FLOOR_POINTS, LEAVE, MIN_EVIDENCE, WINDOW_POINTS, WORST_ATOM};
pub use ledger::EVENT_QUEUE;
pub(crate) use policy::PSEUDO_FORCE_NOISE;
pub use policy::{box_affected_face_density, AtomVerdicts, FallbackPolicy, BOX_AFFECTED_FACE_DENSITY, FORCE_STD_BOUND_MAX, GRADE_MAX};
pub use quarantine::{
    FENCE_VOTES, FENCE_VOTE_LIMIT, NO_MODEL_SAMPLE_FACTOR, QUARANTINE_FRACTION, QUARANTINE_MAX_FRACTION, QUARANTINE_ROUNDS,
    RECOVERY_DEGREE_CAP, SUSPECTS_SHOWN, THIN_EVERY, THIN_FRACTION,
};
pub use trust::{atom_class, env_class, LooRms, Misfit, TrainingResiduals, AUDIT_BURST, AUDIT_REJUDGE_LIMIT, ENV_CLASSES, MIN_RESIDUAL_DOF, STUCK_STEPS};
pub use background::BackgroundPoll;
pub use views::Verdict;
pub use teach::{Blocker, TEACH_MAX_BLOCKERS, TEACH_PATH_POINTS, TEACH_PERTURBATIONS, TEACH_PROMPT_STREAK, TEACH_ROUNDS};

/// What the last evaluate reported, for the refit jump.
struct LastEval {
    pos: Vec<[f64; 3]>,
    z: Vec<u8>,
    energy: f64,
    on_model: bool,
    provenance: Provenance,
    /// A teacher-surface report with the model consulted (rule 5, the first evaluation after an
    /// edit, a gap beyond the tier): the next model-surface evaluation books the transition back
    /// (module doc, "One-surface energy accounting").
    flip_back: bool,
    /// −∇ of the reported surface at `pos`: the model's forces on the model's surface (a
    /// model-surface fallback's `model_forces`), the reported forces otherwise — the m̃ of the
    /// transition booking (`transition_jump`).
    surface_forces: Vec<[f64; 3]>,
    /// A lone-atom reference answer (`lone`, `Source::Reference`), and whose references it summed:
    /// on the model's surface, and a teacher step after it books the transition (module doc, rule 0b).
    reference: Option<LastReference>,
}

/// Whose references a lone-atom reference answer summed (`lone`, "The energy's surface").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LastReference {
    /// The model's own (`Model::ref_energy`): the report moves only with a refit.
    Model,
    /// The provider's (`refs`; the canonical ones in collection mode): a reference arrival moves it.
    Provider,
}

/// The model's answer at the geometry of a teacher step (rules 5–7 and audits).
struct ModelAnswer {
    energy: f64,
    forces: Vec<[f64; 3]>,
    uncertainty: Option<f64>,
    /// The model's predictive stds at the geometry — (total-energy std, per-atom force std, its
    /// epistemic part) — the history's point (M8a Task 1, `surrogate::history`) of an in-step
    /// audit or of a consulted-model fallback. `None` when the model has no variance factor.
    stds: Option<(f64, Vec<f64>, Vec<f64>)>,
    /// The total-energy epistemic std of `stds`' energy std under the model that predicted it
    /// (`epistemic_energy_std`; module doc, "The history's flags"), unsanitised — the history
    /// point sanitises it. `None` with `stds`.
    energy_std_epistemic: Option<f64>,
}

/// One teacher step: why (the note's text, the rule's code and the atom it names), with what
/// from the model, and whether it is an audit.
struct Handoff {
    prov: Provenance,
    pos: Vec<[f64; 3]>,
    z: Vec<u8>,
    reason: Refusal,
    keep_sample: bool,
    /// The geometry is inside a fenced shell band (final review I1): the answer drives the step
    /// and is never fitted (`keep_sample` is false, the outcome `ShellNotKept`) nor judged as an
    /// audit — the relax verification and an audit included (the coordinator's ruling).
    shell_fence: bool,
    /// The kept sample is subject to thinning (module doc).
    thin: bool,
    /// A copy of a stored sample is not kept (`is_label_copy`).
    dedup: bool,
    /// Quarantined samples within the quarantine radius of this geometry (module doc).
    near: Vec<usize>,
    /// An atom of this geometry is near a misfit (`near_misfit_atoms`; spec 2026-11-13 §4.3):
    /// the answer is kept only if it agrees with the model (`admit`). Computed wherever `near` is,
    /// false where `near` is empty for want of a model.
    near_misfit: bool,
    model: Option<ModelAnswer>,
    audit: bool,
    /// Report the step on the teacher's surface although the model was consulted (rule 5, the
    /// first evaluation after an edit; module doc, "One-surface energy accounting").
    teacher_surface: bool,
    /// The audit history's rule code for a consulted-model fallback point (`history_rule_code` of
    /// the refusing rule before any relax-verification override — the Task 1 re-review's Minor 1 —
    /// or `HISTORY_RULE_CLASS_AND_VARIANCE` when the class rule refused and rule 6/7 would have
    /// too; M8a Task 2). Unused for an audit or a gate refusal.
    history_rule: u8,
    /// The history point's flags (module doc, "The history's flags"): the consulted step's
    /// cadence-fresh and fenced bits; empty where no point is recorded (a gate refusal, a capped
    /// segment, a non-finite answer); `TICK` on a calibration tick's point.
    flags: surrogate::history::PointFlags,
    /// An in-step audit that is a calibration tick alone, not rule 8's (module doc, "Calibration
    /// ticks"): judged as an audit, but a box-affected answer does not make rule 8 audit again.
    tick_only: bool,
}

/// The calibration ticks' cadence (module doc, "Calibration ticks"; spec 2026-11-18 §4): every
/// 25th counted in-envelope strict-mode step, per provider, independent of `audit_every`; after
/// every due tick the count pauses for the dead time D (`SurrogateKs::tick_dead_time`).
pub const TICK_EVERY: usize = 25;

/// The weight of the newest sample in the dead time's EMAs (`SurrogateKs::tick_dead_time`).
const TICK_EMA_WEIGHT: f64 = 1.0 / 16.0;

/// The largest in-envelope step wall time one sample may stand for, ms: a pause, an edit or a
/// stalled frame between two steps is not a step's time.
const TICK_STEP_SAMPLE_MAX_MS: f64 = 1000.0;

/// An exponential moving average with [`TICK_EMA_WEIGHT`]; the first sample seeds it.
fn ema(prev: Option<f64>, x: f64) -> f64 {
    prev.map_or(x, |p| p + TICK_EMA_WEIGHT * (x - p))
}

/// What became of a calibration tick that fell due (module doc, "Calibration ticks").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TickOutcome {
    /// Taken, its point recorded.
    Recorded,
    /// Taken, no point: the answer was box-affected, unfit or without a variance factor, or the point was
    /// deferred to a stray attribution check (its point arrives later with `TICK`).
    Unrecorded,
    /// The auditor was busy with other work (rule 8, a display, a reference).
    SkippedBusy,
    /// Not measured: its auditor job abandoned (a Forget, an installed model, the thread gone) or
    /// returned failed or unjudged, its solve failed, or no prediction to compare.
    Lost,
}

/// The calibration ticks' counts (`SurrogateKs::tick_counts`; module doc, "Calibration ticks"):
/// every tick that fell due is in exactly one of the first four, or still in flight.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TickCounts {
    /// Taken, its point recorded.
    pub recorded: u64,
    /// Taken, no point (box-affected, unfit, no variance factor), or the point deferred to a stray attribution check (its point arrives later with `TICK`).
    pub unrecorded: u64,
    /// Skipped: the auditor was busy with other work.
    pub skipped_busy: u64,
    /// Lost: abandoned, unjudged, failed, or nothing to compare.
    pub lost: u64,
    /// Of the taken ones, served by a teacher solve in the step (no auditor).
    pub in_step: u64,
    /// Every tick that fell due (the four above plus the one in flight, if any).
    pub due: u64,
    /// Of `due`, those at a step the model answered (taken or skipped).
    pub answered_due: u64,
    /// Of `answered_due`, skipped because the auditor was busy.
    pub answered_skipped_busy: u64,
    /// Every in-envelope step (the tick predicate's population: in the dead time and with a job in
    /// flight included). σ-blind: the diagnostic's denominator.
    pub population: u64,
    /// Of `population`, the steps the model answered: `population_answered / population` against
    /// `answered_due / due` measures any σ-coupling of which steps become ticks (the Task 7
    /// re-review's R1 diagnostic; plan Task 9 Step 3b).
    pub population_answered: u64,
}

/// The surrogate force provider (module doc).
pub struct SurrogateKs {
    teacher: Box<dyn ForceProvider>,
    name: &'static str,
    teacher_tol: f64,
    policy: FallbackPolicy,
    store: Option<SampleStore>,
    accumulator: Option<Accumulator>,
    /// Store samples already in the accumulator (a prefix).
    accumulated: usize,
    refs: BTreeMap<u8, f64>,
    /// Species whose reference came from a sidecar without the S9 reference rule (re-solved by
    /// the next sampling pass; `save_sidecar` writes no rule while any remain).
    legacy_refs: std::collections::BTreeSet<u8>,
    /// A reference changed since the accumulator was built → rebuild at the next refit.
    refs_dirty: bool,
    /// The store holds a species the accumulator's basis lacks → rebuild at the next refit.
    species_dirty: bool,
    model: Option<Model>,
    report: Option<FitReport>,
    envelope: Option<Envelope>,
    /// The fit's unspanned directions (rule 5d, `surrogate::novelty`).
    null_space: Option<NullSpace>,
    /// Inside a refit (`fit_store_deferred`): `null_space` is not yet the current fit's — computed
    /// once for the fit that stands (`ensure_trust_passes`, or before the accumulator it is built
    /// from takes new rows), like the class pass and the active set (2026-11-17: an O(Q′³)
    /// eigendecomposition per intermediate fit, 3.6 s at Q′ 2 460). Nothing reads it meanwhile.
    null_space_pending: bool,
    /// The fit in place is a trust recovery's trial, searched from the previous fit's
    /// hyperparameters (`fit_trial`; spec 2026-11-17, option B): it never stands as it is —
    /// [`confirm_cold`](SurrogateKs::confirm_cold) searches it again from the data first. Saved
    /// and restored with the fit (`FitState`).
    trial_fit: bool,
    /// Tests only: every fit's solve (`Accumulator::last_refit`) and its basis's degree cap, in
    /// order (`fit_store_with`).
    #[cfg(test)]
    fit_log: Vec<(surrogate::RefitKind, usize)>,
    /// Tests only: the confirmation with this index (0-based, over every `confirm_cold` call) fails
    /// after its cold refit, and that fit then fails `recoverable_fail` as long as it is in place
    /// (`forced_fail_gen`: a trial that passes warm and fails cold); `confirm_failed` keeps the
    /// state around it.
    #[cfg(test)]
    confirm_fail: Option<usize>,
    #[cfg(test)]
    confirm_calls: usize,
    #[cfg(test)]
    forced_fail_gen: Option<u64>,
    #[cfg(test)]
    confirm_failed: Option<ConfirmFailed>,
    /// The extrapolation grade's MaxVol active sets (rule 5e, `surrogate::grade`).
    active_set: Option<ActiveSet>,
    training_residuals: Option<TrainingResiduals>,
    /// The misfit set M of the fit that stands (`trust` module doc; spec 2026-11-13 §4.3): set
    /// with `training_residuals` at every fit, never persisted.
    misfit: Option<Misfit>,
    /// Per environment class (pair, many-body): the fit's leave-one-out numbers and verdict
    /// (module doc, "Class-wise trust"); empty before the first fit.
    class_loo: Vec<ClassLoo>,
    /// The calibrated verdict's hysteresis memory (M8a Task 2, `calibrate`): per class the
    /// verdict that **stood** when the current refit, install or policy change began — (trusted,
    /// `trusted_since`) — committed by `commit_class_memory` once that operation is done, so a
    /// recovery's intermediate fits inside one `refit` neither enter nor leave on their own.
    /// Cleared by a forget or an install. Per environment key (class, Z, neighbour species: M8a
    /// user-test fix 2, `calibrate` "Per environment species").
    class_memory: BTreeMap<surrogate::EnvKey, (bool, Option<u64>)>,
    /// What the class verdicts were last judged from — the fit's per-class statistics, its keys
    /// and the calibration — so new evidence re-judges them without a refit (`rejudge_classes`,
    /// the controller's ruling 3 of user-test fix 2); `None` before the first fit.
    class_judge: Option<ClassJudge>,
    /// Inside a refit (`fit_store_deferred`): the current fit's column count |∪_k S_k| (the sample
    /// floor), while `class_loo`, `class_judge`, `class_worst`, `class_blockers` and `active_set`
    /// are still an earlier fit's — computed once for the fit that stands (`ensure_trust_passes`).
    /// `None` outside a refit: those are the current fit's.
    deferred_trust: Option<usize>,
    /// The per-sample leave-one-out ratios of a fit (`quarantine::LooRatios`, spec ruling R1b): the
    /// class pass's, kept for the recovery's offender ranking and the Model tab's worst-fitting
    /// samples. Keyed to the fit (`model_gen` and its members, `fit_members`); read only through
    /// `fresh_loo_ratios` / `ensure_loo_ratios`, which never return an earlier fit's.
    loo_ratios: Option<LooRatios>,
    /// Tests only: every fit inside a refit computes its trust passes, as before the deferral —
    /// the reference the deferred refit is compared with (`refit_cost` tests).
    #[cfg(test)]
    eager_trust_passes: bool,
    /// Tests only: a refit runs no trust recovery, so the misfit set is tested on the fit as it
    /// comes (`refit_fits`).
    #[cfg(test)]
    pub(super) skip_recovery: bool,
    /// A failed audit's teacher − model gap: untrusted until a refit passes it (rule 8, "A failed
    /// audit stays failed").
    audit_failed: Option<f64>,
    /// A failed audit's worst force residual and its bound, when the forces failed it.
    audit_force: Option<(f64, f64)>,
    /// The failed audit's geometry and labels (and its store index when the answer was kept): what
    /// every refit re-judges it on (module doc, rule 8, "A failed audit stays failed").
    failed_audits: Vec<FailedAudit>,
    /// Failed audits that were not re-earned (`AUDIT_REJUDGE_LIMIT`) or whose composition the scene
    /// no longer holds: their neighbourhoods fall back instead of the whole model (round 1, SHOULD-3).
    audit_fences: Vec<FailedAudit>,
    /// Answered steps still to be audited after a cleared failed audit (`AUDIT_BURST`).
    audit_burst: usize,
    /// Atoms of the geometry whose failed audit set `audit_failed`: the checklist's per-atom
    /// value |gap|/N (activity plan review M1). A readout only; it decides nothing.
    audit_atoms: usize,
    /// Per atom: the neighbour count within r_cut when the variance was last evaluated (a change
    /// forces a fresh verdict: module doc, "Variance cadence").
    verdict_counts: Vec<usize>,
    epoch: u64,
    pending_refit: bool,
    /// What a refit fits changed this many times (`Trainable::fit_inputs_gen`, quiet refits
    /// 2026-11-17): bumped with `pending_refit` by [`arm_refit`](SurrogateKs::arm_refit) — an
    /// appended sample, the user's Forget / Restore, a rebase, samples loaded, imported or
    /// attached — never by a refit's own outcome (a discard or a failure re-arms `pending_refit`
    /// without it).
    fit_inputs_gen: u64,
    /// The caller runs the fits this provider would run in place — an attached collection's first
    /// fit, a pass's end fit — in the background (`Trainable::set_fits_in_background`).
    fits_in_background: bool,
    /// A pass's end left its fit to the caller (`fits_in_background`): due until a background
    /// refit carries it (`first_fit_ready`; quiet refits round 2).
    pass_fit_due: bool,
    /// That pass's tail, completed when its fit lands (`complete_pass_tail`).
    pass_tail: Option<PassTail>,
    /// An attach left its first fit to the worker (`fits_in_background`): until a model forms
    /// (or `forget` clears it), [`first_fit_ready`](SurrogateKs::first_fit_ready) may say so. Only
    /// an attach sets it — a no-model session's kept fallback samples still wait for a pass
    /// (module doc, "No model yet").
    first_fit_due: bool,
    pending_jump: Option<f64>,
    last: Option<LastEval>,
    /// Consultations since the variance was last evaluated (the cadence, module doc).
    consults: usize,
    /// Steps that reached the gate so far, never reset (module doc, "The history's flags"): a step
    /// whose count is a multiple of `variance_cadence` is cadence-fresh. Unlike `consults` its phase
    /// does not follow σ.
    cadence_steps: u64,
    /// The calibration ticks' cadence, in in-envelope steps ([`TICK_EVERY`]; module doc,
    /// "Calibration ticks"). 0 takes no tick: the tests' "ticks off" only.
    tick_every: usize,
    /// In-envelope strict-mode steps so far, never reset (module doc, "Calibration ticks"): every
    /// `tick_every`-th is a tick.
    tick_steps: u64,
    /// What became of the ticks that fell due ([`TickCounts`]; `tick_outcome`).
    ticks: TickCounts,
    /// A tick's audit job is on the auditor, its answer not yet handled (an abandoned one is lost).
    tick_in_flight: bool,
    /// In-envelope steps of the dead time still to pass before the count resumes (module doc,
    /// "Calibration ticks").
    tick_dead: u64,
    /// A fixed dead time (tests: a deterministic schedule), instead of the timed one.
    tick_dead_fixed: Option<u64>,
    /// EMA of the auditor's job wall times, ms, over every job of every kind.
    job_ms_ema: Option<f64>,
    /// EMA of the in-envelope steps' wall times, ms, over every in-envelope step.
    step_ms_ema: Option<f64>,
    /// When the previous in-envelope step was decided (its wall time's start).
    last_in_envelope_at: Option<Instant>,
    last_variance: Option<VarianceVerdict>,
    /// The fit gates' floor, |∪ S_k| over the samples a fit would use in the basis it would build,
    /// keyed by a fingerprint of those samples and that basis (`fit_floor`; the status asks for it
    /// every snapshot, a descriptor pass over the store each time would cost ms). One slot per
    /// body-4 degree: a refit asks for the floor at each degree it weighs
    /// ([`grown_body4_degree`](Self::grown_body4_degree)), and the two must not evict each other.
    fit_floor_cache: std::sync::Mutex<BTreeMap<usize, (u64, usize)>>,
    /// The three-atom coverage map's fitted block, keyed per fit (`triples` module doc, "The
    /// cache"); `triple_map` takes `&self`, as `fit_floor` does.
    triple_cache: std::sync::Mutex<Option<triples::TripleCache>>,
    /// The teach offer's richer-basis words (`richer_basis_floor`), keyed by a fingerprint of all
    /// they read — the fitted samples (count, species, positions, force masks, held-out codes),
    /// the richer basis and the standing degree: the status asks every snapshot, and the count is a
    /// descriptor pass over every fitted sample in the degree-6 basis (60–77 ms on a 1590-sample
    /// collection, 2026-11-17). A changed input changes the key, so no mutating path clears it.
    /// The fingerprint itself hashes every fitted atom's position, O(Σ atoms) per read (0.22 ms at
    /// 1 621 samples); should collections grow far beyond that, a cheaper first-level key —
    /// (`model_gen`, `accumulated`, the store length, the quarantine codes' hash) — could gate it,
    /// the full fingerprint hashed only when that key moves.
    richer_floor_cache: std::sync::Mutex<Option<(u64, Option<String>)>>,
    /// Tests only: the richer-basis key counts `richer_basis_floor` ran (`status_cache` tests).
    #[cfg(test)]
    richer_floor_counts: std::sync::atomic::AtomicUsize,
    /// Tests only: `refresh_class_loo` runs (a policy change that re-judges nothing skips it).
    #[cfg(test)]
    class_loo_refreshes: usize,
    /// Variance evaluations so far (readout; the cadence test).
    variance_checks: u64,
    /// Steps the model answered or would have answered (the audit counter).
    answered: usize,
    /// Per stored sample: quarantined (module doc, "Trust recovery"); shorter than the store
    /// means "not quarantined" for the rest.
    quarantine: Vec<u8>,
    /// Per stored sample held out as lone atoms (`Q_LONE`, set at load — `quarantine` module doc of
    /// [`Q_LONE`]): the code it was loaded with, which a saved file writes back (the provider-only
    /// code is never persisted).
    lone_codes: BTreeMap<usize, u8>,
    /// The body-3 degree cap a recovery settled on (applied to every rebuilt basis).
    degree_cap: Option<usize>,
    /// The body-4 degree D₄ a collection's basis is built at (spec 2026-11-13 §4.1): chosen at every
    /// fit by [`grown_body4_degree`](Self::grown_body4_degree) and written here when the fit
    /// succeeds, set from a loaded file's basis; [`surrogate::BODY4_DEGREE_START`] otherwise. Only a
    /// basis with the body-4 block carries it (`default_basis`): a per-scene basis keeps the start.
    body4_degree: usize,
    /// Thinned candidates since the last kept fallback sample (thinning; the no-model cap).
    since_kept: usize,
    /// Per stored sample: its atoms' (species index, descriptor) under `desc_basis`.
    desc_cache: Vec<Vec<(usize, Vec<f64>)>>,
    desc_basis: Option<Basis>,
    /// What the last refit's recovery did, for the readouts.
    recovery: Option<String>,
    /// The background auditor (ruling S14), when one was attached.
    auditor: Option<Auditor>,
    /// An audit fell due while the auditor was busy, or the last audit's teacher answer was
    /// box-affected (M8a Task 2: not judged, so the cadence is not satisfied): dispatched — or,
    /// without an auditor, run in the step — at the next answered step.
    audit_due: bool,
    /// Consecutive box-affected (unjudged) audits since the last judged one
    /// (`TrainableStatus::audits_box_unjudged`; M8a Task 2).
    box_audits_unjudged: usize,
    /// The audit being dispatched is a draft-mode guess the strict policy would have refused
    /// (`AuditJob::draft_guess`); set by `evaluate` right before `dispatch_audit`.
    dispatch_draft_guess: bool,
    /// The flags of the step dispatching an audit (`AuditJob::flags`); set by `evaluate` right
    /// before `dispatch_audit`, taken by `send_job`.
    dispatch_flags: surrogate::history::PointFlags,
    /// The audit being dispatched is a calibration tick alone, not rule 8's (`AuditJob::tick_only`);
    /// set by `evaluate` right before `dispatch_audit`, taken by `send_job`.
    dispatch_tick_only: bool,
    /// A failed asynchronous audit untrusted the model: the next teacher evaluation books the
    /// Model → Teacher flip at its positions (module doc, "Audit").
    audit_flip: bool,
    last_audit: Option<AuditRecord>,
    /// Every judged audit's prediction against the teacher (M8a Task 1, `surrogate::history`): the
    /// evidence of a calibration. Written with the sidecar, installed from it, cleared by a forget.
    /// A record only; nothing in the policy reads it.
    audit_history: surrogate::AuditHistory,
    /// Fits installed so far (`fit_store`'s successes and installed sidecars): the fit generation
    /// that `last_audit_fit` is compared with. A readout only; it decides nothing.
    fits: u64,
    /// The model generation (activity Task 4 review I2): bumped wherever what the training
    /// views' curves read past the gate changes — every assignment of the model (a fit, an
    /// installed sidecar, a restored state, a forget) and a change of a policy threshold they read
    /// (not draft mode or the audit cadence, `set_policy`) — including a refit that
    /// leaves the model untrusted, which the surface `epoch` does not see. The views' cache key
    /// and their age line; it decides nothing.
    model_gen: u64,
    /// The fit id (trust tightening, spec 2026-11-18 §3): the id a history point records for the
    /// fit that judged it (`record_history_point`). Bumped only by a refit that solved new
    /// coefficients from new teacher data — to max(this, the history's largest id) + 1, so ids
    /// stay unique across sessions (`refit`; `fit_id_samples`) — never by a policy change, a
    /// forget, a restore (it brings back the restored fit's own id), an install (the file's id,
    /// else its history's largest), an attach (the collection history's largest) or a fork (the
    /// copy's refit brings its id back at the install). 0 only while no model has ever stood: a
    /// standing model's id is always ≥ 1 (a legacy install or attach with no id on record takes
    /// 1), so a point's `Some(0)` never occurs and an absent id is `None`, never 0.
    fit_id: u64,
    /// The fit id's watermark (fix round 1, I1): the store length when the fit id was last taken
    /// (the fitted prefix an install brought, the collection's length at an attach; 0 after a
    /// forget). A refit takes a new id only when its standing fit holds a sample at or past it —
    /// new teacher data — so forgetting and restoring a sample, a recovery's holds and releases or
    /// a degree-cap retry re-solve old data and add no fit. It can only under-count.
    fit_id_samples: usize,
    /// (`accumulated`, quarantine codes) when the standing fit was installed: what the training
    /// views call `Fitted` (Task 4 review, the released-sample ruling). A readout only.
    fit_members: (usize, Vec<u8>),
    /// The pair curves of one model generation, everything but the gate and the fence (`views`).
    curve_cache: Option<views::CurveCache>,
    /// The fit generation whose model `last_audit` judged (activity Task 3 review I1: after a
    /// refit the record describes an earlier fit, and the checklist must not read it as a check of
    /// this one). A readout only.
    last_audit_fit: u64,
    /// Notes produced outside a step (an audit verdict), handed out by the next evaluate.
    pending_notes: Vec<String>,
    /// This evaluation's own notes from before its answer (rule 0b's reason it declined): added to
    /// its answer and its Activity row, cleared at the next evaluate (review M5).
    step_notes: Vec<String>,
    /// The next `evaluate` goes to the teacher whatever the model says (the relax verification,
    /// `force_teacher_next`); consumed by that evaluation.
    force_teacher: bool,
    /// The latest `evaluate` was the stepping teacher's and nothing has used the teacher or moved
    /// the nuclei since: `sample_density` delegates (module doc, "Density samples").
    last_fallback: bool,
    /// Consecutive pass steps that failed with "GPU busy" and were kept for a retry
    /// ([`pass_step_is_transient`]); reset by a step that solved.
    pass_busy: u32,
    /// The session's GPU failure (module doc, "A failed GPU"): latched from the first `SolverError::GpuFailed` a
    /// teacher or the auditor returned, or from the stepping teacher's `device_failure`. Never cleared.
    gpu_failed: Option<String>,
    /// Teacher answers `sim::answer_refusal` refused (module doc, "A failed GPU"): each one also left an activity
    /// event with its reason; none was kept.
    refused_answers: u64,
    /// Side jobs (audits and refreshes) dispatched so far, the newest side density, and the
    /// isolated-atom densities that arrived (module doc, "Side densities").
    side_jobs: u64,
    side_display: Option<SideDisplay>,
    /// The newest Harris display payload (`SideKind::Harris`; module doc, "Side densities"), held
    /// beside `side_display` so that neither replaces the other.
    side_harris: Option<SideDisplay>,
    /// The one Harris display job waiting for the auditor (scene, profiles, side-job number): a
    /// newer request replaces it (the latest positions win; module doc, "Side densities").
    harris_queued: Option<(Scene, Arc<sim::HarrisAtoms>, u64)>,
    atom_densities: Vec<(u8, Display)>,
    /// The isolated-atom references that came back (`request_atom_reference`), not yet taken.
    atom_references: Vec<(u8, Result<AtomReference, ProviderError>)>,
    /// Side jobs waiting for the auditor (scene, provenance, kind, side-job number).
    side_queue: std::collections::VecDeque<(Scene, Provenance, JobKind, u64)>,
    /// An audit answer taken outside `evaluate`, judged at the next one (review M7).
    held_audit: Option<AuditAnswer>,
    /// The stray-atom signatures of this session and their states (`stray`; spec 2026-11-18
    /// §4.3, session-only): cleared with the model's basis or provenance (`stray_key`).
    stray_book: stray::StrayBook,
    /// The model's basis and surface (`lone::reference_settings_key`) the book's regions were built under
    /// (`stray_book_current`).
    stray_key: Option<(Basis, String)>,
    /// The composition check in flight on the auditor (at most one; `queue_stray_check`).
    stray_pending: Option<stray::StrayCheck>,
    /// Composition checks queued so far: the id their parts' jobs carry (`AuditJob::id`).
    stray_checks: u64,
    /// Failed stray-atom audits waiting for their attribution check while another is in flight (`stray`,
    /// "Audits of a cleared answer"; Task 5 fix round 1, M4), oldest first.
    stray_attributions: std::collections::VecDeque<stray::AttributionRequest>,
    /// The signatures whose fallbacks were told "no background teacher, so no composition check runs" this session:
    /// the note is written once per signature, not at every step (Task 6, the Task 3 review's minor).
    stray_no_auditor: std::collections::HashSet<stray::Signature>,
    /// The scene-edit generation: bumped by `reset` and `discontinuity` (the driver's edit
    /// signals, `Dynamics::after_edit`); a composition check queued before an edit is dropped.
    scene_gen: u64,
    /// Per stored sample: the teacher's face density (NaN = not known: `add_sample`, a teacher
    /// that reports none, or a file saved without densities — which `save_sidecar` writes only
    /// when no stored sample is box-affected).
    face: Vec<f64>,
    /// Fence voting (module doc): the last disagreeing answers inside a fence, and how many
    /// samples each fenced sample has had kept by a vote.
    fence_votes: Vec<FenceVote>,
    /// Lifting a forgotten sample's fence (module doc): the last fresh answers near only-forgotten
    /// fencing samples that **agreed** with the model; [`FENCE_VOTES`] consistent ones lift it.
    lift_votes: Vec<FenceVote>,
    fence_kept: BTreeMap<usize, u32>,
    /// Samples a vote kept that the next refit has not judged yet: (store index, the fenced
    /// samples it was kept next to). While one is pending its region takes no further vote.
    vote_pending: Vec<(usize, Vec<usize>)>,
    /// Consecutive consultations of a trusted model that fell back, with the last rule and gap.
    streak: usize,
    streak_reason: Option<Refusal>,
    /// Consecutive fallbacks refused by the class rule (4b), and the geometry of the latest — the
    /// teach-around prompt ([`SurrogateKs::teach_prompt`]).
    class_streak: usize,
    class_streak_at: Option<(Vec<[f64; 3]>, Vec<u8>, usize)>,
    streak_gap: Option<f64>,
    /// The trailing run of those fallbacks whose teacher answer was box-affected (never
    /// trained on), and the largest face density in it (`stuck`).
    streak_box: usize,
    streak_box_face: f64,
    /// The orbital view, for the background auditor's teacher only (module doc, "Side
    /// densities": the one orbital producer).
    orbital_view: OrbitalViewRequest,
    /// Whether the density shown includes the partial core ρ_pc (`ForceProvider::set_display_partial_core`):
    /// forwarded to the teacher and sent with every background job, like the orbital view.
    display_core: bool,
    /// The pseudised species of the latest teacher answer and the ionic tag it was solved on: a model answer of the
    /// same surface reports them (the Readouts' "valence" label and the badge keep their words between DFT steps).
    last_pseudised: Option<(String, Vec<sim::PseudoSummary>)>,
    /// The driver's display interrupt (`ForceProvider::set_display_interrupt`), sent with every
    /// background job so the auditor's teacher's unoccupied eigensolve yields to the user too
    /// (module doc, "Orbitals"; Task 4 re-review N1).
    display_interrupt: Option<sim::DisplayInterrupt>,
    /// The orbital shares' free-atom tables (`set_share_tables`), sent with every background job
    /// like the orbital view (module doc, "Orbitals").
    share_tables: Option<Arc<ShareTables>>,
    /// The running sampling pass (`Trainable::start_pass`; `crate::sampling`), and the last one's
    /// final progress.
    pass: Option<Sampler>,
    last_pass: Option<SamplingProgress>,
    /// Isolated-atom references solved by the passes or carried by a loaded model, keyed by the
    /// atom's solve settings (`crate::sampling`, "References"): a later pass at the same settings
    /// skips the scan. Survives a forget.
    ref_cache: AtomReferences,
    /// Messages for the user from the `Trainable` operations (`Trainable::take_notes`).
    user_notes: Vec<String>,
    /// Species whose isolated-atom reference scan converged nothing in a pass, with how it failed
    /// (module doc, "No model yet"); cleared by a reference energy for the species.
    failed_refs: BTreeMap<u8, crate::sampling::FailedReference>,
    /// The last refit's error while no model formed (module doc, "No model yet"); cleared by a fit.
    last_fit_error: Option<String>,
    /// The store length when the running (or last) sampling pass started: its samples are the
    /// pass-tagged ones from there (`finish_pass`'s coverage).
    pass_start: Option<usize>,
    /// What the last finished pass's samples became in the fit that stands (module doc,
    /// "Readiness"; fix (e)).
    pass_coverage: Option<PassCoverage>,
    /// The strict policy's verdict at the last evaluated geometry (module doc, "Readiness"): what
    /// `status` reports while the scene is still that geometry under the same fit and trust.
    strict_verdict: Option<StrictVerdict>,
    /// Per environment class: the fitted sample with the largest leave-one-out ratio and that
    /// ratio (the class rule's "worst sample", named by the readiness readout). A readout only.
    class_worst: Vec<Option<(usize, f64)>>,
    /// Per environment class: the fitted samples beyond 1× in the class rule's leave-one-out, worst
    /// first — the blockers a teach-around pass may target (module doc, "Teaching around the
    /// blockers").
    class_blockers: Vec<Vec<(usize, f64)>>,
    /// Teach-around rounds spent per environment class (0 pair, 1 many-body): at most
    /// `TEACH_ROUNDS` — per class, not per blocker: a round's own samples can become the next
    /// blockers (the user's water + H₂: #438 → #664 → #688), so a per-blocker count never ends.
    teach_rounds: BTreeMap<usize, u32>,
    /// The classes the running teach-around round was counted against (a cancel returns them).
    teach_classes: Vec<usize>,
    /// The running round's starting state (the outcome's words), and the rounds since the counter
    /// last restarted.
    teach_start: Option<teach::TeachStart>,
    teach_session: teach::TeachSession,
    /// The origin of the failed audit that untrusts the model (`sync_audit_failed`).
    audit_origin: AuditOrigin,
    /// The Activity view's live probe (`with_probe`): the refit phase is reported on it.
    probe: Option<Arc<sim::ActivityProbe>>,
    /// One event per teaching solve and the lifetime counts (`ledger`, activity spec A2–A3).
    ledger: Ledger,
    /// The shared collection this provider learns into (M8a Task 4a; `shared` module doc), if
    /// attached: the process's one writable handle of it.
    shared: Option<shared::SharedCollection>,
    /// The attached collection's migration from its predecessor (`shared` module doc, "The migration"; plan 2026-11-13
    /// Task 14): the offer, or the `migration.txt` it keeps. `None` without a collection or anything to migrate.
    migration: Option<shared::MigrationSlot>,
    /// The re-solved samples (code 9, `Q_RESOLVED`) whose last review **kept** them (`quarantine` module,
    /// `review_resolved`; Fermi–Dirac spec §7.6 step 4): store indices. Such a sample fences (`fences`) while its code
    /// is 9. Read back from `migration.txt`'s review log at attach (`shared::find_migration`, ruling R13), remapped by a
    /// purge, cleared with the collection.
    review_kept: std::collections::BTreeSet<usize>,
    /// The store's stamp (`review_stamp_now`) when the re-solved samples were last reviewed: the review repeats only
    /// when it moved. `None`: not reviewed in this session (an attach starts here).
    review_stamp: Option<u64>,
    /// The migration log's review entries (or its state) changed since `migration.txt` was written
    /// (`flush_migration_log`).
    review_log_dirty: bool,
    /// Tests only: how many reviews of the re-solved samples ran (`review_resolved`).
    #[cfg(test)]
    pub(super) reviews_run: usize,
    /// Tests only: an attach runs no refit (`shared::attach_inner`), so what the attach itself installed is seen.
    #[cfg(test)]
    pub(super) attach_refit_off: bool,
    /// Tests only: a collection fitted without the body-4 block (the basis collections had before
    /// M8a Task 5 — the user's stuck collection's repro is pinned to it).
    #[cfg(test)]
    body4_off: bool,
    /// Tests only: the next migrated append stops after its sample record (`shared`, the status window).
    #[cfg(test)]
    status_window_crash: bool,
    /// Tests only: the next migrated append fails before it writes anything.
    #[cfg(test)]
    migrated_append_fails: bool,
    /// Tests only: the body-4 degree every fit takes instead of the growth rule's
    /// ([`grown_body4_degree`](Self::grown_body4_degree)), in collection mode.
    #[cfg(test)]
    pub(super) body4_force: Option<usize>,
    /// Tests only: the recovery's rounds rank by the training residual alone, as before spec R1
    /// (`training_offenders`, no leave-one-out rounds first) — the premise a test compares with.
    #[cfg(test)]
    training_ranking: bool,
    /// A shared collection's cutoff (Bohr): [`surrogate::COLLECTION_R_CUT`] unless the reach
    /// measurement set another ([`SurrogateKs::with_collection_r_cut`]).
    collection_r_cut: f64,
    /// Why the collection refused the sample just stored (taken by `keep_sample` for the step's
    /// note; `shared::SurrogateKs::take_collection_note`).
    collection_note: Vec<String>,
    /// The model file the model was loaded from (`TrainableStatus::model_source`).
    source_file: Option<std::path::PathBuf>,
    /// The species of the specialised model loaded from `source_file` (M8a Task 7,
    /// `shared::SurrogateKs::load_specialised`); empty otherwise. Cleared with `source_file`.
    specialised: Vec<u8>,
    /// The isolated-atom reference jobs queued on or running in the auditor, one per (Z, box n),
    /// with who asked (the driver, whose answers are always handed on, or the provider itself) and
    /// the collection the requesting scene shared (`shared` module doc, "References per box").
    ref_jobs: Vec<shared::RefJob>,
    /// The change of the reported energy at the last evaluated positions that a box reference's
    /// arrival made (the canonical surface moved under a teacher-surface report): booked into the
    /// next evaluation's `refit_jump` (fix round 3; `shared::book_reference_arrival`).
    offset_jump: f64,
    /// The refit running on its own thread (`background` module doc), if any.
    background: Option<background::BackgroundRefit>,
    /// This provider is a background refit's copy: its `refit` writes nothing to the collection
    /// (the live provider does after the install).
    refit_fork: bool,
    /// A background refit's copy: set when the live provider let it go (dropped or replaced) —
    /// its refit stops at its next fit (`fit_store_with`).
    refit_cancel: Option<Arc<std::sync::atomic::AtomicBool>>,
    /// The running refit's stages so far (`refit_stage`; `sim::refit_eta`), `None` outside a
    /// refit: a fit or leave-one-out pass outside one (a view's) is not a stage.
    refit_counts: Option<sim::RefitStages>,
    /// A background refit's copy: the live provider's probe, for the stages only (the phase and
    /// the history are the live provider's).
    stage_probe: Option<Arc<sim::ActivityProbe>>,
}

impl SurrogateKs {
    /// A stage of the running refit starts (`sim::refit_eta`, "The countdown"): counted, and
    /// published on the probe; a no-op outside a refit. Bookkeeping only.
    pub(super) fn refit_stage(&mut self, stage: sim::RefitStage) {
        let Some(c) = self.refit_counts.as_mut() else { return };
        match stage {
            sim::RefitStage::Fit => c.fits += 1,
            sim::RefitStage::LeaveOneOut => c.loo += 1,
            sim::RefitStage::Recovery => c.recovery = true,
        }
        if let Some(p) = self.probe.as_ref().or(self.stage_probe.as_ref()) {
            p.refit_stage(stage);
        }
    }

    /// The species of the samples held (a refit's record and estimate key).
    pub(super) fn store_species(&self) -> usize {
        let mut z: Vec<u8> = self.store.as_ref().map(|st| st.samples.iter().flat_map(|s| s.z.iter().copied()).collect()).unwrap_or_default();
        z.sort_unstable();
        z.dedup();
        z.len()
    }

    /// A refit that ended with a fit (`refit`, or a background one's install or discard): its
    /// record joins the probe's history (`sim::refit_eta`), written to the attached collection's
    /// `refit_times.csv` when it is writable. Never read by a fit or a verdict; without a probe
    /// nothing is kept.
    pub(super) fn record_refit(&mut self, wall_s: f64, samples: usize, species: usize, background: bool, c: sim::RefitStages) {
        let Some(p) = self.probe.clone() else { return };
        let at_unix_s = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        let rec = sim::RefitRecord {
            at_unix_s,
            wall_s,
            samples: samples.min(u32::MAX as usize) as u32,
            species: species.min(u32::MAX as usize) as u32,
            background,
            kind: if c.recovery { sim::RefitKind::Recovery } else { sim::RefitKind::Plain },
            fits: c.fits,
            loo: c.loo,
        };
        let history = p.record_refit(rec);
        let failed = match self.shared.as_ref().filter(|sh| !sh.coll.read_only) {
            Some(sh) => sh.coll.save_refit_times(&sim::refit_eta::encode(&history)).err(),
            None => None,
        };
        if let Some(e) = failed {
            self.note_write_failure(&e);
        }
    }

    /// A refit's outcome in `Trainable::refit`'s words (one line), noting the error while no model
    /// formed.
    fn refit_summary(&mut self, r: Result<FitReport, FitError>) -> Result<String, String> {
        if let Err(e) = &r {
            if self.model.is_none() {
                self.last_fit_error = Some(e.to_string());
            }
        }
        // a fit without energy rows measured no energy rmse (Task 10a review, finding 5)
        r.map(|r| {
            let e = if r.has_energy_rows() { format!("rmse_E {:.1e} Ha/atom", r.rmse_energy_per_atom) } else { format!("rmse_E — ({})", surrogate::fit::NO_ENERGY_ROWS) };
            format!("{} samples, {e}, rmse_F {:.1e} Ha/Bohr", r.n_samples, r.rmse_force)
        })
        .map_err(|e| e.to_string())
    }

    pub fn new(teacher: Box<dyn ForceProvider>, policy: FallbackPolicy) -> SurrogateKs {
        // `name()` returns `&'static str`: a few bytes leaked once per provider built (the
        // worker builds one per backend switch), not per evaluation
        let name: &'static str = Box::leak(format!("surrogate({})", teacher.name()).into_boxed_str());
        SurrogateKs {
            teacher,
            name,
            teacher_tol: 0.0,
            policy,
            store: None,
            accumulator: None,
            accumulated: 0,
            refs: BTreeMap::new(),
            legacy_refs: std::collections::BTreeSet::new(),
            refs_dirty: false,
            species_dirty: false,
            model: None,
            report: None,
            envelope: None,
            null_space: None,
            null_space_pending: false,
            trial_fit: false,
            #[cfg(test)]
            fit_log: Vec::new(),
            #[cfg(test)]
            confirm_fail: None,
            #[cfg(test)]
            confirm_calls: 0,
            #[cfg(test)]
            forced_fail_gen: None,
            #[cfg(test)]
            confirm_failed: None,
            active_set: None,
            training_residuals: None,
            misfit: None,
            class_loo: Vec::new(),
            class_memory: BTreeMap::new(),
            class_judge: None,
            audit_failed: None,
            audit_force: None,
            failed_audits: Vec::new(),
            audit_fences: Vec::new(),
            audit_burst: 0,
            audit_atoms: 0,
            verdict_counts: Vec::new(),
            epoch: 0,
            pending_refit: false,
            fit_inputs_gen: 0,
            fits_in_background: false,
            pass_fit_due: false,
            pass_tail: None,
            first_fit_due: false,
            pending_jump: None,
            last: None,
            consults: 0,
            cadence_steps: 0,
            tick_every: TICK_EVERY,
            tick_steps: 0,
            ticks: TickCounts::default(),
            tick_in_flight: false,
            tick_dead: 0,
            tick_dead_fixed: None,
            job_ms_ema: None,
            step_ms_ema: None,
            last_in_envelope_at: None,
            last_variance: None,
            fit_floor_cache: std::sync::Mutex::new(BTreeMap::new()),
            triple_cache: std::sync::Mutex::new(None),
            richer_floor_cache: std::sync::Mutex::new(None),
            #[cfg(test)]
            richer_floor_counts: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(test)]
            class_loo_refreshes: 0,
            variance_checks: 0,
            answered: 0,
            quarantine: Vec::new(),
            lone_codes: BTreeMap::new(),
            degree_cap: None,
            body4_degree: surrogate::BODY4_DEGREE_START,
            since_kept: THIN_EVERY,
            desc_cache: Vec::new(),
            desc_basis: None,
            recovery: None,
            auditor: None,
            audit_due: false,
            box_audits_unjudged: 0,
            dispatch_draft_guess: false,
            dispatch_flags: surrogate::history::PointFlags::default(),
            dispatch_tick_only: false,
            audit_flip: false,
            force_teacher: false,
            last_fallback: false,
            pass_busy: 0,
            gpu_failed: None,
            refused_answers: 0,
            last_audit: None,
            audit_history: surrogate::AuditHistory::default(),
            fits: 0,
            last_audit_fit: 0,
            model_gen: 0,
            fit_id: 0,
            fit_id_samples: 0,
            fit_members: (0, Vec::new()),
            curve_cache: None,
            pending_notes: Vec::new(),
            step_notes: Vec::new(),
            side_jobs: 0,
            side_display: None,
            side_harris: None,
            harris_queued: None,
            atom_densities: Vec::new(),
            atom_references: Vec::new(),
            side_queue: std::collections::VecDeque::new(),
            held_audit: None,
            stray_book: stray::StrayBook::default(),
            stray_key: None,
            stray_pending: None,
            stray_checks: 0,
            stray_attributions: std::collections::VecDeque::new(),
            stray_no_auditor: std::collections::HashSet::new(),
            scene_gen: 0,
            face: Vec::new(),
            fence_votes: Vec::new(),
            lift_votes: Vec::new(),
            fence_kept: BTreeMap::new(),
            vote_pending: Vec::new(),
            streak: 0,
            streak_reason: None,
            class_streak: 0,
            class_streak_at: None,
            streak_gap: None,
            streak_box: 0,
            streak_box_face: 0.0,
            orbital_view: OrbitalViewRequest::default(),
            display_core: false,
            last_pseudised: None,
            display_interrupt: None,
            share_tables: None,
            pass: None,
            last_pass: None,
            ref_cache: AtomReferences::default(),
            user_notes: Vec::new(),
            failed_refs: BTreeMap::new(),
            last_fit_error: None,
            probe: None,
            ledger: Ledger::default(),
            strict_verdict: None,
            class_worst: Vec::new(),
            class_blockers: Vec::new(),
            deferred_trust: None,
            loo_ratios: None,
            #[cfg(test)]
            eager_trust_passes: false,
            #[cfg(test)]
            skip_recovery: false,
            teach_rounds: BTreeMap::new(),
            teach_classes: Vec::new(),
            teach_start: None,
            teach_session: teach::TeachSession::default(),
            audit_origin: AuditOrigin::Audit,
            pass_start: None,
            pass_coverage: None,
            shared: None,
            migration: None,
            review_kept: std::collections::BTreeSet::new(),
            review_stamp: None,
            review_log_dirty: false,
            #[cfg(test)]
            reviews_run: 0,
            #[cfg(test)]
            attach_refit_off: false,
            #[cfg(test)]
            body4_off: false,
            #[cfg(test)]
            status_window_crash: false,
            #[cfg(test)]
            migrated_append_fails: false,
            #[cfg(test)]
            body4_force: None,
            #[cfg(test)]
            training_ranking: false,
            collection_r_cut: surrogate::COLLECTION_R_CUT,
            collection_note: Vec::new(),
            source_file: None,
            specialised: Vec::new(),
            ref_jobs: Vec::new(),
            offset_jump: 0.0,
            background: None,
            refit_fork: false,
            refit_cancel: None,
            refit_counts: None,
            stage_probe: None,
        }
    }

    /// Records the teacher's SCF tolerance in the samples' provenance (module doc).
    pub fn with_teacher_tol(mut self, tol: f64) -> SurrogateKs {
        self.teacher_tol = tol;
        self
    }

    /// Test hook: the calibration ticks' cadence in in-envelope steps (module doc, "Calibration
    /// ticks"; [`TICK_EVERY`] by default), **0 = no ticks**. It exists so measurements of the
    /// model's own dynamics (its frequency, NVE, tier) can switch off the in-step ticks a provider
    /// without an auditor serves — each one a teacher step on the trajectory (plan 2026-11-19
    /// Task 9 ruling). Not for the app.
    #[doc(hidden)]
    pub fn with_tick_every(mut self, n: usize) -> SurrogateKs {
        self.tick_every = n;
        self
    }

    /// A shared collection's fit uses the cutoff `r_cut` (Bohr) instead of
    /// [`surrogate::COLLECTION_R_CUT`]: the reach measurement of the transfer test (M8a Task 6,
    /// `validate/tests/transfer.rs`), which re-fits one collection's samples at another reach. A
    /// per-scene model keeps `Basis::new`'s cutoff whatever this says. Set before
    /// [`attach`](Self::attach): the fit's basis follows it at the next rebuild.
    pub fn with_collection_r_cut(mut self, r_cut: f64) -> SurrogateKs {
        assert!(r_cut.is_finite() && r_cut > 0.0, "with_collection_r_cut: r_cut = {r_cut}");
        self.collection_r_cut = r_cut;
        self
    }

    /// Report the refit phase on `p` (the Activity view, spec §4.3 A1): every
    /// [`refit`](SurrogateKs::refit), the pass's own included, marks it for its duration. The
    /// teachers' solves are reported by the teachers themselves (`crate::ks::GpuKs::with_probe`),
    /// attached by whoever builds them.
    pub fn with_probe(mut self, p: Arc<sim::ActivityProbe>) -> SurrogateKs {
        self.probe = Some(p);
        self
    }

    /// The accumulator's basis (`None` before the first refit).
    pub fn basis(&self) -> Option<&Basis> {
        self.accumulator.as_ref().map(|a| &a.basis)
    }

    pub fn model(&self) -> Option<&Model> {
        self.model.as_ref()
    }

    /// The last fit's report (`None` before the first refit).
    pub fn report(&self) -> Option<&FitReport> {
        self.report.as_ref()
    }

    /// Surface counter: changes when a refit moves the energy surface (module doc); the same
    /// number as [`ForceProvider::surface_epoch`].
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// The training-descriptor envelope of the current fit (rule 5; `None` before the first
    /// refit).
    pub fn envelope(&self) -> Option<&Envelope> {
        self.envelope.as_ref()
    }

    /// Whether a fallback step has been kept since the last refit.
    pub fn pending_refit(&self) -> bool {
        self.pending_refit
    }

    /// What a refit fits changed (`fit_inputs_gen`): a refit is due. Every change from outside a
    /// refit arms it through here; a refit's own outcome (a discard, a failure) never does.
    pub(super) fn arm_refit(&mut self) {
        self.pending_refit = true;
        self.fit_inputs_gen += 1;
    }

    /// `Trainable::fit_inputs_gen`.
    pub fn fit_inputs_gen(&self) -> u64 {
        self.fit_inputs_gen
    }

    /// `Trainable::first_fit_ready`: a fit this provider left to the caller is due now, no pass or
    /// refit running — an attach's first fit (`fits_in_background`; no model yet and a fit can
    /// form from the samples held, the "No model yet" state whose Retry would fit at once), or a
    /// pass's end fit (`pass_fit_due`, quiet refits round 2: with or without a model).
    pub fn first_fit_ready(&self) -> bool {
        (self.first_fit_due && self.first_fit_can_form()) || (self.pass_fit_due && self.pass.is_none() && self.background.is_none())
    }

    /// No model, no pass or refit running, and a fit can form from the samples held now.
    fn first_fit_can_form(&self) -> bool {
        self.model.is_none() && self.pass.is_none() && self.background.is_none() && self.no_model().is_some_and(|nm| nm.retry && nm.missing.is_empty())
    }

    /// `Trainable::request_first_fit`: the Retry of a fit that can form now is left to the caller
    /// (in the background) — `first_fit_due` — instead of [`retry_fit`](Self::retry_fit)'s fit in
    /// place; `false`, nothing changed, otherwise.
    pub fn request_first_fit(&mut self) -> bool {
        if !self.first_fit_can_form() {
            return false;
        }
        self.first_fit_due = true;
        // due: the caller starts it (no change of what it fits — the caller counts the Retry)
        self.pending_refit = true;
        true
    }

    /// The user's Forget / Restore changes no fit holds yet (2026-11-14, the batched forget): made
    /// since the fit that stands, or — while a background refit runs — since it started (the ones
    /// before are its). Each one made a refit due (`pending_refit`); the worker's pending-refit
    /// debounce starts that refit, and the Model tab's banner counts them.
    ///
    /// **Derived, never counted** (review I1, 2026-11-14): the samples whose forgotten state now
    /// differs from their state in the fit it is measured against — the running background refit's
    /// start (`background::Snapshot::quarantine`) while one runs, else the standing fit's members
    /// (`fit_members`). So a Forget then a Restore of one sample is no change, and every reset —
    /// a fit, a forget of the model, a load, an attach, a pass's fit — is right by construction.
    /// 0 without a model (nothing to refit).
    pub fn user_changes_pending(&self) -> usize {
        if self.model.is_none() {
            return 0;
        }
        let base: &[u8] = match self.background.as_ref() {
            Some(b) => b.snap_quarantine(),
            None => &self.fit_members.1,
        };
        let was = |i: usize| base.get(i).is_some_and(|&q| q == Q_FORGOTTEN || q == Q_FORGOTTEN_UNFENCED);
        (0..self.quarantine.len()).filter(|&i| self.is_forgotten(i) != was(i)).count()
    }

    /// How many times the variance (rules 6–7) has been evaluated.
    pub fn variance_checks(&self) -> u64 {
        self.variance_checks
    }

    /// Calibration ticks taken that left a history point (module doc, "Calibration ticks"): at a
    /// fallback, in the step, or by the auditor once its answer is judged. A readout; the ticks
    /// taken without a point are [`TickCounts::unrecorded`] — among them those whose point was
    /// deferred to a stray attribution check (its point arrives later with `TICK`).
    pub fn ticks_taken(&self) -> u64 {
        self.ticks.recorded
    }

    /// Calibration ticks that fell due and were skipped: the auditor busy with other work
    /// ([`TickCounts::skipped_busy`], whatever answered the step) or the tick lost
    /// ([`TickCounts::lost`]). A readout (the Model tab's "calibration checks skipped").
    pub fn ticks_skipped(&self) -> u64 {
        self.ticks.skipped_busy + self.ticks.lost
    }

    /// Calibration ticks served by a teacher solve in the step (no auditor). A readout.
    pub fn ticks_in_step(&self) -> u64 {
        self.ticks.in_step
    }

    /// Every count of the calibration ticks: each tick that fell due is in exactly one of
    /// `recorded`, `unrecorded`, `skipped_busy` and `lost`, or still in flight. A readout.
    pub fn tick_counts(&self) -> TickCounts {
        self.ticks
    }

    /// The tick counts the readouts show ([`sim::TrainableStatus::ticks`], plan Task 8; Task 7
    /// review M5: through the status, never a downcast): the outcomes with their causes, the
    /// in-step ones and the due count.
    pub fn calibration_ticks(&self) -> sim::CalibrationTicks {
        let t = self.ticks;
        sim::CalibrationTicks { recorded: t.recorded, unrecorded: t.unrecorded, skipped_busy: t.skipped_busy, lost: t.lost, in_step: t.in_step, due: t.due }
    }

    /// Books what became of a tick that fell due (the one place the counts move).
    pub(super) fn tick_outcome(&mut self, o: TickOutcome) {
        match o {
            TickOutcome::Recorded => self.ticks.recorded += 1,
            TickOutcome::Unrecorded => self.ticks.unrecorded += 1,
            TickOutcome::SkippedBusy => self.ticks.skipped_busy += 1,
            TickOutcome::Lost => self.ticks.lost += 1,
        }
    }

    /// A teacher solve cancelled: the step did not happen, so whatever it booked toward the ticks
    /// is undone — `before` = (`tick_steps`, `tick_dead`, the counts) as they stood at its decision —
    /// and the retry is counted afresh (and is the tick, if this one was): the Task 7 review's M2.
    /// The timing EMAs are not rolled back (wall clock, not a count). Returns whether it was
    /// cancelled.
    fn untick_cancelled(&mut self, r: &Result<ForceResult, ProviderError>, before: (u64, u64, TickCounts)) -> bool {
        let cancelled = matches!(r, Err(ProviderError::Cancelled));
        if cancelled {
            (self.tick_steps, self.tick_dead, self.ticks) = before;
        }
        cancelled
    }

    /// The dead time after a due tick, in in-envelope steps (module doc, "Calibration ticks"; the
    /// controller's re-ruling on the Task 7 re-review's R1): the fixed one a test set, else
    /// D = ⌈EMA(auditor job wall ms) / EMA(in-envelope step wall ms)⌉ — both EMAs over every job of
    /// every kind and every in-envelope step, never selected by an outcome — and 0 without both (no
    /// auditor, or no job measured yet: no job to outlive).
    fn tick_dead_time(&self) -> u64 {
        if let Some(d) = self.tick_dead_fixed {
            return d;
        }
        match (self.job_ms_ema, self.step_ms_ema) {
            (Some(job), Some(step)) if step > 0.0 && (job / step).is_finite() => (job / step).ceil().max(0.0) as u64,
            _ => 0,
        }
    }

    /// One in-envelope step's wall time, from the previous in-envelope step's decision (capped at
    /// `TICK_STEP_SAMPLE_MAX_MS`, so a pause or an edit does not stand for a step), into its EMA.
    fn time_in_envelope_step(&mut self) {
        let now = Instant::now();
        if let Some(prev) = self.last_in_envelope_at {
            let ms = (now - prev).as_secs_f64() * 1e3;
            self.step_ms_ema = Some(ema(self.step_ms_ema, ms.min(TICK_STEP_SAMPLE_MAX_MS)));
        }
        self.last_in_envelope_at = Some(now);
    }

    /// One auditor job's wall time (any kind: an audit, a tick, a display, a reference) into its EMA.
    pub(super) fn time_auditor_job(&mut self, ms: f64) {
        if ms.is_finite() && ms >= 0.0 {
            self.job_ms_ema = Some(ema(self.job_ms_ema, ms));
        }
    }

    /// The next `evaluate` goes to the teacher regardless of the model (M6 spec §6.2: the relax
    /// verification). The answer takes the normal fallback path — kept as a sample (never
    /// thinned), reported as a model-surface fallback with the teacher gap when the model could
    /// answer — so nothing else changes; draft mode does not override it.
    pub fn force_teacher_next(&mut self) {
        self.force_teacher = true;
    }

    /// The teacher, for work the policy must not see: the sampling pass (`crate::sampling`)
    /// solves its geometries with it directly — they are training data, not steps.
    /// Its state is no longer the last step's afterwards: `sample_density` says `None` until the
    /// next fallback.
    pub fn teacher_mut(&mut self) -> &mut dyn ForceProvider {
        self.last_fallback = false;
        self.teacher.as_mut()
    }

    /// The session's GPU failure (module doc, "A failed GPU"), latched from the stepping teacher's device when it
    /// was not yet: `Some(cause)` once any teacher of this provider reported it.
    pub fn gpu_failure(&mut self) -> Option<String> {
        if self.gpu_failed.is_none() {
            if let Some(why) = self.teacher.device_failure() {
                self.latch_gpu_failure(why);
            }
        }
        self.gpu_failed.clone()
    }

    /// [`gpu_failure`](Self::gpu_failure) as the error every evaluation and pass step returns from then on.
    pub(crate) fn gpu_failure_error(&mut self) -> Option<ProviderError> {
        self.gpu_failure().map(|why| ProviderError::Solver(sim::SolverError::GpuFailed(why)))
    }

    /// Records the session's GPU failure (module doc, "A failed GPU"), once: a note, and nothing queued for the
    /// auditor is sent any more (a job in flight comes back failing and is handled as a failed job).
    pub(crate) fn latch_gpu_failure(&mut self, why: String) {
        if self.gpu_failed.is_some() {
            return;
        }
        self.user_notes.push(format!(
            "surrogate: the GPU failed ({why}) — every teacher solve is refused and the model answers no more steps; nothing from the failed solve was kept"
        ));
        self.audit_due = false;
        self.side_queue.clear();
        self.harris_queued = None;
        self.gpu_failed = Some(why);
    }

    /// A teacher error: latched when it is the session's GPU failure ([`latch_gpu_failure`](Self::latch_gpu_failure)).
    pub(crate) fn note_teacher_error(&mut self, e: &ProviderError) {
        if let Some(why) = e.gpu_failure() {
            self.latch_gpu_failure(why.to_string());
        }
    }

    /// One teacher answer `sim::answer_refusal` refused (module doc, "A failed GPU"): counted and noted; the caller
    /// leaves the activity event with the reason and keeps nothing.
    pub(crate) fn count_refused_answer(&mut self, why: &str) {
        self.refused_answers += 1;
        self.user_notes.push(format!("surrogate: a teacher answer was refused ({why}) — not kept ({} refused this session)", self.refused_answers));
    }

    /// Teacher answers refused by `sim::answer_refusal` so far (module doc, "A failed GPU").
    pub fn refused_answers(&self) -> u64 {
        self.refused_answers
    }

    /// The teacher's step: evaluate, keep the sample, build the result on the right surface
    /// (module doc, "One-surface accounting" and "Audit"), and leave one activity event (`ledger`):
    /// the rule that sent the step here, what `admit` did with the answer, or `Failed` when the
    /// teacher failed (a cancel leaves none — the step is retried).
    fn teach(&mut self, scene: &Scene, h: Handoff, t0: Instant, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        let solve_started = Instant::now();
        let r = match self.teacher.evaluate(scene, should_abort) {
            Ok(r) => r,
            Err(e) => {
                self.note_teacher_error(&e);
                if !matches!(e, ProviderError::Cancelled) {
                    let mut ev = TeachEvent::new(origin_of(h.audit, h.reason.rule), h.reason.rule, TeachOutcome::Failed);
                    ev.atom = h.reason.atom;
                    ev.detail = format!("{} — the teacher failed: {e}", h.reason.text);
                    ev.n_atoms = h.pos.len();
                    // the time the failed solve took; its SCF steps are not reported (None)
                    ev.solve_ms = solve_started.elapsed().as_secs_f64() * 1e3;
                    self.push_event(ev);
                }
                return Err(e);
            }
        };
        // the gpu-fault fix (module doc, "A failed GPU"): a physically empty answer — not converged, no iteration, a
        // density without the scene's electrons — is neither kept nor used for the step (a non-finite one keeps its own
        // handling: the store never keeps it, and as an audit it fails: activity Task 3b)
        if let Some(why) = sim::scf_refusal(&r.stats) {
            self.count_refused_answer(&why);
            let mut ev = TeachEvent::new(origin_of(h.audit, h.reason.rule), h.reason.rule, TeachOutcome::Failed);
            ev.atom = h.reason.atom;
            ev.detail = format!("{} — the teacher's answer was refused ({why}): not kept, not used", h.reason.text);
            ev.n_atoms = h.pos.len();
            ev.solve_ms = solve_started.elapsed().as_secs_f64() * 1e3;
            ev.scf_steps = Some(r.stats.scf_steps);
            self.push_event(ev);
            return Err(ProviderError::Solver(sim::SolverError::Numerical(format!("the teacher's answer was refused ({why}); nothing was kept"))));
        }
        // a rule-5a fallback of a bonded part plus stray atoms the floor lets through: its answer is a composition
        // check's scene answer (`stray` module doc, "The checks on the auditor"), raw — before the canonical surface
        let stray_check = (h.reason.rule == TeachRule::IsolatedAtom && !h.audit && h.model.is_some() && scene.train_mask().is_none())
            .then(|| self.stray_split(&h.pos, &h.z, &h.prov))
            .flatten()
            .filter(|(_, _, floor)| floor.is_none())
            .map(|(split, sig, _)| (split, sig, stray::part_answer(&r, r.energy, h.prov.h)));
        // in collection mode every comparison and report is on the collection's canonical surface
        // (`shared` module doc, "References per box"; the identity per scene)
        let mut r = r;
        let raw_energy = r.energy;
        r.energy = self.to_canonical(&h.z, h.prov.grid_n, r.energy);
        // the event's record of the solve (read before `r` and `h` are consumed below)
        let (teacher_ms, scf, ts) = (r.stats.solve_ms as f64, r.stats.scf_steps, r.stats.energy.ts);
        // the event's atom count is N_e (`audit_n`), what the step's per-atom gap divides by (the Activity hover's)
        let (h_audit, rule_code, atom, n_atoms) = (h.audit, h.reason.rule, h.reason.atom, self.audit_n(&h.pos));
        let history_code = h.history_rule;
        let gap = h.model.as_ref().map(|m| r.energy - m.energy);
        let mut notes = Vec::new();
        let face = r.stats.face_density;
        let multiplicity = r.stats.multiplicity;
        let model_answer = h.model.as_ref().map(|m| (m.energy, &m.forces[..]));
        let essential = h.model.is_none() && h.keep_sample && !h.thin;
        let rule = box_affected_face_density(h.prov.h);
        let unfit = r.stats.untrainable();
        // the origin: an audit's answer is tagged as one (fix (b), 2026-10-18; an audit in the step
        // is never a draft-mode one — draft mode audits only in the background)
        let tag = if h.audit { SampleTag::Audit } else { SampleTag::Fallback };
        let outcome = if h.shell_fence {
            notes.push(format!(
                "surrogate: shell fence — the teacher's answer drives this step and is not fitted{} (the model's cutoff cannot represent it; a fitted shell row would be a structural outlier)",
                if h.audit { " nor judged as an audit" } else { "" }
            ));
            TeachOutcome::ShellNotKept
        } else {
            // a capped segment's answer is kept with the scene's mask (module doc, "Masked scenes")
            self.admit_and_keep(&h.pos, &h.z, r.energy, raw_energy, &r.forces, face, multiplicity, model_answer, &h.near, h.near_misfit, h.keep_sample, h.thin && !h.audit, h.dedup && !h.audit, essential, &h.prov, unfit, tag, scene.train_mask(), &mut notes)
        };
        if let Some((split, sig, part)) = stray_check {
            let kept = outcome.is_kept().then(|| self.samples().saturating_sub(1));
            let label = stray::signature_label(&sig);
            if self.auditor.is_none() {
                // controller ruling C: a synchronous fragment solve would stall the step; said once per signature
                if self.stray_no_auditor.insert(sig) {
                    notes.push(format!("surrogate: stray atoms ({label}) — no background teacher, so no composition check runs; DFT answers (said once this session)"));
                }
            } else if self.queue_stray_check(scene, part, split, sig, kept) {
                notes.push(format!("surrogate: stray atoms ({label}) — a composition check (the bonded part alone, each stray atom alone) is queued on the background teacher"));
            }
        }
        // an audit at a fenced-band geometry is not a judgement of the model (strict mode refuses
        // it there): the teacher answered, nothing is judged
        let judged_audit = h.audit && !h.shell_fence;
        // the per-atom tiers below (the over-tier report, the audit) per bonded atom at a scene with stray atoms
        // (`stray` module doc, "Audits of a cleared answer"), else per atom
        let n = self.audit_n(&h.pos) as f64;
        let mut stats = r.stats;
        if stats.pseudised.is_empty() && crate::pseudo::is_all_electron_tag(&h.prov.ionic) {
            self.last_pseudised = None;
        } else if !stats.pseudised.is_empty() {
            self.last_pseudised = Some((h.prov.ionic.clone(), stats.pseudised.clone()));
        }
        stats.source = Source::Fallback;
        stats.uncertainty = h.model.as_ref().and_then(|m| m.uncertainty);
        // a consulted model that fell back (an audit is not the model failing to answer)
        let boxed = face.filter(|&rho| rho > rule);
        // the history's fallback point (M8a Task 1 fix round 1, the physics review's I3): the
        // model's prediction against the teacher's answer at a geometry a rule refused — evidence
        // at the edge of what the rules accept. Not inside a fenced shell band (nothing is judged
        // there, an audit included) and not for a non-finite prediction (no finite stds exist)
        if let Some(m) = h.model.as_ref().filter(|_| !h.audit && !h.shell_fence && h.reason.rule != TeachRule::NonFiniteAnswer) {
            let kind = surrogate::history::PointKind::Fallback { rule: h.history_rule };
            self.record_history_point(kind, &h.pos, &h.z, m.stds.as_ref().map(|(e, f, fe)| (*e, &f[..], &fe[..])), m.energy, &m.forces, r.energy, &r.forces, boxed.is_some(), false, unfit.is_some(), h.flags, m.energy_std_epistemic);
        }
        if h.model.is_some() && !h.audit {
            self.streak += 1;
            self.streak_reason = Some(h.reason.clone());
            if let TeachRule::Class(c) = h.reason.rule {
                self.class_streak += 1;
                let class = match c {
                    sim::activity::EnvClass::Pair => 0,
                    sim::activity::EnvClass::ManyBody => 1,
                };
                self.class_streak_at = Some((h.pos.clone(), h.z.clone(), class));
            } else {
                self.class_streak = 0;
            }
            // the trailing run of box-affected answers: none of them is trained on (`stuck`)
            match boxed {
                Some(rho) => {
                    self.streak_box += 1;
                    self.streak_box_face = self.streak_box_face.max(rho);
                }
                None => {
                    self.streak_box = 0;
                    self.streak_box_face = 0.0;
                }
            }
        } else if h.model.is_none() {
            self.streak = 0;
            self.streak_box = 0;
            self.streak_box_face = 0.0;
            self.class_streak = 0;
        }
        // a fallback whose gap exceeds the energy tier is reported on the teacher's surface too
        // (not an audit: a failed one untrusts as well, below): the model's energy there is not
        // what the model claims, and E_total must stay within the tier of the teacher's
        // ("not within the tier": a NaN gap is reported on the teacher's surface, Task 3b)
        let over_tier = !h.audit && h.model.as_ref().is_some_and(|m| !((r.energy - m.energy).abs() / n <= self.policy.energy_std_per_atom));
        let to_teacher = h.teacher_surface || over_tier;
        let flip_back = to_teacher && h.model.is_some();
        let mut surface_forces: Option<Vec<[f64; 3]>> = None;
        let (energy, on_model) = match h.model {
            Some(m) if to_teacher => {
                // rule 5, the first evaluation after an edit, or a gap beyond the tier (module
                // doc, "One-surface energy accounting"): the model's energy here is an
                // extrapolation, the start of a new history, or wrong by more than it claims, so
                // the step is reported on the teacher's surface. Coming from the model's surface
                // the transition is booked along the forces that moved the atoms
                // (`transition_jump`) — never through the model's energy here, which may be an
                // extrapolation; from the teacher's, or after an edit, nothing is booked.
                let gap = r.energy - m.energy;
                self.streak_gap = Some(gap);
                stats.refit_jump = self.transition_jump(r.energy, false, &r.forces, &h.pos);
                notes.insert(0, format!("{} — teacher answered, reported on the teacher's surface (teacher − model {gap:+.2e} Ha)", h.reason.text));
                (r.energy, false)
            }
            Some(m) => {
                let gap = r.energy - m.energy;
                self.streak_gap = Some(gap);
                let box_rho = face.filter(|&rho| rho > rule);
                // an audit judges the forces too (M6 Task 11): the worst atom's residual against
                // its bound, the variance rule's max(2e-3, 10 %|F|) capped at 1e-2
                // "not within": a NaN gap or force residual fails the audit (activity Task 3b)
                let force_fail = worst_force_residual(&self.policy, &m.forces, &r.forces, audit_force_floor(&h.prov.ionic)).filter(|(d, b)| !(d <= b));
                let audit_fails = box_rho.is_none() && (!(gap.abs() / n <= self.policy.energy_std_per_atom) || force_fail.is_some());
                if judged_audit {
                    let verdict = if box_rho.is_some() {
                        AuditVerdict::BoxAffected
                    } else if audit_fails {
                        AuditVerdict::Failed
                    } else {
                        AuditVerdict::Passed
                    };
                    self.last_audit = Some(AuditRecord { gap, verdict, face_density: face, unaudited_steps: 0, asynchronous: false, teacher_ms: stats.solve_ms as f64 });
                    self.last_audit_fit = self.fits;
                    // the history's point (M8a Task 1): an in-step audit is never a draft guess; a failed one
                    // records it below, unless a check attributes it (ruling 9b)
                    if !audit_fails {
                        self.record_history_point(surrogate::history::PointKind::Audit, &h.pos, &h.z, m.stds.as_ref().map(|(e, f, fe)| (*e, &f[..], &fe[..])), m.energy, &m.forces, r.energy, &r.forces, box_rho.is_some(), false, unfit.is_some(), h.flags, m.energy_std_epistemic);
                    }
                    // a box-affected audit judged nothing: it does not satisfy the cadence — the
                    // next answered step is audited again — and it is counted for the checklist
                    // (M8a Task 2, the Task 1 physics review's I4); the step keeps the model's
                    // energy: the gap is the Dirichlet confinement, not the model's error
                    // — a calibration tick alone is not rule 8's audit: its cadence and the count
                    // stay as they were (module doc, "Calibration ticks")
                    if !h.tick_only {
                        self.audit_due = box_rho.is_some();
                        self.box_audits_unjudged = if box_rho.is_some() { self.box_audits_unjudged + 1 } else { 0 };
                    }
                    if let Some(rho) = box_rho {
                        notes.push(format!(
                            "surrogate audit not judged: the teacher's answer is box-affected (face density {rho:.1e} > {rule:.1e}) — its gap is the box's confinement as much as the model's error{}",
                            if h.tick_only { " (a calibration tick: rule 8's cadence is unchanged)" } else { "; the next answered step is audited again" }
                        ));
                    }
                }
                if judged_audit && audit_fails {
                    // failed audit: this step is already on the teacher surface, and the jump
                    // of the reported energy at these positions is the gap itself
                    let index = outcome.is_kept().then(|| self.samples().saturating_sub(1));
                    let case = FailedAudit { pos: h.pos.clone(), z: h.z.clone(), energy: r.energy, forces: r.forces.clone(), index, ionic: h.prov.ionic.clone(), rejudges: 0, origin: AuditOrigin::Audit, pair: None };
                    // at a scene with stray atoms the signature is refused and its samples held, and a check
                    // attributes the failure when one can run (`stray` module doc, "Audits of a cleared answer")
                    let why = match force_fail {
                        Some((d, b)) if gap.abs() / n <= self.policy.energy_std_per_atom => format!("worst atom's force off by {d:.1e} > {b:.1e} Ha/Bohr"),
                        _ => format!("teacher − model {gap:+.2e} Ha, {:.1e} > {:.1e} Ha per bonded atom", gap.abs() / n, self.policy.energy_std_per_atom),
                    };
                    let stray = self.stray_audit_failed(scene, raw_energy, &r.forces, &case, gap, why, h.flags);
                    let attributing = matches!(stray, stray::StrayAudit::Attributing(_));
                    if let stray::StrayAudit::Attributing(note) | stray::StrayAudit::CountsNow(note) = stray {
                        notes.push(note);
                    }
                    if !attributing {
                        self.record_history_point(surrogate::history::PointKind::Audit, &h.pos, &h.z, m.stds.as_ref().map(|(e, f, fe)| (*e, &f[..], &fe[..])), m.energy, &m.forces, r.energy, &r.forces, false, false, unfit.is_some(), h.flags, m.energy_std_epistemic);
                        self.audit_failed = Some(gap);
                        self.audit_force = force_fail;
                        self.audit_atoms = n as usize;
                        self.failed_audits.push(case);
                    }
                    // on the refresh right after a refit the refit's own jump (E_new − E_reported
                    // at these positions) is still pending: the reported energy moves by both.
                    // (From a teacher-surface step there is nothing to book: `jump_to_teacher`.)
                    stats.refit_jump = self.jump_to_teacher(gap);
                    notes.insert(0, match force_fail {
                        _ if attributing => format!(
                            "surrogate audit failed at a scene with stray atoms: teacher − model {gap:+.2e} Ha ({:.1e} Ha per bonded atom) — a check of the bonded part alone says whether the model or the composition failed",
                            gap.abs() / n
                        ),
                        Some((d, b)) if gap.abs() / n <= self.policy.energy_std_per_atom => format!(
                            "surrogate audit failed on forces: worst atom off by {d:.1e} > {b:.1e} Ha/Bohr (teacher − model {gap:+.2e} Ha) — untrusted until a refit predicts that geometry without the audited sample"
                        ),
                        _ => format!(
                            "surrogate audit failed: teacher − model {gap:+.2e} Ha ({:.1e} > {:.1e} {}) — untrusted until a refit predicts that geometry without the audited sample",
                            gap.abs() / n, self.policy.energy_std_per_atom, self.per_atom_unit(&h.pos)
                        ),
                    });
                    (r.energy, false)
                } else {
                    stats.teacher_gap = Some(gap);
                    stats.refit_jump = self.jump_back(m.energy, &m.forces, &h.pos);
                    surface_forces = Some(m.forces.clone());
                    stats.model_forces = Some(m.forces);
                    notes.insert(0, format!("{} — teacher answered (teacher − model {gap:+.2e} Ha)", h.reason.text));
                    (m.energy, true)
                }
            }
            None => {
                // a failed background audit untrusted the model since the last evaluation, which
                // was on the model surface: the reported energy moves from the model's to the
                // teacher's at THESE positions — book that (plus a refit's pending jump), exactly
                // as a failed audit in the step does
                let flip = std::mem::take(&mut self.audit_flip);
                let at_model = self
                    .model
                    .as_ref()
                    .filter(|m| flip && self.same_surface(&m.provenance, &h.prov) && m.basis.species_indices(&h.z).is_ok())
                    .and_then(|m| m.predict(&h.pos, &h.z, false).ok());
                // after a lone-atom reference answer (rule 0b, on the model's surface) the switch to
                // the teacher's is booked along the forces, as rule 5's (`transition_jump`)
                let after_reference = self.last.as_ref().is_some_and(|l| l.reference.is_some());
                match at_model {
                    _ if after_reference => stats.refit_jump = self.transition_jump(r.energy, false, &r.forces, &h.pos),
                    Some(p) => stats.refit_jump = Some(self.pending_jump.take().unwrap_or(0.0) + r.energy - p.energy),
                    None => self.pending_jump = None,
                }
                notes.insert(0, format!("{} — teacher answered", h.reason.text));
                (r.energy, false)
            }
        };
        stats.solve_ms = t0.elapsed().as_millis() as u64;
        self.take_offset_jump(&mut stats);
        // this step's own notes (rule 0b's reason: review M5) — in the Activity row's detail too
        notes.append(&mut self.step_notes);
        stats.notes.extend(notes);
        // every note of this solve — the teacher's own and the step's, the admission's numbers
        // included (plan review S12) — but not the pending notes of other events (a background
        // audit judged in this evaluate carries its own as its own event's detail; review M3)
        let detail = stats.notes.join("\n");
        stats.notes.append(&mut self.pending_notes);
        let mut ev = TeachEvent::new(origin_of(h_audit, rule_code), rule_code, outcome);
        ev.atom = atom;
        ev.detail = detail;
        ev.face_density = face;
        ev.box_affected = boxed.is_some();
        ev.gap = gap;
        ev.n_atoms = n_atoms;
        ev.solve_ms = teacher_ms;
        ev.scf_steps = Some(scf);
        ev.ts = Some(ts);
        ev.audit = if h_audit { self.last_audit.as_ref().map(|a| a.verdict) } else { None };
        // the history code (Task 8 review M4: a code-24 row says the model was too confident)
        ev.history_code = Some(history_code);
        self.push_event(ev);
        let surface_forces = surface_forces.unwrap_or_else(|| r.forces.clone());
        self.last = Some(LastEval { pos: h.pos, z: h.z, energy, on_model, provenance: h.prov, flip_back, surface_forces, reference: None });
        self.last_fallback = true;
        Ok(ForceResult { energy, forces: r.forces, stats, display: r.display, orbitals: None })
    }

    /// The jump of the reported energy when a step goes to the teacher's surface at its own
    /// positions with the teacher − model `gap` there: from a model-surface report, the gap plus a
    /// refit's pending jump (that is E_teacher − E_reported at unchanged positions); from a
    /// teacher-surface report, or after an edit (no previous report: the driver restarts the
    /// history), nothing — and a pending jump, computed for a next answer on the model's surface,
    /// is dropped.
    fn jump_to_teacher(&mut self, gap: f64) -> Option<f64> {
        let pending = self.pending_jump.take();
        match self.last.as_ref().map(|l| l.on_model) {
            Some(true) => Some(pending.unwrap_or(0.0) + gap),
            _ => None,
        }
    }

    /// The booking when a step is reported on the model's surface (`energy`, its forces
    /// `model_forces` at `pos`): after a teacher-surface report with the model consulted
    /// (`LastEval::flip_back`), the transition back (`transition_jump`); else a refit's pending
    /// jump (E_new − E_reported at unchanged positions).
    fn jump_back(&mut self, energy: f64, model_forces: &[[f64; 3]], pos: &[[f64; 3]]) -> Option<f64> {
        if self.last.as_ref().is_some_and(|l| l.flip_back) {
            return self.transition_jump(energy, true, model_forces, pos);
        }
        self.pending_jump.take()
    }

    /// The booking of a step whose reported surface differs from the last report's (module doc,
    /// "One-surface energy accounting", transitions): the step from r₀ (last report: energy E₀,
    /// surface forces m̃₀) to r₁ (this one: E₁, m̃₁) is booked as
    ///
    ///   J = E₁ − E₀ + ½ (m̃₀ + m̃₁)·(r₁ − r₀),
    ///
    /// so that with the driver's own force-mismatch term ½(ΔF₀ + ΔF₁)·Δr (ΔF = raw − m̃, zero
    /// where no model forces are reported) the work booked is E₁ − E₀ + ½(F₀ + F₁)·Δr, F the
    /// forces the two half-kicks used: E_conserved then changes by exactly what Verlet's kinetic
    /// energy does beyond the trapezoid, ΔK − ½(F₀ + F₁)·Δr = dt²(|F₁|² − |F₀|²)/(8m) per atom —
    /// the bounded shadow term of every step — and by nothing that depends on either surface
    /// between the two points. Neither the model's energy at a teacher-surface point (an
    /// extrapolation there) nor the teacher's at a model point (never solved) enters. At
    /// unchanged positions (the refresh after a refit) J is the plain change E₁ − E₀, which is
    /// the refit's pending jump plus the flip. Same surface: a refit's pending jump if the report
    /// is the model's (the refit moved it), nothing if it is the teacher's (a pending jump was
    /// computed for a next answer on the model's surface and is dropped); no previous report (an
    /// edit): `None` — the driver restarts the history there.
    fn transition_jump(&mut self, energy: f64, on_model: bool, surface_forces: &[[f64; 3]], pos: &[[f64; 3]]) -> Option<f64> {
        let pending = self.pending_jump.take();
        let l = self.last.as_ref()?;
        if l.on_model == on_model {
            return pending.filter(|_| on_model);
        }
        if l.pos.len() != pos.len() || l.surface_forces.len() != surface_forces.len() {
            return None;
        }
        let mut w = 0.0;
        for ((m0, m1), (r0, r1)) in l.surface_forces.iter().zip(surface_forces).zip(l.pos.iter().zip(pos)) {
            for a in 0..3 {
                w += 0.5 * (m0[a] + m1[a]) * (r1[a] - r0[a]);
            }
        }
        Some(energy - l.energy + w)
    }
}

impl ForceProvider for SurrogateKs {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        // set again only by a teacher answer (`teach`)
        self.last_fallback = false;
        // a note of a step that failed is not carried to the next (review M5)
        self.step_notes.clear();
        // a failed GPU (module doc, "A failed GPU"): no answer at all — neither the teacher's nor the model's
        if let Some(e) = self.gpu_failure_error() {
            return Err(e);
        }
        // the empty scene (`sim::provider`, "Nothing to solve"): no species, so no model, no
        // fallback, no sample, no audit — the known answer, before any rule runs
        if let Some(r) = sim::nothing_to_solve(scene) {
            return r;
        }
        let t0 = Instant::now();
        // R12 under the surrogate (M6 Task 11): the teacher's own inside-cell check, the same
        // `ProviderError::Invalid` and message, before any rule — a model answer is a function of
        // the positions alone and would carry atoms out of the cell (module doc, "The cell").
        // Rule 0's activity row: nothing solved, nothing counted (a drag held against the margin
        // leaves one per refused evaluate, bounded by the queue: plan review N3)
        if let Err(e) = scene.check_inside_cell() {
            let mut ev = TeachEvent::new(TeachOrigin::Fallback, TeachRule::Cell, TeachOutcome::NotSolved);
            ev.detail = e.to_string();
            ev.n_atoms = scene.nuclei.len();
            self.push_event(ev);
            return Err(e);
        }
        // a live evaluation between a pass's sample solves (review m2): neither may extrapolate
        // the teacher's density through the other's geometry
        if let Some(pass) = self.pass.as_mut() {
            self.teacher.discontinuity();
            pass.teacher_used_elsewhere();
        }
        // a box without its isolated-atom references in collection mode: ask the auditor for them
        // (`shared` module doc, "References per box")
        self.request_box_references(scene);
        // a background audit's verdict, if it arrived (module doc, "Audit")
        self.poll_audit();
        // ...whose job may have found the GPU failed (module doc, "A failed GPU")
        if let Some(e) = self.gpu_failure_error() {
            return Err(e);
        }
        // a failed audit of a composition this scene does not hold turns into a local fence, and back
        // when the composition returns (round 1, SHOULD-3; R1-4)
        if !self.failed_audits.is_empty() || !self.audit_fences.is_empty() {
            let zs: Vec<u8> = scene.nuclei.iter().map(|n| n.z).collect();
            self.fence_failed_audits_off_composition(&zs);
        }
        let forced = std::mem::take(&mut self.force_teacher);
        // nothing reported since an edit (or ever): a fallback starts the history on the teacher
        let post_edit = self.last.is_none();
        let prov = self.scene_provenance(scene);
        let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        let z: Vec<u8> = scene.nuclei.iter().map(|n| n.z).collect();
        // a capped segment (module doc, "Masked scenes"): the teacher answers every step and the
        // answer is kept with the scene's mask; the model is never consulted, so nothing is
        // audited, judged or thinned against it
        if scene.train_mask().is_some() {
            let reason = Refusal::new("surrogate: a capped segment teaches its interior atoms — the model does not answer here (its caps and edge atoms are not the source scene's)", TeachRule::Segment, None);
            let near = if self.model.is_some() { self.near_quarantined(&pos, &z) } else { Vec::new() };
            let near_misfit = self.model.is_some() && self.near_misfit(&pos, &z).contains(&true);
            let h = Handoff { prov, pos, z, reason, keep_sample: true, shell_fence: false, thin: false, dedup: true, near, near_misfit, model: None, audit: false, teacher_surface: false, history_rule: history_rule_code(TeachRule::Segment), flags: surrogate::history::PointFlags::default(), tick_only: false };
            return self.teach(scene, h, t0, should_abort);
        }
        // rules 1–4 (draft mode answers past rule 4 only: module doc, "Draft mode")
        let draft = self.policy.never_fallback;
        // rule 0b, lone atoms (module doc; `lone`): in strict mode, no atom with a neighbour within
        // r_cut is answered from the isolated-atom references — the model's own answer there,
        // whatever its trust; not in draft mode, nor when the teacher was asked for (the relax
        // verification). A lone scene it declines goes to the teacher (rules 1–4, else the refusal
        // below, after the gate), with the reason in this step's notes (review M5)
        let lone = self.lone_reference(scene, &prov);
        let lone_answers = matches!(lone, Lone::Answer { .. });
        let mut lone_aside: Option<String> = None;
        if !draft && !forced {
            match lone {
                Lone::Answer { energy, from_model } => return Ok(self.reference_answer(energy, from_model, prov, pos, z, t0)),
                Lone::StandAside(why) => {
                    self.step_notes.push(format!("surrogate: lone atoms, not answered from the atom references — {why}"));
                    lone_aside = Some(why);
                }
                Lone::NotLone => {}
            }
        }
        let mut untrusted = None;
        // the strict policy's refusal here, before draft mode or the relax verification override
        // it: the readiness readout's (module doc, "Readiness")
        let mut strict_gate: Option<Refusal> = None;
        // the history's cadence-fresh bit (module doc, "The history's flags"): this step's count on
        // the never-reset counter, a multiple of k — not `consults`, whose phase a rule-6/7 refusal
        // restarts. Every step reaching the gate counts (the Task 3 review's minor), so the phase
        // does not follow rule 4's untrust either (audit failures, on an answered-step cadence that
        // σ-refusals shift). A draft-mode step is never cadence-fresh for the history: its points
        // are not in P_A (`calibrate`, "The answer sample P_A")
        let cadence_fresh = {
            let k = self.policy.variance_cadence.max(1) as u64;
            let c = self.cadence_steps;
            self.cadence_steps += 1;
            c % k == 0 && !draft
        };
        match self.gate(&z, &prov) {
            Err(Gate { reason, untrusted: true, .. }) if draft => {
                strict_gate = Some(reason.clone());
                untrusted = Some(reason.text);
            }
            Err(Gate { reason, keep_sample, thin, dedup, .. }) => {
                if self.model.is_some() {
                    self.record_verdict(&pos, &z, Some(reason.clone()));
                }
                // a quarantined neighbourhood is never trained on again (module doc)
                let near = if keep_sample && self.model.is_some() { self.near_quarantined(&pos, &z) } else { Vec::new() };
                let near_misfit = keep_sample && self.model.is_some() && self.near_misfit(&pos, &z).contains(&true);
                let history_rule = history_rule_code(reason.rule);
                let h = Handoff { prov, pos, z, reason, keep_sample, shell_fence: false, thin, dedup, near, near_misfit, model: None, audit: false, teacher_surface: false, history_rule, flags: surrogate::history::PointFlags::default(), tick_only: false };
                return self.teach(scene, h, t0, should_abort);
            }
            Ok(()) => {}
        }
        // rules 5–8: consult the model; the variance on the cadence, the envelope every time
        // the variance cadence (module doc): a fresh check after an edit, a refit, a policy change
        // or a rejection, else at least every k-th consultation — an accepting verdict is reused
        // at most k − 1 times, a rejecting one never
        // a neighbour entering or leaving r_cut is a new environment for the verdict (M6 Task 11)
        let mut counts = {
            let rc2 = self.model.as_ref().map_or(0.0, |m| m.basis.r_cut * m.basis.r_cut);
            neighbour_counts(&pos, rc2)
        };
        // a bonded part plus stray atoms whose composition was measured for bonded geometries like this one (`stray`
        // module doc, "Answering inside a region"; spec §4.4): the stray atoms are exempt from 5a–5e (and 6: their
        // force rows are zero), rule 7 is per bonded atom, 5d and the fences judge the bonded part alone; every other
        // rule as before. The answer is the model's own `predict(scene)`, the bonded part's plus Σ E_ref. Only a scene
        // with an atom without neighbours can have stray atoms (the split's test is the counts')
        // (the scene's descriptors, shared with the envelope below: no extra description on this path)
        let scene_desc = if counts.contains(&0) { self.model.as_ref().and_then(|m| describe_values(&m.basis, &pos, &z).ok()) } else { None };
        let cleared = scene_desc.as_deref().and_then(|d| self.stray_cleared_with(&pos, &z, &prov, Some(d)));
        let part = cleared.as_ref().map(|c| &c.0);
        // an exempt stray atom marks another environment for the variance cadence than a refused one: a verdict reached
        // before the clearance is never reused for a cleared step, nor the reverse. Since ruling 9a rules 7 and 5d are
        // per energy-row atom, so the verdict no longer depends on the exemption; the marker only forces a fresh
        // evaluation at the switch (kept: it costs one variance evaluation there)
        for &i in part.map_or(&[][..], |s| &s.stray) {
            counts[i] = usize::MAX;
        }
        // rule 5's envelope (5a–5c, 5e), the class rule (4b) and the fences, before the variance:
        // none reads σ, and the calibration tick is decided on them (module doc, "Calibration
        // ticks"); the reason chain below assembles them in its own order, as before
        let (mut atom_envelope, mut envelope) = {
            let model = self.model.as_ref().expect("gate passed (or draft past rule 4: a model exists)");
            self.envelope_outside_except(model, &pos, &z, part.map_or(&[][..], |s| &s.stray), scene_desc.as_deref())
        };
        // a lone scene rule 0b declined is never the model's in strict mode (review I1): rule 5a's
        // refusal, which covers a scene of two or more lone atoms already, also for one atom (5a's
        // one-atom exemption assumed the reference composes — 0b checks it) and without an envelope
        if let Some(why) = lone_aside.as_deref() {
            if envelope.is_none() {
                envelope = Some(Self::lone_refusal(why));
                if let Some(a) = atom_envelope.first_mut() {
                    *a = f64::INFINITY;
                }
            }
        }
        // rule 4b, the class rule (module doc, "Class-wise trust"): the query's environment class
        // must be trusted by the fit's leave-one-out errors
        let class = {
            let rc2 = self.model.as_ref().map_or(0.0, |m| m.basis.r_cut * m.basis.r_cut);
            self.class_untrusted(&env_keys(&pos, &z, rc2))
        };
        // the fences (each consulted by the chain below only while no reason is set): the
        // quarantine fence, the misfit set, the shell fence and the audit fence (which returns the
        // shell fence's refusal first) — at a cleared stray-atom scene each judges the bonded part
        // alone (`policy::judged_part`; `stray` module doc, "Answering inside a region"), the atoms
        // named in the scene's numbering
        let near = policy::judged_part(&pos, &z, part, |p, q| self.near_quarantined(p, q));
        let near_misfit_atoms = match part {
            Some(sp) => {
                let (p, q) = sp.bonded_part(&pos, &z);
                let near = self.near_misfit(&p, &q);
                sp.scene_values(&near, pos.len(), false)
            }
            None => self.near_misfit(&pos, &z),
        };
        let near_misfit = near_misfit_atoms.contains(&true);
        let in_shell = self.shell_fence_refusal_judged(&pos, &z, part);
        let audit_fence = self.audit_fence_refusal_judged(&pos, &z, part);
        // the history's fenced bit (module doc, "The history's flags"; spec review I4): any fence
        // that applies here, whichever rule names the refusal
        let fenced = !near.is_empty() || near_misfit || in_shell.is_some() || audit_fence.is_some();
        // the calibration tick (module doc, "Calibration ticks"; spec 2026-11-18 §4): decided here,
        // before the variance, on counters, the envelope, the class rule and the fences alone —
        // never on σ, a λ or what rules 6/7 will say. Never in draft mode (its points are not in
        // P_A). After **every** due tick, answered or refused, taken or skipped, the count pauses for
        // a dead time of D in-envelope steps (the controller's re-ruling on the Task 7 re-review's
        // R1): D is σ-blind (`tick_dead_time`), so the tick-eligible steps are the same whatever the
        // ticks' outcomes. Due while the auditor is busy — an answered tick's job outliving D, or
        // other work — skipped and counted, whatever answers the step
        let in_envelope = !draft && envelope.is_none() && class.is_none() && !fenced;
        // what this step books, undone if its teacher solve is cancelled (`untick_cancelled`)
        let tick_before = (self.tick_steps, self.tick_dead, self.ticks);
        if in_envelope {
            self.ticks.population += 1;
            self.time_in_envelope_step();
        }
        let tick = in_envelope && {
            if self.tick_dead > 0 {
                self.tick_dead -= 1;
                false
            } else {
                self.tick_steps += 1;
                self.tick_every > 0 && self.tick_steps % self.tick_every as u64 == 0
            }
        };
        if tick {
            self.ticks.due += 1;
            self.tick_dead = self.tick_dead_time();
        }
        let due_tick = tick;
        let tick = tick && {
            let busy = self.auditor.as_ref().is_some_and(|a| a.busy.is_some());
            if busy {
                self.tick_outcome(TickOutcome::SkippedBusy);
            }
            !busy
        };
        let with_variance = {
            let k = self.policy.variance_cadence.max(1);
            let fresh = self.last_variance.as_ref().is_none_or(|v| !v.accept) || self.consults + 1 >= k || counts != self.verdict_counts;
            self.consults = if fresh { 0 } else { self.consults + 1 };
            if fresh {
                self.variance_checks += 1;
            }
            fresh
        };
        let model = self.model.as_ref().expect("gate passed (or draft past rule 4: a model exists)");
        let pred = match model.predict(&pos, &z, with_variance) {
            Ok(p) => p,
            Err(e) => {
                // a tick with no prediction to compare is lost
                if tick {
                    self.tick_outcome(TickOutcome::Lost);
                }
                return Err(ProviderError::Invalid(format!("surrogate: {e}")));
            }
        };
        // a prediction that is not finite is never an answer, draft mode and the relax
        // verification included (activity Task 3b): the teacher answers on its own surface, the
        // answer is kept, nothing is audited (the gap is not a number), the switch is booked
        // along the forces (`transition_jump`, never through the model's energy)
        if let Some(reason) = non_finite_answer(&pred) {
            self.record_verdict(&pos, &z, Some(strict_gate.unwrap_or_else(|| reason.clone())));
            let h = Handoff {
                prov,
                pos,
                z,
                reason,
                keep_sample: true,
                shell_fence: false,
                thin: false,
                dedup: false,
                near,
                near_misfit,
                model: Some(ModelAnswer { energy: pred.energy, forces: pred.forces, uncertainty: None, stds: None, energy_std_epistemic: None }),
                audit: false,
                teacher_surface: true,
                history_rule: history_rule_code(TeachRule::NonFiniteAnswer),
                flags: surrogate::history::PointFlags::default(),
                tick_only: false,
            };
            let r = self.teach(scene, h, t0, should_abort);
            // no point is recorded for a non-finite answer: a tick here has nothing to compare and is
            // lost — unless the solve was cancelled: the step did not happen (`untick_cancelled`)
            if !self.untick_cancelled(&r, tick_before) && tick {
                self.tick_outcome(TickOutcome::Lost);
            }
            return r;
        }
        // rule 5d's novelty at this geometry: `Some((ν, atom))` when ν > 1
        // (the bonded part's alone at a cleared stray-atom scene: `novelty_judged`, the atom in the scene's numbering)
        let novelty = |s: &Self| s.novelty_judged(&pos, &z, part).filter(|(nu, _)| *nu > 1.0);
        // the history's novel bit (module doc, "The history's flags"): rule 5d's novelty wherever it
        // is evaluated — with a fresh variance, and read-only on every cadence-fresh step that
        // reuses a verdict (Task 3 review MUST), so a cadence-fresh point's bit does not follow the
        // σ-coupled variance phase
        let mut novel_bit = false;
        // rule 6 on the epistemic force std (σ_f excluded, as rule 7 excludes the energy floor)
        let variance = match (pred.energy_std, pred.force_std_epistemic.as_ref()) {
            (Some(e), Some(f)) => {
                let mut v = self.variance_check(model, &pred, e, f, &pos, &z);
                // rule 5d with the variance (the same cadence; a rejection is never reused)
                if let Some((nu, at)) = novelty(self) {
                    v.accept = false;
                    v.novel = true;
                    v.by_tail = false;
                    v.reason = Some(novelty_refusal(nu, at));
                    novel_bit = true;
                }
                self.verdict_counts = counts.clone();
                self.last_variance = Some(v.clone());
                v
            }
            _ => {
                // read-only: the reused verdict and every rule's decision stay as they were — on a
                // tick too (the tick forces the novelty and the stds for its point, never a fresh
                // verdict: no rule-6/7 decision moves on a tick step)
                novel_bit = (cadence_fresh || tick) && novelty(self).is_some();
                self.last_variance.clone().expect("a variance verdict exists when none is evaluated")
            }
        };
        let uncertainty = Some(variance.uncertainty);
        let atom_uncertainty = Some(variance.atoms.clone());
        // rule 5 includes 5d (an unspanned direction): outside what the model has seen
        let by_envelope = envelope.is_some() || (variance.novel && !variance.accept);
        // the class rule (4b, above): when rule 5 fires as well its note (which atom, which
        // feature) is the more specific one and is shown; the variance's (6–7) yields to the class's
        let class_short = class.as_ref().is_some_and(|c| c.1);
        let by_class = class.is_some();
        let novel = (variance.novel && !variance.accept).then(|| variance.reason.clone()).flatten();
        let mut reason = envelope
            .or(novel)
            .or(class.map(|c| c.0))
            .or_else(|| if variance.accept { None } else { variance.reason.clone() });
        // a quarantined geometry is outside what the model may answer (ruling S15), whichever
        // rule fired first; its answer is kept only if it agrees with the model (teach) — `near`,
        // above. Refused by the fence alone (final review I1): its answer is thinned and copy-checked as an
        // envelope (rule 5) fallback's is — which reaches `admit`'s checks only where forgotten samples
        // alone fence (a held one is released or superseded by an agreeing answer, and a
        // disagreeing one is never kept but by a fence vote)
        let by_fence = reason.is_none() && !near.is_empty();
        if by_fence {
            reason = Some(Refusal::new(
                self.fence_refusal_text(&near),
                TeachRule::NearQuarantine,
                None,
            ));
        }
        // next to a training sample the model cannot reproduce (spec 2026-11-13 §4.3; `trust`
        // module doc, "M, the misfit set"): refused when **any** atom is near — broader than the
        // all-atom quarantine test above, deliberately — and the teacher's answer is kept only if
        // it agrees with the model (`admit`), whichever rule fired first (`near_misfit_atoms`, above)
        if reason.is_none() {
            if let Some(i) = near_misfit_atoms.iter().position(|&b| b) {
                reason = Some(Refusal::new(
                    format!(
                        "surrogate: atom {i} is within {:.2} training spreads of a training sample the model cannot reproduce",
                        QUARANTINE_FRACTION * self.policy.max_descriptor_distance
                    ),
                    TeachRule::NearMisfit,
                    Some(i),
                ));
            }
        }
        // the shell fence (Ruling shell → fence; final review I1): whichever rule names the
        // refusal, a geometry inside a fenced shell band is answered by the teacher and the answer
        // is **held out of the fit** — a band row is the leverage/structural outlier R3-1 stopped
        // fitting (f_c → 0 there), and a flood of them would untrust a class everywhere or let the
        // fit absorb them; then an audit fence (round 1, SHOULD-3), whose fallbacks are never
        // copies (R1-2) — both evaluated above, on every consulted step
        let by_shell_fence = in_shell.is_some();
        if reason.is_none() {
            reason = in_shell;
        }
        let mut by_audit_fence = false;
        if reason.is_none() {
            reason = audit_fence;
            by_audit_fence = reason.is_some();
        }
        // the history's flags: `fenced` (above) whichever rule named the refusal; `TICK` on the
        // point a tick leaves (`tick_flags`, below)
        let flags = surrogate::history::PointFlags(
            if cadence_fresh { surrogate::history::PointFlags::CADENCE_FRESH } else { 0 }
                | if fenced { surrogate::history::PointFlags::FENCED } else { 0 }
                | if novel_bit { surrogate::history::PointFlags::NOVEL } else { 0 },
        );
        let tick_flags = if tick { surrogate::history::PointFlags(flags.0 | surrogate::history::PointFlags::TICK) } else { flags };
        self.record_verdict(&pos, &z, strict_gate.clone().or_else(|| reason.clone()));
        // never at lone atoms the references answer (rule 0b — reached here in draft mode only): the
        // gap there is the isolated atoms' grid/box artefacts, not a model error, and the answer is
        // never kept — an audit could only untrust a good model on it. A lone scene 0b declines
        // (another spin or charge state) stays auditable (review I1)
        let due = (reason.is_none() || draft) && !lone_answers && self.policy.audit_every > 0 && {
            self.answered += 1;
            self.answered % self.policy.audit_every == 0 || self.audit_due || self.audit_burst > 0
        };
        if due {
            self.audit_burst = self.audit_burst.saturating_sub(1);
        }
        // asynchronous when an auditor is attached (the model answers this step), else in the step
        // — never in draft mode, where every step is the model's (module doc, "Draft mode")
        // a draft-mode audit at a geometry the strict policy refuses is a guess's sample (fix (b))
        // — only when the refusal there is a fact about the geometry (round 1, MUST-1): refused only
        // by rule 4/4b/6/7, strict mode may answer there once trust returns, so the audit is ordinary
        self.dispatch_draft_guess = draft && reason.as_ref().is_some_and(|r| geometry_specific(r.rule));
        // a tick at a step the model answers is an audit (module doc, "Calibration ticks"): on the
        // auditor (free: a busy one skipped the tick above), else in the step; judged by rule 8 as
        // well, and rule 8's own cadence (`due`, `answered`, `audit_due`) is untouched by it. A tick
        // at a fallback (a rule refused, or the relax verification) needs no solve: its point is
        // the fallback's, with `TICK`
        let tick_audit = tick && reason.is_none() && !forced;
        // the answered share among due ticks against the population's (spec §4's σ-coupling
        // measurement, the re-review's R1 diagnostic): a step's outcome, known only now — every
        // in-envelope step's, in the dead time and with a job in flight included
        if in_envelope && reason.is_none() && !forced {
            self.ticks.population_answered += 1;
            if due_tick {
                self.ticks.answered_due += 1;
                self.ticks.answered_skipped_busy += u64::from(!tick);
            }
        }
        let tick_only = tick_audit && !due;
        self.dispatch_flags = if tick_audit { tick_flags } else { flags };
        self.dispatch_tick_only = tick_only;
        let audit = (due || tick_audit) && !self.dispatch_audit(scene, &prov) && !draft;
        self.dispatch_draft_guess = false;
        self.dispatch_flags = surrogate::history::PointFlags::default();
        self.dispatch_tick_only = false;
        // a tick served by this step's teacher solve: at a fallback, or in the step's audit
        let tick_here = tick && (!tick_audit || audit);
        // draft mode: what the model guesses beyond, and it answers anyway
        let extrapolation = (draft && (untrusted.is_some() || reason.is_some())).then(|| Extrapolation {
            atoms_outside: atom_envelope.iter().filter(|&&d| d > 1.0).count(),
            untrusted: untrusted.clone(),
            reason: reason.as_ref().map(|r| r.text.clone()),
            atom_envelope: atom_envelope.clone(),
        });
        // the history's rule code of this step's fallback point, from the rule that fired (before
        // the relax verification overrides the reason — the Task 1 re-review's Minor 1): a class
        // refusal the variance verdict would have refused as well is coded apart, since the model
        // would not have answered it had the class been trusted (M8a Task 2, `calibrate`); a rule-6/7
        // refusal that passes at λ_sum — only λ_T or λ_E refused — is coded apart as well (trust
        // tightening, spec 2026-11-18 §4: λ selected it, so it is never in the answer sample P_A)
        let history_rule = match reason.as_ref() {
            Some(r) if matches!(r.rule, TeachRule::Class(_)) && !variance.accept => HISTORY_RULE_CLASS_AND_VARIANCE,
            Some(r) if matches!(r.rule, TeachRule::ForceUncertainty | TeachRule::EnergyUncertainty) && variance.by_tail => HISTORY_RULE_TAIL_INFLATION,
            Some(r) => history_rule_code(r.rule),
            None => history_rule_code(TeachRule::RelaxVerification),
        };
        if draft {
            reason = None;
        }
        // the relax verification (`force_teacher_next`): the teacher answers whatever the rules
        // and draft mode say, and the answer is an essential sample (never thinned)
        if forced {
            reason = Some(Refusal::new("surrogate: relax verification — the teacher checks the relaxed geometry", TeachRule::RelaxVerification, None));
        }
        if reason.is_some() || audit {
            let reason = reason.unwrap_or_else(|| {
                if tick_only && audit {
                    Refusal::new(format!("surrogate calibration tick (every {} in-envelope steps; no background teacher, so in the step)", self.tick_every), TeachRule::Audit, None)
                } else {
                    Refusal::new(format!("surrogate audit (every {} answered steps)", self.policy.audit_every), TeachRule::Audit, None)
                }
            });
            // the history's point (M8a Task 1, `surrogate::history`) of this in-step audit or
            // consulted-model fallback: the model's stds at this geometry — the cadence may have
            // reused an earlier verdict, so when the prediction carries none they are evaluated
            // here (one variance evaluation, O(N·Q′²), against the teacher solve this step makes);
            // `None` without a variance factor (counted as such — a `predict` error here is
            // counted the same way, though it cannot happen: the same model just predicted at
            // this geometry)
            let stds = match (pred.energy_std, pred.force_std.as_ref(), pred.force_std_epistemic.as_ref()) {
                (Some(e), Some(f), Some(fe)) => Some((e, f.clone(), fe.clone())),
                _ => self.model.as_ref().and_then(|m| m.predict(&pos, &z, true).ok()).and_then(|p| Some((p.energy_std?, p.force_std?, p.force_std_epistemic?))),
            };
            // under the model that predicted them (the same one: nothing refits inside `evaluate`)
            let energy_std_epistemic = stds.as_ref().zip(self.model.as_ref()).map(|((e, _, _), m)| epistemic_energy_std(m, surrogate::energy_row_atoms(&m.basis, &pos), *e));
            let h = Handoff {
                prov,
                pos,
                z,
                reason,
                // a geometry inside a fenced shell band is never fitted (I1, the coordinator's
                // ruling): the relax verification still has the teacher's answer, an audit there
                // is not a judgement of the model (`teach` neither fits nor judges it)
                keep_sample: !by_shell_fence,
                shell_fence: by_shell_fence,
                // the variance rules' fallbacks are the answers active learning asks for, and so
                // are a class's while it is short of samples (rule 4b); never a copy where copies
                // could satisfy the rule (4b, 5: round 1 of the 2026-10-18 review, MUST-2); next
                // to a misfit (whichever rule named the refusal) an agreeing answer is kept but M
                // does not lift on it as a quarantine fence does, so a parked scene would keep one
                // per step: thinned and never a copy, as rule 4's answers (Task 6 review, ruling 10);
                // and refused by the quarantine fence alone (forgotten samples' permanent fence,
                // sample-recovery final review I1): both thin, a geometry near either is thinned
                thin: (by_envelope || by_class || by_fence || near_misfit) && !class_short && !forced,
                dedup: (by_envelope || by_class || by_audit_fence || by_fence || near_misfit) && !forced,
                near,
                near_misfit,
                model: Some(ModelAnswer { energy: pred.energy, forces: pred.forces, uncertainty, stds, energy_std_epistemic }),
                audit,
                // rules 4b and 5 (the model's energy is an unvalidated guess or an extrapolation)
                // and the first evaluation after an edit report the teacher's surface; audits
                // and the relax verification keep theirs (module doc, "One-surface energy
                // accounting")
                teacher_surface: !audit && !forced && (by_class || by_envelope || post_edit),
                history_rule,
                flags: if tick_here { tick_flags } else { flags },
                tick_only: audit && tick_only,
            };
            let before = self.audit_history.recorded();
            let r = self.teach(scene, h, t0, should_abort);
            // a cancelled solve: the step did not happen, its count is rolled back and its retry is
            // the tick (the Task 7 review's M2: fallback ticks are the σ-refused ones, so losing them
            // to the user's cancels would be σ-coupled); a failed one loses the tick
            if !self.untick_cancelled(&r, tick_before) && tick_here {
                let recorded = self.audit_history.recorded() > before;
                self.tick_outcome(match (r.is_ok(), recorded) {
                    (false, _) => TickOutcome::Lost,
                    (true, true) => TickOutcome::Recorded,
                    (true, false) => TickOutcome::Unrecorded,
                });
                if r.is_ok() && tick_audit {
                    self.ticks.in_step += 1;
                }
            }
            return r;
        }
        let mut stats = EvalStats::default();
        stats.source = Source::Surrogate;
        if let Some((tag, summaries)) = &self.last_pseudised {
            if *tag == prov.ionic && !crate::pseudo::is_all_electron_tag(&prov.ionic) {
                stats.pseudised = summaries.clone();
            }
        }
        stats.uncertainty = uncertainty;
        stats.atom_uncertainty = atom_uncertainty;
        stats.extrapolation = extrapolation;
        stats.refit_jump = self.jump_back(pred.energy, &pred.forces, &pos);
        self.take_offset_jump(&mut stats);
        stats.energy.total = pred.energy;
        stats.solve_ms = t0.elapsed().as_millis() as u64;
        if let Some((_, residual, margin, passes)) = cleared {
            // the answer identity E = E_model(bonded) + Σ E_ref needs offset_s = 0 (mean 0, no intercept): `stray_cleared`
            // answers no stray scene with a nonzero mean, in every build (final review minor 8)
            // rule 7 of the doc's honesty (review M6): what answered, and on what measurement — the passing checks booked
            stats.notes.push(format!(
                "stray atoms: model for the bonded part + free atoms (composition measured, {passes} checks; worst residual {:.2} mHa, frontier margin {:.0} mHa)",
                residual * 1e3,
                margin * 1e3
            ));
        }
        stats.notes.append(&mut self.step_notes);
        stats.notes.append(&mut self.pending_notes);
        self.streak = 0;
        self.streak_box = 0;
        self.streak_box_face = 0.0;
        self.class_streak = 0;
        self.last = Some(LastEval { pos, z, energy: pred.energy, on_model: true, provenance: prov, flip_back: false, surface_forces: pred.forces.clone(), reference: None });
        Ok(ForceResult { energy: pred.energy, forces: pred.forces, stats, display: None, orbitals: None })
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn reset(&mut self) {
        self.teacher.reset();
        self.last_fallback = false;
        self.last = None;
        self.offset_jump = 0.0;
        self.last_variance = None; // the next consultation checks the variance afresh
        // an edit: a composition check in flight is of the old scene (`stray`)
        self.scene_gen += 1;
    }

    fn discontinuity(&mut self) {
        self.scene_gen += 1;
        self.teacher.discontinuity();
        self.last_fallback = false;
        self.last = None;
        self.offset_jump = 0.0;
        self.last_variance = None;
    }

    fn surface_epoch(&self) -> u64 {
        self.epoch
    }

    fn as_trainable(&mut self) -> Option<&mut dyn Trainable> {
        Some(self)
    }

    fn side_jobs(&self) -> u64 {
        self.side_jobs
    }

    /// The density job's payload (audit or refresh) first, then the Harris one (module doc, "Side
    /// densities"): the driver takes until `None`.
    fn take_side_display(&mut self) -> Option<SideDisplay> {
        self.poll_side_jobs();
        self.side_display.take().or_else(|| self.side_harris.take())
    }

    /// The teacher's side solver (a fresh instance): the references are its surface's, never the
    /// model's and never the stepping teacher's warm state. Used by the driver only when there is
    /// no background auditor (the CPU teacher); with one, the references go to its job slot
    /// (`request_atom_reference`, module doc "Side densities").
    fn side_solver(&self) -> Option<Box<dyn ForceProvider>> {
        self.teacher.side_solver()
    }

    /// The session's GPU failure (module doc, "A failed GPU"): the one latched from any teacher's answer (the
    /// auditor's included), else the stepping teacher's device's.
    fn device_failure(&self) -> Option<String> {
        self.gpu_failed.clone().or_else(|| self.teacher.device_failure())
    }

    /// The teacher's ionic surface (M7): the references and the model's provenance are its.
    fn ionic_tag(&self, scene: &Scene) -> String {
        self.teacher.ionic_tag(scene)
    }

    /// The teacher's: its densities (stepping and background) are the ones partitioned.
    fn valence_charges(&self, scene: &Scene) -> Option<Vec<f64>> {
        self.teacher.valence_charges(scene)
    }

    /// Kept, sent with every background job (the auditor's teacher is the one producer of the shown density in
    /// estimated mode) and applied to the stepping teacher.
    fn set_display_partial_core(&mut self, on: bool) {
        self.display_core = on;
        self.teacher.set_display_partial_core(on);
    }

    /// The stepping teacher's density after a fallback, else `None` (module doc, "Density samples").
    fn sample_density(&mut self, points: &[[f64; 3]]) -> Option<Vec<f64>> {
        if self.last_fallback { self.teacher.sample_density(points) } else { None }
    }

    /// Kept for the background auditor's teacher, the one producer (module doc, "Side
    /// densities"); sent with its next job. The stepping teacher never sees it.
    fn set_orbital_view(&mut self, request: OrbitalViewRequest) {
        self.orbital_view = request;
    }

    /// Always `None`: the producer is the background teacher, whose state is the last side
    /// solve's geometry, not this one's; ask it for a display refresh (`request_display`).
    fn orbital_display_now(&mut self, _should_abort: &mut dyn FnMut() -> bool) -> Option<OrbitalDisplay> {
        None
    }

    /// Kept and sent with every background job: the auditor's teacher, the one producer, polls it
    /// in its display work (the unoccupied eigensolve of a paused refresh), so a forced refresh
    /// yields its job slot to the user's next command (Task 4 re-review N1). The stepping teacher
    /// never shows orbitals, so it needs none.
    fn set_display_interrupt(&mut self, interrupt: sim::DisplayInterrupt) {
        self.display_interrupt = Some(interrupt);
    }

    /// Kept and sent with every background job (the auditor's teacher, the one orbital producer,
    /// weights them at its job's positions); forwarded to the stepping teacher too, which shows no
    /// orbitals (module doc, "Orbitals").
    fn set_share_tables(&mut self, tables: Option<Arc<ShareTables>>) {
        self.teacher.set_share_tables(tables.clone());
        self.share_tables = tables;
    }

    /// Queued on the background auditor (module doc, "Side densities"): `Queued`, the payload
    /// arriving as a `SideKind::Harris` side display; `None` without an auditor, for an empty
    /// scene, or while a sampling pass runs. Never a fallback, an audit, a sample or a teach event.
    fn harris_display(&mut self, scene: &Scene, atoms: &Arc<sim::HarrisAtoms>, _should_abort: &mut dyn FnMut() -> bool) -> Option<sim::HarrisAnswer> {
        self.queue_harris(scene, atoms).map(|job| sim::HarrisAnswer::Queued { job })
    }
}

/// A pass's end, waiting for its fit (`finish_pass` with `fits_in_background`; quiet refits round
/// 2): everything `complete_pass_tail` does once the fit is in.
pub(super) struct PassTail {
    kind: &'static str,
    how: &'static str,
    added: usize,
    progress: SamplingProgress,
    shells: Vec<crate::sampling::ShellCheck>,
    unrun: Vec<((u8, u8), f64)>,
    teach: Option<(Vec<usize>, Vec<crate::sampling::Holdout>)>,
    summary: Option<String>,
    /// Background refits that carried it and were discarded (`background` module doc, "A pass's
    /// fit"): one more is allowed, then the tail completes as [`PassFit::Dropped`].
    pub(super) tries: u32,
}

/// Why a waiting pass's tail completes as not landed when a new pass starts (`drop_pass_tail`).
const NEW_PASS_FIRST: &str = "a new pass started first — its fit at its end holds these samples too";

/// How a pass's fit came out, for [`SurrogateKs::complete_pass_tail`].
pub(super) enum PassFit {
    NoSamples,
    BelowFloor { fittable: usize, floor: usize, q_prime: usize },
    Fitted(Result<FitReport, FitError>),
    /// It never landed: the reason (discarded twice, a later pass, the samples forgotten).
    Dropped(String),
}

// ---- the Trainable face (M6 Task 1): what the driver and the worker reach through
// `ForceProvider::as_trainable` ----

impl SurrogateKs {
    /// Every pass starts here (a new pass, a resumed one, a reference retry, a teach-around pass):
    /// the pass owns the display's turn, so the queued Harris job goes — the caller asks again
    /// after it (module doc, "Side densities").
    pub(super) fn begin_pass(&mut self, sampler: Sampler) {
        // every entry dropped a waiting tail before it touched any state (`drop_pass_tail`); a
        // future entry that forgets to still cannot leave one behind its new pass
        debug_assert!(self.pass_tail.is_none(), "a pass entry left the previous pass's tail waiting");
        self.drop_pass_tail(NEW_PASS_FIRST);
        self.pass = Some(sampler);
        self.harris_queued = None;
    }

    /// A pass's tail still waiting for its fit, completed as not landed (`PassFit::Dropped`) —
    /// called by every pass entry (`start_pass`, `resume_pass`, the reference retry, `start_teach`)
    /// **before** it touches any state (quiet refits round 3, review I1): the completion reads the
    /// teaching state (`teach_rounds`, `teach_classes`, `teach_start`) and `pass_start` of *its*
    /// pass, and a regular pass's restarts the round count — read after the new pass set them, an
    /// old teach round's judgement measured from the new start and uncounted the new round's
    /// classes, and an old regular pass wiped the new round's count. The new pass's fit at its end
    /// holds the tail's samples too. A running background refit no longer carries it.
    pub(super) fn drop_pass_tail(&mut self, why: &str) {
        let Some(tail) = self.pass_tail.take() else { return };
        self.pass_fit_due = false;
        if let Some(b) = self.background.as_mut() {
            b.pass_fit = false;
        }
        self.complete_pass_tail(tail, PassFit::Dropped(why.into()));
    }

    /// A pass's tail whose samples go with the store — a forget, a loaded model or samples file,
    /// an attach (quiet refits round 3, review I2 and Minor 6): nothing of it is completed (its
    /// outcome, shell checks and a teach-around round's judgement would read a model that does not
    /// hold its samples), one note says so, and a teach-around round's own state goes with it
    /// (not counted, as a cancelled round). A running background refit no longer carries it, so
    /// its discard makes nothing due again.
    pub(super) fn discard_pass_tail(&mut self, why: &str) {
        self.pass_fit_due = false;
        if let Some(b) = self.background.as_mut() {
            b.pass_fit = false;
        }
        let Some(tail) = self.pass_tail.take() else { return };
        if tail.teach.is_some() {
            self.uncount_teach_round();
            self.teach_start = None;
        }
        self.user_notes.push(format!(
            "{} {}: its refit will not land ({why}) — the pass's outcome, its shell checks{} go with its {} samples",
            tail.kind,
            tail.how,
            if tail.teach.is_some() { " and the teach-around round's judgement" } else { "" },
            tail.added
        ));
    }

    /// After a pass (complete, cancelled or ended by a failed reference): the reference cache
    /// back, the teacher told its state is a sample geometry's, the model refitted over
    /// everything it holds — only when the samples a fit would use reach Q′ (below it the model
    /// cannot earn trust, and the fit with its trust recovery took 45–50 s for 3 H₂O samples on
    /// the worker thread inside a Cancel: the samples are kept and the refit waits; M6 Task 3
    /// fix round 2) — and the outcome noted (what `Dynamics` did before M6).
    fn finish_pass(&mut self, mut sampler: Sampler) {
        if sampler.is_migration() {
            return self.finish_migration(sampler);
        }
        let progress = sampler.progress();
        // a teach-around pass is judged after its refit (`teach`)
        let teach = sampler.is_targeted().then(|| (sampler.targets().to_vec(), sampler.take_holdouts()));
        let shells = sampler.take_shell_checks();
        let unrun = sampler.pending_shell_checks();
        self.ref_cache = sampler.take_references();
        self.user_notes.extend(sampler.take_notes());
        if let Some(f) = sampler.take_failed_reference() {
            self.failed_refs.insert(f.z, f);
        }
        self.last_pass = Some(progress.clone());
        let added = sampler.samples_added();
        let how = if progress.cancelled { "cancelled" } else { "finished" };
        let kind = if teach.is_some() {
            "teach-around pass"
        } else if sampler.is_references_only() {
            "reference retry"
        } else {
            "sampling pass"
        };
        self.teacher.discontinuity();
        let summary = self.take_collection_summary(sampler.collection_skipped());
        // an earlier pass's tail still waiting for its fit (unreachable: this pass's entry dropped
        // it, `drop_pass_tail`): this pass's fit holds its samples too
        self.drop_pass_tail("a later pass ended first — its fit holds these samples too");
        let tail = PassTail { kind, how, added, progress, shells, unrun, teach, summary, tries: 0 };
        let (fittable, floor, q_prime) = self.fittable_and_floor();
        if self.samples() == 0 {
            self.complete_pass_tail(tail, PassFit::NoSamples);
        } else if fittable < floor {
            // below the floor the model cannot earn trust (the trust rule needs n ≥ |∪ S_k|,
            // `sample_floor`), and the fit with its trust recovery costs seconds on the worker
            // thread (Task 3 fix round 2): keep the samples, skip the refit until there are
            // enough. The floor is the columns the samples' atoms touch, not the basis's Q′ (M8a
            // user-fix 3, the controller's ruling on concern 2): a new species in a large
            // collection raises Q′ by its own block, and N₂ trained into a 1200-sample water
            // store would otherwise never refit
            self.complete_pass_tail(tail, PassFit::BelowFloor { fittable, floor, q_prime });
        } else if self.fits_in_background {
            // the caller fits it in the background (`Trainable::set_fits_in_background`; quiet
            // refits round 2: a collection's pass-end fit froze the app for 30–160 s): the fit is
            // due (`first_fit_ready`), and the tail — the outcome, the shell checks, a teach-around
            // round's judgement — waits for its install (`complete_pass_tail`)
            self.user_notes.push(format!(
                "{kind} {how}: {added} samples added ({} skipped, {} failed); the refit runs in the background — its outcome follows when it lands",
                tail.progress.skipped, tail.progress.failed
            ));
            self.pass_tail = Some(tail);
            self.pass_fit_due = true;
            self.arm_refit();
        } else {
            let r = SurrogateKs::refit(self);
            self.complete_pass_tail(tail, PassFit::Fitted(r));
        }
    }

    /// What `finish_pass` does once its fit is in (`PassFit`; in place, or at the install of the
    /// background refit that carried it — `background` module doc, "A pass's fit"): the outcome
    /// noted, the pass's coverage, its collection summary, its shell checks judged against this
    /// fit (or left unjudged, said), the shell checks it did not run, and a teach-around round
    /// judged (a regular pass restarts the round count instead, SHOULD-T6).
    pub(super) fn complete_pass_tail(&mut self, tail: PassTail, fit: PassFit) {
        let PassTail { kind, how, added, progress, shells, unrun, teach, summary, .. } = tail;
        // the shell checks are judged against this pass's own refit only (final review M2): with
        // no refit the model standing is none or an earlier fit, whose verdict on the shell is
        // not this pass's measurement
        let mut unjudged: Option<String> = None;
        // a check judged against a fit that is not trusted, or holds none of this pass's samples, says so (physics review
        // of the M7 leftovers, Minor 2): its fence is kept (the safe side) but may be the fit's error, not the cutoff's
        let mut caveat: Option<&'static str> = None;
        // the fit never landed: a teach-around round judges nothing (`drop_teach`)
        let mut dropped: Option<String> = None;
        let msg = match fit {
            PassFit::NoSamples => {
                unjudged = Some("the pass fitted no model".into());
                format!("{kind} {how}: no samples to fit")
            }
            PassFit::BelowFloor { fittable, floor, q_prime } => {
                unjudged = Some(format!("the pass's refit did not run: {fittable} samples, below the {floor} model columns their atoms touch"));
                format!("{kind} {how}: {fittable} samples kept — not enough to fit ({floor} model columns their atoms touch; Q′ {q_prime}); sample more to train")
            }
            PassFit::Fitted(Ok(r)) => {
                let coverage = self.compute_pass_coverage();
                let trust = match (self.untrusted_reason(), coverage.as_ref()) {
                    (Some(why), _) => {
                        caveat = Some("judged against a fit that is not trusted");
                        format!("untrusted ({why})")
                    }
                    // fix (e): never "trusted" for a scene whose own pass is not in the fit
                    (None, Some(c)) if c.fitted == 0 && c.total > 0 => {
                        caveat = Some("judged against a fit that holds none of this pass's samples");
                        "trusted elsewhere, but NOT for this scene: none of its pass samples is in the fit".to_string()
                    }
                    (None, _) => "trusted".to_string(),
                };
                format!(
                    "{kind} {how}: {added} samples added ({} skipped, {} failed); refit on {} samples, rmse {:.1e} Ha/atom, {:.1e} Ha/Bohr — {trust}{}",
                    progress.skipped,
                    progress.failed,
                    r.n_samples,
                    r.rmse_energy_per_atom,
                    r.rmse_force,
                    coverage.as_ref().and_then(PassCoverage::words).map_or(String::new(), |w| format!(". {w}"))
                )
            }
            PassFit::Fitted(Err(e)) => {
                unjudged = Some(format!("the pass's refit failed ({e})"));
                if self.model.is_none() {
                    self.last_fit_error = Some(e.to_string());
                }
                format!("{kind} {how}: {added} samples added; the refit failed ({e})")
            }
            PassFit::Dropped(why) => {
                unjudged = Some(format!("the pass's refit did not land ({why})"));
                let msg = format!("{kind} {how}: the refit of its {added} samples did not land ({why}) — Refit now fits them");
                dropped = Some(why);
                msg
            }
        };
        self.pass_coverage = self.compute_pass_coverage();
        self.user_notes.push(msg);
        if let Some(summary) = summary {
            self.user_notes.push(summary);
        }
        match unjudged {
            None => self.judge_shell_checks_with(shells, caveat),
            Some(why) => self.leave_shell_checks_unjudged(&shells, &why),
        }
        for ((za, zb), r) in unrun {
            self.user_notes.push(format!(
                "sampling: the pass ended before its shell check of {}–{} at r = {r:.2} Bohr — the cutoff shell of this pair stays unmeasured",
                units::elements::symbol(za),
                units::elements::symbol(zb)
            ));
        }
        match (teach, dropped) {
            // judged against no fit of its own (quiet refits round 3, review I1): kept, unjudged
            (Some((targets, holdouts)), Some(why)) => self.drop_teach(&targets, holdouts, &progress, &why),
            (Some((targets, holdouts)), None) => self.finish_teach(&targets, holdouts, &progress),
            // a regular pass brings new data: the teach-around round count restarts (SHOULD-T6)
            (None, _) => self.reset_teach_rounds(),
        }
    }

    /// What the last pass's samples (the pass-tagged ones stored since `pass_start`) are in the fit
    /// that stands (fix (e)); `None` before any pass.
    fn compute_pass_coverage(&self) -> Option<PassCoverage> {
        let start = self.pass_start?;
        let store = self.store.as_ref()?;
        let mut c = PassCoverage::default();
        for (i, s) in store.samples.iter().enumerate().skip(start) {
            if !matches!(s.tag, SampleTag::Perturbation | SampleTag::Scan | SampleTag::HotBurst | SampleTag::Targeted) {
                continue;
            }
            if c.total == 0 {
                c.species = s.z.clone();
                c.species.sort_unstable();
            }
            c.total += 1;
            match self.quarantine.get(i).copied().unwrap_or(Q_NONE) {
                Q_BOX => c.box_excluded += 1,
                Q_HELD | Q_SUPERSEDED => c.quarantined += 1,
                Q_FORGOTTEN | Q_FORGOTTEN_UNFENCED => c.forgotten += 1,
                Q_NONE if i < self.accumulated => c.fitted += 1,
                _ => c.other += 1,
            }
            if self.is_box(i) {
                c.box_affected += 1;
            }
        }
        (c.total > 0).then_some(c)
    }

    /// The stored samples a fit would use (not quarantined, superseded, box-excluded or held
    /// out), the floor they must reach — the independent columns their atoms touch in the basis
    /// the fit would build, |∪ S_k| ([`fit_floor`](Self::fit_floor); the trust rule's
    /// `sample_floor`) — and that basis's Q′ (a readout). (0, 0, 0) without samples.
    ///
    /// No count is needed by the fit itself: a column no fitted atom touches has an all-zero Gram
    /// row and column, and the ridge prior holds its coefficient at 0 (`surrogate::fit`, "Ridge
    /// floor"; the solve fails only for an all-zero Gram at λ = 0, `FitError::Singular`).
    fn fittable_and_floor(&self) -> (usize, usize, usize) {
        let Some(store) = self.store.as_ref() else { return (0, 0, 0) };
        let n = (0..store.samples.len()).filter(|&i| !self.excluded(i)).count();
        let mut species: Vec<u8> = store.samples.iter().flat_map(|s| s.z.iter().copied()).collect();
        species.sort_unstable();
        species.dedup();
        let basis = self.fit_basis(&species);
        (n, self.fit_floor(&basis), basis.column_classes().len())
    }

    /// The independent columns of `basis` the atoms of the samples a fit would use touch,
    /// |∪ S_k| (`trust::key_column_counts`), or the basis's Q′ when they touch none (no usable
    /// sample): the floor every gate on fitting counts against (M8a user-fix 3, the controller's
    /// ruling on concern 2: the global Q′ gates fitting nowhere). Cached on a fingerprint of the
    /// samples (count, energies, held-out codes) and the basis (species, shape).
    pub(super) fn fit_floor(&self, basis: &Basis) -> usize {
        use std::hash::{Hash, Hasher};
        let Some(store) = self.store.as_ref() else { return basis.n_independent_columns() };
        let mut h = std::collections::hash_map::DefaultHasher::new();
        store.samples.len().hash(&mut h);
        for s in &store.samples {
            s.energy.to_bits().hash(&mut h);
            // the fitted atoms of a capped segment are the ones counted (M8a Task 10)
            (&s.force_mask, s.use_energy).hash(&mut h);
        }
        self.quarantine.hash(&mut h);
        (&basis.species, basis.n_max, basis.l_max, basis.degree_cap, basis.body3, basis.body4, basis.body4_degree, basis.r_cut.to_bits()).hash(&mut h);
        let key = h.finish();
        if let Some(&(k, f)) = self.fit_floor_cache.lock().unwrap().get(&basis.body4_degree) {
            if k == key {
                return f;
            }
        }
        let union = trust::key_column_counts(basis, store.samples.iter().enumerate().filter(|(i, _)| !self.excluded(*i)).map(|(_, s)| s)).union;
        let f = if union > 0 { union } else { basis.n_independent_columns() };
        self.fit_floor_cache.lock().unwrap().insert(basis.body4_degree, (key, f));
        f
    }

    /// The body-4 degree a fit of these species is built at (spec 2026-11-13 §4.1, "grown with the
    /// store"): the highest D₄ from [`surrogate::BODY4_DEGREE_CAP`] down to
    /// [`surrogate::BODY4_DEGREE_START`] at which the samples a fit would use
    /// (`FitReport::n_samples`, the count the `Samples` row compares) meet the trust floor — the
    /// `Samples` floor ([`fit_floor`](Self::fit_floor) of [`fit_basis`](Self::fit_basis) at that
    /// degree, the columns the fitted atoms touch) plus [`MIN_RESIDUAL_DOF`], the convention of
    /// `trust_floor` (controller ruling 4: at the bare floor the residual degrees of freedom are
    /// short, and a collection trusted at 2 would step up into an untrusted fit). The same test
    /// at every fit: it steps down when samples are forgotten or held below that floor, or when
    /// appended samples touch new columns and raise the floor above the count. Without the body-4
    /// block (per scene) the start, which such a basis carries unused.
    pub(super) fn grown_body4_degree(&self, species: &[u8]) -> usize {
        let _prof = surrogate::prof::span("P3 grown_body4_degree");
        let start = surrogate::BODY4_DEGREE_START;
        let basis = self.fit_basis(species);
        if !basis.body4 {
            return start;
        }
        #[cfg(test)]
        if let Some(d) = self.body4_force {
            return d;
        }
        let Some(store) = self.store.as_ref() else { return start };
        let fitted = (0..store.samples.len()).filter(|&i| !self.excluded(i)).count();
        (start + 1..=surrogate::BODY4_DEGREE_CAP)
            .rev()
            .find(|&d| {
                let mut at = basis.clone();
                at.body4_degree = d;
                fitted >= self.fit_floor(&at) + MIN_RESIDUAL_DOF as usize
            })
            .unwrap_or(start)
    }

    /// The standing model's body-4 degree D₄ (spec 2026-11-13 §4.1; the Model tab's fit line):
    /// `Some` only while its basis has the body-4 block (a collection's), `None` per scene or
    /// without a model.
    pub fn body4_degree(&self) -> Option<usize> {
        self.model.as_ref().filter(|m| m.basis.body4).map(|m| m.basis.body4_degree)
    }

    /// The basis a fit of these species is built on: the default (`Basis::new`), the recovery
    /// degree cap when one stands, and the body-4 block exactly when a shared collection is
    /// attached (M8a Task 5: the collection's capacity; a per-scene model keeps its basis bit for
    /// bit, `surrogate::descriptor` "Body-4 features").
    pub(super) fn fit_basis(&self, species: &[u8]) -> Basis {
        let _prof = surrogate::prof::span("P3 fit_basis");
        let mut b = self.default_basis(species);
        if let Some(d) = self.degree_cap.filter(|_| b.body3) {
            b.degree_cap = d;
        }
        b
    }

    /// [`fit_basis`](Self::fit_basis) without the recovery degree cap: `Basis::new` with body-4 and
    /// the collection's reach (`collection_r_cut`) exactly when a shared collection is attached (the
    /// trust floor's and the no-model cap's count).
    pub(super) fn default_basis(&self, species: &[u8]) -> Basis {
        let mut b = Basis::new(species);
        b.body4 = self.shared.is_some();
        // a collection's reach (M8a Task 6: `surrogate::COLLECTION_R_CUT`); per scene, `Basis::new`'s
        if self.shared.is_some() {
            b.r_cut = self.collection_r_cut;
        }
        #[cfg(test)]
        {
            b.body4 &= !self.body4_off;
        }
        // the degree the last fit chose (`grown_body4_degree`); a basis without body-4 keeps
        // `Basis::new`'s, bit for bit
        if b.body4 {
            b.body4_degree = self.body4_degree;
        }
        b
    }

    /// The model's provenance against `scene`'s, for the readouts (`TrainableStatus`).
    fn provenance_mismatch(&self, scene: &Scene) -> Option<String> {
        let want = self.provenance_for(scene);
        if self.shared.is_some() {
            return self.collection_refuses(&want);
        }
        self.model.as_ref().filter(|m| m.provenance != want).map(|m| {
            let p = &m.provenance;
            format!(
                "the model was trained at {}³ h {} σ {}, charge {}, multiplicity {} — the scene is {}³ h {} σ {}, charge {}, multiplicity {}",
                p.grid_n,
                p.h,
                p.sigma,
                p.charge,
                p.multiplicity.map_or("lowest".into(), |m| m.to_string()),
                want.grid_n,
                want.h,
                want.sigma,
                want.charge,
                want.multiplicity.map_or("lowest".into(), |m| m.to_string()),
            )
        })
    }
}

/// No atom of `pos` has a neighbour within `r_cut` (Bohr; strictly inside, as rule 5a's f_c > 0):
/// every descriptor is empty. Such a geometry teaches the model nothing — its energy is the sum
/// of the references, which the per-box reference solves cover — and its teacher energies carry
/// only an isolated atom's grid/box artefacts (user test, M8a: 202 lone-H samples spread by
/// 1.3 mHa failed rule 4's self-check and the whole collection went untrusted). Never kept.
/// The predicate is the `surrogate` crate's (`surrogate::no_neighbours`), which a collection's
/// import uses too; samples of this kind stored before the rule are held out at load (`Q_LONE`).
pub(crate) fn no_neighbours(pos: &[[f64; 3]], r_cut: f64) -> bool {
    surrogate::no_neighbours(pos, r_cut)
}

/// Why a pass on a scene where no atom has a neighbour within `r_cut` is refused.
pub(crate) fn lone_pass_refusal(r_cut: f64) -> String {
    format!("a lone atom has nothing to learn — the model learns from neighbours within {r_cut} Bohr; train a molecule (e.g. H₂)")
}

/// The Model tab's note on atoms without a neighbour within `r_cut` (Bohr): every atom — the pass's
/// refusal ([`lone_pass_refusal`]); some — they are not learned (the pass still runs); none, or
/// no atoms — `None`. Informational: decides nothing.
pub fn lone_atoms_note(pos: &[[f64; 3]], r_cut: f64) -> Option<String> {
    let r2 = r_cut * r_cut;
    let near = |a: &[f64; 3], b: &[f64; 3]| (0..3).map(|k| (a[k] - b[k]) * (a[k] - b[k])).sum::<f64>() < r2;
    let lone = pos.iter().enumerate().filter(|&(i, a)| !pos.iter().enumerate().any(|(j, b)| j != i && near(a, b))).count();
    let n = pos.len();
    match lone {
        0 => None,
        _ if lone == n => Some(lone_pass_refusal(r_cut)),
        1 => Some(format!(
            "1 of {n} atoms has no neighbour within {r_cut} Bohr — it won't be learned; move it within {r_cut} Bohr (≈ {:.1} Å) of another atom",
            units::bohr_to_angstrom(r_cut)
        )),
        _ => Some(format!(
            "{lone} of {n} atoms have no neighbour within {r_cut} Bohr — they won't be learned; move them within {r_cut} Bohr (≈ {:.1} Å) of another atom",
            units::bohr_to_angstrom(r_cut)
        )),
    }
}

impl SurrogateKs {
    /// The model's cutoff (Bohr): its basis's, else the basis a fit would use (a shared
    /// collection's reach, `Basis::new`'s per scene).
    pub fn cutoff(&self) -> f64 {
        self.basis().map_or(self.default_basis(&[]).r_cut, |b| b.r_cut)
    }
}

impl Trainable for SurrogateKs {
    fn start_pass(&mut self, scene: &Scene, plan: &SamplingPlan, refs: &mut AtomReferences) -> Result<(), String> {
        if self.pass.is_some() {
            return Err("a pass is already running (cancel it first)".into());
        }
        if let Some(why) = self.collection_refuses(&self.provenance_for(scene)) {
            return Err(format!("sampling: {why}"));
        }
        let r_cut = self.cutoff();
        let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        if !pos.is_empty() && no_neighbours(&pos, r_cut) {
            return Err(lone_pass_refusal(r_cut));
        }
        let sampler = Sampler::with_cutoff(scene, plan, r_cut)?;
        // a previous pass's tail first, before this pass touches any state (`drop_pass_tail`)
        self.drop_pass_tail(NEW_PASS_FIRST);
        // the caller's references seed the cache (an entry the cache already holds stays)
        for (key, r) in refs.iter() {
            if self.ref_cache.get(key).is_none() {
                self.ref_cache.insert(key.clone(), r.clone());
            }
        }
        let sampler = sampler.with_references(std::mem::take(&mut self.ref_cache)).with_body4(self.default_basis(&[]).body4);
        self.pass_start = Some(self.samples());
        if let Some(sh) = self.shared.as_mut() {
            sh.pass_start = Some(sh.coll.samples.len());
        }
        let p = sampler.progress();
        self.user_notes.push(format!(
            "sampling pass started: {} jobs ({} pair scans), the dynamics are paused until it ends",
            p.total,
            sampler.scans().len()
        ));
        self.begin_pass(sampler);
        Ok(())
    }

    fn pass_step(&mut self, should_abort: &mut dyn FnMut() -> bool) -> Result<Option<SamplingProgress>, ProviderError> {
        let Some(mut sampler) = self.pass.take() else { return Ok(None) };
        self.last_fallback = false; // the teacher solves a pass geometry
        // a failed GPU ends the pass (module doc, "A failed GPU"): before its next job, and after the job that found it
        let r = match self.gpu_failure_error() {
            Some(e) => Err(e),
            None => sampler.step_own_teacher(self, should_abort),
        };
        let r = match (r, self.gpu_failure_error()) {
            (Ok(_), Some(e)) => Err(e),
            (r, _) => r,
        };
        self.user_notes.extend(sampler.take_notes());
        match r {
            Ok(Some(p)) => {
                self.pass_busy = 0;
                self.pass = Some(sampler);
                Ok(Some(p))
            }
            Err(ProviderError::Cancelled) => {
                self.pass = Some(sampler);
                Err(ProviderError::Cancelled)
            }
            Ok(None) => {
                self.finish_pass(sampler);
                Ok(None)
            }
            // a busy device is not the pass's failure: the job went back on the queue, so the pass
            // is kept and the driver (told "cancelled", i.e. nothing to report) tries again next tick
            Err(e) if pass_step_is_transient(&e, sampler.progress().finished, self.pass_busy) => {
                self.pass_busy += 1;
                self.user_notes.push(format!("sampling: {e}; the job is kept and retried ({}/{PASS_BUSY_RETRIES})", self.pass_busy));
                self.pass = Some(sampler);
                Err(ProviderError::Cancelled)
            }
            Err(e) => {
                self.pass_busy = 0;
                self.finish_pass(sampler);
                Err(e)
            }
        }
    }

    fn start_teach(&mut self, scene: &Scene) -> Result<(), String> {
        SurrogateKs::start_teach(self, scene)
    }

    fn cancel_pass(&mut self) {
        if let Some(mut sampler) = self.pass.take() {
            sampler.cancel();
            self.finish_pass(sampler);
        }
    }

    fn pass_checkpoint(&self) -> Option<String> {
        let sampler = self.pass.as_ref()?;
        let (prov, samples, start) = self.checkpoint_counts(sampler.scene());
        sampler.checkpoint(&prov, samples, start)
    }

    fn resume_pass(&mut self, scene: &Scene, checkpoint: &str, refs: &mut AtomReferences) -> Result<(), String> {
        if self.pass.is_some() {
            return Err("a pass is already running (cancel it first)".into());
        }
        let info = crate::sampling::checkpoint_info(checkpoint)?;
        // a pass the previous solver started never goes on here (Fermi–Dirac spec §7.7)
        if let Some(why) = self.legacy_checkpoint_refusal(scene, &info.provenance) {
            return Err(why);
        }
        let (want, _, _) = self.checkpoint_counts(scene);
        if info.provenance != want {
            return Err(format!("it was taught at {}, the scene is solved at {want}", info.provenance));
        }
        let same = |a: &Scene, b: &Scene| {
            a.cell == b.cell
                && a.charge == b.charge
                && a.multiplicity == b.multiplicity
                && a.sigma == b.sigma
                && a.ionic == b.ionic
                && a.nuclei.len() == b.nuclei.len()
                && a.nuclei.iter().zip(&b.nuclei).all(|(x, y)| x.z == y.z && x.a == y.a && x.pos == y.pos)
        };
        if !same(&info.scene, scene) {
            return Err("the scene is not the one the interrupted pass was sampling".into());
        }
        if self.shared.is_some() {
            // collection mode: the collection kept the pass's samples (`checkpoint_counts`)
            if let Some(why) = self.collection_resume_refusal(info.samples) {
                return Err(why);
            }
        } else if info.samples != self.samples() {
            return Err(format!(
                "its samples file holds {} samples, the checkpoint was written with {} — they were not saved together",
                self.samples(),
                info.samples
            ));
        }
        let sampler = crate::sampling::Sampler::resume(checkpoint)?;
        // a previous pass's tail first, before this pass touches any state (`drop_pass_tail`)
        self.drop_pass_tail(NEW_PASS_FIRST);
        for (key, r) in refs.iter() {
            if self.ref_cache.get(key).is_none() {
                self.ref_cache.insert(key.clone(), r.clone());
            }
        }
        let sampler = sampler.with_references(std::mem::take(&mut self.ref_cache)).with_body4(self.default_basis(&[]).body4);
        self.pass_start = Some(info.pass_start.min(self.samples()));
        if let Some(sh) = self.shared.as_mut() {
            sh.pass_start = Some(info.pass_start);
        }
        let p = sampler.progress();
        self.user_notes.push(format!(
            "sampling pass resumed at sample {} of {} ({}): the {} samples taught before it stopped are kept, the dynamics are paused until it ends",
            p.done + 1,
            p.total,
            p.stage,
            self.samples().saturating_sub(info.pass_start)
        ));
        self.begin_pass(sampler);
        Ok(())
    }

    fn rescale_pass(&mut self, plan: &SamplingPlan) {
        if let Some(sampler) = &mut self.pass {
            sampler.rescale(plan);
        }
    }

    fn migration(&self) -> Option<sim::MigrationStatus> {
        self.migration_status()
    }

    fn start_migration(&mut self) -> Result<(), String> {
        self.start_migration_pass()
    }

    fn pass_active(&self) -> bool {
        self.pass.is_some()
    }

    fn progress(&self) -> Option<SamplingProgress> {
        self.pass.as_ref().map(|s| s.progress()).or_else(|| self.last_pass.clone())
    }

    fn pass_ghost(&self) -> Option<sim::sampling::PassGhost> {
        self.pass.as_ref().and_then(|s| s.ghost(self))
    }

    fn refit(&mut self) -> Result<String, String> {
        let r = SurrogateKs::refit(self);
        // a pass's tail waiting for its fit (a caller that refits in place after all): this is it
        if let Some(tail) = self.pass_tail.take() {
            self.pass_fit_due = false;
            // (a running background refit no longer carries it: a later pass's tail is not its)
            if let Some(b) = self.background.as_mut() {
                b.pass_fit = false;
            }
            self.complete_pass_tail(tail, PassFit::Fitted(r.clone()));
        }
        self.refit_summary(r)
    }

    fn begin_refit_background(&mut self) -> bool {
        self.begin_background_refit()
    }

    fn fit_inputs_gen(&self) -> Option<u64> {
        Some(SurrogateKs::fit_inputs_gen(self))
    }

    fn set_fits_in_background(&mut self, on: bool) {
        self.fits_in_background = on;
    }

    fn first_fit_ready(&self) -> bool {
        SurrogateKs::first_fit_ready(self)
    }

    fn request_first_fit(&mut self) -> bool {
        SurrogateKs::request_first_fit(self)
    }

    fn poll_refit_background(&mut self) -> sim::RefitPoll {
        match self.poll_background_refit() {
            BackgroundPoll::Idle => sim::RefitPoll::Idle,
            BackgroundPoll::Running => sim::RefitPoll::Running,
            BackgroundPoll::Discarded(why) => sim::RefitPoll::Discarded(why),
            BackgroundPoll::Installed(r) => sim::RefitPoll::Done(self.refit_summary(r)),
        }
    }

    fn forget(&mut self) {
        // a running pass ends without a refit; its references stay cached
        if let Some(mut sampler) = self.pass.take() {
            self.teacher.discontinuity();
            self.ref_cache = sampler.take_references();
        }
        let n = self.samples();
        let shared = self.shared.as_ref().map(|s| (s.coll.key.slug(), s.coll.samples.len()));
        SurrogateKs::forget(self);
        self.last_pass = None;
        self.user_notes.push(match shared {
            Some((slug, k)) => format!("model forgotten: the shared collection {slug} is let go — it stays on disk with its {k} samples (attach it again to use it); the teacher answers every step"),
            None => format!("model forgotten ({n} samples dropped): the teacher answers every step"),
        });
    }

    fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        self.save_sidecar(path)
    }

    fn load(&mut self, path: &std::path::Path, scene: &Scene) -> Result<(), String> {
        if self.pass.is_some() {
            return Err("load model: cancel the running sampling pass first".into());
        }
        let sc = match surrogate::sidecar::read_any(path).map_err(|e| format!("load model: {e}"))? {
            surrogate::sidecar::Contents::Fitted(sc) => *sc,
            surrogate::sidecar::Contents::Samples(set) if !set.specialised.is_empty() => return self.load_specialised(path, set, scene),
            surrogate::sidecar::Contents::Samples(set) => return self.load_samples(path, set, scene),
        };
        // a model of the previous solver (spec §7.9): its reason, never the generic "another surface"
        if let Some(why) = legacy_model_refusal(&sc.model.provenance.ionic) {
            let msg = format!("model {} refused: {why}", path.display());
            self.user_notes.push(msg.clone());
            return Err(msg);
        }
        let want = self.provenance_for(scene);
        if sc.model.provenance != want {
            let msg = format!(
                "model {} refused: it was trained at {} but the scene is solved at {} — another Kohn-Sham surface",
                path.display(),
                sc.model.provenance.describe(),
                want.describe()
            );
            self.user_notes.push(msg.clone());
            return Err(msg);
        }
        let prov = sc.model.provenance.clone();
        let refs: Vec<(u8, f64)> = sc.model.basis.species.iter().copied().zip(sc.model.ref_energy.iter().copied()).collect();
        let n = sc.samples.len();
        let s9 = sc.reference_rule.as_deref() == Some(surrogate::sidecar::REFERENCE_RULE);
        // a model file switches the provider to it (a specialised, per-scene model); the shared
        // collection comes back if the install fails (`shared` module doc)
        let shared = self.detach_collection();
        let installed = self.install_sidecar(sc);
        if installed.is_err() {
            if let Some(c) = shared {
                self.reattach_after_failed_load(c);
            }
        }
        match installed {
            Ok(()) => {
                self.set_source_file(Some(path.to_path_buf()));
                let trusted = self.trusted();
                if s9 {
                    for (z, e) in refs {
                        // an energy without a table (the sidecar carries none): a solved entry
                        // under the same key is kept
                        self.ref_cache.seed_energy(atom_key(&prov, z), e, None, prov.tol, &prov.teacher);
                    }
                } else {
                    // pre-S9 references: kept for the loaded fit (they are consistent with it),
                    // never cached, so the next pass re-solves them (or takes an S9 value an
                    // earlier pass cached) and its refit rebuilds on them
                    self.user_notes.push(format!(
                        "model {}: its reference energies predate ruling S9 (lowest-energy multiplicity) — the next sampling pass re-solves them and the refit rebuilds the model on them",
                        path.display()
                    ));
                }
                self.last_pass = None;
                self.user_notes.push(format!("model loaded from {} ({n} samples, {})", path.display(), if trusted { "trusted" } else { "untrusted" }));
                Ok(())
            }
            Err(e) => Err(format!("load model {}: {e}", path.display())),
        }
    }

    fn teacher_mut(&mut self) -> &mut dyn ForceProvider {
        SurrogateKs::teacher_mut(self)
    }

    fn attach_collection(&mut self, root: &std::path::Path, scene: &Scene) -> Result<(), String> {
        self.attach_collection_at(root, scene)
    }

    fn model_source(&self) -> sim::ModelSource {
        SurrogateKs::model_source(self)
    }

    fn save_session_only(&self, path: &std::path::Path) -> std::io::Result<usize> {
        SurrogateKs::save_session_only(self, path)
    }

    fn save_specialised(&self, path: &std::path::Path, species: &[u8]) -> Result<usize, String> {
        SurrogateKs::save_specialised(self, path, species)
    }

    fn specialised(&self) -> Vec<u8> {
        self.specialised.clone()
    }

    fn model_species(&self) -> Vec<u8> {
        self.basis().map_or_else(Vec::new, |b| b.species.clone())
    }

    fn load_session_only(&mut self, path: &std::path::Path) -> Result<usize, String> {
        SurrogateKs::load_session_only(self, path)
    }

    fn import_model(&mut self, path: &std::path::Path) -> Result<sim::ImportSummary, String> {
        let r = self.import_into_collection(path)?;
        let refused = r.refused.into_iter().map(|(i, why)| ((i != surrogate::collection::WHOLE_FILE).then_some(i), why)).collect();
        Ok(sim::ImportSummary { imported: r.imported, refused, notes: r.notes })
    }

    fn retry_fit(&mut self, scene: &Scene, refs: &mut AtomReferences) -> Result<bool, String> {
        SurrogateKs::retry_fit(self, scene, refs)
    }

    fn status(&self, scene: &Scene, forces: &[[f64; 3]]) -> TrainableStatus {
        let fmin = forces.iter().map(|f| (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt()).fold(f64::INFINITY, f64::min);
        TrainableStatus {
            teacher: self.teacher.name(),
            fitted: self.model.is_some(),
            trusted: self.trusted(),
            untrusted_reason: self.model.as_ref().and_then(|_| self.untrusted_reason()),
            samples: self.samples(),
            pending_refit: self.pending_refit(),
            epoch: self.epoch(),
            fit: self.report().map(|r| (r.n_samples, r.q_prime, r.rmse_energy_per_atom, r.rmse_force)),
            body4_degree: self.body4_degree(),
            residuals: self.training_residuals().map(|t| (t.energy_per_atom, t.force, t.force_bound)),
            force_bound: self.policy.force_bound(if fmin.is_finite() { fmin } else { 0.0 }),
            provenance_mismatch: self.provenance_mismatch(scene),
            provenance: match &self.shared {
                Some(sh) => self.model.as_ref().map(|_| format!("shared collection {}: {} — any box, neutral, automatic multiplicity", sh.coll.key.slug(), sh.coll.key.describe())),
                None => self.model.as_ref().map(|m| m.provenance.describe_recorded()),
            },
            audit_every: self.policy.audit_every,
            draft: self.policy.never_fallback,
            quarantined: self.quarantined(),
            superseded: self.superseded(),
            forgotten: self.forgotten(),
            forgotten_in_fit: self.forgotten_in_fit(),
            box_excluded: self.box_excluded(),
            held_out: self.held_out(),
            lone: self.lone_held_out(),
            box_affected: self.box_affected(),
            trust_floor: self.trust_floor(),
            unfitted: self.unfitted(),
            pending: self.pending_samples(),
            recovery: self.recovery_note().map(str::to_string),
            audits_background: self.audits_asynchronous(),
            audit_in_flight: self.audit_in_flight(),
            last_audit: self.last_audit().cloned(),
            last_audit_this_fit: self.last_audit_this_fit(),
            audits_box_unjudged: self.box_audits_unjudged,
            stuck: self.stuck(),
            class_loo: self.class_loo.clone(),
            solves: self.solve_counts(),
            trust_checks: self.trust_checks(),
            stuck_info: self.stuck_info(),
            model_gen: self.model_generation(),
            readiness: self.readiness(scene),
            teach_offer: self.teach_offer(),
            no_model: self.no_model().map(|n| n.words),
            // (not while the first fit runs in the background: a Retry would fit in place)
            retry_fit: self.pass.is_none() && self.background.is_none() && self.no_model().is_some_and(|n| n.retry),
            model_source: self.model_source(),
            lone_note: lone_atoms_note(&scene.nuclei.iter().map(|n| n.pos).collect::<Vec<_>>(), self.cutoff()),
            stray: self.stray_status(),
            stray_held_out: self.stray_held_out(),
            ticks: self.calibration_ticks(),
        }
    }

    fn fitted(&self) -> bool {
        self.model.is_some()
    }

    fn samples(&self) -> usize {
        SurrogateKs::samples(self)
    }

    fn pending_refit(&self) -> bool {
        SurrogateKs::pending_refit(self)
    }

    fn audits_background(&self) -> bool {
        self.audits_asynchronous()
    }

    fn forget_sample(&mut self, id: sim::SampleId) -> Result<(), String> {
        SurrogateKs::forget_sample(self, id)
    }

    fn restore_sample(&mut self, id: sim::SampleId) -> Result<(), String> {
        SurrogateKs::restore_sample(self, id)
    }

    fn forget_samples(&mut self, ids: &[sim::SampleId]) -> Result<(), String> {
        SurrogateKs::forget_samples(self, ids)
    }

    fn restore_samples(&mut self, ids: &[sim::SampleId]) -> Result<(), String> {
        SurrogateKs::restore_samples(self, ids)
    }

    fn user_changes_pending(&self) -> usize {
        SurrogateKs::user_changes_pending(self)
    }

    fn forget_by_rule(&mut self, rule: &sim::knowledge::ForgetRule, expected: usize) -> Result<usize, String> {
        SurrogateKs::forget_by_rule(self, rule, expected)
    }

    fn sample_suspects(&self) -> Vec<sim::SampleSuspect> {
        SurrogateKs::sample_suspects(self)
    }

    fn forgotten_samples(&self) -> Vec<sim::SampleSuspect> {
        SurrogateKs::forgotten_samples(self)
    }

    fn purge_forgotten(&mut self) -> Result<sim::PurgeReport, String> {
        SurrogateKs::purge_forgotten(self)
    }

    fn purge_refusal(&self) -> Option<String> {
        SurrogateKs::purge_refusal(self)
    }

    fn forgotten_in_collection(&self) -> usize {
        SurrogateKs::forgotten_in_collection(self)
    }

    fn set_audit_every(&mut self, n: usize) {
        if self.policy.audit_every != n {
            self.set_policy(FallbackPolicy { audit_every: n, ..self.policy });
        }
    }

    fn set_draft(&mut self, on: bool) {
        if self.policy.never_fallback != on {
            self.set_policy(FallbackPolicy { never_fallback: on, ..self.policy });
        }
    }

    fn force_teacher_next(&mut self) {
        SurrogateKs::force_teacher_next(self)
    }

    fn take_notes(&mut self) -> Vec<String> {
        std::mem::take(&mut self.user_notes)
    }

    fn take_teach_events(&mut self) -> Vec<TeachEvent> {
        self.take_events()
    }

    fn training_views(&mut self) -> Option<sim::TrainingViews> {
        SurrogateKs::training_views(self)
    }

    fn knowledge_map(&self) -> Option<sim::KnowledgeMap> {
        SurrogateKs::knowledge_map(self)
    }

    fn triple_map(&self) -> Option<sim::knowledge::TripleMap> {
        SurrogateKs::triple_map(self)
    }

    fn request_display(&mut self, scene: &Scene) -> bool {
        SurrogateKs::request_display(self, scene)
    }

    /// A Harris payload polled in (`poll_side_jobs`, e.g. by the worker's per-round
    /// `take_atom_densities`) but not yet taken by the driver is still in flight: else the worker's
    /// lost-job check would read the job as lost and re-send it (final review M2).
    fn side_job_in_flight(&self) -> bool {
        self.auditor.as_ref().is_some_and(|a| a.busy.is_some()) || !self.side_queue.is_empty() || self.harris_queued.is_some() || self.side_harris.is_some()
    }

    fn request_atom_density(&mut self, z: u8, scene: &Scene) -> bool {
        SurrogateKs::request_atom_density(self, z, scene)
    }

    fn take_atom_densities(&mut self) -> Vec<(u8, Display)> {
        SurrogateKs::take_atom_densities(self)
    }

    fn request_atom_reference(&mut self, z: u8, mass: f64, scene: &Scene) -> bool {
        SurrogateKs::request_atom_reference(self, z, mass, scene)
    }

    fn take_atom_references(&mut self) -> Vec<(u8, Result<AtomReference, ProviderError>)> {
        SurrogateKs::take_atom_references(self)
    }

    fn atom_reference_done(&self) -> Option<usize> {
        SurrogateKs::atom_reference_done(self)
    }
}

/// Why samples are held but no model formed (module doc, "No model yet"): the plain words, and
/// whether the Retry (`SurrogateKs::retry_fit`) can help.
#[derive(Clone, Debug, PartialEq)]
pub struct NoModel {
    pub words: String,
    pub retry: bool,
    /// The species without a reference energy (sorted).
    pub missing: Vec<u8>,
}

impl SurrogateKs {
    /// `Trainable::load` of a samples-only file (`surrogate::sidecar` version 6): the provenance
    /// checked against `scene` as for a model, then `install_samples` — which fits when it can —
    /// and its S9 references seeded into the cache (a later pass skips their scans).
    fn load_samples(&mut self, path: &std::path::Path, set: surrogate::SampleSet, scene: &Scene) -> Result<(), String> {
        if let Some(why) = legacy_model_refusal(&set.provenance.ionic) {
            let msg = format!("samples {} refused: {why}", path.display());
            self.user_notes.push(msg.clone());
            return Err(msg);
        }
        let want = self.provenance_for(scene);
        if set.provenance != want {
            let msg = format!(
                "samples {} refused: they were taught at {} but the scene is solved at {} — another Kohn-Sham surface",
                path.display(),
                set.provenance.describe(),
                want.describe()
            );
            self.user_notes.push(msg.clone());
            return Err(msg);
        }
        let prov = set.provenance.clone();
        let refs = set.references.clone();
        let s9 = set.reference_rule.as_deref() == Some(surrogate::sidecar::REFERENCE_RULE);
        let n = set.samples.len();
        let shared = self.detach_collection();
        if let Err(e) = self.install_samples(set) {
            if let Some(c) = shared {
                self.reattach_after_failed_load(c);
            }
            return Err(format!("load model {}: {e}", path.display()));
        }
        self.set_source_file(Some(path.to_path_buf()));
        if s9 {
            for (z, e) in refs {
                self.ref_cache.seed_energy(atom_key(&prov, z), e, None, prov.tol, &prov.teacher);
            }
        }
        self.last_pass = None;
        let state = match self.no_model() {
            None if self.trusted() => "fitted, trusted".to_string(),
            None => "fitted, untrusted".to_string(),
            Some(nm) => nm.words,
        };
        self.user_notes.push(format!("samples loaded from {} ({n} samples, saved before a model formed): {state}", path.display()));
        Ok(())
    }

    /// Why there is no model although samples are held (module doc, "No model yet"); `None` with a
    /// model or without samples. In order: a species whose isolated-atom reference is missing (and
    /// how its scan failed, if it did), too few samples to fit (Q′), the last fit error, else a
    /// fit not yet made.
    pub fn no_model(&self) -> Option<NoModel> {
        if self.model.is_some() || self.samples() == 0 {
            return None;
        }
        let store = self.store.as_ref()?;
        let mut species: Vec<u8> = store.samples.iter().flat_map(|s| s.z.iter().copied()).collect();
        species.sort_unstable();
        species.dedup();
        let missing: Vec<u8> = species.iter().copied().filter(|z| !self.refs.get(z).is_some_and(|e| e.is_finite())).collect();
        let (fittable, floor, q_prime) = self.fittable_and_floor();
        let n = self.samples();
        // a reference whose SCF itself fails: a cold solve is deterministic (`sim::references`,
        // "The retry round"), so neither Retry nor another pass on this surface can change it
        let stuck: Vec<&crate::sampling::FailedReference> = missing.iter().filter_map(|z| self.failed_refs.get(z)).filter(|f| !f.transient).collect();
        if !stuck.is_empty() {
            let parts: Vec<String> = stuck.iter().map(|f| format!("the {} atom reference did not solve ({})", symbol(f.z), f.why)).collect();
            return Some(NoModel {
                words: format!(
                    "no model yet: {}. A cold solve is deterministic, so retrying here fails the same way: this solver does not converge that isolated atom at this grid and ionic potential, and a fit needs every species' isolated-atom energy. The {n} samples taught so far are kept — Save model… writes them — but they belong to this surface (another grid or ionic potential is another one)",
                    parts.join("; ")
                ),
                retry: false,
                missing,
            });
        }
        if !missing.is_empty() {
            let parts: Vec<String> = missing
                .iter()
                .map(|&z| match self.failed_refs.get(&z) {
                    Some(f) => format!("the {} atom reference was lost to the device ({}; GPU busy — a retry may succeed)", symbol(z), f.why),
                    None => format!("the {} atom reference is not solved yet", symbol(z)),
                })
                .collect();
            if fittable < floor {
                // a pass solves the references first: sampling more is the remedy for both
                return Some(NoModel {
                    words: format!(
                        "no model yet: {} usable samples, a fit needs {floor} (the model columns their atoms touch; Q′ {q_prime}), and {} — sample more (a pass solves the references first; the {n} samples so far are kept)",
                        fittable,
                        parts.join("; ")
                    ),
                    retry: false,
                    missing,
                });
            }
            return Some(NoModel {
                words: format!(
                    "no model yet: {} — a fit needs every species' isolated-atom energy. The {n} samples taught so far are kept (Save model… writes them); Retry solves the reference (every allowed multiplicity, cold) and fits",
                    parts.join("; ")
                ),
                retry: true,
                missing,
            });
        }
        if fittable < floor {
            return Some(NoModel {
                words: format!("no model yet: {fittable} usable samples, a fit needs {floor} (the model columns their atoms touch; Q′ {q_prime}) — sample more (the {n} samples so far are kept)"),
                retry: false,
                missing,
            });
        }
        // the first fit running in the background (quiet refits 2026-11-17): no Retry meanwhile
        if self.background.is_some() {
            return Some(NoModel { words: format!("no model yet: the first fit of the {n} samples is running in the background — Play once it lands"), retry: false, missing });
        }
        let words = match &self.last_fit_error {
            Some(e) => format!("no model yet: the fit failed ({e}) — Retry fits the {n} samples again"),
            None => format!("no model yet: the {n} samples are not fitted — Retry fits them"),
        };
        Some(NoModel { words, retry: true, missing })
    }

    /// The Retry (`Trainable::retry_fit`; module doc, "No model yet"): a references-only pass over
    /// the missing species on `scene` (whose settings must be the samples' — another grid's
    /// reference would be another surface's), started like any pass; else a refit at once. `Err`
    /// while a pass runs, with nothing to retry, for another surface, with too few samples, for a
    /// reference whose SCF itself failed (deterministic), and — under an ionic policy — for a
    /// missing species with no nucleus in the scene: the provenance tag says nothing about a
    /// species the scene lacks, so its lone atom could be solved on another surface (pseudised
    /// against all-electron samples) and cached under the samples' key (physics review of the fix).
    pub fn retry_fit(&mut self, scene: &Scene, refs: &mut AtomReferences) -> Result<bool, String> {
        if self.pass.is_some() {
            return Err("a pass is already running (cancel it first)".into());
        }
        if self.background.is_some() {
            return Err("the fit is running in the background — Play once it lands".into());
        }
        let Some(nm) = self.no_model() else {
            return Err(if self.model.is_some() { "the model is fitted".into() } else { "no samples yet: start sampling".into() });
        };
        if !nm.retry {
            return Err(nm.words);
        }
        if !nm.missing.is_empty() {
            let want = self.provenance_for(scene);
            if let Some(why) = self.collection_refuses(&want) {
                return Err(format!("the reference must be on the samples' surface: {why}"));
            }
            if let Some(store) = &self.store {
                if !self.same_surface(&store.provenance, &want) {
                    return Err(format!(
                        "the samples were taught at {} but the scene is solved at {} — the reference must be on the samples' surface (switch the scene back, or forget the samples)",
                        store.provenance.describe(),
                        want.describe()
                    ));
                }
            }
            if scene.ionic != sim::IonicPolicy::None {
                if let Some(&z) = nm.missing.iter().find(|&&z| !scene.nuclei.iter().any(|n| n.z == z)) {
                    return Err(format!(
                        "the scene has no {} atom: under an ionic potential its reference must be solved with one in the scene, so that the scene's settings are the samples' for it too (put one back, or set Physics ▸ Ionic potential to none if the samples are all-electron)",
                        symbol(z)
                    ));
                }
            }
            let sampler = Sampler::references_only(scene, &nm.missing)?;
            // a previous pass's tail first, before this pass touches any state (`drop_pass_tail`)
            self.drop_pass_tail(NEW_PASS_FIRST);
            for (key, r) in refs.iter() {
                if self.ref_cache.get(key).is_none() {
                    self.ref_cache.insert(key.clone(), r.clone());
                }
            }
            let sampler = sampler.with_references(std::mem::take(&mut self.ref_cache));
            self.pass_start = Some(self.samples());
            let names: Vec<&str> = nm.missing.iter().map(|&z| symbol(z)).collect();
            self.user_notes.push(format!(
                "reference retry started: the isolated {} atom{} (every allowed multiplicity, cold); the model is fitted at its end",
                names.join(", "),
                if names.len() == 1 { "" } else { "s" }
            ));
            self.begin_pass(sampler);
            return Ok(true);
        }
        match Trainable::refit(self) {
            Ok(summary) => {
                self.user_notes.push(format!("retry: fitted ({summary})"));
                Ok(false)
            }
            Err(e) => Err(format!("the fit failed again ({e})")),
        }
    }
}

/// What a pass's samples became in the fit (fix (e), 2026-10-18: the user's water + 2 H pass
/// added 185 samples and the model read "trusted" while none of them was fitted).
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct PassCoverage {
    /// The pass scene's species (sorted).
    pub(crate) species: Vec<u8>,
    pub(crate) total: usize,
    pub(crate) fitted: usize,
    pub(crate) box_excluded: usize,
    pub(crate) quarantined: usize,
    /// Forgotten by the user (`Q_FORGOTTEN`, Task 2 fix round 1 M5).
    pub(crate) forgotten: usize,
    /// Kept since the refit, or held out otherwise.
    pub(crate) other: usize,
    /// Box-affected, fitted or not (below the trust floor they are fitted).
    pub(crate) box_affected: usize,
}

impl PassCoverage {
    /// The plain words when some of the pass's samples are not in the fit; `None` when all are.
    pub(crate) fn words(&self) -> Option<String> {
        let out = self.total - self.fitted;
        if out == 0 {
            return None;
        }
        let mut why = Vec::new();
        if self.box_excluded > 0 {
            why.push(format!(
                "{} touch the cell's walls (box-affected: the density reaches the faces, so the label depends on where the molecule sits — a bigger cell, or keeping the atoms nearer the centre, fixes that)",
                self.box_excluded
            ));
        }
        if self.quarantined > 0 {
            why.push(format!("{} are quarantined (held out by the trust recovery: the rest of the data could not predict it, or the fit could not reproduce it)", self.quarantined));
        }
        if self.forgotten > 0 {
            why.push(format!("{} were forgotten by you (Model tab, Worst-fitting samples — Restore puts them back)", self.forgotten));
        }
        if self.other > 0 {
            why.push(format!("{} are otherwise left out", self.other));
        }
        Some(format!("{out} of this pass's {} samples are NOT in the fit: {}", self.total, why.join("; ")))
    }
}

/// Consecutive "GPU busy" pass steps kept for a retry before the pass ends as it does for any
/// other failure (a device that stays busy for ~15 solve attempts is not transient).
const PASS_BUSY_RETRIES: u32 = 5;

/// Whether a failed pass step is a transient busy device to retry rather than the pass's end: a
/// `SolverError::Device` (a readback the shared device did not deliver, after `GpuKs`'s own
/// retries) while the sampler can continue (`finished` is false — the failed job went back on its
/// queue; a failed reference scan or a hot burst handles its own failures) and fewer than
/// [`PASS_BUSY_RETRIES`] steps in a row have been kept. A multi-minute pass otherwise ended, with
/// a refit on what it had, on one glitch.
fn pass_step_is_transient(e: &ProviderError, finished: bool, busy: u32) -> bool {
    matches!(e, ProviderError::Solver(sim::SolverError::Device(_))) && !finished && busy < PASS_BUSY_RETRIES
}

#[cfg(test)]
mod tests;
