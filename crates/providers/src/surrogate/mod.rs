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
//! 1. **No model** (nothing fitted yet) → teacher; the sample is kept; `Source::Fallback`.
//! 2. **Unknown species** (a Z the model's `Basis` has no channel for) → teacher,
//!    unconditionally; the sample is kept and the model is marked for a rebuild with the
//!    enlarged species set at the next [`refit`](SurrogateKs::refit). Never a guess.
//! 3. **Other settings** — the scene's (grid n, h, σ, charge, multiplicity) differ from the
//!    model's `Provenance` → teacher, unconditionally; the sample is NOT kept (it belongs to a
//!    different Kohn-Sham surface than the store's).
//! 4. **Untrusted fit** (trust rule below) → teacher; the sample is kept so the fit can earn
//!    trust. The note says which condition failed, e.g. "surrogate untrusted: n_samples 12 <
//!    q_prime 16".
//!    - **4b, untrusted class** (M6 Task 11 round 6; "Class-wise trust" below): the query's
//!      environment class (pair or many-body, by the largest neighbour count within r_cut) is
//!      not trusted by the fit's exact leave-one-out errors → teacher; sample kept (never thinned
//!      while the class is short of Q′ samples — those earn its trust; thinned like rule 5
//!      otherwise); reported on the teacher's surface (the model's energy there is unvalidated).
//!      "surrogate: this kind of environment isn't learned well enough yet (many-body: LOO force
//!      error 3.2× the bound, RMS 0.41×, 45 samples)". Evaluated with the model consulted; when
//!      rule 5 fires as well its more specific note is shown. Draft mode answers anyway.
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
//!      teacher's own isolated atom (rulings S5, S9).
//!    - **5b, coordination ratio** (M6 Task 11): an atom whose coordination C_t = Σ_{j∈t} f_c(r_ij)
//!      (the n = 0 body-2 feature of neighbour species t) is more than
//!      [`surrogate::envelope::COORDINATION_RATIO`] = 3 below the smallest the fit has seen for
//!      its species — the vanishing direction judged relatively, because one kept stretched-pair
//!      sample (the user's r = 5.125, f_c 0.05) put an empty neighbourhood half a training spread
//!      inside the old spread box (derivation and the old/new decision on the repro: the
//!      envelope module doc).
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
//!    `√(energy_std² − (N s_E)²) / N ≤ energy_std_per_atom` (1e-3 Ha). The floor is excluded
//!    because it is the teacher's own scatter (egg-box, SCF lag) — the part of the residual no
//!    number of samples removes, so a fallback triggered by it would teach the model nothing;
//!    the epistemic part is what new samples shrink, so it is the part a fallback can act on.
//!    (Per atom the floor contributes s_E ≈ 1e-5 in quadrature, far below 1e-3: including it
//!    would be inert in practice, not wrong; it is excluded for the reason, not the size.)
//!    Otherwise → teacher; sample kept.
//! 8. **Audit** (ruling S7): every `audit_every`-th step the model answers (2000 by default;
//!    0 disables) the teacher checks it at that geometry, the sample is kept and the gap is
//!    recorded ([`SurrogateKs::last_audit`]). If |gap|/N > `energy_std_per_atom` — or, since
//!    M6 Task 11 round 2, any atom's force residual ‖F_model − F_teacher‖/√3 exceeds the variance
//!    rule's bound min(max(2e-3, 10 %|F_teacher|), 1e-2) — the audit **fails**: the model is
//!    untrusted until the next refit (the reason names the forces when they failed it; an energy
//!    audit alone passed the round-1 repro's 4× force errors). The force audit is **strict, at 1×
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
//! answers are kept as samples and the worker's refits continue, so the model keeps learning — but
//! a failed audit no longer stops it answering (it untrusts the model, which draft mode ignores;
//! the flip to the teacher surface is booked when draft mode is switched off, at that step's
//! positions — after a failed audit only: switched off while untrusted for another reason, or on
//! from an untrusted state, the surface changes without a booked jump and the driver restarts the
//! conserved-energy history with a note). A refit in draft mode books its jump whether or not
//! the model is trusted: the next answer is the model's either way (review I1). The energy
//! bookkeeping is otherwise unchanged: every answer is on the model surface. A region whose
//! samples were quarantined is answered by the model too, and re-learned only from the audits
//! there (review M4).
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
//! between them). An isolated-atom job runs with the view off (its teacher is reset around it
//! anyway: that is a new scene for the tracker). So **the tracks restart after an atom or
//! reference job** (Task 3 review M3): the view's producer is dropped with the view off, and the
//! next audit or refresh hands out fresh tracks (never a reused id, `crate::orbitals`), so a
//! selected orbital falls back — "selected orbital no longer exists — showing …" — even though
//! the same orbital is still there. Rare (one reference per species, mostly early in a session);
//! Task 7's tab shows that note as it stands. The stepping teacher never gets the request:
//! a fallback step carries no orbitals, and a model step has no electrons. So
//! `orbital_display_now` is `None` here; the caller asks for a display refresh instead
//! (`request_display`, which is also how "refresh every N s" updates the orbitals). Without an
//! auditor (the CPU teacher) there is no producer and the estimated solver shows no orbitals.
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
//! The model may answer only when the last `FitReport` has `n_samples ≥ q_prime` (distinct
//! geometries, not rows: the 3N force rows of one sample probe one geometry) AND
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
//! Ha/atom"). A failed audit (rule 8) is the fourth condition, until the next refit.
//! [`SurrogateKs::trusted`] exposes the verdict; every evaluate of an untrusted model is a
//! Fallback (rule 4) and the samples accumulate until a refit earns trust.
//!
//! # Class-wise trust (M6 Task 11 round 6, controller ruling)
//!
//! The training residuals ask whether the fit reproduces data it was fitted *to*; the round-5
//! strict hot run showed that is not enough: a trusted H₂ model with a few dozen kept H₃
//! fallbacks answered three-body geometries at interpolation grades (γ 0.86–1.16) 3.1–6.2× off
//! in force — the many-body samples were reproduced, but a fit without any one of them could not
//! predict it. So each refit (and installed model, and policy change) also asks the
//! leave-one-out question, per **environment class**:
//! - **Class rule** (universal, nothing per element, no constant beyond the model's r_cut): per
//!   atom its neighbour count within r_cut; per geometry the largest — 0 **isolated**, 1
//!   **pair**, ≥ 2 **many-body** ([`env_class`]). A geometry whose atoms have at most one
//!   neighbour each is a set of dimers whatever the species; one atom with two neighbours makes
//!   it many-body.
//! - **LOO residuals**: exact, closed-form, at the fit's own hyperparameters and
//!   standardisation, for the whole sample (its energy row and 3N force rows) at once:
//!   r_s = (I − H_ss)⁻¹ e_s with H_ss = Φ_s A⁻¹ Φ_sᵀ W_s (derivation and cost: `surrogate::loo`;
//!   verified against an explicit downdate of the fit to 2.4e-12). Over the fitted samples (box-excluded,
//!   quarantined and superseded skipped, as the training residuals).
//! - **Trusted class**: ≥ Q′ samples (the trust rule's minimum count), and over its samples'
//!   atoms the LOO force error ‖r_F,j‖/√3 against the atom's bound `force_bound(|F_j|)` (the
//!   variance rule's: max(2e-3, 10 %|F|) capped at 1e-2) has RMS ≤ 1 **and** max ≤ 1, and every
//!   sample's LOO energy |r_E|/N ≤ `energy_std_per_atom` (the energy tier). A sample whose LOO is
//!   undefined (I − H_ss singular) counts as infinitely off.
//! - **The isolated class is exempt**: in a scene of two or more atoms rule 5a refuses it
//!   anyway, and a one-atom scene's answer is E_ref, the teacher's own isolated atom (S9).
//!
//! The per-class numbers (samples, LOO force RMS and max ratios, energy ratio, verdict) are in
//! `TrainableStatus::class_loo` for the Model tab. **Costs of the max** (measured, rule 7): one
//! sample alone in its region decides its class — the GPU repro's kept tail sample at r = 5.125
//! (LOO force 14.9× its bound) untrusts the H₂ pair class until the region fills in (all 1600
//! heating steps fall back), and a first H–He sample (LOO energy 1.1e3× the tier: no other row
//! has an H–He feature) untrusts H₂ with it until H–He is learned
//! (`new_species_falls_back_and_is_learned_at_the_next_refit`: trusted again with 40 H–He
//! dimers). Both are teacher steps, never wrong answers.
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
//!    bent molecules with wide bond excursions) and keeps it if the residuals then pass;
//! 2. otherwise **quarantines** the worst offenders — the samples whose per-atom energy or worst
//!    force residual, relative to its threshold, is within a factor 2 of the worst — and refits
//!    without them, up to [`QUARANTINE_ROUNDS`] rounds, never more than
//!    [`QUARANTINE_MAX_FRACTION`] of the store. A quarantined sample stays in the store (and in
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
//! and once the store holds the trust floor (Q′ + 30) without them, the next refit excludes
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
//! but are **mutually consistent** (every pair within `energy_std_per_atom` per atom and every
//! atom's force bound), the third is kept (tag Fallback) and the next refit decides: if the new
//! model reproduces the kept answer it stays in the fit — the fenced glitch is then released if
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
//! Thinned: the envelope rule (5), a residual-untrusted model (rule 4) and an untrusted class
//! that already holds Q′ samples (4b; short of them its fallbacks are exempt).
//!
//! **No model yet** (final review B2): before the first fit there is no envelope to thin by,
//! and the worker refits only once a model exists (the sampling pass makes the first fit and
//! solves the references it needs) — so while the user plays with a pass offer waiting, every
//! DFT step used to add a sample with no bound. Beyond [`NO_MODEL_SAMPLE_FACTOR`] · Q′ samples
//! (Q′ of the held species' default basis: past the trust floor Q′ + 30 for every species set)
//! one teacher answer in [`THIN_EVERY`] is kept, with a note saying how many are held and
//! that a sampling pass is what builds the model.
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

use crate::sampling::{atom_key, Sampler};
use sim::activity::{SolveCounts, TeachEvent, TeachOrigin, TeachOutcome, TeachRule};
use sim::references::{AtomReference, AtomReferences};
use sim::sampling::{SamplingPlan, SamplingProgress};
use sim::{
    ClassLoo, Display, EvalStats, Extrapolation, ForceProvider, ForceResult, OrbitalDisplay, OrbitalViewRequest, ProviderError, Scene,
    SideDisplay, SideKind, Source, Trainable, TrainableStatus,
};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Instant;
use surrogate::{
    describe_values, Accumulator, ActiveSet, Basis, Envelope, FitError, FitOptions, FitReport, Model, NullSpace, Provenance,
    Sample, SampleStore, SampleTag, StoreError,
};
use surrogate::sidecar::TeacherSolves;

mod audit;
mod ledger;
mod policy;
mod quarantine;
mod store;
mod trust;
mod views;

use audit::*;
use ledger::*;
use policy::*;
use quarantine::*;
pub use audit::{AuditRecord, AuditVerdict};
pub use ledger::EVENT_QUEUE;
pub use policy::{box_affected_face_density, FallbackPolicy, BOX_AFFECTED_FACE_DENSITY, FORCE_STD_BOUND_MAX, GRADE_MAX};
pub use quarantine::{
    FENCE_VOTES, FENCE_VOTE_LIMIT, NO_MODEL_SAMPLE_FACTOR, QUARANTINE_FRACTION, QUARANTINE_MAX_FRACTION, QUARANTINE_ROUNDS,
    RECOVERY_DEGREE_CAP, THIN_EVERY, THIN_FRACTION,
};
pub use trust::{env_class, TrainingResiduals, ENV_CLASSES, MIN_RESIDUAL_DOF, STUCK_STEPS};
pub use views::Verdict;

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
}

/// The model's answer at the geometry of a teacher step (rules 5–7 and audits).
struct ModelAnswer {
    energy: f64,
    forces: Vec<[f64; 3]>,
    uncertainty: Option<f64>,
}

/// One teacher step: why (the note's text, the rule's code and the atom it names), with what
/// from the model, and whether it is an audit.
struct Handoff {
    prov: Provenance,
    pos: Vec<[f64; 3]>,
    z: Vec<u8>,
    reason: Refusal,
    keep_sample: bool,
    /// The kept sample is subject to thinning (module doc).
    thin: bool,
    /// Quarantined samples within the quarantine radius of this geometry (module doc).
    near: Vec<usize>,
    model: Option<ModelAnswer>,
    audit: bool,
    /// Report the step on the teacher's surface although the model was consulted (rule 5, the
    /// first evaluation after an edit; module doc, "One-surface energy accounting").
    teacher_surface: bool,
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
    /// The extrapolation grade's MaxVol active sets (rule 5e, `surrogate::grade`).
    active_set: Option<ActiveSet>,
    training_residuals: Option<TrainingResiduals>,
    /// Per environment class (pair, many-body): the fit's leave-one-out numbers and verdict
    /// (module doc, "Class-wise trust"); empty before the first fit.
    class_loo: Vec<ClassLoo>,
    /// A failed audit's teacher − model gap: untrusted until the next refit.
    audit_failed: Option<f64>,
    /// A failed audit's worst force residual and its bound, when the forces failed it.
    audit_force: Option<(f64, f64)>,
    /// Atoms of the geometry whose failed audit set `audit_failed`: the checklist's per-atom
    /// value |gap|/N (activity plan review M1). A readout only; it decides nothing.
    audit_atoms: usize,
    /// Per atom: the neighbour count within r_cut when the variance was last evaluated (a change
    /// forces a fresh verdict: module doc, "Variance cadence").
    verdict_counts: Vec<usize>,
    epoch: u64,
    pending_refit: bool,
    pending_jump: Option<f64>,
    last: Option<LastEval>,
    /// Consultations since the variance was last evaluated (the cadence, module doc).
    consults: usize,
    last_variance: Option<VarianceVerdict>,
    /// Variance evaluations so far (readout; the cadence test).
    variance_checks: u64,
    /// Steps the model answered or would have answered (the audit counter).
    answered: usize,
    /// Per stored sample: quarantined (module doc, "Trust recovery"); shorter than the store
    /// means "not quarantined" for the rest.
    quarantine: Vec<u8>,
    /// The body-3 degree cap a recovery settled on (applied to every rebuilt basis).
    degree_cap: Option<usize>,
    /// Thinned candidates since the last kept fallback sample (thinning; the no-model cap).
    since_kept: usize,
    /// Per stored sample: its atoms' (species index, descriptor) under `desc_basis`.
    desc_cache: Vec<Vec<(usize, Vec<f64>)>>,
    desc_basis: Option<Basis>,
    /// What the last refit's recovery did, for the readouts.
    recovery: Option<String>,
    /// The background auditor (ruling S14), when one was attached.
    auditor: Option<Auditor>,
    /// An audit fell due while the auditor was busy: dispatched at the next answered step.
    audit_due: bool,
    /// A failed asynchronous audit untrusted the model: the next teacher evaluation books the
    /// Model → Teacher flip at its positions (module doc, "Audit").
    audit_flip: bool,
    last_audit: Option<AuditRecord>,
    /// Fits installed so far (`fit_store`'s successes and installed sidecars): the fit generation
    /// that `last_audit_fit` is compared with. A readout only; it decides nothing.
    fits: u64,
    /// The model generation (activity Task 4 review I2): bumped wherever what the training
    /// views' curves read past the gate changes — every assignment of the model (a fit, an
    /// installed sidecar, a restored state, a forget) and a policy change — including a refit that
    /// leaves the model untrusted, which the surface `epoch` does not see. The views' cache key
    /// and their age line; it decides nothing.
    model_gen: u64,
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
    /// The next `evaluate` goes to the teacher whatever the model says (the relax verification,
    /// `force_teacher_next`); consumed by that evaluation.
    force_teacher: bool,
    /// The latest `evaluate` was the stepping teacher's and nothing has used the teacher or moved
    /// the nuclei since: `sample_density` delegates (module doc, "Density samples").
    last_fallback: bool,
    /// Side jobs (audits and refreshes) dispatched so far, the newest side density, and the
    /// isolated-atom densities that arrived (module doc, "Side densities").
    side_jobs: u64,
    side_display: Option<SideDisplay>,
    atom_densities: Vec<(u8, Display)>,
    /// The isolated-atom references that came back (`request_atom_reference`), not yet taken.
    atom_references: Vec<(u8, Result<AtomReference, ProviderError>)>,
    /// Side jobs waiting for the auditor (scene, provenance, kind, side-job number).
    side_queue: std::collections::VecDeque<(Scene, Provenance, JobKind, u64)>,
    /// An audit answer taken outside `evaluate`, judged at the next one (review M7).
    held_audit: Option<AuditAnswer>,
    /// Per stored sample: the teacher's face density (NaN = not known: `add_sample`, a teacher
    /// that reports none, or a file saved without densities — which `save_sidecar` writes only
    /// when no stored sample is box-affected).
    face: Vec<f64>,
    /// Fence voting (module doc): the last disagreeing answers inside a fence, and how many
    /// samples each fenced sample has had kept by a vote.
    fence_votes: Vec<FenceVote>,
    fence_kept: BTreeMap<usize, u32>,
    /// Samples a vote kept that the next refit has not judged yet: (store index, the fenced
    /// samples it was kept next to). While one is pending its region takes no further vote.
    vote_pending: Vec<(usize, Vec<usize>)>,
    /// Consecutive consultations of a trusted model that fell back, with the last rule and gap.
    streak: usize,
    streak_reason: Option<Refusal>,
    streak_gap: Option<f64>,
    /// The trailing run of those fallbacks whose teacher answer was box-affected (never
    /// trained on), and the largest face density in it (`stuck`).
    streak_box: usize,
    streak_box_face: f64,
    /// The orbital view, for the background auditor's teacher only (module doc, "Side
    /// densities": the one orbital producer).
    orbital_view: OrbitalViewRequest,
    /// The driver's display interrupt (`ForceProvider::set_display_interrupt`), sent with every
    /// background job so the auditor's teacher's unoccupied eigensolve yields to the user too
    /// (module doc, "Orbitals"; Task 4 re-review N1).
    display_interrupt: Option<sim::DisplayInterrupt>,
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
    /// The Activity view's live probe (`with_probe`): the refit phase is reported on it.
    probe: Option<Arc<sim::ActivityProbe>>,
    /// One event per teaching solve and the lifetime counts (`ledger`, activity spec A2–A3).
    ledger: Ledger,
}

impl SurrogateKs {
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
            active_set: None,
            training_residuals: None,
            class_loo: Vec::new(),
            audit_failed: None,
            audit_force: None,
            audit_atoms: 0,
            verdict_counts: Vec::new(),
            epoch: 0,
            pending_refit: false,
            pending_jump: None,
            last: None,
            consults: 0,
            last_variance: None,
            variance_checks: 0,
            answered: 0,
            quarantine: Vec::new(),
            degree_cap: None,
            since_kept: THIN_EVERY,
            desc_cache: Vec::new(),
            desc_basis: None,
            recovery: None,
            auditor: None,
            audit_due: false,
            audit_flip: false,
            force_teacher: false,
            last_fallback: false,
            last_audit: None,
            fits: 0,
            last_audit_fit: 0,
            model_gen: 0,
            fit_members: (0, Vec::new()),
            curve_cache: None,
            pending_notes: Vec::new(),
            side_jobs: 0,
            side_display: None,
            atom_densities: Vec::new(),
            atom_references: Vec::new(),
            side_queue: std::collections::VecDeque::new(),
            held_audit: None,
            face: Vec::new(),
            fence_votes: Vec::new(),
            fence_kept: BTreeMap::new(),
            vote_pending: Vec::new(),
            streak: 0,
            streak_reason: None,
            streak_gap: None,
            streak_box: 0,
            streak_box_face: 0.0,
            orbital_view: OrbitalViewRequest::default(),
            display_interrupt: None,
            pass: None,
            last_pass: None,
            ref_cache: AtomReferences::default(),
            user_notes: Vec::new(),
            probe: None,
            ledger: Ledger::default(),
        }
    }

    /// Records the teacher's SCF tolerance in the samples' provenance (module doc).
    pub fn with_teacher_tol(mut self, tol: f64) -> SurrogateKs {
        self.teacher_tol = tol;
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

    /// How many times the variance (rules 6–7) has been evaluated.
    pub fn variance_checks(&self) -> u64 {
        self.variance_checks
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

    /// The teacher's step: evaluate, keep the sample, build the result on the right surface
    /// (module doc, "One-surface accounting" and "Audit"), and leave one activity event (`ledger`):
    /// the rule that sent the step here, what `admit` did with the answer, or `Failed` when the
    /// teacher failed (a cancel leaves none — the step is retried).
    fn teach(&mut self, scene: &Scene, h: Handoff, t0: Instant, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        let solve_started = Instant::now();
        let r = match self.teacher.evaluate(scene, should_abort) {
            Ok(r) => r,
            Err(e) => {
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
        // the event's record of the solve (read before `r` and `h` are consumed below)
        let (teacher_ms, scf) = (r.stats.solve_ms as f64, r.stats.scf_steps);
        let (h_audit, rule_code, atom, n_atoms) = (h.audit, h.reason.rule, h.reason.atom, h.pos.len());
        let gap = h.model.as_ref().map(|m| r.energy - m.energy);
        let mut notes = Vec::new();
        let face = r.stats.face_density;
        let model_answer = h.model.as_ref().map(|m| (m.energy, &m.forces[..]));
        let essential = h.model.is_none() && h.keep_sample && !h.thin;
        let rule = box_affected_face_density(h.prov.h);
        let outcome = self.admit_and_keep(&h.pos, &h.z, r.energy, &r.forces, face, model_answer, &h.near, h.keep_sample, h.thin && !h.audit, essential, &h.prov, &mut notes);
        let n = h.pos.len().max(1) as f64;
        let mut stats = r.stats;
        stats.source = Source::Fallback;
        stats.uncertainty = h.model.as_ref().and_then(|m| m.uncertainty);
        // a consulted model that fell back (an audit is not the model failing to answer)
        let boxed = face.filter(|&rho| rho > rule);
        if h.model.is_some() && !h.audit {
            self.streak += 1;
            self.streak_reason = Some(h.reason.clone());
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
                let force_fail = worst_force_residual(&self.policy, &m.forces, &r.forces).filter(|(d, b)| !(d <= b));
                let audit_fails = box_rho.is_none() && (!(gap.abs() / n <= self.policy.energy_std_per_atom) || force_fail.is_some());
                if h.audit {
                    let verdict = if box_rho.is_some() {
                        AuditVerdict::BoxAffected
                    } else if audit_fails {
                        AuditVerdict::Failed
                    } else {
                        AuditVerdict::Passed
                    };
                    self.last_audit = Some(AuditRecord { gap, verdict, face_density: face, unaudited_steps: 0, asynchronous: false, teacher_ms: stats.solve_ms as f64 });
                    self.last_audit_fit = self.fits;
                    if let Some(rho) = box_rho {
                        notes.push(format!(
                            "surrogate audit not judged: the teacher's answer is box-affected (face density {rho:.1e} > {rule:.1e}) — its gap is the box's confinement as much as the model's error"
                        ));
                    }
                }
                if h.audit && audit_fails {
                    // failed audit: this step is already on the teacher surface, and the jump
                    // of the reported energy at these positions is the gap itself
                    self.audit_failed = Some(gap);
                    self.audit_force = force_fail;
                    self.audit_atoms = n as usize;
                    // on the refresh right after a refit the refit's own jump (E_new − E_reported
                    // at these positions) is still pending: the reported energy moves by both.
                    // (From a teacher-surface step there is nothing to book: `jump_to_teacher`.)
                    stats.refit_jump = self.jump_to_teacher(gap);
                    notes.insert(0, match force_fail {
                        Some((d, b)) if gap.abs() / n <= self.policy.energy_std_per_atom => format!(
                            "surrogate audit failed on forces: worst atom off by {d:.1e} > {b:.1e} Ha/Bohr (teacher − model {gap:+.2e} Ha) — untrusted until the next refit"
                        ),
                        _ => format!(
                            "surrogate audit failed: teacher − model {gap:+.2e} Ha ({:.1e} > {:.1e} Ha/atom) — untrusted until the next refit",
                            gap.abs() / n, self.policy.energy_std_per_atom
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
                    .filter(|m| flip && m.provenance == h.prov && m.basis.species_indices(&h.z).is_ok())
                    .and_then(|m| m.predict(&h.pos, &h.z, false).ok());
                match at_model {
                    Some(p) => stats.refit_jump = Some(self.pending_jump.take().unwrap_or(0.0) + r.energy - p.energy),
                    None => self.pending_jump = None,
                }
                notes.insert(0, format!("{} — teacher answered", h.reason.text));
                (r.energy, false)
            }
        };
        stats.solve_ms = t0.elapsed().as_millis() as u64;
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
        ev.audit = if h_audit { self.last_audit.as_ref().map(|a| a.verdict) } else { None };
        self.push_event(ev);
        let surface_forces = surface_forces.unwrap_or_else(|| r.forces.clone());
        self.last = Some(LastEval { pos: h.pos, z: h.z, energy, on_model, provenance: h.prov, flip_back, surface_forces });
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
        // a background audit's verdict, if it arrived (module doc, "Audit")
        self.poll_audit();
        let forced = std::mem::take(&mut self.force_teacher);
        // nothing reported since an edit (or ever): a fallback starts the history on the teacher
        let post_edit = self.last.is_none();
        let prov = self.scene_provenance(scene);
        let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        let z: Vec<u8> = scene.nuclei.iter().map(|n| n.z).collect();
        // rules 1–4 (draft mode answers past rule 4 only: module doc, "Draft mode")
        let draft = self.policy.never_fallback;
        let mut untrusted = None;
        match self.gate(&z, &prov) {
            Err(Gate { reason, untrusted: true, .. }) if draft => untrusted = Some(reason.text),
            Err(Gate { reason, keep_sample, thin, .. }) => {
                // a quarantined neighbourhood is never trained on again (module doc)
                let near = if keep_sample && self.model.is_some() { self.near_quarantined(&pos, &z) } else { Vec::new() };
                let h = Handoff { prov, pos, z, reason, keep_sample, thin, near, model: None, audit: false, teacher_surface: false };
                return self.teach(scene, h, t0, should_abort);
            }
            Ok(()) => {}
        }
        // rules 5–8: consult the model; the variance on the cadence, the envelope every time
        // the variance cadence (module doc): a fresh check after an edit, a refit, a policy change
        // or a rejection, else at least every k-th consultation — an accepting verdict is reused
        // at most k − 1 times, a rejecting one never
        // a neighbour entering or leaving r_cut is a new environment for the verdict (M6 Task 11)
        let counts = {
            let rc2 = self.model.as_ref().map_or(0.0, |m| m.basis.r_cut * m.basis.r_cut);
            neighbour_counts(&pos, rc2)
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
        let pred = model
            .predict(&pos, &z, with_variance)
            .map_err(|e| ProviderError::Invalid(format!("surrogate: {e}")))?;
        // a prediction that is not finite is never an answer, draft mode and the relax
        // verification included (activity Task 3b): the teacher answers on its own surface, the
        // answer is kept, nothing is audited (the gap is not a number), the switch is booked
        // along the forces (`transition_jump`, never through the model's energy)
        if let Some(reason) = non_finite_answer(&pred) {
            let near = if self.model.is_some() { self.near_quarantined(&pos, &z) } else { Vec::new() };
            let h = Handoff {
                prov,
                pos,
                z,
                reason,
                keep_sample: true,
                thin: false,
                near,
                model: Some(ModelAnswer { energy: pred.energy, forces: pred.forces, uncertainty: None }),
                audit: false,
                teacher_surface: true,
            };
            return self.teach(scene, h, t0, should_abort);
        }
        // rule 6 on the epistemic force std (σ_f excluded, as rule 7 excludes the energy floor)
        let variance = match (pred.energy_std, pred.force_std_epistemic.as_ref()) {
            (Some(e), Some(f)) => {
                let mut v = self.variance_check(model, &pred, e, f);
                // rule 5d with the variance (the same cadence; a rejection is never reused)
                if let Some((nu, at)) = self.null_space.as_ref().and_then(|ns| ns.novelty(&pos, &z).ok()).filter(|(nu, _)| *nu > 1.0) {
                    v.accept = false;
                    v.novel = true;
                    v.reason = Some(Refusal::new(
                        format!(
                            "surrogate: new environment (feature never seen in training) — {} {nu:.1e}× farther into directions no training row spans than rounding allows",
                            at.map_or("the energy row reaches".to_string(), |j| format!("atom {j}'s force rows reach"))
                        ),
                        TeachRule::NewEnvironment,
                        at,
                    ));
                }
                self.verdict_counts = counts.clone();
                self.last_variance = Some(v.clone());
                v
            }
            _ => self.last_variance.clone().expect("a variance verdict exists when none is evaluated"),
        };
        let uncertainty = Some(variance.uncertainty);
        let atom_uncertainty = Some(variance.atoms.clone());
        let (atom_envelope, envelope) = self.envelope_outside(model, &pos, &z);
        // rule 5 includes 5d (an unspanned direction): outside what the model has seen
        let by_envelope = envelope.is_some() || (variance.novel && !variance.accept);
        // rule 4b, the class rule (module doc, "Class-wise trust"): the query's environment class
        // must be trusted by the fit's leave-one-out errors. When rule 5 fires as well its note
        // (which atom, which feature) is the more specific one and is shown; the variance's
        // (6–7) yields to the class's
        let class = self.class_untrusted(&counts);
        let class_short = class.as_ref().is_some_and(|c| c.1);
        let by_class = class.is_some();
        let novel = (variance.novel && !variance.accept).then(|| variance.reason.clone()).flatten();
        let mut reason = envelope
            .or(novel)
            .or(class.map(|c| c.0))
            .or_else(|| if variance.accept { None } else { variance.reason.clone() });
        // a quarantined geometry is outside what the model may answer (ruling S15), whichever
        // rule fired first; its answer is kept only if it agrees with the model (teach)
        let near = self.near_quarantined(&pos, &z);
        if reason.is_none() && !near.is_empty() {
            reason = Some(Refusal::new(
                format!(
                    "surrogate: within {:.2} training spreads of a quarantined geometry the model cannot represent",
                    QUARANTINE_FRACTION * self.policy.max_descriptor_distance
                ),
                TeachRule::NearQuarantine,
                None,
            ));
        }
        let due = (reason.is_none() || draft) && self.policy.audit_every > 0 && {
            self.answered += 1;
            self.answered % self.policy.audit_every == 0 || self.audit_due
        };
        // asynchronous when an auditor is attached (the model answers this step), else in the step
        // — never in draft mode, where every step is the model's (module doc, "Draft mode")
        let audit = due && !self.dispatch_audit(scene, &prov) && !draft;
        // draft mode: what the model guesses beyond, and it answers anyway
        let extrapolation = (draft && (untrusted.is_some() || reason.is_some())).then(|| Extrapolation {
            atoms_outside: atom_envelope.iter().filter(|&&d| d > 1.0).count(),
            untrusted: untrusted.clone(),
            reason: reason.as_ref().map(|r| r.text.clone()),
            atom_envelope: atom_envelope.clone(),
        });
        if draft {
            reason = None;
        }
        // the relax verification (`force_teacher_next`): the teacher answers whatever the rules
        // and draft mode say, and the answer is an essential sample (never thinned)
        if forced {
            reason = Some(Refusal::new("surrogate: relax verification — the teacher checks the relaxed geometry", TeachRule::RelaxVerification, None));
        }
        if reason.is_some() || audit {
            let reason = reason.unwrap_or_else(|| Refusal::new(format!("surrogate audit (every {} answered steps)", self.policy.audit_every), TeachRule::Audit, None));
            let h = Handoff {
                prov,
                pos,
                z,
                reason,
                keep_sample: true,
                // the variance rules' fallbacks are the answers active learning asks for, and
                // so are a class's while it is short of samples (rule 4b)
                thin: (by_envelope || by_class) && !class_short && !forced,
                near,
                model: Some(ModelAnswer { energy: pred.energy, forces: pred.forces, uncertainty }),
                audit,
                // rules 4b and 5 (the model's energy is an unvalidated guess or an extrapolation)
                // and the first evaluation after an edit report the teacher's surface; audits
                // and the relax verification keep theirs (module doc, "One-surface energy
                // accounting")
                teacher_surface: !audit && !forced && (by_class || by_envelope || post_edit),
            };
            return self.teach(scene, h, t0, should_abort);
        }
        let mut stats = EvalStats::default();
        stats.source = Source::Surrogate;
        stats.uncertainty = uncertainty;
        stats.atom_uncertainty = atom_uncertainty;
        stats.extrapolation = extrapolation;
        stats.refit_jump = self.jump_back(pred.energy, &pred.forces, &pos);
        stats.energy.total = pred.energy;
        stats.solve_ms = t0.elapsed().as_millis() as u64;
        stats.notes.append(&mut self.pending_notes);
        self.streak = 0;
        self.streak_box = 0;
        self.streak_box_face = 0.0;
        self.last = Some(LastEval { pos, z, energy: pred.energy, on_model: true, provenance: prov, flip_back: false, surface_forces: pred.forces.clone() });
        Ok(ForceResult { energy: pred.energy, forces: pred.forces, stats, display: None, orbitals: None })
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn reset(&mut self) {
        self.teacher.reset();
        self.last_fallback = false;
        self.last = None;
        self.last_variance = None; // the next consultation checks the variance afresh
    }

    fn discontinuity(&mut self) {
        self.teacher.discontinuity();
        self.last_fallback = false;
        self.last = None;
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

    fn take_side_display(&mut self) -> Option<SideDisplay> {
        self.poll_side_jobs();
        self.side_display.take()
    }

    /// The teacher's side solver (a fresh instance): the references are its surface's, never the
    /// model's and never the stepping teacher's warm state. Used by the driver only when there is
    /// no background auditor (the CPU teacher); with one, the references go to its job slot
    /// (`request_atom_reference`, module doc "Side densities").
    fn side_solver(&self) -> Option<Box<dyn ForceProvider>> {
        self.teacher.side_solver()
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
}

// ---- the Trainable face (M6 Task 1): what the driver and the worker reach through
// `ForceProvider::as_trainable` ----

impl SurrogateKs {
    /// After a pass (complete, cancelled or ended by a failed reference): the reference cache
    /// back, the teacher told its state is a sample geometry's, the model refitted over
    /// everything it holds — only when the samples a fit would use reach Q′ (below it the model
    /// cannot earn trust, and the fit with its trust recovery took 45–50 s for 3 H₂O samples on
    /// the worker thread inside a Cancel: the samples are kept and the refit waits; M6 Task 3
    /// fix round 2) — and the outcome noted (what `Dynamics` did before M6).
    fn finish_pass(&mut self, mut sampler: Sampler) {
        let progress = sampler.progress();
        self.ref_cache = sampler.take_references();
        self.user_notes.extend(sampler.take_notes());
        self.last_pass = Some(progress.clone());
        let added = sampler.samples_added();
        let how = if progress.cancelled { "cancelled" } else { "finished" };
        self.teacher.discontinuity();
        let (fittable, q_prime) = self.fittable_and_q_prime();
        let msg = if self.samples() == 0 {
            format!("sampling pass {how}: no samples to fit")
        } else if fittable < q_prime {
            // below Q′ the model cannot earn trust (the trust rule needs n ≥ Q′), and the fit
            // with its trust recovery costs seconds on the worker thread (Task 3 fix round 2):
            // keep the samples, skip the refit until there are enough
            format!("sampling pass {how}: {fittable} samples kept — not enough to fit (Q′ = {q_prime}); sample more to train")
        } else {
            match SurrogateKs::refit(self) {
                Ok(r) => format!(
                    "sampling pass {how}: {added} samples added ({} skipped, {} failed); refit on {} samples, rmse {:.1e} Ha/atom, {:.1e} Ha/Bohr — {}",
                    progress.skipped,
                    progress.failed,
                    r.n_samples,
                    r.rmse_energy_per_atom,
                    r.rmse_force,
                    match self.untrusted_reason() {
                        None => "trusted".to_string(),
                        Some(why) => format!("untrusted ({why})"),
                    }
                ),
                Err(e) => format!("sampling pass {how}: {added} samples added; the refit failed ({e})"),
            }
        };
        self.user_notes.push(msg);
    }

    /// The stored samples a fit would use (not quarantined, superseded or box-excluded) and the
    /// Q′ of the basis it would build for them (`surrogate::fit`, the independent columns; the
    /// trust rule needs n ≥ Q′). (0, 0) without samples.
    fn fittable_and_q_prime(&self) -> (usize, usize) {
        let Some(store) = self.store.as_ref() else { return (0, 0) };
        let n = (0..store.samples.len()).filter(|&i| !self.excluded(i)).count();
        let mut species: Vec<u8> = store.samples.iter().flat_map(|s| s.z.iter().copied()).collect();
        species.sort_unstable();
        species.dedup();
        let mut b = Basis::new(&species);
        if let Some(d) = self.degree_cap.filter(|_| b.body3) {
            b.degree_cap = d;
        }
        (n, b.column_classes().len())
    }

    /// The model's provenance against `scene`'s, for the readouts (`TrainableStatus`).
    fn provenance_mismatch(&self, scene: &Scene) -> Option<String> {
        let want = self.provenance_for(scene);
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

impl Trainable for SurrogateKs {
    fn start_pass(&mut self, scene: &Scene, plan: &SamplingPlan, refs: &mut AtomReferences) -> Result<(), String> {
        if self.pass.is_some() {
            return Err("a pass is already running (cancel it first)".into());
        }
        let r_cut = self.basis().map_or(Basis::new(&[]).r_cut, |b| b.r_cut);
        let sampler = Sampler::with_cutoff(scene, plan, r_cut)?;
        // the caller's references seed the cache (an entry the cache already holds stays)
        for (key, r) in refs.iter() {
            if self.ref_cache.get(key).is_none() {
                self.ref_cache.insert(key.clone(), r.clone());
            }
        }
        let sampler = sampler.with_references(std::mem::take(&mut self.ref_cache));
        let p = sampler.progress();
        self.user_notes.push(format!(
            "sampling pass started: {} jobs ({} pair scans), the dynamics are paused until it ends",
            p.total,
            sampler.scans().len()
        ));
        self.pass = Some(sampler);
        Ok(())
    }

    fn pass_step(&mut self, should_abort: &mut dyn FnMut() -> bool) -> Result<Option<SamplingProgress>, ProviderError> {
        let Some(mut sampler) = self.pass.take() else { return Ok(None) };
        self.last_fallback = false; // the teacher solves a pass geometry
        let r = sampler.step_own_teacher(self, should_abort);
        self.user_notes.extend(sampler.take_notes());
        match r {
            Ok(Some(p)) => {
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
            Err(e) => {
                self.finish_pass(sampler);
                Err(e)
            }
        }
    }

    fn cancel_pass(&mut self) {
        if let Some(mut sampler) = self.pass.take() {
            sampler.cancel();
            self.finish_pass(sampler);
        }
    }

    fn rescale_pass(&mut self, plan: &SamplingPlan) {
        if let Some(sampler) = &mut self.pass {
            sampler.rescale(plan);
        }
    }

    fn pass_active(&self) -> bool {
        self.pass.is_some()
    }

    fn progress(&self) -> Option<SamplingProgress> {
        self.pass.as_ref().map(|s| s.progress()).or_else(|| self.last_pass.clone())
    }

    fn refit(&mut self) -> Result<String, String> {
        SurrogateKs::refit(self)
            .map(|r| format!("{} samples, rmse_E {:.1e} Ha/atom, rmse_F {:.1e} Ha/Bohr", r.n_samples, r.rmse_energy_per_atom, r.rmse_force))
            .map_err(|e| e.to_string())
    }

    fn forget(&mut self) {
        // a running pass ends without a refit; its references stay cached
        if let Some(mut sampler) = self.pass.take() {
            self.teacher.discontinuity();
            self.ref_cache = sampler.take_references();
        }
        let n = self.samples();
        SurrogateKs::forget(self);
        self.last_pass = None;
        self.user_notes.push(format!("model forgotten ({n} samples dropped): the teacher answers every step"));
    }

    fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        self.save_sidecar(path)
    }

    fn load(&mut self, path: &std::path::Path, scene: &Scene) -> Result<(), String> {
        if self.pass.is_some() {
            return Err("load model: cancel the running sampling pass first".into());
        }
        let sc = surrogate::sidecar::read(path).map_err(|e| format!("load model: {e}"))?;
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
        match self.install_sidecar(sc) {
            Ok(()) => {
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
            residuals: self.training_residuals().map(|t| (t.energy_per_atom, t.force, t.force_bound)),
            force_bound: self.policy.force_bound(if fmin.is_finite() { fmin } else { 0.0 }),
            provenance_mismatch: self.provenance_mismatch(scene),
            provenance: self.model.as_ref().map(|m| m.provenance.describe_recorded()),
            audit_every: self.policy.audit_every,
            draft: self.policy.never_fallback,
            quarantined: self.quarantined(),
            superseded: self.superseded(),
            box_excluded: self.box_excluded(),
            box_affected: self.box_affected(),
            trust_floor: self.trust_floor(),
            unfitted: self.unfitted(),
            recovery: self.recovery_note().map(str::to_string),
            audits_background: self.audits_asynchronous(),
            audit_in_flight: self.audit_in_flight(),
            last_audit: self.last_audit().cloned(),
            last_audit_this_fit: self.last_audit_this_fit(),
            stuck: self.stuck(),
            class_loo: self.class_loo.clone(),
            solves: self.solve_counts(),
            trust_checks: self.trust_checks(),
            stuck_info: self.stuck_info(),
            model_gen: self.model_generation(),
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

    fn request_display(&mut self, scene: &Scene) -> bool {
        SurrogateKs::request_display(self, scene)
    }

    fn side_job_in_flight(&self) -> bool {
        self.auditor.as_ref().is_some_and(|a| a.busy.is_some()) || !self.side_queue.is_empty()
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

#[cfg(test)]
mod tests;
