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
//! 5. **Out of the training envelope**: any atom whose descriptor lies farther than
//!    `max_descriptor_distance` (3.0) training spreads outside the per-feature range the fit
//!    has seen (`surrogate::Envelope`) → teacher; sample kept. This guards the directions the
//!    variance cannot: a linear model's predictive variance is a leverage in feature space and
//!    is small wherever the features themselves are small — all body-2 features vanish as a
//!    pair separates, so a dissociating dimer would otherwise be answered "confidently" with
//!    Σ ref by a model that never saw an empty neighbourhood (envelope module doc for the unit).
//! 6. **Force uncertainty**: with `force_std[j]` the model's expected ‖δF_j‖ (Task 3's binding
//!    unit), the per-component equivalent `force_std[j]/√3` must satisfy
//!    `force_std[j]/√3 ≤ min(max(force_std_abs, force_std_rel·|F_j|), 1e-2)` for every atom
//!    (2e-3 Ha/Bohr absolute, 10 % relative, the relative bound capped at
//!    [`FORCE_STD_BOUND_MAX`] = 1e-2 Ha/Bohr so a large force never buys an unbounded error).
//!    Otherwise → teacher; sample kept.
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
//!    recorded ([`SurrogateKs::last_audit`]). If |gap|/N > `energy_std_per_atom` the audit
//!    **fails**: the model is untrusted until the next refit. A **box-affected** teacher answer
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
//! nuclei: `reset`, `discontinuity`, `teacher_mut` (the driver's reference-atom solves) and a
//! sampling pass step. The background auditor's state is never sampled.
//!
//! `uncertainty` = max_j force_std[j]/√3, the **per-component** equivalent (S1's "max per-atom
//! force std" is √3 times it); it is reported whenever the model was consulted (rules 5–9), so
//! the readouts can show how far off the model was on a fallback. Between variance checks
//! (the cadence below) it is the reused verdict's number, at most k − 1 consultations old.
//!
//! **Variance cadence** (Task 9). `variance_cadence` = k: the O(Q′²) variance (rules 6–7) is
//! evaluated afresh on the first consultation after an edit (`reset` / `discontinuity`), a
//! refit, an installed model or a policy change, after every rejection, and otherwise at least
//! on every k-th consultation; in between the last *accepting* verdict is reused (a rejecting
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
//! # Trust rule (from the Task 3 fit, binding)
//!
//! The model may answer only when the last `FitReport` has `n_samples ≥ q_prime` (distinct
//! geometries, not rows: the 3N force rows of one sample probe one geometry) AND
//! `residual_dof_e ≥ 30` AND `residual_dof_f ≥ 30` (the two noise levels are then known to
//! ≈ 25 %, so the variance means something). Below that the fit is a curve through too few
//! points with a noise estimate collapsing towards its floor — overconfident inside its span.
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
//! Thinned: the envelope rule (5) and a residual-untrusted model (rule 4).
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
//! model the result is `ForceResult { energy: E_model(r), forces: F_teacher(r) }` and
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
use sim::references::AtomReferences;
use sim::sampling::{SamplingPlan, SamplingProgress};
use sim::{
    Display, EvalStats, Extrapolation, ForceProvider, ForceResult, ProviderError, Scene, SideDisplay, SideKind,
    Source, Trainable, TrainableStatus,
};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Instant;
use surrogate::{
    describe_values, Accumulator, Basis, Envelope, FitError, FitOptions, FitReport, Model, Provenance,
    Sample, SampleStore, SampleTag, StoreError,
};

mod audit;
mod policy;
mod quarantine;
mod store;
mod trust;

use audit::*;
use policy::*;
use quarantine::*;
pub use audit::{AuditRecord, AuditVerdict};
pub use policy::{box_affected_face_density, FallbackPolicy, BOX_AFFECTED_FACE_DENSITY, FORCE_STD_BOUND_MAX};
pub use quarantine::{
    FENCE_VOTES, FENCE_VOTE_LIMIT, NO_MODEL_SAMPLE_FACTOR, QUARANTINE_FRACTION, QUARANTINE_MAX_FRACTION, QUARANTINE_ROUNDS,
    RECOVERY_DEGREE_CAP, THIN_EVERY, THIN_FRACTION,
};
pub use trust::{TrainingResiduals, MIN_RESIDUAL_DOF, STUCK_STEPS};

/// What the last evaluate reported, for the refit jump.
struct LastEval {
    pos: Vec<[f64; 3]>,
    z: Vec<u8>,
    energy: f64,
    on_model: bool,
    provenance: Provenance,
}

/// The model's answer at the geometry of a teacher step (rules 5–7 and audits).
struct ModelAnswer {
    energy: f64,
    forces: Vec<[f64; 3]>,
    uncertainty: Option<f64>,
}

/// One teacher step: why, with what from the model, and whether it is an audit.
struct Handoff {
    prov: Provenance,
    pos: Vec<[f64; 3]>,
    z: Vec<u8>,
    reason: String,
    keep_sample: bool,
    /// The kept sample is subject to thinning (module doc).
    thin: bool,
    /// Quarantined samples within the quarantine radius of this geometry (module doc).
    near: Vec<usize>,
    model: Option<ModelAnswer>,
    audit: bool,
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
    training_residuals: Option<TrainingResiduals>,
    /// A failed audit's teacher − model gap: untrusted until the next refit.
    audit_failed: Option<f64>,
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
    streak_reason: Option<String>,
    streak_gap: Option<f64>,
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
            training_residuals: None,
            audit_failed: None,
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
            pending_notes: Vec::new(),
            side_jobs: 0,
            side_display: None,
            atom_densities: Vec::new(),
            side_queue: std::collections::VecDeque::new(),
            held_audit: None,
            face: Vec::new(),
            fence_votes: Vec::new(),
            fence_kept: BTreeMap::new(),
            vote_pending: Vec::new(),
            streak: 0,
            streak_reason: None,
            streak_gap: None,
            pass: None,
            last_pass: None,
            ref_cache: AtomReferences::default(),
            user_notes: Vec::new(),
        }
    }

    /// Records the teacher's SCF tolerance in the samples' provenance (module doc).
    pub fn with_teacher_tol(mut self, tol: f64) -> SurrogateKs {
        self.teacher_tol = tol;
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
    /// (module doc, "One-surface accounting" and "Audit").
    fn teach(&mut self, scene: &Scene, h: Handoff, t0: Instant, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        let r = self.teacher.evaluate(scene, should_abort)?;
        let mut notes = Vec::new();
        let face = r.stats.face_density;
        let model_answer = h.model.as_ref().map(|m| (m.energy, &m.forces[..]));
        let essential = h.model.is_none() && h.keep_sample && !h.thin;
        let rule = box_affected_face_density(h.prov.h);
        if self.admit(&h.pos, &h.z, r.energy, &r.forces, face, model_answer, &h.near, h.keep_sample, h.thin && !h.audit, essential, h.prov.h, &mut notes) {
            self.keep_sample(&h.pos, &h.z, r.energy, &r.forces, face, &h.prov, &mut notes);
        }
        let n = h.pos.len().max(1) as f64;
        let mut stats = r.stats;
        stats.source = Source::Fallback;
        stats.uncertainty = h.model.as_ref().and_then(|m| m.uncertainty);
        // a consulted model that fell back (an audit is not the model failing to answer)
        if h.model.is_some() && !h.audit {
            self.streak += 1;
            self.streak_reason = Some(h.reason.clone());
        } else if h.model.is_none() {
            self.streak = 0;
        }
        let (energy, on_model) = match h.model {
            Some(m) => {
                let gap = r.energy - m.energy;
                self.streak_gap = Some(gap);
                let box_rho = face.filter(|&rho| rho > rule);
                if h.audit {
                    let verdict = if box_rho.is_some() {
                        AuditVerdict::BoxAffected
                    } else if gap.abs() / n > self.policy.energy_std_per_atom {
                        AuditVerdict::Failed
                    } else {
                        AuditVerdict::Passed
                    };
                    self.last_audit = Some(AuditRecord { gap, verdict, face_density: face, unaudited_steps: 0, asynchronous: false, teacher_ms: stats.solve_ms as f64 });
                    if let Some(rho) = box_rho {
                        notes.push(format!(
                            "surrogate audit not judged: the teacher's answer is box-affected (face density {rho:.1e} > {rule:.1e}) — its gap is the box's confinement as much as the model's error"
                        ));
                    }
                }
                if h.audit && box_rho.is_none() && gap.abs() / n > self.policy.energy_std_per_atom {
                    // failed audit: this step is already on the teacher surface, and the jump
                    // of the reported energy at these positions is the gap itself
                    self.audit_failed = Some(gap);
                    // on the refresh right after a refit the refit's own jump (E_new − E_reported
                    // at these positions) is still pending: the reported energy moves by both
                    stats.refit_jump = Some(self.pending_jump.take().unwrap_or(0.0) + gap);
                    notes.insert(0, format!(
                        "surrogate audit failed: teacher − model {gap:+.2e} Ha ({:.1e} > {:.1e} Ha/atom) — untrusted until the next refit",
                        gap.abs() / n, self.policy.energy_std_per_atom
                    ));
                    (r.energy, false)
                } else {
                    stats.teacher_gap = Some(gap);
                    stats.model_forces = Some(m.forces);
                    stats.refit_jump = self.pending_jump.take();
                    notes.insert(0, format!("{} — teacher answered (teacher − model {gap:+.2e} Ha)", h.reason));
                    (m.energy, true)
                }
            }
            None => {
                // a failed background audit untrusted the model since the last evaluation, which
                // was on the model surface: the reported energy moves from the model's to the
                // teacher's at THESE positions — book that (plus a refit's pending jump), exactly
                // as a failed audit in the step does
                let flip = std::mem::take(&mut self.audit_flip);
                let e_model = self
                    .model
                    .as_ref()
                    .filter(|m| flip && m.provenance == h.prov && m.basis.species_indices(&h.z).is_ok())
                    .and_then(|m| m.predict(&h.pos, &h.z, false).ok())
                    .map(|p| p.energy);
                match e_model {
                    Some(e) => stats.refit_jump = Some(self.pending_jump.take().unwrap_or(0.0) + r.energy - e),
                    None => self.pending_jump = None,
                }
                notes.insert(0, format!("{} — teacher answered", h.reason));
                (r.energy, false)
            }
        };
        stats.solve_ms = t0.elapsed().as_millis() as u64;
        stats.notes.extend(notes);
        stats.notes.append(&mut self.pending_notes);
        self.last = Some(LastEval { pos: h.pos, z: h.z, energy, on_model, provenance: h.prov });
        self.last_fallback = true;
        Ok(ForceResult { energy, forces: r.forces, stats, display: r.display })
    }
}

impl ForceProvider for SurrogateKs {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        let t0 = Instant::now();
        // set again only by a teacher answer (`teach`)
        self.last_fallback = false;
        // a live evaluation between a pass's sample solves (review m2): neither may extrapolate
        // the teacher's density through the other's geometry
        if let Some(pass) = self.pass.as_mut() {
            self.teacher.discontinuity();
            pass.teacher_used_elsewhere();
        }
        // a background audit's verdict, if it arrived (module doc, "Audit")
        self.poll_audit();
        let forced = std::mem::take(&mut self.force_teacher);
        let prov = self.scene_provenance(scene);
        let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        let z: Vec<u8> = scene.nuclei.iter().map(|n| n.z).collect();
        // rules 1–4 (draft mode answers past rule 4 only: module doc, "Draft mode")
        let draft = self.policy.never_fallback;
        let mut untrusted = None;
        match self.gate(&z, &prov) {
            Err(Gate { reason, untrusted: true, .. }) if draft => untrusted = Some(reason),
            Err(Gate { reason, keep_sample, thin, .. }) => {
                // a quarantined neighbourhood is never trained on again (module doc)
                let near = if keep_sample && self.model.is_some() { self.near_quarantined(&pos, &z) } else { Vec::new() };
                let h = Handoff { prov, pos, z, reason, keep_sample, thin, near, model: None, audit: false };
                return self.teach(scene, h, t0, should_abort);
            }
            Ok(()) => {}
        }
        // rules 5–8: consult the model; the variance on the cadence, the envelope every time
        // the variance cadence (module doc): a fresh check after an edit, a refit, a policy change
        // or a rejection, else at least every k-th consultation — an accepting verdict is reused
        // at most k − 1 times, a rejecting one never
        let with_variance = {
            let k = self.policy.variance_cadence.max(1);
            let fresh = self.last_variance.as_ref().is_none_or(|v| !v.accept) || self.consults + 1 >= k;
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
        let variance = match (pred.energy_std, pred.force_std.as_ref()) {
            (Some(e), Some(f)) => {
                let v = self.variance_check(model, &pred, e, f);
                self.last_variance = Some(v.clone());
                v
            }
            _ => self.last_variance.clone().expect("a variance verdict exists when none is evaluated"),
        };
        let uncertainty = Some(variance.uncertainty);
        let atom_uncertainty = Some(variance.atoms.clone());
        let (atom_envelope, envelope) = self.envelope_outside(model, &pos, &z);
        let by_envelope = envelope.is_some();
        let mut reason = envelope.or_else(|| if variance.accept { None } else { variance.reason.clone() });
        // a quarantined geometry is outside what the model may answer (ruling S15), whichever
        // rule fired first; its answer is kept only if it agrees with the model (teach)
        let near = self.near_quarantined(&pos, &z);
        if reason.is_none() && !near.is_empty() {
            reason = Some(format!(
                "surrogate: within {:.2} training spreads of a quarantined geometry the model cannot represent",
                QUARANTINE_FRACTION * self.policy.max_descriptor_distance
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
            reason: reason.clone(),
            atom_envelope: atom_envelope.clone(),
        });
        if draft {
            reason = None;
        }
        // the relax verification (`force_teacher_next`): the teacher answers whatever the rules
        // and draft mode say, and the answer is an essential sample (never thinned)
        if forced {
            reason = Some("surrogate: relax verification — the teacher checks the relaxed geometry".into());
        }
        if reason.is_some() || audit {
            let reason = reason.unwrap_or_else(|| format!("surrogate audit (every {} answered steps)", self.policy.audit_every));
            let h = Handoff {
                prov,
                pos,
                z,
                reason,
                keep_sample: true,
                // the variance rules' fallbacks are the answers active learning asks for
                thin: by_envelope && !forced,
                near,
                model: Some(ModelAnswer { energy: pred.energy, forces: pred.forces, uncertainty }),
                audit,
            };
            return self.teach(scene, h, t0, should_abort);
        }
        let mut stats = EvalStats::default();
        stats.source = Source::Surrogate;
        stats.uncertainty = uncertainty;
        stats.atom_uncertainty = atom_uncertainty;
        stats.extrapolation = extrapolation;
        stats.refit_jump = self.pending_jump.take();
        stats.energy.total = pred.energy;
        stats.solve_ms = t0.elapsed().as_millis() as u64;
        stats.notes.append(&mut self.pending_notes);
        self.streak = 0;
        self.last = Some(LastEval { pos, z, energy: pred.energy, on_model: true, provenance: prov });
        Ok(ForceResult { energy: pred.energy, forces: pred.forces, stats, display: None })
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

    /// The stepping teacher's density after a fallback, else `None` (module doc, "Density samples").
    fn sample_density(&mut self, points: &[[f64; 3]]) -> Option<Vec<f64>> {
        if self.last_fallback { self.teacher.sample_density(points) } else { None }
    }
}

// ---- the Trainable face (M6 Task 1): what the driver and the worker reach through
// `ForceProvider::as_trainable` ----

impl SurrogateKs {
    /// After a pass (complete, cancelled or ended by a failed reference): the reference cache
    /// back, the teacher told its state is a sample geometry's, the model refitted over
    /// everything it holds, and the outcome noted (what `Dynamics` did before M6).
    fn finish_pass(&mut self, mut sampler: Sampler) {
        let progress = sampler.progress();
        self.ref_cache = sampler.take_references();
        self.user_notes.extend(sampler.take_notes());
        self.last_pass = Some(progress.clone());
        let added = sampler.samples_added();
        let how = if progress.cancelled { "cancelled" } else { "finished" };
        self.teacher.discontinuity();
        let msg = if self.samples() == 0 {
            format!("sampling pass {how}: no samples to fit")
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
            audit_every: self.policy.audit_every,
            draft: self.policy.never_fallback,
            quarantined: self.quarantined(),
            recovery: self.recovery_note().map(str::to_string),
            audits_background: self.audits_asynchronous(),
            audit_in_flight: self.audit_in_flight(),
            last_audit: self.last_audit().cloned(),
            stuck: self.stuck(),
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

    fn request_display(&mut self, scene: &Scene) -> bool {
        SurrogateKs::request_display(self, scene)
    }

    fn request_atom_density(&mut self, z: u8, scene: &Scene) -> bool {
        SurrogateKs::request_atom_density(self, z, scene)
    }

    fn take_atom_densities(&mut self) -> Vec<(u8, Display)> {
        SurrogateKs::take_atom_densities(self)
    }
}

#[cfg(test)]
mod tests;
