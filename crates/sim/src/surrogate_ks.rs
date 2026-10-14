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

use crate::provider::{Display, EvalStats, Extrapolation, ForceProvider, ForceResult, ProviderError, SideDisplay, SideKind, Source};
use crate::scene::Scene;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Instant;
use surrogate::{
    describe_values, Accumulator, Basis, Envelope, FitError, FitOptions, FitReport, Model, Provenance,
    Sample, SampleStore, SampleTag, StoreError,
};

/// The relative force criterion's bound never exceeds this (Ha/Bohr).
pub const FORCE_STD_BOUND_MAX: f64 = 1e-2;
/// Trust rule: both residual degrees of freedom of the fit must reach this.
pub const MIN_RESIDUAL_DOF: f64 = 30.0;
/// Fallback thinning and the quarantine neighbourhood, as a fraction of
/// `FallbackPolicy::max_descriptor_distance` (module doc, "Trust recovery").
pub const THIN_FRACTION: f64 = 0.25;
/// One in this many thinned candidates (fallback answers close to a stored sample) is kept
/// anyway (module doc, "Fallback thinning"; also the no-model cap's keep rate).
pub const THIN_EVERY: usize = 10;
/// Without a model, teacher answers are kept unthinned up to this many times Q′ (the default
/// basis of the held species) samples, then one in [`THIN_EVERY`] (module doc, "No model yet").
pub const NO_MODEL_SAMPLE_FACTOR: usize = 4;
/// A geometry within this fraction of `max_descriptor_distance` (training-spread units) of a
/// quarantined sample falls back (module doc): small — the quarantined geometry itself and its
/// immediate neighbourhood, not the region around it the model still represents. A geometric
/// choice, not a measured one (no measurement sets it; `references.md` policy table says so).
pub const QUARANTINE_FRACTION: f64 = 0.1;
/// The body-3 degree cap the first recovery step retries with (the bent-molecule setting).
pub const RECOVERY_DEGREE_CAP: usize = 6;
/// The box rule at h = 0.25 Bohr (e/Bohr³): a teacher answer whose reported face density
/// (`EvalStats::face_density`: max ρ on the outermost grid plane, one spacing h inside the
/// Dirichlet wall) exceeds [`box_affected_face_density`]`(h)` = this × (h / 0.25 Bohr)² is not
/// trained on and not judged as an audit (module doc, "Box-affected answers"). A calibration
/// at h = 0.25 for hydrogen, not a universal scale — see below.
///
/// Why this scale (Task 9; derivation by the Task 9 review, ruling S17). A hard wall against an
/// exponential tail ψ ∝ e^{−κx} (κ = √(−2ε_HOMO)) raises the energy by ΔE = κ ρ₀ per unit area,
/// ρ₀ the *unconfined* density at the wall; for a 1s-like tail at distance d the patch has an
/// effective area π d/κ, so ΔE ≈ π d ρ₀. The solver reports the plane one h inside the wall,
/// where the confined orbital is already pinched: ρ_reported ≈ 4κ²h² ρ₀. Hence
/// ΔE ≈ π d ρ_reported / (4κ²h²) ≈ 50–80 Ha per e/Bohr³ for H₂ (κ ≈ 0.85) at h = 0.25 and
/// d = 2.6–4 Bohr — the 40–55 measured (`diag_confinement_vs_face_density`, 41³ and 49³). So the
/// same face density means 4× the confinement at h = 0.125, so a quarter of the density is
/// tolerated there: the threshold scales as h² (the κ dependence — less confinement per unit
/// face density for more tightly bound species — is not modelled; an energy-relative rule with κ from the HOMO eigenvalue is a handoff item).
/// 2e-5 at h = 0.25 is ≈ 1e-3 Ha of position-dependent confinement — half the H₂ audit
/// threshold (N × 1e-3 Ha): acceptable, not tight (the review prefers 1e-5, the solver's own
/// warning level; 2e-5 keeps the 49³ hot-burst solves that warn at 1.0–1.4e-5 trainable). Fix
/// round 4 had 1e-3 (≈ 5e-2 Ha), which let the confinement through (Task 9 item 1).
pub const BOX_AFFECTED_FACE_DENSITY: f64 = 2e-5;

/// The box rule at grid spacing `h` (Bohr): [`BOX_AFFECTED_FACE_DENSITY`] × (h / 0.25)²
/// (ruling S17, direction corrected in Task 9 fix round 2; the derivation is on the constant):
/// the confinement per unit of reported face density grows as 1/h², so the tolerated density
/// shrinks as h².
pub fn box_affected_face_density(h: f64) -> f64 {
    BOX_AFFECTED_FACE_DENSITY * (h / 0.25).powi(2)
}

/// Fence voting (Task 9 fix round 1, I3): this many mutually consistent fresh teacher answers
/// inside a quarantine fence that all disagree with the model keep one of them for the fit…
pub const FENCE_VOTES: usize = 3;
/// … at most this many times per fenced sample (then the region stays fenced: the model could
/// not learn it from what it was given).
pub const FENCE_VOTE_LIMIT: u32 = 3;

/// A trusted model that has fallen back on this many consecutive consultations is reported as
/// "not answering here" with the rule that fires ([`SurrogateKs::stuck`]; module doc, "Stuck").
pub const STUCK_STEPS: usize = 50;

/// What an audit concluded (module doc, "Audit").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuditVerdict {
    /// |gap|/N within `energy_std_per_atom`.
    Passed,
    /// Beyond it: the model is untrusted until the next refit.
    Failed,
    /// The teacher's answer is box-affected (face density above [`box_affected_face_density`]):
    /// its gap carries the box's confinement, which a translation-invariant model cannot and
    /// must not learn — recorded, not judged, not kept.
    BoxAffected,
}

/// The last audit's outcome (readouts).
#[derive(Clone, Debug, PartialEq)]
pub struct AuditRecord {
    /// E_teacher − E_model at the audited geometry (Hartree; the model current when the answer
    /// was judged).
    pub gap: f64,
    pub verdict: AuditVerdict,
    /// The teacher's face density (e/Bohr³), when its solver reports one.
    pub face_density: Option<f64>,
    /// Answered steps between the audit's dispatch and its verdict (0 for a synchronous audit):
    /// after a failed asynchronous audit these steps were answered by an unaudited model.
    pub unaudited_steps: usize,
    /// Run on the background auditor (ruling S14) rather than in the step.
    pub asynchronous: bool,
    /// The teacher solve's wall time (ms).
    pub teacher_ms: f64,
}

/// What a background job is for (module doc, "Audit" and "Side densities").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JobKind {
    Audit,
    /// A display-only solve at the current geometry.
    Refresh,
    /// The isolated atom of this Z on the scene's cell.
    Atom(u8),
}

/// One asynchronous job (ruling S14): the geometry, its provenance, the auditor generation it
/// belongs to (a Forget or an installed sidecar starts a new one), `answered` at dispatch, what it
/// is for and its side-job number (0 for an atom).
struct AuditJob {
    scene: Scene,
    prov: Provenance,
    generation: u64,
    answered: usize,
    kind: JobKind,
    id: u64,
}

/// The auditor's answer: (energy, forces, face density, display) or the teacher's error.
struct AuditAnswer {
    job: AuditJob,
    result: Result<(f64, Vec<[f64; 3]>, Option<f64>, Option<Display>), String>,
    ms: f64,
}

/// The background auditor (ruling S14, module doc "Audit"): a second teacher instance on its own
/// thread; at most one job in flight. A job of an older generation is abandoned at the teacher's
/// next abort check (GPU: the next SCF step) and its answer discarded.
struct Auditor {
    jobs: Option<mpsc::Sender<AuditJob>>,
    answers: mpsc::Receiver<AuditAnswer>,
    generation: Arc<AtomicU64>,
    handle: Option<std::thread::JoinHandle<()>>,
    /// The job in flight.
    busy: Option<JobKind>,
}

impl Auditor {
    fn spawn(mut teacher: Box<dyn ForceProvider>) -> Auditor {
        let (jtx, jrx) = mpsc::channel::<AuditJob>();
        let (atx, arx) = mpsc::channel::<AuditAnswer>();
        let generation = Arc::new(AtomicU64::new(0));
        let current = generation.clone();
        let handle = std::thread::Builder::new()
            .name("surrogate-audit".into())
            .spawn(move || {
                let mut last_atom = false;
                while let Ok(job) = jrx.recv() {
                    // the audits are thousands of steps apart: no extrapolation through the
                    // previous one's geometry (the density is kept as a warm start) — but an
                    // isolated atom's density is no warm start for a molecule, nor the reverse
                    let atom = matches!(job.kind, JobKind::Atom(_));
                    if atom || last_atom {
                        teacher.reset();
                    } else {
                        teacher.discontinuity();
                    }
                    last_atom = atom;
                    let t = Instant::now();
                    let g = job.generation;
                    let result = teacher
                        .evaluate(&job.scene, &mut || current.load(Ordering::Relaxed) != g)
                        .map(|r| (r.energy, r.forces, r.stats.face_density, r.display))
                        .map_err(|e| e.to_string());
                    let ms = t.elapsed().as_secs_f64() * 1e3;
                    if atx.send(AuditAnswer { job, result, ms }).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn the surrogate audit thread");
        Auditor { jobs: Some(jtx), answers: arx, generation, handle: Some(handle), busy: None }
    }

    fn generation(&self) -> u64 {
        self.generation.load(Ordering::Relaxed)
    }

    /// Abandons the job in flight (its answer is discarded on arrival).
    fn new_generation(&mut self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }
}

impl Drop for Auditor {
    /// Abandons the job in flight and joins the thread (≤ one SCF step on the GPU).
    fn drop(&mut self) {
        self.jobs = None;
        self.generation.store(u64::MAX, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Quarantine codes per stored sample (the sidecar's `sample_quarantine` values).
const Q_NONE: u8 = 0;
const Q_HELD: u8 = 1;
const Q_SUPERSEDED: u8 = 2;
/// A box-affected sample (its face density above the box rule) taken while the model was still
/// being built, excluded from the fit once the trust floor is met without it (Task 9 fix round
/// 1, S1): never a quarantine candidate, never fencing.
const Q_BOX: u8 = 3;
/// Quarantine rounds per refit (module doc).
pub const QUARANTINE_ROUNDS: usize = 5;
/// Never quarantine more than this fraction of the store.
pub const QUARANTINE_MAX_FRACTION: f64 = 0.1;
const SQRT_3: f64 = 1.732_050_807_568_877_2;

/// When the model may answer (module doc, "The fallback policy"). Defaults are the spec's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FallbackPolicy {
    /// Absolute bound on the per-component force std, Ha/Bohr (2e-3).
    pub force_std_abs: f64,
    /// Relative bound, fraction of |F_j| (0.10); the resulting bound is capped at
    /// [`FORCE_STD_BOUND_MAX`].
    pub force_std_rel: f64,
    /// Bound on the epistemic energy std per atom, Ha (1e-3).
    pub energy_std_per_atom: f64,
    /// Out-of-distribution envelope: training spreads outside the per-feature training range
    /// (3.0).
    pub max_descriptor_distance: f64,
    /// The O(Q′²) variance (rules 6–7) is evaluated afresh at least on every k-th consultation,
    /// and always on the first after an edit, a refit, a policy change or a rejection (module
    /// doc); 4 (Task 9: measured, `references.md` speed table). In between, the reported
    /// `EvalStats::uncertainty` is the reused verdict's number (≤ k − 1 consultations old).
    pub variance_cadence: usize,
    /// Audit (ruling S7): every `audit_every`-th step the model would have answered, the
    /// teacher is called instead and the gap recorded; 0 disables (2000, ruling S14 interim;
    /// S7 had 200). With a background auditor that the model outruns (full speed) the effective
    /// cadence is one audit per teacher solve, whatever this says (module doc, "Audit").
    pub audit_every: usize,
    /// Draft mode (module doc): the model answers every step it can be asked at all (rules 1–3
    /// pass), extrapolating beyond rules 4–7 and the quarantine fence; audits only in the
    /// background. Off by default.
    pub never_fallback: bool,
}

impl Default for FallbackPolicy {
    fn default() -> FallbackPolicy {
        FallbackPolicy {
            force_std_abs: 2e-3,
            force_std_rel: 0.10,
            energy_std_per_atom: 1e-3,
            max_descriptor_distance: 3.0,
            variance_cadence: 4,
            audit_every: 2000,
            never_fallback: false,
        }
    }
}

impl FallbackPolicy {
    /// The per-component force bound for an atom with force magnitude `fmag`:
    /// min(max(abs, rel·|F|), [`FORCE_STD_BOUND_MAX`]).
    pub fn force_bound(&self, fmag: f64) -> f64 {
        self.force_std_abs.max(self.force_std_rel * fmag).min(FORCE_STD_BOUND_MAX)
    }

    /// One atom's force residual against a reference force: (‖f − f_ref‖/√3, the per-component
    /// equivalent; [`force_bound`](Self::force_bound)(|f_ref|)). The one definition behind the
    /// training-residual trust rule, the worst-offender ranking and the agreement test.
    pub fn force_residual(&self, f: &[f64; 3], f_ref: &[f64; 3]) -> (f64, f64) {
        let d = ((f[0] - f_ref[0]).powi(2) + (f[1] - f_ref[1]).powi(2) + (f[2] - f_ref[2]).powi(2)).sqrt() / SQRT_3;
        (d, self.force_bound((f_ref[0] * f_ref[0] + f_ref[1] * f_ref[1] + f_ref[2] * f_ref[2]).sqrt()))
    }
}

/// One fresh teacher answer inside a quarantine fence that disagreed with the model (fence voting).
struct FenceVote {
    near: Vec<usize>,
    n: usize,
    energy: f64,
    forces: Vec<[f64; 3]>,
}

/// What the last evaluate reported, for the refit jump.
struct LastEval {
    pos: Vec<[f64; 3]>,
    z: Vec<u8>,
    energy: f64,
    on_model: bool,
    provenance: Provenance,
}

/// The variance verdict (rules 6–7) of the last consultation that evaluated it.
#[derive(Clone)]
struct VarianceVerdict {
    accept: bool,
    uncertainty: f64,
    reason: Option<String>,
    /// Per atom: force_std/√3 ÷ its bound.
    atoms: Vec<f64>,
}

/// Why the model is not consulted (rules 1–4), whether the teacher's answer is kept, and
/// whether that sample is subject to thinning (module doc, "Trust recovery").
struct Gate {
    reason: String,
    keep_sample: bool,
    thin: bool,
    /// Rule 4 (an untrusted fit): the only gate draft mode answers past.
    untrusted: bool,
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

/// Worst training residuals of the current model over the store (trust rule).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainingResiduals {
    /// max over samples of |E_model − E_sample| / N (Ha per atom).
    pub energy_per_atom: f64,
    /// max over atoms of ‖F_model − F_sample‖/√3 (Ha/Bohr, per component equivalent) …
    pub force: f64,
    /// … and the policy's force bound for that atom (its teacher force magnitude).
    pub force_bound: f64,
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
}

/// The fitted state a recovery attempt may have to restore.
struct FitState {
    accumulator: Option<Accumulator>,
    accumulated: usize,
    model: Option<Model>,
    report: Option<FitReport>,
    envelope: Option<Envelope>,
    training_residuals: Option<TrainingResiduals>,
    degree_cap: Option<usize>,
}

/// L∞ distance between two descriptors of species `s` in the envelope's training-spread units
/// (a feature the training never varied counts as infinitely far once the two differ).
fn spread_distance(env: &Envelope, s: usize, a: &[f64], b: &[f64]) -> f64 {
    let mut worst = 0.0f64;
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        let gap = (x - y).abs();
        if gap == 0.0 {
            continue;
        }
        let u = env.unit(s, k);
        worst = worst.max(if u > 0.0 { gap / u } else { f64::INFINITY });
    }
    worst
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
        }
    }

    /// Runs the audits (rule 8) on `teacher`, a second instance of this provider's teacher, on a
    /// background thread (ruling S14; module doc, "Audit"): the model keeps answering while the
    /// audit solves. Without one the audits are synchronous — the CPU teacher's case: a second
    /// `CpuKs` would compete with the first for the same cores, and a CPU solve cannot be
    /// abandoned. Panics if `teacher` is not the same kind of teacher (its `name`).
    pub fn with_auditor(mut self, teacher: Box<dyn ForceProvider>) -> SurrogateKs {
        assert_eq!(teacher.name(), self.teacher.name(), "the auditor must be a second instance of the teacher");
        self.auditor = Some(Auditor::spawn(teacher));
        self
    }

    /// The audits run on a background auditor ([`with_auditor`](Self::with_auditor)).
    pub fn audits_asynchronous(&self) -> bool {
        self.auditor.is_some()
    }

    /// An asynchronous audit is in flight (or its answer arrived and waits for the next evaluate).
    pub fn audit_in_flight(&self) -> bool {
        self.held_audit.is_some() || self.auditor.as_ref().is_some_and(|a| a.busy == Some(JobKind::Audit))
    }

    /// The background auditor exists and has no job in flight (module doc, "Side densities").
    pub fn auditor_idle(&self) -> bool {
        self.auditor.as_ref().is_some_and(|a| a.busy.is_none())
    }

    /// Asks the background auditor for a display-only Kohn-Sham solve at `scene` (module doc,
    /// "Side densities"); its density arrives as a `SideKind::Refresh` [`SideDisplay`]. Sent at
    /// once when the auditor is idle, else queued ahead of the next audit (a newer request
    /// replaces a queued one). False without an auditor. Never judged, never a sample.
    pub fn request_display(&mut self, scene: &Scene) -> bool {
        self.poll_side_jobs();
        // an empty scene has no density to show (`crate::provider`, "Nothing to solve")
        if self.auditor.is_none() || scene.nuclei.is_empty() {
            return false;
        }
        let prov = self.scene_provenance(scene);
        self.side_jobs += 1;
        self.side_queue.retain(|j| j.2 != JobKind::Refresh);
        self.side_queue.push_back((scene.clone(), prov, JobKind::Refresh, self.side_jobs));
        self.pump_side_queue();
        true
    }

    /// Asks the background auditor for the isolated atom Z on `scene`'s cell (σ kept, neutral,
    /// lowest multiplicity, at the origin), queued like a refresh; its display arrives through
    /// [`take_atom_densities`](Self::take_atom_densities). False without an auditor.
    pub fn request_atom_density(&mut self, z: u8, scene: &Scene) -> bool {
        self.poll_side_jobs();
        if self.auditor.is_none() {
            return false;
        }
        let atom = Scene {
            nuclei: vec![crate::scene::Nucleus { pos: [0.0; 3], vel: [0.0; 3], z, mass: 1.0 }],
            cell: scene.cell.clone(),
            charge: 0,
            multiplicity: None,
            sigma: scene.sigma,
            external_field: [0.0; 3],
        };
        let prov = self.scene_provenance(&atom);
        self.side_queue.retain(|j| j.2 != JobKind::Atom(z));
        self.side_queue.push_back((atom, prov, JobKind::Atom(z), 0));
        self.pump_side_queue();
        true
    }

    /// Sends the next queued side job if the auditor is idle (module doc, "Side densities": a
    /// queued request goes ahead of an audit that fell due meanwhile).
    fn pump_side_queue(&mut self) {
        if !self.auditor_idle() {
            return;
        }
        if let Some((scene, prov, kind, id)) = self.side_queue.pop_front() {
            if !self.send_job(scene, prov, kind, id) {
                self.side_queue.clear(); // the thread is gone
            }
        }
    }

    /// The isolated-atom densities that arrived since the last call: (Z, its display).
    pub fn take_atom_densities(&mut self) -> Vec<(u8, Display)> {
        self.poll_side_jobs();
        std::mem::take(&mut self.atom_densities)
    }

    /// Outside `evaluate`: takes whatever answer arrived — a refresh's or an atom's is handled at
    /// once; an audit's is **held** unjudged (its verdict is taken at the first evaluate after it
    /// arrives, by the model current then: module doc, "Audit") — and hands the freed auditor the
    /// next queued side job, so a request made while paused is not stuck behind an audit
    /// (review M7).
    fn poll_side_jobs(&mut self) {
        // (one job is in flight at a time, and no audit is dispatched outside `evaluate`, so a
        // held audit answer is never followed by another before it is judged)
        if let Some(ans) = self.receive_answer() {
            if ans.job.kind == JobKind::Audit {
                self.held_audit = Some(ans);
            } else {
                self.handle_answer(ans);
            }
        }
        self.pump_side_queue();
    }

    /// Sends a job to the idle auditor; false when there is none, it is busy, or its thread is
    /// gone (then the audits run in the step from now on).
    fn send_job(&mut self, scene: Scene, prov: Provenance, kind: JobKind, id: u64) -> bool {
        let answered = self.answered;
        let Some(a) = self.auditor.as_mut() else { return false };
        if a.busy.is_some() {
            return false;
        }
        let job = AuditJob { scene, prov, generation: a.generation(), answered, kind, id };
        match a.jobs.as_ref().map(|tx| tx.send(job)) {
            Some(Ok(())) => {
                a.busy = Some(kind);
                true
            }
            _ => {
                self.auditor = None;
                self.audit_due = false;
                false
            }
        }
    }

    /// How many times the variance (rules 6–7) has been evaluated.
    pub fn variance_checks(&self) -> u64 {
        self.variance_checks
    }

    /// The last audit's outcome.
    pub fn last_audit(&self) -> Option<&AuditRecord> {
        self.last_audit.as_ref()
    }

    /// A trusted model that has fallen back on the last ≥ [`STUCK_STEPS`] consultations: why (the
    /// rule that fires, the latest teacher − model gap) — trusted elsewhere, not answering here
    /// (module doc, "Stuck").
    pub fn stuck(&self) -> Option<String> {
        (self.streak >= STUCK_STEPS && self.trusted()).then(|| {
            let reason = self.streak_reason.as_deref().unwrap_or("surrogate: fallback");
            let reason = reason.strip_prefix("surrogate: ").unwrap_or(reason);
            match self.streak_gap {
                Some(g) => format!("not answering here: the last {} steps it was asked fell back — {reason} (teacher − model {g:+.1e} Ha)", self.streak),
                None => format!("not answering here: the last {} steps it was asked fell back — {reason}", self.streak),
            }
        })
    }

    /// Samples held in quarantine (module doc, "Trust recovery"): excluded from the fit, their
    /// neighbourhood falls back.
    pub fn quarantined(&self) -> usize {
        self.quarantine.iter().filter(|&&q| q == Q_HELD).count()
    }

    /// Quarantined samples a fresh teacher answer contradicted (module doc, "Release"): held,
    /// excluded, no longer fencing.
    pub fn superseded(&self) -> usize {
        self.quarantine.iter().filter(|&&q| q == Q_SUPERSEDED).count()
    }

    /// Whether stored sample `i` is quarantined (and fencing).
    pub fn is_quarantined(&self, i: usize) -> bool {
        self.quarantine.get(i) == Some(&Q_HELD)
    }

    /// Box-affected samples taken while the model was being built and excluded from the fit
    /// since (module doc, "Box-affected answers").
    pub fn box_excluded(&self) -> usize {
        self.quarantine.iter().filter(|&&q| q == Q_BOX).count()
    }

    /// Stored samples whose teacher answer was box-affected (its face density above the box
    /// rule), excluded from the fit yet or not.
    pub fn box_affected(&self) -> usize {
        (0..self.samples()).filter(|&i| self.is_box(i)).count()
    }

    /// Whether stored sample `i`'s teacher answer is box-affected (its face density above the box
    /// rule at the store's grid spacing).
    fn is_box(&self, i: usize) -> bool {
        let h = self.store.as_ref().map_or(0.25, |s| s.provenance.h);
        self.face.get(i).is_some_and(|&rho| rho > box_affected_face_density(h))
    }

    /// Once the store holds enough other samples for the trust floor (Q′ + 30), the box-affected
    /// ones leave the fit (`Q_BOX`; a rebuild). Whether anything changed.
    fn exclude_box_samples(&mut self) -> bool {
        let Some(store) = self.store.as_ref() else { return false };
        let n = store.len();
        let boxed: Vec<usize> = (0..n).filter(|&i| self.is_box(i) && self.quarantine.get(i).is_none_or(|&q| q == Q_NONE)).collect();
        if boxed.is_empty() {
            return false;
        }
        let mut species: Vec<u8> = store.samples.iter().flat_map(|s| s.z.iter().copied()).collect();
        species.sort_unstable();
        species.dedup();
        let floor = Basis::new(&species).column_classes().len() + MIN_RESIDUAL_DOF as usize;
        let others = (0..n).filter(|&i| !self.is_box(i) && self.quarantine.get(i).is_none_or(|&q| q == Q_NONE)).count();
        if others < floor {
            return false;
        }
        self.quarantine.resize(n, Q_NONE);
        for i in boxed {
            self.quarantine[i] = Q_BOX;
        }
        self.species_dirty = true;
        true
    }

    /// Whether stored sample `i` is left out of the fit (quarantined, superseded or box-excluded).
    fn excluded(&self, i: usize) -> bool {
        self.quarantine.get(i).is_some_and(|&q| q != Q_NONE)
    }

    /// What the last refit's trust recovery did ("degree_cap 6 …", "N samples quarantined …"),
    /// `None` when it was not needed.
    pub fn recovery_note(&self) -> Option<&str> {
        self.recovery.as_deref()
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

    /// The current model's worst training residuals (`None` before the first refit).
    pub fn training_residuals(&self) -> Option<TrainingResiduals> {
        self.training_residuals
    }

    /// The teacher − model gap of the audit that made the model untrusted, until the next refit.
    pub fn audit_failed(&self) -> Option<f64> {
        self.audit_failed
    }

    /// Samples held (fallback steps and `add_sample`).
    pub fn samples(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.len())
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

    pub fn policy(&self) -> &FallbackPolicy {
        &self.policy
    }

    /// Replaces the policy; the variance cadence restarts and the training residuals are
    /// re-judged against the new thresholds.
    pub fn set_policy(&mut self, p: FallbackPolicy) {
        self.policy = p;
        self.consults = 0;
        self.last_variance = None;
        self.training_residuals = self.compute_training_residuals();
    }

    /// Teach from an existing sample (a sampling pass, a loaded sidecar). `scene` supplies the
    /// settings the sample was produced at (its provenance); the first sample pins the store's.
    pub fn add_sample(&mut self, s: Sample, scene: &Scene) -> Result<(), StoreError> {
        let prov = self.scene_provenance(scene);
        self.add_sample_with_provenance(s, &prov)
    }

    /// The primitive behind [`add_sample`](Self::add_sample): a sample with the provenance it
    /// was produced at (what a sidecar stores).
    pub fn add_sample_with_provenance(&mut self, s: Sample, prov: &Provenance) -> Result<(), StoreError> {
        self.push_sample(s, prov, f64::NAN)
    }

    /// [`add_sample`](Self::add_sample) with the teacher's reported face density
    /// (`EvalStats::face_density`; `None` when its solver reports none) — what a sampling pass
    /// uses (final review I1), so the box rule applies to pass samples as it does to the
    /// essential live samples (module doc, "Box-affected answers"): kept, stored with their
    /// density, never quarantine candidates, excluded from the fit once the trust floor is met
    /// without them.
    pub fn add_sample_with_face(&mut self, s: Sample, scene: &Scene, face: Option<f64>) -> Result<(), StoreError> {
        let prov = self.scene_provenance(scene);
        self.push_sample(s, &prov, face.unwrap_or(f64::NAN))
    }

    /// Isolated-atom energy of species `z` (Hartree), the model's reference (Z-only: computed
    /// by the teacher, never a table). A change after a fit rebuilds the accumulator at the next
    /// refit. The caller vouches that `e` follows the reference rule (ruling S9: the atom's
    /// lowest-energy multiplicity — what `crate::sampling` computes); the sidecar records it.
    pub fn set_reference_energy(&mut self, z: u8, e: f64) {
        self.legacy_refs.remove(&z);
        if self.refs.insert(z, e) != Some(e) {
            self.refs_dirty = true;
        }
    }

    /// The trust rule (module doc): a model exists and its report, its training residuals and
    /// the last audit pass.
    pub fn trusted(&self) -> bool {
        self.model.is_some() && self.untrusted_reason().is_none()
    }

    /// Why the model may not answer (the trust rule's first failing condition), `None` when it
    /// may (or when nothing was fitted: see [`trusted`](Self::trusted)).
    pub fn untrusted_reason(&self) -> Option<String> {
        let r = self.report.as_ref()?;
        if r.n_samples < r.q_prime {
            return Some(format!("n_samples {} < q_prime {}", r.n_samples, r.q_prime));
        }
        if r.residual_dof_e < MIN_RESIDUAL_DOF {
            return Some(format!("residual_dof_e {:.1} < {MIN_RESIDUAL_DOF:.0}", r.residual_dof_e));
        }
        if r.residual_dof_f < MIN_RESIDUAL_DOF {
            return Some(format!("residual_dof_f {:.1} < {MIN_RESIDUAL_DOF:.0}", r.residual_dof_f));
        }
        if let Some(t) = &self.training_residuals {
            if t.energy_per_atom > self.policy.energy_std_per_atom {
                return Some(format!(
                    "max training energy residual {:.1e} > {:.1e} Ha/atom",
                    t.energy_per_atom, self.policy.energy_std_per_atom
                ));
            }
            if t.force > t.force_bound {
                return Some(format!(
                    "max training force residual {:.1e} > {:.1e} Ha/Bohr",
                    t.force, t.force_bound
                ));
            }
        }
        if let Some(gap) = self.audit_failed {
            return Some(format!("audit found teacher − model {gap:+.1e} Ha; refit to recover"));
        }
        None
    }

    /// One pass over the samples the current model was fitted on (the accumulated prefix of
    /// the store — samples kept since the last refit are not training data yet; module doc,
    /// "Trust rule").
    fn compute_training_residuals(&self) -> Option<TrainingResiduals> {
        let (model, store) = (self.model.as_ref()?, self.store.as_ref()?);
        let mut worst = TrainingResiduals { energy_per_atom: 0.0, force: 0.0, force_bound: f64::INFINITY };
        let mut worst_ratio = 0.0;
        for (i, s) in store.samples[..self.accumulated.min(store.len())].iter().enumerate() {
            if self.excluded(i) {
                continue;
            }
            let Ok(p) = model.predict(&s.pos, &s.z, false) else { continue };
            let n = s.pos.len().max(1) as f64;
            worst.energy_per_atom = worst.energy_per_atom.max((p.energy - s.energy).abs() / n);
            for (fm, fs) in p.forces.iter().zip(&s.forces) {
                let (d, bound) = self.policy.force_residual(fm, fs);
                if d / bound > worst_ratio {
                    worst_ratio = d / bound;
                    worst.force = d;
                    worst.force_bound = bound;
                }
            }
        }
        Some(worst)
    }

    /// Rebuild the model from the samples (module doc): the accumulator takes the samples added
    /// since the last refit, or is rebuilt from the store after a species or reference change;
    /// then one hyperparameter search and solve, and one pass over the store for the training
    /// residuals. Clears a failed audit. Computes the refit jump for the next evaluate and bumps
    /// the surface epoch when the surface moved. On an error the previous model, report and
    /// envelope stay in place.
    pub fn refit(&mut self) -> Result<FitReport, FitError> {
        let was_trusted = self.trusted();
        let was_model = self.last.as_ref().is_some_and(|l| l.on_model);
        let mut report = self.fit_store()?;
        self.recovery = None;
        // fence votes (module doc, "Fence voting"): a voted sample the new model cannot
        // reproduce is quarantined itself and its region takes no further vote
        let rejected = self.judge_votes();
        if rejected > 0 {
            self.species_dirty = true;
            report = self.fit_store()?;
            self.recovery = Some(format!(
                "{rejected} voted samples quarantined: the model cannot learn the fresh teacher answers next to a quarantined geometry — that region stays fenced"
            ));
        }
        // release what the new model reproduces (module doc, "Release")
        let released = self.release_reproduced();
        if released > 0 {
            self.species_dirty = true;
            report = self.fit_store()?;
            self.recovery = Some(format!("{released} quarantined samples released: the model now reproduces them"));
        }
        if self.residuals_fail() {
            report = self.recover_trust(report);
        }
        self.surface_moved(was_trusted, was_model).map_err(FitError::Describe)?;
        Ok(report)
    }

    /// The training residuals exceed the policy's thresholds (the trust rule's S8 part).
    fn residuals_fail(&self) -> bool {
        self.training_residuals
            .is_some_and(|t| t.energy_per_atom > self.policy.energy_std_per_atom || t.force > t.force_bound)
    }

    fn save_state(&self) -> FitState {
        FitState {
            accumulator: self.accumulator.clone(),
            accumulated: self.accumulated,
            model: self.model.clone(),
            report: self.report.clone(),
            envelope: self.envelope.clone(),
            training_residuals: self.training_residuals,
            degree_cap: self.degree_cap,
        }
    }

    fn restore_state(&mut self, s: FitState) {
        self.accumulator = s.accumulator;
        self.accumulated = s.accumulated;
        self.model = s.model;
        self.report = s.report;
        self.envelope = s.envelope;
        self.training_residuals = s.training_residuals;
        self.degree_cap = s.degree_cap;
    }

    /// Ruling S15 (module doc, "Trust recovery"): the degree-cap retry, then quarantine rounds.
    /// Returns the report of the fit that stands; a failed attempt restores the previous fit.
    fn recover_trust(&mut self, report: FitReport) -> FitReport {
        // 1. the bent-molecule setting
        let cap = self.basis().map(|b| (b.body3, b.degree_cap));
        if let Some((true, d)) = cap.filter(|_| self.degree_cap.is_none()) {
            if d < RECOVERY_DEGREE_CAP {
                let saved = self.save_state();
                self.degree_cap = Some(RECOVERY_DEGREE_CAP);
                self.species_dirty = true;
                match self.fit_store() {
                    Ok(r) if !self.residuals_fail() => {
                        self.recovery = Some(format!("refit with degree_cap {RECOVERY_DEGREE_CAP} (the bent-molecule setting) restored the training residuals"));
                        return r;
                    }
                    _ => self.restore_state(saved),
                }
            }
        }
        // 2. quarantine the worst offenders, round by round
        let mut report = report;
        let n_store = self.samples();
        let before = self.quarantined();
        for _ in 0..QUARANTINE_ROUNDS {
            if !self.residuals_fail() {
                break;
            }
            let worst = self.worst_offenders();
            let total = self.quarantined() + worst.len();
            if worst.is_empty() || total as f64 > QUARANTINE_MAX_FRACTION * n_store as f64 {
                break;
            }
            let saved = self.save_state();
            let flags = self.quarantine.clone();
            self.quarantine.resize(n_store, Q_NONE);
            for &i in &worst {
                self.quarantine[i] = Q_HELD;
            }
            self.species_dirty = true; // rebuild without them
            match self.fit_store() {
                Ok(r) => report = r,
                Err(_) => {
                    self.restore_state(saved);
                    self.quarantine = flags;
                    break;
                }
            }
        }
        let added = self.quarantined() - before;
        if added > 0 {
            self.recovery = Some(format!(
                "{} samples quarantined — the model cannot represent them ({added} this refit; kept, excluded from the fit; geometries near them fall back)",
                self.quarantined()
            ));
        }
        report
    }

    /// Whether the model reproduces (energy, forces) at a geometry within the policy's thresholds.
    fn reproduces(&self, pos: &[[f64; 3]], z: &[u8], energy: f64, forces: &[[f64; 3]]) -> bool {
        let Some(model) = self.model.as_ref() else { return false };
        let Ok(pr) = model.predict(pos, z, false) else { return false };
        Self::agree(&self.policy, pos.len(), pr.energy, &pr.forces, energy, forces)
    }

    /// Two answers agree within the policy's thresholds (per-atom energy; every atom's force
    /// against its bound for the reference force).
    fn agree(p: &FallbackPolicy, n: usize, e_a: f64, f_a: &[[f64; 3]], e_b: f64, f_b: &[[f64; 3]]) -> bool {
        (e_a - e_b).abs() / n.max(1) as f64 <= p.energy_std_per_atom
            && f_a.iter().zip(f_b).all(|(a, b)| {
                let (d, bound) = p.force_residual(a, b);
                d <= bound
            })
    }

    /// The refit's verdict on the samples fence votes kept (module doc, "Fence voting"): one the
    /// model (fitted with it) reproduces stays in the fit; one it does not is quarantined — a
    /// geometry the model cannot represent — and its region (the fenced samples it was kept next
    /// to, and itself) takes no further vote. How many were quarantined.
    fn judge_votes(&mut self) -> usize {
        let pending = std::mem::take(&mut self.vote_pending);
        let mut rejected = 0;
        for (k, origin) in pending {
            let Some(s) = self.store.as_ref().and_then(|st| st.samples.get(k)).cloned() else { continue };
            if self.excluded(k) || self.reproduces(&s.pos, &s.z, s.energy, &s.forces) {
                continue;
            }
            self.quarantine.resize(self.samples(), Q_NONE);
            self.quarantine[k] = Q_HELD;
            for i in origin.into_iter().chain([k]) {
                self.fence_kept.insert(i, FENCE_VOTE_LIMIT);
            }
            rejected += 1;
        }
        rejected
    }

    /// Releases the quarantined samples the current model reproduces; how many.
    fn release_reproduced(&mut self) -> usize {
        let held: Vec<usize> = (0..self.quarantine.len()).filter(|&i| self.quarantine[i] == Q_HELD).collect();
        let Some(store) = self.store.as_ref() else { return 0 };
        let back: Vec<usize> = held
            .into_iter()
            .filter(|&i| store.samples.get(i).is_some_and(|s| self.reproduces(&s.pos, &s.z, s.energy, &s.forces)))
            .collect();
        for &i in &back {
            self.quarantine[i] = Q_NONE;
        }
        back.len()
    }

    /// The quarantined (fencing) samples within the quarantine radius of the geometry: for each
    /// held sample, the worst query atom's distance to that sample's nearest atom of its species
    /// (training-spread units, L∞ over features). One description of the query for all of them
    /// (Task 9: it used to re-describe the query once per held sample).
    fn near_quarantined(&mut self, pos: &[[f64; 3]], z: &[u8]) -> Vec<usize> {
        let limit = QUARANTINE_FRACTION * self.policy.max_descriptor_distance;
        self.quarantine_distances(pos, z).into_iter().filter(|&(_, d)| d <= limit).map(|(i, _)| i).collect()
    }

    /// (index, distance) of every held quarantined sample from the geometry (`near_quarantined`).
    fn quarantine_distances(&mut self, pos: &[[f64; 3]], z: &[u8]) -> Vec<(usize, f64)> {
        if self.quarantined() == 0 {
            return Vec::new();
        }
        self.ensure_desc_cache();
        let (Some(basis), Some(env)) = (self.desc_basis.as_ref(), self.envelope.as_ref()) else { return Vec::new() };
        let (Ok(desc), Ok(sp)) = (describe_values(basis, pos, z), basis.species_indices(z)) else { return Vec::new() };
        (0..self.desc_cache.len())
            .filter(|&i| self.quarantine.get(i) == Some(&Q_HELD))
            .map(|i| {
                let atoms = &self.desc_cache[i];
                let worst = sp.iter().zip(&desc).fold(0.0f64, |worst, (qs, qd)| {
                    let best = atoms
                        .iter()
                        .filter(|(s, _)| s == qs)
                        .map(|(s, b)| spread_distance(env, *s, qd, b))
                        .fold(f64::INFINITY, f64::min);
                    worst.max(best)
                });
                (i, worst)
            })
            .collect()
    }

    /// The training samples (not quarantined) whose residual ratio (per-atom energy residual
    /// over its threshold, or worst atom force residual over its bound) exceeds 1 and is within a
    /// factor 2 of the worst ratio.
    fn worst_offenders(&self) -> Vec<usize> {
        let (Some(model), Some(store)) = (self.model.as_ref(), self.store.as_ref()) else { return Vec::new() };
        let p = &self.policy;
        let ratios: Vec<(usize, f64)> = store.samples[..self.accumulated.min(store.len())]
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.excluded(*i) && !self.is_box(*i))
            .filter_map(|(i, s)| {
                let pr = model.predict(&s.pos, &s.z, false).ok()?;
                let n = s.pos.len().max(1) as f64;
                let mut r = (pr.energy - s.energy).abs() / n / p.energy_std_per_atom;
                for (fm, fs) in pr.forces.iter().zip(&s.forces) {
                    let (d, bound) = p.force_residual(fm, fs);
                    r = r.max(d / bound);
                }
                Some((i, r))
            })
            .collect();
        let worst = ratios.iter().map(|x| x.1).fold(0.0, f64::max);
        if worst <= 1.0 {
            return Vec::new();
        }
        ratios.into_iter().filter(|&(_, r)| r > 1.0 && r >= 0.5 * worst).map(|(i, _)| i).collect()
    }

    /// The fit itself (module doc): incremental, or a rebuild from the store skipping the
    /// quarantined samples; no surface bookkeeping (`refit` does it once).
    fn fit_store(&mut self) -> Result<FitReport, FitError> {
        let Some(store) = self.store.as_ref() else { return Err(FitError::NoSamples) };
        if store.is_empty() {
            return Err(FitError::NoSamples);
        }
        self.exclude_box_samples();
        let store = self.store.as_ref().expect("checked");
        let provenance = store.provenance.clone();
        let opts = FitOptions::default();
        // (Re)build the accumulator: from scratch after a species or reference change, else
        // only the samples added since the last refit.
        let rebuild = self.accumulator.is_none() || self.species_dirty || self.refs_dirty;
        let basis = if rebuild {
            let mut species: Vec<u8> = store.samples.iter().flat_map(|s| s.z.iter().copied()).collect();
            species.sort_unstable();
            species.dedup();
            let mut b = Basis::new(&species);
            if let Some(d) = self.degree_cap.filter(|_| b.body3) {
                b.degree_cap = d;
            }
            b
        } else {
            self.accumulator.as_ref().expect("checked").basis.clone()
        };
        // the references first: an error here changes nothing
        let refs = self.reference_vector(&basis)?;
        let (mut acc, mut accumulated) = if rebuild {
            (Accumulator::new(&basis), 0)
        } else {
            (self.accumulator.take().expect("checked"), self.accumulated)
        };
        let store = self.store.as_ref().expect("checked");
        for (i, s) in store.samples.iter().enumerate().skip(accumulated) {
            if self.quarantine.get(i).is_some_and(|&q| q != Q_NONE) {
                accumulated += 1; // held, not fitted (module doc, "Trust recovery")
                continue;
            }
            if let Err(e) = acc.add(s, &basis, &refs, &opts) {
                // cannot happen after a rebuild (the basis covers the store); keep the old
                // accumulator state consistent on the incremental path by rebuilding next time
                self.species_dirty = true;
                if !rebuild {
                    self.accumulator = Some(acc);
                }
                return Err(e.into());
            }
            accumulated += 1;
        }
        let solved = acc.solve(&opts, &provenance, &refs);
        let (model, report) = match solved {
            Ok(x) => x,
            Err(e) => {
                if !rebuild {
                    self.accumulator = Some(acc);
                    self.accumulated = accumulated;
                }
                return Err(e);
            }
        };
        self.envelope = Some(acc.envelope.clone());
        self.accumulator = Some(acc);
        self.accumulated = accumulated;
        self.species_dirty = false;
        self.refs_dirty = false;
        self.model = Some(model);
        self.report = Some(report.clone());
        self.audit_failed = None;
        self.training_residuals = self.compute_training_residuals();
        self.pending_refit = false;
        self.consults = 0;
        self.last_variance = None;
        // the votes disagreed with the previous model
        self.fence_votes.clear();
        Ok(report)
    }

    /// Surface bookkeeping after `self.model` was replaced (a refit or an installed sidecar;
    /// module doc): the jump for the next evaluate, and the epoch. `was_trusted` / `was_model`
    /// describe the state before the replacement. The next answer is the model's when it is
    /// trusted — or, in draft mode, whenever rules 1–3 pass (the `match`): an untrusted refit
    /// under draft mode moves the surface the atoms are on, and its jump must be booked too.
    fn surface_moved(&mut self, was_trusted: bool, was_model: bool) -> Result<(), surrogate::DescribeError> {
        let model = self.model.as_ref().expect("a model was just installed");
        let next_is_model = (self.trusted() || self.policy.never_fallback)
            && match &self.last {
                None => true,
                Some(l) => l.provenance == model.provenance && model.basis.species_indices(&l.z).is_ok(),
            };
        self.pending_jump = match (&self.last, next_is_model) {
            (Some(l), true) => {
                let e_new = model.predict(&l.pos, &l.z, false)?.energy;
                Some(e_new - l.energy)
            }
            _ => None,
        };
        if next_is_model || was_model || (was_trusted != self.trusted()) {
            self.epoch += 1;
        }
        Ok(())
    }

    /// The teacher, for work the policy must not see: the sampling pass (`crate::sampling`)
    /// solves its geometries with it directly — they are training data, not steps.
    pub fn teacher_mut(&mut self) -> &mut dyn ForceProvider {
        self.teacher.as_mut()
    }

    /// The provenance a sample produced at this scene's settings by this provider's teacher
    /// carries (what the store and a loaded model are compared with).
    pub fn provenance_for(&self, scene: &Scene) -> Provenance {
        self.scene_provenance(scene)
    }

    /// The isolated-atom energy set for `z`, if any (Hartree).
    pub fn reference_energy(&self, z: u8) -> Option<f64> {
        self.refs.get(&z).copied()
    }

    /// The training samples held (`None` before the first one).
    pub fn store(&self) -> Option<&SampleStore> {
        self.store.as_ref()
    }

    /// The sufficient statistics of the current fit (`None` before the first refit).
    pub fn accumulator(&self) -> Option<&Accumulator> {
        self.accumulator.as_ref()
    }

    /// Drops the model, its samples, statistics and reference energies: the provider is a
    /// pass-through to its teacher again (rule 1 of the policy). If the model was the reported
    /// surface the epoch moves, so the driver re-evaluates at once; that evaluation is on the
    /// teacher surface without a jump and the driver restarts its conserved-energy history
    /// (the honest outcome: the surface the invariant lived on is gone).
    pub fn forget(&mut self) {
        let was_model = self.last.as_ref().is_some_and(|l| l.on_model);
        if self.trusted() || was_model {
            self.epoch += 1;
        }
        self.store = None;
        self.accumulator = None;
        self.accumulated = 0;
        self.refs.clear();
        self.legacy_refs.clear();
        self.refs_dirty = false;
        self.species_dirty = false;
        self.model = None;
        self.report = None;
        self.envelope = None;
        self.training_residuals = None;
        self.audit_failed = None;
        self.pending_refit = false;
        self.pending_jump = None;
        self.consults = 0;
        self.last_variance = None;
        self.answered = 0;
        self.quarantine.clear();
        self.face.clear();
        self.fence_votes.clear();
        self.fence_kept.clear();
        self.vote_pending.clear();
        self.degree_cap = None;
        self.since_kept = THIN_EVERY;
        self.desc_cache.clear();
        self.desc_basis = None;
        self.recovery = None;
        self.reset_audits();
    }

    /// A model forgotten or replaced: the audit in flight is abandoned, nothing is due or booked.
    fn reset_audits(&mut self) {
        if let Some(a) = self.auditor.as_mut() {
            a.new_generation();
        }
        self.audit_due = false;
        self.audit_flip = false;
        self.last_audit = None;
        self.streak = 0;
        self.streak_reason = None;
        self.streak_gap = None;
    }

    /// Writes the full `.model` sidecar (`surrogate::sidecar`): model, accumulator, last fit
    /// report and every sample held — including any taken since the last refit, which a load
    /// restores as pending. `InvalidInput` if nothing has been fitted yet.
    pub fn save_sidecar(&self, path: &std::path::Path) -> std::io::Result<()> {
        let (Some(model), Some(acc)) = (&self.model, &self.accumulator) else {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "no fitted model to save (sample and refit first)"));
        };
        let samples = self.store.as_ref().map_or(&[][..], |s| &s.samples[..]);
        let rule = self.legacy_refs.is_empty().then_some(surrogate::sidecar::REFERENCE_RULE);
        let flags: Vec<u8> = (0..samples.len()).map(|i| self.quarantine.get(i).copied().unwrap_or(Q_NONE)).collect();
        // the face densities only when one matters (a file is otherwise unchanged)
        let face: Vec<f64> = if (0..samples.len()).any(|i| self.is_box(i)) {
            (0..samples.len()).map(|i| self.face.get(i).copied().unwrap_or(f64::NAN)).collect()
        } else {
            Vec::new()
        };
        surrogate::sidecar::write_with_flags(path, model, acc, self.report.as_ref(), samples, rule, &flags, &face)
    }

    /// Species whose reference energy predates ruling S9 (from a sidecar without the rule).
    pub fn legacy_references(&self) -> Vec<u8> {
        self.legacy_refs.iter().copied().collect()
    }

    /// Replaces everything learned with a sidecar's contents (the caller has checked its
    /// provenance against the scene — `Dynamics` refuses a mismatch). The file must hold the
    /// fit report and at least the samples its accumulator was built from (a model-only file
    /// from `Model::write_sidecar` cannot be judged by the trust rule's training residuals and
    /// is refused). The training residuals are recomputed from the samples; the surface
    /// bookkeeping is a refit's (epoch, jump at the last evaluated positions).
    pub fn install_sidecar(&mut self, sc: surrogate::Sidecar) -> Result<(), String> {
        let surrogate::Sidecar { model, accumulator, report, samples, reference_rule, quarantine, face_density, .. } = sc;
        let Some(report) = report else {
            return Err("the file has no fit report (a model-only sidecar): it cannot be judged by the trust rule".into());
        };
        if accumulator.basis != model.basis {
            return Err("the file's accumulator and model disagree on the basis".into());
        }
        if samples.len() < accumulator.n_samples {
            return Err(format!(
                "the file holds {} samples but its fit used {}: without them the training residuals cannot be checked (a model-only sidecar)",
                samples.len(),
                accumulator.n_samples
            ));
        }
        if model.ref_energy.len() != model.basis.n_species() {
            return Err("the file's reference energies do not match its species".into());
        }
        let mut store = SampleStore::new(model.provenance.clone());
        for (i, smp) in samples.into_iter().enumerate() {
            store.push(smp).map_err(|e| format!("sample {i}: {e}"))?;
        }
        let was_trusted = self.trusted();
        let was_model = self.last.as_ref().is_some_and(|l| l.on_model);
        self.refs = model.basis.species.iter().copied().zip(model.ref_energy.iter().copied()).collect();
        self.legacy_refs = match reference_rule.as_deref() {
            Some(surrogate::sidecar::REFERENCE_RULE) => Default::default(),
            _ => model.basis.species.iter().copied().collect(),
        };
        self.refs_dirty = false;
        self.species_dirty = store.samples.iter().flat_map(|s| s.z.iter()).any(|&z| model.basis.species_index(z).is_none());
        // the accumulated store prefix: the samples the fit used plus the quarantined ones among
        // them (quarantine only ever marks samples inside the prefix)
        let q = |i: usize| quarantine.get(i).is_some_and(|&c| c != Q_NONE);
        let mut prefix = 0usize;
        let mut fitted = 0usize;
        while prefix < store.len() && (fitted < accumulator.n_samples || q(prefix)) {
            if !q(prefix) {
                fitted += 1;
            }
            prefix += 1;
        }
        self.accumulated = prefix;
        self.pending_refit = store.len() > self.accumulated;
        self.envelope = Some(accumulator.envelope.clone());
        self.accumulator = Some(accumulator);
        self.face = if face_density.is_empty() { vec![f64::NAN; store.len()] } else { face_density };
        self.store = Some(store);
        self.quarantine = quarantine;
        self.fence_votes.clear();
        self.fence_kept.clear();
        self.vote_pending.clear();
        self.degree_cap = (model.basis.body3 && model.basis.degree_cap != Basis::new(&[]).degree_cap).then_some(model.basis.degree_cap);
        self.desc_cache.clear();
        self.desc_basis = None;
        self.since_kept = THIN_EVERY;
        self.recovery = None;
        self.reset_audits();
        self.model = Some(model);
        self.report = Some(report);
        self.audit_failed = None;
        self.training_residuals = self.compute_training_residuals();
        self.consults = 0;
        self.last_variance = None;
        self.answered = 0;
        // Cannot fail here: `surface_moved` predicts only at a geometry whose species the new
        // basis covers (it checks `species_indices` first), and `describe` has no other error;
        // the state above is therefore never left half-installed (review m7).
        self.surface_moved(was_trusted, was_model).map_err(|e| e.to_string())
    }

    /// Provenance of a sample produced at this scene's settings by this teacher.
    fn scene_provenance(&self, scene: &Scene) -> Provenance {
        Provenance {
            grid_n: scene.cell.n,
            h: scene.cell.h,
            sigma: scene.sigma.unwrap_or(scene.cell.h),
            tol: self.teacher_tol,
            charge: scene.charge,
            multiplicity: scene.multiplicity,
            teacher: self.teacher.name().to_string(),
        }
    }

    /// Into the store (created with `prov` if absent); marks the basis dirty for a new species.
    fn push_sample(&mut self, s: Sample, prov: &Provenance, face: f64) -> Result<(), StoreError> {
        let z = s.z.clone();
        let n = self.store.as_ref().map_or(0, |st| st.len());
        if !self.quarantine.is_empty() {
            self.quarantine.resize(n, Q_NONE);
        }
        match &mut self.store {
            Some(store) => store.push_checked(s, prov)?,
            None => {
                let mut store = SampleStore::new(prov.clone());
                store.push(s)?;
                self.store = Some(store);
            }
        }
        self.face.resize(n, f64::NAN);
        self.face.push(face);
        if let Some(acc) = &self.accumulator {
            if z.iter().any(|&zi| acc.basis.species_index(zi).is_none()) {
                self.species_dirty = true;
            }
        }
        Ok(())
    }

    /// Reference energies in `basis.species` order.
    fn reference_vector(&self, basis: &Basis) -> Result<Vec<f64>, FitError> {
        basis
            .species
            .iter()
            .map(|&z| match self.refs.get(&z) {
                Some(&e) if e.is_finite() => Ok(e),
                _ => Err(FitError::MissingReference(z)),
            })
            .collect()
    }

    /// Rules 1–4: whether the model may be consulted for this scene.
    fn gate(&self, z: &[u8], prov: &Provenance) -> Result<(), Gate> {
        let Some(model) = &self.model else {
            return Err(Gate { reason: "surrogate: no model yet".into(), keep_sample: true, thin: false, untrusted: false });
        };
        if model.provenance != *prov {
            return Err(Gate {
                reason: format!(
                    "surrogate: scene ({}) differs from the model's ({}) — sample not kept",
                    prov.describe(),
                    model.provenance.describe()
                ),
                keep_sample: false,
                thin: false,
                untrusted: false,
            });
        }
        if let Some(&zu) = z.iter().find(|&&zi| model.basis.species_index(zi).is_none()) {
            let known: Vec<String> = model.basis.species.iter().map(|z| crate::provider::symbol(*z).to_string()).collect();
            return Err(Gate {
                reason: format!(
                    "surrogate: species {} is not in the model ({}) — the model is rebuilt with it at the next refit",
                    crate::provider::symbol(zu),
                    known.join(", ")
                ),
                keep_sample: true,
                thin: false,
                untrusted: false,
            });
        }
        if let Some(why) = self.untrusted_reason() {
            return Err(Gate { reason: format!("surrogate untrusted: {why}"), keep_sample: true, thin: !self.trust_needs_samples(), untrusted: true });
        }
        Ok(())
    }

    /// The trust rule fails for want of data (n_samples < Q′ or too few residual degrees of
    /// freedom): every teacher answer is then worth keeping (no thinning).
    fn trust_needs_samples(&self) -> bool {
        self.report.as_ref().is_none_or(|r| r.n_samples < r.q_prime || r.residual_dof_e < MIN_RESIDUAL_DOF || r.residual_dof_f < MIN_RESIDUAL_DOF)
    }

    /// Fills the per-sample descriptor cache under the current model's basis.
    fn ensure_desc_cache(&mut self) {
        let Some(basis) = self.model.as_ref().map(|m| m.basis.clone()) else { return };
        if self.desc_basis.as_ref() != Some(&basis) {
            self.desc_cache.clear();
            self.desc_basis = Some(basis.clone());
        }
        let Some(store) = self.store.as_ref() else { return };
        for s in &store.samples[self.desc_cache.len()..] {
            let atoms = match (describe_values(&basis, &s.pos, &s.z), basis.species_indices(&s.z)) {
                (Ok(d), Ok(sp)) => sp.into_iter().zip(d).collect(),
                _ => Vec::new(), // a species the basis lacks: never "near" anything
            };
            self.desc_cache.push(atoms);
        }
    }

    /// The distance (training-spread units, L∞ over features; the worst query atom's nearest
    /// stored atom of its species) from the geometry to the nearest stored sample (thinning).
    /// ∞ without a model, an envelope or a candidate.
    fn nearest_stored(&mut self, pos: &[[f64; 3]], z: &[u8]) -> f64 {
        self.ensure_desc_cache();
        let (Some(basis), Some(env)) = (self.desc_basis.as_ref(), self.envelope.as_ref()) else { return f64::INFINITY };
        let (Ok(desc), Ok(sp)) = (describe_values(basis, pos, z), basis.species_indices(z)) else { return f64::INFINITY };
        let mut worst_atom = 0.0f64;
        for (qs, qd) in sp.iter().zip(&desc) {
            let mut best = f64::INFINITY;
            for atoms in &self.desc_cache {
                for (s, b) in atoms {
                    if s == qs {
                        best = best.min(spread_distance(env, *s, qd, b));
                    }
                }
            }
            worst_atom = worst_atom.max(best);
        }
        worst_atom
    }

    /// Rule 5: each atom's distance from the training envelope in units of the limit (> 1 is
    /// outside; draft mode's readout) and the reason naming the worst, if any is outside.
    fn envelope_outside(&self, model: &Model, pos: &[[f64; 3]], z: &[u8]) -> (Vec<f64>, Option<String>) {
        let Some(env) = self.envelope.as_ref() else { return (Vec::new(), None) };
        let (Ok(desc), Ok(s)) = (describe_values(&model.basis, pos, z), model.basis.species_indices(z)) else { return (Vec::new(), None) };
        let mut worst = (0.0f64, 0usize, 0usize);
        let mut atoms = Vec::with_capacity(desc.len());
        for (i, d) in desc.iter().enumerate() {
            let (dist, k) = env.distance(s[i], d);
            atoms.push(dist / self.policy.max_descriptor_distance);
            if dist > worst.0 {
                worst = (dist, i, k);
            }
        }
        (atoms, (worst.0 > self.policy.max_descriptor_distance).then(|| {
            format!(
                "surrogate: atom {} is {:.1} training spreads outside the descriptor envelope (feature {}, limit {})",
                worst.1,
                worst.0,
                model.basis.feature_names()[worst.2],
                self.policy.max_descriptor_distance
            )
        }))
    }

    /// Rules 6–7 on a prediction that carries stds.
    fn variance_check(&self, model: &Model, pred: &surrogate::Prediction, e_std: f64, f_std: &[f64]) -> VarianceVerdict {
        let n = pred.forces.len();
        let p = &self.policy;
        // 6. forces
        let mut uncertainty = 0.0f64;
        let mut worst: Option<(usize, f64, f64)> = None;
        let mut atoms = Vec::with_capacity(n);
        for (j, (&sd, f)) in f_std.iter().zip(&pred.forces).enumerate() {
            let fs = sd / SQRT_3;
            uncertainty = uncertainty.max(fs);
            let bound = p.force_bound((f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt());
            atoms.push(fs / bound);
            if fs > bound && worst.is_none_or(|(_, s, b)| fs / bound > s / b) {
                worst = Some((j, fs, bound));
            }
        }
        if let Some((j, fs, bound)) = worst {
            return VarianceVerdict {
                accept: false,
                uncertainty,
                reason: Some(format!("surrogate: force std {fs:.2e} > bound {bound:.2e} Ha/Bohr on atom {j}")),
                atoms,
            };
        }
        // 7. energy, per atom, epistemic only
        let floor = if model.energy_per_atom { n as f64 * model.sigma_e } else { model.sigma_e };
        let epistemic_per_atom = (e_std * e_std - floor * floor).max(0.0).sqrt() / n as f64;
        if epistemic_per_atom > p.energy_std_per_atom {
            return VarianceVerdict {
                accept: false,
                uncertainty,
                reason: Some(format!(
                    "surrogate: epistemic energy std {epistemic_per_atom:.2e} > {:.2e} Ha/atom",
                    p.energy_std_per_atom
                )),
                atoms,
            };
        }
        VarianceVerdict { accept: true, uncertainty, reason: None, atoms }
    }

    /// Whether a teacher answer becomes a training sample (module doc: box-affected answers,
    /// quarantined neighbourhoods, thinning); when not, the reason goes to `notes`. `model` is the
    /// model's answer at the geometry when it was consulted; `near` the quarantined samples
    /// fencing it; `keep` the rule's own verdict; `thin` whether thinning applies.
    #[allow(clippy::too_many_arguments)]
    fn admit(
        &mut self,
        pos: &[[f64; 3]],
        z: &[u8],
        energy: f64,
        forces: &[[f64; 3]],
        face: Option<f64>,
        model: Option<(f64, &[[f64; 3]])>,
        near: &[usize],
        keep: bool,
        thin: bool,
        essential: bool,
        h: f64,
        notes: &mut Vec<String>,
    ) -> bool {
        if !keep {
            return false;
        }
        // no model yet (module doc, "No model yet"): unthinned up to the cap, then one in THIN_EVERY
        if self.model.is_none() {
            let cap = self.no_model_cap(z);
            if self.samples() >= cap {
                if self.since_kept + 1 < THIN_EVERY {
                    self.since_kept += 1;
                    notes.push(format!(
                        "surrogate: waiting for a sampling pass — {} samples held without a model (≥ {cap}); one in {THIN_EVERY} further teacher answers is kept",
                        self.samples()
                    ));
                    return false;
                }
            }
        }
        let rule = box_affected_face_density(h);
        // box-affected answers are the box, not the molecule (module doc) — except the samples a
        // model is still being built from (no model, a new species, trust short of samples), which
        // a sampling pass would take on this box anyway
        if let Some(rho) = face.filter(|&rho| !essential && rho > rule) {
            notes.push(format!(
                "surrogate: sample not kept — the teacher's density on the outermost grid plane is {rho:.1e} > {rule:.1e} (box-affected: used for this step, not trained on)"
            ));
            return false;
        }
        // a quarantined neighbourhood (module doc, "Release")
        if !near.is_empty() {
            let agrees = model.is_some_and(|(e, f)| Self::agree(&self.policy, pos.len(), e, f, energy, forces));
            if !agrees {
                // fence voting (module doc, "Fence voting"): fresh answers that agree with each
                // other but not with the model are evidence the model is wrong here
                let n = pos.len();
                self.fence_votes.push(FenceVote { near: near.to_vec(), n, energy, forces: forces.to_vec() });
                if self.fence_votes.len() > FENCE_VOTES {
                    self.fence_votes.remove(0);
                }
                // the last k answers: the same fence, mutually consistent (every pair agrees
                // within the per-atom energy threshold and every atom's force bound)
                let votes = &self.fence_votes;
                let consistent = votes.len() == FENCE_VOTES
                    && votes.iter().all(|v| v.n == n && v.near.iter().any(|i| near.contains(i)))
                    && votes.iter().enumerate().all(|(a, va)| {
                        votes[a + 1..].iter().all(|vb| Self::agree(&self.policy, n, va.energy, &va.forces, vb.energy, &vb.forces))
                    });
                // one vote per region until a refit has judged it; at most FENCE_VOTE_LIMIT per
                // fenced sample
                let pending = self.vote_pending.iter().any(|(_, o)| o.iter().any(|i| near.contains(i)));
                let room = !pending && near.iter().any(|i| self.fence_kept.get(i).copied().unwrap_or(0) < FENCE_VOTE_LIMIT);
                if consistent && room {
                    for &i in near {
                        *self.fence_kept.entry(i).or_insert(0) += 1;
                    }
                    self.fence_votes.clear();
                    self.vote_pending.push((self.samples(), near.to_vec()));
                    notes.push(format!(
                        "surrogate: kept — {FENCE_VOTES} mutually consistent fresh teacher answers next to a quarantined geometry disagree with the model; the next refit decides"
                    ));
                    return true;
                }
                notes.push("surrogate: sample not kept — next to a quarantined geometry, and the teacher still disagrees with the model there".into());
                return false;
            }
            self.fence_votes.clear();
            let (mut released, mut superseded) = (0, 0);
            for &i in near {
                let s = self.store.as_ref().expect("a quarantined sample is stored").samples[i].clone();
                if self.reproduces(&s.pos, &s.z, s.energy, &s.forces) {
                    self.quarantine[i] = Q_NONE;
                    released += 1;
                } else {
                    self.quarantine[i] = Q_SUPERSEDED;
                    superseded += 1;
                }
            }
            self.species_dirty = true; // the fit's sample set changed: rebuild
            notes.push(format!(
                "surrogate: a fresh teacher answer near quarantined samples agrees with the model — {released} released, {superseded} superseded"
            ));
            return true;
        }
        // thinning (module doc, "Trust recovery"): audits and the samples trust needs are kept
        if thin && self.since_kept + 1 < THIN_EVERY {
            let d = self.nearest_stored(pos, z);
            let limit = THIN_FRACTION * self.policy.max_descriptor_distance;
            if d <= limit {
                self.since_kept += 1;
                notes.push(format!(
                    "surrogate: sample not kept — {d:.2} training spreads from a stored one (≤ {limit:.2}; one in {THIN_EVERY} such is kept)"
                ));
                return false;
            }
        }
        true
    }

    /// The no-model cap (module doc, "No model yet"): [`NO_MODEL_SAMPLE_FACTOR`] × Q′ of the
    /// default basis over the held species and `z`'s.
    fn no_model_cap(&self, z: &[u8]) -> usize {
        let mut species: Vec<u8> = z.to_vec();
        if let Some(store) = self.store.as_ref() {
            species.extend(store.samples.iter().flat_map(|s| s.z.iter().copied()));
        }
        species.sort_unstable();
        species.dedup();
        NO_MODEL_SAMPLE_FACTOR * Basis::new(&species).column_classes().len()
    }

    /// Into the store as a fallback sample (an admitted teacher answer); marks the refit pending.
    #[allow(clippy::too_many_arguments)]
    fn keep_sample(&mut self, pos: &[[f64; 3]], z: &[u8], energy: f64, forces: &[[f64; 3]], face: Option<f64>, prov: &Provenance, notes: &mut Vec<String>) {
        self.since_kept = 0;
        let sample = Sample { pos: pos.to_vec(), z: z.to_vec(), energy, forces: forces.to_vec(), tag: SampleTag::Fallback, weight: 1.0 };
        match self.push_sample(sample, prov, face.unwrap_or(f64::NAN)) {
            Ok(()) => self.pending_refit = true,
            Err(e) => notes.push(format!("surrogate: sample not kept ({e})")),
        }
    }

    /// Hands an audit that fell due to the background auditor (module doc, "Audit"); `false`
    /// when there is none (the audit then runs in the step). A busy auditor defers it to the
    /// next answered step.
    fn dispatch_audit(&mut self, scene: &Scene, prov: &Provenance) -> bool {
        let every = self.policy.audit_every;
        let Some(a) = self.auditor.as_ref() else { return false };
        if a.busy.is_some() {
            self.audit_due = true;
            return true;
        }
        self.side_jobs += 1;
        if self.send_job(scene.clone(), prov.clone(), JobKind::Audit, self.side_jobs) {
            self.audit_due = false;
            self.pending_notes.push(format!("surrogate audit dispatched to the background teacher (every {every} answered steps; the model answers meanwhile)"));
            true
        } else {
            // the thread is gone (`send_job` dropped it): audit in the step from now on
            self.side_jobs -= 1;
            false
        }
    }

    /// Inside `evaluate`: handles a held audit answer, else the one that arrived
    /// ([`handle_answer`](Self::handle_answer)), then hands the freed auditor the next queued side
    /// job.
    fn poll_audit(&mut self) {
        if let Some(ans) = self.held_audit.take().or_else(|| self.receive_answer()) {
            self.handle_answer(ans);
        }
        self.pump_side_queue();
    }

    /// The background auditor's answer, if one arrived (the auditor is then free).
    fn receive_answer(&mut self) -> Option<AuditAnswer> {
        let a = self.auditor.as_mut()?;
        match a.answers.try_recv() {
            Ok(ans) => {
                a.busy = None;
                Some(ans)
            }
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                self.auditor = None;
                self.audit_due = false;
                self.pending_notes.push("surrogate: the background audit thread stopped — audits run in the step from now on".into());
                None
            }
        }
    }

    /// One answer of the background auditor: a side density is kept for the app, an audit is
    /// judged with the current model (module doc, "Audit"): the sample is admitted like any
    /// teacher answer, the gap recorded; a failed audit untrusts the model and books the flip at
    /// the next evaluation. An answer of an older generation is discarded.
    fn handle_answer(&mut self, ans: AuditAnswer) {
        let Some(a) = self.auditor.as_ref() else { return };
        if ans.job.generation != a.generation() {
            return; // the model it was for was forgotten or replaced
        }
        let positions = || ans.job.scene.nuclei.iter().map(|n| n.pos).collect::<Vec<[f64; 3]>>();
        match (ans.job.kind, ans.result) {
            (JobKind::Atom(z), Ok((.., Some(d)))) => return self.atom_densities.push((z, d)),
            (JobKind::Refresh, Ok((.., Some(d)))) => {
                self.side_display = Some(SideDisplay { display: d, kind: SideKind::Refresh, job: ans.job.id, positions: positions() });
                return;
            }
            (JobKind::Atom(_) | JobKind::Refresh, Ok(_)) => {
                return self.pending_notes.push("surrogate: the background teacher returned no density (its display is off)".into());
            }
            (JobKind::Atom(z), Err(e)) => {
                return self.pending_notes.push(format!("surrogate: the isolated-atom density of Z = {z} failed on the background teacher ({e})"));
            }
            (JobKind::Refresh, Err(e)) => {
                return self.pending_notes.push(format!("surrogate: the density refresh failed on the background teacher ({e})"));
            }
            (JobKind::Audit, Err(e)) => {
                return self.pending_notes.push(format!("surrogate audit: the background teacher failed ({e}) — not judged"));
            }
            (JobKind::Audit, Ok((energy, forces, face, display))) => {
                if let Some(d) = display {
                    self.side_display = Some(SideDisplay { display: d, kind: SideKind::Audit, job: ans.job.id, positions: positions() });
                }
                self.judge_audit(ans.job, energy, forces, face, ans.ms);
            }
        }
    }

    /// A background audit's verdict (module doc, "Audit"): the sample is admitted like any
    /// teacher answer, the gap recorded; a failed audit untrusts the model and books the flip at
    /// the next evaluation.
    fn judge_audit(&mut self, job: AuditJob, energy: f64, forces: Vec<[f64; 3]>, face: Option<f64>, ms: f64) {
        let unaudited = self.answered.saturating_sub(job.answered);
        let AuditJob { scene, prov, .. } = job;
        let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        let z: Vec<u8> = scene.nuclei.iter().map(|n| n.z).collect();
        // the model current now is the one to judge (a refit since the dispatch is what counts)
        let pred = self
            .model
            .as_ref()
            .filter(|m| m.provenance == prov && m.basis.species_indices(&z).is_ok())
            .and_then(|m| m.predict(&pos, &z, false).ok());
        let Some(pred) = pred else {
            self.pending_notes.push("surrogate audit: discarded — the model changed settings or species since its dispatch".into());
            return;
        };
        let n = pos.len().max(1) as f64;
        let gap = energy - pred.energy;
        let near = self.near_quarantined(&pos, &z);
        let mut notes = Vec::new();
        let rule = box_affected_face_density(prov.h);
        if self.admit(&pos, &z, energy, &forces, face, Some((pred.energy, &pred.forces)), &near, true, false, false, prov.h, &mut notes) {
            self.keep_sample(&pos, &z, energy, &forces, face, &prov, &mut notes);
        }
        let box_rho = face.filter(|&rho| rho > rule);
        let thr = self.policy.energy_std_per_atom;
        let verdict = if box_rho.is_some() {
            AuditVerdict::BoxAffected
        } else if gap.abs() / n > thr {
            AuditVerdict::Failed
        } else {
            AuditVerdict::Passed
        };
        notes.insert(0, match verdict {
            AuditVerdict::Passed => format!("surrogate audit passed (background teacher, judged {unaudited} answered steps after its dispatch): teacher − model {gap:+.2e} Ha"),
            AuditVerdict::BoxAffected => format!(
                "surrogate audit not judged: the teacher's answer is box-affected (face density {:.1e} > {rule:.1e}) — its teacher − model {gap:+.2e} Ha is the box's confinement as much as the model's error",
                box_rho.unwrap_or(0.0)
            ),
            AuditVerdict::Failed => {
                self.audit_failed = Some(gap);
                self.audit_flip = self.last.as_ref().is_some_and(|l| l.on_model);
                format!(
                    "surrogate audit failed (background teacher): teacher − model {gap:+.2e} Ha ({:.1e} > {thr:.1e} Ha/atom) — the {unaudited} answered steps since its dispatch were answered by an unaudited model; untrusted until the next refit",
                    gap.abs() / n
                )
            }
        });
        self.last_audit = Some(AuditRecord { gap, verdict, face_density: face, unaudited_steps: unaudited, asynchronous: true, teacher_ms: ms });
        self.pending_notes.extend(notes);
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
        Ok(ForceResult { energy, forces: r.forces, stats, display: r.display })
    }
}

impl ForceProvider for SurrogateKs {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        // the empty scene (`crate::provider`, "Nothing to solve"): no species, so no model, no
        // fallback, no sample, no audit — the known answer, before any rule runs
        if let Some(r) = crate::provider::nothing_to_solve(scene) {
            return r;
        }
        let t0 = Instant::now();
        // a background audit's verdict, if it arrived (module doc, "Audit")
        self.poll_audit();
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
        if reason.is_some() || audit {
            let reason = reason.unwrap_or_else(|| format!("surrogate audit (every {} answered steps)", self.policy.audit_every));
            let h = Handoff {
                prov,
                pos,
                z,
                reason,
                keep_sample: true,
                // the variance rules' fallbacks are the answers active learning asks for
                thin: by_envelope,
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
        self.last = None;
        self.last_variance = None; // the next consultation checks the variance afresh
    }

    fn discontinuity(&mut self) {
        self.teacher.discontinuity();
        self.last = None;
        self.last_variance = None;
    }

    fn surface_epoch(&self) -> u64 {
        self.epoch
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn side_jobs(&self) -> u64 {
        self.side_jobs
    }

    fn take_side_display(&mut self) -> Option<SideDisplay> {
        self.poll_side_jobs();
        self.side_display.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::{DynCommand, Dynamics};
    use crate::provider::Surface;
    use crate::scene::{Cell, Nucleus};
    use crate::test_providers::{MorseDimer, Shared};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::sync::{Arc, Mutex};

    fn never() -> impl FnMut() -> bool {
        || false
    }

    fn cell() -> Cell {
        Cell { n: 33, h: 0.25 }
    }

    /// Two nuclei (Z `z0`, `z1`) `r` apart along `dir`, centred on `origin` (so a dimer sits
    /// near the middle of the 8 Bohr cell: the GPU test's energies must not see the faces).
    fn dimer(r: f64, dir: [f64; 3], origin: [f64; 3], z0: u8, z1: u8) -> Scene {
        let p0: [f64; 3] = std::array::from_fn(|a| origin[a] - 0.5 * r * dir[a]);
        let p1: [f64; 3] = std::array::from_fn(|a| origin[a] + 0.5 * r * dir[a]);
        Scene {
            nuclei: vec![Nucleus::new(z0, p0), Nucleus::new(z1, p1)],
            cell: cell(),
            charge: 0,
            multiplicity: None,
            sigma: None,
            external_field: [0.0; 3],
        }
    }

    fn h2(r: f64) -> Scene {
        dimer(r, [1.0, 0.0, 0.0], [0.0; 3], 1, 1)
    }

    fn unit_vector(rng: &mut StdRng) -> [f64; 3] {
        loop {
            let v: [f64; 3] = [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)];
            let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if (1e-3..1.0).contains(&n) {
                return v.map(|x| x / n);
            }
        }
    }

    /// A Morse-teacher sample of the scene (the teacher's exact answer).
    fn morse_sample(teacher: &MorseDimer, scene: &Scene) -> Sample {
        let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        let (energy, forces) = teacher.energy_and_forces(&pos).unwrap();
        Sample { pos, z: scene.nuclei.iter().map(|n| n.z).collect(), energy, forces, tag: SampleTag::Scan, weight: 1.0 }
    }

    /// A surrogate over a shared Morse teacher with `n` scan samples on r ∈ [lo, hi] (random
    /// orientations and origins), the H reference set, refitted when `n > 0`.
    fn trained(lo: f64, hi: f64, n: usize, policy: FallbackPolicy) -> (SurrogateKs, Arc<Mutex<MorseDimer>>) {
        let teacher = Shared::new(MorseDimer::h2());
        let handle = teacher.handle();
        let mut s = SurrogateKs::new(Box::new(teacher), policy).with_teacher_tol(1e-6);
        s.set_reference_energy(1, MorseDimer::h2().e_atom);
        let mut rng = StdRng::seed_from_u64(4);
        for i in 0..n {
            let r = lo + (hi - lo) * i as f64 / (n.max(2) - 1) as f64;
            let origin = [rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3)];
            let sc = dimer(r, unit_vector(&mut rng), origin, 1, 1);
            s.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
        }
        if n > 0 {
            let rep = s.refit().expect("refit");
            println!("trained on {n} samples r ∈ [{lo}, {hi}]: rmse E/atom {:.1e}, F {:.1e}; s_E {:.1e} s_F {:.1e} λ {:.3e}; dof_E {:.1} dof_F {:.1} q′ {}; converged {}",
                     rep.rmse_energy_per_atom, rep.rmse_force, rep.sigma_e, rep.sigma_f, rep.lambda, rep.residual_dof_e, rep.residual_dof_f, rep.q_prime, rep.converged);
        }
        (s, handle)
    }

    fn calls(h: &Arc<Mutex<MorseDimer>>) -> usize {
        h.lock().unwrap().calls
    }

    fn max_force_diff(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
        a.iter().zip(b).flat_map(|(x, y)| (0..3).map(move |k| (x[k] - y[k]).abs())).fold(0.0, f64::max)
    }

    /// The Verlet swing to compare against: E_conserved's spread and largest per-step change
    /// over `steps` of a plain run on `teacher` from `scene` (same dt, no surrogate).
    fn verlet_reference(scene: &Scene, teacher: MorseDimer, steps: usize) -> (f64, f64) {
        let mut d = Dynamics::new(scene.clone(), Box::new(teacher), 0.25, 1);
        let mut prev = d.snapshot().energy_conserved;
        let (mut lo, mut hi, mut max_step) = (prev, prev, 0.0f64);
        for _ in 0..steps {
            d.apply(DynCommand::Step(1));
            let e = d.tick(f64::INFINITY, &mut never()).energy_conserved;
            lo = lo.min(e);
            hi = hi.max(e);
            max_step = max_step.max((e - prev).abs());
            prev = e;
        }
        (hi - lo, max_step)
    }

    fn surrogate_of(d: &mut Dynamics) -> &mut SurrogateKs {
        d.provider_mut().as_any_mut().and_then(|a| a.downcast_mut::<SurrogateKs>()).expect("a SurrogateKs")
    }

    /// Rule 1: nothing fitted → the teacher answers, the sample is kept, the result is on the
    /// teacher surface.
    #[test]
    fn no_model_means_fallback_and_a_sample() {
        let (mut s, h) = trained(0.0, 0.0, 0, FallbackPolicy::default());
        assert_eq!(s.name(), "surrogate(morse dimer)");
        assert!(!s.trusted() && s.model().is_none() && s.basis().is_none());
        let scene = h2(1.5);
        let r = s.evaluate(&scene, &mut never()).unwrap();
        assert_eq!(calls(&h), 1);
        assert_eq!(s.samples(), 1);
        assert!(s.pending_refit());
        assert_eq!(r.stats.source, Source::Fallback);
        assert_eq!(r.stats.energy_surface(), Surface::Teacher);
        assert_eq!(r.stats.teacher_gap, None);
        assert_eq!(r.stats.refit_jump, None);
        assert_eq!(r.stats.uncertainty, None);
        let (e, f) = MorseDimer::h2().energy_and_forces(&[[-0.75, 0.0, 0.0], [0.75, 0.0, 0.0]]).unwrap();
        assert_eq!(r.energy, e);
        assert_eq!(r.forces, f);
        assert!(r.stats.notes.iter().any(|n| n.contains("no model yet")), "{:?}", r.stats.notes);
        assert_eq!(s.epoch(), 0);
    }

    /// Rules 6–8 and the learning loop: a window-trained dimer answers inside the window to
    /// the teacher's accuracy, falls back outside it (keeping the sample), and answers there
    /// after the refit.
    #[test]
    fn window_trained_model_answers_inside_and_falls_back_outside() {
        let (mut s, h) = trained(1.2, 2.2, 40, FallbackPolicy::default());
        assert!(s.trusted(), "{:?}", s.report());
        assert_eq!(s.basis().unwrap().species, vec![1]);
        let teacher = MorseDimer::h2();

        let inside = h2(1.6);
        let r = s.evaluate(&inside, &mut never()).unwrap();
        let (e, f) = teacher.energy_and_forces(&[[-0.8, 0.0, 0.0], [0.8, 0.0, 0.0]]).unwrap();
        println!("r = 1.6: {:?}, |ΔE| {:.1e}, max|ΔF| {:.1e}, uncertainty {:?}", r.stats.source, (r.energy - e).abs(), max_force_diff(&r.forces, &f), r.stats.uncertainty);
        assert_eq!(r.stats.source, Source::Surrogate, "{:?}", r.stats.notes);
        assert_eq!(calls(&h), 0, "the teacher was not called");
        assert!((r.energy - e).abs() < 1e-5);
        assert!(max_force_diff(&r.forces, &f) < 1e-4);
        assert!(r.stats.uncertainty.is_some_and(|u| u < 2e-3));
        assert_eq!(r.stats.teacher_gap, None);
        assert!(r.display.is_none());
        assert_eq!(r.stats.scf_steps, 0);
        assert_eq!(r.stats.energy.total, r.energy);

        // outside the window: the teacher answers on the model surface and the sample is kept
        let near = h2(3.0);
        let r = s.evaluate(&near, &mut never()).unwrap();
        println!("r = 3.0: {:?}, notes {:?}", r.stats.source, r.stats.notes);
        assert_eq!(r.stats.source, Source::Fallback);
        assert_eq!(calls(&h), 1);
        assert_eq!(s.samples(), 41);
        assert!(s.pending_refit());
        let (e3, f3) = teacher.energy_and_forces(&[[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]]).unwrap();
        assert_eq!(r.forces, f3, "the teacher's forces");
        let gap = r.stats.teacher_gap.expect("on the model surface");
        assert!((r.energy + gap - e3).abs() < 1e-12, "energy is the model's: {} + {gap} vs {e3}", r.energy);
        assert_eq!(r.stats.energy_surface(), Surface::Model);
        assert!(r.stats.energy.total == e3, "the teacher's breakdown is kept");
        assert!(r.stats.model_forces.is_some(), "the model forces travel with a model-surface fallback");

        let rep = s.refit().unwrap();
        let tr = s.training_residuals().unwrap();
        println!("after the refit with the r = 3.0 sample: rmse E/atom {:.1e}, F {:.1e}; s_E {:.1e} s_F {:.1e}; training residuals E {:.1e}/atom, F {:.1e} (bound {:.1e})",
                 rep.rmse_energy_per_atom, rep.rmse_force, rep.sigma_e, rep.sigma_f, tr.energy_per_atom, tr.force, tr.force_bound);
        assert_eq!(rep.n_samples, 41);
        assert!(!s.pending_refit());
        assert!(s.trusted(), "{tr:?}");
        assert_eq!(s.epoch(), 2, "the surface moved twice (first fit, this refit)");
        let r = s.evaluate(&near, &mut never()).unwrap();
        println!("r = 3.0 after refit: {:?}, |ΔE| {:.1e}, max|ΔF| {:.1e}, uncertainty {:?}, notes {:?}", r.stats.source, (r.energy - e3).abs(), max_force_diff(&r.forces, &f3), r.stats.uncertainty, r.stats.notes);
        assert_eq!(r.stats.source, Source::Surrogate, "{:?}", r.stats.notes);
        assert_eq!(calls(&h), 1);
        assert!((r.energy - e3).abs() < 1e-4);
        assert!(max_force_diff(&r.forces, &f3) < 1e-4);
        assert!(r.stats.refit_jump.is_some(), "the first evaluate after a refit reports the jump");
        assert_eq!(s.evaluate(&near, &mut never()).unwrap().stats.refit_jump, None, "once");

    }

    /// A surrogate over a shared Morse teacher trained on `n` window samples of the scripted
    /// teacher Morse(r) + a·sin(k r), k = 2π·`periods`/(hi − lo) — the sinusoid rides on the
    /// Morse energy with its consistent force −a k cos(k r) along the bond. With a = 0 this is
    /// `trained`.
    fn trained_oscillatory(lo: f64, hi: f64, n: usize, a: f64, periods: f64) -> (SurrogateKs, Arc<Mutex<MorseDimer>>) {
        let teacher = Shared::new(MorseDimer::h2());
        let handle = teacher.handle();
        let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_teacher_tol(1e-6);
        s.set_reference_energy(1, MorseDimer::h2().e_atom);
        let k = 2.0 * std::f64::consts::PI * periods / (hi - lo);
        let mut rng = StdRng::seed_from_u64(4);
        for i in 0..n {
            let r = lo + (hi - lo) * i as f64 / (n.max(2) - 1) as f64;
            let origin = [rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3)];
            let dir = unit_vector(&mut rng);
            let sc = dimer(r, dir, origin, 1, 1);
            let mut smp = morse_sample(&MorseDimer::h2(), &sc);
            smp.energy += a * (k * r).sin();
            let de = a * k * (k * r).cos(); // dE/dr; nucleus 1 sits at +r/2·dir
            for c in 0..3 {
                smp.forces[1][c] -= de * dir[c];
                smp.forces[0][c] += de * dir[c];
            }
            s.add_sample(smp, &sc).unwrap();
        }
        s.refit().expect("refit");
        (s, handle)
    }

    /// Ruling S8 (review C): basis misspecification is invisible to the posterior and caught by
    /// the training residuals. The fixture must be something NO basis the provider can build
    /// represents at the training points. The original one (40 Morse points on [1.2, 2.2] plus
    /// one at r = 4.0, whose −3.4e-3 Ha tail the f_c-damped body-2 n_max = 8 basis could not
    /// carry: 1e-2 Ha offset at a training point) was a *body-2* misspecification: the body-3
    /// default basis adds the radial shapes T_{n₁}T_{n₂}f_c²(r/r_cut)^{2l} and fits window + r =
    /// 4.0 (and + r = 5.0) to ~1e-8 Ha/atom, correctly earning trust (Task 6 fix round). The
    /// fixture now is Morse plus a sinusoid of 6 periods across the window, amplitude 1e-2 Ha:
    /// every feature of the default basis is a polynomial in r of degree ≤ 7 (body-2) or ≤ 4
    /// (body-3 at D = 4) times the smooth f_c, so no fit can follow 12 extrema on 1 Bohr — the
    /// two-noise evidence settles on "the data are noisy" (s_E ≈ 4e-3) while the max training
    /// energy residual per atom sits at ≈ a/2 ≥ 3 × `energy_std_per_atom`; the refit must leave
    /// the model untrusted with the note naming that maximum, and the fallback teacher answers.
    /// The same construction with a = 0 is the Morse window and is trusted (the control).
    #[test]
    fn misspecified_fit_is_untrusted_by_its_training_residuals() {
        let (mut s, h) = trained_oscillatory(1.2, 2.2, 40, 1e-2, 6.0);
        let rep = s.report().unwrap().clone();
        let tr = s.training_residuals().unwrap();
        println!("Morse + 1e-2·sin(6 periods): rmse E/atom {:.1e}, F {:.1e}; s_E {:.1e} s_F {:.1e}; γ_E {:.2} γ_F {:.2}; training residuals E {:.1e}/atom, F {:.1e} (bound {:.1e}); trusted {}; reason {:?}",
                 rep.rmse_energy_per_atom, rep.rmse_force, rep.sigma_e, rep.sigma_f, rep.gamma_e, rep.gamma_f, tr.energy_per_atom, tr.force, tr.force_bound, s.trusted(), s.untrusted_reason());
        assert!(rep.n_samples >= rep.q_prime && rep.residual_dof_e >= MIN_RESIDUAL_DOF && rep.residual_dof_f >= MIN_RESIDUAL_DOF, "{rep:?}: the sample-count conditions pass; only the residuals can refuse");
        assert!(!s.trusted(), "the misspecified fit must not be trusted");
        assert!(tr.energy_per_atom > 3e-3, "{tr:?}: max training energy residual ≥ 3× energy_std_per_atom");
        assert!(s.untrusted_reason().unwrap().starts_with("max training energy residual"), "{:?}", s.untrusted_reason());
        let r = s.evaluate(&h2(1.6), &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback);
        assert_eq!(r.stats.energy_surface(), Surface::Teacher);
        assert!(r.stats.notes.iter().any(|n| n.contains("surrogate untrusted: max training energy residual")), "{:?}", r.stats.notes);
        assert_eq!(calls(&h), 1);
        // the epistemic energy criterion alone (the floor N·s_E excluded, as the provider does)
        // would not have caught it at an in-window geometry: the point of the rule
        let model = s.model().unwrap();
        let pred = model.predict(&[[-0.8, 0.0, 0.0], [0.8, 0.0, 0.0]], &[1, 1], true).unwrap();
        let std_e = pred.energy_std.unwrap();
        let epistemic_per_atom = (std_e * std_e - (2.0 * model.sigma_e).powi(2)).max(0.0).sqrt() / 2.0;
        println!("at r = 1.6 the model's epistemic energy std is {epistemic_per_atom:.1e} Ha/atom (total {std_e:.1e}, floor 2·s_E {:.1e}) against a {:.1e} Ha/atom training residual", 2.0 * model.sigma_e, tr.energy_per_atom);
        assert!(epistemic_per_atom < tr.energy_per_atom / 3.0, "the training residual is what refuses, not the posterior");

        // control: the same window without the sinusoid is the Morse fit and is trusted
        let (s0, _) = trained_oscillatory(1.2, 2.2, 40, 0.0, 6.0);
        let tr0 = s0.training_residuals().unwrap();
        println!("a = 0 control: training residuals E {:.1e}/atom, F {:.1e}; trusted {}", tr0.energy_per_atom, tr0.force, s0.trusted());
        assert!(s0.trusted(), "{:?}", s0.untrusted_reason());
        assert!(tr0.energy_per_atom < 1e-6);
    }

    /// Rule 4: a fit with too few samples is never consulted, and the note says why.
    #[test]
    fn untrusted_model_falls_back_with_the_reason() {
        let (mut s, h) = trained(1.2, 2.2, 5, FallbackPolicy::default());
        assert!(!s.trusted());
        let rep = s.report().unwrap().clone();
        assert!(rep.n_samples < rep.q_prime, "{rep:?}");
        let r = s.evaluate(&h2(1.6), &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback);
        assert_eq!(r.stats.energy_surface(), Surface::Teacher);
        assert_eq!(calls(&h), 1);
        assert_eq!(s.samples(), 6);
        let want = format!("surrogate untrusted: n_samples {} < q_prime {}", rep.n_samples, rep.q_prime);
        assert!(r.stats.notes.iter().any(|n| n.contains(&want)), "{:?}", r.stats.notes);

        // enough samples for n_samples ≥ q′ (29 with the body-3 default basis, S = 1) but not
        // for 30 residual energy degrees of freedom (n_E − γ_E < 30 at 30 samples)
        let (mut s, _) = trained(1.2, 2.2, 30, FallbackPolicy::default());
        let rep = s.report().unwrap().clone();
        assert!(rep.n_samples >= rep.q_prime && rep.residual_dof_e < MIN_RESIDUAL_DOF, "{rep:?}");
        assert!(!s.trusted());
        let r = s.evaluate(&h2(1.6), &mut never()).unwrap();
        assert!(r.stats.notes.iter().any(|n| n.contains("untrusted: residual_dof_e")), "{:?}", r.stats.notes);
    }

    /// Rule 3: a scene at other settings than the model's is never answered, even where the
    /// variance would pass; the teacher's answer is not kept.
    #[test]
    fn unseen_species_or_charge_falls_back_unconditionally() {
        let (mut s, h) = trained(1.2, 2.2, 40, FallbackPolicy::default());
        let mut cation = h2(1.6);
        cation.charge = 1;
        let r = s.evaluate(&cation, &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback);
        assert_eq!(calls(&h), 1);
        assert_eq!(s.samples(), 40, "a sample from another surface is not kept");
        assert_eq!(r.stats.teacher_gap, None);
        assert_eq!(r.stats.uncertainty, None);
        assert_eq!(r.stats.energy_surface(), Surface::Teacher);
        assert!(r.stats.notes.iter().any(|n| n.contains("charge 1") && n.contains("differs from the model's")), "{:?}", r.stats.notes);
        // a different σ, grid or multiplicity likewise
        let mut other = h2(1.6);
        other.sigma = Some(0.3);
        assert_eq!(s.evaluate(&other, &mut never()).unwrap().stats.source, Source::Fallback);
        let mut other = h2(1.6);
        other.cell = Cell { n: 49, h: 0.25 };
        assert_eq!(s.evaluate(&other, &mut never()).unwrap().stats.source, Source::Fallback);
        let mut other = h2(1.6);
        other.multiplicity = Some(3);
        assert_eq!(s.evaluate(&other, &mut never()).unwrap().stats.source, Source::Fallback);
        assert_eq!(s.samples(), 40);
        // an unknown species at a foreign provenance: one note (rule 3 precedes rule 2), the
        // sample not kept, no rebuild scheduled
        let mut h_he_cation = dimer(1.6, [0.0, 1.0, 0.0], [0.0; 3], 1, 2);
        h_he_cation.charge = 1;
        let r = s.evaluate(&h_he_cation, &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback);
        let surrogate_notes: Vec<&String> = r.stats.notes.iter().filter(|n| n.starts_with("surrogate")).collect();
        assert_eq!(surrogate_notes.len(), 1, "{:?}", r.stats.notes);
        assert!(surrogate_notes[0].contains("differs from the model's") && !surrogate_notes[0].contains("species"), "{:?}", r.stats.notes);
        assert_eq!(s.samples(), 40);
        assert!(!s.species_dirty);
        // the matching scene is still answered
        assert_eq!(s.evaluate(&h2(1.6), &mut never()).unwrap().stats.source, Source::Surrogate);
    }

    /// Rule 2 and the rebuild: a species the model does not know falls back unconditionally and
    /// is learned at the next refit — once its reference energy exists.
    #[test]
    fn new_species_falls_back_and_is_learned_at_the_next_refit() {
        // 180 samples: the rebuilt two-species body-3 basis has Q′ = 176 (ruling S11/S12), and
        // the trust rule needs n_samples ≥ Q′ after the rebuild
        let (mut s, h) = trained(1.2, 2.2, 180, FallbackPolicy::default());
        let h_he = dimer(1.6, [0.0, 1.0, 0.0], [0.1, -0.1, 0.0], 1, 2);
        let r = s.evaluate(&h_he, &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback);
        assert_eq!(r.stats.energy_surface(), Surface::Teacher);
        assert_eq!(calls(&h), 1);
        assert_eq!(s.samples(), 181, "kept: the rebuild needs it");
        assert!(r.stats.notes.iter().any(|n| n.contains("species He is not in the model (H)")), "{:?}", r.stats.notes);
        // no reference for He yet: the refit fails and the old model stands
        let old_epoch = s.epoch();
        assert_eq!(s.refit(), Err(FitError::MissingReference(2)));
        assert_eq!(s.basis().unwrap().species, vec![1]);
        assert!(s.trusted() && s.epoch() == old_epoch);
        assert_eq!(s.evaluate(&h2(1.6), &mut never()).unwrap().stats.source, Source::Surrogate, "still answers H₂");
        // with the reference the basis is rebuilt with both species
        s.set_reference_energy(2, -2.83);
        let rep = s.refit().unwrap();
        assert_eq!(s.basis().unwrap().species, vec![1, 2]);
        assert_eq!(rep.q_prime, 8 * 3 + 2 * 76, "n_max·S(S+1)/2 body-2 classes plus S·P₃ body-3 columns (D = 4) with S = 2");
        assert_eq!(rep.n_samples, 181);
        let r = s.evaluate(&h_he, &mut never()).unwrap();
        assert!(!r.stats.notes.iter().any(|n| n.contains("species")), "{:?}", r.stats.notes);
        println!("H–He after the rebuild: {:?} {:?}", r.stats.source, r.stats.notes);
        assert_eq!(s.evaluate(&h2(1.6), &mut never()).unwrap().stats.source, Source::Surrogate, "H₂ still answered after the rebuild");
    }

    /// Rule 5: near r_cut every body-2 feature vanishes, the variance is at its floor and would
    /// pass, and only the envelope knows the model has never seen an empty neighbourhood.
    #[test]
    fn envelope_catches_the_vanishing_feature_direction_the_variance_cannot() {
        let policy = FallbackPolicy::default();
        let (mut s, h) = trained(1.2, 2.2, 40, policy);
        // Beyond r_cut = 6.0 every body-2 feature AND its gradient is exactly zero: the leverage
        // vanishes, the stds sit on their floors, and the model would answer "Σ ref, no force"
        // where the Morse still binds by 2e-3 Ha with a 2e-3 Ha/Bohr force. (Just inside r_cut,
        // r = 5.9, the force variance still fires: the Chebyshev gradients do not vanish there.)
        let far = h2(6.5);
        let model = s.model().unwrap();
        let pred = model.predict(&[[-3.25, 0.0, 0.0], [3.25, 0.0, 0.0]], &[1, 1], true).unwrap();
        let fs = pred.force_std.as_ref().unwrap().iter().fold(0.0f64, |m, &x| m.max(x)) / SQRT_3;
        let floor = 2.0 * model.sigma_e;
        let e_epi = (pred.energy_std.unwrap().powi(2) - floor * floor).max(0.0).sqrt() / 2.0;
        let (e_t, f_t) = MorseDimer::h2().energy_and_forces(&[[-3.25, 0.0, 0.0], [3.25, 0.0, 0.0]]).unwrap();
        println!("r = 6.5: per-component force std {fs:.1e} (bound {:.0e}), epistemic energy std/atom {e_epi:.1e} (bound {:.0e}); model E − teacher {:+.1e} Ha, model F {:?} vs teacher {:+.1e}",
                 policy.force_std_abs, policy.energy_std_per_atom, pred.energy - e_t, pred.forces[0], f_t[0][0]);
        assert_eq!(pred.forces, vec![[0.0; 3]; 2], "beyond r_cut the model has no force");
        assert!(fs < 0.1 * policy.force_std_abs && e_epi < 0.1 * policy.energy_std_per_atom, "test premise: the variance passes");
        let r = s.evaluate(&far, &mut never()).unwrap();
        println!("notes {:?}", r.stats.notes);
        assert_eq!(r.stats.source, Source::Fallback);
        assert_eq!(calls(&h), 1);
        assert!(r.stats.notes.iter().any(|n| n.contains("outside the descriptor envelope")), "{:?}", r.stats.notes);
        assert!(r.stats.uncertainty.is_some_and(|u| u < policy.force_std_abs), "the variance is still reported");
        assert_eq!(r.stats.energy_surface(), Surface::Model);
        // inside the window the envelope is quiet
        let r = s.evaluate(&h2(1.7), &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Surrogate);
        // the envelope's limit is the policy's
        s.set_policy(FallbackPolicy { max_descriptor_distance: f64::INFINITY, ..policy });
        let r = s.evaluate(&far, &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Surrogate, "{:?}", r.stats.notes);
    }

    /// One-surface accounting through the driver: with the policy forcing fallbacks, the atoms
    /// move on the teacher's forces while `energy_ks` is the model's and `teacher_gap` the known
    /// difference between the two surfaces.
    #[test]
    fn fallback_step_uses_dft_forces_but_books_the_model_energy() {
        // trained on D = 0.17; the live teacher then has D′ = 1.25 D (energies and forces differ).
        // Fallbacks are forced through the envelope (a negative limit rejects every geometry):
        // zero force thresholds would also fail the training-residual trust condition, which
        // uses the same numbers, and an untrusted model is a teacher-surface fallback.
        let always_fallback = FallbackPolicy { max_descriptor_distance: -1.0, audit_every: 0, ..FallbackPolicy::default() };
        let (s, h) = trained(1.2, 2.2, 40, always_fallback);
        let model = s.model().unwrap().clone();
        let mut live = MorseDimer::h2();
        live.d *= 1.25;
        h.lock().unwrap().d = live.d;

        let v0 = [3e-4, -1e-4, 2e-4];
        let mut scene = h2(1.55);
        scene.nuclei[1].vel = v0;
        let (m0, m1) = (scene.nuclei[0].mass, scene.nuclei[1].mass);
        let dt = units::fs_to_au_time(0.25);
        let mut d = Dynamics::new(scene.clone(), Box::new(s), 0.25, 1);
        let snap0 = d.snapshot();
        assert_eq!(snap0.source, Source::Fallback);
        let p0: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        let (e_t0, f_t0) = live.energy_and_forces(&p0).unwrap();
        let e_m0 = model.predict(&p0, &[1, 1], false).unwrap().energy;
        assert_eq!(snap0.energy_ks, e_m0, "the model's energy");
        assert_eq!(snap0.forces, f_t0, "the teacher's forces");
        assert!((snap0.teacher_gap.unwrap() - (e_t0 - e_m0)).abs() < 1e-14);

        d.apply(DynCommand::Step(1));
        let snap1 = d.tick(f64::INFINITY, &mut never());
        assert_eq!(snap1.step, 1);
        // Verlet drift with the teacher's forces
        let r1: Vec<[f64; 3]> = [(p0[0], [0.0; 3], m0), (p0[1], v0, m1)]
            .iter()
            .zip(&f_t0)
            .map(|((r, v, m), f)| std::array::from_fn(|a| r[a] + dt * v[a] + 0.5 * dt * dt * f[a] / m))
            .collect();
        for (n, r) in snap1.scene.nuclei.iter().zip(&r1) {
            for a in 0..3 {
                assert!((n.pos[a] - r[a]).abs() < 1e-15, "{:?} vs {r:?}", n.pos);
            }
        }
        let p1: Vec<[f64; 3]> = snap1.scene.nuclei.iter().map(|n| n.pos).collect();
        let (e_t1, f_t1) = live.energy_and_forces(&p1).unwrap();
        let e_m1 = model.predict(&p1, &[1, 1], false).unwrap().energy;
        assert_eq!(snap1.source, Source::Fallback);
        assert_eq!(snap1.energy_ks, e_m1);
        assert_eq!(snap1.forces, f_t1);
        let gap = snap1.teacher_gap.unwrap();
        assert!((gap - (e_t1 - e_m1)).abs() < 1e-14, "gap {gap} vs {}", e_t1 - e_m1);
        // the known offset between the surfaces: 0.25 D [(1 − x)² − 1] at r₁
        let r = ((p1[1][0] - p1[0][0]).powi(2) + (p1[1][1] - p1[0][1]).powi(2) + (p1[1][2] - p1[0][2]).powi(2)).sqrt();
        let x = (-(r - 1.4)).exp();
        let offset = 0.25 * 0.17 * ((1.0 - x) * (1.0 - x) - 1.0);
        println!("gap {gap:+.6e} Ha, analytic offset {offset:+.6e} Ha, model error {:+.1e}", gap - offset);
        assert!((gap - offset).abs() < 1e-5, "the gap is the teacher change plus the model's fit error");
        assert_eq!(snap1.eval.energy.total, e_t1, "the teacher's breakdown");
        assert_eq!(snap1.fallback_fraction_100, 1.0);
        assert_eq!(calls(&h), 2);
    }

    /// The accounting across a refit: `E_conserved` is continuous to rounding when the model
    /// first becomes trusted (teacher surface → model surface), across the re-evaluation the
    /// driver does when the epoch changes, and `teacher_gap` is `None` on Surrogate steps.
    #[test]
    fn energy_conserved_is_continuous_across_a_forced_refit() {
        let (s, _h) = trained(0.0, 0.0, 0, FallbackPolicy::default());
        let mut scene = h2(1.55);
        scene.nuclei[1].vel = [4e-4, 1e-4, 0.0];
        let (_, ref_step) = verlet_reference(&scene, MorseDimer::h2(), 100);
        let mut d = Dynamics::new(scene.clone(), Box::new(s), 0.25, 1);
        let mut e_c = Vec::new();
        let mut sources = Vec::new();
        for _ in 0..50 {
            d.apply(DynCommand::Step(1));
            let snap = d.tick(f64::INFINITY, &mut never());
            e_c.push(snap.energy_conserved);
            sources.push(snap.source);
        }
        assert!(sources.iter().all(|&s| s == Source::Fallback));
        let before = d.snapshot();
        assert_eq!(before.model_epoch, 0);
        assert_eq!(before.eval.energy_surface(), Surface::Teacher);
        {
            let s = surrogate_of(&mut d);
            assert_eq!(s.samples(), 51);
            // 20 scan samples from a slightly different well (D 0.5 % deeper): the fit then
            // compromises between them and the 51 live samples, so the model surface differs
            // visibly from the teacher's at r_50 and the jump is a real number, not rounding —
            // while the training residuals (~2e-4/atom) stay inside the trust condition
            let mut deeper = MorseDimer::h2();
            deeper.d *= 1.005;
            let mut rng = StdRng::seed_from_u64(9);
            for i in 0..20 {
                let r = 1.2 + i as f64 / 19.0;
                let sc = dimer(r, unit_vector(&mut rng), [0.0; 3], 1, 1);
                s.add_sample(morse_sample(&deeper, &sc), &sc).unwrap();
            }
            let rep = s.refit().unwrap();
            println!("mid-run refit: {} samples, dof_E {:.1}, dof_F {:.1}, s_F {:.1e}, training residuals {:?}", rep.n_samples, rep.residual_dof_e, rep.residual_dof_f, rep.sigma_f, s.training_residuals());
            assert!(s.trusted(), "{:?}", s.training_residuals());
            assert_eq!(s.epoch(), 1);
        }
        // a paused tick with nothing queued re-evaluates at the unchanged positions (the epoch
        // changed) and books the jump: nothing else moves, so E_conserved must not move either
        let refreshed = d.tick(f64::INFINITY, &mut never());
        assert_eq!(refreshed.step, 50);
        assert_eq!(refreshed.model_epoch, 1);
        assert_eq!(refreshed.eval.energy_surface(), Surface::Model);
        assert_eq!(refreshed.scene, before.scene, "nothing moved");
        let jump = refreshed.eval.refit_jump.expect("the first evaluate after the refit reports the jump");
        let delta = refreshed.energy_conserved - before.energy_conserved;
        println!("E_c before {:.12}, after the refresh {:.12} (Δ {:+.2e}); refit jump {jump:+.3e} Ha (E_ks {:.9} → {:.9}); source {:?}",
                 before.energy_conserved, refreshed.energy_conserved, delta, before.energy_ks, refreshed.energy_ks, refreshed.source);
        assert!((refreshed.energy_ks - before.energy_ks - jump).abs() < 1e-12, "the jump is the change of the reported energy");
        assert!(jump.abs() > 1e-5, "test premise: the surfaces differ visibly");
        assert!(delta.abs() < 1e-12, "E_conserved moved by {delta:+.3e} across the refit");
        assert!(refreshed.history.len() == before.history.len(), "no history restart");
        for _ in 0..50 {
            d.apply(DynCommand::Step(1));
            let snap = d.tick(f64::INFINITY, &mut never());
            if snap.source == Source::Surrogate {
                assert_eq!(snap.teacher_gap, None);
            }
            e_c.push(snap.energy_conserved);
            sources.push(snap.source);
        }
        let n_sur = sources.iter().filter(|&&s| s == Source::Surrogate).count();
        let n_fb = sources.iter().filter(|&&s| s == Source::Fallback).count();
        let fluct = e_c.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f64::max);
        println!("100 steps: {n_sur} surrogate, {n_fb} fallback; max |ΔE_c| per step {fluct:.2e} Ha; fallback fraction (last 100) {:.2}", d.snapshot().fallback_fraction_100);
        assert!(n_sur > 0, "the trained model answers somewhere on the path");
        // no step jumps: the fallback steps' force-mismatch work is booked, so only the Verlet
        // swing of this (0.15 Bohr off r_e) trajectory remains, the same as on the teacher alone
        println!("force-mismatch work booked over the run: {:+.3e} Ha; plain Verlet max per-step change {ref_step:.2e}", d.snapshot().work_mismatch);
        assert!(fluct < 1.5 * ref_step, "a jump: {fluct:.2e} vs the Verlet swing {ref_step:.2e}");
        assert!((d.snapshot().fallback_fraction_100 - n_fb as f64 / 100.0).abs() < 1e-12);
        // the history records each step's source (the viewer's sparkline colours by it): its
        // last 100 entries are exactly the 100 steps' sources, in order
        let snap = d.snapshot();
        assert_eq!(snap.history_source.len(), snap.history.len());
        assert_eq!(&snap.history_source[snap.history_source.len() - 100..], &sources[sources.len() - 100..]);
    }

    /// Review (d), ruling S8: on model-surface fallback steps the teacher's force moves the atoms
    /// while the invariant lives on the model surface; ΔF·Δr is booked into W, so E_conserved is
    /// flat to the Verlet swing, while the unbooked quantity (E_c + work_mismatch) oscillates with
    /// the difference of the two surfaces along the path.
    #[test]
    fn fallback_force_mismatch_work_is_booked() {
        // forced through the envelope (see `fallback_step_uses_dft_forces_but_books_the_model_energy`)
        let always_fallback = FallbackPolicy { max_descriptor_distance: -1.0, audit_every: 0, ..FallbackPolicy::default() };
        let (s, h) = trained(1.2, 2.2, 40, always_fallback);
        h.lock().unwrap().d *= 1.25; // the live teacher's well is 25 % deeper than the model's
        let mut scene = h2(1.55);
        scene.nuclei[0].vel = [-3e-4, 0.0, 0.0];
        scene.nuclei[1].vel = [3e-4, 0.0, 0.0];
        // the same trajectory on the teacher alone: what Verlet itself does to E_conserved
        // (the dimer starts 0.15 Bohr off r_e, 4e-3 Ha of oscillation energy, so the swing is
        // tens of µHa — far above the 1e-7 of a near-equilibrium start)
        let mut live = MorseDimer::h2();
        live.d *= 1.25;
        let (ref_spread, _) = verlet_reference(&scene, live, 200);
        let mut d = Dynamics::new(scene, Box::new(s), 0.25, 1);
        let e0 = d.snapshot().energy_conserved;
        let (mut lo_b, mut hi_b, mut lo_u, mut hi_u) = (e0, e0, e0, e0);
        for _ in 0..200 {
            d.apply(DynCommand::Step(1));
            let snap = d.tick(f64::INFINITY, &mut never());
            assert_eq!(snap.source, Source::Fallback);
            assert!(snap.eval.model_forces.is_some());
            let b = snap.energy_conserved;
            let u = snap.energy_conserved + snap.work_mismatch;
            lo_b = lo_b.min(b);
            hi_b = hi_b.max(b);
            lo_u = lo_u.min(u);
            hi_u = hi_u.max(u);
        }
        let snap = d.snapshot();
        println!("200 forced-fallback steps: E_conserved spread {:.2e} Ha booked, {:.2e} Ha unbooked; plain Verlet on the teacher {:.2e}; Σ mismatch work {:+.3e} Ha; last teacher − model gap {:+.2e}",
                 hi_b - lo_b, hi_u - lo_u, ref_spread, snap.work_mismatch, snap.teacher_gap.unwrap());
        // booked: the Verlet swing and nothing else (a wrong sign would double the unbooked spread)
        assert!(hi_b - lo_b < 1.5 * ref_spread, "booked {:.2e} vs the Verlet swing {ref_spread:.2e}", hi_b - lo_b);
        assert!(hi_u - lo_u > 10.0 * (hi_b - lo_b), "unbooked must visibly move: {:.2e} vs {:.2e}", hi_u - lo_u, hi_b - lo_b);
        assert!(snap.work_mismatch.abs() > 1e-6);
        assert_eq!(snap.fallback_fraction_100, 1.0);
    }

    /// Ruling S7: every `audit_every`-th answered step calls the teacher anyway; a passing audit
    /// is an ordinary model-surface fallback, a failed one (the teacher drifted away) marks the
    /// model untrusted and flips to the teacher surface with the gap booked as `refit_jump`, so
    /// E_conserved stays continuous and the next refit clears the flag.
    #[test]
    fn audit_calls_the_teacher_and_a_failed_audit_flips_the_surface_continuously() {
        let policy = FallbackPolicy { audit_every: 5, ..FallbackPolicy::default() };
        let (s, h) = trained(1.2, 2.2, 40, policy);
        let mut scene = h2(1.5);
        scene.nuclei[1].vel = [2e-4, 0.0, 0.0];
        let mut d = Dynamics::new(scene, Box::new(s), 0.25, 1); // answered step 1
        assert_eq!(d.snapshot().source, Source::Surrogate);
        let mut e_c = vec![d.snapshot().energy_conserved];
        for i in 0..4 {
            d.apply(DynCommand::Step(1));
            let snap = d.tick(f64::INFINITY, &mut never());
            e_c.push(snap.energy_conserved);
            if i == 3 {
                // the 5th answered step is audited: teacher called, gap tiny, still trusted
                assert_eq!(snap.source, Source::Fallback, "{:?}", snap.eval.notes);
                assert!(snap.eval.notes.iter().any(|n| n.contains("surrogate audit (every 5 answered steps)")), "{:?}", snap.eval.notes);
                let gap = snap.teacher_gap.expect("a passing audit is a model-surface fallback");
                println!("passing audit: teacher − model {gap:+.2e} Ha");
                assert!(gap.abs() < 1e-5);
                assert_eq!(snap.eval.refit_jump, None);
            } else {
                assert_eq!(snap.source, Source::Surrogate, "{:?}", snap.eval.notes);
            }
        }
        assert_eq!(calls(&h), 1);
        assert!(surrogate_of(&mut d).trusted());
        assert_eq!(surrogate_of(&mut d).samples(), 41);
        // the teacher drifts away from the model
        h.lock().unwrap().d *= 1.1;
        for i in 0..5 {
            d.apply(DynCommand::Step(1));
            let snap = d.tick(f64::INFINITY, &mut never());
            e_c.push(snap.energy_conserved);
            if i == 4 {
                let jump = snap.eval.refit_jump.expect("the failed audit books the flip");
                let delta = e_c[e_c.len() - 1] - e_c[e_c.len() - 2];
                println!("failed audit: jump {jump:+.3e} Ha, ΔE_c across the flip {delta:+.2e} Ha, notes {:?}", snap.eval.notes);
                assert_eq!(snap.source, Source::Fallback);
                assert!(snap.eval.notes.iter().any(|n| n.contains("surrogate audit failed")), "{:?}", snap.eval.notes);
                assert_eq!(snap.teacher_gap, None);
                assert_eq!(snap.eval.energy_surface(), Surface::Teacher);
                assert!(jump.abs() / 2.0 > 1e-3, "test premise: the drift exceeds the per-atom bound");
                // The jump itself is booked exactly (same positions). What remains is the
                // mixed-force half-kick of this one step: its first half-kick used the model's
                // forces at r_prev (a Surrogate step), its second the teacher's at r_new, and
                // relative to the teacher surface the first was an external force
                // F_model − F_teacher whose work ½ dt v·ΔF (≈ 5·10·2e-4·5e-3 ≈ 5e-6 here,
                // measured 1e-6) cannot be booked without F_teacher(r_prev). A refit avoids
                // this by refreshing at unchanged positions; an audit flips mid-trajectory,
                // once. Bound it at 1e-5 against the 1e-2 Ha jump it replaced.
                assert!(delta.abs() < 1e-5, "E_conserved must not see the flip beyond the half-kick remainder: {delta:+.2e}");
                assert!(snap.notes.iter().all(|n| !n.contains("history restarts")), "{:?}", snap.notes);
            } else {
                assert_eq!(snap.source, Source::Surrogate, "{:?}", snap.eval.notes);
            }
        }
        assert_eq!(calls(&h), 2);
        assert!(!surrogate_of(&mut d).trusted());
        assert!(surrogate_of(&mut d).audit_failed().is_some());
        // every step until a refit is a teacher-surface fallback with the reason
        d.apply(DynCommand::Step(1));
        let snap = d.tick(f64::INFINITY, &mut never());
        assert_eq!(snap.source, Source::Fallback);
        assert!(snap.eval.notes.iter().any(|n| n.contains("surrogate untrusted: audit found teacher − model")), "{:?}", snap.eval.notes);
        assert_eq!(snap.eval.energy_surface(), Surface::Teacher);
        let fluct = e_c.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f64::max);
        println!("max |ΔE_c| per step over the run incl. the flip: {fluct:.2e} Ha");
        assert!(fluct < 1e-5);
        // a refit clears the audit flag (whatever the residuals then say)
        let s = surrogate_of(&mut d);
        s.refit().unwrap();
        assert!(s.audit_failed().is_none());
        println!("after the refit over the drifted samples: trusted {} ({:?})", s.trusted(), s.training_residuals());
    }

    /// Task 5 carry-over (Task 4 re-review): a failed audit on the refresh evaluation right after
    /// a refit must book the refit's pending jump AND the audit's gap — the reported energy at the
    /// unchanged positions moves from the old model's to the teacher's, which is both.
    #[test]
    fn a_failed_audit_on_the_refresh_after_a_refit_books_the_pending_jump_too() {
        let policy = FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() };
        let (s, h) = trained(1.2, 2.2, 40, policy);
        let mut scene = h2(1.5);
        scene.nuclei[1].vel = [2e-4, 0.0, 0.0];
        let mut d = Dynamics::new(scene, Box::new(s), 0.25, 1);
        for _ in 0..5 {
            d.apply(DynCommand::Step(1));
            assert_eq!(d.tick(f64::INFINITY, &mut never()).source, Source::Surrogate);
        }
        let before = d.snapshot();
        {
            let s = surrogate_of(&mut d);
            // a refit that moves the model surface a little (scan samples from a 0.5 % deeper well)
            let mut deeper = MorseDimer::h2();
            deeper.d *= 1.005;
            let mut rng = StdRng::seed_from_u64(9);
            for i in 0..20 {
                let r = 1.2 + i as f64 / 19.0;
                let sc = dimer(r, unit_vector(&mut rng), [0.0; 3], 1, 1);
                s.add_sample(morse_sample(&deeper, &sc), &sc).unwrap();
            }
            s.refit().unwrap();
            assert!(s.trusted(), "{:?}", s.untrusted_reason());
            let pos: Vec<[f64; 3]> = before.scene.nuclei.iter().map(|n| n.pos).collect();
            let pending = s.model().unwrap().predict(&pos, &[1, 1], false).unwrap().energy - before.energy_ks;
            println!("the refit's pending jump at r_last: {pending:+.3e} Ha");
            assert!(pending.abs() > 1e-6, "test premise: the refit moved the surface visibly");
            // every answered step is audited from now on, and the teacher has drifted 10 %:
            // the refresh evaluation is a failing audit
            s.set_policy(FallbackPolicy { audit_every: 1, ..FallbackPolicy::default() });
        }
        h.lock().unwrap().d *= 1.1;
        let refreshed = d.tick(f64::INFINITY, &mut never());
        assert_eq!(refreshed.scene, before.scene, "nothing moved");
        assert!(refreshed.eval.notes.iter().any(|n| n.contains("surrogate audit failed")), "{:?}", refreshed.eval.notes);
        assert_eq!(refreshed.eval.energy_surface(), Surface::Teacher);
        let jump = refreshed.eval.refit_jump.expect("the flip is booked");
        let delta = refreshed.energy_conserved - before.energy_conserved;
        let reported = refreshed.energy_ks - before.energy_ks;
        println!("refresh = failed audit: booked jump {jump:+.6e} Ha, reported energy change {reported:+.6e} Ha, ΔE_c {delta:+.2e}");
        assert!((jump - reported).abs() < 1e-12, "the booked jump is the whole change of the reported energy");
        assert!(delta.abs() < 1e-12, "E_conserved moved by {delta:+.3e} across the refresh");
        assert!(refreshed.notes.iter().all(|n| !n.contains("history restarts")), "{:?}", refreshed.notes);
    }

    /// Ruling S15, fallback thinning (module doc, "Trust recovery"): fallbacks at one geometry
    /// while the model is untrusted for its residuals (not for want of data) keep one sample in
    /// [`THIN_EVERY`]; a geometry far from every stored one is kept at once; while the trust rule
    /// still needs samples (n_samples < Q′) every answer is kept.
    #[test]
    fn thinning_keeps_one_in_ten_near_identical_fallbacks_and_every_far_one() {
        let (mut s, _) = trained(1.2, 2.2, 40, FallbackPolicy::default());
        assert!(s.trusted());
        // untrusted by its residuals (an impossible energy threshold), not by its sample count
        s.set_policy(FallbackPolicy { energy_std_per_atom: 1e-12, ..FallbackPolicy::default() });
        assert!(s.untrusted_reason().unwrap().starts_with("max training energy residual"));
        let n0 = s.samples();
        let mut thinned_notes = 0;
        for _ in 0..25 {
            let r = s.evaluate(&h2(1.6), &mut never()).unwrap();
            assert_eq!(r.stats.source, Source::Fallback);
            thinned_notes += r.stats.notes.iter().filter(|n| n.contains("sample not kept")).count();
        }
        println!("25 fallbacks at r = 1.6: {} kept, {thinned_notes} thinned", s.samples() - n0);
        assert_eq!(s.samples() - n0, 3, "the 1st, 11th and 21st are kept");
        assert_eq!(thinned_notes, 22);
        // far from every stored geometry: kept right away (4 thinned since the last kept one)
        let far = s.nearest_stored(&h2(4.0).nuclei.iter().map(|n| n.pos).collect::<Vec<_>>(), &[1, 1]);
        s.evaluate(&h2(4.0), &mut never()).unwrap();
        println!("r = 4.0 is {far:.1} training spreads from the nearest stored sample: kept");
        assert!(far > THIN_FRACTION * FallbackPolicy::default().max_descriptor_distance);
        assert_eq!(s.samples() - n0, 4);
        // a model that still needs data keeps every answer
        let (mut few, _) = trained(1.2, 2.2, 5, FallbackPolicy::default());
        assert!(few.untrusted_reason().unwrap().starts_with("n_samples"));
        let m0 = few.samples();
        for _ in 0..5 {
            few.evaluate(&h2(1.6), &mut never()).unwrap();
        }
        assert_eq!(few.samples() - m0, 5);
    }

    /// Ruling S15, quarantine and supersession: one stray sample (the teacher's energy at an
    /// in-window geometry 0.1 Ha off, as a glitched solve would give) untrusted the whole model
    /// before fix round 3, for good. The refit quarantines it: trusted again, the sample held
    /// (store, sidecar) but excluded, the window around it still the model's. A fresh teacher
    /// answer at that geometry agrees with the model, so the stray sample is superseded
    /// (contradicted, still held and excluded, no longer fencing) and the fresh answer kept; the
    /// geometry is the model's again. The codes survive the sidecar; Forget clears them.
    #[test]
    fn a_stray_sample_is_quarantined_then_superseded_and_trust_returns() {
        let (mut s, h) = trained(1.2, 2.2, 40, FallbackPolicy::default());
        let stray = dimer(1.63, [0.0, 1.0, 0.0], [0.1, 0.0, 0.0], 1, 1);
        let mut bad = morse_sample(&MorseDimer::h2(), &stray);
        bad.energy += 0.1;
        s.add_sample(bad, &stray).unwrap();
        let rep = s.refit().unwrap();
        println!("after the stray sample: {} samples, rmse E/atom {:.1e}; trusted {}; quarantined {}; note {:?}",
                 rep.n_samples, rep.rmse_energy_per_atom, s.trusted(), s.quarantined(), s.recovery_note());
        assert!(s.trusted(), "{:?}", s.untrusted_reason());
        assert_eq!(s.quarantined(), 1);
        assert!(s.is_quarantined(40), "the stray sample (index 40) is the one held back");
        assert!(s.recovery_note().unwrap().contains("1 samples quarantined"));
        assert_eq!(s.samples(), 41, "never silently dropped");
        assert_eq!(rep.n_samples, 40, "excluded from the fit");
        let near = |s: &mut SurrogateKs, r: f64| {
            s.quarantine_distances(&h2(r).nuclei.iter().map(|n| n.pos).collect::<Vec<_>>(), &[1, 1]).into_iter().map(|x| x.1).fold(f64::INFINITY, f64::min)
        };
        let (d18, d20) = (near(&mut s, 1.8), near(&mut s, 2.0));
        let r18 = s.evaluate(&h2(1.8), &mut never()).unwrap();
        println!("from the quarantined geometry: r = 1.8 at {d18:.2}, r = 2.0 at {d20:.2} training spreads; r = 1.8 answered by {:?}", r18.stats.source);
        assert_eq!(r18.stats.source, Source::Surrogate, "the window around the stray sample is still the model's");
        // the code in the sidecar: a reloaded model fences the geometry too
        let path = std::env::temp_dir().join(format!("quarantine-{}.model", std::process::id()));
        s.save_sidecar(&path).unwrap();
        let mut back = SurrogateKs::new(Box::new(MorseDimer::h2()), FallbackPolicy::default()).with_teacher_tol(1e-6);
        back.install_sidecar(surrogate::sidecar::read(&path).unwrap()).unwrap();
        assert_eq!((back.quarantined(), back.is_quarantined(40), back.trusted()), (1, true, true));
        // the geometry falls back; the teacher (an exact Morse) agrees with the model there, so
        // the stray sample is superseded and the fresh answer kept
        let calls0 = calls(&h);
        let r = s.evaluate(&stray, &mut never()).unwrap();
        println!("at the stray geometry: {:?}, notes {:?}", r.stats.source, r.stats.notes);
        assert_eq!(r.stats.source, Source::Fallback);
        assert!(r.stats.notes.iter().any(|n| n.contains("quarantined geometry")), "{:?}", r.stats.notes);
        assert!(r.stats.notes.iter().any(|n| n.contains("0 released, 1 superseded")), "{:?}", r.stats.notes);
        assert_eq!((s.samples(), calls(&h), s.quarantined(), s.superseded()), (42, calls0 + 1, 0, 1));
        s.refit().unwrap();
        assert!(s.trusted());
        assert_eq!(s.evaluate(&stray, &mut never()).unwrap().stats.source, Source::Surrogate, "no longer fenced");
        let _ = std::fs::remove_file(&path);
        // Forget clears it
        s.forget();
        assert_eq!((s.quarantined(), s.superseded()), (0, 0));
    }

    /// A long-range Morse teacher (a = 0.3: at R = 8 the pair still binds by 4.4e-2 Ha, while
    /// every feature of the 6 Bohr cutoff model has vanished and it can only say Σ E_ref) — a
    /// geometry genuinely unrepresentable. Task 8 fix round 4 (the re-review's Important): the
    /// envelope rule fired first there, the near-quarantine check was skipped, every 10th
    /// answer was kept and quarantined in turn until the 10 % cap untrusted the model for good.
    /// Now 600 teacher steps beyond the cutoff (R 7–10), with a refit every 25 as the worker
    /// does, keep the trust, and R = 1.6 is the model's. Since Task 9 fix round 1 (fence voting,
    /// I3) the store grows by exactly one sample: the first three mutually consistent answers
    /// that disagree with the model are a vote, the next refit cannot reproduce the voted sample,
    /// quarantines it and closes the region to further votes — the chain stays bounded.
    #[test]
    fn a_molecule_parked_beyond_the_cutoff_keeps_one_vote_and_never_untrusts_the_model() {
        let long = MorseDimer { a: 0.3, ..MorseDimer::h2() };
        let teacher = Shared::new(MorseDimer { a: 0.3, ..MorseDimer::h2() });
        let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_teacher_tol(1e-6);
        s.set_reference_energy(1, long.e_atom);
        let mut rng = StdRng::seed_from_u64(9);
        for i in 0..40 {
            let r = 1.2 + 1.0 * i as f64 / 39.0;
            let sc = dimer(r, unit_vector(&mut rng), [0.0; 3], 1, 1);
            s.add_sample(morse_sample(&long, &sc), &sc).unwrap();
        }
        s.refit().unwrap();
        assert!(s.trusted(), "{:?}", s.untrusted_reason());
        // the first answer beyond the cutoff: kept (far from everything), then quarantined
        let r = s.evaluate(&h2(8.0), &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback);
        s.refit().unwrap();
        let (n1, q1) = (s.samples(), s.quarantined());
        println!("after R = 8: {n1} samples, {q1} quarantined, trusted {}, note {:?}", s.trusted(), s.recovery_note());
        assert!(s.trusted() && q1 == 1, "{:?}", s.untrusted_reason());
        let mut rejected_vote = false;
        for k in 0..600 {
            let r = 7.0 + 3.0 * ((k as f64) * 0.37).sin().abs();
            let ev = s.evaluate(&h2(r), &mut never()).unwrap();
            assert_eq!(ev.stats.source, Source::Fallback, "R = {r} beyond the cutoff");
            if k % 25 == 24 && s.pending_refit() {
                s.refit().unwrap();
                rejected_vote |= s.recovery_note().is_some_and(|n| n.contains("1 voted samples quarantined"));
            }
        }
        println!("after 600 steps beyond the cutoff: {} samples, {} quarantined, trusted {}", s.samples(), s.quarantined(), s.trusted());
        assert!(rejected_vote, "the refit judged the vote");
        assert_eq!((s.samples(), s.quarantined()), (n1 + 1, q1 + 1), "one vote, quarantined; nothing more kept");
        assert!(s.trusted(), "{:?}", s.untrusted_reason());
        assert_eq!(s.evaluate(&h2(1.6), &mut never()).unwrap().stats.source, Source::Surrogate);
    }

    /// Release at a refit (module doc, "Release"): a sample 5e-3 Ha off (2.5e-3 Ha/atom) is
    /// quarantined under the default 1e-3 Ha/atom threshold; once the policy accepts 1e-2 the
    /// next refit finds the model reproducing it, releases it into the fit and stays trusted.
    /// (A Morse tail beyond what the cutoff can carry was tried first and is never reproduced:
    /// that is what quarantine is for.)
    #[test]
    fn a_quarantined_sample_is_released_when_the_model_reproduces_it() {
        let (mut s, _) = trained(1.2, 2.2, 40, FallbackPolicy::default());
        let off = dimer(1.7, [0.0, 0.0, 1.0], [0.0; 3], 1, 1);
        let mut sample = morse_sample(&MorseDimer::h2(), &off);
        sample.energy += 5e-3;
        s.add_sample(sample, &off).unwrap();
        s.refit().unwrap();
        println!("5e-3 Ha off: trusted {}, quarantined {}, note {:?}", s.trusted(), s.quarantined(), s.recovery_note());
        assert!(s.trusted() && s.is_quarantined(40), "{:?}", s.untrusted_reason());
        s.set_policy(FallbackPolicy { energy_std_per_atom: 1e-2, ..FallbackPolicy::default() });
        let rep = s.refit().unwrap();
        println!("threshold 1e-2: trusted {}, quarantined {}, fitted {} samples, note {:?}", s.trusted(), s.quarantined(), rep.n_samples, s.recovery_note());
        assert!(s.trusted(), "{:?}", s.untrusted_reason());
        assert_eq!((s.quarantined(), rep.n_samples), (0, 41));
        assert!(s.recovery_note().unwrap().contains("1 quarantined samples released"), "{:?}", s.recovery_note());
    }

    /// A Morse teacher that reports a face density of the test's choosing (the number a grid
    /// solver puts in `EvalStats::face_density`).
    struct FaceDensity {
        inner: Arc<Mutex<MorseDimer>>,
        rho: Arc<Mutex<f64>>,
    }

    impl ForceProvider for FaceDensity {
        fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
            let mut r = self.inner.lock().unwrap().evaluate(scene, should_abort)?;
            r.stats.face_density = Some(*self.rho.lock().unwrap());
            Ok(r)
        }
        fn name(&self) -> &'static str {
            "morse dimer"
        }
        fn reset(&mut self) {}
    }

    /// Box-affected answers (fix round 4; Task 9: the solver's number, `EvalStats::face_density`,
    /// not its warning text): a fallback answer whose teacher saw more than
    /// `BOX_AFFECTED_FACE_DENSITY` on the faces is used for the step and not kept; one below it is
    /// kept. The geometry (R = 2.6 against training on [1.1, 2.0]) is out of the envelope and far
    /// from every stored sample, so nothing else would drop it.
    #[test]
    fn a_box_affected_answer_is_used_but_not_trained_on() {
        let rho = Arc::new(Mutex::new(0.0));
        let teacher = FaceDensity { inner: Arc::new(Mutex::new(MorseDimer::h2())), rho: rho.clone() };
        let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_teacher_tol(1e-6);
        s.set_reference_energy(1, MorseDimer::h2().e_atom);
        let mut rng = StdRng::seed_from_u64(4);
        for i in 0..40 {
            let r = 1.1 + 0.9 * i as f64 / 39.0;
            let origin = [rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3)];
            let sc = dimer(r, unit_vector(&mut rng), origin, 1, 1);
            s.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
        }
        s.refit().unwrap();
        assert!(s.trusted(), "{:?}", s.untrusted_reason());
        let n0 = s.samples();
        *rho.lock().unwrap() = 2.0 * BOX_AFFECTED_FACE_DENSITY;
        let r = s.evaluate(&h2(2.6), &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback);
        assert_eq!(r.stats.face_density, Some(2.0 * BOX_AFFECTED_FACE_DENSITY));
        assert!(r.stats.notes.iter().any(|n| n.contains("box-affected")), "{:?}", r.stats.notes);
        assert_eq!(s.samples(), n0, "a box-affected answer was kept");
        *rho.lock().unwrap() = 0.5 * BOX_AFFECTED_FACE_DENSITY;
        let r = s.evaluate(&h2(2.6), &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback);
        assert_eq!(s.samples(), n0 + 1, "an answer inside the box rule was not kept: {:?}", r.stats.notes);
    }

    /// Final review B2 (module doc, "No model yet"): with no model (a pass offer waiting, a refused
    /// autoload) every DFT step used to keep a sample with no refit and no bound. Up to
    /// NO_MODEL_SAMPLE_FACTOR · Q′ (4 × 29 for H at D = 4) they are kept; beyond, one in
    /// THIN_EVERY, with a note saying how many are held.
    #[test]
    fn without_a_model_the_kept_answers_are_capped() {
        let mut s = SurrogateKs::new(Box::new(MorseDimer::h2()), FallbackPolicy::default()).with_teacher_tol(1e-6);
        let cap = NO_MODEL_SAMPLE_FACTOR * Basis::new(&[1]).column_classes().len();
        assert_eq!(cap, 116);
        let mut last = None;
        for i in 0..cap + 34 {
            last = Some(s.evaluate(&h2(1.2 + 0.004 * i as f64), &mut never()).unwrap());
        }
        // 116 kept, then of 34 more: 3 × (9 dropped, 1 kept) + 4 dropped
        assert_eq!(s.samples(), cap + 3);
        let notes = last.unwrap().stats.notes;
        assert!(notes.iter().any(|n| n.contains("waiting for a sampling pass — 119 samples held")), "{notes:?}");
    }

    /// The auditor's teacher for the asynchronous-audit tests: the same (shared, drift-able) Morse
    /// as the main teacher, answering only once `open` is set (or abandoning the solve when the
    /// surrogate asks), and reporting a face density of the test's choosing.
    struct Gated {
        inner: Arc<Mutex<MorseDimer>>,
        open: Arc<std::sync::atomic::AtomicBool>,
        rho: f64,
        calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl ForceProvider for Gated {
        fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
            use std::sync::atomic::Ordering::SeqCst;
            while !self.open.load(SeqCst) {
                if should_abort() {
                    return Err(ProviderError::Cancelled);
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            self.calls.fetch_add(1, SeqCst);
            // a stand-in density (side densities: what the app shows), tagged with the nuclei
            let display = Some(Display::Cpu { cell: scene.cell.clone(), origin: [0.0; 3], rho: Arc::new(vec![scene.nuclei.len() as f32]), max: 1.0 });
            if scene.nuclei.len() == 1 {
                // an isolated atom (the Morse needs two): its reference energy, no force
                let e = self.inner.lock().unwrap().e_atom;
                return Ok(ForceResult { energy: e, forces: vec![[0.0; 3]], stats: EvalStats::default(), display });
            }
            let mut r = self.inner.lock().unwrap().evaluate(scene, should_abort)?;
            r.stats.face_density = Some(self.rho);
            r.display = display;
            Ok(r)
        }
        fn name(&self) -> &'static str {
            "morse dimer"
        }
        fn reset(&mut self) {}
    }

    /// A trained H₂ surrogate (audit every `every`) with a gated background auditor on the same
    /// Morse; returns it with the teacher handle, the gate and the auditor's call counter.
    #[allow(clippy::type_complexity)]
    fn with_gated_auditor(every: usize, rho: f64) -> (SurrogateKs, Arc<Mutex<MorseDimer>>, Arc<std::sync::atomic::AtomicBool>, Arc<std::sync::atomic::AtomicUsize>) {
        let (s, h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: every, ..FallbackPolicy::default() });
        let open = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let auditor = Gated { inner: h.clone(), open: open.clone(), rho, calls: calls.clone() };
        (s.with_auditor(Box::new(auditor)), h, open, calls)
    }

    /// Steps `d` one step at a time until `f` holds (≤ 2 s; the background answer arrives on its
    /// own thread), returning the snapshots seen.
    fn step_until(d: &mut Dynamics, what: &str, f: impl Fn(&mut Dynamics) -> bool) -> Vec<crate::dynamics::DynSnapshot> {
        let t0 = Instant::now();
        let mut seen = Vec::new();
        while !f(d) {
            assert!(t0.elapsed().as_secs_f64() < 2.0, "{what}: not within 2 s");
            d.apply(DynCommand::Step(1));
            seen.push(d.tick(f64::INFINITY, &mut never()));
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        seen
    }

    /// Ruling S14: with a background auditor the audit step is the model's own answer — the
    /// teacher solve runs on the auditor's thread while the model keeps stepping (here: held
    /// back by a gate for 20 steps), and the verdict is recorded when it arrives, judged by the
    /// current model, with the sample kept and the steps in between counted. A Forget abandons an
    /// audit in flight: its answer is discarded.
    #[test]
    fn a_background_audit_runs_while_the_model_answers() {
        let (s, h, open, audit_calls) = with_gated_auditor(5, 0.0);
        assert!(s.audits_asynchronous());
        let mut scene = h2(1.5);
        scene.nuclei[1].vel = [2e-4, 0.0, 0.0];
        let mut d = Dynamics::new(scene, Box::new(s), 0.25, 1); // answered step 1
        let teacher_calls = calls(&h);
        let mut dispatched = None;
        for i in 0..24 {
            d.apply(DynCommand::Step(1));
            let snap = d.tick(f64::INFINITY, &mut never());
            assert_eq!(snap.source, Source::Surrogate, "step {i}: {:?}", snap.eval.notes);
            if snap.eval.notes.iter().any(|n| n.contains("surrogate audit dispatched")) {
                dispatched.get_or_insert(i);
            }
        }
        assert_eq!(dispatched, Some(3), "the 5th answered step dispatches the audit");
        let sk = surrogate_of(&mut d);
        assert!(sk.audit_in_flight() && sk.last_audit().is_none());
        assert_eq!(calls(&h), teacher_calls, "the main teacher was never called");
        let n0 = sk.samples();
        // the gate opens: the verdict arrives with a later step
        open.store(true, std::sync::atomic::Ordering::SeqCst);
        let seen = step_until(&mut d, "the background verdict", |d| surrogate_of(d).last_audit().is_some());
        let note = seen.last().unwrap().eval.notes.iter().find(|n| n.contains("surrogate audit passed")).cloned();
        let sk = surrogate_of(&mut d);
        let a = sk.last_audit().unwrap().clone();
        println!("background audit: {a:?}; note {note:?}");
        assert_eq!((a.verdict, a.asynchronous), (AuditVerdict::Passed, true));
        assert!(a.gap.abs() < 1e-5 && a.unaudited_steps >= 20, "{a:?}");
        assert!(note.is_some(), "{:?}", seen.last().unwrap().eval.notes);
        assert_eq!(sk.samples(), n0 + 1, "an audit's answer is a sample");
        assert!(sk.trusted() && sk.pending_refit());
        assert!(seen.iter().all(|s| s.source == Source::Surrogate));
        // a Forget abandons the audit in flight (gate shut again; the next one is dispatched)
        open.store(false, std::sync::atomic::Ordering::SeqCst);
        step_until(&mut d, "the next dispatch", |d| surrogate_of(d).audit_in_flight());
        let before = audit_calls.load(std::sync::atomic::Ordering::SeqCst);
        surrogate_of(&mut d).forget();
        open.store(true, std::sync::atomic::Ordering::SeqCst);
        std::thread::sleep(std::time::Duration::from_millis(50));
        d.apply(DynCommand::Step(1));
        d.tick(f64::INFINITY, &mut never());
        let sk = surrogate_of(&mut d);
        assert!(sk.last_audit().is_none() && !sk.audit_in_flight(), "{:?}", sk.last_audit());
        assert_eq!(audit_calls.load(std::sync::atomic::Ordering::SeqCst), before, "the abandoned job never reached the teacher");
    }

    /// Draft mode (module doc): an out-of-envelope geometry is answered by the model — the
    /// teacher never called — with the extrapolation flagged (atoms outside, the rule that would
    /// have fallen back) and the per-atom uncertainty ratios; the same geometry falls back with
    /// draft mode off; an untrusted fit answers too, flagged with the trust rule's reason; a
    /// provenance or species mismatch still goes to the teacher (nothing to guess with).
    #[test]
    fn draft_mode_answers_beyond_the_envelope_and_flags_it() {
        let draft = FallbackPolicy { audit_every: 0, never_fallback: true, ..FallbackPolicy::default() };
        let (mut s, h) = trained(1.2, 2.2, 40, draft);
        assert!(s.trusted());
        let c0 = calls(&h);
        let r = s.evaluate(&h2(4.0), &mut never()).unwrap();
        let x = r.stats.extrapolation.clone().expect("flagged");
        println!("draft at R = 4: {:?}; per-atom uncertainty {:?}", x, r.stats.atom_uncertainty);
        assert_eq!(r.stats.source, Source::Surrogate);
        assert!(x.atoms_outside == 2 && x.untrusted.is_none() && x.atom_envelope.iter().all(|&d| d > 1.0), "{x:?}");
        assert!(x.reason.as_deref().is_some_and(|m| m.contains("outside the descriptor envelope")), "{x:?}");
        assert_eq!(r.stats.atom_uncertainty.as_ref().map(Vec::len), Some(2));
        assert_eq!(calls(&h), c0, "the teacher was never called");
        // inside the envelope: an ordinary answer, nothing flagged, the ratios below 1
        let r = s.evaluate(&h2(1.6), &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Surrogate);
        assert!(r.stats.extrapolation.is_none());
        assert!(r.stats.atom_uncertainty.as_ref().unwrap().iter().all(|&u| u < 1.0), "{:?}", r.stats.atom_uncertainty);
        // draft off: R = 4 is the teacher's
        s.set_policy(FallbackPolicy { never_fallback: false, ..*s.policy() });
        assert_eq!(s.evaluate(&h2(4.0), &mut never()).unwrap().stats.source, Source::Fallback);
        // mismatches still refuse in draft mode: another grid (rule 3), another species (rule 2)
        s.set_policy(FallbackPolicy { never_fallback: true, ..*s.policy() });
        let mut other = h2(1.6);
        other.cell = Cell { n: 41, h: 0.25 };
        let r = s.evaluate(&other, &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback, "{:?}", r.stats.notes);
        assert!(r.stats.extrapolation.is_none());
        let r = s.evaluate(&dimer(1.6, [1.0, 0.0, 0.0], [0.0; 3], 1, 2), &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback, "{:?}", r.stats.notes);
        // an untrusted fit (10 samples < Q′) answers, flagged with the trust rule's reason
        let (mut u, _) = trained(1.2, 2.2, 10, draft);
        assert!(u.model().is_some() && !u.trusted());
        let r = u.evaluate(&h2(1.6), &mut never()).unwrap();
        let x = r.stats.extrapolation.clone().expect("flagged");
        println!("draft, untrusted: {x:?}");
        assert_eq!(r.stats.source, Source::Surrogate);
        assert!(x.untrusted.as_deref().is_some_and(|m| m.contains("untrusted")), "{x:?}");
    }

    /// Review M7 (fix round 1): paused with an audit in flight, a side request is not stuck until
    /// Play — the audit's answer is taken outside `evaluate` and held unjudged, the queued atom job
    /// goes out and arrives, and the audit's verdict is taken at the next evaluate.
    #[test]
    fn a_side_request_made_while_paused_behind_an_audit_still_runs() {
        let (s, _h, open, _) = with_gated_auditor(5, 0.0);
        let mut scene = h2(1.5);
        scene.nuclei[1].vel = [2e-4, 0.0, 0.0];
        let mut d = Dynamics::new(scene, Box::new(s), 0.25, 1);
        step_until(&mut d, "an audit in flight", |d| surrogate_of(d).audit_in_flight());
        let sc = d.snapshot().scene;
        assert!(surrogate_of(&mut d).request_atom_density(1, &sc), "queued behind the audit");
        open.store(true, std::sync::atomic::Ordering::SeqCst);
        let t0 = std::time::Instant::now();
        let atoms = loop {
            let a = surrogate_of(&mut d).take_atom_densities(); // no tick, no evaluate
            if !a.is_empty() {
                break a;
            }
            assert!(t0.elapsed().as_secs_f64() < 2.0, "the atom job never went out while paused");
            std::thread::sleep(std::time::Duration::from_millis(1));
        };
        assert_eq!(atoms[0].0, 1);
        let sk = surrogate_of(&mut d);
        assert!(sk.last_audit().is_none() && sk.audit_in_flight(), "the audit's answer is held, not judged");
        d.apply(DynCommand::Step(1));
        d.tick(f64::INFINITY, &mut never());
        assert!(surrogate_of(&mut d).last_audit().is_some(), "judged at the next evaluate");
    }

    /// Review I1 (fix round 1): in draft mode a refit while the model is untrusted still moves the
    /// surface the atoms are on, so its jump is booked like a trusted refit's — the driver's
    /// refresh at the unchanged positions books `refit_jump` into W and E_conserved is continuous
    /// to rounding (pinned: |ΔE_c| ≤ 1e-12 Ha against a jump of ≥ 1e-4 Ha); without the fix the
    /// jump went unbooked (Model → Model, no restart, no note).
    #[test]
    fn draft_mode_books_the_jump_of_an_untrusted_refit() {
        let draft = FallbackPolicy { audit_every: 0, never_fallback: true, ..FallbackPolicy::default() };
        let (s, _h) = trained(1.2, 2.2, 10, draft);
        assert!(s.model().is_some() && !s.trusted(), "the premise: an untrusted fit");
        let mut scene = h2(1.6);
        scene.nuclei[1].vel = [2e-4, 0.0, 0.0];
        let mut d = Dynamics::new(scene, Box::new(s), 0.25, 1);
        d.apply(DynCommand::Step(20));
        let before = d.tick(f64::INFINITY, &mut never());
        assert_eq!(before.source, Source::Surrogate, "draft answers the untrusted model");
        // five labels from a deeper well: the refit moves the surface, still short of Q′ (untrusted)
        let deeper = MorseDimer { d: 0.2, ..MorseDimer::h2() };
        let sk = surrogate_of(&mut d);
        for k in 0..5 {
            let sc = h2(1.3 + 0.15 * k as f64);
            sk.add_sample(morse_sample(&deeper, &sc), &sc).unwrap();
        }
        sk.refit().expect("refit");
        assert!(!sk.trusted(), "still untrusted after the refit");
        let after = d.tick(f64::INFINITY, &mut never()); // paused: the refresh at the same positions
        let jump = after.eval.refit_jump;
        println!("untrusted draft refit: jump {jump:?}, ΔE_conserved {:.2e} Ha, positions unchanged {}",
                 after.energy_conserved - before.energy_conserved, after.scene == before.scene);
        assert_eq!(after.scene, before.scene);
        assert_eq!(after.source, Source::Surrogate);
        assert!(jump.is_some_and(|j| j.abs() >= 1e-4), "the jump is booked: {jump:?}");
        assert!((after.energy_conserved - before.energy_conserved).abs() <= 1e-12, "{:.2e}", after.energy_conserved - before.energy_conserved);
        assert!(after.work_refit != before.work_refit, "booked into W as a surface jump");
    }

    /// Draft mode keeps learning (module doc): the background audits run while the model answers
    /// every step — out of the envelope here — their answers are kept as samples and a refit
    /// runs on them; a failed audit (the teacher drifted) untrusts the model, which goes on
    /// answering; switching draft mode off then hands the step to the teacher with the flip
    /// booked (`refit_jump`), so E_conserved does not jump by the gap.
    #[test]
    fn draft_mode_audits_keep_teaching_and_never_block() {
        let (mut s, h, open, audit_calls) = with_gated_auditor(5, 0.0);
        s.set_policy(FallbackPolicy { never_fallback: true, ..*s.policy() });
        open.store(true, std::sync::atomic::Ordering::SeqCst);
        let mut scene = h2(3.2); // outside the trained 1.2–2.2
        scene.nuclei[1].vel = [1e-4, 0.0, 0.0];
        let mut d = Dynamics::new(scene, Box::new(s), 0.25, 1);
        let n0 = surrogate_of(&mut d).samples();
        let seen = step_until(&mut d, "a draft-mode audit verdict", |d| surrogate_of(d).last_audit().is_some());
        assert!(seen.iter().all(|x| x.source == Source::Surrogate), "every step is the model's");
        assert!(seen.iter().any(|x| x.eval.extrapolation.as_ref().is_some_and(|e| e.atoms_outside > 0)));
        // the history marks the guessed steps (review M2: the sparkline draws them red)
        let last = seen.last().unwrap();
        assert_eq!(last.history_guess.len(), last.history.len());
        assert!(last.history_guess.iter().any(|&g| g));
        assert!(audit_calls.load(std::sync::atomic::Ordering::SeqCst) >= 1);
        let sk = surrogate_of(&mut d);
        assert!(sk.samples() > n0 && sk.pending_refit(), "the audit's answer is a sample");
        sk.refit().expect("a refit on it");
        // the teacher drifts: the next audit fails, the model is untrusted and still answers
        h.lock().unwrap().d *= 1.1;
        step_until(&mut d, "a failed audit", |d| surrogate_of(d).last_audit().is_some_and(|a| a.verdict == AuditVerdict::Failed));
        assert!(!surrogate_of(&mut d).trusted());
        d.apply(DynCommand::Step(3));
        let t = d.tick(f64::INFINITY, &mut never());
        assert_eq!(t.source, Source::Surrogate, "untrusted, still answering in draft mode");
        // draft off: the teacher's step, the flip booked at its positions
        let e0 = t.energy_conserved;
        let p = FallbackPolicy { never_fallback: false, ..*surrogate_of(&mut d).policy() };
        surrogate_of(&mut d).set_policy(p);
        d.apply(DynCommand::Step(1));
        let t = d.tick(f64::INFINITY, &mut never());
        println!("draft off after a failed audit: {:?}, refit_jump {:?}, ΔE_conserved {:.2e} Ha", t.source, t.eval.refit_jump, t.energy_conserved - e0);
        assert_eq!(t.source, Source::Fallback);
        assert!(t.eval.refit_jump.is_some(), "the flip is booked");
        // booked: E_conserved moves by the step's mixed forces only (first half-kick on the
        // model's, second on the teacher's: O(ΔF·Δr), the 10 % drift of a 0.05 Ha/Bohr force over
        // the step's ~0.05 Bohr), not by the 8e-3 Ha flip itself
        let jump = t.eval.refit_jump.unwrap();
        assert!((t.energy_conserved - e0).abs() < 0.1 * jump.abs(), "{:.2e} vs the flip {jump:.2e}", t.energy_conserved - e0);
    }

    /// Side densities (module doc): a background audit's density reaches the driver's snapshot
    /// as `SideKind::Audit`, stamped with the simulated time of the geometry it was dispatched at
    /// (the positions it carries are that snapshot's); a refresh asked for between ticks is at
    /// the current positions and time, display-only (no sample, no verdict); an isolated-atom
    /// density asked for while the auditor is busy is queued and arrives through
    /// `take_atom_densities` after it. An edit clears the side density.
    #[test]
    fn side_densities_carry_their_kind_time_and_positions() {
        let (s, _h, open, _) = with_gated_auditor(5, 0.0);
        let mut scene = h2(1.5);
        scene.nuclei[1].vel = [2e-4, 0.0, 0.0];
        let mut d = Dynamics::new(scene, Box::new(s), 0.25, 1);
        open.store(true, std::sync::atomic::Ordering::SeqCst);
        let seen = step_until(&mut d, "an audit density", |d| d.snapshot().side_display.is_some());
        let side = d.snapshot().side_display.unwrap();
        assert_eq!(side.kind, SideKind::Audit);
        let at = seen.iter().find(|s| s.scene.nuclei.iter().map(|n| n.pos).collect::<Vec<_>>() == side.positions)
            .expect("the audited positions are a snapshot's");
        println!("audit density: solved at t = {:.3} fs (step {}), arrived at step {}", side.t_fs(), at.step, seen.last().unwrap().step);
        assert_eq!(side.t, at.sim_time, "stamped with the dispatch geometry's time");
        assert!(matches!(&side.display, Display::Cpu { rho, .. } if rho[0] == 2.0));
        // a refresh between ticks: the current positions and time, display-only
        let n0 = surrogate_of(&mut d).samples();
        let (gap0, seq0) = (surrogate_of(&mut d).last_audit().cloned(), side.seq);
        step_until(&mut d, "an idle auditor", |d| surrogate_of(d).auditor_idle());
        let now = d.snapshot();
        assert!(surrogate_of(&mut d).request_display(&now.scene));
        assert!(!surrogate_of(&mut d).auditor_idle(), "sent");
        // queued behind it (taken after the refresh, below)
        assert!(surrogate_of(&mut d).request_atom_density(1, &now.scene));
        let t0 = std::time::Instant::now();
        let refresh = loop {
            let snap = d.tick(f64::INFINITY, &mut never()); // paused: no step, the poll only
            if let Some(sd) = snap.side_display.filter(|sd| sd.seq != seq0) {
                break sd;
            }
            assert!(t0.elapsed().as_secs_f64() < 2.0, "no refresh within 2 s");
            std::thread::sleep(std::time::Duration::from_millis(1));
        };
        assert_eq!(refresh.kind, SideKind::Refresh);
        assert_eq!(refresh.t, now.sim_time);
        assert_eq!(refresh.positions, now.scene.nuclei.iter().map(|n| n.pos).collect::<Vec<_>>());
        let sk = surrogate_of(&mut d);
        assert_eq!(sk.samples(), n0, "a refresh is not a sample");
        assert_eq!(sk.last_audit().cloned(), gap0, "nor an audit");
        // the queued isolated atom: on the scene's cell, through its own queue
        let t0 = std::time::Instant::now();
        let atoms = loop {
            let a = surrogate_of(&mut d).take_atom_densities();
            if !a.is_empty() {
                break a;
            }
            assert!(t0.elapsed().as_secs_f64() < 2.0, "no atom within 2 s");
            std::thread::sleep(std::time::Duration::from_millis(1));
        };
        assert!(matches!(atoms.as_slice(), [(1, Display::Cpu { cell, rho, .. })] if *cell == now.scene.cell && rho[0] == 1.0), "{atoms:?}");
        // an edit clears it
        d.apply(DynCommand::RemoveNucleus(1));
        assert!(d.snapshot().side_display.is_none());
    }

    /// Ruling S14 + S7: a failed background audit untrusts the model when its verdict arrives;
    /// the next evaluation (a teacher step, rule 4) books the Model → Teacher flip at its own
    /// positions as `refit_jump`, so E_conserved stays continuous, and the note says how many
    /// answered steps went unaudited.
    #[test]
    fn a_failed_background_audit_untrusts_and_books_the_flip() {
        let (s, h, open, _) = with_gated_auditor(5, 0.0);
        // at r_e, where a 10 % deeper well changes the energy by 1.7e-2 Ha but the forces hardly:
        // the one mixed-force half-kick of the flip step (see the synchronous test) stays small
        let mut scene = h2(1.4);
        scene.nuclei[1].vel = [2e-4, 0.0, 0.0];
        let mut d = Dynamics::new(scene, Box::new(s), 0.25, 1);
        h.lock().unwrap().d *= 1.1; // the teacher drifts away from the model
        open.store(true, std::sync::atomic::Ordering::SeqCst);
        let mut e_c = vec![d.snapshot().energy_conserved];
        let mut flip = None;
        let t0 = Instant::now();
        while flip.is_none() {
            assert!(t0.elapsed().as_secs_f64() < 2.0, "no flip within 2 s");
            d.apply(DynCommand::Step(1));
            let snap = d.tick(f64::INFINITY, &mut never());
            e_c.push(snap.energy_conserved);
            if snap.source == Source::Fallback {
                flip = Some(snap);
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let snap = flip.unwrap();
        let jump = snap.eval.refit_jump.expect("the flip is booked");
        let delta = e_c[e_c.len() - 1] - e_c[e_c.len() - 2];
        println!("failed background audit: jump {jump:+.3e} Ha, ΔE_c across the flip {delta:+.2e} Ha, notes {:?}", snap.eval.notes);
        assert!(snap.eval.notes.iter().any(|n| n.contains("surrogate audit failed (background teacher)") && n.contains("unaudited")), "{:?}", snap.eval.notes);
        assert_eq!(snap.eval.energy_surface(), Surface::Teacher);
        assert!(jump.abs() / 2.0 > 1e-3, "test premise: the drift exceeds the per-atom bound");
        // the mixed-force half-kick remainder only, as for the synchronous flip
        assert!(delta.abs() < 1e-5, "E_conserved saw the flip: {delta:+.2e}");
        assert!(snap.notes.iter().all(|n| !n.contains("history restarts")), "{:?}", snap.notes);
        let sk = surrogate_of(&mut d);
        assert!(!sk.trusted() && sk.audit_failed().is_some());
        assert_eq!(sk.last_audit().map(|a| a.verdict), Some(AuditVerdict::Failed));
        // a refit clears it
        sk.refit().unwrap();
        assert!(sk.audit_failed().is_none());
    }

    /// Task 9 item 1: an audit whose teacher answer is box-affected is recorded but not judged
    /// — its gap is the box's confinement, which a translation-invariant model cannot learn —
    /// and its answer is not kept: the model stays trusted and answers on, nothing is
    /// quarantined. (Before: the audit failed on the confinement, the refit quarantined the
    /// confined sample, and the quarantine fenced the equilibrium bond everywhere in the cell —
    /// the user's "trusted, 100 % fallback, stable gap".) Both audit paths.
    #[test]
    fn a_box_affected_audit_is_not_judged_and_not_kept() {
        for background in [false, true] {
            let rho = 2.0 * BOX_AFFECTED_FACE_DENSITY;
            let (mut d, h) = if background {
                let (s, h, open, _) = with_gated_auditor(5, rho);
                open.store(true, std::sync::atomic::Ordering::SeqCst);
                (Dynamics::new(h2(1.5), Box::new(s), 0.25, 1), h)
            } else {
                let (s, h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 5, ..FallbackPolicy::default() });
                // the synchronous path: the main teacher reports the face density
                let teacher = FaceDensity { inner: h.clone(), rho: Arc::new(Mutex::new(rho)) };
                let mut s2 = SurrogateKs::new(Box::new(teacher), FallbackPolicy { audit_every: 5, ..FallbackPolicy::default() }).with_teacher_tol(1e-6);
                s2.set_reference_energy(1, MorseDimer::h2().e_atom);
                for smp in s.store().unwrap().samples.clone() {
                    let sc = Scene { nuclei: smp.pos.iter().zip(&smp.z).map(|(p, &z)| Nucleus::new(z, *p)).collect(), ..h2(1.5) };
                    s2.add_sample(smp, &sc).unwrap();
                }
                s2.refit().unwrap();
                (Dynamics::new(h2(1.5), Box::new(s2), 0.25, 1), h)
            };
            h.lock().unwrap().d *= 1.05; // the "box" moves the teacher 5 % off the model
            let n0 = surrogate_of(&mut d).samples();
            let seen = step_until(&mut d, "the audit verdict", |d| surrogate_of(d).last_audit().is_some());
            for _ in 0..10 {
                d.apply(DynCommand::Step(1));
                d.tick(f64::INFINITY, &mut never());
            }
            let sk = surrogate_of(&mut d);
            let a = sk.last_audit().unwrap().clone();
            println!("background {background}: {a:?}");
            assert_eq!(a.verdict, AuditVerdict::BoxAffected);
            assert!(a.gap.abs() / 2.0 > 1e-3, "test premise: the gap alone would fail the audit");
            assert!(sk.trusted() && sk.audit_failed().is_none(), "{:?}", sk.untrusted_reason());
            assert_eq!(sk.samples(), n0, "a box-affected answer is not trained on");
            sk.refit().ok();
            assert!(sk.trusted() && sk.quarantined() == 0);
            let notes: Vec<String> = seen.iter().flat_map(|s| s.eval.notes.clone()).collect();
            assert!(notes.iter().any(|n| n.contains("surrogate audit not judged")), "{notes:?}");
            d.apply(DynCommand::Step(1));
            assert_eq!(d.tick(f64::INFINITY, &mut never()).source, Source::Surrogate);
        }
    }

    /// Ruling (Task 9 item 1): a trusted model that falls back on every consultation says so —
    /// after `STUCK_STEPS` consecutive fallbacks `stuck()` names the rule and the gap — and the
    /// next answer clears it.
    #[test]
    fn a_trusted_model_that_keeps_falling_back_says_so() {
        let (mut s, _h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
        // R = 2.8 is outside the envelope: every consultation there falls back
        for i in 0..STUCK_STEPS {
            assert!(s.stuck().is_none(), "after {i} fallbacks");
            let r = s.evaluate(&h2(2.8), &mut never()).unwrap();
            assert_eq!(r.stats.source, Source::Fallback);
        }
        let why = s.stuck().expect("stuck after STUCK_STEPS fallbacks");
        println!("{why}");
        assert!(s.trusted());
        assert!(why.contains("descriptor envelope") && why.contains("teacher − model"), "{why}");
        assert_eq!(s.evaluate(&h2(1.6), &mut never()).unwrap().stats.source, Source::Surrogate);
        assert!(s.stuck().is_none());
    }

    /// Task 9 item 5, the cadence invariant: with `variance_cadence` = k the variance is evaluated
    /// on the first consultation after an edit (`discontinuity` / `reset`), a refit or a policy
    /// change, never more than k consultations apart otherwise, and afresh after every rejection
    /// (a rejecting verdict is never reused).
    #[test]
    fn the_variance_is_checked_at_least_every_k_consultations_and_after_every_edit() {
        let k = 4;
        let (mut s, _h) = trained(1.2, 2.2, 40, FallbackPolicy { variance_cadence: k, audit_every: 0, ..FallbackPolicy::default() });
        let scene = h2(1.6);
        let checked = |s: &mut SurrogateKs| {
            let before = s.variance_checks();
            let r = s.evaluate(&scene, &mut never()).unwrap();
            (s.variance_checks() > before, r.stats.source)
        };
        // the first consultation after the refit checks; then at most k apart
        let pattern: Vec<bool> = (0..13).map(|_| checked(&mut s).0).collect();
        println!("checks over 13 consultations at k = {k}: {pattern:?}");
        assert!(pattern[0], "the first consultation after a refit checks");
        let mut gap = 0;
        for &c in &pattern {
            gap = if c { 0 } else { gap + 1 };
            assert!(gap < k, "{k} or more consultations without a check: {pattern:?}");
        }
        assert_eq!(pattern.iter().filter(|&&c| c).count(), 4, "k = {k}: checks at 0, 4, 8, 12 (no more than needed)");
        // an edit, a refit, a policy change: the next consultation checks
        checked(&mut s);
        s.discontinuity();
        assert!(checked(&mut s).0, "after discontinuity()");
        checked(&mut s);
        s.reset();
        assert!(checked(&mut s).0, "after reset()");
        checked(&mut s);
        s.refit().unwrap();
        assert!(checked(&mut s).0, "after a refit");
        checked(&mut s);
        s.set_policy(*s.policy());
        assert!(checked(&mut s).0, "after a policy change");
        // a rejecting verdict is never reused: every consultation checks. A force bound (1e-5
        // Ha/Bohr) above the model's worst training residual — trust holds — but below its
        // uncertainty in the stretched tail (envelope off, so the variance rule is what refuses)
        let residual = s.training_residuals().unwrap().force;
        s.set_policy(FallbackPolicy { max_descriptor_distance: 1e9, force_std_abs: 1e-5, force_std_rel: 0.0, variance_cadence: k, audit_every: 0, ..FallbackPolicy::default() });
        assert!(s.trusted() && residual < 1e-5, "{:?} ({residual:.1e})", s.untrusted_reason());
        let r_far = [5.5, 5.0, 4.5]
            .into_iter()
            .find(|&r| {
                s.discontinuity();
                s.evaluate(&h2(r), &mut never()).unwrap().stats.notes.iter().any(|n| n.contains("force std"))
            })
            .expect("a geometry the variance rule refuses");
        println!("the variance rule refuses R = {r_far} at a 1e-5 Ha/Bohr bound (worst training residual {residual:.1e})");
        s.discontinuity();
        let stretched = h2(r_far);
        for i in 0..6 {
            let before = s.variance_checks();
            let src = s.evaluate(&stretched, &mut never()).unwrap().stats.source;
            let c = s.variance_checks() > before;

            assert!(c && src == Source::Fallback, "consultation {i} after a rejection: checked {c}, {src:?}");
        }
    }

    /// Ruling S17 (direction corrected in fix round 2): the reported face density is the
    /// outermost grid plane, one h inside the wall, where the confined orbital is pinched ∝ h²,
    /// so the confinement per unit of it grows as 1/h² and the tolerated density scales as h² —
    /// a quarter at h = 0.125, four times at h = 0.5.
    #[test]
    fn the_box_rule_scales_with_the_grid_spacing() {
        assert_eq!(box_affected_face_density(0.25), BOX_AFFECTED_FACE_DENSITY);
        assert!((box_affected_face_density(0.125) - 0.25 * BOX_AFFECTED_FACE_DENSITY).abs() < 1e-18);
        assert!((box_affected_face_density(0.5) - 4.0 * BOX_AFFECTED_FACE_DENSITY).abs() < 1e-18);
    }

    /// A Morse teacher whose answers carry a confinement offset of the test's choosing (energy
    /// +`bias`) and report a face density `rho` — a box-affected teacher.
    struct Confined {
        inner: MorseDimer,
        rho: Arc<Mutex<(f64, f64)>>,
    }

    impl ForceProvider for Confined {
        fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
            let mut r = self.inner.evaluate(scene, should_abort)?;
            let (rho, bias) = *self.rho.lock().unwrap();
            r.energy += bias;
            r.stats.face_density = Some(rho);
            Ok(r)
        }
        fn name(&self) -> &'static str {
            "morse dimer"
        }
        fn reset(&mut self) {}
    }

    /// Task 9 fix round 1, S1: box-affected answers taken while the model was still being built
    /// (no model yet: the "essential" exemption) carry their face density; they are never
    /// quarantine candidates — so their descriptor-space fence cannot recreate the user's
    /// "trusted, 100 % fallback" — and they leave the fit once the trust floor (Q′ + 30) is met
    /// without them. The flags and densities survive the sidecar.
    #[test]
    fn box_affected_samples_never_fence_and_leave_the_fit_at_the_trust_floor() {
        let state = Arc::new(Mutex::new((2.0 * BOX_AFFECTED_FACE_DENSITY, 0.05)));
        let mut s = SurrogateKs::new(Box::new(Confined { inner: MorseDimer::h2(), rho: state.clone() }), FallbackPolicy::default()).with_teacher_tol(1e-6);
        s.set_reference_energy(1, MorseDimer::h2().e_atom);
        // no model yet: three confined answers (+0.05 Ha) are kept, as a pass on this box would
        for r in [1.5, 1.6, 1.7] {
            s.evaluate(&h2(r), &mut never()).unwrap();
        }
        assert_eq!(s.samples(), 3);
        let mut rng = StdRng::seed_from_u64(4);
        let mut add = |s: &mut SurrogateKs, n: usize| {
            for i in 0..n {
                let r = 1.2 + 1.0 * i as f64 / (n - 1) as f64;
                let sc = dimer(r, unit_vector(&mut rng), [0.0; 3], 1, 1);
                s.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
            }
        };
        // 40 clean samples: below the floor (29 + 30), the confined ones stay in the fit; the
        // residuals fail but none of them may be quarantined
        add(&mut s, 40);
        s.refit().unwrap();
        println!("below the floor: trusted {}, {:?}, quarantined {}, box-excluded {}", s.trusted(), s.untrusted_reason(), s.quarantined(), s.box_excluded());
        assert!(!s.trusted() && s.quarantined() == 0 && s.box_excluded() == 0, "{:?}", s.untrusted_reason());
        // 30 more: the floor is met without them — they leave the fit, never quarantined
        add(&mut s, 30);
        let rep = s.refit().unwrap();
        println!("at the floor: trusted {}, fitted {}, quarantined {}, box-excluded {}", s.trusted(), rep.n_samples, s.quarantined(), s.box_excluded());
        assert!(s.trusted(), "{:?}", s.untrusted_reason());
        assert_eq!((s.quarantined(), s.box_excluded(), rep.n_samples), (0, 3, 70));
        // the model answers at the confined samples' geometries: no fence there
        assert_eq!(s.evaluate(&h2(1.6), &mut never()).unwrap().stats.source, Source::Surrogate);
        // the sidecar keeps the code and the densities
        let path = std::env::temp_dir().join(format!("box-flags-{}.model", std::process::id()));
        s.save_sidecar(&path).unwrap();
        let mut back = SurrogateKs::new(Box::new(MorseDimer::h2()), FallbackPolicy::default()).with_teacher_tol(1e-6);
        back.install_sidecar(surrogate::sidecar::read(&path).unwrap()).unwrap();
        assert_eq!((back.box_excluded(), back.quarantined(), back.trusted()), (3, 0, true));
        let _ = std::fs::remove_file(&path);
    }

    /// A Morse teacher with a Gaussian bump of `amp` Ha at r0 (width w): the first fit, trained on
    /// the plain Morse, misses it.
    struct Bumped {
        inner: MorseDimer,
        amp: f64,
        r0: f64,
        w: f64,
    }

    impl ForceProvider for Bumped {
        fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
            let mut r = self.inner.evaluate(scene, should_abort)?;
            let (p0, p1) = (scene.nuclei[0].pos, scene.nuclei[1].pos);
            let d: [f64; 3] = std::array::from_fn(|a| p1[a] - p0[a]);
            let rr = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let g = self.amp * (-((rr - self.r0) / self.w).powi(2)).exp();
            let de = -2.0 * (rr - self.r0) / (self.w * self.w) * g;
            r.energy += g;
            for a in 0..3 {
                // the Morse convention: F₀ = +dE/dr û, F₁ = −dE/dr û (û from 0 to 1)
                r.forces[0][a] += de * d[a] / rr;
                r.forces[1][a] -= de * d[a] / rr;
            }
            r.stats.energy.total = r.energy;
            Ok(r)
        }
        fn name(&self) -> &'static str {
            "morse dimer"
        }
        fn reset(&mut self) {}
    }

    /// Task 9 fix round 1, I3 (fence voting): a quarantined *glitch* (a bad label, not
    /// box-affected) inside a region where the model is genuinely wrong by more than the tier
    /// used to leave no learning path — every fresh answer disagreed with the model, none was
    /// kept, the fence never released: DFT speed under a trusted badge. The teacher here carries
    /// a Gaussian bump (2e-2 Ha at r = 2.1, width 0.3) the first fit saw only the foot of (its
    /// samples stop at r = 1.6), so at r ≈ 1.9 the model is off by ≈ 4e-3 Ha (2e-3 Ha/atom).
    /// Now the third mutually consistent fresh answer that disagrees with the model is kept, the
    /// refit learns the region from it, the next fresh answer agrees and supersedes the glitch,
    /// and the model answers.
    #[test]
    fn consistent_answers_inside_a_fence_teach_the_model_the_region() {
        let teacher = || Bumped { inner: MorseDimer::h2(), amp: 2e-2, r0: 2.1, w: 0.3 };
        let mut s = SurrogateKs::new(Box::new(teacher()), FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() }).with_teacher_tol(1e-6);
        s.set_reference_energy(1, MorseDimer::h2().e_atom);
        let mut rng = StdRng::seed_from_u64(4);
        let mut t = teacher();
        for i in 0..60 {
            let r = 1.0 + 0.6 * i as f64 / 59.0;
            let sc = dimer(r, unit_vector(&mut rng), [0.0; 3], 1, 1);
            let res = t.evaluate(&sc, &mut never()).unwrap();
            let smp = Sample { pos: sc.nuclei.iter().map(|n| n.pos).collect(), z: vec![1, 1], energy: res.energy, forces: res.forces, tag: SampleTag::Scan, weight: 1.0 };
            s.add_sample(smp, &sc).unwrap();
        }
        // the glitch: 0.1 Ha off at r = 1.9 — quarantined, fencing r ≈ 1.9
        let stray = h2(1.9);
        let mut bad = morse_sample(&MorseDimer::h2(), &stray);
        bad.energy += 0.1;
        s.add_sample(bad, &stray).unwrap();
        s.refit().unwrap();
        println!("{} samples; trusted {} {:?}; quarantined {}", s.samples(), s.trusted(), s.untrusted_reason(), s.quarantined());
        assert!(s.trusted() && s.is_quarantined(60) && s.quarantined() == 1, "{:?} {}", s.untrusted_reason(), s.quarantined());
        let model_gap = |s: &SurrogateKs, r: f64| {
            let pos: Vec<[f64; 3]> = h2(r).nuclei.iter().map(|n| n.pos).collect();
            teacher().evaluate(&h2(r), &mut never()).unwrap().energy - s.model().unwrap().predict(&pos, &[1, 1], false).unwrap().energy
        };
        let before = model_gap(&s, 1.9);
        println!("teacher − model at r = 1.9 before: {before:+.2e} Ha");
        assert!(before.abs() / 2.0 > 1e-3, "test premise: the model misses the region by more than the tier");
        // a molecule moving slowly through the fenced region; the worker refits when a sample
        // is pending
        let (mut answered, mut refits, mut voted) = (None, 0, None);
        for i in 0..30 {
            let r = 1.9 + 0.002 * (i % 5) as f64;
            let res = s.evaluate(&h2(r), &mut never()).unwrap();
            if res.stats.source == Source::Surrogate {
                answered = Some(i);
                break;
            }
            if voted.is_none() && res.stats.notes.iter().any(|n| n.contains("mutually consistent fresh teacher answers")) {
                voted = Some(i);
            }
            if s.pending_refit() {
                s.refit().unwrap();
                refits += 1;
            }
        }
        let after = model_gap(&s, 1.9);
        println!("vote at evaluation {voted:?}, model answered at {answered:?} after {refits} refits; teacher − model now {after:+.2e} Ha; quarantined {}, superseded {}, trusted {}",
                 s.quarantined(), s.superseded(), s.trusted());
        assert_eq!(voted, Some(2), "the third disagreeing answer is the vote");
        let i = answered.expect("the model never learned the fenced region");
        assert!(i <= 5 && refits <= 2, "{i} evaluations, {refits} refits");
        assert!(s.trusted() && s.quarantined() == 0 && s.superseded() == 1, "the glitch is superseded, not released");
        assert!(after.abs() / 2.0 <= 1e-3, "the region is learned: {after:+.2e}");
    }

    /// The driver's safety net: a change of energy surface between consecutive evaluations that
    /// carries no `refit_jump` restarts the conserved-energy history with a note.
    #[test]
    fn a_surface_change_without_a_jump_restarts_the_history() {
        /// A Morse teacher that starts labelling its answers `Source::Surrogate` after `flip_at`
        /// evaluations (the energy stays the teacher's: a bare class flip, no jump reported).
        struct Flipper {
            inner: MorseDimer,
            flip_at: usize,
        }
        impl ForceProvider for Flipper {
            fn evaluate(&mut self, scene: &Scene, a: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
                let mut r = self.inner.evaluate(scene, a)?;
                if self.inner.calls > self.flip_at {
                    r.stats.source = Source::Surrogate;
                }
                Ok(r)
            }
            fn name(&self) -> &'static str {
                "flipper"
            }
            fn reset(&mut self) {}
        }
        let mut scene = h2(1.5);
        scene.nuclei[1].vel = [2e-4, 0.0, 0.0];
        // evaluate 1 = Dynamics::new; evaluates 2–4 = steps 1–3 (Teacher); evaluate 5 = step 4 (Model)
        let mut d = Dynamics::new(scene, Box::new(Flipper { inner: MorseDimer::h2(), flip_at: 4 }), 0.25, 1);
        for _ in 0..3 {
            d.apply(DynCommand::Step(1));
            let snap = d.tick(f64::INFINITY, &mut never());
            assert_eq!(snap.eval.energy_surface(), Surface::Teacher);
        }
        assert_eq!(d.snapshot().history.len(), 3);
        d.apply(DynCommand::Step(1));
        let snap = d.tick(f64::INFINITY, &mut never());
        assert_eq!(snap.source, Source::Surrogate);
        assert_eq!(snap.eval.energy_surface(), Surface::Model);
        assert_eq!(snap.history.len(), 1, "the history restarted at this step");
        assert!(snap.notes.last().is_some_and(|n| n.contains("energy surface changed (Teacher → Model) without a booked jump")), "{:?}", snap.notes);
        assert_eq!(snap.work_mismatch, 0.0);
    }

    /// GPU H₂ round trip: the teacher's own samples (40 centred scan geometries at random
    /// orientations, r ∈ [1.1, 2.0]) train a model that reproduces the teacher at R_e (body-2
    /// only; Task 7 pins the tiered targets). The brief's bounds, 1e-4 Ha / 2e-3 Ha/Bohr, are
    /// pinned at 41³ (10 Bohr box), where the teacher's own orientation scatter is at the
    /// egg-box level (s_E 1.3e-5/atom, s_F 1.4e-4). At 33³ (8 Bohr) the dimer's far atom is
    /// 2.8 Bohr from a face and the energy depends on the orientation at 7e-5/atom (the H atom
    /// itself is 8 mHa higher than at 41³: confinement, the provider's own "face density"
    /// warning) — a bound below the teacher's reproducibility would measure the box, so the 33³
    /// run asserts the force bound and a calibration bound (|ΔE| within 3× the fit's total
    /// energy scatter) and records the numbers.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn gpu_teacher_h2_round_trip() {
        let Some(ctx) = gpu::GpuContext::new_headless() else {
            eprintln!("sim: no GPU adapter — skipped");
            return;
        };
        let (de, df, s_e) = gpu_round_trip(ctx.clone(), 41);
        assert!(de < 1e-4, "41³: |ΔE| = {de:.2e}");
        assert!(df < 2e-3, "41³: max |ΔF| = {df:.2e}");
        assert!(s_e < 3e-5, "41³: the teacher's energy scatter is at the egg-box level, s_E = {s_e:.1e}");
        let (de, df, s_e) = gpu_round_trip(ctx, 33);
        assert!(df < 2e-3, "33³: max |ΔF| = {df:.2e}");
        assert!(de < 3.0 * 2.0 * s_e, "33³: |ΔE| = {de:.2e} against the fit's total energy scatter 2·s_E = {:.2e}", 2.0 * s_e);
    }

    /// Diagnostic (Task 9): the box's confinement energy against the face density — H₂ at R
    /// translated toward a face of the 41³ box (and stretched at the centre), E(x) − E(0) next
    /// to the teacher's face density. Sets `BOX_AFFECTED_FACE_DENSITY`.
    #[test]
    #[ignore = "diagnostic: cargo test -p sim --release diag_confinement -- --ignored --nocapture"]
    fn diag_confinement_vs_face_density() {
        let Some(ctx) = gpu::GpuContext::new_headless() else { return };
        for n_grid in [41usize, 49] {
            let cell = Cell { n: n_grid, h: 0.25 };
            let mut t = crate::provider::GpuKs::new(ctx.clone(), 1e-6);
            t.set_display(false);
            let half = 0.5 * (n_grid - 1) as f64 * 0.25;
            for r in [1.45, 2.5, 3.5, 4.5] {
                let mut e0 = None;
                for x in [0.0, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0] {
                    let far = x + 0.5 * r;
                    if far > half - 0.8 {
                        continue;
                    }
                    let mut sc = dimer(r, [1.0, 0.0, 0.0], [x, 0.0, 0.0], 1, 1);
                    sc.cell = cell.clone();
                    let res = t.evaluate(&sc, &mut never()).unwrap();
                    let e = res.energy;
                    let base = *e0.get_or_insert(e);
                    println!("{n_grid}³ R {r:.2} centre x {x:.1} (far atom {:.2} Bohr from the face): E {e:+.6}  E − E(centre) {:+.2e}  face density {:.2e}",
                             half - far, e - base, res.stats.face_density.unwrap_or(f64::NAN));
                }
            }
        }
    }

    /// One round trip on an n³ grid: (|ΔE| at R_e, max |ΔF| at R_e, the fit's s_E per atom).
    fn gpu_round_trip(ctx: gpu::GpuContext, n_grid: usize) -> (f64, f64, f64) {
        let cell = Cell { n: n_grid, h: 0.25 };
        let with_cell = |mut sc: Scene| {
            sc.cell = cell.clone();
            sc
        };
        let t0 = Instant::now();
        let mut teacher = crate::provider::GpuKs::new(ctx.clone(), 1e-6);
        teacher.set_display(false);
        let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_teacher_tol(1e-6);
        // the H reference from the teacher itself (an isolated atom at the same settings)
        let atom = Scene { nuclei: vec![Nucleus::new(1, [0.0; 3])], cell: cell.clone(), charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3] };
        let e_h = s.evaluate(&atom, &mut never()).unwrap().energy;
        s.set_reference_energy(1, e_h);
        // 40 scan geometries on r ∈ [1.1, 2.0]: every fallback keeps its sample
        let mut rng = StdRng::seed_from_u64(7);
        let n = 40;
        for i in 0..n {
            let r = 1.1 + 0.9 * i as f64 / (n - 1) as f64;
            let origin = [rng.gen_range(-0.2..0.2), rng.gen_range(-0.2..0.2), rng.gen_range(-0.2..0.2)];
            let sc = with_cell(dimer(r, unit_vector(&mut rng), origin, 1, 1));
            // unrelated geometries: keep the warm state, drop the density extrapolation (R13)
            s.discontinuity();
            let r = s.evaluate(&sc, &mut never()).expect("GPU teacher");
            assert_eq!(r.stats.source, Source::Fallback);
        }
        assert_eq!(s.samples(), n + 1);
        let t_scan = t0.elapsed().as_secs_f64();
        let rep = s.refit().expect("refit");
        println!("GPU H₂ {n_grid}³: E_H = {e_h:+.6} Ha; {} samples in {t_scan:.1} s; fit rmse E/atom {:.2e} Ha, F {:.2e} Ha/Bohr; s_E {:.1e} s_F {:.1e} λ {:.3e}; dof_E {:.1} dof_F {:.1}; trusted {}",
                 rep.n_samples, rep.rmse_energy_per_atom, rep.rmse_force, rep.sigma_e, rep.sigma_f, rep.lambda, rep.residual_dof_e, rep.residual_dof_f, s.trusted());
        assert!(s.trusted(), "{rep:?}");
        let re = with_cell(h2(1.45));
        s.discontinuity();
        let t1 = Instant::now();
        let r = s.evaluate(&re, &mut never()).unwrap();
        let t_sur = t1.elapsed().as_secs_f64() * 1e3;
        println!("surrogate at R = 1.45: {:?} E {:+.6} Ha, F_x(0) {:+.4e}, uncertainty {:?}, {t_sur:.2} ms, notes {:?}", r.stats.source, r.energy, r.forces[0][0], r.stats.uncertainty, r.stats.notes);
        assert_eq!(r.stats.source, Source::Surrogate, "{:?}", r.stats.notes);
        let mut direct = crate::provider::GpuKs::new(ctx, 1e-6);
        direct.set_display(false);
        let d = direct.evaluate(&re, &mut never()).unwrap();
        let de = (r.energy - d.energy).abs();
        let df = max_force_diff(&r.forces, &d.forces);
        println!("direct GPU: E {:+.6} Ha, F_x(0) {:+.4e}; |ΔE| {de:.2e} Ha, max|ΔF| {df:.2e} Ha/Bohr", d.energy, d.forces[0][0]);
        (de, df, rep.sigma_e)
    }
}
