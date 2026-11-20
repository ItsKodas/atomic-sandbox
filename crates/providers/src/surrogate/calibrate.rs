//! The calibrated class verdict (M8a Task 2; spec §4.1): rule 4b's verdict per environment class
//! from the error the teacher **measured** at the geometries the class verdict decides, with the
//! fit's leave-one-out error as a floor and hysteresis. It replaces the max rule of M6 Task 11
//! round 6 (one sample alone decided its class) wherever evidence exists; without evidence that
//! rule stands unchanged. Pure functions of the audit history, the policy's bounds and the
//! per-class leave-one-out statistics; the hysteresis state is the standing verdict the caller
//! passes in (`SurrogateKs::class_memory`). Atomic units throughout (Hartree, Bohr); every number
//! below is a ratio to the policy's own bounds.
//!
//! # The sample floor per key (M8a user-fix 3 and its review)
//!
//! The verdict's first condition is a count: enough samples for the parameters the key's
//! predictions can move with. An atom's energy term is E_i = Σ_k c_{s_i,k} B_{i,k}; a feature of
//! a neighbour channel it does not have is exactly 0, so its prediction depends only on the
//! independent columns ([`surrogate::Basis::column_classes`]) with B_{i,k} ≠ 0 there
//! ([`surrogate::touched_columns`]). Per key (`trust::key_column_counts`):
//!
//! - **q_k** = the independent columns non-zero on at least one fitted atom of the key — the key's
//!   column support S_k, data-driven, nothing per element (rule 1);
//! - **n_k** = the fitted samples (those the fit uses: quarantined, superseded, box-excluded,
//!   held-out and lone `Q_LONE` samples are not) holding at least one atom **whose key is k**;
//! - the key is short while n_k < q_k ("12 of 29 samples needed").
//!
//! Before, every key was held to its class's count against the **whole** basis's Q′. A species
//! added to a collection brings columns no other key touches, so it raised every key's bar: the
//! user's water (H/O, Q′ 452, its many-body class 649+ samples, trusted) read "967 samples < Q′
//! 1590" on every key once N₂ was trained in, though none of its data or columns changed.
//!
//! **Why n_k counts only the key's own samples** (the review's ruling 1). The first version counted
//! every sample of the key's class with an atom touching one of k's columns. Columns are shared
//! between keys, so other keys' samples stood in for k's: in an H/N collection of 120 H₃ and 3
//! NH₃, NH₃'s many-body H{H, N} key shares its H–H columns with H₃'s H{H} and counted 123 samples
//! against its 92 columns — its N channels resting on three samples
//! (`a_key_counts_only_the_samples_holding_an_atom_of_it`: now 3 of 92, short); and water's H
//! atoms counted for H₃'s key. A dimer's atom is non-zero on every many-body column of its species
//! (a one-neighbour body-3 feature is R_n R_n′ (2l+1)/4π) but cannot tell their angular directions
//! apart; a pair key never counts a many-body sample's atoms, nor the reverse. The cost: a key seen
//! in few samples is short on its own floor even where its columns are shared with a well-sampled
//! key (a stretched water's pair H{O}, say) — it falls back until it has its own samples.
//!
//! **Why the column support and not the rank** of the key's design block: the support is what the
//! parameters *are*; the rank is what the data *identify*, and a count of samples per parameter is
//! a floor below which the leave-one-out and DFT checks that follow cannot be read —
//! identifiability itself is theirs to judge (an undefined leave-one-out refuses as ∞). The rank
//! would also cost an eigen-decomposition per key per refit.
//!
//! A key no fitted atom holds has **no samples of its own** (n_k = 0) and is refused on the sample
//! floor (`ClassKey::Samples`, 0 of Q′ shown — its own q_k needs an atom of it) — never judged on
//! its class's leave-one-out, which measures other keys' atoms (whole-branch final review 5:
//! before, it took its class's count against Q′ and, with the class above Q′, could be trusted on
//! the other keys' numbers). A class-wide verdict (no key of the class fitted) keeps its class's
//! count against the whole Q′.
//!
//! **The whole-model floor** (`TrustKey::Samples`; the review's ruling 2) is the same count for the
//! whole fit: n_samples ≥ |∪_k S_k|, the columns any fitted atom touches
//! (`SurrogateKs::sample_floor`), not the whole basis's Q′ — else the bug returns whenever the
//! store is below Q′ (1200 water samples and 5 N₂ against Q′ ≈ 1590). Measured
//! (`the_whole_model_floor_is_the_columns_the_fitted_atoms_touch`): 440 H/O samples, floor 142 of
//! Q′ 176; five N₂ join, floor 171 = 142 + N₂'s own 29 against Q′ 543, n 445 — trusted, water
//! answered. The gates on fitting count the same floor (the controller's ruling on concern 2):
//! the end-of-pass refit and the no-model readout need the usable samples to reach the columns
//! their atoms touch in the basis the fit would build (`SurrogateKs::fit_floor`), and the box
//! exclusion waits for that floor + 30 (`trust_floor`) — so a pass bringing N₂ into a 1200-sample
//! water store refits (`a_pass_bringing_a_new_species_into_a_large_store_refits`: 480 samples,
//! floor 171, Q′ 543, the N₂ key 40 of 29 and learned). The fit itself needs no count: a column no
//! fitted atom touches has an all-zero Gram row and column and the ridge prior holds its
//! coefficient at 0. The basis's Q′ stays only where no class judgement exists (no store) and
//! as the no-model keep cap ([`super::NO_MODEL_SAMPLE_FACTOR`] · Q′, which only bounds how many
//! answers are kept before a first fit).
//!
//! # What the verdict asks
//!
//! Rule 4b asks, per environment class c (pair, many-body; [`super::trust::atom_class`]): may the
//! model answer a geometry whose atoms are in c? The claim behind "yes" is the one rule 6 makes
//! per step — the expected error is within the policy's bound. Three measurements bear on it:
//!
//! 1. The **audit history**'s points in this verdict's *jurisdiction* (`surrogate::history`):
//!    audits (every rule passed and the model answered) and the fallbacks rule 4b itself refused
//!    (every other rule passed; only the class verdict stood between the model and an answer) —
//!    the geometries this verdict decides, with the teacher's answer beside the model's. Per atom
//!    the error ‖F_t,j − F_m,j‖ and |F_t,j|, from which the policy's own per-atom bound
//!    b_j = force_bound(|F_t,j|) = min(max(2e-3, 10 % |F|), 1e-2) is formed; per point the energy
//!    gap E_t − E_m.
//! 2. The fit's **leave-one-out residuals** at its training samples (`surrogate::loo`): per fitted
//!    atom of class c the force error of the fit without its sample, ‖r_F,j‖/√3 ÷ b_j, and per
//!    sample the energy error |r_E|/N over the tier. Exact, measured, in-domain.
//! 3. The model's **predictive std** at the audited geometries against the error there: z_j =
//!    ‖ΔF_j‖ / pred_force_std_j, E[z²] = 1 for a calibrated posterior — the spec's "predicted
//!    uncertainty vs measured error", reported as the calibration factor κ_c.
//!
//! # Per environment species (M8a user-test fix 2, the controller's rulings 1–3)
//!
//! The evidence, its window, W, λ, the worst-atom guard and the hysteresis memory are kept per
//! **environment key** (`surrogate::EnvKey`): the atom's class, its Z and the sorted set of its
//! neighbours' species within r_cut — Z as a key only (rule 1), nothing read from it. An atom's
//! verdict is its own key's ([`Calibration::key_verdict`]); the class's leave-one-out statistics
//! stay per class (the fit's residuals are per class), and the sample floor is the key's own —
//! its own samples against the columns its atoms touch (M8a user-fix 3 and its review, "The sample
//! floor per key" above). The user's case: a
//! hot-H₃ fallback measured an H atom at 5.66× its bound (generation 416); with one evidence
//! window per class it kept every many-body environment out, water's included, although water's 81
//! comparisons were all within 0.03× the bound — and with fewer than [`WINDOW_POINTS`] many-body
//! points in the ring nothing aged. A key of the central species alone would not have separated
//! them: water's H atoms have an O and an H within r_cut (two neighbours, many-body), the same
//! "many-body H" as H₃'s. With the neighbour species, water is O{H} and H{H, O}, H₃ is H{H}.
//!
//! A history point recorded before species were kept (`AuditPoint::species` empty) counts for
//! **every** key of its atoms' classes — the conservative reading (ruling 2): an old collection's
//! evidence keeps blocking until the window of newer, attributed points ages it out.
//! [`Calibration::per_class`] is still the class-wide calibration (every point of the class): a
//! readout. **Re-judged on new evidence** (ruling 3): a new audit or rule-4b point re-runs this
//! calibration and the per-key verdicts against the class statistics of the fit that stands
//! (`SurrogateKs::rejudge_classes`; cost one pass over the ≤ 4096-point ring), so a parked scene
//! whose fallbacks are all "skipped: similar" — no refit — can enter or leave. **Entering between
//! refits needs a full window** (`ClassKey::Window`): an untrusted key whose re-judged verdict
//! would trust it on fewer than [`WINDOW_POINTS`] comparisons keeps waiting for the next refit —
//! the gated repro's hot run measured why: re-judged at once, its many-body class entered on its
//! first three benign rule-4b fallbacks (rms inside ENTER) and answered two compact collisions at
//! 3.9× and 6.1× the force bound, the heavy tail a short window cannot see ("Why 300"). Leaving is
//! immediate. **λ does not move between refits**: the variance inflation is the posterior's and is
//! re-measured with it (refits, installs, policy changes); re-measured on every new verdict point it
//! let rule 6 admit the same run's tail (one answer at 3.19× with the window rule alone; both
//! guards: 85 answers, worst 0.97×, the record of before — `references.md`).
//!
//! # The evidence window (fix round 2: evidence ages by model generation)
//!
//! A history point is a measurement of the model that judged it (`AuditPoint::model_gen`). Once a
//! refit has taught the model at the geometries a point measured — every fallback and audit answer
//! is a training sample — that point describes a model that no longer answers, and the ring would
//! otherwise keep it for 4096 points: a class refused on its audits receives only rule-4b
//! fallbacks afterwards, capped at the audits' weight, so perfect fallbacks of every later model
//! could never bring W below the refusing audits' share (the task review's I4: 8 audits at 1.46×
//! leave W ≥ 1.03 > LEAVE), and one 3× atom would block the entry guard below for the ring's
//! lifetime. So per class the evidence is a **window of whole generations**: walking the class's
//! jurisdiction points from the newest generation back, generations are taken until they hold at
//! least [`WINDOW_POINTS`] = 300 points; older points are aged out (`ClassCalibration::aged`).
//! With fewer than 300 points in the ring nothing ages. A generation is never split: the points one
//! model produced are judged together.
//!
//! What this does, both ways: evidence produced by a model **does not age while that model
//! stands** — more fallbacks under the refused model are added to the same window as the refusing
//! audits, capped, and cannot outvote them; after a refit (which has the failing geometries in its
//! training set) the newer models' own comparisons accumulate, and once they reach 300 points the
//! verdict computed at the following refit reads them alone. Released after a teaching refit and
//! the newer models' own evidence, blocked until then (`evidence_ages_by_model_generation_both_ways`,
//! `a_class_refused_on_its_evidence_reenters_after_a_teaching_refit`) — and never after more than
//! 300 newer points, where the whole ring kept the refusing points for 4096.
//!
//! Why 300: the window feeds two statistics with two requirements, and the stricter one sets it.
//! (1) The rms must not be noisier than the hysteresis band tolerates: the rms of n χ²₁-like draws
//! scatters by 1/√(2n) relative (below), and two sigma of that inside the band √2 − 1 = 0.414 needs
//! n ≥ 11.7. (2) The worst-atom guard must be able to **see a tail**: a window free of 3× atoms has
//! to make a 3× tail at the rate the audited runs measured unlikely — fix round 1 measured 1 judged
//! answer of 67 beyond 3× the bound (1.5 %) on a many-body class whose rms sat inside the bound;
//! for a tail rate p = 1 % a window of n points shows none with probability 0.99ⁿ, ≤ 5 % from
//! n = 298 → 300. Consecutive MD points are correlated, so the effective n is smaller than the
//! count (the window errs on the side of seeing too little); that is what the hysteresis and the
//! per-step rules 6/8 are for. The first fix-round-2 draft used 12 (requirement 1 alone) and the
//! gated repro measured why that is wrong: the many-body class's bad first rule-4b points (NVE run,
//! per-step rms up to 2.1× the bound) aged out behind 12 benign newer ones, the class entered on 3
//! points, and — the run's audits off — no comparison of its answers ever reached the window, so it
//! answered the hot run (673 steps; 27 judged answers beyond 3× the force bound or the energy tier,
//! up to 5.4× and 3.4e-3 Ha/atom). Generation numbers are monotone within a provider and an install
//! moves the provider's generation past every generation in the installed history
//! (`install_sidecar`), so "newer" is a comparison of numbers.
//!
//! # The statistics
//!
//! Per class c, over its jurisdiction points p in the window (weights w_p below; n_c = the number
//! of points):
//!
//! ```text
//! W_c    = √( Σ_p w_p w̄_p² / Σ_p w_p ),   w̄_p² = mean over the point's atoms of class c of (‖ΔF_j‖/√3 / b_j)²     the measured force error, in bound units
//! W_E,c  = √( Σ_p w_p (|ΔE_p|/N_p / tier)² / Σ_p w_p )                                                        the measured energy error, in tier units
//! κ_c    = max(1, ẑ_c) · (1 + 1/√(2 n_c)),   ẑ_c² = Σ_p w_p z̄_p² / Σ_p w_p,  z̄_p² = mean of z_j² over the class atoms   the calibration factor (a readout)
//! A_c    = max over the window's points and their class atoms of ‖ΔF_j‖/√3 / b_j                       (ClassLoo::worst_atom)      the worst atom
//! R_c    = RMS over the class's fitted atoms of ‖r_F,j‖/√3 / b_j      (ClassLoo::force_rms)       the LOO force error
//! R_E,c  = RMS over the class's samples of |r_E|/N / tier               (ClassLoo::energy_rms)      the LOO energy error
//! V_c    = max( R_c,  W_c · (1 + 1/√(2 n_c)) )                          (ClassLoo::measured for the second term)   the verdict's force value
//! V_E,c  = W_E,c · (1 + 1/√(2 n_c))                                     (ClassLoo::measured_energy)                the verdict's energy value
//! ```
//!
//! **Why the measured error and not κ·R** (the fix round's A.1, the physics review's I1/I2). The
//! first version decided on κ_c · R_c. That product equals the error at visited geometries only if
//! the posterior's std there matches the leave-one-out error at a training point — false for a
//! sparse class, where removing a sample is a high-leverage removal (large R) while a visited point
//! between samples has a small std: V overstated by the leverage ratio (measured: V 1.8–2.6
//! claimed where the test's own checker found 0.25–0.58× the bound). And z has no floor on its
//! denominator: one confident comparison with a modest error (z = 62 among 266 points) set a
//! class's κ for the ring's lifetime — the max rule's knife-edge relocated. The teacher's error in
//! bound units has neither defect: it is the quantity the policy claims, bounded per atom by the
//! error itself over a fixed scale. κ_c is still computed and reported (the Model tab, a later
//! variance inflation) but decides nothing.
//!
//! **One draw per point, not per atom**: the model's ΣF = 0 makes a dimer's two atoms one draw
//! (the Task 1 physics review's I1); w̄_p² is the mean over the point's class atoms. **Weights**:
//! an audit point weighs 1; a class's rule-4b fallback points (both class codes together) weigh
//! together at most max(n_audits, [`MIN_EVIDENCE`]) — spread evenly — so one class refusal
//! repeated over a heating run (3313 near-identical pair points in the baseline) is worth one
//! class's evidence and never outweighs the audits (controller ruling), yet a refused class with
//! no audits can still be calibrated by its own fallbacks. The evidence count n_a + min(n_f, cap)
//! is formed in integers (fix round 2, the task review's I1: as n_a + (min(n_f, cap)/n_f)·n_f it
//! came out 2.9999999999999996 at n_f = 47, 94, 147, … and silently dropped the class to the max
//! rule — a refit-time knife-edge from rounding). Points refused by rules 5/5b/5d/5e/6/7 or the
//! fences are not evidence: those rules refuse whatever the class verdict says, and their errors
//! (z rms 12–54 in the baseline: the posterior extrapolating) argue for *those* rules, not about
//! the model inside their reach — including a class refusal the variance verdict would have
//! refused as well, which `evaluate` codes apart (`super::HISTORY_RULE_CLASS_AND_VARIANCE`). The
//! **inflation** (1 + 1/√(2n)) is one sigma of the estimate's own scatter — the rms of n draws of a
//! χ²₁-like variable (a dimer's error has one effective component; Var = 2) scatters by ≈ 1/√(2n)
//! relative: ×1.41 at 3 points, ×1.14 at 26, ×1.04 at 336. It is applied with n = the number of
//! points (the scatter is a property of the draws), while the weights settle influence (review
//! M1). The energy is judged by its measured error, never by z_E (biased: +1.7 σ_E median at n 26,
//! Task 1 review M5). A comparison that is not a number (NaN or ∞) makes W, A and κ infinite: the
//! worst evidence, never dropped (the activity Task 3b ruling). R_c stays as the floor — the model
//! is never trusted beyond its own leave-one-out; robust, being an RMS over hundreds of atoms.
//!
//! # The variance inflation (fix round 3; the controller's ruling on C1)
//!
//! The class verdict judges a class's **rms** error; rules 6/7 judge each step by the posterior's
//! epistemic std. Where the posterior is overconfident (κ 2–12 for the hot many-body class) a
//! trusted class's heavy tail passes rule 6 — answers to 5.4× the bound in fix rounds 1–2. So the
//! posterior's std is recalibrated per class by the error the teacher measured where that std
//! decides (the standard post-hoc variance recalibration: one scale factor on the predictive
//! variance, fitted on held-out comparisons — Levi, Gispan, Giladi & Fetaya, Sensors 22, 5540
//! (2022), "Evaluating and calibrating uncertainty prediction in regression tasks" (not verified);
//! Busk et al., Mach. Learn.: Sci. Technol. 3, 015012 (2022), for molecular models (not verified) —
//! both cited from memory without access to the papers; the method stands on the measurement here,
//! not on them):
//!
//! ```text
//! λ_c = max(1, √( Σ_p max(0, w̄_p² − ā_p²) / Σ_p s̄_p² )) · (1 + 1/√(2 n_c)),   s̄_p² = mean over the point's class atoms of (σ_epi,j/√3 / b_j)²
//!       ā_p² = mean over the point's class atoms of (ā_c / b_j)²,   ā_c = min(σ_F, the class's training force-residual rms per component)
//! rule 6:  λ_c(j) · σ_epi,j/√3 ≤ b_j             for every atom j (its own class's λ; 1 for an isolated atom)
//! rule 7:  max_j λ_c(j) · σ_E,epi/N ≤ tier        (one scale of the posterior covariance Σ: both stds are linear in √Σ)
//! ```
//!
//! **The aleatoric part is removed from the numerator** (M8a Task 5 fix round 1, the review's I2).
//! At a fresh geometry the error the teacher measures has the expectation E[(‖ΔF‖/√3)²] =
//! σ_epi²/3 + σ_F² — the epistemic part *and* the fit's misfit floor σ_F, which no amount of data
//! at that geometry removes and which rule 6 (the epistemic std only) is not about. Without the
//! subtraction even a calibrated posterior got λ² ≈ 1 + σ_F²/s̄² — 24.9 / 18.9 (223 a day earlier)
//! on the user's collection — and λ·s̄ was the total rms error: rule 6 then refused any atom whose
//! epistemic std exceeded a few times the class's typical one, and every fallback it added shrank
//! s̄ and grew λ. Per point the subtraction is clamped at 0 (a point whose error is below the floor
//! is no evidence of overconfidence, not negative evidence). ā_c is min(σ_F, the class's own
//! training residual rms): σ_F is the fit's estimate over all rows, a class that the fit
//! represents better than average must not have more subtracted than its own misfit. Dividing by
//! the total std instead was rejected: it under-inflates a truly overconfident posterior (σ_F
//! dominates the denominator exactly where the epistemic std is too small).
//!
//! over the class's points in the **posterior's jurisdiction** — the verdict's (audits, rule-4b
//! fallbacks) plus the fallbacks rules 6/7 refused alone (codes 14, 15, and 24 where only λ_T or
//! λ_E refused: rule 6/7 at the inflated factor, still the std deciding) or with the class (21): the
//! geometries where the std decides. Points of rules 5/5a–5e and the fences are not: those refuse
//! whatever the std says. The same window of generations as the verdict (≥ [`WINDOW_POINTS`]),
//! one draw per point, no weight cap (every comparison is equally a draw of the posterior; the cap
//! protected the verdict's audits from heating floods). The guards the ruling asked for:
//!
//! - **The ratio of sums, not a mean of z²** (the physics review's I2: z = ‖ΔF‖/σ has no floor
//!   on its denominator, and one confident comparison at z = 62 among 266 set κ ≥ 5.4 for the
//!   ring's lifetime). Both sums are in the bound's units, so a point with a tiny std *and* a small
//!   error adds little to either: the same 266 + 1 points give λ_raw = 1.0000 where the mean of z²
//!   gives 3.9 (`the_variance_inflation_is_a_ratio_of_sums_over_the_posteriors_jurisdiction`).
//!   The sums are dominated by the comparisons whose error or std is a sizeable fraction of the
//!   bound — the ones rule 6 decides.
//! - **Only with ≥ [`MIN_EVIDENCE`] points**; below, λ = 1 exactly (the uninflated rule of before:
//!   nothing loosened, nothing tightened without evidence).
//! - **Floored at 1**: never deflated — a posterior measured underconfident keeps its own std
//!   (rule 6 is not loosened by calibration).
//! - **× (1 + 1/√(2n))**: the conservative side of the estimate's one-sigma scatter, as for W.
//! - **No cap**, and why that is safe from a lockout: a large λ only sends steps to the teacher,
//!   and those steps are rule-6/7 points — the factor's own evidence — so λ is re-measured from the
//!   refused geometries themselves as the window moves; a class whose model improves comes back.
//!   A comparison that is not a number makes λ = ∞ (every atom of the class refused) until it ages.
//!
//! λ moves at refits, installs and policy changes, with the verdict (`refresh_class_loo`, which
//! also drops a cached variance verdict). Rule 6's reported uncertainty is the inflated std; the
//! history records the model's own (uninflated) stds, which λ is measured from.
//!
//! # The verdict, with hysteresis
//!
//! A key short of samples is untrusted (`ClassKey::Samples`; "The sample floor per key" below; a
//! class-wide verdict: fewer than Q′ samples, as before). Otherwise:
//!
//! - **The worst-atom guard** ([`WORST_ATOM`] = 3; `ClassKey::WorstAtom`): an untrusted class does
//!   not **enter** while any point in its window has an atom measured beyond 3× its bound — at
//!   every evidence count (fix round 2, the controller's ruling on the entry question); while the
//!   class is below its evidence floor ("The evidence floor" below; before it, fewer than
//!   [`MIN_EVIDENCE`] points) such an atom also makes a trusted class leave.
//!   For a calibrated posterior at std = bound, P(χ²₃/3 > 9) = 5.9e-6 per atom: such an error is a
//!   posterior that is wrong somewhere the class would answer, not an unlucky draw (the repro
//!   test's own derivation). Why at entry and at every n: the many-body error is heavy-tailed
//!   (refused-step max/rms 4–6 against ≈ 1.9 for a calibrated Gaussian), so an rms over the
//!   benign early points of a refused class — at any n — admitted it before the compact
//!   collisions that followed reached 3–5× (the fix-round-1 measurement). A trusted class with
//!   evidence at the floor keeps being judged by its rms (rule 8's audits fail a trusted model's 3×
//!   answer per step and the window then holds the failing audit). The window ages the guard's
//!   points too.
//! - **calibrated** (n_c ≥ [`MIN_EVIDENCE`] = 3 points): untrusted → trusted when V_c ≤ [`ENTER`]
//!   = 1/√2 and V_E,c ≤ ENTER; trusted → stays while V_c ≤ [`LEAVE`] = 1 and V_E,c ≤ LEAVE. The
//!   class is never trusted with its measured rms error beyond the bound — the tolerance rules 6
//!   and 8 use — and enters only once that error is clearly inside: the band is a factor √2 in
//!   error (2 in variance), equal to the estimate's one-sigma scatter at n = 3 and wider from n = 4
//!   (review M3), so refit-to-refit jitter of W and R does not flip the verdict. The refusal names
//!   which term decided: the measured error (`ClassKey::Calibrated`), the leave-one-out floor
//!   (`ClassKey::LooFloor`), the energy LOO RMS with M (`ClassKey::EnergyRms`, below) or the
//!   measured energy (`ClassKey::MeasuredEnergy`). **Below the evidence floor** the rule of before
//!   must pass as well (`ClassKey::Floor`; "The evidence floor" below).
//! - **uncalibrated** (fewer than 3 points): the rule of before, unchanged — LOO force max ≤ 1 and
//!   rms ≤ 1, LOO energy max ≤ 1 (`super::trust::class_refusal`) — plus the worst-atom guard both
//!   ways.
//! - **the energy half of C1, both paths** (open-shell model Task 7c, the user's decision
//!   2026-11-13): refused (`ClassKey::EnergyRms`) when the energy LOO RMS, the misfit set M
//!   included, is beyond the tier (> 1, NaN fails) **and** an M sample contributes to that energy
//!   sum (`ClassStats::energy_misfits` > 0). With M empty it never decides: uncalibrated the RMS is
//!   at most the max (≤ 1 already), calibrated the verdict never read the energy LOO. The limit is
//!   1 in both paths — no ½ entry band, the calibrated energy check is the measured one. Without evidence nothing is loosened; the fallbacks a refused class produces are
//!   themselves evidence (rule-4b points), so a class the max rule refuses for one outlier earns
//!   the calibrated verdict from teacher steps of its own — since the evidence floor, once they
//!   span 12 fits and 60 points (before it, three steps and a refit). A class whose only
//!   recent comparisons were box-affected has no evidence from them (they leave no point):
//!   unverified, never passing by calibration.
//!
//! Three is the smallest count from which an rms can be told from a single draw — the count the
//! audit burst already uses after a cleared failure ([`super::AUDIT_BURST`]). The hysteresis state
//! is the verdict that stood when the refit, install or policy change began (`class_memory`,
//! committed once it is done) and the generation it was entered at (`ClassLoo::trusted_since`);
//! it is reset by a forget, and **saved with the model** (the sidecar's `class_memory` header key,
//! fix round 2, the task review's M4): a reload, the worker's re-spawn with its carried model and a
//! resumed training keep a verdict that sat inside the band, instead of re-entering. The refusal's
//! words are [`refusal_words`], one function for the verdict and the provider's Log note (fix
//! round 2, the task review's M2). Numbers of record: `validate/references.md`, "M8a calibrated
//! verdict".
//!
//! # The evidence floor (trust tightening, spec 2026-11-18 §3; the user: "Keep it, gated")
//!
//! Three points tell an rms from a single draw; they do not make it a number to open a class on.
//! The measured case (bisect 762159b): a many-body key opened on 3 rule-4b points — LOO force rms
//! 0.32×, max 3.07× — and answered a hot run beyond the gate. So the calibrated verdict opens a key
//! **alone** only at the floor:
//!
//! ```text
//! n_fit = |{ fit_id(p) }|,  n_pts = |{ p }|,   p over the key's verdict points in its window (audits and rule-4b
//!                                              fallbacks, uncapped) that carry species and a fit id
//! floor met  ⇔  n_fit ≥ FLOOR_FITS = 12  and  n_pts ≥ FLOOR_POINTS = 60
//!
//! floor met:                 today's calibrated verdict (ENTER / LEAVE, the worst-atom guard on entry)
//! evidence ≥ 3, floor unmet: today's calibrated verdict AND the rule of before (LOO force rms, max ≤ 1, energy max ≤ 1),
//!                            the worst-atom guard both ways
//! evidence < 3:              the rule of before, the guard both ways (unchanged)
//! ```
//!
//! At every count this is today's verdict ANDed with a further condition, or today's verdict — it
//! can only refuse more. **The criterion**: two sigma of the verdict value's relative scatter,
//! ≈ ½√(CV²/n) for an rms over n draws whose squares have coefficient of variation CV², inside the
//! √2 band: n ≥ CV²/(√2 − 1)² = 5.83·CV² (the spec's review I2). **12 fits**: one fit pools one
//! decorrelated block (in the harness a 25-step refit interval, 6–9 τ_int); the measured block CV²
//! 0.84 needs n ≥ 4.9. **60 points**: where a fit holds 1–3 points (the app refits for every kept
//! sample) the per-answer CV² proxy 10 needs n ≥ 59; the points are correlated (τ_int 2.8–4.4: 14–21
//! effective draws), which is why both conditions are needed. W keeps its per-point rms: re-weighting
//! per fit could dilute one fit of many bad points by benign one-point fits.
//!
//! **A fit, not a generation**: `model_gen` is also bumped by policy changes, restores, forgets
//! and installs, so a parked scene could reach 12 by settings changes; the fit id is bumped only by
//! a refit that solved new coefficients from new teacher data (`SurrogateKs::fit_id`). **Which points
//! count**: a point needs species (legacy species-less points still enter W, the evidence and the
//! guard for every key of their class, never the floor) **and** a fit id (every point saved before
//! the fit id existed has none) — the stricter of the spec's two readings, so n_pts never counts a
//! point n_fit cannot see. Over-count, documented (spec §3): one key's refit advances every key's
//! fit id.
//!
//! **The refusal's name**: `ClassKey::Floor` — (the LOO ratio that decided, 1), "within the bound
//! over n_fit of 12 fits / n_pts of 60 checks, and its leave-one-out is X× the bound" (a count that
//! has reached its floor reads "80 checks (60 needed)", [`floor_count`]) — only when the
//! calibrated verdict itself passes; when it fails, its own failing term is named (the floor is not
//! why the key is shut, and the words would claim "within the bound"). The decision is the AND
//! either way.
//!
//! **No grandfathering**: the verdict is re-judged at every install and attach (`refresh_class_loo`
//! with the file's `class_memory` as the hysteresis), so a key a file saved open on legacy points
//! with LOO max > 1 loads closed (`a_class_saved_open_on_three_points_loads_closed`). The way back
//! from a parked scene (its thinned fallbacks add no fit) is the existing one for a LOO-blocked
//! class: the teach prompt after `TEACH_PROMPT_STREAK` class fallbacks, or moving the scene.
//! Measured on the Morse fixtures: a pair class refused for one outlier (LOO max 1.08×) now enters
//! after 120 fallback steps — the 12th fit — instead of 10.
//!
//! # The tail-aware inflations (trust tightening, spec 2026-11-18 §4; plan Task 5)
//!
//! λ_sum (above) is a ratio of sums: it is dominated by the high-std points and cannot see a
//! confident point's large error — the hot run's 4.48× answer needs λ ≈ 7.7 to be refused (spec
//! §4; its std is not in the log), and with such an atom at s = 0.13 b among 80 benign points
//! λ_sum is 1.21 (`the_tail_inflation_refuses_the_measured_tail`, where λ_T is 7.69). λ_T (forces)
//! and λ_E (energy) ask the question rule 6/7 answer, of exactly the
//! measured atoms (points) those rules would admit. Pure functions of the history's **answer
//! sample** P_A ([`in_answer_sample`]) and the policy's bounds; they decide nothing here — rules 6
//! and 7 read them (`SurrogateKs::key_factors`, `variance_check`) as λ_F(j) = max(λ_sum(k),
//! λ_T(k)) on s_j and max(max_j λ_F(j), max_k λ_E(k)) on σ_E,epi/N, N the one normalisation rule 7
//! and λ_E share (`policy::energy_atoms`), a sparse key (fewer than [`MIN_EVIDENCE`] P_A points of
//! its own) at its class's λ_T and λ_E as floors. Each comes with its [`TailCause`], which the
//! refusal names. All quantities are ratios, no unit enters:
//!
//! ```text
//! forces, per P_A atom j of the key (bound units; force_bound(|F|) = min(max(2e-3, 10 %|F|), 1e-2) Ha/Bohr):
//!   r_j = s_j / b⁶_j,  s_j = σ_epi,j/√3,  b⁶_j = force_bound(|F_m,j|)   rule 6's own ratio (uninflated): its bound is the model's
//!   e_j = ‖F_t,j − F_m,j‖/√3 / b_j,      b_j = force_bound(|F_t,j|)     the measured error on the teacher's bound (rule 8's)
//!   (a point recorded without |F_m,j|: b⁶_j = force_bound(|F_t,j| + ‖ΔF_j‖) ≥ force_bound(|F_m,j|))
//! energy, per P_A point p holding the key (tier units, tier = energy_std_per_atom = 1e-3 Ha/atom; N_p its atoms):
//!   r_p = σ_E,epi,p / N_p / tier,   e_p = |E_t − E_m| / N_p / tier      (the aleatoric floor not subtracted)
//!
//! A(λ) = { items with λ·r ≤ 1 }                 the measured items rule 6 (7) would admit at factor λ
//! A good  ⇔  max_A e ≤ M                        M = WORST_ATOM = 3 for λ_T, ENERGY_MAX = 1 (1× the tier) for λ_E
//!         ∧  (A meets the floor ⇒ √( Σ_{p∈A} ē_p² / n_A ) · (1 + 1/√(2 n_A)) ≤ ENTER)
//!            ē_p² = the mean of e_j² over p's atoms in A (one draw per point; λ_E: e_p² itself),
//!            n_A = the points with an item in A; the floor: ≥ FLOOR_FITS distinct fits and ≥ FLOOR_POINTS points
//! λ = min{ λ ≥ 1 : A(λ′) good for every λ′ ≥ λ }
//! ```
//!
//! **The one pass.** Sorted by r ascending, the items with r ≤ 1 form groups of equal r, and every
//! A(λ), λ ≥ 1, is a prefix P_m of whole groups (P_m grows as λ falls; A(1) = every item with
//! r ≤ 1). Let m* be the largest m with P_1 … P_m all good. If m* is every group, λ = 1. Else the
//! first bad prefix P_{m*+1} must be excluded: λ·r_{m*+1} > 1, so λ = the next float above
//! 1/r_{m*+1} (fl(1/r) is within half an ulp of 1/r, its successor strictly above), and ∞ when
//! r_{m*+1} = 0 — no finite factor excludes an atom whose std is 0. Goodness is not monotone in
//! the prefix (a later benign group can bring the rms back), which is why the *first* bad prefix
//! decides: "for every λ′ ≥ λ". The max is a running max; the rms a compensated running sum of the
//! per-point means, each point's old mean taken out and its new one put in as its atoms join; the
//! floor is monotone in the set, so its first prefix is found by bisection. One sort, one pass and
//! O(log n) floor counts.
//!
//! **Which bound r is on** (fix round 1, review I1). The bound is piecewise in |F| (2e-3, 10 %|F|,
//! 1e-2), and the model's force differs from the teacher's by the very error measured: at
//! |F_t| = 0.05 Ha/Bohr and e/b = 4.48 the two bounds differ by up to 1.78×. On the teacher's
//! bound, A(λ_T) was not rule 6's set: at λ_T rule 6 still admitted the measured bad atom. So r is
//! formed on rule 6's own bound, force_bound(|F_m,j|) with |F_m,j| recorded per atom
//! (`AuditPoint::model_force`), the same expression as `variance_check`'s — A(λ) is the set rule 6
//! admits at factor λ (to the rounding of a different evaluation order: λ = next_up(1/r) is not a
//! bit-exact boundary). A point recorded without |F_m| (an older file) uses |F_t,j| + ‖ΔF_j‖ ≥
//! |F_m,j| (the triangle inequality; the bound is non-decreasing in |F|): its r is never above rule
//! 6's, so the atom the max condition names is refused by rule 6 at λ_T, never let through; the
//! cost is a λ_T up to 1.78× larger than rule 6's own bound would need. Never the teacher's bound
//! alone. e stays on the teacher's bound: the error is judged against the policy's claim at the
//! true force, as rule 8 does.
//!
//! **Why the rms only at the floor** (the spec's addition to the review's definition): below it a
//! single admitted atom at 0.8× the bound would fail the rms and make λ_T = ∞ — the knife edge of
//! one draw. The max condition holds at every size. **Why bound (tier) units**: only errors that
//! matter to the policy inflate — a harmless confident point (e = 0.3 b at r = 0.02, a mean of
//! ratios 15) leaves λ_T = 1 (`a_harmless_confident_point_does_not_inflate`). **Why 1× the tier for
//! λ_E** (rev. 2): the gate checks every answer's energy at 1× the tier, so an admitted energy
//! beyond it is an answer already called wrong; rev. 1's 4.53 never fired on the hot run's
//! energy-only failures (e_p 1.03–1.39), and their rms (≈ 0.26, below the floor anyway) gives 1.
//!
//! **NaN and zero (M1).** An item with e = r = 0 adds nothing: not to the max, an rms, a point
//! count or the floor (it is no draw — counted, it would dilute a failing rms). Any NaN r or e in
//! the sample, admitted or not, gives NaN (the inflation is undefined; rule 6/7 refuse a NaN
//! factor). An infinite error is the worst item (∞ at r = 0, else the next float above 1/r). An
//! item with r > 1 (r = ∞ included) is never admitted at λ ≥ 1 and decides nothing.
//!
//! **What it does not do** (C1, change 3): it bounds the tail *of what was measured*. An answer at
//! a geometry like none in P_A is still unmeasured; rule 5's envelope and novelty are for that.
//!
//! # The answer sample P_A: chosen without reading σ or λ (spec §4, the self-reference guard)
//!
//! ```text
//! p ∈ P_A  ⇔  ¬FENCED ∧ ¬NOVEL ∧ ( TICK ∨ (CADENCE_FRESH ∧ class-refused(p)) )
//! class-refused(p)  ⇔  p is a fallback the class rule (4b) named: codes 5, 6 (the class codes) or 21 (the class with rule 6/7 failing too)
//! ```
//!
//! The selection reads the point's flags word and kind only — never its stds, its errors or any λ.
//! (i) **Class-refused in-envelope steps**: the class verdict is decided by measured errors and the
//! LOO, never by this step's σ, and every such step is recorded. The rule that names a refusal is
//! the first of envelope (rule 5: 5a/5b/5e, every step) → novelty (5d) → class (4b) → variance
//! (6/7) → the fences; so a class code means "the class refused and neither the envelope nor the
//! novelty did". The envelope is evaluated every step (σ-blind). The novelty is evaluated only
//! with the variance, whose phase follows σ — so on a reused verdict a novel geometry the class
//! refuses carries a class code, on a fresh one 5d's code 11. The provider therefore evaluates the
//! novelty on **every cadence-fresh step**, read-only where the verdict was reused, and records
//! `NOVEL` (Task 3 review MUST, carried; `surrogate` module doc, "The history's flags"): with the
//! bit, a cadence-fresh point is in P_A exactly when the class refused it, the envelope and the
//! novelty did not and no fence holds it — the predicate, whichever code the precedence chain
//! wrote. Below the class rule only 6/7 remain, and 5/6 versus 21 (whether a stale or fresh
//! variance verdict also refused) is inside P_A either way. (ii) **Calibration ticks** (Task 7):
//! σ-blind by their own counter; a fenced or novel tick is still left out.
//!
//! **Left out**: ordinary audits, codes 14/15 and 24 (λ selects them), fenced points (`FENCED`:
//! codes 5/6/21 can hold geometries the key will never answer), novel ones, and class-refused
//! points fresh only by a refusal (not `CADENCE_FRESH`). **Draft mode**: a draft-mode step is never
//! cadence-fresh for the history (its only consulted-model points are relax verifications, coded
//! by the rule that fired first, and background audits), and a draft guess's audit is never
//! recorded (`Unrecorded::draft`; its sample is a `DraftAudit`). **Legacy points** (flags 0) are
//! never in P_A.
//!
//! **The window**: the same rule as the verdict's and λ_sum's — the newest whole generations of
//! the key's (class's) own P_A points holding at least [`WINDOW_POINTS`] — over P_A itself.
//! `ClassCalibration::answer_points` is the window's P_A count (the sparse-key rule, Task 6). Each
//! point counts for every key it holds an atom of, like the LOO energy; the floor counts a point
//! only with species and a fit id, like the evidence floor's.

use sim::activity::{EnvClass, TeachRule};
use sim::ClassKey;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use surrogate::history::{AuditHistory, AuditPoint, PointFlags, PointKind};
use surrogate::EnvKey;

use super::FallbackPolicy;

/// An untrusted class becomes trusted when its verdict values (force and energy) are at most
/// this: 1/√2 — half the bound's variance.
pub const ENTER: f64 = std::f64::consts::FRAC_1_SQRT_2;
/// A trusted class stays trusted while its verdict values are at most this: the bound itself.
pub const LEAVE: f64 = 1.0;
/// Evidence points (audits plus the capped rule-4b fallback weight) a class needs before the
/// calibrated verdict judges it; below it the rule of before stands, with the [`WORST_ATOM`] guard.
pub const MIN_EVIDENCE: usize = 3;
/// One jurisdiction point in the window with an atom's measured force error beyond this many
/// times its bound keeps an untrusted class from entering (at every evidence count) and, with
/// its key below the evidence floor ([`FLOOR_FITS`] fits and [`FLOOR_POINTS`] points), makes a trusted one leave: P(χ²₃/3 > 9) = 5.9e-6 per atom
/// for a calibrated posterior at std = bound.
pub const WORST_ATOM: f64 = 3.0;
/// The evidence window (module doc, "The evidence window"): the newest whole generations of a
/// class's jurisdiction points holding at least this many points — enough that a window with no
/// atom beyond [`WORST_ATOM`] makes a 3× tail at 1 % unlikely (0.99ⁿ ≤ 5 %: n ≥ 298), which also
/// keeps two sigma of the rms's own scatter, 2/√(2n), far inside the hysteresis band √2 − 1.
pub const WINDOW_POINTS: usize = 300;
/// The evidence floor's fits (trust tightening, spec 2026-11-18 §3; module doc, "The evidence
/// floor"): a key's calibrated verdict opens it alone only over at least this many distinct fits
/// among its attributed verdict points in the window — one decorrelated block per fit, and two
/// sigma of the rms's scatter inside the √2 band needs n ≥ 5.83·CV² = 4.9 blocks at the measured
/// block CV² 0.84.
pub const FLOOR_FITS: usize = 12;
/// The evidence floor's points: at least this many attributed verdict points carrying a fit id in
/// the window — the per-answer CV² proxy of 10 needs n ≥ 5.83 × 10 = 59 (fits of 1–3 points, as in
/// the app, where the worker refits for every kept sample).
pub const FLOOR_POINTS: usize = 60;

const SQRT_3: f64 = 1.732_050_807_568_877_2;

/// One class's calibration from the history (module doc).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClassCalibration {
    /// κ_c: measured over predicted, ≥ 1 (1 without any evidence; ∞ when a comparison in the
    /// evidence was not a number). A readout; it decides nothing.
    pub factor: f64,
    /// Audit points of the class in the window.
    pub audits: usize,
    /// Rule-4b fallback points of the class in the window (before the weight cap).
    pub fallbacks: usize,
    /// n_c = Σw: the audits plus the capped fallback weight, in points — an integer, formed
    /// exactly (module doc).
    pub evidence: usize,
    /// The jurisdiction points of the class in the window (audits + fallbacks): what the
    /// inflation counts.
    pub points: usize,
    /// The class's jurisdiction points older than the window (aged out; module doc).
    pub aged: usize,
    /// The oldest model generation in the window, `None` when the window is the whole ring
    /// (fewer than [`WINDOW_POINTS`] points of the class held) or there is no evidence.
    pub window_from: Option<u64>,
    /// ẑ_c before the floor and the inflation (NaN without evidence).
    pub z_rms: f64,
    /// W_c · (1 + 1/√(2n)): the measured force error in bound units, inflated (NaN without
    /// evidence) — the verdict's force value beside the LOO floor.
    pub measured: f64,
    /// W_E,c · (1 + 1/√(2n)): the measured energy error in tier units, inflated (NaN without
    /// evidence).
    pub measured_energy: f64,
    /// A_c: the largest per-atom measured force error over the window's points, in bound units
    /// (the [`WORST_ATOM`] guard's input; 0 without evidence, ∞ for a comparison that is not a
    /// number).
    pub worst_atom: f64,
    /// λ_c, the variance inflation rules 6 and 7 apply to the class's atoms (fix round 3; module
    /// doc, "The variance inflation"): ≥ 1, exactly 1 with fewer than [`MIN_EVIDENCE`] points in
    /// its window, ∞ when a comparison there was not a number.
    pub inflation: f64,
    /// The points behind `inflation` (its own window over the posterior's jurisdiction).
    pub inflation_points: usize,
    /// √(Σ w̄² / Σ s̄²) before the floor at 1 and the scatter factor (NaN with fewer than
    /// [`MIN_EVIDENCE`] points): the measured error over the epistemic std, both in bound units.
    pub inflation_raw: f64,
    /// The evidence floor's n_fit: the distinct fit ids among the window's verdict points that
    /// carry species and a fit id (module doc, "The evidence floor").
    pub fits: usize,
    /// The evidence floor's n_pts: the window's verdict points (audits and rule-4b fallbacks,
    /// uncapped) that carry species and a fit id.
    pub floor_points: usize,
    /// λ_T, the tail-aware force inflation over the answer sample P_A (module doc, "The tail-aware
    /// inflations"): ≥ 1, ∞ when a bad atom has s = 0, NaN when a comparison in P_A is NaN.
    pub tail_inflation: f64,
    /// λ_E, the tail-aware energy inflation over P_A's points with an epistemic energy std (module
    /// doc): ≥ 1, ∞ / NaN as λ_T.
    pub energy_inflation: f64,
    /// The P_A points in λ_T's window (the sparse-key rule reads it: fewer than [`MIN_EVIDENCE`]).
    pub answer_points: usize,
    /// What decided `tail_inflation` (the refusal's words name it; Task 6).
    pub tail_cause: TailCause,
    /// What decided `energy_inflation`.
    pub energy_cause: TailCause,
}

impl ClassCalibration {
    /// Enough evidence for the calibrated verdict ([`MIN_EVIDENCE`]).
    pub fn calibrated(&self) -> bool {
        self.evidence >= MIN_EVIDENCE
    }

    /// The evidence floor is met ([`FLOOR_FITS`] distinct fits **and** [`FLOOR_POINTS`] points):
    /// the calibrated verdict may open the key alone (module doc, "The evidence floor"). Met
    /// implies [`calibrated`](Self::calibrated): 60 points weigh at least min(60, 3) evidence.
    pub fn floor_met(&self) -> bool {
        self.fits >= FLOOR_FITS && self.floor_points >= FLOOR_POINTS
    }
}

/// The calibration: class-wide (indexed like `atom_class` / `ENV_CLASSES`: 0 pair, 1 many-body; a
/// readout) and per environment key (module doc, "Per environment species": what decides).
#[derive(Clone, Debug, PartialEq)]
pub struct Calibration {
    pub per_class: [ClassCalibration; 2],
    /// Per key: its attributed points with the class's species-less points.
    pub per_key: BTreeMap<EnvKey, ClassCalibration>,
    /// Per class: the species-less points alone — what a key without attributed points reads.
    pub legacy: [ClassCalibration; 2],
}

impl Calibration {
    /// The calibration a key's verdict reads: its own, or its class's species-less points for a
    /// key with no attributed point.
    pub fn key(&self, k: &EnvKey) -> ClassCalibration {
        self.per_key.get(k).copied().unwrap_or(self.legacy[(k.class.clamp(1, 2) - 1) as usize])
    }
}

/// The per-class leave-one-out statistics the verdict reads (what
/// `SurrogateKs::compute_class_loo_full` measures): ratios to the policy's bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClassStats {
    pub samples: usize,
    pub force_rms: f64,
    pub force_max: f64,
    pub energy_rms: f64,
    pub energy_max: f64,
    /// The misfit set M's energy samples in `energy_rms` (open-shell Task 7c): the energy-RMS
    /// guard ([`ClassKey::EnergyRms`]) is judged only when this is above 0.
    pub energy_misfits: usize,
}

/// The energy half of C1 (open-shell model Task 7c, the user's decision 2026-11-13): the energy
/// LOO RMS, M included, beyond the tier — only when an M sample contributes to it, so with M
/// empty no verdict changes. NaN or ∞ with M contributing fails (not within the limit).
#[allow(clippy::neg_cmp_op_on_partial_ord)] // the negation is the point: NaN fails
fn energy_rms_fails(st: &ClassStats) -> bool {
    st.energy_misfits > 0 && !(st.energy_rms <= 1.0)
}

/// The verdict for one class.
#[derive(Clone, Debug, PartialEq)]
pub enum ClassVerdict {
    Trusted,
    /// The first failing condition (its key, value and limit — `ClassLoo::refusal`) and the reason
    /// in the refusal's words ([`refusal_words`]: the part after "isn't learned well enough yet
    /// (<class>: ").
    Untrusted { key: ClassKey, value: f64, limit: f64, reason: String },
}

/// A history point in the verdict's jurisdiction: an audit, or a fallback refused by rule 4b.
pub(super) fn in_jurisdiction(kind: PointKind) -> bool {
    match kind {
        PointKind::Audit => true,
        PointKind::Fallback { rule } => {
            rule == super::history_rule_code(TeachRule::Class(EnvClass::Pair)) || rule == super::history_rule_code(TeachRule::Class(EnvClass::ManyBody))
        }
    }
}

/// A non-finite square sum: the worst evidence, never dropped (Task 3b).
fn sq(x: f64) -> f64 {
    if x.is_finite() { x * x } else { f64::INFINITY }
}

/// Per point and class: (w̄², z̄², the energy term, the worst atom ratio, s̄² the epistemic force
/// std in bound units), its generation and kind.
#[derive(Clone, Copy)]
struct PointEvidence {
    gen: u64,
    audit: bool,
    w2: f64,
    z2: f64,
    e2: f64,
    worst: f64,
    s2: f64,
    /// ā̄² = mean over the point's class atoms of (ā_c/b_j)²: the aleatoric part of w̄² (fix round 1)
    a2: f64,
    /// The point's fit id when it counts toward the evidence floor: it carries species and a fit
    /// id (module doc, "The evidence floor"); `None` otherwise.
    floor_fit: Option<u64>,
}

/// A history point in the **posterior's** jurisdiction (fix round 3, the variance inflation): the
/// geometries where the model's predictive std decided or would have decided — the class
/// verdict's jurisdiction (audits, rule-4b fallbacks) plus the fallbacks rules 6/7 refused alone
/// (codes 14, 15; code 24 where only the tail-aware factor refused, trust tightening Task 6) and
/// those the class and rules 6/7 refused together (code 21). Not rules
/// 5/5a–5e or the fences: those refuse whatever the std says (the extrapolating posterior's z rms
/// 12–54 there argues for those rules, not for the std inside their reach).
fn in_posterior_jurisdiction(kind: PointKind) -> bool {
    match kind {
        PointKind::Audit => true,
        PointKind::Fallback { rule } => {
            in_jurisdiction(kind)
                || rule == super::history_rule_code(TeachRule::ForceUncertainty)
                || rule == super::history_rule_code(TeachRule::EnergyUncertainty)
                || rule == super::HISTORY_RULE_TAIL_INFLATION
                || rule == super::HISTORY_RULE_CLASS_AND_VARIANCE
        }
    }
}

/// The oldest generation of the window (module doc, "The evidence window"): from the newest
/// generation back, whole generations until they hold [`WINDOW_POINTS`]; `None` = every point.
fn window_from(points: &[&PointEvidence]) -> Option<u64> {
    window_from_gens(points.iter().map(|p| p.gen))
}

/// [`window_from`] over the points' generations.
fn window_from_gens(gens: impl Iterator<Item = u64>) -> Option<u64> {
    let mut per_gen: BTreeMap<u64, usize> = BTreeMap::new();
    for g in gens {
        *per_gen.entry(g).or_default() += 1;
    }
    let mut held = 0usize;
    for (&gen, &n) in per_gen.iter().rev() {
        held += n;
        if held >= WINDOW_POINTS {
            return Some(gen);
        }
    }
    None
}

/// The calibration per class from the history and the policy's bounds (module doc), with no
/// aleatoric level removed from the inflation's numerator ([`calibrate_with`] with ā = 0) and no
/// keys beyond the history's own.
pub fn calibrate(history: &AuditHistory, policy: &FallbackPolicy) -> Calibration {
    calibrate_with(history, policy, [0.0; 2], &BTreeSet::new())
}

/// One point's evidence over the atoms `atoms` (all of one class, aleatoric level `aleatoric`).
fn point_evidence(p: &AuditPoint, atoms: &[usize], aleatoric: f64, policy: &FallbackPolicy) -> Option<PointEvidence> {
    if atoms.is_empty() {
        return None;
    }
    let (mut w2, mut z2, mut worst, mut s2, mut a2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for &j in atoms {
        let bound = policy.force_bound(p.teacher_force[j]);
        let w = p.err_force[j] / SQRT_3 / bound;
        w2 += sq(w);
        z2 += sq(p.err_force[j] / p.pred_force_std[j]);
        s2 += sq(p.pred_force_std_epistemic[j] / SQRT_3 / bound);
        a2 += sq(aleatoric / bound);
        worst = if w.is_finite() { worst.max(w) } else { f64::INFINITY };
    }
    let n = atoms.len() as f64;
    Some(PointEvidence {
        gen: p.model_gen,
        audit: p.kind.is_audit(),
        w2: w2 / n,
        z2: z2 / n,
        // per N_e through the one normalisation (`policy::energy_atoms`; a point's `n_atoms` is its N_e)
        e2: sq(p.err_energy.abs() / super::policy::energy_atoms(p.n_atoms.max(1)) / policy.energy_std_per_atom),
        worst,
        s2: s2 / n,
        a2: a2 / n,
        // a species-less (legacy) point or one without a fit id never counts toward the floor
        floor_fit: p.fit_id.filter(|_| !p.species.is_empty()),
    })
}

/// The verdict's and the posterior's evidence lists of one class or key, and its answer sample
/// (module doc, "The answer sample P_A").
#[derive(Default)]
struct Lists {
    verdict: Vec<PointEvidence>,
    posterior: Vec<PointEvidence>,
    answer: Vec<AnswerEvidence>,
}

impl Lists {
    fn push(&mut self, e: PointEvidence, verdict: bool) {
        if verdict {
            self.verdict.push(e);
        }
        self.posterior.push(e);
    }
}

/// [`calibrate`] with each class's aleatoric force level ā_c (Ha/Bohr per component; module doc,
/// "The variance inflation": min(σ_F, the class's training force-residual rms)) removed from the
/// inflation's measured errors, and a calibration for every key in `keys` (the fit's own atoms'
/// keys) besides the history's attributed ones (module doc, "Per environment species").
pub fn calibrate_with(history: &AuditHistory, policy: &FallbackPolicy, aleatoric: [f64; 2], keys: &BTreeSet<EnvKey>) -> Calibration {
    let mut class: [Lists; 2] = Default::default();
    let mut legacy: [Lists; 2] = Default::default();
    let mut by_key: BTreeMap<EnvKey, Lists> = BTreeMap::new();
    for p in history.iter() {
        // the posterior's jurisdiction (λ_sum, the verdict) and the answer sample (λ_T, λ_E) are
        // selected apart: a tick may be a point of either kind
        let posterior = in_posterior_jurisdiction(p.kind);
        let answer = in_answer_sample(p);
        if !posterior && !answer {
            continue;
        }
        let verdict = in_jurisdiction(p.kind);
        for c in 0..2 {
            let code = c as u8 + 1;
            let atoms: Vec<usize> = (0..p.n_atoms).filter(|&j| p.classes[j] == code).collect();
            if posterior {
                if let Some(e) = point_evidence(p, &atoms, aleatoric[c], policy) {
                    class[c].push(e, verdict);
                    if p.species.is_empty() {
                        legacy[c].push(e, verdict);
                    }
                }
            }
            if answer {
                if let Some(a) = answer_evidence(p, &atoms, policy) {
                    if p.species.is_empty() {
                        legacy[c].answer.push(a.clone());
                    }
                    class[c].answer.push(a);
                }
            }
        }
        let mut atoms_of: BTreeMap<EnvKey, Vec<usize>> = BTreeMap::new();
        for j in 0..p.n_atoms {
            if let Some(k) = p.key(j) {
                atoms_of.entry(k).or_default().push(j);
            }
        }
        for (k, atoms) in atoms_of {
            let c = (k.class - 1) as usize;
            if posterior {
                if let Some(e) = point_evidence(p, &atoms, aleatoric[c], policy) {
                    by_key.entry(k.clone()).or_default().push(e, verdict);
                }
            }
            if answer {
                if let Some(a) = answer_evidence(p, &atoms, policy) {
                    by_key.entry(k).or_default().answer.push(a);
                }
            }
        }
    }
    fn refs<T>(l: &[T]) -> Vec<&T> {
        l.iter().collect()
    }
    let per_class = std::array::from_fn(|c| summarise(&refs(&class[c].verdict), &refs(&class[c].posterior), &refs(&class[c].answer)));
    let legacy_cal: [ClassCalibration; 2] = std::array::from_fn(|c| summarise(&refs(&legacy[c].verdict), &refs(&legacy[c].posterior), &refs(&legacy[c].answer)));
    let all: BTreeSet<EnvKey> = keys.iter().filter(|k| (1..=2).contains(&k.class)).cloned().chain(by_key.keys().cloned()).collect();
    let empty = Lists::default();
    let per_key = all
        .into_iter()
        .map(|k| {
            let own = by_key.get(&k).unwrap_or(&empty);
            let old = &legacy[(k.class - 1) as usize];
            let v: Vec<&PointEvidence> = own.verdict.iter().chain(&old.verdict).collect();
            let q: Vec<&PointEvidence> = own.posterior.iter().chain(&old.posterior).collect();
            let a: Vec<&AnswerEvidence> = own.answer.iter().chain(&old.answer).collect();
            (k, summarise(&v, &q, &a))
        })
        .collect();
    Calibration { per_class, per_key, legacy: legacy_cal }
}

/// One class's or key's calibration from its verdict and posterior evidence and its answer sample
/// (module doc).
fn summarise(all: &[&PointEvidence], posterior: &[&PointEvidence], answer: &[&AnswerEvidence]) -> ClassCalibration {
    let (inflation_var, inflation_points, inflation_raw) = variance_inflation(posterior);
    let ((tail_inflation, tail_cause), (energy_inflation, energy_cause), answer_points) = tail_inflations(answer);
    let from = window_from(all);
    let inside = |e: &&&PointEvidence| from.is_none_or(|g| e.gen >= g);
    let audits: Vec<&PointEvidence> = all.iter().filter(inside).filter(|e| e.audit).copied().collect();
    let fallbacks: Vec<&PointEvidence> = all.iter().filter(inside).filter(|e| !e.audit).copied().collect();
    let (n_a, n_f) = (audits.len(), fallbacks.len());
    let points = n_a + n_f;
    let aged = all.len() - points;
    // the evidence floor: the window's verdict points carrying species and a fit id, uncapped, and
    // their distinct fits
    let floor_ids: Vec<u64> = audits.iter().chain(&fallbacks).filter_map(|e| e.floor_fit).collect();
    let floor_points = floor_ids.len();
    let fits = floor_ids.iter().collect::<BTreeSet<_>>().len();
    if points == 0 {
        return ClassCalibration { factor: 1.0, audits: 0, fallbacks: 0, evidence: 0, points: 0, aged, window_from: None, z_rms: f64::NAN, measured: f64::NAN, measured_energy: f64::NAN, worst_atom: 0.0, inflation: inflation_var, inflation_points, inflation_raw, fits, floor_points, tail_inflation, energy_inflation, answer_points, tail_cause, energy_cause };
    }
    // the fallbacks of the class together weigh at most the audits (never fewer than three);
    // the count in integers (review I1), the per-point weight only for the sums
    let cap = n_a.max(MIN_EVIDENCE);
    let evidence = n_a + n_f.min(cap);
    let w_f = if n_f == 0 { 0.0 } else { n_f.min(cap) as f64 / n_f as f64 };
    let den = evidence as f64;
    let sum = |f: &dyn Fn(&PointEvidence) -> f64| audits.iter().map(|e| f(e)).sum::<f64>() + w_f * fallbacks.iter().map(|e| f(e)).sum::<f64>();
    let inflation = 1.0 + 1.0 / (2.0 * points as f64).sqrt();
    let z_rms = (sum(&|e| e.z2) / den).sqrt();
    let measured = (sum(&|e| e.w2) / den).sqrt() * inflation;
    let measured_energy = (sum(&|e| e.e2) / den).sqrt() * inflation;
    let worst_atom = audits.iter().chain(&fallbacks).map(|e| e.worst).fold(0.0f64, |m, w| if w.is_nan() || m.is_nan() { f64::NAN } else { m.max(w) });
    ClassCalibration { factor: z_rms.max(1.0) * inflation, audits: n_a, fallbacks: n_f, evidence, points, aged, window_from: from, z_rms, measured, measured_energy, worst_atom, inflation: inflation_var, inflation_points, inflation_raw, fits, floor_points, tail_inflation, energy_inflation, answer_points, tail_cause, energy_cause }
}

/// λ_c (module doc, "The variance inflation"): over the class's points in the posterior's
/// jurisdiction, in their own window of the newest generations ([`WINDOW_POINTS`]), one draw per
/// point, λ = max(1, √(Σ w̄² / Σ s̄²)) · (1 + 1/√(2n)) — the measured force error over the
/// epistemic std, both in the policy's bound units, as a ratio of sums (not a mean of ratios: a
/// confident point with a small error adds little to either sum, so no point's tiny std can set
/// the class's factor — the physics review's I2). 1 with fewer than [`MIN_EVIDENCE`] points; 1
/// when every std is exactly 0 (rule 6 then compares 0 whatever the factor); ∞ when a
/// comparison was not a number. Returns (λ, n, the raw ratio).
fn variance_inflation(points: &[&PointEvidence]) -> (f64, usize, f64) {
    let from = window_from(points);
    let inside: Vec<&PointEvidence> = points.iter().filter(|e| from.is_none_or(|g| e.gen >= g)).copied().collect();
    let n = inside.len();
    if n < MIN_EVIDENCE {
        return (1.0, n, f64::NAN);
    }
    // the measured error less its aleatoric part, per point and never below 0 (fix round 1, the
    // review's I2): E[w̄²] = s̄² + ā̄² for a calibrated posterior at a fresh geometry
    let sw: f64 = inside.iter().map(|e| (e.w2 - e.a2).max(0.0)).sum();
    let ss: f64 = inside.iter().map(|e| e.s2).sum();
    if !(sw.is_finite() && ss.is_finite()) {
        return (f64::INFINITY, n, f64::INFINITY);
    }
    if ss == 0.0 {
        return (1.0, n, f64::NAN);
    }
    let raw = (sw / ss).sqrt();
    (raw.max(1.0) * (1.0 + 1.0 / (2.0 * n as f64).sqrt()), n, raw)
}

// ---- the tail-aware inflations (module doc, "The tail-aware inflations"; spec 2026-11-18 §4) ----

/// Whether `p` is in the answer sample P_A (module doc, "The answer sample P_A"): not fenced, not
/// novel, and a calibration tick or a cadence-fresh class-refused point (codes 5, 6, 21). Reads the
/// point's flags word and kind only — never a std, an error or a λ (the self-reference guard).
pub(super) fn in_answer_sample(p: &AuditPoint) -> bool {
    let f = p.flags;
    let tick = f.has(PointFlags::TICK);
    !f.has(PointFlags::FENCED) && !f.has(PointFlags::NOVEL) && (tick || (f.has(PointFlags::CADENCE_FRESH) && class_refused(p.kind)))
}

/// The class rule (4b) named the refusal: a class code (5, 6) or the class with rule 6/7 failing
/// too (21) — from the provider's own code table, never literal numbers.
fn class_refused(kind: PointKind) -> bool {
    match kind {
        PointKind::Audit => false,
        PointKind::Fallback { rule } => in_jurisdiction(kind) || rule == super::HISTORY_RULE_CLASS_AND_VARIANCE,
    }
}

/// One measured atom of the answer sample, in the policy's bound units (module doc): r = s_j/b_j,
/// e = e_j/b_j (s_j = σ_epi,j/√3, e_j = ‖ΔF_j‖/√3), of point `point` judged by fit `fit` (`None`:
/// it does not count toward the floor).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TailAtom {
    pub point: usize,
    pub fit: Option<u64>,
    pub r: f64,
    pub e: f64,
}

/// One measured point of the answer sample, in tier units (module doc): r = σ_E,epi/N/tier,
/// e = |ΔE|/N/tier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TailPoint {
    pub point: usize,
    pub fit: Option<u64>,
    pub r: f64,
    pub e: f64,
}

/// λ_E's max condition: an admitted point's measured energy error is at most 1× the tier (spec §4
/// rev. 2 — the gate's own per-answer criterion).
pub const ENERGY_MAX: f64 = 1.0;

/// What decided a tail-aware inflation (module doc, "The one pass"): the first bad prefix's failing
/// condition and the r of the group that made it bad — what a refusal at the factor names (Task 6).
/// All values are ratios: bound units for λ_T, tier units for λ_E.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum TailCause {
    /// Every admitted set is good: λ = 1.
    #[default]
    None,
    /// The max condition: an item of the decisive group at `e`× the bound (tier) with the model's std
    /// `r`× of it (the previous prefix was good, so the running max that fails is this group's).
    Max { e: f64, r: f64 },
    /// The rms condition on a set meeting the floor: its rms × (1 + 1/√(2n)) — the measured rms with
    /// its small-sample margin, not the measured rms itself — is `value` (> [`ENTER`])
    /// over `points` points, the decisive group's std `r`× the bound (tier).
    Rms { value: f64, points: usize, r: f64 },
    /// A comparison in the sample is not a number: λ = NaN.
    NotANumber,
}

/// What [`tail_factor`] reads of an item.
trait Tail: Copy {
    fn point(&self) -> usize;
    fn fit(&self) -> Option<u64>;
    fn r(&self) -> f64;
    fn e(&self) -> f64;
}

impl Tail for TailAtom {
    fn point(&self) -> usize {
        self.point
    }
    fn fit(&self) -> Option<u64> {
        self.fit
    }
    fn r(&self) -> f64 {
        self.r
    }
    fn e(&self) -> f64 {
        self.e
    }
}

impl Tail for TailPoint {
    fn point(&self) -> usize {
        self.point
    }
    fn fit(&self) -> Option<u64> {
        self.fit
    }
    fn r(&self) -> f64 {
        self.r
    }
    fn e(&self) -> f64 {
        self.e
    }
}

/// The evidence floor over a set of items: ≥ [`FLOOR_FITS`] distinct fits and ≥ [`FLOOR_POINTS`]
/// distinct points among the items that carry a fit id. Monotone in the set.
fn meets_floor<T: Tail>(items: &[T]) -> bool {
    let mut points = BTreeSet::new();
    let mut fits = BTreeSet::new();
    for t in items {
        if let Some(f) = t.fit() {
            points.insert(t.point());
            fits.insert(f);
        }
    }
    fits.len() >= FLOOR_FITS && points.len() >= FLOOR_POINTS
}

/// The floor over a set of answer-sample atoms ([`tail_inflation`]'s `floor_met` in `calibrate`).
pub fn atoms_meet_floor(atoms: &[TailAtom]) -> bool {
    meets_floor(atoms)
}

/// The floor over a set of answer-sample points ([`energy_inflation`]'s `floor_met`).
pub fn points_meet_floor(points: &[TailPoint]) -> bool {
    meets_floor(points)
}

/// λ_T over a key's answer-sample atoms (module doc, "The tail-aware inflations"): the max
/// condition at [`WORST_ATOM`] = 3, the rms condition at [`ENTER`] on the sets `floor_met` (monotone
/// in the set) accepts. ≥ 1; ∞ when a bad atom has r = 0; NaN when any r or e is NaN.
pub fn tail_inflation(atoms: &[TailAtom], floor_met: impl Fn(&[TailAtom]) -> bool) -> f64 {
    tail_factor(atoms, WORST_ATOM, floor_met).0
}

/// λ_E over a key's answer-sample points with an epistemic energy std: as [`tail_inflation`], the
/// max condition at [`ENERGY_MAX`] = 1× the tier.
pub fn energy_inflation(points: &[TailPoint], floor_met: impl Fn(&[TailPoint]) -> bool) -> f64 {
    tail_factor(points, ENERGY_MAX, floor_met).0
}

/// [`energy_inflation`] with another max condition: a test's counterfactual (rev. 1's 4.53).
#[cfg(test)]
pub(super) fn energy_inflation_with_max(points: &[TailPoint], max: f64, floor_met: impl Fn(&[TailPoint]) -> bool) -> f64 {
    tail_factor(points, max, floor_met).0
}

/// Neumaier's compensated running sum (CLAUDE.md rule 4): the rms condition's Σ of per-point means,
/// whose terms are taken out and put back as a point's atoms join.
#[derive(Clone, Copy, Default)]
struct Compensated {
    sum: f64,
    c: f64,
}

impl Compensated {
    fn add(&mut self, x: f64) {
        let t = self.sum + x;
        if self.sum.abs() >= x.abs() {
            self.c += (self.sum - t) + x;
        } else {
            self.c += (x - t) + self.sum;
        }
        self.sum = t;
    }

    fn value(&self) -> f64 {
        self.sum + self.c
    }
}

/// The smallest λ ≥ 1 with A(λ′) good for every λ′ ≥ λ (module doc, "The tail-aware inflations",
/// "The one pass"): `max` the max condition's limit, the rms condition at [`ENTER`] on the prefixes
/// `floor_met` accepts. With it, what decided ([`TailCause`]).
fn tail_factor<T: Tail>(items: &[T], max: f64, floor_met: impl Fn(&[T]) -> bool) -> (f64, TailCause) {
    // any NaN, admitted or not: the inflation is undefined (M1)
    if items.iter().any(|t| t.r().is_nan() || t.e().is_nan()) {
        return (f64::NAN, TailCause::NotANumber);
    }
    // admitted at some λ ≥ 1 (λ·r ≤ 1 needs r ≤ 1), and not e = r = 0 (no draw: adds nothing)
    let mut a: Vec<T> = items.iter().copied().filter(|t| t.r() <= 1.0 && !(t.r() == 0.0 && t.e() == 0.0)).collect();
    a.sort_by(|x, y| x.r().total_cmp(&y.r()));
    // the first prefix length that meets the floor (the floor is monotone in the set): bisection
    let first_floor = {
        let (mut lo, mut hi) = (0usize, a.len() + 1);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if floor_met(&a[..mid]) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo
    };
    let mut worst = 0.0f64;
    // per point: Σ e² over its admitted atoms and their count; Σ over points of their means
    let mut per_point: HashMap<usize, (Compensated, usize)> = HashMap::new();
    let mut total = Compensated::default();
    let mut i = 0;
    while i < a.len() {
        // the next group of equal r enters together
        let r = a[i].r();
        let mut j = i;
        while j < a.len() && a[j].r() == r {
            let e = a[j].e();
            worst = worst.max(e); // no NaN here
            let (sum, n) = per_point.entry(a[j].point()).or_default();
            if *n > 0 {
                total.add(-(sum.value() / *n as f64));
            }
            sum.add(sq(e));
            *n += 1;
            total.add(sum.value() / *n as f64);
            j += 1;
        }
        // (an infinite e fails the max first, so a NaN the sum may then hold is never read)
        let cause = if !(worst <= max) {
            Some(TailCause::Max { e: worst, r })
        } else if j < first_floor {
            None
        } else {
            let n_a = per_point.len() as f64;
            let v = total.value() / n_a;
            let v = if v < 0.0 { 0.0 } else { v }; // rounding below 0; a NaN stays NaN and fails
            let value = v.sqrt() * (1.0 + 1.0 / (2.0 * n_a).sqrt());
            (!(value <= ENTER)).then_some(TailCause::Rms { value, points: per_point.len(), r })
        };
        if let Some(cause) = cause {
            // λ·r > 1 excludes this group; no finite λ excludes r = 0
            return (if r <= 0.0 { f64::INFINITY } else { (1.0 / r).next_up() }, cause);
        }
        i = j;
    }
    (1.0, TailCause::None)
}

/// One answer-sample point's measurements over the atoms of one class or key (module doc): per
/// atom (r_j, e_j) in bound units, and (r_p, e_p) in tier units when it carries an epistemic
/// energy std; its generation (the window) and its fit id when it counts toward the floor.
#[derive(Clone)]
struct AnswerEvidence {
    gen: u64,
    fit: Option<u64>,
    atoms: Vec<(f64, f64)>,
    energy: Option<(f64, f64)>,
}

fn answer_evidence(p: &AuditPoint, atoms: &[usize], policy: &FallbackPolicy) -> Option<AnswerEvidence> {
    if atoms.is_empty() {
        return None;
    }
    let per_atom = atoms
        .iter()
        .map(|&j| {
            // e on the teacher's bound (rule 8's); r on rule 6's own, force_bound(|F_m|) — or, for a
            // point recorded without |F_m|, the bound at |F_t| + ‖ΔF‖ ≥ |F_m| (the triangle
            // inequality; the bound is non-decreasing), never larger r than rule 6's (module doc)
            let bound = policy.force_bound(p.teacher_force[j]);
            let rule6 = policy.force_bound(p.model_force.get(j).copied().unwrap_or(p.teacher_force[j] + p.err_force[j]));
            (p.pred_force_std_epistemic[j] / SQRT_3 / rule6, p.err_force[j] / SQRT_3 / bound)
        })
        .collect();
    // N: rule 7's own normalisation, through the one shared function (spec §5; `atoms` is not empty,
    // so n_atoms ≥ 1). A point is recorded over its bonded part (`record_history_point`), so its n_atoms
    // is its N_e, the energy-row atoms rule 7 divides by (stray-atoms ruling 9a)
    let (n, tier) = (super::policy::energy_atoms(p.n_atoms), policy.energy_std_per_atom);
    Some(AnswerEvidence {
        gen: p.model_gen,
        // as the evidence floor: a point counts only with species and a fit id
        fit: p.fit_id.filter(|_| !p.species.is_empty()),
        atoms: per_atom,
        energy: p.energy_std_epistemic.map(|s| (s / n / tier, p.err_energy.abs() / n / tier)),
    })
}

/// (λ_T, λ_E — each with its cause — and the P_A points in the window) of one class or key: over its
/// own window of the newest generations holding [`WINDOW_POINTS`] of its P_A points (module doc,
/// "The window").
fn tail_inflations(points: &[&AnswerEvidence]) -> ((f64, TailCause), (f64, TailCause), usize) {
    let from = window_from_gens(points.iter().map(|p| p.gen));
    let inside: Vec<&AnswerEvidence> = points.iter().filter(|p| from.is_none_or(|g| p.gen >= g)).copied().collect();
    let atoms: Vec<TailAtom> = inside.iter().enumerate().flat_map(|(i, p)| p.atoms.iter().map(move |&(r, e)| TailAtom { point: i, fit: p.fit, r, e })).collect();
    let energies: Vec<TailPoint> = inside.iter().enumerate().filter_map(|(i, p)| p.energy.map(|(r, e)| TailPoint { point: i, fit: p.fit, r, e })).collect();
    (tail_factor(&atoms, WORST_ATOM, atoms_meet_floor), tail_factor(&energies, ENERGY_MAX, points_meet_floor), inside.len())
}

/// What the refusal's words read (a `ClassLoo` holds all of it; `class_verdict` passes its own).
pub struct WordsInput {
    pub samples: usize,
    pub force_rms: f64,
    pub force_max: f64,
    pub energy_max: f64,
    /// The measured force / energy errors (inflated), `None` without evidence.
    pub measured: Option<f64>,
    pub measured_energy: Option<f64>,
    /// The evidence count and the calibration factor κ.
    pub evidence: usize,
    pub calibration: f64,
    /// The evidence floor's counts ([`ClassCalibration::fits`], [`ClassCalibration::floor_points`]).
    pub fits: usize,
    pub floor_points: usize,
}

/// The words of a class refusal (key, value, limit): the part after "isn't learned well enough yet
/// (<class>: " — the verdict's `reason` and the provider's Log note alike (one wording, review M2).
/// One count of the evidence floor in words (trust tightening Task 8; the Task 4 review's minor 3:
/// never "80 of 60 checks"): "11 of 12 fits" while it is short of its floor, "80 checks (60
/// needed)" once it reaches it. `unit` may be empty ("4 of 60", "80 (60 needed)") where the
/// sentence names it. The provider's refusal words and the app's panel read this one function.
pub fn floor_count(n: usize, floor: usize, unit: &str) -> String {
    let unit = if unit.is_empty() { String::new() } else { format!(" {unit}") };
    if n < floor { format!("{n} of {floor}{unit}") } else { format!("{n}{unit} ({floor} needed)") }
}

/// The uncalibrated keys keep their words of before (pinned byte for byte by the refusal-string
/// tests).
pub fn refusal_words(key: ClassKey, value: f64, limit: f64, w: &WordsInput) -> String {
    let n = w.samples;
    // "the model's own uncertainty was κ× too small" only when it says something (κ is floored at
    // 1 and ∞ when a comparison failed: review I3)
    let posterior = if w.calibration.is_finite() && w.calibration > 1.05 { format!("; the model's own uncertainty was {:.1}× too small there", w.calibration) } else { String::new() };
    let over = format!("{} DFT comparison{}", w.evidence, if w.evidence == 1 { "" } else { "s" });
    match key {
        // M8a user-fix 3: a key's floor is the columns its atoms touch, not the whole basis's Q′
        ClassKey::Samples => format!("{n} of {limit:.0} samples needed"),
        ClassKey::Force => {
            let energy = if w.energy_max > 1.0 { format!(", energy {:.1}× the tier", w.energy_max) } else { String::new() };
            format!("LOO force error {:.1}× the bound, RMS {:.2}×{energy}, {n} samples", w.force_max, w.force_rms)
        }
        ClassKey::Energy => format!("LOO energy error {:.1}× the tier, {n} samples", w.energy_max),
        ClassKey::EnergyRms => format!("LOO energy error RMS {value:.1}× the tier (training samples the model cannot reproduce included), {n} samples"),
        // the guard applies both ways below the evidence floor (trust tightening §3)
        ClassKey::WorstAtom if w.fits < FLOOR_FITS || w.floor_points < FLOOR_POINTS => format!(
            "a DFT check found an atom's force {value:.1}× its bound, limit {limit:.0}× while the class is below its evidence floor: {} / {}; {n} samples",
            floor_count(w.fits, FLOOR_FITS, "fits"),
            floor_count(w.floor_points, FLOOR_POINTS, "checks")
        ),
        ClassKey::WorstAtom => format!("a DFT check found an atom's force {value:.1}× its bound — the class does not enter while its recent evidence holds an atom beyond {limit:.0}×; {n} samples"),
        ClassKey::Calibrated => format!("DFT-measured force error {value:.2}× the bound over {over}, limit {limit:.2} (leave-one-out RMS {:.2}×){posterior}; {n} samples", w.force_rms),
        ClassKey::LooFloor => format!(
            "leave-one-out force RMS {value:.2}× the bound, limit {limit:.2} (DFT-measured {} over {over}); {n} samples",
            w.measured.map_or("—".to_string(), |m| format!("{m:.2}×"))
        ),
        ClassKey::MeasuredEnergy => format!("DFT-measured energy error {value:.2}× the tier over {over}, limit {limit:.2}; {n} samples"),
        ClassKey::Floor => format!(
            "within the bound over {} / {}, and its leave-one-out is {value:.1}× the bound; {n} samples",
            floor_count(w.fits, FLOOR_FITS, "fits"),
            floor_count(w.floor_points, FLOOR_POINTS, "checks")
        ),
        ClassKey::Window => format!("within its bound over {value:.0} DFT comparisons, but entering before the next refit needs {limit:.0}; {n} samples"),
    }
}

impl Calibration {
    /// The verdict for `class` (0 pair, 1 many-body) from its leave-one-out statistics, the fit's
    /// Q′ and the standing verdict `was_trusted` (the hysteresis; module doc, "The verdict").
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // the negation is the point: NaN fails
    pub fn class_verdict(&self, class: usize, st: &ClassStats, q_prime: usize, was_trusted: bool) -> ClassVerdict {
        verdict(self.per_class[class], st, q_prime, was_trusted)
    }

    /// The verdict for one environment key (module doc, "Per environment species"): its own
    /// calibration ([`Calibration::key`]) with its class's leave-one-out statistics `st`.
    pub fn key_verdict(&self, key: &EnvKey, st: &ClassStats, q_prime: usize, was_trusted: bool) -> ClassVerdict {
        verdict(self.key(key), st, q_prime, was_trusted)
    }
}

/// The verdict from one calibration (module doc, "The verdict, with hysteresis").
#[allow(clippy::neg_cmp_op_on_partial_ord)] // the negation is the point: NaN fails
pub fn verdict(cal: ClassCalibration, st: &ClassStats, q_prime: usize, was_trusted: bool) -> ClassVerdict {
    {
        let finite = |x: f64| (!x.is_nan()).then_some(x);
        let words = WordsInput {
            samples: st.samples,
            force_rms: st.force_rms,
            force_max: st.force_max,
            energy_max: st.energy_max,
            measured: if cal.points > 0 { finite(cal.measured) } else { None },
            measured_energy: if cal.points > 0 { finite(cal.measured_energy) } else { None },
            evidence: cal.evidence,
            calibration: cal.factor,
            fits: cal.fits,
            floor_points: cal.floor_points,
        };
        let untrusted = |key: ClassKey, value: f64, limit: f64| ClassVerdict::Untrusted { key, value, limit, reason: refusal_words(key, value, limit, &words) };
        if st.samples < q_prime {
            return untrusted(ClassKey::Samples, st.samples as f64, q_prime as f64);
        }
        // the worst-atom guard: on entry at every evidence count, both ways below the evidence
        // floor (trust tightening §3; before it, below three points)
        if (!was_trusted || !cal.floor_met()) && !(cal.worst_atom <= WORST_ATOM) {
            return untrusted(ClassKey::WorstAtom, cal.worst_atom, WORST_ATOM);
        }
        // the rule of before: LOO force rms and max ≤ 1, LOO energy max ≤ 1
        let before = super::trust::class_refusal(st.samples, q_prime, st.force_rms, st.force_max, st.energy_max);
        if !cal.calibrated() {
            // unchanged — no evidence loosens nothing
            return match before {
                None if energy_rms_fails(st) => untrusted(ClassKey::EnergyRms, st.energy_rms, 1.0),
                None => ClassVerdict::Trusted,
                Some((key, value, limit)) => untrusted(key, value, limit),
            };
        }
        // today's calibrated verdict; below the floor it must pass **and** the rule of before. The
        // refusal names what shuts the key: the calibrated verdict's own failing term when it
        // fails (the floor is not why), else the floor with the leave-one-out ratio that decided
        // (`ClassKey::Floor`, limit 1)
        match calibrated_refusal(&cal, st, was_trusted) {
            Some((key, value, limit)) => untrusted(key, value, limit),
            None => match before {
                Some((_, value, limit)) if !cal.floor_met() => untrusted(ClassKey::Floor, value, limit),
                _ => ClassVerdict::Trusted,
            },
        }
    }
}

/// Today's calibrated verdict (module doc, "The verdict, with hysteresis"): the first failing term
/// as (key, value, limit), `None` when it passes.
#[allow(clippy::neg_cmp_op_on_partial_ord)] // the negation is the point: NaN fails
fn calibrated_refusal(cal: &ClassCalibration, st: &ClassStats, was_trusted: bool) -> Option<(ClassKey, f64, f64)> {
    let limit = if was_trusted { LEAVE } else { ENTER };
    // the force: the measured error with the leave-one-out as the floor (NaN fails either way);
    // the key names the term that decided
    let (w, r) = (cal.measured, st.force_rms);
    if !(w <= limit) && !(w < r) {
        return Some((ClassKey::Calibrated, w, limit));
    }
    if !(r <= limit) {
        return Some((ClassKey::LooFloor, r, limit));
    }
    // the energy half of C1: at the bound itself (no ENTER band — the calibrated energy check
    // is the measured audit energy below), and only when M contributes
    if energy_rms_fails(st) {
        return Some((ClassKey::EnergyRms, st.energy_rms, 1.0));
    }
    if !(cal.measured_energy <= limit) {
        return Some((ClassKey::MeasuredEnergy, cal.measured_energy, limit));
    }
    None
}
