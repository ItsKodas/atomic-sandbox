# Live orbitals in estimated mode — design

Status: approved by the user 2026-11-14; builder-check rulings folded in (§9). Builder: read `CLAUDE.md` first. The rules there apply: no fitted per-element data, atomic units, f32 on the GPU, the `ForceProvider` boundary, validation gates and physics honesty.

## 1. Intent

The user wants the **All orbitals coloured** view, and the other orbital views, to stay live in estimated mode. They want to see atoms share and dent each other's orbitals as they move. In their words: "seeing the atoms interact like that is incredible".

They also asked for it to learn "in a similar way to the learned bonds", that is, from the app's own past DFT solves.

**What estimated mode shows today:**
- Its orbitals are the background auditor's last DFT solve (`SideDisplay.orbitals` from Audit/Refresh jobs).
- Between solves they are faded and warped along with the atoms ("fields follow the atoms").
- With audits off, which is the usual case under Model only, they do not refresh at all.
- The density view's "atomic superposition" is isolated atoms placed on top of each other. It knows nothing about how the atoms interact.

**What this adds:** orbitals solved at the **current** geometry, in a potential built from a guessed density. The density guess is:
- **Phase A:** the superposed isolated-atom densities (Harris-style, non-self-consistent).
- **Phase B:** that superposition plus each atom's deformation remembered from past DFT solves, in the spirit of `bond_experience`.

Because the orbitals are actually solved, they mix across atoms and stay orthogonal, which is what produces the dents. Nothing is stretched from an old frame.

## 2. Goals, non-goals, anti-goals

**Goals**
- G1. Under the estimated backend, the Orbitals, All orbitals coloured and Spin density views show orbitals solved at the current nuclei. They refresh while playing and after a geometry change while paused, with no SCF loop.
- G2. Phase B: the density guess improves with experience. Each DFT solve the app already runs contributes per-atom deformation densities. Atoms whose surroundings match remembered ones get that deformation added back.
- G3. Honest labelling everywhere these orbitals appear: not self-consistent, eigenvalues approximate, which atoms used experience. The labels must never read like a DFT result.
- G4. Display only. Nothing here reaches forces, energies, the integrator, the surrogate's fit, trust or audits.

**Non-goals**
- Learning orbitals directly. They are not unique: sign, the freedom to rotate occupied orbitals among themselves, level crossings. This was agreed with the user.
- Bonds or partial charges from the Harris density. That would be a later, separately approved follow-on (§9 Q2).
- Any change to SCF results, forces, the surrogate or its trust.
- The CPU teacher. With `--backend cpu` estimated mode keeps today's `ORBITALS_CPU_TEACHER` note.

**Anti-goals** (failures even if it works)
- Under Model only, it starts a solve that shows up as a "DFT fallback", or counts as a fallback in the counters.
- It noticeably slows the run. The orbital display must not make a playing estimated run visibly slower.
- It makes Harris orbitals look like converged DFT. That includes showing their ε or gap as if exact, or hiding which atoms used experience.
- Experience that silently produces garbage. A remembered deformation applied to a mismatched environment must be refused, never applied.

## 3. Constraints

- **Rule 1.** The basis sizes, cutoffs, match tolerances and "how many examples" thresholds are display choices, and identical for every element. Learned values come only from this app's own DFT solves. Nothing is typed in per element.
- **Rule 3.** Everything is in atomic units internally.
- **Rule 4.** Grid fields are f32 on the GPU. Experience coefficients are stored as f32; projections and fits are done in f64 on the CPU. Integrated charges use compensated sums.
- **Rule 5.** `sim` does not learn what computes the display. The Harris solve lives in `qm-gpu` (the solver method) and `providers` (the job). Its payload is the existing `OrbitalDisplay` with a new "kind".
- **Rule 7.** Each place that shows these orbitals states the approximation (§6).
- **One GPU, one heavy job at a time.** The Harris solve shares the `GpuContext` queue with the render thread, the stepping teacher and the auditor.
- **Hardware.** RX 7900 XTX, 24 GB, Vulkan, with no f64 in WGSL.

## 4. Phase A — Harris display orbitals

### 4.1 Physics (decided)

The input density is ρ_in = Σ_A ρ_A(|r − R_A|). Each ρ_A is the isolated-atom radial density of species Z_A on the scene's surface (h, σ, ionic policy), taken from the cloud's `AtomProfile`s (`app/src/cloud.rs`).
- Those come from `Atom` jobs: one solve per species on the auditor, which the superposition cloud already requests. The reference multiplicity scans (`AtomReference.radial`) are not used here, because each is several cold DFT solves, and under Model only `queue_missing_references` may never run them.
- So under Model only the one DFT cost of this feature is one isolated-atom solve per species, once per surface. It is the same solve the superposition cloud already makes, and Activity names it as such.
- The profiles must be the valence density without ρ_pc. Today the auditor applies `set_display_partial_core(job.core)` to every job, Atom jobs included. The Harris path requests or filters profiles without the partial core, and a test pins this.
- Each atom's profile is renormalised to its own electron count (Z, or Z_v under a pseudopotential) before summing.
- The sum is then rescaled to the scene's electron count N = Σ Z_v − charge, so that ∫ρ_in = N. The rescale factor is shown in the debug readout. A charged scene's factor is ≠ 1, and the label says so.
- **Partial core.** ρ_pc is never part of ρ_in's Hartree term. The xc term includes the core exactly as the SCF does (NLCC), so ρ_in matches what the SCF's own ρ would be.

The potential is V_σ = v_ion + V_H[ρ_in] + v_xc,σ[ρ_in↑, ρ_in↓].
- v_ion is the scene's nuclear or local-pseudo potential, plus the nonlocal KB projectors in H.
- V_H comes from the existing GPU Poisson solve.
- v_xc is LSDA. Its spin split is the input's: ρ_in↑ = ρ_in·n↑/N and ρ_in↓ = ρ_in·n↓/N, with n↑ and n↓ from the scene's multiplicity rule (`spin_split`, auto = N mod 2).
- This is a decided simplification. Superposed atoms carry no well-defined molecular spin, and it is documented.

The solve is one LOBPCG for the lowest n_occ,σ + m states per spin channel. m is the existing "show unoccupied" count, at most `UNOCCUPIED_MAX`. It is warm-started from the previous refresh's vectors; cold, it falls back to `initial_orbitals`.
- **Occupations** use the provider's current rule, applied once to the Harris eigenvalues: Aufbau for all-electron, `DegenerateShells` window semantics under pseudopotentials, and Fermi–Dirac at the provider's kT once the smearing work lands.
- That needs a public one-shot entry point next to `qm::scf::degenerate_shell_occupations` (and the FD one). The SCF's private freeze/witness machinery does not apply to a single diagonalisation.

The displayed products use the **output** orbitals with those occupations. The Spin density view under this mode shows the Harris output's ρ_out↑ − ρ_out↓ (ρ_out = Σ f|ψ|²), never ρ_in's proportional split. The **Density** view is not part of this design: it keeps today's behaviour (the last DFT density, or the atomic superposition), so the default view never starts Harris solves.

### 4.2 Where it runs (decided)

- **Thread.** It runs on the background auditor's teacher thread, as a new side job `JobKind::HarrisDisplay`. That thread already owns a GPU teacher, the side-job queue and `SideDisplay` delivery. The stepping teacher stays untouched, so a playing run never waits for a display solve.
- **No warm-start contamination.** It never touches the auditor solver's SCF warm state (`warm`). A new `GpuSolver` method borrows the solver's Poisson, kernels and LOBPCG work, and writes into a separate `HarrisState`. It keeps its own orbital vectors, ρ_in, V_H[ρ_in], evals and occupations. `OrbitalView` reads that state through the same products path, so it no longer requires `solver.state()`.
- **The occupied-agreement guard** (`occupied_disagreement`) must hold for a Harris state: the unoccupied step rebuilds v from the same ρ_in and V_H[ρ_in]. Simplest is to solve occupied and unoccupied states in the one LOBPCG and skip the separate unoccupied step. That choice is left to the builder.
- **Scheduling.**
  - One Harris job is queued at most. A newer geometry replaces the queued one, so the latest positions win.
  - Audits that are due go first, because they are a safety check.
  - Atom and Reference jobs go first too, since Harris needs the atom references.
  - While playing, the rate follows the existing `Cadence`/budget pattern. The target is at most 15 % of wall time on Harris, with a minimum interval of 1 s and a View setting "refresh orbitals every N s" (default 2 s). Paused, one solve runs after each geometry change, debounced 300 ms after a drag's last move.
- **Its own ionic cache.** A Harris solve is at a geometry other than the auditor's converged state. Under pseudopotentials, `HarrisState` therefore owns its own `GpuIonic` (V_loc, the KB projectors) and v_nuc field, rebuilt per Harris geometry. It never rebuilds the solver's shared per-geometry cache, which `display_work` relies on for the auditor state's own display.
- **VRAM.** It counts against the auditor's half: the vectors, ρ_in, V_H, the ionic cache and v_nuc. The `orbital_view_footprint_bytes` guard is extended to cover them. If over budget, it shows a note and no Harris job runs.
- **Provider entry point.** The auditor's teacher is a `Box<dyn ForceProvider>`, and nothing may downcast it. So `sim`'s `ForceProvider` gets one new default-`None` method for a Harris display solve, which `GpuKs` implements. That is the only `sim` API change.
- **Existing DFT refresh paths.**
  - Outside Model only, nothing changes: the view-change `Refresh` in `worker/commands.rs` `orbital_view`, and "refresh cloud every N s" in `cloud_jobs` (default off), still run real DFT side solves. Harris fills the time between them.
  - Under Model only, the view-change `Refresh` becomes a Harris job, so Model only starts no DFT solve for a view change. "refresh cloud every N s" is already off there.
  - The new "refresh orbitals every N s" sets the Harris cadence only. It sits beside the cloud setting and does not replace it.
- **When it runs.** Only when the backend is Estimated with a GPU teacher, an orbital view is shown, and the atom references for every species present are in. Opening an orbital view is the user's opt-in, including under Model only (§9 Q1).
- **Stopping.** It stops when the view closes, the backend changes or a sampling pass starts. A pass owns the auditor, so Harris is paused for its duration, with a note.
- **Precedence.** A real DFT side payload (Audit/Refresh) at the current geometry, meaning the same positions to 1e-6 Bohr, replaces a Harris payload, because it is self-consistent. The newer geometry always wins.

### 4.3 Delivery

- `OrbitalDisplay` gains a `kind`: `Converged` (today's) or `Harris { experience: Option<ExperienceUse> }`. The kind is carried through `SideDisplay`, `poll_side_display` and the snapshot.
- A Harris payload at the current geometry is **not faded** (it is current). It carries its own label (§6).
- It does not count as a fallback, an audit or a teacher solve in any counter. Activity shows it as its own job: "orbital display solve (not self-consistent) · 0.3 s". It is never labelled "DFT fallback".
- The probe gets a third display slot, or the existing `display_solve`. The builder chooses, but the Activity row must name it correctly.

## 5. Phase B — density experience

### 5.1 What is remembered (decided)

After every converged DFT solve the app already runs, the deformation density Δρ = ρ_SCF − ρ_in is computed (with ρ_in as in §4.1). The solves include fallbacks, audits, sampling-pass samples, refreshes and Bonds now.

Δρ is split between atoms by Hirshfeld weights, w_A = ρ_A / Σ_B ρ_B. Each atom's share is projected onto an atom-centred basis in a frame tied to that atom's neighbours. **Baseline:** n_max = 6 radial functions (Gaussians on a fixed geometric ladder, the same for every element, inside r_c = 4 Bohr) × real spherical harmonics up to l_max = 2, giving 54 coefficients. The builder may change n_max, l_max or the ladder, recording the reason and the measured effect (§7 V-B2), provided the values stay the same for every element.

Each record holds:
- Z;
- the neighbours within 6 Bohr (nearest 6 at most, as in bond_experience): Z and their positions in the atom's frame;
- the coefficients;
- the source solve's surface tag.

Records are stored in the shared collection's folder as `density_experience.bin`, keeping the newest 50 000 atoms, or per session when there is no shared collection. Two running copies may append; each record is self-delimiting, and a partial tail record is dropped on read.

Only solves on the collection's own surface tag are stored. A tag change, for example after the smearing migration, means a new file.

### 5.1a Which solves record, and where (decided)

- **Recording:** only solves on the estimated backend's surfaces that have a collection. That covers fallbacks, audits, refreshes, Bonds now and sampling-pass samples. The plain GPU/CPU backends do not record, because every MD step there is a solve and they have no collection.
- **Off the step path:** the readback and projection never run on the stepping thread. A fallback's converged density is handed to the auditor thread, or to a worker-side helper thread (the builder chooses), so a playing run never waits for a projection.
- **Rate limit:** at most one record per atom per 0.05 Bohr of that atom's environment change (RMSD of its nearest 6 neighbours), so a sampling pass's hundreds of solves do not fill the store with near-duplicates.
- **Keys:** records are keyed by surface tag, scene charge and multiplicity. They are matched only within the same key, as `bond_experience`'s in-memory store is.
- **Ownership:** the store lives in `providers`, next to the auditor, since that is where Harris needs it. `app` only displays its counts.

### 5.1b Pre-registered spike before Phase B's plan (decided, ruling 2026-11-14)

The builder check showed that a near-duplicate lookup cannot work as first written. Sampling perturbations (σ = 0.15 Bohr per coordinate) and thermal MD move neighbours out to 6 Bohr by 0.4–0.5 Bohr RMSD between any two frames, so a 0.1 Bohr match would almost never find 3 candidates.

Phase B therefore uses **kernel-weighted regression** over stored atoms in aligned frames, not exact lookup, as follows:
- **Candidates:** the same Z, the same species among the nearest 6 neighbours within 6 Bohr. Filtering and alignment both use exactly those nearest 6.
- **Alignment:** RMSD r from Kabsch plus the assignment search.
- **Weight:** w = exp(−(r/ℓ)²).
- **Applied when:** Σw ≥ W_min and the weighted spread passes the §5.2 check.

Before Phase B's plan is written, a throwaway spike measures ℓ, W_min and the cutoff on one H₂O sampling pass, with V-B2's held-out split. **Pre-registered stop:** if no (ℓ, W_min) with ℓ ≤ 0.5 Bohr reduces the held-out L1 density error by at least 30 % against plain superposition, Phase B goes back to the user. The candidates then would be an equivariant regression, or dropping Phase B. §5.2 below states the rule with the spike's values to be filled in. Its 0.1 Bohr / 3-candidate numbers are superseded.

### 5.2 How it is used (decided, with the spike's ℓ and W_min)

For each atom at display time:
1. **Candidates:** stored atoms with the same Z and the same multiset of neighbour species within 6 Bohr.
2. **Alignment:** for each candidate, find the rigid rotation and the same-species neighbour assignment that best overlay its neighbours onto this atom's (Kabsch with an assignment search, nearest 6 neighbours at most). Accept the candidate if the RMSD is ≤ 0.1 Bohr.
3. **Threshold:** if at least 3 candidates are accepted, rotate each one's coefficients into this atom's frame (real Wigner matrices for l ≤ 2) and average them, weighted by exp(−(RMSD/0.05)²). The deformation is added to ρ_in. Otherwise this atom gets none.
4. **Spread check:** if the accepted candidates' coefficients disagree, with relative spread above 0.3 in the l ≤ 2 norm, the deformation is not applied, as with bond_experience's "all agree" rule.

A lone atom and a linear environment need care:
- A lone atom has a spherical deformation only (l = 0). Neighbourless matching is trivial.
- Linear environments leave the rotation about the axis free. Only the axially symmetric part (m = 0) of the averaged coefficients is kept.

After adding the deformations, ρ_in is renormalised to N again. The atoms that used experience, and their count, go into `ExperienceUse`.

The thresholds (0.1 Bohr, 3 examples, 0.3 spread, 6 Bohr, 50 000 records) are display choices, the same for every element, stated in the README as such.

### 5.3 What it buys

The Harris functional's error is second order in ρ_in − ρ_SCF, so a better ρ_in improves the orbitals and their ε quadratically. Phase B is the same Harris solve with a better input. No other display path changes.

## 6. Honesty labels (rule 7)

Every place listed below gets wording for the Harris kind:
- `ui/dock/orbitals.rs` notes;
- `view.rs` "show" info;
- `viewer.rs` `OrbitalShown::label`;
- `jobs.rs`;
- README "Orbitals and spin".

The wording is the builder's, but it must say:
- "orbitals solved once at t = … in the potential of [superposed atoms | superposed atoms + experience on k of n atoms], not self-consistent";
- ε and gaps are approximate (they are shown with "≈" and a tooltip);
- the shapes and mixing are qualitative;
- the density and spin views show this solve's output, not a DFT density;
- unoccupied states are mostly box states, as today.

A charged scene also says the atoms were rescaled to its electron count.

## 7. Validation

Display-only work, but rule 7 needs measured numbers. These are recorded in `crates/validate/references.md` under a new "Live orbitals" section and run in a new `cargo test -p validate --release --test harris` (GPU).

- **V-A1 (identity, gate).** Feed the **converged** SCF density as ρ_in. The Harris eigenvalues must equal the SCF's to ≤ 1e-5 Ha, and the occupied-subspace overlap must be ≥ 1 − 1e-6. Cases: H₂, H₂O, and a pseudo-O atom. This proves the potential assembly matches the SCF's.
- **V-A2 (record + loose gate).** Superposed atoms at equilibrium for H₂, H₂O and NH₃, and the user's scene: neutral H₂ + H, an open-shell doublet with the third H 3 Bohr from H₂'s centre on its axis, built inline. H₃⁺ (the near-equilateral triangle with sides 2.1 Bohr, charge +1) is recorded only, not gated, as the charged-rescale case.
  - Recorded: the occupied-subspace overlap with SCF, the ε error per level, and ∫|ρ_out − ρ_SCF|/N.
  - Gate: overlap ≥ 0.95 for each case. If one fails, investigate; do not loosen it.
- **V-A3 (cost, record).** A warm Harris refresh at 81³ for H₂ and for the neutral H₂ + H, and at 97³ for H₂O, timed. **Target:** ≤ 0.3 s for H₂ and H₂ + H at 81³. If it isn't met, record it, and the cadence keeps the 15 % budget.
- **V-B1 (gate).** Experience round trip, testing the projection, not the thresholds. Store a solve's deformation, then rebuild ρ_in at the same geometry with the count and weight thresholds bypassed by a test-only switch. ∫|ρ_in − ρ_SCF|/N must fall by ≥ 80 % against plain superposition (the basis's truncation error is what is left).
- **V-B2 (record + gate).** Held-out. Train experience on one sampling pass of H₂O, then evaluate at held-out perturbed geometries of the same pass. Recorded: the L1 density error of superposition vs experience. Gate: experience is lower on average, and the occupied overlap improves.
- **V-B3 (gate).** A mismatched environment (a different neighbour species, or RMSD > 0.1) applies no deformation, and the spread check refuses disagreeing candidates.

Other gates that apply:
- `qm-gpu` gets a new public method, but `run()` is unchanged: `--test gpu_vs_cpu`, plus the `qm-gpu` and `providers` unit tests.
- `--test surrogate`, because the auditor's job queue changes.
- `--test orbitals`.
- `cargo test -p app --release`.

## 8. Phasing

- **Phase A** ships on its own: Harris from plain superposition, with labels, cadence and V-A1–3.
- **Phase B** adds experience: storage, matching, V-B1–3.

Each phase is merged after its gates pass. The README is updated at each merge.

## 9. Decisions and open questions

- **Approved** by the user on 2026-11-14 ("i approve the live orbitals"), with the builder-check rulings below folded in.
- **Q1** (my call, check it): opening the Orbitals, All orbitals coloured or Spin density view in estimated mode is the opt-in to Harris display solves, including under Model only. Model only's promise becomes "no DFT for forces or motion; display solves only for a view you opened, plus the one isolated-atom solve per species that the cloud already makes". The alternative is a separate checkbox.
- **Q2** (yours, later): should a follow-on read **bonds and partial charges** from the Harris/experience density too? It would replace bond_experience's lookup with a density-based guess. It is not in this design either way.
- **Builder-check rulings (2026-11-14):**
  - The atom densities come from the cloud's `AtomProfile`s, without ρ_pc (§4.1).
  - The Density view is out of scope (§4.1).
  - Harris has its own `GpuIonic` and v_nuc (§4.2).
  - It enters through a new default-`None` `ForceProvider` method (§4.2).
  - Model only's view-change refresh becomes Harris (§4.2).
  - Phase B is kernel regression, behind a pre-registered spike (§5.1b), with recording off the step path, a rate limit and keys by charge and multiplicity (§5.1a).
  - The V-A2 user's scene is neutral H₂ + H, with H₃⁺ recorded only (§7).
  - V-B1 bypasses the thresholds (§7).

## 10. Left to the builder

- The exact API of the new `GpuSolver` Harris method and of `HarrisState`.
- Whether unoccupied states join the one LOBPCG.
- The probe slot.
- The record format of `density_experience.bin`.
- The assignment search's algorithm, within "nearest 6 neighbours".
- The label wording, within §6.
- The basis ladder's radii, within §5.1's rule.
