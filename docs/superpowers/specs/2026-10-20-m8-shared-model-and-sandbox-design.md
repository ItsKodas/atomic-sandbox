# M8 — the shared model and the Sandbox solver — design

**Date:** 2026-10-20 · **Status:** written for review · **Owner:** Dakoda (decisions quoted below are theirs) · **Supersedes** the M8 outline in `docs/superpowers/notes/2026-10-13-m5-user-requests.md` ("Locality and transfer", item D "per-atom model storage") where they differ.

## 1. Intent

Today a trained model belongs to one scene: it answers only for that scene's exact settings (grid, box, charge, species), and the DFT grid limits every scene to a few dozen atoms in a ~20 Bohr box. The user wants to "simulate much much larger scenes, possibly with hundreds of atoms or large polymer chains", interactively, on models that "support many different scenes with the same model, at increased complexities too". M8 delivers that in two milestones:

- **M8a — the shared model:** one model per set of solver settings that keeps learning from everything trained, transfers to scenes it never saw whole (it is local: an atom's energy depends on its neighbourhood within r_cut), shows what it knows, and offers "Train this segment".
- **M8b — the Sandbox solver:** a fourth solver beside CPU / GPU / Estimated that answers every step from the shared model only — no DFT, so no grid and no size limit — freezing the atoms it has not learned instead of guessing.

## 2. Decisions taken with the user (2026-10-20)

- "One big model that keeps learning from everything" — with **optional smaller specialised models** "trained on specific scenes for faster output".
- **Sandbox = model only, no fallback**, "only guessing / preventing the simulation from breaking down (but still warn of poor precision)".
- **Unlearned atoms are frozen and ignored**, with a warning that they need training, and a **"Train this segment"** button that "grabs the selected atoms requiring training, and creates a new scene with them and settings ready to go for training".
- **Sandbox keeps its own scene**, separate from the scene of the other three solvers ("make sure the sandbox area is kept").
- **Scale:** "hundreds being watchable, thousands expected to take some time".
- **A view of what the model knows** ("some flow chart or something") — with **mockups as part of M8** before it is built.
- **Bonds, molecule names, the electron cloud, orbitals and charges in Sandbox wait** for the learned-density milestone (they are read off a solved density, which Sandbox does not have) — "fine to wait".
- **Split into M8a then M8b**; "this seems fair".
- Working style: the user hand-tests visible work; Claude does not wait for them (memory `keep-working-note-tests`): build, launch, list what to test, continue unless blocked.

## 3. Goals, non-goals, anti-goals

**Goals:** the shared model (M8a §4.2–4.3), the calibrated trust verdict (§4.1), the knowledge map with mockups (§4.4), Train this segment (§4.5); the Sandbox solver with its own scene, unbounded space, freeze-and-warn, data-driven safety guards and a GPU model evaluator for hundreds–thousands of atoms (M8b §5).

**Non-goals:** charged scenes in the shared model (a local model cannot see the total charge — M9, long-range electrostatics); density / orbitals / bonds / names / charges in Sandbox (the learned-density milestone); DFT inside Sandbox; periodic cells (M10); new physics in the DFT solvers.

**Anti-goals** (failure even if it runs):
1. **Confidently wrong without saying so** — an atom moving where the model has no business answering. Freezing exists to prevent it; every answered atom carries its uncertainty.
2. **Physics silently mixed** — samples from different solver settings (grid spacing, pseudopotentials, tolerance, smearing) in one model.
3. **A hand-made shortcut** — any per-element or per-pair parameter not learned from this solver's own data (CLAUDE.md rule 1), e.g. a repulsion to stop crashes.
4. **Big scenes that run but are not watchable** at the stated scale.

## 4. M8a — the shared model

### 4.1 Calibrated trust verdict (first task; physics-critical, Fable + physics review)

The class-wise trust rule (providers::surrogate rule 4b, the max leave-one-out error per class) is a knife-edge: the verdict flips at r_cut and on rare classes (the user's screenshots: #251 at r_cut, #344 compressed H–H in H₃⁺). **Requirement:** a verdict calibrated against the model's audit history — predicted uncertainty vs. the errors the audits actually measured, per class — with hysteresis so it does not flip between neighbouring steps, and the same safety as today (the surrogate suite's stuck-then-escape repro still answered honestly; no rise in missed large errors). The first step **measures today's rule 4b as the baseline** on the surrogate suite's audit records (missed large errors: an audit error above the fit's own 99th-percentile in-domain error while the verdict said trusted) (my call). The method is the physics designer's to choose and justify in `validate/references.md`; acceptance: both repro cases answered stably, the surrogate suite green, the physics reviewer's sign-off.

### 4.2 The shared model ("collection")

- **One collection per surface key** = the solver settings a sample's energy depends on: the exchange-correlation functional (LSDA today; PBE from the PBE milestone, §7), grid spacing h, σ, SCF tolerance, the teacher's kind, the ionic surface tag (pseudopotential recipe hash). **Not** the box size (grid n): the box rule already marks box-affected samples, and transfer across box sizes is the point. Today's `Provenance` comparison (which includes `grid_n`, `charge`, `multiplicity`) becomes, for collections, this key; per-scene models keep today's rule.
- **Neutral scenes only (M8):** samples from charged scenes stay in per-scene models (M9 brings charge); existing charged `.model` files (e.g. `models/nh4+.model`) are not imported — they stay per-scene models (my call).
- **Box-affected samples** (the box rule's flag) are kept out of the shared collection and stay only in their per-scene model, so finite-box error never enters the shared fit or its calibration (my call; physics review). **Spin:** a sample keeps its own multiplicity as solved (the scene's auto = the lowest spin the electron count allows — paired, one unpaired electron for an odd count, `qm::scf::spin_split` — not a search for the lowest-energy state; corrected 2026-10-21, M8a Task 4a review: an auto O₂ is the singlet surface while the O reference is the triplet, a known gap in `validate/references.md`); whether samples of different spin states of the same atoms may share a collection is a physics ruling for the design review (default: yes when the scene's multiplicity is auto, no when the user forced one).
- **Contents:** the sample store (every kept teacher answer with its geometry, energy, forces, tags), the isolated-atom references per species, the fitted coefficients and the per-class calibration (§4.1). The species set **grows** when a sample with a new element arrives: the basis gains that channel and the model refits from the stored samples (the accumulated normal equations make this cheap for added samples; a new species is a full refit). Fitting stays interactive as the collection grows (the M8 notes' "S = 4 performance fixes": spectral Σ, warm evidence) — target: a refit after a teaching sample under 1 s at 10 000 samples.
- **Storage:** `models/collections/<surface key>/` (samples, references, fit, a human-readable `README.txt` of what it knows), shareable like profiles; the app autosaves into it as it learns. Existing `.model` sidecars **import** into the matching collection (their samples, if the surface key matches; charged ones refused with the reason) — never silently.
- **Specialised models:** "Save a specialised model…" exports a model fitted only on the samples relevant to a scene (its species and pair classes), smaller and faster; it is a normal model file, chosen in the Model tab or as Sandbox's model.
- **Estimated uses the collection** of its surface key by default (the Model tab says which, with a switch to a specialised model).

### 4.2b Model capacity (added after the design review, 2026-10-20)

Today's descriptors stop at three-body terms (pairs and triples of neighbours under a degree cap). One collection covering water, hydrocarbons and polymers may run out of capacity and degrade for everything at once. **Goal:** the model's capacity grows with the collection — four-body invariants (the notes' "S = 4" work: the spectral Σ and warm-evidence fixes that keep a larger fit interactive) and whatever basis growth the measurements call for — and the transfer test (§4.3) must pass **with all of its training in one collection**, not only in a small specialised model. The Bayesian (linear) form stays: its closed-form uncertainty is what the trust verdict and Sandbox's freezing rest on.

### 4.3 What transfers, and how it is checked

The model is local, so a collection trained on fragments answers larger molecules built from the same environments. **Validation (new `validate` test):** a collection taught only on CH₄ and C₂H₆ (and H₂O + (H₂O)₂ for water) reproduces fresh DFT points of propane (and a water trimer) with force and energy-per-atom errors at most 2× the collection's own in-domain errors (its leave-one-out RMSE) where it says it is trusted (my call; the default-grid DFT references of propane and a water trimer are seconds each on the GPU), and refuses (falls back / freezes) where it is not. **Reach:** today's cutoff r_cut = 6 Bohr (≈ 3.2 Å) puts a water molecule's hydrogen-bonded neighbour (≈ 5.3 Bohr) at the edge and the next shell out of sight; published potentials of this kind use 5–6 Å. The transfer test runs at r_cut = 6 and ≈ 9 Bohr and the collection's default is chosen from the result (accuracy vs. evaluation cost and the larger training boxes a longer reach needs), justified in `references.md`. Numbers recorded in `references.md`.

### 4.4 The knowledge map (mockups first)

A view of what the collection knows, designed as a **mockup page first** (`docs/superpowers/notes/2026-10-2x-m8-mockup.html`, like the UI redesign's), approved by the user before it is built:
- **Graph:** elements as nodes (size = samples involving them), pairs as edges coloured by confidence (calibrated, §4.1: green trusted / amber thin / grey never seen).
- **Pair detail:** the distances taught (a histogram from compressed to stretched) with a marker where the current scene asks; **element detail:** the bond angles and coordination counts seen.
- **3D overlay:** each atom tinted by how well its environment is known; in Sandbox the frozen atoms and *why* ("never seen N–N").
- Home: a right-dock tab next to Model plus a larger pop-out window; Ctrl-K entries.
The mockup page also covers the Sandbox solver panel, the frozen-atom warnings and the Train-this-segment flow (M8b).

### 4.5 Train this segment (from Estimated in M8a; from Sandbox in M8b)

Selected atoms (the Edit tool's selection) or the atoms the model refuses, **plus their neighbours within r_cut** (an interaction needs both sides), cut out into a **new scene**. **Capping** (signed off by the user 2026-10-20, "happy with that if it trains the same behaviour outcomes"): every bond the cut breaks gets a capping hydrogen, placed with the molecule builder's tables (`crates/app/src/builder/chem.rs`: covalent radii and valences — the builder's Rule-1 sign-off extended to placing caps; they place atoms only, never touch the physics), so the segment is a closed-shell molecule, not a radical; **only the atoms whose whole r_cut neighbourhood matches the original scene are trained on** (their forces; energies through the per-atom masking the plan designs with the physics reviewer) — the caps and the edge atoms' environments are not taught. **Condition (the user's):** a validate test shows that a collection taught on capped segments of a molecule (polyethylene, a water cluster) reproduces fresh DFT points of the *whole* molecule as well as a collection taught on the whole molecule's own samples (errors within 1.5× of each other); if not, capping is replaced by the uncapped two-shell cut (larger segments, only the centre trained). The segment scene is centred, a box fitted with the solver's margin, the default grid and ionic policy, opened in Estimated with a sampling plan ready (the existing teach-around / sampling pass). What it learns goes into the collection. The scene it came from is kept (M8a: saved and offered as "back to the scene"; M8b: the Sandbox slot).

## 5. M8b — the Sandbox solver

- **A fourth solver, "Sandbox"**, in the toolbar's solver group. Its **own scene slot**: switching between Sandbox and the other solvers swaps scenes; both persist with the session and save/open like any scene.
- **Its model:** picked in the Sandbox panel from the collections and specialised models on disk (the scene file records which); atoms of species the model does not cover are frozen like any unlearned atom (my call). **Sandbox scenes are neutral:** a charged scene waits for M9 (refused with the reason).
- **No grid:** the space is as large as wanted. Walls stay available with a freely set half-width (default on, sized to the scene with room); off = unbounded (the escape guard of rule 0 is replaced by the walls or a "far away" warning).
- **Model only:** every step from the chosen collection or specialised model. No DFT, no fallback, no audits in Sandbox (learning happens through Train this segment in the DFT solvers).
- **Freeze-and-warn:** an atom whose environment the model does not know (the verdict of §4.1 per atom: untrusted class, outside the envelope, novelty / extrapolation grade) is **frozen** — held still, velocity zero — and marked; a banner: "N atoms frozen: the model has not learned their surroundings — Train this segment". Atoms answered but uncertain move, tinted by their uncertainty, with a precision warning (as draft mode today). **Cascade guard** (added 2026-10-20): a frozen atom is an immovable obstacle its neighbours can be pushed into unusual geometries against, which may freeze them in turn; when the frozen count keeps growing (the plan sets the rule, e.g. rising over a window of steps or beyond a share of the scene), the run pauses and says so instead of slowly turning to stone. The Sandbox energy is the sum over **unfrozen atoms' own terms only** (E = Σ_{i not frozen} E_i): a frozen atom's term is dropped, so no force from an unlearned environment acts on a moving atom and the forces stay the gradient of one energy; the jump when an atom freezes or unfreezes is booked as work (like a pin), so the conserved-energy readout stays continuous (my call; physics review).
- **Safety guards from data only (rule 1):** no pair of atoms closer than the shortest distance the collection was ever taught for that pair (a constraint like the walls, from the samples, not a table); a force cap and a speed guard that pause with a reason instead of exploding. No hand-made repulsion.
- **GPU model evaluator:** descriptors, energy and forces on the GPU with a cell list (neighbours within r_cut), matching the CPU model within a tested tolerance; the CPU path stays the reference. **Targets** ("watchable" = the toolbar's watchable rate for the scene, with the view at ≥ 30 frames/s — my call): a ~200-atom scene (a polyethylene chain, a box of ~60 waters) at the toolbar's "watchable" rate on the RX 7900 XTX; 1000+ atoms run, slower, with the rate shown.
- **What Sandbox shows:** atoms, trails, motion blur, the Camera view, all tools (Inspect, Place, Grab, Edit), the readouts (energy, temperature, conserved energy), the uncertainty tint and the knowledge overlay. **Not** the density, orbitals, bonds, names or charges (§2; the learned-density milestone).
- **Train this segment from Sandbox:** as §4.5; the Sandbox scene stays in its slot; when the collection has learned the segment, the frozen atoms unfreeze.

## 6. Testing and gates (CLAUDE.md rule 6; user preference: Claude does not wait on hand-testing)

- **Physics gates:** the surrogate suite (`validate --test surrogate`) and `providers` tests on every M8a task that touches trust, collections or fitting; the new transfer test (§4.3); a Sandbox test: the GPU evaluator vs the CPU model (energies, forces), NVE energy drift of a small trusted scene in Sandbox, freeze/unfreeze bookkeeping. The dynamics suite if `sim` changes. Physics reviewer on §4.1, §4.2's sharing rules and §5's freeze rules.
- **The user hand-tests:** the mockup page (approval gate for §4.4's UI), the knowledge map, Train this segment, Sandbox's feel and speed. Claude builds and launches after each visible piece, lists what to test, and continues.
- **One heavy GPU job at a time** on this machine.

## 7. Order and timing

**PBE becomes the default functional between M8a and M8b** (decided with the user 2026-10-20; LSDA kept as an option): every collection is tied to the functional that taught it (§4.2's key), so the serious training M8b invites should land on PBE from the start. The PBE milestone has its own spec (functional on the grid, CPU f64 reference then WGSL, low-density thresholds, pseudopotentials regenerated and re-gated for PBE, oracles against published PBE numbers, models keyed by functional); ≈ 2–4 days, dominated by regenerating and re-validating the pseudopotentials. M8a's development training is small and is redone on PBE.

M8a: design review → mockup page (user approval) → §4.1 → §4.2 → §4.3 → §4.4 → §4.5 → merge. PBE milestone (its own spec and plan). M8b: plan → GPU evaluator → Sandbox solver + slot + walls → freeze/guards → overlay + Train this segment → merge. Implementation starts after the five sessions in flight on 2026-10-20 (training fixes, capture, rewind, orbital polish, M7 leftovers) merge — §4.2 touches the same training code. Estimated: M8a ≈ 2–3 days, M8b ≈ 2–3 days of wall clock.

## 8. Left to the builder

The calibration method (§4.1, with the physics designer and reviewer), the collection file format and its versioning, the GPU evaluator's kernel layout, the knowledge map's exact visuals (from the approved mockup), the walls' default size in Sandbox, and the exact thresholds of the guards — each justified in `references.md` or the plan where it is physics.
