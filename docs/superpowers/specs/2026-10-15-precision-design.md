# Precision: near-exact electrons offline — precision renders, precision models, the model manager — Design Spec

**Date:** 2026-10-15
**Status:** Written from the user's request and the controller's binding commitments (§1). Decisions this spec had to make on its own are marked **[provisional]** and listed in §13. Physics-reviewed 2026-10-15 ("ready with fixes"; every MUST, SHOULD and NOTE folded in — §18; review text `docs/superpowers/notes/2026-10-15-precision-spec-review.md`): stages 1–3 may start; stage 4 starts with the QMC pseudopotential variant (plan Task 4.7).
**Author:** Claude (Opus 5.5) with Dakoda
**Parents:** `docs/superpowers/specs/2026-10-12-atomic-sandbox-design.md` (§2 "accurate means: as accurate as the best method that runs interactively", §5.1 `ForceProvider`, §11 roadmap); `docs/superpowers/specs/2026-10-13-m5-surrogate-provider-design.md` (the linear ACE model, its evidence-chosen two-noise posterior, trust/fallback/audit, one-surface bookkeeping); `docs/superpowers/specs/2026-10-14-m6-tools-design.md` §9 (the `providers` crate, `Trainable`); `docs/superpowers/specs/2026-10-15-user-data-design.md` (the per-user data root, the model library, the canonical key with `TEACHER_REV`, the Model tab — extended here, never duplicated); `docs/superpowers/specs/2026-10-14-pseudopotential-design.md` (M7: the in-house TM/KB pseudopotentials the QMC teacher needs for Z ≥ 3); `docs/superpowers/specs/2026-10-15-orbital-viewer-design.md` §4.1 (where the occupied orbitals live — the trial wavefunction reads them).
**Inputs:** the user's request of 2026-10-15 (recorded in `docs/superpowers/notes/2026-10-13-m5-user-requests.md` § "Precision milestone"); the UI mockup's style and the redesign's no-floating-windows rule (`docs/superpowers/notes/2026-10-14-ui-mockup.html`); `crates/validate/references.md` (every number of this solver quoted below).
**Plan:** `docs/superpowers/plans/2026-10-15-precision.md` (five stages, each shipping something usable).

---

## 1. What the user asked, and what is binding

After a discussion of why LSDA's bond energies are ~30 kcal/mol off and that the estimator could be trained on a better teacher, the user (2026-10-15):

> "perhaps we go ahead and add that to our plan? a different model to run that is highly precise using the real equations for almost perfect precision, but cannot be played live, you can either prep a scene then have it solve each step that you could then play back as a video whilst it solves each step. Or train a highly precise model. Again we need to have a super easy to use UI to configure, manage and understand each model on the system."

The controller's commitments (binding on this design):

1. **Precision render.** A background solve of every MD step of a prepared scene; progressive playback (scrub, pause, resume) while it computes; checkpointed so it survives an app restart; usable with **any** slow solver — including today's DFT at the **Precise** quality level — so it ships before the precise method exists.
2. **Precision models.** The estimator trained on the precise method, mainly by **Δ-learning** (the model learns E_precise − E_LSDA, a smooth function that needs far fewer expensive samples), then playable live.
3. **A model manager.** Every model on the system in one place: teacher and accuracy tier, training data, coverage, cost, side-by-side comparison, a chooser for which model answers the scene, long jobs as a queue with progress and ETA; plain-language explanations for every number.
4. **The precise method** is "most likely real-space quantum Monte Carlo from the sandbox's own orbitals (Z-only, rule 1), with essentially exact methods for 1–3-electron systems". This spec decides, honestly, including fixed-node error, noisy forces and where pseudopotentials are required (§3–§5).

## 2. The gap being closed, in numbers

- **LSDA's energetics.** For the 148-molecule G2 set, LSD atomization energies have a mean absolute error of **83.8 kcal/mol** (PBE 17.1) — Tao, Perdew, Staroverov & Scuseria, *Phys. Rev. Lett.* 91, 146401 (2003), Table 1. LSDA *overbinds*; the ~30 kcal/mol the user quoted is the small-molecule end of that distribution. The functional is the one fitted ingredient of this project (design §6) and no grid refinement removes its error.
- **This solver on top of that** (`references.md`): the smeared nucleus (σ = h) moves H₂'s bond length to 1.002 / 0.947 / 0.847 / 0.807 Å at Fast / Balanced / Accurate / Precise against 0.741 Å (experiment); the σ → 0 LSDA limit is 1.451 Bohr (0.768 Å). M7 removes the smearing for cores; it does not touch the functional.
- **The target.** The non-relativistic, clamped-nucleus (Born–Oppenheimer) Schrödinger equation of the electrons — "the real equations". Its exact answers for the systems the oracles use: H₂ at R = 1.4011 Bohr, **E = −1.1744759314002167 Ha** (Pachucki, *Phys. Rev. A* 82, 032509 (2010)); hence the BO well depth **D_e = 0.17447593 Ha = 38 293.0 cm⁻¹ = 109.49 kcal/mol = 4.7477 eV** (derived from that number and E(H) = −0.5). H₃⁺ (equilateral, R = 1.65 Bohr) **−1.34383562502 Ha** (Pavanello & Adamowicz 2009, explicitly correlated Gaussians, as tabulated by Lopez Vieyra, Turbiner & Medel Cobaxin, arXiv:1105.5435; the earlier ECG record −1.343835624 is Cencek, Komasa & Rychlewski, *Chem. Phys. Lett.* 246, 417 (1995)). HeH⁺ at R = 1.463283 Bohr **−2.978708310771 Ha** (Pachucki, *Phys. Rev. A* 85, 042511 (2012)). He atom **−2.903724377034 Ha** (Schwartz, arXiv:math-ph/0605018: −2.90372437703411959831…; the Hylleraas-CI value of Sims & Hagstrom agrees). H₂O: exact non-relativistic total energy **−76.4389 Ha** (Klopper, *Mol. Phys.* 99, 481 (2001), as used by Caffarel, Applencourt, Giner & Scemama, arXiv:1604.02183); non-relativistic clamped-nucleus atomization energy **TAE_e ≈ 233.0 kcal/mol (0.3713 Ha)**, derived from the W4 breakdown of Karton, Rabinovich, Martin & Ruscic, *J. Chem. Phys.* 125, 144108 (2006), Table 5 (TAE_e 232.65 including scalar-relativistic −0.27, spin–orbit −0.22 and DBOC +0.13 kcal/mol; removing those three gives 233.01; the sum of the non-relativistic components gives 233.02). Of that, **0.38 kcal/mol is core–valence correlation**, which any frozen-core (pseudopotential) method misses by construction. Experiment (ATcT, same table): TAE₀ 219.36 kcal/mol with ZPVE 13.29.

What "almost perfect precision" can honestly mean is fixed by those numbers: an error budget measured in **kcal/mol** (1 kcal/mol = 1.594 mHa), against LSDA's tens.

## 3. Method choice

### 3.1 The candidates

| method | what it solves | error vs exact BO | cost scaling | rule 1 | verdict |
|---|---|---|---|---|---|
| **Full CI on the real-space grid** | the N-electron Schrödinger equation on the product grid | exact to the grid's discretisation (cusps are badly resolved by a uniform grid: the e–e and e–n cusps are the same "stencil turns a cusp into noise" problem as `nuclear.rs`'s, now in 3N dimensions) | (n³)^N / N! amplitudes: 2 e⁻ on 33³ → 6.5 × 10⁸ (2.6 GB f32 — fits the card at a cusp-ruining h = 0.375); 2 e⁻ on 49³ → 7 × 10⁹; 3 e⁻ impossible | trivially holds | **rejected**: only ≤ 2 electrons, only on grids too coarse for the cusps the answer depends on |
| **Explicitly correlated wavefunctions** (exponential/Kołos–Wolniewicz-type for H₂; explicitly correlated Gaussians, ECG, for few-electron systems — Mitroy et al., *Rev. Mod. Phys.* 85, 693 (2013)) | variational, basis functions that depend on r_ij | H₂ to ~10⁻¹⁵ Ha (Pachucki 2010); HeH⁺ ~10⁻¹² (Pachucki 2012); H₃⁺ with 1000 ECGs to the digits of §2 | grows steeply with N (N! permutations, K² matrix elements, nonlinear optimisation); practical for N ≤ 3–4 | holds **if** the nonlinear parameters are found variationally from Z alone (§5.1) — no tabulated basis | **adopted for N ≤ 3** (stage 3): deterministic, smooth, forces available, μHa-class with a few hundred functions [est, measured in Task 3.1] |
| **Variational + diffusion Monte Carlo** (Slater–Jastrow; Reynolds, Ceperley, Alder & Lester, *J. Chem. Phys.* 77, 5593 (1982); review Foulkes, Mitas, Needs & Rajagopal, *Rev. Mod. Phys.* 73, 33 (2001)) | projects the ground state in imaginary time with the nodes of a trial function fixed | **exact for nodeless ground states** (1 e⁻; 2 e⁻ singlets: H₂, H₃⁺, HeH⁺, He — the spatial ground state of a two-electron singlet has no nodes); otherwise the **fixed-node error** (§3.3) | N_e³–N_e⁴ per fixed absolute error; all-electron cost grows as Z^5.5–Z^6.5 (Ceperley 1986; Hammond, Reynolds & Lester 1987 — as summarised by Ma, Drummond, Towler & Needs, arXiv:0801.2736, who also quote an earlier practical test at ≈ Z^5.2 and fit Z^5.47 to their own noble-gas runs) → pseudopotentials for Z ≥ 3 (§5.2) | holds: the trial orbitals are this solver's Kohn–Sham orbitals; the Jastrow has a universal functional form and per-scene variational parameters (§5.2.3) | **adopted** as the general precise method (stage 4) |
| **Δ-learning on top of LSDA** (Ramakrishnan, Dral, Rupp & von Lilienfeld, *J. Chem. Theory Comput.* 11, 2087 (2015)) | a model of E_precise − E_LSDA | the precise teacher's error + the fit's (posterior σ) | a precise label per sample; the residual is smoother and smaller than the surface, so far fewer labels | holds: a linear ACE model on Z-only labels (the M5 argument) | **adopted** (stage 5) — this is how precision becomes *live* |
| **Coupled cluster CCSD(T) in a Gaussian basis** | the chemistry gold standard | ~1 kcal/mol at the basis-set limit | N⁷ | **violates rule 1** (§3.2) | **rejected** |
| **Coupled cluster in a basis-free (grid) representation** | CC with the grid's Kohn–Sham virtuals as the one-particle space | basis convergence in the number of virtuals is slow (the e–e cusp); unbound virtuals become states of the Dirichlet box (orbital-viewer §4.5, §5.7) | every two-electron integral (ab|cd) over grid orbitals is a Poisson solve: ~N_virt²/2 solves per geometry — ~5 × 10⁵ at a modest 1000 virtuals, each ~10–100 ms on this GPU [est] → hours to days per geometry before the N⁷ contraction | holds (the one-particle space is derived) | **not realistic here** — kept as a note |

**Decision.** The precise method is a **ladder chosen by electron count**, all Z-only:
- **≤ 3 electrons: explicitly correlated Gaussians** (`ecg`), the "exact" tier — deterministic energies and analytic forces, μHa-class.
- **Everything else: fixed-node diffusion Monte Carlo** (`qmc`) with a single-determinant Slater–Jastrow trial function built from **this solver's own LSDA orbitals**, all-electron point nuclei for H and He, this project's own M7 pseudopotentials (with T-moves) for Z ≥ 3. Two-electron singlets are run by DMC too (they are nodeless, so the same code is exact there) — that is the cross-check of the DMC engine against the ECG tier (§9).
- **Δ-learning** turns either into a live model.

### 3.2 Why a Gaussian-basis CCSD(T) violates rule 1

Standard correlated quantum chemistry expands orbitals in a **per-element basis set**: for every element, a table of Gaussian exponents and contraction coefficients optimised for that element's atom (the correlation-consistent cc-pVXZ family is built by optimising exponents per element on atomic correlated energies). Those tables are *fitted per-element numbers that determine the answer*: change oxygen's exponents and water's CCSD(T) energy changes. That is exactly what CLAUDE.md rule 1 forbids ("any fitted per-element … parameter is a design violation"); the parent design rejected Gaussian-basis DFT for the same reason (design §3). A **universal** Gaussian basis (exponents from one Z-scaled even-tempered rule, the way M7 derives its radii) would satisfy rule 1, but reaching the 1 kcal/mol class then needs very large bases and basis-set extrapolation, plus an integral engine and a CC code — a second quantum-chemistry program, CPU-bound, larger than everything this project has built. And CCSD(T) is itself an approximation whose accuracy is established *empirically* on main-group molecules. It stays what it is in `validate`: a source of published reference numbers (test oracles only, design §3), never a solver.

### 3.3 Fixed-node error: how large, from the literature

- **All-electron**, single-determinant Slater–Jastrow DMC with **Hartree–Fock and DFT orbitals** (Slater basis, cusp-projected) on the 55 G2 molecules: bond energies to a **mean absolute deviation of 3.2 kcal/mol**; DMC recovers on average **95 %** of the correlation energy — Nemec, Towler & Needs, *J. Chem. Phys.* 132, 034111 (2010) (abstract, checked 2026-10-15; the VMC figure of 82 % and the cost of "40 CPU hours per electron" for ≈ 25 µHa per electron with τ = 0.01 and 0.001 extrapolated are from the paper's text as rendered by ar5iv, checked the same day).
- With **Burkatzki–Filippi–Dolg (Hartree–Fock-generated) pseudopotentials** on the G2 set: single determinant with **Hartree–Fock orbitals** (largest basis) **3.0 kcal/mol**; with VMC-optimised orbitals **2.1**; with a small complete-active-space multideterminant trial function **1.2**; τ = 0.01 Ha⁻¹, experimental references from Feller and co-workers — Petruzielo, Toulouse & Umrigar, *J. Chem. Phys.* 136, 124116 (2012) (the three MADs checked against the abstract, the potentials and τ against the paper's text, 2026-10-15).
- Two-electron singlets have no nodal error at all (§3.1). Correlated-sampling DMC forces and bond lengths for first-row diatomics are in Filippi & Umrigar, *Phys. Rev. B* 61, R16291 (2000); their H₂ digits are **unverified** here and not quoted.
- **None of these studies used LSDA nodes from a grid together with LSDA- or HF-TM pseudopotentials of our own recipe**, which is what this project will have: transferring their numbers to ours is an extrapolation, and the water oracle (§9) is the measurement.

So the honest expectation for this project's first DMC (single determinant, LSDA nodes, own pseudopotentials in the QMC variant of §5.2.1): **atomization energies within a few kcal/mol** for first-row molecules, an order of magnitude better than LSDA and short of "chemical accuracy" (1 kcal/mol), which needs multideterminant nodes (§12, not planned). Water's own number is *measured* by the oracle (§9) and recorded, never assumed; the per-molecule water rows of the cited studies are pinned from their supplements at task time.

**The honesty statement (the physics reviewer's estimate, 2026-10-15, adopted).** With the review's M1/M2 fixes in place, H₂O's TAE_e at the end of stage 5 is expected to be **underbound by ~3 ± 1.5 kcal/mol** against the exact 233.0: fixed node from single-determinant LSDA nodes ~1.5–3; the HF-reference, no-NLCC TM pseudopotential ~0.5–2 (unverified); T-moves vs locality ~0.3; the missing core–valence correlation 0.38; the live Δ-model adds ≤ 0.6. That is a ~60–65 % chance of landing within 4 kcal/mol. With M7's NLCC/KB potentials used as they are (rejected, §5.2.1) the error would be unpredictable, 5–10 kcal/mol plausible. LSDA, for comparison, is off by tens of kcal/mol.

### 3.4 Cost per geometry (estimates — marked [est]; Task 4.0 and Task 3.1 measure them first)

Anchors: (i) **all-electron DMC to ≈ 25 µHa per electron took 40 CPU-hours per electron** on a 3 GHz Xeon core, two time steps extrapolated (Nemec et al. 2010); (ii) this solver's measured warm DFT solves (`references.md`, M3 Task 10: H₂ 49³ ~0.1 s, H₂O 57³ ~1.0 s, H₂O 97³ ~1.8 s on the RX 7900 XTX). Derivations:

- *DFT at Precise* (h = 0.10): points ∝ h⁻³, LOBPCG iterations ∝ 1/h (pseudopotential spec §1.3). H₂O in its 14 Bohr box is 141³: from 97³ (h = 0.146), ×(141/97)³ × (0.146/0.10) ≈ 4.5 → **~8 s per MD step** [est]. H₂ (12 Bohr, 121³): ×(121/49)³ × 2.5 ≈ 38 from 0.1 s → **~4 s** [est, an overestimate: the GPU is latency-bound below ~10⁶ points].
- *DMC, H₂O*: from anchor (i), ±0.25 mHa all-electron ≈ 400 core-hours with two time steps → ±1 mHa (16× fewer samples) with one time step ≈ 12 core-hours (2009 core). This CPU core is ~2–3× faster, and a GPU walker code is ~10–50 cores' worth for this gather-bound workload [est]; pseudopotential O removes most of the Z^5.5–6.5 core cost (×5–10 [est]). → **all-electron ~10–30 min, pseudo-O ~1–5 min per geometry at ±1 mHa** [est]. Plus a one-off VMC Jastrow optimisation per job (~10–30 min [est]) and one Precise-quality DFT solve for the orbitals per geometry (~8 s).
- *DMC, H₂ / H₃⁺* (2 e⁻, Z = 1): no core, tiny variance; **seconds to a minute** at ±0.1 mHa [est]. 
- *DMC, a 10-atom molecule* — glycine NH₂CH₂COOH (10 atoms, 30 valence electrons with pseudo C/N/O): fixed-error cost ∝ N_e³–N_e⁴, so ×(30/8)³⁻⁴ ≈ 50–200 over pseudo-H₂O → **~1–6 h per geometry** [est]; its DFT orbitals at Precise (20 Bohr box, 201³) ~1–3 min.
- *ECG*: K functions, K² matrix elements of closed form (overlap, kinetic, Coulomb through the Boys function), a K³ generalised eigenproblem, plus stochastic-variational refinement (Varga & Suzuki, *Phys. Rev. C* 52, 2885 (1995)). K = 200 for 2 e⁻: ms per energy, ~0.5 s for a refinement sweep warm-started from the previous geometry [est]. 3 e⁻ (6 permutations, K ~ 500–1000): ~10 s–1 min [est].

| per geometry, RX 7900 XTX class [est] | H₂ | H₃⁺ | H₂O | glycine (10 atoms) |
|---|---|---|---|---|
| DFT LSDA at Precise, one warm MD step | ~4 s | ~4 s | ~8 s | ~1–3 min |
| exact tier (ECG) | ~0.5 s | ~2–5 s | — (10 e⁻) | — |
| DMC ±1 mHa (pseudo for Z ≥ 3) | ~10 s (exact: nodeless) | ~30 s (exact: nodeless) | ~1–5 min (+ orbitals 8 s) | ~1–6 h |
| precise labels a Δ-model needs [est, §7.6] | 20–50 | 50–100 | 100–300 | 300–1000 |
| a 1000-frame precision render, every frame solved | ~1 h (DFT) / ~10 min (ECG) | ~1.5 h (ECG) | ~2 h (DFT) / 1–3 days (DMC) | ~1 day (DFT) / months (DMC) |

The last row is why the DMC render integrates a learned surface (§4) and why Δ-learning is the main route (§7): a 10-atom molecule with DMC is at the edge of this hardware even then (a Δ-model costs days of GPU time).

## 4. Forces for dynamics: what a precision render integrates

### 4.1 The difficulty

- **ECG** gives forces cleanly: with the linear coefficients normalised (cᵀSc = 1), F_I = −cᵀ(∂H/∂R_I)c − Σ_k (∂E/∂s_k)·(∂s_k/∂R_I); ∂H/∂R_I holds the electron–nucleus attraction derivative (closed-form over Gaussians) **and the ion–ion repulsion term**, and the second sum is the **Pulay term** of centres that move with their nuclei (§5.1.4).
- **DMC forces are noisy and biased.** The bare Hellmann–Feynman estimator has infinite variance (the force density diverges at the nuclei); the renormalised and zero-variance zero-bias estimators (Assaraf & Caffarel, *J. Chem. Phys.* 113, 4028 (2000); *J. Chem. Phys.* 119, 10536 (2003)) cure the variance and remove the leading bias of the mixed distribution, and correlated sampling with the space-warp transformation gives finite-difference forces (Filippi & Umrigar 2000). **All of them need ∂Ψ_T/∂R_I** — the derivative of the trial wavefunction with respect to a nuclear position. For a Gaussian-basis trial function that is analytic; for **this project's trial function** the orbitals are grid Kohn–Sham orbitals whose R-derivative is a *response* of the SCF (one linear-response or finite-difference KS solve per nuclear coordinate, 3N per geometry), and the space-warp route needs 3N + 1 correlated DMC runs per geometry. Noisy-force QMC molecular dynamics has been done — Attaccalite & Sorella, *Phys. Rev. Lett.* 100, 114501 (2008) absorb the force noise into a Langevin thermostat; Zen, Luo, Mazzola, Guidoni & Sorella, *J. Chem. Phys.* 142, 144111 (2015) ran liquid water on VMC forces — with analytically differentiable trial functions and supercomputer time.
- **Per-frame DMC energies along a trajectory are affordable** with continuous walker propagation: Grossman & Mitas, *Phys. Rev. Lett.* 94, 056403 (2005) exploit the 10–100× ratio of electronic to ionic time scales and propagate the same walkers along the MD path, so each frame costs a fraction of an independent DMC.

### 4.2 The decision (what each render kind integrates)

| render kind | the motion comes from | per-frame energies | what "precise" honestly means here (rule 7) |
|---|---|---|---|
| **R1 — every step solved, deterministic teacher** (DFT at any quality, incl. Precise; the ECG exact tier) | velocity Verlet on the teacher's own forces — the existing `sim::Dynamics`, NVE or CSVR | the teacher's | DFT at Precise: *the best LSDA this solver can do* — LSDA's bond energies stay tens of kcal/mol off; at h = 0.10 the smeared-H bond is 0.807 Å vs 0.741. ECG: **the exact Born–Oppenheimer electrons** (μHa) for ≤ 3 e⁻; what remains is classical nuclei (no zero-point energy: for H₂ ω_e/2 = 2200.6 cm⁻¹ = 6.3 kcal/mol, harmonic, from ω_e = 4401.21 cm⁻¹) and the non-relativistic BO Hamiltonian |
| **R2 — learned precise surface, refined on the fly** (DMC teacher; also ECG) — **the default for DMC** | velocity Verlet on **E_base + E_Δ**, where E_Δ is a Δ-model (§7) that is refitted whenever the precise teacher labels a new geometry; the trajectory pauses for a label when the Δ-model is uncertain at the current geometry, or every k frames | E_base + E_Δ ± σ_Δ every frame; **E_DMC ± σ_stat** at every labelled frame (continuous walkers) | motion on the DMC surface *as learned*: errors are the fixed-node error (§3.3, the tier statement), the pseudopotential error (measured, §9), the DMC time-step bias (measured), and the fit's posterior σ (shown per frame). The forces are the analytic gradient of a smooth model — conservative, noise-free, NVE-capable; refits are booked like M5's `refit_jump`, so E_conserved stays continuous |
| **R3 — precise energies along a cheap trajectory** (any stochastic or slow teacher) | the base surface (LSDA DFT or the LSDA surrogate) | E_base every frame; E_precise ± σ every k-th frame (continuous walkers) | *LSDA motion with precise energies*: the bond lengths and frequencies are LSDA's; the energies on that path are DMC's. Labelled exactly so ("motion: LSDA") |
| ~~R4 — every step on DMC forces with a noise-absorbing Langevin thermostat~~ | — | — | **rejected for this milestone [provisional, P3]**: needs ∂Ψ_T/∂R for grid orbitals (3N response solves or 3N + 1 correlated runs per step — §4.1), samples only the canonical ensemble (no NVE), and at the §3.4 costs a 1000-step water run is weeks. A later item if analytically differentiable trial orbitals ever exist |

Classical nuclei remain throughout (design §2); path-integral quantum nuclei — which matter at the same kcal/mol scale for X–H bonds (water's ZPVE is 13.29 kcal/mol, W4 table) — are a separate later item, and the render's honesty line says "classical nuclei: zero-point motion not included".

R2 is the honest answer to "solve each step" for DMC: every frame *is* answered by a model of the DMC surface whose uncertainty is shown, and the frames the user cares about can be checked by real DMC (cadence k, default "when uncertain + every 50 frames" [provisional, P4]).

## 5. The precise teachers

### 5.1 Exact tier: explicitly correlated Gaussians (`crates/ecg`, CPU f64)

#### 5.1.1 Wavefunction
Ψ = 𝒜 Σ_k c_k Θ_k φ_k(r), with φ_k(r) = exp(−(r − s_k)ᵀ A_k (r − s_k)) over the 3N electron coordinates (A_k a symmetric positive-definite N × N matrix ⊗ I₃, s_k a floating centre per electron — "shifted" ECGs, the form Pavanello & Adamowicz used for H₃⁺), 𝒜 the antisymmetriser over the N! permutations with the spin function Θ of the requested multiplicity. Matrix elements of overlap, kinetic energy, electron–electron and electron–nucleus Coulomb are closed-form (the Coulomb ones through the Boys function F₀; Mitroy et al. 2013 §III).

#### 5.1.2 Rule 1
No basis table exists. The nonlinear parameters are found by the **stochastic variational method** (Varga & Suzuki 1995): candidates are drawn from one universal distribution — widths log-uniform in a range scaled by the scene's own Z² (core) and by the scene's own interatomic distances (bonds), centres near nuclei or bond midpoints of the scene — and each candidate is kept if it lowers the energy most. The universal constants of the draw (§14) are the whole recipe, as M7's §4.9 ledger is for pseudopotentials. Different elements end up with different parameters because their Z and geometry differ — outputs, not inputs.

#### 5.1.3 Scope
N ≤ 3 electrons, any charge and multiplicity the solver allows (H, H₂⁺, H₂, H₃⁺, HeH⁺, He, H₃, HeH, Li, Li⁺ …). N = 4 is possible (24 permutations) and is a later extension. Electron count above 3 → the exact tier refuses the scene with "the exact tier handles at most 3 electrons; use the DMC tier".

#### 5.1.4 Forces and the moving nuclei
Within an MD *segment* the widths A_k are frozen and **each centre is attached to a nucleus**: s_k = R_{I(k)} + d_k with the offset d_k frozen (review M3). A space-fixed frozen basis would not do: the force on a rigid translation of the nuclei at fixed centres is not zero for an incomplete basis (ΣF_I ≠ 0), and that surface is not rotation-invariant, so NVE would drift in momentum. With attached centres the frozen surface is translation- and rotation-invariant, the force of §4.1 (with its Pulay term Σ_k ∂E/∂s_k · ∂s_k/∂R_I, analytic: the ∂E/∂s_k are the derivatives the gradient optimisation of the centres already needs) is the exact derivative of the energy the trajectory sees, ΣF_I = 0 exactly, and Verlet conserves energy on that surface. Every `ECG_REFRESH_STEPS` steps or when any nucleus has moved more than `ECG_REFRESH_DIST` (§14) from where the basis was optimised, the job runs an SVM refinement sweep warm-started from the current basis (centres translated with their nearest nucleus first); the energy change at fixed positions is booked as a surface jump (the M5 `refit_jump` mechanism, `EvalStats::refit_jump`), so E_conserved stays continuous and the drift readout measures the integrator. The accuracy loss of a frozen basis over a segment is measured (Task 3.2) and is the reason for the two constants.

#### 5.1.5 Accuracy target
1e-6 Ha (μHa) on the oracle geometries, which is three orders below the DMC tier and below anything a frame displays. ECG energies are variational upper bounds: the oracle asserts E_ECG ≥ E_exact − 1e-10 *and* E_ECG ≤ E_exact + tol (§9).

### 5.2 DMC tier (`crates/qmc` CPU f64 reference; `crates/qmc-gpu` walkers in WGSL)

#### 5.2.1 The Hamiltonian
H = −½Σ∇_i² + Σ_{i<j} 1/r_ij + Σ_{i,I} v_I(r_iI) + Σ_{I<J} Z_v,I Z_v,J/R_IJ, with v_I = −Z/r (**point nuclei**, no smearing) for species without a core (H, He), and an **M7 pseudopotential in its QMC variant** for species with one: the **semilocal** form v_I = V_loc(r) + Σ_l δV_l(r) P_l(cos θ) — the standard QMC form (review M1: the KB-separable operator ⟨r|V_KB|r′⟩ = E_KB β(r)β(r′) Σ_m Y_lm Y*_lm would need a radial × angular integral of Ψ, which a sphere quadrature at the electron's radius does not evaluate) — generated by the recipe variant **`xc = HF`, `nlcc off`** (review M2, below), unfiltered: the Fourier filter of M7 §5.1 is a grid device and QMC has no grid. The trial orbitals (§5.2.2) are solved on the grid with the **same variant**, in its KB form and filtered; KB on the grid versus semilocal in DMC is a trial-function difference only, which DMC's energy inherits only through the nodes. Li, Na, Mg (refused by M7 until ONCV) are refused by the DMC tier with M7's reason.

**Why pseudopotentials for Z ≥ 3** (not only cost): the trial orbitals come from the grid solver (§5.2.2), and an all-electron O orbital from the grid is the smeared-nucleus 1s, twice its true size at σ = 0.25 (M7 §1.2) and still 0.8 Zσ at Precise — a trial function whose core is wrong by tens of Ha has a variance that makes DMC unaffordable, on top of the Z^5.5–6.5 cost. With M7 (in its QMC variant), the grid orbitals of a pseudised species are smooth, cusp-free valence orbitals of nearly the same pseudo-Hamiltonian — a good trial function, not the exact orbitals of the operator DMC solves.

**Nonlocal pseudopotentials in DMC.** The nonlocal operator is not a multiplicative potential, so DMC needs an approximation. The **locality approximation** (Mitas, Shirley & Ceperley, *J. Chem. Phys.* 95, 3467 (1991)) replaces V_NL by Ψ_T⁻¹V_NLΨ_T: its error is second order in Ψ_T − Ψ₀ but it is not variational and depends on the Jastrow. **T-moves** (Casula, *Phys. Rev. B* 74, 161102 (2006)) restore an upper bound; the original form loses size consistency at finite time step (Casula, Moroni, Sorella & Filippi, *J. Chem. Phys.* 132, 154113 (2010) give size-consistent versions), and the T-move schemes carry larger time-step errors than the locality approximation, which Anderson & Umrigar (*J. Chem. Phys.*, 2021, doi:10.1063/5.0052838) reduce with a modified scheme while keeping the upper bound. **Decision: the size-consistent T-moves of Casula et al. 2010** — variational, size-consistent (the Anderson–Umrigar variant is the upgrade if the measured time-step bias at the tier τ exceeds 0.5 mHa on H₂O); the locality approximation is implemented too, as the cross-check, and the difference between the two is reported per oracle as the pseudopotential-localisation uncertainty (rule 7). The angular integral of the semilocal projector over the sphere of the electron's radius, ∫ dΩ′ P_l(r̂·r̂′) Ψ(…r r̂′…)/Ψ, uses a fixed universal quadrature (§14), randomly rotated per evaluation; for l ≤ 2 the projected angular dependence has degree 2l ≤ 4, inside the rule's exactness.

**Why M7's potentials cannot be used as they are (review M2).** M7 unscreens with v_xc[ρ_v + ρ_pc] from a spin-restricted LSDA reference and relies on the nonlinear core correction to restore the core–valence exchange-correlation in the molecule. QMC has no ρ_pc: with M7's V_ion as it is, the valence electrons would lack exactly the core–valence XC the unscreening removed — an uncontrolled error. QMC potentials (Burkatzki, Filippi & Dolg, *J. Chem. Phys.* 126, 234105 (2007), "energy-consistent pseudopotentials for quantum Monte Carlo calculations") are generated from Hartree–Fock atoms without NLCC: in HF the core–valence interaction is Hartree plus exchange, both carried by the unscreened potential, and correlation is left to the QMC. **Decision:** an M7 recipe variant **`xc = HF`, `nlcc off`** — a radial spin-restricted Hartree–Fock reference atom feeding M7's TM/KB pipeline and gates (Task 4.7, unconditional) — part of `recipe_hash` and of the precise key's `pseudo` line; the trial orbitals use the same variant. Rule 1 holds (Z in, potential out; a universal switch). Its remaining error (frozen core, the 0.38 kcal/mol core–valence correlation of water, the TM construction) is *bounded* offline against all-electron DMC (§9, Task 4.6) — a bound, not a decomposition (§9 note). The choice of potential is part of the Task 4.0 go/no-go, and M7's spec records the QMC variant as a future requirement.

#### 5.2.2 Trial wavefunction: this solver's orbitals
Ψ_T = e^J · det[φ_i↑(r_j↑)] · det[φ_i↓(r_j↓)]. The orbitals φ are the occupied LSDA Kohn–Sham orbitals of the scene at the **trial grid** (the Precise quality, h = 0.10, by default; part of the tier key, §5.3), solved by the teacher's own `GpuKs` (warm along a trajectory) under `IonicPolicy::CoresOnly` — pseudised cores, H and He at σ = h. They are read once per geometry from `GpuState::orbitals_up/dn` (orbital-viewer §4.1) through a `providers::ks` method that is internal to the `providers` crate (rule 5: `sim` never sees orbitals), prefiltered on the CPU in f64 into cubic **B-spline coefficients** (C², so the local energy's Laplacian is continuous) and uploaded as f32. Evaluation inside the box: a 64-point gather per orbital with value, gradient and Laplacian.

- **Cusps.** The e–e cusp (½ antiparallel, ¼ parallel; Kato) is in the Jastrow. For H and He the grid orbitals are cusp-free at the nucleus (σ-smeared: zero slope there), so the **e–n Jastrow term carries the whole cusp −Z** exactly; the remaining inner shape error within ~σ of the proton is a trial-function error that costs variance, not accuracy (DMC is exact for H, H₂, H₃⁺, HeH⁺, He whatever the trial's shape, since they are nodeless). For pseudised species there is no cusp (the potential is finite).
- **The box node.** Grid orbitals vanish at the Dirichlet faces, so Ψ_T has a node there and walkers are confined to the cell. The bias is the box's confinement, the quantity the M5 box rule (`box_affected_face_density`) already measures. A DMC label whose trial solve is box-affected is **used but never trained on** (the S17 rule, unchanged). Measured on the H atom (§9).
- **Nodes** are those of the single LSDA determinant. VMC orbital optimisation (which took the G2 MAD from 3.0 to 2.1 kcal/mol in Petruzielo et al.) is a later, separately measured step (§12).

#### 5.2.3 Jastrow: universal form, per-scene variational parameters
J = Σ_{i<j} u(r_ij) + Σ_{i,I} χ_{Z_I}(r_iI) + Σ_{i<j,I} f_{Z_I}(r_iI, r_jI, r_ij), the functional form of Drummond, Towler & Needs, *Phys. Rev. B* 70, 235119 (2004): polynomials in r with a smooth cutoff (r − L)^C Θ(L − r), the cusp conditions imposed as linear constraints on the coefficients, separate u for parallel and antiparallel spins. **Rule 1.** The *form*, the expansion orders, the cutoff rule (L from the scene's own size, §14) and C are universal. The coefficients — per species (χ, f) and per spin pair (u) — are **variational parameters of this scene's wavefunction**, determined by minimising the VMC energy (with a variance term) by the **linear method** (Umrigar, Toulouse, Filippi, Sorella & Hennig, *Phys. Rev. Lett.* 98, 110201 (2007)). They have the status of the orbital coefficients: produced by the variational principle from Z and the geometry, never typed in, never shipped, never fitted to experiment. They are cached with the job and the label store (warm start for the next geometry), and a fresh scene starts from zero (cusp terms only).

#### 5.2.4 DMC algorithm
Importance-sampled fixed-node DMC with the small-time-step-error algorithm of Umrigar, Nightingale & Runge (*J. Chem. Phys.* 99, 2865 (1993)): drift-diffusion moves electron by electron with the UNR drift limiting near nodes, Metropolis acceptance, branching on the effective time step, a large population (≥ `DMC_WALKERS`, §14) with population control by reconfiguration; walkers that cross the trial node are rejected (fixed node). The time step is a tier constant; the time-step bias is measured on each oracle by a τ ladder and reported (Task 4.4), and the oracles extrapolate τ → 0. Energies: the mixed estimator ⟨E_L⟩. **Statistical errors** come from reblocking (Flyvbjerg & Petersen, *J. Chem. Phys.* 91, 461 (1989)) of the generation-averaged energies; an error bar is published only when the blocking plateau is detected, otherwise the label says "error bar not yet reliable" and the job runs on. Accumulation: per-walker local energies in f32 on the GPU, workgroup partial sums, Neumaier-compensated f64 on the CPU (rule 4); determinant inverses refreshed in f64 on the CPU every `DET_REFRESH` sweeps (f32 Sherman–Morrison drifts); branching weights and population in f64 on the CPU.

#### 5.2.5 Continuous walkers along a trajectory
Between consecutive labelled geometries of one job, the walker population, the Jastrow and the B-spline orbitals are carried over (Grossman & Mitas 2005): after the nuclei move, the orbitals are re-solved (warm DFT), the population is re-equilibrated for `DMC_REEQUIL` generations (discarded), and accumulation restarts. The first geometry of a job pays the full VMC optimisation and equilibration.

#### 5.2.6 GPU
One walker per invocation group; positions f32 **relative to the cell centre** (|r| ≤ ~10 Bohr, f32 ulp ~1e-6 Bohr), orbitals as storage buffers, Jastrow in registers. CPU f64 reference first (`qmc`), WGSL against it (`qmc-gpu`, Opus via `gpu-kernel-engineer`): bitwise-equal moves for a fixed random stream are not expected (f32), so the GPU's energies are gated statistically against the CPU's (§9). The DMC job shares the device with the live viewer (§6.3).

### 5.3 Tiers, revisions and keys

**`PreciseKey`** (canonical text, the user-data §3.2 rules: one field per line, floats with `{:?}`, FNV-1a 64 file names):

```text
atomic-sandbox precise key v1
species 1,8
charge 0
multiplicity lowest
method dmc
tier sj1-tm
method_rev 1
trial grid 141 h 0.1 sigma 0.1 tol 1e-6 teacher Kohn-Sham (GPU) teacher_rev 2
pseudo recipe 3f2a… xc hf nlcc off form semilocal policy cores-only        (illustrative values; the recipe is M7's recipe_hash)
tau 0.005
```

- `method` ∈ {`ecg`, `dmc`, `dft`}; `tier` names the approximation level inside it (`ecg`: `exact`; `dmc`: `sj1-tm` = single-determinant Slater–Jastrow, size-consistent T-moves; `sj1-la` = locality approximation; later `sjm-…` multideterminant).
- **`method_rev: u32`** plays `TEACHER_REV`'s role for the precise teacher: any change to the method's output (a Jastrow form change, a new quadrature, a DMC algorithm change) bumps it; a `validate` tripwire per method (H₂ by ECG at R = 1.4011 to 1e-9 Ha; H₂ by DMC with a fixed seed on the CPU reference, bitwise) enforces it, exactly like the user-data tripwire.
- **The trial line** is the full user-data key of the orbital solve (it carries `TEACHER_REV`): DMC's nodes depend on it.
- **Isotopes are recorded, not keyed** (ruling P2: the BO surface is mass-independent).

**Tier badges and their plain-language meaning** (§8.6 has the exact copy): `LSDA` · `LSDA (Precise grid)` · `exact` (ECG) · `DMC` · `LSDA + Δ(exact)` · `LSDA + Δ(DMC)`.

## 6. The precision render

### 6.1 The job model
A **job** is a scene snapshot plus settings, queued, run by the job runner, and written to disk as it goes:

```text
Job {
  id: "2026-10-15T1746-h2o-dmc-3f9c", name: "water, DMC, 300 K",
  kind: Render { render: R1 | R2 | R3, steps, dt_fs, thermostat, field, walls, density_every, check_every }
      | Label { geometries | from_pool(plan), target_sigma }        // precise labels for a Δ-model (§7)
      | Rebuild { delta_key }                                       // base energies at stored labels, then refit
  scene: Scene (nuclei incl. velocities and masses, cell, charge, multiplicity, σ, ionic policy)
  teacher: TeacherSpec (the precise key's fields, or a DFT quality level), base: Option<LibraryKey>
  seed: u64, created, state: Queued | Running | Paused{why} | Done | Failed{error} | Cancelled,
  progress: { done, total, stage }, cost: { per_unit_s: rolling median, wall_s }, eta_s
}
```

- **Create.** Simulation ▸ **Precision render…** (and Ctrl-K, and the toolbar's jobs chip) opens the render form (§8.4) prefilled from the live scene, the live dt, thermostat, walls and field. Submitting snapshots the scene at that instant; **later edits of the live scene never touch the job**.
- **Rule 8.** The job's dt is fixed at submission; playback speed is frames per wall second and never changes dt.
- **The render runs `sim::Dynamics`** — the same driver, integrator, thermostat, walls, field and bookkeeping as the live run — with the job's provider: `GpuKs`/`CpuKs` at the chosen quality (R1 DFT), `EcgKs` (R1 exact), or `DeltaKs` over a base with the job's Δ-model and a DMC labeller (R2), or the base plus a side labeller (R3). One `tick` per frame.

### 6.2 Files and checkpoint/resume
Jobs live in `<root>/jobs/<id>/` (the user-data root; tests pass their own; automation runs use a temp root — user-data L2):

```text
job.json          # the Job record (atomic write, userdata::atomic_write) — the truth for state and settings
frames.asfr       # append-only frame log
state.ckpt        # teacher warm state (atomic write, every CKPT_EVERY s): DMC walkers + RNG + Jastrow; ECG basis; Δ-model; DFT: none
density/000120.f16  # keyframe densities (every density_every frames; f16, the frame's grid)
log.txt           # the job's own log lines (also shown in the manager)
```

- **`frames.asfr`**: magic `ASFRAME1`, a JSON header (job id, n atoms, Z, masses, dt, units "atomic"), then length-prefixed records each closed by its FNV-1a 64: step, t, positions and velocities (f64), forces (f64), the thermostat's RNG state, E_pot, KE, E_conserved, booked work rows, T, the per-frame error bars (σ_model, σ_stat, label flag), the source (teacher / model / labelled), solve wall time, the density keyframe index, notes. **Every record holds the full integrator state**, so a job resumes from its last complete record; a torn last record (crash mid-write) fails its checksum and is truncated on open.
- **Checkpoint consistency (review S3).** `state.ckpt` records the frame index it belongs to and is written **with every teacher refresh, label and Δ refit** (and every `CKPT_EVERY` s otherwise). On resume, frames newer than the checkpoint's frame are **truncated** and recomputed, so a warm state is never paired with frames it did not produce. For DFT (no warm state) the last complete record is the checkpoint.
- **Resume.** On launch, jobs that were `Running` become `Paused{"interrupted by exit"}`; with **Resume jobs at launch** (factory on [provisional, P5]) they restart in queue order. A resumed DFT job starts cold from the last frame (one cold solve) — **it continues the same physics, not bitwise the run that would have happened** (the SCF lands within its tolerance from a different start, and MD is chaotic); ECG and DMC jobs restore their warm state from `state.ckpt` (DMC with its RNG: a CPU-reference resume is bitwise). The manager says which.
- **Disk.** Frames are ~200 B per atom; densities dominate (141³ f16 = 5.6 MB per keyframe). The form shows the estimate before submitting ("≈ 560 MB: 100 densities at 141³"); the manager shows each job's size; **deleting** a job renames its folder to `<id>.deleted-<unix>` (user-data §5.5 style: the app never destroys data the user did not explicitly delete; a footer offers "free X MB" which then removes the renamed job folders, with a two-click confirmation; it never touches `models/` or `models/precise/`, whose forgotten files stay the user's to delete, user-data §5.5).

### 6.3 The runner and the GPU
- One **runner thread** owns the queue; it runs **one GPU job at a time** and, alongside it, CPU jobs (ECG; CPU DFT) on up to `cores − 2` threads.
- **Sharing the GPU with the live sandbox** [provisional, P6]: the runner submits work in chunks (one SCF step; `DMC_CHUNK` generations) and, by the setting **Background jobs while the live run plays**: `pause` / `share` (factory) / `full speed`. In `share` the runner yields after each chunk when the live worker has a solve pending, so the live run's latency is at most one chunk (≤ ~50 ms of GPU time [est], measured in Task 1.3); the render thread was never blocked by a solve and is not now.
- **VRAM.** Jobs allocate against the same budget guard as the live solver (`ATOMIC_SANDBOX_VRAM_GB`, default 12 GB); a job that would exceed it waits ("waiting for GPU memory: 3.1 GB needed, 1.2 GB free") instead of failing.
- **Errors.** A provider error fails the job with its message, keeps every frame, and offers "Retry from frame k".
- The **live sampling pass** (M5/M6, run by the live worker) appears in the queue as a read-only row with its own progress and Cancel — it is not moved into the runner [provisional, P7]: it pauses the live run by design and belongs to the live worker.

### 6.4 Progressive playback
**Playback mode** turns the 3D view into a player of a job's frames, reusing the viewer (nuclei, density volume/isosurface, bond lines, labels, the dock) with a **timeline** in place of the transport while it is active:

```text
┌ water, DMC, 300 K · R2 · LSDA + Δ(DMC) ─────────────────────── ● rendering 412 / 1000 · ETA 3 h 10 min ┐
│ ⏮  ▶  ⏭   [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  frame 388 · t 97.00 fs  ×1 (30 frames/s) │
│              ▲ labelled frames (DMC)   ·   ░ not computed yet   ·   ▓ computed     [follow live] [exit] │
│ E: ─╮╭─╮╭──── ±0.6 mHa (model σ) · frame 380 DMC −17.2134 ± 0.0008 Ha · drift −3e-6 Ha/ps · T 296 K │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

- Scrubbing and play stop at the **computed frontier**; **Follow live** keeps the head at the frontier (it advances as frames arrive).
- Playback speed is frames per wall second (1–120), never dt (rule 8).
- **Density** shows the nearest keyframe at or before the frame, labelled "density from frame 380 (t = 95.00 fs)" when it is not the frame itself — the M5b honesty rule; frames between keyframes can blend the superposition of the M5b "atomic superposition" cloud, labelled likewise. **Orbitals** are not stored: **Re-solve this frame** runs the job's teacher (or the base DFT) at that frame's geometry and shows its density and, with the orbital viewer, its orbitals.
- **The sparkline above the timeline** shows E_conserved, the per-frame error band, and labelled frames as dots with their ± bars.
- **Edits.** A recording cannot be edited ("this is a recording — Fork to edit"). **Fork live scene from this frame** loads the frame's positions and velocities into the live sandbox (a `SetScene`, so the model library resolves as for any scene load). **Re-render from this frame…** opens the render form prefilled with that frame and the job's settings, as a new job.
- **Exit** returns to the live scene exactly as it was (the live worker kept running or paused per §6.3).

### 6.5 Per-frame error bars (rule 7)
Each frame shows what it is and how sure:
- R1 DFT: "LSDA at h = 0.10 (Precise) · SCF tol 1e-6" — no stochastic bar; the method's systematic error is the tier statement, not a per-frame number.
- R1 ECG: "exact electrons · classical nuclei", with the **measured frozen-basis error of the current segment** (the gap between the segment's energy and the re-optimised energy at its end, Task 3.2) — the 1 µHa accuracy belongs to the oracle geometries and is not claimed per frame (review S4).
- R2: "model of DMC: ± σ" every frame, σ = √(σ_base² + σ_Δ²) (§7.4); labelled frames add "DMC: E ± σ_stat"; the tier line adds "fixed-node (single determinant) with our pseudopotentials: bond energies typically a few kcal/mol too weak (water expected ≈ 3 ± 1.5 kcal/mol); time step τ = 0.005: bias ≤ (measured) mHa".
- R3: "motion: LSDA; energy: DMC ± σ_stat at labelled frames".

## 7. Precision models (Δ-learning)

### 7.1 What is learned
For a scene key (species, charge, multiplicity) and a base surface (a user-data library key: the LSDA surface at some grid/σ/teacher), the **Δ-model** fits the *binding-energy residual*

  Δ(R) = [E_precise(R) − Σ_I E_precise,atom(Z_I)] − [E_base(R) − Σ_I E_base,atom(Z_I)],

so Δ → 0 when the atoms separate beyond r_cut (the M5 reference-energy argument, §3 of the surrogate spec): the atoms' precise energies come from the same teacher (H: −0.5 exactly by DMC/ECG; pseudo-O: its own DMC, with the multiplicity chosen by a **DMC scan** over the allowed multiplicities — review S2 — not by the LSDA S9 scan).

The **live precision model** is E(R) = E_base-model(R) + E_Δ(R), with forces the analytic gradients of both (M5 descriptors and derivatives, reused). A smooth, small Δ needs a smaller basis than the base surface (default `DELTA_D` = 2 total degree, body order 3 [provisional, measured in Task 5.2]).

### 7.2 Labels are stored once and reused across base surfaces
Precise labels are expensive and **do not depend on the base**. They live in a **label store** keyed by the `PreciseKey` alone: `<root>/models/precise/<hash>.labels` — geometry, Z, E_precise, σ_stat, the trial-solve face density, cost, date, and the warm state needed to continue (Jastrow parameters per geometry cluster; ECG bases). A **Δ-model** is *derived*: (label store) × (base key) → evaluate the base at every stored geometry (one DFT solve each — seconds, a `Rebuild` job) → fit. Changing the Quality level therefore rebuilds the Δ-model in minutes from the same precise data instead of re-running hours of DMC. The derived Δ-model is cached in `<root>/models/precise/<hash>.delta` (key: base key + precise key, both canonical texts, user-data §3 rules). Older builds never look in `precise/`.

### 7.3 The fit with noisy labels
The M5 fit is a Bayesian linear regression whose evidence chooses (λ, s_E, s_F). For Δ:
- Rows are **energies only** for DMC labels (no DMC forces, §4.1). ECG labels carry energies (σ ≈ 0 at the µHa level) and forces. ECG force labels are **frozen-basis derivatives**, not derivatives of the re-optimised surface, so they enter with a **measured σ_F**: the rms of (frozen-basis force − central difference of warm-start re-optimised energies) on the oracle systems (Task 3.2).
- **Heteroscedastic noise:** row i is whitened by √(s_E² + σ_i²), σ_i the label's own blocked DMC error and s_E the evidence-chosen model-error floor ("two-noise" becomes "known per-row statistical noise + learned representation noise"). With σ_i = 0 (ECG) this is M5's fit unchanged.
- Posterior Σ_Δ = (ΦᵀWΦ + λI)⁻¹; predictive variance of a new geometry σ_Δ²(x) = φ(x)ᵀΣ_Δφ(x) (the epistemic part, which is what the trajectory's error bar shows); of a new *label* add s_E² + σ_stat².
- **Combined uncertainty.** The base model and the Δ-model are fitted on independent data, so their epistemic variances add: σ²(E) = σ_base² + σ_Δ². With the base answered by DFT itself, σ_base = 0. The fixed-node, time-step and pseudopotential errors are **systematic** and are never folded into σ: they are the tier statement (rule 7).
- **Difference labels** [measured in Task 5.4, adopted only if they pay]: correlated-sampling DMC of a geometry pair (R, R + δ) with the space-warp transformation gives E(R + δ) − E(R) with a much smaller σ than two independent runs; it enters the linear fit as one row φ(R + δ) − φ(R). Cheap gradient information without ∂Ψ_T/∂R.

### 7.4 Trust, fallback and audit for a model of noisy labels
The M5 rules are kept in shape; the thresholds change meaning:
- **Trust** needs n ≥ Q′_Δ + `DELTA_MIN_DOF` labels; the reduced χ² of the standardised training residuals r_i/√(s_E² + σ_i²) not below its 99 % interval (the labels' noise overstated); and the closed-form leave-one-out standardised residuals' rms ≤ `DELTA_LOO_MAX` (a linear model's LOO residual is r_i/(1 − h_ii)).
- **χ² cannot catch understated σ_i** (review M4): the evidence raises s_E until the reduced χ² ≈ 1. Instead the fit flags **s_E² ≫ median σ_i²** (above `DELTA_SE_RATIO`, §14) as "label errors understated or the model underfits" — a warning shown with the model, and untrusted above the ratio. The calibration of the σ_i themselves is the DMC suite's job (`error_bars_are_honest`, §9).
- **The base's noise floor.** E_base labels from DFT carry the egg-box (not translation-invariant), which no invariant Δ can represent; it shows up as a floor in s_E, reported with every Δ-model.
- **Answering.** The Δ-model answers where its predictive σ_Δ ≤ `DELTA_ANSWER_STD` = **min(1 mHa, 0.5 mHa × N_atoms)** [provisional] — the per-atom term binds for one- and two-atom scenes, the 1 mHa total from three atoms up — and every atom is inside its training envelope (M5 rule 5).
- **What the user sees.** The per-frame and readout band is the **total** σ = √(σ_base² + σ_Δ²); the per-atom force uncertainty is √(∇φᵀΣ∇φ) per component. Δ forces are never checked against DMC directly (DMC gives no forces): only through energies, and against the ECG tier where it applies.
- **When it cannot** (live play): the step is answered by the base alone and the **precision is visibly lost**: a red banner "precision unavailable here: this geometry is outside what the DMC correction has learned — these steps are LSDA", the sparkline marks the steps (like `history_guess`), and the surface change base+Δ ↔ base is **booked** as a surface jump so E_conserved stays continuous. If **Learn while I play** is on, the geometry is queued as a `Label` job (background, never blocking the run); its label refits the Δ-model, booked again. In a render (R2) the job instead pauses its trajectory for the label.
- **Audits.** At cadence, a fresh precise label at a geometry the Δ-model answered (a background job). Passed iff |E_label − E_model| ≤ 3√(σ_stat² + σ_Δ²) **and** within the tier's answer bound; a failure untrusts the Δ-model until the next refit (M5 audit semantics).

### 7.5 Choosing what to label (active learning against the teacher's cost)
A label costs minutes to hours, a candidate costs microseconds, so selection is worth compute:
1. **Pool.** Geometries from the cheap side: the M5 `SamplingPlan` generators (perturbations, pair scans whose ranges come from the scene, hot bursts) run on the **base surrogate** (or base + current Δ), at the temperature the user plans to simulate — thousands of candidates.
2. **Greedy batch D-optimal selection.** Pick the candidate of largest σ_Δ²(x); update Σ_Δ by the rank-1 formula as if it were labelled (its label is not needed for the variance); repeat until the batch is full or every candidate's σ_Δ is below target. Box-affected candidates are skipped.
3. **Budget first.** The Label form shows "≈ 40 DMC solves × ~3 min ≈ 2 h · until every pooled geometry is below ±1 mHa" before anything runs, and the user can cap time or count.
4. **ECG teachers** label with forces, so their batches are smaller.

### 7.6 How many labels [est]
A Δ-model with energy rows only needs n ≳ Q′_Δ + dofs; at D = 2 the Q′ of one species is a few, of H/O a few tens [est]. The §3.4 table's 20–1000 is that count plus margin and is replaced by Task 5.2's measured learning curve (Δ error vs n on held-out labels, first on the cheap proxy Δ = E_DFT(Precise) − E_DFT(Balanced), whose labels cost seconds, then confirmed on ECG for H₃⁺ and on DMC for H₂O).

### 7.7 The live provider
`providers::delta::DeltaKs: ForceProvider` wraps the base provider (the estimated `SurrogateKs`, or a DFT provider) and a Δ-model; `as_trainable()` exposes the base's `Trainable` unchanged (the base keeps learning from DFT as today) and a new `PreciseTrainable` for the Δ side (label queue, refit, status) — no downcasts (M6 §9). `EvalStats` gains `delta: Option<DeltaStats { value, std, answered, tier }>`; `Source` stays (Dft / Surrogate / Fallback describe the *base*); the readouts show both lines.

### 7.8 Provenance and the library
- A Δ-model's identity is its two canonical keys (base library key + precise key); its file name is FNV-1a of the two texts joined by `"\n--\n"`. The model is **never installed on a scene whose base key or precise key differs** (the user-data gate, extended).
- **The tier, method and `method_rev` are in the precise key**, so an ECG-taught and a DMC-taught Δ for the same scene are different models, and a DMC algorithm change makes old labels inert under their own key ("method rev 1 → 2"), exactly as `TEACHER_REV` does for LSDA models.
- The **Model tab's** "This model" group (user-data §12) gains rows for an active Δ-model (§8.5); the manager lists label stores and Δ-models next to the LSDA library entries (§8.2).

## 8. The model manager

### 8.1 Where it lives
All numbers in this section's mocks are illustrative layout, not results. **[provisional, P1]** A full-width **Models & jobs** page that temporarily replaces the 3D view (View ▸ Models & jobs; Ctrl-K "models", "jobs", "model manager"; the toolbar's jobs chip; the Model tab's "Open model manager"). It is a page, not a floating window, so the redesign's rule holds; Esc or "Back to the scene" returns. The Model dock tab keeps the compact, this-scene view (user-data §12), and its **Library** section (user-data §11) becomes the manager's *Models* list in compact form, with the same rows and actions — one implementation, two sizes (no duplication). Three tabs: **Models**, **Jobs**, **Compare**. Every number has a tooltip saying what it is and where it comes from (the user-data §12 rule), and every tier badge has a one-sentence meaning (§8.6).

### 8.2 Models
```text
Models & jobs                                                        [Models] Jobs  Compare      ← Back to the scene
 Search… [________]   show: ☑ LSDA models  ☑ precision models  ☑ precise data      12 models · 3 label sets · 41.6 MB
 ┌────┬──────────────────────────────────────┬──────────────────┬──────────┬──────────────┬─────────────┬────────┐
 │ ●  │ model                                │ accuracy         │ trained  │ samples      │ teacher time│ size   │
 ├────┼──────────────────────────────────────┼──────────────────┼──────────┼──────────────┼─────────────┼────────┤
 │ ●  │ H O · 141³ · 0.1 Bohr · GPU 1e-6     │ [LSDA]           │ 3 h ago  │ 412 (2 excl.)│ 1 min 44 s  │ 1.8 MB │
 │ ●  │   + Δ(DMC) · sj1-tm · τ 0.005      │ [LSDA + Δ(DMC)]  │ 1 h ago  │ 96 labels    │ 4 h 12 min  │ 0.2 MB │
 │    │ H O · DMC labels (sj1-tm)          │ [precise data]   │ 1 h ago  │ 96 ± 0.6 mHa │ 4 h 12 min  │ 3.1 MB │
 │    │ H · 49³ · 0.25 Bohr · GPU 1e-6       │ [LSDA]           │ 2 d ago  │ 273          │ 41 s        │ 0.4 MB │
 │    │   + Δ(exact) · ECG                   │ [LSDA + Δ(exact)]│ 2 d ago  │ 40 (+forces) │ 31 s        │ 0.1 MB │
 └────┴──────────────────────────────────────┴──────────────────┴──────────┴──────────────┴─────────────┴────────┘
 ● answering this scene   ◐ for this scene, not chosen   ○ another scene    (amber: read-only, in use elsewhere · red: unreadable)
```
**Selecting a row** opens its detail on the right:
```text
 H O · 141³ · 0.1 Bohr · GPU 1e-6  + Δ(DMC)                                   [LSDA + Δ(DMC)]  ⓘ
 What it is      LSDA energies of water-like scenes, corrected by a model of what DMC adds.
 How good        bond energies: DMC-class (single determinant: typically a few kcal/mol too weak) where
                 trained; ± 0.4 mHa (0.25 kcal/mol) fit uncertainty on its own labels ⓘ
                 measured on water's atomization: 3.1 ± 0.3 kcal/mol from the exact value ⓘ (oracle, §9)
 Trained on      96 DMC labels (± 0.6 mHa each), 2 h 58 min pass + 1 h 14 min learning-while-playing
 Where           O–H 1.62–2.31 Bohr (96) · H–H 2.40–3.61 Bohr (96) · bursts 300 K, 600 K
                 this scene now: O–H 1.81 ✓  H–H 2.86 ✓
 Trust           trusted · χ² 1.04 (fine: 0.81–1.22) · LOO rms 1.1 σ
 Cost            4 h 12 min of GPU · one label ≈ 2.6 min · a new label for this scene ≈ 2.6 min
 Files           …\models\precise\9c1e….delta (0.2 MB) · labels …\9c1e….labels (3.1 MB)
 [Use for this scene]  [Label more…]  [Compare…]  [Export]  [Delete…]
```
- **Delete…** follows user-data §5.5 (two clicks, renamed to `.forgotten-<unix>`, never destroyed); deleting a label store says how many Δ-models are derived from it and that they go too.
- **Use for this scene** is the chooser (§8.5); disabled with the reason when the keys do not match (the `LibraryKey::diff` phrases: "this model is for species H O; the scene is H").

### 8.3 Compare
Two to four models of the **same species/charge/multiplicity**, side by side, on the current scene and on a scan the user picks:
```text
 Compare  · scene: water (3 atoms) · scan: O–H stretch 1.4–3.0 Bohr (other atoms fixed)   [scan ▾] [Run (≈ 2 s)]
 ┌──────────────────────────────┬─────────────────┬──────────────────┬──────────────────┐
 │                              │ LSDA (141³)     │ LSDA + Δ(DMC)    │ DFT Precise      │
 ├──────────────────────────────┼─────────────────┼──────────────────┼──────────────────┤
 │ energy here (binding)        │ −0.4021 Ha      │ −0.3655 ± 0.0004 │ −0.4019          │
 │ O–H bond length (scan min)   │ 1.83 Bohr       │ 1.81 ± 0.01      │ 1.83             │
 │ stretch stiffness            │ 0.51 Ha/Bohr²   │ 0.55 ± 0.02      │ 0.51             │
 │ breaking one O–H (scan end)  │ 0.214 Ha        │ 0.188 ± 0.002    │ 0.213            │
 │ cost per step                │ 6 µs            │ 9 µs             │ 8 s              │
 │ trusted here?                │ yes             │ yes (σ 0.4 mHa)  │ —                │
 └──────────────────────────────┴─────────────────┴──────────────────┴──────────────────┘
 [curve: E(R) of each, with the Δ-model's ± band, and the DMC labels as points with error bars]
```
Each row's ⓘ explains the quantity in a sentence ("how much energy it takes to pull one H off, along this scan"). Scans are cheap for models (µs per point) and are offered for DFT only with a cost line.

### 8.4 Jobs
```text
 Jobs                                     GPU: running · background jobs while playing: [pause | share● | full]
 ┌──┬────────────────────────────────┬─────────┬─────────────────────────────┬──────────┬──────────┬─────────┐
 │  │ job                            │ kind    │ progress                    │ per unit │ ETA      │ size    │
 ├──┼────────────────────────────────┼─────────┼─────────────────────────────┼──────────┼──────────┼─────────┤
 │▶ │ water, DMC, 300 K              │ render  │ ▓▓▓▓▓░░░░ 412/1000 frames   │ 11 s/fr  │ 3 h 10 m │ 210 MB  │
 │  │   └ labelling frame 412 (DMC)  │         │ ▓▓▓▓▓▓▓░ σ 0.8 → 0.6 mHa    │ 2.6 m/lb │ 40 s     │         │
 │⏸ │ H₂ exact, 1 ps                 │ render  │ ▓▓▓▓▓▓▓▓ 3900/4000          │ 0.4 s/fr │ (paused) │ 18 MB   │
 │… │ H O: label more (40)           │ label   │ queued #3                   │ 2.6 m/lb │ ≈ 1 h 44 │ —       │
 │• │ sampling pass (live worker)    │ pass    │ ▓▓▓░░ 112/273 perturbations │ 1.0 s    │ 2 min    │ —       │
 └──┴────────────────────────────────┴─────────┴─────────────────────────────┴──────────┴──────────┴─────────┘
 [Watch] [Pause] [Resume] [Cancel…] [▲ ▼ reorder] [Open log] [Delete…]       finished: 7 · 1.3 GB · [free 1.1 MB deleted]
```
- **ETA** = remaining units × the rolling median cost per unit (the M6 median rule), shown as "≈" until 3 units are done; the reference-stage and orbital-stage costs are counted separately (the user-requests "Sampling-pass ETA" lesson).
- **Cancel…** keeps every computed frame or label (two clicks); a cancelled render stays playable.
- **The toolbar chip** (next to the solver chooser): "⚙ 2 jobs · 41 % · 3 h 10 m" — click opens Jobs; a finished job raises a toast "water, DMC, 300 K finished — Watch".
- **The render form** (Simulation ▸ Precision render…):
```text
 Precision render — water (3 atoms, from the live scene at t = 412.75 fs)
 Solver      ( ) DFT at Precise (LSDA, h 0.10)   ≈ 8 s/frame
             ( ) exact (ECG)  — needs ≤ 3 electrons (this scene has 10)   [disabled, reason shown]
             (●) DMC — learned surface, refined on the fly (R2)   ≈ 11 s/frame + labels ≈ 2.6 min each
             ( ) DMC energies along an LSDA trajectory (R3)
 Length      [1000] steps × dt 0.25 fs = 250 fs        thermostat: CSVR 300 K (from the live run)  [change]
 Keep        density every [10] frames (≈ 560 MB)      check with DMC: [when uncertain + every 50 frames ▾]
 Estimate    ≈ 3 h 30 min on this GPU (≈ 40 labels) · runs in the background; the sandbox stays usable
 [Queue render]   [Cancel]
```

### 8.5 The chooser: which model answers this scene
In the **Model dock tab** (and the manager's detail): "Answering this scene:" with one radio per model that matches the scene's keys, each with its badge and one plain line:
```text
 Answering this scene
 (●) LSDA model            [LSDA]            fast; bond energies as LSDA (tens of kcal/mol off)
 ( ) LSDA + DMC correction [LSDA + Δ(DMC)]   fast; bond energies DMC-class where trained (96 labels) 
 [Label more for this scene…]  [Open model manager]
```
- The choice is per key and remembered in the index (`chosen: true`), so returning to water picks the same model. The solver chooser's **Estimated** button shows the active tier as a badge ("Estimated · LSDA + Δ(DMC)").
- Switching the answering model at run time changes the surface: booked as a surface jump (E_conserved continuous) and logged ("model switched: LSDA → LSDA + Δ(DMC) — the energy at these positions changed by +0.0366 Ha").
- The This-model group (user-data §12) gains, for an active Δ: **precise teacher** (method, tier, rev, trial grid, τ, pseudo recipe), **labels** (count, mean σ, by origin: pass / learning-while-playing / audit / render), **Δ fit** (rmse, χ², LOO, Q′, dofs), **Δ coverage** (pair ranges of the labels), **precise teacher time**.

### 8.6 Plain language (the copy is pinned in the plan, Task 2.4)
- **LSDA** — "This sandbox's everyday physics: density-functional theory with the simplest functional. Shapes are roughly right; bond energies are typically tens of kcal/mol too strong."
- **LSDA (Precise grid)** — "The same physics on the finest grid: the best LSDA can do, still with LSDA's bond-energy error."
- **exact** — "The electrons' Schrödinger equation solved to within a millionth of a Hartree. Only possible for up to 3 electrons. The nuclei are still classical."
- **DMC** — "Diffusion Monte Carlo: solves the many-electron equation by random walks. Exact except for where the wavefunction changes sign, which comes from an LSDA guess; bond energies typically a few kcal/mol too weak (water: expected about 3). Every number has a statistical ± that shrinks the longer it runs."
- **Δ (correction model)** — "A small learned model of the difference between the precise method and LSDA. It makes precision fast enough to play live, but only where it has been taught; elsewhere it says so and falls back to LSDA."
- **±** — "One standard deviation: about two times in three the true value of *this method* is inside the bar. It does not include the method's own approximation (see the badge)."
- **kcal/mol** — shown next to every Hartree difference that is a bond-energy-like quantity (1 kcal/mol = 1.594 mHa), converted at the UI edge (rule 3).

## 9. Validation (rules 6 and 7)

New `validate` suites, release-only, GPU tests skipping without an adapter; every number of record into `references.md` § "Precision". **Pre-measurement rule:** bounds marked "measured" are pinned at the measured value × a stated margin by the task that first runs them, never loosened afterwards.

| suite / test | what | bound | reference |
|---|---|---|---|
| `precision_ecg::h2_at_1_4011_matches_pachucki` | ECG H₂ at R = 1.4011 | E − E_exact ∈ [−1e-10, +1e-6] Ha | Pachucki 2010: −1.1744759314002167 |
| `precision_ecg::h2_well_depth` | D_e = E(R = 40) − E(1.4011) vs the exact | within 2e-6 Ha of 0.17447593 | derived from Pachucki 2010 and E(H) = −0.5 |
| `precision_ecg::h3plus_matches_the_ecg_record` | H₃⁺, R = 1.65 | E − E_ref ∈ [−1e-10, +1e-5] | −1.34383562502 (Pavanello & Adamowicz 2009) |
| `precision_ecg::heh_plus_matches_pachucki` | HeH⁺, R = 1.463283 | E − E_ref ∈ [−1e-10, +1e-5] | −2.978708310771 (Pachucki 2012) |
| `precision_ecg::helium_atom` | He atom | E − E_exact ∈ [−1e-10, +1e-6] | −2.903724377034 (Schwartz; Sims & Hagstrom) |
| `ecg::hf_force_is_the_frozen_basis_derivative` | analytic force (attached centres: Pulay and ion–ion terms included) vs central FD of E at the frozen basis; ΣF | ≤ 1e-7 Ha/Bohr; \|ΣF\| ≤ 1e-10 | — (consistency) |
| `ecg::force_label_sigma_is_measured` | frozen-basis force vs central FD of warm-start re-optimised energies | rms recorded → the σ_F of ECG force labels | — |
| `ecg_render_conserves_energy_across_refreshes` | 2000 steps of H₂ at 300 K, NVE, with basis refreshes booked | drift ≤ measured × 3, continuity across refreshes 1e-10 | — |
| `precision_qmc::hydrogen_atom_dmc` | DMC, 1 e⁻, point nucleus, grid trial | |E + 0.5| ≤ 3σ + box bias (measured) | exact −0.5 |
| `precision_qmc::h2_nodeless_dmc_matches_ecg` | H₂ at 1.4011, τ ladder → 0 | |E − E_ECG| ≤ 3σ, σ ≤ 0.1 mHa | the ECG tier (and Pachucki) |
| `precision_qmc::h3plus_nodeless_dmc_matches_ecg` | H₃⁺ R = 1.65 | same | same |
| `precision_qmc::error_bars_are_honest` | 40 independent DMC runs of H₂ (seeds), each E_i ± σ_i | fraction within 1σ of E_ECG + the measured τ-bias ∈ [0.49, 0.87]; within 2σ ∈ [0.87, 1.0]; χ²/40 ∈ [0.52, 1.67] (99 % binomial and χ²₄₀ intervals for 40 runs) | — (statistics) |
| `qmc_gpu::gpu_dmc_matches_cpu_reference` | the same Ψ_T, τ, H₂O, GPU vs CPU reference DMC | |ΔE| ≤ 3√(σ_g² + σ_c²) | — |
| `qmc::time_step_bias_is_measured_and_extrapolated` | H₂O, τ ∈ {0.02, 0.01, 0.005, 0.0025} | bias at the tier τ recorded; extrapolation fit residual ≤ 2σ | — |
| `qmc::t_moves_and_locality_are_both_reported` | pseudo-O atom and H₂O, T-moves vs LA | difference reported (the localisation uncertainty); the τ dependence of each reported with its sign — the T-move upper bound is relative to the exact E₀ of the pseudo-Hamiltonian, so it implies nothing about E(τ) against E(τ → 0) (review S5) | — |
| `qmc::nonlocal_quadrature_matches_the_semilocal_radial_integral` | for f = R(r)Y_lm, l ≤ 2: the sphere rule's ⟨f\|δV_l P_l\|f⟩ vs ∫δV_l R² r² dr | 1e-10 (degree 2l ≤ 4, inside the rule's exactness) | — (review M1) |
| `precision_qmc::water_atomization_is_internally_consistent` | TAE_e = E(O, DMC-scanned multiplicity) + 2E(H) − E(H₂O) at the experimental equilibrium geometry, pseudo-O (`xc hf, nlcc off`, semilocal) + AE H, T-moves | **hard gate (review S1): internal consistency** — τ → 0 extrapolation residuals ≤ 2σ, σ ≤ 0.3 kcal/mol with a detected blocking plateau, the population bias inside 2σ, and the decomposition rows (T-moves vs LA, the pseudopotential bound below, the 0.38 frozen-core term) all present | — |
| `precision_qmc::water_atomization_vs_exact` (stage-4 acceptance, **reported to the user**, not a pass/fail gate) | the same TAE_e against the exact 233.0 kcal/mol | **\|ΔTAE\| ≤ 4 kcal/mol is the stage-4 acceptance reported to the user**, pinned "measured first" (the measured value × the recorded margin becomes the regression bound); the cited studies used HF/DFT nodes with all-electron or BFD potentials, so 4 is an extrapolation (§3.3) | W4 (Karton et al. 2006), non-relativistic, ZPVE removed |
| `precision_qmc::pseudopotential_error_offline` (offline, `#[ignore]` by default: hours) | all-electron point-nucleus DMC of O, OH, H₂O (trial orbitals from a fine all-electron grid plus Ma et al. cusp correction — Ma, Towler, Drummond & Needs, *J. Chem. Phys.* 122, 224322 (2005)) vs the pseudo-O DMC | TAE difference reported as a **bound** on the pseudopotential error, not a decomposition: at h = 0.05 the all-electron O still has Zσ = 0.4 (its core is distorted) and the Ma et al. correction assumes correct orbitals outside its radius | — |
| `precision_qmc::box_bias` | H atom in 12 / 14 / 16 Bohr boxes | bias reported, monotone | — |
| `precision_delta::h3plus_from_ecg_labels` | Δ(ECG) on Accurate-LSDA H₃⁺; held-out geometries | |E − E_ECG| ≤ 3σ_Δ on ≥ 95 % of 100 held-out points; bend/stretch frequencies on the corrected surface within 2 % of the ECG surface's | the ECG tier |
| `precision_delta::delta_uncertainty_is_calibrated_on_dmc` | standardised held-out residuals of a DMC Δ-model (H₂O) | rms ∈ [0.7, 1.4] | — |
| `precision_delta::water_live_reproduces_dmc_atomization` | the live LSDA+Δ(DMC) at the oracle geometry (labels at the tier τ) | within 3σ + \|the measured τ bias at the tier τ\| of the τ → 0 DMC value of `water_atomization_is_internally_consistent` (review S6), σ the total of §7.4 | — |
| `jobs::a_dft_render_equals_live_dynamics_bitwise` | an R1 CPU-DFT render of H₂ vs `sim::Dynamics` driven directly, 20 steps | bitwise (`==`) — the same code path | — |
| `jobs::resume_after_a_killed_process_is_bitwise_for_deterministic_teachers` | kill (a real child process) and resume, Morse and ECG | frames after resume bitwise equal to an uninterrupted run | — |
| `jobs::a_torn_frame_is_truncated_and_recomputed` | a record cut mid-write | the log opens; that frame is recomputed | — |

Notes from the review: the Δ fit's s_E carries the DFT egg-box floor (§7.4), reported with each Δ-model; GPU-vs-CPU local energies are compared with a tolerance **relative to |V_en|** near nuclei (f32), not a fixed 1e-5 Ha (Task 4.5).

The existing suites are untouched (this milestone changes no LSDA solver code; the M7 recipe variant of Task 4.7 is an added option whose default is M7's, so M7's suites are unchanged); stage 4's orbital accessor is read-only (pinned bitwise like the orbital viewer's §4.4). Statistical honesty is a first-class oracle (`error_bars_are_honest`), not a footnote: an error bar that is wrong is worse than none.

## 10. Milestone placement and stages

Current order (user-data spec, M7 spec): M6 (tools) → user data → orbital viewer → M7 (pseudopotentials) → transfer/electrostatics → periodic cells. This milestone threads in as five stages, **each shipping something usable**:

| stage | ships | depends on | placement |
|---|---|---|---|
| **1 — Precision render** with the existing solvers (DFT at any Quality incl. Precise), the job runner and queue, checkpoint/resume, progressive playback | "render this at Precise overnight and watch it fill in" | user data (the root, atomic writes) | **right after user data** |
| **2 — Model manager** (Models, Jobs, Compare pages; chooser; plain-language copy) over the LSDA library and the jobs of stage 1 | every model and job in one place | stage 1, user data's library | after stage 1 |
| **3 — Exact tier (ECG)** for ≤ 3 electrons as a render teacher and a model teacher | exact H₂ / H₃⁺ / HeH⁺ dynamics; "exact" badges | stage 1 | after stage 2; independent of M7 (H and He only) |
| **4 — DMC tier** (VMC optimisation, DMC, T-moves, GPU walkers, R3 renders, label jobs) | DMC energies along trajectories; DMC labels | M7 (pseudo-O), stages 1–2; the orbital accessor | **after M7** |
| **5 — Δ-learned precision models** (label stores, Δ fit, live `DeltaKs`, R2 renders, active learning, audits) | precision played live | stage 3 (the ECG half: 5.1–5.3), stage 4 (the DMC half: 5.4–5.7) | the ECG half can follow stage 3 directly |

## 11. Risks

1. **DMC cost is underestimated** (§3.4 is [est]). Mitigation: Task 4.0 measures the GPU walker throughput and the H₂O variance before any DMC feature is built; the forms always show measured costs; Δ-learning is the path that survives a 10× miss.
2. **f32 walkers.** Determinant ratios and local energies in f32 can bias. Mitigation: f64 CPU inverse refresh, compensated sums, and `gpu_matches_cpu` gated statistically against the f64 reference.
3. **The pseudopotential's error in DMC.** M7's LSDA/NLCC potentials are unusable in QMC as they are (review M2); the HF-reference, no-NLCC variant (Task 4.7, unconditional) is bounded offline against all-electron DMC (§9). If that variant fails M7's gates for oxygen, stage 4 stops at the Task 4.0 go/no-go and reports.
4. **Single-determinant nodes from LSDA** are the dominant error and are worst at stretched bonds (and M6 found this solver stays on the spin-symmetric LSDA solution past the Coulson–Fischer point, so the base and the nodes are both restricted there). Mitigation: stated in the tier copy ("bond breaking is where this tier is weakest"); orbital optimisation and multideterminant nodes are listed for later (§12). For two-electron systems it does not matter (nodeless).
5. **Δ is not small everywhere.** Where the base is qualitatively wrong (restricted H₂ at dissociation), Δ is large; it is still smooth, but needs labels there. Mitigation: the selection is variance-driven, so the labels go where Δ is uncertain; the coverage view shows it.
6. **The box node** biases DMC for diffuse states. Mitigation: the box rule (used, never trained on), `precision_qmc::box_bias`.
7. **GPU contention** with the live run. Mitigation: chunking, the share setting, measured latency (Task 1.3).
8. **Disk growth** from densities. Mitigation: cadence and size shown before submitting; deletion renames, a footer frees.
9. **Scope.** This is the largest milestone yet. Mitigation: five independently shippable stages; stage 4 starts with a measurement spike that can stop it.

## 12. Non-goals and later items

- **Multideterminant trial functions, VMC orbital optimisation, backflow** — the route from ~3 to ~1 kcal/mol (Petruzielo et al. 2012); later, each measured.
- **Noisy-force Langevin QMC-MD (R4)** — rejected here (§4.2).
- **Path-integral quantum nuclei** — a separate later item (§4.2).
- **Periodic systems** — the DMC tier is for finite clusters; periodic QMC (finite-size corrections, twist averaging) comes with periodic cells, if ever.
- **Coupled cluster** — never as a solver (§3.2).
- **N = 4 ECG**, **excited states**, **relativistic corrections** — not now.

## 13. Provisional rulings (made here, flagged for the user)

- **P1** The model manager is a full-width page replacing the 3D view (View ▸ Models & jobs), not a dock tab or a window; the Model tab keeps the compact this-scene view, and its Library section is the manager's list in compact form (§8.1).
- **P2** Two-electron singlets are run by DMC as well as ECG; ECG is the exact tier for N ≤ 3 and the DMC engine's oracle (§3.1).
- **P3** Noisy-force Langevin DMC-MD (R4) is not built; DMC renders integrate the learned surface (R2, default) or LSDA motion with DMC energies (R3) (§4.2).
- **P4** R2's DMC check cadence defaults to "when uncertain + every 50 frames" (§4.2).
- **P5** Jobs interrupted by exit resume at the next launch by default (a setting, `jobs.resume_at_launch`, profile class "run") (§6.2).
- **P6** Background jobs *share* the GPU with the live run by default (chunked; settings pause / share / full) (§6.3).
- **P7** The live sampling pass is shown in the job queue but stays on the live worker (§6.3).
- **P8** Precise labels are stored per precise key, independent of the base surface; Δ-models are derived and rebuilt when the base changes (§7.2).
- **P9** The Δ-model's default basis is total degree 2, body order 3, and its answer bound min(1 mHa, 0.5 mHa × N_atoms) (§7.1, §7.4) — both re-decided by Task 5.2's measurements.
- **P10** Size-consistent T-moves (Casula et al. 2010) are the DMC tier's nonlocal treatment; the locality approximation is the cross-check (§5.2.1).
- **P11** All-electron point nuclei for H and He in DMC; M7 pseudopotentials for Z ≥ 3 in the QMC variant (`xc = HF`, `nlcc off`, semilocal form in DMC) (§5.2.1; review rulings M1, M2).
- **P12** Trial orbitals come from a Precise-quality (h = 0.10) LSDA solve under `CoresOnly` with the same pseudopotential variant, part of the tier key (§5.2.2).

## 14. Constants (the complete ledger; universal — none per element)

| constant | value | role | why universal |
|---|---|---|---|
| `ECG_REFRESH_STEPS`, `ECG_REFRESH_DIST` | 20 steps, 0.05 Bohr [measured in Task 3.2] | frozen-basis segments | integrator/geometry, not element |
| ECG candidate draw [provisional, re-decided by Task 3.1] | widths log-uniform in [0.1, 4]·Z² (core) and [0.05, 2]/R_min² (bonds); centres within 0.5 Bohr of a nucleus or a bond midpoint; 20 candidates per function | SVM | a fixed distribution scaled by the scene's own Z and distances |
| Jastrow orders | u: 8, χ: 8, f: 3 (each variable) | Jastrow | fixed expansion orders |
| Jastrow cutoffs | L_u = min(half the cell's inner width, 10 Bohr); L_χ = L_f = 4 Bohr (pseudo) / 3 Bohr (AE H, He); C = 3 | Jastrow | from the scene's cell; fixed exponent |
| nonlocal quadrature | 12-point icosahedral rule (exact for spherical harmonics through degree 5), random rotation per evaluation | T-moves, LA | one rule for every channel l ≤ 2 (the first and second rows need s and p projectors); an l = 3 channel needs a higher-degree rule, added with it |
| `DMC_WALKERS` | 4096 per GPU job (target) | population | population-control bias measured, not assumed |
| τ (tier) | 0.005 Ha⁻¹ (pseudo), 0.0025 (all-electron oracles) [measured in Task 4.4] | DMC | a time step |
| `DMC_REEQUIL` | 200 generations | continuation | — |
| `DET_REFRESH` | 16 sweeps | f64 inverse refresh | numerics |
| `DMC_CHUNK` | the generations fitting ~50 ms of GPU [measured] | GPU sharing | — |
| `DELTA_D`, body order | 2, 3 [P9] | Δ basis | fixed |
| `DELTA_MIN_DOF`, `DELTA_LOO_MAX` | 10, 1.5 σ | Δ trust | fixed |
| `DELTA_ANSWER_STD` | min(1 mHa, 0.5 mHa × N_atoms) [P9] | Δ answering | fixed |
| `DELTA_SE_RATIO` | s_E² ≤ 9 × median σ_i² (above: "label errors understated or the model underfits", untrusted) [provisional, Task 5.2] | Δ trust (review M4) | fixed |
| `CKPT_EVERY` | 30 s | job checkpoints | — |

Anything per element appearing in this list is a design violation (rule 1).

## 15. Model policy

Fable 5.1: the ECG solver (integrals, SVM, forces), everything in `qmc` (Jastrow and its derivatives, VMC optimisation, DMC, T-moves, estimators, error analysis), the Δ fit with heteroscedastic noise and its trust rules, active-learning selection, all oracles and the statistical-honesty tests, and physics review of R1–R3 bookkeeping. Opus 5.5: the `jobs` crate (queue, runner, files, checkpoints), the playback mode and timeline, the manager pages and chooser, the GPU walker kernels against the CPU reference (`gpu-kernel-engineer`), the library integration, README. Fable tasks may fall back to Opus only with a mandatory `physics-reviewer` pass before they land.

## 16. Self-review (done before commit)

- **Every commitment of §1 is answered:** (1) §6 (render with any solver, DFT at Precise in stage 1, checkpoints, progressive playback with scrub/pause/resume); (2) §7 (Δ-learning as the main route, then live via `DeltaKs`); (3) §8 (one place, tiers, data, coverage, cost, compare, chooser, queue with progress/ETA/pause/resume/cancel, plain language); (4) §3–§5 (the method, with fixed-node error, noisy forces and pseudopotentials stated).
- **Rule 1, paranoid pass:** no basis table (ECG parameters are SVM outputs from a universal draw scaled by the scene's Z and distances; the DMC orbitals are the solver's; the Jastrow form, orders and cutoff rule are universal and its coefficients are variational outputs per scene); the pseudopotentials are M7's; Gaussian-basis CC rejected with the reason (§3.2). The residual tension is the same as M7's: per-scene variational parameters differ between elements because their Z differs — they are outputs, never inputs, never stored as data, never fitted to experiment.
- **Rule 3:** atomic units in every file and solver; kcal/mol and eV only at the UI edge.
- **Rule 4:** ECG and every DMC reduction in f64 on the CPU; walkers f32 with f64 inverse refresh and compensated sums.
- **Rule 5:** `sim` gains only plain records (`DeltaStats`, tier text); orbitals cross only inside `providers`; the render runs `sim::Dynamics` with an opaque provider.
- **Rule 6:** new suites gate the new methods; the old suites are untouched; tripwires per precise method (`method_rev`).
- **Rule 7:** every tier states its systematic error in words; per-frame bars are statistical/posterior only and say so; classical nuclei and LSDA motion (R3) are labelled.
- **Rule 8:** dt fixed per job; playback speed is frames per second.
- **Numbers:** every literature number in this spec was checked against its source (or a source quoting it, named) on 2026-10-15, re-checked after the review (§18: what was verified where; Filippi–Umrigar's H₂ digits are marked unverified and not quoted); every cost is marked [est] with its derivation and a measuring task; the mock numbers in §8 are labelled illustrative.
- **Placeholders:** the per-molecule water rows of Nemec/Petruzielo and LSDA's own water atomization are deliberately left to the tasks that pin them from the sources.

## 17. References

- K. Pachucki, "Born-Oppenheimer potential for H₂", *Phys. Rev. A* 82, 032509 (2010), arXiv:1007.0322.
- C. Schwartz, "Further computations of the He atom ground state", arXiv:math-ph/0605018 — He: −2.90372437703411959831…; J. S. Sims & S. A. Hagstrom's Hylleraas-CI helium value (NIST) agrees.
- K. Pachucki, "Accurate Born-Oppenheimer potential for HeH⁺", *Phys. Rev. A* 85, 042511 (2012), arXiv:1203.3927.
- M. Pavanello & L. Adamowicz (2009), H₃⁺ with 1000 shifted ECGs, as tabulated in J. C. Lopez Vieyra, A. V. Turbiner & H. Medel Cobaxin, arXiv:1105.5435; W. Cencek, J. Komasa & J. Rychlewski, *Chem. Phys. Lett.* 246, 417 (1995).
- J. Mitroy et al., "Theory and application of explicitly correlated Gaussians", *Rev. Mod. Phys.* 85, 693 (2013).
- K. Varga & Y. Suzuki, *Phys. Rev. C* 52, 2885 (1995) — the stochastic variational method.
- P. J. Reynolds, D. M. Ceperley, B. J. Alder & W. A. Lester, "Fixed-node quantum Monte Carlo for molecules", *J. Chem. Phys.* 77, 5593 (1982).
- W. M. C. Foulkes, L. Mitas, R. J. Needs & G. Rajagopal, "Quantum Monte Carlo simulations of solids", *Rev. Mod. Phys.* 73, 33 (2001).
- C. J. Umrigar, M. P. Nightingale & K. J. Runge, *J. Chem. Phys.* 99, 2865 (1993).
- C. J. Umrigar, J. Toulouse, C. Filippi, S. Sorella & R. G. Hennig, *Phys. Rev. Lett.* 98, 110201 (2007) — the linear method.
- N. D. Drummond, M. D. Towler & R. J. Needs, "Jastrow correlation factor for atoms, molecules, and solids", *Phys. Rev. B* 70, 235119 (2004).
- A. Ma, M. D. Towler, N. D. Drummond & R. J. Needs, "Scheme for adding electron–nucleus cusps to Gaussian orbitals", *J. Chem. Phys.* 122, 224322 (2005).
- A. Ma, N. D. Drummond, M. D. Towler & R. J. Needs, "All-electron quantum Monte Carlo calculations for the noble gas atoms He to Xe", arXiv:0801.2736 — summarises the Z^5.5–6.5 cost estimates of D. M. Ceperley, *J. Stat. Phys.* 43, 815 (1986) and B. L. Hammond, P. J. Reynolds & W. A. Lester, *J. Chem. Phys.* 87, 1130 (1987).
- L. Mitas, E. L. Shirley & D. M. Ceperley, "Nonlocal pseudopotentials and diffusion Monte Carlo", *J. Chem. Phys.* 95, 3467 (1991).
- M. Casula, "Beyond the locality approximation in the standard diffusion Monte Carlo method", *Phys. Rev. B* 74, 161102 (2006).
- M. Casula, S. Moroni, S. Sorella & C. Filippi, "Size-consistent variational approaches to nonlocal pseudopotentials", *J. Chem. Phys.* 132, 154113 (2010).
- T. A. Anderson & C. J. Umrigar, "Nonlocal pseudopotentials and time-step errors in diffusion Monte Carlo", *J. Chem. Phys.* (2021), doi:10.1063/5.0052838.
- M. Burkatzki, C. Filippi & M. Dolg, "Energy-consistent pseudopotentials for quantum Monte Carlo calculations", *J. Chem. Phys.* 126, 234105 (2007).
- M. Nemec, M. D. Towler & R. J. Needs, "Benchmark all-electron ab initio quantum Monte Carlo calculations for small molecules", *J. Chem. Phys.* 132, 034111 (2010), arXiv:0908.2041.
- F. R. Petruzielo, J. Toulouse & C. J. Umrigar, "Approaching chemical accuracy with quantum Monte Carlo", *J. Chem. Phys.* 136, 124116 (2012), arXiv:1202.0317.
- M. Caffarel, T. Applencourt, E. Giner & A. Scemama, "Toward an improved control of the fixed-node error in quantum Monte Carlo: the case of the water molecule", arXiv:1604.02183 (quotes Klopper's −76.4389 Ha: W. Klopper, *Mol. Phys.* 99, 481 (2001)).
- R. Assaraf & M. Caffarel, "Computing forces with quantum Monte Carlo", *J. Chem. Phys.* 113, 4028 (2000); "Zero-variance zero-bias principle for observables in quantum Monte Carlo: application to forces", *J. Chem. Phys.* 119, 10536 (2003), arXiv:physics/0310035.
- C. Filippi & C. J. Umrigar, "Correlated sampling in quantum Monte Carlo: a route to forces", *Phys. Rev. B* 61, R16291 (2000).
- C. Attaccalite & S. Sorella, "Stable liquid hydrogen at high pressure by a novel ab initio molecular-dynamics calculation", *Phys. Rev. Lett.* 100, 114501 (2008), arXiv:0802.1316.
- A. Zen, Y. Luo, G. Mazzola, L. Guidoni & S. Sorella, "Ab initio molecular dynamics simulation of liquid water by quantum Monte Carlo", *J. Chem. Phys.* 142, 144111 (2015).
- J. C. Grossman & L. Mitas, "Efficient quantum Monte Carlo energies for molecular dynamics simulations", *Phys. Rev. Lett.* 94, 056403 (2005).
- C. Huang & B. M. Rubenstein, "Machine learning diffusion Monte Carlo forces", arXiv:2211.07103 — DMC energies without forces as ML training labels.
- R. Ramakrishnan, P. O. Dral, M. Rupp & O. A. von Lilienfeld, "Big data meets quantum chemistry approximations: the Δ-machine learning approach", *J. Chem. Theory Comput.* 11, 2087 (2015).
- H. Flyvbjerg & H. G. Petersen, "Error estimates on averages of correlated data", *J. Chem. Phys.* 91, 461 (1989).
- J. Tao, J. P. Perdew, V. N. Staroverov & G. E. Scuseria, "Climbing the density functional ladder", *Phys. Rev. Lett.* 91, 146401 (2003), arXiv:cond-mat/0306203 — G2/148 MAE: LSD 83.8, PBE 17.1 kcal/mol.
- A. Karton, E. Rabinovich, J. M. L. Martin & B. Ruscic, "W4 theory for computational thermochemistry", *J. Chem. Phys.* 125, 144108 (2006), arXiv:physics/0608123 — H₂O TAE breakdown, ATcT TAE₀ and ZPVE.
- Huber & Herzberg via NIST CCCBDB — H₂ ω_e = 4401.21 cm⁻¹ (already in the M6 references).

## 18. Review (physics review 2026-10-15, `docs/superpowers/notes/2026-10-15-precision-spec-review.md`; verdict "ready with fixes": stages 1–3 may start, stage 4 only with M1/M2 folded in)

| item | change |
|---|---|
| **M1** a 12-point sphere rule evaluates the semilocal form δV_l(r)P_l at the electron's radius, not the KB operator E_KB β(r)β(r′)ΣYY*; the quadrature test was ill-posed | §5.2.1: DMC uses M7's **semilocal** V_l^ion (standard QMC practice); the claim that the grid orbitals are the exact pseudo-orbitals of DMC's Hamiltonian is struck — KB on the grid vs semilocal in DMC is a trial-function difference only; §9 and Task 4.4 test the quadrature against ∫δV_l R² r² dr (degree 2l ≤ 4, exact). |
| **M2** M7 potentials are unscreened with v_xc[ρ_v + ρ_pc] and rely on NLCC; QMC has no ρ_pc | §5.2.1: an M7 recipe variant **`xc = HF`, `nlcc off`** (in `recipe_hash` and the precise key's `pseudo` line; the trial orbitals use the same variant); Task 4.7 is unconditional; the potential decision joins the Task 4.0 go/no-go; M7's spec gains a line recording the QMC variant as a future requirement; risk 3 rewritten. |
| **M3** a space-fixed frozen floating basis gives ΣF ≠ 0 and a non-rotation-invariant surface | §4.1, §5.1.4: centres attached to their nuclei, the analytic Pulay term Σ_k ∂E/∂s_k·∂s_k/∂R_I, the ion–ion term named in F_I = −cᵀ∂H/∂R_I c (cᵀSc = 1), ΣF = 0 exactly; §7.3: ECG force labels are frozen-basis derivatives with a **measured σ_F** (§9 `force_label_sigma_is_measured`; Task 3.2). |
| **M4** χ² cannot catch understated σ_i when s_E is evidence-chosen | §7.4: flag s_E² ≫ median σ_i² (`DELTA_SE_RATIO`, §14) as "label errors understated or the model underfits"; σ_i calibration is `error_bars_are_honest`'s; the overstated-noise χ² check is kept; Task 5.2's test changed accordingly. |
| **S1** the 4 kcal/mol water bound is extrapolated | §9: the hard gate is internal consistency (τ → 0, σ, plateau, decomposition rows); \|ΔTAE\| ≤ 4 is the stage-4 acceptance **reported** to the user, pinned "measured first"; §3.3 says the cited studies' nodes and potentials. |
| **S2** O reference multiplicity | §7.1: chosen by a DMC scan, not the LSDA S9 scan. |
| **S3** frames newer than `state.ckpt` | §6.2: the checkpoint records its frame and is written with every refresh/label/refit; resume truncates frames newer than it. |
| **S4** "±1 µHa" per R1 ECG frame | §6.5: the measured frozen-basis error of the current segment is shown; 1 µHa is the oracle geometries' accuracy. |
| **S5** "E_TM(τ) ≥ E_TM(τ→0) − 3σ" does not follow from Casula | §9 and Task 4.4: the τ dependence is reported with its sign; the 30 Bohr dimer trial is the product of two single-atom trials. |
| **S6** τ bias in the live water check | §9: the bound adds the measured τ bias at the tier τ. |
| **S7** He digits | §2, §9, Task 3.1: −2.903724377034 with the variational bound (Schwartz, arXiv:math-ph/0605018, checked). |
| **S8** unverified literature | Re-checked 2026-10-15: Nemec 3.2 kcal/mol and 95 % (abstract; all-electron, HF and DFT orbitals), 82 % and 40 CPU-h per electron (paper text via ar5iv); Petruzielo 3.0 / 2.1 / 1.2 (abstract, HF orbitals; BFD potentials and τ = 0.01 from the paper text); HeH⁺ −2.978708310771 at 1.463283 (Pachucki's arXiv abstract); H₃⁺ −1.34383562502 (quoted by Lopez Vieyra et al.); Ma et al.: Z^5.5–6.5 estimates, Z^5.2 earlier practical test, Z^5.47 their fit (paper text). Filippi–Umrigar's H₂ R_e digits could not be checked and are no longer quoted. |
| **Notes** | §9: the all-electron h = 0.05 comparison is a bound (Zσ = 0.4), not a decomposition; the DFT egg-box floor in s_E is reported (§7.4); the displayed band is the total √(σ_base² + σ_Δ²) and the force σ is √(∇φᵀΣ∇φ), with Δ forces never checked against DMC (§7.4); `DELTA_ANSWER_STD` = min(1 mHa, 0.5 mHa × N_atoms) states which term binds; Task 4.5's E_L tolerance is relative to \|V_en\|; the jobs "free X MB" footer never touches `models/precise/` (§6.2). |
| **Headline** | §3.3 carries the reviewer's estimate as the honesty statement: H₂O TAE_e underbound by ~3 ± 1.5 kcal/mol at the end of stage 5 (~60–65 % chance of ≤ 4); unpredictable (5–10 plausible) with M7's NLCC/KB potentials as they are. |
