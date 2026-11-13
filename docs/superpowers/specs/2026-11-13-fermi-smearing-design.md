# Fermi–Dirac occupations and the Mermin free energy (2026-11-13)

Status: design. The user approved the fix and the collection option on 2026-11-13 (section 2); everything else here
is a ruling of mine and is listed in section 13. Evidence: `docs/superpowers/notes/2026-11-13-scf-stall-diagnosis.md`
("the diagnosis" below; its section numbers are cited as D§n). Implementation starts only **after the trust rework
on branch `claude/charming-keller-0bc26b` has merged to `main`** and `scf-stall` has been rebased onto it (section 11).

## 1. Intent

The SCF stalls on small, lopsided clusters, which are what a user builds by hand. Under the app defaults 5 of 23
reproduced H clusters stall, and 6 more need a shell release and 30–55 iterations (D§2). The cause is a discontinuity
in the occupation rule (D§4). `Occupations::DegenerateShells` shares levels equally inside a window
|ε − ε_F| < τ_c = 0.1 Ha, so a self-consistent level sitting at the window edge flips the occupations between
iterations, and neither branch is self-consistent. The all-electron integer aufbau fails on a different set (H₅ ring,
H₆ seed 5, H₂O + 2 H) with the textbook level-flipping problem.

This spec replaces both rules with one continuous rule, **Fermi–Dirac occupations at a fixed electronic temperature**.
The solver then minimises and reports the **Mermin free energy F = E − TS**. The prototype (D§6) converged all 23
default-policy cases and the 3 all-electron failures, and left gapped systems unchanged (H₂ ΔE = 0, H₂O 1e-6).

It also keeps the user's trained collections. F differs from the old E for open-shell and near-degenerate samples,
so the surface key has to change. The old collection is carried onto the new key by re-solving every sample once with
the new solver, never by deleting or converting the old data.

## 2. Decisions already made by the user (2026-11-13)

1. Fermi–Dirac occupations at a fixed kT = 1e-3 Ha (≈ 316 K), with a chemical potential per spin channel. They
   replace the τ_c equal share of `DegenerateShells` and its two-phase release.
   - The solver reports and minimises F = E − TS.
   - The forces stay as they are. They are already f-weighted Hellmann–Feynman forces, i.e. −dF/dR.
   - MD, relax and the surrogate's energy labels use F.
   - The DIIS clear/push order at a release must be fixed if any release logic survives. Removing the release path
     is preferred.
2. **Collections, option (b).** The user's collections are kept. A sample whose label may change is re-solved with
   the new solver and passes a review before it is trusted; it never re-enters the fit silently. Samples whose label
   is unchanged stay.

## 3. Goals, non-goals, anti-goals

**Goals**

- **G1. Convergence.** Every staller in the diagnosis converges within a stated iteration budget (section 9, V2).
  That is 5 default-policy stallers and 3 all-electron failures, plus the all-electron lopsided tetrahedron, which
  takes 56 iterations under aufbau today. None of them may end by the stall stop or the cap.
- **G2. One rule.** One occupation rule for every scene the providers solve, all-electron and pseudised alike.
  `DegenerateShells`, its release and its "pattern frozen by rank" logic are deleted.
- **G3. One energy.** F is *the* energy: what the solver minimises, what `ForceProvider` returns, what MD conserves
  (with K), what relax minimises and what the surrogate learns. E and TS are available next to it, labelled.
- **G4. CPU = GPU.** The f64 reference (`qm`) and the GPU solver (`qm-gpu`) implement the rule with the same shared
  host code and agree within `gpu_vs_cpu`'s existing bounds.
- **G5. The collection is kept.** Unchanged samples carry over as they are. Changed samples are re-solved and reviewed
  by a rule. Nothing in the old collection folder is modified. An older build never misreads a migrated collection.
- **G6. Honest documentation.** The entropy term, the restricted-vs-broken-symmetry gap and the cost are stated
  wherever a number moves (CLAUDE.md rule 7).

**Non-goals**

- A joint chemical potential, where the spin moment floats (section 4.5).
- Gaussian or Methfessel–Paxton smearing.
- Tying T_el to the thermostat temperature.
- A Kerker preconditioner. All-electron H₂O + 2 H takes 79 iterations (D§6); that is noted here, not fixed.
- A symmetry-breaking nudge in the cold guess (section 4.7).
- Periodic cells. The roadmap's "Fermi smearing" item is done for finite boxes only.
- Migrating `.model` files inside the app (section 7.9).
- Splitting collections by spin. This was rejected with the user on the trust-rework branch.
- The radial generator's occupations (`qm::radial`, its own `Occupations` enum). They are unchanged, so the
  pseudopotential tables are unchanged.

**Anti-goals** (each one is a failure even if everything else works)

- A tolerance or bound is loosened to make a test pass (rule 6). Every moved reference gets its TS term stated in
  `references.md`.
- kT, or anything else in the rule, is per element (rule 1).
- E is reported with F's forces, or F with E's.
- Labels from the old rule and the new rule are mixed in one collection or one fit.
- The user's old collection folder is written, cut, re-locked or deleted.
- A re-solved sample whose electronic state changed enters the fit before its review.
- dt, the stall stop or the mixer is changed to make the new rule pass.

## 4. Physics

### 4.1 The occupation function

For each spin channel σ with N_σ > 0, take the k_σ solved levels ε_iσ (k_σ ≥ N_σ + 1; section 4.2):

```text
f_iσ = 1 / (1 + exp(x_iσ)),   x_iσ = (ε_iσ − μ_σ) / kT,   Σ_i f_iσ = N_σ
```

- Evaluate it in the overflow-free form: e^{−x}/(1 + e^{−x}) for x ≥ 0, and 1/(1 + e^{x}) for x < 0.
- A channel with N_σ = 0 is not solved, as today. All its f are 0 and μ_σ is undefined (reported as `None`).

**μ_σ by bisection**

- Bracket: [ε_min − 50 kT, ε_max + 50 kT]. At the lower end every f ≤ e^{−50}, so Σf < N_σ. At the upper end every
  f ≥ 1 − e^{−50}, and k_σ > N_σ gives Σf > N_σ. No other bracket logic is needed.
- Step: halve the bracket on the sign of Σf(mid) − N_σ. Stop when the midpoint equals an endpoint in f64, i.e. the
  bracket is exhausted. That takes about 60 halvings for a 20 Ha bracket (all-electron O 1s), a few µs.
- After it, |Σf − N_σ| ≲ k_σ·ulp(μ)/(4kT) ≈ 1e-13. The density is still charge-renormalised per channel by the
  mixer, as today.
- No tolerance parameter and no iteration cap are needed, and the function is deterministic: the same f64 inputs give
  the same μ on every run and in both solvers.

**Per spin, not joint (the user's decision; why).** A per-channel μ_σ at fixed N_σ is the fixed-spin-moment
ensemble.

- It keeps what `multiplicity` means today: the user's forced M, or "auto", which is the lowest M of the electron
  count's parity.
- It keeps the collections' per-sample multiplicity record meaningful.
- A joint μ would let the moment float. That is physical, but it changes the solver's electron counts, its
  normalisation and every sample's spin record, and at 316 K in LSDA it rarely moves the moment off an integer anyway.

### 4.2 Witness levels (the FD tail)

The rule needs every level with non-negligible f, so each channel grows its block until the highest solved level
satisfies:

```text
ε_kσ − μ_σ ≥ max(WITNESS_GAP, 36 kT),   WITNESS_GAP = 0.1 Ha
```

- At 36 kT, f < 2.4e-16, below f64 resolution next to 1. At 0.1 Ha = 100 kT, f ≈ 4e-44.
- WITNESS_GAP keeps today's τ_c reach, which is the configuration the prototype ran (D§6) and today's cost on the
  pseudo path. The 36 kT arm only matters if a harness raises kT above 2.8 mHa.
- Growth is one vector at a time, with μ recomputed each time. Adding a level above μ only lowers μ, so the test is
  monotone.
- The cap stays: `MAX_WITNESS_GROWTH` = 16, with the same error when it is exceeded.
- **Cost (measured in Task A1, section 11).** All-electron scenes have not grown witnesses until now. Under `Aufbau`
  they solve exactly N_σ vectors, so they now pay at least one extra vector per occupied channel. The default policy
  (`Accepted`, pseudised H) already pays it. The change in all-electron per-solve time is measured on H₂ and H₃⁺ at
  49³ and reported in `references.md` and in the README's performance numbers.

### 4.3 Energies

- **E**, the internal energy, is today's Kohn–Sham energy with f-weights:
  E = T_s + E_ext (incl. E_NL) + E_H + E_xc + E_nuc.
  - T_s = Σ f_iσ ⟨ψ|−½∇²|ψ⟩, E_NL = Σ f_iσ Σ E_KB c², and ρ_σ = Σ f_iσ |ψ_iσ|².
  - Every one of these is already f-weighted under `DegenerateShells`, so nothing changes here.
- **S**, the electronic entropy (dimensionless; k_B = 1), is S = Σ_σ Σ_i s(f_iσ) with
  s(f) = −[f ln f + (1−f) ln(1−f)].
  - Evaluate it from x, not from f: s = ln(1 + e^{−|x|}) + |x|·f_<, where f_< = 1/(1 + e^{|x|}) = min(f, 1−f), using
    `ln_1p`.
  - This form has the 0·ln 0 = 0 limit built in. Both terms go to 0 as |x| → ∞, and it has no cancellation at f → 0
    or f → 1.
  - Sum with Neumaier compensation (rule 4).
- **F = E − kT·S**, the Mermin free energy. It is the reported total.
- **The per-iteration Harris–Foulkes energy** (`ScfProgress::energy`, shown in the Activity tab) becomes
  Σ f ε + double counting − kT·S of that iteration's occupations, i.e. F_HF.

**Reference values (exact degeneracy, from the formula, not tabulated):**

| case | occupation | TS at kT = 1e-3 Ha |
|---|---|---|
| s(½) | — | ln 2 = 0.693147 |
| s(⅓) = s(⅔) | — | 0.636514 |
| H₄ square | two half-filled pairs | 4 kT ln 2 = 2.7726e-3 Ha (measured 2.77e-3, D§6) |
| H₅ ring | one half-filled ↓ pair | 2 kT ln 2 = 1.3863e-3 Ha |
| linear OH | ↓ π pair holds 1 | 2 kT ln 2 = 1.3863e-3 Ha |
| lone O, M = 3 | ↓ 2p holds 1 of 3 | 3 kT s(⅓) = 1.9095e-3 Ha |
| lone O, M = 1 | ↑ and ↓ each hold 2 of 3 | 3.8191e-3 Ha |

The lone-O rows hold only if the symmetric ensemble survives at this kT; section 4.7(d) says why that is not certain.

**Gapped systems.** With a gap of ≥ 37 kT on each side of μ, every f is exactly 0.0 or 1.0 in f64 and S ≤ 1e-14.
So TS = 0 on every closed shell, every integer-per-spin open shell, and 4 of the 5 stallers (D§6: TS ≤ 2e-7).

**Not computed: the kT → 0 estimate.** Gillan's E₀ ≈ ½(E + F) is exact to O(T²) for a smooth density of states. It is
off by TS/2 exactly in the degenerate case, the one where it would matter, so it is not reported.

### 4.4 Forces: −dF/dR, unchanged code

F[{ψ}, {f}] at fixed N_σ is stationary in both ψ and f:
- ∂F/∂ψ = 0 gives the KS equations.
- ∂F/∂f_iσ = ε_iσ − μ_σ + kT ln(f/(1−f)) = 0 is exactly the FD condition, with μ_σ the Lagrange multiplier of
  Σf = N_σ.

So dF/dR = ∂F/∂R|explicit. The occupation derivatives drop out (Σ_i μ_σ df_i/dR = μ_σ dN_σ/dR = 0), which leaves the
Hellmann–Feynman force with f-weighted density and projector sums. The existing code computes exactly that
(`crates/qm/src/forces.rs`, local term and nonlocal projector sum; `crates/qm-gpu/src/forces.rs`). It has been
f-weighted since M7, so **no force code changes.**

What changes is what the force is the derivative of. Paired with E, these forces would be inconsistent by
d(TS)/dR. That is 0 in gapped systems and up to a few mHa/Bohr where a level crosses μ. Hence G3.

### 4.5 Spin: fixed multiplicity and "auto"

- **Forced M.** The scene's `multiplicity` gives N↑ and N↓ (`qm::scf::spin_split`, unchanged). The FD rule runs per
  channel at those counts.
- **Auto** (`multiplicity: None`). This is the lowest M of the electron count's parity, as today, with no scan in a
  scene solve.
  - With FD this is a spin-restricted-count ensemble. A diradical solved at auto (O₂ at M = 1, singlet NH) shares its
    half-filled degenerate pair per spin, as `DegenerateShells` did, and now carries −TS.
  - The true triplet is lower. That is the existing, documented symmetry dilemma, unchanged.
- **The isolated-atom multiplicity scan** (`sim::references`, ruling S9) solves every allowed M and keeps the
  **lowest F**. Under the new rule F is the variational quantity at T_el, and it is the energy every sample's fit target
  is rebased against, so the scan has to compare F too.
- **Collections admit only auto-multiplicity samples**, as today. The re-solve (section 7.3) also runs at auto and must
  reproduce the sample's recorded multiplicity.

### 4.6 Where kT lives

- `qm::scf::ELECTRON_KT = 1e-3` (Ha), a solver constant in atomic units. It is not per element, not in any table, and
  the same for every scene.
- `ScfSettings.occupations = Occupations::FermiDirac { kt }`. The constant goes in through `ScfSettings::for_grid` and
  `providers::ks::problem`.
- **Not exposed in the app, and no dev flag.** Harnesses and `validate` measurements set `kt` through `ScfSettings`
  directly.
  - A user-visible T_el would move the surface (and the collection key) with a UI control.
  - A dev flag would need worker plumbing, recording and a provenance path for something no current measurement needs
    inside the app.
  - The provenance tag carries kT (section 6.2), so if a flag is added later, data at another kT can never mix.
- **Display.** T_el = kT/k_B = 315.8 K. It is converted only at the UI boundary through `units` (rule 3).

### 4.7 What the new state is (rule 7)

- **(a) Gapped systems.** F = E bit for bit or within the SCF tolerance: H₂ 0, H₂O 1e-6 (D§6). Nothing physical
  changes.
- **(b) Exactly degenerate partly filled shells** (symmetric H₄, the H₅ ring's ↓ pair, linear OH's π, a symmetric
  atom).
  - FD gives the same equal-weight ensemble `DegenerateShells` gave, so E is unchanged and F = E − TS (table above).
  - The lattice-split pairs (≤ 5.3e-5 Ha) get f within 1.3 % of equal instead of exactly equal.
  - The pseudopotential spec's §5.7 argument (levels this close are thermally co-occupied) is now the rule itself,
    made continuous.
- **(c) Near-degenerate frontier ("pinned") states.** F lies below E by TS, which is O(kT·N_pinned). This is a real
  T_el = 316 K effect, not an error: the system has an electronic temperature. It biases nothing in gapped systems.
- **(d) Open-shell atoms below the polarisation threshold: an open question gated by Task A1.**
  - The pseudopotential spec §5.7 says the symmetric FD state is unstable to orbital polarisation below ≈ 3 mHa. Its
    stated reason is that integer occupation of an O 2p set splits it by 11.6–11.8 mHa.
  - A linear estimate of the gain of the occupation mode at a symmetric shell is f(1−f)·U/kT, with U ≈ 11.8 mHa the
    splitting per unit occupation difference. For M = 3's minority p (f = ⅓) at kT = 1e-3 that is ≈ 2.6 > 1, so the
    symmetric ensemble is probably a saddle of F, not a minimum.
  - DIIS can still converge onto it: the diagnosis's lopsided H₄ converged to its spin-symmetric stationary state.
    But at a low-symmetry lattice site, where the cold guess is split by ~1.5e-4 Ha, it may polarise instead.
  - That is the round-1 path-dependence §5.7's two phases were built against (a lone O 4.1 mHa lower at
    (0, h/2, h/2)). On the M = 3 integer branch, it would also bring back the orientation zero mode (Δρ stalled at
    7e-5).
  - The diagnosis did not test lone open-shell atoms. **Task A1 measures this before any production code, and stops
    for a user ruling if it fails** (section 9, V0).
- **(e) Broken symmetry is not addressed.** Smearing converges to the nearest stationary state, not the lowest.
  - The lopsided H₄ tetrahedron (D§6, pseudo-H, 81³ × 0.15) converges to a spin-symmetric state with its frontier pair
    pinned at μ (f = 0.57 / 0.43): F = −1.932837 Ha.
  - Unsmeared runs reach spin-broken integer states at E = −1.944984 (α = 0.1) and −1.956852 (α = 0.05), 12 and
    24 mHa lower.
  - All-electron: F = −1.813947 against the integer −1.826310 (12.4 mHa).
  - This is LSDA's static-correlation regime. A spin-symmetric cold guess (ρ↑ = ρ↓ when N↑ = N↓) stays symmetric.
  - Documented in `references.md`, not hidden. A symmetry-breaking nudge is a separate decision (non-goal).
- **(f) What MD conserves.** BO-MD on F conserves F + K_nuc. E + K_nuc is not conserved where TS varies along the path,
  and the readouts say so (section 6.6).
- **(g) A new failure mode, stated.** The occupation-pattern warning (`EvalStats::occupation_changed`) goes away,
  because F(R) is smooth in the levels.
  - What smoothness does not prevent is a warm-started trajectory crossing from one stationary state to another, such
    as the restricted fractional and broken-symmetry states of (e). F then jumps.
  - Occupations do not flag this. It shows as NVE drift, and in a collection as a label the leave-one-out machinery
    cannot fit.
  - Documented as a limitation.

## 5. Solver changes (`qm`, `qm-gpu`)

**Goes**

- `Occupations::DegenerateShells`.
- `degenerate_shell_occupations`, `released_occupations`, `decide`, `shared_runs`, `FractionalShell`.
- `DEGENERACY_TAU`, and `CANDIDATE_TAU` (renamed `WITNESS_GAP`, same value, new meaning; section 4.2).
- `ScfStats.released`, the phase-2 restart of the stall watch, and the "pattern frozen by rank" state in both loops.
- **The release path is removed, so the DIIS clear-then-push defect at a release (D§3) disappears with it (the user's
  preference).**
  - The other DIIS reset, when frozen orbitals are re-solved (`qm-gpu/src/scf.rs`, "a frozen orbital set re-solved
    changes the map"), is unaffected and stays. Its current entry's residual is already measured on the new map, so
    clearing first and then pushing it is correct.
  - A test pins that no code path clears the history and then pushes an entry measured on the old map. It drives a
    synthetic map change and checks that the first post-reset mix depends only on new-map residuals.

**Comes**

- `Occupations::FermiDirac { kt }`, and `pub fn fermi_dirac(evals, n, kt) -> FdOccupations { f, mu, entropy }` in
  `qm::scf`.
  - It is host f64 code, used by the CPU loop and imported by `qm-gpu`, just as `qm-gpu` imports
    `degenerate_shell_occupations` today. CPU and GPU therefore share every operation after the eigenvalues
    (G4: the eigenvalues come from the f64 Rayleigh–Ritz on the host in both).
- **`Energy`** gains `internal` (E) and `ts` (kT·S ≥ 0).
  - `total` becomes F = `internal` − `ts`, so every consumer of `total` (providers, MD, relax, surrogate) gets F with
    no further change.
  - Invariant tested: `total == internal − ts` exactly, and the components sum to `internal`.
- **`ScfStats`** gains `entropy` (S) and `mu: [Option<f64>; 2]`. `fractional` becomes `partial: Vec<PartialLevel>`
  with `{ spin, index, eps, f }` for every level with min(f, 1−f) ≥ `PARTIAL_OCCUPATION_MIN` = 1e-6.
  - 1e-6 corresponds to |ε − μ| ≤ 13.8 kT, so "no partial level" means a gap of at least 28 mHa around μ in every
    channel. Section 6.4 relies on that.
- **`Aufbau`** stays in `qm` only as the kT → 0 reference. One equivalence test uses it: FD and Aufbau agree on H₂ and
  H₂O within the SCF tolerance. Nothing in `providers` can select it.
  - `ScfSettings::for_grid` defaults to `FermiDirac { kt: ELECTRON_KT }`, so the oracles test the path the app runs.

**Unchanged**

- The mixer (Pulay/DIIS on stacked spin densities, α = 0.3, m = 6, clamp, per-channel renormalisation).
- Eigen tolerance tied to Δρ.
- The stall stop (W = 50, level 1e-3 per electron). It is still checked after the convergence test, still a safety net.
- The convergence criterion (Δρ < tol on a tight eigen step).
- The cold guess (ρ↑ : ρ↓ = N↑ : N↓). Occupations are recomputed from each step's levels; they are not mixed.

**Warm start.** `KsState.occupations` records the scheme including kT, and the existing guard requires the same scheme.
A state converged at another kT (a harness) never seeds a solve.

## 6. Providers and consumers

### 6.1 `providers::ks::problem`

Every scene gets `FermiDirac { kt: ELECTRON_KT }`, whatever the ionic policy, so G2 holds.

- The module doc's "nothing pseudised is exactly today's problem, the same bits" no longer holds: all-electron scenes
  now solve witness vectors and are no longer bitwise equal to the old aufbau.
- `gpu_none_policy_is_bitwise_today` and the stall-stop bitwise pins are re-pinned, each with the reason recorded in
  the test and in `references.md`. They are not deleted.

### 6.2 The provenance tag (the surface)

| surface | old tag | new tag |
|---|---|---|
| pseudised | `<word>/<surface_hash>/shared1e-3/1e-1/stop2x<floor>` | `<word>/<surface_hash>/fd1e-3/stop2x<floor>` (`fd{kT:e}`) |
| all-electron | `none` | `none/fd1e-3` |

- The all-electron tag must change too: aufbau labels of near-degenerate all-electron samples (stretched bonds, open
  shells) move.
- Every comparison with the literal `"none"` that means "the all-electron surface" goes through one predicate,
  `providers::pseudo::is_all_electron_tag`. About 50 literal uses exist, many of them the scene's `ionic none` policy
  word, which is unrelated and unchanged.
- `SurfaceKey::slug` needs no special case: `none/fd1e-3` hashes like any other tag (`none-<12 hex>`).
- **The dimer gate's procedure** (`qm::radial::ledger::DIMER_PROCEDURE`) runs 3-D dimer solves. It uses
  `DegenerateShells` on the pseudised dimer and `Aufbau` on the all-electron one, and both become FD.
  - Its identity is "every constant of the gate's 3-D computation, hashed mechanically" (M7 Task 11, review SHOULD-2),
    so the occupation rule and kT are added to it.
  - `recipe_hash` then moves for Z ≤ `dimer_z_max`, and H and He tables regenerate once on first use (the H gate takes
    about 1 min on the GPU).
  - The generation-records test must show the regenerated tables are bitwise identical: the radial generator does not
    depend on the 3-D rule. It must also show the gate's verdicts and pinned R_e/ω reproduce within the backend margin.
  - `core_recipe_hash`, and so the `CoresOnly` surface hash, is unchanged.
- **`legacy_tag_of(policy)`** reproduces the tag the last `DegenerateShells` build computes, including the
  pre-change surface hash for each policy and dimer backend. Its only use is to find a predecessor collection
  (section 7.2).
  - It is pinned by a test against strings recorded from the last DS commit for every policy and both dimer backends.

### 6.3 `EvalStats` and the occupation warning

- `EvalStats.fractional` (`sim::SharedShell`) becomes the per-level list (`sim::PartialLevel { spin, index, eps, f }`).
  `EvalStats.energy` (`EnergyBreakdown`) gains `internal` and `ts`.
- `EvalStats::occupation_changed`, its watch in `providers::ks` and the surrogate's refusal to train on such a step are
  **retired**. They existed because E(R) jumped when a DS pattern switched (pseudo spec §5.7, "Molecules"), and F(R)
  does not jump. Section 4.7(g) states what replaces that guard.

### 6.4 The lone-atom composition rule (`providers::surrogate`, lone atoms)

`closed_spin_channels(z, all_electron, fractional)` asks whether an isolated atom's reference ends at a gap in each
spin channel, so that k ≥ 2 far-apart copies compose to k·E_ref. It becomes: the species has one allowed multiplicity
(Z = 1), **or** the reference's winning solve has no partial level (`ScfStats.partial` empty), on either surface.

- The `all_electron` argument goes. FD exposes an all-electron open shell as partial occupations, where aufbau hid it.
- "No partial level" means a gap of at least 28 mHa around μ. That is larger than the O self-polarisation gap
  (11.6–11.8 mHa) plus the largest box splitting of two copies' levels (1.8 mHa).
- A polarised-integer open shell (the outcome of section 4.7(d) if it happens) shows partial levels at that gap, so it
  is declined, which is the safe side.
- `AtomReference::fractional` is recorded from `partial` as before.

### 6.5 Reference scan, MD, relax, surrogate

- The scan keeps the lowest F (section 4.5).
- `ForceProvider` returns F (through `Energy.total`).
- The integrator, FIRE and the quench need no change.
- The surrogate's teacher labels, audits and teacher gap are all F.
- Hirshfeld charges and bonds read the density, which is f-weighted as before.

### 6.6 What the user sees

| where | today | new |
|---|---|---|
| Readouts row | "E_KS" / "E_KS (valence, pseudopotential)" | "F (free energy, T_el 316 K)" / "F (valence, pseudopotential, T_el 316 K)" |
| Readouts tooltip | — | "Kohn–Sham free energy F = E − T_el·S at the electronic temperature T_el = 316 K (kT = 1 mHa). For a gapped molecule TS = 0 and F = E. Partly occupied levels (open shells, nearly degenerate frontier levels) carry TS > 0. The forces, the dynamics and the trained model all use F." |
| "E total" row and the graph | KE + E_KS | KE + F, so "E total" reads "KE + F: conserved in NVE without tools" |
| Energy breakdown | kinetic … nuclear, total | kinetic, external, Hartree, xc, nuclear, **E (internal)**, **−TS (electronic entropy, T_el 316 K)**, **F** |
| Activity iteration energy | E_HF | F_HF (same row, renamed) |
| Solve log line | — | "−TS = 1.9 mHa", only when TS ≥ 1e-6 Ha |
| Orbital viewer | shared-shell rows | per-level f; a level with f < 1e-6 shows as empty; the `DegenerateShells` wording in `ui/dock/orbitals.rs` is rewritten |
| Compare with reference (PySCF) | forces, dipole | when TS ≥ 1e-6 Ha: "ours is at T_el = 316 K with partly occupied levels; PySCF's is integer-occupied", next to the force deviations |

## 7. Collection migration (option b)

### 7.1 Why the key changes, and how the collection is still kept

A collection is "one append-only store of teacher answers per solver surface" (`surrogate::collection`), and the
diagnosis is right that the surface changed. An unchanged tag would let an old build append old-rule labels to a
collection a new build has written F labels into, or the reverse. The tag therefore changes (section 6.2), the slug
changes with it, and the new surface's collection lives in a **new folder**.

"Keeping the collection" means **carrying its samples onto the new key, re-solved by the new teacher and checked**.
It does not mean reusing the old folder:

- **The old folder is never written.** It is opened by a new strictly read-only open, `Collection::open_read_only`:
  no `.lock` taken or taken over, no tail cut, no history or reference rewrite.
  - A damaged tail is read up to the last whole record and reported.
  - If the old folder is held by a live process (an older build running), the migration refuses until it closes. The
    message says why: it would be reading a store that is still being appended to.
- Older builds keep using the old folder exactly as it was at migration time. They compute the legacy tag, hence the
  old slug, and never reach the new folder.

### 7.2 Finding the predecessor

The new build attaches a surface S_new (functional, h, σ, tol, teacher, new tag). The predecessor of S_new is the folder
whose key equals S_new with the ionic tag replaced by `legacy_tag_of(policy)` (`"none"` for the all-electron surface).
It is matched exactly against `key.txt`, as every open already is. There is one predecessor per surface and no fuzzy
matching.

### 7.3 The re-solve pass

**When.** When S_new is attached and has a predecessor with samples not yet migrated, the Model tab shows an offer
(wording in section 7.8). One click starts it. "Later" leaves the offer in the Model tab's collection section.

- It does not start by itself: it takes GPU time on the order of an hour.
- Nothing else waits on it. Before and during the pass, the new collection takes live samples as usual.

**How.** It is a pass in the existing pass machinery (`providers::sampling`), with that machinery's rules for sharing
the GPU with the live scene, Pause/Resume/Cancel, and an Activity progress line. For each sample of the old folder in
file order:

1. Solve it **cold** with S_new's teacher, using:
   - the key's h, σ and tolerance (the sample tolerance, 1e-6, and the GPU stop rule as for any teacher answer);
   - the sample's own box (`grid_n`) and its stored positions, bit for bit;
   - multiplicity auto.
   - Cold, because a cold GPU solve is bit-reproducible (`sim::references`), so results do not depend on order and a
     resume reproduces them exactly.
2. Check the solve. It must have converged (not stalled, not capped), carry no solver warning (core overlap, LSDA
   domain, Poisson cap), have a face density within the box rule (`box_affected_face_density(h)`), and reproduce the
   recorded multiplicity when one is recorded.
   - A sample that fails is **not migrated**. It is listed with its reason in `migration.txt` and stays in the old
     folder.
   - Its old label is never carried: it belongs to the old surface.
3. Classify it (section 7.4) and append it to the new folder (section 7.5).

**References.** Every (species, box) in the old `references.txt` is re-solved by the multiplicity scan (lowest F;
section 4.5) and offered to the new collection through the existing `offer_reference` rules (box rule, odd n). Old and
new values go into `migration.txt`. References are few and cheap: about 13–15 s per O multiplicity on the GPU
(`references.md`, sampling-pass cost).

**Resume.** `migration.txt` holds a cursor (the next old sample index), rewritten through a temporary file after each
sample's append.

- After a crash between the append and the cursor write, the sample at the cursor is solved again. The cold solve is
  bit-reproducible, so the append is refused as a duplicate (`REFUSE_DUPLICATE`), and that counts as done.
- The pass resumes on the next attach or app start. It is never re-offered once complete.

**Cost.** One cold teacher solve per sample, the same order as teaching it originally: about 1–3 s per sample on the
RX 7900 XTX at the app's grids (`references.md`: H₂O 1.1 s cold at 57³, more at 81³). The user's H/N/O collection
(≈ 1 700–2 500 samples) needs about 1–2 h. Before starting, the offer shows a count and an estimate. The estimate is
updated from the measured median after the first 20 solves.

**Why re-solve everything (the classification criterion).** Nothing the collection stores tells an unchanged label
from a changed one. A sample keeps Z, positions, energy, forces, the box, the multiplicity, the tag, the weight, the
mask and the scene name. It keeps no eigenvalues, no gap and no occupations. Whether F = E at a geometry depends on its
frontier gap, which only a solve gives. Element or electron-count heuristics would be wrong both ways:
- a stretched closed shell can change;
- an odd-electron radical such as H₃O usually does not.

So the criterion is a comparison with a fresh solve, and its cost is stated above.

### 7.4 Classification

Thresholds: ε_E = 1e-5 Ha per sample, and ε_F = 1.5e-4 Ha/Bohr per force component.

| class | condition | meaning |
|---|---|---|
| **U**, unchanged | \|F_new − E_old\| ≤ ε_E, and every force component within ε_F | a gapped state, the same label within solver noise |
| **T**, entropy only | not U, \|E_new − E_old\| ≤ ε_E (the internal energy), every force within ε_F | the same electronic state (an equal-share ensemble before and the same f now); only the reported quantity changed, by −TS |
| **S**, state changed | anything else | the old rule had a different state here: e.g. a released integer pattern where FD now pins the frontier at μ (H₆ seed 5: E −3.148291 before, F −3.148504 now), or a different SCF branch |

- **ε_E** is 10× the measured per-solve path scatter of a converged GPU energy (6e-7 Ha, `references.md` "Dynamics")
  and 100× below the model's 1 mHa/atom energy bound.
- **ε_F** is 3 × `PSEUDO_FORCE_NOISE` (5e-5 Ha/Bohr, the teacher's measured path scatter), the tier an audit already
  judges against.
- **Both errors are safe.** A false "changed" costs one review. A false "unchanged" keeps a label within 1e-5 Ha of the
  new one.
- **Masked samples** (capped segments) are compared on all atoms, which is conservative. Their mask and
  energy-not-fitted flag are carried over.
- `migration.txt` reports the largest |ΔE| and |ΔF| seen in class U, so the thresholds' margin is visible on real
  data.

### 7.5 What is written to the new folder

- **U.** The old record is appended as it was: energy and force bits, tag, weight, flags, mask, `grid_n`, multiplicity
  and scene name. Its old quarantine code is carried as a status record when it is not 0. It is the same answer, so it
  inherits its trust state.
- **T.** The re-solved label (F_new, forces_new) is appended with the old record's metadata, and the old code is
  carried. The review is the T test itself:
  - the internal energy and every force agree with an answer the old trust machinery had already judged;
  - the label moved by exactly the defined −TS term, which is smooth across geometries (a constant 1.386 mHa for every
    linear OH, say).
  - The refit after migration judges it like any fitted sample.
- **S.** The re-solved label is appended with the old metadata and status code **8, `Q_RESOLVED`** ("re-solved for the
  new solver, awaiting review"). Two exceptions keep their old code: a forgotten sample keeps code 7, because the
  user's decision outranks the review, and a superseded one keeps code 2, because what superseded it is migrated too.
  An old hold (code 1) does not survive into class S: it judged the old label, so the sample goes to review instead.
- **`references.txt`**: the re-solved references only (section 7.3). Old references are never carried.
- **`history.bin`**: not carried. The audit history measured the model against the old teacher, and a calibration
  built on it would mix surfaces.
  - The new folder starts an empty history.
  - Without evidence, the calibrated class verdict falls back to its no-evidence rule (`calibrate` module doc), so the
    first refits are more conservative until audits accumulate. The summary says so.
- **`bond_experience.txt`** (display only, safe to delete): copied unchanged.
- **`README.txt`**: regenerated, with the migration paragraph (section 7.8).
- **`migration.txt`** (new). It holds:
  - the source slug and its `key.txt`, the build version and the date;
  - the cursor and the state (running / complete / reviewed);
  - the counts per class;
  - every not-migrated sample, with its old index and reason;
  - the reference table (old and new values);
  - class U's largest deviations;
  - the review log (section 7.6).

The new folder can already hold samples taught before or during the migration. Migrated records are appended after
them, interleaved in arrival order. Status records name samples by their index in the new file, as today.

### 7.6 Code 8 and its review: automatic, by the leave-one-out rule

A `Q_RESOLVED` sample is out of the fit, the envelope, the grade, the leave-one-out and the class statistics, like a
held sample.

- It is **not** counted against the trust recovery's 10 % held cap: it is not a misfit hold.
- It is **not** the user's: Forget and Restore do not apply to it.
- **Before its review it does not fence.** The fit simply has no data there. Old labels never enter the new folder, so
  the extrapolation grade and novelty rules already send nearby geometries to DFT.

**The review** runs inside a refit once the migration pass is complete, so every sibling of a sample is present before
any is judged. It repeats at every later refit while code-8 samples remain and the store has changed since the last
review.

1. Fit a candidate on the current fit's members **plus every code-8 sample**.
2. For each code-8 sample, compute its **leave-one-out ratio in that candidate**: the reading the trust recovery
   already ranks offenders by (`quarantine::loo_ratio` today): how far the fit without the sample misses it, against
   its own energy and force bounds.
3. A sample with ratio ≤ 1 is **released** (8 → 0). The refit then proceeds with the released samples as ordinary
   fitted samples, and the ordinary trust machinery (residuals, recovery, misfit) judges the result as it judges any
   new data.
4. A sample with ratio > 1 stays at 8. From then on it **is treated as a stored label the model cannot reproduce**:
   - it fences like a held sample, or, after the trust rework merges, is refused near like a member of its misfit
     set, whichever mechanism `main` then uses for "a stored label the model cannot reproduce";
   - a later review can still release it;
   - it is never converted to code 1, so it never enters the recovery's holds or its cap. The trust rework's "no
     ratchet" invariant (G2 of its spec) stays intact.

**Why this rule.**
- *Automatic and rule-based, as the user prefers.* It uses the trust machinery's own reading, the leave-one-out ratio,
  against the policy's own bounds, with nothing new to calibrate.
- *Candidate fit with siblings, not the plain release-by-reproduction of a held sample.* When a whole region changed
  together, such as every near-linear OH or every auto-multiplicity O₂, a fit without any of them cannot reproduce
  any of them, so release-by-reproduction would never fire. With the siblings in the candidate, each is judged by
  whether its neighbours predict it, which is the same question the class rule asks.
- A set of mutually consistent but wrong re-solves would pass. A wrong re-solve here means a teacher pathology, and the
  solver checks in section 7.3 step 2 guard against that. The released samples then face every check any fitted
  sample faces.

**Cost.** One extra fit plus a leave-one-out pass per refit while code-8 samples remain. That is about 1 s at
1 700 samples (`references.md`, leave-one-out recovery ranking). There is none once all are reviewed.

### 7.7 Formats and older builds

- **Collection records: version 1 → 2** (`surrogate::collection::RECORD_VERSION`, the `collection` entry of
  `app::update::formats`).
  - Every record the new build writes is version 2. That changes the meaning of `energy` (F on an FD surface) and
    allows status code 8. The byte layout is unchanged.
  - The reader takes 1 and 2. It never writes into a folder holding version-1 records, because every pre-existing
    folder is a predecessor (section 7.1), so no file mixes versions.
  - What an older build does:
    - it never computes the new key, so it never opens the new folder in normal use;
    - if pointed at it, `Collection::open` refuses the version-2 record with "written by a newer build" and leaves the
      file untouched (existing behaviour);
    - the downgrade dialog lists the folder as "can't open (newer format)", which is true.
- **Model files: the sidecar version is bumped** (one more entry in `surrogate::sidecar::MAGICS`).
  - The layout is unchanged, but the meaning changes: energies are F and the provenance tag format is new.
  - An older build already refuses such a file by its tag. The bump makes the refusal a clean "newer format" and puts
    `.model` files in the downgrade warnings.
  - Code 8 is never written into a sidecar: a sidecar writer leaves code-8 samples out, because the collection is the
    store of record. So `is_persisted_code` admits 8 for collections only.
- **`history.bin` (`ASAUDIT1`): unchanged.** Same layout, same meaning: the model against the teacher of the folder's
  own surface.
- **`pseudo_cache`: no format change.** H/He entries are re-keyed by the moved `recipe_hash` (a cache, rebuilt).
- **`checkpoint`** (`training/pass.ckpt`): unchanged format. An interrupted old pass carries the old provenance, so its
  resume is refused with the reason ("started with the previous solver"). The file is left as it is.
- **The downgrade warning** (`app::update::formats::downgrade_warnings`) gains one line. For each new folder whose
  `migration.txt` names a source folder that still exists:
  "<version> will use <old folder>, the collection as it was before it was re-solved for the Fermi–Dirac solver, without
  what was learned since. Both folders stay on disk."
- **`surrogate::quarantine` docs** gain "an older build meeting code 8", in the style of code 7's paragraph. Its
  content is short: no older build can read a version-2 record, and no sidecar holds 8.

### 7.8 What the user sees

**The offer** (Model tab, collection section), filled from the predecessor's counts:

> **This collection was taught with the previous solver.** Atomic Sandbox now fills electron levels with Fermi–Dirac
> smearing at 316 K, which fixes SCF stalls on loose clusters. Your collection "H, N, O · 2 412 samples"
> (models/collections/<old>) was taught with the old rule. Its samples need one re-solve with the new solver before the
> model uses them. Most will come back identical within solver noise. Open-shell and nearly degenerate ones change by
> up to a few mHa; those are checked against the rest before they rejoin the fit. About 1 h 30 min of GPU time. You
> can pause it and keep working. The original folder is kept unchanged, and older versions keep using it.
> [Re-solve now] [Later]

**Progress (Activity tab):**

> Re-solving for the new solver: sample 412 of 2 412 · 380 identical · 22 changed only by −TS · 10 changed state ·
> 0 not migrated · ≈ 52 min left

**Summary** (Model tab, and the README.txt paragraph):

> Migrated from <old slug> on 2026-11-20:
> - 2 302 identical within solver noise (kept as they were);
> - 64 changed only by the electronic-entropy term −TS (re-labelled, kept in the fit);
> - 41 changed state (re-solved and reviewed: 35 rejoined the fit because the model without each predicts it within
>   its bound; 6 are kept out, and the model falls back to DFT near them);
> - 5 not migrated (2 stalled, 3 now box-affected; they remain in the old collection).
>
> References re-solved: O −TS = 1.9 mHa (or as measured). The audit history starts empty, so trust is rebuilt from new
> audits.

The wording says "identical within solver noise", not "identical". It says "kept out", not "discarded". It names the
old folder.

### 7.9 Model files

- **The shipped models** (`models/hydrogen.model`, `h2o.model`, `h3+.model`, `SiH4.model`) carry old tags, so the new
  build would refuse them.
  - They are re-made by a headless `#[ignore]`d harness that runs the same engine: re-solve, classify U/T/S, review in
    a refit, write.
  - A re-made model must pass the trust rule it passed before. If one does not, it is re-taught from scratch with the
    sampling pass, and the README's model list says which.
  - The numbers go into `references.md`.
- **A user's own `.model` file with an old tag** is refused on load with the reason: "taught with the solver before
  <version> (integer / shared occupations); this version uses Fermi–Dirac smearing: re-teach it, or open it in
  <version>". The Load-model card's "apply the model's settings" offer does not offer the occupation difference, which
  cannot be applied.
  - Migrating model files in the app is a non-goal. The engine exists, so it is a small follow-up if the user wants it.

## 8. Interaction with the trust rework (`claude/charming-keller-0bc26b`)

That branch rewrites `providers::surrogate`'s trust and quarantine code: the misfit set M, body-4 degree 3, no-ratchet
recovery and `NearMisfit`. It merges to `main` first. This spec touches the surrogate provider at the following points
only, named by role so they survive the merge:

1. **The set of codes excluded from the fit** (today, the store's "excluded" predicate): add code 8.
2. **The fence / near-misfit refusal predicate**: add a reviewed-and-failed code-8 sample (section 7.6, step 4).
3. **The held-cap count of the recovery**: confirm that code 8 is not counted.
4. **The refit pipeline**: one review step (section 7.6), placed before the trust judgement. It reuses the recovery's
   leave-one-out ratio, whatever name it has on `main`.
5. **The lone-atom rule's "ends at a gap" test** (section 6.4).
6. **The shared-collection attach**: predecessor detection and the migration pass hook (section 7.2–7.3).
7. **The policy's label-noise prior** (`force_noise_prior`): it reads "all-electron" from the tag through
   `is_all_electron_tag`.
8. **The occupation-change training refusal**: deleted (section 6.3).

The plan re-reads these against `main` after the merge. If the rework changed what one of them means, the plan records
the mapping, and this spec is amended rather than reinterpreted silently.

## 9. Validation (rule 6; one heavy job at a time)

Every gate in CLAUDE.md's table applies, because this change touches the solver, dynamics, the surrogate, pseudo
scenes, the orbital display and the viewer. The full serial set is about 4.5–5 h. No tolerance is loosened. Where a
pinned number moves, the reason is the TS term or the re-pinned bitwise path, and that is stated beside the number.

| # | suite / command | what it must show |
|---|---|---|
| **V0** | **Task A1 measurement (entry gate)**, before production code. The diagnosis's env-gated `SCF_SMEAR=1e-3` path on the GPU, plus a CPU f64 cross-check of the worst case. | See "V0 in detail" below. |
| V1 | `cargo test -p units -p qm -p sim -p surrogate --release` | **FD unit tests:** two levels with N = 1, degenerate (f = ½, μ = ε, S = 2 ln 2); a gap Δ (f = 1/(1 + e^{±Δ/2kT}) at the symmetric μ); a gap ≥ 37 kT (f exactly 0.0 / 1.0, S ≤ 1e-14); N_σ = 0; the bracket's validity; determinism (same bits twice); s(f) against the direct formula on a grid of f, plus the 0 ln 0 limit. **Witness rule.** **Energy invariant** (total = internal − ts). **CPU H₄ square a = 1.8** (41³ × 0.30, axes-aligned, exactly degenerate): TS = 4 kT ln 2 within 1e-8 Ha, and E equal to the DS-era E within 1e-6. **FD ≡ Aufbau** on H₂ and H₂O within tol. The DIIS-reset test (section 5). The collection: v2 records, code 8, the read-only open leaving every file byte-identical. |
| V2 | **Regression (new; `cargo test -p providers --release`, GPU, release-only).** The diagnosis harness reduced and committed. Scenes go through `providers::ks::problem`, cold, at 81³ × 0.15, the app's tolerance raised by `effective_gpu_tol`, stall stop on. | See "V2 cases" below. |
| V3 | `cargo test -p validate --release --test oracles` | **σ → 0 oracles** (H, He, H₂): numbers unchanged within the existing tolerances, and **TS ≤ 1e-12** asserted on every oracle system (they are gapped or single-electron). **New FD-consistency oracle** (CPU f64, 41³ × 0.30, tol 1e-9): see "V3 force oracle" below. |
| V4 | `cargo test -p validate --release --test gpu_vs_cpu` | Existing cases within existing bounds. New cases at the suite's fixed σ = h = 0.25, 49³: H₄ square (degenerate) and a lone pseudo-O at the M V0 settled on. Each checks \|F_gpu − F_cpu\|, \|TS_gpu − TS_cpu\|, μ_σ, and per-level f within 1e-6, plus forces, all within the suite's existing energy and force bounds. |
| V5 | `cargo test -p validate --release --test dynamics` | Existing tests unchanged (gapped H₂ and H₃⁺: the conserved quantity is F = E). See "V5 new test" below. |
| V6 | `--test pseudo`, `--test pseudo_gen`, `--test pseudo_oracles` | **Generation records:** bitwise unchanged, with `recipe_hash` moved by `DIMER_PROCEDURE` only. **Dimer-gate verdicts and R_e / ω:** reproduce within the backend margin. **Lone-atom references and the O multiplicity scan** (`pseudo_o_triplet_matches_cpu`, `gpu_oxygen_reference_multiplicity`, `degenerate_spread_at_low_symmetry_sites` repurposed as V0's site study): these move by the measured −TS. Each moved reference is stated in `references.md` with its TS and, for an exactly degenerate shell, the closed-form value (section 4.3). The scan keeps the same M (lowest F). |
| V7 | `cargo test -p validate --release --test surrogate` | Passes, with labels now F; the H₂ and H₃⁺ tests are gapped, so their pins hold. New test: a model file or collection with a legacy tag is refused, and its collection is found as the predecessor. New migration integration test (GPU, small): a collection of H₂ samples (U), linear OH samples (T: ΔF = −2 kT ln 2, E and forces unchanged) and one near-square H₄ (S, reviewed). It checks the counts, the codes, the review's release, and the old folder hashing identical before and after. |
| V8 | `cargo test -p providers --release` | The provider unit tests. The tag strings (`none/fd1e-3`, `…/fd1e-3/stop2x…`). `legacy_tag_of` pinned against the recorded strings. The re-pinned bitwise tests, each with its reason. The lone-rule test on a polarised-or-partial O (declined). |
| V9 | `--test tools`, `--test charges`, `--test orbitals` | Density and orbital consumers. Gapped systems leave the numbers unchanged. The orbital display shows per-level f. |
| V10 | `cargo test -p app --release` (incl. replays) | Label tests (`energy_label` → the F wording). Formats (`collection` = 2, the sidecar bumped, the downgrade line for a migrated folder). The offer / progress / summary text from a fixture predecessor. The replay scripts (they pin quality, walls and ionic; any energy expectation is gapped). |
| V11 | **Dry run on a copy of the user's collection** (measurement, `#[ignore]`d, never on the original) | Counts per class, the class U maximum deviations, the time per sample, and the review outcome (released / kept out). Then the refit's trust verdict against the pre-migration one, recorded in `references.md` "Fermi–Dirac migration". **Must:** the source copy is byte-identical afterwards, and no sample changes code without a logged reason. **Reported, not gated:** whether the migrated collection is trusted as before. If it is not, the reason is investigated before merge. |

**V0 in detail.**
- **Cases.** Lone pseudo-O at M = 1 and M = 3, each at four sites: the node, the cube centre, (0, h/2, h/2) and a
  generic site. Grids: h_req and Accurate. Plus OH (bent and linear), the O₂ singlet and triplet, H₂O, and NH at auto.
- **Proceed if every case converges within 60 iterations without the stall stop.** Also, for each species and M, the
  **site spread of F** must not exceed the `DegenerateShells` site spread of E, measured the same way, by more than
  1e-4 Ha. Comparing like with like cancels the egg-box. The DS failure V0 guards against was 4.1 mHa.
- **Record which state each case reaches:** symmetric (TS ≈ the closed form) or polarised (TS ≈ 0).
- **If V0 fails, stop and ask the user.** Neither of these is chosen by me:
  - (i) raise kT to the measured stability threshold, predicted ≈ 3 mHa by pseudo spec §5.7. That threshold would be
    measured as the smallest kT with a site spread within bound. It costs a larger F bias on stretched and pinned
    states.
  - (ii) keep 1e-3 and document the site dependence of open-shell O as an LSDA symmetry-dilemma artefact, with the
    lone-atom rule already declining such atoms in pairs.

**V2 cases.** Coordinates are D§2.1's centred ones; the rhombus and tetrahedron as listed in D§2.

| policy | case | measured (prototype) | budget |
|---|---|---|---|
| default | H₄ rhombus | 10 | ≤ 20 |
| default | H₄ lopsided tetra | 32 | ≤ 50 |
| default | H₄ seed 1 | 12 | ≤ 20 |
| default | H₅ seed 3 | 18 | ≤ 30 |
| default | H₈ seed 8 | 14 | ≤ 25 |
| `ionic none` | H₅ ring (radius 1.531, xy plane, vertex phase as the harness builds it) | 13 | ≤ 25 |
| `ionic none` | H₆ seed 5 | 47 | ≤ 75 |
| `ionic none` | H₂O + 2 H | 79 | ≤ 120 |
| `ionic none` | lopsided tetra | 28 | ≤ 45 |

- Each case must **converge** (no stall stop, no cap) within its budget, about 1.5× the prototype's count.
- **F and TS** must match the prototype's D§6 values within 1e-5 Ha. Examples: rhombus F −2.079319, tetra
  F −1.932837 with TS 2.7e-3, H₂O + 2 H AE F −47.339721.
- A mismatch is investigated, never re-pinned blindly. Once checked, the pins are the production values.
- The controls H₂ and H₂O must keep the D§6 energies within 1e-6.

**V3 force oracle.**
- **Geometry.** A near-square H₄ rectangle whose frontier gap is a few kT. Task A1 picks the aspect from a short scan
  so that |d(TS)/dx| ≥ 2 mHa/Bohr.
- **Method.** A central difference with δ = 0.01 Bohr along one atom's in-plane coordinate.
- **Must show both:**
  - |F_HF,x + ΔF/2δ| is within the existing HF-vs-finite-difference force bound of the `qm` force tests (the number
    stated in `references.md` "M7 forces", not a new one);
  - |F_HF,x + ΔE/2δ| ≥ 5× that bound, so the test tells F's derivative from E's.

**V5 new test.**
- **Setup.** NVE on an all-electron H₄ square, a = 1.8, at 49³ × 0.25, released from rest with a 1e-3 Bohr
  rectangular perturbation. Run 400 steps of 0.25 fs. The square is a saddle: the gap opens as it falls toward a
  rectangle, and TS swings by ≈ 2.8 mHa.
- **Must show:**
  - the drift of F + K stays within the suite's existing NVE drift bound (`nve_energy_drift_is_bounded`'s bound, per
    atom);
  - E + K moves by ≥ 1 mHa over the run, so the test sees the difference.

**Bitwise pins.** `gpu_none_policy_is_bitwise_today`, the stall-stop bitwise tests and every other pin of the old
all-electron or DS bits are re-pinned in the same commit as the rule change. Each gets a one-line reason: "all-electron
now solves witness vectors (FD, spec 2026-11-13)". They are not deleted.

## 10. Documentation to update

- **`qm::scf` module doc, "Occupations".** Rewritten: the rule, μ, the witness, the energies, section 4.7's states,
  and why the two phases are gone. Plus `qm-gpu::scf`'s matching paragraph.
- **The pseudopotential spec, §5.7.** It is the section that specifies `DegenerateShells`; the main spec has no §5.7.
  Add an amendment block at the top: "superseded 2026-11-13 by `2026-11-13-fermi-smearing-design.md`". The
  why-not-integer evidence stays as history.
  - Its last paragraph's stability claim ("below ≈ 3 mHa its symmetric state is unstable") is answered by V0's
    measurement.
  - §5.5's provenance-tag text gets the new format.
- **The main design spec** (`2026-10-12-atomic-sandbox-design.md`).
  - §6 "SCF loop": "Integer aufbau occupations in v1; Fermi smearing later" becomes "Fermi–Dirac occupations at
    kT = 1 mHa; the solver minimises F = E − TS (spec 2026-11-13)".
  - §11 roadmap item 2: Fermi smearing is done for finite boxes; periodic cells and PBE remain.
- **The orbital-viewer spec, item 4.** Note that fractional, spherical occupations arrived with this spec, so the
  shared-shell rows are now per-level f.
- **`crates/validate/references.md`**:
  - "Approximation gaps that are expected": F = E − TS at T_el = 316 K. TS = 0 on gapped systems. Closed-form TS for
    exactly degenerate shells. The restricted-vs-broken-symmetry H₄ gap (12–24 mHa pseudo, 12.4 mHa all-electron;
    section 4.7(e)). F, not E, is what MD conserves.
  - Every moved reference, with its TS (V6).
  - The V0 site study, V2's pins, V3's force-oracle numbers, V5's drift, the all-electron witness cost (section 4.2),
    the shipped models' re-make, and V11's dry run under a new section, "Fermi–Dirac migration".
- **`docs/dev-reference.md`**:
  - New section "Occupations (Fermi–Dirac)": the constants and where they live, the tag format, `legacy_tag_of`, and
    the bitwise re-pins.
  - New section "Collection migration": the predecessor, the pass, `migration.txt`, the classes, code 8 and its review,
    record v2, the sidecar bump, the downgrade line.
  - "Shared collections": record v2 and code 8.
  - "SCF stall stop": no release restart.
  - "Test suites": V2 and the new oracle, NVE and migration tests.
- **`README.md`** (CLAUDE.md, the user's rule):
  - The readouts (F, T_el, E and −TS in the breakdown).
  - "SCF converges on loose hand-built clusters".
  - The all-electron performance numbers, if they moved.
  - A short "Upgrading: collections are re-solved once" paragraph with the time estimate, and the note that the old
    folder is kept.
  - The shipped models' status.
- **Code docs:**
  - `surrogate::collection` (records v2, the read-only open, the migration files in the layout table);
  - `surrogate::quarantine` (code 8, and an older build meeting it);
  - `providers::pseudo` (the tag);
  - `providers::ks` (no longer "exactly today's problem");
  - the lone-atom rule's doc;
  - `app::update::formats` (the table and the downgrade line).
- **DEVLOG.md** at the merge (memory: DEVLOG upkeep). Link the diagnosis note forward to this spec.

## 11. Implementation order

Model tags are per CLAUDE.md. All are dispatched on Opus, per the 2026-10-21 ruling. Fable-class tasks get a
`physics-reviewer` pass before they land.

0. **Prerequisite.** `claude/charming-keller-0bc26b` has merged to `main`. Rebase `scf-stall`. Re-read the touch points
   in section 8. The diagnostic files (`scf_stall_diag.rs`, the env knobs in `qm-gpu/src/scf.rs`) stay uncommitted
   until A2, where the harness is reduced into V2 and the env knobs are removed.
1. **A1 (Fable-class, measurement):** V0. Also choose V3's aspect and measure the all-electron witness cost.
   **Stop here if V0 fails.**
2. **A2 (Fable-class):** `qm` FD, energies and stats; delete DS; V1; the V3 oracle; the physics review.
3. **A3 (Opus):** `qm-gpu` on the shared host function; V4.
4. **A4 (Opus):** providers: `problem`, the tag, `legacy_tag_of`, `is_all_electron_tag`, `DIMER_PROCEDURE`, the lone
   rule, `EvalStats`, the scan's F; V2, V6, V8.
5. **A5 (Opus):** UI readouts, breakdown, orbital rows, Compare note; V10 (labels).
6. **A6 (Fable-class):** V5 dynamics, plus a physics review of the whole solver half.
7. **B1 (Opus):** `surrogate::collection`: the read-only open, records v2, code 8, `migration.txt`; the sidecar bump;
   `formats`.
8. **B2 (Opus, physics-reviewed):** the providers' migration pass and the code-8 review step at section 8's touch
   points; V7.
9. **B3 (Opus):** app offer, progress and summary; the downgrade line; the Load-model refusal text; V10.
10. **B4 (Opus):** re-make the shipped models; V11 dry run on a copy; the docs (section 10); full serial gate run; then
    merge.

It ships as **one merge**. The solver change without the migration would strand the user's collection behind a new
key.

## 12. Risks

1. **The lone open-shell O at kT = 1e-3** (section 4.7(d)). Gated by V0, and the user rules if it fails.
2. **All-electron per-solve cost goes up** (witness vectors). Measured and reported, not hidden. The default policy
   already pays it.
3. **The migration takes about 1–2 h of GPU time.** It is optional to start, pausable and resumable, and live use is
   not blocked.
4. **Many S-class samples in one region.** That region falls back to DFT until the review runs after the pass; the
   review then judges siblings together (section 7.6).
5. **The trust rework changes a touch point's meaning.** The plan re-reads them after the merge and amends this spec
   (section 8).
6. **Stationary-state switches along trajectories** (section 4.7(g)). Documented, and seen as drift or a leave-one-out
   misfit.
7. **Slow convergence on pinned states** (the tetra 28–32 iterations, all-electron H₂O + 2 H 79). V2's budgets cover
   them. The stall stop must not fire on any V2 case.

## 13. Rulings I made (not decided by the user)

1. **One rule everywhere.** FD also replaces aufbau on all-electron scenes (`ionic none`). It fixes the all-electron
   failures (D§6), and `DegenerateShells` is deleted outright. *Reversible:* keeping aufbau for `none` would keep that
   surface's tag and bits, and its stalls.
2. `Aufbau` stays in `qm` only as the kT → 0 test reference. `for_grid` defaults to FD, so the oracles test the app's
   path.
3. **The μ bisection.** Bracket [ε_min − 50 kT, ε_max + 50 kT], halving to f64 exhaustion, with no tolerance
   parameter. Shared host f64 code for both solvers.
4. **The witness condition** is ε_k − μ ≥ max(0.1 Ha, 36 kT). It keeps today's 0.1 Ha reach (the prototype's
   configuration) and `MAX_WITNESS_GROWTH` = 16.
5. **The entropy** is evaluated from x as ln(1 + e^{−|x|}) + |x|·min(f, 1−f), with Neumaier summation.
6. **`Energy.total` = F**, with `internal` (E) and `ts` added. The per-iteration energy shown is F_HF.
7. **Gillan's E₀ = ½(E + F) is not computed or shown.**
8. **kT is a `qm` constant, `ELECTRON_KT` = 1e-3 Ha.** It is not exposed in the app and there is **no dev flag**;
   harnesses set it through `ScfSettings`. The tag carries kT.
9. **"Auto" multiplicity keeps its meaning** (lowest M of the parity). The reference scan compares F.
10. **`PARTIAL_OCCUPATION_MIN` = 1e-6** defines a partial level. The lone-atom "ends at a gap" test becomes "no partial
    level" on both surfaces, and the `all_electron` argument goes.
11. **`occupation_changed` and its training refusal are retired.** The new failure mode (state switches) is
    documented instead.
12. **The DIIS fix** is achieved by deleting the release path. The frozen-orbital reset stays (its order is correct),
    pinned by a test.
13. **Tags:** `…/fd1e-3/stop2x…` and `none/fd1e-3`. Comparisons go through `is_all_electron_tag`.
14. **The occupation rule and kT are added to `DIMER_PROCEDURE`.** `recipe_hash` moves for Z ≤ `dimer_z_max`, so H and
    He tables regenerate once. The records must be bitwise identical and the verdicts reproduced.
15. **`legacy_tag_of`**, pinned against recorded strings, finds the predecessor exactly. There is no fuzzy matching.
16. **Migration = a new folder on the new key, filled by re-solving.** The old folder is opened strictly read-only and
    never written. An older build still runs on it as it was.
17. **Classification by re-solving every sample** (no stored-data criterion exists), with three classes U / T / S.
    ε_E = 1e-5 Ha per sample, ε_F = 1.5e-4 Ha/Bohr per component.
18. **Class U carries the old record bit for bit.** Class T takes the new label and keeps its trust state, with the
    T test as its review. Class S takes the new label at code 8. Codes 7 and 2 carry over; an old code 1 does not
    survive into S.
19. **Code 8 `Q_RESOLVED`.** It is out of the fit and outside the held cap. It does not fence before review, and does
    fence (or is refused near, as the misfit rule) after a failed review. It is never converted to code 1.
20. **The review** happens in a refit after the pass completes: a candidate fit with all code-8 samples, and release
    where the leave-one-out ratio is ≤ 1. It repeats at later refits.
21. **Failed re-solves are not migrated** and stay in the old folder. They are listed in `migration.txt`.
22. **References are re-solved** (lowest F), never carried. **The audit history is not carried.** `bond_experience.txt`
    is copied.
23. **The pass is user-started** (an offer, with "Later"), runs in the existing pass machinery, re-solves cold for
    reproducibility, and resumes from a cursor in `migration.txt`.
24. **The migration refuses while an older build holds the old folder's lock.**
25. **Collection `RECORD_VERSION` 1 → 2** for everything the new build writes. **The sidecar version is bumped.**
    `history`, `pseudo_cache` and `checkpoint` formats are unchanged. Code 8 is never written to a sidecar.
26. **A downgrade-warning line** names the pre-migration folder the older version will use.
27. **The shipped `.model` files are re-made** by the same engine in a headless harness, or re-taught if one fails its
    trust rule. **In-app migration of user `.model` files is a non-goal**; they get a clear refusal reason.
28. **V0 is an entry gate with a pre-registered stop.** If lone open-shell O fails at kT = 1e-3, the user chooses
    between raising kT and documenting the site dependence. I do not pick.
29. **V2 budgets are about 1.5× the prototype's iteration counts**, and F and TS are pinned to D§6 within 1e-5 Ha.
    Bitwise pins are re-pinned with a reason, never deleted.
30. **Readout wording:** "F (free energy, T_el 316 K)", E and −TS in the breakdown, a −TS note in the solve log when
    TS ≥ 1e-6 Ha, and a note in the PySCF compare.
31. **It ships as one merge** (solver plus migration), and implementation starts after the trust rework merges.
