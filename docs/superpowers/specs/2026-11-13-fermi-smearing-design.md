# Fermi–Dirac occupations and the Mermin free energy (2026-11-13)

Status: design. The user approved the fix and the collection option on 2026-11-13 (section 2); everything else here
is a ruling of mine and is listed in section 13. **Amended 2026-11-14 for the user's ruling after V0** (section 2,
item 3): kT = 5 mHa, and a 1e-5 GPU stop for every solve with a partly occupied level (section 5.1). The V0 note's
2026-11-14 section holds the measurements. These edits are tagged [kT-5]. The design has been amended for the physics review of 682960e
(MUST 6, SHOULD 10, NOTE 9), and again for its re-review of 3be0b63 (the κ̂ estimator, V3's third step, V5's work
bound and cases, the M = 1 threshold, two notes; tagged "re-review"). Each finding is applied where its section is; the review IDs appear in brackets, e.g.
[MUST-1], and section 13 lists the rulings. Evidence: `docs/superpowers/notes/2026-11-13-scf-stall-diagnosis.md`
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
The solver then converges to a stationary point of, and reports, the **Mermin free energy F = E − TS** [SHOULD-3]. The prototype (D§6) converged all 23
default-policy cases and the 3 all-electron failures, and left gapped systems unchanged (H₂ ΔE = 0, H₂O 1e-6).

It also keeps the user's trained collections. F differs from the old E for open-shell and near-degenerate samples,
so the surface key has to change. The old collection is carried onto the new key by re-solving every sample once with
the new solver, never by deleting or converting the old data.

## 2. Decisions already made by the user (2026-11-13)

1. Fermi–Dirac occupations at a fixed electronic temperature, with a chemical potential per spin channel. They
   replace the τ_c equal share of `DegenerateShells` and its two-phase release. The user first set kT = 1e-3 Ha
   (≈ 316 K); V0 failed there, and item 3 replaces it with **kT = 5e-3 Ha (≈ 1 579 K)** [kT-5].
   - The solver reports and minimises F = E − TS. Strictly, the SCF makes F stationary on the branch it reaches
     (section 4.7(e)) [SHOULD-3].
   - The forces stay as they are. They are already f-weighted Hellmann–Feynman forces, i.e. −dF/dR.
   - MD, relax and the surrogate's energy labels use F.
   - The DIIS clear/push order at a release must be fixed if any release logic survives. Removing the release path
     is preferred.
2. **Collections, option (b).** The user's collections are kept. A sample whose label may change is re-solved with
   the new solver and passes a review before it is trusted; it never re-enters the fit silently. Samples whose label
   is unchanged stay.
3. **kT = 5 mHa, and the lone-atom reference scan converges at 1e-5** (2026-11-14, after V0 failed at 1e-3; the V0
   note, `docs/superpowers/notes/2026-11-13-v0-open-shell-stability.md`). The user chose option (i), raising kT, at
   5 mHa, with the scan's convergence stop at 1e-5 instead of 1e-6. Where that stop applies is my ruling (section 5.1,
   ruling 42): on the GPU, every solve whose occupations have a partly occupied level, not only the scan.

## 3. Goals, non-goals, anti-goals

**Goals**

- **G1. Convergence.** Every staller in the diagnosis converges within a stated iteration budget (section 9, V2).
  That is 5 default-policy stallers and 3 all-electron failures, plus the all-electron lopsided tetrahedron, which
  takes 56 iterations under aufbau today. None of them may end by the stall stop or the cap.
- **G2. One rule.** One occupation rule for every scene the providers solve, all-electron and pseudised alike.
  `DegenerateShells`, its release and its "pattern frozen by rank" logic are deleted.
- **G3. One energy.** F is *the* energy: what the solver makes stationary (self-consistent) — the minimum of F on the
  branch the SCF reaches, which is not always the global one (section 4.7(e)) [SHOULD-3] —, what `ForceProvider`
  returns, what MD conserves
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
- **Non-finite input** [SHOULD-1]. A non-finite eigenvalue raises `QmError::Numerical` before the bisection starts.
  Without this check a NaN would never compare equal to an endpoint, so the stop rule below would never fire.
  - The bracket is also checked once, as built (Task 4; kept by its review): both ends finite and
    Σf(lo) < N_σ < Σf(hi), else `QmError::Numerical`. It changes no bracket; it fails only when |ε|/kT ≳ 1e14 absorbs
    the 50 kT margin, where the bisection would otherwise converge silently onto an end. k_σ < N_σ + 1 and a kT that
    is not finite and > 0 are refused the same way.
- **Step.** Halve the bracket on the sign of Σf(mid) − N_σ. Stop when the midpoint equals an endpoint in f64, i.e. the
  bracket is exhausted.
  - The count is about 60 halvings when |μ| ~ 1 (a 20 Ha bracket, as for all-electron O 1s).
  - Near μ = 0 at kT = 5 mHa it is ≲ 120 (measured 113–116, Task 4): within |x| < 1.1e-16, e^x rounds to 1, so Σf is
    flat over ≈ ±1.1e-16·kT around a root at 0 and the bisection stops there. About 1 100 is reached only for a
    subnormal kT, whose flat zone lies below the subnormal spacing [NOTE-1]. That is still µs.
  - A backstop cap of 2 048 halvings never binds. It is there only so that termination never depends on the stop rule
    alone.
- **Ties and the gapped plateau** [MUST-1]. In a gapped channel, Σf is exactly N_σ in f64 over a whole plateau of μ.
  - The plateau runs from about ε_HOMO + 37 kT, where 1 − f_HOMO drops below ½ ulp(1), to about ε_LUMO − 36 kT, where
    f_LUMO drops below ½ ulp(N_σ). It exists whenever the gap exceeds about 74 kT. At kT = 5 mHa that is 0.37 Ha
    [kT-5], so only very wide gaps have it and most closed shells do not. Without a plateau the two bisections
    meet at the single root, and the f above μ are e^{−(ε−μ)/kT}, small but not below f64 resolution.
  - On the plateau, "halve on the sign" is undefined, and a tie rule would drive μ to one edge rather than to the FD
    root near mid-gap.
  - The solver therefore runs the bisection twice:
    - with Σf(mid) − N_σ ≥ 0 moving the upper end, which converges to a crossing (the least μ with Σf ≥ N_σ, up to
      rounding);
    - with Σf(mid) − N_σ ≤ 0 moving the lower end, which converges to a crossing (the greatest μ with Σf ≤ N_σ, up
      to rounding).
  - It reports the midpoint of the two. Without a plateau the two coincide to f64.
  - Only the f matter for the physics, and they are identical to ≤ e^{−36} anywhere on the plateau.
  - μ_σ is a reported diagnostic only where the channel has a partial level (section 5). In a gapped channel it is
    printed as the plateau midpoint and never compared across solvers or used by any test as a physical number.
- After the bisection, |Σf − N_σ| ≲ k_σ·ulp(μ)/(4kT) ≈ 1e-13. The density is still charge-renormalised per channel by
  the mixer, as today.
- No tolerance parameter is needed, and the function is deterministic: the same f64 inputs give the same μ and f on
  every run and in both solvers.

**Per spin, not joint (the user's decision; why).** A per-channel μ_σ at fixed N_σ is the fixed-spin-moment
ensemble.

- It keeps what `multiplicity` means today: the user's forced M, or "auto", which is the lowest M of the electron
  count's parity.
- It keeps the collections' per-sample multiplicity record meaningful.
- A joint μ would let the moment float. That is physical, but it changes the solver's electron counts, its
  normalisation and every sample's spin record, and at T_el in LSDA it rarely moves the moment off an integer anyway.

### 4.2 Witness levels (the FD tail)

The rule needs every level with non-negligible f, so each channel grows its block until the highest solved level
satisfies:

```text
ε_kσ ≥ ε_{N_σ,σ} + WITNESS_GAP   and   ε_kσ − μ_σ ≥ 36 kT,   WITNESS_GAP = 0.1 Ha
```

Here ε_{N_σ,σ} is the N_σ-th lowest level [MUST-1].

- **The first arm** is today's τ_c reach, measured from the N_σ-th level and not from μ. It is exactly the configuration
  the prototype ran: the prototype applied FD only after `degenerate_shell_occupations` had accepted that witness
  (`qm-gpu/src/scf.rs`, D§6). So the cost claim below and V2's "match the prototype" premise rest on the configuration
  production will run.
  - A μ-relative witness would have meant ε_HOMO + 0.137 Ha or ε_LUMO + 0.064 Ha in a gapped channel, depending on the
    tie rule.
- **The second arm** is the FD tail condition. At 36 kT, f < 2.4e-16, below f64 resolution next to 1; at 0.1 Ha =
  100 kT, f ≈ 4e-44. At kT = 1e-3 it bound only when μ lay more than 0.064 Ha above ε_{N_σ}.
  - **At kT = 5 mHa it binds in every channel** [kT-5]: 36 kT = 0.18 Ha, and μ is never far below ε_{N_σ} (at mid-gap
    in a gapped channel, within a few kT of it in a partial one), so ε_max ≥ μ + 0.18 Ha implies the first arm.
  - **Measured (V0, 2026-11-14; 14 Bohr box, both grids, four sites):** it fits under the cap in every case. The
    extra vectors per channel reached 2–5 (B, C, N, O, F, Cl at their winning M, the molecules, all-electron O at
    M = 1, 3), 7–12 (Si and S at M = 3, P at every M, F and Cl at M = 4, O and all-electron O at M = 5, Si at M = 1, 5)
    and at most **15 of 16** (S at M = 5 on its h_req grid). The reach then lands at 40–82 kT, because a single added
    vector often jumps across a cluster of box states.
  - **The margin is thin for the diffuse multiplicities** (the 4s-like ↑ level of S at M = 5 and Cl at M = 4, O's 3s-like
    level at M = 5; all-electron O at M = 5: a diffuse ↑ manifold among the box states, TS ≈ 2× the three-level closed form). Box states scale as L³, so the same atom in a box a few per cent larger than 14 Bohr can exceed 16. A
    failed multiplicity is left out of the scan (`sim::references::solve_atom`), so it cannot change the winner
    unless the winner fails, and every winner needs ≤ 10. The cap error is NOTE-6's risk, now reached by the 3p
    atoms' non-winning multiplicities as well as by anions.
  - **Its cost is large.** Growth is one vector at a time, each growth re-solving the block. Cold solves of the 3p
    atoms took 5–80× the `DegenerateShells` time (Si at M = 5, h_req: 80 s against 1 s; S at M = 3, 0.15: 60 s
    against 4 s); O at M = 1 and 3 (2–3 extras) cost the same as DS. Warm solves keep the grown block and cost the
    same.
  - **No change to the multiple is needed for V0** (it fits). If the user wants margin or speed, the smallest changes,
    neither loosening a tolerance, are: (a) `MAX_WITNESS_GROWTH` 16 → 32, a cost cap only; (b) growth by several
    vectors per re-solve (e.g. doubling the extra block), a cost change only; (c) a tail arm derived from the stop
    instead of from f64 resolution: at X kT the first missing level holds f ≤ e^{−X}, and the neglected charge is about
    g·kT·e^{−X} for a level density g (≈ 60 per Ha above μ in the 14 Bohr box: 15 levels over 0.26 Ha for S at
    M = 5), so X = 25 leaves ≈ 4e-12 electrons and kT·that ≈ 2e-14 Ha in F, five orders below the 1e-6 Δρ stop and the
    µHa energy scatter. (c) shrinks the window from 0.18 to 0.125 Ha. This spec keeps 36 kT and 16 (ruling 43).
  - **Amended by plan Task 9c (2026-11-16/17), the cold-start window.** Tight steps (the step that can converge)
    keep the full 36 kT window, so the converged answer and every convergence criterion are unchanged. On
    loosened steps only, the window is a step-accuracy tail X = ln(n_w/(c·Δρ_prev)), c = 1e-3, clamped to
    [6, 36] kT. The bound is Σf ≲ g·kT·e^{−X}, conservative while n_w ≥ g·kT. Cold FD solves start from free-atom
    densities (qm::radial, Z only) for all-electron centres and Z_v blobs for pseudised ones. The GPU block carries a
    buffer column, which counts as the witness only when its residual is ≤ the guard tolerance. The VRAM guard
    counts the measured start allocation (3 per spin), and growth past it is capped by the remaining budget with a
    named error. A loose-column stop was measured and dropped (stretched-H₂ 43–102 steps > 30). Measurements:
    `crates/validate/references.md`; rulings: the SDD ledger.
- Growth is one vector at a time, re-testing both arms on the re-solved levels. Growing the block re-solves every
  level, so no monotonicity is claimed. The loop is bounded by the cap.
- The cap stays: `MAX_WITNESS_GROWTH` (16 at this spec's writing; 32 since plan Task 9b, and the solver's cap is
  min(32, what the VRAM budget fits)), with the same error when it is exceeded.
  - **A new error path** [NOTE-6]. All-electron scenes now grow witnesses. In a large box, or with an unbound LSDA
    HOMO (anions), the density of box states inside 0.1 Ha of ε_{N_σ} can exceed 16. N(E) ∝ L³E^{3/2} gives about 28
    states per 0.1 Ha at L = 20 Bohr near 0.2 Ha.
  - The ε_N-relative arm keeps the window as today's, so the risk is the pseudo path's existing one, now also on the
    all-electron path.
  - V8 pins it with an all-electron H⁻ at the app's largest box. Its outcome (converges, or the documented cap error)
    is recorded.
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

| case | occupation | TS at kT = 5e-3 Ha [kT-5] | (at the first ruling's 1e-3) |
|---|---|---|---|
| s(½) | — | ln 2 = 0.693147 | |
| s(⅓) = s(⅔) | — | 0.636514 | |
| H₄ square | two half-filled pairs | 4 kT ln 2 = 1.38629e-2 Ha | 2.7726e-3 (measured 2.77e-3, D§6) |
| H₅ ring | one half-filled ↓ pair | 2 kT ln 2 = 6.9315e-3 Ha | 1.3863e-3 |
| linear OH | ↓ π pair holds 1 | 2 kT ln 2 = 6.9315e-3 Ha | 1.3863e-3 |
| one open p channel: lone O and S at M = 3, C and Si at M = 3, B at M = 2 and 4, F and Cl at M = 2 and 4 | 1 or 2 of 3 | 3 kT s(⅓) = 9.5477e-3 Ha | 1.9095e-3 |
| two open p channels: lone O, S, C, Si at M = 1; N and P at M = 2 | ↑ and ↓ each hold 1 or 2 of 3 | 1.90954e-2 Ha | 3.8191e-3 |

The atom rows hold where the symmetric ensemble is what the SCF reaches. V0 measured that it is, for every accepted
open-shell species at kT = 5 mHa (section 4.7(d) and the V0 note): TS equals the closed form to ≤ 1e-6 relative, or
lies 0.3–0.6 % above it where a diffuse level is also partly occupied (F and Cl at M = 4).

**Gapped systems** [SHOULD-5]. With a gap of at least 37 kT on each side of μ:
- every f below μ is exactly 1.0 in f64;
- every f above μ is ≤ e^{−36}. It is representable down to 1e-308, so it is never 0.0. (The prototype's
  `x > 40 → 0.0` clip hid this.)
- S ≤ 1e-14.

So TS ≤ 1e-14 Ha on every closed shell and every integer-per-spin open shell gapped by ≥ 74 kT. On 4 of the 5 stallers
D§6 measured TS ≤ 2e-7 Ha [NOTE-7]: they are integer-occupied, but not by a 74 kT margin.

**Below 74 kT, at kT = 5 mHa** [kT-5]. A gap Δ with μ at mid-gap gives, per channel, TS ≈ 2kT(1 + y)e^{−y} with
y = Δ/(2kT). That is 4e-10 Ha at Δ = 0.2 Ha, 5e-6 Ha at 0.1 Ha, and 1e-5 Ha (the migration's ε_E, section 7.4) at
0.092 Ha. At the first ruling's 1e-3 the same 1e-5 was reached only below Δ ≈ 0.015 Ha. So every frontier gap below
about 0.1 Ha now carries a visible TS: stretched bonds, near-degenerate pinned states and diffuse open levels (the
lone O at M = 5 carries 6.3e-5 Ha from its 3s-like ↑ level against the box states; V0 note).

**Not computed: the kT → 0 estimate** [NOTE-8]. For a smooth density of states the T² terms of E and F cancel in
Gillan's E₀ ≈ ½(E + F), so its error starts beyond T². It is off by TS/2 exactly in the degenerate case, the one where
it would matter, so it is not reported.

### 4.4 Forces: −dF/dR, unchanged code

F[{ψ}, {f}] at fixed N_σ is stationary in both ψ and f:
- ∂F/∂ψ = 0 gives the KS equations.
- ∂F/∂f_iσ = ε_iσ − μ_σ + kT ln(f/(1−f)) = 0 is exactly the FD condition, with μ_σ the Lagrange multiplier of
  Σf = N_σ.

So dF/dR = ∂F/∂R|explicit. The occupation derivatives drop out (Σ_i μ_σ df_i/dR = μ_σ dN_σ/dR = 0), which leaves the
Hellmann–Feynman force with f-weighted density and projector sums. The existing code computes exactly that
(`crates/qm/src/forces.rs`, local term and nonlocal projector sum; `crates/qm-gpu/src/forces.rs`), to the same floors as
today: the SCF and eigen residuals, and the discrete Hartree map's asymmetry (≲ 1e-5 Ha/Bohr, `forces.rs` module doc)
[SHOULD-8]. The code has been f-weighted since M7, so **no force code changes**. Only the module docs change, from
"−dE/dR at fixed occupations" to "−dF/dR of the Mermin functional, stationary in ψ and f" (section 10).

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

- `qm::scf::ELECTRON_KT = 5e-3` (Ha; the user's ruling of 2026-11-14, section 2 item 3 [kT-5]), a solver constant in
  atomic units. It is not per element, not in any table, and the same for every scene.
- `ScfSettings.occupations = Occupations::FermiDirac { kt }`. The constant goes in through `ScfSettings::for_grid` and
  `providers::ks::problem`.
- **Not exposed in the app, and no dev flag.** Harnesses and `validate` measurements set `kt` through `ScfSettings`
  directly.
  - A user-visible T_el would move the surface (and the collection key) with a UI control.
  - A dev flag would need worker plumbing, recording and a provenance path for something no current measurement needs
    inside the app.
  - The provenance tag carries kT (section 6.2), so if a flag is added later, data at another kT can never mix.
- **Display.** T_el = kT/k_B = 1 578.9 K, shown as 1 579 K (315.8 K at the first ruling's 1e-3). It is converted only
  at the UI boundary through `units` (rule 3).

### 4.7 What the new state is (rule 7)

- **(a) Gapped systems.** F = E to within the GPU path scatter [SHOULD-6]. D§6 measured H₂ ΔE = 0 and H₂O ΔE = 1e-6 Ha,
  against warm-vs-cold path scatters of up to 1.87e-6 Ha (pseudo H₂O, `references.md` "SCF, GPU against CPU").
  Nothing physical changes.
- **(b) Exactly degenerate partly filled shells** (symmetric H₄, the H₅ ring's ↓ pair, linear OH's π, a symmetric
  atom).
  - FD gives the same equal-weight ensemble `DegenerateShells` gave, so E is unchanged and F = E − TS (table above).
  - The lattice-split pairs (≤ 5.3e-5 Ha) get f = ½ ∓ Δ/(8kT) = ½ ∓ 0.0013 at kT = 5 mHa instead of exactly ½:
    0.0013 absolute, 0.27 % relative (½ ∓ 0.0066 at 1e-3) [NOTE-2] [kT-5].
  - The pseudopotential spec's §5.7 argument (levels this close are thermally co-occupied) is now the rule itself,
    made continuous.
- **(c) Near-degenerate frontier ("pinned") states.** F lies below E by TS, which is O(kT·N_pinned). This is the
  Mermin–Kohn–Sham functional at the fixed electronic temperature T_el, with ground-state LSDA xc [SHOULD-4].
  - S is the entropy of the KS occupations, not the molecule's physical electronic entropy. The H₄ square's 4 ln 2
    counts two independent half-filled pairs per spin, not the many-body multiplet degeneracy.
  - It is a defined, documented part of the surface, not a solver error. It biases nothing in gapped systems.
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
  - **The threshold is species-dependent** [MUST-2]. U grows with orbital compactness (F 2p > O 2p > C 2p). Rule 1
    forbids a per-element kT, so the one global kT must be stable for the worst accepted species, on both surfaces.
    The all-electron open-shell atoms ran aufbau until now: pseudo spec §6.1 says "the O triplet stops at the SCF cap".
  - **Predictions pre-registered for V0** [SHOULD-9]:
    - *M = 3:* kT* = f(1−f)·U = (2/9)·U, with f = ⅓ and U ≈ 11.8 mHa from the integer splitting, so ≈ 2.6 mHa.
      The symmetric state is predicted to be a saddle at kT = 1e-3.
    - *M = 1:* both channels have f = ⅔ in three p levels, so the occupation modes couple across channels.
      - The charge-like mode (the same pattern in both channels) has U_c = U↑↑ + U↑↓. The measured M = 1 integer
        behaviour (the occupied pair's oblate density pushes the empty p *below* an occupied one) says this mode is
        anti-aufbau, U_c < 0, and stable.
      - The spin-like mode (opposite patterns) has U_s = U↑↑ − U↑↓, with kT*_s = (2/9)·U_s.
      - **The two premises together fix the sign** (re-review R-SHOULD-D). U↑↑ > 0 (from M = 3) and U_c < 0 imply
        U↑↓ < −U↑↑. So U_s > 2U↑↑ and kT*_s ≳ (4/9)·U↑↑ ≈ 5 mHa, if M = 1's U↑↑ matches M = 3's (it could be smaller:
        M = 1's minority is a hole in three p, not a single electron).
      - **So the M = 1 symmetric ensemble is predicted to be a saddle at kT = 1e-3, with gain ≳ 5.** This is the
        expected outcome of the spec's own premises, not a side risk. V0 may well fail at M = 1 with kT = 1e-3, and
        option (i)'s kT would then be set by the M = 1 spin mode (≈ 5 mHa), not by M = 3 (≈ 2.6 mHa).
      - "Symmetric at M = 3, polarised at M = 1" is therefore anticipated. V0 extracts U↑↑, U_c and U_s from the
        integer states' splittings and checks them against the observed state at each M.
    - *GPU occupation jitter:* FD turns f32 eigenvalue noise into density noise, at up to 1/(4kT) = 250 Ha⁻¹; the
      equal share of `DegenerateShells` was exactly flat. V0 records the per-iteration jitter δε of the shell's
      eigenvalues near convergence on the GPU. It must keep the implied Δρ floor, ≈ 2δε/(4kT)·∫ ||ψ_i|² − |ψ_j|²|, a
      factor 3 below the GPU stop rule max(tol, 2·floor·N). Failing that is a V0 failure.
  - The diagnosis did not test lone open-shell atoms. **Task A1 measures this before any production code, for every
    accepted open-shell species, both multiplicities and both surfaces, and stops for a user ruling if any fails**
    (section 9, V0).
  - **What V0 found** [kT-5] (the V0 note):
    - *At 1e-3 (2026-11-13): fail.* O at M = 1 and N at M = 2 polarise site-dependently (21–39 mHa spreads; the CPU f64
      cross-check confirms the physics), and every open shell is held up by the f32 noise floor against the 1e-6
      stop. The linear thresholds from the measured U are O M = 3 4.3 mHa, the O M = 1 spin mode 19.5 mHa, F M = 4
      3.2 mHa; the SCF does better than the linear estimate and is symmetric everywhere from 2–3 mHa.
    - *At 5 mHa with the 1e-5 stop (2026-11-14):* for every accepted open-shell species (B, C, N, O, F, Si, P, S, Cl,
      and all-electron O), every converged solve at every site, on both grids and at every scanned M is in the
      **symmetric ensemble** (or a gapped state); none polarises. TS equals the closed form to 1e-6 relative (0.3–0.6 %
      above it where a diffuse level is also partly occupied). The scan's winner is the same at every site and equals
      the `DegenerateShells` winner (all-electron O on Balanced: M = 1 everywhere, where DS stalled at M = 1). The
      remaining misses are convergence-noise ones, listed in section 9, V0.
- **(e) Broken symmetry is not addressed.** Smearing converges to the nearest stationary state, not the lowest.
  - The lopsided H₄ tetrahedron (D§6, pseudo-H, 81³ × 0.15) converges to a spin-symmetric state with its frontier pair
    pinned at μ (f = 0.57 / 0.43): F = −1.932837 Ha.
  - Unsmeared runs reach spin-broken integer states at E = −1.944984 (α = 0.1) and −1.956852 (α = 0.05), 12 and
    24 mHa lower.
  - All-electron: F = −1.813947 against the integer −1.826310 (12.4 mHa).
  - This is LSDA's static-correlation regime.
  - **Two claims here are empirical until Task A1 tests them** [SHOULD-2]:
    - **"A spin-symmetric cold guess stays symmetric."** The two channels start from different random vectors
      (seeds 1 and 2; the witness seeds differ by spin), so ρ↑ = ρ↓ holds only to eigensolver noise. The symmetry
      survives because the DIIS map damps the antisymmetric mode within one solve. Nothing guarantees that across
      thousands of warm-started MD steps that restart DIIS.
    - **"The 12–24 mHa gap is LSDA broken symmetry."** It compares F of the FD state with E of two integer states
      found under the old rule. It is physics only if those broken-symmetry states are also stationary points of F at
      kT = 1e-3. Their gaps (0.031 / 0.027 Ha ≫ kT) make that likely, but it is not yet tested.
    - Task A1 seeds FD with the α = 0.1 and α = 0.05 broken-symmetry densities. It confirms that each stays (TS ≲ 1e-6)
      at F ≈ E within the path scatter, i.e. that the spin-symmetric state is a higher stationary point of the same F.
  - **A structural alternative, for the user to decide** (offered, not adopted; ruling 33). Under auto with N↑ = N↓ and
    a symmetric guess, the two channels could share seeds, or be one solve. ρ↑ = ρ↓ would then be exact, and the auto
    surface would be the spin-restricted one by construction: one branch and no jumps, at the documented
    restricted-vs-broken cost.
  - Documented in `references.md`, not hidden. A symmetry-breaking nudge is a separate decision (non-goal).
- **(f) What MD conserves.** BO-MD on F conserves F + K_nuc. E + K_nuc is not conserved where TS varies along the path,
  and the readouts say so (section 6.6).
- **(g) Branch jumps, and their guard** [MUST-3]. On one branch F(R) is smooth in the levels, so the
  occupation-pattern warning (`EvalStats::occupation_changed`) loses its reason.
  - Smoothness does not stop a warm-started trajectory crossing from one stationary state to another, such as the
    restricted fractional and broken-symmetry states of (e). F then jumps by up to 12–24 mHa for the lopsided H₄. That
    is 2–4× the nuclear KE of four atoms at 300 K (5.7 mHa).
  - Occupations cannot flag this, NVE drift is not a detector, and leave-one-out catches it only after both branches
    are stored.
  - So the old watch is **replaced, not just retired**, by a work-consistency guard, `EvalStats::branch_jump`
    (section 6.3). It warns and refuses surrogate training on a step whose ΔF does not match the work the forces did.
    That is stricter than the old watch.
- **(h) Separated fragments share μ_σ** [SHOULD-10]. One μ_σ per channel spans the whole box. Where two far-apart
  fragments' frontier levels align within a few kT, FD puts fractional charge or spin on each: LSDA's delocalisation
  error, made continuous.
  - `DegenerateShells` did the same inside its 0.1 Ha window, and aufbau did it as integer flips. It is not new
    physics, but the user can build it (H and O far apart, a dissociating heteronuclear bond).
  - Documented in `references.md` "Approximation gaps".
- **(i) Atomisation and binding energies** [NOTE-4]. Energies taken against isolated-atom references shift by
  +TS_atom: at kT = 5 mHa a lone O carries −TS = −9.55 mHa at M = 3 (V0 measured the symmetric state at every site)
  [kT-5]. Every open-shell reference carries 9.55 mHa per open p channel (B, C, O, F, Si, S, Cl at their winning M),
  so the atomisation energy of H₂O against O + 2 H is 9.55 mHa (0.26 eV) smaller than the integer-occupation number,
  and that of a molecule made of k such atoms ≈ k × 9.55 mHa smaller. A gapped molecule's own TS is ≤ 4e-10 Ha for
  frontier gaps ≥ 0.2 Ha (section 4.3), so the shift is the atoms'. This is stated in `references.md`, and wherever
  the app shows E − ΣE_ref, the text says the references are free energies at T_el.

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

- `Occupations::FermiDirac { kt }`, and `pub fn fermi_dirac(evals, n, kt) -> Result<FdOccupations, QmError>`
  (`FdOccupations { f, mu, entropy, halvings }`) in `qm::occupations`, re-exported from `qm::scf` and the crate root.
  - It is host f64 code, used by the CPU loop and imported by `qm-gpu`, just as `qm-gpu` imports
    `degenerate_shell_occupations` today. CPU and GPU therefore share every operation after the eigenvalues
    (G4: the eigenvalues come from the f64 Rayleigh–Ritz on the host in both).
- **`Energy`** gains `internal` (E) and `ts` (kT·S ≥ 0).
  - `total` becomes F = `internal` − `ts`, so every consumer of `total` (providers, MD, relax, surrogate) gets F with
    no further change.
  - Invariant tested: `total == internal − ts` exactly, and the components sum to `internal`.
- **`ScfStats`** gains `entropy` (S) and `mu: [Option<f64>; 2]`. `fractional` becomes `partial: Vec<PartialLevel>`
  with `{ spin, index, eps, f }` for every level with min(f, 1−f) ≥ `PARTIAL_OCCUPATION_MIN` = 1e-6.
  - 1e-6 corresponds to |ε − μ| ≤ 13.8 kT, so "no partial level" means a gap of at least 27.6 kT around μ in every
    channel: 138 mHa at kT = 5 mHa (28 mHa at 1e-3) [kT-5]. Sections 5.1 and 6.4 rely on that.
- **`Aufbau`** stays in `qm` only as the kT → 0 reference. One equivalence test uses it: FD and Aufbau agree on H₂ and
  H₂O within |ΔE| ≤ 1e-8 Ha (CPU f64, tol = eig_tol = 1e-9; V1) (re-review SHOULD-6). Nothing in `providers` can select it.
  - At kT = 5 mHa the f agree only to about e^{−Δ/2kT}: TS is below 1e-8 Ha only for a frontier gap ≥ 0.17 Ha
    (TS(0.15 Ha) = 4.9e-8, TS(0.2 Ha) = 4.3e-10 per spin pair), and E moves by about f_LUMO·Δ [kT-5]. The equivalence
    test is about the rule's kT → 0 limit, not the production kT, so it runs at an explicit `FermiDirac { kt: 1e-3 }`,
    where the 1e-8 bound holds for any gap ≥ 0.05 Ha, and the 1e-8 bound is not raised. The production kT's effect on
    the same H₂O (its gap, TS and F − E) is printed next to it and recorded in `references.md`.
  - `ScfSettings::for_grid` defaults to `FermiDirac { kt: ELECTRON_KT }`, so the oracles test the path the app runs.

**Unchanged**

- The mixer (Pulay/DIIS on stacked spin densities, α = 0.3, m = 6, clamp, per-channel renormalisation).
- Eigen tolerance tied to Δρ.
- The stall stop (W = 50, level 1e-3 per electron). It is still checked after the convergence test, still a safety net.
- The convergence criterion (Δρ < tol on a tight eigen step). The tolerance it compares against is unchanged on the
  CPU and for gapped GPU solves; a GPU solve with a partly occupied level stops at 1e-5 (section 5.1) [kT-5].
- The cold guess (ρ↑ : ρ↓ = N↑ : N↓). Occupations are recomputed from each step's levels; they are not mixed.

**Warm start.** `KsState.occupations` records the scheme including kT, and the existing guard requires the same scheme.
A state converged at another kT (a harness) never seeds a solve.

### 5.1 The GPU stop for a solve with a partly occupied level [kT-5]

**Why.** FD's slope at μ, 1/(4kT), turns the f32 eigenvalue jitter δε of a partly occupied level into occupation
noise, and so into a Δρ floor the GPU cannot get under (spec risk 8; the V0 note, section 4 and its 2026-11-14
section). `DegenerateShells` had no such floor: its equal share was exactly flat. The floor is a property of the
partly occupied level, not of the lone-atom scan: an O₂ singlet, an OH radical or a stretched bond in a collection
sample carries the same pinned levels, and the sample teacher's 1e-6 would stall on them as the scan did in V0.

**The rule.** On the GPU, under `FermiDirac`, the convergence test of an SCF step is

```text
Δρ < tol_eff                                   if the step's occupations have no partial level
Δρ < max(tol_eff, GPU_PARTIAL_TOL)             if they have one (min(f, 1 − f) ≥ PARTIAL_OCCUPATION_MIN in either channel)
tol_eff = effective_gpu_tol(tol, N)            under an ionic setup (unchanged: max(tol, 2·5e-8·N)); tol otherwise
GPU_PARTIAL_TOL = 1e-5                         (a solver constant defined once in qm::occupations, re-exported by
                                               qm-gpu (preflight F11); the user's 2026-11-14 value)
```

- "Partial" is the spec's existing definition (section 5, `partial_levels`): a level within 13.8 kT = 69 mHa of μ.
  It is evaluated on the occupations of the step being tested, so a cold solve stays bit-reproducible.
- **Where it applies:** every GPU SCF (pseudised and all-electron; the reference scan, the sample teacher, the
  migration's re-solves, the sampling pass, live MD), not only the scan. Gapped GPU solves keep exactly today's stop,
  so every gapped pin (H₂, H₂O, the oracles) is unaffected by it. The CPU f64 path has no f32 floor and keeps `tol`.
  The interactive 1e-5 is already at or above it.
- **The tolerance that is recorded.** `ScfStats` reports the stop it used (`tol_used`), and `EvalStats::scf_tol_used`
  carries it, so the `branch_jump` guard picks the 1e-5 tier for such a solve (section 6.3). `Provenance::tol` and
  `ReferenceKey::tol` keep the nominal tolerance, as they already do for `effective_gpu_tol`; the rule itself is in
  the provenance tag (section 6.2).
- **The stall stop** (W = 50, 1e-3 per electron) and the iteration cap are unchanged.

**What stopping at 1e-5 costs** (V0 note, 2026-11-14). F is stationary in ρ, so the error of a state with SCF
residual Δρ is second order in it; the Hellmann–Feynman forces are first order.
- *Estimate.* A residual δρ of L1 norm 1e-5 spread over a ~1 Bohr region costs about its Hartree self-energy,
  ‖δρ‖²/R ≈ 1e-10 Ha, in F. The force on nucleus I moves by at most ‖δρ‖₁·max|∇V_loc,I|; for a pseudo-O local part
  (Z_v = 6, r_c ≈ 1.1 Bohr) max|∇V| ≈ 2 Ha/Bohr, so ≤ 2e-5 Ha/Bohr, a strict upper bound.
- *Measured* (every V0 case at kT = 5 mHa: the cold solve stopped at 1e-5 against a warm re-solve to 3e-6 from the
  noise floor): on the pseudo surface |ΔF| ≤ 1.7 µHa, and |ΔF_I| ≤ 8.8e-7 Ha/Bohr per component on every solve with a
  partial level (atoms and molecules), ≤ 3.2e-6 including gapped ones (P at M = 4, a cold stop at Δρ = 9.1e-6); on
  all-electron O ≤ 2.9 µHa and ≤ 2.1e-6 Ha/Bohr. The energy differences are the size of the
  warm/cold path scatter σ_path (0.04–0.7 µHa pseudo, 0.6–1.1 µHa all-electron), so the stop's own second-order
  error is not resolved above the f32 scatter, as the estimate says. The force differences sit ≥ 20× below the linear
  bound (the residual is not where |∇V| peaks, and on a lone atom it is largely inversion-symmetric).
- Against the surface's own tolerances: ≥ 47× below the migration's ε_F = 1.5e-4 Ha/Bohr, ≥ 15× below
  `PSEUDO_FORCE_NOISE` = 5e-5 (≥ 57× on the partial-level solves the rule affects), and ≤ ε_E/3 in energy
  (ε_E = 1e-5 Ha, section 7.4).

**What it does not fix.** For the M = 1-type states (O, C, Si, S at M = 1; N and P at M = 2; the O₂ singlet and NH at
auto; all-electron O at M = 5) the f32 floor's median is 1e-6 to 2.8e-5, so 1e-5 is reached by a downward
fluctuation rather than with a factor-3 margin (V0 note). They converge within 60 iterations at every site measured
except S at M = 1 on Accurate (61, one site) and all-electron O at M = 5 (over 60 at 5 of 8 sites, not converged in 100 at
3 of them), and the spec's jitter rule
(floor ≤ stop/3) fails for them; see section 9, V0.

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
| pseudised | `<word>/<surface_hash>/shared1e-3/1e-1/stop2x<floor>` | `<word>/<surface_hash>/fd5e-3/stop2x<floor>/pin1e-5` (`fd{kT:e}`, `pin{GPU_PARTIAL_TOL:e}`) |
| all-electron | `none` | `none/fd5e-3/pin1e-5` |

[kT-5] The tag names kT (5e-3, the 2026-11-14 ruling) and the partial-level GPU stop of section 5.1, so data solved
at another kT or under another stop rule never shares a collection. The `pin` part is the same on every teacher (the
CPU teacher simply never reaches it), as `stop2x<floor>` already is.

- The all-electron tag must change too: aufbau labels of near-degenerate all-electron samples (stretched bonds, open
  shells) move.
- Every comparison with the literal `"none"` that means "the all-electron surface" goes through one predicate,
  `providers::pseudo::is_all_electron_tag`. About 50 literal uses exist, many of them the scene's `ionic none` policy
  word, which is unrelated and unchanged.
- `SurfaceKey::slug` needs no special case: `none/fd5e-3/pin1e-5` hashes like any other tag (`none-<12 hex>`).
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
- **`EvalStats::occupation_changed` is replaced by `EvalStats::branch_jump`** [MUST-3]. The old flag existed because
  E(R) jumped when a DS pattern switched (pseudo spec §5.7, "Molecules"). On one branch F(R) does not jump, but it does
  across branches (section 4.7(g)).

  **The check.** After each evaluation, the Kohn–Sham adapter in `providers::ks` compares it with *the same provider's
  previous evaluation*, as the old watch did, and with the same "differs from the last teacher call" meaning under the
  surrogate. With ΔR the per-atom displacement between the two geometries and F_t, F_{t+1} the forces at each:

  ```text
  ΔF = F(R_{t+1}) − F(R_t)          (the energy change)
  W  = −½ Σ_I (F_I,t + F_I,t+1)·ΔR_I   (the trapezoid work estimate)
  m  = |ΔF − W|                        (the mismatch)
  b  = 3·(√(σ_E,t² + σ_E,t+1²) + ½‖ΔR‖·√(σ_F,t² + σ_F,t+1²)) + 3·κ_max·‖ΔR‖³
  ```

  (each evaluation brings its own tier; with equal tiers the noise term is 3(√2σ_E + ‖ΔR‖σ_F/√2); plan Task 8 review M4)

  - **σ_E and σ_F** are the surface's measured per-solve path scatters of the energy and of a force component, **at
    the tolerance the evaluation was solved at** (re-review R-MUST-A).
    - At the sample tolerance 1e-6 on the pseudo surface: 1.87e-6 Ha (H₂O warm vs cold) and `PSEUDO_FORCE_NOISE` =
      5e-5 Ha/Bohr. On the all-electron surface: its measured values (`references.md`; 5.4e-6 Ha for all-electron
      H₂O).
    - Live MD runs at the app tolerance raised by `effective_gpu_tol`. Its scatters are measured once in Task A4 and
      recorded in `references.md`. The guard picks the tier from `EvalStats::scf_tol_used`.
    - They are taken from the references, not fitted per scene.
  - **‖ΔR‖** is the Euclidean norm of the 3N-vector of displacements.
  - **The third-order term uses κ_max, always** (plan Task 8 fix round 1, review C1). κ_max bounds the trapezoid's
    third-order term, (1/12)|∂³E/∂u³| along the step's unit 3N direction u = ΔR/‖ΔR‖ (re-review R-MUST-A).
    - **κ_max = 1 Ha/Bohr³** is a global constant, not per element: (1/12)·6Da³ for a Morse bond with D = 0.2 Ha and
      a = 1.5 Bohr⁻¹, rounded up. The "M7 forces" OH gives (1/12)·E‴ ≈ 0.18, below it. Along one bond, ∂³E/∂u³ is the
      bond's V‴ times 1 when one atom moves, 2√2 when two equal masses move oppositely, ≈ 1.19 for O–H in MD; the measured
      (1/12)|V‴| at 0.8 R_e (O–H 0.40, N–H 0.35) give 1.13 and 0.99 with the 2√2, under 3κ_max = 3 (review M2).
    - **Superseded: a running median κ̂ = min(κ_med, κ_max)** over the last 8 judged, unflagged steps. The running
      median is biased low on large-amplitude bond motion and locks in, because flagged steps never enter it (Task 8
      review C1: a replica flagged 655 of 2999 steps of H₂ forming from rest at 4 Bohr). Morse V‴ vanishes at
      R_e + ln 4/a, steps there fill the window with m/‖ΔR‖³ ≈ 0.001–0.005, and the third derivative then rises 100–300×
      within a few steps into the well. The window and its constant are removed; the guard flags on the derived bound.
    - **What it can see** (rule 7; review I2). The reach is per atom, the bound uses the 3N norm: with N atoms each
      moving δ, b ≈ 3κ_max·N^{3/2}·δ³ + n. At the reach (δ = 0.1 Bohr) the third-order term is 3.0, 8.5, 15.6, 33.5 and
      94.9 mHa for N = 1, 2, 3, 5 and 10, so **a 12 mHa jump (the smallest section 4.7(e) documents) is not detectable on
      a step that moves three or more atoms near the reach**. MD steps are far shorter: an H moves about 0.013, 0.024 and
      0.041 Bohr per 0.25 fs step at 300, 1000 and 3000 K (b ≈ 1e-5–7e-3 Ha + n). At a typical 0.02 Bohr one-atom step
      on the pseudo surface, b ≈ 1.2e-5 Ha + 2.4e-5 ≤ 4e-5 Ha.
  - **The step is judged** only when the largest single-atom displacement since the previous evaluation is ≤ 0.1 Bohr:
    consecutive MD, relax or drag steps, and consecutive teacher calls during a fallback streak. Otherwise
    `branch_jump` is `None` (not judged).
  - **The history restarts** (the next evaluation is not judged) on:
    - an edit that resets the provider;
    - an evaluation on a different grid, box or box placement than the previous one. A box change alters the discrete
      surface by egg-box and wall terms, and would otherwise be flagged (re-review R-MUST-A).
  - **On m > b:** `branch_jump = Some(m)`, a warning in `EvalStats` ("F changed by m more than the forces' work: the
    SCF moved to another stationary state"), and **the surrogate refuses to train on that step**. That is exactly the
    role the occupation-pattern watch had.
  - **Spin measure.** For N↑ = N↓ the change of ∫|ρ↑ − ρ↓| between the two evaluations is reported with it, since the
    restricted ↔ broken-symmetry switch shows there.
  - **The constants' derivation** (the factor 3, κ_max = 1, the 0.1 Bohr reach) is reviewed by the physics reviewer in
    the plan. None of them may be loosened to make a test pass. (The window of 8 went with the running median, review C1.)
  - **Coverage, stated honestly.** The guard does not judge:
    - sampling-pass solves, which are of unrelated perturbed geometries (usually more than 0.1 Bohr from the previous
      solve; dense scans and teach path points closer than that are judged, review M8);
    - the surrogate's sparse audits, whose teacher calls are usually more than 0.1 Bohr apart (re-review R-NOTE-b).

    Under the surrogate, only fallback streaks are judged. Every other label relies on the leave-one-out machinery, as
    today.

### 6.4 The lone-atom composition rule (`providers::surrogate`, lone atoms)

`closed_spin_channels(z, all_electron, fractional)` asks whether an isolated atom's reference ends at a gap in each
spin channel, so that k ≥ 2 far-apart copies compose to k·E_ref. It becomes: the species has one allowed multiplicity
(Z = 1), **or** the reference's winning solve has no partial level (`ScfStats.partial` empty), on either surface.

- The `all_electron` argument goes. FD exposes an all-electron open shell as partial occupations, where aufbau hid it.
- "No partial level" means a gap of at least 27.6 kT around μ: 138 mHa at kT = 5 mHa [kT-5]. That is larger than the
  O self-polarisation gap (11.6–11.8 mHa) plus the largest box splitting of two copies' levels (1.8 mHa).
- At 5 mHa every accepted open-shell species' winning reference (B, C, O, F, Si, S, Cl) is the symmetric partial
  ensemble (V0), so all of them are declined in pairs; so is all-electron O on Accurate, whose gapped M = 1 winner
  has a 113 mHa gap, inside 138 mHa (TS 5e-6 Ha, partial levels in both channels; on Balanced it has none). Z = 1 and
  the gapped winners (N and P at M = 4, no partial level measured) compose.
- A polarised-integer open shell (the outcome of section 4.7(d) if it happens) shows partial levels at that gap, so it
  is declined, which is the safe side.
- `AtomReference::fractional` is recorded from `partial` as before.
- **Z = 1 is not "DFT-equivalent"** [NOTE-9]. The Z = 1 branch composes k far-apart H as k·E_ref(M = 2). A DFT solve of
  an even number of far-apart H at auto is the restricted ½/½ ensemble, tens of mHa above that sum. That gap
  pre-exists (the lone-atom doc's 4.6e-2 Ha at 6.4 Bohr). FD lowers the DFT side by a further 2 kT ln 2 per pair.
  "Compose to k·E_ref" is the rule's answer, not a claim that DFT gives it. The lone-atom doc says so.

### 6.5 Reference scan, MD, relax, surrogate

- The scan keeps the lowest F (section 4.5).
- `ForceProvider` returns F (through `Energy.total`).
- The integrator, FIRE and the quench need no change.
- The surrogate's teacher labels, audits and teacher gap are all F.
- Hirshfeld charges and bonds read the density, which is f-weighted as before.

### 6.6 What the user sees

| where | today | new |
|---|---|---|
| Readouts row | "E_KS" / "E_KS (valence, pseudopotential)" | "F (electronic free energy, T_el 1579 K)" / "F (electronic free energy, valence, pseudopotential, T_el 1579 K)" [SHOULD-4] [kT-5] |
| Readouts tooltip | — | "Kohn–Sham free energy F = E − T_el·S at the electronic temperature T_el = 1579 K (kT = 5 mHa). For a gapped molecule TS = 0 and F = E. Partly occupied levels (open shells, nearly degenerate frontier levels) carry TS > 0. The forces, the dynamics and the trained model all use F. T_el is a fixed numerical electronic temperature, independent of the thermostat; F does not include the nuclei's entropy." [SHOULD-4] |
| Branch-jump warning | — | "F changed by … more than the forces' work: the SCF moved to another stationary state (not trained on)" (section 6.3) |
| "E total" row and the graph | KE + E_KS | KE + F, so "E total" reads "KE + F: conserved in NVE without tools" |
| Energy breakdown | kinetic … nuclear, total | kinetic, external, Hartree, xc, nuclear, **E (internal)**, **−TS (electronic entropy, T_el 1579 K)**, **F** |
| Activity iteration energy | E_HF | F_HF (same row, renamed) |
| Solve log line | — | "−TS = −9.5 mHa", only when TS ≥ 1e-6 Ha (sign corrected 2026-11-16 at plan Task 12: the value printed is −TS, negative; the earlier string dropped the sign) |
| Orbital viewer | shared-shell rows | per-level f; a level with f < 1e-6 shows as empty; the `DegenerateShells` wording in `ui/dock/orbitals.rs` is rewritten |
| Compare with reference (PySCF) | forces, dipole | when TS ≥ 1e-6 Ha: "ours is at T_el = 1579 K with partly occupied levels; PySCF's is integer-occupied", next to the force deviations |

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
   recorded multiplicity when one is recorded. (That last check is a consistency check of the settings, not a state
   guard: under multiplicity auto the electron count's parity fixes M. The class decides the state; plan Task 14 fix
   round 1, M1.)
   - A sample that fails is **not migrated**. It is listed with its reason in `migration.txt` and stays in the old
     folder.
   - Its old label is never carried: it belongs to the old surface.
   - Refused without a solve (plan Task 14): a sample with no neighbour within the model's cutoff (lone atoms teach
     nothing; an import refuses them too), a status code no collection holds, a forced multiplicity. A device error
     ("GPU busy") is not the sample's: the cursor stays and the pass retries, or stops resumable.
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

Thresholds, per sample, from the teacher's noise model (plan Task 14 fix round 1, physics review I2):
**ε_E = max(1e-5 Ha, 5σ_E)** and **ε_F = 3σ_F** (Ha/Bohr per force component), with σ_E, σ_F the per-solve path scatter
the `branch_jump` guard uses (`providers::branch_jump::noise_tier`): by surface (pseudo or all-electron), by the
tolerance the new solve stopped at (a GPU solve with a partly occupied level stops at `GPU_PARTIAL_TOL` = 1e-5, the app
tier), and σ_E = max(the tier's, c·|E_old|), c = 9e-8 pseudo, 1.0e-7 all-electron.

| | ε_E | ε_F |
|---|---|---|
| pseudo, sample tier, \|E\| ≲ 22 Ha (H₂, H₂O, OH) | 1e-5 | 1.5e-4 (3 × `PSEUDO_FORCE_NOISE`) |
| pseudo, 35 Ha | 1.58e-5 | 1.5e-4 |
| pseudo, 70 Ha | 3.15e-5 | 1.5e-4 |
| pseudo, partial stop (app tier) | as the sample tier's (its σ_E 1.6e-6 is below the floor) | 2.7e-4 |
| all-electron, sample tier, \|E\| ≤ 54 Ha | 2.7e-5 | 7.2e-5 |
| all-electron, 70 Ha | 3.5e-5 | 7.2e-5 |
| all-electron, partial stop (app tier) | as for its \|E\| (2.35e-5 up to 47 Ha) | 7.2e-5: the larger σ_F of the two tiers (the app tier's 1.4e-5 is gapped H₂O's; an all-electron partial stop is unmeasured) |

A fixed 1e-5 was 1.6σ_E at 70 Ha (about a quarter of gapped samples falsely S) and below 2σ_F of a partial stop. The
classes decide the review only: a T or S label is always the new solve's.

| class | condition | meaning |
|---|---|---|
| **U**, unchanged | \|F_new − E_old\| ≤ ε_E, and every force component within ε_F | a gapped state, the same label within solver noise |
| **T**, entropy only | not U, \|E_new − E_old\| ≤ ε_E (the internal energy), every force within ε_F | the same electronic state (an equal-share ensemble before and the same f now); only the reported quantity changed, by −TS |
| **S**, state changed | anything else | the old rule had a different state here: e.g. a released integer pattern where FD now pins the frontier at μ (H₆ seed 5: E −3.148291 before, F −3.148504 now), or a different SCF branch |

- **ε_E's floor** [SHOULD-7], 1e-5 Ha, is 5× the largest measured path scatter on the pseudo surface (1.87e-6 Ha,
  H₂O warm vs cold, `references.md` "SCF, GPU against CPU") and 100× below the model's 1 mHa/atom energy bound; above
  it, 5σ_E of the noise model (table above).
  - The earlier "10× the 6e-7 Ha scatter" cited the H₂ number, which is the wrong surface for an H/N/O collection.
  - Because the margin is thinner on O-containing samples, V11 reports the fraction of class S whose only failing test
    is |ΔE| ∈ (ε_E, 3ε_E] as the false-S estimate.
- **ε_F** is 3σ_F: on the pseudo sample tier 3 × `PSEUDO_FORCE_NOISE` (5e-5 Ha/Bohr, the teacher's measured path
  scatter), the tier an audit already judges against.
- **Both errors are safe.** A false "changed" costs one review (and re-judges a misfit hold or a lifted fence). A false
  "unchanged" keeps a label within ε_E of the new one.
- **Class S can replace a lower-F label** (rule 7; plan Task 10 review M4, V7). At V7's near-square H₄ (aspect 1.005)
  the shared-shell rule released the e pair into the broken-symmetry integer state, E_DS = −2.028985 Ha; the cold
  Fermi–Dirac solve reaches the symmetric ensemble, F = −2.014486 Ha, 14.5 mHa higher (section 4.7(e)). The new label is
  the state the new teacher converges to from its cold start, not the lowest state at that geometry. V7 checks that its
  TS is the symmetric ensemble's (within 3 % of the fixture's 1.3836e-2 Ha), so a broken-symmetry landing fails loudly.
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
  - the label moved by exactly the defined −TS term. That term is C^∞ but steep near a degeneracy [NOTE-5]: 2 kT ln 2 =
    6.93 mHa for a linear OH at kT = 5 mHa [kT-5], falling to 0 over a range of width ~kT/|dΔ/dR|, a few degrees of bend. The fit sees it
    as any other feature of the surface.
  - The refit after migration judges it like any fitted sample.
- **S.** The re-solved label is appended with the old metadata and status code **9, `Q_RESOLVED`** ("re-solved for the
  new solver, awaiting review"; code 8 is `purge-forgotten`'s "forgotten, fence lifted", section 8). Two exceptions keep
  their old code: a forgotten sample keeps code 7, because the user's decision outranks the review, and a superseded one
  keeps code 2, because what superseded it is migrated too. A forgotten sample whose fence was lifted (code 8) goes to
  7: the Forget survives, and the fence returns, because the lift judged the old label (under U and T it keeps 8).
  An old hold (code 1) does not survive into class S: it judged the old label, so the sample goes to review instead.
- **`references.txt`**: the re-solved references only (section 7.3). Old references are never carried.
- **`history.bin`**: not carried. The audit history measured the model against the old teacher, and a calibration
  built on it would mix surfaces.
  - The new folder starts an empty history.
  - Without evidence, the calibrated class verdict falls back to its no-evidence rule (`calibrate` module doc), so the
    first refits are more conservative until audits accumulate. The summary says so.
- **`bond_experience.txt`** (display only, safe to delete): copied unchanged [NOTE-9].
  - It holds bond verdicts (Z, neighbour counts, R, bonded), not energies.
  - Those verdicts came from old-surface densities. They differ only where levels were partly occupied, and they guess
    a display only.
  - The README paragraph says so.
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

### 7.6 Code 9 and its review: automatic, by the leave-one-out rule

A `Q_RESOLVED` sample is out of the fit, the envelope, the grade, the leave-one-out and the class statistics, like a
held sample.

- It is **not** counted against the trust recovery's 10 % held cap: it is not a misfit hold.
- It is **not** the user's: Forget and Restore do not apply to it.
- **Before its review it does not fence.** The fit simply has no data there. Old labels never enter the new folder, so
  the extrapolation grade and novelty rules already send nearby geometries to DFT.

**The review** runs inside a refit once the migration pass is complete, so every sibling of a sample is present before
any is judged. It repeats at every later refit while code-9 samples remain and the store has changed since the last
review.

1. Fit a candidate on the current fit's members **plus every code-9 sample**.
2. For each code-9 sample, compute its **leave-one-out ratio in that candidate**: the reading the trust recovery
   already ranks offenders by (`quarantine::loo_ratio` today): how far the fit without the sample misses it, against
   its own energy and force bounds.
3. A sample with ratio ≤ 1 is **released** (9 → 0). The refit then proceeds with the released samples as ordinary
   fitted samples, and the ordinary trust machinery (residuals, recovery, misfit) judges the result as it judges any
   new data.
4. A sample with ratio > 1 stays at 9. From then on it **is treated as a stored label the model cannot reproduce**:
   - it fences like a held sample (the quarantine fence, `fences()`; section 8, preflight F4). It never joins the
     misfit set M, is not counted by `fencing()`, and `purge-forgotten`'s lift votes never lift it;
   - a later review can still release it;
   - like a held sample (code 1), one fresh teacher answer next to it that agrees with the model settles it: it is
     released when the model reproduces it, else superseded (controller ruling B, plan Task 15; `admit`'s release and
     supersede of the fencing samples next to an agreeing answer). Purge's lift votes still never lift it: they lift
     forgotten samples only, and a fence that also holds a non-forgotten sample casts no lift vote;
   - it is never converted to code 1, so it never enters the recovery's holds or its cap. The trust rework's "no
     ratchet" invariant (G2 of its spec) stays intact.

**Not judged (plan Task 15, review M1).** "Every code-9 sample" means every one a candidate can hold:
- **A sample with stray atoms** (atoms with no neighbour within the cutoff) is not judged. A stray composition is
  never fitted unmeasured (the stray-atom spec, 2026-11-18), so it cannot be in the candidate. It waits at 9: out of
  the fit, not fencing, never reviewed. `migration.txt` lists it on a `waiting` line, the review's note counts it
  ("N waiting (stray atoms)"), and the state stays "complete" while any waits; "reviewed" means every code-9 sample
  has a verdict. This is a known limit. A follow-up could judge it by plain prediction and release it to the
  stray-atom hold rather than to 0.
- **A sample whose box lacks its isolated-atom references** waits too: its energy is the teacher's raw one. The
  reference's arrival rebases it, which changes the store, so the next refit reviews it.

The review also repeats when the bounds it reads (the policy's energy and force bounds) or the basis (degree cap,
body-4 degree) change (review M3). Its note appears only when the verdicts change (M4).

**Why this rule.**
- *Automatic and rule-based, as the user prefers.* It uses the trust machinery's own reading, the leave-one-out ratio,
  against the policy's own bounds, with nothing new to calibrate.
- *Candidate fit with siblings, not the plain release-by-reproduction of a held sample.* When a whole region changed
  together, such as every near-linear OH or every auto-multiplicity O₂, a fit without any of them cannot reproduce
  any of them, so release-by-reproduction would never fire. With the siblings in the candidate, each is judged by
  whether its neighbours predict it, which is the same question the class rule asks.
- The opposite error is the single-pass screen's swamping: gross outliers among the code-9 samples inflate their
  neighbours' ratios. Measured on the analytic fixture (plan Task 15): ten consistent siblings read 0.02–0.11 alone,
  0.42–0.55 beside one 50 mHa outlier, and 0.89–1.02 beside two, so 3 of 10 were kept. The error is on the safe side
  (kept samples fence, so DFT answers nearby) and heals itself: fresh answers there join the next candidate, and
  ruling B settles a kept sample at the first agreeing answer. An iterated screen (refit without the kept ones,
  re-judge them) is a possible refinement.
- A set of mutually consistent but wrong re-solves would pass. A wrong re-solve here means a teacher pathology, and the
  solver checks in section 7.3 step 2 guard against that. The released samples then face every check any fitted
  sample faces.

**Cost.** One extra fit plus a leave-one-out pass per refit while code-9 samples remain. That is about 1 s at
1 700 samples (`references.md`, leave-one-out recovery ranking). There is none once all are reviewed.

### 7.7 Formats and older builds

- **Collection records: version 1 → 2** (`surrogate::collection::RECORD_VERSION`, the `collection` entry of
  `app::update::formats`). [F1] The numbers in this section assume `main`'s version 1; they are symbolic: the new
  version is the merged `main`'s + 1, and the sidecar's new magic is the merged `MAGICS.len()` + 1 (plan Task 2).
  - Every record the new build writes is version 2. That changes the meaning of `energy` (F on an FD surface) and
    allows status code 9. The byte layout is unchanged.
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
  - Code 9 is never written into a sidecar: a sidecar writer leaves code-9 samples out, because the collection is the
    store of record. So 9 is admitted for collections only (`is_collection_code`); `is_persisted_code` keeps purge's 8.
- **`history.bin` (`ASAUDIT1`): unchanged.** Same layout, same meaning: the model against the teacher of the folder's
  own surface.
- **`pseudo_cache`: no format change.** H/He entries are re-keyed by the moved `recipe_hash` (a cache, rebuilt).
- **`checkpoint`** (`training/pass.ckpt`): unchanged format. An interrupted old pass carries the old provenance, so its
  resume is refused with the reason ("started with the previous solver"). The file is left as it is.
- **The downgrade warning** (`app::update::formats::downgrade_warnings`) gains one line. For each new folder whose
  `migration.txt` names a source folder that still exists:
  "<version> will use <old folder>, the collection as it was before it was re-solved for the Fermi–Dirac solver, without
  what was learned since. Both folders stay on disk."
- **`surrogate::quarantine` docs** gain "an older build meeting code 9", in the style of code 7's paragraph. Its
  content is short: no older build can read a version-2 record, and no sidecar holds 9.

### 7.8 What the user sees

**The offer** (Model tab, collection section), filled from the predecessor's counts:

> **This collection was taught with the previous solver.** Atomic Sandbox now fills electron levels with Fermi–Dirac
> smearing at 1 579 K, which fixes SCF stalls on loose clusters. Your collection "H, N, O · 2 412 samples"
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
> References re-solved: O −TS = −9.5 mHa (or as measured; sign corrected 2026-11-16 at plan Task 12). The audit history starts empty, so trust is rebuilt from new
> audits.

The wording says "identical within solver noise", not "identical". It says "kept out", not "discarded". It names the
old folder.

As built (plan Task 16, `app::ui::migration`):
- Code-9 samples the review cannot judge (stray atoms, §7.6) get their own clause in the state-changed line, in the
  same style: "…; 2 waiting (stray atoms)". State-changed samples with no verdict and not waiting (they kept code 7
  or 2, or wait for an isolated-atom reference) are "N not reviewed (…)". Before the first review the line says what
  the review will do.
- The references line reads "References re-solved (the new F against the old E): O −9.5 mHa": the log holds the old
  and new values only, and their difference is −TS only where E did not move, so it is labelled as what it is.
- The not-migrated reasons are worded from the pass's reasons ("stalled", "now box-affected", "not converged", …).
- Numbers are grouped with a space as above ("2 412", "1 579 K"); times are "1 h 30 min", to 5 min from an hour on.
- **Later** folds the offer into one line of the Model tab's *Model file* section, which keeps Re-solve now; there the
  section shows the progress line while the pass runs, "stopped at sample k of n … It goes on from there at the next
  start of the app, or now with Resume" for a stopped pass, and the summary once complete. A stopped pass is started
  again by the worker when it next attaches the collection (§7.3, "Resume").
- The `README.txt` paragraph is not written yet (the collection's README writer is `surrogate::collection`).

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

1. **The set of codes excluded from the fit** (today, the store's "excluded" predicate): add code 9.
2. **The fence / near-misfit refusal predicate**: add a reviewed-and-failed code-9 sample (section 7.6, step 4) to the
   quarantine fence `fences()`, not to the misfit set M (preflight F4, below).
3. **The held-cap count of the recovery**: confirm that code 9 is not counted.
4. **The refit pipeline**: one review step (section 7.6), placed before the trust judgement. It reuses the recovery's
   leave-one-out ratio, whatever name it has on `main`.
5. **The lone-atom rule's "ends at a gap" test** (section 6.4).
6. **The shared-collection attach**: predecessor detection and the migration pass hook (section 7.2–7.3).
7. **The policy's label-noise prior** (`force_noise_prior`): it reads "all-electron" from the tag through
   `is_all_electron_tag`.
8. **The occupation-change training refusal**: replaced, at the same place, by the branch-jump refusal (section 6.3)
   [MUST-3].

The plan re-reads these against `main` after the merge. If the rework changed what one of them means, the plan records
the mapping, and this spec is amended rather than reinterpreted silently.

**Amendment (preflight scan, 2026-11-14): `purge-forgotten`.** That branch also merges to `main` first. It takes code 8
for `Q_FORGOTTEN_UNFENCED` ("forgotten, fence lifted": a forgotten sample whose fence three agreeing answers lifted),
and it can delete forgotten samples from `samples.bin`, renumbering the rest.
- **Codes and formats (F1).** `Q_RESOLVED` is **9** everywhere in this spec. The collection record version and the
  sidecar magic are the merged `main`'s values + 1 (section 7.7).
- **Code 8 in the migration (F2).** Under U and T a code-8 sample keeps 8. Under S it goes to 7 (section 7.5): the
  Forget survives, and the fence returns, because the lift judged the old label.
- **Renumbering (F3).** A purge renumbers the new folder, so `migration.txt`'s review log (section 7.6) is remapped
  through `PurgeOutcome::map`. A purge is refused while a migration pass runs. `migration.txt` records the source
  `samples.bin`'s length and hash; on a mismatch at resume the pass restarts from 0. The read-only open (section 7.3)
  never cleans purge's `samples.bin.new` / `samples.bin.bak.new`.
- **The fence for a failed review (F4).** A reviewed-and-failed code 9 joins the quarantine fence (`fences()`), like a
  held sample. It stays out of the held cap and out of `fencing()`, purge's lift votes cannot lift it, and it never
  joins the misfit set M.

**Amendment (plan Task 2, 2026-11-14): the touch points on merged `main` (fe30578, merge 97da009).** The names are
in the plan's Touch-point map. Roles 1 and 3–10 mean what this section says; on `main` code 9 is excluded from the
fit (`excluded` is "code ≠ 0"), left out of the held cap (`quarantined() + forgotten()` counts codes 1, 7 and 8),
kept out of M (M is built from fitted samples) and never lifted (the lift takes code 7 only) without a code change.
The formats: R = 1 and K = 8, so the new collection record version is 2 and the new magic `ASMODEL9`; the old
sidecar bodies are 6, 7 and 8 (section 7.7's numbers hold as written). Two points of role 2/10 do not read as
preflight F4 assumed, and need the controller's ruling before plan Task 15 (no meaning is chosen here):
- **`fencing()` is the fence's on-switch.** Its only readers are the early returns of the quarantine distance scan
  and the views' "any fence" checks. A failed code 9 in `fences()` but not counted by `fencing()` would fence nothing
  whenever no code-1 or code-7 sample exists. So "not counted by `fencing()`" (section 7.6 step 4, F4 above)
  conflicts with "joins the quarantine fence". `fencing()` feeds no cap.
- **A fresh agreeing answer moves fencing samples.** `admit` releases (when the model reproduces it) or supersedes
  every non-forgotten fencing sample next to a fresh answer that agrees with the model. Section 7.6 does not say
  whether a failed code 9 is moved that way (like a held sample) or only by a later review (like a forgotten one).

**Controller rulings on the two points (plan Task 15).**
- **A:** a reviewed-and-failed code 9 fences. The readers that returned early on `fencing() == 0` (the quarantine
  distance scan, cached or not, and the views' "any fence" checks) read a separate "any sample fences" predicate
  (`any_fences()`) that includes it; `fencing()` stays the count of codes 1 and 7, so code 9 stays out of it and out
  of the held cap, as F4 says.
- **B:** a failed code 9 is moved like a held sample (code 1): one agreeing fresh teacher answer releases or
  supersedes it through `admit` (section 7.6 step 4). Purge's lift votes still never lift it.

## 9. Validation (rule 6; one heavy job at a time)

Every gate in CLAUDE.md's table applies, because this change touches the solver, dynamics, the surrogate, pseudo
scenes, the orbital display and the viewer. The full serial set is about 4.5–5 h. No tolerance is loosened. Where a
pinned number moves, the reason is the TS term or the re-pinned bitwise path, and that is stated beside the number.

| # | suite / command | what it must show |
|---|---|---|
| **V0** | **Task A1 measurement (entry gate)**, before production code. The diagnosis's env-gated `SCF_SMEAR` path on the GPU (1e-3, then 5e-3 after the ruling), plus a CPU f64 cross-check of each species' worst case. | See "V0 in detail" below. |
| V1 | `cargo test -p units -p qm -p sim -p surrogate --release` | **FD unit tests:**<br>• two levels with N = 1, degenerate (f = ½, μ = ε, S = 2 ln 2);<br>• a gap Δ (f = 1/(1 + e^{±Δ/2kT}) at the symmetric μ);<br>• a gap ≥ 74 kT [SHOULD-5]: f below μ exactly 1.0, f above μ ≤ e^{−36}, S ≤ 1e-14, μ the plateau midpoint of section 4.1 (both bisections) [MUST-1];<br>• N_σ = 0; the bracket's validity;<br>• a NaN eigenvalue → `QmError::Numerical` before bisecting [SHOULD-1];<br>• the worst-case halving count near μ = 0 stays under the backstop;<br>• determinism (same bits twice);<br>• s(f) against the direct formula on a grid of f, plus the 0 ln 0 limit.<br>**Witness rule**, both arms. **Energy invariant** (total = internal − ts).<br>**CPU H₄ square a = 1.8** (41³ × 0.30, axes-aligned, exactly degenerate, tol 1e-9): TS = 4 kT ln 2 within 1e-8 Ha, and E equal to the DS-era E within 1e-8 Ha (same f by symmetry) [SHOULD-6].<br>**FD ≡ Aufbau** on H₂ and H₂O (CPU f64, tol = eig_tol = 1e-9): \|ΔE\| ≤ 1e-8 Ha, since the f agree to e^{−100} [SHOULD-6].<br>The DIIS-reset test (section 5). The collection: v2 records, code 9, the read-only open leaving every file byte-identical. |
| V2 | **Regression (new; `cargo test -p providers --release`, GPU, release-only).** The diagnosis harness reduced and committed. Scenes go through `providers::ks::problem`, cold, at 81³ × 0.15, the app's tolerance raised by `effective_gpu_tol`, stall stop on. | See "V2 cases" below. |
| V3 | `cargo test -p validate --release --test oracles` | **σ → 0 oracles** (H, He, H₂): numbers unchanged within the existing tolerances, and **TS ≤ 1e-12** asserted on every oracle system (they are gapped or single-electron).<br>**New FD-consistency oracle** (CPU f64, 41³ × 0.30, tol 1e-9, eig_tol 1e-10): see "V3 force oracle" below. |
| V4 | `cargo test -p validate --release --test gpu_vs_cpu` | Existing cases within existing bounds. New cases at the suite's fixed σ = h = 0.25, 49³: H₄ square (degenerate), and a lone pseudo-O at each M V0 settled on [MUST-5]: M = 3 and M = 1 at 5 mHa [kT-5]. The GPU side stops at the section 5.1 partial-level 1e-5 and the CPU at the suite's tol; V0 measured the stop's own error on the pseudo surface at ≤ 1.7 µHa and ≤ 3.2e-6 Ha/Bohr, inside the bounds below, which do not move. Each case checks:<br>• \|F_gpu − F_cpu\| and \|TS_gpu − TS_cpu\| within the suite's energy bounds (TS is second order in Δε: \|ΔTS\| ≲ Δε²/(16kT) per pair);<br>• μ_σ within `EPS_BOUND_PSEUDO` (1e-5 Ha) where a partial level exists, and not compared in gapped channels (section 4.1);<br>• per-level \|Δf\| ≤ 2·`EPS_BOUND_PSEUDO`·max f(1−f)/kT = `EPS_BOUND_PSEUDO`/(2kT) = 1e-3 at kT = 5 mHa (5e-3 at 1e-3) (re-review R-NOTE-a) [kT-5]. The earlier EPS/(4kT) was exact only for a pair; a three-level shell at f = ⅓ can reach 0.30·EPS/kT under an adversarial error pattern. The measured \|Δε\| = 2.2e-6 gives \|Δf\| ≤ 1.3e-4 at 5 mHa;<br>• the sum of f over each degenerate cluster within 1e-9. A cluster is the levels the CPU finds within 1e-6 Ha of each other;<br>• **both solvers in the same state class** (symmetric vs polarised, by TS against the closed form), so that an f32-vs-f64 split onto different stationary states is detected, not averaged over.<br>Forces are checked against the existing force bounds. |
| V5 | `cargo test -p validate --release --test dynamics` | Existing tests unchanged (gapped H₂ and H₃⁺: the conserved quantity is F = E). See "V5 new tests" below. |
| V6 | `--test pseudo`, `--test pseudo_gen`, `--test pseudo_oracles` | **Generation records:** bitwise unchanged, with `recipe_hash` moved by `DIMER_PROCEDURE` only.<br>**Dimer-gate verdicts and R_e / ω:** reproduce within the backend margin.<br>**Lone-atom references and the O multiplicity scan** (`pseudo_o_triplet_matches_cpu`, `gpu_oxygen_reference_multiplicity`; `degenerate_spread_at_low_symmetry_sites` repurposed as V0's site study): these move by the measured −TS. Each moved reference is stated in `references.md` with its TS and, for an exactly degenerate shell, the closed-form value (section 4.3). The scan keeps the same M (lowest F). |
| V7 | `cargo test -p validate --release --test surrogate` | Passes, with labels now F; the H₂ and H₃⁺ tests are gapped, so their pins hold.<br>A model file or collection with a legacy tag is refused, and its collection is found as the predecessor.<br>**Migration integration test** (GPU, small): a collection of H₂ samples (U), linear OH samples (T: ΔF = −2 kT ln 2, E and forces unchanged) and one near-square H₄ (S, reviewed). It checks the counts, the codes, the review's release, and the old folder hashing identical before and after.<br>**Branch-jump refusal** [MUST-3]: a teacher answer stream with a synthetic stationary-state switch (the same geometry path, F shifted by 12 mHa from one step on) is warned and not trained on; the steps before and after it are trained on. |
| V8 | `cargo test -p providers --release` | The provider unit tests.<br>The tag strings (`none/fd5e-3/pin1e-5`, `…/fd5e-3/stop2x…/pin1e-5`) [kT-5]. `legacy_tag_of` pinned against the recorded strings. The re-pinned bitwise tests, each with its reason.<br>The lone-rule test on a polarised-or-partial O (declined).<br>**`branch_jump`** [MUST-3]: a synthetic jump is flagged; a smooth analytic path (a Morse dimer stepped at MD-sized displacements) is never flagged; a displacement > 0.1 Bohr is not judged; a relax-like sequence of 1e-3 Bohr steps followed by a 0.02 Bohr step carrying a synthetic 12 mHa jump is flagged; a stream of alternating jumps stays flagged at every jump; a grid or box change restarts the history and is not flagged (re-review R-MUST-A).<br>**Witness cap** [NOTE-6]: all-electron H⁻ at the app's largest box. |
| V9 | `--test tools`, `--test charges`, `--test orbitals` | Density and orbital consumers. Gapped systems leave the numbers unchanged. The orbital display shows per-level f. |
| V10 | `cargo test -p app --release` (incl. replays) | Label tests (`energy_label` → the F wording).<br>Formats (`collection` = 2, the sidecar bumped, the downgrade line for a migrated folder).<br>The offer / progress / summary text from a fixture predecessor.<br>The replay scripts (they pin quality, walls and ionic; any energy expectation is gapped). |
| V11 | **Dry run on a copy of the user's collection** (measurement, `#[ignore]`d, never on the original) | Counts per class, the class U maximum deviations, the time per sample, and the review outcome (released / kept out). Then the refit's trust verdict against the pre-migration one, recorded in `references.md` "Fermi–Dirac migration".<br>**The false-S estimate** [SHOULD-7]: the fraction of class S whose only failing test is \|ΔE\| ∈ (ε_E, 3ε_E].<br>**Must:** the source copy is byte-identical afterwards, and no sample changes code without a logged reason.<br>**Reported, not gated:** whether the migrated collection is trusted as before. If it is not, the reason is investigated before merge. |

**V0 in detail** [MUST-2, SHOULD-6, SHOULD-9].

**Outcome** [kT-5] (the V0 note). At 1e-3 (2026-11-13) V0 failed and the user ruled (i): kT = 5 mHa, the scan's stop
at 1e-5 (section 2 item 3; the stop's scope is section 5.1). Re-run at 5 mHa / 1e-5 on 2026-11-14 over B, C, N, O, F,
Si, P, S, Cl, all-electron O (M = 1, 3, 5) and the molecules below:
- **States, site spread, same winner: pass everywhere.** Winners B 2, C 3, N 4, O 3, F 2, Si 3, P 4, S 3, Cl 2,
  all-electron O 1, at every site on both grids, equal to the DS winners.
- **Convergence (≤ 60 iterations): two misses.** S at M = 1 on Accurate (61 iterations at one site), and all-electron
  O at M = 5 (over 60 iterations at 2 of 4 sites on Balanced and 3 of 4 on Accurate, 5 of 8; of them not converged in 100
  at 3: Balanced centre and generic, Accurate node; a diffuse ↑ level manifold at ≈ −0.02 Ha among the box states,
  TS 19–21 mHa ≈ 2× the three-level closed form 3kT·s(⅓) = 9.5 mHa, 670 mHa above the winner; plan Task 11's CPU f64
  cross-check: the Balanced generic site does not converge in f64 either, a slow mode of the SCF map and mixer on this
  state, not f32 noise; references.md "V0's two accepted convergence misses on the CPU f64 path"). Neither is a winner; the scan leaves a failed
  multiplicity out.
- **Jitter (floor ≤ stop/3 = 3.3e-6): fails for the M = 1-type states** (O, C, S at M = 1 on both grids, Si at M = 1
  on Accurate, N and P at M = 2, all-electron O at M = 5, one Si M = 3 site, the O₂ singlet and NH at auto): medians
  1e-6 to 2.8e-5. Every scan winner, OH and every gapped case passes, by 4.5× to > 1e4×.
- **Molecules:** OH (along an axis and off-axis), the O₂ triplet, H₂O and NH at M = 3 pass every rule; the O₂ singlet and
  NH at auto pass all but the jitter rule (DS itself stalls on singlet NH at every site).
- **Witness:** fits (at most 15 of 16, section 4.2).

These misses are what remains for the user: the pre-registered rules, applied as written, still fail at 5 mHa for
non-winning M = 1-type states (and two convergence misses). Nothing here is re-bounded.

**Stated up front: V0 may well fail at M = 1 with kT = 1e-3** (section 4.7(d), re-review R-SHOULD-D). On the spec's own
premises, the M = 1 spin-like occupation mode of an open p shell has a stability threshold of kT*_s ≳ 5 mHa, five
times the user's kT. The symmetric M = 1 ensemble is then a saddle of F, and the SCF may polarise, become
site-dependent, or fail to settle. kT stays at 1e-3 Ha in this spec (the user's decision; superseded on 2026-11-14 by 5 mHa, see the outcome above). If V0 fails, the
pre-registered stop below applies and the user chooses among (i)–(iii).

- **Cases.**
  - Every species the recipe accepts with an open-shell ground state: pseudo C, O and F, and B if accepted, with N as
    the gapped control.
  - Each at every multiplicity the reference scan solves.
  - Each at four sites (the node, the cube centre, (0, h/2, h/2) and a generic site) and on both grids (h_req and
    Accurate).
  - All-electron O at M = 1 and M = 3 (Balanced and Accurate).
  - Plus OH (bent, linear), the O₂ singlet and triplet, H₂O, and NH at auto.
- **Proceed, per species, if all three hold:**
  - **Convergence.** Every case converges within 60 iterations without the stall stop.
  - **Site spread.** For each species and M, the site spread of F does not exceed the `DegenerateShells` site spread of
    E, measured the same way, by more than **6σ_path + 1e-6 Ha**.
    - σ_path is that species' GPU per-solve path scatter, measured in V0 itself by a warm and a cold solve at each
      site. The factor 6 covers the difference of two spreads, each from 4 solves, at 3σ.
    - The 1e-6 Ha term bounds the derived lattice-split differences between the FD and the equal-share f: ≈ Δ·δf ≈
      5.3e-5 × 0.0066 = 3.5e-7 in E at 1e-3 (5.3e-5 × 0.0013 = 7e-8 at 5 mHa), and second order in TS.
    - Comparing like with like cancels the egg-box. The DS failure V0 guards against was 4.1 mHa.
  - **Same winner.** The reference scan's winning M is the same at every site.
- **Record** for each case:
  - which state it reaches: symmetric (TS ≈ the closed form) or polarised (TS ≈ 0);
  - U↑↑, U_c and U_s from the integer states' splittings, against the section 4.7(d) predictions;
  - the GPU eigenvalue jitter and its implied Δρ floor, which must sit a factor 3 below the GPU stop rule (otherwise
    V0 fails);
  - the lopsided-tetrahedron seeding of section 4.7(e) [SHOULD-2].
- **If any species or multiplicity fails, stop and ask the user.** As stated above, M = 1 is the expected place for a
  failure. The user rules on one global kT for all species (rule 1). None of these options is chosen by me:
  - **(i)** Raise kT to the measured stability threshold of the worst species and multiplicity. The predictions are
    ≈ 3 mHa for O at M = 3 (pseudo spec §5.7), ≈ 5 mHa for the M = 1 spin mode, and higher for F. It is measured as
    the smallest kT that passes every rule above. It costs a larger F bias on stretched and pinned states (TS and the
    stretched-bond energies scale with kT).
  - **(ii)** Keep 1e-3 and document the site dependence of the failing open-shell species as an LSDA symmetry-dilemma
    artefact, with the lone-atom rule already declining such atoms in pairs.
    - Option (ii) still requires the reference scan to converge at every M at every site [SHOULD-9]. A lone-atom
      reference that stalls blocks every sample of that species, so (ii) is available only if the convergence rule
      above holds.
  - **(iii)** [NOTE-3] Raise the width with Marzari–Vanderbilt cold smearing instead of FD. The failure is set by the
    occupation function's slope at μ (FD: 1/(4kT)), not by FD as such.
    - Cold smearing has non-negative occupations, a variational generalised free energy, and an energy bias of O(σ³)
      for non-degenerate pinned states. That answers (i)'s cost.
    - It keeps a −TS-like term at exact degeneracies.
    - It departs from the user's FD choice, so it is only offered.

**V2 cases.** Coordinates are D§2.1's centred ones; the rhombus and tetrahedron as listed in D§2.

**[A1, 2026-11-14] Amended: measured at kT = 5 mHa** with the prototype (plan Task 3 step 0; the A1 note,
`docs/superpowers/notes/2026-11-13-a1-measurements.md` section 0), cold, 81³ × 0.15, the app's tolerance 1e-5 raised
by `effective_gpu_tol` (1e-5 for every case; so the section 5.1 partial-level stop is the same number), stall stop
on. Every case converged without the stall stop or the cap. The 1e-3 counts and D§6's F/TS values are superseded.

| policy | case | measured at 5 mHa (iterations) | budget | F (Ha) | TS (Ha) |
|---|---|---|---|---|---|
| default | H₄ rhombus | 12 | ≤ 20 | −2.079322 | 0.000027 |
| default | H₄ lopsided tetra | 16 | ≤ 25 | −1.943830 | 0.013889 |
| default | H₄ seed 1 | 13 | ≤ 20 | −2.078602 | 0.002076 |
| default | H₅ seed 3 | 15 | ≤ 25 | −2.564249 | 0.003006 |
| default | H₈ seed 8 | 14 | ≤ 25 | −4.220394 | 0.000163 |
| `ionic none` | H₅ ring (radius 1.531, xy plane, vertex phase as the harness builds it) | 11 | ≤ 20 | −2.470682 | 0.006932 (= 2 kT ln 2) |
| `ionic none` | H₆ seed 5 | 15 | ≤ 25 | −2.973255 | 0.009715 |
| `ionic none` | H₂O + 2 H (its own test function, section below) | 33 | ≤ 50 | −47.350019 | 0.013163 |
| `ionic none` | lopsided tetra | 15 | ≤ 25 | −1.824952 | 0.013978 |

(`ionic none` runs with the witness configuration, `DegenerateShells` + FD, as production will. TS is E − F of the
printed six-decimal E and F, so it carries ±1e-6 of rounding. The two lopsided tetrahedra exceed 4 kT ln 2 =
0.013863, the most one pinned pair per channel can give: a third level lies near μ.)

- Each case must **converge** (no stall stop, no cap) within its budget, ⌈1.5 × the measured count⌉ rounded up to a
  multiple of 5 (the "about 1.5×" rule, unchanged).
- **F and TS** must match the table within 1e-5 Ha (the values are printed to 6 decimals, a ±5e-7 rounding inside
  that tolerance).
  - Production runs the prototype's witness rule (section 4.2), so this comparison is like with like.
- **All-electron H₂O + 2 H is its own test function** (review fix round 1): it costs 434 s of GPU at 5 mHa
  (13 s per iteration), so it runs as a separate GPU batch from the other cases, and it logs the extra (witness)
  vectors each channel grows, so that its cost can be explained.
- A mismatch is investigated, never re-pinned blindly. Once checked, the pins are the production values.
- **The controls** [SHOULD-6]: H₂ and H₂O must keep the D§6 energies within 3 × max(achieved, path scatter): H₂ 2e-6 Ha,
  H₂O 6e-6 Ha (the suite's rule). Production changes the block size (witnesses), and so the bits. At 5 mHa they are
  unchanged to every printed digit (H₂ −1.136119 in 9 iterations, H₂O −17.476023 in 13; TS 0 and 1.4e-12).
- What moved from 1e-3 (A1 note section 0): the partly occupied cases converge in fewer or equal iterations; TS
  scales by 5 for the exactly degenerate H₅ ring and non-linearly for the pinned ones; seeds 1, 3 and 8, gapped at
  1e-3, now carry a partial level (TS 2.1e-3, 3.0e-3, 1.6e-4). All-electron H₂O + 2 H costs 434 s of GPU (13 s per
  iteration), the suite's slowest case.

**V3 force oracle** [MUST-4].
- **Geometry.** A near-square H₄ rectangle whose frontier gap is a few kT. Task A1 picks the aspect from a short scan
  so that:
  - |d(TS)/dx| ≥ 2 mHa/Bohr;
  - the plain |FD(δ) − FD(2δ)| ≤ 1e-3 Ha/Bohr.
  - [kT-5] At kT = 5 mHa the admissible window of c = dΔ/dx widens to about [4.4e-3, 0.13] Ha/Bohr (the TS condition
    does not depend on kT; F‴ ≤ 0.048·c³/(kT)² is 25× smaller at a given c). The near-square aspect moves accordingly.

  The two pull in opposite directions: d(TS)/dx ≤ 0.45·c while F‴ ≤ 0.048·c³/(kT)² (the earlier 0.2 was the canonical
  two-state form, 4× conservative; the pre-registered window [5e-3, 1e-1] is unchanged), with c = dΔ/dx. At kT = 1e-3 any c
  between 5e-3 and 5e-2 Ha/Bohr met both; at 5 mHa the scan and its fallback use c ∈ [5e-3, 1e-1] Ha/Bohr (preflight
  F15; plan Tasks 3 and 6), inside the window above with margin below 0.13. A1 also checks that the whole 4δ span of the stencil lies in that regime.
- **Method.** Central differences at δ/2 = 0.005, δ = 0.01 and 2δ = 0.02 Bohr along one atom's in-plane coordinate,
  combined by Richardson as in `oh_hf_force_matches_fd`. The reported value is R(δ/2, δ).
- **Certifying the Richardson value** (re-review R-SHOULD-B).
  - The two Richardson values R(δ/2, δ) and R(δ, 2δ) must agree within 2.5e-5 Ha/Bohr. Their difference, −F⁽⁵⁾δ⁴/32, is
    3.75× FD(δ)'s own δ⁴ term (F⁽⁵⁾δ⁴/120) and 15× the remainder of R(δ/2, δ) (F⁽⁵⁾δ⁴/480; Task 6 wording fix), so this
    bounds that remainder (to ≤ 1.7e-6) independently of HF.
  - A ratio window alone does not do it: at r = 5 the Richardson residual is still a third of the plain error.
- **The δ-scaling check** asserts (FD(2δ) − HF)/(FD(δ) − HF) ∈ [3, 5], i.e. that the plain FD error is the δ² term, so
  that δ resolves the TS swing. It is asserted only when |FD(δ) − HF| ≥ 1e-5 Ha/Bohr; below that the ratio is noise.
  - Near a pinned pair, F contains −2kT ln(2 cosh(Δ/4kT)) per channel (FD occupations); its third derivative is
    sech²u·tanh u·c³/(8kT²) for two channels (u = Δ/4kT), at most 0.048·c³/(kT)². (Historical: the earlier example,
    a plain FD(0.01) off by ~2e-3 on truncation alone, was computed with the 4× canonical form at kT = 1e-3; with
    0.048 it is ~5e-4, still 5× the bound.)
  - **[A1, review fix round 1] Pre-registered for the A1 candidate** (H₄ 1.8 × 1.95, measured F‴ ≈ 0.58 Ha/Bohr³ on the
    GPU): the plain truncation e = F‴δ²/6 ≈ 1e-5 Ha/Bohr, so the δ-scaling check sits at its own 1e-5 gate. Task 6
    records e (= FD_F(δ) − R_F(δ/2, δ)) and the offset b = HF − R_F(δ/2, δ) separately. With them the ratio is
    (4e − b)/(e − b), which stays in [3, 5] exactly when **b/e ∈ [−½, ¼]**: −e/2 ≤ b ≤ e/4 for e > b, e/4 ≤ b ≤ −e/2 for
    e < b (Task 6 wording fix: the literal interval [−e/2, e/4] is empty when e < 0, and on the CPU F‴ > 0 makes e < 0).
    **If the ratio check is asserted and b/e lies outside [−½, ¼], stop and report:** that is a spec decision
    for the user, never a re-pin of the window or the gate.
  - **[Task 6] Solver tolerances near a pinned pair.** The FD occupation response ∂f/∂ε = f(1 − f)/kT (≈ 34/Ha here)
    amplifies inner-solver error into a Δρ floor: at the CPU's fixed Poisson tolerance 1e-10 the V3 centre wandered at
    Δρ 2e-9 … 2e-7 for 200 steps whatever eig_tol, and with eig_tol 1e-9 a warm re-solve sat at 1e-8 … 2e-7 (the pair's
    rotation θ ≈ r/Δ). Ruling (option A, factor amended in Task 6 fix round 1): `qm`'s SCF asks its Poisson solves for
    min(1e-10, 1e-3·tol) (bit-identical for tol ≥ 1e-7; 1e-12 at tol 1e-9; 1e-2·tol left the floor at tol, with warm
    solves at 184 and 92 of 200 steps), and V3 sets eig_tol 1e-10 in its own settings. Loosening V3's tol was rejected.
- **Must show both, on the Richardson value:**
  - |HF_x + dF/dx|_Rich ≤ 1e-4 Ha/Bohr (the "M7 forces" bound, which is defined on Richardson);
  - |HF_x + dE/dx|_Rich ≥ 5e-4 Ha/Bohr, so the test tells F's derivative from E's.

**V5 new tests** [MUST-6]. The draft's H₄-square NVE run is withdrawn, for four reasons:
- its ≈ 0.27 Ha release toward 2 H₂ gives a Verlet swing of ≥ 1 mHa, the size of the signal;
- the fragments reach the faces of the 12 Bohr box;
- its "per atom" scaling of the drift bound was underived;
- "E + K moves" is the identity E + K = (F + K) + TS.

The two tests that replace it:
1. **Work-integral oracle** (CPU f64, the V3 geometry). Move one atom along a prescribed 0.2 Bohr path crossing the TS
   swing, in 40 steps, and integrate W = −∫F_HF·dx by Simpson.
   - Assert |W + ΔF| ≤ (Simpson error, estimated as |S₄₀ − S₂₀|/15) + 3(L·σ_F + √2·σ_E) (re-review R-SHOULD-C).
     - L = 0.2 Bohr is the path length.
     - σ_F and σ_E are the CPU f64 path scatters at tol 1e-9, measured in the test.
     - The noise of a Simpson W over the path is at most Σ|w_i|·σ_F = L·σ_F, and ΔF carries √2·σ_E.
     - (The earlier "40× the per-point force scatter" added Ha/Bohr to Ha.)
   - Assert |W + ΔE| = |ΔTS| ≥ 5× that bound. It is easily met: ΔTS ≈ 0.4 mHa over 0.2 Bohr at
     |d(TS)/dx| ≥ 2 mHa/Bohr.
2. **NVE** on a case whose released energy keeps h²/4·E_mode ≤ 1e-4 Ha, chosen by Task A1. For example, the bending
   vibration of a linear ²Π radical, started from a bend that splits the π pair by about 10 kT (re-review R-SHOULD-C); 50 mHa at kT = 5 mHa, so a larger bend than at 1e-3, which A1 checks against h²/4·E_mode [kT-5].
   - **[A1, 2026-11-14] The case is CO₂⁺** (charge +1, auto doublet, `Accepted`: pseudised C and O), 71³ × 0.20.
     N₃ was measured and failed: its ↓ π split peaks at 39 mHa (7.9 kT) near 60°, then the state changes (A1 note
     §2.1). CO₂⁺ is linear ²Π_g (R_e = 2.225 Bohr, TS = 2 kT ln 2 within 1e-9), with k_β = 0.107 Ha/rad² and
     ω_b = 2.335e-3 a.u.
   - Start: C on a grid node, β = 70° (bond angle 110°) at R = 2.226, at rest. ↓ split 47.3 mHa (9.5 kT), ↑ gap
     95 mHa, TS 0.514 mHa, so the swing is 6.4 mHa.
   - dt = 0.25 fs: h_b = 0.024 with E_mode = 98.75 mHa, giving h²/4·E_mode = 1.4e-5 Ha. The stretch holds 5.8 mHa,
     giving 5.6e-6. Face density ≤ 1.1e-6.
   - Steps: ≥ 2 bend periods (≈ 600 steps; 300 steps = 0.075 ps cannot resolve drift at DRIFT_BOUND 2e-4 Ha/ps
     against the Verlet swing). Task 11 times the per-step cost and measures the path scatter σ_path for this
     system under the 1e-5 stop before fixing the count and the fluctuation bound 3 × (h²/4·E_mode + 2σ_path); if
     600 steps exceed one 15-min GPU batch, split the run into checkpointed batches.
   - At linearity the π pair holds one hole, so TS = 2 kT ln 2 = 6.93 mHa at 5 mHa (1.39 at 1e-3). Bending splits the
     pair (∝ θ²), so the bend sweeps TS between 6.93 mHa and ≈ 0.
   - Only the initial bend energy is released, so nothing leaves the box.
   - **Withdrawn:** the earlier examples. A near-square H₄ at larger a still falls to 2 H₂, releasing ≈ 0.24 Ha
     (h²/4·E ≈ 2.6e-3 Ha). A stretched H₂ needs R ≳ 6–8 Bohr to bring its gap to a few kT, which does not fit the box
     under the face-density rule.
   - The box must keep the face density below the box rule throughout.
   - The drift and fluctuation bounds are re-derived for that trajectory by the suite's own rule (the
     `nve_energy_drift_is_bounded` method), not scaled per atom.
   - Assert F + K within those bounds, and max TS − min TS ≥ 1 mHa along the run, so the trajectory really crosses the
     TS swing.
3. **[Task 11 fix round 1, 2026-11-17] Both pre-registered bounds failed and were re-derived** from error sources
   documented before Task 11 (the V5 bound review and its revision; `references.md` "V5 fix round 1"):
   - work integral: + L·B_box, the discrete force's box term (`qm::forces` (iii); it falls 26× from a 12 to an 18 Bohr
     box), B_box = V3's 1e-4 Ha/Bohr because the CO₂ bend measured that term at ≈ 1.6e-5 Ha/Bohr, at (iii)'s ceiling;
     σ_F, σ_E from the larger of the two ends;
   - NVE: 730 steps (2.5 bend periods, two batches); (A) F + K ≤ 3·(swing + 2σ_path) + B_W and (B) the shadow Hamiltonian
     H̃ ≤ 3·2σ_path + B_W + h_s⁴·E_mode, swing = h_b²/4·(E_mode − E_s) + h_s²/4·E_s from the run's stretch amplitude
     (the bend feeds the stiffer stretch; no resonance), B_W = 3 × the measured GPU work bias along the bend and the
     stretch, attributed to the same box term by the CPU f64 Q_β (14 → 18 Bohr: 12× smaller). The drift bound is kept.

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
    kT = 5 mHa; the SCF converges to a stationary point of F = E − TS (spec 2026-11-13)".
  - §11 roadmap item 2: Fermi smearing is done for finite boxes; periodic cells and PBE remain.
- **The orbital-viewer spec, item 4.** Note that fractional, spherical occupations arrived with this spec, so the
  shared-shell rows are now per-level f.
- **`crates/validate/references.md`**:
  - "Approximation gaps that are expected": F = E − TS at T_el = 1 579 K (kT = 5 mHa). TS = 0 on gapped systems. Closed-form TS for
    exactly degenerate shells. The restricted-vs-broken-symmetry H₄ gap (12–24 mHa pseudo, 12.4 mHa all-electron;
    section 4.7(e)). F, not E, is what MD conserves.
  - Also under "Approximation gaps":
    - S is the KS-occupation entropy with ground-state LSDA xc, not the physical electronic entropy [SHOULD-4].
    - Separated fragments share μ_σ (LSDA delocalisation, section 4.7(h)) [SHOULD-10].
    - Atomisation and binding energies shift by +TS_atom (section 4.7(i)) [NOTE-4].
    - The Z = 1 lone-rule composition is not DFT-equivalent (section 6.4) [NOTE-9].
  - Every moved reference, with its TS (V6).
  - Under a new section, "Fermi–Dirac migration":
    - the V0 study (every species, both surfaces, U↑↑/U_c/U_s, the GPU jitter floor, the lopsided-tetra seeding);
    - V2's pins, V3's Richardson and ratio numbers, V5's work-integral and NVE bounds with their derivation;
    - the `branch_jump` constants and their derivation;
    - the all-electron witness cost and the H⁻ witness-cap outcome (section 4.2);
    - the shipped models' re-make;
    - V11's dry run, including the false-S estimate.
- **`docs/dev-reference.md`**:
  - New section "Occupations (Fermi–Dirac)": the constants and where they live, the tag format, `legacy_tag_of`, and
    the bitwise re-pins.
  - New section "Collection migration": the predecessor, the pass, `migration.txt`, the classes, code 9 and its review,
    record v2, the sidecar bump, the downgrade line.
  - "Shared collections": record v2 and code 9 (beside purge's code 8).
  - "SCF stall stop": no release restart.
  - "Test suites": V2 and the new oracle, the work-integral, NVE, branch-jump and migration tests.
  - The `branch_jump` guard (section 6.3), in the occupations section.
- **`README.md`** (CLAUDE.md, the user's rule):
  - The readouts (F, T_el, E and −TS in the breakdown).
  - "SCF converges on loose hand-built clusters".
  - The all-electron performance numbers, if they moved.
  - A short "Upgrading: collections are re-solved once" paragraph with the time estimate, and the note that the old
    folder is kept.
  - The shipped models' status.
- **Code docs:**
  - `surrogate::collection` (records v2, the read-only open, the migration files in the layout table);
  - `surrogate::quarantine` (code 9, and an older build meeting it);
  - `providers::pseudo` (the tag);
  - `providers::ks` (no longer "exactly today's problem"; `branch_jump` replaces the occupation watch);
  - the `qm::forces` and `qm_gpu::forces` module docs [SHOULD-8]: the forces are −dF/dR of the Mermin functional,
    stationary in ψ and f, not −dE/dR. The "Janak … at fixed occupations" grounding and the "spec §5.7" citation are
    rewritten.
  - the lone-atom rule's doc;
  - `app::update::formats` (the table and the downgrade line).
- **DEVLOG.md** at the merge (memory: DEVLOG upkeep). Link the diagnosis note forward to this spec.

## 11. Implementation order

Model tags are per CLAUDE.md. All are dispatched on Opus, per the 2026-10-21 ruling. Fable-class tasks get a
`physics-reviewer` pass before they land.

0. **Prerequisite.** `claude/charming-keller-0bc26b` has merged to `main`. Rebase `scf-stall`. Re-read the touch points
   in section 8. The diagnostic files (`scf_stall_diag.rs`, the env knobs in `qm-gpu/src/scf.rs`) stay uncommitted
   until A2, where the harness is reduced into V2 and the env knobs are removed.
1. **A1 (Fable-class, measurement):** V0, over every accepted open-shell species, both M and both surfaces (MUST-2). It
   includes the U↑↑/U_c/U_s extraction, the GPU jitter floor, and the seeding of the lopsided tetrahedron with its
   α = 0.1 and α = 0.05 broken-symmetry densities (SHOULD-2). A1 also:
   - chooses V3's aspect and checks its 4δ span;
   - chooses V5's NVE case (h²/4·E_mode ≤ 1e-4 Ha, the face density within the box rule). It confirms that LSDA keeps
     N₃ linear with a ²Π_g ground state on the pseudo surface at h_req, or substitutes another linear ²Π radical
     (A1 chose CO₂⁺; N₃ failed the split);
   - checks V3's plain-FD condition (|FD(δ) − FD(2δ)| ≤ 1e-3 Ha/Bohr);
   - measures the all-electron witness cost.

   **Stop here if V0 fails.**
2. **A2 (Fable-class):** `qm` FD, energies and stats; delete DS; V1; the V3 oracle; the physics review.
3. **A3 (Opus):** `qm-gpu` on the shared host function; V4.
4. **A4 (Opus; `branch_jump` physics-reviewed):** providers: `problem`, the tag, `legacy_tag_of`,
   `is_all_electron_tag`, `DIMER_PROCEDURE`, the lone rule, `EvalStats` with `branch_jump` and its training refusal, the
   scan's F; V2, V6, V8. The plan also greps every consumer that rebuilds a total from `Energy` components instead of
   reading `total` (the app's breakdown, the sidecar, `sim::references`); the review did not trace these.
5. **A5 (Opus):** UI readouts, breakdown, orbital rows, Compare note; V10 (labels).
6. **A6 (Fable-class):** V5 dynamics, plus a physics review of the whole solver half.
7. **B1 (Opus):** `surrogate::collection`: the read-only open, records v2, code 9, `migration.txt`; the sidecar bump;
   `formats`.
8. **B2 (Opus, physics-reviewed):** the providers' migration pass and the code-9 review step at section 8's touch
   points; V7.
9. **B3 (Opus):** app offer, progress and summary; the downgrade line; the Load-model refusal text; V10.
10. **B4 (Opus):** re-make the shipped models; V11 dry run on a copy; the docs (section 10); full serial gate run; then
    merge.

It ships as **one merge**. The solver change without the migration would strand the user's collection behind a new
key.

## 12. Risks

1. **Open-shell atoms** (section 4.7(d)). V0 failed at kT = 1e-3; the user ruled kT = 5 mHa with a 1e-5 scan stop
   [kT-5]. At 5 mHa every species reaches the symmetric state at every site with the same winner; what remains is
   convergence noise at the M = 1-type multiplicities (section 9, V0).
2. **All-electron per-solve cost goes up** (witness vectors). Measured and reported, not hidden. The default policy
   already pays it.
3. **The migration takes about 1–2 h of GPU time.** It is optional to start, pausable and resumable, and live use is
   not blocked.
4. **Many S-class samples in one region.** That region falls back to DFT until the review runs after the pass; the
   review then judges siblings together (section 7.6).
5. **The trust rework changes a touch point's meaning.** The plan re-reads them after the merge and amends this spec
   (section 8).
6. **Stationary-state switches along trajectories** (section 4.7(g)). Caught by the `branch_jump` guard on judged
   steps (warning plus training refusal). Sampling-pass solves are not judged by it and rely on leave-one-out.
7. **The `branch_jump` bound is too tight or too loose.** Too tight refuses good steps; too loose misses jumps. V8 pins
   both directions on analytic paths, its constants are derived and reviewed, and none may be loosened to pass.
8. **f32 eigenvalue jitter at degenerate shells becomes density noise on the GPU** (1/(4kT) = 50 Ha⁻¹ at 5 mHa).
   Measured in V0: the floor's median reaches 1e-6 to 2.8e-5 for the M = 1-type states (two open channels or two
   half-filled pairs: O, Si, S at M = 1; the O₂ singlet and NH at auto), so the 1e-5 stop of section 5.1 is reached
   by a downward fluctuation there, not with the factor-3 margin [kT-5].
9. **Slow convergence on pinned states** (the tetra 28–32 iterations, all-electron H₂O + 2 H 79). V2's budgets cover
   them. The stall stop must not fire on any V2 case.
10. **A larger electronic temperature biases more of the surface** [kT-5]. TS > 1e-5 Ha wherever a frontier gap is
    below ≈ 0.09 Ha (section 4.3): stretched bonds, dissociation, near-degenerate clusters. Atomisation energies
    against open-shell references shrink by 9.55 mHa per open p channel (section 4.7(i)). More migrated samples fall
    into class T or S than at 1e-3. Stated in `references.md`, never hidden.
11. **The witness costs more at 5 mHa** [kT-5]. The 36 kT arm binds in every channel; cold solves of the 3p atoms
    took 5–80× the DS time, and S at M = 5 needed 15 of the 16 allowed extra vectors (section 4.2). Plan Task 3
    measures the cost on molecules; a box a few per cent larger than 14 Bohr can hit the cap for the diffuse
    multiplicities.

## 13. Rulings I made (not decided by the user)

1. **One rule everywhere.** FD also replaces aufbau on all-electron scenes (`ionic none`). It fixes the all-electron
   failures (D§6), and `DegenerateShells` is deleted outright. *Reversible:* keeping aufbau for `none` would keep that
   surface's tag and bits, and its stalls.
2. `Aufbau` stays in `qm` only as the kT → 0 test reference. `for_grid` defaults to FD, so the oracles test the app's
   path.
3. **The μ bisection** (amended for MUST-1, SHOULD-1 and NOTE-1).
   - Bracket [ε_min − 50 kT, ε_max + 50 kT], halving to f64 exhaustion, with no tolerance parameter.
   - Two tie-directed bisections; the plateau midpoint is reported.
   - A non-finite eigenvalue is `QmError::Numerical` before bisecting.
   - A 2 048-halving backstop.
   - μ is a physical diagnostic only where a partial level exists.
   - Shared host f64 code for both solvers.
4. **The witness condition** (amended for MUST-1) is ε_k ≥ ε_{N_σ} + 0.1 Ha **and** ε_k − μ ≥ 36 kT.
   - The first arm is exactly today's reach and the prototype's configuration.
   - `MAX_WITNESS_GROWTH` = 16 stays, and its new all-electron error path is pinned by V8 (H⁻) [NOTE-6].
5. **The entropy** is evaluated from x as ln(1 + e^{−|x|}) + |x|·min(f, 1−f), with Neumaier summation.
6. **`Energy.total` = F**, with `internal` (E) and `ts` added. The per-iteration energy shown is F_HF.
7. **Gillan's E₀ = ½(E + F) is not computed or shown.**
8. **kT is a `qm` constant, `ELECTRON_KT` = 5e-3 Ha** (the user's 2026-11-14 ruling; 1e-3 before) [kT-5]. It is not exposed in the app and there is **no dev flag**;
   harnesses set it through `ScfSettings`. The tag carries kT.
9. **"Auto" multiplicity keeps its meaning** (lowest M of the parity). The reference scan compares F.
10. **`PARTIAL_OCCUPATION_MIN` = 1e-6** defines a partial level. The lone-atom "ends at a gap" test becomes "no partial
    level" on both surfaces, and the `all_electron` argument goes.
11. **`occupation_changed` is replaced by `branch_jump`** (amended for MUST-3), a stricter guard.
    - It compares ΔF with the trapezoid work of the forces between consecutive evaluations of one provider (largest
      atom displacement ≤ 0.1 Bohr).
    - The bound is 3× the path-scatter noise at the evaluation's own tolerance, plus 3κ̂‖ΔR‖³. Here κ̂ = min(running
      median over the last 8 judged, unflagged, curvature-dominated steps; κ_max = 1 Ha/Bohr³), with κ_max used alone
      until the window fills (re-review). **Superseded by plan Task 8 fix round 1 (review C1): the bound uses κ_max
      always (section 6.3).**
    - A grid, box or placement change restarts the history.
    - On a failure: a warning, and no surrogate training on that step.
    - For N↑ = N↓ the change of ∫|ρ↑ − ρ↓| is reported with it.
    - The constants are physics-reviewed in the plan.
12. **The DIIS fix** is achieved by deleting the release path. The frozen-orbital reset stays (its order is correct),
    pinned by a test.
13. **Tags:** `…/fd5e-3/stop2x…/pin1e-5` and `none/fd5e-3/pin1e-5` [kT-5]. Comparisons go through `is_all_electron_tag`.
14. **The occupation rule and kT are added to `DIMER_PROCEDURE`.** `recipe_hash` moves for Z ≤ `dimer_z_max`, so H and
    He tables regenerate once. The records must be bitwise identical and the verdicts reproduced.
15. **`legacy_tag_of`**, pinned against recorded strings, finds the predecessor exactly. There is no fuzzy matching.
16. **Migration = a new folder on the new key, filled by re-solving.** The old folder is opened strictly read-only and
    never written. An older build still runs on it as it was.
17. **Classification by re-solving every sample** (no stored-data criterion exists), with three classes U / T / S.
    ε_E = 1e-5 Ha per sample, ε_F = 1.5e-4 Ha/Bohr per component.
18. **Class U carries the old record bit for bit.** Class T takes the new label and keeps its trust state, with the
    T test as its review. Class S takes the new label at code 9. Codes 7 and 2 carry over, and purge's 8 becomes 7; an old
    code 1 does not survive into S.
19. **Code 9 `Q_RESOLVED`** (8 is `purge-forgotten`'s). It is out of the fit and outside the held cap. It does not
    fence before review, and does fence (`fences()`, never the misfit set M) after a failed review. It is never
    converted to code 1.
20. **The review** happens in a refit after the pass completes: a candidate fit with all code-9 samples, and release
    where the leave-one-out ratio is ≤ 1. It repeats at later refits.
21. **Failed re-solves are not migrated** and stay in the old folder. They are listed in `migration.txt`.
22. **References are re-solved** (lowest F), never carried. **The audit history is not carried.** `bond_experience.txt`
    is copied.
23. **The pass is user-started** (an offer, with "Later"), runs in the existing pass machinery, re-solves cold for
    reproducibility, and resumes from a cursor in `migration.txt`.
24. **The migration refuses while an older build holds the old folder's lock.**
25. **Collection `RECORD_VERSION` 1 → 2** for everything the new build writes. **The sidecar version is bumped.**
    `history`, `pseudo_cache` and `checkpoint` formats are unchanged. Code 9 is never written to a sidecar. (Versions
    symbolic: the merged `main`'s + 1; section 7.7.)
26. **A downgrade-warning line** names the pre-migration folder the older version will use.
27. **The shipped `.model` files are re-made** by the same engine in a headless harness, or re-taught if one fails its
    trust rule. **In-app migration of user `.model` files is a non-goal**; they get a clear refusal reason.
28. **V0 is an entry gate with a pre-registered stop** (amended for MUST-2, SHOULD-6, SHOULD-9 and NOTE-3).
    - It covers every accepted open-shell species at every scanned M, on both grids and at four sites, plus
      all-electron O at M = 1 and 3.
    - The site-spread margin is derived: 6σ_path + 1e-6 Ha.
    - The scan's winning M must be the same at every site.
    - The M = 1 two-channel predictions, the GPU jitter floor and the convergence precondition of option (ii) are
      pre-registered.
    - The spec states up front that V0 may well fail at M = 1, where the spin-mode threshold is ≳ 5 mHa by the spec's
      own premises (re-review).
    - If any species or multiplicity fails, the user rules on one global kT: (i) raise kT (≈ 3 mHa for O at M = 3,
      ≈ 5 mHa for the M = 1 spin mode, more for F), (ii) keep 1e-3 and document the site dependence, or (iii)
      Marzari–Vanderbilt cold smearing (offered only). kT stays 1e-3 here. I do not pick.
29. **V2 budgets are about 1.5× the prototype's iteration counts**, and F and TS are pinned to D§6 within 1e-5 Ha.
    The controls are pinned at 3 × max(achieved, path scatter): H₂ 2e-6, H₂O 6e-6 [SHOULD-6]. Bitwise pins are
    re-pinned with a reason, never deleted.
30. **Readout wording** (amended for SHOULD-4):
    - the row reads "F (electronic free energy, T_el 1579 K)";
    - the tooltip adds that T_el is a fixed numerical temperature, independent of the thermostat, and that F excludes
      the nuclei's entropy;
    - E and −TS appear in the breakdown;
    - the solve log gets a −TS note when TS ≥ 1e-6 Ha;
    - the PySCF compare gets a note.
31. **It ships as one merge** (solver plus migration), and implementation starts after the trust rework merges.

**Added after the physics review of 682960e** (each is my call on how to apply the finding):

32. **V3** [MUST-4] uses Richardson on δ/2, δ and 2δ (0.005 / 0.01 / 0.02 Bohr) against the "M7 forces" Richardson
    bound, 1e-4 Ha/Bohr.
    - R(δ/2, δ) and R(δ, 2δ) must agree within 2.5e-5 Ha/Bohr, which certifies the δ⁴ remainder (re-review).
    - It asserts (FD(2δ) − HF)/(FD(δ) − HF) ∈ [3, 5] only when |FD(δ) − HF| ≥ 1e-5 Ha/Bohr.
    - A1 picks the aspect so that the plain |FD(δ) − FD(2δ)| ≤ 1e-3 Ha/Bohr.
    - Its discrimination threshold is |HF + dE/dx|_Rich ≥ 5e-4 Ha/Bohr.
33. **The shared-seed spin-restricted auto surface** [SHOULD-2] is written into section 4.7(e) as an alternative for
    the user, not adopted. Task A1 seeds the tetrahedron with its broken-symmetry densities, to test the 12–24 mHa
    "LSDA broken symmetry" claim under F.
34. **V4** [MUST-5]:
    - μ is compared within `EPS_BOUND_PSEUDO` only where a partial level exists;
    - per-level |Δf| ≤ `EPS_BOUND_PSEUDO`/(2kT) = 1e-3 at kT = 5 mHa (5e-3 at 1e-3) [kT-5]. This is 2·EPS·max f(1−f)/kT, which also covers a three-level
      shell at f = ⅓; EPS/(4kT) was exact only for pairs (re-review).
    - the sum of f per degenerate cluster within 1e-9, a cluster being the levels the CPU finds within 1e-6 Ha of each
      other;
    - TS against the energy bounds;
    - the same state class required on both solvers.
35. **V5** [MUST-6]: the H₄-square NVE run is withdrawn. In its place:
    - a CPU work-integral oracle: Simpson over 40 steps on the V3 geometry, with
      |W + ΔF| ≤ |S₄₀ − S₂₀|/15 + 3(L·σ_F + √2·σ_E) (L = 0.2 Bohr; the units are corrected in the re-review), and
      |W + ΔE| ≥ 5× that;
    - an NVE run on a low-release case chosen by A1, the Renner–Teller bend of linear ²Π_g CO₂⁺ (A1: N₃ failed the 10 kT split). The bounds are re-derived by the suite's own rule
      (never scaled per atom), and TS must swing ≥ 1 mHa along it. The near-square H₄ and stretched-H₂ examples are
      withdrawn.
36. **Wording and documentation fixes:**
    - "minimises" becomes "makes stationary on the branch reached" [SHOULD-3];
    - "a real T_el effect" becomes "the Mermin–Kohn–Sham functional with ground-state xc; S is the KS-occupation
      entropy" [SHOULD-4];
    - the gapped f is "1.0 below μ, ≤ e^{−36} above, never 0.0" [SHOULD-5];
    - the force-module docs say −dF/dR, to today's floors [SHOULD-8];
    - new sections 4.7(h) (shared μ_σ across fragments, SHOULD-10) and 4.7(i) (atomisation energies shift by +TS_atom,
      NOTE-4);
    - NOTE-2, 5, 7, 8 and 9 are applied as wording.
37. **ε_E stays 1e-5 Ha** [SHOULD-7], with its derivation corrected: 5× the pseudo H₂O path scatter of 1.87e-6.
    V11 reports the false-S estimate. I kept the value rather than raising it, because a false S costs only one review,
    while a larger ε_E would carry real changes of up to that size as "unchanged".
38. **Every SHOULD is applied; none is declined.** All NOTEs are applied, NOTE-3 as option (iii) for the user only.

**Added after the physics re-review of 3be0b63:**

39. **The branch-jump bound** (R-MUST-A):
    - **κ̂** = min(κ_med, κ_max). κ_med skips noise-dominated and flagged steps. κ_max = 1 Ha/Bohr³ is global and is
      confirmed by the plan's physics review against measured E‴. (Superseded by plan Task 8 fix round 1: κ_max
      always, section 6.3.)
    - **Noise tier:** σ_E and σ_F are those of the evaluation's own tolerance; the app-tolerance tier is measured in A4.
    - **History restart:** on any grid, box or placement change.
    - **Coverage:** the guard is stated not to judge sparse surrogate audits (R-NOTE-b).
    - **V8 gains:** the relax-then-jump case and the alternating-jump case.
40. **The §5 `Aufbau` equivalence** is stated in Ha (|ΔE| ≤ 1e-8, CPU f64, tol 1e-9) (SHOULD-6 remainder).
41. **The M = 1 prediction is stated as expected, not as an "if".** On the spec's premises kT*_s ≳ 5 mHa, so V0 may well
    fail at M = 1 with kT = 1e-3, and option (i) would be set by it. This is stated up front in V0, and the
    pre-registered stop is unchanged in form: the user chooses (i) raise kT, (ii) keep 1e-3 and document the site
    dependence, or (iii) cold smearing. kT is not changed by me.

**Added for the user's ruling of 2026-11-14** [kT-5] (kT = 5 mHa and a 1e-5 scan stop; rulings 8, 13, 28 and 41
above are superseded where they name 1e-3):

42. **The 1e-5 stop applies to every GPU solve with a partial level** (section 5.1), not only to the reference scan:
    `max(tol_eff, GPU_PARTIAL_TOL = 1e-5)` on a step whose occupations have a level with min(f, 1 − f) ≥ 1e-6;
    gapped GPU solves and the CPU keep their tolerance; `effective_gpu_tol` is unchanged and applied first; the tag
    gains `/pin1e-5`; `ScfStats.tol_used` and `EvalStats::scf_tol_used` report it, so `branch_jump` judges such a
    solve on the 1e-5 tier. Reason: the f32 floor belongs to the partly occupied level (the O₂ singlet and NH at auto
    have it too), and V0 measured the stop's cost at ≤ 1.7 µHa in F and ≤ 8.8e-7 Ha/Bohr in the forces of the
    partial-level solves (pseudo; ≤ 3.2e-6 including gapped ones).
43. **The witness stays at 36 kT with `MAX_WITNESS_GROWTH` = 16.** It fits every V0 case at 5 mHa (at most 15 of 16,
    S at M = 5). The margin and the cost are stated in section 4.2 with three changes the user may choose (a larger
    cap, multi-vector growth, a 25 kT arm derived from the stop); none is adopted here.
44. **V0's "OH (bent, linear)"** is run as OH along a lattice axis and OH along a low-symmetry direction, since a
    diatomic has no bend. If a bent radical was meant, it is not yet measured (the V0 note says so).
45. **V2's pins are re-measured at 5 mHa** before they are written (plan Task 3, step 0); a V2 case that does not
    converge at 5 mHa is a stop.
46. **The FD ≡ Aufbau equivalence test runs at an explicit kT = 1e-3** (section 5), because at 5 mHa a 0.15–0.25 Ha
    gap leaves f differences of e^{−Δ/2kT} that the 1e-8 bound would not cover; the bound is not raised, and the
    production kT's TS on the same H₂O is reported beside it.
