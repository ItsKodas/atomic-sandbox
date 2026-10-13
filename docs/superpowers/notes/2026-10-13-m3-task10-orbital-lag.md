# M3 Task 10 note: orbital lag at the interactive tolerance (M4 item)

**What:** the GPU SCF calls LOBPCG with tol = max(eig_tol, 3 × f32 residual floor) ≈ 5.4e-6
at h = 0.25, and keeps a warm orbital set as it is when its residual is within 1.5× of that
(`qm_gpu::scf`, `EIG_GUESS_FACTOR`). A residual of 8e-6 admits an orbital error of
~8e-6/gap ≈ 2e-5 per orbital, i.e. ~5e-5 in ∫|δρ| for H₂ — above the viewer's SCF tol 1e-5.
So in a warm solve the orbitals freeze (0-iteration eigensolves) while the mixing moves the
potential; once the potential's drift pushes the residual back over the band they are re-solved
to tol, ρ_out jumps by ~8e-5, and the DIIS history (built on the frozen map) is stale.

**Measured (Task 10, app `moving_a_nucleus_warm_starts`, H₂ 49³, nucleus moved 0.02 Bohr):**
15 mixing steps warm vs 9 cold, 210 vs 304 ms, energies equal to 1e-6; the readback count
(the deterministic cost) is 553 warm vs 593 cold (15 vs 9 steps; the app test's tol 1e-5 —
at the validate suite's tol 1e-6 the same comparison is 340 vs 609). Variants measured and rejected (Task 10
report §6): a 2× band (fixes this case, breaks the CPU-parity step count at tol 1e-7),
allowing the band only once Δρ ≤ 10× tol (re-solving while the potential moves injects ~1e-7
of Δρ per update: slower everywhere), forcing ≥ 1 step per call (a Δρ floor at 1e-7).

**Why it is not a correctness issue:** the energy is second order in the orbital error, the
forces first order but inside their validate bounds (achieved 3e-6; bounds 5e-6 / 6e-5), and every late step is one cheap
frozen eigensolve (one readback).

**Remedies to evaluate in M4 (when dynamics makes warm starts the common case):**
1. A per-call eigen tol tied to the current Δρ (e.g. tol_eig = min(3× floor, c·Δρ·gap)) so the
   orbitals' admitted density error never exceeds the SCF's.
2. A cheaper floor: error-compensated (double-f32) Ritz lincombs would lower the residual floor
   below 1.8e-6 and let the SCF ask for a tighter eigen tol at the same cost.
3. Resetting the DIIS history when a frozen set is re-solved (cheap, but it only shortens the
   crawl after a jump; it does not remove the jump).
