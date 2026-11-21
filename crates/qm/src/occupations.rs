//! Fermi–Dirac occupations at a fixed electronic temperature, per spin channel (host f64; the one rule both
//! solvers call: the CPU loop in `scf` and `qm-gpu`). Spec `docs/superpowers/specs/2026-11-13-fermi-smearing-design.md`
//! §4.1–4.3, §5, §5.1. Atomic units: ε, μ, kT in Hartree; k_B = 1, so S is dimensionless and T_el = kT.
//!
//! # The rule (one channel σ with N_σ = n > 0 electrons and k ≥ n + 1 solved levels ε_i)
//! ```text
//! f_i = 1 / (1 + e^{x_i}),   x_i = (ε_i − μ) / kT,   Σ_i f_i = n
//! ```
//! f is the occupation of spin orbital i, 0 ≤ f ≤ 1; it decreases with x, so Σf(μ) increases with μ
//! (dΣf/dμ = Σ f(1 − f)/kT > 0). It is evaluated overflow-free: f = e^{−x}/(1 + e^{−x}) for x ≥ 0 and
//! 1/(1 + e^{x}) for x < 0, so neither branch forms an e^{|x|} > 1; x = ±∞ gives exactly 0 or 1, and nothing is
//! clipped (a level above μ keeps f ≈ e^{−x} down to the subnormals).
//!
//! # μ by bisection to the plateau midpoint [MUST-1]
//! - **Bracket** [ε_min − 50 kT, ε_max + 50 kT] (`MU_BRACKET_KT`). At the lower end every x ≥ 50, f ≤ e^{−50},
//!   so Σf ≤ k·e^{−50} < n; at the upper end every f ≥ 1 − e^{−50} (= 1.0 in f64), so Σf = k > n. Both ends are
//!   evaluated once in f64 before bisecting; they fail only if |ε|/kT ≳ 1e14 absorbs the 50 kT margin.
//! - **Two bisections**, each halving [lo, hi] at mid = ½lo + ½hi (cannot overflow) and stopping when mid equals
//!   an endpoint in f64, i.e. when lo and hi are adjacent floats:
//!   - (≥) mid → hi when Σf(mid) − n ≥ 0, else → lo. Invariant Σf(lo) < n ≤ Σf(hi), so hi converges to a crossing
//!     μ₋ (the least f64 μ with Σf ≥ n, up to rounding: the f64 Σf need not be monotone in its last ulp);
//!   - (≤) mid → lo when Σf(mid) − n ≤ 0, else → hi. Invariant Σf(lo) ≤ n < Σf(hi), so lo converges to a crossing
//!     μ₊ (the greatest f64 μ with Σf ≤ n, up to rounding).
//!   - μ = ½μ₋ + ½μ₊. In a gapped channel Σf == n *exactly* in f64 over a plateau of μ (1 − f_HOMO below ½ ulp(1),
//!     f_LUMO below ½ ulp(n)), from about ε_HOMO + 37 kT to ε_LUMO − 36 kT once the gap exceeds ≈ 74 kT, and a
//!     single tie rule would put μ on one edge. μ₋ and μ₊ are the two edges, so μ is the plateau midpoint (near the
//!     FD root at mid-gap); every f is the same to ≤ e^{−36} anywhere on the plateau. Without a plateau μ₋ and μ₊
//!     coincide or are adjacent floats around the root, μ is one of them, and
//!     |Σf − n| ≤ (dΣf/dμ)·ulp(μ) ≤ k·ulp(μ)/(4kT).
//! - **Termination** [NOTE-1]. A bisection exhausts its bracket after ≈ 60 halvings at |μ| ~ 1; ≲ 120 near μ = 0 at
//!   kT = 5 mHa (measured 113–116: within |x| < 1.1e-16 e^x rounds to 1, so Σf is flat over ≈ ±1.1e-16·kT around a
//!   root at 0 and the bisection stops there); ≈ 1 100 only for a subnormal kT, whose flat zone is itself below the
//!   subnormal spacing, so the bisection runs through the subnormals. `MAX_MU_HALVINGS` = 2 048 is a backstop that never binds, so termination never rests on the stop rule
//!   alone. `halvings` reports the larger of the two counts.
//! - **Inputs refused before bisecting** (`QmError::Numerical`) [SHOULD-1]: k < n + 1 (then Σf ≤ k = n at every μ,
//!   no bracket), a non-finite eigenvalue (a NaN never equals an endpoint, so the stop would never fire), a kT
//!   that is not finite and > 0, and a bracket that fails its one check (an end not finite, or not
//!   Σf(lo) < n < Σf(hi)). n = 0 is an unsolved channel: every f = 0, μ = None, S = 0, no level needed.
//! - No tolerance parameter; the result is a deterministic function of the f64 inputs (the same bits on every run
//!   and in both solvers).
//!
//! # Entropy (spec §4.3)
//! S = Σ_i s(f_i), s(f) = −[f ln f + (1 − f) ln(1 − f)] ≥ 0, evaluated from x, not from f:
//! ```text
//! s(x) = ln(1 + e^{−|x|}) + |x| / (1 + e^{|x|})        (= ln_1p(e^{−|x|}) + |x|·min(f, 1 − f))
//! ```
//! Derivation: with a = |x|, f_< = 1/(1 + e^{a}) = min(f, 1 − f) and f_> = 1 − f_< = e^{a}/(1 + e^{a}),
//! −f_< ln f_< − f_> ln f_> = f_< ln(1 + e^{a}) + f_>[ln(1 + e^{a}) − a] = ln(1 + e^{a}) − a f_>
//! = ln(1 + e^{−a}) + a(1 − f_>) = ln(1 + e^{−a}) + a f_<. Both terms are ≥ 0 and → 0 as a → ∞, so the
//! 0·ln 0 = 0 limit is built in and nothing cancels at f → 0 or 1; s(±∞) is defined as 0 (∞/∞ otherwise). A
//! channel's S is a Neumaier sum (rule 4); the solver's free energy is F = E − kT·Σ_σ S_σ.
//!
//! # Witness and partial levels (spec §4.2, §5)
//! - The highest solved level ε_max must satisfy ε_max ≥ ε_(n) + `WITNESS_GAP` (ε_(n) the n-th lowest) and
//!   ε_max − μ ≥ `WITNESS_TAIL_KT`·kT; at 36 kT the first unsolved level would hold f < 2.4e-16.
//! - A level is partial when min(f, 1 − f) ≥ `PARTIAL_OCCUPATION_MIN` = 1e-6, i.e. |ε − μ| ≲ 13.8 kT; the GPU
//!   stop is max(tol_eff, `GPU_PARTIAL_TOL`) on a step with one (spec §5.1).
//!
//! # Witness columns at a guard tolerance (plan 2026-11-13 Task 9b; both solvers)
//! One channel σ, a block of k columns: Ritz pairs (θ_j, x_j), ‖x_j‖ = 1 in the grid's L² product, residual
//! r_j = ‖H x_j − θ_j x_j‖, occupations f_j = f(θ_j; μ) of this module's rule, ρ_σ = Σ_j f_j x_j². The witness columns
//! carry f ≤ e^{−36}: only their eigenvalue bound is used, so holding them to the SCF's eigen tolerance (Task 7: LOBPCG
//! at its 200-iteration cap among box states) is pure cost. The rule splits the block **per column by f**, never by
//! position (a column between the occupied levels and the witness can be partly occupied: it stays tight).
//! - **What a loose column can do to Δρ.** Replacing x_j by any unit vector changes ρ_σ by f_j(x_j'² − x_j²), of L1
//!   norm ≤ 2f_j. Its level shift δθ_j moves μ (Σ_i f_i = N_σ fixed; ∂f_i/∂θ_i = −f_i(1 − f_i)/kT = −∂f_i/∂μ) by
//!   δμ = Σ_{j∈L} f_j(1 − f_j)δθ_j / Σ_i f_i(1 − f_i), so the tight columns' weights move by
//!   Σ_{i∈T} |δf_i| ≤ |δμ|·Σ_{i∈T} f_i(1 − f_i)/kT ≤ Σ_{j∈L} f_j |δθ_j| / kT, each |δf_i| moving ∫|δρ| by ≤ |δf_i|;
//!   and the loose columns' own weights move too, Σ_{j∈L} |δf_j| ≤ Σ_{j∈L} f_j |δθ_j − δμ| / kT, by as much again
//!   (Σ_T δf = −Σ_L δf; plan Task 9b review M1, which corrected a first version that omitted this term). A loose
//!   column held to r_j ≤ g has |δθ_j| ≤ r_j ≤ g (below), so over the loose set L
//!   ```text
//!   ∫|δρ| ≤ (2 + 2g/kT) · Σ_{j∈L} f_j            (= 2.4 Σ_L f_j at g = 1e-3 Ha, kT = 5 mHa)
//!   ```
//!   (2.4 is the first-order constant; the reviewer's random check of 20 000 FD channels gave a worst 2.44, the
//!   e^{g/kT} nonlinearity of f, inside the ≲ 2.9 below — not a violated bound). It assumes r_j ≤ g: a loose column
//!   that leaves LOBPCG on a floor exit above g (`qm_gpu::eigen`, the f32-floor diagnosis reads the scaled split
//!   residual) is bounded by its own per-column term (2 + 2r_j/kT)·f̂_j instead, and the 2.4e-2·tol figure does not
//!   apply to it.
//! - **The rule** (`tight_columns`). L is the set of the highest-ranked columns whose tail sum of the **upper-bound**
//!   occupations f̂_j = f(θ_j − r_j; μ) ≥ f_j (the level lies above θ_j − r_j) stays below `LOOSE_F_FACTOR`·tol,
//!   tol the SCF's Δρ stop: ∫|δρ| ≤ 2.4e-2·tol, ≲ 2.9e-2·tol with the loose levels' own shift of μ (a factor
//!   e^{g/kT} ≤ e^{0.2} on f), so no loose column can move Δρ across the stop. The threshold comes from the
//!   stop, not from a fixed f: H₃⁺ at 1.6 r₀ has its ↑ level 2 at f ≈ 5e-11, and at tol = 1e-6 (threshold 1e-8)
//!   that level is loose, while a level at 2e-10 would be too (task 9 review C1's caution). Every other column is
//!   **tight**: held to the SCF's eigen tolerance, as before. Because f falls with θ, L is a top segment of the ranks.
//! - **The energy and forces.** F is stationary in ψ at fixed f (spec §4.4), so a loose column's error enters F at
//!   second order, f_j r_j²/δ_j (δ_j its distance to the rest of the spectrum), and through ρ: ≤ (2.9e-2·tol)²-sized.
//!   The Hellmann–Feynman force moves by ≤ ∫|δρ|·max|∇V_I| ≲ 2.9e-2·tol·max|∇V_I|. Tested on the CPU f64 reference
//!   (`scf` tests, loose against all-tight): |ΔF| ≤ 1e-10 Ha, |ΔF_I| ≤ 1e-9 Ha/Bohr.
//! - **The guard tolerance** g = `WITNESS_GUARD_TOL` = `WITNESS_GAP`/100 = 1e-3 Ha (the Harris display's `GUARD_TOL`),
//!   or the step's eigen tol when that is looser (the GPU's Δρ-tied early steps).
//! - **The witness test reads a lower bound** (`witness_bound_reached`): for a unit x there is an eigenvalue λ with
//!   |λ − θ| ≤ ‖Hx − θx‖ (the Krylov–Weinstein bound; Parlett, *The Symmetric Eigenvalue Problem*, Thm 4.5.1), so
//!   the highest column's level is ≥ θ_top − r_top, and both arms are tested on it:
//!   ```text
//!   θ_top − r_top ≥ ε_(N_σ) + WITNESS_GAP   and   θ_top − r_top − μ ≥ WITNESS_TAIL_KT · kT
//!   ```
//!   At r ≤ g it costs ≤ 1e-3 Ha of the 0.18 Ha window; a tight top (r ~ 1e-6) gives the Ritz test to 1e-6. Never the
//!   Ritz value alone.
//! - **The premise of both bounds** (review M2, restated by the task 9c review I1): Ritz values bound the levels from
//!   above (θ_j ≥ λ_j), but Krylov–Weinstein only puts *some* eigenvalue within r of θ — it does not bound a column's
//!   lowest spectral component: the weight a unit x with residual r carries below θ − D is ≤ r²/D² (Chebyshev), and up
//!   to half of it can lie just below θ − r. So θ_j − r_j is used as a lower bound on λ_j — for f̂ and for the witness —
//!   under the premise that the spectral weight below θ_j − r_j is negligible (≤ r_j²/D²), i.e. the block holds the
//!   lowest k levels and no level below the block's top was missed. That holds for columns converged to r ≤ g (hidden
//!   weight ≲ 3e-5 at g = 1e-3 against the window's D ≈ 0.18 Ha), which is why an untested column (the GPU's buffer)
//!   is a witness only at r ≤ g. Both solvers let the top tested level decide a (tight) step only at
//!   r ≤ 1.5 g and otherwise grow the block by one vector (`qm-gpu` `solve_fd`; the CPU's `scf::solve_fd` since plan
//!   2026-11-13 Task 11 fix round 1, solver-half review M5, `WITNESS_RESIDUAL_FACTOR`) — on the GPU after the lowest
//!   converged witness below (the next bullet) found none. The GPU solver checks the rest the way it can: a Rayleigh–Ritz step whose formed
//!   vectors do not reproduce their Ritz values (a lost column, `qm_gpu::eigen` "Ritz vectors that do not match their
//!   Ritz values") is undone; one it cannot undo is an error on the SCF path.
//! - **The lowest converged witness** (GPU, `witness_keep_within`; branch `eig-floor`, 2026-11-21). The bound above is per
//!   level: λ_t ≥ θ_t − r_t needs only column t at r_t ≲ g and the columns below it holding the lowest t − 1 levels
//!   (premise M2); every level above t is then ≥ λ_t, whatever its own column's residual. So when the top tested level
//!   passes both arms at r > 1.5 g, the GPU first looks for the lowest tested rank t ≥ n that clears both arms on
//!   θ_t − r_t **at r_t ≤ 1.5 g**; if there is one it decides the step and the block is trimmed above it exactly as
//!   `witness_keep` would (t + 1, or t + 2 with a passing spare) — the level the old rule's eventual trim kept anyway —
//!   and only if there is none does the block grow by one. Why the GPU needs it: the top column of a block converges
//!   slowest (its rate is set by the gap to the first level outside the block), and `qm_gpu::eigen`'s floor diagnosis
//!   reads the split residual scaled by tol/g, whose band, 10·floor·g/tol = 3.3 g at the floor-limited tight tol
//!   3·floor, lets a loose top leave on a `Floor` exit anywhere up to 3.3 g. Each growth then made a fresh vector the
//!   new top, in the box continuum, and the chain ran on: measured on a warm water step at 119³ × 0.15 (17.7 Bohr, the
//!   app's Estimated-mode box), a block of 7 grown one vector at a time to 20 per spin, 3281 LOBPCG iterations and
//!   68.6 s, while the first level that clears both arms (θ = 0.0389 Ha against a 0.009 Ha window) was at r = 4.3e-5
//!   from the second call on and ~1e-5 after; with the rule that level decides at the second call (1054 iterations,
//!   peak block 8; `qm_gpu::scf` module doc). The CPU keeps the top-level rule: its f64 eigensolver meets g except at
//!   its cap.
//! - **Growth** (`grown_extras`): when the witness fails the block grows by its current extra count (the extra block
//!   doubles: 1, 2, 4, 8, 16, 32), clamped to the cap, instead of one vector per re-solve; each grown vector keeps
//!   its index-seeded guess. Reaching e extras costs ⌈log₂ e⌉ re-solves instead of e − 1.
//! - **The step-accuracy tail** (plan 2026-11-13 Task 9c (b), `step_tail_kt`; the task 9b review's lever). On a
//!   *loosened* GPU step (its eigen tol above the tight one, i.e. far from convergence) the tail arm is
//!   X = ln(n_w/(c·Δρ_prev)) clamped to [6, 36] kT, c = `STEP_TAIL_C` = 1e-3, n_w the block's levels. The levels it
//!   leaves out are all those above μ + X·kT; with g the density of states there (levels per Ha), their charge is
//!   Σf ≲ ∫ g e^{−(ε−μ)/kT} dε = g·kT·e^{−X} (spec 2026-11-13 §4.2 (c); task 9c review M1), which is ≤ n_w·e^{−X} =
//!   c·Δρ_prev — the same error class as the Δρ-tied eigen tol, a thousandth of the step's own error — while
//!   n_w ≥ g·kT. That holds in the boxes measured: g·kT ≈ 0.3 at 14 Bohr, ≈ 1.6 at 24.5, ≈ 3 at 30 (Weyl), against
//!   n_w ≥ n_σ + 2. The gap arm and the lower bound stay. The tail cannot change a converged answer (review M2): a
//!   solve converges only on a tight step (step tol = eig tol), which runs the full 36 kT window, so a level that
//!   crosses into the window can never be missed at convergence; a descending level enters on a transient step once
//!   its f exceeds c·Δρ, at the cost of one growth. The [6, 36] clamp binds only for Δρ_prev ≳ n_w/(c·e⁶) ≈ 2.5·n_w (the floor) or
//!   tiny Δρ_prev (36 kT). A step whose window widened and grew the block resets the DIIS history (the map changed, like
//!   an unfreeze; by the bound above the change is ≤ c·Δρ — measured, removing the reset changed neither gate case).
//! - **No loose stop** (plan 2026-11-13 Task 9c (d), measured and not adopted). The bound above holds per column,
//!   ∫|δρ| ≤ Σ_L (2 + 2r_j/kT)·f̂_j at any r_j, so a GPU eigensolve could end once that sum is below 1e-2·tol and the
//!   witness passes, the loose columns wherever they are. Tried on every step, on loosened steps and the first one, and
//!   on the first step alone: it cut the cold all-electron H₂O 97³'s LOBPCG iterations by a quarter, but the loose
//!   columns it leaves far from converged (residuals up to 0.1 Ha) seed the next step, whose tight solve then
//!   floor-exits or freezes, and the stretched H₂ (`providers` `the_level_gated_stall_stop_on_the_measured_cases`, σ_g/σ_u
//!   at f 0.93/0.07) took 43, 102 and 41 steps (65³ R 6 / 65³ / 107³ R 5.75) against 11 and 25 without it (gate ≤ 30).
//!   The loose columns keep the guard tolerance.
//! - **Trim** (`witness_keep`): a doubled block overshoots (the 14 Bohr H⁻ needs 10–11 extras and doubled to 16, which
//!   made every LOBPCG iteration of the solve ~2× dearer: 185 s against 75 s). Once the witness passes, the columns
//!   above the lowest level t that passes it alone (with one spare above t when that spare passes too) are dropped:
//!   their f are below e^{−36} each, so the occupations of what is kept change by less than f64 resolves next to 1, and
//!   F, ρ and the forces by nothing the stop can see. The kept block still passes the witness by construction.

use crate::error::QmError;
use crate::grid::neumaier;
use crate::scf::ascending;

/// The electronic temperature kT (Ha): 5 mHa ≈ 1 579 K, the user's ruling of 2026-11-14 (spec §2 item 3). One
/// solver constant for every scene; not per element.
pub const ELECTRON_KT: f64 = 5e-3;
/// Witness arm 1 (Ha): the highest solved level lies at least this far above the n-th lowest (spec §4.2; the
/// value of the shared-shell rule's candidate window τ_c, deleted in plan 2026-11-13 Task 10).
pub const WITNESS_GAP: f64 = 0.1;
/// Witness arm 2: the highest solved level lies at least this many kT above μ (f < 2.4e-16 beyond it).
pub const WITNESS_TAIL_KT: f64 = 36.0;
/// A level with min(f, 1 − f) at or above this is partly occupied (|ε − μ| ≲ 13.8 kT).
pub const PARTIAL_OCCUPATION_MIN: f64 = 1e-6;
/// The step-accuracy tail (plan 2026-11-13 Task 9c (b)): the omitted charge's fraction of the previous step's Δρ.
pub const STEP_TAIL_C: f64 = 1e-3;
/// Its floor, in kT (f ≤ e^{−6} ≈ 2.5e-3 per omitted level at the widest Δρ).
pub const STEP_TAIL_MIN_KT: f64 = 6.0;
/// The GPU SCF's convergence stop on a step whose occupations have a partial level (spec §5.1; the user's
/// 2026-11-14 value). The one definition (preflight F11): `qm-gpu`, the tags and the dimer ledger read it.
pub const GPU_PARTIAL_TOL: f64 = 1e-5;
/// A column is loose when the tail sum of the upper-bound occupations over it and every column above it is below this
/// × the SCF's Δρ tolerance (module doc, "Witness columns at a guard tolerance"): ∫|δρ| ≤ 2.4e-2·tol (≲ 2.9e-2·tol).
pub const LOOSE_F_FACTOR: f64 = 1e-2;
/// The loose (guard) columns' residual tolerance (Ha): `WITNESS_GAP`/100 = 1e-3, so the witness's lower bound
/// θ − r loses ≤ 1e-3 Ha of its 0.18 Ha window (module doc). The Harris display's `GUARD_TOL` is the same value.
pub const WITNESS_GUARD_TOL: f64 = WITNESS_GAP / 100.0;
/// The μ bracket's margin beyond the extreme levels, in kT.
pub const MU_BRACKET_KT: f64 = 50.0;
/// Backstop on each μ bisection; never binds for finite inputs (≲ 120 halvings at kT = 5 mHa, ≈ 1 100 only for a
/// subnormal kT; module doc, "Termination").
pub const MAX_MU_HALVINGS: usize = 2048;

/// One channel's Fermi–Dirac occupations.
#[derive(Clone, Debug, PartialEq)]
pub struct FdOccupations {
    /// f_i in the order of the `evals` passed in.
    pub f: Vec<f64>,
    /// μ (Ha), the plateau midpoint; `None` for an empty channel. A physical diagnostic only where a level is
    /// partial (spec §4.1).
    pub mu: Option<f64>,
    /// S = Σ s(f_i), dimensionless (k_B = 1); the free-energy term is kT·S.
    pub entropy: f64,
    /// The larger halving count of the two bisections (< `MAX_MU_HALVINGS` for finite inputs).
    pub halvings: usize,
}

/// f at x = (ε − μ)/kT, overflow-free; never clipped.
fn occupation(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        e / (1.0 + e)
    } else {
        1.0 / (1.0 + x.exp())
    }
}

/// Σf at μ, compensated.
fn total(evals: &[f64], mu: f64, kt: f64) -> f64 { neumaier(evals.iter().map(|&e| occupation((e - mu) / kt))) }

/// One tie-directed bisection on [lo, hi]. `ge`: Σf − n ≥ 0 moves hi, and hi (μ₋) is returned; otherwise Σf − n ≤ 0
/// moves lo, and lo (μ₊) is returned. Also returns the halving count.
fn bisect(evals: &[f64], n: f64, kt: f64, mut lo: f64, mut hi: f64, ge: bool) -> (f64, usize) {
    let mut halvings = 0;
    while halvings < MAX_MU_HALVINGS {
        let mid = 0.5 * lo + 0.5 * hi;
        if mid <= lo || mid >= hi {
            break;
        }
        let d = total(evals, mid, kt) - n;
        let to_hi = if ge { d >= 0.0 } else { d > 0.0 };
        if to_hi {
            hi = mid
        } else {
            lo = mid
        }
        halvings += 1;
    }
    (if ge { hi } else { lo }, halvings)
}

/// Fermi–Dirac occupations of one channel with `n` electrons over `evals` (any order) at temperature `kt` (Ha):
/// f in `evals`' order, μ the plateau midpoint of the two tie-directed bisections, S with Neumaier compensation.
/// The module doc has the rule and the inputs it refuses.
pub fn fermi_dirac(evals: &[f64], n: usize, kt: f64) -> Result<FdOccupations, QmError> {
    if n == 0 {
        return Ok(FdOccupations { f: vec![0.0; evals.len()], mu: None, entropy: 0.0, halvings: 0 });
    }
    if evals.len() < n + 1 {
        return Err(QmError::Numerical(format!("Fermi–Dirac needs at least n + 1 = {} levels, got {}", n + 1, evals.len())));
    }
    if let Some(e) = evals.iter().find(|e| !e.is_finite()) {
        return Err(QmError::Numerical(format!("non-finite eigenvalue {e} in the Fermi–Dirac occupations")));
    }
    if !(kt.is_finite() && kt > 0.0) {
        return Err(QmError::Numerical(format!("Fermi–Dirac kT = {kt} is not finite and > 0")));
    }
    let nf = n as f64;
    let lo = evals.iter().copied().fold(f64::INFINITY, f64::min) - MU_BRACKET_KT * kt;
    let hi = evals.iter().copied().fold(f64::NEG_INFINITY, f64::max) + MU_BRACKET_KT * kt;
    if !(lo.is_finite() && hi.is_finite() && total(evals, lo, kt) < nf && total(evals, hi, kt) > nf) {
        return Err(QmError::Numerical(format!("the Fermi–Dirac μ bracket [{lo:e}, {hi:e}] does not bracket n = {n} at kT = {kt:e}")));
    }
    let (mu_minus, h_ge) = bisect(evals, nf, kt, lo, hi, true);
    let (mu_plus, h_le) = bisect(evals, nf, kt, lo, hi, false);
    let mu = 0.5 * mu_minus + 0.5 * mu_plus;
    let f = evals.iter().map(|&e| occupation((e - mu) / kt)).collect();
    let entropy = neumaier(evals.iter().map(|&e| fd_entropy((e - mu) / kt)));
    Ok(FdOccupations { f, mu: Some(mu), entropy, halvings: h_ge.max(h_le) })
}

/// The entropy s ≥ 0 of one level at x = (ε − μ)/kT: ln_1p(e^{−|x|}) + |x|/(1 + e^{|x|}) (module doc); s(±∞) = 0.
pub fn fd_entropy(x: f64) -> f64 {
    let a = x.abs();
    if a.is_infinite() {
        return 0.0;
    }
    (-a).exp().ln_1p() + a / (1.0 + a.exp())
}

/// Whether a channel's highest solved level is a witness for `n` electrons at μ = `mu` (spec §4.2): ε_max ≥ ε_(n) +
/// `WITNESS_GAP` (ε_(n) the n-th lowest) and ε_max − μ ≥ `WITNESS_TAIL_KT`·kT. An empty channel (n = 0) needs none;
/// a channel with no level above its n-th never has one.
pub fn witness_reached(evals: &[f64], n: usize, mu: f64, kt: f64) -> bool {
    if n == 0 {
        return true;
    }
    if evals.len() <= n {
        return false;
    }
    let order = ascending(evals);
    let (eps_n, eps_max) = (evals[order[n - 1]], evals[order[order.len() - 1]]);
    eps_max >= eps_n + WITNESS_GAP && eps_max - mu >= WITNESS_TAIL_KT * kt
}

/// `witness_reached` on the levels' lower bounds (module doc, "Witness columns at a guard tolerance"): the highest
/// level's θ − r (θ_top the largest Ritz value, r its residual) must lie ≥ ε_(n) + `WITNESS_GAP` and ≥ μ +
/// `WITNESS_TAIL_KT`·kT; ε_(n) is the n-th lowest Ritz value. An empty channel (n = 0) needs none; a channel with no
/// level above its n-th never has one. `residuals[i]` belongs to `evals[i]`; a non-finite residual never passes.
pub fn witness_bound_reached(evals: &[f64], residuals: &[f64], n: usize, mu: f64, kt: f64) -> bool {
    witness_bound_reached_tail(evals, residuals, n, mu, kt, WITNESS_TAIL_KT)
}

/// `witness_bound_reached` with the tail arm at `tail_kt`·kT (plan 2026-11-13 Task 9c (b): `step_tail_kt` on a loosened
/// GPU step; `WITNESS_TAIL_KT` everywhere else).
pub fn witness_bound_reached_tail(evals: &[f64], residuals: &[f64], n: usize, mu: f64, kt: f64, tail_kt: f64) -> bool {
    assert_eq!(evals.len(), residuals.len(), "witness_bound_reached: one residual per level");
    if n == 0 {
        return true;
    }
    if evals.len() <= n {
        return false;
    }
    let order = ascending(evals);
    let (eps_n, top) = (evals[order[n - 1]], order[order.len() - 1]);
    let lower = evals[top] - residuals[top];
    lower >= eps_n + WITNESS_GAP && lower - mu >= tail_kt * kt
}

/// The witness tail of a loosened SCF step (plan 2026-11-13 Task 9c (b); module doc, "The step-accuracy tail"):
/// X = ln(n_w / (`STEP_TAIL_C`·Δρ_prev)) clamped to [`STEP_TAIL_MIN_KT`, `WITNESS_TAIL_KT`], n_w the block's levels
/// (a bound on the omitted charge while n_w ≥ g·kT, g the density of states above μ + X·kT: module doc).
/// A non-finite or non-positive Δρ_prev gives the full `WITNESS_TAIL_KT`.
pub fn step_tail_kt(n_w: usize, delta_rho_prev: f64) -> f64 {
    if !(delta_rho_prev.is_finite() && delta_rho_prev > 0.0) { return WITNESS_TAIL_KT; }
    (n_w.max(1) as f64 / (STEP_TAIL_C * delta_rho_prev)).ln().clamp(STEP_TAIL_MIN_KT, WITNESS_TAIL_KT)
}

/// How many columns are **tight** (module doc, "Witness columns at a guard tolerance"), as a position count: the
/// columns of `evals` (positions 0..k, residuals `residuals`) whose upper-bound occupation f̂ = f(θ − r; μ) is needed
/// to keep the loose tail sum below `LOOSE_F_FACTOR`·`tol` are tight; the returned count covers every one of them,
/// i.e. it is 1 + the largest position of a tight column (0 if none). By rank the loose columns are the top segment
/// whose Σ f̂ < `LOOSE_F_FACTOR`·tol; positions are ranks once the block is ascending, and a column out of rank order
/// (a warm guess) only raises the count. An empty channel (`mu` None) has no tight column.
pub fn tight_columns(evals: &[f64], residuals: &[f64], mu: Option<f64>, kt: f64, tol: f64) -> usize {
    assert_eq!(evals.len(), residuals.len(), "tight_columns: one residual per level");
    let Some(mu) = mu else { return 0 };
    let order = ascending(evals);
    let limit = LOOSE_F_FACTOR * tol;
    // from the top rank down: the loose ranks are those whose running tail sum stays below the limit
    let mut tail = 0.0;
    let mut loose_from = order.len(); // ranks ≥ loose_from are loose
    for (rank, &i) in order.iter().enumerate().rev() {
        tail += occupation((evals[i] - residuals[i] - mu) / kt);
        if !(tail < limit) {
            break;
        }
        loose_from = rank;
    }
    order[..loose_from].iter().map(|&i| i + 1).max().unwrap_or(0)
}

/// The tight count a solve starts from (both solvers' `solve_fd`): the positions whose stored occupation `f` (a warm state's,
/// ascending) the tail rule of `tight_columns` needs at the SCF's `tol`, at least `n`; `n` on a cold start (no f yet).
/// Only a start: every solve verifies it on its own levels and re-solves when a column needs the eigen tol.
pub fn tight_start(f: Option<&[f64]>, n: usize, tol: f64) -> usize {
    let Some(f) = f else { return n };
    let limit = LOOSE_F_FACTOR * tol;
    let mut tail = 0.0;
    let mut loose_from = f.len();
    for (i, &fi) in f.iter().enumerate().rev() {
        tail += fi;
        if !(tail < limit) { break; }
        loose_from = i;
    }
    loose_from.max(n)
}

/// After `witness_bound_reached` has passed on `evals` (any order; `residuals` theirs), how many of the lowest-ranked
/// levels to keep (module doc, "Trim"): t + 1 for the lowest rank t ≥ n whose lower bound θ_t − r_t clears both witness
/// arms (with μ = `mu`, the full block's), or t + 2 when the level above t clears them too (a spare above the witness);
/// `evals.len()` when nothing passes (never after a passed test) or n = 0.
pub fn witness_keep(evals: &[f64], residuals: &[f64], n: usize, mu: f64, kt: f64) -> usize {
    witness_keep_tail(evals, residuals, n, mu, kt, WITNESS_TAIL_KT)
}

/// `witness_keep` with the tail arm at `tail_kt`·kT (`witness_bound_reached_tail`).
pub fn witness_keep_tail(evals: &[f64], residuals: &[f64], n: usize, mu: f64, kt: f64, tail_kt: f64) -> usize {
    assert_eq!(evals.len(), residuals.len(), "witness_keep: one residual per level");
    if n == 0 || evals.len() <= n {
        return evals.len();
    }
    let order = ascending(evals);
    let eps_n = evals[order[n - 1]];
    let passes = |rank: usize| {
        let i = order[rank];
        let lower = evals[i] - residuals[i];
        lower >= eps_n + WITNESS_GAP && lower - mu >= tail_kt * kt
    };
    match (n..order.len()).find(|&t| passes(t)) {
        Some(t) if t + 1 < order.len() && passes(t + 1) => t + 2,
        Some(t) => t + 1,
        None => evals.len(),
    }
}

/// The witness of a tight GPU step whose top level left its eigensolve above `r_max` (module doc, "The lowest converged
/// witness"): `witness_keep_tail`'s keep count, but read at the lowest rank t ≥ n whose lower bound θ_t − r_t clears both
/// arms **and** whose residual r_t ≤ `r_max` (premise M2: a lower bound only at r ≲ g) — t + 1, or t + 2 when the level
/// above t clears both arms on its lower bound too (the spare, as in `witness_keep`). `None` when no level qualifies (the
/// block must grow); `Some(evals.len())` for an empty channel (n = 0), which needs no witness.
pub fn witness_keep_within(evals: &[f64], residuals: &[f64], n: usize, mu: f64, kt: f64, tail_kt: f64, r_max: f64) -> Option<usize> {
    assert_eq!(evals.len(), residuals.len(), "witness_keep_within: one residual per level");
    if n == 0 {
        return Some(evals.len());
    }
    if evals.len() <= n {
        return None;
    }
    let order = ascending(evals);
    let eps_n = evals[order[n - 1]];
    let passes = |rank: usize| {
        let i = order[rank];
        let lower = evals[i] - residuals[i];
        lower >= eps_n + WITNESS_GAP && lower - mu >= tail_kt * kt
    };
    // (a non-finite residual fails `≤ r_max` and, through the lower bound, both arms)
    let t = (n..order.len()).find(|&t| passes(t) && residuals[order[t]] <= r_max)?;
    Some(if t + 1 < order.len() && passes(t + 1) { t + 2 } else { t + 1 })
}

/// The extra-vector count after one growth of a block holding `extras` vectors beyond its n (module doc, "Growth"):
/// the extra block doubles (at least one vector), clamped to `cap`.
pub fn grown_extras(extras: usize, cap: usize) -> usize { (2 * extras).max(extras + 1).min(cap) }

/// One partly occupied level: min(f, 1 − f) ≥ `PARTIAL_OCCUPATION_MIN`.
#[derive(Clone, Debug, PartialEq)]
pub struct PartialLevel {
    /// 0 = ↑, 1 = ↓.
    pub spin: u8,
    /// The level's rank in ascending order of ε within its channel (equal ε keep their input order).
    pub index: usize,
    /// ε (Ha).
    pub eps: f64,
    /// Its occupation f.
    pub f: f64,
}

/// The channel's partly occupied levels, in ascending order of ε.
pub fn partial_levels(evals: &[f64], f: &[f64], spin: u8) -> Vec<PartialLevel> {
    assert_eq!(evals.len(), f.len(), "partial_levels: one f per level");
    ascending(evals)
        .into_iter()
        .enumerate()
        .filter(|&(_, i)| f[i].min(1.0 - f[i]) >= PARTIAL_OCCUPATION_MIN)
        .map(|(rank, i)| PartialLevel { spin, index: rank, eps: evals[i], f: f[i] })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::QmError;
    use crate::grid::neumaier;
    use std::f64::consts::LN_2;

    /// Σf with the rule's own compensated sum.
    fn sum(f: &[f64]) -> f64 { neumaier(f.iter().copied()) }

    /// splitmix64: a deterministic stream for the randomised tests (no `rand` in `qm`).
    struct Mix(u64);
    impl Mix {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        /// Uniform in [0, 1).
        fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
    }

    #[test]
    fn fd_degenerate_pair_half_filled() {
        let o = fermi_dirac(&[-0.3, -0.3], 1, 1e-3).unwrap();
        assert_eq!(o.f, vec![0.5, 0.5]);
        let mu = o.mu.unwrap();
        assert!((mu + 0.3).abs() <= 1e-15, "μ = {mu:e}");
        assert!((o.entropy - 2.0 * LN_2).abs() <= 1e-14, "S = {}", o.entropy);
    }

    #[test]
    fn fd_gap_symmetric_mu() {
        let kt = ELECTRON_KT;
        let o = fermi_dirac(&[-0.3, -0.3 + 5.0 * kt], 1, kt).unwrap();
        let (f0, f1) = (1.0 / (1.0 + (-2.5f64).exp()), 1.0 / (1.0 + 2.5f64.exp()));
        assert!((o.f[0] - f0).abs() <= 1e-14, "f0 = {} vs {f0}", o.f[0]);
        assert!((o.f[1] - f1).abs() <= 1e-14, "f1 = {} vs {f1}", o.f[1]);
        let mu = o.mu.unwrap();
        assert!((mu - (-0.3 + 2.5 * kt)).abs() <= 1e-12, "μ = {mu}");
    }

    /// `f` follows `evals`' order, whatever that order is.
    #[test]
    fn fd_order_follows_evals() {
        let kt = ELECTRON_KT;
        let a = fermi_dirac(&[-0.4, -0.31, -0.3, 0.2], 2, kt).unwrap();
        let b = fermi_dirac(&[0.2, -0.3, -0.4, -0.31], 2, kt).unwrap();
        assert_eq!(b.f, vec![a.f[3], a.f[2], a.f[0], a.f[1]]);
        assert_eq!(a.mu, b.mu);
    }

    /// [MUST-1, SHOULD-5] A gap of 0.6 Ha = 120 kT: Σf == 2 in f64 over a plateau of μ, and μ is its midpoint.
    #[test]
    fn fd_gapped_plateau_midpoint() {
        let kt = ELECTRON_KT;
        let o = fermi_dirac(&[-0.5, -0.5, 0.1], 2, kt).unwrap();
        assert_eq!(o.f[0], 1.0);
        assert_eq!(o.f[1], 1.0);
        assert!(o.f[2] > 0.0 && o.f[2] <= (-36.0f64).exp(), "f_LUMO = {:e}", o.f[2]);
        assert!(o.entropy <= 1e-14, "S = {:e}", o.entropy);
        let mu = o.mu.unwrap();
        assert!((-0.5 + 30.0 * kt..=0.1 - 30.0 * kt).contains(&mu), "μ = {mu}");
        assert!((mu - (-0.2 + 0.5 * kt)).abs() <= 2.0 * kt, "μ = {mu}");
        eprintln!("plateau midpoint μ = {mu:.6} (−0.2 + 0.5 kT = {:.6}), halvings {}", -0.2 + 0.5 * kt, o.halvings);
    }

    #[test]
    fn fd_empty_channel() {
        let o = fermi_dirac(&[-0.3, 0.1], 0, ELECTRON_KT).unwrap();
        assert_eq!(o.f, vec![0.0, 0.0]);
        assert_eq!(o.mu, None);
        assert_eq!(o.entropy, 0.0);
        assert_eq!(o.halvings, 0);
        // n = 0 needs no extra level.
        assert_eq!(fermi_dirac(&[], 0, ELECTRON_KT).unwrap().f, Vec::<f64>::new());
    }

    /// 200 spectra shaped like a channel's levels: k ≤ 20 levels in [−spread, 0], spread log-uniform in
    /// [1e-6, 20] Ha (both ends included), 0 < n < k, in random order. After the bisection
    /// |Σf − n| ≲ (slope at μ)·ulp(μ) ≤ k·ulp(μ)/(4kT); levels are sparse where |μ| is large, so ≈ 1e-13.
    #[test]
    fn fd_sums_to_n() {
        let kt = ELECTRON_KT;
        let mut rng = Mix(0x5EED_F00D);
        let (mut worst, mut max_halvings) = (0.0f64, 0usize);
        for t in 0..200 {
            let k = 2 + (rng.next() % 19) as usize;
            let n = 1 + (rng.next() % (k as u64 - 1)) as usize;
            let spread = match t { 0 => 1e-6, 1 => 20.0, _ => 1e-6 * (2e7f64).powf(rng.unit()) };
            let evals: Vec<f64> = (0..k).map(|_| -spread * rng.unit()).collect();
            let o = fermi_dirac(&evals, n, kt).unwrap();
            let dev = (sum(&o.f) - n as f64).abs();
            assert!(dev <= 1e-12, "set {t}: k {k}, n {n}, spread {spread:e}: |Σf − n| = {dev:e}");
            assert!(o.halvings < MAX_MU_HALVINGS, "set {t}: {} halvings", o.halvings);
            assert!(o.f.iter().all(|&f| (0.0..=1.0).contains(&f)));
            worst = worst.max(dev);
            max_halvings = max_halvings.max(o.halvings);
        }
        eprintln!("fd_sums_to_n: max |Σf − n| = {worst:e}, max halvings {max_halvings}");
    }

    /// [SHOULD-1] A NaN never equals a bisection endpoint, so it must be refused before the bisection.
    #[test]
    fn fd_non_finite_is_numerical() {
        for evals in [[-0.3, f64::NAN], [-0.3, f64::INFINITY], [f64::NEG_INFINITY, 0.1]] {
            assert!(matches!(fermi_dirac(&evals, 1, ELECTRON_KT), Err(QmError::Numerical(_))), "{evals:?}");
        }
    }

    /// Too few levels (k < n + 1) and a kT that is not finite and > 0 are refused too.
    #[test]
    fn fd_bad_count_or_kt_is_numerical() {
        assert!(matches!(fermi_dirac(&[-0.3, 0.1], 2, ELECTRON_KT), Err(QmError::Numerical(_))));
        assert!(matches!(fermi_dirac(&[-0.3], 1, ELECTRON_KT), Err(QmError::Numerical(_))));
        for kt in [0.0, -1e-3, f64::NAN, f64::INFINITY] {
            assert!(matches!(fermi_dirac(&[-0.3, 0.1], 1, kt), Err(QmError::Numerical(_))), "kT = {kt}");
        }
    }

    /// [NOTE-1] μ at 0, where the f64 spacing reaches subnormals: still far under the backstop.
    #[test]
    fn fd_halvings_near_zero_mu() {
        let o = fermi_dirac(&[-1e-300, 1e-300], 1, ELECTRON_KT).unwrap();
        assert!(o.halvings < MAX_MU_HALVINGS, "{} halvings", o.halvings);
        assert!((sum(&o.f) - 1.0).abs() <= 1e-15);
        eprintln!("μ ≈ 0: μ = {:e}, halvings {}", o.mu.unwrap(), o.halvings);
    }

    #[test]
    fn fd_is_deterministic() {
        let evals = [-0.71, -0.30, -0.297, -0.2955, 0.05, -0.30 + 1e-9];
        let (a, b) = (fermi_dirac(&evals, 3, ELECTRON_KT).unwrap(), fermi_dirac(&evals, 3, ELECTRON_KT).unwrap());
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a.f), bits(&b.f));
        assert_eq!(a.mu.unwrap().to_bits(), b.mu.unwrap().to_bits());
        assert_eq!(a.entropy.to_bits(), b.entropy.to_bits());
        assert_eq!(a.halvings, b.halvings);
    }

    /// s(x) from x against −f ln f − (1−f) ln(1−f) from f = 1/(1 + eˣ), over f in [1e-12, 1 − 1e-12]: 500 points
    /// log-spaced in [1e-12, ½] and their mirrors 1 − f (the tails are where a cancellation would show).
    #[test]
    fn fd_entropy_matches_direct_formula() {
        let mut worst = 0.0f64;
        for j in 0..500 {
            let low = 10f64.powf(-12.0 + (12.0 - LN_2 / std::f64::consts::LN_10) * j as f64 / 499.0);
            for f in [low, 1.0 - low] {
                let x = ((1.0 - f) / f).ln();
                let direct = -f * f.ln() - (1.0 - f) * (1.0 - f).ln();
                let d = (fd_entropy(x) - direct).abs();
                assert!(d <= 1e-14, "f = {f:e}, x = {x}: |Δs| = {d:e}");
                worst = worst.max(d);
            }
        }
        eprintln!("fd_entropy vs direct: max |Δs| = {worst:e}");
        for x in [800.0, -800.0] {
            let s = fd_entropy(x);
            assert!(s.is_finite() && s <= 1e-300, "s({x}) = {s:e}");
        }
        assert!((fd_entropy(2f64.ln()) - 0.636514).abs() <= 1e-6, "s(⅓) = {}", fd_entropy(2f64.ln()));
        assert!((fd_entropy(0.0) - LN_2).abs() <= 1e-15);
    }

    #[test]
    fn witness_both_arms() {
        // ε_(2) = −0.3, the levels in any order.
        let lvl = |top: f64| vec![-0.3, top, -0.5];
        // Arm 1 fails (ε_max = ε_(n) + 0.099), arm 2 holds (0.199 Ha ≥ 36 kT = 0.18 Ha).
        assert!(!witness_reached(&lvl(-0.3 + 0.099), 2, -0.4, 5e-3));
        // Arm 1 met (ε_max = ε_(n) + 0.2), arm 2 fails at kT = 3e-3: ε_max − μ = 0.105 < 36 kT = 0.108.
        assert!(!witness_reached(&lvl(-0.1), 2, -0.1 - 0.105, 3e-3));
        // Both met.
        assert!(witness_reached(&lvl(-0.1), 2, -0.1 - 0.2, 3e-3));
        // No level above the n-th: never reached.
        assert!(!witness_reached(&[-0.5, -0.3], 2, -0.9, 3e-3));
        // An empty channel needs no witness.
        assert!(witness_reached(&[], 0, 0.0, 3e-3));
    }

    /// Plan Task 9b: the tight count follows f and the Δρ stop, not position. One electron, a 0.2 Ha gap: μ at mid-gap,
    /// the level above holds f = e^{−20} ≈ 2.1e-9.
    #[test]
    fn tight_columns_follow_f_and_the_stop() {
        let kt = ELECTRON_KT;
        let evals = [-0.5, -0.3, 0.0, 0.1];
        let zero = [0.0; 4];
        let mu = fermi_dirac(&evals, 1, kt).unwrap().mu;
        let f1 = 1.0 / (1.0 + ((-0.3 - mu.unwrap()) / kt).exp());
        assert!((1e-9..1e-8).contains(&f1), "f(−0.3) = {f1:e}");
        // tol 1e-6: limit 1e-8 > f1 + the tail: the level above is loose
        assert_eq!(tight_columns(&evals, &zero, mu, kt, 1e-6), 1);
        // tol 1e-7: limit 1e-9 < f1: it stays tight (the threshold comes from the stop)
        assert_eq!(tight_columns(&evals, &zero, mu, kt, 1e-7), 2);
        // its residual raises the upper bound f̂ = f(θ − r): r = 0.03 Ha gives e^{6}·f1 ≈ 8.6e-7 ≥ 1e-8
        assert_eq!(tight_columns(&evals, &[0.0, 0.03, 0.0, 0.0], mu, kt, 1e-6), 2);
        // positions, not ranks: the same levels out of order — the tight level sits at position 1, so 2 columns
        assert_eq!(tight_columns(&[-0.3, -0.5, 0.1, 0.0], &zero, mu, kt, 1e-6), 2);
        // a partly occupied level between the occupied one and the witness stays tight whatever its position
        let p = [-0.5, -0.49, 0.3];
        let mp = fermi_dirac(&p, 1, kt).unwrap().mu;
        assert_eq!(tight_columns(&p, &[0.0; 3], mp, kt, 1e-6), 2);
        // the tail is a sum: twenty levels each at 1e-9 (alone below 1e-8) are not all loose together
        let mut many = vec![-0.5];
        let e9 = mu.unwrap() + kt * (1e9f64).ln(); // f = 1/(1 + 1e9) ≈ 1e-9
        many.extend(std::iter::repeat_n(e9, 20));
        many.push(0.3);
        let t = tight_columns(&many, &vec![0.0; many.len()], mu, kt, 1e-6);
        assert!((2..=many.len() - 9).contains(&t), "tight {t} of {}", many.len());
        // an empty channel has no tight column
        assert_eq!(tight_columns(&[0.1, 0.2], &[0.0, 0.0], None, kt, 1e-6), 0);
        // a non-finite residual is never loose
        assert_eq!(tight_columns(&evals, &[0.0, 0.0, 0.0, f64::NAN], mu, kt, 1e-6), 4);
    }

    /// Plan Task 9b: the witness test reads θ − r, never the Ritz value alone.
    #[test]
    fn witness_reads_the_lower_bound() {
        let kt = ELECTRON_KT;
        let mu = -0.4;
        let top = mu + 36.0 * kt + 0.001; // the Ritz value clears the tail arm by 1 mHa (the gap arm by far)
        let evals = [-0.5, top, -0.3];
        assert!(witness_reached(&evals, 1, mu, kt));
        assert!(witness_bound_reached(&evals, &[0.0, 0.0009, 0.0], 1, mu, kt), "θ − r still clears it");
        assert!(!witness_bound_reached(&evals, &[0.0, 0.002, 0.0], 1, mu, kt), "θ − r = top − 2 mHa does not");
        // the gap arm on the lower bound: ε_(1) = −0.5, top − r must be ≥ −0.4
        let near = [-0.5, -0.399];
        assert!(witness_reached(&near, 1, -0.9, kt));
        assert!(!witness_bound_reached(&near, &[0.0, 0.002], 1, -0.9, kt));
        assert!(!witness_bound_reached(&evals, &[0.0, f64::NAN, 0.0], 1, mu, kt), "a NaN residual never passes");
        assert!(witness_bound_reached(&[], &[], 0, 0.0, kt));
        assert!(!witness_bound_reached(&[-0.5], &[0.0], 1, -0.9, kt));
    }

    /// Plan Task 9b: the trim keeps the lowest witness and one passing spare above it.
    #[test]
    fn witness_keep_trims_the_overshoot() {
        let kt = ELECTRON_KT;
        let mu = -0.4;
        let evals = [-0.5, -0.3, -0.25, -0.1, -0.05, 0.0, 0.3];
        let zero = [0.0; 7];
        // the first level ≥ μ + 0.18 = −0.22 (and ≥ ε_(1) + 0.1 = −0.4) is −0.1 at rank 3; the spare −0.05 passes too
        assert_eq!(witness_keep(&evals, &zero, 1, mu, kt), 5);
        // the spare's residual fails it: the witness alone on top
        assert_eq!(witness_keep(&evals, &[0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0], 1, mu, kt), 4);
        // the witness's own residual pushes it up a rank
        assert_eq!(witness_keep(&evals, &[0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0], 1, mu, kt), 6);
        // whatever is kept passes the witness on its own
        for res in [zero, [0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0]] {
            let k = witness_keep(&evals, &res, 1, mu, kt);
            assert!(witness_bound_reached(&evals[..k], &res[..k], 1, mu, kt));
        }
        assert_eq!(witness_keep(&evals, &zero, 0, mu, kt), 7, "an empty channel keeps its block");
    }

    /// `witness_keep_within`: the lowest tested level that clears both arms on its lower bound **at a residual within
    /// `r_max`** decides, with `witness_keep`'s spare rule above it; a passing level above `r_max` is skipped (it is not
    /// yet a lower bound, premise M2), and none at all is `None` (the block must grow).
    #[test]
    fn witness_keep_within_reads_the_lowest_converged_witness() {
        let kt = ELECTRON_KT;
        let mu = -0.4; // the arms: lower bound ≥ μ + 0.18 = −0.22 and ≥ ε_(1) + 0.1 = −0.4
        let evals = [-0.5, -0.3, -0.25, -0.1, -0.05, 0.0, 0.3];
        let r_max = 1.5e-3;
        // everything converged: `witness_keep`'s answer (−0.1 at rank 3 and its spare −0.05)
        let zero = [0.0; 7];
        assert_eq!(witness_keep_within(&evals, &zero, 1, mu, kt, WITNESS_TAIL_KT, r_max), Some(witness_keep(&evals, &zero, 1, mu, kt)));
        // the top (rank 6) above r_max, as a floor exit leaves it: rank 3 still decides, the same keep
        let top_loose = [0.0, 0.0, 0.0, 1e-4, 0.0, 0.0, 3e-3];
        assert_eq!(witness_keep_within(&evals, &top_loose, 1, mu, kt, WITNESS_TAIL_KT, r_max), Some(5));
        // the first passing level (rank 3) above r_max: rank 4 decides; its spare rank 5 passes on its lower bound
        let first_loose = [0.0, 0.0, 0.0, 2e-3, 1e-3, 3e-3, 3e-3];
        assert_eq!(witness_keep_within(&evals, &first_loose, 1, mu, kt, WITNESS_TAIL_KT, r_max), Some(6));
        // … and with only the top within r_max, the top decides alone
        let only_top = [0.0, 0.0, 0.0, 2e-3, 2e-3, 2e-3, 1e-3];
        assert_eq!(witness_keep_within(&evals, &only_top, 1, mu, kt, WITNESS_TAIL_KT, r_max), Some(7));
        // no passing level within r_max: no witness
        let none = [0.0, 0.0, 0.0, 2e-3, 2e-3, 2e-3, 2e-3];
        assert_eq!(witness_keep_within(&evals, &none, 1, mu, kt, WITNESS_TAIL_KT, r_max), None);
        // a level within r_max whose lower bound misses the tail arm is no witness (rank 2, −0.25 < −0.22)
        assert_eq!(witness_keep_within(&evals[..3], &zero[..3], 1, mu, kt, WITNESS_TAIL_KT, r_max), None);
        // the tail arm is the caller's (a loosened step's 6 kT window lets rank 1 decide, rank 2 its spare)
        assert_eq!(witness_keep_within(&evals[..3], &zero[..3], 1, mu, kt, 6.0, r_max), Some(3));
        // whatever is kept passes the witness with its top level's residual within r_max, or its top is the spare
        for res in [zero, top_loose, first_loose, only_top] {
            let k = witness_keep_within(&evals, &res, 1, mu, kt, WITNESS_TAIL_KT, r_max).expect("a witness");
            assert!(witness_bound_reached(&evals[..k], &res[..k], 1, mu, kt));
            assert!(res[k - 1] <= r_max || res[k - 2] <= r_max, "the witness is the kept top or the one below the spare");
        }
        // an empty channel keeps its block, a block without a level above n has none
        assert_eq!(witness_keep_within(&evals, &zero, 0, mu, kt, WITNESS_TAIL_KT, r_max), Some(7));
        assert_eq!(witness_keep_within(&evals[..1], &zero[..1], 1, mu, kt, WITNESS_TAIL_KT, r_max), None);
        // order-independent: the same block shuffled keeps the same count
        let (e2, r2) = ([0.0, -0.5, 0.3, -0.1, -0.3, -0.05, -0.25], [0.0, 0.0, 3e-3, 2e-3, 0.0, 1e-3, 0.0]);
        assert_eq!(witness_keep_within(&e2, &r2, 1, mu, kt, WITNESS_TAIL_KT, r_max), Some(6));
    }

    #[test]
    fn growth_doubles_the_extra_block() {
        let seq: Vec<usize> = std::iter::successors(Some(1usize), |&e| (e < 32).then(|| grown_extras(e, 32))).collect();
        assert_eq!(seq, vec![1, 2, 4, 8, 16, 32]);
        assert_eq!(grown_extras(0, 32), 1);
        assert_eq!(grown_extras(3, 32), 6);
        assert_eq!(grown_extras(20, 32), 32);
        assert_eq!(grown_extras(4, 5), 5);
    }

    #[test]
    fn partial_levels_threshold() {
        let evals = [0.2, -0.1, 0.05, -0.4, 0.3];
        let f = [1e-6, 0.5, 9.9e-7, 1.0, 1.0 - 9.9e-7];
        let p = partial_levels(&evals, &f, 1);
        assert_eq!(
            p,
            vec![
                PartialLevel { spin: 1, index: 1, eps: -0.1, f: 0.5 },
                PartialLevel { spin: 1, index: 3, eps: 0.2, f: 1e-6 },
            ]
        );
        assert!(partial_levels(&[-0.5, 0.2], &[1.0, 0.0], 0).is_empty());
    }

    /// Task 9c (b): X = ln(n_w/(c·Δρ_prev)) kT, clamped to [6, 36]; no Δρ yet (∞, NaN, 0) is the full window.
    #[test]
    fn step_tail_follows_the_step_accuracy() {
        // n_w = 10, Δρ = 1e-2: ln(10 / 1e-5) = ln 1e6 = 13.8155
        assert!((step_tail_kt(10, 1e-2) - (1e6f64).ln()).abs() < 1e-12);
        assert!((step_tail_kt(10, 10.0) - (1e3f64).ln()).abs() < 1e-12); // ln 1000 = 6.91, just above the floor
        assert_eq!(step_tail_kt(10, 100.0), STEP_TAIL_MIN_KT); // ln 100 = 4.6 → 6
        assert_eq!(step_tail_kt(10, 1e-15), WITNESS_TAIL_KT); // ln 1e19 = 43.7 → 36
        for d in [f64::INFINITY, f64::NAN, 0.0, -1.0] { assert_eq!(step_tail_kt(7, d), WITNESS_TAIL_KT); }
        // monotone: a smaller Δρ_prev never shortens the window
        let xs: Vec<f64> = (0..60).map(|e| step_tail_kt(8, 10f64.powf(2.0 - e as f64 / 4.0))).collect();
        assert!(xs.windows(2).all(|w| w[1] >= w[0]) && xs[0] == STEP_TAIL_MIN_KT && xs[59] == WITNESS_TAIL_KT, "{xs:?}");
    }
}
