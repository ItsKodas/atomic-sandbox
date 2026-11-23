//! Learning the chemical-channel mix W of the channel surrogate (spec
//! `docs/superpowers/specs/2026-11-22-chemical-channels-design.md`, "Learning W" as amended by its
//! "Physics review, 2026-11-22: binding changes" I2, I4 and C1; the review in full:
//! `docs/superpowers/research/2026-11-22-chemical-channels-physics-review.md`; background:
//! `docs/superpowers/research/2026-11-22-chemical-channels-literature.md` §1, tensor-reduced ACE).
//! Pure f64, CPU, single-threaded.
//!
//! # What W is
//!
//! The channel model mixes the per-species neighbour densities of the atomic basis
//! ([`crate::descriptor`]) into K chemical channels,
//!
//!   Ā_{k,(n,lm)}(i) = Σ_t W[t, k] A_{(t,n),lm}(i),      W ∈ ℝ^{S×K}, stored `w[t·K + k]`,
//!
//! and builds the body-3/4 products over channel indices (k, n, l) instead of species (t, n, l).
//! Body 2 stays per species pair. Only col-span(W) sets the model's span (review I2c: the degree
//! caps involve n and l, never k, so an invertible K×K transform maps the term set to itself);
//! W is stored with **orthonormal columns** (column scale and sign are exact gauges under the
//! fit's per-feature standardisation, a rotation within the span changes only the prior).
//!
//! **Rule 1 (CLAUDE.md, the signed-off exception of 2026-11-22).** W is a per-element parameter,
//! so it is learnt here **only from our own DFT samples**, through a fit of those samples, and is
//! never initialised from or regularised towards a tabulated property (electronegativity, radii,
//! group). The species index t is categorical (sorted Z), as everywhere in the surrogate.
//!
//! # How W is learnt (spec "Learning W"; review I2, I4)
//!
//! 1. **The low-order species fit.** A body-2 + body-3 species model on the samples, in the reduced
//!    basis of review I4 ([`LowOrder::REVIEW_I4`]: n_max 4, l_max 2, degree cap 2, no body 4), by
//!    the ordinary fit ([`Accumulator`], evidence-chosen λ and noise levels; review M7: W comes from
//!    the posterior **mean**). Its posterior over the Q′ standardised reduced coefficients θ is
//!    N(m, Σ_low), Σ_low = A⁻¹, A = L Lᵀ (`Model::sigma_inv_chol`). **Cost:** a body-4-free basis
//!    goes through the reference hyperparameter search (`Accumulator::solve`, ∝ Q′³ per point):
//!    60 s at S = 5 (Q′ 735, 1 000 six-atom samples, 6 rounds; `learn_mix_timing_probe`), so
//!    ~400× that at S = 10 (Q′ 5 420). Beyond S ≈ 6 the low-order fit needs the spectral search
//!    of the large-basis path, which `Accumulator::refit` selects only for a body-4 basis today.
//!
//! 2. **The symmetric coefficient tensor (I2a).** A body-3 column class i is (centre s, stored term
//!    q = (t₁n₁, t₂n₂, l) with channel c₁ = t₁n_max + n₁ ≤ c₂), with raw coefficient
//!    c_i = d_i θ_i, d_i = 1/scale_k (k = n_body2 + q; body-3 classes are single columns). The
//!    stored feature is Σ_m A_{c₁,lm} A_{c₂,lm} once per unordered pair, so as a symmetric
//!    quadratic form over ordered channel pairs
//!
//!      E_i^(3) = Σ_l Σ_{c₁,c₂} C^{s_i,l}_{c₁c₂} A_{c₁,l}·A_{c₂,l},   C_{cc} = c_i,  C_{c₁c₂} = C_{c₂c₁} = c_i/2 (c₁ ≠ c₂),
//!
//!    which includes the same-species, different-n pairs. Without the halving the hetero entries
//!    would carry twice their weight and bias the subspace.
//!
//! 3. **Energy weighting (I2b).** a_t = the standardisation scale of the body-2 coordination
//!    feature b2[(t, 0)] (its pooled RMS over the energy rows; [`MixSetup::energy_weighting`]
//!    false sets a_t = 1 for the gate's comparison). With the normalised densities Ã_t = A_t/a_t
//!    the energy reads Σ C a_{t₁}a_{t₂} Ã Ã, so the weighted coefficients are energy per
//!    normalised density²: a raw coefficient that is large only because its feature is small (a
//!    usually distant or rare neighbour) does not dominate, and a rare species' noisy fibre pulls
//!    the subspace less while its row is still projected.
//!
//! 4. **The unfolding (spec step 2).** Rows t (one neighbour slot), columns
//!    j = (s, t′, n, n′, l) (everything else), N = S·S·n_max²·(l_max + 1) columns:
//!
//!      M̃[t, (s, t′, n, n′, l)] = a_t a_{t′} C^{s,l}_{(t n),(t′ n′)}.
//!
//!    C is symmetric in its two channel slots, so the unfolding along the other slot is the same
//!    matrix with its columns permuted: the symmetrisation over the two neighbour slots is built
//!    in. Absent terms (beyond the degree cap) are zeros.
//!
//! 5. **The left singular subspace (I2e).** G = M̃ M̃ᵀ (S×S, every entry a Neumaier-compensated
//!    sum), decomposed by the cyclic Jacobi [`symmetric_eigen`] (deterministic, no threads):
//!    G = Ũ Λ Ũᵀ, eigenpairs sorted by λ₁ ≥ … ≥ λ_S (ties by index), σ_j = √max(λ_j, 0) the
//!    singular values of M̃ ([`LearntMix::singular_values`]).
//!
//! 6. **K_eff and the identity (I2f).** K_eff = min(K, S, r) with the numerical rank
//!    r = #{λ_j > S·ε·λ₁}. Jacobi resolves the Gram's eigenvalues to O(ε‖G‖) = O(ε λ₁) absolutely,
//!    so this is I2f's "σ_k > S·ε·σ₁" moved to where it can be resolved: σ_j > √(S ε)·σ₁ (forming
//!    the Gram squares the condition number, which I2e accepts because only the subspace is used;
//!    a test of σ_j at S·ε·σ₁ would sit below the Gram's rounding). **When K_eff = S, W = I**: the
//!    exact species span, the code path the identity test proves, with Σ_W = 0 (a W that does not
//!    depend on the data carries no error from it) and ρ_t = 0.
//!
//! 7. **W (I2b, I2c, I2d).** The channel directions in raw-density units: Ā_k = Σ_t Ũ[t,k] Ã_t =
//!    Σ_t (Ũ[t,k]/a_t) A_t, i.e. B = D_a⁻¹ Ũ_K (D_a = diag a). I2b's W = D_a⁻¹Ũ_K and I2c's
//!    "orthonormal columns" are both met by storing an orthonormal basis of col-span(B): the thin
//!    QR B = Q R_qr by modified Gram–Schmidt run twice (deterministic, in singular-value order, so
//!    the dominant direction stays first), then each column's sign fixed so that its entry of
//!    largest |value| (lowest species index on a tie) is positive (I2d: the sign is an exact
//!    prediction gauge; this is for determinism and the file). So
//!
//!      W = B T,   T = R_qr⁻¹ S_g,   S_g = diag(±1).
//!
//!    With a_t = 1 for all t, R_qr = I and W = Ũ_K up to the signs.
//!
//! 8. **Diagnostics (I2f).** The per-species truncation residual
//!    ρ_t = ‖M̃_t − Ũ_{t,K} Σ_K V_Kᵀ‖² / ‖M̃_t‖² = Σ_{j > K_eff} Ũ_tj² λ_j⁺ / G_tt (row t of M̃ is
//!    Σ_j Ũ_tj σ_j v_jᵀ with orthonormal v_j); NaN for a species with no body-3 fibre (G_tt = 0:
//!    no information, which is not "perfectly represented"). The gap is σ_K/σ_{K+1}
//!    ([`LearntMix::gap`]).
//!
//! # The W covariance Σ_W = R Σ_low Rᵀ (review C1)
//!
//! **The review's form.** At the mean, V_K = M̃ᵀŨ_K Σ_K⁻¹, so W = D_a⁻¹ M̃ V_K Σ_K⁻¹ T. With the
//! right factor frozen at the mean, W(θ) = D_a⁻¹ M̃(θ) M̃(m)ᵀ Ũ_K Λ_K⁻¹ T, which is linear in θ
//! (M̃ is) and equals the stored W at θ = m. Write a perturbation δM̃ of the unfolding in the
//! complete eigenbasis of G, δŨ_K = Σ_m ũ_m γ_mk. The frozen form has, for every m,
//!
//!   γ_mk = ũ_mᵀ δM̃ p_k / λ_k,   p_k = M̃ᵀ ũ_k (= σ_k v_k).
//!
//! **Its first-order error, and the form implemented.** For m > K (off the span) first-order
//! perturbation theory of the symmetric G (Rayleigh–Schrödinger; Golub & Van Loan §8.1) gives
//! the exact coefficient
//!
//!   γ_mk = ũ_mᵀ δG ũ_k / (λ_k − λ_m),   δG = δM̃ M̃ᵀ + M̃ δM̃ᵀ,   i.e. (ũ_mᵀ δM̃ p_k + ũ_kᵀ δM̃ p_m)/(λ_k − λ_m),
//!
//! which equals the frozen form only when λ_m = 0 (a tensor of exactly K channels) and otherwise
//! differs at first order, by terms of relative size σ_m/σ_k and λ_m/(λ_k − λ_m). Measured on a
//! noisy planted fit (σ = 3.8, 2.3, 1.1, 0.87, K = 2): the frozen form's off-span variance is about
//! half of the sampled one (worst species 0.97 relative error), the exact one is within 3.3 %
//! (`sigma_w_is_the_exact_first_order_covariance_off_the_span`). **Implemented:** the exact γ_mk
//! off the span; within the span (m ≤ K) the frozen form's γ_mk is kept. Within-span changes are
//! the K² gauge directions δW = W δT that a refit of c absorbs (review C1.8: conservative); the
//! exact within-span derivative would divide by λ_k − λ_m between two *kept* eigenvalues, which
//! is ill-conditioned for a gauge, while the off-span division is by the kept/dropped gap — the
//! quantity that decides whether the subspace is determined at all (a gap below the Gram's
//! resolution is [`MixError::DegenerateGap`]). Freezing T (instead of re-orthonormalising) also
//! differs only within the span: the off-span part of δQ is (I − QQᵀ) δB R_qr⁻¹ either way.
//!
//! **R.** So δW = D_a⁻¹ (Σ_m ũ_m γ_mk) T, linear in δθ: vec W ↦ R θ with W[t, c] at row t·K + c.
//! Each body-3 class i of step 2 enters δM̃ = Σ_i δθ_i E_i through one or two entries of value
//! a_{t₁} a_{t₂} h d_i (h = 1 for c₁ = c₂, ½ otherwise): (t₁, (s, t₂, n₁, n₂, l)) and, for
//! c₁ ≠ c₂, (t₂, (s, t₁, n₂, n₁, l)); body-2 classes have R = 0. At the mean R m = W, as for the
//! review's form (γ(m) = δ_mk within the span, 0 off it). Then
//!
//!   Σ_W = R Σ_low Rᵀ = Xᵀ X,   X = L⁻¹ Rᵀ   (S·K forward substitutions through the fit's factor),
//!
//! an (S·K)² matrix with the cross-species terms, row-major over vec W
//! ([`LearntMix::sigma_w`]). The pieces held fixed are data scales, not coefficients: a_t and d_i.
//!
//! **Validity: η.** Σ_W is first order in the posterior spread of M̃. The diagnostic
//!
//!   η = √E‖δM̃‖²_F / (σ_K − σ_{K+1}),   E‖δM̃‖²_F = Σ_i n_i (a_{t₁}a_{t₂} h d_i)² Σ_low[i, i]
//!
//! ([`LearntMix::nonlinearity`]; Σ_low[i, i] by forward substitution, O(Q′³/6)) compares that
//! spread with the singular gap that holds the subspace (Wedin's sin Θ theorem: the subspace
//! moves by at most ‖δM̃‖₂/gap). Measured by drawing θ from the posterior and recomputing W
//! through the whole pipeline: at η = 0.067 the off-span covariance agrees with Σ_W within 6.4 %
//! (`sigma_w_holds_at_full_width_when_eta_is_small`); at η = 1.9 the sampled off-span variance is
//! 2.0× Σ_W's (`sigma_w_is_the_exact_first_order_covariance_off_the_span`, printed) — beyond
//! first order the subspace is not determined by the data and Σ_W **understates** it. A consumer
//! must not read Σ_W as an error bar when η is not small.
//!
//! **Behaviour (C1.4), with no new constant.** A species the data barely see has its body-3
//! fibre's posterior at the prior (1/λ in standardised units) and a small feature scale (large
//! d_i), so its Σ_W block is wide even when its mean row is small (`rare_species_have_wider_sigma_w`:
//! tr Σ_W per species 0.09 / 1.7e5 / 0.14 for H / C (12 of 312 samples) / O); a well-sampled
//! species has Σ_W ∝ ~1/n. Σ_W is **frozen with W** (C1.5): it is the error of the frozen W and is
//! recomputed only at a relearn. With W = I (K_eff = S) there is no W to be wrong: Σ_W = 0.
//!
//! **Σ_W is a diagnostic only** (spec "Physics review 2, 2026-11-23: Task 5 replaced", item 1;
//! `docs/superpowers/research/2026-11-23-chemical-channels-task4-task5-review.md` §C). The day-1
//! gate measured η = 242–6 030 ≫ 1, where Σ_W understates the subspace's spread by an unknown
//! factor, so C1's predictive term gᵀΣ_W g is **withdrawn**: nothing adds Σ_W to an error bar.
//! It, η and ρ_t are recorded in `channels.txt` with W; a species W cannot represent is refused
//! (below), not widened.
//!
//! # Which species W represents (review 2, items 2–3; the review's R1/R2)
//!
//! The provider installs the **represented** mix ([`MixSetup::eligible`] `Some`): each species
//! carries a flag, [`LearntMix::represented`], decided here at learn time, frozen with W and stored
//! by Z — **never inferred from W's values** (a zero row and a represented species with a small row
//! are different things). Species t is represented when both hold:
//!
//! - **(a) N_W** — the caller's eligibility mask: some many-body environment key with t among its
//!   neighbours meets its sample floor n_k ≥ q_k in the low-order fit (the existing per-key floor
//!   applied to the data W is learnt from; `providers::surrogate`, "Chemical channels: which
//!   species are represented");
//! - **(b) ρ_t ≤ [`RHO_MAX`] = 0.01**, the truncation residual of step 8. √ρ_t is the relative RMS
//!   error of t's weighted body-3 fibre after the truncation; requiring it ≤ the policy's 10 %
//!   relative force bound gives 0.1². Conservative: the many-body part is a fraction of a force,
//!   and a refit of c absorbs some of the discarded part. A NaN ρ (no fibre) fails.
//!
//! The procedure (single pass, review R2): the rows of M̃ of species failing (a) are set to zero
//! **before** the Gram, so a pair-only species neither spends a channel nor tilts the subspace;
//! K_eff = min(K, #eligible, rank); ρ_t is taken from that truncation (NaN for a species failing
//! (a): its Gram row is 0); species failing (b) are marked unrepresented too. When K_eff covers
//! every represented species W is **the identity on them** (S×S_rep, one unit column per
//! represented species in species order, ρ = 0 and Σ_W = 0 for them: I2f generalised — with every
//! species represented that is exactly W = I, the species path); otherwise B = D_a⁻¹Ũ_K with the
//! unrepresented rows set to zero goes through the thin QR and the sign rule as in step 7 (Σ_W and η
//! are computed from the zeroed-row unfolding: diagnostics). **Unrepresented rows are exact +0.0**
//! in every case, so `add_neighbour` skips them and the support of `crate::model` never reaches a
//! channel through them. With no species represented W is a single all-zero column (the body-3/4
//! blocks then vanish: a body-2 model).
//!
//! [`MixSetup::eligible`] `None` is the **raw** Task 2 mix the day-1 gate measured: every row used,
//! no ρ test, every species flagged represented. The provider never installs it.
//!
//! # Determinism
//!
//! The same samples, setup and K give bit-identical W and Σ_W on the same machine: the
//! accumulation of a body-4-free basis and the reference hyperparameter search are sequential
//! ([`Accumulator::add_many`] loops over `add` for such a basis; `Accumulator::solve`), and the
//! unfolding, the compensated Gram, Jacobi, the sort, Gram–Schmidt, the sign rule and the
//! substitutions run in a fixed order on one thread (`learn_mix_is_deterministic`).

use crate::descriptor::Basis;
use crate::fit::{Accumulator, FitError, FitOptions, FitReport, Sample};
use crate::linalg::{forward_substitute, symmetric_eigen};
use crate::model::{Model, Provenance};
use std::fmt;

/// The basis of the low-order species fit W is learnt from (review I4): body 2 + body 3 on
/// species, no body 4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LowOrder {
    pub n_max: usize,
    pub l_max: usize,
    pub degree_cap: usize,
}

impl LowOrder {
    /// Review I4's reduced basis: n_max 4, l_max 2, degree cap 2. Its Q′ is
    /// 4·S(S+1)/2 + S·P₃ with P₃ = (10 S² + 4 S)/2 under the cap: 735 / 2 832 / 5 420 at
    /// S = 5 / 8 / 10 (`low_order_column_counts`). The review's 795 / 2 976 / 5 640 are the same
    /// body 3 with an n_max 8 body 2 (under degree cap 2 the body-3 terms need n₁ + n₂ ≤ 2, so they
    /// do not depend on n_max ≥ 3; only the body-2 block does).
    pub const REVIEW_I4: LowOrder = LowOrder { n_max: 4, l_max: 2, degree_cap: 2 };

    /// The low-order basis over `species` at cutoff `r_cut` (Bohr).
    pub fn basis(&self, species: &[u8], r_cut: f64) -> Basis {
        let mut b = Basis::new(species);
        b.n_max = self.n_max;
        b.l_max = self.l_max;
        b.degree_cap = self.degree_cap;
        b.r_cut = r_cut;
        b.body3 = true;
        b.body4 = false;
        b
    }
}

/// What [`learn_mix`] needs besides the samples and K.
#[derive(Clone, Debug, PartialEq)]
pub struct MixSetup {
    /// Sorted, unique Z: the rows of W.
    pub species: Vec<u8>,
    /// Isolated-atom energy per species (Hartree, indexed like `species`), as for any fit.
    pub ref_energy: Vec<f64>,
    /// The collection's cutoff (Bohr).
    pub r_cut: f64,
    /// The fit options of the low-order fit (the collection's: `sigma_f_min` as there).
    pub fit: FitOptions,
    pub low_order: LowOrder,
    /// Review I2b's energy weighting (default on; off only to compare in the gate).
    pub energy_weighting: bool,
    /// Per species (indexed like `species`): whether it meets N_W (module doc, "Which species W
    /// represents", (a)). `Some`: the represented mix the provider installs; `None` (the default):
    /// the raw Task 2 mix (every row, no ρ test).
    pub eligible: Option<Vec<bool>>,
}

impl MixSetup {
    /// Defaults: review I4's low-order basis, default fit options, energy weighting on, the raw
    /// mix (`eligible` `None`).
    pub fn new(species: &[u8], ref_energy: &[f64], r_cut: f64) -> MixSetup {
        MixSetup {
            species: species.to_vec(),
            ref_energy: ref_energy.to_vec(),
            r_cut,
            fit: FitOptions::default(),
            low_order: LowOrder::REVIEW_I4,
            energy_weighting: true,
            eligible: None,
        }
    }
}

/// The largest truncation residual ρ_t of a represented species (module doc, "Which species W
/// represents", (b)): √ρ ≤ the policy's 10 % relative force bound.
pub const RHO_MAX: f64 = 0.01;

/// The representation rule (module doc, "Which species W represents"): (a) N_W met and (b)
/// ρ_t ≤ [`RHO_MAX`]; a NaN ρ (no fibre) fails.
pub fn is_represented(n_w_met: bool, rho: f64) -> bool {
    n_w_met && rho <= RHO_MAX
}

/// A learnt channel mix and its uncertainty (module doc).
#[derive(Clone, Debug, PartialEq)]
pub struct LearntMix {
    /// K_eff = min(K, S, numerical rank): the number of columns of `w`.
    pub k: usize,
    /// The K asked for.
    pub k_requested: usize,
    /// Sorted Z: row t of `w` belongs to `species[t]`.
    pub species: Vec<u8>,
    /// S×K row-major, `w[t * k + c]`, orthonormal columns; the identity when `k` = S.
    pub w: Vec<f64>,
    /// Σ_W = R Σ_low Rᵀ, (S·K)×(S·K) row-major over vec W (index t·K + c); zero when `k` = S.
    pub sigma_w: Vec<f64>,
    /// The S singular values of the weighted unfolding M̃, descending.
    pub singular_values: Vec<f64>,
    /// ρ_t per species (module doc, step 8); 0 when `k` = S, NaN for a species with no body-3 fibre.
    pub truncation_residual: Vec<f64>,
    /// η = √E‖δM̃‖²_F / (σ_K − σ_{K+1}) under the low-order posterior (module doc, "Validity"):
    /// Σ_W is first order and holds for η ≪ 1; 0 when `k` = S.
    pub nonlinearity: f64,
    /// The energy weights a_t (1 when unweighted).
    pub weights: Vec<f64>,
    /// Numerical rank of M̃ (module doc, step 6).
    pub rank: usize,
    /// The low-order fit's report (Q′, λ, σ: review M7 records them with W); `None` from
    /// [`mix_from_model`].
    pub low_order: Option<FitReport>,
    /// Per species: represented (module doc, "Which species W represents") — a flag decided at
    /// learn time and stored by Z, never read from `w`. An unrepresented species' row is exact 0.
    pub represented: Vec<bool>,
    /// Per species: N_W was met at learn time (the eligibility mask; all `true` for the raw mix).
    pub n_w_met: Vec<bool>,
    /// Per species: its many-body sample count at learn time (the provider's relearn trigger, review
    /// 2 item 8: a represented species' count doubling); 0 until the provider records it.
    pub many_body: Vec<usize>,
    /// The first audit-history model generation measured under this W (review 2 item 8: points of
    /// an earlier W leave the evidence counts and λ); 0 until the provider records it.
    pub history_from: u64,
}

impl LearntMix {
    /// Whether species `z` is represented (`false` for a species without a row).
    pub fn represents(&self, z: u8) -> bool {
        self.species.iter().position(|&s| s == z).is_some_and(|t| self.represented[t])
    }
}

impl LearntMix {
    /// σ_K/σ_{K+1} (∞ when σ_{K+1} = 0); `None` when K = S.
    pub fn gap(&self) -> Option<f64> {
        if self.k >= self.singular_values.len() {
            return None;
        }
        let (a, b) = (self.singular_values[self.k - 1], self.singular_values[self.k]);
        Some(if b > 0.0 { a / b } else { f64::INFINITY })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum MixError {
    /// K = 0 was asked for.
    ZeroK,
    /// The species list is empty, unsorted or has duplicates.
    Species,
    /// The model's basis has no body-3 block (or has body 4: the low-order fit never does).
    NotLowOrder,
    /// The model carries no posterior factor of the right size (Σ_W needs it).
    NoPosterior,
    /// The body-3 coefficient tensor is zero (no many-body information to project).
    NoManyBodySignal,
    /// λ_K and λ_{K+1} are equal to the Gram's resolution: the K-channel subspace is not
    /// determined (choose another K).
    DegenerateGap { k: usize },
    Fit(FitError),
}

impl fmt::Display for MixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MixError::ZeroK => write!(f, "a channel mix needs K ≥ 1"),
            MixError::Species => write!(f, "the species list must be non-empty, sorted and unique"),
            MixError::NotLowOrder => write!(f, "the low-order model must have body 3 and no body 4"),
            MixError::NoPosterior => write!(f, "the low-order model has no posterior factor"),
            MixError::NoManyBodySignal => write!(f, "the low-order fit's body-3 coefficients are all zero"),
            MixError::DegenerateGap { k } => write!(f, "singular values {k} and {} are equal: no unique {k}-channel subspace", k + 1),
            MixError::Fit(e) => write!(f, "low-order fit: {e}"),
        }
    }
}

impl std::error::Error for MixError {}

impl From<FitError> for MixError {
    fn from(e: FitError) -> MixError {
        MixError::Fit(e)
    }
}

impl From<crate::descriptor::DescribeError> for MixError {
    fn from(e: crate::descriptor::DescribeError) -> MixError {
        MixError::Fit(FitError::Describe(e))
    }
}

/// Learn the channel mix from `samples` (module doc): the low-order species fit, then
/// [`mix_from_model`]. Samples of a species outside `setup.species` are refused (the caller
/// leaves them out: review I5a).
pub fn learn_mix(samples: &[&Sample], setup: &MixSetup, k: usize) -> Result<LearntMix, MixError> {
    if k == 0 {
        return Err(MixError::ZeroK);
    }
    let sp = &setup.species;
    if sp.is_empty() || sp.windows(2).any(|p| p[0] >= p[1]) {
        return Err(MixError::Species);
    }
    let basis = setup.low_order.basis(sp, setup.r_cut);
    if samples.is_empty() {
        return Err(MixError::Fit(FitError::NoSamples));
    }
    if setup.eligible.as_ref().is_some_and(|e| e.len() != sp.len()) {
        return Err(MixError::Species);
    }
    let mut acc = Accumulator::new(&basis);
    acc.add_many(samples, &basis, &setup.ref_energy, &setup.fit)?;
    let (model, report) = acc.solve(&setup.fit, &low_order_provenance(), &setup.ref_energy)?;
    let mut mix = mix_from_model_with(&model, k, setup.energy_weighting, setup.eligible.as_deref())?;
    mix.low_order = Some(report);
    Ok(mix)
}

/// The raw mix of a fitted low-order model (module doc, steps 2–8, and Σ_W): body 3 on, body 4
/// off, with its posterior factor.
pub fn mix_from_model(model: &Model, k: usize, energy_weighting: bool) -> Result<LearntMix, MixError> {
    mix_and_map(model, k, energy_weighting, Map::Exact).map(|(mix, _)| mix)
}

/// [`mix_from_model`] with the eligibility mask of N_W (`Some`: the represented mix, module doc
/// "Which species W represents"; `None`: the raw mix).
pub fn mix_from_model_with(model: &Model, k: usize, energy_weighting: bool, eligible: Option<&[bool]>) -> Result<LearntMix, MixError> {
    mix_and_map_with(model, k, energy_weighting, Map::Exact, eligible).map(|(mix, _)| mix)
}

/// Which linear map [`mix_and_map`] builds Σ_W from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Map {
    /// The frozen form within the span, the exact first order off it (module doc).
    Exact,
    /// The review's frozen form everywhere (tests: the comparison of "Validity").
    #[cfg_attr(not(test), allow(dead_code))]
    Frozen,
    /// No Σ_W, η or R (tests: the Monte Carlo draws need W only).
    #[cfg_attr(not(test), allow(dead_code))]
    None,
}

/// [`mix_from_model`] and the linear map R of vec W = R θ ((S·K)×Q′ row-major; empty when W = I).
fn mix_and_map(model: &Model, k: usize, energy_weighting: bool, map: Map) -> Result<(LearntMix, Vec<f64>), MixError> {
    mix_and_map_with(model, k, energy_weighting, map, None)
}

/// [`mix_and_map`] with the N_W eligibility mask (`Some`: the represented mix, module doc "Which
/// species W represents").
fn mix_and_map_with(model: &Model, k: usize, energy_weighting: bool, map: Map, eligible: Option<&[bool]>) -> Result<(LearntMix, Vec<f64>), MixError> {
    if k == 0 {
        return Err(MixError::ZeroK);
    }
    let basis = &model.basis;
    if !basis.body3 || basis.body4 || basis.n_body3() == 0 {
        return Err(MixError::NotLowOrder);
    }
    let s_n = basis.n_species();
    if eligible.is_some_and(|e| e.len() != s_n) {
        return Err(MixError::Species);
    }
    let q_prime = basis.n_independent_columns();
    let chol = match &model.sigma_inv_chol {
        Some(l) if l.len() == q_prime * q_prime => l,
        _ => return Err(MixError::NoPosterior),
    };
    let a = weights(model, energy_weighting);
    let mut un = Unfolding::of(model, &a);
    // review R2: the rows of the species failing N_W are zero before the Gram (they neither spend
    // a channel nor tilt the subspace)
    if let Some(e) = eligible {
        for (t, &ok) in e.iter().enumerate() {
            if !ok {
                un.m[t * un.cols..(t + 1) * un.cols].iter_mut().for_each(|x| *x = 0.0);
            }
        }
    }
    if let Some(e) = eligible {
        return represented_mix(model, k, map, e, &a, &un, chol, q_prime);
    }

    // step 5: the compensated Gram and its Jacobi eigenpairs, descending
    let Spectrum { g, lambda, u_mat, singular_values, rank } = spectrum(&un, s_n)?;

    // step 6: K_eff
    let k_eff = k.min(s_n).min(rank);
    let species = basis.species.clone();
    if k_eff == s_n {
        let mut w = vec![0.0; s_n * s_n];
        for t in 0..s_n {
            w[t * s_n + t] = 1.0;
        }
        let mix = LearntMix {
            k: s_n,
            k_requested: k,
            species,
            w,
            sigma_w: vec![0.0; s_n * s_n * s_n * s_n],
            singular_values,
            truncation_residual: vec![0.0; s_n],
            nonlinearity: 0.0,
            weights: a,
            rank,
            low_order: None,
            represented: vec![true; s_n],
            n_w_met: vec![true; s_n],
            many_body: vec![0; s_n],
            history_from: 0,
        };
        return Ok((mix, Vec::new()));
    }

    // step 7: B = D_a⁻¹ Ũ_K, its thin QR, the sign rule; W = B T, T = R_qr⁻¹ S_g
    let kk = k_eff;
    let b: Vec<f64> = (0..s_n).flat_map(|t| (0..kk).map(move |c| (t, c))).map(|(t, c)| u_mat[t * s_n + c] / a[t]).collect();
    let (mut w, r_qr) = thin_qr(&b, s_n, kk);
    let mut signs = vec![1.0; kk];
    for c in 0..kk {
        let mut best = 0;
        for t in 1..s_n {
            if w[t * kk + c].abs() > w[best * kk + c].abs() {
                best = t;
            }
        }
        if w[best * kk + c] < 0.0 {
            signs[c] = -1.0;
            for t in 0..s_n {
                w[t * kk + c] = -w[t * kk + c];
            }
        }
    }
    let t_mat = upper_inverse_times_signs(&r_qr, kk, &signs);

    // step 8: ρ_t
    let truncation_residual: Vec<f64> = (0..s_n)
        .map(|t| {
            let gtt = g[t * s_n + t];
            if gtt > 0.0 {
                let mut r = Neumaier::default();
                for j in kk..s_n {
                    r.add(u_mat[t * s_n + j] * u_mat[t * s_n + j] * lambda[j].max(0.0));
                }
                r.sum() / gtt
            } else {
                f64::NAN
            }
        })
        .collect();

    // the kept/dropped gap must be resolvable for the subspace (and its derivative) to exist
    if kk < s_n && !(lambda[kk - 1] - lambda[kk] > s_n as f64 * f64::EPSILON * lambda[0]) {
        return Err(MixError::DegenerateGap { k: kk });
    }

    // Σ_W (module doc): R, then X = L⁻¹ Rᵀ, Σ_W = Xᵀ X; and the linearity diagnostic η
    let (mut sigma_w, mut nonlinearity, mut r) = (Vec::new(), f64::NAN, Vec::new());
    if map != Map::None {
        let classes = class_entries(model, &un, &a);
        nonlinearity = expected_unfolding_spread(chol, q_prime, &classes).sqrt() / (singular_values[kk - 1] - singular_values[kk]);
        r = linear_map(model, &un, &classes, &a, &u_mat, &lambda, &t_mat, kk, map == Map::Exact);
        let rows = s_n * kk;
        let mut x = r.clone();
        for row in x.chunks_mut(q_prime) {
            forward_substitute(chol, q_prime, row);
        }
        sigma_w = vec![0.0; rows * rows];
        for p in 0..rows {
            for q in 0..=p {
                let v = compensated_dot(&x[p * q_prime..(p + 1) * q_prime], &x[q * q_prime..(q + 1) * q_prime]);
                sigma_w[p * rows + q] = v;
                sigma_w[q * rows + p] = v;
            }
        }
    }

    let mix = LearntMix {
        k: kk,
        k_requested: k,
        species,
        w,
        sigma_w,
        singular_values,
        truncation_residual,
        nonlinearity,
        weights: a,
        rank,
        low_order: None,
        represented: vec![true; s_n],
        n_w_met: vec![true; s_n],
        many_body: vec![0; s_n],
        history_from: 0,
    };
    Ok((mix, r))
}

/// The Gram of the unfolding's rows and its eigenpairs, descending (module doc, steps 5–6).
struct Spectrum {
    g: Vec<f64>,
    lambda: Vec<f64>,
    /// Ũ[t, j], column j the j-th largest.
    u_mat: Vec<f64>,
    singular_values: Vec<f64>,
    rank: usize,
}

fn spectrum(un: &Unfolding, s_n: usize) -> Result<Spectrum, MixError> {
    let mut g = vec![0.0; s_n * s_n];
    for t in 0..s_n {
        for u in 0..=t {
            let v = compensated_dot(un.row(t), un.row(u));
            g[t * s_n + u] = v;
            g[u * s_n + t] = v;
        }
    }
    let (vals, vecs) = symmetric_eigen(&g, s_n);
    let mut order: Vec<usize> = (0..s_n).collect();
    order.sort_by(|&i, &j| vals[j].total_cmp(&vals[i]).then(i.cmp(&j)));
    let lambda: Vec<f64> = order.iter().map(|&j| vals[j]).collect();
    let u_mat: Vec<f64> = (0..s_n).flat_map(|t| order.iter().map(move |&j| (t, j))).map(|(t, j)| vecs[t * s_n + j]).collect();
    let singular_values: Vec<f64> = lambda.iter().map(|&l| l.max(0.0).sqrt()).collect();
    if !(lambda[0] > 0.0) {
        return Err(MixError::NoManyBodySignal);
    }
    let rank = lambda.iter().filter(|&&l| l > s_n as f64 * f64::EPSILON * lambda[0]).count();
    Ok(Spectrum { g, lambda, u_mat, singular_values, rank })
}

/// The represented mix (module doc, "Which species W represents"): `un`'s rows of the species
/// failing N_W already zero.
#[allow(clippy::too_many_arguments)]
fn represented_mix(model: &Model, k: usize, map: Map, eligible: &[bool], a: &[f64], un: &Unfolding, chol: &[f64], q_prime: usize) -> Result<(LearntMix, Vec<f64>), MixError> {
    let s_n = eligible.len();
    let species = model.basis.species.clone();
    let s_elig = eligible.iter().filter(|&&e| e).count();
    if s_elig == 0 {
        return Err(MixError::NoManyBodySignal);
    }
    let Spectrum { g, lambda, u_mat, singular_values, rank } = spectrum(un, s_n)?;
    let k_eff = k.min(s_elig).min(rank);
    // ρ_t of the K_eff truncation; NaN without a fibre (a species failing N_W has none here)
    let mut truncation_residual: Vec<f64> = (0..s_n)
        .map(|t| {
            let gtt = g[t * s_n + t];
            if !eligible[t] || !(gtt > 0.0) {
                return f64::NAN;
            }
            let mut r = Neumaier::default();
            for j in k_eff..s_n {
                r.add(u_mat[t * s_n + j] * u_mat[t * s_n + j] * lambda[j].max(0.0));
            }
            r.sum() / gtt
        })
        .collect();
    let represented: Vec<bool> = (0..s_n).map(|t| is_represented(eligible[t], truncation_residual[t])).collect();
    let reps: Vec<usize> = (0..s_n).filter(|&t| represented[t]).collect();
    let base = |k_out: usize, w: Vec<f64>, sigma_w: Vec<f64>, truncation_residual: Vec<f64>, nonlinearity: f64| LearntMix {
        k: k_out,
        k_requested: k,
        species: species.clone(),
        w,
        sigma_w,
        singular_values: singular_values.clone(),
        truncation_residual,
        nonlinearity,
        weights: a.to_vec(),
        rank,
        low_order: None,
        represented: represented.clone(),
        n_w_met: eligible.to_vec(),
        many_body: vec![0; s_n],
        history_from: 0,
    };
    if reps.is_empty() {
        // nothing represented: one all-zero channel (the body-3/4 blocks vanish)
        return Ok((base(1, vec![0.0; s_n], vec![0.0; s_n * s_n], truncation_residual, 0.0), Vec::new()));
    }
    if k_eff >= reps.len() {
        // K covers every represented species: the identity on them (I2f generalised)
        let kk = reps.len();
        let mut w = vec![0.0; s_n * kk];
        for (c, &t) in reps.iter().enumerate() {
            w[t * kk + c] = 1.0;
            truncation_residual[t] = 0.0;
        }
        let sk = s_n * kk;
        return Ok((base(kk, w, vec![0.0; sk * sk], truncation_residual, 0.0), Vec::new()));
    }
    let kk = k_eff;
    if !(lambda[kk - 1] - lambda[kk] > s_n as f64 * f64::EPSILON * lambda[0]) {
        return Err(MixError::DegenerateGap { k: kk });
    }
    // step 7 on the represented rows: B = D_a⁻¹ Ũ_K, unrepresented rows exact 0
    let b: Vec<f64> = (0..s_n).flat_map(|t| (0..kk).map(move |c| (t, c))).map(|(t, c)| if represented[t] { u_mat[t * s_n + c] / a[t] } else { 0.0 }).collect();
    let (mut w, r_qr) = thin_qr(&b, s_n, kk);
    if (0..kk).any(|c| !(r_qr[c * kk + c] > 0.0) || !r_qr[c * kk + c].is_finite()) {
        // the represented rows alone do not hold K independent directions
        return Err(MixError::DegenerateGap { k: kk });
    }
    let mut signs = vec![1.0; kk];
    for c in 0..kk {
        let mut best = 0;
        for t in 1..s_n {
            if w[t * kk + c].abs() > w[best * kk + c].abs() {
                best = t;
            }
        }
        if w[best * kk + c] < 0.0 {
            signs[c] = -1.0;
            for t in 0..s_n {
                w[t * kk + c] = -w[t * kk + c];
            }
        }
    }
    // exact +0.0 (a sign flip leaves −0.0)
    for t in (0..s_n).filter(|&t| !represented[t]) {
        w[t * kk..(t + 1) * kk].iter_mut().for_each(|x| *x = 0.0);
    }
    let t_mat = upper_inverse_times_signs(&r_qr, kk, &signs);
    // Σ_W and η: diagnostics (module doc, "Σ_W is a diagnostic only")
    let (mut sigma_w, mut nonlinearity, mut r) = (Vec::new(), f64::NAN, Vec::new());
    if map != Map::None {
        let classes = class_entries(model, un, a);
        nonlinearity = expected_unfolding_spread(chol, q_prime, &classes).sqrt() / (singular_values[kk - 1] - singular_values[kk]);
        r = linear_map(model, un, &classes, a, &u_mat, &lambda, &t_mat, kk, map == Map::Exact);
        let rows = s_n * kk;
        let mut x = r.clone();
        for row in x.chunks_mut(q_prime) {
            forward_substitute(chol, q_prime, row);
        }
        sigma_w = vec![0.0; rows * rows];
        for p in 0..rows {
            for q in 0..=p {
                let v = compensated_dot(&x[p * q_prime..(p + 1) * q_prime], &x[q * q_prime..(q + 1) * q_prime]);
                sigma_w[p * rows + q] = v;
                sigma_w[q * rows + p] = v;
            }
        }
    }
    Ok((base(kk, w, sigma_w, truncation_residual, nonlinearity), r))
}

/// The energy weights a_t (module doc, step 3): the scale of b2[(t, 0)], or 1.
fn weights(model: &Model, energy_weighting: bool) -> Vec<f64> {
    let basis = &model.basis;
    (0..basis.n_species())
        .map(|t| {
            if !energy_weighting {
                return 1.0;
            }
            let sc = model.scale[basis.body2_index(t, 0)];
            if sc > 0.0 {
                sc
            } else {
                1.0
            }
        })
        .collect()
}

/// The weighted unfolding M̃ (module doc, step 4), S rows of N columns.
struct Unfolding {
    s: usize,
    n_max: usize,
    n_l: usize,
    cols: usize,
    m: Vec<f64>,
}

impl Unfolding {
    fn col(&self, centre: usize, t2: usize, n1: usize, n2: usize, l: usize) -> usize {
        (((centre * self.s + t2) * self.n_max + n1) * self.n_max + n2) * self.n_l + l
    }

    fn row(&self, t: usize) -> &[f64] {
        &self.m[t * self.cols..(t + 1) * self.cols]
    }

    fn of(model: &Model, a: &[f64]) -> Unfolding {
        let basis = &model.basis;
        let s = basis.n_species();
        let (n_max, n_l) = (basis.n_max, basis.l_max + 1);
        let cols = s * s * n_max * n_max * n_l;
        let mut un = Unfolding { s, n_max, n_l, cols, m: vec![0.0; s * cols] };
        let (raw, _) = model.effective_coefficients();
        let p2 = basis.n_body2();
        let terms = basis.body3_terms();
        for (centre, row) in raw.iter().enumerate() {
            for (q, term) in terms.iter().enumerate() {
                let c = row[p2 + q];
                let same = term.t1 == term.t2 && term.n1 == term.n2;
                let h = if same { 1.0 } else { 0.5 };
                let v = a[term.t1] * a[term.t2] * h * c;
                let j = un.col(centre, term.t2, term.n1, term.n2, term.l);
                un.m[term.t1 * cols + j] = v;
                if !same {
                    let j = un.col(centre, term.t1, term.n2, term.n1, term.l);
                    un.m[term.t2 * cols + j] = v;
                }
            }
        }
        un
    }
}

/// One body-3 column class seen from the unfolding: the class index i, its value per unit θ_i in
/// M̃ (a_{t₁}a_{t₂} h d_i) and its one or two (row, column) entries (module doc, "Σ_W").
struct ClassEntries {
    i: usize,
    value: f64,
    entries: [(usize, usize); 2],
    n: usize,
}

/// The body-3 classes of the model with their unfolding entries.
fn class_entries(model: &Model, un: &Unfolding, a: &[f64]) -> Vec<ClassEntries> {
    let basis = &model.basis;
    let p = basis.n_features();
    let p2 = basis.n_body2();
    let terms = basis.body3_terms();
    let mut out = Vec::new();
    for (i, class) in basis.column_classes().iter().enumerate() {
        let (centre, feat) = (class[0] / p, class[0] % p);
        if feat < p2 || feat >= p2 + terms.len() {
            continue;
        }
        let term = terms[feat - p2];
        let sc = model.scale[feat];
        let d = 1.0 / if sc == 0.0 { 1.0 } else { sc };
        let same = term.t1 == term.t2 && term.n1 == term.n2;
        let h = if same { 1.0 } else { 0.5 };
        let e1 = (term.t1, un.col(centre, term.t2, term.n1, term.n2, term.l));
        let e2 = (term.t2, un.col(centre, term.t1, term.n2, term.n1, term.l));
        out.push(ClassEntries { i, value: a[term.t1] * a[term.t2] * h * d, entries: [e1, e2], n: if same { 1 } else { 2 } });
    }
    out
}

/// R of vec W = R θ (module doc, "Σ_W"), (S·K)×Q′ row-major: the frozen form within the span,
/// the exact first-order eigenvector derivative off it.
#[allow(clippy::too_many_arguments)]
fn linear_map(model: &Model, un: &Unfolding, classes: &[ClassEntries], a: &[f64], u_mat: &[f64], lambda: &[f64], t_mat: &[f64], kk: usize, exact: bool) -> Vec<f64> {
    let s_n = model.basis.n_species();
    let q_prime = model.basis.n_independent_columns();
    // p_m = M̃ᵀ ũ_m for every m (N×S, `pm[col * S + m]`)
    let mut pm = vec![0.0; un.cols * s_n];
    for col in 0..un.cols {
        for m in 0..s_n {
            let mut acc = Neumaier::default();
            for t in 0..s_n {
                acc.add(un.m[t * un.cols + col] * u_mat[t * s_n + m]);
            }
            pm[col * s_n + m] = acc.sum();
        }
    }
    let mut r = vec![0.0; s_n * kk * q_prime];
    // per class: x_mᵀ E_i p_b = value · Σ_entries x_m[t] p_b[col]
    let mut e_p = vec![0.0; s_n * s_n]; // (E_i p_b)[t] at t·S + b
    let mut gamma = vec![0.0; s_n * kk]; // the Ũ-basis coefficients of δŨ_K: γ[m·K + k]
    for ce in classes {
        e_p.iter_mut().for_each(|x| *x = 0.0);
        for &(t, col) in &ce.entries[..ce.n] {
            for b in 0..s_n {
                e_p[t * s_n + b] += ce.value * pm[col * s_n + b];
            }
        }
        // ũ_mᵀ E_i p_b
        let proj = |m: usize, b: usize| -> f64 { (0..s_n).map(|t| u_mat[t * s_n + m] * e_p[t * s_n + b]).sum() };
        for k in 0..kk {
            for m in 0..s_n {
                gamma[m * kk + k] = if m < kk || !exact {
                    // within the span: the frozen form's ũ_mᵀ δM̃ M̃ᵀ ũ_k / λ_k
                    proj(m, k) / lambda[k]
                } else {
                    // off the span: ũ_mᵀ δG ũ_k / (λ_k − λ_m), δG = δM̃ M̃ᵀ + M̃ δM̃ᵀ
                    (proj(m, k) + proj(k, m)) / (lambda[k] - lambda[m])
                };
            }
        }
        // δW[t, c] = (1/a_t) Σ_k (Σ_m Ũ[t, m] γ[m, k]) T[k, c]
        for t in 0..s_n {
            for k in 0..kk {
                let du: f64 = (0..s_n).map(|m| u_mat[t * s_n + m] * gamma[m * kk + k]).sum::<f64>() / a[t];
                for c in 0..kk {
                    r[(t * kk + c) * q_prime + ce.i] += du * t_mat[k * kk + c];
                }
            }
        }
    }
    r
}

/// E‖δM̃‖²_F under the posterior: Σ_i n_i value_i² Σ_low[i, i], with Σ_low[i, i] = ‖L⁻¹ e_i‖²
/// (forward substitution from row i).
fn expected_unfolding_spread(chol: &[f64], q_prime: usize, classes: &[ClassEntries]) -> f64 {
    let mut total = Neumaier::default();
    let mut x = vec![0.0; q_prime];
    for ce in classes {
        let i = ce.i;
        x[i] = 1.0 / chol[i * q_prime + i];
        let mut norm = Neumaier::default();
        norm.add(x[i] * x[i]);
        for r in i + 1..q_prime {
            let row = &chol[r * q_prime + i..r * q_prime + r];
            let s: f64 = row.iter().zip(&x[i..r]).map(|(l, v)| l * v).sum();
            x[r] = -s / chol[r * q_prime + r];
            norm.add(x[r] * x[r]);
        }
        total.add(ce.n as f64 * ce.value * ce.value * norm.sum());
    }
    total.sum()
}

/// Thin QR of the S×K row-major `b` by modified Gram–Schmidt run twice ("twice is enough"):
/// (Q S×K with orthonormal columns, R K×K upper triangular), b = Q R.
fn thin_qr(b: &[f64], s_n: usize, kk: usize) -> (Vec<f64>, Vec<f64>) {
    let mut q = vec![0.0; s_n * kk];
    let mut r = vec![0.0; kk * kk];
    for c in 0..kk {
        let mut v: Vec<f64> = (0..s_n).map(|t| b[t * kk + c]).collect();
        for _ in 0..2 {
            for j in 0..c {
                let proj: f64 = compensated_dot_strided(&q, j, kk, &v);
                r[j * kk + c] += proj;
                for t in 0..s_n {
                    v[t] -= proj * q[t * kk + j];
                }
            }
        }
        let norm = compensated_dot(&v, &v).sqrt();
        r[c * kk + c] = norm;
        for t in 0..s_n {
            q[t * kk + c] = v[t] / norm;
        }
    }
    (q, r)
}

/// T = R⁻¹ S_g for the K×K upper-triangular R (back substitution per column).
fn upper_inverse_times_signs(r: &[f64], kk: usize, signs: &[f64]) -> Vec<f64> {
    let mut t = vec![0.0; kk * kk];
    for c in 0..kk {
        for i in (0..=c).rev() {
            let mut s = if i == c { signs[c] } else { 0.0 };
            for j in i + 1..=c {
                s -= r[i * kk + j] * t[j * kk + c];
            }
            t[i * kk + c] = s / r[i * kk + i];
        }
    }
    t
}

fn low_order_provenance() -> Provenance {
    Provenance {
        grid_n: 0,
        h: 0.0,
        sigma: 0.0,
        tol: 0.0,
        charge: 0,
        multiplicity: None,
        teacher: "channels: low-order fit".to_string(),
        masses: Vec::new(),
        ionic: String::new(),
    }
}

/// Neumaier compensated sum.
#[derive(Default)]
struct Neumaier {
    sum: f64,
    comp: f64,
}

impl Neumaier {
    fn add(&mut self, x: f64) {
        let t = self.sum + x;
        if self.sum.abs() >= x.abs() {
            self.comp += (self.sum - t) + x;
        } else {
            self.comp += (x - t) + self.sum;
        }
        self.sum = t;
    }

    fn sum(&self) -> f64 {
        self.sum + self.comp
    }
}

fn compensated_dot(x: &[f64], y: &[f64]) -> f64 {
    let mut acc = Neumaier::default();
    for (a, b) in x.iter().zip(y) {
        acc.add(a * b);
    }
    acc.sum()
}

/// Σ_t q[t·kk + j]·v[t].
fn compensated_dot_strided(q: &[f64], j: usize, kk: usize, v: &[f64]) -> f64 {
    let mut acc = Neumaier::default();
    for (t, vt) in v.iter().enumerate() {
        acc.add(q[t * kk + j] * vt);
    }
    acc.sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::SampleTag;
    use crate::model::{energy_and_forces, SupportCache};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::sync::OnceLock;

    const R_CUT: f64 = 6.0;
    const EPS: f64 = f64::EPSILON;

    fn prov() -> Provenance {
        Provenance {
            grid_n: 49,
            h: 0.25,
            sigma: 0.25,
            tol: 1e-6,
            charge: 0,
            multiplicity: None,
            teacher: "synthetic".to_string(),
            masses: Vec::new(),
            ionic: "none".into(),
        }
    }

    fn normal(rng: &mut StdRng) -> f64 {
        let u1: f64 = rng.gen_range(f64::EPSILON..1.0);
        let u2: f64 = rng.gen_range(0.0..1.0);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Random cluster in a cube of side `side`, no pair closer than 1 Bohr.
    fn cluster(rng: &mut StdRng, n: usize, side: f64) -> Vec<[f64; 3]> {
        loop {
            let pos: Vec<[f64; 3]> = (0..n).map(|_| [rng.gen_range(0.0..side), rng.gen_range(0.0..side), rng.gen_range(0.0..side)]).collect();
            let ok = (0..n).all(|i| (i + 1..n).all(|j| (0..3).map(|a| (pos[i][a] - pos[j][a]).powi(2)).sum::<f64>() > 1.0));
            if ok {
                return pos;
            }
        }
    }

    /// A low-order teacher whose body-3 raw coefficients factor through `w_star` (S×K):
    /// C^{s,l}_{(t n),(t′ n′)} = Σ_{k,k′} W*[t,k] W*[t′,k′] Γ^{s,l}_{(k n),(k′ n′)} with Γ symmetric
    /// and random, stored as C (c₁ = c₂) or 2C (c₁ ≠ c₂, the unordered term carries both orders).
    /// Body 2 random; the standardisation scale random in [0.5, 2] (so the weights a_t and d_i
    /// are exercised); the posterior factor the identity (Σ_low = I).
    fn planted_teacher(rng: &mut StdRng, species: &[u8], w_star: &[f64], k: usize) -> Model {
        let basis = LowOrder::REVIEW_I4.basis(species, R_CUT);
        let s_n = basis.n_species();
        let p = basis.n_features();
        let p2 = basis.n_body2();
        let n_max = basis.n_max;
        let n_l = basis.l_max + 1;
        let nc = k * n_max;
        let mut gamma = vec![vec![vec![0.0; nc * nc]; n_l]; s_n];
        for g_s in gamma.iter_mut() {
            for g in g_s.iter_mut() {
                for a in 0..nc {
                    for b in a..nc {
                        let v = rng.gen_range(-1.0..1.0);
                        g[a * nc + b] = v;
                        g[b * nc + a] = v;
                    }
                }
            }
        }
        let scale: Vec<f64> = (0..p).map(|_| rng.gen_range(0.5..2.0)).collect();
        let terms = basis.body3_terms();
        let coeffs: Vec<Vec<f64>> = (0..s_n)
            .map(|s| {
                let mut row = vec![0.0; p];
                for (f, c) in row.iter_mut().enumerate().take(p2) {
                    *c = rng.gen_range(-0.5..0.5) * scale[f];
                }
                for (q, t) in terms.iter().enumerate() {
                    let mut c = 0.0;
                    for k1 in 0..k {
                        for k2 in 0..k {
                            c += w_star[t.t1 * k + k1] * w_star[t.t2 * k + k2] * gamma[s][t.l][(k1 * n_max + t.n1) * nc + k2 * n_max + t.n2];
                        }
                    }
                    let same = t.t1 == t.t2 && t.n1 == t.n2;
                    row[p2 + q] = if same { c } else { 2.0 * c } * scale[p2 + q];
                }
                row
            })
            .collect();
        let qp = basis.n_independent_columns();
        let mut eye = vec![0.0; qp * qp];
        for i in 0..qp {
            eye[i * qp + i] = 1.0;
        }
        Model {
            coeffs,
            mean: vec![0.0; p],
            scale,
            ref_energy: species.iter().map(|&z| -0.5 * z as f64).collect(),
            sigma_e: 0.0,
            sigma_f: 0.0,
            energy_per_atom: true,
            lambda: 1.0,
            sigma_inv_chol: Some(eye),
            n_samples: 0,
            provenance: prov(),
            representatives: OnceLock::new(),
            variance_support: SupportCache::new(),
            basis,
        }
    }

    fn random_w(rng: &mut StdRng, s_n: usize, k: usize) -> Vec<f64> {
        (0..s_n * k).map(|_| normal(rng)).collect()
    }

    /// `n` teacher samples with species drawn by `pick`, optional Gaussian noise (total energy,
    /// force component).
    fn samples(rng: &mut StdRng, teacher: &Model, n: usize, pick: &dyn Fn(&mut StdRng) -> Vec<u8>, noise: Option<(f64, f64)>) -> Vec<Sample> {
        (0..n)
            .map(|_| {
                let z = pick(rng);
                let pos = cluster(rng, z.len(), 3.5);
                let (mut e, mut f) = energy_and_forces(teacher, &pos, &z).unwrap();
                if let Some((se, sf)) = noise {
                    e += se * normal(rng);
                    for fj in f.iter_mut() {
                        for c in fj.iter_mut() {
                            *c += sf * normal(rng);
                        }
                    }
                }
                Sample::new(pos, z, e, f, SampleTag::Scan, 1.0)
            })
            .collect()
    }

    fn uniform_pick(species: Vec<u8>, atoms: usize) -> impl Fn(&mut StdRng) -> Vec<u8> {
        move |rng: &mut StdRng| (0..atoms).map(|_| species[rng.gen_range(0..species.len())]).collect()
    }

    /// The largest principal-angle sine between col-span(w) (orthonormal columns, S×k) and
    /// col-span(w_star) (S×k_star): ‖(I − P_*) W‖₂.
    fn sin_theta(w: &[f64], k: usize, w_star: &[f64], k_star: usize, s_n: usize) -> f64 {
        let (q, _) = thin_qr(w_star, s_n, k_star);
        let mut e = w.to_vec();
        for c in 0..k {
            for j in 0..k_star {
                let proj: f64 = (0..s_n).map(|t| q[t * k_star + j] * w[t * k + c]).sum();
                for t in 0..s_n {
                    e[t * k + c] -= proj * q[t * k_star + j];
                }
            }
        }
        let mut ete = vec![0.0; k * k];
        for a in 0..k {
            for b in 0..k {
                ete[a * k + b] = (0..s_n).map(|t| e[t * k + a] * e[t * k + b]).sum();
            }
        }
        let (vals, _) = symmetric_eigen(&ete, k);
        vals.iter().cloned().fold(0.0, f64::max).max(0.0).sqrt()
    }

    fn orthonormality_error(w: &[f64], s_n: usize, k: usize) -> f64 {
        let mut worst: f64 = 0.0;
        for a in 0..k {
            for b in 0..k {
                let d: f64 = (0..s_n).map(|t| w[t * k + a] * w[t * k + b]).sum();
                worst = worst.max((d - if a == b { 1.0 } else { 0.0 }).abs());
            }
        }
        worst
    }

    fn sign_rule_holds(w: &[f64], s_n: usize, k: usize) -> bool {
        (0..k).all(|c| {
            let mut best = 0;
            for t in 1..s_n {
                if w[t * k + c].abs() > w[best * k + c].abs() {
                    best = t;
                }
            }
            w[best * k + c] > 0.0
        })
    }

    /// The standardised coefficient θ of every column class (body-3 classes are single columns).
    fn theta(model: &Model) -> Vec<f64> {
        let p = model.basis.n_features();
        model.basis.column_classes().iter().map(|c| model.coeffs[c[0] / p][c[0] % p]).collect()
    }

    fn apply(r: &[f64], th: &[f64]) -> Vec<f64> {
        r.chunks(th.len()).map(|row| row.iter().zip(th).map(|(a, b)| a * b).sum()).collect()
    }

    fn setup(species: &[u8]) -> MixSetup {
        MixSetup::new(species, &species.iter().map(|&z| -0.5 * z as f64).collect::<Vec<_>>(), R_CUT)
    }

    #[test]
    fn low_order_column_counts() {
        // review I4's reduced basis (LowOrder::REVIEW_I4 doc): Q′ = 4·S(S+1)/2 + S·(10S² + 4S)/2
        for (s_n, q) in [(5usize, 735usize), (8, 2_832), (10, 5_420)] {
            let species: Vec<u8> = (1..=s_n as u8).collect();
            let b = LowOrder::REVIEW_I4.basis(&species, R_CUT);
            assert_eq!(b.n_body3(), (10 * s_n * s_n + 4 * s_n) / 2);
            assert_eq!(b.n_independent_columns(), q, "S = {s_n}");
            // under degree cap 2 the body-3 block does not depend on n_max ≥ 3 (only body 2 does):
            // the review's 795 / 2 976 / 5 640 are this body 3 with an n_max 8 body 2
            let mut b8 = b.clone();
            b8.n_max = 8;
            assert_eq!(b8.n_body3(), b.n_body3());
            assert_eq!(b8.n_independent_columns(), q + 4 * s_n * (s_n + 1) / 2);
        }
    }

    /// The algebra alone, from an exactly planted tensor: the learnt span is span(W*) to the
    /// rounding of the S×S eigensolve, the map R reproduces W at the mean, and W is orthonormal
    /// with the sign rule.
    #[test]
    fn a_planted_mix_is_recovered_from_its_coefficients() {
        let mut rng = StdRng::seed_from_u64(11);
        let species = [1u8, 3, 6, 8];
        let (s_n, k) = (4, 2);
        let w_star = random_w(&mut rng, s_n, k);
        let teacher = planted_teacher(&mut rng, &species, &w_star, k);
        let (mix, r) = mix_and_map(&teacher, k, true, Map::Exact).unwrap();
        assert_eq!(mix.k, k);
        let sv = &mix.singular_values;
        let kappa_a = mix.weights.iter().cloned().fold(0.0, f64::max) / mix.weights.iter().cloned().fold(f64::INFINITY, f64::min);
        let sin = sin_theta(&mix.w, k, &w_star, k, s_n);
        // Davis–Kahan: sin Θ(Ũ_K) ≤ ‖δG‖/λ_K with the Gram formed compensated (ε per entry) and
        // Jacobi's backward error O(ε‖G‖), taken as ≤ 64·S·ε·λ₁; the map D_a⁻¹ to W's span
        // multiplies a principal-angle sine by at most κ(D_a) (module doc, step 7)
        let bound = 64.0 * s_n as f64 * EPS * (sv[0] / sv[k - 1]).powi(2) * kappa_a;
        eprintln!("planted algebra: sin Θ = {sin:.3e}, bound {bound:.3e}, σ = {sv:?}, ρ = {:?}", mix.truncation_residual);
        assert!(sin <= bound, "sin Θ {sin:e} > {bound:e}");
        // the planted tensor has exactly K channels: σ_{K+1..S} are rounding and ρ_t ≈ 0
        assert!(sv[k] <= 1e-6 * sv[0], "σ_K+1 = {:e}", sv[k]);
        assert!(mix.truncation_residual.iter().all(|&r| r <= 1e-12), "{:?}", mix.truncation_residual);
        // vec W = R θ at the mean (module doc, "Σ_W")
        let w_lin = apply(&r, &theta(&teacher));
        let err = w_lin.iter().zip(&mix.w).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        eprintln!("planted algebra: max |Rm − W| = {err:.3e}");
        assert!(err <= 1e-12, "R m differs from W by {err:e}");
        assert!(orthonormality_error(&mix.w, s_n, k) <= 10.0 * s_n as f64 * EPS);
        assert!(sign_rule_holds(&mix.w, s_n, k));
        // Σ_low = I here, so Σ_W = R Rᵀ
        let rows = s_n * k;
        let qp = r.len() / rows;
        for a in 0..rows {
            for b in 0..rows {
                let rr: f64 = (0..qp).map(|i| r[a * qp + i] * r[b * qp + i]).sum();
                assert!((rr - mix.sigma_w[a * rows + b]).abs() <= 1e-12 * (1.0 + rr.abs()));
            }
        }
    }

    /// Off the span, R is the exact derivative of the whole pipeline (Gram, eigenvectors, QR,
    /// signs; module doc, "Its first-order error, and the form implemented"): central differences
    /// in θ agree, on a tensor of three channels learnt at K = 2 (σ₃ ≠ 0, where the review's frozen
    /// form differs at first order).
    #[test]
    fn sigma_w_map_is_the_out_of_span_derivative() {
        let mut rng = StdRng::seed_from_u64(12);
        let species = [1u8, 3, 6, 8];
        let (s_n, k) = (4, 2);
        let w_star = random_w(&mut rng, s_n, 3);
        let teacher = planted_teacher(&mut rng, &species, &w_star, 3);
        let (mix, r) = mix_and_map(&teacher, k, true, Map::Exact).unwrap();
        let (_, r_frozen) = mix_and_map(&teacher, k, true, Map::Frozen).unwrap();
        eprintln!("FD teacher: σ = {:?}", mix.singular_values);
        let p = teacher.basis.n_features();
        let p2 = teacher.basis.n_body2();
        let classes = teacher.basis.column_classes();
        for trial in 0..3 {
            let delta: Vec<f64> = classes.iter().map(|c| if c[0] % p >= p2 { normal(&mut rng) } else { 0.0 }).collect();
            let h = 1e-6;
            let shifted = |sign: f64| {
                let mut m = teacher.clone();
                for (i, c) in classes.iter().enumerate() {
                    m.coeffs[c[0] / p][c[0] % p] += sign * h * delta[i];
                }
                mix_from_model(&m, k, true).unwrap().w
            };
            let (wp, wm) = (shifted(1.0), shifted(-1.0));
            let fd: Vec<f64> = wp.iter().zip(&wm).map(|(a, b)| (a - b) / (2.0 * h)).collect();
            let lin = apply(&r, &delta);
            // project both off span(W)
            let off = |v: &[f64]| -> Vec<f64> {
                let mut e = v.to_vec();
                for c in 0..k {
                    for j in 0..k {
                        let proj: f64 = (0..s_n).map(|t| mix.w[t * k + j] * v[t * k + c]).sum();
                        for t in 0..s_n {
                            e[t * k + c] -= proj * mix.w[t * k + j];
                        }
                    }
                }
                e
            };
            let (fd_off, lin_off, frozen_off) = (off(&fd), off(&lin), off(&apply(&r_frozen, &delta)));
            let norm = lin_off.iter().map(|x| x * x).sum::<f64>().sqrt();
            let dist = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
            let diff = dist(&fd_off, &lin_off);
            let diff_frozen = dist(&fd_off, &frozen_off);
            let whole = dist(&fd, &lin) / norm;
            eprintln!(
                "trial {trial}: off-span |FD − Rδ| / |Rδ_⊥| = {:.3e} (frozen form {:.3e}); whole-vector difference {whole:.3e} (the within-span gauge)",
                diff / norm,
                diff_frozen / norm
            );
            // central difference: O(h²) truncation (h = 1e-6 against O(1) curvature over the gap)
            // plus ε/h ≈ 2e-10 rounding; 1e-6 is far above both and far below a first-order error
            assert!(diff <= 1e-6 * norm, "out-of-span derivative mismatch {:e}", diff / norm);
            assert!(diff_frozen > 1e-3 * norm, "the frozen form matches too: σ₃ is not exercised");
        }
    }

    /// Through the fit: noise-free samples of a planted K-channel teacher. The span error obeys
    /// Wedin's sin Θ theorem with the fit's own coefficient error (a bound computed from the truth,
    /// not a tuned tolerance), and that error is small enough to recover the span.
    #[test]
    fn a_planted_mix_is_recovered() {
        let mut rng = StdRng::seed_from_u64(13);
        let species = [1u8, 6, 8];
        let (s_n, k) = (3, 2);
        let w_star = random_w(&mut rng, s_n, k);
        let teacher = planted_teacher(&mut rng, &species, &w_star, k);
        let set = samples(&mut rng, &teacher, 400, &uniform_pick(species.to_vec(), 5), None);
        let refs: Vec<&Sample> = set.iter().collect();
        let st = setup(&species);
        let mix = learn_mix(&refs, &st, k).unwrap();
        assert_eq!(mix.k, k);
        // the fitted model itself, for the bound (the same path as learn_mix: bit for bit)
        let basis = st.low_order.basis(&species, R_CUT);
        let mut acc = Accumulator::new(&basis);
        acc.add_many(&refs, &basis, &st.ref_energy, &st.fit).unwrap();
        let (model, report) = acc.solve(&st.fit, &low_order_provenance(), &st.ref_energy).unwrap();
        let again = mix_from_model(&model, k, true).unwrap();
        assert_eq!(again.w, mix.w);
        let a = &mix.weights;
        let (m_fit, m_true) = (Unfolding::of(&model, a), Unfolding::of(&teacher, a));
        let e_f = m_fit.m.iter().zip(&m_true.m).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt();
        let m_norm = m_true.m.iter().map(|x| x * x).sum::<f64>().sqrt();
        let kappa_a = a.iter().cloned().fold(0.0, f64::max) / a.iter().cloned().fold(f64::INFINITY, f64::min);
        // Wedin (σ_{K+1} of the true M̃ is 0): sin Θ(Ũ_fit, Ũ_true) ≤ ‖E‖₂/σ_K(M̃_fit) ≤ ‖E‖_F/σ_K;
        // ×κ(D_a) into W's span; plus the eigensolve's own rounding (the previous test's bound)
        let sv = &mix.singular_values;
        let bound = kappa_a * (e_f / sv[k - 1] + 64.0 * s_n as f64 * EPS * (sv[0] / sv[k - 1]).powi(2));
        let sin = sin_theta(&mix.w, k, &w_star, k, s_n);
        eprintln!(
            "planted fit: Q′ {} λ {:.3e} rmse E {:.3e} F {:.3e}; ‖E‖_F/‖M̃‖ = {:.3e}; σ = {sv:?}; sin Θ = {sin:.3e}, Wedin bound {bound:.3e}; ρ = {:?}",
            report.q_prime,
            report.lambda,
            report.rmse_energy_per_atom,
            report.rmse_force,
            e_f / m_norm,
            mix.truncation_residual
        );
        assert!(sin <= bound, "sin Θ {sin:e} above Wedin's bound {bound:e}");
        // recovery: on noise-free data the fit determines the body-3 coefficients to its
        // floor-ridge level; a span error of 1e-4 is far below what the channel model could notice
        // (its column-space change is O(sin Θ))
        assert!(sin <= 1e-4, "planted span not recovered: sin Θ {sin:e}");
    }

    #[test]
    fn w_has_orthonormal_columns() {
        let mut rng = StdRng::seed_from_u64(14);
        for (species, k_true) in [(vec![1u8, 6, 8], 3usize), (vec![1u8, 3, 6, 8], 4), (vec![1u8, 3, 6, 8, 9], 3)] {
            let s_n = species.len();
            let w_star = random_w(&mut rng, s_n, k_true);
            let teacher = planted_teacher(&mut rng, &species, &w_star, k_true);
            for k in 1..s_n {
                for weighting in [true, false] {
                    let mix = mix_from_model(&teacher, k, weighting).unwrap();
                    assert_eq!(mix.k, k.min(mix.rank));
                    let err = orthonormality_error(&mix.w, s_n, mix.k);
                    // modified Gram–Schmidt run twice: orthogonal to O(ε) whatever the conditioning
                    assert!(err <= 10.0 * s_n as f64 * EPS, "S {s_n} K {k}: |WᵀW − I| = {err:e}");
                    assert!(sign_rule_holds(&mix.w, s_n, mix.k));
                    assert_eq!(mix.sigma_w.len(), (s_n * mix.k).pow(2));
                    assert!(mix.w.iter().chain(&mix.sigma_w).all(|x| x.is_finite()));
                    // Σ_W is a covariance: symmetric with a non-negative diagonal
                    let rows = s_n * mix.k;
                    for a in 0..rows {
                        assert!(mix.sigma_w[a * rows + a] >= 0.0);
                        for b in 0..rows {
                            assert_eq!(mix.sigma_w[a * rows + b], mix.sigma_w[b * rows + a]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn learn_mix_caps_k_at_species_count() {
        let mut rng = StdRng::seed_from_u64(15);
        for species in [vec![1u8, 8], vec![1u8, 6, 8]] {
            let s_n = species.len();
            let w_star = random_w(&mut rng, s_n, s_n);
            let teacher = planted_teacher(&mut rng, &species, &w_star, s_n);
            let set = samples(&mut rng, &teacher, 120, &uniform_pick(species.clone(), 4), Some((1e-3, 1e-3)));
            let refs: Vec<&Sample> = set.iter().collect();
            for k in [s_n, s_n + 1, 7] {
                let mix = learn_mix(&refs, &setup(&species), k).unwrap();
                assert_eq!(mix.rank, s_n, "a generic fit is full rank");
                assert_eq!((mix.k, mix.k_requested), (s_n, k));
                for t in 0..s_n {
                    for c in 0..s_n {
                        assert_eq!(mix.w[t * s_n + c], if t == c { 1.0 } else { 0.0 });
                    }
                }
                assert_eq!(mix.sigma_w.len(), s_n.pow(4));
                assert!(mix.sigma_w.iter().all(|&x| x == 0.0));
                assert!(mix.truncation_residual.iter().all(|&x| x == 0.0));
                assert_eq!(mix.gap(), None);
                assert!(mix.low_order.is_some());
            }
        }
    }

    #[test]
    fn learn_mix_is_deterministic() {
        let mut rng = StdRng::seed_from_u64(16);
        let species = [1u8, 6, 8];
        let w_star = random_w(&mut rng, 3, 3);
        let teacher = planted_teacher(&mut rng, &species, &w_star, 3);
        let set = samples(&mut rng, &teacher, 150, &uniform_pick(species.to_vec(), 5), Some((1e-3, 1e-3)));
        let refs: Vec<&Sample> = set.iter().collect();
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let first = learn_mix(&refs, &setup(&species), 2).unwrap();
        let second = learn_mix(&refs, &setup(&species), 2).unwrap();
        assert_eq!(first.k, 2);
        assert_eq!(bits(&first.w), bits(&second.w));
        assert_eq!(bits(&first.sigma_w), bits(&second.sigma_w));
        assert_eq!(bits(&first.singular_values), bits(&second.singular_values));
        assert_eq!(bits(&first.truncation_residual), bits(&second.truncation_residual));
        assert!(first.gap().unwrap() > 1.0);
    }

    /// Review C1.4: a species the data barely see has its fibre's posterior near the prior and a
    /// small feature scale, so its Σ_W block is wider than a well-sampled species'.
    #[test]
    fn rare_species_have_wider_sigma_w() {
        let mut rng = StdRng::seed_from_u64(17);
        let species = [1u8, 6, 8];
        let (s_n, k) = (3, 2);
        let w_star = random_w(&mut rng, s_n, 3);
        let teacher = planted_teacher(&mut rng, &species, &w_star, 3);
        let noise = Some((1e-3, 1e-3));
        // H and O in every sample; C (t = 1) in 12 of 312
        let mut set = samples(&mut rng, &teacher, 300, &uniform_pick(vec![1, 8], 5), noise);
        let with_c = |rng: &mut StdRng| {
            let mut z: Vec<u8> = (0..4).map(|_| if rng.gen_bool(0.5) { 1 } else { 8 }).collect();
            z.push(6);
            z
        };
        set.extend(samples(&mut rng, &teacher, 12, &with_c, noise));
        let refs: Vec<&Sample> = set.iter().collect();
        let mix = learn_mix(&refs, &setup(&species), k).unwrap();
        assert_eq!(mix.k, k);
        let rows = s_n * k;
        let block: Vec<f64> = (0..s_n).map(|t| (0..k).map(|c| mix.sigma_w[(t * k + c) * rows + t * k + c]).sum()).collect();
        eprintln!("rare species: tr Σ_W per species (H, C, O) = {block:?}; ρ = {:?}; σ = {:?}", mix.truncation_residual, mix.singular_values);
        assert!(block[1] > block[0] && block[1] > block[2], "the rare species' Σ_W block is not the widest: {block:?}");
    }

    /// The low-order fit of noisy samples of a planted 2-channel teacher over four species.
    fn planted_fit(seed: u64, noise: f64) -> (Model, LearntMix) {
        let mut rng = StdRng::seed_from_u64(seed);
        let species = [1u8, 3, 6, 8];
        let w_star = random_w(&mut rng, 4, 2);
        let teacher = planted_teacher(&mut rng, &species, &w_star, 2);
        let set = samples(&mut rng, &teacher, 300, &uniform_pick(species.to_vec(), 5), Some((noise, noise)));
        let refs: Vec<&Sample> = set.iter().collect();
        let st = setup(&species);
        let basis = st.low_order.basis(&species, R_CUT);
        let mut acc = Accumulator::new(&basis);
        acc.add_many(&refs, &basis, &st.ref_energy, &st.fit).unwrap();
        let (model, _) = acc.solve(&st.fit, &low_order_provenance(), &st.ref_energy).unwrap();
        let (mix, _) = mix_and_map(&model, 2, true, Map::Exact).unwrap();
        (model, mix)
    }

    /// Σ_W against the posterior itself: θ = m + shrink·δθ with δθ ~ N(0, Σ_low) drawn through the
    /// fit's factor, W recomputed by the whole pipeline (Gram, Jacobi, QR, signs), the deviation
    /// divided by shrink. Off the span — the part a refit of c cannot absorb — the sample
    /// covariance is compared with (P⊥ ⊗ I) Σ_W (P⊥ ⊗ I); within the span Σ_W keeps the frozen
    /// gauge part (review C1.8) and the pipeline re-orthonormalises, so they are not compared.
    /// P⊥ = I − W Wᵀ on the species index, applied to every column of vec W.
    fn off_span(w: &[f64], s_n: usize, k: usize, v: &[f64]) -> Vec<f64> {
        let mut e = v.to_vec();
        for c in 0..k {
            for j in 0..k {
                let proj: f64 = (0..s_n).map(|t| w[t * k + j] * v[t * k + c]).sum();
                for t in 0..s_n {
                    e[t * k + c] -= proj * w[t * k + j];
                }
            }
        }
        e
    }

    /// Per species, tr of the off-span block (P⊥ ⊗ I) Σ_W (P⊥ ⊗ I).
    fn off_span_traces(mix: &LearntMix) -> Vec<f64> {
        let (s_n, k) = (mix.species.len(), mix.k);
        let rows = s_n * k;
        let mut half = vec![0.0; rows * rows];
        for b in 0..rows {
            let col: Vec<f64> = (0..rows).map(|a| mix.sigma_w[a * rows + b]).collect();
            for (a, v) in off_span(&mix.w, s_n, k, &col).into_iter().enumerate() {
                half[a * rows + b] = v;
            }
        }
        let mut lin = vec![0.0; rows * rows];
        for a in 0..rows {
            for (b, v) in off_span(&mix.w, s_n, k, &half[a * rows..(a + 1) * rows]).into_iter().enumerate() {
                lin[a * rows + b] = v;
            }
        }
        (0..s_n).map(|t| (0..k).map(|c| lin[(t * k + c) * rows + t * k + c]).sum()).collect()
    }

    /// Σ_W against the posterior itself: θ = m + shrink·δθ with δθ ~ N(0, Σ_low) drawn through the
    /// fit's factor, W recomputed by the whole pipeline (Gram, Jacobi, QR, signs), the deviation
    /// divided by shrink. Off the span — the part a refit of c cannot absorb — the sample
    /// covariance is compared with (P⊥ ⊗ I) Σ_W (P⊥ ⊗ I); within the span Σ_W keeps the frozen
    /// gauge part (review C1.8) and the pipeline re-orthonormalises, so they are not compared.
    /// Returns the sampled off-span traces per species.
    fn monte_carlo_off_span(model: &Model, mix: &LearntMix, draws: usize, shrink: f64, rng: &mut StdRng) -> Vec<f64> {
        use crate::linalg::backward_substitute_transposed;
        let (s_n, k) = (mix.species.len(), mix.k);
        let rows = s_n * k;
        let off = |v: &[f64]| off_span(&mix.w, s_n, k, v);
        let basis = &model.basis;
        let qp = basis.n_independent_columns();
        let l = model.sigma_inv_chol.as_ref().unwrap();
        let p = basis.n_features();
        let classes = basis.column_classes();
        let mut sum = vec![0.0; rows];
        let mut outer = vec![0.0; rows * rows];
        for _ in 0..draws {
            let mut z: Vec<f64> = (0..qp).map(|_| normal(rng)).collect();
            backward_substitute_transposed(l, qp, &mut z); // δθ = L⁻ᵀ z ~ N(0, A⁻¹)
            let mut m = model.clone();
            for (i, c) in classes.iter().enumerate() {
                m.coeffs[c[0] / p][c[0] % p] += z[i] * shrink;
            }
            let w = mix_and_map(&m, k, true, Map::None).unwrap().0.w;
            let d: Vec<f64> = off(&w).iter().map(|x| x / shrink).collect();
            for a in 0..rows {
                sum[a] += d[a];
                for b in 0..rows {
                    outer[a * rows + b] += d[a] * d[b];
                }
            }
        }
        let n = draws as f64;
        let mc: Vec<f64> = (0..rows * rows).map(|ab| (outer[ab] - sum[ab / rows] * sum[ab % rows] / n) / (n - 1.0)).collect();
        (0..s_n).map(|t| (0..k).map(|c| mc[(t * k + c) * rows + t * k + c]).sum()).collect()
    }

    /// The relative std of a sample variance from n Gaussian draws is √(2/(n−1)); a trace over
    /// correlated entries has no more. Five of them is the tolerance.
    fn mc_tolerance(draws: usize) -> f64 {
        5.0 * (2.0 / (draws as f64 - 1.0)).sqrt()
    }

    /// The worst |sampled / linear − 1| over the species and the total.
    fn worst_ratio_error(lin: &[f64], mc: &[f64]) -> f64 {
        let total = (mc.iter().sum::<f64>() / lin.iter().sum::<f64>() - 1.0).abs();
        lin.iter().zip(mc).map(|(a, b)| (b / a - 1.0).abs()).fold(total, f64::max)
    }

    fn sci(v: &[f64]) -> String {
        v.iter().map(|x| format!("{x:.3e}")).collect::<Vec<_>>().join(", ")
    }

    /// The first order is exact: in the linear limit (shrink 1e-4) the sampled off-span
    /// covariance is Σ_W's, on a noisy fit whose dropped singular values are not small — where the
    /// review's frozen form alone is measurably wrong (asserted, so the test has the power to tell
    /// the two apart; module doc, "Validity").
    #[test]
    fn sigma_w_is_the_exact_first_order_covariance_off_the_span() {
        let (model, mix) = planted_fit(19, 1e-2);
        let frozen = mix_and_map(&model, 2, true, Map::Frozen).unwrap().0;
        let mut rng = StdRng::seed_from_u64(20);
        let draws = 2000;
        let mc = monte_carlo_off_span(&model, &mix, draws, 1e-4, &mut rng);
        let (lin, lin_frozen) = (off_span_traces(&mix), off_span_traces(&frozen));
        let tol = mc_tolerance(draws);
        let (err, err_frozen) = (worst_ratio_error(&lin, &mc), worst_ratio_error(&lin_frozen, &mc));
        eprintln!(
            "first order (noise 1e-2, η {:.3e}, σ [{}]): off-span traces sampled [{}], exact [{}] (worst error {err:.3}), frozen [{}] (worst error {err_frozen:.3}); tolerance {tol:.3}",
            mix.nonlinearity,
            sci(&mix.singular_values),
            sci(&mc),
            sci(&lin),
            sci(&lin_frozen)
        );
        assert!(err <= tol, "the exact first order misses the sampled covariance by {err}");
        assert!(err_frozen > tol, "the frozen form is indistinguishable here ({err_frozen}): the test has no power");
        // the same fit at its full posterior width, measured (module doc, "Validity"): not a gate
        let mc_full = monte_carlo_off_span(&model, &mix, 400, 1.0, &mut rng);
        eprintln!("full width at η {:.3e}: sampled / linear off-span trace {:.3}", mix.nonlinearity, mc_full.iter().sum::<f64>() / lin.iter().sum::<f64>());
    }

    /// Where η ≪ 1 the first order holds at the posterior's full width.
    #[test]
    fn sigma_w_holds_at_full_width_when_eta_is_small() {
        let (model, mix) = planted_fit(21, 3e-4);
        assert!(mix.nonlinearity < 0.1, "the premise: η = {} is small", mix.nonlinearity);
        let mut rng = StdRng::seed_from_u64(22);
        let draws = 800;
        let mc = monte_carlo_off_span(&model, &mix, draws, 1.0, &mut rng);
        let lin = off_span_traces(&mix);
        let err = worst_ratio_error(&lin, &mc);
        eprintln!("full width (noise 3e-4, η {:.3e}): off-span traces sampled [{}], linear [{}], worst error {err:.3}", mix.nonlinearity, sci(&mc), sci(&lin));
        assert!(err <= mc_tolerance(draws), "full width: sampled / linear off by {err}");
    }

    /// Wall time of `learn_mix` at five species (Q′ 735), split into the low-order fit and the mix,
    /// for the day-1 gate's timing (not a gate: `#[ignore]`d;
    /// `cargo test -p surrogate --release --lib -- channels::tests::learn_mix_timing_probe --ignored --nocapture`).
    #[test]
    #[ignore]
    fn learn_mix_timing_probe() {
        let mut rng = StdRng::seed_from_u64(23);
        let species = [1u8, 3, 8, 16, 17];
        let w_star = random_w(&mut rng, 5, 3);
        let teacher = planted_teacher(&mut rng, &species, &w_star, 3);
        let set = samples(&mut rng, &teacher, 1000, &uniform_pick(species.to_vec(), 6), Some((1e-3, 1e-3)));
        let refs: Vec<&Sample> = set.iter().collect();
        let st = setup(&species);
        let basis = st.low_order.basis(&species, R_CUT);
        let t0 = std::time::Instant::now();
        let mut acc = Accumulator::new(&basis);
        acc.add_many(&refs, &basis, &st.ref_energy, &st.fit).unwrap();
        let t1 = std::time::Instant::now();
        let (model, report) = acc.solve(&st.fit, &low_order_provenance(), &st.ref_energy).unwrap();
        let t2 = std::time::Instant::now();
        let mix = mix_from_model(&model, 4, true).unwrap();
        let t3 = std::time::Instant::now();
        eprintln!(
            "S 5, Q′ {}, 1000 samples of 6 atoms: accumulate {:.2} s, search + solve {:.2} s ({} rounds), mix + Σ_W {:.3} s; σ [{}], η {:.3e}",
            report.q_prime,
            (t1 - t0).as_secs_f64(),
            (t2 - t1).as_secs_f64(),
            report.hyper_iterations,
            (t3 - t2).as_secs_f64(),
            sci(&mix.singular_values),
            mix.nonlinearity
        );
    }

    #[test]
    fn learn_mix_refuses_bad_input() {
        let mut rng = StdRng::seed_from_u64(18);
        let species = [1u8, 8];
        let w_star = random_w(&mut rng, 2, 2);
        let teacher = planted_teacher(&mut rng, &species, &w_star, 2);
        let set = samples(&mut rng, &teacher, 5, &uniform_pick(species.to_vec(), 3), None);
        let refs: Vec<&Sample> = set.iter().collect();
        assert_eq!(learn_mix(&refs, &setup(&species), 0), Err(MixError::ZeroK));
        let mut bad = setup(&species);
        bad.species = vec![8, 1];
        assert_eq!(learn_mix(&refs, &bad, 1), Err(MixError::Species));
        assert_eq!(learn_mix(&[], &setup(&species), 1), Err(MixError::Fit(FitError::NoSamples)));
        let mut no_post = teacher.clone();
        no_post.sigma_inv_chol = None;
        assert_eq!(mix_from_model(&no_post, 1, true), Err(MixError::NoPosterior));
        // a sample with a species outside the list is the describe error, not a guess
        assert!(matches!(learn_mix(&refs, &setup(&[1]), 1), Err(MixError::Fit(FitError::Describe(_)))));
        let mut wrong_mask = setup(&species);
        wrong_mask.eligible = Some(vec![true]);
        assert_eq!(learn_mix(&refs, &wrong_mask, 1), Err(MixError::Species));
    }

    /// Spec "Physics review 2" item 10 (review T5): the represented status is a flag decided by
    /// the rule — N_W and ρ_t ≤ 0.01, a NaN failing — and stored, never read from W: a species
    /// failing N_W keeps an exact-zero row even with a non-zero fibre of its own, a species failing
    /// ρ is unrepresented with its row zeroed, the others are represented.
    #[test]
    fn representation_status_is_a_flag_not_a_value() {
        // the rule itself: ρ = 0.5 fails, ρ < 0.01 passes, NaN fails, N_W unmet fails
        assert!(!is_represented(true, 0.5));
        assert!(is_represented(true, 0.004));
        assert!(is_represented(true, RHO_MAX));
        assert!(!is_represented(true, f64::NAN));
        assert!(!is_represented(false, 0.0));
        // a planted tensor: H, C, O on two strong channels, S on a third, weak one of its own; K = 2
        let mut rng = StdRng::seed_from_u64(41);
        let species = [1u8, 6, 8, 16];
        let w_star = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.6, 0.8, 0.0, 0.0, 0.0, 0.05];
        let teacher = planted_teacher(&mut rng, &species, &w_star, 3);
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        // every species eligible: S's fibre lies along the dropped direction, so ρ_S ≫ 0.01
        let all = mix_from_model_with(&teacher, 2, false, Some(&[true; 4])).unwrap();
        assert_eq!(all.k, 2);
        for t in 0..4 {
            let rho = all.truncation_residual[t];
            assert_eq!(all.represented[t], rho <= RHO_MAX, "species {} ρ {rho}", species[t]);
        }
        assert!(all.represented[..3].iter().all(|&r| r), "H, C, O represented: ρ {:?}", all.truncation_residual);
        assert!(!all.represented[3] && all.truncation_residual[3] > RHO_MAX, "S fails (b): ρ {}", all.truncation_residual[3]);
        assert_eq!(bits(&all.w[3 * 2..4 * 2]), vec![0u64; 2], "its row is exactly +0.0");
        assert!(orthonormality_error(&all.w, 4, 2) < 1e-12, "the represented rows carry W's orthonormal columns");
        assert!(sign_rule_holds(&all.w, 4, 2));
        assert_eq!(all.n_w_met, vec![true; 4]);
        // S failing N_W instead: its Gram row is zero before the SVD (NaN ρ, a flag, row exactly 0)
        // although its own fibre is non-zero in the fit's coefficients
        let un = Unfolding::of(&teacher, &weights(&teacher, false));
        assert!(un.row(3).iter().any(|&x| x != 0.0), "the premise: S has a non-zero body-3 fibre");
        let masked = mix_from_model_with(&teacher, 2, false, Some(&[true, true, true, false])).unwrap();
        assert!(masked.truncation_residual[3].is_nan());
        assert_eq!(masked.represented, vec![true, true, true, false]);
        assert_eq!(masked.n_w_met, vec![true, true, true, false]);
        assert_eq!(bits(&masked.w[3 * 2..4 * 2]), vec![0u64; 2]);
        // a full-rank tensor: K covering every represented species gives the identity on them,
        // S's row zero
        let w_dense = random_w(&mut rng, 4, 4);
        let dense = planted_teacher(&mut rng, &species, &w_dense, 4);
        let id = mix_from_model_with(&dense, 3, false, Some(&[true, true, true, false])).unwrap();
        assert_eq!(id.k, 3);
        assert_eq!(id.w, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(id.truncation_residual[..3], [0.0; 3]);
        assert_eq!(id.represented, vec![true, true, true, false]);
        assert!(id.sigma_w.iter().all(|&x| x == 0.0));
        // with every species eligible and K ≥ S: exactly the raw mix's W = I, the species path
        let full = mix_from_model_with(&dense, 4, false, Some(&[true; 4])).unwrap();
        let raw = mix_from_model(&dense, 4, false).unwrap();
        assert_eq!((full.k, bits(&full.w)), (raw.k, bits(&raw.w)));
        assert_eq!(full.represented, vec![true; 4]);
        // and below S with every species represented: the raw mix's W bit for bit
        let (low_raw, low_rep) = (mix_from_model(&teacher, 3, false).unwrap(), mix_from_model_with(&teacher, 3, false, Some(&[true; 4])).unwrap());
        if low_rep.represented.iter().all(|&r| r) {
            assert_eq!(bits(&low_raw.w), bits(&low_rep.w));
        }
        // no species eligible: refused (nothing to learn from)
        assert_eq!(mix_from_model_with(&teacher, 2, false, Some(&[false; 4])), Err(MixError::NoManyBodySignal));
    }
}
