//! The recipe's universal constants — spec §4.9, "the complete rule-1 ledger" — as ONE `const`
//! table. Every *recipe* constant the generator (`qm::radial`) and its consumers (the filter,
//! `qm::ionic`, the dimer gate) use — every number that can move a generated table — is a field of
//! [`LEDGER`]; no other file under `crates/qm/src/radial/` declares a numeric constant
//! (`gate_thresholds_are_constants`, Task 4). The method-convergence literals that cannot move a
//! converged result beyond its own tolerance (bisection width, WKB end, overflow rescale, DIIS
//! regularisation, …) stay inline and are listed in the module doc of the file that uses them
//! (`numerov`, `atom`; review S2, 2026-10-15). Nothing is indexed by Z. A change to
//! any value is a recipe change: every element is regenerated and re-gated, and `recipe_hash`
//! (Task 4: FNV-1a 64 of this table's `Debug` rendering, which includes `version` and `xc`) changes,
//! invalidating cached tables and surrogate provenances.
//!
//! `numerics` holds the solver knobs the spec names in §4.1/§4.6 (radial SCF tolerance, DIIS) and the
//! plan's G2 zero-count mesh (Spec question 5): they do not change a converged result beyond their
//! tolerance, but they are hashed too.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Xc { LsdaPw92 }

/// Nonlinear core correction mode: `Lfc` (Louie–Froyen–Cohen, spec §4.5); `Off` is the switch the
/// precision milestone's QMC variant needs (spec §3, "Future requirement").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NlccMode { Lfc, Off }

/// Local-channel candidates in the universal preference order (spec §4.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalCandidate { HighestChannel, Polynomial }

/// Gate thresholds (spec §4.6, §4.8). Energies in Hartree.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gates {
    pub t1_eig: f64, pub t2_norm: f64, pub t3_delta_e: f64, pub t4_eig: f64,
    pub t5_de_equiv: f64, pub t5_window: f64, pub t5_mesh: f64, pub t5_rd_factor: f64,
    pub t6_hardness: f64, pub t6_dq: f64,
    pub g2_eig: f64, pub g2_window: f64,
    pub h_req_tol: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Numerics {
    pub scf_tol: f64, pub scf_max_iter: usize, pub diis_history: usize, pub mixing: f64, pub g2_mesh: f64,
    /// Consecutive β-halvings of the linear-mixing fallback before an SCF step that unbinds an
    /// occupied shell is reported as genuinely unbound (review M1: β_min = mixing/2^6 ≈ 5e-3).
    pub backtrack_max: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ledger {
    pub version: &'static str,
    /// Radial log grid r_i = r0·e^{iδ} up to r_max (Bohr) — §4.1.
    pub r0: f64, pub r_max: f64, pub delta: f64,
    /// Ruling S9's multiplicity cap (`sim::sampling::allowed_multiplicities`; a `sim` test pins them equal).
    pub multiplicity_max: u32,
    /// Semicore promotion window and occupation-scan window (Ha) — §4.2, §4.1.
    pub delta_core: f64, pub delta_scan: f64,
    /// l_max = max occupied valence l + this — §4.2.
    pub l_max_above_valence: u8,
    /// The r_c ladder, softest first — §4.4.
    pub f_ladder: [f64; 4],
    pub local_order: [LocalCandidate; 2],
    pub nlcc: NlccMode,
    /// r_nlc where ρ_c = nlcc_crossover · ρ_v (LFC's factor 1) — §4.5.
    pub nlcc_crossover: f64,
    /// TM polynomial degree (7 coefficients) — §4.3.
    pub tm_degree: u8,
    /// r_cmp: where |V_ion + Z_v/r| < this (Ha) for good — §4.3.
    pub v_ion_tail: f64,
    pub gates: Gates,
    /// q_c = this · π/h; mask at gamma_r · r_c — §5.1.
    pub q_c_over_nyquist: f64, pub gamma_r: f64,
    /// σ_LR = this · h — §5.1.
    pub sigma_lr_over_h: f64,
    /// V_tail is sampled out to where it is below this (Ha) — §5.1.
    pub v_tail_floor: f64,
    /// The h ladder for h_req (Bohr), coarsest first — §4.8 T8.
    pub h_ladder: [f64; 5],
    /// r_core encloses all but this fraction of the core electrons — §5.6.
    pub eps_core: f64,
    /// A shell's r_peak (spec §4.4: r_c = f·r_peak) is its outermost local maximum of |u| above this
    /// fraction of max|u| — recipe-affecting, hence here (review S2).
    pub r_peak_floor: f64,
    /// The dimer gate on f applies for Z ≤ dimer_z_max: R_e within dimer_re (Bohr), ω within dimer_omega_rel — §4.4, §6.4.
    pub dimer_z_max: u8, pub dimer_re: f64, pub dimer_omega_rel: f64,
    /// The dimer gate's bond-length bracket, in units of the element's outermost valence r_peak
    /// (Task 11; Spec question 9): a bound AE dimer must have its minimum inside it.
    pub dimer_bracket: [f64; 2],
    pub xc: Xc,
    pub numerics: Numerics,
}

pub const LEDGER: Ledger = Ledger {
    version: "m7-tm-kb-nlcc-1",
    r0: 1e-5, r_max: 60.0, delta: 0.008,
    multiplicity_max: 5,
    delta_core: 2.0, delta_scan: 0.1,
    l_max_above_valence: 1,
    f_ladder: [1.6, 1.4, 1.2, 1.0],
    local_order: [LocalCandidate::HighestChannel, LocalCandidate::Polynomial],
    nlcc: NlccMode::Lfc, nlcc_crossover: 1.0,
    tm_degree: 12,
    v_ion_tail: 1e-8,
    gates: Gates {
        t1_eig: 1e-3, t2_norm: 1e-10, t3_delta_e: 5e-3, t4_eig: 5e-3,
        t5_de_equiv: 20e-3, t5_window: 1.0, t5_mesh: 0.01, t5_rd_factor: 1.2,
        t6_hardness: 10e-3, t6_dq: 0.25,
        g2_eig: 1e-3, g2_window: 0.5,
        h_req_tol: 1e-3,
    },
    q_c_over_nyquist: 0.7, gamma_r: 1.5,
    sigma_lr_over_h: 2.0,
    v_tail_floor: 1e-6,
    h_ladder: [0.35, 0.30, 0.25, 0.20, 0.15],
    eps_core: 1e-3,
    r_peak_floor: 1e-3,
    dimer_z_max: 2, dimer_re: 0.02, dimer_omega_rel: 0.05, dimer_bracket: [1.0, 4.0],
    xc: Xc::LsdaPw92,
    numerics: Numerics { scf_tol: 1e-10, scf_max_iter: 400, diis_history: 6, mixing: 0.3, g2_mesh: 1e-3, backtrack_max: 6 },
};
