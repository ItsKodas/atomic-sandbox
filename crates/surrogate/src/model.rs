//! The fitted linear model and its evaluation: total energy and analytic forces, and the
//! Bayesian predictive variance when the model carries the posterior factor from the fit.
//!
//! # Energy
//!
//! The coefficients live in standardised feature space (`fit` solves there so the ridge penalty
//! is scale-free): with per-feature `mean_k` and `scale_k`, the energy of atom i of species s_i is
//!
//!   E_i = ref_energy[s_i] + Σ_k c_{s_i,k} (B_{i,k} − mean_k)/scale_k,
//!
//! and a feature with `scale_k == 0` (constant over the training set) is divided by 1 instead.
//! Expanding gives the form the hot path evaluates,
//!
//!   E_i = ref_energy[s_i] + offset_{s_i} + Σ_k c̃_{s_i,k} B_{i,k},
//!   c̃_{s,k} = c_{s,k}/scale'_k,   offset_s = −Σ_k c̃_{s,k} mean_k,
//!
//! so standardisation costs one S×P pass per call and nothing per pair. Note the isolated-atom
//! limit (no neighbour inside r_cut, B = 0) is ref_energy[s] + offset_s: it equals the reference
//! energy only when offset_s = 0, which `fit` guarantees by never centring (mean = 0, scale-only
//! standardisation, no intercept — ruling S5); "two atoms beyond r_cut → Σ ref" then holds
//! exactly.
//!
//! # Forces, pair-symmetric
//!
//! F_j = −∂E/∂r_j = −Σ_i Σ_k c̃_{s_i,k} ∂B_{i,k}/∂r_j. For body-2 features every B_{i,k} is a sum
//! over pairs, so E = Σ_i (ref + offset) + Σ_{pairs i<j} e_ij(r_ij) with
//!
//!   e_ij(r) = Σ_n [c̃_{s_i,(s_j,n)} + c̃_{s_j,(s_i,n)}] R_n(r) f_c(r)
//!
//! (the pair contributes to i's descriptor on j's species channel and to j's on i's). Its
//! derivative a_ij = de_ij/dr uses the same bracket with g_n = d[R_n f_c]/dr, and with
//! u_ij = (r_j − r_i)/r the chain rule gives F_j −= a_ij u_ij, F_i += a_ij u_ij. Each pair is
//! visited once and the same vector is added to one atom and subtracted from the other, so
//! ΣF = 0 holds to rounding by construction (translation invariance of the descriptors), and
//! Σ r×F = Σ_pairs a_ij (r_i − r_j) × u_ij = −Σ a_ij r_ij u_ij × u_ij = 0 (rotation invariance:
//! every pair force is central). No per-atom gradient list is built; the per-pair work is one
//! `radial` call plus 2·n_max fused multiply-adds, and the (s_i, s_j) coefficient blocks are
//! contiguous slices because features are ordered (neighbour species, n).
//!
//! The energy is accumulated with Neumaier (compensated) summation (CLAUDE.md rule 4: energy
//! reductions are compensated); forces are plain sums of ≤ N terms per atom.
//!
//! # Body-3 forces by the adjoint of the atomic basis
//!
//! With the body-3 block on, atom i's energy also holds E_i^{(3)} = Σ_q c̃_{s_i,q} Σ_m
//! A_{c₁(q),lm}(i) A_{c₂(q),lm}(i) (`descriptor` docs). Instead of forming the P₃ feature
//! gradients, the hot path contracts the coefficients first,
//!
//!   W_{c,lm}(i) = ∂E_i/∂A_{c,lm} = Σ_{q: c₁ = c} c̃_q A_{c₂,lm} + Σ_{q: c₂ = c} c̃_q A_{c₁,lm}
//!
//! (both sums contribute when c₁ = c₂, giving 2c̃A), and then, since A_{(t,n),lm}(i) =
//! Σ_{j∈t} φ_{n,lm}(r_j − r_i),
//!
//!   ∂E_i/∂r_j = Σ_n Σ_{lm} W_{(s_j,n),lm}(i) ∇φ_{n,lm}(r_j − r_i)   (j ≠ i),   ∂E_i/∂r_i = −Σ_j ∂E_i/∂r_j,
//!
//! so F_j −= ∂E_i/∂r_j and F_i += the same vector: ΣF = 0 by construction again. Per centre
//! the cost is one `radial` + one `solid_harmonics` per neighbour (their ∇φ kept in a scratch
//! slot), O(P₃·(2l+1)) for the energy and W, and O(neighbours · n_max · (l_max+1)²) for the
//! forces — O(N · neighbours · features) overall with no per-pair allocation (the scratch is
//! sized once per call). The per-neighbour body-3 force is not central, so Σ r×F vanishes by
//! the rotation invariance of E_i (cancellation between neighbours, to rounding) rather than
//! pair by pair. The body-2 block keeps its pair sweep above.
//!
//! # Coincident atoms (policy)
//!
//! Two atoms at the same position have r_ij = 0 and no direction. The descriptor keeps the value
//! R_n(0) f_c(0) = 1 (so the pair counts as a neighbour and the energy stays finite:
//! E = Σ ref + Σ_n [c̃_i + c̃_j]_n) and the hot path adds exactly zero force to both atoms
//! (`pair_geometry` returns u = 0). The energy has a |r| cusp there; 0 is a valid subgradient,
//! it keeps the pair antisymmetry (ΣF = 0 exactly) and, unlike a NaN or an arbitrary direction,
//! it cannot inject momentum. Dynamics never reaches r = 0 on a repulsive surface; the policy is
//! for edits that drop one atom onto another, where the next step's forces are finite.
//!
//! # Predictive variance (`predict` with `with_variance`)
//!
//! With the posterior covariance Σ of the standardised coefficients on the Q′ independent
//! columns, held as the Cholesky factor L_A of Σ⁻¹ = G̃_E/σ_e² + G̃_F/σ_f² + λI (`fit` docs), a
//! linear functional φ̃ᵀc of the coefficients has epistemic variance φ̃ᵀΣφ̃ = ‖L_A⁻¹φ̃‖². The
//! outputs:
//!
//! * `energy_std`: the TOTAL energy, φ̃ = Σ_i φ̃_i with φ̃_i the standardised descriptor of atom i
//!   in its species block, plus the aleatoric floor of the energy rows: (N·σ_e)² when the fit
//!   used energies per atom (the row was E/N with noise σ_e), σ_e² otherwise.
//! * `force_std[j]`: √(Σ_a var_{j,a}) with var_{j,a} = σ_f² + g̃_{j,a}ᵀΣg̃_{j,a}, where g̃_{j,a}
//!   is the force-row feature vector −Σ_i ∂B̃_i/∂r_{j,a} — the expected squared norm of the
//!   force error on atom j, the quantity that compares to |F_j| in the fallback threshold; a
//!   per-component std lies between force_std/√3 and force_std.
//! * `force_std_epistemic[j]`: √(Σ_a g̃_{j,a}ᵀΣg̃_{j,a}), the same sum without the aleatoric
//!   term. The split is exact by construction — the two terms are accumulated separately and
//!   force_std[j]² = 3σ_f² + force_std_epistemic[j]² (to rounding of one addition): the first is
//!   the label noise of the three force rows of atom j (what a fresh teacher answer scatters
//!   about the model however many samples are added), the second the coefficient uncertainty
//!   (what new samples shrink). The provider's force rule compares the second (M6 Task 11; the
//!   energy rule already removes its floor).
//!
//! ## On the composition's support (spec `2026-11-20-variance-support-design.md`, review round 1)
//!
//! A scene's query vectors are non-zero only on columns its species can reach. For a species set
//! C let S(C) be every reduced column whose centre species and channel species (body-2: the
//! neighbour's; body-3: both channels'; body-4: all three) lie in C. A feature with a channel
//! species the scene lacks is a sum with no term (or a product with one), exactly +0.0, and so is
//! its gradient; so every query vector of a scene of composition C vanishes off S(C), and with
//! A = L_A L_Aᵀ the posterior precision the block-inverse rule gives
//!
//!   φ̃ᵀA⁻¹φ̃ = φ̃_Sᵀ (A⁻¹)_SS φ̃_S = ‖F⁻¹ J φ̃_S‖²,
//!
//! F the trailing Cholesky block of A ordered (N, reversed S) (`linalg::support_factor`, built in
//! double-double: in f64 the cancellation that block carries cost up to 3.5e-11 relative on fitted
//! factors). The forms are the full path's own substitution through a reordered exact factor, at
//! |S|²/2 per vector instead of Q′²/2. They are in the full path's accuracy class, measured
//! against a double-double evaluation through L on the user's BEST fit (every form of 1 000
//! H₂/H₃ geometries: support max 3.2e-12, median 4.3e-13; full max 3.3e-12, median 4.2e-13), so
//! the two paths can differ by about their sum (1.9e-12 on the stds there); the unit-test
//! fixtures pin them within 1e-12 on the stds (4.4e-13 measured). Every build is guarded by the
//! 1e-10 self-check (`SupportCache`). On the H/O/P collection basis (Q′ 1 902): |S({H})| = 52, a
//! pair 468, all three 1 902.
//!
//! The model caches one factor per composition (`Model::variance_support`, a [`SupportCache`]).
//! A composition with |S(C)| > Q′/2 takes the full factor (the support gains little there), and so
//! does every call of a model without a posterior (none is asked for) or with a factor the full
//! substitution cannot use (a non-finite entry or a zero pivot: the full path then reports the
//! non-finite stds L gives). S(C) depends on C alone and the build is bitwise deterministic, so a
//! std is a pure function of (L, geometry, species set): call order, threads and clones do not
//! move a bit (`stds_do_not_depend_on_the_call_history`).
//!
//! Cost: one `describe` (body-3/4 terms of absent channel species skipped, bit for bit), then the
//! forms: on the support |S|²/2 multiply-adds per vector (1 + 3N of them), and one build per
//! (model, composition) (double-double, Σ_s (Q′ − s)²/2 + Q′|S|²/2 and O(|S|³): 9.5 ms at |S| = 52,
//! 57 ms at 282, 0.26 s at 950 for Q′ = 1 902); on the full factor 3N + 1 triangular solves of
//! size Q′ (the independent columns, `Basis::n_independent_columns`), O(Q′²) each.

use crate::descriptor::{describe, gaunt, pair_geometry, t4_channels, AtomicBasisScratch, Basis, Body4Term, DescribeError, Gaunt};
use crate::basis::radial;
use crate::linalg::{cholesky_quad_forms, support_factor};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

/// Settings the training data were produced with; a model is only valid for a scene solved with
/// the same ones (the Kohn-Sham surface depends on grid, σ and tolerance). `masses` is a
/// **record, not a gate** (M6 spec §5.3, P2): the isotopes — sorted unique (Z, A) pairs — the
/// samples' scenes carried, kept because a hot burst's trajectory depends on the masses, but the
/// Born-Oppenheimer surface the model learns does not, so `PartialEq` ignores it (refusing an H₂
/// model for D₂ would be physically wrong). Empty when unrecorded (a sidecar before version 3).
#[derive(Clone, Debug)]
pub struct Provenance {
    pub grid_n: usize,
    pub h: f64,
    pub sigma: f64,
    pub tol: f64,
    pub charge: i32,
    pub multiplicity: Option<u32>,
    pub teacher: String,
    pub masses: Vec<(u8, u16)>,
    /// The ionic surface the samples were solved on (M7, spec §5.5; `providers::pseudo::provenance_tag`):
    /// `"none"` for the all-electron surface (every model before M7), else `"<policy>/<recipe hash>"`.
    /// **Compared**: the surfaces differ by tens of Ha per pseudised atom and in bond angles, so a model
    /// taught on one is never consulted on another. The tag follows the physics, not the menu — H₂ under
    /// `CoresOnly` is the all-electron surface and carries `"none"`.
    pub ionic: String,
}

/// Everything but `masses` (struct doc) — and so does every derived `PartialEq` that embeds a
/// `Provenance` (`Model`, `SampleStore`): two models differing only in the recorded mass set
/// compare equal.
impl PartialEq for Provenance {
    fn eq(&self, o: &Provenance) -> bool {
        self.grid_n == o.grid_n
            && self.h == o.h
            && self.sigma == o.sigma
            && self.tol == o.tol
            && self.charge == o.charge
            && self.multiplicity == o.multiplicity
            && self.teacher == o.teacher
            && self.ionic == o.ionic
    }
}

impl Provenance {
    /// Everything a provenance comparison looks at, for messages (a refused load, a scene the
    /// model was not taught at): grid, h, σ, charge, multiplicity, teacher and its tolerance.
    /// The recorded mass set is NOT part of it — a refusal must not name a field that was not
    /// compared (P2); [`describe_recorded`](Self::describe_recorded) adds it for info lines.
    pub fn describe(&self) -> String {
        // grid_n 0 = any box: a shared collection's provenance (`providers::surrogate`, "shared")
        let grid = if self.grid_n == 0 { "any box,".to_string() } else { format!("{}³", self.grid_n) };
        format!(
            "{} h {} σ {}, charge {}, multiplicity {}, teacher '{}' at tol {:e}, ionic {}",
            grid,
            self.h,
            self.sigma,
            self.charge,
            self.multiplicity.map_or("lowest".to_string(), |m| m.to_string()),
            self.teacher,
            self.tol,
            self.ionic
        )
    }

    /// The recorded mass set as labels, "¹H ²H", or "unrecorded" (a sidecar before version 3).
    pub fn masses_label(&self) -> String {
        if self.masses.is_empty() {
            "unrecorded".to_string()
        } else {
            self.masses.iter().map(|&(z, a)| units::isotopes::isotope_label(z, a)).collect::<Vec<_>>().join(" ")
        }
    }

    /// [`describe`](Self::describe) plus the record: "…, masses ¹H ²H (recorded, not compared)".
    pub fn describe_recorded(&self) -> String {
        format!("{}, masses {} (recorded, not compared)", self.describe(), self.masses_label())
    }

    /// Merges another record's mass set into this one (sorted, unique) — the store's record
    /// grows with the scenes its samples came from; equality is unaffected (struct doc).
    pub fn record_masses(&mut self, masses: &[(u8, u16)]) {
        for &m in masses {
            if let Err(i) = self.masses.binary_search(&m) {
                self.masses.insert(i, m);
            }
        }
    }

    /// The solver surface these settings put a sample on ([`SurfaceKey`]): everything a sample's
    /// energy depends on — h, σ, the SCF tolerance, the teacher, the ionic tag and the functional
    /// (`"lsda"`, the only one this solver has) — and NOT the box (`grid_n`), the charge or the
    /// multiplicity (spec §4.2: a collection's admission rules handle those, `crate::collection`).
    pub fn surface_key(&self) -> SurfaceKey {
        SurfaceKey {
            functional: FUNCTIONAL_LSDA.to_string(),
            h: self.h,
            sigma: self.sigma,
            tol: self.tol,
            teacher: self.teacher.clone(),
            ionic: self.ionic.clone(),
        }
    }
}

/// The exchange-correlation functional of every surface this solver has produced (LSDA; PBE
/// arrives with its own milestone and its own key).
pub const FUNCTIONAL_LSDA: &str = "lsda";

/// One solver surface (spec `2026-10-20-m8-shared-model-and-sandbox-design.md` §4.2): the solve
/// settings a sample's energy depends on, so samples with equal keys can share one fit whatever
/// box they were solved in. Compared exactly (f64 `==`, like [`Provenance`]); the box size
/// (`grid_n`), the charge and the multiplicity are not part of it — a shared collection refuses
/// charged and spin-forced samples instead, and the box rule keeps box-affected ones out.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceKey {
    /// The exchange-correlation functional ([`FUNCTIONAL_LSDA`]).
    pub functional: String,
    /// Grid spacing, Bohr.
    pub h: f64,
    /// Nuclear smearing width σ, Bohr.
    pub sigma: f64,
    /// SCF tolerance of the teacher's solves.
    pub tol: f64,
    /// The teacher's kind (`ForceProvider::name`, e.g. "Kohn-Sham (GPU)").
    pub teacher: String,
    /// The ionic surface tag ([`Provenance::ionic`]).
    pub ionic: String,
}

/// FNV-1a, 64 bit: a short, platform-independent fingerprint of a string for [`SurfaceKey::slug`]
/// (not a security hash: a slug collision is caught by `key.txt`, `crate::collection`).
pub(crate) fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Lower-case ASCII letters, digits, `.`, `+` and `-` are kept; every other run of characters
/// becomes one `-` (and none leads or trails).
fn slug_part(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        let c = c.to_ascii_lowercase();
        if c.is_ascii_lowercase() || c.is_ascii_digit() || c == '.' || c == '+' {
            out.push(c);
        } else if !out.is_empty() && !out.ends_with('-') {
            out.push('-');
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() {
        out.push('x');
    }
    out
}

impl SurfaceKey {
    /// A filesystem-safe folder name for this key, the same on every run and platform: the
    /// functional, h, σ and the tolerance (each f64 in Rust's shortest round-trip form, so two
    /// different values never share a spelling), the teacher, and the ionic tag — `none` as it is,
    /// any other tag as its policy word plus a 12-hex-digit FNV-1a hash of the whole tag (the full
    /// tags run to ~60 characters with `/`). A teacher name over 32 characters is shortened the same
    /// way. Only `[a-z0-9.+_-]`; e.g. `lsda_h0.15_s0.15_tol1e-6_kohn-sham-gpu_none`.
    pub fn slug(&self) -> String {
        let hashed = |s: &str, head: &str| format!("{}-{:012x}", slug_part(head), fnv1a64(s.as_bytes()) & 0xffff_ffff_ffff);
        let teacher = slug_part(&self.teacher);
        let teacher = if teacher.len() > 32 { hashed(&self.teacher, &teacher[..20]) } else { teacher };
        let ionic = if self.ionic == "none" { "none".to_string() } else { hashed(&self.ionic, self.ionic.split('/').next().unwrap_or("")) };
        format!(
            "{}_h{}_s{}_tol{}_{}_{}",
            slug_part(&self.functional),
            slug_part(&format!("{:?}", self.h)),
            slug_part(&format!("{:?}", self.sigma)),
            slug_part(&format!("{:?}", self.tol)),
            teacher,
            ionic
        )
    }

    /// The first field where `other` differs from `self`, worded for a refusal:
    /// "h 0.25 (the collection's 0.15)"; `None` when the keys are equal.
    pub fn difference(&self, other: &SurfaceKey) -> Option<String> {
        if self.functional != other.functional {
            return Some(format!("functional {} (the collection's {})", other.functional, self.functional));
        }
        if self.h != other.h {
            return Some(format!("h {:?} (the collection's {:?})", other.h, self.h));
        }
        if self.sigma != other.sigma {
            return Some(format!("σ {:?} (the collection's {:?})", other.sigma, self.sigma));
        }
        if self.tol != other.tol {
            return Some(format!("tol {:?} (the collection's {:?})", other.tol, self.tol));
        }
        if self.teacher != other.teacher {
            return Some(format!("teacher '{}' (the collection's '{}')", other.teacher, self.teacher));
        }
        if self.ionic != other.ionic {
            return Some(format!("ionic {} (the collection's {})", other.ionic, self.ionic));
        }
        None
    }

    /// One line for messages and the README: "lsda, h 0.15 Bohr, σ 0.15 Bohr, tol 1e-6, teacher
    /// 'Kohn-Sham (GPU)', ionic none".
    pub fn describe(&self) -> String {
        format!(
            "{}, h {:?} Bohr, σ {:?} Bohr, tol {:?}, teacher '{}', ionic {}",
            self.functional, self.h, self.sigma, self.tol, self.teacher, self.ionic
        )
    }

    /// The `key.txt` form: one `name value` line per field, floats in shortest round-trip form
    /// (read back bit-exactly by [`from_text`](Self::from_text)); `#` lines are comments.
    pub fn to_text(&self) -> String {
        format!(
            "# Atomic Sandbox collection: the solver surface every sample here was solved on\nfunctional {}\nh {:?}\nsigma {:?}\ntol {:?}\nteacher {}\nionic {}\n",
            self.functional, self.h, self.sigma, self.tol, self.teacher, self.ionic
        )
    }

    /// Reads [`to_text`](Self::to_text)'s form; every field is required, once.
    pub fn from_text(text: &str) -> Result<SurfaceKey, String> {
        let mut fields: [Option<&str>; 6] = [None; 6];
        const NAMES: [&str; 6] = ["functional", "h", "sigma", "tol", "teacher", "ionic"];
        for line in text.lines() {
            let line = line.trim_end_matches('\r');
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }
            let (name, value) = line.split_once(' ').ok_or_else(|| format!("line '{line}' is not 'name value'"))?;
            let k = NAMES.iter().position(|&n| n == name).ok_or_else(|| format!("unknown field '{name}'"))?;
            if fields[k].replace(value).is_some() {
                return Err(format!("field '{name}' appears twice"));
            }
        }
        let get = |k: usize| fields[k].ok_or_else(|| format!("field '{}' is missing", NAMES[k]));
        let num = |k: usize| -> Result<f64, String> { get(k)?.parse::<f64>().map_err(|e| format!("field '{}': {e}", NAMES[k])) };
        Ok(SurfaceKey { functional: get(0)?.to_string(), h: num(1)?, sigma: num(2)?, tol: num(3)?, teacher: get(4)?.to_string(), ionic: get(5)?.to_string() })
    }
}

/// A fitted linear model. All fields are public: `fit` (Task 3) builds it, the sidecar (Task 5)
/// serialises it.
#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub basis: Basis,
    /// `coeffs[centre species][feature]`, in STANDARDISED feature space.
    pub coeffs: Vec<Vec<f64>>,
    /// Per-feature mean of the standardisation (shared by all centre species). **Always zero in
    /// a fitted model** (ruling S5: scale-only standardisation, so the binding energy vanishes
    /// beyond r_cut by construction; `fit` writes zeros and its tests assert it). The field is
    /// kept for sidecar format stability (format v2 stores it) and because `predict` still folds
    /// a hand-built model's nonzero mean into the per-species offsets correctly
    /// ([`Model::effective_coefficients`]) — the cost is one multiply-add per feature per
    /// species per call, not per atom.
    pub mean: Vec<f64>,
    /// Per-feature scale of the standardisation; 0 marks a constant feature (divided by 1).
    pub scale: Vec<f64>,
    /// Isolated-atom energy per species index, Hartree.
    pub ref_energy: Vec<f64>,
    /// Noise level of an energy row: per atom when `energy_per_atom`, else per sample (Hartree).
    pub sigma_e: f64,
    /// Noise level of a force component, Hartree/Bohr.
    pub sigma_f: f64,
    /// Whether the energy rows of the fit were E/N (decides the energy floor in `predict`).
    pub energy_per_atom: bool,
    /// Prior precision λ of the standardised coefficients (`fit` docs).
    pub lambda: f64,
    /// Cholesky factor (lower, row-major Q′×Q′) of the posterior precision
    /// Σ⁻¹ = G̃_E/σ_e² + G̃_F/σ_f² + λI over the Q′ independent columns of
    /// [`Basis::column_classes`] (representative order), for the predictive variance. `None`
    /// for a model without a fit; `predict` then reports no uncertainty.
    pub sigma_inv_chol: Option<Vec<f64>>,
    pub n_samples: usize,
    pub provenance: Provenance,
    /// Lazily cached representative columns of `basis.column_classes()` (one full column index
    /// per class); derived data, `OnceLock::new()` in a constructor. Replace the cell if `basis`
    /// is changed in place.
    pub representatives: OnceLock<Vec<usize>>,
    /// Lazily cached factors of the predictive variance, one per species composition (module
    /// doc, "Predictive variance"); derived data of `sigma_inv_chol` and `representatives`,
    /// `SupportCache::new()` in a constructor. Replace the cell if `sigma_inv_chol` or `basis` is
    /// changed in place.
    pub variance_support: SupportCache,
}

/// Energy (Hartree, total including Σ ref_energy) and forces (Hartree/Bohr), with the
/// predictive standard deviations when the model carries a covariance factor.
#[derive(Clone, Debug, PartialEq)]
pub struct Prediction {
    pub energy: f64,
    pub forces: Vec<[f64; 3]>,
    pub energy_std: Option<f64>,
    /// Per atom, Hartree/Bohr.
    pub force_std: Option<Vec<f64>>,
    /// Per atom, Hartree/Bohr: the epistemic part of `force_std` (module doc, "Predictive
    /// variance"): force_std² = 3σ_f² + force_std_epistemic².
    pub force_std_epistemic: Option<Vec<f64>>,
}

impl Model {
    /// Total energy and forces; `with_variance` additionally asks for the predictive standard
    /// deviations (module doc), which are `None` while `sigma_inv_chol` is `None`.
    ///
    /// # Panics
    /// If `sigma_inv_chol` is `Some` with a length other than Q′² (`Basis::n_independent_columns`).
    pub fn predict(
        &self,
        pos: &[[f64; 3]],
        z: &[u8],
        with_variance: bool,
    ) -> Result<Prediction, DescribeError> {
        let (energy, forces) = energy_and_forces(self, pos, z)?;
        let (energy_std, force_std, force_std_epistemic) = match (&self.sigma_inv_chol, with_variance) {
            (Some(l), true) => {
                let (e, f, f_epi) = self.predictive_std(l, pos, z)?;
                (Some(e), Some(f), Some(f_epi))
            }
            _ => (None, None, None),
        };
        Ok(Prediction {
            energy,
            forces,
            energy_std,
            force_std,
            force_std_epistemic,
        })
    }

    /// Standardised feature index map: 1/scale'_k per feature.
    fn inverse_scale(&self) -> Vec<f64> {
        self.scale
            .iter()
            .map(|&s| if s == 0.0 { 1.0 } else { 1.0 / s })
            .collect()
    }

    /// Representative full-column index of every class of `basis.column_classes()`, cached.
    pub fn representatives(&self) -> &[usize] {
        self.representatives
            .get_or_init(|| self.basis.column_classes().into_iter().map(|c| c[0]).collect())
    }

    /// Total-energy std, per-atom force std and its epistemic part (module doc).
    fn predictive_std(
        &self,
        l: &[f64],
        pos: &[[f64; 3]],
        z: &[u8],
    ) -> Result<(f64, Vec<f64>, Vec<f64>), DescribeError> {
        self.predictive_std_with(l, pos, z, VariancePath::Auto)
    }

    /// [`Model::predictive_std`] with the quadratic forms on the composition's support when it is
    /// small enough (`Auto`, module doc) or always on the full factor (`Full`, the tests' oracle).
    fn predictive_std_with(
        &self,
        l: &[f64],
        pos: &[[f64; 3]],
        z: &[u8],
        path: VariancePath,
    ) -> Result<(f64, Vec<f64>, Vec<f64>), DescribeError> {
        let qr = self.representatives().len();
        assert_eq!(l.len(), qr * qr, "Model.sigma_inv_chol: Q′×Q′ over Basis::column_classes()");
        let n = pos.len();
        let s = self.basis.species_indices(z)?;
        let desc = describe(&self.basis, pos, z)?;
        let q = QueryInputs { species: &s, desc: &desc, inv_scale: self.inverse_scale() };
        let quad = match path {
            VariancePath::Auto if !forced_full() => self.variance_support.quad_forms(self, l, &q),
            _ => full_quad_forms(l, qr, &self.full_query_vectors(&q)),
        };
        let floor_e = if self.energy_per_atom {
            (n as f64 * self.sigma_e).powi(2)
        } else {
            self.sigma_e.powi(2)
        };
        let energy_std = (floor_e + quad[0]).sqrt();
        let mut force_std = Vec::with_capacity(n);
        let mut force_std_epistemic = Vec::with_capacity(n);
        for j in 0..n {
            // the epistemic sum alone, then the aleatoric 3σ_f² added once: the split is exact
            let mut epistemic = 0.0;
            for a in 0..3 {
                epistemic += quad[1 + 3 * j + a];
            }
            force_std.push((3.0 * self.sigma_f * self.sigma_f + epistemic).sqrt());
            force_std_epistemic.push(epistemic.sqrt());
        }
        Ok((energy_std, force_std, force_std_epistemic))
    }

    /// The column map of the full path: every class representative to its reduced index
    /// (`representatives`), every other column `IGNORED`.
    fn full_column_map(&self) -> Vec<u32> {
        let mut map = vec![IGNORED; self.basis.n_species() * self.basis.n_features()];
        for (r, &c) in self.representatives().iter().enumerate() {
            map[c] = r as u32;
        }
        map
    }

    /// The 1 + 3N query vectors on the Q′ independent columns (the full path's layout).
    fn full_query_vectors(&self, q: &QueryInputs) -> Vec<Vec<f64>> {
        accumulate_query_vectors(&self.basis, q, &self.full_column_map(), self.representatives().len())
            .expect("the full map has no column outside it")
    }

    /// The query vectors of a geometry in the full path's layout (the tests' entry point).
    #[cfg(test)]
    fn query_vectors(&self, pos: &[[f64; 3]], z: &[u8]) -> Result<Vec<Vec<f64>>, DescribeError> {
        let s = self.basis.species_indices(z)?;
        let desc = describe(&self.basis, pos, z)?;
        Ok(self.full_query_vectors(&QueryInputs { species: &s, desc: &desc, inv_scale: self.inverse_scale() }))
    }

    /// Effective coefficients c̃_{s,k} = c_{s,k}/scale'_k and per-species offsets
    /// −Σ_k c̃_{s,k} mean_k (see the module doc), so E_i = ref + offset + c̃·B.
    pub fn effective_coefficients(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        self.check_shapes();
        let p = self.basis.n_features();
        let mut c_tilde = Vec::with_capacity(self.coeffs.len());
        let mut offset = Vec::with_capacity(self.coeffs.len());
        for row in &self.coeffs {
            let mut ct = Vec::with_capacity(p);
            let mut off = Neumaier::default();
            for (k, &c_k) in row.iter().enumerate() {
                let scale = if self.scale[k] == 0.0 { 1.0 } else { self.scale[k] };
                let c = c_k / scale;
                ct.push(c);
                off.add(-c * self.mean[k]);
            }
            c_tilde.push(ct);
            offset.push(off.sum());
        }
        (c_tilde, offset)
    }

    /// Shape consistency of the public fields against the basis.
    ///
    /// # Panics
    /// With a message naming the field, if any length disagrees with the basis.
    pub fn check_shapes(&self) {
        let s = self.basis.n_species();
        let p = self.basis.n_features();
        assert_eq!(self.coeffs.len(), s, "Model.coeffs: one row per species");
        for (i, row) in self.coeffs.iter().enumerate() {
            assert_eq!(row.len(), p, "Model.coeffs[{i}]: one entry per feature");
        }
        assert_eq!(self.mean.len(), p, "Model.mean: one entry per feature");
        assert_eq!(self.scale.len(), p, "Model.scale: one entry per feature");
        assert_eq!(self.ref_energy.len(), s, "Model.ref_energy: one entry per species");
    }
}

/// Which factor the predictive variance's quadratic forms use (module doc, "Predictive
/// variance").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VariancePath {
    /// The composition's support factor when |S(C)| ≤ Q′/2, else the full L.
    Auto,
    /// Always the full factor L (the tests' oracle; `Auto` falls back to the same code).
    #[cfg_attr(not(test), allow(dead_code))]
    Full,
}

#[cfg(any(test, feature = "test-hooks"))]
static FORCE_FULL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Test hook (the `test-hooks` feature, for the provider's tests): while on, every variance call
/// of every model in the process takes the full path (the support path "forced off"). Never used
/// by the `surrogate` crate's own tests: it is process-wide.
#[cfg(any(test, feature = "test-hooks"))]
pub fn force_full_variance_path(on: bool) {
    FORCE_FULL.store(on, std::sync::atomic::Ordering::SeqCst);
}

#[inline]
fn forced_full() -> bool {
    #[cfg(any(test, feature = "test-hooks"))]
    {
        FORCE_FULL.load(std::sync::atomic::Ordering::SeqCst)
    }
    #[cfg(not(any(test, feature = "test-hooks")))]
    {
        false
    }
}

/// A column a query-vector layout neither holds nor needs: a tied duplicate (not a class
/// representative), which the posterior never sees.
const IGNORED: u32 = u32::MAX;
/// A representative the layout does not hold: its entries must be exactly zero.
const OUTSIDE: u32 = u32::MAX - 1;

/// What the query vectors are accumulated from: the species index of every atom, its
/// descriptor and gradients, and 1/scale′ per feature.
struct QueryInputs<'a> {
    species: &'a [usize],
    desc: &'a [crate::descriptor::AtomDescriptor],
    inv_scale: Vec<f64>,
}

/// The 1 + 3N query vectors (the energy's φ̃, then g̃_{j,a} for j = 0..N, a = x, y, z) laid out by
/// `map` (full column → slot < `width`, `IGNORED` or `OUTSIDE`). Each slot receives the same
/// additions in the same order as a full-width row restricted afterwards, so its value is that
/// row's bit for bit (`query_vectors_are_the_restricted_full_rows`). `None` if a non-zero
/// contribution lands on an `OUTSIDE` column (impossible for the composition's support, module
/// doc; the caller then takes the full path).
fn accumulate_query_vectors(basis: &Basis, q: &QueryInputs, map: &[u32], width: usize) -> Option<Vec<Vec<f64>>> {
    let p = basis.n_features();
    let n = q.desc.len();
    let mut out = vec![vec![0.0; width]; 1 + 3 * n];
    for (i, d) in q.desc.iter().enumerate() {
        let block = &map[q.species[i] * p..(q.species[i] + 1) * p];
        // Energy: φ̃ = Σ_i φ̃_i.
        for k in 0..p {
            match block[k] {
                IGNORED => {}
                OUTSIDE => {
                    if d.b[k] * q.inv_scale[k] != 0.0 {
                        return None;
                    }
                }
                r => out[0][r as usize] += d.b[k] * q.inv_scale[k],
            }
        }
        // Forces: g̃_{j,a} = −Σ_i ∂B̃_i/∂r_{j,a}, gathered from every centre's gradient list.
        for (j, g) in &d.grad {
            for k in 0..p {
                let r = block[k];
                if r == IGNORED {
                    continue;
                }
                let gk = g[k];
                let sc = q.inv_scale[k];
                if r == OUTSIDE {
                    if gk[0] * sc != 0.0 || gk[1] * sc != 0.0 || gk[2] * sc != 0.0 {
                        return None;
                    }
                    continue;
                }
                let r = r as usize;
                out[1 + 3 * j][r] -= gk[0] * sc;
                out[2 + 3 * j][r] -= gk[1] * sc;
                out[3 + 3 * j][r] -= gk[2] * sc;
            }
        }
    }
    Some(out)
}

/// ‖L⁻¹v‖² of every query vector: every quadratic form of the step in one batched substitution
/// (`cholesky_quad_forms`: bit for bit the one-at-a-time values, L read once per eight rows;
/// 2026-11-17).
fn full_quad_forms(l: &[f64], qr: usize, vs: &[Vec<f64>]) -> Vec<f64> {
    cholesky_quad_forms(l, qr, &vs.iter().map(|v| &v[..]).collect::<Vec<_>>())
}

/// The composition's support S(C) (module doc): every reduced column whose centre species and
/// channel species all lie in `mask` (bit s = species index s), ascending.
fn composition_support(basis: &Basis, rep: &[usize], mask: u32) -> Vec<usize> {
    let p = basis.n_features();
    let p2 = basis.n_body2();
    let terms3 = basis.body3_terms();
    let terms4 = basis.body4_terms();
    let p3 = terms3.len();
    let bit = |t: usize| 1u32 << t;
    rep.iter()
        .enumerate()
        .filter(|&(_, &c)| {
            let (centre, k) = (c / p, c % p);
            let channels = if k < p2 {
                bit(k / basis.n_max)
            } else if k < p2 + p3 {
                let t = &terms3[k - p2];
                bit(t.t1) | bit(t.t2)
            } else {
                terms4[k - p2 - p3].t.iter().fold(0, |m, &t| m | bit(t))
            };
            (bit(centre) | channels) & !mask == 0
        })
        .map(|(r, _)| r)
        .collect()
}

/// The species bitmask of a scene (bit s = species index s); `None` past 32 species (the full
/// path then).
fn mask_of(species: &[usize]) -> Option<u32> {
    species.iter().try_fold(0u32, |m, &s| (s < 32).then(|| m | 1 << s))
}

/// A cheap fingerprint of L (review I2, 2026-11-20): its size, its whole last row (contiguous)
/// and 32 evenly spaced diagonal entries, by bits. A rescaling, refit or refactorisation of L
/// changes it; an in-place edit confined to other entries would not, so the field rule (replace
/// the cell if `sigma_inv_chol` changes in place) still stands. O(Q′) reads.
fn fingerprint(l: &[f64], qr: usize) -> u64 {
    let mix = |h: u64, x: u64| (h ^ x).wrapping_mul(0x0000_0100_0000_01b3);
    let mut h = mix(0xcbf2_9ce4_8422_2325, qr as u64);
    if qr == 0 {
        return h;
    }
    for &v in &l[(qr - 1) * qr..] {
        h = mix(h, v.to_bits());
    }
    for i in (0..qr).step_by(qr.div_ceil(32)) {
        h = mix(h, l[i * qr + i].to_bits());
    }
    h
}

/// Whether the full substitution through L gives finite numbers for finite vectors: every entry
/// it reads (the lower triangle) finite and no zero on the diagonal. O(Q′²) reads, once per model
/// (`SupportState::usable`), and only when a support factor is about to be built (review I1).
fn factor_usable(l: &[f64], qr: usize) -> bool {
    (0..qr).all(|i| l[i * qr + i] != 0.0 && l[i * qr..i * qr + i + 1].iter().all(|v| v.is_finite()))
}

/// The self-check's bound (review I3): a new factor whose own forms differ from the full path's
/// by more than this, relative, is not used.
const SELF_CHECK_BOUND: f64 = 1e-10;

/// The predictive variance's factors per species composition (module doc, "Predictive
/// variance"): one entry per species set C (a bitmask of species indices, at most 2^S − 1),
/// holding either the factor F on S(C) (`linalg::support_factor`) or the decision to take the
/// full path (|S(C)| > Q′/2, an unusable L, a failed build or a failed self-check). Derived data
/// of the model's `sigma_inv_chol`, `basis` and `representatives`, built lazily by `predict`.
///
/// * **Determinism.** S(C) is a function of the basis and C, `support_factor` is bitwise
///   deterministic whatever the thread count, and the self-check runs on a canonical cluster of
///   the composition (not on the triggering call), so whether C takes the support or the full
///   path, and every std, is a pure function of (L, geometry, species set): the same across
///   runs, threads, clones and call orders. A call whose query vectors are all zero (no
///   neighbour inside r_cut) settles nothing and answers +0.0 forms, as the full path does.
/// * **Threads.** The model is shared with the auditor's thread: entries are immutable `Arc`s
///   behind a `Mutex` held only to read or insert one, never while a factor is built or L is
///   checked (no lock is held across a call into rayon, where a stolen job could re-enter
///   `predict`). Two threads that miss the same C build the same bits; the first insert stands.
///   A poisoned lock is recovered: entries are only ever inserted whole.
/// * **Staleness.** Every call compares a fingerprint of L (`fingerprint`, O(Q′)) with the one
///   the entries were built for and empties the cache on a mismatch. Replace the cell anyway if
///   `sigma_inv_chol` or `basis` is changed in place.
/// * **Self-check.** A new factor is used only if every form of the composition's canonical
///   cluster (`canonical_cluster`: one atom of each species and one more, all within r_cut, so
///   the forms are non-zero) agrees with the full path's within `SELF_CHECK_BOUND`
///   (NaN-propagating), and at least one checked form is non-zero; else the composition takes the
///   full path for good. The largest difference seen is kept (`self_check_max`); a fallback is
///   reported once on stderr, and so is a call that found a non-zero outside S(C) (counted per
///   factor, `outside_support_calls`; impossible by the module doc's argument).
/// * **Equality and clone.** Not part of `PartialEq` (derived data); `Clone` copies the entries
///   (the `Arc`s): a clone has the same L, and an entry either builds is the same bits.
pub struct SupportCache {
    state: Mutex<SupportState>,
}

#[derive(Clone)]
enum Slot {
    /// This composition takes the full path.
    Full,
    /// This composition's support factor.
    Support(Arc<SupportFactor>),
}

#[derive(Clone, Default)]
struct SupportState {
    entries: std::collections::BTreeMap<u32, Slot>,
    /// The fingerprint of the L the entries and `usable` belong to.
    fingerprint: Option<u64>,
    /// `factor_usable(L)`, once per model (review I1).
    usable: Option<bool>,
    builds: u64,
    /// The largest relative difference a self-check has seen.
    self_check_max: f64,
    /// A full-path fallback (unusable L, failed build or self-check) has been reported.
    reported: bool,
    /// The self-check's bound when not `SELF_CHECK_BOUND` (the tests lower it).
    #[cfg(test)]
    bound: Option<f64>,
}

/// S(C) and its factor (struct doc).
struct SupportFactor {
    /// Reduced column indices, ascending.
    support: Vec<usize>,
    /// Full column → slot in the factor's (reversed) order, `IGNORED` or `OUTSIDE`.
    map: Vec<u32>,
    /// The |S| × |S| lower factor F of `linalg::support_factor` (row-major, S reversed).
    factor: Vec<f64>,
    /// Calls that found a non-zero outside S(C) (impossible by the module doc's argument; each
    /// took the full path, and the first is reported on stderr).
    outside: std::sync::atomic::AtomicU64,
}

impl SupportFactor {
    /// ‖F⁻¹ J v_S‖² of vectors laid out by `map` (so already in the reversed order).
    fn quad_forms(&self, vs: &[Vec<f64>]) -> Vec<f64> {
        cholesky_quad_forms(&self.factor, self.support.len(), &vs.iter().map(|v| &v[..]).collect::<Vec<_>>())
    }
}

/// The self-check's query set for composition `mask` (struct doc, "Self-check"): one fixed
/// cluster, a function of the basis and the composition alone — one atom of every species of C
/// in ascending order, then one more of the first (two of it when C has one species), on a
/// fixed spiral 1.4–2.4 Bohr from the origin, so every pair is inside r_cut and every block of
/// S(C) the cluster reaches is non-zero. Returns (pos, z).
fn canonical_cluster(basis: &Basis, mask: u32) -> (Vec<[f64; 3]>, Vec<u8>) {
    let mut z: Vec<u8> = (0..basis.n_species()).filter(|&s| mask & (1 << s) != 0).map(|s| basis.species[s]).collect();
    z.push(z[0]);
    let n = z.len();
    let pos = (0..n)
        .map(|k| {
            if k == 0 {
                return [0.0; 3];
            }
            let r = 1.4 + 1.0 * k as f64 / n as f64;
            let cos_t = 1.0 - 2.0 * (k as f64 - 0.5) / n as f64;
            let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
            let phi = 2.399_963_229_728_653 * k as f64; // the golden angle
            [r * sin_t * phi.cos(), r * sin_t * phi.sin(), r * cos_t]
        })
        .collect();
    (pos, z)
}

/// The largest relative difference between two lists of forms, NaN-propagating (a NaN on either
/// side, or a NaN quotient, gives NaN, which fails every `diff <= bound` test).
fn max_relative_difference(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| if x == y { 0.0 } else { (x - y).abs() / y.abs() })
        .fold(0.0, |m: f64, d| if d.is_nan() || m.is_nan() { f64::NAN } else { m.max(d) })
}

impl SupportCache {
    /// An empty cache (nothing built).
    pub fn new() -> SupportCache {
        SupportCache { state: Mutex::new(SupportState::default()) }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, SupportState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The support S(C) held for the species set of `species` (species indices), `None` if no
    /// support factor is held for it (not built yet, or the full path).
    pub fn support_for(&self, species: &[usize]) -> Option<Vec<usize>> {
        match self.state().entries.get(&mask_of(species)?) {
            Some(Slot::Support(e)) => Some(e.support.clone()),
            _ => None,
        }
    }

    /// How many compositions hold a support factor.
    pub fn support_entries(&self) -> usize {
        self.state().entries.values().filter(|e| matches!(e, Slot::Support(_))).count()
    }

    /// How many compositions are settled (a support factor or the full path).
    pub fn settled(&self) -> usize {
        self.state().entries.len()
    }

    /// How many support factors this cache has built (the tests count them).
    pub fn builds(&self) -> u64 {
        self.state().builds
    }

    /// The largest relative difference between a build's self-check forms on the support and on
    /// the full path seen so far (struct doc, "Self-check"); 0 before the first build, NaN if a
    /// check met a NaN.
    pub fn self_check_max(&self) -> f64 {
        self.state().self_check_max
    }

    /// Calls that found a non-zero query entry outside their composition's S(C), over every held
    /// factor (0 by the module doc's argument).
    pub fn outside_support_calls(&self) -> u64 {
        self.state()
            .entries
            .values()
            .map(|e| match e {
                Slot::Support(f) => f.outside.load(std::sync::atomic::Ordering::Relaxed),
                Slot::Full => 0,
            })
            .sum()
    }

    /// Report a full-path fallback, once per cache (struct doc).
    fn report(&self, why: &str) {
        let mut st = self.state();
        if !st.reported {
            st.reported = true;
            eprintln!("surrogate: the predictive variance takes the full path ({why})");
        }
    }

    /// vᵀA⁻¹v for the call's 1 + 3N query vectors: on the composition's support factor when it
    /// has one, else on the full factor.
    fn quad_forms(&self, model: &Model, l: &[f64], q: &QueryInputs) -> Vec<f64> {
        let qr = model.representatives().len();
        let full = || full_quad_forms(l, qr, &model.full_query_vectors(q));
        // every query vector zero (no atom has a neighbour inside r_cut): ‖L⁻¹0‖² = +0.0 exactly,
        // as on the full path; nothing is built or settled (review round 2, I1)
        if q.desc.iter().all(|d| d.b.iter().all(|&x| x == 0.0) && d.grad.iter().all(|(_, g)| g.iter().all(|v| v.iter().all(|&x| x == 0.0)))) {
            return vec![0.0; 1 + 3 * q.desc.len()];
        }
        let Some(mask) = mask_of(q.species) else {
            return full();
        };
        let fp = fingerprint(l, qr);
        let slot = {
            let mut st = self.state();
            if st.fingerprint != Some(fp) {
                // a new L (struct doc, "Staleness"): nothing held belongs to it
                st.entries.clear();
                st.usable = None;
                st.fingerprint = Some(fp);
            }
            st.entries.get(&mask).cloned()
        };
        let entry = match slot {
            Some(Slot::Full) => return full(),
            Some(Slot::Support(e)) => e,
            None => match self.build(model, l, qr, mask, fp) {
                Some(e) => e,
                None => return full(),
            },
        };
        match accumulate_query_vectors(&model.basis, q, &entry.map, entry.support.len()) {
            Some(vs) => entry.quad_forms(&vs),
            // a non-zero value outside S(C) contradicts the module doc's argument: never seen
            None => {
                debug_assert!(false, "a query vector is non-zero outside the composition's support");
                if entry.outside.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
                    eprintln!("surrogate: a query vector was non-zero outside its composition's support (the call took the full path)");
                }
                full()
            }
        }
    }

    /// Settle composition `mask`: insert its support factor or the full path, and return the
    /// factor if it has one. The outcome is a function of (L, basis, composition) alone — the
    /// self-check runs on `canonical_cluster`, not on the triggering call — so racing builders
    /// insert the same thing. Nothing O(Q′²) runs under the lock.
    fn build(&self, model: &Model, l: &[f64], qr: usize, mask: u32, fp: u64) -> Option<Arc<SupportFactor>> {
        let insert = |slot: Slot| -> Option<Arc<SupportFactor>> {
            let mut st = self.state();
            if st.fingerprint != Some(fp) {
                return None;
            }
            match st.entries.entry(mask).or_insert(slot) {
                Slot::Support(e) => Some(e.clone()),
                Slot::Full => None,
            }
        };
        let support = composition_support(&model.basis, model.representatives(), mask);
        if 2 * support.len() > qr {
            return insert(Slot::Full);
        }
        let known = self.state().usable;
        let usable = known.unwrap_or_else(|| {
            let u = factor_usable(l, qr);
            self.state().usable = Some(u);
            u
        });
        if !usable {
            self.report("the posterior factor has a non-finite entry or a zero pivot");
            return insert(Slot::Full);
        }
        let factor = support_factor(l, qr, &support);
        self.state().builds += 1;
        let Some(factor) = factor else {
            self.report("a support factor's Gram matrix is not numerically positive definite");
            return insert(Slot::Full);
        };
        // the factor's (reversed) slots for the representatives in S, OUTSIDE for the others
        let m = support.len();
        let mut map = model.full_column_map();
        for v in map.iter_mut().filter(|v| **v != IGNORED) {
            *v = OUTSIDE;
        }
        let rep = model.representatives();
        for (t, &r) in support.iter().enumerate() {
            map[rep[r]] = (m - 1 - t) as u32;
        }
        let entry = SupportFactor { support, map, factor, outside: std::sync::atomic::AtomicU64::new(0) };
        // the self-check on the composition's canonical cluster: every form on both paths
        let (pos, z) = canonical_cluster(&model.basis, mask);
        let species = model.basis.species_indices(&z).expect("the basis's own species");
        let desc = describe(&model.basis, &pos, &z).expect("the basis's own species");
        let cq = QueryInputs { species: &species, desc: &desc, inv_scale: model.inverse_scale() };
        let Some(on_s) = accumulate_query_vectors(&model.basis, &cq, &entry.map, m) else {
            debug_assert!(false, "the canonical cluster is non-zero outside its composition's support");
            self.report("the self-check's cluster was non-zero outside its composition's support");
            return insert(Slot::Full);
        };
        let b = full_quad_forms(l, qr, &model.full_query_vectors(&cq));
        let a = entry.quad_forms(&on_s);
        let diff = max_relative_difference(&a, &b);
        let bound = {
            let mut st = self.state();
            st.self_check_max = if diff.is_nan() || st.self_check_max.is_nan() { f64::NAN } else { st.self_check_max.max(diff) };
            #[cfg(test)]
            let bound = st.bound.unwrap_or(SELF_CHECK_BOUND);
            #[cfg(not(test))]
            let bound = SELF_CHECK_BOUND;
            bound
        };
        if !b.iter().any(|&x| x > 0.0) {
            // nothing was checked (a posterior whose forms all vanish on the cluster)
            self.report("a support factor's self-check had no non-zero form");
            return insert(Slot::Full);
        }
        if !(diff <= bound) {
            self.report(&format!("a support factor's self-check differed by {diff:.1e} relative"));
            return insert(Slot::Full);
        }
        insert(Slot::Support(Arc::new(entry)))
    }
}

impl Default for SupportCache {
    fn default() -> SupportCache {
        SupportCache::new()
    }
}

impl Clone for SupportCache {
    /// The same entries (struct doc).
    fn clone(&self) -> SupportCache {
        SupportCache { state: Mutex::new(self.state().clone()) }
    }
}

impl PartialEq for SupportCache {
    /// Always equal: derived data takes no part in a model's equality (struct doc).
    fn eq(&self, _: &SupportCache) -> bool {
        true
    }
}

impl std::fmt::Debug for SupportCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let st = self.state();
        let entries: Vec<(u32, Option<usize>)> = st
            .entries
            .iter()
            .map(|(k, v)| match v {
                Slot::Support(e) => (*k, Some(e.support.len())),
                Slot::Full => (*k, None),
            })
            .collect();
        f.debug_struct("SupportCache").field("entries", &entries).field("builds", &st.builds).finish()
    }
}

/// The per-step hot path: total energy and forces without materialising per-atom descriptor
/// gradients (module doc). O(N²) pair sweep, `radial` once per pair inside r_cut.
///
/// # Panics
/// If `pos.len() != z.len()`, or the model's field lengths disagree with its basis.
pub fn energy_and_forces(
    model: &Model,
    pos: &[[f64; 3]],
    z: &[u8],
) -> Result<(f64, Vec<[f64; 3]>), DescribeError> {
    assert_eq!(pos.len(), z.len(), "energy_and_forces: pos and z lengths differ");
    let basis = &model.basis;
    let s = basis.species_indices(z)?;
    let (c_tilde, offset) = model.effective_coefficients();
    let n_max = basis.n_max;
    let n = pos.len();

    let mut energy = Neumaier::default();
    for &si in &s {
        energy.add(model.ref_energy[si] + offset[si]);
    }
    let mut forces = vec![[0.0; 3]; n];
    let mut rn = vec![0.0; n_max];
    let mut drn = vec![0.0; n_max];
    for i in 0..n {
        for j in i + 1..n {
            let Some((r, u)) = pair_geometry(pos[i], pos[j], basis.r_cut) else {
                continue;
            };
            radial(n_max, r, basis.r_cut, &mut rn, &mut drn);
            let ci = &c_tilde[s[i]][basis.body2_index(s[j], 0)..basis.body2_index(s[j], n_max)];
            let cj = &c_tilde[s[j]][basis.body2_index(s[i], 0)..basis.body2_index(s[i], n_max)];
            let mut e_pair = 0.0;
            let mut a = 0.0;
            for nn in 0..n_max {
                let c = ci[nn] + cj[nn];
                e_pair += c * rn[nn];
                a += c * drn[nn];
            }
            energy.add(e_pair);
            let f = [a * u[0], a * u[1], a * u[2]];
            for k in 0..3 {
                forces[i][k] += f[k];
                forces[j][k] -= f[k];
            }
        }
    }
    if basis.body3 || basis.body4 {
        body3_energy_and_forces(basis, &c_tilde, &s, pos, &mut energy, &mut forces);
    }
    Ok((energy.sum(), forces))
}

/// The body-3 block of the hot path (module doc, "Body-3 forces by the adjoint"): adds every
/// atom's E_i^{(3)} to `energy` and the pair-antisymmetric force contributions to `forces`.
fn body3_energy_and_forces(
    basis: &Basis,
    c_tilde: &[Vec<f64>],
    s: &[usize],
    pos: &[[f64; 3]],
    energy: &mut Neumaier,
    forces: &mut [[f64; 3]],
) {
    let n = pos.len();
    let n_max = basis.n_max;
    let p2 = basis.n_body2();
    let terms = basis.body3_terms();
    let p4 = p2 + terms.len();
    let terms4: Vec<(Body4Term, &[Gaunt])> = basis.body4_terms().into_iter().map(|t| (t, gaunt(t.l))).collect();
    let mut sc = AtomicBasisScratch::new(basis, n);
    let nlm = sc.nlm;
    let mut w = vec![0.0; basis.n_body2() * nlm];
    for i in 0..n {
        sc.clear();
        for j in 0..n {
            if j == i {
                continue;
            }
            let Some((r, u)) = pair_geometry(pos[i], pos[j], basis.r_cut) else {
                continue;
            };
            let d = [pos[j][0] - pos[i][0], pos[j][1] - pos[i][1], pos[j][2] - pos[i][2]];
            sc.add_neighbour(basis, j, s[j], d, r, u);
        }
        if sc.neighbours.is_empty() {
            continue;
        }
        // Energy and the adjoint W = ∂E_i/∂A.
        let ci = &c_tilde[s[i]][p2..];
        w.iter_mut().for_each(|v| *v = 0.0);
        let mut e3 = Neumaier::default();
        for (q, t) in terms.iter().enumerate() {
            let c = ci[q];
            if c == 0.0 {
                continue;
            }
            let (c1, c2) = (basis.body2_index(t.t1, t.n1), basis.body2_index(t.t2, t.n2));
            let range = AtomicBasisScratch::l_range(t.l);
            let mut dot = 0.0;
            for lm in range.clone() {
                let (a1, a2) = (sc.a[c1 * nlm + lm], sc.a[c2 * nlm + lm]);
                dot += a1 * a2;
                w[c1 * nlm + lm] += c * a2;
                w[c2 * nlm + lm] += c * a1;
            }
            e3.add(c * dot);
        }
        // Body-4 (descriptor module doc): E += c Σ G A₁A₂A₃, W_f += c Σ G ∏_{other two} A.
        let ci4 = &c_tilde[s[i]][p4..];
        for (q, (t, entries)) in terms4.iter().enumerate() {
            let c = ci4[q];
            if c == 0.0 {
                continue;
            }
            let ch = t4_channels(basis, t);
            let mut val = 0.0;
            for e in entries.iter() {
                let a = [sc.a[ch[0] * nlm + e.lm[0]], sc.a[ch[1] * nlm + e.lm[1]], sc.a[ch[2] * nlm + e.lm[2]]];
                val += e.g * a[0] * a[1] * a[2];
                let cg = c * e.g;
                w[ch[0] * nlm + e.lm[0]] += cg * a[1] * a[2];
                w[ch[1] * nlm + e.lm[1]] += cg * a[0] * a[2];
                w[ch[2] * nlm + e.lm[2]] += cg * a[0] * a[1];
            }
            e3.add(c * val);
        }
        energy.add(e3.sum());
        // Forces: ∂E_i/∂r_j = Σ_{n,lm} W_{(s_j,n),lm} ∇φ_{n,lm}(d_ij).
        for (slot, &j) in sc.neighbours.iter().enumerate() {
            let mut g = [0.0; 3];
            for nn in 0..n_max {
                let wrow = &w[(s[j] * n_max + nn) * nlm..(s[j] * n_max + nn + 1) * nlm];
                let dp = sc.dphi(slot, nn);
                for lm in 0..nlm {
                    let wv = wrow[lm];
                    g[0] += wv * dp[lm][0];
                    g[1] += wv * dp[lm][1];
                    g[2] += wv * dp[lm][2];
                }
            }
            for a in 0..3 {
                forces[j][a] -= g[a];
                forces[i][a] += g[a];
            }
        }
    }
}

/// Neumaier compensated sum (Kahan with the improved correction for |term| > |sum|).
#[derive(Default)]
struct Neumaier {
    sum: f64,
    comp: f64,
}

impl Neumaier {
    #[inline]
    fn add(&mut self, x: f64) {
        let t = self.sum + x;
        if self.sum.abs() >= x.abs() {
            self.comp += (self.sum - t) + x;
        } else {
            self.comp += (x - t) + self.sum;
        }
        self.sum = t;
    }

    #[inline]
    fn sum(&self) -> f64 {
        self.sum + self.comp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neumaier_recovers_a_cancelling_sum_that_plain_addition_loses() {
        let mut n = Neumaier::default();
        let terms = [1.0, 1e100, 1.0, -1e100];
        let plain: f64 = terms.iter().sum();
        for t in terms {
            n.add(t);
        }
        assert_eq!(plain, 0.0);
        assert_eq!(n.sum(), 2.0);
    }

    fn tiny_model() -> Model {
        let mut basis = Basis::new(&[1, 8]);
        basis.body3 = false; // body-2 regression fixture (the default is body-3)
        let p = basis.n_features();
        Model {
            basis,
            coeffs: vec![vec![0.5; p], vec![-0.25; p]],
            mean: vec![2.0; p],
            scale: vec![4.0; p],
            ref_energy: vec![-0.5, -75.0],
            sigma_e: 0.0,
            sigma_f: 0.0,
            energy_per_atom: true,
            lambda: 0.0,
            sigma_inv_chol: None,
            n_samples: 0,
            provenance: Provenance {
                grid_n: 1,
                h: 1.0,
                sigma: 1.0,
                tol: 1.0,
                charge: 0,
                multiplicity: None,
                teacher: String::new(),
                masses: Vec::new(),
                ionic: "none".into(),
            },
            representatives: OnceLock::new(),
            variance_support: SupportCache::new(),
        }
    }

    #[test]
    fn coincident_atoms_in_the_hot_path_give_a_finite_energy_and_zero_force() {
        let mut m = tiny_model();
        m.mean = vec![0.0; m.basis.n_features()];
        let n_max = m.basis.n_max;
        let pos = [[0.3, -0.1, 2.0], [0.3, -0.1, 2.0], [0.3, -0.1, 2.0 + 1.5]];
        let z = [1, 8, 1];
        let (e, f) = energy_and_forces(&m, &pos, &z).unwrap();
        assert!(e.is_finite());
        // The coincident pair (H at 0, O at 1) contributes Σ_n [c̃_H,(O,n) + c̃_O,(H,n)]·1 (R_n(0) f_c(0) = 1).
        let (ct, _) = m.effective_coefficients();
        let pair_01: f64 = (0..n_max)
            .map(|nn| ct[0][m.basis.body2_index(1, nn)] + ct[1][m.basis.body2_index(0, nn)])
            .sum();
        // Remove atom 2's pairs by evaluating the two-atom system separately.
        let (e01, f01) = energy_and_forces(&m, &pos[..2], &z[..2]).unwrap();
        assert!((e01 - (m.ref_energy[0] + m.ref_energy[1] + pair_01)).abs() < 1e-12);
        assert_eq!(f01, vec![[0.0; 3]; 2], "no direction at r = 0 → exactly zero force");
        // In the three-atom system the coincident pair still adds no force; atoms 0 and 1 feel
        // only atom 2, along z, and momentum is conserved.
        assert_eq!((f[0][0], f[0][1], f[1][0], f[1][1]), (0.0, 0.0, 0.0, 0.0));
        let sum: f64 = f.iter().map(|v| v[2]).sum();
        assert!(sum.abs() < 1e-15);
        assert!(f[2][2].abs() > 0.0);
        // Variance path with a coincident pair stays finite too (identity posterior factor on
        // the Q′ independent columns).
        let q = m.basis.n_independent_columns();
        let mut l = vec![0.0; q * q];
        for i in 0..q {
            l[i * q + i] = 1.0;
        }
        m.sigma_inv_chol = Some(l);
        let pred = m.predict(&pos, &z, true).unwrap();
        assert!(pred.energy_std.unwrap().is_finite());
        assert!(pred.force_std.unwrap().iter().all(|s| s.is_finite()));
    }

    #[test]
    fn effective_coefficients_fold_the_standardisation() {
        let mut m = tiny_model();
        m.scale[3] = 0.0; // constant feature: divided by 1
        let (ct, off) = m.effective_coefficients();
        assert_eq!(ct[0][0], 0.125);
        assert_eq!(ct[0][3], 0.5);
        assert_eq!(ct[1][0], -0.0625);
        // offset_s = −Σ_k c̃_{s,k}·mean_k = −(15·0.125 + 0.5)·2 for s = 0.
        assert!((off[0] + 2.0 * (15.0 * 0.125 + 0.5)).abs() < 1e-15);
        assert!((off[1] - 2.0 * (15.0 * 0.0625 + 0.25)).abs() < 1e-15);
    }

    #[test]
    #[should_panic(expected = "Model.mean: one entry per feature")]
    fn check_shapes_names_the_offending_field() {
        let mut m = tiny_model();
        m.mean.pop();
        m.check_shapes();
    }

    // ---- the predictive variance on the support (spec 2026-11-20-variance-support-design.md) ----

    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn provenance() -> Provenance {
        tiny_model().provenance
    }

    /// A model on `species` (body-3 on or off) with seeded random coefficients and scales and a
    /// posterior factor L = chol(λI + BBᵀ/k) over its Q′ independent columns (B random Q′ × k).
    fn random_posterior_model(rng: &mut StdRng, species: &[u8], body3: bool) -> Model {
        let mut basis = Basis::new(species);
        basis.body3 = body3;
        random_posterior_on(rng, basis)
    }

    /// [`random_posterior_model`] on a given basis.
    fn random_posterior_on(rng: &mut StdRng, basis: Basis) -> Model {
        let p = basis.n_features();
        let s = basis.n_species();
        let q = basis.n_independent_columns();
        let k = 24;
        let b: Vec<f64> = (0..q * k).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let mut a = vec![0.0; q * q];
        for i in 0..q {
            for j in 0..=i {
                let d: f64 = (0..k).map(|t| b[i * k + t] * b[j * k + t]).sum::<f64>() / k as f64;
                a[i * q + j] = d;
                a[j * q + i] = d;
            }
            a[i * q + i] += 1e-2;
        }
        crate::linalg::cholesky_in_place(&mut a, q).unwrap();
        Model {
            basis,
            coeffs: (0..s).map(|_| (0..p).map(|_| rng.gen_range(-1.0..1.0)).collect()).collect(),
            mean: vec![0.0; p],
            scale: (0..p).map(|_| rng.gen_range(0.5..2.0)).collect(),
            ref_energy: (0..s).map(|_| rng.gen_range(-80.0..-0.4)).collect(),
            sigma_e: 1e-4,
            sigma_f: 1e-3,
            energy_per_atom: true,
            lambda: 1e-2,
            sigma_inv_chol: Some(a),
            n_samples: 0,
            provenance: provenance(),
            representatives: OnceLock::new(),
            variance_support: SupportCache::new(),
        }
    }

    /// A seeded geometry of `n` atoms drawn from `species`, every pair at least 0.9 Bohr apart,
    /// in a 3.5 Bohr box (inside r_cut, so every atom neighbours every other).
    fn random_scene(rng: &mut StdRng, species: &[u8], n: usize) -> (Vec<[f64; 3]>, Vec<u8>) {
        loop {
            let pos: Vec<[f64; 3]> = (0..n).map(|_| [rng.gen_range(0.0..3.5), rng.gen_range(0.0..3.5), rng.gen_range(0.0..3.5)]).collect();
            let ok = (0..n).all(|i| (i + 1..n).all(|j| (0..3).map(|a| (pos[i][a] - pos[j][a]).powi(2)).sum::<f64>() > 0.81));
            if ok {
                return (pos, (0..n).map(|_| species[rng.gen_range(0..species.len())]).collect());
            }
        }
    }

    /// A model fitted by `fit` to a seeded random linear teacher with noise (`samples` scenes of
    /// 2–4 atoms over `species`): the posterior factor of a real fit, its conditioning included.
    fn fitted_model(rng: &mut StdRng, species: &[u8], body3: bool, samples: usize) -> Model {
        let teacher = {
            let mut t = random_posterior_model(rng, species, body3);
            t.sigma_inv_chol = None;
            t.scale = vec![1.0; t.basis.n_features()];
            t
        };
        let mut store = crate::fit::SampleStore::new(provenance());
        for _ in 0..samples {
            let n = rng.gen_range(2..=4);
            let (pos, z) = random_scene(rng, species, n);
            let (e, f) = energy_and_forces(&teacher, &pos, &z).unwrap();
            let e = e + 1e-4 * rng.gen_range(-1.0..1.0);
            let f = f
                .iter()
                .map(|v| [v[0] + 1e-3 * rng.gen_range(-1.0..1.0), v[1] + 1e-3 * rng.gen_range(-1.0..1.0), v[2] + 1e-3 * rng.gen_range(-1.0..1.0)])
                .collect();
            store.push(crate::fit::Sample::new(pos, z, e, f, crate::fit::SampleTag::Perturbation, 1.0)).unwrap();
        }
        let (m, _) = crate::fit::fit(&store, &teacher.basis, &teacher.ref_energy, &crate::fit::FitOptions::default()).unwrap();
        assert!(m.sigma_inv_chol.is_some());
        m
    }

    /// The largest relative difference of the support path's three outputs to the full path's,
    /// asserting the spec's bound |a − b| ≤ 1e-12|b| + 1e-15 on each.
    fn assert_matches_full(m: &Model, pos: &[[f64; 3]], z: &[u8], what: &str) -> f64 {
        let l = m.sigma_inv_chol.as_ref().unwrap();
        let (e_s, f_s, fe_s) = m.predictive_std_with(l, pos, z, VariancePath::Auto).unwrap();
        let (e_f, f_f, fe_f) = m.predictive_std_with(l, pos, z, VariancePath::Full).unwrap();
        let mut worst = 0.0f64;
        let mut check = |a: f64, b: f64, name: &str| {
            assert!((a - b).abs() <= 1e-12 * b.abs() + 1e-15, "{what}: {name} {a:e} (support) vs {b:e} (full)");
            if b != 0.0 {
                worst = worst.max((a - b).abs() / b.abs());
            }
        };
        check(e_s, e_f, "energy_std");
        for j in 0..pos.len() {
            check(f_s[j], f_f[j], "force_std");
            check(fe_s[j], fe_f[j], "force_std_epistemic");
        }
        // and `predict` itself reports the support path's numbers, bit for bit
        let pred = m.predict(pos, z, true).unwrap();
        assert_eq!(pred.energy_std.unwrap().to_bits(), e_s.to_bits(), "{what}: predict takes the same path");
        worst
    }

    #[test]
    fn support_variance_matches_the_full_path() {
        let mut rng = StdRng::seed_from_u64(2026_11_20);
        let mut worst = 0.0f64;
        // (model species, body-3, the scenes' species): the scenes touch a part of the basis
        let cases: [(&[u8], bool, &[u8]); 5] = [
            (&[1, 8], false, &[1]),
            (&[1, 8], true, &[1]),
            (&[1, 8], true, &[8]),
            (&[1, 8, 15], true, &[1, 8]),
            (&[1, 6, 8], true, &[6]),
        ];
        let mut supported = 0;
        let mut n_models = 0;
        for (k, &(species, body3, scene_species)) in cases.iter().enumerate() {
            // a fit on a three-species body-3 basis (Q′ 543) takes minutes: those cases fit the
            // body-2 basis of the same species instead (the posterior's conditioning is the fit's)
            let models = [random_posterior_model(&mut rng, species, body3), fitted_model(&mut rng, species, body3 && species.len() < 3, 30)];
            n_models += models.len();
            for (which, m) in models.iter().enumerate() {
                for t in 0..6 {
                    let n = 2 + t % 3;
                    let (pos, z) = random_scene(&mut rng, scene_species, n);
                    worst = worst.max(assert_matches_full(m, &pos, &z, &format!("case {k} model {which} scene {t}")));
                }
                if m.variance_support.support_entries() > 0 {
                    supported += 1;
                }
                // the self-check compares forms (variances), whose relative difference is twice
                // a std's: the spec's 1e-12 on the stds is 2e-12 here (round 2: the canonical
                // cluster's forms measured 1.0e-12 on these fixtures)
                assert!(m.variance_support.self_check_max() <= 2e-12, "self-check {:.2e}", m.variance_support.self_check_max());
            }
        }
        assert_eq!(supported, n_models, "every model ran on the support path");
        println!("support variance vs full path: worst relative difference {worst:.2e}");
    }

    /// S(C) holds every column a scene of composition C can touch (a feature of an absent
    /// channel species is an empty sum, exactly 0.0), and for one species it is exactly that
    /// species' own basis (the reviewer's sizes on the collection basis: 52 for {H}, 468 for a
    /// pair, 1 902 for all three).
    #[test]
    fn the_composition_support_covers_every_scene_of_it() {
        let mut basis = Basis::new(&[1, 8, 15]);
        basis.body4 = true;
        basis.body4_degree = 3;
        let mut model = tiny_model();
        model.basis = basis.clone();
        let rep = model.representatives().to_vec();
        assert_eq!(rep.len(), 1902);
        let size = |mask: u32| composition_support(&basis, &rep, mask).len();
        assert_eq!((size(0b001), size(0b011), size(0b101), size(0b110), size(0b111)), (52, 468, 468, 468, 1902));
        let mut single = Basis::new(&[1]);
        single.body4 = true;
        single.body4_degree = 3;
        assert_eq!(size(0b001), single.n_independent_columns());
        // every geometric support inside its composition's S(C)
        let mut rng = StdRng::seed_from_u64(71);
        let mut body4 = Basis::new(&[1, 8]);
        body4.body4 = true;
        body4.body4_degree = 3;
        let models = [
            (vec![1u8, 8], random_posterior_model(&mut rng, &[1, 8], false)),
            (vec![1, 8], random_posterior_model(&mut rng, &[1, 8], true)),
            (vec![1, 8, 15], random_posterior_model(&mut rng, &[1, 8, 15], true)),
            (vec![1, 8], random_posterior_on(&mut rng, body4)),
        ];
        for (species, m) in &models {
            let species = &species[..];
            for t in 0..12 {
                let pick: Vec<u8> = species.iter().copied().filter(|_| rng.gen_bool(0.6)).collect();
                let pick = if pick.is_empty() { vec![species[0]] } else { pick };
                let (pos, z) = random_scene(&mut rng, &pick, 2 + t % 3);
                let s = m.basis.species_indices(&z).unwrap();
                let sc = composition_support(&m.basis, m.representatives(), mask_of(&s).unwrap());
                for v in m.query_vectors(&pos, &z).unwrap() {
                    assert!(v.iter().enumerate().all(|(k, &x)| x == 0.0 || sc.binary_search(&k).is_ok()), "{species:?} scene {t}");
                }
            }
        }
    }

    /// The std is a pure function of (L, geometry, species set): two different orders of
    /// compositions, a clone taken mid-way and a fresh model give the same bits.
    #[test]
    fn stds_do_not_depend_on_the_call_history() {
        let mut rng = StdRng::seed_from_u64(73);
        let base = random_posterior_model(&mut rng, &[1, 8, 15], true);
        let probes: Vec<(Vec<[f64; 3]>, Vec<u8>)> = [&[1u8][..], &[8], &[1, 8], &[15], &[1, 15]]
            .iter()
            .map(|sp| random_scene(&mut rng, sp, 3))
            .collect();
        let bits = |m: &Model, order: &[usize]| -> Vec<Vec<u64>> {
            let mut out = vec![Vec::new(); probes.len()];
            for &k in order {
                let p = m.predict(&probes[k].0, &probes[k].1, true).unwrap();
                let mut b = vec![p.energy_std.unwrap().to_bits()];
                b.extend(p.force_std.unwrap().iter().chain(&p.force_std_epistemic.unwrap()).map(|x| x.to_bits()));
                out[k] = b;
            }
            out
        };
        let fresh = |m: &Model| {
            let mut c = m.clone();
            c.variance_support = SupportCache::new();
            c
        };
        let forward = bits(&fresh(&base), &[0, 1, 2, 3, 4]);
        let backward = bits(&fresh(&base), &[4, 3, 2, 1, 0]);
        let a = fresh(&base);
        let _ = bits(&a, &[2, 4]);
        let clone = a.clone();
        let from_clone = bits(&clone, &[1, 0, 3, 2, 4]);
        let from_original = bits(&a, &[3, 0, 4, 1, 2]);
        assert_eq!(forward, backward);
        assert_eq!(forward, from_clone);
        assert_eq!(forward, from_original);
        assert!(a.variance_support.support_entries() >= 4, "the support path was taken");
    }

    #[test]
    fn one_factor_per_composition() {
        let mut rng = StdRng::seed_from_u64(41);
        // three species, so H₂ with an O stays within Q′/2 (with two, it does not); body-2, as a
        // three-species body-3 fit takes minutes
        let m = fitted_model(&mut rng, &[1, 8, 15], false, 30);
        let q = m.basis.n_independent_columns();
        let h2 = (vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], vec![1u8, 1]);
        let h2_o = (vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.4], [2.5, 0.3, 0.7]], vec![1u8, 1, 8]);
        assert_eq!(m.variance_support.support_entries(), 0);
        assert_matches_full(&m, &h2.0, &h2.1, "H2");
        let s1 = m.variance_support.support_for(&[0]).expect("H2 is on the support path");
        assert_eq!(m.variance_support.builds(), 1);
        assert_matches_full(&m, &[[0.0, 0.0, 0.0], [0.1, 0.0, 1.6]], &[1, 1], "H2 again");
        assert_eq!(m.variance_support.builds(), 1, "the same composition: no build");
        assert_matches_full(&m, &h2_o.0, &h2_o.1, "H2 + O");
        let s2 = m.variance_support.support_for(&[0, 1]).unwrap();
        assert!(s1.iter().all(|k| s2.contains(k)) && 2 * s2.len() <= q);
        assert_eq!(m.variance_support.builds(), 2);
        assert_eq!(m.variance_support.support_for(&[0]).unwrap(), s1, "{{H}} keeps its own factor");
        println!("|S|: {{H}} {}, {{H, O}} {} (Q′ {q})", s1.len(), s2.len());
    }

    #[test]
    fn a_new_model_starts_with_an_empty_cache() {
        let mut rng = StdRng::seed_from_u64(43);
        let m1 = fitted_model(&mut rng, &[1, 8], true, 30);
        let (pos, z) = random_scene(&mut rng, &[1], 3);
        assert_matches_full(&m1, &pos, &z, "first fit");
        assert_eq!(m1.variance_support.support_entries(), 1);
        // a clone has the same L, so it shares the built factors (derived data of equal inputs)
        let c = m1.clone();
        assert_eq!(c.variance_support.support_for(&[0]), m1.variance_support.support_for(&[0]));
        assert_eq!(c, m1, "the cache takes no part in equality");
        // a refit (here a fit to more data, as a background refit's fork makes) is a new Model
        let m2 = fitted_model(&mut StdRng::seed_from_u64(43), &[1, 8], true, 45);
        assert_ne!(m2.sigma_inv_chol, m1.sigma_inv_chol);
        assert_eq!(m2.variance_support.support_entries(), 0, "a new model starts with an empty cache");
        assert_eq!(m2.variance_support.builds(), 0);
        assert_matches_full(&m2, &pos, &z, "the refit");
        let fresh = m2.clone();
        assert_matches_full(&fresh, &pos, &z, "a clone of the refit");
    }

    /// L changed in place behind the cache's back (as the providers' `always_unsure` does): the
    /// fingerprint drops the stale factor and the answer is the new L's.
    #[test]
    fn a_factor_changed_in_place_drops_the_stale_entries() {
        let mut rng = StdRng::seed_from_u64(79);
        let mut m = random_posterior_model(&mut rng, &[1, 8], true);
        let (pos, z) = random_scene(&mut rng, &[1], 2);
        assert_matches_full(&m, &pos, &z, "before");
        let before = m.predict(&pos, &z, true).unwrap().energy_std.unwrap();
        for x in m.sigma_inv_chol.as_mut().unwrap().iter_mut() {
            *x *= 1e-9;
        }
        assert_matches_full(&m, &pos, &z, "after the in-place change");
        let after = m.predict(&pos, &z, true).unwrap().energy_std.unwrap();
        assert!(after > 1e8 * before, "{after:e} vs {before:e}: the inflated posterior is used");
        assert_eq!(m.variance_support.builds(), 2, "rebuilt for the new L");
    }

    /// A build whose self-check fails (the bound lowered below any difference) is never used:
    /// that composition takes the full path for good, bit for bit, and the difference is kept.
    #[test]
    fn a_failed_self_check_takes_the_full_path() {
        let mut rng = StdRng::seed_from_u64(83);
        let m = random_posterior_model(&mut rng, &[1, 8], true);
        m.variance_support.state().bound = Some(-1.0);
        let (pos, z) = random_scene(&mut rng, &[1], 3);
        let l = m.sigma_inv_chol.as_ref().unwrap();
        for _ in 0..2 {
            let auto = m.predictive_std_with(l, &pos, &z, VariancePath::Auto).unwrap();
            let full = m.predictive_std_with(l, &pos, &z, VariancePath::Full).unwrap();
            assert_eq!(auto.0.to_bits(), full.0.to_bits());
            assert!(auto.1.iter().zip(&full.1).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        assert_eq!(m.variance_support.support_entries(), 0);
        assert_eq!(m.variance_support.builds(), 1, "built once, then the full path for good");
        assert!(m.variance_support.self_check_max() < 1e-12, "the measured difference {:.2e}", m.variance_support.self_check_max());
    }

    #[test]
    fn large_support_uses_the_full_path() {
        let mut rng = StdRng::seed_from_u64(47);
        // one species: S({H}) is the whole basis
        let m = random_posterior_model(&mut rng, &[1], true);
        let (pos, z) = random_scene(&mut rng, &[1], 4);
        let l = m.sigma_inv_chol.as_ref().unwrap();
        let auto = m.predictive_std_with(l, &pos, &z, VariancePath::Auto).unwrap();
        let full = m.predictive_std_with(l, &pos, &z, VariancePath::Full).unwrap();
        assert_eq!(m.variance_support.support_entries(), 0, "|S| > Q′/2: no factor");
        assert_eq!(m.variance_support.builds(), 0);
        assert_eq!(auto.0.to_bits(), full.0.to_bits());
        assert!(auto.1.iter().zip(&full.1).all(|(a, b)| a.to_bits() == b.to_bits()));
        assert!(auto.2.iter().zip(&full.2).all(|(a, b)| a.to_bits() == b.to_bits()));
        // all three species: the full path, while {H} keeps its factor
        let m3 = random_posterior_model(&mut rng, &[1, 8, 15], true);
        let (p_small, z_small) = random_scene(&mut rng, &[1], 2);
        assert_matches_full(&m3, &p_small, &z_small, "small first");
        let s = m3.variance_support.support_for(&[0]).unwrap();
        let p_big = vec![[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.6, 0.0], [0.0, 0.0, 1.7], [1.4, 1.5, 1.6], [-1.2, 0.8, 0.3]];
        let z_big = vec![1u8, 8, 15, 1, 8, 15];
        let l3 = m3.sigma_inv_chol.as_ref().unwrap();
        let a = m3.predictive_std_with(l3, &p_big, &z_big, VariancePath::Auto).unwrap();
        let f = m3.predictive_std_with(l3, &p_big, &z_big, VariancePath::Full).unwrap();
        assert_eq!(a.0.to_bits(), f.0.to_bits(), "the large call is the full path bit for bit");
        assert!(a.1.iter().zip(&f.1).all(|(x, y)| x.to_bits() == y.to_bits()));
        assert_eq!(m3.variance_support.support_for(&[0]).unwrap(), s);
        assert_eq!(m3.variance_support.builds(), 1);
    }

    /// The model is shared with the auditor's thread: concurrent calls over several compositions
    /// neither deadlock nor see a torn factor; every answer matches the full path, bit for bit
    /// with `predict` (the std does not depend on which thread built the factor).
    #[test]
    fn concurrent_calls_agree_with_the_full_path() {
        let mut rng = StdRng::seed_from_u64(53);
        let m = random_posterior_model(&mut rng, &[1, 8, 15], true);
        let kinds: [&[u8]; 4] = [&[1], &[8], &[15], &[1, 8]];
        let scenes: Vec<(Vec<[f64; 3]>, Vec<u8>)> = (0..24).map(|t| random_scene(&mut rng, kinds[t % 4], 2 + t % 2)).collect();
        std::thread::scope(|sc| {
            for w in 0..4 {
                let (m, scenes) = (&m, &scenes);
                sc.spawn(move || {
                    for (t, (pos, z)) in scenes.iter().enumerate().skip(w).chain(scenes.iter().enumerate()) {
                        assert_matches_full(m, pos, z, &format!("thread {w} scene {t}"));
                    }
                });
            }
        });
        assert!(m.variance_support.support_entries() >= 3);
    }

    /// The query vectors as built before 2026-11-20: full-width rows, restricted afterwards.
    fn query_vectors_by_full_rows(m: &Model, pos: &[[f64; 3]], z: &[u8]) -> Result<Vec<Vec<f64>>, DescribeError> {
        let basis = &m.basis;
        let p = basis.n_features();
        let q = basis.n_species() * p;
        // The posterior lives on the independent columns (`fit` docs, "Tied columns"): restrict
        // every query vector to the class representatives before the quadratic form.
        let rep = m.representatives();
        let qr = rep.len();
        let s = basis.species_indices(z)?;
        let desc = describe(basis, pos, z)?;
        let inv_scale = m.inverse_scale();
        let n = pos.len();
        let mut reduced = vec![0.0; qr];
        fn restrict(dst: &mut [f64], rep: &[usize], v: &[f64]) {
            for (r, &c) in dst.iter_mut().zip(rep) {
                *r = v[c];
            }
        }

        // Energy: φ̃ = Σ_i φ̃_i.
        let mut phi = vec![0.0; q];
        for (i, d) in desc.iter().enumerate() {
            let block = &mut phi[s[i] * p..(s[i] + 1) * p];
            for k in 0..p {
                block[k] += d.b[k] * inv_scale[k];
            }
        }
        restrict(&mut reduced, rep, &phi);

        // Forces: g̃_{j,a} = −Σ_i ∂B̃_i/∂r_{j,a}, gathered from every centre's gradient list.
        let mut rows = vec![vec![0.0; q]; 3 * n];
        for (i, d) in desc.iter().enumerate() {
            let off = s[i] * p;
            for (j, g) in &d.grad {
                for k in 0..p {
                    let gk = g[k];
                    let sc = inv_scale[k];
                    rows[3 * j][off + k] -= gk[0] * sc;
                    rows[3 * j + 1][off + k] -= gk[1] * sc;
                    rows[3 * j + 2][off + k] -= gk[2] * sc;
                }
            }
        }
        let mut restricted = Vec::with_capacity(1 + 3 * n);
        restricted.push(reduced);
        for row in &rows {
            let mut r = vec![0.0; qr];
            restrict(&mut r, rep, row);
            restricted.push(r);
        }
        Ok(restricted)
    }

    #[test]
    fn query_vectors_are_the_restricted_full_rows() {
        let mut rng = StdRng::seed_from_u64(61);
        for (species, body3) in [(&[1u8, 8][..], false), (&[1, 8], true), (&[1, 8, 15], true)] {
            let m = random_posterior_model(&mut rng, species, body3);
            for t in 0..4 {
                let (pos, z) = random_scene(&mut rng, species, 2 + t);
                let got = m.query_vectors(&pos, &z).unwrap();
                let want = query_vectors_by_full_rows(&m, &pos, &z).unwrap();
                assert_eq!(got.len(), want.len());
                for (g, w) in got.iter().zip(&want) {
                    assert!(g.iter().zip(w).all(|(a, b)| a.to_bits() == b.to_bits()), "{species:?} scene {t}");
                }
            }
        }
    }

    /// A factor with a zero on its diagonal, or a NaN in a row the scene's support never reaches
    /// (hand-built models; `fit` never writes one): the cache finds L unusable before building
    /// and every call gives the full path's numbers bit for bit (non-finite here), never a panic
    /// and never a finite answer the full path would not give.
    #[test]
    fn an_unusable_factor_takes_the_full_path() {
        let mut rng = StdRng::seed_from_u64(67);
        let base = random_posterior_model(&mut rng, &[1, 8, 15], true);
        let q = base.basis.n_independent_columns();
        let (p_h, z_h) = random_scene(&mut rng, &[1], 2);
        let (p_o, z_o) = random_scene(&mut rng, &[8], 2);
        for (what, at, value, pos, z) in [("zero pivot", 0, 0.0, &p_h, &z_h), ("NaN before S", q + 0, f64::NAN, &p_o, &z_o)] {
            let mut m = base.clone();
            m.variance_support = SupportCache::new();
            m.sigma_inv_chol.as_mut().unwrap()[at] = value;
            let l = m.sigma_inv_chol.clone().unwrap();
            let auto = m.predictive_std_with(&l, pos, z, VariancePath::Auto).unwrap();
            let full = m.predictive_std_with(&l, pos, z, VariancePath::Full).unwrap();
            assert!(!full.0.is_finite(), "{what}: the full path gives {:e}", full.0);
            assert_eq!(auto.0.to_bits(), full.0.to_bits(), "{what}");
            assert!(auto.1.iter().zip(&full.1).all(|(a, b)| a.to_bits() == b.to_bits()), "{what}");
            assert_eq!(m.variance_support.support_entries(), 0);
            assert_eq!(m.variance_support.builds(), 0, "{what}: nothing built");
        }
    }

    /// A call with no neighbour inside r_cut (its query vectors all zero) answers +0.0 forms and
    /// settles nothing (review round 2, I1); the composition is settled, and checked, by the
    /// first call that has neighbours: with the bound below any difference it takes the full path.
    #[test]
    fn a_call_without_neighbours_settles_nothing() {
        let mut rng = StdRng::seed_from_u64(89);
        let m = random_posterior_model(&mut rng, &[1, 8], true);
        m.variance_support.state().bound = Some(-1.0);
        let l = m.sigma_inv_chol.as_ref().unwrap();
        for (pos, z) in [(vec![[0.0, 0.0, 0.0]], vec![1u8]), (vec![[0.0, 0.0, 0.0], [0.0, 0.0, 7.0]], vec![1u8, 1])] {
            let auto = m.predictive_std_with(l, &pos, &z, VariancePath::Auto).unwrap();
            let full = m.predictive_std_with(l, &pos, &z, VariancePath::Full).unwrap();
            assert_eq!(auto.0.to_bits(), full.0.to_bits());
            assert!(auto.2.iter().zip(&full.2).all(|(a, b)| a.to_bits() == b.to_bits() && *a == 0.0));
        }
        assert_eq!((m.variance_support.settled(), m.variance_support.builds()), (0, 0), "nothing settled");
        let h2 = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]];
        let auto = m.predictive_std_with(l, &h2, &[1, 1], VariancePath::Auto).unwrap();
        let full = m.predictive_std_with(l, &h2, &[1, 1], VariancePath::Full).unwrap();
        assert_eq!(auto.0.to_bits(), full.0.to_bits());
        assert_eq!((m.variance_support.settled(), m.variance_support.support_entries(), m.variance_support.builds()), (1, 0, 1), "checked by the real call: the full path");
        assert!(m.variance_support.self_check_max() >= 0.0);
    }

    /// The self-check's maximum propagates NaN (review round 2, M2).
    #[test]
    fn a_nan_difference_fails_the_self_check() {
        assert!(max_relative_difference(&[1.0, f64::NAN, 2.0], &[1.0, 1.0, 2.0]).is_nan());
        assert!(max_relative_difference(&[1.0, 1.0], &[f64::NAN, 1.0]).is_nan());
        assert!(!(max_relative_difference(&[f64::NAN], &[1.0]) <= SELF_CHECK_BOUND));
        assert_eq!(max_relative_difference(&[1.0, 3.0], &[1.0, 2.0]), 0.5);
    }

    /// The canonical cluster holds every species of its composition, inside r_cut.
    #[test]
    fn the_canonical_cluster_spans_its_composition() {
        let basis = Basis::new(&[1, 6, 8, 15]);
        for mask in 1u32..16 {
            let (pos, z) = canonical_cluster(&basis, mask);
            let want: Vec<u8> = (0..4).filter(|s| mask & (1 << s) != 0).map(|s| basis.species[s]).collect();
            assert!(want.iter().all(|w| z.contains(w)) && z.iter().all(|x| want.contains(x)));
            for i in 0..pos.len() {
                for j in i + 1..pos.len() {
                    let d = (0..3).map(|a| (pos[i][a] - pos[j][a]).powi(2)).sum::<f64>().sqrt();
                    assert!(d > 0.5 && d < basis.r_cut, "mask {mask:b}: atoms {i} {j} at {d}");
                }
            }
        }
    }

    /// The support path against the full one on a real fit (review I3b, round 2): 1 000 seeded H₂
    /// and H₃ geometries on the model in `MODEL_FILE`. For every geometry and every form (the
    /// 1 + 3N quadratic forms), each path's relative error against a double-double evaluation
    /// through L; and the largest relative std difference between the paths. A measurement.
    /// `MODEL_FILE=<a model sidecar> cargo test -p surrogate --release --lib support_accuracy_on_a_model_file -- --ignored --nocapture`
    #[test]
    #[ignore = "a measurement on a model file (MODEL_FILE)"]
    fn support_accuracy_on_a_model_file() {
        use rayon::prelude::*;
        let path = std::path::PathBuf::from(std::env::var("MODEL_FILE").expect("MODEL_FILE"));
        let m = crate::sidecar::read(&path).expect("a readable model sidecar").model;
        let l = m.sigma_inv_chol.clone().expect("a posterior");
        let qr = m.representatives().len();
        let mut rng = StdRng::seed_from_u64(0xbe57);
        let geometries: Vec<(Vec<[f64; 3]>, Vec<u8>)> = (0..1000)
            .map(|k| {
                let n = if k % 2 == 0 { 2 } else { 3 };
                let mut pos = vec![[0.0; 3]; n];
                pos[1] = [0.0, 0.0, rng.gen_range(0.9..3.5)];
                if n == 3 {
                    pos[2] = [rng.gen_range(-2.5..2.5), rng.gen_range(0.6..3.0), rng.gen_range(-1.0..4.0)];
                }
                (pos, vec![1u8; n])
            })
            .collect();
        let t = std::time::Instant::now();
        // the std difference between the paths, and the factor for {H} (built by the first call)
        let mut worst_std = 0.0f64;
        for (pos, z) in &geometries {
            let a = m.predictive_std_with(&l, pos, z, VariancePath::Auto).unwrap();
            let f = m.predictive_std_with(&l, pos, z, VariancePath::Full).unwrap();
            let pairs: Vec<(f64, f64)> = std::iter::once((a.0, f.0)).chain(a.1.iter().copied().zip(f.1.iter().copied())).chain(a.2.iter().copied().zip(f.2.iter().copied())).collect();
            let (x, y): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();
            worst_std = worst_std.max(max_relative_difference(&x, &y));
        }
        let entry = match m.variance_support.state().entries.get(&1).cloned() {
            Some(Slot::Support(e)) => e,
            _ => panic!("{{H}} is on the support path"),
        };
        // every form of every geometry against double-double (in parallel; each is independent)
        let errors: Vec<(f64, f64, f64)> = geometries
            .par_iter()
            .map(|(pos, z)| {
                let s = m.basis.species_indices(z).unwrap();
                let desc = describe(&m.basis, pos, z).unwrap();
                let q = QueryInputs { species: &s, desc: &desc, inv_scale: m.inverse_scale() };
                let full_vs = m.full_query_vectors(&q);
                let on_s = accumulate_query_vectors(&m.basis, &q, &entry.map, entry.support.len()).unwrap();
                let (sup, full) = (entry.quad_forms(&on_s), full_quad_forms(&l, qr, &full_vs));
                let (mut es, mut ef, mut between) = (0.0f64, 0.0f64, 0.0f64);
                for (j, v) in full_vs.iter().enumerate() {
                    let exact = crate::linalg::tests::quad_form_dd(&l, qr, v);
                    if exact > 0.0 {
                        es = es.max((sup[j] - exact).abs() / exact);
                        ef = ef.max((full[j] - exact).abs() / exact);
                        between = between.max((sup[j] - full[j]).abs() / full[j]);
                    }
                }
                (es, ef, between)
            })
            .collect();
        let max = |f: fn(&(f64, f64, f64)) -> f64| errors.iter().map(f).fold(0.0, f64::max);
        let median = |f: fn(&(f64, f64, f64)) -> f64| {
            let mut v: Vec<f64> = errors.iter().map(f).collect();
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let ratio = errors.iter().filter(|e| e.1 > 0.0).map(|e| e.0 / e.1).fold(0.0, f64::max);
        println!(
            "Q′ {qr}, |S({{H}})| {}: 1000 H₂/H₃ geometries, every form against double-double through L —\n  support: max {:.2e}, median {:.2e}; full: max {:.2e}, median {:.2e}; support vs full on the forms: max {:.2e}; on the stds: max {worst_std:.2e}\n  largest per-geometry ratio (support error / full error) {ratio:.1}; self-check (canonical cluster) {:.2e}; {:.0} s",
            entry.support.len(),
            max(|e| e.0),
            median(|e| e.0),
            max(|e| e.1),
            median(|e| e.1),
            max(|e| e.2),
            m.variance_support.self_check_max(),
            t.elapsed().as_secs_f64()
        );
    }
}
