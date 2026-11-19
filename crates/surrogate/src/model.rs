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
//! ## On the scene's support (spec `2026-11-20-variance-support-design.md`)
//!
//! A scene's query vectors are non-zero on few of the Q′ columns (H₂ in the H/O/P collection
//! basis: 52 of 1 902 — the H-centre features with H neighbours). Let S contain the support of
//! every query vector of the call (an entry that is exactly 0.0 adds nothing to any product, so
//! leaving it out is exact). With A = L_A L_Aᵀ the posterior precision, the block-inverse rule
//! gives
//!
//!   φ̃ᵀA⁻¹φ̃ = φ̃_Sᵀ (A⁻¹)_SS φ̃_S = ‖R φ̃_S‖²,   (A⁻¹)_SS = XᵀX = RᵀR,   X = L_A⁻¹E_S,
//!
//! the same quantity at |S|²/2 per vector instead of Q′²/2. R v_S is not formed by multiplying:
//! in f64 its products cancel (the unit columns L_A⁻¹e_s carry the prior's variance ~1/λ, a query
//! in a direction the data pin down has a variance orders of magnitude smaller; measured up to
//! 3.5e-11 relative on fitted factors for the spec's f64 QR of X, against ≤ 2.2e-13 for the full
//! substitution). `linalg::support_factor` builds R in double-double and keeps W = R⁻¹ (in
//! reversed order, a lower triangle), so the forms are one more batched substitution
//! (`cholesky_quad_forms` of size |S|) — componentwise stable like the full one, and agreeing
//! with it to its own rounding (≤ 4.4e-13 relative on the stds in the tests).
//!
//! The model caches S and the factor (`Model::variance_support`, a [`SupportCache`]). A call whose
//! support is inside S uses it; one that is not grows S to the union and rebuilds, so S grows to
//! the composition's reachable set instead of churning as neighbours come and go. When the union
//! would exceed Q′/2 the call takes the full factor (the support gains little there; the cached
//! entry stays for the smaller calls), and so does every call of a model without a posterior
//! (none is asked for) or with a factor the full substitution cannot use (a non-finite entry or
//! a zero pivot: the full path then reports the non-finite stds L gives). The value depends on S only at rounding level: the same geometry can read
//! a last-bit-different std before and after S grows; every trust rule compares against bounds
//! many orders wider.
//!
//! Cost: one `describe`, then the forms: on the support |S|²/2 multiply-adds per vector (1 + 3N
//! of them) and a rebuild when S grows (double-double, Σ_s (Q′ − s)²/2 + Q′|S|²/2 and O(|S|³):
//! 9.5 ms at |S| = 52, 57 ms at 282, 0.26 s at 950 for Q′ = 1 902); on the full factor 3N + 1
//! triangular solves of size Q′ (the independent columns, `Basis::n_independent_columns`),
//! O(Q′²) each.

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
    /// Lazily cached factor of the predictive variance on the scenes' support (module doc,
    /// "Predictive variance"); derived data of `sigma_inv_chol` and `representatives`,
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

    /// [`Model::predictive_std`] with the quadratic forms on the support when it is small enough
    /// (`Auto`, module doc) or always on the full factor (`Full`, the oracle of the tests).
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
        let restricted = self.query_vectors(pos, z)?;
        let quad = match path {
            VariancePath::Auto => self.variance_support.quad_forms(l, qr, &restricted),
            VariancePath::Full => full_quad_forms(l, qr, &restricted),
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

    /// The 1 + 3N query vectors of the predictive variance on the Q′ independent columns (module
    /// doc): the energy's φ̃, then g̃_{j,a} for j = 0..N, a = x, y, z. Accumulated straight onto
    /// the class representatives (`representatives`): each entry sees the same additions in the
    /// same order as a full-width row restricted afterwards, so the values are those bit for bit
    /// (`query_vectors_are_the_restricted_full_rows`).
    fn query_vectors(&self, pos: &[[f64; 3]], z: &[u8]) -> Result<Vec<Vec<f64>>, DescribeError> {
        let basis = &self.basis;
        let p = basis.n_features();
        let q = basis.n_species() * p;
        // The posterior lives on the independent columns (`fit` docs, "Tied columns"): only the
        // class representatives' entries are formed.
        let rep = self.representatives();
        let qr = rep.len();
        let mut reduced_of = vec![usize::MAX; q];
        for (r, &c) in rep.iter().enumerate() {
            reduced_of[c] = r;
        }
        let s = basis.species_indices(z)?;
        let desc = describe(basis, pos, z)?;
        let inv_scale = self.inverse_scale();
        let n = pos.len();
        let mut out = vec![vec![0.0; qr]; 1 + 3 * n];
        for (i, d) in desc.iter().enumerate() {
            let block = &reduced_of[s[i] * p..(s[i] + 1) * p];
            // Energy: φ̃ = Σ_i φ̃_i.
            for k in 0..p {
                if block[k] != usize::MAX {
                    out[0][block[k]] += d.b[k] * inv_scale[k];
                }
            }
            // Forces: g̃_{j,a} = −Σ_i ∂B̃_i/∂r_{j,a}, gathered from every centre's gradient list.
            for (j, g) in &d.grad {
                for k in 0..p {
                    let r = block[k];
                    if r == usize::MAX {
                        continue;
                    }
                    let gk = g[k];
                    let sc = inv_scale[k];
                    out[1 + 3 * j][r] -= gk[0] * sc;
                    out[2 + 3 * j][r] -= gk[1] * sc;
                    out[3 + 3 * j][r] -= gk[2] * sc;
                }
            }
        }
        Ok(out)
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
    /// The cached support's factor when the call's support fits (|S| ≤ Q′/2), else the full L.
    Auto,
    /// Always the full factor L (the tests' oracle; `Auto` falls back to the same code).
    #[cfg_attr(not(test), allow(dead_code))]
    Full,
}

/// ‖L⁻¹v‖² of every query vector: every quadratic form of the step in one batched substitution
/// (`cholesky_quad_forms`: bit for bit the one-at-a-time values, L read once per eight rows;
/// 2026-11-17).
fn full_quad_forms(l: &[f64], qr: usize, vs: &[Vec<f64>]) -> Vec<f64> {
    cholesky_quad_forms(l, qr, &vs.iter().map(|v| &v[..]).collect::<Vec<_>>())
}

/// The predictive variance's factor on the scenes' support (module doc, "Predictive variance"):
/// the support S (reduced column indices, ascending) and its factor F (`linalg::support_factor`:
/// the R of L⁻¹E_S built in double-double and kept as R⁻¹, solved rather than multiplied). Derived
/// data of the model's `sigma_inv_chol` and `representatives`, built lazily by `predict`.
///
/// * **Growth.** A call whose support is not inside S replaces the entry by one on the union, so
///   S only grows (to the composition's reachable set) and never churns as neighbours come and
///   go. A call whose support is inside S uses the entry as it is.
/// * **Threads.** The model is shared with the auditor's thread: the entry is an immutable
///   `Arc` behind a `Mutex` that is held only to read or swap the `Arc`, never while F is built,
///   so a build can use the thread pool and nothing waits on another thread's build (no lock is
///   held across a call into rayon, where a stolen job could re-enter `predict`). Two threads
///   that grow S at once may both build; the entry installed is always a superset of the one it
///   replaces (a build that is not is merged and rebuilt), and a reader holds a whole entry, so
///   no F is ever torn. A poisoned lock is recovered: the entry is only ever replaced whole.
/// * **Equality and clone.** Not part of `PartialEq` (two models equal in their fields are equal
///   whatever they cached); `Clone` copies the entry (the `Arc`), valid because the clone's L is
///   the same bit for bit — replace the cell if `sigma_inv_chol` or `basis` is then changed in
///   place. A refit makes a new `Model` and so a new, empty cache.
pub struct SupportCache {
    state: Mutex<SupportState>,
}

#[derive(Clone, Default)]
struct SupportState {
    entry: Option<Arc<SupportFactor>>,
    builds: u64,
    /// L is not a usable factor (a non-finite entry in its lower triangle or a zero on its
    /// diagonal, checked before the first build, or `linalg::support_factor` gave `None`): every
    /// call takes the full path, which reports what L gives (non-finite stds for such a factor;
    /// the support path could miss a bad entry outside S and answer finitely).
    unusable: bool,
}

/// S and its factor (module doc).
struct SupportFactor {
    /// Reduced column indices, ascending.
    support: Vec<usize>,
    /// Whether each reduced column is in S (length Q′).
    member: Vec<bool>,
    /// The |S| × |S| lower factor F of `linalg::support_factor` (row-major, S reversed).
    factor: Vec<f64>,
}

impl SupportFactor {
    fn covers(&self, call: &[usize]) -> bool {
        call.iter().all(|&k| self.member[k])
    }

    /// The factor on `support` (ascending; `linalg::support_factor`), `None` for an unusable L.
    fn build(l: &[f64], qr: usize, support: Vec<usize>) -> Option<SupportFactor> {
        let factor = support_factor(l, qr, &support)?;
        let mut member = vec![false; qr];
        for &k in &support {
            member[k] = true;
        }
        Some(SupportFactor { support, member, factor })
    }

    /// vᵀA⁻¹v = ‖F⁻¹ J v_S‖² for every query vector (each supported on S): the entries on S
    /// gathered in reverse order, then one batched substitution of size |S|.
    fn quad_forms(&self, vs: &[Vec<f64>]) -> Vec<f64> {
        let gathered: Vec<Vec<f64>> = vs.iter().map(|v| self.support.iter().rev().map(|&k| v[k]).collect()).collect();
        cholesky_quad_forms(&self.factor, self.support.len(), &gathered.iter().map(|v| &v[..]).collect::<Vec<_>>())
    }
}

impl SupportCache {
    /// An empty cache (nothing built).
    pub fn new() -> SupportCache {
        SupportCache { state: Mutex::new(SupportState::default()) }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, SupportState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The cached support S (reduced column indices, ascending), `None` before the first build.
    pub fn support(&self) -> Option<Vec<usize>> {
        self.state().entry.as_ref().map(|e| e.support.clone())
    }

    /// How many factors this cache has built (a measure of the rebuilds; the tests count them).
    pub fn builds(&self) -> u64 {
        self.state().builds
    }

    /// vᵀA⁻¹v for every query vector (each of length Q′), on the support when the union of the
    /// cached S and the call's support has |S| ≤ Q′/2, else on the full factor (module doc).
    fn quad_forms(&self, l: &[f64], qr: usize, vs: &[Vec<f64>]) -> Vec<f64> {
        // the call's support: a value that is exactly 0.0 adds nothing to any product
        let call: Vec<usize> = (0..qr).filter(|&k| vs.iter().any(|v| v[k] != 0.0)).collect();
        if call.is_empty() {
            // every vector is zero: ‖L⁻¹0‖² = 0 exactly, as the full path gives
            return vec![0.0; vs.len()];
        }
        let Some(entry) = self.entry_covering(l, qr, &call) else {
            return full_quad_forms(l, qr, vs);
        };
        entry.quad_forms(vs)
    }

    /// The entry whose S covers `call`, built (on the union with the current S) if need be;
    /// `None` when that union exceeds Q′/2 (the full path; the cached entry is kept for the
    /// smaller calls) or L is unusable.
    fn entry_covering(&self, l: &[f64], qr: usize, call: &[usize]) -> Option<Arc<SupportFactor>> {
        let mut current = {
            let mut st = self.state();
            if st.entry.is_none() && !st.unusable && !factor_usable(l, qr) {
                st.unusable = true;
            }
            if st.unusable {
                return None;
            }
            st.entry.clone()
        };
        loop {
            if let Some(e) = current.as_ref().filter(|e| e.covers(call)) {
                return Some(e.clone());
            }
            let mut union: Vec<usize> = current.as_ref().map_or_else(Vec::new, |e| e.support.clone());
            union.extend_from_slice(call);
            union.sort_unstable();
            union.dedup();
            if 2 * union.len() > qr {
                return None;
            }
            // built without the lock (struct doc, "Threads")
            let built = SupportFactor::build(l, qr, union);
            let mut st = self.state();
            st.builds += 1;
            let Some(built) = built.map(Arc::new) else {
                st.unusable = true;
                return None;
            };
            match &st.entry {
                // another thread installed a superset of ours meanwhile (it covers this call)
                Some(e) if e.covers(&built.support) => return Some(e.clone()),
                // the two grew apart: merge with the installed one and build again
                Some(e) if !built.covers(&e.support) => {
                    current = Some(e.clone());
                    continue;
                }
                // ours contains the installed one, or nothing is installed: install it
                _ => {
                    st.entry = Some(built.clone());
                    return Some(built);
                }
            }
        }
    }
}

/// Whether the full substitution through L gives finite numbers for finite vectors: every entry
/// it reads (the lower triangle) finite and no zero on the diagonal. O(Q′²) reads, once per cache.
fn factor_usable(l: &[f64], qr: usize) -> bool {
    (0..qr).all(|i| l[i * qr + i] != 0.0 && l[i * qr..i * qr + i + 1].iter().all(|v| v.is_finite()))
}

impl Default for SupportCache {
    fn default() -> SupportCache {
        SupportCache::new()
    }
}

impl Clone for SupportCache {
    /// The same entry (the clone's L is the same; struct doc).
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
        f.debug_struct("SupportCache").field("support_len", &st.entry.as_ref().map(|e| e.support.len())).field("builds", &st.builds).finish()
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
        // and `predict` itself reports the support path's numbers
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
                if m.variance_support.support().is_some() {
                    supported += 1;
                }
            }
        }
        assert_eq!(supported, n_models, "every model ran on the support path");
        println!("support variance vs full path: worst relative difference {worst:.2e}");
    }

    #[test]
    fn the_support_grows_and_never_shrinks() {
        let mut rng = StdRng::seed_from_u64(41);
        // three species, so H₂ with an O stays within Q′/2 (with two, it does not); body-2, as a
        // three-species body-3 fit takes minutes
        let m = fitted_model(&mut rng, &[1, 8, 15], false, 30);
        let q = m.basis.n_independent_columns();
        let h2 = (vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], vec![1u8, 1]);
        let h2_o = (vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.4], [2.5, 0.3, 0.7]], vec![1u8, 1, 8]);
        assert!(m.variance_support.support().is_none());
        assert_matches_full(&m, &h2.0, &h2.1, "H2");
        let s1 = m.variance_support.support().expect("H2 is on the support path");
        let b1 = m.variance_support.builds();
        assert_matches_full(&m, &h2_o.0, &h2_o.1, "H2 + O");
        let s2 = m.variance_support.support().unwrap();
        assert!(s2.len() > s1.len() && s1.iter().all(|k| s2.contains(k)), "S grows to a strict superset: {} -> {}", s1.len(), s2.len());
        assert!(2 * s2.len() <= q, "|S| {} still within Q′/2 = {}", s2.len(), q / 2);
        let b2 = m.variance_support.builds();
        assert!(b2 > b1);
        // plain H₂ again, at another bond length: the grown factor answers it without a rebuild
        assert_matches_full(&m, &[[0.0, 0.0, 0.0], [0.1, 0.0, 1.6]], &[1, 1], "H2 again");
        assert_eq!(m.variance_support.support().unwrap(), s2, "S never shrinks");
        assert_eq!(m.variance_support.builds(), b2, "no rebuild for a covered support");
        println!("|S|: H2 {}, H2 + O {} (Q′ {q}); builds {b1}, {b2}", s1.len(), s2.len());
    }

    #[test]
    fn a_new_model_starts_with_an_empty_cache() {
        let mut rng = StdRng::seed_from_u64(43);
        let m1 = fitted_model(&mut rng, &[1, 8], true, 30);
        let (pos, z) = random_scene(&mut rng, &[1], 3);
        assert_matches_full(&m1, &pos, &z, "first fit");
        assert!(m1.variance_support.support().is_some());
        // a clone has the same L, so it shares the built support factor (derived data of equal inputs)
        let c = m1.clone();
        assert_eq!(c.variance_support.support(), m1.variance_support.support());
        assert_eq!(c, m1, "the cache takes no part in equality");
        // a refit (here a fit to more data, as a background refit's fork makes) is a new Model
        let m2 = fitted_model(&mut StdRng::seed_from_u64(43), &[1, 8], true, 45);
        assert_ne!(m2.sigma_inv_chol, m1.sigma_inv_chol);
        assert!(m2.variance_support.support().is_none(), "a new model starts with an empty cache");
        assert_eq!(m2.variance_support.builds(), 0);
        assert_matches_full(&m2, &pos, &z, "the refit");
        let fresh = m2.clone();
        assert_matches_full(&fresh, &pos, &z, "a clone of the refit");
    }

    #[test]
    fn large_support_uses_the_full_path() {
        let mut rng = StdRng::seed_from_u64(47);
        // one species: an H₄ cluster touches almost every column
        let m = random_posterior_model(&mut rng, &[1], true);
        let q = m.basis.n_independent_columns();
        let (pos, z) = random_scene(&mut rng, &[1], 4);
        let l = m.sigma_inv_chol.as_ref().unwrap();
        let auto = m.predictive_std_with(l, &pos, &z, VariancePath::Auto).unwrap();
        let full = m.predictive_std_with(l, &pos, &z, VariancePath::Full).unwrap();
        assert!(m.variance_support.support().is_none(), "|S| > Q′/2 = {}: nothing cached", q / 2);
        assert_eq!(m.variance_support.builds(), 0);
        assert_eq!(auto.0.to_bits(), full.0.to_bits());
        assert!(auto.1.iter().zip(&full.1).all(|(a, b)| a.to_bits() == b.to_bits()));
        assert!(auto.2.iter().zip(&full.2).all(|(a, b)| a.to_bits() == b.to_bits()));
        // a model with a small support cached keeps it while a large call takes the full path
        let m3 = random_posterior_model(&mut rng, &[1, 8, 15], true);
        let (p_small, z_small) = random_scene(&mut rng, &[1], 2);
        assert_matches_full(&m3, &p_small, &z_small, "small first");
        let s = m3.variance_support.support().unwrap();
        let p_big = vec![[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.6, 0.0], [0.0, 0.0, 1.7], [1.4, 1.5, 1.6], [-1.2, 0.8, 0.3]];
        let z_big = vec![1u8, 8, 15, 1, 8, 15];
        let l3 = m3.sigma_inv_chol.as_ref().unwrap();
        let a = m3.predictive_std_with(l3, &p_big, &z_big, VariancePath::Auto).unwrap();
        let f = m3.predictive_std_with(l3, &p_big, &z_big, VariancePath::Full).unwrap();
        assert_eq!(a.0.to_bits(), f.0.to_bits(), "the large call is the full path bit for bit");
        assert!(a.1.iter().zip(&f.1).all(|(x, y)| x.to_bits() == y.to_bits()));
        assert_eq!(m3.variance_support.support().unwrap(), s, "the small support stays cached");
    }

    /// The model is shared with the auditor's thread: concurrent calls that grow S in different
    /// directions neither deadlock nor see a torn factor; every answer matches the full path.
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
        assert!(m.variance_support.support().is_some());
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
            assert!(m.variance_support.support().is_none());
            assert_eq!(m.variance_support.builds(), 0, "{what}: nothing built");
        }
    }
}
