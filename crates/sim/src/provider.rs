//! The force boundary (CLAUDE.md rule 5): [`ForceProvider`] turns a [`Scene`]'s nuclear
//! positions into the Born-Oppenheimer energy and the forces on the nuclei; the integrator never
//! knows what computes them. The implementations live in the `providers` crate (M6, Task 1):
//! the two Kohn-Sham adapters (`providers::ks`: `CpuKs`, `GpuKs` — their module doc has the warm
//! starts, the R15 cold retry and the display copies) and the surrogate (`providers::surrogate`).
//! A provider that learns — the surrogate — is reached through [`ForceProvider::as_trainable`]
//! and the [`Trainable`] trait: the driver's sampling pass and model commands and the worker's
//! refits and readouts go through it, and no caller downcasts a provider (the M5 `Any` accessor
//! was deleted with the move, review S4).
//!
//! # Display payload
//! Each successful evaluation can carry the converged electron density for the viewer
//! ([`Display`]; on by default, `set_display(false)` turns it off): the CPU adapter as f32
//! values, the GPU adapter as a 3D texture filled on the device (`copy_density_to_volume`, no
//! CPU round trip) from a pool that never overwrites a texture someone still holds
//! (`providers::ks`'s `TexturePool`). `sim` carries the payload opaquely — nothing here interprets it — so the
//! renderer still never learns what computed the density.
//!
//! # Orbital display (orbital-viewer spec §4.3–4.4; plan Task 3)
//! The Kohn-Sham orbitals reach the viewer the same way: [`ForceProvider::set_orbital_view`]
//! says what to show ([`OrbitalViewRequest`]), and each successful evaluation then carries an
//! [`OrbitalDisplay`] in [`ForceResult::orbitals`] — the list ([`OrbitalInfo`] rows, which mirror
//! `qm::orbitals::Row` so that `sim` does not depend on `qm`), the spin readout and the field (a
//! texture on the provider's device or values to upload, [`OrbitalField`]) — computed **after**
//! the step's energy and forces are final, from the converged state, never touching the solve
//! (rule 6; `providers::ks`'s bitwise on/off test). [`ForceProvider::orbital_display_now`] redoes
//! it on the current converged state without a solve (the view switched on while paused).
//! **The rate bound lives in the provider**: during play at most one payload per
//! [`ORBITAL_MIN_INTERVAL`] of wall time; `force_every_step` (set by the driver while paused or
//! single-stepping) and the first evaluation after the view changed always carry one. A
//! surrogate has **one producer**, its background side solver (review SHOULD 12): only its audits'
//! and display refreshes' [`SideDisplay`]s carry orbitals; its model and fallback steps carry
//! none. `sim` never interprets a payload: the driver only keeps the newest
//! (`DynSnapshot::orbitals`) and drops it when the view is switched off.
//!
//! # Errors
//! - [`ProviderError::Cancelled`]: `should_abort` returned true (the GPU keeps the in-progress
//!   state as its warm start).
//! - [`ProviderError::Solver`]: the solver's own errors ([`SolverError`], `qm::QmError` mapped
//!   variant for variant by `providers::ks`) — coincident nuclei (raised before any work), no SCF
//!   convergence, and, GPU only, `Numerical` (non-finite numerics; the solver drops its state and
//!   the next evaluation is cold). After an error the CPU provider holds no warm state either.
//! - [`ProviderError::Invalid`]: the scene cannot be posed — a charge above the proton count, a
//!   multiplicity impossible for the electron count (what `qm::System::new` would panic on), or
//!   a nucleus closer than [`CELL_MARGIN_H`](crate::CELL_MARGIN_H)·h to a cell face or outside
//!   the cell ([`Scene::check_inside_cell`], R12: the density is cut off there, so the forces
//!   would be artefacts of the box). No solver error fits these: they are input errors, not
//!   solver outcomes.
//!
//! # Nothing to solve (the empty scene)
//! A scene with no nuclei (the app's `empty` preset) is answered without a solve by every
//! provider (`providers::ks`, `providers::surrogate`) through [`nothing_to_solve`]: E = 0, no
//! forces, no density, `EvalStats::default()` — the exact answer, since there is no nucleus and
//! (charge 0) no electron. `qm`/`qm-gpu` are never called with it. With no nuclei a non-zero
//! charge or a multiplicity other than 1 cannot be posed ([`ProviderError::Invalid`]): a positive
//! charge has no protons to come from, and a negative one would put free electrons in an empty
//! box — not a scene this sandbox poses. (Nuclei with no electrons, e.g. H⁺ or H₂²⁺, still go to
//! the solvers, which handle empty spin channels: `qm::scf` module doc.) The driver itself never
//! asks a provider about an empty scene (`crate::dynamics`, "The empty scene"); this guard is for
//! the other callers.
use crate::references::{AtomReference, AtomReferences};
use crate::sampling::{SamplingPlan, SamplingProgress};
use crate::scene::{Cell, Scene};
use gpu::Volume3D;
use std::sync::Arc;

/// One evaluation: E_KS (Hartree, nuclear repulsion included; the same number as
/// `stats.energy.total`) and the force on each nucleus (Hartree/Bohr, in `scene.nuclei` order),
/// plus the density for the viewer when the provider makes one (module doc, "Display payload")
/// and, when an orbital view is on and its rate allows, the orbital display (module doc,
/// "Orbital display"; `None` otherwise — the viewer keeps the last one).
#[derive(Clone, Debug)]
pub struct ForceResult {
    pub energy: f64,
    pub forces: Vec<[f64; 3]>,
    pub stats: EvalStats,
    pub display: Option<Display>,
    pub orbitals: Option<OrbitalDisplay>,
}

/// Wall time between two orbital payloads of one producer during play (orbital-viewer spec §4.4,
/// "Rate"): a step inside the interval carries none. Paused, single-stepping (the driver sets
/// [`OrbitalViewRequest::force_every_step`]) and the first evaluation after the view changed
/// always carry one.
pub const ORBITAL_MIN_INTERVAL: std::time::Duration = std::time::Duration::from_millis(100);

/// A spin channel of the orbital display (`qm::orbitals::Spin` mirrored so that `sim` does not
/// depend on `qm`; `providers` converts).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Spin {
    Up,
    Down,
}

/// The track an [`OrbitalRequest::Orbital`] uses to ask for "the HOMO": no producer ever issues it,
/// so the selection fallback shows the highest occupied orbital — and says nothing about a
/// vanished selection, because none was made (the viewer adopts the track the payload shows).
pub const TRACK_HOMO: u32 = u32::MAX;

/// What the orbital view shows (orbital-viewer spec §3). Tracks are the producer's continuity
/// tracks (spec §5.4: an orbital keeps its track while the atoms move and levels reorder); a
/// provider never reuses a track id for another orbital, also across its resets.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum OrbitalRequest {
    /// No orbital view: no payloads, and the producer's display state is dropped.
    #[default]
    Off,
    /// The list (rows, spin summary) without a field.
    List,
    /// One orbital by track: ψ̃ (signed, `squared` false) or ψ̃².
    Orbital { spin: Spin, track: u32, squared: bool },
    /// Every included orbital in its own colour (spec §5.6): `include` the tracks (`None` = all
    /// occupied), `physical` scales by max ρ instead of each orbital's peak. (The blend, additive
    /// or weighted, is a render setting: the producer's field does not depend on it, so it is not
    /// part of the request.)
    AllColoured { include: Option<Vec<u32>>, physical: bool },
    /// m = ρ↑ − ρ↓.
    SpinDensity,
}

impl OrbitalRequest {
    pub fn is_off(&self) -> bool {
        matches!(self, OrbitalRequest::Off)
    }
}

/// The orbital view as the app asks for it ([`ForceProvider::set_orbital_view`]): what, the
/// unoccupied orbitals wanted (spec §4.5; computed from plan Task 4 on), and whether every
/// evaluation must carry a payload regardless of [`ORBITAL_MIN_INTERVAL`] — the driver sets it
/// while paused or single-stepping, the no-impact tests always.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OrbitalViewRequest {
    pub what: OrbitalRequest,
    pub unoccupied: u8,
    pub force_every_step: bool,
}

impl OrbitalViewRequest {
    /// `what` with no unoccupied orbitals, rate-bounded.
    pub fn new(what: OrbitalRequest) -> OrbitalViewRequest {
        OrbitalViewRequest { what, unoccupied: 0, force_every_step: false }
    }

    /// The same view (what and unoccupied), whatever the rate flag.
    pub fn same_view(&self, other: &OrbitalViewRequest) -> bool {
        self.what == other.what && self.unoccupied == other.unoccupied
    }
}

/// A row label (`qm::orbitals::Label` mirrored): `#k` (1-based index in its channel by energy) or
/// an atomic n l m name (a lone atom whose shells could be labelled, spec §5.3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrbitalLabel {
    Index(usize),
    Atomic { n: u8, l: u8, m: &'static str },
}

impl std::fmt::Display for OrbitalLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrbitalLabel::Index(k) => write!(f, "#{k}"),
            OrbitalLabel::Atomic { n, l, m } => {
                write!(f, "{n}{}", ['s', 'p', 'd', 'f'][(*l as usize).min(3)])?;
                if m.is_empty() { Ok(()) } else { write!(f, "_{m}") }
            }
        }
    }
}

/// One row of the orbital list (`qm::orbitals::Row` mirrored, spec §3.1): channel, index in the
/// channel (energy order), track, colour slot (0..12, the palette is the renderer's), ε (Hartree;
/// ⟨ψ̃|H|ψ̃⟩ for a rotated member), `eps_range` (the lowest and highest **eigenvalue** of the row's
/// set — equal to ε for a row of its own; HOMO, LUMO and the gap are read off these, not off ε,
/// which for a rotated member is a Rayleigh quotient), occupation, label, the degenerate set (size, spread Ha), the
/// rotation and grid-split tags, box state, smeared 1s, and — unoccupied rows, plan Task 4 — the
/// eigen-residual and whether it ended above the eigensolve's tolerance.
///
/// Unoccupied rows have `occupation` 0. `box_state` (unoccupied rows only) marks a row whose set
/// has ε ≥ 0, more than `BOX_STATE_OUTER_WEIGHT` = 5 % of a member within L/6 of a face, or a set
/// density on the outermost grid plane above the surrogate's box-affected face density
/// (`qm::orbitals` module doc §8). Task 7 words the tag by the sign of ε (review M2): ε ≥ 0 →
/// "state of the box — unbound in LSDA at this box"; ε < 0 → "bound, but its tail reaches the box
/// walls: energy box-affected" (Li 2p: ε −0.0396 Ha, decay length 1.8 Bohr, outer weight 0.095).
/// `residual` is the largest ‖(H − ε)φ‖ (Ha, ‖φ‖ = 1) among the members of the row's degenerate
/// set — the **eigensolve residual of the set** (Task 7's label), not of the shown ψ̃, whose own
/// residual can reach ≈ max r + s/2 for a rotated row (review M4); `not_converged`: one of them
/// ended above the tolerance (the LOBPCG cap), shown anyway and tagged "not converged
/// (residual …)".
#[derive(Clone, Debug, PartialEq)]
pub struct OrbitalInfo {
    pub spin: Spin,
    pub index: usize,
    pub track: u32,
    pub slot: u8,
    pub energy: f64,
    pub eps_range: (f64, f64),
    pub occupation: f64,
    pub label: OrbitalLabel,
    pub cluster: Option<(usize, f64)>,
    pub rotated: bool,
    pub grid_split: Option<f64>,
    pub box_state: bool,
    pub smeared_1s: bool,
    pub residual: Option<f64>,
    pub not_converged: bool,
}

/// The spin readout (spec §5.5; `qm::orbitals::SpinSummary` with its display scale): occupied
/// orbitals per channel, ∫m, ∫|m| (electrons), max|m|, max ρ (e/Bohr³) and the scale of the spin
/// field, max(max|m|, `SPIN_FLOOR`·max ρ).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SpinSummary {
    pub n_up: usize,
    pub n_dn: usize,
    pub integral: f64,
    pub abs_integral: f64,
    pub max_abs: f64,
    pub rho_max: f64,
    pub scale: f64,
    /// The L1 norm (electrons) up to which ∫|m| is the solver's noise, not spin
    /// (`qm::orbitals::SpinSummary::noise_l1`: SCF tolerance, eigensolver tolerance, f32 precision
    /// — derived there). 0 = not known: nothing is judged to be noise.
    pub noise: f64,
}

impl SpinSummary {
    /// A closed shell: ∫|m| is within the solver's noise, so the spin density is not drawn (the
    /// provider sends no field) and the tab says "no net spin".
    pub fn no_net_spin(&self) -> bool {
        self.noise > 0.0 && self.abs_integral <= self.noise
    }
}

/// Whether a lone atom got atomic labels (`qm::orbitals::LoneAtom` mirrored, spec §5.3):
/// `Unlabelled` carries the reason for the `#k` labels, for the tab's honesty text (rule 7).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LoneAtomLabels {
    /// Not exactly one nucleus: molecular orbitals.
    #[default]
    NotLoneAtom,
    /// n l m labels; `shell_order_inverted`: a lower-l shell above a higher-l shell of the same n
    /// (the smeared nucleus, spec §2 item 5 — not physical).
    Labelled { shell_order_inverted: bool },
    Unlabelled(&'static str),
}

/// The unoccupied rows' state (spec §4.5, plan Task 4; `None` in a payload when none were asked
/// for). The rows come from a display-only eigensolve on the converged ground-state potential
/// (`qm::orbitals` module doc §8); it never touches the SCF, the energy or the forces.
///
/// **What the tab must say about them (rule 7; the facts behind Task 7's text):**
/// - Kohn–Sham unoccupied orbitals are eigenstates of the *ground-state* potential, not excited
///   states; their ε are not excitation energies (ε_a − ε_i is only the zeroth-order, uncoupled
///   estimate of a neutral excitation). LSDA, a continuous functional, has no derivative
///   discontinuity; its KS gap typically underestimates IP − EA by a lot.
/// - LSDA's potential decays faster than −1/r (the exact KS potential has a −1/r tail and a
///   Rydberg series of bound virtuals; LSDA has few or none), so its virtual orbitals are poorly
///   bound or unbound: H₂ has no unoccupied state below zero in boxes up to 24 Bohr, and H₂O's LSDA
///   virtual at ≈ −0.002 Ha (24 Bohr box, decay length ≈ 16 Bohr) is box-lifted above zero in the
///   12 Bohr box, not absent. In the finite Dirichlet box an unbound state becomes a **state of
///   the box** ([`OrbitalInfo::box_state`]): its energy and shape are the box's, and they change
///   with the box size.
///
/// Fields: asked for (m), shown and hidden per channel (the block-edge rule: a degenerate set cut
/// by the block edge is hidden, so fewer than m may be shown), the eigensolve's tolerance (Ha; the
/// SCF's effective f32-floor tolerance on the GPU), whether the rows were solved at this payload's
/// geometry (`current`; otherwise at `positions`, kept because the budget runs the eigensolve only
/// when paused — `paused_only` — or because a newer command interrupted it), the last eigensolve's
/// wall (s), a count of eigensolves so far (monotone: a new one changes it), and lines for the tab
/// (`notes`: "m requested, k shown…", a solve discarded by the occupied-eigenvalue check or
/// interrupted, a view refused by the GPU memory guard). The LOBPCG cap is accepted with honest
/// tags (Task 4 review I3): every shown row is at or below `tol` or tagged
/// [`OrbitalInfo::not_converged`].
///
/// `interrupted`: the last eigensolve attempt yielded to a newer command (Task 4 review I2), so
/// the rows are an earlier solve's; the driver retries once the worker is idle again (Task 4
/// re-review N2, `Dynamics::tick`). `held`: the m of the set the rows come from (0: none held).
/// It differs from `requested` (capped at the provider's maximum) after a change of m whose
/// eigensolve was interrupted (re-review N3): the rows are then the previous request's set, and
/// Task 7 shows `held` against `requested`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UnoccupiedStatus {
    pub requested: u8,
    pub shown: [usize; 2],
    pub hidden: [usize; 2],
    pub tol: f64,
    pub current: bool,
    pub positions: Vec<[f64; 3]>,
    pub paused_only: bool,
    pub wall_s: f64,
    pub solves: u64,
    pub notes: Vec<String>,
    pub interrupted: bool,
    pub held: u8,
}

/// The field of an orbital payload (spec §3's table): on the provider's device as a texture
/// (r32float [`Volume3D`] for an orbital or the spin density, rgba16float
/// [`ColourVolume3D`](gpu::ColourVolume3D) for the colour volume), or as values for the viewer
/// to upload (n³ in `qm::Grid` layout). `signed`: ψ̃ or m (else ψ̃²); `scale`: what 1.0 means —
/// max|ψ| over the orbital's set (its square for ψ̃²), the spin scale, 1 for colour (the
/// weights are divided out).
#[derive(Clone)]
pub enum OrbitalField {
    Gpu { volume: Arc<Volume3D>, signed: bool, scale: f32 },
    GpuColour { volume: Arc<gpu::ColourVolume3D>, scale: f32 },
    Cpu { values: Arc<Vec<f32>>, signed: bool, scale: f32 },
    CpuColour { rgba: Arc<Vec<[f32; 4]>>, scale: f32 },
}

impl std::fmt::Debug for OrbitalField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrbitalField::Gpu { volume, signed, scale } => write!(f, "OrbitalField::Gpu {{ dims {:?}, signed {signed}, scale {scale} }}", volume.dims),
            OrbitalField::GpuColour { volume, scale } => write!(f, "OrbitalField::GpuColour {{ dims {:?}, scale {scale} }}", volume.dims),
            OrbitalField::Cpu { values, signed, scale } => write!(f, "OrbitalField::Cpu {{ len {}, signed {signed}, scale {scale} }}", values.len()),
            OrbitalField::CpuColour { rgba, scale } => write!(f, "OrbitalField::CpuColour {{ len {}, scale {scale} }}", rgba.len()),
        }
    }
}

/// One orbital payload (spec §4.3), display only and opaque to `sim` (module doc, "Orbital
/// display"): the geometry it was solved at (`cell`, `origin` of grid point 0 as in [`Display`],
/// the nuclear `positions`), the list, the field (`None` for the list or when there is nothing to
/// draw — `notes` says why), the spin readout, the unoccupied state, and notes for the tab. `what`
/// is the request this payload answers *as resolved* — the orbital actually shown after a
/// selection fallback, the tracks actually included — and `lone_atom` the labelling verdict.
#[derive(Clone, Debug)]
pub struct OrbitalDisplay {
    pub cell: Cell,
    pub origin: [f64; 3],
    pub positions: Vec<[f64; 3]>,
    pub rows: Vec<OrbitalInfo>,
    pub field: Option<OrbitalField>,
    pub spin: SpinSummary,
    pub unoccupied: Option<UnoccupiedStatus>,
    pub notes: Vec<String>,
    pub what: OrbitalRequest,
    pub lone_atom: LoneAtomLabels,
}

/// The converged total density ρ↑ + ρ↓ (e/Bohr³) on the evaluation's grid, for display only.
/// `cell` is the scene's; `origin` is the position of grid point (0, 0, 0) (`qm::Grid::origin`);
/// values are in `qm::Grid` layout (x fastest); `max` is the largest value.
#[derive(Clone)]
pub enum Display {
    /// n³ f32 values.
    Cpu { cell: Cell, origin: [f64; 3], rho: Arc<Vec<f32>>, max: f32 },
    /// An n³ r32float texture on the provider's device (the `GpuContext` it was built with).
    Gpu { cell: Cell, origin: [f64; 3], volume: Arc<Volume3D>, max: f32 },
}

impl std::fmt::Debug for Display {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Display::Cpu { cell, origin, rho, max } => f.debug_struct("Display::Cpu").field("cell", cell).field("origin", origin)
                .field("len", &rho.len()).field("max", max).finish(),
            Display::Gpu { cell, origin, volume, max } => f.debug_struct("Display::Gpu").field("cell", cell).field("origin", origin)
                .field("dims", &volume.dims).field("max", max).finish(),
        }
    }
}

/// Who produced an evaluation's energy and forces (spec §5; ruling S1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Source {
    /// A Kohn-Sham provider (`providers::ks`: `CpuKs`, `GpuKs`) or any provider that is not a
    /// surrogate.
    #[default]
    Dft,
    /// The surrogate model answered (`providers::surrogate`).
    Surrogate,
    /// The surrogate handed the step to its teacher: the forces are the teacher's; the energy
    /// is the model's when a trusted model exists (`teacher_gap` is `Some`) and the teacher's
    /// otherwise (`teacher_gap` is `None`) — see [`EvalStats::energy_surface`].
    Fallback,
}

/// Which potential-energy surface an evaluation's `energy` is on (the surrogate's one-surface
/// accounting, `providers::surrogate` module doc). The driver restarts the `E_conserved` history
/// when consecutive evaluations change surface without a booked `refit_jump`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Surface {
    /// The provider's own (DFT) surface: `Source::Dft`, or a `Fallback` without a trusted model,
    /// at a geometry the model may not answer (out of its envelope, M6 Task 11), on the first
    /// evaluation after an edit, or after a failed audit (no `teacher_gap`).
    Teacher,
    /// The surrogate model's surface: `Source::Surrogate`, or a `Fallback` whose energy is the
    /// model's (`teacher_gap` present).
    Model,
}

/// The Kohn-Sham energy breakdown of an evaluation (Hartree): kinetic, external (electron–nucleus),
/// Hartree, exchange-correlation, nuclear repulsion and their total — `EvalStats::energy`. The
/// same fields as `qm::Energy`, which it carried until M6 (`providers::ks` converts at the
/// boundary, so `sim` does not depend on `qm`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EnergyBreakdown {
    pub kinetic: f64,
    pub external: f64,
    pub hartree: f64,
    pub xc: f64,
    pub nuclear: f64,
    pub total: f64,
}

/// Cost and diagnostics of one evaluation.
/// - `source`, `uncertainty`, `teacher_gap`, `refit_jump`, `model_forces` (ruling S1; the
///   surrogate provider's fields, `Source::Dft` and `None` for the Kohn-Sham adapters): who
///   answered; the surrogate's epistemic force uncertainty **per component**, max_j ‖δF_j‖/√3 (Ha/Bohr —
///   the number its policy compares with its thresholds; S1's "max per-atom force std" is √3
///   times it) when the model was consulted; E_teacher − E_model on a Fallback step booked on
///   the model surface (Ha); the change of the reported energy at the current positions caused
///   by a refit or a failed audit (once, on that evaluation) — or, when the surrogate's report
///   moves between the model's and the teacher's surface from one step to the next, that step's
///   booking E₁ − E₀ + ½(m̃₀ + m̃₁)·Δr (M6 Task 11) — which the driver books into its work term
///   so `E_conserved` stays continuous; and the model's forces on a Fallback step booked on the
///   model surface, so the driver can book the force-mismatch work ½(ΔF_old + ΔF_new)·Δr, ΔF =
///   `forces` − `model_forces` (the `providers::surrogate` module doc, "One-surface energy
///   accounting").
///   Between the surrogate's variance checks (`FallbackPolicy::variance_cadence` k) `uncertainty`
///   is the last check's number, reused for at most k − 1 consultations.
/// - `scf_steps`: SCF iterations; `readbacks`: GPU→CPU readbacks (GPU only — the GPU's cost
///   measure, `SolveProfile::readbacks`).
///   After a warm solve that stalled and was retried cold (module doc, R15) both include the
///   failed attempt's, and `notes` says so.
/// - `solve_ms`: wall of the evaluation — solve, forces, the display copy and, when the
///   evaluation carries an orbital payload, the orbital update (≈ 1–4 ms at 97³) and any
///   display eigensolve for unoccupied orbitals it ran (seconds; `UnoccupiedStatus::wall_s`) —
///   so the driver's step cost and budget packing see the worker's real time — **excluding**
///   the `GpuSolver` build on a grid change (one-off setup, ~0.3 s at 97³, kept out so a
///   per-step cost estimate is not spiked by it); reported, never a pass/fail measure.
/// - `warnings`: the solver's warnings (`ScfStats::warnings`: box too small, Poisson or
///   final-step LOBPCG not converged); `notes`: informational lines that are not warnings (GPU:
///   the `SolveProfile` notes and whether the start density was extrapolated; CPU: LOBPCG
///   stopping at its iteration cap).
/// - `face_density`: max |ρ| on the outermost grid plane inside the Dirichlet wall, one spacing
///   h in from it (e/Bohr³; `ScfStats::boundary_density`, the number behind the "box too small"
///   warning, which stays): how much the box confines this answer — the confinement energy per
///   unit of it scales as 1/h² (`providers::surrogate::box_affected_face_density`). `None` from a
///   provider without a grid (analytic, surrogate-answered steps).
/// - `eig_tol_used`: the LOBPCG tolerance of the final SCF step (GPU: the Δρ-tied tol,
///   `SolveProfile::eig_tol_used`; CPU: the fixed `ScfSettings::eig_tol`).
/// - `scf_tol_used`: the Δρ tolerance this solve actually stopped at (M7): the provider's `scf_tol`, except
///   where a GPU solver cannot reach it — under a pseudo setup its stop is max(tol, 2·floor(N)), the f32
///   density floor growing with the valence electron count (spec §7 risk 6). **A record, never compared**: the
///   surrogate's provenance carries the stop *rule* (in the ionic tag), the nominal tolerance as before, and
///   this number is what each answer ran at (`None` from an analytic provider or a model answer).
/// - `energy`: the breakdown (kinetic, external, Hartree, xc, nuclear, total — Hartree);
///   `max_eigen_residual`, `poisson_residual`: as `ScfStats` defines them (the worst LOBPCG
///   exit residual of the tight-tol steps; the worst relative Poisson residual). On a surrogate
///   `Fallback` step the breakdown is the TEACHER's (its `total` = `ForceResult::energy` +
///   `teacher_gap`); on a `Surrogate` step only `total` is filled (the model has no breakdown).
#[derive(Clone, Debug)]
pub struct EvalStats {
    pub source: Source,
    pub uncertainty: Option<f64>,
    pub teacher_gap: Option<f64>,
    pub refit_jump: Option<f64>,
    pub model_forces: Option<Vec<[f64; 3]>>,
    pub scf_steps: usize,
    pub readbacks: Option<usize>,
    pub solve_ms: u64,
    pub warnings: Vec<String>,
    pub notes: Vec<String>,
    pub face_density: Option<f64>,
    pub eig_tol_used: Option<f64>,
    pub scf_tol_used: Option<f64>,
    pub energy: EnergyBreakdown,
    pub max_eigen_residual: f64,
    pub poisson_residual: f64,
    /// A surrogate consultation's per-atom force uncertainty as a fraction of that atom's bound
    /// (force_std/√3 ÷ `FallbackPolicy::force_bound`; ≥ 1 is beyond what the model may answer
    /// with), from the latest variance evaluation (`providers::surrogate`, the cadence).
    pub atom_uncertainty: Option<Vec<f64>>,
    /// Draft mode (`FallbackPolicy::never_fallback`): the model answered a step it would
    /// otherwise have handed to its teacher — what it guessed beyond.
    pub extrapolation: Option<Extrapolation>,
    /// M7 (spec §4.7, §5.5): the species of this answer that carry a pseudopotential (empty under
    /// `IonicPolicy::None` and when nothing in the scene has a core), one summary per Z.
    pub pseudised: Vec<PseudoSummary>,
    /// M7 (spec §5.6): the pairs (I, J, R_IJ) whose frozen cores overlap — an answer with any is used
    /// (every field is finite) and never kept as training data. Each is also a warning, prefixed
    /// [`CORE_OVERLAP_WARNING`].
    pub core_overlaps: Vec<(usize, usize, f64)>,
    /// M7 (spec §5.7; Task 6 review A.6): the shared (fractionally occupied) degenerate shells of
    /// the final SCF step under `Occupations::DegenerateShells`; empty for integer occupations.
    pub fractional: Vec<SharedShell>,
    /// The shared-shell pattern (spin, first level, size) differs from the previous step's
    /// evaluation by the same provider: E(R) can jump by the orbital-polarisation energy there
    /// (a few mHa), so the answer is used and never kept as training data (Task 6 review A.6).
    /// Also a warning, [`OCCUPATION_CHANGE_WARNING`].
    pub occupation_changed: bool,
}

/// The prefix of the per-step core-overlap warning (spec §5.6): "cores overlap: I–J at R = … Bohr".
/// Pinned equal to `qm::ionic::CORE_OVERLAP_WARNING` in `providers`.
pub const CORE_OVERLAP_WARNING: &str = "cores overlap";

/// The prefix of the warning for a shared-shell pattern that changed between consecutive steps.
pub const OCCUPATION_CHANGE_WARNING: &str = "occupation pattern changed";

/// One shared (fractionally occupied) cluster of one spin channel (`qm::scf::FractionalShell`): spin
/// (0 ↑, 1 ↓), the index of its first level in ascending order, its size, the occupation f of each
/// level and the spread of its eigenvalues (Ha).
#[derive(Clone, Debug, PartialEq)]
pub struct SharedShell {
    pub spin: u8,
    pub first: usize,
    pub count: usize,
    pub f: f64,
    pub spread: f64,
}

/// What the generator recorded for one pseudised species, for the badge, the Selection tab and the
/// Quality menu (spec §4.7, §5.5): Z, the valence charge Z_v, the softness factor f, the local
/// channel, the largest r_c (Bohr), the coarsest grid spacing h_req at which the table is certified
/// (`None`: none on the ladder), the derived core radius, T8(e)'s egg-box grid h_eb (`None`: none —
/// reported, never refused on) and the full generation record as text.
#[derive(Clone, Debug, PartialEq)]
pub struct PseudoSummary {
    pub z: u8,
    pub z_v: f64,
    pub f: f64,
    pub local: String,
    pub r_c_max: f64,
    pub h_req: Option<f64>,
    pub h_eb: Option<f64>,
    pub r_core: f64,
    pub record: String,
}

/// What a draft-mode answer guessed beyond (`providers::surrogate`, "Draft mode").
#[derive(Clone, Debug, PartialEq)]
pub struct Extrapolation {
    /// Atoms outside the descriptor envelope (beyond what the model has learned).
    pub atoms_outside: usize,
    /// The model is untrusted (the trust rule's reason).
    pub untrusted: Option<String>,
    /// The rule that would have fallen back (envelope, variance, quarantine fence).
    pub reason: Option<String>,
    /// Per atom: its distance from the descriptor envelope in units of the limit (> 1 is outside).
    pub atom_envelope: Vec<f64>,
}

/// All zeros (an analytic provider fills `energy.total` and leaves the rest).
impl Default for EvalStats {
    fn default() -> EvalStats {
        EvalStats {
            source: Source::Dft,
            uncertainty: None,
            teacher_gap: None,
            refit_jump: None,
            model_forces: None,
            scf_steps: 0,
            readbacks: None,
            solve_ms: 0,
            warnings: Vec::new(),
            notes: Vec::new(),
            face_density: None,
            eig_tol_used: None,
            scf_tol_used: None,
            energy: EnergyBreakdown::default(),
            max_eigen_residual: 0.0,
            poisson_residual: 0.0,
            atom_uncertainty: None,
            extrapolation: None,
            pseudised: Vec::new(),
            core_overlaps: Vec::new(),
            fractional: Vec::new(),
            occupation_changed: false,
        }
    }
}

impl EvalStats {
    /// Why this answer must never be a training sample, if it must not (M7): the cores of two ions overlap
    /// ([`CORE_OVERLAP_WARNING`], spec §5.6 — the frozen-core approximation is outside its domain) or the
    /// shared-shell occupation pattern changed since the previous evaluation ([`OCCUPATION_CHANGE_WARNING`] —
    /// E(R) jumps by the orbital-polarisation energy there). The answer is still used for its step.
    pub fn untrainable(&self) -> Option<&'static str> {
        if !self.core_overlaps.is_empty() {
            Some(CORE_OVERLAP_WARNING)
        } else if self.occupation_changed {
            Some(OCCUPATION_CHANGE_WARNING)
        } else {
            None
        }
    }

    /// The surface `ForceResult::energy` is on (see [`Surface`]).
    pub fn energy_surface(&self) -> Surface {
        match self.source {
            Source::Dft => Surface::Teacher,
            Source::Surrogate => Surface::Model,
            Source::Fallback => if self.teacher_gap.is_some() { Surface::Model } else { Surface::Teacher },
        }
    }
}

/// A solver's own failure ([`ProviderError::Solver`]): `qm::QmError` variant for variant, with the
/// same messages (`providers::ks` maps one onto the other; `sim` does not depend on `qm`).
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum SolverError {
    /// Two nuclei closer than the coincidence threshold: the point-charge repulsion is undefined.
    #[error("nuclei {0} and {1} coincide")]
    CoincidentNuclei(usize, usize),
    /// The SCF loop hit its iteration cap with Δρ (the second field) still above tolerance.
    #[error("SCF did not converge after {0} iterations (Δρ = {1:.3e})")]
    ScfNotConverged(usize, f64),
    /// The SCF stall stop (`qm::QmError::ScfStalled`): the best Δρ did not improve 2× within
    /// `window` iterations, so the solve ended after `iterations`, before the cap.
    #[error("SCF stalled: Δρ best {best_delta_rho:.1e} not improved 2× in {window} iterations (target {tol:.0e}; stopped after {iterations} iterations)")]
    ScfStalled { iterations: usize, best_delta_rho: f64, tol: f64, window: usize },
    /// Non-finite or rank-deficient numerics inside a GPU solve; the message names the check.
    #[error("numerical failure in the solver: {0}")]
    Numerical(String),
    /// A GPU → CPU readback failed (the device was busy past a wait, or lost): "GPU busy". Raised
    /// by `qm::QmError::Device` after the provider's own retries ran out; the message names the
    /// cause and the attempts. Nothing is wrong with the scene — Play or any edit tries again.
    #[error("GPU busy, a readback from the device failed ({0})")]
    Device(String),
}

impl SolverError {
    /// How a failed SCF ended and after how many iterations (the solve log's iteration cell):
    /// `None` for an error that is not an SCF non-convergence.
    pub fn scf_end(&self) -> Option<(crate::activity::ScfEnd, usize)> {
        use crate::activity::ScfEnd;
        match *self {
            SolverError::ScfStalled { iterations, .. } => Some((ScfEnd::Stalled, iterations)),
            SolverError::ScfNotConverged(n, d) if d.is_finite() => Some((ScfEnd::Cap, n)),
            SolverError::ScfNotConverged(n, _) => Some((ScfEnd::NonFinite, n)),
            _ => None,
        }
    }
}

impl ProviderError {
    /// [`SolverError::scf_end`] of a solver error; `None` otherwise.
    pub fn scf_end(&self) -> Option<(crate::activity::ScfEnd, usize)> {
        match self {
            ProviderError::Solver(e) => e.scf_end(),
            _ => None,
        }
    }
}

/// Why an evaluation produced no forces (module doc, "Errors").
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("cancelled")]
    Cancelled,
    #[error("{0}")]
    Solver(#[from] SolverError),
    /// The scene cannot be posed as a Kohn-Sham problem (charge or multiplicity).
    #[error("{0}")]
    Invalid(String),
}

/// The empty scene's answer (module doc, "Nothing to solve"): `None` when the scene has nuclei
/// (there is something to solve); otherwise E = 0, no forces, no density — or
/// [`ProviderError::Invalid`] for a charge ≠ 0 or a multiplicity ≠ 1 with no nuclei.
pub fn nothing_to_solve(scene: &Scene) -> Option<Result<ForceResult, ProviderError>> {
    if !scene.nuclei.is_empty() {
        return None;
    }
    if scene.charge != 0 {
        return Some(Err(ProviderError::Invalid(format!(
            "charge {} with no nuclei: an empty scene has no protons and no electrons (place a nucleus, or set the charge to 0)",
            scene.charge
        ))));
    }
    if let Some(m) = scene.multiplicity.filter(|&m| m != 1) {
        return Some(Err(ProviderError::Invalid(format!("multiplicity {m} impossible for 0 electrons (an empty scene)"))));
    }
    Some(Ok(ForceResult { energy: 0.0, forces: Vec::new(), stats: EvalStats::default(), display: None, orbitals: None }))
}

pub trait ForceProvider: Send {
    /// Energy and forces for the scene's current nuclear positions. `should_abort` is polled at solver-step granularity.
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError>;
    fn name(&self) -> &'static str;
    /// Forget warm state: the next evaluation is cold (grid change, added or removed nuclei, a
    /// replaced scene).
    fn reset(&mut self);
    /// A teleport of existing nuclei (R13): the next evaluation may warm-start from the old
    /// geometry's state but must not extrapolate through the jump. Default: [`reset`](Self::reset).
    fn discontinuity(&mut self) {
        self.reset();
    }
    /// A counter that changes whenever the provider's energy surface changes for a reason other
    /// than the positions — a surrogate refit between ticks (`providers::surrogate`). Constant
    /// (0) for a provider whose surface is fixed. The driver re-evaluates at the current
    /// positions when it changes and books that evaluation's `refit_jump`, so `E_conserved`
    /// stays continuous and the next step's first half-kick already uses the new surface's
    /// forces.
    fn surface_epoch(&self) -> u64 {
        0
    }
    /// The provider as a [`Trainable`] — one that learns from a teacher (the surrogate) — for the
    /// driver's sampling pass and model commands and the worker's refits and readouts (module
    /// doc). `None` by default: a provider that cannot be taught. This is the only way past the
    /// boundary; nothing downcasts a provider.
    fn as_trainable(&mut self) -> Option<&mut dyn Trainable> {
        None
    }
    /// Side jobs (densities solved off the stepping path: the surrogate's background audits and
    /// display refreshes, [`SideDisplay`]) dispatched so far — monotone, 0 for a provider without
    /// any. The driver stamps each new job with the simulated time of the geometry it was
    /// dispatched at (`crate::dynamics`, "Side densities").
    fn side_jobs(&self) -> u64 {
        0
    }
    /// The newest density a side job produced since the last call (older ones are dropped).
    fn take_side_display(&mut self) -> Option<SideDisplay> {
        None
    }
    /// The latest converged density ρ↑ + ρ↓ (e/Bohr³) of this provider at arbitrary points (Bohr,
    /// trilinear interpolation on its grid; 0 outside), or `None` when the provider holds no density
    /// (a surrogate-answered step, an analytic provider, nothing solved yet). M6 spec §3.3: the
    /// bond detector's only view of the electrons — numbers at points, never a grid (rule 5).
    /// The Kohn-Sham adapters (`providers::ks`) answer `Some` iff their latest `evaluate`
    /// succeeded and neither [`reset`](Self::reset) nor [`discontinuity`](Self::discontinuity)
    /// was called since, so the density is always that of the geometry last evaluated.
    fn sample_density(&mut self, _points: &[[f64; 3]]) -> Option<Vec<f64>> {
        None
    }
    /// The SCF tolerance this provider solves at (the Kohn-Sham adapters' Δρ tolerance), or
    /// `None` for a provider without one (analytic; a surrogate answers for its teacher). M6
    /// spec §3.2: the isolated-atom references are keyed by the solver and its tolerance
    /// (`crate::references::ReferenceKey`), and the driver builds that key from here.
    fn scf_tol(&self) -> Option<f64> {
        None
    }
    /// A fresh, independent instance of this solver at the same settings, for work off the
    /// stepping path — the isolated-atom references (M6 spec §3.2): the driver solves them on it
    /// so the stepping provider's warm state is never touched and no step after a reference is
    /// cold. A surrogate answers with its teacher's. `None` for a provider that cannot (the
    /// driver then detects no bonds and says so); the analytic test providers clone themselves.
    fn side_solver(&self) -> Option<Box<dyn ForceProvider>> {
        None
    }
    /// The orbital view from now on (module doc, "Orbital display"). A change of `what` or
    /// `unoccupied` makes the next payload due regardless of the rate; `Off` drops the producer's
    /// display state (fields, textures, tracker). No-op by default: a provider without orbitals.
    fn set_orbital_view(&mut self, _request: OrbitalViewRequest) {}
    /// The orbital display of the provider's current converged state, **without an SCF solve**
    /// (the view switched on or changed while paused, or the state paused into): `None` when the
    /// view is off, nothing is converged at the geometry last evaluated (a reset, a teleport, a
    /// failed or cancelled evaluation since), or the provider has no orbitals here (analytic; a
    /// surrogate, whose producer is its side solver — the caller asks it for a display refresh
    /// instead). Counts as a payload for the rate bound. With unoccupied orbitals asked for it may
    /// run the display eigensolve (seconds); `should_abort` (Task 4 review I2: true when the worker
    /// has *any* command waiting) cancels it — the payload is still made, with the previous
    /// unoccupied rows stamped not current.
    fn orbital_display_now(&mut self, _should_abort: &mut dyn FnMut() -> bool) -> Option<OrbitalDisplay> {
        None
    }
    /// The interrupt that display-only work done inside an evaluation (the unoccupied-orbital
    /// eigensolve of a payload) polls to yield to the user ([`DisplayInterrupt`]; the driver
    /// installs it, `Dynamics::set_display_interrupt`). `evaluate`'s own `should_abort` is the
    /// stepping solve's (R11: only commands that make the step pointless) and is not used for
    /// display work. Default: ignored.
    fn set_display_interrupt(&mut self, _interrupt: DisplayInterrupt) {}
    /// M7: whether the density shown (`Display`) includes the partial core ρ_pc of the pseudised
    /// species — a display of the generator's core model, never solved and never fed back
    /// (spec §6.5). Applies from the next evaluation; the Kohn-Sham adapters honour it, a
    /// surrogate forwards it to its teacher. No-op by default.
    fn set_display_partial_core(&mut self, _on: bool) {}
    /// M7: the identity of the ionic surface this provider solves `scene` on — `"none"` when
    /// nothing in it is pseudised (the all-electron surface), else `"<policy word>/<recipe hash>"`
    /// (`providers::pseudo::provenance_tag`). Keys the isolated-atom references
    /// ([`crate::references::ReferenceKey::ionic`]): a reference solved on one surface is never
    /// reused on another (the valence energies differ by tens of Ha). The default — for the
    /// analytic providers, which have no pseudopotentials — is the policy's own word.
    fn ionic_tag(&self, scene: &Scene) -> String {
        scene.ionic.word().to_string()
    }
}

/// True when the worker has a command waiting — any command, not only R11's aborting ones (Task
/// 4 review I2): display-only work (the unoccupied-orbital eigensolve) yields to the user at once,
/// since it writes nothing a command could depend on. Shared across threads (the app's sender
/// counts what it sends, the worker what it applies).
pub type DisplayInterrupt = Arc<dyn Fn() -> bool + Send + Sync>;

/// What solved a [`SideDisplay`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SideKind {
    /// A background audit (`providers::surrogate`, rule 8): a teacher solve that also checks the
    /// model.
    Audit,
    /// A display-only refresh the app asked for (`Trainable::request_display`).
    Refresh,
}

/// A real Kohn-Sham density solved off the stepping path (M5b, "Density in estimated mode"): at
/// `positions`, by side job number `job` ([`ForceProvider::side_jobs`]). Display only — the step
/// it arrives with was answered by someone else. `orbitals`: the side solver's orbital payload
/// when an orbital view is on (module doc, "Orbital display": in estimated mode the only
/// producer).
#[derive(Clone, Debug)]
pub struct SideDisplay {
    pub display: Display,
    pub orbitals: Option<OrbitalDisplay>,
    pub kind: SideKind,
    pub job: u64,
    pub positions: Vec<[f64; 3]>,
}

/// A provider that learns from a teacher (M6 spec §9; the surrogate, `providers::surrogate`):
/// what the driver's sampling pass and model commands (`crate::dynamics`, "Sampling") and the
/// worker's refits, readouts and density requests need, reached through
/// [`ForceProvider::as_trainable`] — never by downcasting. Messages for the user that an
/// operation produces (a pass started or finished, a reference chosen, a model loaded, refused or
/// forgotten) are queued and handed out by [`take_notes`](Self::take_notes), in order.
pub trait Trainable {
    /// Plans a sampling pass over `scene` (module doc of `providers::sampling`); `refs` seeds its
    /// isolated-atom reference cache (a reference found there costs no solve). Err with the
    /// plan's problem (or when a pass is already running); nothing changes then.
    fn start_pass(&mut self, scene: &Scene, plan: &SamplingPlan, refs: &mut AtomReferences) -> Result<(), String>;
    /// Starts a **teach-around** pass (`providers::surrogate`, "Teaching around the blockers"): a
    /// targeted sampling pass around the training samples that keep an environment class untrusted
    /// (`TrainableStatus::teach_offer`), driven by `pass_step` like any pass. Err when there is
    /// nothing to teach or a pass is running. The default: unsupported.
    fn start_teach(&mut self, _scene: &Scene) -> Result<(), String> {
        Err("this provider cannot teach around blockers".into())
    }
    /// One teacher solve of the running pass: `Ok(Some(progress))` after it; `Ok(None)` when the
    /// pass has finished — it then refits over everything it holds and notes the outcome;
    /// `Err(Cancelled)` when `should_abort` abandoned the solve (the job stays queued); any other
    /// `Err` ends the pass (refitted when there are enough samples to fit, noted) with that error.
    fn pass_step(&mut self, should_abort: &mut dyn FnMut() -> bool) -> Result<Option<SamplingProgress>, ProviderError>;
    /// Ends a running pass early: what it sampled is kept and refitted once there are enough
    /// samples to fit (Q′; fewer are kept unfitted) — noted either way. No-op without one.
    fn cancel_pass(&mut self);
    /// A pass is running.
    fn pass_active(&self) -> bool;
    /// The running pass's progress, else the last pass's final state; `None` before any pass and
    /// after a forget or a load.
    fn progress(&self) -> Option<SamplingProgress>;
    /// Rebuilds the model from the samples held (between ticks, never inside `evaluate`): `Ok`
    /// with a one-line summary ("N samples, rmse_E … Ha/atom, rmse_F … Ha/Bohr"), `Err` with the
    /// fit's error (the previous model stays). A refit that moves the energy surface changes
    /// [`ForceProvider::surface_epoch`].
    fn refit(&mut self) -> Result<String, String>;
    /// Drops the model, its samples and references (a running pass is cancelled without a refit;
    /// the reference cache survives): the provider is a pass-through to its teacher again. Noted.
    fn forget(&mut self);
    /// Writes the full model file (the `.model` sidecar). `InvalidInput` with nothing fitted.
    fn save(&self, path: &std::path::Path) -> std::io::Result<()>;
    /// Reads a model file and installs it if it was taught on `scene`'s Kohn-Sham surface (grid,
    /// h, σ, charge, multiplicity, teacher and tolerance); `Err` with the message for the user
    /// otherwise (a provenance refusal is noted too) and nothing changes. Refused while a pass
    /// runs.
    fn load(&mut self, path: &std::path::Path, scene: &Scene) -> Result<(), String>;
    /// The teacher, for work the model must not see (the pass's solves, the reference atoms).
    fn teacher_mut(&mut self) -> &mut dyn ForceProvider;
    /// The readouts at `scene` (whose settings the model's provenance is compared with) and the
    /// latest `forces` (whose smallest magnitude sets the reported force bound).
    fn status(&self, scene: &Scene, forces: &[[f64; 3]]) -> TrainableStatus;
    /// A model has been fitted (`status().fitted`, without building the readouts).
    fn fitted(&self) -> bool;
    /// Training samples held (`status().samples`, without building the readouts).
    fn samples(&self) -> usize;
    /// A sample was kept since the last refit (`status().pending_refit`, without building the
    /// readouts).
    fn pending_refit(&self) -> bool;
    /// The audit cadence (answered steps per audit; 0 disables).
    fn set_audit_every(&mut self, n: usize);
    /// Draft mode: the model answers every step it can be asked at all, extrapolating (the
    /// surrogate's `FallbackPolicy::never_fallback`). A provider without one ignores it.
    fn set_draft(&mut self, _on: bool) {}
    /// The next `evaluate` goes to the teacher regardless of the model (the relax verification,
    /// M6 spec §6.2); the answer takes the normal fallback path and is kept as a sample.
    fn force_teacher_next(&mut self);
    /// Shrinks the running pass to `plan`'s counts (M6 spec §9, the automatic pass re-sized from
    /// the median of its first solves): only jobs not yet run are dropped, the references and
    /// the trust minimum are kept, and a pass never grows (`providers::sampling::Sampler::rescale`).
    /// No-op without a running pass, and by default (a provider without passes).
    fn rescale_pass(&mut self, _plan: &SamplingPlan) {}
    /// The queued messages for the user (trait doc), oldest first.
    fn take_notes(&mut self) -> Vec<String> {
        Vec::new()
    }
    /// The teaching solves since the last call, oldest first (`crate::activity::TeachEvent`; the
    /// Activity tab's solve log). A provider without a teacher has none.
    fn take_teach_events(&mut self) -> Vec<crate::activity::TeachEvent> {
        Vec::new()
    }
    /// The training views' data built from what the provider holds, without a teacher solve
    /// (`crate::activity::TrainingViews`); `None` without a fitted model.
    fn training_views(&mut self) -> Option<crate::activity::TrainingViews> {
        None
    }
    /// Asks for a display-only Kohn-Sham solve at `scene` off the stepping path; it arrives as a
    /// `SideKind::Refresh` [`SideDisplay`] through [`ForceProvider::take_side_display`]. False
    /// when the provider has no background teacher.
    fn request_display(&mut self, _scene: &Scene) -> bool {
        false
    }
    /// A background job (audit, display refresh, isolated atom or reference) is in flight or queued:
    /// the worker keeps ticking while paused so that its answer is collected without another
    /// command (final review I1). False without a background teacher.
    fn side_job_in_flight(&self) -> bool {
        false
    }
    /// Asks for the isolated-atom density of species `z` on `scene`'s cell off the stepping path,
    /// taken with [`take_atom_densities`](Self::take_atom_densities). False when the provider has
    /// no background teacher.
    fn request_atom_density(&mut self, _z: u8, _scene: &Scene) -> bool {
        false
    }
    /// The isolated-atom densities that arrived since the last call: (Z, its display).
    fn take_atom_densities(&mut self) -> Vec<(u8, Display)> {
        Vec::new()
    }
    /// Asks the background teacher for species `z`'s isolated-atom **reference** — the whole
    /// multiplicity scan of `crate::references::solve_atom` on `scene`'s cell and σ, the atom of
    /// mass `mass` — off the stepping path (M6 spec §3.2, Task 3 review I2: the stepping
    /// teacher is never used or reset for it), taken with
    /// [`take_atom_references`](Self::take_atom_references). It shares the background teacher's
    /// job slot with the audits and side densities. False when the provider has no background
    /// teacher (the driver then uses [`ForceProvider::side_solver`]).
    fn request_atom_reference(&mut self, _z: u8, _mass: f64, _scene: &Scene) -> bool {
        false
    }
    /// The isolated-atom references that came back since the last call: (Z, the scan's outcome).
    /// Every accepted request is answered exactly once — `Err(Cancelled)` when the background
    /// teacher abandoned it (a Forget, an installed model), another error when it failed or the
    /// background thread stopped.
    fn take_atom_references(&mut self) -> Vec<(u8, Result<AtomReference, ProviderError>)> {
        Vec::new()
    }
    /// The multiplicities done so far by the reference scan the background teacher is running
    /// (`None` when it runs none): the progress the driver shows.
    fn atom_reference_done(&self) -> Option<usize> {
        None
    }
}

/// What a [`Trainable`] reports for the readouts (the app's `ModelStatus` is built from it plus
/// the worker's own fields).
#[derive(Clone, Debug, PartialEq)]
pub struct TrainableStatus {
    /// The teacher's name.
    pub teacher: &'static str,
    /// A model has been fitted.
    pub fitted: bool,
    /// The trust rule passes.
    pub trusted: bool,
    /// Why not (a fitted model only).
    pub untrusted_reason: Option<String>,
    /// Training samples held.
    pub samples: usize,
    /// A sample was kept since the last refit.
    pub pending_refit: bool,
    /// The surface epoch ([`ForceProvider::surface_epoch`]).
    pub epoch: u64,
    /// The last fit: (samples, Q′, rmse E per atom, rmse F).
    pub fit: Option<(usize, usize, f64, f64)>,
    /// Worst training residuals: (energy per atom, force, that atom's force bound).
    pub residuals: Option<(f64, f64, f64)>,
    /// The per-component force bound at the smallest force of `status`'s `forces` (Ha/Bohr).
    pub force_bound: f64,
    /// The model was trained at other settings than the scene's: both, for the user.
    pub provenance_mismatch: Option<String>,
    /// The fitted model's provenance for the info line, with the recorded (not compared) mass
    /// set (`surrogate::Provenance::describe_recorded`); `None` without a model.
    pub provenance: Option<String>,
    /// Answered steps per audit (0: off).
    pub audit_every: usize,
    /// Draft mode is on.
    pub draft: bool,
    /// Samples held in quarantine.
    pub quarantined: usize,
    /// Quarantined samples a fresh answer contradicted (held, excluded, no longer fencing).
    pub superseded: usize,
    /// Box-affected samples excluded from the fit (`providers::surrogate`, "Box-affected answers").
    pub box_excluded: usize,
    /// Draft-mode audit answers at geometries strict mode refuses, held out of the strict fit
    /// (`providers::surrogate`, "Draft mode").
    pub held_out: usize,
    /// Stored samples whose teacher answer was box-affected, excluded from the fit yet or not
    /// (diag-3h: the Model tab names the box when it keeps the model untrusted).
    pub box_affected: usize,
    /// The sample count the box-affected samples' exclusion waits for (Q′ + `MIN_RESIDUAL_DOF`
    /// of the stored species' basis, the trust floor); `None` with an empty store.
    pub trust_floor: Option<usize>,
    /// Samples kept since the last refit (not in the fit yet). Right after a refit
    /// `samples` = fit samples + `quarantined` + `superseded` + `box_excluded` + `held_out` + `unfitted`
    /// exactly (M6 Task 11: the panel's "fit vs held" reconciliation).
    pub unfitted: usize,
    /// What the last refit's trust recovery did.
    pub recovery: Option<String>,
    /// The audits run on a background teacher.
    pub audits_background: bool,
    /// A background audit is in flight.
    pub audit_in_flight: bool,
    /// The last audit's outcome.
    pub last_audit: Option<AuditRecord>,
    /// `last_audit` judged the current fit. With `last_audit` this gives three states: `None`
    /// (never audited), `Some` and `false` (no audit since the last refit: the record is an
    /// earlier fit's), `Some` and `true` (activity Task 3 review I1).
    pub last_audit_this_fit: bool,
    /// A trusted model that keeps falling back says why.
    pub stuck: Option<String>,
    /// Per environment class, the fit's leave-one-out errors and whether the class is trusted
    /// (`providers::surrogate`, "Class-wise trust"; M6 Task 11 round 6).
    pub class_loo: Vec<ClassLoo>,
    /// The model's lifetime teaching solves (`crate::activity::SolveCounts`).
    pub solves: crate::activity::SolveCounts,
    /// The trust rule's conditions in its order; empty before the first fit.
    pub trust_checks: Vec<crate::activity::TrustCheck>,
    /// `stuck` in structured form (the plain wording's input).
    pub stuck_info: Option<crate::activity::StuckInfo>,
    /// The model generation (`TrainingViews::model_gen`): changes at every installed, restored or
    /// forgotten model and every policy change, also where the surface epoch does not (an
    /// untrusted refit). The app keys its training-view rebuilds on it (activity Task 5).
    pub model_gen: u64,
    /// Whether the strict policy answers at the scene's current geometry, and if not the rule that
    /// refuses and why ([`Readiness`]; `None` without a model or nuclei). The `trusted` flag is the
    /// global trust rule only; this is what "trusted" must mean to a user — "strict mode will answer
    /// here" (the 2026-10-18 water investigation: a model shown trusted fell back on every step).
    pub readiness: Option<Readiness>,
    /// What a teach-around pass would do now (`providers::surrogate`, "Teaching around the
    /// blockers"): `None` when no class is blocked by teachable samples.
    pub teach_offer: Option<TeachOffer>,
}

/// A teach-around offer (`TrainableStatus::teach_offer`).
#[derive(Clone, Debug, PartialEq)]
pub struct TeachOffer {
    /// The blockers it would teach around: (store index, class name, leave-one-out ratio,
    /// in-sample ratio) — leverage outliers (in-sample ≤ 1, LOO > 1), at most five.
    pub blockers: Vec<(usize, &'static str, f64, f64)>,
    /// Blockers it would not teach: the model cannot represent them (in-sample > 1), or two rounds
    /// were spent on them already — (store index, class name, in-sample ratio, why); the worst
    /// five by leave-one-out.
    pub refused: Vec<(usize, &'static str, f64, &'static str)>,
    /// How many blockers were refused in all (`refused` lists at most five).
    pub refused_total: usize,
    /// The teacher solves the pass would make (the held-out checks included).
    pub solves: usize,
    /// What the offer says before it runs (the teach-around review, SHOULD-T3/T5): the cost and
    /// the risk (a failed check untrusts the model until re-earned), the sample floor a richer
    /// basis would set, and a pair blocker's pair against r_cut. Informational: decides nothing.
    pub notes: Vec<String>,
    /// Set when consecutive class-rule fallbacks are blocked by a sample far from the geometry
    /// being asked (`providers::surrogate`, `teach_prompt`): the status chip and a card offer the
    /// pass in these words. Never runs anything by itself.
    pub prompt: Option<String>,
}

/// The strict fallback policy's verdict at the current geometry ([`TrainableStatus::readiness`];
/// `providers::surrogate`, "Readiness"): whether the model would answer a step there with draft
/// mode off, the rule that refuses (the activity code the step's event would carry) and the
/// refusal in words — for the class rule (4b) with the class's worst leave-one-out sample named.
#[derive(Clone, Debug, PartialEq)]
pub struct Readiness {
    /// The strict policy answers here.
    pub answers: bool,
    /// The refusing rule (`None` when it answers).
    pub rule: Option<crate::activity::TeachRule>,
    /// The atom the rule names, if any.
    pub atom: Option<usize>,
    /// "strict mode answers here", or the refusal's text without the "surrogate: " prefix.
    pub text: String,
    /// The scene's own sampling pass is not in the fit: none of the last pass's samples of this
    /// composition survived into the fit that stands (box-excluded, quarantined …), in words —
    /// the model has not learned this scene, whatever the policy says about this one geometry.
    pub unlearned: Option<String>,
}

/// One environment class's leave-one-out numbers (`TrainableStatus::class_loo`): the class
/// ("pair": every atom has at most one neighbour within r_cut; "many-body": some atom has two or
/// more), its samples, the LOO per-atom force error over its atoms as a ratio to each atom's
/// force bound (RMS and max), the largest LOO per-atom energy error as a ratio to the energy
/// tier, and whether the class is trusted.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassLoo {
    pub class: &'static str,
    pub samples: usize,
    pub force_rms: f64,
    pub force_max: f64,
    pub energy_max: f64,
    pub trusted: bool,
    /// The class rule's verdict in structured form: (the first failing condition, its value, its
    /// limit), `None` when trusted. `trusted == refusal.is_none()` always (one function decides
    /// both; activity plan review M3). Only the first failing condition is carried: a Force
    /// refusal may have an energy co-failure too, which `energy_max > 1` shows (the Log's text
    /// names both).
    pub refusal: Option<(ClassKey, f64, f64)>,
}

/// The condition that keeps an environment class untrusted ([`ClassLoo::refusal`]), in the order
/// the class rule tests them (`providers::surrogate`, "Class-wise trust").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClassKey {
    /// Fewer samples of the class than Q′: (samples, Q′).
    Samples,
    /// The leave-one-out force error over the class's atoms: the larger of max and RMS, as a
    /// ratio to each atom's bound (limit 1; NaN when either ratio is NaN).
    Force,
    /// The largest leave-one-out per-atom energy error, as a ratio to the tier (limit 1).
    Energy,
}

/// What an audit concluded (`providers::surrogate` module doc, "Audit").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuditVerdict {
    /// |gap|/N within `energy_std_per_atom`.
    Passed,
    /// Beyond it: the model is untrusted until the next refit.
    Failed,
    /// The teacher's answer is box-affected (face density above
    /// `providers::surrogate::box_affected_face_density`):
    /// its gap carries the box's confinement, which a translation-invariant model cannot and
    /// must not learn — recorded, not judged, not kept.
    BoxAffected,
}

/// The last audit's outcome (readouts).
#[derive(Clone, Debug, PartialEq)]
pub struct AuditRecord {
    /// E_teacher − E_model at the audited geometry (Hartree; the model current when the answer
    /// was judged).
    pub gap: f64,
    pub verdict: AuditVerdict,
    /// The teacher's face density (e/Bohr³), when its solver reports one.
    pub face_density: Option<f64>,
    /// Answered steps between the audit's dispatch and its verdict (0 for a synchronous audit):
    /// after a failed asynchronous audit these steps were answered by an unaudited model.
    pub unaudited_steps: usize,
    /// Run on the background auditor (ruling S14) rather than in the step.
    pub asynchronous: bool,
    /// The teacher solve's wall time (ms).
    pub teacher_ms: f64,
}

#[cfg(test)]
mod ionic_tests {
    use super::*;

    /// M7 Task 9 (spec §5.6; Task 6 review A.6): an answer with overlapping cores or a changed shared-shell pattern
    /// is untrainable and says why; an ordinary one is not.
    #[test]
    fn untrainable_answers_say_why() {
        let mut s = EvalStats::default();
        assert_eq!(s.untrainable(), None);
        s.occupation_changed = true;
        assert_eq!(s.untrainable(), Some(OCCUPATION_CHANGE_WARNING));
        s.core_overlaps = vec![(0, 1, 0.8)];
        assert_eq!(s.untrainable(), Some(CORE_OVERLAP_WARNING), "overlapping cores are named first");
        assert_eq!(CORE_OVERLAP_WARNING, "cores overlap");
    }
}
