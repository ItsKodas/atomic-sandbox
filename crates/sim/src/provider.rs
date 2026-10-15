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
/// plus the density for the viewer when the provider makes one (module doc, "Display payload").
#[derive(Clone, Debug)]
pub struct ForceResult {
    pub energy: f64,
    pub forces: Vec<[f64; 3]>,
    pub stats: EvalStats,
    pub display: Option<Display>,
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
/// - `solve_ms`: wall of the evaluation — solve, forces and the display copy — **excluding**
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
            energy: EnergyBreakdown::default(),
            max_eigen_residual: 0.0,
            poisson_residual: 0.0,
            atom_uncertainty: None,
            extrapolation: None,
        }
    }
}

impl EvalStats {
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
    /// Non-finite or rank-deficient numerics inside a GPU solve; the message names the check.
    #[error("numerical failure in the solver: {0}")]
    Numerical(String),
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
    Some(Ok(ForceResult { energy: 0.0, forces: Vec::new(), stats: EvalStats::default(), display: None }))
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
}

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
/// it arrives with was answered by someone else.
#[derive(Clone, Debug)]
pub struct SideDisplay {
    pub display: Display,
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
    /// Asks for a display-only Kohn-Sham solve at `scene` off the stepping path; it arrives as a
    /// `SideKind::Refresh` [`SideDisplay`] through [`ForceProvider::take_side_display`]. False
    /// when the provider has no background teacher.
    fn request_display(&mut self, _scene: &Scene) -> bool {
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
    /// Stored samples whose teacher answer was box-affected, excluded from the fit yet or not
    /// (diag-3h: the Model tab names the box when it keeps the model untrusted).
    pub box_affected: usize,
    /// The sample count the box-affected samples' exclusion waits for (Q′ + `MIN_RESIDUAL_DOF`
    /// of the stored species' basis, the trust floor); `None` with an empty store.
    pub trust_floor: Option<usize>,
    /// Samples kept since the last refit (not in the fit yet). Right after a refit
    /// `samples` = fit samples + `quarantined` + `superseded` + `box_excluded` + `unfitted`
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
    /// A trusted model that keeps falling back says why.
    pub stuck: Option<String>,
    /// Per environment class, the fit's leave-one-out errors and whether the class is trusted
    /// (`providers::surrogate`, "Class-wise trust"; M6 Task 11 round 6).
    pub class_loo: Vec<ClassLoo>,
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
