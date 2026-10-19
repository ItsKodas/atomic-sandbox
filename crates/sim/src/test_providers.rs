//! Analytic [`ForceProvider`]s for tests and the `validate` dynamics suite — none of them touches
//! a grid or a solver, so integrator and driver behaviour can be pinned against closed forms.
//! `pub` because Tasks 5 and 7 (the `Dynamics` driver tests, `validate/tests/dynamics.rs`)
//! reuse them; nothing here is used by the application itself.
use crate::provider::{Display, EvalStats, ForceProvider, ForceResult, OrbitalDisplay, OrbitalViewRequest, ProviderError, SideDisplay, SolverError, Trainable, TrainableStatus};
use crate::references::AtomReferences;
use crate::sampling::{SamplingPlan, SamplingProgress};
use crate::scene::{neumaier, Scene};
use std::sync::{Arc, Mutex};

mod gaussians;
pub use gaussians::{Gaussians, GriddedGaussians, Term};

/// Independent isotropic wells: F_i = −k (r_i − r₀ᵢ), E = ½ k Σ|r_i − r₀ᵢ|² (Hartree, Bohr).
/// Each nucleus has its own centre, so the force on one never depends on another; the exact
/// motion is a sinusoid at ω_i = √(k/m_i). Never errors except for a centre-count mismatch
/// (`ProviderError::Invalid`). `stats` is `EvalStats::default()` with `energy.total` filled.
#[derive(Clone, Debug)]
pub struct Harmonic {
    pub k: f64,
    pub centres: Vec<[f64; 3]>,
}

impl Harmonic {
    /// Wells at the scene's current positions (every nucleus starts at its own minimum).
    pub fn at_rest_positions(scene: &Scene, k: f64) -> Harmonic {
        Harmonic { k, centres: scene.nuclei.iter().map(|n| n.pos).collect() }
    }
}

impl ForceProvider for Harmonic {
    fn evaluate(&mut self, scene: &Scene, _should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        if self.centres.len() != scene.nuclei.len() {
            return Err(ProviderError::Invalid(format!("harmonic: {} centres for {} nuclei", self.centres.len(), scene.nuclei.len())));
        }
        let k = self.k;
        let forces: Vec<[f64; 3]> = scene.nuclei.iter().zip(&self.centres)
            .map(|(n, c)| std::array::from_fn(|a| -k * (n.pos[a] - c[a])))
            .collect();
        let energy = 0.5 * k * neumaier(scene.nuclei.iter().zip(&self.centres).map(|(n, c)| {
            let d: [f64; 3] = std::array::from_fn(|a| n.pos[a] - c[a]);
            d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
        }));
        let mut stats = EvalStats::default();
        stats.energy.total = energy;
        Ok(ForceResult { energy, forces, stats, display: None, orbitals: None })
    }

    fn name(&self) -> &'static str {
        "harmonic"
    }

    fn reset(&mut self) {}
}

/// No forces, zero energy: free motion (plus whatever the integrator's external field and the
/// caller's `augment` add).
#[derive(Clone, Copy, Debug, Default)]
pub struct Zero;

impl ForceProvider for Zero {
    fn evaluate(&mut self, scene: &Scene, _should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        Ok(ForceResult { energy: 0.0, forces: vec![[0.0; 3]; scene.nuclei.len()], stats: EvalStats::default(), display: None, orbitals: None })
    }

    fn name(&self) -> &'static str {
        "zero"
    }

    fn reset(&mut self) {}
}

/// Which error [`Failing`] injects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailKind {
    /// `ProviderError::Cancelled` — what a GPU solve returns when `should_abort` fired.
    Cancelled,
    /// `ProviderError::Invalid("injected failure …")` — an unposable scene.
    Invalid,
    /// `ProviderError::Solver(SolverError::Numerical("injected failure …"))` — a GPU numerical failure.
    Numerical,
}

/// Wraps a provider and fails exactly on evaluation number `fail_at_call` (1-based, counted
/// over this wrapper's lifetime in `calls`) with the error `kind` selects (default
/// [`FailKind::Invalid`]). Every other call is forwarded. For the "provider failure must not
/// corrupt the scene" contracts of `Integrator::step` and the `Dynamics` driver.
pub struct Failing {
    pub inner: Box<dyn ForceProvider>,
    pub fail_at_call: usize,
    pub calls: usize,
    pub kind: FailKind,
}

impl Failing {
    pub fn new(inner: impl ForceProvider + 'static, fail_at_call: usize) -> Failing {
        Failing { inner: Box::new(inner), fail_at_call, calls: 0, kind: FailKind::Invalid }
    }

    pub fn with_kind(self, kind: FailKind) -> Failing {
        Failing { kind, ..self }
    }
}

impl ForceProvider for Failing {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        self.calls += 1;
        if self.calls == self.fail_at_call {
            let msg = format!("injected failure at call {}", self.calls);
            return Err(match self.kind {
                FailKind::Cancelled => ProviderError::Cancelled,
                FailKind::Invalid => ProviderError::Invalid(msg),
                FailKind::Numerical => ProviderError::Solver(SolverError::Numerical(msg)),
            });
        }
        self.inner.evaluate(scene, should_abort)
    }

    fn name(&self) -> &'static str {
        "failing"
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// Morse dimer: E = 2 e_atom + D[(1 − e^{−a(r − r_e)})² − 1] for exactly two nuclei (any Z;
/// `ProviderError::Invalid` otherwise), with the analytic forces F₁ = +(dE/dr) û, F₂ = −(dE/dr) û,
/// û from nucleus 0 to nucleus 1 — translation and rotation invariant, so the surrogate can
/// represent it (unlike [`Harmonic`], whose wells are tied to space). `h2()` is the Task 3
/// fixture (D = 0.17 Ha, a = 1 Bohr⁻¹, r_e = 1.4 Bohr, e_atom = −0.478 Ha: an H₂-like well).
/// `calls` counts evaluations; wrap in [`Shared`] to read or retune it after boxing.
#[derive(Clone, Debug)]
pub struct MorseDimer {
    pub d: f64,
    pub a: f64,
    pub r_e: f64,
    pub e_atom: f64,
    pub calls: usize,
}

impl MorseDimer {
    pub fn h2() -> MorseDimer {
        MorseDimer { d: 0.17, a: 1.0, r_e: 1.4, e_atom: -0.478, calls: 0 }
    }

    /// (E(r), dE/dr).
    pub fn energy_and_derivative(&self, r: f64) -> (f64, f64) {
        let x = (-self.a * (r - self.r_e)).exp();
        let e = 2.0 * self.e_atom + self.d * ((1.0 - x) * (1.0 - x) - 1.0);
        let de = 2.0 * self.d * self.a * (1.0 - x) * x;
        (e, de)
    }

    /// Energy and forces of a two-nucleus geometry (what `evaluate` returns, without the count).
    pub fn energy_and_forces(&self, pos: &[[f64; 3]]) -> Result<(f64, Vec<[f64; 3]>), ProviderError> {
        if pos.len() != 2 {
            return Err(ProviderError::Invalid(format!("morse dimer: {} nuclei, need 2", pos.len())));
        }
        let d: [f64; 3] = std::array::from_fn(|a| pos[1][a] - pos[0][a]);
        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if r == 0.0 {
            return Err(ProviderError::Invalid("morse dimer: coincident nuclei".into()));
        }
        let (e, de) = self.energy_and_derivative(r);
        let u = d.map(|x| x / r);
        Ok((e, vec![u.map(|x| de * x), u.map(|x| -de * x)]))
    }
}

impl ForceProvider for MorseDimer {
    fn evaluate(&mut self, scene: &Scene, _should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        self.calls += 1;
        let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        let (energy, forces) = self.energy_and_forces(&pos)?;
        let mut stats = EvalStats::default();
        stats.energy.total = energy;
        Ok(ForceResult { energy, forces, stats, display: None, orbitals: None })
    }

    fn name(&self) -> &'static str {
        "morse dimer"
    }

    fn reset(&mut self) {}
}

/// A provider behind an `Arc<Mutex<_>>`: the test keeps a clone of the handle and can read or
/// retune the provider after it was boxed into another one (a surrogate's teacher). Every
/// trait method forwards under the lock.
pub struct Shared<P>(pub Arc<Mutex<P>>);

impl<P> Shared<P> {
    pub fn new(p: P) -> Shared<P> {
        Shared(Arc::new(Mutex::new(p)))
    }

    pub fn handle(&self) -> Arc<Mutex<P>> {
        self.0.clone()
    }
}

impl<P: ForceProvider> ForceProvider for Shared<P> {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).evaluate(scene, should_abort)
    }

    fn name(&self) -> &'static str {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).name()
    }

    fn reset(&mut self) {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).reset()
    }

    fn discontinuity(&mut self) {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).discontinuity()
    }

    fn sample_density(&mut self, points: &[[f64; 3]]) -> Option<Vec<f64>> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).sample_density(points)
    }

    fn scf_tol(&self) -> Option<f64> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).scf_tol()
    }

    fn side_solver(&self) -> Option<Box<dyn ForceProvider>> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).side_solver()
    }

    fn set_orbital_view(&mut self, request: OrbitalViewRequest) {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).set_orbital_view(request)
    }

    fn orbital_display_now(&mut self, a: &mut dyn FnMut() -> bool) -> Option<OrbitalDisplay> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).orbital_display_now(a)
    }

    fn set_display_interrupt(&mut self, interrupt: crate::provider::DisplayInterrupt) {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).set_display_interrupt(interrupt)
    }
}

/// A [`Trainable`] provider behind an `Arc<Mutex<_>>` — [`Shared`] for the surrogate: the test
/// keeps a clone of the handle and reads or drives the provider (its own methods, beyond the
/// trait) after it was boxed into a `Dynamics`, which reaches it through `as_trainable` as it
/// would the provider itself. Every `ForceProvider` and `Trainable` method forwards under the
/// lock; `teacher_mut` lends a proxy that forwards to the inner provider's teacher under the
/// same lock. (No downcast: the handle is how a test sees the concrete type.) Never hold a guard
/// of the handle across a call into the `Dynamics` that owns the wrapper — the lock is not
/// re-entrant.
pub struct SharedTrainable<P> {
    inner: Arc<Mutex<P>>,
    teacher: TeacherOf<P>,
}

/// The teacher of a [`SharedTrainable`]'s provider, reached under its lock.
pub struct TeacherOf<P>(Arc<Mutex<P>>);

impl<P> SharedTrainable<P> {
    pub fn new(p: P) -> SharedTrainable<P> {
        let inner = Arc::new(Mutex::new(p));
        SharedTrainable { teacher: TeacherOf(inner.clone()), inner }
    }

    pub fn handle(&self) -> Arc<Mutex<P>> {
        self.inner.clone()
    }

    /// A second wrapper over the same provider (a second `Dynamics` on one model; never drive
    /// both at once — the lock is not re-entrant across them either).
    pub fn from_handle(inner: Arc<Mutex<P>>) -> SharedTrainable<P> {
        SharedTrainable { teacher: TeacherOf(inner.clone()), inner }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, P> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl<P: ForceProvider + Trainable> ForceProvider for SharedTrainable<P> {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        self.lock().evaluate(scene, should_abort)
    }

    fn name(&self) -> &'static str {
        self.lock().name()
    }

    fn reset(&mut self) {
        self.lock().reset()
    }

    fn discontinuity(&mut self) {
        self.lock().discontinuity()
    }

    fn sample_density(&mut self, points: &[[f64; 3]]) -> Option<Vec<f64>> {
        self.lock().sample_density(points)
    }

    fn scf_tol(&self) -> Option<f64> {
        self.lock().scf_tol()
    }

    fn side_solver(&self) -> Option<Box<dyn ForceProvider>> {
        self.lock().side_solver()
    }

    fn surface_epoch(&self) -> u64 {
        self.lock().surface_epoch()
    }

    fn as_trainable(&mut self) -> Option<&mut dyn Trainable> {
        Some(self)
    }

    fn side_jobs(&self) -> u64 {
        self.lock().side_jobs()
    }

    fn take_side_display(&mut self) -> Option<SideDisplay> {
        self.lock().take_side_display()
    }

    fn set_orbital_view(&mut self, request: OrbitalViewRequest) {
        self.lock().set_orbital_view(request)
    }

    fn orbital_display_now(&mut self, a: &mut dyn FnMut() -> bool) -> Option<OrbitalDisplay> {
        self.lock().orbital_display_now(a)
    }

    fn set_display_interrupt(&mut self, interrupt: crate::provider::DisplayInterrupt) {
        self.lock().set_display_interrupt(interrupt)
    }
}

impl<P: ForceProvider + Trainable> Trainable for SharedTrainable<P> {
    fn start_pass(&mut self, scene: &Scene, plan: &SamplingPlan, refs: &mut AtomReferences) -> Result<(), String> {
        self.lock().start_pass(scene, plan, refs)
    }

    fn start_teach(&mut self, scene: &Scene) -> Result<(), String> {
        self.lock().start_teach(scene)
    }

    fn pass_step(&mut self, should_abort: &mut dyn FnMut() -> bool) -> Result<Option<SamplingProgress>, ProviderError> {
        self.lock().pass_step(should_abort)
    }

    fn cancel_pass(&mut self) {
        self.lock().cancel_pass()
    }

    fn pass_active(&self) -> bool {
        self.lock().pass_active()
    }

    fn progress(&self) -> Option<SamplingProgress> {
        self.lock().progress()
    }

    fn pass_ghost(&self) -> Option<crate::sampling::PassGhost> {
        self.lock().pass_ghost()
    }

    fn refit(&mut self) -> Result<String, String> {
        Trainable::refit(&mut *self.lock())
    }

    fn forget(&mut self) {
        Trainable::forget(&mut *self.lock())
    }

    fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        self.lock().save(path)
    }

    fn load(&mut self, path: &std::path::Path, scene: &Scene) -> Result<(), String> {
        self.lock().load(path, scene)
    }

    fn teacher_mut(&mut self) -> &mut dyn ForceProvider {
        &mut self.teacher
    }

    fn status(&self, scene: &Scene, forces: &[[f64; 3]]) -> TrainableStatus {
        self.lock().status(scene, forces)
    }

    fn fitted(&self) -> bool {
        Trainable::fitted(&*self.lock())
    }

    fn samples(&self) -> usize {
        Trainable::samples(&*self.lock())
    }

    fn pending_refit(&self) -> bool {
        Trainable::pending_refit(&*self.lock())
    }

    fn set_audit_every(&mut self, n: usize) {
        self.lock().set_audit_every(n)
    }

    fn set_draft(&mut self, on: bool) {
        self.lock().set_draft(on)
    }

    fn rescale_pass(&mut self, plan: &SamplingPlan) {
        self.lock().rescale_pass(plan)
    }

    fn force_teacher_next(&mut self) {
        Trainable::force_teacher_next(&mut *self.lock())
    }

    fn take_notes(&mut self) -> Vec<String> {
        self.lock().take_notes()
    }

    fn take_teach_events(&mut self) -> Vec<crate::activity::TeachEvent> {
        Trainable::take_teach_events(&mut *self.lock())
    }

    fn training_views(&mut self) -> Option<crate::activity::TrainingViews> {
        Trainable::training_views(&mut *self.lock())
    }

    fn request_display(&mut self, scene: &Scene) -> bool {
        Trainable::request_display(&mut *self.lock(), scene)
    }

    fn request_atom_density(&mut self, z: u8, scene: &Scene) -> bool {
        Trainable::request_atom_density(&mut *self.lock(), z, scene)
    }

    fn take_atom_densities(&mut self) -> Vec<(u8, Display)> {
        Trainable::take_atom_densities(&mut *self.lock())
    }

    fn request_atom_reference(&mut self, z: u8, mass: f64, scene: &Scene) -> bool {
        Trainable::request_atom_reference(&mut *self.lock(), z, mass, scene)
    }

    fn take_atom_references(&mut self) -> Vec<(u8, Result<crate::references::AtomReference, ProviderError>)> {
        Trainable::take_atom_references(&mut *self.lock())
    }

    fn atom_reference_done(&self) -> Option<usize> {
        Trainable::atom_reference_done(&*self.lock())
    }
}

impl<P: ForceProvider + Trainable> TeacherOf<P> {
    fn with<R>(&self, f: impl FnOnce(&mut dyn ForceProvider) -> R) -> R {
        let mut p = self.0.lock().unwrap_or_else(|e| e.into_inner());
        f(p.teacher_mut())
    }
}

impl<P: ForceProvider + Trainable> ForceProvider for TeacherOf<P> {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        self.with(|t| t.evaluate(scene, should_abort))
    }

    fn name(&self) -> &'static str {
        self.with(|t| t.name())
    }

    fn reset(&mut self) {
        self.with(|t| t.reset())
    }

    fn discontinuity(&mut self) {
        self.with(|t| t.discontinuity())
    }

    fn sample_density(&mut self, points: &[[f64; 3]]) -> Option<Vec<f64>> {
        self.with(|t| t.sample_density(points))
    }

    fn scf_tol(&self) -> Option<f64> {
        self.with(|t| t.scf_tol())
    }

    fn side_solver(&self) -> Option<Box<dyn ForceProvider>> {
        self.with(|t| t.side_solver())
    }

    fn surface_epoch(&self) -> u64 {
        self.with(|t| t.surface_epoch())
    }

    fn side_jobs(&self) -> u64 {
        self.with(|t| t.side_jobs())
    }

    fn take_side_display(&mut self) -> Option<SideDisplay> {
        self.with(|t| t.take_side_display())
    }

    fn set_orbital_view(&mut self, request: OrbitalViewRequest) {
        self.with(|t| t.set_orbital_view(request))
    }

    fn orbital_display_now(&mut self, a: &mut dyn FnMut() -> bool) -> Option<OrbitalDisplay> {
        self.with(|t| t.orbital_display_now(a))
    }

    fn set_display_interrupt(&mut self, interrupt: crate::provider::DisplayInterrupt) {
        self.with(|t| t.set_display_interrupt(interrupt))
    }
}
