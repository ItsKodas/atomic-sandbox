//! Analytic [`ForceProvider`]s for tests and the `validate` dynamics suite — none of them touches
//! a grid or a solver, so integrator and driver behaviour can be pinned against closed forms.
//! `pub` because Tasks 5 and 7 (the `Dynamics` driver tests, `validate/tests/dynamics.rs`)
//! reuse them; nothing here is used by the application itself.
use crate::provider::{EvalStats, ForceProvider, ForceResult, ProviderError};
use crate::scene::{neumaier, Scene};
use std::sync::{Arc, Mutex};

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
        Ok(ForceResult { energy, forces, stats, display: None })
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
        Ok(ForceResult { energy: 0.0, forces: vec![[0.0; 3]; scene.nuclei.len()], stats: EvalStats::default(), display: None })
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
    /// `ProviderError::Solver(QmError::Numerical("injected failure …"))` — a GPU numerical failure.
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
                FailKind::Numerical => ProviderError::Solver(qm::QmError::Numerical(msg)),
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
        Ok(ForceResult { energy, forces, stats, display: None })
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
}
