//! Analytic [`ForceProvider`]s for tests and the `validate` dynamics suite — none of them touches
//! a grid or a solver, so integrator and driver behaviour can be pinned against closed forms.
//! `pub` because Tasks 5 and 7 (the `Dynamics` driver tests, `validate/tests/dynamics.rs`)
//! reuse them; nothing here is used by the application itself.
use crate::provider::{EvalStats, ForceProvider, ForceResult, ProviderError};
use crate::scene::{neumaier, Scene};

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

/// Wraps a provider and fails exactly on evaluation number `fail_at_call` (1-based, counted
/// over this wrapper's lifetime in `calls`): `ProviderError::Cancelled` when `cancelled`, else
/// `ProviderError::Invalid("injected failure …")`. Every other call is forwarded. For the
/// "provider failure must not corrupt the scene" contracts of `Integrator::step` and the
/// `Dynamics` driver.
pub struct Failing {
    pub inner: Box<dyn ForceProvider>,
    pub fail_at_call: usize,
    pub calls: usize,
    pub cancelled: bool,
}

impl Failing {
    pub fn new(inner: impl ForceProvider + 'static, fail_at_call: usize) -> Failing {
        Failing { inner: Box::new(inner), fail_at_call, calls: 0, cancelled: false }
    }
}

impl ForceProvider for Failing {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        self.calls += 1;
        if self.calls == self.fail_at_call {
            return Err(if self.cancelled { ProviderError::Cancelled } else { ProviderError::Invalid(format!("injected failure at call {}", self.calls)) });
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
