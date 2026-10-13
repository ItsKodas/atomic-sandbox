//! Nuclear dynamics (M4).
pub mod dynamics;
pub mod integrator;
pub mod provider;
pub mod sampling;
pub mod scene;
pub mod surrogate_ks;
pub mod test_providers;
pub mod thermostat;

pub use dynamics::{DynCommand, DynSnapshot, Dynamics, SideDensity, Walls};
pub use integrator::{Integrator, StepStats};
pub use provider::{CpuKs, Display, EvalStats, ForceProvider, ForceResult, GpuKs, ProviderError, SideDisplay, SideKind, Source, Surface};
pub use sampling::{run_sampling, PairScan, ReferenceCache, Sampler, SamplingPlan, SamplingProgress};
pub use scene::{Cell, Nucleus, Scene, CELL_MARGIN_H};
pub use surrogate_ks::{FallbackPolicy, SurrogateKs};
pub use test_providers::{FailKind, Failing, Harmonic, MorseDimer, Shared, Zero};
pub use thermostat::Csvr;
