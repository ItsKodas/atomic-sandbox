//! Nuclear dynamics (M4): the scene, the integrator, the thermostat, the driver and the relax
//! tool, behind the force boundary ([`ForceProvider`], with [`Trainable`] for a provider that
//! learns). The providers themselves — Kohn-Sham and surrogate — live in the `providers` crate
//! (M6, Task 1); `sim` depends on `units` and `gpu` (the display texture handle) only.
pub mod dynamics;
pub mod integrator;
pub mod provider;
pub mod references;
pub mod relax;
pub mod sampling;
pub mod scene;
pub mod test_providers;
pub mod thermostat;

pub use dynamics::{DynCommand, DynSnapshot, Dynamics, SideDensity, Walls};
pub use integrator::{Integrator, StepStats};
pub use relax::{Fire, FireOutcome, FireParams, FireStep, RelaxProgress, F_TOL};
pub use provider::{
    AuditRecord, AuditVerdict, Display, EnergyBreakdown, EvalStats, Extrapolation, ForceProvider, ForceResult, ProviderError, SideDisplay,
    SideKind, SolverError, Source, Surface, Trainable, TrainableStatus,
};
pub use references::{AtomReference, AtomReferences, ReferenceKey};
pub use sampling::{allowed_multiplicities, ReferenceCache, SamplingPlan, SamplingProgress};
pub use scene::{Cell, Nucleus, Scene, CELL_MARGIN_H};
pub use test_providers::{FailKind, Failing, Harmonic, MorseDimer, Shared, SharedTrainable, Zero};
pub use thermostat::Csvr;
