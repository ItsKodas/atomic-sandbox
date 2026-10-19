//! Nuclear dynamics (M4): the scene, the integrator, the thermostat, the driver and the relax
//! tool, behind the force boundary ([`ForceProvider`], with [`Trainable`] for a provider that
//! learns). The providers themselves — Kohn-Sham and surrogate — live in the `providers` crate
//! (M6, Task 1); `sim` depends on `units` and `gpu` (the display texture handle) only. The bond
//! detector ([`bonds`]) and the isolated-atom references ([`references`]) it reads the density
//! against are M6 Task 3.
pub mod activity;
pub mod bonds;
pub mod charges;
pub mod dynamics;
pub mod integrator;
pub mod provider;
pub mod references;
pub mod relax;
pub mod sampling;
pub mod scene;
pub mod test_providers;
pub mod thermostat;

pub use activity::{ActivityProbe, ProbeView, ScfEnd, ScfIteration, ScfLive, Slot, SolveCounts, TeachBuffer, TeachEvent, TrainingViews};
pub use dynamics::{DynCommand, DynSnapshot, Dynamics, ReferenceProgress, SideDensity, Walls};
pub use integrator::{Integrator, StepStats};
pub use relax::{Fire, FireOutcome, FireParams, FireStep, RelaxProgress, F_TOL};
pub use provider::{
    AuditRecord, AuditVerdict, ClassKey, ClassLoo, Readiness, TeachOffer, Display, DisplayInterrupt, EnergyBreakdown, EvalStats, Extrapolation, ForceProvider, ForceResult,
    LoneAtomLabels, OrbitalDisplay, OrbitalField, OrbitalInfo, OrbitalLabel, OrbitalRequest, OrbitalViewRequest, ProviderError, SideDisplay, SideKind,
    SolverError, Source, Spin, SpinSummary, Surface, Trainable, TrainableStatus, UnoccupiedStatus, nothing_to_solve, PseudoSummary, SharedShell, CORE_OVERLAP_WARNING, OCCUPATION_CHANGE_WARNING, ORBITAL_MIN_INTERVAL, TRACK_HOMO,
};
pub use charges::{ChargeSource, PartialCharges};
pub use bonds::{components, Bond, BondGraph, BondKind, Detector, MarchStop, PairReport, Promolecule, Route, Verdict};
pub use references::{solve_atom, AtomReference, AtomReferences, AtomScan, RadialDensity, ReferenceKey};
pub use sampling::{allowed_multiplicities, GhostAtom, GhostFinished, GhostJob, GhostOutcome, PassGhost, ReferenceCache, SamplingPlan, SamplingProgress};
pub use scene::{Cell, IonicPolicy, Nucleus, Scene, CELL_MARGIN_H};
pub use test_providers::{FailKind, Failing, Gaussians, GriddedGaussians, Harmonic, MorseDimer, Shared, SharedTrainable, Term, Zero};
pub use thermostat::Csvr;
