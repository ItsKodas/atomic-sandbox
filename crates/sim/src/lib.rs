//! Nuclear dynamics (M4): the scene, the integrator, the thermostat, the driver and the relax
//! tool, behind the force boundary ([`ForceProvider`], with [`Trainable`] for a provider that
//! learns). The providers themselves — Kohn-Sham and surrogate — live in the `providers` crate
//! (M6, Task 1); `sim` depends on `units` and `gpu` (the display texture handle) only. The bond
//! detector ([`bonds`]) and the isolated-atom references ([`references`]) it reads the density
//! against are M6 Task 3. The pure part of the viewer's bonds from experience
//! ([`bond_experience`]) moved here for the validate suite (live orbitals, Task 9).
pub mod activity;
pub mod bond_experience;
pub mod bonds;
pub mod charges;
pub mod dynamics;
pub mod integrator;
pub mod knowledge;
pub mod profiles;
pub mod provider;
pub mod references;
pub mod refit_eta;
pub mod relax;
pub mod sampling;
pub mod scene;
pub mod test_providers;
pub mod thermostat;

pub use refit_eta::{RefitEstimate, RefitEta, RefitKind, RefitRecord, RefitStage, RefitStages};
pub use activity::{ActivityProbe, DisplayJob, DisplaySolve, ProbeView, RefitView, ScfEnd, ScfIteration, ScfLive, Slot, SolveCounts, TeachBuffer, TeachEvent, TrainingViews};
pub use dynamics::{DynCommand, DynSnapshot, Dynamics, ReferenceProgress, SideDensity, Walls, HARRIS_SAME_GEOMETRY};
pub use integrator::{Integrator, StepStats};
pub use knowledge::{atom_coverage, Confidence, ElementKnowledge, KnowledgeMap, PairKnowledge};
pub use relax::{Fire, FireOutcome, FireParams, FireStep, RelaxProgress, F_TOL};
pub use provider::{
    AuditRecord, AuditVerdict, ClassKey, ClassLoo, SpeciesVerdict, key_words, ImportSummary, MigrationPhase, MigrationStatus, ModelSource, PurgeReport, Readiness, TeachOffer, Display, DisplayInterrupt, EnergyBreakdown, EvalStats, Extrapolation, ForceProvider, ForceResult,
    ColourLook, ExperienceUse, HarrisAnswer, HarrisAtoms, LoneAtomLabels, OrbitalDisplay, OrbitalKind, OrbitalField, OrbitalInfo, OrbitalLabel, OrbitalRequest, OrbitalViewRequest, CORE_GAP, ProviderError, SideDisplay, SideKind,
    RefitPoll, SampleId, SampleSuspect, ShareTables, SolverError, Source, Spin, SpinSummary, Surface, Trainable, TrainableStatus, UnoccupiedStatus, nothing_to_solve, PseudoSummary, PartialLevel, Frontier, CORE_OVERLAP_WARNING, BRANCH_JUMP_WARNING, branch_jump_warning, untrainable_citation, ORBITAL_MIN_INTERVAL, TRACK_HOMO,
};
pub use charges::{orbital_shares, orbital_shares_many, set_average, share_sum, ChargeSource, PartialCharges, Weights};
pub use bonds::{components, Bond, BondGraph, BondKind, Detector, MarchStop, PairReport, Promolecule, Route, Verdict};
pub use profiles::{radial_profile, AtomProfile};
pub use references::{solve_atom, AtomReference, AtomReferences, AtomScan, RadialDensity, ReferenceKey};
pub use sampling::{allowed_multiplicities, GhostAtom, GhostFinished, GhostJob, GhostOutcome, PassGhost, ReferenceCache, SamplingPlan, SamplingProgress};
pub use scene::{Cell, IonicPolicy, Nucleus, Scene, CELL_MARGIN_H};
pub use test_providers::{FailKind, Failing, Gaussians, GriddedGaussians, Harmonic, MorseDimer, Shared, SharedTrainable, Term, Zero};
pub use thermostat::Csvr;
