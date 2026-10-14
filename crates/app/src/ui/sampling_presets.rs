//! The sampling plan's presets (user request 2026-10-15, "In the sampling area can we get some
//! presets with descriptions for use cases etc?"): a const table of named plans for the Model
//! tab's Sampling plan section. UI data only — no physics: each preset is a set of
//! `sim::SamplingPlan` field values, Standard is `SamplingPlan::default()`, the scan range stays
//! at its default and the seed is never touched. Which preset is selected is derived from the
//! plan by comparison ([`matching`]), never stored, so it cannot drift from the fields.
use sim::SamplingPlan;

/// One named plan: the fields it sets (temperatures in kelvin, σ in Bohr) and its use case.
#[derive(Clone, Copy, Debug)]
pub struct SamplingPreset {
    pub name: &'static str,
    pub perturbations: usize,
    pub perturbation_sigma: f64,
    pub hot_burst_steps: usize,
    pub hot_burst_t: f64,
    pub scan_points_per_pair: usize,
    pub description: &'static str,
}

/// The preset row's hover text: how to pick a preset.
pub const RULE_OF_THUMB: &str = "the hot burst only explores shapes reachable at its temperature, so sample at roughly twice the temperature you will play at; a hotter pass covers more but is harder to fit";

/// The presets, in the row's order. Standard must equal `SamplingPlan::default()` (tested).
pub const PRESETS: [SamplingPreset; 4] = [
    SamplingPreset {
        name: "Quick look",
        perturbations: 30,
        perturbation_sigma: 0.10,
        hot_burst_steps: 100,
        hot_burst_t: 800.0,
        scan_points_per_pair: 8,
        description: "fastest pass; a first model for a molecule sitting near rest. Expect fallbacks as soon as it moves much.",
    },
    SamplingPreset {
        name: "Gentle",
        perturbations: 60,
        perturbation_sigma: 0.10,
        hot_burst_steps: 200,
        hot_burst_t: 800.0,
        scan_points_per_pair: 12,
        description: "room-temperature play up to ~400 K; the easiest surface to fit — try this first for 3 or more atoms.",
    },
    SamplingPreset {
        name: "Standard",
        perturbations: 60,
        perturbation_sigma: 0.15,
        hot_burst_steps: 200,
        hot_burst_t: 2000.0,
        scan_points_per_pair: 12,
        description: "general play including heating to ~1000 K; works well for pairs like H₂.",
    },
    SamplingPreset {
        name: "Hot / reactive",
        perturbations: 120,
        perturbation_sigma: 0.20,
        hot_burst_steps: 400,
        hot_burst_t: 3000.0,
        scan_points_per_pair: 16,
        description: "collisions and bond breaking; about twice the solves. With 3 or more atoms the fit may not reach the trust bar (it then says so and DFT keeps answering).",
    },
];

/// The label shown when the plan matches no preset.
pub const CUSTOM: &str = "Custom";

impl SamplingPreset {
    /// The plan with this preset's fields, the default scan range and `seed`.
    pub fn plan(&self, seed: u64) -> SamplingPlan {
        SamplingPlan {
            perturbations: self.perturbations,
            perturbation_sigma: self.perturbation_sigma,
            hot_burst_steps: self.hot_burst_steps,
            hot_burst_t: self.hot_burst_t,
            scan_points_per_pair: self.scan_points_per_pair,
            scan_range: SamplingPlan::default().scan_range,
            seed,
        }
    }

    /// Sets the plan's fields to this preset's, keeping its seed.
    pub fn apply(&self, plan: &mut SamplingPlan) {
        *plan = self.plan(plan.seed);
    }
}

/// The index in [`PRESETS`] of the preset the plan equals exactly, the seed ignored; `None`
/// ("Custom") when any other field differs.
pub fn matching(plan: &SamplingPlan) -> Option<usize> {
    PRESETS.iter().position(|p| p.plan(plan.seed) == *plan)
}
