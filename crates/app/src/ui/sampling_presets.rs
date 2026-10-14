//! The sampling plan's presets (user request 2026-10-15, "In the sampling area can we get some
//! presets with descriptions for use cases etc?"): a const table of named plans for the Model
//! tab's Sampling plan section. UI data only — no physics: each preset is a set of
//! `sim::SamplingPlan` field values, Standard is `SamplingPlan::default()`, the scan range stays
//! at its default and the seed is never touched. Which preset is selected is derived from the
//! plan by comparison ([`matching`]), never stored, so it cannot drift from the fields.
//!
//! **The copy promises only what the surrogate's rules allow** (M6 final review, core I1, rule 7):
//! no preset's trust outcome is measured, so no description says a preset reaches trust or that
//! the model will answer a regime. A preset only changes what the pass samples; whether the fit
//! answers is decided after each pass by the trust checks (Model tab), and two rules hold whatever
//! was sampled — an atom with no neighbour inside the model's cutoff in a scene of two or more
//! atoms is always answered by DFT (rule 5a), and one poorly predicted sample untrusts its whole
//! class (pair or many-body; the class-wise leave-one-out rule, `crates/validate/references.md`
//! "`the_stuck_escape_repro_is_answered_honestly`"), so a hotter pass can leave more on DFT, not
//! less.
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
pub const RULE_OF_THUMB: &str = "a preset only sets what the pass samples. The hot burst only explores shapes reachable at its temperature, so sample at roughly twice the temperature you will play at; a hotter pass covers more shapes, but one sample the fit predicts poorly keeps that whole kind of environment (pairs, or three or more atoms together) on DFT until more are learned. Whether the model is trusted is checked after every pass and shown in this tab; anything it has not seen falls back to DFT.";

/// The presets, in the row's order. Standard must equal `SamplingPlan::default()` (tested).
pub const PRESETS: [SamplingPreset; 4] = [
    SamplingPreset {
        name: "Quick look",
        perturbations: 30,
        perturbation_sigma: 0.10,
        hot_burst_steps: 100,
        hot_burst_t: 800.0,
        scan_points_per_pair: 8,
        description: "fewest solves; samples only small displacements and a short 800 K burst around the current shape. Expect DFT fallbacks as soon as the molecule moves far from it.",
    },
    SamplingPreset {
        name: "Gentle",
        perturbations: 60,
        perturbation_sigma: 0.10,
        hot_burst_steps: 200,
        hot_burst_t: 800.0,
        scan_points_per_pair: 12,
        description: "small displacements and an 800 K burst: aimed at play near room temperature (up to ~400 K). Shapes it has not seen still fall back to DFT.",
    },
    SamplingPreset {
        name: "Standard",
        perturbations: 60,
        perturbation_sigma: 0.15,
        hot_burst_steps: 200,
        hot_burst_t: 2000.0,
        scan_points_per_pair: 12,
        description: "the default: wider displacements and a 2000 K burst, aimed at play with moderate heating (~1000 K). A stretched sample the fit predicts poorly keeps pair environments on DFT until more are learned.",
    },
    SamplingPreset {
        name: "Hot / reactive",
        perturbations: 120,
        perturbation_sigma: 0.20,
        hot_burst_steps: 400,
        hot_burst_t: 3000.0,
        scan_points_per_pair: 16,
        description: "about twice the solves: the widest displacements, longer pair scans and a 3000 K burst widen what the model has seen. It does not make bond breaking a model answer: atoms pulled beyond the model's cutoff, and arrangements it has not seen, are answered by DFT, and one badly fit stretched sample can keep pair environments on DFT.",
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
