//! What the chrome says about pseudopotentials (M7 Task 9; spec §4.7, §5.5, §6.5; NLCC ruling 6):
//! the words, kept apart from the widgets so they are testable and the menu, the badge, the
//! Selection tab and the Readouts say the same thing.
//!
//! Every number comes from the generation record (`sim::PseudoSummary`, filled from
//! `qm::radial::GenerationRecord`): nothing here is a per-element table (rule 1). A scene refused
//! for a grid coarser than a species' `h_req` never reaches a summary — the provider's error says
//! it; the Quality menu marks those levels from the same record before the click.
use super::PanelInput;
use sim::PseudoSummary;

/// The pseudised species the chrome can describe now: the latest evaluation's, else — a scene refused for its
/// grid reached no evaluation — those already generated for the scene (`providers::pseudo::cached_summaries`,
/// which never generates and never waits).
pub fn pseudised(input: &PanelInput<'_>) -> Vec<PseudoSummary> {
    match input.snapshot.map(|s| &s.dynamics.eval.pseudised).filter(|p| !p.is_empty()) {
        Some(p) => p.clone(),
        None if input.scene.ionic != sim::IonicPolicy::None => input.scene.to_sim().map_or_else(|_| Vec::new(), |s| providers::pseudo::cached_summaries(&s)),
        None => Vec::new(),
    }
}

/// "needs h ≤ 0.20 (O)" for a grid spacing `h` above what the pseudised species allow, else `None`.
pub fn needs_finer(pseudised: &[PseudoSummary], h: f64) -> Option<String> {
    coarsest_allowed(pseudised).filter(|&(allowed, _)| h > allowed + 1e-9).map(|(allowed, z)| format!("needs h ≤ {allowed:.2} ({})", sym(z)))
}

/// The tooltip of Physics ▸ Ionic potential (spec §4.7).
pub const IONIC_TIP: &str = "how the nuclei and their core electrons enter: all-electron — every electron on the grid, nuclei smeared at σ = h (heavy atoms lose most of their energy to the smearing); cores only — each species with a core gets a pseudopotential generated from this program's own all-electron atom (H, He stay all-electron); all species — H and He too (their first use checks the pseudised dimer against this program's own all-electron dimer: ≈ 1 min on the GPU, tens of minutes without one). Pseudopotential energies are valence energies: not comparable with all-electron totals; energy differences are the physics";

/// The honesty line under the policy switch (spec §6.5).
pub const VALENCE_NOTE: &str = "E_KS is a valence energy under a pseudopotential (spec §6.5): totals are not comparable with all-electron ones — differences are the physics";

/// The tooltip of View ▸ Density ▸ with partial core.
pub const PARTIAL_CORE_TIP: &str = "adds ρ_pc — a display of the generator's core model, not solved. The partial core is the smooth stand-in for the frozen core's density that the exchange–correlation functional sees (spec §4.5); it never enters the Hartree potential, and the display never feeds the solve. Applies from the next step";

/// The egg-box measurements the line cites (`references.md` "Task 7c round 2", recipe `m7-tm-kb-nlcc-4`,
/// `Cores only`, O₂ net lattice force): reported, not part of any element's certificate. The dimer gate (Task 11) moved only
/// H and He, not the recipe version, so the O figures stand.
pub const EGG_BOX_RECORD: &str = "measured on O₂ with this recipe (m7-tm-kb-nlcc-4): net lattice force 4.26e-5 Ha/Bohr at h = 0.20 Bohr (the 3e-5 target missed), 9.17e-6 at 0.175, 6.75e-6 at 0.15 (met); all-electron water at 0.25 Bohr: 3.4e-4. Energies are unaffected; total energy is still conserved; the centre of mass drifts unless the COM correction is on";

/// The grid spacing up to which this species' egg-box force is certified: `min(h_eb, h_req)` (T8(e)'s
/// `h_eb` above `h_req` carries no meaning — a scene coarser than `h_req` is refused). `None`: no grid on
/// the ladder certifies it (F, at this recipe version).
pub fn egg_box_grid(p: &PseudoSummary) -> Option<f64> {
    match (p.h_eb, p.h_req) {
        (Some(e), Some(r)) => Some(e.min(r)),
        (Some(e), None) => Some(e),
        (None, _) => None,
    }
}

/// Whether the species' egg-box force is certified at spacing `h` (a hair of slack for the typed value).
pub fn egg_box_certified(p: &PseudoSummary, h: f64) -> bool {
    egg_box_grid(p).is_some_and(|g| h <= g + 1e-9)
}

fn sym(z: u8) -> &'static str {
    units::elements::symbol(z)
}

/// "egg-box force not certified at this grid" with the element, h and what certifies it; `None` when
/// every pseudised species is certified at `h`. Never a refusal (NLCC ruling 6).
pub fn egg_box_warning(pseudised: &[PseudoSummary], h: f64) -> Option<String> {
    let bad: Vec<String> = pseudised
        .iter()
        .filter(|p| !egg_box_certified(p, h))
        .map(|p| match egg_box_grid(p) {
            Some(g) => format!("{} needs h ≤ {g:.3}", sym(p.z)),
            None => format!("{} has no certified grid", sym(p.z)),
        })
        .collect();
    (!bad.is_empty()).then(|| format!("egg-box force not certified at this grid (h = {h:.3} Bohr: {})", bad.join("; ")))
}

/// One row of the Ionic potential section: the species, its valence charge, the largest r_c and the
/// grids its energies (`h_req`) and its egg-box force (`min(h_eb, h_req)`) are certified to.
pub fn species_line(p: &PseudoSummary) -> String {
    let grid = |g: Option<f64>| g.map_or("none on the ladder".to_string(), |g| format!("h ≤ {g:.2}"));
    format!(
        "{}  Z_v {} · r_c {:.2} Bohr · {} · energies certified at {} · egg-box certified at {}",
        sym(p.z),
        p.z_v,
        p.r_c_max,
        p.local,
        grid(p.h_req),
        grid(egg_box_grid(p))
    )
}

/// The badge's and the atom card's tail for a pseudised nucleus: "pseudo, Z_v = 6, r_c = 1.35 Bohr, needs h ≤ 0.20".
pub fn badge_tail(p: &PseudoSummary) -> String {
    let needs = p.h_req.map_or(String::new(), |h| format!(", needs h ≤ {h:.2}"));
    format!("pseudo, Z_v = {}, r_c = {:.2} Bohr{needs}", p.z_v, p.r_c_max)
}

/// The summary for species `z` in the latest evaluation, if it was pseudised.
pub fn of_species(pseudised: &[PseudoSummary], z: u8) -> Option<&PseudoSummary> {
    pseudised.iter().find(|p| p.z == z)
}

/// The coarsest grid spacing every pseudised species allows, and the first species that sets it: the
/// Quality menu marks a level above it "needs h ≤ …". `None` when nothing is pseudised or no species
/// has an `h_req` (then no grid on the ladder certifies its energies; the provider's error says so).
pub fn coarsest_allowed(pseudised: &[PseudoSummary]) -> Option<(f64, u8)> {
    pseudised.iter().filter_map(|p| p.h_req.map(|h| (h, p.z))).min_by(|a, b| a.0.total_cmp(&b.0))
}

/// The Readouts row label of the Kohn-Sham energy ("E_KS", or its valence reading under a
/// pseudopotential — spec §6.5).
pub fn energy_label(pseudised: &[PseudoSummary]) -> &'static str {
    if pseudised.is_empty() { "E_KS" } else { "E_KS (valence, pseudopotential)" }
}

/// The occupation line of the Task 6 review ruling A.6: "open shell: n e shared equally over k degenerate
/// levels (↑/↓), the spherical-atom ensemble; LSDA's symmetry-broken integer state lies ΔE lower (references.md)".
/// One line per shared shell of the final step, `None` for integer occupations.
pub fn occupation_lines(shells: &[sim::SharedShell]) -> Vec<String> {
    shells
        .iter()
        .map(|c| {
            let arrow = if c.spin == 0 { "↑" } else { "↓" };
            // f of the k levels of the cluster: n = f·k electrons (of this spin)
            let n = c.f * c.count as f64;
            format!(
                "open shell: {} e shared equally over {} degenerate levels ({arrow}), the spherical-atom ensemble; LSDA's symmetry-broken integer state lies ΔE lower (references.md)",
                trim(n),
                c.count
            )
        })
        .collect()
}

/// A count of electrons without trailing zeros ("2", "1.5").
fn trim(x: f64) -> String {
    let s = format!("{x:.3}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(z: u8, z_v: f64, h_req: Option<f64>, h_eb: Option<f64>) -> PseudoSummary {
        PseudoSummary { z, z_v, f: 1.0, local: "s-local".into(), r_c_max: 1.35, h_req, h_eb, r_core: 0.8, record: String::new() }
    }

    /// NLCC ruling 6 and its re-review (NOTE-R1): the egg-box grid is min(h_eb, h_req); a scene coarser than it
    /// gets the line "egg-box force not certified at this grid", F (no h_eb) always; never a refusal (text only).
    #[test]
    fn egg_box_grid_is_min_of_h_eb_and_h_req_and_the_line_says_who() {
        let o = summary(8, 6.0, Some(0.20), Some(0.15));
        let he = summary(2, 2.0, Some(0.20), Some(0.35));
        let h = summary(1, 1.0, Some(0.35), Some(0.35));
        let f = summary(9, 7.0, Some(0.15), None);
        assert_eq!(egg_box_grid(&o), Some(0.15));
        assert_eq!(egg_box_grid(&he), Some(0.20), "He: h_eb 0.35 > h_req 0.20 carries no meaning");
        assert_eq!(egg_box_grid(&h), Some(0.35));
        assert_eq!(egg_box_grid(&f), None);
        assert!(egg_box_warning(&[o.clone(), h.clone()], 0.15).is_none(), "certified at the Accurate default");
        let w = egg_box_warning(&[o.clone(), h.clone()], 0.20).expect("O is not certified at 0.20");
        assert!(w.starts_with("egg-box force not certified at this grid") && w.contains("O needs h ≤ 0.150") && !w.contains('H'), "{w}");
        let w = egg_box_warning(&[f], 0.15).expect("F has no certified grid at any h");
        assert!(w.contains("F has no certified grid"), "{w}");
        assert!(egg_box_warning(&[], 0.5).is_none(), "nothing pseudised: nothing to say");
    }

    #[test]
    fn badge_and_species_lines_carry_the_record() {
        let o = summary(8, 6.0, Some(0.20), Some(0.15));
        assert_eq!(badge_tail(&o), "pseudo, Z_v = 6, r_c = 1.35 Bohr, needs h ≤ 0.20");
        assert!(species_line(&o).contains("energies certified at h ≤ 0.20") && species_line(&o).contains("egg-box certified at h ≤ 0.15"));
        assert!(species_line(&summary(9, 7.0, Some(0.15), None)).contains("egg-box certified at none on the ladder"));
        assert_eq!(coarsest_allowed(&[summary(1, 1.0, Some(0.35), None), o]), Some((0.20, 8)));
        assert_eq!(coarsest_allowed(&[]), None);
    }

    #[test]
    fn the_occupation_line_is_ruling_a6s() {
        let shells = vec![sim::SharedShell { spin: 1, first: 1, count: 3, f: 1.0 / 3.0, spread: 1e-7 }];
        let lines = occupation_lines(&shells);
        assert_eq!(lines.len(), 1);
        assert_eq!(
            lines[0],
            "open shell: 1 e shared equally over 3 degenerate levels (↓), the spherical-atom ensemble; LSDA's symmetry-broken integer state lies ΔE lower (references.md)"
        );
        assert!(occupation_lines(&[]).is_empty());
        assert_eq!(energy_label(&[]), "E_KS");
        assert_eq!(energy_label(&[summary(8, 6.0, None, None)]), "E_KS (valence, pseudopotential)");
    }
}
