//! The "atom card": a labelled 2D Bohr-style schematic of the selected nucleus — protons, neutrons,
//! electrons in shells — drawn in a small card at the top-right of the 3D view (it can be hidden
//! from the dock's Selection tab, which repeats its text lines).
//!
//! This is explicitly a diagram, not the simulation: the sandbox never resolves nucleons (the nucleus
//! is a point of charge +Z and a mass; see the design spec §11), and in a molecule the electrons are
//! shared — the density cloud in the 3D view is the real picture. Everything shown here derives from
//! Z and the isotope mass table, the only per-element data the project carries (rule 1): N = A − Z with
//! A the nucleus's own mass number (`SceneNucleus::mass_number`: its isotope, else the most abundant
//! one — M6 spec §5.4), and the shell occupation is the textbook 2 / 8 / 8 / 18 / 18 / 32 filling of a
//! neutral atom.
use crate::names::{element_name, superscript_digits};
use crate::scene::SceneNucleus;
use egui::{Align2, Color32, FontId, Pos2, Stroke, Vec2};
use render::elements;

/// Textbook Bohr shell capacities (K, L, M, N, O, P). Only the counts are shown — no orbital shapes.
const SHELL_CAPACITY: [usize; 6] = [2, 8, 8, 18, 18, 32];

/// The nucleus's mass number (its own isotope, else the most abundant one), or `None` beyond the
/// table.
pub fn mass_number(nucleus: &SceneNucleus) -> Option<u32> {
    nucleus.mass_number().map(u32::from)
}

/// Electrons per shell for a neutral atom with `z` electrons.
pub fn shells(z: u8) -> Vec<usize> {
    let mut left = z as usize;
    let mut out = Vec::new();
    for cap in SHELL_CAPACITY {
        if left == 0 {
            break;
        }
        let n = left.min(cap);
        out.push(n);
        left -= n;
    }
    out
}

/// The card's title: "¹⁶O oxygen" (the nucleus's isotope, symbol, name).
pub fn title(nucleus: &SceneNucleus) -> String {
    let z = nucleus.z;
    match mass_number(nucleus) {
        Some(a) => format!("{}{} {}", superscript_digits(a), elements::symbol(z), element_name(z)),
        None => format!("{} (Z = {z})", elements::symbol(z)),
    }
}

/// "8 protons · 8 neutrons" (the nucleus's own isotope: ²H has 1 neutron).
pub fn composition(nucleus: &SceneNucleus) -> String {
    let z = nucleus.z;
    let n = mass_number(nucleus).map(|a| a.saturating_sub(z as u32));
    format!("{} protons · {} neutrons", z, n.map_or("?".to_string(), |n| n.to_string()))
}

/// "8 electrons (neutral atom), shells 2 / 6".
pub fn electrons_line(z: u8) -> String {
    format!(
        "{} electrons (neutral atom), shells {}",
        z,
        shells(z).iter().map(|s| s.to_string()).collect::<Vec<_>>().join(" / ")
    )
}

/// "solved as a pseudo-ion: 6 valence electrons, the 2-electron core is frozen (not in the diagram's solve)" — the card
/// draws the textbook neutral atom, the simulation solves Z_v electrons for a pseudised species.
pub fn pseudo_line(p: &sim::PseudoSummary) -> String {
    let core = (p.z as f64 - p.z_v).round() as i64;
    format!("pseudo-ion in the solve: {} valence electrons; the {core} core electrons are frozen into the potential", p.z_v)
}

/// The caveat under the drawing (short) and behind its "i" (long).
const CAVEAT: &str = "In a molecule the electrons are shared: the cloud in the 3D view is the real picture. The simulation treats the nucleus as a point of charge +Z.";

/// Draws the card for `nucleus` (index `i` in the scene) with its top-right corner at
/// `top_right` (the 3D view's corner); returns the card's rect. Call once per frame while a
/// nucleus is selected.
pub fn show(ctx: &egui::Context, top_right: Pos2, nucleus: &SceneNucleus, i: usize, pseudo: Option<&sim::PseudoSummary>) -> egui::Rect {
    let z = nucleus.z;
    let n_neutrons = mass_number(nucleus).map(|a| a.saturating_sub(z as u32));
    let shells = shells(z);
    egui::Area::new(egui::Id::new("atom-card"))
        .pivot(Align2::RIGHT_TOP)
        .fixed_pos(top_right)
        .order(egui::Order::Middle)
        .fade_in(false)
        .show(ctx, |ui| {
            crate::ui::theme::card().show(ui, |ui| {
                ui.set_width(220.0);
                ui.horizontal(|ui| {
                    ui.strong(title(nucleus));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        crate::ui::theme::info(ui, CAVEAT);
                    });
                });
                ui.label(format!("nucleus #{i}: {}", composition(nucleus)));
                ui.label(electrons_line(z));
                // M7: under a pseudopotential only the valence electrons are solved
                if let Some(p) = pseudo {
                    ui.label(pseudo_line(p));
                }
                let (rect, _) =
                    ui.allocate_exact_size(Vec2::new(220.0, 190.0), egui::Sense::hover());
                let p = ui.painter_at(rect);
                let c = rect.center();
                // nucleus: a compact cluster of red (p) and green (n) dots on a small spiral
                // interleave protons (red) and neutrons (green) outward along a sunflower spiral
                let (mut p_left, mut n_left) = (z as usize, n_neutrons.unwrap_or(0) as usize);
                let golden = 2.399_963_f32;
                let mut k = 0usize;
                while p_left + n_left > 0 {
                    let proton = if p_left == 0 { false } else if n_left == 0 { true } else { k % 2 == 0 };
                    if proton { p_left -= 1 } else { n_left -= 1 }
                    let r = 2.4 * (k as f32 + 0.5).sqrt();
                    let ang = k as f32 * golden;
                    let q = c + Vec2::new(ang.cos(), ang.sin()) * r;
                    let colour = if proton { Color32::from_rgb(232, 86, 86) } else { Color32::from_rgb(96, 190, 110) };
                    p.circle_filled(q, 3.2, colour);
                    k += 1;
                }
                // shells: rings with electron dots
                let max_r = 86.0;
                let n_shells = shells.len().max(1) as f32;
                for (s, &count) in shells.iter().enumerate() {
                    let r = 22.0 + (s as f32 + 1.0) * (max_r - 22.0) / n_shells;
                    p.circle_stroke(c, r, Stroke::new(1.0, Color32::from_white_alpha(70)));
                    for e in 0..count {
                        let ang = std::f32::consts::TAU * e as f32 / count as f32 - std::f32::consts::FRAC_PI_2;
                        let q = c + Vec2::new(ang.cos(), ang.sin()) * r;
                        p.circle_filled(q, 2.6, Color32::from_rgb(120, 180, 255));
                    }
                }
                p.text(
                    Pos2::new(rect.left() + 4.0, rect.bottom() - 4.0),
                    Align2::LEFT_BOTTOM,
                    "schematic — not to scale",
                    FontId::proportional(10.0),
                    Color32::from_white_alpha(140),
                );
                ui.small(CAVEAT);
            });
        })
        .response
        .rect
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pseudised_species_says_what_is_solved() {
        let p = sim::PseudoSummary { z: 8, z_v: 6.0, f: 1.0, local: "s-local".into(), r_c_max: 1.35, h_req: Some(0.2), h_eb: Some(0.15), r_core: 0.8, record: String::new() };
        assert_eq!(pseudo_line(&p), "pseudo-ion in the solve: 6 valence electrons; the 2 core electrons are frozen into the potential");
    }

    #[test]
    fn shells_fill_the_textbook_way() {
        assert_eq!(shells(1), vec![1]);
        assert_eq!(shells(2), vec![2]);
        assert_eq!(shells(3), vec![2, 1]);
        assert_eq!(shells(8), vec![2, 6]);
        assert_eq!(shells(10), vec![2, 8]);
    }

    #[test]
    fn mass_numbers_come_from_the_nucleus_then_the_isotope_table() {
        let n = |z: u8, a: Option<u16>| SceneNucleus { z, a, pos: [0.0; 3], vel: [0.0; 3] };
        assert_eq!(mass_number(&n(1, None)), Some(1));
        assert_eq!(mass_number(&n(1, Some(2))), Some(2));
        assert_eq!(mass_number(&n(3, None)), Some(7));
        assert_eq!(mass_number(&n(8, None)), Some(16));
        assert_eq!(mass_number(&n(8, Some(18))), Some(18));
        assert_eq!(mass_number(&n(11, None)), None);
    }

    /// The card's lines follow the nucleus's isotope (merge of main's card refactor with M6's
    /// per-nucleus mass number).
    #[test]
    fn title_and_composition_follow_the_isotope() {
        let n = |z: u8, a: Option<u16>| SceneNucleus { z, a, pos: [0.0; 3], vel: [0.0; 3] };
        assert_eq!(title(&n(8, None)), "¹⁶O oxygen");
        assert_eq!(title(&n(1, Some(2))), "²H hydrogen");
        assert_eq!(composition(&n(1, Some(2))), "1 protons · 1 neutrons");
        assert_eq!(composition(&n(1, None)), "1 protons · 0 neutrons");
    }
}
