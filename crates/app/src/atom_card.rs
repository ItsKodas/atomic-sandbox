//! The "atom card": a labelled 2D Bohr-style schematic of the selected nucleus — protons, neutrons,
//! electrons in shells — drawn in a small card at the top-right of the 3D view (it can be hidden
//! from the dock's Selection tab, which repeats its text lines).
//!
//! This is explicitly a diagram, not the simulation: the sandbox never resolves nucleons (the nucleus
//! is a point of charge +Z and a mass; see the design spec §11), and in a molecule the electrons are
//! shared — the density cloud in the 3D view is the real picture. Everything shown here derives from
//! Z and the isotope mass table, the only per-element data the project carries (rule 1): N = A − Z with
//! A the mass number of the most abundant isotope (`units::isotope_mass_amu` rounded), and the shell
//! occupation is the textbook 2 / 8 / 8 / 18 / 18 / 32 filling of a neutral atom.
use crate::scene::SceneNucleus;
use egui::{Align2, Color32, FontId, Pos2, Stroke, Vec2};
use render::elements;

/// Textbook Bohr shell capacities (K, L, M, N, O, P). Only the counts are shown — no orbital shapes.
const SHELL_CAPACITY: [usize; 6] = [2, 8, 8, 18, 18, 32];

/// Element names for the display badge (display-only, like `render::elements`).
pub fn element_name(z: u8) -> &'static str {
    [
        "?", "hydrogen", "helium", "lithium", "beryllium", "boron", "carbon", "nitrogen", "oxygen",
        "fluorine", "neon",
    ]
    .get(z as usize)
    .copied()
    .unwrap_or("?")
}

/// Mass number of the most abundant isotope (the mass table rounded), or `None` beyond the table.
pub fn mass_number(z: u8) -> Option<u32> {
    units::isotope_mass_amu(z).map(|m| m.round() as u32)
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

/// Superscript digits for the isotope label (⁷Li).
fn superscript(n: u32) -> String {
    const DIGITS: [char; 10] = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
    n.to_string()
        .chars()
        .map(|c| DIGITS[c.to_digit(10).unwrap_or(0) as usize])
        .collect()
}

/// The card's title: "¹⁶O oxygen" (isotope, symbol, name).
pub fn title(z: u8) -> String {
    match mass_number(z) {
        Some(a) => format!("{}{} {}", superscript(a), elements::symbol(z), element_name(z)),
        None => format!("{} (Z = {z})", elements::symbol(z)),
    }
}

/// "8 protons · 8 neutrons".
pub fn composition(z: u8) -> String {
    let n = mass_number(z).map(|a| a.saturating_sub(z as u32));
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

/// The caveat under the drawing (short) and behind its "i" (long).
const CAVEAT: &str = "In a molecule the electrons are shared: the cloud in the 3D view is the real picture. The simulation treats the nucleus as a point of charge +Z.";

/// Draws the card for `nucleus` (index `i` in the scene) with its top-right corner at
/// `top_right` (the 3D view's corner); returns the card's rect. Call once per frame while a
/// nucleus is selected.
pub fn show(ctx: &egui::Context, top_right: Pos2, nucleus: &SceneNucleus, i: usize) -> egui::Rect {
    let z = nucleus.z;
    let n_neutrons = mass_number(z).map(|a| a.saturating_sub(z as u32));
    let shells = shells(z);
    egui::Area::new(egui::Id::new("atom-card"))
        .pivot(Align2::RIGHT_TOP)
        .fixed_pos(top_right)
        .order(egui::Order::Middle)
        .show(ctx, |ui| {
            crate::ui::theme::card().show(ui, |ui| {
                ui.set_width(220.0);
                ui.horizontal(|ui| {
                    ui.strong(title(z));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        crate::ui::theme::info(ui, CAVEAT);
                    });
                });
                ui.label(format!("nucleus #{i}: {}", composition(z)));
                ui.label(electrons_line(z));
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
    fn shells_fill_the_textbook_way() {
        assert_eq!(shells(1), vec![1]);
        assert_eq!(shells(2), vec![2]);
        assert_eq!(shells(3), vec![2, 1]);
        assert_eq!(shells(8), vec![2, 6]);
        assert_eq!(shells(10), vec![2, 8]);
    }

    #[test]
    fn mass_numbers_come_from_the_isotope_table() {
        assert_eq!(mass_number(1), Some(1));
        assert_eq!(mass_number(3), Some(7));
        assert_eq!(mass_number(8), Some(16));
        assert_eq!(mass_number(11), None);
    }

    #[test]
    fn superscripts_render_every_digit() {
        assert_eq!(superscript(16), "¹⁶");
        assert_eq!(superscript(7), "⁷");
    }
}
