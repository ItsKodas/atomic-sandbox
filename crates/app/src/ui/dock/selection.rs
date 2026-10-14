//! The Selection tab: the selected nucleus (element, index, position, composition, electrons),
//! its Pin / Focus / Delete actions with their keys, the distance guides as a list, the atom
//! card toggle and its caveat; the queued bonds slot (rows 86–90, 131–133, 136, 141, 147).
use super::super::{theme, Action, PanelInput, UiState};
use crate::atom_card;
use crate::tools;
use egui::{RichText, Ui};
use render::elements;

pub fn show(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    let Some((i, n)) = input.selection() else {
        theme::note(ui, "no selection (Select tool, click a nucleus)");
        card_toggle(ui, st);
        return;
    };
    super::group(ui, st, "d-selection", |ui, _| {
        ui.horizontal(|ui| {
            ui.label(RichText::new(atom_card::title(n)).strong().size(14.0));
            ui.label(RichText::new(format!("nucleus #{i}")).monospace().color(theme::INK_DIM));
        });
        egui::Grid::new("selection").num_columns(2).striped(true).max_col_width(250.0).show(ui, |ui| {
            // long values wrap in the dock instead of widening it
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
            ui.label("position");
            ui.monospace(format!("({:.2}, {:.2}, {:.2}) Bohr", n.pos[0], n.pos[1], n.pos[2]));
            ui.end_row();
            ui.label("composition");
            ui.label(atom_card::composition(n));
            ui.end_row();
            ui.label("electrons");
            ui.label(atom_card::electrons_line(n.z));
            ui.end_row();
        });
        ui.horizontal_wrapped(|ui| {
            let pinned = input.pinned(i);
            let mut p = pinned.unwrap_or(false);
            if ui
                .add_enabled(pinned.is_some(), egui::Checkbox::new(&mut p, "pinned"))
                .on_hover_text("hold this nucleus still (velocity zeroed every step)")
                .changed()
            {
                acts.push(Action::Pin(i, p));
            }
            if ui.button("Focus selected").clicked() {
                acts.push(Action::Focus);
            }
            if ui.button(RichText::new("Delete selected").color(theme::DANGER_INK)).on_hover_text("also the Delete key, also while playing").clicked() {
                acts.push(Action::Delete(i));
            }
            if ui.button("Deselect").on_hover_text("Esc").clicked() {
                acts.push(Action::Deselect);
            }
        });
        theme::note(ui, "Del removes it · Esc deselects");
    });
    super::group(ui, st, "d-guides", |ui, _| {
        theme::section(ui, "Distance guides");
        let guides = tools::distance_guides(n.pos, &input.scene.nuclei, Some(i));
        if guides.is_empty() {
            theme::note(ui, format!("no other nucleus within {} Bohr", tools::GUIDE_RADIUS));
        }
        egui::Grid::new("guides").num_columns(3).striped(true).show(ui, |ui| {
            for g in guides {
                let other = &input.scene.nuclei[g.index];
                let pin = if input.pinned(g.index) == Some(true) { " (pinned)" } else { "" };
                ui.label(format!("to {} #{}{pin}", elements::symbol(other.z), g.index));
                ui.monospace(format!("{:.2} Bohr", g.distance));
                ui.label(RichText::new(format!("{:.3} Å", units::bohr_to_angstrom(g.distance))).monospace().color(theme::INK_DIM));
                ui.end_row();
            }
        });
    });
    // SEAM(Q6, Q7): bonds read off the density, the molecule name and the composition badge.
    ui.horizontal(|ui| {
        theme::section(ui, "Bonds (read off the density)");
        theme::queued_tag(ui);
    });
    theme::note(ui, "bonded atoms and the molecule's name will show here once a solve converges");
    card_toggle(ui, st);
}

fn card_toggle(ui: &mut Ui, st: &mut UiState) {
    super::group(ui, st, "d-card", |ui, st| {
        theme::section(ui, "Atom card");
        ui.checkbox(&mut st.show_atom_card, "show the card in the 3D view");
        theme::note(ui, "In a molecule the electrons are shared: the cloud in the 3D view is the real picture. The simulation treats the nucleus as a point of charge +Z. The card is a schematic, not to scale.");
    });
}
