//! The Selection tab: the selected nucleus (element, index, position, composition, electrons,
//! its isotope with the listed alternatives — M6 spec §5.4), its Pin / Focus / Delete actions with
//! their keys, the distance guides as a list, its bonds read off the density with each pair's
//! numbers, the scene's molecules (formula and name, M6 spec §3.5 and §4) with the honesty lines,
//! and the atom card toggle and its caveat (rows 86–90, 131–133, 136, 141, 147; Q6, Q7).
use super::super::{theme, Action, PanelInput, UiState};
use crate::atom_card;
use crate::tools;
use egui::{RichText, Ui};
use render::elements;

pub fn show(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    let Some((i, n)) = input.selection() else {
        theme::note(ui, "no selection (Select tool, click a nucleus)");
        molecules(ui, st, input);
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
        isotopes(ui, i, n, acts);
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
    bonds(ui, st, input, i);
    molecules(ui, st, input);
    card_toggle(ui, st);
}

/// The selected nucleus's isotope: the listed mass numbers of its Z with their masses
/// (`Action::SetIsotope`; mass only — the forces are the same for every isotope).
fn isotopes(ui: &mut Ui, i: usize, n: &crate::scene::SceneNucleus, acts: &mut Vec<Action>) {
    let listed = units::isotopes_of(n.z);
    if listed.is_empty() {
        return;
    }
    let current = n.mass_number();
    ui.horizontal_wrapped(|ui| {
        ui.label("isotope").on_hover_text("only the mass changes: the Born-Oppenheimer surface (the forces) is the same for every isotope; at fixed velocity the kinetic energy changes, booked as work done by tools (E conserved stays continuous). Atomic masses, AME2020");
        for &a in listed {
            let mass = units::isotope_mass_amu_a(n.z, a).unwrap_or(f64::NAN);
            let text = format!("{} {mass:.6} u", units::isotopes::isotope_label(n.z, a));
            let r = ui.add(egui::Button::selectable(current == Some(a), text)).on_hover_text(format!("A = {a}: {} protons, {} neutrons", n.z, a.saturating_sub(n.z as u16)));
            if a == 2 && n.z == 1 {
                super::super::probe("selection-isotope-2h", &r);
            }
            if r.clicked() && current != Some(a) {
                acts.push(Action::SetIsotope { index: i, a });
            }
        }
    });
}

/// The selected nucleus's bonds (each pair's numbers on hover) and its molecule.
fn bonds(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, i: usize) {
    super::group(ui, st, "d-bonds", |ui, _| {
        theme::section(ui, "Bonds (read off the density)");
        let Some(s) = input.snapshot else { return };
        let d = &s.dynamics;
        let Some(g) = d.bonds.as_ref().filter(|g| g.n_nuclei == input.scene.nuclei.len()) else {
            if d.bonds_pending.is_empty() {
                theme::note(ui, "no bond graph yet: bonds are read off the density after a converged DFT step");
            } else {
                let pending: Vec<&str> = d.bonds_pending.iter().map(|&z| elements::symbol(z)).collect();
                theme::note(ui, format!("solving the free-atom references first ({}): the bonds appear when they are in", pending.join(", ")));
            }
            return;
        };
        let mine: Vec<&sim::PairReport> = d.pair_reports.iter().filter(|r| r.i == i || r.j == i).collect();
        let bonded: Vec<usize> = g.bonds.iter().filter_map(|b| if b.i == i { Some(b.j) } else if b.j == i { Some(b.i) } else { None }).collect();
        if bonded.is_empty() {
            theme::note(ui, "not bonded to anything");
        } else {
            let names: Vec<String> = bonded.iter().map(|&j| format!("{} #{j}", input.scene.nuclei.get(j).map_or("?", |n| elements::symbol(n.z)))).collect();
            ui.label(format!("bonded to {}", names.join(", ")));
        }
        for r in mine {
            let line = super::super::pair_line(input.scene, r);
            let bonded = matches!(r.verdict, sim::Verdict::Bonded { .. });
            ui.add(egui::Label::new(RichText::new(&line).size(11.0).color(if bonded { theme::INK } else { theme::INK_DIM })).wrap());
        }
        if let Some((k, _)) = g.components.iter().enumerate().find(|(_, c)| c.contains(&i)) {
            if let Some(name) = s.labels.get(k) {
                ui.label(super::super::glyphs(&format!("molecule {}", name.label()))).on_hover_text(super::super::glyphs(&name.tooltip()));
            }
        }
        if d.bonds_stale {
            theme::note(ui, super::super::bonds_label(d).unwrap_or_default());
        }
    });
}

/// The scene's molecules (formula — name), the unassigned net charge, the honesty lines.
fn molecules(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>) {
    super::group(ui, st, "d-molecules", |ui, _| {
        theme::section(ui, "Molecules");
        let Some(s) = input.snapshot else { return };
        let Some(g) = s.dynamics.bonds.as_ref().filter(|g| g.n_nuclei == input.scene.nuclei.len() && s.labels.len() == g.components.len()) else {
            theme::note(ui, "none yet: molecules are the connected pieces of the bond graph");
            return;
        };
        for (name, members) in s.labels.iter().zip(&g.components) {
            let who: Vec<String> = members.iter().map(|m| format!("#{m}")).collect();
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(super::super::glyphs(&name.label())).strong()).on_hover_text(super::super::glyphs(&name.tooltip()));
                ui.label(RichText::new(who.join(" ")).monospace().size(10.5).color(theme::INK_FAINT));
            });
        }
        if let Some(n) = &s.charge_note {
            theme::note(ui, n.as_str());
        }
        theme::note(ui, super::super::glyphs(crate::ui::menus::HONEST_BONDS));
    });
}

fn card_toggle(ui: &mut Ui, st: &mut UiState) {
    super::group(ui, st, "d-card", |ui, st| {
        theme::section(ui, "Atom card");
        ui.checkbox(&mut st.show_atom_card, "show the card in the 3D view");
        theme::note(ui, "In a molecule the electrons are shared: the cloud in the 3D view is the real picture. The simulation treats the nucleus as a point of charge +Z. The card is a schematic, not to scale.");
    });
}
