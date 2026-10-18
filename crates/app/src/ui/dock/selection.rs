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
        super::wide_grid(ui, "selection", 2).striped(true).show(ui, |ui| {
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
        pseudopotential(ui, input, n.z);
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
        super::wide_grid(ui, "guides", 3).striped(true).show(ui, |ui| {
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

/// A pseudised species' record (M7): what the generator chose and measured for it (`qm::radial::GenerationRecord`),
/// collapsed; nothing when the species is all-electron in the latest solve.
fn pseudopotential(ui: &mut Ui, input: &PanelInput<'_>, z: u8) {
    let Some(p) = input.snapshot.and_then(|s| super::super::pseudo::of_species(&s.dynamics.eval.pseudised, z)) else { return };
    ui.label(RichText::new(atom_card::pseudo_line(p)).size(11.0).color(theme::INK_DIM));
    egui::CollapsingHeader::new("pseudopotential (generation record)")
        .id_salt("pseudo-record")
        .default_open(false)
        .show(ui, |ui| {
            ui.label(RichText::new(&p.record).monospace().size(10.5).color(theme::INK_DIM));
            if let Some(w) = super::super::pseudo::egg_box_warning(std::slice::from_ref(p), input.scene.grid_h) {
                ui.label(RichText::new(w).size(11.0).color(super::super::AMBER)).on_hover_text(super::super::pseudo::EGG_BOX_RECORD);
            }
        });
}

/// The selected nucleus's isotope: the listed mass numbers of its Z with their masses
/// (`Action::SetIsotope`; mass only — the forces are the same for every isotope).
fn isotopes(ui: &mut Ui, i: usize, n: &crate::scene::SceneNucleus, acts: &mut Vec<Action>) {
    if units::isotopes_of(n.z).is_empty() {
        return;
    }
    let current = n.mass_number();
    ui.horizontal_wrapped(|ui| {
        ui.label("isotope").on_hover_text("only the mass changes: the Born-Oppenheimer surface (the forces) is the same for every isotope; at fixed velocity the kinetic energy changes, booked as work done by tools (E conserved stays continuous). Atomic masses, AME2020");
        if let Some(a) = super::super::isotope_buttons(ui, n.z, current, "selection-isotope").filter(|&a| current != Some(a)) {
            acts.push(Action::SetIsotope { index: i, a });
        }
    });
}

/// The selected nucleus's bonds (each pair's numbers on hover) and its molecule.
fn bonds(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, i: usize) {
    super::group(ui, st, "d-bonds", |ui, st| {
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
        let pos: Vec<[f64; 3]> = input.scene.nuclei.iter().map(|n| n.pos).collect();
        let zs: Vec<u8> = input.scene.nuclei.iter().map(|n| n.z).collect();
        // the scene's pseudised species (the latest solve's `EvalStats.pseudised`, else those already
        // generated for it: `pseudo::pseudised`): which angles sit on the pseudopotential surface, and
        // which bond points lie inside a core radius
        let pseudised = super::super::pseudo::pseudised(input);
        let pseudised_z: Vec<u8> = pseudised.iter().map(|p| p.z).collect();
        let r_c_max_of = |z: u8| super::super::pseudo::of_species(&pseudised, z).map(|p| p.r_c_max);
        let mut listed = false;
        for r in mine {
            let mut line = super::super::pair_line(input.scene, r);
            let bonded = matches!(r.verdict, sim::Verdict::Bonded { .. });
            // plan review Q10: a bond point inside a pseudised atom's r_c has a valence-density kind
            let valence = bonded && r.kind().is_some() && r.s.is_some_and(|s| super::super::valence_density_kind(&pos, &zs, r.i, r.j, s, r_c_max_of));
            if valence {
                line.push_str(" (valence-density kind)");
            }
            let label = ui.add(egui::Label::new(RichText::new(&line).size(11.0).color(if bonded { theme::INK } else { theme::INK_DIM })).wrap());
            if let (true, Some(kind), Some(band)) = (bonded, r.kind(), r.laplacian_noise) {
                listed = true;
                let valence_note = if valence { format!("\n{}", super::super::VALENCE_KIND_NOTE) } else { String::new() };
                label.on_hover_text(format!(
                    "{}\nThe kind is the sign of the Laplacian of the density at the bond point (QTAIM; Bader & Essén 1984): negative, charge concentrated between the nuclei — shared (covalent); positive, depleted — closed-shell (weak, stretched or ionic-like); within ±{band:.3} e/Bohr⁵ (3× its estimated grid noise) — not resolved. The line in the view is solid, dashed or dotted accordingly, thinner and fainter for a smaller ρ_b.\n{}{valence_note}",
                    super::super::kind_words(kind),
                    super::super::LSDA_STRETCHED_NOTE
                ));
            }
        }
        // whenever a bond is listed, not only a non-shared one: the user's own stretched H₃ contact
        // is shared-shell while bonded (review r1)
        if listed {
            theme::note(ui, super::super::LSDA_STRETCHED_NOTE);
        }
        // the bond angles at this atom (M7 Task 13): geometry only, whatever the declutter rule shows
        // in the view; each tagged with the surface it is measured on
        let sym = |k: usize| input.scene.nuclei.get(k).map_or("?", |n| elements::symbol(n.z));
        let mut notes: Vec<&'static str> = Vec::new();
        for m in crate::overlay::bond_angles(&pos, g).iter().filter(|m| m.vertex == i) {
            let note = super::super::angle_surface_note(&pseudised_z, [zs[m.a], zs[m.vertex], zs[m.b]]);
            if !notes.contains(&note) {
                notes.push(note);
            }
            ui.label(RichText::new(format!("∠{}–{}–{} {}", sym(m.a), sym(m.vertex), sym(m.b), crate::overlay::angle_label(m))).size(11.0).color(theme::INK))
                .on_hover_text(format!("the angle between the bonds to #{} and #{} at #{} — geometry of the nuclear positions only; {note}", m.a, m.b, m.vertex));
        }
        for note in notes {
            theme::note(ui, note);
        }
        if let Some((k, _)) = g.components.iter().enumerate().find(|(_, c)| c.contains(&i)) {
            if let Some(name) = s.labels.get(k) {
                ui.label(super::super::rich(&format!("molecule {}", name.label_in(st.name_style)), 13.0, theme::INK)).on_hover_text(super::super::rich(&name.tooltip_in(st.name_style), 12.5, theme::INK));
            }
        }
        if d.bonds_stale {
            theme::note(ui, super::super::bonds_label(d).unwrap_or_default());
        }
    });
}

/// The scene's molecules (formula — name), the unassigned net charge, the honesty lines.
fn molecules(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>) {
    super::group(ui, st, "d-molecules", |ui, st| {
        theme::section(ui, "Molecules");
        let Some(s) = input.snapshot else { return };
        let Some(g) = s.dynamics.bonds.as_ref().filter(|g| g.n_nuclei == input.scene.nuclei.len() && s.labels.len() == g.components.len()) else {
            theme::note(ui, "none yet: molecules are the connected pieces of the bond graph");
            return;
        };
        for (name, members) in s.labels.iter().zip(&g.components) {
            let who: Vec<String> = members.iter().map(|m| format!("#{m}")).collect();
            ui.horizontal_wrapped(|ui| {
                ui.label(super::super::rich(&name.label_in(st.name_style), 13.5, egui::Color32::WHITE)).on_hover_text(super::super::rich(&name.tooltip_in(st.name_style), 12.5, theme::INK));
                ui.label(RichText::new(who.join(" ")).monospace().size(10.5).color(theme::INK_FAINT));
            });
        }
        if let Some(n) = &s.charge_note {
            theme::note(ui, n.as_str());
        }
        ui.label(super::super::rich(crate::ui::menus::HONEST_BONDS, 11.0, theme::INK_DIM));
    });
}

fn card_toggle(ui: &mut Ui, st: &mut UiState) {
    super::group(ui, st, "d-card", |ui, st| {
        theme::section(ui, "Atom card");
        ui.checkbox(&mut st.show_atom_card, "show the card in the 3D view");
        theme::note(ui, "In a molecule the electrons are shared: the cloud in the 3D view is the real picture. The simulation treats the nucleus as a point of charge +Z. The card is a schematic, not to scale.");
    });
}
