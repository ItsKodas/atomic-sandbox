//! The Selection panel (restructured 2026-11-14 from the user's approved mockup). Top to bottom:
//!
//! 1. **The atom card** (anchor `d-selection`): an element badge in the atom's display colour,
//!    "Hydrogen #0", a sub-line with the pseudopotential's short form and the isotope as a mass
//!    label that opens today's isotope picker (`Action::SetIsotope`), the partial charge as a
//!    coloured pill (faded when not current); below, an icon row — pin, focus, follow, delete,
//!    deselect — with today's tooltips and keys, the "Del · Esc" hint at its right.
//! 2. **NEIGHBOURS** (anchors `d-guides`, `d-bonds`): the distance guides and the bonds in one list
//!    sorted by distance ([`neighbours`]), each with a bond chip from the density's verdict
//!    ([`BondChip`]: covalent, weak contact, multi-centre, bond, no bond; mint "guessed" where the
//!    guess from experience draws the bond; faded while the bonds are an earlier step's), the bond
//!    angles at the atom indented under the pair they join, the honesty notes under the list.
//! 3. **The molecule line**: the molecule's name in the chosen style and its atoms.
//! 4. Folds ([`Sections`], `UiState::selection_sections`, the profile's `selection_sections`):
//!    **Details** (position, composition, electrons, the pseudopotential line and its generation
//!    record; an amber ⚠ on the header when the egg-box warning stands), **Scene molecules (n)**,
//!    **Partial charges** (Σq in its header).
//!
//! With the Edit tool's group (more than one atom): "3 atoms selected", their chips, the pairwise
//! distances as NEIGHBOURS, the last one clicked as a small line. With nothing selected: a short
//! empty state, then Scene molecules and Partial charges shown open. Display only but the actions
//! the row and the isotope picker emit, as before.
use super::super::{icons, probe_rect, theme, Action, PanelInput, UiState, AMBER};
use crate::atom_card;
use crate::tools;
use egui::{vec2, Color32, RichText, Stroke, Ui};
use render::elements;

/// An atom's name in the panels: its element symbol and its 0-based index in the scene ("O #0",
/// "H #1") — the Selection tab's charges and guides and the Orbitals tab's groups (orbital-atom-shares
/// spec, goal 2: the same names in both).
pub(crate) fn atom_name(z: u8, k: usize) -> String {
    format!("{} #{k}", elements::symbol(z))
}

/// One of the panel's folds (`UiState::selection_sections`). All closed by default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Section {
    Details,
    Molecules,
    Charges,
}

impl Section {
    pub const ALL: [Section; 3] = [Section::Details, Section::Molecules, Section::Charges];

    /// Its word in the profile's `selection_sections` line (its header's probe: `selection-fold-<id>`).
    pub fn id(self) -> &'static str {
        match self {
            Section::Details => "details",
            Section::Molecules => "molecules",
            Section::Charges => "charges",
        }
    }

    /// The search anchor its contents mark: a Ctrl K reveal unfolds it.
    pub fn anchor(self) -> &'static str {
        match self {
            Section::Details => "d-selection-details",
            Section::Molecules => "d-molecules",
            Section::Charges => "d-charges",
        }
    }
}

/// Which folds are open (`UiState::selection_sections`): none by default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sections([bool; 3]);

impl Sections {
    pub fn is_open(&self, s: Section) -> bool {
        self.0[s as usize]
    }

    pub fn set(&mut self, s: Section, open: bool) {
        self.0[s as usize] = open;
    }

    /// Every fold open (or none).
    #[cfg(test)]
    pub fn all(open: bool) -> Sections {
        Sections([open; 3])
    }

    /// The profile's value: the open folds' ids, comma separated.
    pub fn encode(&self) -> String {
        Section::ALL.into_iter().filter(|&s| self.is_open(s)).map(Section::id).collect::<Vec<_>>().join(",")
    }

    /// The profile's value read back (unknown words dropped).
    pub fn parse(v: &str) -> Sections {
        let mut s = Sections::default();
        for w in v.split(',').map(str::trim) {
            if let Some(x) = Section::ALL.into_iter().find(|x| x.id().eq_ignore_ascii_case(w)) {
                s.set(x, true);
            }
        }
        s
    }
}

/// The empty state's line (its tooltip is the old note).
pub const EMPTY: &str = "Click an atom to inspect it (Inspect: key 1, Edit: key 4)";

/// A fold's header (probed `selection-fold-<id>`); `forced` draws it open without a caret (nothing
/// selected: the molecules and the charges are the content then). Returns whether it is open.
fn fold_header(ui: &mut Ui, st: &mut UiState, s: Section, title: &str, summary: Option<(String, Color32)>, forced: bool) -> bool {
    if forced {
        let r = ui.horizontal(|ui| {
            theme::section(ui, title);
            if let Some((t, c)) = &summary {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| ui.label(RichText::new(t).size(11.0).color(*c)));
            }
        });
        probe_rect(&format!("selection-fold-{}", s.id()), r.response.rect);
        return true;
    }
    let (r, _) = theme::fold_header(ui, title, st.selection_sections.is_open(s), summary, None, false);
    probe_rect(&format!("selection-fold-{}", s.id()), r.rect);
    if r.clicked() {
        let open = st.selection_sections.is_open(s);
        st.selection_sections.set(s, !open);
    }
    st.selection_sections.is_open(s)
}

/// A card (the Model tab's verdict card look), probed as `probe`.
fn card<R>(ui: &mut Ui, probe: &str, body: impl FnOnce(&mut Ui) -> R) -> R {
    let r = egui::Frame::new().fill(theme::PANEL_2).stroke(Stroke::new(1.0, theme::LINE)).corner_radius(8).inner_margin(egui::Margin::symmetric(10, 8)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        body(ui)
    });
    probe_rect(probe, r.response.rect);
    r.inner
}

pub fn show(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    // the visible part of the panel (the tests' width checks)
    probe_rect("selection-panel", ui.clip_rect());
    if let Some(a) = st.search.unrevealed() {
        if let Some(s) = Section::ALL.into_iter().find(|s| s.anchor() == a) {
            st.selection_sections.set(s, true);
        }
    }
    let Some((i, n)) = input.selection() else {
        let r = ui.label(RichText::new(EMPTY).color(theme::INK_DIM)).on_hover_text("no selection (Inspect or Edit: click a nucleus)");
        probe_rect("selection-empty", r.rect);
        segment_row(ui, st, input, "d-segment", acts);
        molecules(ui, st, input, true);
        charges(ui, st, input, None, true);
        card_toggle(ui, st);
        return;
    };
    let group = st.edit_group.len() > 1;
    super::group(ui, st, "d-selection", |ui, st| {
        if group {
            group_card(ui, st, input, i, n, acts);
        } else {
            atom_card(ui, st, input, i, n, acts);
        }
    });
    segment_row(ui, st, input, "d-segment", acts);
    if group {
        group_distances(ui, st, input);
    } else {
        neighbours_card(ui, st, input, i, n);
        molecule_line(ui, st, input, i);
    }
    details(ui, st, input, n);
    molecules(ui, st, input, false);
    charges(ui, st, input, Some(i), false);
    card_toggle(ui, st);
}

/// The element badge: a disc in the atom's display colour with its symbol.
fn badge(ui: &mut Ui, z: u8, size: f32) -> egui::Response {
    let (rect, r) = ui.allocate_exact_size(vec2(size, size), egui::Sense::hover());
    let [cr, cg, cb] = elements::colour_srgb(z);
    let light = 0.299 * cr as f32 + 0.587 * cg as f32 + 0.114 * cb as f32 > 150.0;
    ui.painter().circle(rect.center(), size / 2.0, Color32::from_rgb(cr, cg, cb), Stroke::new(1.0, Color32::from_white_alpha(64)));
    ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, elements::symbol(z), egui::FontId::proportional(size * 0.45), if light { Color32::from_rgb(23, 17, 42) } else { Color32::WHITE });
    r
}

/// The partial charge's pill (today's colour rule and hover; faded when not current), if measured.
fn charge_pill(ui: &mut Ui, input: &PanelInput<'_>, i: usize) {
    let Some((c, current, q, counted)) = input.snapshot.and_then(|s| super::super::shown_charges(&s.dynamics)).and_then(|(c, cur)| Some((c, cur, *c.q.get(i)?, c.counted[i]))) else { return };
    let alpha = if current { 255 } else { 140 };
    let r = theme::pill(ui, &format!("{} e", super::super::charge_text(q)), super::super::charge_colour(q, alpha));
    probe_rect("selection-charge", r.rect);
    r.on_hover_text(format!(
        "{}\nHirshfeld: Z {counted} − {:.4} electrons in its share = {q:+.4} e{}",
        super::super::CHARGES_TIP,
        counted - q,
        if current { String::new() } else { format!(" (from t = {:.2} fs)", c.t_fs()) }
    ));
}

/// The pseudopotential's short form ("pseudo · Z_v 6" / "all-electron") for the sub-line.
fn pseudo_short(input: &PanelInput<'_>, z: u8) -> (String, String) {
    match input.snapshot.and_then(|s| super::super::pseudo::of_species(&s.dynamics.eval.pseudised, z)) {
        Some(p) => (format!("pseudo · Z_v {}", p.z_v), atom_card::pseudo_line(p)),
        None => ("all-electron".into(), "all-electron in the latest solve: every electron on the grid, the nucleus a smeared point charge".into()),
    }
}

/// 1. The atom card: badge, name, sub-line (pseudopotential, isotope picker), charge pill, the
/// icon row.
fn atom_card(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, i: usize, n: &crate::scene::SceneNucleus, acts: &mut Vec<Action>) {
    card(ui, "selection-card", |ui| {
        ui.horizontal(|ui| {
            badge(ui, n.z, 34.0).on_hover_text(atom_card::title(n));
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 2.0;
                ui.horizontal(|ui| {
                    let name = crate::names::element_name(n.z);
                    let mut c = name.chars();
                    let name = c.next().map_or(String::new(), |f| f.to_uppercase().collect::<String>() + c.as_str());
                    let r = ui.label(RichText::new(name).strong().size(14.0));
                    probe_rect("selection-name", r.rect);
                    ui.label(RichText::new(format!("#{i}")).monospace().color(theme::INK_DIM));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| charge_pill(ui, input, i));
                });
                ui.horizontal(|ui| {
                    let (short, long) = pseudo_short(input, n.z);
                    ui.label(RichText::new(short).size(11.0).color(theme::INK_DIM)).on_hover_text(long);
                    isotope_picker(ui, i, n, acts);
                });
            });
        });
        ui.add_space(2.0);
        action_row(ui, st, input, i, acts, "Del removes it · Esc deselects");
    });
}

/// The isotope as a mass label ("¹H ▾") opening today's picker (`Action::SetIsotope`).
fn isotope_picker(ui: &mut Ui, i: usize, n: &crate::scene::SceneNucleus, acts: &mut Vec<Action>) {
    if units::isotopes_of(n.z).is_empty() {
        return;
    }
    let current = n.mass_number();
    let label = match current {
        Some(a) => format!("{} ▾", units::isotopes::isotope_label(n.z, a)),
        None => "isotope ▾".into(),
    };
    let r = ui.add(egui::Button::new(RichText::new(label).size(11.5).color(theme::ACCENT_INK)).frame(false)).on_hover_text(
        "isotope: only the mass changes: the Born-Oppenheimer surface (the forces) is the same for every isotope; at fixed velocity the kinetic energy changes, booked as work done by tools (E conserved stays continuous). Atomic masses, AME2020",
    );
    probe_rect("selection-isotope", r.rect);
    egui::Popup::from_toggle_button_response(&r).id(egui::Id::new("selection-isotope-popup")).show(|ui| {
        ui.vertical(|ui| {
            if let Some(a) = super::super::isotope_buttons(ui, n.z, current, "selection-isotope").filter(|&a| current != Some(a)) {
                acts.push(Action::SetIsotope { index: i, a });
            }
        });
    });
}

/// The icon row: pin (a toggle; off when pinning is not available), focus, follow (a toggle),
/// delete (danger), deselect — today's actions and tooltips with their keys — and `hint` muted at
/// its right.
fn action_row(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, i: usize, acts: &mut Vec<Action>, hint: &str) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        let pinned = input.pinned(i);
        let r = icons::button(ui, icons::Icon::Pin, 22.0, pinned == Some(true), pinned.is_some()).on_hover_text("pinned: hold this nucleus still (velocity zeroed every step)");
        probe_rect("sel-pin", r.rect);
        if r.clicked() {
            acts.push(Action::Pin(i, !pinned.unwrap_or(false)));
        }
        let r = icons::button(ui, icons::Icon::Search, 22.0, false, true).on_hover_text("Focus selected: centre the camera on it");
        probe_rect("sel-focus", r.rect);
        if r.clicked() {
            acts.push(Action::Focus);
        }
        let r = icons::button(ui, icons::Icon::Camera, 22.0, st.follow_selected, true)
            .on_hover_text("follow: keep the camera centred on the selected atom as it moves (orbit and zoom still work; panning is overridden while on)");
        super::super::probe("follow-selected", &r);
        if r.clicked() {
            st.follow_selected = !st.follow_selected;
        }
        let r = ui.add(egui::Button::new(RichText::new("Delete").size(11.5).color(theme::DANGER_INK)).min_size(vec2(22.0, 22.0))).on_hover_text("Delete selected (Del): also the Delete key, also while playing");
        probe_rect("sel-delete", r.rect);
        if r.clicked() {
            acts.push(Action::Delete(i));
        }
        let r = icons::button(ui, icons::Icon::Close, 22.0, false, true).on_hover_text("Deselect (Esc)");
        probe_rect("sel-deselect", r.rect);
        if r.clicked() {
            acts.push(Action::Deselect);
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add(egui::Label::new(RichText::new(hint).size(10.5).color(theme::INK_FAINT)).truncate());
        });
    });
}

/// The Edit tool's group: "3 atoms selected", their chips, today's note as a hint, the icon row
/// (acting as before: Delete removes them all), and the last one clicked as a small line.
fn group_card(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, i: usize, n: &crate::scene::SceneNucleus, acts: &mut Vec<Action>) {
    card(ui, "selection-card", |ui| {
        let atoms: Vec<(u8, usize)> = st.edit_group.iter().filter_map(|&j| input.scene.nuclei.get(j).map(|m| (m.z, j))).collect();
        let r = ui.label(RichText::new(format!("{} atoms selected", atoms.len())).strong().size(14.0));
        probe_rect("selection-name", r.rect);
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = vec2(4.0, 4.0);
            for (z, j) in &atoms {
                theme::pill(ui, &atom_name(*z, *j), theme::INK_DIM);
            }
        });
        ui.add(egui::Label::new(RichText::new("the Edit tool's handles move them together (velocities kept); Delete removes them all").size(11.0).color(theme::INK_FAINT)).wrap());
        action_row(ui, st, input, i, acts, "Del removes them · Esc");
        let r = ui.label(RichText::new(format!("last clicked: {} · {}", atom_name(n.z, i), atom_card::title(n))).size(11.0).color(theme::INK_DIM));
        probe_rect("selection-last", r.rect);
    });
}

/// The pairwise distances among the group's atoms, nearest first.
pub fn pairwise(input: &PanelInput<'_>, group: &[usize]) -> Vec<(usize, usize, f64)> {
    let nuclei = &input.scene.nuclei;
    let mut v = Vec::new();
    for (a, &i) in group.iter().enumerate() {
        for &j in &group[a + 1..] {
            if let (Some(p), Some(q)) = (nuclei.get(i), nuclei.get(j)) {
                let d = (0..3).map(|k| (p.pos[k] - q.pos[k]).powi(2)).sum::<f64>().sqrt();
                v.push((i.min(j), i.max(j), d));
            }
        }
    }
    v.sort_by(|a, b| a.2.total_cmp(&b.2));
    v
}

fn group_distances(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>) {
    let group = st.edit_group.clone();
    super::group(ui, st, "d-guides", |ui, _| {
        card(ui, "selection-neighbours", |ui| {
            ui.label(theme::caps("Distances in the group"));
            super::wide_grid(ui, "group-distances", 2).striped(true).show(ui, |ui| {
                for (k, (i, j, d)) in pairwise(input, &group).into_iter().enumerate() {
                    let name = |m: usize| input.scene.nuclei.get(m).map_or(format!("? #{m}"), |n| atom_name(n.z, m));
                    let r = ui.label(format!("{}–{}", name(i), name(j)));
                    probe_rect(&format!("selection-pair-{k}"), r.rect);
                    ui.label(RichText::new(format!("{d:.2} Bohr · {:.2} Å", units::bohr_to_angstrom(d))).monospace().size(11.0).color(theme::INK_DIM));
                    ui.end_row();
                }
            });
        });
    });
}

/// A neighbour's bond chip, from the density's verdict for the pair (`DynSnapshot::pair_reports`)
/// and the guess from experience (`PanelInput::experience_bonds`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BondChip {
    /// Bonded, the Laplacian's kind: shared.
    Covalent,
    /// Bonded, closed-shell.
    WeakContact,
    /// Bonded, a multi-centre bond (a ring).
    MultiCentre,
    /// Bonded, the kind not resolved (or not computed).
    Bond,
    /// The density says no bond (or no pair was examined: farther than the bond search reaches).
    NoBond,
    /// The guess from experience draws a bond here (the bonds are an earlier step's).
    Guessed,
}

impl BondChip {
    pub fn text(self) -> &'static str {
        match self {
            BondChip::Covalent => "covalent",
            BondChip::WeakContact => "weak contact",
            BondChip::MultiCentre => "multi-centre",
            BondChip::Bond => "bond",
            BondChip::NoBond => "no bond",
            BondChip::Guessed => "guessed",
        }
    }

    fn colour(self) -> Color32 {
        match self {
            BondChip::NoBond => theme::INK_FAINT,
            BondChip::Guessed => {
                let [r, g, b] = crate::overlay::EXPERIENCE_TINT;
                Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
            }
            BondChip::WeakContact => theme::INK_DIM,
            _ => theme::ACCENT_INK,
        }
    }
}

/// One neighbour row: the other atom, the distance, the chip (`None`: no bond graph yet).
#[derive(Clone, Debug, PartialEq)]
pub struct Neighbour {
    pub index: usize,
    pub distance: f64,
    pub chip: Option<BondChip>,
}

/// The neighbours of atom `i`: the distance guides (within `tools::GUIDE_RADIUS`, at most
/// `tools::GUIDE_MAX`) and every atom it is bonded to, one row each, nearest first, each with its
/// chip.
pub fn neighbours(input: &PanelInput<'_>, i: usize) -> Vec<Neighbour> {
    let nuclei = &input.scene.nuclei;
    let Some(me) = nuclei.get(i) else { return Vec::new() };
    let mut rows: Vec<(usize, f64)> = tools::distance_guides(me.pos, nuclei, Some(i)).into_iter().map(|g| (g.index, g.distance)).collect();
    let d = input.snapshot.map(|s| &s.dynamics);
    let graph = d.and_then(|d| d.bonds.as_ref().filter(|g| g.n_nuclei == nuclei.len()));
    if let Some(g) = graph {
        for b in g.bonds.iter().filter(|b| b.i == i || b.j == i) {
            let j = if b.i == i { b.j } else { b.i };
            if !rows.iter().any(|r| r.0 == j) {
                if let Some(q) = nuclei.get(j) {
                    rows.push((j, (0..3).map(|k| (q.pos[k] - me.pos[k]).powi(2)).sum::<f64>().sqrt()));
                }
            }
        }
    }
    rows.sort_by(|a, b| a.1.total_cmp(&b.1));
    rows.into_iter()
        .map(|(j, distance)| {
            let chip = graph.map(|g| {
                let key = (i.min(j), i.max(j));
                if input.experience_bonds.contains(&key) {
                    return BondChip::Guessed;
                }
                let bonded = g.bonds.iter().any(|b| (b.i.min(b.j), b.i.max(b.j)) == key);
                let report = d.and_then(|d| d.pair_reports.iter().find(|r| (r.i.min(r.j), r.i.max(r.j)) == key));
                if !bonded {
                    return BondChip::NoBond;
                }
                match report.map(|r| (r.verdict, r.kind())) {
                    Some((sim::Verdict::Bonded { multi_centre: true }, _)) => BondChip::MultiCentre,
                    Some((_, Some(sim::BondKind::Shared))) => BondChip::Covalent,
                    Some((_, Some(sim::BondKind::ClosedShell))) => BondChip::WeakContact,
                    _ => BondChip::Bond,
                }
            });
            Neighbour { index: j, distance, chip }
        })
        .collect()
}

/// 2. NEIGHBOURS: the merged guides and bonds, the angles under their pairs, the notes.
fn neighbours_card(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, i: usize, _n: &crate::scene::SceneNucleus) {
    let rows = neighbours(input, i);
    let r = ui.scope(|ui| {
        card(ui, "selection-neighbours", |ui| {
            ui.label(theme::caps(&format!("Neighbours · within {} Bohr", tools::GUIDE_RADIUS)));
            let Some(s) = input.snapshot else {
                rows_only(ui, input, &rows, None);
                return;
            };
            let d = &s.dynamics;
            let graph = d.bonds.as_ref().filter(|g| g.n_nuclei == input.scene.nuclei.len());
            if rows.is_empty() {
                theme::note(ui, format!("no other nucleus within {} Bohr", tools::GUIDE_RADIUS));
            }
            let pos: Vec<[f64; 3]> = input.scene.nuclei.iter().map(|n| n.pos).collect();
            let zs: Vec<u8> = input.scene.nuclei.iter().map(|n| n.z).collect();
            let pseudised = super::super::pseudo::pseudised(input);
            let pseudised_z: Vec<u8> = pseudised.iter().map(|p| p.z).collect();
            let r_c_max_of = |z: u8| super::super::pseudo::of_species(&pseudised, z).map(|p| p.r_c_max);
            let angles: Vec<crate::overlay::AngleMark> = graph.map(|g| crate::overlay::bond_angles(&pos, g).into_iter().filter(|m| m.vertex == i).collect()).unwrap_or_default();
            let at = |k: usize| rows.iter().position(|r| r.index == k);
            let mut notes: Vec<&'static str> = Vec::new();
            let mut listed = false;
            let mut shown_angles = vec![false; angles.len()];
            for (row_k, nb) in rows.iter().enumerate() {
                let key = (i.min(nb.index), i.max(nb.index));
                let report = d.pair_reports.iter().find(|r| (r.i.min(r.j), r.i.max(r.j)) == key);
                let valence = report.is_some_and(|r| matches!(r.verdict, sim::Verdict::Bonded { .. }) && r.kind().is_some() && r.s.is_some_and(|s| super::super::valence_density_kind(&pos, &zs, r.i, r.j, s, r_c_max_of)));
                if report.is_some_and(|r| matches!(r.verdict, sim::Verdict::Bonded { .. }) && r.kind().is_some()) {
                    listed = true;
                }
                neighbour_row(ui, input, row_k, nb, report, valence, d.bonds_stale.then(|| super::super::bonds_label(d)).flatten());
                // the angles whose both neighbours are now listed
                for (a, m) in angles.iter().enumerate() {
                    if shown_angles[a] {
                        continue;
                    }
                    if let (Some(pa), Some(pb)) = (at(m.a), at(m.b)) {
                        if pa.max(pb) == row_k {
                            shown_angles[a] = true;
                            angle_row(ui, input, m, &pseudised_z, &zs, &mut notes);
                        }
                    }
                }
            }
            for (a, m) in angles.iter().enumerate() {
                if !shown_angles[a] {
                    angle_row(ui, input, m, &pseudised_z, &zs, &mut notes);
                }
            }
            // the honesty lines, under the list
            if graph.is_none() {
                if d.bonds_pending.is_empty() {
                    theme::note(ui, "no bond graph yet: bonds are read off the density after a converged DFT step");
                } else {
                    let pending: Vec<&str> = d.bonds_pending.iter().map(|&z| elements::symbol(z)).collect();
                    theme::note(ui, format!("solving the free-atom references first ({}): the bonds appear when they are in", pending.join(", ")));
                }
            }
            // whenever a bond with a kind is listed: the user's own stretched H₃ contact is shared-shell while bonded (review r1)
            if listed {
                small_note(ui, super::super::LSDA_STRETCHED_NOTE);
            }
            for note in notes {
                small_note(ui, note);
            }
            if d.bonds_stale {
                small_note(ui, &input.stale_bonds_line.clone().or_else(|| super::super::bonds_label(d)).unwrap_or_default());
            }
        });
    });
    // the guides' and the bonds' search anchors (one card)
    super::super::search::mark(ui, st, "d-guides", r.response.rect);
    super::super::search::mark(ui, st, "d-bonds", r.response.rect);
}

fn small_note(ui: &mut Ui, text: &str) {
    ui.add(egui::Label::new(RichText::new(text).size(10.5).color(theme::INK_FAINT)).wrap());
}

/// The rows without a snapshot (the guides only, no chip).
fn rows_only(ui: &mut Ui, input: &PanelInput<'_>, rows: &[Neighbour], stale: Option<String>) {
    if rows.is_empty() {
        theme::note(ui, format!("no other nucleus within {} Bohr", tools::GUIDE_RADIUS));
    }
    for (k, nb) in rows.iter().enumerate() {
        neighbour_row(ui, input, k, nb, None, false, stale.clone());
    }
}

/// One neighbour: its name (+ " pinned"), "1.44 Bohr · 0.76 Å", its chip (faded while the bonds
/// are an earlier step's) with the pair's numbers on hover.
fn neighbour_row(ui: &mut Ui, input: &PanelInput<'_>, k: usize, nb: &Neighbour, report: Option<&sim::PairReport>, valence: bool, stale: Option<String>) {
    let other = &input.scene.nuclei[nb.index];
    let r = ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        let pin = if input.pinned(nb.index) == Some(true) { " pinned" } else { "" };
        ui.label(RichText::new(format!("{}{pin}", atom_name(other.z, nb.index))).size(12.0));
        ui.label(RichText::new(format!("{:.2} Bohr · {:.2} Å", nb.distance, units::bohr_to_angstrom(nb.distance))).monospace().size(11.0).color(theme::INK_DIM));
        if let Some(chip) = nb.chip {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let c = chip.colour();
                let c = if stale.is_some() && chip != BondChip::Guessed { c.gamma_multiply(0.5) } else { c };
                let mut tip = match report {
                    Some(r) => super::super::pair_line(input.scene, r),
                    None => format!("not a bond — farther apart than the bond search reaches ({} Bohr)", sim::bonds::R_MAX),
                };
                if let (Some(kind), Some(band), true) = (report.and_then(|r| r.kind()), report.and_then(|r| r.laplacian_noise), matches!(chip, BondChip::Covalent | BondChip::WeakContact | BondChip::Bond)) {
                    tip += &format!(
                        "\n{}\nThe kind is the sign of the Laplacian of the density at the bond point (QTAIM; Bader & Essén 1984): negative, charge concentrated between the nuclei — shared (covalent); positive, depleted — closed-shell (weak, stretched or ionic-like); within ±{band:.3} e/Bohr⁵ (3× its estimated grid noise) — not resolved.\n{}",
                        super::super::kind_words(kind),
                        super::super::LSDA_STRETCHED_NOTE
                    );
                }
                if valence {
                    tip += &format!("\n{}", super::super::VALENCE_KIND_NOTE);
                }
                if chip == BondChip::Guessed {
                    tip = format!("guessed from experience: pairs seen before with these elements and surroundings were bonded (View ▸ Bonds and labels ▸ guess bonds from experience; display only). The last DFT step's verdict: {tip}");
                }
                if let Some(s) = &stale {
                    tip = format!("{s}\n{tip}");
                }
                let r = egui::Frame::new()
                    .fill(c.gamma_multiply(0.12))
                    .stroke(Stroke::new(1.0, c.gamma_multiply(0.6)))
                    .corner_radius(255)
                    .inner_margin(egui::Margin::symmetric(6, 0))
                    .show(ui, |ui| ui.label(RichText::new(chip.text()).size(10.5).color(c)))
                    .response
                    .on_hover_text(tip);
                probe_rect(&format!("selection-chip-{k}"), r.rect);
            });
        }
    });
    probe_rect(&format!("selection-neighbour-{k}"), r.response.rect);
}

/// An angle at the atom, indented and muted, with the surface it is measured on.
fn angle_row(ui: &mut Ui, input: &PanelInput<'_>, m: &crate::overlay::AngleMark, pseudised_z: &[u8], zs: &[u8], notes: &mut Vec<&'static str>) {
    let sym = |k: usize| input.scene.nuclei.get(k).map_or("?", |n| elements::symbol(n.z));
    let note = super::super::angle_surface_note(pseudised_z, [zs[m.a], zs[m.vertex], zs[m.b]]);
    if !notes.contains(&note) {
        notes.push(note);
    }
    ui.horizontal(|ui| {
        ui.add_space(16.0);
        ui.label(RichText::new(format!("∠{}–{}–{} {}", sym(m.a), sym(m.vertex), sym(m.b), crate::overlay::angle_label(m))).size(11.0).color(theme::INK_DIM))
            .on_hover_text(format!("the angle between the bonds to #{} and #{} at #{} — geometry of the nuclear positions only; {note}", m.a, m.b, m.vertex));
    });
}

/// 3. The molecule the atom belongs to: its name in the chosen style (tooltip as before) and its
/// atoms.
fn molecule_line(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, i: usize) {
    let Some(s) = input.snapshot else { return };
    let Some(g) = s.dynamics.bonds.as_ref().filter(|g| g.n_nuclei == input.scene.nuclei.len()) else { return };
    let Some((k, members)) = g.components.iter().enumerate().find(|(_, c)| c.contains(&i)) else { return };
    let Some(name) = s.labels.get(k) else { return };
    let r = ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        ui.label(super::super::rich(&name.label_in(st.name_style), 13.0, theme::INK)).on_hover_text(super::super::rich(&name.tooltip_in(st.name_style), 12.5, theme::INK));
        for &m in members {
            if let Some(n) = input.scene.nuclei.get(m) {
                let text = RichText::new(atom_name(n.z, m)).monospace().size(10.5).color(if m == i { theme::INK } else { theme::INK_FAINT });
                ui.label(text);
            }
        }
    });
    probe_rect("selection-molecule", r.response.rect);
}

/// 4. Details: position, composition, electrons, the pseudopotential line and its generation
/// record (an amber ⚠ on the header while the egg-box warning stands).
fn details(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, n: &crate::scene::SceneNucleus) {
    let pseudo = input.snapshot.and_then(|s| super::super::pseudo::of_species(&s.dynamics.eval.pseudised, n.z));
    let egg = pseudo.and_then(|p| super::super::pseudo::egg_box_warning(std::slice::from_ref(p), input.scene.grid_h));
    let summary = if egg.is_some() { Some(("⚠ egg-box".to_string(), AMBER)) } else { Some((format!("({:.2}, {:.2}, {:.2}) Bohr", n.pos[0], n.pos[1], n.pos[2]), theme::INK_FAINT)) };
    if !fold_header(ui, st, Section::Details, "Details", summary, false) {
        return;
    }
    super::group(ui, st, "d-selection-details", |ui, _| {
        ui.indent("selection-details", |ui| {
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
            if let Some(p) = pseudo {
                ui.label(RichText::new(atom_card::pseudo_line(p)).size(11.0).color(theme::INK_DIM));
                egui::CollapsingHeader::new("pseudopotential (generation record)").id_salt("pseudo-record").default_open(false).show(ui, |ui| {
                    ui.label(RichText::new(&p.record).monospace().size(10.5).color(theme::INK_DIM));
                });
                if let Some(w) = &egg {
                    ui.add(egui::Label::new(RichText::new(w).size(11.0).color(AMBER)).wrap()).on_hover_text(super::super::pseudo::EGG_BOX_RECORD);
                }
            }
        });
    });
}

/// The atoms "Train this segment" cuts out: the Edit tool's group, else the selected nucleus.
pub fn segment_atoms(st: &UiState, input: &PanelInput<'_>) -> Vec<usize> {
    if st.edit_group.is_empty() {
        input.selection().map(|(i, _)| vec![i]).unwrap_or_default()
    } else {
        st.edit_group.clone()
    }
}

/// What the button does, on hover.
pub const TRAIN_SEGMENT_TIP: &str = "teach the shared model the selected atoms' surroundings without solving the whole scene: the selection and every atom within twice the model's cutoff (plus the sampling pass's largest planned move) are cut out, each cut bond capped with an H (placed with the molecule builder's tables), and opened as a scene of its own in the estimated solver with the Standard plan ready. Only atoms whose whole neighbourhood is this scene's are taught; the caps and the edge atoms are not. The current scene is saved, and a chip by the scene chip brings you back. A preview with the counts, the box and the GPU memory comes first.";

/// "Train this segment" (M8a Task 10; Dock ▸ Selection and the Model tab): `Action::TrainSegment`
/// with [`segment_atoms`]; disabled, with the reason, with nothing selected or during a pass.
/// Not drawn while the feature is deferred ([`crate::segment::SEGMENT_TRAINING`]).
pub fn segment_row(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, anchor: &str, acts: &mut Vec<Action>) {
    if !crate::segment::SEGMENT_TRAINING {
        return;
    }
    let atoms = segment_atoms(st, input);
    let sampling = input.snapshot.is_some_and(|s| crate::sim::sampling_active(&s.dynamics));
    super::group(ui, st, anchor, |ui, _| {
        ui.horizontal_wrapped(|ui| {
            let b = ui
                .add_enabled(!atoms.is_empty() && !sampling, egui::Button::new("Train this segment"))
                .on_disabled_hover_text(if sampling { "a sampling pass is running: let it finish (or Cancel it) first" } else { crate::segment::NOTHING_SELECTED })
                .on_hover_text(TRAIN_SEGMENT_TIP);
            if super::super::probed(if anchor == "d-segment" { "train-segment" } else { "train-segment-model" }, b).clicked() {
                acts.push(Action::TrainSegment(atoms.clone()));
            }
            let what = match atoms.len() {
                0 => "no selection".to_string(),
                1 => "1 atom selected".to_string(),
                n => format!("{n} atoms selected"),
            };
            ui.label(RichText::new(what).size(11.0).color(theme::INK_FAINT));
        });
    });
}

/// Partial charges: every atom's (`sim::charges`, Hirshfeld; the selected one marked), their sum
/// against the scene's charge (in the header), the measurement's time and the honesty line.
/// `open`: nothing selected, shown open.
fn charges(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, selected: Option<usize>, open: bool) {
    let d = input.snapshot.map(|s| &s.dynamics);
    let shown = d.and_then(super::super::shown_charges);
    let summary = shown.map(|(c, _)| (format!("Σq {:+.4} e", c.sum()), theme::INK_FAINT));
    if !fold_header(ui, st, Section::Charges, "Partial charges", summary, open) {
        return;
    }
    super::group(ui, st, "d-charges", |ui, _| {
        ui.indent("selection-charges", |ui| {
            ui.label(RichText::new("Hirshfeld").size(11.0).color(theme::INK_DIM)).on_hover_text(super::super::CHARGES_TIP);
            let Some(d) = d else { return };
            let Some((c, current)) = shown else {
                let why = match d.charges.as_deref() {
                    Some(c) if c.fits(&d.scene) && !c.balanced() => format!("not shown: they add up to {:+.4} e against the scene's charge {:+} (the density does not hold the solve's electrons)", c.sum(), c.system_charge),
                    _ if !d.bonds_pending.is_empty() => "solving the free-atom references first: the charges appear with the bonds".into(),
                    _ => "none yet: measured from the density after a converged DFT step".into(),
                };
                theme::note(ui, why);
                return;
            };
            let alpha = if current { 255 } else { 140 };
            super::wide_grid(ui, "charges", 3).striped(true).show(ui, |ui| {
                for (k, (z, q)) in c.z.iter().zip(&c.q).enumerate() {
                    let who = atom_name(*z, k);
                    ui.label(if selected == Some(k) { RichText::new(who).strong() } else { RichText::new(who) });
                    ui.label(RichText::new(super::super::charge_text(*q)).monospace().color(super::super::charge_colour(*q, alpha)));
                    ui.label(RichText::new(format!("Z {}", c.counted[k])).size(10.5).color(theme::INK_FAINT))
                        .on_hover_text("the charge its electrons are counted against: Z, or the valence charge Z_v of a pseudopotential atom");
                    ui.end_row();
                }
            });
            ui.label(RichText::new(format!("Σq = {:+.4} e (scene charge {:+}) · {:.4} electrons", c.sum(), c.system_charge, c.electrons)).size(11.0).color(theme::INK_DIM))
                .on_hover_text("the weights add up to one at every point, so the charges add up to the scene's charge exactly (to rounding): a check that the density holds the solve's electrons");
            if !current {
                theme::note(ui, super::super::charges_label(d).unwrap_or_default());
            }
            theme::note(ui, "one partition scheme of several (Hirshfeld magnitudes are small): read the signs and trends; measured from this LSDA density — Help ▸ Partial charges");
        });
    });
}

/// Scene molecules: formula — name, each with its atoms, the unassigned net charge, the honesty
/// lines. `open`: nothing selected, shown open.
fn molecules(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, open: bool) {
    let s = input.snapshot;
    let g = s.and_then(|s| s.dynamics.bonds.as_ref().filter(|g| g.n_nuclei == input.scene.nuclei.len() && s.labels.len() == g.components.len()));
    let title = match g {
        Some(g) => format!("Scene molecules ({})", g.components.len()),
        None => "Scene molecules".to_string(),
    };
    if !fold_header(ui, st, Section::Molecules, &title, None, open) {
        return;
    }
    super::group(ui, st, "d-molecules", |ui, st| {
        ui.indent("selection-molecules", |ui| {
            let (Some(s), Some(g)) = (s, g) else {
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
    });
}

fn card_toggle(ui: &mut Ui, st: &mut UiState) {
    ui.add_space(4.0);
    super::group(ui, st, "d-card", |ui, st| {
        ui.checkbox(&mut st.show_atom_card, RichText::new("show the atom card in the 3D view").size(11.0).color(theme::INK_DIM))
            .on_hover_text("In a molecule the electrons are shared: the cloud in the 3D view is the real picture. The simulation treats the nucleus as a point of charge +Z. The card is a schematic, not to scale.");
    });
}
