//! The View panel (docking spec D-6; user decision 1, "B's inspector", and the panel mockup the
//! user approved 2026-11-03): every control of the old View menu, which was "such a mess" (user,
//! 2026-10-21), regrouped into five folding sections in the order one reaches for them — What you
//! see, How it looks, Camera, Bonds & labels, Display aids — then Reset view settings.
//!
//! - A control that does not apply is hidden, not greyed: the volume's gamma and opacity only while
//!   rendering as a volume, the iso level only as an isosurface, the dots' look only as dots, the
//!   dots' orbital colouring only while All orbitals coloured is shown, the Camera view's settings
//!   only while it is on, the estimated solver's cloud options only under that solver.
//! - Each section is a folding header (probe and id `view-<id>`) whose open state is display state
//!   kept in the profile (`view_sections = what,looks`, [`Sections`]); What you see and How it
//!   looks start open. A Ctrl K reveal of a control here unfolds its section (it stays open).
//! - Each section is one search anchor (`m-density`, `m-looks`, `m-camera`, `m-bonds`,
//!   `m-display`): the View menu's `m-density` group was split in two, its look half becoming
//!   `m-looks`; every widget keeps its test probe name.
//!
//! Like the menu it replaces, this is display state: the controls that were `Action`s (show, render
//! as, partial core, the cloud, bonds and labels, name style, smooth motion) still are — recorded,
//! some of them sent — and the rest write `UiState` directly.
use super::super::menus::{ANGLES_TIP, BONDS_TIP, HONEST_BONDS, NAME_STYLE_TIP};
use super::super::{knowledge, probe_rect, probed, pseudo, rich, theme, Action, ShowMode, UiState, CHARGES_TIP, VOLUME_DOTS, VOLUME_ISO, VOLUME_VOLUME};
use super::orbitals;
use crate::sim::Backend;
use egui::{pos2, vec2, Sense, Shape, Stroke, Ui};

/// One of the panel's folding sections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Section {
    What,
    Looks,
    Camera,
    Bonds,
    Aids,
}

impl Section {
    pub const ALL: [Section; 5] = [Section::What, Section::Looks, Section::Camera, Section::Bonds, Section::Aids];

    /// Its word in the profile's `view_sections` line (its header's probe: `view-<id>`).
    pub fn id(self) -> &'static str {
        match self {
            Section::What => "what",
            Section::Looks => "looks",
            Section::Camera => "camera",
            Section::Bonds => "bonds",
            Section::Aids => "aids",
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            Section::What => "What you see",
            Section::Looks => "How it looks",
            Section::Camera => "Camera",
            Section::Bonds => "Bonds & labels",
            Section::Aids => "Display aids",
        }
    }

    /// The search anchor its contents mark (`search::ENTRIES`).
    pub fn anchor(self) -> &'static str {
        match self {
            Section::What => "m-density",
            Section::Looks => "m-looks",
            Section::Camera => "m-camera",
            Section::Bonds => "m-bonds",
            Section::Aids => "m-display",
        }
    }
}

/// Which sections are open (`UiState::view_sections`): What you see and How it looks by default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sections([bool; 5]);

impl Default for Sections {
    fn default() -> Self {
        Sections([true, true, false, false, false])
    }
}

impl Sections {
    pub fn is_open(&self, s: Section) -> bool {
        self.0[s as usize]
    }

    pub fn set(&mut self, s: Section, open: bool) {
        self.0[s as usize] = open;
    }

    /// Every section open (or none).
    #[cfg(test)]
    pub fn all(open: bool) -> Sections {
        Sections([open; 5])
    }

    /// The profile's value: the open sections' ids, comma separated, in panel order.
    pub fn encode(&self) -> String {
        Section::ALL.into_iter().filter(|&s| self.is_open(s)).map(Section::id).collect::<Vec<_>>().join(",")
    }

    /// The profile's value read back: the ids named are open, the rest closed; unknown words are
    /// dropped, case and spaces ignored.
    pub fn parse(v: &str) -> Sections {
        let mut s = Sections([false; 5]);
        for w in v.split(',').map(str::trim) {
            if let Some(x) = Section::ALL.into_iter().find(|x| x.id().eq_ignore_ascii_case(w)) {
                s.set(x, true);
            }
        }
        s
    }
}

/// A section's header row: a caret, its title; a click folds or unfolds it. Returns whether it is
/// open now.
fn header(ui: &mut Ui, st: &mut UiState, s: Section) -> bool {
    let (rect, r) = ui.allocate_exact_size(vec2(ui.available_width(), 26.0), Sense::click());
    probe_rect(&format!("view-{}", s.id()), rect);
    if r.clicked() {
        let open = st.view_sections.is_open(s);
        st.view_sections.set(s, !open);
    }
    let open = st.view_sections.is_open(s);
    let p = ui.painter();
    if r.hovered() {
        p.rect_filled(rect, 5.0, theme::PANEL_2);
    }
    let ink = if r.hovered() { theme::INK } else { theme::INK_DIM };
    // the caret: ▸ folded, ▾ open
    let c = pos2(rect.left() + 10.0, rect.center().y);
    let tri = if open { vec![c + vec2(-4.0, -2.0), c + vec2(4.0, -2.0), c + vec2(0.0, 3.0)] } else { vec![c + vec2(-2.0, -4.0), c + vec2(3.0, 0.0), c + vec2(-2.0, 4.0)] };
    p.add(Shape::convex_polygon(tri, ink, Stroke::NONE));
    p.text(pos2(rect.left() + 22.0, rect.center().y), egui::Align2::LEFT_CENTER, s.title(), egui::FontId::proportional(13.5), theme::INK);
    r.on_hover_text(if open { "fold this section" } else { "unfold this section" });
    open
}

/// The panel's contents (inside its scroll area, `dock::body`).
pub fn show(ui: &mut Ui, st: &mut UiState, acts: &mut Vec<Action>) {
    // the sliders leave room for their value and name in a narrow panel (egui's 100 pt at 300 pt)
    ui.spacing_mut().slider_width = (ui.available_width() * 0.32).clamp(70.0, 180.0);
    // the visible part of the panel (the reveal test: a revealed control lands inside it)
    probe_rect("view-panel", ui.clip_rect());
    // a Ctrl K reveal unfolds the section holding its anchor until the control has scrolled into
    // view; the section then stays open
    if let Some(a) = st.search.unrevealed() {
        if let Some(s) = Section::ALL.into_iter().find(|s| s.anchor() == a) {
            st.view_sections.set(s, true);
        }
    }
    for s in Section::ALL {
        if header(ui, st, s) {
            super::group(ui, st, s.anchor(), |ui, st| {
                ui.indent(("view-section", s.id()), |ui| match s {
                    Section::What => what(ui, st, acts),
                    Section::Looks => looks(ui, st, acts),
                    Section::Camera => camera(ui, st),
                    Section::Bonds => bonds(ui, st, acts),
                    Section::Aids => aids(ui, st, acts),
                });
            });
            ui.add_space(4.0);
        }
    }
    ui.separator();
    super::group(ui, st, "m-view-reset", |ui, st| {
        if probed("view-reset", ui.button("Reset view settings"))
            .on_hover_text("How it looks, Camera, Bonds & labels and Display aids back to the defaults (What you see stays as it is); the panel's folds stay")
            .clicked()
        {
            reset(st, acts);
        }
    });
}

/// What you see: the field the view shows (orbital-viewer spec §3; the Orbitals panel holds the
/// list and the per-mode controls) and the estimated solver's cloud (Q5 of Appendix B, M5b).
fn what(ui: &mut Ui, st: &mut UiState, acts: &mut Vec<Action>) {
    ui.horizontal_wrapped(|ui| {
        ui.label("show");
        theme::info(ui, format!("{} {}", orbitals::KS_NOTE, orbitals::SPIN_NOTE));
    });
    if let Some(m) = orbitals::show_selector(ui, st.show, &mut st.hide_cloud, "menu-show") {
        acts.push(Action::Show(m));
    }
    theme::section(ui, "Estimated solver's cloud");
    cloud_controls(ui, st, acts);
}

/// The estimated solver's cloud (`crate::cloud`; `sim` module doc, "The cloud"): the display warp
/// (every solver), atomic superposition between DFT clouds and the display refresh (that solver
/// only: hidden under the others).
fn cloud_controls(ui: &mut Ui, st: &UiState, acts: &mut Vec<Action>) {
    let mut warp = st.warp_fields;
    if ui
        .checkbox(&mut warp, "fields follow the atoms")
        .on_hover_text("the density and orbitals were solved at one geometry; while the atoms move on (the estimated solver, or between DFT steps) the field is warped with them — each point moves with its nearby atoms — until the next solve. A display approximation: it moves what was solved and cannot make new chemistry (a bond that broke since is drawn stretched until the next solve)")
        .changed()
    {
        acts.push(Action::WarpFields(warp));
    }
    if st.backend != Backend::Estimated {
        theme::note(ui, "(cloud options apply to the estimated solver; DFT steps show their own density)");
        return;
    }
    let (mut sup, mut refresh, mut every) = (st.cloud_superposition, st.cloud_refresh, st.cloud_refresh_s);
    let mut changed = ui
        .checkbox(&mut sup, "atomic superposition between DFT clouds")
        .on_hover_text("estimated solver: the model has no electrons, so between real densities the cloud is the teacher's own isolated atoms placed at the current nuclei — where the atoms are, not their bonds — blended with the last real density as the atoms move away from it. Labelled \"atomic superposition — not solved\". Needs the GPU teacher (its background solver computes the atoms once per species and grid)")
        .changed();
    ui.horizontal_wrapped(|ui| {
        changed |= probed("cloud-refresh", ui.checkbox(&mut refresh, "refresh cloud every"))
            .on_hover_text("estimated solver, while playing: a display-only DFT solve at the current geometry on the background solver (≈ 0.1–0.2 s for H₂ at 49³; it waits while an audit runs, and goes before the next one — at full speed, where audits run back to back, it costs ≈ solve time / N of them), labelled with its time. Not a training sample. Needs the GPU teacher")
            .changed();
        if refresh {
            changed |= ui.add(egui::Slider::new(&mut every, 1.0..=10.0).suffix(" s").fixed_decimals(0)).changed();
        }
    });
    if changed {
        acts.push(Action::Cloud { superposition: sup, refresh, refresh_s: every });
    }
}

/// How it looks: volume / isosurface / dots and each one's look, the partial core, bloom.
fn looks(ui: &mut Ui, st: &mut UiState, acts: &mut Vec<Action>) {
    ui.horizontal_wrapped(|ui| {
        ui.label("render as");
        let ori = "volume / isosurface: for an orbital or the spin density, a glowing volume with the sign's colour, or the ± isosurface. Dots: random samples of the field";
        if let Some(m) = theme::segmented(ui, st.volume_mode, &[(VOLUME_VOLUME, "volume", ori), (VOLUME_ISO, "isosurface", ori), (VOLUME_DOTS, "dots", "random samples of the field, ∝ ρ (or |ψ|², or |m|) per cell: dots are samples of where an electron is likely to be — not electrons")]) {
            acts.push(Action::VolumeMode(m));
        }
    });
    match st.volume_mode {
        VOLUME_VOLUME => {
            ui.add(egui::Slider::new(&mut st.gamma, 0.2..=1.0).text("gamma"));
            ui.add(egui::Slider::new(&mut st.opacity, 0.1..=20.0).logarithmic(true).text("opacity"));
        }
        VOLUME_ISO => {
            ui.add(egui::Slider::new(&mut st.iso, 0.01..=0.9).logarithmic(true).text("iso (× ρ max)"));
        }
        _ => dots_look(ui, st),
    }
    // M7: the pseudised species' partial core in the density shown (a display of the generator's model)
    let mut core = st.show_partial_core;
    if probed("show-partial-core", ui.checkbox(&mut core, "with partial core")).on_hover_text(pseudo::PARTIAL_CORE_TIP).changed() {
        acts.push(Action::ShowPartialCore(core));
    }
    // bloom: the bright parts glow (display only)
    ui.horizontal(|ui| {
        ui.label("bloom");
        ui.add(egui::Slider::new(&mut st.bloom, 0.0..=1.5).fixed_decimals(2))
            .on_hover_text("a soft glow around the bright parts of the view — dense glowing dots, a hot density core, the Camera view's atoms (0 = off). Display only");
    });
    if st.bloom > 0.0 {
        ui.horizontal(|ui| {
            ui.label("bloom from");
            ui.add(egui::Slider::new(&mut st.bloom_threshold, 0.1..=1.0).fixed_decimals(2))
                .on_hover_text("how bright a part must be to glow (1 = only full white)");
        });
    }
}

/// The dots' look, while rendering as dots: floating (each dot wanders by a Metropolis walk on the
/// field, so the cloud stays a true sample of it), glowing (soft sprites added: dense regions burn
/// to white), their amount, feather, size and blur; the orbital colouring under All orbitals
/// coloured.
fn dots_look(ui: &mut Ui, st: &mut UiState) {
    ui.checkbox(&mut st.shimmer, "shimmer (new dots every frame)");
    ui.horizontal_wrapped(|ui| {
        ui.checkbox(&mut st.dots_float, "floating").on_hover_text("each dot wanders: a random step each frame, kept or refused by the density there (a Metropolis walk) — the cloud stays a true sample of the field while the dots drift");
        ui.checkbox(&mut st.dots_glow, "glow").on_hover_text("soft dots whose light adds up: dense regions burn toward white");
    });
    if st.dots_float {
        ui.add(egui::Slider::new(&mut st.dots_drift, 0.002..=1.5).logarithmic(true).text("drift (grid cells per frame)"));
    }
    ui.add(egui::Slider::new(&mut st.dots_amount, 0.5..=6.0).text("amount (× the default)"));
    // the view switch's feather (`overlays::view_switch`), the same setting
    ui.add(egui::Slider::new(&mut st.dots_feather, 0.0..=0.8).text("feather"))
        .on_hover_text("fade out the faint outskirts: the dots where the field is low disappear softly, leaving the main lobes (0 = every dot)");
    ui.add(egui::Slider::new(&mut st.dots_size, 0.05..=3.0).logarithmic(true).text("dot size"));
    ui.add(egui::Slider::new(&mut st.dots_blur, 0.0..=1.0).text("blur (0 crisp … 1 soft)"));
    if st.dots_glow {
        ui.add(egui::Slider::new(&mut st.dots_brightness, 0.2..=4.0).logarithmic(true).text("brightness"));
    }
    if st.show == ShowMode::Coloured {
        ui.horizontal(|ui| {
            ui.label("colour dots by");
            if let Some(by_orbital) = theme::segmented(ui, st.dots_by_orbital, &[(false, "density", "the density ramp at the dot"), (true, "orbital", "each dot in its orbital's own colour (All orbitals coloured)")]) {
                st.dots_by_orbital = by_orbital;
            }
        });
    }
}

/// Camera: the Camera view (also the view's bottom-left switch) and, while it is on, its lens and
/// look; motion blur and the background, which apply to either view.
fn camera(ui: &mut Ui, st: &mut UiState) {
    ui.checkbox(&mut st.camera_view, "camera view").on_hover_text("the atoms as they look rather than as markers: from a soft electron glow to solid balls, blurred over the shutter (also the view switch, bottom left)");
    if st.camera_view {
        let row = |ui: &mut Ui, name: &str, v: &mut f32, range: std::ops::RangeInclusive<f32>, text: String, tip: &str| {
            ui.horizontal(|ui| {
                ui.add_sized([72.0, 18.0], egui::Label::new(name));
                ui.add(egui::Slider::new(v, range).show_value(false)).on_hover_text(tip);
                ui.label(text);
            });
        };
        let t = format!("{:+.1} EV", st.exposure_ev);
        row(ui, "exposure", &mut st.exposure_ev, -3.0..=3.0, t, "brightness, in stops (EV)");
        let t = format!("{:.0} %", st.shutter * 100.0);
        row(ui, "shutter", &mut st.shutter, 0.0..=1.0, t, "how much of the motion between frames one picture collects: longer streaks to the right, a sharp still at 0");
        let t = format!("{:.0}°", st.fov_deg);
        row(ui, "field of view", &mut st.fov_deg, 15.0..=100.0, t, "the lens: narrow (telephoto, flat) to wide (close, strong perspective)");
        let t = if st.camera_solidity <= 0.0 { "glow".into() } else if st.camera_solidity >= 1.0 { "solid".into() } else { format!("{:.0} %", st.camera_solidity * 100.0) };
        row(ui, "solidity", &mut st.camera_solidity, 0.0..=1.0, t, "how the atoms look: a soft electron glow (left, closest to what an atom is) to solid shaded balls (right, like a space-filling model)");
        let t = format!("{:.1}×", st.camera_size);
        row(ui, "size", &mut st.camera_size, 0.6..=4.0, t, "the atoms' size, × the diagram's radius: about 1× is the diagram's small balls, about 3–4× is space-filling");
        let t = if st.camera_smooth <= 0.0 { "balls".into() } else { format!("{:.2} Bohr", st.camera_smooth) };
        row(ui, "smooth", &mut st.camera_smooth, 0.0..=2.0, t, "with any solidity: one smooth surface instead of separate balls — atoms close together flow into each other through a soft neck this wide (0 = separate balls, the hard crease)");
    }
    ui.horizontal(|ui| {
        ui.label("motion blur");
        ui.add(egui::Slider::new(&mut st.motion_blur, 0.0..=1.0).show_value(false))
            .on_hover_text("while playing, faint copies of each atom along the path its last steps took — a fast vibration smears instead of jittering. Shows when many steps run per frame (the estimated solver); 0 turns it off. Display only");
        ui.label(format!("{:.0} %", st.motion_blur * 100.0));
    });
    // the 3D view's background: an inline picker, presets, and a reset to the default dark blue
    egui::CollapsingHeader::new("background colour").id_salt("bg-colour").show(ui, |ui| {
        let [r, g, b] = st.background.unwrap_or([29, 31, 41]);
        let mut c = egui::Color32::from_rgb(r, g, b);
        if egui::widgets::color_picker::color_picker_color32(ui, &mut c, egui::widgets::color_picker::Alpha::Opaque) {
            st.background = Some([c.r(), c.g(), c.b()]);
        }
        ui.horizontal(|ui| {
            for (name, rgb) in [("black", [0, 0, 0]), ("charcoal", [24, 24, 28]), ("slate", [46, 52, 64]), ("grey", [128, 128, 132]), ("white", [245, 245, 245])] {
                let (rect, resp) = ui.allocate_exact_size(egui::vec2(22.0, 22.0), Sense::click());
                ui.painter().rect(rect, 4.0, egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]), Stroke::new(1.0, theme::LINE), egui::StrokeKind::Inside);
                if resp.on_hover_text(name).clicked() {
                    st.background = Some(rgb);
                }
            }
            if st.background.is_some() {
                ui.add_space(8.0);
                if ui.button("Reset").on_hover_text("the default dark blue").clicked() {
                    st.background = None;
                }
            }
        });
    });
}

/// Bonds & labels (Q6 and Q7 of Appendix B, M6): the bonds read off the density, the molecule
/// names and their style, the composition badge, angles, lengths, partial charges, label opacity.
fn bonds(ui: &mut Ui, st: &mut UiState, acts: &mut Vec<Action>) {
    ui.horizontal_wrapped(|ui| {
        let (mut b, mut l) = (st.show_bonds, st.show_labels);
        if probed("show-bonds", ui.checkbox(&mut b, "bonds")).on_hover_text(rich(BONDS_TIP, 12.5, theme::INK)).changed() {
            acts.push(Action::ShowBonds(b));
        }
        if probed("show-labels", ui.checkbox(&mut l, "molecule names"))
            .on_hover_text("each molecule's formula and name next to it in the view (also listed in the Selection panel): a molecule is a connected piece of the bond graph; names come from a small table of trivial names or the IUPAC rules for binary compounds, marked \"(systematic)\"")
            .changed()
        {
            acts.push(Action::ShowLabels(l));
        }
        theme::info(ui, rich(BONDS_TIP, 12.5, theme::INK));
    });
    ui.label("name style").on_hover_text(NAME_STYLE_TIP);
    let opts: Vec<(crate::names::NameStyle, &str, &str)> = crate::names::NameStyle::ALL.iter().map(|&s| (s, s.title(), NAME_STYLE_TIP)).collect();
    if let Some(s) = theme::segmented_wrapped(ui, st.name_style, &opts) {
        acts.push(Action::NameStyle(s));
    }
    let mut g = st.show_badge;
    if probed("show-badge", ui.checkbox(&mut g, "composition badge"))
        .on_hover_text("hovering a nucleus with the Select tool (or selecting one) shows its isotope, protons, neutrons and the force on it: \"¹⁶O · 8 p · 8 n · |F| 0.012 Ha/Bohr\"")
        .changed()
    {
        acts.push(Action::ShowBadge(g));
    }
    let mut a = st.show_angles;
    if probed("show-angles", ui.checkbox(&mut a, "bond angles")).on_hover_text(ANGLES_TIP).changed() {
        acts.push(Action::ShowAngles(a));
    }
    probed("show-bond-lengths", ui.checkbox(&mut st.show_bond_lengths, "bond lengths")).on_hover_text(
        "each drawn bond's length in Å at its middle (the nuclei's distance as drawn now; Bohr in the hover of the Selection tab). With more than 12 bonds only those of the selected or hovered atom are labelled. Display only",
    );
    probed("show-charges", ui.checkbox(&mut st.show_charges, "partial charges")).on_hover_text(CHARGES_TIP);
    ui.horizontal(|ui| {
        ui.label("label opacity");
        ui.add(egui::Slider::new(&mut st.label_opacity, 0.0..=1.0).show_value(false))
            .on_hover_text("the labels drawn in the 3D view: molecule names, bond lengths, angles and partial charges (display only)");
        ui.label(format!("{:.0} %", st.label_opacity * 100.0));
    });
    ui.label(rich(HONEST_BONDS, 11.0, theme::INK_DIM));
}

/// Display aids: the cell, floor grid and axes, the placement plane, trails, the knowledge
/// overlay, what training is solving, smooth motion, the nuclei's size.
fn aids(ui: &mut Ui, st: &mut UiState, acts: &mut Vec<Action>) {
    ui.horizontal_wrapped(|ui| {
        ui.checkbox(&mut st.show_cell, "cell box").on_hover_text("the simulation box: nuclei must stay inside it");
        ui.checkbox(&mut st.show_floor, "floor grid");
        ui.checkbox(&mut st.show_axes, "axes");
    });
    ui.horizontal_wrapped(|ui| {
        ui.checkbox(&mut st.show_place_grid, "placement plane").on_hover_text("Place tool: the plane the next click lands on");
        ui.label("spacing");
        ui.radio_value(&mut st.grid_in_angstrom, false, "1 Bohr");
        ui.radio_value(&mut st.grid_in_angstrom, true, "1 Å");
    });
    trail_controls(ui, st);
    knowledge::overlay_toggle(ui, st);
    probed("show-ghost", ui.checkbox(&mut st.show_ghost, "show what training is solving"))
        .on_hover_text("while a sampling pass runs, the view shows the geometry the teacher is solving right now as a wireframe ghost (a lone reference atom, a random nudge with arrows, the scanned pair, the hot burst with a trail) instead of your scene, flashing green / red / grey as each sample is kept / fails / is skipped; your scene returns when the pass ends or is cancelled. Display only");
    let mut smooth = st.smooth_motion;
    if ui
        .checkbox(&mut smooth, "smooth motion")
        .on_hover_text("interpolates between simulation snapshots (one publish interval of latency): the nuclei glide, and the density cloud crossfades between the last two solved densities with the same weight — a blend for display, not a transported density")
        .changed()
    {
        acts.push(Action::SmoothMotion(smooth));
    }
    nucleus_size(ui, st);
}

/// Trails (`crate::trails`, user request 2026-10-17): a fading line behind each atom. Display only:
/// written straight to `UiState`, never sent to the worker.
fn trail_controls(ui: &mut Ui, st: &mut UiState) {
    use crate::trails;
    ui.checkbox(&mut st.show_trails, "trails").on_hover_text(
        "a thin fading line behind each atom, in its element colour, showing where it has been (display only; turning it on starts from now). Teleporting an atom, changing the nucleus count, loading a scene or a rewind starts a fresh trail; dragging keeps it",
    );
    if !st.show_trails {
        return;
    }
    let mut x = trails::length_to_slider(st.trail_fs);
    let length = egui::Slider::new(&mut x, 0.0..=1.0)
        .text("trail length")
        .custom_formatter(|v, _| trails::length_text(trails::slider_to_length(v)))
        .custom_parser(|s| trails::parse_length(s).map(trails::length_to_slider));
    if ui
        .add(length)
        .on_hover_text("how long a trail point stays visible, in SIMULATED time (5 fs – 10 ps, logarithmic): the trail covers the same motion at any play speed, whether the simulation runs at 1 or 10 000 steps per tick. Type a value like 250 or 2 ps")
        .changed()
    {
        st.trail_fs = trails::slider_to_length(x);
    }
    let mut pct = (st.trail_opacity * 100.0).round();
    if ui
        .add(egui::Slider::new(&mut pct, 0.0..=100.0).text("trail opacity").suffix(" %").fixed_decimals(0))
        .on_hover_text("the newest end's opacity; the trail fades linearly to fully transparent at the chosen length")
        .changed()
    {
        st.trail_opacity = pct / 100.0;
    }
}

/// The nuclei's sphere size (`UiState::nucleus_scale`, `gfx::nucleus_scale`): automatic — full
/// size, smaller while an orbital is drawn so the spheres do not hide it — or a fixed factor.
fn nucleus_size(ui: &mut Ui, st: &mut UiState) {
    ui.horizontal_wrapped(|ui| {
        let mut auto = st.nucleus_scale.is_none();
        if probed("nucleus-auto", ui.checkbox(&mut auto, "nucleus size: auto"))
            .on_hover_text("the nuclei's spheres at their display radius, shrunk while an orbital is drawn so they do not hide its lobes; untick to fix the size yourself (display only)")
            .changed()
        {
            st.nucleus_scale = if auto { None } else { Some(1.0) };
        }
        if let Some(s) = &mut st.nucleus_scale {
            ui.add(egui::Slider::new(s, 0.1..=2.0).fixed_decimals(2).suffix("×")).on_hover_text("× each element's display radius");
        }
    });
}

/// Reset view settings: How it looks, Camera, Bonds & labels and Display aids back to a fresh
/// state's values. The settings that are `Action`s are reset by their actions (so a recording
/// sees it), only where they differ; the rest are written directly. What you see — the field
/// shown, the cloud — and the panel's folds are left alone.
fn reset(st: &mut UiState, acts: &mut Vec<Action>) {
    let d = UiState::new("h2", &crate::scene::Scene::preset("h2").expect("the h2 preset"));
    if st.volume_mode != d.volume_mode {
        acts.push(Action::VolumeMode(d.volume_mode));
    }
    if st.show_partial_core != d.show_partial_core {
        acts.push(Action::ShowPartialCore(d.show_partial_core));
    }
    for (now, was, act) in [
        (st.show_bonds, d.show_bonds, Action::ShowBonds as fn(bool) -> Action),
        (st.show_labels, d.show_labels, Action::ShowLabels),
        (st.show_badge, d.show_badge, Action::ShowBadge),
        (st.show_angles, d.show_angles, Action::ShowAngles),
        (st.smooth_motion, d.smooth_motion, Action::SmoothMotion),
    ] {
        if now != was {
            acts.push(act(was));
        }
    }
    if st.name_style != d.name_style {
        acts.push(Action::NameStyle(d.name_style));
    }
    macro_rules! back {
        ($($f:ident),*) => { $(st.$f = d.$f;)* };
    }
    back!(
        gamma, opacity, iso, shimmer, dots_float, dots_glow, dots_drift, dots_amount, dots_feather, dots_size, dots_blur, dots_brightness, dots_by_orbital, bloom, bloom_threshold,
        camera_view, exposure_ev, shutter, fov_deg, camera_solidity, camera_size, camera_smooth, motion_blur, background,
        show_bond_lengths, show_charges, label_opacity,
        show_cell, show_floor, show_axes, show_place_grid, grid_in_angstrom, show_trails, trail_fs, trail_opacity, show_knowledge, show_ghost, nucleus_scale
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sections_round_trip_through_the_profile_word_list() {
        assert_eq!(Sections::default().encode(), "what,looks");
        assert_eq!(Sections::parse("what,looks"), Sections::default());
        assert_eq!(Sections::parse(""), Sections::all(false));
        assert_eq!(Sections::all(true).encode(), "what,looks,camera,bonds,aids");
        let s = Sections::parse(" Bonds , aids,bogus,camera ");
        assert!(!s.is_open(Section::What) && s.is_open(Section::Camera) && s.is_open(Section::Bonds) && s.is_open(Section::Aids));
        assert_eq!(s.encode(), "camera,bonds,aids", "panel order");
        for x in Section::ALL {
            assert_eq!(Sections::parse(x.id()).encode(), x.id());
        }
    }
}
