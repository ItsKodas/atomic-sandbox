//! The View panel (docking spec D-6; restructured 2026-11-14 from the user's approved mockup).
//! Top to bottom:
//!
//! 1. **LOOK** (a card, anchor `m-looks`): four look presets — [`CLEAN`], [`GLOW`], [`SCIENTIFIC`],
//!    [`PRESENT`], each a fixed bundle of display settings ([`Look`]) — the active one highlighted,
//!    "custom" once any of its settings is touched; render as (volume / isosurface / dots), colour
//!    by (dots under All orbitals coloured, as before) and the two controls of the render mode that
//!    matter most ([`main_controls`]); the rest of that mode's look in the "Fine-tune <mode>" fold.
//!    Its ↺ puts the look back to the app's defaults (what Reset view settings did for it).
//! 2. The `show` selector is not here any more: it heads the Orbitals panel (the same state).
//! 3. **DISPLAY AIDS** (a card, anchor `m-display`): the old checkboxes as toggle chips; a chip with
//!    settings (placement plane, trails, nucleus size) opens them in a row under the chips with its ▾.
//! 4. **ESTIMATED MODE CLOUD** (a card, anchor `m-density`, only under the estimated solver): fields
//!    follow the atoms, atoms between solves, refresh every N s. Under the other solvers "fields
//!    follow the atoms" (which applies between DFT steps too) is a chip of Display aids.
//! 5. Folds, closed by default: **Bonds and labels** (`m-bonds`) and **Camera** (`m-camera`), each
//!    with a summary in its header and a ↺ of its own; then a muted Reset all view settings
//!    (`view-reset`).
//!
//! The folds' open state ([`Sections`]: Fine-tune, Camera, Bonds) is display state kept in the
//! profile (`view_sections = finetune,camera,bonds`); a Ctrl K reveal unfolds the fold holding its
//! anchor. Like the menu it replaced, this is display state: the controls that were `Action`s
//! (render as, partial core, the cloud, the warp, bonds and labels, name style, smooth motion)
//! still are — recorded, some of them sent — and the rest write `UiState` directly.
use super::super::menus::{ANGLES_TIP, BONDS_TIP, HONEST_BONDS, NAME_STYLE_TIP};
use super::super::{knowledge, probe_rect, probed, pseudo, rich, theme, Action, ShowMode, UiState, CHARGES_TIP, VOLUME_DOTS, VOLUME_ISO, VOLUME_VOLUME};
use crate::sim::Backend;
use egui::{vec2, RichText, Sense, Stroke, Ui};

/// One of the panel's folds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Section {
    /// "Fine-tune <mode>" in the LOOK card: the render mode's other look settings.
    FineTune,
    Camera,
    Bonds,
}

impl Section {
    pub const ALL: [Section; 3] = [Section::FineTune, Section::Camera, Section::Bonds];

    /// Its word in the profile's `view_sections` line (its header's probe: `view-<id>`).
    pub fn id(self) -> &'static str {
        match self {
            Section::FineTune => "finetune",
            Section::Camera => "camera",
            Section::Bonds => "bonds",
        }
    }

    /// The search anchor its contents mark (`search::ENTRIES`).
    pub fn anchor(self) -> &'static str {
        match self {
            Section::FineTune => "m-finetune",
            Section::Camera => "m-camera",
            Section::Bonds => "m-bonds",
        }
    }
}

/// Which folds are open (`UiState::view_sections`): none by default.
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

    /// The profile's value: the open folds' ids, comma separated, in panel order.
    pub fn encode(&self) -> String {
        Section::ALL.into_iter().filter(|&s| self.is_open(s)).map(Section::id).collect::<Vec<_>>().join(",")
    }

    /// The profile's value read back: the ids named are open, the rest closed; unknown words (the
    /// old sections' `what`, `looks`, `aids`, now cards) are dropped, case and spaces ignored.
    pub fn parse(v: &str) -> Sections {
        let mut s = Sections([false; 3]);
        for w in v.split(',').map(str::trim) {
            if let Some(x) = Section::ALL.into_iter().find(|x| x.id().eq_ignore_ascii_case(w)) {
                s.set(x, true);
            }
        }
        s
    }
}

// ---- the look presets ----

/// A look: every display setting a preset fixes — the render mode (an `Action`), the dots'
/// colouring and look, bloom, and the volume's and isosurface's levels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Look {
    pub volume_mode: u32,
    pub dots_by_orbital: bool,
    pub shimmer: bool,
    pub dots_float: bool,
    pub dots_glow: bool,
    pub dots_drift: f32,
    pub dots_amount: f32,
    pub dots_feather: f32,
    pub dots_size: f32,
    pub dots_blur: f32,
    pub dots_brightness: f32,
    pub bloom: f32,
    pub bloom_threshold: f32,
    pub gamma: f32,
    pub opacity: f32,
    pub iso: f32,
}

/// Clean: the app's default look (a glowing volume; the dots' defaults) without the shimmer and
/// without bloom — a quiet, steady picture.
pub const CLEAN: Look = Look {
    volume_mode: VOLUME_VOLUME,
    dots_by_orbital: true,
    shimmer: false,
    dots_float: true,
    dots_glow: true,
    dots_drift: 0.35,
    dots_amount: 2.0,
    dots_feather: 0.0,
    dots_size: 1.0,
    dots_blur: 0.5,
    dots_brightness: 1.0,
    bloom: 0.0,
    bloom_threshold: 0.6,
    gamma: 0.5,
    opacity: 2.0,
    iso: 0.1,
};

/// Glow: the user's own style (the screenshot of 2026-11-14) — dots coloured by orbital, shimmering,
/// floating and glowing, drift 0.15, amount 0.5, feather 0, size 0.42, blur 0.6, brightness 1.0,
/// bloom 0.87 from 0.56.
pub const GLOW: Look = Look {
    volume_mode: VOLUME_DOTS,
    dots_by_orbital: true,
    shimmer: true,
    dots_float: true,
    dots_glow: true,
    dots_drift: 0.15,
    dots_amount: 0.5,
    dots_feather: 0.0,
    dots_size: 0.42,
    dots_blur: 0.6,
    dots_brightness: 1.0,
    bloom: 0.87,
    bloom_threshold: 0.56,
    ..CLEAN
};

/// Scientific: the ± isosurface (a definite contour reads as a measurement better than a glowing
/// volume), dots coloured by density, no glow, bloom or shimmer.
pub const SCIENTIFIC: Look = Look { volume_mode: VOLUME_ISO, dots_by_orbital: false, shimmer: false, dots_glow: false, bloom: 0.0, ..CLEAN };

/// Present: Glow tuned for screen sharing (video compression eats faint, fast detail) — brighter
/// dots (1.4), a little more of them (0.7) and larger (0.5), slower drift (0.08), a little more
/// bloom (1.0 from 0.5).
pub const PRESENT: Look = Look { dots_drift: 0.08, dots_amount: 0.7, dots_size: 0.5, dots_brightness: 1.4, bloom: 1.0, bloom_threshold: 0.5, ..GLOW };

/// The four presets, in the card's order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LookPreset {
    Clean,
    Glow,
    Scientific,
    Present,
}

impl LookPreset {
    pub const ALL: [LookPreset; 4] = [LookPreset::Clean, LookPreset::Glow, LookPreset::Scientific, LookPreset::Present];

    pub fn title(self) -> &'static str {
        match self {
            LookPreset::Clean => "Clean",
            LookPreset::Glow => "Glow",
            LookPreset::Scientific => "Scientific",
            LookPreset::Present => "Present",
        }
    }

    /// Its button's probe: `look-<word>`.
    pub fn word(self) -> &'static str {
        match self {
            LookPreset::Clean => "clean",
            LookPreset::Glow => "glow",
            LookPreset::Scientific => "scientific",
            LookPreset::Present => "present",
        }
    }

    pub fn look(self) -> Look {
        match self {
            LookPreset::Clean => CLEAN,
            LookPreset::Glow => GLOW,
            LookPreset::Scientific => SCIENTIFIC,
            LookPreset::Present => PRESENT,
        }
    }

    fn tip(self) -> &'static str {
        match self {
            LookPreset::Clean => "the default glowing volume, steady: no shimmer, no bloom",
            LookPreset::Glow => "glowing dots coloured by orbital, shimmering and drifting, with bloom",
            LookPreset::Scientific => "the isosurface, coloured by density: no glow, bloom or shimmer",
            LookPreset::Present => "Glow tuned for screen sharing: brighter, a little more bloom, slower drift",
        }
    }
}

impl Look {
    /// The look in force.
    pub fn of(st: &UiState) -> Look {
        Look {
            volume_mode: st.volume_mode,
            dots_by_orbital: st.dots_by_orbital,
            shimmer: st.shimmer,
            dots_float: st.dots_float,
            dots_glow: st.dots_glow,
            dots_drift: st.dots_drift,
            dots_amount: st.dots_amount,
            dots_feather: st.dots_feather,
            dots_size: st.dots_size,
            dots_blur: st.dots_blur,
            dots_brightness: st.dots_brightness,
            bloom: st.bloom,
            bloom_threshold: st.bloom_threshold,
            gamma: st.gamma,
            opacity: st.opacity,
            iso: st.iso,
        }
    }

    /// Puts this look in force: the render mode by its `Action` (where it differs), the rest written
    /// directly (display state).
    pub fn apply(self, st: &mut UiState, acts: &mut Vec<Action>) {
        if st.volume_mode != self.volume_mode {
            acts.push(Action::VolumeMode(self.volume_mode));
        }
        st.dots_by_orbital = self.dots_by_orbital;
        st.shimmer = self.shimmer;
        st.dots_float = self.dots_float;
        st.dots_glow = self.dots_glow;
        st.dots_drift = self.dots_drift;
        st.dots_amount = self.dots_amount;
        st.dots_feather = self.dots_feather;
        st.dots_size = self.dots_size;
        st.dots_blur = self.dots_blur;
        st.dots_brightness = self.dots_brightness;
        st.bloom = self.bloom;
        st.bloom_threshold = self.bloom_threshold;
        st.gamma = self.gamma;
        st.opacity = self.opacity;
        st.iso = self.iso;
    }
}

/// The preset the look in force is, if any (each setting within 1e-4 of the bundle's): after a
/// slider is touched it is none ("custom").
pub fn active_preset(st: &UiState) -> Option<LookPreset> {
    let now = Look::of(st);
    let near = |a: f32, b: f32| (a - b).abs() < 1e-4;
    LookPreset::ALL.into_iter().find(|p| {
        let l = p.look();
        l.volume_mode == now.volume_mode
            && l.dots_by_orbital == now.dots_by_orbital
            && l.shimmer == now.shimmer
            && l.dots_float == now.dots_float
            && l.dots_glow == now.dots_glow
            && [
                (l.dots_drift, now.dots_drift),
                (l.dots_amount, now.dots_amount),
                (l.dots_feather, now.dots_feather),
                (l.dots_size, now.dots_size),
                (l.dots_blur, now.dots_blur),
                (l.dots_brightness, now.dots_brightness),
                (l.bloom, now.bloom),
                (l.bloom_threshold, now.bloom_threshold),
                (l.gamma, now.gamma),
                (l.opacity, now.opacity),
                (l.iso, now.iso),
            ]
            .into_iter()
            .all(|(a, b)| near(a, b))
    })
}

// ---- the panel ----

/// A card (the Model tab's verdict card look) with a caps title, a ↺ at its right (probed
/// `{probe}-reset`, its tooltip `reset_tip`; returns whether it was clicked) and its body.
fn card(ui: &mut Ui, title: &str, probe: &str, reset_tip: &str, body: impl FnOnce(&mut Ui)) -> bool {
    let mut reset = false;
    let r = egui::Frame::new().fill(theme::PANEL_2).stroke(Stroke::new(1.0, theme::LINE)).corner_radius(8).inner_margin(egui::Margin::symmetric(10, 8)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            ui.label(theme::caps(title));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let b = ui.add(egui::Button::new(RichText::new("↺").size(12.0).color(theme::INK_DIM)).frame(false)).on_hover_text(reset_tip);
                probe_rect(&format!("{probe}-reset"), b.rect);
                reset = b.clicked();
            });
        });
        body(ui);
    });
    probe_rect(probe, r.response.rect);
    reset
}

/// A fold's header (`theme::fold_header`, probed `view-<id>`) with its summary; returns whether it is open.
fn fold_header(ui: &mut Ui, st: &mut UiState, s: Section, title: &str, summary: String, sub: bool) -> bool {
    let (r, _) = theme::fold_header(ui, title, st.view_sections.is_open(s), Some((summary, theme::INK_FAINT)), None, sub);
    probe_rect(&format!("view-{}", s.id()), r.rect);
    if r.clicked() {
        let open = st.view_sections.is_open(s);
        st.view_sections.set(s, !open);
    }
    st.view_sections.is_open(s)
}

/// A toggle chip: a rounded selectable button (on: accent), probed `probe`.
fn chip(ui: &mut Ui, on: bool, text: &str, probe: &str) -> egui::Response {
    let ink = if on { theme::ACCENT_INK } else { theme::INK_DIM };
    let b = egui::Button::new(RichText::new(text).size(11.5).color(ink))
        .fill(if on { theme::ACCENT_BG } else { theme::PANEL })
        .stroke(Stroke::new(1.0, if on { theme::ACCENT.gamma_multiply(0.6) } else { theme::LINE }))
        .corner_radius(255);
    probed(probe, ui.add(b))
}

/// The panel's contents (inside its scroll area, `dock::body`).
pub fn show(ui: &mut Ui, st: &mut UiState, acts: &mut Vec<Action>) {
    // the sliders leave room for their value and name in a narrow panel (egui's 100 pt at 300 pt)
    ui.spacing_mut().slider_width = (ui.available_width() * 0.32).clamp(70.0, 180.0);
    // the visible part of the panel (the reveal test: a revealed control lands inside it)
    probe_rect("view-panel", ui.clip_rect());
    // a Ctrl K reveal unfolds the fold holding its anchor until the control has scrolled into view;
    // the fold then stays open
    if let Some(a) = st.search.unrevealed() {
        if let Some(s) = Section::ALL.into_iter().find(|s| s.anchor() == a) {
            st.view_sections.set(s, true);
        }
    }
    super::group(ui, st, "m-looks", |ui, st| look_card(ui, st, acts));
    ui.add_space(4.0);
    super::group(ui, st, "m-display", |ui, st| aids_card(ui, st, acts));
    if st.backend == Backend::Estimated {
        ui.add_space(4.0);
        super::group(ui, st, "m-density", |ui, st| cloud_card(ui, st, acts));
    }
    ui.add_space(4.0);
    // the folds
    let on: Vec<&str> = [(st.show_bonds, "bonds"), (st.show_labels, "names"), (st.show_angles, "angles"), (st.show_bond_lengths, "lengths"), (st.show_charges, "charges")]
        .into_iter()
        .filter_map(|(o, w)| o.then_some(w))
        .collect();
    let summary = if on.is_empty() { "all off".to_string() } else { on.join(" · ") };
    if fold_header(ui, st, Section::Bonds, "Bonds and labels", summary, false) {
        super::group(ui, st, "m-bonds", |ui, st| {
            ui.indent("view-bonds-body", |ui| {
                reset_row(ui, "view-bonds-reset", "Bonds and labels back to the defaults", || reset_bonds(st, acts));
                bonds(ui, st, acts);
            });
        });
        ui.add_space(4.0);
    }
    let mut words: Vec<String> = Vec::new();
    if st.camera_view {
        words.push("camera view".into());
    }
    words.push(format!("blur {:.0} %", st.motion_blur * 100.0));
    words.push(if st.background.is_some() { "background set".into() } else { "background".into() });
    if fold_header(ui, st, Section::Camera, "Camera", words.join(" · "), false) {
        super::group(ui, st, "m-camera", |ui, st| {
            ui.indent("view-camera-body", |ui| {
                reset_row(ui, "view-camera-reset", "the Camera view, motion blur and the background back to the defaults", || reset_camera(st));
                camera(ui, st);
            });
        });
        ui.add_space(4.0);
    }
    ui.add_space(6.0);
    super::group(ui, st, "m-view-reset", |ui, st| {
        if probed("view-reset", ui.add(egui::Button::new(RichText::new("Reset all view settings").size(11.0).color(theme::INK_FAINT)).frame(false)))
            .on_hover_text("the look, Display aids, Bonds and labels and Camera back to the defaults (what the view shows — the Orbitals panel's show — and the estimated cloud stay as they are); the panel's folds stay")
            .clicked()
        {
            reset(st, acts);
        }
    });
}

/// A fold's ↺, right-aligned on its own row; `f` runs when it is clicked.
fn reset_row(ui: &mut Ui, probe: &str, tip: &str, f: impl FnOnce()) {
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
        let b = ui.add(egui::Button::new(RichText::new("↺ reset").size(11.0).color(theme::INK_DIM)).frame(false)).on_hover_text(tip);
        probe_rect(probe, b.rect);
        if b.clicked() {
            f();
        }
    });
}

/// LOOK: the presets, render as, colour by, the mode's two main controls, the Fine-tune fold.
fn look_card(ui: &mut Ui, st: &mut UiState, acts: &mut Vec<Action>) {
    let reset = card(ui, "Look", "view-look", "the look back to the app's defaults (render as volume, the default dots, bloom 0.35)", |ui| {
        let active = active_preset(st);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            let w = ((ui.available_width() - 3.0 * 4.0) / 4.0).max(40.0);
            for p in LookPreset::ALL {
                let on = active == Some(p);
                let b = egui::Button::selectable(on, RichText::new(p.title()).size(12.0)).min_size(vec2(w, 22.0));
                let r = probed(&format!("look-{}", p.word()), ui.add_sized([w, 22.0], b)).on_hover_text(p.tip());
                if r.clicked() {
                    p.look().apply(st, acts);
                }
            }
        });
        if active.is_none() {
            let r = ui.label(RichText::new("custom").size(10.5).color(theme::INK_FAINT)).on_hover_text("the look differs from every preset (a setting was changed): pick one to go back to it");
            probe_rect("look-custom", r.rect);
        }
        ui.horizontal_wrapped(|ui| {
            ui.label("render as");
            let ori = "volume / isosurface: for an orbital or the spin density, a glowing volume with the sign's colour, or the ± isosurface. Dots: random samples of the field";
            if let Some(m) = theme::segmented(ui, st.volume_mode, &[(VOLUME_VOLUME, "volume", ori), (VOLUME_ISO, "isosurface", ori), (VOLUME_DOTS, "dots", "random samples of the field, ∝ ρ (or |ψ|², or |m|) per cell: dots are samples of where an electron is likely to be — not electrons")]) {
                acts.push(Action::VolumeMode(m));
            }
        });
        if st.volume_mode == VOLUME_DOTS && st.show == ShowMode::Coloured {
            ui.horizontal(|ui| {
                ui.label("colour by");
                if let Some(by_orbital) = theme::segmented(ui, st.dots_by_orbital, &[(false, "density", "the density ramp at the dot"), (true, "orbital", "each dot in its orbital's own colour (All orbitals coloured)")]) {
                    st.dots_by_orbital = by_orbital;
                }
            });
        }
        main_controls(ui, st);
        let (title, summary) = match st.volume_mode {
            VOLUME_VOLUME => ("Fine-tune volume", "bloom · partial core"),
            VOLUME_ISO => ("Fine-tune isosurface", "bloom from · partial core"),
            _ => ("Fine-tune dots", "shimmer · float · glow · size · blur · bloom"),
        };
        if fold_header(ui, st, Section::FineTune, title, summary.to_string(), true) {
            super::group(ui, st, "m-finetune", |ui, st| ui.indent("view-finetune", |ui| fine_tune(ui, st, acts)));
        }
    });
    if reset {
        reset_look(st, acts);
    }
}

/// The render mode's two controls that matter most: dots — amount and brightness (with glow; the
/// dot size without it, since brightness acts only on glowing dots); volume — opacity and gamma;
/// isosurface — the iso level and bloom (it has no other level).
fn main_controls(ui: &mut Ui, st: &mut UiState) {
    match st.volume_mode {
        VOLUME_VOLUME => {
            probed("look-opacity", ui.add(egui::Slider::new(&mut st.opacity, 0.1..=20.0).logarithmic(true).text("opacity")));
            probed("look-gamma", ui.add(egui::Slider::new(&mut st.gamma, 0.2..=1.0).text("gamma")));
        }
        VOLUME_ISO => {
            probed("look-iso", ui.add(egui::Slider::new(&mut st.iso, 0.01..=0.9).logarithmic(true).text("iso (× ρ max)")));
            bloom_slider(ui, st);
        }
        _ => {
            probed("look-amount", ui.add(egui::Slider::new(&mut st.dots_amount, 0.5..=6.0).text("amount"))).on_hover_text("how many dots: × the default count");
            if st.dots_glow {
                probed("look-brightness", ui.add(egui::Slider::new(&mut st.dots_brightness, 0.2..=4.0).logarithmic(true).text("brightness")));
            } else {
                probed("look-size", ui.add(egui::Slider::new(&mut st.dots_size, 0.05..=3.0).logarithmic(true).text("dot size")));
            }
        }
    }
}

/// Bloom: the bright parts glow (display only).
fn bloom_slider(ui: &mut Ui, st: &mut UiState) {
    probed("look-bloom", ui.add(egui::Slider::new(&mut st.bloom, 0.0..=1.5).fixed_decimals(2).text("bloom")))
        .on_hover_text("a soft glow around the bright parts of the view — dense glowing dots, a hot density core, the Camera view's atoms (0 = off). Display only");
}

/// The Fine-tune fold: the render mode's other look settings, then bloom (where it is not a main
/// control), bloom from (with bloom) and the partial core.
fn fine_tune(ui: &mut Ui, st: &mut UiState, acts: &mut Vec<Action>) {
    if st.volume_mode == VOLUME_DOTS {
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut st.shimmer, "shimmer").on_hover_text("new dots every frame");
            ui.checkbox(&mut st.dots_float, "floating").on_hover_text("each dot wanders: a random step each frame, kept or refused by the density there (a Metropolis walk) — the cloud stays a true sample of the field while the dots drift");
            ui.checkbox(&mut st.dots_glow, "glow").on_hover_text("soft dots whose light adds up: dense regions burn toward white");
        });
        if st.dots_float {
            ui.add(egui::Slider::new(&mut st.dots_drift, 0.002..=1.5).logarithmic(true).text("drift")).on_hover_text("how far a floating dot wanders: grid cells per frame");
        }
        // the view switch's feather (`overlays::view_switch`), the same setting
        // clamped only when typed or dragged: drawing never rewrites a profile's value outside the range
        ui.add(egui::Slider::new(&mut st.dots_feather, 0.0..=0.8).clamping(egui::SliderClamping::Edits).text("feather"))
            .on_hover_text("fade out the faint outskirts: the dots where the field is low disappear softly, leaving the main lobes (0 = every dot)");
        if st.dots_glow {
            ui.add(egui::Slider::new(&mut st.dots_size, 0.05..=3.0).logarithmic(true).text("dot size"));
        }
        ui.add(egui::Slider::new(&mut st.dots_blur, 0.0..=1.0).text("blur")).on_hover_text("0 crisp … 1 soft");
    }
    if st.volume_mode != VOLUME_ISO {
        bloom_slider(ui, st);
    }
    if st.bloom > 0.0 {
        ui.add(egui::Slider::new(&mut st.bloom_threshold, 0.1..=1.0).fixed_decimals(2).text("bloom from")).on_hover_text("how bright a part must be to glow (1 = only full white)");
    }
    // M7: the pseudised species' partial core in the density shown (a display of the generator's model)
    let mut core = st.show_partial_core;
    if probed("show-partial-core", ui.checkbox(&mut core, "with partial core")).on_hover_text(pseudo::PARTIAL_CORE_TIP).changed() {
        acts.push(Action::ShowPartialCore(core));
    }
}

/// The settings row a chip with settings opens (display-only, in egui's memory).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AidRow {
    Plane,
    Trails,
}

/// A chip with settings: the chip toggles, its ▾ (probed `{probe}-more`) opens or closes its row.
fn chip_with_row(ui: &mut Ui, on: bool, text: &str, probe: &str, row: AidRow, open: &mut Option<AidRow>) -> egui::Response {
    // both straight into the wrapping row (a nested horizontal would not wrap, and would widen the card)
    let more_w = 14.0;
    let need = ui.painter().layout_no_wrap(text.to_string(), egui::FontId::proportional(11.5), theme::INK).size().x + 2.0 * ui.spacing().button_padding.x + more_w + ui.spacing().item_spacing.x;
    if ui.available_width() < need {
        ui.end_row();
    }
    let r = chip(ui, on, text, probe);
    let more = ui.add_sized([more_w, 18.0], egui::Button::new(RichText::new(if *open == Some(row) { "▴" } else { "▾" }).size(10.0).color(theme::INK_DIM)).frame(false)).on_hover_text("its settings");
    probe_rect(&format!("{probe}-more"), more.rect);
    if more.clicked() {
        *open = if *open == Some(row) { None } else { Some(row) };
    }
    r
}

/// DISPLAY AIDS: the old checkboxes as chips; placement plane and trails open their settings in a
/// row under the chips, nucleus size shows its slider while not automatic.
fn aids_card(ui: &mut Ui, st: &mut UiState, acts: &mut Vec<Action>) {
    let id = ui.id().with("view-aid-row");
    let mut open: Option<AidRow> = ui.data(|d| d.get_temp(id)).flatten();
    let reset = card(ui, "Display aids", "view-aids", "the display aids back to the defaults", |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = vec2(5.0, 5.0);
            if chip(ui, st.show_cell, "cell box", "aid-cell").on_hover_text("the simulation box: nuclei must stay inside it").clicked() {
                st.show_cell = !st.show_cell;
            }
            if chip(ui, st.show_floor, "floor grid", "aid-floor").clicked() {
                st.show_floor = !st.show_floor;
            }
            if chip(ui, st.show_axes, "axes", "aid-axes").clicked() {
                st.show_axes = !st.show_axes;
            }
            if chip_with_row(ui, st.show_place_grid, "placement plane", "aid-place", AidRow::Plane, &mut open).on_hover_text("Place tool: the plane the next click lands on (▾: its spacing)").clicked() {
                st.show_place_grid = !st.show_place_grid;
            }
            let trails = if st.show_trails { format!("trails {}", crate::trails::length_text(st.trail_fs)) } else { "trails".to_string() };
            if chip_with_row(ui, st.show_trails, &trails, "aid-trails", AidRow::Trails, &mut open)
                .on_hover_text("a thin fading line behind each atom, in its element colour, showing where it has been (display only; turning it on starts from now). Teleporting an atom, changing the nucleus count, loading a scene or a rewind starts a fresh trail; dragging keeps it. ▾: its length and opacity")
                .clicked()
            {
                st.show_trails = !st.show_trails;
            }
            if chip(ui, st.show_knowledge, "knowledge overlay", "show-knowledge").on_hover_text(knowledge::OVERLAY_TIP).clicked() {
                st.show_knowledge = !st.show_knowledge;
            }
            if chip(ui, st.show_ghost, "training ghost", "show-ghost")
                .on_hover_text("show what training is solving: while a sampling pass runs, the view shows the geometry the teacher is solving right now as a wireframe ghost (a lone reference atom, a random nudge with arrows, the scanned pair, the hot burst with a trail) instead of your scene, flashing green / red / grey as each sample is kept / fails / is skipped; your scene returns when the pass ends or is cancelled. Display only")
                .clicked()
            {
                st.show_ghost = !st.show_ghost;
            }
            if chip(ui, st.smooth_motion, "smooth motion", "aid-smooth")
                .on_hover_text("interpolates between simulation snapshots (one publish interval of latency): the nuclei glide, and the density cloud crossfades between the last two solved densities with the same weight — a blend for display, not a transported density")
                .clicked()
            {
                acts.push(Action::SmoothMotion(!st.smooth_motion));
            }
            let auto = st.nucleus_scale.is_none();
            if chip(ui, auto, "nucleus size auto", "nucleus-auto")
                .on_hover_text("the nuclei's spheres at their display radius, shrunk while an orbital is drawn so they do not hide its lobes; off, you fix the size yourself (display only)")
                .clicked()
            {
                st.nucleus_scale = if auto { Some(1.0) } else { None };
            }
            if st.backend != Backend::Estimated && chip(ui, st.warp_fields, "fields follow the atoms", "aid-warp").on_hover_text(WARP_TIP).clicked() {
                acts.push(Action::WarpFields(!st.warp_fields));
            }
        });
        match open {
            Some(AidRow::Plane) => {
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new("plane spacing").size(11.0).color(theme::INK_DIM));
                    probed("aid-place-bohr", ui.radio_value(&mut st.grid_in_angstrom, false, "1 Bohr"));
                    probed("aid-place-angstrom", ui.radio_value(&mut st.grid_in_angstrom, true, "1 Å"));
                });
            }
            Some(AidRow::Trails) => trail_controls(ui, st),
            None => {}
        }
        if let Some(s) = &mut st.nucleus_scale {
            // the profile and `--ui nuclei=` take any factor: drawing must not clamp it to the slider's range
            probed("nucleus-scale", ui.add(egui::Slider::new(s, 0.1..=2.0).clamping(egui::SliderClamping::Edits).fixed_decimals(2).suffix("×").text("nucleus size")))
                .on_hover_text("× each element's display radius");
        }
    });
    ui.data_mut(|d| d.insert_temp(id, open));
    if reset {
        reset_aids(st, acts);
    }
}

/// The trails' length and opacity (`crate::trails`, user request 2026-10-17): display only, written
/// straight to `UiState`, never sent to the worker.
fn trail_controls(ui: &mut Ui, st: &mut UiState) {
    use crate::trails;
    let mut x = trails::length_to_slider(st.trail_fs);
    let length = egui::Slider::new(&mut x, 0.0..=1.0)
        .text("trail length")
        .custom_formatter(|v, _| trails::length_text(trails::slider_to_length(v)))
        .custom_parser(|s| trails::parse_length(s).map(trails::length_to_slider));
    if probed("trail-length", ui.add(length))
        .on_hover_text("how long a trail point stays visible, in SIMULATED time (5 fs – 10 ps, logarithmic): the trail covers the same motion at any play speed, whether the simulation runs at 1 or 10 000 steps per tick. Type a value like 250 or 2 ps")
        .changed()
    {
        st.trail_fs = trails::slider_to_length(x);
    }
    let mut pct = (st.trail_opacity * 100.0).round();
    if probed("trail-opacity", ui.add(egui::Slider::new(&mut pct, 0.0..=100.0).text("trail opacity").suffix(" %").fixed_decimals(0)))
        .on_hover_text("the newest end's opacity; the trail fades linearly to fully transparent at the chosen length")
        .changed()
    {
        st.trail_opacity = pct / 100.0;
    }
}

/// "fields follow the atoms" (every solver).
pub const WARP_TIP: &str = "the density and orbitals were solved at one geometry; while the atoms move on (the estimated solver, or between DFT steps) the field is warped with them — each point moves with its nearby atoms — until the next solve. A display approximation: it moves what was solved and cannot make new chemistry (a bond that broke since is drawn stretched until the next solve)";

/// ESTIMATED MODE CLOUD (`crate::cloud`; `sim` module doc, "The cloud"): the display warp, atomic
/// superposition between DFT clouds and the display refresh, as chips.
fn cloud_card(ui: &mut Ui, st: &mut UiState, acts: &mut Vec<Action>) {
    egui::Frame::new().fill(theme::PANEL_2).stroke(Stroke::new(1.0, theme::LINE)).corner_radius(8).inner_margin(egui::Margin::symmetric(10, 8)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(theme::caps("Estimated mode cloud"));
        let (mut sup, mut refresh, mut every) = (st.cloud_superposition, st.cloud_refresh, st.cloud_refresh_s);
        let mut changed = false;
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = vec2(5.0, 5.0);
            if chip(ui, st.warp_fields, "fields follow the atoms", "aid-warp").on_hover_text(WARP_TIP).clicked() {
                acts.push(Action::WarpFields(!st.warp_fields));
            }
            if chip(ui, sup, "atoms between solves", "cloud-superposition")
                .on_hover_text("atomic superposition between DFT clouds — estimated solver: the model has no electrons, so between real densities the cloud is the teacher's own isolated atoms placed at the current nuclei — where the atoms are, not their bonds — blended with the last real density as the atoms move away from it. Labelled \"atomic superposition — not solved\". Needs the GPU teacher (its background solver computes the atoms once per species and grid)")
                .clicked()
            {
                sup = !sup;
                changed = true;
            }
            if chip(ui, refresh, &format!("refresh every {every:.0} s"), "cloud-refresh")
                .on_hover_text("estimated solver, while playing: a display-only DFT solve at the current geometry on the background solver (≈ 0.1–0.2 s for H₂ at 49³; it waits while an audit runs, and goes before the next one — at full speed, where audits run back to back, it costs ≈ solve time / N of them), labelled with its time. Not a training sample. Needs the GPU teacher")
                .clicked()
            {
                refresh = !refresh;
                changed = true;
            }
        });
        if refresh {
            changed |= probed("cloud-refresh-s", ui.add(egui::Slider::new(&mut every, 1.0..=10.0).suffix(" s").fixed_decimals(0).text("refresh every"))).changed();
        }
        if changed {
            acts.push(Action::Cloud { superposition: sup, refresh, refresh_s: every });
        }
    });
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


/// The tooltip of View ▸ Bonds & labels ▸ guess bonds from experience (`crate::bond_experience`).
pub const GUESS_BONDS_TIP: &str = "estimated solver: between DFT solves the bonds shown are the last DFT step's (dashed and faded). Every fresh DFT bond graph is remembered — each pair's elements, distance, how many atoms sit within 8 Bohr of each end, and whether the density bonded it (in the shared collection's folder when the model learns into one, else for this session). With this on, while the bonds are stale each pair is looked up among pairs seen before with the same elements and neighbour counts within 0.1 Bohr: if at least three were seen and all agree, the view follows them — a mint bond where they were bonded, no bond where they were not — and otherwise keeps the last DFT step's bond, dashed. Angle arcs between guessed bonds are mint too, and the angle and length labels on them mint-tinted and half-faded. Learned from this app's own DFT, nothing typed in per element; molecule names and the Selection panel stay the last DFT graph's. Display only; on by default";

/// Bonds & labels (Q6 and Q7 of Appendix B, M6): the bonds read off the density, the molecule
/// names and their style, the composition badge, angles, lengths, partial charges, the guess from
/// experience (estimated solver), label opacity.
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
    if st.backend == Backend::Estimated {
        probed("guess-bonds", ui.checkbox(&mut st.guess_bonds, "guess bonds from experience")).on_hover_text(GUESS_BONDS_TIP);
    }
    ui.horizontal(|ui| {
        ui.label("label opacity");
        ui.add(egui::Slider::new(&mut st.label_opacity, 0.0..=1.0).show_value(false))
            .on_hover_text("the labels drawn in the 3D view: molecule names, bond lengths, angles and partial charges (display only)");
        ui.label(format!("{:.0} %", st.label_opacity * 100.0));
    });
    ui.label(rich(HONEST_BONDS, 11.0, theme::INK_DIM));
}

/// A fresh state's values (the app's defaults), for the resets.
fn defaults() -> UiState {
    UiState::new("h2", &crate::scene::Scene::preset("h2").expect("the h2 preset"))
}

/// The LOOK card's ↺: the look and the partial core back to the app's defaults — the render mode
/// and the partial core by their actions (so a recording sees it), only where they differ; the rest
/// written directly.
fn reset_look(st: &mut UiState, acts: &mut Vec<Action>) {
    let d = defaults();
    Look::of(&d).apply(st, acts);
    if st.show_partial_core != d.show_partial_core {
        acts.push(Action::ShowPartialCore(d.show_partial_core));
    }
}

/// Camera's ↺: the Camera view and its lens and look, motion blur, the background.
fn reset_camera(st: &mut UiState) {
    let d = defaults();
    macro_rules! back {
        ($($f:ident),*) => { $(st.$f = d.$f;)* };
    }
    back!(camera_view, exposure_ev, shutter, fov_deg, camera_solidity, camera_size, camera_smooth, motion_blur, background);
}

/// Bonds and labels' ↺: the switches that are actions by their actions, the rest directly.
fn reset_bonds(st: &mut UiState, acts: &mut Vec<Action>) {
    let d = defaults();
    for (now, was, act) in [
        (st.show_bonds, d.show_bonds, Action::ShowBonds as fn(bool) -> Action),
        (st.show_labels, d.show_labels, Action::ShowLabels),
        (st.show_badge, d.show_badge, Action::ShowBadge),
        (st.show_angles, d.show_angles, Action::ShowAngles),
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
    back!(show_bond_lengths, show_charges, guess_bonds, label_opacity);
}

/// Display aids' ↺: smooth motion by its action, the rest directly.
fn reset_aids(st: &mut UiState, acts: &mut Vec<Action>) {
    let d = defaults();
    if st.smooth_motion != d.smooth_motion {
        acts.push(Action::SmoothMotion(d.smooth_motion));
    }
    macro_rules! back {
        ($($f:ident),*) => { $(st.$f = d.$f;)* };
    }
    back!(show_cell, show_floor, show_axes, show_place_grid, grid_in_angstrom, show_trails, trail_fs, trail_opacity, show_knowledge, show_ghost, nucleus_scale);
}

/// Reset all view settings: the look, Camera, Bonds and labels and Display aids back to a fresh
/// state's values (each section's own ↺ in turn). What the view shows (the Orbitals panel's
/// `show`), the estimated cloud and the panel's folds are left alone.
fn reset(st: &mut UiState, acts: &mut Vec<Action>) {
    reset_look(st, acts);
    reset_camera(st);
    reset_bonds(st, acts);
    reset_aids(st, acts);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sections_round_trip_through_the_profile_word_list() {
        assert_eq!(Sections::default().encode(), "");
        assert_eq!(Sections::parse(""), Sections::default());
        assert_eq!(Sections::all(true).encode(), "finetune,camera,bonds");
        let s = Sections::parse(" Bonds , what,looks,aids,bogus,camera ");
        assert!(!s.is_open(Section::FineTune) && s.is_open(Section::Camera) && s.is_open(Section::Bonds), "the old sections' words are dropped");
        assert_eq!(s.encode(), "camera,bonds", "panel order");
        for x in Section::ALL {
            assert_eq!(Sections::parse(x.id()).encode(), x.id());
        }
    }

    /// The presets are four different bundles; Glow is the user's style to the digit.
    #[test]
    fn the_presets_are_distinct_bundles() {
        for (i, a) in LookPreset::ALL.iter().enumerate() {
            for b in &LookPreset::ALL[i + 1..] {
                assert_ne!(a.look(), b.look(), "{a:?} / {b:?}");
            }
        }
        let g = GLOW;
        assert_eq!((g.volume_mode, g.dots_by_orbital, g.shimmer, g.dots_float, g.dots_glow), (VOLUME_DOTS, true, true, true, true));
        assert_eq!((g.dots_drift, g.dots_amount, g.dots_feather, g.dots_size, g.dots_blur, g.dots_brightness, g.bloom, g.bloom_threshold), (0.15, 0.5, 0.0, 0.42, 0.6, 1.0, 0.87, 0.56));
        assert!(!SCIENTIFIC.dots_glow && SCIENTIFIC.bloom == 0.0 && !SCIENTIFIC.shimmer && !SCIENTIFIC.dots_by_orbital && SCIENTIFIC.volume_mode == VOLUME_ISO);
        assert!(!CLEAN.shimmer && CLEAN.bloom == 0.0);
        assert!(PRESENT.dots_brightness > GLOW.dots_brightness && PRESENT.bloom > GLOW.bloom && PRESENT.dots_drift < GLOW.dots_drift);
    }
}
