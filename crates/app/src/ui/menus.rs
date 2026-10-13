//! The top bar: brand, the File / Scene / View / Physics / Help menus (what is set once and left
//! alone), the scene chip (preset, grid, nuclei, electrons, charge, multiplicity; click for the
//! preset picker and Reset) and the Ctrl K search button. In the compact layout the five menus
//! fold into one "☰ Menu". Also the Help windows ("Physics, honestly").
use super::search;
use super::{gravity_au, icons, probed, theme, Action, HelpTopic, PanelInput, UiState};
use crate::scene::{model_path, Scene};
use crate::sim::{Backend, DEFAULT_DT_FS};
use egui::{Button, Frame, Margin, Popup, PopupCloseBehavior, PopupKind, RichText, Stroke, Ui, Vec2};
use sim::dynamics::DT_WARN_FS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MenuId {
    File,
    Scene,
    View,
    Physics,
    Help,
}

impl MenuId {
    pub const ALL: [MenuId; 5] = [MenuId::File, MenuId::Scene, MenuId::View, MenuId::Physics, MenuId::Help];

    pub fn title(self) -> &'static str {
        match self {
            MenuId::File => "File",
            MenuId::Scene => "Scene",
            MenuId::View => "View",
            MenuId::Physics => "Physics",
            MenuId::Help => "Help",
        }
    }
}

/// The popup id of a menu (the search opens menus by it).
pub fn popup_id(m: MenuId) -> egui::Id {
    egui::Id::new(("menu", m as u8))
}

pub fn top_bar(root: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, compact: bool, acts: &mut Vec<Action>) {
    egui::Panel::top("top-bar")
        .exact_size(if compact { 34.0 } else { 38.0 })
        .resizable(false)
        .frame(Frame::new().fill(theme::PANEL).inner_margin(Margin::symmetric(10, 0)).stroke(Stroke::new(1.0, theme::LINE)))
        .show(root, |ui| {
            ui.horizontal_centered(|ui| {
                brand(ui, compact);
                if compact {
                    let r = ui.add(Button::new("☰ Menu").frame_when_inactive(false));
                    menu_popup(&r, egui::Id::new("menu-all")).show(|ui| {
                        for m in MenuId::ALL {
                            egui::containers::menu::SubMenuButton::new(m.title()).ui(ui, |ui| {
                                ui.set_min_width(340.0);
                                contents(ui, st, input, m, acts)
                            });
                        }
                    });
                } else {
                    for m in MenuId::ALL {
                        let open = Popup::is_id_open(ui.ctx(), popup_id(m));
                        let r = ui.add(Button::selectable(open, m.title()).frame_when_inactive(false));
                        menu_popup(&r, popup_id(m)).show(|ui| {
                            ui.set_min_width(350.0);
                            contents(ui, st, input, m, acts)
                        });
                    }
                }
                ui.add_space(12.0);
                scene_chip(ui, st, input, compact, acts);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| search_button(ui, st, compact));
            });
        });
}

/// A menu's popup: toggled by its button, kept open while its own widgets are used (sliders,
/// fields), closed by a click elsewhere or Esc.
fn menu_popup(r: &egui::Response, id: egui::Id) -> Popup<'static> {
    Popup::from_toggle_button_response(r)
        .id(id)
        .kind(PopupKind::Menu)
        .layout(egui::Layout::top_down(egui::Align::Min))
        .close_behavior(PopupCloseBehavior::CloseOnClickOutside)
        .gap(4.0)
}

fn brand(ui: &mut Ui, compact: bool) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(18.0), egui::Sense::hover());
    let c = rect.center();
    ui.painter().circle_filled(c, 9.0, theme::GLOW.gamma_multiply(0.25));
    ui.painter().circle_filled(c, 6.0, theme::ACCENT.gamma_multiply(0.8));
    ui.painter().circle_filled(c - Vec2::splat(1.0), 2.2, egui::Color32::WHITE);
    if !compact {
        ui.label(RichText::new("Atomic Sandbox").strong().size(13.5));
        ui.add_space(8.0);
    }
}

/// Electrons in the scene (Σ Z − charge, at least 0).
fn electrons(scene: &Scene) -> i32 {
    let protons: i32 = scene.nuclei.iter().map(|n| n.z as i32).sum();
    (protons - scene.charge).max(0)
}

fn scene_chip(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, compact: bool, acts: &mut Vec<Action>) {
    let s = input.scene;
    let e = electrons(s);
    let mult = match s.multiplicity {
        None => format!("auto ({})", 1 + (e as u32 % 2)),
        Some(m) => m.to_string(),
    };
    let detail = if compact {
        format!("{}³ · {} nuclei", s.grid_n, s.nuclei.len())
    } else {
        format!(
            "{}³ × {:.3} Bohr · {} nuclei · {e} e⁻ · charge {} · mult {mult}",
            s.grid_n,
            s.grid_h,
            s.nuclei.len(),
            s.charge
        )
    };
    let mut job = egui::text::LayoutJob::default();
    let fmt = |c: egui::Color32, mono: bool| egui::TextFormat {
        color: c,
        font_id: if mono { egui::FontId::monospace(11.5) } else { egui::FontId::proportional(12.5) },
        valign: egui::Align::Center,
        ..Default::default()
    };
    if !compact {
        job.append("scene  ", 0.0, fmt(theme::INK_DIM, false));
    }
    job.append(&st.preset, 0.0, fmt(theme::INK, false));
    job.append(&format!("   {detail}  ▾"), 0.0, fmt(theme::INK_DIM, true));
    let id = egui::Id::new("scene-chip");
    let r = ui
        .add(Button::new(job).fill(theme::PANEL_2).stroke(Stroke::new(1.0, theme::LINE)))
        .on_hover_text("the scene: click to switch preset or reset it");
    menu_popup(&r, id).show(|ui| presets(ui, st, acts));
}

fn search_button(ui: &mut Ui, st: &mut UiState, compact: bool) {
    let text = if compact { "      Ctrl K".to_string() } else { "      Find a control or setting…          Ctrl K".to_string() };
    let r = ui
        .add(Button::new(RichText::new(text).color(theme::INK_FAINT)).fill(theme::APP).stroke(Stroke::new(1.0, theme::LINE)))
        .on_hover_text("search every control by name (Ctrl K)");
    icons::paint(
        ui.painter(),
        egui::Rect::from_center_size(egui::pos2(r.rect.left() + 16.0, r.rect.center().y), Vec2::splat(14.0)),
        icons::Icon::Search,
        theme::INK_FAINT,
    );
    if r.clicked() {
        st.search.toggle();
    }
}

/// A menu's contents.
fn contents(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, m: MenuId, acts: &mut Vec<Action>) {
    match m {
        MenuId::File => file_menu(ui, st, input, acts),
        MenuId::Scene => scene_menu(ui, st, input, acts),
        MenuId::View => view_menu(ui, st, acts),
        MenuId::Physics => physics_menu(ui, st, input, acts),
        MenuId::Help => help_menu(ui, st),
    }
}

/// Runs `add` and marks its rect with a search anchor.
fn anchored<R>(ui: &mut Ui, st: &mut UiState, anchor: &str, add: impl FnOnce(&mut Ui, &mut UiState) -> R) -> R {
    let r = ui.scope(|ui| add(ui, st));
    search::mark(ui, st, anchor, r.response.rect);
    r.inner
}

/// The preset picker and Reset (File ▸ Presets and the scene chip).
fn presets(ui: &mut Ui, st: &mut UiState, acts: &mut Vec<Action>) {
    theme::section(ui, "Presets");
    ui.horizontal(|ui| {
        for name in Scene::PRESETS {
            if ui.selectable_label(st.preset == *name, *name).clicked() && st.preset != *name {
                acts.push(Action::Preset(name.to_string()));
            }
        }
        ui.add_space(8.0);
        if ui.button("Reset").on_hover_text("reload the preset (pauses; time restarts)").clicked() {
            acts.push(Action::Preset(st.preset.clone()));
        }
    });
}

fn file_menu(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    anchored(ui, st, "m-presets", |ui, st| presets(ui, st, acts));
    ui.separator();
    anchored(ui, st, "m-scene-file", |ui, st| {
        theme::section(ui, "Scene file");
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(&mut st.scene_path).desired_width(170.0));
            if ui.button("Save").on_hover_text("positions and velocities as they are now").clicked() {
                acts.push(Action::SaveScene);
            }
            if ui
                .button("Load")
                .on_hover_text("estimated solver: the model sidecar next to the file (same name, .model) is loaded with it when it exists")
                .clicked()
            {
                acts.push(Action::LoadScene);
            }
        });
        theme::note(
            ui,
            format!(
                "Load also loads {} in estimated mode (unless a sampling pass is running).",
                model_path(&st.scene_path).display()
            ),
        );
    });
    ui.separator();
    anchored(ui, st, "m-model-file", |ui, st| {
        ui.horizontal(|ui| {
            theme::section(ui, "Estimated model");
            ui.label(RichText::new(model_path(&st.scene_path).display().to_string()).monospace().size(11.0).color(theme::INK_DIM));
        });
        super::dock::model_file_buttons(ui, st, input, acts);
        if st.backend != Backend::Estimated {
            theme::note(ui, "the estimated solver's model: pick Estimated in the toolbar's Solver group first");
        }
    });
}

fn scene_menu(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    let scene = input.scene;
    let protons: i32 = scene.nuclei.iter().map(|n| n.z as i32).sum();
    let electrons = electrons(scene);
    theme::note(ui, "charge, multiplicity, σ and grid edits pause the run and restart its time");
    anchored(ui, st, "m-electrons", |ui, _| {
        theme::section(ui, "Electrons");
        egui::Grid::new("electrons").num_columns(2).show(ui, |ui| {
            ui.label("charge");
            let mut c = scene.charge;
            // at least one electron must remain; anions down to −4
            if ui.add(egui::DragValue::new(&mut c).range(-4..=(protons - 1).max(0)).speed(0.05)).changed() {
                acts.push(Action::SetCharge(c));
            }
            ui.end_row();
            ui.label("multiplicity");
            ui.horizontal(|ui| {
                let mut auto = scene.multiplicity.is_none();
                let mut m = scene.multiplicity.unwrap_or(1 + (electrons as u32 % 2));
                let toggled = ui.checkbox(&mut auto, "auto").changed();
                let edited = ui.add_enabled(!auto, egui::DragValue::new(&mut m).range(1..=11).speed(0.05)).changed();
                if toggled || edited {
                    acts.push(Action::SetMultiplicity(if auto { None } else { Some(m) }));
                }
            });
            ui.end_row();
            ui.label("σ (nuclear smearing)");
            ui.horizontal(|ui| {
                let mut auto = scene.sigma.is_none();
                let mut s = scene.sigma.unwrap_or(scene.grid_h);
                let toggled = ui.checkbox(&mut auto, "auto = h").changed();
                let edited = ui
                    .add_enabled(!auto, egui::DragValue::new(&mut s).range(0.05..=1.0).speed(0.005).suffix(" Bohr"))
                    .changed();
                if toggled || edited {
                    acts.push(Action::SetSigma(if auto { None } else { Some(s) }));
                }
            });
            ui.end_row();
        });
    });
    ui.separator();
    anchored(ui, st, "m-grid", |ui, st| {
        theme::section(ui, "Grid");
        ui.horizontal(|ui| {
            ui.label("n");
            // n must be odd: edit k with n = 2k + 1, shown and typed as n
            let mut k = (st.grid_n.max(1) - 1) / 2;
            ui.add(
                egui::DragValue::new(&mut k)
                    .range(8..=48)
                    .speed(0.1)
                    .custom_formatter(|v, _| format!("{}", 2 * v as usize + 1))
                    .custom_parser(|s| s.trim().parse::<f64>().ok().map(|n| ((n - 1.0) / 2.0).round())),
            );
            st.grid_n = 2 * k + 1;
            ui.label("h");
            ui.add(egui::DragValue::new(&mut st.grid_h).range(0.1..=0.5).speed(0.002).fixed_decimals(3).suffix(" Bohr"));
            let pending = st.grid_n != scene.grid_n || st.grid_h != scene.grid_h;
            if probed("apply-grid", ui.add_enabled(pending, Button::new("Apply grid"))).clicked() {
                acts.push(Action::ApplyGrid);
            }
        });
        let points = (st.grid_n as f64).powi(3);
        ui.label(
            RichText::new(format!(
                "cost {}³ × {electrons} electrons ≈ {:.2} M points (box {:.1} Bohr; now {}³ at {})",
                st.grid_n,
                points * electrons.max(1) as f64 / 1e6,
                (st.grid_n - 1) as f64 * st.grid_h,
                scene.grid_n,
                scene.grid_h
            ))
            .monospace()
            .size(10.5)
            .color(theme::INK_DIM),
        );
    });
    ui.separator();
    // SEAM(Q4, m5b-viewer-extras): container walls + size slider land here (Appendix B).
    anchored(ui, st, "m-container", |ui, _| {
        theme::section(ui, "Container");
        theme::queued_row(ui, "Container walls");
        theme::note(ui, "Smooth repulsive wall inside the cell, booked into E conserved; size ≤ the DFT cell.");
    });
}

fn view_menu(ui: &mut Ui, st: &mut UiState, acts: &mut Vec<Action>) {
    anchored(ui, st, "m-density", |ui, st| {
        theme::section(ui, "Density");
        ui.horizontal(|ui| {
            ui.label("render as");
            if let Some(m) = theme::segmented(ui, st.volume_mode, &[(0, "volume", ""), (1, "isosurface", "")]) {
                acts.push(Action::VolumeMode(m));
            }
        });
        let volume = st.volume_mode == 0;
        ui.add_enabled(volume, egui::Slider::new(&mut st.gamma, 0.2..=1.0).text("gamma"));
        ui.add_enabled(volume, egui::Slider::new(&mut st.opacity, 0.1..=20.0).logarithmic(true).text("opacity"));
        ui.add_enabled(!volume, egui::Slider::new(&mut st.iso, 0.01..=0.9).logarithmic(true).text("iso (× ρ max)"));
        // SEAM(Q5, m5b-viewer-extras): the live-cloud refresh control lands here.
        ui.horizontal(|ui| {
            ui.add_enabled(false, egui::Label::new("refresh cloud every 1.0 s"));
            theme::queued_tag(ui);
        });
    });
    ui.separator();
    anchored(ui, st, "m-display", |ui, st| {
        theme::section(ui, "Display aids");
        ui.horizontal(|ui| {
            ui.checkbox(&mut st.show_cell, "cell").on_hover_text("the simulation box: nuclei must stay inside it");
            ui.checkbox(&mut st.show_floor, "floor grid");
            ui.checkbox(&mut st.show_axes, "axes");
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut st.show_place_grid, "placement plane").on_hover_text("Place tool: the plane the next click lands on");
            ui.label("spacing");
            ui.radio_value(&mut st.grid_in_angstrom, false, "1 Bohr");
            ui.radio_value(&mut st.grid_in_angstrom, true, "1 Å");
        });
        ui.checkbox(&mut st.smooth_motion, "smooth motion")
            .on_hover_text("interpolates between simulation snapshots (one publish interval of latency)");
        // SEAM(Q6, Q7): bond lines / molecule names and the composition badge land here.
        theme::queued_row(ui, "bond lines and molecule names");
        theme::queued_row(ui, "composition badge on hover");
    });
}

/// The dt field (toolbar and Physics menu): amber when above `DT_WARN_FS`.
pub fn dt_field(ui: &mut Ui, st: &UiState, acts: &mut Vec<Action>) -> egui::Response {
    let mut dt = st.dt_fs;
    let warn = dt > DT_WARN_FS;
    let r = ui.scope(|ui| {
        if warn {
            ui.visuals_mut().override_text_color = Some(super::AMBER);
            ui.visuals_mut().widgets.inactive.bg_stroke = Stroke::new(1.0, super::AMBER);
        }
        ui.add(egui::DragValue::new(&mut dt).range(0.01..=2.0).speed(0.005).fixed_decimals(3).suffix(" fs"))
    });
    let r = r.inner.on_hover_text(format!(
        "time step; the default {DEFAULT_DT_FS} fs resolves the ~10 fs H–X stretch, above {DT_WARN_FS} fs it is under-resolved. Changing dt restarts the drift history."
    ));
    if r.changed() {
        acts.push(Action::SetDt(dt));
    }
    r
}

fn physics_menu(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    anchored(ui, st, "m-thermostat", |ui, st| {
        ui.horizontal(|ui| {
            theme::section(ui, "Thermostat");
            theme::info(ui, "Bussi CSVR (canonical velocity rescaling). Switching it on with the nuclei at rest draws Maxwell–Boltzmann velocities at T first.");
        });
        ui.horizontal(|ui| {
            let (mut on, mut t, mut tau) = (st.thermostat_on, st.thermostat_t, st.thermostat_tau);
            let toggled = probed("thermostat", ui.checkbox(&mut on, "thermostat (Bussi CSVR)"))
                .on_hover_text("Bussi CSVR (canonical velocity rescaling). Switching it on with the nuclei at rest draws Maxwell–Boltzmann velocities at T first.")
                .changed();
            let tc = ui
                .add(egui::DragValue::new(&mut t).range(0.0..=5000.0).speed(1.0).suffix(" K"))
                .on_hover_text("target temperature")
                .changed();
            let tauc = ui
                .add(egui::DragValue::new(&mut tau).range(1.0..=1000.0).speed(0.2).prefix("τ ").suffix(" fs"))
                .on_hover_text("coupling time: smaller = tighter control, more disturbed dynamics")
                .changed();
            if toggled || tc || tauc {
                acts.push(Action::Thermostat { on, t, tau });
            }
        });
    });
    ui.separator();
    anchored(ui, st, "m-field", |ui, st| {
        ui.horizontal(|ui| {
            theme::section(ui, "External field");
            theme::info(ui, "invisible at atomic scale, by design (½ g t² over 1 ps ≈ 1e-11 Bohr)");
        });
        ui.horizontal(|ui| {
            ui.label("uniform acceleration (gravity hook)").on_hover_text("invisible at atomic scale, by design");
            if probed("gravity-off", ui.selectable_label(!st.gravity, "off")).clicked() {
                acts.push(Action::Gravity(false));
            }
            if probed("gravity-g", ui.selectable_label(st.gravity, "g = 9.8 m/s² (−y)"))
                .on_hover_text(format!(
                    "{:.3e} Bohr per a.u.² — invisible at atomic scale, by design (½ g t² over 1 ps ≈ 1e-11 Bohr)",
                    gravity_au()
                ))
                .clicked()
            {
                acts.push(Action::Gravity(true));
            }
        });
        if let Some(s) = input.snapshot {
            let a = s.dynamics.scene.external_field;
            if a != [0.0; 3] {
                theme::note(
                    ui,
                    format!(
                        "field |a| = {:.3e} Bohr/a.u.² (invisible at atomic scale, by design)",
                        (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
                    ),
                );
            }
        }
    });
    ui.separator();
    anchored(ui, st, "m-com", |ui, st| {
        let mut com = st.com_correction;
        if probed("com", ui.checkbox(&mut com, "COM correction"))
            .on_hover_text("subtract the mean provider force: the grid's egg-box net force (~3e-4 Ha/Bohr at h = 0.25) would otherwise push a free molecule across the box")
            .changed()
        {
            acts.push(Action::ComCorrection(com));
        }
    });
    anchored(ui, st, "m-dt", |ui, st| {
        ui.horizontal(|ui| {
            ui.label("time step dt");
            dt_field(ui, st, acts);
        });
        theme::note(ui, format!("Also in the toolbar. Above {DT_WARN_FS} fs the field turns amber and the warning appears in Readouts."));
    });
    ui.separator();
    // SEAM(Q1, Q8, m5b-viewer-extras): draft mode and the relax button land here (Appendix B).
    anchored(ui, st, "m-queued", |ui, _| {
        ui.horizontal(|ui| {
            theme::section(ui, "Estimated solver");
            theme::queued_tag(ui);
        });
        theme::queued_row(ui, "Draft mode: guess instead of falling back (unsafe)");
        theme::note(ui, "Session only, off by default. Audits keep running and teaching the model.");
        ui.horizontal(|ui| {
            ui.add_enabled(false, Button::new("Relax geometry (FIRE)"));
            theme::queued_tag(ui);
        });
    });
}

fn help_menu(ui: &mut Ui, st: &mut UiState) {
    anchored(ui, st, "m-controls", |ui, st| {
        theme::section(ui, "Controls");
        egui::Grid::new("help-controls").num_columns(2).show(ui, |ui| {
            for (what, how) in CONTROLS.iter().take(6) {
                ui.label(*what);
                ui.label(RichText::new(*how).color(theme::INK_DIM));
                ui.end_row();
            }
        });
        if ui.button("All controls…").clicked() {
            st.help = Some(HelpTopic::Controls);
        }
    });
    ui.separator();
    anchored(ui, st, "m-honest", |ui, st| {
        theme::section(ui, "Physics, honestly");
        for (topic, label) in [
            (HelpTopic::Model, "LSDA, smeared nuclei, what the numbers mean"),
            (HelpTopic::Estimated, "The estimated solver is a model of this DFT"),
            (HelpTopic::Gravity, "Gravity is invisible at atomic scale, by design"),
            (HelpTopic::AtomCard, "The atom card is a schematic"),
            (HelpTopic::FastPlay, "Fast play: shimmer, not swing"),
        ] {
            if ui.add(Button::new(format!("{label}  ›")).frame_when_inactive(false)).clicked() {
                st.help = Some(topic);
            }
        }
        theme::note(ui, "Each opens the long-form text (also README \"Physics, honestly\").");
    });
}

/// The mouse and keyboard controls (Help ▸ Controls, row 146–148 of the mapping).
const CONTROLS: &[(&str, &str)] = &[
    ("Orbit · pan · zoom", "middle-drag or left-drag on empty space · right-drag · wheel"),
    ("Play / pause", "Space (or the Play button)"),
    ("Delete the selected nucleus", "Delete (also while playing)"),
    ("Deselect", "Esc"),
    ("Find a control", "Ctrl K"),
    ("Place an atom", "tool Place (or pick an element chip), left-click: it lands on the camera's focus plane (the blue placement grid)"),
    ("Move an atom", "tool Select, click a nucleus, drag. Paused: it moves where you drop it. Playing: it is pulled toward the cursor by a critically damped spring capped at 0.1 Ha/Bohr (the orange ring marks the target). Targets and placements stay at least 3 grid steps inside the cell; faint lines show distances to nearby atoms in Bohr and Å"),
    ("Hold an atom", "select it, Pin (a blue square marks it)"),
    ("Time", "Step advances one time step; Rewind reverses every velocity (an undisturbed trajectory retraces itself); speed = steps per ≈16 ms tick, never dt"),
];

/// The Help windows' long-form texts. The physics numbers in them (H atom, H₂ frequency, …) are a
/// second copy of README "Physics, honestly" and `crates/validate/references.md`: update them
/// together.
fn help_text(t: HelpTopic) -> (&'static str, String) {
    match t {
        HelpTopic::Controls => ("Controls", CONTROLS.iter().map(|(a, b)| format!("{a}: {b}.")).collect::<Vec<_>>().join("\n\n")),
        HelpTopic::Model => (
            "LSDA, smeared nuclei, what the numbers mean",
            "Born-Oppenheimer separation; electrons in spin-polarised Kohn-Sham DFT with the LSDA exchange-correlation functional (Dirac exchange + Perdew–Wang 92 correlation); nuclei represented as narrow Gaussians (σ = grid spacing by default) with an analytic potential; Hellmann-Feynman forces. Atomic units inside the solver; the UI converts.\n\n\
             Known gaps, documented rather than hidden: LSDA underbinds the hydrogen atom (−0.4777 Ha here vs −0.5 exact); absolute energies of anything containing oxygen or heavier are not physical at the default 0.25 Bohr spacing because the nucleus is heavily smeared — the density shapes are right, the numbers are not (smeared nuclei: absolute energies depend on σ and h); the f32 GPU path scatters energies by up to ~1e-5 Ha between solve paths.\n\n\
             Dynamics run on the smeared surface: H₂ vibrates at 3237 cm⁻¹ (period 10.3 fs) against 4401 cm⁻¹ measured, because the smeared nuclei lengthen the bond and soften its well. Water's minimum on this surface is near-linear, so pressing Play on h2o heats the molecule. These numbers are a copy: README \"Physics, honestly\" and crates/validate/references.md are where they are kept current.".into(),
        ),
        HelpTopic::Estimated => (
            "The estimated solver is a model of this DFT",
            "Estimated = a model fitted to this scene's DFT; valid only where it has been taught — it falls back to DFT when unsure.\n\n\
             It inherits every approximation of its teacher (smearing, LSDA, the box) and adds its own fit error; the uncertainty it shows is a MODEL uncertainty — how sure it is of reproducing the DFT — not a physical error bar. An untrusted model never answers: every step is DFT and is kept as a sample until a refit earns trust. A sample the model cannot represent is quarantined: kept, excluded from the fit, its neighbourhood answered by DFT.\n\n\
             Box-affected teacher answers (density on the outermost grid plane above 2e-5 × (h / 0.25 Bohr)² e/Bohr³) are used for that step but never trained on, and an audit there is recorded but not judged: the cure is a bigger box. A surrogate step has no electrons, so the cloud you see is the last DFT density, drawn faded and labelled with its time.".into(),
        ),
        HelpTopic::Gravity => (
            "Gravity is invisible at atomic scale, by design",
            format!(
                "The uniform acceleration g = 9.8 m/s² along −y is {:.3e} Bohr per a.u.². Over a picosecond it moves a nucleus by ½ g t² ≈ 1e-11 Bohr: invisible at atomic scale, which is the honest answer. Gravity exists as a hook, not as a visible effect.",
                gravity_au()
            ),
        ),
        HelpTopic::AtomCard => (
            "The atom card is a schematic",
            "The atom card is a labelled 2D Bohr-style schematic, not the simulation: the sandbox never resolves nucleons (the nucleus is a point of charge +Z and a mass), and in a molecule the electrons are shared — the density cloud in the 3D view is the real picture. Everything on it derives from Z and the isotope mass table: N = A − Z with A the mass number of the most abundant isotope, and the shell occupation is the textbook 2 / 8 / 8 / 18 filling of a neutral atom.".into(),
        ),
        HelpTopic::FastPlay => (
            "Fast play: shimmer, not swing",
            "Speed is time steps attempted per ≈16 ms tick; it never changes dt. One DFT step usually fills a tick by itself, so the measured fs/s is what counts; with the estimated solver answering (~10 µs a step) a tick runs all its steps and then waits out its 16 ms, except at 1000 (as fast as possible). Above ~30 fs/s a 10 fs bond vibration is faster than the frame rate — you will see a shimmer, not a swing; use the watchable preset (one ~10 fs H–X vibration in ≈ 0.5 s of wall time) to see motion.".into(),
        ),
    }
}

/// The open Help window.
pub fn help_window(ctx: &egui::Context, st: &mut UiState) {
    let Some(topic) = st.help else { return };
    let (title, text) = help_text(topic);
    let mut open = true;
    egui::Window::new(title)
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .default_width(460.0)
        .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
        .show(ctx, |ui| {
            ui.set_max_width(460.0);
            for para in text.split("\n\n") {
                ui.label(para);
                ui.add_space(4.0);
            }
        });
    if !open {
        st.help = None;
    }
}

