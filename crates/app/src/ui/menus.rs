//! The top bar: brand, the File / Scene / View / Physics / Help menus (what is set once and left
//! alone), the scene chip (preset, grid, nuclei, electrons, charge, multiplicity; click for the
//! preset picker and Reset) and the Ctrl K search button. In the compact layout the five menus
//! fold into one "☰ Menu". Also the Help windows ("Physics, honestly").
use super::dock::orbitals;
use super::search;
use super::{gravity_au, icons, probed, theme, Action, HelpTopic, PanelInput, ShowMode, UiState, VOLUME_DOTS, VOLUME_ISO, VOLUME_VOLUME};
use crate::scene::{model_path, Scene};
use crate::sim::{Backend, DEFAULT_DT_FS};
use egui::{Button, Frame, Margin, Popup, PopupCloseBehavior, PopupKind, RichText, Stroke, Ui, Vec2};
use sim::dynamics::{DT_WARN_FS, F_DRAG_HI, F_DRAG_LO, F_DRAG_MAX};

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
    job.append(&st.source.label(), 0.0, fmt(theme::INK, false));
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

/// The preset picker and Reset (File ▸ Presets and the scene chip): `Scene::PRESETS` in order —
/// empty, h, h2, he, lih, h2o. A scene loaded from a file highlights no preset (every one can be
/// picked), and Reset reloads that file (final review app I1).
fn presets(ui: &mut Ui, st: &mut UiState, acts: &mut Vec<Action>) {
    theme::section(ui, "Presets");
    ui.horizontal(|ui| {
        let current = st.source.preset().map(str::to_string);
        for name in Scene::PRESETS {
            let on = current.as_deref() == Some(*name);
            // probed by its own name (the click tests press "empty")
            if probed(name, ui.selectable_label(on, *name)).clicked() && !on {
                acts.push(Action::Preset(name.to_string()));
            }
        }
        ui.add_space(8.0);
        match st.source.clone() {
            super::SceneSource::Preset(p) => {
                if probed("reset", ui.button("Reset")).on_hover_text("reload the preset (pauses; time restarts)").clicked() {
                    acts.push(Action::Preset(p));
                }
            }
            super::SceneSource::File(f) => {
                if probed("reset", ui.button("Reset")).on_hover_text(format!("reload the scene file {f} (pauses; time restarts)")).clicked() {
                    st.scene_path = f;
                    acts.push(Action::LoadScene { sidecar: true });
                }
            }
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
                acts.push(Action::LoadScene { sidecar: true });
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
    anchored(ui, st, "m-quality", |ui, st| quality_controls(ui, st, input, acts));
    ui.separator();
    anchored(ui, st, "m-grid", |ui, st| {
        theme::section(ui, "Grid");
        ui.horizontal(|ui| {
            ui.label("n");
            // n must be odd: edit k with n = 2k + 1, shown and typed as n
            let mut k = (st.grid_n.max(1) - 1) / 2;
            ui.add(
                egui::DragValue::new(&mut k)
                    .range(8..=(crate::scene::MAX_GRID_N - 1) / 2)
                    .speed(0.1)
                    .custom_formatter(|v, _| format!("{}", 2 * v as usize + 1))
                    .custom_parser(|s| s.trim().parse::<f64>().ok().map(|n| ((n - 1.0) / 2.0).round())),
            );
            st.grid_n = 2 * k + 1;
            ui.label("h");
            ui.add(egui::DragValue::new(&mut st.grid_h).range(crate::quality::H_MIN..=0.5).speed(0.002).fixed_decimals(3).suffix(" Bohr"));
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
    // Q4 of the mockup's Appendix B (M5b): container walls and their size, the piston.
    anchored(ui, st, "m-container", |ui, st| {
        theme::section(ui, "Container");
        walls_controls(ui, st, input, acts);
    });
}

/// The quality levels' probe names (the click tests).
const QUALITY_PROBES: [&str; 4] = ["quality-fast", "quality-balanced", "quality-accurate", "quality-precise"];

/// Scene ▸ Quality (`crate::quality`): Fast / Balanced / Accurate / Precise and a Custom h, each
/// with its grid for this box, its cost estimated from the scene's last solve, and the measured
/// H₂ bond length as the accuracy hint. A change is a scene edit (pauses, re-solves).
fn quality_controls(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    use crate::quality::{accuracy_hint, grid_n, level_of, scaled_cost, title, LEVELS, H_MAX, H_MIN};
    let scene = input.scene;
    theme::section(ui, "Quality");
    // M7: a level coarser than a pseudised species' h_req is refused by the provider (T8's certificate);
    // the menu says so before the click, from the same record
    let pseudised = super::pseudo::pseudised(input);
    let cost = |n: usize| match st.last_solve {
        Some((n0, s0)) => format!("≈ {}", super::duration_label(scaled_cost(s0, n0, n))),
        None => "—".into(),
    };
    egui::Grid::new("quality").num_columns(4).spacing([10.0, 3.0]).show(ui, |ui| {
        for ((name, h), probe) in LEVELS.iter().zip(QUALITY_PROBES) {
            let n = grid_n(st.box_len, *h);
            let on = (scene.grid_h - h).abs() < 5e-4;
            let r = probed(probe, ui.selectable_label(on, format!("{} · h {h:.2}", title(name))))
                .on_hover_text(format!("h = σ = {h} Bohr → {n}³ for this {:.1} Bohr box. {}", st.box_len, accuracy_hint(name)));
            if r.clicked() && !on {
                acts.push(Action::Quality(*h));
            }
            ui.label(RichText::new(format!("{n}³")).monospace().size(10.5));
            ui.label(RichText::new(cost(n)).monospace().size(10.5).color(theme::INK_DIM));
            match super::pseudo::needs_finer(&pseudised, *h) {
                Some(why) => ui.label(RichText::new(why).size(10.5).color(super::AMBER)).on_hover_text("this level's grid is coarser than the pseudopotential is certified for (T8): the solver refuses such a scene rather than answer it wrongly"),
                None => ui.label(RichText::new(accuracy_hint(name)).size(10.5).color(theme::INK_DIM)),
            };
            ui.end_row();
        }
    });
    ui.horizontal(|ui| {
        ui.label("custom h");
        let r = probed(
            "quality-custom",
            ui.add(egui::Slider::new(&mut st.quality_slider, H_MIN..=H_MAX).step_by(0.001).fixed_decimals(3).suffix(" Bohr")),
        )
        .on_hover_text("any spacing in 0.08–0.35 Bohr (snapped to 3 decimals); applied when you let go. Smaller is slower and closer to the σ → 0 answer");
        if r.drag_stopped() || (r.changed() && !r.dragged()) {
            acts.push(Action::Quality(st.quality_slider));
        }
        let n = grid_n(st.box_len, crate::quality::snap(st.quality_slider));
        ui.label(RichText::new(format!("{n}³ · {}", cost(n))).monospace().size(10.5).color(theme::INK_DIM));
        if let Some(why) = super::pseudo::needs_finer(&pseudised, crate::quality::snap(st.quality_slider)) {
            ui.label(RichText::new(why).size(10.5).color(super::AMBER));
        }
    });
    let level = level_of(scene.grid_h).map_or_else(|| format!("Custom h {:.3}", scene.grid_h), title);
    let l = (scene.grid_n as f64 - 1.0) * scene.grid_h;
    ui.label(
        RichText::new(format!(
            "now: {level} — {}³ × {:.3} Bohr, box {:.2} Bohr = {:.2} Å; σ {}",
            scene.grid_n,
            scene.grid_h,
            l,
            units::bohr_to_angstrom(l),
            scene.sigma.map_or("= h".to_string(), |s| format!("{s} Bohr (set)"))
        ))
        .monospace()
        .size(10.5)
        .color(theme::INK_DIM),
    );
    theme::note(
        ui,
        match st.last_solve {
            Some((n0, s0)) => format!("costs ≈, rough: this scene's last solve ({n0}³, {}) × (new n / n)³ — the GPU is latency-bound on small grids, so this can be well off either way", super::duration_label(s0)),
            None => "costs appear after the first solve (estimated from it × (new n / n)³)".into(),
        },
    );
}

/// Container walls (`sim::dynamics` module doc, "Container walls"): the checkbox and the
/// half-width slider, capped at the cell's allowed maximum (re-clamped after a grid change: the
/// driver moves the walls in by itself, the slider follows). A move while playing is the piston:
/// its work is booked (Readouts' "of which walls (piston)" row).
fn walls_controls(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    let max = super::max_wall_half_width(input.scene);
    let min = super::MIN_WALL_HALF_WIDTH.min(max);
    st.wall_half_width = st.wall_half_width.clamp(min, max.max(min));
    let (mut on, mut half) = (st.walls_on, st.wall_half_width);
    let mut send = false;
    ui.horizontal(|ui| {
        send |= probed("walls", ui.checkbox(&mut on, "container walls"))
            .on_hover_text(format!(
                "smooth repulsive walls {} Bohr thick inside the box drawn in the view: a nucleus with up to 1 Ha of kinetic energy per axis turns before the face, so the solver's cell is never left. On by default, 1.5 Bohr inside the cell's face (every solver, the estimated one included); untick for unconfined physics. Booked in E conserved (potential + the work of moving them)",
                sim::dynamics::WALL_LAYER
            ))
            .changed();
        let slider = ui
            .add_enabled(on, egui::Slider::new(&mut half, min..=max.max(min)).suffix(" Bohr").fixed_decimals(2))
            .on_hover_text("half-width of the container (the DFT cell is the limit: grow the grid to go further). Shrinking it while playing is a piston: it squeezes the atoms, and its work is booked");
        send |= slider.changed();
    });
    if on {
        theme::note(
            ui,
            format!(
                "box ±{:.2} Bohr = ±{:.2} Å (max {:.2} Bohr: the cell less its solver margin)",
                half,
                units::bohr_to_angstrom(half),
                max
            ),
        );
        if let Some(w) = input.snapshot.and_then(|s| s.dynamics.walls) {
            if (w.half_width - half).abs() > 1e-9 && !send {
                theme::note(ui, format!("the run's walls: ±{:.2} Bohr", w.half_width));
            }
        }
    }
    if send {
        acts.push(Action::Walls { on, half_width: half });
    }
}

/// The estimated solver's cloud (`crate::cloud`; `sim` module doc, "The cloud"): atomic
/// superposition between DFT clouds, and the display refresh.
fn cloud_controls(ui: &mut Ui, st: &UiState, acts: &mut Vec<Action>) {
    let estimated = st.backend == Backend::Estimated;
    let (mut sup, mut refresh, mut every) = (st.cloud_superposition, st.cloud_refresh, st.cloud_refresh_s);
    let mut changed = false;
    ui.add_enabled_ui(estimated, |ui| {
        changed |= ui
            .checkbox(&mut sup, "atomic superposition between DFT clouds")
            .on_hover_text("estimated solver: the model has no electrons, so between real densities the cloud is the teacher's own isolated atoms placed at the current nuclei — where the atoms are, not their bonds — blended with the last real density as the atoms move away from it. Labelled \"atomic superposition — not solved\". Needs the GPU teacher (its background solver computes the atoms once per species and grid)")
            .changed();
        ui.horizontal(|ui| {
            changed |= probed("cloud-refresh", ui.checkbox(&mut refresh, "refresh cloud every"))
                .on_hover_text("estimated solver, while playing: a display-only DFT solve at the current geometry on the background solver (≈ 0.1–0.2 s for H₂ at 49³; it waits while an audit runs, and goes before the next one — at full speed, where audits run back to back, it costs ≈ solve time / N of them), labelled with its time. Not a training sample. Needs the GPU teacher")
                .changed();
            changed |= ui.add_enabled(refresh, egui::Slider::new(&mut every, 1.0..=10.0).suffix(" s").fixed_decimals(0)).changed();
        });
    });
    if !estimated {
        theme::note(ui, "(cloud options apply to the estimated solver; DFT steps show their own density)");
    }
    if changed {
        acts.push(Action::Cloud { superposition: sup, refresh, refresh_s: every });
    }
}

/// Draft mode (`providers::surrogate`, "Draft mode"): Physics ▸ Estimated solver and Dock ▸ Model.
pub fn draft_control(ui: &mut Ui, st: &UiState, acts: &mut Vec<Action>) {
    let mut draft = st.draft;
    if probed("draft", ui.add_enabled(st.backend == Backend::Estimated, egui::Checkbox::new(&mut draft, "guess instead of falling back (unsafe)")))
        .on_hover_text("draft mode: the model answers EVERY step — also where it is untrusted, uncertain, or outside what it has learned — instead of handing the step to DFT. Beyond its 6 Bohr cutoff a bond contributes nothing (a pulled-apart pair is two free atoms), and geometries, species or charges it never saw are answered confidently wrong. The background DFT audits keep running and the model keeps learning from them, but nothing stops it answering. A red banner says when it extrapolates; atoms are tinted by their uncertainty. Scenes it cannot be asked about at all (another grid, charge or species) still go to DFT. Needs the GPU teacher for audits (none in the step).")
        .on_disabled_hover_text("the estimated solver's draft mode: pick Estimated in the toolbar's Solver group first")
        .changed()
    {
        acts.push(Action::Draft(draft));
    }
    if st.draft {
        ui.colored_label(super::RED, "draft mode: forces may be educated guesses — every step is the model's");
    }
}

fn view_menu(ui: &mut Ui, st: &mut UiState, acts: &mut Vec<Action>) {
    anchored(ui, st, "m-density", |ui, st| {
        theme::section(ui, "Density");
        // orbital-viewer spec §3: what the view shows; the Orbitals dock tab holds the list and the per-mode controls
        ui.horizontal_wrapped(|ui| {
            ui.label("show");
            theme::info(ui, format!("{} {}", orbitals::KS_NOTE, orbitals::SPIN_NOTE));
        });
        if let Some(m) = orbitals::show_selector(ui, st.show, "menu-show") {
            acts.push(Action::Show(m));
        }
        ui.horizontal(|ui| {
            ui.label("render as");
            let ori = "volume / isosurface: for an orbital or the spin density, a glowing volume with the sign's colour, or the ± isosurface. Dots: random samples of the field";
            if let Some(m) = theme::segmented(ui, st.volume_mode, &[(VOLUME_VOLUME, "volume", ori), (VOLUME_ISO, "isosurface", ori), (VOLUME_DOTS, "dots", "random samples of the field, ∝ ρ (or |ψ|², or |m|) per cell: dots are samples of where an electron is likely to be — not electrons")]) {
                acts.push(Action::VolumeMode(m));
            }
        });
        let volume = st.volume_mode == super::VOLUME_VOLUME;
        ui.add_enabled(volume, egui::Slider::new(&mut st.gamma, 0.2..=1.0).text("gamma"));
        ui.add_enabled(volume, egui::Slider::new(&mut st.opacity, 0.1..=20.0).logarithmic(true).text("opacity"));
        ui.add_enabled(st.volume_mode == super::VOLUME_ISO, egui::Slider::new(&mut st.iso, 0.01..=0.9).logarithmic(true).text("iso (× ρ max)"));
        ui.horizontal(|ui| {
            ui.add_enabled(st.volume_mode == super::VOLUME_DOTS, egui::Checkbox::new(&mut st.shimmer, "shimmer (new dots every frame)"));
        });
        // M7: the pseudised species' partial core in the density shown (a display of the generator's model)
        let mut core = st.show_partial_core;
        if probed("show-partial-core", ui.checkbox(&mut core, "with partial core")).on_hover_text(super::pseudo::PARTIAL_CORE_TIP).changed() {
            acts.push(Action::ShowPartialCore(core));
        }
        ui.horizontal(|ui| {
            ui.label("colour dots by");
            ui.add_enabled_ui(st.volume_mode == super::VOLUME_DOTS && st.show == ShowMode::Coloured, |ui| {
                if let Some(by_orbital) = theme::segmented(ui, st.dots_by_orbital, &[(false, "density", "the density ramp at the dot"), (true, "orbital", "each dot in its orbital's own colour (All orbitals coloured)")]) {
                    st.dots_by_orbital = by_orbital;
                }
            });
        });
        // Q5 of Appendix B (M5b): the estimated solver's cloud
        theme::section(ui, "Estimated solver's cloud");
        cloud_controls(ui, st, acts);
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
        probed("show-ghost", ui.checkbox(&mut st.show_ghost, "show what training is solving"))
            .on_hover_text("while a sampling pass runs, the view shows the geometry the teacher is solving right now as a wireframe ghost (a lone reference atom, a random nudge with arrows, the scanned pair, the hot burst with a trail) instead of your scene, flashing green / red / grey as each sample is kept / fails / is skipped; your scene returns when the pass ends or is cancelled. Display only"
            );
        let mut smooth = st.smooth_motion;
        if ui
            .checkbox(&mut smooth, "smooth motion")
            .on_hover_text("interpolates between simulation snapshots (one publish interval of latency): the nuclei glide, and the density cloud crossfades between the last two solved densities with the same weight — a blend for display, not a transported density")
            .changed()
        {
            acts.push(Action::SmoothMotion(smooth));
        }
    });
    ui.separator();
    // Q6 and Q7 of Appendix B (M6): the bonds read off the density, the molecule labels, the badge
    anchored(ui, st, "m-bonds", |ui, st| {
        ui.horizontal(|ui| {
            theme::section(ui, "Bonds and molecules");
            theme::info(ui, super::rich(BONDS_TIP, 12.5, theme::INK));
        });
        ui.horizontal(|ui| {
            let (mut b, mut l, mut g) = (st.show_bonds, st.show_labels, st.show_badge);
            if probed("show-bonds", ui.checkbox(&mut b, "bonds")).on_hover_text(super::rich(BONDS_TIP, 12.5, theme::INK)).changed() {
                acts.push(Action::ShowBonds(b));
            }
            if probed("show-labels", ui.checkbox(&mut l, "molecule names"))
                .on_hover_text("each molecule's formula and name next to it in the view (also listed in Dock ▸ Selection): a molecule is a connected piece of the bond graph; names come from a small table of trivial names or the IUPAC rules for binary compounds, marked \"(systematic)\"")
                .changed()
            {
                acts.push(Action::ShowLabels(l));
            }
            if probed("show-badge", ui.checkbox(&mut g, "composition badge"))
                .on_hover_text("hovering a nucleus with the Select tool (or selecting one) shows its isotope, protons, neutrons and the force on it: \"¹⁶O · 8 p · 8 n · |F| 0.012 Ha/Bohr\"")
                .changed()
            {
                acts.push(Action::ShowBadge(g));
            }
        });
        ui.horizontal(|ui| {
            ui.label("name style").on_hover_text(NAME_STYLE_TIP);
            let opts: Vec<(crate::names::NameStyle, &str, &str)> = crate::names::NameStyle::ALL.iter().map(|&s| (s, s.title(), NAME_STYLE_TIP)).collect();
            if let Some(s) = theme::segmented(ui, st.name_style, &opts) {
                acts.push(Action::NameStyle(s));
            }
        });
        ui.label(super::rich(HONEST_BONDS, 11.0, theme::INK_DIM));
    });
}

/// The name-style tooltip (View ▸ Bonds and molecules).
pub const NAME_STYLE_TIP: &str = "how molecules are named (the formula always shows): common — water, ammonia (trivial names, else IUPAC binary names marked \"(systematic)\"); systematic (stoichiometric) — the IUPAC Red Book IR-5 name from the composition: dihydrogen monoxide, nitrogen trihydride, trihydrogen(1+) (a heteronuclear ion or a compound of three elements has none: formula only); IUPAC hydride — oxidane, azane, methane, borane where a parent hydride exists, else the stoichiometric name; formula only";

/// The bonds' tooltip (View ▸ Bonds and molecules).
pub const BONDS_TIP: &str = "a bond is drawn where the solved density piles up between two nuclei beyond what their free atoms would put there (a density saddle or a deformation-density maximum on the line between them, M6 spec §3). Read off the density after each converged DFT step — under the estimated solver the bonds are the last DFT step's, faded and drawn between the current positions. The line shows the strength: thicker and more opaque for more density at the bond point (log ρ_b from 0.01 to 0.3 e/Bohr³, a display scale), solid for a shared-shell (covalent) bond, dashed for a closed-shell (weak / stretched) contact, dotted when the sign of the Laplacian of the density at the bond point is within its grid noise (QTAIM, Bader). Violet: a multi-centre (ring) bond, as in H₃⁺. LSDA overbinds stretched contacts (delocalisation error): a weak contact here may be weaker or absent in reality. Heavy atoms are smeared (at the Balanced 0.25 Bohr spacing, and still at the default Accurate 0.15): the topology is expected right, the bond-point numbers are not physical";

/// The honesty lines of spec §3.5 (View menu, Selection tab).
pub const HONEST_BONDS: &str = "Bonds and names are read off the density after a converged DFT step. H₃ is not a stable molecule; H₃⁺ is.";

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

/// Physics ▸ Ionic potential (M7, spec §4.7): the three policies, the valence-energy honesty line, and for each
/// pseudised species of the latest solve its record and the grids it is certified at. The egg-box line (NLCC
/// ruling 6) is text only — a scene is never refused for it.
fn ionic_controls(ui: &mut Ui, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    use super::pseudo;
    ui.horizontal(|ui| {
        theme::section(ui, "Ionic potential");
        theme::info(ui, pseudo::IONIC_TIP);
    });
    let opts: Vec<(sim::IonicPolicy, &str, &str)> = sim::IonicPolicy::ALL.iter().map(|&p| (p, p.title(), pseudo::IONIC_TIP)).collect();
    if let Some(p) = theme::segmented(ui, input.scene.ionic, &opts) {
        acts.push(Action::Ionic(p));
    }
    theme::note(ui, "a policy change replaces the scene (pauses; time and the energy history restart). A pseudopotential is generated on first use (seconds per element) and kept for the run");
    let pseudised = pseudo::pseudised(input);
    let pseudised = &pseudised[..];
    if pseudised.is_empty() {
        if input.scene.ionic != sim::IonicPolicy::None && !input.scene.nuclei.is_empty() {
            theme::note(ui, "nothing in this scene is pseudised (H and He stay all-electron under cores only)");
        }
        return;
    }
    ui.label(RichText::new(pseudo::VALENCE_NOTE).size(11.0).color(theme::INK_DIM));
    for p in pseudised {
        ui.label(RichText::new(pseudo::species_line(p)).monospace().size(10.5).color(theme::INK_DIM));
    }
    if let Some(w) = pseudo::egg_box_warning(pseudised, input.scene.grid_h) {
        ui.label(RichText::new(w).size(11.0).color(super::AMBER)).on_hover_text(pseudo::EGG_BOX_RECORD);
    }
}

fn physics_menu(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    anchored(ui, st, "m-ionic", |ui, _| ionic_controls(ui, input, acts));
    ui.separator();
    anchored(ui, st, "m-thermostat", |ui, st| {
        ui.horizontal(|ui| {
            theme::section(ui, "Thermostat");
            theme::info(ui, "Bussi CSVR (canonical velocity rescaling). Switching it on with the nuclei at rest draws Maxwell–Boltzmann velocities at T first.");
            // display only (rule 3): the simulation stays in kelvin
            if let Some(u) = theme::segmented(
                ui,
                st.temp_units,
                &[(super::TempUnits::K, "K", "kelvin"), (super::TempUnits::C, "°C", "Celsius (display only)"), (super::TempUnits::F, "°F", "Fahrenheit (display only)")],
            ) {
                acts.push(Action::TempUnits(u));
            }
        });
        ui.horizontal(|ui| {
            let (mut on, mut t, mut tau) = (st.thermostat_on, st.thermostat_t, st.thermostat_tau);
            let toggled = probed("thermostat", ui.checkbox(&mut on, "thermostat (Bussi CSVR)"))
                .on_hover_text("Bussi CSVR (canonical velocity rescaling). Switching it on with the nuclei at rest draws Maxwell–Boltzmann velocities at T first.")
                .changed();
            // typed in the chosen scale, kept in kelvin, clamped at absolute zero
            let tu = st.temp_units;
            let mut shown = tu.from_k(t);
            let tc = probed(
                "thermostat-t",
                ui.add(egui::DragValue::new(&mut shown).range(tu.range(0.0, 5000.0)).speed(1.0).max_decimals(2).suffix(tu.suffix())),
            )
            .on_hover_text("target temperature (absolute zero is 0 K = −273.15 °C = −459.67 °F)")
            .changed();
            if tc {
                t = tu.to_k(shown);
            }
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
    anchored(ui, st, "m-drag", |ui, st| drag_controls(ui, st, acts));
    ui.separator();
    // Q1 of Appendix B (M5b): draft mode.
    anchored(ui, st, "m-draft", |ui, st| {
        theme::section(ui, "Estimated solver");
        draft_control(ui, st, acts);
        theme::note(ui, "Session only, off by default. Audits keep running and teaching the model.");
    });
    ui.separator();
    // Q8 of Appendix B (M6): relax (FIRE); also in the toolbar's Time group
    anchored(ui, st, "m-relax", |ui, _| relax_controls(ui, input, acts));
}

/// Physics ▸ Relax geometry: the button (Stop while running), the readout and the note.
fn relax_controls(ui: &mut Ui, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    ui.horizontal(|ui| {
        theme::section(ui, "Relax geometry (FIRE)");
        theme::info(ui, super::RELAX_TIP);
    });
    ui.horizontal(|ui| super::relax_toggle(ui, input, Vec2::ZERO, ("m-relax", "m-relax-stop"), acts));
    match input.snapshot.and_then(|s| s.dynamics.relax.as_ref()) {
        Some(r) => {
            for l in super::relax_lines(r) {
                theme::note(ui, l);
            }
        }
        None => {
            theme::note(ui, format!("stops when every atom's force is below {:.0e} Ha/Bohr; the dynamics pause while it runs", sim::F_TOL));
        }
    }
}

/// The drag pull's tooltip (Physics ▸ Drag pull, the Select tool, the cursor label's meaning).
pub const DRAG_TIP: &str = "how hard a drag pulls while playing: the spring's force cap. The wheel while dragging an atom (Select tool) sets it, ×1.25 per notch, 0.01–1 Ha/Bohr; it stays for the next drag. Stronger pulls tear bonds faster and heat the molecule more (all of it booked as tool work); the spring's stiffness and damping, and dt, never change";

/// The drag pull (`sim::dynamics` module doc, "Tool forces"): a slider over the driver's range
/// and Reset to the default; the wheel while dragging sets the same value.
fn drag_controls(ui: &mut Ui, st: &UiState, acts: &mut Vec<Action>) {
    ui.horizontal(|ui| {
        theme::section(ui, "Drag pull");
        theme::info(ui, DRAG_TIP);
    });
    ui.horizontal(|ui| {
        let mut f = st.drag_force;
        // no fixed decimals: egui would round the stored value to them (0.01, 0.02 … at the bottom)
        let slider = egui::Slider::new(&mut f, F_DRAG_LO..=F_DRAG_HI)
            .logarithmic(true)
            .custom_formatter(|v, _| super::sig3(v))
            .custom_parser(|t| t.trim().parse().ok())
            .suffix(" Ha/Bohr");
        if probed("drag-pull", ui.add(slider))
            .on_hover_text(DRAG_TIP)
            .changed()
        {
            acts.push(Action::DragForce(f));
        }
        if probed("drag-reset", ui.add_enabled(st.drag_force != F_DRAG_MAX, Button::new(format!("Reset to {F_DRAG_MAX}"))))
            .on_hover_text("back to the default pull, 0.1 Ha/Bohr (5.1 eV/Å)")
            .clicked()
        {
            acts.push(Action::DragForce(F_DRAG_MAX));
        }
    });
    theme::note(ui, format!("{} — the wheel while dragging an atom sets it", super::pull_label(st.drag_force)));
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
    ("Orbit · pan · zoom", "middle-drag or left-drag on empty space · right-drag · wheel (while dragging an atom the wheel sets its pull instead)"),
    ("Play / pause", "Space (or the Play button)"),
    ("Delete the selected nucleus", "Delete (also while playing)"),
    ("Deselect", "Esc"),
    ("Find a control", "Ctrl K"),
    ("Place an atom", "tool Place (or pick an element chip), left-click: it lands on the camera's focus plane (the blue placement grid)"),
    ("Move an atom", "tool Select, click a nucleus, drag. Paused: it moves where you drop it. Playing: it is pulled toward the cursor by a critically damped spring whose force is capped at the pull — 0.1 Ha/Bohr by default; the wheel while dragging sets it (×1.25 per notch, 0.01–1 Ha/Bohr, shown next to the cursor; Physics ▸ Drag pull resets it) — the orange ring marks the target. Targets and placements stay at least 3 grid steps inside the cell; faint lines show distances to nearby atoms in Bohr and Å"),
    ("Hold an atom", "select it, Pin (a blue square marks it)"),
    ("Time", "Step advances one time step; Rewind reverses every velocity (an undisturbed trajectory retraces itself); speed = steps per ≈16 ms tick, never dt"),
    ("Relax", "Relax (toolbar Time group, or Physics ▸ Relax geometry) moves the nuclei downhill with FIRE until every force is below 1e-3 Ha/Bohr, then stops at rest; Stop ends it. Under the estimated solver one DFT solve checks the result"),
    ("Element and isotope", "the element chips, or \"periodic table ›\" under them: search by symbol, name or Z; Z 1–10 can be placed, heavier elements need pseudopotentials (planned); the isotope row under the table picks the mass number. A selected nucleus's isotope: Dock ▸ Selection"),
    ("Bonds and molecules", "drawn from the density after each DFT step (View ▸ Bonds and molecules); hover a nucleus with the Select tool for its isotope, protons, neutrons and force"),
    ("Record and replay input", "start the app with --record <file> to write the 3D view's mouse and keys and the panel's actions to a replay script; --replay <file> [--preset p] [--backend b] [--screenshot out.png] runs one headlessly and checks its expect lines (exit code 1 on a failed one). The checklist's scripts are in crates/app/replays; README \"Recording and replaying input\""),
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
             Known gaps, documented rather than hidden: LSDA underbinds the hydrogen atom (−0.4777 Ha here vs −0.5 exact); absolute energies of anything containing oxygen or heavier are not physical at the Balanced 0.25 Bohr spacing (and still smeared at the default Accurate 0.15) because the nucleus is heavily smeared — the density shapes are right, the numbers are not (smeared nuclei: absolute energies depend on σ and h); the f32 GPU path carries ~1e-6 Ha of scatter between solve paths. A stretched H₂ beyond ~3 Bohr keeps a bond line and a too-high energy (≈ +0.05 Ha at dissociation) because the solver stays on the spin-symmetric solution; a broken-symmetry start is a planned solver item.\n\n\
             Dynamics run on the smeared surface: at σ = h = 0.25 H₂ vibrates at 3237 cm⁻¹ (period 10.3 fs) against 4401 cm⁻¹ measured, because the smeared nuclei lengthen the bond and soften its well. Water's minimum on the 0.25 Bohr surface is near-linear, so pressing Play on h2o heats the molecule. These numbers are a copy: README \"Physics, honestly\" and crates/validate/references.md are where they are kept current.".into(),
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

