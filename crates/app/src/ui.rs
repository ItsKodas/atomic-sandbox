//! The egui side panel: scene, tools, electrons, grid, view controls, readouts, file I/O.
//!
//! The panel never edits the scene directly: edits become [`Command`]s that `main` sends to the
//! worker (and applies to its local mirror of the scene). Only view-only settings (tool mode,
//! element, volume look, pending grid edits, file path) live in [`UiState`].
use crate::scene::{validate_grid, Scene};
use crate::sim::{Backend, Command, Snapshot};
use egui::{Color32, RichText};
use render::elements;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolMode {
    Orbit,
    Place,
    Select,
}

pub struct UiState {
    /// Z placed by the Place tool.
    pub element: u8,
    pub mode: ToolMode,
    pub preset: String,
    /// Pending grid edit, sent on "Apply grid".
    pub grid_n: usize,
    pub grid_h: f64,
    /// 0 = volume, 1 = isosurface (the `VolumeUniform::mode` values).
    pub volume_mode: u32,
    pub gamma: f32,
    pub opacity: f32,
    /// Isosurface threshold as a fraction of the density maximum.
    pub iso: f32,
    pub scene_path: String,
    /// Neutral feedback (saved, loaded); errors go to `PanelOutput::error`.
    pub status: String,
    /// The backend the user picked; the viewer re-spawns the worker when it differs from the
    /// running one (`Viewer::run_ui`).
    pub backend: Backend,
    /// A GPU context exists (else the GPU choice is disabled).
    pub gpu_available: bool,
    /// Display aids (`overlay::build`): the simulation cell, a floor grid in the xz plane, xyz
    /// axes, and (Place mode) the plane the next click lands on; grid spacing 1 Bohr or 1 Å.
    pub show_cell: bool,
    pub show_floor: bool,
    pub show_axes: bool,
    pub show_place_grid: bool,
    pub grid_in_angstrom: bool,
}

impl UiState {
    pub fn new(preset: &str, scene: &Scene) -> UiState {
        UiState {
            backend: Backend::Cpu,
            gpu_available: false,
            element: 1,
            mode: ToolMode::Orbit,
            preset: preset.to_string(),
            grid_n: scene.grid_n,
            grid_h: scene.grid_h,
            volume_mode: 0,
            gamma: 0.5,
            opacity: 2.0,
            iso: 0.1,
            scene_path: "my.scene".into(),
            status: String::new(),
            show_cell: true,
            show_floor: true,
            show_axes: true,
            show_place_grid: true,
            grid_in_angstrom: false,
        }
    }

    /// Resets the pending edits after the whole scene was replaced (preset or file load).
    fn adopt(&mut self, scene: &Scene) {
        self.grid_n = scene.grid_n;
        self.grid_h = scene.grid_h;
    }
}

/// What the user's edits in one frame imply; `main` acts on it.
#[derive(Default)]
pub struct PanelOutput {
    pub commands: Vec<Command>,
    pub focus_selected: bool,
    /// A UI-side failure (invalid grid, unreadable file) to show in red.
    pub error: Option<String>,
}

/// Draws the side panel into the root `ui` (egui 0.36 panels attach to a `Ui`, not the context).
pub fn side_panel(
    root: &mut egui::Ui,
    st: &mut UiState,
    scene: &Scene,
    snapshot: Option<&Snapshot>,
    busy: bool,
    last_error: Option<&str>,
    selected: Option<usize>,
) -> PanelOutput {
    let mut out = PanelOutput::default();
    egui::Panel::left("side_panel")
        .resizable(true)
        .default_size(310.0)
        .show(root, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Atomic Sandbox");
                status_line(ui, scene, snapshot, busy, last_error);
                ui.separator();
                backend_row(ui, st);
                scene_section(ui, st, &mut out);
                tool_section(ui, st, scene, selected, &mut out);
                electrons_section(ui, st, scene, &mut out);
                view_section(ui, st);
                display_section(ui, st);
                readouts_section(ui, snapshot);
                file_section(ui, st, scene, &mut out);
            });
        });
    out
}

fn status_line(
    ui: &mut egui::Ui,
    scene: &Scene,
    snapshot: Option<&Snapshot>,
    busy: bool,
    last_error: Option<&str>,
) {
    ui.horizontal(|ui| {
        if busy {
            ui.spinner();
            ui.label("solving…");
        } else if snapshot.is_some_and(|s| s.scene == *scene) {
            ui.label("converged");
        } else if snapshot.is_some() {
            ui.label("density is from an earlier geometry");
        } else {
            ui.label("waiting for the first solve");
        }
    });
    if let Some(e) = last_error {
        ui.colored_label(Color32::from_rgb(255, 90, 90), e);
    }
}

fn backend_row(ui: &mut egui::Ui, st: &mut UiState) {
    ui.horizontal(|ui| {
        ui.label("solver");
        ui.radio_value(&mut st.backend, Backend::Cpu, "CPU")
            .on_hover_text("qm::scf, f64 reference; a solve cannot be interrupted");
        ui.add_enabled_ui(st.gpu_available, |ui| {
            ui.radio_value(&mut st.backend, Backend::Gpu, "GPU")
                .on_hover_text("wgpu compute, f32 grid; a newer edit abandons the solve in flight")
                .on_disabled_hover_text("no GPU adapter");
        });
    });
}

fn scene_section(ui: &mut egui::Ui, st: &mut UiState, out: &mut PanelOutput) {
    ui.horizontal(|ui| {
        let before = st.preset.clone();
        egui::ComboBox::from_label("preset")
            .selected_text(&st.preset)
            .show_ui(ui, |ui| {
                for name in Scene::PRESETS {
                    ui.selectable_value(&mut st.preset, name.to_string(), *name);
                }
            });
        let reload = ui
            .button("Reset")
            .on_hover_text("reload the preset")
            .clicked();
        if st.preset != before || reload {
            if let Some(s) = Scene::preset(&st.preset) {
                st.adopt(&s);
                out.commands.push(Command::SetScene(s));
            }
        }
    });
}

fn tool_section(
    ui: &mut egui::Ui,
    st: &mut UiState,
    scene: &Scene,
    selected: Option<usize>,
    out: &mut PanelOutput,
) {
    ui.separator();
    ui.horizontal(|ui| {
        ui.label("tool");
        ui.radio_value(&mut st.mode, ToolMode::Orbit, "Orbit");
        ui.radio_value(&mut st.mode, ToolMode::Place, "Place");
        ui.radio_value(&mut st.mode, ToolMode::Select, "Select");
    });
    let name = |z: u8| format!("{} (Z = {z})", elements::symbol(z));
    egui::ComboBox::from_label("element to place")
        .selected_text(name(st.element))
        .show_ui(ui, |ui| {
            for z in 1..=10u8 {
                ui.selectable_value(&mut st.element, z, name(z));
            }
        });
    let Some(n) = selected.and_then(|i| scene.nuclei.get(i).map(|n| (i, n))) else {
        ui.weak("no selection (Select tool, click a nucleus)");
        return;
    };
    let (i, nuc) = n;
    ui.label(format!(
        "selected: {} #{i} at ({:.2}, {:.2}, {:.2}) Bohr",
        elements::symbol(nuc.z),
        nuc.pos[0],
        nuc.pos[1],
        nuc.pos[2]
    ));
    ui.horizontal(|ui| {
        if ui.button("Delete selected").clicked() {
            out.commands.push(Command::RemoveNucleus(i));
        }
        if ui.button("Focus selected").clicked() {
            out.focus_selected = true;
        }
    });
}

fn electrons_section(ui: &mut egui::Ui, st: &mut UiState, scene: &Scene, out: &mut PanelOutput) {
    ui.separator();
    let protons: i32 = scene.nuclei.iter().map(|n| n.z as i32).sum();
    let electrons = (protons - scene.charge).max(0);
    egui::Grid::new("electrons").num_columns(2).show(ui, |ui| {
        ui.label("charge");
        let mut c = scene.charge;
        if ui
            // at least one electron must remain; anions down to −4
            .add(
                egui::DragValue::new(&mut c)
                    .range(-4..=(protons - 1).max(0))
                    .speed(0.05),
            )
            .changed()
            && c != scene.charge
        {
            out.commands.push(Command::SetCharge(c));
        }
        ui.end_row();
        ui.label("multiplicity");
        ui.horizontal(|ui| {
            let mut auto = scene.multiplicity.is_none();
            let mut m = scene.multiplicity.unwrap_or(1 + (electrons as u32 % 2));
            let toggled = ui.checkbox(&mut auto, "auto").changed();
            let edited = ui
                .add_enabled(
                    !auto,
                    egui::DragValue::new(&mut m).range(1..=11).speed(0.05),
                )
                .changed();
            let want = if auto { None } else { Some(m) };
            if (toggled || edited) && want != scene.multiplicity {
                out.commands.push(Command::SetMultiplicity(want));
            }
        });
        ui.end_row();
        ui.label("σ (nuclear smearing)");
        ui.horizontal(|ui| {
            let mut auto = scene.sigma.is_none();
            let mut s = scene.sigma.unwrap_or(scene.grid_h);
            let toggled = ui.checkbox(&mut auto, "auto = h").changed();
            let edited = ui
                .add_enabled(
                    !auto,
                    egui::DragValue::new(&mut s)
                        .range(0.05..=1.0)
                        .speed(0.005)
                        .suffix(" Bohr"),
                )
                .changed();
            let want = if auto { None } else { Some(s) };
            if (toggled || edited) && want != scene.sigma {
                out.commands.push(Command::SetSigma(want));
            }
        });
        ui.end_row();
    });
    grid_controls(ui, st, scene, electrons, out);
}

fn grid_controls(
    ui: &mut egui::Ui,
    st: &mut UiState,
    scene: &Scene,
    electrons: i32,
    out: &mut PanelOutput,
) {
    ui.horizontal(|ui| {
        ui.label("grid n");
        // n must be odd: edit k with n = 2k + 1, shown and typed as n
        let mut k = (st.grid_n.max(1) - 1) / 2;
        ui.add(
            egui::DragValue::new(&mut k)
                .range(8..=48)
                .speed(0.1)
                .custom_formatter(|v, _| format!("{}", 2 * v as usize + 1))
                .custom_parser(|s| {
                    s.trim()
                        .parse::<f64>()
                        .ok()
                        .map(|n| ((n - 1.0) / 2.0).round())
                }),
        );
        st.grid_n = 2 * k + 1;
        ui.label("h");
        ui.add(
            egui::DragValue::new(&mut st.grid_h)
                .range(0.1..=0.5)
                .speed(0.002)
                .fixed_decimals(3)
                .suffix(" Bohr"),
        );
    });
    let points = (st.grid_n as f64).powi(3);
    ui.weak(format!(
        "cost {}³ × {electrons} electrons ≈ {:.2} M points (box {:.1} Bohr; now {}³ at {})",
        st.grid_n,
        points * electrons.max(1) as f64 / 1e6,
        (st.grid_n - 1) as f64 * st.grid_h,
        scene.grid_n,
        scene.grid_h
    ));
    let pending = st.grid_n != scene.grid_n || st.grid_h != scene.grid_h;
    if ui
        .add_enabled(pending, egui::Button::new("Apply grid"))
        .clicked()
    {
        match validate_grid(st.grid_n, st.grid_h) {
            Ok(()) => out.commands.push(Command::SetGrid {
                n: st.grid_n,
                h: st.grid_h,
            }),
            Err(e) => out.error = Some(e),
        }
    }
}

fn view_section(ui: &mut egui::Ui, st: &mut UiState) {
    ui.separator();
    ui.horizontal(|ui| {
        ui.label("density");
        ui.radio_value(&mut st.volume_mode, 0, "volume");
        ui.radio_value(&mut st.volume_mode, 1, "isosurface");
    });
    let volume = st.volume_mode == 0;
    ui.add_enabled(
        volume,
        egui::Slider::new(&mut st.gamma, 0.2..=1.0).text("gamma"),
    );
    ui.add_enabled(
        volume,
        egui::Slider::new(&mut st.opacity, 0.1..=20.0)
            .logarithmic(true)
            .text("opacity"),
    );
    ui.add_enabled(
        !volume,
        egui::Slider::new(&mut st.iso, 0.01..=0.9)
            .logarithmic(true)
            .text("iso (× ρ max)"),
    );
}

fn display_section(ui: &mut egui::Ui, st: &mut UiState) {
    ui.separator();
    ui.horizontal(|ui| {
        ui.label("display");
        ui.checkbox(&mut st.show_cell, "cell")
            .on_hover_text("the simulation box: nuclei must stay inside it");
        ui.checkbox(&mut st.show_floor, "floor grid");
        ui.checkbox(&mut st.show_axes, "axes");
    });
    ui.horizontal(|ui| {
        ui.checkbox(&mut st.show_place_grid, "placement plane")
            .on_hover_text("Place tool: the plane the next click lands on");
        ui.label("spacing");
        ui.radio_value(&mut st.grid_in_angstrom, false, "1 Bohr");
        ui.radio_value(&mut st.grid_in_angstrom, true, "1 Å");
    });
}

fn readouts_section(ui: &mut egui::Ui, snapshot: Option<&Snapshot>) {
    ui.separator();
    let Some(s) = snapshot else {
        ui.weak("no solution yet");
        return;
    };
    let e = &s.energy;
    let ev = units::hartree_to_ev;
    egui::Grid::new("energy")
        .num_columns(3)
        .striped(true)
        .show(ui, |ui| {
            let mut row = |name: &str, x: f64, strong: bool| {
                let t = |s: String| {
                    if strong {
                        RichText::new(s).strong()
                    } else {
                        RichText::new(s)
                    }
                };
                ui.label(t(name.to_string()));
                ui.monospace(t(format!("{x:+.6} Ha")));
                ui.monospace(t(format!("{:+.3} eV", ev(x))));
                ui.end_row();
            };
            row("E total", e.total, true);
            row("kinetic", e.kinetic, false);
            row("external", e.external, false);
            row("Hartree", e.hartree, false);
            row("xc", e.xc, false);
            row("nuclear", e.nuclear, false);
        });
    let sigma = s.scene.sigma.unwrap_or(s.grid_h);
    ui.weak(format!(
        "smeared nuclei (σ = {sigma} Bohr): absolute energies depend on σ and h"
    ));
    if s.backend == Backend::Gpu {
        ui.weak("GPU (f32 grid): energies scatter by up to ~1e-5 Ha between solve paths");
    }
    egui::Grid::new("scf").num_columns(2).show(ui, |ui| {
        let mut row = |name: &str, v: String| {
            ui.label(name);
            ui.monospace(v);
            ui.end_row();
        };
        row("SCF steps", s.iterations.to_string());
        // `ScfStats::max_eigen_residual`: the worst LOBPCG residual among the solves the
        // converged state rests on (the GPU's deliberately loosened early SCF steps excluded)
        row(
            "eigen residual (final)",
            format!("{:.1e}", s.max_eigen_residual),
        );
        row("Poisson residual", format!("{:.1e}", s.poisson_residual));
        row(
            "solve",
            match s.readbacks {
                Some(r) => format!("{} ms, {r} readbacks ({})", s.solve_ms, s.backend.label()),
                None => format!("{} ms ({})", s.solve_ms, s.backend.label()),
            },
        );
        row("grid", format!("{}³ × {} Bohr", s.grid_n, s.grid_h));
        row("ρ max", format!("{:.3} e/Bohr³", s.density_max));
    });
    for w in &s.warnings {
        ui.colored_label(Color32::from_rgb(240, 200, 60), w);
    }
}

fn file_section(ui: &mut egui::Ui, st: &mut UiState, scene: &Scene, out: &mut PanelOutput) {
    ui.separator();
    ui.horizontal(|ui| {
        ui.label("file");
        ui.text_edit_singleline(&mut st.scene_path);
    });
    ui.horizontal(|ui| {
        if ui.button("Save").clicked() {
            match std::fs::write(&st.scene_path, scene.to_text()) {
                Ok(()) => {
                    st.status = format!("saved {} nuclei to {}", scene.nuclei.len(), st.scene_path)
                }
                Err(e) => out.error = Some(format!("save {}: {e}", st.scene_path)),
            }
        }
        if ui.button("Load").clicked() {
            match load(&st.scene_path) {
                Ok(s) => {
                    st.status = format!("loaded {}", st.scene_path);
                    st.adopt(&s);
                    out.commands.push(Command::SetScene(s));
                }
                Err(e) => out.error = Some(e),
            }
        }
    });
    if !st.status.is_empty() {
        ui.weak(&st.status);
    }
}

fn load(path: &str) -> Result<Scene, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("load {path}: {e}"))?;
    let scene = Scene::parse(&text).map_err(|e| format!("{path}: {e}"))?;
    scene.validate().map_err(|e| format!("{path}: {e}"))?;
    Ok(scene)
}
