//! What a click in the chrome means, separate from where the control sits: the toolbar, a menu,
//! a dock tab and the Ctrl K search all emit these, and [`apply`] turns each into exactly the
//! [`Command`] / [`DynCommand`] (and `UiState` change) the old side panel produced for the same
//! control — so a control can move without its meaning changing, and the mapping is testable
//! without a window (`ui::tests`).
use super::{duration_label, load, PanelInput, PanelOutput, ToolMode, UiState, FORGET_CONFIRM};
use crate::scene::{model_path, validate_grid, Scene};
use crate::sim::{Backend, Command};
use sim::dynamics::{F_DRAG_HI, F_DRAG_LO};
use sim::{DynCommand, SamplingPlan};
use std::time::Instant;

#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    /// Play ↔ Pause (the viewer flips its play state).
    TogglePlay,
    /// Pause and advance one time step.
    Step,
    /// v → −v.
    Rewind,
    /// Steps per tick (the slider, the typed field, the watchable preset). Never dt (rule 8).
    SetSpeed(u32),
    /// The time step (fs).
    SetDt(f64),
    Tool(ToolMode),
    /// The element to place; picking one also switches the tool to Place.
    Element(u8),
    /// Drop the selection (Esc).
    Deselect,
    /// Remove nucleus i (Delete).
    Delete(usize),
    /// Centre the camera on the selection.
    Focus,
    Pin(usize, bool),
    /// The solver; the viewer re-spawns the worker when it differs from the running one.
    Backend(Backend),
    /// Load a preset (also Reset: the current preset again).
    Preset(String),
    /// The thermostat's switch and settings as edited (sent when switched, or retuned while on).
    Thermostat { on: bool, t: f64, tau: f64 },
    Gravity(bool),
    ComCorrection(bool),
    SetCharge(i32),
    SetMultiplicity(Option<u32>),
    SetSigma(Option<f64>),
    /// Send the pending grid edit (`UiState::grid_n`, `grid_h`).
    ApplyGrid,
    /// Scene ▸ Quality: a level's or the Custom slider's h (snapped; the box is kept, σ follows h
    /// unless set — `crate::quality`).
    Quality(f64),
    /// Write the scene to `UiState::scene_path`.
    SaveScene,
    /// Read the scene at `UiState::scene_path` (and its `.model` sidecar in estimated mode).
    LoadScene,
    SaveModel,
    LoadModel,
    /// "Forget model": the first click arms, a second within 4 s forgets.
    Forget,
    /// "Start sampling" / "Sample more" with the panel's plan; `long` = the pass would take longer
    /// than `sim::PASS_TARGET_S`, which needs a second click ("Start anyway").
    Sample { long: bool },
    CancelSampling,
    /// The automatic first pass's "Start anyway" with the plan it offered.
    StartOffered(SamplingPlan),
    /// "Forget it and sample this scene" (after a refused autoload).
    SampleAfterRefusal,
    /// The estimated solver's audit cadence (answered steps; 0 = never).
    SetAudit(usize),
    /// Density: 0 = volume, 1 = isosurface.
    VolumeMode(u32),
    /// Container walls on / off and their half-width (Bohr); a move while playing is the piston.
    Walls { on: bool, half_width: f64 },
    /// The estimated solver's cloud: atomic superposition between DFT clouds, and the display
    /// refresh (on, every s seconds).
    Cloud { superposition: bool, refresh: bool, refresh_s: f64 },
    /// Draft mode: the model answers every step (unsafe).
    Draft(bool),
    /// The drag pull (Ha/Bohr): Physics ▸ Drag pull's slider and Reset; clamped to the driver's
    /// range, sent only when it changes.
    DragForce(f64),
    /// The wheel while dragging (the 3D view, `Viewer::wheel`): the pull × [`super::DRAG_WHEEL_STEP`]
    /// per line up, ÷ per line down.
    DragForceWheel(f32),
}

/// Applies one action: the `UiState` change and the commands, as the old panel did.
pub fn apply(st: &mut UiState, input: &PanelInput<'_>, a: Action, out: &mut PanelOutput) {
    let dyn_ = |c: DynCommand| Command::Dyn(c);
    match a {
        Action::TogglePlay => out.toggle_play = true,
        Action::Step => out.commands.push(dyn_(DynCommand::Step(1))),
        Action::Rewind => out.commands.push(dyn_(DynCommand::Rewind)),
        Action::SetSpeed(n) => {
            st.speed = n;
            out.commands.push(dyn_(DynCommand::SetSpeed(n)));
        }
        Action::SetDt(dt) => {
            st.dt_fs = dt;
            out.commands.push(dyn_(DynCommand::SetDt(dt)));
        }
        Action::Tool(m) => st.mode = m,
        Action::Element(z) => {
            st.element = z;
            st.mode = ToolMode::Place;
        }
        Action::Deselect => out.deselect = true,
        Action::Delete(i) => out.commands.push(Command::RemoveNucleus(i)),
        Action::Focus => out.focus_selected = true,
        Action::Pin(i, on) => out.commands.push(dyn_(DynCommand::Pin(i, on))),
        Action::Backend(b) => {
            if b != Backend::Gpu || st.gpu_available {
                st.backend = b;
            }
        }
        Action::Preset(name) => {
            // at the panel's quality (Accurate by default), the preset's own box kept
            if let (Some(file), Some(s)) = (Scene::preset(&name), crate::quality::preset(&name, st.quality_h)) {
                if let Err(e) = vram_check(st, &s) {
                    out.error = Some(e);
                    return;
                }
                st.preset = name;
                st.adopt(&s);
                st.box_len = crate::quality::box_len(&file);
                // the container follows the new cell (h2 → h2o: 4.5 → 5.55 Bohr)
                st.wall_half_width = super::default_wall_half_width(&s);
                out.commands.push(Command::SetScene(s));
                if st.walls_on {
                    out.commands.push(dyn_(DynCommand::SetWalls(st.walls())));
                }
            }
        }
        Action::Quality(h) => {
            let h = crate::quality::snap(h);
            st.quality_h = h;
            let s = crate::quality::regrid(input.scene, st.box_len, h);
            if let Err(e) = vram_check(st, &s) {
                out.error = Some(e);
                return;
            }
            st.grid_n = s.grid_n;
            st.grid_h = s.grid_h;
            if (s.grid_n, s.grid_h) != (input.scene.grid_n, input.scene.grid_h) {
                out.commands.push(Command::SetGrid { n: s.grid_n, h: s.grid_h });
            }
        }
        Action::Thermostat { on, t, tau } => {
            let toggled = on != st.thermostat_on;
            let retuned = t != st.thermostat_t || tau != st.thermostat_tau;
            st.thermostat_on = on;
            st.thermostat_t = t;
            st.thermostat_tau = tau;
            if toggled || (on && retuned) {
                out.commands.push(dyn_(DynCommand::SetThermostat(st.thermostat())));
            }
        }
        Action::Gravity(on) => {
            st.gravity = on;
            out.commands.push(dyn_(DynCommand::SetExternalField(st.field())));
        }
        Action::ComCorrection(on) => {
            st.com_correction = on;
            out.commands.push(dyn_(DynCommand::SetComCorrection(on)));
        }
        Action::SetCharge(c) => {
            if c != input.scene.charge {
                out.commands.push(Command::SetCharge(c));
            }
        }
        Action::SetMultiplicity(m) => {
            if m != input.scene.multiplicity {
                out.commands.push(Command::SetMultiplicity(m));
            }
        }
        Action::SetSigma(s) => {
            if s != input.scene.sigma {
                out.commands.push(Command::SetSigma(s));
            }
        }
        Action::ApplyGrid => match validate_grid(st.grid_n, st.grid_h).and_then(|()| {
            let s = Scene { grid_n: st.grid_n, grid_h: st.grid_h, ..input.scene.clone() };
            vram_check(st, &s)
        }) {
            Ok(()) => {
                // an explicit grid: its box is the one later quality changes keep
                st.box_len = (st.grid_n as f64 - 1.0) * st.grid_h;
                out.commands.push(Command::SetGrid { n: st.grid_n, h: st.grid_h });
            }
            Err(e) => out.error = Some(e),
        },
        Action::SaveScene => match std::fs::write(&st.scene_path, input.scene.to_text()) {
            Ok(()) => {
                let msg = format!("saved {} nuclei to {}", input.scene.nuclei.len(), st.scene_path);
                st.set_status(msg)
            }
            Err(e) => out.error = Some(format!("save {}: {e}", st.scene_path)),
        },
        Action::LoadScene => match load(&st.scene_path) {
            Ok(s) => {
                let mut msg = format!("loaded {}", st.scene_path);
                st.adopt(&s);
                st.wall_half_width = super::default_wall_half_width(&s);
                out.commands.push(Command::SetScene(s));
                if st.walls_on {
                    out.commands.push(dyn_(DynCommand::SetWalls(st.walls())));
                }
                let sidecar = model_path(&st.scene_path);
                if st.backend == Backend::Estimated && sidecar.exists() {
                    if input.sampling() {
                        // the new scene cancels the pass, but a load would be refused while it
                        // runs (and abort its solve for nothing)
                        msg += &format!(" — its model {} was not loaded (a sampling pass was running): press Load model", sidecar.display());
                    } else {
                        msg += &format!(" and its model {}", sidecar.display());
                        out.commands.push(dyn_(DynCommand::LoadModel(sidecar)));
                    }
                }
                st.set_status(msg);
            }
            Err(e) => out.error = Some(e),
        },
        Action::SaveModel => {
            let path = model_path(&st.scene_path);
            out.commands.push(dyn_(DynCommand::SaveModel(path.clone())));
            st.set_status(format!("saving the model to {}", path.display()));
        }
        Action::LoadModel => {
            let path = model_path(&st.scene_path);
            if path.exists() {
                out.commands.push(dyn_(DynCommand::LoadModel(path)));
            } else {
                out.error = Some(format!("no model file {}", path.display()));
            }
        }
        Action::Forget => {
            if forget_armed(st) {
                st.forget_armed = None;
                out.commands.push(dyn_(DynCommand::ForgetModel));
            } else {
                st.forget_armed = Some(Instant::now());
            }
        }
        Action::Sample { long } => {
            if long && !start_armed(st) {
                st.start_armed = Some(Instant::now());
            } else {
                st.start_armed = None;
                out.commands.push(dyn_(DynCommand::StartSampling(st.plan.clone())));
                st.plan.seed += 1;
            }
        }
        Action::CancelSampling => out.commands.push(dyn_(DynCommand::CancelSampling)),
        Action::StartOffered(plan) => out.commands.push(dyn_(DynCommand::StartSampling(plan))),
        Action::SampleAfterRefusal => {
            out.commands.push(dyn_(DynCommand::StartSampling(st.plan.clone())));
            st.plan.seed += 1;
        }
        Action::SetAudit(n) => {
            st.audit_every = n;
            out.commands.push(Command::SetAudit(n));
        }
        Action::VolumeMode(m) => st.volume_mode = m,
        Action::Walls { on, half_width } => {
            st.walls_on = on;
            st.wall_half_width = half_width;
            out.commands.push(dyn_(DynCommand::SetWalls(st.walls())));
        }
        Action::Cloud { superposition, refresh, refresh_s } => {
            st.cloud_superposition = superposition;
            st.cloud_refresh = refresh;
            st.cloud_refresh_s = refresh_s;
            out.commands.push(st.cloud_command());
        }
        Action::Draft(on) => {
            st.draft = on;
            out.commands.push(Command::SetDraft(on));
        }
        Action::DragForce(f) => set_drag_force(st, f.clamp(F_DRAG_LO, F_DRAG_HI), out),
        Action::DragForceWheel(lines) => set_drag_force(st, super::wheel_pull(st.drag_force, lines), out),
    }
}

/// The pull as set (already in range); a no-op when unchanged (the wheel at an end of the range).
fn set_drag_force(st: &mut UiState, f: f64, out: &mut PanelOutput) {
    if f.is_finite() && f != st.drag_force {
        st.drag_force = f;
        out.commands.push(Command::Dyn(DynCommand::SetDragForce { f_max: f }));
    }
}

/// The first "Forget model" click is still armed.
pub fn forget_armed(st: &UiState) -> bool {
    st.forget_armed.is_some_and(|t| t.elapsed() < FORGET_CONFIRM)
}

/// The first click on a long pass is still armed.
pub fn start_armed(st: &UiState) -> bool {
    st.start_armed.is_some_and(|t| t.elapsed() < FORGET_CONFIRM)
}

/// Drops an expired arming (the buttons go back to their first-click label).
pub fn expire_arming(st: &mut UiState) {
    if st.forget_armed.is_some() && !forget_armed(st) {
        st.forget_armed = None;
    }
    if st.start_armed.is_some() && !start_armed(st) {
        st.start_armed = None;
    }
}

/// The sampling button's state (Dock ▸ Model).
pub struct SampleButton {
    pub label: String,
    /// The pass would take longer than `sim::PASS_TARGET_S` (the panel's plan × the teacher's cost
    /// per solve): it needs a second click.
    pub long: bool,
    /// The first click on a long pass is armed: the label is "Start anyway (≈ t)".
    pub armed: bool,
    /// The plan's cost line ("n solves, ≈ t"), when not sampling.
    pub cost: Option<String>,
}

/// The sampling button's label, length, arming and cost.
pub fn sample_button(st: &UiState, input: &PanelInput<'_>) -> SampleButton {
    let m = input.model();
    let d = input.snapshot.map(|s| &s.dynamics);
    let fitted = m.is_some_and(|m| m.fitted);
    let solves = d.and_then(|d| st.plan.solves(&d.scene).ok());
    let eta = solves.zip(m.and_then(|m| m.solve_s)).map(|(n, c)| n as f64 * c);
    let long = eta.is_some_and(|t| t > crate::sim::PASS_TARGET_S);
    let armed = long && start_armed(st);
    let label = if armed {
        format!("Start anyway (≈ {})", duration_label(eta.unwrap_or(0.0)))
    } else if fitted || m.is_some_and(|m| m.samples > 0) {
        "Sample more".into()
    } else {
        "Start sampling".into()
    };
    let cost = solves.filter(|_| !input.sampling()).map(|n| match eta {
        Some(t) => format!("{n} solves, ≈ {}", duration_label(t)),
        None => format!("{n} solves"),
    });
    SampleButton { label, long, armed, cost }
}

/// The GPU memory guard for a scene the panel is about to send (`quality::vram_check` with the
/// backend's solver count and the budget): refused before the worker allocates anything.
fn vram_check(st: &UiState, s: &Scene) -> Result<(), String> {
    crate::quality::vram_check(s, crate::quality::gpu_solvers(st.backend, st.gpu_available), crate::quality::vram_budget_bytes())
}
