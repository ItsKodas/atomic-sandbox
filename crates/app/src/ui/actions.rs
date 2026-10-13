//! What a click in the chrome means, separate from where the control sits: the toolbar, a menu,
//! a dock tab and the Ctrl K search all emit these, and [`apply`] turns each into exactly the
//! [`Command`] / [`DynCommand`] (and `UiState` change) the old side panel produced for the same
//! control — so a control can move without its meaning changing, and the mapping is testable
//! without a window (`ui::tests`).
use super::{duration_label, load, PanelInput, PanelOutput, ToolMode, UiState, FORGET_CONFIRM};
use crate::scene::{model_path, validate_grid, Scene};
use crate::sim::{Backend, Command};
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
            if let Some(s) = Scene::preset(&name) {
                st.preset = name;
                st.adopt(&s);
                out.commands.push(Command::SetScene(s));
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
        Action::ApplyGrid => match validate_grid(st.grid_n, st.grid_h) {
            Ok(()) => out.commands.push(Command::SetGrid { n: st.grid_n, h: st.grid_h }),
            Err(e) => out.error = Some(e),
        },
        Action::SaveScene => match std::fs::write(&st.scene_path, input.scene.to_text()) {
            Ok(()) => st.status = format!("saved {} nuclei to {}", input.scene.nuclei.len(), st.scene_path),
            Err(e) => out.error = Some(format!("save {}: {e}", st.scene_path)),
        },
        Action::LoadScene => match load(&st.scene_path) {
            Ok(s) => {
                st.status = format!("loaded {}", st.scene_path);
                st.adopt(&s);
                out.commands.push(Command::SetScene(s));
                let sidecar = model_path(&st.scene_path);
                if st.backend == Backend::Estimated && sidecar.exists() {
                    if input.sampling() {
                        // the new scene cancels the pass, but a load would be refused while it
                        // runs (and abort its solve for nothing)
                        st.status += &format!(" — its model {} was not loaded (a sampling pass was running): press Load model", sidecar.display());
                    } else {
                        st.status += &format!(" and its model {}", sidecar.display());
                        out.commands.push(dyn_(DynCommand::LoadModel(sidecar)));
                    }
                }
            }
            Err(e) => out.error = Some(e),
        },
        Action::SaveModel => {
            let path = model_path(&st.scene_path);
            out.commands.push(dyn_(DynCommand::SaveModel(path.clone())));
            st.status = format!("saving the model to {}", path.display());
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

/// The sampling button's label and whether the pass is long (the panel's plan × the teacher's
/// cost per solve above `sim::PASS_TARGET_S`), with the plan's cost line ("n solves, ≈ t").
pub fn sample_button(st: &UiState, input: &PanelInput<'_>) -> (String, bool, Option<String>) {
    let m = input.model();
    let d = input.snapshot.map(|s| &s.dynamics);
    let fitted = m.is_some_and(|m| m.fitted);
    let solves = d.and_then(|d| st.plan.solves(&d.scene).ok());
    let eta = solves.zip(m.and_then(|m| m.solve_s)).map(|(n, c)| n as f64 * c);
    let long = eta.is_some_and(|t| t > crate::sim::PASS_TARGET_S);
    let label = if long && start_armed(st) {
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
    (label, long, cost)
}
