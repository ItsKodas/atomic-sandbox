//! What a click in the chrome means, separate from where the control sits: the toolbar, a menu,
//! a dock tab and the Ctrl K search all emit these, and [`apply`] turns each into exactly the
//! [`Command`] / [`DynCommand`] (and `UiState` change) the old side panel produced for the same
//! control — so a control can move without its meaning changing, and the mapping is testable
//! without a window (`ui::tests`). The same actions are the panel half of the input-replay
//! harness (`crate::replay`, M6 spec §8): `Viewer::apply` runs one through [`apply`], the recorder
//! writes each as a script line ([`Action::describe`]) and the replayer reads it back
//! ([`Action::parse`]) — the egui layout itself is never replayed.
use super::{duration_label, load, PanelInput, PanelOutput, ToolMode, UiState, FORGET_CONFIRM};
use crate::scene::{model_path, validate_grid, Scene};
use crate::sim::{Backend, Command};
use providers::PlanSolves;
use sim::dynamics::{F_DRAG_HI, F_DRAG_LO};
use sim::{DynCommand, SamplingPlan};
use std::time::Instant;

#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    /// Play ↔ Pause (the viewer flips its play state).
    TogglePlay,
    /// Play / Pause outright (a replay script names the state it wants; the button toggles).
    Play,
    Pause,
    /// Pause and advance one time step.
    Step,
    /// v → −v.
    Rewind,
    /// Steps per tick (the slider, the typed field, the watchable preset). Never dt (rule 8).
    SetSpeed(u32),
    /// The time step (fs).
    SetDt(f64),
    Tool(ToolMode),
    /// The element to place (its most abundant isotope); picking one also switches the tool to
    /// Place.
    Element(u8),
    /// The element and isotope (mass number) to place — the periodic table's isotope row; also
    /// switches the tool to Place.
    ElementIsotope(u8, u16),
    /// The selected nucleus's isotope (Dock ▸ Selection): `DynCommand::SetIsotope` (mass only:
    /// the Born-Oppenheimer surface is unchanged, ΔK booked — M6 spec §5.3).
    SetIsotope { index: usize, a: u16 },
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
    /// Temperatures on screen: K, °C or °F (display only).
    TempUnits(super::TempUnits),
    /// Scene ▸ Quality: a level's or the Custom slider's h (snapped; the box is kept, σ follows h
    /// unless set — `crate::quality`).
    Quality(f64),
    /// Write the scene to `UiState::scene_path`.
    SaveScene,
    /// Read the scene at `UiState::scene_path` (and, with `sidecar`, its `.model` sidecar in
    /// estimated mode). A replay's `scene` line loads without the sidecar (final review app M8: a
    /// replay touches only its script and the files it names).
    LoadScene { sidecar: bool },
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
    /// View ▸ Bonds and molecules: the bond lines, the molecule labels in the view, the
    /// composition badge on hover (display only).
    ShowBonds(bool),
    ShowLabels(bool),
    ShowBadge(bool),
    /// View ▸ Bonds and molecules ▸ name style (display only).
    NameStyle(crate::names::NameStyle),
    /// Relax the geometry (FIRE, M6 spec §6): `DynCommand::Relax` — refused while a sampling pass
    /// runs (the driver refuses it too).
    Relax,
    /// End the relax now: `DynCommand::StopRelax`.
    StopRelax,
    /// View ▸ Density ▸ smooth motion: draw the nuclei between the last two snapshots (display
    /// only; every replay script turns it off first so what it checks is the snapshot's).
    SmoothMotion(bool),
}

/// Applies one action: the `UiState` change and the commands, as the old panel did.
pub fn apply(st: &mut UiState, input: &PanelInput<'_>, a: Action, out: &mut PanelOutput) {
    let dyn_ = |c: DynCommand| Command::Dyn(c);
    match a {
        Action::TogglePlay => out.toggle_play = true,
        Action::Play => out.commands.push(dyn_(DynCommand::Play)),
        Action::Pause => out.commands.push(dyn_(DynCommand::Pause)),
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
        // only an element that can be placed (final review app M3: the table and chips emit Z
        // 1–10, but a script could pick any Z, which the worker then refused at the click)
        Action::Element(z) | Action::ElementIsotope(z, _) if !super::ptable::enabled(z) => {
            out.error = Some(not_placeable(z));
        }
        Action::Element(z) => {
            st.element = z;
            st.isotope = None;
            st.mode = ToolMode::Place;
        }
        Action::ElementIsotope(z, a) => {
            if units::isotope_mass_amu_a(z, a).is_some() {
                st.element = z;
                st.isotope = crate::scene::canonical_a(z, a);
                st.mode = ToolMode::Place;
            }
        }
        Action::SetIsotope { index, a } => out.commands.push(dyn_(DynCommand::SetIsotope { index, a })),
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
                st.source = super::SceneSource::Preset(name);
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
        Action::TempUnits(u) => st.temp_units = u,
        Action::Quality(h) => {
            let h = crate::quality::snap(h);
            let s = crate::quality::regrid(input.scene, st.box_len, h);
            if let Err(e) = vram_check(st, &s) {
                // refused: the slider goes back to the accepted quality
                st.quality_slider = st.quality_h;
                out.error = Some(e);
                return;
            }
            st.quality_h = h;
            st.quality_slider = h;
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
        Action::LoadScene { sidecar } => match load(&st.scene_path) {
            Ok(s) => {
                let mut msg = format!("loaded {}", st.scene_path);
                let path = st.scene_path.clone();
                st.adopt_file(&path, &s);
                out.commands.push(Command::SetScene(s));
                if st.walls_on {
                    out.commands.push(dyn_(DynCommand::SetWalls(st.walls())));
                }
                let model = model_path(&st.scene_path);
                if sidecar && st.backend == Backend::Estimated && model.exists() {
                    let sidecar = model;
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
        Action::ShowBonds(on) => st.show_bonds = on,
        Action::ShowLabels(on) => st.show_labels = on,
        Action::ShowBadge(on) => st.show_badge = on,
        Action::NameStyle(s) => st.name_style = s,
        Action::Relax => {
            if input.sampling() {
                st.set_status(super::RELAX_REFUSED_SAMPLING);
            } else {
                out.commands.push(dyn_(DynCommand::Relax));
            }
        }
        Action::StopRelax => out.commands.push(dyn_(DynCommand::StopRelax)),
        Action::SmoothMotion(on) => st.smooth_motion = on,
    }
}

impl Action {
    /// The action as a replay script's words (`action <words>`, `crate::replay`), the inverse of
    /// [`Action::parse`]. `None` for the actions whose meaning lives in panel state a script does
    /// not carry — the pending grid edit (Apply grid), the file path (save / load scene or model),
    /// the sampling plan (Sample, the offered pass): the recorder writes those as a comment.
    pub fn describe(&self) -> Option<String> {
        use Action::*;
        let on = |b: bool| if b { "on" } else { "off" };
        let opt = |v: Option<String>| v.unwrap_or_else(|| "auto".into());
        Some(match self {
            TogglePlay => "toggle-play".into(),
            Play => "play".into(),
            Pause => "pause".into(),
            Step => "step".into(),
            Rewind => "rewind".into(),
            SetSpeed(n) => format!("speed {n}"),
            SetDt(dt) => format!("dt {dt}"),
            Tool(m) => format!("tool {}", tool_word(*m)),
            Element(z) => format!("element {z}"),
            ElementIsotope(z, a) => format!("element {z} {a}"),
            SetIsotope { index, a } => format!("isotope {index} {a}"),
            Deselect => "deselect".into(),
            Delete(i) => format!("delete {i}"),
            Focus => "focus".into(),
            Pin(i, p) => format!("pin {i} {p}"),
            Backend(b) => format!("backend {}", backend_word(*b)),
            Preset(name) => format!("preset {name}"),
            Thermostat { on: o, t, tau } => format!("thermostat {} {t} {tau}", on(*o)),
            Gravity(b) => format!("gravity {b}"),
            ComCorrection(b) => format!("com-correction {b}"),
            SetCharge(c) => format!("charge {c}"),
            SetMultiplicity(m) => format!("multiplicity {}", opt(m.map(|m| m.to_string()))),
            SetSigma(s) => format!("sigma {}", opt(s.map(|s| s.to_string()))),
            Forget => "forget".into(),
            CancelSampling => "cancel-sampling".into(),
            SetAudit(n) => format!("audit {n}"),
            VolumeMode(m) => format!("volume-mode {m}"),
            Walls { on: o, half_width } => format!("walls {} {half_width}", on(*o)),
            Cloud { superposition, refresh, refresh_s } => format!("cloud {superposition} {refresh} {refresh_s}"),
            Draft(b) => format!("draft {b}"),
            DragForce(f) => format!("drag-force {f}"),
            DragForceWheel(l) => format!("drag-force-wheel {l}"),
            ShowBonds(b) => format!("bonds {b}"),
            ShowLabels(b) => format!("labels {b}"),
            ShowBadge(b) => format!("badge {b}"),
            NameStyle(s) => format!("names {}", style_word(*s)),
            Relax => "relax".into(),
            StopRelax => "stop-relax".into(),
            SmoothMotion(b) => format!("smooth-motion {b}"),
            Quality(h) => format!("quality {h}"),
            TempUnits(u) => format!(
                "temp-units {}",
                match u {
                    super::TempUnits::K => "k",
                    super::TempUnits::C => "c",
                    super::TempUnits::F => "f",
                }
            ),
            ApplyGrid | SaveScene | LoadScene { .. } | SaveModel | LoadModel | Sample { .. } | StartOffered(_) | SampleAfterRefusal => return None,
        })
    }

    /// An `action` line's words (without `action`), the inverse of [`Action::describe`].
    pub fn parse(words: &[&str]) -> Result<Action, String> {
        use Action::*;
        let (&head, args) = words.split_first().ok_or("an action needs a name")?;
        let want = |n: usize| -> Result<(), String> {
            if args.len() == n {
                Ok(())
            } else {
                Err(format!("action {head} takes {n} argument{}, got {}", if n == 1 { "" } else { "s" }, args.len()))
            }
        };
        fn num<T: std::str::FromStr>(head: &str, w: &str) -> Result<T, String> {
            w.parse().map_err(|_| format!("action {head}: bad number '{w}'"))
        }
        let flag = |w: &str| match w {
            "true" | "on" => Ok(true),
            "false" | "off" => Ok(false),
            _ => Err(format!("action {head}: expected true/false (or on/off), got '{w}'")),
        };
        Ok(match head {
            "toggle-play" | "play" | "pause" | "step" | "rewind" | "deselect" | "focus" | "forget" | "cancel-sampling" | "relax" | "stop-relax" => {
                want(0)?;
                match head {
                    "toggle-play" => TogglePlay,
                    "play" => Play,
                    "pause" => Pause,
                    "step" => Step,
                    "rewind" => Rewind,
                    "deselect" => Deselect,
                    "focus" => Focus,
                    "forget" => Forget,
                    "cancel-sampling" => CancelSampling,
                    "relax" => Relax,
                    _ => StopRelax,
                }
            }
            "speed" => {
                want(1)?;
                SetSpeed(num(head, args[0])?)
            }
            "dt" => {
                want(1)?;
                SetDt(num(head, args[0])?)
            }
            "tool" => {
                want(1)?;
                Tool(match args[0] {
                    "orbit" => ToolMode::Orbit,
                    "place" => ToolMode::Place,
                    "select" => ToolMode::Select,
                    w => return Err(format!("action tool: orbit, place or select, got '{w}'")),
                })
            }
            "element" => {
                let z: u8 = match args.first() {
                    Some(w) => num(head, w)?,
                    None => return Err("action element takes Z [A], got 0 arguments".into()),
                };
                if !super::ptable::enabled(z) {
                    return Err(format!("action {}", not_placeable(z)));
                }
                match args.len() {
                    1 => Element(z),
                    2 => ElementIsotope(z, num(head, args[1])?),
                    n => return Err(format!("action element takes Z [A], got {n} arguments")),
                }
            }
            "isotope" => {
                want(2)?;
                SetIsotope { index: num(head, args[0])?, a: num(head, args[1])? }
            }
            "delete" => {
                want(1)?;
                Delete(num(head, args[0])?)
            }
            "pin" => {
                want(2)?;
                Pin(num(head, args[0])?, flag(args[1])?)
            }
            "backend" => {
                want(1)?;
                Backend(match args[0] {
                    "gpu" => crate::sim::Backend::Gpu,
                    "cpu" => crate::sim::Backend::Cpu,
                    "estimated" => crate::sim::Backend::Estimated,
                    w => return Err(format!("action backend: gpu, cpu or estimated, got '{w}'")),
                })
            }
            "preset" => {
                want(1)?;
                Preset(args[0].to_string())
            }
            "thermostat" => {
                want(3)?;
                Thermostat { on: flag(args[0])?, t: num(head, args[1])?, tau: num(head, args[2])? }
            }
            "gravity" | "com-correction" | "draft" | "bonds" | "labels" | "badge" | "smooth-motion" => {
                want(1)?;
                let b = flag(args[0])?;
                match head {
                    "gravity" => Gravity(b),
                    "com-correction" => ComCorrection(b),
                    "draft" => Draft(b),
                    "bonds" => ShowBonds(b),
                    "labels" => ShowLabels(b),
                    "badge" => ShowBadge(b),
                    _ => SmoothMotion(b),
                }
            }
            "charge" => {
                want(1)?;
                SetCharge(num(head, args[0])?)
            }
            "multiplicity" => {
                want(1)?;
                SetMultiplicity(if args[0] == "auto" { None } else { Some(num(head, args[0])?) })
            }
            "sigma" => {
                want(1)?;
                SetSigma(if args[0] == "auto" { None } else { Some(num(head, args[0])?) })
            }
            "audit" => {
                want(1)?;
                SetAudit(num(head, args[0])?)
            }
            "volume-mode" => {
                want(1)?;
                VolumeMode(num(head, args[0])?)
            }
            "walls" => {
                want(2)?;
                Walls { on: flag(args[0])?, half_width: num(head, args[1])? }
            }
            "cloud" => {
                want(3)?;
                Cloud { superposition: flag(args[0])?, refresh: flag(args[1])?, refresh_s: num(head, args[2])? }
            }
            "drag-force" => {
                want(1)?;
                DragForce(num(head, args[0])?)
            }
            "drag-force-wheel" => {
                want(1)?;
                DragForceWheel(num(head, args[0])?)
            }
            "names" => {
                want(1)?;
                use crate::names::NameStyle as N;
                NameStyle(match args[0] {
                    "common" => N::Common,
                    "stoichiometric" => N::Stoichiometric,
                    "hydride" => N::Hydride,
                    "formula" => N::FormulaOnly,
                    w => return Err(format!("action names: common, stoichiometric, hydride or formula, got '{w}'")),
                })
            }
            "quality" => {
                // a level's name (`--quality` spells them the same) or h in Bohr
                want(1)?;
                Quality(match crate::quality::parse(args[0]) {
                    Some(h) => h,
                    None => num(head, args[0])?,
                })
            }
            "temp-units" => {
                want(1)?;
                TempUnits(match args[0] {
                    "k" | "K" => super::TempUnits::K,
                    "c" | "C" => super::TempUnits::C,
                    "f" | "F" => super::TempUnits::F,
                    w => return Err(format!("action temp-units: k, c or f, got '{w}'")),
                })
            }
            other => return Err(format!("unknown action '{other}'")),
        })
    }
}

/// The refusal of an element that cannot be placed (`ptable::enabled`).
fn not_placeable(z: u8) -> String {
    format!("element {z}: only Z 1–{} can be placed (heavier elements need pseudopotentials)", super::ptable::Z_ENABLED)
}

fn tool_word(m: ToolMode) -> &'static str {
    match m {
        ToolMode::Orbit => "orbit",
        ToolMode::Place => "place",
        ToolMode::Select => "select",
    }
}

/// The solver's script word (`--backend` spells it the same).
pub fn backend_word(b: Backend) -> &'static str {
    match b {
        Backend::Gpu => "gpu",
        Backend::Cpu => "cpu",
        Backend::Estimated => "estimated",
    }
}

fn style_word(s: crate::names::NameStyle) -> &'static str {
    use crate::names::NameStyle as N;
    match s {
        N::Common => "common",
        N::Stoichiometric => "stoichiometric",
        N::Hydride => "hydride",
        N::FormulaOnly => "formula",
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
    /// The first click on a long pass is armed: the label is "Start anyway (≈ t at the first
    /// solve)" or "… at the last pass's rate" (`ModelStatus::solve_s_measured`).
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
        // the estimate's basis (M6 Task 9): the last pass's measured rate, else the first solve
        let basis = if m.is_some_and(|m| m.solve_s_measured) { "at the last pass's rate" } else { "at the first solve" };
        format!("Start anyway (≈ {} {basis})", duration_label(eta.unwrap_or(0.0)))
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
