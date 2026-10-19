//! What a click in the chrome means, separate from where the control sits: the toolbar, a menu,
//! a dock tab and the Ctrl K search all emit these, and [`apply`] turns each into exactly the
//! [`Command`] / [`DynCommand`] (and `UiState` change) the old side panel produced for the same
//! control — so a control can move without its meaning changing, and the mapping is testable
//! without a window (`ui::tests`). The same actions are the panel half of the input-replay
//! harness (`crate::replay`, M6 spec §8): `Viewer::apply` runs one through [`apply`], the recorder
//! writes each as a script line ([`Action::describe`]) and the replayer reads it back
//! ([`Action::parse`]) — the egui layout itself is never replayed.
use super::{duration_label, load, DockTab, PanelInput, PanelOutput, ShowMode, ToolMode, UiState, FORGET_CONFIRM, UNOCCUPIED_MAX};
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
    /// Physics ▸ Reverse velocities: v → −v (`DynCommand::Rewind`; time reversal).
    ReverseVelocities,
    /// The toolbar's Rewind: back to Play snapshot i (0 = the newest; `ui::rewind`).
    RewindTo(usize),
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
    /// Physics ▸ Ionic potential (M7): all-electron, pseudopotentials for species with a core, or for
    /// every species. Replaces the scene (`Command::SetIonicPolicy`); presets loaded afterwards keep it.
    Ionic(sim::IonicPolicy),
    /// View ▸ Density ▸ with partial core (M7): the density shown includes the pseudised species' ρ_pc —
    /// a display of the generator's core model, not solved (`Command::DisplayPartialCore`).
    ShowPartialCore(bool),
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
    /// Load the model file chosen in the dialog (`UiState::model_file`, else next to the scene). Its provenance is read
    /// first (`surrogate::sidecar::read_provenance`): equal to the scene's → `DynCommand::LoadModel`; different → no
    /// command, the offer card opens (`UiState::model_offer`, `ui::model_offer`).
    LoadModel,
    /// The offer card's "Apply the model's settings and load": the scene edits (`SetGrid`, `SetSigma`, `SetCharge`,
    /// `SetMultiplicity`, `SetIonicPolicy`, `SetIsotope`) that make the scene match the model's provenance, then
    /// `DynCommand::LoadModel`. Refused (an error, nothing sent) when a setting cannot be applied.
    ApplyModelSettings,
    /// The File window's "Use its settings": the model file's settings applied to the scene (as
    /// the card's Apply does), the model itself not loaded.
    ApplyModelSettingsOnly(std::path::PathBuf),
    /// The offer card's "Cancel": closes it; nothing is sent, nothing changes.
    CancelModelOffer,
    /// "Forget model": the first click arms, a second within 4 s forgets.
    Forget,
    /// "Start sampling" / "Sample more" with the panel's plan; `long` = the pass would take longer
    /// than `sim::PASS_TARGET_S`, which needs a second click ("Start anyway").
    Sample { long: bool },
    CancelSampling,
    /// "Teach around N blockers": a teach-around pass (`DynCommand::StartTeaching`; the surrogate's
    /// "Teaching around the blockers"). Recorded like Sample, not replayable (it solves).
    TeachAroundBlockers,
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
    /// View ▸ Bonds and molecules ▸ bond angles (display only; script word `angles`).
    ShowAngles(bool),
    /// View ▸ Bonds and molecules ▸ name style (display only).
    NameStyle(crate::names::NameStyle),
    /// Relax the geometry (FIRE, M6 spec §6): `DynCommand::Relax` — refused while a sampling pass
    /// runs (the driver refuses it too).
    Relax,
    /// The estimated view's "Bonds now" (`DynCommand::RefreshBonds`): a teacher solve between
    /// steps for the bond graph, and the bonds, lengths and angles shown.
    RefreshBonds,
    /// End the relax now: `DynCommand::StopRelax`.
    StopRelax,
    /// View ▸ Density ▸ smooth motion: draw the nuclei between the last two snapshots (display
    /// only; every replay script turns it off first so what it checks is the snapshot's).
    SmoothMotion(bool),
    /// View ▸ Density ▸ show (orbital-viewer spec §3): what the 3D view shows. Anything but
    /// Density opens the Orbitals tab and asks the worker for the orbital view
    /// (`Command::OrbitalView`); Density switches it off.
    Show(ShowMode),
    /// A row of the Orbitals tab: show that orbital (by channel and track — the selection follows
    /// it as the levels reorder) in Orbitals mode.
    PickOrbital { spin: sim::Spin, track: u32 },
    /// Orbitals mode: the signed ± phase (true) or ψ̃² (false).
    OrbitalPhase(bool),
    /// "show unoccupied": m per spin channel (0–8).
    Unoccupied(u8),
    /// All orbitals coloured: include or exclude one orbital (by track); the first change makes
    /// the default (every occupied orbital) explicit.
    Include { track: u32, on: bool },
    /// All orbitals coloured: back to the default, every occupied orbital.
    IncludeAllOccupied,
    /// All orbitals coloured: additive (emission only) or weighted blend — a render setting, no
    /// request to the worker.
    ColourBlend { additive: bool },
    /// All orbitals coloured: each orbital scaled to its own peak (false) or to the physical ρ max.
    ColourScale { physical: bool },
    /// All orbitals coloured: boundary surfaces (enclosed %, shell or filled) or the density, and
    /// whether the core orbitals are left out (`sim::ColourLook`).
    ColourLook(sim::ColourLook),
    /// View: the shown fields follow the atoms (`UiState::warp_fields`; display only).
    WarpFields(bool),
}

/// Applies one action: the `UiState` change and the commands, as the old panel did.
pub fn apply(st: &mut UiState, input: &PanelInput<'_>, a: Action, out: &mut PanelOutput) {
    let dyn_ = |c: DynCommand| Command::Dyn(c);
    match a {
        Action::TogglePlay => {
            if starts_play(st, input) {
                super::rewind::take(st, input);
            }
            out.toggle_play = true;
        }
        Action::Play => {
            // estimated with no model: nothing to play on (the viewer's toggle refuses the same)
            if no_model(st, input) {
                out.error = Some("the estimated solver has no model to play on: load one (File ▸ Open) or press Start sampling in the Model tab".into());
            } else {
                if starts_play(st, input) {
                    super::rewind::take(st, input);
                }
                out.commands.push(dyn_(DynCommand::Play));
            }
        }
        Action::Pause => out.commands.push(dyn_(DynCommand::Pause)),
        Action::Step => out.commands.push(dyn_(DynCommand::Step(1))),
        Action::ReverseVelocities => out.commands.push(dyn_(DynCommand::Rewind)),
        Action::RewindTo(i) => super::rewind::restore(st, i, out),
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
            if let (Some(file), Some(mut s)) = (Scene::preset(&name), crate::quality::preset(&name, st.quality_h)) {
                // the policy in force (Accepted by default in the app since M7 Task 11): a preset file itself is all-electron
                s.ionic = st.ionic;
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
        Action::Ionic(p) => {
            // the policy of the scene: the pseudo footprint is checked like a grid change
            let s = Scene { ionic: p, ..input.scene.clone() };
            match vram_check(st, &s) {
                Ok(()) => {
                    st.ionic = p;
                    if p != input.scene.ionic {
                        out.commands.push(Command::SetIonicPolicy(p));
                    }
                }
                Err(e) => out.error = Some(e),
            }
        }
        Action::ShowPartialCore(on) => {
            st.show_partial_core = on;
            out.commands.push(Command::DisplayPartialCore(on));
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
            let path = st.model_file.clone().unwrap_or_else(|| model_path(&st.scene_path));
            out.commands.push(dyn_(DynCommand::SaveModel(path.clone())));
            st.set_status(format!("saving the model to {}", path.display()));
        }
        Action::LoadModel => {
            let path = st.model_file.clone().unwrap_or_else(|| model_path(&st.scene_path));
            st.model_offer = None;
            if !path.exists() {
                out.error = Some(format!("no model file {}", path.display()));
            } else {
                match model_offer_for(st, input, &path) {
                    // the settings match (or the header cannot be read: the driver says why, as it always did)
                    None => out.commands.push(dyn_(DynCommand::LoadModel(path))),
                    Some(offer) => {
                        st.set_status(format!("{} was taught at other settings than this scene's — see the card over the view", path.display()));
                        st.model_offer = Some(offer);
                    }
                }
            }
        }
        Action::CancelModelOffer => st.model_offer = None,
        Action::ApplyModelSettings => {
            let Some(shown) = st.model_offer.take() else { return };
            if input.sampling() {
                st.model_offer = Some(shown);
                out.error = Some("a sampling pass is running: cancel it first, then load the model".into());
                return;
            }
            // against the scene as it is now, not as the card first saw it
            match model_offer_for(st, input, &shown.path) {
                None => out.commands.push(dyn_(DynCommand::LoadModel(shown.path))),
                Some(offer) => {
                    if let Some(why) = offer.blocked() {
                        out.error = Some(format!("the model's settings cannot be applied: {why}"));
                        st.model_offer = Some(offer);
                        return;
                    }
                    apply_edits(st, &offer, out);
                    st.set_status(format!("applied the settings of {} and loading it", offer.path.display()));
                    out.commands.push(dyn_(DynCommand::LoadModel(offer.path)));
                }
            }
        }
        Action::ApplyModelSettingsOnly(path) => {
            if input.sampling() {
                out.error = Some("a sampling pass is running: cancel it first".into());
                return;
            }
            match model_offer_for(st, input, &path) {
                None => st.set_status(format!("the scene already has the settings of {}", path.display())),
                Some(offer) => match offer.blocked() {
                    Some(why) => out.error = Some(format!("the model's settings cannot be applied: {why}")),
                    None => {
                        apply_edits(st, &offer, out);
                        st.set_status(format!("applied the settings of {} (the model is not loaded)", path.display()));
                    }
                },
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
        Action::TeachAroundBlockers => out.commands.push(dyn_(DynCommand::StartTeaching)),
        Action::StartOffered(plan) => out.commands.push(dyn_(DynCommand::StartSampling(plan))),
        Action::SampleAfterRefusal => {
            out.commands.push(dyn_(DynCommand::StartSampling(st.plan.clone())));
            st.plan.seed += 1;
        }
        Action::SetAudit(n) => {
            st.audit_every = n;
            out.commands.push(Command::SetAudit(n));
        }
        Action::VolumeMode(m) => {
            // dots draw an orbital as ψ̃² (`UiState::orbital_request`): the request follows the mode
            let before = st.orbital_request();
            st.volume_mode = m;
            if st.show != ShowMode::Density && !st.orbital_request().same_view(&before) {
                send_orbital_view(st, out);
            }
        }
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
        Action::ShowAngles(on) => st.show_angles = on,
        Action::NameStyle(s) => st.name_style = s,
        Action::RefreshBonds => {
            out.commands.push(dyn_(DynCommand::RefreshBonds));
            (st.show_bonds, st.show_bond_lengths, st.show_angles) = (true, true, true);
            st.bonds_refresh = Some(Instant::now());
            st.set_status("solving the density once for the bonds and angles");
        }
        Action::Relax => {
            if input.sampling() {
                st.set_status(super::RELAX_REFUSED_SAMPLING);
            } else {
                out.commands.push(dyn_(DynCommand::Relax));
            }
        }
        Action::StopRelax => out.commands.push(dyn_(DynCommand::StopRelax)),
        Action::SmoothMotion(on) => st.smooth_motion = on,
        Action::Show(mode) => {
            let was = st.show;
            st.show = mode;
            if mode != ShowMode::Density {
                st.open_tab(DockTab::Orbitals);
                // Orbitals mode before anything was picked: the HOMO of the payload on screen, if
                // there is one (else the provider's fallback shows it and the viewer adopts it)
                if mode == ShowMode::Orbitals && st.orbital_pick.is_none() {
                    st.orbital_pick = homo_of(input);
                }
            }
            // a view freshly switched on asks for no unoccupied orbitals until its first payload
            // is in: the display eigensolve takes seconds and the list should not wait for it
            st.unocc_deferred = was == ShowMode::Density && mode != ShowMode::Density;
            send_orbital_view(st, out);
        }
        Action::PickOrbital { spin, track } => {
            st.orbital_pick = Some((spin, track));
            st.orbital_pick_adopted = false;
            st.show = ShowMode::Orbitals;
            st.open_tab(DockTab::Orbitals);
            send_orbital_view(st, out);
        }
        Action::OrbitalPhase(phase) => {
            st.orbital_phase = phase;
            if st.show == ShowMode::Orbitals {
                send_orbital_view(st, out);
            }
        }
        Action::Unoccupied(m) => {
            st.unoccupied = m.min(UNOCCUPIED_MAX);
            st.unocc_deferred = false;
            if matches!(st.show, ShowMode::Orbitals | ShowMode::Coloured) {
                send_orbital_view(st, out);
            }
        }
        Action::Include { track, on } => {
            let mut set = st.included.take().unwrap_or_else(|| occupied_tracks(input));
            if on {
                set.insert(track);
            } else {
                set.remove(&track);
            }
            st.included = Some(set);
            if st.show == ShowMode::Coloured {
                send_orbital_view(st, out);
            }
        }
        Action::IncludeAllOccupied => {
            st.included = None;
            if st.show == ShowMode::Coloured {
                send_orbital_view(st, out);
            }
        }
        Action::ColourBlend { additive } => st.colour_additive = additive,
        Action::ColourScale { physical } => {
            st.colour_physical = physical;
            if st.show == ShowMode::Coloured {
                send_orbital_view(st, out);
            }
        }
        Action::WarpFields(on) => st.warp_fields = on,
        Action::ColourLook(look) => {
            st.colour_look = look;
            if st.show == ShowMode::Coloured {
                send_orbital_view(st, out);
            }
        }
    }
}

/// The orbital view as the panel now means it, to the worker (display only).
fn send_orbital_view(st: &UiState, out: &mut PanelOutput) {
    out.commands.push(Command::OrbitalView(st.orbital_request()));
}

/// The newest orbital payload's rows, if any.
fn payload_rows<'a>(input: &'a PanelInput<'_>) -> &'a [sim::OrbitalInfo] {
    input.snapshot.and_then(|s| s.dynamics.orbitals.as_ref()).map_or(&[][..], |o| &o.rows[..])
}

/// The tracks of the occupied orbitals on screen (the include set's default).
fn occupied_tracks(input: &PanelInput<'_>) -> std::collections::BTreeSet<u32> {
    payload_rows(input).iter().filter(|r| r.occupation > 0.0).map(|r| r.track).collect()
}

/// The HOMO of the payload on screen — the occupied row whose set has the highest eigenvalue — as
/// a pick.
fn homo_of(input: &PanelInput<'_>) -> Option<(sim::Spin, u32)> {
    payload_rows(input)
        .iter()
        .filter(|r| r.occupation > 0.0)
        // the first of equal levels (↑ before ↓): a closed shell's HOMO is the ↑ orbital
        .fold(None::<&sim::OrbitalInfo>, |best, r| match best {
            Some(b) if b.eps_range.1 >= r.eps_range.1 => Some(b),
            _ => Some(r),
        })
        .map(|r| (r.spin, r.track))
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
            ReverseVelocities => "reverse-velocities".into(),
            RewindTo(0) => "rewind".into(),
            RewindTo(i) => format!("rewind {i}"),
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
            Ionic(p) => format!("ionic {}", p.word()),
            ShowPartialCore(b) => format!("partial-core {b}"),
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
            ShowAngles(b) => format!("angles {b}"),
            NameStyle(s) => format!("names {}", style_word(*s)),
            Relax => "relax".into(),
            RefreshBonds => "refresh-bonds".into(),
            StopRelax => "stop-relax".into(),
            SmoothMotion(b) => format!("smooth-motion {b}"),
            Show(m) => format!("show {}", m.word()),
            PickOrbital { spin, track } => format!("orbital {} {track}", spin_word(*spin)),
            OrbitalPhase(b) => format!("orbital-phase {b}"),
            Unoccupied(m) => format!("unoccupied {m}"),
            Include { track, on: o } => format!("orbital-include {track} {}", on(*o)),
            IncludeAllOccupied => "orbital-include-reset".into(),
            ColourBlend { additive } => format!("colour-blend {}", if *additive { "additive" } else { "weighted" }),
            ColourScale { physical } => format!("colour-scale {}", if *physical { "physical" } else { "peak" }),
            WarpFields(on) => format!("warp-fields {on}"),
            ColourLook(l) => format!(
                "colour-look {} {} {}",
                match l.enclose {
                    None => "density",
                    Some(_) if l.shell => "shell",
                    Some(_) => "filled",
                },
                l.enclose.unwrap_or(90),
                if l.hide_core { "valence" } else { "all" }
            ),
            Quality(h) => format!("quality {h}"),
            TempUnits(u) => format!(
                "temp-units {}",
                match u {
                    super::TempUnits::K => "k",
                    super::TempUnits::C => "c",
                    super::TempUnits::F => "f",
                }
            ),
            ApplyGrid | SaveScene | LoadScene { .. } | SaveModel | LoadModel | ApplyModelSettings | ApplyModelSettingsOnly(_) | CancelModelOffer | Sample { .. } | StartOffered(_) | SampleAfterRefusal | TeachAroundBlockers => return None,
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
            "toggle-play" | "play" | "pause" | "step" | "reverse-velocities" | "deselect" | "focus" | "forget" | "cancel-sampling" | "relax" | "stop-relax" | "orbital-include-reset" => {
                want(0)?;
                match head {
                    "toggle-play" => TogglePlay,
                    "play" => Play,
                    "pause" => Pause,
                    "step" => Step,
                    "reverse-velocities" => ReverseVelocities,
                    "deselect" => Deselect,
                    "focus" => Focus,
                    "forget" => Forget,
                    "cancel-sampling" => CancelSampling,
                    "relax" => Relax,
                    "refresh-bonds" => RefreshBonds,
                    "orbital-include-reset" => IncludeAllOccupied,
                    _ => StopRelax,
                }
            }
            "rewind" => match args {
                [] => RewindTo(0),
                [i] => RewindTo(num(head, i)?),
                _ => return Err(format!("action rewind takes 0 or 1 arguments (the snapshot, 0 = the newest), got {}", args.len())),
            },
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
                    "edit" => ToolMode::Edit,
                    w => return Err(format!("action tool: orbit, place, select or edit, got '{w}'")),
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
            "ionic" => {
                want(1)?;
                Ionic(sim::IonicPolicy::parse(args[0]).ok_or_else(|| format!("action ionic: none, cores, accepted or all, got '{}'", args[0]))?)
            }
            "gravity" | "com-correction" | "draft" | "bonds" | "labels" | "badge" | "angles" | "smooth-motion" | "partial-core" => {
                want(1)?;
                let b = flag(args[0])?;
                match head {
                    "partial-core" => ShowPartialCore(b),
                    "gravity" => Gravity(b),
                    "com-correction" => ComCorrection(b),
                    "draft" => Draft(b),
                    "bonds" => ShowBonds(b),
                    "labels" => ShowLabels(b),
                    "badge" => ShowBadge(b),
                    "angles" => ShowAngles(b),
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
            "show" => {
                want(1)?;
                Show(ShowMode::from_word(args[0]).ok_or_else(|| format!("action show: density, orbitals, colour or spin, got '{}'", args[0]))?)
            }
            "orbital" => {
                want(2)?;
                PickOrbital {
                    spin: match args[0] {
                        "up" => sim::Spin::Up,
                        "down" => sim::Spin::Down,
                        w => return Err(format!("action orbital: up or down, got '{w}'")),
                    },
                    track: num(head, args[1])?,
                }
            }
            "orbital-phase" => {
                want(1)?;
                OrbitalPhase(flag(args[0])?)
            }
            "unoccupied" => {
                want(1)?;
                Unoccupied(num(head, args[0])?)
            }
            "orbital-include" => {
                want(2)?;
                Include { track: num(head, args[0])?, on: flag(args[1])? }
            }
            "colour-blend" => {
                want(1)?;
                ColourBlend {
                    additive: match args[0] {
                        "additive" => true,
                        "weighted" => false,
                        w => return Err(format!("action colour-blend: additive or weighted, got '{w}'")),
                    },
                }
            }
            "colour-scale" => {
                want(1)?;
                ColourScale {
                    physical: match args[0] {
                        "peak" => false,
                        "physical" => true,
                        w => return Err(format!("action colour-scale: peak or physical, got '{w}'")),
                    },
                }
            }
            "warp-fields" => {
                want(1)?;
                WarpFields(match args[0] {
                    "true" => true,
                    "false" => false,
                    w => return Err(format!("action warp-fields: true or false, got '{w}'")),
                })
            }
            "colour-look" => {
                want(3)?;
                let pct: u8 = num(head, args[1])?;
                if !(50..=99).contains(&pct) {
                    return Err(format!("action colour-look: the enclosed share is 50–99 %, got {pct}"));
                }
                let (enclose, shell) = match args[0] {
                    "shell" => (Some(pct), true),
                    "filled" => (Some(pct), false),
                    "density" => (None, true),
                    w => return Err(format!("action colour-look: shell, filled or density, got '{w}'")),
                };
                let hide_core = match args[2] {
                    "all" => false,
                    "valence" => true,
                    w => return Err(format!("action colour-look: all or valence, got '{w}'")),
                };
                ColourLook(sim::ColourLook { enclose, shell, hide_core })
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

fn spin_word(s: sim::Spin) -> &'static str {
    match s {
        sim::Spin::Up => "up",
        sim::Spin::Down => "down",
    }
}

fn tool_word(m: ToolMode) -> &'static str {
    match m {
        ToolMode::Orbit => "orbit",
        ToolMode::Place => "place",
        ToolMode::Select => "select",
        ToolMode::Edit => "edit",
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

/// The offer for loading `path` into the scene in force (`ui::model_offer`): `None` when the model's
/// settings match the scene's, or when its header cannot be read (the driver's own refusal then says why).
fn model_offer_for(st: &UiState, input: &PanelInput<'_>, path: &std::path::Path) -> Option<super::model_offer::ModelOffer> {
    let prov = surrogate::sidecar::read_provenance(path).ok()?;
    // the teacher the session has: the provider's name when a model is up, else what the machine decides
    let teacher = input.model().map_or(if st.gpu_available { "Kohn-Sham (GPU)" } else { "Kohn-Sham (CPU)" }, |m| m.teacher);
    let vram = |s: &Scene| vram_check(st, s);
    let ctx = super::model_offer::Context {
        teacher,
        sample_tol: crate::worker::SAMPLE_TOL,
        pseudised: !super::pseudo::pseudised(input).is_empty(),
        vram: &vram,
    };
    super::model_offer::offer(path, &prov, input.scene, &ctx)
}

/// The GPU memory guard for a scene the panel is about to send (`quality::vram_check` with the
/// backend's solver count and the budget): refused before the worker allocates anything.
/// The estimated solver with no fitted model and no pass that will make one: nothing to play on.
fn no_model(st: &UiState, input: &PanelInput<'_>) -> bool {
    st.backend == Backend::Estimated && !input.model().is_some_and(|m| m.fitted) && !input.sampling()
}

/// Play from paused that the driver will run (not during a pass or a relax, which refuse it): the
/// moment a Play snapshot is taken (`ui::rewind`).
fn starts_play(st: &UiState, input: &PanelInput<'_>) -> bool {
    !input.playing && !no_model(st, input) && !input.sampling() && !input.relaxing()
}

pub(super) fn vram_check(st: &UiState, s: &Scene) -> Result<(), String> {
    crate::quality::vram_check(s, crate::quality::gpu_solvers(st.backend, st.gpu_available), crate::quality::vram_budget())
}

/// The settings edits of a model offer (`model_offer::Edit`), sent as the ordinary commands.
fn apply_edits(st: &mut UiState, offer: &super::model_offer::ModelOffer, out: &mut PanelOutput) {
        for e in &offer.edits {
            use super::model_offer::Edit;
            match e {
                Edit::Grid { n, h } => {
                    // as Apply grid: the new box is the one later quality changes keep
                    (st.grid_n, st.grid_h) = (*n, *h);
                    st.box_len = (*n as f64 - 1.0) * *h;
                    out.commands.push(Command::SetGrid { n: *n, h: *h });
                }
                Edit::Sigma(x) => out.commands.push(Command::SetSigma(*x)),
                Edit::Charge(c) => out.commands.push(Command::SetCharge(*c)),
                Edit::Multiplicity(m) => out.commands.push(Command::SetMultiplicity(*m)),
                Edit::Ionic(p) => {
                    st.ionic = *p;
                    out.commands.push(Command::SetIonicPolicy(*p));
                }
                Edit::Isotope { index, a } => out.commands.push(Command::Dyn(DynCommand::SetIsotope { index: *index, a: *a })),
            }
        }
}
