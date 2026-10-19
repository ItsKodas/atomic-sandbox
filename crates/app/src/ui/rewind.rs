//! Rewind to the Play snapshot (user request 2026-10-19, notes `2026-10-13-m5-user-requests.md`
//! "Placement assist and rewind-to-Play snapshot": "if the simulation breaks, hit rewind to go back
//! to when you hit play"). Pressing Play ([`Action::Play`], [`Action::TogglePlay`] from paused)
//! stores the scene as the viewer has it — positions, velocities, isotopes, charge, multiplicity,
//! grid, σ, ionic policy — with the pins and the run settings in force (walls, field, thermostat,
//! COM correction) and when it was taken (simulated time, step, wall clock). The newest
//! [`KEPT`] are kept, newest first. The toolbar's Rewind ([`Action::RewindTo`]) puts one back
//! through the scene-replacement path every preset load uses (`Command::SetScene` →
//! `DynCommand::ReplaceScene`: the run pauses, the snapshot's velocities are kept, time, pins and
//! the drag restart), then re-sends the settings and the pins; `sim` is unchanged. Time restarts
//! at 0 (a replaced scene is a new run); the estimated model is not rolled back — it only learns.
//! The old Rewind (v → −v) is Physics ▸ Reverse velocities ([`Action::ReverseVelocities`]).
use super::activity::Mark;
use super::{theme, Action, PanelInput, PanelOutput, SceneSource, UiState};
use crate::scene::Scene;
use crate::sim::Command;
use egui::Ui;
use sim::DynCommand;
use std::time::{Duration, Instant};

/// Snapshots kept (the oldest is dropped).
pub const KEPT: usize = 5;

/// The toolbar button's tooltip.
pub const TIP: &str = "back to when you pressed Play: the run pauses and the scene returns to the newest Play snapshot — positions, velocities, pins, walls, field and thermostat — so Play replays from there (time restarts at 0; with the thermostat on the replay differs, its kicks are random). Right-click or ▾ for an older one. The estimated model is not rolled back: it keeps what it learned";

/// The disabled button's tooltip.
pub const NONE_YET: &str = "no snapshot yet: one is taken each time you press Play";

/// What a snapshot puts back: the scene and the run settings in force when Play was pressed.
#[derive(Clone, Debug, PartialEq)]
pub struct RunState {
    /// The viewer's scene: positions, velocities, isotopes, charge, multiplicity, grid, σ, policy.
    pub scene: Scene,
    pub pins: Vec<bool>,
    pub walls_on: bool,
    pub wall_half_width: f64,
    pub gravity: bool,
    /// (on, T in K, τ in fs).
    pub thermostat: (bool, f64, f64),
    pub com_correction: bool,
    /// The scene chip's source (Reset reloads it), the box quality changes keep, the quality.
    pub source: SceneSource,
    pub box_len: f64,
    pub quality_h: f64,
}

/// One Play snapshot and when it was taken.
#[derive(Clone, Debug)]
pub struct PlaySnapshot {
    pub state: RunState,
    pub sim_time_fs: f64,
    pub step: u64,
    /// Wall clock.
    pub taken: Instant,
}

impl PlaySnapshot {
    /// The menu's line: "t = 12.5 fs · step 50 · H₂O · 2 min ago".
    pub fn label(&self) -> String {
        let z: Vec<u8> = self.state.scene.nuclei.iter().map(|n| n.z).collect();
        let what = if z.is_empty() { "no atoms".to_string() } else { crate::names::formula(&z, None) };
        format!("t = {:.1} fs · step {} · {what} · {}", self.sim_time_fs, self.step, ago(self.taken.elapsed()))
    }
}

/// "just now", "12 s ago", "3 min ago", "1 h 5 min ago".
pub fn ago(d: Duration) -> String {
    let s = d.as_secs();
    match s {
        0..=1 => "just now".into(),
        2..=59 => format!("{s} s ago"),
        60..=3599 => format!("{} min ago", s / 60),
        _ => format!("{} h {} min ago", s / 3600, s / 60 % 60),
    }
}

/// Play was pressed: store the state the run starts from. Pressing Play on a state already kept
/// (Play right after a rewind) moves that snapshot to the front instead of keeping a copy, so
/// Rewind returns to it.
pub fn take(st: &mut UiState, input: &PanelInput<'_>) {
    let d = input.snapshot.map(|s| &s.dynamics);
    let n = input.scene.nuclei.len();
    // the driver's pins, when its scene has the mirror's nuclei (an add in flight: none)
    let pins = d.map(|d| d.pins.clone()).filter(|p| p.len() == n).unwrap_or_else(|| vec![false; n]);
    let state = RunState {
        scene: input.scene.clone(),
        pins,
        walls_on: st.walls_on,
        wall_half_width: st.wall_half_width,
        gravity: st.gravity,
        thermostat: (st.thermostat_on, st.thermostat_t, st.thermostat_tau),
        com_correction: st.com_correction,
        source: st.source.clone(),
        box_len: st.box_len,
        quality_h: st.quality_h,
    };
    let at = st.activity.clock_now();
    if let Some(i) = st.play_snapshots.iter().position(|s| s.state == state) {
        let s = st.play_snapshots.remove(i).expect("found");
        st.activity.timeline.push(Mark::Snapshot(at, s.sim_time_fs));
        st.play_snapshots.push_front(s);
        return;
    }
    let (sim_time_fs, step) = d.map_or((0.0, 0), |d| (d.sim_time_fs(), d.step));
    st.activity.timeline.push(Mark::Snapshot(at, sim_time_fs));
    st.play_snapshots.push_front(PlaySnapshot { state, sim_time_fs, step, taken: Instant::now() });
    st.play_snapshots.truncate(KEPT);
}

/// Rewind to snapshot `i` (0 = the newest): the panel's settings follow it, then the commands —
/// Pause, the scene (velocities kept; time, pins and drag restart), the field, the COM
/// correction, the walls, the thermostat when it differs, and the pins.
pub fn restore(st: &mut UiState, i: usize, out: &mut PanelOutput) {
    let Some(snap) = st.play_snapshots.get(i).cloned() else {
        out.error = Some(if st.play_snapshots.is_empty() { NONE_YET.into() } else { format!("no snapshot {i}: {} kept (0 = the newest)", st.play_snapshots.len()) });
        return;
    };
    let s = snap.state;
    if let Err(e) = super::actions::vram_check(st, &s.scene) {
        out.error = Some(e);
        return;
    }
    let dyn_ = Command::Dyn;
    st.source = s.source;
    st.adopt(&s.scene);
    st.box_len = s.box_len;
    st.quality_h = s.quality_h;
    st.quality_slider = s.quality_h;
    st.ionic = s.scene.ionic;
    st.gravity = s.gravity;
    st.com_correction = s.com_correction;
    st.walls_on = s.walls_on;
    st.wall_half_width = s.wall_half_width;
    // Pause first: the scene replacement would pause too, but the viewer would then word it as an
    // edit that paused the run ("paused: this edit replaces the scene…") over the rewind's status
    out.commands.push(dyn_(DynCommand::Pause));
    out.commands.push(Command::SetScene(s.scene));
    out.commands.push(dyn_(DynCommand::SetExternalField(st.field())));
    out.commands.push(dyn_(DynCommand::SetComCorrection(st.com_correction)));
    out.commands.push(dyn_(DynCommand::SetWalls(st.walls())));
    // only a changed thermostat: re-sending one that is on would redraw velocities at rest
    let (on, t, tau) = s.thermostat;
    if (on, t, tau) != (st.thermostat_on, st.thermostat_t, st.thermostat_tau) {
        st.thermostat_on = on;
        st.thermostat_t = t;
        st.thermostat_tau = tau;
        out.commands.push(dyn_(DynCommand::SetThermostat(st.thermostat())));
    }
    for (k, _) in s.pins.iter().enumerate().filter(|(_, &p)| p) {
        out.commands.push(dyn_(DynCommand::Pin(k, true)));
    }
    let at = st.activity.clock_now();
    st.activity.timeline.push(Mark::Restore(at, snap.sim_time_fs));
    st.set_status(format!(
        "rewound to the Play snapshot at t = {:.1} fs (step {}, {}): paused — press Play to run it again from there; time restarts at 0. The estimated model keeps what it learned",
        snap.sim_time_fs,
        snap.step,
        ago(snap.taken.elapsed())
    ));
}

/// The Rewind button's menu (right-click or ▾): the kept snapshots, newest first.
pub fn menu(ui: &mut Ui, st: &UiState, acts: &mut Vec<Action>) {
    ui.set_min_width(260.0);
    theme::section(ui, "Rewind to a Play snapshot");
    if st.play_snapshots.is_empty() {
        theme::note(ui, NONE_YET);
        return;
    }
    for (i, s) in st.play_snapshots.iter().enumerate() {
        let text = if i == 0 { format!("{} (newest)", s.label()) } else { s.label() };
        if ui.button(text).clicked() {
            acts.push(Action::RewindTo(i));
            ui.close();
        }
    }
    theme::note(ui, format!("taken each time you press Play; the newest {KEPT} are kept. The estimated model is not rolled back."));
}

/// The Rewind button and its ▾ (the toolbar's Time group).
pub fn button(ui: &mut Ui, st: &UiState, compact: bool, acts: &mut Vec<Action>) {
    let any = !st.play_snapshots.is_empty();
    let r = super::icons::button(ui, super::icons::Icon::Rewind, 28.0, false, any).on_hover_text(TIP).on_disabled_hover_text(NONE_YET);
    if r.clicked() {
        acts.push(Action::RewindTo(0));
    }
    r.context_menu(|ui| menu(ui, st, acts));
    if !compact {
        let more = ui.add_enabled(any, egui::Button::new(egui::RichText::new("▾").size(11.0)).min_size(egui::vec2(14.0, 28.0)).frame(false)).on_hover_text("pick an older Play snapshot").on_disabled_hover_text(NONE_YET);
        egui::Popup::menu(&more).show(|ui| menu(ui, st, acts));
    }
}
