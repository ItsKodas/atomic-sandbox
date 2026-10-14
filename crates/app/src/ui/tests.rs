//! The chrome without a window: the old panel's unit tests, one test per control path (a
//! toolbar / menu / dock action emits the same `Command` / `DynCommand` as the old side panel
//! did for that control), the search, the log, and a headless egui smoke run of every layout.
use super::actions::{apply, sample_button};
use super::logbook::LogKind;
use super::*;
use super::PROBES;
use crate::scene::Scene;
use crate::sim::{Command, ModelStatus, PassOffer, Snapshot};
use render::camera::OrbitCamera;

/// g = 9.8 m/s² in atomic units: a₀ / t_au² = 0.529177e-10 m / (2.418884e-17 s)²
/// = 9.0442e22 m/s², so g = 1.0836e-22 Bohr/a.u.².
#[test]
fn gravity_in_atomic_units() {
    let g = gravity_au();
    assert!((g - 1.0836e-22).abs() < 2e-26, "g = {g:e}");
}

/// The box warning names the nucleus nearest the face: none for the preset (6 Bohr half-box,
/// nuclei at ±0.8), amber within 1.5 Bohr, red outside.
#[test]
fn box_warning_flags_nuclei_near_or_beyond_the_face() {
    let mut s = Scene::preset("h2").unwrap();
    assert_eq!(box_warning(&s), None);
    s.nuclei[1].pos = [0.0, -5.0, 0.0];
    let (c, m, i) = box_warning(&s).unwrap();
    assert_eq!((c, i), (AMBER, 1));
    assert!(m.contains("#1") && m.contains("1.0 Bohr"), "{m}");
    s.nuclei[0].pos = [0.0, 0.0, 7.5];
    let (c, m, i) = box_warning(&s).unwrap();
    assert_eq!((c, i), (RED, 0));
    assert!(m.contains("#0") && m.contains("outside"), "{m}");
}

/// A new worker gets the panel's run settings; defaults send only speed and COM correction.
#[test]
fn run_settings_mirror_the_panel() {
    let mut st = UiState::new("h2", &Scene::preset("h2").unwrap());
    assert!(st.com_correction, "COM correction defaults on in the app");
    assert_eq!(st.run_settings().len(), 2);
    st.thermostat_on = true;
    st.gravity = true;
    st.dt_fs = 0.4;
    let v = st.run_settings();
    assert!(v.iter().any(|c| matches!(c, DynCommand::SetThermostat(Some((t, tau))) if *t == 300.0 && *tau == 10.0)));
    assert!(v.iter().any(|c| matches!(c, DynCommand::SetDt(x) if *x == 0.4)));
    assert!(v.iter().any(|c| matches!(c, DynCommand::SetExternalField(a) if a[1] < 0.0)));
    st.walls_on = true;
    let v = st.run_settings();
    assert!(v.iter().any(|c| matches!(c, DynCommand::SetWalls(Some(w)) if w.half_width == 4.5)), "h2: 6 − 1.5 Bohr");
}

/// The container's default sits 1.5 Bohr inside the face and within the allowed range; the
/// maximum is the cell less its margin and h/2 (49 × 0.25: 6 − 0.75 − 0.125).
#[test]
fn wall_defaults_fit_the_cell() {
    let s = Scene::preset("h2").unwrap();
    assert!((max_wall_half_width(&s) - 5.125).abs() < 1e-12);
    assert_eq!(UiState::new("h2", &s).wall_half_width, 4.5);
    let mut tiny = s.clone();
    tiny.grid_n = 21; // half 2.5: max 1.625, default clamped up to 1.5
    assert_eq!(UiState::new("h2", &tiny).wall_half_width, 1.5);
}

/// Scene ▸ Container, View ▸ the estimated solver's cloud, Physics ▸ draft mode (M5b): the
/// commands `main`'s side panel sent.
#[test]
fn m5b_actions_send_walls_cloud_and_draft() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let mut st = UiState::new("h2", &scene);
    let c = act(&mut st, &inp, Action::Walls { on: true, half_width: 3.0 });
    assert!(matches!(one_dyn(&c), DynCommand::SetWalls(Some(w)) if w.half_width == 3.0));
    let c = act(&mut st, &inp, Action::Walls { on: false, half_width: 3.0 });
    assert!(matches!(one_dyn(&c), DynCommand::SetWalls(None)));
    let c = act(&mut st, &inp, Action::Cloud { superposition: false, refresh: true, refresh_s: 4.0 });
    assert!(matches!(c.commands.as_slice(), [Command::SetCloud { superposition: false, refresh_s: Some(x) }] if *x == 4.0));
    let c = act(&mut st, &inp, Action::Draft(true));
    assert!(matches!(c.commands.as_slice(), [Command::SetDraft(true)]) && st.draft);
}

/// "watchable": one ~10 fs vibration in ≈ 0.5 s — at dt 0.25 fs on the surrogate (≈ 0.01 ms
/// a step) 1 step per paced 16 ms tick, 15.6 fs/s; a DFT step (100 ms) is slower than that
/// whatever the speed (1 step, 2.5 fs/s); a small dt asks for more steps per tick.
#[test]
fn the_watchable_speed_shows_a_vibration_in_about_half_a_second() {
    let (n, r) = watchable_speed(0.25, 0.01);
    assert_eq!(n, 1);
    assert!((r - 15.625).abs() < 1e-9, "{r}");
    let (n, r) = watchable_speed(0.25, 100.0);
    assert_eq!(n, 1);
    assert!((r - 2.5).abs() < 1e-9, "{r}");
    let (n, r) = watchable_speed(0.01, 0.01);
    assert_eq!(n, 32);
    assert!((r - 20.0).abs() < 1e-9, "{r}");
}

// ---- helpers ----

fn camera() -> OrbitCamera {
    OrbitCamera::new(glam::Vec3::ZERO, 14.0)
}

fn input<'a>(scene: &'a Scene, snapshot: Option<&'a Snapshot>, cam: &'a OrbitCamera) -> PanelInput<'a> {
    PanelInput {
        scene,
        snapshot,
        busy: false,
        playing: false,
        last_error: None,
        selected: None,
        measured_fs_per_s: None,
        camera: cam,
        density_label: None,
        draft_banner: None,
    }
}

fn model() -> ModelStatus {
    ModelStatus {
        teacher: "GpuKs",
        fitted: true,
        trusted: true,
        untrusted_reason: None,
        samples: 273,
        pending_refit: false,
        epoch: 3,
        fit: Some((271, 40, 2.1e-6, 1.6e-5)),
        residuals: None,
        force_bound: 2e-3,
        provenance_mismatch: None,
        last_refit: None,
        refits: 0,
        autoload_refused: None,
        audit_every: 2000,
        draft: false,
        last_audit_step: Some(1600),
        last_gap: Some((-5.3e-5, 1600)),
        quarantined: 2,
        recovery: None,
        audits_background: true,
        audit_in_flight: true,
        last_audit: Some(sim::surrogate_ks::AuditRecord {
            gap: -5.3e-5,
            verdict: sim::surrogate_ks::AuditVerdict::Passed,
            face_density: None,
            unaudited_steps: 18400,
            asynchronous: true,
            teacher_ms: 274.0,
        }),
        stuck: None,
        solve_s: Some(1.0),
        pass_eta_s: Some(25.0),
        pass_offer: None,
    }
}

fn snapshot(scene: &Scene, backend: Backend, model: Option<ModelStatus>) -> Snapshot {
    let mut dynamics = sim::DynSnapshot::default();
    dynamics.scene = scene.to_sim().unwrap();
    dynamics.forces_stale = false;
    dynamics.history = (0..50).map(|k| (k as f64 * 10.0, -1.1 + 1e-4 * (k as f64).sin(), 300.0 + k as f64)).collect();
    dynamics.history_source = (0..50).map(|k| if k % 7 == 0 { sim::Source::Fallback } else { sim::Source::Surrogate }).collect();
    dynamics.notes = vec!["thermostat on: drew Maxwell–Boltzmann velocities at 300 K".into()];
    dynamics.notes_total = 1;
    dynamics.drift_per_ps = Some(2.4e-4);
    Snapshot {
        dynamics,
        scene: scene.clone(),
        backend,
        engine: if backend == Backend::Cpu { Backend::Cpu } else { Backend::Gpu },
        model,
        atom_profiles: Default::default(),
        edits: 0,
        driver_commands: 0,
        working: false,
    }
}

fn act(st: &mut UiState, inp: &PanelInput<'_>, a: Action) -> PanelOutput {
    let mut out = PanelOutput::default();
    apply(st, inp, a, &mut out);
    out
}

fn one_dyn(out: &PanelOutput) -> &DynCommand {
    assert_eq!(out.commands.len(), 1, "{:?}", out.commands);
    match &out.commands[0] {
        Command::Dyn(c) => c,
        c => panic!("not a driver command: {c:?}"),
    }
}

// ---- one test per control path ----

/// Toolbar ▸ Time and Speed and dt: the old Play / Step / Rewind / speed slider / watchable /
/// dt paths. Speed never touches dt (rule 8).
#[test]
fn toolbar_time_speed_and_dt_send_the_old_commands() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let mut st = UiState::new("h2", &scene);
    let out = act(&mut st, &inp, Action::TogglePlay);
    assert!(out.toggle_play && out.commands.is_empty());
    assert!(matches!(one_dyn(&act(&mut st, &inp, Action::Step)), DynCommand::Step(1)));
    assert!(matches!(one_dyn(&act(&mut st, &inp, Action::Rewind)), DynCommand::Rewind));
    let dt = st.dt_fs;
    assert!(matches!(one_dyn(&act(&mut st, &inp, Action::SetSpeed(120))), DynCommand::SetSpeed(120)));
    assert_eq!((st.speed, st.dt_fs), (120, dt), "speed changes steps per tick, never dt");
    assert!(matches!(one_dyn(&act(&mut st, &inp, Action::SetDt(0.4))), DynCommand::SetDt(x) if *x == 0.4));
    assert_eq!(st.dt_fs, 0.4);
}

/// Toolbar ▸ Tool and the element chips: picking an element also switches to Place (new).
#[test]
fn element_chip_picks_the_element_and_switches_to_place() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let mut st = UiState::new("h2", &scene);
    act(&mut st, &inp, Action::Tool(ToolMode::Select));
    assert_eq!(st.mode, ToolMode::Select);
    let out = act(&mut st, &inp, Action::Element(8));
    assert!(out.commands.is_empty());
    assert_eq!((st.element, st.mode), (8, ToolMode::Place));
}

/// Toolbar / dock ▸ Selection: Delete → RemoveNucleus, Focus, Pin → Dyn(Pin), Esc → deselect.
#[test]
fn selection_actions_send_the_old_commands() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let mut st = UiState::new("h2", &scene);
    let out = act(&mut st, &inp, Action::Delete(1));
    assert!(matches!(out.commands.as_slice(), [Command::RemoveNucleus(1)]));
    assert!(act(&mut st, &inp, Action::Focus).focus_selected);
    assert!(matches!(one_dyn(&act(&mut st, &inp, Action::Pin(0, true))), DynCommand::Pin(0, true)));
    assert!(act(&mut st, &inp, Action::Deselect).deselect);
}

/// Toolbar ▸ Solver: GPU is refused without an adapter (the old radio was disabled).
#[test]
fn solver_segments_pick_the_backend() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let mut st = UiState::new("h2", &scene);
    act(&mut st, &inp, Action::Backend(Backend::Gpu));
    assert_eq!(st.backend, Backend::Cpu, "no adapter: GPU stays disabled");
    st.gpu_available = true;
    act(&mut st, &inp, Action::Backend(Backend::Gpu));
    assert_eq!(st.backend, Backend::Gpu);
    act(&mut st, &inp, Action::Backend(Backend::Estimated));
    assert_eq!(st.backend, Backend::Estimated);
}

/// File ▸ Presets and the scene chip: a preset (or Reset) sends SetScene and adopts its grid.
#[test]
fn presets_replace_the_scene_and_adopt_its_grid() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let mut st = UiState::new("h2", &scene);
    st.grid_n = 33;
    let out = act(&mut st, &inp, Action::Preset("h2o".into()));
    let want = Scene::preset("h2o").unwrap();
    assert!(matches!(out.commands.as_slice(), [Command::SetScene(s)] if *s == want));
    assert_eq!((st.preset.as_str(), st.grid_n), ("h2o", want.grid_n));
    let out = act(&mut st, &inp, Action::Preset("nope".into()));
    assert!(out.commands.is_empty() && st.preset == "h2o");
}

/// Physics menu: the thermostat sends on a switch or a retune while on (never while off), the
/// gravity preset sends the field, COM correction its flag.
#[test]
fn physics_menu_sends_thermostat_field_and_com() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let mut st = UiState::new("h2", &scene);
    let out = act(&mut st, &inp, Action::Thermostat { on: false, t: 350.0, tau: 10.0 });
    assert!(out.commands.is_empty(), "a retune while off sends nothing");
    assert_eq!(st.thermostat_t, 350.0);
    let out = act(&mut st, &inp, Action::Thermostat { on: true, t: 350.0, tau: 10.0 });
    assert!(matches!(one_dyn(&out), DynCommand::SetThermostat(Some((t, tau))) if *t == 350.0 && *tau == 10.0));
    let out = act(&mut st, &inp, Action::Thermostat { on: true, t: 350.0, tau: 20.0 });
    assert!(matches!(one_dyn(&out), DynCommand::SetThermostat(Some((_, tau))) if *tau == 20.0));
    let out = act(&mut st, &inp, Action::Thermostat { on: false, t: 350.0, tau: 20.0 });
    assert!(matches!(one_dyn(&out), DynCommand::SetThermostat(None)));
    let out = act(&mut st, &inp, Action::Gravity(true));
    assert!(matches!(one_dyn(&out), DynCommand::SetExternalField(a) if a[1] == -gravity_au()));
    let out = act(&mut st, &inp, Action::Gravity(false));
    assert!(matches!(one_dyn(&out), DynCommand::SetExternalField(a) if *a == [0.0; 3]));
    let out = act(&mut st, &inp, Action::ComCorrection(false));
    assert!(matches!(one_dyn(&out), DynCommand::SetComCorrection(false)));
    assert!(!st.com_correction);
}

/// Scene menu: charge / multiplicity / σ send only a change; Apply grid validates.
#[test]
fn scene_menu_sends_electron_and_grid_edits() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let mut st = UiState::new("h2", &scene);
    assert!(act(&mut st, &inp, Action::SetCharge(scene.charge)).commands.is_empty());
    assert!(matches!(act(&mut st, &inp, Action::SetCharge(1)).commands.as_slice(), [Command::SetCharge(1)]));
    assert!(matches!(act(&mut st, &inp, Action::SetMultiplicity(Some(3))).commands.as_slice(), [Command::SetMultiplicity(Some(3))]));
    assert!(act(&mut st, &inp, Action::SetMultiplicity(scene.multiplicity)).commands.is_empty());
    assert!(matches!(act(&mut st, &inp, Action::SetSigma(Some(0.3))).commands.as_slice(), [Command::SetSigma(Some(x))] if *x == 0.3));
    st.grid_n = 41;
    st.grid_h = 0.3;
    assert!(matches!(act(&mut st, &inp, Action::ApplyGrid).commands.as_slice(), [Command::SetGrid { n: 41, h }] if *h == 0.3));
    st.grid_n = 40;
    let out = act(&mut st, &inp, Action::ApplyGrid);
    assert!(out.commands.is_empty() && out.error.is_some(), "an even n is refused");
}

/// File ▸ Scene file: Save writes the file and says so; Load sends SetScene (and, estimated, the
/// `.model` sidecar's LoadModel); a missing model file is a UI error.
#[test]
fn file_menu_saves_and_loads_scene_and_model() {
    let dir = std::env::temp_dir().join(format!("atomic-ui-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("t.scene");
    let scene = Scene::preset("lih").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let mut st = UiState::new("lih", &scene);
    st.scene_path = path.display().to_string();
    let out = act(&mut st, &inp, Action::SaveScene);
    assert!(out.error.is_none() && out.commands.is_empty());
    assert!(st.status.starts_with("saved 2 nuclei to"), "{}", st.status);
    let out = act(&mut st, &inp, Action::LoadScene);
    assert!(matches!(out.commands.as_slice(), [Command::SetScene(s)] if *s == scene), "{:?}", out.commands);
    let out = act(&mut st, &inp, Action::LoadModel);
    assert!(out.error.as_deref().is_some_and(|e| e.starts_with("no model file")));
    // estimated with a sidecar: the scene and its model
    std::fs::write(dir.join("t.model"), b"x").unwrap();
    st.backend = Backend::Estimated;
    let out = act(&mut st, &inp, Action::LoadScene);
    assert!(matches!(out.commands.as_slice(), [Command::SetScene(_), Command::Dyn(DynCommand::LoadModel(_))]), "{:?}", out.commands);
    assert!(matches!(one_dyn(&act(&mut st, &inp, Action::LoadModel)), DynCommand::LoadModel(_)));
    assert!(matches!(one_dyn(&act(&mut st, &inp, Action::SaveModel)), DynCommand::SaveModel(p) if p.ends_with("t.model")));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Dock ▸ Model and File ▸ Estimated model: Forget needs two clicks; a long pass needs two
/// clicks ("Start anyway"), a short one starts at once and advances the seed; Cancel, the offer's
/// Start anyway, "Forget it and sample" and the audit cadence send their old commands.
#[test]
fn model_actions_keep_their_two_click_confirms() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let mut st = UiState::new("h2", &scene);
    assert!(act(&mut st, &inp, Action::Forget).commands.is_empty(), "the first click arms");
    assert!(matches!(one_dyn(&act(&mut st, &inp, Action::Forget)), DynCommand::ForgetModel));
    assert!(st.forget_armed.is_none());
    let seed = st.plan.seed;
    assert!(act(&mut st, &inp, Action::Sample { long: true }).commands.is_empty(), "a long pass arms first");
    assert!(matches!(one_dyn(&act(&mut st, &inp, Action::Sample { long: true })), DynCommand::StartSampling(p) if p.seed == seed));
    assert_eq!(st.plan.seed, seed + 1);
    assert!(matches!(one_dyn(&act(&mut st, &inp, Action::Sample { long: false })), DynCommand::StartSampling(_)));
    assert!(matches!(one_dyn(&act(&mut st, &inp, Action::CancelSampling)), DynCommand::CancelSampling));
    let plan = sim::SamplingPlan { perturbations: 7, ..Default::default() };
    assert!(matches!(one_dyn(&act(&mut st, &inp, Action::StartOffered(plan.clone()))), DynCommand::StartSampling(p) if *p == plan));
    assert!(matches!(one_dyn(&act(&mut st, &inp, Action::SampleAfterRefusal)), DynCommand::StartSampling(_)));
    let out = act(&mut st, &inp, Action::SetAudit(0));
    assert!(matches!(out.commands.as_slice(), [Command::SetAudit(0)]));
    assert_eq!(st.audit_every, 0);
}

/// The sampling button's label follows the model ("Start sampling" → "Sample more") and the
/// armed long pass ("Start anyway (≈ t)").
#[test]
fn the_sampling_button_names_its_state() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let mut st = UiState::new("h2", &scene);
    let mut m = model();
    m.fitted = false;
    m.samples = 0;
    let snap = snapshot(&scene, Backend::Estimated, Some(m));
    let inp = input(&scene, Some(&snap), &cam);
    let b = sample_button(&st, &inp);
    assert_eq!(b.label, "Start sampling");
    assert!(b.cost.is_some_and(|c| c.contains("solves")));
    assert!(b.long && !b.armed, "the default plan at 1 s a solve is a long pass");
    let snap = snapshot(&scene, Backend::Estimated, Some(model()));
    let inp = input(&scene, Some(&snap), &cam);
    assert_eq!(sample_button(&st, &inp).label, "Sample more");
    st.start_armed = Some(std::time::Instant::now());
    let b = sample_button(&st, &inp);
    assert!(b.armed && b.label.starts_with("Start anyway"), "{}", b.label);
}

/// View ▸ Density and the quick switch in the view.
#[test]
fn the_quick_view_switch_sets_the_volume_mode() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let mut st = UiState::new("h2", &scene);
    act(&mut st, &inp, Action::VolumeMode(1));
    assert_eq!(st.volume_mode, 1);
}

/// Ctrl K: fuzzy matches by name first; every control the mapping names has an entry.
#[test]
fn the_control_search_finds_controls_by_fuzzy_name() {
    assert_eq!(search::results("therm", 3)[0].name, "Thermostat (Bussi CSVR)");
    assert_eq!(search::results("forget", 1)[0].name, "Forget model");
    assert_eq!(search::results("apply grid", 1)[0].name, "Apply grid");
    assert!(search::results("zzzzqq", 5).is_empty());
    assert_eq!(search::results("", 1000).len(), search::ENTRIES.len());
    for e in search::ENTRIES {
        assert!(!e.anchor.is_empty() && !e.name.is_empty());
    }
}

/// The log records a status line once per `set_status` (a repeat of the same text is a new
/// message: fresh toast), counts an error as unread, logs new dynamics notes by `notes_total`, the
/// box warning once per nucleus, each evaluation note once per run, and a repeated refusal of the
/// same command again.
#[test]
fn the_log_records_what_is_new() {
    let mut scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let mut st = UiState::new("h2", &scene);
    let inp = input(&scene, None, &cam);
    st.set_status("saved 2 nuclei to my.scene");
    st.log.record(&st.status, st.status_seq, &inp);
    st.log.record(&st.status, st.status_seq, &inp);
    assert_eq!(st.log.count(LogKind::Status), 1, "one frame after another: one message");
    st.set_status("saved 2 nuclei to my.scene");
    st.log.record(&st.status, st.status_seq, &inp);
    assert_eq!(st.log.count(LogKind::Status), 2, "Save twice: confirmed twice");
    assert!(st.log.current_toast().is_some());
    let inp = PanelInput { last_error: Some("bad grid"), ..input(&scene, None, &cam) };
    st.log.record("", 0, &inp);
    assert_eq!((st.log.count(LogKind::Error), st.log.unread), (1, 1));
    scene.nuclei[1].pos = [0.0, -5.0, 0.0];
    let mut snap = snapshot(&scene, Backend::Gpu, None);
    snap.dynamics.eval.notes = vec!["start density extrapolated".into()];
    let inp = input(&scene, Some(&snap), &cam);
    st.log.record("", 0, &inp);
    st.log.record("", 0, &inp);
    assert_eq!(st.log.count(LogKind::Note), 2, "a dynamics note and the evaluation note, once each: {:?}", st.log.entries);
    assert_eq!(st.log.count(LogKind::Warning), 1, "the box warning once");
    // the same evaluation note on later steps of the run is not logged again, even past the
    // dedupe window
    for k in 0..30 {
        snap.dynamics.step += 1;
        snap.dynamics.eval.notes = vec![format!("other note {k}"), "start density extrapolated".into()];
        st.log.record("", 0, &input(&scene, Some(&snap), &cam));
    }
    assert_eq!(st.log.entries.iter().filter(|e| e.text == "start density extrapolated").count(), 1);
    // a refusal, then the same refusal again after another (refused) Play: two entries
    snap.dynamics.last_rejection = Some("Play while a sampling pass runs".into());
    st.log.record("", 0, &input(&scene, Some(&snap), &cam));
    st.log.record("", 0, &input(&scene, Some(&snap), &cam));
    assert_eq!(st.log.count(LogKind::Refused), 1);
    snap.driver_commands += 1;
    st.log.record("", 0, &input(&scene, Some(&snap), &cam));
    assert_eq!(st.log.count(LogKind::Refused), 2, "a repeated refusal is new feedback");
    // the same for a refused non-edit command typed again (SetDt / SetSpeed / SetWalls): the
    // driver-command count moves, the mirrored-edit count does not
    st.log.toast = None;
    snap.dynamics.last_rejection = Some("dt must be finite and > 0".into());
    st.log.record("", 0, &input(&scene, Some(&snap), &cam));
    assert!(matches!(st.log.current_toast(), Some((LogKind::Refused, t)) if t.contains("dt must be")));
    st.log.toast = None;
    snap.driver_commands += 1;
    st.log.record("", 0, &input(&scene, Some(&snap), &cam));
    assert!(st.log.current_toast().is_some(), "the repeat re-raises the toast");
    assert_eq!(st.log.count(LogKind::Refused), 4);
    assert!(matches!(st.log.current_toast(), Some((LogKind::Refused, _))));
}

/// The status chip keeps the pause notice (amber: it asks for Play) and the driver's refusal in
/// place until they are answered — the mockup's "the status chip says so".
#[test]
fn the_status_chip_keeps_the_pause_notice_and_the_refusal() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let mut st = UiState::new("h2", &scene);
    let mut snap = snapshot(&scene, Backend::Gpu, None);
    assert!(overlays::chip_notes(&st, &input(&scene, Some(&snap), &cam)).is_empty());
    st.set_status("paused: this edit replaces the scene (time, pins and the drift history restart) — press Play to continue");
    snap.dynamics.last_rejection = Some("Step while a sampling pass runs".into());
    let notes = overlays::chip_notes(&st, &input(&scene, Some(&snap), &cam));
    assert_eq!(notes.len(), 2);
    assert!(notes.iter().all(|(c, _)| *c == AMBER), "{notes:?}");
    assert!(notes[0].1.starts_with("paused: this edit") && notes[1].1 == "refused: Step while a sampling pass runs");
    st.status.clear(); // Play answers it (Viewer::send)
    snap.dynamics.last_rejection = None; // the next accepted command clears it
    assert!(overlays::chip_notes(&st, &input(&scene, Some(&snap), &cam)).is_empty());
}

// ---- headless smoke runs of the whole chrome ----

fn frame(ctx: &egui::Context, st: &mut UiState, inp: &PanelInput<'_>, w: f32, events: Vec<egui::Event>) -> PanelOutput {
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w, 900.0))),
        events,
        ..Default::default()
    };
    let mut out = None;
    let mut full = ctx.run_ui(raw, |root| out = Some(show(root, st, inp)));
    // the window hands these to the painter; dropping them unapplied trips epaint's debug assertion
    full.textures_delta.clear();
    out.unwrap()
}

/// Every layout (wide and compact; dock open on each tab and collapsed; every menu open; the
/// search with a query; a running pass with its card; the first-pass offer; a selection with
/// its card) lays out without a panic and, with no input, sends nothing.
#[test]
fn every_layout_draws_and_sends_nothing_without_input() {
    let scene = Scene::preset("h2o").unwrap();
    let cam = camera();
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2o", &scene);
    st.gpu_available = true;
    st.backend = Backend::Estimated;
    search::MARKED.with(|m| m.borrow_mut().clear());
    let mut m = model();
    m.pass_offer = Some(PassOffer {
        plan: sim::SamplingPlan::default(),
        solves: 365,
        eta_s: 366.0,
        per_solve_s: 1.0,
        full_solves: 512,
        full_eta_s: 512.0,
    });
    let mut sampling = snapshot(&scene, Backend::Estimated, Some(model()));
    sampling.dynamics.sampling = Some(sim::SamplingProgress {
        done: 112,
        total: 273,
        stage: "perturbations",
        last_energy: -76.4,
        cancelled: false,
        skipped: 2,
        failed: 0,
        finished: false,
    });
    let snaps = [
        None,
        Some(snapshot(&scene, Backend::Gpu, None)),
        Some(snapshot(&scene, Backend::Estimated, Some(m))),
        Some(sampling),
    ];
    for snap in &snaps {
        for (w, sel) in [(1400.0, None), (1000.0, Some(0)), (1400.0, Some(2))] {
            let inp = PanelInput {
                selected: sel,
                playing: sel.is_some(),
                density_label: Some("density from the last DFT step".into()),
                last_error: sel.map(|_| "an error"),
                ..input(&scene, snap.as_ref(), &cam)
            };
            for tab in DockTab::ALL {
                st.dock_tab = tab;
                for _ in 0..2 {
                    let out = frame(&ctx, &mut st, &inp, w, vec![]);
                    assert!(out.commands.is_empty() && !out.toggle_play && !out.deselect && out.error.is_none(), "{tab:?} at {w}");
                }
            }
            st.dock_open = false;
            frame(&ctx, &mut st, &inp, w, vec![]);
            st.dock_open = true;
            for mid in menus::MenuId::ALL {
                egui::Popup::open_id(&ctx, menus::popup_id(mid));
                for _ in 0..2 {
                    assert!(frame(&ctx, &mut st, &inp, w, vec![]).commands.is_empty());
                }
            }
            egui::Popup::close_all(&ctx);
            st.search.toggle();
            st.search.query = "therm".into();
            frame(&ctx, &mut st, &inp, w, vec![]);
            st.search.open = false;
        }
    }
    // every search entry's anchor is marked by a widget group in some layout (a misspelt anchor
    // would silently stop revealing)
    let marked = search::MARKED.with(|m| m.borrow().clone());
    for e in search::ENTRIES {
        assert!(marked.contains(e.anchor), "search entry {:?}: anchor {:?} is never marked", e.name, e.anchor);
    }
}

fn key(key: egui::Key, modifiers: egui::Modifiers) -> egui::Event {
    egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers }
}

/// Ctrl K opens the search, Enter reveals the best match (its dock tab opens), Esc closes the
/// search; Esc with nothing open deselects.
#[test]
fn ctrl_k_reveals_and_esc_deselects() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let ctx = egui::Context::default();
    let mut st = UiState::new("h2", &scene);
    let inp = PanelInput { selected: Some(0), ..input(&scene, None, &cam) };
    frame(&ctx, &mut st, &inp, 1400.0, vec![key(egui::Key::K, egui::Modifiers::COMMAND)]);
    assert!(st.search.open);
    st.search.query = "distance guides".into();
    st.dock_tab = DockTab::Log;
    frame(&ctx, &mut st, &inp, 1400.0, vec![key(egui::Key::Enter, egui::Modifiers::NONE)]);
    assert!(!st.search.open);
    assert_eq!(st.dock_tab, DockTab::Selection);
    assert_eq!(st.search.revealing(), Some("d-guides"));
    frame(&ctx, &mut st, &inp, 1400.0, vec![]);
    let out = frame(&ctx, &mut st, &inp, 1400.0, vec![key(egui::Key::Escape, egui::Modifiers::NONE)]);
    assert!(out.deselect);
}

/// The driver's refusal stays in place in the Log tab (the `last refusal` line) for as long as the
/// driver holds it, besides the 6 s toast and the log entry.
#[test]
fn the_last_refusal_stays_in_the_log_tab() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let mut snap = snapshot(&scene, Backend::Gpu, None);
    assert_eq!(dock::last_refusal(&input(&scene, Some(&snap), &cam)), None);
    snap.dynamics.last_rejection = Some("Play while a sampling pass runs".into());
    let inp = input(&scene, Some(&snap), &cam);
    assert_eq!(dock::last_refusal(&inp).as_deref(), Some("last refusal: refused: Play while a sampling pass runs"));
    let mut st = UiState::new("h2", &scene);
    st.log.record("", 0, &inp);
    assert_eq!(st.log.count(LogKind::Refused), 1);
    assert!(st.log.current_toast().is_some());
}

// ---- the real widgets, clicked: pointer events through egui into the chrome ----

/// The rect `name`'s widget had in the last frame (`ui::probe`).
fn probed_rect(name: &str) -> egui::Rect {
    PROBES.with(|p| p.borrow().get(name).copied()).unwrap_or_else(|| panic!("no widget probed as {name:?}"))
}

fn pointer(pos: egui::Pos2, pressed: bool) -> egui::Event {
    egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE }
}

/// Lays the chrome out (a few frames: popups size themselves first), then presses and releases the
/// primary button on the centre of widget `name`, moving by `drag` (points, in steps) in between.
/// Returns every frame's output.
fn press_widget(ctx: &egui::Context, st: &mut UiState, inp: &PanelInput<'_>, name: &str, drag: f32) -> Vec<PanelOutput> {
    for _ in 0..3 {
        frame(ctx, st, inp, 1400.0, vec![]);
    }
    let c = probed_rect(name).center();
    let mut outs = vec![frame(ctx, st, inp, 1400.0, vec![egui::Event::PointerMoved(c), pointer(c, true)])];
    let mut at = c;
    for _ in 0..4 {
        if drag != 0.0 {
            at.x += drag / 4.0;
            outs.push(frame(ctx, st, inp, 1400.0, vec![egui::Event::PointerMoved(at)]));
        }
    }
    outs.push(frame(ctx, st, inp, 1400.0, vec![pointer(at, false)]));
    outs.push(frame(ctx, st, inp, 1400.0, vec![egui::Event::PointerGone]));
    outs
}

fn click_widget(ctx: &egui::Context, st: &mut UiState, inp: &PanelInput<'_>, name: &str) -> Vec<PanelOutput> {
    press_widget(ctx, st, inp, name, 0.0)
}

fn dyn_commands(outs: &[PanelOutput]) -> Vec<DynCommand> {
    outs.iter()
        .flat_map(|o| &o.commands)
        .filter_map(|c| match c {
            Command::Dyn(d) => Some(d.clone()),
            _ => None,
        })
        .collect()
}

/// Toolbar: Play, the watchable preset, a drag on the speed field (SetSpeed only, never dt —
/// rule 8), a drag on the dt field (SetDt only), an element chip (→ Place), the solver segments
/// and Delete, each clicked as a user would.
#[test]
fn clicking_the_toolbar_sends_the_old_commands() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let snap = snapshot(&scene, Backend::Gpu, None);
    let inp = PanelInput { selected: Some(1), ..input(&scene, Some(&snap), &cam) };
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    st.gpu_available = true;
    let outs = click_widget(&ctx, &mut st, &inp, "play");
    assert_eq!(outs.iter().filter(|o| o.toggle_play).count(), 1, "Play clicked once");
    st.speed = 50;
    let c = dyn_commands(&click_widget(&ctx, &mut st, &inp, "watchable"));
    assert!(matches!(c.as_slice(), [DynCommand::SetSpeed(1)]), "{c:?}");
    let dt = st.dt_fs;
    let c = dyn_commands(&press_widget(&ctx, &mut st, &inp, "speed", 60.0));
    assert!(!c.is_empty() && c.iter().all(|d| matches!(d, DynCommand::SetSpeed(n) if *n > 1)), "{c:?}");
    assert!(st.speed > 1 && st.dt_fs == dt, "speed {} dt {}", st.speed, st.dt_fs);
    let speed = st.speed;
    let c = dyn_commands(&press_widget(&ctx, &mut st, &inp, "dt", 60.0));
    assert!(!c.is_empty() && c.iter().all(|d| matches!(d, DynCommand::SetDt(x) if *x > 0.25)), "{c:?}");
    assert_eq!(st.speed, speed);
    let outs = click_widget(&ctx, &mut st, &inp, "element-8");
    assert!(outs.iter().all(|o| o.commands.is_empty()));
    assert_eq!((st.element, st.mode), (8, ToolMode::Place));
    click_widget(&ctx, &mut st, &inp, "solver-estimated");
    assert_eq!(st.backend, Backend::Estimated);
    click_widget(&ctx, &mut st, &inp, "solver-cpu");
    assert_eq!(st.backend, Backend::Cpu);
    click_widget(&ctx, &mut st, &inp, "solver-gpu");
    assert_eq!(st.backend, Backend::Gpu);
    let outs = click_widget(&ctx, &mut st, &inp, "delete");
    let removes: Vec<_> = outs.iter().flat_map(|o| &o.commands).filter(|c| matches!(c, Command::RemoveNucleus(1))).collect();
    assert_eq!(removes.len(), 1, "Delete removes the selected nucleus once");
}

/// Menus: the Physics menu's thermostat checkbox, gravity "g" and "off", COM correction; the
/// Scene menu's Apply grid — clicked inside the open popup.
#[test]
fn clicking_menu_controls_sends_the_old_commands() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let snap = snapshot(&scene, Backend::Gpu, None);
    let inp = input(&scene, Some(&snap), &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    let in_menu = |st: &mut UiState, m: menus::MenuId, name: &str| {
        egui::Popup::open_id(&ctx, menus::popup_id(m));
        let outs = click_widget(&ctx, st, &inp, name);
        egui::Popup::close_all(&ctx);
        outs
    };
    let c = dyn_commands(&in_menu(&mut st, menus::MenuId::Physics, "thermostat"));
    assert!(matches!(c.as_slice(), [DynCommand::SetThermostat(Some((t, tau)))] if *t == 300.0 && *tau == 10.0), "{c:?}");
    assert!(st.thermostat_on);
    let c = dyn_commands(&in_menu(&mut st, menus::MenuId::Physics, "gravity-g"));
    assert!(matches!(c.as_slice(), [DynCommand::SetExternalField(a)] if a[1] == -gravity_au()), "{c:?}");
    let c = dyn_commands(&in_menu(&mut st, menus::MenuId::Physics, "gravity-off"));
    assert!(matches!(c.as_slice(), [DynCommand::SetExternalField(a)] if *a == [0.0; 3]), "{c:?}");
    let c = dyn_commands(&in_menu(&mut st, menus::MenuId::Physics, "com"));
    assert!(matches!(c.as_slice(), [DynCommand::SetComCorrection(false)]), "{c:?}");
    st.grid_n = 41;
    let outs = in_menu(&mut st, menus::MenuId::Scene, "apply-grid");
    let grids: Vec<_> = outs.iter().flat_map(|o| &o.commands).filter(|c| matches!(c, Command::SetGrid { n: 41, .. })).collect();
    assert_eq!(grids.len(), 1, "Apply grid sends the pending grid once");
}

/// Dock ▸ Model: Forget model needs two clicks; a long pass needs two clicks on the sampling button.
#[test]
fn clicking_the_model_tab_keeps_the_two_click_confirms() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let snap = snapshot(&scene, Backend::Estimated, Some(model()));
    let inp = input(&scene, Some(&snap), &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    st.backend = Backend::Estimated;
    st.dock_tab = DockTab::Model;
    assert!(dyn_commands(&click_widget(&ctx, &mut st, &inp, "forget")).is_empty(), "the first click arms");
    let c = dyn_commands(&click_widget(&ctx, &mut st, &inp, "forget"));
    assert!(matches!(c.as_slice(), [DynCommand::ForgetModel]), "{c:?}");
    let seed = st.plan.seed;
    assert!(dyn_commands(&click_widget(&ctx, &mut st, &inp, "sample")).is_empty(), "a long pass arms first");
    let c = dyn_commands(&click_widget(&ctx, &mut st, &inp, "sample"));
    assert!(matches!(c.as_slice(), [DynCommand::StartSampling(p)] if p.seed == seed), "{c:?}");
    assert_eq!(st.plan.seed, seed + 1);
}

/// The M5b controls in their new homes, clicked: Scene ▸ Container's walls checkbox, View ▸ the
/// cloud refresh, Physics ▸ draft mode (estimated solver).
#[test]
fn clicking_the_m5b_controls_sends_their_commands() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let snap = snapshot(&scene, Backend::Estimated, Some(model()));
    let inp = input(&scene, Some(&snap), &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    st.backend = Backend::Estimated;
    let in_menu = |st: &mut UiState, m: menus::MenuId, name: &str| {
        egui::Popup::open_id(&ctx, menus::popup_id(m));
        let outs = click_widget(&ctx, st, &inp, name);
        egui::Popup::close_all(&ctx);
        outs
    };
    let c = dyn_commands(&in_menu(&mut st, menus::MenuId::Scene, "walls"));
    assert!(matches!(c.as_slice(), [DynCommand::SetWalls(Some(w))] if w.half_width == 4.5), "{c:?}");
    let outs = in_menu(&mut st, menus::MenuId::View, "cloud-refresh");
    let clouds: Vec<_> = outs.iter().flat_map(|o| &o.commands).filter(|c| matches!(c, Command::SetCloud { superposition: true, refresh_s: Some(x) } if *x == 2.0)).collect();
    assert_eq!(clouds.len(), 1);
    let outs = in_menu(&mut st, menus::MenuId::Physics, "draft");
    let drafts: Vec<_> = outs.iter().flat_map(|o| &o.commands).filter(|c| matches!(c, Command::SetDraft(true))).collect();
    assert_eq!(drafts.len(), 1);
    assert!(st.draft);
}
