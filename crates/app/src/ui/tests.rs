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

/// A new worker gets the panel's run settings; defaults send speed, COM correction and the
/// container walls (on by default since 2026-10-14) — only speed and COM with the walls off.
#[test]
fn run_settings_mirror_the_panel() {
    let mut st = UiState::new("h2", &Scene::preset("h2").unwrap());
    assert!(st.com_correction, "COM correction defaults on in the app");
    let v = st.run_settings();
    assert_eq!(v.len(), 3);
    assert!(v.iter().any(|c| matches!(c, DynCommand::SetWalls(Some(w)) if w.half_width == 4.5)), "{v:?}");
    st.walls_on = false;
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

/// Container walls are on by default in the app (user request 2026-10-14) for every preset, the
/// empty one included, at the slider's default: 1.5 Bohr inside the cell's face (4.5 Bohr at
/// 49³ × 0.25, 5.5 at h2o's 57³), within the allowed range — and that is what a fresh worker is
/// sent.
#[test]
fn fresh_panels_start_with_walls_on_at_the_default_half_width() {
    for name in Scene::PRESETS {
        let s = Scene::preset(name).unwrap();
        let st = UiState::new(name, &s);
        let half = (s.grid_n as f64 - 1.0) * s.grid_h / 2.0;
        assert!(st.walls_on, "{name}");
        assert_eq!(st.wall_half_width, default_wall_half_width(&s), "{name}");
        assert!((st.wall_half_width - (half - 1.5)).abs() < 1e-12 && st.wall_half_width <= max_wall_half_width(&s), "{name}: {}", st.wall_half_width);
        let v = st.run_settings();
        assert!(v.iter().any(|c| matches!(c, DynCommand::SetWalls(Some(w)) if w.half_width == st.wall_half_width)), "{name}: {v:?}");
    }
    assert_eq!(UiState::new("empty", &Scene::preset("empty").unwrap()).wall_half_width, 4.5);
    assert_eq!(UiState::new("h2o", &Scene::preset("h2o").unwrap()).wall_half_width, 5.5);
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
        bonds_label: None,
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
        provenance: Some("49³ h 0.25 σ 0.25, charge 0, multiplicity lowest, teacher 'GpuKs' at tol 1e-6, masses ¹H (recorded, not compared)".into()),
        last_refit: None,
        refits: 0,
        autoload_refused: None,
        audit_every: 2000,
        draft: false,
        last_audit_step: Some(1600),
        last_gap: Some((-5.3e-5, 1600)),
        quarantined: 2,
        recovery: None,
        superseded: 0,
        box_excluded: 0,
        unfitted: 0,
        fit_vs_held: Some(crate::sim::fit_vs_held(271, 273, 0, 2, 0, 0)),
        audits_background: true,
        audit_in_flight: true,
        last_audit: Some(sim::AuditRecord {
            gap: -5.3e-5,
            verdict: sim::AuditVerdict::Passed,
            face_density: None,
            unaudited_steps: 18400,
            asynchronous: true,
            teacher_ms: 274.0,
        }),
        stuck: None,
        class_loo: vec![sim::ClassLoo { class: "pair", samples: 273, force_rms: 0.009, force_max: 0.1, energy_max: 0.152, trusted: true }],
        solve_s: Some(1.0),
        solve_s_measured: false,
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
        commands: 0,
        working: false,
        labels: Default::default(),
        charge_note: None,
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

/// File ▸ Presets and the scene chip: a preset (or Reset) sends SetScene at the panel's quality
/// (Accurate by default: h2o's 14 Bohr box at h = 0.15 is 95³) and adopts its grid and box.
#[test]
fn presets_replace_the_scene_and_adopt_its_grid() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let mut st = UiState::new("h2", &scene);
    st.grid_n = 33;
    let out = act(&mut st, &inp, Action::Preset("h2o".into()));
    let want = crate::quality::preset("h2o", crate::quality::DEFAULT_H).unwrap();
    assert_eq!((want.grid_n, want.grid_h), (95, 0.15));
    // the scene, then the container moved to the new cell's default (walls on by default)
    assert!(matches!(out.commands.as_slice(), [Command::SetScene(s), Command::Dyn(DynCommand::SetWalls(Some(w)))] if *s == want && (w.half_width - 5.55).abs() < 1e-12), "{:?}", out.commands);
    assert_eq!((st.preset.as_str(), st.grid_n, st.box_len), ("h2o", 95, 14.0));
    assert!((st.wall_half_width - 5.55).abs() < 1e-12);
    // at another quality the preset follows it
    st.quality_h = 0.30;
    let out = act(&mut st, &inp, Action::Preset("h2".into()));
    assert!(matches!(out.commands.first(), Some(Command::SetScene(s)) if (s.grid_n, s.grid_h) == (41, 0.30)), "{:?}", out.commands);
    // walls off: the scene alone
    st.walls_on = false;
    let out = act(&mut st, &inp, Action::Preset("he".into()));
    assert!(matches!(out.commands.as_slice(), [Command::SetScene(_)]), "{:?}", out.commands);
    let out = act(&mut st, &inp, Action::Preset("nope".into()));
    assert!(out.commands.is_empty() && st.preset == "he");
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
    // the scene, then the container at the loaded cell's default (walls on by default)
    assert!(matches!(out.commands.as_slice(), [Command::SetScene(s), Command::Dyn(DynCommand::SetWalls(Some(_)))] if *s == scene), "{:?}", out.commands);
    let out = act(&mut st, &inp, Action::LoadModel);
    assert!(out.error.as_deref().is_some_and(|e| e.starts_with("no model file")));
    // estimated with a sidecar: the scene and its model
    std::fs::write(dir.join("t.model"), b"x").unwrap();
    st.backend = Backend::Estimated;
    let out = act(&mut st, &inp, Action::LoadScene);
    assert!(matches!(out.commands.as_slice(), [Command::SetScene(_), Command::Dyn(DynCommand::SetWalls(_)), Command::Dyn(DynCommand::LoadModel(_))]), "{:?}", out.commands);
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
    // the estimate's basis (M6 Task 9, ported from the old panel): the first solve until a pass
    // has measured its own rate
    assert!(b.label.ends_with(" at the first solve)"), "{}", b.label);
    let mut measured = model();
    measured.solve_s_measured = true;
    let snap = snapshot(&scene, Backend::Estimated, Some(measured));
    let b = sample_button(&st, &input(&scene, Some(&snap), &cam));
    assert!(b.label.ends_with(" at the last pass's rate)"), "{}", b.label);
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

/// M6 Task 3's bond-reference status line, ported from the old panel's status line into the
/// status chip (merge of main's UI redesign): dim, background work, gone when the job ends.
#[test]
fn the_status_chip_shows_the_bond_references_in_the_background() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let st = UiState::new("h2", &scene);
    let mut snap = snapshot(&scene, Backend::Gpu, None);
    snap.dynamics.bond_reference = Some(sim::ReferenceProgress { z: 8, done: 1, total: 3, elapsed_s: 14.2 });
    let notes = overlays::chip_notes(&st, &input(&scene, Some(&snap), &cam));
    assert_eq!(notes, vec![(theme::INK_DIM, "bond references: O 2/3… (14 s)".to_string())]);
    snap.dynamics.bond_reference = Some(sim::ReferenceProgress { z: 1, done: 3, total: 3, elapsed_s: 2.0 });
    assert_eq!(overlays::chip_notes(&st, &input(&scene, Some(&snap), &cam))[0].1, "bond references: H 3/3… (2 s)");
    snap.dynamics.bond_reference = None;
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
    // walls are on by default since 2026-10-14: start from off (`--no-walls`) so the click turns
    // them on, then click again for off
    st.walls_on = false;
    let in_menu = |st: &mut UiState, m: menus::MenuId, name: &str| {
        egui::Popup::open_id(&ctx, menus::popup_id(m));
        let outs = click_widget(&ctx, st, &inp, name);
        egui::Popup::close_all(&ctx);
        outs
    };
    let c = dyn_commands(&in_menu(&mut st, menus::MenuId::Scene, "walls"));
    assert!(matches!(c.as_slice(), [DynCommand::SetWalls(Some(w))] if w.half_width == 4.5), "{c:?}");
    let c = dyn_commands(&in_menu(&mut st, menus::MenuId::Scene, "walls"));
    assert!(matches!(c.as_slice(), [DynCommand::SetWalls(None)]), "{c:?}");
    let outs = in_menu(&mut st, menus::MenuId::View, "cloud-refresh");
    let clouds: Vec<_> = outs.iter().flat_map(|o| &o.commands).filter(|c| matches!(c, Command::SetCloud { superposition: true, refresh_s: Some(x) } if *x == 2.0)).collect();
    assert_eq!(clouds.len(), 1);
    let outs = in_menu(&mut st, menus::MenuId::Physics, "draft");
    let drafts: Vec<_> = outs.iter().flat_map(|o| &o.commands).filter(|c| matches!(c, Command::SetDraft(true))).collect();
    assert_eq!(drafts.len(), 1);
    assert!(st.draft);
}

// ---- the drag pull (user request 2026-10-14: the wheel while dragging) ----

/// The wheel's action scales the pull geometrically (× 1.25 per line up, ÷ per line down),
/// clamps it to the driver's 0.01–1 Ha/Bohr and sends `SetDragForce` only when it changes; the
/// Physics menu's slider value is clamped the same way and Reset goes back to 0.1; a new worker
/// gets a non-default pull with the run settings.
#[test]
fn the_drag_pull_actions_scale_clamp_and_send() {
    use sim::dynamics::{F_DRAG_HI, F_DRAG_LO, F_DRAG_MAX};
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let mut st = UiState::new("h2", &scene);
    assert_eq!(st.drag_force, F_DRAG_MAX);
    assert!(!st.run_settings().iter().any(|c| matches!(c, DynCommand::SetDragForce { .. })), "the default is not re-sent");
    let pull = |out: &PanelOutput| match one_dyn(out) {
        DynCommand::SetDragForce { f_max } => *f_max,
        c => panic!("{c:?}"),
    };
    let f = pull(&act(&mut st, &inp, Action::DragForceWheel(1.0)));
    assert!((f - 0.125).abs() < 1e-15 && st.drag_force == f, "{f}");
    let f = pull(&act(&mut st, &inp, Action::DragForceWheel(-2.0)));
    assert!((f - 0.08).abs() < 1e-15, "{f}");
    // a pixel wheel's fraction of a line scales by the fraction
    let f = pull(&act(&mut st, &inp, Action::DragForceWheel(0.5)));
    assert!((f - 0.08 * 1.25f64.sqrt()).abs() < 1e-15, "{f}");
    assert_eq!(pull(&act(&mut st, &inp, Action::DragForceWheel(40.0))), F_DRAG_HI);
    assert!(act(&mut st, &inp, Action::DragForceWheel(1.0)).commands.is_empty(), "at the ceiling: nothing to send");
    assert!(st.run_settings().iter().any(|c| matches!(c, DynCommand::SetDragForce { f_max } if *f_max == F_DRAG_HI)));
    assert_eq!(pull(&act(&mut st, &inp, Action::DragForceWheel(-40.0))), F_DRAG_LO);
    assert_eq!(pull(&act(&mut st, &inp, Action::DragForce(F_DRAG_MAX))), F_DRAG_MAX);
    assert_eq!(pull(&act(&mut st, &inp, Action::DragForce(5.0))), F_DRAG_HI);
    assert!(act(&mut st, &inp, Action::DragForce(f64::NAN)).commands.is_empty() && st.drag_force == F_DRAG_HI);
    // the label converts at the UI edge (units): 0.16 Ha/Bohr = 8.2 eV/Å
    assert_eq!(pull_label(0.16), "pull 0.160 Ha/Bohr · 8.23 eV/Å");
    // three significant figures keep the wheel's steps distinct at the bottom of the range (M5)
    assert_eq!([0.01, 0.0125, 0.015625, 1.0].map(sig3), ["0.0100", "0.0125", "0.0156", "1.00"]);
}

/// A queued action (the 3D view's wheel during a drag, `Viewer::wheel`) is applied with the next
/// frame's chrome actions: the frame sends the `SetDragForce` and empties the queue.
#[test]
fn a_queued_wheel_action_is_sent_with_the_next_frame() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let ctx = egui::Context::default();
    let mut st = UiState::new("h2", &scene);
    st.queued.push(Action::DragForceWheel(1.0));
    st.queued.push(Action::DragForceWheel(1.0));
    let out = frame(&ctx, &mut st, &inp, 1400.0, vec![]);
    let c = dyn_commands(std::slice::from_ref(&out));
    assert!(matches!(c.as_slice(), [DynCommand::SetDragForce { f_max: a }, DynCommand::SetDragForce { f_max: b }] if (*a - 0.125).abs() < 1e-15 && (*b - 0.15625).abs() < 1e-15), "{c:?}");
    assert!(st.queued.is_empty());
    assert!(frame(&ctx, &mut st, &inp, 1400.0, vec![]).commands.is_empty());
}

/// Physics ▸ Drag pull, clicked: Reset sends the default pull; Ctrl K finds it as "pull" and as
/// "drag force".
#[test]
fn clicking_the_drag_pull_reset_and_finding_it() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let snap = snapshot(&scene, Backend::Gpu, None);
    let inp = input(&scene, Some(&snap), &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    st.drag_force = 0.5;
    egui::Popup::open_id(&ctx, menus::popup_id(menus::MenuId::Physics));
    let c = dyn_commands(&click_widget(&ctx, &mut st, &inp, "drag-reset"));
    egui::Popup::close_all(&ctx);
    assert!(matches!(c.as_slice(), [DynCommand::SetDragForce { f_max }] if *f_max == sim::dynamics::F_DRAG_MAX), "{c:?}");
    assert_eq!(st.drag_force, sim::dynamics::F_DRAG_MAX);
    for q in ["pull", "drag force"] {
        let r = search::results(q, 3);
        assert!(r.first().is_some_and(|e| e.anchor == "m-drag"), "{q}: {:?}", r.iter().map(|e| e.name).collect::<Vec<_>>());
    }
}

// ---- the empty scene (the `empty` preset) ----

/// An empty-scene snapshot as the driver publishes it: no nuclei, no history, no drift, T = 0.
fn empty_snapshot(backend: Backend, model: Option<ModelStatus>) -> Snapshot {
    let scene = Scene::preset("empty").unwrap();
    let mut s = snapshot(&scene, backend, model);
    s.dynamics.history = std::sync::Arc::from(Vec::new());
    s.dynamics.history_source = std::sync::Arc::from(Vec::new());
    s.dynamics.drift_per_ps = None;
    s.dynamics.temperature = 0.0;
    s
}

/// The scene chip's Presets popup lists `empty` first, and clicking it sends the same
/// `SetScene` (→ the worker's `ReplaceScene`) as every other preset, adopting its grid.
#[test]
fn the_presets_popup_lists_empty_first_and_sends_its_scene() {
    assert_eq!(Scene::PRESETS, &["empty", "h", "h2", "he", "lih", "h2o"]);
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let snap = snapshot(&scene, Backend::Gpu, None);
    let inp = input(&scene, Some(&snap), &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    st.grid_n = 33;
    egui::Popup::open_id(&ctx, egui::Id::new("scene-chip"));
    let outs = click_widget(&ctx, &mut st, &inp, "empty");
    egui::Popup::close_all(&ctx);
    // every preset label was drawn, in order, left to right
    let xs: Vec<f32> = Scene::PRESETS.iter().map(|n| probed_rect(n).left()).collect();
    assert!(xs.windows(2).all(|w| w[0] < w[1]), "{xs:?}");
    let want = crate::quality::preset("empty", crate::quality::DEFAULT_H).unwrap();
    let sent: Vec<&Command> = outs.iter().flat_map(|o| &o.commands).collect();
    assert!(matches!(sent.as_slice(), [Command::SetScene(s), Command::Dyn(DynCommand::SetWalls(Some(_)))] if *s == want), "{sent:?}");
    assert_eq!((st.preset.as_str(), st.grid_n), ("empty", 81), "the empty preset at the default quality");
    // the same path as any other preset
    let out = act(&mut st, &inp, Action::Preset("h2o".into()));
    assert!(matches!(out.commands.first(), Some(Command::SetScene(s)) if s.nuclei.len() == 3));
}

/// The chrome on an empty scene: the status says so (playing or paused), T reads "—", the
/// sampling button is disabled with "place atoms to sample", no atom card; every layout draws
/// and sends nothing.
#[test]
fn the_chrome_on_an_empty_scene_says_place_atoms() {
    let scene = Scene::preset("empty").unwrap();
    let cam = camera();
    let gpu = empty_snapshot(Backend::Gpu, None);
    let mut m = model();
    (m.fitted, m.trusted, m.samples, m.fit) = (false, false, 0, None);
    let est = empty_snapshot(Backend::Estimated, Some(m));
    for (snap, playing) in [(&gpu, false), (&gpu, true), (&est, true)] {
        let inp = PanelInput { playing, ..input(&scene, Some(snap), &cam) };
        assert_eq!(overlays::state(&inp), (false, EMPTY_STATUS.to_string()));
    }
    assert_eq!(temperature_text(&gpu.dynamics, 1, TempUnits::K), "—");
    let h = snapshot(&Scene::preset("h").unwrap(), Backend::Gpu, None);
    assert_eq!(temperature_text(&h.dynamics, 1, TempUnits::K), "0.0 K");

    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("empty", &scene);
    st.backend = Backend::Estimated;
    st.dock_tab = DockTab::Model;
    let inp = input(&scene, Some(&est), &cam);
    for _ in 0..3 {
        assert!(frame(&ctx, &mut st, &inp, 1400.0, vec![]).commands.is_empty());
    }
    // the sampling button is there but disabled: a click sends nothing
    let outs = click_widget(&ctx, &mut st, &inp, "sample");
    assert!(outs.iter().all(|o| o.commands.is_empty()), "an empty scene cannot be sampled");
    for snap in [None, Some(&gpu), Some(&est)] {
        for tab in DockTab::ALL {
            st.dock_tab = tab;
            for w in [1400.0, 1000.0] {
                let inp = PanelInput { playing: true, ..input(&scene, snap, &cam) };
                assert!(frame(&ctx, &mut st, &inp, w, vec![]).commands.is_empty(), "{tab:?} at {w}");
            }
        }
    }
}

// ---- quality presets (Scene ▸ Quality, `crate::quality`) ----

/// A quality change is a grid edit that keeps the box (12 Bohr: 81³ at Accurate, 41³ at Fast,
/// Custom 0.1234 → 0.123 at 99³) — even after a rounded-up Custom box, the next level is back on
/// the preset's 12 Bohr — and leaves σ alone (None follows h; an explicit σ stays); the level the
/// scene is already at sends nothing.
#[test]
fn quality_changes_keep_the_box_and_leave_sigma_alone() {
    let cam = camera();
    let mut scene = Scene::preset("h2").unwrap();
    let mut st = UiState::new("h2", &scene);
    let step = |scene: &mut Scene, st: &mut UiState, h: f64| -> Option<(usize, f64)> {
        let inp = input(scene, None, &cam);
        let out = act(st, &inp, Action::Quality(h));
        let c = out.commands.first().cloned();
        if let Some(c) = &c {
            c.apply(scene);
        }
        match c {
            Some(Command::SetGrid { n, h }) => Some((n, h)),
            None => None,
            Some(other) => panic!("{other:?}"),
        }
    };
    assert_eq!(step(&mut scene, &mut st, 0.15), Some((81, 0.15)));
    assert_eq!(step(&mut scene, &mut st, 0.15), None, "already Accurate");
    assert_eq!(step(&mut scene, &mut st, 0.30), Some((41, 0.30)));
    assert_eq!(step(&mut scene, &mut st, 0.1234), Some((99, 0.123)), "Custom snaps to 3 decimals");
    assert_eq!(st.quality_h, 0.123);
    assert_eq!(step(&mut scene, &mut st, 0.15), Some((81, 0.15)), "the box does not creep");
    assert_eq!(st.box_len, 12.0);
    assert_eq!(scene.sigma, None, "σ follows h");
    scene.sigma = Some(0.2);
    assert_eq!(step(&mut scene, &mut st, 0.10), Some((121, 0.10)));
    assert_eq!(scene.sigma, Some(0.2), "an explicit σ is not overwritten");
    scene.validate().unwrap();
}

/// Scene ▸ Quality, clicked: Accurate on a 49³ scene sends the grid edit to 81³ × 0.15, and the
/// Custom slider is there; every level is listed with its grid.
#[test]
fn clicking_a_quality_level_sends_the_grid_edit() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let mut snap = snapshot(&scene, Backend::Gpu, None);
    snap.dynamics.eval.scf_steps = 9;
    snap.dynamics.eval.solve_ms = 150;
    let inp = input(&scene, Some(&snap), &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    egui::Popup::open_id(&ctx, menus::popup_id(menus::MenuId::Scene));
    let outs = click_widget(&ctx, &mut st, &inp, "quality-accurate");
    egui::Popup::close_all(&ctx);
    let grids: Vec<_> = outs.iter().flat_map(|o| &o.commands).filter(|c| matches!(c, Command::SetGrid { n: 81, h } if *h == 0.15)).collect();
    assert_eq!(grids.len(), 1, "{:?}", outs.iter().flat_map(|o| &o.commands).collect::<Vec<_>>());
    assert_eq!(st.last_solve, Some((49, 0.15)), "the cost estimates scale the last solve");
    for p in ["quality-fast", "quality-balanced", "quality-precise", "quality-custom"] {
        probed_rect(p);
    }
}

// ---- a toolbar that holds still (user report 2026-10-14: the bar grew and shrank while playing) ----

/// The fs/s readouts: three significant figures with an SI suffix, never more than five
/// characters (so the monospace slot fits every value), rounding carried into the next unit.
#[test]
fn fs_per_s_shows_three_figures_in_five_characters_at_most() {
    use toolbar::fs_per_s;
    let r = [0.0, 0.25, 9.996, 15.6, 99.97, 126.0, 999.6, 1240.0, 79_365.08, 999_400.0, 999_999.0, 1.5e6].map(fs_per_s);
    assert_eq!(r, ["0.00", "0.25", "10.0", "15.6", "100", "126", "1.00k", "1.24k", "79.4k", "999k", "1.00M", "1.50M"]);
    let mut x = 1e-4;
    while x < 1e16 {
        assert!(fs_per_s(x).chars().count() <= 5, "{x}: {}", fs_per_s(x));
        x *= 1.013;
    }
    assert_eq!([f64::NAN, -1.0, f64::INFINITY].map(fs_per_s), ["—"; 3]);
}

/// Every toolbar group keeps its rect (to 0.5 pt) whatever the live values — fs/s 1 vs 999 999,
/// this tick's rate, fallbacks 0 vs 100 %, epoch 1 vs 9999, the solver line long (untrusted,
/// sampling) vs short (DFT) vs absent, the selection, the dt warning, the pull, the speed and
/// dt fields, the element — in the full layout and the compact one; and no group outgrows the
/// width it was given (`toolbar::Widths`), nor the window.
#[test]
fn the_toolbar_groups_hold_still_when_live_values_change() {
    use toolbar::GROUPS;
    let scene = Scene::preset("h2o").unwrap();
    let cam = camera();
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    // quiet: small numbers everywhere
    let mut quiet_model = model();
    quiet_model.epoch = 1;
    let mut quiet = snapshot(&scene, Backend::Estimated, Some(quiet_model));
    quiet.dynamics.wall_ratio = 1.0;
    quiet.dynamics.fallback_fraction_100 = 0.0;
    quiet.dynamics.source = sim::Source::Surrogate;
    quiet.dynamics.ms_per_step = 0.01;
    // loud: the widest numbers and the longest lines
    let mut loud_model = model();
    loud_model.epoch = 9999;
    loud_model.trusted = false;
    loud_model.untrusted_reason = Some("residual_dof_e 3.0 < 30 after the last refit of a model with a very long reason".into());
    let mut loud = snapshot(&scene, Backend::Estimated, Some(loud_model));
    loud.dynamics.wall_ratio = 79_365.08;
    loud.dynamics.fallback_fraction_100 = 1.0;
    loud.dynamics.source = sim::Source::Fallback;
    loud.dynamics.ms_per_step = 900.0;
    loud.dynamics.dt_warning = Some("dt 2 fs is above 1 fs: under-resolved".into());
    let mut sampling = snapshot(&scene, Backend::Estimated, Some(model()));
    sampling.dynamics.sampling = Some(sim::SamplingProgress {
        done: 7,
        total: 1273,
        stage: "perturbations of the whole scene",
        last_energy: -76.4,
        cancelled: false,
        skipped: 2,
        failed: 0,
        finished: false,
    });
    let dft = snapshot(&scene, Backend::Gpu, None);
    // (snapshot, measured fs/s, selection, speed, dt, element, pull, playing)
    let states: [(Option<&Snapshot>, Option<f64>, Option<usize>, u32, f64, u8, f64, bool); 5] = [
        (Some(&quiet), Some(1.0), None, 1, 0.01, 1, 0.01, false),
        (Some(&loud), Some(999_999.0), Some(2), sim::dynamics::MAX_SPEED, 2.0, 10, 1.0, true),
        (None, None, Some(0), 37, 0.25, 8, 0.0125, false),
        (Some(&sampling), Some(15.6), None, 1000, 0.5, 2, 0.1, true),
        (Some(&dft), Some(79_365.08), Some(1), 120, 1.234, 6, 0.0156, true),
    ];
    for w in [1600.0, 1400.0, 1240.0, 1000.0] {
        let mut first: Option<std::collections::BTreeMap<&'static str, (egui::Rect, f32)>> = None;
        for (k, &(snap, rate, selected, speed, dt, element, pull, playing)) in states.iter().enumerate() {
            let mut st = UiState::new("h2o", &scene);
            st.gpu_available = true;
            st.backend = snap.map_or(Backend::Estimated, |s| s.backend);
            st.mode = ToolMode::Select;
            (st.speed, st.dt_fs, st.element, st.drag_force) = (speed, dt, element, pull);
            let inp = PanelInput { selected, playing, measured_fs_per_s: rate, ..input(&scene, snap, &cam) };
            for _ in 0..2 {
                GROUPS.with(|g| g.borrow_mut().clear());
                frame(&ctx, &mut st, &inp, w, vec![]);
            }
            let groups = GROUPS.with(|g| g.borrow().clone());
            assert_eq!(groups.len(), 7, "{w}: {:?}", groups.keys());
            for (anchor, (rect, width)) in &groups {
                assert!(rect.width() <= width + 0.5, "{w} state {k}: {anchor} is {} wide, given {width}", rect.width());
                assert!(rect.right() <= w, "{w} state {k}: {anchor} ends at {} past the window", rect.right());
            }
            match &first {
                None => first = Some(groups),
                Some(f) => {
                    for (anchor, (rect, _)) in &groups {
                        let r0 = f[anchor].0;
                        let moved = (rect.min - r0.min).length().max((rect.max - r0.max).length());
                        assert!(moved <= 0.5, "{w}: {anchor} moved {moved:.2} pt in state {k}: {r0:?} -> {rect:?}");
                    }
                }
            }
        }
        let f = first.unwrap();
        let compact = f["tb-time"].0.height() < 40.0;
        assert_eq!(compact, w < 1300.0, "{w}: the full layout fits 1400 and 1600, not 1240");
        eprintln!("{w} (compact {compact}): {}", f.iter().map(|(a, (r, gw))| format!("{a} {:.0}..{:.0} ({gw:.0})", r.left(), r.right())).collect::<Vec<_>>().join(", "));
    }
}

// ---- M6 Task 10: bonds, molecules, badge, relax, isotopes, the periodic table ----

/// The M6 actions emit exactly the driver commands of the engine features: Relax / StopRelax
/// (Relax refused during a sampling pass, with the status line), SetIsotope for the selection;
/// the placement isotope and the view toggles send nothing.
#[test]
fn m6_actions_send_relax_isotope_and_toggle_the_view() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let mut st = UiState::new("h2", &scene);
    assert!(matches!(one_dyn(&act(&mut st, &inp, Action::Relax)), DynCommand::Relax));
    assert!(matches!(one_dyn(&act(&mut st, &inp, Action::StopRelax)), DynCommand::StopRelax));
    assert!(matches!(one_dyn(&act(&mut st, &inp, Action::SetIsotope { index: 1, a: 2 })), DynCommand::SetIsotope { index: 1, a: 2 }));
    // a pass running: Relax is refused here (the driver refuses it too)
    let mut pass = snapshot(&scene, Backend::Estimated, Some(model()));
    pass.dynamics.sampling = Some(sim::SamplingProgress { done: 1, total: 9, stage: "perturbations", last_energy: -1.1, cancelled: false, skipped: 0, failed: 0, finished: false });
    let busy = input(&scene, Some(&pass), &cam);
    assert!(act(&mut st, &busy, Action::Relax).commands.is_empty());
    assert_eq!(st.status, RELAX_REFUSED_SAMPLING);
    // the placement isotope: canonical (the most abundant one is None), unlisted ones ignored
    st.mode = ToolMode::Orbit;
    assert!(act(&mut st, &inp, Action::ElementIsotope(1, 2)).commands.is_empty());
    assert_eq!((st.element, st.isotope, st.mode), (1, Some(2), ToolMode::Place));
    act(&mut st, &inp, Action::ElementIsotope(8, 18));
    assert_eq!((st.element, st.isotope), (8, Some(18)));
    act(&mut st, &inp, Action::ElementIsotope(8, 16));
    assert_eq!((st.element, st.isotope), (8, None));
    act(&mut st, &inp, Action::ElementIsotope(1, 5));
    assert_eq!((st.element, st.isotope), (8, None), "an unlisted isotope changes nothing");
    act(&mut st, &inp, Action::ElementIsotope(1, 3));
    act(&mut st, &inp, Action::Element(2));
    assert_eq!((st.element, st.isotope), (2, None), "an element chip is the most abundant isotope");
    // view toggles: display only
    assert!(st.show_bonds && st.show_labels && st.show_badge, "on by default");
    assert!(act(&mut st, &inp, Action::ShowBonds(false)).commands.is_empty());
    assert!(act(&mut st, &inp, Action::ShowLabels(false)).commands.is_empty());
    assert!(act(&mut st, &inp, Action::ShowBadge(false)).commands.is_empty());
    assert!(!st.show_bonds && !st.show_labels && !st.show_badge);
}

/// Clicked as a user would: the toolbar's Relax (Stop while relaxing), Physics ▸ Relax, the
/// Selection tab's ²H, View ▸ bonds, and the periodic table's cells (O places, Ar is greyed and
/// does nothing) and its isotope row.
#[test]
fn clicking_the_m6_controls_sends_their_commands() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let snap = snapshot(&scene, Backend::Gpu, None);
    let inp = PanelInput { selected: Some(0), ..input(&scene, Some(&snap), &cam) };
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    let c = dyn_commands(&click_widget(&ctx, &mut st, &inp, "relax"));
    assert!(matches!(c.as_slice(), [DynCommand::Relax]), "{c:?}");
    let mut relaxing = snapshot(&scene, Backend::Gpu, None);
    relaxing.dynamics.relax = Some(relax_progress(None));
    let inp_r = PanelInput { selected: Some(0), ..input(&scene, Some(&relaxing), &cam) };
    let c = dyn_commands(&click_widget(&ctx, &mut st, &inp_r, "relax"));
    assert!(matches!(c.as_slice(), [DynCommand::StopRelax]), "{c:?}");
    let in_menu = |st: &mut UiState, inp: &PanelInput<'_>, m: menus::MenuId, name: &str| {
        egui::Popup::open_id(&ctx, menus::popup_id(m));
        let outs = click_widget(&ctx, st, inp, name);
        egui::Popup::close_all(&ctx);
        outs
    };
    let c = dyn_commands(&in_menu(&mut st, &inp, menus::MenuId::Physics, "m-relax"));
    assert!(matches!(c.as_slice(), [DynCommand::Relax]), "{c:?}");
    let c = dyn_commands(&in_menu(&mut st, &inp_r, menus::MenuId::Physics, "m-relax-stop"));
    assert!(matches!(c.as_slice(), [DynCommand::StopRelax]), "{c:?}");
    assert!(in_menu(&mut st, &inp, menus::MenuId::View, "show-bonds").iter().all(|o| o.commands.is_empty()));
    assert!(!st.show_bonds);
    st.dock_tab = DockTab::Selection;
    let c = dyn_commands(&click_widget(&ctx, &mut st, &inp, "selection-isotope-2"));
    assert!(matches!(c.as_slice(), [DynCommand::SetIsotope { index: 0, a: 2 }]), "{c:?}");
    // the periodic table
    let in_table = |st: &mut UiState, name: &str| {
        open_ptable(&ctx);
        let outs = click_widget(&ctx, st, &inp, name);
        egui::Popup::close_all(&ctx);
        outs
    };
    assert!(in_table(&mut st, "ptable-8").iter().all(|o| o.commands.is_empty()));
    assert_eq!((st.element, st.isotope, st.mode), (8, None, ToolMode::Place));
    in_table(&mut st, "ptable-18");
    assert_eq!(st.element, 8, "Ar is greyed: needs pseudopotentials");
    in_table(&mut st, "ptable-1");
    in_table(&mut st, "ptable-isotope-2");
    assert_eq!((st.element, st.isotope), (1, Some(2)));
}

fn relax_progress(converged: Option<bool>) -> sim::RelaxProgress {
    sim::RelaxProgress {
        steps: 42,
        max_force: 3.1e-3,
        max_force_augmented: 3.1e-3,
        energy: -1.1021,
        energy_start: -1.1,
        dt_fs: 0.6,
        alpha: 0.08,
        converged,
        verified_max_force: None,
        note: None,
    }
}

/// The relax readout: steps, max|F| against F_TOL, the energy drop, FIRE's Δt; the result line;
/// the teacher's verification under the estimated solver; the driver's note.
#[test]
fn the_relax_readout_shows_progress_the_drop_and_the_verification() {
    let r = relax_progress(None);
    assert_eq!(relax_lines(&r), vec!["relaxing: step 42 · max|F| 3.1e-3 Ha/Bohr (stop < 1e-3) · ΔE -2.10e-3 Ha · Δt 0.60 fs"]);
    let mut done = relax_progress(Some(true));
    done.max_force = 8.2e-4;
    done.max_force_augmented = 8.2e-4;
    done.verified_max_force = Some(6.1e-4);
    assert_eq!(
        relax_lines(&done),
        vec!["relaxed in 42 steps: max|F| 8.2e-4 Ha/Bohr (< 1e-3) · ΔE -2.10e-3 Ha", "verified by one teacher solve: DFT max|F| 6.1e-4 Ha/Bohr"]
    );
    let mut held = relax_progress(Some(false));
    held.max_force_augmented = 5e-4;
    held.note = Some("stopped after 42 steps".into());
    let l = relax_lines(&held);
    assert!(l[0].starts_with("relax ended after 42 steps: max|F| 3.1e-3 Ha/Bohr (with tools 5.0e-4)"), "{l:?}");
    assert_eq!(l[1], "stopped after 42 steps");
}

fn graph(n: usize, bonds: &[(usize, usize)], components: Vec<Vec<usize>>, t: f64) -> std::sync::Arc<sim::BondGraph> {
    std::sync::Arc::new(sim::BondGraph {
        bonds: bonds.iter().map(|&(i, j)| sim::Bond { i, j, rho_bcp: 0.18, enhancement: 1.3, s: 0.6, route: sim::Route::Deformation, multi_centre: false }).collect(),
        components,
        step: 4,
        sim_time: t,
        n_nuclei: n,
    })
}

/// Under the surrogate the graph is the last DFT step's: the faded bonds get spec §3.4's label
/// with that step's time; a fresh graph (or none) gets none.
#[test]
fn the_stale_bonds_are_labelled_under_the_surrogate() {
    let scene = Scene::preset("h2").unwrap();
    let mut s = snapshot(&scene, Backend::Estimated, Some(model()));
    assert_eq!(bonds_label(&s.dynamics), None, "no graph");
    s.dynamics.bonds = Some(graph(2, &[(0, 1)], vec![vec![0, 1]], 12.5 / units::au_time_to_fs(1.0)));
    assert_eq!(bonds_label(&s.dynamics), None, "a DFT step's own graph");
    s.dynamics.bonds_stale = true;
    s.dynamics.source = sim::Source::Surrogate;
    assert_eq!(bonds_label(&s.dynamics).as_deref(), Some("bonds from the last DFT step (t = 12.50 fs) — the surrogate has no electrons"));
}

/// Spec §4.4's tooltip lines: a bonded pair's numbers and route, a rejected one's reason.
#[test]
fn pair_lines_carry_the_numbers_behind_the_verdict() {
    let scene = Scene::preset("h2o").unwrap();
    let r = |i, j, verdict, route| sim::PairReport {
        i,
        j,
        route,
        s: Some(0.3),
        rho_min: Some(0.18),
        transverse_ratio: Some(1.01),
        rho_pro: Some(0.14),
        enhancement: Some(1.3),
        march_end: None,
        march_stop: None,
        verdict,
    };
    assert_eq!(
        pair_line(&scene, &r(0, 1, sim::Verdict::Bonded { multi_centre: false }, Some(sim::Route::Deformation))),
        "O–H (#0–#1): bonded — ρ_b 0.180 e/Bohr³, 1.30× the free atoms' density there, deformation route"
    );
    assert_eq!(
        pair_line(&scene, &r(1, 2, sim::Verdict::NeighbourRegion { atom: 0 }, Some(sim::Route::Deformation))),
        "H–H (#1–#2): not a bond — the feature between them lies in O #0's share of space: another bond seen from the side"
    );
    assert!(pair_line(&scene, &r(1, 2, sim::Verdict::NotAccumulated, Some(sim::Route::Saddle))).ends_with("(×1.30 < 1.11)"));
}

/// The Selection tab and the molecule list draw with a bond graph and labels (and with a graph
/// whose count is not the scene's: then nothing is listed), sending nothing.
#[test]
fn the_selection_tab_lists_bonds_and_molecules() {
    let scene = Scene::preset("h2o").unwrap();
    let cam = camera();
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2o", &scene);
    st.dock_tab = DockTab::Selection;
    let mut s = snapshot(&scene, Backend::Gpu, None);
    s.dynamics.bonds = Some(graph(3, &[(0, 1), (0, 2)], vec![vec![0, 1, 2]], 0.0));
    s.labels = std::sync::Arc::new(vec![crate::names::classify(&[8, 1, 1], &[(0, 1), (0, 2)], Some(0))]);
    assert_eq!(s.labels[0].label(), "H₂O — water");
    for (sel, n) in [(Some(0), 3), (None, 3), (Some(1), 4)] {
        if n == 4 {
            s.dynamics.bonds = Some(graph(4, &[(0, 1)], vec![vec![0, 1], vec![2], vec![3]], 0.0));
        }
        let inp = PanelInput { selected: sel, ..input(&scene, Some(&s), &cam) };
        for _ in 0..2 {
            let out = frame(&ctx, &mut st, &inp, 1400.0, vec![]);
            assert!(out.commands.is_empty());
        }
    }
}

/// View ▸ Bonds and molecules ▸ name style: display only, no command; the labels follow it.
#[test]
fn the_name_style_is_a_view_setting() {
    use crate::names::NameStyle;
    let scene = Scene::preset("h2o").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let mut st = UiState::new("h2o", &scene);
    assert_eq!(st.name_style, NameStyle::Common);
    assert!(act(&mut st, &inp, Action::NameStyle(NameStyle::Stoichiometric)).commands.is_empty());
    assert_eq!(st.name_style, NameStyle::Stoichiometric);
    let water = crate::names::classify(&[8, 1, 1], &[(0, 1), (0, 2)], Some(0));
    assert_eq!(water.label_in(st.name_style), "H₂O — dihydrogen monoxide");
    assert!(search::results("dihydrogen monoxide", 5).iter().any(|e| e.name == "Name style"));
}

/// The superscripts of a formula are drawn as small raised text (the default font lacks ⁺ ⁻): one
/// raised run per superscript group, signs mapped to + and −, subscripts kept.
#[test]
fn superscripts_are_drawn_raised() {
    let job = rich("H₃⁺ — ¹⁶O²⁻", 12.0, theme::INK);
    assert_eq!(job.text, "H₃+ — 16O2−");
    let raised: Vec<&str> = job.sections.iter().filter(|s| s.format.valign == egui::Align::TOP).map(|s| &job.text[s.byte_range.start.0..s.byte_range.end.0]).collect();
    assert_eq!(raised, vec!["+", "16", "2−"]);
    assert!(job.sections.iter().all(|s| s.format.valign == egui::Align::TOP || s.format.font_id.size == 12.0));
    assert_eq!(rich("water", 12.0, theme::INK).sections.len(), 1);
}

/// Review I2: a scene or quality change over the GPU memory budget is refused with a message
/// before anything is sent (wgpu would otherwise panic on the allocation): 8 Ne in a 16 Bohr box at
/// Custom 0.08 (201³, 80 electrons, ≈ 25.5 GB) on the GPU backend — and the refused value does not
/// stay on the Custom slider. H₂O at Custom 0.08 on the estimated solver (177³, two GPU solvers —
/// teacher and auditor — ≈ 9.9 GB) is under the 12 GB default and goes
/// through. (The action path reads `ATOMIC_SANDBOX_VRAM_GB`; unset here.)
#[test]
fn a_change_over_the_vram_budget_is_refused() {
    use crate::quality::{gpu_solvers, vram_check, VRAM_BUDGET_GB};
    let budget = (VRAM_BUDGET_GB * 1e9) as u64;
    let cam = camera();
    let mut big = Scene::preset("empty").unwrap();
    big.grid_n = 65; // 16 Bohr at 0.25
    for i in 0..8 {
        let a = i as f64 * std::f64::consts::FRAC_PI_4;
        big.nuclei.push(crate::scene::SceneNucleus { z: 10, pos: [3.0 * a.cos(), 3.0 * a.sin(), 0.0], vel: [0.0; 3], a: None });
    }
    let fine = crate::quality::regrid(&big, 16.0, 0.08);
    assert_eq!(fine.grid_n, 201);
    let e = vram_check(&fine, 1, budget).unwrap_err();
    assert!(e.contains("GPU memory") && e.contains("budget") && e.contains("201³"), "{e}");
    assert!(vram_check(&fine, 0, budget).is_ok(), "the CPU backend allocates no GPU memory");
    let mut st = UiState::new("empty", &big);
    (st.backend, st.gpu_available) = (Backend::Gpu, true);
    let before = st.quality_h;
    st.quality_slider = 0.08; // dragged there
    let out = act(&mut st, &input(&big, None, &cam), Action::Quality(0.08));
    assert!(out.commands.is_empty() && out.error.as_deref().is_some_and(|e| e.contains("GPU memory")), "{:?}", out.error);
    assert_eq!((st.grid_n, st.quality_h, st.quality_slider), (65, before, before), "nothing changed, the slider back too");

    let w = Scene::preset("h2o").unwrap();
    let w8 = crate::quality::regrid(&w, 14.0, 0.08);
    assert_eq!(w8.grid_n, 177);
    let need = 2 * providers::gpu_footprint_bytes(&w8.to_sim().unwrap());
    eprintln!("H₂O at 0.08 (177³), estimated mode: ≈ {:.2} GB", need as f64 / 1e9);
    assert!((9.5e9..10.3e9).contains(&(need as f64)), "{need}");
    assert!(vram_check(&w8, gpu_solvers(Backend::Estimated, true), budget).is_ok());
    let mut st = UiState::new("h2o", &w);
    (st.backend, st.gpu_available) = (Backend::Estimated, true);
    let out = act(&mut st, &input(&w, None, &cam), Action::Quality(0.08));
    assert!(out.error.is_none(), "{:?}", out.error);
    assert!(matches!(out.commands.as_slice(), [Command::SetGrid { n: 177, h }] if *h == 0.08), "{:?}", out.commands);
}

// ---- temperature units (user request 2026-10-14: K / °C / °F, display only) ----

/// The scale converts at the display edge only: readouts in the chosen unit, "—" with no nuclei,
/// typed values back to kelvin and clamped at absolute zero; the toggle is an action.
#[test]
fn temperatures_show_in_the_chosen_scale_and_stay_kelvin_inside() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let mut snap = snapshot(&scene, Backend::Gpu, None);
    snap.dynamics.temperature = 300.0;
    assert_eq!(temperature_text(&snap.dynamics, 1, TempUnits::K), "300.0 K");
    assert_eq!(temperature_text(&snap.dynamics, 1, TempUnits::C), "26.9 °C");
    assert_eq!(temperature_text(&snap.dynamics, 0, TempUnits::F), "80 °F");
    assert_eq!(temperature_text(&empty_snapshot(Backend::Gpu, None).dynamics, 1, TempUnits::F), "—");
    assert_eq!(TempUnits::C.to_k(-300.0), 0.0, "clamped at absolute zero");
    assert!((TempUnits::F.to_k(32.0) - 273.15).abs() < 1e-9);
    assert_eq!(*TempUnits::C.range(0.0, 5000.0).start(), -273.15);
    let mut st = UiState::new("h2", &scene);
    assert_eq!(st.temp_units, TempUnits::K, "kelvin by default");
    let out = act(&mut st, &input(&scene, Some(&snap), &cam), Action::TempUnits(TempUnits::F));
    assert!(out.commands.is_empty(), "display only: nothing reaches the driver");
    assert_eq!(st.temp_units, TempUnits::F);
    for q in ["celsius", "fahrenheit"] {
        assert!(search::results(q, 3).first().is_some_and(|e| e.anchor == "m-thermostat"), "{q}");
    }
}

/// The thermostat target typed in °F: dragged far below absolute zero it stops there, and the
/// panel keeps 0 K; every tab and menu draws in °C.
#[test]
fn the_thermostat_target_is_typed_in_the_chosen_scale() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let mut snap = snapshot(&scene, Backend::Estimated, Some(model()));
    snap.dynamics.thermostat = Some((300.0, 10.0));
    let inp = input(&scene, Some(&snap), &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    st.temp_units = TempUnits::F;
    egui::Popup::open_id(&ctx, menus::popup_id(menus::MenuId::Physics));
    press_widget(&ctx, &mut st, &inp, "thermostat-t", -4000.0);
    egui::Popup::close_all(&ctx);
    assert!(st.thermostat_t.abs() < 1e-9, "clamped at 0 K: {}", st.thermostat_t);
    st.temp_units = TempUnits::C;
    st.backend = Backend::Estimated;
    for tab in DockTab::ALL {
        st.dock_tab = tab;
        assert!(frame(&ctx, &mut st, &inp, 1400.0, vec![]).commands.is_empty());
    }
    for mid in menus::MenuId::ALL {
        egui::Popup::open_id(&ctx, menus::popup_id(mid));
        frame(&ctx, &mut st, &inp, 1400.0, vec![]);
    }
    egui::Popup::close_all(&ctx);
}

/// A typed 300 K thermostat target survives switching the display to °C, °F and back: the scales
/// only convert what is shown, so the kelvin value stays exactly 300.0.
#[test]
fn toggling_the_temperature_unit_keeps_a_typed_target_exact() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let snap = snapshot(&scene, Backend::Gpu, None);
    let inp = input(&scene, Some(&snap), &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    st.thermostat_t = 300.0;
    egui::Popup::open_id(&ctx, menus::popup_id(menus::MenuId::Physics));
    for u in [TempUnits::C, TempUnits::F, TempUnits::K, TempUnits::F, TempUnits::C, TempUnits::K] {
        act(&mut st, &inp, Action::TempUnits(u));
        for _ in 0..3 {
            frame(&ctx, &mut st, &inp, 1400.0, vec![]);
        }
        assert_eq!(st.thermostat_t, 300.0, "after showing it in {u:?}");
    }
    egui::Popup::close_all(&ctx);
}

/// The sampling-plan presets (user request 2026-10-15): every one is a valid plan, Standard is
/// `sim`'s default, and they are distinct (so matching is unambiguous).
#[test]
fn sampling_presets_are_valid_and_standard_is_the_default() {
    use super::sampling_presets::{matching, PRESETS};
    for p in &PRESETS {
        assert_eq!(p.plan(0).validate(), Ok(()), "{}", p.name);
        assert!(!p.description.is_empty(), "{}", p.name);
    }
    let standard = PRESETS.iter().find(|p| p.name == "Standard").unwrap();
    assert_eq!(standard.plan(0), SamplingPlan::default());
    assert_eq!(PRESETS.iter().filter(|p| p.plan(0) == SamplingPlan::default()).count(), 1);
    // the panel starts on Standard
    let st = UiState::new("h2", &Scene::preset("h2").unwrap());
    assert_eq!(matching(&st.plan).map(|i| PRESETS[i].name), Some("Standard"));
}

/// Each preset is recognised whatever the seed; any single-field edit makes the plan "Custom".
#[test]
fn sampling_preset_matching_is_derived_from_the_fields() {
    use super::sampling_presets::{matching, PRESETS};
    for (i, p) in PRESETS.iter().enumerate() {
        for seed in [0, 7, u64::MAX] {
            assert_eq!(matching(&p.plan(seed)), Some(i), "{} at seed {seed}", p.name);
        }
        let edits: [(&str, fn(&mut SamplingPlan)); 7] = [
            ("perturbations", |q| q.perturbations += 1),
            ("σ_r", |q| q.perturbation_sigma += 0.001),
            ("burst steps", |q| q.hot_burst_steps -= 1),
            ("burst T", |q| q.hot_burst_t += 1.0),
            ("scan points", |q| q.scan_points_per_pair += 1),
            ("scan range lo", |q| q.scan_range.0 += 0.05),
            ("scan range hi", |q| q.scan_range.1 -= 0.05),
        ];
        for (what, edit) in edits {
            let mut q = p.plan(3);
            edit(&mut q);
            assert_eq!(matching(&q), None, "{} with {what} edited", p.name);
        }
    }
}

/// Picking a preset sets its fields and keeps the plan's seed.
#[test]
fn selecting_a_sampling_preset_keeps_the_seed() {
    use super::sampling_presets::{matching, PRESETS};
    let mut plan = SamplingPlan { perturbations: 5, perturbation_sigma: 0.42, hot_burst_t: 77.0, seed: 1234, ..SamplingPlan::default() };
    assert_eq!(matching(&plan), None);
    for (i, p) in PRESETS.iter().enumerate() {
        p.apply(&mut plan);
        assert_eq!(plan.seed, 1234, "{}", p.name);
        assert_eq!(plan, p.plan(1234));
        assert_eq!(matching(&plan), Some(i));
    }
}
